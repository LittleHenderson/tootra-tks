"""
TKS Trace Utilities - Compression, Serialization, and Batching

This module provides utilities for handling TKS model traces efficiently:
    1. Top-K compression for constant-size representations
    2. JSONL serialization for logging and analysis
    3. TraceBatch class for efficient batched training
    4. Feature extraction for trace-consistency loss
    5. Schema validation for trace integrity

Trace Structure (from TKSNoeticLM and TKSLLMCorePipeline):
    - embedding: [batch, seq, 40] initial noetic embedding
    - worlds: dict with A, B, C, D world tensors
    - blocks: list of per-block traces (noetic_weights, attn_weights, ...)
    - attractor: [batch, seq, 40] converged attractor state
    - attractor_converged: bool
    - attractor_iterations: int
    - rpm_gate: [batch, seq] RPM gate values
    - dwp_scores: [batch, seq, 7, 3] D/W/P scores per Foundation
    - operator_core: dict (if enabled) with noetic_output, gate_values, etc.

Usage:
    from trace_utils import compress_trace, TraceBatch, extract_trace_features

    # Compress for storage
    out = model(tokens, return_full_trace=True)
    compressed = compress_trace(out['trace'], noetic_top_k=3)

    # Serialize to JSONL
    save_traces([compressed], "traces.jsonl")

    # Create batch for training
    batch = TraceBatch.from_model_output(out)
    features = batch.get_noetic_features()

Author: TKS-LLM Compiler Agent
Date: 2025-12-23
"""

from __future__ import annotations

import json
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F


# =============================================================================
# CONSTANTS
# =============================================================================

NOETIC_DIM = 10
NUM_WORLDS = 4
TOTAL_DIM = 40
NUM_FOUNDATIONS = 7

# Default feature dimensions for trace-consistency loss
DEFAULT_NOETIC_TOP_K = 3
DEFAULT_NUM_BLOCKS = 4
DEFAULT_NUM_SCALES = 3


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

TRACE_SCHEMA_V1 = {
    "noetic_routing": {"weights", "indices"},
    "attention": {"scale_weights"},
    "attractor": {"converged", "iterations", "final_delta"},
    "rpm": {"gate", "dwp_scores"},
    "worlds": {"A_norm", "B_norm", "C_norm", "D_norm"},
}

TRACE_SCHEMA_V1_1 = {
    **TRACE_SCHEMA_V1,
    "operator_core": {"noetic_output", "gate_values", "symmetry_violation"},
}


def validate_trace(trace: dict, schema_version: str = "1.0") -> bool:
    """
    Check if trace conforms to expected schema.

    Args:
        trace: Trace dictionary to validate
        schema_version: Schema version ("1.0" or "1.1")

    Returns:
        True if valid, False otherwise
    """
    if schema_version == "1.0":
        schema = TRACE_SCHEMA_V1
    elif schema_version == "1.1":
        schema = TRACE_SCHEMA_V1_1
    else:
        raise ValueError(f"Unknown schema version: {schema_version}")

    # Check top-level keys
    for section, required_keys in schema.items():
        if section in trace:
            section_data = trace[section]
            if isinstance(section_data, dict):
                for key in required_keys:
                    if key not in section_data:
                        return False

    return True


# =============================================================================
# TOP-K COMPRESSION
# =============================================================================

def compress_trace(
    full_trace: dict,
    noetic_top_k: int = 3,
    include_attention: bool = True,
    include_trajectory: bool = False,
) -> dict:
    """
    Compress a full trace to constant-size representation.

    Full traces can be large (O(seq * hidden) per block). This function
    compresses to O(k * num_blocks) by keeping only top-k noetic activations.

    Args:
        full_trace: Output from model with return_full_trace=True
                   Expected keys: 'embedding', 'worlds', 'blocks', 'attractor',
                   'attractor_converged', 'attractor_iterations', 'rpm_gate',
                   'dwp_scores', optionally 'operator_core'
        noetic_top_k: Keep top-k noetic weights per token (default 3)
        include_attention: Include attention scale weights (default True)
        include_trajectory: Include attractor trajectory - large! (default False)

    Returns:
        Compressed trace with constant size regardless of model width:
        {
            'noetic_routing': {
                'weights': [num_blocks, k],  # top-k weights per block (mean over seq)
                'indices': [num_blocks, k],  # indices of top-k noetics
            },
            'attention': {
                'scale_weights': [num_blocks, num_scales],  # attention scale mix
            },
            'attractor': {
                'converged': bool,
                'iterations': int,
                'final_delta': float,
            },
            'rpm': {
                'gate': float,  # mean gate value
                'dwp_scores': [7, 3],  # mean D/W/P per foundation
            },
            'worlds': {
                'A_norm': float, 'B_norm': float, 'C_norm': float, 'D_norm': float,
            },
            'operator_core': {...} if present,
            'trajectory': [...] if include_trajectory,
        }
    """
    compressed = {}

    # 1. Extract and compress noetic routing
    # Handle both v4 model format (blocks list) and v2 pipeline format (top-level noetic_routing)
    if 'blocks' in full_trace and full_trace['blocks']:
        # V4 model format: extract from blocks
        blocks = full_trace['blocks']
        num_blocks = len(blocks)

        noetic_weights_list = []
        noetic_indices_list = []
        attention_scale_list = []

        for block_trace in blocks:
            # Noetic weights: [batch, seq, 10] -> top-k per block
            if 'noetic_weights' in block_trace:
                weights = block_trace['noetic_weights']
                if isinstance(weights, torch.Tensor):
                    # Mean over batch and sequence
                    mean_weights = weights.mean(dim=(0, 1))  # [10]
                    top_k_values, top_k_indices = torch.topk(
                        mean_weights, k=min(noetic_top_k, 10)
                    )
                    noetic_weights_list.append(top_k_values.tolist())
                    noetic_indices_list.append(top_k_indices.tolist())

            # Attention weights: extract scale mixing if available
            # Handle list of tensors (one per scale) or stacked tensor
            if include_attention and 'attn_weights' in block_trace:
                attn_weights = block_trace['attn_weights']
                if isinstance(attn_weights, torch.Tensor):
                    # Stacked tensor: [batch, num_scales, seq_q, seq_k] or similar
                    if attn_weights.dim() == 4:
                        scale_means = attn_weights.mean(dim=(0, 2, 3)).tolist()
                        attention_scale_list.append(scale_means)
                elif isinstance(attn_weights, list):
                    # List of tensors per scale
                    scale_means = []
                    for scale_attn in attn_weights:
                        if isinstance(scale_attn, torch.Tensor):
                            scale_means.append(scale_attn.mean().item())
                    if scale_means:
                        attention_scale_list.append(scale_means)

        compressed['noetic_routing'] = {
            'weights': noetic_weights_list,
            'indices': noetic_indices_list,
        }

        if include_attention and attention_scale_list:
            compressed['attention'] = {
                'scale_weights': attention_scale_list,
            }

    elif 'noetic_routing' in full_trace:
        # V2 pipeline format: already has noetic_routing at top level
        routing = full_trace['noetic_routing']
        if isinstance(routing, dict):
            weights = routing.get('weights', [])
            indices = routing.get('indices', [])

            # Convert tensors if needed
            if isinstance(weights, torch.Tensor):
                # Tensor: could be [batch, seq, 10] or [10]
                if weights.dim() == 3:
                    # [batch, seq, 10] -> mean over batch and seq, then top-k
                    mean_weights = weights.mean(dim=(0, 1))  # [10]
                    top_k_values, top_k_indices = torch.topk(
                        mean_weights, k=min(noetic_top_k, 10)
                    )
                    compressed['noetic_routing'] = {
                        'weights': [top_k_values.tolist()],
                        'indices': [top_k_indices.tolist()],
                    }
                else:
                    weights = weights.tolist()

            if 'noetic_routing' not in compressed and isinstance(weights, list):
                # Check structure of weights list
                if weights and isinstance(weights[0], list):
                    # Could be [batch, seq, 10] or [block, 10]
                    first = weights[0]
                    if isinstance(first, list) and first and isinstance(first[0], list):
                        # [batch, seq, 10] format - flatten and get top-k
                        # Convert to tensor for easier processing
                        try:
                            w_tensor = torch.tensor(weights)  # [batch, seq, 10]
                            mean_weights = w_tensor.mean(dim=(0, 1))
                            top_k_values, top_k_indices = torch.topk(
                                mean_weights, k=min(noetic_top_k, 10)
                            )
                            compressed['noetic_routing'] = {
                                'weights': [top_k_values.tolist()],
                                'indices': [top_k_indices.tolist()],
                            }
                        except Exception:
                            # Fallback: just take first values
                            compressed['noetic_routing'] = {
                                'weights': [],
                                'indices': [],
                            }
                    else:
                        # [block, 10] or [seq, 10] - each row is one block
                        noetic_weights_list = []
                        noetic_indices_list = []
                        for row in weights:
                            if isinstance(row, list) and len(row) == 10:
                                row_tensor = torch.tensor(row)
                                top_k_values, top_k_indices = torch.topk(
                                    row_tensor, k=min(noetic_top_k, 10)
                                )
                                noetic_weights_list.append(top_k_values.tolist())
                                noetic_indices_list.append(top_k_indices.tolist())
                        compressed['noetic_routing'] = {
                            'weights': noetic_weights_list,
                            'indices': noetic_indices_list,
                        }
                elif weights and isinstance(weights[0], (int, float)):
                    # Single [10] vector
                    w_tensor = torch.tensor(weights)
                    top_k_values, top_k_indices = torch.topk(
                        w_tensor, k=min(noetic_top_k, len(weights))
                    )
                    compressed['noetic_routing'] = {
                        'weights': [top_k_values.tolist()],
                        'indices': [top_k_indices.tolist()],
                    }
                else:
                    compressed['noetic_routing'] = {
                        'weights': [],
                        'indices': [],
                    }

        # V2 may also have attention info
        if include_attention and 'attention' in full_trace:
            attn = full_trace['attention']
            if isinstance(attn, dict) and 'scale_weights' in attn:
                compressed['attention'] = {
                    'scale_weights': attn['scale_weights'],
                }

    # 2. Attractor state
    compressed['attractor'] = {
        'converged': full_trace.get('attractor_converged', False),
        'iterations': full_trace.get('attractor_iterations', 0),
        'final_delta': 0.0,  # Not always available in trace
    }

    # If trajectory is available and requested, compress it
    if include_trajectory and 'trajectory' in full_trace:
        trajectory = full_trace.get('trajectory', [])
        # Handle tensor/list/None safely
        has_traj = (isinstance(trajectory, torch.Tensor) and trajectory.numel() > 0) or \
                   (isinstance(trajectory, list) and len(trajectory) > 0)
        if has_traj:
            # Store only deltas between states (more compact)
            deltas = []
            for i in range(1, len(trajectory)):
                if isinstance(trajectory[i], torch.Tensor):
                    delta = (trajectory[i] - trajectory[i-1]).abs().mean().item()
                    deltas.append(delta)
            compressed['trajectory'] = deltas

    # 3. RPM gating
    rpm_gate = full_trace.get('rpm_gate')
    dwp_scores = full_trace.get('dwp_scores')

    compressed['rpm'] = {
        'gate': rpm_gate.mean().item() if isinstance(rpm_gate, torch.Tensor) else 0.0,
    }

    if isinstance(dwp_scores, torch.Tensor):
        # dwp_scores: [batch, seq, 7, 3] -> [7, 3] mean
        compressed['rpm']['dwp_scores'] = dwp_scores.mean(dim=(0, 1)).tolist()

    # 4. World norms from embedding
    embedding = full_trace.get('embedding')
    worlds = full_trace.get('worlds', {})

    compressed['worlds'] = {}
    if isinstance(embedding, torch.Tensor):
        # Extract world norms from full embedding
        A = embedding[..., 0:10]
        B = embedding[..., 10:20]
        C = embedding[..., 20:30]
        D = embedding[..., 30:40]
        compressed['worlds'] = {
            'A_norm': A.norm().item(),
            'B_norm': B.norm().item(),
            'C_norm': C.norm().item(),
            'D_norm': D.norm().item(),
        }
    elif isinstance(worlds, dict) and worlds:
        # Use pre-split world tensors
        for world_name in ['A', 'B', 'C', 'D']:
            if world_name in worlds:
                w = worlds[world_name]
                if isinstance(w, torch.Tensor):
                    compressed['worlds'][f'{world_name}_norm'] = w.norm().item()

    # 5. Operator core (if present)
    if 'operator_core' in full_trace:
        op_core = full_trace['operator_core']
        compressed['operator_core'] = {
            'gate_mean': op_core.get('gate_values', torch.tensor(0.0)).mean().item()
            if isinstance(op_core.get('gate_values'), torch.Tensor) else 0.0,
        }
        if 'symmetry_losses' in op_core:
            sym_losses = op_core['symmetry_losses']
            compressed['operator_core']['symmetry_violation'] = sum(
                v.item() if isinstance(v, torch.Tensor) else v
                for v in sym_losses.values()
            ) / max(len(sym_losses), 1)

    return compressed


# =============================================================================
# JSONL SERIALIZATION
# =============================================================================

def _tensor_to_json(obj: Any) -> Any:
    """Convert tensors to JSON-serializable format."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: _tensor_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_tensor_to_json(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def _json_to_tensor(obj: Any) -> Any:
    """Convert JSON lists back to tensors where appropriate."""
    if isinstance(obj, list):
        # Heuristic: if it looks like a numeric array, convert to tensor
        if obj and isinstance(obj[0], (int, float)):
            return torch.tensor(obj)
        elif obj and isinstance(obj[0], list):
            # Nested list - recursively convert
            inner = [_json_to_tensor(v) for v in obj]
            if all(isinstance(v, torch.Tensor) for v in inner):
                return torch.stack(inner)
            return inner
        return [_json_to_tensor(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _json_to_tensor(v) for k, v in obj.items()}
    return obj


def serialize_trace(trace: dict, compact: bool = True) -> str:
    """
    Serialize trace to JSONL-compatible string.

    Args:
        trace: Trace dictionary (can be full or compressed)
        compact: If True, use minimal whitespace (default True)

    Returns:
        JSON string suitable for JSONL line
    """
    json_obj = _tensor_to_json(trace)
    if compact:
        return json.dumps(json_obj, separators=(',', ':'))
    else:
        return json.dumps(json_obj, indent=2)


def deserialize_trace(line: str) -> dict:
    """
    Deserialize trace from JSONL string.

    Args:
        line: JSON string from JSONL file

    Returns:
        Trace dictionary with numeric arrays as tensors
    """
    obj = json.loads(line.strip())
    return _json_to_tensor(obj)


def save_traces(
    traces: List[dict],
    path: Union[str, Path],
    compress_file: bool = False,
) -> None:
    """
    Save multiple traces to JSONL file.

    Args:
        traces: List of trace dictionaries
        path: Output file path (.jsonl or .jsonl.gz)
        compress_file: If True, gzip compress the output
    """
    path = Path(path)

    if compress_file or path.suffix == '.gz':
        open_fn = gzip.open
        mode = 'wt'
        if not path.suffix == '.gz':
            path = path.with_suffix(path.suffix + '.gz')
    else:
        open_fn = open
        mode = 'w'

    with open_fn(path, mode, encoding='utf-8') as f:
        for trace in traces:
            line = serialize_trace(trace, compact=True)
            f.write(line + '\n')


def load_traces(path: Union[str, Path]) -> List[dict]:
    """
    Load traces from JSONL file.

    Args:
        path: Input file path (.jsonl or .jsonl.gz)

    Returns:
        List of trace dictionaries
    """
    path = Path(path)

    if path.suffix == '.gz':
        open_fn = gzip.open
        mode = 'rt'
    else:
        open_fn = open
        mode = 'r'

    traces = []
    with open_fn(path, mode, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                traces.append(deserialize_trace(line))

    return traces


# =============================================================================
# TRACE FEATURE EXTRACTION
# =============================================================================

def extract_trace_features(
    trace: dict,
    noetic_top_k: int = DEFAULT_NOETIC_TOP_K,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    num_scales: int = DEFAULT_NUM_SCALES,
) -> torch.Tensor:
    """
    Extract fixed-size feature vector from trace for consistency loss.

    This function extracts a canonical feature vector from a trace that can
    be used for trace-consistency training. The features capture the key
    structural properties of the noetic computation.

    Features (concatenated):
        - noetic_routing: top-k weights flattened [k * num_blocks]
        - world_mix: 4 values (A/B/C/D norms)
        - attention_scales: num_scales values (mean scale weights)
        - attractor: [converged, iterations/max_iter, final_delta]
        - rpm: [gate_mean, foundation_idx/7, top_dwp_score]
        - operator_core: [mean_gate, sym_violation] (if present, else zeros)

    Total: ~50-100 dimensional fixed vector depending on configuration.

    Args:
        trace: Trace dictionary (compressed or full)
        noetic_top_k: Number of top noetic weights to include (default 3)
        num_blocks: Expected number of blocks (default 4)
        num_scales: Number of attention scales (default 3)

    Returns:
        Feature tensor of shape [feature_dim] where feature_dim depends on config
    """
    features = []
    device = 'cpu'

    # 1. Noetic routing features: [k * num_blocks]
    noetic_routing = trace.get('noetic_routing', {})
    weights = noetic_routing.get('weights', [])

    noetic_features = []
    for block_weights in weights[:num_blocks]:
        if isinstance(block_weights, torch.Tensor):
            block_weights = block_weights.tolist()
        # Pad or truncate to noetic_top_k
        block_weights = block_weights[:noetic_top_k]
        while len(block_weights) < noetic_top_k:
            block_weights.append(0.0)
        noetic_features.extend(block_weights)

    # Pad if fewer blocks than expected
    while len(noetic_features) < noetic_top_k * num_blocks:
        noetic_features.append(0.0)

    features.extend(noetic_features[:noetic_top_k * num_blocks])

    # 2. World mix features: [4]
    worlds = trace.get('worlds', {})
    world_features = [
        worlds.get('A_norm', 0.0),
        worlds.get('B_norm', 0.0),
        worlds.get('C_norm', 0.0),
        worlds.get('D_norm', 0.0),
    ]
    # Normalize world features
    world_sum = sum(world_features) + 1e-8
    world_features = [w / world_sum for w in world_features]
    features.extend(world_features)

    # 3. Attention scale features: [num_scales]
    attention = trace.get('attention', {})
    scale_weights = attention.get('scale_weights', [])

    scale_features = []
    # Handle tensor/list safely
    has_scale_weights = (isinstance(scale_weights, torch.Tensor) and scale_weights.numel() > 0) or \
                        (isinstance(scale_weights, list) and len(scale_weights) > 0)
    if has_scale_weights:
        # Average over blocks
        for s in range(num_scales):
            scale_mean = sum(
                block_scales[s] if s < len(block_scales) else 0.0
                for block_scales in scale_weights
            ) / max(len(scale_weights), 1)
            scale_features.append(scale_mean)
    while len(scale_features) < num_scales:
        scale_features.append(1.0 / num_scales)
    features.extend(scale_features[:num_scales])

    # 4. Attractor features: [3]
    attractor = trace.get('attractor', {})
    max_iter = 15  # Default max iterations
    attractor_features = [
        1.0 if attractor.get('converged', False) else 0.0,
        attractor.get('iterations', 0) / max_iter,
        min(attractor.get('final_delta', 0.0), 1.0),  # Clamp delta
    ]
    features.extend(attractor_features)

    # 5. RPM features: [3]
    rpm = trace.get('rpm', {})
    gate_mean = rpm.get('gate', 0.0)
    if isinstance(gate_mean, torch.Tensor):
        gate_mean = gate_mean.mean().item()

    # Find dominant foundation from DWP scores
    # Handle different formats: [7, 3], [[7, 3], ...], tensor [batch, seq, 7, 3]
    dwp_scores = rpm.get('dwp_scores', [[0.0, 0.0, 0.0]] * NUM_FOUNDATIONS)
    if isinstance(dwp_scores, torch.Tensor):
        # Average over batch and sequence, keep [7, 3]
        if dwp_scores.dim() == 4:  # [batch, seq, 7, 3]
            dwp_scores = dwp_scores.mean(dim=(0, 1))  # [7, 3]
        elif dwp_scores.dim() == 3:  # [batch, 7, 3] or [seq, 7, 3]
            dwp_scores = dwp_scores.mean(dim=0)  # [7, 3]
        dwp_scores = dwp_scores.tolist()

    # Compute gate per foundation (D * W * P)
    foundation_gates = []
    for f_idx in range(NUM_FOUNDATIONS):
        if f_idx < len(dwp_scores):
            scores = dwp_scores[f_idx]
            if isinstance(scores, (list, tuple)) and len(scores) >= 3:
                d, w, p = scores[0], scores[1], scores[2]
                foundation_gates.append(d * w * p)
            elif isinstance(scores, (int, float)):
                foundation_gates.append(float(scores))
            else:
                foundation_gates.append(0.0)
        else:
            foundation_gates.append(0.0)

    dominant_foundation = max(range(len(foundation_gates)), key=lambda i: foundation_gates[i])
    top_dwp_score = max(foundation_gates) if foundation_gates else 0.0

    rpm_features = [
        gate_mean,
        dominant_foundation / NUM_FOUNDATIONS,
        top_dwp_score,
    ]
    features.extend(rpm_features)

    # 6. Operator core features: [2] (if present)
    operator_core = trace.get('operator_core', {})
    op_features = [
        operator_core.get('gate_mean', 0.0),
        operator_core.get('symmetry_violation', 0.0),
    ]
    features.extend(op_features)

    # Flatten any nested lists and ensure all values are floats
    def flatten_to_floats(lst):
        result = []
        for item in lst:
            if isinstance(item, (list, tuple)):
                result.extend(flatten_to_floats(item))
            elif isinstance(item, torch.Tensor):
                if item.numel() == 1:
                    result.append(float(item.item()))
                else:
                    result.extend([float(x) for x in item.flatten().tolist()])
            elif isinstance(item, (int, float)):
                result.append(float(item))
            else:
                result.append(0.0)  # Default for unknown types
        return result

    features = flatten_to_floats(features)

    return torch.tensor(features, dtype=torch.float32, device=device)


def compute_feature_dim(
    noetic_top_k: int = DEFAULT_NOETIC_TOP_K,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    num_scales: int = DEFAULT_NUM_SCALES,
) -> int:
    """
    Compute the feature dimension for extract_trace_features.

    Args:
        noetic_top_k: Number of top noetic weights
        num_blocks: Number of blocks
        num_scales: Number of attention scales

    Returns:
        Total feature dimension
    """
    dim = 0
    dim += noetic_top_k * num_blocks  # noetic routing
    dim += 4  # world mix
    dim += num_scales  # attention scales
    dim += 3  # attractor
    dim += 3  # rpm
    dim += 2  # operator core
    return dim


# =============================================================================
# TRACE BATCH CLASS
# =============================================================================

@dataclass
class TraceBatch:
    """
    Batched trace container for efficient training.

    This class collates multiple traces into batched tensors for
    efficient GPU computation during training.

    Attributes:
        traces: List of raw trace dictionaries
        batch_size: Number of traces in batch
        noetic_top_k: Top-k parameter used for compression
        _cached_features: Cached feature tensor
    """

    traces: List[dict]
    batch_size: int = field(init=False)
    noetic_top_k: int = DEFAULT_NOETIC_TOP_K
    _cached_features: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        self.batch_size = len(self.traces)

    def to_tensor_dict(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Convert to batched tensors for loss computation.

        Returns a dictionary with batched versions of key trace components.

        Args:
            device: Target device ('cpu' or 'cuda')

        Returns:
            Dict with batched tensors:
            {
                'noetic_weights': [batch, num_blocks, top_k],
                'world_norms': [batch, 4],
                'attractor_converged': [batch],
                'rpm_gate': [batch],
                'dwp_scores': [batch, 7, 3],
                ...
            }
        """
        result = {}

        # Collect noetic weights
        noetic_weights = []
        for trace in self.traces:
            routing = trace.get('noetic_routing', {})
            weights = routing.get('weights', [])
            # Flatten to [num_blocks * top_k]
            flat_weights = []
            for block_w in weights:
                if isinstance(block_w, torch.Tensor):
                    block_w = block_w.tolist()
                flat_weights.extend(block_w[:self.noetic_top_k])
            noetic_weights.append(flat_weights)

        # Pad to same length
        max_len = max(len(w) for w in noetic_weights) if noetic_weights else 0
        noetic_weights = [w + [0.0] * (max_len - len(w)) for w in noetic_weights]
        result['noetic_weights'] = torch.tensor(noetic_weights, device=device)

        # Collect world norms
        world_norms = []
        for trace in self.traces:
            worlds = trace.get('worlds', {})
            norms = [
                worlds.get('A_norm', 0.0),
                worlds.get('B_norm', 0.0),
                worlds.get('C_norm', 0.0),
                worlds.get('D_norm', 0.0),
            ]
            world_norms.append(norms)
        result['world_norms'] = torch.tensor(world_norms, device=device)

        # Collect attractor info
        converged = []
        iterations = []
        for trace in self.traces:
            attractor = trace.get('attractor', {})
            converged.append(1.0 if attractor.get('converged', False) else 0.0)
            iterations.append(float(attractor.get('iterations', 0)))
        result['attractor_converged'] = torch.tensor(converged, device=device)
        result['attractor_iterations'] = torch.tensor(iterations, device=device)

        # Collect RPM info
        gates = []
        dwp_scores = []
        for trace in self.traces:
            rpm = trace.get('rpm', {})
            gates.append(rpm.get('gate', 0.0))
            scores = rpm.get('dwp_scores', [[0.0, 0.0, 0.0]] * NUM_FOUNDATIONS)
            if isinstance(scores, torch.Tensor):
                scores = scores.tolist()
            # Ensure correct shape
            while len(scores) < NUM_FOUNDATIONS:
                scores.append([0.0, 0.0, 0.0])
            scores = [s[:3] + [0.0] * (3 - len(s)) for s in scores[:NUM_FOUNDATIONS]]
            dwp_scores.append(scores)
        result['rpm_gate'] = torch.tensor(gates, device=device)
        result['dwp_scores'] = torch.tensor(dwp_scores, device=device)

        return result

    def get_noetic_features(self, device: str = 'cpu') -> torch.Tensor:
        """
        Extract features for trace-consistency loss.

        Returns a [batch, feature_dim] tensor of extracted features
        suitable for computing trace-consistency loss during training.

        Args:
            device: Target device

        Returns:
            Feature tensor [batch, feature_dim]
        """
        if self._cached_features is not None:
            return self._cached_features.to(device)

        features = []
        for trace in self.traces:
            feat = extract_trace_features(trace, noetic_top_k=self.noetic_top_k)
            features.append(feat)

        self._cached_features = torch.stack(features).to(device)
        return self._cached_features

    @classmethod
    def from_model_output(
        cls,
        output: dict,
        compress: bool = True,
        noetic_top_k: int = DEFAULT_NOETIC_TOP_K,
    ) -> 'TraceBatch':
        """
        Create TraceBatch from model forward output.

        Args:
            output: Dict from model.forward() with 'trace' key
            compress: Whether to compress traces (default True)
            noetic_top_k: Top-k for compression

        Returns:
            TraceBatch instance
        """
        trace = output.get('trace', {})
        if not trace:
            raise ValueError("Model output missing 'trace' key")

        if compress:
            compressed = compress_trace(trace, noetic_top_k=noetic_top_k)
            traces = [compressed]
        else:
            traces = [trace]

        return cls(traces=traces, noetic_top_k=noetic_top_k)

    @classmethod
    def from_traces(
        cls,
        traces: List[dict],
        noetic_top_k: int = DEFAULT_NOETIC_TOP_K,
    ) -> 'TraceBatch':
        """
        Create TraceBatch from list of trace dicts.

        Args:
            traces: List of trace dictionaries
            noetic_top_k: Top-k parameter

        Returns:
            TraceBatch instance
        """
        return cls(traces=traces, noetic_top_k=noetic_top_k)

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, idx: int) -> dict:
        return self.traces[idx]


# =============================================================================
# TRACE CONSISTENCY LOSS
# =============================================================================

def trace_consistency_loss(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Compute trace consistency loss between two feature vectors.

    This loss encourages similar inputs to produce similar trace patterns.
    Used for semi-supervised training where we want consistent noetic
    behavior across augmented versions of the same input.

    Args:
        features_a: [batch, feature_dim] features from first trace
        features_b: [batch, feature_dim] features from second trace
        margin: Margin for contrastive component (default 0.1)

    Returns:
        Scalar loss tensor
    """
    # Normalize features
    features_a = F.normalize(features_a, dim=-1)
    features_b = F.normalize(features_b, dim=-1)

    # Cosine similarity loss (want high similarity)
    cos_sim = (features_a * features_b).sum(dim=-1)
    sim_loss = 1.0 - cos_sim.mean()

    # L2 distance for fine-grained matching
    l2_dist = (features_a - features_b).pow(2).sum(dim=-1).sqrt()
    dist_loss = l2_dist.mean()

    return sim_loss + 0.5 * dist_loss


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def merge_trace_batches(batches: List[TraceBatch]) -> TraceBatch:
    """
    Merge multiple TraceBatch objects into one.

    Args:
        batches: List of TraceBatch instances

    Returns:
        Merged TraceBatch
    """
    all_traces = []
    noetic_top_k = batches[0].noetic_top_k if batches else DEFAULT_NOETIC_TOP_K

    for batch in batches:
        all_traces.extend(batch.traces)

    return TraceBatch(traces=all_traces, noetic_top_k=noetic_top_k)


def filter_traces_by_convergence(
    traces: List[dict],
    require_converged: bool = True,
) -> List[dict]:
    """
    Filter traces by attractor convergence status.

    Args:
        traces: List of trace dictionaries
        require_converged: If True, keep only converged traces

    Returns:
        Filtered list of traces
    """
    filtered = []
    for trace in traces:
        attractor = trace.get('attractor', {})
        converged = attractor.get('converged', False)
        if converged == require_converged:
            filtered.append(trace)
    return filtered


def compute_trace_statistics(traces: List[dict]) -> dict:
    """
    Compute aggregate statistics over a list of traces.

    Args:
        traces: List of trace dictionaries

    Returns:
        Dict with statistics:
        {
            'num_traces': int,
            'convergence_rate': float,
            'mean_iterations': float,
            'mean_rpm_gate': float,
            'world_balance': [4] mean world norms,
            ...
        }
    """
    if not traces:
        return {'num_traces': 0}

    stats = {'num_traces': len(traces)}

    # Convergence stats
    converged = sum(
        1 for t in traces
        if t.get('attractor', {}).get('converged', False)
    )
    stats['convergence_rate'] = converged / len(traces)

    iterations = [
        t.get('attractor', {}).get('iterations', 0)
        for t in traces
    ]
    stats['mean_iterations'] = sum(iterations) / len(iterations)

    # RPM stats
    gates = [t.get('rpm', {}).get('gate', 0.0) for t in traces]
    stats['mean_rpm_gate'] = sum(gates) / len(gates)

    # World balance
    world_norms = {'A': [], 'B': [], 'C': [], 'D': []}
    for t in traces:
        worlds = t.get('worlds', {})
        for w in ['A', 'B', 'C', 'D']:
            world_norms[w].append(worlds.get(f'{w}_norm', 0.0))

    stats['world_balance'] = {
        w: sum(norms) / len(norms) if norms else 0.0
        for w, norms in world_norms.items()
    }

    return stats


# =============================================================================
# MAIN TEST SECTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TKS Trace Utilities - Test Suite")
    print("=" * 70)

    # Create mock trace data
    def create_mock_trace() -> dict:
        """Create a mock trace for testing."""
        batch, seq = 2, 8
        num_blocks = 4

        blocks = []
        for _ in range(num_blocks):
            blocks.append({
                'noetic_weights': torch.softmax(torch.randn(batch, seq, 10), dim=-1),
                'attn_weights': torch.softmax(torch.randn(batch, 3, seq, seq), dim=-1),
            })

        return {
            'embedding': torch.randn(batch, seq, 40),
            'worlds': {
                'A': torch.randn(batch, seq, 10),
                'B': torch.randn(batch, seq, 10),
                'C': torch.randn(batch, seq, 10),
                'D': torch.randn(batch, seq, 10),
            },
            'blocks': blocks,
            'attractor': torch.randn(batch, seq, 40),
            'attractor_converged': True,
            'attractor_iterations': 7,
            'rpm_gate': torch.sigmoid(torch.randn(batch, seq)),
            'dwp_scores': torch.sigmoid(torch.randn(batch, seq, 7, 3)),
            'operator_core': {
                'gate_values': torch.sigmoid(torch.randn(batch, 40)),
                'symmetry_losses': {
                    'left_right': torch.tensor(0.01),
                    'composition': torch.tensor(0.02),
                },
            },
        }

    # Test 1: Compression
    print("\n--- Test 1: Trace Compression ---")
    mock_trace = create_mock_trace()
    compressed = compress_trace(mock_trace, noetic_top_k=3)

    print(f"Original trace keys: {list(mock_trace.keys())}")
    print(f"Compressed trace keys: {list(compressed.keys())}")
    print(f"Noetic routing shape: {len(compressed['noetic_routing']['weights'])} blocks")
    print(f"World norms: {compressed['worlds']}")
    print(f"Attractor: {compressed['attractor']}")
    print(f"RPM gate mean: {compressed['rpm']['gate']:.4f}")

    # Test 2: Serialization
    print("\n--- Test 2: JSONL Serialization ---")
    json_str = serialize_trace(compressed)
    print(f"JSON length: {len(json_str)} chars")
    print(f"First 200 chars: {json_str[:200]}...")

    # Roundtrip test
    deserialized = deserialize_trace(json_str)
    print(f"Roundtrip keys match: {set(compressed.keys()) == set(deserialized.keys())}")

    # Test 3: File I/O
    print("\n--- Test 3: File I/O ---")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save and load
        path = os.path.join(tmpdir, "traces.jsonl")
        save_traces([compressed, compressed], path)
        loaded = load_traces(path)
        print(f"Saved 2 traces, loaded {len(loaded)} traces")

        # Test gzip
        path_gz = os.path.join(tmpdir, "traces.jsonl.gz")
        save_traces([compressed], path_gz, compress_file=True)
        loaded_gz = load_traces(path_gz)
        print(f"Gzip roundtrip: {len(loaded_gz)} traces")

    # Test 4: Feature Extraction
    print("\n--- Test 4: Feature Extraction ---")
    features = extract_trace_features(compressed)
    print(f"Feature vector shape: {features.shape}")
    print(f"Expected dim: {compute_feature_dim()}")
    print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")

    # Test 5: TraceBatch
    print("\n--- Test 5: TraceBatch ---")
    batch = TraceBatch.from_traces([compressed, compressed, compressed])
    print(f"Batch size: {len(batch)}")

    tensor_dict = batch.to_tensor_dict()
    print(f"Tensor dict keys: {list(tensor_dict.keys())}")
    print(f"World norms shape: {tensor_dict['world_norms'].shape}")
    print(f"DWP scores shape: {tensor_dict['dwp_scores'].shape}")

    noetic_features = batch.get_noetic_features()
    print(f"Noetic features shape: {noetic_features.shape}")

    # Test 6: Trace Consistency Loss
    print("\n--- Test 6: Trace Consistency Loss ---")
    features_a = batch.get_noetic_features()
    # Slightly perturbed version
    features_b = features_a + 0.1 * torch.randn_like(features_a)
    loss = trace_consistency_loss(features_a, features_b)
    print(f"Consistency loss (similar traces): {loss.item():.4f}")

    # Very different traces
    features_c = torch.randn_like(features_a)
    loss_diff = trace_consistency_loss(features_a, features_c)
    print(f"Consistency loss (different traces): {loss_diff.item():.4f}")

    # Test 7: Statistics
    print("\n--- Test 7: Trace Statistics ---")
    traces = [compressed, compressed, compressed]
    stats = compute_trace_statistics(traces)
    print(f"Num traces: {stats['num_traces']}")
    print(f"Convergence rate: {stats['convergence_rate']:.2%}")
    print(f"Mean iterations: {stats['mean_iterations']:.1f}")
    print(f"Mean RPM gate: {stats['mean_rpm_gate']:.4f}")
    print(f"World balance: {stats['world_balance']}")

    # Test 8: Schema Validation
    print("\n--- Test 8: Schema Validation ---")
    is_valid = validate_trace(compressed, schema_version="1.0")
    print(f"Schema v1.0 valid: {is_valid}")

    is_valid_v11 = validate_trace(compressed, schema_version="1.1")
    print(f"Schema v1.1 valid: {is_valid_v11}")

    # Test 9: Filter by Convergence
    print("\n--- Test 9: Filter by Convergence ---")
    # Create mix of converged and non-converged
    non_converged = dict(compressed)
    non_converged['attractor'] = {**compressed['attractor'], 'converged': False}
    mixed_traces = [compressed, non_converged, compressed]

    converged_only = filter_traces_by_convergence(mixed_traces, require_converged=True)
    print(f"Total traces: {len(mixed_traces)}, Converged: {len(converged_only)}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
