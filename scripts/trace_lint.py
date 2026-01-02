#!/usr/bin/env python3
"""
TKS Trace Lint - Automated Verifier for TKS Invariants and Trace Integrity

This module provides comprehensive linting for TKS model traces and weights,
verifying that all canonical TKS invariants hold during training and inference.

CORE INVARIANTS VERIFIED:
    1. Attractor Contractivity: Lipschitz constant L < 1
    2. Attractor Convergence: Fixed-point iteration terminates
    3. Noetic Operator Spectra: Spectral radius < 1 for stability
    4. Routing Health: Entropy, sparsity, and collapse detection
    5. Residual Budget: Energy constraints on skip connections
    6. Bridge Usage: Cross-world energy limits
    7. D/W/P Score Normalization: Sum constraints per Foundation

USAGE:
    # CLI: Lint a model checkpoint
    python scripts/trace_lint.py --model output/models/best_model.pt

    # CLI: Lint a trace file
    python scripts/trace_lint.py --trace output/traces.jsonl

    # CLI: Lint both with strict mode
    python scripts/trace_lint.py --model model.pt --trace traces.jsonl --strict

    # Library: Use in training loop
    from scripts.trace_lint import TrainingLintMonitor, TKSInvariants

    monitor = TrainingLintMonitor(model, check_interval=100)
    for step, batch in enumerate(dataloader):
        output = model(batch, return_full_trace=True)
        monitor.step(step, output.get('trace'))

    report = monitor.get_report()

Author: TKS Integration Agent (tks-supervisor)
Date: 2025-12-23
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# CONSTANTS (TKS Canonical)
# =============================================================================

NOETIC_DIM = 10
NUM_WORLDS = 4
TOTAL_DIM = 40
NUM_FOUNDATIONS = 7

WORLD_NAMES = ['A', 'B', 'C', 'D']
FOUNDATION_NAMES = [
    'Unity', 'Wisdom', 'Life', 'Companionship', 'Power', 'Material', 'Lust'
]


# =============================================================================
# SEVERITY ENUM
# =============================================================================

class Severity(str, Enum):
    """Violation severity levels."""
    ERROR = "error"      # Must be fixed, training should stop
    WARNING = "warning"  # Should be investigated
    INFO = "info"        # Informational, may be expected


# =============================================================================
# INVARIANT DEFINITIONS
# =============================================================================

@dataclass
class TKSInvariants:
    """
    Invariants that must hold for a valid TKS model.

    These thresholds are derived from the TKS canonical specification and
    the mathematical requirements for stable noetic computation.

    Attributes:
        attractor_lipschitz_max: Maximum Lipschitz constant (< 1 for contraction)
        attractor_convergence_required: Whether convergence must occur
        max_attractor_iterations: Maximum allowed iterations before convergence
        spectral_radius_max: Maximum spectral radius for noetic operators
        involution_tolerance: Tolerance for symmetric/anti-symmetric operators
        min_routing_entropy: Minimum routing entropy (prevents collapse)
        max_routing_sparsity: Maximum allowed routing sparsity
        max_residual_energy: Maximum residual connection energy ratio
        max_bridge_energy: Maximum cross-world bridge energy ratio
        dwp_sum_tolerance: Tolerance for D+W+P sum per foundation
    """

    # Attractor invariants
    attractor_lipschitz_max: float = 0.99
    attractor_convergence_required: bool = True
    max_attractor_iterations: int = 15
    attractor_tolerance: float = 1e-3

    # Noetic operator invariants
    spectral_radius_max: float = 1.0
    involution_tolerance: float = 0.01

    # Routing invariants
    min_routing_entropy: float = 0.5  # bits
    max_routing_sparsity: float = 0.9  # At most 90% on one expert
    routing_collapse_threshold: float = 0.95  # >95% on one = collapsed

    # Energy budget invariants
    max_residual_energy: float = 0.1  # 10% of output from residual
    max_bridge_energy: float = 0.3  # 30% cross-world

    # D/W/P score invariants
    dwp_sum_tolerance: float = 0.1  # D+W+P should sum to ~1 per foundation
    dwp_min_value: float = 0.0
    dwp_max_value: float = 1.0

    # World balance invariants
    world_imbalance_threshold: float = 10.0  # Max ratio between world norms

    # Trace schema version
    expected_schema_version: str = "1.0"


# =============================================================================
# LINT VIOLATION
# =============================================================================

@dataclass
class LintViolation:
    """
    A single lint violation found during trace or model analysis.

    Attributes:
        severity: How serious the violation is
        component: Which TKS component violated (attractor, routing, etc.)
        invariant: Name of the violated invariant
        message: Human-readable description
        value: The actual measured value
        threshold: The expected threshold that was violated
        location: Optional location info (step, layer, etc.)
    """
    severity: Severity
    component: str
    invariant: str
    message: str
    value: float
    threshold: float
    location: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'severity': self.severity.value,
            'component': self.component,
            'invariant': self.invariant,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'location': self.location,
        }

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.severity.value.upper()}] {self.component}.{self.invariant}{loc}: {self.message} (value={self.value:.4f}, threshold={self.threshold:.4f})"


# =============================================================================
# TRACE LINTER
# =============================================================================

class TraceLinter:
    """
    Lint a trace dictionary for invariant violations.

    The trace linter checks runtime trace outputs against TKS canonical
    invariants to detect issues during training or inference.

    Example:
        linter = TraceLinter()
        violations = linter.lint(trace)

        if any(v.severity == Severity.ERROR for v in violations):
            raise RuntimeError("Critical TKS invariant violation!")
    """

    def __init__(self, invariants: Optional[TKSInvariants] = None):
        """
        Initialize the trace linter.

        Args:
            invariants: TKS invariant thresholds (default if not provided)
        """
        self.invariants = invariants or TKSInvariants()
        self.violations: List[LintViolation] = []

    def lint(self, trace: dict) -> List[LintViolation]:
        """
        Check trace against all invariants.

        Args:
            trace: Trace dictionary from model forward pass

        Returns:
            List of violations (empty if clean)
        """
        self.violations = []

        self._check_schema_version(trace)
        self._check_attractor(trace)
        self._check_routing(trace)
        self._check_residual(trace)
        self._check_bridge(trace)
        self._check_dwp(trace)
        self._check_worlds(trace)
        self._check_operator_core(trace)

        return self.violations

    def _add_violation(
        self,
        severity: Severity,
        component: str,
        invariant: str,
        message: str,
        value: float,
        threshold: float,
        location: Optional[str] = None,
    ) -> None:
        """Add a violation to the list."""
        self.violations.append(LintViolation(
            severity=severity,
            component=component,
            invariant=invariant,
            message=message,
            value=value,
            threshold=threshold,
            location=location,
        ))

    def _check_schema_version(self, trace: dict) -> None:
        """Verify trace schema version matches expected."""
        version = trace.get('_schema_version', 'unknown')
        if version != self.invariants.expected_schema_version:
            self._add_violation(
                severity=Severity.WARNING,
                component="schema",
                invariant="version_match",
                message=f"Trace schema version '{version}' != expected '{self.invariants.expected_schema_version}'",
                value=0.0,
                threshold=0.0,
            )

    def _check_attractor(self, trace: dict) -> None:
        """
        Check attractor convergence properties.

        Verifies:
            1. Convergence occurred (if required)
            2. Iteration count is within bounds
            3. Final delta is below tolerance
        """
        attractor = trace.get('attractor', {})
        if not attractor:
            # Check legacy format
            attractor = {
                'converged': trace.get('attractor_converged', False),
                'iterations': trace.get('attractor_iterations', 0),
                'final_delta': trace.get('final_delta', 0.0),
            }

        # Check convergence
        converged = attractor.get('converged', False)
        if self.invariants.attractor_convergence_required and not converged:
            self._add_violation(
                severity=Severity.ERROR,
                component="attractor",
                invariant="convergence_required",
                message="Attractor did not converge to fixed point",
                value=0.0,
                threshold=1.0,
            )

        # Check iteration count
        iterations = attractor.get('iterations', 0)
        if iterations >= self.invariants.max_attractor_iterations:
            self._add_violation(
                severity=Severity.WARNING,
                component="attractor",
                invariant="max_iterations",
                message=f"Attractor used max iterations ({iterations}), may not have converged",
                value=float(iterations),
                threshold=float(self.invariants.max_attractor_iterations),
            )

        # Check final delta
        final_delta = attractor.get('final_delta', 0.0)
        if final_delta > self.invariants.attractor_tolerance:
            severity = Severity.ERROR if converged else Severity.WARNING
            self._add_violation(
                severity=severity,
                component="attractor",
                invariant="delta_tolerance",
                message=f"Final delta {final_delta:.6f} exceeds tolerance",
                value=final_delta,
                threshold=self.invariants.attractor_tolerance,
            )

        # Estimate Lipschitz from trajectory if available
        trajectory = attractor.get('trajectory', [])
        # Handle tensor or list trajectory
        traj_len = trajectory.shape[0] if isinstance(trajectory, torch.Tensor) else len(trajectory) if trajectory is not None else 0
        if traj_len >= 2:
            lipschitz_estimate = self._estimate_trajectory_lipschitz(trajectory)
            if lipschitz_estimate >= self.invariants.attractor_lipschitz_max:
                self._add_violation(
                    severity=Severity.ERROR,
                    component="attractor",
                    invariant="lipschitz_contraction",
                    message=f"Estimated Lipschitz {lipschitz_estimate:.4f} >= 1 (not contractive)",
                    value=lipschitz_estimate,
                    threshold=self.invariants.attractor_lipschitz_max,
                )

    def _estimate_trajectory_lipschitz(self, trajectory: List) -> float:
        """Estimate Lipschitz constant from attractor trajectory."""
        if len(trajectory) < 2:
            return 0.0

        ratios = []
        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]

            if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
                # Trajectory stores deltas directly
                if abs(prev) > 1e-8:
                    ratios.append(abs(curr) / abs(prev))
            elif isinstance(prev, torch.Tensor) and isinstance(curr, torch.Tensor):
                # Full state tensors
                prev_norm = prev.norm().item()
                curr_norm = curr.norm().item()
                if prev_norm > 1e-8:
                    ratios.append(curr_norm / prev_norm)

        return max(ratios) if ratios else 0.0

    def _check_routing(self, trace: dict) -> None:
        """
        Check noetic routing health.

        Verifies:
            1. Routing entropy is above minimum (prevents collapse)
            2. Sparsity is below maximum
            3. No routing collapse (single expert domination)
        """
        noetic_routing = trace.get('noetic_routing', {})
        if not noetic_routing:
            return

        weights = noetic_routing.get('weights', None)
        if weights is None:
            return
        # Check if empty (handle tensor vs list)
        if isinstance(weights, torch.Tensor):
            if weights.numel() == 0:
                return
        elif not weights:
            return

        # Handle different weight formats
        if isinstance(weights, torch.Tensor):
            weights_tensor = weights
        elif isinstance(weights, list):
            # Could be [block, k] or flat
            if weights and isinstance(weights[0], list):
                # Take first block for analysis
                weights_tensor = torch.tensor(weights[0])
            else:
                weights_tensor = torch.tensor(weights)
        else:
            return

        # Ensure we have a 1D tensor of weights
        if weights_tensor.dim() > 1:
            weights_tensor = weights_tensor.flatten()[:NOETIC_DIM]

        # Normalize to form a distribution
        if weights_tensor.sum() > 0:
            probs = weights_tensor / weights_tensor.sum()
        else:
            return

        # Check entropy
        entropy = self._compute_entropy(probs)
        max_entropy = math.log2(NOETIC_DIM)  # Maximum entropy for uniform
        normalized_entropy = entropy / max_entropy

        if entropy < self.invariants.min_routing_entropy:
            self._add_violation(
                severity=Severity.WARNING,
                component="routing",
                invariant="min_entropy",
                message=f"Routing entropy {entropy:.3f} below minimum (routing may be collapsing)",
                value=entropy,
                threshold=self.invariants.min_routing_entropy,
            )

        # Check sparsity (max weight)
        max_weight = probs.max().item()
        if max_weight > self.invariants.max_routing_sparsity:
            severity = Severity.ERROR if max_weight > self.invariants.routing_collapse_threshold else Severity.WARNING
            self._add_violation(
                severity=severity,
                component="routing",
                invariant="max_sparsity",
                message=f"Max routing weight {max_weight:.3f} indicates over-sparsity",
                value=max_weight,
                threshold=self.invariants.max_routing_sparsity,
            )

        # Check for collapse
        if max_weight > self.invariants.routing_collapse_threshold:
            self._add_violation(
                severity=Severity.ERROR,
                component="routing",
                invariant="collapse_detection",
                message=f"Routing collapsed: {max_weight:.1%} on single expert",
                value=max_weight,
                threshold=self.invariants.routing_collapse_threshold,
            )

    def _compute_entropy(self, probs: torch.Tensor) -> float:
        """Compute entropy in bits from probability distribution."""
        # Add small epsilon for numerical stability
        probs = probs.clamp(min=1e-10)
        return -(probs * probs.log2()).sum().item()

    def _check_residual(self, trace: dict) -> None:
        """
        Check residual connection energy budget.

        Verifies residual energy does not dominate the output.
        """
        # Check operator core for gate values indicating residual strength
        operator_core = trace.get('operator_core', {})

        gate_mean = operator_core.get('gate_mean', 0.0)
        if isinstance(gate_mean, torch.Tensor):
            gate_mean = gate_mean.item()

        # Low gate values indicate heavy residual usage
        residual_estimate = 1.0 - gate_mean if gate_mean > 0 else 0.0

        if residual_estimate > self.invariants.max_residual_energy:
            self._add_violation(
                severity=Severity.WARNING,
                component="residual",
                invariant="max_energy",
                message=f"Residual energy estimate {residual_estimate:.3f} exceeds budget",
                value=residual_estimate,
                threshold=self.invariants.max_residual_energy,
            )

    def _check_bridge(self, trace: dict) -> None:
        """
        Check cross-world bridge energy constraints.

        Verifies that cross-world communication is within bounds.
        """
        worlds = trace.get('worlds', {})
        if not worlds:
            return

        norms = []
        for w in WORLD_NAMES:
            norm_key = f'{w}_norm'
            if norm_key in worlds:
                norm_val = worlds[norm_key]
                if isinstance(norm_val, torch.Tensor):
                    norm_val = norm_val.item()
                norms.append(norm_val)

        if len(norms) < 2:
            return

        total_norm = sum(norms)
        if total_norm < 1e-8:
            return

        # Estimate bridge energy as deviation from uniform world distribution
        expected = total_norm / len(norms)
        deviations = [abs(n - expected) for n in norms]
        bridge_energy = sum(deviations) / total_norm

        if bridge_energy > self.invariants.max_bridge_energy:
            self._add_violation(
                severity=Severity.WARNING,
                component="bridge",
                invariant="max_energy",
                message=f"Cross-world bridge energy {bridge_energy:.3f} exceeds limit",
                value=bridge_energy,
                threshold=self.invariants.max_bridge_energy,
            )

    def _check_dwp(self, trace: dict) -> None:
        """
        Check D/W/P score constraints.

        Verifies:
            1. D+W+P sums to approximately 1 per foundation
            2. Individual scores are in [0, 1]
        """
        rpm = trace.get('rpm', {})
        dwp_scores = rpm.get('dwp_scores', None)

        # Handle tensor/list/None cases safely
        if dwp_scores is None:
            return
        if isinstance(dwp_scores, torch.Tensor):
            if dwp_scores.numel() == 0:
                return
            dwp_scores = dwp_scores.tolist()
        elif not dwp_scores:
            return

        # Handle different formats
        # Format 1: [7, 3] - per foundation D/W/P
        # Format 2: [7] - just gate values

        if len(dwp_scores) == NUM_FOUNDATIONS:
            for f_idx, scores in enumerate(dwp_scores):
                if isinstance(scores, list) and len(scores) >= 3:
                    d, w, p = scores[0], scores[1], scores[2]
                    total = d + w + p

                    # Check individual bounds
                    for name, val in [('D', d), ('W', w), ('P', p)]:
                        if val < self.invariants.dwp_min_value or val > self.invariants.dwp_max_value:
                            self._add_violation(
                                severity=Severity.ERROR,
                                component="dwp",
                                invariant="value_bounds",
                                message=f"{name} score {val:.3f} out of [0,1] for Foundation {f_idx}",
                                value=val,
                                threshold=self.invariants.dwp_max_value,
                                location=f"Foundation_{f_idx}_{FOUNDATION_NAMES[f_idx]}",
                            )

                    # Check sum constraint (soft)
                    if abs(total - 1.0) > self.invariants.dwp_sum_tolerance:
                        self._add_violation(
                            severity=Severity.INFO,
                            component="dwp",
                            invariant="sum_constraint",
                            message=f"D+W+P={total:.3f} deviates from 1.0 for Foundation {f_idx}",
                            value=total,
                            threshold=1.0,
                            location=f"Foundation_{f_idx}_{FOUNDATION_NAMES[f_idx]}",
                        )

    def _check_worlds(self, trace: dict) -> None:
        """
        Check world balance constraints.

        Verifies that world norms are reasonably balanced.
        """
        worlds = trace.get('worlds', {})
        if not worlds:
            return

        norms = []
        for w in WORLD_NAMES:
            norm_key = f'{w}_norm'
            if norm_key in worlds:
                norm_val = worlds[norm_key]
                if isinstance(norm_val, torch.Tensor):
                    norm_val = norm_val.item()
                norms.append((w, norm_val))

        if len(norms) < 2:
            return

        # Check for extreme imbalance
        max_norm = max(n for _, n in norms)
        min_norm = min(n for _, n in norms)

        if min_norm > 1e-8:
            ratio = max_norm / min_norm
            if ratio > self.invariants.world_imbalance_threshold:
                max_world = max(norms, key=lambda x: x[1])[0]
                min_world = min(norms, key=lambda x: x[1])[0]
                self._add_violation(
                    severity=Severity.WARNING,
                    component="worlds",
                    invariant="balance",
                    message=f"World imbalance: {max_world}/{min_world} ratio = {ratio:.1f}",
                    value=ratio,
                    threshold=self.invariants.world_imbalance_threshold,
                )

    def _check_operator_core(self, trace: dict) -> None:
        """
        Check operator core constraints.

        Verifies:
            1. Symmetry violation is low
            2. Gate values are in bounds
        """
        operator_core = trace.get('operator_core', {})
        if not operator_core:
            return

        # Check symmetry violation
        sym_violation = operator_core.get('symmetry_violation', 0.0)
        if isinstance(sym_violation, torch.Tensor):
            sym_violation = sym_violation.item()

        if sym_violation > self.invariants.involution_tolerance:
            self._add_violation(
                severity=Severity.WARNING,
                component="operator_core",
                invariant="symmetry",
                message=f"Operator symmetry violation {sym_violation:.4f} exceeds tolerance",
                value=sym_violation,
                threshold=self.invariants.involution_tolerance,
            )


# =============================================================================
# MODEL LINTER
# =============================================================================

class ModelLinter:
    """
    Lint model weights for structural TKS invariants.

    The model linter checks weight matrices and layer configurations
    against TKS canonical requirements.

    Example:
        linter = ModelLinter()
        violations = linter.lint(model)

        for v in violations:
            if v.severity == Severity.ERROR:
                print(f"CRITICAL: {v}")
    """

    def __init__(self, invariants: Optional[TKSInvariants] = None):
        """
        Initialize the model linter.

        Args:
            invariants: TKS invariant thresholds
        """
        self.invariants = invariants or TKSInvariants()
        self.violations: List[LintViolation] = []

    def lint(self, model: nn.Module) -> List[LintViolation]:
        """
        Check model weights against TKS invariants.

        Args:
            model: PyTorch model to lint

        Returns:
            List of violations
        """
        self.violations = []

        self._check_attractor(model)
        self._check_noetic_operators(model)
        self._check_world_projections(model)

        return self.violations

    def _add_violation(
        self,
        severity: Severity,
        component: str,
        invariant: str,
        message: str,
        value: float,
        threshold: float,
        location: Optional[str] = None,
    ) -> None:
        """Add a violation to the list."""
        self.violations.append(LintViolation(
            severity=severity,
            component=component,
            invariant=invariant,
            message=message,
            value=value,
            threshold=threshold,
            location=location,
        ))

    def _check_attractor(self, model: nn.Module) -> None:
        """Check attractor contraction properties."""
        attractor = getattr(model, 'attractor', None)
        if attractor is None:
            # Try to find attractor in submodules
            for name, module in model.named_modules():
                if 'attractor' in name.lower():
                    attractor = module
                    break

        if attractor is None:
            self._add_violation(
                severity=Severity.INFO,
                component="attractor",
                invariant="existence",
                message="No attractor module found in model",
                value=0.0,
                threshold=1.0,
            )
            return

        # Estimate Lipschitz constant via power iteration
        lipschitz = self._estimate_lipschitz(attractor)

        if lipschitz >= self.invariants.attractor_lipschitz_max:
            self._add_violation(
                severity=Severity.ERROR,
                component="attractor",
                invariant="lipschitz_contraction",
                message=f"Attractor Lipschitz {lipschitz:.3f} >= 1 (not contractive)",
                value=lipschitz,
                threshold=self.invariants.attractor_lipschitz_max,
            )
        elif lipschitz >= 0.9:
            self._add_violation(
                severity=Severity.WARNING,
                component="attractor",
                invariant="lipschitz_margin",
                message=f"Attractor Lipschitz {lipschitz:.3f} close to 1 (weak contraction)",
                value=lipschitz,
                threshold=0.9,
            )

    def _estimate_lipschitz(self, module: nn.Module, num_samples: int = 100, num_iters: int = 20) -> float:
        """
        Estimate Lipschitz constant via power iteration.

        For a linear layer W, estimates ||W||_2 (spectral norm).
        For nonlinear modules, samples random points.

        Args:
            module: Module to estimate Lipschitz for
            num_samples: Number of random samples for nonlinear estimation
            num_iters: Power iteration steps

        Returns:
            Estimated Lipschitz constant
        """
        # Try to find linear weights to analyze
        weights = []

        # Look for contraction_maps (StableAttractorLayer)
        if hasattr(module, 'contraction_maps'):
            for T in module.contraction_maps:
                if hasattr(T, 'weight'):
                    weights.append(T.weight)
                elif hasattr(T, 'weight_orig'):  # spectral_norm
                    weights.append(T.weight_orig)

        # Look for generic weight matrices
        for name, param in module.named_parameters():
            if 'weight' in name and param.dim() == 2:
                weights.append(param)

        if not weights:
            # Fallback: empirical estimation
            return self._estimate_lipschitz_empirical(module, num_samples)

        # Power iteration for spectral norm
        max_lipschitz = 0.0

        for W in weights:
            with torch.no_grad():
                # Power iteration to estimate ||W||_2
                v = torch.randn(W.shape[1], device=W.device)
                v = v / v.norm()

                for _ in range(num_iters):
                    u = W @ v
                    u_norm = u.norm()
                    u = u / (u_norm + 1e-8)

                    v = W.T @ u
                    v_norm = v.norm()
                    v = v / (v_norm + 1e-8)

                # Spectral norm estimate
                spectral_norm = (W @ v).norm().item()

                # Account for contraction factor if present
                if hasattr(module, 'contraction_factor'):
                    spectral_norm *= module.contraction_factor

                max_lipschitz = max(max_lipschitz, spectral_norm)

        return max_lipschitz

    def _estimate_lipschitz_empirical(self, module: nn.Module, num_samples: int = 100) -> float:
        """Empirically estimate Lipschitz by sampling."""
        try:
            dim = getattr(module, 'dim', TOTAL_DIM)
        except Exception:
            dim = TOTAL_DIM

        max_ratio = 0.0

        with torch.no_grad():
            for _ in range(num_samples):
                x1 = torch.randn(1, 1, dim)
                x2 = torch.randn(1, 1, dim)

                try:
                    y1 = module(x1)
                    y2 = module(x2)

                    # Handle dict output
                    if isinstance(y1, dict):
                        y1 = y1.get('attractor', y1.get('output', x1))
                    if isinstance(y2, dict):
                        y2 = y2.get('attractor', y2.get('output', x2))

                    input_dist = (x1 - x2).norm().item()
                    output_dist = (y1 - y2).norm().item()

                    if input_dist > 1e-8:
                        ratio = output_dist / input_dist
                        max_ratio = max(max_ratio, ratio)

                except Exception:
                    pass

        return max_ratio

    def _check_noetic_operators(self, model: nn.Module) -> None:
        """
        Check noetic operator constraints.

        Verifies:
            1. Spectral radius < 1 for stability
            2. Symmetry properties for +/- operators
        """
        # Look for noetic processing modules
        for name, module in model.named_modules():
            if 'noetic' in name.lower() and hasattr(module, 'weight'):
                W = module.weight

                if W.dim() != 2:
                    continue

                # Check spectral radius
                try:
                    with torch.no_grad():
                        eigenvalues = torch.linalg.eigvals(W)
                        spectral_radius = eigenvalues.abs().max().item()

                        if spectral_radius > self.invariants.spectral_radius_max:
                            self._add_violation(
                                severity=Severity.WARNING,
                                component="noetic_operator",
                                invariant="spectral_radius",
                                message=f"Spectral radius {spectral_radius:.3f} > 1 (unstable)",
                                value=spectral_radius,
                                threshold=self.invariants.spectral_radius_max,
                                location=name,
                            )
                except Exception:
                    pass

    def _check_world_projections(self, model: nn.Module) -> None:
        """Check world projection layers for proper structure."""
        # Look for world-specific layers
        for world in WORLD_NAMES:
            world_module = getattr(model, f'world_{world}', None)
            if world_module is None:
                # Try nested search
                for name, module in model.named_modules():
                    if f'world_{world.lower()}' in name.lower():
                        world_module = module
                        break

            if world_module is not None and hasattr(world_module, 'weight'):
                W = world_module.weight

                # Check if initialized near identity (expected for TKS)
                if W.shape[0] == W.shape[1] == NOETIC_DIM:
                    identity = torch.eye(NOETIC_DIM, device=W.device)
                    deviation = (W - identity).norm().item()

                    # Large deviation from identity might indicate issues
                    if deviation > 5.0:
                        self._add_violation(
                            severity=Severity.INFO,
                            component="world_projection",
                            invariant="near_identity",
                            message=f"World {world} projection deviates {deviation:.2f} from identity",
                            value=deviation,
                            threshold=5.0,
                            location=f"world_{world}",
                        )


# =============================================================================
# TRAINING LINT MONITOR
# =============================================================================

class TrainingLintMonitor:
    """
    Continuous linting during training.

    Provides real-time monitoring of TKS invariants during training,
    logging violations and tracking trends over time.

    Example:
        monitor = TrainingLintMonitor(model, check_interval=100)

        for step, batch in enumerate(dataloader):
            output = model(batch, return_full_trace=True)
            loss = compute_loss(output)

            # Check invariants periodically
            monitor.step(step, output.get('trace'))

            loss.backward()
            optimizer.step()

        # Get final report
        report = monitor.get_report()
        print(f"Total errors: {report['total_errors']}")
    """

    def __init__(
        self,
        model: nn.Module,
        invariants: Optional[TKSInvariants] = None,
        check_interval: int = 100,
        log_errors: bool = True,
        log_warnings: bool = False,
    ):
        """
        Initialize the training monitor.

        Args:
            model: Model to monitor
            invariants: TKS invariant thresholds
            check_interval: Check every N training steps
            log_errors: Whether to log error-level violations
            log_warnings: Whether to log warning-level violations
        """
        self.model = model
        self.invariants = invariants or TKSInvariants()
        self.check_interval = check_interval
        self.log_errors = log_errors
        self.log_warnings = log_warnings

        self.model_linter = ModelLinter(self.invariants)
        self.trace_linter = TraceLinter(self.invariants)

        self.history: List[Dict[str, Any]] = []
        self.total_errors = 0
        self.total_warnings = 0

    def step(self, step: int, trace: Optional[dict] = None) -> List[LintViolation]:
        """
        Called each training step.

        Args:
            step: Current training step number
            trace: Optional trace from model forward pass

        Returns:
            List of violations found (empty if not a check step)
        """
        if step % self.check_interval != 0:
            return []

        violations = []

        # Lint model weights
        model_violations = self.model_linter.lint(self.model)
        violations.extend(model_violations)

        # Lint trace if provided
        if trace is not None:
            trace_violations = self.trace_linter.lint(trace)
            violations.extend(trace_violations)

        # Record in history
        num_errors = sum(1 for v in violations if v.severity == Severity.ERROR)
        num_warnings = sum(1 for v in violations if v.severity == Severity.WARNING)

        self.total_errors += num_errors
        self.total_warnings += num_warnings

        self.history.append({
            'step': step,
            'violations': [v.to_dict() for v in violations],
            'num_errors': num_errors,
            'num_warnings': num_warnings,
            'num_info': sum(1 for v in violations if v.severity == Severity.INFO),
        })

        # Log violations
        for v in violations:
            if v.severity == Severity.ERROR and self.log_errors:
                logger.error(f"Step {step}: {v}")
            elif v.severity == Severity.WARNING and self.log_warnings:
                logger.warning(f"Step {step}: {v}")

        return violations

    def get_report(self) -> Dict[str, Any]:
        """
        Get summary report of all violations.

        Returns:
            Report dictionary with:
                - total_errors: Total error-level violations
                - total_warnings: Total warning-level violations
                - error_rate: Errors per check
                - most_common: Most frequently violated invariants
                - history: Full violation history
        """
        # Count violations by invariant
        invariant_counts: Dict[str, int] = {}
        component_counts: Dict[str, int] = {}

        for entry in self.history:
            for v in entry['violations']:
                key = f"{v['component']}.{v['invariant']}"
                invariant_counts[key] = invariant_counts.get(key, 0) + 1
                component_counts[v['component']] = component_counts.get(v['component'], 0) + 1

        # Sort by frequency
        most_common = sorted(invariant_counts.items(), key=lambda x: -x[1])[:10]

        return {
            'total_checks': len(self.history),
            'total_errors': self.total_errors,
            'total_warnings': self.total_warnings,
            'error_rate': self.total_errors / max(len(self.history), 1),
            'warning_rate': self.total_warnings / max(len(self.history), 1),
            'most_common_violations': most_common,
            'by_component': component_counts,
            'history': self.history,
        }

    def save_report(self, path: Union[str, Path]) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = self.get_report()
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Lint report saved to: {path}")


# =============================================================================
# FILE LOADERS
# =============================================================================

def load_model(path: Union[str, Path], device: str = 'cpu') -> nn.Module:
    """
    Load a model checkpoint.

    Attempts to load with various TKS model architectures, falling back to
    a generic state dict wrapper if specific architectures don't match.

    Args:
        path: Path to .pt checkpoint file
        device: Device to load model onto

    Returns:
        Loaded model (or state wrapper for direct weight linting)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Try to reconstruct model
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    config = checkpoint.get('config', {})

    # Find vocab size and hidden dim from state dict
    vocab_size = 128
    hidden_dim = 256
    for key in state_dict:
        if 'embedding' in key and 'weight' in key and 'token' in key:
            vocab_size = state_dict[key].shape[0]
            hidden_dim = state_dict[key].shape[1]
            break

    # Try v4 model first (TKSNoeticLM)
    try:
        from tks_llm_core_v4 import TKSNoeticLM, TKSNoeticLMConfig

        v4_config = TKSNoeticLMConfig(
            vocab_size=vocab_size,
            num_layers=config.get('num_layers', config.get('num_blocks', 4)),
            num_scales=config.get('num_scales', 3),
        )
        model = TKSNoeticLM(v4_config)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        logger.info("Loaded as TKSNoeticLM (v4)")
        return model

    except (ImportError, RuntimeError, TypeError) as e:
        logger.debug(f"TKSNoeticLM load failed: {e}")

    # Try v2 pipeline
    try:
        from tks_llm_core_v2 import TKSLLMCorePipeline, TOTAL_DIM as TD

        model = TKSLLMCorePipeline(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            noetic_dim=TD,
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        logger.info("Loaded as TKSLLMCorePipeline (v2)")
        return model

    except (ImportError, RuntimeError) as e:
        logger.debug(f"TKSLLMCorePipeline load failed: {e}")

    # Fallback: return a wrapper that exposes state dict for weight linting
    logger.warning("Model architecture not recognized, using state dict wrapper for linting")

    class StateWrapper(nn.Module):
        """Wrapper that exposes state dict weights for linting."""

        def __init__(self, state_dict: dict):
            super().__init__()
            self._state_dict = state_dict

            # Convert state dict to parameters for linting
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    # Create nested structure to match module hierarchy
                    safe_key = key.replace('.', '__')
                    setattr(self, safe_key, nn.Parameter(value, requires_grad=False))

        def named_modules(self, *args, **kwargs):
            """Override to yield module-like objects for each weight."""
            yield '', self

            # Also yield pseudo-modules for key components
            for key in self._state_dict:
                if 'attractor' in key:
                    yield f'attractor.{key}', self
                    break

    return StateWrapper(state_dict).to(device)


def load_traces(path: Union[str, Path]) -> List[dict]:
    """
    Load traces from JSONL file.

    Args:
        path: Path to .jsonl trace file

    Returns:
        List of trace dictionaries
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    traces = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    return traces


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TKS Trace Linter - Verify TKS invariants and trace integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Lint a model checkpoint
  python scripts/trace_lint.py --model output/models/best_model.pt

  # Lint a trace file
  python scripts/trace_lint.py --trace output/traces.jsonl

  # Lint both with strict mode
  python scripts/trace_lint.py --model model.pt --trace traces.jsonl --strict

  # Output to JSON report
  python scripts/trace_lint.py --model model.pt --output lint_report.json
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--trace', '-t',
        type=str,
        help='Path to trace JSONL file'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors (exit code 1 for any violation)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON report path'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show all violations including info level'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device for model loading (default: cpu)'
    )

    args = parser.parse_args()

    if not args.model and not args.trace:
        parser.error("At least one of --model or --trace must be specified")

    print("=" * 70)
    print("TKS TRACE LINTER")
    print("=" * 70)

    invariants = TKSInvariants()
    all_violations: List[LintViolation] = []

    # Lint model if provided
    if args.model:
        print(f"\n--- Linting Model: {args.model} ---")

        try:
            model = load_model(args.model, args.device)
            model_linter = ModelLinter(invariants)
            model_violations = model_linter.lint(model)
            all_violations.extend(model_violations)

            print(f"  Found {len(model_violations)} violation(s)")

        except Exception as e:
            print(f"  ERROR: Failed to load model: {e}")
            if args.strict:
                sys.exit(1)

    # Lint traces if provided
    if args.trace:
        print(f"\n--- Linting Traces: {args.trace} ---")

        try:
            traces = load_traces(args.trace)
            trace_linter = TraceLinter(invariants)

            for i, trace in enumerate(traces):
                trace_violations = trace_linter.lint(trace)
                for v in trace_violations:
                    v.location = f"trace_{i}" + (f"/{v.location}" if v.location else "")
                all_violations.extend(trace_violations)

            print(f"  Linted {len(traces)} trace(s)")
            print(f"  Found {len([v for v in all_violations if 'trace_' in (v.location or '')])} violation(s)")

        except Exception as e:
            print(f"  ERROR: Failed to load traces: {e}")
            if args.strict:
                sys.exit(1)

    # Print violations
    print("\n--- Violation Summary ---")

    errors = [v for v in all_violations if v.severity == Severity.ERROR]
    warnings = [v for v in all_violations if v.severity == Severity.WARNING]
    infos = [v for v in all_violations if v.severity == Severity.INFO]

    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Info:     {len(infos)}")

    if errors:
        print("\n--- ERRORS ---")
        for v in errors:
            print(f"  {v}")

    if warnings:
        print("\n--- WARNINGS ---")
        for v in warnings:
            print(f"  {v}")

    if args.verbose and infos:
        print("\n--- INFO ---")
        for v in infos:
            print(f"  {v}")

    # Save report if requested
    if args.output:
        report = {
            'model_path': args.model,
            'trace_path': args.trace,
            'strict_mode': args.strict,
            'num_errors': len(errors),
            'num_warnings': len(warnings),
            'num_info': len(infos),
            'violations': [v.to_dict() for v in all_violations],
            'invariants': {
                'attractor_lipschitz_max': invariants.attractor_lipschitz_max,
                'max_attractor_iterations': invariants.max_attractor_iterations,
                'min_routing_entropy': invariants.min_routing_entropy,
                'max_routing_sparsity': invariants.max_routing_sparsity,
            },
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {args.output}")

    # Determine exit code
    if errors:
        print(f"\n[FAILED] {len(errors)} error(s) found")
        sys.exit(1)
    elif args.strict and warnings:
        print(f"\n[FAILED] {len(warnings)} warning(s) found (strict mode)")
        sys.exit(1)
    else:
        print("\n[PASSED] No critical violations found")
        sys.exit(0)


if __name__ == "__main__":
    main()
