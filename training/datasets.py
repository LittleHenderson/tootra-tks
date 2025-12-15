"""
TKS-LLM Training Datasets — PyTorch Dataset Classes

This module provides dataset classes for TKS-LLM training:
    - TKSElementsDataset: Element prediction task
    - TKSCompositionsDataset: Noetic composition task
    - TKSRPMDataset: RPM prediction task
    - TKSMultiTaskDataset: Combined multi-task dataset
    - PilotDataset: Simple dataset for pilot training

Author: TKS-LLM Data-Agent
Date: 2025-12-12
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import torch
from torch.utils.data import Dataset


# =============================================================================
# CONSTANTS
# =============================================================================

WORLDS = ["A", "B", "C", "D"]
WORLD_OFFSETS = {"A": 0, "B": 10, "C": 20, "D": 30}
NOETICS = list(range(1, 11))
NOETIC_NAMES = {
    1: "Mind", 2: "Positive", 3: "Negative", 4: "Vibration",
    5: "Female", 6: "Male", 7: "Rhythm", 8: "Cause", 9: "Effect", 10: "Idea"
}
INVOLUTION_PAIRS = [(2, 3), (5, 6), (8, 9)]
FOUNDATIONS = {
    1: {"name": "Unity", "noetics": [10]},
    2: {"name": "Wisdom", "noetics": [1, 4, 6, 7]},
    3: {"name": "Life", "noetics": [4]},
    4: {"name": "Companionship", "noetics": [2, 5]},
    5: {"name": "Power", "noetics": [6, 8]},
    6: {"name": "Material", "noetics": [10, 4, 9]},
    7: {"name": "Lust", "noetics": [5, 6, 7]},
}


# =============================================================================
# ELEMENT UTILITIES
# =============================================================================

def element_to_index(element: str) -> int:
    """
    Convert TKS element to 0-39 index.

    Args:
        element: Element string like 'B4'

    Returns:
        Index in 0-39 range
    """
    world = element[0]
    noetic = int(element[1:])
    return WORLD_OFFSETS[world] + (noetic - 1)


def index_to_element(index: int) -> str:
    """
    Convert 0-39 index to TKS element.

    Args:
        index: Index in 0-39 range

    Returns:
        Element string like 'B4'
    """
    world_idx = index // 10
    noetic = (index % 10) + 1
    world = WORLDS[world_idx]
    return f"{world}{noetic}"


def element_to_token_id(element: str, vocab_offset: int = 0) -> int:
    """
    Convert element to token ID with vocabulary offset.

    Args:
        element: Element string
        vocab_offset: Offset for special tokens (default 0)

    Returns:
        Token ID
    """
    return element_to_index(element) + vocab_offset


def compute_rpm_from_noetics(noetics: List[int]) -> Dict[str, float]:
    """
    Compute RPM scores from noetic indices.

    Args:
        noetics: List of noetic indices (1-10)

    Returns:
        Dict with desire, wisdom, power scores
    """
    desire_noetics = [n for n in noetics if n in [2, 3]]
    wisdom_noetics = [n for n in noetics if n in [1, 4, 5, 6, 7]]
    power_noetics = [n for n in noetics if n in [8, 9]]

    desire = 0.3 + 0.35 * min(len(desire_noetics), 2)
    wisdom = 0.3 + 0.14 * min(len(wisdom_noetics), 5)
    power = 0.3 + 0.35 * min(len(power_noetics), 2)

    return {
        "desire": min(1.0, desire),
        "wisdom": min(1.0, wisdom),
        "power": min(1.0, power),
    }


def assign_foundation(noetics: List[int]) -> int:
    """
    Assign foundation based on noetic overlap.

    Args:
        noetics: List of noetic indices

    Returns:
        Foundation index (1-7)
    """
    best_f = 1
    best_overlap = 0

    for f_idx, f_info in FOUNDATIONS.items():
        overlap = len(set(noetics) & set(f_info["noetics"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_f = f_idx

    return best_f


# =============================================================================
# BASE DATASET CLASS
# =============================================================================

class TKSBaseDataset(Dataset):
    """Base class for TKS datasets."""

    def __init__(
        self,
        max_seq_len: int = 8,
        vocab_size: int = 1000,
        element_vocab_offset: int = 0,
    ):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.element_vocab_offset = element_vocab_offset
        self.examples: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.examples)

    def _pad_sequence(self, seq: List[int], pad_value: int = 0) -> torch.Tensor:
        """Pad sequence to max_seq_len."""
        seq = seq[:self.max_seq_len]
        padded = seq + [pad_value] * (self.max_seq_len - len(seq))
        return torch.tensor(padded, dtype=torch.long)

    def _create_attention_mask(self, seq_len: int) -> torch.Tensor:
        """Create attention mask for sequence."""
        mask = [1] * min(seq_len, self.max_seq_len)
        mask += [0] * (self.max_seq_len - len(mask))
        return torch.tensor(mask, dtype=torch.long)


# =============================================================================
# ELEMENT PREDICTION DATASET
# =============================================================================

class TKSElementsDataset(TKSBaseDataset):
    """
    Dataset for element prediction task.

    Each example is a sequence of elements with target being next element.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        examples: Optional[List[Dict]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if data_path:
            self._load_from_file(data_path)
        elif examples:
            self.examples = examples
        else:
            raise ValueError("Must provide data_path or examples")

    def _load_from_file(self, path: str) -> None:
        """Load examples from JSONL file."""
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        # Get elements
        elements = ex.get("elements", ex.get("input_elements", []))
        target = ex.get("target_element", elements[-1] if elements else "A1")

        # Convert to token IDs
        input_ids = [element_to_token_id(e, self.element_vocab_offset) for e in elements]
        target_id = element_to_token_id(target, self.element_vocab_offset)

        # Pad and create targets (repeat target for all positions for simplicity)
        input_tensor = self._pad_sequence(input_ids)
        target_tensor = torch.full((self.max_seq_len,), target_id, dtype=torch.long)
        attention_mask = self._create_attention_mask(len(elements))

        return {
            "input_ids": input_tensor,
            "targets": target_tensor,
            "attention_mask": attention_mask,
        }

    @classmethod
    def generate_synthetic(
        cls,
        count: int = 200,
        seed: int = 42,
        **kwargs,
    ) -> "TKSElementsDataset":
        """
        Generate synthetic element prediction examples.

        Args:
            count: Number of examples to generate
            seed: Random seed
            **kwargs: Additional dataset arguments

        Returns:
            TKSElementsDataset with synthetic data
        """
        random.seed(seed)
        examples = []

        for i in range(count):
            world = WORLDS[i % 4]
            seq_len = random.randint(3, 5)
            noetics = random.sample(NOETICS, seq_len + 1)

            input_elements = [f"{world}{n}" for n in noetics[:-1]]
            target_element = f"{world}{noetics[-1]}"

            examples.append({
                "id": f"synth-elem-{i:05d}",
                "task": "element_prediction",
                "input_elements": input_elements,
                "target_element": target_element,
                "world": world,
                "complexity": "simple",
            })

        return cls(examples=examples, **kwargs)


# =============================================================================
# NOETIC COMPOSITION DATASET
# =============================================================================

class TKSCompositionsDataset(TKSBaseDataset):
    """
    Dataset for noetic composition task.

    Each example is a pair of elements with target being their composition result.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        examples: Optional[List[Dict]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if data_path:
            self._load_from_file(data_path)
        elif examples:
            self.examples = examples

    def _load_from_file(self, path: str) -> None:
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        op1 = ex["operand_1"]
        op2 = ex["operand_2"]
        result = ex["result"]

        # Input is [op1, op2], target is result
        input_ids = [
            element_to_token_id(op1, self.element_vocab_offset),
            element_to_token_id(op2, self.element_vocab_offset),
        ]
        target_id = element_to_token_id(result, self.element_vocab_offset)

        input_tensor = self._pad_sequence(input_ids)
        target_tensor = torch.full((self.max_seq_len,), target_id, dtype=torch.long)
        attention_mask = self._create_attention_mask(2)

        # Additional labels
        is_involution = torch.tensor(ex.get("is_involution", False), dtype=torch.bool)

        return {
            "input_ids": input_tensor,
            "targets": target_tensor,
            "attention_mask": attention_mask,
            "is_involution": is_involution,
        }

    @classmethod
    def generate_synthetic(
        cls,
        count: int = 200,
        seed: int = 42,
        **kwargs,
    ) -> "TKSCompositionsDataset":
        """Generate synthetic composition examples."""
        random.seed(seed)
        examples = []

        # Involution examples
        for i, (a, b) in enumerate(INVOLUTION_PAIRS):
            for world in WORLDS:
                per_pair = count // (len(INVOLUTION_PAIRS) * len(WORLDS) * 2)
                for _ in range(per_pair):
                    examples.append({
                        "id": f"synth-comp-{len(examples):05d}",
                        "task": "noetic_composition",
                        "operand_1": f"{world}{a}",
                        "operand_2": f"{world}{b}",
                        "operation": "compose",
                        "result": f"{world}10",
                        "is_involution": True,
                        "world": world,
                    })

        # Non-involution examples
        while len(examples) < count:
            world = random.choice(WORLDS)
            n1, n2 = random.sample(NOETICS, 2)
            if (n1, n2) not in INVOLUTION_PAIRS and (n2, n1) not in INVOLUTION_PAIRS:
                result_n = random.choice(NOETICS)
                examples.append({
                    "id": f"synth-comp-{len(examples):05d}",
                    "task": "noetic_composition",
                    "operand_1": f"{world}{n1}",
                    "operand_2": f"{world}{n2}",
                    "operation": "compose",
                    "result": f"{world}{result_n}",
                    "is_involution": False,
                    "world": world,
                })

        return cls(examples=examples[:count], **kwargs)


# =============================================================================
# RPM PREDICTION DATASET
# =============================================================================

class TKSRPMDataset(TKSBaseDataset):
    """
    Dataset for RPM (Desire/Wisdom/Power) prediction task.

    Each example has elements with target RPM scores.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        examples: Optional[List[Dict]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if data_path:
            self._load_from_file(data_path)
        elif examples:
            self.examples = examples

    def _load_from_file(self, path: str) -> None:
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        elements = ex["elements"]
        rpm = ex["rpm"]
        foundation = ex.get("foundation", 1)

        # Convert elements to IDs
        input_ids = [element_to_token_id(e, self.element_vocab_offset) for e in elements]

        input_tensor = self._pad_sequence(input_ids)
        attention_mask = self._create_attention_mask(len(elements))

        # RPM targets
        dwp_labels = torch.tensor([
            rpm["desire"],
            rpm["wisdom"],
            rpm["power"],
        ], dtype=torch.float)

        # Foundation target (0-indexed)
        foundation_target = torch.tensor(foundation - 1, dtype=torch.long)

        # Create dummy targets for compatibility
        target_tensor = input_tensor.clone()

        return {
            "input_ids": input_tensor,
            "targets": target_tensor,
            "attention_mask": attention_mask,
            "dwp_labels": dwp_labels,
            "target_foundation": foundation_target,
        }

    @classmethod
    def generate_synthetic(
        cls,
        count: int = 200,
        seed: int = 42,
        **kwargs,
    ) -> "TKSRPMDataset":
        """Generate synthetic RPM examples."""
        random.seed(seed)
        examples = []

        for i in range(count):
            num_elements = random.randint(2, 4)
            worlds = random.sample(WORLDS, min(2, num_elements))

            elements = []
            noetics_used = []
            for j in range(num_elements):
                world = worlds[j % len(worlds)]
                noetic = random.choice(NOETICS)
                elements.append(f"{world}{noetic}")
                noetics_used.append(noetic)

            rpm = compute_rpm_from_noetics(noetics_used)
            foundation = assign_foundation(noetics_used)

            examples.append({
                "id": f"synth-rpm-{i:05d}",
                "task": "rpm_prediction",
                "elements": elements,
                "foundation": foundation,
                "foundation_name": FOUNDATIONS[foundation]["name"],
                "rpm": rpm,
                "gate": round(rpm["desire"] * rpm["wisdom"] * rpm["power"], 3),
                "complexity": "simple" if len(worlds) == 1 else "medium",
            })

        return cls(examples=examples, **kwargs)


# =============================================================================
# MULTI-TASK DATASET
# =============================================================================

class TKSMultiTaskDataset(TKSBaseDataset):
    """
    Combined dataset supporting multiple task types.

    Supports:
        - element_prediction: Next element prediction
        - noetic_composition: Composition result prediction
        - rpm_prediction: D/W/P score prediction
        - foundation_classification: Foundation prediction
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        examples: Optional[List[Dict]] = None,
        task_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.task_weights = task_weights or {
            "element_prediction": 1.0,
            "noetic_composition": 1.0,
            "rpm_prediction": 1.0,
        }

        if data_path:
            self._load_from_file(data_path)
        elif examples:
            self.examples = examples

    def _load_from_file(self, path: str) -> None:
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        task = ex.get("task", "element_prediction")

        # Get elements
        elements = ex.get("elements", ex.get("input_elements", []))

        # Convert to IDs
        input_ids = [element_to_token_id(e, self.element_vocab_offset) for e in elements]
        input_tensor = self._pad_sequence(input_ids)
        attention_mask = self._create_attention_mask(len(elements))

        result = {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "task": task,
        }

        # Task-specific targets
        if task == "element_prediction":
            target = ex.get("target_element", elements[-1] if elements else "A1")
            target_id = element_to_token_id(target, self.element_vocab_offset)
            result["targets"] = torch.full((self.max_seq_len,), target_id, dtype=torch.long)

        elif task == "noetic_composition":
            result_elem = ex.get("result", "A10")
            target_id = element_to_token_id(result_elem, self.element_vocab_offset)
            result["targets"] = torch.full((self.max_seq_len,), target_id, dtype=torch.long)
            result["is_involution"] = torch.tensor(ex.get("is_involution", False))

        elif task == "rpm_prediction":
            rpm = ex.get("rpm", {"desire": 0.5, "wisdom": 0.5, "power": 0.5})
            result["dwp_labels"] = torch.tensor([
                rpm["desire"], rpm["wisdom"], rpm["power"]
            ], dtype=torch.float)
            result["targets"] = input_tensor.clone()

            if "foundation" in ex:
                result["target_foundation"] = torch.tensor(ex["foundation"] - 1, dtype=torch.long)

        else:
            # Default: copy input as target
            result["targets"] = input_tensor.clone()

        # Add task weight
        result["weight"] = torch.tensor(self.task_weights.get(task, 1.0), dtype=torch.float)

        return result

    @classmethod
    def generate_synthetic(
        cls,
        count: int = 400,
        seed: int = 42,
        **kwargs,
    ) -> "TKSMultiTaskDataset":
        """Generate synthetic multi-task examples."""
        elem_ds = TKSElementsDataset.generate_synthetic(count // 4, seed)
        comp_ds = TKSCompositionsDataset.generate_synthetic(count // 4, seed + 1)
        rpm_ds = TKSRPMDataset.generate_synthetic(count // 2, seed + 2)

        combined = elem_ds.examples + comp_ds.examples + rpm_ds.examples
        random.seed(seed)
        random.shuffle(combined)

        return cls(examples=combined[:count], **kwargs)


# =============================================================================
# PILOT DATASET (SIMPLE)
# =============================================================================

class PilotDataset(TKSBaseDataset):
    """
    Simple dataset for pilot training.

    Loads from JSONL and handles multiple task types.
    """

    def __init__(
        self,
        jsonl_path: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._load_from_file(jsonl_path)

    def _load_from_file(self, path: str) -> None:
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        # Get elements
        elements = ex.get("elements", ex.get("input_elements", []))
        if not elements:
            elements = ["A1"]  # Fallback

        # Convert to token IDs
        input_ids = [element_to_token_id(e, self.element_vocab_offset) for e in elements]

        # Determine target
        if "target_element" in ex:
            target_id = element_to_token_id(ex["target_element"], self.element_vocab_offset)
        elif "result" in ex:
            target_id = element_to_token_id(ex["result"], self.element_vocab_offset)
        else:
            target_id = input_ids[-1] if input_ids else 0

        # Pad sequences
        input_tensor = self._pad_sequence(input_ids)
        target_tensor = torch.full((self.max_seq_len,), target_id, dtype=torch.long)
        attention_mask = self._create_attention_mask(len(elements))

        result = {
            "input_ids": input_tensor,
            "targets": target_tensor,
            "attention_mask": attention_mask,
        }

        # Add RPM labels if present
        if "rpm" in ex:
            rpm = ex["rpm"]
            result["dwp_labels"] = torch.tensor([
                rpm["desire"], rpm["wisdom"], rpm["power"]
            ], dtype=torch.float)

        # Add foundation if present
        if "foundation" in ex:
            result["target_foundation"] = torch.tensor(ex["foundation"] - 1, dtype=torch.long)

        return result


# =============================================================================
# DATA COLLATION
# =============================================================================

def tks_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for TKS datasets.

    Handles variable presence of optional fields.
    """
    result = {}

    # Required fields
    result["input_ids"] = torch.stack([b["input_ids"] for b in batch])
    result["targets"] = torch.stack([b["targets"] for b in batch])

    # Optional fields (stack only if present in any item)
    if any("attention_mask" in b for b in batch):
        result["attention_mask"] = torch.stack([
            b.get("attention_mask", torch.zeros_like(batch[0]["input_ids"])) for b in batch
        ])

    if any("dwp_labels" in b for b in batch):
        result["dwp_labels"] = torch.stack([
            b.get("dwp_labels", torch.zeros(3, dtype=torch.float)) for b in batch
        ])

    if any("target_foundation" in b for b in batch):
        result["target_foundation"] = torch.stack([
            b.get("target_foundation", torch.tensor(0, dtype=torch.long)) for b in batch
        ])

    if any("weight" in b for b in batch):
        result["weight"] = torch.stack([
            b.get("weight", torch.tensor(1.0, dtype=torch.float)) for b in batch
        ])

    if any("is_involution" in b for b in batch):
        result["is_involution"] = torch.stack([
            b.get("is_involution", torch.tensor(False)) for b in batch
        ])

    return result


# =============================================================================
# DATA GENERATION UTILITIES
# =============================================================================

def generate_pilot_data(output_dir: str = "data/pilot", seed: int = 42) -> None:
    """
    Generate all pilot training datasets.

    Creates:
        - stage1_elements.jsonl (200 examples)
        - stage2_compositions.jsonl (200 examples)
        - stage3_rpm.jsonl (200 examples)
        - stage4_combined.jsonl (400 examples)
        - eval_holdout.jsonl (100 examples)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Stage 1: Element prediction
    elem_ds = TKSElementsDataset.generate_synthetic(200, seed)
    _save_jsonl(output_path / "stage1_elements.jsonl", elem_ds.examples)
    print(f"Generated stage1_elements.jsonl: {len(elem_ds.examples)} examples")

    # Stage 2: Compositions
    comp_ds = TKSCompositionsDataset.generate_synthetic(200, seed + 1)
    _save_jsonl(output_path / "stage2_compositions.jsonl", comp_ds.examples)
    print(f"Generated stage2_compositions.jsonl: {len(comp_ds.examples)} examples")

    # Stage 3: RPM
    rpm_ds = TKSRPMDataset.generate_synthetic(200, seed + 2)
    _save_jsonl(output_path / "stage3_rpm.jsonl", rpm_ds.examples)
    print(f"Generated stage3_rpm.jsonl: {len(rpm_ds.examples)} examples")

    # Stage 4: Combined
    multi_ds = TKSMultiTaskDataset.generate_synthetic(400, seed + 3)
    _save_jsonl(output_path / "stage4_combined.jsonl", multi_ds.examples)
    print(f"Generated stage4_combined.jsonl: {len(multi_ds.examples)} examples")

    # Eval holdout
    eval_ds = TKSMultiTaskDataset.generate_synthetic(100, seed + 100)
    _save_jsonl(output_path / "eval_holdout.jsonl", eval_ds.examples)
    print(f"Generated eval_holdout.jsonl: {len(eval_ds.examples)} examples")


def _save_jsonl(path: Path, examples: List[Dict]) -> None:
    """Save examples to JSONL file."""
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


# =============================================================================
# TESTING
# =============================================================================

def test_datasets():
    """Test dataset classes."""
    print("Testing TKS Datasets")
    print("=" * 40)

    # Test element dataset
    elem_ds = TKSElementsDataset.generate_synthetic(50)
    print(f"\nElements dataset: {len(elem_ds)} examples")
    sample = elem_ds[0]
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  targets shape: {sample['targets'].shape}")

    # Test composition dataset
    comp_ds = TKSCompositionsDataset.generate_synthetic(50)
    print(f"\nCompositions dataset: {len(comp_ds)} examples")
    sample = comp_ds[0]
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  is_involution: {sample['is_involution']}")

    # Test RPM dataset
    rpm_ds = TKSRPMDataset.generate_synthetic(50)
    print(f"\nRPM dataset: {len(rpm_ds)} examples")
    sample = rpm_ds[0]
    print(f"  dwp_labels: {sample['dwp_labels']}")

    # Test multi-task dataset
    multi_ds = TKSMultiTaskDataset.generate_synthetic(100)
    print(f"\nMulti-task dataset: {len(multi_ds)} examples")

    # Test collation
    from torch.utils.data import DataLoader
    loader = DataLoader(multi_ds, batch_size=8, collate_fn=tks_collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_datasets()
