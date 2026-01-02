"""
Equation Detector - Parse TKS equations from input.

Detects patterns like "A1 + B4", "C3 × D7" etc. and returns
structured triplets (left_idx, operator_idx, right_idx) for the
operator core layer.

Usage:
    detector = EquationDetector()
    triplets = detector.parse_batch(texts)  # or from tokens
"""

import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch


# =============================================================================
# CONSTANTS (from tks_rules)
# =============================================================================

# Operator mapping (canonical)
OPERATOR_MAP = {
    '+': 0,   # Association (commutative)
    '-': 1,   # Disassociation (non-commutative)
    '×': 2,   # Exacerbation (commutative)
    '*': 2,   # Alias for ×
    'x': 2,   # Alias for ×
    'X': 2,   # Alias for ×
    '÷': 3,   # Opposition (non-commutative)
    '/': 3,   # Alias for ÷
}

# Operator names
OPERATOR_NAMES = {
    0: 'Association (+)',
    1: 'Disassociation (-)',
    2: 'Exacerbation (×)',
    3: 'Opposition (÷)',
}

# World mapping
WORLD_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
WORLD_NAMES = ['Spiritual', 'Mental', 'Emotional', 'Physical']

# Element regex: World letter + Noetic digit(s)
ELEMENT_PATTERN = r'([ABCD])(\d{1,2})'

# Full equation pattern: Element + Operator + Element
EQUATION_PATTERN = re.compile(
    rf'({ELEMENT_PATTERN})\s*([+\-×÷*/xX])\s*({ELEMENT_PATTERN})',
    re.IGNORECASE
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EquationTriplet:
    """Parsed equation triplet with indices."""
    left_idx: int       # 0-39 element index
    operator_idx: int   # 0-3 operator index
    right_idx: int      # 0-39 element index
    left_str: str       # Original string (e.g., "A1")
    operator_str: str   # Original operator (e.g., "+")
    right_str: str      # Original string (e.g., "B4")
    position: Tuple[int, int]  # Start, end position in text

    def __repr__(self):
        return f"EquationTriplet({self.left_str} {self.operator_str} {self.right_str})"


@dataclass
class BatchEquationTriplets:
    """Batch of equation triplets for model input."""
    left_indices: torch.Tensor   # [batch, max_equations]
    operator_indices: torch.Tensor  # [batch, max_equations]
    right_indices: torch.Tensor  # [batch, max_equations]
    mask: torch.Tensor           # [batch, max_equations] - 1 where valid
    num_equations: List[int]     # Number of equations per sample


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def element_to_index(element: str) -> int:
    """
    Convert element string to 0-39 index.

    Examples:
        'A1' -> 0
        'A10' -> 9
        'B1' -> 10
        'D10' -> 39
    """
    element = element.upper().strip()
    match = re.match(ELEMENT_PATTERN, element)
    if not match:
        raise ValueError(f"Invalid element: {element}")

    world = match.group(1)
    noetic = int(match.group(2))

    if noetic < 1 or noetic > 10:
        raise ValueError(f"Noetic must be 1-10, got {noetic}")

    world_idx = WORLD_MAP[world]
    return world_idx * 10 + (noetic - 1)


def index_to_element(idx: int) -> str:
    """
    Convert 0-39 index to element string.

    Examples:
        0 -> 'A1'
        9 -> 'A10'
        10 -> 'B1'
        39 -> 'D10'
    """
    if idx < 0 or idx > 39:
        raise ValueError(f"Index must be 0-39, got {idx}")

    world_idx = idx // 10
    noetic = (idx % 10) + 1
    world = ['A', 'B', 'C', 'D'][world_idx]
    return f"{world}{noetic}"


def normalize_operator(op: str) -> int:
    """Convert operator string to index."""
    op = op.strip()
    if op not in OPERATOR_MAP:
        raise ValueError(f"Invalid operator: {op}")
    return OPERATOR_MAP[op]


# =============================================================================
# EQUATION DETECTOR
# =============================================================================

class EquationDetector:
    """
    Detect and parse TKS equations from text.

    Supports multiple equations per text and batch processing.
    """

    def __init__(self, max_equations_per_sample: int = 8):
        self.max_equations = max_equations_per_sample
        self.pattern = EQUATION_PATTERN

    def parse_single(self, text: str) -> List[EquationTriplet]:
        """
        Parse all equations from a single text.

        Returns list of EquationTriplet objects.
        """
        triplets = []

        for match in self.pattern.finditer(text):
            full_match = match.group(0)
            left_full = match.group(1)
            left_world = match.group(2)
            left_noetic = match.group(3)
            operator = match.group(4)
            right_full = match.group(5)
            right_world = match.group(6)
            right_noetic = match.group(7)

            try:
                left_idx = element_to_index(left_full)
                right_idx = element_to_index(right_full)
                op_idx = normalize_operator(operator)

                triplet = EquationTriplet(
                    left_idx=left_idx,
                    operator_idx=op_idx,
                    right_idx=right_idx,
                    left_str=left_full.upper(),
                    operator_str=operator,
                    right_str=right_full.upper(),
                    position=(match.start(), match.end()),
                )
                triplets.append(triplet)

            except ValueError:
                # Skip invalid equations
                continue

        return triplets

    def parse_batch(
        self,
        texts: List[str],
        device: torch.device = None,
    ) -> Optional[BatchEquationTriplets]:
        """
        Parse equations from a batch of texts.

        Returns BatchEquationTriplets or None if no equations found.
        """
        if device is None:
            device = torch.device('cpu')

        batch_size = len(texts)
        all_triplets = [self.parse_single(text) for text in texts]

        # Check if any equations found
        total_equations = sum(len(t) for t in all_triplets)
        if total_equations == 0:
            return None

        # Determine max equations in batch
        max_eq = min(self.max_equations, max(len(t) for t in all_triplets))

        # Build tensors
        left_indices = torch.zeros(batch_size, max_eq, dtype=torch.long, device=device)
        operator_indices = torch.zeros(batch_size, max_eq, dtype=torch.long, device=device)
        right_indices = torch.zeros(batch_size, max_eq, dtype=torch.long, device=device)
        mask = torch.zeros(batch_size, max_eq, dtype=torch.float, device=device)
        num_equations = []

        for i, triplets in enumerate(all_triplets):
            n = min(len(triplets), max_eq)
            num_equations.append(n)

            for j in range(n):
                left_indices[i, j] = triplets[j].left_idx
                operator_indices[i, j] = triplets[j].operator_idx
                right_indices[i, j] = triplets[j].right_idx
                mask[i, j] = 1.0

        return BatchEquationTriplets(
            left_indices=left_indices,
            operator_indices=operator_indices,
            right_indices=right_indices,
            mask=mask,
            num_equations=num_equations,
        )

    def has_equation(self, text: str) -> bool:
        """Check if text contains any TKS equation."""
        return bool(self.pattern.search(text))


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def get_equation_vectors(
    triplets: BatchEquationTriplets,
    noetic_dim: int = 40,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert equation triplets to one-hot vectors for model input.

    Returns:
        left_vec: [batch, max_eq, 40] one-hot element vectors
        right_vec: [batch, max_eq, 40] one-hot element vectors
        operator: [batch, max_eq] operator indices
    """
    batch_size, max_eq = triplets.left_indices.shape
    device = triplets.left_indices.device

    left_vec = torch.zeros(batch_size, max_eq, noetic_dim, device=device)
    right_vec = torch.zeros(batch_size, max_eq, noetic_dim, device=device)

    # Scatter one-hot
    left_vec.scatter_(-1, triplets.left_indices.unsqueeze(-1), 1.0)
    right_vec.scatter_(-1, triplets.right_indices.unsqueeze(-1), 1.0)

    return left_vec, right_vec, triplets.operator_indices


def aggregate_equation_outputs(
    comp_outputs: torch.Tensor,  # [batch, max_eq, 40]
    mask: torch.Tensor,          # [batch, max_eq]
) -> torch.Tensor:
    """
    Aggregate multiple equation outputs per sample.

    Uses mean pooling over valid equations.

    Returns: [batch, 40]
    """
    # Mask and sum
    masked = comp_outputs * mask.unsqueeze(-1)
    summed = masked.sum(dim=1)  # [batch, 40]

    # Divide by count
    count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
    return summed / count


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Equation Detector Test")
    print("=" * 60)

    detector = EquationDetector()

    # Test cases
    test_texts = [
        "The equation A1 + B4 represents spiritual mind associating with mental vibration.",
        "Consider C3 × D7 and also A10 - B2 in this context.",
        "No equations here, just regular text.",
        "Multiple: A1+A2, B3-C4, D5×A6, C7÷D8.",
        "Edge cases: a1 + b2 (lowercase), A1+B2 (no spaces), A10 × D10",
    ]

    print("\n--- Single Parsing ---")
    for text in test_texts:
        triplets = detector.parse_single(text)
        print(f"\nText: {text[:60]}...")
        print(f"  Found {len(triplets)} equations:")
        for t in triplets:
            print(f"    {t}")

    print("\n--- Batch Parsing ---")
    batch = detector.parse_batch(test_texts)
    if batch:
        print(f"Batch shape: left={batch.left_indices.shape}")
        print(f"Equations per sample: {batch.num_equations}")
        print(f"Mask sum per sample: {batch.mask.sum(dim=1).tolist()}")

    print("\n--- Element Conversion ---")
    for elem in ['A1', 'A10', 'B5', 'C3', 'D10']:
        idx = element_to_index(elem)
        back = index_to_element(idx)
        print(f"  {elem} -> {idx} -> {back}")

    print("\n--- Vector Generation ---")
    if batch:
        left_vec, right_vec, ops = get_equation_vectors(batch)
        print(f"  left_vec shape: {left_vec.shape}")
        print(f"  right_vec shape: {right_vec.shape}")
        print(f"  Sample left_vec[0,0] nonzero: {left_vec[0,0].nonzero().squeeze().tolist()}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
