"""
Synthetic DPS Training Data Generator

Generates training data for the Depth Permission System (DPS) layer.
DPS controls adaptive computation depth based on novelty.

Formula: NW = V × I × (1-S)
Where:
- V = Validity (is the thought correct/consistent?) [0, 1]
- I = Impact (does it help solve the problem?) [0, 1]
- S = Similarity (have we seen this before?) [0, 1] - lower = more novel

Classifications:
- HEAVY (class 0): NW >= 0.70 - Immediate unlock, can think deeper
- COUNT (class 1): 0.45 <= NW < 0.70 - Accumulate tokens toward unlock
- NOCOUNT (class 2): NW < 0.45 - No progress

Author: TKS Agent
Date: 2026-01-07
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class DPSSample:
    """A single DPS training sample."""
    # Input context (simulated noetic embedding features)
    context_features: List[float]  # 40-dim canonical embedding

    # V, I, S components
    validity: float
    impact: float
    similarity: float

    # Computed novelty weight
    novelty_weight: float

    # Classification (0=HEAVY, 1=COUNT, 2=NOCOUNT)
    novelty_class: int

    # DPS State info
    current_p_max: int
    current_tokens: int
    is_in_cooldown: bool

    # Expected next state
    should_unlock: bool
    next_p_max: int
    next_tokens: int

    # Metadata
    scenario_type: str
    description: str


def compute_novelty_weight(v: float, i: float, s: float) -> float:
    """Compute NW = V × I × (1-S)."""
    return v * i * (1.0 - s)


def classify_novelty(nw: float, heavy_thresh: float = 0.70, count_thresh: float = 0.45) -> int:
    """
    Classify novelty weight.
    Returns: 0=HEAVY, 1=COUNT, 2=NOCOUNT
    """
    if nw >= heavy_thresh:
        return 0  # HEAVY
    elif nw >= count_thresh:
        return 1  # COUNT
    else:
        return 2  # NOCOUNT


def should_unlock_dps(
    novelty_class: int,
    current_tokens: int,
    tokens_required: int = 5,
) -> bool:
    """
    Determine if DPS should unlock based on novelty.

    Rules:
    - HEAVY: Immediate unlock
    - COUNT: Unlock if tokens >= tokens_required
    - NOCOUNT: Never unlock (reset tokens)
    """
    if novelty_class == 0:  # HEAVY
        return True
    elif novelty_class == 1:  # COUNT
        return (current_tokens + 1) >= tokens_required
    else:  # NOCOUNT
        return False


def compute_next_state(
    novelty_class: int,
    current_p_max: int,
    current_tokens: int,
    max_depth: int = 5,
    tokens_required: int = 5,
    cooldown_k: int = 10,
) -> Tuple[bool, int, int]:
    """
    Compute next DPS state.

    Returns:
        should_unlock: Whether to unlock
        next_p_max: New depth limit
        next_tokens: New token count
    """
    if novelty_class == 0:  # HEAVY
        # Immediate unlock
        new_p_max = min(current_p_max + 1, max_depth)
        return True, new_p_max, 0

    elif novelty_class == 1:  # COUNT
        new_tokens = current_tokens + 1
        if new_tokens >= tokens_required:
            # Accumulated enough, unlock
            new_p_max = min(current_p_max + 1, max_depth)
            return True, new_p_max, 0
        else:
            # Keep accumulating
            return False, current_p_max, new_tokens

    else:  # NOCOUNT
        # Reset tokens, no unlock
        return False, current_p_max, 0


def generate_random_context(dim: int = 40) -> List[float]:
    """Generate random 40-dim context features (simulated noetic embedding)."""
    return [random.gauss(0, 1) for _ in range(dim)]


def generate_heavy_sample(
    current_p_max: int,
    current_tokens: int,
    in_cooldown: bool = False,
) -> DPSSample:
    """Generate a HEAVY scenario (NW >= 0.70)."""
    # High validity, high impact, low similarity (novel)
    v = random.uniform(0.85, 1.0)
    i = random.uniform(0.85, 1.0)
    s = random.uniform(0.0, 0.15)

    nw = compute_novelty_weight(v, i, s)
    nc = classify_novelty(nw)
    unlock, next_p, next_t = compute_next_state(nc, current_p_max, current_tokens)

    return DPSSample(
        context_features=generate_random_context(),
        validity=v,
        impact=i,
        similarity=s,
        novelty_weight=nw,
        novelty_class=nc,
        current_p_max=current_p_max,
        current_tokens=current_tokens,
        is_in_cooldown=in_cooldown,
        should_unlock=unlock,
        next_p_max=next_p,
        next_tokens=next_t,
        scenario_type="HEAVY",
        description="High validity, high impact, low similarity - genuine novel insight",
    )


def generate_count_sample(
    current_p_max: int,
    current_tokens: int,
    in_cooldown: bool = False,
) -> DPSSample:
    """Generate a COUNT scenario (0.45 <= NW < 0.70)."""
    # Moderate values that produce NW in COUNT range
    # Target: 0.45 <= V*I*(1-S) < 0.70

    # Try to generate valid sample
    for _ in range(100):
        v = random.uniform(0.6, 0.9)
        i = random.uniform(0.6, 0.9)
        s = random.uniform(0.1, 0.4)
        nw = compute_novelty_weight(v, i, s)
        if 0.45 <= nw < 0.70:
            break

    nc = classify_novelty(nw)
    unlock, next_p, next_t = compute_next_state(nc, current_p_max, current_tokens)

    return DPSSample(
        context_features=generate_random_context(),
        validity=v,
        impact=i,
        similarity=s,
        novelty_weight=nw,
        novelty_class=nc,
        current_p_max=current_p_max,
        current_tokens=current_tokens,
        is_in_cooldown=in_cooldown,
        should_unlock=unlock,
        next_p_max=next_p,
        next_tokens=next_t,
        scenario_type="COUNT",
        description="Moderate novelty - contributes to token accumulation",
    )


def generate_nocount_sample(
    current_p_max: int,
    current_tokens: int,
    in_cooldown: bool = False,
) -> DPSSample:
    """Generate a NOCOUNT scenario (NW < 0.45)."""
    # Low values or high similarity
    scenario = random.choice(["low_validity", "low_impact", "high_similarity", "mixed_low"])

    if scenario == "low_validity":
        v = random.uniform(0.1, 0.4)
        i = random.uniform(0.5, 0.8)
        s = random.uniform(0.2, 0.5)
        desc = "Low validity - thought is incorrect or inconsistent"
    elif scenario == "low_impact":
        v = random.uniform(0.5, 0.8)
        i = random.uniform(0.1, 0.4)
        s = random.uniform(0.2, 0.5)
        desc = "Low impact - thought doesn't help solve the problem"
    elif scenario == "high_similarity":
        v = random.uniform(0.6, 0.9)
        i = random.uniform(0.6, 0.9)
        s = random.uniform(0.7, 0.95)
        desc = "High similarity - we've seen this thought before"
    else:  # mixed_low
        v = random.uniform(0.3, 0.5)
        i = random.uniform(0.3, 0.5)
        s = random.uniform(0.4, 0.6)
        desc = "Mixed low values - mediocre thought overall"

    nw = compute_novelty_weight(v, i, s)

    # Ensure it's actually NOCOUNT
    if nw >= 0.45:
        s = min(0.95, s + 0.3)  # Increase similarity to push below threshold
        nw = compute_novelty_weight(v, i, s)

    nc = classify_novelty(nw)
    unlock, next_p, next_t = compute_next_state(nc, current_p_max, current_tokens)

    return DPSSample(
        context_features=generate_random_context(),
        validity=v,
        impact=i,
        similarity=s,
        novelty_weight=nw,
        novelty_class=nc,
        current_p_max=current_p_max,
        current_tokens=current_tokens,
        is_in_cooldown=in_cooldown,
        should_unlock=unlock,
        next_p_max=next_p,
        next_tokens=next_t,
        scenario_type="NOCOUNT",
        description=desc,
    )


def generate_episode_sequence(
    episode_length: int = 10,
    initial_p_max: int = 2,
    max_depth: int = 5,
) -> List[DPSSample]:
    """
    Generate a sequence of DPS samples simulating an episode.

    This creates more realistic training data where samples are correlated
    within an episode (e.g., unlocks lead to higher p_max for subsequent samples).
    """
    samples = []
    p_max = initial_p_max
    tokens = 0
    in_cooldown = False
    cooldown_remaining = 0

    for step in range(episode_length):
        # Decide what type of sample to generate based on distribution
        r = random.random()
        if r < 0.15:  # 15% HEAVY
            sample = generate_heavy_sample(p_max, tokens, in_cooldown)
        elif r < 0.45:  # 30% COUNT
            sample = generate_count_sample(p_max, tokens, in_cooldown)
        else:  # 55% NOCOUNT
            sample = generate_nocount_sample(p_max, tokens, in_cooldown)

        samples.append(sample)

        # Update state for next sample
        if sample.should_unlock:
            p_max = sample.next_p_max
            tokens = 0
            in_cooldown = True
            cooldown_remaining = 10
        else:
            tokens = sample.next_tokens

        # Decrement cooldown
        if in_cooldown:
            cooldown_remaining -= 1
            if cooldown_remaining <= 0:
                in_cooldown = False

    return samples


def generate_dataset(
    num_samples: int = 3000,
    episode_length: int = 10,
    class_balance: bool = True,
) -> List[DPSSample]:
    """
    Generate a complete DPS training dataset.

    Args:
        num_samples: Total number of samples to generate
        episode_length: Length of each episode sequence
        class_balance: If True, balance classes (1000 each for HEAVY/COUNT/NOCOUNT)

    Returns:
        List of DPSSample objects
    """
    samples = []

    if class_balance:
        # Generate balanced classes
        samples_per_class = num_samples // 3

        # HEAVY samples
        for _ in range(samples_per_class):
            p_max = random.randint(1, 4)
            tokens = random.randint(0, 4)
            samples.append(generate_heavy_sample(p_max, tokens))

        # COUNT samples
        for _ in range(samples_per_class):
            p_max = random.randint(1, 4)
            tokens = random.randint(0, 4)
            samples.append(generate_count_sample(p_max, tokens))

        # NOCOUNT samples
        for _ in range(samples_per_class):
            p_max = random.randint(1, 4)
            tokens = random.randint(0, 4)
            samples.append(generate_nocount_sample(p_max, tokens))

        random.shuffle(samples)
    else:
        # Generate episode sequences (more realistic)
        num_episodes = num_samples // episode_length
        for _ in range(num_episodes):
            episode = generate_episode_sequence(episode_length)
            samples.extend(episode)

    return samples[:num_samples]


def save_dataset(samples: List[DPSSample], output_path: Path, format: str = "jsonl"):
    """Save dataset to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(asdict(sample)) + "\n")
    elif format == "json":
        with open(output_path, "w") as f:
            json.dump([asdict(s) for s in samples], f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Saved {len(samples)} samples to {output_path}")


def print_dataset_stats(samples: List[DPSSample]):
    """Print dataset statistics."""
    heavy = sum(1 for s in samples if s.novelty_class == 0)
    count = sum(1 for s in samples if s.novelty_class == 1)
    nocount = sum(1 for s in samples if s.novelty_class == 2)
    unlocks = sum(1 for s in samples if s.should_unlock)

    nw_values = [s.novelty_weight for s in samples]

    print("\n" + "=" * 60)
    print("DPS Training Data Statistics")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"\nClass distribution:")
    print(f"  HEAVY (class 0):   {heavy:5d} ({heavy/len(samples)*100:.1f}%)")
    print(f"  COUNT (class 1):   {count:5d} ({count/len(samples)*100:.1f}%)")
    print(f"  NOCOUNT (class 2): {nocount:5d} ({nocount/len(samples)*100:.1f}%)")
    print(f"\nUnlock rate: {unlocks/len(samples)*100:.1f}%")
    print(f"\nNovelty Weight stats:")
    print(f"  Mean: {np.mean(nw_values):.3f}")
    print(f"  Std:  {np.std(nw_values):.3f}")
    print(f"  Min:  {np.min(nw_values):.3f}")
    print(f"  Max:  {np.max(nw_values):.3f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate DPS training data")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3000,
        help="Number of samples to generate (default: 3000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dps_training_data.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Generate class-balanced dataset",
    )
    parser.add_argument(
        "--episodes",
        action="store_true",
        help="Generate as episode sequences (more realistic, less balanced)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Generate dataset
    print(f"Generating {args.num_samples} DPS training samples...")
    samples = generate_dataset(
        num_samples=args.num_samples,
        class_balance=not args.episodes,
    )

    # Print stats
    print_dataset_stats(samples)

    # Save
    output_path = Path(args.output)
    save_dataset(samples, output_path, args.format)

    print(f"\nDone! Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
