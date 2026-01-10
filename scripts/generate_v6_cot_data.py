#!/usr/bin/env python3
"""
TKS v6 Chain-of-Thought Data Generator

Generates RPM-based training data with:
1. D x W x P across 4 Worlds (12 prerequisites)
2. Scalar gradation levels (0.0-1.0)
3. Step-by-step RPM reasoning chains
4. Proper Chain-of-Thought format

Data Format:
    Input:  Natural language goal
    Output: Step-by-step RPM chain showing prerequisite checking

Example:
    Input:  "I want to start a business"
    Step 1: "Check Physical Desire (10D): 0.9 [STRONG] - PASS"
    Step 2: "Check Physical Wisdom (11W): 0.3 [LEARNING] - INSUFFICIENT"
    Step 3: "BLOCK at 11W. Sub-goal: Acquire business knowledge"
    Step 4: "Check Mental Wisdom (5W) for learning: 0.7 [KNOWING] - PASS"
    ...
    Final:  "Chain: 10D->11W(block)->5W->6P->11W(resolved)->12P->EFFECT"

Author: TKS Development
Date: 2026-01-10
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import hashlib


# =============================================================================
# CONSTANTS: TKS PREREQUISITE STRUCTURE
# =============================================================================

WORLDS = ["Spiritual", "Mental", "Emotional", "Physical"]
WORLD_CODES = ["A", "B", "C", "D"]
PREREQ_TYPES = ["Desire", "Wisdom", "Power"]
PREREQ_TYPE_CODES = ["D", "W", "P"]

# Full prerequisite names with indices
PREREQUISITES = [
    (0, "1D", "Spiritual Desire", "A", "D"),
    (1, "2W", "Spiritual Wisdom", "A", "W"),
    (2, "3P", "Spiritual Power", "A", "P"),
    (3, "4D", "Mental Desire", "B", "D"),
    (4, "5W", "Mental Wisdom", "B", "W"),
    (5, "6P", "Mental Power", "B", "P"),
    (6, "7D", "Emotional Desire", "C", "D"),
    (7, "8W", "Emotional Wisdom", "C", "W"),
    (8, "9P", "Emotional Power", "C", "P"),
    (9, "10D", "Physical Desire", "D", "D"),
    (10, "11W", "Physical Wisdom", "D", "W"),
    (11, "12P", "Physical Power", "D", "P"),
]

# Gradation levels with names
GRADATION_LEVELS = {
    "D": [  # Desire (Kings)
        (0.0, "ABSENT", "Night"),
        (0.33, "BEGINNING", "Sunrise"),
        (0.67, "WEAKENING", "Sunset"),
        (1.0, "STRONG", "Noon"),
    ],
    "W": [  # Wisdom (Jacks)
        (0.0, "NONE", "No Grip"),
        (0.33, "LEARNING", "Weak Grip"),
        (0.67, "KNOWING", "Firmer Grip"),
        (1.0, "MASTERING", "Kung Fu Grip"),
    ],
    "P": [  # Power (Queens)
        (0.0, "NONE", "No Capacity"),
        (0.33, "BIRTH", "Seedling"),
        (0.67, "ADOLESCENT", "Growing"),
        (1.0, "MATURE", "Strong Stem"),
    ],
}

# Thresholds for sufficiency (what level is "enough"?)
DEFAULT_THRESHOLD = 0.6  # Need at least KNOWING/ADOLESCENT level


# =============================================================================
# GOAL TEMPLATES
# =============================================================================

@dataclass
class Goal:
    """A goal/foundation that requires prerequisites."""
    name: str
    description: str
    primary_world: str  # Which world this goal primarily manifests in
    required_prereqs: List[int]  # Indices of prerequisites needed
    sub_goals: Dict[int, str] = field(default_factory=dict)  # prereq_idx -> sub-goal name


# Library of goals for training data
GOAL_LIBRARY = [
    Goal(
        name="Start a Business",
        description="Launch a new commercial venture",
        primary_world="D",
        required_prereqs=[9, 10, 11],  # Physical D, W, P
        sub_goals={
            9: "Cultivate entrepreneurial desire",
            10: "Learn business fundamentals",
            11: "Acquire startup capital and resources",
        }
    ),
    Goal(
        name="Write a Book",
        description="Complete a written manuscript",
        primary_world="D",
        required_prereqs=[3, 4, 5, 9, 10, 11],  # Mental + Physical
        sub_goals={
            3: "Find motivation to write",
            4: "Learn writing craft",
            5: "Develop mental discipline",
            9: "Commit to daily writing practice",
            10: "Understand publishing process",
            11: "Create writing environment and tools",
        }
    ),
    Goal(
        name="Achieve Emotional Balance",
        description="Attain stable emotional state",
        primary_world="C",
        required_prereqs=[6, 7, 8],  # Emotional D, W, P
        sub_goals={
            6: "Desire emotional growth",
            7: "Learn emotional regulation techniques",
            8: "Build emotional resilience capacity",
        }
    ),
    Goal(
        name="Deepen Spiritual Practice",
        description="Advance in spiritual development",
        primary_world="A",
        required_prereqs=[0, 1, 2],  # Spiritual D, W, P
        sub_goals={
            0: "Awaken spiritual longing",
            1: "Study spiritual teachings",
            2: "Develop spiritual discipline",
        }
    ),
    Goal(
        name="Learn a New Skill",
        description="Acquire proficiency in a new domain",
        primary_world="B",
        required_prereqs=[3, 4, 5, 9, 10],  # Mental + some Physical
        sub_goals={
            3: "Desire to learn",
            4: "Understand learning methods",
            5: "Maintain focus and concentration",
            9: "Commit to practice schedule",
            10: "Know what resources to use",
        }
    ),
    Goal(
        name="Improve Physical Health",
        description="Enhance bodily wellbeing",
        primary_world="D",
        required_prereqs=[6, 7, 9, 10, 11],  # Emotional + Physical
        sub_goals={
            6: "Want to be healthy",
            7: "Understand body-emotion connection",
            9: "Desire lifestyle change",
            10: "Know health principles",
            11: "Have access to healthy options",
        }
    ),
    Goal(
        name="Build a Relationship",
        description="Form meaningful connection with another",
        primary_world="C",
        required_prereqs=[3, 6, 7, 8, 9],
        sub_goals={
            3: "Conceptualize ideal relationship",
            6: "Open heart to connection",
            7: "Learn relationship skills",
            8: "Develop emotional availability",
            9: "Willing to invest time",
        }
    ),
    Goal(
        name="Overcome Fear",
        description="Transcend limiting fear pattern",
        primary_world="C",
        required_prereqs=[0, 1, 6, 7, 8],
        sub_goals={
            0: "Seek liberation from fear",
            1: "Understand fear's spiritual purpose",
            6: "Want to be free",
            7: "Learn fear management",
            8: "Build courage capacity",
        }
    ),
    Goal(
        name="Create Art",
        description="Produce creative artistic work",
        primary_world="D",
        required_prereqs=[0, 3, 4, 6, 9, 10, 11],
        sub_goals={
            0: "Connect to creative source",
            3: "Envision the artwork",
            4: "Understand artistic techniques",
            6: "Feel inspired",
            9: "Commit to creation",
            10: "Know materials and methods",
            11: "Have tools and space",
        }
    ),
    Goal(
        name="Solve a Problem",
        description="Find solution to a challenge",
        primary_world="B",
        required_prereqs=[3, 4, 5, 6],
        sub_goals={
            3: "Desire to solve it",
            4: "Understand problem domain",
            5: "Apply analytical thinking",
            6: "Stay motivated through difficulty",
        }
    ),
    Goal(
        name="Manifest Abundance",
        description="Attract prosperity and resources",
        primary_world="D",
        required_prereqs=[0, 1, 2, 3, 4, 6, 9, 10, 11],
        sub_goals={
            0: "Align with abundance spiritually",
            1: "Understand laws of manifestation",
            2: "Command spiritual power",
            3: "Conceive abundance mentally",
            4: "Know wealth-building principles",
            6: "Feel deserving of abundance",
            9: "Desire material prosperity",
            10: "Know how to earn/create",
            11: "Take action to manifest",
        }
    ),
    Goal(
        name="Heal from Trauma",
        description="Process and integrate past wounds",
        primary_world="C",
        required_prereqs=[0, 1, 3, 4, 6, 7, 8],
        sub_goals={
            0: "Seek healing from higher source",
            1: "Understand spiritual dimension of trauma",
            3: "Desire to heal",
            4: "Learn about trauma recovery",
            6: "Feel ready to face pain",
            7: "Know healing techniques",
            8: "Develop processing capacity",
        }
    ),
]


# =============================================================================
# STATE GENERATION
# =============================================================================

def generate_random_state(
    blocked_prereqs: Optional[List[int]] = None,
    strong_prereqs: Optional[List[int]] = None
) -> List[float]:
    """
    Generate a random prerequisite state vector.

    Args:
        blocked_prereqs: Indices to set low (blocked)
        strong_prereqs: Indices to set high (satisfied)

    Returns:
        List of 12 gradation values [0.0-1.0]
    """
    state = []

    for i in range(12):
        if blocked_prereqs and i in blocked_prereqs:
            # Blocked: low value (0.0-0.4)
            val = random.uniform(0.0, 0.4)
        elif strong_prereqs and i in strong_prereqs:
            # Strong: high value (0.7-1.0)
            val = random.uniform(0.7, 1.0)
        else:
            # Random: full range
            val = random.uniform(0.0, 1.0)
        state.append(round(val, 2))

    return state


def get_gradation_label(value: float, prereq_type: str) -> Tuple[str, str]:
    """Get the label for a gradation value."""
    levels = GRADATION_LEVELS[prereq_type]

    # Find closest level
    for i, (threshold, label, metaphor) in enumerate(levels):
        if i == len(levels) - 1:
            return label, metaphor
        next_threshold = levels[i + 1][0]
        if value < (threshold + next_threshold) / 2:
            return label, metaphor

    return levels[-1][1], levels[-1][2]


def is_sufficient(value: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Check if a prerequisite value is sufficient."""
    return value >= threshold


# =============================================================================
# RPM CHAIN GENERATION
# =============================================================================

@dataclass
class RPMStep:
    """A single step in the RPM chain."""
    step_num: int
    action: str  # "CHECK", "BLOCK", "RECURSE", "RESOLVE", "EFFECT"
    prereq_idx: Optional[int]
    prereq_name: str
    value: Optional[float]
    label: Optional[str]
    result: str  # "PASS", "FAIL", "BLOCKED", etc.
    sub_goal: Optional[str] = None


def generate_rpm_chain(
    goal: Goal,
    state: List[float],
    threshold: float = DEFAULT_THRESHOLD,
    max_depth: int = 3
) -> List[RPMStep]:
    """
    Generate an RPM reasoning chain for a goal.

    This implements the core RPM algorithm:
    1. Check prerequisites in D->W->P order per world
    2. Find first block
    3. Recurse on block (create sub-goal)
    4. Resolve and continue
    5. Output EFFECT when all satisfied
    """
    steps = []
    step_num = 0

    # Track which prereqs we've checked and resolved
    checked = set()
    resolved = set()

    # Sort prerequisites by world priority (D->B->C->A for most goals)
    # Physical first, then down the chain
    world_order = {"D": 0, "C": 1, "B": 2, "A": 3}
    if goal.primary_world == "A":
        world_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    elif goal.primary_world == "B":
        world_order = {"B": 0, "A": 1, "C": 2, "D": 3}
    elif goal.primary_world == "C":
        world_order = {"C": 0, "B": 1, "D": 2, "A": 3}

    # Sort required prereqs
    sorted_prereqs = sorted(
        goal.required_prereqs,
        key=lambda idx: (world_order[PREREQUISITES[idx][3]], PREREQUISITES[idx][4])
    )

    def check_prereq(prereq_idx: int, depth: int = 0) -> bool:
        nonlocal step_num

        if prereq_idx in resolved:
            return True

        idx, code, name, world, ptype = PREREQUISITES[prereq_idx]
        value = state[prereq_idx]
        label, metaphor = get_gradation_label(value, ptype)
        sufficient = is_sufficient(value, threshold)

        step_num += 1
        steps.append(RPMStep(
            step_num=step_num,
            action="CHECK",
            prereq_idx=prereq_idx,
            prereq_name=f"{code} ({name})",
            value=value,
            label=label,
            result="PASS" if sufficient else "INSUFFICIENT"
        ))

        checked.add(prereq_idx)

        if sufficient:
            resolved.add(prereq_idx)
            return True

        # BLOCK - need to recurse
        step_num += 1
        sub_goal = goal.sub_goals.get(prereq_idx, f"Acquire {name}")
        steps.append(RPMStep(
            step_num=step_num,
            action="BLOCK",
            prereq_idx=prereq_idx,
            prereq_name=f"{code} ({name})",
            value=value,
            label=label,
            result="BLOCKED",
            sub_goal=sub_goal
        ))

        if depth < max_depth:
            # Recurse - check what's needed for this sub-goal
            # For simplicity, check related prerequisites
            related = get_related_prereqs(prereq_idx)

            step_num += 1
            steps.append(RPMStep(
                step_num=step_num,
                action="RECURSE",
                prereq_idx=prereq_idx,
                prereq_name=sub_goal,
                value=None,
                label=None,
                result=f"Sub-goal: {sub_goal}"
            ))

            # Check related prereqs
            all_related_ok = True
            for rel_idx in related:
                if rel_idx not in resolved:
                    if not check_prereq(rel_idx, depth + 1):
                        all_related_ok = False

            if all_related_ok:
                # Simulate resolution after sub-goal work
                step_num += 1
                steps.append(RPMStep(
                    step_num=step_num,
                    action="RESOLVE",
                    prereq_idx=prereq_idx,
                    prereq_name=f"{code} ({name})",
                    value=threshold + 0.1,  # Now sufficient
                    label="SUFFICIENT",
                    result="RESOLVED"
                ))
                resolved.add(prereq_idx)
                return True

        return False

    # Main RPM loop
    all_satisfied = True
    for prereq_idx in sorted_prereqs:
        if not check_prereq(prereq_idx):
            all_satisfied = False

    # Final EFFECT step
    step_num += 1
    if all_satisfied or len(resolved) == len(goal.required_prereqs):
        steps.append(RPMStep(
            step_num=step_num,
            action="EFFECT",
            prereq_idx=None,
            prereq_name="Noetic 9",
            value=1.0,
            label="COMPLETE",
            result=f"MANIFEST: {goal.name}"
        ))
    else:
        steps.append(RPMStep(
            step_num=step_num,
            action="EFFECT",
            prereq_idx=None,
            prereq_name="Noetic 9",
            value=0.0,
            label="INCOMPLETE",
            result=f"BLOCKED: Cannot manifest {goal.name}"
        ))

    return steps


def get_related_prereqs(prereq_idx: int) -> List[int]:
    """Get prerequisites related to a given one (for sub-goal recursion)."""
    idx, code, name, world, ptype = PREREQUISITES[prereq_idx]

    related = []

    # Same world, different types
    world_start = (prereq_idx // 3) * 3
    for i in range(world_start, world_start + 3):
        if i != prereq_idx:
            related.append(i)

    # Same type, adjacent world (higher level)
    type_offset = prereq_idx % 3
    if prereq_idx >= 3:  # Has a higher world
        related.append(prereq_idx - 3)

    return related[:2]  # Limit to 2 for manageable chains


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_chain_as_text(goal: Goal, steps: List[RPMStep]) -> str:
    """Format RPM chain as readable text for training."""
    lines = []

    for step in steps:
        if step.action == "CHECK":
            lines.append(
                f"Step {step.step_num}: CHECK {step.prereq_name} = "
                f"{step.value:.2f} [{step.label}] -> {step.result}"
            )
        elif step.action == "BLOCK":
            lines.append(
                f"Step {step.step_num}: BLOCK at {step.prereq_name}. "
                f"Sub-goal needed: {step.sub_goal}"
            )
        elif step.action == "RECURSE":
            lines.append(
                f"Step {step.step_num}: RECURSE -> {step.result}"
            )
        elif step.action == "RESOLVE":
            lines.append(
                f"Step {step.step_num}: RESOLVE {step.prereq_name} = "
                f"{step.value:.2f} -> {step.result}"
            )
        elif step.action == "EFFECT":
            lines.append(
                f"Step {step.step_num}: EFFECT (Noetic 9) -> {step.result}"
            )

    return "\n".join(lines)


def format_chain_as_tks(goal: Goal, steps: List[RPMStep]) -> str:
    """Format RPM chain as TKS notation."""
    chain_parts = []

    for step in steps:
        if step.action == "CHECK" and step.result == "PASS":
            chain_parts.append(f"{PREREQUISITES[step.prereq_idx][1]}+")
        elif step.action == "BLOCK":
            chain_parts.append(f"{PREREQUISITES[step.prereq_idx][1]}!")
        elif step.action == "RESOLVE":
            chain_parts.append(f"{PREREQUISITES[step.prereq_idx][1]}*")
        elif step.action == "EFFECT":
            if "MANIFEST" in step.result:
                chain_parts.append("N9+")
            else:
                chain_parts.append("N9-")

    return " -> ".join(chain_parts)


def generate_training_example(
    goal: Goal,
    difficulty: str = "medium"
) -> Dict:
    """
    Generate a single training example.

    Args:
        goal: The goal to manifest
        difficulty: "easy" (few blocks), "medium", "hard" (many blocks)

    Returns:
        Dict with input, output, metadata
    """
    # Generate state based on difficulty
    if difficulty == "easy":
        # Most prereqs satisfied
        blocked = random.sample(goal.required_prereqs, k=min(1, len(goal.required_prereqs)))
        strong = [p for p in goal.required_prereqs if p not in blocked]
    elif difficulty == "hard":
        # Most prereqs blocked
        strong = random.sample(goal.required_prereqs, k=min(1, len(goal.required_prereqs)))
        blocked = [p for p in goal.required_prereqs if p not in strong]
    else:  # medium
        n_blocked = len(goal.required_prereqs) // 2
        blocked = random.sample(goal.required_prereqs, k=max(1, n_blocked))
        strong = [p for p in goal.required_prereqs if p not in blocked]

    state = generate_random_state(blocked_prereqs=blocked, strong_prereqs=strong)

    # Generate RPM chain
    chain = generate_rpm_chain(goal, state)

    # Format outputs
    text_chain = format_chain_as_text(goal, chain)
    tks_chain = format_chain_as_tks(goal, chain)

    # Create training example
    example = {
        "input": f"Goal: {goal.name}\nDescription: {goal.description}",
        "output": text_chain,
        "tks_notation": tks_chain,
        "metadata": {
            "goal_name": goal.name,
            "primary_world": goal.primary_world,
            "difficulty": difficulty,
            "initial_state": state,
            "num_steps": len(chain),
            "num_blocks": sum(1 for s in chain if s.action == "BLOCK"),
            "success": any(s.action == "EFFECT" and "MANIFEST" in s.result for s in chain),
        }
    }

    return example


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_dataset(
    num_examples: int = 1000,
    output_path: str = "data/v6_cot_training.jsonl",
    seed: int = 42
) -> None:
    """
    Generate full training dataset.

    Args:
        num_examples: Number of examples to generate
        output_path: Output file path
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    difficulties = ["easy", "medium", "hard"]

    print(f"Generating {num_examples} training examples...")

    for i in range(num_examples):
        # Cycle through goals and difficulties
        goal = GOAL_LIBRARY[i % len(GOAL_LIBRARY)]
        difficulty = difficulties[i % len(difficulties)]

        example = generate_training_example(goal, difficulty)
        examples.append(example)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_examples}")

    # Write to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"\nDataset saved to {output_path}")

    # Print statistics
    num_easy = sum(1 for e in examples if e['metadata']['difficulty'] == 'easy')
    num_medium = sum(1 for e in examples if e['metadata']['difficulty'] == 'medium')
    num_hard = sum(1 for e in examples if e['metadata']['difficulty'] == 'hard')
    num_success = sum(1 for e in examples if e['metadata']['success'])
    avg_steps = sum(e['metadata']['num_steps'] for e in examples) / len(examples)
    avg_blocks = sum(e['metadata']['num_blocks'] for e in examples) / len(examples)

    print(f"\nDataset Statistics:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Difficulty distribution: Easy={num_easy}, Medium={num_medium}, Hard={num_hard}")
    print(f"  Success rate: {num_success}/{len(examples)} ({100*num_success/len(examples):.1f}%)")
    print(f"  Average steps per chain: {avg_steps:.1f}")
    print(f"  Average blocks per chain: {avg_blocks:.1f}")

    # Print sample
    print(f"\n{'='*70}")
    print("SAMPLE TRAINING EXAMPLE")
    print('='*70)
    sample = examples[0]
    print(f"\nINPUT:\n{sample['input']}")
    print(f"\nOUTPUT:\n{sample['output']}")
    print(f"\nTKS NOTATION:\n{sample['tks_notation']}")


def generate_validation_set(
    num_examples: int = 100,
    output_path: str = "data/v6_cot_validation.jsonl",
    seed: int = 123
) -> None:
    """Generate validation dataset with different seed."""
    generate_dataset(num_examples, output_path, seed)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate TKS v6 CoT training data")
    parser.add_argument(
        "--num-train", type=int, default=5000,
        help="Number of training examples"
    )
    parser.add_argument(
        "--num-val", type=int, default=500,
        help="Number of validation examples"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TKS V6 CHAIN-OF-THOUGHT DATA GENERATOR")
    print("=" * 70)
    print(f"\nGenerating RPM-based training data...")
    print(f"  Training examples: {args.num_train}")
    print(f"  Validation examples: {args.num_val}")
    print(f"  Output directory: {args.output_dir}")

    # Generate training set
    train_path = f"{args.output_dir}/v6_cot_training.jsonl"
    generate_dataset(args.num_train, train_path, args.seed)

    # Generate validation set
    val_path = f"{args.output_dir}/v6_cot_validation.jsonl"
    generate_validation_set(args.num_val, val_path, args.seed + 1000)

    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"\nNext steps:")
    print(f"  1. Review sample data for correctness")
    print(f"  2. Fine-tune v5 model on this data")
    print(f"  3. Evaluate syntactic generalization")


if __name__ == "__main__":
    main()
