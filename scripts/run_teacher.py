#!/usr/bin/env python3
"""
TKS Multi-LLM Teacher CLI

Command-line interface for the Multi-LLM Teacher ensemble system.

Commands:
    interpret   - Interpret a single TKS equation
    batch       - Process multiple equations from file
    generate    - Generate training data from equations
    validate    - Validate text against TKS canon
    test        - Run teacher with mock providers

Supported Providers:
    openai      - OpenAI (GPT-4, GPT-3.5, etc.)
    anthropic   - Anthropic (Claude 3, etc.)
    gemini      - Google Gemini (Gemini Pro, Gemini Ultra, etc.)
    local       - Local models (Ollama, vLLM, etc.)
    mock        - Mock provider for testing

Usage:
    python scripts/run_teacher.py interpret "B4 + C10 + A2"
    python scripts/run_teacher.py interpret "B4 + D10" --providers openai:gpt-4 gemini:gemini-1.5-pro
    python scripts/run_teacher.py batch equations.jsonl --output interpretations.jsonl
    python scripts/run_teacher.py generate equations.jsonl --output training_data.jsonl
    python scripts/run_teacher.py validate "The N6 (Male) represents..."
    python scripts/run_teacher.py test

NOTE: Only canonical TKS worlds (A, B, C, D) are allowed; non-canonical codes (Y, Z, etc.) are rejected by the validator.

Environment Variables:
    OPENAI_API_KEY      - OpenAI API key
    ANTHROPIC_API_KEY   - Anthropic API key
    GOOGLE_API_KEY      - Google/Gemini API key (or GEMINI_API_KEY)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from teacher import (
    MultiLLMTeacher,
    TeacherConfig,
    CanonicalValidator,
    TrainingDataTransformer,
    TKSEquation,
    TaskType,
    create_provider,
)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def parse_equation(equation_str: str) -> TKSEquation:
    """Parse equation string into TKSEquation object."""
    # Handle formats like "B4 + C10 + A2" or "B4, C10, A2" or "B4 C10 A2"
    # Only canonical worlds A, B, C, D are valid
    import re
    elements = re.findall(r'[ABCD]\d{1,2}', equation_str.upper())

    if not elements:
        raise ValueError(f"No valid elements found in: {equation_str}. Only canonical worlds A, B, C, D are allowed.")

    return TKSEquation(elements=elements)


def create_teacher_from_args(args) -> MultiLLMTeacher:
    """Create teacher from command-line arguments."""
    providers = []

    # Parse provider arguments
    if hasattr(args, 'providers') and args.providers:
        for pspec in args.providers:
            parts = pspec.split(':')
            if len(parts) >= 2:
                providers.append({
                    "type": parts[0],
                    "model": parts[1]
                })
            else:
                providers.append({"type": parts[0], "model": "default"})

    # Default to mock if no providers specified
    if not providers:
        providers = [{"type": "mock", "model": "mock-teacher"}]

    config = TeacherConfig(
        providers=providers,
        min_canon_score=getattr(args, 'min_canon', 0.8),
        strict_validation=not getattr(args, 'lenient', False)
    )

    return MultiLLMTeacher(config)


def load_equations_from_file(filepath: str) -> List[TKSEquation]:
    """Load equations from JSONL file."""
    equations = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'elements' in data:
                equations.append(TKSEquation(
                    elements=data['elements'],
                    pattern=data.get('pattern'),
                    rpm=data.get('rpm', {}),
                    foundations=data.get('foundations', [])
                ))
            elif 'equation' in data:
                equations.append(parse_equation(data['equation']))
    return equations


# ==============================================================================
# COMMANDS
# ==============================================================================

def cmd_interpret(args):
    """Interpret a single TKS equation."""
    print("=" * 60)
    print("TKS EQUATION INTERPRETATION")
    print("=" * 60)

    # Parse equation
    try:
        equation = parse_equation(args.equation)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"\nEquation: {equation.to_string()}")
    print(f"Elements: {equation.elements}")
    print(f"Noetics: {equation.noetics}")

    # Create teacher
    teacher = create_teacher_from_args(args)
    print(f"\nProviders: {[p.provider_name for p in teacher.providers]}")

    # Query
    print("\nQuerying teacher ensemble...")
    result = teacher.query(equation)

    # Display results
    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)

    print(f"\nBest Provider: {result.best_provider}")
    print(f"Best Score: {result.best_score:.2f}")
    print(f"Valid Responses: {result.valid_count}/{result.total_count}")

    print("\nScores:")
    print(f"  Agreement: {result.scores.agreement_score:.2f}")
    print(f"  Canon: {result.scores.canon_score:.2f}")
    print(f"  Confidence: {result.scores.confidence_score:.2f}")

    print("\n" + "-" * 40)
    print("INTERPRETATION")
    print("-" * 40)
    print(result.best_response)

    if args.verbose:
        print("\n" + "-" * 40)
        print("ALL RESPONSES")
        print("-" * 40)
        for i, resp in enumerate(result.responses):
            print(f"\n[{i+1}] {resp.provider_name} ({resp.model})")
            print(f"    Valid: {resp.is_valid}, Canon: {resp.canon_score:.2f}")
            if resp.validation_issues:
                print(f"    Issues: {resp.validation_issues}")
            print(f"    Text: {resp.text[:200]}...")

    return 0


def cmd_batch(args):
    """Process multiple equations from file."""
    print("=" * 60)
    print("BATCH EQUATION PROCESSING")
    print("=" * 60)

    # Load equations
    try:
        equations = load_equations_from_file(args.input)
        print(f"Loaded {len(equations)} equations from {args.input}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return 1

    # Create teacher
    teacher = create_teacher_from_args(args)

    # Process equations
    results = []
    for i, equation in enumerate(equations):
        print(f"\n[{i+1}/{len(equations)}] Processing: {equation.to_string()}")

        result = teacher.query(equation)

        results.append({
            "equation": equation.to_string(),
            "elements": equation.elements,
            "interpretation": result.best_response,
            "provider": result.best_provider,
            "scores": {
                "agreement": result.scores.agreement_score,
                "canon": result.scores.canon_score,
                "confidence": result.scores.confidence_score
            },
            "valid": result.valid_count > 0
        })

        print(f"    -> {result.best_provider}, confidence: {result.scores.confidence_score:.2f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\nSaved {len(results)} results to {args.output}")

    # Statistics
    valid_count = sum(1 for r in results if r["valid"])
    avg_confidence = sum(r["scores"]["confidence"] for r in results) / len(results)
    print(f"Valid: {valid_count}/{len(results)}, Avg Confidence: {avg_confidence:.2f}")

    return 0


def cmd_generate(args):
    """Generate training data from equations."""
    print("=" * 60)
    print("TRAINING DATA GENERATION")
    print("=" * 60)

    # Load equations
    try:
        equations = load_equations_from_file(args.input)
        print(f"Loaded {len(equations)} equations from {args.input}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return 1

    # Parse task types
    task_types = None
    if args.tasks:
        task_map = {
            "e2i": TaskType.E2I,
            "i2e": TaskType.I2E,
            "s2e": TaskType.S2E,
            "e2rpm": TaskType.E2RPM,
            "e2f": TaskType.E2F,
            "full": TaskType.FULL
        }
        task_types = [task_map[t.lower()] for t in args.tasks if t.lower() in task_map]

    # Create teacher
    teacher = create_teacher_from_args(args)

    # Generate training data
    print(f"\nGenerating training data with tasks: {[t.value for t in (task_types or list(TaskType))]}")
    examples = teacher.generate_training_data(
        equations,
        task_types=task_types,
        show_progress=True
    )

    # Save examples
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + '\n')

    print(f"\nGenerated {len(examples)} training examples")
    print(f"Saved to {args.output}")

    # Statistics
    stats = teacher.get_statistics()
    print(f"\nTeacher Statistics:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Canonical rejections: {stats['canonical_rejections']}")

    return 0


def cmd_validate(args):
    """Validate text against TKS canon."""
    print("=" * 60)
    print("CANONICAL VALIDATION")
    print("=" * 60)

    validator = CanonicalValidator(strict_mode=not args.lenient)

    # Get text to validate
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.text

    print(f"\nText ({len(text)} chars):")
    print("-" * 40)
    print(text[:500] + ("..." if len(text) > 500 else ""))
    print("-" * 40)

    # Validate
    result = validator.validate(text)

    print(f"\nVALID: {result.is_valid}")
    print(f"Canon Score: {result.canon_score:.2f}")
    print(f"Errors: {result.error_count}")
    print(f"Warnings: {result.warning_count}")

    if result.issues:
        print("\nIssues Found:")
        for issue in result.issues:
            print(f"  [{issue.severity.value.upper()}] {issue.rule}: {issue.message}")
            if issue.suggestion:
                print(f"      Suggestion: {issue.suggestion}")

    return 0 if result.is_valid else 1


def cmd_test(args):
    """Run teacher with mock providers for testing."""
    print("=" * 60)
    print("TEACHER TEST MODE")
    print("=" * 60)

    # Create mock responses
    from teacher.providers import MockProvider

    mock1 = MockProvider(model="mock-gpt4")
    mock1.add_response("B4", "The element B4 (Mental-Vibration) represents the vibrational nature of mental force. In the Mental world, N4 manifests as the underlying frequency that shapes thought. RPM: Wisdom-dominant through N4.")

    mock2 = MockProvider(model="mock-claude")
    mock2.add_response("B4", "B4 combines the mental potential of the Mental world with the principle of Vibration (N4). This element embodies the rhythmic pulse of thought, operating through the wisdom channel. The frequency foundation is primary here.")

    mock3 = MockProvider(model="mock-local")
    mock3.add_response("B4", "Element B4 is Mental-Vibration, the mental vibration. N4 in world B. Wisdom component active. Foundation: Frequency.")

    # Create teacher with mock providers
    config = TeacherConfig(providers=[])
    teacher = MultiLLMTeacher(config)
    teacher.providers = [mock1, mock2, mock3]

    # Test equation (using canonical worlds A, B, C, D only)
    test_equation = TKSEquation(elements=["B4", "C10", "A2"])

    print(f"\nTest Equation: {test_equation.to_string()}")
    print(f"Providers: {[p.model for p in teacher.providers]}")

    # Query
    print("\nQuerying...")
    result = teacher.query(test_equation)

    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)

    print(f"Best Provider: {result.best_provider}")
    print(f"Valid: {result.valid_count}/{result.total_count}")

    print("\nScores:")
    print(f"  Agreement: {result.scores.agreement_score:.3f}")
    print(f"  Canon: {result.scores.canon_score:.3f}")
    print(f"  Confidence: {result.scores.confidence_score:.3f}")

    print("\nBest Response:")
    print(result.best_response)

    print("\n" + "-" * 40)
    print("INDIVIDUAL RESPONSES")
    print("-" * 40)

    for resp in result.responses:
        print(f"\n{resp.provider_name}:")
        print(f"  Valid: {resp.is_valid}, Canon: {resp.canon_score:.2f}")
        print(f"  Text: {resp.text[:100]}...")

    # Test canonical validation
    print("\n" + "-" * 40)
    print("CANONICAL VALIDATION TESTS")
    print("-" * 40)

    validator = CanonicalValidator()

    test_cases = [
        ("Valid: N6 is Male", True),
        ("Invalid: N6 is MEL", False),
        ("Valid: Elements B4, C10, A2", True),  # Canonical worlds
        ("Invalid: Element Y10", False),         # Y is non-canonical
        ("Invalid: Element Z5", False),          # Z is non-canonical
        ("Invalid: Element X5", False),          # X is non-canonical
        ("Valid: Involution pair (2,3)", True),
    ]

    for text, expected in test_cases:
        result = validator.validate(text)
        status = "PASS" if result.is_valid == expected else "FAIL"
        print(f"  [{status}] '{text}' -> valid={result.is_valid} (expected {expected})")

    print("\nTest complete!")
    return 0


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TKS Multi-LLM Teacher CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s interpret "B4 + C10"
  %(prog)s interpret "B4 + D10" --providers openai:gpt-4 anthropic:claude-3-sonnet
  %(prog)s interpret "A2 + C10" --providers gemini:gemini-1.5-pro
  %(prog)s interpret "B4 + C10" --providers openai:gpt-4 anthropic:claude-3-sonnet gemini:gemini-1.5-pro
  %(prog)s batch equations.jsonl --output results.jsonl
  %(prog)s generate equations.jsonl --output training.jsonl --tasks e2i i2e e2rpm
  %(prog)s validate "The N6 (Male) principle represents..."
  %(prog)s test

Supported Providers:
  openai:<model>      GPT-4, GPT-3.5, etc. (requires OPENAI_API_KEY)
  anthropic:<model>   Claude 3, etc. (requires ANTHROPIC_API_KEY)
  gemini:<model>      Gemini Pro, Gemini Ultra (requires GOOGLE_API_KEY)
  local:<model>       Ollama, vLLM, etc.
  mock:<model>        Mock provider for testing

Canonical Worlds:
  Only A, B, C, D are valid world codes. Non-canonical codes (Y, Z, etc.) are rejected.
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # interpret command
    interpret_parser = subparsers.add_parser("interpret", help="Interpret a TKS equation")
    interpret_parser.add_argument("equation", help="TKS equation (e.g., 'B4 + C10 + A2') - only A,B,C,D worlds")
    interpret_parser.add_argument("--providers", "-p", nargs="+",
                                  help="Providers (format: type:model). Types: openai, anthropic, gemini, local, mock")
    interpret_parser.add_argument("--verbose", "-v", action="store_true", help="Show all responses")
    interpret_parser.add_argument("--lenient", action="store_true", help="Use lenient validation")

    # batch command
    batch_parser = subparsers.add_parser("batch", help="Process multiple equations")
    batch_parser.add_argument("input", help="Input JSONL file with equations")
    batch_parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    batch_parser.add_argument("--providers", "-p", nargs="+",
                              help="Providers (format: type:model). Types: openai, anthropic, gemini, local, mock")
    batch_parser.add_argument("--lenient", action="store_true", help="Use lenient validation")

    # generate command
    generate_parser = subparsers.add_parser("generate", help="Generate training data")
    generate_parser.add_argument("input", help="Input JSONL file with equations")
    generate_parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    generate_parser.add_argument("--providers", "-p", nargs="+",
                                 help="Providers (format: type:model). Types: openai, anthropic, gemini, local, mock")
    generate_parser.add_argument("--tasks", "-t", nargs="+", help="Task types (e2i, i2e, s2e, e2rpm, e2f, full)")
    generate_parser.add_argument("--min-canon", type=float, default=0.8, help="Min canon score")
    generate_parser.add_argument("--lenient", action="store_true", help="Use lenient validation")

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate text against canon")
    validate_parser.add_argument("text", nargs="?", help="Text to validate")
    validate_parser.add_argument("--file", "-f", help="File to validate")
    validate_parser.add_argument("--lenient", action="store_true", help="Use lenient validation")

    # test command
    test_parser = subparsers.add_parser("test", help="Run with mock providers")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "interpret": cmd_interpret,
        "batch": cmd_batch,
        "generate": cmd_generate,
        "validate": cmd_validate,
        "test": cmd_test
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
