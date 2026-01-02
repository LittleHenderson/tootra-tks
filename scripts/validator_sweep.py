#!/usr/bin/env python3
"""
TKS Canon Validation Sweep

Runs CanonicalValidator across datasets to ensure 100% canon compliance.
Reports pass rates and identifies any failures.

Usage:
    python scripts/validator_sweep.py data/attractor_stress.jsonl
    python scripts/validator_sweep.py data/lacunary_benchmark.jsonl --output report.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from teacher.validator import CanonicalValidator


def validate_dataset(input_file: str, verbose: bool = False) -> dict:
    """Validate all equations in a JSONL dataset."""
    validator = CanonicalValidator(strict_mode=True)

    total = 0
    valid = 0
    invalid = 0
    scores = []
    failures = []

    with open(input_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                failures.append({
                    'line': line_num,
                    'error': f'JSON parse error: {e}',
                    'content': line[:100]
                })
                invalid += 1
                total += 1
                continue

            equation = entry.get('equation', '')
            if not equation:
                failures.append({
                    'line': line_num,
                    'error': 'Missing equation field',
                    'content': str(entry)[:100]
                })
                invalid += 1
                total += 1
                continue

            result = validator.validate(equation)
            total += 1
            scores.append(result.canon_score)

            if result.is_valid:
                valid += 1
                if verbose:
                    print(f"[PASS] Line {line_num}: {equation} (score: {result.canon_score})")
            else:
                invalid += 1
                failures.append({
                    'line': line_num,
                    'equation': equation,
                    'canon_score': result.canon_score,
                    'issues': result.issues
                })
                if verbose:
                    print(f"[FAIL] Line {line_num}: {equation}")
                    for issue in result.issues:
                        print(f"       - {issue}")

    pass_rate = (valid / total * 100) if total > 0 else 0.0

    return {
        'input_file': str(input_file),
        'total_count': total,
        'valid_count': valid,
        'invalid_count': invalid,
        'pass_rate': round(pass_rate, 2),
        'canon_score_stats': {
            'min': round(min(scores), 4) if scores else 0.0,
            'max': round(max(scores), 4) if scores else 0.0,
            'mean': round(sum(scores) / len(scores), 4) if scores else 0.0
        },
        'failures': failures
    }


def main():
    parser = argparse.ArgumentParser(description="TKS Canon Validation Sweep")
    parser.add_argument("input", nargs="+", help="Input JSONL file(s) to validate")
    parser.add_argument("--output", "-o", help="Output JSON report file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fail-threshold", type=float, default=100.0,
                       help="Minimum pass rate to succeed (default: 100%%)")
    args = parser.parse_args()

    print("TKS Canon Validation Sweep")
    print("=" * 50)

    all_results = []
    overall_pass = True

    for input_file in args.input:
        if not Path(input_file).exists():
            print(f"WARNING: File not found: {input_file}")
            continue

        print(f"\nValidating: {input_file}")
        result = validate_dataset(input_file, args.verbose)
        all_results.append(result)

        status = "PASS" if result['pass_rate'] >= args.fail_threshold else "FAIL"
        print(f"  Total: {result['total_count']}")
        print(f"  Valid: {result['valid_count']}")
        print(f"  Invalid: {result['invalid_count']}")
        print(f"  Pass Rate: {result['pass_rate']}% [{status}]")
        print(f"  Canon Scores: min={result['canon_score_stats']['min']}, "
              f"max={result['canon_score_stats']['max']}, "
              f"mean={result['canon_score_stats']['mean']}")

        if result['pass_rate'] < args.fail_threshold:
            overall_pass = False
            if result['failures']:
                print(f"  First few failures:")
                for f in result['failures'][:3]:
                    print(f"    Line {f.get('line', '?')}: {f.get('equation', f.get('error', 'unknown'))}")

    if args.output:
        report = {
            'datasets': all_results,
            'overall_pass': overall_pass,
            'fail_threshold': args.fail_threshold
        }
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to: {args.output}")

    print()
    print("=" * 50)
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
