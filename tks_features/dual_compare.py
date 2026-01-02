"""
Dual Provider Comparison Module

This module provides functionality to run the same TKS query through both
Claude and Gemini providers and compare their results.

Usage:
    from tks_features.dual_compare import DualProviderComparison

    comparator = DualProviderComparison(
        claude_api_key="...",
        gemini_api_key="..."
    )

    result = comparator.compare("I want to start a business but I'm scared")
    print(result['claude']['equation'])
    print(result['gemini']['equation'])
    print(result['comparison']['equations_match'])

    # Or get formatted output
    print(comparator.compare_formatted("Your query here"))
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tks_wrapper import TKSWrapper, AnalysisMode, TKSAnalysisResult


@dataclass
class ComparisonResult:
    """Result of comparing two provider outputs."""
    equations_match: bool
    common_foundations: List[str]
    common_elements: List[str]
    disagreements: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "equations_match": self.equations_match,
            "common_foundations": self.common_foundations,
            "common_elements": self.common_elements,
            "disagreements": self.disagreements
        }


class DualProviderComparison:
    """
    Dual Provider Comparison System

    Runs the same TKS query through both Claude and Gemini and compares
    their analysis results, highlighting agreements and disagreements.
    """

    def __init__(
        self,
        claude_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        claude_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
        strict_validation: bool = True
    ):
        """
        Initialize dual provider comparison.

        Args:
            claude_api_key: Anthropic API key (or use env var)
            gemini_api_key: Google API key (or use env var)
            claude_model: Claude model to use
            gemini_model: Gemini model to use
            strict_validation: Enforce strict canonical validation
        """
        self.strict_validation = strict_validation

        # Initialize both providers
        try:
            self.claude_wrapper = TKSWrapper(
                provider="claude",
                api_key=claude_api_key,
                model=claude_model,
                strict_validation=strict_validation
            )
            self.claude_available = True
        except Exception as e:
            self.claude_wrapper = None
            self.claude_available = False
            self.claude_error = str(e)

        try:
            self.gemini_wrapper = TKSWrapper(
                provider="gemini",
                api_key=gemini_api_key,
                model=gemini_model,
                strict_validation=strict_validation
            )
            self.gemini_available = True
        except Exception as e:
            self.gemini_wrapper = None
            self.gemini_available = False
            self.gemini_error = str(e)

        if not self.claude_available and not self.gemini_available:
            raise RuntimeError(
                "Neither Claude nor Gemini providers could be initialized. "
                f"Claude error: {self.claude_error if hasattr(self, 'claude_error') else 'Unknown'}. "
                f"Gemini error: {self.gemini_error if hasattr(self, 'gemini_error') else 'Unknown'}"
            )

    def compare(
        self,
        user_input: str,
        mode: AnalysisMode = AnalysisMode.FULL,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the same query through both providers and compare results.

        Args:
            user_input: The situation/question to analyze
            mode: Analysis mode
            context: Optional additional context

        Returns:
            Dictionary containing:
                - 'claude': Claude's analysis result (or error)
                - 'gemini': Gemini's analysis result (or error)
                - 'comparison': Comparison dictionary
        """
        result = {
            'claude': None,
            'gemini': None,
            'comparison': None
        }

        # Run Claude analysis
        if self.claude_available:
            try:
                claude_result = self.claude_wrapper.analyze(
                    user_input=user_input,
                    mode=mode,
                    context=context
                )
                result['claude'] = claude_result.to_dict()
            except Exception as e:
                result['claude'] = {'error': str(e)}
        else:
            result['claude'] = {
                'error': f"Claude not available: {getattr(self, 'claude_error', 'Unknown error')}"
            }

        # Run Gemini analysis
        if self.gemini_available:
            try:
                gemini_result = self.gemini_wrapper.analyze(
                    user_input=user_input,
                    mode=mode,
                    context=context
                )
                result['gemini'] = gemini_result.to_dict()
            except Exception as e:
                result['gemini'] = {'error': str(e)}
        else:
            result['gemini'] = {
                'error': f"Gemini not available: {getattr(self, 'gemini_error', 'Unknown error')}"
            }

        # Generate comparison if both succeeded
        if (result['claude'] and 'error' not in result['claude'] and
            result['gemini'] and 'error' not in result['gemini']):
            result['comparison'] = self._compare_results(
                result['claude'],
                result['gemini']
            ).to_dict()
        elif result['claude'] and 'error' not in result['claude']:
            # Only Claude succeeded
            result['comparison'] = {
                'note': 'Only Claude analysis available',
                'gemini_error': result['gemini'].get('error', 'Unknown')
            }
        elif result['gemini'] and 'error' not in result['gemini']:
            # Only Gemini succeeded
            result['comparison'] = {
                'note': 'Only Gemini analysis available',
                'claude_error': result['claude'].get('error', 'Unknown')
            }
        else:
            # Both failed
            result['comparison'] = {
                'error': 'Both providers failed',
                'claude_error': result['claude'].get('error', 'Unknown'),
                'gemini_error': result['gemini'].get('error', 'Unknown')
            }

        return result

    def _compare_results(
        self,
        claude_dict: Dict[str, Any],
        gemini_dict: Dict[str, Any]
    ) -> ComparisonResult:
        """
        Compare results from both providers.

        Args:
            claude_dict: Claude's analysis dictionary
            gemini_dict: Gemini's analysis dictionary

        Returns:
            ComparisonResult with comparison details
        """
        # Compare equations
        claude_eq = claude_dict.get('equation', '')
        gemini_eq = gemini_dict.get('equation', '')
        equations_match = claude_eq == gemini_eq

        # Extract elements from equations
        claude_elements = self._extract_elements(claude_eq)
        gemini_elements = self._extract_elements(gemini_eq)
        common_elements = list(set(claude_elements) & set(gemini_elements))

        # Compare foundations
        claude_foundations = set(claude_dict.get('foundations', []))
        gemini_foundations = set(gemini_dict.get('foundations', []))
        common_foundations = list(claude_foundations & gemini_foundations)

        # Identify disagreements
        disagreements = []

        if not equations_match:
            disagreements.append(
                f"Equations differ: Claude='{claude_eq}' vs Gemini='{gemini_eq}'"
            )

        # Foundation disagreements
        claude_only = claude_foundations - gemini_foundations
        gemini_only = gemini_foundations - claude_foundations

        if claude_only:
            disagreements.append(
                f"Foundations only in Claude: {', '.join(claude_only)}"
            )
        if gemini_only:
            disagreements.append(
                f"Foundations only in Gemini: {', '.join(gemini_only)}"
            )

        # Compare RPM analysis
        claude_rpm = claude_dict.get('rpm', {})
        gemini_rpm = gemini_dict.get('rpm', {})

        for dimension in ['desire', 'wisdom', 'power']:
            claude_val = claude_rpm.get(dimension, '').lower()
            gemini_val = gemini_rpm.get(dimension, '').lower()

            # Simple check if they're substantially different
            if claude_val and gemini_val and claude_val != gemini_val:
                # Check if they share some common words
                claude_words = set(claude_val.split())
                gemini_words = set(gemini_val.split())
                common_words = claude_words & gemini_words

                if len(common_words) < 3:  # Less than 3 common words = significant difference
                    disagreements.append(
                        f"RPM {dimension.capitalize()} differs significantly"
                    )

        # Compare canonical validity
        claude_canonical = claude_dict.get('is_canonical', False)
        gemini_canonical = gemini_dict.get('is_canonical', False)

        if claude_canonical != gemini_canonical:
            disagreements.append(
                f"Canonicality differs: Claude={claude_canonical}, Gemini={gemini_canonical}"
            )

        return ComparisonResult(
            equations_match=equations_match,
            common_foundations=common_foundations,
            common_elements=common_elements,
            disagreements=disagreements
        )

    def _extract_elements(self, equation: str) -> List[str]:
        """
        Extract element codes (e.g., A1, B7, D3) from equation string.

        Args:
            equation: Equation string

        Returns:
            List of element codes
        """
        import re
        pattern = r'[ABCD]\d{1,2}'
        return re.findall(pattern, equation)

    def compare_formatted(
        self,
        user_input: str,
        mode: AnalysisMode = AnalysisMode.FULL,
        context: Optional[str] = None
    ) -> str:
        """
        Run comparison and return formatted string output.

        Args:
            user_input: The situation/question to analyze
            mode: Analysis mode
            context: Optional additional context

        Returns:
            Formatted comparison string
        """
        result = self.compare(user_input, mode, context)

        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append("DUAL PROVIDER COMPARISON - CLAUDE VS GEMINI")
        output_lines.append("=" * 80)
        output_lines.append(f"\nQuery: {user_input}\n")

        # Claude results
        output_lines.append("-" * 80)
        output_lines.append("CLAUDE ANALYSIS")
        output_lines.append("-" * 80)

        if result['claude'] and 'error' not in result['claude']:
            claude = result['claude']
            output_lines.append(f"Equation: {claude.get('equation', 'N/A')}")
            output_lines.append(f"Canonical: {claude.get('is_canonical', False)}")
            output_lines.append(f"Foundations: {', '.join(claude.get('foundations', []))}")

            rpm = claude.get('rpm', {})
            output_lines.append("\nRPM Analysis:")
            output_lines.append(f"  Desire: {rpm.get('desire', 'N/A')}")
            output_lines.append(f"  Wisdom: {rpm.get('wisdom', 'N/A')}")
            output_lines.append(f"  Power: {rpm.get('power', 'N/A')}")

            output_lines.append(f"\nGuidance: {claude.get('guidance', 'N/A')}")
        else:
            error = result['claude'].get('error', 'Unknown error') if result['claude'] else 'No result'
            output_lines.append(f"ERROR: {error}")

        # Gemini results
        output_lines.append("\n" + "-" * 80)
        output_lines.append("GEMINI ANALYSIS")
        output_lines.append("-" * 80)

        if result['gemini'] and 'error' not in result['gemini']:
            gemini = result['gemini']
            output_lines.append(f"Equation: {gemini.get('equation', 'N/A')}")
            output_lines.append(f"Canonical: {gemini.get('is_canonical', False)}")
            output_lines.append(f"Foundations: {', '.join(gemini.get('foundations', []))}")

            rpm = gemini.get('rpm', {})
            output_lines.append("\nRPM Analysis:")
            output_lines.append(f"  Desire: {rpm.get('desire', 'N/A')}")
            output_lines.append(f"  Wisdom: {rpm.get('wisdom', 'N/A')}")
            output_lines.append(f"  Power: {rpm.get('power', 'N/A')}")

            output_lines.append(f"\nGuidance: {gemini.get('guidance', 'N/A')}")
        else:
            error = result['gemini'].get('error', 'Unknown error') if result['gemini'] else 'No result'
            output_lines.append(f"ERROR: {error}")

        # Comparison
        output_lines.append("\n" + "=" * 80)
        output_lines.append("COMPARISON")
        output_lines.append("=" * 80)

        comparison = result.get('comparison', {})

        if 'error' in comparison:
            output_lines.append(f"ERROR: {comparison['error']}")
        elif 'note' in comparison:
            output_lines.append(f"NOTE: {comparison['note']}")
        else:
            output_lines.append(f"Equations Match: {comparison.get('equations_match', False)}")

            common_elements = comparison.get('common_elements', [])
            if common_elements:
                output_lines.append(f"Common Elements: {', '.join(common_elements)}")
            else:
                output_lines.append("Common Elements: None")

            common_foundations = comparison.get('common_foundations', [])
            if common_foundations:
                output_lines.append(f"Common Foundations: {', '.join(common_foundations)}")
            else:
                output_lines.append("Common Foundations: None")

            disagreements = comparison.get('disagreements', [])
            if disagreements:
                output_lines.append("\nDisagreements:")
                for i, disagreement in enumerate(disagreements, 1):
                    output_lines.append(f"  {i}. {disagreement}")
            else:
                output_lines.append("\nNo significant disagreements found!")

        output_lines.append("\n" + "=" * 80)

        return "\n".join(output_lines)


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def main():
    """Command-line interface for dual provider comparison."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Dual Provider Comparison - Compare Claude and Gemini TKS analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare both providers
  python -m tks_features.dual_compare "I want to start a business"

  # JSON output
  python -m tks_features.dual_compare "Should I move cities?" --json

  # Specify API keys
  python -m tks_features.dual_compare "I'm stressed" --claude-key sk-... --gemini-key AIza...

Environment Variables:
  ANTHROPIC_API_KEY  - For Claude provider
  GOOGLE_API_KEY     - For Gemini provider
  GEMINI_API_KEY     - Alternative for Gemini provider
"""
    )

    parser.add_argument(
        "input",
        help="Situation to analyze"
    )
    parser.add_argument(
        "--claude-key",
        help="Claude API key"
    )
    parser.add_argument(
        "--gemini-key",
        help="Gemini API key"
    )
    parser.add_argument(
        "--claude-model",
        help="Claude model to use"
    )
    parser.add_argument(
        "--gemini-model",
        help="Gemini model to use"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "equation", "coaching", "diagnostic"],
        default="full",
        help="Analysis mode"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    mode_map = {
        "full": AnalysisMode.FULL,
        "equation": AnalysisMode.EQUATION,
        "coaching": AnalysisMode.COACHING,
        "diagnostic": AnalysisMode.DIAGNOSTIC
    }

    try:
        comparator = DualProviderComparison(
            claude_api_key=args.claude_key,
            gemini_api_key=args.gemini_key,
            claude_model=args.claude_model,
            gemini_model=args.gemini_model
        )

        if args.json:
            result = comparator.compare(args.input, mode_map[args.mode])
            print(json.dumps(result, indent=2))
        else:
            output = comparator.compare_formatted(args.input, mode_map[args.mode])
            print(output)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
