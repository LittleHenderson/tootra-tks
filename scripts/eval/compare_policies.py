#!/usr/bin/env python3
"""
Policy Comparison Tool for TKS strg-llm-stk Integration

Compares baseline vs candidate policies by running replay sets against both
and generating a comparison report.

Usage:
    python scripts/eval/compare_policies.py \
        --baseline checkpoints/baseline.pt \
        --candidate checkpoints/candidate.pt \
        --replay-set replay_sets/rs_core.json

Author: EVAL agent
Date: 2026-01-05
"""
import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ComparisonResult(Enum):
    """Result of policy comparison."""
    IMPROVED = "IMPROVED"      # Candidate better than baseline
    REGRESSED = "REGRESSED"    # Candidate worse than baseline
    UNCHANGED = "UNCHANGED"    # Same behavior
    DIFFERENT = "DIFFERENT"    # Different but not clearly better/worse


@dataclass
class EpisodeComparison:
    """Comparison result for a single episode."""
    episode_id: str
    category: str
    baseline_result: str
    candidate_result: str
    comparison: ComparisonResult
    baseline_metrics: Dict = field(default_factory=dict)
    candidate_metrics: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class PolicyComparisonReport:
    """Complete policy comparison report."""
    baseline_path: str
    candidate_path: str
    replay_set: str
    timestamp: str
    total_episodes: int
    improved: int
    regressed: int
    unchanged: int
    different: int
    recommendation: str
    comparisons: List[EpisodeComparison] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "baseline_path": self.baseline_path,
            "candidate_path": self.candidate_path,
            "replay_set": self.replay_set,
            "timestamp": self.timestamp,
            "summary": {
                "total": self.total_episodes,
                "improved": self.improved,
                "regressed": self.regressed,
                "unchanged": self.unchanged,
                "different": self.different,
            },
            "recommendation": self.recommendation,
            "comparisons": [
                {
                    "id": c.episode_id,
                    "category": c.category,
                    "baseline": c.baseline_result,
                    "candidate": c.candidate_result,
                    "comparison": c.comparison.value,
                    "notes": c.notes,
                }
                for c in self.comparisons
            ],
        }


class PolicyComparator:
    """
    Compares two policy configurations by running replay sets.
    """

    def __init__(
        self,
        baseline_path: Optional[str] = None,
        candidate_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.baseline_path = baseline_path
        self.candidate_path = candidate_path
        self.verbose = verbose

        # Load configurations
        self.baseline_config = self._load_config(baseline_path, "baseline")
        self.candidate_config = self._load_config(candidate_path, "candidate")

    def _load_config(self, path: Optional[str], name: str) -> Dict:
        """Load policy configuration from file or return defaults."""
        if path and Path(path).exists():
            with open(path, 'r') as f:
                return json.load(f)
        else:
            # Return default configuration
            return self._get_default_config(name)

    def _get_default_config(self, name: str) -> Dict:
        """Get default policy configuration."""
        if name == "baseline":
            return {
                "name": "baseline",
                "version": "1.0.0",
                "governance": {
                    "hs_threshold": 0.45,
                    "critical_threshold": 0.70,
                    "alignment_min": 0.80,
                    "rpm_depth_cap_normal": 3,
                    "rpm_depth_cap_high_stakes": 2,
                    "rpm_breadth_cap": 5,
                    "tool_call_cap_normal": 12,
                    "tool_call_cap_high_stakes": 8,
                    "tool_call_cap_critical": 0,
                },
                "dps": {
                    "heavy_threshold": 0.70,
                    "count_threshold": 0.45,
                    "n_tokens_threshold": 5,
                    "cooldown_k": 10,
                    "max_depth": 5,
                    "initial_p_max": 2,
                },
            }
        else:
            # Candidate may have different thresholds for comparison
            return {
                "name": "candidate",
                "version": "1.1.0",
                "governance": {
                    "hs_threshold": 0.45,
                    "critical_threshold": 0.70,
                    "alignment_min": 0.80,
                    "rpm_depth_cap_normal": 3,
                    "rpm_depth_cap_high_stakes": 2,
                    "rpm_breadth_cap": 5,
                    "tool_call_cap_normal": 12,
                    "tool_call_cap_high_stakes": 8,
                    "tool_call_cap_critical": 0,
                },
                "dps": {
                    "heavy_threshold": 0.70,
                    "count_threshold": 0.45,
                    "n_tokens_threshold": 5,
                    "cooldown_k": 10,
                    "max_depth": 5,
                    "initial_p_max": 2,
                },
            }

    def compare(self, replay_set_path: str) -> PolicyComparisonReport:
        """
        Compare baseline and candidate policies on a replay set.

        Args:
            replay_set_path: Path to replay set JSON file

        Returns:
            PolicyComparisonReport with comparison results
        """
        with open(replay_set_path, 'r') as f:
            replay_set = json.load(f)

        episodes = replay_set.get('episodes', [])
        comparisons = []

        for episode in episodes:
            comparison = self._compare_episode(episode)
            comparisons.append(comparison)

            if self.verbose:
                symbol = {
                    ComparisonResult.IMPROVED: "↑",
                    ComparisonResult.REGRESSED: "↓",
                    ComparisonResult.UNCHANGED: "=",
                    ComparisonResult.DIFFERENT: "~",
                }[comparison.comparison]
                print(f"  {symbol} {comparison.episode_id}: {comparison.comparison.value}")

        # Generate recommendation
        improved = sum(1 for c in comparisons if c.comparison == ComparisonResult.IMPROVED)
        regressed = sum(1 for c in comparisons if c.comparison == ComparisonResult.REGRESSED)
        unchanged = sum(1 for c in comparisons if c.comparison == ComparisonResult.UNCHANGED)
        different = sum(1 for c in comparisons if c.comparison == ComparisonResult.DIFFERENT)

        if regressed > 0:
            recommendation = "REJECT - candidate has regressions"
        elif improved > 0 and regressed == 0:
            recommendation = "ACCEPT - candidate improves without regressions"
        elif unchanged == len(comparisons):
            recommendation = "NEUTRAL - no behavioral changes detected"
        else:
            recommendation = "REVIEW - behavioral differences need manual review"

        return PolicyComparisonReport(
            baseline_path=self.baseline_path or "default",
            candidate_path=self.candidate_path or "default",
            replay_set=replay_set_path,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_episodes=len(comparisons),
            improved=improved,
            regressed=regressed,
            unchanged=unchanged,
            different=different,
            recommendation=recommendation,
            comparisons=comparisons,
        )

    def _compare_episode(self, episode: Dict) -> EpisodeComparison:
        """Compare a single episode between baseline and candidate."""
        episode_id = episode.get('id', 'unknown')
        category = episode.get('category', 'unknown')
        expected = episode.get('expected', {})
        input_data = episode.get('input', {})

        # Run with baseline config
        baseline_result, baseline_metrics = self._run_with_config(
            input_data, expected, self.baseline_config
        )

        # Run with candidate config
        candidate_result, candidate_metrics = self._run_with_config(
            input_data, expected, self.candidate_config
        )

        # Compare results
        comparison, notes = self._determine_comparison(
            baseline_result, candidate_result,
            baseline_metrics, candidate_metrics,
            expected
        )

        return EpisodeComparison(
            episode_id=episode_id,
            category=category,
            baseline_result=baseline_result,
            candidate_result=candidate_result,
            comparison=comparison,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            notes=notes,
        )

    def _run_with_config(
        self,
        input_data: Dict,
        expected: Dict,
        config: Dict,
    ) -> Tuple[str, Dict]:
        """Run episode with given configuration."""
        metrics = {}

        # Simulate governance decision based on config
        gov_config = config.get('governance', {})
        dps_config = config.get('dps', {})

        # High-Stakes calculation
        uncertainty = input_data.get('uncertainty', 0.5)
        stakes = input_data.get('stakes', 0.5)
        alignment = input_data.get('alignment', 0.9)

        hs_score = (uncertainty * stakes) * (alignment ** 2)
        metrics['hs_score'] = hs_score

        hs_threshold = gov_config.get('hs_threshold', 0.45)
        critical_threshold = gov_config.get('critical_threshold', 0.70)
        alignment_min = gov_config.get('alignment_min', 0.80)

        # Determine mode
        if hs_score >= critical_threshold:
            mode = "Critical"
        elif hs_score >= hs_threshold:
            mode = "High-Stakes"
        else:
            mode = "Normal"
        metrics['mode'] = mode

        # Determine decision
        decision = "ALLOW"

        # Check alignment
        if alignment < alignment_min:
            decision = "PAUSE"
            metrics['reason'] = "low_alignment"

        # Check mode
        if mode == "Critical":
            decision = "BLOCK"
            metrics['reason'] = "critical_mode"
        elif mode == "High-Stakes" and decision == "ALLOW":
            decision = "PAUSE"
            metrics['reason'] = "high_stakes_mode"

        # Check depth/breadth caps
        current_depth = input_data.get('current_depth', 0)
        depth_cap = gov_config.get('rpm_depth_cap_normal', 3)
        if current_depth > depth_cap:
            decision = "BLOCK"
            metrics['reason'] = "depth_exceeded"

        current_breadth = input_data.get('current_breadth', 0)
        breadth_cap = gov_config.get('rpm_breadth_cap', 5)
        if current_breadth > breadth_cap:
            decision = "BLOCK"
            metrics['reason'] = "breadth_exceeded"

        # Check tool caps
        tool_calls = input_data.get('tool_calls_this_episode', 0)
        tool_cap = gov_config.get('tool_call_cap_normal', 12)
        if mode == "High-Stakes":
            tool_cap = gov_config.get('tool_call_cap_high_stakes', 8)
        elif mode == "Critical":
            tool_cap = gov_config.get('tool_call_cap_critical', 0)
        if tool_calls >= tool_cap:
            decision = "BLOCK"
            metrics['reason'] = "tool_cap_reached"

        # Check irreversibility
        action = input_data.get('action', {})
        if action.get('reversibility') == 'irreversible':
            clearance = input_data.get('clearance_token')
            if not clearance:
                decision = "BLOCK"
                metrics['reason'] = "no_clearance"

        metrics['decision'] = decision
        return decision, metrics

    def _determine_comparison(
        self,
        baseline_result: str,
        candidate_result: str,
        baseline_metrics: Dict,
        candidate_metrics: Dict,
        expected: Dict,
    ) -> Tuple[ComparisonResult, List[str]]:
        """Determine comparison result between baseline and candidate."""
        notes = []

        # Same result
        if baseline_result == candidate_result:
            return ComparisonResult.UNCHANGED, ["No behavioral change"]

        expected_decision = expected.get('decision')

        # Check if candidate matches expected better
        baseline_matches = baseline_result == expected_decision
        candidate_matches = candidate_result == expected_decision

        if candidate_matches and not baseline_matches:
            notes.append(f"Candidate now matches expected decision: {expected_decision}")
            return ComparisonResult.IMPROVED, notes

        if baseline_matches and not candidate_matches:
            notes.append(f"Candidate no longer matches expected decision: {expected_decision}")
            return ComparisonResult.REGRESSED, notes

        # Both match or both don't match but different
        notes.append(f"Baseline: {baseline_result}, Candidate: {candidate_result}")
        return ComparisonResult.DIFFERENT, notes


def main():
    parser = argparse.ArgumentParser(description="Compare TKS policy configurations")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline policy config (JSON)",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Path to candidate policy config (JSON)",
    )
    parser.add_argument(
        "--replay-set",
        type=str,
        default="replay_sets/rs_core.json",
        help="Path to replay set JSON file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for JSON report",
    )

    args = parser.parse_args()

    print(f"Comparing policies on: {args.replay_set}")
    print(f"  Baseline:  {args.baseline or 'default'}")
    print(f"  Candidate: {args.candidate or 'default'}")
    print("-" * 60)

    comparator = PolicyComparator(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        verbose=args.verbose,
    )

    report = comparator.compare(args.replay_set)

    print("-" * 60)
    print("Summary:")
    print(f"  Improved:  {report.improved}")
    print(f"  Regressed: {report.regressed}")
    print(f"  Unchanged: {report.unchanged}")
    print(f"  Different: {report.different}")
    print(f"\nRecommendation: {report.recommendation}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to: {args.output}")

    # Exit with failure if any regressions
    sys.exit(0 if report.regressed == 0 else 1)


if __name__ == "__main__":
    main()
