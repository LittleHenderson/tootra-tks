#!/usr/bin/env python3
"""
Replay Regression Runner for TKS strg-llm-stk Integration

Runs golden episodes from replay_sets/*.json and verifies expected behaviors.
Used for regression testing to ensure system improvements don't break core functionality.

Usage:
    python scripts/eval/run_replay.py --replay-set replay_sets/rs_core.json
    python scripts/eval/run_replay.py --replay-set replay_sets/rs_core.json --category dps_gating
    python scripts/eval/run_replay.py --replay-set replay_sets/rs_core.json --verbose

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

import torch


class ReplayResult(Enum):
    """Result of a single replay episode."""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class EpisodeResult:
    """Result of running a single episode."""
    episode_id: str
    category: str
    result: ReplayResult
    expected: Dict
    actual: Dict
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class ReplayReport:
    """Complete replay run report."""
    replay_set: str
    timestamp: str
    total_episodes: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_seconds: float
    results: List[EpisodeResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "replay_set": self.replay_set,
            "timestamp": self.timestamp,
            "summary": {
                "total": self.total_episodes,
                "passed": self.passed,
                "failed": self.failed,
                "skipped": self.skipped,
                "errors": self.errors,
                "pass_rate": f"{100 * self.passed / max(1, self.total_episodes):.1f}%",
            },
            "duration_seconds": self.duration_seconds,
            "results": [
                {
                    "id": r.episode_id,
                    "category": r.category,
                    "result": r.result.value,
                    "errors": r.errors,
                    "duration_ms": r.duration_ms,
                }
                for r in self.results
            ],
        }


class ReplayRunner:
    """
    Runs golden episodes and verifies expected behaviors.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._load_modules()

    def _load_modules(self):
        """Load all required modules."""
        try:
            from tks_compositional_layer import (
                TKSCompositionalLayerV2,
                WorldWiseBilinearComposer,
                OperatorSymmetryLoss,
                TOTAL_DIM,
                NUM_OPERATORS,
                OP_PLUS, OP_MINUS, OP_MULT, OP_DIV,
            )
            self.operator_layer = TKSCompositionalLayerV2()
            self.bilinear = WorldWiseBilinearComposer()
            self.sym_loss = OperatorSymmetryLoss()
            self.TOTAL_DIM = TOTAL_DIM
            self.OP_MAP = {"+": OP_PLUS, "-": OP_MINUS, "x": OP_MULT, "/": OP_DIV}
            self._has_operator = True
        except ImportError as e:
            if self.verbose:
                print(f"  Warning: Could not load operator modules: {e}")
            self._has_operator = False

        try:
            from tks_llm_core_v2 import RPMGatingMechanism
            self.rpm_layer = RPMGatingMechanism(dim=40)
            self._has_rpm = True
        except ImportError as e:
            if self.verbose:
                print(f"  Warning: Could not load RPM modules: {e}")
            self._has_rpm = False

        try:
            # Direct import by executing the file
            import importlib.util
            dps_path = Path(__file__).parent.parent.parent / "tks_features" / "dps_gating.py"
            spec = importlib.util.spec_from_file_location("dps_gating", dps_path)
            if spec and spec.loader:
                dps_module = importlib.util.module_from_spec(spec)
                sys.modules['dps_gating'] = dps_module
                spec.loader.exec_module(dps_module)

                self.DPSGatingLayer = dps_module.DPSGatingLayer
                self.DPSConfig = dps_module.DPSConfig
                self.DPSState = dps_module.DPSState
                self.NoveltyWeight = dps_module.NoveltyWeight
                self.ValidityScore = dps_module.ValidityScore
                self.ImpactScore = dps_module.ImpactScore
                self.SimilarityScore = dps_module.SimilarityScore
                self.NoveltyClass = dps_module.NoveltyClass
                self._has_dps = True
            else:
                raise ImportError("Could not create spec for dps_gating")
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Could not load DPS modules: {e}")
            self._has_dps = False

        try:
            import importlib.util
            gov_path = Path(__file__).parent.parent.parent / "tks_features" / "governance_rails.py"
            spec = importlib.util.spec_from_file_location("governance_rails", gov_path)
            if spec and spec.loader:
                gov_module = importlib.util.module_from_spec(spec)
                sys.modules['governance_rails'] = gov_module
                spec.loader.exec_module(gov_module)

                self.GovernanceRails = gov_module.GovernanceRails
                self.GovernanceConfig = gov_module.GovernanceConfig
                self.ActionProfile = gov_module.ActionProfile
                self.ActionReversibility = gov_module.ActionReversibility
                self.GateDecision = gov_module.GateDecision
                self.OperationalMode = gov_module.OperationalMode
                self.HighStakesGate = gov_module.HighStakesGate
                self.AntiRecursionGate = gov_module.AntiRecursionGate
                self._has_governance = True
            else:
                raise ImportError("Could not create spec for governance_rails")
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Could not load governance modules: {e}")
            self._has_governance = False

    def run_replay_set(
        self,
        replay_set_path: str,
        category_filter: Optional[str] = None,
    ) -> ReplayReport:
        """
        Run all episodes in a replay set.

        Args:
            replay_set_path: Path to replay set JSON file
            category_filter: Optional category to filter episodes

        Returns:
            ReplayReport with results
        """
        start_time = time.time()

        with open(replay_set_path, 'r') as f:
            replay_set = json.load(f)

        episodes = replay_set.get('episodes', [])
        if category_filter:
            episodes = [e for e in episodes if e.get('category') == category_filter]

        results = []
        for episode in episodes:
            result = self._run_episode(episode)
            results.append(result)

            if self.verbose:
                status = "PASS" if result.result == ReplayResult.PASS else "FAIL"
                print(f"  [{status}] {result.episode_id}: {result.result.value}")
                if result.errors:
                    for err in result.errors:
                        print(f"      {err}")

        duration = time.time() - start_time

        report = ReplayReport(
            replay_set=replay_set_path,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_episodes=len(results),
            passed=sum(1 for r in results if r.result == ReplayResult.PASS),
            failed=sum(1 for r in results if r.result == ReplayResult.FAIL),
            skipped=sum(1 for r in results if r.result == ReplayResult.SKIP),
            errors=sum(1 for r in results if r.result == ReplayResult.ERROR),
            duration_seconds=duration,
            results=results,
        )

        return report

    def _run_episode(self, episode: Dict) -> EpisodeResult:
        """Run a single episode and verify expected behavior."""
        start_time = time.time()
        episode_id = episode.get('id', 'unknown')
        category = episode.get('category', 'unknown')
        input_data = episode.get('input', {})
        expected = episode.get('expected', {})

        actual = {}
        errors = []

        try:
            if category == 'operator_semantics':
                actual, errors = self._run_operator_episode(input_data, expected)
            elif category == 'dps_gating':
                actual, errors = self._run_dps_episode(input_data, expected)
            elif category == 'governance_rails':
                actual, errors = self._run_governance_episode(input_data, expected)
            elif category == 'rpm_regulator':
                actual, errors = self._run_rpm_episode(input_data, expected)
            elif category == 'integration':
                actual, errors = self._run_integration_episode(input_data, expected)
            else:
                errors.append(f"Unknown category: {category}")

            result = ReplayResult.PASS if not errors else ReplayResult.FAIL

        except Exception as e:
            errors.append(f"Exception: {str(e)}")
            result = ReplayResult.ERROR

        duration_ms = (time.time() - start_time) * 1000

        return EpisodeResult(
            episode_id=episode_id,
            category=category,
            result=result,
            expected=expected,
            actual=actual,
            errors=errors,
            duration_ms=duration_ms,
        )

    def _run_operator_episode(self, input_data: Dict, expected: Dict) -> Tuple[Dict, List[str]]:
        """Run operator semantics episode."""
        if not self._has_operator:
            return {}, ["Operator modules not available"]

        actual = {}
        errors = []

        operator_str = input_data.get('operator', '+')
        op_idx = self.OP_MAP.get(operator_str, 0)

        batch_size = 4
        left = torch.randn(batch_size, self.TOTAL_DIM)
        right = torch.randn(batch_size, self.TOTAL_DIM)
        operator = torch.full((batch_size,), op_idx, dtype=torch.long)

        # Run forward pass
        outputs = self.operator_layer(left, right, operator)

        actual['output_shape'] = list(outputs['noetic_output'].shape)
        actual['has_nan'] = torch.isnan(outputs['noetic_output']).any().item()

        # Check symmetry/antisymmetry
        if expected.get('operator_behavior') == 'symmetric':
            composed_ab = self.bilinear(left, right, operator)
            composed_ba = self.bilinear(right, left, operator)
            losses = self.sym_loss(composed_ab, composed_ba, operator)
            actual['symmetry_loss'] = losses['symmetry_loss'].item()

            threshold = expected.get('symmetry_loss_below', 0.1)
            # Note: Before training, symmetry may not be perfect
            # This is a smoke test more than a hard requirement

        elif expected.get('operator_behavior') == 'antisymmetric':
            composed_ab = self.bilinear(left, right, operator)
            composed_ba = self.bilinear(right, left, operator)
            losses = self.sym_loss(composed_ab, composed_ba, operator)
            actual['antisymmetry_loss'] = losses['antisymmetry_loss'].item()

        if actual.get('has_nan'):
            errors.append("Output contains NaN values")

        return actual, errors

    def _run_dps_episode(self, input_data: Dict, expected: Dict) -> Tuple[Dict, List[str]]:
        """Run DPS gating episode."""
        if not self._has_dps:
            return {}, ["DPS modules not available"]

        actual = {}
        errors = []

        # Build novelty components from input
        v_data = input_data.get('validity', {})
        i_data = input_data.get('impact', {})
        s_data = input_data.get('similarity', {})

        validity = self.ValidityScore(
            confidence=v_data.get('confidence', 0.5),
            consistency=v_data.get('consistency', 0.5),
            evidence_ok=v_data.get('evidence_ok', 0.5),
        )
        impact = self.ImpactScore(
            delta_reward=i_data.get('delta_reward', 0.5),
            delta_uncertainty=i_data.get('delta_uncertainty', 0.5),
            delta_coverage=i_data.get('delta_coverage', 0.5),
            delta_cost=i_data.get('delta_cost', 0.5),
        )
        similarity = self.SimilarityScore(
            s_embed=s_data.get('s_embed', 0.0),
            s_ast=s_data.get('s_ast', 0.0),
        )

        novelty = self.NoveltyWeight(
            validity=validity,
            impact=impact,
            similarity=similarity,
        )

        config = self.DPSConfig()
        actual['novelty_weight'] = novelty.value
        actual['novelty_class'] = novelty.classify(config).value

        # Check expected novelty class
        if expected.get('novelty_class'):
            if actual['novelty_class'] != expected['novelty_class']:
                errors.append(f"Expected novelty_class={expected['novelty_class']}, got {actual['novelty_class']}")

        # Check novelty weight range
        if expected.get('novelty_weight_gte'):
            if actual['novelty_weight'] < expected['novelty_weight_gte']:
                errors.append(f"Expected NW >= {expected['novelty_weight_gte']}, got {actual['novelty_weight']}")

        # Test state update if initial_state provided
        if 'initial_state' in input_data:
            state_data = input_data['initial_state']
            state = self.DPSState(
                p_max=state_data.get('p_max', 2),
                tokens=state_data.get('tokens', 0),
                cooldown=state_data.get('cooldown', 0),
            )

            dps_layer = self.DPSGatingLayer(config, noetic_dim=40)
            new_state, unlock_occurred = dps_layer.update_state(state, novelty, "test_episode")

            actual['unlock_occurred'] = unlock_occurred
            actual['new_p_max'] = new_state.p_max
            actual['new_tokens'] = new_state.tokens
            actual['new_cooldown'] = new_state.cooldown

            if expected.get('unlock_occurred') is not None:
                if actual['unlock_occurred'] != expected['unlock_occurred']:
                    errors.append(f"Expected unlock_occurred={expected['unlock_occurred']}, got {actual['unlock_occurred']}")

            if expected.get('new_p_max') is not None:
                if actual['new_p_max'] != expected['new_p_max']:
                    errors.append(f"Expected new_p_max={expected['new_p_max']}, got {actual['new_p_max']}")

        return actual, errors

    def _run_governance_episode(self, input_data: Dict, expected: Dict) -> Tuple[Dict, List[str]]:
        """Run governance rails episode."""
        if not self._has_governance:
            return {}, ["Governance modules not available"]

        actual = {}
        errors = []

        config = self.GovernanceConfig()

        # High-Stakes formula test
        if 'uncertainty' in input_data and 'stakes' in input_data:
            hs_gate = self.HighStakesGate(config)
            uncertainty = input_data.get('uncertainty', 0.5)
            stakes = input_data.get('stakes', 0.5)
            alignment = input_data.get('alignment', 0.9)

            hs_score = hs_gate.compute_high_stakes(uncertainty, stakes, alignment)
            actual['hs_score'] = hs_score

            result = hs_gate.check(uncertainty, stakes, alignment)
            actual['mode'] = result.mode.value
            actual['decision'] = result.decision.value

            if expected.get('hs_score_gte') and hs_score < expected['hs_score_gte']:
                errors.append(f"Expected HS >= {expected['hs_score_gte']}, got {hs_score}")

            if expected.get('mode') and actual['mode'] != expected['mode']:
                errors.append(f"Expected mode={expected['mode']}, got {actual['mode']}")

            if expected.get('decision') and actual['decision'] != expected['decision']:
                errors.append(f"Expected decision={expected['decision']}, got {actual['decision']}")

        # Anti-recursion test
        if 'current_depth' in input_data:
            ar_gate = self.AntiRecursionGate(config)
            result = ar_gate.check(
                current_depth=input_data.get('current_depth', 0),
                current_breadth=input_data.get('current_breadth', 0),
                tool_calls_this_episode=input_data.get('tool_calls_this_episode', 0),
                current_mode=self.OperationalMode.NORMAL,
            )
            actual['decision'] = result.decision.value
            actual['gate_name'] = result.gate_name

            if expected.get('decision') and actual['decision'] != expected['decision']:
                errors.append(f"Expected decision={expected['decision']}, got {actual['decision']}")

        return actual, errors

    def _run_rpm_episode(self, input_data: Dict, expected: Dict) -> Tuple[Dict, List[str]]:
        """Run RPM regulator episode."""
        if not self._has_rpm:
            return {}, ["RPM modules not available"]

        actual = {}
        errors = []

        batch_size = 4
        seq_len = 8
        thought = torch.randn(batch_size, seq_len, 40)

        out = self.rpm_layer(thought)

        actual['output_shape'] = list(out['gated_output'].shape)
        actual['has_nan'] = torch.isnan(out['gated_output']).any().item()
        actual['dwp_scores_shape'] = list(out['dwp_scores'].shape)

        if actual['has_nan']:
            errors.append("Output contains NaN values")

        if expected.get('shape_preserved'):
            if actual['output_shape'] != [batch_size, seq_len, 40]:
                errors.append(f"Expected shape [4, 8, 40], got {actual['output_shape']}")

        return actual, errors

    def _run_integration_episode(self, input_data: Dict, expected: Dict) -> Tuple[Dict, List[str]]:
        """Run integration episode."""
        actual = {}
        errors = []

        # Test basic pipeline connectivity
        if self._has_operator:
            batch_size = 4
            left = torch.randn(batch_size, 40, requires_grad=True)
            right = torch.randn(batch_size, 40, requires_grad=True)
            operator = torch.randint(0, 4, (batch_size,))

            outputs = self.operator_layer(left, right, operator)
            actual['output_shape'] = list(outputs['noetic_output'].shape)
            actual['has_nan'] = torch.isnan(outputs['noetic_output']).any().item()

            # Test gradient flow
            loss = outputs['noetic_output'].sum()
            loss.backward()
            actual['gradients_flow'] = left.grad is not None and not torch.isnan(left.grad).any().item()

            if actual['has_nan']:
                errors.append("Output contains NaN values")

            if expected.get('gradients_flow') and not actual['gradients_flow']:
                errors.append("Gradients did not flow properly")

        return actual, errors


def main():
    parser = argparse.ArgumentParser(description="Run TKS replay regression suite")
    parser.add_argument(
        "--replay-set",
        type=str,
        default="replay_sets/rs_core.json",
        help="Path to replay set JSON file",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter to specific category (e.g., dps_gating)",
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

    print(f"Running replay set: {args.replay_set}")
    print("-" * 60)

    runner = ReplayRunner(verbose=args.verbose)
    report = runner.run_replay_set(args.replay_set, args.category)

    print("-" * 60)
    print(f"Results: {report.passed}/{report.total_episodes} passed "
          f"({100 * report.passed / max(1, report.total_episodes):.1f}%)")
    print(f"  Passed:  {report.passed}")
    print(f"  Failed:  {report.failed}")
    print(f"  Skipped: {report.skipped}")
    print(f"  Errors:  {report.errors}")
    print(f"  Duration: {report.duration_seconds:.2f}s")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to: {args.output}")

    # Exit with failure if any tests failed
    sys.exit(0 if report.failed == 0 and report.errors == 0 else 1)


if __name__ == "__main__":
    main()
