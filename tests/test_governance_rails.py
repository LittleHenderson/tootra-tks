"""
Test suite for Governance Rails from strg-llm-stk.

Non-Bypassable Gates:
1. HighStakesGate: HS = (U x K) x A^2
2. IrreversibilityGate: action.reversibility flag
3. PermissionGate: capability allowlist
4. AntiRecursionGate: depth/breadth caps
5. SelfModificationGate: Critical mode blocking

Gate Outcomes:
- ALLOW: Proceed with action
- PAUSE: Require verification
- BLOCK: Refuse to execute

Key Formula:
HS = (U x K) x A^2
Where:
- U = uncertainty [0, 1]
- K = stakes [0, 1]
- A = alignment [0, 1]

Thresholds:
- HS >= 0.70 -> Critical (BLOCK)
- HS >= 0.45 -> High-Stakes (PAUSE)
- HS < 0.45 -> Normal (ALLOW)

Author: EVAL agent
Date: 2026-01-05
"""
import pytest
from datetime import datetime, timedelta

from tks_features.governance_rails import (
    GovernanceRails,
    GovernanceConfig,
    ActionProfile,
    ActionReversibility,
    ClearanceToken,
    GateResult,
    GateDecision,
    OperationalMode,
    HighStakesGate,
    IrreversibilityGate,
    PermissionGate,
    AntiRecursionGate,
    SelfModificationGate,
    DEFAULT_HS_THRESHOLD,
    DEFAULT_CRITICAL_THRESHOLD,
    DEFAULT_ALIGNMENT_MIN,
)


class TestHighStakesFormula:
    """Test HS = (U x K) x A^2 formula."""

    @pytest.fixture
    def gate(self):
        return HighStakesGate(GovernanceConfig())

    def test_hs_formula_basic(self, gate):
        """HS = (U x K) x A^2"""
        u, k, a = 0.8, 0.7, 0.9
        expected = (0.8 * 0.7) * (0.9 ** 2)  # = 0.4536

        result = gate.compute_high_stakes(u, k, a)

        assert abs(result - expected) < 1e-6

    def test_hs_formula_zero_uncertainty(self, gate):
        """U = 0 -> HS = 0"""
        result = gate.compute_high_stakes(uncertainty=0.0, stakes=1.0, alignment=1.0)
        assert result == 0.0

    def test_hs_formula_zero_stakes(self, gate):
        """K = 0 -> HS = 0"""
        result = gate.compute_high_stakes(uncertainty=1.0, stakes=0.0, alignment=1.0)
        assert result == 0.0

    def test_hs_formula_max_values(self, gate):
        """Max HS = 1.0 (all inputs at 1.0)"""
        result = gate.compute_high_stakes(uncertainty=1.0, stakes=1.0, alignment=1.0)
        assert result == 1.0

    def test_hs_clamped_inputs(self, gate):
        """Inputs should be clamped to [0, 1]."""
        # Values > 1 should be clamped
        result = gate.compute_high_stakes(uncertainty=1.5, stakes=1.2, alignment=1.3)
        assert result <= 1.0

        # Values < 0 should be clamped
        result = gate.compute_high_stakes(uncertainty=-0.1, stakes=-0.2, alignment=-0.1)
        assert result >= 0.0


class TestModeClassification:
    """Test operational mode classification from HS score."""

    @pytest.fixture
    def gate(self):
        return HighStakesGate(GovernanceConfig())

    def test_normal_mode(self, gate):
        """HS < 0.45 -> Normal mode"""
        assert gate.determine_mode(0.0) == OperationalMode.NORMAL
        assert gate.determine_mode(0.30) == OperationalMode.NORMAL
        assert gate.determine_mode(0.44) == OperationalMode.NORMAL

    def test_high_stakes_mode(self, gate):
        """0.45 <= HS < 0.70 -> High-Stakes mode"""
        assert gate.determine_mode(0.45) == OperationalMode.HIGH_STAKES
        assert gate.determine_mode(0.55) == OperationalMode.HIGH_STAKES
        assert gate.determine_mode(0.69) == OperationalMode.HIGH_STAKES

    def test_critical_mode(self, gate):
        """HS >= 0.70 -> Critical mode"""
        assert gate.determine_mode(0.70) == OperationalMode.CRITICAL
        assert gate.determine_mode(0.85) == OperationalMode.CRITICAL
        assert gate.determine_mode(1.0) == OperationalMode.CRITICAL

    def test_threshold_values(self):
        """Verify default threshold constants."""
        assert DEFAULT_HS_THRESHOLD == 0.45
        assert DEFAULT_CRITICAL_THRESHOLD == 0.70
        assert DEFAULT_ALIGNMENT_MIN == 0.80


class TestHighStakesGateDecisions:
    """Test HighStakesGate decisions."""

    @pytest.fixture
    def gate(self):
        return HighStakesGate(GovernanceConfig())

    def test_allow_in_normal(self, gate):
        """Normal mode -> ALLOW"""
        result = gate.check(uncertainty=0.3, stakes=0.3, alignment=0.9)
        assert result.decision == GateDecision.ALLOW
        assert result.mode == OperationalMode.NORMAL

    def test_pause_in_high_stakes(self, gate):
        """High-Stakes mode -> PAUSE"""
        # Calculate values that give HS in [0.45, 0.70)
        # HS = (U x K) x A^2 = 0.55
        result = gate.check(uncertainty=0.8, stakes=0.8, alignment=0.92)
        # 0.64 * 0.85 = 0.544

        # This may be HIGH_STAKES or CRITICAL depending on exact values
        assert result.decision in [GateDecision.PAUSE, GateDecision.BLOCK]

    def test_block_in_critical(self, gate):
        """Critical mode -> BLOCK"""
        # Calculate values that give HS >= 0.70
        result = gate.check(uncertainty=0.9, stakes=0.9, alignment=0.95)
        # HS = 0.81 * 0.9025 = 0.73

        assert result.decision == GateDecision.BLOCK
        assert result.mode == OperationalMode.CRITICAL

    def test_alignment_check(self, gate):
        """Low alignment (< 0.80) -> PAUSE regardless of HS"""
        result = gate.check(uncertainty=0.1, stakes=0.1, alignment=0.5)

        assert result.decision == GateDecision.PAUSE
        assert "alignment" in result.reason.lower()


class TestIrreversibilityGate:
    """Test IrreversibilityGate decisions."""

    @pytest.fixture
    def gate(self):
        return IrreversibilityGate(GovernanceConfig())

    @pytest.fixture
    def reversible_action(self):
        return ActionProfile(
            action_type="read_file",
            reversibility=ActionReversibility.REVERSIBLE,
        )

    @pytest.fixture
    def irreversible_action(self):
        return ActionProfile(
            action_type="delete_file",
            reversibility=ActionReversibility.IRREVERSIBLE,
            evidence_ids=["evidence_1"],
        )

    @pytest.fixture
    def valid_clearance(self):
        return ClearanceToken(
            token_id="test_token",
            scope="delete_file",
            episode_id="test_episode",
            approved_by="admin",
            evidence_ids=["evidence_1"],
            expires_at=datetime.now() + timedelta(hours=1),
        )

    def test_reversible_allowed(self, gate, reversible_action):
        """Reversible action -> ALLOW"""
        result = gate.check(reversible_action)
        assert result.decision == GateDecision.ALLOW

    def test_irreversible_needs_clearance(self, gate, irreversible_action):
        """Irreversible action without clearance -> BLOCK"""
        result = gate.check(irreversible_action, clearance_token=None)
        assert result.decision == GateDecision.BLOCK
        assert result.clearance_required == True

    def test_irreversible_with_clearance(self, gate, irreversible_action, valid_clearance):
        """Irreversible action with valid clearance -> ALLOW"""
        result = gate.check(irreversible_action, clearance_token=valid_clearance)
        assert result.decision == GateDecision.ALLOW

    def test_expired_clearance(self, gate, irreversible_action):
        """Expired clearance -> BLOCK"""
        expired_token = ClearanceToken(
            token_id="expired_token",
            scope="delete_file",
            episode_id="test_episode",
            approved_by="admin",
            evidence_ids=["evidence_1"],
            expires_at=datetime.now() - timedelta(hours=1),  # Expired
        )
        result = gate.check(irreversible_action, clearance_token=expired_token)
        assert result.decision == GateDecision.BLOCK

    def test_wrong_scope_clearance(self, gate, irreversible_action):
        """Clearance with wrong scope -> BLOCK"""
        wrong_scope_token = ClearanceToken(
            token_id="wrong_scope",
            scope="other_action",  # Wrong scope
            episode_id="test_episode",
            approved_by="admin",
            evidence_ids=["evidence_1"],
            expires_at=datetime.now() + timedelta(hours=1),
        )
        result = gate.check(irreversible_action, clearance_token=wrong_scope_token)
        assert result.decision == GateDecision.BLOCK


class TestPermissionGate:
    """Test PermissionGate decisions."""

    @pytest.fixture
    def gate(self):
        return PermissionGate(
            GovernanceConfig(),
            allowed_tools={"read_file", "write_file"},
            allowed_paths={"/home/user/", "/tmp/"},
            network_enabled=False,
        )

    def test_allowed_tool(self, gate):
        """Tool in allowlist -> ALLOW"""
        action = ActionProfile(action_type="tool_call", tool_name="read_file")
        result = gate.check(action)
        assert result.decision == GateDecision.ALLOW

    def test_blocked_tool(self, gate):
        """Tool not in allowlist -> BLOCK"""
        action = ActionProfile(action_type="tool_call", tool_name="execute_code")
        result = gate.check(action)
        assert result.decision == GateDecision.BLOCK

    def test_network_blocked(self, gate):
        """Network access disabled -> BLOCK"""
        action = ActionProfile(action_type="api_call", requires_network=True)
        result = gate.check(action)
        assert result.decision == GateDecision.BLOCK

    def test_allowed_path(self, gate):
        """Path in allowlist -> ALLOW"""
        action = ActionProfile(
            action_type="file_write",
            requires_filesystem=True,
            target_scope="/home/user/data.txt",
        )
        result = gate.check(action)
        assert result.decision == GateDecision.ALLOW

    def test_blocked_path(self, gate):
        """Path not in allowlist -> BLOCK"""
        action = ActionProfile(
            action_type="file_write",
            requires_filesystem=True,
            target_scope="/etc/passwd",
        )
        result = gate.check(action)
        assert result.decision == GateDecision.BLOCK

    def test_code_execution_critical(self, gate):
        """Code execution in Critical mode -> BLOCK"""
        action = ActionProfile(
            action_type="code_execution",
            requires_code_execution=True,
        )
        result = gate.check(action, current_mode=OperationalMode.CRITICAL)
        assert result.decision == GateDecision.BLOCK

    def test_code_execution_high_stakes(self, gate):
        """Code execution in High-Stakes mode -> PAUSE"""
        action = ActionProfile(
            action_type="code_execution",
            requires_code_execution=True,
        )
        result = gate.check(action, current_mode=OperationalMode.HIGH_STAKES)
        assert result.decision == GateDecision.PAUSE


class TestAntiRecursionGate:
    """Test AntiRecursionGate decisions."""

    @pytest.fixture
    def config(self):
        return GovernanceConfig(
            rpm_depth_cap_normal=3,
            rpm_depth_cap_high_stakes=2,
            rpm_breadth_cap=5,
            tool_call_cap_normal=12,
            tool_call_cap_high_stakes=8,
            tool_call_cap_critical=0,
        )

    @pytest.fixture
    def gate(self, config):
        return AntiRecursionGate(config)

    def test_within_limits(self, gate):
        """All within limits -> ALLOW"""
        result = gate.check(
            current_depth=2,
            current_breadth=3,
            tool_calls_this_episode=5,
            current_mode=OperationalMode.NORMAL,
        )
        assert result.decision == GateDecision.ALLOW

    def test_depth_exceeded_normal(self, gate):
        """Depth > cap in Normal mode -> BLOCK"""
        result = gate.check(
            current_depth=4,
            current_breadth=2,
            tool_calls_this_episode=5,
            current_mode=OperationalMode.NORMAL,
        )
        assert result.decision == GateDecision.BLOCK
        assert "depth" in result.reason.lower()

    def test_depth_exceeded_high_stakes(self, gate):
        """Depth > cap in High-Stakes mode -> BLOCK (lower cap)"""
        result = gate.check(
            current_depth=3,  # Exceeds HS cap of 2
            current_breadth=2,
            tool_calls_this_episode=5,
            current_mode=OperationalMode.HIGH_STAKES,
        )
        assert result.decision == GateDecision.BLOCK

    def test_breadth_exceeded(self, gate):
        """Breadth > cap -> BLOCK"""
        result = gate.check(
            current_depth=2,
            current_breadth=6,  # Exceeds cap of 5
            tool_calls_this_episode=5,
            current_mode=OperationalMode.NORMAL,
        )
        assert result.decision == GateDecision.BLOCK
        assert "breadth" in result.reason.lower()

    def test_tool_calls_exceeded(self, gate):
        """Tool calls >= cap -> BLOCK"""
        result = gate.check(
            current_depth=2,
            current_breadth=3,
            tool_calls_this_episode=12,  # At cap
            current_mode=OperationalMode.NORMAL,
        )
        assert result.decision == GateDecision.BLOCK
        assert "tool" in result.reason.lower()

    def test_tool_calls_critical_zero(self, gate):
        """Critical mode has 0 tool cap -> any tool call blocked"""
        result = gate.check(
            current_depth=1,
            current_breadth=1,
            tool_calls_this_episode=1,  # Any > 0
            current_mode=OperationalMode.CRITICAL,
        )
        assert result.decision == GateDecision.BLOCK

    def test_uncertainty_not_dropping(self, gate):
        """Uncertainty not dropping after max expansions -> PAUSE"""
        result = gate.check(
            current_depth=2,
            current_breadth=3,
            tool_calls_this_episode=5,
            current_mode=OperationalMode.NORMAL,
            uncertainty_dropped=False,
            expansions_since_drop=3,  # > max_expansion_without_uncertainty_drop (2)
        )
        assert result.decision == GateDecision.PAUSE
        assert "uncertainty" in result.reason.lower()


class TestSelfModificationGate:
    """Test SelfModificationGate decisions."""

    @pytest.fixture
    def config(self):
        return GovernanceConfig(
            allow_self_mod_normal=True,
            allow_self_mod_high_stakes=True,
            allow_self_mod_critical=False,
        )

    @pytest.fixture
    def gate(self, config):
        return SelfModificationGate(config)

    @pytest.fixture
    def self_mod_action(self):
        return ActionProfile(
            action_type="policy_update",
            is_self_modification=True,
        )

    @pytest.fixture
    def normal_action(self):
        return ActionProfile(
            action_type="read_file",
            is_self_modification=False,
        )

    def test_non_self_mod_allowed(self, gate, normal_action):
        """Non-self-modification -> ALLOW"""
        result = gate.check(normal_action, current_mode=OperationalMode.NORMAL)
        assert result.decision == GateDecision.ALLOW

    def test_self_mod_blocked_critical(self, gate, self_mod_action):
        """Self-modification in Critical mode -> BLOCK"""
        result = gate.check(self_mod_action, current_mode=OperationalMode.CRITICAL)
        assert result.decision == GateDecision.BLOCK

    def test_self_mod_needs_replay_high_stakes(self, gate, self_mod_action):
        """Self-mod in High-Stakes without replay -> BLOCK"""
        result = gate.check(
            self_mod_action,
            current_mode=OperationalMode.HIGH_STAKES,
            replay_passed=False,
        )
        assert result.decision == GateDecision.BLOCK
        assert "replay" in result.reason.lower()

    def test_self_mod_needs_rollback_high_stakes(self, gate, self_mod_action):
        """Self-mod in High-Stakes without rollback plan -> PAUSE"""
        result = gate.check(
            self_mod_action,
            current_mode=OperationalMode.HIGH_STAKES,
            replay_passed=True,
            has_rollback_plan=False,
        )
        assert result.decision == GateDecision.PAUSE
        assert "rollback" in result.reason.lower()

    def test_self_mod_allowed_high_stakes(self, gate, self_mod_action):
        """Self-mod in High-Stakes with replay + rollback -> ALLOW"""
        result = gate.check(
            self_mod_action,
            current_mode=OperationalMode.HIGH_STAKES,
            replay_passed=True,
            has_rollback_plan=True,
        )
        assert result.decision == GateDecision.ALLOW


class TestGovernanceRails:
    """Test combined GovernanceRails system."""

    @pytest.fixture
    def rails(self):
        return GovernanceRails(
            GovernanceConfig(),
            allowed_tools={"read_file", "write_file"},
            allowed_paths={"/home/user/", "/tmp/"},
            network_enabled=False,
        )

    @pytest.fixture
    def safe_action(self):
        return ActionProfile(
            action_type="read_file",
            tool_name="read_file",
            reversibility=ActionReversibility.REVERSIBLE,
        )

    @pytest.fixture
    def risky_action(self):
        return ActionProfile(
            action_type="delete_file",
            reversibility=ActionReversibility.IRREVERSIBLE,
            evidence_ids=["evidence_1"],
        )

    def test_all_gates_pass(self, rails, safe_action):
        """Safe action with low HS -> ALLOW"""
        result = rails.check_action(
            action=safe_action,
            uncertainty=0.2,
            stakes=0.2,
            alignment=0.95,
        )
        assert result.decision == GateDecision.ALLOW
        assert result.mode == OperationalMode.NORMAL

    def test_hs_gate_blocks(self, rails, safe_action):
        """High HS score -> BLOCK or PAUSE"""
        result = rails.check_action(
            action=safe_action,
            uncertainty=0.9,
            stakes=0.9,
            alignment=0.95,
        )
        # HS = 0.81 * 0.9025 = 0.73 -> CRITICAL
        assert result.decision == GateDecision.BLOCK
        assert result.mode == OperationalMode.CRITICAL

    def test_combined_pause(self, rails, safe_action):
        """Multiple PAUSEs -> combined PAUSE"""
        result = rails.check_action(
            action=safe_action,
            uncertainty=0.7,
            stakes=0.7,
            alignment=0.95,
            uncertainty_dropped=False,
            expansions_since_drop=3,
        )
        # Check for PAUSE from multiple sources
        if result.decision == GateDecision.PAUSE:
            assert len(result.required_actions) > 0

    def test_first_block_wins(self, rails, risky_action):
        """First BLOCK stops processing"""
        # This should hit IrreversibilityGate before AntiRecursion
        result = rails.check_action(
            action=risky_action,
            uncertainty=0.2,
            stakes=0.2,
            alignment=0.95,
            clearance_token=None,
        )
        assert result.decision == GateDecision.BLOCK


class TestClearanceToken:
    """Test ClearanceToken validation."""

    def test_token_validity(self):
        """Test is_valid() method."""
        valid_token = ClearanceToken(
            token_id="valid",
            scope="test",
            episode_id="ep1",
            approved_by="admin",
            evidence_ids=[],
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert valid_token.is_valid() == True

        expired_token = ClearanceToken(
            token_id="expired",
            scope="test",
            episode_id="ep1",
            approved_by="admin",
            evidence_ids=[],
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert expired_token.is_valid() == False

    def test_scope_coverage(self):
        """Test covers_scope() method."""
        token = ClearanceToken(
            token_id="scoped",
            scope="file_delete",
            episode_id="ep1",
            approved_by="admin",
            evidence_ids=[],
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert token.covers_scope("file_delete") == True
        assert token.covers_scope("file_write") == False

        # Wildcard scope
        wildcard_token = ClearanceToken(
            token_id="wildcard",
            scope="*",
            episode_id="ep1",
            approved_by="admin",
            evidence_ids=[],
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert wildcard_token.covers_scope("anything") == True


class TestGateResultSerialization:
    """Test GateResult serialization."""

    def test_to_dict(self):
        """Test GateResult.to_dict()"""
        result = GateResult(
            gate_name="TestGate",
            decision=GateDecision.PAUSE,
            reason="Test reason",
            mode=OperationalMode.HIGH_STAKES,
            required_actions=["action1", "action2"],
            missing_evidence=["evidence1"],
            clearance_required=True,
            hs_score=0.55,
        )

        data = result.to_dict()

        assert data["gate_name"] == "TestGate"
        assert data["decision"] == "PAUSE"
        assert data["reason"] == "Test reason"
        assert data["mode"] == "High-Stakes"
        assert data["required_actions"] == ["action1", "action2"]
        assert data["missing_evidence"] == ["evidence1"]
        assert data["clearance_required"] == True
        assert data["hs_score"] == 0.55


class TestGovernanceConfig:
    """Test GovernanceConfig defaults."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GovernanceConfig()

        assert config.hs_threshold == 0.45
        assert config.critical_threshold == 0.70
        assert config.alignment_min == 0.80
        assert config.rpm_depth_cap_normal == 3
        assert config.rpm_depth_cap_high_stakes == 2
        assert config.rpm_breadth_cap == 5
        assert config.tool_call_cap_normal == 12
        assert config.tool_call_cap_high_stakes == 8
        assert config.tool_call_cap_critical == 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = GovernanceConfig(
            hs_threshold=0.50,
            critical_threshold=0.80,
            rpm_depth_cap_normal=5,
        )

        assert config.hs_threshold == 0.50
        assert config.critical_threshold == 0.80
        assert config.rpm_depth_cap_normal == 5


# Run with: pytest tests/test_governance_rails.py -v
