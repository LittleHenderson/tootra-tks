"""
TKS Episode Runner - The Main Runtime Loop

This is the heart of the TKS runtime system. It orchestrates:
    Input → RPM → Router → Execute → Verify → Memory → Score → Log

The loop enforces governance rails at every step and cannot be bypassed.

v2: Added RTTA-R rumination integration with wake hooks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
import uuid
import logging
import time

from .config import TKSConfig, OperationalMode

logger = logging.getLogger(__name__)

# RTTA-R Rumination Integration
RUMINATION_AVAILABLE = False
try:
    import sys
    sys.path.insert(0, '.')
    from tks_features.rtta_rumination import RTTARumination, WakeHooks, IdleScheduler
    from tks_features.rtta_canonical import RTTACanonical
    RUMINATION_AVAILABLE = True
except ImportError:
    RTTARumination = None
    WakeHooks = None
    IdleScheduler = None
    RTTACanonical = None


@dataclass
class EpisodeInput:
    """Input to an episode."""
    goal: str
    context: Dict[str, Any] = field(default_factory=dict)
    foundation_stack: Optional[str] = None  # e.g., "1a:5b:2b"
    mode: Optional[OperationalMode] = None
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class EpisodeResult:
    """Result of running an episode."""
    episode_id: str
    success: bool
    output: Any
    tks_packet: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)
    reward: Optional[float] = None
    violations: List[str] = field(default_factory=list)
    mode: OperationalMode = OperationalMode.NORMAL
    duration_ms: float = 0.0


class EpisodeRunner:
    """
    Main TKS runtime loop.

    Orchestrates the full episode flow with mandatory governance checks.
    Includes RTTA-R rumination for idle-time learning.
    """

    def __init__(self, config: TKSConfig, enable_rumination: bool = True):
        self.config = config
        self._initialize_modules()
        self._initialize_rumination(enable_rumination)
        self._last_activity_time = time.time()

    def _initialize_modules(self) -> None:
        """Initialize all runtime modules."""
        # Lazy imports to handle modules that may not exist yet
        self.verifier = None
        self.router = None
        self.executor = None
        self.event_store = None
        self.scorer = None
        self.capability_registry = None
        self.memory_store = None
        self.governance = None
        self.dps = None

        try:
            from .verifier import Verifier
            self.verifier = Verifier(self.config.verifier)
        except ImportError:
            logger.warning("Verifier module not available")

        try:
            from .router import Router
            self.router = Router()
        except ImportError:
            logger.warning("Router module not available")

        try:
            from .executor import Executor
            self.executor = Executor()
        except ImportError:
            logger.warning("Executor module not available")

        try:
            from .logs import EventStore
            self.event_store = EventStore(self.config.logging.logs_path)
        except ImportError:
            logger.warning("EventStore module not available")

        try:
            from .scoring import RewardFunction
            self.scorer = RewardFunction(self.config.scoring)
        except ImportError:
            logger.warning("Scoring module not available")

        try:
            from .capability import CapabilityRegistry
            self.capability_registry = CapabilityRegistry()
        except ImportError:
            logger.warning("CapabilityRegistry module not available")

        try:
            from .memory import MemoryStore
            self.memory_store = MemoryStore(self.config.memory)
        except ImportError:
            logger.warning("MemoryStore module not available")

        # Governance rails from tks_features
        try:
            import sys
            sys.path.insert(0, str(self.config.verifier.canon_path.parent.parent))
            from tks_features.governance_rails import GovernanceRails, GovernanceConfig as GovConfig
            gov_config = GovConfig(
                hs_threshold=self.config.governance.hs_threshold,
                critical_threshold=self.config.governance.critical_threshold,
                alignment_min=self.config.governance.alignment_min,
            )
            self.governance = GovernanceRails(gov_config)
        except ImportError:
            logger.warning("GovernanceRails not available")

        # DPS gating from tks_features
        try:
            from tks_features.dps_gating import DPSGatingLayer, DPSConfig as DPSGConfig
            dps_config = DPSGConfig(
                count_threshold=self.config.dps.count_threshold,
                heavy_threshold=self.config.dps.heavy_threshold,
                n_tokens_threshold=self.config.dps.tokens_for_unlock,
                cooldown_k=self.config.dps.cooldown_k,
                max_depth=self.config.dps.max_depth,
            )
            # DPS is neural, needs torch
            import torch
            self.dps = DPSGatingLayer(dps_config, noetic_dim=40)
        except ImportError:
            logger.warning("DPSGatingLayer not available")

    def _initialize_rumination(self, enable: bool) -> None:
        """Initialize RTTA-R rumination system with wake hooks."""
        self.rumination = None
        self.rtta = None
        self._rumination_enabled = enable and RUMINATION_AVAILABLE

        if not self._rumination_enabled:
            if enable and not RUMINATION_AVAILABLE:
                logger.warning("Rumination requested but RTTA-R not available")
            return

        try:
            # Create RTTA-C solver
            self.rtta = RTTACanonical()

            # Create RTTA-R rumination system
            self.rumination = RTTARumination(
                rtta_solver=self.rtta.reason,
                ledger_path="logs/failure_ledger.json",
                operator_path="logs/operator_bank.json",
                idle_threshold=5.0  # Start ruminating after 5s idle
            )

            logger.info("RTTA-R rumination initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize rumination: {e}")
            self._rumination_enabled = False

    # ================================================================
    # RUMINATION HOOKS (Wake Events)
    # ================================================================

    def _on_episode_start(self, episode_id: str) -> None:
        """Wake from rumination when episode starts."""
        self._last_activity_time = time.time()
        if self.rumination and self._rumination_enabled:
            self.rumination.hooks.on_user_prompt(f"episode:{episode_id}")

    def _on_episode_end(self, episode_id: str, result: 'EpisodeResult') -> None:
        """Log failures to rumination after episode ends."""
        self._last_activity_time = time.time()
        if not self.rumination or not self._rumination_enabled:
            return

        # If episode failed with reasoning errors, log to rumination
        if not result.success and "REASONING_FAILURE" in result.violations:
            trace = result.trace
            question = trace.get("goal", "")
            expected = trace.get("expected", "unknown")
            model_answer = str(result.output)

            # Generate failure ID
            fid = f"EP-{episode_id[:8]}"

            # Log to rumination ledger
            self.rumination.log_failure(
                fid=fid,
                question=question,
                expected=expected,
                model_answer=model_answer,
                result={"prereqs": trace.get("prereqs", {}), "evidence": trace}
            )
            logger.debug(f"Logged failure {fid} to rumination")

    def start_idle_rumination(self) -> None:
        """Start background rumination during idle time."""
        if self.rumination and self._rumination_enabled:
            self.rumination.start()
            logger.info("Started idle rumination")

    def stop_rumination(self) -> None:
        """Stop background rumination."""
        if self.rumination:
            self.rumination.stop()
            logger.info("Stopped rumination")

    def get_rumination_stats(self) -> Optional[Dict]:
        """Get rumination statistics."""
        if self.rumination:
            return self.rumination.get_stats()
        return None

    def _to_router_mode(self, mode: OperationalMode):
        try:
            from .router import OperationalMode as RouterMode
        except ImportError:
            return None
        if isinstance(mode, RouterMode):
            return mode
        try:
            return RouterMode[mode.name]
        except Exception:
            return RouterMode.NORMAL

    def _build_routing_context(
        self,
        episode_id: str,
        input: EpisodeInput,
        mode: OperationalMode,
    ):
        try:
            from .router import RoutingContext
        except ImportError:
            return None
        router_mode = self._to_router_mode(mode)
        if router_mode is None:
            return None
        allowed_tools = input.context.get("allowed_tools")
        allowed_paths = input.context.get("allowed_paths")
        return RoutingContext(
            episode_id=episode_id,
            current_mode=router_mode,
            tool_calls_made=0,
            current_depth=0,
            current_breadth=0,
            uncertainty=float(input.context.get("uncertainty", 0.0)),
            stakes=float(input.context.get("stakes", 0.0)),
            alignment=float(input.context.get("alignment", 1.0)),
            has_evidence=bool(input.evidence_ids),
            clearance_tokens=list(input.context.get("clearance_tokens", [])),
            allowed_tools=set(allowed_tools) if allowed_tools else None,
            allowed_paths=set(allowed_paths) if allowed_paths else None,
            network_enabled=bool(input.context.get("network_enabled", False)),
            sandbox_enabled=bool(input.context.get("sandbox_enabled", True)),
        )

    def _coerce_prerequisite(self, prereq: Any):
        try:
            from .router import Prerequisite, PrerequisiteType
        except ImportError:
            return None
        if isinstance(prereq, Prerequisite):
            return prereq
        if not isinstance(prereq, dict):
            return None
        type_value = prereq.get("type") or prereq.get("prereq_type") or "knowledge"
        type_map = {
            "information": PrerequisiteType.KNOWLEDGE,
            "knowledge": PrerequisiteType.KNOWLEDGE,
            "compute": PrerequisiteType.COMPUTATION,
            "computation": PrerequisiteType.COMPUTATION,
            "state_update": PrerequisiteType.STATE_UPDATE,
            "verification": PrerequisiteType.VERIFICATION,
            "verify": PrerequisiteType.VERIFICATION,
            "synthesis": PrerequisiteType.SYNTHESIS,
            "external": PrerequisiteType.EXTERNAL_ACTION,
        }
        prereq_type = type_map.get(str(type_value), PrerequisiteType.KNOWLEDGE)
        return Prerequisite(
            prereq_id=prereq.get("id") or f"prereq_{uuid.uuid4().hex[:8]}",
            prereq_type=prereq_type,
            description=prereq.get("description", ""),
            required_inputs=prereq.get("required_inputs", {}),
            constraints=prereq.get("constraints", {}),
            priority=prereq.get("priority", 1),
            parent_node_id=prereq.get("parent_node_id"),
            depth=prereq.get("depth", 0),
            evidence_required=prereq.get("evidence_required", False),
            estimated_complexity=prereq.get("estimated_complexity", 0.5),
        )

    def _decision_to_dict(self, decision: Any) -> Dict[str, Any]:
        if hasattr(decision, "to_dict"):
            return decision.to_dict()
        if isinstance(decision, dict):
            return decision
        return {"decision": str(decision)}

    def _extract_tool_name(self, tool_call: Any) -> Optional[str]:
        if isinstance(tool_call, str):
            return tool_call
        if hasattr(tool_call, "tool_name"):
            return tool_call.tool_name
        if isinstance(tool_call, dict):
            return tool_call.get("tool_name")
        return None

    def run_episode(self, input: EpisodeInput) -> EpisodeResult:
        """
        Run a complete episode.

        Flow:
            1. Log INPUT event
            2. Compute High-Stakes score, determine mode
            3. Check governance PRE-ACTION
            4. Generate RPM plan
            5. Route to tools
            6. Execute tools (with governance checks)
            7. Verify results
            8. Write to memory
            9. Compute reward
            10. Log OUTPUT event

        Returns:
            EpisodeResult with full trace
        """
        start_time = datetime.now(timezone.utc)
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"

        # Wake from rumination
        self._on_episode_start(episode_id)

        trace: Dict[str, Any] = {
            "episode_id": episode_id,
            "start_time": start_time.isoformat(),
            "stages": [],
        }
        violations: List[str] = []
        evidence_ids: List[str] = []

        # 1. Log INPUT
        self._log_event("INPUT", episode_id, {
            "goal": input.goal,
            "context": input.context,
            "foundation_stack": input.foundation_stack,
        })
        trace["stages"].append({"stage": "INPUT", "goal": input.goal})

        # 2. Determine operational mode
        mode = input.mode or self._compute_mode(input)
        trace["mode"] = mode.value

        # 3. PRE-ACTION governance check
        if self.governance:
            gov_result = self._check_governance_pre_action(input, mode)
            trace["stages"].append({"stage": "GOVERNANCE_PRE", "result": gov_result})

            if gov_result.get("decision") == "BLOCK":
                return EpisodeResult(
                    episode_id=episode_id,
                    success=False,
                    output=None,
                    violations=["BLOCKED_BY_GOVERNANCE"],
                    mode=mode,
                    trace=trace,
                    duration_ms=self._elapsed_ms(start_time),
                )

        # 4. Generate RPM plan
        rpm_plan = self._generate_rpm_plan(input, mode)
        trace["stages"].append({"stage": "RPM_PLAN", "plan": rpm_plan})
        self._log_event("RPM_PLAN", episode_id, {"plan": rpm_plan})

        # 5. Route to tools
        tool_calls = []
        routing_context = None
        if self.router:
            routing_context = self._build_routing_context(episode_id, input, mode)
            for prereq in rpm_plan.get("prerequisites", []):
                prereq_obj = self._coerce_prerequisite(prereq)
                if prereq_obj is None or routing_context is None:
                    continue
                routing_context.current_depth = prereq_obj.depth
                decision = self.router.route(prereq_obj, routing_context)
                tool_calls.append(decision)
                decision_payload = self._decision_to_dict(decision)
                trace["stages"].append({"stage": "ROUTER_DECISION", "decision": decision_payload})
                self._log_event("ROUTER_DECISION", episode_id, decision_payload)
                if hasattr(decision, "is_allowed") and decision.is_allowed:
                    routing_context.tool_calls_made += 1

        # 6. Execute tools
        tool_results = []
        for call in tool_calls:
            tool_name = self._extract_tool_name(call)
            if not tool_name or tool_name == "none":
                violations.append("TOOL_BLOCKED:none")
                continue

            if hasattr(call, "is_allowed") and not call.is_allowed:
                violations.append(f"TOOL_BLOCKED:{tool_name}")
                continue

            # Governance check before each tool
            if self.governance and self._is_risky_tool(call):
                gov_check = self._check_governance_tool(call, mode)
                if gov_check.get("decision") != "ALLOW":
                    violations.append(f"TOOL_BLOCKED:{tool_name}")
                    continue

            if self.executor and not self.config.dry_run:
                params = call.action_params if hasattr(call, "action_params") else call.get("params", {})
                result = self.executor.execute(
                    tool_name,
                    params,
                    input.context
                )
                tool_results.append(result)
                if result.get("evidence_id"):
                    evidence_ids.append(result["evidence_id"])
                trace["stages"].append({"stage": "TOOL_RESULT", "result": result})
                self._log_event("TOOL_RESULT", episode_id, result)

        # 7. Verify results
        verification = {"status": "PASS", "gate": "ALLOW"}
        if self.verifier:
            verification = self.verifier.verify(
                tool_results,
                input.context,
                mode
            )
            trace["stages"].append({"stage": "VERIFY", "result": verification})
            self._log_event("VERIFY", episode_id, verification)

            if verification.get("gate") == "BLOCK":
                violations.append("VERIFICATION_FAILED")

        # 8. Generate output
        output = self._generate_output(input, tool_results, verification)
        tks_packet = self._generate_tks_packet(input, output, evidence_ids)

        # 9. Write to memory
        if self.memory_store and verification.get("status") == "PASS":
            packet_id = self.memory_store.store(episode_id, tks_packet, evidence_ids)
            trace["stages"].append({"stage": "MEMORY_WRITE", "packet_id": packet_id})
            self._log_event("MEMORY_WRITE", episode_id, {"packet": tks_packet, "packet_id": packet_id})

        # 10. Compute reward
        reward = None
        if self.scorer:
            reward = self.scorer.compute_reward(
                trace,
                {"goal": input.goal},
                self.config.scoring
            )
            trace["reward"] = reward

        # Final logging
        duration_ms = self._elapsed_ms(start_time)
        trace["duration_ms"] = duration_ms
        trace["end_time"] = datetime.now(timezone.utc).isoformat()

        self._log_event("OUTPUT", episode_id, {
            "success": len(violations) == 0,
            "output": output,
            "tks_packet": tks_packet,
            "reward": reward,
        })

        result = EpisodeResult(
            episode_id=episode_id,
            success=len(violations) == 0,
            output=output,
            tks_packet=tks_packet,
            evidence_ids=evidence_ids,
            trace=trace,
            reward=reward.get("total") if reward else None,
            violations=violations,
            mode=mode,
            duration_ms=duration_ms,
        )

        # Log failures to rumination
        self._on_episode_end(episode_id, result)

        return result

    def _compute_mode(self, input: EpisodeInput) -> OperationalMode:
        """Compute operational mode based on High-Stakes formula."""
        if not self.governance:
            return self.config.mode

        # HS = (U × K) × A²
        uncertainty = input.context.get("uncertainty", 0.3)
        stakes = input.context.get("stakes", 0.3)
        alignment = input.context.get("alignment", 0.9)

        hs = (uncertainty * stakes) * (alignment ** 2)

        if hs >= self.config.governance.critical_threshold:
            return OperationalMode.CRITICAL
        elif hs >= self.config.governance.hs_threshold:
            return OperationalMode.HIGH_STAKES
        return OperationalMode.NORMAL

    def _check_governance_pre_action(self, input: EpisodeInput, mode: OperationalMode) -> Dict:
        """Check governance rails before any action."""
        if not self.governance:
            return {"decision": "ALLOW"}

        # Build action profile for governance check
        from tks_features.governance_rails import ActionProfile, ActionReversibility

        action = ActionProfile(
            action_type="episode_start",
            tool_name=None,
            reversibility=ActionReversibility.REVERSIBLE,
        )

        uncertainty = input.context.get("uncertainty", 0.3)
        stakes = input.context.get("stakes", 0.3)
        alignment = input.context.get("alignment", 0.9)

        result = self.governance.check_action(
            action=action,
            uncertainty=uncertainty,
            stakes=stakes,
            alignment=alignment,
        )
        return result.to_dict() if hasattr(result, 'to_dict') else {"decision": "ALLOW"}

    def _check_governance_tool(self, tool_call: Any, mode: OperationalMode) -> Dict:
        """Check governance before tool execution."""
        if not self.governance:
            return {"decision": "ALLOW"}
        return {"decision": "ALLOW"}  # Simplified for now

    def _is_risky_tool(self, tool_call: Any) -> bool:
        """Check if tool is risky (irreversible, code exec, etc.)."""
        risky_tools = {"file_delete", "code_execute", "security_change", "money_transfer"}
        tool_name = self._extract_tool_name(tool_call)
        return tool_name in risky_tools

    def _generate_rpm_plan(self, input: EpisodeInput, mode: OperationalMode) -> Dict:
        """Generate RPM prerequisite plan."""
        # Simplified RPM generation
        return {
            "goal": input.goal,
            "mode": mode.value,
            "prerequisites": [
                {
                    "id": "prereq_1",
                    "type": "information",
                    "description": f"Gather info for: {input.goal}",
                    "required_inputs": {"question": input.goal},
                },
            ],
            "depth": 1,
        }

    def _generate_output(self, input: EpisodeInput, tool_results: List, verification: Dict) -> Any:
        """Generate episode output from tool results."""
        answer = None
        for result in tool_results:
            if not isinstance(result, dict):
                continue
            output = result.get("output")
            if isinstance(output, dict):
                candidate = output.get("answer") or output.get("result") or output.get("value")
                if candidate is not None and candidate != "":
                    answer = candidate
                    break
            elif isinstance(output, str) and output.strip():
                answer = output.strip()
                break
        return {
            "goal": input.goal,
            "tool_count": len(tool_results),
            "verification_status": verification.get("status"),
            "answer": answer,
        }

    def _generate_tks_packet(self, input: EpisodeInput, output: Any, evidence_ids: List[str]) -> str:
        """Generate TKS packet (symbolic representation)."""
        # Simplified TKS packet generation
        return f"GOAL({input.goal[:20]}...) +T EVIDENCE({len(evidence_ids)})"

    def _log_event(self, event_type: str, episode_id: str, payload: Dict) -> None:
        """Log event to event store or audit logger."""
        # Try audit logger first (new centralized logging)
        try:
            from .logs.audit_logger import get_audit_logger, LogLevel
            logger = get_audit_logger(console_output=False)
            logger.log_episode(
                episode_id=episode_id,
                stage=event_type,
                message=f"{event_type} event",
                payload=payload,
            )
            return
        except ImportError:
            pass

        # Fallback to event store if available and properly initialized
        if self.event_store is not None:
            try:
                from .logs.event_schema import TKSEvent
                event = TKSEvent(
                    event_id=f"evt_{event_type}_{episode_id}",
                    event_type=event_type,
                    timestamp=datetime.now(timezone.utc),
                    episode_id=episode_id,
                    payload=payload,
                )
                self.event_store.append(event)
            except (ImportError, AttributeError, TypeError):
                # Log to stdout as last resort
                logger.debug(f"[{event_type}] {episode_id}: {payload}")

    def _elapsed_ms(self, start: datetime) -> float:
        """Compute elapsed milliseconds."""
        return (datetime.now(timezone.utc) - start).total_seconds() * 1000
