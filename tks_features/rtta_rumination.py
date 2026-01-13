"""
RTTA-R: Rumination Mode for RTTA-C

TKS-native deterministic rumination with JSON Schema compliant structures.
Schemas: urn:tks:failure-ledger, urn:tks:rumination-state

No gradient updates - purely deterministic control flow.
"""

import json
import os
import time
import re
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from enum import Enum


# ============================================================
# UTILITIES
# ============================================================

def now_iso() -> str:
    """Return current time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# ENUMS (Schema Compliant)
# ============================================================

class FailureStatus(Enum):
    UNSOLVED = "unsolved"
    SOLVED = "solved"
    RETIRED = "retired"


class HypothesisStatus(Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"


class WakeReason(Enum):
    USER_PROMPT = "user_prompt"
    TASK_ASSIGNED = "task_assigned"
    MANUAL_WAKE = "manual_wake"
    SHUTDOWN = "shutdown"
    UNKNOWN = "unknown"


class RuminationMode(Enum):
    IDLE = "idle"
    RUMINATING = "ruminating"
    AWAKE = "awake"


# Valid slots per schema
VALID_SLOTS = ["1D", "2W", "3P", "4D", "5W", "6P", "7D", "8W", "9P", "10D", "11W", "12P"]


# ============================================================
# SCHEMA-COMPLIANT DATA STRUCTURES
# ============================================================

@dataclass
class Gist:
    """Minimal structure for later solving (schema: gist)."""
    type: str
    signature: str
    slot_block: str
    missing_operator: Optional[str] = None
    pattern_hint: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {"type": self.type, "signature": self.signature, "slot_block": self.slot_block}
        if self.missing_operator:
            d["missing_operator"] = self.missing_operator
        if self.pattern_hint:
            d["pattern_hint"] = self.pattern_hint
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'Gist':
        return cls(**d)


@dataclass
class Hypothesis:
    """Hypothesis for solving a failure (schema: hypotheses[])."""
    label: str
    status: str = "proposed"  # proposed | validated | rejected
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {"label": self.label, "status": self.status}
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'Hypothesis':
        return cls(**d)


@dataclass
class FailureRecord:
    """
    Complete failure record (schema: FailureRecord).
    All required fields per urn:tks:failure-ledger.
    """
    id: str
    question_text: str
    expected: str
    model_answer: str
    status: str  # unsolved | solved | retired
    first_unmet_slot: str  # Must be in VALID_SLOTS
    impact_score: float
    desire_level: float
    error_severity: float
    novelty_gap: float
    recency: float
    gist: Gist
    attempt_count: int
    rumination_cycles: int
    last_seen_at: str  # ISO datetime

    # Optional fields
    operator_missing: Optional[str] = None
    last_ruminated_at: Optional[str] = None
    resolved_at: Optional[str] = None
    hypotheses: List[Hypothesis] = field(default_factory=list)

    # Internal tracking (not in schema output)
    _last_improvement: int = 0
    _stagnation_penalty: float = 0.0
    _depth_score: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to schema-compliant dict."""
        d = {
            "id": self.id,
            "question_text": self.question_text,
            "expected": self.expected,
            "model_answer": self.model_answer,
            "status": self.status,
            "first_unmet_slot": self.first_unmet_slot,
            "impact_score": round(self.impact_score, 4),
            "desire_level": round(self.desire_level, 4),
            "error_severity": round(self.error_severity, 4),
            "novelty_gap": round(self.novelty_gap, 4),
            "recency": round(self.recency, 4),
            "gist": self.gist.to_dict(),
            "attempt_count": self.attempt_count,
            "rumination_cycles": self.rumination_cycles,
            "last_seen_at": self.last_seen_at,
        }
        if self.operator_missing:
            d["operator_missing"] = self.operator_missing
        if self.last_ruminated_at:
            d["last_ruminated_at"] = self.last_ruminated_at
        if self.resolved_at:
            d["resolved_at"] = self.resolved_at
        if self.hypotheses:
            d["hypotheses"] = [h.to_dict() for h in self.hypotheses]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'FailureRecord':
        gist = Gist.from_dict(d.pop('gist'))
        hypotheses = [Hypothesis.from_dict(h) for h in d.pop('hypotheses', [])]
        # Remove internal fields if present
        d.pop('_last_improvement', None)
        d.pop('_stagnation_penalty', None)
        d.pop('_depth_score', None)
        return cls(gist=gist, hypotheses=hypotheses, **d)


@dataclass
class WakeEvent:
    """Wake event structure (schema: wake_event)."""
    pending: bool = False
    reason: str = "unknown"
    source: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {"pending": self.pending, "reason": self.reason}
        if self.source:
            d["source"] = self.source
        if self.timestamp:
            d["timestamp"] = self.timestamp
        return d

    def set(self, reason: str, source: str = "unknown"):
        self.pending = True
        self.reason = reason
        self.source = source
        self.timestamp = now_iso()

    def clear(self):
        self.pending = False
        self.reason = "unknown"
        self.source = None
        self.timestamp = None


@dataclass
class RuminationState:
    """
    Rumination state (schema: urn:tks:rumination-state).
    """
    enabled: bool = False
    mode: str = "idle"  # idle | ruminating | awake
    current_failure_id: Optional[str] = None
    wake_event: WakeEvent = field(default_factory=WakeEvent)
    tick_ms: int = 100
    idle_backoff_ms: int = 500
    desire_alpha: float = 0.15
    impact_weights: Dict[str, float] = field(default_factory=lambda: {
        "error_severity": 0.4,
        "novelty_gap": 0.3,
        "slot_block": 0.2,
        "recency": 0.1
    })
    slot_weights: Dict[str, float] = field(default_factory=lambda: {
        "4D": 1.0,
        "5W": 0.8,
        "6P": 0.6
    })
    last_cycle_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "current_failure_id": self.current_failure_id or "",
            "wake_event": self.wake_event.to_dict(),
            "tick_ms": self.tick_ms,
            "idle_backoff_ms": self.idle_backoff_ms,
            "desire_alpha": self.desire_alpha,
            "impact_weights": self.impact_weights,
            "slot_weights": self.slot_weights,
            "last_cycle_at": self.last_cycle_at
        }


# ============================================================
# FAILURE LEDGER (Schema: urn:tks:failure-ledger)
# ============================================================

class FailureLedger:
    """
    Schema-compliant failure ledger with indexed access.
    Implements urn:tks:failure-ledger JSON Schema.
    """

    VERSION = "1.0"
    DEFAULT_THRESHOLD = 0.6

    def __init__(self, storage_path: str = "logs/failure_ledger.json"):
        self.storage_path = storage_path
        self.version = self.VERSION
        self.threshold = self.DEFAULT_THRESHOLD
        self.created_at = now_iso()
        self.updated_at = now_iso()
        self.records: Dict[str, FailureRecord] = {}
        self.index = {"unsolved": [], "solved": [], "retired": []}
        self._load()

    def _load(self):
        """Load ledger from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                self.version = data.get('version', self.VERSION)
                self.threshold = data.get('threshold', self.DEFAULT_THRESHOLD)
                self.created_at = data.get('created_at', now_iso())
                self.updated_at = data.get('updated_at', now_iso())
                for rid, rec in data.get('records', {}).items():
                    self.records[rid] = FailureRecord.from_dict(rec)
                self.index = data.get('index', {"unsolved": [], "solved": [], "retired": []})
            except Exception as e:
                print(f"Warning: Could not load ledger: {e}")

    def _save(self):
        """Save ledger to disk (schema-compliant)."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.updated_at = now_iso()
        data = {
            "version": self.version,
            "threshold": self.threshold,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "records": {rid: rec.to_dict() for rid, rec in self.records.items()},
            "index": self.index
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _rebuild_index(self):
        """Rebuild index from records."""
        self.index = {"unsolved": [], "solved": [], "retired": []}
        for rid, rec in self.records.items():
            self.index[rec.status].append(rid)

    def compute_impact_score(self,
                              error_severity: float,
                              novelty_gap: float,
                              slot_block: str,
                              recency: float = 1.0,
                              is_unknown_type: bool = False,
                              stagnation_penalty: float = 0.0,
                              weights: Optional[Dict] = None) -> float:
        """
        Compute impact score using Noetic 4 (Vibration).
        Uses configurable weights from RuminationState.
        """
        w = weights or {"error_severity": 0.4, "novelty_gap": 0.3, "slot_block": 0.2, "recency": 0.1}

        # Force 4D for unknown types
        if is_unknown_type:
            slot_block = '4D'

        slot_weights = {"4D": 1.0, "5W": 0.8, "6P": 0.6}
        slot_weight = slot_weights.get(slot_block, 0.4)

        impact = (w["error_severity"] * error_severity +
                  w["novelty_gap"] * novelty_gap +
                  w["slot_block"] * slot_weight +
                  w["recency"] * recency -
                  stagnation_penalty)

        return min(1.0, max(0.0, impact))

    def _extract_gist(self, question: str, problem_type: str,
                      first_unmet: str) -> Gist:
        """Extract minimal gist from failure."""
        operator_map = {
            ('sequence', '5W'): 'OP-SEQ-UNKNOWN',
            ('sequence', '6P'): 'OP-SEQ-UNKNOWN',
            ('syllogism', '4D'): 'OP-SYLL-VARIANT',
            ('linear_system', '5W'): 'OP-LINSYS-VARIANT',
            ('conditional', '5W'): 'OP-COND-VARIANT',
            ('probability', '5W'): 'OP-PROB-VARIANT',
            ('unknown', '4D'): 'OP-TYPE-UNKNOWN',
        }

        missing_op = operator_map.get((problem_type, first_unmet), f'OP-{problem_type.upper()}-UNKNOWN')
        numbers = re.findall(r'\d+', question)
        signature = f"{problem_type}: {','.join(numbers[:5])} -> ?"

        pattern_hint = None
        if 'fibonacci' in question.lower() or (numbers and numbers[:3] == ['1', '1', '2']):
            pattern_hint = 'fibonacci_like'
        elif 'prime' in question.lower():
            pattern_hint = 'prime_sequence'

        return Gist(
            type=problem_type,
            signature=signature,
            slot_block=first_unmet,
            missing_operator=missing_op,
            pattern_hint=pattern_hint
        )

    def add_failure(self, fid: str, question: str, expected: str,
                    model_answer: str, result: Dict) -> FailureRecord:
        """Add a new failure to the ledger."""
        problem_type = result.get('evidence', {}).get('derived', {}).get('problem_type', 'unknown')
        prereqs = result.get('prereqs', {})

        # Find first unmet slot
        first_unmet = '5W'
        for slot in ['4D', '5W', '6P']:
            if slot in prereqs and not prereqs[slot].get('passed', True):
                first_unmet = slot
                break

        # Determine error severity
        if 'cannot determine' in model_answer.lower() or 'unable' in model_answer.lower():
            error_severity = 0.7
        elif model_answer.lower() != expected.lower():
            error_severity = 1.0
        else:
            error_severity = 0.5

        is_unknown = problem_type == 'unknown'
        novelty_gap = 1.0 if is_unknown else 0.5
        recency = 1.0

        impact = self.compute_impact_score(
            error_severity=error_severity,
            novelty_gap=novelty_gap,
            slot_block=first_unmet,
            recency=recency,
            is_unknown_type=is_unknown
        )

        gist = self._extract_gist(question, problem_type, first_unmet)

        if fid in self.records:
            record = self.records[fid]
            record.attempt_count += 1
            record.last_seen_at = now_iso()
            record.impact_score = max(record.impact_score, impact)
            record.recency = recency
        else:
            record = FailureRecord(
                id=fid,
                question_text=question,
                expected=expected,
                model_answer=model_answer,
                status="unsolved",
                first_unmet_slot=first_unmet,
                impact_score=impact,
                desire_level=0.5,
                error_severity=error_severity,
                novelty_gap=novelty_gap,
                recency=recency,
                gist=gist,
                attempt_count=1,
                rumination_cycles=0,
                last_seen_at=now_iso(),
                operator_missing=gist.missing_operator
            )
            self.records[fid] = record
            self.index["unsolved"].append(fid)

        self._save()
        return record

    def mark_solved(self, fid: str, hypothesis_label: str):
        """Mark a failure as solved."""
        if fid in self.records:
            record = self.records[fid]
            record.status = "solved"
            record.resolved_at = now_iso()
            record.hypotheses.append(Hypothesis(label=hypothesis_label, status="validated"))

            # Update index
            if fid in self.index["unsolved"]:
                self.index["unsolved"].remove(fid)
            if fid not in self.index["solved"]:
                self.index["solved"].append(fid)

            self._save()

    def mark_retired(self, fid: str, reason: str = "manual"):
        """Retire a failure (stop ruminating on it)."""
        if fid in self.records:
            record = self.records[fid]
            record.status = "retired"
            record.gist.notes = f"Retired: {reason}"

            if fid in self.index["unsolved"]:
                self.index["unsolved"].remove(fid)
            if fid not in self.index["retired"]:
                self.index["retired"].append(fid)

            self._save()

    def get_unsolved(self) -> List[FailureRecord]:
        """Get all unsolved failures sorted by impact."""
        unsolved = [self.records[fid] for fid in self.index["unsolved"] if fid in self.records]
        return sorted(unsolved, key=lambda f: f.impact_score, reverse=True)

    def get_by_type(self, problem_type: str) -> List[FailureRecord]:
        """Get failures by problem type."""
        return [self.records[fid] for fid in self.index["unsolved"]
                if fid in self.records and self.records[fid].gist.type == problem_type]


# ============================================================
# OPERATOR BANK
# ============================================================

@dataclass
class OperatorHypothesis:
    """A proposed operator to solve a failure pattern."""
    name: str
    trigger_pattern: str
    description: str
    test_cases: List[Dict] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class OperatorBank:
    """Registry of operators with proposal pathway."""

    def __init__(self, storage_path: str = "logs/operator_bank.json"):
        self.storage_path = storage_path
        self.operators: Dict[str, OperatorHypothesis] = {}
        self.proposed: Dict[str, OperatorHypothesis] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                for name, op_data in data.get('operators', {}).items():
                    self.operators[name] = OperatorHypothesis(**op_data)
                for name, op_data in data.get('proposed', {}).items():
                    self.proposed[name] = OperatorHypothesis(**op_data)
            except Exception as e:
                print(f"Warning: Could not load operator bank: {e}")

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            'timestamp': now_iso(),
            'operators': {name: op.to_dict() for name, op in self.operators.items()},
            'proposed': {name: op.to_dict() for name, op in self.proposed.items()}
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def propose_operator(self, failure: FailureRecord) -> Optional[OperatorHypothesis]:
        """Propose a new operator based on failure pattern."""
        gist = failure.gist
        op_name = gist.missing_operator or f"OP-{gist.type.upper()}-UNKNOWN"

        if op_name in self.proposed or op_name in self.operators:
            return self.proposed.get(op_name) or self.operators.get(op_name)

        trigger_patterns = {
            'sequence': r'sequence|pattern|next|follows',
            'syllogism': r'all\s+\w+\s+are|every\s+\w+|some\s+\w+',
            'linear_system': r'cost|price|\$\d+|bought|total',
            'conditional': r'if\s+.+then|when\s+.+|implies',
            'unknown': r'.*',
        }

        trigger = trigger_patterns.get(gist.type, gist.type)

        hypothesis = OperatorHypothesis(
            name=op_name,
            trigger_pattern=trigger,
            description=f"Proposed operator for {gist.type} (slot: {gist.slot_block})",
            test_cases=[{'input': failure.question_text, 'expected': failure.expected}]
        )

        self.proposed[op_name] = hypothesis
        self._save()
        return hypothesis


# ============================================================
# RUMINATION ENGINE
# ============================================================

class RuminationEngine:
    """
    RTTA-R: Rumination Mode with schema-compliant state.
    Uses RuminationState (urn:tks:rumination-state).
    """

    def __init__(self, rtta_solver: Callable,
                 ledger: Optional[FailureLedger] = None,
                 operator_bank: Optional[OperatorBank] = None):
        self.rtta_solver = rtta_solver
        self.ledger = ledger or FailureLedger()
        self.operator_bank = operator_bank or OperatorBank()

        # Schema-compliant state
        self.state = RuminationState()

        # Stagnation tracking
        self.stagnation_threshold = 5
        self.stagnation_penalty_rate = 0.05

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Stats
        self.total_cycles = 0
        self.solutions_found = 0

    @property
    def is_running(self) -> bool:
        return self.state.mode == "ruminating"

    # --------------------------------------------------------
    # DESIRE UPDATES (Diminishing Returns)
    # --------------------------------------------------------

    def raise_desire(self, failure: FailureRecord):
        """
        Raise 4D with diminishing returns:
        delta = alpha * impact * (1 - desire_level)
        """
        headroom = 1.0 - failure.desire_level
        delta = self.state.desire_alpha * failure.impact_score * headroom
        failure.desire_level = min(1.0, failure.desire_level + delta)

    def decay_desire(self, failure: FailureRecord):
        """Decay desire after failed hypothesis."""
        failure.desire_level = max(0.0, failure.desire_level - 0.08)

    def decay_on_resolve(self, failure: FailureRecord):
        """Quick decay when solved."""
        failure.desire_level *= 0.1
        failure.impact_score *= 0.1

    def apply_stagnation_penalty(self, failure: FailureRecord):
        """Apply penalty if stuck for too many cycles."""
        cycles_since = failure.rumination_cycles - failure._last_improvement
        if cycles_since > self.stagnation_threshold:
            penalty = self.stagnation_penalty_rate * (cycles_since - self.stagnation_threshold)
            failure._stagnation_penalty = min(0.3, penalty)

            is_unknown = failure.gist.type == 'unknown'
            failure.impact_score = self.ledger.compute_impact_score(
                error_severity=failure.error_severity,
                novelty_gap=failure.novelty_gap,
                slot_block=failure.first_unmet_slot,
                recency=0.5,
                is_unknown_type=is_unknown,
                stagnation_penalty=failure._stagnation_penalty,
                weights=self.state.impact_weights
            )

    # --------------------------------------------------------
    # RUMINATION CYCLE
    # --------------------------------------------------------

    def ruminate_once(self, failure: FailureRecord) -> bool:
        """Single rumination cycle. Returns True if solved."""
        self.total_cycles += 1
        failure.rumination_cycles += 1
        failure.last_ruminated_at = now_iso()
        self.state.last_cycle_at = now_iso()
        self.state.current_failure_id = failure.id

        # Raise desire (diminishing returns)
        self.raise_desire(failure)

        # Apply stagnation penalty
        self.apply_stagnation_penalty(failure)

        # Propose operator
        hypothesis = self.operator_bank.propose_operator(failure)
        if not hypothesis:
            self.decay_desire(failure)
            return False

        # Try to solve
        passed = False
        try:
            result = self.rtta_solver(failure.question_text, verbose=False)
            answer = str(result.get('answer', ''))
            if failure.expected.lower() in answer.lower():
                passed = True
                self.ledger.mark_solved(failure.id, hypothesis.name)
                self.decay_on_resolve(failure)
                self.solutions_found += 1
                failure._last_improvement = failure.rumination_cycles
        except:
            pass

        # Log hypothesis
        status = "validated" if passed else "rejected"
        failure.hypotheses.append(Hypothesis(label=hypothesis.name, status=status))

        if not passed:
            self.decay_desire(failure)

        return passed

    # --------------------------------------------------------
    # RUMINATION LOOPS
    # --------------------------------------------------------

    def rumination_loop(self, budget: int = 10):
        """Budgeted rumination loop."""
        cycles = 0
        while cycles < budget:
            if self.state.wake_event.pending:
                break

            unsolved = self.ledger.get_unsolved()
            if not unsolved:
                break

            self.ruminate_once(unsolved[0])
            cycles += 1
            self.ledger._save()
            time.sleep(self.state.tick_ms / 1000.0)

    def rumination_loop_indefinite(self):
        """
        Indefinite rumination - runs until wake signal.
        Implements pseudocode from spec.
        """
        self.state.enabled = True
        self.state.mode = "ruminating"
        self.state.wake_event.clear()

        while self.state.enabled:
            if self.state.wake_event.pending:
                break

            unsolved = self.ledger.get_unsolved()
            if not unsolved:
                time.sleep(self.state.idle_backoff_ms / 1000.0)
                continue

            failure = unsolved[0]
            self.ruminate_once(failure)
            self.ledger._save()
            time.sleep(self.state.tick_ms / 1000.0)

        self.state.mode = "awake"

    def _try_solve(self, failure: FailureRecord, hypothesis) -> bool:
        try:
            result = self.rtta_solver(failure.question_text, verbose=False)
            answer = str(result.get('answer', ''))
            return failure.expected.lower() in answer.lower()
        except:
            return False

    # --------------------------------------------------------
    # CONTROL API
    # --------------------------------------------------------

    def start_rumination(self):
        """Start indefinite rumination in background."""
        with self._lock:
            if self.state.mode == "ruminating":
                return
            self.state.wake_event.clear()
            self._thread = threading.Thread(target=self.rumination_loop_indefinite)
            self._thread.daemon = True
            self._thread.start()

    def wake(self, reason: str = "manual_wake", source: str = "unknown"):
        """Wake from rumination."""
        self.state.wake_event.set(reason, source)
        self.state.mode = "awake"
        self.state.enabled = False

    def stop_rumination(self):
        """Hard stop."""
        self.wake("shutdown", "system")

    def start_idle_rumination(self, budget: int = 10):
        """Start budgeted rumination in background."""
        with self._lock:
            if self.state.mode == "ruminating":
                return
            self.state.mode = "ruminating"
            self.state.wake_event.clear()
            self._thread = threading.Thread(target=self._run_budgeted, args=(budget,))
            self._thread.daemon = True
            self._thread.start()

    def _run_budgeted(self, budget: int):
        try:
            self.rumination_loop(budget)
        finally:
            self.state.mode = "idle"


# ============================================================
# WAKE HOOKS (Event Wiring)
# ============================================================

class WakeHooks:
    """
    Wake hooks for event wiring.
    Connects external events to rumination wake signals.
    """

    def __init__(self, engine: RuminationEngine):
        self.engine = engine

    def on_idle_enter(self):
        """Called when system enters idle state."""
        self.engine.start_rumination()

    def on_user_prompt(self, prompt_text: str):
        """Called when user submits a prompt."""
        self.engine.wake("user_prompt", source="cli")

    def on_task_assigned(self, task_id: str):
        """Called when a task is assigned."""
        self.engine.wake("task_assigned", source=task_id)

    def on_shutdown(self):
        """Called on system shutdown."""
        self.engine.wake("shutdown", source="system")

    def on_manual_wake(self, source: str = "user"):
        """Manual wake command."""
        self.engine.wake("manual_wake", source=source)


# ============================================================
# IDLE SCHEDULER
# ============================================================

class IdleScheduler:
    """
    Scheduler that monitors idle state and triggers rumination.
    """

    def __init__(self, engine: RuminationEngine,
                 idle_threshold: float = 5.0,
                 use_indefinite: bool = True):
        self.engine = engine
        self.hooks = WakeHooks(engine)
        self.idle_threshold = idle_threshold
        self.use_indefinite = use_indefinite

        self.last_query_time = time.time()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

    def record_query(self):
        """Record query and wake from rumination."""
        self.last_query_time = time.time()
        if self.engine.is_running:
            self.hooks.on_user_prompt("query")

    def start_monitoring(self):
        """Start idle monitoring."""
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def _monitor_loop(self):
        while self._running:
            time.sleep(1.0)
            idle_time = time.time() - self.last_query_time
            if idle_time >= self.idle_threshold and not self.engine.is_running:
                self.hooks.on_idle_enter()

    def stop_monitoring(self):
        self._running = False
        self.hooks.on_shutdown()

    def wake(self):
        self.hooks.on_manual_wake()
        self.last_query_time = time.time()


# ============================================================
# RTTA-R INTEGRATION
# ============================================================

class RTTARumination:
    """
    Complete RTTA-R system with schema-compliant structures.

    Control API:
    - start() -> enables idle rumination
    - wake() -> wakes from rumination
    - stop() -> hard stop
    """

    def __init__(self, rtta_solver: Callable,
                 ledger_path: str = "logs/failure_ledger.json",
                 operator_path: str = "logs/operator_bank.json",
                 idle_threshold: float = 5.0):
        self.ledger = FailureLedger(ledger_path)
        self.operator_bank = OperatorBank(operator_path)
        self.engine = RuminationEngine(rtta_solver, self.ledger, self.operator_bank)
        self.scheduler = IdleScheduler(self.engine, idle_threshold=idle_threshold)
        self.hooks = self.scheduler.hooks

        self.total_failures_logged = 0

    def log_failure(self, fid: str, question: str, expected: str,
                    model_answer: str, result: Dict) -> FailureRecord:
        record = self.ledger.add_failure(fid, question, expected, model_answer, result)
        self.total_failures_logged += 1
        return record

    def start(self):
        """Start idle rumination scheduler."""
        self.scheduler.start_monitoring()

    def wake(self):
        """Wake from rumination."""
        self.scheduler.wake()

    def stop(self):
        """Stop rumination."""
        self.scheduler.stop_monitoring()

    def ruminate_now(self, budget: int = 10):
        """Immediate budgeted rumination."""
        self.engine.rumination_loop(budget)

    def get_stats(self) -> Dict:
        unsolved = self.ledger.get_unsolved()
        return {
            'total_failures': len(self.ledger.records),
            'unsolved': len(unsolved),
            'total_cycles': self.engine.total_cycles,
            'solutions_found': self.engine.solutions_found,
            'proposed_operators': len(self.operator_bank.proposed),
            'mode': self.engine.state.mode,
            'highest_impact': unsolved[0].impact_score if unsolved else 0,
            'highest_desire': max((f.desire_level for f in unsolved), default=0),
        }

    def report(self) -> str:
        stats = self.get_stats()
        unsolved = self.ledger.get_unsolved()[:5]

        lines = [
            "=" * 60,
            "RTTA-R RUMINATION REPORT",
            "=" * 60,
            f"Version: {self.ledger.version}",
            f"Mode: {stats['mode']}",
            f"Total failures: {stats['total_failures']}",
            f"Unsolved: {stats['unsolved']}",
            f"Cycles: {stats['total_cycles']}",
            f"Solutions: {stats['solutions_found']}",
            "",
            "Top 5 by impact:",
        ]

        for f in unsolved:
            stag = f" stag={f._stagnation_penalty:.2f}" if f._stagnation_penalty > 0 else ""
            lines.append(f"  [{f.id}] impact={f.impact_score:.2f} desire={f.desire_level:.2f}{stag}")
            lines.append(f"       type={f.gist.type}, slot={f.first_unmet_slot}, cycles={f.rumination_cycles}")

        return "\n".join(lines)

    def get_state(self) -> Dict:
        """Get schema-compliant rumination state."""
        return self.engine.state.to_dict()


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from tks_features.rtta_canonical import RTTACanonical

    # Fresh start
    for f in ["logs/failure_ledger.json", "logs/operator_bank.json"]:
        if os.path.exists(f):
            os.remove(f)

    rtta = RTTACanonical()
    rumination = RTTARumination(rtta.reason)

    test_failures = [
        ("F01", "Pattern: 1, 1, 2, 3, 5, 8, ?", "13"),
        ("F02", "Every bird can fly. Sparrows are birds. Can sparrows fly?", "Yes"),
        ("F03", "Pens cost $3, pencils $1. Bought 5, spent $11. Pens?", "3"),
    ]

    print("=== Logging failures ===")
    for qid, question, expected in test_failures:
        result = rtta.reason(question, verbose=False)
        if expected.lower() not in str(result['answer']).lower():
            record = rumination.log_failure(qid, question, expected, result['answer'], result)
            print(f"{qid}: impact={record.impact_score:.2f} slot={record.first_unmet_slot}")

    print()
    print(rumination.report())

    print()
    print("=== Rumination state (schema) ===")
    print(json.dumps(rumination.get_state(), indent=2))

    print()
    print("=== Running 10 cycles ===")
    rumination.ruminate_now(budget=10)

    print()
    print(rumination.report())
