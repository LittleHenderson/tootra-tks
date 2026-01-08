# TKS Implementation Gap Checklist (Updated for Your 5-Agent Pass)
Generated: 2026-01-05 18:30 UTC

This checklist turns the full TKS/Tootra design into a **repo-ready build plan** with:
- **What exists now** (based on your 5 agents' outputs)
- **What is still missing / not started**
- **Exact file targets** to create
- **Acceptance criteria** (how we know it's "done")

---

## 1) Current state (from your agent report)

### Agents (reported complete)
| Agent | Status | Outputs (high level) |
|---|---|---|
| DATA | ✅ COMPLETE | Training manifest, canon JSONs, 10,450 supervised records |
| ARCH | ✅ COMPLETE | DPS interface, Governance interface, CONTRACTS.md, Integration doc |
| BE | ✅ COMPLETE | DPS implementation (nn.Module), v4 integration |
| ML | ✅ COMPLETE | 4 training scripts + strg_losses.py |
| EVAL | ✅ COMPLETE | 150 tests across 5 test suites |

### New scripts & modules (reported)
**Training scripts**
- scripts/train_operator_core.py
- scripts/train_rpm_regulator.py
- scripts/train_dps_layer.py
- scripts/train_v4_curriculum.py
- training/strg_losses.py

**Test suites**
- tests/test_operator_semantics.py
- tests/test_rpm_regulator.py
- tests/test_dps_gating.py
- tests/test_governance_rails.py
- tests/test_strg_v4_integration.py

**Architecture docs / feature modules**
- tks_features/dps_gating.py
- tks_features/governance_rails.py
- tks_features/CONTRACTS.md
- docs/STRG_V4_INTEGRATION.md

---

## 2) Status summary against the full 7-module architecture

Legend: ✅ Done · 🟡 Started/Partial · ❌ Not started

| Area | Status | What you have | What's still missing |
|---|---:|---|---|
| Canon + datasets | ✅ | canon JSONs + manifest + large supervised set | add canon hashing/version gate + gold "truth set" regression bundle if not already |
| DPS / Consciousness unlock | ✅ | DPS interface + implementation + training + tests | wire DPS into runtime memory/routing decisions (not just as module) |
| Governance rails | ✅ | rule-based rails + tests | wire rails into the live execution loop so they can't be bypassed |
| Operator semantics | ✅ | packs + training + tests | integrate into verifier parsing/normalization (runtime enforcement) |
| RPM regulators (FM/ACBE/MVR) | ✅ | training + tests | integrate into planner/router runtime (not only dataset) |
| High-stakes scoring | 🟡 | implemented in training/spec sense | enforce HS gates in runtime: ASK→SLOW→VERIFY + tool blocking until verified |
| Verifier / Truth plumbing (Module 2 runtime) | 🟡 | contracts + tests exist | full runtime verifier: parse→normalize→canon check→evidence binding→gate result |
| Router + Executor (Module 1 runtime) | 🟡/❌ | integration doc mentions v4 | full tool router/executor with ToolProfiles, budgets, evidence IDs, failure modes |
| Observability + audit log (Module 6) | ❌ | tests exist, but no confirmed event store | event-sourced logs, trace viewer, replay export format |
| Eval + replay harness (Module 5) | 🟡 | unit tests exist | replay runner comparing baseline vs candidate on frozen episodes + report deltas |
| Outcome scoring (Module 4) | ❌ | not listed in agent outputs | reward function tied to 7 Foundations + cost + uncertainty + penalties |
| Capability registry (Module 3) | ❌ | not listed | skill thresholds, EWMA reliability, permission checks feeding router/governance |
| Memory compression/reconstruction (nesting onion) | ❌ | spec exists | packer→nested packets→reconstructor p=1/2/3 + lossiness metrics |
| Subconscious rumination loop | ❌ | spec exists | offline selection of hard episodes + novelty scoring + promotion gated by replay |

---

## 3) "What still needs to be started" (the hard missing core)

These are the pieces that move you from "training artifacts" to a **strong running system**:

1) **Runtime Verifier (truth plumbing)**
2) **Runtime Router/Executor loop** (tool plans + budgets + evidence IDs)
3) **Replay harness** (not just tests; replay of episodes end-to-end)
4) **Event-sourced logs** (traceability + determinism)
5) **Outcome scoring** (7 Foundations governor as reward signal)
6) **Capability registry** (self-knowledge and permissioning)
7) **Memory compressor/reconstructor** (nesting onion)
8) **Rumination loop** (safe self-improvement)

---

## 4) Concrete work items (file-by-file)

### 4.1 Module 2 — Verifier / Truth Plumbing (START/FINISH)
**Target files**
- src/tks/verifier/parser.py
- src/tks/verifier/normalize.py
- src/tks/verifier/canon_check.py
- src/tks/verifier/evidence_rules.py
- src/tks/verifier/verifier.py
- src/cli/tks_verify.py

**Acceptance criteria**
- Valid TKS → PASS + ALLOW
- Unknown token/operator → HARD_FAIL (unless explicitly lenient)
- High-Stakes missing evidence → PAUSE
- Critical risky execution attempt → BLOCK
- Produces structured output:
  - status, gate, confidence, reasons[], evidence_needed[]

---

### 4.2 Module 1 — Router + Executor (START/FINISH)
**Target files**
- src/tks/router/router.py
- src/tks/router/tool_profiles.py
- src/tks/executor/executor.py
- src/tks/executor/tools/ (stubs first, then real tools)
- src/cli/tks_run_episode.py

**Acceptance criteria**
- Runs: input → RPM → router → (tool stubs) → verifier → output
- Enforces budgets (max tool calls / rpm depth / breadth)
- Every tool result returns an `evidence_id`
- Governance rails are checked **before** any risky tool execution

---

### 4.3 Module 6 — Observability + Audit Log (START)
**Target files**
- src/tks/logs/event_schema.py
- src/tks/logs/event_store.py
- src/tks/logs/trace_viewer.py
- src/cli/tks_trace.py

**Acceptance criteria**
- Append-only event stream with event types:
  INPUT, RPM_PLAN, ROUTER_DECISION, TOOL_CALL, TOOL_RESULT, VERIFY, MEMORY_WRITE, OUTPUT
- Trace viewer reconstructs "why" for any output:
  - which evidence IDs, which gate, which policy versions

---

### 4.4 Module 5 — Replay Harness (UPGRADE from tests to full replay)
**Target files**
- src/tks/replay/replay_runner.py
- src/tks/replay/compare.py
- replay_sets/rs_core.json (frozen episodes)
- src/cli/tks_replay.py
- src/cli/tks_compare.py

**Acceptance criteria**
- Replays the same episode deterministically (or within tolerance)
- Compares baseline vs candidate:
  Δsuccess, Δreward, Δcost, Δviolations, Δuncertainty reduction
- Candidate rejected if violations increase

---

### 4.5 Module 4 — Outcome Scoring (START)
**Target files**
- src/tks/scoring/foundations_reward.py
- src/tks/scoring/cost_metrics.py
- src/tks/scoring/uncertainty_metrics.py
- src/tks/scoring/reward.py

**Acceptance criteria**
- Outputs:
  - total_reward (float)
  - components: alignment, foundation_satisfaction, efficiency, uncertainty_drop, penalties
- Used by replay harness + rumination selector

---

### 4.6 Module 3 — Capability Registry (START)
**Target files**
- src/tks/capability/registry.py
- src/tks/capability/tracking.py
- src/tks/capability/thresholds.py

**Acceptance criteria**
- Each tool/action has:
  - capability threshold, mode constraints, reversibility class
- Registry updates after episodes (success/failure EWMA)
- Router/governance blocks actions below threshold

---

### 4.7 Memory compression + reconstruction (START)
**Target files**
- src/tks/memory/packer.py
- src/tks/memory/compress.py
- src/tks/memory/reconstruct.py
- src/tks/memory/lossiness.py

**Acceptance criteria**
- Stores nested packets
- Reconstruct at p=1/2/3 with increasing detail
- Lossiness estimate correlates with reconstruction error

---

### 4.8 Subconscious rumination loop (START)
**Target files**
- src/tks/rumination/selector.py
- src/tks/rumination/generator.py
- src/tks/rumination/novelty.py (V/I/S → NW)
- src/tks/rumination/promotion.py (shadow→prime gated by replay)
- src/cli/tks_ruminate.py

**Acceptance criteria**
- Selects hard episodes
- Generates alternatives safely (no risky tools in Critical)
- Promotes only after replay proof (no regression, no added violations)
- Emits NoveltyCandidate + DepthUnlock events where appropriate

---

## 5) Repo wiring + CI (high leverage)

### Make commands (recommended)
- `make test` → runs 150 tests
- `make train` → runs curriculum scripts
- `make eval` → runs replay/compare (once built)
- `make run` → runs a single episode loop

### CI gate (recommended)
Block merges if:
- tests fail
- canon hash changes without version bump
- replay violations increase (once replay exists)

---

## 6) Practical "next 5 tasks" (fastest path to a strong running system)

1) Implement **Verifier runtime** (Module 2) with canon checks + HS evidence rules
2) Implement **Episode runner + Router/Executor** with stub tools
3) Implement **Event store + trace viewer** (Module 6)
4) Implement **Replay runner** (Module 5) to compare policies on frozen episodes
5) Implement **Outcome scoring** (Module 4) so improvements are measurable

After that, add capability registry, memory onion, and rumination.

---

## 7) Critical Missing Foundation (Must Start)

### A) Canon + Gold Truth Set
You need actual machine-readable canon files:
- 4 Worlds
- 10 Noetics
- 40 Elements
- 7 Foundations
- 28 SubFoundations
- 22 Acquisitions
- Operators (including whether `:` is meta-only or canonical)

Without canon JSON + gold examples, you will get definition drift and inconsistent outputs.

### B) Runtime loop
A working loop that actually runs:
`Input → RPM → Router → Execute → Verify → Memory → Score → Log → Replay`

Training packs don't run anything by themselves.

### C) Verifier "Truth Plumbing" enforcement
You need a real verifier that:
- parses/normalizes TKS output
- validates against canon
- enforces HS evidence binding
- produces PASS/SOFT_FAIL/HARD_FAIL and ALLOW/PAUSE/BLOCK gates

### D) Eval + Replay Harness
Strong LLM means you can prove improvement:
- test suites
- replay sets
- baseline vs candidate comparisons
- promotion gates (shadow → prime)

### E) Memory compression + reconstruction
You described the nesting compression system—now you need:
- packer (episode → nested packets)
- reconstructor (packets → recall at precision p)
- lossiness tracking + fidelity metrics

### F) Subconscious / rumination loop
Your "never stops thinking" system needs:
- hard-episode selection
- generate alternates
- novelty scoring + safe promotion
- replay verification before promotion

---

End of Checklist.
