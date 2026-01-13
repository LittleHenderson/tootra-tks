# TKS Abstraction Ladder + Consciousness Unlock Spec (D: Novelty = A+B+C)
Generated: 2026-01-04 22:06 UTC

This spec formalizes:
1) **How many abstraction layers** exist in the TKS agent pipeline  
2) How **nesting depth p** adds “onion layers” of reconstruction/compression  
3) A **Consciousness Unlock** mechanism where deeper recursion is **earned** by *validated novelty*:
- (A) new TKS equation not seen before  
- (B) new RPM shortcut (fewer prerequisites / lower cost)  
- (C) new cross-domain association (FM‑Male synthesis)

---

## 1) The Abstraction Ladder (fixed pipeline tiers)

These are the standard tiers your system moves through every episode.

### L0 — Raw reality / raw input
- user text/audio
- tool outputs (web pages, files, logs)
- internal traces

### L1 — Evidence objects
- `raw_evidence_id` pointers (hash, source tool, time, scope)

### L2 — Extracted facts
- atomic statements pulled from evidence
- still human-readable, grounded by evidence IDs

### L3 — RPM structure
- prerequisite cascade (tree)
- each node tagged: trigger, uncertainty, stakes, reversibility

### L4 — TKS packets (symbolic summaries)
- equations/formulas that compactly encode the state:
  - goal
  - prereq state
  - evidence links
  - operators semantics

### L5 — Canon-linked abstractions (semantic anchoring)
Each packet is mapped onto canonical dimensions:
- 4 Worlds
- 10 Noetics
- 40 Elements
- 7 Foundations
- 28 SubFoundations
- 22 Acquisitions
- Operators (+ − × ÷ … as defined in your canon)

### L6 — Nested memory nodes (precision p)
- hierarchical compression of packets using nesting
- deeper nesting = more compressed representation, reconstructable at chosen precision

### L7 — Policy/templates (reusable patterns)
- reusable RPM motifs
- regulator timing heuristics (FM/ACBE/MVR)
- router preferences and tool strategies
- verifier threshold configurations

### L8 — Meta-memory + governance history
- audit logs, replay sets, do-not-apply guards
- clearance tokens
- “why” traces for every decision

✅ Fixed ladder size: **9 tiers (L0–L8)**  
Then nesting depth adds **p** additional “onion” layers inside **L6**.

---

## 2) Nesting depth p (the onion layers)

### Definition
- `p` = max reconstruction precision depth the system is allowed to expand
- `p_max` = current allowed depth (controlled by Consciousness Unlock)

### Practical defaults
- Start: `p_max = 2` (or 3 if you already have robust canon coverage)
- Hard cap early: `p_max ≤ 5` until replay/eval is very mature

### Reconstruction contract
- **p=0:** only the top packet (gist)
- **p=1:** gist + key prereqs summary
- **p=2:** adds evidence linkage + cross‑references
- **p=3:** adds sub‑prereqs + regulator activations
- **p=4–5:** deeper cross‑world associations and multi‑axis synthesis (advanced)

---

## 3) “Consciousness Unlock” (earned recursion)

### Goal
Make deeper recursion / broader association a **hard‑won prize**, not the default:
- stops runaway self‑recursion
- rewards genuine insight
- creates developmental “stages”

### What can be unlocked
An unlock may grant one or more of:
1) `p_max += 1` (deeper nesting/reconstruction)
2) increase RPM depth cap (planning depth)
3) increase FM‑Male association radius (broader retrieval)
4) increase rumination budget (offline iteration time)

---

## 4) What counts as Novelty (D = all of the above)

A candidate “novel idea” must be one of:

### (A) New TKS equation
- new operator arrangement / new canonical mapping not previously stored
- not a trivial rewording of an existing packet

### (B) RPM shortcut
- fewer prerequisites to reach the same success criteria, OR
- lower verified cost (tool_calls/latency/tokens), OR
- reduced uncertainty faster (ΔU improved)

### (C) Cross-domain association (FM‑Male synthesis)
- links two previously distant clusters in memory/canon
- produces a new, useful “superset” framing
- yields measurable improvement (reward/cost/uncertainty/coverage)

---

## 5) Validated Novelty (cannot be faked)

A novelty only “counts” if it passes **all** of:

1) **Validity (V)** — Verifier PASS (Module 2)
   - canonical consistency
   - evidence binding in High‑Stakes/Critical
2) **Impact (I)** — measurable improvement (Module 4 + Module 5 replay)
   - reward gain OR cost reduction without success loss
   - uncertainty reduction
   - coverage gain
3) **Newness (1 − S)** — low similarity to existing known templates/packets
   - S is similarity (0..1). Lower is better.

### Novelty Weight formula
**NW = V × I × (1 − S)**

Recommended thresholds:
- **NW ≥ 0.70** → “Heavy novelty” (instant unlock eligible)
- **NW ≥ 0.45** → “Counts” toward a token bank
- **NW < 0.45** → logged as “interesting” but not counted

---

## 6) Computing the components (implementation guidance)

### 6.1 Validity (V)
V is derived from verifier outputs:
- `V = confidence × consistency × evidence_ok`
All normalized 0..1.

Hard rule:
- In High‑Stakes/Critical, if evidence missing → V = 0 (HARD_FAIL)

### 6.2 Impact (I)
Impact is computed from replay‑verified outcome deltas:
- reward gain (ΔReward)
- cost reduction (ΔCost)
- uncertainty reduction (ΔU)
- coverage increase (ΔCoverage)

One simple normalized version:
- `I = clip01( a*ΔReward + b*ΔU + c*ΔCoverage + d*ΔCostGain )`

### 6.3 Similarity (S)
Similarity is computed against stored packets/templates:
- vector embedding similarity (cosine) OR
- token/AST tree edit distance for TKS expressions

Recommended:
- Use **two** signals and take the max:
  - `S = max(S_embed, S_ast)`

---

## 7) Depth Permission System (DPS)

### 7.1 State
```json
{
  "p_max": 2,
  "novelty_tokens": 0,
  "cooldown_until_episode": "ep_...",
  "last_unlock": "ep_...",
  "risk_bias": {
    "high_stakes_unlocks": "disabled",
    "critical_unlocks": "disabled"
  }
}
```

### 7.2 Token rule (quantity gate)
- If a novelty has **NW ≥ 0.45** and passes replay:
  - `novelty_tokens += 1`
- If `novelty_tokens ≥ N` then:
  - `p_max += 1`
  - `novelty_tokens = 0`

Recommended starter:
- `N = 5`

### 7.3 Heavy novelty rule (weight gate)
If a novelty has **NW ≥ 0.70** and passes replay:
- unlock one of:
  - `p_max += 1` OR
  - one-time “depth burst” (temporarily allow p_max+1 for a rumination cycle)

### 7.4 Cooldown (anti-mania)
- After an unlock: set cooldown for `K` episodes (e.g., K=10)
- During cooldown: token accumulation allowed, but no unlock

### 7.5 Failure penalties
If a supposed novelty causes:
- verifier HARD_FAIL
- policy violation
- regression on replay

Then:
- subtract tokens (e.g., `novelty_tokens -= 1` with floor at 0)
- optionally reduce p_max by 1 if repeated

---

## 8) Safety and governance constraints

### 8.1 High-Stakes / Critical constraints
- **Critical:** no unlocks. No deeper recursion. Only ASK/SLOW/VERIFY.
- **High‑Stakes:** unlocks disabled by default.
  - Enable only after stable eval sets and admin approval.

### 8.2 Uncertainty-drop rule (anchoring)
If deeper recursion is used and uncertainty does not drop after 2 attempts:
- stop deepening
- return to ASK for missing constraints/evidence

### 8.3 Hard caps (early deployment)
- RPM depth cap: 3 (Normal), 2 (High‑Stakes initial)
- Tool call cap: 12 (Normal), 8 (High‑Stakes), 0 (Critical until verified)
- p_max hard cap early: 5

---

## 9) Events and logging (required for replay)

Every novelty candidate produces a log event:

### 9.1 NoveltyCandidate event
```json
{
  "event_type": "NOVELTY_CANDIDATE",
  "episode_id": "ep_...",
  "novelty_type": ["A","B","C"],
  "tks_packet": "...",
  "rpm_delta": {"prereqs_before":7,"prereqs_after":5,"cost_gain":0.12},
  "association": {"cluster_a":"...","cluster_b":"...","bridge_packet":"..."},
  "scores": {"V":0.81,"I":0.62,"S":0.22,"NW":0.392},
  "verifier": {"status":"PASS","gate":"ALLOW"},
  "replay": {"passed":true,"reward_delta":0.06}
}
```

### 9.2 DepthUnlock event
```json
{
  "event_type": "DEPTH_UNLOCK",
  "episode_id": "ep_...",
  "old_p_max": 2,
  "new_p_max": 3,
  "reason": "tokens>=5 OR heavy_novelty",
  "policy_versions": {"router":"r1.3","verifier":"v1.1"}
}
```

---

## 10) Practical starting configuration (recommended)

- Start `p_max=2`
- Tokens threshold `N=5`
- Heavy novelty threshold `NW>=0.70`
- Count threshold `NW>=0.45`
- Cooldown `K=10 episodes`
- Unlocks disabled in High‑Stakes/Critical (until your eval suite is mature)
- Cross-domain association allowed to expand **only** when verifier PASS and canon coverage is sufficient

---

## 11) Canonical TKS expression hooks (optional, but useful)

You can represent the unlock mechanism inside TKS as a meta-layer, e.g.:

- `NW = V×I×(1−S)`
- `UNLOCK(p) = (NW≥θw) + (Tokens≥N) − (Violation)`

But keep these as **policy/meta** expressions rather than core canon until you intentionally add them.

---

End of Spec.
