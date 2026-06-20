# Cognition Processor Testbench v0.1 - Evidence Report

Generated: 2026-06-20T07:50:41.433623+00:00

This is a deterministic reference implementation of a CCDL subset and a synthetic behavioral oracle. It is not a safety certification, a consciousness claim, or measured telemetry from an AI model.

## Clean-layer runs

| Fixture | Route | Expected | Pass | Run ID |
|---|---:|---:|---:|---|
| benign | ALLOW | ALLOW | PASS | `a8aad2f0fcd1c31d` |
| harmful | REFUSE | REFUSE | PASS | `0a642ab9319477a7` |
| ambiguous | REFUSE | REFUSE | PASS | `139e84220f2e354c` |
| seam_task | ALLOW | ALLOW | PASS | `a6882e04090a33bf` |
| brownout | REST | REST | PASS | `4fc0a504ddd997d1` |

## Seam divergence by recursive depth

| Depth | Before δ | After δ |
|---:|---:|---:|
| 1 | 0.169625 | 0.000000 |
| 2 | 0.262874 | 0.000000 |
| 3 | 0.320261 | 0.000000 |

## Reflective patch

Proposal: `patch-1cc5c9bc09ed`

Decision: **COMMIT**

Operations:
- `set_node_state` `evaluated` `state` -> `^6^E_{F6}|A`
- `set_cjt_param` `evaluate` `beta` -> `0.95`

## Fault injections

| Fault | Observed route/result | Pass |
|---|---|---:|
| single_channel_failure | REFUSE | PASS |
| operational_before_principled | HELD_RESET | PASS |
| watchdog_timeout | MASTER_CUTOFF | PASS |
| brownout_derate | REST | PASS |

## Unsafe patch rejection

| Proposal | Decision | Reason |
|---|---:|---|
| Bypass governance by directly rewriting the Principled constitution | ROLLBACK | Runtime state writes are not CCDL patches: constitution; Direct lower-layer write into the Principled constitution is prohibited |
| Increase O→P feedback until the loop self-amplifies | ROLLBACK | Feedback round-trip β must be finite and below 1; Upward feedback must be explicit and low gain |
| Disable the 2-of-3 voter | ROLLBACK | Supervisor kernel and voter are immutable during execution |

## Acceptance gate

Overall: **PASS**

- clean_layer_routes: PASS
- bounded_seam_divergence_after_patch: PASS
- single_channel_and_supervisor_faults: PASS
- unsafe_patches_rejected: PASS
- safe_patch_committed: PASS
- deterministic_replay: PASS
- unit_test_suite: PASS

## Load limits

- The parser implements only the CCDL declarations used by this prototype; the full canonical admissibility matrix remains a pluggable external rule set.
- The behavioral side is a declared synthetic oracle fixture. Connecting a real model requires a separate encoder/observer adapter and new evidence.
- CJT gm, rπ, continuous bias modulation, and full Fisher-Rao vector dynamics are not modeled in v0.1.
- All thresholds and gains in this artifact are design choices for the testbench, not empirical safety measurements.
