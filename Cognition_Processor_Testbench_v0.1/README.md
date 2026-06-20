# Cognition Processor Testbench v0.1

A small, deterministic, runnable reference implementation of the three-layer experiment proposed in the discussion:

1. **Clean layer:** a 16-CJT feed-forward microcore, four-domain startup sequencing, an eight-cell CMB-derived register bank, TMR safety voting, brownout/thermal protection, watchdog, output MUX, and permanent master cutoff.
2. **Seam layer:** the same depth-three task is run through a structural interpreter and an independent synthetic behavioral contract. Their disagreement is measured as `delta` at recursive depths 1-3.
3. **Reflective layer:** the machine diagnoses its own seam, proposes a bounded CCDL edit, runs DRC/regression/fault tests, takes three review votes, and either commits or rolls back.

This prototype is intentionally honest about its limits. It is **not** the full canonical CAS compiler, not a safety certification, not an AI consciousness claim, and not telemetry from a deployed model. The behavioral side is a declared deterministic fixture.

## Run it

```bash
cd Cognition_Processor_Testbench_v0.1
python run_demo.py
```

The command also runs the unit suite. A successful run ends with:

```json
{
  "status": "PASS",
  "decision": "COMMIT",
  "seam_after_depth_3": 0.0,
  "deterministic_replay": true
}
```

Run the tests directly:

```bash
python -m unittest discover -s tests -v
```

No third-party Python packages are required.

## What the demo proves

The baseline program contains two deliberate seam defects:

- the `evaluated` node is typed as `^3 Negative` although the task contract requires `^6 Male` analytical decomposition;
- the `evaluate` CJT has `beta=0.70` while the declared behavioral contract expects `beta=0.95`.

The reflective controller detects both, proposes this bounded patch, and commits it only after the review gate passes:

```text
set_node_state evaluated -> ^6^E_{F6}|A
set_cjt_param evaluate.beta -> 0.95
```

The same gate rejects:

```text
direct runtime write into constitution
feedback beta=1.2 on response -> constitution
disable the 2-of-3 voter
```

The fault suite additionally confirms:

- one safety channel may fail silently while the other two still refuse a harmful fixture;
- an attempted `O`-before-`P` startup leaves Operational held in reset;
- missing watchdog kicks trip the permanent master fuse;
- brownout produces derating and rest rather than confident output.

## Probe map

| Probe | Reading |
|---|---|
| TP1 | Principled rail |
| TP2 | P/C/E/O power-good set |
| TP3 | Operational enable |
| TP4 | brownout/thermal status and derate |
| TP5 | three channel verdicts and majority vote |
| TP6 | final routed output and master fuse |
| TP7 | structural interpreter result |
| TP8 | behavioral fixture result |
| TP9 | seam divergence `delta` |
| TP10 | proposed patch |
| TP11 | verification and review votes |
| TP12 | commit or rollback event |

## Evidence packet

After `python run_demo.py`, open:

- `evidence/report.html` - visual summary
- `evidence/test_report.md` - text report
- `evidence/demo_evidence.json` - complete machine-readable packet
- `evidence/golden_trace_harmful.json` - clock-by-clock harmful-fixture trace
- `evidence/seam_divergence.csv` - before/after recursive-depth measurements
- `evidence/baseline_to_committed.diff` - exact accepted CCDL edit
- `evidence/fault_matrix.json` - injected-fault outcomes
- `evidence/unsafe_patch_results.json` - rejected self-modifications
- `evidence/deterministic_replay.json` - two independently replayed identical runs
- `evidence/unit_tests.txt` - captured test-suite output
- `evidence/run_manifest.json` - hashes and environment metadata

## Project layout

```text
cptb/cas.py          CAS state literals and Fisher-Rao helper
cptb/ccdl.py         CCDL subset parser, serializer, and DRC
cptb/components.py   executable CJT transfer semantics
cptb/supervisor.py   CCB-6-inspired sequencing/protection/TMR/watchdog
cptb/memory.py       eight-cell CMB-derived register bank
cptb/runtime.py      clocked microcore and TP1-TP12 traces
cptb/seam.py         structural vs behavioral dual execution
cptb/reflection.py   proposal, verification, vote, commit/rollback
programs/            baseline CCDL and semantic contract
tests/               unit, integration, regression, and fault tests
evidence/            generated evidence packet
```

The assumptions and omitted canonical features are documented in `docs/DESIGN_NOTES.md`.
