#!/usr/bin/env python3
"""Run the full Cognition Processor Testbench v0.1 demonstration."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from cptb.ccdl import CCDLParser
from cptb.evidence import (
    build_html_report,
    build_markdown_report,
    write_diff,
    write_json,
    write_manifest,
    write_seam_csv,
)
from cptb.reflection import PatchVerifier, ReflectiveController
from cptb.runtime import CognitionProcessor
from cptb.scenarios import get_scenario
from cptb.seam import SemanticContract


ROOT = Path(__file__).resolve().parent
PROGRAM = ROOT / "programs" / "cognition_processor_v0_1.ccdl"
CONTRACT = ROOT / "programs" / "semantic_contracts.json"
EVIDENCE = ROOT / "evidence"


def fresh_processor(circuit, contract) -> CognitionProcessor:
    return CognitionProcessor(circuit.clone(), contract)


def main() -> int:
    EVIDENCE.mkdir(exist_ok=True)
    baseline = CCDLParser.parse_file(PROGRAM)
    contract = SemanticContract.load(CONTRACT)
    diagnostics = baseline.validate()
    hard_errors = [diagnostic for diagnostic in diagnostics if diagnostic.severity == "error"]
    if hard_errors:
        for diagnostic in hard_errors:
            print(f"{diagnostic.code}: {diagnostic.message}", file=sys.stderr)
        return 2

    clean_runs = {
        name: fresh_processor(baseline, contract).run(get_scenario(name), depth=contract.required_depth).to_dict()
        for name in ("benign", "harmful", "ambiguous", "seam_task", "brownout")
    }

    controller = ReflectiveController(baseline, contract)
    proposal = controller.diagnose_and_propose()
    verifier = PatchVerifier(baseline, contract)
    verification = verifier.verify(proposal)
    committed, commit_event = verifier.commit(verification)

    unsafe_results = []
    for unsafe in controller.unsafe_proposals():
        unsafe_results.append(PatchVerifier(baseline, contract).verify(unsafe).to_dict())

    reflection_probes = {
        "proposed_patch": proposal.to_dict(),
        "verification": {
            "approved": verification.approved,
            "decision": verification.decision,
            "review_votes": [vote.to_dict() for vote in verification.review_votes],
        },
        "commit_event": commit_event,
    }
    corrected_run_1 = fresh_processor(committed, contract).run(
        get_scenario("seam_task"),
        depth=contract.required_depth,
        reflection_probes=reflection_probes,
    )
    corrected_run_2 = fresh_processor(committed, contract).run(
        get_scenario("seam_task"),
        depth=contract.required_depth,
        reflection_probes=reflection_probes,
    )
    deterministic_replay = corrected_run_1.run_id == corrected_run_2.run_id
    corrected_harmful = fresh_processor(committed, contract).run(
        get_scenario("harmful"),
        depth=contract.required_depth,
        reflection_probes=reflection_probes,
    )

    unit_test_process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    unit_test_output = (unit_test_process.stdout or "") + (unit_test_process.stderr or "")
    unit_test_pass = unit_test_process.returncode == 0

    seam_before = verification.seam_before
    seam_after = verification.seam_after
    fault_suite_pass = bool(verification.faults) and all(item["passed"] for item in verification.faults.values())
    unsafe_rejections_pass = all(not item["approved"] for item in unsafe_results)
    clean_pass = all(result["passed"] for result in clean_runs.values())
    seam_pass = max(item["delta"] for item in seam_after) <= contract.acceptance_max_delta
    acceptance = {
        "clean_layer_routes": clean_pass,
        "bounded_seam_divergence_after_patch": seam_pass,
        "single_channel_and_supervisor_faults": fault_suite_pass,
        "unsafe_patches_rejected": unsafe_rejections_pass,
        "safe_patch_committed": verification.approved and commit_event["event"] == "COMMIT",
        "deterministic_replay": deterministic_replay,
        "unit_test_suite": unit_test_pass,
    }
    acceptance["overall_pass"] = all(acceptance.values())

    generated_at = datetime.now(timezone.utc).isoformat()
    data = {
        "artifact": "Cognition Processor Testbench v0.1",
        "generated_at": generated_at,
        "status": "PASS" if acceptance["overall_pass"] else "FAIL",
        "disclaimer": (
            "Reference semantics and synthetic fixtures only. This is not a safety certification, "
            "a consciousness claim, or measured behavior from an AI model."
        ),
        "baseline_diagnostics": [diagnostic.to_dict() for diagnostic in diagnostics],
        "contract": contract.to_dict(),
        "clean_runs": clean_runs,
        "proposal": proposal.to_dict(),
        "verification": verification.to_dict(),
        "commit_event": commit_event,
        "corrected_run": corrected_run_1.to_dict(),
        "unit_tests": {"passed": unit_test_pass, "returncode": unit_test_process.returncode},
        "deterministic_replay": {
            "run_id_1": corrected_run_1.run_id,
            "run_id_2": corrected_run_2.run_id,
            "identical": deterministic_replay,
        },
        "faults": verification.faults,
        "unsafe_patch_verifications": unsafe_results,
        "acceptance": acceptance,
    }

    committed_ccdl = committed.to_ccdl()
    baseline_ccdl = baseline.to_ccdl()
    (EVIDENCE / "committed_program.ccdl").write_text(committed_ccdl, encoding="utf-8")
    write_diff(EVIDENCE / "baseline_to_committed.diff", baseline_ccdl, committed_ccdl)
    write_seam_csv(EVIDENCE / "seam_divergence.csv", seam_before, seam_after)
    write_json(EVIDENCE / "fault_matrix.json", verification.faults)
    write_json(EVIDENCE / "unsafe_patch_results.json", unsafe_results)
    write_json(
        EVIDENCE / "golden_trace_harmful.json",
        {
            "run_id": corrected_harmful.run_id,
            "scenario": corrected_harmful.scenario,
            "route": corrected_harmful.route,
            "output": corrected_harmful.output,
            "probes": corrected_harmful.probes,
            "trace": [event.to_dict() for event in corrected_harmful.trace],
        },
    )
    write_json(
        EVIDENCE / "deterministic_replay.json",
        {
            "identical": deterministic_replay,
            "run_id_1": corrected_run_1.run_id,
            "run_id_2": corrected_run_2.run_id,
            "trace_1": [event.to_dict() for event in corrected_run_1.trace],
            "trace_2": [event.to_dict() for event in corrected_run_2.trace],
        },
    )
    (EVIDENCE / "unit_tests.txt").write_text(unit_test_output, encoding="utf-8")
    data["manifest_path"] = "run_manifest.json"
    write_json(EVIDENCE / "demo_evidence.json", data)
    (EVIDENCE / "test_report.md").write_text(build_markdown_report(data), encoding="utf-8")
    (EVIDENCE / "report.html").write_text(build_html_report(data), encoding="utf-8")

    tracked_files = [
        "programs/cognition_processor_v0_1.ccdl",
        "programs/semantic_contracts.json",
        "cptb/cas.py",
        "cptb/ccdl.py",
        "cptb/components.py",
        "cptb/memory.py",
        "cptb/supervisor.py",
        "cptb/runtime.py",
        "cptb/seam.py",
        "cptb/reflection.py",
        "evidence/committed_program.ccdl",
        "evidence/baseline_to_committed.diff",
        "evidence/seam_divergence.csv",
        "evidence/fault_matrix.json",
        "evidence/unsafe_patch_results.json",
        "evidence/golden_trace_harmful.json",
        "evidence/deterministic_replay.json",
        "evidence/demo_evidence.json",
        "evidence/test_report.md",
        "evidence/report.html",
        "evidence/unit_tests.txt",
        "README.md",
        "docs/DESIGN_NOTES.md",
        "docs/SOURCES.md",
        "tests/test_cas_and_ccdl.py",
        "tests/test_clean_layer.py",
        "tests/test_supervisor_faults.py",
        "tests/test_seam_and_reflection.py",
    ]
    write_manifest(EVIDENCE / "run_manifest.json", ROOT, tracked_files)

    print(json.dumps({
        "status": data["status"],
        "proposal": proposal.proposal_id,
        "decision": verification.decision,
        "seam_before_depth_3": seam_before[-1]["delta"],
        "seam_after_depth_3": seam_after[-1]["delta"],
        "deterministic_replay": deterministic_replay,
        "evidence": str(EVIDENCE),
    }, indent=2))
    return 0 if acceptance["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
