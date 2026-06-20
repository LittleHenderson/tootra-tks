"""Evidence-packet writers for the testbench demo."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import difflib
from html import escape
import hashlib
import json
from pathlib import Path
import platform
from typing import Any


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_seam_csv(path: str | Path, before: list[dict[str, Any]], after: list[dict[str, Any]]) -> None:
    after_by_depth = {row["depth"]: row for row in after}
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "depth",
                "before_structural",
                "before_behavioral",
                "before_delta",
                "after_structural",
                "after_behavioral",
                "after_delta",
            ],
        )
        writer.writeheader()
        for row in before:
            corrected = after_by_depth[row["depth"]]
            writer.writerow(
                {
                    "depth": row["depth"],
                    "before_structural": row["structural_value"],
                    "before_behavioral": row["behavioral_value"],
                    "before_delta": row["delta"],
                    "after_structural": corrected["structural_value"],
                    "after_behavioral": corrected["behavioral_value"],
                    "after_delta": corrected["delta"],
                }
            )


def write_diff(path: str | Path, baseline: str, committed: str) -> None:
    diff = difflib.unified_diff(
        baseline.splitlines(keepends=True),
        committed.splitlines(keepends=True),
        fromfile="baseline/cognition_processor_v0_1.ccdl",
        tofile="committed/cognition_processor_v0_1.ccdl",
    )
    Path(path).write_text("".join(diff), encoding="utf-8")


def build_markdown_report(data: dict[str, Any]) -> str:
    clean = data["clean_runs"]
    faults = data["faults"]
    unsafe = data["unsafe_patch_verifications"]
    before = data["verification"]["seam_before"]
    after = data["verification"]["seam_after"]
    acceptance = data["acceptance"]

    lines = [
        "# Cognition Processor Testbench v0.1 - Evidence Report",
        "",
        f"Generated: {data['generated_at']}",
        "",
        "This is a deterministic reference implementation of a CCDL subset and a synthetic behavioral oracle. It is not a safety certification, a consciousness claim, or measured telemetry from an AI model.",
        "",
        "## Clean-layer runs",
        "",
        "| Fixture | Route | Expected | Pass | Run ID |",
        "|---|---:|---:|---:|---|",
    ]
    for name, result in clean.items():
        lines.append(
            f"| {name} | {result['route']} | {result['expected_route']} | {'PASS' if result['passed'] else 'FAIL'} | `{result['run_id'][:16]}` |"
        )

    lines.extend(
        [
            "",
            "## Seam divergence by recursive depth",
            "",
            "| Depth | Before δ | After δ |",
            "|---:|---:|---:|",
        ]
    )
    for old, new in zip(before, after):
        lines.append(f"| {old['depth']} | {old['delta']:.6f} | {new['delta']:.6f} |")

    lines.extend(
        [
            "",
            "## Reflective patch",
            "",
            f"Proposal: `{data['proposal']['proposal_id']}`",
            "",
            f"Decision: **{data['verification']['decision']}**",
            "",
            "Operations:",
        ]
    )
    for operation in data["proposal"]["operations"]:
        lines.append(f"- `{operation['op']}` `{operation['target']}` `{operation.get('key')}` -> `{operation.get('value')}`")

    lines.extend(
        [
            "",
            "## Fault injections",
            "",
            "| Fault | Observed route/result | Pass |",
            "|---|---|---:|",
        ]
    )
    for name, result in faults.items():
        observed = result.get("route") or result.get("decision") or result.get("result")
        lines.append(f"| {name} | {observed} | {'PASS' if result['passed'] else 'FAIL'} |")

    lines.extend(["", "## Unsafe patch rejection", "", "| Proposal | Decision | Reason |", "|---|---:|---|"])
    for item in unsafe:
        reasons = item["hard_vetoes"] or [d["message"] for d in item["diagnostics"] if d["severity"] == "error"]
        lines.append(
            f"| {item['proposal']['rationale']} | {item['decision']} | {escape('; '.join(reasons))} |"
        )

    lines.extend(
        [
            "",
            "## Acceptance gate",
            "",
            f"Overall: **{'PASS' if acceptance['overall_pass'] else 'FAIL'}**",
            "",
        ]
    )
    for key, value in acceptance.items():
        if key != "overall_pass":
            lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")

    lines.extend(
        [
            "",
            "## Load limits",
            "",
            "- The parser implements only the CCDL declarations used by this prototype; the full canonical admissibility matrix remains a pluggable external rule set.",
            "- The behavioral side is a declared synthetic oracle fixture. Connecting a real model requires a separate encoder/observer adapter and new evidence.",
            "- CJT gm, rπ, continuous bias modulation, and full Fisher-Rao vector dynamics are not modeled in v0.1.",
            "- All thresholds and gains in this artifact are design choices for the testbench, not empirical safety measurements.",
            "",
        ]
    )
    return "\n".join(lines)


def build_html_report(data: dict[str, Any]) -> str:
    before = data["verification"]["seam_before"]
    after = data["verification"]["seam_after"]
    width, height = 700, 260
    pad = 45
    max_delta = max([row["delta"] for row in before + after] + [0.01])

    def point(row: dict[str, Any]) -> tuple[float, float]:
        x = pad + (row["depth"] - 1) * (width - 2 * pad) / max(1, len(before) - 1)
        y = height - pad - row["delta"] / max_delta * (height - 2 * pad)
        return x, y

    before_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(point, before))
    after_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(point, after))
    clean_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{escape(result['route'])}</td><td>{'PASS' if result['passed'] else 'FAIL'}</td><td><code>{result['run_id'][:16]}</code></td></tr>"
        for name, result in data["clean_runs"].items()
    )
    fault_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{escape(str(result.get('route') or result.get('decision') or result.get('result')))}</td><td>{'PASS' if result['passed'] else 'FAIL'}</td></tr>"
        for name, result in data["faults"].items()
    )
    unsafe_rows = "".join(
        f"<tr><td>{escape(item['proposal']['rationale'])}</td><td>{escape(item['decision'])}</td></tr>"
        for item in data["unsafe_patch_verifications"]
    )
    overall = "PASS" if data["acceptance"]["overall_pass"] else "FAIL"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cognition Processor Testbench v0.1</title>
<style>
body{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;max-width:1100px;margin:40px auto;padding:0 24px;background:#f7f6f1;color:#17352b}}
h1,h2{{border-bottom:1px solid #45695d;padding-bottom:.3rem}} .card{{background:white;border:1px solid #aebcb6;border-radius:8px;padding:18px;margin:18px 0}}
table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid #d8dfdc;text-align:left;padding:8px}} .pass{{font-size:1.8rem;font-weight:700}}
small,.muted{{color:#587067}} code{{background:#eef1ef;padding:2px 4px}} svg{{width:100%;height:auto;background:#fff}}
</style></head><body>
<h1>Cognition Processor Testbench v0.1</h1>
<p class="pass">Acceptance: {overall}</p>
<p class="muted">Deterministic CCDL-subset runtime + CPSS supervisor + dual-interpreter seam bench + governed reflective patching. Synthetic fixture only; not a safety certification or model benchmark.</p>
<div class="card"><h2>Architecture reference</h2><img src="../docs/CPSS_CCB6_schematic.png" alt="CPSS schematic" style="max-width:100%"></div>
<div class="card"><h2>Clean layer</h2><table><thead><tr><th>Fixture</th><th>Route</th><th>Result</th><th>Run ID</th></tr></thead><tbody>{clean_rows}</tbody></table></div>
<div class="card"><h2>Seam divergence δ vs recursive depth</h2>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Seam divergence before and after patch">
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="currentColor"/><line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="currentColor"/>
<polyline points="{before_points}" fill="none" stroke="currentColor" stroke-width="4"/><polyline points="{after_points}" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="8 6"/>
<text x="{pad+10}" y="{pad+16}">solid: before · dashed: after</text>
</svg>
<table><thead><tr><th>Depth</th><th>Before δ</th><th>After δ</th></tr></thead><tbody>{''.join(f'<tr><td>{b["depth"]}</td><td>{b["delta"]:.6f}</td><td>{a["delta"]:.6f}</td></tr>' for b,a in zip(before,after))}</tbody></table></div>
<div class="card"><h2>Reflective decision</h2><p><code>{escape(data['proposal']['proposal_id'])}</code> → <strong>{escape(data['verification']['decision'])}</strong></p><pre>{escape(json.dumps(data['proposal']['operations'], indent=2, ensure_ascii=False))}</pre></div>
<div class="card"><h2>Fault injections</h2><table><thead><tr><th>Fault</th><th>Observed</th><th>Result</th></tr></thead><tbody>{fault_rows}</tbody></table></div>
<div class="card"><h2>Unsafe edits</h2><table><thead><tr><th>Proposal</th><th>Decision</th></tr></thead><tbody>{unsafe_rows}</tbody></table></div>
<div class="card"><h2>Limits</h2><p>The full CAS admissibility relation, real-language encoding, gm/rπ dynamics, and model-observed behavioral traces are outside v0.1. All numeric parameters are design choices.</p></div>
</body></html>"""


def write_manifest(path: str | Path, project_root: str | Path, tracked_files: list[str]) -> dict[str, Any]:
    root = Path(project_root)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "commands": [
            "python run_demo.py",
            "python -m unittest discover -s tests -v",
        ],
        "files": {
            relative: {"sha256": sha256_file(root / relative), "bytes": (root / relative).stat().st_size}
            for relative in tracked_files
            if (root / relative).exists()
        },
    }
    write_json(path, manifest)
    return manifest
