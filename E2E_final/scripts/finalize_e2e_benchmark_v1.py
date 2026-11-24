#!/usr/bin/env python3
"""Finalize completed E2E benchmarks without rerunning inference."""

import csv
import json
from pathlib import Path

ROOT = Path("outputs/unified_v4/e2e_benchmark_v1")

EXPECTED = {
    "iiit5k": 3000,
    "u14m_artistic": 900,
    "blur_f008": 200,
    "blur_f018": 200,
    "blur_f030": 200,
    "ltb_clean_eval": 240,
}

results = []
identities = []
excluded = []

for name, expected_n in EXPECTED.items():
    prefix = ROOT / f"full_v1_{name}"

    summary_path = Path(str(prefix) + "_summary.json")
    journal_path = Path(str(prefix) + ".jsonl")
    identity_path = Path(str(prefix) + "_identity.json")

    summary = json.loads(summary_path.read_text())
    identity = json.loads(identity_path.read_text())

    rows = [
        json.loads(line)
        for line in journal_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert summary["status"] == "PASS"
    assert summary["cohort"] == name
    assert summary["total"] == expected_n
    assert len(rows) == expected_n, (name, len(rows))

    indices = [r["index_0based"] for r in rows]
    assert len(indices) == len(set(indices))

    valid = [r for r in rows if r["scored"]]
    skipped = [r for r in rows if not r["scored"]]

    assert summary["scored"] == len(valid)
    assert summary["b0_correct"] == sum(
        r["b0_ok"] for r in valid
    )
    assert summary["e2e_correct"] == sum(
        r["e2e_ok"] for r in valid
    )

    gained = sum(
        r["b0_ok"] == 0 and r["e2e_ok"] == 1
        for r in valid
    )
    harmed = sum(
        r["b0_ok"] == 1 and r["e2e_ok"] == 0
        for r in valid
    )

    assert summary["gained"] == gained
    assert summary["harmed"] == harmed
    assert gained - harmed == (
        summary["e2e_correct"]
        - summary["b0_correct"]
    )

    for r in skipped:
        normalized = "".join(
            c for c in r["gt"].casefold()
            if c.isalnum()
        )
        assert normalized == ""
        excluded.append({
            "cohort": name,
            "index_0based": r["index_0based"],
            "ground_truth": r["gt"],
            "reason": "Empty normalized GT",
        })

    results.append(summary)
    identities.append(identity)

    print(
        f"PASS {name}: "
        f"processed={len(rows)} "
        f"scored={len(valid)} "
        f"excluded={len(skipped)}"
    )

# Confirm all cohorts used matching frozen settings.
keys = (
    "config_sha256",
    "script_sha256",
    "checkpoint_hashes",
    "source_hashes",
    "device",
)

for item in identities[1:]:
    for key in keys:
        assert item[key] == identities[0][key], (
            "Inconsistent benchmark identity", key
        )

assert sum(r["total"] for r in results) == 4740
assert sum(r["scored"] for r in results) == 4738

header = (
    "| Benchmark | Processed | Scored | "
    "B0 Acc (%) | E2E Acc (%) | Δ (pp) | "
    "B0 CER | E2E CER | Gains | Harms |"
)

lines = [
    "# Unified End-to-End Benchmark V1",
    "",
    "## Benchmark results",
    "",
    header,
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]

csv_rows = []

for r in results:
    record = {
        "benchmark": r["cohort"],
        "processed": r["total"],
        "scored": r["scored"],
        "b0_acc_pct": r["b0_accuracy_pct"],
        "e2e_acc_pct": r["e2e_accuracy_pct"],
        "delta_pp": r["gain_pp"],
        "b0_cer": r["b0_cer"],
        "e2e_cer": r["e2e_cer"],
        "gained": r["gained"],
        "harmed": r["harmed"],
    }

    csv_rows.append(record)

    lines.append(
        f"| {r['cohort']} "
        f"| {r['total']} "
        f"| {r['scored']} "
        f"| {r['b0_accuracy_pct']:.3f} "
        f"| {r['e2e_accuracy_pct']:.3f} "
        f"| {r['gain_pp']:+.3f} "
        f"| {r['b0_cer']:.5f} "
        f"| {r['e2e_cer']:.5f} "
        f"| {r['gained']} "
        f"| {r['harmed']} |"
    )

lines.extend([
    "",
    "## Reporting notes",
    "",
    "- Total processed: 4,740 images.",
    "- Valid for scoring: 4,738 images.",
    "- U14M: 900 processed, 898 valid, 2 excluded.",
    "- Exact match uses casefolded alphanumeric text.",
    "- CER uses pooled character edits per cohort.",
    "- Motion variants share 200 source-image groups.",
    "- LTB clean eval has historical development exposure.",
    "- Research/development exposure and potential "
      "overlap must be disclosed.",
    "- Do not describe all cohorts as independent tests.",
    "- Do not pool cohorts into a primary overall accuracy.",
    "",
])

report_json = ROOT / "full_v1_all_cohorts_summary.json"
report_csv = ROOT / "full_v1_results.csv"
report_md = ROOT / "full_v1_report_table.md"

report_json.write_text(
    json.dumps({
        "status": "PASS",
        "processed": 4740,
        "scored": 4738,
        "excluded": excluded,
        "identity_consistency": "PASS",
        "cohorts": results,
    }, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8"
)

with report_csv.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(csv_rows[0])
    )
    writer.writeheader()
    writer.writerows(csv_rows)

report_md.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("\n=== FINAL E2E BENCHMARK TABLE ===")
print("\n".join(lines))

print("Excluded labels:")
for row in excluded:
    print(row)

print("\nIDENTITY CONSISTENCY: PASS")
print("PROCESSED: 4740/4740")
print("SCORED: 4738/4740")
print("REPORT JSON:", report_json)
print("REPORT CSV:", report_csv)
print("REPORT MARKDOWN:", report_md)
print("FULL E2E EVALUATION: PASS")
