"""Independent recalculation of values supported by the uploaded GitHub ZIP.

No new model inference occurs. Historical aggregate-only rows remain clearly
marked as reported, not independently verified.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from .paths import ROOT, UPSTREAM
from .metrics import score_pairs, compare_paired, normalize
from .evaluation import file_sha256

SAGE_RESULTS = UPSTREAM / "reports/long_text/sage_str_v4/results"
SAGE_SOURCE = UPSTREAM / "experiments/long_text/sage_str_v4/src/sage_str_v4/selector.py"
SAGE_COLUMNS = {"SVTRv2-T_single": "svtrv2", "SVTRv2-T_v3": "geometry_v3",
                "Unified_v1": "unified_v1", "Safe_reference": "reference_prediction",
                "SAGE_v3_locked": "sage_v3", "SAGE_v4": "sage_v4"}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def verify_archive(out=None) -> dict:
    predictions_path = SAGE_RESULTS / "evaluation_predictions.csv"
    metrics_path = SAGE_RESULTS / "evaluation_accuracy_cer_by_dataset.csv"
    preds = read_csv(predictions_path)
    metrics = read_csv(metrics_path)
    if len(preds) != 461:
        raise AssertionError(f"SAGE pilot rows differ from source report: {len(preds)} != 461")
    expected_source = json.loads((SAGE_RESULTS / "run_manifest.json").read_text())
    if file_sha256(SAGE_SOURCE) != expected_source["v4_code_sha256"]:
        raise AssertionError("SAGE v4 selector source SHA changed")
    computed = []
    for original in metrics:
        ds = original["dataset"]
        method = original["method"]
        subset = [r for r in preds if r["dataset"] == ds]
        if method not in SAGE_COLUMNS:
            raise AssertionError(f"Unexpected method in upstream metrics: {method}")
        col = SAGE_COLUMNS[method]
        pairs = [(r["gt"], r[col]) for r in subset if normalize(r["gt"])]
        score = score_pairs(pairs)
        if score["n"] != int(original["n"]) or score["correct"] != int(original["exact_correct"]):
            raise AssertionError(f"WA mismatch: {ds}/{method}")
        if score["character_edits"] != int(original["total_edit_errors"]):
            raise AssertionError(f"CER edit mismatch: {ds}/{method}")
        if abs(score["cer_pct"] - float(original["CER_pct"])) > 1e-7:
            raise AssertionError(f"CER rate mismatch: {ds}/{method}")
        computed.append({"dataset": ds, "method": method, **score})

    verify_rows = []
    for ds in sorted({r["dataset"] for r in preds}):
        group = [r for r in preds if r["dataset"] == ds]
        a = score_pairs([(r["gt"], r["svtrv2"]) for r in group])
        b = score_pairs([(r["gt"], r["sage_v4"]) for r in group])
        paired = compare_paired([(r["gt"], r["svtrv2"], r["sage_v4"]) for r in group], bootstraps=2000)
        verify_rows.append({"dataset": ds, "n": len(group), "baseline_method": "SVTRv2-T_single",
                            "method": "SAGE_STR_v4", "baseline_wa_pct": a["word_accuracy_pct"],
                            "method_wa_pct": b["word_accuracy_pct"],
                            "baseline_cer_pct": a["cer_pct"], "method_cer_pct": b["cer_pct"],
                            "delta_wa_pp": paired["delta_wa_pp"], "gained": paired["gained"],
                            "harmed": paired["harmed"],
                            "evidence_type": "RECALCULATED_FROM_PAIRED_UPSTREAM_PREDICTIONS",
                            "source": str(predictions_path.relative_to(ROOT))})
    # These are provenance-labelled upstream aggregate-only values, not replayed OCR.
    tracker = read_csv(UPSTREAM / "tracker/experiments.csv")
    aggregate_rows = []
    for row in tracker:
        if row["experiment_id"] not in {"EXP-001", "EXP-004", "EXP-005", "EXP-006"}:
            continue
        aggregate_rows.append({"experiment_id": row["experiment_id"], "method": row["method"],
                               "dataset": row["dataset"], "split": row["split"],
                               "baseline_wa_pct": row["word_accuracy_before"],
                               "method_wa_pct": row["word_accuracy_after"],
                               "baseline_cer_pct": row["cer_before"],
                               "method_cer_pct": row["cer_after"],
                               "status": row["status"],
                               "evidence_type": "REPORTED_IN_UPSTREAM_TRACKER_NOT_REEVALUATED",
                               "source": "research_upstream/tracker/experiments.csv"})
    report = {"status": "ARCHIVED_EVIDENCE_VERIFIED_NO_NEW_INFERENCE", "source_upstream_zip": "svtrv2-scene-text-recognition-main.zip",
              "sage_selector_sha256": file_sha256(SAGE_SOURCE),
              "sage_predictions_sha256": file_sha256(predictions_path),
              "sage_per_method_metric_rows_verified": len(computed),
              "sage_eval_rows_verified": len(preds),
              "sage_vs_baseline": verify_rows,
              "upstream_aggregate_only": aggregate_rows,
              "real09_cell38_user_colab_log": {
                "dataset": "WATER-T validation512 (historically exposed)",
                "n": 512, "baseline_wa_pct": 48.83, "a1_wa_pct": 47.27,
                "a2_wa_pct": 47.27, "evidence_type": "USER_SUPPLIED_COLAB_LOG_NOT_IN_UPLOADED_ZIP",
                "claim": "A2 did not beat A1; not selected as production Our Method"},
              "integrated_our_method": {"fresh_full_system_metrics": None,
                "status": "NOT_EVALUATED_REQUIRES_ALL_MODEL_CHECKPOINTS_AND_IMAGES",
                "note": "No aggregate across distinct datasets/cohorts, and no merging 898/900 artistic results"}}
    if out:
        dest = Path(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return report
