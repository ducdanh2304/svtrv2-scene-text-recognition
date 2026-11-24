"""Evaluation harness: same images, labels, normalization and paired rows."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Iterable
from .metrics import evaluate_rows, normalize
from .types import Sample

FIELDS = ["id", "gt", "baseline", "method", "baseline_name", "method_name", "route",
          "baseline_confidence", "method_confidence", "baseline_ms", "method_ms",
          "image_sha256_rgb", "details"]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rows_to_csv(rows: list[dict]) -> bytes:
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError("Predictions CSV has an unexpected column order")
        return list(reader)


def run_evaluation(samples: Iterable[Sample], system, *, mode: str,
                   output_csv: str | Path, output_json: str | Path | None = None,
                   save_every: int = 25, bootstrap: int = 2000,
                   provenance: dict | None = None) -> dict:
    """Run or resume. GT is only accessed AFTER predict_pair returns."""
    output = Path(output_csv)
    final = Path(output_json) if output_json else output.with_suffix(".report.json")
    lock_file = output.with_suffix(output.suffix + ".contract.json")
    contract = {"mode": mode, "provenance": provenance or {},
                "normalization": "lowercase-ASCII-alphanumeric", "format": "str_e2e_eval_v1"}
    contract_bytes = (json.dumps(contract, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if lock_file.is_file():
        if lock_file.read_bytes() != contract_bytes:
            raise RuntimeError("Existing evaluation output was generated using a different contract")
    else:
        if output.is_file():
            raise RuntimeError("Existing predictions have no integrity contract. Refusing unsafe resume")
        atomic_write(lock_file, contract_bytes)
    rows = read_rows(output) if output.is_file() else []
    seen = set()
    n = 0
    for sample in samples:
        n += 1
        if sample.sample_id in seen:
            raise ValueError(f"Repeated sample ID: {sample.sample_id}")
        seen.add(sample.sample_id)
        if sample.gt is None or not normalize(sample.gt):
            raise ValueError(f"GT missing or empty after normalization for ID {sample.sample_id}")
        if n <= len(rows):
            previous = rows[n - 1]
            if previous["id"] != sample.sample_id or previous["gt"] != sample.gt:
                raise RuntimeError(f"Resume cohort mismatch at row {n}")
            # Also verify actual RGB image bytes against prior run, not just GT/ID.
            image = sample.load_image()
            if hashlib.sha256(image.tobytes()).hexdigest() != previous["image_sha256_rgb"]:
                raise RuntimeError(f"Resume image-content SHA mismatch at ID {sample.sample_id}")
            continue
        image = sample.load_image()
        # Content-only RGB hash is invariant to PNG/JPEG container formatting.
        im_sha = hashlib.sha256(image.tobytes()).hexdigest()
        baseline, ours = system.predict_pair(image, mode=mode)
        rows.append({"id": sample.sample_id, "gt": sample.gt, "baseline": baseline.text,
                     "method": ours.text, "baseline_name": baseline.method,
                     "method_name": ours.method, "route": ours.route,
                     "baseline_confidence": "" if baseline.confidence is None else baseline.confidence,
                     "method_confidence": "" if ours.confidence is None else ours.confidence,
                     "baseline_ms": "" if baseline.latency_ms is None else f"{baseline.latency_ms:.5f}",
                     "method_ms": "" if ours.latency_ms is None else f"{ours.latency_ms:.5f}",
                     "image_sha256_rgb": im_sha,
                     "details": json.dumps(ours.details, ensure_ascii=False, sort_keys=True)})
        if len(rows) % save_every == 0:
            atomic_write(output, rows_to_csv(rows))
            print(f"EVAL saved {len(rows)} images", flush=True)
    if len(rows) != n:
        raise RuntimeError(f"Existing prediction count {len(rows)} exceeds input count {n}")
    if not rows:
        raise ValueError("No samples to evaluate")
    atomic_write(output, rows_to_csv(rows))
    result = evaluate_rows(rows, baseline="baseline", method="method", bootstraps=bootstrap)
    branches = {}
    for name in sorted(set(r["route"] for r in rows)):
        branch_rows = [r for r in rows if r["route"] == name]
        branches[name] = evaluate_rows(branch_rows, bootstraps=bootstrap)
    result.update({"status": "FRESH_EVAL_COMPLETED", "dataset_image_count": n,
                   "mode": mode, "branches": branches,
                   "predictions_sha256": file_sha256(output),
                   "contract_sha256": file_sha256(lock_file), "provenance": provenance or {},
                   "warning": "Router and combined system must be evaluated on untouched test data; archived metrics are not merged-run metrics."})
    data = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if final.is_file() and final.read_bytes() != data:
        raise RuntimeError("Evaluation report already exists with a different result")
    atomic_write(final, data)
    return result


def evaluate_existing_predictions(csv_path: str | Path, *, baseline_column: str,
                                   method_column: str, gt_column: str="gt",
                                   id_column: str="id", bootstrap: int=2000,
                                   filter_column: str | None = None,
                                   filter_value: str | None = None) -> dict:
    rows = []
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as f:
        for i, source in enumerate(csv.DictReader(f), 1):
            if filter_column is not None and source.get(filter_column) != filter_value:
                continue
            if not all(column in source for column in (baseline_column, method_column, gt_column)):
                raise ValueError("Required prediction columns missing")
            rows.append({"id": source.get(id_column) or str(i),
                         "gt": source[gt_column], "baseline": source[baseline_column],
                         "method": source[method_column]})
    result = evaluate_rows(rows, bootstraps=bootstrap)
    result["status"] = "RECALCULATED_FROM_EXISTING_PAIRED_PREDICTIONS"
    result["source_predictions_sha256"] = file_sha256(csv_path)
    return result
