"""Consistent STR word accuracy, CER and paired comparisons (no GT in inference)."""
from __future__ import annotations
import math
import re
from collections import defaultdict

import numpy as np


def normalize(text: str) -> str:
    """Follow archived STR evaluation: lowercase, keep ASCII a-z0-9."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        cur = [i]
        for j, char_b in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (char_a != char_b)))
        prev = cur
    return prev[-1]


def score_pairs(pairs: list[tuple[str, str]]) -> dict:
    """Input raw (GT, prediction) pairs. Empty normalized GT is invalid."""
    if not pairs:
        raise ValueError("Cannot score an empty evaluation cohort")
    data = [(normalize(gt), normalize(pred)) for gt, pred in pairs]
    if any(not gt for gt, _ in data):
        raise ValueError("Empty normalized GT; refusing to change evaluation denominator")
    chars = sum(len(gt) for gt, _ in data)
    correct = sum(gt == pred for gt, pred in data)
    edits = sum(edit_distance(gt, pred) for gt, pred in data)
    return {"n": len(data), "correct": correct, "word_accuracy": correct / len(data),
            "word_accuracy_pct": 100 * correct / len(data), "character_edits": edits,
            "gt_characters": chars, "cer": edits / chars, "cer_pct": 100 * edits / chars}


def exact_mcnemar(gains: int, harms: int) -> float:
    """Exact two-sided McNemar test, numerically stable for large n."""
    if gains < 0 or harms < 0:
        raise ValueError("gains and harms must be nonnegative")

    n = gains + harms
    if n == 0:
        return 1.0

    k = min(gains, harms)

    # log P(X=k), X ~ Binomial(n, 0.5)
    log_p_k = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        - n * math.log(2.0)
    )

    # Sum P(X <= k) relative to P(X=k).
    # This avoids enormous binomial coefficients and 2**n.
    relative = 1.0
    tail_sum = 1.0

    for i in range(k, 0, -1):
        relative *= i / (n - i + 1)
        tail_sum += relative

    return min(1.0, 2.0 * math.exp(log_p_k) * tail_sum)


def compare_paired(rows: list[tuple[str, str, str]], *, seed: int = 20261122,
                   bootstraps: int = 2000) -> dict:
    """Triples of raw GT, baseline, candidate from identical IDs and order."""
    if not rows:
        raise ValueError("No paired observations")
    if bootstraps < 1:
        raise ValueError("bootstraps must be positive")
    gt = [normalize(r[0]) for r in rows]
    baseline = [normalize(r[1]) for r in rows]
    candidate = [normalize(r[2]) for r in rows]
    if any(not x for x in gt):
        raise ValueError("At least one GT normalizes to empty")
    b = np.asarray([int(x == y) for x, y in zip(gt, baseline)], dtype=np.int16)
    c = np.asarray([int(x == y) for x, y in zip(gt, candidate)], dtype=np.int16)
    be = np.asarray([edit_distance(x, y) for x, y in zip(gt, baseline)], dtype=np.int32)
    ce = np.asarray([edit_distance(x, y) for x, y in zip(gt, candidate)], dtype=np.int32)
    lens = np.asarray([len(x) for x in gt], dtype=np.int32)
    gains = int(((b == 0) & (c == 1)).sum())
    harms = int(((b == 1) & (c == 0)).sum())
    rng = np.random.default_rng(seed)
    dwa, dcer = [], []
    # Streaming bootstrap avoids allocating large (#reps x #samples) arrays.
    for _ in range(bootstraps):
        ix = rng.integers(0, len(rows), len(rows))
        dwa.append(float(100 * (c[ix] - b[ix]).mean()))
        dcer.append(float(100 * (ce[ix] - be[ix]).sum() / lens[ix].sum()))
    return {"n": len(rows), "gained": gains, "harmed": harms,
            "both_correct": int(((b == 1) & (c == 1)).sum()),
            "both_wrong": int(((b == 0) & (c == 0)).sum()),
            "delta_wa_pp": float(100 * (c.mean() - b.mean())),
            "delta_cer_pp": float(100 * (ce.sum() - be.sum()) / lens.sum()),
            "delta_wa_ci95_pp": [float(v) for v in np.quantile(dwa, [0.025, 0.975])],
            "delta_cer_ci95_pp": [float(v) for v in np.quantile(dcer, [0.025, 0.975])],
            "mcnemar_exact_two_sided_p": exact_mcnemar(gains, harms),
            "bootstrap_seed": seed, "bootstrap_replicates": bootstraps}


def evaluate_rows(rows: list[dict], baseline: str = "baseline", method: str = "method",
                  bootstraps: int = 2000) -> dict:
    """Rows carry id, gt, baseline, method; enforce nonmissing and unique IDs."""
    if not rows:
        raise ValueError("No rows")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs in evaluation")
    for row in rows:
        for key in ("gt", baseline, method):
            if row.get(key) is None:
                raise ValueError(f"Missing column {key} for ID {row['id']}")
    pairs_base = [(r["gt"], r[baseline]) for r in rows]
    pairs_method = [(r["gt"], r[method]) for r in rows]
    return {"baseline": score_pairs(pairs_base), "method": score_pairs(pairs_method),
            "paired": compare_paired([(r["gt"], r[baseline], r[method]) for r in rows],
                                     bootstraps=bootstraps),
            "metric_normalization": "lowercase ASCII alphanumeric [a-z0-9]",
            "source_cohort_n": len(ids)}
