#!/usr/bin/env python3
"""Frozen STRSystem auto benchmark: Official B0 vs E2E."""

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import lmdb
import torch
import yaml
from PIL import Image

from str_e2e.system import STRSystem
from str_e2e.routing import image_route

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/unified_v4/e2e_benchmark_v1"
CFG = ROOT / "configs/system.local.yaml"
MAP = ROOT / (
    "outputs/unified_v4/provenance_official/"
    "official_rgb_mapping.csv"
)

DATA = {
    "iiit5k": (
        "datasets/iiit5k/evaluation/IIIT5k", 3000
    ),
    "u14m_artistic": (
        "datasets/union_artistic/u14m/artistic", 900
    ),
    "blur_f008": (
        "datasets/router_motion_dev/blur_f008", 200
    ),
    "blur_f018": (
        "datasets/router_motion_dev/blur_f018", 200
    ),
    "blur_f030": (
        "datasets/router_motion_dev/blur_f030", 200
    ),
    "ltb_clean_eval": (
        "datasets/ltb_unpacked/ltb/long_lmdb", 4789
    ),
}

NOTES = {
    "iiit5k": "Evaluation; upstream overlap not certified",
    "u14m_artistic": "Research-exposed Artistic benchmark",
    "blur_f008": "Development blur; shared source groups",
    "blur_f018": "Development blur; shared source groups",
    "blur_f030": "Development blur; shared source groups",
    "ltb_clean_eval": "Historical clean eval, not independent",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def norm(text):
    return "".join(
        c for c in str(text).casefold()
        if c.isalnum()
    )


def dist(a, b):
    if len(a) < len(b):
        a, b = b, a

    row = list(range(len(b) + 1))

    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(
                nxt[-1] + 1,
                row[j] + 1,
                row[j-1] + (x != y)
            ))
        row = nxt

    return row[-1]


def indices_for(name):
    if name != "ltb_clean_eval":
        return list(range(DATA[name][1]))

    with MAP.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 4789

    ix = {
        int(r["index_0based"]): r
        for r in rows
    }
    assert len(ix) == 4789

    dev = {
        r["rgb_sha256"]
        for r in rows
        if r["split"] in (
            "train", "calibration", "tune"
        )
    }

    eval_all = {
        i for i, r in ix.items()
        if r["split"] == "eval"
    }

    keep = sorted(
        i for i in eval_all
        if ix[i]["rgb_sha256"] not in dev
    )

    assert len(eval_all) == 241
    assert len(keep) == 240
    assert eval_all - set(keep) == {2927}

    return keep


def read_item(txn, i):
    key = f"{i+1:09d}".encode()

    raw = txn.get(b"image-" + key)
    gt = txn.get(b"label-" + key)

    if raw is None or gt is None:
        raise RuntimeError(
            f"Missing image/label index {i}"
        )

    with Image.open(io.BytesIO(raw)) as im:
        return im.convert("RGB"), gt.decode("utf-8")


def sync(device):
    if (
        str(device).startswith("cuda")
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize()


def conf(x):
    try:
        value = float(x)
        return (
            value if math.isfinite(value)
            else None
        )
    except (TypeError, ValueError):
        return None


def summary(name, rows):
    valid = [r for r in rows if r["scored"]]
    n = len(valid)

    a = sum(r["b0_ok"] for r in valid)
    b = sum(r["e2e_ok"] for r in valid)

    gained = sum(
        not r["b0_ok"] and r["e2e_ok"]
        for r in valid
    )
    harmed = sum(
        r["b0_ok"] and not r["e2e_ok"]
        for r in valid
    )

    assert gained - harmed == b - a

    chars = sum(r["gt_chars"] for r in valid)

    by_route = {}

    for route in sorted({
        r["route"] for r in rows
    }):
        sub = [
            r for r in valid
            if r["route"] == route
        ]

        by_route[route] = {
            "n": len(sub),
            "official_b0_ok": sum(
                r["b0_ok"] for r in sub
            ),
            "route_baseline_ok": sum(
                r["route_b0_ok"] for r in sub
            ),
            "e2e_ok": sum(
                r["e2e_ok"] for r in sub
            ),
        }

    return {
        "status": "PASS",
        "cohort": name,
        "total": len(rows),
        "scored": n,
        "source_groups": len({
            r["source_group_id"] for r in rows
        }),
        "routes": dict(Counter(
            r["route"] for r in rows
        )),
        "by_route": by_route,
        "b0_correct": a,
        "e2e_correct": b,
        "gained": gained,
        "harmed": harmed,
        "changed": sum(
            r["b0"] != r["e2e"]
            for r in rows
        ),
        "b0_accuracy_pct": (
            round(100 * a / n, 4) if n else None
        ),
        "e2e_accuracy_pct": (
            round(100 * b / n, 4) if n else None
        ),
        "gain_pp": (
            round(100 * (b-a) / n, 4)
            if n else None
        ),
        "b0_cer": (
            round(sum(
                r["b0_edits"] for r in valid
            ) / chars, 6)
            if chars else None
        ),
        "e2e_cer": (
            round(sum(
                r["e2e_edits"] for r in valid
            ) / chars, 6)
            if chars else None
        ),
        "official_median_ms_includes_warmup": (
            round(statistics.median(
                r["official_ms"] for r in rows
            ), 2)
            if rows else None
        ),
        "e2e_pair_median_ms_includes_warmup": (
            round(statistics.median(
                r["e2e_pair_ms"] for r in rows
            ), 2)
            if rows else None
        ),
        "scoring": (
            "Unicode alnum casefold exact-match; "
            "pooled normalized character error rate"
        ),
        "note": NOTES[name],
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--cohorts", nargs="+",
        choices=DATA,
        default=list(DATA)
    )
    p.add_argument("--tag", required=True)
    p.add_argument(
        "--limit", type=int, default=0,
        help="0=all, N=first N eligible samples"
    )

    args = p.parse_args()

    if (
        not re.fullmatch(
            r"[A-Za-z0-9_-]{1,40}", args.tag
        )
        or args.limit < 0
    ):
        p.error("Invalid tag/limit")

    if len(args.cohorts) != len(set(args.cohorts)):
        p.error("Duplicated cohort")

    cfg = yaml.safe_load(
        CFG.read_text(encoding="utf-8")
    )

    assert isinstance(cfg, dict)
    assert isinstance(
        cfg.get("routing", {}), dict
    )

    if cfg.get("longtext", {}).get(
        "strict_frozen_assets", True
    ) is False:
        raise RuntimeError(
            "Frozen SCOPE verification must remain enabled"
        )

    device = cfg.get("device", "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    asset_hashes = {
        key: sha(cfg["assets"][key])
        for key in (
            "official_checkpoint",
            "r3_checkpoint",
            "waterec_checkpoint",
        )
    }

    source_files = (
        "system.py",
        "routing.py",
        "backends/official.py",
        "backends/motion.py",
        "backends/artistic.py",
        "backends/longtext.py",
        "backends/waterec.py",
    )

    source_hashes = {
        s: sha(ROOT / "src/str_e2e" / s)
        for s in source_files
    }

    reports = []

    for name in args.cohorts:
        db = ROOT / DATA[name][0]

        if not (db / "data.mdb").is_file():
            raise FileNotFoundError(db / "data.mdb")

        wanted = indices_for(name)

        if args.limit:
            wanted = wanted[:args.limit]

        journal = OUT / f"{args.tag}_{name}.jsonl"
        lock = OUT / f"{args.tag}_{name}_identity.json"
        report_path = OUT / f"{args.tag}_{name}_summary.json"

        identity = {
            "dataset": name,
            "selected_indices": wanted,
            "db_path": str(db),
            "db_size": (
                db / "data.mdb"
            ).stat().st_size,
            "db_mtime_ns": (
                db / "data.mdb"
            ).stat().st_mtime_ns,
            "config_sha256": sha(CFG),
            "script_sha256": sha(__file__),
            "checkpoint_hashes": asset_hashes,
            "source_hashes": source_hashes,
            "device": device,
            "mapping_sha256": (
                sha(MAP)
                if name == "ltb_clean_eval"
                else None
            ),
        }

        if lock.exists():
            old = json.loads(
                lock.read_text(encoding="utf-8")
            )
            if old != identity:
                raise RuntimeError(
                    "Run identity changed; use a new --tag"
                )
        else:
            if journal.exists():
                raise RuntimeError(
                    "Journal exists without identity lock"
                )
            lock.write_text(
                json.dumps(identity, indent=2) + "\n",
                encoding="utf-8"
            )

        done = {}

        if journal.exists():
            for line in journal.read_text(
                encoding="utf-8"
            ).splitlines():
                row = json.loads(line)
                i = row["index_0based"]

                if i in done or i not in wanted:
                    raise RuntimeError(
                        "Duplicate/unexpected journal row"
                    )

                done[i] = row

        print(
            f"\n=== {name}: "
            f"{len(done)}/{len(wanted)} completed ===",
            flush=True
        )

        if len(done) != len(wanted):
            env = lmdb.open(
                str(db),
                readonly=True,
                lock=False,
                readahead=False,
                subdir=True
            )

            system = STRSystem(cfg)

            try:
                with (
                    env.begin(write=False) as txn,
                    journal.open(
                        "a", encoding="utf-8"
                    ) as f
                ):
                    raw_n = txn.get(b"num-samples")
                    assert raw_n is not None
                    n = int(raw_n.decode())

                    assert n == DATA[name][1], (
                        name, n
                    )

                    for step, i in enumerate(
                        wanted, 1
                    ):
                        if i in done:
                            continue

                        try:
                            image, gt = read_item(txn, i)

                            route, _ = image_route(
                                image,
                                **cfg.get("routing", {})
                            )

                            # Independent Official B0 inference.
                            sync(device)
                            t0 = time.perf_counter()

                            official = system._backend(
                                "official"
                            ).predict(image)

                            sync(device)
                            official_ms = (
                                time.perf_counter() - t0
                            ) * 1000

                            # Real End-to-End AUTO inference.
                            sync(device)
                            t0 = time.perf_counter()

                            branch_base, method = (
                                system.predict_pair(
                                    image, mode="auto"
                                )
                            )

                            sync(device)
                            pair_ms = (
                                time.perf_counter() - t0
                            ) * 1000

                            assert (
                                method.route
                                == branch_base.route
                                == route
                            )
                            assert (
                                method.details.get("branch")
                                == route
                            )

                            w, h = image.size

                            digest = hashlib.sha256(
                                f"{w}x{h}:".encode()
                                + image.tobytes()
                            ).hexdigest()

                            ng, nb, ne, nr = map(
                                norm,
                                (
                                    gt,
                                    official.text,
                                    method.text,
                                    branch_base.text
                                )
                            )

                            valid = bool(ng)

                            if name.startswith("blur_"):
                                group = f"iiit_val:{i:06d}"
                            elif name == "ltb_clean_eval":
                                group = f"ltb:{digest}"
                            else:
                                group = f"{name}:{i:06d}"

                            rec = {
                                "cohort": name,
                                "index_0based": i,
                                "source_group_id": group,
                                "rgb_sha256": digest,
                                "gt": gt,
                                "width": w,
                                "height": h,
                                "route": route,
                                "official_method": official.method,
                                "route_baseline_method": (
                                    branch_base.method
                                ),
                                "e2e_method": method.method,
                                "selected_component": (
                                    method.details.get("selected")
                                ),
                                "scope_active": (
                                    method.details.get(
                                        "safe_scope_active"
                                    )
                                ),
                                "b0": official.text,
                                "route_baseline": branch_base.text,
                                "e2e": method.text,
                                "b0_confidence": conf(
                                    official.confidence
                                ),
                                "e2e_confidence": conf(
                                    method.confidence
                                ),
                                "scored": valid,
                                "b0_ok": (
                                    int(nb == ng)
                                    if valid else None
                                ),
                                "route_b0_ok": (
                                    int(nr == ng)
                                    if valid else None
                                ),
                                "e2e_ok": (
                                    int(ne == ng)
                                    if valid else None
                                ),
                                "gt_chars": (
                                    len(ng) if valid else 0
                                ),
                                "b0_edits": (
                                    dist(nb, ng)
                                    if valid else 0
                                ),
                                "e2e_edits": (
                                    dist(ne, ng)
                                    if valid else 0
                                ),
                                "official_ms": round(
                                    official_ms, 2
                                ),
                                "e2e_pair_ms": round(
                                    pair_ms, 2
                                ),
                            }

                            f.write(
                                json.dumps(
                                    rec, ensure_ascii=False
                                ) + "\n"
                            )
                            f.flush()
                            os.fsync(f.fileno())

                            done[i] = rec

                            if (
                                step <= 3
                                or step % 25 == 0
                                or step == len(wanted)
                            ):
                                print(
                                    f"[{step}/{len(wanted)}] "
                                    f"{name} route={route} "
                                    f"B0={rec['b0_ok']} "
                                    f"E2E={rec['e2e_ok']} "
                                    f"pair_ms={pair_ms:.0f}",
                                    flush=True
                                )

                        except Exception:
                            print(
                                f"FAILED {name} index={i}; "
                                "safe to resume",
                                flush=True
                            )
                            raise

            finally:
                system.close()
                env.close()
                del system
                gc.collect()

                if (
                    str(device).startswith("cuda")
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

        rows = [done[i] for i in wanted]

        report = summary(name, rows)
        report.update({
            "journal": str(journal),
            "identity": str(lock),
        })

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False
            ) + "\n",
            encoding="utf-8"
        )

        reports.append(report)

        print(
            f"PASS {name}: "
            f"n={report['scored']} "
            f"B0={report['b0_accuracy_pct']}% "
            f"E2E={report['e2e_accuracy_pct']}% "
            f"gain={report['gain_pp']}pp "
            f"CER={report['e2e_cer']} "
            f"routes={report['routes']}",
            flush=True
        )

    suite = OUT / f"{args.tag}_suite_summary.json"

    suite.write_text(
        json.dumps({
            "status": "PASS",
            "cohorts": reports,
            "warning": (
                "Do not pool research-exposed cohorts "
                "into an independent score. "
                "Latency includes paired inference "
                "and warmup."
            )
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    print("\n=== UNIFIED E2E BENCHMARK SUMMARY ===")

    for r in reports:
        print(
            f"{r['cohort']}: "
            f"{r['e2e_correct']}/{r['scored']}, "
            f"B0={r['b0_correct']}/{r['scored']}, "
            f"gains={r['gained']}, "
            f"harms={r['harmed']}, "
            f"routes={r['routes']}",
            flush=True
        )

    print("REPORT:", suite)
    print("UNIFIED E2E BENCHMARK: PASS")


if __name__ == "__main__":
    main()
