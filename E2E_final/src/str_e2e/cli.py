"""Commands: predict, evaluate, score-csv, verify-archive, check-assets."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .datasets import from_csv, from_lmdb, from_directory
from .evaluation import run_evaluation, evaluate_existing_predictions, file_sha256
from .system import STRSystem, load_config, MODES


def gather_assets(config: dict) -> dict:
    from .paths import ROOT
    assets = {}
    for key, val in config.get("assets", {}).items():
        if val is None:
            assets[key] = {"status": "MISSING"}
            continue
        p = Path(val).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file():
            assets[key] = {"path": str(p.resolve()), "sha256": file_sha256(p)}
        elif p.is_dir():
            assets[key] = {"path": str(p.resolve()), "status": "DIRECTORY"}
        else:
            assets[key] = {"path": str(p), "status": "NOT_FOUND"}
    return assets


def render(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True))


def build_parser():
    p = argparse.ArgumentParser(prog="str-e2e", description="Modular cropped scene-text recognizer and paired evaluation")
    sub = p.add_subparsers(dest="command", required=True)
    pred = sub.add_parser("predict", help="Run one word/line crop (NO ground truth)")
    pred.add_argument("--config", required=True)
    pred.add_argument("--image", required=True)
    pred.add_argument("--mode", choices=sorted(MODES))
    evalp = sub.add_parser("evaluate", help="Paired eval on CSV manifest or LMDB")
    evalp.add_argument("--config", required=True)
    source = evalp.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="CSV columns: id,image_path,gt")
    source.add_argument("--lmdb", help="OpenOCR LMDB directory with data.mdb")
    evalp.add_argument("--image-root")
    evalp.add_argument("--mode", choices=sorted(MODES))
    evalp.add_argument("--output", required=True)
    evalp.add_argument("--bootstrap", type=int, default=2000)
    evalp.add_argument("--save-every", type=int, default=25)
    score = sub.add_parser("score-csv", help="Recompute paired metrics from saved predictions")
    score.add_argument("--csv", required=True)
    score.add_argument("--baseline-column", default="baseline")
    score.add_argument("--method-column", default="method")
    score.add_argument("--gt-column", default="gt")
    score.add_argument("--id-column", default="id")
    score.add_argument("--bootstrap", type=int, default=2000)
    score.add_argument("--filter-column", help="Use with --filter-value to score a single dataset")
    score.add_argument("--filter-value")
    assets = sub.add_parser("check-assets", help="Report missing checkpoints/datasets without inventing results")
    assets.add_argument("--config", required=True)
    archive = sub.add_parser("verify-archive", help="Recompute existing SAGE pilot metrics and report historical rows")
    archive.add_argument("--out", help="Write evidence JSON to this path")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "score-csv":
        render(evaluate_existing_predictions(args.csv, baseline_column=args.baseline_column,
                                            method_column=args.method_column,
                                            gt_column=args.gt_column, id_column=args.id_column,
                                            bootstrap=args.bootstrap,
                                            filter_column=args.filter_column,
                                            filter_value=args.filter_value))
    elif args.command == "verify-archive":
        from .archive import verify_archive
        render(verify_archive(args.out))
    else:
        cfg = load_config(args.config)
        if args.command == "check-assets":
            render(gather_assets(cfg))
            return
        mode = args.mode or cfg.get("mode", "auto")
        provenance = {"config_sha256": file_sha256(args.config), "assets": gather_assets(cfg)}
        with STRSystem(cfg) as system:
            if args.command == "predict":
                from PIL import Image
                with Image.open(args.image) as inp:
                    baseline, ours = system.predict_pair(inp.convert("RGB"), mode=mode)
                render({"mode": mode, "baseline": baseline.as_dict(), "our_method": ours.as_dict(),
                        "caveat": "Prototype integrated routing is not an independently benchmarked performance claim"})
            elif args.command == "evaluate":
                samples = (from_csv(args.manifest, args.image_root) if args.manifest
                           else from_lmdb(args.lmdb))
                report = run_evaluation(samples, system, mode=mode, output_csv=args.output,
                                        bootstrap=args.bootstrap, save_every=args.save_every,
                                        provenance=provenance)
                render(report)

if __name__ == "__main__":
    main()
