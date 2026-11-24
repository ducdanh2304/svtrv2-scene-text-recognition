# E2E_final — frozen SVTRv2 OCR End-to-End V1

Final source + evaluation scripts corresponding to **4,740 processed images / 4,738 valid for scoring**. Results in `reports/`. No retraining or Gate V4 promotion.

- `src/str_e2e`: production system, routing and expert backends.
- `scripts/predict_auto.py`: sample Auto inference CLI.
- `scripts/evaluate_unified_e2e_v1.py`: original full benchmark runner and resume.
- `scripts/launch_e2e_clock_safe_v1.py`: CUDA startup guard.
- `scripts/finalize_e2e_benchmark_v1.py`: summarize complete JSONL journals.
- `tests`: frozen dispatch regression tests.
- `release_manifest.json`: SHA256 for packaged files and checkpoint assets.

**Weights, datasets and external OpenOCR repositories are intentionally NOT included.** Place them at the relative paths listed in `release_manifest.json`; obtain the exact external repo commits also listed there. Run `python scripts/setup_local.py` to create untracked absolute-path config files, then `PYTHONPATH=src python -m pytest -q tests/test_scope_dispatch.py tests/test_system_dispatch.py`. Use `PYTHONPATH=src python scripts/predict_auto.py PATH_TO_IMAGE` after assets are installed. Benchmark requires LMDB datasets and provenance mapping in the original expected directory structure.

Results (B0 -> E2E): IIIT5K 98.600% -> 99.300%; U14M 70.713% -> 82.739% (898 valid); Blur f008 96.500% -> 96.500%; f018 91.000% -> 93.500%; f030 70.500% -> 80.500%; LTB 33.333% -> 51.667%.

**Scientific limitations:** historical LTB development exposure; Motion blur variants share 200 image sources; independent training-overlap validation is still needed. Do not present an aggregate independent-test accuracy. Runtime config is regenerated per machine, so original results cannot be claimed reproduced on a new host until revalidated.
