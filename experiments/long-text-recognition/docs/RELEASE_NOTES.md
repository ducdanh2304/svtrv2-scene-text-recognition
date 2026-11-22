# SCOPE-CTC V4 / CS406 Frozen Release

This repository is based on `SCOPE_V4_SPEED_OPTIMIZED_ALL_DATASETS_RESULTS.zip`, the **GPU-verified V4** experiment. The `src/` source files (except `alex_e7_experiment.py`, extracted from the matching V4 notebook for CPU parity tests) match the ZIP's `source_modules/` files byte-for-byte.

## Frozen properties
- SVTRv2-T architecture and weights unchanged.
- Original full-image baseline is `full12`.
- Core segment verification, two-group spatial support, and max edit budget unchanged.
- Baseline-preserving cover probe and accelerated Numba prefix beam enabled.
- 4,789 LTB + 21,903 valid external images evaluated; 26,692/26,692 V3→V4 prediction parity verified in historical GPU audit.

## Known limitations
- LTB and external benchmarks were examined during development; reported results are exploratory.
- No independent long-text holdout; no cross-model generalization claim.
- 38 conventional-text regressions observed in paired external benchmark evaluation.
- Mean LTB latency is GPU instrumented, not a matched standalone model end-to-end benchmark.
- An alternative one-witness decoder outperformed V4 on the reused LTB in later exploratory P1/P2 studies. The CS406 V4 release is intentionally **not updated** by that result.

## Release integrity
- Public repo excludes weights, image data, original label texts, prediction-level CSVs, and experiment ZIPs.
- The notebook contains the exact embedded modules and historical parity data needed for the course experiment.
- Results are aggregate tables only.
