# SAGE-STR v4 — Kaggle GPU exploratory development report

UTC: 2026-11-18T16:46:08.932345+00:00
OpenOCR commit: 1ccfc6ee6161f7133b192e16af3a9265273997ba
Checkpoint SHA256: 684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6
V4 code SHA256: f5ff65b03239c56cd2a7fb5ab39262bde8e10ee86dc25116e9713cf6f94e1c55
Fixed split salt: cs406-sage-v4-locked-splits-v1
MAX_LTB=1200, MAX_EXTERNAL=256; 0 means full cohort
Both v3 and v4 are re-fitted using identical fixed partitions and same GPU predictions.

## EVAL-ONLY (within-run disjoint, but historical benchmarks inspected)
| split   | dataset             | method          |   n |   exact_correct |   exact_accuracy_pct |   CER_pct |   total_edit_errors |
|:--------|:--------------------|:----------------|----:|----------------:|---------------------:|----------:|--------------------:|
| eval    | LTB                 | SVTRv2-T_single | 241 |              80 |               33.195 |    32.686 |                2694 |
| eval    | LTB                 | SVTRv2-T_v3     | 241 |             117 |               48.548 |    17.811 |                1468 |
| eval    | LTB                 | Unified_v1      | 241 |             124 |               51.452 |     9.706 |                 800 |
| eval    | LTB                 | Safe_reference  | 241 |             117 |               48.548 |    17.811 |                1468 |
| eval    | LTB                 | SAGE_v3_locked  | 241 |             127 |               52.697 |     9.330 |                 769 |
| eval    | LTB                 | SAGE_v4         | 241 |             127 |               52.697 |     9.706 |                 800 |
| eval    | u14m/multi_oriented | SVTRv2-T_single |  58 |              42 |               72.414 |     7.295 |                  24 |
| eval    | u14m/multi_oriented | SVTRv2-T_v3     |  58 |              42 |               72.414 |     7.295 |                  24 |
| eval    | u14m/multi_oriented | Unified_v1      |  58 |              47 |               81.034 |    10.638 |                  35 |
| eval    | u14m/multi_oriented | Safe_reference  |  58 |              42 |               72.414 |     7.295 |                  24 |
| eval    | u14m/multi_oriented | SAGE_v3_locked  |  58 |              49 |               84.483 |     4.255 |                  14 |
| eval    | u14m/multi_oriented | SAGE_v4         |  58 |              50 |               86.207 |     3.951 |                  13 |
| eval    | u14m/multi_words    | SVTRv2-T_single |  60 |              49 |               81.667 |     5.248 |                  36 |
| eval    | u14m/multi_words    | SVTRv2-T_v3     |  60 |              49 |               81.667 |     5.248 |                  36 |
| eval    | u14m/multi_words    | Unified_v1      |  60 |              46 |               76.667 |     6.122 |                  42 |
| eval    | u14m/multi_words    | Safe_reference  |  60 |              49 |               81.667 |     5.248 |                  36 |
| eval    | u14m/multi_words    | SAGE_v3_locked  |  60 |              50 |               83.333 |     4.810 |                  33 |
| eval    | u14m/multi_words    | SAGE_v4         |  60 |              51 |               85.000 |     4.810 |                  33 |
| eval    | evaluation/IIIT5k   | SVTRv2-T_single |  47 |              46 |               97.872 |     0.417 |                   1 |
| eval    | evaluation/IIIT5k   | SVTRv2-T_v3     |  47 |              46 |               97.872 |     0.417 |                   1 |
| eval    | evaluation/IIIT5k   | Unified_v1      |  47 |              38 |               80.851 |     5.417 |                  13 |
| eval    | evaluation/IIIT5k   | Safe_reference  |  47 |              46 |               97.872 |     0.417 |                   1 |
| eval    | evaluation/IIIT5k   | SAGE_v3_locked  |  47 |              46 |               97.872 |     0.417 |                   1 |
| eval    | evaluation/IIIT5k   | SAGE_v4         |  47 |              46 |               97.872 |     0.417 |                   1 |
| eval    | evaluation/CUTE80   | SVTRv2-T_single |  55 |              51 |               92.727 |     1.845 |                   5 |
| eval    | evaluation/CUTE80   | SVTRv2-T_v3     |  55 |              51 |               92.727 |     1.845 |                   5 |
| eval    | evaluation/CUTE80   | Unified_v1      |  55 |              33 |               60.000 |    11.070 |                  30 |
| eval    | evaluation/CUTE80   | Safe_reference  |  55 |              51 |               92.727 |     1.845 |                   5 |
| eval    | evaluation/CUTE80   | SAGE_v3_locked  |  55 |              52 |               94.545 |     1.476 |                   4 |
| eval    | evaluation/CUTE80   | SAGE_v4         |  55 |              53 |               96.364 |     0.738 |                   2 |

## Paired test
| split   | dataset             | comparison                 |   win |   loss |   net |   McNemar_exact_p |
|:--------|:--------------------|:---------------------------|------:|-------:|------:|------------------:|
| eval    | LTB                 | SAGE_v4 vs SVTRv2-T_single |    53 |      6 |    47 |             0.000 |
| eval    | LTB                 | SAGE_v4 vs SVTRv2-T_v3     |    12 |      2 |    10 |             0.013 |
| eval    | LTB                 | SAGE_v4 vs Unified_v1      |     4 |      1 |     3 |             0.375 |
| eval    | LTB                 | SAGE_v4 vs Safe_reference  |    12 |      2 |    10 |             0.013 |
| eval    | LTB                 | SAGE_v4 vs SAGE_v3_locked  |     0 |      0 |     0 |             1.000 |
| eval    | u14m/multi_oriented | SAGE_v4 vs SVTRv2-T_single |     8 |      0 |     8 |             0.008 |
| eval    | u14m/multi_oriented | SAGE_v4 vs SVTRv2-T_v3     |     8 |      0 |     8 |             0.008 |
| eval    | u14m/multi_oriented | SAGE_v4 vs Unified_v1      |     4 |      1 |     3 |             0.375 |
| eval    | u14m/multi_oriented | SAGE_v4 vs Safe_reference  |     8 |      0 |     8 |             0.008 |
| eval    | u14m/multi_oriented | SAGE_v4 vs SAGE_v3_locked  |     2 |      1 |     1 |             1.000 |
| eval    | u14m/multi_words    | SAGE_v4 vs SVTRv2-T_single |     2 |      0 |     2 |             0.500 |
| eval    | u14m/multi_words    | SAGE_v4 vs SVTRv2-T_v3     |     2 |      0 |     2 |             0.500 |
| eval    | u14m/multi_words    | SAGE_v4 vs Unified_v1      |     6 |      1 |     5 |             0.125 |
| eval    | u14m/multi_words    | SAGE_v4 vs Safe_reference  |     2 |      0 |     2 |             0.500 |
| eval    | u14m/multi_words    | SAGE_v4 vs SAGE_v3_locked  |     1 |      0 |     1 |             1.000 |
| eval    | evaluation/IIIT5k   | SAGE_v4 vs SVTRv2-T_single |     0 |      0 |     0 |             1.000 |
| eval    | evaluation/IIIT5k   | SAGE_v4 vs SVTRv2-T_v3     |     0 |      0 |     0 |             1.000 |
| eval    | evaluation/IIIT5k   | SAGE_v4 vs Unified_v1      |     8 |      0 |     8 |             0.008 |
| eval    | evaluation/IIIT5k   | SAGE_v4 vs Safe_reference  |     0 |      0 |     0 |             1.000 |
| eval    | evaluation/IIIT5k   | SAGE_v4 vs SAGE_v3_locked  |     0 |      0 |     0 |             1.000 |
| eval    | evaluation/CUTE80   | SAGE_v4 vs SVTRv2-T_single |     2 |      0 |     2 |             0.500 |
| eval    | evaluation/CUTE80   | SAGE_v4 vs SVTRv2-T_v3     |     2 |      0 |     2 |             0.500 |
| eval    | evaluation/CUTE80   | SAGE_v4 vs Unified_v1      |    20 |      0 |    20 |             0.000 |
| eval    | evaluation/CUTE80   | SAGE_v4 vs Safe_reference  |     2 |      0 |     2 |             0.500 |
| eval    | evaluation/CUTE80   | SAGE_v4 vs SAGE_v3_locked  |     1 |      0 |     1 |             1.000 |

## LTB buckets
| bucket   | method          |   n |   exact_correct |   exact_accuracy_pct |
|:---------|:----------------|----:|----------------:|---------------------:|
| 26-35    | SVTRv2-T_single | 172 |              72 |               41.860 |
| 26-35    | Unified_v1      | 172 |              90 |               52.326 |
| 26-35    | Safe_reference  | 172 |              85 |               49.419 |
| 26-35    | SAGE_v3_locked  | 172 |              92 |               53.488 |
| 26-35    | SAGE_v4         | 172 |              92 |               53.488 |
| 36-55    | SVTRv2-T_single |  55 |               8 |               14.545 |
| 36-55    | Unified_v1      |  55 |              31 |               56.364 |
| 36-55    | Safe_reference  |  55 |              29 |               52.727 |
| 36-55    | SAGE_v3_locked  |  55 |              32 |               58.182 |
| 36-55    | SAGE_v4         |  55 |              32 |               58.182 |
| 56+      | SVTRv2-T_single |  14 |               0 |                0.000 |
| 56+      | Unified_v1      |  14 |               3 |               21.429 |
| 56+      | Safe_reference  |  14 |               3 |               21.429 |
| 56+      | SAGE_v3_locked  |  14 |               3 |               21.429 |
| 56+      | SAGE_v4         |  14 |               3 |               21.429 |

## V4 safety audit
{'passed': True, 'n': 306, 'summary': {'wins': 20, 'harms': 2, 'edit_gain': 366, 'promoted': 45, 'n': 306}, 'by_source': {'LTB': {'wins': 12, 'harms': 2, 'edit_gain': 348, 'promoted': 37, 'n': 168}, 'u14m/multi_oriented': {'wins': 6, 'harms': 0, 'edit_gain': 13, 'promoted': 6, 'n': 45}, 'u14m/multi_words': {'wins': 2, 'harms': 0, 'edit_gain': 5, 'promoted': 2, 'n': 28}, 'evaluation/IIIT5k': {'wins': 0, 'harms': 0, 'edit_gain': 0, 'promoted': 0, 'n': 25}, 'evaluation/CUTE80': {'wins': 0, 'harms': 0, 'edit_gain': 0, 'promoted': 0, 'n': 40}}, 'note': 'Development audit; no OOD guarantee or conformal claim'}

## Limitations
- This is a development evaluation. Previously studied benchmark images are NOT independent held-out generalization evidence.
- No threshold, candidate selection or feature is permitted to read eval ground-truth at inference.
- The stable hash split changes the rows compared with the historical v3 pilot; only within-notebook paired v3_locked vs v4 comparison is valid.
- Safety audit is empirical only, NOT guaranteed OOD non-inferiority or conformal validity.
- The official SVTRv2-T backbone is FROZEN; only lightweight calibration/selector is learned.
- Candidate pool is the same as v3; gains are due to the decision layer, not new OCR models.
- To make final conclusions, evaluate a truly untouched long-text and rotated-word benchmark.