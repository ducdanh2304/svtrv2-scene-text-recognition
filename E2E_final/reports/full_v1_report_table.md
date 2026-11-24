# Unified End-to-End Benchmark V1

## Benchmark results

| Benchmark | Processed | Scored | B0 Acc (%) | E2E Acc (%) | Δ (pp) | B0 CER | E2E CER | Gains | Harms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iiit5k | 3000 | 3000 | 98.600 | 99.300 | +0.700 | 0.00360 | 0.00229 | 29 | 8 |
| u14m_artistic | 900 | 898 | 70.713 | 82.739 | +12.027 | 0.07508 | 0.04211 | 127 | 19 |
| blur_f008 | 200 | 200 | 96.500 | 96.500 | +0.000 | 0.01005 | 0.01507 | 2 | 2 |
| blur_f018 | 200 | 200 | 91.000 | 93.500 | +2.500 | 0.03015 | 0.02111 | 8 | 3 |
| blur_f030 | 200 | 200 | 70.500 | 80.500 | +10.000 | 0.12362 | 0.06533 | 23 | 3 |
| ltb_clean_eval | 240 | 240 | 33.333 | 51.667 | +18.333 | 0.32465 | 0.10144 | 50 | 6 |

## Reporting notes

- Total processed: 4,740 images.
- Valid for scoring: 4,738 images.
- U14M: 900 processed, 898 valid, 2 excluded.
- Exact match uses casefolded alphanumeric text.
- CER uses pooled character edits per cohort.
- Motion variants share 200 source-image groups.
- LTB clean eval has historical development exposure.
- Research/development exposure and potential overlap must be disclosed.
- Do not describe all cohorts as independent tests.
- Do not pool cohorts into a primary overall accuracy.
