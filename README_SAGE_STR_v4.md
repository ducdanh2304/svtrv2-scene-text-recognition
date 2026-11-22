# SAGE-STR v4 — Contextual Risk-Aware Adaptive Inference for STR

**CS406 · Frozen SVTRv2-T backbone · Lightweight learned selector · Experimental research release**

> **Status:** The code corresponds to the **Kaggle SAGE-STR v4 GPU pilot** (2,224 inference images, 461 evaluation images). This is **not a proven SOTA method, a guarantee of zero regressions, or untouched out-of-distribution evaluation**.

SAGE-STR studies when an alternative prediction should replace a reference STR prediction. It uses the same **frozen SVTRv2-T** recognizer to generate geometry/crop, orientation and resolution candidates and a **Contextual Dual-Arbiter** to estimate expected recognition gain and harm.

- [Run on Kaggle](experiments/long_text/sage_str_v4/README.md)
- [Complete Vietnamese research report](reports/long_text/sage_str_v4/BAO_CAO_TONG_HOP_DAY_DU.md)
- [Detailed method (Vietnamese)](reports/long_text/sage_str_v4/02_PHUONG_PHAP_CHI_TIET.md)
- [Historical experiments and negative results (Vietnamese)](reports/long_text/sage_str_v4/01_HANH_TRINH_THU_NGHIEM.md)
- [Pilot metrics and limitations (Vietnamese)](reports/long_text/sage_str_v4/03_KET_QUA_VA_GIOI_HAN.md)
- [Reproducibility protocol](reports/long_text/sage_str_v4/04_TAI_LAP_VA_KIEM_CHUNG.md)
- [GitHub push guide](docs/long_text/SAGE_STR_V4_GITHUB_PUSH.md)

## Pilot evaluation snapshot

| Dataset | Eval n | SVTRv2-T original | Unified v1 | SAGE-STR v4 |
|---|---:|---:|---:|---:|
| LTB | 241 | 33.20% | 51.45% | **52.70%** |
| Union14M Multi-Oriented | 58 | 72.41% | 81.03% | **86.21%** |
| Union14M Multi-Words | 60 | 81.67% | 76.67% | **85.00%** |
| IIIT5K | 47 | 97.87% | 80.85% | **97.87%** |
| CUTE80 | 55 | 92.73% | 60.00% | **96.36%** |

Numbers above are **development pilot exact-line accuracy**, not benchmark-wide or independent generalization. LTB CER: Unified v1 **9.706%**, SAGE v4 **9.706%**, SAGE v3 locked **9.330%** (the latter is better on CER); LTB 56+ is only **3/14**. Exact-line improvement v4 vs Unified on LTB and Multi-Oriented is small (paired McNemar **p=0.375** per set).

## Attribution

SVTRv2/OpenOCR is third-party work by Du et al. (ICCV 2025). This repo includes **the authors' local experimental glue code, not OpenOCR pretrained model weights or raw benchmark images**. The notebook clones official OpenOCR at a pinned commit. Datasets and pretrained weights must be used subject to their respective licenses/terms. See [attribution](reports/long_text/sage_str_v4/05_NGUON_VA_TRICH_DAN.md).

## Folder placement

This archive is **additive**: extract it in the root of the existing `svtrv2-scene-text-recognition` repo. It places code under `experiments/long_text/sage_str_v4/` and reports under `reports/long_text/sage_str_v4/`, leaving Unified v1 untouched.
