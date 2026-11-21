# SCOPE-CTC V4 — CS406 Frozen Release

**Spatially Verified Cross-Scale CTC Segment Reconstruction for Long-Text Scene Text Recognition**

A **training-free inference-time pipeline** around the **official, frozen SVTRv2-T** checkpoint. The model's visual backbone, feature rearrangement, CTC head, and learned weights are **not modified**. The pipeline collects crop/scale CTC logits, proposes localized edits, verifies them with local CTC sequence likelihoods across spatially grouped views, and reconstructs the recognized line.

> **Release boundary:** This is the **CS406 V4 final artifact**. Do not mix the exploratory paper-track P1/P2 decoders or their results into this release. No independent long-text generalization or SOTA claim is made.

## Contents

```text
notebooks/SCOPE_CTC_V4_CS406_RunAll.ipynb   Reproducible Colab GPU Run All
src/                                       Exact V4 source from verified GPU ZIP
scripts/infer.py                           Standalone single-image inference (environment-dependent)
scripts/smoke_test.py                      CPU regression checks
scripts/verify_sources.py                  Verify frozen source module SHA-256 digests
docs/SCOPE_CTC_V4_CS406_Report.docx          Approved four-section CS406 technical report
docs/results/                             Aggregate metrics only, no raw image/GT/prediction dumps
docs/V4_PROVENANCE.json                    Official checkpoint, config, revision and method parameters
```

## Reproduce the frozen experiment (recommended)

1. Open `notebooks/SCOPE_CTC_V4_CS406_RunAll.ipynb` in **Google Colab** and enable a CUDA GPU.
2. Run all cells. The notebook fetches the official SVTRv2-T weights/config and the OpenOCR evaluation data. Its frozen SHA-256 checks must pass.
3. Optional: edit `ENABLE_DRIVE_CHECKPOINTS = False` if you do not want Google Drive checkpointing. The notebook otherwise writes to a dedicated folder in your Drive.
4. For complete published cohort results, retain `MAX_IMAGES = 0` and the default General sampling protocol. The notebook creates a results ZIP with raw prediction audits, accuracy, CER, paired wins/losses, and instrumented latency.

**Note:** The notebook depends on third-party data and checkpoint hosts; download availability, Colab package versions, and GPU environment are external dependencies. Keep notebook source and method thresholds frozen when reproducing the CS406 results.

## Run a single image (optional local API)

The code in `src/` is importable as **flat modules** using `PYTHONPATH=src`; it is not a trained model by itself. Clone OpenOCR at the pinned revision and supply the **official matching checkpoint/config files**, which are intentionally **not** included.

```bash
git clone https://github.com/Topdu/OpenOCR.git vendor/OpenOCR
git -C vendor/OpenOCR checkout 1ccfc6ee6161f7133b192e16af3a9265273997ba
python -m pip install -r requirements.txt
python scripts/infer.py --image /path/to/line.png --openocr-repo vendor/OpenOCR --config /path/to/config.yml --checkpoint /path/to/best.pth --device cuda
```

The standalone CLI is provided as a convenience wrapper around the frozen modules; **the verified experimental entrypoint remains the Colab notebook**. OpenOCR itself may need additional dependencies from its upstream `requirements.txt` and compatibility fixes contained in the Colab notebook. The CLI has not been validated on every GPU/OS combination.

## Verified results (development/reused benchmarks)

| Metric | Full-image SVTRv2-T, matched `full12` | SCOPE-CTC V4 |
|---|---:|---:|
| LTB extended, 4,789, exact-line accuracy | 33.85% | **56.13%** |
| LTB extended, A-Avg of 3 length buckets | 18.40% | **51.10%** |
| LTB extended, CER ↓ | 31.24% | **8.12%** |
| External Common/Union14M/OST, 21,903, micro accuracy | 81.975% | 81.984% |
| LTB instrumented mean pipeline latency | standalone matched model-forward only | **135.69 ms/image** |

**Interpretation:** The **full-pipeline gain** includes crop/stitch, routing, reference construction, CTC verification, and reconstruction. The incremental **SCOPE Core** ablation on LTB was 2,666 correct C1 reference lines → 2,688 SCOPE lines (**+22 lines**), with +16 attributable to the cross-scale component over the native-only decoder. Do not attribute the full 33.85% → 56.13% gain to segment verification alone.

### Full LTB breakdown

| Length (GT characters) | n | Matched SVTRv2-T | V4 |
|---|---:|---:|---:|
| 26–35 | 3,376 | 44.31% | 59.39% |
| 36–55 | 1,147 | 10.90% | 49.17% |
| ≥56 | 266 | 0.00% | 44.74% |

External benchmarks here are overwhelmingly short text. They test robustness against destroying ordinary STR accuracy, **not independent long-text generalization**. These benchmarks were inspected during iterative development; treat all significance estimates as exploratory. General is a fixed **2,000-item sample** (1,938 valid) rather than full 400,000 images. See `docs/results/` and the DOCX report.

### Latency caution

- V4 recorded **135.69 ms** mean on the LTB extended corpus, versus **355.35 ms** for V3 in a *different* GPU run; there is output parity on 26,692 evaluated images.
- This is **not** a controlled standalone-SVTRv2-T-vs-SCOPE end-to-end slowdown claim. Resize is **not proven** to be the main bottleneck; per-view neural forward and decoding both matter.

## Integrity and non-leakage

- Official frozen SVTRv2-T checkpoint SHA-256: `684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6`
- Official matching config SHA-256: `f1ab35a0f5e44d3c24c940cdf1a07404068ad9191bad9fe232ad4fcf38bbbe29`
- OpenOCR revision: `1ccfc6ee6161f7133b192e16af3a9265273997ba`
- Image inference API does **not** receive `gt`, test-set name, or evaluation correctness; labels are joined after predictions are sealed.
- Every source module under `src/` (except the notebook-derived `alex_e7_experiment.py` helper) is byte-for-byte copied from the successful V4 GPU result ZIP. Run `python scripts/verify_sources.py` to verify.

## Local development smoke test

```bash
python scripts/verify_sources.py
python scripts/smoke_test.py
```

## Dependencies and licensing

- **No checkpoint, benchmarks, tokens, or personal Drive files are committed.** The notebook downloads external resources separately.
- Upstream OpenOCR and benchmark data have their own respective licenses/terms. Review them before releasing the repository or redistributing derived assets.
- **The repository owner must select a license** after reviewing authorship and applicable third-party licenses. No license is assumed here.

## References

- SVTRv2, ICCV 2025 — https://openaccess.thecvf.com/content/ICCV2025/html/Du_SVTRv2_CTC_Beats_Encoder-Decoder_Models_in_Scene_Text_Recognition_ICCV_2025_paper.html
- OpenOCR reference repository — https://github.com/Topdu/OpenOCR
- SMTR, AAAI 2025 — https://ojs.aaai.org/index.php/AAAI/article/view/32285

---

**CS406 is frozen.** P1/P2 and any independent holdout work belong in a separate research branch/repository; they are not implied by this release.
