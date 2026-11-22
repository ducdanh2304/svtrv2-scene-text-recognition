# SAGE-STR v4 — Kaggle run guide

Open `SAGE_v4_Kaggle.ipynb` in Kaggle (Code -> New Notebook -> File -> Import notebook); under Notebook Settings enable GPU and Internet for the first run. Run All. The pilot defaults to `MAX_LTB=1200` and `MAX_EXTERNAL=256`; set both to `0` for all available samples after the pilot. Check the `EVAL ONLY` and `Paired` tables printed in the notebook. Outputs are saved as **individual CSV files** at `/kaggle/working/SAGE_STR_V4_GPU_RESULTS/`. You do **not** need to download a ZIP to read the results. The notebook creates an optional ZIP at `/kaggle/working/SAGE_STR_V4_GPU_RESULTS.zip`, but does not force any download.

## What has changed

- Same frozen SVTRv2-T checkpoint and same proposal generator as v3.
- Context-aware orientation and recovery selection: compares candidate/reference/original text and evidence; uses both orientations when available.
- Deterministic ID-hash-based splits fixed independently of cohort size.
- Re-fits v3 control and v4 on the same split and evaluates them on the same images. Do not directly compare these new eval percentages against the old v3 pilot's **different evaluation split**.
- Batched score computation with parity tests against per-image selection.
- Explicit Multi-Oriented losses vs Unified and LTB character-error regression audit printed in output.

## Key outputs

- `evaluation_accuracy_cer_by_dataset.csv`: exact accuracy, CER, and edit counts on eval split.
- `evaluation_paired_win_loss.csv`: paired win/loss vs all controls, including v3-locked and Unified v1.
- `evaluation_predictions.csv`: per-image eval predictions and choices.
- `evaluation_per_candidate_trace.csv`: score and probability per candidate; ground-truth correctness only for posthoc analysis.
- `evaluation_multi_oriented_unified_wins_v4_losses.csv`: Multi-Oriented cases where Unified v1 succeeds but v4 fails.
- `evaluation_ltb_error_audit.csv`: LTB character-error differences.
- `ltb_eval_length_buckets.csv`, `ltb_eval_AAvg.csv`: three LTB length groups and aggregate.
- `all_descriptive_accuracy_cer.csv`: ALL rows including train/tune, **not independent evaluation**.
- `v4_locked_splits.csv`: per-image stable split assignments.
- `v4_policy.json`, `run_manifest.json`, `REPORT_SAGE_V4.md`: policy, source and hash metadata.

## Scientific caveat

The notebook was syntactically validated and the train-to-output pipeline passed an end-to-end **CPU dry-run on synthetic data**. It has **not** been tested with official SVTRv2-T on a Kaggle GPU in this environment; actual accuracy improvements are unverified. LTB/IIIT5K/CUTE80/Union14M have been repeatedly analyzed during previous development and thus do not constitute untouched external generalization tests.
