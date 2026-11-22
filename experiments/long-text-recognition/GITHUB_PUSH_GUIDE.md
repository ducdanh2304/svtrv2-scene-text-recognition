# Publish the frozen CS406 V4 release to GitHub

## Before publishing

- Read `README.md` and `LICENSE_POLICY.md`. Choose and add a repository license **after checking upstream conditions**.
- Review the report and metric summary. Do not upload checkpoint binaries, image datasets, personal Google Drive contents, or prediction-level data.
- Verify sources and tests locally:

```bash
python scripts/verify_sources.py
python scripts/smoke_test.py
```

## Create a repository

Create an **empty** repository in your own GitHub account, e.g. `scope-ctc-cs406`. Do **not** initialize the remote with a README or `.gitignore` because these are already included here. Choose visibility (private/public) yourself.

Open a terminal inside the extracted `SCOPE_CTC_CS406_GitHub_Ready` folder and run:

```bash
git init
git add .
git commit -m "Release frozen SCOPE-CTC V4 for CS406"
git branch -M main
```

Then add the actual remote URL copied from your GitHub repository page:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Do not use the placeholder URL literally; substitute your real repository's link.

## Recommended release tag

After confirming all files are correct, you may tag the immutable course version:

```bash
git tag -a cs406-v4.0 -m "CS406 frozen SCOPE-CTC V4"
git push origin cs406-v4.0
```

Use **a separate branch or a separate repository** for paper-track experiments; do not modify this release tag. Keep the original V4 checkpoint hash and benchmark protocol unchanged.
