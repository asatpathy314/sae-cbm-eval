# sae-cbm-eval

Incremental implementation of the SAE-CBM pruning pipeline from [SPEC.md](/Users/habichuela/Developer/projects/sae-cbm-eval/SPEC.md).

Current status:

- `scripts/00_verify_sae.py` is implemented.
- `scripts/01_extract_features.py` is implemented.
- `scripts/02_train_baseline.py` is implemented.
- `scripts/03_run_pruning.py` is implemented.
- `scripts/04_l1_baseline.py` is implemented.
- It verifies the locked SAE checkpoint/config contract, checks the matching CLIP config, runs a dummy encoder pass, and writes `results/verify_sae.json`.
- It parses CUB-200 metadata, optionally downloads the dataset if missing, extracts CLS-token SAE features into memmapped `.npy` caches, and writes `features/extraction_meta.json`.
- It creates the reproducible train/validation split, cross-validates the L2 multinomial logistic baseline, and writes `results/split_indices.json`, `results/cv_results.json`, and `results/baseline_summary.json`.
- It computes `features/sigma_train.npy`, runs the iterative pruning schedule, derives `k_delta`, and writes the pruning artifacts under `results/`.
- It runs the standardized L1 `saga` baseline and writes `results/l1_baseline.json`.

Setup:

```bash
uv sync
uv run python scripts/00_verify_sae.py
uv run python scripts/01_extract_features.py
uv run python scripts/02_train_baseline.py
uv run python scripts/03_run_pruning.py
uv run python scripts/04_l1_baseline.py
```

Notes:

- `.env` is loaded automatically if present.
- Set `HUGGINGFACE_TOKEN` in `.env` or the shell if you want authenticated Hugging Face downloads.
