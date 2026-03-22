# sae-cbm-eval

Incremental implementation of the SAE-CBM pruning pipeline from [SPEC.md](/Users/habichuela/Developer/projects/sae-cbm-eval/SPEC.md).

Current status:

- `scripts/00_verify_sae.py` is implemented.
- `scripts/01_extract_features.py` is implemented.
- It verifies the locked SAE checkpoint/config contract, checks the matching CLIP config, runs a dummy encoder pass, and writes `results/verify_sae.json`.
- It parses CUB-200 metadata, optionally downloads the dataset if missing, extracts CLS-token SAE features into memmapped `.npy` caches, and writes `features/extraction_meta.json`.

Setup:

```bash
uv sync
uv run python scripts/00_verify_sae.py
uv run python scripts/01_extract_features.py
```

Notes:

- `.env` is loaded automatically if present.
- Set `HUGGINGFACE_TOKEN` in `.env` or the shell if you want authenticated Hugging Face downloads.
