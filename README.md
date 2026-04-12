# SAE-CBM Evaluation Pipeline

Evaluates whether Sparse Autoencoder (SAE) features can serve as a concept bottleneck for fine-grained classification on CUB-200-2011. The pipeline extracts SAE activations from a CLIP ViT-B/32, iteratively prunes features by importance, and compares against several baselines.

## Prerequisites

- Python 3.10+
- A CUDA GPU (scripts 00, 01, 08, 12 use PyTorch and benefit significantly from GPU)
- ~5 GB disk for the CUB-200-2011 dataset + cached features
- A HuggingFace token (for downloading the SAE checkpoint)
- An OpenAI API key (for script 10 only -- MLLM labeling with GPT-4o-mini)

## Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd sae-cbm-eval

# Install dependencies (uses uv, install it first if needed: https://docs.astral.sh/uv/)
uv sync

# Create .env from the example and fill in your tokens
cp .env.example .env
# Edit .env:
#   HUGGINGFACE_TOKEN=hf_...
#   OPENAI_API_KEY=sk-...
```

## Running the Pipeline

Run each script in order. Every script is self-contained and writes its outputs to `results/`, `features/`, `figures/`, or `exemplars/`. If a script has already been run, it will skip by default -- pass `--overwrite` to re-run.

All scripts accept `--verbose` for detailed logging.

```bash
# Stage 1: Verify SAE checkpoint and extract features (GPU)
uv run python scripts/00_verify_sae.py
uv run python scripts/01_extract_features.py

# Stage 2: Train baseline classifier and find best regularization
uv run python scripts/02_train_baseline.py

# Stage 3: Iterative pruning
uv run python scripts/03_run_pruning.py

# Stage 4: L1 sparsity baseline
uv run python scripts/04_l1_baseline.py

# Stage 5: Final test-set evaluation at selected operating points
uv run python scripts/05_final_test.py

# Stage 6: Plot initial pruning curve
uv run python scripts/06_plot_results.py

# Stage 7: Random feature subset baseline (validates pruning is non-trivial)
uv run python scripts/07_random_baseline.py

# Stage 8: Raw CLIP dimension baseline (GPU -- extracts CLIP features, then prunes)
uv run python scripts/08_clip_baseline.py

# Stage 9: Collect exemplar images for MLLM labeling
uv run python scripts/09_collect_exemplars.py

# Stage 10: Label features via GPT-4o-mini (requires OPENAI_API_KEY in .env)
uv run python scripts/10_label_features.py

# Stage 11: Feature-attribute alignment (AUROC analysis)
uv run python scripts/11_attribute_alignment.py

# Stage 12: Semantic agreement between MLLM labels and CUB attributes (GPU -- uses CLIP text encoder)
uv run python scripts/12_semantic_agreement.py

# Stage 13: Generate all final paper figures
uv run python scripts/13_plot_all.py
```

### GPU Notes

- Scripts **00, 01** load the CLIP model + SAE and do a forward pass over all ~12k images. This is the most GPU-intensive step.
- Script **08** extracts raw CLIP features (similar to 01 but without the SAE). Also GPU-heavy.
- Script **12** encodes text labels with CLIP's text encoder. Light GPU usage.
- All other scripts are CPU-only (logistic regression, numpy operations, plotting).

### API Cost Note

Script **10** calls GPT-4o-mini to label each retained feature. At the default operating point (`delta=0.01`), this is ~100-150 API calls with image inputs. Cost should be well under $1.

## Output Structure

```
features/           Cached feature matrices (.npy) and extraction metadata
results/            JSON files from each stage (accuracies, pruning curves, etc.)
figures/            PDF figures for the paper
exemplars/          Montage images and prompts used for MLLM labeling
```

## Interpreting Results

### Key output files

| File | What it tells you |
|------|-------------------|
| `results/baseline_summary.json` | Full-feature accuracy and best regularization C |
| `results/pruning_results.json` | Accuracy at each pruning round (the main pruning curve) |
| `results/pruning_summary.json` | k_delta values -- how many features needed at each accuracy-drop threshold |
| `results/l1_baseline.json` | L1-sparse logistic regression accuracy + number of nonzero features |
| `results/final_test_results.json` | Test-set accuracy at selected operating points |
| `results/random_baseline.json` | Random subset accuracy distribution (should be much worse than pruned) |
| `results/clip_baseline.json` | Raw CLIP pruning curve (should degrade faster than SAE pruning) |
| `results/attribute_alignment.json` | Per-feature best-match AUROC against CUB attributes |
| `results/semantic_agreement.json` | Joint AUROC + CLIP-text-similarity for labeled features |

### Key figures

| Figure | What to look for |
|--------|-----------------|
| `figures/pruning_curve.pdf` | Basic pruning curve from script 06 |
| `figures/pruning_curve_full.pdf` | Enhanced curve with random baseline, CLIP baseline, and L1 overlaid |
| `figures/alignment_histogram.pdf` | Distribution of best-match AUROCs -- features above 0.5 are better than chance |
| `figures/semantic_agreement_scatter.pdf` | AUROC vs CLIP similarity -- top-right quadrant = high-quality interpretable features |

### What "good" results look like

1. **Pruning curve**: Accuracy should stay relatively flat as features are removed, then drop sharply. A long plateau means the SAE has many redundant features and pruning finds a compact, informative subset.
2. **k_delta**: At `delta=0.01` (1% accuracy drop), we hope to retain only a small fraction of the original 49,152 features (e.g., ~100-200).
3. **Random baseline**: Random subsets of matched size should perform significantly worse than pruned subsets. This confirms pruning selects specifically useful features.
4. **CLIP baseline**: Raw CLIP dimensions (768) should degrade faster under pruning than SAE features, showing the SAE basis is better suited for concept-level selection.
5. **Alignment**: Many retained features should have best-match AUROC > 0.6 against CUB attributes, meaning they correlate with human-defined visual properties.
6. **Semantic agreement**: Features where both AUROC is high AND the MLLM label matches the attribute name (high CLIP text similarity) are "high-quality interpretable concepts."
