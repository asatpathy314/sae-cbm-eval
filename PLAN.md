# Plan: Fix Experimental Issues + Switch to Vanilla CLS-Only SAE

## Context

The current pipeline uses a vanilla all-tokens SAE (layer 11) with maxpool aggregation, which conflates patch-specialized and CLS-specialized features from the mixed dictionary. The fix is to use **vanilla CLS-only SAEs** — these have variable L0 per image (~424-493 active features at layers 6-7) and a full dictionary, making the pruning experiment meaningful.

**Research narrative:** "CLIP's CLS representation can be decomposed into N interpretable SAE features, but only M are needed for classification." The pruning curve is the core contribution.

**Primary SAE: Vanilla CLS-only, layer 6 or 7**
- Layer 6: 92% explained variance, CLS L0 ~424
- Layer 7: 88% explained variance, CLS L0 ~493
- Repo pattern: `Prisma-Multimodal/sae-vanilla-cls_only-layer_{N}-hook_resid_post` (exact name TBD — need to verify on HuggingFace)
- Dictionary size: TBD from config.json (likely 49,152 with x64 expansion on 768)
- Extraction: CLS token only → vanilla SAE encode → full sparse feature vector per image
- Pruning: core experiment, prune from ~500 active features down

**Optional ablation:** Vanilla all-patches SAE with mean-pooled patch features, same pruning pipeline. Compares CLS vs patch concept compressibility.

---

## Part A: SAE switch (requires verification + re-extraction)

### A0. Verify new SAE exists and inspect config
- [x] Download `config.json` from the Prisma CLS-only vanilla SAE repo on HuggingFace
- [x] Confirm exact repo ID and weight filename
- [x] Confirm `d_sae`, `d_in`, `activation_fn_str`, `architecture`
- [x] Confirm `hook_point_layer`, `layer_subtype`
- [x] Confirm whether `sae.encode()` returns the same `(sae_in, features)` tuple

This unblocks all other Part A changes.

### A1. Update constants
**File:** `src/sae_cbm_eval/constants.py`

Update after A0 verification:
```
SAE_REPO_ID → new repo ID
SAE_WEIGHT_FILENAME → from repo
EXPECTED_HOOK_LAYER → 6 or 7
EXPECTED_SAE_DIM → from config.json
EXPECTED_ACTIVATION_FN → "relu" (vanilla SAE, should stay the same)
EXPECTED_ARCHITECTURE → "standard" (vanilla, should stay the same)
```

- [x] Update `SAE_REPO_ID`
- [x] Update `SAE_WEIGHT_FILENAME`
- [x] Update `EXPECTED_HOOK_LAYER` (6)
- [x] Update `EXPECTED_SAE_DIM` (unchanged: 49,152)
- [x] Verify `EXPECTED_ACTIVATION_FN` and `EXPECTED_ARCHITECTURE` stay the same (relu, standard)

### A2. Revert extraction to CLS-only
**File:** `src/sae_cbm_eval/extraction.py`

Lines 91-95 (current maxpool) → CLS-only:
```python
h_cls = activations[:, 0, :]        # (B, 768)
_, features = sae.encode(h_cls)     # (B, d_sae)
```

- [x] Replace maxpool logic with CLS-only indexing

### A3. Update script 01 token_policy
**File:** `scripts/01_extract_features.py`

- [x] Import `TOKEN_POLICY_CLS_ONLY` instead of `TOKEN_POLICY_ALL_MAXPOOL`
- [x] Record `"token_policy": TOKEN_POLICY_CLS_ONLY` in extraction metadata

### A4. Update script 02 validation
**File:** `scripts/02_train_baseline.py`

- [x] Validate `TOKEN_POLICY_CLS_ONLY` instead of `TOKEN_POLICY_ALL_MAXPOOL`

### A5. Update script 00 verification
**File:** `scripts/00_verify_sae.py`

- [x] Verification checks compare against constants, so updating A1 handles most of it
- [x] Review `verify_sae_runtime` — no hardcoded layer references, clean

### A6. Full re-extraction required
- [ ] After A1-A5, re-run pipeline from script 00 onward. All cached features are invalidated (new SAE, new layer, CLS-only).

---

## Part B: Fix existing issues

### B1. Fix attribute path (scripts 11/12 broken)
**File:** `src/sae_cbm_eval/attributes.py`

`parse_attribute_names` expects `dataset_root / "attributes" / "attributes.txt"` but the file is at `data/attributes.txt` (one level up).

- [x] If standard CUB path doesn't exist, fall back to `dataset_root.parent / "attributes.txt"`
- [x] Also applied fallback to `parse_image_attribute_labels` via shared `_resolve_attributes_dir`

### B2. Certainty filtering
**Files:** `src/sae_cbm_eval/attributes.py`, `scripts/11_attribute_alignment.py`

- [x] `build_attribute_matrix`: add `min_certainty: int | None = None` param, filter `labels_df` before matrix loop
- [x] `11_attribute_alignment.py`: add `--min-certainty` CLI arg (default `3`)

### B3. Random baseline: dynamic feature counts
**File:** `scripts/07_random_baseline.py`

- [x] Auto-derive from `final_test_results.json` `n_features_final` when `--feature-counts` not passed
- [x] Fallback to `DEFAULT_FEATURE_COUNTS` if file missing

### B4. Label validation: LLM-based
**File:** `scripts/10_label_features.py`

- [x] Keep `is_refusal()` fast-path for "I cannot"/"sorry"
- [x] Replace `is_valid_label()` with `validate_label_llm(client, model, label) -> bool`
- [x] Prompt: "Does this label describe a visual property (color, shape, pattern, body part) rather than naming a specific bird species? Answer yes or no."
- [x] Record `"validation_method": "llm"` per entry

### B5. Semantic agreement: LLM judge
**Files:** `scripts/12_semantic_agreement.py`, `scripts/13_plot_all.py`

**12_semantic_agreement.py:**
- [x] Remove CLIP text encoding (`encode_texts_clip`, `torch`, `resolve_device`, `--device`, `--sim-threshold`)
- [x] Add OpenAI client init, `--model` (gpt-4o-mini), `--delay`
- [x] `judge_semantic_match(client, model, label, attribute) -> bool`
- [x] Output: `llm_agreement: bool` replaces `clip_sim: float`
- [x] `high_quality` = AUROC >= threshold AND `llm_agreement`

**13_plot_all.py:**
- [x] Replace AUROC-vs-clip_sim scatter with bar chart of high-quality fractions
- [x] Loop over all operating points, not just `[0]`

### B6. Alignment on test set (post-selection optimism)
**File:** `scripts/11_attribute_alignment.py`

- [x] Compute alignment on Z_test / A_test instead of Z_train / A_train
- [x] Feature indices from `final_test_results.json` were selected on Z_train, so Z_test is held-out
- [x] Add `--split` arg (default `test`)

### B7. Enforce --overwrite in scripts 09 and 13
**Files:** `scripts/09_collect_exemplars.py`, `scripts/13_plot_all.py`

- [x] Add `check_overwrite` calls before writing outputs

---

## Implementation order

```
Phase 1 — Verify new SAE:
  A0: Download and inspect CLS-only vanilla SAE config from HuggingFace

Phase 2 — SAE switch + re-extraction:
  A1 → A2 → A3 → A4 → A5
  Re-run: 00 → 01 → 02 → 03 → 04 → 05 → 06

Phase 3 — Bug fixes + measurement improvements:
  B1 (attr path) + B2 (certainty) + B7 (overwrite) — can do in parallel
  B3 (random counts) — after 05 produces final_test_results.json
  B4 (LLM validation) → B5 (LLM judge) → B6 (test-set alignment)
  Re-run: 07, 09, 10, 11, 12, 13
```

---

## Verification

| Change | Check |
|--------|-------|
| A0 | Config matches expectations (d_in=768, vanilla, relu, correct layer) |
| A2 | `extraction.py` uses `activations[:, 0, :]`, no maxpool |
| A1-A5 | `verify_sae.json` passes with new SAE, `extraction_meta.json` says `cls_only` |
| B1 | `parse_attribute_names` succeeds with `data/attributes.txt` location |
| B2 | Attribute matrix has fewer nonzero entries with min_certainty=3 |
| B3 | Feature counts auto-derived from `final_test_results.json` |
| B4 | Species labels rejected, visual labels accepted by LLM |
| B5 | `semantic_agreement.json` has `llm_agreement` booleans |
| B6 | Script 11 loads Z_test / A_test, not Z_train / A_train |
| B7 | Scripts 09/13 refuse to overwrite without flag |

---

## Critical files

```
src/sae_cbm_eval/constants.py           # A1
src/sae_cbm_eval/extraction.py          # A2
src/sae_cbm_eval/attributes.py          # B1, B2
scripts/00_verify_sae.py                # A5
scripts/01_extract_features.py          # A3
scripts/02_train_baseline.py            # A4
scripts/07_random_baseline.py           # B3
scripts/09_collect_exemplars.py         # B7
scripts/10_label_features.py            # B4
scripts/11_attribute_alignment.py       # B2, B6
scripts/12_semantic_agreement.py        # B5
scripts/13_plot_all.py                  # B5, B7
```
