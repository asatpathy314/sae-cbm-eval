# Next Steps

## Purpose

This document records the most promising follow-up directions after the completed
midterm pipeline, incorporating literature checks and the current empirical
results.

Current empirical status:

- The SAE feature space is highly compressible for CUB classification.
- Iterative pruning retains strong validation and test accuracy at very small
  feature budgets.
- This supports a "compact predictive core" story.
- It does **not** yet establish interpretability, faithfulness, or attribute
  alignment.

The main goal of the next phase is to convert the current compressibility result
into a stronger claim about whether the retained SAE features correspond to
meaningful bird attributes rather than arbitrary predictive structure.

## Primary Directions

### 1. Attribute Alignment of the Pruned Core

This is the highest-priority direction and the clearest candidate contribution.

Core question:

- Do the features that survive pruning align with CUB-200's 312 expert
  attributes?

Why this matters:

- DN-CBM evaluates naming quality on object-level datasets but does not measure
  alignment to expert attribute annotations.
- M-CBM uses MLLM labeling but does not run systematic attribute-alignment
  evaluation.
- Existing vision SAE work does not appear to evaluate a pruned sparse core
  against a dataset with expert attributes in this way.

Important framing:

- MLLM labeling is **not** the contribution by itself.
- The contribution is the **alignment evaluation**.
- Labeling is a tool used to interpret the retained features.

Recommended experiments:

1. For each retained feature at selected sparsity levels, collect top-activating
   images.
2. Score alignment between feature activation and each CUB attribute.
3. Record the best-matched attribute per feature.
4. Report distribution of best-match scores across retained features.
5. Add shuffled-label and random-attribute baselines.
6. Optionally compare MLLM-generated labels to best-matched attribute names as a
   secondary semantic check.

Deliverable:

- A quantitative answer to whether the pruned core corresponds to recognizable
  bird attributes.

### 2. Matched-Sparsity Baselines

This is essential for interpreting the pruning result.

Core question:

- Are the retained features genuinely special, or would almost any subset of the
  same size work?

Most important baseline:

- Random subsets of SAE features at matched `k`.

Why this matters:

- If random `k=100` SAE subsets perform similarly to the pruned set, the current
  pruning result is much less interesting.
- If random subsets fail badly while the pruned set remains strong, that is
  strong evidence that the selected features are meaningfully informative.

Required baselines:

1. Random SAE feature subsets at matched `k` values such as `139`, `103`, `52`,
   and optionally `23`.
2. Raw CLIP dimension selection at matched `k`, without the SAE basis.

Raw CLIP baseline question:

- Is the SAE decomposition actually helping, or could similar sparsity/performance
  be achieved by selecting dimensions directly from the original CLIP
  representation?

Deliverable:

- A fair statement of whether pruning in SAE space is better than chance and
  better than pruning directly in raw CLIP space.

### 3. NEC Comparison

This should be framed as a comparison or validation study rather than a new
faithfulness theory.

Core question:

- How does the pruning-derived effective feature count compare with VLG-CBM's NEC
  estimate on the same model?

Why this matters:

- NEC is an existing way to summarize effective concept count.
- The pruning curve is a more direct empirical measurement of how many features
  are actually needed for performance.
- Agreement would support NEC as a useful proxy.
- Disagreement would reveal a gap between NEC and empirical compressibility.

Recommended outcome framing:

- "We compare NEC to pruning-derived effective feature count."
- Do **not** pitch this as an entirely new notion of faithfulness by itself.

Deliverable:

- A direct empirical comparison between a published metric and the pruning curve.

## Secondary Directions

### 4. Stability Across Train/Validation Seeds

This is a strengthening experiment rather than a central contribution.

Core question:

- Do the same features survive across multiple splits and random seeds?

Why this matters:

- Stable selected features suggest a real reusable core.
- Unstable subsets would imply redundancy without clear semantic uniqueness.

Recommended metrics:

- Overlap / Jaccard similarity of retained feature sets.
- Variance in `k_delta`.
- Variance in validation accuracy at matched `k`.

### 5. Raw CLIP Pruning as a Standalone Comparison

This overlaps with the matched-sparsity baseline section but is worth calling
out explicitly.

Core question:

- Does the SAE basis provide a better sparse prediction substrate than the raw
  CLIP feature basis?

If time is limited, this can be implemented as part of the matched-sparsity
baseline suite rather than a separate full section.

## Demoted or Cut

### 6. CLS vs. Patch Tokens

Demote heavily or cut from the main paper.

Reason:

- Patch-level SAE work on vision transformers is already well covered by
  PatchSAE and related literature.
- A broad CLS-vs-patch comparison would mostly replicate existing work.

Only keep if framed narrowly as:

- whether patch-level extraction changes which features survive pruning on
  CUB-200.

Even then, it should be a secondary ablation, not a main direction.

### 7. Feature Ablations / Simple Causal Tests

Demote to optional or supplementary material.

Reason:

- CaFE and SAEV already investigate causal interventions on vision SAE features.
- A simple single-feature ablation study here would be weaker than those methods.

Use only if:

- it helps explain a small set of especially important retained features.

### 8. Expansion-Factor or Layer Sweeps

Demote to appendix or future work.

Reason:

- Existing literature already analyzes expansion ratio / layer effects in related
  settings.
- This is expensive and likely not the highest-value use of time right now.

## Recommended Order of Execution

### Phase 1: Establish whether the current pruning result is meaningful

1. Random SAE subsets at matched `k`.
2. Raw CLIP matched-sparsity baseline.
3. Raw CLIP pruning baseline if feasible.

This phase determines whether the current result is actually about informative
feature selection or just generic redundancy.

### Phase 2: Test interpretability of the retained core

1. Top-activating image collection for retained features.
2. Attribute alignment against CUB annotations.
3. Label retained features with an MLLM only as an interpretive aid.
4. Compare labels to best-matched attribute names where useful.

This phase is the most likely source of a novel result.

### Phase 3: Metric comparison and robustness

1. NEC comparison.
2. Stability across seeds.

These experiments strengthen the story and make the claims more defensible.

## Target Paper Claim

If the next experiments succeed, the paper's main claim should look something
like this:

- A highly overcomplete SAE concept space can be pruned to a small predictive
  core on CUB-200.
- That core is not matched by random subsets or raw CLIP baselines at the same
  feature budget.
- The retained features show measurable alignment with expert-annotated bird
  attributes.

Until those experiments are done, the current defensible claim is weaker:

- The SAE feature space is highly compressible for fine-grained classification,
  but interpretability and faithfulness remain unverified.
