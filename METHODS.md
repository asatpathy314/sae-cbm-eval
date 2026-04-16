# Methods

## Overview

We evaluate whether sparse autoencoder (SAE) features extracted from a CLIP vision transformer can serve as an interpretable concept bottleneck for fine-grained image classification. Our pipeline decomposes the CLIP CLS representation into sparse SAE features, trains a linear classifier on those features, iteratively prunes uninformative features, and then assesses the interpretability of the retained features through multimodal LLM labeling and alignment with human-annotated visual attributes.

## 1. Dataset

We use CUB-200-2011 (Wah et al., 2011), a fine-grained bird classification benchmark containing 11,788 images across 200 species. The dataset provides an official train/test split (5,994 training images, 5,794 test images) with approximately 30 images per class per split. CUB-200-2011 also provides 312 binary visual attributes per image (e.g., `wing_color::red`, `bill_shape::hooked`) annotated by human workers with certainty ratings from 1 (guessing) to 4 (definite). We use these ground-truth attributes as a reference for evaluating feature interpretability.

## 2. Feature Extraction

### 2.1 Vision Encoder

We use CLIP ViT-B/32 (Radford et al., 2021), specifically the DataComp-XL variant (`laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K`), implemented via the vit-prisma library's hooked transformer interface. Each 224x224 input image is tokenized into 49 patch tokens plus one CLS token. We extract intermediate activations at layer 6 (of 12), specifically at the residual stream post-attention hook point (`blocks.6.hook_resid_post`), yielding a 768-dimensional representation for the CLS token.

### 2.2 Sparse Autoencoder

We apply a pre-trained vanilla (standard architecture) sparse autoencoder from the Prisma-Multimodal project (`imagenet-sweep-vanilla-x64-CLS_6`). This SAE was trained exclusively on CLS token activations from ImageNet using a 64x expansion factor, producing a 49,152-dimensional sparse feature space from the 768-dimensional input. The SAE uses a ReLU activation function with layer normalization on inputs, and was trained with ghost gradients and an L1 sparsity penalty (coefficient 1.05e-08).

For each image, we extract the CLS token activation at layer 6, encode it through the SAE, and obtain a 49,152-dimensional sparse feature vector:

$$\mathbf{z} = \text{ReLU}(\mathbf{W}_{enc} \cdot \text{LayerNorm}(\mathbf{h}_{CLS}) + \mathbf{b}_{enc})$$

where $\mathbf{h}_{CLS} \in \mathbb{R}^{768}$ is the CLS activation and $\mathbf{z} \in \mathbb{R}^{49152}$ is the resulting sparse feature vector. The SAE achieves 92% explained variance on the CLS representation with an average L0 (number of nonzero features) of approximately 431 per image.

### 2.3 Feature Statistics

Across the training set, we observe that 45,880 of the 49,152 dictionary features are "dead" (never activate for any training image), leaving approximately 3,272 live features. Per-image density is approximately 0.88%, consistent with the expected L0 of ~431 active features per image. All feature values are non-negative by construction (ReLU activation).

## 3. Classification

### 3.1 Baseline Classifier

We train a multinomial logistic regression classifier (L2-regularized, lbfgs solver) on the full 49,152-dimensional SAE feature vectors to establish a baseline classification accuracy. The 5,994 training images are further split into a fitting set (80%, ~4,795 images) and a validation set (20%, ~1,199 images) via stratified sampling, preserving class balance.

### 3.2 Regularization Selection

The regularization strength $C$ is selected via 5-fold cross-validation over the fitting set, searching over $C \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}$. The $C$ value maximizing mean cross-validated accuracy is used for all subsequent experiments.

## 4. Iterative Feature Pruning

The core experiment is an iterative pruning procedure that removes the least important features at each round, producing a pruning curve that maps feature count to classification accuracy.

### 4.1 Importance Score

At each pruning round, we fit the logistic regression classifier on the current feature subset and compute per-feature importance scores:

$$\text{importance}_j = \|\mathbf{w}_j\|_2 \cdot \sigma_j$$

where $\|\mathbf{w}_j\|_2$ is the L2 norm of the weight vector for feature $j$ across all 200 classes, and $\sigma_j$ is the standard deviation of feature $j$ across the training set. This product captures both how sensitive the classifier is to a feature and how much that feature actually varies across images. A feature with large weights but near-constant activation contributes little to discrimination and is pruned early.

### 4.2 Pruning Schedule

Starting from all live features, we iteratively:
1. Fit the logistic regression on the current feature subset (using the selected $C$)
2. Compute importance scores for all remaining features
3. Remove the bottom 10% of features by importance
4. Record the validation accuracy at the current feature count

This continues until 5 or fewer features remain (up to 1,000 rounds). The result is a pruning curve: a sequence of (feature count, validation accuracy) pairs.

### 4.3 Operating Points

From the pruning curve, we derive a $k$-$\delta$ table: for each accuracy drop threshold $\delta \in \{0.01, 0.02, 0.05, 0.10\}$, we find the minimum number of features $k_\delta$ such that validation accuracy remains within $\delta$ of the full-feature baseline. These operating points define the feature subsets used in downstream interpretability analysis.

### 4.4 Final Test Evaluation

At each operating point, we prune to the target $k_\delta$ features using the full training set, retrain the classifier on all 5,994 training images (restricted to the selected features), and evaluate on the held-out 5,794 test images. This provides an honest estimate of generalization performance. The list of retained feature indices at each operating point is recorded for interpretability analysis.

## 5. Baseline Comparisons

### 5.1 L1 Baseline

We train L1-regularized logistic regression (saga solver) at several regularization strengths ($C \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1\}$), which naturally produces sparse solutions. For each $C$, we report the number of nonzero weight features and validation accuracy. This compares the built-in L1 sparsity mechanism against our iterative importance-based pruning. Features are scaled by variance (not mean-centered) to preserve the non-negative sparsity structure of SAE features.

### 5.2 Random Feature Baseline

For each operating point's feature count $k$, we sample 50 random subsets of $k$ features (drawn from the pool of live features) and train a classifier on each subset. This yields a distribution of random-subset accuracies (mean, standard deviation, min, max) at each $k$. The gap between pruned accuracy and random-subset accuracy quantifies the value of importance-based selection over arbitrary selection.

### 5.3 Raw CLIP Baseline

We extract 768-dimensional raw CLIP CLS embeddings (without SAE encoding) for all images and apply the same iterative pruning procedure to the raw dimensions. This baseline tests whether the SAE's learned dictionary provides a more compressible basis than the entangled raw representation. If the SAE pruning curve dominates the raw CLIP pruning curve, it demonstrates that the SAE has reorganized the representation into more individually selectable dimensions.

## 6. Interpretability Analysis

### 6.1 Exemplar Collection

For each retained feature at each operating point, we identify the 15 highest-activating and 15 lowest-activating training images (with class diversity constraints: at most 3 images per species). These are assembled into visual montages showing contrast between images that strongly activate the feature and those that do not. The montages serve as visual evidence for what each feature detects.

### 6.2 Multimodal LLM Labeling

Each montage is presented to a multimodal language model (GPT-5.4-mini) with the prompt:

> "You are shown two sets of bird images. TOP ROW(S): These images strongly activate a particular visual feature. BOTTOM ROW(S): These images do NOT activate this feature. Describe the visual property that is present in the top images but absent in the bottom images. Use a short phrase (3-10 words). Focus only on visible properties: color, texture, shape, pattern, or body part. Do NOT name any bird species."

Labels are validated in two stages:
1. **Refusal detection** (heuristic): Labels containing "I cannot", "I can't", or "sorry" are flagged as refusals and marked invalid.
2. **Visual property validation** (LLM): Remaining labels are checked by the same model with the prompt: "Does this label describe a visual property (color, shape, pattern, texture, or body part) rather than naming a specific bird species? Answer yes or no." Labels receiving a "no" are marked invalid.

### 6.3 Feature-Attribute Alignment

We measure how well each retained SAE feature aligns with CUB's 312 ground-truth binary attributes by computing the area under the ROC curve (AUROC). For each (feature, attribute) pair, the feature's activation values serve as the predictor and the binary attribute annotation as the label. We filter attribute annotations to those with certainty $\geq$ 3 (annotator was reasonably confident) to reduce label noise.

Alignment is computed on the **test set** to avoid post-selection optimism: since feature indices were selected by the pruning procedure on the training set, the test set provides a held-out evaluation of whether the alignment generalizes.

For each retained feature, we record its best-matched attribute (highest AUROC) and the corresponding AUROC value. We report the mean and median best-match AUROC across all retained features, and the fraction exceeding a threshold of 0.65 (indicating meaningful alignment). A permutation baseline (shuffling attribute labels 100 times) establishes the chance-level AUROC distribution.

### 6.4 Semantic Agreement

We assess whether the MLLM-generated labels agree with the statistically best-matched CUB attributes. For each (label, best-matched attribute) pair, a language model (GPT-5.4-mini) judges whether the two descriptions refer to the same or very similar visual property:

> "Do these two descriptions refer to the same or very similar visual property of a bird? Answer yes or no.
> Description A: [MLLM label]
> Description B: [CUB attribute name]"

A feature is classified as **high-quality** if both conditions hold:
1. Its best-match AUROC $\geq$ 0.65 (statistical alignment)
2. The LLM judge confirms semantic agreement between the MLLM label and the best-matched attribute

The fraction of high-quality features is reported at each operating point as the headline interpretability metric. This joint criterion requires that a feature is both statistically predictive of a human-annotated attribute and that the MLLM independently describes the same visual property when shown exemplar images.

## 7. Reproducibility

All random operations use a fixed seed (42). Feature extraction uses float32 precision throughout. SAE checkpoint integrity is verified via SHA256 hashes of both the config and weight files before any extraction. The train/validation split indices, cross-validation results, pruning trajectories, and all intermediate results are serialized to JSON for full reproducibility. Hardware configuration and software versions (PyTorch, vit-prisma, scikit-learn) are recorded in each run manifest.
