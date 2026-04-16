# Implementation Specification v3: SAE-CBM Feature Pruning on CUB-200
# Locked model/SAE pair: CLIP ViT-B/32 DataComp + Prisma vanilla SAE x64

## 0. Scope

This spec covers the midterm pipeline:

1. Download and parse CUB-200-2011.
2. Extract SAE features from CLIP ViT-B/32 using a confirmed Prisma checkpoint.
3. Train a linear classifier with cross-validated regularization.
4. Run iterative pruning and produce an accuracy-vs-feature-count curve.
5. Run one robustness comparison: an L1 baseline.
6. Evaluate selected operating points on the held-out test set.

Deliverables:

- Cached feature matrices and labels.
- A reproducible train/val split file.
- Baseline validation accuracy.
- A pruning curve and `k_delta` table.
- L1 baseline overlay.
- Final test-set accuracy at selected operating points.

MLLM labeling and concept alignment are still out of scope for this spec.

---

## 1. Locked Choices And Key Numbers

These are now confirmed and should not be changed unless `00_verify_sae.py`
fails and the replacement is documented explicitly.

| Item | Value | Status |
|------|-------|--------|
| CLIP backbone | `CLIP-ViT-B-32-DataComp.XL-s13B-b90K` | Confirmed |
| Hooked model string | `open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K` | Confirmed |
| open_clip fallback | `create_model_and_transforms("ViT-B-32", pretrained="datacomp_xl_s13b_b90k")` | Confirmed |
| SAE repo | `Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-8e-05` | Confirmed |
| SAE weight file | `n_images_2600058.pt` | Confirmed |
| SAE type | vanilla ReLU SAE | Confirmed |
| Hook point | `hook_resid_post`, layer 11 | Confirmed |
| Input dimension | 768 | Confirmed |
| SAE dimension | 49,152 | Confirmed |
| Context size | 50 tokens = 1 CLS + 49 patches | Confirmed |
| SAE training regime | trained on all tokens | Confirmed |
| Model checkpoint license | Apache-2.0 | Confirmed |
| Reported explained variance | 79.3% | Confirmed |
| Reported mean L0 | about 287.6 active features/token | Confirmed |

Primary image-level feature design choice for this project:

- Use the CLS token only for the main pipeline.
- Reason: we need one feature vector per image, and CLS-only matches the
  standard classification convention for CLIP-style image representations.
- Caveat: the SAE was trained on all tokens, so this is an experimental design
  choice, not a theorem. Document it in the report.

Do not use OpenAI CLIP ViT-B/32 for extraction. That would produce a model
mismatch with the confirmed SAE checkpoint.

---

## 2. Environment, Reproducibility, And Dataset

### 2.1 Hardware

- GPU: 8 GB VRAM minimum for extraction, 12 GB preferred.
- Disk: about 2.5 GB for cached `float32` features plus metadata and plots.
- RAM: 32 GB recommended, 16 GB bare minimum.
- Reason for the RAM recommendation: sklearn may materialize `float64` working
  arrays during multinomial logistic regression, especially in early pruning
  rounds with 49,152 features.

### 2.2 Python Environment

Use `uv` for environment management and locking.

```txt
python>=3.10
torch>=2.1
vit-prisma
huggingface-hub
open-clip-torch
scikit-learn>=1.3
scipy
numpy
pandas
matplotlib
Pillow
torchvision
tqdm
```

Project files:

- `pyproject.toml`
- `uv.lock`

After the first successful environment setup, lock exact versions in
`uv.lock`. Do not rely on ad hoc `pip freeze` snapshots as the primary record
of what was used.

### 2.3 Reproducibility Requirements

All scripts must:

- Set `np.random.seed(42)`.
- Set `torch.manual_seed(42)`.
- Use `random_state=42` for sklearn components.
- Record GPU model and CUDA version in `results/hardware.txt`.
- Save split indices to `results/split_indices.json`.
- Save extraction metadata to `features/extraction_meta.json`.
- Save row-to-image mapping so cached feature rows can be traced back to source
  images without ambiguity.
- Write a per-run manifest to `results/run_manifests/<script_name>.json`.

Each run manifest should include at least:

- script name
- timestamp
- python version
- torch version
- vit-prisma version
- sklearn version
- `uv.lock` hash
- git commit if available
- CUDA version
- checkpoint repo id
- CLIP model id

### 2.4 Dataset Acquisition And Parsing

Download CUB-200-2011 from the official Caltech Data record:

```bash
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar xzf CUB_200_2011.tgz
```

Expected directory contents:

- `images.txt`: `<image_id> <relative_path>`
- `image_class_labels.txt`: `<image_id> <class_id>`
- `train_test_split.txt`: `<image_id> <is_training>`
- `images/`: JPEG files
- `attributes/`: attribute labels, not used in Stages 1-3

Parse the metadata into a single table keyed by `image_id` with columns:

- `image_id`
- `relative_path`
- `class_id_raw`
- `class_id`
- `is_training`

Rules:

- Remap class ids from 1-based to 0-based: `class_id = class_id_raw - 1`.
- Sort rows by `image_id` before any feature extraction.
- Build train and test splits from `is_training`.
- Preserve the row order used to write `Z_train.npy` and `Z_test.npy`.

Expected counts:

- Total images: 11,788
- Train images: 5,994
- Test images: 5,794
- Classes: 200

Save the row mapping used for the feature caches:

```txt
features/row_mapping.csv
```

with columns:

- `split`
- `row_index`
- `image_id`
- `relative_path`
- `class_id`

This file is required for debugging label alignment and any later qualitative
analysis of surviving features.

---

## 3. Stage 1: SAE Feature Extraction

### 3.1 Verify The SAE Checkpoint First

Do not run dataset extraction until this script verifies the checkpoint,
config, encoder API, and hook contract:

```python
from huggingface_hub import hf_hub_download
from vit_prisma.sae import SparseAutoencoder

repo_id = (
    "Prisma-Multimodal/"
    "sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-8e-05"
)

sae_path = hf_hub_download(repo_id, filename="n_images_2600058.pt")
config_path = hf_hub_download(repo_id, filename="config.json")

sae = SparseAutoencoder.load_from_pretrained(sae_path)

print(sae.W_enc.shape)
print(sae.W_dec.shape)
print(config_path)
```

Expected contract to verify from the checkpoint config and loaded weights:

- Input dimension is 768.
- SAE dimension is 49,152.
- Hook point is `hook_resid_post`.
- Layer index is 11.
- SAE type is a vanilla ReLU SAE.
- The base model in the config matches the DataComp ViT-B/32 backbone.
- A single dummy encoder pass on shape `(1, 768)` works using the library's
  actual encoder API, for example `sae.encode(...)` if that method exists.

This verification script must save `results/verify_sae.json` with:

- checkpoint repo id
- weight filename
- config filename
- observed encoder call used
- observed dummy output shape
- observed hook name and layer
- whether the fallback equivalence test was run
- whether the fallback equivalence test passed

If any of those fail, stop and debug before proceeding.

### 3.2 Load The Matching CLIP Model

Primary path using Prisma:

```python
from vit_prisma.models.model_loader import load_hooked_model

model = load_hooked_model("open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K")
model.eval()
model.to("cuda")
```

Fallback path using open_clip:

```python
import open_clip

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="datacomp_xl_s13b_b90k",
)
clip_model.eval().cuda()
```

The fallback is disabled by default. It may be enabled only if a fixed-image
equivalence test passes against Prisma's hooked model output.

Required fallback gate:

- same preprocessing recipe
- same token count
- same hidden size
- cosine similarity above `0.9999` on the extracted CLS residual stream for at
  least one fixed image
- max absolute difference below `1e-4` on the same extracted tensor

### 3.3 Confirm The Hook Output Before Running The Full Dataset

The SAE was trained on all 50 tokens at `hook_resid_post` layer 11. For this
project, the primary pipeline uses the CLS token only.

Verify the hidden state shape on one image:

```python
# Expected hidden state shape before token selection:
# (batch, 50, 768)
```

Expected token semantics:

- token 0: CLS
- tokens 1-49: patch tokens

Primary protocol:

```python
h = resid_post            # (batch, 50, 768)
h_cls = h[:, 0, :]        # (batch, 768)
z = sae.encode(h_cls)     # (batch, 49152)
```

Do not average over tokens in the main pipeline. If time remains later, mean
pooled token features can be added as an auxiliary robustness check, but they
are not required for the midterm.

### 3.4 Extraction Loop

For each image:

1. Load the image from `images/<relative_path>`.
2. Apply CLIP preprocessing only. No random crop, flip, or color jitter.
3. Run the CLIP forward pass up to the confirmed hook point.
4. Select the CLS token.
5. Encode with the SAE.
6. Write the resulting `float32` feature row into the appropriate split matrix.

Implementation requirement:

- Do not accumulate all batch outputs in Python lists.
- Preallocate the output arrays and write incrementally.
- Prefer `np.lib.format.open_memmap(...)` so extraction uses a single on-disk
  target array rather than building a second full in-memory copy.

Example:

```python
Z_train_mm = np.lib.format.open_memmap(
    "features/Z_train.npy",
    mode="w+",
    dtype=np.float32,
    shape=(5994, 49152),
)
```

After the first batch, assert:

```python
assert z.shape[1] == 49152
z_min = float(z.min().item())
print(f"z_min={z_min:.8f}")
assert z_min >= -1e-6
sparsity = (z == 0).float().mean().item()
print(f"sparsity={sparsity:.3f}")
```

The exact sparsity will vary, but it should be high and clearly non-dense.

### 3.5 Caching

Save:

```txt
features/
  Z_train.npy            # (5994, 49152), float32
  Z_test.npy             # (5794, 49152), float32
  y_train.npy            # (5994,), int64
  y_test.npy             # (5794,), int64
  row_mapping.csv
  extraction_meta.json
```

`extraction_meta.json` must include:

- `repo_id`
- `weight_filename`
- `config_filename`
- `clip_model_name`
- `preprocess_id`
- `image_size`
- `hook_name`
- `layer`
- `token_policy`
- `n_features`
- `input_dim`
- `vit_prisma_version`
- `torch_version`
- `sae_config_path`
- `sae_config_hash`
- `dtype_policy`
- `observed_hook_shape`

### 3.6 Post-Cache Sanity Checks

```python
import numpy as np
from collections import Counter

Z = np.load("features/Z_train.npy")
y = np.load("features/y_train.npy")

assert Z.shape == (5994, 49152)
assert y.shape == (5994,)

counts = Counter(y.tolist())
assert len(counts) == 200
assert all(29 <= c <= 30 for c in counts.values())

dead = (Z.max(axis=0) == 0).sum()
sigma = Z.std(axis=0)

print(f"dead={dead}")
print(f"sigma_zero={(sigma == 0).sum()}")
print(f"median_sigma={np.median(sigma):.6f}")
```

If `dead` is extremely large or `sigma` is nearly zero everywhere, the
extraction path is probably wrong.

---

## 4. Stage 2: Linear Classifier

### 4.1 Train/Val Split

Split indices, not just the arrays, so the split is reproducible and explicit:

```python
import json
import numpy as np
from sklearn.model_selection import train_test_split

idx = np.arange(len(Z_train))

train_idx, val_idx = train_test_split(
    idx,
    test_size=0.2,
    stratify=y_train,
    random_state=42,
)

Z_tr = Z_train[train_idx]
Z_val = Z_train[val_idx]
y_tr = y_train[train_idx]
y_val = y_train[val_idx]

with open("results/split_indices.json", "w") as f:
    json.dump(
        {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
        },
        f,
        indent=2,
    )
```

Expected sizes:

- `D_train`: about 4,795
- `D_val`: about 1,199
- `D_test`: 5,794

### 4.2 Cross-Validate Regularization

Use `cross_validate(..., return_estimator=True)` so convergence can be checked
per fold.

```python
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

C_candidates = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for C in C_candidates:
    clf = LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        multi_class="multinomial",
        random_state=42,
    )
    result = cross_validate(
        clf,
        Z_tr,
        y_tr,
        cv=cv,
        scoring="accuracy",
        return_estimator=True,
        n_jobs=1,
    )
    iters = [int(est.n_iter_[0]) for est in result["estimator"]]
    converged = all(it < 5000 for it in iters)
    cv_results[str(C)] = {
        "mean_acc": float(result["test_score"].mean()),
        "std_acc": float(result["test_score"].std()),
        "fold_accs": result["test_score"].tolist(),
        "fold_iters": iters,
        "converged": converged,
    }

best_C = max(C_candidates, key=lambda c: cv_results[str(c)]["mean_acc"])

with open("results/cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2)
```

Why `n_jobs=1` by default:

- `cross_validate` can parallelize folds.
- Five concurrent multinomial `lbfgs` fits at 49,152 features can consume a lot
  of RAM.
- Increase `n_jobs` only if the machine has enough headroom.
- `return_estimator=True` is useful for convergence debugging, but it also keeps
  all fold estimators in memory. Treat that as a deliberate debugging tradeoff.

If any fold hits `max_iter`, increase it to 10,000 or 20,000 and rerun.

### 4.3 Baseline Classifier

```python
clf_full = LogisticRegression(
    C=best_C,
    penalty="l2",
    solver="lbfgs",
    max_iter=5000,
    multi_class="multinomial",
    random_state=42,
)
clf_full.fit(Z_tr, y_tr)

acc_val_full = clf_full.score(Z_val, y_val)
iters_full = clf_full.n_iter_[0]
```

Do not evaluate on `D_test` yet.

Operational note:

- sklearn may cast dense arrays to `float64` internally.
- Runtime memory use can therefore exceed the on-disk `float32` cache size.

Heuristic accuracy guidance, not a hard literature bound:

- If validation accuracy is materially below 40%, suspect a feature-extraction
  or label-alignment bug.
- If it lands in the rough 50%+ regime, the pipeline is at least plausible.

Record:

- `best_C`
- validation accuracy
- `clf_full.n_iter_`
- dead feature count

---

## 5. Stage 3: Iterative Pruning

### 5.1 Precompute Feature Statistics

Compute standard deviations on `D_train` only:

```python
sigma_tr = Z_tr.std(axis=0)
np.save("features/sigma_train.npy", sigma_tr)
```

### 5.2 Pruning Rule

At each round:

1. Fit multinomial logistic regression on the active feature set.
2. Compute feature importance
   `importance_j = ||W[:, j]||_2 * sigma_tr[j]`.
3. Remove the bottom `p` fraction of active features.
4. Record validation accuracy and the active feature indices.

Reference implementation:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def iterative_pruning(Z_tr, y_tr, Z_val, y_val, sigma_tr, C, p=0.1, k_min=5):
    active = np.arange(Z_tr.shape[1])
    results = []

    for t in range(1000):
        clf = LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            multi_class="multinomial",
            random_state=42,
        )
        clf.fit(Z_tr[:, active], y_tr)
        val_acc = clf.score(Z_val[:, active], y_val)

        results.append(
            {
                "round": t,
                "n_features": int(len(active)),
                "val_acc": float(val_acc),
                "feature_indices": active.copy(),
            }
        )

        if len(active) <= k_min:
            break

        W = clf.coef_
        importance = np.linalg.norm(W, axis=0) * sigma_tr[active]

        m = max(int(np.floor(p * len(active))), 1)
        prune_idx = np.argsort(importance)[:m]

        keep_mask = np.ones(len(active), dtype=bool)
        keep_mask[prune_idx] = False
        active = active[keep_mask]

    return results
```

### 5.3 Pruning Schedule

Use:

- `p = 0.1`
- `k_min = 5`
- `C = best_C` from Section 4.2 for the primary run

With `k_0 = 49,152`, `p = 0.1`, and floor-based pruning:

- There are 91 pruning steps to go from 49,152 to 5 features.
- The recorded curve contains 92 points including the initial full-feature
  model.

The first few active counts are:

```txt
49152
44237
39814
35833
32250
...
```

### 5.4 Fixed-C Sensitivity Check

Keeping `C` fixed across all rounds is a practical choice, not a proven
invariant. Treat it as the primary protocol and add a small sensitivity check.

At approximately these feature counts:

- 49,152
- 10,000
- 5,000
- 1,000
- 500
- 100
- 50

run a lightweight 3-fold CV over:

- `0.1 * best_C`
- `best_C`
- `10 * best_C`

If the best `C` changes consistently and materially, note it in the report.
Do not make this sensitivity pass the main experimental loop.

### 5.5 Save Artifacts

Save both a lightweight curve and the full per-round index sets:

```python
import json
import numpy as np

curve = np.array([(r["n_features"], r["val_acc"]) for r in results])
np.save("results/pruning_curve.npy", curve)

serializable = []
for r in results:
    serializable.append(
        {
            "round": r["round"],
            "n_features": r["n_features"],
            "val_acc": r["val_acc"],
            "feature_indices": r["feature_indices"].tolist(),
        }
    )

with open("results/pruning_results.json", "w") as f:
    json.dump(serializable, f)
```

### 5.6 Compute `k_delta`

```python
import json

a_val_0 = results[0]["val_acc"]
deltas = [0.01, 0.02, 0.05, 0.10]
k_delta_table = {}

for delta in deltas:
    threshold = a_val_0 - delta
    valid = [r for r in results if r["val_acc"] >= threshold]
    k_delta_table[str(delta)] = (
        min(r["n_features"] for r in valid) if valid else None
    )

with open("results/k_delta_table.json", "w") as f:
    json.dump(k_delta_table, f, indent=2)
```

### 5.7 Final Test-Set Protocol

This is the corrected protocol.

Do not reuse the exact feature subset found on `D_train` alone. Instead:

1. Use `D_train` and `D_val` to choose `k_delta`.
2. For each selected `k_delta`, rerun the pruning procedure on the full 5,994
   training images to obtain a full-data feature subset of that size.
3. Fit the final classifier on all 5,994 training images using that subset.
4. Evaluate once on `D_test`.

No-leakage rule:

- `D_test` is touched only in this stage.
- It is never used for model selection, hyperparameter selection, or pruning
  decisions.
- It is evaluated once per selected operating point.

Reference helper:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def prune_to_k(Z_fit, y_fit, sigma_fit, C, k_target, p=0.1):
    active = np.arange(Z_fit.shape[1])

    while len(active) > k_target:
        clf = LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            multi_class="multinomial",
            random_state=42,
        )
        clf.fit(Z_fit[:, active], y_fit)

        W = clf.coef_
        importance = np.linalg.norm(W, axis=0) * sigma_fit[active]

        m = max(int(np.floor(p * len(active))), 1)
        m = min(m, len(active) - k_target)
        prune_idx = np.argsort(importance)[:m]

        keep_mask = np.ones(len(active), dtype=bool)
        keep_mask[prune_idx] = False
        active = active[keep_mask]

    clf_final = LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        multi_class="multinomial",
        random_state=42,
    )
    clf_final.fit(Z_fit[:, active], y_fit)
    return active, clf_final
```

Final evaluation loop:

```python
import json
import numpy as np

with open("results/k_delta_table.json") as f:
    k_delta_table = json.load(f)

sigma_full = Z_train_full.std(axis=0)
final_test_results = []

for delta in ["0.01", "0.02", "0.05"]:
    k_target = k_delta_table[delta]
    if k_target is None:
        continue

    active_full, clf_final = prune_to_k(
        Z_train_full,
        y_train_full,
        sigma_full,
        best_C,
        k_target=int(k_target),
        p=0.1,
    )
    test_acc = clf_final.score(Z_test[:, active_full], y_test)
    final_test_results.append(
        {
            "delta": delta,
            "k_target": int(k_target),
            "n_features_final": int(len(active_full)),
            "test_acc": float(test_acc),
            "active_feature_indices": active_full.tolist(),
        }
    )

with open("results/final_test_results.json", "w") as f:
    json.dump(final_test_results, f, indent=2)
```

---

## 6. Robustness Check: L1 Baseline

L1 selection is scale-sensitive, so the baseline must be standardized.

Important caveat:

- The iterative pruning pipeline uses raw SAE activations and a
  weight-times-standard-deviation ranking.
- The L1 baseline should be standardized first.
- This makes the comparison useful but not perfectly apples-to-apples.
- Treat it as an auxiliary robustness check, not a direct replacement for the
  pruning curve.

Reference implementation:

```python
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

C_l1_candidates = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
l1_results = []

for C_l1 in C_l1_candidates:
    pipe = make_pipeline(
        StandardScaler(copy=False),
        LogisticRegression(
            penalty="l1",
            C=C_l1,
            solver="saga",
            max_iter=10000,
            tol=1e-3,
            multi_class="multinomial",
            random_state=42,
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(Z_tr, y_tr)
    val_acc = pipe.score(Z_val, y_val)

    clf_l1 = pipe.named_steps["logisticregression"]
    W_l1 = clf_l1.coef_
    nonzero = int((np.abs(W_l1).sum(axis=0) > 1e-10).sum())
    convergence_warnings = sum(
        issubclass(w.category, ConvergenceWarning) for w in caught
    )

    l1_results.append(
        {
            "C": C_l1,
            "val_acc": float(val_acc),
            "nonzero_features": nonzero,
            "convergence_warnings": int(convergence_warnings),
        }
    )

with open("results/l1_baseline.json", "w") as f:
    json.dump(l1_results, f, indent=2)
```

Overlay these points on the pruning curve.

---

## 7. Plotting

```python
import json
import matplotlib.pyplot as plt
import numpy as np

curve = np.load("results/pruning_curve.npy")

with open("results/l1_baseline.json") as f:
    l1_results = json.load(f)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    curve[:, 0],
    curve[:, 1] * 100,
    "o-",
    markersize=2,
    linewidth=1.5,
    label="Iterative pruning",
)

for r in l1_results:
    ax.plot(
        r["nonzero_features"],
        r["val_acc"] * 100,
        "s",
        color="red",
        markersize=7,
        zorder=5,
    )
ax.plot([], [], "s", color="red", markersize=7, label="L1 baseline")

ax.axhline(
    curve[0, 1] * 100,
    color="gray",
    linestyle="--",
    alpha=0.5,
    label=f"Full ({curve[0, 1] * 100:.1f}%)",
)

for delta in [0.01, 0.02, 0.05, 0.10]:
    ax.axhline(
        (curve[0, 1] - delta) * 100,
        color="orange",
        linestyle=":",
        alpha=0.3,
    )

ax.set_xlabel("Number of SAE features")
ax.set_ylabel("Validation accuracy (%)")
ax.set_xscale("log")
ax.invert_xaxis()
ax.legend()
ax.set_title("Iterative Feature Pruning on CUB-200")
plt.tight_layout()
plt.savefig("figures/pruning_curve.pdf", dpi=300, bbox_inches="tight")
```

---

## 8. Failure Modes And Debug Checklist

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| SAE checkpoint download fails | Wrong repo id or filename | Use the confirmed repo id and `n_images_2600058.pt` |
| `z.shape[1] != 49152` | Wrong checkpoint or wrong config | Inspect `config.json` and loaded weight shapes |
| Baseline accuracy is very low | Wrong CLIP backbone | Make sure the base model is DataComp, not OpenAI CLIP |
| Baseline accuracy is very low | Wrong hook point or wrong token policy | Verify `hook_resid_post`, layer 11, and CLS extraction |
| Rows and labels do not match | Bad ordering between metadata and extraction loop | Save and inspect `row_mapping.csv` |
| L-BFGS is too slow or fails to converge | Large multiclass problem | Increase `max_iter`, reduce `C`, or lower fold parallelism |
| Early pruning rounds are very slow | 49,152 features is large | Accept long early rounds; later rounds shrink quickly |
| L1 produces strange sparsity | Features were not standardized | Use `StandardScaler` in the L1 pipeline |

---

## 9. File Structure

```txt
sae-cbm-pruning/
├── pyproject.toml
├── uv.lock
├── config.py
├── scripts/
│   ├── 00_verify_sae.py
│   ├── 01_extract_features.py
│   ├── 02_train_baseline.py
│   ├── 03_run_pruning.py
│   ├── 04_l1_baseline.py
│   ├── 05_final_test.py
│   └── 06_plot_results.py
├── data/
│   └── CUB_200_2011/
├── features/
│   ├── Z_train.npy
│   ├── Z_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── sigma_train.npy
│   ├── row_mapping.csv
│   └── extraction_meta.json
├── results/
│   ├── hardware.txt
│   ├── verify_sae.json
│   ├── split_indices.json
│   ├── cv_results.json
│   ├── pruning_results.json
│   ├── pruning_curve.npy
│   ├── l1_baseline.json
│   ├── k_delta_table.json
│   ├── final_test_results.json
│   └── run_manifests/
└── figures/
    └── pruning_curve.pdf
```

---

## 10. Execution Order

```bash
# 0. Create/sync the environment
uv sync

# 1. Verify the checkpoint and model contract
uv run python scripts/00_verify_sae.py

# 2. Extract features
uv run python scripts/01_extract_features.py

# 3. Train the baseline and choose C
uv run python scripts/02_train_baseline.py

# 4. Run iterative pruning
uv run python scripts/03_run_pruning.py

# 5. Run the L1 robustness baseline
uv run python scripts/04_l1_baseline.py

# 6. Rerun pruning on the full train split at selected k values and evaluate on test
uv run python scripts/05_final_test.py

# 7. Plot
uv run python scripts/06_plot_results.py
```

Expected wall-clock estimate:

- SAE verification: 1-2 minutes
- Extraction: about 15-25 minutes with a GPU
- Baseline + CV: about 30-60 minutes
- Pruning: a few hours, dominated by early rounds
- L1 baseline: about 30-60 minutes

---

## 11. Paper Language To Update Later

These text changes are now directionally correct, but do not edit the paper
until the code path has actually run successfully at least once.

| Old text | New text |
|----------|----------|
| CLIP ViT-B/16 | CLIP ViT-B/32 DataComp |
| 8x expansion | 64x expansion |
| `k ~= 6144` | `k = 49152` |
| about 65 pruning rounds | about 90 pruning rounds |
| stronger leakage evidence | greater capacity for leakage |

Important wording change:

- A larger `k / d` ratio increases the model's capacity to preserve or
  reconstruct information.
- It does not by itself prove leakage.
- Phrase this as a capacity argument, not empirical evidence.

---

## 12. Midterm Checklist

- [ ] Confirmed Prisma checkpoint and weight filename.
- [ ] Confirmed matching CLIP backbone is DataComp ViT-B/32.
- [ ] Saved reproducible split indices.
- [ ] Saved row-to-image mapping.
- [ ] Saved `verify_sae.json` and per-script run manifests.
- [ ] Reported dead-feature count and basic sparsity statistics.
- [ ] Reported baseline validation accuracy with all 49,152 features.
- [ ] Produced the pruning curve.
- [ ] Produced the `k_delta` table for `delta in {0.01, 0.02, 0.05, 0.10}`.
- [ ] Overlaid the L1 baseline.
- [ ] Evaluated selected operating points on the held-out test set using the
  corrected full-data rerun protocol.
