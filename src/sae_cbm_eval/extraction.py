from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from vit_prisma.models.model_loader import load_hooked_model
from vit_prisma.sae import SparseAutoencoder
from vit_prisma.transforms.model_transforms import get_model_transforms

from sae_cbm_eval.constants import CLIP_MODEL_ID


def load_sae_for_extraction(
    *,
    repo_id: str,
    weight_filename: str,
    config_filename: str,
    device: str,
    hf_token: str | None,
):
    config_path = Path(
        hf_hub_download(repo_id=repo_id, filename=config_filename, token=hf_token)
    )
    weights_path = Path(
        hf_hub_download(repo_id=repo_id, filename=weight_filename, token=hf_token)
    )
    sae = SparseAutoencoder.load_from_pretrained(
        str(weights_path),
        config_path=str(config_path),
        current_cfg={"_device": device, "_dtype": "float32"},
    )
    sae = sae.to(device=device, dtype=torch.float32)
    sae.eval()
    return sae, config_path, weights_path


def load_clip_model_for_extraction(*, clip_model_id: str, device: str):
    # `move_to_device=True` is not reliable for this model on MPS in the current
    # vit_prisma loader, so we load on CPU and move explicitly.
    model = load_hooked_model(
        clip_model_id,
        device="cpu",
        move_to_device=False,
        dtype=torch.float32,
    )
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def build_preprocess(clip_model_id: str = CLIP_MODEL_ID):
    return get_model_transforms(clip_model_id)


def open_feature_memmap(path: Path, n_rows: int, n_features: int) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        str(path),
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, n_features),
    )


def extract_split_features(
    *,
    model,
    sae,
    dataloader,
    hook_name: str,
    output_path: Path,
    n_rows: int,
    n_features: int,
    device: str,
) -> dict[str, Any]:
    feature_mm = open_feature_memmap(output_path, n_rows=n_rows, n_features=n_features)

    observed_hook_shape: list[int] | None = None
    first_batch_stats: dict[str, float] | None = None

    with torch.inference_mode():
        for images, row_indices, _class_ids in dataloader:
            images = images.to(device=device, dtype=torch.float32)
            _, cache = model.run_with_cache(images, names_filter=hook_name)
            activations = cache[hook_name]
            observed_hook_shape = list(activations.shape)

            h_cls = activations[:, 0, :]
            _, features = sae.encode(h_cls)

            if features.ndim != 2 or features.shape[1] != n_features:
                raise ValueError(
                    f"Expected feature shape (batch, {n_features}), observed {tuple(features.shape)}"
                )

            if first_batch_stats is None:
                z_min = float(features.min().item())
                if z_min < -1e-6:
                    raise ValueError(
                        f"SAE features should be non-negative for ReLU SAE, observed min {z_min}"
                    )
                sparsity = float((features == 0).float().mean().item())
                first_batch_stats = {
                    "z_min": z_min,
                    "z_max": float(features.max().item()),
                    "z_mean": float(features.mean().item()),
                    "sparsity": sparsity,
                    "density": 1.0 - sparsity,
                }

            feature_mm[row_indices.numpy()] = (
                features.detach().to(device="cpu", dtype=torch.float32).numpy()
            )
            del cache

    feature_mm.flush()

    if observed_hook_shape is None or first_batch_stats is None:
        raise ValueError(f"No batches were processed for {output_path}")

    return {
        "observed_hook_shape": observed_hook_shape,
        "first_batch_stats": first_batch_stats,
    }
