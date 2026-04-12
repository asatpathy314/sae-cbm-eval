from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sae_cbm_eval.attributes import parse_attribute_names
from sae_cbm_eval.constants import (
    CLIP_MODEL_ID,
    CUB_DIR_NAME,
    RANDOM_SEED,
    SAE_REPO_ID,
)
from sae_cbm_eval.runtime import (
    configure_runtime_logging,
    ensure_dir,
    load_project_env,
    project_path,
    resolve_device,
    set_reproducibility,
    write_json,
    write_run_manifest,
)


SCRIPT_NAME = "12_semantic_agreement"
DEFAULT_SIM_THRESHOLD = 0.75
DEFAULT_AUROC_THRESHOLD = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute semantic agreement between MLLM labels and CUB attributes."
    )
    parser.add_argument("--dataset-root", type=Path, default=project_path("data", CUB_DIR_NAME))
    parser.add_argument("--results-dir", type=Path, default=project_path("results"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--operating-points",
        nargs="+",
        default=["0.01"],
    )
    parser.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    parser.add_argument("--auroc-threshold", type=float, default=DEFAULT_AUROC_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text())


def encode_texts_clip(texts: list[str], device: str) -> np.ndarray:
    """Encode texts using open_clip's CLIP text encoder."""
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="datacomp_xl_s13b_b90k",
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    tokens = tokenizer(texts).to(device)
    with torch.inference_mode():
        embeddings = model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return embeddings.cpu().numpy().astype(np.float32)


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)
    results_dir = ensure_dir(args.results_dir)
    device = resolve_device(args.device)
    output_path = results_dir / "semantic_agreement.json"

    try:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")
        load_project_env()
        set_reproducibility(RANDOM_SEED)

        attr_names_df = parse_attribute_names(Path(args.dataset_root))
        attr_names = attr_names_df["attribute_name"].tolist()
        attr_texts = [name.replace("_", " ").replace("::", " ") for name in attr_names]

        alignment_results = load_json(results_dir / "attribute_alignment.json")

        all_agreement: list[dict[str, Any]] = []

        for delta in args.operating_points:
            labels_path = results_dir / f"feature_labels_delta_{delta}.json"
            if not labels_path.exists():
                logging.warning("No labels file for delta=%s, skipping.", delta)
                continue

            labels_data = load_json(labels_path)

            alignment_data = None
            for ar in alignment_results:
                if ar["delta"] == delta:
                    alignment_data = ar
                    break
            if alignment_data is None:
                logging.warning("No alignment data for delta=%s, skipping.", delta)
                continue

            valid_labels = [l for l in labels_data if l.get("valid") and l.get("label")]
            if not valid_labels:
                logging.warning("No valid labels for delta=%s", delta)
                continue

            label_texts = [l["label"] for l in valid_labels]
            logging.info("Encoding %s MLLM labels + %s attribute names", len(label_texts), len(attr_texts))

            all_texts = label_texts + attr_texts
            all_embeddings = encode_texts_clip(all_texts, device)
            label_embeddings = all_embeddings[:len(label_texts)]
            attr_embeddings = all_embeddings[len(label_texts):]

            best_matches = alignment_data["best_matches"]
            match_lookup = {m["feature_rank"]: m for m in best_matches if isinstance(m, dict)}

            feature_agreements: list[dict[str, Any]] = []
            n_high_quality = 0

            for li, label_entry in enumerate(valid_labels):
                rank = label_entry["rank"]
                match = match_lookup.get(rank)

                if match is None or match["best_attr_idx"] < 0:
                    feature_agreements.append({
                        "rank": rank,
                        "label": label_entry["label"],
                        "best_attr": "none",
                        "best_auroc": None,
                        "clip_sim": None,
                        "high_quality": False,
                    })
                    continue

                best_attr_idx = match["best_attr_idx"]
                best_auroc = match["best_auroc"]

                sim = float(label_embeddings[li] @ attr_embeddings[best_attr_idx])

                high_quality = (
                    best_auroc >= args.auroc_threshold
                    and sim >= args.sim_threshold
                )
                if high_quality:
                    n_high_quality += 1

                feature_agreements.append({
                    "rank": rank,
                    "label": label_entry["label"],
                    "best_attr": attr_names[best_attr_idx],
                    "best_auroc": best_auroc,
                    "clip_sim": sim,
                    "high_quality": high_quality,
                })

            sims = [fa["clip_sim"] for fa in feature_agreements if fa["clip_sim"] is not None]
            sims_arr = np.array(sims)

            delta_result = {
                "delta": delta,
                "n_labeled": len(valid_labels),
                "n_high_quality": n_high_quality,
                "frac_high_quality": n_high_quality / len(valid_labels) if valid_labels else 0,
                "mean_clip_sim": float(sims_arr.mean()) if len(sims_arr) > 0 else None,
                "median_clip_sim": float(np.median(sims_arr)) if len(sims_arr) > 0 else None,
                "sim_threshold": args.sim_threshold,
                "auroc_threshold": args.auroc_threshold,
                "feature_agreements": feature_agreements,
            }
            all_agreement.append(delta_result)

            logging.info(
                "delta=%s: %s/%s high-quality concepts (%.1f%%), mean_sim=%.4f",
                delta, n_high_quality, len(valid_labels),
                100 * delta_result["frac_high_quality"],
                delta_result["mean_clip_sim"] or 0,
            )

        write_json(output_path, all_agreement)
        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=results_dir,
            extra={"status": "ok", "operating_points": args.operating_points},
        )
        return 0
    except Exception as exc:
        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=results_dir,
            extra={"status": "error", "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
