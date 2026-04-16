from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

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
    set_reproducibility,
    write_json,
    write_run_manifest,
)


SCRIPT_NAME = "12_semantic_agreement"
DEFAULT_AUROC_THRESHOLD = 0.65
DEFAULT_MODEL = "gpt-5.4-mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute semantic agreement between MLLM labels and CUB attributes."
    )
    parser.add_argument("--dataset-root", type=Path, default=project_path("data", CUB_DIR_NAME))
    parser.add_argument("--results-dir", type=Path, default=project_path("results"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls to respect rate limits.")
    parser.add_argument(
        "--operating-points",
        nargs="+",
        default=["0.01"],
    )
    parser.add_argument("--auroc-threshold", type=float, default=DEFAULT_AUROC_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text())


def judge_semantic_match(client, model: str, label: str, attribute: str) -> bool:
    """Ask an LLM whether the MLLM label semantically matches the CUB attribute."""
    attr_clean = attribute.replace("_", " ").replace("::", " ")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Do these two descriptions refer to the same or very similar "
                    "visual property of a bird? Answer yes or no.\n\n"
                    f"Description A: \"{label}\"\n"
                    f"Description B: \"{attr_clean}\""
                ),
            }
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )
    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)
    results_dir = ensure_dir(args.results_dir)
    output_path = results_dir / "semantic_agreement.json"

    try:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")
        load_project_env()
        set_reproducibility(RANDOM_SEED)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Add it to .env or export it in your shell."
            )

        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        attr_names_df = parse_attribute_names(Path(args.dataset_root))
        attr_names = attr_names_df["attribute_name"].tolist()

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
                        "llm_agreement": False,
                        "high_quality": False,
                    })
                    continue

                best_attr_idx = match["best_attr_idx"]
                best_auroc = match["best_auroc"]

                llm_agree = judge_semantic_match(
                    client, args.model,
                    label_entry["label"], attr_names[best_attr_idx],
                )

                high_quality = (
                    best_auroc >= args.auroc_threshold
                    and llm_agree
                )
                if high_quality:
                    n_high_quality += 1

                feature_agreements.append({
                    "rank": rank,
                    "label": label_entry["label"],
                    "best_attr": attr_names[best_attr_idx],
                    "best_auroc": best_auroc,
                    "llm_agreement": llm_agree,
                    "high_quality": high_quality,
                })

                if (li + 1) % 10 == 0:
                    logging.info("Judged %s/%s features", li + 1, len(valid_labels))

                if args.delay > 0:
                    time.sleep(args.delay)

            delta_result = {
                "delta": delta,
                "n_labeled": len(valid_labels),
                "n_high_quality": n_high_quality,
                "frac_high_quality": n_high_quality / len(valid_labels) if valid_labels else 0,
                "auroc_threshold": args.auroc_threshold,
                "judge_model": args.model,
                "feature_agreements": feature_agreements,
            }
            all_agreement.append(delta_result)

            logging.info(
                "delta=%s: %s/%s high-quality concepts (%.1f%%)",
                delta, n_high_quality, len(valid_labels),
                100 * delta_result["frac_high_quality"],
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
