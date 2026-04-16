from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from sae_cbm_eval.constants import CLIP_MODEL_ID, RANDOM_SEED, SAE_REPO_ID
from sae_cbm_eval.runtime import (
    configure_runtime_logging,
    ensure_dir,
    load_project_env,
    project_path,
    set_reproducibility,
    write_json,
    write_run_manifest,
)


SCRIPT_NAME = "10_label_features"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_OPERATING_POINTS = ["0.01"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label retained features using a multimodal LLM."
    )
    parser.add_argument("--exemplar-dir", type=Path, default=project_path("exemplars"))
    parser.add_argument("--results-dir", type=Path, default=project_path("results"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--operating-points",
        nargs="+",
        default=DEFAULT_OPERATING_POINTS,
    )
    parser.add_argument("--max-features", type=int, default=None,
                        help="Cap on number of features to label (for cost control).")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls to respect rate limits.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def encode_image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def label_feature(
    client,
    model: str,
    montage_path: Path,
    prompt: str,
) -> dict[str, Any]:
    """Call the MLLM API with the montage image and prompt."""
    b64 = encode_image_base64(montage_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        max_completion_tokens=100,
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()
    return {
        "label": text,
        "model": model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }


def is_refusal(label: str) -> bool:
    """Fast-path check for obvious refusals."""
    lower = label.lower()
    return any(marker in lower for marker in ("i cannot", "i can't", "sorry"))


def validate_label_llm(client, model: str, label: str) -> bool:
    """Ask an LLM whether the label describes a visual property."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Does this label describe a visual property (color, shape, "
                    "pattern, texture, or body part) rather than naming a specific "
                    "bird species? Answer yes or no.\n\n"
                    f"Label: \"{label}\""
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

    try:
        load_project_env()
        set_reproducibility(RANDOM_SEED)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Add it to .env or export it in your shell."
            )

        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        for delta in args.operating_points:
            op_dir = args.exemplar_dir / f"delta_{delta}"
            if not op_dir.exists():
                logging.warning("No exemplar directory for delta=%s, skipping.", delta)
                continue

            prompts_path = op_dir / "prompts.json"
            if not prompts_path.exists():
                logging.warning("No prompts.json in %s, skipping.", op_dir)
                continue

            prompts_data = json.loads(prompts_path.read_text())
            output_path = results_dir / f"feature_labels_delta_{delta}.json"

            if output_path.exists() and not args.overwrite:
                logging.info("Labels already exist at %s, skipping.", output_path)
                continue

            if args.max_features is not None:
                prompts_data = prompts_data[:args.max_features]

            labels: list[dict[str, Any]] = []
            n_valid = 0
            n_invalid = 0

            for i, entry in enumerate(prompts_data):
                montage_path = op_dir / entry["montage_file"]
                if not montage_path.exists():
                    logging.warning("Montage not found: %s", montage_path)
                    labels.append({
                        "rank": entry["rank"],
                        "feature_index": entry["feature_index"],
                        "label": None,
                        "valid": False,
                        "error": "montage_not_found",
                    })
                    continue

                try:
                    result = label_feature(client, args.model, montage_path, entry["prompt"])
                    if is_refusal(result["label"]):
                        valid = False
                        validation_method = "refusal_heuristic"
                    else:
                        valid = validate_label_llm(client, args.model, result["label"])
                        validation_method = "llm"
                    labels.append({
                        "rank": entry["rank"],
                        "feature_index": entry["feature_index"],
                        "label": result["label"],
                        "valid": valid,
                        "validation_method": validation_method,
                        "model": result["model"],
                        "usage": result["usage"],
                    })
                    if valid:
                        n_valid += 1
                    else:
                        n_invalid += 1
                except Exception as exc:
                    logging.warning("API error for feature %s: %s", entry["rank"], exc)
                    labels.append({
                        "rank": entry["rank"],
                        "feature_index": entry["feature_index"],
                        "label": None,
                        "valid": False,
                        "error": str(exc),
                    })

                if (i + 1) % 10 == 0:
                    logging.info("Labeled %s/%s features", i + 1, len(prompts_data))

                if args.delay > 0:
                    time.sleep(args.delay)

            write_json(output_path, labels)
            logging.info(
                "delta=%s: labeled %s features (%s valid, %s invalid)",
                delta, len(labels), n_valid, n_invalid,
            )

        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=SAE_REPO_ID,
            clip_model_id=CLIP_MODEL_ID,
            results_dir=results_dir,
            extra={
                "status": "ok",
                "model": args.model,
                "operating_points": args.operating_points,
            },
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
