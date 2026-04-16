from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from vit_prisma.models.model_loader import load_config
from vit_prisma.sae import SparseAutoencoder
from vit_prisma.utils.enums import ModelType

from sae_cbm_eval.constants import (
    CLIP_MODEL_ID,
    EXPECTED_ACTIVATION_FN,
    EXPECTED_ARCHITECTURE,
    EXPECTED_CONTEXT_SIZE,
    EXPECTED_HOOK_LAYER,
    EXPECTED_HOOK_NAME,
    EXPECTED_HOOK_SUBTYPE,
    EXPECTED_INPUT_DIM,
    EXPECTED_MODEL_CLASS_NAME,
    EXPECTED_SAE_DIM,
    RANDOM_SEED,
    SAE_CONFIG_FILENAME,
    SAE_REPO_ID,
    SAE_WEIGHT_FILENAME,
)
from sae_cbm_eval.runtime import (
    configure_runtime_logging,
    load_project_env,
    preferred_torch_device,
    project_path,
    set_reproducibility,
    sha256_file,
    write_hardware_report,
    write_json,
    write_run_manifest,
)

warnings.filterwarnings(
    "ignore",
    message=r".*Plotly version 5\.19\.0, which is not compatible with this version of Kaleido.*",
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"kaleido\._sync_server")
SCRIPT_NAME = "00_verify_sae"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the locked SAE checkpoint contract."
    )
    parser.add_argument("--repo-id", default=SAE_REPO_ID)
    parser.add_argument("--weight-filename", default=SAE_WEIGHT_FILENAME)
    parser.add_argument("--config-filename", default=SAE_CONFIG_FILENAME)
    parser.add_argument(
        "--clip-model-id",
        default=CLIP_MODEL_ID,
        help="Prisma hooked model identifier expected by the SAE config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_path("results", "verify_sae.json"),
        help="Verification report path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for the verifier itself.",
    )
    return parser.parse_args()


def expect_equal(problems: list[str], label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        problems.append(f"{label}: expected {expected!r}, observed {observed!r}")


def load_raw_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text())


def verify_checkpoint_metadata(
    raw_config: dict[str, Any], clip_model_id: str
) -> dict[str, Any]:
    problems: list[str] = []
    observed_hook_name = (
        f"blocks.{raw_config.get('hook_point_layer')}.{raw_config.get('layer_subtype')}"
    )

    expect_equal(
        problems,
        "model_class_name",
        raw_config.get("model_class_name"),
        EXPECTED_MODEL_CLASS_NAME,
    )
    expect_equal(problems, "model_name", raw_config.get("model_name"), clip_model_id)
    expect_equal(
        problems,
        "hook_point_layer",
        raw_config.get("hook_point_layer"),
        EXPECTED_HOOK_LAYER,
    )
    expect_equal(
        problems,
        "layer_subtype",
        raw_config.get("layer_subtype"),
        EXPECTED_HOOK_SUBTYPE,
    )
    expect_equal(problems, "hook_name", observed_hook_name, EXPECTED_HOOK_NAME)
    expect_equal(problems, "d_in", raw_config.get("d_in"), EXPECTED_INPUT_DIM)
    expect_equal(problems, "d_sae", raw_config.get("d_sae"), EXPECTED_SAE_DIM)
    expect_equal(
        problems, "context_size", raw_config.get("context_size"), EXPECTED_CONTEXT_SIZE
    )
    expect_equal(
        problems,
        "activation_fn_str",
        raw_config.get("activation_fn_str"),
        EXPECTED_ACTIVATION_FN,
    )
    expect_equal(
        problems,
        "architecture",
        raw_config.get("architecture"),
        EXPECTED_ARCHITECTURE,
    )

    return {
        "observed_hook_name": observed_hook_name,
        "observed_hook_layer": raw_config.get("hook_point_layer"),
        "problems": problems,
    }


def verify_matching_clip_config(clip_model_id: str) -> dict[str, Any]:
    clip_cfg = load_config(clip_model_id, model_type=ModelType.VISION)
    patch_size = getattr(clip_cfg, "patch_size", None)
    image_size = getattr(clip_cfg, "image_size", None)
    token_count = None
    if patch_size and image_size:
        token_count = (image_size // patch_size) ** 2 + 1

    problems: list[str] = []
    expect_equal(
        problems,
        "clip_cfg.model_name",
        getattr(clip_cfg, "model_name", None),
        clip_model_id,
    )
    expect_equal(
        problems,
        "clip_cfg.d_model",
        getattr(clip_cfg, "d_model", None),
        EXPECTED_INPUT_DIM,
    )
    expect_equal(problems, "clip_cfg.image_size", image_size, 224)
    expect_equal(problems, "clip_cfg.patch_size", patch_size, 32)
    # CLIP always produces 50 tokens (49 patches + CLS); SAE context_size may differ
    # (e.g. 1 for CLS-only SAEs). Validate CLIP token count independently.
    expect_equal(problems, "clip token count", token_count, 50)

    return {
        "clip_config_type": type(clip_cfg).__name__,
        "clip_config_d_model": getattr(clip_cfg, "d_model", None),
        "clip_config_n_layers": getattr(clip_cfg, "n_layers", None),
        "clip_config_patch_size": patch_size,
        "clip_config_image_size": image_size,
        "clip_config_token_count": token_count,
        "problems": problems,
    }


def verify_weight_shapes(weights_path: Path) -> dict[str, Any]:
    raw = torch.load(weights_path, map_location="cpu", weights_only=False)
    # Handle both flat state dicts and nested {cfg, state_dict} wrappers
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) and "state_dict" in raw else raw
    problems: list[str] = []
    expected_keys = {"W_enc", "W_dec", "b_enc", "b_dec"}
    observed_keys = set(state_dict.keys())

    if observed_keys != expected_keys:
        problems.append(
            f"state_dict keys: expected {sorted(expected_keys)!r}, observed {sorted(observed_keys)!r}"
        )

    if "W_enc" in state_dict:
        expect_equal(
            problems,
            "W_enc.shape",
            tuple(state_dict["W_enc"].shape),
            (EXPECTED_INPUT_DIM, EXPECTED_SAE_DIM),
        )
    if "W_dec" in state_dict:
        expect_equal(
            problems,
            "W_dec.shape",
            tuple(state_dict["W_dec"].shape),
            (EXPECTED_SAE_DIM, EXPECTED_INPUT_DIM),
        )
    if "b_enc" in state_dict:
        expect_equal(
            problems,
            "b_enc.shape",
            tuple(state_dict["b_enc"].shape),
            (EXPECTED_SAE_DIM,),
        )
    if "b_dec" in state_dict:
        expect_equal(
            problems,
            "b_dec.shape",
            tuple(state_dict["b_dec"].shape),
            (EXPECTED_INPUT_DIM,),
        )

    return {
        "state_dict_keys": sorted(observed_keys),
        "weight_shapes": {
            key: list(state_dict[key].shape)
            for key in ["W_enc", "W_dec", "b_enc", "b_dec"]
            if key in state_dict
        },
        "problems": problems,
    }


def verify_sae_runtime(weights_path: Path, config_path: Path) -> dict[str, Any]:
    device = preferred_torch_device()
    sae = SparseAutoencoder.load_from_pretrained(
        str(weights_path),
        config_path=str(config_path),
        current_cfg={"_device": device, "_dtype": "float32"},
    )
    sae.eval()

    dummy = torch.zeros(
        (1, EXPECTED_INPUT_DIM), dtype=torch.float32, device=sae.W_enc.device
    )
    sae_in, features = sae.encode(dummy)

    problems: list[str] = []
    expect_equal(problems, "sae runtime input dim", sae.d_in, EXPECTED_INPUT_DIM)
    expect_equal(problems, "sae runtime feature dim", sae.d_sae, EXPECTED_SAE_DIM)
    expect_equal(
        problems, "dummy sae_in shape", tuple(sae_in.shape), (1, EXPECTED_INPUT_DIM)
    )
    expect_equal(
        problems, "dummy feature shape", tuple(features.shape), (1, EXPECTED_SAE_DIM)
    )

    z_min = float(features.min().item())
    if z_min < -1e-6:
        problems.append(
            f"dummy feature minimum should be non-negative for ReLU SAE, observed {z_min}"
        )

    return {
        "observed_encoder_call_used": "sae.encode(dummy)[1]",
        "observed_dummy_output_shape": list(features.shape),
        "dummy_input_device": str(dummy.device),
        "sae_parameter_device": str(sae.W_enc.device),
        "dummy_feature_min": z_min,
        "dummy_feature_max": float(features.max().item()),
        "dummy_feature_mean": float(features.mean().item()),
        "runtime_hook_name": sae.cfg.hook_point,
        "runtime_hook_layer": sae.cfg.hook_point_layer,
        "runtime_layer_subtype": sae.cfg.layer_subtype,
        "runtime_model_name": sae.cfg.model_name,
        "runtime_architecture": sae.cfg.architecture,
        "runtime_activation_fn": sae.cfg.activation_fn_str,
        "runtime_context_size": sae.cfg.context_size,
        "problems": problems,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    hf_token = load_project_env()
    set_reproducibility(RANDOM_SEED)
    write_hardware_report()

    logging.info("Downloading SAE config and weights from %s", args.repo_id)
    config_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.config_filename,
            token=hf_token,
        )
    )
    weights_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.weight_filename,
            token=hf_token,
        )
    )

    raw_config = load_raw_config(config_path)
    checkpoint_checks = verify_checkpoint_metadata(raw_config, args.clip_model_id)
    clip_checks = verify_matching_clip_config(args.clip_model_id)
    weight_checks = verify_weight_shapes(weights_path)
    runtime_checks = verify_sae_runtime(weights_path, config_path)

    all_problems = (
        checkpoint_checks["problems"]
        + clip_checks["problems"]
        + weight_checks["problems"]
        + runtime_checks["problems"]
    )

    return {
        "checkpoint_repo_id": args.repo_id,
        "weight_filename": args.weight_filename,
        "config_filename": args.config_filename,
        "clip_model_id": args.clip_model_id,
        "config_path": str(config_path),
        "weights_path": str(weights_path),
        "sae_config_hash": sha256_file(config_path),
        "weights_hash": sha256_file(weights_path),
        "observed_encoder_call_used": runtime_checks["observed_encoder_call_used"],
        "observed_dummy_output_shape": runtime_checks["observed_dummy_output_shape"],
        "observed_hook_name": checkpoint_checks["observed_hook_name"],
        "observed_hook_layer": checkpoint_checks["observed_hook_layer"],
        "fallback_equivalence_test_run": False,
        "fallback_equivalence_test_passed": False,
        "fallback_equivalence_test_reason": "disabled by default",
        "checks": {
            "checkpoint": checkpoint_checks,
            "clip_config": clip_checks,
            "weights": weight_checks,
            "runtime": runtime_checks,
        },
        "problems": all_problems,
        "ok": not all_problems,
    }


def main() -> int:
    args = parse_args()
    configure_runtime_logging(verbose=args.verbose)

    try:
        report = build_report(args)
        write_json(args.output, report)
        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=args.repo_id,
            clip_model_id=args.clip_model_id,
            extra={
                "status": "ok" if report["ok"] else "failed_verification",
                "verify_report_path": str(args.output),
                "problem_count": len(report["problems"]),
            },
        )
    except Exception as exc:
        write_run_manifest(
            script_name=SCRIPT_NAME,
            checkpoint_repo_id=args.repo_id,
            clip_model_id=args.clip_model_id,
            extra={
                "status": "error",
                "error": str(exc),
            },
        )
        logging.exception("SAE verification failed")
        return 1

    if not report["ok"]:
        logging.error("SAE verification failed:\n- %s", "\n- ".join(report["problems"]))
        return 1

    logging.info("SAE verification passed. Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
