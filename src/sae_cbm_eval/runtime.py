from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_reproducibility(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_runtime_logging(verbose: bool = False) -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*Plotly version 5\.19\.0, which is not compatible with this version of Kaleido.*",
    )
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    for logger_name in [
        "filelock",
        "httpcore",
        "httpx",
        "huggingface_hub",
        "urllib3",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def load_project_env() -> str | None:
    load_dotenv(project_path(".env"), override=False)
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
    return token


def get_package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def get_uv_lock_hash() -> str | None:
    uv_lock = project_path("uv.lock")
    if not uv_lock.exists():
        return None
    return sha256_file(uv_lock)


def preferred_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str | None) -> str:
    if device in {None, "", "auto"}:
        return preferred_torch_device()
    return device


def collect_hardware_info() -> dict[str, Any]:
    gpu_names: list[str] = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": gpu_names,
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
        "preferred_device": preferred_torch_device(),
    }


def write_hardware_report(results_dir: Path | None = None) -> dict[str, Any]:
    info = collect_hardware_info()
    lines = [
        f"platform: {info['platform']}",
        f"machine: {info['machine']}",
        f"processor: {info['processor']}",
        f"python_version: {info['python_version']}",
        f"torch_version: {info['torch_version']}",
        f"cuda_available: {info['cuda_available']}",
        f"cuda_version: {info['cuda_version']}",
        f"cuda_device_count: {info['cuda_device_count']}",
        f"gpu_names: {', '.join(info['gpu_names']) if info['gpu_names'] else 'none'}",
        f"mps_available: {info['mps_available']}",
        f"preferred_device: {info['preferred_device']}",
    ]
    base_dir = results_dir or project_path("results")
    path = base_dir / "hardware.txt"
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n")
    return info


def build_run_manifest(
    *,
    script_name: str,
    checkpoint_repo_id: str | None,
    clip_model_id: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "script_name": script_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "vit_prisma_version": get_package_version("vit-prisma"),
        "sklearn_version": get_package_version("scikit-learn"),
        "uv_lock_hash": get_uv_lock_hash(),
        "git_commit": get_git_commit(),
        "cuda_version": torch.version.cuda,
        "checkpoint_repo_id": checkpoint_repo_id,
        "clip_model_id": clip_model_id,
    }
    if extra:
        payload.update(extra)
    return payload


def write_run_manifest(
    *,
    script_name: str,
    checkpoint_repo_id: str | None,
    clip_model_id: str | None,
    extra: dict[str, Any] | None = None,
    results_dir: Path | None = None,
) -> Path:
    manifest = build_run_manifest(
        script_name=script_name,
        checkpoint_repo_id=checkpoint_repo_id,
        clip_model_id=clip_model_id,
        extra=extra,
    )
    base_dir = results_dir or project_path("results")
    path = base_dir / "run_manifests" / f"{script_name}.json"
    write_json(path, manifest)
    return path
