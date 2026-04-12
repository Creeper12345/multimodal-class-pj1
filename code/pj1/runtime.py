"""Runtime configuration helpers shared by project scripts."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from pathlib import Path
import platform
import sys
from typing import Any


@dataclass(frozen=True)
class RuntimePaths:
    output_dir: str
    hf_home: str
    transformers_cache: str
    torch_home: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def configure_platform_env() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def configure_offline_mode(local_files_only: bool) -> None:
    if not local_files_only:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def configure_hf_endpoint(endpoint: str | None) -> None:
    if not endpoint:
        return
    normalized = endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = normalized
    os.environ["HUGGINGFACE_CO_RESOLVE_ENDPOINT"] = normalized


def configure_runtime_env(output_dir: str | Path) -> RuntimePaths:
    configure_platform_env()
    output_dir = Path(output_dir)
    hf_home = Path(os.environ.setdefault("HF_HOME", str(output_dir / "hf_cache")))
    transformers_cache = Path(
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    )
    torch_home = Path(os.environ.setdefault("TORCH_HOME", str(output_dir / "torch_cache")))
    return RuntimePaths(
        output_dir=str(output_dir),
        hf_home=str(hf_home),
        transformers_cache=str(transformers_cache),
        torch_home=str(torch_home),
    )


def managed_bert_tokenizer_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    return output_dir / "hf_cache" / "manual" / "bert-base-uncased"


def detect_torch_runtime() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "torch_installed": False,
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "cuda_device_count": 0,
            "mps_available": False,
        }

    cuda_available = torch.cuda.is_available()
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    return {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_available": bool(cuda_available),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "mps_available": mps_available,
    }


def detect_system_info() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
