"""Caption model adapters for Task 2."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Protocol, Sequence

from PIL import Image

from code.pj1.runtime import configure_platform_env
from code.pj1.task1.models import _batched, _patch_lavis_bert_tokenizer, _resolve_bert_tokenizer_path


configure_platform_env()


class CaptionModel(Protocol):
    name: str

    def generate_captions(
        self,
        image_paths: Sequence[Path],
        batch_size: int,
        generation_config: dict[str, Any],
    ) -> list[str]:
        ...


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LavisCaptionGenerator:
    model_name: str
    model_type: str
    device: str
    local_files_only: bool = False
    bert_tokenizer_path: str | None = None

    def __post_init__(self) -> None:
        import torch
        from lavis.models import load_model_and_preprocess

        self.name = f"lavis_{self.model_name}_{self.model_type}"
        self.device = resolve_device(self.device)
        self._torch = torch
        tokenizer_path = _resolve_bert_tokenizer_path(self.bert_tokenizer_path)
        with _patch_lavis_bert_tokenizer(tokenizer_path):
            self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name=self.model_name,
                model_type=self.model_type,
                is_eval=True,
                device=self.device,
            )
        self.model.eval()

    def generate_captions(
        self,
        image_paths: Sequence[Path],
        batch_size: int,
        generation_config: dict[str, Any],
    ) -> list[str]:
        processor = self.vis_processors["eval"]
        captions: list[str] = []
        for batch_paths in _batched(list(image_paths), batch_size):
            images = [processor(Image.open(path).convert("RGB")) for path in batch_paths]
            tensor = self._torch.stack(images, dim=0).to(self.device)
            samples: dict[str, Any] = {"image": tensor}
            prompt = generation_config.get("prompt")
            if prompt:
                samples["prompt"] = prompt
            with self._torch.no_grad():
                outputs = self.model.generate(
                    samples,
                    **_generate_kwargs(self.model.generate, generation_config),
                )
            captions.extend(str(text).strip() for text in outputs)
        return captions


def _generate_kwargs(generate_fn, config: dict[str, Any]) -> dict[str, Any]:
    supported = set(inspect.signature(generate_fn).parameters)
    all_kwargs = {
        "use_nucleus_sampling": bool(config["use_nucleus_sampling"]),
        "num_beams": int(config["num_beams"]),
        "max_length": int(config["max_length"]),
        "min_length": int(config["min_length"]),
        "top_p": float(config["top_p"]),
        "repetition_penalty": float(config["repetition_penalty"]),
        "length_penalty": float(config["length_penalty"]),
        "num_captions": int(config["num_captions"]),
        "temperature": float(config["temperature"]),
    }
    return {key: value for key, value in all_kwargs.items() if key in supported}


def load_model_from_spec(
    spec: str,
    device: str = "auto",
    local_files_only: bool = False,
    bert_tokenizer_path: str | None = None,
) -> CaptionModel:
    parts = spec.split(":")
    if len(parts) != 3 or parts[0] != "lavis":
        raise ValueError("Task 2 currently supports only lavis:<name>:<model_type> specs.")
    return LavisCaptionGenerator(
        model_name=parts[1],
        model_type=parts[2],
        device=device,
        local_files_only=local_files_only,
        bert_tokenizer_path=bert_tokenizer_path,
    )
