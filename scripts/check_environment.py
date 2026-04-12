"""Check the local runtime for PJ1 before starting a task run."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.pj1.runtime import configure_runtime_env, detect_system_info, detect_torch_runtime


DEFAULT_IMPORTS = (
    "numpy",
    "PIL",
    "torch",
    "torchvision",
    "decord",
    "transformers",
    "lavis",
    "open_clip",
    "pycocoevalcap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/task1_retrieval")
    parser.add_argument("--annotation", default="datasets/annotations/captions_val2017.json")
    parser.add_argument("--image-dir", default="datasets/val2017")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--imports", nargs="*", default=list(DEFAULT_IMPORTS))
    return parser.parse_args()


def check_imports(names: list[str]) -> dict[str, dict[str, str | bool | None]]:
    results: dict[str, dict[str, str | bool | None]] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            results[name] = {
                "ok": True,
                "version": getattr(module, "__version__", None),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            results[name] = {
                "ok": False,
                "version": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return results


def main() -> int:
    args = parse_args()
    runtime_paths = configure_runtime_env(args.output_dir)

    payload = {
        "system": detect_system_info(),
        "runtime_paths": runtime_paths.as_dict(),
        "torch": detect_torch_runtime(),
        "imports": check_imports(args.imports),
    }

    if args.check_data:
        annotation = Path(args.annotation)
        image_dir = Path(args.image_dir)
        payload["data"] = {
            "annotation_exists": annotation.exists(),
            "image_dir_exists": image_dir.exists(),
            "annotation": str(annotation),
            "image_dir": str(image_dir),
        }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
