"""Package PJ1 submission files while excluding large local artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
import zipfile


DEFAULT_OUTPUT = "handin/pj1_submission.zip"
DEFAULT_EXCLUDES = [
    ".DS_Store",
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".venv/",
    "venv/",
    "env/",
    ".idea/",
    ".vscode/",
    "datasets/",
    "outputs/**/cache/",
    "outputs/**/logs/",
    "outputs/**/*.npy",
    "outputs/**/*.npz",
    "outputs/**/*.pt",
    "outputs/**/*.pth",
    "outputs/**/*.ckpt",
    "outputs/**/*.safetensors",
    "handin/",
    "*.zip",
    "*.tar.gz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--ignore-file", default=".submissionignore")
    parser.add_argument(
        "--include-assignment-pdf",
        action="store_true",
        help="Include PJ1 assignment PDF in the archive if present.",
    )
    return parser.parse_args()


def load_patterns(root: Path, ignore_file: str) -> list[str]:
    path = root / ignore_file
    if not path.exists():
        return list(DEFAULT_EXCLUDES)

    patterns: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def is_ignored(relative_path: str, is_dir: bool, patterns: list[str]) -> bool:
    path = relative_path.replace("\\", "/")
    path_for_dir = f"{path}/" if is_dir and not path.endswith("/") else path

    for pattern in patterns:
        normalized = pattern.replace("\\", "/")
        if normalized.endswith("/"):
            prefix = normalized.rstrip("/")
            if path == prefix or path.startswith(prefix + "/") or path_for_dir.startswith(normalized):
                return True
        if fnmatch(path, normalized) or fnmatch(path_for_dir, normalized):
            return True
    return False


def iter_submission_files(root: Path, patterns: list[str], include_assignment_pdf: bool):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if not include_assignment_pdf and relative == "PJ1(1).pdf":
            continue
        if is_ignored(relative, path.is_dir(), patterns):
            continue
        if path.is_file():
            yield path, relative


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    patterns = load_patterns(root, args.ignore_file)

    output.parent.mkdir(parents=True, exist_ok=True)
    files = list(iter_submission_files(root, patterns, args.include_assignment_pdf))

    manifest_lines = [
        "PJ1 submission manifest",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Files:",
    ]
    manifest_lines.extend(f"- {relative}" for _, relative in files)
    manifest = "\n".join(manifest_lines) + "\n"

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, relative in files:
            zf.write(path, arcname=relative)
        zf.writestr("SUBMISSION_MANIFEST.txt", manifest)

    print(f"Wrote {output}")
    print(f"Packaged {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

