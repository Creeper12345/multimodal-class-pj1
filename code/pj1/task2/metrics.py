"""Caption evaluation utilities for Task 2."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
import tempfile
from typing import Iterable, Sequence


BLEU_METHODS = ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4")
SUPPORTED_METRICS = ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE")
JAVA_REQUIRED_METRICS = ("METEOR", "SPICE")
JAVA_METRICS_ENV = "PJ1_ENABLE_JAVA_METRICS"
STANFORD_CORENLP_3_4_1_JAR = "stanford-corenlp-3.4.1.jar"
PUNCTUATIONS = ("''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", ".", "?", "!", ",", ":", "-", "--", "...")


@dataclass(frozen=True)
class CaptionMetrics:
    scores: dict[str, float]
    warnings: tuple[str, ...] = ()


def _normalize_caption_records(
    records: dict[int, list[dict[str, str]]],
) -> dict[int, list[dict[str, str]]]:
    normalized: dict[int, list[dict[str, str]]] = {}
    for image_id, items in records.items():
        normalized[int(image_id)] = [{"caption": str(item["caption"]).strip()} for item in items]
    return normalized


def _identity_tokenize(
    records: dict[int, list[dict[str, str]]],
) -> dict[int, list[str]]:
    return {
        image_id: [" ".join(item["caption"].lower().split()) for item in items]
        for image_id, items in records.items()
    }


def _tokenize_records(
    records: dict[int, list[dict[str, str]]],
    tokenizer_fallback: str,
) -> tuple[dict[int, list[str]], list[str]]:
    warnings: list[str] = []
    try:
        tokenized = _project_ptb_tokenize(records)
        return tokenized, warnings
    except Exception as exc:
        if tokenizer_fallback == "error":
            raise RuntimeError(
                "PTBTokenizer failed. Install Java or rerun with tokenizer fallback enabled."
            ) from exc
        warnings.append(f"PTBTokenizer unavailable; falling back to identity tokenization: {exc}")
        return _identity_tokenize(records), warnings


def _project_ptb_tokenize(records: dict[int, list[dict[str, str]]]) -> dict[int, list[str]]:
    from pycocoevalcap.tokenizer import ptbtokenizer

    jar_dir = os.path.dirname(os.path.abspath(ptbtokenizer.__file__))
    image_ids = [image_id for image_id, items in records.items() for _ in range(len(items))]
    sentences = "\n".join(
        item["caption"].replace("\n", " ")
        for _, items in records.items()
        for item in items
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        dir=tempfile.gettempdir(),
        encoding="utf-8",
    ) as handle:
        handle.write(sentences)
        tmp_path = handle.name

    cmd = [
        "java",
        "-cp",
        STANFORD_CORENLP_3_4_1_JAR,
        "edu.stanford.nlp.process.PTBTokenizer",
        "-preserveLines",
        "-lowerCase",
        tmp_path,
    ]
    try:
        token_lines = subprocess.check_output(
            cmd,
            cwd=jar_dir,
            stderr=subprocess.STDOUT,
        ).decode("utf-8")
    finally:
        os.remove(tmp_path)

    lines = token_lines.split("\n")
    tokenized: dict[int, list[str]] = {}
    for image_id, line in zip(image_ids, lines):
        tokenized.setdefault(int(image_id), []).append(
            " ".join(word for word in line.rstrip().split(" ") if word not in PUNCTUATIONS)
        )
    return tokenized


def _metric_scorers(requested: set[str]) -> list[tuple[object, Sequence[str]]]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.spice.spice import Spice

    scorers: list[tuple[object, Sequence[str]]] = []
    if any(metric in requested for metric in BLEU_METHODS):
        scorers.append((Bleu(4), BLEU_METHODS))
    if "METEOR" in requested:
        scorers.append((Meteor(), ("METEOR",)))
    if "ROUGE_L" in requested:
        scorers.append((Rouge(), ("ROUGE_L",)))
    if "CIDEr" in requested:
        scorers.append((Cider(), ("CIDEr",)))
    if "SPICE" in requested:
        scorers.append((Spice(), ("SPICE",)))
    return scorers


def _drop_unavailable_metrics(requested: list[str], strict: bool) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    available = list(requested)
    unavailable: list[str] = []
    java_metrics_requested = [metric for metric in available if metric in JAVA_REQUIRED_METRICS]
    java_metrics_enabled = os.environ.get(JAVA_METRICS_ENV) == "1"
    if java_metrics_requested and not java_metrics_enabled:
        unavailable.extend(java_metrics_requested)
    elif shutil.which("java") is None:
        unavailable.extend(metric for metric in available if metric in JAVA_REQUIRED_METRICS)
    if "SPICE" in available and "SPICE" not in unavailable and not _spice_dependencies_available():
        unavailable.append("SPICE")

    if unavailable and strict:
        raise RuntimeError(
            "Required local dependencies are unavailable for these caption metrics: "
            + ", ".join(unavailable)
        )
    for metric in unavailable:
        available.remove(metric)
    if unavailable:
        warnings.append(
            "Skipped caption metrics with unavailable local dependencies: "
            + ", ".join(unavailable)
            + f". Set {JAVA_METRICS_ENV}=1 only when Java metric dependencies are ready."
        )
    return available, warnings


def _spice_dependencies_available() -> bool:
    try:
        from pycocoevalcap.spice import get_stanford_models
    except Exception:
        return False
    spice_dir = os.path.dirname(os.path.abspath(get_stanford_models.__file__))
    jar_path = os.path.join(spice_dir, "lib", "stanford-corenlp-3.6.0.jar")
    models_path = os.path.join(spice_dir, "lib", "stanford-corenlp-3.6.0-models.jar")
    return os.path.exists(jar_path) and os.path.exists(models_path)


def evaluate_captions(
    references: dict[int, list[dict[str, str]]],
    predictions: dict[int, list[dict[str, str]]],
    metrics: Iterable[str],
    tokenizer_fallback: str = "identity",
    strict: bool = False,
) -> CaptionMetrics:
    requested = [metric for metric in metrics if metric in SUPPORTED_METRICS]
    if not requested:
        raise ValueError("No supported caption metrics were requested.")
    requested, availability_warnings = _drop_unavailable_metrics(requested, strict=strict)
    if not requested:
        raise ValueError("No requested caption metrics are available in this environment.")

    reference_ids = set(references)
    prediction_ids = set(predictions)
    if reference_ids != prediction_ids:
        missing = sorted(reference_ids - prediction_ids)[:5]
        extra = sorted(prediction_ids - reference_ids)[:5]
        raise ValueError(
            f"Prediction image ids do not match references. Missing={missing}, extra={extra}"
        )

    normalized_references = _normalize_caption_records(references)
    normalized_predictions = _normalize_caption_records(predictions)
    tokenized_references, warnings = _tokenize_records(
        normalized_references,
        tokenizer_fallback=tokenizer_fallback,
    )
    warnings.extend(availability_warnings)
    tokenized_predictions, tokenization_warnings = _tokenize_records(
        normalized_predictions,
        tokenizer_fallback=tokenizer_fallback,
    )
    warnings.extend(tokenization_warnings)

    scores: dict[str, float] = {}
    for scorer, method_names in _metric_scorers(set(requested)):
        try:
            score, _ = scorer.compute_score(tokenized_references, tokenized_predictions)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to compute {','.join(method_names)}.") from exc
            warnings.append(f"Skipped {','.join(method_names)} because scoring failed: {exc}")
            continue

        if len(method_names) == 1:
            metric_name = method_names[0]
            scores[metric_name] = float(score)
            continue

        for metric_name, metric_score in zip(method_names, score):
            if metric_name in requested:
                scores[metric_name] = float(metric_score)

    return CaptionMetrics(scores=scores, warnings=tuple(warnings))
