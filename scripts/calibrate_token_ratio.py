#!/usr/bin/env python3
"""Calibration tool: how many characters per token does our Spanish catalog cost?

`EMBEDDING_FALLBACK_MAX_CHARS` is a character cap standing in for a token limit. The
fallback embedding model accepts 2048 tokens, but we truncate on characters because a
deterministic slice costs zero CPU, adds no dependency, and needs no extra round-trip
in the hot retrieval path. That trade is only sound if the chars-per-token ratio behind
the cap is measured rather than guessed -- and accented Spanish product copy tokenizes
noticeably denser than the 3 chars/token rule of thumb inherited from English.

This script answers the question with the real Gemini tokenizer (`count_tokens`) over
real catalog text, and prints the number to write into `app/core/config.py`.

Operating instructions:
    * Run it manually and rarely: after a large catalog import, after switching
      embedding models, or when revisiting the truncation settings.
    * Copy the recommended value into `EMBEDDING_FALLBACK_MAX_CHARS` in
      `app/core/config.py`. Nothing reads this script at runtime.
    * NEVER call it on the hot path. `count_tokens` is a network round-trip per sample;
      that is fine for a one-off calibration and unacceptable per search request.

Usage:
    python3 scripts/calibrate_token_ratio.py
    python3 scripts/calibrate_token_ratio.py --file data/real_descriptions.jsonl --limit 50
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Import-guarded exactly like `app/services/llm_client.py`: this module must stay
# importable and py_compile-clean on a machine that has never installed the SDK.
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    GENAI_AVAILABLE = False

# The token ceiling of the fallback embedding model (gemini-embedding-001).
FALLBACK_MODEL_TOKEN_LIMIT = 2048

DEFAULT_FIXTURE_PATH = "data/catalog_fixture.json"
DEFAULT_SAMPLE_LIMIT = 25
DEFAULT_MODEL = "gemini-embedding-001"

PLACEHOLDER_API_KEYS = frozenset({
    "your-google-ai-studio-api-key-here",
    "your_api_key_here",
    "test-mock-gemini-key-12345",
    "",
})


def resolve_path(target_path_str: str) -> Path:
    """Resolves a data path against cwd, the repository root and /app.

    Args:
        target_path_str: Relative or absolute path to the sample file.

    Returns:
        The first existing candidate, or the unresolved original when none exist.
    """
    candidate = Path(target_path_str)
    if candidate.is_file():
        return candidate.resolve()

    if not candidate.is_absolute():
        cwd_candidate = (Path.cwd() / target_path_str).resolve()
        if cwd_candidate.is_file():
            return cwd_candidate

        repo_root = Path(__file__).resolve().parent.parent
        repo_candidate = (repo_root / target_path_str).resolve()
        if repo_candidate.is_file():
            return repo_candidate

        docker_candidate = Path("/app") / target_path_str
        if docker_candidate.is_file():
            return docker_candidate.resolve()

    return candidate


def _texts_from_object(obj: Any) -> list[str]:
    """Extracts embeddable text from one decoded JSON object.

    Args:
        obj: A decoded JSON value: a catalog item dict, a plain string, or a list.

    Returns:
        The texts found, in the same order.
    """
    if isinstance(obj, str):
        return [obj] if obj.strip() else []

    if isinstance(obj, list):
        collected: list[str] = []
        for element in obj:
            collected.extend(_texts_from_object(element))
        return collected

    if isinstance(obj, dict):
        # Mirrors django_api._build_embedding_text: what actually gets embedded is the
        # title plus the category plus the description, not the description alone.
        title = str(obj.get("title") or obj.get("name") or "").strip()
        category = str(obj.get("category") or "").strip()
        description = str(obj.get("description") or obj.get("text") or "").strip()
        if not (title or description):
            return []
        parts = [part for part in (title, f"Categoría: {category}" if category else "", description) if part]
        return [". ".join(parts).strip()]

    return []


def load_samples(file_path: str, limit: int) -> list[str]:
    """Loads sample texts from a JSON, JSONL or plain-text file.

    Args:
        file_path: Path to the sample file. JSON documents may be a catalog fixture
            (`{"items": [...]}`), a bare list, or a single object; JSONL is decoded
            line by line; anything else is treated as one text per non-empty line.
        limit: Maximum number of samples to return.

    Returns:
        The collected texts, truncated to `limit`.

    Raises:
        FileNotFoundError: If the file cannot be found.
        ValueError: If the file yields no usable text.
    """
    resolved = resolve_path(file_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de muestras: '{file_path}' ({resolved}).")

    raw = resolved.read_text(encoding="utf-8")
    texts: list[str] = []

    # 1. A single JSON document (the catalog fixture is the default case).
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        document = None

    if document is not None:
        if isinstance(document, dict) and isinstance(document.get("items"), list):
            texts = _texts_from_object(document["items"])
        else:
            texts = _texts_from_object(document)

    # 2. JSONL: one JSON value per line.
    if not texts:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                texts.extend(_texts_from_object(json.loads(line)))
            except json.JSONDecodeError:
                continue

    # 3. Plain text: one sample per non-empty line.
    if not texts:
        texts = [line.strip() for line in raw.splitlines() if line.strip()]

    texts = [text for text in texts if text.strip()]
    if not texts:
        raise ValueError(f"El archivo '{resolved}' no contiene textos utilizables.")

    return texts[: max(1, limit)]


def resolve_api_key() -> str:
    """Returns the configured Gemini API key, or an empty string when unusable."""
    key = (
        os.environ.get("GEMINI_API_KEY", "")
        or os.environ.get("GOOGLE_API_KEY", "")
        or os.environ.get("GEMINI_KEY", "")
        or os.environ.get("GOOGLE_GENAI_API_KEY", "")
    ).strip().strip("'").strip('"').strip()

    if key in PLACEHOLDER_API_KEYS or key.startswith("test-mock"):
        return ""
    return key


def percentile(values: list[float], fraction: float) -> float:
    """Computes a percentile with linear interpolation between adjacent ranks.

    Args:
        values: Non-empty list of measurements.
        fraction: Percentile as a fraction in [0, 1] (0.10 for the 10th percentile).

    Returns:
        The interpolated percentile value.
    """
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = fraction * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]

    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def median(values: list[float]) -> float:
    """Returns the median of a non-empty list of measurements."""
    return percentile(values, 0.5)


def count_tokens(client: Any, model: str, text: str) -> Optional[int]:
    """Counts the tokens of one text with the real Gemini tokenizer.

    Args:
        client: An initialized `genai.Client`.
        model: Model identifier whose tokenizer is used.
        text: The text to measure.

    Returns:
        The token count, or None when the SDK response carries no usable total.
    """
    try:
        result = client.models.count_tokens(model=model, contents=text)
    except Exception as exc:
        print(f"  [warn] count_tokens falló para una muestra: {exc}", file=sys.stderr)
        return None

    total = getattr(result, "total_tokens", None)
    if total is None and isinstance(result, dict):
        total = result.get("total_tokens") or result.get("totalTokens")
    if total is None:
        return None

    try:
        return int(total)
    except (TypeError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    """Builds the command line parser."""
    parser = argparse.ArgumentParser(
        prog="calibrate_token_ratio.py",
        description=(
            "Measures the chars-per-token ratio of real Spanish catalog text with the "
            "Gemini tokenizer and recommends a value for EMBEDDING_FALLBACK_MAX_CHARS. "
            "Calibration tool: run it manually, write the number into app/core/config.py, "
            "and never call it on the hot path."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/calibrate_token_ratio.py\n"
            "  python3 scripts/calibrate_token_ratio.py --limit 50\n"
            "  python3 scripts/calibrate_token_ratio.py --file data/descriptions.jsonl\n"
        ),
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_FIXTURE_PATH,
        help=(
            "JSON, JSONL or plain-text file holding real product descriptions "
            f"(default: {DEFAULT_FIXTURE_PATH})."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help=f"Maximum number of samples to measure (default: {DEFAULT_SAMPLE_LIMIT}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model whose tokenizer is queried (default: {DEFAULT_MODEL}).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Command line arguments, defaulting to `sys.argv[1:]`.

    Returns:
        0 on success, 1 on a usage/data error, 2 when the SDK or API key is missing.
    """
    args = build_parser().parse_args(argv)

    if not GENAI_AVAILABLE:
        print(
            "El SDK 'google-genai' no está instalado, así que no hay tokenizador real "
            "disponible.\nInstálelo con: pip install google-genai",
            file=sys.stderr,
        )
        return 2

    api_key = resolve_api_key()
    if not api_key:
        print(
            "No hay una GEMINI_API_KEY real configurada (se revisaron GEMINI_API_KEY, "
            "GOOGLE_API_KEY, GEMINI_KEY y GOOGLE_GENAI_API_KEY).\n"
            "Exporte una clave válida antes de calibrar: export GEMINI_API_KEY=...",
            file=sys.stderr,
        )
        return 2

    try:
        samples = load_samples(args.file, args.limit)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        print(f"No se pudo inicializar el cliente de Google GenAI: {exc}", file=sys.stderr)
        return 2

    print(f"Fuente : {resolve_path(args.file)}")
    print(f"Modelo : {args.model}")
    print(f"Muestras: {len(samples)}\n")
    print(f"{'#':>3}  {'chars':>7}  {'tokens':>7}  {'chars/token':>12}")
    print("-" * 36)

    ratios: list[float] = []
    for index, text in enumerate(samples, start=1):
        tokens = count_tokens(client, args.model, text)
        if not tokens:
            print(f"{index:>3}  {len(text):>7}  {'n/d':>7}  {'n/d':>12}")
            continue
        ratio = len(text) / tokens
        ratios.append(ratio)
        print(f"{index:>3}  {len(text):>7}  {tokens:>7}  {ratio:>12.3f}")

    if not ratios:
        print(
            "\nNinguna muestra pudo medirse; no hay datos para recomendar un límite.",
            file=sys.stderr,
        )
        return 1

    mean_ratio = sum(ratios) / len(ratios)
    median_ratio = median(ratios)
    p10_ratio = percentile(ratios, 0.10)
    recommended = int(math.floor(FALLBACK_MODEL_TOKEN_LIMIT * p10_ratio))

    print("\nAgregados (chars/token)")
    print(f"  media    : {mean_ratio:.3f}")
    print(f"  mediana  : {median_ratio:.3f}")
    print(f"  p10      : {p10_ratio:.3f}")
    print(f"  mínimo   : {min(ratios):.3f}")
    print(f"  máximo   : {max(ratios):.3f}")

    print(f"\nRecomendación: EMBEDDING_FALLBACK_MAX_CHARS = {recommended}")
    print(
        f"  = floor({FALLBACK_MODEL_TOKEN_LIMIT} tokens x {p10_ratio:.3f} chars/token). "
        "Se usa el percentil 10 y no la media porque un tope de truncado debe aguantar "
        "el texto MÁS denso del catálogo, no el promedio: dimensionarlo con la media "
        "deja que la mitad de los productos supere el límite de tokens del modelo."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
