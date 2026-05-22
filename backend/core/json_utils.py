import json
import re
from typing import Any


def extract_json_from_text(text: str) -> str | None:
    if not text:
        return None

    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    starts = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx != -1]
    if not starts:
        return None
    start = min(starts)

    opening = cleaned[start]
    closing = "}" if opening == "{" else "]"
    end = cleaned.rfind(closing)
    if end == -1 or end <= start:
        return None
    return cleaned[start : end + 1].strip()


def repair_json_like_text(text: str) -> str:
    if not text:
        return text
    repaired = extract_json_from_text(text) or text.strip()
    repaired = repaired.replace("\ufeff", "")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def safe_json_loads(text: str) -> Any:
    if isinstance(text, (dict, list)):
        return text

    attempts = [text, extract_json_from_text(text), repair_json_like_text(text)]
    last_error: Exception | None = None
    for candidate in attempts:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception as exc:  # noqa: BLE001 - preserve fallback behavior for LLM output
            last_error = exc
    raise ValueError(f"Invalid JSON output: {last_error}")


def normalize_json_result(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"items": value}
    return {"value": value}
