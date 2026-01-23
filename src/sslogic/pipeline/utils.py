"""Utility helpers for the SSLogic orchestration layer."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict


class StructuredOutputError(RuntimeError):
    """Raised when an agent fails to return the expected structured payload."""


_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)```$", re.DOTALL)


def _strip_code_fence(candidate: str) -> str:
    """Remove a surrounding Markdown code fence if present."""

    match = _CODE_FENCE_PATTERN.match(candidate.strip())
    if match:
        return match.group(1).strip()
    return candidate.strip()


def _extract_json_substring(text: str) -> str | None:
    """Attempt to locate a JSON object within free-form text."""

    stack: list[int] = []
    start_idx: int | None = None
    for idx, ch in enumerate(text):
        if ch == "{" and not stack:
            start_idx = idx
            stack.append(idx)
        elif ch == "{" and stack:
            stack.append(idx)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start_idx is not None:
                return text[start_idx : idx + 1]
    return None


def _try_parse_json(text: str) -> Dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _try_literal_eval(text: str) -> Dict[str, Any] | None:
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, dict):
        return json.loads(json.dumps(parsed, ensure_ascii=False))
    return None


def _candidate_payloads(payload: str) -> list[str]:
    cleaned = _strip_code_fence(payload)
    candidates = [cleaned] if cleaned else []
    candidate = _extract_json_substring(cleaned)
    if candidate and candidate not in candidates:
        candidates.append(candidate)
    return candidates


def coerce_json_dict(payload: str) -> Dict[str, Any]:
    """Parse *payload* into a JSON dictionary.

    Agents occasionally wrap JSON in code fences or include leading commentary.
    This helper strips common wrappers and attempts a tolerant parse.
    """

    if not payload:
        raise StructuredOutputError("Empty output from agent; expected JSON object.")

    snippets: list[str] = []
    for candidate in _candidate_payloads(payload):
        for parser in (_try_parse_json, _try_literal_eval):
            parsed = parser(candidate)
            if parsed is not None:
                return parsed
        snippets.append(candidate[:200])

    preview = " | ".join(payload.strip().splitlines()[:5])
    if snippets:
        raise StructuredOutputError(
            "Failed to parse structured payload. Sample snippets: " + preview
        )
    raise StructuredOutputError("Unable to locate JSON object in agent output.")


def dump_json(data: Dict[str, Any]) -> str:
    """Compact ``json.dumps`` wrapper with ASCII-safe output."""

    return json.dumps(data, ensure_ascii=False, indent=2)
