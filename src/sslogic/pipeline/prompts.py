"""Prompt builders for the SSLogic orchestration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from textwrap import dedent, indent
from typing import Any, Dict, List, Optional

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - dependency check
    raise RuntimeError(
        "PyYAML is required to load prompt templates. Install with `pip install pyyaml`."
    ) from exc


_STAGE_FILES: Dict[str, str] = {
    "propose": "propose.yaml",
    "execute": "execute.yaml",
    "revision": "revision.yaml",
    "validator": "validator.yaml",
    "blind_review": "blind_review.yaml",
}


@lru_cache(maxsize=None)
def _load_stage_config(stage: str) -> Dict[str, Any]:
    if stage not in _STAGE_FILES:
        raise KeyError(f"Unknown stage '{stage}'.")
    config_path = Path(__file__).with_name("prompts") / _STAGE_FILES[stage]
    if not config_path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt file {config_path} must contain a mapping.")
    return data


def _stage_template(stage: str) -> str:
    template = _load_stage_config(stage).get("template")
    if not isinstance(template, str):
        raise TypeError(f"Stage '{stage}' template must be a string.")
    return template


def _stage_schema(stage: str) -> Dict[str, str]:
    schema = _load_stage_config(stage).get("schema")
    if not isinstance(schema, dict):
        raise TypeError(f"Stage '{stage}' schema must be a mapping.")
    title = schema.get("title")
    body = schema.get("body")
    if not isinstance(title, str) or not isinstance(body, str):
        raise TypeError(
            f"Stage '{stage}' schema must include string 'title' and 'body'."
        )
    return {"title": title, "body": body}


def _stage_stop_hint(stage: str) -> str:
    stop_hint = _load_stage_config(stage).get("stop_hint")
    if not isinstance(stop_hint, str):
        raise TypeError(f"Stage '{stage}' stop_hint must be a string.")
    return stop_hint


def _format_errors(errors: List[str]) -> str:
    if not errors:
        return "None"
    return "; ".join(errors)


def _format_preview_block(preview: Optional[str], language: str = "json") -> str:
    if not preview:
        return "No preview available; use File Agent to read the full content."
    return f"```{language}\n{preview}\n```"


def _artifact_section(
    title: str,
    context: Dict[str, Any],
    *,
    include_preview: bool = True,
    preview_key: str = "payload_preview",
) -> str:
    lines = [
        f"- File path: {context.get('artifact_path', 'unknown')}",
        f"- Content hash: {context.get('content_hash', 'unknown')}",
        f"- Known parse issues: {_format_errors(context.get('errors') or [])}",
    ]
    meta = context.get("meta") or {}
    if meta:
        lines.append(f"- Additional metadata: {meta}")
    if include_preview:
        lines.append("- Preview:")
        lines.append(_format_preview_block(context.get(preview_key)))
    body = "\n".join(lines)
    return f"{title}\n{body}" if title else body


def _validator_sections(contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return "No validator feedback available."
    parts: List[str] = []
    for idx, ctx in enumerate(contexts, 1):
        meta = ctx.get("meta") or {}
        validator_name = meta.get("validator_id") or f"Validator #{idx}"
        section = _artifact_section("Reference info", ctx)
        parts.append(f"{idx}. {validator_name}\n" + indent(section, "   "))
    return "\n".join(parts)


def _seed_section(seed_idea: Optional[str]) -> str:
    if seed_idea:
        return f"Seed idea supplied by user:\n{seed_idea.strip()}"
    return (
        "No explicit seed idea supplied. Craft an original baseline algorithm problem "
        "from scratch while ensuring it exercises non-trivial reasoning."
    )


def _schema_section(title: str, schema: str) -> str:
    return dedent(
        f"""
        ## {title}
        - Output must be parseable as **standard JSON** (double-quoted keys; no Python-style quotes or comments).
        - Field names, nesting, and array structure must exactly match the template below; replace only example values without adding/removing/renaming fields.
        - Ensure the final output is **single-line** JSON so it can be written directly to a `.jsonl` file.
        - Recommended: build a `payload` dict first, then call `json.dumps(payload, ensure_ascii=False, separators=(",", ":"))` for a compact form.
        ```json
        {schema}
        ```
        """
    ).strip()


def _stop_section(log_hint: str) -> str:
    return dedent(
        f"""
        ## Submission
        - After the final JSON is ready, run: `print(stop(output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), log="{log_hint}"))`.
        - The system writes `output` to a hash directory for the next stage; ensure it is complete and parseable.
        """
    ).strip()


def render_propose_task(seed_idea: str | None) -> str:
    template = _stage_template("propose")
    schema = _stage_schema("propose")
    return (
        dedent(template)
        .format(
            seed_section=_seed_section(seed_idea),
            schema_section=_schema_section(schema["title"], schema["body"]),
            _stop=_stop_section(_stage_stop_hint("propose")),
        )
        .strip()
    )


def render_execute_task(propose_context: Dict[str, Any]) -> str:
    baseline_section = indent(_artifact_section("", propose_context), "    ")
    template = _stage_template("execute")
    schema = _stage_schema("execute")

    # Add retry feedback if present
    retry_section = ""
    if "retry_feedback" in propose_context:
        retry_attempt = propose_context.get("retry_attempt", 1)
        retry_section = f"""

WARNING: Retry #{retry_attempt}

{propose_context["retry_feedback"]}

Complete the required validation steps before continuing.
"""

    return (
        dedent(template)
        .format(
            baseline_section=baseline_section,
            schema_section=_schema_section(schema["title"], schema["body"]),
            _stop=_stop_section(_stage_stop_hint("execute")),
        )
        .strip()
        + retry_section
    )


def render_revision_task(
    evolved_context: Dict[str, Any],
    validator_context: List[Dict[str, Any]],
    blind_context: Dict[str, Any],
    propose_context: Dict[str, Any],
) -> str:
    evolved_section = indent(_artifact_section("", evolved_context), "    ")
    validator_section = indent(_validator_sections(validator_context), "    ")
    blind_section = indent(
        _artifact_section("", blind_context, preview_key="payload_preview"), "    "
    )
    propose_section = indent(_artifact_section("", propose_context), "    ")
    template = _stage_template("revision")
    schema = _stage_schema("revision")
    return (
        dedent(template)
        .format(
            evolved_section=evolved_section,
            validator_section=validator_section,
            blind_section=blind_section,
            propose_section=propose_section,
            schema_section=_schema_section(schema["title"], schema["body"]),
            _stop=_stop_section(_stage_stop_hint("revision")),
        )
        .strip()
    )


def render_validator_task(
    validator_id: str,
    focus: str,
    evolved_context: Dict[str, Any],
) -> str:
    artifact_section = indent(_artifact_section("", evolved_context), "    ")
    template = _stage_template("validator")
    schema = _stage_schema("validator")
    schema_body = schema["body"].replace("__VALIDATOR_ID__", validator_id)
    return (
        dedent(template)
        .format(
            validator_id=validator_id,
            focus=focus,
            artifact_section=artifact_section,
            schema_section=_schema_section(schema["title"], schema_body),
            _stop=_stop_section(_stage_stop_hint("validator")),
        )
        .strip()
    )


def render_blind_review_task(evolved_context: Dict[str, Any]) -> str:
    """Render blind-solver task with problem statement injected directly."""
    # Extract problem statement from evolved context
    problem_data = evolved_context.get("payload", {})
    evolved_problem = problem_data.get("evolved_problem", {})

    # Format problem statement - match execute.yaml schema
    problem_lines = []
    if evolved_problem.get("title"):
        problem_lines.append(f"**Title**: {evolved_problem['title']}\n")
    if evolved_problem.get("scenario"):
        problem_lines.append(f"**Scenario**: {evolved_problem['scenario']}\n")
    if evolved_problem.get("givens"):
        givens = evolved_problem["givens"]
        if isinstance(givens, list):
            givens_text = "\n".join([f"  - {g}" for g in givens])
            problem_lines.append(f"**Givens**:\n{givens_text}\n")
        else:
            problem_lines.append(f"**Givens**: {givens}\n")
    if evolved_problem.get("question"):
        problem_lines.append(f"**Question**: {evolved_problem['question']}\n")
    if evolved_problem.get("twist"):
        problem_lines.append(f"**Twist**: {evolved_problem['twist']}\n")

    problem_statement = (
        "\n".join(problem_lines) if problem_lines else "No problem statement available."
    )

    template = _stage_template("blind_review")
    schema = _stage_schema("blind_review")
    return (
        dedent(template)
        .format(
            problem_statement=problem_statement,
            schema_section=_schema_section(schema["title"], schema["body"]),
            _stop=_stop_section(_stage_stop_hint("blind_review")),
        )
        .strip()
    )
