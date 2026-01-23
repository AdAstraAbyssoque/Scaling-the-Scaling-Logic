"""Multi-agent orchestration for the SSLogic task synthesis pipeline."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import hashlib
import uuid

from .agents import AgentProxyTool, AgentRunResult, CKProAgentWrapper
from .artifacts import StageArtifact, create_session_dir, persist_artifact
from .prompts import (
    render_blind_review_task,
    render_execute_task,
    render_propose_task,
    render_revision_task,
    render_validator_task,
)
from .utils import StructuredOutputError, coerce_json_dict, dump_json
from ..ck_pro.agents.phoenix_tracer import (
    TaskTracer,
    init_phoenix_tracing,
    is_phoenix_enabled,
)


@dataclass
class ValidatorConfig:
    validator_id: str
    focus: str
    config_overrides: Optional[Dict[str, Any]] = None


@dataclass
class AttemptRecord:
    iteration: int
    evolved: "StageOutput"
    accepted: bool
    raw_outputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageOutput:
    stage: str
    artifact: StageArtifact
    raw_output: str
    log: str
    parsed: Optional[Dict[str, Any]]
    errors: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def reference(self, include_payload: bool = True) -> Dict[str, Any]:
        ref: Dict[str, Any] = {
            "stage": self.stage,
            "artifact_path": _rel_path(self.artifact.path),
            "session_dir": _rel_path(self.artifact.session_dir),
            "content_hash": self.artifact.content_hash,
            "log_path": _rel_path(self.artifact.log_path),
            "errors": list(self.errors),
            "meta": dict(self.meta),
        }
        if include_payload and self.parsed is not None:
            ref["payload"] = self.parsed
        elif include_payload:
            ref["payload"] = None
        return ref


def _find_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    stack: List[int] = []
    start_idx: Optional[int] = None
    for idx, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start_idx = idx
            stack.append(idx)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    return text[start_idx : idx + 1]
    return None


def _rel_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _assert_non_empty_validation(payload: Dict[str, Any]) -> None:
    validation = (
        payload.get("baseline_solution", {}) if isinstance(payload, dict) else {}
    )
    validation_text = ""
    if isinstance(validation, dict):
        validation_text = str(validation.get("validation", ""))
    if not validation_text.strip():
        raise StructuredOutputError(
            "baseline_solution.validation must provide a self-check explanation."
        )


class SSLogicPipeline:
    """Main orchestrator implementing the Propose → Execute flow with inline validation."""

    def __init__(
        self,
        main_agent: Optional[CKProAgentWrapper] = None,
        validator_configs: Optional[List[ValidatorConfig]] = None,
        blind_agent: Optional[CKProAgentWrapper] = None,
        max_iterations: int = 1,
    ) -> None:
        self.phoenix_enabled = is_phoenix_enabled() or init_phoenix_tracing(
            project_name="sslogic"
        )
        self.main_agent = main_agent or CKProAgentWrapper(name="ckpro-main")
        validators_to_use = validator_configs or [
            ValidatorConfig(
                "Schema-Validator", "Schema completeness and statement clarity"
            ),
            ValidatorConfig(
                "Algorithm-Validator",
                "Solvability, algorithmic soundness, and edge-case coverage",
            ),
        ]
        self.validators: List[tuple[ValidatorConfig, CKProAgentWrapper]] = []
        for cfg in validators_to_use:
            wrapper = CKProAgentWrapper(
                name=f"validator-{cfg.validator_id.lower()}",
                config_overrides=cfg.config_overrides,
            )
            self.validators.append((cfg, wrapper))
        self.blind_agent = blind_agent or CKProAgentWrapper(name="ckpro-blind-solver")
        self.max_iterations = max(1, max_iterations)

        self.attempt_history: List[AttemptRecord] = []
        self._session_dir: Optional[Path] = None
        self._inline_counter = 0

        # Track tool invocations during execute stage
        self._validator_tool_called = False
        self._blind_tool_called = False
        self._stop_guard_enabled = False

        self._register_executor_tools()
        self.main_agent.install_stop_guard(self._stop_guard_check)

    # -- stage helpers -------------------------------------------------
    def _stage_run(
        self, agent: CKProAgentWrapper, task: str, stage_label: str
    ) -> AgentRunResult:
        if self.phoenix_enabled:
            session_id = self._session_dir.name if self._session_dir else "unknown"
            safe_stage = stage_label.replace(" ", "_")
            context = TaskTracer(
                task=f"{session_id}:{stage_label}",
                agent_name=f"pipeline.{safe_stage}",
            )
        else:
            context = nullcontext()
        with context:
            return agent.run(task=task)

    def _persist_stage_output(
        self,
        stage_label: str,
        run: AgentRunResult,
        *,
        meta: Optional[Dict[str, Any]] = None,
        extra_checks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
        suffix: str = ".json",
    ) -> StageOutput:
        if self._session_dir is None:
            raise RuntimeError(
                "Session directory is not initialized. Call run() to start a session."
            )
        errors: List[str] = []
        parsed: Optional[Dict[str, Any]] = None
        meta_payload: Dict[str, Any] = dict(meta or {})
        raw_output_obj = run.output
        raw_output: str

        if isinstance(raw_output_obj, dict):
            if "output" in raw_output_obj:
                candidate = raw_output_obj.get("output")
                if isinstance(candidate, dict):
                    raw_output = dump_json(candidate)
                elif candidate is None:
                    raw_output = ""
                else:
                    raw_output = str(candidate)
            else:
                raw_output = dump_json(raw_output_obj)
        elif raw_output_obj is None:
            raw_output = ""
        else:
            raw_output = str(raw_output_obj)

        if raw_output.strip().lower() == "none":
            raw_output = ""

        if not raw_output.strip():
            raw_step = run.raw_step or {}
            fallback_texts: List[tuple[str, str]] = []

            def _add_candidate(value: Any, label: str) -> None:
                if value is None:
                    return
                if isinstance(value, dict):
                    serialized = dump_json(value)
                else:
                    serialized = str(value)
                if not serialized.strip():
                    return
                if any(serialized == existing for _, existing in fallback_texts):
                    return
                fallback_texts.append((label, serialized))

            if isinstance(raw_step, dict):
                end_section = raw_step.get("end", {})
                if isinstance(end_section, dict):
                    final_results = end_section.get("final_results", {})
                    if isinstance(final_results, dict):
                        _add_candidate(
                            final_results.get("output"), "end.final_results.output"
                        )
                        _add_candidate(
                            final_results.get("observation"),
                            "end.final_results.observation",
                        )
                    _add_candidate(end_section.get("output"), "end.output")
                    _add_candidate(end_section.get("observation"), "end.observation")

                action_section = raw_step.get("action", {})
                if isinstance(action_section, dict):
                    _add_candidate(action_section.get("input"), "action.input")
                    _add_candidate(action_section.get("raw_input"), "action.raw_input")
                    _add_candidate(
                        action_section.get("observation"), "action.observation"
                    )

                _add_candidate(raw_step.get("output"), "step.output")
                _add_candidate(raw_step.get("observation"), "step.observation")
                _add_candidate(raw_step.get("log"), "step.log")

            if fallback_texts and "fallback_candidates" not in meta_payload:
                meta_payload["fallback_candidates"] = [
                    label for label, _ in fallback_texts
                ]

            for label, text in fallback_texts:
                extracted = _find_json_block(text)
                if not extracted:
                    continue
                try:
                    coerce_json_dict(extracted)
                except StructuredOutputError:
                    continue
                raw_output = extracted
                meta_payload.setdefault("derived_from", label)
                break

        if not raw_output.strip():
            errors.append("empty-output")
        else:
            try:
                parsed = coerce_json_dict(raw_output)
            except StructuredOutputError as exc:
                errors.append(f"parse_error: {exc}")

        log_obj: Any = run.log
        if (not log_obj) and isinstance(raw_output_obj, dict):
            log_obj = raw_output_obj.get("log")
        if isinstance(log_obj, dict):
            log_text = dump_json(log_obj)
        elif log_obj is None:
            log_text = ""
        else:
            log_text = str(log_obj)

        # For blind review stages, enrich parsed data and raw_output with official answer
        if parsed is not None and "blind-review" in stage_label:
            # Ensure both blind_answer and official_answer are preserved in the artifact
            if "official_answer" in meta_payload:
                # Add official answer to parsed data if not already present from agent output
                if "official_answer" not in parsed or parsed["official_answer"] is None:
                    parsed["official_answer_from_evolved"] = meta_payload[
                        "official_answer"
                    ]

                # Rebuild raw_output to include the enriched data
                raw_output = dump_json(parsed)

        artifact = persist_artifact(
            stage_label,
            raw_output,
            log=log_text,
            suffix=suffix,
            session_dir=self._session_dir,
        )
        if parsed is not None and extra_checks:
            for check in extra_checks:
                try:
                    check(parsed)
                except StructuredOutputError as exc:
                    errors.append(f"schema_error: {exc}")

        return StageOutput(
            stage=stage_label,
            artifact=artifact,
            raw_output=raw_output,
            log=log_text,
            parsed=parsed,
            errors=errors,
            meta=meta_payload,
        )

    def _next_inline_stage_label(self, prefix: str) -> str:
        self._inline_counter += 1
        return f"{prefix}-{self._inline_counter}"

    def _register_executor_tools(self) -> None:
        validator_tool = AgentProxyTool(
            name="validator_agent",
            short_doc=(
                "- def validator_agent(payload: dict | str = None, artifact_path: str | None = None, "
                "validator_id: str | None = None, focus: str | None = None) -> Dict:  # "
                "Run validators on a candidate payload and return structured feedback."
            ),
            long_doc="""
- validator_agent
```python
def validator_agent(
    payload: dict | str = None,
    artifact_path: str | None = None,
    validator_id: str | None = None,
    focus: str | None = None,
) -> dict:
    \"""Run configured validators on a candidate payload and aggregate their feedback.

    Args:
        payload: Candidate JSON payload (dict or JSON string). Provide either `payload` or `artifact_path`.
        artifact_path: Path to a JSON artifact containing the candidate output.
        validator_id: Optional specific validator identifier to invoke.
        focus: Optional focus override to pass to the validator prompt.
    Returns:
        dict: JSON-safe structure summarizing each validator's raw output, parsed payload, and parse errors if any.
    \"""
```
""".strip(),
            runner=self._executor_validator_tool,
        )
        blind_tool = AgentProxyTool(
            name="blind_review_agent",
            short_doc=(
                "- def blind_review_agent(payload: dict | str = None, artifact_path: str | None = None) -> Dict:  # "
                "Run blind review on the candidate payload and return reasoning/answer feedback."
            ),
            long_doc="""
- blind_review_agent
```python
def blind_review_agent(
    payload: dict | str = None,
    artifact_path: str | None = None,
) -> dict:
    \"""Trigger the blind-review agent on a candidate payload.

    Args:
        payload: Candidate JSON payload (dict or JSON string). Provide either `payload` or `artifact_path`.
        artifact_path: Path to a JSON artifact containing the candidate output.
    Returns:
        dict: JSON-safe structure including the blind-review agent's reasoning, submit result, and parse diagnostics.
    \"""
```
""".strip(),
            runner=self._executor_blind_tool,
        )
        self.main_agent.register_tool(validator_tool)
        self.main_agent.register_tool(blind_tool)

    def _stop_guard_check(self) -> tuple[bool, str]:
        if not self._stop_guard_enabled:
            return True, ""
        missing: List[str] = []
        if not self._validator_tool_called:
            missing.append("validator_agent")
        if not self._blind_tool_called:
            missing.append("blind_review_agent")
        if not missing:
            return True, ""
        if len(missing) == 1:
            missing_desc = missing[0]
        else:
            missing_desc = " and ".join(missing)
        message = (
            "Execution incomplete: you must call "
            f"{missing_desc} at least once to validate the problem you generated.\n"
            "Complete the required validation steps, then call stop again to submit the final result."
        )
        return False, message

    def _resolve_inline_payload(
        self,
        payload: Any = None,
        artifact_path: Optional[str] = None,
    ) -> tuple[Dict[str, Any], str, str, Optional[str]]:
        inline_origin: Optional[str] = None
        if payload is not None:
            if isinstance(payload, dict):
                raw_text = dump_json(payload)
                parsed = coerce_json_dict(raw_text)
                inline_origin = "<payload>"
            elif isinstance(payload, str):
                raw_text = payload
                parsed = coerce_json_dict(payload)
                inline_origin = "<payload-str>"
            else:
                raise ValueError("payload must be a dict or JSON string.")
            source_path = self._persist_inline_payload_file(raw_text)
        elif artifact_path:
            path = Path(artifact_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"artifact_path '{artifact_path}' does not exist."
                )
            raw_text = path.read_text(encoding="utf-8")
            parsed = coerce_json_dict(raw_text)
            inline_origin = _rel_path(path)
            source_path = inline_origin or str(path)
        else:
            raise ValueError("Provide either `payload` or `artifact_path`.")
        return parsed, raw_text, source_path, inline_origin

    def _persist_inline_payload_file(self, raw_text: str, suffix: str = ".json") -> str:
        if self._session_dir is None:
            self._session_dir = create_session_dir()
        inline_dir = self._session_dir / "inline"
        inline_dir.mkdir(parents=True, exist_ok=True)
        filename = f"inline-{uuid.uuid4().hex}{suffix}"
        path = inline_dir / filename
        path.write_text(raw_text, encoding="utf-8")
        return _rel_path(path) or str(path)

    def _inline_stage_context(
        self,
        payload: Dict[str, Any],
        raw_text: str,
        source: str,
        inline_origin: Optional[str] = None,
    ) -> Dict[str, Any]:
        preview_limit = 800
        preview = (
            raw_text
            if len(raw_text) <= preview_limit
            else raw_text[:preview_limit] + "\n... (truncated)"
        )
        meta = {"source": source}
        if inline_origin and inline_origin != source:
            meta["inline_origin"] = inline_origin
        return {
            "artifact_path": source,
            "session_dir": _rel_path(self._session_dir) if self._session_dir else "<inline>",
            "content_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "errors": [],
            "meta": meta,
            "has_payload": True,
            "payload_preview": preview,
            "payload": payload,
        }

    def _executor_validator_tool(
        self,
        payload: Any = None,
        artifact_path: Optional[str] = None,
        validator_id: Optional[str] = None,
        focus: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Mark that validator tool was called
        self._validator_tool_called = True

        parsed_payload, raw_text, source, inline_origin = self._resolve_inline_payload(
            payload=payload,
            artifact_path=artifact_path,
        )
        context = self._inline_stage_context(
            parsed_payload, raw_text, source, inline_origin
        )

        results: List[Dict[str, Any]] = []
        for cfg, wrapper in self.validators:
            if validator_id and cfg.validator_id != validator_id:
                continue
            stage_label = self._next_inline_stage_label(
                f"validator-inline-{cfg.validator_id.lower()}"
            )
            task_focus = focus or cfg.focus
            task = render_validator_task(cfg.validator_id, task_focus, context)
            run = self._stage_run(wrapper, task, stage_label)
            raw_output = (
                run.output if isinstance(run.output, str) else dump_json(run.output)
            )
            parsed_output = None
            parse_error = None
            if raw_output and raw_output.strip():
                try:
                    parsed_output = coerce_json_dict(raw_output)
                except StructuredOutputError as exc:
                    parse_error = str(exc)
            results.append(
                {
                    "validator_id": cfg.validator_id,
                    "focus": task_focus,
                    "raw_output": raw_output,
                    "parsed": parsed_output,
                    "parse_error": parse_error,
                    "log": run.log,
                }
            )

        if validator_id and not results:
            return {
                "error": f"No validator matched id '{validator_id}'.",
                "source": source,
            }

        return {
            "source": source,
            "origin": inline_origin,
            "payload_preview": context.get("payload_preview"),
            "validators": results,
        }

    def _executor_blind_tool(
        self,
        payload: Any = None,
        artifact_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Mark that blind review tool was called
        self._blind_tool_called = True

        parsed_payload, raw_text, source, inline_origin = self._resolve_inline_payload(
            payload=payload,
            artifact_path=artifact_path,
        )
        context = self._inline_stage_context(
            parsed_payload, raw_text, source, inline_origin
        )
        context["payload"] = parsed_payload
        trimmed = {"evolved_problem": parsed_payload.get("evolved_problem", {})}
        problem_preview = dump_json(trimmed)
        if len(problem_preview) > 800:
            problem_preview = problem_preview[:800] + "\n... (truncated)"
        context["problem_preview"] = problem_preview

        stage_label = self._next_inline_stage_label("blind-solver-inline")
        task = render_blind_review_task(context)
        run = self._stage_run(self.blind_agent, task, stage_label)
        raw_output = (
            run.output if isinstance(run.output, str) else dump_json(run.output)
        )
        parsed_output = None
        parse_error = None
        if raw_output and raw_output.strip():
            try:
                parsed_output = coerce_json_dict(raw_output)
            except StructuredOutputError as exc:
                parse_error = str(exc)

        return {
            "source": source,
            "origin": inline_origin,
            "payload_preview": context.get("payload_preview"),
            "raw_output": raw_output,
            "parsed": parsed_output,
            "parse_error": parse_error,
            "log": run.log,
        }

    @staticmethod
    def _stage_prompt_context(
        stage: StageOutput, preview_limit: int = 0
    ) -> Dict[str, Any]:
        preview: Optional[str] = None
        if preview_limit and stage.parsed is not None:
            candidate = dump_json(stage.parsed)
            if len(candidate) > preview_limit:
                preview = candidate[:preview_limit] + "\n... (truncated)"
            else:
                preview = candidate
        return {
            "artifact_path": _rel_path(stage.artifact.path),
            "session_dir": _rel_path(stage.artifact.session_dir),
            "content_hash": stage.artifact.content_hash,
            "errors": list(stage.errors),
            "meta": dict(stage.meta),
            "has_payload": stage.parsed is not None,
            "payload_preview": preview,
        }

    def run_propose(self, seed_idea: Optional[str]) -> StageOutput:
        task = render_propose_task(seed_idea)
        run = self._stage_run(self.main_agent, task, "propose")
        stage_output = self._persist_stage_output(
            "propose",
            run,
            extra_checks=[_assert_non_empty_validation],
        )
        return stage_output

    def run_execute(
        self, propose_stage: StageOutput, iteration: int = 1, max_retries: int = 3
    ) -> StageOutput:
        context = self._stage_prompt_context(propose_stage, preview_limit=800)
        stage_label = f"execute-iter{iteration}"

        self._stop_guard_enabled = True

        try:
            for retry_attempt in range(max_retries):
                # Reset tool call tracking before each run
                self._validator_tool_called = False
                self._blind_tool_called = False

                task = render_execute_task(context)
                run = self._stage_run(self.main_agent, task, stage_label)

                # Check if required tools were called
                if not self._validator_tool_called or not self._blind_tool_called:
                    missing_tools = []
                    if not self._validator_tool_called:
                        missing_tools.append("validator_agent")
                    if not self._blind_tool_called:
                        missing_tools.append("blind_review_agent")

                    feedback_msg = (
                        "Execution incomplete: you must call "
                        f"{' and '.join(missing_tools)} at least once to validate the problem you generated. "
                        "Use these tools to check your output, then submit the final answer."
                    )

                    # If this is not the last retry, continue the agent with feedback
                    if retry_attempt < max_retries - 1:
                        # Update context with feedback for next iteration
                        context["retry_feedback"] = feedback_msg
                        context["retry_attempt"] = retry_attempt + 1
                        continue
                    else:
                        # On last retry, raise an error
                        raise RuntimeError(
                            f"Execute stage failed after {max_retries} attempts: "
                            f"Agent did not call required tools ({', '.join(missing_tools)})"
                        )

                # If tools were called, return the stage output
                stage_output = self._persist_stage_output(
                    stage_label,
                    run,
                    meta={"iteration": iteration, "retry_attempt": retry_attempt},
                )
                return stage_output
        finally:
            self._stop_guard_enabled = False

        # This should not be reached, but just in case
        raise RuntimeError(f"Execute stage failed after {max_retries} attempts")

    def run_revision(
        self,
        evolved_stage: StageOutput,
        validator_feedback: List[StageOutput],
        blind_feedback: StageOutput,
        propose_stage: StageOutput,
        iteration: int,
    ) -> StageOutput:
        evolved_context = self._stage_prompt_context(evolved_stage, preview_limit=600)
        validator_context = [
            self._stage_prompt_context(stage, preview_limit=400)
            for stage in validator_feedback
        ]
        blind_context = self._stage_prompt_context(blind_feedback, preview_limit=400)
        propose_context = self._stage_prompt_context(propose_stage, preview_limit=600)
        task = render_revision_task(
            evolved_context,
            validator_context,
            blind_context,
            propose_context,
        )
        stage_label = f"revision-iter{iteration}"
        run = self._stage_run(self.main_agent, task, stage_label)
        return self._persist_stage_output(
            stage_label, run, meta={"iteration": iteration}
        )

    def run_validators(
        self, evolved_stage: StageOutput, iteration: int
    ) -> List[StageOutput]:
        feedback: List[StageOutput] = []
        evolved_context = self._stage_prompt_context(evolved_stage, preview_limit=800)
        for cfg, wrapper in self.validators:
            task = render_validator_task(cfg.validator_id, cfg.focus, evolved_context)
            stage_label = f"validator-{cfg.validator_id.lower()}-iter{iteration}"
            run = self._stage_run(wrapper, task, stage_label)
            stage_output = self._persist_stage_output(
                stage_label,
                run,
                meta={
                    "validator_id": cfg.validator_id,
                    "focus": cfg.focus,
                    "iteration": iteration,
                },
            )
            feedback.append(stage_output)
        return feedback

    # -- public orchestration -----------------------------------------
    def run(self, seed_idea: Optional[str] = None) -> Dict[str, Any]:
        self._session_dir = create_session_dir()
        session_dir = self._session_dir

        pipeline_context = (
            TaskTracer(
                task=f"{session_dir.name}:pipeline",
                agent_name="pipeline.root",
            )
            if self.phoenix_enabled
            else nullcontext()
        )

        with pipeline_context:
            propose_stage = self.run_propose(seed_idea)
            final_stage = self.run_execute(propose_stage, iteration=1)

        attempt = AttemptRecord(iteration=1, evolved=final_stage, accepted=True)
        self.attempt_history.append(attempt)

        final_answer_path = session_dir / "final_answer.json"
        final_output_text = final_stage.raw_output or ""
        if (not final_output_text.strip()) and final_stage.parsed is not None:
            final_output_text = dump_json(final_stage.parsed)
        final_answer_path.write_text(final_output_text, encoding="utf-8")

        def _stage_ref(
            stage_output: StageOutput, include_payload: bool = True
        ) -> Dict[str, Any]:
            return stage_output.reference(include_payload=include_payload)

        return {
            "status": "completed",
            "phoenix_enabled": bool(self.phoenix_enabled),
            "session_dir": _rel_path(session_dir),
            "session_id": session_dir.name,
            "final_answer_path": _rel_path(final_answer_path),
            "propose": _stage_ref(propose_stage, include_payload=True),
            "final_evolved": _stage_ref(final_stage, include_payload=True),
            "attempts_made": len(self.attempt_history),
            "history": [
                {
                    "iteration": att.iteration,
                    "accepted": att.accepted,
                    "evolved_artifact": att.evolved.reference(include_payload=False),
                }
                for att in self.attempt_history
            ],
        }


def format_pipeline_result(result: Dict[str, Any]) -> str:
    """Convenience helper for pretty-printing pipeline output."""

    return dump_json(result)
