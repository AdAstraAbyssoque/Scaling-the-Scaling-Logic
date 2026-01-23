"""Agent wrappers reused by the SSLogic pipeline."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..ck_pro.ck_main.agent import CKAgent
from ..ck_pro.ck_main.main import default_main_configs
from ..ck_pro.agents.tool import Tool, StopTool


class AgentProxyTool(Tool):
    """Generic wrapper to expose callables as CKAgent tools."""

    def __init__(
        self,
        name: str,
        short_doc: str,
        long_doc: str,
        runner: Callable[..., Any],
    ) -> None:
        super().__init__(name=name)
        self._short_doc = short_doc
        self._long_doc = long_doc
        self._runner = runner

    def get_function_definition(self, short: bool):
        return self._short_doc if short else self._long_doc

    def _execute(self, *args, **kwargs):
        try:
            return self._runner(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            return {
                "error": f"{type(exc).__name__}: {exc}",
            }


def _extract_final_output(session) -> Dict[str, Any]:
    """Pull the `output` and `log` fields from the last session step."""

    step = session.get_current_step()
    end_payload = (step or {}).get("end", {})
    final = end_payload.get("final_results", {})
    return {
        "output": final.get("output", ""),
        "log": final.get("log", ""),
        "raw_step": step,
    }


@dataclass
class AgentRunResult:
    output: str
    log: str
    session: Any
    raw_step: Dict[str, Any]


class GuardedStopTool(StopTool):
    """Stop tool that consults a guard callback before finalizing."""

    def __init__(
        self,
        agent=None,
        guard_fn: Optional[Callable[[], tuple[bool, str]]] = None,
    ):
        super().__init__(agent=agent)
        self.guard_fn = guard_fn

    def _execute(self, output: str, log: str):
        if self.guard_fn is not None:
            allowed, message = self.guard_fn()
            if not allowed:
                return {
                    "error": "stop_guard_blocked",
                    "message": message,
                }
        return super()._execute(output, log)


class CKProAgentWrapper:
    """Light wrapper around :class:`CKAgent` with sane defaults."""

    def __init__(self, name: str, config_overrides: Optional[Dict[str, Any]] = None):
        cfg = copy.deepcopy(default_main_configs)
        cfg.setdefault("name", name)
        if config_overrides:
            for key, value in config_overrides.items():
                if isinstance(value, dict) and key in cfg:
                    cfg[key].update(value)
                else:
                    cfg[key] = value
        cfg.setdefault("name", name)
        self.agent = CKAgent(**cfg)
        self.name = cfg.get("name", name)

    def run(self, task: str, **run_kwargs) -> AgentRunResult:
        session = self.agent.run(task=task, **run_kwargs)
        final_payload = _extract_final_output(session)
        return AgentRunResult(
            output=final_payload.get("output", ""),
            log=final_payload.get("log", ""),
            session=session,
            raw_step=final_payload.get("raw_step", {}),
        )

    def register_tool(self, tool: Tool) -> None:
        """Register an additional tool so the agent can invoke it during execution."""

        agent = self.agent
        existing_names = {existing_tool.name for existing_tool in agent.tools}
        if tool.name not in existing_names:
            agent.tools.append(tool)
        if tool.name not in agent.active_functions:
            agent.active_functions.append(tool.name)
        agent.ACTIVE_FUNCTIONS[tool.name] = tool

    def install_stop_guard(
        self, guard_fn: Callable[[], tuple[bool, str]]
    ) -> None:
        """Replace the default stop tool with a guarded variant."""

        agent = self.agent
        guarded_tool = GuardedStopTool(agent=agent, guard_fn=guard_fn)

        replaced = False
        for idx, tool in enumerate(agent.tools):
            if isinstance(tool, StopTool):
                agent.tools[idx] = guarded_tool
                replaced = True
                break
        if not replaced:
            agent.tools.append(guarded_tool)

        agent.ACTIVE_FUNCTIONS["stop"] = guarded_tool
        if "stop" not in agent.active_functions:
            agent.active_functions.append("stop")
