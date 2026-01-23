from __future__ import annotations

import copy
import json
import random
from collections import Counter
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    TimeoutError as FuturesTimeoutError,
)
import os
import multiprocessing as mp
import ast
from dataclasses import dataclass, field
from datetime import datetime
import traceback
from pathlib import Path
from queue import Empty
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from ..agents.phoenix_tracer import (
    TaskTracer,
    ToolTracer,
    init_phoenix_tracing,
    is_phoenix_enabled,
)
from ..agents.tool import StopTool, Tool
from ..agents.model import LLM
from ..ck_main.agent import CKAgent
from ..ck_main.main import default_main_configs

from .experience import ExperienceManager
from .experience_curator import auto_curate_experience_pool
from .prompts import (
    render_blind_prompt,
    render_main_task,
    render_question_quality_prompt,
    render_validator_builder_prompt,
)

BLIND_AGENT_SYSTEM_PROMPT = """You are a math/logic problem-solving agent. Your task is to read the problem carefully and solve it independently.

IMPORTANT:
- After finishing reasoning and giving the answer, you must call the `stop` tool to end the task.
- Do not output only the answer without calling the stop tool.
- The stop tool is the only correct way to end the task."""

QUALITY_AGENT_SYSTEM_PROMPT = """You are a problem-quality review agent. Your task is to review the problem quality and output the review result in JSON format.

IMPORTANT:
- After completing the review, you must call the `stop` tool to end the task.
- Do not output only JSON without calling the stop tool.
- The stop tool is the only correct way to end the task."""

BLIND_ATTEMPT_TIMEOUT_BUFFER = 2.0
BLIND_ATTEMPT_DEFAULT_TIMEOUT = 600  # 与 exec_timeout_wo_call 对齐
MP_CONTEXT = mp.get_context("spawn")
VALIDATOR_EXEC_TIMEOUT = float(os.environ.get("CKPRO_VALIDATOR_TIMEOUT", "120"))


def _normalize_answer(text: Any) -> str:
    """提取并标准化 \\boxed{} 中的答案"""
    if text is None:
        return ""
    value = str(text).strip()

    # 提取 \boxed{...} 中的内容
    import re

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", value)
    if boxed_match:
        value = boxed_match.group(1)

    return value.strip().lower()


def _ensure_json_dict(text: str | None) -> Dict[str, Any]:
    """
    确保文本可以解析为 JSON 字典

    Args:
        text: 要解析的文本，可能是 None

    Returns:
        解析后的字典

    Raises:
        ValueError: 如果无法解析为字典
    """
    if text is None:
        raise ValueError(
            "输出内容为 None。请确保 agent 在最后一步（end 阶段）正确输出了包含 'output' 和 'log' 字段的字典。"
        )

    if not isinstance(text, str):
        # 如果已经是字典，直接返回
        if isinstance(text, dict):
            return text
        raise ValueError(
            f"输出内容类型错误：期望字符串或字典，实际为 {type(text)}。内容：{text}"
        )

    # 去除首尾空白
    text = text.strip()
    if not text:
        raise ValueError(
            "输出内容为空字符串。请确保 agent 在最后一步（end 阶段）正确输出了包含 'output' 和 'log' 字段的字典。"
        )

    try:
        return json.loads(text)
    except Exception as exc:  # pragma: no cover - fallback to literal_eval
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        raise ValueError(f"无法解析为 JSON：{exc}\n原始内容:\n{text}") from exc


def _clone_agent_configs(
    name: str, call_target: Optional[str] = None
) -> Dict[str, Any]:
    cfg = copy.deepcopy(default_main_configs)
    cfg.setdefault("name", name)
    if call_target:
        cfg.setdefault("model", {})
        cfg["model"]["call_target"] = call_target
    return cfg


def _disable_ckagent_default_tools(agent: CKAgent) -> None:
    """
    移除 CKAgent 中默认启用但在演化流程中不允许使用的工具。
    主要禁用 ask_llm、web_agent、file_agent 相关功能，防止兜底提问或调用外部子代理。
    """

    blocked_names = {"ask_llm", "web_agent", "file_agent"}

    # 清理工具列表
    if hasattr(agent, "tools"):
        agent.tools = [
            tool
            for tool in agent.tools
            if getattr(tool, "name", None) not in blocked_names
        ]

    # 清理 active functions 与 ACTIVE_FUNCTIONS 注册表
    if hasattr(agent, "active_functions"):
        agent.active_functions = [
            fn for fn in agent.active_functions if fn not in blocked_names
        ]
    if hasattr(agent, "ACTIVE_FUNCTIONS"):
        for name in blocked_names:
            agent.ACTIVE_FUNCTIONS.pop(name, None)

    # 移除子代理名称，防止再次调用
    if hasattr(agent, "sub_agent_names"):
        agent.sub_agent_names = [
            name for name in agent.sub_agent_names if name not in blocked_names
        ]

    # 解除对 ask_llm / web_agent / file_agent 的引用
    if hasattr(agent, "tool_ask_llm"):
        agent.tool_ask_llm = None
    if hasattr(agent, "web_agent"):
        agent.web_agent = None
    if hasattr(agent, "file_agent"):
        agent.file_agent = None


def _configure_agent_system_prompt(
    agent: CKAgent, system_prompt: str, debug: bool = False
) -> None:
    """
    为 agent 配置自定义 system prompt，通过重写 step_call 方法实现

    这种方式更优雅，符合 ckpro 的设计模式：
    1. 在 agent 创建时一次性配置，而不是每次调用时动态设置
    2. 通过重写 step_call 方法，在调用 LLM 前注入 system prompt
    3. 保持模板系统的完整性，不破坏原有的模板结构

    Args:
        agent: 要配置的 CKAgent
        system_prompt: 要添加的 system prompt（会追加到原有的 system prompt 后面）
        debug: 是否启用调试模式，打印完整的 prompt 内容（可通过环境变量 DEBUG_AGENT_PROMPT 控制）
    """
    if not system_prompt:
        return

    # 从环境变量读取调试模式设置（如果未显式指定）
    if not debug:
        from ..agents.utils import GET_ENV_VAR

        debug_val = GET_ENV_VAR("DEBUG_AGENT_PROMPT", df=False)
        debug = bool(debug_val) if debug_val else False

    # 保存原始的 step_call 方法
    original_step_call = agent.step_call

    def step_call_with_system_prompt(self, messages, session, model=None):
        """包装 step_call，在调用 LLM 前注入 system prompt

        注意：函数签名必须与原始的 step_call 方法完全一致：
        def step_call(self, messages, session, model=None)
        """
        # 确保 messages 是列表格式
        if not isinstance(messages, list):
            messages = [messages] if messages else []

        # 找到或创建 system 消息
        system_msg_found = False
        original_system_content = None
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                original_system_content = msg["content"]
                msg["content"] = f"{msg['content']}\n\n{system_prompt}"
                system_msg_found = True
                break

        # 如果没有 system 消息，在开头添加一个
        if not system_msg_found:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # 调试模式：打印完整的 prompt
        if debug:
            import json

            print("\n" + "=" * 80)
            print(f"[DEBUG] Agent: {self.name}")
            print("=" * 80)
            print("\n[完整 Prompt 消息列表]:")
            print(json.dumps(messages, ensure_ascii=False, indent=2))
            print("\n[System Prompt 详情]:")
            if system_msg_found:
                print(f"原始英文 System Prompt:\n{original_system_content}")
                print(f"\nAppended system prompt:\n{system_prompt}")
                print(f"\n合并后的完整 System Prompt:\n{messages[0]['content']}")
            else:
                print(f"新增的 System Prompt:\n{system_prompt}")
            print("=" * 80 + "\n")

        # 调用原始的 step_call（注意：这里直接调用原始方法，不需要 self）
        return original_step_call(messages, session, model)

    # 重写 step_call 方法（直接绑定到 agent 实例）
    agent.step_call = step_call_with_system_prompt.__get__(agent, type(agent))


def _run_agent(
    agent: CKAgent,
    task: str,
    session: Any | None = None,
) -> Dict[str, Any]:
    """
    运行 agent 执行任务

    Args:
        agent: 要运行的 CKAgent（应该在创建时已配置好 system prompt）
        task: 任务描述
        session: 可选的会话对象
    """
    session_obj = cast(Any, agent.run(task=task, session=session))
    last_step = session_obj.get_current_step()
    end_block = (last_step or {}).get("end") or {}
    final_results = end_block.get("final_results") or {}
    output = final_results.get("output", "")
    log = final_results.get("log", "")

    # 如果 output 为 None 或空字符串，尝试从 end_block 的 code 中提取
    if (
        not output
        or output == "None"
        or (isinstance(output, str) and output.strip() == "")
    ):
        try:
            end_code = end_block.get("code", "")
            if end_code:
                # 清理 code，移除可能的 markdown 代码块标记
                cleaned_code = end_code.strip()
                if cleaned_code.startswith("```"):
                    # 移除 ```python 或 ``` 标记
                    lines = cleaned_code.split("\n")
                    if lines[0].startswith("```"):
                        cleaned_code = "\n".join(lines[1:])
                    if cleaned_code.endswith("```"):
                        cleaned_code = cleaned_code[:-3].strip()

                # 尝试从 code 中 eval 出 final_results
                parsed = None
                try:
                    parsed = eval(cleaned_code)
                except Exception:
                    # 尝试移除所有换行符后再次 eval（处理 LLM 在字符串中意外换行的情况）
                    try:
                        parsed = eval(cleaned_code.replace("\n", ""))
                    except Exception:
                        pass

                if parsed and isinstance(parsed, dict) and "output" in parsed:
                    extracted_output = parsed.get("output", "")
                    if extracted_output and extracted_output != "None":
                        output = extracted_output
                    if not log:
                        log = parsed.get("log", "")
        except Exception as e:
            # 如果 eval 失败，尝试从 code 中提取 JSON
            try:
                import json
                import re

                # 尝试从 code 中提取 JSON 字符串（更宽松的匹配）
                # 匹配包含 "output" 和 "log" 的字典
                json_pattern = (
                    r'\{[^{}]*(?:"output"[^{}]*"log"|"log"[^{}]*"output")[^{}]*\}'
                )
                json_match = re.search(json_pattern, end_code, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    if "output" in parsed:
                        extracted_output = parsed.get("output", "")
                        if extracted_output and extracted_output != "None":
                            output = extracted_output
                        if not log:
                            log = parsed.get("log", "")
            except Exception:
                pass

    if output is None:
        print(
            f"[WARN] _run_agent failed to extract output. end_code preview: {end_block.get('code', '')[:200]}..."
        )

    return {
        "output": output,
        "log": log,
        "session": session_obj,
        "raw_step": last_step,
    }


def _blind_attempt_subprocess_runner(
    queue: mp.Queue,
    prompt: str,
    sample_answer: str,
    call_target: str,
) -> None:
    """在子进程中运行盲评，便于超时直接 kill"""
    try:
        agent = CKAgent(**_clone_agent_configs("seed-blind-worker", call_target))
        _configure_agent_system_prompt(agent, BLIND_AGENT_SYSTEM_PROMPT)
        agent_result = _run_agent(agent, prompt)
        raw_output = agent_result["output"]

        normalized_pred = _normalize_answer(raw_output)
        normalized_gold = _normalize_answer(sample_answer)
        matched = normalized_pred == normalized_gold and normalized_pred != ""
        queue.put(
            (
                "ok",
                {
                    "raw_output": raw_output,
                    "extracted_answer": normalized_pred,
                    "expected_answer": normalized_gold,
                    "matched": matched,
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        queue.put(
            (
                "error",
                exc.__class__.__name__,
                str(exc),
                traceback.format_exc(),
            )
        )


def _validator_subprocess_runner(queue: mp.Queue, code: str, inputs: Any) -> None:
    """在子进程中运行 validator，防止长时间阻塞主进程"""
    try:
        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        fn = namespace.get("solution")
        if not callable(fn):
            queue.put(
                (
                    "error",
                    "InvalidValidator",
                    "validator_code 必须定义可调用的 solution(inputs) 函数。",
                    "",
                )
            )
            return
        result = fn(inputs)
        queue.put(("ok", result))
    except Exception as exc:  # noqa: BLE001
        queue.put(
            (
                "error",
                exc.__class__.__name__,
                str(exc),
                traceback.format_exc(),
            )
        )


class ProxyTool(Tool):
    def __init__(self, name: str, short_doc: str, long_doc: str, runner):
        super().__init__(name=name)
        self._short_doc = short_doc
        self._long_doc = long_doc
        self._runner = runner

    def get_function_definition(self, short: bool):
        return self._short_doc if short else self._long_doc

    def _execute(self, *args, **kwargs):
        with ToolTracer(self.name, args=args, kwargs=kwargs) as tracer:
            result = self._runner(*args, **kwargs)
            tracer.set_output(result)

            # 对于盲评工具，直接将结果格式化为文本并注入到对话上下文中
            if self.name == "seed_submit_blind_review" and isinstance(result, dict):
                return self._format_blind_review_result(result)

            return result

    def _format_blind_review_result(self, result: Dict[str, Any]) -> str:
        """将盲评结果格式化为友好的文本消息，直接注入到对话上下文中"""
        success = result.get("success", False)
        passed_samples = result.get("passed_samples", 0)
        total_samples = result.get("total_samples", 0)
        required_pass = result.get("required_pass", 3)
        summary = result.get("summary", "")

        # 构建格式化的消息
        lines = [
            "=" * 60,
            "盲评结果",
            "=" * 60,
            f"状态: {'✓ 通过' if success else '✗ 未通过'}",
            f"通过题目数: {passed_samples}/{total_samples}",
            f"要求通过数: {required_pass}",
            f"总结: {summary}",
            "",
        ]

        # 添加每道题的详细结果
        samples = result.get("samples", [])
        if samples:
            lines.append("详细结果:")
            for sample in samples:
                sample_id = sample.get("sample_id", "?")
                difficulty = sample.get("difficulty", "?")
                sample_pass = sample.get("pass", False)
                sample_summary = sample.get("summary", "")
                lines.append(
                    f"  题目 {sample_id} (难度 {difficulty}): {sample_summary}"
                )

        # 如果未通过，显示失败详情
        if not success:
            # 从 samples 中找到失败的题目
            failed_samples = [
                sample for sample in samples if not sample.get("pass", False)
            ]
            if failed_samples:
                lines.append("")
                lines.append("失败题目详情:")
                for sample in failed_samples:
                    sample_id = sample.get("sample_id", "?")
                    difficulty = sample.get("difficulty", "?")
                    question = sample.get("question", "")
                    attempts = sample.get("attempts", [])

                    # 对于每道失败题目，显示所有尝试的完整信息
                    for attempt in attempts:
                        if not attempt.get("match", False):
                            blind_info = {
                                "blind_review": {
                                    "sample_id": sample_id,
                                    "extracted_answer": attempt.get("blind_answer", ""),
                                    "expected_answer": attempt.get(
                                        "official_answer", ""
                                    ),
                                    "matched": attempt.get("match", False),
                                    "attempt": attempt.get("attempt", 1),
                                    "question": question,
                                }
                            }
                            # 格式化为 JSON，确保可读性
                            lines.append("")
                            lines.append(
                                json.dumps(blind_info, ensure_ascii=False, indent=2)
                            )

        lines.append("=" * 60)

        return "\n".join(lines)


class GuardedStopTool(StopTool):
    def __init__(self, agent: CKAgent, guard_fn):
        super().__init__(agent=agent)
        self.guard_fn = guard_fn

    def _execute(self, output: str, log: str = "", **kwargs):
        # 将 output 传递给 guard_fn 以便执行最终提交前的合规性检查
        allowed, message = self.guard_fn(output)
        if not allowed:
            return {
                "error": "stop_guard_blocked",
                "message": message,
            }
        return super()._execute(output, log)


class SeedRunContext:
    def __init__(
        self,
        seed: Dict[str, Any],
        experience_manager: ExperienceManager,
        validator_builder_llm,
        blind_agent: CKAgent,
        quality_agent: CKAgent,
        session_id: str,
        blind_model_call_target: str,
        blind_attempt_timeout: float,
    ):
        self.seed = seed
        self.experience_manager = experience_manager
        self.validator_builder_llm = (
            validator_builder_llm  # 用于自动生成 validator 的纯 LLM
        )
        self.blind_agent = blind_agent  # 用于盲评的 CKAgent
        self.quality_agent = quality_agent
        self.session_id = session_id
        self.blind_model_call_target = blind_model_call_target
        self.blind_attempt_timeout = blind_attempt_timeout
        self._blind_agent_lock: Lock = Lock()

        self.samples: List[Dict[str, Any]] = []
        self.blind_history: List[Dict[str, Any]] = []
        self.blind_success: bool = False
        self.new_experiences: List[str] = []

        # 代码草稿存储（三个独立组件）
        self.saved_generator_code: str = ""
        self.saved_question_template: str = ""
        self.saved_validator_code: str = ""
        self.saved_evolution_strategy: str = ""
        self.validator_generation_logs: List[Dict[str, Any]] = []
        self.validator_timeout: float = VALIDATOR_EXEC_TIMEOUT

        # 会话目录（用于保存文件）
        self.session_dir: Optional[Path] = None

        # stop 尝试计数
        self.stop_fail_count: int = 0

        self._script_globals: Dict[str, Any] = {}
        self._generator_namespace: Dict[str, Any] = {}
        self.validator_pool: List[Dict[str, Any]] = []
        self._auto_validator_index: int = 0
        self.question_check_result: Optional[Dict[str, Any]] = None
        self.quality_history: List[Dict[str, Any]] = []
        self.quality_agent_session: Any | None = None
        self._prepare_script_environment()

    # ------------------------------------------------------------------ #
    # Script helpers
    def _prepare_script_environment(self) -> None:
        script = self.seed.get("full_script") or ""
        generator_code = self.seed.get("generator_code") or ""
        validator_code = self.seed.get("validator_code") or ""

        compiled: Dict[str, Any] = {}
        try:
            if script.strip():
                exec(script, compiled)
            else:
                exec(generator_code, compiled)
                exec(validator_code, compiled)
        except Exception as exc:
            raise RuntimeError(
                f"加载 seed 代码失败：{exc}\n\nGenerator:\n{generator_code}\n\nValidator:\n{validator_code}"
            ) from exc

        self._script_globals = compiled

    def _set_generator_from_code(self, code: str) -> None:
        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        input_fn = namespace.get("input")
        if not callable(input_fn):
            raise RuntimeError(
                "generator_code 必须定义可调用的 input(difficulty) 函数。"
            )
        self._generator_namespace = namespace

    def _register_validator_code(
        self, name: str, code: str, source: str
    ) -> Dict[str, Any]:
        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        fn = namespace.get("solution")
        if not callable(fn):
            raise RuntimeError(
                "validator_code 必须定义可调用的 solution(inputs) 函数。"
            )

        # 去重：同名 validator 覆盖旧版本
        self.validator_pool = [
            item for item in self.validator_pool if item["name"] != name
        ]

        record = {
            "name": name,
            "code": code,
            "fn": fn,
            "source": source,
            "registered_at": datetime.now().isoformat(),
        }
        self.validator_pool.append(record)
        return record

    def _stringify_answer(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return repr(value)

    def _run_validator_in_subprocess(self, code: str, inputs: Any, timeout: float):
        """在独立子进程中执行 validator，避免主进程被长耗时阻塞"""
        queue: mp.Queue = MP_CONTEXT.Queue()
        process = MP_CONTEXT.Process(
            target=_validator_subprocess_runner,
            args=(queue, code, inputs),
            daemon=True,
        )
        try:
            process.start()
        except Exception as exc:  # noqa: BLE001
            queue.close()
            queue.join_thread()
            raise RuntimeError(f"validator 子进程启动失败：{exc}") from exc

        process.join(timeout)
        if process.is_alive():
            process.kill()
            process.join()
            queue.close()
            queue.join_thread()
            raise TimeoutError(f"validator timeout>{timeout}s")

        try:
            status, payload = queue.get_nowait()
        except Empty:
            queue.close()
            queue.join_thread()
            raise TimeoutError(f"validator timeout>{timeout}s (无返回值)")
        finally:
            queue.close()
            queue.join_thread()

        if status == "ok":
            return payload

        exc_type, exc_message, exc_tb = payload
        raise RuntimeError(f"{exc_type}: {exc_message}")

    def _run_validator_in_thread(self, fn, inputs: Any, timeout: float):
        """线程降级版本的执行器（无法被强杀，但兼容不可序列化输入）"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, inputs)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError as exc:
                raise TimeoutError(f"validator timeout>{timeout}s") from exc

    def _execute_validator(self, item: Dict[str, Any], inputs: Any):
        """统一的 validator 执行入口，优先使用子进程执行并加超时保护"""
        timeout = self.validator_timeout
        code_text = item.get("code") or ""
        if code_text:
            try:
                return self._run_validator_in_subprocess(
                    code=code_text, inputs=inputs, timeout=timeout
                )
            except TimeoutError:
                raise
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[warn] validator {item.get('name')} 子进程执行失败，降级到线程：{exc}"
                )

        fn = item.get("fn")
        if not callable(fn):
            raise RuntimeError("solution 未找到或不可调用")
        return self._run_validator_in_thread(fn, inputs, timeout)

    def _aggregate_validator_answers(self, inputs: Any) -> Dict[str, Any]:
        """
        使用 validator 池对答案进行一致性投票，返回汇总结构
        """
        if not self.validator_pool:
            legacy_fn = self._solution_fn()
            if not legacy_fn:
                raise RuntimeError("当前未注册任何 validator，无法计算答案。")
            try:
                result = self._execute_validator(
                    {
                        "name": "legacy_validator",
                        "source": "legacy",
                        "fn": legacy_fn,
                        "code": self.saved_validator_code
                        or self.seed.get("validator_code")
                        or "",
                    },
                    copy.deepcopy(inputs),
                )
                answer_str = self._stringify_answer(result)
                return {
                    "answer": result,
                    "consensus": answer_str,
                    "votes": [
                        {
                            "name": "legacy_validator",
                            "source": "legacy",
                            "answer": answer_str,
                            "status": "ok",
                            "is_winner": True,
                            "vote_share": 1.0,
                        }
                    ],
                    "disagreement": False,
                    "support_ratio": 1.0,
                }
            except Exception as exc:
                raise RuntimeError(f"legacy validator 执行失败：{exc}") from exc

        votes: List[Dict[str, Any]] = []
        answer_cache: List[Dict[str, Any]] = []

        for item in self.validator_pool:
            try:
                result = self._execute_validator(item, copy.deepcopy(inputs))
                if isinstance(result, dict) and result.get("status") == "schema_error":
                    votes.append(
                        {
                            "name": item["name"],
                            "source": item.get("source"),
                            "answer": None,
                            "status": "schema_error",
                            "error": result.get("detail", "inputs schema mismatch"),
                            "is_winner": False,
                        }
                    )
                    continue
                answer_str = self._stringify_answer(result)
                answer_cache.append(
                    {
                        "name": item["name"],
                        "source": item.get("source"),
                        "answer": result,
                        "answer_str": answer_str,
                    }
                )
                votes.append(
                    {
                        "name": item["name"],
                        "source": item.get("source"),
                        "answer": answer_str,
                        "status": "ok",
                        "is_winner": False,
                    }
                )
            except TimeoutError as exc:
                votes.append(
                    {
                        "name": item.get("name"),
                        "source": item.get("source"),
                        "answer": None,
                        "status": "timeout",
                        "error": str(exc),
                        "is_winner": False,
                        "vote_share": 0.0,
                    }
                )
            except Exception as exc:
                votes.append(
                    {
                        "name": item["name"],
                        "source": item.get("source"),
                        "answer": None,
                        "status": "error",
                        "error": str(exc),
                        "is_winner": False,
                    }
                )

        valid_answers = [entry for entry in answer_cache]
        if not valid_answers:
            # 检查是否有 auto_validator（盲审阶段才会有）
            has_auto_validator = any(
                item.get("source") == "auto_llm" for item in self.validator_pool
            )
            if has_auto_validator:
                # 盲审阶段：所有 validator 都失败了，这是严重问题
                raise RuntimeError(
                    "validator 池未能返回有效答案，请检查各个 validator 的实现。"
                )
            else:
                # 第一次保存阶段：只有 primary_validator，失败时给出更友好的提示
                error_details = [
                    f"{vote.get('name', 'unknown')}: {vote.get('error', 'unknown error')}"
                    for vote in votes
                    if vote.get("status") != "ok"
                ]
                error_msg = "primary_validator 执行失败"
                if error_details:
                    error_msg += f"：{'; '.join(error_details)}"
                raise RuntimeError(
                    f"{error_msg}。请检查 validator_code 的实现是否正确。"
                )

        counter = Counter(entry["answer_str"] for entry in valid_answers)
        top_answer, top_count = counter.most_common(1)[0]
        total_valid = sum(counter.values())
        disagreement = len(counter) > 1
        winner_entry = next(
            entry for entry in valid_answers if entry["answer_str"] == top_answer
        )

        for vote in votes:
            if vote["answer"] == top_answer:
                vote["is_winner"] = True
            vote["vote_share"] = (
                counter[vote["answer"]] / total_valid
                if vote.get("answer") in counter
                else 0.0
            )

        return {
            "answer": winner_entry["answer"],
            "consensus": top_answer,
            "votes": votes,
            "disagreement": disagreement,
            "support_ratio": top_count / total_valid,
        }

    def _run_contract_smoke_test(
        self, difficulties: Optional[List[int]] = None
    ) -> None:
        """
        生成少量样本，确保 generator 与所有已注册 validator 的输入输出契约一致
        """
        input_fn = self._generator_namespace.get("input")
        if not callable(input_fn):
            return
        validators = [item for item in self.validator_pool if callable(item.get("fn"))]
        if not validators:
            return

        if not difficulties:
            difficulties = [1, 3, 5, 7]

        errors: List[str] = []
        for diff in difficulties:
            diff = max(1, min(10, diff))
            try:
                generated = input_fn(diff)
            except Exception as exc:
                errors.append(f"generator(difficulty={diff}) 运行失败：{exc}")
                continue

            if (
                not isinstance(generated, tuple)
                or len(generated) != 2
                or generated[0] is None
            ):
                errors.append(
                    f"generator(difficulty={diff}) 返回格式应为 (inputs, slot_texts)"
                )
                continue

            inputs = generated[0]

            for item in validators:
                try:
                    result = self._execute_validator(item, copy.deepcopy(inputs))
                    if (
                        isinstance(result, dict)
                        and result.get("status") == "schema_error"
                    ):
                        errors.append(
                            f"validator {item.get('name')} 在 difficulty={diff} 报告 schema_error：{result.get('detail')}"
                        )
                except Exception as exc:
                    errors.append(
                        f"validator {item.get('name')} 在 difficulty={diff} 执行失败：{exc}"
                    )

        if errors:
            raise RuntimeError("契约自检失败：" + "；".join(errors[:3]))

    def _extract_python_code(self, raw_output: Any) -> str:
        text = raw_output if isinstance(raw_output, str) else str(raw_output)
        text = text.strip()
        if not text:
            return ""

        if "```" in text:
            lines = text.split("\n")
            code_lines: List[str] = []
            in_block = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("```"):
                    if in_block:
                        break
                    in_block = True
                    continue
                if in_block:
                    code_lines.append(line)
            code = "\n".join(code_lines).strip()
            if code:
                return code
        return text

    def _spawn_validator_via_llm(self) -> Dict[str, Any]:
        if not self.saved_generator_code or not self.saved_question_template:
            raise RuntimeError("生成自动 validator 前需要先保存 generator 与题面模板。")

        prompt = render_validator_builder_prompt(
            generator_code=self.saved_generator_code,
            question_template=self.saved_question_template,
            reference_validator=self.saved_validator_code,
        )
        raw_output = self.validator_builder_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a Python validator-writing assistant. Output only executable validator code."
                        "Do not output explanations, natural language, or extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )
        code = self._extract_python_code(raw_output)
        if not code:
            raise RuntimeError("自动生成 validator 失败：未获取到代码。")

        self._auto_validator_index += 1
        name = f"auto_validator_{self._auto_validator_index}"
        record = self._register_validator_code(name=name, code=code, source="auto_llm")
        record["raw_output"] = raw_output
        self.validator_generation_logs.append(
            {
                "name": name,
                "raw_output": raw_output,
                "registered_at": record["registered_at"],
            }
        )

        if self.session_dir:
            file_path = self.session_dir / f"{name}.py"
            try:
                file_path.write_text(code, encoding="utf-8")
            except Exception as exc:
                print(f"[warn] 自动 validator 写入失败: {exc}")

        return record

    def _ensure_additional_validators(self, required_total: int = 3) -> None:
        """
        确保 validator 池中至少有 required_total 个可用验证器
        （包含主验证器 + 自动生成的验证器）

        注意：生成 auto_validator 只需要 generator_code 和 question_template，
        不需要 saved_validator_code（虽然会作为参考传递给 LLM）
        """
        if not self.saved_generator_code or not self.saved_question_template:
            print(
                f"[warn] 无法生成 auto_validator：缺少 generator_code 或 question_template"
            )
            return

        current_validators = len(
            [item for item in self.validator_pool if callable(item.get("fn"))]
        )
        attempts = 0
        max_attempts = required_total * 3
        while current_validators < required_total and attempts < max_attempts:
            attempts += 1
            try:
                self._spawn_validator_via_llm()
            except Exception as exc:
                print(f"[warn] 自动生成 validator 失败（尝试 {attempts}）: {exc}")
            current_validators = len(
                [item for item in self.validator_pool if callable(item.get("fn"))]
            )

        if current_validators < required_total:
            raise RuntimeError(
                f"自动生成备用 validator 未达预期数量（需要 {required_total} 个，实际 {current_validators} 个）。"
            )

    def _run_question_quality_check(self, difficulty: int = 7) -> Dict[str, Any]:
        if self.question_check_result:
            return self.question_check_result

        if not self.samples:
            # 如果尚未生成样本，自动生成一个中等偏高难度样本供检查
            try:
                self.tool_generate_sample(difficulty=difficulty, rng_seed=42)
            except Exception as exc:
                fallback_used = False
                if isinstance(exc, TimeoutError) or "timeout" in str(exc).lower():
                    # 如果 validator 在较高难度下超时，自动降级难度重试一次
                    fallback_difficulty = max(3, min(5, difficulty - 2))
                    try:
                        self.tool_generate_sample(
                            difficulty=fallback_difficulty, rng_seed=42
                        )
                        fallback_used = True
                        print(
                            f"[warn] 自动生成检查样本在难度 {difficulty} 超时，已降级到难度 {fallback_difficulty} 重试。"
                        )
                    except Exception as exc2:
                        raise RuntimeError(
                            f"自动生成检查样本失败：{exc}；降级到难度 {fallback_difficulty} 仍失败：{exc2}"
                        ) from exc2
                if not fallback_used:
                    raise RuntimeError(f"自动生成检查样本失败：{exc}") from exc

        latest_sample = self.samples[-1]
        prompt = render_question_quality_prompt(
            question=latest_sample.get("question", ""),
            answer=latest_sample.get("answer_str", ""),
            difficulty=latest_sample.get("difficulty"),
        )

        agent_result = _run_agent(
            self.quality_agent,
            prompt,
            session=self.quality_agent_session,
        )
        self.quality_agent_session = agent_result.get("session")
        raw_output = agent_result["output"]
        result_text = (
            raw_output if isinstance(raw_output, str) else str(raw_output)
        ).strip()

        try:
            if result_text.startswith("```"):
                result_text = self._extract_python_code(result_text)
            parsed = json.loads(result_text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(result_text)
            except Exception as exc:
                raise RuntimeError(
                    f"题面检查结果解析失败：{exc}。原始输出：{result_text}"
                ) from exc
            else:
                if not isinstance(parsed, dict):
                    raise RuntimeError(
                        f"题面检查结果格式无效（需要 dict）。原始输出：{result_text}"
                    )
        except Exception as exc:
            raise RuntimeError(
                f"题面检查结果解析失败：{exc}。原始输出：{result_text}"
            ) from exc

        action = (parsed.get("action") or "").lower()
        parsed["raw_output"] = result_text
        self.quality_history.append(parsed)

        if action not in {"proceed", "pass"}:
            parsed["status"] = "revise"
            self.question_check_result = parsed
            return parsed

        parsed["status"] = "pass"
        self.question_check_result = parsed
        return parsed

    def _input_fn(self):
        if self._generator_namespace:
            func = self._generator_namespace.get("input")
            if callable(func):
                return func
        func = self._script_globals.get("input")
        return func if callable(func) else None

    def _solution_fn(self):
        if self.validator_pool:
            primary = self.validator_pool[0]
            fn = primary.get("fn")
            if callable(fn):
                return fn
        func = self._script_globals.get("solution")
        return func if callable(func) else None

    # ------------------------------------------------------------------ #
    # Tool implementations
    def tool_generate_sample(
        self,
        difficulty: int = 5,
        template_index: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        生成题目样本

        工作流程：
        1. 使用 generator 生成题目输入数据和槽位文本
        2. 使用题面模板渲染题目文本
        3. 使用 validator 计算答案
        """
        if rng_seed is not None:
            random.seed(rng_seed)

        input_fn = self._input_fn()
        if not input_fn:
            raise RuntimeError(
                "尚未检测到有效的 generator。请先调用 seed_save_code() 保存新的 generator 代码。"
            )

        generated = input_fn(difficulty=difficulty)
        if not isinstance(generated, (list, tuple)) or len(generated) != 2:
            raise RuntimeError(
                "input(difficulty) 必须返回 (inputs, slot_texts)。请检查 generator_code 的返回值。"
            )
        inputs, slot_texts_raw = generated
        if isinstance(slot_texts_raw, (list, tuple)):
            slot_texts = list(slot_texts_raw)
        elif slot_texts_raw is None:
            slot_texts = []
        else:
            slot_texts = [slot_texts_raw]

        question_template = self.saved_question_template or ""
        if not question_template:
            templates = self.seed.get("question_templates") or []
            if templates:
                if template_index is not None and 0 <= template_index < len(templates):
                    question_template = templates[template_index]
                else:
                    question_template = random.choice(templates)

        question: Optional[str]
        if question_template:
            rendered_question = str(question_template)
            for slot_idx, text in enumerate(slot_texts, start=1):
                rendered_question = rendered_question.replace(
                    f"[Input Slot {slot_idx}]", str(text)
                )
            question = rendered_question
        else:
            question = None

        aggregation = self._aggregate_validator_answers(inputs)
        answer = aggregation["answer"]

        if not question or not str(question).strip():
            raise RuntimeError(
                "生成的样本缺少题面描述。请确保：\n"
                "1. 使用 seed_save_code() 保存了题面模板 (question_template)\n"
                "2. generator 的 slot_texts 与模板槽位匹配"
            )
        question_str = str(question).strip()
        sample_question = (self.seed.get("sample_question") or "").strip()
        if sample_question and question_str == sample_question:
            raise RuntimeError(
                "检测到题面仍然与原始 seed 示例完全一致。请在题面模板中设计新的题目描述。"
            )
        if "[Input Slot" in question_str:
            raise RuntimeError(
                "The statement still contains template placeholders (e.g., [Input Slot 1]). Please check:\n"
                "1. The generator returned enough slot_texts\n"
                "2. The slot indices in the template are correct"
            )

        sample_id = len(self.samples) + 1
        sample = {
            "sample_id": sample_id,
            "difficulty": difficulty,
            "question": question_str,
            "answer": answer,
            "answer_str": self._stringify_answer(answer),
            "validator_votes": aggregation.get("votes") if aggregation else [],
            "validator_consensus": (
                aggregation.get("consensus") if aggregation else None
            ),
            "validator_disagreement": (
                aggregation.get("disagreement") if aggregation else False
            ),
            "validator_support_ratio": (
                aggregation.get("support_ratio") if aggregation else 1.0
            ),
            "template_index": template_index,
        }
        if aggregation.get("disagreement"):
            sample["validator_warning"] = (
                "不同 validator 返回了不一致的答案，请检查 validator_votes 并修复主验证器。"
            )
            print(
                f"[warn] Validator disagreement detected (sample #{sample_id}, difficulty {difficulty})."
            )
        self.samples.append(sample)
        return sample

    def tool_list_samples(self) -> List[Dict[str, Any]]:
        return [
            {
                "sample_id": item["sample_id"],
                "difficulty": item["difficulty"],
                "answer": item["answer_str"],
                "question_preview": item["question"][:120]
                + ("..." if len(item["question"]) > 120 else ""),
            }
            for item in self.samples
        ]

    def tool_check_question_quality(self, difficulty: int = 7) -> Dict[str, Any]:
        result = self._run_question_quality_check(difficulty=difficulty)
        if self.session_dir:
            try:
                quality_file = self.session_dir / "question_quality_check.json"
                quality_file.write_text(
                    json.dumps(
                        {
                            "latest": result,
                            "history": self.quality_history,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                result["_persist_warning"] = str(exc)
        return {"latest": result, "history": self.quality_history}

    def _run_blind_attempt_with_timeout(
        self, prompt: str, sample_answer: str, attempt_idx: int
    ) -> Dict[str, Any]:
        """在子进程中执行盲评，超时直接强杀"""
        timeout = self.blind_attempt_timeout or BLIND_ATTEMPT_DEFAULT_TIMEOUT
        queue: mp.Queue = MP_CONTEXT.Queue()
        process = MP_CONTEXT.Process(
            target=_blind_attempt_subprocess_runner,
            args=(
                queue,
                prompt,
                sample_answer,
                self.blind_model_call_target,
            ),
            daemon=True,
        )
        process.start()
        process.join(timeout + BLIND_ATTEMPT_TIMEOUT_BUFFER)
        if process.is_alive():
            process.kill()
            process.join()
            queue.close()
            queue.join_thread()
            raise TimeoutError(
                f"blind_attempt_timeout>{timeout}s (attempt {attempt_idx + 1})"
            )
        try:
            status, payload = queue.get_nowait()
        except Empty:
            raise TimeoutError(
                f"blind_attempt_timeout>{timeout}s (attempt {attempt_idx + 1})"
            )
        finally:
            queue.close()
            queue.join_thread()

        if status == "ok":
            return payload

        exc_type, exc_message, exc_tb = payload
        if exc_type == "TimeoutError":
            raise TimeoutError(
                f"blind_attempt_timeout>{timeout}s (attempt {attempt_idx + 1})"
            )
        raise RuntimeError(f"{exc_type}: {exc_message}\n{exc_tb}")

    def _run_single_blind_attempt_with_context(
        self,
        sample_id: int,
        attempt_idx: int,
        question: str,
        experiences: List[str],
        sample_answer: str,
        parent_context,
    ) -> Dict[str, Any]:
        """单次盲评执行（带 context 传递，用于并行调用）- 纯 LLM 调用"""
        from ..agents.phoenix_tracer import is_phoenix_enabled

        prompt = render_blind_prompt(
            question=question,
            experiences=experiences,
            attempt_idx=attempt_idx,
        )

        # 在子线程中恢复父线程的 OpenTelemetry context
        if is_phoenix_enabled() and parent_context:
            try:
                from opentelemetry import context, trace  # type: ignore[import]

                # 附加父 context
                token = context.attach(parent_context)
                try:
                    tracer = trace.get_tracer("ck-pro.seed")
                    with tracer.start_as_current_span(
                        f"blind_review_attempt_{attempt_idx+1}",
                        attributes={
                            "blind_review.sample_id": sample_id,
                            "blind_review.attempt": attempt_idx + 1,
                            "blind_review.question": question[:200],  # 截断避免过长
                        },
                    ) as span:
                        result = self._run_blind_attempt_with_timeout(
                            prompt, sample_answer, attempt_idx
                        )
                        raw_output = result["raw_output"]
                        normalized_pred = result["extracted_answer"]
                        normalized_gold = result["expected_answer"]
                        matched = result["matched"]

                        # 记录结果到 span
                        span.set_attribute(
                            "blind_review.extracted_answer", normalized_pred
                        )
                        span.set_attribute(
                            "blind_review.expected_answer", normalized_gold
                        )
                        span.set_attribute("blind_review.matched", matched)

                        result["attempt"] = attempt_idx + 1
                        return result
                finally:
                    # 恢复原来的 context
                    context.detach(token)
            except Exception as e:
                print(f"Phoenix span 创建失败: {e}")
                # 降级到无追踪模式
                pass

        result = self._run_blind_attempt_with_timeout(
            prompt, sample_answer, attempt_idx
        )
        result["attempt"] = attempt_idx + 1
        return result

    def tool_submit_blind_review(self) -> Dict[str, Any]:
        """
        自动生成多难度样本并进行盲评校验（每题并行 LLM 调用）

        盲评的目的：测试 generator 和 validator 代码是否可用、答案是否正确

        Returns:
            盲评结果，包含每道题的尝试记录和整体通过情况
        """
        if not self.saved_generator_code:
            raise RuntimeError(
                "尚未检测到演化后的 generator/validator 代码。请先调用 seed_save_code() 保存新代码后再执行盲评。"
            )

        # 题面质量审查
        if not self.question_check_result:
            self._run_question_quality_check(difficulty=7)

        # 确保 validator 池具备足够的版本进行一致性投票
        self._ensure_additional_validators(required_total=3)

        difficulties = [1, 3, 5, 5, 7]
        attempts_per_sample = 1
        required_pass = 3

        experiences = self.experience_manager.all()

        # 获取当前 OpenTelemetry context 以便传递到子线程
        from ..agents.phoenix_tracer import is_phoenix_enabled

        current_context = None
        if is_phoenix_enabled():
            try:
                from opentelemetry import context  # type: ignore[import]

                current_context = context.get_current()
            except Exception:
                pass

        sample_records: List[Dict[str, Any]] = []

        for difficulty in difficulties:
            sample = self.tool_generate_sample(difficulty=difficulty)
            sample_id = sample["sample_id"]
            question = sample["question"]
            official_answer = sample["answer_str"]

            blind_attempts: List[Optional[Dict[str, Any]]] = [
                None
            ] * attempts_per_sample
            with ThreadPoolExecutor(max_workers=attempts_per_sample) as executor:
                futures = {
                    executor.submit(
                        self._run_single_blind_attempt_with_context,
                        sample_id,
                        attempt_idx,
                        question,
                        experiences,
                        sample["answer"],
                        current_context,
                    ): attempt_idx
                    for attempt_idx in range(attempts_per_sample)
                }

                for future in as_completed(futures):
                    attempt_idx = futures[future]
                    try:
                        result = future.result()
                        blind_attempts[attempt_idx] = result
                    except Exception as exc:
                        print(f"盲评尝试 {attempt_idx + 1} 出错: {exc}")
                        blind_attempts[attempt_idx] = {
                            "attempt": attempt_idx + 1,
                            "raw_output": f"Error: {exc}",
                            "extracted_answer": "",
                            "expected_answer": sample["answer_str"],
                            "matched": False,
                        }

            formatted_attempts: List[Dict[str, Any]] = []
            matched_attempts = 0
            for attempt_idx, item in enumerate(blind_attempts):
                if item:
                    raw_output = item["raw_output"]
                    if len(raw_output) > 500:
                        raw_output_display = (
                            raw_output[:200]
                            + f"...(详细推理已截断，共 {len(raw_output)} 字符)"
                        )
                    else:
                        raw_output_display = raw_output

                    matched = item["matched"]
                    if matched:
                        matched_attempts += 1

                    formatted_attempts.append(
                        {
                            "attempt": item["attempt"],
                            "blind_answer": item["extracted_answer"],
                            "blind_reasoning": raw_output_display,
                            "official_answer": item["expected_answer"],
                            "match": matched,
                            "comparison": (
                                "✓ 结果一致"
                                if matched
                                else (
                                    f"✗ 结果不一致（blind: {item['extracted_answer']} | "
                                    f"official: {item['expected_answer']}）"
                                )
                            ),
                        }
                    )
                else:
                    formatted_attempts.append(
                        {
                            "attempt": attempt_idx + 1,
                            "blind_answer": "",
                            "blind_reasoning": "Error: 未获取答案",
                            "official_answer": official_answer,
                            "match": False,
                            "comparison": "✗ 结果不一致（blind 未返回答案）",
                        }
                    )

            sample_pass = matched_attempts > 0
            failed_attempts = [
                attempt for attempt in formatted_attempts if not attempt.get("match")
            ]
            sample_records.append(
                {
                    "sample_id": sample_id,
                    "difficulty": difficulty,
                    "question": question,
                    "official_answer": official_answer,
                    "attempts": formatted_attempts,
                    "matched_attempts": matched_attempts,
                    "pass": sample_pass,
                    "summary": f"难度 {difficulty}: {matched_attempts}/{attempts_per_sample} 次匹配"
                    + (" ✓" if sample_pass else " ✗"),
                    "failed_attempts": failed_attempts,
                }
            )

        passed_samples = sum(1 for item in sample_records if item["pass"])
        total_samples = len(sample_records)
        success = passed_samples >= required_pass
        if success and not self.blind_success:
            self.blind_success = True

        flat_attempts: List[Dict[str, Any]] = []
        for sample_record in sample_records:
            for attempt in sample_record["attempts"]:
                attempt_copy = dict(attempt)
                attempt_copy["sample_id"] = sample_record["sample_id"]
                attempt_copy["difficulty"] = sample_record["difficulty"]
                flat_attempts.append(attempt_copy)

        difficulty_str = ", ".join(str(d) for d in difficulties)

        record = {
            "samples": sample_records,
            "blind_results": flat_attempts,
            "success": success,
            "passed_samples": passed_samples,
            "pass_count": passed_samples,
            "required_pass": required_pass,
            "total_samples": total_samples,
            "attempts_per_sample": attempts_per_sample,
            "difficulties": difficulties,
            "question_quality_check": self.question_check_result,
            "question_quality_history": self.quality_history,
            "validator_pool": [
                {
                    "name": item["name"],
                    "source": item.get("source"),
                    "registered_at": item.get("registered_at"),
                    "code": item.get("code"),
                }
                for item in self.validator_pool
            ],
            "auto_validator_logs": self.validator_generation_logs,
            "summary": (
                f"盲评共 {total_samples} 道题（难度 {difficulty_str}），"
                f"通过 {passed_samples} 道（要求 {required_pass} 道）"
                + ("，判定通过 ✓" if success else "，判定未通过 ✗")
            ),
            # 添加 Agent 可能期望的字段别名，确保结果能被正确访问
            "status": "passed" if success else "failed",
            "Passed": passed_samples,
            "Total Questions": total_samples,
            "Correct Answers": passed_samples,
            "Blind Review Summary": (
                f"盲评共 {total_samples} 道题（难度 {difficulty_str}），"
                f"通过 {passed_samples} 道（要求 {required_pass} 道）"
                + ("，判定通过 ✓" if success else "，判定未通过 ✗")
            ),
        }
        if not success:
            record["failed_samples_detail"] = [
                {
                    "sample_id": item["sample_id"],
                    "difficulty": item["difficulty"],
                    "failed_attempts": item.get("failed_attempts", []),
                }
                for item in sample_records
                if not item["pass"]
            ]

        self.blind_history.append(record)
        return record

    def tool_record_experience(self, note: str) -> Dict[str, Any]:
        added = self.experience_manager.add([note])
        self.new_experiences.extend(added)
        return {"recorded": added}

    def tool_save_code(
        self,
        generator_code: str,
        question_template: str,
        validator_code: str,
        evolution_strategy: str = "",
    ) -> Dict[str, Any]:
        """
        保存演化后的三个独立组件

        Args:
            generator_code: Evolved generator code (input function that produces input data)
            question_template: Evolved statement template (must include [Input Slot N] placeholders)
            validator_code: Evolved validator code (solution function that computes the answer)
            evolution_strategy: Evolution strategy description (optional)

        Returns:
            保存确认信息
        """

        def _ast_signature(code_text: str) -> Optional[str]:
            try:
                tree = ast.parse(code_text)
                return ast.dump(tree, include_attributes=False)
            except Exception:
                return None

        # AST 结构校验：generator 必须有实质性改动
        orig_gen = self.seed.get("generator_code", "")
        new_gen_sig = _ast_signature(generator_code)
        orig_gen_sig = _ast_signature(orig_gen) if orig_gen else None
        if orig_gen_sig and new_gen_sig and new_gen_sig == orig_gen_sig:
            raise RuntimeError(
                "检测到 generator_code 的结构与原始 seed 完全一致。请重写生成逻辑，而非只改动题面。"
            )

        # 保存三个组件
        self.saved_generator_code = generator_code
        self.saved_question_template = question_template
        self.saved_validator_code = validator_code
        self.saved_evolution_strategy = evolution_strategy
        self.question_check_result = None
        self.quality_history = []
        self.quality_agent_session = None
        if "[Input Slot" not in question_template:
            raise RuntimeError(
                "question_template must include at least one placeholder (e.g., [Input Slot 1])."
            )

        # 保存到文件系统
        if self.session_dir:
            try:
                generator_file = self.session_dir / "generator.py"
                template_file = self.session_dir / "question_template.txt"
                validator_file = self.session_dir / "validator.py"
                strategy_file = self.session_dir / "evolution_strategy.txt"

                generator_file.write_text(generator_code, encoding="utf-8")
                template_file.write_text(question_template, encoding="utf-8")
                validator_file.write_text(validator_code, encoding="utf-8")
                if evolution_strategy:
                    strategy_file.write_text(evolution_strategy, encoding="utf-8")

                files_saved = f"已保存到 {generator_file.name}, {template_file.name}, {validator_file.name}"
            except Exception as e:
                files_saved = f"文件保存失败: {e}"
        else:
            files_saved = "会话目录未初始化，仅保存在内存中"

        # 更新脚本环境以便后续生成样本时使用新代码
        try:
            compiled: Dict[str, Any] = {}
            exec(generator_code, compiled)
            exec(validator_code, compiled)
            self._script_globals = compiled
            self._set_generator_from_code(generator_code)
            self.validator_pool = []
            self._auto_validator_index = 0
            self.validator_generation_logs = []
            self.samples = []
            self.blind_history = []
            primary_record = self._register_validator_code(
                name="primary_validator", code=validator_code, source="agent"
            )
        except Exception as exc:
            return {
                "status": "warning",
                "message": f"代码已保存，但加载时出现警告：{exc}。{files_saved}",
                "generator_length": len(generator_code),
                "validator_length": len(validator_code),
            }

        contract_error: Optional[str] = None
        try:
            self._run_contract_smoke_test()
        except Exception as exc:
            contract_error = str(exc)

        if contract_error:
            return {
                "status": "warning",
                "message": f"代码已保存，但契约自检失败：{contract_error}。{files_saved}",
                "generator_length": len(generator_code),
                "validator_length": len(validator_code),
                "validator_pool_size": len(self.validator_pool),
                "primary_validator_registered_at": primary_record["registered_at"],
                "contract_check": "failed",
            }

        return {
            "status": "success",
            "message": f"代码已保存并加载到执行环境。{files_saved}",
            "generator_length": len(generator_code),
            "validator_length": len(validator_code),
            "validator_pool_size": len(self.validator_pool),
            "primary_validator_registered_at": primary_record["registered_at"],
            "contract_check": "passed",
        }

    def tool_prepare_submission(self) -> Dict[str, Any]:
        """
        准备最终提交的数据结构

        Returns:
            包含所有必要字段的 dict，可直接转为 JSON 提交
        """
        import json

        # 获取通过盲评的样本摘要
        passed_sample_details: List[Dict[str, Any]] = []
        for entry in self.blind_history:
            if "samples" in entry:
                for sample in entry.get("samples", []):
                    if sample.get("pass"):
                        passed_sample_details.append(sample)
            elif entry.get("success"):
                passed_sample_details.append(
                    {
                        "sample_id": entry.get("sample_id"),
                        "question": entry.get("question", ""),
                        "official_answer": entry.get("official_answer", ""),
                        "difficulty": entry.get("difficulty"),
                    }
                )

        sample_summary = ""
        if passed_sample_details:
            sample = passed_sample_details[0]
            difficulty_note = (
                f"(难度 {sample.get('difficulty')}) "
                if sample.get("difficulty") is not None
                else ""
            )
            question_preview = (sample.get("question") or "")[:100]
            sample_summary = (
                f"样本 {sample.get('sample_id', '-')}: "
                f"{difficulty_note}题面前100字: {question_preview}..., "
                f"标准答案: {sample.get('official_answer', '')}"
            )

        # 盲评结果摘要（以最近一次盲评为准）
        if self.blind_history:
            latest_review = self.blind_history[-1]
            if "samples" in latest_review:
                samples = latest_review.get("samples", [])
                total_samples = latest_review.get("total_samples", len(samples))
                passed_samples = latest_review.get("passed_samples")
                if passed_samples is None:
                    passed_samples = sum(1 for sample in samples if sample.get("pass"))
                required_pass = latest_review.get("required_pass", 3)
                difficulties = latest_review.get("difficulties") or [
                    sample.get("difficulty") for sample in samples
                ]
                difficulty_str = ", ".join(
                    str(d) for d in difficulties if d is not None
                )
                status = "通过" if latest_review.get("success") else "未通过"
                summary_parts = [
                    f"最近一次盲评：共 {total_samples} 道题",
                ]
                if difficulty_str:
                    summary_parts.append(f"（难度 {difficulty_str}）")
                summary_parts.append(
                    f"，通过 {passed_samples} 道，要求 {required_pass} 道，判定{status}"
                )
                blind_review_summary = "".join(summary_parts)
            else:
                latest_attempts = len(latest_review.get("blind_results", []))
                latest_passed = latest_review.get("pass_count", 0)
                if latest_attempts > 0:
                    latest_rate = latest_passed / latest_attempts * 100
                    blind_review_summary = (
                        f"最近一次盲评：样本 {latest_review.get('sample_id', '-')}, "
                        f"总尝试 {latest_attempts} 次，通过 {latest_passed} 次，"
                        f"通过率 {latest_rate:.1f}%"
                    )
                else:
                    blind_review_summary = (
                        f"最近一次盲评：样本 {latest_review.get('sample_id', '-')}, "
                        "尚无有效尝试"
                    )
        else:
            blind_review_summary = "尚未发起盲评"

        submission_data = {
            "task_id": self.seed.get("task_id", ""),
            "generator_code": self.saved_generator_code
            or self.seed.get("generator_code", ""),
            "question_template": self.saved_question_template
            or (
                self.seed.get("question_templates", [""])[0]
                if self.seed.get("question_templates")
                else ""
            ),
            "validator_code": self.saved_validator_code
            or self.seed.get("validator_code", ""),
            "evolution_strategy": self.saved_evolution_strategy,
            "sample_summary": sample_summary,
            "blind_review_summary": blind_review_summary,
            "experience_updates": self.new_experiences,
            "validator_pool_info": [
                {
                    "name": item["name"],
                    "source": item.get("source"),
                    "registered_at": item.get("registered_at"),
                    "code": item.get("code"),
                }
                for item in self.validator_pool
            ],
            "question_quality_check": self.question_check_result,
            "question_quality_history": self.quality_history,
            "notes": (
                f"完成 {len(self.samples)} 个样本生成，{len(self.blind_history)} 次盲评；"
                f"题面检查状态：{(self.question_check_result or {}).get('action', '未执行')}"
            ),
        }

        # 保存详细的盲评结果和样本信息到文件
        if self.session_dir:
            try:
                # 保存盲评详细结果
                blind_detail_file = self.session_dir / "blind_review_details.json"
                blind_detail_file.write_text(
                    json.dumps(self.blind_history, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                # 保存样本信息
                samples_file = self.session_dir / "samples.json"
                samples_file.write_text(
                    json.dumps(self.samples, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                # 保存提交摘要
                summary_file = self.session_dir / "submission_summary.json"
                summary_file.write_text(
                    json.dumps(submission_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                submission_data["_files_saved"] = (
                    "已保存详细信息到 blind_review_details.json, samples.json, submission_summary.json"
                )
            except Exception as e:
                submission_data["_files_save_error"] = f"保存文件失败: {e}"

        return submission_data

    # ------------------------------------------------------------------ #
    def build_tools(self) -> Iterable[Tool]:
        yield ProxyTool(
            "seed_generate_sample",
            "- def seed_generate_sample(difficulty: int = 5, template_index: int | None = None, rng_seed: int | None = None) -> dict",
            """- seed_generate_sample
```python
def seed_generate_sample(difficulty: int = 5, template_index: int | None = None, rng_seed: int | None = None) -> dict:
    # Generate a sample using the current seed code and return a structure with sample_id, question, answer, etc.
    # difficulty controls the difficulty; template_index selects a template (optional); rng_seed sets the random seed (optional).
    pass
```""".strip(),
            self.tool_generate_sample,
        )
        yield ProxyTool(
            "seed_list_samples",
            "- def seed_list_samples() -> list",
            """- seed_list_samples
```python
def seed_list_samples() -> list:
    # List summaries of samples generated in the current session for selection and blind review submission.
    pass
```""".strip(),
            self.tool_list_samples,
        )
        yield ProxyTool(
            "seed_check_question_quality",
            "- def seed_check_question_quality(difficulty: int = 7) -> dict",
            """- seed_check_question_quality
```python
def seed_check_question_quality(difficulty: int = 7) -> dict:
    # Evaluate the current statement with the readability/novelty checker.
    # If no sample exists, auto-generate one at the specified difficulty before checking.
    # Returns JSON with fields such as readability, novelty, difficulty_alignment, action.
    pass
```""".strip(),
            self.tool_check_question_quality,
        )
        yield ProxyTool(
            "seed_submit_blind_review",
            "- def seed_submit_blind_review() -> dict",
            """- seed_submit_blind_review
```python
def seed_submit_blind_review() -> dict:
    # First run readability/novelty checks, then auto-complete the validator pool (primary + 2 auto-generated validators).
    # Then generate 5 blind-review questions at difficulties (1,3,5,5,7), use the validator pool for consensus answers,
    # and call a pure LLM for blind solving. At least 3 correct answers are required to pass; returns detailed records.
    pass
```""".strip(),
            self.tool_submit_blind_review,
        )
        yield ProxyTool(
            "seed_record_experience",
            "- def seed_record_experience(note: str) -> dict",
            """- seed_record_experience
```python
def seed_record_experience(note: str) -> dict:
    # Write new experience to the shared repository (deduplicated) for reuse in later tasks.
    # WARNING: only record generalizable methods/tactics, not task-specific details.
    # Example: OK \"Add multi-step reasoning chains to increase difficulty\" vs NOT OK \"Used modulo in a meeting-room problem\".
    # Returns the records that were actually written.
    pass
```""".strip(),
            self.tool_record_experience,
        )
        yield ProxyTool(
            "seed_save_code",
            "- def seed_save_code(generator_code: str, question_template: str, validator_code: str, evolution_strategy: str = '') -> dict",
            """- seed_save_code
```python
def seed_save_code(generator_code: str, question_template: str, validator_code: str, evolution_strategy: str = '') -> dict:
    # Save the three evolved components: generator, question template, validator.
    #
    # Parameters (all must be strings):
    # - generator_code: full Python code string defining input(difficulty) and returning (inputs, slot_texts)
    # - question_template: statement template string containing [Input Slot N] placeholders
    # - validator_code: full Python code string defining solution(inputs) and returning the answer
    # - evolution_strategy: evolution strategy description string (optional)
    #
    # Example:
    # generator_code = '''
    # def input(difficulty):
    #     # your generator code
    #     return inputs, slot_texts
    # '''
    # question_template = "Given [Input Slot 1] and [Input Slot 2], find ..."
    # validator_code = '''
    # def solution(inputs):
    #     # your validator code
    #     return answer
    # '''
    # result = seed_save_code(generator_code, question_template, validator_code, "evolution strategy")
    #
    # Important:
    # 1. Define these variables (as strings) before calling the function.
    # 2. generator_code and validator_code must be full Python code strings (use triple-quoted multi-line strings).
    # 3. question_template is a plain string containing [Input Slot N] placeholders.
    # 4. You can call multiple times; each call overwrites the previous version.
    # 5. After saving, code is loaded into the execution environment for later sample generation.
    pass
```""".strip(),
            self.tool_save_code,
        )
        yield ProxyTool(
            "seed_prepare_submission",
            "- def seed_prepare_submission() -> dict",
            """- seed_prepare_submission
```python
def seed_prepare_submission() -> dict:
    # Prepare the full data structure for final submission.
    # Returns a dict with all required fields, ready to be converted to JSON for stop.
    # Includes: task_id, generator_code, validator_code, evolution_strategy,
    #           sample_summary, blind_review_summary, experience_updates, notes
    pass
```""".strip(),
            self.tool_prepare_submission,
        )


@dataclass
class SeedRunResult:
    task_id: str
    success: bool
    session_dir: Path
    output_text: str
    parsed_output: Dict[str, Any]
    blind_history: List[Dict[str, Any]] = field(default_factory=list)
    samples: List[Dict[str, Any]] = field(default_factory=list)
    experience_added: List[str] = field(default_factory=list)
    raw_log: str = ""
    raw_session: Any = None


class SeedGenerationPipeline:
    def __init__(
        self,
        seed_path: Optional[Path] = None,
        output_root: Path = Path("artifacts/seed_pipeline_v2"),
        phoenix_project: str = "seed-generation",
        model_call_target: Optional[str] = None,
        blind_model_call_target: Optional[str] = "api:gpt-4o",
    ):
        default_seed_path = (
            Path(__file__).resolve().parents[2] / "data" / "seed" / "Seed.jsonl"
        )
        self.seed_path = Path(seed_path) if seed_path else default_seed_path
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.experience_manager = ExperienceManager()
        self.seeds = self._load_seeds()
        self.model_call_target = (
            model_call_target or default_main_configs["model"]["call_target"]
        )
        self.blind_model_call_target = blind_model_call_target or self.model_call_target
        self.blind_attempt_timeout = BLIND_ATTEMPT_DEFAULT_TIMEOUT

        self.phoenix_enabled = is_phoenix_enabled() or init_phoenix_tracing(
            project_name=phoenix_project
        )

    # ------------------------------------------------------------------ #
    def _load_seeds(self) -> List[Dict[str, Any]]:
        if not self.seed_path.exists():
            raise FileNotFoundError(f"Seed JSONL 不存在：{self.seed_path}")
        records: List[Dict[str, Any]] = []
        with self.seed_path.open("r", encoding="utf-8") as fd:
            for line in fd:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _create_session_dir(self, task_id: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_dir = self.output_root / f"{task_id}_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _register_tool(self, agent: CKAgent, tool: Tool) -> None:
        existing = {t.name for t in agent.tools}
        if tool.name not in existing:
            agent.tools.append(tool)
        if tool.name not in agent.active_functions:
            agent.active_functions.append(tool.name)
        agent.ACTIVE_FUNCTIONS[tool.name] = tool

    def _install_stop_guard(self, agent: CKAgent, guard_fn) -> None:
        guarded = GuardedStopTool(agent=agent, guard_fn=guard_fn)
        replaced = False
        for idx, tool in enumerate(agent.tools):
            if isinstance(tool, StopTool):
                agent.tools[idx] = guarded
                replaced = True
                break
        if not replaced:
            agent.tools.append(guarded)
        agent.ACTIVE_FUNCTIONS["stop"] = guarded
        if "stop" not in agent.active_functions:
            agent.active_functions.append("stop")

    def _find_seed(self, task_id: Optional[str] = None, index: Optional[int] = None):
        if task_id:
            for item in self.seeds:
                if item.get("task_id") == task_id:
                    return item
            raise ValueError(f"task_id={task_id} 不存在。")
        if index is not None:
            if 0 <= index < len(self.seeds):
                return self.seeds[index]
            raise IndexError(f"index={index} 越界（共有 {len(self.seeds)} 条记录）。")
        if not self.seeds:
            raise RuntimeError("Seed 列表为空。")
        return self.seeds[0]

    def _create_agent(self, name: str, call_target: Optional[str] = None) -> CKAgent:
        cfg = _clone_agent_configs(name, call_target or self.model_call_target)
        # 增加超时时间以适应盲评任务（5道题并发执行可能需要更长时间）
        cfg["exec_timeout_wo_call"] = 600  # 10分钟（原来是200秒）
        cfg["exec_timeout_with_call"] = 1200  # 20分钟（原来是1000秒）
        cfg["max_time_limit"] = 5400  # 90分钟总时长（原来是4200秒/70分钟）
        agent = CKAgent(**cfg)
        _disable_ckagent_default_tools(agent)
        return agent

    def _auto_curate_experiences(
        self, new_experiences: List[str], call_target: str
    ) -> None:
        """
        任务结束后自动整理经验池
        整理失败时保留原经验池，不影响主流程
        """
        try:
            auto_curate_experience_pool(
                experience_manager=self.experience_manager,
                new_experiences=new_experiences,
                call_target=call_target,
                max_items=20,
            )
        except Exception as exc:
            print(f"[warn] 经验池自动整理出错（已跳过）: {exc}")

    # ------------------------------------------------------------------ #
    def run(
        self,
        task_id: Optional[str] = None,
        *,
        index: Optional[int] = None,
        resume: bool = False,
    ) -> SeedRunResult:
        seed = self._find_seed(task_id=task_id, index=index)
        task_id = cast(str, seed.get("task_id") or f"seed_{index}")

        if resume and self.output_root.exists():
            candidates = []
            for p in self.output_root.iterdir():
                if p.is_dir() and p.name.startswith(f"{task_id}_"):
                    # 简单的名称匹配可能不够精确，但配合 timestamp 格式通常足够
                    # 检查是否存在 final_output.json 视为任务完成
                    if (p / "final_output.json").exists():
                        candidates.append(p)

            if candidates:
                # 按名称（时间戳）倒序排列，取最新的
                candidates.sort(key=lambda x: x.name, reverse=True)
                latest_dir = candidates[0]
                print(
                    f"Task {task_id} 已在 {latest_dir} 完成，跳过生成 (resume mode)。"
                )

                try:
                    final_output = json.loads(
                        (latest_dir / "final_output.json").read_text(encoding="utf-8")
                    )
                    return SeedRunResult(
                        task_id=task_id,
                        success=final_output.get("blind_success", False),
                        session_dir=latest_dir,
                        output_text=json.dumps(
                            final_output.get("output", {}), ensure_ascii=False
                        ),
                        parsed_output=final_output.get("output", {}),
                        blind_history=final_output.get("blind_history", []),
                        samples=final_output.get("samples", []),
                        experience_added=final_output.get("experience_added", []),
                        raw_log=final_output.get("log", ""),
                        raw_session=None,
                    )
                except Exception as e:
                    print(f"[warn] 加载已有结果失败 ({latest_dir}): {e}。将重新运行。")

        session_dir = self._create_session_dir(task_id)
        session_id = session_dir.name

        main_agent = self._create_agent(
            name=f"seed-main-{task_id}", call_target=self.model_call_target
        )
        quality_agent = self._create_agent(
            name=f"seed-quality-{task_id}", call_target=self.model_call_target
        )
        # 为 quality_agent 配置 system prompt，提醒使用 stop 工具
        _configure_agent_system_prompt(
            quality_agent,
            QUALITY_AGENT_SYSTEM_PROMPT,
        )

        blind_agent = self._create_agent(
            name=f"seed-blind-{task_id}", call_target=self.blind_model_call_target
        )
        # 为 blind_agent 配置 system prompt，提醒使用 stop 工具
        # 注意：这个 system prompt 会追加到 ckpro 原有的英文 system prompt 后面
    # Full system prompt = base English prompt (from ck_action template) + appended system prompt
        # 要查看完整的 prompt，可以设置环境变量 DEBUG_AGENT_PROMPT=1
        _configure_agent_system_prompt(
            blind_agent,
            BLIND_AGENT_SYSTEM_PROMPT,
        )
        # 使用 o4-mini 生成 auto_validator（与 blind_agent 使用相同的模型）
        validator_builder_llm = LLM(
            call_target=self.blind_model_call_target,
            max_token_num=20000,
        )

        context = SeedRunContext(
            seed=seed,
            experience_manager=self.experience_manager,
            validator_builder_llm=validator_builder_llm,
            blind_agent=blind_agent,
            quality_agent=quality_agent,
            session_id=session_id,
            blind_model_call_target=self.blind_model_call_target,
            blind_attempt_timeout=self.blind_attempt_timeout,
        )
        # 设置会话目录用于保存文件
        context.session_dir = session_dir

        for tool in context.build_tools():
            self._register_tool(main_agent, tool)

        def _guard(output_data):
            """检查题面自检与盲评是否完整通过，否则直接阻断 stop"""
            # 先确保题面质量检查已完成且通过
            qc_result = context.question_check_result
            if not qc_result:
                return (
                    False,
                    "尚未调用 seed_check_question_quality()。请先完成题面自检并确保返回 action=proceed。",
                )
            qc_action = (qc_result or {}).get("action") or qc_result.get("status")
            if qc_action and qc_action.lower() != "proceed":
                return (
                    False,
                    "题面检查仍为 revise 状态，请根据 feedback 调整题面后重新检查，直至 action=proceed。",
                )

            # 再检查盲评是否运行过
            if not context.blind_history:
                return (
                    False,
                    "尚未运行 seed_submit_blind_review() 或无任何盲评记录，请先完成盲评并确保成功。",
                )

            # 仅当盲评通过时才允许提交最终结果
            if context.blind_success:
                return True, ""

            context.stop_fail_count += 1
            if context.stop_fail_count >= 3:
                return (
                    False,
                    "连续 3 次 stop 被阻止：盲评仍未通过。请修复题面/代码直至 seed_submit_blind_review() 通过至少一次。",
                )
            return (
                False,
                "盲评尚未通过，禁止提交最终结果。请先调用 seed_submit_blind_review() 并确保成功，再次尝试 stop。",
            )

        self._install_stop_guard(main_agent, _guard)

        experiences = self.experience_manager.all()
        prompt = render_main_task(seed, experiences)

        with TaskTracer(
            task=f"{session_id}:main",
            agent_name="seed.main",
        ):
            result = _run_agent(main_agent, prompt)

        output_text = result["output"]
        log_text = result["log"]
        raw_session = result["session"]

        # 检查 output_text 是否为 None，如果是则尝试从 session 中提取
        if output_text is None:
            # 尝试从 session 的最后一步中提取 final_results
            try:
                last_step = raw_session.get_current_step() if raw_session else None
                if last_step and "end" in last_step:
                    end_block = last_step["end"]
                    final_results = end_block.get("final_results")
                    if final_results and isinstance(final_results, dict):
                        output_text = final_results.get("output")
                        if log_text is None or not log_text:
                            log_text = final_results.get("log", "")
            except Exception as e:
                print(f"[warn] 尝试从 session 提取输出失败: {e}")

        stop_accepted = context.blind_success
        stop_blocked_output: Optional[Any] = None

        if stop_accepted:
            try:
                parsed_output = (
                    output_text
                    if isinstance(output_text, dict)
                    else _ensure_json_dict(output_text)
                )
            except ValueError as e:
                # 如果解析失败，提供更详细的错误信息
                error_msg = str(e)
                if "None" in error_msg or output_text is None:
                    raise ValueError(
                        f"Agent 最终输出为 None。这可能是因为：\n"
                        f"1. `stop` 工具没有被正确调用\n"
                        f"2. `end` 阶段没有正确输出包含 'output' 和 'log' 字段的字典\n"
                        f"3. 请确保在最后一步使用 `stop(output=json.dumps({{...}}), log='...')` 输出有效的 JSON 字符串\n"
                        f"原始错误: {error_msg}"
                    ) from e
                raise
            experience_added = context.new_experiences or parsed_output.get(
                "experience_updates", []
            )
            if experience_added:
                self.experience_manager.add(experience_added)
                self._auto_curate_experiences(
                    new_experiences=experience_added,
                    call_target=main_agent.model.call_target,
                )
        else:
            parsed_output = {}
            experience_added = []
            stop_blocked_output = output_text

        blind_history = context.blind_history
        samples = context.samples

        artifact_payload = {
            "task_id": task_id,
            "prompt": prompt,
            "output": parsed_output,
            "log": log_text,
            "blind_history": blind_history,
            "samples": samples,
            "experience_added": experience_added,
            "blind_success": context.blind_success,
            "stop_guard_blocked": not stop_accepted,
            "question_quality_history": context.quality_history,
        }
        if stop_blocked_output is not None:
            artifact_payload["blocked_output"] = stop_blocked_output
        (session_dir / "final_output.json").write_text(
            json.dumps(artifact_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (session_dir / "raw_output.txt").write_text(str(output_text), encoding="utf-8")
        (session_dir / "blind_history.json").write_text(
            json.dumps(blind_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (session_dir / "samples.json").write_text(
            json.dumps(samples, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        success = stop_accepted

        return SeedRunResult(
            task_id=task_id,
            success=success,
            session_dir=session_dir,
            output_text=str(output_text),
            parsed_output=parsed_output,
            blind_history=blind_history,
            samples=samples,
            experience_added=experience_added,
            raw_log=log_text,
            raw_session=raw_session,
        )
