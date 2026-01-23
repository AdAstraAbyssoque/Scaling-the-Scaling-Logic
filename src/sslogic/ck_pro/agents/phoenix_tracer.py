"""
Phoenix Tracer 模块 - CognitiveKernel-Pro 轻量化监控
基于 code_coach 的 Phoenix 实现，为 CK-Pro 提供可观测性支持
"""

import os
from typing import Optional

# Phoenix 相关配置
PHOENIX_COLLECTOR_ENDPOINT = "http://localhost:6006"

# 默认开启 Phoenix，除非显式设置 PHOENIX_ENABLE=false/0/no
PHOENIX_ENABLE = os.getenv("PHOENIX_ENABLE", "true").lower() in (
    "true",
    "1",
    "yes",
)

# 全局状态
_tracer_provider = None
_instrumentor = None
_is_initialized = False


def init_phoenix_tracing(
    project_name: str = "ck-pro",
    endpoint: str = None,
    enable: bool = None,
) -> bool:
    """
    初始化 Phoenix instrumentation

    Args:
        project_name: Phoenix 项目名称
        endpoint: Phoenix 收集器端点
        enable: 是否启用追踪（None 时使用环境变量）

    Returns:
        是否成功初始化
    """
    global _tracer_provider, _instrumentor, _is_initialized

    # 检查是否启用
    if enable is None:
        enable = PHOENIX_ENABLE

    if not enable:
        print("Phoenix tracing 未启用（设置 PHOENIX_ENABLE=true 以启用）")
        return False

    # 避免重复初始化
    if _is_initialized:
        print("Phoenix instrumentation 已经初始化")
        return True

    try:
        from phoenix.otel import register
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor

        # 设置端点
        if endpoint is None:
            endpoint = f"{PHOENIX_COLLECTOR_ENDPOINT}/v1/traces"

        # 注册 Phoenix
        _tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint,
            auto_instrument=True,
        )

        # 初始化 SmolagentsInstrumentor（如果项目使用 smolagents）
        _instrumentor = SmolagentsInstrumentor()
        if not _instrumentor.is_instrumented_by_opentelemetry:
            _instrumentor.instrument()

        _is_initialized = True
        print(f"✓ Phoenix instrumentation 已初始化")
        print(f"  项目: {project_name}")
        print(f"  端点: {endpoint}")
        print(f"  UI: {PHOENIX_COLLECTOR_ENDPOINT}")

        return True

    except ImportError as e:
        print(f"✗ Phoenix 依赖未安装: {e}")
        print(
            "  请运行: uv pip install arize-phoenix openinference-instrumentation-smolagents"
        )
        return False
    except Exception as e:
        print(f"✗ Phoenix instrumentation 初始化失败: {e}")
        return False


def shutdown_phoenix_tracing():
    """关闭 Phoenix instrumentation"""
    global _instrumentor, _is_initialized

    if not _is_initialized:
        return

    try:
        if _instrumentor and _instrumentor.is_instrumented_by_opentelemetry:
            _instrumentor.uninstrument()
        _is_initialized = False
        print("Phoenix instrumentation 已关闭")
    except Exception as e:
        print(f"关闭 Phoenix instrumentation 时出错: {e}")


def is_phoenix_enabled() -> bool:
    """检查 Phoenix 是否已启用并初始化"""
    return _is_initialized


def get_tracer_provider():
    """获取 tracer provider"""
    return _tracer_provider


def get_tracer(name: str = "ck-pro"):
    """获取 tracer 实例"""
    if not _is_initialized:
        return None
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:
        return None


def trace_llm_call(func):
    """
    装饰器：为 LLM 调用添加追踪

    遵循 OpenInference 语义约定，详细记录 LLM 输入输出

    使用方法：
        @trace_llm_call
        def call_llm(messages, **kwargs):
            ...
    """

    def wrapper(*args, **kwargs):
        if not _is_initialized:
            return func(*args, **kwargs)

        try:
            import json
            from opentelemetry import trace
            from opentelemetry.trace import Status, StatusCode

            tracer = trace.get_tracer("ck-pro.llm")

            # 提取模型信息和调用参数
            model_name = kwargs.get("model", "unknown")
            provider = "unknown"
            llm_instance = None

            if hasattr(args[0], "call_target"):
                llm_instance = args[0]
                model_name = llm_instance.call_target
                if "api:" in str(model_name):
                    provider = "custom_api"
                elif "gpt:" in str(model_name):
                    provider = "openai"
                elif "claude" in str(model_name):
                    provider = "anthropic"

            # 提取 messages（第一个参数通常是 self，第二个是 messages）
            messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])

            # 收集所有调用参数（包括 temperature, top_p, max_tokens 等）
            invocation_params = {
                "model": model_name,
                "provider": provider,
            }

            # 从 LLM 实例获取默认参数
            if llm_instance and hasattr(llm_instance, "call_kwargs"):
                invocation_params.update(llm_instance.call_kwargs)

            # 添加本次调用的额外参数
            for k, v in kwargs.items():
                if k not in ["messages"]:  # 排除 messages，单独处理
                    invocation_params[k] = v

            # 添加 seed 和其他 LLM 属性
            if llm_instance:
                if hasattr(llm_instance, "seed") and llm_instance.seed != 0:
                    invocation_params["seed"] = llm_instance.seed
                if hasattr(llm_instance, "thinking"):
                    invocation_params["thinking"] = llm_instance.thinking

            # 预处理 messages 以便在 span 创建时就包含
            messages_str = ""
            user_msg = ""
            system_msg = ""
            message_count = 0
            conversation_history = []  # 前文对话历史
            current_user_input = ""  # 本轮新输入

            if messages:
                message_count = len(messages) if isinstance(messages, list) else 0

                # 分离前文和本轮输入
                for i, msg in enumerate(messages):
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")

                        # 处理字符串或列表格式的 content
                        if isinstance(content, list):
                            content_text = " ".join(
                                [
                                    (
                                        item.get("text", "")
                                        if isinstance(item, dict)
                                        and item.get("type") == "text"
                                        else ""
                                    )
                                    for item in content
                                ]
                            )
                        else:
                            content_text = str(content)

                        # 最后一条 user message 是本轮输入
                        if role == "user" and i == len(messages) - 1:
                            current_user_input = content_text
                        elif role == "system":
                            system_msg = (
                                content_text[:1000]
                                if len(content_text) > 1000
                                else content_text
                            )
                        else:
                            # 其他消息作为前文历史
                            conversation_history.append(
                                {"role": role, "content": content_text}
                            )

            from opentelemetry.trace import SpanKind

            # 创建初始 attributes（包含输入信息）
            initial_attributes = {
                # OpenInference 语义约定 - 用于 Phoenix UI 识别
                "openinference.span.kind": "LLM",  # 关键：标记为 LLM 类型
                "llm.model_name": model_name,
                "llm.provider": provider,
                "llm.invocation_parameters": json.dumps(
                    invocation_params, ensure_ascii=False, default=str
                ),
            }

            # 添加输入信息（在 span 创建时就设置）
            # 优先展示本轮输入，前文历史单独展示
            if current_user_input:
                initial_attributes["input.value"] = (
                    current_user_input  # 主输入：本轮新消息（不截断）
                )
                initial_attributes["llm.current_user_input"] = current_user_input

            if conversation_history:
                # 前文历史用 JSON 格式存储（Phoenix 会自动展开，不需要截断）
                initial_attributes["llm.conversation_history"] = json.dumps(
                    conversation_history, ensure_ascii=False, default=str
                )

            if system_msg:
                initial_attributes["llm.prompt_system"] = system_msg

            if message_count > 0:
                initial_attributes["llm.message_count"] = message_count

            # 完整 messages 用于调试（JSON 格式，Phoenix 自动展开）
            if messages:
                initial_attributes["llm.input_messages_full"] = json.dumps(
                    messages, ensure_ascii=False, default=str
                )

            with tracer.start_as_current_span(
                model_name,  # 直接使用模型名作为 span 名称
                kind=SpanKind.INTERNAL,
                attributes=initial_attributes,
            ) as span:
                try:
                    # 记录调用时间
                    import time, re

                    start_time = time.time()

                    # 记录调用前的 token 统计（用于计算本次调用的增量）
                    prev_tokens = {}
                    if llm_instance and hasattr(llm_instance, "call_stat"):
                        prev_tokens = {
                            "prompt_tokens": llm_instance.call_stat.get(
                                "prompt_tokens", 0
                            ),
                            "completion_tokens": llm_instance.call_stat.get(
                                "completion_tokens", 0
                            ),
                            "total_tokens": llm_instance.call_stat.get(
                                "total_tokens", 0
                            ),
                        }

                    # 执行 LLM 调用
                    result = func(*args, **kwargs)

                    # 记录调用耗时
                    duration = time.time() - start_time
                    span.set_attribute("llm.duration_seconds", round(duration, 3))

                    # 记录 Token 使用情况（OpenInference 标准字段）
                    # Phoenix UI 会自动在 AGENT span 层级聚合累计 token 统计
                    if llm_instance and hasattr(llm_instance, "call_stat"):
                        current_tokens = llm_instance.call_stat
                        input_tokens = current_tokens.get(
                            "prompt_tokens", 0
                        ) - prev_tokens.get("prompt_tokens", 0)
                        output_tokens = current_tokens.get(
                            "completion_tokens", 0
                        ) - prev_tokens.get("completion_tokens", 0)
                        total_tokens = current_tokens.get(
                            "total_tokens", 0
                        ) - prev_tokens.get("total_tokens", 0)

                        # OpenInference 标准字段（必须使用整数类型）
                        span.set_attribute(
                            "llm.token_count.prompt",
                            int(input_tokens) if input_tokens else 0,
                        )
                        span.set_attribute(
                            "llm.token_count.completion",
                            int(output_tokens) if output_tokens else 0,
                        )
                        span.set_attribute(
                            "llm.token_count.total",
                            int(total_tokens) if total_tokens else 0,
                        )

                        # 调试信息（可以在 Phoenix span attributes 中查看）
                        span.set_attribute(
                            "debug.token_stats",
                            f"prompt={input_tokens}, completion={output_tokens}, total={total_tokens}",
                        )

                    span.set_status(Status(StatusCode.OK))

                    # 记录输出信息（使用 Phoenix UI 优先显示的字段名）
                    if isinstance(result, str):
                        # 检测是否包含 thinking/reasoning
                        has_thinking = "Thought:" in result or "Reasoning:" in result

                        # 主输出：完整响应（不截断，Phoenix 有展开功能）
                        span.set_attribute("output.value", result)
                        span.set_attribute("llm.response_length", len(result))
                        span.set_attribute("llm.has_thinking", has_thinking)

                        # 提取结构化信息（方便快速查看）
                        import re

                        # 提取 thought 部分（如果有）
                        if has_thinking:
                            thought_match = re.search(
                                r"(?:Thought|Reasoning):\s*(.+?)(?:\n(?:Action|Code|Final Answer):|$)",
                                result,
                                re.IGNORECASE | re.DOTALL,
                            )
                            if thought_match:
                                thought_text = thought_match.group(1).strip()
                                span.set_attribute(
                                    "llm.thought", thought_text
                                )  # 不截断

                        # 提取 Code 部分（如果有）
                        code_match = re.search(
                            r"(?:Code|Action):\s*```(?:python)?\s*(.+?)\s*```",
                            result,
                            re.IGNORECASE | re.DOTALL,
                        )
                        if code_match:
                            code_text = code_match.group(1).strip()
                            span.set_attribute("llm.code", code_text)  # 不截断

                        # 提取 Final Answer（如果有）
                        final_answer_match = re.search(
                            r"Final Answer:\s*(.+)", result, re.IGNORECASE | re.DOTALL
                        )
                        if final_answer_match:
                            final_answer = final_answer_match.group(1).strip()
                            span.set_attribute("llm.final_answer", final_answer)

                        # 如果响应很长，添加简短预览（500字符）
                        if len(result) > 1000:
                            span.set_attribute(
                                "llm.response_preview", result[:500] + "..."
                            )

                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        except ImportError:
            # OpenTelemetry 未安装，直接调用原函数
            return func(*args, **kwargs)

    return wrapper


class TaskTracer:
    """
    任务级别的追踪器
    使用 context manager 来自动管理 span 的生命周期
    """

    def __init__(self, task: str, agent_name: str = "ck-agent"):
        self.task = task
        self.agent_name = agent_name
        self.span = None
        self.token = None

    def __enter__(self):
        if not _is_initialized:
            return self

        try:
            from opentelemetry import trace, context

            tracer = trace.get_tracer("ck-pro.agent")

            from opentelemetry.trace import SpanKind

            # 创建新的 span 并设置为当前 context
            self.span = tracer.start_span(
                f"agent.{self.agent_name}",  # 使用 agent 名称
                kind=SpanKind.INTERNAL,
                attributes={
                    # OpenInference 语义约定
                    "openinference.span.kind": "AGENT",  # 标记为 Agent 类型
                    "agent.name": self.agent_name,
                    "agent.task": (
                        self.task[:1000] if len(self.task) > 1000 else self.task
                    ),
                },
            )
            # 使 span 成为当前 context，这样子 span 会自动成为它的 child
            self.token = context.attach(trace.set_span_in_context(self.span))

        except Exception as e:
            print(f"Failed to start task span: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not _is_initialized or not self.span:
            return False

        try:
            from opentelemetry import context
            from opentelemetry.trace import Status, StatusCode

            if exc_type:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(Status(StatusCode.OK))

            self.span.end()

            # 恢复之前的 context
            if self.token:
                context.detach(self.token)

        except Exception as e:
            print(f"Failed to end task span: {e}")

        return False


class StepTracer:
    """
    步骤级别的追踪器 (REACT 循环)
    自动成为当前 task span 的子 span
    """

    def __init__(self, step_idx: int, step_type: str = "react"):
        self.step_idx = step_idx
        self.step_type = step_type
        self.span = None
        self.token = None

    def __enter__(self):
        if not _is_initialized:
            return self

        try:
            from opentelemetry import trace, context

            tracer = trace.get_tracer("ck-pro.agent")

            from opentelemetry.trace import SpanKind

            # 在当前 context 下创建子 span
            self.span = tracer.start_span(
                f"step_{self.step_idx:02d}",  # 简化名称
                kind=SpanKind.INTERNAL,
                attributes={
                    # OpenInference 语义约定
                    "openinference.span.kind": "CHAIN",  # 标记为 Chain/Step 类型
                    "step.index": self.step_idx,
                    "step.type": self.step_type,
                },
            )
            self.token = context.attach(trace.set_span_in_context(self.span))

        except Exception as e:
            print(f"Failed to start step span: {e}")

        return self

    def add_planning(self, thought: str, state: dict = None):
        """添加规划阶段的内容"""
        if not self.span or not _is_initialized:
            return
        try:
            import json

            # 只作为 span attributes（主视图可见）- 遵循简洁原则
            self.span.set_attribute(
                "react.planning.thought",
                thought[:1500] if len(thought) > 1500 else thought,
            )
            if state:
                # 提取关键信息避免过长
                state_str = json.dumps(state, ensure_ascii=False)
                self.span.set_attribute(
                    "react.planning.state",
                    state_str[:800] if len(state_str) > 800 else state_str,
                )

                # 如果有 todo_list 和 completed_list，单独记录
                if isinstance(state, dict):
                    if "todo_list" in state:
                        self.span.set_attribute(
                            "react.planning.todo", str(state["todo_list"])[:500]
                        )
                    if "completed_list" in state:
                        self.span.set_attribute(
                            "react.planning.completed",
                            str(state["completed_list"])[:500],
                        )
        except Exception as e:
            print(f"Failed to add planning: {e}")

    def add_action(self, thought: str, action_code: str):
        """添加动作阶段的内容"""
        if not self.span or not _is_initialized:
            return
        try:
            # 只作为 span attributes（主视图可见）
            self.span.set_attribute(
                "react.action.thought",
                thought[:1500] if len(thought) > 1500 else thought,
            )
            self.span.set_attribute(
                "react.action.code",
                action_code[:2000] if len(action_code) > 2000 else action_code,
            )

            # 提取工具名称（从代码中解析，如 tool_name(...) 格式）
            import re

            tool_match = re.search(r"(\w+)\s*\(", action_code)
            if tool_match:
                self.span.set_attribute("react.action.tool", tool_match.group(1))
        except Exception as e:
            print(f"Failed to add action: {e}")

    def add_observation(self, observation: str):
        """添加观察结果"""
        if not self.span or not _is_initialized:
            return
        try:
            obs_str = (
                str(observation) if not isinstance(observation, str) else observation
            )

            # 只作为 span attribute（主视图可见）
            self.span.set_attribute(
                "react.observation.result",
                obs_str[:2000] if len(obs_str) > 2000 else obs_str,
            )

            # 记录结果长度和是否成功（用于快速分析）
            self.span.set_attribute("react.observation.length", len(obs_str))

            # 判断是否包含错误信息
            is_error = any(
                keyword in obs_str.lower()
                for keyword in [
                    "error",
                    "exception",
                    "failed",
                    "traceback",
                    "code execution error",
                ]
            )
            self.span.set_attribute("react.observation.has_error", is_error)

            # 如果是错误,提取详细错误信息
            if is_error:
                from opentelemetry.trace import Status, StatusCode

                # 设置 span 状态为 ERROR
                self.span.set_status(Status(StatusCode.ERROR, "Step execution error"))

                # 提取错误类型和消息
                import re

                # 提取 "Code Execution Error:" 后面的内容
                if "Code Execution Error:" in obs_str:
                    error_content = obs_str.split("Code Execution Error:", 1)[1].strip()
                    self.span.set_attribute(
                        "react.observation.error_details",
                        (
                            error_content[:1500]
                            if len(error_content) > 1500
                            else error_content
                        ),
                    )

                    # 提取具体的异常类型 (如 ValueError, TypeError 等)
                    exception_match = re.search(r"(\w+Error|Exception):", error_content)
                    if exception_match:
                        self.span.set_attribute(
                            "react.observation.error_type", exception_match.group(1)
                        )

                    # 提取错误的代码行
                    line_match = re.search(r"line (\d+)", error_content)
                    if line_match:
                        self.span.set_attribute(
                            "react.observation.error_line", int(line_match.group(1))
                        )

                # 完整的错误信息(不截断)存储为 event
                self.span.add_event(
                    "execution_error",
                    attributes={
                        "error.message": (
                            obs_str[:3000] if len(obs_str) > 3000 else obs_str
                        ),
                        "error.type": "code_execution",
                    },
                )
        except Exception as e:
            print(f"Failed to add observation: {e}")

    def add_llm_io(self, phase: str, messages: list, response: str):
        """添加 LLM 输入输出（这个方法现在可能不需要了，因为 trace_llm_call 装饰器已经处理）"""
        if not self.span or not _is_initialized:
            return
        try:
            import json

            # 提取用户消息和系统消息用于快速查看
            user_message = ""
            system_message = ""
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get("role") == "user":
                        user_message = msg.get("content", "")
                    elif msg.get("role") == "system":
                        system_message = msg.get("content", "")

            messages_str = json.dumps(messages, ensure_ascii=False)

            # 只作为 span attributes（主视图可见）
            # 使用更清晰的命名，与 trace_llm_call 中的 LLM span 区分开
            self.span.set_attribute(
                f"react.{phase}.prompt",
                user_message[:1000] if len(user_message) > 1000 else user_message,
            )
            self.span.set_attribute(
                f"react.{phase}.response",
                response[:1000] if len(response) > 1000 else response,
            )
            self.span.set_attribute(
                f"react.{phase}.messages_full",
                messages_str[:2000] if len(messages_str) > 2000 else messages_str,
            )
        except Exception as e:
            print(f"Failed to add LLM I/O: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not _is_initialized or not self.span:
            return False

        try:
            from opentelemetry import context
            from opentelemetry.trace import Status, StatusCode

            if exc_type:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(Status(StatusCode.OK))

            self.span.end()

            if self.token:
                context.detach(self.token)

        except Exception as e:
            print(f"Failed to end step span: {e}")

        return False


class ToolTracer:
    """
    工具调用追踪器
    用于追踪工具的输入、输出和执行过程
    """

    def __init__(self, tool_name: str, args: tuple = None, kwargs: dict = None):
        self.tool_name = tool_name
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.span = None
        self.token = None

    def __enter__(self):
        if not _is_initialized:
            return self

        try:
            import json
            from opentelemetry import trace, context
            from opentelemetry.trace import SpanKind

            tracer = trace.get_tracer("ck-pro.tool")

            # 创建工具调用 span（使用简洁的名称，设置为 TOOL span kind）
            self.span = tracer.start_span(
                self.tool_name,  # 直接使用工具名作为 span 名称
                kind=SpanKind.INTERNAL,  # 标记为内部调用
                attributes={
                    # OpenInference 语义约定 - 用于 Phoenix UI 识别
                    "openinference.span.kind": "TOOL",  # 关键：标记为工具类型
                    "tool.name": self.tool_name,
                    "tool.description": f"Tool: {self.tool_name}",
                },
            )

            # 记录输入参数（完整保留，符合 Phoenix UI 显示要求）
            input_parts = []

            if self.args:
                args_str = json.dumps(self.args, ensure_ascii=False, default=str)
                self.span.set_attribute(
                    "tool.parameters.args",
                    args_str[:3000] if len(args_str) > 3000 else args_str,
                )
                input_parts.append(args_str)

            if self.kwargs:
                kwargs_str = json.dumps(self.kwargs, ensure_ascii=False, default=str)
                self.span.set_attribute(
                    "tool.parameters.kwargs",
                    kwargs_str[:3000] if len(kwargs_str) > 3000 else kwargs_str,
                )
                input_parts.append(kwargs_str)

                # 特别记录常见的参数（方便快速查看）
                if "query" in self.kwargs:
                    query_val = str(self.kwargs["query"])
                    self.span.set_attribute(
                        "tool.parameters.query",
                        query_val[:1500] if len(query_val) > 1500 else query_val,
                    )
                if "task" in self.kwargs:
                    task_val = str(self.kwargs["task"])
                    self.span.set_attribute(
                        "tool.parameters.task",
                        task_val[:1500] if len(task_val) > 1500 else task_val,
                    )

            # 设置统一的输入字段（Phoenix UI 主要显示这个）
            if input_parts:
                combined_input = " | ".join(input_parts)
                self.span.set_attribute(
                    "input.value",
                    (
                        combined_input[:5000]
                        if len(combined_input) > 5000
                        else combined_input
                    ),
                )

            self.token = context.attach(trace.set_span_in_context(self.span))

        except Exception as e:
            print(f"Failed to start tool span: {e}")

        return self

    def set_output(self, output):
        """设置工具输出（完整保留，符合 Phoenix UI 显示要求）"""
        if not self.span or not _is_initialized:
            return
        try:
            output_str = str(output)
            output_len = len(output_str)

            # 设置多个字段确保在 Phoenix UI 中可见
            self.span.set_attribute(
                "output.value", output_str[:5000] if output_len > 5000 else output_str
            )
            self.span.set_attribute(
                "tool.output", output_str[:3000] if output_len > 3000 else output_str
            )
            self.span.set_attribute("tool.output_length", output_len)

            # 如果输出过长，记录摘要
            if output_len > 5000:
                self.span.set_attribute("tool.output_truncated", True)
                self.span.set_attribute("tool.output_preview", output_str[:500] + "...")
        except Exception as e:
            print(f"Failed to set tool output: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not _is_initialized or not self.span:
            return False

        try:
            from opentelemetry import context
            from opentelemetry.trace import Status, StatusCode

            if exc_type:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
                self.span.set_attribute("tool.error", str(exc_val)[:500])
            else:
                self.span.set_status(Status(StatusCode.OK))

            self.span.end()

            if self.token:
                context.detach(self.token)

        except Exception as e:
            print(f"Failed to end tool span: {e}")

        return False


class SubAgentTracer:
    """
    Sub-Agent 调用追踪器
    用于追踪子 agent 的任务和执行过程
    """

    def __init__(self, agent_name: str, task: str):
        self.agent_name = agent_name
        self.task = task
        self.span = None
        self.token = None

    def __enter__(self):
        if not _is_initialized:
            return self

        try:
            from opentelemetry import trace, context

            tracer = trace.get_tracer("ck-pro.agent")

            # 创建 sub-agent 调用 span
            self.span = tracer.start_span(
                f"agent.call.{self.agent_name}",
                attributes={
                    "agent.name": self.agent_name,
                    "agent.type": "sub_agent",
                    "agent.task": (
                        self.task[:1000] if len(self.task) > 1000 else self.task
                    ),
                },
            )

            self.token = context.attach(trace.set_span_in_context(self.span))

        except Exception as e:
            print(f"Failed to start sub-agent span: {e}")

        return self

    def set_result(self, result):
        """设置 sub-agent 结果"""
        if not self.span or not _is_initialized:
            return
        try:
            result_str = str(result)
            self.span.set_attribute(
                "agent.result",
                result_str[:2000] if len(result_str) > 2000 else result_str,
            )
            self.span.set_attribute("agent.result_length", len(result_str))
        except Exception as e:
            print(f"Failed to set agent result: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not _is_initialized or not self.span:
            return False

        try:
            from opentelemetry import context
            from opentelemetry.trace import Status, StatusCode

            if exc_type:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
                self.span.set_attribute("agent.error", str(exc_val)[:500])
            else:
                self.span.set_status(Status(StatusCode.OK))

            self.span.end()

            if self.token:
                context.detach(self.token)

        except Exception as e:
            print(f"Failed to end sub-agent span: {e}")

        return False


# 便捷函数：用于上下文管理器
class PhoenixContext:
    """Phoenix 追踪上下文管理器"""

    def __init__(self, project_name: str = "ck-pro", **kwargs):
        self.project_name = project_name
        self.kwargs = kwargs
        self.initialized = False

    def __enter__(self):
        self.initialized = init_phoenix_tracing(
            project_name=self.project_name, **self.kwargs
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.initialized:
            shutdown_phoenix_tracing()


# 自动初始化（可选）
def auto_init_if_enabled():
    """如果环境变量启用，自动初始化 Phoenix"""
    if PHOENIX_ENABLE and not _is_initialized:
        init_phoenix_tracing()


if __name__ == "__main__":
    # 测试代码
    print("测试 Phoenix Tracer 模块...")
    print(f"环境变量 PHOENIX_ENABLE: {os.getenv('PHOENIX_ENABLE', 'not set')}")

    # 强制启用进行测试
    result = init_phoenix_tracing(project_name="ck-pro-test", enable=True)
    print(f"初始化结果: {result}")

    if result:
        print(f"Phoenix 已启用: {is_phoenix_enabled()}")
        shutdown_phoenix_tracing()
