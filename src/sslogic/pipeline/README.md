## 任务合成（SSLogic）

本目录实现“Propose → Execute → Validate/Blind Review → Revision”多智能体范式，
基于 CognitiveKernel-Pro (`CKAgent`) 的主/子代理架构，保留论文中描述的全部
关键组件与流程约束。

### 组件总览

- **主代理（CKPro Main Agent）**：一个 `CKAgent` 实例，负责生成与改写题目。
  - **Propose**：生成基线题、完整解法与难度评估，并规划多个演化方向。
  - **Execute**：选择演化方向生成新题目与官方解答，按要求调用验证工具。
  - **Revision**：在校验与盲评反馈存在阻塞时触发修订。
- **验证器（Validator V1/V2/...）**：多个 `CKAgent` 子实例，聚焦题面完备性、
  可解性与边界覆盖，输出 `verdict`、阻塞问题、警告与建议。
- **盲评代理（Blind Review Agent）**：独立 `CKAgent`，仅凭题面求解并给出最终答案，
  用于对比官方答案的一致性。
- **Experience / Context Playbook**：主代理的进度状态中包含 `experience` 与
  `information` 列表，用于积累成功模式、失败经验与复用策略。

### 代码结构

| 文件              | 说明                                                                      |
| ----------------- | ------------------------------------------------------------------------- |
| `__init__.py`     | 导出 `SSLogicPipeline` 及辅助类型                                         |
| `agents.py`       | 对 `CKAgent` 的轻量封装，便于统一调用                                     |
| `prompts.py`      | 构建 Propose / Execute / Revision / Validator / Blind Review 所需提示语    |
| `prompts/`        | 原始提示词模板（YAML）                                                     |
| `pipeline.py`     | 多轮协同管线实现，包含验证工具与 stop guard                               |
| `utils.py`        | JSON 解析等通用工具                                                       |
| `run_pipeline.py` | CLI，便于直接运行整个流程                                                 |

### Phoenix 追踪

`SSLogicPipeline` 在初始化时会调用 `ck_pro.agents.phoenix_tracer.init_phoenix_tracing`
（项目名 `sslogic`）。确保安装 `arize-phoenix` 与
`openinference-instrumentation-smolagents`，并以 `PHOENIX_ENABLE=true` 启动 CLI：

```bash
PHOENIX_ENABLE=true uv run python -m sslogic.pipeline.run_pipeline --seed "..."
```

若成功启用，最终 JSON 会包含 `"phoenix_enabled": true`，所有代理调用会写入 Phoenix DB
（默认端点 `http://localhost:6006`）。

### 快速开始

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
uv run python -m sslogic.pipeline.run_pipeline --seed "A logic puzzle"
```

- `--seed`：可选的原始灵感；为空时主代理会自行构造基线题。
- `--seed-file`：指向 JSONL 文件时，将按 `--seed-id` 或 `--seed-index` 选中记录，
  并提取指定字段（默认 `question`）。
- `--seed-id` / `--seed-index` / `--seed-key`：结合 `--seed-file` 精确挑选种子文本。
- `--max-iterations`：主代理在验证失败后允许的修订轮数（≥1）。

示例：

```bash
uv run python -m sslogic.pipeline.run_pipeline \
  --seed-file sslogic/src/sslogic/pipeline/example/logic_reasoning.jsonl \
  --seed-id Add_one_eliminate_20250919 \
  --output sslogic/src/sslogic/pipeline/example/answer/answer.jsonl
```

CLI 会打印最终的结构化 JSON，其中包含：

- `status`：`completed` 或 `needs_manual_review`
- `propose`：基线题与演化规划
- `final_evolved`：最终题目与官方答案
- `history`：每次迭代的通过情况摘要

### 最小可运行路径（Mock）

默认 `call_target` 为 `mock`，可直接离线跑通流程并生成结构化输出：

```bash
uv run python -m sslogic.pipeline.run_pipeline --seed "mock seed"
```

如需真实模型调用，请使用 `--model` 覆盖或通过配置文件指定。

> Note: All prompt templates instruct agents to reason and respond in English for consistent review.
