<div align="center">

# Scaling the Scaling Logic (SSLogic)
### Agentic Meta-Synthesis of Verifiable Logic Reasoning

**Bowen Liu, Zhi Wu, Runquan Xie, Zhanhui Kang, Jia Li**  
Tencent Hunyuan, Tencent | HKUST (Guangzhou)

[![arXiv](https://img.shields.io/badge/arXiv-2602.13218-b31b1b.svg)](https://arxiv.org/abs/2602.13218)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2602.13218-blue.svg)](https://doi.org/10.48550/arXiv.2602.13218)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)

**[Paper](https://arxiv.org/abs/2602.13218) | [Code](https://github.com/AdAstraAbyssoque/Scaling-the-Scaling-Logic)**

</div>

<p align="center">
  <img src="figures/Image%20(1).png" width="96%" alt="SSLogic multi-gate framework overview" />
</p>
<p align="center">
  <em>Figure 1. Multi-Gate Agentic Meta-Synthesis in SSLogic: a closed loop over task-family synthesis, consensus-based verification, blind review, and failure-driven refinement.</em>
</p>

## Abstract

Scaling verifiable training signals remains a core bottleneck for reinforcement learning from verifiable rewards (RLVR). Existing logic-data pipelines typically scale at the instance level (template perturbations and parameter sweeps), which limits structural diversity and long-horizon reasoning value. SSLogic moves scaling to the task-family level: agents iteratively synthesize and refine executable `Generator-Validator` program pairs in a Generate-Validate-Refine loop. To control data quality, SSLogic introduces a Multi-Gate Validation Protocol that combines static quality checks, consensus-based dynamic verification, and adversarial blind review. Starting from 400 human seed families, SSLogic expands to 953 families and 21,389 verifiable instances, yielding consistent gains under fixed-step RL training.

## Core Contributions

1. **Task-family-level scaling**: SSLogic treats executable task specifications `(G, V)` as evolvable objects, not fixed templates.
2. **Multi-Gate reliability control**: Quality gate + consensus validation + blind review reduce ambiguity, unsolvability, and implementation leakage.
3. **Downstream RL effectiveness**: Evolved data improves logic and math performance at matched optimization steps, with measurable trajectory-level behavior shifts.

## Framework Overview

SSLogic instantiates a three-phase agentic pipeline:

1. **Phase I - Context-Aware Specification Synthesis**
Main Agent receives a seed idea, injects reusable experience, and writes `generator.py` / `validator.py` with execution checks.

2. **Phase II - Multi-Gate Validation Protocol**
- Gate 1: static quality assurance (format, clarity, solvability signals).
- Gate 2: consensus-based dynamic verification across independent validators.
- Blind Review: independent agents solve from text-only statements; acceptance requires thresholded agreement with canonical answers.

3. **Phase III - Feedback-Driven Refinement**
Failed candidates trigger structured debugging (ambiguity, algorithmic bug, logic gap), then re-enter refinement until acceptance or stop.

## Empirical Snapshot

### Data Scaling

| Metric | Seed | Evolved | Delta |
| :-- | --: | --: | --: |
| Task families | 400 | **953** | +553 |
| Verifiable instances | 5,718 | **21,389** | +15,671 |

### RL Gains (paper-reported, matched-step setting)

| Benchmark | Improvement |
| :-- | --: |
| SynLogic | **+5.2** |
| BBEH | **+1.4** |
| AIME25 | **+3.0** |
| Brumo25 | **+3.7** |

### Pipeline Diagnostics (100 sampled synthesis traces)

| Diagnostic | Value |
| :-- | --: |
| Gate 1 pass rate | 67.0% |
| Gate 2 consensus OK | 93.0% |
| Blind-review pass (overall acceptance) | **55.0%** |
| Runtime gap (rejected vs accepted) | **6.5x** |
| Amortized cost per accepted family | **$1.18** |

## Analysis Glimpses

<p align="center">
  <img src="figures/figure6.png" width="78%" alt="Code complexity metrics across seed and evolved sources" />
</p>
<p align="center">
  <em>Code-level complexity analysis: evolved task families show richer control flow and algorithmic depth.</em>
</p>

<p align="center">
  <img src="figures/figure3.png" width="42%" alt="Reflection token frequency" />
  <img src="figures/figure4.png" width="42%" alt="Response length dynamics" />
</p>
<p align="center">
  <em>Training dynamics: reflection-like signals and response length co-evolve with logic-data training.</em>
</p>

## Installation

```bash
git clone https://github.com/AdAstraAbyssoque/Scaling-the-Scaling-Logic.git
cd Scaling-the-Scaling-Logic

uv sync
uv pip install -e .
```

## Quick Start

### 1) Run a single-seed synthesis session

```bash
uv run python -m sslogic.pipeline.run_pipeline \
  --seed "Design a verifiable logic task involving constrained swaps on a sequence" \
  --max-iterations 3
```

### 2) Run from a JSONL seed file

```bash
uv run python -m sslogic.pipeline.run_pipeline \
  --seed-file src/sslogic/pipeline/example/logic_reasoning.jsonl \
  --seed-id Add_one_eliminate_20250919 \
  --seed-key question \
  --max-iterations 3 \
  --output src/sslogic/pipeline/example/answer/answer.jsonl
```

### 3) Override model backend

```bash
uv run python -m sslogic.pipeline.run_pipeline \
  --seed "Construct a shortest-path logic puzzle with hidden constraints" \
  --model openai:gpt-4o-mini
```

## Output Artifacts

Each run writes structured artifacts under:

```text
src/sslogic/pipeline/artifacts/session/<xx>/<session_id>/
```

Typical outputs include:

- `propose-*.json`
- `execute-iter*.json`
- validator and blind-review traces (`*.json`, `*.log`)
- `final_answer.json`

## Repository Layout

```text
.
|-- README.md
|-- pyproject.toml
|-- figures/
|-- scripts/
|   `-- run_pipeline.py
|-- src/
|   `-- sslogic/
|       |-- pipeline/      # Core multi-agent synthesis pipeline
|       `-- ck_pro/        # Cognitive Kernel-Pro based agent stack
`-- eval/
```

## Citation

```bibtex
@article{liu2026sslogic,
  title   = {Scaling the Scaling Logic: Agentic Meta-Synthesis of Logic Reasoning},
  author  = {Bowen Liu and Zhi Wu and Runquan Xie and Zhanhui Kang and Jia Li},
  journal = {arXiv preprint arXiv:2602.13218},
  year    = {2026},
  doi     = {10.48550/arXiv.2602.13218},
  url     = {https://arxiv.org/abs/2602.13218}
}
```

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
