# Scaling the Scaling Logic

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **Scaling the Scaling Logic: An Agentic Meta-Synthesis Framework for Verifiable Logic Tasks**

**SSLogic** represents a paradigm shift from manual data curation to **agentic meta-synthesis**. Instead of merely generating static question-answer pairs, SSLogic synthesizes and evolves **executable programs** (Generators and Validators) that define entire families of logical tasks. This approach ensures infinite scalability, controllable difficulty, and rigorous verifiability.

<p align="center">
  <img src="figures/figure1.png" alt="From Manual Curation to Agentic Meta-Synthesis">
  <br>
  <em>Figure 1: From Manual Curation to Agentic Meta-Synthesis. SSLogic evolves task families through a closed Generate–Validate–Repair loop.</em>
</p>

---

## Key Features

- **Agentic Meta-Synthesis**: Shifts availability from instance-level automation to task-family synthesis. Agents write Python code to generate new logic problems.
- **Multi-Gate Validation Protocol**:
  - **Ensemble Consistency**: Uses multiple validator implementations to reduce bias.
  - **Adversarial Blind Review**: Independent code agents must be able to solve the task from the description alone.
- **Effective Scaling**: Scales 5,718 seed tasks to 15,671 evolved instances with controllable difficulty (no collapse).
- **Improved Training Dynamics**: Models trained on SSLogic data exhibit longer reasoning trajectories and deeper self-reflection.

## Performance & Impact

SSLogic significantly enhances the training value of synthetic data. Reinforcement Learning (RL) on SSLogic-evolved tasks yields stable gains across logic and math benchmarks.

| Metric               | Seed Baseline | **SSLogic Evolved** | Usage                 |
| :------------------- | :-----------: | :-----------------: | :-------------------- |
| **Logic (SynLogic)** |     14.6      |   **18.7** (+4.1)   | Direct Logic Training |
| **Math (AIME24)**    |     13.2      |   **17.3** (+4.1)   | Cross-Domain Transfer |

> _Results under fixed optimization steps (Step 240)._

<details>
<summary><strong>View Training Dynamics (Click to Expand)</strong></summary>

### Evolution of Reasoning

SSLogic training drives the model to develop longer reasoning chains and more frequent self-reflection tokens.

|                                                                              |                                                                            |
| :--------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| <img src="figures/figure3.png" width="400" alt="Reflection Token Frequency"> | <img src="figures/figure4.png" width="400" alt="Response Length Dynamics"> |
|                           **Reflection Frequency**                           |                            **Response Length**                             |

</details>

## Methodology

SSLogic operates on a **Generate-Validate-Repair** closed loop:

1.  **Synthesis**: An agent generates a `Generator` (creates problem instances) and a `Validator` (verifies solutions).
2.  **Gated Validation**:
    - _Gate 1_: Quality checks.
    - _Gate 2_: Adversarial Review – Can an independent agent solve it strictly from the text description?
3.  **Feedback-Driven Repair**: If circulation fails, the agent receives structured error logs and attempts to patch the code.

<p align="center">
  <img src="figures/figure6.png" width="80%" alt="Code Complexity Analysis">
  <br>
  <em>Code-level complexity analysis showing increased algorithmic depth in evolved tasks.</em>
</p>

## Installation

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
git clone https://github.com/your-username/sslogic.git
cd sslogic

uv sync
```

## Quick Start

The main entry point for the pipeline is `scripts/run_pipeline.py`.

### 1. Evolve a Seed Idea

Generate a new verifiable task family from a simple text description.

```bash
uv run scripts/run_pipeline.py \
    --seed "Calculate the sum of two numbers where one is double the other, subject to a modulo constraint." \
    --max-iterations 3 \
    --output-dir artifacts/my_experiment
```

### 2. Run from a Seed File

Process a batch of seed ideas defined in a JSONL file.

```bash
uv run scripts/run_pipeline.py \
    --seed-file data/seeds.jsonl \
    --seed-key "question" \
    --max-iterations 5
```

## Project Structure

```text
sslogic/
├── artifacts/             #  Generated Task Families (Code & Data)
├── eval/                  #  Evaluation Harness
├── figures/               #  Paper Figures & Visuals
├── scripts/               #  Utility Scripts
│   └── run_pipeline.py    #  Main Pipeline Entry Point
├── src/
│   └── sslogic/           #  Core Package
│       ├── pipeline/
│       └── ck_pro/
├── LICENSE
└── pyproject.toml
```

## Algorithmic Coverage

SSLogic evolves tasks that cover a wide range of algorithmic patterns, shifting distribution from simple linear logic to complex graph and dynamic programming problems.

<p align="center">
  <img src="figures/figure8.png" width="40%" alt="Algorithmic Patern Coverage">
  <br>
  <em>Algorithmic Patern Coverage</em>
</p>

## Citation

If you find SSLogic useful for your research, please cite our paper.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
