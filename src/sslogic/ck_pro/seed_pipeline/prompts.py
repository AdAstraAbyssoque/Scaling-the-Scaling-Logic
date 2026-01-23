from __future__ import annotations

from typing import Dict, List, Optional


def _trim_text(text: str, max_chars: int = 300) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]} ... (truncated; original length {len(text)})"


def _select_recent_items(items: List[str], limit: Optional[int]) -> List[str]:
    if limit is None or limit <= 0 or len(items) <= limit:
        return items
    return items[-limit:]


def _format_experience_highlights(experiences: List[str], limit: int = 20) -> str:
    if not experiences:
        return "1. (No shared experience yet. Record new experience after finishing for reuse.)"
    selected = _select_recent_items(experiences, limit)
    return "\n".join(f"{idx + 1}. {exp}" for idx, exp in enumerate(selected, start=1))


def _format_experience_bullets(experiences: List[str], limit: int = 20) -> str:
    if not experiences:
        return "(No shared experience yet. Record new experience after finishing for reuse.)"
    selected = _select_recent_items(experiences, limit)
    return "\n".join(f"- {exp}" for exp in selected)


def _trim_code_block(code: str, max_lines: int = 40) -> str:
    if not code:
        return "# (no code available)"
    lines = code.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    trimmed = lines[:max_lines]
    trimmed.append("# ... (remaining code omitted; refer to the full seed during evolution)")
    return "\n".join(trimmed)


# NOTE: Experience highlights are currently disabled below.
## Experience highlights
# {experience_highlights}


def render_main_task(seed: Dict[str, str], experiences: List[str]) -> str:
    """Render instruction prompt for the code reasoning agent."""

    experience_highlights = _format_experience_highlights(experiences)
    templates_preview = ""
    if seed.get("question_templates"):
        template_list = seed["question_templates"]
        preview_items = template_list[:3]
        templates_preview = "\n".join(
            f"{idx + 1}. {tpl}" for idx, tpl in enumerate(preview_items)
        )
        if len(template_list) > len(preview_items):
            templates_preview += "\n... (more templates omitted)"

    generator_code_block = _trim_code_block(
        seed.get("generator_code", ""), max_lines=80
    )
    validator_code_block = _trim_code_block(
        seed.get("validator_code", ""), max_lines=80
    )
    sample_question = _trim_text(
        seed.get("sample_question", "(no sample question provided)"), max_chars=260
    )

    return f"""
You are a code reasoning agent responsible for evolving high-quality logical reasoning problems.

## Core task
Based on the original seed, design three independent components to build a deeper logical reasoning problem so the reasoning chain evolves substantially and the difficulty increases, rather than merely changing the story surface:

1. **generator**: Python function `input(difficulty)` that generates input data and slot texts
   - Return format: `(inputs, slot_texts)`
   - `inputs`: input data passed to the validator
   - `slot_texts`: text list used to fill the question template

2. **question_template**: a text template containing placeholders like `[Input Slot 1]` and `[Input Slot 2]`
   - Example: `"Given [Input Slot 1] and [Input Slot 2], find ..."`
   - Slots will be filled by the generator's slot_texts in order

3. **validator**: Python function `solution(inputs)` that computes the correct answer
   - Accepts `inputs` produced by the generator
   - Returns the standard answer; must be a pure function with no side effects so the system can auto-generate independent validators for consistency checks

### Delivery format
- Use `stop` to output JSON: `task_id`, `generator_code`, `question_template`, `validator_code`, `evolution_strategy`, `sample_summary`, `blind_review_summary`, `experience_updates`, `notes`.
- IMPORTANT: the parameter to `stop` must be a valid JSON string containing all fields above. The system parses this JSON in the final step (end stage); if the output is None or cannot be parsed, the task fails.

### Mandatory constraints
- **Reasoning evolution first**: the evolution strategy must focus on innovations/deepening in reasoning logic, constraints, or variable relationships. You may reference `mutation_hint` but do not merely reskin the story or replace terms; avoid problems that look complex but are actually easy.
- **High difficulty target**: our goal is for top-tier reasoning models to have a high error rate at difficulty 10. Problems must build long reasoning chains, interdependent constraints, or counterintuitive setups; do not intentionally simplify or lower difficulty.
- **Keep task lineage**: the core objective must follow the original seed (see the sample question and `task_path` below). You may add new constraints or change the narrative background, but do not turn the problem into an unrelated type; the final answer format must still align with the original task requirements.
- **All three components must evolve**: `generator_code` (must be rewritten), `question_template` (new statement), `validator_code` (structure can be retained but must match new rules).
- **Question self-check**: before blind review, the system calls `seed_check_question_quality()` to check readability, novelty, and difficulty. If it fails, blind review is blocked.
- **Validator pool consistency**: after saving, the system auto-generates 2 independent validators based on the generator/template spec. A majority vote becomes the standard answer. If they disagree, fix the main validator and record the experience.
- **Your code may be wrong**: both generator and validator may contain bugs. If blind review fails, first verify your generator outputs and validator computations in Python before blaming the blind-review LLM.
- **Answer self-proof**: do not rely on the seed answer. Use prints/assertions/small Python scripts to repeatedly verify solvability, correctness, and validator judgments.
- **Single-step tooling**: keep operations atomic; do not call tools inside `stop`.
- **Save before testing**: after evolution, you must call `seed_save_code(generator_code, question_template, validator_code, evolution_strategy)`. IMPORTANT: define these variables (as strings) before calling, e.g., `generator_code = "def input(difficulty): ..."`; do not use undefined variable names.
- **Blind review constraints**: `seed_submit_blind_review()` runs question self-checks and validator pool completion, then generates 5 questions with difficulties `[1, 3, 5, 5, 7]`. At least 3/5 must pass for success, and blind review must pass at least once before `stop`; otherwise submission is blocked. If blind review fails, the system returns `failed_samples_detail` to help diagnose failed samples. IMPORTANT: even if blind review passes (3/5 or 4/5), verify validator correctness in Python; low pass rate may indicate validator bugs.
- **Submission gate**: call `seed_prepare_submission()` before `stop`. If `stop` is rejected 3 times (blind review not passed), the workflow terminates.
- **Difficulty tiers**: the generator receives an integer difficulty in `[1, 10]`. In difficulty control, 1-3 should be computationally simple (but can still require hard reasoning), 4-6 medium, 7-10 difficult. Do not make problems too easy; difficulty >= 7 should be computationally challenging. Note: you do not need to shrink data size to increase difficulty; increase constraint complexity or reasoning chain length instead.
- **Difficulty control**: overly simple narratives are stale and resemble traditional LeetCode-style tasks that do not trigger deep reasoning. Try to create more novel problems that encourage multi-step reasoning and puzzle-like relations, and include necessary details and background to increase interest and complexity.
- **Avoid shortcut solving**: ensure the evolved problem cannot be solved via information shortcuts or pattern matching; avoid tasks that look data-heavy but require trivial reasoning.
- **Avoid pure algorithm tasks**: do not create fully independent traditional algorithm problems (e.g., LCS, shortest path). Problems should require multi-step reasoning and logical analysis, though Codeforces 2000+ greedy/insight tasks are acceptable as a reference.
- **Generator coverage**: when generating samples, explicitly construct multiple scenarios (solvable/unsolvable or complex/simple). For difficulty >= 5, at least half the samples must have non-trivial solutions. Forbid the answer from degenerating into a single constant (e.g., always 0).
- **Consistency**: the statement must clearly highlight the core concept. If terms like "middle part" or "lexicographic order" are used, define the mathematical/algorithmic meaning (e.g., "after sorting, take the floor(n/2)-th element") to avoid ambiguity and non-unique answers.
- **Readability requirements**: the statement should be concise and clear, reducing ambiguity. Avoid ASCII art for trees/grids; use parentheses or array notation instead. Ensure information is complete and unambiguous.
- **Code difficulty**: avoid high resource usage (excessive computation or infinite loops). Ensure reasonable runtime to prevent timeouts and system overload.

### Pre-submission checklist
1. Call `seed_save_code()` after defining generator/template/validator, then manually run `input()` and `solution()` in Python to check typical samples.
2. Run `seed_generate_sample()` at least once and manually inspect the statement and `validator_votes` to ensure answers are non-degenerate and validators agree.
3. Call `seed_check_question_quality()` and obtain `action: "proceed"`; if `revise`, adjust per feedback and pass again.
4. Trigger `seed_submit_blind_review()` and record matching details for a passing attempt; if it fails, fix your code and retry.
5. Before `stop`, re-check statement wording, difficulty labels, answer format, and validator outputs. Ensure `sample_summary` and `blind_review_summary` in the final JSON are truthful and reproducible. IMPORTANT: the `output` parameter to `stop` must be a valid JSON string (use `json.dumps()`), containing all required fields; it cannot be None or empty.

### Suggested workflow
1. Deconstruct the original problem -> map its reasoning chain -> design a new reasoning path/constraints -> implement three components.
2. **Repeated verification**: run generator and validator repeatedly in Python, print intermediate variables, and confirm correctness; if needed, sample across difficulties to check solvable/unsolvable ratios.
3. **Save code**: define variables (e.g., `generator_code = "..."`), call `seed_save_code(generator_code, question_template, validator_code)`, then run `seed_generate_sample()` to inspect statements and answers.
4. Call `seed_check_question_quality()` to confirm readability/novelty/difficulty alignment; iterate if needed.
5. Use `seed_generate_sample()` to inspect `validator_votes`; if inconsistent, fix the main validator and retry.
6. Trigger blind review (auto-completes validator pool); if it fails, use `failed_samples_detail` to locate and fix issues before retrying.
7. Record **generalizable** problem-design experience (not task-specific details) -> organize deliverables -> call `stop`.


## Seed summary (mutation_hint is for inspiration only; define a stronger reasoning goal yourself)

- question_templates preview:
{templates_preview if templates_preview else '(no templates provided; you may design your own)'}
- Reference sample question (original seed example, for understanding only; you may completely change it):
{sample_question}
  (Do not reuse the original problem or use blind review to test the original sample.)

## Original seed code excerpts (read for understanding only; do not copy directly)
```python
# generator_code
{generator_code_block}
```

```python
# validator_code
{validator_code_block}
```
""".strip()


def render_blind_prompt(
    question: str,
    experiences: List[str],
    attempt_idx: int,
) -> str:
    """Render instruction prompt for the blind reviewer (pure LLM call)."""

    experience_block = _format_experience_bullets(experiences)

    return f"""
You are a math/logic problem solver (attempt {attempt_idx + 1}). Please read the problem carefully and solve it independently.

### Problem
{question}

### Requirements
1. Understand the problem and perform clear reasoning.
2. You may call the `python` tool to write and run code to assist reasoning; print intermediate results when needed.
3. **The final answer must be strictly enclosed in `\\boxed{{final answer}}`.**
4. If unsolvable or information is insufficient, use `\\boxed{{N/A}}`.
5. **IMPORTANT**: After finishing reasoning and giving the answer, you must call the `stop` tool to end the task.
6. Avoid high resource usage (excessive computation or infinite loops); ensure reasonable runtime.

### Output example
Reasoning: ... therefore ... so ...

Final answer: \\boxed{{42}}

Then call `stop` to finish.
""".strip()


def render_validator_builder_prompt(
    generator_code: str,
    question_template: str,
    reference_validator: str,
) -> str:
    generator_snippet = _trim_code_block(generator_code, max_lines=120)
    template_text = (
        question_template.strip() or "(question template is empty; infer input structure from the generator)"
    )
    reference_note = ""
    if reference_validator:
        reference_snippet = _trim_code_block(reference_validator, max_lines=120)
        reference_note = f"""
### Current main validator (for IO contract only; do not copy line-by-line)
```python
{reference_snippet}
```
"""

    return f"""
You will receive the generator code and question template. Please write a **brand-new** Python validator function:

1. Function signature must be `def solution(inputs):`
2. Solve using only data in `inputs`; do not call the generator, random functions, or I/O.
3. Parse the `inputs` schema at the beginning. If required fields are missing or types do not match expectations, return `{{"status": "schema_error", "detail": "..."}}` and do not raise exceptions.
4. Output must be the unique standard answer. You may return a number, string, tuple, or dict, but it must match the statement.
5. Helper functions are allowed, but must ultimately return `solution(inputs)`.
6. Avoid side effects; keep pure functions so the system can generate multiple validators for voting.
7. Your implementation must be independently derived. If it disagrees with the main validator, the system will expose `validator_votes` for comparison.
8. Avoid high resource usage (excessive computation or infinite loops). Ensure reasonable runtime.

### Generator (for input structure)
```python
{generator_snippet}
```

### Question template (slots will be filled)
{template_text}

{reference_note}

Output Python code only. Do not add explanatory text or Markdown.
""".strip()


def render_question_quality_prompt(
    question: str,
    answer: str,
    difficulty: Optional[int],
) -> str:
    difficulty_note = difficulty if difficulty is not None else "(not provided)"

    return f"""
Please act as a question quality reviewer to judge whether the following problem meets our publication standards. Focus on logical correctness and solvability; tolerate flashy or long descriptions unless they clearly obstruct understanding.

### Problem
{question}

### Official answer preview
{answer}

### Targets (follow these standards)
- **Readability**: Is the problem statement largely clear and solvable as written? If the issue is only length or terminology, mark `pass`. Mark `revise` only when there is ambiguity, missing information, or contradictions. **Special note**: If the statement contains ASCII art for trees/grids, mark `revise` due to structural ambiguity.
- **Novelty**: Does it introduce a new reasoning path or constraint combination? Pass if it substantially strengthens the original task even with a similar story; revise only when it copies existing templates.
- **DifficultyAlignment**: Does difficulty match the target level ({difficulty_note})? Revise only if clearly too easy/too hard or mismatched with stated complexity. If the problem allows shortcut solving, mark `revise`.
- Avoid high resource usage (excessive computation or infinite loops); ensure reasonable runtime.

Output JSON with fields (if revisions are needed, include brief guidance in `feedback`; acceptable problems should have `action: "proceed"`):
{{
  "readability": "pass" | "revise",
  "novelty": "pass" | "revise",
  "difficulty_alignment": "pass" | "revise",
  "action": "proceed" | "revise",
  "feedback": "one-sentence suggestion"
}}

- If any criterion fails, set `action` to "revise" and provide concrete guidance in `feedback`.
- Keep the reply as valid JSON; do not include extra text or Markdown.
- **IMPORTANT**: After outputting JSON, you must call the `stop` tool to end the review.
""".strip()
