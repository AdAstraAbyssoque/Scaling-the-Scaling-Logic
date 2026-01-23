"""CLI for executing the SSLogic multi-agent pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from . import SSLogicPipeline, format_pipeline_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SSLogic synthesis pipeline.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="",
        help="Seed idea text for the baseline problem. Overrides --seed-file if provided.",
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help="Path to a JSONL file supplying seed ideas (uses the first matching record).",
    )
    parser.add_argument(
        "--seed-id",
        type=str,
        default="",
        help="If --seed-file is used, optional task_id to select a specific record.",
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        default=0,
        help="If --seed-file is used without --seed-id, 0-based record index to select.",
    )
    parser.add_argument(
        "--seed-key",
        type=str,
        default="question",
        help="JSON key to extract from the seed record (default: question).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Maximum number of validation/revision cycles (>=1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model call target (e.g., openai:gpt-4o-mini, mock). Overrides default model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to append the final answer JSON (JSONL if *.jsonl).",
    )
    return parser.parse_args()


def _load_seed_from_file(
    path: Path,
    seed_id: str = "",
    seed_index: int = 0,
    seed_key: str = "question",
) -> Optional[str]:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")

    selected = None
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if seed_id:
                if str(record.get("task_id")) != seed_id:
                    continue
                selected = record
                break
            if idx == seed_index and selected is None:
                selected = record
                if not seed_id:
                    break

    if not selected:
        return None

    value = selected.get(seed_key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def main() -> None:
    args = parse_args()
    seed_text = args.seed.strip()
    if not seed_text and args.seed_file:
        seed_text = (
            _load_seed_from_file(
                args.seed_file,
                seed_id=args.seed_id.strip(),
                seed_index=args.seed_index,
                seed_key=args.seed_key,
            )
            or ""
        )

    # Prepare model config override if specified
    main_agent = None
    if args.model and args.model.strip():
        from .agents import CKProAgentWrapper

        model_override = {"model": {"call_target": args.model.strip()}}
        main_agent = CKProAgentWrapper(
            name="ckpro-main", config_overrides=model_override
        )

    pipeline = SSLogicPipeline(
        max_iterations=args.max_iterations,
        main_agent=main_agent,
    )
    result = pipeline.run(seed_idea=seed_text or None)

    final_answer_path = result.get("final_answer_path")
    if final_answer_path and args.output:
        final_file = Path(final_answer_path)
        if final_file.exists():
            args.output.parent.mkdir(parents=True, exist_ok=True)
            data_text = final_file.read_text(encoding="utf-8").strip()
            if args.output.suffix.lower() == ".jsonl":
                if data_text:
                    try:
                        payload = json.loads(data_text)
                        line = json.dumps(payload, ensure_ascii=False)
                    except json.JSONDecodeError:
                        line = data_text.replace("\n", " ").strip()
                    with args.output.open("a", encoding="utf-8") as handle:
                        handle.write(line)
                        handle.write("\n")
            else:
                args.output.write_text(data_text, encoding="utf-8")

            final_answer_copy = args.output.parent / "final_answer.json"
            final_answer_copy.parent.mkdir(parents=True, exist_ok=True)
            final_answer_copy.write_text(data_text, encoding="utf-8")

    print(format_pipeline_result(result))


if __name__ == "__main__":
    main()
