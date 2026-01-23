import argparse
import json
import os
from typing import Any, Dict, List, Optional

from transformers import AutoConfig, AutoTokenizer

from utils import construct_prompt, load_jsonl, save_jsonl
from parser import parse_ground_truth, parse_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="all", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--output_dir", default="./data_filtered", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--prompt_type", default="qwen3-base-training", type=str)
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--adapt_few_shot", action="store_true")
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B-Base", type=str)
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str)
    parser.add_argument("--max_model_len", default=None, type=int)
    parser.add_argument("--save_dropped", action="store_true")
    return parser.parse_args()


def resolve_data_names(data_dir: str, data_names: str, split: str) -> List[str]:
    if data_names in {"all", "*"}:
        names = []
        for entry in os.listdir(data_dir):
            path = os.path.join(data_dir, entry)
            if not os.path.isdir(path):
                continue
            if os.path.exists(os.path.join(path, f"{split}.jsonl")):
                names.append(entry)
        return sorted(names)
    return [name.strip() for name in data_names.split(",") if name.strip()]


def infer_max_len(tokenizer, config: Optional[Any]) -> Optional[int]:
    candidates = []
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max_length, int) and model_max_length < 10**9:
        candidates.append(model_max_length)
    if config is not None:
        for key in [
            "max_position_embeddings",
            "max_seq_len",
            "max_sequence_length",
            "n_positions",
        ]:
            value = getattr(config, key, None)
            if isinstance(value, int):
                candidates.append(value)
    if not candidates:
        return None
    return min(candidates)


def token_length(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def main() -> None:
    args = parse_args()
    tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True, use_fast=True
    )
    config = None
    try:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    except Exception:
        config = None

    max_model_len = args.max_model_len or infer_max_len(tokenizer, config)
    if max_model_len is None:
        raise ValueError(
            "Unable to infer max_model_len. Please pass --max_model_len explicitly."
        )

    data_names = resolve_data_names(args.data_dir, args.data_names, args.split)
    if not data_names:
        raise ValueError("No datasets found to filter.")

    report: Dict[str, Dict[str, Any]] = {}

    for data_name in data_names:
        data_path = os.path.join(args.data_dir, data_name, f"{args.split}.jsonl")
        if not os.path.exists(data_path):
            print(f"Skip missing dataset: {data_path}")
            continue

        examples = list(load_jsonl(data_path))
        if not examples:
            print(f"Skip empty dataset: {data_path}")
            continue

        if "idx" not in examples[0]:
            examples = [{"idx": i, **example} for i, example in enumerate(examples)]

        kept: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        max_prompt_len = 0
        max_idx = None

        for example in examples:
            example_for_prompt = dict(example)
            question = parse_question(example_for_prompt, data_name)
            if not question:
                dropped.append({"idx": example.get("idx"), "reason": "empty_question"})
                continue

            example_for_prompt["question"] = question
            _, gt_ans = parse_ground_truth(example_for_prompt, data_name)
            example_for_prompt["gt_ans"] = gt_ans

            prompt = construct_prompt(example_for_prompt, data_name, args)
            if args.apply_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt.strip()}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

            prompt_len = token_length(tokenizer, prompt)
            if prompt_len > max_prompt_len:
                max_prompt_len = prompt_len
                max_idx = example.get("idx")

            if prompt_len <= max_model_len:
                kept.append(example)
            else:
                dropped.append({"idx": example.get("idx"), "prompt_len": prompt_len})

        out_path = os.path.join(args.output_dir, data_name, f"{args.split}.jsonl")
        save_jsonl(kept, out_path)
        if args.save_dropped:
            dropped_path = os.path.join(
                args.output_dir, data_name, f"{args.split}.dropped.jsonl"
            )
            save_jsonl(dropped, dropped_path)

        report[data_name] = {
            "total": len(examples),
            "kept": len(kept),
            "dropped": len(dropped),
            "max_prompt_len": max_prompt_len,
            "max_prompt_idx": max_idx,
        }
        print(
            f"{data_name}: total={len(examples)} kept={len(kept)} "
            f"dropped={len(dropped)} max_prompt_len={max_prompt_len}"
        )

    report_path = os.path.join(args.output_dir, "length_filter_report.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
