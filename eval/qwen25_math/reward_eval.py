#!/usr/bin/env python3
import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval_service import RewardService


def normalize_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def select_prompt(sample):
    for key in ["question", "problem", "input", "Question"]:
        if key in sample and sample[key]:
            return str(sample[key])
    return ""


def select_response(code_list, pred_list, idx):
    raw = ""
    if idx < len(code_list):
        raw = code_list[idx]
    raw = "" if raw is None else str(raw)

    if "</think>" in raw or "</reasoning>" in raw:
        selected = raw
    else:
        pred = ""
        if idx < len(pred_list):
            pred = pred_list[idx]
        pred = "" if pred is None else str(pred)
        selected = f"</reasoning>\n{pred}" if pred else raw

    is_empty = not selected or not selected.strip()
    return selected, is_empty


def extract_prompt_type_from_filename(path: Path) -> str:
    name = path.stem
    if not name.startswith("test_"):
        return ""
    prefix = name[len("test_") :]
    if "_seed" in prefix:
        prefix = prefix.split("_seed", 1)[0]
    if "_" in prefix:
        maybe_model, maybe_num = prefix.rsplit("_", 1)
        if maybe_num.lstrip("-").isdigit():
            return maybe_model
    return prefix


def parse_model_dataset(path: Path):
    parts = path.parts
    dataset = ""
    model = ""

    if "math_eval" in parts:
        idx = parts.index("math_eval")
        if idx + 1 < len(parts):
            dataset = parts[idx + 1]

    if "ckpts" in parts:
        idx = parts.index("ckpts")
        if idx + 1 < len(parts):
            model = parts[idx + 1]
    elif "model" in parts:
        idx = parts.index("model")
        if idx + 1 < len(parts):
            model = parts[idx + 1]
    elif "code" in parts:
        idx = parts.index("code")
        if idx + 1 < len(parts):
            model = parts[idx + 1]
    else:
        model = extract_prompt_type_from_filename(path)

    if not model:
        model = extract_prompt_type_from_filename(path)
    if not dataset:
        dataset = path.parent.name
    return model or "unknown", dataset or "unknown"


def load_cache(cache_path):
    cache = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = record.get("idx")
            if idx is None:
                continue
            cache[idx] = record
    return cache


def score_response(service, prompt, response, ref_answer, exp_id, router_id, extra):
    if not response or not response.strip():
        return 0.0, False, "empty"

    result = service.get_reward(
        prompt=prompt,
        response=response,
        ref_answer=ref_answer,
        exp_id=exp_id,
        router_id=router_id,
        extra_param=extra,
    )
    if not result:
        return 0.0, False, "failed"

    data = result.get("data", {})
    reward_score = data.get("score", 0.0)
    correct = data.get("correct", False)
    return reward_score, bool(correct), "ok"


def pass_at_k(n, c, k):
    if k > n:
        return None
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def compute_metrics(entries, timeout_samples, empty_samples, time_use):
    score_mat = [list(entry["correct"]) for entry in entries]
    if not score_mat:
        return {
            "num_samples": 0,
            "num_scores": 0,
            "timeout_samples": timeout_samples,
            "empty_samples": empty_samples,
            "acc": 0.0,
            "passk_excluded_samples": 0,
            "pass@1": None,
            "pass@2": None,
            "pass@4": None,
            "pass@8": None,
            "pass@16": None,
            "pass@32": None,
            "pass@64": None,
            "time_use_in_second": time_use,
            "time_use_in_minite": f"{int(time_use // 60)}:{int(time_use % 60):02d}",
        }

    max_len = max(len(s) for s in score_mat)
    for s in score_mat:
        if len(s) < max_len:
            s.extend([s[-1]] * (max_len - len(s)))

    col_means = np.mean(np.array(score_mat, dtype=float), axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result = {
        "num_samples": len(entries),
        "num_scores": sum(len(entry["correct"]) for entry in entries),
        "timeout_samples": timeout_samples,
        "empty_samples": empty_samples,
        "acc": mean_score[0],
    }

    if any(entry.get("type") for entry in entries):
        type_scores = defaultdict(list)
        for entry in entries:
            entry_type = entry.get("type")
            if not entry_type:
                continue
            type_scores[entry_type].append(entry["correct"][-1])
        type_scores = {
            k: round(sum(v) / len(v) * 100, 1) for k, v in type_scores.items()
        }
        type_scores = dict(sorted(type_scores.items(), key=lambda x: x[0]))
        result["type_acc"] = type_scores

    passk_entries = [entry for entry in entries if not entry.get("skip_passk")]
    result["passk_excluded_samples"] = len(entries) - len(passk_entries)

    for k in [1, 2, 4, 8, 16, 32, 64]:
        per_sample = []
        invalid = False
        for entry in passk_entries:
            n = len(entry["correct"])
            c = sum(entry["correct"])
            val = pass_at_k(n, c, k)
            if val is None:
                invalid = True
                break
            per_sample.append(val)
        if invalid or not per_sample:
            result[f"pass@{k}"] = None
        else:
            result[f"pass@{k}"] = float(
                round(sum(per_sample) / len(per_sample) * 100, 1)
            )

    result["time_use_in_second"] = time_use
    result["time_use_in_minite"] = f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    return result


def discover_jsonl_files(input_path):
    if input_path.is_file():
        if input_path.suffix == ".jsonl":
            return [input_path]
        if input_path.name.endswith("_metrics.json"):
            return [input_path]
        return []

    jsonl_files = [
        p
        for p in input_path.rglob("*.jsonl")
        if not p.name.endswith("_reward_system.jsonl")
    ]
    if jsonl_files:
        return jsonl_files

    metrics_files = list(input_path.rglob("*_metrics.json"))
    return metrics_files


def resolve_jsonl_from_metrics(metrics_path):
    parent = metrics_path.parent
    candidates = list(parent.glob("*.jsonl"))
    if not candidates:
        return None
    base = metrics_path.name
    best = None
    best_len = -1
    for candidate in candidates:
        stem = candidate.name[:-6]
        if base.startswith(stem + "_") and base.endswith("_metrics.json"):
            if len(stem) > best_len:
                best = candidate
                best_len = len(stem)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="evaluation/outputs",
        help="Directory or file (jsonl or *_metrics.json).",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--url",
        type=str,
        default="http://api-reward-platform-api-reward-platform-354ba565.turbotke.production.polaris:10001/reward/RewardService",
    )
    parser.add_argument(
        "--token", type=str, default="890f24d0-d398-4bf7-8a1b-5b9509766b1a"
    )
    parser.add_argument("--router-id", type=str, default="Logic-Traditional-11")
    parser.add_argument("--exp-id", type=str, default="reward_eval")
    parser.add_argument("--suffix", type=str, default="metric_reward_system")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    targets = discover_jsonl_files(input_path)
    if not targets:
        raise SystemExit(f"No jsonl or *_metrics.json found under {input_path}")

    if not args.url and not args.mock:
        raise SystemExit("Missing reward service URL (set --url or REWARD_SERVICE_URL)")
    if not args.token and not args.mock:
        raise SystemExit(
            "Missing reward service token (set --token or REWARD_SERVICE_TOKEN)"
        )

    service = RewardService(
        url=args.url, token=args.token, mock_mode=args.mock, verbose=True
    )

    for target in targets:
        start_time = time.time()
        jsonl_path = target
        if jsonl_path.suffix != ".jsonl":
            jsonl_path = resolve_jsonl_from_metrics(target)
            if not jsonl_path:
                print(f"Skip {target}: no matching jsonl found.")
                continue

        metrics_path = jsonl_path.with_suffix("")
        metrics_path = metrics_path.with_name(
            metrics_path.name + f"_{args.suffix}.json"
        )
        cache_path = jsonl_path.with_suffix("")
        cache_path = cache_path.with_name(cache_path.name + "_reward_system.jsonl")

        if args.overwrite and cache_path.exists():
            cache_path.unlink()
        cache = {} if args.overwrite else load_cache(cache_path)

        entries = []
        entry_map = {}
        pending = []
        timeout_samples = 0
        empty_samples = 0

        with jsonl_path.open("r", encoding="utf-8") as f:
            first_sample = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                first_sample = json.loads(line)
                break
            if not first_sample or (
                "code" not in first_sample and "pred" not in first_sample
            ):
                print(f"Skip {jsonl_path}: no code/pred fields.")
                continue
            f.seek(0)

            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                idx = sample.get("idx")
                if idx is None:
                    idx = len(entries)

                code_list = normalize_list(sample.get("code"))
                pred_list = normalize_list(sample.get("pred"))
                if not code_list and pred_list:
                    code_list = [""] * len(pred_list)
                if not pred_list and code_list:
                    pred_list = [""] * len(code_list)

                if not code_list:
                    code_list = [""]
                    pred_list = [""]

                prompt = select_prompt(sample)
                ref_answer = sample.get("gt", sample.get("answer", ""))
                ref_answer = "" if ref_answer is None else str(ref_answer)

                n_responses = len(code_list)
                _, last_empty = select_response(code_list, pred_list, n_responses - 1)
                if last_empty:
                    empty_samples += 1

                if not ref_answer.strip():
                    correct = [False] * n_responses
                    reward_scores = [0.0] * n_responses
                    entry = {
                        "idx": idx,
                        "type": sample.get("type"),
                        "correct": correct,
                        "reward_scores": reward_scores,
                        "updated": True,
                        "skip_passk": True,
                    }
                    entries.append(entry)
                    entry_map[idx] = entry
                    continue

                cache_entry = cache.get(idx)
                cached_correct = []
                cached_reward = []
                if cache_entry:
                    cached_correct = normalize_list(cache_entry.get("reward_correct"))
                    cached_reward = normalize_list(cache_entry.get("reward_scores"))

                correct = [None] * n_responses
                reward_scores = [None] * n_responses
                for i in range(min(len(cached_correct), n_responses)):
                    correct[i] = cached_correct[i]
                for i in range(min(len(cached_reward), n_responses)):
                    reward_scores[i] = cached_reward[i]

                updated = False
                for i in range(n_responses):
                    if correct[i] is not None:
                        continue
                    response, is_empty = select_response(code_list, pred_list, i)
                    if is_empty:
                        correct[i] = False
                        reward_scores[i] = 0.0
                        updated = True
                        continue
                    pending.append((idx, i, prompt, response, ref_answer))

                entry = {
                    "idx": idx,
                    "type": sample.get("type"),
                    "correct": correct,
                    "reward_scores": reward_scores,
                    "updated": updated,
                    "skip_passk": False,
                }
                entries.append(entry)
                entry_map[idx] = entry

        if pending:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {}
                for idx, resp_idx, prompt, response, ref_answer in pending:
                    extra = {}
                    future = executor.submit(
                        score_response,
                        service,
                        prompt,
                        response,
                        ref_answer,
                        args.exp_id,
                        args.router_id,
                        extra,
                    )
                    futures[future] = (idx, resp_idx)

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc=jsonl_path.name
                ):
                    idx, resp_idx = futures[future]
                    reward_score, correct_flag, status = future.result()
                    if status == "failed":
                        timeout_samples += 1
                    entry = entry_map.get(idx)
                    if not entry:
                        continue
                    entry["correct"][resp_idx] = correct_flag
                    entry["reward_scores"][resp_idx] = reward_score
                    entry["updated"] = True

        # Fill any remaining Nones (failed or skipped) as 0
        for entry in entries:
            for i in range(len(entry["correct"])):
                if entry["correct"][i] is None:
                    entry["correct"][i] = False
                if entry["reward_scores"][i] is None:
                    entry["reward_scores"][i] = 0.0

        # Append updates to cache
        if entries:
            with cache_path.open("a", encoding="utf-8") as f:
                for entry in entries:
                    if not entry["updated"]:
                        continue
                    f.write(
                        json.dumps(
                            {
                                "idx": entry["idx"],
                                "reward_scores": entry["reward_scores"],
                                "reward_correct": entry["correct"],
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

        time_use = time.time() - start_time
        metrics = compute_metrics(
            entries, timeout_samples, empty_samples, time_use=time_use
        )
        metrics["scorer"] = "reward_service"
        metrics["router_id"] = args.router_id

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=True)

        model_label, dataset_label = parse_model_dataset(jsonl_path)
        print(
            f"[reward_eval] model={model_label} dataset={dataset_label} acc={metrics.get('acc')} "
            f"timeout_samples={metrics.get('timeout_samples')}"
        )
        print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
