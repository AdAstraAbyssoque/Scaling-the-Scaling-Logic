"""
经验池自动整理模块
在每次任务结束后自动调用 LLM 精选经验条目，确保池子不超过指定数量
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List

from ..agents.model import LLM


_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text.strip()
    lines = text.splitlines()
    block_lines = []
    in_block = False
    for line in lines:
        if line.strip().startswith("```"):
            in_block = not in_block
            continue
        if in_block:
            block_lines.append(line)
    if block_lines:
        return "\n".join(block_lines).strip()
    return text.strip()


def _extract_json_array(text: str) -> str | None:
    if not text:
        return None
    cleaned = _strip_code_fences(text)
    start = cleaned.find("[")
    if start == -1:
        return None
    depth = 0
    for idx, ch in enumerate(cleaned[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return cleaned[start : idx + 1]
    return None


def _clean_experience_item(item: str) -> str:
    text = item.strip()
    text = _BULLET_PREFIX_RE.sub("", text)
    return text.strip()


def curate_experiences(
    existing: List[str],
    new_candidates: List[str],
    call_target: str,
    max_items: int = 20,
    max_token_num: int = 6000,
) -> List[str] | None:
    """
    使用 LLM 精选经验池

    Args:
        existing: 现有经验列表
        new_candidates: 新增候选经验
        call_target: LLM 调用目标
        max_items: 最大条目数
        max_token_num: LLM 上下文窗口

    Returns:
        精选后的经验列表，失败返回 None
    """
    if not existing and not new_candidates:
        return []
    if max_items <= 0:
        return []

    existing_block = "\n".join(f"- {item}" for item in existing) or "(none)"
    candidate_block = "\n".join(f"- {item}" for item in new_candidates) or "(none)"

    prompt = f"""
You are an assistant responsible for maintaining the quality of the experience repository. The goal is to help the team continuously produce high-difficulty logical problems where advanced reasoning models still make mistakes at difficulty 7-10. The experience list must be capped at no more than {max_items} items while preserving the most reusable, generalizable strategies.

Please compare \"Existing Experience\" and \"New Candidates\" and perform the following:
1. **Align with task goals**: remove items that encourage simplifying problems, lowering difficulty, weakening constraints, or prioritizing easy solvability; keep or rewrite items that strengthen multi-stage reasoning, complex constraint combinations, or counterintuitive setups.
2. **Quality filtering**: deduplicate and merge semantic overlaps; rewrite overly specific items to make them transferable and focused on increasing reasoning difficulty.
3. **Coverage**: prioritize experience involving generator/validator co-verification, improving blind-review pass rates (while keeping high difficulty), and preventing shortcut solutions.
4. **Quantity control**: limit output to {max_items} items; each item must be a single, concise, actionable sentence.
5. **Output format**: return a strictly valid JSON array, for example:
[
  "Experience 1",
  "Experience 2"
]
Do not include any extra text or code blocks, and do not prefix items with numbering or bullets.
6. If no usable items remain, return [].

### Existing experience
{existing_block}

### New candidates
{candidate_block}
""".strip()

    messages = [
        {
            "role": "system",
            "content": "You are a rigorous knowledge-base curator. Output must be a pure JSON array with no extra text or Markdown code blocks.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        llm = LLM(call_target=call_target, max_token_num=max_token_num)
        raw_output = llm(messages)

        if isinstance(raw_output, dict):
            raw_output = raw_output.get("content") or ""
        else:
            raw_output = str(raw_output)

        raw_output = raw_output.strip()
        json_payload = _extract_json_array(raw_output) or raw_output

        result = json.loads(json_payload)

        if not isinstance(result, list):
            print(f"[warn] LLM 输出格式错误（不是列表），跳过整理")
            return None

        curated = []
        seen = set()
        for item in result:
            if not isinstance(item, str):
                continue
            cleaned = _clean_experience_item(item)
            if not cleaned:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            curated.append(cleaned)
            if max_items and len(curated) >= max_items:
                break

        return curated

    except Exception as exc:
        print(f"[warn] 经验整理失败: {exc}，保留原经验池")
        return None


def backup_experience_file(path: Path, keep_recent: int = 5) -> None:
    """
    备份经验文件并清理旧备份

    Args:
        path: 经验文件路径
        keep_recent: 保留最近的备份数量（默认5个）
    """
    if not path.exists():
        return

    # 创建新备份
    backup_path = path.with_suffix(path.suffix + f".bak-{datetime.now():%Y%m%d-%H%M%S}")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[info] 已备份经验文件: {backup_path.name}")

    # 清理旧备份
    try:
        backup_pattern = f"{path.name}.bak-*"
        all_backups = sorted(
            path.parent.glob(backup_pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,  # 最新的在前
        )

        if len(all_backups) > keep_recent:
            old_backups = all_backups[keep_recent:]
            for old_backup in old_backups:
                old_backup.unlink()
            print(f"[info] 已清理 {len(old_backups)} 个旧备份文件")
    except Exception as exc:
        print(f"[warn] 清理旧备份失败: {exc}")


def auto_curate_experience_pool(
    experience_manager,
    new_experiences: List[str],
    call_target: str,
    max_items: int = 20,
) -> bool:
    """
    自动整理经验池（在任务结束时调用）

    Args:
        experience_manager: ExperienceManager 实例
        new_experiences: 本次新增的经验
        call_target: LLM 调用目标
        max_items: 最大条目数

    Returns:
        是否成功整理
    """
    existing = experience_manager.all()
    total_count = len(existing)

    # 如果条目数不超过阈值，跳过整理
    if total_count <= max_items and not new_experiences:
        print(f"[info] 经验池条目数 ({total_count}) 未超过阈值 ({max_items})，无需整理")
        return True

    print(
        f"[info] 开始整理经验池: 现有 {total_count} 条，新增 {len(new_experiences)} 条"
    )

    curated = curate_experiences(
        existing=existing,
        new_candidates=new_experiences,
        call_target=call_target,
        max_items=max_items,
    )

    if curated is None:
        print("[warn] 经验整理失败，保留原经验池")
        return False

    if curated == existing:
        print("[info] 经验池无变化，跳过写入")
        return True

    # 备份并写入
    try:
        backup_experience_file(experience_manager.path)
        experience_manager._cached = curated
        experience_manager._persist()
        print(f"[info] 经验池已更新: {len(existing)} → {len(curated)} 条")
        return True
    except Exception as exc:
        print(f"[error] 写入经验池失败: {exc}")
        return False
