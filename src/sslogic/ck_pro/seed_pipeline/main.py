from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import SeedGenerationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed generation pipeline built on CK-Pro."
    )
    parser.add_argument(
        "--task-id",
        dest="task_id",
        type=str,
        default=None,
        help="指定要处理的 seed task_id（与 --index 互斥）。",
    )
    parser.add_argument(
        "--index",
        dest="index",
        type=int,
        default=None,
        help="按索引选择 seed（从 0 开始，与 --task-id 互斥）。",
    )
    parser.add_argument(
        "--seed-path",
        dest="seed_path",
        type=Path,
        default=None,
        help="自定义 Seed.jsonl 路径，默认读取仓库 data/Seed.jsonl。",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("artifacts/seed_pipeline"),
        help="运行结果输出目录。",
    )
    parser.add_argument(
        "--phoenix-project",
        dest="phoenix_project",
        type=str,
        default="seed-generation",
        help="Phoenix 追踪使用的项目名称。",
    )
    parser.add_argument(
        "--model-call-target",
        dest="model_call_target",
        type=str,
        default=None,
        help="自定义 LLM call_target（默认使用 DeepSeek），例如 api:gpt-4o。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="如果任务已完成（存在输出目录且包含 final_output.json），则跳过生成。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = SeedGenerationPipeline(
        seed_path=args.seed_path,
        output_root=args.output_dir,
        phoenix_project=args.phoenix_project,
        model_call_target=args.model_call_target,
    )

    result = pipeline.run(task_id=args.task_id, index=args.index, resume=args.resume)

    print("==== 运行完成 ====")
    print(f"task_id: {result.task_id}")
    print(f"盲评是否通过: {'是' if result.success else '否'}")
    print(f"session_dir: {result.session_dir}")
    print(f"experience_added: {result.experience_added}")
    print(f"model_call_target: {pipeline.model_call_target}")


if __name__ == "__main__":
    main()
