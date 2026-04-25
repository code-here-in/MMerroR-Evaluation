#!/usr/bin/env python3
"""
Run QA correctness evaluation for all models and both sampled subsets (correct/incorrect)
in one go. It expects the ID lists produced by scripts/sample_from_label_for_qa.py:
  samples_for_qa/<model>/correct_ids.txt
  samples_for_qa/<model>/incorrect_ids.txt

For each model, it will invoke legacy/test_answer_correctness.py twice
(once per subset) with the provided API/base/key and judge options.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        default="meta-llama/llama-4-maverick,o4-mini,gpt-5.2,qwen3-vl-32b-instruct",
        help="Comma-separated API model names to evaluate.",
    )
    parser.add_argument(
        "--id_root",
        type=str,
        default="../samples_for_qa",
        help="Root directory containing <model>/correct_ids.txt and incorrect_ids.txt.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/jsons",
        help="Path to ground-truth JSONs.",
    )
    parser.add_argument(
        "--qa_script",
        type=str,
        default="legacy/test_answer_correctness.py",
        help="Path to QA evaluation script.",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="../result_answer_sampled",
        help="Where QA results will be written.",
    )
    parser.add_argument("--api_base", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--key", type=str, default="")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge_mode", choices=["llm", "string"], default="llm")
    parser.add_argument("--llm_on_incorrect_only", action="store_true", help="First string-match, LLM only when incorrect.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--use_max_tokens", action="store_true")
    return parser.parse_args()


def gather_subsets(id_root: Path, model: str) -> List[Tuple[str, Path]]:
    """
    Return list of (tag, path) pairs for correct/incorrect id files.
    Tags will be used as run_tag values.
    """
    safe_model = model.replace("/", "--")
    model_dir = id_root / safe_model
    subsets = []
    for tag in ("correct", "incorrect"):
        f = model_dir / f"{tag}_ids.txt"
        if f.exists():
            subsets.append((tag, f))
    return subsets


def read_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len([ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()])


def build_cmd(
    args: argparse.Namespace,
    model: str,
    ids_file: Path,
    tag: str,
) -> List[str]:
    cmd = [
        sys.executable,
        args.qa_script,
        "--data_dir",
        args.data_dir,
        "--question_ids_file",
        str(ids_file),
        "--models",
        model,
        "--api_base",
        args.api_base,
        "--key",
        args.key,
        "--judge_mode",
        args.judge_mode,
        "--result_root",
        args.result_root,
        "--run_tag",
        tag,
        "--workers",
        str(args.workers),
        "--timeout",
        str(args.timeout),
        "--max_retries",
        str(args.max_retries),
        "--temperature",
        str(args.temperature),
        "--max_tokens",
        str(args.max_tokens),
    ]
    if args.use_max_tokens:
        cmd.append("--use_max_tokens")
    if args.judge_mode == "llm":
        cmd.extend(["--judge_model", args.judge_model])
        if args.llm_on_incorrect_only:
            cmd.append("--llm_on_incorrect_only")
    return cmd


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    qa_script = repo_root / args.qa_script if not Path(args.qa_script).is_absolute() else Path(args.qa_script)
    id_root = repo_root / args.id_root if not Path(args.id_root).is_absolute() else Path(args.id_root)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not qa_script.exists():
        raise SystemExit(f"QA script not found: {qa_script}")
    if not id_root.exists():
        raise SystemExit(f"ID root not found: {id_root}")

    for model in models:
        subsets = gather_subsets(id_root, model)
        if not subsets:
            print(f"[SKIP] {model}: no correct/incorrect id files under {id_root}")
            continue
        for tag, ids_file in subsets:
            count = read_count(ids_file)
            if count == 0:
                print(f"[SKIP] {model} {tag}: empty ids file {ids_file}")
                continue
            tag_with_count = f"{tag}{count}"
            cmd = build_cmd(args, model, ids_file, tag_with_count)
            print(f"[RUN] {model} {tag} ({count} ids): {' '.join(cmd)}")
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
