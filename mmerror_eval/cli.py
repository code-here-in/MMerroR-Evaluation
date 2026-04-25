from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .config import load_config
from .labels import normalize_task_mode
from .runner import run_project


def build_parser(default_config_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MMErroR evaluations from a config file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help=f"Path to YAML config file. Default: {default_config_path}",
    )
    parser.add_argument("--env-file", type=Path, default=None, help="Optional .env file to load before evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Override dataset.limit.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override dataset.json_dir.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Override dataset.image_dir.")
    parser.add_argument("--task-mode", type=str, default=None, help="Override evaluation.task_mode with etc/epd/4/5.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override evaluation.output_dir.")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to enable. Only matching config entries will run.",
    )
    return parser


def _apply_overrides(config: any, args: argparse.Namespace) -> None:
    if args.limit is not None:
        config.dataset.limit = args.limit
    if args.data_dir is not None:
        config.dataset.json_dir = args.data_dir.resolve()
    if args.image_dir is not None:
        config.dataset.image_dir = args.image_dir.resolve()
    if args.task_mode is not None:
        config.evaluation.task_mode = normalize_task_mode(args.task_mode)
    if args.output_dir is not None:
        config.evaluation.output_dir = args.output_dir.resolve()
    if args.models:
        chosen = {item.strip() for item in args.models.split(",") if item.strip()}
        for model in config.models:
            model.enabled = model.name in chosen


def main(argv: Optional[List[str]] = None) -> int:
    default_config_path = Path(__file__).resolve().parents[1] / "eval_config.yaml"
    parser = build_parser(default_config_path)
    args = parser.parse_args(argv)

    config = load_config(args.config, env_file=args.env_file)
    _apply_overrides(config, args)
    run_project(config)
    return 0
