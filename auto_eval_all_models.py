import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-run MMErroR evaluation for ETC and EPD using the packaged evaluator."
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--runner", type=str, default="run.py")
    parser.add_argument("--config", type=str, default="eval_config.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--api-base", type=str, default=os.environ.get("MMERROR_API_BASE", os.environ.get("API_BASE", "")))
    parser.add_argument("--key", type=str, default=os.environ.get("MMERROR_API_KEY", os.environ.get("API_KEY", "")))
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names. Only models present in the config will run.",
    )
    parser.add_argument("--tasks", type=str, default="etc,epd", help="Comma-separated task list, e.g. etc,epd")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--env-file", type=str, default="")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def pretty_command(cmd: list[str], key: str) -> str:
    masked = []
    skip_next_key = False
    for token in cmd:
        if skip_next_key:
            masked.append(mask_secret(key))
            skip_next_key = False
            continue
        if token == "--key":
            masked.append(token)
            skip_next_key = True
            continue
        masked.append(token)
    return " ".join(shlex.quote(item) for item in masked)


def resolve_from_bundle_root(bundle_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (bundle_root / path).resolve()


def build_command(args: argparse.Namespace, task: str) -> list[str]:
    bundle_root = Path(__file__).resolve().parent
    runner_path = resolve_from_bundle_root(bundle_root, args.runner)
    config_path = resolve_from_bundle_root(bundle_root, args.config)

    command = [
        args.python,
        str(runner_path),
        "--config",
        str(config_path),
        "--task-mode",
        task,
    ]
    if args.models:
        command.extend(["--models", args.models])
    if args.data_dir:
        data_dir = resolve_from_bundle_root(bundle_root, args.data_dir)
        command.extend(["--data-dir", str(data_dir)])
    if args.image_dir:
        image_dir = resolve_from_bundle_root(bundle_root, args.image_dir)
        command.extend(["--image-dir", str(image_dir)])
    if args.output_dir:
        output_dir = resolve_from_bundle_root(bundle_root, args.output_dir)
        command.extend(["--output-dir", str(output_dir)])
    if args.limit > 0:
        command.extend(["--limit", str(args.limit)])
    if args.env_file:
        env_file_path = resolve_from_bundle_root(bundle_root, args.env_file)
        command.extend(["--env-file", str(env_file_path)])
    return command


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    bundle_root = Path(__file__).resolve().parent
    runner_path = resolve_from_bundle_root(bundle_root, args.runner)
    if not runner_path.exists():
        print(f"[ERROR] Runner not found: {runner_path}")
        return 2

    config_path = resolve_from_bundle_root(bundle_root, args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return 2

    task_list = [task.strip().lower() for task in args.tasks.split(",") if task.strip()]
    if not task_list:
        print("[ERROR] No tasks specified.")
        return 2

    env = os.environ.copy()
    if args.api_base:
        env["MMERROR_API_BASE"] = args.api_base
    if args.key:
        env["MMERROR_API_KEY"] = args.key

    print("=" * 72)
    print("MMErroR batch evaluation")
    print(f"runner: {runner_path}")
    print(f"config: {config_path}")
    print(f"tasks: {', '.join(task_list)}")
    print(f"models: {args.models or '(from config)'}")
    print(f"data_dir: {args.data_dir or '(from config)'}")
    print(f"image_dir: {args.image_dir or '(from config)'}")
    print(f"output_dir: {args.output_dir or '(from config)'}")
    print(f"api_base: {args.api_base or '(from env/.env)'}")
    print(f"key: {mask_secret(args.key)}" if args.key else "key: (from env/.env)")
    print("=" * 72)

    results = []
    for task in task_list:
        command = build_command(args, task)
        print(f"\n[RUN] task={task}")
        print(pretty_command(command, args.key))

        if args.dry_run:
            exit_code = 0
            print("[DRY-RUN] Skipped actual execution.")
        else:
            completed = subprocess.run(command, env=env)
            exit_code = completed.returncode

        results.append((task, exit_code))
        status = "OK" if exit_code == 0 else "FAIL"
        print(f"[DONE] task={task}, exit_code={exit_code}, status={status}")

        if exit_code != 0 and args.stop_on_error:
            print("[STOP] stop-on-error enabled, aborting early.")
            break

    print("\n" + "=" * 72)
    print("Summary")
    all_ok = True
    for task, exit_code in results:
        status = "OK" if exit_code == 0 else "FAIL"
        print(f"- task={task}: {status} (exit_code={exit_code})")
        if exit_code != 0:
            all_ok = False
    print("=" * 72)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
