from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except Exception:
    class tqdm:  # type: ignore[override]
        def __init__(self, total: int = 0, desc: str = "") -> None:
            self.total = total
            self.desc = desc

        def update(self, count: int = 1) -> None:
            _ = count

        def write(self, message: str) -> None:
            print(message)

        def close(self) -> None:
            return

from .config import AppConfig, ModelConfig
from .dataset import DatasetRecord, load_dataset
from .labels import (
    PRESENCE_ERROR_LETTER,
    PRESENCE_NO_ERROR_LETTER,
    LabelSpec,
    build_label_spec,
    normalize_label,
    normalize_presence_decision,
    presence_letter_for_label,
)
from .prompts import load_prompt_template, render_presence_prompt, render_prompt
from .providers import ProviderContext, create_provider, merge_provider_settings
from .reporting import build_summary_rows, compute_model_metrics, render_summary_markdown
from .utils import dump_json, slugify, utc_now_iso, write_text


class RateLimiter:
    def __init__(self, interval_sec: float) -> None:
        self.interval_sec = interval_sec
        self._lock = Lock()
        self._last_request_ts = 0.0

    def wait(self) -> None:
        if self.interval_sec <= 0:
            return
        with self._lock:
            now = time.monotonic()
            remaining = self.interval_sec - (now - self._last_request_ts)
            if remaining > 0:
                time.sleep(remaining)
                now = time.monotonic()
            self._last_request_ts = now


def _task_mode_dir_name(task_mode: str) -> str:
    normalized = str(task_mode).strip().lower()
    if normalized == "etc":
        return "ETC"
    if normalized == "epd":
        return "EPD"
    return normalized.upper()


def _build_message(record: DatasetRecord, prompt: str) -> List[Any]:
    if record.image_path and record.image_path.exists():
        return [str(record.image_path), prompt]
    return [prompt]


def _evaluate_record(
    provider: Any,
    rate_limiter: RateLimiter,
    record: DatasetRecord,
    prompt_templates: Dict[str, str],
    label_spec: LabelSpec,
    max_retries: int,
) -> Dict[str, Any]:
    if label_spec.task_mode == "epd":
        return _evaluate_record_epd(
            provider=provider,
            rate_limiter=rate_limiter,
            record=record,
            prompt_templates=prompt_templates,
            label_spec=label_spec,
            max_retries=max_retries,
        )

    return _evaluate_record_etc(
        provider=provider,
        rate_limiter=rate_limiter,
        record=record,
        prompt_template=prompt_templates["type"],
        label_spec=label_spec,
        max_retries=max_retries,
    )


def _evaluate_record_etc(
    provider: Any,
    rate_limiter: RateLimiter,
    record: DatasetRecord,
    prompt_template: str,
    label_spec: LabelSpec,
    max_retries: int,
) -> Dict[str, Any]:
    gt_label = normalize_label(record.label, label_spec=label_spec, is_prediction=False)
    if not gt_label:
        return {
            "index": record.index,
            "question_id": record.question_id,
            "status": "error",
            "error": f"Unsupported ground-truth label: {record.label}",
        }

    gold_letter = label_spec.reverse_pred_map.get(gt_label)
    prompt = render_prompt(prompt_template, record.question, record.error_reason, label_spec)
    message = _build_message(record, prompt)
    attempts = max(1, max_retries + 1)

    last_error: Optional[str] = None
    response: Optional[str] = None
    pred_label: Optional[str] = None

    for attempt in range(1, attempts + 1):
        try:
            rate_limiter.wait()
            response = provider.generate(message, context=ProviderContext(gold_letter=gold_letter))
            pred_label = normalize_label(response, label_spec=label_spec, is_prediction=True)
            if response and "Failed to obtain answer via API" not in response and pred_label:
                return {
                    "index": record.index,
                    "question_id": record.question_id,
                    "domain": record.domain,
                    "subdomain": record.subdomain,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "correct": pred_label == gt_label,
                    "raw_response": response,
                    "status": "ok",
                    "attempts": attempt,
                }
            last_error = f"Invalid response: {response!r}; parsed={pred_label!r}"
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
        if attempt < attempts:
            time.sleep(2)

    return {
        "index": record.index,
        "question_id": record.question_id,
        "domain": record.domain,
        "subdomain": record.subdomain,
        "gt_label": gt_label,
        "status": "error",
        "error": last_error or "Unknown failure",
        "raw_response": response,
        "attempts": attempts,
    }


def _evaluate_record_epd(
    provider: Any,
    rate_limiter: RateLimiter,
    record: DatasetRecord,
    prompt_templates: Dict[str, str],
    label_spec: LabelSpec,
    max_retries: int,
) -> Dict[str, Any]:
    gt_label = normalize_label(record.label, label_spec=label_spec, is_prediction=False)
    if not gt_label:
        return {
            "index": record.index,
            "question_id": record.question_id,
            "status": "error",
            "error": f"Unsupported ground-truth label: {record.label}",
        }

    gold_letter = label_spec.reverse_pred_map.get(gt_label)
    expected_presence = presence_letter_for_label(gt_label)
    type_label_spec = build_label_spec("etc")
    attempts = max(1, max_retries + 1)

    presence_prompt = render_presence_prompt(
        prompt_templates["presence"],
        record.question,
        record.error_reason,
    )
    presence_message = _build_message(record, presence_prompt)

    type_prompt = render_prompt(
        prompt_templates["type"],
        record.question,
        record.error_reason,
        type_label_spec,
        task_mode="epd-type",
        task_instruction=(
            "Stage 2 of EPD: an error has already been determined to be present. "
            "Classify the error type from A to D."
        ),
    )
    type_message = _build_message(record, type_prompt)

    last_error: Optional[str] = None
    presence_response: Optional[str] = None
    presence_decision: Optional[str] = None
    type_response: Optional[str] = None
    pred_label: Optional[str] = None

    for attempt in range(1, attempts + 1):
        try:
            rate_limiter.wait()
            presence_response = provider.generate(
                presence_message,
                context=ProviderContext(gold_letter=gold_letter, stage="presence"),
            )
            presence_decision = normalize_presence_decision(presence_response)
            if not presence_decision:
                last_error = f"Invalid presence response: {presence_response!r}"
                if attempt < attempts:
                    time.sleep(2)
                continue

            if presence_decision == PRESENCE_NO_ERROR_LETTER:
                pred_label = "E_NoError"
                return {
                    "index": record.index,
                    "question_id": record.question_id,
                    "domain": record.domain,
                    "subdomain": record.subdomain,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "correct": pred_label == gt_label,
                    "raw_response": {
                        "presence": presence_response,
                        "type": None,
                    },
                    "epd_presence_decision": presence_decision,
                    "epd_expected_presence": expected_presence,
                    "status": "ok",
                    "attempts": attempt,
                }

            if presence_decision != PRESENCE_ERROR_LETTER:
                last_error = f"Unsupported presence decision: {presence_decision!r}"
                if attempt < attempts:
                    time.sleep(2)
                continue

            rate_limiter.wait()
            type_response = provider.generate(
                type_message,
                context=ProviderContext(gold_letter=gold_letter, stage="type"),
            )
            pred_label = normalize_label(type_response, label_spec=type_label_spec, is_prediction=True)
            if pred_label:
                return {
                    "index": record.index,
                    "question_id": record.question_id,
                    "domain": record.domain,
                    "subdomain": record.subdomain,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "correct": pred_label == gt_label,
                    "raw_response": {
                        "presence": presence_response,
                        "type": type_response,
                    },
                    "epd_presence_decision": presence_decision,
                    "epd_expected_presence": expected_presence,
                    "status": "ok",
                    "attempts": attempt,
                }

            last_error = (
                f"Invalid type response after presence={presence_response!r}: "
                f"type_response={type_response!r}; parsed={pred_label!r}"
            )
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
        if attempt < attempts:
            time.sleep(2)

    return {
        "index": record.index,
        "question_id": record.question_id,
        "domain": record.domain,
        "subdomain": record.subdomain,
        "gt_label": gt_label,
        "status": "error",
        "error": last_error or "Unknown failure",
        "raw_response": {
            "presence": presence_response,
            "type": type_response,
        },
        "epd_presence_decision": presence_decision,
        "epd_expected_presence": expected_presence,
        "attempts": attempts,
    }


def _load_resume(output_file: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if not output_file.exists():
        return [], {}

    content = json.loads(output_file.read_text(encoding="utf-8"))
    results = content.get("results", [])
    if not isinstance(results, list):
        return [], {}

    successful_map: Dict[str, Dict[str, Any]] = {}
    for result in results:
        if result.get("status") == "ok" and result.get("question_id"):
            successful_map[result["question_id"]] = result
    return results, successful_map


def _write_model_report(
    output_file: Path,
    config: AppConfig,
    model: ModelConfig,
    results: Sequence[Dict[str, Any]],
    dataset_total: int,
    started_at: str,
) -> Dict[str, Any]:
    ordered_results = sorted(results, key=lambda item: item.get("index", -1))
    metrics = compute_model_metrics(ordered_results, dataset_total)
    report = {
        "project": config.project.name,
        "description": config.project.description,
        "task_mode": config.evaluation.task_mode,
        "model": model.name,
        "config_file": str(config.config_path),
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        **metrics,
        "results": ordered_results,
    }
    dump_json(report, output_file)
    return report


def run_model(
    config: AppConfig,
    model: ModelConfig,
    records: Sequence[DatasetRecord],
    label_spec: LabelSpec,
    prompt_templates: Dict[str, str],
) -> Dict[str, Any]:
    settings = merge_provider_settings(model, config.provider_defaults, config.evaluation)
    provider = create_provider(settings)
    rate_limiter = RateLimiter(config.evaluation.request_interval_sec)

    safe_model_name = slugify(model.name)
    task_mode_dir = _task_mode_dir_name(config.evaluation.task_mode)
    model_dir = config.evaluation.output_dir / task_mode_dir / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    output_file = model_dir / "latest.json"
    started_at = utc_now_iso()
    existing_results: List[Dict[str, Any]] = []
    completed_map: Dict[str, Dict[str, Any]] = {}
    if config.evaluation.resume:
        existing_results, completed_map = _load_resume(output_file)

    pending_records = [record for record in records if record.question_id not in completed_map]
    results = list(existing_results)

    progress = tqdm(total=len(pending_records), desc=f"Evaluating {model.name}")

    if pending_records:
        with ThreadPoolExecutor(max_workers=max(1, config.evaluation.concurrency)) as executor:
            futures = {
                executor.submit(
                    _evaluate_record,
                    provider,
                    rate_limiter,
                    record,
                    prompt_templates,
                    label_spec,
                    config.evaluation.max_retries,
                ): record.question_id
                for record in pending_records
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result.get("status") != "ok":
                    progress.write(f"[{model.name}] {result.get('question_id')}: {result.get('error')}")
                _write_model_report(output_file, config, model, results, len(records), started_at)
                progress.update(1)

    progress.close()

    report = _write_model_report(output_file, config, model, results, len(records), started_at)
    report["output_file"] = str(output_file)
    return report


def run_project(config: AppConfig) -> Dict[str, Any]:
    label_spec = build_label_spec(config.evaluation.task_mode)
    prompt_templates = {
        "type": load_prompt_template(config.prompt_path),
    }
    if config.evaluation.task_mode == "epd":
        prompt_templates["presence"] = load_prompt_template(config.epd_presence_prompt_path)
        prompt_templates["type"] = load_prompt_template(config.epd_type_prompt_path)
    records = load_dataset(config.dataset.json_dir, config.dataset.image_dir, config.dataset.limit)
    enabled_models = [model for model in config.models if model.enabled]
    if not enabled_models:
        raise ValueError("No enabled models found in config.")

    model_reports: List[Dict[str, Any]] = []
    for model in enabled_models:
        model_reports.append(run_model(config, model, records, label_spec, prompt_templates))

    summary_rows = build_summary_rows(model_reports)
    summary = {
        "project": config.project.name,
        "task_mode": config.evaluation.task_mode,
        "dataset_total": len(records),
        "models": summary_rows,
    }
    summary_dir = config.evaluation.output_dir / _task_mode_dir_name(config.evaluation.task_mode)
    dump_json(summary, summary_dir / "summary.json")
    write_text(render_summary_markdown(summary_rows), summary_dir / "summary.md")
    return summary
