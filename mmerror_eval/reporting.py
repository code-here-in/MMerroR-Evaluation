from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List


def compute_model_metrics(results: Iterable[Dict[str, Any]], dataset_total: int) -> Dict[str, Any]:
    successful = [result for result in results if result.get("status") == "ok"]
    failures = [result for result in results if result.get("status") != "ok"]
    correct = sum(1 for result in successful if result.get("correct"))
    scored_total = len(successful)
    accuracy = (correct / scored_total) if scored_total else 0.0
    strict_accuracy = (correct / dataset_total) if dataset_total else 0.0
    coverage = (scored_total / dataset_total) if dataset_total else 0.0

    def group_stats(key: str) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0.0})
        for result in successful:
            group_name = result.get(key) or "Unknown"
            grouped[group_name]["total"] += 1
            if result.get("correct"):
                grouped[group_name]["correct"] += 1
        ordered = dict(sorted(grouped.items(), key=lambda item: item[0]))
        for data in ordered.values():
            data["accuracy"] = (data["correct"] / data["total"]) if data["total"] else 0.0
        return ordered

    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in successful:
        confusion[result["gt_label"]][result["pred_label"]] += 1

    confusion_matrix = {
        gold: dict(sorted(predictions.items(), key=lambda item: item[0]))
        for gold, predictions in sorted(confusion.items(), key=lambda item: item[0])
    }

    domain_accuracy = group_stats("domain")
    macro_accuracy = 0.0
    if domain_accuracy:
        macro_accuracy = sum(item["accuracy"] for item in domain_accuracy.values()) / len(domain_accuracy)

    return {
        "dataset_total": dataset_total,
        "scored_total": scored_total,
        "failed_total": len(failures),
        "correct": correct,
        "accuracy": accuracy,
        "strict_accuracy": strict_accuracy,
        "coverage": coverage,
        "macro_accuracy": macro_accuracy,
        "overall_weighted_accuracy": accuracy,
        "domain_accuracy": domain_accuracy,
        "subdomain_accuracy": group_stats("subdomain"),
        "label_accuracy": group_stats("gt_label"),
        "confusion_matrix": confusion_matrix,
    }


def build_summary_rows(model_reports: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in model_reports:
        rows.append(
            {
                "model": report["model"],
                "task_mode": report["task_mode"],
                "correct": report["correct"],
                "scored_total": report["scored_total"],
                "dataset_total": report["dataset_total"],
                "accuracy": report["accuracy"],
                "macro_accuracy": report.get("macro_accuracy", 0.0),
                "strict_accuracy": report["strict_accuracy"],
                "coverage": report["coverage"],
                "output_file": report["output_file"],
            }
        )
    rows.sort(key=lambda row: row["strict_accuracy"], reverse=True)
    return rows


def render_summary_markdown(rows: Iterable[Dict[str, Any]]) -> str:
    header = [
        "| Model | Task | Correct | Scored | Dataset | Accuracy | Macro | Strict Accuracy | Coverage | Output |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines = []
    for row in rows:
        lines.append(
            "| {model} | {task_mode} | {correct} | {scored_total} | {dataset_total} | {accuracy:.2%} | {macro_accuracy:.2%} | {strict_accuracy:.2%} | {coverage:.2%} | {output_file} |".format(
                **row
            )
        )
    return "\n".join(header + lines)
