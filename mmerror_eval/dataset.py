from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class DatasetRecord:
    index: int
    question_id: str
    question: str
    correct_answer: str
    error_reason: str
    label: str
    domain: str
    subdomain: str
    image_path: Optional[Path]


def infer_image_dir(json_dir: Path) -> Path:
    if json_dir.name == "jsons":
        sibling = json_dir.parent / "images"
        return sibling
    return json_dir / "images"


def find_image(image_dir: Optional[Path], question_id: str) -> Optional[Path]:
    if not image_dir or not question_id:
        return None
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = image_dir / f"{question_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_dataset(json_dir: Path, image_dir: Optional[Path] = None, limit: int = -1) -> List[DatasetRecord]:
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory does not exist: {json_dir}")

    resolved_image_dir = image_dir or infer_image_dir(json_dir)
    records: List[DatasetRecord] = []
    json_files = sorted(path for path in json_dir.glob("MMErroR_*.json") if path.name != "MMErroR_all.json")

    for path in json_files:
        content = json.loads(path.read_text(encoding="utf-8"))
        label = content.get("label")
        if not label:
            continue

        records.append(
            DatasetRecord(
                index=len(records),
                question_id=str(content.get("question_id", "")),
                question=str(content.get("question", "")),
                correct_answer=str(content.get("correct_answer", "")),
                error_reason=str(content.get("error_reason", "")),
                label=str(label),
                domain=str(content.get("domain", "")),
                subdomain=str(content.get("subdomain", "")),
                image_path=find_image(resolved_image_dir, str(content.get("question_id", ""))),
            )
        )

        if limit and limit > 0 and len(records) >= limit:
            break

    return records

