from __future__ import annotations

from pathlib import Path
from typing import Optional

from .labels import LabelSpec, format_letter_list


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_prompt(
    template: str,
    question: str,
    error_reason: str,
    label_spec: LabelSpec,
    task_mode: Optional[str] = None,
    task_instruction: Optional[str] = None,
) -> str:
    letters_slash = "/".join(label_spec.letters)
    letters_csv = format_letter_list(label_spec.letters)
    letters_regex = "".join(label_spec.letters)
    categories = "\n".join(label_spec.categories)
    return template.format(
        task_mode=(task_mode or label_spec.task_mode).upper(),
        task_instruction=(task_instruction or label_spec.task_instruction),
        letters_slash=letters_slash,
        letters_csv=letters_csv,
        letters_regex=letters_regex,
        categories=categories,
        question=question,
        error_reason=error_reason,
    )


def render_presence_prompt(template: str, question: str, error_reason: str) -> str:
    categories = "\n".join(
        [
            "P. Error Present",
            "   The reasoning contains at least one error and should proceed to error-type diagnosis.",
            "N. No Error",
            "   The reasoning is fully correct and should not proceed to error-type diagnosis.",
        ]
    )
    return template.format(
        letters_slash="P/N",
        letters_csv="P or N",
        letters_regex="PN",
        categories=categories,
        question=question,
        error_reason=error_reason,
    )
