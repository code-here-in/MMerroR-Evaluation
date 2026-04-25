from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


TASK_MODE_ALIASES = {
    "4": "etc",
    "5": "epd",
    "etc": "etc",
    "epd": "epd",
}

PRESENCE_ERROR_LETTER = "P"
PRESENCE_NO_ERROR_LETTER = "N"


@dataclass(frozen=True)
class LabelSpec:
    task_mode: str
    letters: List[str]
    pred_map: Dict[str, str]
    reverse_pred_map: Dict[str, str]
    categories: List[str]
    allow_noerror: bool
    task_instruction: str


def normalize_task_mode(value: str) -> str:
    key = str(value).strip().lower()
    if key not in TASK_MODE_ALIASES:
        raise ValueError("task_mode must be one of: etc, epd, 4, 5")
    return TASK_MODE_ALIASES[key]


def build_label_spec(task_mode: str) -> LabelSpec:
    normalized = normalize_task_mode(task_mode)
    base_categories = [
        (
            "A",
            "A_Visual_Perception_Error",
            "Misread or misinterpret the visual content, such as text, numbers, objects, locations, colors, or counts.",
        ),
        (
            "B",
            "B_Reasoning_Error",
            "Visual facts are correct, but the logic, arithmetic, or multi-step deduction is wrong.",
        ),
        (
            "C",
            "C_Question_Comprehension_Error",
            "The image is read reasonably well, but the question intent or constraint is misunderstood.",
        ),
        (
            "D",
            "D_Knowledge_Deployment_Error",
            "The image and question are understood, but the wrong fact, formula, or external knowledge is applied.",
        ),
    ]

    categories = list(base_categories)
    if normalized == "epd":
        categories.append(
            (
                "E",
                "E_NoError",
                "The reasoning is correct and does not contain a visual, reasoning, question-understanding, or knowledge-use error.",
            )
        )
        task_instruction = (
            "First decide whether the reasoning contains an error. "
            "If there is no error, choose E. Otherwise choose the single best error type."
        )
    else:
        task_instruction = (
            "The reasoning chain is guaranteed to contain exactly one error. "
            "Choose the single best error type from A to D."
        )

    pred_map = {letter: name for letter, name, _ in categories}
    reverse_pred_map = {name: letter for letter, name, _ in categories}
    category_lines: List[str] = []
    for letter, name, description in categories:
        category_lines.append(f"{letter}. {name}")
        category_lines.append(f"   {description}")

    return LabelSpec(
        task_mode=normalized,
        letters=[letter for letter, _, _ in categories],
        pred_map=pred_map,
        reverse_pred_map=reverse_pred_map,
        categories=category_lines,
        allow_noerror=normalized == "epd",
        task_instruction=task_instruction,
    )


def format_letter_list(letters: List[str]) -> str:
    if not letters:
        return ""
    if len(letters) == 1:
        return letters[0]
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    return f"{', '.join(letters[:-1])}, or {letters[-1]}"


def normalize_label(label: str, label_spec: LabelSpec, is_prediction: bool = False) -> Optional[str]:
    if label is None:
        return None

    text = str(label).strip()
    if not text:
        return None

    letters = "".join(label_spec.letters)
    letters_lower = letters.lower()

    if is_prediction:
        candidate = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
        candidate = re.sub(
            r"<\|\s*begin_of_box\s*\|>|<\|\s*end_of_box\s*\|>",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        if not candidate:
            return None

        def match_simple(input_text: str) -> Optional[str]:
            match = re.fullmatch(
                rf"\s*[*_`~$]*\(?\s*([{letters}])\s*\)?[*_`~$]*[.!?:]*\s*",
                input_text,
                re.IGNORECASE,
            )
            if match:
                return match.group(1).upper()
            return None

        boxed_pattern = re.compile(
            rf"\\boxed\s*\{{\s*(?:\\mathbf|\\mathrm|\\text)?\s*\{{?\s*([{letters}])\s*\}}?\s*\}}",
            re.IGNORECASE,
        )

        def match_boxed(input_text: str) -> Optional[str]:
            matches = [match.group(1).upper() for match in boxed_pattern.finditer(input_text)]
            if len(set(matches)) == 1:
                return matches[0]
            return None

        direct = match_simple(candidate)
        if direct:
            return label_spec.pred_map[direct]

        boxed = match_boxed(candidate)
        if boxed:
            return label_spec.pred_map[boxed]

        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        line_matches = [match_simple(line) for line in lines]
        line_matches = [match for match in line_matches if match]
        if len(set(line_matches)) == 1:
            return label_spec.pred_map[line_matches[0]]

        answer_pattern = re.compile(r"^\s*(?:final\s*answer|answer|output)\b\s*[:\-]?\s*(.+)$", re.IGNORECASE)
        answer_matches: List[str] = []
        for line in lines:
            answer_match = answer_pattern.match(line)
            if not answer_match:
                continue
            tail = answer_match.group(1).strip()
            simple = match_simple(tail)
            if simple:
                answer_matches.append(simple)
                continue
            boxed = match_boxed(tail)
            if boxed:
                answer_matches.append(boxed)
        if len(set(answer_matches)) == 1 and answer_matches:
            return label_spec.pred_map[answer_matches[0]]

        label_name = text.strip()
        if label_name in label_spec.reverse_pred_map:
            return label_name
        return None

    lowered = text.lower()
    start_match = re.match(rf"^\s*([{letters_lower}])\b", lowered)
    if start_match:
        letter = start_match.group(1).upper()
        return label_spec.pred_map.get(letter)

    if label_spec.allow_noerror and ("noerror" in lowered or "no error" in lowered):
        return label_spec.pred_map.get("E")
    if "visual" in lowered or "perception" in lowered:
        return "A_Visual_Perception_Error"
    if "reasoning" in lowered:
        return "B_Reasoning_Error"
    if "question" in lowered or "comprehension" in lowered:
        return "C_Question_Comprehension_Error"
    if "knowledge" in lowered:
        return "D_Knowledge_Deployment_Error"
    return None


def presence_letter_for_label(label_name: str) -> str:
    if label_name == "E_NoError":
        return PRESENCE_NO_ERROR_LETTER
    return PRESENCE_ERROR_LETTER


def normalize_presence_decision(value: str) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    simple_match = re.fullmatch(r"\s*([PN])\s*[.!?:]?\s*", text, re.IGNORECASE)
    if simple_match:
        return simple_match.group(1).upper()

    lowered = text.lower()
    if "no error" in lowered or lowered == "none":
        return PRESENCE_NO_ERROR_LETTER
    if "error present" in lowered or lowered == "error":
        return PRESENCE_ERROR_LETTER
    return None
