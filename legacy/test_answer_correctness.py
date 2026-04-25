#!/usr/bin/env python3
"""
Call an OpenAI-compatible API to have a model answer each question, then judge
whether the answer matches the ground-truth `correct_answer`.

Key features:
- Skips samples whose `correct_answer` is empty or equals "？".
- Sends the paired image (PNG/JPG) when available.
- Supports multiple models, auto resume, and concurrency.
- Judging can be done by an LLM (default) or by simple string matching.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------- API wrapper (mirrors the current config-driven evaluator behavior) -------- #
class OpenAIWrapper:
    def __init__(self, model, api_base, key, img_detail="high", timeout=300, temperature=0, use_max_tokens=False, max_tokens=256):
        self.model = model
        self.api_base = api_base
        self.key = key or os.environ.get("OPENAI_API_KEY", "")
        self.img_detail = img_detail
        self.timeout = timeout
        self.temperature = temperature
        self.use_max_tokens = use_max_tokens
        self.max_tokens = max_tokens
        self.fail_msg = "Failed to obtain answer via API."

        if not self.key:
            raise ValueError("API key is required. Provide --key or set OPENAI_API_KEY.")
        if self.img_detail not in ("low", "high", "auto"):
            raise ValueError("img_detail must be 'low', 'high', or 'auto'.")

    def _auth_header(self):
        if self.key.lower().startswith("bearer "):
            return self.key
        return f"Bearer {self.key}"

    def _encode_image(self, image_path: str) -> str:
        import base64
        import mimetypes

        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _image_content(self, image_path: str) -> dict:
        image_url = {"url": self._encode_image(image_path)}
        if self.img_detail:
            image_url["detail"] = self.img_detail
        return {"type": "image_url", "image_url": image_url}

    def _part_to_content(self, part: Any) -> List[dict]:
        if isinstance(part, dict):
            if "type" in part and ("text" in part or "image_url" in part):
                return [part]
            if part.get("type") == "text":
                return [{"type": "text", "text": str(part.get("value", ""))}]
            if part.get("type") == "image":
                value = part.get("value", "")
                if value:
                    return [self._image_content(value)]
                return []
            return [{"type": "text", "text": str(part)}]

        if isinstance(part, str) and os.path.isfile(part):
            return [self._image_content(part)]

        return [{"type": "text", "text": str(part)}]

    def _build_messages(self, message: Any) -> List[dict]:
        if isinstance(message, list) and message and isinstance(message[0], dict) and "role" in message[0]:
            return message

        content: List[dict] = []
        if isinstance(message, list):
            for part in message:
                content.extend(self._part_to_content(part))
        else:
            content.extend(self._part_to_content(message))

        return [{"role": "user", "content": content}]

    def _use_max_completion_tokens(self) -> bool:
        model_lower = self.model.lower()
        return ("o1" in model_lower) or ("o3" in model_lower) or ("o4" in model_lower) or ("gpt-5" in model_lower)

    def generate(self, message: Any) -> str:
        payload = {
            "model": self.model,
            "messages": self._build_messages(message),
            "n": 1,
            "temperature": self.temperature,
        }

        if self.use_max_tokens:
            if self._use_max_completion_tokens():
                payload["max_completion_tokens"] = self.max_tokens
                payload.pop("temperature", None)
            else:
                payload["max_tokens"] = self.max_tokens

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": self._auth_header(),
        }
        import urllib.request

        request = urllib.request.Request(self.api_base, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"].strip()


# -------- Helpers -------- #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/jsons")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of samples (after filtering).")
    parser.add_argument("--question_ids_file", type=str, default="", help="Optional file with question_ids (one per line) to restrict evaluation.")
    parser.add_argument("--api_base", type=str, default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions"))
    parser.add_argument("--key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--models", type=str, default="gpt-4o-mini", help="Comma-separated answer models.")
    parser.add_argument("--judge_model", type=str, default="", help="LLM used to judge correctness. Default: same as answer model.")
    parser.add_argument("--judge_api_base", type=str, default="", help="Optional different API base for the judge.")
    parser.add_argument("--judge_key", type=str, default="", help="Optional different API key for the judge.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--img_detail", type=str, default="high")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--use_max_tokens", action="store_true", help="Use max_tokens/max_completion_tokens for generations.")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--result_root", type=str, default="../result_answer", help="Directory to save QA results.")
    parser.add_argument("--resume_file", type=str, default="", help="Resume from an existing result JSON.")
    parser.add_argument("--judge_mode", choices=["llm", "string"], default="llm", help="How to judge correctness.")
    parser.add_argument("--llm_on_incorrect_only", action="store_true", help="If set and judge_mode=llm, first try string match; only call LLM judge when string deems incorrect.")
    parser.add_argument("--seed", type=int, default=42, help="For deterministic sampling when limit > 0.")
    parser.add_argument("--run_tag", type=str, default="", help="Optional tag stored in output for distinguishing runs (e.g., correct500/incorrect500).")
    return parser.parse_args()


def load_dataset(data_dir: Path, limit: int = -1, seed: int = 42, id_allowlist: Optional[set] = None) -> List[dict]:
    import random

    json_files = sorted([f for f in data_dir.glob("*.json")])
    items: List[dict] = []
    for path in json_files:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        correct = str(obj.get("correct_answer", "")).strip()
        if not correct or correct == "？":
            continue
        if id_allowlist is not None:
            qid = str(obj.get("question_id", "")).strip()
            if not qid or qid not in id_allowlist:
                continue
        obj["_source_path"] = path
        items.append(obj)

    if limit and limit > 0 and limit < len(items):
        random.Random(seed).shuffle(items)
        items = items[:limit]
    return items


def build_answer_prompt(question: str) -> str:
    return (
        "Answer the question concisely. Provide only the final answer; no reasoning, no formatting, no markdown.\n\n"
        f"Question: {question}"
    )


def build_judge_prompt(question: str, gt_answer: str, model_answer: str) -> str:
    return f"""
You are a strict grader. Decide if the model's answer matches the ground truth in meaning (ignoring casing and punctuation).

Respond with exactly one token: correct or incorrect.

Question: {question}
Ground truth answer: {gt_answer}
Model answer: {model_answer}

Output: correct/incorrect only.
""".strip()


def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.lower()
    text = re.sub(r"[\s\u00a0]+", " ", text)
    text = re.sub(r"[\"'“”`]+", "", text)
    text = text.strip(" .,!?:;")
    return text


def simple_match(gt: str, pred: str) -> bool:
    gt_norm = normalize_text(gt)
    pred_norm = normalize_text(pred)
    if not pred_norm:
        return False
    if gt_norm == pred_norm:
        return True
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True
    # Basic plural handling (e.g., herbivore vs herbivores)
    if gt_norm.endswith("s") and gt_norm[:-1] == pred_norm:
        return True
    if pred_norm.endswith("s") and pred_norm[:-1] == gt_norm:
        return True
    return False


def parse_judge_response(resp: str) -> Optional[bool]:
    if not resp:
        return None
    text = resp.strip().lower()
    if text.startswith("correct"):
        return True
    if text.startswith("incorrect"):
        return False
    if text in {"yes", "true", "1"}:
        return True
    if text in {"no", "false", "0"}:
        return False
    return None


def find_image_path(question_id: str, image_dir: Path) -> Optional[Path]:
    png = image_dir / f"{question_id}.png"
    if png.exists():
        return png
    jpg = image_dir / f"{question_id}.jpg"
    if jpg.exists():
        return jpg
    return None


def process_sample(
    idx: int,
    item: dict,
    answer_model: OpenAIWrapper,
    judge_model: Optional[OpenAIWrapper],
    image_dir: Path,
    judge_mode: str,
    llm_on_incorrect_only: bool,
    max_retries: int,
) -> dict:
    question = item.get("question", "")
    correct_answer = item.get("correct_answer", "")
    question_id = item.get("question_id", "")

    # Build answer message
    prompt = build_answer_prompt(question)
    message: List[Any] = [prompt]
    image_path = find_image_path(question_id, image_dir) if question_id else None
    if image_path:
        message = [str(image_path), prompt]

    # Call answer model with retries
    answer_raw = None
    last_error = None
    attempts = max(0, max_retries) + 1
    for attempt in range(attempts):
        try:
            answer_raw = answer_model.generate(message)
            if answer_raw:
                break
            last_error = "Empty response."
        except Exception as e:
            last_error = f"Exception: {e}"
        if attempt < attempts - 1:
            time.sleep(2)

    if not answer_raw:
        return {"index": idx, "question_id": question_id, "error": f"Failed to get answer. Last error: {last_error}"}

    answer_norm = normalize_text(answer_raw)

    # Judge correctness
    judged_correct: Optional[bool] = None
    judge_response = ""
    if judge_mode == "llm" and judge_model is not None:
        # Optional pre-check via string match to save LLM calls
        if llm_on_incorrect_only:
            quick = simple_match(correct_answer, answer_norm)
            if quick:
                judged_correct = True
                judge_response = "string_match"
            else:
                judge_response = ""
        judge_prompt = build_judge_prompt(question, correct_answer, answer_raw)
        judge_message: List[Any] = [judge_prompt]
        if image_path:
            judge_message = [str(image_path), judge_prompt]

        # Only call LLM judge if we still need a decision
        if judged_correct is None:
            for attempt in range(attempts):
                try:
                    judge_response = judge_model.generate(judge_message)
                    judged_correct = parse_judge_response(judge_response)
                    if judged_correct is not None:
                        break
                except Exception as e:
                    judge_response = f"Exception: {e}"
                if attempt < attempts - 1:
                    time.sleep(2)
    else:
        judged_correct = simple_match(correct_answer, answer_norm)
        judge_response = "string_match"

    if judged_correct is None:
        # Fallback to string match if judge failed
        judged_correct = simple_match(correct_answer, answer_norm)

    return {
        "index": idx,
        "question_id": question_id,
        "gt_answer": correct_answer,
        "pred_answer_raw": answer_raw,
        "pred_answer_norm": answer_norm,
        "correct": bool(judged_correct),
        "judge_response": judge_response,
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = (repo_root / args.data_dir) if not os.path.isabs(args.data_dir) else Path(args.data_dir)
    result_root = (repo_root / args.result_root) if not os.path.isabs(args.result_root) else Path(args.result_root)

    if not data_dir.exists():
        raise SystemExit(f"Data directory {data_dir} does not exist.")

    # Infer image directory using the same jsons/images convention as the main evaluator.
    if str(data_dir).endswith("jsons"):
        image_dir = Path(str(data_dir).replace("jsons", "images"))
    else:
        image_dir = data_dir / "images"

    id_allowlist = None
    if args.question_ids_file:
        qfile = Path(args.question_ids_file)
        if not qfile.exists():
            raise SystemExit(f"question_ids_file not found: {qfile}")
        id_allowlist = {line.strip() for line in qfile.read_text(encoding="utf-8").splitlines() if line.strip()}
        print(f"Loaded {len(id_allowlist)} question_ids from {qfile}.")

    data = load_dataset(data_dir, limit=args.limit, seed=args.seed, id_allowlist=id_allowlist)
    print(f"Loaded {len(data)} samples after filtering empty/'？' answers from {data_dir}.")

    models_to_test = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_name in models_to_test:
        safe_model_name = model_name.replace("/", "--")
        model_dir = result_root / safe_model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Determine output/resume file
        if args.resume_file:
            output_file = Path(args.resume_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            pattern = f"results_{safe_model_name}_*.json"
            existing = sorted(model_dir.glob(pattern))
            if existing:
                output_file = existing[-1]
                print(f"Auto-resuming from latest file: {output_file}")
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = model_dir / f"results_{safe_model_name}_{ts}.json"

        # Load resume content
        results: List[Dict[str, Any]] = []
        correct = 0
        skip_indices = set()
        try:
            if output_file.exists():
                with output_file.open("r", encoding="utf-8") as f:
                    resume_data = json.load(f)
                if resume_data.get("model") and resume_data["model"] != model_name:
                    print(f"Resume file model mismatch: {resume_data.get('model')} vs {model_name}; starting fresh.")
                else:
                    results = resume_data.get("results", [])
                    correct = sum(1 for r in results if r.get("correct"))
                    skip_indices = {r.get("index") for r in results if "error" not in r}
                    print(f"Resuming: loaded {len(results)} records, skipping {len(skip_indices)} samples.")
        except Exception as e:
            print(f"Failed to load resume file {output_file}: {e}")

        # Initialize answer and judge models
        try:
            answer_model = OpenAIWrapper(
                model=model_name,
                api_base=args.api_base,
                key=args.key,
                img_detail=args.img_detail,
                timeout=args.timeout,
                temperature=args.temperature,
                use_max_tokens=args.use_max_tokens,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            print(f"Failed to initialize answer model {model_name}: {e}")
            continue

        judge_model_name = args.judge_model.strip() or model_name
        judge_api_base = args.judge_api_base.strip() or args.api_base
        judge_key = args.judge_key.strip() or args.key
        judge_model = None
        if args.judge_mode == "llm":
            try:
                judge_model = OpenAIWrapper(
                    model=judge_model_name,
                    api_base=judge_api_base,
                    key=judge_key,
                    img_detail=args.img_detail,
                    timeout=args.timeout,
                    temperature=0,
                    use_max_tokens=False,
                    max_tokens=128,
                )
            except Exception as e:
                print(f"Failed to initialize judge model {judge_model_name}: {e}")
                judge_model = None

        pending = [(i, item) for i, item in enumerate(data) if i not in skip_indices]
        print(f"Testing model {model_name}: {len(pending)} samples to process (total {len(data)}).")

        def write_progress():
            sorted_results = sorted(results, key=lambda x: x["index"])
            total = len(sorted_results)
            acc = (correct / total) if total else 0.0
            payload = {
                "model": model_name,
                "run_tag": args.run_tag,
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "judge_mode": args.judge_mode,
                "judge_model": judge_model_name if args.judge_mode == "llm" else "string_match",
                "results": sorted_results,
            }
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_sample,
                    i,
                    item,
                    answer_model,
                    judge_model,
                    image_dir,
                    args.judge_mode,
                    args.llm_on_incorrect_only,
                    args.max_retries,
                ): i
                for i, item in pending
            }
            for future in as_completed(futures):
                res = future.result()
                if res is None:
                    continue
                results.append(res)
                if res.get("correct"):
                    correct += 1
                write_progress()

        # Final write with sorted results
        results.sort(key=lambda x: x["index"])
        total = len(results)
        final_acc = (correct / total) if total else 0.0
        payload = {
            "model": model_name,
            "run_tag": args.run_tag,
            "accuracy": final_acc,
            "correct": correct,
            "total": total,
            "judge_mode": args.judge_mode,
            "judge_model": judge_model_name if args.judge_mode == "llm" else "string_match",
            "results": results,
        }
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Done. Accuracy {final_acc:.4f} ({correct}/{total}). Results saved to {output_file}")


if __name__ == "__main__":
    main()
