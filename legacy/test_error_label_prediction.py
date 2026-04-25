import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def dump(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


class OpenAIWrapper:
    def __init__(self, model, api_base, key, img_detail='high', timeout=300, temperature=0, use_max_tokens=False, max_tokens=2048):
        self.model = model
        self.api_base = api_base
        self.key = key or os.environ.get('OPENAI_API_KEY', '')
        self.img_detail = img_detail
        self.timeout = timeout
        self.temperature = temperature
        self.use_max_tokens = use_max_tokens
        self.max_tokens = max_tokens
        self.fail_msg = 'Failed to obtain answer via API.'

        if not self.key:
            raise ValueError('API key is required. Provide --key or set OPENAI_API_KEY.')
        if self.img_detail not in ('low', 'high', 'auto'):
            raise ValueError("img_detail must be 'low', 'high', or 'auto'.")

    def _auth_header(self):
        if self.key.lower().startswith('bearer '):
            return self.key
        return f'Bearer {self.key}'

    def _encode_image(self, image_path):
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('ascii')
        return f'data:{mime_type};base64,{encoded}'

    def _image_content(self, image_path):
        image_url = {'url': self._encode_image(image_path)}
        if self.img_detail:
            image_url['detail'] = self.img_detail
        return {'type': 'image_url', 'image_url': image_url}

    def _part_to_content(self, part):
        if isinstance(part, dict):
            if 'type' in part and ('text' in part or 'image_url' in part):
                return [part]
            if part.get('type') == 'text':
                return [{'type': 'text', 'text': str(part.get('value', ''))}]
            if part.get('type') == 'image':
                value = part.get('value', '')
                if value:
                    return [self._image_content(value)]
                return []
            return [{'type': 'text', 'text': str(part)}]

        if isinstance(part, str) and os.path.isfile(part):
            return [self._image_content(part)]

        return [{'type': 'text', 'text': str(part)}]

    def _build_messages(self, message):
        if isinstance(message, list) and message and isinstance(message[0], dict) and 'role' in message[0]:
            return message

        content = []
        if isinstance(message, list):
            for part in message:
                content.extend(self._part_to_content(part))
        else:
            content.extend(self._part_to_content(message))

        return [{'role': 'user', 'content': content}]

    def _use_max_completion_tokens(self):
        model_lower = self.model.lower()
        return ('o1' in model_lower) or ('o3' in model_lower) or ('o4' in model_lower) or ('gpt-5' in model_lower)

    def generate(self, message):
        payload = {
            'model': self.model,
            'messages': self._build_messages(message),
            'n': 1,
            'temperature': self.temperature,
        }

        if self.use_max_tokens:
            if self._use_max_completion_tokens():
                payload['max_completion_tokens'] = self.max_tokens
                payload.pop('temperature', None)
            else:
                payload['max_tokens'] = self.max_tokens

        body = json.dumps(payload).encode('utf-8')
        headers = {
            'Content-Type': 'application/json',
            'Authorization': self._auth_header(),
        }
        request = urllib.request.Request(self.api_base, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode('utf-8', errors='replace')
            data = json.loads(raw)
            return data['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f'{self.fail_msg} {e}'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/jsons')
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--api_base', type=str, default=os.environ.get('OPENAI_API_BASE', "https://api.openai.com/v1/chat/completions"))
    parser.add_argument('--key', type=str, default=os.environ.get('OPENAI_API_KEY', ''))
    parser.add_argument('--models', type=str, default="gpt-4.1-mini")
    parser.add_argument('--workers', type=int, default=64, help='Number of concurrent workers')
    parser.add_argument('--img_detail', type=str, default='high', help='Image detail level: low, high, or auto')
    parser.add_argument('--timeout', type=int, default=600, help='Per-request timeout in seconds')
    parser.add_argument('--resume_file', type=str, default='', help='Resume from an existing results JSON (skips already-successful samples).')
    parser.add_argument('--model_concurrency', type=int, default=1, help='Number of models to run in parallel')
    parser.add_argument('--max_retries', type=int, default=5, help='Retries for exceptions/invalid responses per sample.')
    return parser.parse_args()

def normalize_label(label, is_pred=False):
    if not label:
        return None
    label_str = str(label).strip()
    import re

    pred_map = {
        'A': 'A_Visual_Perception_Error',
        'B': 'B_Reasoning_Error',
        'C': 'C_Question_Comprehension_Error',
        'D': 'D_Knowledge_Deployment_Error'
    }

    # For model predictions: accept ONLY a bare single letter (or boxed letter). Any extra text invalidates.
    if is_pred:
        # If the model included deliberate traces like <think>...</think>, keep only the last non-empty line
        label_candidate = label_str
        label_candidate = re.sub(r"</?think>", "", label_candidate, flags=re.IGNORECASE)
        lines = [ln.strip() for ln in label_candidate.splitlines() if ln.strip()]
        if lines:
            label_candidate = lines[-1]
        else:
            label_candidate = label_candidate.strip()

        boxed_match = re.fullmatch(r"\s*\\boxed\{([A-D])\}\s*", label_candidate, re.IGNORECASE)
        if boxed_match:
            return pred_map[boxed_match.group(1).upper()]

        # Allow markdown-styled single letters such as **B** or `C`
        markdown_match = re.fullmatch(r"\s*[*_`~]*([A-D])[*_`~]*\s*", label_candidate, re.IGNORECASE)
        if markdown_match:
            return pred_map[markdown_match.group(1).upper()]

        letter_match = re.fullmatch(r"\s*([A-D])\s*", label_candidate, re.IGNORECASE)
        if letter_match:
            return pred_map[letter_match.group(1).upper()]

        return None  # Any other text is invalid

    # For ground-truth labels: allow letters at start or keyword names
    label_lower = label_str.lower()
    start_match = re.match(r'^\s*([a-d])\b', label_lower)
    if start_match:
        char = start_match.group(1).upper()
        return pred_map.get(char)

    if 'knowledge' in label_lower:
        return 'D_Knowledge_Deployment_Error'
    if 'comprehension' in label_lower:
        return 'C_Question_Comprehension_Error'
    if 'perception' in label_lower:
        return 'A_Visual_Perception_Error'
    if 'reasoning' in label_lower:
        return 'B_Reasoning_Error'
    return None

def process_sample(model, item, image_dir, i, max_retries=0):
    question = item.get('question', '')
    error_reason = item.get('error_reason', '')
    question_id = item.get('question_id', '')
    gt_label = normalize_label(item.get('label'))
    
    if not gt_label:
        return None

    prompt = f"""
You are grading a reasoning for the question below. Decide which single category (A/B/C/D) best describes the situation.

STRICT OUTPUT FORMAT:
- Respond with EXACTLY one uppercase letter: A, B, C, or D.
- No spaces, no punctuation, no Markdown/bolding, no <think>, no analysis, no explanation, no restatement.
- Do NOT include any chain-of-thought; your reply must be exactly 1 character matching ^[A-D]$.

Question: {question}
Reasoning: {error_reason}

Categories:
A. A_Visual_Perception_Error
    Misread or misinterpret the visual content (numbers/text/objects/locations/colors/counts). Reasoning steps are otherwise fine given the misread input.
B. B_Reasoning_Error
    Visual facts are correct, but the logic/math/formula application is wrong (bad arithmetic, wrong unit handling, faulty deduction).
C. C_Question_Comprehension_Error
    Visual reading is fine, but the question is misunderstood (answers a different question, ignores constraints, swaps entities/targets).
D. D_Knowledge_Deployment_Error
    Visual reading and question understanding are fine, but wrong external knowledge/formula/fact is applied.

Output: one letter only (A/B/C/D).
"""
    # Prepare input message
    message = [prompt]
    
    # Try to find image
    if question_id:
        image_path = os.path.join(image_dir, f"{question_id}.png")
        if os.path.exists(image_path):
            message = [image_path, prompt]
        else:
            # Try jpg
            image_path_jpg = os.path.join(image_dir, f"{question_id}.jpg")
            if os.path.exists(image_path_jpg):
                message = [image_path_jpg, prompt]
            else:
                # print(f"Warning: Image for {question_id} not found at {image_path} or {image_path_jpg}")
                pass

    # Call model with retries on exceptions or invalid responses
    response = None
    pred_label = None
    last_error = None
    max_retries = max(0, max_retries)
    attempts = max_retries + 1
    for attempt in range(attempts):
        try:
            response = model.generate(message)  # single request; rely on long timeout
            pred_label = normalize_label(response, is_pred=True)
            if response and "Failed to obtain answer via API" not in response and pred_label:
                break
            last_error = f"Invalid response: {response}"
        except Exception as e:
            last_error = f"Exception: {e}"
        if attempt < max_retries:
            time.sleep(2)

    if response is None or "Failed to obtain answer via API" in response or not pred_label:
        return {'index': i, 'error': f"Failed after {attempts} attempts. Last error: {last_error}"}
        
    
    return {
        'index': i,
        'question_id': question_id,
        'gt_label': gt_label,
        'pred_label': pred_label,
        'correct': pred_label == gt_label
    }

def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    result_root = repo_root / "result"

    def compute_domain_stats(output_file: Path):
        try:
            import sys
            import tempfile
            import os
            scripts_dir = repo_root / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from compute_domain_accuracy import summarize

            json_dir = repo_root / "finaldata" / "jsons"
            with tempfile.NamedTemporaryFile('w+', delete=False, suffix='.json') as tmpf:
                summarize(Path(output_file), json_dir, Path(tmpf.name))
                tmpf.seek(0)
                domain_data = json.load(tmpf)
            os.remove(tmpf.name)
            return domain_data
        except Exception as e:
            print(f"Failed to compute domain stats: {e}")
            return {}
    
    models_to_test = [x.strip() for x in args.models.split(',')]

    def run_single_model(model_name: str):
        print(f"\n{'='*30}")
        print(f"Testing model: {model_name}")
        print(f"{'='*30}")

        # Decide output paths (default or resume-specified)
        results = []
        correct = 0
        total = 0
        skip_indices = set()

        safe_model_name = model_name.replace('/', '--')
        model_dir = result_root / safe_model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # 优化：每次检测输出文件夹是否已有对应results文件，若有则自动继续写入该文件
        # 优先使用--resume_file参数，其次自动检测最新文件，否则新建
        if args.resume_file:
            output_file = Path(args.resume_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            domain_file = output_file.with_name(f"{output_file.stem}_domain.json")
        else:
            # 检查是否已有同名results文件（按模型名匹配）
            pattern = f"results_{safe_model_name}_*.json"
            existing_main = sorted(p for p in model_dir.glob(pattern) if not p.name.endswith('_domain.json'))
            existing_domain_only = sorted(p for p in model_dir.glob(pattern) if p.name.endswith('_domain.json'))
            if existing_main:
                output_file = existing_main[-1]
                domain_file = output_file.with_name(f"{output_file.stem}_domain.json")
                print(f"Auto-resuming from latest file: {output_file}")
            elif existing_domain_only:
                # 兼容只有 _domain.json 的旧输出
                output_file = existing_domain_only[-1]
                domain_file = output_file
                print(f"Auto-resuming from domain-only file: {output_file}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = f"results_{safe_model_name}_{timestamp}"
                output_file = model_dir / f"{base_name}.json"
                domain_file = model_dir / f"{base_name}_domain.json"

        # Load resume content if the chosen output file already exists
        try:
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    resume_data = json.load(f)
                # Ensure resume file corresponds to the same model
                resumed_model = resume_data.get('model')
                if resumed_model and resumed_model != model_name:
                    print(f"Resume file model mismatch: file model={resumed_model}, current model={model_name}. Abort.")
                    return
                results = resume_data.get('results', [])
                correct = sum(1 for r in results if r.get('correct'))
                total = len(results)
                # Rerun error entries; skip only non-error ones
                skip_indices = {r.get('index') for r in results if 'error' not in r and r.get('index') is not None}
                if results:
                    print(f"Resuming from {output_file}: loaded {len(results)} records, skipping {len(skip_indices)} completed samples.")
                # If the file already covers every sample without errors, skip rerun
                total_samples = len(data)
                has_errors = any('error' in r for r in results)
                if total_samples > 0 and len(skip_indices) >= total_samples and not has_errors:
                    print(f"Existing results cover all {total_samples} samples; skipping rerun.")
                    return
        except Exception as e:
            print(f"Failed to load resume file {output_file}: {e}")
        
        # Initialize model
        try:
            print(f"Initializing CustomAPI with model {model_name}, img_detail={args.img_detail}...")
            model = OpenAIWrapper(model=model_name, api_base=args.api_base, key=args.key, img_detail=args.img_detail, timeout=args.timeout)
        except Exception as e:
            print(f"Failed to initialize model {model_name}: {e}")
            return
        
        def write_progress():
            # Sort for readability and stable output
            sorted_results = sorted(results, key=lambda x: x['index'])
            acc = (correct / total) if total > 0 else 0
            output_data = {
                'model': model_name,
                'accuracy': acc,
                'correct': correct,
                'total': total,
            }
            # Lightweight progress write: domain stats are computed once at the end
            output_data['results'] = sorted_results
            dump(output_data, output_file)
        
        print(f"Testing on {len(data)} samples with up to {args.workers} workers...")
        pending = [(i, item) for i, item in enumerate(data) if i not in skip_indices]
        if skip_indices:
            print(f"Skipping {len(skip_indices)} already-processed samples; processing {len(pending)} remaining.")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_sample, model, item, image_dir, i, args.max_retries): i for i, item in pending}

            for future in as_completed(futures):
                res = future.result()
                if res and 'error' not in res:
                    results.append(res)
                    if res['correct']:
                        correct += 1
                    total += 1
                    write_progress()
                    print(f"Sample {res['index']+1}: GT: {res['gt_label']}, Pred: {res['pred_label']}")
                elif res and 'error' in res:
                    print(f"Error on sample {res['index']}: {res['error']}")
        
        # Sort results by index
        results.sort(key=lambda x: x['index'])

        if total > 0:
            acc = correct/total
            print(f"Accuracy for {model_name}: {acc:.2f} ({correct}/{total})")
        else:
            acc = 0
            print(f"No valid samples processed for {model_name}.")
            
        # Save results into model-specific folder under result/<model_name>/
        output_data = {
            'model': model_name,
            'accuracy': acc,
            'correct': correct,
            'total': total,
        }
        domain_data = compute_domain_stats(output_file)
        for k in ['summary', 'overall', 'domain_accuracy', 'missing_question_ids', 'table_row']:
            if k in domain_data:
                output_data[k] = domain_data[k]
        output_data['results'] = results
        dump(output_data, output_file)
        print(f"Results saved to {output_file}")

        # 不再生成_domain.json文件，领域统计已合并到主文件

    # Load data
    if not os.path.exists(args.data_dir):
        print(f"Data directory {args.data_dir} does not exist.")
        return

    # Assume images are in a sibling directory 'images' if data_dir ends with 'jsons'
    if args.data_dir.endswith('jsons'):
        image_dir = args.data_dir.replace('jsons', 'images')
    else:
        # Fallback or assume data_dir is the root MMMU folder
        image_dir = os.path.join(args.data_dir, 'images')
    
    if not os.path.exists(image_dir):
        print(f"Warning: Image directory {image_dir} does not exist. Proceeding with text-only if possible.")

    json_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.json')])
    data = []
    for f in json_files:
        try:
            with open(os.path.join(args.data_dir, f), 'r') as f_in:
                content = json.load(f_in)
            if 'label' in content and content['label']:
                data.append(content)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    print(f"Found {len(data)} samples with labels.")
    if args.limit > 0:
        data = data[:args.limit]
    

    # Loop over models (optionally in parallel)
    model_concurrency = max(1, min(args.model_concurrency, len(models_to_test)))
    if model_concurrency == 1 or len(models_to_test) == 1:
        for model_name in models_to_test:
            run_single_model(model_name)
    else:
        print(f"Running models in parallel (concurrency={model_concurrency})...")
        with ThreadPoolExecutor(max_workers=model_concurrency) as executor:
            futures = {executor.submit(run_single_model, model_name): model_name for model_name in models_to_test}
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Model {model_name} failed: {e}")

if __name__ == '__main__':
    main()
