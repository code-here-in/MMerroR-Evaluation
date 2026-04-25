from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import EvaluationConfig, ModelConfig, ProviderDefaults


REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh")
MODE_CHOICES = ("off", "thinking", "minimal", "low", "medium", "high", "xhigh")
MODE_API_CHOICES = (
    "auto",
    "openai",
    "responses",
    "anthropic",
    "gemini",
    "kimi",
    "doubao",
    "glm",
    "stepfun",
    "internvl",
    "momo",
)


@dataclass
class ProviderContext:
    gold_letter: Optional[str] = None
    stage: str = "type"


@dataclass
class ProviderSettings:
    provider_type: str
    model_name: str
    api_base: Optional[str]
    api_key: Optional[str]
    img_detail: str
    timeout_sec: int
    temperature: float
    use_max_completion_tokens: bool
    max_completion_tokens: int
    mode: str
    mode_api: str
    thinking_budget_tokens: int
    include_thoughts: bool
    extra_body: Dict[str, Any]
    strategy: Optional[str] = None
    constant_label: Optional[str] = None


class BaseProvider:
    def generate(self, message: List[Any], context: Optional[ProviderContext] = None) -> str:
        raise NotImplementedError


def _env_first(*names: Optional[str]) -> Optional[str]:
    for name in names:
        if not name:
            continue
        value = os.environ.get(name)
        if value:
            return value
    return None


def merge_provider_settings(
    model: ModelConfig,
    defaults: ProviderDefaults,
    evaluation: EvaluationConfig,
) -> ProviderSettings:
    api_base = model.api_base or _env_first(model.api_base_env, defaults.api_base_env, "OPENAI_API_BASE") or defaults.api_base
    api_key = model.api_key or _env_first(model.api_key_env, defaults.api_key_env, "OPENAI_API_KEY") or defaults.api_key
    extra_body = dict(defaults.extra_body)
    extra_body.update(model.extra_body)

    return ProviderSettings(
        provider_type=model.provider_type or defaults.provider_type,
        model_name=model.name,
        api_base=api_base,
        api_key=api_key,
        img_detail=model.img_detail or defaults.img_detail,
        timeout_sec=evaluation.timeout_sec,
        temperature=evaluation.temperature,
        use_max_completion_tokens=evaluation.use_max_completion_tokens,
        max_completion_tokens=evaluation.max_completion_tokens,
        mode=(model.mode or defaults.mode or "off").strip().lower(),
        mode_api=(model.mode_api or defaults.mode_api or "auto").strip().lower(),
        thinking_budget_tokens=(
            int(model.thinking_budget_tokens)
            if model.thinking_budget_tokens is not None
            else int(defaults.thinking_budget_tokens)
        ),
        include_thoughts=(
            bool(model.include_thoughts)
            if model.include_thoughts is not None
            else bool(defaults.include_thoughts)
        ),
        extra_body=extra_body,
        strategy=model.strategy,
        constant_label=model.constant_label,
    )


def create_provider(settings: ProviderSettings) -> BaseProvider:
    if settings.provider_type == "mock":
        return MockProvider(settings)
    if settings.provider_type != "openai_compatible":
        raise ValueError(f"Unsupported provider type: {settings.provider_type}")
    return OpenAICompatibleProvider(settings)


class MockProvider(BaseProvider):
    def __init__(self, settings: ProviderSettings) -> None:
        self.strategy = (settings.strategy or "gold_label").strip().lower()
        self.constant_label = settings.constant_label or "A"

    def generate(self, message: List[Any], context: Optional[ProviderContext] = None) -> str:
        _ = message
        if self.strategy == "gold_label":
            if not context or not context.gold_letter:
                raise ValueError("Mock provider strategy=gold_label requires context.gold_letter.")
            if context.stage == "presence":
                return "N" if context.gold_letter == "E" else "P"
            return context.gold_letter
        if self.strategy == "constant_label":
            return self.constant_label
        raise ValueError(f"Unsupported mock strategy: {self.strategy}")


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.fail_message = "Failed to obtain answer via API."

        if not self.settings.api_key:
            raise ValueError("API key is required. Set MMERROR_API_KEY (or OPENAI_API_KEY as a compatibility alias).")
        if not self.settings.api_base:
            raise ValueError("API base is required. Set MMERROR_API_BASE (or OPENAI_API_BASE as a compatibility alias), or configure api_base.")
        if self.settings.img_detail not in ("low", "high", "auto"):
            raise ValueError("img_detail must be low, high, or auto.")
        if self.settings.mode not in MODE_CHOICES:
            raise ValueError(f"mode must be one of: {', '.join(MODE_CHOICES)}")
        if self.settings.mode_api not in MODE_API_CHOICES:
            raise ValueError(f"mode_api must be one of: {', '.join(MODE_API_CHOICES)}")

    def generate(self, message: List[Any], context: Optional[ProviderContext] = None) -> str:
        _ = context
        payload: Dict[str, Any] = {
            "model": self.settings.model_name,
            "messages": self._build_messages(message),
            "n": 1,
            "temperature": self.settings.temperature,
        }

        if self.settings.use_max_completion_tokens:
            if self._use_max_completion_tokens():
                payload["max_completion_tokens"] = self.settings.max_completion_tokens
                payload.pop("temperature", None)
            else:
                payload["max_tokens"] = self.settings.max_completion_tokens

        self._apply_mode_payload(payload)
        if self._detect_mode_api() == "anthropic":
            self._ensure_anthropic_thinking_constraints(payload)
        if self._detect_mode_api() == "kimi" and self.settings.temperature == 0:
            payload.pop("temperature", None)

        try:
            return self._send(payload)
        except Exception as error:
            if self.settings.mode != "off":
                try:
                    fallback_payload = json.loads(json.dumps(payload))
                    fallback_payload.pop("reasoning_effort", None)
                    fallback_payload.pop("reasoning", None)
                    fallback_payload.pop("thinking", None)
                    extra_body = fallback_payload.get("extra_body")
                    if isinstance(extra_body, dict):
                        google_cfg = extra_body.get("google")
                        if isinstance(google_cfg, dict):
                            google_cfg.pop("thinking_config", None)
                            if not google_cfg:
                                extra_body.pop("google", None)
                        if not extra_body:
                            fallback_payload.pop("extra_body", None)
                    return self._send(fallback_payload)
                except Exception as fallback_error:
                    return f"{self.fail_message} mode_error={error}; fallback_error={fallback_error}"
            return f"{self.fail_message} {error}"

    def _send(self, payload: Dict[str, Any]) -> str:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": self._auth_header(),
        }
        request = urllib.request.Request(self.settings.api_base, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=self.settings.timeout_sec) as response:
            raw = response.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"].strip()

    def _auth_header(self) -> str:
        api_key = self.settings.api_key or ""
        if api_key.lower().startswith("bearer "):
            return api_key
        return f"Bearer {api_key}"

    def _use_max_completion_tokens(self) -> bool:
        model_lower = self.settings.model_name.lower()
        return any(token in model_lower for token in ("o1", "o3", "o4", "gpt-5"))

    def _encode_image(self, image_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"
        with open(image_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _image_content(self, image_path: str) -> Dict[str, Any]:
        image_url = {"url": self._encode_image(image_path)}
        if self.settings.img_detail:
            image_url["detail"] = self.settings.img_detail
        return {"type": "image_url", "image_url": image_url}

    def _part_to_content(self, part: Any) -> List[Dict[str, Any]]:
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

        if isinstance(part, (str, Path)) and Path(part).is_file():
            return [self._image_content(str(part))]

        return [{"type": "text", "text": str(part)}]

    def _build_messages(self, message: List[Any]) -> List[Dict[str, Any]]:
        if message and isinstance(message[0], dict) and "role" in message[0]:
            return message

        content: List[Dict[str, Any]] = []
        for part in message:
            content.extend(self._part_to_content(part))
        return [{"role": "user", "content": content}]

    def _mode_to_effort(self) -> Optional[str]:
        if self.settings.mode == "off":
            return None
        if self.settings.mode == "thinking":
            return "high"
        if self.settings.mode in REASONING_EFFORT_VALUES:
            return self.settings.mode
        return None

    def _detect_mode_api(self) -> str:
        if self.settings.mode_api != "auto":
            return self.settings.mode_api

        model_lower = self.settings.model_name.lower()
        if any(token in model_lower for token in ("glm", "chatglm", "zhipu")):
            return "glm"
        if "kimi" in model_lower:
            return "kimi"
        if "doubao" in model_lower:
            return "doubao"
        if any(token in model_lower for token in ("step-3", "step3", "step")):
            return "stepfun"
        if "internvl" in model_lower:
            return "internvl"
        if "momo" in model_lower:
            return "momo"
        if "gemini" in model_lower:
            return "gemini"
        if "claude" in model_lower:
            return "anthropic"
        if "grok" in model_lower:
            return "responses"
        if any(token in model_lower for token in ("gpt", "o1", "o3", "o4")):
            return "openai"
        return "openai"

    def _thinking_type_from_mode(self) -> str:
        if self.settings.mode == "off":
            return "disabled"
        if self.settings.mode == "minimal":
            return "auto"
        return "enabled"

    def _thinking_type_for_kimi(self) -> str:
        return "disabled" if self.settings.mode in ("off", "minimal") else "enabled"

    def _thinking_type_for_glm(self) -> str:
        return "disabled" if self.settings.mode in ("off", "minimal") else "enabled"

    def _effort_for_doubao(self) -> Optional[str]:
        if self.settings.mode == "off":
            return None
        if self.settings.mode in ("thinking", "xhigh"):
            return "high"
        if self.settings.mode in ("minimal", "low", "medium", "high"):
            return self.settings.mode
        return "high"

    def _merge_nested_dict(self, base: Dict[str, Any], incoming: Dict[str, Any]) -> None:
        for key, value in incoming.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._merge_nested_dict(base[key], value)
            else:
                base[key] = value

    def _apply_mode_payload(self, payload: Dict[str, Any]) -> None:
        effort = self._mode_to_effort()
        if effort is None:
            if self.settings.extra_body:
                self._merge_nested_dict(payload, self.settings.extra_body)
            return

        mode_api = self._detect_mode_api()

        if mode_api == "openai":
            payload["reasoning_effort"] = effort
        elif mode_api == "responses":
            payload["reasoning"] = {"effort": effort}
        elif mode_api == "anthropic":
            if self.settings.mode not in ("off", "minimal"):
                budget_by_mode = {
                    "minimal": 1024,
                    "low": 2048,
                    "medium": 4096,
                    "high": 8192,
                    "xhigh": 16384,
                    "thinking": 8192,
                }
                budget = self.settings.thinking_budget_tokens
                if budget < 0:
                    budget = budget_by_mode.get(self.settings.mode, 4096)
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": int(max(1024, budget)),
                }
        elif mode_api == "gemini":
            payload["reasoning_effort"] = effort
            thinking_config: Dict[str, Any] = {}
            if self.settings.mode in ("minimal", "low", "medium", "high", "xhigh", "thinking"):
                level = "high" if self.settings.mode in ("xhigh", "thinking") else self.settings.mode
                thinking_config["thinking_level"] = level
            if self.settings.thinking_budget_tokens >= 0:
                thinking_config["thinking_budget"] = int(self.settings.thinking_budget_tokens)
            if self.settings.include_thoughts:
                thinking_config["include_thoughts"] = True
            if thinking_config:
                payload.setdefault("extra_body", {})
                payload["extra_body"].setdefault("google", {})
                payload["extra_body"]["google"]["thinking_config"] = thinking_config
        elif mode_api == "kimi":
            payload["thinking"] = {"type": self._thinking_type_for_kimi()}
            payload.pop("reasoning_effort", None)
        elif mode_api == "doubao":
            payload["thinking"] = {"type": self._thinking_type_from_mode()}
            db_effort = self._effort_for_doubao()
            if db_effort is None:
                payload.pop("reasoning_effort", None)
            else:
                payload["reasoning_effort"] = db_effort
        elif mode_api == "glm":
            payload["thinking"] = {"type": self._thinking_type_for_glm()}
            payload.pop("reasoning_effort", None)
        elif mode_api in ("stepfun", "internvl", "momo"):
            payload["thinking"] = {"type": self._thinking_type_from_mode()}
            payload.pop("reasoning_effort", None)

        if self.settings.extra_body:
            self._merge_nested_dict(payload, self.settings.extra_body)

    def _ensure_anthropic_thinking_constraints(self, payload: Dict[str, Any]) -> None:
        thinking = payload.get("thinking")
        if not isinstance(thinking, dict):
            return
        if thinking.get("type") != "enabled":
            return

        budget = int(max(1024, thinking.get("budget_tokens", 1024)))
        max_tokens = payload.get("max_tokens")

        if max_tokens is None:
            max_tokens = max(4096, budget + 1024)
        else:
            max_tokens = int(max_tokens)
            if max_tokens <= budget:
                max_tokens = budget + 1024

        payload["max_tokens"] = max_tokens
        payload.pop("max_completion_tokens", None)
        thinking["budget_tokens"] = min(budget, max_tokens - 1)
