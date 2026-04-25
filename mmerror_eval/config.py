from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .labels import normalize_task_mode
from .utils import load_env_file


@dataclass
class ProjectConfig:
    name: str
    description: str = ""


@dataclass
class DatasetConfig:
    json_dir: Path
    image_dir: Optional[Path]
    limit: int = -1


@dataclass
class EvaluationConfig:
    task_mode: str
    output_dir: Path
    concurrency: int = 8
    max_retries: int = 5
    request_interval_sec: float = 0.0
    timeout_sec: int = 600
    temperature: float = 0.0
    max_completion_tokens: int = 2048
    use_max_completion_tokens: bool = True
    resume: bool = True


@dataclass
class ProviderDefaults:
    provider_type: str = "openai_compatible"
    api_base: Optional[str] = None
    api_base_env: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    img_detail: str = "high"
    mode: str = "off"
    mode_api: str = "auto"
    thinking_budget_tokens: int = -1
    include_thoughts: bool = False
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str
    enabled: bool = True
    provider_type: Optional[str] = None
    api_base: Optional[str] = None
    api_base_env: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    img_detail: Optional[str] = None
    mode: Optional[str] = None
    mode_api: Optional[str] = None
    thinking_budget_tokens: Optional[int] = None
    include_thoughts: Optional[bool] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)
    strategy: Optional[str] = None
    constant_label: Optional[str] = None


@dataclass
class AppConfig:
    root_dir: Path
    config_path: Path
    project: ProjectConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    prompt_path: Path
    epd_presence_prompt_path: Path
    epd_type_prompt_path: Path
    provider_defaults: ProviderDefaults
    models: List[ModelConfig]


def _resolve_path(root_dir: Path, value: Optional[str]) -> Optional[Path]:
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (root_dir / path).resolve()
    return path


def _merge_env(env_file_values: Dict[str, str]) -> None:
    for key, value in env_file_values.items():
        if key not in os.environ:
            os.environ[key] = value


def _read_yaml(path: Path) -> Dict[str, Any]:
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError("Config file must contain a YAML object.")
    return content


def load_config(config_path: Path, env_file: Optional[Path] = None) -> AppConfig:
    resolved_config_path = config_path.resolve()
    root_dir = resolved_config_path.parent

    default_env_file = root_dir / ".env"
    chosen_env_file = env_file.resolve() if env_file else default_env_file
    _merge_env(load_env_file(chosen_env_file))

    raw = _read_yaml(resolved_config_path)

    project_raw = raw.get("project", {})
    dataset_raw = raw.get("dataset", {})
    evaluation_raw = raw.get("evaluation", {})
    prompt_raw = raw.get("prompt", {})
    provider_raw = raw.get("provider_defaults", {})
    models_raw = raw.get("models", [])

    if not models_raw:
        raise ValueError("Config must define at least one model in models.")

    task_mode = normalize_task_mode(str(evaluation_raw.get("task_mode", "epd")))

    project = ProjectConfig(
        name=str(project_raw.get("name", resolved_config_path.stem)),
        description=str(project_raw.get("description", "")),
    )
    dataset = DatasetConfig(
        json_dir=_resolve_path(root_dir, str(dataset_raw.get("json_dir", ""))) or Path(),
        image_dir=_resolve_path(root_dir, dataset_raw.get("image_dir")),
        limit=int(dataset_raw.get("limit", -1)),
    )
    evaluation = EvaluationConfig(
        task_mode=task_mode,
        output_dir=_resolve_path(root_dir, str(evaluation_raw.get("output_dir", "../result"))) or (root_dir.parent / "result"),
        concurrency=max(1, int(evaluation_raw.get("concurrency", 8))),
        max_retries=max(0, int(evaluation_raw.get("max_retries", 5))),
        request_interval_sec=float(evaluation_raw.get("request_interval_sec", 0.0)),
        timeout_sec=max(1, int(evaluation_raw.get("timeout_sec", 600))),
        temperature=float(evaluation_raw.get("temperature", 0.0)),
        max_completion_tokens=max(1, int(evaluation_raw.get("max_completion_tokens", 2048))),
        use_max_completion_tokens=bool(evaluation_raw.get("use_max_completion_tokens", True)),
        resume=bool(evaluation_raw.get("resume", True)),
    )
    prompt_path = _resolve_path(root_dir, str(prompt_raw.get("file", "prompts/error_label.prompt.txt")))
    if prompt_path is None:
        raise ValueError("prompt.file must be set.")
    epd_presence_prompt_path = _resolve_path(
        root_dir,
        str(prompt_raw.get("epd_presence_file", "prompts/epd_presence.prompt.txt")),
    )
    epd_type_prompt_path = _resolve_path(
        root_dir,
        str(prompt_raw.get("epd_type_file", "prompts/epd_type.prompt.txt")),
    )
    if epd_presence_prompt_path is None or epd_type_prompt_path is None:
        raise ValueError("prompt.epd_presence_file and prompt.epd_type_file must be set.")

    provider_defaults = ProviderDefaults(
        provider_type=str(provider_raw.get("type", "openai_compatible")),
        api_base=provider_raw.get("api_base"),
        api_base_env=provider_raw.get("api_base_env"),
        api_key=provider_raw.get("api_key"),
        api_key_env=provider_raw.get("api_key_env"),
        img_detail=str(provider_raw.get("img_detail", "high")),
        mode=str(provider_raw.get("mode", "off")),
        mode_api=str(provider_raw.get("mode_api", "auto")),
        thinking_budget_tokens=int(provider_raw.get("thinking_budget_tokens", -1)),
        include_thoughts=bool(provider_raw.get("include_thoughts", False)),
        extra_body=dict(provider_raw.get("extra_body", {}) or {}),
    )

    models: List[ModelConfig] = []
    for model_raw in models_raw:
        if not isinstance(model_raw, dict):
            raise ValueError("Each model entry must be a YAML object.")
        models.append(
            ModelConfig(
                name=str(model_raw["name"]),
                enabled=bool(model_raw.get("enabled", True)),
                provider_type=model_raw.get("type"),
                api_base=model_raw.get("api_base"),
                api_base_env=model_raw.get("api_base_env"),
                api_key=model_raw.get("api_key"),
                api_key_env=model_raw.get("api_key_env"),
                img_detail=model_raw.get("img_detail"),
                mode=model_raw.get("mode"),
                mode_api=model_raw.get("mode_api"),
                thinking_budget_tokens=model_raw.get("thinking_budget_tokens"),
                include_thoughts=model_raw.get("include_thoughts"),
                extra_body=dict(model_raw.get("extra_body", {}) or {}),
                strategy=model_raw.get("strategy"),
                constant_label=model_raw.get("constant_label"),
            )
        )

    return AppConfig(
        root_dir=root_dir,
        config_path=resolved_config_path,
        project=project,
        dataset=dataset,
        evaluation=evaluation,
        prompt_path=prompt_path,
        epd_presence_prompt_path=epd_presence_prompt_path,
        epd_type_prompt_path=epd_type_prompt_path,
        provider_defaults=provider_defaults,
        models=models,
    )
