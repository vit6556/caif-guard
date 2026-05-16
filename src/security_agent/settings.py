from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "/workspace/configs")).resolve()
DEFAULT_RUNTIME_PATH = DEFAULT_CONFIG_DIR / "runtime.yaml"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


@lru_cache(maxsize=4)
def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _runtime(*keys: str, default: Any = None) -> Any:
    data: Any = _load_yaml(str(DEFAULT_RUNTIME_PATH))
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data



@dataclass(frozen=True)
class Settings:
    # Only the gateway mode is selected by Docker Compose per service. Shared
    # runtime defaults are loaded from configs/runtime.yaml.
    guard_mode: str = os.getenv("GUARD_MODE", "raw").strip().lower()

    ollama_base_url: str = str(_runtime("ollama_base_url", default="http://host.docker.internal:11434")).rstrip("/")
    target_model: str = str(_runtime("target_model", default="llama3.2:3b"))  # agent model; guard models configured separately
    llama_guard_model: str = str(_runtime("guard_models", "llama_guard", default="llama3.2:3b"))
    nemo_guard_model: str = str(_runtime("guard_models", "nemo_self_check", default="llama3.2:3b"))

    log_dir: Path = Path(str(_runtime("paths", "log_dir", default="/workspace/reports/agent_logs"))).resolve()
    workspace_dir: Path = Path(str(_runtime("paths", "workspace_dir", default="/workspace/data/workspace"))).resolve()
    rag_dir: Path = Path(str(_runtime("paths", "rag_dir", default="/workspace/data/rag"))).resolve()
    caif_policy_path: Path = Path(str(_runtime("paths", "caif_policy_path", default="/workspace/configs/policies/caif_policy.yaml"))).resolve()

    max_agent_steps: int = _as_int(_runtime("limits", "max_agent_steps", default=3), 3)
    max_tokens: int = _as_int(_runtime("limits", "max_tokens", default=512), 512)
    temperature: float = _as_float(_runtime("limits", "temperature", default=0), 0.0)
    request_timeout_s: float = _as_float(_runtime("limits", "request_timeout_s", default=120), 120.0)

    nemo_use_library: bool = _as_bool(_runtime("nemo", "use_library", default=True), True)
    nemo_fail_open: bool = _as_bool(_runtime("nemo", "fail_open", default=True), True)
    nemo_config_dir: Path = Path(str(_runtime("nemo", "config_dir", default="/workspace/configs/nemo"))).resolve()

    llama_guard_profile: str = str(_runtime("llama_guard", "profile", default="input_output_tools"))
    llama_guard_check_input: bool = _as_bool(_runtime("llama_guard", "check_input", default=True), True)
    llama_guard_check_output: bool = _as_bool(_runtime("llama_guard", "check_output", default=True), True)
    llama_guard_check_tools: bool = _as_bool(_runtime("llama_guard", "check_tools", default=True), True)
    llama_guard_check_tool_results: bool = _as_bool(_runtime("llama_guard", "check_tool_results", default=False), False)


settings = Settings()
