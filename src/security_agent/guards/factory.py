from __future__ import annotations

from typing import Any

from security_agent.guards.base import BaseGuard
from security_agent.guards.caif_guard import CAIFGuard
from security_agent.guards.llama_guard import LlamaGuard
from security_agent.guards.nemo_guard import NemoGuard
from security_agent.guards.raw import RawGuard


def make_guard(mode: str, **kwargs: Any) -> BaseGuard:
    normalized = (mode or "raw").strip().lower()
    if normalized == "raw":
        return RawGuard(**kwargs)
    if normalized == "nemo":
        return NemoGuard(**kwargs)
    if normalized in {"llama_guard", "llamaguard", "llama"}:
        return LlamaGuard(**kwargs)
    if normalized in {"caif", "caif_guard", "caif-guard"}:
        return CAIFGuard(**kwargs)
    raise ValueError(f"Unknown guard mode: {mode}")
