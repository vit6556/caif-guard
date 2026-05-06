from __future__ import annotations

from dataclasses import asdict
from typing import Any

from security_agent.audit import AuditLogger
from security_agent.ollama_client import OllamaClient
from security_agent.policy import GuardDecision


class BaseGuard:
    name = "base"

    def __init__(self, *, audit: AuditLogger, ollama: OllamaClient, settings: Any) -> None:
        self.audit = audit
        self.ollama = ollama
        self.settings = settings

    async def check_input(self, text: str, *, trace_id: str, model: str) -> GuardDecision:
        return GuardDecision(allowed=True)

    async def check_tool_call(self, tool_name: str, args: dict[str, Any], *, trace_id: str, model: str) -> GuardDecision:
        return GuardDecision(allowed=True)

    async def check_tool_result(
        self,
        tool_name: str,
        result_text: str,
        *,
        trace_id: str,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> GuardDecision:
        return GuardDecision(allowed=True)

    async def check_output(self, text: str, *, trace_id: str, model: str) -> GuardDecision:
        return GuardDecision(allowed=True)

    def log_decision(self, trace_id: str, stage: str, decision: GuardDecision, **extra: Any) -> None:
        self.audit.write(
            trace_id=trace_id,
            event_type="guard_decision",
            guard=self.name,
            stage=stage,
            decision=asdict(decision),
            **extra,
        )
