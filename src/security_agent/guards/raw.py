from __future__ import annotations

from typing import Any

from security_agent.guards.base import BaseGuard
from security_agent.policy import GuardDecision


class RawGuard(BaseGuard):
    name = "raw"

    async def check_input(self, text: str, *, trace_id: str, model: str) -> GuardDecision:
        decision = GuardDecision(allowed=True, reason="raw mode: no guardrail")
        self.log_decision(trace_id, "input", decision, model=model)
        return decision

    async def check_tool_call(self, tool_name: str, args: dict[str, Any], *, trace_id: str, model: str) -> GuardDecision:
        decision = GuardDecision(allowed=True, reason="raw mode: tool call is not blocked")
        self.log_decision(trace_id, "tool_call", decision, model=model, tool_name=tool_name, args=args)
        return decision

    async def check_tool_result(self, tool_name: str, result_text: str, *, trace_id: str, model: str, metadata: dict[str, Any] | None = None) -> GuardDecision:
        decision = GuardDecision(allowed=True, reason="raw mode: tool result is not sanitized")
        self.log_decision(trace_id, "tool_result", decision, model=model, tool_name=tool_name, metadata=metadata or {})
        return decision

    async def check_output(self, text: str, *, trace_id: str, model: str) -> GuardDecision:
        decision = GuardDecision(allowed=True, reason="raw mode: output is not filtered")
        self.log_decision(trace_id, "output", decision, model=model)
        return decision
