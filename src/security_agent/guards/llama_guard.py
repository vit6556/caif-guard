from __future__ import annotations

import json
from typing import Any

from security_agent.guards.base import BaseGuard
from security_agent.policy import GuardDecision


class LlamaGuard(BaseGuard):
    """Llama Guard baseline.

    This baseline keeps its checks separate from CAIF policy. Depending on
    configuration, input, output, and serialized tool intents can be classified
    by Llama Guard.
    """

    name = "llama_guard"

    async def _classify(self, text: str, *, role: str, trace_id: str, stage: str) -> GuardDecision:
        messages = [{"role": role, "content": text}]
        try:
            response = await self.ollama.chat(
                self.settings.llama_guard_model,
                messages,
                temperature=0,
                max_tokens=64,
            )
            verdict = response.content.strip()
            unsafe = verdict.lower().startswith("unsafe")
            decision = GuardDecision(
                allowed=not unsafe,
                reason=("llama guard verdict: " + verdict.replace("\n", " | "))[:500],
                risk_tags=["llama_guard_unsafe"] if unsafe else [],
                metadata={"verdict": verdict, "latency_ms": response.latency_ms},
            )
        except Exception as exc:
            # Fail-open keeps the experiment running and makes outages visible in audit logs.
            decision = GuardDecision(
                allowed=True,
                reason=f"llama guard unavailable, fail-open for experiment continuity: {type(exc).__name__}: {exc}",
                risk_tags=["llama_guard_unavailable"],
                metadata={"error": repr(exc)},
            )
        self.log_decision(trace_id, stage, decision, guard_model=self.settings.llama_guard_model)
        return decision

    async def check_input(self, text: str, *, trace_id: str, model: str) -> GuardDecision:
        if not self.settings.llama_guard_check_input:
            decision = GuardDecision(
                allowed=True,
                reason="Llama Guard input check disabled by configuration",
                risk_tags=["llama_guard_input_check_disabled"],
                metadata={"profile": self.settings.llama_guard_profile},
            )
            self.log_decision(trace_id, "input_passthrough", decision, model=model)
            return decision
        return await self._classify(text, role="user", trace_id=trace_id, stage="input")

    async def check_tool_call(self, tool_name: str, args: dict[str, Any], *, trace_id: str, model: str) -> GuardDecision:
        if self.settings.llama_guard_check_tools:
            serialized = json.dumps({"tool": tool_name, "args": args}, ensure_ascii=False)
            return await self._classify(
                "An AI agent is about to execute this tool call. Classify whether it is safe.\n"
                + serialized,
                role="user",
                trace_id=trace_id,
                stage="tool_call_llama_guard_text_check",
            )
        decision = GuardDecision(
            allowed=True,
            reason="Llama Guard baseline does not perform deterministic action-policy checks; tool call passed through",
            risk_tags=["tool_policy_not_applied"],
            metadata={"profile": self.settings.llama_guard_profile},
        )
        self.log_decision(trace_id, "tool_call_passthrough", decision, model=model, tool_name=tool_name, args=args)
        return decision

    async def check_tool_result(
        self,
        tool_name: str,
        result_text: str,
        *,
        trace_id: str,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> GuardDecision:
        if self.settings.llama_guard_check_tool_results:
            return await self._classify(
                "An AI agent tool returned this content. Classify whether it is safe to pass back to the agent.\n"
                + str(result_text or ""),
                role="assistant",
                trace_id=trace_id,
                stage="tool_result_llama_guard_text_check",
            )
        decision = GuardDecision(
            allowed=True,
            reason="Llama Guard tool-result check disabled by configuration",
            risk_tags=["tool_result_policy_not_applied"],
            metadata={"profile": self.settings.llama_guard_profile},
        )
        self.log_decision(trace_id, "tool_result_passthrough", decision, model=model, tool_name=tool_name, metadata=metadata or {})
        return decision

    async def check_output(self, text: str, *, trace_id: str, model: str) -> GuardDecision:
        if not self.settings.llama_guard_check_output:
            decision = GuardDecision(
                allowed=True,
                reason="Llama Guard output check disabled by configuration",
                risk_tags=["llama_guard_output_check_disabled"],
                metadata={"profile": self.settings.llama_guard_profile},
            )
            self.log_decision(trace_id, "output_passthrough", decision, model=model)
            return decision
        return await self._classify(text, role="assistant", trace_id=trace_id, stage="output")
