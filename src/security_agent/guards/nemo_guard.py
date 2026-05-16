from __future__ import annotations

import asyncio
import json
from typing import Any

from security_agent.guards.base import BaseGuard
from security_agent.policy import GuardDecision


class NemoGuard(BaseGuard):
    """Baseline: NeMo Guardrails only (`LLMRails.check_async`).

    Behaviour is driven by `configs/nemo/` (rails flows + prompts). No extra heuristics
    beyond mapping tool proposals to input rails as plain user-role text for evaluation.
    """

    name = "nemo_guardrails"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._rails = None
        self._nemo_error: str | None = None
        if self.settings.nemo_use_library:
            self._init_nemo()

    def _init_nemo(self) -> None:
        try:
            from nemoguardrails import LLMRails, RailsConfig  # type: ignore

            config = RailsConfig.from_path(str(self.settings.nemo_config_dir))
            self._rails = LLMRails(config)
        except Exception as exc:
            self._rails = None
            self._nemo_error = f"{type(exc).__name__}: {exc}"

    async def _rails_check_async(
        self,
        messages: list[dict[str, str]],
        *,
        rail_types: list[Any],
        stage: str,
        trace_id: str,
    ) -> GuardDecision:
        """Run NeMo moderation rails via the documented check_async API."""

        if self._rails is None:
            decision = GuardDecision(
                allowed=self.settings.nemo_fail_open,
                reason=(
                    "NeMo library unavailable; fail-open for experiment continuity"
                    if self.settings.nemo_fail_open
                    else "NeMo library unavailable; fail-closed"
                ),
                risk_tags=["nemo_unavailable"],
                metadata={"error": self._nemo_error, "fail_open": self.settings.nemo_fail_open},
            )
            self.log_decision(trace_id, f"{stage}_nemo", decision)
            return decision

        try:
            # Import local to nemoguardrails install (evaluator Docker image).
            from nemoguardrails.rails.llm.options import RailStatus, RailType  # type: ignore

            result = await self._rails.check_async(messages, rail_types=rail_types)
            blocked = result.status == RailStatus.BLOCKED
            replaced = getattr(result, "content", "") or ""

            if blocked:
                decision = GuardDecision(
                    allowed=False,
                    reason=("NeMo Guardrails blocked: " + replaced.replace("\n", " | "))[:500],
                    risk_tags=["nemo_block"],
                    metadata={
                        "nemo_status": str(result.status),
                        "rail": getattr(result, "rail", None),
                        "stage": stage,
                    },
                )
            elif result.status == RailStatus.MODIFIED and stage == "output" and replaced:
                decision = GuardDecision(
                    allowed=True,
                    reason="NeMo Guardrails modified output content",
                    risk_tags=["nemo_modified"],
                    replacement=replaced,
                    metadata={"nemo_status": str(result.status), "stage": stage},
                )
            else:
                decision = GuardDecision(
                    allowed=True,
                    reason=("NeMo Guardrails allowed: " + str(result.status)),
                    metadata={"nemo_status": str(result.status), "stage": stage},
                )
        except Exception as exc:
            decision = GuardDecision(
                allowed=self.settings.nemo_fail_open,
                reason=f"NeMo check failed: {type(exc).__name__}: {exc}",
                risk_tags=["nemo_check_error"],
                metadata={"error": repr(exc), "fail_open": self.settings.nemo_fail_open},
            )
        self.log_decision(trace_id, f"{stage}_nemo", decision)
        return decision

    async def _nemo_check(self, text: str, *, stage: str, trace_id: str) -> GuardDecision:
        from nemoguardrails.rails.llm.options import RailType  # type: ignore

        if stage == "output":
            msgs = [{"role": "assistant", "content": text}]
            rails = [RailType.OUTPUT]
        else:
            msgs = [{"role": "user", "content": text}]
            rails = [RailType.INPUT]
        return await self._rails_check_async(msgs, rail_types=rails, stage=stage, trace_id=trace_id)

    async def check_input(self, text: str, *, trace_id: str, model: str) -> GuardDecision:
        return await self._nemo_check(text, stage="input", trace_id=trace_id)

    async def check_tool_call(self, tool_name: str, args: dict[str, Any], *, trace_id: str, model: str) -> GuardDecision:
        payload = json.dumps({"tool": tool_name, "args": args}, ensure_ascii=False)
        return await self._nemo_check(payload, stage="tool_call", trace_id=trace_id)

    async def check_tool_result(
        self,
        tool_name: str,
        result_text: str,
        *,
        trace_id: str,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> GuardDecision:
        decision = GuardDecision(
            allowed=True,
            reason="NeMo tool-result check disabled by configuration",
            risk_tags=["tool_result_policy_not_applied"],
        )
        self.log_decision(
            trace_id,
            "tool_result_passthrough",
            decision,
            model=model,
            tool_name=tool_name,
            metadata=metadata or {},
        )
        return decision

    async def check_output(self, text: str, *, trace_id: str, model: str) -> GuardDecision:
        return await self._nemo_check(text, stage="output", trace_id=trace_id)
