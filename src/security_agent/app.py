from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from security_agent.agent import SecurityAgent
from security_agent.audit import AuditLogger, new_trace_id
from security_agent.guards.factory import make_guard
from security_agent.ollama_client import OllamaClient
from security_agent.settings import settings
from security_agent.tools.executor import ToolExecutor


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = False
    metadata: dict[str, Any] | None = None
    user: str | None = None


class AgentChatRequest(BaseModel):
    message: str
    model: str | None = None
    trace_id: str | None = None
    test_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CAIFInputCheckRequest(BaseModel):
    text: str
    trace_id: str | None = None
    model: str | None = None


class CAIFToolCallCheckRequest(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    model: str | None = None


class CAIFToolResultCheckRequest(BaseModel):
    tool_name: str
    result_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    model: str | None = None


class CAIFOutputCheckRequest(BaseModel):
    text: str
    trace_id: str | None = None
    model: str | None = None


audit = AuditLogger(settings.log_dir, service_name=f"agent_{settings.guard_mode}")
ollama = OllamaClient(settings.ollama_base_url, timeout_s=settings.request_timeout_s)
guard = make_guard(settings.guard_mode, audit=audit, ollama=ollama, settings=settings)
tools = ToolExecutor(workspace_dir=settings.workspace_dir, rag_dir=settings.rag_dir, audit=audit)
agent = SecurityAgent(
    guard=guard,
    tools=tools,
    ollama=ollama,
    audit=audit,
    guard_mode=settings.guard_mode,
    max_steps=settings.max_agent_steps,
    max_tokens=settings.max_tokens,
    temperature=settings.temperature,
)

app = FastAPI(title="Local AI Agent Security Benchmark + CAIF Gateway", version="0.15.0")


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "guard_mode": settings.guard_mode,
        "ollama_base_url": settings.ollama_base_url,
        "target_model": settings.target_model,
        "caif_policy_path": str(settings.caif_policy_path),
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    try:
        names = await ollama.list_models()
    except Exception:
        names = [settings.target_model]
    return {
        "object": "list",
        "data": [{"id": name, "object": "model", "owned_by": "ollama"} for name in names],
    }


@app.post("/admin/reset-workspace")
async def reset_workspace() -> dict[str, Any]:
    tools.reset_workspace()
    return {"ok": True, "workspace_dir": str(settings.workspace_dir)}


def _ensure_caif_gateway() -> None:
    if settings.guard_mode not in {"caif", "caif_guard", "caif-guard"}:
        raise HTTPException(status_code=400, detail="CAIF check endpoints require GUARD_MODE=caif")


def _decision_response(trace_id: str, stage: str, decision: Any) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "stage": stage,
        "guard_mode": settings.guard_mode,
        "allowed": decision.allowed,
        "reason": decision.reason,
        "risk_tags": decision.risk_tags,
        "cia": decision.cia,
        "replacement": decision.replacement,
        "metadata": decision.metadata,
        "decision": asdict(decision),
    }


@app.get("/caif/policy")
async def caif_policy() -> dict[str, Any]:
    _ensure_caif_gateway()
    policy = getattr(guard, "policy", None)
    return {
        "guard_mode": settings.guard_mode,
        "policy_path": str(settings.caif_policy_path),
        "policy": policy.summary() if policy and hasattr(policy, "summary") else None,
    }


@app.post("/caif/check/input")
async def caif_check_input(request: CAIFInputCheckRequest) -> dict[str, Any]:
    _ensure_caif_gateway()
    trace_id = request.trace_id or new_trace_id()
    decision = await guard.check_input(request.text, trace_id=trace_id, model=request.model or settings.target_model)
    return _decision_response(trace_id, "input", decision)


@app.post("/caif/check/tool-call")
async def caif_check_tool_call(request: CAIFToolCallCheckRequest) -> dict[str, Any]:
    _ensure_caif_gateway()
    trace_id = request.trace_id or new_trace_id()
    decision = await guard.check_tool_call(
        request.tool_name,
        request.args,
        trace_id=trace_id,
        model=request.model or settings.target_model,
    )
    return _decision_response(trace_id, "tool_call", decision)


@app.post("/caif/check/tool-result")
async def caif_check_tool_result(request: CAIFToolResultCheckRequest) -> dict[str, Any]:
    _ensure_caif_gateway()
    trace_id = request.trace_id or new_trace_id()
    decision = await guard.check_tool_result(
        request.tool_name,
        request.result_text,
        trace_id=trace_id,
        model=request.model or settings.target_model,
        metadata=request.metadata,
    )
    return _decision_response(trace_id, "tool_result", decision)


@app.post("/caif/check/output")
async def caif_check_output(request: CAIFOutputCheckRequest) -> dict[str, Any]:
    _ensure_caif_gateway()
    trace_id = request.trace_id or new_trace_id()
    decision = await guard.check_output(request.text, trace_id=trace_id, model=request.model or settings.target_model)
    return _decision_response(trace_id, "output", decision)


@app.post("/agent/chat")
async def agent_chat(request: AgentChatRequest) -> dict[str, Any]:
    trace_id = request.trace_id or new_trace_id()
    try:
        result = await agent.run(
            user_message=request.message,
            model=request.model or settings.target_model,
            trace_id=trace_id,
            test_id=request.test_id,
            metadata=request.metadata,
        )
        return {
            "trace_id": result.trace_id,
            "answer": result.answer,
            "blocked": result.blocked,
            "guard_mode": result.guard_mode,
            "model": result.model,
            "latency_ms": result.latency_ms,
            "events": result.events,
        }
    except Exception as exc:
        audit.write(
            trace_id=trace_id,
            event_type="request_exception",
            guard_mode=settings.guard_mode,
            model=request.model or settings.target_model,
            error=repr(exc),
        )
        return {
            "trace_id": trace_id,
            "answer": "Agent request failed safely inside the local gateway. No tool action was completed.",
            "blocked": False,
            "guard_mode": settings.guard_mode,
            "model": request.model or settings.target_model,
            "latency_ms": 0,
            "events": audit.read_trace(trace_id),
            "error": repr(exc),
        }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported by this benchmark endpoint")
    user_messages = [m.content for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="at least one user message is required")
    user_message = user_messages[-1]
    trace_id = new_trace_id()
    try:
        result = await agent.run(
            user_message=user_message,
            model=request.model or settings.target_model,
            trace_id=trace_id,
            test_id=(request.metadata or {}).get("test_id") if request.metadata else None,
            metadata={"openai_compatible": True, **(request.metadata or {})},
            max_tokens=request.max_tokens,
        )
    except Exception as exc:
        audit.write(
            trace_id=trace_id,
            event_type="request_exception",
            guard_mode=settings.guard_mode,
            model=request.model or settings.target_model,
            error=repr(exc),
        )
        class _FallbackResult:
            answer = "Agent request failed safely inside the local gateway. No tool action was completed."
            blocked = False
            guard_mode = settings.guard_mode
        result = _FallbackResult()
    created = int(time.time())
    return {
        "id": f"chatcmpl-{trace_id}",
        "object": "chat.completion",
        "created": created,
        "model": request.model or settings.target_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "metadata": {"trace_id": trace_id, "blocked": result.blocked, "guard_mode": result.guard_mode},
    }
