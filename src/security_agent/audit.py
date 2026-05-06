from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any


def new_trace_id() -> str:
    return f"tr_{uuid.uuid4().hex[:16]}"


def now_ms() -> int:
    return int(time.time() * 1000)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


class AuditLogger:
    """Append-only JSONL audit trail for all guard, model and tool events."""

    def __init__(self, log_dir: Path, service_name: str = "agent") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{service_name}.jsonl"
        self._lock = Lock()

    def write(self, **event: Any) -> dict[str, Any]:
        record = {"ts_ms": now_ms(), **event}
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return record

    def read_trace(self, trace_id: str) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("trace_id") == trace_id:
                    events.append(item)
        return events
