"""JSON-лог траєкторії агента (крок 4)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any


class TrajectoryLogger:
    """Append-only лог кроків для post-mortem аналізу."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.start_time = time.monotonic()
        self.last_step_time = self.start_time
        self.meta: dict[str, Any] = {}

    def reset(self) -> None:
        self.steps.clear()
        self.start_time = time.monotonic()
        self.last_step_time = self.start_time
        self.meta = {}

    def log_step(
        self,
        step_num: int,
        node: str,
        input_data: str,
        output_data: str,
        tool_calls: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        now = time.monotonic()
        elapsed_ms = int((now - self.start_time) * 1000)
        duration_ms = int((now - self.last_step_time) * 1000)
        self.last_step_time = now

        entry: dict[str, Any] = {
            "step": step_num,
            "node": node,
            "input": str(input_data)[:500],
            "output": str(output_data)[:500],
            "tool_calls": tool_calls or [],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "elapsed_ms": elapsed_ms,
            "duration_ms": duration_ms,
            "duration_sec": round(duration_ms / 1000.0, 3),
        }
        if extra:
            entry["extra"] = extra
        self.steps.append(entry)

    def save(self, filepath: str) -> str:
        payload = {
            "total_steps": len(self.steps),
            "total_time_ms": int((time.monotonic() - self.start_time) * 1000),
            "meta": self.meta,
            "trajectory": self.steps,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return filepath

    def summary(self) -> dict[str, Any]:
        tools_used = [
            name for step in self.steps for name in (step.get("tool_calls") or [])
        ]
        return {
            "total_steps": len(self.steps),
            "total_time_ms": int((time.monotonic() - self.start_time) * 1000),
            "tools_used": tools_used,
            "unique_tools": sorted(set(tools_used)),
        }
