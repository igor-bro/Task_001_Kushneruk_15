"""Захисні механізми ReAct-агента (крок 5).

- max_steps — ліміт ітерацій agent-вузла
- timeout — загальний час виконання (time.monotonic)
- LoopDetector — повтор одного й того ж tool+args N разів поспіль
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional


class LoopDetector:
    """Детекція зациклення: однаковий tool call N разів поспіль."""

    def __init__(self, max_repeats: int = 3) -> None:
        self.max_repeats = max_repeats
        self.recent_calls: list[str] = []

    def reset(self) -> None:
        self.recent_calls.clear()

    @staticmethod
    def _hash_call(tool_name: str, args: dict) -> str:
        payload = json.dumps(
            {"name": tool_name, "args": args},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def check(self, tool_name: str, args: dict | None = None) -> bool:
        """True, якщо виявлено зациклення."""
        call_hash = self._hash_call(tool_name, args or {})
        self.recent_calls.append(call_hash)
        if len(self.recent_calls) >= self.max_repeats:
            last_n = self.recent_calls[-self.max_repeats :]
            if len(set(last_n)) == 1:
                return True
        return False


@dataclass
class SafetyLimits:
    """Ліміти одного прогону агента."""

    max_steps: int = 10
    timeout_seconds: float = 120.0
    max_repeats: int = 3
    loop_detector: LoopDetector = field(default_factory=lambda: LoopDetector(3))
    start_monotonic: float = field(default_factory=time.monotonic)
    stop_reason: Optional[str] = None

    def __post_init__(self) -> None:
        self.loop_detector = LoopDetector(self.max_repeats)
        self.reset()

    def reset(self) -> None:
        self.start_monotonic = time.monotonic()
        self.loop_detector.reset()
        self.stop_reason = None

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_monotonic

    def check_limits(self, step_count: int) -> Optional[str]:
        if step_count >= self.max_steps:
            return (
                f"Досягнуто ліміт кроків ({self.max_steps}). "
                "Завершую з наявними даними."
            )
        if self.elapsed_seconds() > self.timeout_seconds:
            return (
                f"Перевищено таймаут ({self.timeout_seconds:.0f} с). "
                "Завершую з наявними даними."
            )
        return None

    def check_tool_loop(self, tool_name: str, args: dict | None = None) -> Optional[str]:
        if self.loop_detector.check(tool_name, args):
            return (
                f"Виявлено повторний виклик {tool_name} "
                f"({self.max_repeats} однакові виклики поспіль)."
            )
        return None
