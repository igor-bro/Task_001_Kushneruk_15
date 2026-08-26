"""Демо захисних механізмів (крок 5) без зайвих LLM-викликів."""

from __future__ import annotations

import sys

from safety import LoopDetector, SafetyLimits


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== LoopDetector ===")
    det = LoopDetector(max_repeats=3)
    for i in range(1, 4):
        looped = det.check("device_status", {"device_id": "hvac_living"})
        print(f"виклик #{i}: loop={looped}")

    print("\n=== max_steps / timeout ===")
    limits = SafetyLimits(max_steps=2, timeout_seconds=0.0)
    print("max_steps:", limits.check_limits(2))
    limits.reset()
    limits.timeout_seconds = 0.0
    # після reset start свіжий — timeout 0 спрацює одразу
    import time

    time.sleep(0.01)
    limits.timeout_seconds = 0.0
    print("timeout:", limits.check_limits(0))
    print("OK safety demo")


if __name__ == "__main__":
    main()
