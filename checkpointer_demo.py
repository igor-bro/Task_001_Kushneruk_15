"""Демо SqliteSaver (крок 7): пам'ять між запитами + get_state."""

from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from react_agent import build_react_graph
from tools import reset_house


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    reset_house()
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        app, _, _ = build_react_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "home-session-001"}}

        print("=== Запит 1 ===")
        r1 = app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Запам'ятай: я люблю температуру у вітальні 22°C."
                    )
                ],
                "step_count": 0,
                "stop_reason": None,
            },
            config=config,
        )
        print(r1["messages"][-1].content)

        print("\n=== Запит 2 (той самий thread) ===")
        r2 = app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Яку температуру я люблю у вітальні? "
                        "Також прочитай living_temp."
                    )
                ],
                "step_count": 0,
                "stop_reason": None,
            },
            config=config,
        )
        print(r2["messages"][-1].content)

        snap = app.get_state(config)
        print("\n=== get_state ===")
        print("messages:", len(snap.values.get("messages", [])))
        print("config:", snap.config)
        print("OK checkpointer demo")


if __name__ == "__main__":
    main()
