"""HITL для device_control (крок 9): interrupt_before=['risky_tools'].

Команди:
    python hitl.py start
    python hitl.py approve
    python hitl.py reject
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tools import LEDGER_PATH, SAFE_TOOLS, device_control, reset_house

load_dotenv()

HITL_DB = Path(__file__).with_name("hitl_checkpoints.db")
HITL_TOOLS = SAFE_TOOLS + [device_control]
SYSTEM = """\
Ти Smart Home асистент. Відповідай українською.
Якщо користувач просить УВІМКНУТИ/ВИМКНУТИ пристрій — виклич device_control.
Для читання стану використовуй sensor_read / device_status.
"""


def build_llm():
    import os

    key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not key or key.startswith("your-"):
        raise RuntimeError("Немає MISTRAL_API_KEY")
    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    return ChatMistralAI(model=model, temperature=0.1)


def is_risky_tool(state: MessagesState) -> str:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            name = tc["name"] if isinstance(tc, dict) else tc.get("name", "")
            if name == "device_control":
                return "risky"
        return "safe_tools"
    return END


def build_hitl_graph():
    llm = build_llm().bind_tools(HITL_TOOLS)

    def agent_node(state: MessagesState) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM)] + list(messages)
        return {"messages": [llm.invoke(messages)]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("safe_tools", ToolNode(SAFE_TOOLS))
    g.add_node("risky_tools", ToolNode([device_control]))
    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent",
        is_risky_tool,
        {"safe_tools": "safe_tools", "risky": "risky_tools", END: END},
    )
    g.add_edge("safe_tools", "agent")
    g.add_edge("risky_tools", "agent")
    return g


def _thread() -> dict:
    # Фіксований thread для resume між процесами в одному демо-сеансі
    path = Path(__file__).with_name("hitl_thread.txt")
    if path.exists():
        tid = path.read_text(encoding="utf-8").strip()
    else:
        tid = f"hitl-{uuid.uuid4().hex[:8]}"
        path.write_text(tid, encoding="utf-8")
    return {"configurable": {"thread_id": tid}}


def cmd_start() -> None:
    reset_house()
    LEDGER_PATH.unlink(missing_ok=True)
    HITL_DB.unlink(missing_ok=True)
    Path(__file__).with_name("hitl_thread.txt").unlink(missing_ok=True)

    graph = build_hitl_graph()
    with SqliteSaver.from_conn_string(str(HITL_DB)) as cp:
        app = graph.compile(checkpointer=cp, interrupt_before=["risky_tools"])
        config = _thread()
        app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Увімкни опалення hvac_living на 22°C — "
                            "я скоро повернуся додому."
                        )
                    )
                ]
            },
            config=config,
        )
        snap = app.get_state(config)
        print("[interrupt] next =", snap.next)
        pending = snap.values["messages"][-1]
        if getattr(pending, "tool_calls", None):
            for tc in pending.tool_calls:
                print("[tool]", tc["name"], json.dumps(tc["args"], ensure_ascii=False))
        print("[ledger]", "порожньо" if not LEDGER_PATH.exists() else "є запис")


def cmd_approve() -> None:
    graph = build_hitl_graph()
    with SqliteSaver.from_conn_string(str(HITL_DB)) as cp:
        app = graph.compile(checkpointer=cp, interrupt_before=["risky_tools"])
        config = _thread()
        result = app.invoke(None, config=config)
        print("[approve]", result["messages"][-1].content)
        if LEDGER_PATH.exists():
            print("[ledger]", LEDGER_PATH.read_text(encoding="utf-8"))


def cmd_reject() -> None:
    graph = build_hitl_graph()
    with SqliteSaver.from_conn_string(str(HITL_DB)) as cp:
        app = graph.compile(checkpointer=cp, interrupt_before=["risky_tools"])
        config = _thread()
        snap = app.get_state(config)
        pending = snap.values["messages"][-1]
        tc_id = pending.tool_calls[0]["id"]
        app.update_state(
            config,
            {
                "messages": [
                    ToolMessage(
                        content="Операцію відхилено оператором.",
                        tool_call_id=tc_id,
                    )
                ]
            },
            as_node="risky_tools",
        )
        result = app.invoke(None, config=config)
        print("[reject]", result["messages"][-1].content)
        print(
            "[ledger]",
            "порожньо (tool не виконано)"
            if not LEDGER_PATH.exists()
            else LEDGER_PATH.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    if cmd == "start":
        cmd_start()
    elif cmd == "approve":
        cmd_approve()
    elif cmd == "reject":
        cmd_reject()
    else:
        raise SystemExit(f"Невідома команда: {cmd}")
