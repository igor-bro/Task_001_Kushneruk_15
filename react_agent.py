"""ReAct-агент Smart Home (кроки 3–5).

Цикл: START → agent → tools | END
Захист: max_steps=10, timeout=120 с, детекція повторів.
Лог: TrajectoryLogger → trajectory.json

Запуск:
    python react_agent.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from logger import TrajectoryLogger
from safety import SafetyLimits
from tools import SAFE_TOOLS, reset_house

load_dotenv()

SYSTEM_PROMPT = """\
Ти — асистент розумного дому (Home Assistant). Відповідай українською.

Доступні безпечні інструменти:
- sensor_read — температура / вологість / рух
- device_status — стан пристрою (on/off, settings)
- energy_consumption — споживання електроенергії
- schedule_list — заплановані автоматизації

Правила:
1. Для фактів про дім ЗАВЖДИ викликай відповідний tool (не вигадуй показники).
2. Не керуй пристроями в цьому режимі (device_control недоступний) — лише читай стан.
3. Якщо зима і холодно — рекомендуй увімкнути опалення, але не виконуй керування.
4. Відповідай стисло й по суті.
"""

DEFAULT_MAX_STEPS = 10
DEFAULT_TIMEOUT_SECONDS = 120.0
TRAJECTORY_PATH = Path(__file__).with_name("trajectory.json")


class ReactState(TypedDict):
    messages: Annotated[list, add_messages]
    step_count: int
    stop_reason: Optional[str]


def build_llm() -> ChatMistralAI:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key or api_key.startswith("your-"):
        raise RuntimeError("Немає MISTRAL_API_KEY у .env")
    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    return ChatMistralAI(model=model, temperature=0.1)


def _tool_calls_as_dicts(message: AIMessage) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for call in message.tool_calls or []:
        if isinstance(call, dict):
            result.append(
                {"name": call.get("name", "?"), "args": call.get("args") or {}}
            )
        else:
            result.append(
                {
                    "name": getattr(call, "name", "?"),
                    "args": getattr(call, "args", {}) or {},
                }
            )
    return result


def _content_preview(msg: Any, limit: int = 400) -> str:
    content = getattr(msg, "content", "") or ""
    return str(content)[:limit]


def build_react_graph(
    *,
    tools: list | None = None,
    system_prompt: str | None = None,
    safety: SafetyLimits | None = None,
    logger: TrajectoryLogger | None = None,
    checkpointer=None,
):
    """Збирає ReAct-граф із захистом і логуванням.

    tools / system_prompt — щоб той самий цикл можна було використати
    як вкладеного executor у Plan-and-Execute (з knowledge_search тощо).
    """
    agent_tools = list(tools) if tools is not None else list(SAFE_TOOLS)
    prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    limits = safety or SafetyLimits(
        max_steps=DEFAULT_MAX_STEPS,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    )
    traj = logger or TrajectoryLogger()
    llm = build_llm().bind_tools(agent_tools)

    def agent_node(state: ReactState) -> dict:
        step = int(state.get("step_count") or 0) + 1
        reason = limits.check_limits(step - 1)
        if reason:
            limits.stop_reason = reason
            msg = AIMessage(content=f"⚠️ {reason}")
            traj.log_step(
                step,
                "agent",
                "safety check",
                reason,
                extra={"stopped": True},
            )
            return {
                "messages": [msg],
                "step_count": step,
                "stop_reason": reason,
            }

        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=prompt)] + messages

        response = llm.invoke(messages)
        calls = _tool_calls_as_dicts(response)

        # Детекція повторів по кожному tool call
        for call in calls:
            loop_reason = limits.check_tool_loop(call["name"], call["args"])
            if loop_reason:
                limits.stop_reason = loop_reason
                msg = AIMessage(content=f"⚠️ {loop_reason} Завершую цикл.")
                traj.log_step(
                    step,
                    "agent",
                    _content_preview(messages[-1]),
                    loop_reason,
                    tool_calls=[c["name"] for c in calls],
                    extra={"stopped": True, "loop": True},
                )
                return {
                    "messages": [msg],
                    "step_count": step,
                    "stop_reason": loop_reason,
                }

        traj.log_step(
            step,
            "agent",
            _content_preview(messages[-1]),
            _content_preview(response),
            tool_calls=[c["name"] for c in calls],
        )
        return {
            "messages": [response],
            "step_count": step,
            "stop_reason": None,
        }

    def tools_node(state: ReactState) -> dict:
        """Обгортка ToolNode з логуванням."""
        node = ToolNode(agent_tools)
        result = node.invoke(state)
        tool_msgs = [
            m for m in result.get("messages", []) if isinstance(m, ToolMessage)
        ]
        names = [getattr(m, "name", "?") for m in tool_msgs]
        preview = " | ".join(_content_preview(m, 200) for m in tool_msgs) or "ok"
        traj.log_step(
            int(state.get("step_count") or 0),
            "tools",
            f"tools={names}",
            preview,
            tool_calls=names,
        )
        return result

    def route_after_agent(state: ReactState) -> Literal["tools", "__end__"]:
        if state.get("stop_reason"):
            return "__end__"
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    graph = StateGraph(ReactState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer), limits, traj


def invoke_react(
    query: str,
    *,
    tools: list | None = None,
    system_prompt: str | None = None,
    safety: SafetyLimits | None = None,
    config: dict | None = None,
    trajectory_path: Path | str | None = None,
    reset_house_state: bool = False,
) -> dict:
    """Один повний ReAct-цикл (agent ⇄ tools) для підзадачі.

    Використовується і в demo, і як вкладений executor у Plan-and-Execute.
    За замовчуванням НЕ скидає mock-дім — щоб кроки P&E бачили спільний стан.
    """
    if reset_house_state:
        reset_house()

    app, limits, traj = build_react_graph(
        tools=tools,
        system_prompt=system_prompt,
        safety=safety,
    )
    limits.reset()
    traj.reset()
    traj.meta["query"] = query

    result = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "step_count": 0,
            "stop_reason": None,
        },
        config=config,
    )
    traj.meta["stop_reason"] = result.get("stop_reason") or limits.stop_reason
    traj.meta["summary"] = traj.summary()
    if trajectory_path:
        traj.save(str(trajectory_path))
    return result


def run_query(
    query: str,
    *,
    config: dict | None = None,
    trajectory_path: Path | str | None = TRAJECTORY_PATH,
) -> dict:
    return invoke_react(
        query,
        config=config,
        trajectory_path=trajectory_path,
        reset_house_state=True,
    )


DEMO_QUERIES = [
    "Яка температура в вітальні і на вулиці?",
    "Чи увімкнено опалення hvac_living і які в нього налаштування?",
    "Скільки електроенергії витрачено сьогодні і які автоматизації у living?",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for i, query in enumerate(DEMO_QUERIES, start=1):
        print(f"\n{'=' * 60}")
        print(f"[USER {i}] {query}")
        path = Path(__file__).with_name(f"trajectory_{i}.json")
        if i == len(DEMO_QUERIES):
            path = TRAJECTORY_PATH
        result = run_query(query, trajectory_path=path)
        final = result["messages"][-1]
        print(f"[ASSISTANT] {_content_preview(final, 800)}")
        tools = [
            m.name
            for m in result["messages"]
            if isinstance(m, ToolMessage)
        ]
        print(f"[tools] {tools or '(немає)'}")
        print(f"[steps] {result.get('step_count')}  stop={result.get('stop_reason')}")
        print(f"[log] {path.name}")


if __name__ == "__main__":
    main()
