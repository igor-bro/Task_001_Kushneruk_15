"""Plan-and-Execute для Smart Home (крок 6).

Граф: START → planner → executor → replanner ⇄ executor | END

Запуск:
    python plan_execute.py
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from knowledge import knowledge_search
from models import Plan, PlanStep, ReplanDecision
from tools import SAFE_TOOLS, TOOLS_BY_NAME, reset_house

load_dotenv()

EXECUTOR_TOOLS = SAFE_TOOLS + [knowledge_search]
TOOLS_MAP = {**TOOLS_BY_NAME, knowledge_search.name: knowledge_search}

SYSTEM_PLANNER = """\
Ти planner для Smart Home. Склади план 3–6 кроків.
Кожен крок: step_id, description, expected_tool
(sensor_read | device_status | energy_consumption | schedule_list |
knowledge_search | null).
НЕ плануй device_control — лише збір стану і рекомендації.
"""


class PlanExecuteState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    plan: list[dict[str, Any]]
    completed_steps: list[str]
    current_step_idx: int
    response: str
    replan_count: int


def build_llm() -> ChatMistralAI:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key or api_key.startswith("your-"):
        raise RuntimeError("Немає MISTRAL_API_KEY у .env")
    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    return ChatMistralAI(model=model, temperature=0.1)


def _steps_to_dicts(steps: list[PlanStep]) -> list[dict[str, Any]]:
    return [
        {
            "step_id": s.step_id,
            "description": s.description,
            "expected_tool": s.expected_tool,
        }
        for s in steps
    ]


def _format_plan(plan: list[dict[str, Any]]) -> str:
    lines = []
    for s in plan:
        tool = s.get("expected_tool") or "none"
        lines.append(f"{s.get('step_id', '?')}. [{tool}] {s.get('description')}")
    return "\n".join(lines)


def _tool_calls(message: AIMessage) -> list[dict[str, Any]]:
    out = []
    for call in message.tool_calls or []:
        if isinstance(call, dict):
            out.append({"name": call.get("name", "?"), "args": call.get("args") or {}})
        else:
            out.append(
                {
                    "name": getattr(call, "name", "?"),
                    "args": getattr(call, "args", {}) or {},
                }
            )
    return out


def build_pe_graph(checkpointer=None):
    llm = build_llm()
    planner_llm = llm.with_structured_output(Plan)
    replanner_llm = llm.with_structured_output(ReplanDecision)
    executor_llm = llm.bind_tools(EXECUTOR_TOOLS)

    def planner_node(state: PlanExecuteState) -> dict:
        plan_obj = None
        for _ in range(3):
            plan_obj = planner_llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PLANNER),
                    HumanMessage(
                        content=(
                            f"Задача: {state['task']}\n"
                            "Склади план підготовки дому (читання стану + рекомендації)."
                        )
                    ),
                ]
            )
            if plan_obj and plan_obj.steps:
                break
        if not plan_obj or not plan_obj.steps:
            steps = [
                {
                    "step_id": 1,
                    "description": "Прочитати температуру у вітальні",
                    "expected_tool": "sensor_read",
                },
                {
                    "step_id": 2,
                    "description": "Перевірити статус HVAC",
                    "expected_tool": "device_status",
                },
                {
                    "step_id": 3,
                    "description": "Знайти рекомендації для зими",
                    "expected_tool": "knowledge_search",
                },
            ]
            return {
                "plan": steps,
                "current_step_idx": 0,
                "completed_steps": [],
                "replan_count": 0,
                "messages": [
                    AIMessage(content="[PLAN fallback]\n" + _format_plan(steps))
                ],
            }
        steps = _steps_to_dicts(plan_obj.steps)
        return {
            "plan": steps,
            "current_step_idx": 0,
            "completed_steps": [],
            "replan_count": 0,
            "messages": [
                AIMessage(
                    content=f"[PLAN]\ngoal: {plan_obj.goal}\n{_format_plan(steps)}"
                )
            ],
        }

    def executor_node(state: PlanExecuteState) -> dict:
        idx = state["current_step_idx"]
        plan = state["plan"]
        if idx >= len(plan):
            return {"response": "Усі кроки плану виконано."}

        step = plan[idx]
        desc = step.get("description", "")
        expected = step.get("expected_tool")
        context = ""
        if state["completed_steps"]:
            context = "\nПопередні результати:\n" + "\n".join(
                f"- {r}" for r in state["completed_steps"]
            )

        prompt = (
            "Ти executor Smart Home. Виконай РІВНО один крок.\n"
            f"expected_tool: {expected}\n"
            f"Крок {idx + 1}/{len(plan)}: {desc}\n"
            f"{context}\n"
            "Якщо expected_tool заданий — виклич саме його. "
            "device_control недоступний."
        )
        response = executor_llm.invoke([HumanMessage(content=prompt)])
        calls = _tool_calls(response)

        if expected and expected in TOOLS_MAP:
            matching = [c for c in calls if c["name"] == expected]
            if not matching:
                defaults = {
                    "sensor_read": {
                        "device_id": "living_temp",
                        "sensor_type": "temperature",
                    },
                    "device_status": {"device_id": "hvac_living"},
                    "energy_consumption": {"period": "today"},
                    "schedule_list": {"room": "living"},
                    "knowledge_search": {"query": desc},
                }
                calls = [{"name": expected, "args": defaults.get(expected, {})}]
            else:
                calls = matching

        if not calls:
            step_result = str(response.content).strip() or "Крок без tool."
        else:
            chunks = []
            for call in calls:
                fn = TOOLS_MAP.get(call["name"])
                if fn is None:
                    chunks.append(f"{call['name']}: невідомий tool")
                    continue
                try:
                    chunks.append(f"{call['name']}: {fn.invoke(call['args'] or {})}")
                except Exception as exc:  # noqa: BLE001
                    chunks.append(f"{call['name']}: помилка {exc}")
            step_result = "\n".join(chunks)

        completed = state["completed_steps"] + [f"Крок {idx + 1}: {step_result}"]
        return {
            "completed_steps": completed,
            "current_step_idx": idx + 1,
            "messages": [
                AIMessage(
                    content=(
                        f"[EXECUTOR] step={idx + 1} tool={expected}\n{step_result}"
                    )
                )
            ],
        }

    def replanner_node(state: PlanExecuteState) -> dict:
        remaining = state["plan"][state["current_step_idx"] :]
        completed_summary = "\n".join(state["completed_steps"]) or "(немає)"
        remaining_summary = _format_plan(remaining) if remaining else "(немає)"

        decision = None
        for _ in range(3):
            decision = replanner_llm.invoke(
                f"Завдання: {state['task']}\n\n"
                f"Виконано:\n{completed_summary}\n\n"
                f"Залишок:\n{remaining_summary}\n\n"
                "Виріши: finish (якщо достатньо даних для рекомендацій), "
                "continue або replan. Для finish обов'язково final_answer "
                "з конкретним планом дій українською."
            )
            if decision is not None:
                break

        if decision is None:
            if remaining:
                return {}
            return {"response": "Завдання виконано (fallback)."}

        if decision.action == "finish":
            answer = decision.final_answer or "Підготовка дому проаналізована."
            return {
                "response": answer,
                "messages": [
                    AIMessage(content=f"[REPLANNER finish] {decision.reasoning}")
                ],
            }
        if decision.action == "replan" and decision.updated_steps:
            new_plan = _steps_to_dicts(decision.updated_steps)
            return {
                "plan": new_plan,
                "current_step_idx": 0,
                "replan_count": state.get("replan_count", 0) + 1,
                "messages": [
                    AIMessage(
                        content=(
                            f"[REPLANNER replan] {decision.reasoning}\n"
                            f"{_format_plan(new_plan)}"
                        )
                    )
                ],
            }
        return {
            "messages": [
                AIMessage(content=f"[REPLANNER continue] {decision.reasoning}")
            ]
        }

    def should_continue(state: PlanExecuteState) -> Literal["execute", "finish"]:
        if state.get("response"):
            return "finish"
        if state.get("replan_count", 0) > 3:
            return "finish"
        if state["current_step_idx"] >= len(state["plan"]):
            return "finish"
        return "execute"

    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("replanner", replanner_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")
    graph.add_conditional_edges(
        "replanner", should_continue, {"execute": "executor", "finish": END}
    )
    return graph.compile(checkpointer=checkpointer)


ARRIVAL_TASK = (
    "Я повернуся додому через 2 години. Зараз зима, -10 на вулиці. Підготуй дім. "
    "Перевір стан пристроїв, прочитай температуру, запропонуй план дій."
)


def run_arrival_scenario() -> dict:
    reset_house()
    app = build_pe_graph()
    return app.invoke(
        {
            "task": ARRIVAL_TASK,
            "messages": [],
            "plan": [],
            "completed_steps": [],
            "current_step_idx": 0,
            "response": "",
            "replan_count": 0,
        }
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[TASK]", ARRIVAL_TASK)
    result = run_arrival_scenario()
    print("\n[PLAN]")
    print(_format_plan(result.get("plan") or []))
    print("\n[COMPLETED]")
    for item in result.get("completed_steps") or []:
        print("-", item[:300])
    print("\n[RESPONSE]")
    print(result.get("response"))
