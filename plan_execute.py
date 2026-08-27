"""Plan-and-Execute для Smart Home (крок 6).

Граф: START → planner → executor → replanner ⇄ executor | END

Executor виконує КОЖЕН крок плану вкладеним ReAct-агентом
(повний цикл agent ⇄ tools), а не одноразовим LLM-викликом.

Запуск:
    python plan_execute.py
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from knowledge import knowledge_search
from models import Plan, PlanStep, ReplanDecision
from react_agent import invoke_react
from safety import SafetyLimits
from tools import SAFE_TOOLS, reset_house

load_dotenv()

EXECUTOR_TOOLS = SAFE_TOOLS + [knowledge_search]

SYSTEM_PLANNER = """\
Ти planner для Smart Home. Склади план 3–4 кроків.
Кожен крок: step_id, description, expected_tool
(sensor_read | device_status | energy_consumption | schedule_list |
knowledge_search | null).
НЕ плануй device_control — лише збір стану і рекомендації.
"""

SYSTEM_EXECUTOR = """\
Ти executor Smart Home (вкладений ReAct). Відповідай українською.
Мета: виконати РІВНО один крок плану мінімальною кількістю викликів.

Правила:
1. Якщо в підказці є expected_tool — виклич його ОДИН раз з коректними args.
2. Після отримання ToolMessage одразу дай стислий підсумок кроку БЕЗ нових tool-calls.
3. Не викликай зайві tools «про всяк випадок».
4. device_control недоступний.
5. Не вигадуй показники — лише дані з tools.
"""

# Ліміт внутрішнього ReAct на ОДИН крок плану (зовнішній цикл — окремо).
EXECUTOR_MAX_STEPS = 4
EXECUTOR_TIMEOUT_SECONDS = 90.0


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


def _step_result_from_react(messages: list) -> str:
    """Підсумок вкладеного ReAct: фінальна відповідь + факти з tools."""
    if not messages:
        return ""

    tool_chunks: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "tool")
            preview = str(msg.content or "")[:500]
            tool_chunks.append(f"{name}: {preview}")

    last = messages[-1]
    final = str(getattr(last, "content", "") or "").strip()
    # Якщо цикл обрізало safety-лімітом — все одно віддаємо зібрані tool-факти.
    if final.startswith("⚠️") and tool_chunks:
        return " | ".join(tool_chunks) + f"\n({final})"
    if final:
        if tool_chunks and "status" not in final.lower():
            return final + "\n[tools] " + " | ".join(tool_chunks[:3])
        return final
    if tool_chunks:
        return " | ".join(tool_chunks)
    return ""


def build_pe_graph(checkpointer=None):
    llm = build_llm()
    planner_llm = llm.with_structured_output(Plan)
    replanner_llm = llm.with_structured_output(ReplanDecision)

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
        """Один крок плану = повний вкладений ReAct-цикл (як у практикумі step6)."""
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
            f"Виконай наступний крок плану: {desc}\n"
            f"Це крок {idx + 1} з {len(plan)}.\n"
            f"expected_tool (підказка): {expected or 'будь-який релевантний'}\n"
            f"{context}\n"
            "Якщо expected_tool заданий — віддай перевагу саме йому. "
            "Використовуй інструменти в ReAct-циклі. Поверни стислий результат кроку."
        )

        # Вкладений граф: порожня історія + лише стиснутий context у промпті.
        try:
            nested = invoke_react(
                prompt,
                tools=EXECUTOR_TOOLS,
                system_prompt=SYSTEM_EXECUTOR,
                safety=SafetyLimits(
                    max_steps=EXECUTOR_MAX_STEPS,
                    timeout_seconds=EXECUTOR_TIMEOUT_SECONDS,
                ),
                reset_house_state=False,
                trajectory_path=None,
            )
            step_result = (
                _step_result_from_react(nested.get("messages") or [])
                or "Крок без відповіді."
            )
            react_steps = nested.get("step_count")
            stop = nested.get("stop_reason")
        except Exception as exc:  # noqa: BLE001
            # Transient API / мережа не повинні валити весь P&E-граф.
            step_result = f"помилка вкладеного ReAct: {type(exc).__name__}: {exc}"
            react_steps = None
            stop = "executor_error"

        completed = state["completed_steps"] + [f"Крок {idx + 1}: {step_result}"]
        return {
            "completed_steps": completed,
            "current_step_idx": idx + 1,
            "messages": [
                AIMessage(
                    content=(
                        f"[EXECUTOR/ReAct] step={idx + 1} "
                        f"expected={expected} react_steps={react_steps} "
                        f"stop={stop}\n{step_result}"
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
                "Виріши: finish лише якщо зібрано достатньо ФАКТИЧНИХ даних "
                "(температура/статуси/розклад/знання) для рекомендацій; "
                "інакше continue або replan. Не finish після одного кроку, "
                "якщо залишились непрочитані кроки з expected_tool. "
                "Для finish обов'язково final_answer українською."
            )
            if decision is not None:
                break

        if decision is None:
            if remaining:
                return {}
            return {"response": "Завдання виконано (fallback)."}

        if decision.action == "finish":
            # Код > модель: не дозволяємо finish, поки виконано менше 2 кроків
            # (або весь короткий план), якщо ще є залишок.
            min_done = min(2, len(state["plan"]))
            if remaining and state["current_step_idx"] < min_done:
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "[REPLANNER continue forced] замало виконаних кроків "
                                f"({state['current_step_idx']}/{min_done})"
                            )
                        )
                    ]
                }
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
