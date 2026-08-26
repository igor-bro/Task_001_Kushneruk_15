"""Pydantic-моделі для structured outputs (крок 4 / 6)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """Один крок плану з явним expected_tool для routing."""

    step_id: int = Field(..., description="Номер кроку (з 1)", ge=1)
    description: str = Field(..., description="Опис дії", max_length=500)
    expected_tool: Optional[str] = Field(
        default=None,
        description=(
            "Очікуваний tool: sensor_read, device_status, energy_consumption, "
            "schedule_list, knowledge_search, device_control або null для тексту"
        ),
    )


class Plan(BaseModel):
    """Структурований план виконання задачі."""

    goal: str = Field(..., description="Мета задачі")
    steps: list[PlanStep] = Field(
        ...,
        description="Список кроків",
        min_length=1,
        max_length=15,
    )
    reasoning: str = Field(..., description="Обґрунтування плану")


class ReplanDecision(BaseModel):
    """Рішення replanner після кроку."""

    action: Literal["continue", "replan", "finish"] = Field(
        ...,
        description=(
            "continue=наступний крок, replan=оновити план, finish=завершити"
        ),
    )
    updated_steps: Optional[list[PlanStep]] = Field(
        default=None,
        description="Новий план, якщо action=replan",
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="Фінальна відповідь, якщо action=finish",
    )
    reasoning: str = Field(..., description="Обґрунтування рішення")
