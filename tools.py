"""Доменні tools Smart Home (варіант 15).

Кожен tool має Pydantic v2 схему з Field + field_validator
і повертає JSON-рядок: {status, data} або {status, error}.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

HOUSE: dict[str, Any] = {
    "outside_temp_c": -10.0,
    "season": "winter",
    "sensors": {
        "living_temp": {
            "room": "living",
            "type": "temperature",
            "value": 18.5,
            "unit": "C",
        },
        "living_humidity": {
            "room": "living",
            "type": "humidity",
            "value": 42.0,
            "unit": "%",
        },
        "living_motion": {
            "room": "living",
            "type": "motion",
            "value": False,
            "unit": "bool",
        },
        "bedroom_temp": {
            "room": "bedroom",
            "type": "temperature",
            "value": 17.0,
            "unit": "C",
        },
        "bedroom_humidity": {
            "room": "bedroom",
            "type": "humidity",
            "value": 45.0,
            "unit": "%",
        },
        "kitchen_temp": {
            "room": "kitchen",
            "type": "temperature",
            "value": 19.0,
            "unit": "C",
        },
        "hallway_motion": {
            "room": "hallway",
            "type": "motion",
            "value": False,
            "unit": "bool",
        },
    },
    "devices": {
        "hvac_living": {
            "room": "living",
            "kind": "climate",
            "on": False,
            "settings": {"target_c": 21.0, "mode": "heat"},
        },
        "hvac_bedroom": {
            "room": "bedroom",
            "kind": "climate",
            "on": False,
            "settings": {"target_c": 20.0, "mode": "heat"},
        },
        "light_living": {
            "room": "living",
            "kind": "light",
            "on": False,
            "settings": {"brightness": 70},
        },
        "light_hallway": {
            "room": "hallway",
            "kind": "light",
            "on": False,
            "settings": {"brightness": 40},
        },
        "lock_front": {
            "room": "hallway",
            "kind": "lock",
            "on": True,
            "settings": {"auto_lock": True},
        },
        "boiler": {
            "room": "utility",
            "kind": "boiler",
            "on": False,
            "settings": {"target_c": 55.0},
        },
    },
    "schedules": {
        "living": [
            {
                "id": "liv-heat-morning",
                "action": "hvac_on",
                "time": "06:30",
                "enabled": True,
            },
            {
                "id": "liv-light-evening",
                "action": "light_on",
                "time": "18:00",
                "enabled": True,
            },
        ],
        "bedroom": [
            {
                "id": "bed-heat-night",
                "action": "hvac_on",
                "time": "21:00",
                "enabled": False,
            },
        ],
        "kitchen": [
            {
                "id": "kit-light-morning",
                "action": "light_on",
                "time": "07:00",
                "enabled": True,
            },
        ],
        "hallway": [
            {
                "id": "hall-motion-light",
                "action": "light_on_motion",
                "time": "always",
                "enabled": True,
            },
        ],
    },
    "energy_kwh": {"today": 12.4, "week": 78.2, "month": 310.5},
}

_HOUSE_TEMPLATE: dict[str, Any] = deepcopy(HOUSE)
LEDGER_PATH = Path(__file__).with_name("device_actions.json")
ALLOWED_SENSOR_TYPES = frozenset({"temperature", "humidity", "motion"})
ALLOWED_PERIODS = frozenset({"today", "week", "month"})
ALLOWED_ACTIONS = frozenset({"on", "off", "set", "lock", "unlock"})
ALLOWED_ROOMS = frozenset(
    {"living", "bedroom", "kitchen", "hallway", "utility", "all"}
)


def _ok(data: dict[str, Any]) -> str:
    return json.dumps({"status": "ok", "data": data}, ensure_ascii=False)


def _err(message: str, **extra: Any) -> str:
    return json.dumps({"status": "error", "error": message, **extra}, ensure_ascii=False)


def reset_house() -> None:
    """Скидає mock-стан до початкового шаблону."""
    global HOUSE
    HOUSE = deepcopy(_HOUSE_TEMPLATE)


class SensorReadInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    device_id: str = Field(
        ...,
        description="ID сенсора, напр. living_temp",
        min_length=3,
        max_length=64,
    )
    sensor_type: str = Field(
        ..., description="Тип сенсора: temperature | humidity | motion"
    )

    @field_validator("sensor_type")
    @classmethod
    def sensor_type_ok(cls, v: str) -> str:
        value = v.strip().lower()
        if value not in ALLOWED_SENSOR_TYPES:
            raise ValueError(
                f"sensor_type має бути одним із: {sorted(ALLOWED_SENSOR_TYPES)}"
            )
        return value


class DeviceStatusInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    device_id: str = Field(
        ...,
        description="ID пристрою, напр. hvac_living",
        min_length=3,
        max_length=64,
    )


class EnergyConsumptionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    period: str = Field(..., description="Період: today | week | month")

    @field_validator("period")
    @classmethod
    def period_ok(cls, v: str) -> str:
        value = v.strip().lower()
        if value not in ALLOWED_PERIODS:
            raise ValueError(f"period має бути одним із: {sorted(ALLOWED_PERIODS)}")
        return value


class ScheduleListInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    room: str = Field(
        ..., description="Кімната: living | bedroom | kitchen | hallway | all"
    )

    @field_validator("room")
    @classmethod
    def room_ok(cls, v: str) -> str:
        value = v.strip().lower()
        if value not in ALLOWED_ROOMS:
            raise ValueError(f"room має бути одним із: {sorted(ALLOWED_ROOMS)}")
        return value


class DeviceControlInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    device_id: str = Field(..., description="ID пристрою", min_length=3, max_length=64)
    action: str = Field(..., description="Дія: on | off | set | lock | unlock")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Параметри, напр. {target_c: 22}",
    )

    @field_validator("action")
    @classmethod
    def action_ok(cls, v: str) -> str:
        value = v.strip().lower()
        if value not in ALLOWED_ACTIONS:
            raise ValueError(f"action має бути одним із: {sorted(ALLOWED_ACTIONS)}")
        return value


@tool(args_schema=SensorReadInput)
def sensor_read(device_id: str, sensor_type: str) -> str:
    """Прочитати mock-сенсор (температура, вологість, рух)."""
    sensor = HOUSE["sensors"].get(device_id)
    if sensor is None:
        return _err(f"Невідомий сенсор: {device_id}", device_id=device_id)
    if sensor["type"] != sensor_type:
        return _err(
            f"Сенсор {device_id} має тип {sensor['type']}, а не {sensor_type}",
            device_id=device_id,
        )
    return _ok(
        {
            "device_id": device_id,
            "sensor_type": sensor_type,
            "room": sensor["room"],
            "value": sensor["value"],
            "unit": sensor["unit"],
            "outside_temp_c": HOUSE["outside_temp_c"],
            "season": HOUSE["season"],
        }
    )


@tool(args_schema=DeviceStatusInput)
def device_status(device_id: str) -> str:
    """Отримати статус пристрою (on/off і settings)."""
    device = HOUSE["devices"].get(device_id)
    if device is None:
        return _err(f"Невідомий пристрій: {device_id}", device_id=device_id)
    return _ok(
        {
            "device_id": device_id,
            "room": device["room"],
            "kind": device["kind"],
            "on": device["on"],
            "settings": device["settings"],
        }
    )


@tool(args_schema=EnergyConsumptionInput)
def energy_consumption(period: str) -> str:
    """Повернути mock-споживання електроенергії за період."""
    kwh = HOUSE["energy_kwh"].get(period)
    if kwh is None:
        return _err(f"Немає даних для period={period}")
    return _ok({"period": period, "kwh": kwh, "unit": "kWh"})


@tool(args_schema=ScheduleListInput)
def schedule_list(room: str) -> str:
    """Список запланованих автоматизацій для кімнати (або all)."""
    if room == "all":
        items = []
        for room_name, schedules in HOUSE["schedules"].items():
            for item in schedules:
                items.append({"room": room_name, **item})
        return _ok({"room": "all", "count": len(items), "schedules": items})
    schedules = HOUSE["schedules"].get(room, [])
    return _ok({"room": room, "count": len(schedules), "schedules": schedules})


@tool(args_schema=DeviceControlInput)
def device_control(device_id: str, action: str, params: dict | None = None) -> str:
    """Керувати пристроєм. РИЗИКОВА дія — потрібен HITL."""
    params = params or {}
    device = HOUSE["devices"].get(device_id)
    if device is None:
        return _err(f"Невідомий пристрій: {device_id}", device_id=device_id)

    kind = device["kind"]
    if action in {"on", "off"}:
        device["on"] = action == "on"
        for key, value in params.items():
            if key != "on":
                device["settings"][key] = value
    elif action == "lock":
        if kind != "lock":
            return _err(f"lock доступний лише для lock-пристроїв, не {kind}")
        device["on"] = True
    elif action == "unlock":
        if kind != "lock":
            return _err(f"unlock доступний лише для lock-пристроїв, не {kind}")
        device["on"] = False
    elif action == "set":
        for key, value in params.items():
            device["settings"][key] = value
        if "on" in params:
            device["on"] = bool(params["on"])
    else:
        return _err(f"Невідома дія: {action}")

    record = {
        "device_id": device_id,
        "action": action,
        "params": params,
        "result_on": device["on"],
        "settings": deepcopy(device["settings"]),
    }
    ledger: list[dict] = []
    if LEDGER_PATH.exists():
        ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    ledger.append(record)
    LEDGER_PATH.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return _ok(record)


SAFE_TOOLS = [sensor_read, device_status, energy_consumption, schedule_list]
RISKY_TOOLS = [device_control]
ALL_TOOLS = SAFE_TOOLS + RISKY_TOOLS
RISKY_TOOL_NAMES = frozenset(t.name for t in RISKY_TOOLS)
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
