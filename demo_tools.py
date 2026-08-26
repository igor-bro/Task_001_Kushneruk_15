"""Демо tools Smart Home (крок 2)."""

from __future__ import annotations

import sys

from pydantic import ValidationError

from tools import (
    DeviceControlInput,
    SensorReadInput,
    device_control,
    device_status,
    energy_consumption,
    reset_house,
    schedule_list,
    sensor_read,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    reset_house()
    try:
        SensorReadInput(device_id="abc", sensor_type="pressure")
    except ValidationError as exc:
        print("OK validation sensor_type:", exc.errors()[0]["msg"])

    try:
        DeviceControlInput(device_id="hvac_living", action="explode")
    except ValidationError as exc:
        print("OK validation action:", exc.errors()[0]["msg"])

    print(sensor_read.invoke({"device_id": "living_temp", "sensor_type": "temperature"}))
    print(device_status.invoke({"device_id": "hvac_living"}))
    print(energy_consumption.invoke({"period": "today"}))
    print(schedule_list.invoke({"room": "living"}))
    print(
        device_control.invoke(
            {"device_id": "hvac_living", "action": "on", "params": {"target_c": 22}}
        )
    )
    print(device_status.invoke({"device_id": "hvac_living"}))
    print("tools OK")


if __name__ == "__main__":
    main()
