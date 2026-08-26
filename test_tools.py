"""Тести Pydantic-схем і tools."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tools import (
    DeviceControlInput,
    DeviceStatusInput,
    EnergyConsumptionInput,
    ScheduleListInput,
    SensorReadInput,
    device_control,
    device_status,
    energy_consumption,
    reset_house,
    schedule_list,
    sensor_read,
)


@pytest.fixture(autouse=True)
def _clean_house():
    reset_house()
    yield
    reset_house()


class TestPydanticSchemas:
    def test_sensor_read_valid(self):
        inp = SensorReadInput(device_id="living_temp", sensor_type="temperature")
        assert inp.sensor_type == "temperature"

    def test_sensor_read_invalid_type(self):
        with pytest.raises(ValidationError):
            SensorReadInput(device_id="living_temp", sensor_type="pressure")

    def test_device_status_valid(self):
        assert DeviceStatusInput(device_id="hvac_living").device_id == "hvac_living"

    def test_energy_invalid_period(self):
        with pytest.raises(ValidationError):
            EnergyConsumptionInput(period="year")

    def test_energy_valid_period(self):
        assert EnergyConsumptionInput(period="week").period == "week"

    def test_schedule_invalid_room(self):
        with pytest.raises(ValidationError):
            ScheduleListInput(room="garage")

    def test_device_control_invalid_action(self):
        with pytest.raises(ValidationError):
            DeviceControlInput(device_id="light_living", action="explode")

    def test_device_control_valid(self):
        inp = DeviceControlInput(
            device_id="light_living", action="on", params={"brightness": 80}
        )
        assert inp.action == "on"


class TestTools:
    def test_sensor_read_ok(self):
        data = json.loads(
            sensor_read.invoke(
                {"device_id": "living_temp", "sensor_type": "temperature"}
            )
        )
        assert data["status"] == "ok"
        assert data["data"]["value"] == 18.5

    def test_sensor_read_wrong_type(self):
        data = json.loads(
            sensor_read.invoke({"device_id": "living_temp", "sensor_type": "humidity"})
        )
        assert data["status"] == "error"

    def test_device_status_unknown(self):
        data = json.loads(device_status.invoke({"device_id": "no_such_device"}))
        assert data["status"] == "error"

    def test_energy_and_schedule(self):
        e = json.loads(energy_consumption.invoke({"period": "today"}))
        s = json.loads(schedule_list.invoke({"room": "living"}))
        assert e["status"] == "ok" and e["data"]["kwh"] == 12.4
        assert s["status"] == "ok" and s["data"]["count"] >= 1

    def test_device_control_changes_state(self):
        before = json.loads(device_status.invoke({"device_id": "hvac_living"}))
        assert before["data"]["on"] is False
        data = json.loads(
            device_control.invoke(
                {
                    "device_id": "hvac_living",
                    "action": "on",
                    "params": {"target_c": 22},
                }
            )
        )
        assert data["status"] == "ok"
        after = json.loads(device_status.invoke({"device_id": "hvac_living"}))
        assert after["data"]["on"] is True
        assert after["data"]["settings"]["target_c"] == 22
