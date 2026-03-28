import asyncio
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from greenlang.v2.reliability_runtime import (
    ErrorClass,
    classify_connector_error,
    validate_connector_registry,
)

_BASE_CONNECTOR_PATH = (
    Path(__file__).resolve().parents[2] / "greenlang" / "integration" / "integrations" / "base_connector.py"
)
_BASE_CONNECTOR_SPEC = importlib.util.spec_from_file_location(
    "phase3_base_connector_module",
    _BASE_CONNECTOR_PATH,
)
_BASE_CONNECTOR_MODULE = importlib.util.module_from_spec(_BASE_CONNECTOR_SPEC)
assert _BASE_CONNECTOR_SPEC and _BASE_CONNECTOR_SPEC.loader
_BASE_CONNECTOR_SPEC.loader.exec_module(_BASE_CONNECTOR_MODULE)

BaseConnector = _BASE_CONNECTOR_MODULE.BaseConnector
ConnectorConfig = _BASE_CONNECTOR_MODULE.ConnectorConfig
CircuitBreakerError = _BASE_CONNECTOR_MODULE.CircuitBreakerError


class Query(BaseModel):
    key: str = Field(...)
    value: int = Field(default=1)


class Payload(BaseModel):
    result: int = Field(...)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Config(ConnectorConfig):
    pass


class AcceptanceConnector(BaseConnector[Query, Payload, Config]):
    connector_id = "sap-erp"
    connector_version = "2.0.0"

    def __init__(self, config: Config):
        super().__init__(config)
        self.calls = 0
        self.fail_mode = None

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> bool:
        return True

    async def _health_check_impl(self) -> bool:
        return True

    async def _fetch_data_impl(self, query: Query) -> Payload:
        self.calls += 1
        if self.fail_mode == "transient" and self.calls < 3:
            raise ConnectionError("transient connection reset")
        if self.fail_mode == "always":
            raise ConnectionError("transient failure")
        if self.fail_mode == "slow":
            await asyncio.sleep(2)
        if self.fail_mode == "schema":
            raise ValueError("schema validation mismatch")
        return Payload(result=query.value * 2, ts=datetime(2026, 1, 1, tzinfo=timezone.utc))


@pytest.mark.asyncio
async def test_prioritized_connector_registry_validates() -> None:
    result = validate_connector_registry()
    assert result.ok, result.errors
    assert any(p.connector_id == "sap-erp" for p in result.profiles)


@pytest.mark.asyncio
async def test_runtime_profile_binding_applies_registry_limits() -> None:
    config = Config(connector_id="sap-erp", connector_type="erp")
    conn = AcceptanceConnector(config)
    assert conn.reliability_profile is not None
    assert conn.config.max_retries == conn.reliability_profile.retry.max_attempts
    assert conn.config.timeout_seconds >= 1


@pytest.mark.asyncio
async def test_idempotency_acceptance_same_input_same_result_hash() -> None:
    config = Config(connector_id="unlisted-idempotency", connector_type="erp")
    conn = AcceptanceConnector(config)
    await conn.connect()
    q = Query(key="k1", value=5)
    payload1, prov1 = await conn.fetch_data(q)
    payload2, prov2 = await conn.fetch_data(q)
    assert payload1.result == payload2.result
    assert prov1.query_hash == prov2.query_hash
    assert prov1.response_hash == prov2.response_hash


@pytest.mark.asyncio
async def test_retry_and_timeout_budget_behavior() -> None:
    config = Config(connector_id="unlisted-retry", connector_type="erp")
    conn = AcceptanceConnector(config)
    await conn.connect()
    conn.fail_mode = "transient"
    payload, _ = await conn.fetch_data(Query(key="k2", value=3))
    assert payload.result == 6
    assert conn.calls >= 3

    slow = AcceptanceConnector(Config(connector_id="unlisted-slow", connector_type="erp"))
    await slow.connect()
    slow.fail_mode = "slow"
    with pytest.raises(TimeoutError):
        await slow.fetch_data(Query(key="k3", value=1), timeout=1)


@pytest.mark.asyncio
async def test_circuit_breaker_open_and_half_open_behavior() -> None:
    config = Config(
        connector_id="unlisted-circuit",
        connector_type="erp",
        circuit_breaker_threshold=1,
        circuit_breaker_timeout=1,
        max_retries=0,
    )
    conn = AcceptanceConnector(config)
    await conn.connect()
    conn.fail_mode = "always"

    with pytest.raises(Exception):
        await conn.fetch_data(Query(key="k4", value=1))
    with pytest.raises(CircuitBreakerError):
        await conn.fetch_data(Query(key="k4b", value=1))

    await asyncio.sleep(1.1)
    conn.fail_mode = None
    payload, _ = await conn.fetch_data(Query(key="k4c", value=2))
    assert payload.result == 4


def test_error_classification_behavior() -> None:
    assert classify_connector_error(ConnectionError("network")) == ErrorClass.TRANSIENT
    assert classify_connector_error(Exception("429 rate limit")) == ErrorClass.THROTTLING
    assert classify_connector_error(Exception("auth token expired")) == ErrorClass.AUTH
    assert classify_connector_error(ValueError("schema mismatch")) == ErrorClass.SCHEMA
    assert classify_connector_error(RuntimeError("boom")) == ErrorClass.PERMANENT
