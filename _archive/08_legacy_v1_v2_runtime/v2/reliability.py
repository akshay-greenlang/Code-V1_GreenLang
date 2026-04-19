# -*- coding: utf-8 -*-
"""Connector reliability policy models for V2."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RetryPolicy(BaseModel):
    max_attempts: int = Field(ge=1, le=10)
    backoff_strategy: str = Field(default="exponential")
    retryable_status_codes: list[int] = Field(default_factory=list)


class TimeoutBudget(BaseModel):
    connect_ms: int = Field(ge=100, le=60000)
    read_ms: int = Field(ge=100, le=60000)
    overall_ms: int = Field(ge=200, le=120000)

    @field_validator("overall_ms")
    @classmethod
    def _overall_must_cover_connect_read(cls, value: int, info) -> int:
        connect = info.data.get("connect_ms", 0)
        read = info.data.get("read_ms", 0)
        if value < connect + read:
            raise ValueError("overall_ms must be >= connect_ms + read_ms")
        return value


class CircuitBreakerPolicy(BaseModel):
    failure_rate_threshold: float = Field(ge=0.01, le=1.0)
    open_state_seconds: int = Field(ge=1, le=300)
    half_open_probe_requests: int = Field(ge=1, le=50)


class ConnectorReliabilityProfile(BaseModel):
    connector_id: str
    owner_team: str
    retry: RetryPolicy
    timeout: TimeoutBudget
    circuit_breaker: CircuitBreakerPolicy
    idempotency_required: bool = True
    dead_letter_enabled: bool = True

    @field_validator("connector_id", "owner_team")
    @classmethod
    def _non_empty_strings(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

