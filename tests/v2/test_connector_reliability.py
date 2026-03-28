import pytest

from greenlang.v2.reliability import ConnectorReliabilityProfile


def test_connector_reliability_profile_valid() -> None:
    profile = ConnectorReliabilityProfile(
        connector_id="sap-erp",
        owner_team="data-integration",
        retry={
            "max_attempts": 3,
            "backoff_strategy": "exponential",
            "retryable_status_codes": [429, 502, 503],
        },
        timeout={"connect_ms": 1000, "read_ms": 2000, "overall_ms": 4000},
        circuit_breaker={
            "failure_rate_threshold": 0.5,
            "open_state_seconds": 30,
            "half_open_probe_requests": 3,
        },
    )
    assert profile.idempotency_required
    assert profile.dead_letter_enabled


def test_connector_overall_timeout_must_cover_connect_plus_read() -> None:
    with pytest.raises(ValueError):
        ConnectorReliabilityProfile(
            connector_id="sap-erp",
            owner_team="data-integration",
            retry={
                "max_attempts": 3,
                "backoff_strategy": "exponential",
                "retryable_status_codes": [429, 502, 503],
            },
            timeout={"connect_ms": 3000, "read_ms": 3000, "overall_ms": 5000},
            circuit_breaker={
                "failure_rate_threshold": 0.5,
                "open_state_seconds": 30,
                "half_open_probe_requests": 3,
            },
        )

