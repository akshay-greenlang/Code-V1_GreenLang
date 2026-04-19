# -*- coding: utf-8 -*-
"""
Shared pytest fixtures and configuration for Factors load testing.

Provides test API keys, base URL configuration, and common load-test
utilities used by both Locust and k6 test runners.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest


# ── SLA thresholds ──────────────────────────────────────────────────

SLA_P95_MS = 50
SLA_ERROR_RATE_PCT = 1.0
SLA_TARGET_RPS = 1000

# Scenario weights (must sum to 100)
SCENARIO_WEIGHTS = {
    "search": 40,
    "list": 20,
    "get": 20,
    "match": 10,
    "edition": 5,
    "export": 5,
}

# Test API key for load testing (non-production)
LOAD_TEST_API_KEY = os.getenv(
    "GL_LOAD_TEST_API_KEY",
    "gl_test_load_key_xxxxxxxxxxxxxxxxxxxxxxxx",
)

# Base URL for the Factors API under test
BASE_URL = os.getenv("GL_FACTORS_BASE_URL", "http://localhost:8000")


@dataclass
class SLAThreshold:
    """SLA threshold definition for a single endpoint."""

    endpoint: str
    p50_ms: float = 20.0
    p95_ms: float = SLA_P95_MS
    p99_ms: float = 100.0
    max_error_rate_pct: float = SLA_ERROR_RATE_PCT
    min_rps: float = 0.0


# Per-endpoint SLA thresholds
ENDPOINT_SLAS: Dict[str, SLAThreshold] = {
    "/v2/factors/search": SLAThreshold(
        endpoint="/v2/factors/search", p50_ms=15, p95_ms=50, p99_ms=100,
    ),
    "/v2/factors": SLAThreshold(
        endpoint="/v2/factors", p50_ms=10, p95_ms=40, p99_ms=80,
    ),
    "/v2/factors/{factor_id}": SLAThreshold(
        endpoint="/v2/factors/{factor_id}", p50_ms=5, p95_ms=25, p99_ms=50,
    ),
    "/v2/factors/match": SLAThreshold(
        endpoint="/v2/factors/match", p50_ms=30, p95_ms=50, p99_ms=150,
    ),
    "/v2/editions": SLAThreshold(
        endpoint="/v2/editions", p50_ms=5, p95_ms=20, p99_ms=40,
    ),
    "/v2/factors/export": SLAThreshold(
        endpoint="/v2/factors/export", p50_ms=100, p95_ms=500, p99_ms=1000,
    ),
}


@dataclass
class EndpointResult:
    """Aggregated results for a single endpoint from a load test run."""

    endpoint: str
    request_count: int = 0
    error_count: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    avg_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    rps: float = 0.0

    @property
    def error_rate_pct(self) -> float:
        """Calculate error rate as a percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100.0

    def exceeds_sla(self, sla: SLAThreshold) -> List[str]:
        """Return list of SLA violations for this endpoint."""
        violations: List[str] = []
        if self.p95_ms > sla.p95_ms:
            violations.append(
                "p95 %.1fms > SLA %.1fms" % (self.p95_ms, sla.p95_ms)
            )
        if self.p99_ms > sla.p99_ms:
            violations.append(
                "p99 %.1fms > SLA %.1fms" % (self.p99_ms, sla.p99_ms)
            )
        if self.error_rate_pct > sla.max_error_rate_pct:
            violations.append(
                "error_rate %.2f%% > SLA %.2f%%"
                % (self.error_rate_pct, sla.max_error_rate_pct)
            )
        return violations


# ── Sample data for load tests ──────────────────────────────────────

SAMPLE_SEARCH_QUERIES = [
    "diesel",
    "natural gas",
    "electricity",
    "gasoline",
    "coal",
    "fuel oil",
    "propane",
    "biomass",
    "solar",
    "wind",
    "jet fuel",
    "LPG",
    "CNG",
    "hydrogen",
    "ethanol",
    "biodiesel",
    "refrigerant R-410A",
    "HFC-134a",
    "methane fugitive",
    "waste landfill",
]

SAMPLE_GEOGRAPHIES = ["US", "GB", "DE", "FR", "JP", "AU", "CA", "IN", "BR", "CN"]

SAMPLE_SCOPES = ["scope_1", "scope_2", "scope_3"]

SAMPLE_FUEL_TYPES = [
    "diesel",
    "natural_gas",
    "gasoline",
    "coal_bituminous",
    "fuel_oil_2",
    "propane",
    "electricity",
]


@pytest.fixture(scope="session")
def load_test_config() -> Dict[str, Any]:
    """Session-scoped load test configuration."""
    return {
        "base_url": BASE_URL,
        "api_key": LOAD_TEST_API_KEY,
        "sla_p95_ms": SLA_P95_MS,
        "sla_error_rate_pct": SLA_ERROR_RATE_PCT,
        "sla_target_rps": SLA_TARGET_RPS,
        "scenario_weights": SCENARIO_WEIGHTS,
    }


@pytest.fixture(scope="session")
def endpoint_slas() -> Dict[str, SLAThreshold]:
    """Session-scoped endpoint SLA thresholds."""
    return ENDPOINT_SLAS
