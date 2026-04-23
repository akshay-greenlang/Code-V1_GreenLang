# -*- coding: utf-8 -*-
"""
Factors API load test (DEP12).

Two user scenarios:

    * ``FactorsResolveUser`` — hammers ``/v1/resolve`` with realistic single-row
      resolve payloads. Target: 1000 rps sustained for 10 minutes.
    * ``FactorsBatchUser``   — posts 10k-row payloads to ``/v1/batch/resolve``.
      Target: 100 rps sustained.

This script is designed to **run against the staging cluster only**
(``factors-staging.greenlang.com``). It refuses to start if the target host
matches ``factors.greenlang.com`` or ``*.prod.*`` unless the operator sets
``GL_FACTORS_LOAD_TEST_ALLOW_PROD=1``. The guard is deliberately annoying.

Run via:

    locust -f tests/factors/load/factors_load_test.py \\
        --host https://factors-staging.greenlang.com \\
        --users 1000 --spawn-rate 50 --run-time 10m \\
        --tags resolve

    locust -f tests/factors/load/factors_load_test.py \\
        --host https://factors-staging.greenlang.com \\
        --users 100  --spawn-rate 10 --run-time 10m \\
        --tags batch

Or headless with CSV output:

    locust -f tests/factors/load/factors_load_test.py \\
        --host https://factors-staging.greenlang.com \\
        --headless -u 1000 -r 50 -t 10m \\
        --csv out/factors-load-$(date +%Y%m%d)

The CSV + `*_stats.csv` output feeds the template
``tests/factors/load/load_report_template.md``.
"""
from __future__ import annotations

import json
import os
import random
import sys
from typing import Dict, List

try:
    from locust import HttpUser, between, events, tag, task  # type: ignore
except ImportError:
    print(
        "locust is required; install with `pip install locust` before running.",
        file=sys.stderr,
    )
    raise


# ---------------------------------------------------------------------------
# Safety guard: refuse to hit production.
# ---------------------------------------------------------------------------


PROD_HOST_BLOCKLIST = (
    "factors.greenlang.com",
    "factors.greenlang.ai",
    "factors.prod.",
    ".prod.greenlang",
)


@events.test_start.add_listener
def _assert_staging_only(environment, **_kwargs):
    host = (environment.host or "").lower()
    allow_prod = os.getenv("GL_FACTORS_LOAD_TEST_ALLOW_PROD") == "1"
    for bad in PROD_HOST_BLOCKLIST:
        if bad in host and not allow_prod:
            raise SystemExit(
                f"Refusing to load-test {host!r}. "
                "Load tests are staging-only. Set "
                "GL_FACTORS_LOAD_TEST_ALLOW_PROD=1 if you know what you're doing."
            )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


RESOLVE_FIXTURES: List[Dict[str, object]] = [
    {
        "family": "electricity",
        "jurisdiction": "US-CA",
        "year": 2024,
        "method_profile": "ghg_protocol_scope2_market_based",
        "unit": "kWh",
        "quantity": 1000,
    },
    {
        "family": "combustion",
        "jurisdiction": "UK",
        "year": 2024,
        "method_profile": "defra_2024",
        "fuel": "natural_gas",
        "unit": "therm",
        "quantity": 150,
    },
    {
        "family": "freight",
        "jurisdiction": "EU-27",
        "year": 2024,
        "mode": "truck_hgv",
        "distance_km": 500,
        "payload_t": 10,
    },
    {
        "family": "material",
        "jurisdiction": "DE",
        "year": 2024,
        "cn_code": "72083900",
        "mass_kg": 10000,
    },
    {
        "family": "land",
        "jurisdiction": "BR",
        "year": 2024,
        "hectares": 5,
        "land_use": "cropland_to_pasture",
    },
]


def _batch_payload(n: int = 10_000) -> Dict[str, object]:
    """Return a realistic batch payload with ``n`` rows."""
    rows = []
    for i in range(n):
        base = dict(random.choice(RESOLVE_FIXTURES))
        base["client_row_id"] = f"row-{i:06d}"
        rows.append(base)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


class FactorsResolveUser(HttpUser):
    """Single-row /v1/resolve user. Target: 1000 rps."""

    weight = 10
    wait_time = between(0.01, 0.05)

    def on_start(self):
        self.token = os.getenv("GL_FACTORS_LOAD_TEST_TOKEN", "")
        if not self.token:
            raise SystemExit(
                "GL_FACTORS_LOAD_TEST_TOKEN must be set (Pro-tier JWT for the "
                "staging tenant `loadtest`)."
            )

    @tag("resolve")
    @task(10)
    def resolve(self):
        fixture = random.choice(RESOLVE_FIXTURES)
        with self.client.post(
            "/v1/resolve",
            json=fixture,
            headers={
                "Authorization": f"Bearer {self.token}",
                "X-GreenLang-Load-Test": "1",
            },
            name="POST /v1/resolve",
            catch_response=True,
        ) as resp:
            if resp.status_code == 429:
                # Don't fail the test on rate-limit; just mark it.
                resp.success()
            elif resp.status_code >= 500:
                resp.failure(f"5xx: {resp.status_code}")


class FactorsBatchUser(HttpUser):
    """Batch 10k-row /v1/batch/resolve user. Target: 100 rps."""

    weight = 1
    wait_time = between(0.5, 1.0)

    def on_start(self):
        self.token = os.getenv("GL_FACTORS_LOAD_TEST_TOKEN", "")
        if not self.token:
            raise SystemExit("GL_FACTORS_LOAD_TEST_TOKEN must be set")
        self.payload = _batch_payload()

    @tag("batch")
    @task
    def batch_resolve(self):
        with self.client.post(
            "/v1/batch/resolve",
            data=json.dumps(self.payload),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GreenLang-Load-Test": "1",
            },
            name="POST /v1/batch/resolve (10k rows)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 429:
                resp.success()
            elif resp.status_code >= 500:
                resp.failure(f"5xx: {resp.status_code}")


# ---------------------------------------------------------------------------
# Custom stats summary on test stop
# ---------------------------------------------------------------------------


@events.quitting.add_listener
def _print_summary(environment, **_kwargs):
    stats = environment.stats
    for endpoint in ("POST /v1/resolve", "POST /v1/batch/resolve (10k rows)"):
        entry = stats.get(endpoint, "POST")
        if entry is None:
            continue
        print(
            f"[summary] {endpoint}: "
            f"count={entry.num_requests} "
            f"p50={entry.get_response_time_percentile(0.50):.0f}ms "
            f"p95={entry.get_response_time_percentile(0.95):.0f}ms "
            f"p99={entry.get_response_time_percentile(0.99):.0f}ms "
            f"fail%={entry.fail_ratio * 100:.2f}"
        )
