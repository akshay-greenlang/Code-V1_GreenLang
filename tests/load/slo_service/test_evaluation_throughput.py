# -*- coding: utf-8 -*-
"""
Evaluation Throughput Load Tests for SLO Service (OBS-005)

Tests the performance of SLO evaluation, budget calculation, recording
rule generation, and sustained load with many SLOs.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time

import pytest

from greenlang.infrastructure.slo_service.burn_rate import (
    calculate_burn_rate,
    evaluate_burn_rate_windows,
)
from greenlang.infrastructure.slo_service.error_budget import (
    calculate_error_budget,
)
from greenlang.infrastructure.slo_service.models import (
    SLI,
    SLIType,
    SLO,
    SLOWindow,
)
from greenlang.infrastructure.slo_service.recording_rules import (
    generate_all_recording_rules,
)
from greenlang.infrastructure.slo_service.slo_manager import SLOManager


def _generate_slos(count: int) -> list:
    """Generate a batch of SLOs for load testing."""
    slos = []
    for i in range(count):
        sli = SLI(
            name=f"sli_{i}",
            sli_type=SLIType.AVAILABILITY,
            good_query=f'http_requests_total{{code!~"5..",service="svc-{i}"}}',
            total_query=f'http_requests_total{{service="svc-{i}"}}',
        )
        slo = SLO(
            slo_id=f"load-test-slo-{i}",
            name=f"Load Test SLO {i}",
            service=f"service-{i % 10}",
            sli=sli,
            target=99.9 if i % 3 == 0 else (99.0 if i % 3 == 1 else 99.99),
            team=f"team-{i % 5}",
        )
        slos.append(slo)
    return slos


@pytest.mark.performance
class TestEvaluationThroughput:
    """Performance tests for SLO evaluation."""

    def test_100_slos_evaluated_under_30s(self):
        """100 SLOs can be evaluated (budget calculated) under 30 seconds."""
        slos = _generate_slos(100)

        start = time.monotonic()
        for slo in slos:
            calculate_error_budget(slo, current_sli=0.9995)
        elapsed = time.monotonic() - start

        assert elapsed < 30.0, f"100 SLO evaluations took {elapsed:.2f}s"

    def test_concurrent_evaluation_performance(self):
        """Budget calculations in bulk complete quickly."""
        slos = _generate_slos(500)
        sli_values = [0.999 + (i % 10) * 0.0001 for i in range(500)]

        start = time.monotonic()
        results = []
        for slo, sli_val in zip(slos, sli_values):
            budget = calculate_error_budget(slo, current_sli=sli_val)
            results.append(budget)
        elapsed = time.monotonic() - start

        assert len(results) == 500
        assert elapsed < 5.0, f"500 budget calculations took {elapsed:.2f}s"

    def test_budget_calculation_throughput(self):
        """10,000 budget calculations complete in under 10 seconds."""
        slo = _generate_slos(1)[0]

        start = time.monotonic()
        for i in range(10000):
            sli_val = 0.998 + (i % 20) * 0.0001
            calculate_error_budget(slo, current_sli=sli_val)
        elapsed = time.monotonic() - start

        throughput = 10000 / elapsed
        assert throughput > 1000, f"Throughput: {throughput:.0f} calcs/sec"

    def test_recording_rule_generation_performance(self):
        """Recording rule generation for 100 SLOs under 5 seconds."""
        slos = _generate_slos(100)

        start = time.monotonic()
        result = generate_all_recording_rules(slos)
        elapsed = time.monotonic() - start

        total_rules = sum(len(g["rules"]) for g in result["groups"])
        assert total_rules == 500  # 100 * 5 rules per SLO
        assert elapsed < 5.0, f"Rule generation took {elapsed:.2f}s"

    def test_sustained_evaluation_load(self):
        """Sustained evaluation loop for 1000 iterations."""
        slos = _generate_slos(10)

        start = time.monotonic()
        for iteration in range(1000):
            for slo in slos:
                sli_val = 0.999 + (iteration % 10) * 0.0001
                budget = calculate_error_budget(slo, current_sli=sli_val)
                # Also compute burn rate
                error_rate = 1.0 - sli_val
                calculate_burn_rate(error_rate, slo.error_budget_fraction)
        elapsed = time.monotonic() - start

        total_ops = 1000 * 10 * 2  # iterations * slos * (budget + burn_rate)
        throughput = total_ops / elapsed
        assert throughput > 1000, f"Sustained throughput: {throughput:.0f} ops/sec"
