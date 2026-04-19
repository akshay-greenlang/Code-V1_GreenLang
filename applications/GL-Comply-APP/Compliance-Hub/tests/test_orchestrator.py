# -*- coding: utf-8 -*-
"""Orchestrator + framework adapter integration tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from agents.orchestrator_agent import ComplianceOrchestrator
from schemas.models import ComplianceRequest, EntitySnapshot, FrameworkEnum
from services import registry
from greenlang.schemas.enums import ComplianceStatus


@pytest.fixture(autouse=True)
def _ensure_adapters_registered():
    # Importing services triggers adapter registration
    import services  # noqa: F401


def _base_request(frameworks, **overrides) -> ComplianceRequest:
    base = dict(
        entity=EntitySnapshot(
            entity_id="acme-de",
            legal_name="ACME GmbH",
            jurisdiction="DE",
            revenue_eur=75_000_000,
            employees=400,
        ),
        reporting_period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        reporting_period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        frameworks=frameworks,
        data_sources={
            "activities": [
                {
                    "activity_id": "a1",
                    "activity_type": "stationary_combustion",
                    "fuel_type": "diesel",
                    "quantity": "1000",
                    "unit": "gallons",
                    "year": 2024,
                }
            ]
        },
    )
    base.update(overrides)
    return ComplianceRequest(**base)


def test_all_10_adapters_registered():
    assert len(registry.available()) == 10


def test_orchestrator_runs_single_framework():
    req = _base_request([FrameworkEnum.GHG_PROTOCOL])
    report = asyncio.run(ComplianceOrchestrator().run(req))
    assert FrameworkEnum.GHG_PROTOCOL in report.results
    assert report.overall_status == ComplianceStatus.COMPLIANT


def test_orchestrator_runs_multiple_frameworks():
    frameworks = [
        FrameworkEnum.GHG_PROTOCOL,
        FrameworkEnum.CSRD,
        FrameworkEnum.CBAM,
        FrameworkEnum.ISO_14064,
        FrameworkEnum.SBTI,
    ]
    report = asyncio.run(ComplianceOrchestrator().run(_base_request(frameworks)))
    assert set(report.results.keys()) == set(frameworks)


def test_orchestrator_surfaces_eudr_pending_in_gap_analysis():
    report = asyncio.run(
        ComplianceOrchestrator().run(_base_request([FrameworkEnum.EUDR]))
    )
    eudr_result = report.results[FrameworkEnum.EUDR]
    assert eudr_result.compliance_status == ComplianceStatus.UNDER_REVIEW
    assert any(g["framework"] == "eudr" for g in report.gap_analysis)
    # Overall downgraded to partially_compliant when any adapter is under_review
    assert report.overall_status == ComplianceStatus.PARTIALLY_COMPLIANT


def test_orchestrator_aggregate_hash_deterministic():
    req = _base_request([FrameworkEnum.GHG_PROTOCOL])
    h1 = asyncio.run(ComplianceOrchestrator().run(req)).aggregate_provenance_hash
    h2 = asyncio.run(ComplianceOrchestrator().run(req)).aggregate_provenance_hash
    assert h1 == h2
    assert len(h1) == 64


def test_taxonomy_adapter_computes_alignment_ratio():
    req = _base_request(
        [FrameworkEnum.EU_TAXONOMY],
        data_sources={
            "taxonomy_activities": [
                {"turnover_eur": 50_000_000, "eligible": True, "aligned": True},
                {"turnover_eur": 50_000_000, "eligible": False, "aligned": False},
            ]
        },
    )
    report = asyncio.run(ComplianceOrchestrator().run(req))
    taxonomy = report.results[FrameworkEnum.EU_TAXONOMY]
    assert taxonomy.compliance_status == ComplianceStatus.COMPLIANT
    assert taxonomy.metrics["eligible_share"] == "0.5"
    assert taxonomy.metrics["aligned_share"] == "0.5"
