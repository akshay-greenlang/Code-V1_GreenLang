# -*- coding: utf-8 -*-
"""Applicability rule tests."""

from __future__ import annotations

from schemas.models import ApplicabilityRequest, EntitySnapshot, FrameworkEnum
from services import applicability


def _req(**entity_kwargs):
    base = dict(
        entity_id="e1",
        legal_name="Test Corp",
        jurisdiction="DE",
        revenue_eur=10_000_000,
        employees=50,
    )
    base.update(entity_kwargs)
    return ApplicabilityRequest(entity=EntitySnapshot(**base), reporting_year=2026)


def test_baseline_frameworks_always_apply():
    result = applicability.evaluate(_req())
    assert FrameworkEnum.GHG_PROTOCOL in result.applicable_frameworks
    assert FrameworkEnum.ISO_14064 in result.applicable_frameworks


def test_csrd_threshold_triggers_for_large_eu_entity():
    result = applicability.evaluate(
        _req(jurisdiction="DE", revenue_eur=60_000_000, employees=300)
    )
    assert FrameworkEnum.CSRD in result.applicable_frameworks
    assert FrameworkEnum.EU_TAXONOMY in result.applicable_frameworks


def test_csrd_not_triggered_under_threshold():
    result = applicability.evaluate(
        _req(jurisdiction="DE", revenue_eur=20_000_000, employees=100)
    )
    assert FrameworkEnum.CSRD not in result.applicable_frameworks


def test_cbam_triggered_when_flag_set():
    result = applicability.evaluate(_req(imports_cbam_goods=True))
    assert FrameworkEnum.CBAM in result.applicable_frameworks


def test_eudr_triggered_when_flag_set():
    result = applicability.evaluate(_req(handles_eudr_commodities=True))
    assert FrameworkEnum.EUDR in result.applicable_frameworks


def test_sb253_triggered_for_california():
    result = applicability.evaluate(_req(jurisdiction="US-CA", operates_in_us_ca=True))
    assert FrameworkEnum.SB253 in result.applicable_frameworks


def test_sbti_tcfd_cdp_for_large_entity():
    result = applicability.evaluate(_req(revenue_eur=200_000_000))
    for fw in (FrameworkEnum.SBTI, FrameworkEnum.TCFD, FrameworkEnum.CDP):
        assert fw in result.applicable_frameworks


def test_rationale_covers_all_applicable_frameworks():
    result = applicability.evaluate(_req(imports_cbam_goods=True, revenue_eur=200_000_000))
    assert set(result.rationale.keys()) == set(result.applicable_frameworks)
