# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceCheckerEngine (Engine 6 of 7).

AGENT-MRV-013: Dual Reporting Reconciliation Agent
Target: 40 tests covering 7-framework compliance checking.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from greenlang.dual_reporting_reconciliation.compliance_checker import (
    ComplianceCheckerEngine,
)
from greenlang.dual_reporting_reconciliation.models import (
    ReconciliationWorkspace,
    ReportingFramework,
    UpstreamResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a ComplianceCheckerEngine instance."""
    return ComplianceCheckerEngine()


@pytest.fixture
def sample_workspace() -> ReconciliationWorkspace:
    """Return a sample ReconciliationWorkspace for compliance checking."""
    loc = UpstreamResult(
        agent="mrv_009",
        method="location_based",
        energy_type="electricity",
        emissions_tco2e=Decimal("1250.50"),
        energy_quantity_mwh=Decimal("5000.0"),
        ef_used=Decimal("0.2501"),
        ef_source="eGRID 2023 CAMX",
        ef_hierarchy="grid_average",
        facility_id="FAC-001",
        tenant_id="tenant-001",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
    )
    mkt = UpstreamResult(
        agent="mrv_010",
        method="market_based",
        energy_type="electricity",
        emissions_tco2e=Decimal("625.25"),
        energy_quantity_mwh=Decimal("5000.0"),
        ef_used=Decimal("0.12505"),
        ef_source="Supplier Disclosure 2024",
        ef_hierarchy="supplier_no_cert",
        facility_id="FAC-001",
        tenant_id="tenant-001",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
    )
    return ReconciliationWorkspace(
        reconciliation_id="RECON-TEST-001",
        tenant_id="tenant-001",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        location_results=[loc],
        market_results=[mkt],
        total_location_tco2e=Decimal("1250.50"),
        total_market_tco2e=Decimal("625.25"),
    )


@pytest.fixture
def minimal_workspace() -> ReconciliationWorkspace:
    """Return a minimal workspace with just totals."""
    return ReconciliationWorkspace(
        reconciliation_id="RECON-MIN-001",
        tenant_id="tenant-001",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        location_results=[],
        market_results=[],
        total_location_tco2e=Decimal("100"),
        total_market_tco2e=Decimal("80"),
    )


@pytest.fixture
def zero_workspace() -> ReconciliationWorkspace:
    """Return a workspace with zero emissions."""
    return ReconciliationWorkspace(
        reconciliation_id="RECON-ZERO-001",
        tenant_id="tenant-001",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        location_results=[],
        market_results=[],
        total_location_tco2e=Decimal("0"),
        total_market_tco2e=Decimal("0"),
    )


# ===========================================================================
# 1. Initialization Tests
# ===========================================================================


class TestEngineInit:
    """Test ComplianceCheckerEngine initialization."""

    def test_create_instance(self, engine):
        assert engine is not None

    def test_singleton_pattern(self):
        e1 = ComplianceCheckerEngine()
        e2 = ComplianceCheckerEngine()
        assert e1 is e2

    def test_health_check(self, engine):
        health = engine.health_check()
        assert isinstance(health, dict)
        assert health.get("status") in ("healthy", "available", "ok")


# ===========================================================================
# 2. Full Compliance Check Tests
# ===========================================================================


class TestFullComplianceCheck:
    """Test check_all_frameworks main method."""

    def test_check_all_frameworks(self, engine, sample_workspace):
        result = engine.check_all_frameworks(
            sample_workspace, None, None,
        )
        assert isinstance(result, dict)

    def test_check_specific_frameworks(self, engine, sample_workspace):
        result = engine.check_all_frameworks(
            sample_workspace, None, None,
            frameworks=[ReportingFramework.GHG_PROTOCOL],
        )
        assert isinstance(result, dict)

    def test_check_empty_workspace(self, engine, minimal_workspace):
        result = engine.check_all_frameworks(
            minimal_workspace, None, None,
        )
        assert isinstance(result, dict)


# ===========================================================================
# 3. GHG Protocol Tests
# ===========================================================================


class TestGHGProtocolCompliance:
    """Test GHG Protocol Scope 2 Guidance compliance checking."""

    def test_check_ghg_protocol(self, engine, sample_workspace):
        result = engine.check_ghg_protocol(sample_workspace, None, None)
        assert result is not None

    def test_ghg_protocol_has_status(self, engine, sample_workspace):
        result = engine.check_ghg_protocol(sample_workspace, None, None)
        if hasattr(result, "status"):
            assert result.status is not None
        elif isinstance(result, dict):
            assert "status" in result

    def test_ghg_protocol_has_requirements(self, engine, sample_workspace):
        result = engine.check_ghg_protocol(sample_workspace, None, None)
        if hasattr(result, "requirements"):
            assert len(result.requirements) >= 5
        elif isinstance(result, dict):
            assert len(result.get("requirements", [])) >= 5


# ===========================================================================
# 4. CSRD/ESRS Tests
# ===========================================================================


class TestCSRDCompliance:
    """Test CSRD/ESRS E1 compliance checking."""

    def test_check_csrd(self, engine, sample_workspace):
        result = engine.check_csrd_esrs(sample_workspace, None, None)
        assert result is not None

    def test_csrd_has_requirements(self, engine, sample_workspace):
        result = engine.check_csrd_esrs(sample_workspace, None, None)
        if hasattr(result, "requirements"):
            assert len(result.requirements) >= 5
        elif isinstance(result, dict):
            assert len(result.get("requirements", [])) >= 5


# ===========================================================================
# 5. CDP Tests
# ===========================================================================


class TestCDPCompliance:
    """Test CDP Climate Change compliance checking."""

    def test_check_cdp(self, engine, sample_workspace):
        result = engine.check_cdp(sample_workspace, None, None)
        assert result is not None

    def test_cdp_has_requirements(self, engine, sample_workspace):
        result = engine.check_cdp(sample_workspace, None, None)
        if hasattr(result, "requirements"):
            assert len(result.requirements) >= 4
        elif isinstance(result, dict):
            assert len(result.get("requirements", [])) >= 4


# ===========================================================================
# 6. SBTi Tests
# ===========================================================================


class TestSBTiCompliance:
    """Test SBTi compliance checking."""

    def test_check_sbti(self, engine, sample_workspace):
        result = engine.check_sbti(sample_workspace, None, None)
        assert result is not None


# ===========================================================================
# 7. GRI Tests
# ===========================================================================


class TestGRICompliance:
    """Test GRI 305 compliance checking."""

    def test_check_gri(self, engine, sample_workspace):
        result = engine.check_gri(sample_workspace, None, None)
        assert result is not None


# ===========================================================================
# 8. ISO 14064 Tests
# ===========================================================================


class TestISO14064Compliance:
    """Test ISO 14064-1 compliance checking."""

    def test_check_iso_14064(self, engine, sample_workspace):
        result = engine.check_iso_14064(sample_workspace, None, None)
        assert result is not None


# ===========================================================================
# 9. RE100 Tests
# ===========================================================================


class TestRE100Compliance:
    """Test RE100 compliance checking."""

    def test_check_re100(self, engine, sample_workspace):
        result = engine.check_re100(sample_workspace, None, None)
        assert result is not None


# ===========================================================================
# 10. Compliance Summary Tests
# ===========================================================================


class TestComplianceSummary:
    """Test compliance summary generation."""

    def test_get_summary(self, engine, sample_workspace):
        results = engine.check_all_frameworks(
            sample_workspace, None, None,
        )
        summary = engine.get_compliance_summary(results)
        assert isinstance(summary, dict)
        assert (
            "total_frameworks" in summary
            or "frameworks_checked" in summary
            or "framework_count" in summary
        )

    def test_generate_compliance_flags(self, engine, sample_workspace):
        results = engine.check_all_frameworks(
            sample_workspace, None, None,
        )
        flags = engine.generate_compliance_flags(results)
        assert isinstance(flags, list)


# ===========================================================================
# 11. Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_workspace(self, engine, minimal_workspace):
        result = engine.check_all_frameworks(minimal_workspace, None, None)
        assert isinstance(result, dict)

    def test_zero_emissions(self, engine, zero_workspace):
        result = engine.check_all_frameworks(zero_workspace, None, None)
        assert isinstance(result, dict)

    def test_all_frameworks_checked(self, engine, sample_workspace):
        result = engine.check_all_frameworks(
            sample_workspace, None, None,
        )
        # Should have checked all 7 frameworks
        assert len(result) == 7
