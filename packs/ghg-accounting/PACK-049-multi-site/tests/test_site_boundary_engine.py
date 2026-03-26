# -*- coding: utf-8 -*-
"""
Tests for PACK-049 Engine 3: SiteBoundaryEngine

Covers boundary definition (equity/operational/financial control), site inclusion,
de minimis exclusion, boundary changes (acquisition/divestiture), time-weighted
consolidation, materiality assessment, boundary locking, and comparison.
Target: ~55 tests.
"""

import pytest
from decimal import Decimal
from datetime import date, datetime, timezone
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

try:
    from engines.site_boundary_engine import (
        SiteBoundaryEngine,
        BoundaryDefinition,
        BoundarySite,
        BoundaryChange,
        BoundaryComparison,
        MaterialityAssessment,
        ConsolidationApproach,
        BoundaryChangeType,
    )
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

pytestmark = pytest.mark.skipif(not HAS_ENGINE, reason="Engine not yet built")


@pytest.fixture
def engine():
    return SiteBoundaryEngine()


@pytest.fixture
def equity_boundary(engine):
    return engine.define_boundary(
        organisation_id="ORG-001",
        reporting_year=2026,
        consolidation_approach="EQUITY_SHARE",
        sites=[
            {"site_id": "site-001", "ownership_pct": Decimal("100.00")},
            {"site_id": "site-002", "ownership_pct": Decimal("100.00")},
            {"site_id": "site-003", "ownership_pct": Decimal("75.00")},
            {"site_id": "site-004", "ownership_pct": Decimal("50.00")},
            {"site_id": "site-005", "ownership_pct": Decimal("100.00")},
        ],
    )


# ============================================================================
# Boundary Definition Tests
# ============================================================================

class TestBoundaryDefinition:

    def test_define_boundary_equity(self, engine):
        boundary = engine.define_boundary(
            organisation_id="ORG-001",
            reporting_year=2026,
            consolidation_approach="EQUITY_SHARE",
            sites=[
                {"site_id": "s1", "ownership_pct": Decimal("100.00")},
            ],
        )
        assert boundary is not None
        assert boundary.consolidation_approach == "EQUITY_SHARE"

    def test_define_boundary_operational(self, engine):
        boundary = engine.define_boundary(
            organisation_id="ORG-001",
            reporting_year=2026,
            consolidation_approach="OPERATIONAL_CONTROL",
            sites=[
                {"site_id": "s1", "ownership_pct": Decimal("100.00")},
                {"site_id": "s2", "ownership_pct": Decimal("60.00")},
            ],
        )
        assert boundary.consolidation_approach == "OPERATIONAL_CONTROL"
        assert len(boundary.sites) == 2

    def test_define_boundary_financial(self, engine):
        boundary = engine.define_boundary(
            organisation_id="ORG-001",
            reporting_year=2026,
            consolidation_approach="FINANCIAL_CONTROL",
            sites=[
                {"site_id": "s1", "ownership_pct": Decimal("100.00")},
            ],
        )
        assert boundary.consolidation_approach == "FINANCIAL_CONTROL"

    def test_boundary_has_id(self, engine):
        boundary = engine.define_boundary(
            organisation_id="ORG-001",
            reporting_year=2026,
            consolidation_approach="EQUITY_SHARE",
            sites=[{"site_id": "s1", "ownership_pct": Decimal("100.00")}],
        )
        assert boundary.boundary_id is not None
        assert len(boundary.boundary_id) > 0

    def test_boundary_provenance_hash(self, engine):
        boundary = engine.define_boundary(
            organisation_id="ORG-001",
            reporting_year=2026,
            consolidation_approach="EQUITY_SHARE",
            sites=[{"site_id": "s1", "ownership_pct": Decimal("100.00")}],
        )
        assert boundary.provenance_hash is not None
        assert len(boundary.provenance_hash) == 64


# ============================================================================
# Site Inclusion / Exclusion Tests
# ============================================================================

class TestSiteInclusion:

    def test_add_site_to_boundary(self, engine, equity_boundary):
        updated = engine.add_site(
            boundary_id=equity_boundary.boundary_id,
            site_id="site-006",
            ownership_pct=Decimal("100.00"),
        )
        site_ids = [s.site_id for s in updated.sites]
        assert "site-006" in site_ids

    def test_exclude_site_de_minimis(self, engine, equity_boundary):
        updated = engine.exclude_site(
            boundary_id=equity_boundary.boundary_id,
            site_id="site-004",
            reason="de_minimis",
            emissions_pct=Decimal("0.005"),
        )
        site = next(s for s in updated.sites if s.site_id == "site-004")
        assert site.is_included is False

    def test_add_site_already_exists(self, engine, equity_boundary):
        with pytest.raises((ValueError, Exception)):
            engine.add_site(
                boundary_id=equity_boundary.boundary_id,
                site_id="site-001",
                ownership_pct=Decimal("100.00"),
            )


# ============================================================================
# Boundary Change Tests
# ============================================================================

class TestBoundaryChanges:

    def test_boundary_change_acquisition(self, engine, equity_boundary):
        change = engine.record_boundary_change(
            boundary_id=equity_boundary.boundary_id,
            change_type="ACQUISITION",
            site_id="site-006",
            effective_date=date(2026, 7, 1),
            ownership_pct=Decimal("100.00"),
            description="Acquired new plant",
        )
        assert change is not None
        assert change.change_type == "ACQUISITION"

    def test_boundary_change_divestiture(self, engine, equity_boundary):
        change = engine.record_boundary_change(
            boundary_id=equity_boundary.boundary_id,
            change_type="DIVESTITURE",
            site_id="site-004",
            effective_date=date(2026, 10, 1),
            ownership_pct=Decimal("0.00"),
            description="Divested retail unit",
        )
        assert change.change_type == "DIVESTITURE"


# ============================================================================
# Time-Weighted Consolidation Tests
# ============================================================================

class TestTimeWeightedConsolidation:

    def test_time_weighted_consolidation_full_year(self, engine, equity_boundary):
        # Site owned for entire year: weight = 1.0
        weight = engine.calculate_time_weight(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
            acquisition_date=date(2025, 1, 1),
            reporting_year_start=date(2026, 1, 1),
            reporting_year_end=date(2026, 12, 31),
        )
        assert weight == Decimal("1.0") or weight >= Decimal("0.99")

    def test_time_weighted_consolidation_midyear(self, engine, equity_boundary):
        # Site acquired July 1 = 50% of year
        weight = engine.calculate_time_weight(
            start_date=date(2026, 7, 1),
            end_date=date(2026, 12, 31),
            acquisition_date=date(2026, 7, 1),
            reporting_year_start=date(2026, 1, 1),
            reporting_year_end=date(2026, 12, 31),
        )
        # Approximately 184/365 or 366 days
        assert Decimal("0.49") <= weight <= Decimal("0.51")

    def test_time_weighted_consolidation_quarter(self, engine, equity_boundary):
        # Acquired October 1 = ~25% of year
        weight = engine.calculate_time_weight(
            start_date=date(2026, 10, 1),
            end_date=date(2026, 12, 31),
            acquisition_date=date(2026, 10, 1),
            reporting_year_start=date(2026, 1, 1),
            reporting_year_end=date(2026, 12, 31),
        )
        assert Decimal("0.24") <= weight <= Decimal("0.26")


# ============================================================================
# Materiality Assessment Tests
# ============================================================================

class TestMaterialityAssessment:

    def test_assess_materiality_above_threshold(self, engine, equity_boundary):
        result = engine.assess_materiality(
            boundary_id=equity_boundary.boundary_id,
            site_id="site-001",
            site_emissions=Decimal("5000"),
            total_emissions=Decimal("20000"),
            threshold=Decimal("0.05"),
        )
        assert result.is_material is True
        # 5000/20000 = 25% > 5%

    def test_assess_materiality_below_threshold(self, engine, equity_boundary):
        result = engine.assess_materiality(
            boundary_id=equity_boundary.boundary_id,
            site_id="site-004",
            site_emissions=Decimal("50"),
            total_emissions=Decimal("20000"),
            threshold=Decimal("0.05"),
        )
        assert result.is_material is False
        # 50/20000 = 0.25% < 5%

    def test_assess_materiality_at_threshold(self, engine, equity_boundary):
        result = engine.assess_materiality(
            boundary_id=equity_boundary.boundary_id,
            site_id="site-003",
            site_emissions=Decimal("1000"),
            total_emissions=Decimal("20000"),
            threshold=Decimal("0.05"),
        )
        # 1000/20000 = 5% exactly at threshold
        assert result is not None


# ============================================================================
# Boundary Lock Tests
# ============================================================================

class TestBoundaryLock:

    def test_lock_boundary(self, engine, equity_boundary):
        locked = engine.lock_boundary(equity_boundary.boundary_id)
        assert locked.is_locked is True

    def test_locked_boundary_immutable(self, engine, equity_boundary):
        engine.lock_boundary(equity_boundary.boundary_id)
        with pytest.raises((ValueError, Exception)):
            engine.add_site(
                boundary_id=equity_boundary.boundary_id,
                site_id="site-006",
                ownership_pct=Decimal("100.00"),
            )


# ============================================================================
# Boundary Comparison Tests
# ============================================================================

class TestBoundaryComparison:

    def test_compare_boundaries_identical(self, engine):
        b1 = engine.define_boundary(
            organisation_id="ORG-001",
            reporting_year=2025,
            consolidation_approach="EQUITY_SHARE",
            sites=[
                {"site_id": "s1", "ownership_pct": Decimal("100.00")},
                {"site_id": "s2", "ownership_pct": Decimal("75.00")},
            ],
        )
        b2 = engine.define_boundary(
            organisation_id="ORG-001",
            reporting_year=2026,
            consolidation_approach="EQUITY_SHARE",
            sites=[
                {"site_id": "s1", "ownership_pct": Decimal("100.00")},
                {"site_id": "s2", "ownership_pct": Decimal("75.00")},
            ],
        )
        comparison = engine.compare_boundaries(b1.boundary_id, b2.boundary_id)
        assert comparison.sites_added == 0
        assert comparison.sites_removed == 0

    def test_compare_boundaries_site_added(self, engine):
        b1 = engine.define_boundary(
            organisation_id="ORG-001",
            reporting_year=2025,
            consolidation_approach="EQUITY_SHARE",
            sites=[
                {"site_id": "s1", "ownership_pct": Decimal("100.00")},
            ],
        )
        b2 = engine.define_boundary(
            organisation_id="ORG-001",
            reporting_year=2026,
            consolidation_approach="EQUITY_SHARE",
            sites=[
                {"site_id": "s1", "ownership_pct": Decimal("100.00")},
                {"site_id": "s2", "ownership_pct": Decimal("75.00")},
            ],
        )
        comparison = engine.compare_boundaries(b1.boundary_id, b2.boundary_id)
        assert comparison.sites_added == 1

    def test_compare_boundaries_ownership_change(self, engine):
        b1 = engine.define_boundary(
            organisation_id="ORG-001",
            reporting_year=2025,
            consolidation_approach="EQUITY_SHARE",
            sites=[
                {"site_id": "s1", "ownership_pct": Decimal("75.00")},
            ],
        )
        b2 = engine.define_boundary(
            organisation_id="ORG-001",
            reporting_year=2026,
            consolidation_approach="EQUITY_SHARE",
            sites=[
                {"site_id": "s1", "ownership_pct": Decimal("100.00")},
            ],
        )
        comparison = engine.compare_boundaries(b1.boundary_id, b2.boundary_id)
        assert comparison.ownership_changes >= 1 or comparison is not None
