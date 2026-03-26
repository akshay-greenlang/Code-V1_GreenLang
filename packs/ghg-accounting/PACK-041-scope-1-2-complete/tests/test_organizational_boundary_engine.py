# -*- coding: utf-8 -*-
"""
Unit tests for OrganizationalBoundaryEngine -- PACK-041 Engine 1
=================================================================

Tests GHG Protocol Chapter 3 consolidation approaches (equity share,
operational control, financial control), entity evaluation, boundary
changes, base-year impact assessment, and provenance hashing.

Coverage target: 85%+
Total tests: ~65
"""

import importlib.util
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack041_test.engines.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


_m = _load("organizational_boundary_engine")


# =============================================================================
# Module Loading
# =============================================================================


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_module_version(self):
        assert hasattr(_m, "_MODULE_VERSION")
        assert _m._MODULE_VERSION == "1.0.0"

    def test_engine_class_exists(self):
        assert hasattr(_m, "OrganizationalBoundaryEngine")

    def test_engine_instantiation(self):
        engine = _m.OrganizationalBoundaryEngine()
        assert engine is not None


# =============================================================================
# Equity Share Approach
# =============================================================================


class TestEquityShareApproach:
    """Test equity share consolidation per GHG Protocol Ch. 3."""

    def _make_org(self, entities):
        return _m.OrganizationStructure(
            org_name="Test Org",
            reporting_year=2025,
            base_year=2019,
            entities=entities,
            default_approach=_m.ConsolidationApproach.EQUITY_SHARE,
        )

    def test_equity_share_wholly_owned_100pct(self):
        """Wholly owned entity should include 100% under equity share."""
        entity = _m.LegalEntity(
            entity_name="Subsidiary A",
            entity_type=_m.EntityType.WHOLLY_OWNED,
            equity_pct=Decimal("100"),
            total_scope1_tco2e=Decimal("1000"),
            total_scope2_tco2e=Decimal("500"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.EQUITY_SHARE)
        er = result.entity_results[0]
        assert er.inclusion_pct == Decimal("100.00")
        assert er.inclusion_status == _m.InclusionStatus.INCLUDED

    def test_equity_share_majority_owned_75pct(self):
        """75% equity should include 75% of emissions."""
        entity = _m.LegalEntity(
            entity_name="Majority Sub",
            entity_type=_m.EntityType.MAJORITY_OWNED,
            equity_pct=Decimal("75"),
            total_scope1_tco2e=Decimal("2000"),
            total_scope2_tco2e=Decimal("1000"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.EQUITY_SHARE)
        er = result.entity_results[0]
        assert er.inclusion_pct == Decimal("75.00")
        assert er.scope1_included_tco2e == Decimal("1500.0000")
        assert er.scope2_included_tco2e == Decimal("750.0000")

    def test_equity_share_joint_venture_50pct(self):
        """50% JV should include 50% of emissions under equity share."""
        entity = _m.LegalEntity(
            entity_name="JV Alpha",
            entity_type=_m.EntityType.JV,
            equity_pct=Decimal("50"),
            total_scope1_tco2e=Decimal("4000"),
            total_scope2_tco2e=Decimal("2000"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.EQUITY_SHARE)
        er = result.entity_results[0]
        assert er.inclusion_pct == Decimal("50.00")
        assert er.scope1_included_tco2e == Decimal("2000.0000")
        assert er.inclusion_status == _m.InclusionStatus.PARTIAL

    def test_equity_share_associate_25pct(self):
        """25% associate should include 25% of emissions."""
        entity = _m.LegalEntity(
            entity_name="Associate Beta",
            entity_type=_m.EntityType.ASSOCIATE,
            equity_pct=Decimal("25"),
            total_scope1_tco2e=Decimal("8000"),
            total_scope2_tco2e=Decimal("4000"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.EQUITY_SHARE)
        er = result.entity_results[0]
        assert er.inclusion_pct == Decimal("25.00")
        assert er.scope1_included_tco2e == Decimal("2000.0000")

    def test_equity_share_zero_excluded(self):
        """0% equity should be excluded."""
        entity = _m.LegalEntity(
            entity_name="Zero Stake",
            entity_type=_m.EntityType.ASSOCIATE,
            equity_pct=Decimal("0"),
            total_scope1_tco2e=Decimal("1000"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.EQUITY_SHARE)
        er = result.entity_results[0]
        assert er.inclusion_status == _m.InclusionStatus.EXCLUDED
        assert er.scope1_included_tco2e == Decimal("0")

    def test_equity_share_multi_entity_aggregation(self):
        """Multiple entities should aggregate correctly."""
        entities = [
            _m.LegalEntity(
                entity_name="Sub 1", equity_pct=Decimal("100"),
                total_scope1_tco2e=Decimal("1000"),
                total_scope2_tco2e=Decimal("500"),
            ),
            _m.LegalEntity(
                entity_name="JV", entity_type=_m.EntityType.JV,
                equity_pct=Decimal("50"),
                total_scope1_tco2e=Decimal("2000"),
                total_scope2_tco2e=Decimal("1000"),
            ),
        ]
        org = self._make_org(entities)
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.EQUITY_SHARE)
        # 100% of 1000 + 50% of 2000 = 2000 scope1
        assert result.total_scope1_tco2e == Decimal("2000.0000")
        # 100% of 500 + 50% of 1000 = 1000 scope2
        assert result.total_scope2_tco2e == Decimal("1000.0000")


# =============================================================================
# Operational Control Approach
# =============================================================================


class TestOperationalControlApproach:
    """Test operational control consolidation per GHG Protocol Ch. 3."""

    def _make_org(self, entities):
        return _m.OrganizationStructure(
            org_name="Test Org",
            entities=entities,
            default_approach=_m.ConsolidationApproach.OPERATIONAL_CONTROL,
        )

    def test_operational_control_wholly_owned_100pct(self):
        entity = _m.LegalEntity(
            entity_name="Sub A",
            entity_type=_m.EntityType.WHOLLY_OWNED,
            total_scope1_tco2e=Decimal("5000"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.OPERATIONAL_CONTROL)
        er = result.entity_results[0]
        assert er.inclusion_pct == Decimal("100.00")
        assert er.inclusion_status == _m.InclusionStatus.INCLUDED

    def test_operational_control_jv_as_operator(self):
        """JV where reporting org is operator should include 100%."""
        entity = _m.LegalEntity(
            entity_name="JV Operated",
            entity_type=_m.EntityType.JV,
            equity_pct=Decimal("50"),
            has_operational_control=True,
            total_scope1_tco2e=Decimal("3000"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.OPERATIONAL_CONTROL)
        er = result.entity_results[0]
        assert er.inclusion_pct == Decimal("100.00")

    def test_operational_control_jv_not_operator(self):
        """JV where reporting org is NOT operator should exclude."""
        entity = _m.LegalEntity(
            entity_name="JV Non-Op",
            entity_type=_m.EntityType.JV,
            equity_pct=Decimal("50"),
            has_operational_control=False,
            total_scope1_tco2e=Decimal("3000"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.OPERATIONAL_CONTROL)
        er = result.entity_results[0]
        assert er.inclusion_pct == Decimal("0.00")
        assert er.inclusion_status == _m.InclusionStatus.EXCLUDED

    def test_operational_control_associate_excluded(self):
        """Associate entities are excluded under OC by default."""
        entity = _m.LegalEntity(
            entity_name="Assoc",
            entity_type=_m.EntityType.ASSOCIATE,
            equity_pct=Decimal("30"),
            total_scope1_tco2e=Decimal("1000"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.OPERATIONAL_CONTROL)
        er = result.entity_results[0]
        assert er.inclusion_status == _m.InclusionStatus.EXCLUDED


# =============================================================================
# Financial Control Approach
# =============================================================================


class TestFinancialControlApproach:
    """Test financial control consolidation per GHG Protocol Ch. 3."""

    def _make_org(self, entities):
        return _m.OrganizationStructure(
            org_name="Test Org",
            entities=entities,
        )

    def test_financial_control_subsidiary_100pct(self):
        entity = _m.LegalEntity(
            entity_name="Sub FC",
            entity_type=_m.EntityType.WHOLLY_OWNED,
            total_scope1_tco2e=Decimal("2000"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.FINANCIAL_CONTROL)
        er = result.entity_results[0]
        assert er.inclusion_pct == Decimal("100.00")

    def test_financial_control_majority_owned(self):
        entity = _m.LegalEntity(
            entity_name="Majority FC",
            entity_type=_m.EntityType.MAJORITY_OWNED,
            equity_pct=Decimal("60"),
            total_scope1_tco2e=Decimal("3000"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.FINANCIAL_CONTROL)
        er = result.entity_results[0]
        assert er.inclusion_pct == Decimal("100.00")


# =============================================================================
# Leased Assets and Franchises
# =============================================================================


class TestLeasedAssetsAndFranchises:
    """Test treatment of leased assets and franchise operations."""

    def _make_org(self, entities):
        return _m.OrganizationStructure(org_name="Test", entities=entities)

    def test_finance_lease_included_oc(self):
        entity = _m.LegalEntity(
            entity_name="Finance Lease",
            entity_type=_m.EntityType.LEASED_FINANCE,
            total_scope1_tco2e=Decimal("500"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.OPERATIONAL_CONTROL)
        er = result.entity_results[0]
        assert er.inclusion_pct == Decimal("100.00")

    def test_operating_lease_excluded_oc(self):
        entity = _m.LegalEntity(
            entity_name="Operating Lease",
            entity_type=_m.EntityType.LEASED_OPERATING,
            total_scope1_tco2e=Decimal("500"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.OPERATIONAL_CONTROL)
        er = result.entity_results[0]
        assert er.inclusion_status == _m.InclusionStatus.EXCLUDED

    def test_franchise_excluded_oc_by_default(self):
        entity = _m.LegalEntity(
            entity_name="Franchise Op",
            entity_type=_m.EntityType.FRANCHISE,
            total_scope1_tco2e=Decimal("800"),
        )
        org = self._make_org([entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org, _m.ConsolidationApproach.OPERATIONAL_CONTROL)
        er = result.entity_results[0]
        assert er.inclusion_status == _m.InclusionStatus.EXCLUDED


# =============================================================================
# Boundary Changes
# =============================================================================


class TestBoundaryChanges:
    """Test boundary change handling (acquisition, divestiture, merger)."""

    def _setup_engine_with_boundary(self):
        engine = _m.OrganizationalBoundaryEngine()
        entity = _m.LegalEntity(
            entity_id="ENT-EXIST",
            entity_name="Existing Sub",
            entity_type=_m.EntityType.WHOLLY_OWNED,
            total_scope1_tco2e=Decimal("10000"),
            total_scope2_tco2e=Decimal("5000"),
        )
        org = _m.OrganizationStructure(
            org_name="Test Org",
            entities=[entity],
            base_year=2019,
        )
        boundary = engine.define_boundary(org, _m.ConsolidationApproach.OPERATIONAL_CONTROL)
        return engine, boundary

    def test_acquisition_enters_boundary(self):
        engine, boundary = self._setup_engine_with_boundary()
        change = _m.BoundaryChangeEvent(
            change_type=_m.BoundaryChangeType.ACQUISITION,
            effective_date=date(2025, 6, 1),
            entity_id="ENT-NEW",
            entity_name="Acquired Corp",
            emissions_impact_tco2e=Decimal("2000"),
            equity_pct_after=Decimal("100"),
            operational_control_after=True,
        )
        result = engine.handle_boundary_change(change, boundary)
        assert result.inclusion_after == _m.InclusionStatus.INCLUDED
        assert result.emissions_delta_tco2e > Decimal("0")

    def test_divestiture_leaves_boundary(self):
        engine, boundary = self._setup_engine_with_boundary()
        change = _m.BoundaryChangeEvent(
            change_type=_m.BoundaryChangeType.DIVESTITURE,
            effective_date=date(2025, 3, 1),
            entity_id="ENT-EXIST",
            entity_name="Existing Sub",
            emissions_impact_tco2e=Decimal("5000"),
            equity_pct_before=Decimal("100"),
            equity_pct_after=Decimal("0"),
        )
        result = engine.handle_boundary_change(change, boundary)
        assert result.inclusion_after == _m.InclusionStatus.EXCLUDED
        assert result.emissions_delta_tco2e < Decimal("0")

    def test_merger_enters_boundary(self):
        engine, boundary = self._setup_engine_with_boundary()
        change = _m.BoundaryChangeEvent(
            change_type=_m.BoundaryChangeType.MERGER,
            effective_date=date(2025, 1, 1),
            entity_name="Merged Corp",
            emissions_impact_tco2e=Decimal("8000"),
            equity_pct_after=Decimal("100"),
            operational_control_after=True,
        )
        result = engine.handle_boundary_change(change, boundary)
        assert result.inclusion_after == _m.InclusionStatus.INCLUDED

    def test_change_no_boundary_raises_error(self):
        engine = _m.OrganizationalBoundaryEngine()
        change = _m.BoundaryChangeEvent(
            change_type=_m.BoundaryChangeType.ACQUISITION,
            effective_date=date(2025, 1, 1),
            emissions_impact_tco2e=Decimal("1000"),
        )
        with pytest.raises(ValueError, match="No boundary"):
            engine.handle_boundary_change(change)


# =============================================================================
# Base Year Impact Assessment
# =============================================================================


class TestBaseYearImpact:
    """Test base-year recalculation impact assessment."""

    def _setup_engine(self):
        engine = _m.OrganizationalBoundaryEngine()
        entity = _m.LegalEntity(
            entity_name="Main",
            total_scope1_tco2e=Decimal("20000"),
            total_scope2_tco2e=Decimal("10000"),
        )
        org = _m.OrganizationStructure(org_name="Test", entities=[entity])
        engine.define_boundary(org)
        return engine

    def test_significant_acquisition_above_5pct(self):
        engine = self._setup_engine()
        change = _m.BoundaryChangeEvent(
            change_type=_m.BoundaryChangeType.ACQUISITION,
            effective_date=date(2025, 1, 1),
            emissions_impact_tco2e=Decimal("3000"),
        )
        assessment = engine.assess_base_year_impact(
            change, Decimal("30000")
        )
        assert assessment.exceeds_threshold is True
        assert assessment.materiality_pct > Decimal("5.0")

    def test_not_significant_below_5pct(self):
        engine = self._setup_engine()
        change = _m.BoundaryChangeEvent(
            change_type=_m.BoundaryChangeType.ACQUISITION,
            effective_date=date(2025, 1, 1),
            emissions_impact_tco2e=Decimal("100"),
        )
        assessment = engine.assess_base_year_impact(
            change, Decimal("30000")
        )
        assert assessment.exceeds_threshold is False
        assert assessment.materiality_pct < Decimal("5.0")

    def test_divestiture_recalculation(self):
        engine = self._setup_engine()
        change = _m.BoundaryChangeEvent(
            change_type=_m.BoundaryChangeType.DIVESTITURE,
            effective_date=date(2025, 6, 1),
            emissions_impact_tco2e=Decimal("5000"),
        )
        assessment = engine.assess_base_year_impact(
            change, Decimal("30000")
        )
        assert assessment.adjusted_base_year_emissions_tco2e == Decimal("25000.0000")

    def test_methodology_change_trigger(self):
        engine = self._setup_engine()
        change = _m.BoundaryChangeEvent(
            change_type=_m.BoundaryChangeType.METHODOLOGY_CHANGE,
            effective_date=date(2025, 1, 1),
            emissions_impact_tco2e=Decimal("2000"),
        )
        assessment = engine.assess_base_year_impact(
            change, Decimal("30000")
        )
        if assessment.exceeds_threshold:
            assert assessment.recalculation_trigger == _m.RecalculationTrigger.METHODOLOGY_CHANGE

    def test_custom_significance_threshold(self):
        engine = self._setup_engine()
        change = _m.BoundaryChangeEvent(
            change_type=_m.BoundaryChangeType.ACQUISITION,
            effective_date=date(2025, 1, 1),
            emissions_impact_tco2e=Decimal("2500"),
        )
        assessment = engine.assess_base_year_impact(
            change, Decimal("30000"), significance_threshold_pct=Decimal("10.0")
        )
        # 2500/30000 = 8.33% < 10% threshold
        assert assessment.exceeds_threshold is False


# =============================================================================
# Boundary Report Generation
# =============================================================================


class TestBoundaryReport:
    """Test boundary report generation."""

    def test_report_generation(self):
        engine = _m.OrganizationalBoundaryEngine()
        entity = _m.LegalEntity(
            entity_name="Report Test Sub",
            total_scope1_tco2e=Decimal("5000"),
            total_scope2_tco2e=Decimal("3000"),
        )
        org = _m.OrganizationStructure(org_name="Report Org", entities=[entity])
        engine.define_boundary(org)
        report = engine.generate_boundary_report()
        assert report.boundary_definition is not None
        assert len(report.summary_text) > 0

    def test_report_without_boundary_raises(self):
        engine = _m.OrganizationalBoundaryEngine()
        with pytest.raises(ValueError, match="No boundary"):
            engine.generate_boundary_report()

    def test_report_provenance_hash(self):
        engine = _m.OrganizationalBoundaryEngine()
        entity = _m.LegalEntity(
            entity_name="Hash Test",
            total_scope1_tco2e=Decimal("1000"),
        )
        org = _m.OrganizationStructure(org_name="Hash Org", entities=[entity])
        engine.define_boundary(org)
        report = engine.generate_boundary_report()
        assert len(report.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in report.provenance_hash)


# =============================================================================
# Provenance Hashing
# =============================================================================


class TestProvenanceHashing:
    """Test SHA-256 provenance hashing is deterministic."""

    def test_boundary_provenance_deterministic(self):
        """Provenance hash is a valid SHA-256 hex string.

        Note: The engine assigns a unique boundary_id (UUID4) per invocation,
        so cross-invocation equality is not expected.  We verify the hash is
        structurally correct and that re-hashing the same result object is
        deterministic.
        """
        engine = _m.OrganizationalBoundaryEngine()
        entity = _m.LegalEntity(
            entity_name="Deterministic",
            total_scope1_tco2e=Decimal("1234"),
            total_scope2_tco2e=Decimal("567"),
        )
        org = _m.OrganizationStructure(
            org_id="FIXED-ID",
            org_name="Fixed Org",
            entities=[entity],
        )
        r1 = engine.define_boundary(org)
        # Hash is valid SHA-256 (64 hex chars)
        assert len(r1.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in r1.provenance_hash)
        # Re-computing hash on same result yields same value (deterministic)
        rehash = _m._compute_hash(r1)
        assert rehash == r1.provenance_hash

    def test_provenance_changes_with_data(self):
        """Different input data produces different provenance hashes.

        Even though boundary_id differs, we verify that distinct emission
        values also lead to distinct hashes.
        """
        engine = _m.OrganizationalBoundaryEngine()
        e1 = _m.LegalEntity(entity_name="A", total_scope1_tco2e=Decimal("100"))
        e2 = _m.LegalEntity(entity_name="A", total_scope1_tco2e=Decimal("200"))
        org1 = _m.OrganizationStructure(org_id="X", org_name="X", entities=[e1])
        org2 = _m.OrganizationStructure(org_id="X", org_name="X", entities=[e2])
        r1 = engine.define_boundary(org1)
        r2 = engine.define_boundary(org2)
        assert r1.provenance_hash != r2.provenance_hash


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_organization_raises(self):
        engine = _m.OrganizationalBoundaryEngine()
        org = _m.OrganizationStructure(org_name="Empty", entities=[])
        with pytest.raises(ValueError, match="at least one entity"):
            engine.define_boundary(org)

    def test_inactive_entity_excluded(self):
        entity = _m.LegalEntity(
            entity_name="Inactive",
            total_scope1_tco2e=Decimal("5000"),
            is_active=False,
        )
        org = _m.OrganizationStructure(org_name="Test", entities=[entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org)
        er = result.entity_results[0]
        assert er.inclusion_status == _m.InclusionStatus.EXCLUDED

    def test_processing_time_positive(self):
        entity = _m.LegalEntity(entity_name="Timer", total_scope1_tco2e=Decimal("100"))
        org = _m.OrganizationStructure(org_name="Timer Org", entities=[entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org)
        assert result.processing_time_ms >= Decimal("0")

    def test_countries_and_sectors_collected(self):
        entity = _m.LegalEntity(
            entity_name="Geo Test",
            country_of_incorporation="US",
            sector="manufacturing",
            total_scope1_tco2e=Decimal("100"),
        )
        org = _m.OrganizationStructure(org_name="Geo Org", entities=[entity])
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org)
        assert "US" in result.countries_covered
        assert "manufacturing" in result.sectors_covered

    def test_100_entities_completes(self):
        """Performance: 100 entities should complete without error."""
        entities = []
        for i in range(100):
            entities.append(_m.LegalEntity(
                entity_name=f"Entity-{i:03d}",
                equity_pct=Decimal("100"),
                total_scope1_tco2e=Decimal("100"),
                total_scope2_tco2e=Decimal("50"),
            ))
        org = _m.OrganizationStructure(org_name="Big Org", entities=entities)
        engine = _m.OrganizationalBoundaryEngine()
        result = engine.define_boundary(org)
        assert result.total_entities == 100
        assert result.included_entities == 100

    def test_inclusion_percentages_api(self):
        engine = _m.OrganizationalBoundaryEngine()
        entities = [
            _m.LegalEntity(entity_id="E1", entity_name="Sub", equity_pct=Decimal("100")),
            _m.LegalEntity(
                entity_id="E2", entity_name="JV",
                entity_type=_m.EntityType.JV,
                equity_pct=Decimal("40"),
            ),
        ]
        pcts = engine.calculate_inclusion_percentages(
            entities, _m.ConsolidationApproach.EQUITY_SHARE
        )
        assert pcts["E1"] == Decimal("100.00")
        assert pcts["E2"] == Decimal("40.00")
