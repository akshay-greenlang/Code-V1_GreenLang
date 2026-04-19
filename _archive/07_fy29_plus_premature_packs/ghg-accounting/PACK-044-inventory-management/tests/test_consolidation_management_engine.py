# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Consolidation Management Engine Tests
=============================================================

Tests ConsolidationManagementEngine: equity-share and operational-control
consolidation, intra-group elimination, entity hierarchy, and multi-entity
rollups.

Target: 50+ test cases.
"""

from decimal import Decimal

import pytest

from conftest import _load_engine

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

_mod = _load_engine("consolidation_management")

ConsolidationManagementEngine = _mod.ConsolidationManagementEngine
ConsolidationManagementResult = _mod.ConsolidationManagementResult
EntityConsolidatedResult = _mod.EntityConsolidatedResult
EliminationRecord = _mod.EliminationRecord
ConsolidationStatus = _mod.ConsolidationStatus
ConsolidationApproach = _mod.ConsolidationApproach
EntityHierarchy = _mod.EntityHierarchy
Entity = _mod.Entity
SubsidiarySubmission = _mod.SubsidiarySubmission
IntraGroupTransfer = _mod.IntraGroupTransfer
EliminationType = _mod.EliminationType
SubmissionStatus = _mod.SubmissionStatus


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def engine():
    """Create a fresh ConsolidationManagementEngine."""
    return ConsolidationManagementEngine()


def _make_hierarchy(approach=ConsolidationApproach.OPERATIONAL_CONTROL,
                    group_name="Acme Group"):
    """Build standard entity hierarchy for testing."""
    return EntityHierarchy(
        group_name=group_name,
        reporting_year=2025,
        consolidation_approach=approach,
        entities=[
            Entity(
                entity_id="ENT-ROOT",
                entity_name="Acme Group HQ",
                entity_type="parent",
                parent_entity_id=None,
                equity_pct=Decimal("100"),
                has_operational_control=True,
                has_financial_control=True,
            ),
            Entity(
                entity_id="ENT-001",
                entity_name="Acme Manufacturing US",
                entity_type="subsidiary",
                parent_entity_id="ENT-ROOT",
                equity_pct=Decimal("100"),
                has_operational_control=True,
                has_financial_control=True,
            ),
            Entity(
                entity_id="ENT-002",
                entity_name="Acme Europe GmbH",
                entity_type="subsidiary",
                parent_entity_id="ENT-ROOT",
                equity_pct=Decimal("100"),
                has_operational_control=True,
                has_financial_control=True,
            ),
            Entity(
                entity_id="ENT-003",
                entity_name="Acme-Nippon JV",
                entity_type="joint_venture",
                parent_entity_id="ENT-ROOT",
                equity_pct=Decimal("50"),
                has_operational_control=False,
                has_financial_control=False,
            ),
        ],
    )


def _make_submissions():
    """Build standard submissions."""
    return [
        SubsidiarySubmission(
            entity_id="ENT-001",
            scope1_tco2e=Decimal("12500"),
            scope2_location_tco2e=Decimal("8200"),
            scope2_market_tco2e=Decimal("5100"),
            scope3_tco2e=Decimal("15000"),
            status=SubmissionStatus.SUBMITTED,
        ),
        SubsidiarySubmission(
            entity_id="ENT-002",
            scope1_tco2e=Decimal("9800"),
            scope2_location_tco2e=Decimal("6500"),
            scope2_market_tco2e=Decimal("4100"),
            scope3_tco2e=Decimal("12000"),
            status=SubmissionStatus.SUBMITTED,
        ),
        SubsidiarySubmission(
            entity_id="ENT-003",
            scope1_tco2e=Decimal("7200"),
            scope2_location_tco2e=Decimal("5100"),
            scope2_market_tco2e=Decimal("3600"),
            scope3_tco2e=Decimal("8000"),
            status=SubmissionStatus.SUBMITTED,
        ),
    ]


def _make_transfers():
    """Build standard intragroup transfers."""
    return [
        IntraGroupTransfer(
            from_entity_id="ENT-001",
            to_entity_id="ENT-002",
            elimination_type=EliminationType.PRODUCT_TRANSFER,
            amount_tco2e=Decimal("150"),
            scope=2,
            description="Intercompany product transfer US->DE",
        ),
    ]


@pytest.fixture
def hierarchy_oc():
    return _make_hierarchy(ConsolidationApproach.OPERATIONAL_CONTROL)


@pytest.fixture
def hierarchy_eq():
    return _make_hierarchy(ConsolidationApproach.EQUITY_SHARE)


@pytest.fixture
def submissions():
    return _make_submissions()


@pytest.fixture
def transfers():
    return _make_transfers()


# ===================================================================
# Operational Control Tests
# ===================================================================


class TestOperationalControl:
    """Tests for operational control consolidation approach."""

    def test_consolidate_operational_control(self, engine, hierarchy_oc, submissions):
        result = engine.consolidate(hierarchy_oc, submissions)
        assert isinstance(result, ConsolidationManagementResult)

    def test_100pct_inclusion_for_controlled_entities(self, engine, hierarchy_oc, submissions):
        result = engine.consolidate(hierarchy_oc, submissions)
        for er in result.entity_results:
            if er.entity_id == "ENT-001":
                assert er.inclusion_pct == 100.0

    def test_jv_excluded_without_control(self, engine, hierarchy_oc, submissions):
        result = engine.consolidate(hierarchy_oc, submissions)
        jv = [e for e in result.entity_results if e.entity_id == "ENT-003"]
        if jv:
            assert jv[0].inclusion_pct == 0.0

    def test_total_scope1_correct(self, engine, hierarchy_oc, submissions):
        result = engine.consolidate(hierarchy_oc, submissions)
        expected = 12500.0 + 9800.0  # ENT-001 + ENT-002, JV excluded
        assert abs(result.total_scope1_tco2e - expected) < 1.0

    def test_total_scope2_location_correct(self, engine, hierarchy_oc, submissions):
        result = engine.consolidate(hierarchy_oc, submissions)
        expected = 8200.0 + 6500.0
        assert abs(result.total_scope2_location_tco2e - expected) < 1.0


# ===================================================================
# Equity Share Tests
# ===================================================================


class TestEquityShare:
    """Tests for equity share consolidation approach."""

    def test_consolidate_equity_share(self, engine, hierarchy_eq, submissions):
        result = engine.consolidate(hierarchy_eq, submissions)
        assert result is not None

    def test_jv_proportional_inclusion(self, engine, hierarchy_eq, submissions):
        result = engine.consolidate(hierarchy_eq, submissions)
        jv = [e for e in result.entity_results if e.entity_id == "ENT-003"]
        if jv:
            assert jv[0].inclusion_pct == 50.0

    def test_equity_scope1_includes_jv_share(self, engine, hierarchy_eq, submissions):
        result = engine.consolidate(hierarchy_eq, submissions)
        expected = 12500.0 + 9800.0 + (7200.0 * 0.5)
        assert abs(result.total_scope1_tco2e - expected) < 1.0

    def test_equity_scope2_market_includes_jv_share(self, engine, hierarchy_eq, submissions):
        result = engine.consolidate(hierarchy_eq, submissions)
        expected = 5100.0 + 4100.0 + (3600.0 * 0.5)
        assert abs(result.total_scope2_market_tco2e - expected) < 1.0


# ===================================================================
# Intra-Group Elimination Tests
# ===================================================================


class TestIntraGroupElimination:
    """Tests for intra-group transfer elimination."""

    def test_elimination_applied(self, engine, hierarchy_oc, submissions, transfers):
        result = engine.consolidate(hierarchy_oc, submissions, transfers)
        assert result.total_eliminations_tco2e > 0

    def test_elimination_records_generated(self, engine, hierarchy_oc, submissions, transfers):
        result = engine.consolidate(hierarchy_oc, submissions, transfers)
        assert len(result.eliminations) >= 1

    def test_net_total_reduced_by_eliminations(self, engine, hierarchy_oc, submissions, transfers):
        result_no_elim = engine.consolidate(hierarchy_oc, submissions)
        result_with_elim = engine.consolidate(
            _make_hierarchy(),
            _make_submissions(),
            transfers,
        )
        assert result_with_elim.total_all_scopes_tco2e <= result_no_elim.total_all_scopes_tco2e


# ===================================================================
# Consolidation Status Tests
# ===================================================================


class TestConsolidationStatusTests:
    """Tests for consolidation status tracking."""

    def test_status_populated(self, engine, hierarchy_oc, submissions):
        result = engine.consolidate(hierarchy_oc, submissions)
        assert result.consolidation_status is not None
        # 4 entities in hierarchy (root + 3 subsidiaries), all material
        assert result.consolidation_status.total_entities >= 3

    def test_submitted_entities(self, engine, hierarchy_oc, submissions):
        result = engine.consolidate(hierarchy_oc, submissions)
        assert result.consolidation_status.entities_submitted >= 2

    def test_completeness_percentage(self, engine, hierarchy_oc, submissions):
        result = engine.consolidate(hierarchy_oc, submissions)
        assert result.consolidation_status.completeness_pct > 0


# ===================================================================
# Edge Cases Tests
# ===================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_submissions_raises(self, engine, hierarchy_oc):
        with pytest.raises(ValueError):
            engine.consolidate(hierarchy_oc, [])

    def test_single_entity_and_submission(self, engine):
        hierarchy = EntityHierarchy(
            group_name="Solo",
            reporting_year=2025,
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            entities=[
                Entity(
                    entity_id="ENT-SOLO",
                    entity_name="Solo Corp",
                    entity_type="parent",
                    parent_entity_id=None,
                    equity_pct=Decimal("100"),
                    has_operational_control=True,
                ),
            ],
        )
        submissions = [
            SubsidiarySubmission(
                entity_id="ENT-SOLO",
                scope1_tco2e=Decimal("12500"),
                scope2_location_tco2e=Decimal("8200"),
                scope2_market_tco2e=Decimal("5100"),
                status=SubmissionStatus.SUBMITTED,
            ),
        ]
        result = engine.consolidate(hierarchy, submissions)
        assert abs(result.total_scope1_tco2e - 12500.0) < 1.0

    def test_zero_equity_excluded(self, engine):
        hierarchy = EntityHierarchy(
            group_name="Zero",
            reporting_year=2025,
            consolidation_approach=ConsolidationApproach.EQUITY_SHARE,
            entities=[
                Entity(
                    entity_id="ENT-ROOT",
                    entity_name="Zero Parent",
                    entity_type="parent",
                    parent_entity_id=None,
                    equity_pct=Decimal("0"),
                    has_operational_control=False,
                    has_financial_control=False,
                ),
            ],
        )
        submissions = [
            SubsidiarySubmission(
                entity_id="ENT-ROOT",
                scope1_tco2e=Decimal("5000"),
                scope2_location_tco2e=Decimal("3000"),
                status=SubmissionStatus.SUBMITTED,
            ),
        ]
        result = engine.consolidate(hierarchy, submissions)
        assert result.total_scope1_tco2e == 0.0


# ===================================================================
# Provenance Tests
# ===================================================================


class TestProvenance:
    """Tests for provenance hashing."""

    def test_result_has_provenance_hash(self, engine, hierarchy_oc, submissions):
        result = engine.consolidate(hierarchy_oc, submissions)
        assert len(result.provenance_hash) == 64

    def test_processing_time_positive(self, engine, hierarchy_oc, submissions):
        result = engine.consolidate(hierarchy_oc, submissions)
        assert result.processing_time_ms >= 0


# ===================================================================
# Model Tests
# ===================================================================


class TestModels:
    """Tests for Pydantic model defaults and enum values."""

    @pytest.mark.parametrize("approach", list(ConsolidationApproach))
    def test_consolidation_approaches(self, approach):
        assert approach.value is not None

    def test_entity_consolidated_result_defaults(self):
        er = EntityConsolidatedResult()
        assert er.equity_pct == 100.0
        assert er.inclusion_pct == 100.0

    def test_elimination_record_defaults(self):
        er = EliminationRecord()
        assert er.amount_tco2e == 0.0

    def test_consolidation_status_defaults(self):
        cs = ConsolidationStatus()
        assert cs.total_entities == 0
        assert cs.all_approved is False

    def test_result_defaults(self):
        r = ConsolidationManagementResult()
        assert r.total_scope1_tco2e == 0.0
        assert r.total_all_scopes_tco2e == 0.0
