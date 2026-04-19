# -*- coding: utf-8 -*-
"""
Unit tests for MultiEntityEngine (PACK-022 Engine 8).

Tests multi-entity GHG consolidation using equity share, financial control,
and operational control approaches, intercompany elimination, base year
recalculation, target allocation, and completeness tracking.
"""

import pytest
from decimal import Decimal

from engines.multi_entity_engine import (
    MultiEntityEngine,
    MultiEntityConfig,
    EntityEmissions,
    IntercompanyElimination,
    StructuralChange,
    EntityTargetAllocation,
    GroupEmissions,
    BaseYearRecalculation,
    ConsolidationResult,
    ConsolidationMethod,
    EntityType,
    ReportingStatus,
    EliminationType,
    StructuralChangeType,
    TargetAllocationType,
    MAX_ENTITIES,
    MAX_HIERARCHY_DEPTH,
    CURRENCY_RATES_TO_USD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(eid, name, year, s1, s2m, s3, revenue=1000000,
                 ownership=100, fin_ctrl=True, ops_ctrl=True,
                 status=ReportingStatus.ACTUAL, entity_type=EntityType.SUBSIDIARY,
                 currency="USD", s2l=None, hierarchy=1, parent=None):
    return EntityEmissions(
        entity_id=eid,
        entity_name=name,
        entity_type=entity_type,
        parent_entity_id=parent,
        hierarchy_level=hierarchy,
        reporting_year=year,
        scope1_emissions=Decimal(str(s1)),
        scope2_location=Decimal(str(s2l if s2l is not None else s2m)),
        scope2_market=Decimal(str(s2m)),
        scope3_emissions=Decimal(str(s3)),
        revenue=Decimal(str(revenue)),
        ownership_pct=Decimal(str(ownership)),
        has_financial_control=fin_ctrl,
        has_operational_control=ops_ctrl,
        reporting_status=status,
        currency=currency,
    )


@pytest.fixture
def engine():
    return MultiEntityEngine()


@pytest.fixture
def loaded_engine():
    eng = MultiEntityEngine()
    eng.set_group_name("GlobalCorp Group")
    eng.add_entity(_make_entity("e1", "Sub A", 2024, 500, 200, 1000, revenue=5000000))
    eng.add_entity(_make_entity("e2", "Sub B", 2024, 300, 150, 800, revenue=3000000))
    eng.add_entity(_make_entity("e3", "Sub C", 2024, 100, 50, 300, revenue=1000000,
                                status=ReportingStatus.ESTIMATED))
    return eng


@pytest.fixture
def two_year_engine():
    eng = MultiEntityEngine()
    eng.set_group_name("TwoYear Corp")
    # Base year
    eng.add_entity(_make_entity("e1", "Sub A", 2022, 600, 250, 1200, revenue=5000000))
    eng.add_entity(_make_entity("e2", "Sub B", 2022, 400, 180, 900, revenue=3000000))
    # Reporting year
    eng.add_entity(_make_entity("e1", "Sub A", 2024, 500, 200, 1000, revenue=5500000))
    eng.add_entity(_make_entity("e2", "Sub B", 2024, 350, 160, 800, revenue=3200000))
    return eng


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestMultiEntityInit:

    def test_default_config(self, engine):
        assert isinstance(engine.config, MultiEntityConfig)
        assert engine.config.default_consolidation == ConsolidationMethod.OPERATIONAL_CONTROL

    def test_custom_config_dict(self):
        eng = MultiEntityEngine({"default_consolidation": "equity_share", "max_entities": 20})
        assert eng.config.default_consolidation == ConsolidationMethod.EQUITY_SHARE
        assert eng.config.max_entities == 20

    def test_set_group_name(self, engine):
        engine.set_group_name("  TestGroup  ")
        assert engine._group_name == "TestGroup"


# ---------------------------------------------------------------------------
# Entity Management Tests
# ---------------------------------------------------------------------------


class TestEntityManagement:

    def test_add_entity(self, engine):
        entity = _make_entity("e1", "Sub A", 2024, 100, 50, 200)
        result = engine.add_entity(entity)
        assert len(result.provenance_hash) == 64

    def test_auto_calculate_total(self, engine):
        entity = _make_entity("e1", "Sub A", 2024, 100, 50, 200)
        result = engine.add_entity(entity)
        assert result.total_emissions == Decimal("350")  # 100 + 50 + 200

    def test_add_entities_batch(self, engine):
        entities = [
            _make_entity("e1", "A", 2024, 100, 50, 200),
            _make_entity("e2", "B", 2024, 200, 100, 300),
        ]
        count = engine.add_entities(entities)
        assert count == 2

    def test_max_entities_exceeded(self):
        eng = MultiEntityEngine({"max_entities": 2})
        eng.add_entity(_make_entity("e1", "A", 2024, 100, 50, 200))
        eng.add_entity(_make_entity("e2", "B", 2024, 100, 50, 200))
        with pytest.raises(ValueError, match="Maximum"):
            eng.add_entity(_make_entity("e3", "C", 2024, 100, 50, 200))

    def test_hierarchy_depth_exceeded(self, engine):
        with pytest.raises(ValueError, match="exceeds max depth"):
            engine.add_entity(
                _make_entity("e1", "A", 2024, 100, 50, 200, hierarchy=MAX_HIERARCHY_DEPTH + 1)
            )

    def test_clear(self, loaded_engine):
        loaded_engine.clear()
        assert len(loaded_engine._entities) == 0
        assert loaded_engine._group_name == ""


# ---------------------------------------------------------------------------
# Consolidation Tests
# ---------------------------------------------------------------------------


class TestConsolidation:

    def test_operational_control(self, loaded_engine):
        result = loaded_engine.consolidate(2024, ConsolidationMethod.OPERATIONAL_CONTROL)
        assert isinstance(result, GroupEmissions)
        assert result.entities_consolidated == 3
        assert result.consolidation_method == ConsolidationMethod.OPERATIONAL_CONTROL
        assert float(result.scope1_total) > 0

    def test_equity_share(self, engine):
        engine.add_entity(_make_entity("e1", "A", 2024, 100, 50, 200, ownership=60))
        engine.add_entity(_make_entity("e2", "B", 2024, 100, 50, 200, ownership=40))
        result = engine.consolidate(2024, ConsolidationMethod.EQUITY_SHARE)
        # S1 = 100*0.6 + 100*0.4 = 100
        assert float(result.scope1_total) == pytest.approx(100.0, rel=1e-2)

    def test_financial_control_excludes(self, engine):
        engine.add_entity(_make_entity("e1", "A", 2024, 100, 50, 200, fin_ctrl=True))
        engine.add_entity(_make_entity("e2", "B", 2024, 100, 50, 200, fin_ctrl=False))
        result = engine.consolidate(2024, ConsolidationMethod.FINANCIAL_CONTROL)
        # Only e1 should be included
        assert float(result.scope1_total) == pytest.approx(100.0, rel=1e-2)

    def test_no_data_for_year_raises(self, loaded_engine):
        with pytest.raises(ValueError, match="No entity data"):
            loaded_engine.consolidate(2020)

    def test_completeness_pct(self, loaded_engine):
        result = loaded_engine.consolidate(2024)
        # 2 actual, 1 estimated -> 66.67%
        assert float(result.completeness_pct) == pytest.approx(66.67, rel=1e-1)

    def test_eliminations_reduce_total(self, loaded_engine):
        loaded_engine.add_elimination(IntercompanyElimination(
            seller_entity_id="e1",
            buyer_entity_id="e2",
            elimination_type=EliminationType.SCOPE1_TO_SCOPE3,
            scope_at_seller="scope1",
            scope_at_buyer="scope3",
            emissions_eliminated=Decimal("50"),
        ))
        result = loaded_engine.consolidate(2024)
        # Total should be reduced by 50
        assert float(result.eliminations_total) == pytest.approx(50.0, rel=1e-3)

    def test_provenance_hash(self, loaded_engine):
        result = loaded_engine.consolidate(2024)
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)


# ---------------------------------------------------------------------------
# Currency Normalization Tests
# ---------------------------------------------------------------------------


class TestCurrencyNormalization:

    def test_same_currency_no_conversion(self, engine):
        amount = engine._normalize_currency(Decimal("1000"), "USD")
        assert amount == Decimal("1000")

    def test_eur_to_usd(self, engine):
        amount = engine._normalize_currency(Decimal("1000"), "EUR")
        assert float(amount) > 1000  # EUR > USD

    def test_unknown_currency_defaults(self, engine):
        amount = engine._normalize_currency(Decimal("1000"), "XYZ")
        assert float(amount) > 0


# ---------------------------------------------------------------------------
# Base Year Recalculation Tests
# ---------------------------------------------------------------------------


class TestBaseYearRecalculation:

    def test_below_threshold_no_recalc(self, two_year_engine):
        two_year_engine.add_structural_change(StructuralChange(
            change_type=StructuralChangeType.ACQUISITION,
            entity_id="e3",
            entity_name="New Sub",
            effective_year=2023,
            emissions_impact=Decimal("50"),
        ))
        result = two_year_engine.assess_base_year_recalculation(2022)
        assert isinstance(result, BaseYearRecalculation)
        assert result.recalculation_required is False
        assert result.recalculated_base_year_emissions is None

    def test_above_threshold_triggers_recalc(self, two_year_engine):
        two_year_engine.add_structural_change(StructuralChange(
            change_type=StructuralChangeType.MERGER,
            entity_id="e3",
            entity_name="MergedCo",
            effective_year=2023,
            emissions_impact=Decimal("500"),
        ))
        result = two_year_engine.assess_base_year_recalculation(2022)
        assert result.recalculation_required is True
        assert result.recalculated_base_year_emissions is not None

    def test_no_base_year_data_raises(self, engine):
        engine.add_structural_change(StructuralChange(
            change_type=StructuralChangeType.ACQUISITION,
            entity_id="e1",
            effective_year=2023,
            emissions_impact=Decimal("100"),
        ))
        with pytest.raises(ValueError, match="No entity data for base year"):
            engine.assess_base_year_recalculation(2020)


# ---------------------------------------------------------------------------
# Target Allocation Tests
# ---------------------------------------------------------------------------


class TestTargetAllocation:

    def test_proportional_allocation(self, two_year_engine):
        allocations = two_year_engine.allocate_targets(
            group_target_pct=Decimal("30"),
            base_year=2022,
            reporting_year=2024,
            allocation_type=TargetAllocationType.TOP_DOWN_PROPORTIONAL,
        )
        assert len(allocations) == 2
        total_alloc = sum(a.allocated_target_absolute for a in allocations)
        assert float(total_alloc) > 0

    def test_equal_allocation(self, two_year_engine):
        allocations = two_year_engine.allocate_targets(
            group_target_pct=Decimal("30"),
            base_year=2022,
            reporting_year=2024,
            allocation_type=TargetAllocationType.TOP_DOWN_EQUAL,
        )
        abs_targets = [float(a.allocated_target_absolute) for a in allocations]
        assert abs_targets[0] == pytest.approx(abs_targets[1], rel=1e-3)

    def test_no_base_year_raises(self, engine):
        with pytest.raises(ValueError, match="No entity data for base year"):
            engine.allocate_targets(Decimal("30"), 2020, 2024)

    def test_progress_tracking(self, two_year_engine):
        allocations = two_year_engine.allocate_targets(
            group_target_pct=Decimal("30"),
            base_year=2022,
            reporting_year=2024,
        )
        for alloc in allocations:
            assert float(alloc.progress_pct) >= 0


# ---------------------------------------------------------------------------
# Completeness Report Tests
# ---------------------------------------------------------------------------


class TestCompletenessReport:

    def test_completeness_report(self, loaded_engine):
        report = loaded_engine.get_completeness_report(2024)
        assert report["total_entities"] == 3
        assert report["status_breakdown"]["actual"] == 2
        assert report["status_breakdown"]["estimated"] == 1
        assert "provenance_hash" in report

    def test_no_entities_for_year(self, loaded_engine):
        report = loaded_engine.get_completeness_report(2020)
        assert report["total_entities"] == 0


# ---------------------------------------------------------------------------
# Progress Aggregation Tests
# ---------------------------------------------------------------------------


class TestProgressAggregation:

    def test_aggregate_progress(self, two_year_engine):
        result = two_year_engine.aggregate_progress(2022, 2024, Decimal("30"))
        assert "base_year_emissions" in result
        assert "progress_pct" in result
        assert "on_track" in result

    def test_progress_error_on_missing_year(self, engine):
        result = engine.aggregate_progress(2020, 2024, Decimal("30"))
        assert "error" in result


# ---------------------------------------------------------------------------
# Full Consolidation Pipeline Tests
# ---------------------------------------------------------------------------


class TestFullConsolidation:

    def test_full_consolidation_structure(self, loaded_engine):
        result = loaded_engine.run_full_consolidation(2024)
        assert isinstance(result, ConsolidationResult)
        assert isinstance(result.group_emissions, GroupEmissions)
        assert len(result.entity_breakdown) == 3
        assert len(result.provenance_hash) == 64

    def test_full_with_target_allocation(self, two_year_engine):
        result = two_year_engine.run_full_consolidation(
            reporting_year=2024,
            base_year=2022,
            group_target_pct=Decimal("30"),
        )
        assert len(result.target_allocation) == 2

    def test_full_with_structural_changes(self, two_year_engine):
        two_year_engine.add_structural_change(StructuralChange(
            change_type=StructuralChangeType.ACQUISITION,
            entity_id="e3",
            effective_year=2023,
            emissions_impact=Decimal("200"),
        ))
        result = two_year_engine.run_full_consolidation(
            reporting_year=2024, base_year=2022
        )
        assert result.base_year_recalculation is not None


# ---------------------------------------------------------------------------
# Edge Cases & Constants
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_max_entities_constant(self):
        assert MAX_ENTITIES == 50

    def test_max_hierarchy_depth_constant(self):
        assert MAX_HIERARCHY_DEPTH == 3

    def test_currency_table_has_usd(self):
        assert "USD" in CURRENCY_RATES_TO_USD
        assert CURRENCY_RATES_TO_USD["USD"] == Decimal("1.000")

    def test_enum_values(self):
        assert ConsolidationMethod.EQUITY_SHARE.value == "equity_share"
        assert EntityType.PARENT.value == "parent"
        assert ReportingStatus.ACTUAL.value == "actual"
        assert EliminationType.SCOPE1_TO_SCOPE3.value == "scope1_to_scope3"
        assert StructuralChangeType.ACQUISITION.value == "acquisition"
        assert TargetAllocationType.BOTTOM_UP.value == "bottom_up"

    def test_entity_with_all_scopes(self, engine):
        entity = _make_entity("e1", "Full", 2024, 1000, 500, 3000, s2l=600)
        result = engine.add_entity(entity)
        assert result.total_emissions == Decimal("4500")
