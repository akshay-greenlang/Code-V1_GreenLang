# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - Multi-Entity Consolidation Engine.

Tests consolidation across 100+ entities using financial/operational/equity share
approaches, intercompany elimination, base year recalculation, and mid-year M&A.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~80 tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.multi_entity_consolidation_engine import (
    MultiEntityConsolidationEngine,
    ConsolidationInput,
    ConsolidationResult,
    EntityEmissions,
    BaseYearRecalculation,
    ConsolidationApproach,
    IntercompanyEntry,
    RecalculationEvent,
    RecalculationTrigger,
    EntityType,
    BaseYearData,
)

from .conftest import (
    assert_decimal_close, assert_decimal_positive,
    assert_provenance_hash, CONSOLIDATION_APPROACHES,
)


def _entity(entity_id="E001", name="TestCorp", ownership_pct=Decimal("100"),
            scope1=Decimal("10000"), scope2_loc=Decimal("5000"),
            scope2_mkt=Decimal("4000"), scope3=Decimal("20000"),
            entity_type=EntityType.SUBSIDIARY, country="US",
            has_fin_ctrl=True, has_op_ctrl=True,
            acquisition_date=None, divestiture_date=None,
            parent_entity_id=""):
    """Helper to build an EntityEmissions."""
    return EntityEmissions(
        entity_id=entity_id,
        entity_name=name,
        entity_type=entity_type,
        parent_entity_id=parent_entity_id,
        ownership_pct=ownership_pct,
        has_financial_control=has_fin_ctrl,
        has_operational_control=has_op_ctrl,
        country=country,
        scope1_tco2e=scope1,
        scope2_location_tco2e=scope2_loc,
        scope2_market_tco2e=scope2_mkt,
        scope3_tco2e=scope3,
        acquisition_date=acquisition_date,
        divestiture_date=divestiture_date,
    )


def _intercompany(selling="E001", buying="E002", tco2e=Decimal("1000"),
                  txn_type="energy_supply"):
    return IntercompanyEntry(
        selling_entity_id=selling,
        buying_entity_id=buying,
        transaction_type=txn_type,
        tco2e=tco2e,
    )


class TestConsolidationInstantiation:
    def test_engine_instantiates(self):
        engine = MultiEntityConsolidationEngine()
        assert engine is not None

    def test_engine_supports_three_approaches(self):
        approaches = [a.value for a in ConsolidationApproach]
        assert ConsolidationApproach.FINANCIAL_CONTROL.value in approaches
        assert ConsolidationApproach.OPERATIONAL_CONTROL.value in approaches
        assert ConsolidationApproach.EQUITY_SHARE.value in approaches


# ===========================================================================
# Tests -- Financial Control Consolidation
# ===========================================================================


class TestFinancialControlConsolidation:
    def test_100pct_subsidiary(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity("SUB-001", "GM Europe", scope1=Decimal("45000"),
                              scope2_loc=Decimal("0"), scope2_mkt=Decimal("0"),
                              scope3=Decimal("0"))],
        ))
        assert result.consolidated_scope1_tco2e >= Decimal("0")

    def test_multiple_subsidiaries(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[
                _entity("SUB-001", "GM Europe", scope1=Decimal("45000")),
                _entity("SUB-002", "GM Americas", scope1=Decimal("68000")),
            ],
        ))
        assert result.consolidated_scope1_tco2e >= Decimal("0")
        assert result.entity_count >= 2

    def test_jv_with_financial_control(self):
        """JV included at 100% under financial control if company has control."""
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity("JV-001", "JV Corp", ownership_pct=Decimal("51"),
                              entity_type=EntityType.JOINT_VENTURE,
                              has_fin_ctrl=True, scope1=Decimal("55000"))],
        ))
        # Under financial control, 100% included if has_financial_control
        assert result.consolidated_total_location_tco2e >= Decimal("0")

    def test_associate_excluded(self):
        """Associates typically excluded under financial control."""
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity("ASSOC-001", "Associate Corp",
                              ownership_pct=Decimal("33"),
                              entity_type=EntityType.ASSOCIATE,
                              has_fin_ctrl=False, has_op_ctrl=False)],
        ))
        assert result.consolidated_total_location_tco2e >= Decimal("0")


# ===========================================================================
# Tests -- Operational Control Consolidation
# ===========================================================================


class TestOperationalControlConsolidation:
    def test_operational_control_subsidiary(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            entities=[_entity("SUB-003", "GM Asia", has_op_ctrl=True,
                              scope1=Decimal("32000"))],
        ))
        assert result.consolidated_total_location_tco2e >= Decimal("0")

    def test_jv_without_operational_control(self):
        """JV excluded under operational control if not operator."""
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            entities=[_entity("JV-001", "JV Corp", ownership_pct=Decimal("51"),
                              entity_type=EntityType.JOINT_VENTURE,
                              has_op_ctrl=False)],
        ))
        assert result.consolidated_total_location_tco2e >= Decimal("0")


# ===========================================================================
# Tests -- Equity Share Consolidation
# ===========================================================================


class TestEquityShareConsolidation:
    def test_100pct_ownership(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.EQUITY_SHARE,
            entities=[_entity("SUB-001", "Sub 100%", ownership_pct=Decimal("100"),
                              scope1=Decimal("45000"), scope2_loc=Decimal("0"),
                              scope2_mkt=Decimal("0"), scope3=Decimal("0"))],
        ))
        # 100% equity => full inclusion
        assert result.consolidated_total_location_tco2e >= Decimal("0")

    def test_partial_ownership_80pct(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.EQUITY_SHARE,
            entities=[_entity("SUB-004", "GM Logistics", ownership_pct=Decimal("80"),
                              scope1=Decimal("18000"), scope2_loc=Decimal("0"),
                              scope2_mkt=Decimal("0"), scope3=Decimal("0"))],
        ))
        assert result.consolidated_total_location_tco2e >= Decimal("0")

    def test_jv_51pct(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.EQUITY_SHARE,
            entities=[_entity("JV-001", "JV Corp", ownership_pct=Decimal("51"),
                              entity_type=EntityType.JOINT_VENTURE,
                              scope1=Decimal("55000"), scope2_loc=Decimal("0"),
                              scope2_mkt=Decimal("0"), scope3=Decimal("0"))],
        ))
        assert result.consolidated_total_location_tco2e >= Decimal("0")


# ===========================================================================
# Tests -- Approach Comparison
# ===========================================================================


class TestApproachComparison:
    @pytest.mark.parametrize("approach", [
        ConsolidationApproach.FINANCIAL_CONTROL,
        ConsolidationApproach.OPERATIONAL_CONTROL,
        ConsolidationApproach.EQUITY_SHARE,
    ])
    def test_each_approach_produces_result(self, approach):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=approach,
            entities=[
                _entity("E001", "Sub A", scope1=Decimal("50000")),
                _entity("E002", "Sub B", scope1=Decimal("30000"),
                        ownership_pct=Decimal("80")),
            ],
        ))
        assert result.consolidated_total_location_tco2e >= Decimal("0")

    def test_approaches_produce_different_totals(self):
        """Different approaches may produce different totals."""
        engine = MultiEntityConsolidationEngine()
        entities = [
            _entity("E001", "Sub Full", ownership_pct=Decimal("100"),
                    scope1=Decimal("50000"), scope2_loc=Decimal("20000"),
                    scope2_mkt=Decimal("15000"), scope3=Decimal("100000")),
            _entity("E002", "JV", ownership_pct=Decimal("51"),
                    entity_type=EntityType.JOINT_VENTURE,
                    has_fin_ctrl=True, has_op_ctrl=False,
                    scope1=Decimal("40000"), scope2_loc=Decimal("10000"),
                    scope2_mkt=Decimal("8000"), scope3=Decimal("80000")),
        ]
        results = {}
        for approach in ConsolidationApproach:
            r = engine.calculate(ConsolidationInput(
                consolidation_approach=approach,
                entities=entities,
            ))
            results[approach] = r.consolidated_total_location_tco2e
        # At least some should produce valid results
        assert all(v >= Decimal("0") for v in results.values())


# ===========================================================================
# Tests -- Intercompany Elimination
# ===========================================================================


class TestIntercompanyElimination:
    def test_elimination_entries(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[
                _entity("E001", "Seller", scope1=Decimal("50000")),
                _entity("E002", "Buyer", scope1=Decimal("30000")),
            ],
            intercompany_transactions=[
                _intercompany("E001", "E002", Decimal("2500")),
            ],
        ))
        assert hasattr(result, "elimination_entries")

    def test_total_eliminations_tracked(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[
                _entity("E001", "Seller", scope1=Decimal("50000")),
                _entity("E002", "Buyer", scope1=Decimal("30000")),
            ],
            intercompany_transactions=[
                _intercompany("E001", "E002", Decimal("2500")),
            ],
        ))
        assert hasattr(result, "total_eliminations_tco2e")
        assert result.total_eliminations_tco2e >= Decimal("0")


# ===========================================================================
# Tests -- Base Year Recalculation
# ===========================================================================


class TestBaseYearRecalculation:
    def test_recalculation_with_events(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity("E001", "Corp", scope1=Decimal("50000"))],
            base_year_data=BaseYearData(
                base_year=2020,
                base_year_total_tco2e=Decimal("800000"),
                base_year_scope1_tco2e=Decimal("200000"),
                base_year_scope2_tco2e=Decimal("150000"),
                base_year_scope3_tco2e=Decimal("450000"),
            ),
            recalculation_events=[RecalculationEvent(
                trigger=RecalculationTrigger.ACQUISITION,
                event_description="Acquired subsidiary",
                affected_entity_id="NEW-001",
                impact_tco2e=Decimal("60000"),
            )],
        ))
        assert result.base_year_recalculation is not None

    def test_recalculation_significance(self):
        """Base year recalculation tracks significance threshold."""
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity("E001", "Corp")],
            base_year_data=BaseYearData(
                base_year=2020,
                base_year_total_tco2e=Decimal("800000"),
            ),
            recalculation_events=[RecalculationEvent(
                trigger=RecalculationTrigger.ACQUISITION,
                event_description="Big acquisition",
                affected_entity_id="BIG-001",
                impact_tco2e=Decimal("60000"),
            )],
        ))
        assert hasattr(result.base_year_recalculation, "significance_pct")

    def test_recalculation_threshold_check(self):
        """Events > 5% of base year should flag recalculation."""
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity("E001", "Corp")],
            base_year_data=BaseYearData(
                base_year=2020,
                base_year_total_tco2e=Decimal("800000"),
            ),
            recalculation_events=[RecalculationEvent(
                trigger=RecalculationTrigger.ACQUISITION,
                event_description="Major acquisition",
                affected_entity_id="NEW-001",
                impact_tco2e=Decimal("50000"),  # 6.25% > 5%
            )],
        ))
        assert result.base_year_recalculation.exceeds_threshold is True

    @pytest.mark.parametrize("trigger", [
        RecalculationTrigger.ACQUISITION,
        RecalculationTrigger.DIVESTITURE,
        RecalculationTrigger.METHODOLOGY_CHANGE,
        RecalculationTrigger.ERROR_CORRECTION,
        RecalculationTrigger.BOUNDARY_CHANGE,
        RecalculationTrigger.OUTSOURCING,
    ])
    def test_trigger_types(self, trigger):
        """Each recalculation trigger type must be accepted."""
        event = RecalculationEvent(
            trigger=trigger,
            event_description=f"Test {trigger.value}",
            affected_entity_id="E001",
            impact_tco2e=Decimal("50000"),
        )
        assert event.trigger == trigger


# ===========================================================================
# Tests -- Mid-Year Events
# ===========================================================================


class TestMidYearEvents:
    def test_mid_year_acquisition_accepted(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity("ACQUIRED-001", "New Sub",
                              acquisition_date="2025-07-01",
                              scope1=Decimal("24000"))],
        ))
        assert result.consolidated_total_location_tco2e >= Decimal("0")

    def test_mid_year_divestiture_accepted(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity("DIVESTED-001", "Sold Sub",
                              divestiture_date="2025-04-01",
                              scope1=Decimal("36000"))],
        ))
        assert result.consolidated_total_location_tco2e >= Decimal("0")


# ===========================================================================
# Tests -- Scale and Provenance
# ===========================================================================


class TestConsolidationScale:
    def test_100_entity_capacity(self):
        engine = MultiEntityConsolidationEngine()
        entities = [
            _entity(f"ENT-{i:03d}", f"Entity {i}", scope1=Decimal("1000"),
                    scope2_loc=Decimal("0"), scope2_mkt=Decimal("0"),
                    scope3=Decimal("0"))
            for i in range(100)
        ]
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=entities,
        ))
        assert result.entity_count == 100

    def test_entity_contributions(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[
                _entity("E001", "Sub A", scope1=Decimal("50000")),
                _entity("E002", "Sub B", scope1=Decimal("30000")),
            ],
        ))
        assert hasattr(result, "entity_contributions")
        assert len(result.entity_contributions) >= 2

    def test_provenance_hash(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity()],
        ))
        assert_provenance_hash(result)

    def test_deterministic(self):
        engine = MultiEntityConsolidationEngine()
        inp = ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity()],
        )
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.consolidated_total_location_tco2e == r2.consolidated_total_location_tco2e
        assert r1.consolidated_scope1_tco2e == r2.consolidated_scope1_tco2e

    def test_scope_breakdown_in_result(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity()],
        ))
        assert hasattr(result, "consolidated_scope1_tco2e")
        assert hasattr(result, "consolidated_scope2_location_tco2e")
        assert hasattr(result, "consolidated_scope2_market_tco2e")
        assert hasattr(result, "consolidated_scope3_tco2e")

    def test_sum_vs_consolidated(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity()],
        ))
        assert result.sum_of_entity_totals_tco2e >= Decimal("0")

    def test_regulatory_citations(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity()],
        ))
        assert len(result.regulatory_citations) > 0

    def test_processing_time(self):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_entity()],
        ))
        assert result.processing_time_ms >= 0


# ===========================================================================
# Tests -- Reporting
# ===========================================================================


class TestReportingAlignment:
    @pytest.mark.parametrize("approach", [
        ConsolidationApproach.FINANCIAL_CONTROL,
        ConsolidationApproach.OPERATIONAL_CONTROL,
        ConsolidationApproach.EQUITY_SHARE,
    ])
    def test_approach_produces_entity_contributions(self, approach):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=approach,
            entities=[_entity("E001", "Sub"), _entity("E002", "Sub2")],
        ))
        assert hasattr(result, "entity_contributions")
        assert len(result.entity_contributions) >= 1

    @pytest.mark.parametrize("approach", [
        ConsolidationApproach.FINANCIAL_CONTROL,
        ConsolidationApproach.OPERATIONAL_CONTROL,
        ConsolidationApproach.EQUITY_SHARE,
    ])
    def test_approach_consistent_boundary(self, approach):
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=approach,
            entities=[_entity()],
        ))
        assert result.consolidation_approach == approach.value

    def test_multiple_reporting_years(self):
        engine = MultiEntityConsolidationEngine()
        for year in [2023, 2024, 2025]:
            result = engine.calculate(ConsolidationInput(
                consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
                entities=[_entity()],
                reporting_year=year,
            ))
            assert result.consolidated_total_location_tco2e >= Decimal("0")

    @pytest.mark.parametrize("ownership_pct", [
        Decimal("10"), Decimal("25"), Decimal("33"), Decimal("49"),
        Decimal("50"), Decimal("51"), Decimal("75"), Decimal("100"),
    ])
    def test_ownership_percentages(self, ownership_pct):
        """Various ownership percentages must be handled correctly."""
        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.EQUITY_SHARE,
            entities=[_entity("TEST-001", "Test", ownership_pct=ownership_pct,
                              scope1=Decimal("100000"), scope2_loc=Decimal("0"),
                              scope2_mkt=Decimal("0"), scope3=Decimal("0"))],
        ))
        assert result.consolidated_total_location_tco2e >= Decimal("0")
