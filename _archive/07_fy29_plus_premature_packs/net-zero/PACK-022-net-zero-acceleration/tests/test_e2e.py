# -*- coding: utf-8 -*-
"""
End-to-end integration tests for PACK-022 Net Zero Acceleration engines 6-10.

Tests multi-engine pipelines that chain outputs from one engine as context
for another, validating cross-engine consistency and complete workflows.
"""

import pytest
from decimal import Decimal

from engines.temperature_scoring_engine import (
    TemperatureScoringEngine,
    PortfolioEntity,
    EmissionsTarget,
    TargetScope,
    TargetTimeframe,
    ScoreType,
    TemperatureBand,
    WhatIfScenario,
)
from engines.variance_decomposition_engine import (
    VarianceDecompositionEngine,
    SegmentData,
    ScopeFilter,
    ForecastHorizon,
    AlertSeverity,
)
from engines.multi_entity_engine import (
    MultiEntityEngine,
    EntityEmissions,
    IntercompanyElimination,
    ConsolidationMethod,
    EntityType,
    ReportingStatus,
    EliminationType,
    TargetAllocationType,
)
from engines.vcmi_validation_engine import (
    VCMIValidationEngine,
    EmissionsData,
    CarbonCreditPortfolio,
    VCMITier,
    CriterionStatus,
)
from engines.assurance_workpaper_engine import (
    AssuranceWorkpaperEngine,
    MethodologyEntry,
    DataLineageEntry,
    ExceptionEntry,
    CompletenessEntry,
    CalculationMethod,
    DataSourceType,
    ExceptionSeverity,
    AssuranceLevel,
    CrossCheckStatus,
)


# ---------------------------------------------------------------------------
# E2E: Multi-Entity Consolidation + Temperature Scoring
# ---------------------------------------------------------------------------


class TestMultiEntityTemperatureScoring:
    """Test consolidation followed by temperature scoring of entities."""

    def test_consolidate_then_score(self):
        # Step 1: Consolidate three subsidiaries
        me = MultiEntityEngine()
        me.set_group_name("GreenCorp")
        me.add_entity(EntityEmissions(
            entity_id="sub-a", entity_name="Sub A", reporting_year=2025,
            scope1_emissions=Decimal("500"), scope2_market=Decimal("200"),
            scope3_emissions=Decimal("1000"), revenue=Decimal("5000000"),
            has_operational_control=True,
        ))
        me.add_entity(EntityEmissions(
            entity_id="sub-b", entity_name="Sub B", reporting_year=2025,
            scope1_emissions=Decimal("300"), scope2_market=Decimal("150"),
            scope3_emissions=Decimal("800"), revenue=Decimal("3000000"),
            has_operational_control=True,
        ))
        group = me.consolidate(2025)
        assert float(group.total_emissions) > 0

        # Step 2: Score each subsidiary in a temperature portfolio
        ts = TemperatureScoringEngine()
        for eid, ename, rev in [("sub-a", "Sub A", 5000000), ("sub-b", "Sub B", 3000000)]:
            ts.add_entity(PortfolioEntity(
                entity_id=eid, entity_name=ename,
                revenue=Decimal(str(rev)),
                total_emissions=Decimal("1700") if eid == "sub-a" else Decimal("1250"),
                targets=[EmissionsTarget(
                    entity_id=eid,
                    scope=TargetScope.S1S2,
                    timeframe=TargetTimeframe.NEAR_TERM,
                    base_year=2020, target_year=2030,
                    base_year_emissions=Decimal("2000"),
                    target_year_emissions=Decimal("1000"),
                )],
            ))

        result = ts.run_full_assessment()
        assert result.entities_assessed == 2
        assert len(result.portfolio_scores) == 6
        for ps in result.portfolio_scores:
            assert float(ps.temperature_score) <= 3.2

    def test_temperature_what_if_on_group(self):
        ts = TemperatureScoringEngine()
        ts.add_entity(PortfolioEntity(
            entity_id="a", entity_name="A", revenue=Decimal("10000000"),
            targets=[EmissionsTarget(
                entity_id="a", scope=TargetScope.S1S2,
                timeframe=TargetTimeframe.NEAR_TERM,
                base_year=2020, target_year=2030,
                base_year_emissions=Decimal("1000"),
                target_year_emissions=Decimal("600"),
            )],
        ))
        ts.add_entity(PortfolioEntity(
            entity_id="b", entity_name="B", revenue=Decimal("5000000"),
            targets=[],
        ))
        scenario = WhatIfScenario(
            entity_id="b", new_annual_reduction_rate=Decimal("4.2"),
        )
        what_if = ts.run_what_if(scenario, ScoreType.WATS, TargetScope.S1S2, TargetTimeframe.NEAR_TERM)
        assert float(what_if.temperature_change) < 0


# ---------------------------------------------------------------------------
# E2E: Variance Decomposition Across Multiple Periods
# ---------------------------------------------------------------------------


class TestVarianceDecompositionMultiPeriod:
    """Test decomposition across 3+ years with forecasting and alerts."""

    def test_full_decomposition_with_alerts(self):
        vd = VarianceDecompositionEngine()
        for year, seg_data in [
            (2022, [("plant-a", 600, 1000), ("plant-b", 400, 800)]),
            (2023, [("plant-a", 550, 1050), ("plant-b", 380, 820)]),
            (2024, [("plant-a", 520, 1100), ("plant-b", 350, 850)]),
        ]:
            vd.add_segment_data([
                SegmentData(segment_id=sid, segment_name=sid, year=year,
                            emissions=Decimal(str(e)), activity=Decimal(str(a)))
                for sid, e, a in seg_data
            ])

        # Set planned emissions that will trigger an alert
        vd.set_planned_emissions({2024: Decimal("700")})

        result = vd.run_full_decomposition()
        assert result.years_analyzed == 3
        assert len(result.decomposition_by_year) == 2
        assert len(result.rolling_forecast) == 3  # 3-year default
        assert len(result.cumulative_effects) == 4

        # Verify forecast years are future
        for fp in result.rolling_forecast:
            assert fp.year > 2024

        # Check alerts fired
        assert len(result.alerts) >= 1
        assert result.provenance_hash != ""


# ---------------------------------------------------------------------------
# E2E: VCMI Validation + Assurance Workpaper Generation
# ---------------------------------------------------------------------------


class TestVCMIWithAssurance:
    """Test VCMI validation followed by assurance workpaper generation."""

    def test_vcmi_then_assurance(self):
        # Step 1: Run VCMI validation
        vcmi = VCMIValidationEngine()
        emissions = EmissionsData(
            reporting_year=2025,
            scope1_emissions=Decimal("800"),
            scope2_emissions=Decimal("300"),
            scope3_emissions=Decimal("2000"),
            base_year=2019,
            base_year_emissions=Decimal("5500"),
            reductions_achieved=Decimal("500"),
            has_sbti_target=True,
            target_reduction_pct=Decimal("42"),
            has_public_disclosure=True,
            disclosure_platform="CDP",
            inventory_year=2025,
        )
        credits = CarbonCreditPortfolio(
            total_credits_retired=Decimal("3000"),
            ccp_approved_credits=Decimal("3000"),
            credit_vintage_year=2024,
            registries=["verra"],
            ccp_compliance={f"CCP-{i}": True for i in range(1, 11)},
        )
        vcmi_result = vcmi.validate(emissions, credits, entity_name="TestCo")

        assert vcmi_result.highest_eligible_tier != VCMITier.NOT_ELIGIBLE

        # Step 2: Generate assurance workpapers using VCMI data
        aw = AssuranceWorkpaperEngine()
        total_emissions = emissions.scope1_emissions + emissions.scope2_emissions + emissions.scope3_emissions

        aw.add_methodology(MethodologyEntry(
            scope="scope1", source_name="Combustion",
            calculation_method=CalculationMethod.EMISSION_FACTOR,
        ))

        aw.create_calculation_trace(
            source_name="Scope 1 Combustion",
            scope="scope1",
            activity_data=Decimal("50000"),
            activity_unit="litres",
            emission_factor=Decimal("2.68"),
            ef_unit="kgCO2e/litre",
            ef_source="DEFRA 2025",
        )

        aw.add_exception(ExceptionEntry(
            source_name="Scope 3 estimation",
            scope="scope3",
            severity=ExceptionSeverity.MEDIUM,
            exception_type="estimation",
            description="Spend-based estimate for Cat 1",
            impact_tco2e=Decimal("100"),
        ))

        aw.add_completeness_entry(CompletenessEntry(
            source_name="Scope 1 All",
            scope="scope1",
            data_source_type=DataSourceType.METERED,
            coverage_pct=Decimal("100"),
            is_actual=True,
        ))

        workpapers = aw.generate_workpapers(
            entity_name="TestCo",
            reporting_year=2025,
            total_emissions=total_emissions,
            scope1=emissions.scope1_emissions,
            scope2_location=emissions.scope2_emissions,
            scope2_market=emissions.scope2_emissions,
            scope3=emissions.scope3_emissions,
        )

        assert workpapers.engagement_summary.entity_name == "TestCo"
        assert len(workpapers.cross_checks) >= 1
        assert len(workpapers.provenance_chain) >= 1
        assert workpapers.provenance_hash != ""


# ---------------------------------------------------------------------------
# E2E: Multi-Entity + Target Allocation + Variance Decomposition
# ---------------------------------------------------------------------------


class TestMultiEntityWithDecomposition:
    """Consolidate, allocate targets, then decompose changes."""

    def test_target_allocation_then_decomposition(self):
        me = MultiEntityEngine()
        # Base year
        me.add_entity(EntityEmissions(
            entity_id="div-a", entity_name="Div A", reporting_year=2022,
            scope1_emissions=Decimal("800"), scope2_market=Decimal("300"),
            scope3_emissions=Decimal("1500"), revenue=Decimal("10000000"),
            has_operational_control=True,
        ))
        me.add_entity(EntityEmissions(
            entity_id="div-b", entity_name="Div B", reporting_year=2022,
            scope1_emissions=Decimal("500"), scope2_market=Decimal("200"),
            scope3_emissions=Decimal("1000"), revenue=Decimal("6000000"),
            has_operational_control=True,
        ))
        # Current year
        me.add_entity(EntityEmissions(
            entity_id="div-a", entity_name="Div A", reporting_year=2024,
            scope1_emissions=Decimal("700"), scope2_market=Decimal("250"),
            scope3_emissions=Decimal("1300"), revenue=Decimal("11000000"),
            has_operational_control=True,
        ))
        me.add_entity(EntityEmissions(
            entity_id="div-b", entity_name="Div B", reporting_year=2024,
            scope1_emissions=Decimal("450"), scope2_market=Decimal("180"),
            scope3_emissions=Decimal("900"), revenue=Decimal("6500000"),
            has_operational_control=True,
        ))

        # Allocate targets
        allocations = me.allocate_targets(
            group_target_pct=Decimal("30"),
            base_year=2022, reporting_year=2024,
            allocation_type=TargetAllocationType.TOP_DOWN_PROPORTIONAL,
        )
        assert len(allocations) == 2

        # Use divisional emissions for variance decomposition
        vd = VarianceDecompositionEngine()
        vd.add_segment_data([
            SegmentData(segment_id="div-a", segment_name="Div A", year=2022,
                        emissions=Decimal("2600"), activity=Decimal("10000000")),
            SegmentData(segment_id="div-b", segment_name="Div B", year=2022,
                        emissions=Decimal("1700"), activity=Decimal("6000000")),
            SegmentData(segment_id="div-a", segment_name="Div A", year=2024,
                        emissions=Decimal("2250"), activity=Decimal("11000000")),
            SegmentData(segment_id="div-b", segment_name="Div B", year=2024,
                        emissions=Decimal("1530"), activity=Decimal("6500000")),
        ])
        decomp = vd.decompose_year(2022, 2024)
        assert decomp.segment_count == 2
        assert float(decomp.total_change) < 0  # Emissions decreased


# ---------------------------------------------------------------------------
# E2E: Full Pipeline - All Five Engines
# ---------------------------------------------------------------------------


class TestFullPipelineAllEngines:
    """Comprehensive test using all five engines in sequence."""

    def test_complete_net_zero_assessment(self):
        # 1. Multi-entity consolidation
        me = MultiEntityEngine()
        me.set_group_name("GlobalInc")
        me.add_entity(EntityEmissions(
            entity_id="hq", entity_name="HQ", reporting_year=2025,
            scope1_emissions=Decimal("2000"), scope2_market=Decimal("800"),
            scope3_emissions=Decimal("5000"), revenue=Decimal("50000000"),
            has_operational_control=True, entity_type=EntityType.PARENT,
            hierarchy_level=0,
        ))
        me.add_entity(EntityEmissions(
            entity_id="sub-1", entity_name="Factory", reporting_year=2025,
            scope1_emissions=Decimal("1500"), scope2_market=Decimal("600"),
            scope3_emissions=Decimal("3000"), revenue=Decimal("30000000"),
            has_operational_control=True,
        ))
        group = me.consolidate(2025)
        total = group.total_emissions

        # 2. Temperature scoring
        ts = TemperatureScoringEngine()
        for eid, rev in [("hq", 50000000), ("sub-1", 30000000)]:
            ts.add_entity(PortfolioEntity(
                entity_id=eid, entity_name=eid,
                revenue=Decimal(str(rev)),
                targets=[EmissionsTarget(
                    entity_id=eid, scope=TargetScope.S1S2,
                    timeframe=TargetTimeframe.NEAR_TERM,
                    base_year=2020, target_year=2030,
                    base_year_emissions=Decimal("3000"),
                    target_year_emissions=Decimal("1500"),
                )],
            ))
        temp_result = ts.run_full_assessment()
        assert temp_result.entities_assessed == 2

        # 3. VCMI validation
        vcmi = VCMIValidationEngine()
        vcmi_result = vcmi.validate(
            EmissionsData(
                reporting_year=2025,
                scope1_emissions=Decimal("3500"),
                scope2_emissions=Decimal("1400"),
                scope3_emissions=Decimal("8000"),
                base_year=2020,
                base_year_emissions=Decimal("15000"),
                reductions_achieved=Decimal("500"),
                has_sbti_target=True,
                target_reduction_pct=Decimal("42"),
                has_public_disclosure=True,
                disclosure_platform="CDP",
                inventory_year=2025,
            ),
            CarbonCreditPortfolio(
                total_credits_retired=Decimal("3000"),
                ccp_approved_credits=Decimal("3000"),
                registries=["verra"],
                ccp_compliance={f"CCP-{i}": True for i in range(1, 11)},
            ),
            entity_name="GlobalInc",
        )
        assert vcmi_result.reporting_year == 2025

        # 4. Assurance workpapers
        aw = AssuranceWorkpaperEngine()
        aw.add_methodology(MethodologyEntry(
            scope="scope1", source_name="All Scope 1",
            calculation_method=CalculationMethod.EMISSION_FACTOR,
        ))
        aw.add_completeness_entry(CompletenessEntry(
            source_name="All Scope 1", scope="scope1",
            data_source_type=DataSourceType.METERED,
            coverage_pct=Decimal("95"), is_actual=True,
        ))
        workpapers = aw.generate_workpapers(
            entity_name="GlobalInc", reporting_year=2025,
            total_emissions=total,
            scope1=group.scope1_total,
            scope2_location=group.scope2_location_total,
            scope2_market=group.scope2_market_total,
            scope3=group.scope3_total,
        )
        assert workpapers.engagement_summary.entity_name == "GlobalInc"

        # 5. Verify provenance hashes exist across all results
        assert len(group.provenance_hash) == 64
        assert len(temp_result.provenance_hash) == 64
        assert len(vcmi_result.provenance_hash) == 64
        assert len(workpapers.provenance_hash) == 64
