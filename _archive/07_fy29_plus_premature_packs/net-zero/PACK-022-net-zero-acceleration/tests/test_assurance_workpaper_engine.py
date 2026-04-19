# -*- coding: utf-8 -*-
"""
Unit tests for AssuranceWorkpaperEngine (PACK-022 Engine 10).

Tests ISAE 3410 workpaper generation including engagement summaries,
methodology documentation, calculation traces, data lineage, control
evidence, exception registers, completeness matrices, change registers,
cross-checks, and provenance chains.
"""

import json
import pytest
from decimal import Decimal

from engines.assurance_workpaper_engine import (
    AssuranceWorkpaperEngine,
    AssuranceWorkpaperConfig,
    AssuranceResult,
    EngagementSummary,
    MethodologyEntry,
    CalculationStep,
    CalculationTrace,
    DataLineageEntry,
    ControlEvidence,
    ExceptionEntry,
    CompletenessEntry,
    ChangeEntry,
    CrossCheckResult,
    AssuranceLevel,
    WorkpaperSection,
    MaterialityBasis,
    DataSourceType,
    CalculationMethod,
    ExceptionSeverity,
    CrossCheckStatus,
    DEFAULT_MATERIALITY_PCT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return AssuranceWorkpaperEngine()


@pytest.fixture
def loaded_engine():
    """Engine pre-loaded with sample workpaper data."""
    eng = AssuranceWorkpaperEngine()
    eng.add_methodology(MethodologyEntry(
        scope="scope1",
        source_name="Natural Gas Combustion",
        calculation_method=CalculationMethod.EMISSION_FACTOR,
        emission_factor_source="DEFRA 2025",
        emission_factor_value=Decimal("2.02"),
        emission_factor_unit="kgCO2e/m3",
    ))
    eng.add_methodology(MethodologyEntry(
        scope="scope2",
        source_name="Grid Electricity",
        calculation_method=CalculationMethod.EMISSION_FACTOR,
        emission_factor_source="IEA 2024",
        emission_factor_value=Decimal("0.42"),
        emission_factor_unit="kgCO2e/kWh",
    ))
    eng.create_calculation_trace(
        source_name="Natural Gas",
        scope="scope1",
        activity_data=Decimal("50000"),
        activity_unit="m3",
        emission_factor=Decimal("2.02"),
        ef_unit="kgCO2e/m3",
        ef_source="DEFRA 2025",
    )
    eng.add_data_lineage(DataLineageEntry(
        data_point_name="Gas Consumption",
        source_system="ERP",
        source_type=DataSourceType.METERED,
        raw_value=Decimal("50000"),
        raw_unit="m3",
        transformations=["Unit conversion", "Gap fill Feb"],
        final_value=Decimal("50000"),
        final_unit="m3",
    ))
    eng.add_control_evidence(ControlEvidence(
        control_name="Gas meter reconciliation",
        control_type="reconciliation",
        expected_value=Decimal("50000"),
        actual_value=Decimal("49800"),
        tolerance_pct=Decimal("1"),
        status=CrossCheckStatus.PASSED,
    ))
    eng.add_exception(ExceptionEntry(
        source_name="Fleet Diesel",
        scope="scope1",
        severity=ExceptionSeverity.MEDIUM,
        exception_type="estimation",
        description="Q3 diesel estimated from Q2 extrapolation",
        impact_tco2e=Decimal("15"),
    ))
    eng.add_completeness_entry(CompletenessEntry(
        source_name="Natural Gas",
        scope="scope1",
        data_source_type=DataSourceType.METERED,
        coverage_pct=Decimal("100"),
        is_actual=True,
    ))
    eng.add_completeness_entry(CompletenessEntry(
        source_name="Fleet Diesel",
        scope="scope1",
        data_source_type=DataSourceType.ESTIMATED,
        coverage_pct=Decimal("75"),
        is_actual=False,
        estimation_method="Linear extrapolation from Q2",
        months_actual=9,
        months_estimated=3,
    ))
    eng.add_change(ChangeEntry(
        change_type="methodology",
        description="Updated gas emission factor from DEFRA 2024 to 2025",
        affected_scope="scope1",
        affected_source="Natural Gas",
        previous_value=Decimal("98"),
        new_value=Decimal("101"),
        rationale="Annual EF update",
        approved_by="Sustainability Director",
    ))
    return eng


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestAssuranceInit:

    def test_default_config(self, engine):
        assert isinstance(engine.config, AssuranceWorkpaperConfig)
        assert engine.config.default_assurance_level == AssuranceLevel.LIMITED
        assert engine.config.materiality_pct == DEFAULT_MATERIALITY_PCT

    def test_custom_config(self):
        eng = AssuranceWorkpaperEngine({"materiality_pct": "3", "cross_check_tolerance_pct": "2"})
        assert eng.config.materiality_pct == Decimal("3")
        assert eng.config.cross_check_tolerance_pct == Decimal("2")

    def test_config_object(self):
        cfg = AssuranceWorkpaperConfig(default_assurance_level=AssuranceLevel.REASONABLE)
        eng = AssuranceWorkpaperEngine(cfg)
        assert eng.config.default_assurance_level == AssuranceLevel.REASONABLE


# ---------------------------------------------------------------------------
# Engagement Summary Tests
# ---------------------------------------------------------------------------


class TestEngagementSummary:

    def test_create_summary(self, engine):
        summary = engine.create_engagement_summary(
            entity_name="TestCo", reporting_year=2025,
            total_emissions=Decimal("10000"),
        )
        assert isinstance(summary, EngagementSummary)
        assert summary.entity_name == "TestCo"
        assert summary.reporting_year == 2025
        assert summary.assurance_level == AssuranceLevel.LIMITED
        assert len(summary.provenance_hash) == 64

    def test_materiality_calculated(self, engine):
        summary = engine.create_engagement_summary(
            entity_name="TestCo", reporting_year=2025,
            total_emissions=Decimal("10000"),
        )
        # 5% of 10000 = 500
        assert float(summary.materiality_threshold) == pytest.approx(500.0, rel=1e-3)

    def test_custom_assurance_level(self, engine):
        summary = engine.create_engagement_summary(
            entity_name="TestCo", reporting_year=2025,
            total_emissions=Decimal("10000"),
            assurance_level=AssuranceLevel.REASONABLE,
        )
        assert summary.assurance_level == AssuranceLevel.REASONABLE


# ---------------------------------------------------------------------------
# Methodology Documentation Tests
# ---------------------------------------------------------------------------


class TestMethodologyDocumentation:

    def test_add_methodology(self, engine):
        entry = MethodologyEntry(
            scope="scope1",
            source_name="Diesel",
            calculation_method=CalculationMethod.EMISSION_FACTOR,
        )
        result = engine.add_methodology(entry)
        assert len(result.provenance_hash) == 64
        assert len(engine._methodology) == 1


# ---------------------------------------------------------------------------
# Calculation Trace Tests
# ---------------------------------------------------------------------------


class TestCalculationTrace:

    def test_create_trace_three_steps(self, engine):
        trace = engine.create_calculation_trace(
            source_name="Diesel",
            scope="scope1",
            activity_data=Decimal("10000"),
            activity_unit="litres",
            emission_factor=Decimal("2.68"),
            ef_unit="kgCO2e/litre",
            ef_source="DEFRA 2025",
        )
        assert len(trace.steps) == 3
        assert trace.steps[0].step_number == 1
        assert trace.steps[2].step_number == 3
        assert float(trace.final_emissions) > 0

    def test_trace_with_gwp(self, engine):
        trace = engine.create_calculation_trace(
            source_name="CH4 fugitive",
            scope="scope1",
            activity_data=Decimal("100"),
            activity_unit="kg",
            emission_factor=Decimal("1.0"),
            ef_unit="kgCH4/kg",
            ef_source="IPCC",
            gwp=Decimal("27"),
            gwp_gas="CH4",
        )
        assert float(trace.final_emissions) == pytest.approx(2700.0, rel=1e-3)

    def test_trace_provenance_hash(self, engine):
        trace = engine.create_calculation_trace(
            source_name="Gas",
            scope="scope1",
            activity_data=Decimal("1000"),
            activity_unit="m3",
            emission_factor=Decimal("2.0"),
            ef_unit="kgCO2e/m3",
            ef_source="Test",
        )
        assert len(trace.provenance_hash) == 64


# ---------------------------------------------------------------------------
# Data Lineage Tests
# ---------------------------------------------------------------------------


class TestDataLineage:

    def test_add_lineage(self, engine):
        entry = DataLineageEntry(
            data_point_name="Electricity kWh",
            source_system="Smart Meter",
            source_type=DataSourceType.METERED,
            raw_value=Decimal("100000"),
            final_value=Decimal("100000"),
        )
        result = engine.add_data_lineage(entry)
        assert result.source_hash != ""
        assert result.transformation_hash != ""
        assert result.output_hash != ""
        assert result.provenance_hash != ""


# ---------------------------------------------------------------------------
# Control Evidence Tests
# ---------------------------------------------------------------------------


class TestControlEvidence:

    def test_auto_evaluate_passed(self, engine):
        control = ControlEvidence(
            control_name="Test",
            control_type="reconciliation",
            expected_value=Decimal("100"),
            actual_value=Decimal("100.5"),
            tolerance_pct=Decimal("1"),
            status=CrossCheckStatus.SKIPPED,
        )
        result = engine.add_control_evidence(control)
        assert result.status == CrossCheckStatus.PASSED

    def test_auto_evaluate_warning(self, engine):
        control = ControlEvidence(
            control_name="Test",
            control_type="reconciliation",
            expected_value=Decimal("100"),
            actual_value=Decimal("101.5"),
            tolerance_pct=Decimal("1"),
            status=CrossCheckStatus.SKIPPED,
        )
        result = engine.add_control_evidence(control)
        assert result.status == CrossCheckStatus.WARNING

    def test_auto_evaluate_failed(self, engine):
        control = ControlEvidence(
            control_name="Test",
            control_type="reconciliation",
            expected_value=Decimal("100"),
            actual_value=Decimal("120"),
            tolerance_pct=Decimal("1"),
            status=CrossCheckStatus.SKIPPED,
        )
        result = engine.add_control_evidence(control)
        assert result.status == CrossCheckStatus.FAILED


# ---------------------------------------------------------------------------
# Exception Register Tests
# ---------------------------------------------------------------------------


class TestExceptionRegister:

    def test_add_exception(self, engine):
        exc = ExceptionEntry(
            source_name="Diesel",
            severity=ExceptionSeverity.HIGH,
            exception_type="data_quality",
            description="Missing invoices for Q4",
            impact_tco2e=Decimal("50"),
        )
        result = engine.add_exception(exc)
        assert len(result.provenance_hash) == 64


# ---------------------------------------------------------------------------
# Completeness Matrix Tests
# ---------------------------------------------------------------------------


class TestCompletenessMatrix:

    def test_add_completeness(self, engine):
        entry = CompletenessEntry(
            source_name="Gas",
            scope="scope1",
            data_source_type=DataSourceType.METERED,
            coverage_pct=Decimal("100"),
            is_actual=True,
        )
        result = engine.add_completeness_entry(entry)
        assert len(result.provenance_hash) == 64


# ---------------------------------------------------------------------------
# Change Register Tests
# ---------------------------------------------------------------------------


class TestChangeRegister:

    def test_add_change_auto_impact(self, engine):
        change = ChangeEntry(
            change_type="methodology",
            description="EF update",
            previous_value=Decimal("100"),
            new_value=Decimal("110"),
        )
        result = engine.add_change(change)
        assert result.impact == Decimal("10")

    def test_explicit_impact_preserved(self, engine):
        change = ChangeEntry(
            change_type="boundary",
            description="Added new facility",
            previous_value=Decimal("0"),
            new_value=Decimal("500"),
            impact=Decimal("500"),
        )
        result = engine.add_change(change)
        assert result.impact == Decimal("500")


# ---------------------------------------------------------------------------
# Cross-Check Tests
# ---------------------------------------------------------------------------


class TestCrossChecks:

    def test_sum_check_passes(self, engine):
        checks = engine.run_cross_checks(
            scope1=Decimal("100"),
            scope2_location=Decimal("50"),
            scope2_market=Decimal("50"),
            scope3=Decimal("300"),
            reported_total=Decimal("450"),
        )
        sum_check = next(c for c in checks if "Sum" in c.check_name)
        assert sum_check.status == CrossCheckStatus.PASSED

    def test_sum_check_fails(self, engine):
        checks = engine.run_cross_checks(
            scope1=Decimal("100"),
            scope2_location=Decimal("50"),
            scope2_market=Decimal("50"),
            scope3=Decimal("300"),
            reported_total=Decimal("600"),
        )
        sum_check = next(c for c in checks if "Sum" in c.check_name)
        assert sum_check.status == CrossCheckStatus.FAILED

    def test_yoy_trend_warning(self, engine):
        checks = engine.run_cross_checks(
            scope1=Decimal("100"),
            scope2_location=Decimal("50"),
            scope2_market=Decimal("50"),
            scope3=Decimal("300"),
            reported_total=Decimal("450"),
            prior_year_total=Decimal("250"),
        )
        yoy = next((c for c in checks if "Year" in c.check_name), None)
        assert yoy is not None
        assert yoy.status == CrossCheckStatus.WARNING

    def test_scope2_reasonableness(self, engine):
        checks = engine.run_cross_checks(
            scope1=Decimal("100"),
            scope2_location=Decimal("50"),
            scope2_market=Decimal("45"),
            scope3=Decimal("300"),
            reported_total=Decimal("445"),
        )
        s2_check = next((c for c in checks if "Scope 2" in c.check_name), None)
        assert s2_check is not None

    def test_scope3_proportion_warning_low(self, engine):
        checks = engine.run_cross_checks(
            scope1=Decimal("900"),
            scope2_location=Decimal("90"),
            scope2_market=Decimal("90"),
            scope3=Decimal("10"),
            reported_total=Decimal("1000"),
        )
        s3 = next((c for c in checks if "Scope 3 Proportion" in c.check_name), None)
        assert s3 is not None
        assert s3.status == CrossCheckStatus.WARNING


# ---------------------------------------------------------------------------
# Materiality Calculation Tests
# ---------------------------------------------------------------------------


class TestMateriality:

    def test_default_materiality(self, engine):
        result = engine.calculate_materiality(Decimal("10000"))
        assert float(result) == pytest.approx(500.0, rel=1e-3)

    def test_custom_materiality_pct(self):
        eng = AssuranceWorkpaperEngine({"materiality_pct": "3"})
        result = eng.calculate_materiality(Decimal("10000"))
        assert float(result) == pytest.approx(300.0, rel=1e-3)

    def test_zero_emissions(self, engine):
        result = engine.calculate_materiality(Decimal("0"))
        assert result == Decimal("0")


# ---------------------------------------------------------------------------
# Provenance Chain Tests
# ---------------------------------------------------------------------------


class TestProvenanceChain:

    def test_chain_genesis(self, engine):
        chain = engine.build_provenance_chain()
        assert len(chain) >= 1
        assert len(chain[0]) == 64

    def test_chain_grows_with_data(self, loaded_engine):
        chain = loaded_engine.build_provenance_chain()
        # 1 genesis + 2 methodology + 1 trace + 1 lineage + 1 control + 1 exception
        assert len(chain) >= 7


# ---------------------------------------------------------------------------
# Full Workpaper Generation Tests
# ---------------------------------------------------------------------------


class TestGenerateWorkpapers:

    def test_generate_structure(self, loaded_engine):
        result = loaded_engine.generate_workpapers(
            entity_name="TestCo", reporting_year=2025,
            total_emissions=Decimal("10000"),
            scope1=Decimal("4000"),
            scope2_location=Decimal("2000"),
            scope2_market=Decimal("1800"),
            scope3=Decimal("4200"),
        )
        assert isinstance(result, AssuranceResult)
        assert result.engagement_summary.entity_name == "TestCo"
        assert len(result.methodology_entries) == 2
        assert len(result.calculation_traces) == 1
        assert len(result.data_lineage) == 1
        assert len(result.control_evidence) == 1
        assert len(result.exceptions) == 1
        assert len(result.completeness_matrix) == 2
        assert len(result.change_register) == 1
        assert len(result.cross_checks) >= 1
        assert len(result.provenance_chain) >= 1
        assert len(result.provenance_hash) == 64

    def test_material_exception_flagging(self, loaded_engine):
        result = loaded_engine.generate_workpapers(
            entity_name="TestCo", reporting_year=2025,
            total_emissions=Decimal("100"),
        )
        # materiality = 5% of 100 = 5; exception has impact 15 > 5
        assert result.material_exceptions_count >= 1

    def test_overall_completeness(self, loaded_engine):
        result = loaded_engine.generate_workpapers(
            entity_name="TestCo", reporting_year=2025,
            total_emissions=Decimal("10000"),
        )
        # 1 actual out of 2 sources = 50%
        assert float(result.overall_completeness_pct) == pytest.approx(50.0, rel=1e-1)


# ---------------------------------------------------------------------------
# Export Tests
# ---------------------------------------------------------------------------


class TestExport:

    def test_export_to_json(self, loaded_engine):
        result = loaded_engine.generate_workpapers(
            entity_name="TestCo", reporting_year=2025,
            total_emissions=Decimal("10000"),
        )
        json_str = loaded_engine.export_to_json(result)
        data = json.loads(json_str)
        assert "engagement_summary" in data
        assert "methodology_entries" in data


# ---------------------------------------------------------------------------
# Section Summary & Clear Tests
# ---------------------------------------------------------------------------


class TestUtilities:

    def test_section_summary(self, loaded_engine):
        summary = loaded_engine.get_section_summary()
        assert summary["methodology_documentation"] == 2
        assert summary["calculation_trace"] == 1
        assert summary["data_lineage"] == 1
        assert summary["control_evidence"] == 1
        assert summary["exception_register"] == 1
        assert summary["completeness_matrix"] == 2
        assert summary["change_register"] == 1

    def test_clear(self, loaded_engine):
        loaded_engine.clear()
        summary = loaded_engine.get_section_summary()
        assert all(v == 0 for v in summary.values())


# ---------------------------------------------------------------------------
# Edge Cases & Constants
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_default_materiality_pct(self):
        assert DEFAULT_MATERIALITY_PCT == Decimal("5")

    def test_enum_values(self):
        assert AssuranceLevel.LIMITED.value == "limited"
        assert AssuranceLevel.REASONABLE.value == "reasonable"
        assert WorkpaperSection.CALCULATION_TRACE.value == "calculation_trace"
        assert DataSourceType.METERED.value == "metered"
        assert ExceptionSeverity.CRITICAL.value == "critical"
        assert CrossCheckStatus.PASSED.value == "passed"
