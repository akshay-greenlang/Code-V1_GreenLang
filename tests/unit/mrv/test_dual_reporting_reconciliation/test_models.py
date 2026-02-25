# -*- coding: utf-8 -*-
"""
Unit tests for Dual Reporting Reconciliation data models.

AGENT-MRV-013: Dual Reporting Reconciliation Agent
Target: 80 tests covering all 22 enums, 25 models, and 7 constant tables.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict

import pytest

from greenlang.dual_reporting_reconciliation.models import (
    # Module-level constants
    AGENT_ID,
    AGENT_COMPONENT,
    VERSION,
    TABLE_PREFIX,
    MAX_UPSTREAM_RESULTS,
    MAX_BATCH_PERIODS,
    MAX_FACILITIES,
    MAX_DISCREPANCIES,
    MAX_FRAMEWORKS,
    MAX_TREND_POINTS,
    MAX_REQUIREMENTS_PER_FRAMEWORK,
    DEFAULT_CONFIDENCE_LEVEL,
    DECIMAL_INF,
    DECIMAL_PLACES,
    ZERO,
    ONE_HUNDRED,
    # Enumerations
    EnergyType,
    Scope2Method,
    UpstreamAgent,
    DiscrepancyType,
    DiscrepancyDirection,
    MaterialityLevel,
    QualityDimension,
    QualityGrade,
    EFHierarchyPriority,
    ReportingFramework,
    FlagType,
    FlagSeverity,
    ReconciliationStatus,
    IntensityMetric,
    TrendDirection,
    PipelineStage,
    ExportFormat,
    ComplianceStatus,
    DataQualityTier,
    GWPSource,
    EmissionGas,
    BatchStatus,
    # Constant tables
    GWP_VALUES,
    MATERIALITY_THRESHOLDS,
    QUALITY_WEIGHTS,
    QUALITY_GRADE_THRESHOLDS,
    EF_HIERARCHY_QUALITY_SCORES,
    RESIDUAL_MIX_FACTORS,
    UPSTREAM_AGENT_MAPPING,
    FRAMEWORK_REQUIRED_DISCLOSURES,
    # Data models
    ResidualMixFactor,
    UpstreamResult,
    EnergyTypeBreakdown,
    FacilityBreakdown,
    ReconciliationWorkspace,
    Discrepancy,
    WaterfallItem,
    WaterfallDecomposition,
    Flag,
    DiscrepancyReport,
    QualityScore,
    QualityAssessment,
    FrameworkTable,
    ReportingTableSet,
    TrendDataPoint,
    TrendReport,
    IntensityResult,
    ComplianceRequirement,
    ComplianceCheckResult,
    ReconciliationRequest,
    ReconciliationReport,
    BatchReconciliationRequest,
    BatchReconciliationResult,
    ExportRequest,
    AggregationResult,
    # Aliases
    DiscrepancyItem,
    ComplianceIssue,
)


# ===========================================================================
# 1. Module Constants
# ===========================================================================


class TestModuleConstants:
    """Test module-level constant values."""

    def test_agent_id(self):
        assert AGENT_ID == "GL-MRV-X-024"

    def test_agent_component(self):
        assert AGENT_COMPONENT == "AGENT-MRV-013"

    def test_version(self):
        assert VERSION == "1.0.0"

    def test_table_prefix(self):
        assert TABLE_PREFIX == "gl_drr_"

    def test_max_upstream_results(self):
        assert MAX_UPSTREAM_RESULTS == 10_000

    def test_max_batch_periods(self):
        assert MAX_BATCH_PERIODS == 120

    def test_max_facilities(self):
        assert MAX_FACILITIES == 50_000

    def test_max_discrepancies(self):
        assert MAX_DISCREPANCIES == 5_000

    def test_max_frameworks(self):
        assert MAX_FRAMEWORKS == 20

    def test_max_trend_points(self):
        assert MAX_TREND_POINTS == 240

    def test_max_requirements_per_framework(self):
        assert MAX_REQUIREMENTS_PER_FRAMEWORK == 200

    def test_default_confidence_level(self):
        assert DEFAULT_CONFIDENCE_LEVEL == Decimal("95.0")

    def test_decimal_inf(self):
        assert DECIMAL_INF == Decimal("Infinity")

    def test_decimal_places(self):
        assert DECIMAL_PLACES == 8

    def test_zero(self):
        assert ZERO == Decimal("0")

    def test_one_hundred(self):
        assert ONE_HUNDRED == Decimal("100")


# ===========================================================================
# 2. Enumerations (22)
# ===========================================================================


class TestEnums:
    """Test all 22 enumerations."""

    def test_energy_type_values(self):
        assert len(EnergyType) == 4
        assert EnergyType.ELECTRICITY.value == "electricity"
        assert EnergyType.STEAM.value == "steam"
        assert EnergyType.DISTRICT_HEATING.value == "district_heating"
        assert EnergyType.DISTRICT_COOLING.value == "district_cooling"

    def test_scope2_method_values(self):
        assert len(Scope2Method) == 2
        assert Scope2Method.LOCATION_BASED.value == "location_based"
        assert Scope2Method.MARKET_BASED.value == "market_based"

    def test_upstream_agent_values(self):
        assert len(UpstreamAgent) == 4
        assert UpstreamAgent.MRV_009.value == "mrv_009"
        assert UpstreamAgent.MRV_012.value == "mrv_012"

    def test_discrepancy_type_values(self):
        assert len(DiscrepancyType) == 8
        assert DiscrepancyType.REC_GO_IMPACT.value == "rec_go_impact"
        assert DiscrepancyType.RESIDUAL_MIX_UPLIFT.value == "residual_mix_uplift"
        assert DiscrepancyType.SUPPLIER_EF_DELTA.value == "supplier_ef_delta"

    def test_discrepancy_direction_values(self):
        assert len(DiscrepancyDirection) == 3
        assert DiscrepancyDirection.MARKET_LOWER.value == "market_lower"
        assert DiscrepancyDirection.MARKET_HIGHER.value == "market_higher"
        assert DiscrepancyDirection.EQUAL.value == "equal"

    def test_materiality_level_values(self):
        assert len(MaterialityLevel) == 5
        assert MaterialityLevel.IMMATERIAL.value == "immaterial"
        assert MaterialityLevel.EXTREME.value == "extreme"

    def test_quality_dimension_values(self):
        assert len(QualityDimension) == 4

    def test_quality_grade_values(self):
        assert len(QualityGrade) == 5

    def test_ef_hierarchy_priority_values(self):
        assert len(EFHierarchyPriority) >= 5

    def test_reporting_framework_values(self):
        assert len(ReportingFramework) == 7

    def test_flag_type_values(self):
        assert len(FlagType) >= 3

    def test_flag_severity_values(self):
        assert len(FlagSeverity) >= 3

    def test_reconciliation_status_values(self):
        assert len(ReconciliationStatus) >= 3

    def test_intensity_metric_values(self):
        assert len(IntensityMetric) == 4
        assert IntensityMetric.REVENUE.value == "revenue"

    def test_trend_direction_values(self):
        assert len(TrendDirection) == 3
        assert TrendDirection.STABLE.value == "stable"

    def test_pipeline_stage_values(self):
        assert len(PipelineStage) == 10

    def test_export_format_values(self):
        assert len(ExportFormat) >= 3

    def test_compliance_status_values(self):
        assert len(ComplianceStatus) >= 3

    def test_data_quality_tier_values(self):
        assert len(DataQualityTier) == 3

    def test_gwp_source_values(self):
        assert len(GWPSource) >= 3

    def test_emission_gas_values(self):
        assert len(EmissionGas) >= 3

    def test_batch_status_values(self):
        assert len(BatchStatus) >= 4


# ===========================================================================
# 3. Constant Tables
# ===========================================================================


class TestConstantTables:
    """Test constant table structures."""

    def test_gwp_values_has_ar5(self):
        assert "AR5" in GWP_VALUES or "ar5" in str(GWP_VALUES).lower()

    def test_materiality_thresholds_structure(self):
        assert isinstance(MATERIALITY_THRESHOLDS, dict)
        assert len(MATERIALITY_THRESHOLDS) >= 4

    def test_quality_weights_sum_to_one(self):
        total = sum(QUALITY_WEIGHTS.values())
        assert abs(float(total) - 1.0) < 0.01

    def test_quality_weights_has_four_dimensions(self):
        assert len(QUALITY_WEIGHTS) == 4

    def test_quality_grade_thresholds(self):
        assert isinstance(QUALITY_GRADE_THRESHOLDS, dict)
        assert len(QUALITY_GRADE_THRESHOLDS) >= 4

    def test_ef_hierarchy_quality_scores(self):
        assert isinstance(EF_HIERARCHY_QUALITY_SCORES, dict)
        assert len(EF_HIERARCHY_QUALITY_SCORES) >= 4

    def test_residual_mix_factors(self):
        assert isinstance(RESIDUAL_MIX_FACTORS, dict)
        assert len(RESIDUAL_MIX_FACTORS) >= 10

    def test_upstream_agent_mapping(self):
        assert isinstance(UPSTREAM_AGENT_MAPPING, dict)
        assert len(UPSTREAM_AGENT_MAPPING) >= 4

    def test_framework_required_disclosures(self):
        assert isinstance(FRAMEWORK_REQUIRED_DISCLOSURES, dict)
        assert len(FRAMEWORK_REQUIRED_DISCLOSURES) >= 5


# ===========================================================================
# 4. Data Models
# ===========================================================================


class TestUpstreamResult:
    """Test UpstreamResult model creation and validation."""

    def test_create_basic(self, sample_location_result):
        result = UpstreamResult(**sample_location_result)
        assert result.facility_id == "FAC-001"
        assert result.energy_type == EnergyType.ELECTRICITY
        assert result.method == Scope2Method.LOCATION_BASED

    def test_emissions_value(self, sample_location_result):
        result = UpstreamResult(**sample_location_result)
        assert result.emissions_tco2e == Decimal("1250.50")

    def test_market_result(self, sample_market_result):
        result = UpstreamResult(**sample_market_result)
        assert result.method == Scope2Method.MARKET_BASED
        assert result.ef_hierarchy == EFHierarchyPriority.SUPPLIER_NO_CERT


class TestReconciliationWorkspace:
    """Test ReconciliationWorkspace model."""

    def test_create_empty(self):
        ws = ReconciliationWorkspace(
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            location_results=[],
            market_results=[],
        )
        assert ws.total_location_tco2e == Decimal("0")
        assert ws.total_market_tco2e == Decimal("0")


class TestDiscrepancy:
    """Test Discrepancy model."""

    def test_create_basic(self):
        d = Discrepancy(
            discrepancy_type=DiscrepancyType.REC_GO_IMPACT,
            direction=DiscrepancyDirection.MARKET_LOWER,
            materiality=MaterialityLevel.MATERIAL,
            absolute_tco2e=Decimal("625.25"),
            percentage=Decimal("50.0"),
            description="REC/GO impact drives 50% reduction in market-based",
        )
        assert d.discrepancy_type == DiscrepancyType.REC_GO_IMPACT
        assert d.direction == DiscrepancyDirection.MARKET_LOWER


class TestTrendDataPoint:
    """Test TrendDataPoint model."""

    def test_create_basic(self):
        tdp = TrendDataPoint(
            period="2024",
            location_tco2e=Decimal("1800.0"),
            market_tco2e=Decimal("1000.0"),
            pif=Decimal("0.4444"),
            re100_pct=Decimal("55.0"),
        )
        assert tdp.period == "2024"
        assert tdp.location_tco2e == Decimal("1800.0")


class TestComplianceRequirement:
    """Test ComplianceRequirement model."""

    def test_create_basic(self):
        cr = ComplianceRequirement(
            requirement_id="GHG-001",
            description="Report both location-based and market-based totals",
            met=True,
        )
        assert cr.requirement_id == "GHG-001"
        assert cr.met is True


class TestReconciliationRequest:
    """Test ReconciliationRequest model."""

    def test_create_with_mandatory_fields(self):
        req = ReconciliationRequest(
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )
        assert req.tenant_id == "tenant-001"


# ===========================================================================
# 5. Type Aliases
# ===========================================================================


class TestTypeAliases:
    """Test backward-compatible type aliases."""

    def test_discrepancy_item_is_discrepancy(self):
        assert DiscrepancyItem is Discrepancy

    def test_compliance_issue_is_compliance_requirement(self):
        assert ComplianceIssue is ComplianceRequirement
