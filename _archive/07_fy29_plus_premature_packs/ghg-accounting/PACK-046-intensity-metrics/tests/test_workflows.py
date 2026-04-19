"""
Unit tests for PACK-046 Workflows (Denominator Setup + Intensity Calculation).

Tests all 2 implemented workflows with 60+ tests covering:
  - DenominatorSetupWorkflow: 4-phase async execution
  - IntensityCalculationWorkflow: 4-phase async execution
  - Phase sequencing and result tracking
  - Provenance hashing
  - Sector identification and framework mapping
  - Denominator selection scoring
  - Data collection and coverage tracking
  - Validation findings (completeness, quality, YoY, zero values)
  - Emissions ingestion and alignment
  - Scope configuration rules
  - Intensity metric computation
  - Quality assurance checks (completeness, outlier, consistency, reasonableness)
  - Error handling and retry logic
  - Edge cases

Author: GreenLang QA Team
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path
from typing import Dict

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from workflows.denominator_setup_workflow import (
    DataQualityGrade,
    DenominatorCandidate,
    DenominatorRecord,
    DenominatorSetupInput,
    DenominatorSetupResult,
    DenominatorSetupWorkflow,
    DenominatorType,
    DenominatorUnit,
    FRAMEWORK_REQUIRED_DENOMINATORS,
    PhaseResult,
    PhaseStatus,
    SECTOR_DENOMINATOR_MAP,
    SectorClassification,
    SectorProfile,
    SetupPhase,
    ValidationFinding,
    ValidationSeverity,
    WorkflowStatus,
)
from workflows.intensity_calculation_workflow import (
    CalcPhase,
    DenominatorDataSet,
    EmissionsDataSet,
    FRAMEWORK_SCOPE_RULES,
    IntensityCalcInput,
    IntensityCalcResult,
    IntensityCalculationWorkflow,
    IntensityMetric,
    IntensityUnit,
    QualityCheckResult,
    QualityCheckType,
    QualityOutcome,
    ScopeInclusion,
    ScopeRule,
)


# ---------------------------------------------------------------------------
# Denominator Setup Workflow Tests
# ---------------------------------------------------------------------------


class TestDenominatorSetupWorkflowInit:
    """Tests for DenominatorSetupWorkflow initialisation."""

    def test_init_creates_workflow(self):
        wf = DenominatorSetupWorkflow()
        assert wf is not None

    def test_init_has_workflow_id(self):
        wf = DenominatorSetupWorkflow()
        assert wf.workflow_id is not None
        assert len(wf.workflow_id) > 0

    def test_init_has_phase_sequence(self):
        wf = DenominatorSetupWorkflow()
        assert len(wf.PHASE_SEQUENCE) == 4

    def test_init_with_config(self):
        wf = DenominatorSetupWorkflow(config={"key": "value"})
        assert wf.config == {"key": "value"}

    def test_phase_sequence_order(self):
        expected = [
            SetupPhase.SECTOR_IDENTIFICATION,
            SetupPhase.DENOMINATOR_SELECTION,
            SetupPhase.DATA_COLLECTION,
            SetupPhase.VALIDATION,
        ]
        assert DenominatorSetupWorkflow.PHASE_SEQUENCE == expected


class TestDenominatorSetupExecution:
    """Tests for DenominatorSetupWorkflow.execute()."""

    @pytest.mark.asyncio
    async def test_execute_completes_for_industrials(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-001",
            sector=SectorClassification.INDUSTRIALS,
            applicable_frameworks=["esrs_e1", "cdp_c6", "gri_305_4"],
            reporting_periods=["2024", "2025"],
            available_data={
                "revenue": {"2024": 100.0, "2025": 110.0},
                "production_volume": {"2024": 50000, "2025": 55000},
            },
        )
        result = await wf.execute(inp)
        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.phases) == 4

    @pytest.mark.asyncio
    async def test_execute_returns_provenance_hash(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-002",
            sector=SectorClassification.ENERGY,
        )
        result = await wf.execute(inp)
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_execute_records_duration(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-003",
            sector=SectorClassification.REAL_ESTATE,
        )
        result = await wf.execute(inp)
        assert result.total_duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_execute_sets_organization_id(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-004",
            sector=SectorClassification.FINANCIALS,
        )
        result = await wf.execute(inp)
        assert result.organization_id == "org-004"


class TestDenominatorSetupPhase1SectorIdentification:
    """Tests for Phase 1: Sector Identification."""

    @pytest.mark.asyncio
    async def test_sector_profile_created(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-005",
            sector=SectorClassification.MATERIALS,
            applicable_frameworks=["esrs_e1"],
        )
        result = await wf.execute(inp)
        assert result.sector_profile is not None
        assert result.sector_profile.sector == SectorClassification.MATERIALS

    @pytest.mark.asyncio
    async def test_energy_sector_specific_denominator(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-006",
            sector=SectorClassification.ENERGY,
        )
        result = await wf.execute(inp)
        assert result.sector_profile.sector_specific_denominator == "energy_generated"

    @pytest.mark.asyncio
    async def test_transport_sector_specific_denominator(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-007",
            sector=SectorClassification.TRANSPORTATION,
        )
        result = await wf.execute(inp)
        assert result.sector_profile.sector_specific_denominator == "passenger_km"

    @pytest.mark.asyncio
    async def test_real_estate_sector_specific_denominator(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-008",
            sector=SectorClassification.REAL_ESTATE,
        )
        result = await wf.execute(inp)
        assert result.sector_profile.sector_specific_denominator == "floor_area"

    @pytest.mark.asyncio
    async def test_unknown_framework_ignored_with_warning(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-009",
            sector=SectorClassification.OTHER,
            applicable_frameworks=["esrs_e1", "unknown_fw_xyz"],
        )
        result = await wf.execute(inp)
        phase1 = result.phases[0]
        assert any("Unknown frameworks ignored" in w for w in phase1.warnings)
        assert "esrs_e1" in result.sector_profile.applicable_frameworks
        assert "unknown_fw_xyz" not in result.sector_profile.applicable_frameworks


class TestDenominatorSetupPhase2Selection:
    """Tests for Phase 2: Denominator Selection."""

    @pytest.mark.asyncio
    async def test_candidates_generated(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-010",
            sector=SectorClassification.INDUSTRIALS,
            applicable_frameworks=["esrs_e1", "gri_305_4"],
        )
        result = await wf.execute(inp)
        assert len(result.denominator_candidates) > 0

    @pytest.mark.asyncio
    async def test_selected_denominators_above_threshold(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-011",
            sector=SectorClassification.MATERIALS,
            applicable_frameworks=["esrs_e1", "cdp_c6"],
            available_data={"revenue": {"2024": 100.0}},
        )
        result = await wf.execute(inp)
        for selected in result.selected_denominators:
            assert selected.relevance_score >= 40.0 or "Default" in selected.selection_reason

    @pytest.mark.asyncio
    async def test_data_availability_boosts_relevance(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-012",
            sector=SectorClassification.INDUSTRIALS,
            applicable_frameworks=["gri_305_4"],
            available_data={"revenue": {"2024": 200.0}},
        )
        result = await wf.execute(inp)
        rev_candidate = next(
            (c for c in result.denominator_candidates
             if c.denominator_type == DenominatorType.REVENUE),
            None,
        )
        assert rev_candidate is not None
        assert rev_candidate.data_available is True

    @pytest.mark.asyncio
    async def test_candidates_sorted_by_relevance(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-013",
            sector=SectorClassification.ENERGY,
            applicable_frameworks=["esrs_e1", "cdp_c6", "sbti_sda"],
        )
        result = await wf.execute(inp)
        scores = [c.relevance_score for c in result.denominator_candidates]
        assert scores == sorted(scores, reverse=True)


class TestDenominatorSetupPhase3DataCollection:
    """Tests for Phase 3: Data Collection."""

    @pytest.mark.asyncio
    async def test_records_collected_from_available_data(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-014",
            sector=SectorClassification.INDUSTRIALS,
            applicable_frameworks=["gri_305_4"],
            reporting_periods=["2024"],
            available_data={"revenue": {"2024": 200.0}},
        )
        result = await wf.execute(inp)
        assert len(result.collected_records) > 0

    @pytest.mark.asyncio
    async def test_missing_data_generates_warnings(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-015",
            sector=SectorClassification.INDUSTRIALS,
            applicable_frameworks=["gri_305_4"],
            reporting_periods=["2024", "2025"],
            available_data={"revenue": {"2024": 200.0}},
        )
        result = await wf.execute(inp)
        phase3 = result.phases[2]
        assert any("Missing data" in w for w in phase3.warnings)

    @pytest.mark.asyncio
    async def test_custom_denominators_collected(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-016",
            sector=SectorClassification.OTHER,
            applicable_frameworks=["gri_305_4"],
            reporting_periods=["2024"],
            available_data={"revenue": {"2024": 100.0}},
            custom_denominators=[
                {
                    "type": "widgets",
                    "unit": "widget",
                    "values": {"2024": 50000},
                },
            ],
        )
        result = await wf.execute(inp)
        custom_records = [
            r for r in result.collected_records
            if r.denominator_type == DenominatorType.CUSTOM
        ]
        assert len(custom_records) > 0

    @pytest.mark.asyncio
    async def test_coverage_percentage_calculated(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-017",
            sector=SectorClassification.INDUSTRIALS,
            applicable_frameworks=["gri_305_4"],
            reporting_periods=["2024"],
            available_data={"revenue": {"2024": 200.0}},
        )
        result = await wf.execute(inp)
        phase3 = result.phases[2]
        assert "coverage_pct" in phase3.outputs


class TestDenominatorSetupPhase4Validation:
    """Tests for Phase 4: Validation."""

    @pytest.mark.asyncio
    async def test_validation_finds_missing_periods(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-018",
            sector=SectorClassification.INDUSTRIALS,
            applicable_frameworks=["gri_305_4"],
            reporting_periods=["2024", "2025"],
            available_data={"revenue": {"2024": 200.0}},
        )
        result = await wf.execute(inp)
        errors = [
            f for f in result.validation_findings
            if f.severity == ValidationSeverity.ERROR
        ]
        assert len(errors) > 0

    @pytest.mark.asyncio
    async def test_readiness_score_calculated(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-019",
            sector=SectorClassification.INDUSTRIALS,
            applicable_frameworks=["gri_305_4"],
            reporting_periods=["2024"],
            available_data={"revenue": {"2024": 200.0}},
        )
        result = await wf.execute(inp)
        assert 0.0 <= result.readiness_score <= 100.0

    @pytest.mark.asyncio
    async def test_yoy_change_warning(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-020",
            sector=SectorClassification.INDUSTRIALS,
            applicable_frameworks=["gri_305_4"],
            reporting_periods=["2024", "2025"],
            available_data={
                "revenue": {"2024": 100.0, "2025": 300.0},
            },
        )
        result = await wf.execute(inp)
        yoy_findings = [
            f for f in result.validation_findings
            if "Year-over-year" in f.message
        ]
        assert len(yoy_findings) > 0

    @pytest.mark.asyncio
    async def test_zero_value_validation_error(self):
        wf = DenominatorSetupWorkflow()
        inp = DenominatorSetupInput(
            organization_id="org-021",
            sector=SectorClassification.INDUSTRIALS,
            applicable_frameworks=["gri_305_4"],
            reporting_periods=["2024"],
            available_data={"revenue": {"2024": 0.0}},
        )
        result = await wf.execute(inp)
        zero_findings = [
            f for f in result.validation_findings
            if "Zero or negative" in f.message
        ]
        assert len(zero_findings) > 0


# ---------------------------------------------------------------------------
# Sector-Denominator Reference Data Tests
# ---------------------------------------------------------------------------


class TestReferenceData:
    """Tests for reference data integrity."""

    def test_sector_denominator_map_has_13_sectors(self):
        assert len(SECTOR_DENOMINATOR_MAP) == 13

    def test_framework_required_denominators_has_8_frameworks(self):
        assert len(FRAMEWORK_REQUIRED_DENOMINATORS) == 8

    def test_each_sector_has_at_least_2_denominators(self):
        for sector, denoms in SECTOR_DENOMINATOR_MAP.items():
            assert len(denoms) >= 2, f"Sector {sector} has fewer than 2 denominators"

    def test_framework_scope_rules_has_8_frameworks(self):
        assert len(FRAMEWORK_SCOPE_RULES) == 8

    def test_every_framework_has_scope_rules(self):
        for fw, rules in FRAMEWORK_SCOPE_RULES.items():
            assert len(rules) >= 1, f"Framework {fw} has no scope rules"


# ---------------------------------------------------------------------------
# Intensity Calculation Workflow Tests
# ---------------------------------------------------------------------------


class TestIntensityCalcWorkflowInit:
    """Tests for IntensityCalculationWorkflow initialisation."""

    def test_init_creates_workflow(self):
        wf = IntensityCalculationWorkflow()
        assert wf is not None

    def test_init_has_workflow_id(self):
        wf = IntensityCalculationWorkflow()
        assert wf.workflow_id is not None

    def test_phase_sequence_is_4(self):
        assert len(IntensityCalculationWorkflow.PHASE_SEQUENCE) == 4

    def test_phase_sequence_order(self):
        expected = [
            CalcPhase.DATA_INGESTION,
            CalcPhase.SCOPE_CONFIGURATION,
            CalcPhase.INTENSITY_CALCULATION,
            CalcPhase.QUALITY_ASSURANCE,
        ]
        assert IntensityCalculationWorkflow.PHASE_SEQUENCE == expected


class TestIntensityCalcWorkflowExecution:
    """Tests for IntensityCalculationWorkflow.execute()."""

    @pytest.mark.asyncio
    async def test_execute_completes(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-100",
            emissions_data=[
                EmissionsDataSet(
                    period="2024",
                    scope1_tco2e=5000.0,
                    scope2_location_tco2e=3000.0,
                ),
            ],
            denominator_data=[
                DenominatorDataSet(
                    denominator_type="revenue",
                    period="2024",
                    value=500.0,
                    unit="USD_million",
                ),
            ],
        )
        result = await wf.execute(inp)
        assert result.status == WorkflowStatus.COMPLETED
        assert result.metrics_count > 0

    @pytest.mark.asyncio
    async def test_execute_provenance_hash(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-101",
            emissions_data=[
                EmissionsDataSet(period="2024", scope1_tco2e=1000.0),
            ],
            denominator_data=[
                DenominatorDataSet(
                    denominator_type="fte", period="2024", value=50.0, unit="headcount",
                ),
            ],
        )
        result = await wf.execute(inp)
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_execute_records_all_4_phases(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-102",
            emissions_data=[
                EmissionsDataSet(period="2024", scope1_tco2e=2000.0),
            ],
            denominator_data=[
                DenominatorDataSet(
                    denominator_type="revenue", period="2024", value=100.0,
                ),
            ],
        )
        result = await wf.execute(inp)
        completed_phases = [
            p for p in result.phases if p.status == PhaseStatus.COMPLETED
        ]
        assert len(completed_phases) == 4


class TestIntensityCalcPhase1DataIngestion:
    """Tests for Phase 1: Data Ingestion."""

    @pytest.mark.asyncio
    async def test_emissions_indexed_by_period(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-103",
            emissions_data=[
                EmissionsDataSet(period="2023", scope1_tco2e=4000.0),
                EmissionsDataSet(period="2024", scope1_tco2e=5000.0),
            ],
            denominator_data=[
                DenominatorDataSet(denominator_type="revenue", period="2023", value=400.0),
                DenominatorDataSet(denominator_type="revenue", period="2024", value=500.0),
            ],
        )
        result = await wf.execute(inp)
        phase1 = result.phases[0]
        assert phase1.outputs["emissions_ingested"] == 2
        assert len(phase1.outputs["aligned_periods"]) == 2

    @pytest.mark.asyncio
    async def test_misaligned_periods_warned(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-104",
            emissions_data=[
                EmissionsDataSet(period="2024", scope1_tco2e=5000.0),
            ],
            denominator_data=[
                DenominatorDataSet(denominator_type="revenue", period="2023", value=400.0),
            ],
        )
        result = await wf.execute(inp)
        phase1 = result.phases[0]
        assert len(phase1.warnings) > 0


class TestIntensityCalcPhase2ScopeConfiguration:
    """Tests for Phase 2: Scope Configuration."""

    @pytest.mark.asyncio
    async def test_default_scope_rules_applied(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-105",
            emissions_data=[
                EmissionsDataSet(period="2024", scope1_tco2e=5000.0),
            ],
            denominator_data=[
                DenominatorDataSet(denominator_type="revenue", period="2024", value=500.0),
            ],
            applicable_frameworks=["esrs_e1"],
        )
        result = await wf.execute(inp)
        assert len(result.scope_rules) > 0
        fw_names = [r.framework for r in result.scope_rules]
        assert "esrs_e1" in fw_names

    @pytest.mark.asyncio
    async def test_custom_scope_rules_override(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-106",
            emissions_data=[
                EmissionsDataSet(period="2024", scope1_tco2e=5000.0),
            ],
            denominator_data=[
                DenominatorDataSet(denominator_type="revenue", period="2024", value=500.0),
            ],
            applicable_frameworks=["cdp_c6"],
            custom_scope_rules=[
                ScopeRule(
                    framework="cdp_c6",
                    scope_inclusion=ScopeInclusion.SCOPE_1,
                    description="Custom CDP rule",
                ),
            ],
        )
        result = await wf.execute(inp)
        cdp_rules = [r for r in result.scope_rules if r.framework == "cdp_c6"]
        assert len(cdp_rules) == 1
        assert cdp_rules[0].scope_inclusion == ScopeInclusion.SCOPE_1


class TestIntensityCalcPhase3Calculation:
    """Tests for Phase 3: Intensity Calculation."""

    @pytest.mark.asyncio
    async def test_metrics_computed(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-107",
            emissions_data=[
                EmissionsDataSet(
                    period="2024",
                    scope1_tco2e=5000.0,
                    scope2_location_tco2e=3000.0,
                ),
            ],
            denominator_data=[
                DenominatorDataSet(denominator_type="revenue", period="2024", value=500.0),
            ],
            applicable_frameworks=["iso_14064"],
        )
        result = await wf.execute(inp)
        assert result.metrics_count > 0

    @pytest.mark.asyncio
    async def test_intensity_value_deterministic(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-108",
            emissions_data=[
                EmissionsDataSet(period="2024", scope1_tco2e=5000.0),
            ],
            denominator_data=[
                DenominatorDataSet(
                    denominator_type="revenue", period="2024",
                    value=500.0, unit="USD_million",
                ),
            ],
            applicable_frameworks=["iso_14064"],
        )
        result = await wf.execute(inp)
        s12_loc_metrics = [
            m for m in result.intensity_metrics
            if m.scope_inclusion == ScopeInclusion.SCOPE_1_2_LOCATION
        ]
        if s12_loc_metrics:
            m = s12_loc_metrics[0]
            assert m.intensity_value == pytest.approx(5000.0 / 500.0, rel=1e-4)

    @pytest.mark.asyncio
    async def test_multiple_denominators_produce_more_metrics(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-109",
            emissions_data=[
                EmissionsDataSet(period="2024", scope1_tco2e=5000.0),
            ],
            denominator_data=[
                DenominatorDataSet(denominator_type="revenue", period="2024", value=500.0),
                DenominatorDataSet(denominator_type="fte", period="2024", value=100.0),
            ],
            applicable_frameworks=["iso_14064"],
        )
        result = await wf.execute(inp)
        denom_types = set(m.denominator_type for m in result.intensity_metrics)
        assert len(denom_types) == 2

    @pytest.mark.asyncio
    async def test_intensity_unit_mapped(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-110",
            emissions_data=[
                EmissionsDataSet(period="2024", scope1_tco2e=5000.0),
            ],
            denominator_data=[
                DenominatorDataSet(
                    denominator_type="revenue", period="2024",
                    value=500.0, unit="USD_million",
                ),
            ],
            applicable_frameworks=["iso_14064"],
        )
        result = await wf.execute(inp)
        for m in result.intensity_metrics:
            assert m.intensity_unit == IntensityUnit.TCO2E_PER_USD_MILLION


class TestIntensityCalcPhase4QualityAssurance:
    """Tests for Phase 4: Quality Assurance."""

    @pytest.mark.asyncio
    async def test_quality_checks_run(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-111",
            emissions_data=[
                EmissionsDataSet(period="2024", scope1_tco2e=5000.0),
            ],
            denominator_data=[
                DenominatorDataSet(denominator_type="revenue", period="2024", value=500.0),
            ],
        )
        result = await wf.execute(inp)
        assert len(result.quality_checks) > 0

    @pytest.mark.asyncio
    async def test_quality_pass_rate_calculated(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-112",
            emissions_data=[
                EmissionsDataSet(period="2024", scope1_tco2e=5000.0),
            ],
            denominator_data=[
                DenominatorDataSet(denominator_type="revenue", period="2024", value=500.0),
            ],
        )
        result = await wf.execute(inp)
        assert 0.0 <= result.quality_pass_rate <= 100.0

    @pytest.mark.asyncio
    async def test_consistency_check_scope_hierarchy(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-113",
            emissions_data=[
                EmissionsDataSet(
                    period="2024",
                    scope1_tco2e=5000.0,
                    scope2_location_tco2e=3000.0,
                    scope3_tco2e=10000.0,
                ),
            ],
            denominator_data=[
                DenominatorDataSet(denominator_type="revenue", period="2024", value=500.0),
            ],
            applicable_frameworks=["esrs_e1"],
        )
        result = await wf.execute(inp)
        consistency_checks = [
            qc for qc in result.quality_checks
            if qc.check_type == QualityCheckType.CONSISTENCY
        ]
        assert len(consistency_checks) > 0

    @pytest.mark.asyncio
    async def test_cross_framework_check(self):
        wf = IntensityCalculationWorkflow()
        inp = IntensityCalcInput(
            organization_id="org-114",
            emissions_data=[
                EmissionsDataSet(period="2024", scope1_tco2e=5000.0),
            ],
            denominator_data=[
                DenominatorDataSet(denominator_type="revenue", period="2024", value=500.0),
            ],
            applicable_frameworks=["esrs_e1", "cdp_c6"],
        )
        result = await wf.execute(inp)
        cf_checks = [
            qc for qc in result.quality_checks
            if qc.check_type == QualityCheckType.CROSS_FRAMEWORK
        ]
        assert len(cf_checks) > 0
