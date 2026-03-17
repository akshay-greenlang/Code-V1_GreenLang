# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep Pack - Workflow Tests
========================================================

Tests all 8 workflows: CarbonFootprint, RecycledContent, PassportCompilation,
PerformanceTesting, DueDiligenceAssessment, LabellingVerification,
EndOfLifePlanning, RegulatorySubmission. Validates class instantiation,
phase execution, error handling, provenance tracking, and result schemas.

Author: GreenLang Platform Team (GL-TestEngineer)
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = PACK_ROOT / "workflows"


def _load_module(file_name: str, module_name: str, subdir: str = ""):
    if subdir:
        file_path = PACK_ROOT / subdir / file_name
    else:
        file_path = PACK_ROOT / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Workflow modules loaded dynamically
# ---------------------------------------------------------------------------

cf_mod = _load_module(
    "carbon_footprint_assessment_workflow.py",
    "pack020_tw.wf_cf",
    "workflows",
)
rc_mod = _load_module(
    "recycled_content_tracking_workflow.py",
    "pack020_tw.wf_rc",
    "workflows",
)
pc_mod = _load_module(
    "passport_compilation_workflow.py",
    "pack020_tw.wf_pc",
    "workflows",
)
pt_mod = _load_module(
    "performance_testing_workflow.py",
    "pack020_tw.wf_pt",
    "workflows",
)
dd_mod = _load_module(
    "due_diligence_assessment_workflow.py",
    "pack020_tw.wf_dd",
    "workflows",
)
lv_mod = _load_module(
    "labelling_verification_workflow.py",
    "pack020_tw.wf_lv",
    "workflows",
)
eol_mod = _load_module(
    "end_of_life_planning_workflow.py",
    "pack020_tw.wf_eol",
    "workflows",
)
rs_mod = _load_module(
    "regulatory_submission_workflow.py",
    "pack020_tw.wf_rs",
    "workflows",
)


# =========================================================================
# CarbonFootprintWorkflow
# =========================================================================
class TestCarbonFootprintWorkflow:
    """Tests for CarbonFootprintWorkflow."""

    def test_instantiation(self):
        wf = cf_mod.CarbonFootprintWorkflow()
        assert wf.workflow_id is not None
        assert wf.config is None

    def test_instantiation_with_config(self):
        wf = cf_mod.CarbonFootprintWorkflow(config={"key": "val"})
        assert wf.config == {"key": "val"}

    def test_get_phases_returns_four(self):
        wf = cf_mod.CarbonFootprintWorkflow()
        phases = wf.get_phases()
        assert len(phases) == 4
        names = [p["name"] for p in phases]
        assert "data_collection" in names
        assert "lca_calculation" in names

    def test_validate_inputs_valid(self):
        wf = cf_mod.CarbonFootprintWorkflow()
        inp = cf_mod.CarbonFootprintInput(
            battery_capacity_kwh=75.0,
            material_records=[
                cf_mod.MaterialRecord(material_name="lithium_carbonate", mass_kg=10.0)
            ],
        )
        issues = wf.validate_inputs(inp)
        assert issues == []

    def test_validate_inputs_zero_capacity(self):
        wf = cf_mod.CarbonFootprintWorkflow()
        inp = cf_mod.CarbonFootprintInput(battery_capacity_kwh=0.0)
        issues = wf.validate_inputs(inp)
        assert any("capacity" in i.lower() for i in issues)

    def test_validate_inputs_no_records(self):
        wf = cf_mod.CarbonFootprintWorkflow()
        inp = cf_mod.CarbonFootprintInput(battery_capacity_kwh=75.0)
        issues = wf.validate_inputs(inp)
        assert len(issues) > 0

    @pytest.mark.asyncio
    async def test_execute_completes(self):
        wf = cf_mod.CarbonFootprintWorkflow()
        inp = cf_mod.CarbonFootprintInput(
            battery_capacity_kwh=75.0,
            material_records=[
                cf_mod.MaterialRecord(
                    material_name="lithium_carbonate",
                    mass_kg=10.0,
                    emission_factor_kgco2e_per_kg=8.5,
                )
            ],
            energy_records=[
                cf_mod.EnergyRecord(
                    energy_source="grid_eu_avg",
                    consumption_kwh=500.0,
                    emission_factor_kgco2e_per_kwh=0.256,
                )
            ],
        )
        result = await wf.execute(inp)
        assert result.status.value == "completed"
        assert result.phases_completed == 4
        assert result.total_carbon_footprint_kgco2e > 0

    @pytest.mark.asyncio
    async def test_execute_provenance_hash(self):
        wf = cf_mod.CarbonFootprintWorkflow()
        inp = cf_mod.CarbonFootprintInput(
            battery_capacity_kwh=75.0,
            material_records=[
                cf_mod.MaterialRecord(
                    material_name="copper_foil",
                    mass_kg=5.0,
                    emission_factor_kgco2e_per_kg=4.1,
                )
            ],
        )
        result = await wf.execute(inp)
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_execute_performance_class_a(self):
        wf = cf_mod.CarbonFootprintWorkflow()
        inp = cf_mod.CarbonFootprintInput(
            battery_capacity_kwh=100.0,
            material_records=[
                cf_mod.MaterialRecord(
                    material_name="lithium_carbonate",
                    mass_kg=1.0,
                    emission_factor_kgco2e_per_kg=8.5,
                )
            ],
        )
        result = await wf.execute(inp)
        assert result.performance_class == "A"

    @pytest.mark.asyncio
    async def test_execute_default_emission_factors(self):
        wf = cf_mod.CarbonFootprintWorkflow()
        inp = cf_mod.CarbonFootprintInput(
            battery_capacity_kwh=75.0,
            material_records=[
                cf_mod.MaterialRecord(material_name="lithium_carbonate", mass_kg=10.0)
            ],
        )
        result = await wf.execute(inp)
        assert result.status.value == "completed"
        assert result.material_emissions_kgco2e > 0

    @pytest.mark.asyncio
    async def test_execute_with_recycling_credit(self):
        wf = cf_mod.CarbonFootprintWorkflow()
        inp = cf_mod.CarbonFootprintInput(
            battery_capacity_kwh=75.0,
            recycling_credit_kgco2e=50.0,
            material_records=[
                cf_mod.MaterialRecord(
                    material_name="lithium_carbonate",
                    mass_kg=10.0,
                    emission_factor_kgco2e_per_kg=8.5,
                )
            ],
        )
        result = await wf.execute(inp)
        assert result.recycling_credit_kgco2e == 50.0

    @pytest.mark.asyncio
    async def test_execute_none_input(self):
        wf = cf_mod.CarbonFootprintWorkflow()
        result = await wf.execute(None)
        assert result.status.value in ("completed", "failed")


# =========================================================================
# RecycledContentWorkflow
# =========================================================================
class TestRecycledContentWorkflow:
    """Tests for RecycledContentWorkflow."""

    def test_instantiation(self):
        wf = rc_mod.RecycledContentWorkflow()
        assert wf.workflow_id is not None

    def test_get_phases_returns_four(self):
        wf = rc_mod.RecycledContentWorkflow()
        assert len(wf.get_phases()) == 4

    def test_validate_inputs_empty(self):
        wf = rc_mod.RecycledContentWorkflow()
        inp = rc_mod.RecycledContentInput()
        issues = wf.validate_inputs(inp)
        assert any("no material" in i.lower() for i in issues)

    @pytest.mark.asyncio
    async def test_execute_with_entries(self):
        wf = rc_mod.RecycledContentWorkflow()
        inp = rc_mod.RecycledContentInput(
            material_entries=[
                rc_mod.MaterialEntry(
                    material_type=rc_mod.RecoveredMaterialType.COBALT,
                    total_mass_kg=10.0,
                    recycled_mass_kg=2.0,
                ),
                rc_mod.MaterialEntry(
                    material_type=rc_mod.RecoveredMaterialType.LITHIUM,
                    total_mass_kg=8.0,
                    recycled_mass_kg=0.5,
                ),
            ],
        )
        result = await wf.execute(inp)
        assert result.status.value == "completed"
        assert result.overall_recycled_content_pct > 0
        assert len(result.material_summaries) == 2

    @pytest.mark.asyncio
    async def test_target_comparison_phase_2031(self):
        wf = rc_mod.RecycledContentWorkflow()
        inp = rc_mod.RecycledContentInput(
            material_entries=[
                rc_mod.MaterialEntry(
                    material_type=rc_mod.RecoveredMaterialType.COBALT,
                    total_mass_kg=100.0,
                    recycled_mass_kg=20.0,
                ),
            ],
            target_phase=rc_mod.ComplianceTargetPhase.PHASE_1_2031,
        )
        result = await wf.execute(inp)
        cobalt_summary = next(
            (s for s in result.material_summaries if s.material_type == "cobalt"), None
        )
        assert cobalt_summary is not None
        assert cobalt_summary.meets_2031_target is True

    @pytest.mark.asyncio
    async def test_provenance_hash_present(self):
        wf = rc_mod.RecycledContentWorkflow()
        inp = rc_mod.RecycledContentInput(
            material_entries=[
                rc_mod.MaterialEntry(
                    material_type=rc_mod.RecoveredMaterialType.NICKEL,
                    total_mass_kg=50.0,
                    recycled_mass_kg=5.0,
                ),
            ],
        )
        result = await wf.execute(inp)
        assert len(result.provenance_hash) == 64


# =========================================================================
# PassportCompilationWorkflow
# =========================================================================
class TestPassportCompilationWorkflow:
    """Tests for PassportCompilationWorkflow."""

    def test_instantiation(self):
        wf = pc_mod.PassportCompilationWorkflow()
        assert wf.workflow_id is not None

    def test_get_phases_returns_five(self):
        wf = pc_mod.PassportCompilationWorkflow()
        assert len(wf.get_phases()) == 5

    def test_validate_inputs_missing_battery_id(self):
        wf = pc_mod.PassportCompilationWorkflow()
        inp = pc_mod.PassportCompilationInput(battery_id="", manufacturer_name="Foo")
        issues = wf.validate_inputs(inp)
        assert any("battery" in i.lower() for i in issues)

    @pytest.mark.asyncio
    async def test_execute_full(self):
        wf = pc_mod.PassportCompilationWorkflow()
        inp = pc_mod.PassportCompilationInput(
            battery_id="BAT-TEST-001",
            manufacturer_name="TestCo",
            manufacturing_place="Berlin",
            manufacturing_date="2027-01-01",
            battery_weight_kg=450.0,
            carbon_footprint_data={
                "carbon_footprint_total_kgco2e": 5000,
                "carbon_footprint_per_kwh": 66.0,
                "performance_class": "B",
                "lifecycle_stages": ["raw_material", "manufacturing"],
            },
        )
        result = await wf.execute(inp)
        assert result.status.value == "completed"
        assert result.passport_id.startswith("BP-")
        assert result.qr_code is not None

    @pytest.mark.asyncio
    async def test_execute_incomplete_validation(self):
        wf = pc_mod.PassportCompilationWorkflow()
        inp = pc_mod.PassportCompilationInput(
            battery_id="BAT-TEST-002",
            manufacturer_name="",
        )
        result = await wf.execute(inp)
        assert result.validation_passed is False
        assert result.validation_error_count > 0


# =========================================================================
# PerformanceTestingWorkflow
# =========================================================================
class TestPerformanceTestingWorkflow:
    """Tests for PerformanceTestingWorkflow."""

    def test_instantiation(self):
        wf = pt_mod.PerformanceTestingWorkflow()
        assert wf.workflow_id is not None

    def test_get_phases_returns_four(self):
        assert len(pt_mod.PerformanceTestingWorkflow().get_phases()) == 4

    @pytest.mark.asyncio
    async def test_execute_with_test_records(self):
        wf = pt_mod.PerformanceTestingWorkflow()
        inp = pt_mod.PerformanceTestingInput(
            rated_capacity_ah=75.0,
            battery_category="ev_battery",
            test_records=[
                pt_mod.TestRecord(
                    test_type=pt_mod.TestType.RATED_CAPACITY,
                    measured_value=75.0,
                    unit="Ah",
                ),
                pt_mod.TestRecord(
                    test_type=pt_mod.TestType.ROUND_TRIP_EFFICIENCY,
                    measured_value=93.0,
                    unit="%",
                ),
                pt_mod.TestRecord(
                    test_type=pt_mod.TestType.CYCLE_LIFE,
                    measured_value=1500.0,
                    unit="cycles",
                ),
                pt_mod.TestRecord(
                    test_type=pt_mod.TestType.CAPACITY_RETENTION,
                    measured_value=85.0,
                    unit="%",
                    cycle_count=500,
                ),
            ],
        )
        result = await wf.execute(inp)
        assert result.status.value == "completed"
        assert result.metrics_compliant > 0

    @pytest.mark.asyncio
    async def test_overall_compliance_non_compliant(self):
        wf = pt_mod.PerformanceTestingWorkflow()
        inp = pt_mod.PerformanceTestingInput(
            rated_capacity_ah=75.0,
            battery_category="ev_battery",
            test_records=[
                pt_mod.TestRecord(
                    test_type=pt_mod.TestType.ROUND_TRIP_EFFICIENCY,
                    measured_value=70.0,
                    unit="%",
                ),
            ],
        )
        result = await wf.execute(inp)
        assert result.overall_compliance == "non_compliant"


# =========================================================================
# DueDiligenceAssessmentWorkflow
# =========================================================================
class TestDueDiligenceAssessmentWorkflow:
    """Tests for DueDiligenceAssessmentWorkflow."""

    def test_instantiation(self):
        wf = dd_mod.DueDiligenceAssessmentWorkflow()
        assert wf.workflow_id is not None

    def test_get_phases(self):
        assert len(dd_mod.DueDiligenceAssessmentWorkflow().get_phases()) == 4

    @pytest.mark.asyncio
    async def test_execute_with_suppliers(self):
        wf = dd_mod.DueDiligenceAssessmentWorkflow()
        inp = dd_mod.DueDiligenceInput(
            suppliers=[
                dd_mod.SupplierRecord(
                    supplier_name="CobaltMine Corp",
                    country_code="COD",
                    tier=dd_mod.SupplierTier.TIER_3,
                    materials_supplied=["cobalt"],
                ),
                dd_mod.SupplierRecord(
                    supplier_name="NickelPure Finland",
                    country_code="FI",
                    tier=dd_mod.SupplierTier.TIER_1,
                    materials_supplied=["nickel"],
                    has_due_diligence_policy=True,
                    certification="ISO 14001",
                ),
            ],
        )
        result = await wf.execute(inp)
        assert result.status.value == "completed"
        assert result.suppliers_mapped == 2
        assert result.high_risk_supplier_count >= 1

    @pytest.mark.asyncio
    async def test_provenance_hash(self):
        wf = dd_mod.DueDiligenceAssessmentWorkflow()
        inp = dd_mod.DueDiligenceInput(
            suppliers=[
                dd_mod.SupplierRecord(
                    supplier_name="Test Supplier",
                    country_code="DE",
                    tier=dd_mod.SupplierTier.TIER_1,
                ),
            ],
        )
        result = await wf.execute(inp)
        assert len(result.provenance_hash) == 64


# =========================================================================
# LabellingVerificationWorkflow
# =========================================================================
class TestLabellingVerificationWorkflow:
    """Tests for LabellingVerificationWorkflow."""

    def test_instantiation(self):
        wf = lv_mod.LabellingVerificationWorkflow()
        assert wf.workflow_id is not None

    def test_get_phases(self):
        assert len(lv_mod.LabellingVerificationWorkflow().get_phases()) == 4

    @pytest.mark.asyncio
    async def test_execute_conformant(self):
        wf = lv_mod.LabellingVerificationWorkflow()
        required = lv_mod.LABEL_REQUIREMENTS["ev_battery"]
        elements = [
            lv_mod.LabelElement(
                element_type=et,
                present=True,
                legible=True,
                indelible=True,
                height_mm=20.0,
                width_mm=20.0,
                content="Valid content" if et not in (
                    "qr_code", "ce_marking", "hazard_symbols",
                    "crossed_out_wheelie_bin", "separate_collection",
                    "carbon_footprint_class", "recycled_content_info",
                    "material_composition",
                ) else "",
            )
            for et in required
        ]
        inp = lv_mod.LabellingVerificationInput(
            battery_category="ev_battery",
            label_elements=elements,
        )
        result = await wf.execute(inp)
        assert result.status.value == "completed"

    @pytest.mark.asyncio
    async def test_execute_missing_elements(self):
        wf = lv_mod.LabellingVerificationWorkflow()
        inp = lv_mod.LabellingVerificationInput(
            battery_category="ev_battery",
            label_elements=[],
        )
        result = await wf.execute(inp)
        assert result.elements_non_conformant > 0
        assert result.overall_conformant is False


# =========================================================================
# EndOfLifePlanningWorkflow
# =========================================================================
class TestEndOfLifePlanningWorkflow:
    """Tests for EndOfLifePlanningWorkflow."""

    def test_instantiation(self):
        wf = eol_mod.EndOfLifePlanningWorkflow()
        assert wf.workflow_id is not None

    def test_get_phases(self):
        assert len(eol_mod.EndOfLifePlanningWorkflow().get_phases()) == 4

    @pytest.mark.asyncio
    async def test_execute_with_data(self):
        wf = eol_mod.EndOfLifePlanningWorkflow()
        inp = eol_mod.EndOfLifeInput(
            battery_category="ev",
            battery_chemistry="lithium_ion",
            target_year="2028",
            waste_streams=[
                eol_mod.WasteStreamRecord(
                    batteries_placed_on_market_tonnes=100.0,
                    batteries_collected_tonnes=100.0,
                    batteries_recycled_tonnes=72.0,
                )
            ],
            material_recoveries=[
                eol_mod.MaterialRecoveryRecord(
                    material_name="cobalt",
                    input_mass_kg=1000.0,
                    recovered_mass_kg=920.0,
                ),
                eol_mod.MaterialRecoveryRecord(
                    material_name="lithium",
                    input_mass_kg=500.0,
                    recovered_mass_kg=260.0,
                ),
            ],
        )
        result = await wf.execute(inp)
        assert result.status.value == "completed"
        assert result.recycling_summary is not None


# =========================================================================
# RegulatorySubmissionWorkflow
# =========================================================================
class TestRegulatorySubmissionWorkflow:
    """Tests for RegulatorySubmissionWorkflow."""

    def test_instantiation(self):
        wf = rs_mod.RegulatorySubmissionWorkflow()
        assert wf.workflow_id is not None

    def test_get_phases(self):
        assert len(rs_mod.RegulatorySubmissionWorkflow().get_phases()) == 4

    @pytest.mark.asyncio
    async def test_execute_empty_documents(self):
        wf = rs_mod.RegulatorySubmissionWorkflow()
        inp = rs_mod.RegulatorySubmissionInput(
            battery_id="BAT-001",
            manufacturer_name="Test Corp",
        )
        result = await wf.execute(inp)
        assert result.status.value == "completed"
        assert result.submission_ready is False

    @pytest.mark.asyncio
    async def test_execute_with_documents(self):
        wf = rs_mod.RegulatorySubmissionWorkflow()
        docs = []
        for doc_type in rs_mod.REQUIRED_DOCUMENTS.get("ev_battery", []):
            docs.append(
                rs_mod.DocumentRecord(
                    document_type=doc_type,
                    title=f"Test {doc_type}",
                    fields={
                        k: f"value_{k}"
                        for k in rs_mod.DOC_REQUIRED_FIELDS.get(doc_type, [])
                    },
                )
            )
        inp = rs_mod.RegulatorySubmissionInput(
            battery_id="BAT-FULL",
            manufacturer_name="Complete Corp",
            battery_category="ev_battery",
            documents=docs,
        )
        result = await wf.execute(inp)
        assert result.status.value == "completed"
        assert result.overall_completeness_pct == 100.0
        assert result.submission_ready is True

    @pytest.mark.asyncio
    async def test_provenance_hash(self):
        wf = rs_mod.RegulatorySubmissionWorkflow()
        result = await wf.execute(
            rs_mod.RegulatorySubmissionInput(
                battery_id="BAT-P",
                manufacturer_name="Prov Corp",
            )
        )
        assert len(result.provenance_hash) == 64
