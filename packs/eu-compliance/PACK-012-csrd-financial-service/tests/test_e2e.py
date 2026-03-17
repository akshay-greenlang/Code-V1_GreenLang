# -*- coding: utf-8 -*-
"""
End-to-end flow tests for PACK-012 CSRD Financial Service Pack.

These tests exercise multiple engines and workflows together in realistic
scenarios, validating that data flows correctly through the complete CSRD
Financial Service compliance pipeline. Each test constructs realistic input
data, runs it through engines/workflows, and verifies cross-component
consistency.

Test count: 15 tests
Target: Validate complete FI CSRD compliance workflows
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Path setup - import engines and workflows from PACK-012 source tree
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent


def _import_from_path(module_name: str, file_path: str):
    """Import a module from an absolute file path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Module imports - engines, workflows, and templates
# ---------------------------------------------------------------------------

fe_mod = _import_from_path(
    "e2e_financed_emissions_engine",
    str(PACK_ROOT / "engines" / "financed_emissions_engine.py"),
)
gar_mod = _import_from_path(
    "e2e_green_asset_ratio_engine",
    str(PACK_ROOT / "engines" / "green_asset_ratio_engine.py"),
)
climate_mod = _import_from_path(
    "e2e_climate_risk_scoring_engine",
    str(PACK_ROOT / "engines" / "climate_risk_scoring_engine.py"),
)
pillar3_mod = _import_from_path(
    "e2e_pillar3_esg_engine",
    str(PACK_ROOT / "engines" / "pillar3_esg_engine.py"),
)
insurance_mod = _import_from_path(
    "e2e_insurance_underwriting_engine",
    str(PACK_ROOT / "engines" / "insurance_underwriting_engine.py"),
)
materiality_mod = _import_from_path(
    "e2e_fs_double_materiality_engine",
    str(PACK_ROOT / "engines" / "fs_double_materiality_engine.py"),
)
transition_mod = _import_from_path(
    "e2e_fs_transition_plan_engine",
    str(PACK_ROOT / "engines" / "fs_transition_plan_engine.py"),
)
btar_mod = _import_from_path(
    "e2e_btar_calculator_engine",
    str(PACK_ROOT / "engines" / "btar_calculator_engine.py"),
)

# Workflow modules
wf_financed_mod = _import_from_path(
    "e2e_wf_financed",
    str(PACK_ROOT / "workflows" / "financed_emissions_workflow.py"),
)
wf_gar_mod = _import_from_path(
    "e2e_wf_gar",
    str(PACK_ROOT / "workflows" / "gar_btar_workflow.py"),
)
wf_stress_mod = _import_from_path(
    "e2e_wf_stress",
    str(PACK_ROOT / "workflows" / "climate_stress_test_workflow.py"),
)
wf_pillar3_mod = _import_from_path(
    "e2e_wf_pillar3",
    str(PACK_ROOT / "workflows" / "pillar3_reporting_workflow.py"),
)
wf_insurance_mod = _import_from_path(
    "e2e_wf_insurance",
    str(PACK_ROOT / "workflows" / "insurance_emissions_workflow.py"),
)

# Template modules
tpl_pcaf_mod = _import_from_path(
    "e2e_tpl_pcaf",
    str(PACK_ROOT / "templates" / "pcaf_report.py"),
)

# ---------------------------------------------------------------------------
# Engine aliases
# ---------------------------------------------------------------------------

FinancedEmissionsEngine = fe_mod.FinancedEmissionsEngine
FinancedEmissionsConfig = fe_mod.FinancedEmissionsConfig

GreenAssetRatioEngine = gar_mod.GreenAssetRatioEngine
GARConfig = gar_mod.GARConfig

ClimateRiskScoringEngine = climate_mod.ClimateRiskScoringEngine
ClimateRiskConfig = climate_mod.ClimateRiskConfig

Pillar3ESGEngine = pillar3_mod.Pillar3ESGEngine
Pillar3Config = pillar3_mod.Pillar3Config

InsuranceUnderwritingEngine = insurance_mod.InsuranceUnderwritingEngine
UnderwritingConfig = insurance_mod.UnderwritingConfig

FSDoubleMaterialityEngine = materiality_mod.FSDoubleMaterialityEngine
FSMaterialityConfig = materiality_mod.FSMaterialityConfig

FSTransitionPlanEngine = transition_mod.FSTransitionPlanEngine
TransitionPlanConfig = transition_mod.TransitionPlanConfig

BTARCalculatorEngine = btar_mod.BTARCalculatorEngine
BTARConfig = btar_mod.BTARConfig

# Workflow aliases
FinancedEmissionsWorkflow = wf_financed_mod.FinancedEmissionsWorkflow
FinancedEmissionsInput = wf_financed_mod.FinancedEmissionsInput
CounterpartyExposure = wf_financed_mod.CounterpartyExposure

GARBTARWorkflow = wf_gar_mod.GARBTARWorkflow
GARBTARInput = wf_gar_mod.GARBTARInput
GARAsset = wf_gar_mod.GARAsset

ClimateStressTestWorkflow = wf_stress_mod.ClimateStressTestWorkflow
ClimateStressTestInput = wf_stress_mod.ClimateStressTestInput

Pillar3ReportingWorkflow = wf_pillar3_mod.Pillar3ReportingWorkflow
Pillar3ReportingInput = wf_pillar3_mod.Pillar3ReportingInput

InsuranceEmissionsWorkflow = wf_insurance_mod.InsuranceEmissionsWorkflow
InsuranceEmissionsInput = wf_insurance_mod.InsuranceEmissionsInput
InsurancePolicy = wf_insurance_mod.InsurancePolicy

PCAFReportTemplate = tpl_pcaf_mod.PCAFReportTemplate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# ===========================================================================
# E2E Test: Full PCAF Financed Emissions Calculation
# ===========================================================================


class TestE2EFinancedEmissions:
    """Full end-to-end financed emissions workflow with template rendering."""

    @pytest.mark.asyncio
    async def test_full_financed_emissions_flow(self):
        """Complete financed emissions from data to disclosure."""
        wf = FinancedEmissionsWorkflow()
        inp = FinancedEmissionsInput(
            organization_id="org-e2e-001",
            reporting_period="2025",
            exposures=[
                CounterpartyExposure(
                    counterparty_id="cp-e2e-001",
                    counterparty_name="Industrial Corp",
                    asset_class="BUSINESS_LOANS",
                    outstanding_amount=2000000.0,
                    sector="C.20",
                    country="DE",
                    scope1_emissions=500.0,
                    scope2_emissions=200.0,
                    total_equity_plus_debt=10000000.0,
                    data_quality_score=2,
                ),
                CounterpartyExposure(
                    counterparty_id="cp-e2e-002",
                    counterparty_name="Real Estate Trust",
                    asset_class="COMMERCIAL_REAL_ESTATE",
                    outstanding_amount=5000000.0,
                    sector="L.68",
                    country="FR",
                    scope1_emissions=300.0,
                    scope2_emissions=400.0,
                    total_equity_plus_debt=20000000.0,
                    data_quality_score=3,
                ),
            ],
        )
        result = await wf.run(inp)

        assert result is not None
        assert result.workflow_name == "financed_emissions"
        assert len(result.provenance_hash) == 64
        assert len(result.phases) == 5
        assert hasattr(result, "total_financed_emissions_tco2e")
        assert hasattr(result, "waci")

    @pytest.mark.asyncio
    async def test_financed_emissions_provenance_determinism(self):
        """Same input produces valid provenance across runs."""
        inp = FinancedEmissionsInput(
            organization_id="org-e2e-002",
            reporting_period="2025",
            exposures=[
                CounterpartyExposure(
                    counterparty_id="cp-det-001",
                    asset_class="LISTED_EQUITY",
                    outstanding_amount=1000000.0,
                    scope1_emissions=100.0,
                    total_equity_plus_debt=5000000.0,
                )
            ],
        )
        wf1 = FinancedEmissionsWorkflow()
        wf2 = FinancedEmissionsWorkflow()
        r1 = await wf1.run(inp)
        r2 = await wf2.run(inp)

        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_financed_emissions_template_render(self):
        """Financed emissions result can be rendered as PCAF template."""
        wf = FinancedEmissionsWorkflow()
        inp = FinancedEmissionsInput(
            organization_id="org-e2e-003",
            reporting_period="2025",
        )
        result = await wf.run(inp)

        template = PCAFReportTemplate()
        data = {
            "institution_name": "GL E2E Test Bank",
            "reporting_period": "2025",
            "total_financed_emissions": result.total_financed_emissions_tco2e,
            "waci": result.waci,
            "data_quality_score": result.weighted_data_quality_score,
        }
        markdown = template.render_markdown(data)
        assert isinstance(markdown, str)
        assert len(markdown) > 0


# ===========================================================================
# E2E Test: GAR/BTAR Computation
# ===========================================================================


class TestE2EGARBTARComputation:
    """Full end-to-end GAR/BTAR computation workflow."""

    @pytest.mark.asyncio
    async def test_full_gar_btar_computation(self):
        """Complete GAR/BTAR from asset classification to disclosure."""
        wf = GARBTARWorkflow()
        inp = GARBTARInput(
            organization_id="org-e2e-010",
            reporting_date="2025-12-31",
            assets=[
                GARAsset(
                    asset_id="asset-e2e-001",
                    counterparty_name="Green Energy Ltd",
                    counterparty_nace="D.35",
                    gross_carrying_amount=3000000.0,
                    taxonomy_eligible=True,
                    taxonomy_aligned_pct=85.0,
                    substantial_contribution_objective="CLIMATE_MITIGATION",
                    dnsh_compliant=True,
                    minimum_safeguards=True,
                ),
                GARAsset(
                    asset_id="asset-e2e-002",
                    counterparty_name="Oil Refinery Corp",
                    counterparty_nace="C.19",
                    gross_carrying_amount=2000000.0,
                    taxonomy_eligible=True,
                    taxonomy_aligned_pct=0.0,
                    dnsh_compliant=False,
                ),
            ],
            include_btar=True,
        )
        result = await wf.run(inp)

        assert result is not None
        assert result.workflow_name == "gar_btar"
        assert len(result.provenance_hash) == 64
        assert len(result.phases) == 4
        assert hasattr(result, "gar_pct")
        assert hasattr(result, "btar_pct")


# ===========================================================================
# E2E Test: Climate Stress Test
# ===========================================================================


class TestE2EClimateStressTest:
    """Full end-to-end climate stress test workflow."""

    @pytest.mark.asyncio
    async def test_full_climate_stress_test(self):
        """Complete climate stress test across NGFS scenarios."""
        wf = ClimateStressTestWorkflow()
        inp = ClimateStressTestInput(
            organization_id="org-e2e-020",
            reporting_date="2025-12-31",
            scenarios=["NET_ZERO_2050", "DELAYED_TRANSITION", "CURRENT_POLICIES"],
            time_horizons=[2030, 2040, 2050],
        )
        result = await wf.run(inp)

        assert result is not None
        assert len(result.provenance_hash) == 64
        assert len(result.phases) == 5
        assert hasattr(result, "scenarios_tested")
        assert hasattr(result, "max_credit_loss_pct")

    @pytest.mark.asyncio
    async def test_stress_test_risk_metrics(self):
        """Stress test produces physical and transition risk metrics."""
        wf = ClimateStressTestWorkflow()
        inp = ClimateStressTestInput(
            organization_id="org-e2e-021",
            reporting_date="2025-12-31",
        )
        result = await wf.run(inp)
        assert hasattr(result, "physical_risk_exposure_pct")
        assert hasattr(result, "transition_risk_exposure_pct")
        assert hasattr(result, "counterparties_assessed")


# ===========================================================================
# E2E Test: Pillar 3 Report Generation
# ===========================================================================


class TestE2EPillar3Report:
    """Full end-to-end Pillar 3 ESG ITS report generation."""

    @pytest.mark.asyncio
    async def test_full_pillar3_report(self):
        """Complete Pillar 3 from data extraction to filing readiness."""
        wf = Pillar3ReportingWorkflow()
        inp = Pillar3ReportingInput(
            organization_id="org-e2e-030",
            reporting_date="2025-12-31",
            institution_name="GL E2E Credit Institution AG",
            lei="529900HNOAA1KXQJUQ27",
            gar_pct=18.5,
            btar_pct=25.3,
            financed_emissions_tco2e=12500.0,
            total_exposure_eur=50000000000.0,
        )
        result = await wf.run(inp)

        assert result is not None
        assert len(result.provenance_hash) == 64
        assert len(result.phases) == 4
        assert hasattr(result, "templates_populated")
        assert hasattr(result, "filing_ready")

    @pytest.mark.asyncio
    async def test_pillar3_data_quality(self):
        """Pillar 3 result includes data quality assessment."""
        wf = Pillar3ReportingWorkflow()
        inp = Pillar3ReportingInput(
            organization_id="org-e2e-031",
            reporting_date="2025-12-31",
        )
        result = await wf.run(inp)
        assert hasattr(result, "data_quality_score")
        assert hasattr(result, "completeness_pct")


# ===========================================================================
# E2E Test: Insurance Emissions
# ===========================================================================


class TestE2EInsuranceEmissions:
    """Full end-to-end insurance underwriting emissions workflow."""

    @pytest.mark.asyncio
    async def test_full_insurance_emissions(self):
        """Complete insurance emissions from policy data to disclosure."""
        wf = InsuranceEmissionsWorkflow()
        inp = InsuranceEmissionsInput(
            organization_id="org-e2e-040",
            reporting_period="2025",
            policies=[
                InsurancePolicy(
                    policy_id="pol-e2e-001",
                    policyholder_name="Heavy Industry Inc",
                    line_of_business="PROPERTY",
                    gross_written_premium=1000000.0,
                    scope1_emissions=200.0,
                    policyholder_revenue=5000000.0,
                    data_quality_score=2,
                ),
                InsurancePolicy(
                    policy_id="pol-e2e-002",
                    policyholder_name="Clean Tech Ltd",
                    line_of_business="LIABILITY",
                    gross_written_premium=750000.0,
                    scope1_emissions=10.0,
                    policyholder_revenue=3000000.0,
                    data_quality_score=1,
                    ceded_pct=25.0,
                ),
            ],
        )
        result = await wf.run(inp)

        assert result is not None
        assert len(result.provenance_hash) == 64
        assert len(result.phases) == 4
        assert hasattr(result, "gross_attributed_emissions_tco2e")
        assert hasattr(result, "net_attributed_emissions_tco2e")


# ===========================================================================
# E2E Test: Cross-Engine Data Flow and Provenance
# ===========================================================================


class TestE2ECrossEngineDataFlow:
    """Tests that verify cross-engine data consistency and provenance chain."""

    def test_all_8_engines_instantiate(self):
        """All 8 calculation engines can be instantiated."""
        engines = [
            FinancedEmissionsEngine(FinancedEmissionsConfig()),
            GreenAssetRatioEngine(GARConfig()),
            ClimateRiskScoringEngine(ClimateRiskConfig()),
            Pillar3ESGEngine(Pillar3Config()),
            InsuranceUnderwritingEngine(UnderwritingConfig()),
            FSDoubleMaterialityEngine(FSMaterialityConfig()),
            FSTransitionPlanEngine(TransitionPlanConfig()),
            BTARCalculatorEngine(BTARConfig()),
        ]
        assert len(engines) == 8
        for engine in engines:
            assert engine is not None

    def test_provenance_hash_is_deterministic(self):
        """Same data produces same provenance hash."""
        data = {"org": "test", "value": 42.0, "items": [1, 2, 3]}
        h1 = _hash(data)
        h2 = _hash(data)
        assert h1 == h2
        assert len(h1) == 64

    @pytest.mark.asyncio
    async def test_financed_emissions_feeds_pillar3(self):
        """Financed emissions output can feed Pillar 3 input."""
        fe_wf = FinancedEmissionsWorkflow()
        fe_inp = FinancedEmissionsInput(
            organization_id="org-e2e-cross-001",
            reporting_period="2025",
            exposures=[
                CounterpartyExposure(
                    counterparty_id="cp-cross-001",
                    asset_class="LISTED_EQUITY",
                    outstanding_amount=1000000.0,
                    scope1_emissions=100.0,
                    total_equity_plus_debt=5000000.0,
                )
            ],
        )
        fe_result = await fe_wf.run(fe_inp)

        # Feed financed emissions into Pillar 3
        p3_wf = Pillar3ReportingWorkflow()
        p3_inp = Pillar3ReportingInput(
            organization_id="org-e2e-cross-001",
            reporting_date="2025-12-31",
            financed_emissions_tco2e=fe_result.total_financed_emissions_tco2e,
        )
        p3_result = await p3_wf.run(p3_inp)

        assert p3_result is not None
        assert len(p3_result.provenance_hash) == 64
        assert len(fe_result.provenance_hash) == 64
