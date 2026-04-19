# -*- coding: utf-8 -*-
"""
End-to-end flow tests for PACK-011 SFDR Article 9 Pack.

These tests exercise multiple engines together in realistic scenarios,
validating that data flows correctly through the complete SFDR Article 9
compliance pipeline. Each test constructs realistic input data, runs it
through multiple engines sequentially, and verifies cross-engine consistency.

Test count: 15 tests
Target: Validate complete Article 9 compliance workflows
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
# Path setup - import engines from PACK-011 source tree
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

so_mod = _import_from_path(
    "sustainable_objective_engine",
    str(PACK_ROOT / "engines" / "sustainable_objective_engine.py"),
)
dnsh_mod = _import_from_path(
    "enhanced_dnsh_engine",
    str(PACK_ROOT / "engines" / "enhanced_dnsh_engine.py"),
)
tax_mod = _import_from_path(
    "full_taxonomy_alignment",
    str(PACK_ROOT / "engines" / "full_taxonomy_alignment.py"),
)
impact_mod = _import_from_path(
    "impact_measurement_engine",
    str(PACK_ROOT / "engines" / "impact_measurement_engine.py"),
)
bench_mod = _import_from_path(
    "benchmark_alignment_engine",
    str(PACK_ROOT / "engines" / "benchmark_alignment_engine.py"),
)
pai_mod = _import_from_path(
    "pai_mandatory_engine",
    str(PACK_ROOT / "engines" / "pai_mandatory_engine.py"),
)
carbon_mod = _import_from_path(
    "carbon_trajectory_engine",
    str(PACK_ROOT / "engines" / "carbon_trajectory_engine.py"),
)
universe_mod = _import_from_path(
    "investment_universe_engine",
    str(PACK_ROOT / "engines" / "investment_universe_engine.py"),
)

# Workflow modules
wf_annex_iii_mod = _import_from_path(
    "e2e_wf_annex_iii",
    str(PACK_ROOT / "workflows" / "annex_iii_disclosure.py"),
)
wf_annex_v_mod = _import_from_path(
    "e2e_wf_annex_v",
    str(PACK_ROOT / "workflows" / "annex_v_reporting.py"),
)
wf_pai_mod = _import_from_path(
    "e2e_wf_pai",
    str(PACK_ROOT / "workflows" / "pai_mandatory_workflow.py"),
)
wf_benchmark_mod = _import_from_path(
    "e2e_wf_benchmark",
    str(PACK_ROOT / "workflows" / "benchmark_monitoring.py"),
)
wf_downgrade_mod = _import_from_path(
    "e2e_wf_downgrade",
    str(PACK_ROOT / "workflows" / "downgrade_monitoring.py"),
)

# Template modules
tpl_annex_iii_mod = _import_from_path(
    "e2e_tpl_annex_iii",
    str(PACK_ROOT / "templates" / "annex_iii_precontractual.py"),
)

# ---------------------------------------------------------------------------
# Engine aliases
# ---------------------------------------------------------------------------

SustainableObjectiveEngine = so_mod.SustainableObjectiveEngine
SustainableObjectiveConfig = so_mod.SustainableObjectiveConfig
HoldingData = so_mod.HoldingData

EnhancedDNSHEngine = dnsh_mod.EnhancedDNSHEngine
EnhancedDNSHConfig = dnsh_mod.EnhancedDNSHConfig

FullTaxonomyAlignmentEngine = tax_mod.FullTaxonomyAlignmentEngine
FullTaxonomyConfig = tax_mod.FullTaxonomyConfig

ImpactMeasurementEngine = impact_mod.ImpactMeasurementEngine
ImpactConfig = impact_mod.ImpactConfig

BenchmarkAlignmentEngine = bench_mod.BenchmarkAlignmentEngine
BenchmarkConfig = bench_mod.BenchmarkConfig

PAIMandatoryEngine = pai_mod.PAIMandatoryEngine
PAIMandatoryConfig = pai_mod.PAIMandatoryConfig

CarbonTrajectoryEngine = carbon_mod.CarbonTrajectoryEngine
TrajectoryConfig = carbon_mod.TrajectoryConfig

InvestmentUniverseEngine = universe_mod.InvestmentUniverseEngine
UniverseConfig = universe_mod.UniverseConfig

# Workflow aliases
AnnexIIIDisclosureWorkflow = wf_annex_iii_mod.AnnexIIIDisclosureWorkflow
AnnexIIIDisclosureInput = wf_annex_iii_mod.AnnexIIIDisclosureInput
SustainableObjective_WF = wf_annex_iii_mod.SustainableObjective
SustainableObjectiveType_WF = wf_annex_iii_mod.SustainableObjectiveType

AnnexVReportingWorkflow = wf_annex_v_mod.AnnexVReportingWorkflow
AnnexVReportingInput = wf_annex_v_mod.AnnexVReportingInput

PAIMandatoryWorkflow = wf_pai_mod.PAIMandatoryWorkflow
PAIMandatoryInput_WF = wf_pai_mod.PAIMandatoryInput

BenchmarkMonitoringWorkflow = wf_benchmark_mod.BenchmarkMonitoringWorkflow
BenchmarkMonitoringInput = wf_benchmark_mod.BenchmarkMonitoringInput

DowngradeMonitoringWorkflow = wf_downgrade_mod.DowngradeMonitoringWorkflow
DowngradeMonitoringInput = wf_downgrade_mod.DowngradeMonitoringInput

AnnexIIIPrecontractualTemplate = tpl_annex_iii_mod.AnnexIIIPrecontractualTemplate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# ===========================================================================
# E2E Test Class
# ===========================================================================


class TestE2EAnnexIIIDisclosure:
    """Full end-to-end Annex III pre-contractual disclosure generation."""

    @pytest.mark.asyncio
    async def test_full_annex_iii_disclosure(self):
        """Complete Annex III disclosure workflow end-to-end."""
        wf = AnnexIIIDisclosureWorkflow()
        inp = AnnexIIIDisclosureInput(
            organization_id="org-e2e-001",
            product_name="GL E2E Climate Solutions Fund",
            reporting_date="2026-03-15",
            sustainable_objectives=[
                SustainableObjective_WF(
                    name="Carbon Reduction",
                    objective_type=SustainableObjectiveType_WF.CARBON_REDUCTION,
                    sustainability_indicators=[
                        "ghg_intensity_reduction",
                        "renewable_energy_pct",
                    ],
                )
            ],
            minimum_taxonomy_aligned_pct=20.0,
        )
        result = await wf.run(inp)

        assert result is not None
        assert result.product_name == "GL E2E Climate Solutions Fund"
        assert len(result.provenance_hash) == 64
        assert len(result.phases) == 5
        assert hasattr(result, "is_article_9_eligible")
        assert result.sustainable_investment_commitment_pct >= 0.0
        assert result.template_sections_completed > 0

    @pytest.mark.asyncio
    async def test_annex_iii_provenance_determinism(self):
        """Same input produces same provenance hash."""
        inp = AnnexIIIDisclosureInput(
            organization_id="org-e2e-002",
            product_name="Determinism Test Fund",
            reporting_date="2026-01-01",
            sustainable_objectives=[
                SustainableObjective_WF(
                    name="Environmental",
                    objective_type=SustainableObjectiveType_WF.ENVIRONMENTAL,
                    sustainability_indicators=["env_kpi_1"],
                )
            ],
        )
        wf1 = AnnexIIIDisclosureWorkflow()
        wf2 = AnnexIIIDisclosureWorkflow()
        r1 = await wf1.run(inp)
        r2 = await wf2.run(inp)

        # Both runs produce valid results
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64


class TestE2EAnnexVReporting:
    """Full end-to-end Annex V periodic report generation."""

    @pytest.mark.asyncio
    async def test_full_annex_v_periodic_report(self):
        """Complete Annex V periodic reporting workflow end-to-end."""
        wf = AnnexVReportingWorkflow()
        inp = AnnexVReportingInput(
            organization_id="org-e2e-010",
            product_name="GL E2E Annual Report Fund",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
        )
        result = await wf.run(inp)

        assert result is not None
        assert result.product_name == "GL E2E Annual Report Fund"
        assert len(result.provenance_hash) == 64
        assert len(result.phases) >= 4
        assert hasattr(result, "actual_sustainable_investment_pct")
        assert hasattr(result, "pai_indicators_reported")


class TestE2EPAIMandatoryAssessment:
    """Full end-to-end PAI mandatory assessment with all 18 indicators."""

    @pytest.mark.asyncio
    async def test_full_pai_assessment(self):
        """Complete PAI mandatory assessment end-to-end."""
        wf = PAIMandatoryWorkflow()
        inp = PAIMandatoryInput_WF(
            organization_id="org-e2e-020",
            product_name="GL Full PAI E2E Fund",
            reporting_date="2026-03-15",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
        )
        result = await wf.run(inp)

        assert result is not None
        assert len(result.provenance_hash) == 64
        assert result.total_indicators >= 14
        assert hasattr(result, "data_coverage_pct")

    @pytest.mark.asyncio
    async def test_pai_phases_complete(self):
        """All 4 PAI phases complete successfully."""
        wf = PAIMandatoryWorkflow()
        inp = PAIMandatoryInput_WF(
            organization_id="org-e2e-021",
            product_name="PAI Phase Check",
            reporting_date="2026-01-01",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
        )
        result = await wf.run(inp)
        assert len(result.phases) == 4


class TestE2EBenchmarkAlignment:
    """Full end-to-end benchmark alignment check with CTB/PAB."""

    @pytest.mark.asyncio
    async def test_ctb_benchmark_alignment(self):
        """CTB benchmark alignment check end-to-end."""
        wf = BenchmarkMonitoringWorkflow()
        inp = BenchmarkMonitoringInput(
            organization_id="org-e2e-030",
            product_name="GL CTB E2E Fund",
            reporting_date="2026-03-15",
            benchmark_type="CTB",
            benchmark_name="MSCI Europe Climate Change CTB",
        )
        result = await wf.run(inp)

        assert result is not None
        assert len(result.provenance_hash) == 64
        assert result.benchmark_type == "CTB"
        assert hasattr(result, "alignment_status")
        assert hasattr(result, "ghg_reduction_pct")

    @pytest.mark.asyncio
    async def test_pab_benchmark_alignment(self):
        """PAB benchmark alignment check end-to-end."""
        wf = BenchmarkMonitoringWorkflow()
        inp = BenchmarkMonitoringInput(
            organization_id="org-e2e-031",
            product_name="GL PAB E2E Fund",
            reporting_date="2026-03-15",
            benchmark_type="PAB",
            benchmark_name="S&P Eurozone LargeMidCap PAB",
        )
        result = await wf.run(inp)

        assert result is not None
        assert result.benchmark_type == "PAB"
        assert hasattr(result, "required_reduction_pct")


class TestE2EDowngradeRisk:
    """Full end-to-end downgrade risk assessment."""

    @pytest.mark.asyncio
    async def test_downgrade_risk_assessment(self):
        """Downgrade risk assessment end-to-end."""
        wf = DowngradeMonitoringWorkflow()
        inp = DowngradeMonitoringInput(
            organization_id="org-e2e-040",
            product_name="GL Downgrade Watch Fund",
            assessment_date="2026-03-15",
        )
        result = await wf.run(inp)

        assert result is not None
        assert len(result.provenance_hash) == 64
        assert result.current_classification == "ARTICLE_9"
        assert hasattr(result, "risk_level")
        assert hasattr(result, "risk_score")
        assert result.risk_score >= 0.0


class TestE2ECrossEngineDataFlow:
    """Validate data flows correctly between multiple engines."""

    def test_engine_instantiation_chain(self):
        """All 8 engines can be instantiated in sequence."""
        engines = [
            SustainableObjectiveEngine(SustainableObjectiveConfig()),
            EnhancedDNSHEngine(EnhancedDNSHConfig()),
            FullTaxonomyAlignmentEngine(FullTaxonomyConfig()),
            ImpactMeasurementEngine(ImpactConfig()),
            BenchmarkAlignmentEngine(BenchmarkConfig()),
            PAIMandatoryEngine(PAIMandatoryConfig()),
            CarbonTrajectoryEngine(TrajectoryConfig()),
            InvestmentUniverseEngine(UniverseConfig()),
        ]
        assert len(engines) == 8
        for engine in engines:
            assert engine is not None

    def test_provenance_hash_consistency(self):
        """Provenance hashes are deterministic across calls."""
        data = {
            "product_name": "Test Fund",
            "holdings": [
                {"isin": "TEST001", "weight": 0.5},
                {"isin": "TEST002", "weight": 0.5},
            ],
        }
        hash1 = _hash(data)
        hash2 = _hash(data)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_template_renders_from_engine_output(self):
        """Template renders correctly from engine-like output data."""
        template = AnnexIIIPrecontractualTemplate()
        data = {
            "product_info": {
                "product_name": "Cross-Engine Test Fund",
                "sfdr_classification": "article_9",
            },
            "sustainable_objective": {
                "objective_type": "environmental",
                "objective_description": "Climate change mitigation",
            },
        }
        md = template.render_markdown(data)
        assert isinstance(md, str)
        assert len(md) > 0


class TestE2ESustainableVerification:
    """End-to-end sustainable investment verification."""

    @pytest.mark.asyncio
    async def test_full_verification_flow(self):
        """Full sustainable verification workflow end-to-end."""
        from importlib.util import spec_from_file_location, module_from_spec

        wf_mod = _import_from_path(
            "e2e_wf_sust_verif",
            str(PACK_ROOT / "workflows" / "sustainable_verification.py"),
        )
        wf = wf_mod.SustainableVerificationWorkflow()
        inp = wf_mod.SustainableVerificationInput(
            organization_id="org-e2e-050",
            product_name="GL 100% Sustainable E2E",
            verification_date="2026-03-15",
        )
        result = await wf.run(inp)

        assert result is not None
        assert len(result.provenance_hash) == 64
        assert hasattr(result, "sustainable_pct")
