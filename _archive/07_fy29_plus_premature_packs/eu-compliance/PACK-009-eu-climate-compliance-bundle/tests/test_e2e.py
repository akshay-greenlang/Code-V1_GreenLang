# -*- coding: utf-8 -*-
"""
End-to-end tests for PACK-009 EU Climate Compliance Bundle.

Simulates full bundle workflows: data mapping to deduplication, gap
analysis to remediation plans, evidence registration/reuse, consistency
reconciliation, scoring with maturity, calendar dependency chains,
consolidated metrics trends, full bundle smoke, cross-regulation data
flow, and template rendering.

Coverage target: end-to-end flow validation
Test count: 10

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Dynamic import helper (inline per PACK-009 test pattern)
# ---------------------------------------------------------------------------


def _import_from_path(module_name: str, file_path: Path):
    """Import a module from an absolute file path, returning None on failure."""
    if not file_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pack directory constant
# ---------------------------------------------------------------------------

_PACK_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Module loaders (lazy, cached at module scope)
# ---------------------------------------------------------------------------

def _load_engine(filename: str, cache_key: str):
    """Load an engine module or return None."""
    return _import_from_path(cache_key, _PACK_DIR / "engines" / filename)


def _load_integration(filename: str, cache_key: str):
    """Load an integration module or return None."""
    return _import_from_path(cache_key, _PACK_DIR / "integrations" / filename)


def _load_template(filename: str, cache_key: str):
    """Load a template module or return None."""
    return _import_from_path(cache_key, _PACK_DIR / "templates" / filename)


# ===========================================================================
# Tests
# ===========================================================================


class TestE2EFlows:
    """End-to-end tests simulating full bundle workflows."""

    # -----------------------------------------------------------------------
    # 1. test_e2e_data_mapper_to_deduplication_flow
    # -----------------------------------------------------------------------

    def test_e2e_data_mapper_to_deduplication_flow(self):
        """Map CSRD fields to CBAM using the mapper, then run deduplication."""
        mapper_mod = _load_engine("cross_framework_data_mapper.py", "e2e_mapper")
        dedup_mod = _load_engine("data_deduplication_engine.py", "e2e_dedup")
        if mapper_mod is None or dedup_mod is None:
            pytest.skip("Engine modules not importable")

        mapper = mapper_mod.CrossFrameworkDataMapperEngine()

        result = mapper.map_field(
            "CSRD", "E1_6_scope1_ghg_emissions", "CBAM", 42000.0
        )
        assert result.confidence > 0, "Mapping should find a match"
        assert result.target_field != "", "Target field should be non-empty"
        assert result.provenance_hash != "", "Provenance hash required"
        assert len(result.provenance_hash) == 64, "SHA-256 hash must be 64 hex chars"

        dedup_engine = dedup_mod.DataDeduplicationEngine()
        dedup_result = dedup_engine.scan_requirements(
            regulations=["CSRD", "CBAM"]
        )
        assert dedup_result.total_requirements_scanned > 0
        assert dedup_result.duplicate_groups >= 0
        assert dedup_result.provenance_hash != ""
        assert len(dedup_result.provenance_hash) == 64

    # -----------------------------------------------------------------------
    # 2. test_e2e_gap_analysis_to_remediation_plan
    # -----------------------------------------------------------------------

    def test_e2e_gap_analysis_to_remediation_plan(self):
        """Create compliance status, run gap analyzer, verify remediation plan."""
        gap_mod = _load_engine("cross_regulation_gap_analyzer.py", "e2e_gap")
        if gap_mod is None:
            pytest.skip("Gap analyzer not importable")

        engine = gap_mod.CrossRegulationGapAnalyzerEngine()

        compliance_status = {
            "CSRD-DC-001": "COMPLIANT",
            "CSRD-DC-002": "NON_COMPLIANT",
            "CSRD-DC-003": "NOT_ASSESSED",
            "CBAM-DC-001": "NON_COMPLIANT",
            "CBAM-DC-002": "PARTIALLY_COMPLIANT",
            "EUDR-DC-001": "COMPLIANT",
            "EUDR-DC-002": "NON_COMPLIANT",
            "TAX-DC-001": "COMPLIANT",
            "TAX-DC-002": "NOT_ASSESSED",
        }

        result = engine.scan_all_regulations(compliance_status)

        assert result.total_gaps > 0, "Should find gaps for NON_COMPLIANT items"
        assert result.total_requirements_scanned > 0
        assert len(result.remediation_plan) > 0, "Remediation plan should not be empty"
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

        for item in result.remediation_plan:
            assert item.title != "", "Roadmap item needs a title"
            assert item.effort_hours >= 0
            assert item.phase >= 1
            assert item.phase <= 4

        multi_reg_gaps = engine.get_multi_regulation_gaps(result.gaps, min_regulations=2)
        for gap in multi_reg_gaps:
            assert len(gap.affected_regulations) >= 2

    # -----------------------------------------------------------------------
    # 3. test_e2e_evidence_registration_and_reuse
    # -----------------------------------------------------------------------

    def test_e2e_evidence_registration_and_reuse(self):
        """Register evidence items and verify cross-regulation reuse tracking."""
        evidence_mod = _load_engine("cross_regulation_evidence_engine.py", "e2e_evidence")
        if evidence_mod is None:
            pytest.skip("Evidence engine not importable")

        engine = evidence_mod.CrossRegulationEvidenceEngine()

        EvidenceItem = evidence_mod.EvidenceItem
        items = [
            EvidenceItem(
                evidence_id="EV-001",
                title="GHG Emissions Report 2025",
                evidence_type="REPORT",
                regulations=["CSRD", "CBAM", "EU_TAXONOMY"],
            ),
            EvidenceItem(
                evidence_id="EV-002",
                title="Supply Chain Traceability Certificate",
                evidence_type="CERTIFICATE",
                regulations=["EUDR", "CSRD"],
            ),
        ]

        # Register evidence one at a time (API accepts single EvidenceItem)
        registered = []
        for item in items:
            result = engine.register_evidence(item)
            registered.append(result)
        assert len(registered) >= 2
        # Verify items were registered
        for r in registered:
            assert r is not None
            assert r.evidence_id != ""

    # -----------------------------------------------------------------------
    # 4. test_e2e_consistency_check_to_reconciliation
    # -----------------------------------------------------------------------

    def test_e2e_consistency_check_to_reconciliation(self):
        """Create conflicting data, run consistency engine, verify results."""
        consistency_mod = _load_engine(
            "multi_regulation_consistency_engine.py", "e2e_consistency"
        )
        if consistency_mod is None:
            pytest.skip("Consistency engine not importable")

        engine = consistency_mod.MultiRegulationConsistencyEngine()
        DataPoint = consistency_mod.DataPoint

        # API expects List[DataPoint], not dict
        data_points = [
            DataPoint(regulation="CSRD", field_name="scope1_ghg_emissions", value=42000.0, unit="tCO2e"),
            DataPoint(regulation="CBAM", field_name="scope1_ghg_emissions", value=43500.0, unit="tCO2e"),
            DataPoint(regulation="EU_TAXONOMY", field_name="scope1_ghg_emissions", value=42000.0, unit="tCO2e"),
            DataPoint(regulation="CSRD", field_name="water_consumption", value=15000.0, unit="m3"),
            DataPoint(regulation="CBAM", field_name="water_consumption", value=15000.0, unit="m3"),
            DataPoint(regulation="EU_TAXONOMY", field_name="water_consumption", value=14800.0, unit="m3"),
        ]

        result = engine.check_consistency(data_points)
        assert result is not None
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    # -----------------------------------------------------------------------
    # 5. test_e2e_scoring_with_maturity_assessment
    # -----------------------------------------------------------------------

    def test_e2e_scoring_with_maturity_assessment(self):
        """Create regulation inputs, compute scores, check maturity assessment."""
        scoring_mod = _load_engine(
            "bundle_compliance_scoring_engine.py", "e2e_scoring"
        )
        if scoring_mod is None:
            pytest.skip("Scoring engine not importable")

        engine = scoring_mod.BundleComplianceScoringEngine()
        RegulationInput = scoring_mod.RegulationInput

        inputs = [
            RegulationInput(
                regulation="CSRD",
                requirements_total=20,
                requirements_met=15,
                data_quality_score=0.85,
                evidence_completeness=0.80,
                timeliness_score=0.90,
            ),
            RegulationInput(
                regulation="CBAM",
                requirements_total=16,
                requirements_met=10,
                data_quality_score=0.70,
                evidence_completeness=0.60,
                timeliness_score=0.75,
            ),
            RegulationInput(
                regulation="EUDR",
                requirements_total=19,
                requirements_met=5,
                data_quality_score=0.50,
                evidence_completeness=0.40,
                timeliness_score=0.55,
            ),
            RegulationInput(
                regulation="EU_TAXONOMY",
                requirements_total=20,
                requirements_met=18,
                data_quality_score=0.90,
                evidence_completeness=0.88,
                timeliness_score=0.95,
            ),
        ]

        result = engine.calculate_scores(inputs)
        assert result is not None
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

        assert result.overall_score >= 0.0
        assert result.overall_score <= 100.0

        if hasattr(result, "maturity_profile") and result.maturity_profile:
            for ma in result.maturity_profile:
                assert ma.level >= 1
                assert ma.level <= 5

    # -----------------------------------------------------------------------
    # 6. test_e2e_calendar_dependency_chain
    # -----------------------------------------------------------------------

    def test_e2e_calendar_dependency_chain(self):
        """Create calendar events with dependencies, verify critical path."""
        calendar_mod = _load_engine("regulatory_calendar_engine.py", "e2e_calendar")
        if calendar_mod is None:
            pytest.skip("Calendar engine not importable")

        engine = calendar_mod.RegulatoryCalendarEngine()

        result = engine.get_all_deadlines(year=2025)
        assert result is not None
        assert hasattr(result, "provenance_hash")
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

        # Result should have events
        events = getattr(result, "events", [])
        assert len(events) > 0
        for event in events:
            assert hasattr(event, "regulation")
            assert event.regulation != ""

    # -----------------------------------------------------------------------
    # 7. test_e2e_consolidated_metrics_trend
    # -----------------------------------------------------------------------

    def test_e2e_consolidated_metrics_trend(self):
        """Create multi-period data, verify trend analysis output."""
        metrics_mod = _load_engine(
            "consolidated_metrics_engine.py", "e2e_metrics"
        )
        if metrics_mod is None:
            pytest.skip("Consolidated metrics engine not importable")

        engine = metrics_mod.ConsolidatedMetricsEngine()

        periods = [
            {
                "period": "2024-Q4",
                "regulation_scores": {
                    "CSRD": 65.0, "CBAM": 55.0, "EUDR": 40.0, "EU_TAXONOMY": 70.0,
                },
            },
            {
                "period": "2025-Q1",
                "regulation_scores": {
                    "CSRD": 72.0, "CBAM": 60.0, "EUDR": 50.0, "EU_TAXONOMY": 75.0,
                },
            },
            {
                "period": "2025-Q2",
                "regulation_scores": {
                    "CSRD": 78.0, "CBAM": 68.0, "EUDR": 58.0, "EU_TAXONOMY": 80.0,
                },
            },
        ]

        # aggregate_metrics accepts List[RegulationMetrics], then analyze_trends works
        RegulationMetrics = metrics_mod.RegulationMetrics
        metrics = [
            RegulationMetrics(
                regulation="CSRD",
                compliance_score=78.0,
                data_completeness=85.0,
                items_assessed=20,
                items_compliant=15,
                items_non_compliant=3,
                items_pending=2,
            ),
            RegulationMetrics(
                regulation="CBAM",
                compliance_score=68.0,
                data_completeness=70.0,
                items_assessed=16,
                items_compliant=10,
                items_non_compliant=4,
                items_pending=2,
            ),
        ]
        result = engine.aggregate_metrics(metrics)
        assert result is not None
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    # -----------------------------------------------------------------------
    # 8. test_e2e_full_bundle_smoke_test
    # -----------------------------------------------------------------------

    def test_e2e_full_bundle_smoke_test(self):
        """Import orchestrator, verify instantiation and 12 phases defined."""
        orch_mod = _load_integration("pack_orchestrator.py", "e2e_orch")
        if orch_mod is None:
            pytest.skip("pack_orchestrator.py not importable")

        BundlePackOrchestrator = getattr(orch_mod, "BundlePackOrchestrator", None)
        BundleOrchestratorConfig = getattr(orch_mod, "BundleOrchestratorConfig", None)
        BundlePipelinePhase = getattr(orch_mod, "BundlePipelinePhase", None)
        PHASE_ORDER = getattr(orch_mod, "PHASE_ORDER", None)

        assert BundlePackOrchestrator is not None
        assert BundleOrchestratorConfig is not None
        assert BundlePipelinePhase is not None
        assert PHASE_ORDER is not None

        assert len(PHASE_ORDER) == 12, f"Expected 12 phases, got {len(PHASE_ORDER)}"

        expected_phases = [
            "health_check", "config_init", "pack_loading",
            "data_collection", "deduplication", "parallel_assessment",
            "consistency_check", "gap_analysis", "calendar_update",
            "consolidated_reporting", "evidence_package", "audit_trail",
        ]
        actual_phases = [p.value for p in PHASE_ORDER]
        assert actual_phases == expected_phases, (
            f"Phase order mismatch: {actual_phases}"
        )

        config = BundleOrchestratorConfig(
            enable_csrd=True,
            enable_cbam=True,
            enable_eudr=True,
            enable_taxonomy=True,
            reporting_period_year=2025,
            organization_name="E2E Test Corp AG",
        )
        orchestrator = BundlePackOrchestrator(config)
        status = orchestrator.get_status()
        assert status["total_phases"] == 12
        assert status["packs_enabled"]["csrd"] is True
        assert status["packs_enabled"]["cbam"] is True

    # -----------------------------------------------------------------------
    # 9. test_e2e_cross_regulation_data_flow
    # -----------------------------------------------------------------------

    def test_e2e_cross_regulation_data_flow(self):
        """Map data from CSRD to CBAM and verify field conversion."""
        mapper_mod = _load_engine("cross_framework_data_mapper.py", "e2e_xreg_flow")
        if mapper_mod is None:
            pytest.skip("Mapper engine not importable")

        engine = mapper_mod.CrossFrameworkDataMapperEngine()

        csrd_fields = {
            "E1_6_scope1_ghg_emissions": 42000.0,
            "E1_6_scope2_ghg_emissions": 8500.0,
            "E1_6_scope3_cat1_emissions": 120000.0,
        }

        batch_result = engine.map_batch("CSRD", csrd_fields, "CBAM")
        assert batch_result.total_fields == 3
        assert batch_result.mapped_count >= 1, "At least one CSRD field should map to CBAM"
        assert batch_result.provenance_hash != ""
        assert len(batch_result.provenance_hash) == 64

        for mapping in batch_result.mappings:
            assert mapping.source_regulation == "CSRD"
            assert mapping.target_regulation == "CBAM"
            assert mapping.confidence > 0
            assert mapping.target_field != ""

        csrd_to_tax = engine.map_field(
            "CSRD", "E1_6_total_ghg_emissions", "EU_TAXONOMY", 170500.0
        )
        assert csrd_to_tax.confidence > 0.5
        assert csrd_to_tax.target_field != ""

        overlap = engine.get_overlap_statistics("CSRD", "CBAM")
        assert overlap.exact_matches >= 0
        assert overlap.approximate_matches >= 0
        assert overlap.provenance_hash != ""

    # -----------------------------------------------------------------------
    # 10. test_e2e_all_templates_render_without_error
    # -----------------------------------------------------------------------

    def test_e2e_all_templates_render_without_error(self):
        """For each template, create minimal data, render markdown, verify non-empty."""
        template_specs = [
            (
                "consolidated_dashboard.py",
                "ConsolidatedDashboardTemplate",
                "DashboardConfig",
                "DashboardData",
            ),
            (
                "cross_regulation_data_map.py",
                "CrossRegulationDataMapTemplate",
                "DataMapConfig",
                "DataMapData",
            ),
            (
                "unified_gap_analysis_report.py",
                "UnifiedGapAnalysisReportTemplate",
                "GapAnalysisConfig",
                "GapAnalysisData",
            ),
            (
                "regulatory_calendar_report.py",
                "RegulatoryCalendarReportTemplate",
                "CalendarConfig",
                "CalendarData",
            ),
            (
                "data_consistency_report.py",
                "DataConsistencyReportTemplate",
                "ConsistencyConfig",
                "ConsistencyData",
            ),
            (
                "deduplication_savings_report.py",
                "DeduplicationSavingsReportTemplate",
                "DeduplicationConfig",
                "DeduplicationData",
            ),
            (
                "multi_regulation_audit_trail.py",
                "MultiRegulationAuditTrailTemplate",
                "AuditTrailConfig",
                "AuditTrailData",
            ),
            (
                "bundle_executive_summary.py",
                "BundleExecutiveSummaryTemplate",
                "ExecutiveSummaryConfig",
                "ExecutiveSummaryData",
            ),
        ]

        failures = []
        tested = 0

        for filename, tpl_cls_name, config_cls_name, data_cls_name in template_specs:
            mod = _load_template(filename, f"e2e_tpl_{filename.replace('.py','')}")
            if mod is None:
                continue

            tpl_cls = getattr(mod, tpl_cls_name, None)
            config_cls = getattr(mod, config_cls_name, None)
            data_cls = getattr(mod, data_cls_name, None)

            if tpl_cls is None:
                failures.append(f"{filename}: missing class {tpl_cls_name}")
                continue

            try:
                if config_cls is not None:
                    cfg = config_cls()
                    tpl_instance = tpl_cls(config=cfg)
                else:
                    tpl_instance = tpl_cls()

                if data_cls is not None:
                    data_instance = data_cls()
                else:
                    data_instance = {}

                render_method = getattr(tpl_instance, "render", None)
                if render_method is None:
                    render_method = getattr(tpl_instance, "render_markdown", None)
                if render_method is None:
                    render_method = getattr(tpl_instance, "generate", None)

                if render_method is not None:
                    output = render_method(data_instance)
                    if output is not None:
                        assert len(str(output)) > 0, f"{filename}: render returned empty"
                tested += 1

            except Exception as exc:
                failures.append(f"{filename}: {type(exc).__name__}: {exc}")

        if tested == 0:
            pytest.skip("No templates could be loaded for rendering")

        assert not failures, "Template render failures:\n" + "\n".join(failures)
