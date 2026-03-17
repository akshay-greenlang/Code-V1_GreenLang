# -*- coding: utf-8 -*-
"""
Demo / smoke tests for PACK-009 EU Climate Compliance Bundle.

Validates that the demo configuration YAML is present, parseable, and
contains required keys. Verifies that all engine, workflow, template,
and integration modules are importable and expose expected classes.

Coverage target: smoke-level
Test count: 8

Author: GreenLang QA Team
Version: 1.0.0
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

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
# Pack directory
# ---------------------------------------------------------------------------

_PACK_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Module file inventories
# ---------------------------------------------------------------------------

ENGINE_FILES = [
    "cross_framework_data_mapper.py",
    "data_deduplication_engine.py",
    "cross_regulation_gap_analyzer.py",
    "multi_regulation_consistency_engine.py",
    "cross_regulation_evidence_engine.py",
    "bundle_compliance_scoring_engine.py",
    "regulatory_calendar_engine.py",
    "consolidated_metrics_engine.py",
]

ENGINE_CLASSES = [
    ("cross_framework_data_mapper.py", "CrossFrameworkDataMapperEngine"),
    ("data_deduplication_engine.py", "DataDeduplicationEngine"),
    ("cross_regulation_gap_analyzer.py", "CrossRegulationGapAnalyzerEngine"),
    ("multi_regulation_consistency_engine.py", "MultiRegulationConsistencyEngine"),
    ("cross_regulation_evidence_engine.py", "CrossRegulationEvidenceEngine"),
    ("bundle_compliance_scoring_engine.py", "BundleComplianceScoringEngine"),
    ("regulatory_calendar_engine.py", "RegulatoryCalendarEngine"),
    ("consolidated_metrics_engine.py", "ConsolidatedMetricsEngine"),
]

WORKFLOW_FILES = [
    "unified_data_collection.py",
    "cross_regulation_assessment.py",
    "consolidated_reporting.py",
    "calendar_management.py",
    "cross_framework_gap_analysis.py",
    "bundle_health_check.py",
    "data_consistency_reconciliation.py",
    "annual_compliance_review.py",
]

WORKFLOW_CLASSES = [
    ("unified_data_collection.py", "UnifiedDataCollectionWorkflow"),
    ("cross_regulation_assessment.py", "CrossRegulationAssessmentWorkflow"),
    ("consolidated_reporting.py", "ConsolidatedReportingWorkflow"),
    ("calendar_management.py", "CalendarManagementWorkflow"),
    ("cross_framework_gap_analysis.py", "CrossFrameworkGapAnalysisWorkflow"),
    ("bundle_health_check.py", "BundleHealthCheckWorkflow"),
    ("data_consistency_reconciliation.py", "DataConsistencyReconciliationWorkflow"),
    ("annual_compliance_review.py", "AnnualComplianceReviewWorkflow"),
]

TEMPLATE_FILES = [
    "consolidated_dashboard.py",
    "cross_regulation_data_map.py",
    "unified_gap_analysis_report.py",
    "regulatory_calendar_report.py",
    "data_consistency_report.py",
    "deduplication_savings_report.py",
    "multi_regulation_audit_trail.py",
    "bundle_executive_summary.py",
]

TEMPLATE_CLASSES = [
    ("consolidated_dashboard.py", "ConsolidatedDashboardTemplate"),
    ("cross_regulation_data_map.py", "CrossRegulationDataMapTemplate"),
    ("unified_gap_analysis_report.py", "UnifiedGapAnalysisReportTemplate"),
    ("regulatory_calendar_report.py", "RegulatoryCalendarReportTemplate"),
    ("data_consistency_report.py", "DataConsistencyReportTemplate"),
    ("deduplication_savings_report.py", "DeduplicationSavingsReportTemplate"),
    ("multi_regulation_audit_trail.py", "MultiRegulationAuditTrailTemplate"),
    ("bundle_executive_summary.py", "BundleExecutiveSummaryTemplate"),
]

INTEGRATION_FILES = [
    "pack_orchestrator.py",
    "csrd_pack_bridge.py",
    "cbam_pack_bridge.py",
    "eudr_pack_bridge.py",
    "taxonomy_pack_bridge.py",
    "cross_framework_mapper_bridge.py",
    "shared_data_pipeline_bridge.py",
    "consolidated_evidence_bridge.py",
]

INTEGRATION_CLASSES = [
    ("pack_orchestrator.py", "BundlePackOrchestrator"),
    ("csrd_pack_bridge.py", "CSRDPackBridge"),
    ("cbam_pack_bridge.py", "CBAMPackBridge"),
    ("eudr_pack_bridge.py", "EUDRPackBridge"),
    ("taxonomy_pack_bridge.py", "TaxonomyPackBridge"),
    ("cross_framework_mapper_bridge.py", "CrossFrameworkMapperBridge"),
    ("shared_data_pipeline_bridge.py", "SharedDataPipelineBridge"),
    ("consolidated_evidence_bridge.py", "ConsolidatedEvidenceBridge"),
]


# ===========================================================================
# Tests
# ===========================================================================


class TestDemoSmoke:
    """Smoke tests for PACK-009 demo configuration and module imports."""

    # -----------------------------------------------------------------------
    # 1. test_demo_config_yaml_exists
    # -----------------------------------------------------------------------

    def test_demo_config_yaml_exists(self):
        """Verify the demo YAML configuration file is on disk."""
        demo_yaml = _PACK_DIR / "config" / "demo" / "demo_config.yaml"
        assert demo_yaml.exists(), f"demo_config.yaml not found at {demo_yaml}"
        assert demo_yaml.stat().st_size > 0, "demo_config.yaml is empty"

    # -----------------------------------------------------------------------
    # 2. test_demo_config_yaml_valid
    # -----------------------------------------------------------------------

    def test_demo_config_yaml_valid(self):
        """Verify the demo YAML file is valid YAML that loads without error."""
        yaml = pytest.importorskip("yaml")
        demo_yaml = _PACK_DIR / "config" / "demo" / "demo_config.yaml"
        if not demo_yaml.exists():
            pytest.skip("demo_config.yaml not found")

        with open(demo_yaml, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        assert isinstance(data, dict), "demo_config.yaml root should be a mapping"
        assert len(data) > 0, "demo_config.yaml should have at least one key"

    # -----------------------------------------------------------------------
    # 3. test_demo_config_has_regulation_settings
    # -----------------------------------------------------------------------

    def test_demo_config_has_regulation_settings(self):
        """Verify the demo YAML includes regulation_configs with CSRD, CBAM, EUDR, TAXONOMY."""
        yaml = pytest.importorskip("yaml")
        demo_yaml = _PACK_DIR / "config" / "demo" / "demo_config.yaml"
        if not demo_yaml.exists():
            pytest.skip("demo_config.yaml not found")

        with open(demo_yaml, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        assert "regulation_configs" in data, "Missing 'regulation_configs' key"
        reg_configs = data["regulation_configs"]
        for reg in ("CSRD", "CBAM", "EUDR", "TAXONOMY"):
            assert reg in reg_configs, f"regulation_configs missing '{reg}'"
            reg_block = reg_configs[reg]
            assert "enabled" in reg_block, f"{reg} missing 'enabled' flag"
            assert "pack_id" in reg_block, f"{reg} missing 'pack_id'"
            assert "scoring_weight" in reg_block, f"{reg} missing 'scoring_weight'"

        required_top_keys = [
            "calendar", "deduplication", "consistency", "gap_analysis",
            "evidence", "reporting", "scoring", "data_mapper", "demo",
        ]
        for key in required_top_keys:
            assert key in data, f"demo_config.yaml missing top-level key '{key}'"

    # -----------------------------------------------------------------------
    # 4. test_demo_all_engines_importable
    # -----------------------------------------------------------------------

    def test_demo_all_engines_importable(self):
        """Verify all 8 engine modules can be imported and expose expected classes."""
        engines_dir = _PACK_DIR / "engines"
        failures = []

        for filename, expected_class in ENGINE_CLASSES:
            fpath = engines_dir / filename
            mod = _import_from_path(f"pack009_eng_{filename.replace('.py','')}", fpath)
            if mod is None:
                failures.append(f"{filename}: import returned None")
                continue
            if not hasattr(mod, expected_class):
                failures.append(f"{filename}: missing class {expected_class}")

        assert not failures, "Engine import failures:\n" + "\n".join(failures)

    # -----------------------------------------------------------------------
    # 5. test_demo_all_workflows_importable
    # -----------------------------------------------------------------------

    def test_demo_all_workflows_importable(self):
        """Verify all 8 workflow modules can be imported and expose expected classes."""
        wf_dir = _PACK_DIR / "workflows"
        failures = []

        for filename, expected_class in WORKFLOW_CLASSES:
            fpath = wf_dir / filename
            mod = _import_from_path(f"pack009_wf_{filename.replace('.py','')}", fpath)
            if mod is None:
                failures.append(f"{filename}: import returned None")
                continue
            if not hasattr(mod, expected_class):
                failures.append(f"{filename}: missing class {expected_class}")

        assert not failures, "Workflow import failures:\n" + "\n".join(failures)

    # -----------------------------------------------------------------------
    # 6. test_demo_all_templates_importable
    # -----------------------------------------------------------------------

    def test_demo_all_templates_importable(self):
        """Verify all 8 template modules can be imported and expose expected classes."""
        tpl_dir = _PACK_DIR / "templates"
        failures = []

        for filename, expected_class in TEMPLATE_CLASSES:
            fpath = tpl_dir / filename
            mod = _import_from_path(f"pack009_tpl_{filename.replace('.py','')}", fpath)
            if mod is None:
                failures.append(f"{filename}: import returned None")
                continue
            if not hasattr(mod, expected_class):
                failures.append(f"{filename}: missing class {expected_class}")

        assert not failures, "Template import failures:\n" + "\n".join(failures)

    # -----------------------------------------------------------------------
    # 7. test_demo_all_integrations_importable
    # -----------------------------------------------------------------------

    def test_demo_all_integrations_importable(self):
        """Verify all 8 integration modules can be imported and expose expected classes."""
        integ_dir = _PACK_DIR / "integrations"
        failures = []

        for filename, expected_class in INTEGRATION_CLASSES:
            fpath = integ_dir / filename
            mod = _import_from_path(f"pack009_integ_{filename.replace('.py','')}", fpath)
            if mod is None:
                failures.append(f"{filename}: import returned None")
                continue
            if not hasattr(mod, expected_class):
                failures.append(f"{filename}: missing class {expected_class}")

        assert not failures, "Integration import failures:\n" + "\n".join(failures)

    # -----------------------------------------------------------------------
    # 8. test_demo_bundle_orchestrator_smoke_test
    # -----------------------------------------------------------------------

    def test_demo_bundle_orchestrator_smoke_test(self):
        """Import the BundlePackOrchestrator and instantiate with default config."""
        orch_path = _PACK_DIR / "integrations" / "pack_orchestrator.py"
        mod = _import_from_path("pack009_orch_smoke", orch_path)
        if mod is None:
            pytest.skip("pack_orchestrator.py could not be imported")

        BundlePackOrchestrator = getattr(mod, "BundlePackOrchestrator", None)
        BundleOrchestratorConfig = getattr(mod, "BundleOrchestratorConfig", None)
        BundlePipelinePhase = getattr(mod, "BundlePipelinePhase", None)

        assert BundlePackOrchestrator is not None, "Missing BundlePackOrchestrator"
        assert BundleOrchestratorConfig is not None, "Missing BundleOrchestratorConfig"
        assert BundlePipelinePhase is not None, "Missing BundlePipelinePhase"

        config = BundleOrchestratorConfig(
            enable_csrd=True,
            enable_cbam=False,
            enable_eudr=False,
            enable_taxonomy=True,
            reporting_period_year=2025,
            organization_name="Demo Smoke Corp",
        )
        orchestrator = BundlePackOrchestrator(config)

        assert orchestrator.config.enable_csrd is True
        assert orchestrator.config.enable_taxonomy is True
        assert orchestrator.config.organization_name == "Demo Smoke Corp"

        status = orchestrator.get_status()
        assert status["total_phases"] == 12
        assert status["completed_phases"] == 0

        phases = list(BundlePipelinePhase)
        assert len(phases) == 12
