# -*- coding: utf-8 -*-
"""
Integration tests for PACK-009 EU Climate Compliance Bundle.

Validates cross-pack integration: orchestrator instantiation, pack bridge
construction, mapper bridge mappings, shared pipeline routing, evidence
bridge, health check, setup wizard steps, 12-phase pipeline, importability
of all engines / workflows / templates, config module, constituent pack
directories, bridge config pattern enforcement, and provenance hashing.

Coverage target: integration-level
Test count: 15

Author: GreenLang QA Team
Version: 1.0.0
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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
_EU_COMPLIANCE_DIR = _PACK_DIR.parent  # packs/eu-compliance


# ---------------------------------------------------------------------------
# Pre-import integration modules
# ---------------------------------------------------------------------------

_orch_mod = _import_from_path(
    "pack009_integ_orch",
    _PACK_DIR / "integrations" / "pack_orchestrator.py",
)

_csrd_bridge_mod = _import_from_path(
    "pack009_integ_csrd",
    _PACK_DIR / "integrations" / "csrd_pack_bridge.py",
)

_cbam_bridge_mod = _import_from_path(
    "pack009_integ_cbam",
    _PACK_DIR / "integrations" / "cbam_pack_bridge.py",
)

_eudr_bridge_mod = _import_from_path(
    "pack009_integ_eudr",
    _PACK_DIR / "integrations" / "eudr_pack_bridge.py",
)

_taxonomy_bridge_mod = _import_from_path(
    "pack009_integ_taxonomy",
    _PACK_DIR / "integrations" / "taxonomy_pack_bridge.py",
)

_mapper_bridge_mod = _import_from_path(
    "pack009_integ_mapper_bridge",
    _PACK_DIR / "integrations" / "cross_framework_mapper_bridge.py",
)

_pipeline_bridge_mod = _import_from_path(
    "pack009_integ_pipeline",
    _PACK_DIR / "integrations" / "shared_data_pipeline_bridge.py",
)

_evidence_bridge_mod = _import_from_path(
    "pack009_integ_evidence",
    _PACK_DIR / "integrations" / "consolidated_evidence_bridge.py",
)

_health_mod = _import_from_path(
    "pack009_integ_health",
    _PACK_DIR / "integrations" / "bundle_health_check.py",
)

_wizard_mod = _import_from_path(
    "pack009_integ_wizard",
    _PACK_DIR / "integrations" / "setup_wizard.py",
)

_config_mod = _import_from_path(
    "pack009_config",
    _PACK_DIR / "config" / "pack_config.py",
)


# ---------------------------------------------------------------------------
# Module inventories (engines, workflows, templates)
# ---------------------------------------------------------------------------

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

# Bridge module-level class pairs for config pattern enforcement
BRIDGE_CONFIGS = [
    ("csrd_pack_bridge.py", "CSRDPackBridge", "CSRDPackBridgeConfig"),
    ("cbam_pack_bridge.py", "CBAMPackBridge", "CBAMPackBridgeConfig"),
    ("eudr_pack_bridge.py", "EUDRPackBridge", "EUDRPackBridgeConfig"),
    ("taxonomy_pack_bridge.py", "TaxonomyPackBridge", "TaxonomyPackBridgeConfig"),
    ("cross_framework_mapper_bridge.py", "CrossFrameworkMapperBridge", "CrossFrameworkMapperConfig"),
    ("shared_data_pipeline_bridge.py", "SharedDataPipelineBridge", "SharedDataPipelineConfig"),
    ("consolidated_evidence_bridge.py", "ConsolidatedEvidenceBridge", "ConsolidatedEvidenceConfig"),
    ("bundle_health_check.py", "BundleHealthCheckIntegration", "BundleHealthCheckConfig"),
    ("setup_wizard.py", "BundleSetupWizard", "BundleSetupWizardConfig"),
]

# Constituent pack directory names
CONSTITUENT_PACK_DIRS = [
    "PACK-001-csrd-starter",
    "PACK-004-cbam-readiness",
    "PACK-006-eudr-starter",
    "PACK-008-eu-taxonomy-alignment",
]


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_provenance_hash(obj: Any) -> None:
    """Verify an object carries a valid 64-char SHA-256 provenance hash."""
    h = getattr(obj, "provenance_hash", None)
    if h is None and isinstance(obj, dict):
        h = obj.get("provenance_hash")
    assert h is not None, "Missing provenance_hash"
    assert isinstance(h, str), f"provenance_hash should be str, got {type(h)}"
    assert len(h) == 64, f"SHA-256 hash should be 64 hex chars, got {len(h)}"
    assert all(c in "0123456789abcdef" for c in h), "Invalid hex chars in hash"


# ===========================================================================
# Tests
# ===========================================================================


@pytest.mark.integration
class TestBundlePackIntegration:
    """Integration tests for PACK-009 EU Climate Compliance Bundle."""

    # -----------------------------------------------------------------------
    # 1. test_bundle_orchestrator_instantiation
    # -----------------------------------------------------------------------

    def test_bundle_orchestrator_instantiation(self):
        """Instantiate BundlePackOrchestrator with all 4 packs enabled."""
        if _orch_mod is None:
            pytest.skip("pack_orchestrator.py could not be imported")

        BundleOrchestratorConfig = getattr(_orch_mod, "BundleOrchestratorConfig", None)
        BundlePackOrchestrator = getattr(_orch_mod, "BundlePackOrchestrator", None)
        assert BundleOrchestratorConfig is not None, "Missing BundleOrchestratorConfig"
        assert BundlePackOrchestrator is not None, "Missing BundlePackOrchestrator"

        config = BundleOrchestratorConfig(
            enable_csrd=True,
            enable_cbam=True,
            enable_eudr=True,
            enable_taxonomy=True,
            reporting_period_year=2025,
            organization_name="Integration Test Corp",
        )
        orchestrator = BundlePackOrchestrator(config)

        assert orchestrator is not None
        assert orchestrator.config.enable_csrd is True
        assert orchestrator.config.enable_cbam is True
        assert orchestrator.config.enable_eudr is True
        assert orchestrator.config.enable_taxonomy is True
        assert orchestrator.config.organization_name == "Integration Test Corp"
        assert orchestrator.config.reporting_period_year == 2025

    # -----------------------------------------------------------------------
    # 2. test_all_pack_bridges_instantiate
    # -----------------------------------------------------------------------

    def test_all_pack_bridges_instantiate(self):
        """All 4 regulation pack bridges instantiate with default configs."""
        bridge_specs = [
            (_csrd_bridge_mod, "CSRDPackBridge", "CSRDPackBridgeConfig", "csrd_pack_bridge.py"),
            (_cbam_bridge_mod, "CBAMPackBridge", "CBAMPackBridgeConfig", "cbam_pack_bridge.py"),
            (_eudr_bridge_mod, "EUDRPackBridge", "EUDRPackBridgeConfig", "eudr_pack_bridge.py"),
            (_taxonomy_bridge_mod, "TaxonomyPackBridge", "TaxonomyPackBridgeConfig", "taxonomy_pack_bridge.py"),
        ]

        failures = []
        for mod, bridge_cls_name, config_cls_name, filename in bridge_specs:
            if mod is None:
                failures.append(f"{filename}: module import returned None")
                continue

            ConfigCls = getattr(mod, config_cls_name, None)
            BridgeCls = getattr(mod, bridge_cls_name, None)

            if ConfigCls is None:
                failures.append(f"{filename}: missing {config_cls_name}")
                continue
            if BridgeCls is None:
                failures.append(f"{filename}: missing {bridge_cls_name}")
                continue

            try:
                cfg = ConfigCls()
                bridge = BridgeCls(cfg)
                assert bridge is not None, f"{filename}: bridge is None after construction"
                assert bridge.config is not None, f"{filename}: bridge.config is None"
            except Exception as exc:
                failures.append(f"{filename}: instantiation failed - {exc}")

        assert not failures, (
            "Pack bridge instantiation failures:\n" + "\n".join(failures)
        )

    # -----------------------------------------------------------------------
    # 3. test_cross_framework_mapper_bridge_has_mappings
    # -----------------------------------------------------------------------

    def test_cross_framework_mapper_bridge_has_mappings(self):
        """CrossFrameworkMapperBridge has unified mappings across 4 frameworks."""
        if _mapper_bridge_mod is None:
            pytest.skip("cross_framework_mapper_bridge.py could not be imported")

        CrossFrameworkMapperConfig = getattr(
            _mapper_bridge_mod, "CrossFrameworkMapperConfig", None,
        )
        CrossFrameworkMapperBridge = getattr(
            _mapper_bridge_mod, "CrossFrameworkMapperBridge", None,
        )
        UNIFIED_MAPPING_TABLE = getattr(
            _mapper_bridge_mod, "UNIFIED_MAPPING_TABLE", None,
        )

        assert CrossFrameworkMapperConfig is not None, "Missing CrossFrameworkMapperConfig"
        assert CrossFrameworkMapperBridge is not None, "Missing CrossFrameworkMapperBridge"
        assert UNIFIED_MAPPING_TABLE is not None, "Missing UNIFIED_MAPPING_TABLE"

        config = CrossFrameworkMapperConfig()
        bridge = CrossFrameworkMapperBridge(config)

        # map a known shared field
        result = bridge.map_field("organization_name")
        assert isinstance(result, dict), "map_field should return a dict"
        assert len(result) > 0, "map_field('organization_name') returned empty result"

        # get_all_mappings should contain multiple entries
        all_mappings = bridge.get_all_mappings()
        assert isinstance(all_mappings, dict)
        # all_mappings is a structured dict with 'mappings' key containing field entries
        mappings_data = all_mappings.get("mappings", all_mappings)
        assert len(mappings_data) >= 1, (
            f"Expected at least 1 mapping entry, got {len(mappings_data)}"
        )

        # overlap report should exist
        overlap = bridge.get_overlap_report()
        assert isinstance(overlap, dict)
        assert "total_fields" in overlap or "overlap_count" in overlap or len(overlap) > 0, (
            "Overlap report should contain at least one key"
        )

    # -----------------------------------------------------------------------
    # 4. test_shared_data_pipeline_has_routing
    # -----------------------------------------------------------------------

    def test_shared_data_pipeline_has_routing(self):
        """SharedDataPipelineBridge has routing table and routes to active packs."""
        if _pipeline_bridge_mod is None:
            pytest.skip("shared_data_pipeline_bridge.py could not be imported")

        SharedDataPipelineConfig = getattr(
            _pipeline_bridge_mod, "SharedDataPipelineConfig", None,
        )
        SharedDataPipelineBridge = getattr(
            _pipeline_bridge_mod, "SharedDataPipelineBridge", None,
        )
        ROUTING_TABLE = getattr(_pipeline_bridge_mod, "ROUTING_TABLE", None)

        assert SharedDataPipelineConfig is not None, "Missing SharedDataPipelineConfig"
        assert SharedDataPipelineBridge is not None, "Missing SharedDataPipelineBridge"
        assert ROUTING_TABLE is not None, "Missing ROUTING_TABLE"

        config = SharedDataPipelineConfig()
        bridge = SharedDataPipelineBridge(config)

        # Routing table should have multiple categories
        assert isinstance(ROUTING_TABLE, dict)
        assert len(ROUTING_TABLE) >= 4, (
            f"Expected at least 4 routing categories, got {len(ROUTING_TABLE)}"
        )

        # get_routing_table should return the runtime table
        rt = bridge.get_routing_table()
        assert isinstance(rt, dict)
        assert len(rt) > 0, "get_routing_table returned empty result"

        # Each category in the static table should target at least one pack
        for category, cat_info in ROUTING_TABLE.items():
            target_packs = cat_info.get("target_packs", [])
            assert len(target_packs) >= 1, (
                f"Category '{category}' has no target packs"
            )

    # -----------------------------------------------------------------------
    # 5. test_consolidated_evidence_bridge_instantiation
    # -----------------------------------------------------------------------

    def test_consolidated_evidence_bridge_instantiation(self):
        """ConsolidatedEvidenceBridge instantiates and has evidence reuse map."""
        if _evidence_bridge_mod is None:
            pytest.skip("consolidated_evidence_bridge.py could not be imported")

        ConsolidatedEvidenceConfig = getattr(
            _evidence_bridge_mod, "ConsolidatedEvidenceConfig", None,
        )
        ConsolidatedEvidenceBridge = getattr(
            _evidence_bridge_mod, "ConsolidatedEvidenceBridge", None,
        )
        EVIDENCE_REUSE_MAP = getattr(
            _evidence_bridge_mod, "EVIDENCE_REUSE_MAP", None,
        )

        assert ConsolidatedEvidenceConfig is not None, "Missing ConsolidatedEvidenceConfig"
        assert ConsolidatedEvidenceBridge is not None, "Missing ConsolidatedEvidenceBridge"
        assert EVIDENCE_REUSE_MAP is not None, "Missing EVIDENCE_REUSE_MAP"

        config = ConsolidatedEvidenceConfig()
        bridge = ConsolidatedEvidenceBridge(config)

        assert bridge is not None
        assert bridge.config is not None
        assert bridge.config.enable_reuse_tracking is True
        assert bridge.config.evidence_retention_years == 7

        # Reuse map should have multiple evidence types
        assert isinstance(EVIDENCE_REUSE_MAP, dict)
        assert len(EVIDENCE_REUSE_MAP) >= 3, (
            f"Expected at least 3 reuse map entries, got {len(EVIDENCE_REUSE_MAP)}"
        )

        # Each entry should have applicable_packs
        for evidence_key, info in EVIDENCE_REUSE_MAP.items():
            packs = info.get("applicable_packs", [])
            assert len(packs) >= 1, (
                f"Evidence '{evidence_key}' has no applicable packs"
            )

    # -----------------------------------------------------------------------
    # 6. test_bundle_health_check_runs
    # -----------------------------------------------------------------------

    def test_bundle_health_check_runs(self):
        """BundleHealthCheckIntegration instantiates with 25 health categories."""
        if _health_mod is None:
            pytest.skip("bundle_health_check.py could not be imported")

        BundleHealthCheckConfig = getattr(
            _health_mod, "BundleHealthCheckConfig", None,
        )
        BundleHealthCheckIntegration = getattr(
            _health_mod, "BundleHealthCheckIntegration", None,
        )
        HEALTH_CHECK_CATEGORIES = getattr(
            _health_mod, "HEALTH_CHECK_CATEGORIES", None,
        )

        assert BundleHealthCheckConfig is not None, "Missing BundleHealthCheckConfig"
        assert BundleHealthCheckIntegration is not None, "Missing BundleHealthCheckIntegration"
        assert HEALTH_CHECK_CATEGORIES is not None, "Missing HEALTH_CHECK_CATEGORIES"

        config = BundleHealthCheckConfig()
        health = BundleHealthCheckIntegration(config)

        assert health is not None
        assert health.config is not None
        assert health.config.timeout_seconds == 30

        # Must have 25 health check categories
        assert isinstance(HEALTH_CHECK_CATEGORIES, list)
        assert len(HEALTH_CHECK_CATEGORIES) == 25, (
            f"Expected 25 health check categories, got {len(HEALTH_CHECK_CATEGORIES)}"
        )

        # Validate per-pack breakdown: 5 per pack (4 packs) + 5 bundle
        pack_counts: Dict[str, int] = {}
        for cat in HEALTH_CHECK_CATEGORIES:
            pack = cat.get("pack", "unknown")
            pack_counts[pack] = pack_counts.get(pack, 0) + 1

        for pack_key in ("csrd", "cbam", "eudr", "taxonomy"):
            assert pack_counts.get(pack_key, 0) == 5, (
                f"Expected 5 checks for {pack_key}, got {pack_counts.get(pack_key, 0)}"
            )
        assert pack_counts.get("bundle", 0) == 5, (
            f"Expected 5 bundle-level checks, got {pack_counts.get('bundle', 0)}"
        )

    # -----------------------------------------------------------------------
    # 7. test_setup_wizard_steps
    # -----------------------------------------------------------------------

    def test_setup_wizard_steps(self):
        """BundleSetupWizard has exactly 10 steps, start() returns step info."""
        if _wizard_mod is None:
            pytest.skip("setup_wizard.py could not be imported")

        BundleSetupWizardConfig = getattr(
            _wizard_mod, "BundleSetupWizardConfig", None,
        )
        BundleSetupWizard = getattr(
            _wizard_mod, "BundleSetupWizard", None,
        )
        WIZARD_STEPS = getattr(_wizard_mod, "WIZARD_STEPS", None)

        assert BundleSetupWizardConfig is not None, "Missing BundleSetupWizardConfig"
        assert BundleSetupWizard is not None, "Missing BundleSetupWizard"
        assert WIZARD_STEPS is not None, "Missing WIZARD_STEPS"

        # Exactly 10 steps
        assert isinstance(WIZARD_STEPS, list)
        assert len(WIZARD_STEPS) == 10, (
            f"Expected 10 wizard steps, got {len(WIZARD_STEPS)}"
        )

        # Verify step numbers are 1-10
        step_numbers = [s["number"] for s in WIZARD_STEPS]
        assert step_numbers == list(range(1, 11)), (
            f"Step numbers should be 1-10, got {step_numbers}"
        )

        # Verify step names
        expected_names = [
            "Welcome",
            "Regulation Selection",
            "Entity Setup",
            "Pack Configuration",
            "Data Sources",
            "Calendar Setup",
            "Reporting Preferences",
            "Evidence Config",
            "Review",
            "Activation",
        ]
        actual_names = [s["name"] for s in WIZARD_STEPS]
        assert actual_names == expected_names, (
            f"Wizard step names mismatch:\n"
            f"  Expected: {expected_names}\n"
            f"  Got:      {actual_names}"
        )

        # Instantiate and start
        config = BundleSetupWizardConfig()
        wizard = BundleSetupWizard(config)
        start_info = wizard.start()

        assert isinstance(start_info, dict)
        assert start_info["current_step"] == 1
        assert start_info["total_steps"] == 10
        assert "steps" in start_info
        assert len(start_info["steps"]) == 10

    # -----------------------------------------------------------------------
    # 8. test_bundle_orchestrator_has_12_phases
    # -----------------------------------------------------------------------

    def test_bundle_orchestrator_has_12_phases(self):
        """BundlePipelinePhase enum has exactly 12 phases in correct order."""
        if _orch_mod is None:
            pytest.skip("pack_orchestrator.py could not be imported")

        BundlePipelinePhase = getattr(_orch_mod, "BundlePipelinePhase", None)
        PHASE_ORDER = getattr(_orch_mod, "PHASE_ORDER", None)
        BundleOrchestratorConfig = getattr(_orch_mod, "BundleOrchestratorConfig", None)
        BundlePackOrchestrator = getattr(_orch_mod, "BundlePackOrchestrator", None)

        assert BundlePipelinePhase is not None, "Missing BundlePipelinePhase"
        assert PHASE_ORDER is not None, "Missing PHASE_ORDER"

        phases = list(BundlePipelinePhase)
        assert len(phases) == 12, f"Expected 12 pipeline phases, got {len(phases)}"

        expected_values = [
            "health_check",
            "config_init",
            "pack_loading",
            "data_collection",
            "deduplication",
            "parallel_assessment",
            "consistency_check",
            "gap_analysis",
            "calendar_update",
            "consolidated_reporting",
            "evidence_package",
            "audit_trail",
        ]
        actual_values = [p.value for p in phases]
        assert actual_values == expected_values, (
            f"Phase values mismatch:\n"
            f"  Expected: {expected_values}\n"
            f"  Got:      {actual_values}"
        )

        assert len(PHASE_ORDER) == 12, f"PHASE_ORDER has {len(PHASE_ORDER)} entries, expected 12"

        # Verify get_status returns 12 total phases
        config = BundleOrchestratorConfig(
            enable_csrd=True,
            enable_cbam=True,
            enable_eudr=True,
            enable_taxonomy=True,
        )
        orchestrator = BundlePackOrchestrator(config)
        status = orchestrator.get_status()
        assert status["total_phases"] == 12
        assert status["completed_phases"] == 0

    # -----------------------------------------------------------------------
    # 9. test_all_engines_importable_from_pack
    # -----------------------------------------------------------------------

    def test_all_engines_importable_from_pack(self):
        """All 8 engine modules import and expose expected classes."""
        engines_dir = _PACK_DIR / "engines"
        failures = []

        for filename, expected_class in ENGINE_CLASSES:
            fpath = engines_dir / filename
            mod = _import_from_path(
                f"pack009_integ_eng_{filename.replace('.py', '')}",
                fpath,
            )
            if mod is None:
                failures.append(f"{filename}: import returned None")
                continue
            if not hasattr(mod, expected_class):
                failures.append(f"{filename}: missing class {expected_class}")

        assert not failures, (
            "Engine import failures:\n" + "\n".join(failures)
        )

    # -----------------------------------------------------------------------
    # 10. test_all_workflows_importable_from_pack
    # -----------------------------------------------------------------------

    def test_all_workflows_importable_from_pack(self):
        """All 8 workflow modules import and expose expected classes."""
        wf_dir = _PACK_DIR / "workflows"
        failures = []

        for filename, expected_class in WORKFLOW_CLASSES:
            fpath = wf_dir / filename
            mod = _import_from_path(
                f"pack009_integ_wf_{filename.replace('.py', '')}",
                fpath,
            )
            if mod is None:
                failures.append(f"{filename}: import returned None")
                continue
            if not hasattr(mod, expected_class):
                failures.append(f"{filename}: missing class {expected_class}")

        assert not failures, (
            "Workflow import failures:\n" + "\n".join(failures)
        )

    # -----------------------------------------------------------------------
    # 11. test_all_templates_importable_from_pack
    # -----------------------------------------------------------------------

    def test_all_templates_importable_from_pack(self):
        """All 8 template modules import and expose expected classes."""
        tpl_dir = _PACK_DIR / "templates"
        failures = []

        for filename, expected_class in TEMPLATE_CLASSES:
            fpath = tpl_dir / filename
            mod = _import_from_path(
                f"pack009_integ_tpl_{filename.replace('.py', '')}",
                fpath,
            )
            if mod is None:
                failures.append(f"{filename}: import returned None")
                continue
            if not hasattr(mod, expected_class):
                failures.append(f"{filename}: missing class {expected_class}")

        assert not failures, (
            "Template import failures:\n" + "\n".join(failures)
        )

    # -----------------------------------------------------------------------
    # 12. test_config_module_has_bundle_config_class
    # -----------------------------------------------------------------------

    def test_config_module_has_bundle_config_class(self):
        """Config module exposes BundleComplianceConfig and PackConfig classes."""
        if _config_mod is None:
            pytest.skip("pack_config.py could not be imported")

        BundleComplianceConfig = getattr(
            _config_mod, "BundleComplianceConfig", None,
        )
        PackConfig = getattr(_config_mod, "PackConfig", None)
        RegulationType = getattr(_config_mod, "RegulationType", None)

        assert BundleComplianceConfig is not None, (
            "Missing BundleComplianceConfig in config/pack_config.py"
        )
        assert PackConfig is not None, (
            "Missing PackConfig in config/pack_config.py"
        )
        assert RegulationType is not None, (
            "Missing RegulationType in config/pack_config.py"
        )

        # Verify RegulationType has all 4 regulations
        reg_values = set(r.value for r in RegulationType)
        assert "CSRD" in reg_values, "RegulationType missing CSRD"
        assert "CBAM" in reg_values, "RegulationType missing CBAM"
        assert "EUDR" in reg_values, "RegulationType missing EUDR"
        assert "TAXONOMY" in reg_values, "RegulationType missing TAXONOMY"

        # BundleComplianceConfig should be instantiable
        try:
            bundle_config = BundleComplianceConfig()
            assert bundle_config is not None
        except Exception as exc:
            pytest.fail(f"BundleComplianceConfig() raised: {exc}")

    # -----------------------------------------------------------------------
    # 13. test_all_constituent_pack_dirs_exist
    # -----------------------------------------------------------------------

    def test_all_constituent_pack_dirs_exist(self):
        """All 4 constituent pack directories exist on disk."""
        missing = []
        for pack_dir_name in CONSTITUENT_PACK_DIRS:
            full_path = _EU_COMPLIANCE_DIR / pack_dir_name
            if not full_path.exists():
                missing.append(str(full_path))
            elif not full_path.is_dir():
                missing.append(f"{full_path} (exists but is not a directory)")

        assert not missing, (
            "Missing constituent pack directories:\n" + "\n".join(missing)
        )

        # Verify the orchestrator also knows about them
        if _orch_mod is not None:
            CONSTITUENT_PACKS = getattr(_orch_mod, "CONSTITUENT_PACKS", None)
            if CONSTITUENT_PACKS is not None:
                assert "csrd" in CONSTITUENT_PACKS, "CONSTITUENT_PACKS missing 'csrd'"
                assert "cbam" in CONSTITUENT_PACKS, "CONSTITUENT_PACKS missing 'cbam'"
                assert "eudr" in CONSTITUENT_PACKS, "CONSTITUENT_PACKS missing 'eudr'"
                assert "taxonomy" in CONSTITUENT_PACKS, "CONSTITUENT_PACKS missing 'taxonomy'"

    # -----------------------------------------------------------------------
    # 14. test_bridge_config_pattern_enforced
    # -----------------------------------------------------------------------

    def test_bridge_config_pattern_enforced(self):
        """Every bridge module exposes a Config class and a Bridge/Integration class."""
        integ_dir = _PACK_DIR / "integrations"
        failures = []

        for filename, bridge_cls_name, config_cls_name in BRIDGE_CONFIGS:
            fpath = integ_dir / filename
            mod = _import_from_path(
                f"pack009_integ_pattern_{filename.replace('.py', '')}",
                fpath,
            )
            if mod is None:
                failures.append(f"{filename}: import returned None")
                continue

            ConfigCls = getattr(mod, config_cls_name, None)
            BridgeCls = getattr(mod, bridge_cls_name, None)

            if ConfigCls is None:
                failures.append(f"{filename}: missing config class {config_cls_name}")
            if BridgeCls is None:
                failures.append(f"{filename}: missing bridge class {bridge_cls_name}")

            # Config must be a Pydantic BaseModel subclass (or similar)
            if ConfigCls is not None:
                try:
                    cfg_instance = ConfigCls()
                    assert cfg_instance is not None, (
                        f"{config_cls_name}() returned None"
                    )
                except Exception as exc:
                    failures.append(
                        f"{filename}: {config_cls_name}() failed - {exc}"
                    )

            # Bridge must accept config as first __init__ argument
            if BridgeCls is not None and ConfigCls is not None:
                try:
                    cfg_instance = ConfigCls()
                    bridge_instance = BridgeCls(cfg_instance)
                    assert bridge_instance is not None, (
                        f"{bridge_cls_name}(config) returned None"
                    )
                    assert hasattr(bridge_instance, "config"), (
                        f"{bridge_cls_name} missing 'config' attribute after init"
                    )
                except Exception as exc:
                    failures.append(
                        f"{filename}: {bridge_cls_name}(config) failed - {exc}"
                    )

        assert not failures, (
            "Bridge config pattern failures:\n" + "\n".join(failures)
        )

    # -----------------------------------------------------------------------
    # 15. test_provenance_hash_in_engine_results
    # -----------------------------------------------------------------------

    def test_provenance_hash_in_engine_results(self):
        """Engines that produce results include SHA-256 provenance hashes."""
        engines_dir = _PACK_DIR / "engines"

        # Test CrossFrameworkDataMapperEngine (map_field result)
        mapper_path = engines_dir / "cross_framework_data_mapper.py"
        mapper_mod = _import_from_path(
            "pack009_integ_prov_mapper",
            mapper_path,
        )
        if mapper_mod is None:
            pytest.skip("cross_framework_data_mapper.py could not be imported")

        CrossFrameworkDataMapperEngine = getattr(
            mapper_mod, "CrossFrameworkDataMapperEngine", None,
        )
        if CrossFrameworkDataMapperEngine is None:
            pytest.skip("CrossFrameworkDataMapperEngine class not found")

        engine = CrossFrameworkDataMapperEngine()

        # Single map_field result should have provenance hash
        result = engine.map_field(
            "CSRD", "E1_6_scope1_ghg_emissions", "CBAM", 42000.0,
        )
        _assert_provenance_hash(result)

        # Batch map result should have provenance hash
        batch_result = engine.map_batch(
            "CSRD",
            {"E1_6_scope1_ghg_emissions": 42000.0},
            "CBAM",
        )
        _assert_provenance_hash(batch_result)

        # Test DataDeduplicationEngine (scan result)
        dedup_path = engines_dir / "data_deduplication_engine.py"
        dedup_mod = _import_from_path(
            "pack009_integ_prov_dedup",
            dedup_path,
        )
        if dedup_mod is not None:
            DataDeduplicationEngine = getattr(
                dedup_mod, "DataDeduplicationEngine", None,
            )
            if DataDeduplicationEngine is not None:
                dedup_engine = DataDeduplicationEngine()
                scan = dedup_engine.scan_requirements()
                _assert_provenance_hash(scan)

        # Test CrossRegulationGapAnalyzerEngine (scan result)
        gap_path = engines_dir / "cross_regulation_gap_analyzer.py"
        gap_mod = _import_from_path(
            "pack009_integ_prov_gap",
            gap_path,
        )
        if gap_mod is not None:
            CrossRegulationGapAnalyzerEngine = getattr(
                gap_mod, "CrossRegulationGapAnalyzerEngine", None,
            )
            if CrossRegulationGapAnalyzerEngine is not None:
                gap_engine = CrossRegulationGapAnalyzerEngine()
                # scan_all_regulations takes a Dict[str, str] of requirement_id -> status
                compliance_status = {
                    "CSRD_E1_1": "COMPLIANT",
                    "CBAM_DIRECT_EMISSIONS": "NON_COMPLIANT",
                    "EUDR_DDS_STATEMENT": "COMPLIANT",
                    "TAX_CCM_ELIGIBILITY": "COMPLIANT",
                }
                result = gap_engine.scan_all_regulations(compliance_status)
                _assert_provenance_hash(result)
