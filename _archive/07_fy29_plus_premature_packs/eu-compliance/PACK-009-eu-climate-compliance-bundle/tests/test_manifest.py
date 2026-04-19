# -*- coding: utf-8 -*-
"""
PACK-009 EU Climate Compliance Bundle Pack - Manifest Tests
=============================================================

Tests the pack.yaml manifest for PACK-009 to ensure:
- Pack metadata is correct (ID, version, tier=bundle, category)
- All 4 constituent pack dependencies are listed
- All 4 EU regulation references are present (CSRD, CBAM, EUDR, EU Taxonomy)
- All 8 bundle engines are listed
- All 8 bundle workflows are listed
- All 8 bundle templates are listed
- All 10 bundle integrations are listed
- All 4 bundle presets are defined
- All engine/workflow/template/integration files exist on disk
- Pack summary includes correct constituent pack count and agent totals
- All component IDs are unique (no duplicates)
- Version follows semver format
- Inherited agent counts match expected values

Author: GreenLang QA Team
Version: 1.0.0
"""

import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

_PACK_009_DIR = Path(__file__).resolve().parent.parent
_PACK_YAML = _PACK_009_DIR / "pack.yaml"

PACK_ROOT = _PACK_009_DIR
PACK_YAML_PATH = _PACK_YAML
ENGINES_DIR = _PACK_009_DIR / "engines"
WORKFLOWS_DIR = _PACK_009_DIR / "workflows"
TEMPLATES_DIR = _PACK_009_DIR / "templates"
INTEGRATIONS_DIR = _PACK_009_DIR / "integrations"

REGULATIONS = ["CSRD", "CBAM", "EUDR", "TAXONOMY"]
CONSTITUENT_PACKS = {
    "PACK-001": "csrd-starter",
    "PACK-004": "cbam-readiness",
    "PACK-006": "eudr-starter",
    "PACK-008": "eu-taxonomy-alignment",
}
BUNDLE_ENGINE_IDS = [
    "cross_framework_data_mapper", "data_deduplication",
    "cross_regulation_gap_analyzer", "regulatory_calendar",
    "consolidated_metrics", "multi_regulation_consistency",
    "bundle_compliance_scoring", "cross_regulation_evidence",
]
BUNDLE_WORKFLOW_IDS = [
    "unified_data_collection", "cross_regulation_assessment",
    "consolidated_reporting", "calendar_management",
    "cross_framework_gap_analysis", "bundle_health_check",
    "data_consistency_reconciliation", "annual_compliance_review",
]
BUNDLE_TEMPLATE_IDS = [
    "consolidated_dashboard", "cross_regulation_data_map",
    "unified_gap_analysis", "regulatory_calendar_report",
    "data_consistency_report", "bundle_executive_summary",
    "deduplication_savings", "multi_regulation_audit_trail",
]
BUNDLE_INTEGRATION_IDS = [
    "bundle_orchestrator", "csrd_pack_bridge", "cbam_pack_bridge",
    "eudr_pack_bridge", "taxonomy_pack_bridge",
    "cross_framework_mapper_bridge", "shared_data_pipeline_bridge",
    "consolidated_evidence_bridge", "bundle_health_check_integration",
    "setup_wizard",
]
BUNDLE_PRESET_IDS = ["enterprise_full", "financial_institution", "eu_importer", "sme_essential"]
ENGINE_FILES = {
    "cross_framework_data_mapper": "cross_framework_data_mapper.py",
    "data_deduplication": "data_deduplication_engine.py",
    "cross_regulation_gap_analyzer": "cross_regulation_gap_analyzer.py",
    "regulatory_calendar": "regulatory_calendar_engine.py",
    "consolidated_metrics": "consolidated_metrics_engine.py",
    "multi_regulation_consistency": "multi_regulation_consistency_engine.py",
    "bundle_compliance_scoring": "bundle_compliance_scoring_engine.py",
    "cross_regulation_evidence": "cross_regulation_evidence_engine.py",
}
WORKFLOW_FILES = {
    "unified_data_collection": "unified_data_collection.py",
    "cross_regulation_assessment": "cross_regulation_assessment.py",
    "consolidated_reporting": "consolidated_reporting.py",
    "calendar_management": "calendar_management.py",
    "cross_framework_gap_analysis": "cross_framework_gap_analysis.py",
    "bundle_health_check": "bundle_health_check.py",
    "data_consistency_reconciliation": "data_consistency_reconciliation.py",
    "annual_compliance_review": "annual_compliance_review.py",
}
TEMPLATE_FILES = {
    "consolidated_dashboard": "consolidated_dashboard.py",
    "cross_regulation_data_map": "cross_regulation_data_map.py",
    "unified_gap_analysis": "unified_gap_analysis_report.py",
    "regulatory_calendar_report": "regulatory_calendar_report.py",
    "data_consistency_report": "data_consistency_report.py",
    "bundle_executive_summary": "bundle_executive_summary.py",
    "deduplication_savings": "deduplication_savings_report.py",
    "multi_regulation_audit_trail": "multi_regulation_audit_trail.py",
}
INTEGRATION_FILES = {
    "bundle_orchestrator": "pack_orchestrator.py",
    "csrd_pack_bridge": "csrd_pack_bridge.py",
    "cbam_pack_bridge": "cbam_pack_bridge.py",
    "eudr_pack_bridge": "eudr_pack_bridge.py",
    "taxonomy_pack_bridge": "taxonomy_pack_bridge.py",
    "cross_framework_mapper_bridge": "cross_framework_mapper_bridge.py",
    "shared_data_pipeline_bridge": "shared_data_pipeline_bridge.py",
    "consolidated_evidence_bridge": "consolidated_evidence_bridge.py",
    "bundle_health_check_integration": "bundle_health_check.py",
    "setup_wizard": "setup_wizard.py",
}


@pytest.fixture(scope="module")
def pack_yaml_path() -> Path:
    return _PACK_YAML


@pytest.fixture(scope="module")
def pack_yaml() -> Dict[str, Any]:
    if not _PACK_YAML.exists():
        pytest.skip("pack.yaml not found")
    with open(_PACK_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===========================================================================
# Test class
# ===========================================================================

@pytest.mark.unit
class TestPackManifest:
    """Test suite for PACK-009 pack.yaml manifest."""

    # -----------------------------------------------------------------
    # Existence and validity
    # -----------------------------------------------------------------

    def test_pack_yaml_exists(self, pack_yaml_path: Path):
        """Test that pack.yaml file exists on disk."""
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"

    def test_pack_yaml_valid_yaml(self, pack_yaml_path: Path):
        """Test that pack.yaml is valid YAML that parses to a dictionary."""
        with open(pack_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "pack.yaml must parse to a dictionary"
        assert len(data) > 0, "pack.yaml must not be empty"

    # -----------------------------------------------------------------
    # Metadata fields
    # -----------------------------------------------------------------

    def test_pack_yaml_metadata_fields(self, pack_yaml: Dict[str, Any]):
        """Test that metadata contains name, version, display_name, tier, and category."""
        meta = pack_yaml.get("metadata", {})
        assert "name" in meta, "metadata.name missing"
        assert "version" in meta, "metadata.version missing"
        assert "display_name" in meta, "metadata.display_name missing"
        assert "category" in meta, "metadata.category missing"
        assert "tier" in meta, "metadata.tier missing"

        assert "PACK-009" in meta["name"], (
            f"Expected PACK-009 in metadata.name, got {meta['name']}"
        )
        assert meta["display_name"] == "EU Climate Compliance Bundle Pack", (
            f"Expected 'EU Climate Compliance Bundle Pack', got {meta['display_name']}"
        )
        assert meta["category"] == "eu-compliance", (
            f"Expected eu-compliance category, got {meta['category']}"
        )
        assert meta["tier"] == "bundle", (
            f"Expected bundle tier, got {meta['tier']}"
        )

    def test_pack_yaml_version_format(self, pack_yaml: Dict[str, Any]):
        """Test that version follows semver format X.Y.Z."""
        meta = pack_yaml.get("metadata", {})
        version = meta.get("version", "")
        assert isinstance(version, str), f"version must be string, got {type(version)}"
        assert re.match(r"^\d+\.\d+\.\d+$", version), (
            f"Invalid semver version format: {version}"
        )

    # -----------------------------------------------------------------
    # Dependencies (4 constituent packs)
    # -----------------------------------------------------------------

    def test_pack_yaml_dependencies(self, pack_yaml: Dict[str, Any]):
        """Test that dependencies list all 4 constituent packs."""
        meta = pack_yaml.get("metadata", {})
        deps = meta.get("dependencies", [])

        assert isinstance(deps, list), "dependencies must be a list"
        assert len(deps) == 4, f"Expected 4 dependencies, found {len(deps)}"

        dep_ids = [d.get("id", "") for d in deps]

        expected_packs = [
            "PACK-001-csrd-starter",
            "PACK-004-cbam-readiness",
            "PACK-006-eudr-starter",
            "PACK-008-eu-taxonomy-alignment",
        ]
        for pack_id in expected_packs:
            assert pack_id in dep_ids, (
                f"Missing dependency: {pack_id}. Found: {dep_ids}"
            )

    # -----------------------------------------------------------------
    # Regulation references
    # -----------------------------------------------------------------

    def test_pack_yaml_regulation_references(self, pack_yaml: Dict[str, Any]):
        """Test that regulations section references CSRD, CBAM, EUDR, and EU Taxonomy."""
        meta = pack_yaml.get("metadata", {})
        regulations = meta.get("regulations", [])

        assert isinstance(regulations, list), "regulations must be a list"
        assert len(regulations) >= 4, (
            f"Expected at least 4 regulations, found {len(regulations)}"
        )

        reg_ids = [r.get("id", "") for r in regulations]
        for expected_reg in REGULATIONS:
            # TAXONOMY is listed as EU-TAXONOMY in pack.yaml
            expected_id = expected_reg if expected_reg != "TAXONOMY" else "EU-TAXONOMY"
            assert expected_id in reg_ids, (
                f"Missing regulation reference: {expected_id}. Found: {reg_ids}"
            )

    # -----------------------------------------------------------------
    # Engines section (8 engines)
    # -----------------------------------------------------------------

    def test_pack_yaml_engines_section(self, pack_yaml: Dict[str, Any]):
        """Test that components.engines contains exactly 8 bundle engines."""
        components = pack_yaml.get("components", {})
        engines = components.get("engines", [])

        assert isinstance(engines, list), "engines must be a list"
        assert len(engines) == 8, (
            f"Expected 8 engines, found {len(engines)}"
        )

        engine_ids = [e.get("id", "") for e in engines]
        for expected_id in BUNDLE_ENGINE_IDS:
            assert expected_id in engine_ids, (
                f"Missing engine: {expected_id}. Found: {engine_ids}"
            )

    # -----------------------------------------------------------------
    # Workflows section (8 workflows)
    # -----------------------------------------------------------------

    def test_pack_yaml_workflows_section(self, pack_yaml: Dict[str, Any]):
        """Test that components.workflows contains exactly 8 bundle workflows."""
        components = pack_yaml.get("components", {})
        workflows = components.get("workflows", [])

        assert isinstance(workflows, list), "workflows must be a list"
        assert len(workflows) == 8, (
            f"Expected 8 workflows, found {len(workflows)}"
        )

        workflow_ids = [w.get("id", "") for w in workflows]
        for expected_id in BUNDLE_WORKFLOW_IDS:
            assert expected_id in workflow_ids, (
                f"Missing workflow: {expected_id}. Found: {workflow_ids}"
            )

    # -----------------------------------------------------------------
    # Templates section (8 templates)
    # -----------------------------------------------------------------

    def test_pack_yaml_templates_section(self, pack_yaml: Dict[str, Any]):
        """Test that components.templates contains exactly 8 bundle templates."""
        components = pack_yaml.get("components", {})
        templates = components.get("templates", [])

        assert isinstance(templates, list), "templates must be a list"
        assert len(templates) == 8, (
            f"Expected 8 templates, found {len(templates)}"
        )

        template_ids = [t.get("id", "") for t in templates]
        for expected_id in BUNDLE_TEMPLATE_IDS:
            assert expected_id in template_ids, (
                f"Missing template: {expected_id}. Found: {template_ids}"
            )

    # -----------------------------------------------------------------
    # Integrations section (10 integrations)
    # -----------------------------------------------------------------

    def test_pack_yaml_integrations_section(self, pack_yaml: Dict[str, Any]):
        """Test that components.integrations contains exactly 10 integrations."""
        components = pack_yaml.get("components", {})
        integrations = components.get("integrations", [])

        assert isinstance(integrations, list), "integrations must be a list"
        assert len(integrations) == 10, (
            f"Expected 10 integrations, found {len(integrations)}"
        )

    # -----------------------------------------------------------------
    # Presets section (4 presets)
    # -----------------------------------------------------------------

    def test_pack_yaml_presets_section(self, pack_yaml: Dict[str, Any]):
        """Test that presets.bundle_presets contains exactly 4 deployment presets."""
        presets = pack_yaml.get("presets", {})
        bundle_presets = presets.get("bundle_presets", [])

        assert isinstance(bundle_presets, list), "bundle_presets must be a list"
        assert len(bundle_presets) == 4, (
            f"Expected 4 bundle presets, found {len(bundle_presets)}"
        )

        preset_ids = [p.get("id", "") for p in bundle_presets]
        for expected_id in BUNDLE_PRESET_IDS:
            assert expected_id in preset_ids, (
                f"Missing preset: {expected_id}. Found: {preset_ids}"
            )

    # -----------------------------------------------------------------
    # Pack summary (bundle-specific)
    # -----------------------------------------------------------------

    def test_pack_yaml_pack_summary(self, pack_yaml: Dict[str, Any]):
        """Test pack_summary contains bundle-specific summary data."""
        summary = pack_yaml.get("pack_summary", {})
        assert isinstance(summary, dict), "pack_summary must be a dictionary"

        assert "constituent_packs" in summary, "Missing constituent_packs in summary"
        assert "total_inherited_agents" in summary, "Missing total_inherited_agents"
        assert "bundle_engines" in summary, "Missing bundle_engines in summary"
        assert "bundle_workflows" in summary, "Missing bundle_workflows"
        assert "bundle_templates" in summary, "Missing bundle_templates"
        assert "bundle_integrations" in summary, "Missing bundle_integrations"

        assert summary["bundle_engines"] == 8, (
            f"Expected 8 bundle engines, got {summary['bundle_engines']}"
        )
        assert summary["bundle_workflows"] == 8, (
            f"Expected 8 bundle workflows, got {summary['bundle_workflows']}"
        )
        assert summary["bundle_templates"] == 8, (
            f"Expected 8 bundle templates, got {summary['bundle_templates']}"
        )
        assert summary["bundle_integrations"] == 10, (
            f"Expected 10 bundle integrations, got {summary['bundle_integrations']}"
        )

    # -----------------------------------------------------------------
    # Constituent packs count
    # -----------------------------------------------------------------

    def test_pack_yaml_constituent_packs_count(self, pack_yaml: Dict[str, Any]):
        """Test pack_summary reports exactly 4 constituent packs."""
        summary = pack_yaml.get("pack_summary", {})
        count = summary.get("constituent_packs", 0)
        assert count == 4, f"Expected 4 constituent packs, got {count}"

    # -----------------------------------------------------------------
    # Inherited agent count
    # -----------------------------------------------------------------

    def test_pack_yaml_inherited_agent_count(self, pack_yaml: Dict[str, Any]):
        """Test total_inherited_agents matches sum of per-pack agent counts."""
        summary = pack_yaml.get("pack_summary", {})
        total_inherited = summary.get("total_inherited_agents", 0)

        # Expected sum: CSRD(51) + CBAM(47) + EUDR(59) + Taxonomy(51) = 208
        assert total_inherited == 208, (
            f"Expected 208 total inherited agents, got {total_inherited}"
        )

        # Verify per-pack breakdown if available
        agents_by_pack = summary.get("agents_by_pack", {})
        if agents_by_pack:
            per_pack_total = sum(agents_by_pack.values())
            assert per_pack_total == total_inherited, (
                f"Per-pack sum ({per_pack_total}) does not match "
                f"total_inherited_agents ({total_inherited})"
            )

    # -----------------------------------------------------------------
    # File existence checks - Engines
    # -----------------------------------------------------------------

    def test_pack_yaml_all_engine_files_exist(self):
        """Test that all 8 engine Python files exist on disk."""
        for engine_id, filename in ENGINE_FILES.items():
            engine_path = ENGINES_DIR / filename
            assert engine_path.exists(), (
                f"Engine file missing for '{engine_id}': {engine_path}"
            )

    # -----------------------------------------------------------------
    # File existence checks - Workflows
    # -----------------------------------------------------------------

    def test_pack_yaml_all_workflow_files_exist(self):
        """Test that all 8 workflow Python files exist on disk."""
        for workflow_id, filename in WORKFLOW_FILES.items():
            workflow_path = WORKFLOWS_DIR / filename
            assert workflow_path.exists(), (
                f"Workflow file missing for '{workflow_id}': {workflow_path}"
            )

    # -----------------------------------------------------------------
    # File existence checks - Templates
    # -----------------------------------------------------------------

    def test_pack_yaml_all_template_files_exist(self):
        """Test that all 8 template Python files exist on disk."""
        for template_id, filename in TEMPLATE_FILES.items():
            template_path = TEMPLATES_DIR / filename
            assert template_path.exists(), (
                f"Template file missing for '{template_id}': {template_path}"
            )

    # -----------------------------------------------------------------
    # File existence checks - Integrations
    # -----------------------------------------------------------------

    def test_pack_yaml_all_integration_files_exist(self):
        """Test that all 10 integration Python files exist on disk."""
        for integration_id, filename in INTEGRATION_FILES.items():
            integration_path = INTEGRATIONS_DIR / filename
            assert integration_path.exists(), (
                f"Integration file missing for '{integration_id}': {integration_path}"
            )

    # -----------------------------------------------------------------
    # No duplicate component names
    # -----------------------------------------------------------------

    def test_pack_yaml_no_duplicate_component_names(self, pack_yaml: Dict[str, Any]):
        """Test that all component IDs across engines, workflows, templates, integrations are unique."""
        components = pack_yaml.get("components", {})

        all_ids: List[str] = []

        for engine in components.get("engines", []):
            all_ids.append(f"engine:{engine.get('id', '')}")

        for workflow in components.get("workflows", []):
            all_ids.append(f"workflow:{workflow.get('id', '')}")

        for template in components.get("templates", []):
            all_ids.append(f"template:{template.get('id', '')}")

        for integration in components.get("integrations", []):
            all_ids.append(f"integration:{integration.get('id', '')}")

        duplicates = [cid for cid in all_ids if all_ids.count(cid) > 1]
        assert len(duplicates) == 0, (
            f"Duplicate component IDs found: {set(duplicates)}"
        )

    # -----------------------------------------------------------------
    # Preset file existence
    # -----------------------------------------------------------------

    def test_pack_yaml_all_preset_files_exist(self, pack_yaml: Dict[str, Any]):
        """Test that all 4 bundle preset YAML files exist on disk."""
        presets = pack_yaml.get("presets", {})
        bundle_presets = presets.get("bundle_presets", [])

        for preset in bundle_presets:
            config_file = preset.get("config_file", "")
            if config_file:
                preset_path = _PACK_009_DIR / config_file
                assert preset_path.exists(), (
                    f"Preset file missing for '{preset.get('id', '')}': {preset_path}"
                )
