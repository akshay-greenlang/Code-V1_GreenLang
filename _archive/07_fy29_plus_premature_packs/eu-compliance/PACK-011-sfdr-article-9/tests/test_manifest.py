# -*- coding: utf-8 -*-
"""
PACK-011 SFDR Article 9 Pack - Manifest Tests
===============================================

Tests the pack.yaml manifest for PACK-011 to ensure:
- Pack metadata is correct (ID=PACK-011, version, tier=standalone, category=eu-compliance)
- SFDR regulation references are present (EU 2019/2088, RTS 2022/1288, Benchmarks 2019/2089)
- All 8 SFDR Article 9 engines are listed
- All 8 SFDR Article 9 workflows are listed
- All 8 SFDR Article 9 templates are listed
- All 10 SFDR Article 9 integrations are listed
- All 9 deployment presets are defined (5 product + 4 entity in pack.yaml,
  10 YAML files on disk including extra entity presets)
- Pack summary includes correct agent totals (30 MRV, 10 data, 10 foundation)
- All engine/workflow/template/integration files exist on disk
- All component IDs are unique (no duplicates)
- Version follows semver format
- All preset YAML files exist on disk

Author: GreenLang QA Team
Version: 1.0.0
"""

import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml


# ---------------------------------------------------------------------------
# Inline constants (no conftest imports)
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
PACK_YAML = PACK_ROOT / "pack.yaml"
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"

SFDR_ENGINE_IDS = [
    "sustainable_objective_engine",
    "enhanced_dnsh_engine",
    "full_taxonomy_alignment",
    "impact_measurement_engine",
    "benchmark_alignment_engine",
    "pai_mandatory_engine",
    "carbon_trajectory_engine",
    "investment_universe_engine",
]

SFDR_WORKFLOW_IDS = [
    "annex_iii_disclosure",
    "annex_v_reporting",
    "sustainable_verification",
    "impact_reporting",
    "benchmark_monitoring",
    "pai_mandatory_workflow",
    "downgrade_monitoring",
    "regulatory_update",
]

SFDR_TEMPLATE_IDS = [
    "annex_iii_precontractual",
    "annex_v_periodic",
    "impact_report",
    "benchmark_methodology",
    "sustainable_dashboard",
    "pai_mandatory_report",
    "carbon_trajectory_report",
    "audit_trail_report",
]

SFDR_INTEGRATION_IDS = [
    "pack_orchestrator",
    "article8_pack_bridge",
    "taxonomy_pack_bridge",
    "mrv_emissions_bridge",
    "benchmark_data_bridge",
    "impact_data_bridge",
    "eet_data_bridge",
    "regulatory_bridge",
    "health_check",
    "setup_wizard",
]

# Product presets defined in pack.yaml presets.product_presets
SFDR_PRODUCT_PRESET_IDS = [
    "impact_fund",
    "climate_fund",
    "social_fund",
    "esg_leader_fund",
    "transition_fund",
]

ENGINE_FILES = {
    "sustainable_objective_engine": "sustainable_objective_engine.py",
    "enhanced_dnsh_engine": "enhanced_dnsh_engine.py",
    "full_taxonomy_alignment": "full_taxonomy_alignment.py",
    "impact_measurement_engine": "impact_measurement_engine.py",
    "benchmark_alignment_engine": "benchmark_alignment_engine.py",
    "pai_mandatory_engine": "pai_mandatory_engine.py",
    "carbon_trajectory_engine": "carbon_trajectory_engine.py",
    "investment_universe_engine": "investment_universe_engine.py",
}

WORKFLOW_FILES = {
    "annex_iii_disclosure": "annex_iii_disclosure.py",
    "annex_v_reporting": "annex_v_reporting.py",
    "sustainable_verification": "sustainable_verification.py",
    "impact_reporting": "impact_reporting.py",
    "benchmark_monitoring": "benchmark_monitoring.py",
    "pai_mandatory_workflow": "pai_mandatory_workflow.py",
    "downgrade_monitoring": "downgrade_monitoring.py",
    "regulatory_update": "regulatory_update.py",
}

TEMPLATE_FILES = {
    "annex_iii_precontractual": "annex_iii_precontractual.py",
    "annex_v_periodic": "annex_v_periodic.py",
    "impact_report": "impact_report.py",
    "benchmark_methodology": "benchmark_methodology.py",
    "sustainable_dashboard": "sustainable_dashboard.py",
    "pai_mandatory_report": "pai_mandatory_report.py",
    "carbon_trajectory_report": "carbon_trajectory_report.py",
    "audit_trail_report": "audit_trail_report.py",
}

INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "article8_pack_bridge": "article8_pack_bridge.py",
    "taxonomy_pack_bridge": "taxonomy_pack_bridge.py",
    "mrv_emissions_bridge": "mrv_emissions_bridge.py",
    "benchmark_data_bridge": "benchmark_data_bridge.py",
    "impact_data_bridge": "impact_data_bridge.py",
    "eet_data_bridge": "eet_data_bridge.py",
    "regulatory_bridge": "regulatory_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}


# ---------------------------------------------------------------------------
# Module-scoped fixtures (inline, no conftest)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pack_yaml_path() -> Path:
    """Return the absolute path to pack.yaml."""
    return PACK_YAML


@pytest.fixture(scope="module")
def pack_yaml() -> Dict[str, Any]:
    """Parse and return pack.yaml as a dictionary."""
    if not PACK_YAML.exists():
        pytest.skip("pack.yaml not found")
    with open(PACK_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===========================================================================
# Test class
# ===========================================================================

@pytest.mark.unit
class TestPackManifest:
    """Test suite for PACK-011 pack.yaml manifest."""

    # -----------------------------------------------------------------
    # 1. Existence
    # -----------------------------------------------------------------

    def test_pack_yaml_exists(self, pack_yaml_path: Path):
        """Test that pack.yaml file exists on disk."""
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"

    # -----------------------------------------------------------------
    # 2. Valid YAML
    # -----------------------------------------------------------------

    def test_pack_yaml_valid_yaml(self, pack_yaml_path: Path):
        """Test that pack.yaml is valid YAML that parses to a dictionary."""
        with open(pack_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "pack.yaml must parse to a dictionary"
        assert len(data) > 0, "pack.yaml must not be empty"

    # -----------------------------------------------------------------
    # 3. Metadata fields
    # -----------------------------------------------------------------

    def test_pack_yaml_metadata_fields(self, pack_yaml: Dict[str, Any]):
        """Test metadata contains name, version, display_name, tier, and category."""
        meta = pack_yaml.get("metadata", {})
        assert "name" in meta, "metadata.name missing"
        assert "version" in meta, "metadata.version missing"
        assert "display_name" in meta, "metadata.display_name missing"
        assert "category" in meta, "metadata.category missing"
        assert "tier" in meta, "metadata.tier missing"

        assert "PACK-011" in meta["name"], (
            f"Expected PACK-011 in metadata.name, got {meta['name']}"
        )
        assert meta["display_name"] == "SFDR Article 9 Pack", (
            f"Expected 'SFDR Article 9 Pack', got {meta['display_name']}"
        )
        assert meta["category"] == "eu-compliance", (
            f"Expected eu-compliance category, got {meta['category']}"
        )
        assert meta["tier"] == "standalone", (
            f"Expected standalone tier, got {meta['tier']}"
        )

    # -----------------------------------------------------------------
    # 4. Version format
    # -----------------------------------------------------------------

    def test_pack_yaml_version_format(self, pack_yaml: Dict[str, Any]):
        """Test that version follows semver format X.Y.Z."""
        meta = pack_yaml.get("metadata", {})
        version = meta.get("version", "")
        assert isinstance(version, str), f"version must be string, got {type(version)}"
        assert re.match(r"^\d+\.\d+\.\d+$", version), (
            f"Invalid semver version format: {version}"
        )

    # -----------------------------------------------------------------
    # 5. Regulation references
    # -----------------------------------------------------------------

    def test_pack_yaml_regulation_references(self, pack_yaml: Dict[str, Any]):
        """Test that SFDR regulation references are present including Benchmarks."""
        meta = pack_yaml.get("metadata", {})

        # Check regulation section
        regulation = meta.get("regulation", {})
        assert regulation, "metadata.regulation section missing"
        assert "2019/2088" in regulation.get("reference", ""), (
            "Missing SFDR reference (EU) 2019/2088"
        )

        # Check focus articles include Article 9
        focus_articles = regulation.get("focus_articles", [])
        assert 9 in focus_articles, (
            f"Article 9 must be in focus_articles. Found: {focus_articles}"
        )

        # Check compliance_references for RTS and Benchmarks
        compliance_refs = meta.get("compliance_references", [])
        assert isinstance(compliance_refs, list), "compliance_references must be a list"
        assert len(compliance_refs) >= 4, (
            f"Expected at least 4 compliance references, found {len(compliance_refs)}"
        )

        ref_ids = [r.get("id", "") for r in compliance_refs]
        assert "SFDR-LEVEL1" in ref_ids, (
            f"Missing SFDR-LEVEL1 reference. Found: {ref_ids}"
        )
        assert "SFDR-RTS" in ref_ids, (
            f"Missing SFDR-RTS reference. Found: {ref_ids}"
        )
        assert "BENCHMARKS-REGULATION" in ref_ids, (
            f"Missing BENCHMARKS-REGULATION reference. Found: {ref_ids}"
        )

    # -----------------------------------------------------------------
    # 6. Engines section (8 engines)
    # -----------------------------------------------------------------

    def test_pack_yaml_engines_section(self, pack_yaml: Dict[str, Any]):
        """Test that components.engines contains exactly 8 SFDR Article 9 engines."""
        components = pack_yaml.get("components", {})
        engines = components.get("engines", [])

        assert isinstance(engines, list), "engines must be a list"
        assert len(engines) == 8, (
            f"Expected 8 engines, found {len(engines)}"
        )

        engine_ids = [e.get("id", "") for e in engines]
        for expected_id in SFDR_ENGINE_IDS:
            assert expected_id in engine_ids, (
                f"Missing engine: {expected_id}. Found: {engine_ids}"
            )

    # -----------------------------------------------------------------
    # 7. Workflows section (8 workflows)
    # -----------------------------------------------------------------

    def test_pack_yaml_workflows_section(self, pack_yaml: Dict[str, Any]):
        """Test that components.workflows contains exactly 8 SFDR Article 9 workflows."""
        components = pack_yaml.get("components", {})
        workflows = components.get("workflows", [])

        assert isinstance(workflows, list), "workflows must be a list"
        assert len(workflows) == 8, (
            f"Expected 8 workflows, found {len(workflows)}"
        )

        workflow_ids = [w.get("id", "") for w in workflows]
        for expected_id in SFDR_WORKFLOW_IDS:
            assert expected_id in workflow_ids, (
                f"Missing workflow: {expected_id}. Found: {workflow_ids}"
            )

    # -----------------------------------------------------------------
    # 8. Templates section (8 templates)
    # -----------------------------------------------------------------

    def test_pack_yaml_templates_section(self, pack_yaml: Dict[str, Any]):
        """Test that components.templates contains exactly 8 SFDR Article 9 templates."""
        components = pack_yaml.get("components", {})
        templates = components.get("templates", [])

        assert isinstance(templates, list), "templates must be a list"
        assert len(templates) == 8, (
            f"Expected 8 templates, found {len(templates)}"
        )

        template_ids = [t.get("id", "") for t in templates]
        for expected_id in SFDR_TEMPLATE_IDS:
            assert expected_id in template_ids, (
                f"Missing template: {expected_id}. Found: {template_ids}"
            )

    # -----------------------------------------------------------------
    # 9. Integrations section (10 integrations)
    # -----------------------------------------------------------------

    def test_pack_yaml_integrations_section(self, pack_yaml: Dict[str, Any]):
        """Test that components.integrations contains exactly 10 integrations."""
        components = pack_yaml.get("components", {})
        integrations = components.get("integrations", [])

        assert isinstance(integrations, list), "integrations must be a list"
        assert len(integrations) == 10, (
            f"Expected 10 integrations, found {len(integrations)}"
        )

        integration_ids = [i.get("id", "") for i in integrations]
        for expected_id in SFDR_INTEGRATION_IDS:
            assert expected_id in integration_ids, (
                f"Missing integration: {expected_id}. Found: {integration_ids}"
            )

    # -----------------------------------------------------------------
    # 10. Presets section (5 product presets in pack.yaml)
    # -----------------------------------------------------------------

    def test_pack_yaml_presets_section(self, pack_yaml: Dict[str, Any]):
        """Test that presets.product_presets contains exactly 5 Article 9 product presets."""
        presets = pack_yaml.get("presets", {})
        product_presets = presets.get("product_presets", [])

        assert isinstance(product_presets, list), "product_presets must be a list"
        assert len(product_presets) == 5, (
            f"Expected 5 product presets, found {len(product_presets)}"
        )

        preset_ids = [p.get("id", "") for p in product_presets]
        for expected_id in SFDR_PRODUCT_PRESET_IDS:
            assert expected_id in preset_ids, (
                f"Missing preset: {expected_id}. Found: {preset_ids}"
            )

    # -----------------------------------------------------------------
    # 11. Pack summary
    # -----------------------------------------------------------------

    def test_pack_yaml_pack_summary(self, pack_yaml: Dict[str, Any]):
        """Test pack_summary contains SFDR-specific summary data."""
        summary = pack_yaml.get("pack_summary", {})
        assert isinstance(summary, dict), "pack_summary must be a dictionary"

        assert "total_agents" in summary, "Missing total_agents in summary"
        assert "pack_engines" in summary, "Missing pack_engines in summary"
        assert "pack_workflows" in summary, "Missing pack_workflows"
        assert "pack_templates" in summary, "Missing pack_templates"
        assert "pack_integrations" in summary, "Missing pack_integrations"

        assert summary["pack_engines"] == 8, (
            f"Expected 8 pack engines, got {summary['pack_engines']}"
        )
        assert summary["pack_workflows"] == 8, (
            f"Expected 8 pack workflows, got {summary['pack_workflows']}"
        )
        assert summary["pack_templates"] == 8, (
            f"Expected 8 pack templates, got {summary['pack_templates']}"
        )
        assert summary["pack_integrations"] == 10, (
            f"Expected 10 pack integrations, got {summary['pack_integrations']}"
        )

    # -----------------------------------------------------------------
    # 12. MRV agent count
    # -----------------------------------------------------------------

    def test_pack_yaml_agents_mrv_count(self, pack_yaml: Dict[str, Any]):
        """Test that agents_mrv section declares exactly 30 MRV agents."""
        agents_mrv = pack_yaml.get("components", pack_yaml).get("agents_mrv", {})

        total_mrv = 0
        for scope_key in ("scope_1", "scope_2", "scope_3", "cross_cutting"):
            scope_agents = agents_mrv.get(scope_key, [])
            if isinstance(scope_agents, list):
                total_mrv += len(scope_agents)

        assert total_mrv == 30, (
            f"Expected 30 MRV agents, found {total_mrv}"
        )

        # Also verify pack_summary agrees
        summary = pack_yaml.get("pack_summary", {})
        if "mrv_agents" in summary:
            assert summary["mrv_agents"] == 30, (
                f"pack_summary.mrv_agents expected 30, got {summary['mrv_agents']}"
            )

    # -----------------------------------------------------------------
    # 13. Data agent count
    # -----------------------------------------------------------------

    def test_pack_yaml_agents_data_count(self, pack_yaml: Dict[str, Any]):
        """Test that agents_data section declares exactly 10 data agents."""
        agents_data = pack_yaml.get("components", pack_yaml).get("agents_data", [])
        assert isinstance(agents_data, list), "agents_data must be a list"
        assert len(agents_data) == 10, (
            f"Expected 10 data agents, found {len(agents_data)}"
        )

        summary = pack_yaml.get("pack_summary", {})
        if "data_agents" in summary:
            assert summary["data_agents"] == 10, (
                f"pack_summary.data_agents expected 10, got {summary['data_agents']}"
            )

    # -----------------------------------------------------------------
    # 14. Foundation agent count
    # -----------------------------------------------------------------

    def test_pack_yaml_agents_foundation_count(self, pack_yaml: Dict[str, Any]):
        """Test that agents_foundation section declares exactly 10 foundation agents."""
        agents_found = pack_yaml.get("components", pack_yaml).get(
            "agents_foundation", []
        )
        assert isinstance(agents_found, list), "agents_foundation must be a list"
        assert len(agents_found) == 10, (
            f"Expected 10 foundation agents, found {len(agents_found)}"
        )

        summary = pack_yaml.get("pack_summary", {})
        if "foundation_agents" in summary:
            assert summary["foundation_agents"] == 10, (
                f"pack_summary.foundation_agents expected 10, "
                f"got {summary['foundation_agents']}"
            )

    # -----------------------------------------------------------------
    # 15. Engine files exist
    # -----------------------------------------------------------------

    def test_pack_yaml_all_engine_files_exist(self):
        """Test that all 8 engine Python files exist on disk."""
        for engine_id, filename in ENGINE_FILES.items():
            engine_path = ENGINES_DIR / filename
            assert engine_path.exists(), (
                f"Engine file missing for '{engine_id}': {engine_path}"
            )

    # -----------------------------------------------------------------
    # 16. Workflow files exist
    # -----------------------------------------------------------------

    def test_pack_yaml_all_workflow_files_exist(self):
        """Test that all 8 workflow Python files exist on disk."""
        for workflow_id, filename in WORKFLOW_FILES.items():
            workflow_path = WORKFLOWS_DIR / filename
            assert workflow_path.exists(), (
                f"Workflow file missing for '{workflow_id}': {workflow_path}"
            )

    # -----------------------------------------------------------------
    # 17. Template files exist
    # -----------------------------------------------------------------

    def test_pack_yaml_all_template_files_exist(self):
        """Test that all 8 template Python files exist on disk."""
        for template_id, filename in TEMPLATE_FILES.items():
            template_path = TEMPLATES_DIR / filename
            assert template_path.exists(), (
                f"Template file missing for '{template_id}': {template_path}"
            )

    # -----------------------------------------------------------------
    # 18. Integration files exist
    # -----------------------------------------------------------------

    def test_pack_yaml_all_integration_files_exist(self):
        """Test that all 10 integration Python files exist on disk."""
        for integration_id, filename in INTEGRATION_FILES.items():
            integration_path = INTEGRATIONS_DIR / filename
            assert integration_path.exists(), (
                f"Integration file missing for '{integration_id}': {integration_path}"
            )

    # -----------------------------------------------------------------
    # 19. No duplicate component names
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
    # 20. Preset file existence
    # -----------------------------------------------------------------

    def test_pack_yaml_all_preset_files_exist(self, pack_yaml: Dict[str, Any]):
        """Test that all product preset YAML files referenced in pack.yaml exist on disk."""
        presets = pack_yaml.get("presets", {})
        product_presets = presets.get("product_presets", [])

        for preset in product_presets:
            config_file = preset.get("config_file", "")
            if config_file:
                preset_path = PACK_ROOT / config_file
                assert preset_path.exists(), (
                    f"Preset file missing for '{preset.get('id', '')}': {preset_path}"
                )

    # -----------------------------------------------------------------
    # 21. Total agents count in summary
    # -----------------------------------------------------------------

    def test_pack_yaml_total_agents_count(self, pack_yaml: Dict[str, Any]):
        """Test total_agents in pack_summary equals 50 (30 MRV + 10 data + 10 foundation)."""
        summary = pack_yaml.get("pack_summary", {})
        total = summary.get("total_agents", 0)
        assert total == 50, (
            f"Expected total_agents=50, got {total}"
        )

    # -----------------------------------------------------------------
    # 22. Article 9 specific - Annex III / Annex V templates present
    # -----------------------------------------------------------------

    def test_pack_yaml_article9_annex_templates(self, pack_yaml: Dict[str, Any]):
        """Test that Article 9 specific Annex III and Annex V templates are present."""
        components = pack_yaml.get("components", {})
        template_ids = [t.get("id", "") for t in components.get("templates", [])]

        assert "annex_iii_precontractual" in template_ids, (
            "Missing Annex III pre-contractual template (Article 9 specific)"
        )
        assert "annex_v_periodic" in template_ids, (
            "Missing Annex V periodic report template (Article 9 specific)"
        )

    # -----------------------------------------------------------------
    # 23. Article 9 specific - article8_pack_bridge integration present
    # -----------------------------------------------------------------

    def test_pack_yaml_article8_bridge_present(self, pack_yaml: Dict[str, Any]):
        """Test that the Article 8 Pack Bridge integration is present for downgrade support."""
        components = pack_yaml.get("components", {})
        integration_ids = [i.get("id", "") for i in components.get("integrations", [])]

        assert "article8_pack_bridge" in integration_ids, (
            "Missing article8_pack_bridge integration (required for downgrade pathway)"
        )

    # -----------------------------------------------------------------
    # 24. Article 9 specific - benchmark_data_bridge integration present
    # -----------------------------------------------------------------

    def test_pack_yaml_benchmark_bridge_present(self, pack_yaml: Dict[str, Any]):
        """Test that the Benchmark Data Bridge integration is present for CTB/PAB."""
        components = pack_yaml.get("components", {})
        integration_ids = [i.get("id", "") for i in components.get("integrations", [])]

        assert "benchmark_data_bridge" in integration_ids, (
            "Missing benchmark_data_bridge integration (required for Art.9(2)/9(3))"
        )

    # -----------------------------------------------------------------
    # 25. Entity preset YAML files on disk (asset_manager, etc.)
    # -----------------------------------------------------------------

    def test_entity_preset_files_exist_on_disk(self):
        """Test that entity-level preset YAML files exist on disk."""
        entity_presets = [
            "asset_manager", "insurance", "bank", "pension_fund", "wealth_manager",
        ]
        presets_dir = PACK_ROOT / "config" / "presets"
        for preset_name in entity_presets:
            preset_path = presets_dir / f"{preset_name}.yaml"
            assert preset_path.exists(), (
                f"Entity preset file missing: {preset_path}"
            )
