# -*- coding: utf-8 -*-
"""
PACK-012 CSRD Financial Service Pack - Manifest Tests
======================================================

Tests the pack.yaml manifest for PACK-012 to ensure:
- Pack metadata is correct (ID=PACK-012, version, tier=sector-specific, category=eu-compliance)
- 8 engines, 8 workflows, 8 templates, 10 integrations present
- 6 presets listed (bank, insurance, asset_manager, investment_firm, pension_fund, conglomerate)
- All preset files exist on disk
- 72 agent dependencies (30 MRV + 20 data + 10 foundation + 9 FIN + 2 climate + 1 policy)
- Regulation references (CSRD, ESRS, EU Taxonomy, CRR, SFDR, Solvency II, PCAF, SBTi FI, EBA ITS)
- All component files exist on disk
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

FS_ENGINE_IDS = [
    "financed_emissions",
    "insurance_underwriting",
    "green_asset_ratio",
    "btar_calculator",
    "climate_risk_scoring",
    "fs_double_materiality",
    "fs_transition_plan",
    "pillar3_esg",
]

FS_WORKFLOW_IDS = [
    "financed_emissions_workflow",
    "gar_btar_workflow",
    "insurance_emissions_workflow",
    "climate_stress_test",
    "fs_materiality",
    "transition_plan",
    "pillar3_reporting",
    "regulatory_integration",
]

FS_TEMPLATE_IDS = [
    "pcaf_report",
    "gar_btar_report",
    "pillar3_esg",
    "climate_risk_report",
    "fs_esrs_chapter",
    "financed_emissions_dashboard",
    "insurance_esg",
    "sbti_fi_report",
]

FS_INTEGRATION_IDS = [
    "pack_orchestrator",
    "csrd_pack_bridge",
    "sfdr_pack_bridge",
    "taxonomy_pack_bridge",
    "mrv_investments_bridge",
    "finance_agent_bridge",
    "climate_risk_bridge",
    "eba_pillar3_bridge",
    "health_check",
    "setup_wizard",
]

FS_PRESET_IDS = [
    "bank", "insurance", "asset_manager",
    "investment_firm", "pension_fund", "conglomerate",
]

ENGINE_FILES = {
    "financed_emissions": "financed_emissions_engine.py",
    "insurance_underwriting": "insurance_underwriting_engine.py",
    "green_asset_ratio": "green_asset_ratio_engine.py",
    "btar_calculator": "btar_calculator_engine.py",
    "climate_risk_scoring": "climate_risk_scoring_engine.py",
    "fs_double_materiality": "fs_double_materiality_engine.py",
    "fs_transition_plan": "fs_transition_plan_engine.py",
    "pillar3_esg": "pillar3_esg_engine.py",
}

WORKFLOW_FILES = {
    "financed_emissions_workflow": "financed_emissions_workflow.py",
    "gar_btar_workflow": "gar_btar_workflow.py",
    "insurance_emissions_workflow": "insurance_emissions_workflow.py",
    "climate_stress_test": "climate_stress_test_workflow.py",
    "fs_materiality": "fs_materiality_workflow.py",
    "transition_plan": "transition_plan_workflow.py",
    "pillar3_reporting": "pillar3_reporting_workflow.py",
    "regulatory_integration": "regulatory_integration_workflow.py",
}

TEMPLATE_FILES = {
    "pcaf_report": "pcaf_report.py",
    "gar_btar_report": "gar_btar_report.py",
    "pillar3_esg": "pillar3_esg_template.py",
    "climate_risk_report": "climate_risk_report.py",
    "fs_esrs_chapter": "fs_esrs_chapter.py",
    "financed_emissions_dashboard": "financed_emissions_dashboard.py",
    "insurance_esg": "insurance_esg_template.py",
    "sbti_fi_report": "sbti_fi_report.py",
}

INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "csrd_pack_bridge": "csrd_pack_bridge.py",
    "sfdr_pack_bridge": "sfdr_pack_bridge.py",
    "taxonomy_pack_bridge": "taxonomy_pack_bridge.py",
    "mrv_investments_bridge": "mrv_investments_bridge.py",
    "finance_agent_bridge": "finance_agent_bridge.py",
    "climate_risk_bridge": "climate_risk_bridge.py",
    "eba_pillar3_bridge": "eba_pillar3_bridge.py",
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
    """Test suite for PACK-012 pack.yaml manifest."""

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

        assert "PACK-012" in meta["name"], (
            f"Expected PACK-012 in metadata.name, got {meta['name']}"
        )
        assert meta["display_name"] == "CSRD Financial Service Pack", (
            f"Expected 'CSRD Financial Service Pack', got {meta['display_name']}"
        )
        assert meta["category"] == "eu-compliance", (
            f"Expected eu-compliance category, got {meta['category']}"
        )
        assert meta["tier"] == "sector-specific", (
            f"Expected sector-specific tier, got {meta['tier']}"
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
    # 5. Pack ID is PACK-012
    # -----------------------------------------------------------------

    def test_pack_id(self, pack_yaml: Dict[str, Any]):
        """Test that the pack ID contains PACK-012."""
        meta = pack_yaml.get("metadata", {})
        name = meta.get("name", "")
        assert "PACK-012" in name, f"Expected PACK-012 in name, got {name}"
        assert "csrd-financial-service" in name, (
            f"Expected csrd-financial-service in name, got {name}"
        )

    # -----------------------------------------------------------------
    # 6. Regulation references
    # -----------------------------------------------------------------

    def test_pack_yaml_regulation_references(self, pack_yaml: Dict[str, Any]):
        """Test that regulation references are present including CSRD, CRR, PCAF."""
        meta = pack_yaml.get("metadata", {})

        # Check primary regulation
        regulation = meta.get("regulation", {})
        assert regulation, "metadata.regulation section missing"
        primary = regulation.get("primary", {})
        assert "2022/2464" in primary.get("reference", ""), (
            "Missing CSRD reference (EU) 2022/2464"
        )

        # Check secondary regulations
        secondary = regulation.get("secondary", [])
        assert isinstance(secondary, list), "secondary regulations must be a list"
        assert len(secondary) >= 7, (
            f"Expected at least 7 secondary regulations, found {len(secondary)}"
        )

        # Check compliance_references
        compliance_refs = meta.get("compliance_references", [])
        assert isinstance(compliance_refs, list), "compliance_references must be a list"
        assert len(compliance_refs) >= 8, (
            f"Expected at least 8 compliance references, found {len(compliance_refs)}"
        )

        ref_ids = [r.get("id", "") for r in compliance_refs]
        for required_id in ["CSRD", "ESRS", "EU_TAXONOMY", "CRR_CRD_VI", "PCAF"]:
            assert required_id in ref_ids, (
                f"Missing {required_id} reference. Found: {ref_ids}"
            )

    # -----------------------------------------------------------------
    # 7. Engines section (8 engines)
    # -----------------------------------------------------------------

    def test_pack_yaml_engines_section(self, pack_yaml: Dict[str, Any]):
        """Test that components.engines contains exactly 8 FS engines."""
        components = pack_yaml.get("components", {})
        engines = components.get("engines", [])

        assert isinstance(engines, list), "engines must be a list"
        assert len(engines) == 8, (
            f"Expected 8 engines, found {len(engines)}"
        )

        engine_ids = [e.get("id", "") for e in engines]
        for expected_id in FS_ENGINE_IDS:
            assert expected_id in engine_ids, (
                f"Missing engine: {expected_id}. Found: {engine_ids}"
            )

    # -----------------------------------------------------------------
    # 8. Workflows section (8 workflows)
    # -----------------------------------------------------------------

    def test_pack_yaml_workflows_section(self, pack_yaml: Dict[str, Any]):
        """Test that components.workflows contains exactly 8 FS workflows."""
        components = pack_yaml.get("components", {})
        workflows = components.get("workflows", [])

        assert isinstance(workflows, list), "workflows must be a list"
        assert len(workflows) == 8, (
            f"Expected 8 workflows, found {len(workflows)}"
        )

        workflow_ids = [w.get("id", "") for w in workflows]
        for expected_id in FS_WORKFLOW_IDS:
            assert expected_id in workflow_ids, (
                f"Missing workflow: {expected_id}. Found: {workflow_ids}"
            )

    # -----------------------------------------------------------------
    # 9. Templates section (8 templates)
    # -----------------------------------------------------------------

    def test_pack_yaml_templates_section(self, pack_yaml: Dict[str, Any]):
        """Test that components.templates contains exactly 8 FS templates."""
        components = pack_yaml.get("components", {})
        templates = components.get("templates", [])

        assert isinstance(templates, list), "templates must be a list"
        assert len(templates) == 8, (
            f"Expected 8 templates, found {len(templates)}"
        )

        template_ids = [t.get("id", "") for t in templates]
        for expected_id in FS_TEMPLATE_IDS:
            assert expected_id in template_ids, (
                f"Missing template: {expected_id}. Found: {template_ids}"
            )

    # -----------------------------------------------------------------
    # 10. Integrations section (10 integrations)
    # -----------------------------------------------------------------

    def test_pack_yaml_integrations_section(self, pack_yaml: Dict[str, Any]):
        """Test that components.integrations contains exactly 10 FS integrations."""
        components = pack_yaml.get("components", {})
        integrations = components.get("integrations", [])

        assert isinstance(integrations, list), "integrations must be a list"
        assert len(integrations) == 10, (
            f"Expected 10 integrations, found {len(integrations)}"
        )

        integration_ids = [i.get("id", "") for i in integrations]
        for expected_id in FS_INTEGRATION_IDS:
            assert expected_id in integration_ids, (
                f"Missing integration: {expected_id}. Found: {integration_ids}"
            )

    # -----------------------------------------------------------------
    # 11. Presets section (6 presets)
    # -----------------------------------------------------------------

    def test_pack_yaml_presets_section(self, pack_yaml: Dict[str, Any]):
        """Test that presets section contains exactly 6 FS presets."""
        components = pack_yaml.get("components", {})
        presets = components.get("presets", [])
        assert isinstance(presets, list), "presets must be a list"
        assert len(presets) == 6, (
            f"Expected 6 presets, found {len(presets)}"
        )

        preset_ids = [p.get("id", "") for p in presets]
        for expected_id in FS_PRESET_IDS:
            assert expected_id in preset_ids, (
                f"Missing preset: {expected_id}. Found: {preset_ids}"
            )

    # -----------------------------------------------------------------
    # 12. Preset files exist on disk
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("preset_name", FS_PRESET_IDS)
    def test_preset_files_exist(self, preset_name: str):
        """Test that each preset YAML file exists on disk."""
        preset_path = PACK_ROOT / "config" / "presets" / f"{preset_name}.yaml"
        assert preset_path.is_file(), (
            f"Preset file not found: {preset_path}"
        )

    # -----------------------------------------------------------------
    # 13. Preset files are valid YAML
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("preset_name", FS_PRESET_IDS)
    def test_preset_files_parseable(self, preset_name: str):
        """Test that each preset YAML file is valid YAML."""
        preset_path = PACK_ROOT / "config" / "presets" / f"{preset_name}.yaml"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), (
            f"Preset {preset_name} must parse to a dict"
        )

    # -----------------------------------------------------------------
    # 14. Engine files exist on disk
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("engine_id,filename", list(ENGINE_FILES.items()))
    def test_engine_files_exist(self, engine_id: str, filename: str):
        """Test that each engine Python file exists on disk."""
        engine_path = ENGINES_DIR / filename
        assert engine_path.is_file(), (
            f"Engine file not found: {engine_path} (engine: {engine_id})"
        )

    # -----------------------------------------------------------------
    # 15. Workflow files exist on disk
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("workflow_id,filename", list(WORKFLOW_FILES.items()))
    def test_workflow_files_exist(self, workflow_id: str, filename: str):
        """Test that each workflow Python file exists on disk."""
        workflow_path = WORKFLOWS_DIR / filename
        assert workflow_path.is_file(), (
            f"Workflow file not found: {workflow_path} (workflow: {workflow_id})"
        )

    # -----------------------------------------------------------------
    # 16. Template files exist on disk
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("template_id,filename", list(TEMPLATE_FILES.items()))
    def test_template_files_exist(self, template_id: str, filename: str):
        """Test that each template Python file exists on disk."""
        template_path = TEMPLATES_DIR / filename
        assert template_path.is_file(), (
            f"Template file not found: {template_path} (template: {template_id})"
        )

    # -----------------------------------------------------------------
    # 17. Integration files exist on disk
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("integration_id,filename", list(INTEGRATION_FILES.items()))
    def test_integration_files_exist(self, integration_id: str, filename: str):
        """Test that each integration Python file exists on disk."""
        integration_path = INTEGRATIONS_DIR / filename
        assert integration_path.is_file(), (
            f"Integration file not found: {integration_path} (integration: {integration_id})"
        )

    # -----------------------------------------------------------------
    # 18. Unique component IDs
    # -----------------------------------------------------------------

    def test_unique_component_ids(self, pack_yaml: Dict[str, Any]):
        """Test that component IDs are unique within each section (engines, workflows, etc.)."""
        components = pack_yaml.get("components", {})

        for section in ["engines", "workflows", "templates", "integrations"]:
            items = components.get(section, [])
            ids = [item.get("id", "") for item in items if item.get("id")]
            assert len(ids) == len(set(ids)), (
                f"Duplicate IDs in {section}: "
                f"{[x for x in ids if ids.count(x) > 1]}"
            )

    # -----------------------------------------------------------------
    # 19. Agent dependencies total = 72
    # -----------------------------------------------------------------

    def test_agent_dependencies_total(self, pack_yaml: Dict[str, Any]):
        """Test that total agent dependencies equals 72."""
        deps = pack_yaml.get("dependencies", {})
        total = deps.get("total_agents", 0)
        assert total == 72, (
            f"Expected 72 total agents, got {total}"
        )

    # -----------------------------------------------------------------
    # 20. Agent dependencies breakdown
    # -----------------------------------------------------------------

    def test_agent_dependencies_breakdown(self, pack_yaml: Dict[str, Any]):
        """Test agent dependency breakdown (30 MRV + 20 data + 10 foundation + 12 sector)."""
        deps = pack_yaml.get("dependencies", {})
        breakdown = deps.get("breakdown", {})

        assert breakdown.get("mrv_agents") == 30, (
            f"Expected 30 MRV agents, got {breakdown.get('mrv_agents')}"
        )
        assert breakdown.get("data_agents") == 20, (
            f"Expected 20 data agents, got {breakdown.get('data_agents')}"
        )
        assert breakdown.get("foundation_agents") == 10, (
            f"Expected 10 foundation agents, got {breakdown.get('foundation_agents')}"
        )
        assert breakdown.get("financial_sector_agents") == 9, (
            f"Expected 9 financial sector agents, got {breakdown.get('financial_sector_agents')}"
        )
        assert breakdown.get("climate_agents") == 2, (
            f"Expected 2 climate agents, got {breakdown.get('climate_agents')}"
        )
        assert breakdown.get("policy_agents") == 1, (
            f"Expected 1 policy agent, got {breakdown.get('policy_agents')}"
        )

    # -----------------------------------------------------------------
    # 21. MRV agents section exists
    # -----------------------------------------------------------------

    def test_mrv_agents_section(self, pack_yaml: Dict[str, Any]):
        """Test that agents_mrv section exists with scope_1, scope_2, scope_3."""
        components = pack_yaml.get("components", {})
        agents_mrv = components.get("agents_mrv", {})
        assert isinstance(agents_mrv, dict), "agents_mrv must be a dict"

        # Validate scope sections
        for scope in ["scope_1", "scope_2", "scope_3", "cross_cutting"]:
            assert scope in agents_mrv, f"Missing agents_mrv.{scope} section"

        # Validate counts: 8 scope_1 + 5 scope_2 + 15 scope_3 + 2 cross_cutting = 30
        s1 = agents_mrv.get("scope_1", [])
        s2 = agents_mrv.get("scope_2", [])
        s3 = agents_mrv.get("scope_3", [])
        cc = agents_mrv.get("cross_cutting", [])
        total = len(s1) + len(s2) + len(s3) + len(cc)
        assert total == 30, f"Expected 30 MRV agents, found {total}"

    # -----------------------------------------------------------------
    # 22. Foundation agents section
    # -----------------------------------------------------------------

    def test_foundation_agents_section(self, pack_yaml: Dict[str, Any]):
        """Test that agents_foundation section has 10 agents."""
        components = pack_yaml.get("components", {})
        agents_found = components.get("agents_foundation", [])
        assert isinstance(agents_found, list), "agents_foundation must be a list"
        assert len(agents_found) == 10, (
            f"Expected 10 foundation agents, found {len(agents_found)}"
        )

    # -----------------------------------------------------------------
    # 23. Financial sector agents section
    # -----------------------------------------------------------------

    def test_financial_sector_agents_section(self, pack_yaml: Dict[str, Any]):
        """Test that agents_financial_sector section has 12 agents (9 FIN + 2 climate + 1 policy)."""
        components = pack_yaml.get("components", {})
        agents_fs = components.get("agents_financial_sector", [])
        assert isinstance(agents_fs, list), "agents_financial_sector must be a list"
        assert len(agents_fs) == 12, (
            f"Expected 12 financial sector agents, found {len(agents_fs)}"
        )

    # -----------------------------------------------------------------
    # 24. Tags include key financial terms
    # -----------------------------------------------------------------

    def test_metadata_tags(self, pack_yaml: Dict[str, Any]):
        """Test that metadata tags include key financial services terms."""
        meta = pack_yaml.get("metadata", {})
        tags = meta.get("tags", [])
        assert isinstance(tags, list), "tags must be a list"

        required_tags = [
            "csrd", "financial-services", "pcaf", "gar", "pillar-3",
            "climate-risk", "eu-compliance", "sector-specific",
        ]
        for tag in required_tags:
            assert tag in tags, f"Missing required tag: {tag}. Found: {tags}"

    # -----------------------------------------------------------------
    # 25. Performance targets present
    # -----------------------------------------------------------------

    def test_performance_targets_present(self, pack_yaml: Dict[str, Any]):
        """Test that performance section defines processing targets."""
        performance = pack_yaml.get("performance", {})
        assert isinstance(performance, dict), "performance must be a dict"
        assert "portfolio_processing" in performance, (
            "Missing performance.portfolio_processing"
        )
