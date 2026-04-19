# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - pack.yaml manifest.

Validates the structure, metadata, component counts, regulatory references,
agent dependencies, performance targets, and security configuration.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~50 tests
"""

import re
from pathlib import Path

import pytest
import yaml

PACK_DIR = Path(__file__).resolve().parent.parent
PACK_YAML_PATH = PACK_DIR / "pack.yaml"


@pytest.fixture(scope="module")
def pack_yaml_path() -> Path:
    return PACK_YAML_PATH


@pytest.fixture(scope="module")
def pack_data() -> dict:
    with open(PACK_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None, "pack.yaml parsed as None (empty file?)"
    return data


@pytest.fixture(scope="module")
def metadata(pack_data: dict) -> dict:
    return pack_data.get("metadata", {})


@pytest.fixture(scope="module")
def components(pack_data: dict) -> dict:
    return pack_data.get("components", {})


# ===========================================================================
# Tests -- File Existence & YAML Validity
# ===========================================================================


class TestPackYamlStructure:
    def test_pack_yaml_exists(self, pack_yaml_path: Path) -> None:
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"

    def test_pack_yaml_valid_yaml(self, pack_yaml_path: Path) -> None:
        with open(pack_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_pack_yaml_not_empty(self, pack_data: dict) -> None:
        assert len(pack_data) > 0


# ===========================================================================
# Tests -- Top-Level Fields
# ===========================================================================


class TestTopLevelFields:
    @pytest.mark.parametrize("field", [
        "pack_id", "pack_name", "version", "tier", "category", "description",
    ])
    def test_required_top_level_fields(self, pack_data: dict, field: str) -> None:
        assert field in pack_data, f"Missing required field: {field}"
        assert pack_data[field], f"Field {field} is empty"

    def test_pack_id(self, pack_data: dict) -> None:
        assert pack_data["pack_id"] == "PACK-027"

    def test_pack_name(self, pack_data: dict) -> None:
        assert "Enterprise" in pack_data["pack_name"]
        assert "Net Zero" in pack_data["pack_name"]

    def test_category(self, pack_data: dict) -> None:
        assert pack_data["category"] == "Net Zero"

    def test_tier(self, pack_data: dict) -> None:
        assert pack_data["tier"] == "Enterprise"

    def test_version_format(self, pack_data: dict) -> None:
        version = pack_data["version"]
        assert re.match(r"^\d+\.\d+\.\d+$", version)


# ===========================================================================
# Tests -- Metadata Section
# ===========================================================================


class TestMetadataFields:
    def test_metadata_section_exists(self, pack_data: dict) -> None:
        assert "metadata" in pack_data

    def test_author_present(self, metadata: dict) -> None:
        assert "author" in metadata and metadata["author"]

    def test_keywords_present(self, metadata: dict) -> None:
        keywords = metadata.get("keywords", [])
        assert isinstance(keywords, list)
        assert len(keywords) >= 10

    @pytest.mark.parametrize("expected_keyword", [
        "enterprise", "net-zero", "sbti", "carbon-pricing",
        "scenario-modeling", "assurance", "compliance",
    ])
    def test_critical_keywords_present(self, metadata: dict, expected_keyword: str) -> None:
        keywords = metadata.get("keywords", [])
        assert expected_keyword in keywords, f"Missing keyword: {expected_keyword}"

    def test_homepage(self, metadata: dict) -> None:
        assert "homepage" in metadata

    def test_documentation(self, metadata: dict) -> None:
        assert "documentation" in metadata


# ===========================================================================
# Tests -- Component Counts
# ===========================================================================


class TestComponentCounts:
    def test_engines_count(self, components: dict) -> None:
        engines = components.get("engines", {})
        engine_list = engines.get("list", [])
        assert len(engine_list) >= 8

    def test_workflows_count(self, components: dict) -> None:
        workflows = components.get("workflows", {})
        wf_list = workflows.get("list", [])
        assert len(wf_list) >= 8

    def test_templates_count(self, components: dict) -> None:
        templates = components.get("templates", {})
        tmpl_list = templates.get("list", [])
        assert len(tmpl_list) >= 10

    def test_integrations_count(self, components: dict) -> None:
        integrations = components.get("integrations", {})
        integ_list = integrations.get("list", [])
        assert len(integ_list) >= 13

    def test_presets_count(self, components: dict) -> None:
        presets = components.get("presets", {})
        sectors = presets.get("sectors", [])
        assert len(sectors) >= 8

    @pytest.mark.parametrize("engine_name", [
        "enterprise_baseline_engine", "sbti_target_engine",
        "scenario_modeling_engine", "carbon_pricing_engine",
        "scope4_avoided_emissions_engine", "supply_chain_mapping_engine",
        "multi_entity_consolidation_engine", "financial_integration_engine",
    ])
    def test_expected_engines_present(self, components: dict, engine_name: str) -> None:
        engine_list = components.get("engines", {}).get("list", [])
        assert engine_name in engine_list, f"Missing engine: {engine_name}"

    @pytest.mark.parametrize("workflow_name", [
        "comprehensive_baseline_workflow", "sbti_submission_workflow",
        "annual_inventory_workflow", "scenario_analysis_workflow",
        "supply_chain_engagement_workflow", "regulatory_filing_workflow",
    ])
    def test_expected_workflows_present(self, components: dict, workflow_name: str) -> None:
        wf_list = components.get("workflows", {}).get("list", [])
        assert workflow_name in wf_list, f"Missing workflow: {workflow_name}"

    @pytest.mark.parametrize("template_name", [
        "ghg_inventory_report", "sbti_target_submission",
        "cdp_climate_response", "tcfd_report", "executive_dashboard",
    ])
    def test_expected_templates_present(self, components: dict, template_name: str) -> None:
        tmpl_list = components.get("templates", {}).get("list", [])
        assert template_name in tmpl_list, f"Missing template: {template_name}"


# ===========================================================================
# Tests -- Enterprise-Specific Requirements
# ===========================================================================


class TestEnterpriseRequirements:
    def test_target_audience_present(self, pack_data: dict) -> None:
        assert "target_audience" in pack_data

    def test_target_audience_enterprise(self, pack_data: dict) -> None:
        audience = pack_data.get("target_audience", {})
        size = audience.get("company_size", "")
        assert "enterprise" in size.lower() or "250" in size

    def test_use_cases_defined(self, pack_data: dict) -> None:
        audience = pack_data.get("target_audience", {})
        use_cases = audience.get("use_cases", [])
        assert len(use_cases) >= 5

    def test_erp_integrations(self, components: dict) -> None:
        integ_list = components.get("integrations", {}).get("list", [])
        assert any("sap" in i.lower() for i in integ_list)
        assert any("oracle" in i.lower() for i in integ_list)
        assert any("workday" in i.lower() for i in integ_list)


# ===========================================================================
# Tests -- Platform Dependencies
# ===========================================================================


class TestPlatformDependencies:
    def test_platform_dependencies_present(self, pack_data: dict) -> None:
        assert "platform_dependencies" in pack_data

    def test_infrastructure_deps(self, pack_data: dict) -> None:
        deps = pack_data.get("platform_dependencies", {})
        infra = deps.get("infrastructure", [])
        assert len(infra) >= 5

    def test_security_deps(self, pack_data: dict) -> None:
        deps = pack_data.get("platform_dependencies", {})
        sec = deps.get("security", [])
        assert len(sec) >= 4

    def test_mrv_agents_all(self, pack_data: dict) -> None:
        deps = pack_data.get("platform_dependencies", {})
        mrv = deps.get("mrv_agents", {})
        assert mrv.get("all") is True

    def test_foundation_agents(self, pack_data: dict) -> None:
        deps = pack_data.get("platform_dependencies", {})
        found = deps.get("foundation_agents", [])
        assert len(found) >= 8

    def test_data_agents(self, pack_data: dict) -> None:
        deps = pack_data.get("platform_dependencies", {})
        data = deps.get("data_agents", [])
        assert len(data) >= 8


# ===========================================================================
# Tests -- Technical Specs
# ===========================================================================


class TestTechnicalSpecs:
    def test_technical_specs_present(self, pack_data: dict) -> None:
        assert "technical_specs" in pack_data

    def test_performance_targets(self, pack_data: dict) -> None:
        specs = pack_data.get("technical_specs", {})
        perf = specs.get("performance_targets", {})
        assert len(perf) >= 3

    def test_data_quality_targets(self, pack_data: dict) -> None:
        specs = pack_data.get("technical_specs", {})
        dq = specs.get("data_quality_targets", {})
        assert len(dq) >= 2


# ===========================================================================
# Tests -- Compliance
# ===========================================================================


class TestCompliance:
    def test_compliance_section_present(self, pack_data: dict) -> None:
        assert "compliance" in pack_data

    def test_frameworks_defined(self, pack_data: dict) -> None:
        compliance = pack_data.get("compliance", {})
        frameworks = compliance.get("frameworks", [])
        assert len(frameworks) >= 8

    @pytest.mark.parametrize("framework_name", [
        "GHG Protocol", "SBTi", "ISO 14064", "SEC Climate",
        "CSRD", "CDP", "TCFD",
    ])
    def test_key_frameworks_present(self, pack_data: dict, framework_name: str) -> None:
        compliance = pack_data.get("compliance", {})
        frameworks = compliance.get("frameworks", [])
        names = [f.get("name", "") for f in frameworks]
        assert any(framework_name in n for n in names), \
            f"Missing framework: {framework_name}"


# ===========================================================================
# Tests -- Release & Testing
# ===========================================================================


class TestReleaseInfo:
    def test_release_section(self, pack_data: dict) -> None:
        assert "release" in pack_data

    def test_release_status(self, pack_data: dict) -> None:
        release = pack_data.get("release", {})
        assert release.get("status") == "Production Ready"

    def test_testing_section(self, pack_data: dict) -> None:
        assert "testing" in pack_data

    def test_testing_coverage(self, pack_data: dict) -> None:
        testing = pack_data.get("testing", {})
        assert "target_coverage" in testing
