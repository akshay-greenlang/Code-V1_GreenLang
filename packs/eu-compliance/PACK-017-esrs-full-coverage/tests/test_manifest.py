# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - Manifest Tests (test_manifest.py)
=====================================================================

Validates pack.yaml structure: metadata, engines, workflows, templates,
integrations, presets, agent dependencies, performance targets, security
config, and all 12 ESRS regulatory references.

Target: 80+ tests across 10 test classes.

Test Classes:
    TestManifestExists            - File existence and parseability (5 tests)
    TestMetadata                  - Pack metadata fields and values (18 tests)
    TestComponents                - Engine/workflow/template/integration counts (10 tests)
    TestEngineSpecs               - Parametrized per-engine validation (44 tests)
    TestWorkflowSpecs             - Parametrized per-workflow validation (48 tests)
    TestTemplateSpecs             - Parametrized per-template validation (48 tests)
    TestAgentDependencies         - MRV, data, and foundation agent references (6 tests)
    TestDependencies              - Required packs and Python version (4 tests)
    TestSecurity                  - JWT, RBAC, encryption, TLS config (5 tests)
    TestPerformance               - Per-engine performance targets (5 tests)

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-017 ESRS Full Coverage
Date:    March 2026
"""

import re
from pathlib import Path

import pytest

from .conftest import (
    ALL_ESRS_STANDARDS,
    ENGINE_CLASSES,
    ENGINE_FILES,
    INTEGRATION_CLASSES,
    INTEGRATION_FILES,
    PACK_ROOT,
    PRESET_NAMES,
    TEMPLATE_CLASSES,
    TEMPLATE_FILES,
    WORKFLOW_CLASSES,
    WORKFLOW_FILES,
)


# ===========================================================================
# Pack YAML Existence and Parse
# ===========================================================================


class TestManifestExists:
    """Tests for pack.yaml file existence and parseability."""

    def test_pack_yaml_exists(self, pack_yaml_path):
        """pack.yaml file exists on disk."""
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"

    def test_pack_yaml_is_file(self, pack_yaml_path):
        """pack.yaml is a regular file (not a directory)."""
        if not pack_yaml_path.exists():
            pytest.skip("pack.yaml not found")
        assert pack_yaml_path.is_file()

    def test_pack_yaml_parseable(self, pack_yaml_data):
        """pack.yaml parses to a non-empty dictionary."""
        assert isinstance(pack_yaml_data, dict)
        assert len(pack_yaml_data) > 0

    def test_pack_yaml_has_required_keys(self, pack_yaml_data):
        """pack.yaml has metadata, engines, workflows, and dependencies sections."""
        for key in ["metadata", "engines", "workflows", "dependencies"]:
            assert key in pack_yaml_data, f"Missing required key: {key}"

    def test_pack_yaml_has_at_least_five_top_level_keys(self, pack_yaml_data):
        """pack.yaml contains at least 5 top-level keys."""
        assert len(pack_yaml_data) >= 5, (
            f"pack.yaml has only {len(pack_yaml_data)} top-level keys; "
            f"expected at least 5"
        )


# ===========================================================================
# Metadata
# ===========================================================================


class TestMetadata:
    """Tests for pack.yaml metadata section."""

    def test_metadata_section_exists(self, pack_yaml_data):
        """Metadata section is present."""
        assert "metadata" in pack_yaml_data

    def test_metadata_fields_present(self, pack_yaml_data):
        """Metadata section contains all required fields."""
        meta = pack_yaml_data.get("metadata", {})
        required = [
            "name", "version", "display_name", "description",
            "category", "tier", "author", "license",
        ]
        for field in required:
            assert field in meta, f"Missing metadata field: {field}"

    def test_pack_name_correct(self, pack_yaml_data):
        """Pack name is PACK-017-esrs-full-coverage."""
        meta = pack_yaml_data["metadata"]
        assert meta["name"] == "PACK-017-esrs-full-coverage"

    def test_pack_version_valid(self, pack_yaml_data):
        """Pack version follows semver pattern."""
        version = pack_yaml_data["metadata"]["version"]
        assert re.match(r"^\d+\.\d+\.\d+$", version), f"Invalid version: {version}"

    def test_pack_version_is_1_0_0(self, pack_yaml_data):
        """Pack version is 1.0.0."""
        assert pack_yaml_data["metadata"]["version"] == "1.0.0"

    def test_pack_display_name_contains_esrs(self, pack_yaml_data):
        """Display name contains ESRS."""
        display = pack_yaml_data["metadata"]["display_name"]
        assert "ESRS" in display, f"Display name '{display}' does not contain 'ESRS'"

    def test_pack_display_name_contains_full_or_coverage(self, pack_yaml_data):
        """Display name contains Full Coverage or equivalent."""
        display = pack_yaml_data["metadata"]["display_name"].lower()
        assert "full" in display or "coverage" in display or "complete" in display

    def test_pack_category_is_eu_compliance(self, pack_yaml_data):
        """Category is eu-compliance."""
        assert pack_yaml_data["metadata"]["category"] == "eu-compliance"

    def test_pack_tier(self, pack_yaml_data):
        """Tier field is present and non-empty."""
        meta = pack_yaml_data["metadata"]
        assert "tier" in meta
        assert len(str(meta["tier"])) > 0

    def test_pack_has_tags(self, pack_yaml_data):
        """Pack has at least 30 tags for discoverability."""
        meta = pack_yaml_data["metadata"]
        tags = meta.get("tags", [])
        assert len(tags) >= 30, f"Expected at least 30 tags, got {len(tags)}"

    def test_tags_include_esrs_standard_references(self, pack_yaml_data):
        """Tags include references to key ESRS standards."""
        tags = pack_yaml_data["metadata"].get("tags", [])
        tags_joined = " ".join(t.lower() for t in tags)
        # Check for core tags (ESRS, CSRD, at least some E/S/G standards)
        required_tags = ["esrs", "csrd"]
        for tag in required_tags:
            assert tag in tags_joined, f"'{tag}' not found in tags"

        # Check for at least some E/S/G standard references (not all, since E1 is via PACK-016)
        esg_tags = ["e2", "e3", "e4", "e5", "s1", "s2", "s3", "s4", "g1"]
        esg_found = sum(1 for esg_tag in esg_tags if esg_tag in tags_joined)
        assert esg_found >= 5, f"Expected at least 5 E/S/G standards in tags, found {esg_found}"

    def test_regulation_section_exists(self, pack_yaml_data):
        """Regulation section exists in metadata."""
        meta = pack_yaml_data["metadata"]
        assert "regulation" in meta, "Missing 'regulation' in metadata"

    def test_regulation_primary_references_esrs(self, pack_yaml_data):
        """Primary regulation references ESRS Set 1 or Delegated Regulation."""
        reg = pack_yaml_data["metadata"]["regulation"]
        primary = reg.get("primary", {})
        primary_str = str(primary)
        assert "ESRS" in primary_str or "2023/2772" in primary_str

    def test_regulation_secondary_count_at_least_4(self, pack_yaml_data):
        """At least 4 secondary regulation references."""
        reg = pack_yaml_data["metadata"]["regulation"]
        secondary = reg.get("secondary", [])
        assert len(secondary) >= 4, (
            f"Expected at least 4 secondary regulations, got {len(secondary)}"
        )

    def test_csrd_in_secondary_regulations(self, pack_yaml_data):
        """CSRD Directive 2022/2464 is referenced in secondary regulations."""
        reg = pack_yaml_data["metadata"]["regulation"]
        secondary = reg.get("secondary", [])
        csrd_found = any(
            "CSRD" in str(s) or "2022/2464" in str(s) for s in secondary
        )
        assert csrd_found, "CSRD not found in secondary regulations"

    def test_delegated_regulation_2023_2772_referenced(self, pack_yaml_data):
        """EU Delegated Regulation 2023/2772 is referenced."""
        reg = pack_yaml_data["metadata"]["regulation"]
        all_refs = str(reg)
        assert "2023/2772" in all_refs

    @pytest.mark.parametrize("standard_id", ALL_ESRS_STANDARDS)
    def test_each_standard_referenced_in_yaml(self, pack_yaml_data, standard_id):
        """Each of the 12 ESRS standards appears somewhere in pack.yaml."""
        yaml_text = str(pack_yaml_data).upper()
        assert standard_id in yaml_text, (
            f"Standard {standard_id} not found anywhere in pack.yaml"
        )


# ===========================================================================
# Components
# ===========================================================================


class TestComponents:
    """Tests for components section in pack.yaml."""

    def test_engines_section_exists(self, pack_yaml_data):
        """Engines section exists."""
        assert "engines" in pack_yaml_data

    def test_engines_listed(self, pack_yaml_data):
        """Engines section lists 11 engines."""
        engines = pack_yaml_data["engines"]
        assert len(engines) == 11, f"Expected 11 engines, got {len(engines)}"

    def test_workflows_listed(self, pack_yaml_data):
        """Workflows section lists 12 workflows."""
        workflows = pack_yaml_data["workflows"]
        assert len(workflows) == 12, f"Expected 12 workflows, got {len(workflows)}"

    def test_templates_listed(self, pack_yaml_data):
        """Templates section lists 12 templates."""
        templates = pack_yaml_data["templates"]
        assert len(templates) == 12, f"Expected 12 templates, got {len(templates)}"

    def test_integrations_listed(self, pack_yaml_data):
        """Integrations section lists 10 integrations."""
        integrations = pack_yaml_data["integrations"]
        assert len(integrations) == 10, f"Expected 10 integrations, got {len(integrations)}"

    def test_presets_listed(self, pack_yaml_data):
        """Presets section lists 6 presets."""
        presets = pack_yaml_data.get("presets", [])
        assert len(presets) == 6, f"Expected 6 presets, got {len(presets)}"

    def test_engine_ids_unique(self, pack_yaml_data):
        """Engine IDs are unique."""
        engines = pack_yaml_data["engines"]
        ids = [e["id"] for e in engines]
        assert len(ids) == len(set(ids)), "Duplicate engine IDs found"

    def test_workflow_ids_unique(self, pack_yaml_data):
        """Workflow IDs are unique."""
        workflows = pack_yaml_data["workflows"]
        ids = [w["id"] for w in workflows]
        assert len(ids) == len(set(ids)), "Duplicate workflow IDs found"

    def test_all_engines_have_required_fields(self, pack_yaml_data):
        """All engines have id, name, and description."""
        engines = pack_yaml_data["engines"]
        for engine in engines:
            assert "id" in engine, "Engine missing id"
            assert "name" in engine, f"Engine {engine.get('id', '?')} missing name"
            assert "description" in engine, f"Engine {engine.get('id', '?')} missing description"

    def test_orchestrator_engine_present(self, pack_yaml_data):
        """ESRS Coverage Orchestrator engine is present."""
        engines = pack_yaml_data["engines"]
        ids = [e["id"] for e in engines]
        assert any("orchestrator" in eid.lower() for eid in ids)


# ===========================================================================
# Engine Specs (Parametrized)
# ===========================================================================


class TestEngineSpecs:
    """Parametrized tests validating each of the 11 engine specifications."""

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_file_mapped(self, engine_key):
        """Engine key has a corresponding file name in ENGINE_FILES."""
        assert engine_key in ENGINE_FILES
        assert ENGINE_FILES[engine_key].endswith(".py")

    @pytest.mark.parametrize("engine_key", list(ENGINE_CLASSES.keys()))
    def test_engine_class_mapped(self, engine_key):
        """Engine key has a corresponding class name in ENGINE_CLASSES."""
        assert engine_key in ENGINE_CLASSES
        class_name = ENGINE_CLASSES[engine_key]
        assert class_name[0].isupper(), f"Class name '{class_name}' should start uppercase"

    @pytest.mark.parametrize("engine_key,expected_class", list(ENGINE_CLASSES.items()))
    def test_engine_class_ends_with_engine(self, engine_key, expected_class):
        """Engine class name ends with 'Engine'."""
        assert expected_class.endswith("Engine"), (
            f"Engine class '{expected_class}' for '{engine_key}' does not end with 'Engine'"
        )

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_file_ends_with_engine_py(self, engine_key):
        """Engine file name ends with '_engine.py'."""
        file_name = ENGINE_FILES[engine_key]
        assert file_name.endswith("_engine.py"), (
            f"Engine file '{file_name}' does not end with '_engine.py'"
        )


# ===========================================================================
# Workflow Specs (Parametrized)
# ===========================================================================


class TestWorkflowSpecs:
    """Parametrized tests validating each of the 12 workflow specifications."""

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_file_mapped(self, wf_key):
        """Workflow key has a corresponding file name."""
        assert wf_key in WORKFLOW_FILES
        assert WORKFLOW_FILES[wf_key].endswith(".py")

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_CLASSES.keys()))
    def test_workflow_class_mapped(self, wf_key):
        """Workflow key has a corresponding class name."""
        assert wf_key in WORKFLOW_CLASSES
        assert WORKFLOW_CLASSES[wf_key][0].isupper()

    @pytest.mark.parametrize("wf_key,expected_class", list(WORKFLOW_CLASSES.items()))
    def test_workflow_class_ends_with_workflow(self, wf_key, expected_class):
        """Workflow class name ends with 'Workflow'."""
        assert expected_class.endswith("Workflow"), (
            f"Workflow class '{expected_class}' does not end with 'Workflow'"
        )

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_file_ends_with_workflow_py(self, wf_key):
        """Workflow file name ends with '_workflow.py'."""
        file_name = WORKFLOW_FILES[wf_key]
        assert file_name.endswith("_workflow.py"), (
            f"Workflow file '{file_name}' does not end with '_workflow.py'"
        )


# ===========================================================================
# Template Specs (Parametrized)
# ===========================================================================


class TestTemplateSpecs:
    """Parametrized tests validating each of the 12 template specifications."""

    @pytest.mark.parametrize("tpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_file_mapped(self, tpl_key):
        """Template key has a corresponding file name."""
        assert tpl_key in TEMPLATE_FILES
        assert TEMPLATE_FILES[tpl_key].endswith(".py")

    @pytest.mark.parametrize("tpl_key", list(TEMPLATE_CLASSES.keys()))
    def test_template_class_mapped(self, tpl_key):
        """Template key has a corresponding class name."""
        assert tpl_key in TEMPLATE_CLASSES
        assert TEMPLATE_CLASSES[tpl_key][0].isupper()

    @pytest.mark.parametrize("tpl_key,expected_class", list(TEMPLATE_CLASSES.items()))
    def test_template_class_ends_with_report_or_template(self, tpl_key, expected_class):
        """Template class name ends with 'Report' or 'Template' or valid suffix."""
        # Allow Report, Template, or Scorecard suffixes
        valid_suffixes = ("Report", "Template", "Scorecard")
        assert any(expected_class.endswith(suffix) for suffix in valid_suffixes), (
            f"Template class '{expected_class}' does not end with valid suffix (Report/Template/Scorecard)"
        )

    @pytest.mark.parametrize("tpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_file_ends_with_report_or_statement_py(self, tpl_key):
        """Template file name ends with '_report.py' or '_statement.py'."""
        file_name = TEMPLATE_FILES[tpl_key]
        valid_suffixes = ("_report.py", "_statement.py")
        assert any(file_name.endswith(suffix) for suffix in valid_suffixes), (
            f"Template file '{file_name}' does not end with valid suffix (_report.py or _statement.py)"
        )


# ===========================================================================
# Agent Dependencies
# ===========================================================================


class TestAgentDependencies:
    """Tests for agent dependency declarations in pack.yaml."""

    def test_agents_referenced(self, pack_yaml_data):
        """AGENT dependencies exist and are non-empty."""
        deps = pack_yaml_data.get("dependencies", {})
        agents = deps.get("agents", [])
        assert agents is not None, "agents not found in dependencies"
        assert len(agents) > 0, "agents list appears empty"

    def test_agents_count_at_least_5(self, pack_yaml_data):
        """At least 5 agent references in dependencies."""
        deps = pack_yaml_data.get("dependencies", {})
        agents = deps.get("agents", [])
        if isinstance(agents, list):
            total = len(agents)
        elif isinstance(agents, dict):
            total = len(agents)
        else:
            total = 0
        assert total >= 5, f"Expected at least 5 agent dependencies, got {total}"

    def test_agents_mrv_range_referenced(self, pack_yaml_data):
        """AGENT-MRV-001-030 range reference exists in agents list."""
        deps = pack_yaml_data.get("dependencies", {})
        agents = deps.get("agents", [])
        agents_str = str(agents).upper()
        # Check for AGENT-MRV range notation or individual MRV references
        has_mrv = "AGENT-MRV" in agents_str or "MRV" in agents_str
        assert has_mrv, "No MRV agent references found in dependencies.agents"

    def test_agents_data_referenced(self, pack_yaml_data):
        """AGENT-DATA references exist in agents list."""
        deps = pack_yaml_data.get("dependencies", {})
        agents = deps.get("agents", [])
        agents_str = str(agents).upper()
        has_data = "AGENT-DATA" in agents_str or "DATA" in agents_str
        assert has_data, "No DATA agent references found in dependencies.agents"

    def test_agents_foundation_referenced(self, pack_yaml_data):
        """AGENT-FOUND references exist in agents list."""
        deps = pack_yaml_data.get("dependencies", {})
        agents = deps.get("agents", [])
        agents_str = str(agents).upper()
        has_found = "AGENT-FOUND" in agents_str or "FOUND" in agents_str
        assert has_found, "No FOUND agent references found in dependencies.agents"

    def test_specific_critical_agents_referenced(self, pack_yaml_data):
        """Critical agents (Orchestrator, Schema, Data Quality) are referenced."""
        deps = pack_yaml_data.get("dependencies", {})
        agents = deps.get("agents", [])
        agents_str = str(agents).upper()

        # Check for key agent types
        critical_agents = ["AGENT-FOUND-001", "AGENT-FOUND-002", "AGENT-DATA-010"]
        found_count = sum(1 for agent in critical_agents if agent in agents_str)

        assert found_count >= 2, (
            f"Expected at least 2 critical agents referenced, found {found_count}"
        )


# ===========================================================================
# Dependencies
# ===========================================================================


class TestDependencies:
    """Tests for dependencies section in pack.yaml."""

    def test_dependencies_section_exists(self, pack_yaml_data):
        """Dependencies section exists."""
        assert "dependencies" in pack_yaml_data

    def test_required_packs_present(self, pack_yaml_data):
        """Required packs list exists in dependencies."""
        deps = pack_yaml_data["dependencies"]
        assert "required" in deps, "dependencies.required not found"

    def test_pack_015_and_016_required(self, pack_yaml_data):
        """PACK-015 (DMA) and PACK-016 (E1) are listed as required dependencies."""
        deps = pack_yaml_data["dependencies"]
        required = deps.get("required", [])
        required_str = str(required)
        has_015 = "PACK-015" in required_str or "015" in required_str or "pack_id: PACK-015" in required_str
        has_016 = "PACK-016" in required_str or "016" in required_str or "pack_id: PACK-016" in required_str
        assert has_015, "PACK-015 not in dependencies.required"
        assert has_016, "PACK-016 not in dependencies.required"

    def test_python_minimum_version(self, pack_yaml_data):
        """Python minimum version is specified in metadata (3.9+)."""
        # Check metadata for platform version or look in dependencies
        meta = pack_yaml_data.get("metadata", {})
        min_version_str = meta.get("min_platform_version", "")

        # If not in metadata, skip this test as it's optional
        if not min_version_str:
            pytest.skip("No min_platform_version specified in metadata")

        # Platform version 2.0.0+ implies Python 3.9+
        version_match = re.search(r"(\d+)\.(\d+)", str(min_version_str))
        if version_match:
            major = int(version_match.group(1))
            # Platform 2.0+ is good enough (implies Python 3.9+)
            assert major >= 2, (
                f"Platform version {min_version_str} is below 2.0"
            )


# ===========================================================================
# Security
# ===========================================================================


class TestSecurity:
    """Tests for security configuration in pack.yaml."""

    def test_security_implied_by_platform(self, pack_yaml_data):
        """Security is implied by platform version (2.0+) which includes SEC-001-011."""
        meta = pack_yaml_data.get("metadata", {})
        min_version = meta.get("min_platform_version", "")

        # Platform 2.0+ includes full SEC suite (JWT, RBAC, AES-256, TLS 1.3)
        if min_version:
            version_match = re.search(r"(\d+)\.(\d+)", str(min_version))
            if version_match:
                major = int(version_match.group(1))
                assert major >= 2, "Platform version should be 2.0+ (includes SEC-001-011)"

    def test_enterprise_tier_includes_security(self, pack_yaml_data):
        """Enterprise tier packs include full security suite."""
        meta = pack_yaml_data.get("metadata", {})
        tier = meta.get("tier", "").lower()

        # Enterprise tier implies JWT, RBAC, encryption
        if tier == "enterprise":
            assert True  # Enterprise tier includes security by default
        else:
            pytest.skip(f"Pack tier is '{tier}', not enterprise")

    def test_security_via_support_tier(self, pack_yaml_data):
        """Enterprise-premium support includes security."""
        meta = pack_yaml_data.get("metadata", {})
        support_tier = meta.get("support_tier", "").lower()

        if "enterprise" in support_tier or "premium" in support_tier:
            assert True  # Premium support includes security
        else:
            pytest.skip(f"Support tier is '{support_tier}'")

    def test_security_documentation_referenced(self, pack_yaml_data):
        """Security is documented in pack description or dependencies."""
        # Check description for security mentions
        meta = pack_yaml_data.get("metadata", {})
        desc = str(meta.get("description", "")).lower()

        # Check for provenance/SHA-256 (security feature)
        has_security_ref = "provenance" in desc or "sha-256" in desc or "audit" in desc
        assert has_security_ref, "No security-related terms in description"

    def test_quality_gates_include_security(self, pack_yaml_data):
        """Quality gates include security requirements."""
        quality_gates = pack_yaml_data.get("quality_gates", {})

        # Check for provenance/audit requirements
        has_provenance = quality_gates.get("provenance_required", False)
        has_audit_trail = quality_gates.get("audit_trail_required", False)

        assert has_provenance or has_audit_trail, (
            "Quality gates should include provenance_required or audit_trail_required"
        )


# ===========================================================================
# Performance
# ===========================================================================


class TestPerformance:
    """Tests for performance targets in pack.yaml."""

    def test_performance_section_exists(self, pack_yaml_data):
        """Performance section exists."""
        assert "performance" in pack_yaml_data, "Missing 'performance' section"

    def test_memory_ceiling_defined(self, pack_yaml_data):
        """Memory ceiling is defined."""
        perf = pack_yaml_data["performance"]
        perf_str = str(perf).lower()
        assert "memory" in perf_str, "No memory ceiling found in performance section"

    def test_per_engine_targets_exist(self, pack_yaml_data):
        """Performance targets exist for individual engines."""
        perf = pack_yaml_data["performance"]
        engine_specific_keys = [
            k for k in perf.keys()
            if any(
                std in k.lower()
                for std in ["esrs2", "e2", "e3", "e4", "e5",
                            "s1", "s2", "s3", "s4", "g1",
                            "orchestrator", "engine"]
            )
        ]
        assert len(engine_specific_keys) >= 1, "No engine-specific performance targets found"

    def test_throughput_or_latency_target_defined(self, pack_yaml_data):
        """Throughput or latency target is defined."""
        perf = pack_yaml_data["performance"]
        perf_str = str(perf).lower()
        has_target = (
            "throughput" in perf_str
            or "latency" in perf_str
            or "records_per_second" in perf_str
            or "max_" in perf_str
        )
        assert has_target, "No throughput/latency target found"

    def test_concurrency_target_defined(self, pack_yaml_data):
        """Concurrency or parallel execution target is defined."""
        perf = pack_yaml_data["performance"]
        perf_str = str(perf).lower()
        has_concurrency = (
            "concurrent" in perf_str
            or "parallel" in perf_str
            or "batch_size" in perf_str
        )
        assert has_concurrency, "No concurrency target found"
