# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional Pack - Pack Manifest Validation Tests
=================================================================

Validates the integrity and structure of pack.yaml for the Professional tier.
Ensures all professional components, workflows, templates, and compliance
references are properly defined and internally consistent.

Test count: 15
Author: GreenLang QA Team
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest
import yaml

from .conftest import (
    PROFESSIONAL_COMPLIANCE_REFS,
    PROFESSIONAL_ENGINES,
    PROFESSIONAL_TEMPLATES,
    PROFESSIONAL_WORKFLOWS,
)


class TestPackManifest:
    """Validates pack.yaml structure, fields, and referential integrity."""

    # ------------------------------------------------------------------
    # 1. pack.yaml is valid YAML
    # ------------------------------------------------------------------
    def test_manifest_yaml_valid(self, pack_yaml_path: Path, pack_yaml: Dict[str, Any]):
        """Verify pack.yaml exists and parses as valid YAML without errors."""
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"
        assert isinstance(pack_yaml, dict), "pack.yaml did not parse into a dictionary"
        assert len(pack_yaml) > 0, "pack.yaml is empty"

    # ------------------------------------------------------------------
    # 2. Required fields present
    # ------------------------------------------------------------------
    def test_manifest_has_required_fields(self, pack_yaml: Dict[str, Any]):
        """Verify pack.yaml contains all mandatory top-level sections.

        Required: metadata, dependencies, components, workflows,
        templates, presets, requirements, performance.
        """
        required_sections = [
            "metadata", "dependencies", "components", "workflows",
            "templates", "presets", "requirements", "performance",
        ]
        for section in required_sections:
            assert section in pack_yaml, (
                f"Missing required top-level section '{section}' in pack.yaml"
            )

        # metadata sub-fields
        metadata = pack_yaml["metadata"]
        for field in ["name", "version", "category", "tier", "display_name", "description"]:
            assert field in metadata, f"Missing metadata field '{field}'"

    # ------------------------------------------------------------------
    # 3. Extends PACK-001
    # ------------------------------------------------------------------
    def test_manifest_extends_pack001(self, pack_yaml: Dict[str, Any]):
        """Verify PACK-002 declares a dependency on PACK-001."""
        deps = pack_yaml["dependencies"]
        assert isinstance(deps, list), "dependencies must be a list"
        assert len(deps) >= 1, "Expected at least one dependency (PACK-001)"

        pack001_dep = deps[0]
        assert pack001_dep["pack_id"] == "PACK-001", (
            f"First dependency should be PACK-001, got '{pack001_dep.get('pack_id')}'"
        )
        assert pack001_dep["required"] is True, "PACK-001 dependency must be required"
        assert "csrd-starter" in pack001_dep.get("name", ""), (
            "PACK-001 dependency should reference csrd-starter"
        )

    # ------------------------------------------------------------------
    # 4. Component count 93+
    # ------------------------------------------------------------------
    def test_manifest_component_count(self, pack_yaml: Dict[str, Any]):
        """Verify the total component count is 93+ (66 inherited + 27 professional)."""
        components = pack_yaml["components"]
        inherited_count = components.get("inherited", {}).get("agent_count", 0)
        assert inherited_count >= 66, (
            f"Expected at least 66 inherited agents, got {inherited_count}"
        )

        # Count professional components
        pro_groups = [
            "professional_apps", "cdp_engines", "tcfd_engines",
            "sbti_engines", "taxonomy_engines", "professional_engines",
        ]
        pro_count = 0
        for group in pro_groups:
            entries = components.get(group, [])
            pro_count += len(entries)

        total = inherited_count + pro_count
        assert total >= 93, (
            f"Expected at least 93 total components, got {total} "
            f"({inherited_count} inherited + {pro_count} professional)"
        )

    # ------------------------------------------------------------------
    # 5. Professional engines (7)
    # ------------------------------------------------------------------
    def test_manifest_professional_engines(self, pack_yaml: Dict[str, Any]):
        """Verify all 7 professional engines are defined."""
        engines = pack_yaml["components"].get("professional_engines", [])
        engine_ids = {e["id"] for e in engines}

        for expected_id in PROFESSIONAL_ENGINES:
            assert expected_id in engine_ids, (
                f"Missing professional engine '{expected_id}'"
            )

        assert len(engines) == 7, f"Expected 7 professional engines, got {len(engines)}"

    # ------------------------------------------------------------------
    # 6. Workflows (8 professional)
    # ------------------------------------------------------------------
    def test_manifest_workflows(self, pack_yaml: Dict[str, Any]):
        """Verify all 8 professional workflows are defined with phases."""
        workflows = pack_yaml["workflows"]
        for wf_name in PROFESSIONAL_WORKFLOWS:
            assert wf_name in workflows, f"Missing workflow '{wf_name}'"
            wf = workflows[wf_name]
            assert "display_name" in wf, f"Workflow '{wf_name}' missing display_name"
            assert "phases" in wf, f"Workflow '{wf_name}' missing phases"
            phases = wf["phases"]
            assert isinstance(phases, list) and len(phases) > 0, (
                f"Workflow '{wf_name}' must have at least one phase"
            )
            for phase in phases:
                assert "name" in phase, f"Phase in '{wf_name}' missing 'name'"
                assert "agents" in phase, (
                    f"Phase '{phase.get('name', '?')}' in '{wf_name}' missing 'agents'"
                )

    # ------------------------------------------------------------------
    # 7. Templates (10 professional)
    # ------------------------------------------------------------------
    def test_manifest_templates(self, pack_yaml: Dict[str, Any]):
        """Verify all 10 professional templates are defined."""
        templates = pack_yaml["templates"]
        assert isinstance(templates, list), "templates must be a list"
        template_ids = {t["id"] for t in templates}

        for expected_id in PROFESSIONAL_TEMPLATES:
            assert expected_id in template_ids, (
                f"Missing professional template '{expected_id}'"
            )

        assert len(templates) >= 10, (
            f"Expected at least 10 professional templates, got {len(templates)}"
        )

    # ------------------------------------------------------------------
    # 8. Presets (4 size + 5 sector)
    # ------------------------------------------------------------------
    def test_manifest_presets(self, pack_yaml: Dict[str, Any]):
        """Verify 4 size presets and 5 sector presets are defined."""
        presets = pack_yaml["presets"]
        assert "size_presets" in presets, "Missing size_presets"
        assert "sector_presets" in presets, "Missing sector_presets"

        size_ids = {p["id"] for p in presets["size_presets"]}
        expected_sizes = {"enterprise_group", "listed_company", "financial_institution", "multinational"}
        assert expected_sizes == size_ids, (
            f"Size preset mismatch: expected {expected_sizes}, got {size_ids}"
        )

        sector_ids = {p["id"] for p in presets["sector_presets"]}
        expected_sectors = {
            "manufacturing_pro", "financial_services_pro", "technology_pro",
            "energy_pro", "heavy_industry_pro",
        }
        assert expected_sectors == sector_ids, (
            f"Sector preset mismatch: expected {expected_sectors}, got {sector_ids}"
        )

    # ------------------------------------------------------------------
    # 9. Compliance references (6)
    # ------------------------------------------------------------------
    def test_manifest_compliance_references(self, pack_yaml: Dict[str, Any]):
        """Verify 6 compliance references: CSRD, ESRS, ESEF, ISAE 3000, ISAE 3410, EU Taxonomy."""
        refs = pack_yaml["metadata"]["compliance_references"]
        assert isinstance(refs, list), "compliance_references must be a list"
        ref_ids = {r["id"] for r in refs}

        for expected_id in PROFESSIONAL_COMPLIANCE_REFS:
            assert expected_id in ref_ids, (
                f"Missing compliance reference '{expected_id}'"
            )

        assert len(refs) >= 6, f"Expected at least 6 compliance references, got {len(refs)}"

        for ref in refs:
            assert "regulation" in ref, f"Ref '{ref['id']}' missing 'regulation'"
            assert "effective_date" in ref, f"Ref '{ref['id']}' missing 'effective_date'"

    # ------------------------------------------------------------------
    # 10. Performance targets
    # ------------------------------------------------------------------
    def test_manifest_performance_targets(self, pack_yaml: Dict[str, Any]):
        """Verify performance targets are defined for professional workloads."""
        perf = pack_yaml["performance"]
        expected_categories = [
            "data_ingestion", "ghg_calculation", "report_generation",
            "scenario_analysis", "api_response", "data_quality",
        ]
        for cat in expected_categories:
            assert cat in perf, f"Missing performance category '{cat}'"

        # Verify consolidated report target exists
        report_gen = perf["report_generation"]
        assert "consolidated_report_max_seconds" in report_gen, (
            "Missing consolidated_report_max_seconds target"
        )
        assert report_gen["consolidated_report_max_seconds"] <= 600, (
            "Consolidated report should complete in 600s or less"
        )

    # ------------------------------------------------------------------
    # 11. Semver format
    # ------------------------------------------------------------------
    def test_manifest_semver_format(self, pack_yaml: Dict[str, Any]):
        """Verify version follows semantic versioning (MAJOR.MINOR.PATCH)."""
        version = pack_yaml["metadata"]["version"]
        semver_pattern = r"^\d+\.\d+\.\d+$"
        assert re.match(semver_pattern, version), (
            f"Version '{version}' does not follow semver (MAJOR.MINOR.PATCH)"
        )
        major, minor, patch = [int(x) for x in version.split(".")]
        assert major >= 1, "Major version should be >= 1 for release"

    # ------------------------------------------------------------------
    # 12. Tier is professional
    # ------------------------------------------------------------------
    def test_manifest_tier_is_professional(self, pack_yaml: Dict[str, Any]):
        """Verify the pack tier is 'professional'."""
        assert pack_yaml["metadata"]["tier"] == "professional", (
            f"Expected tier 'professional', got '{pack_yaml['metadata'].get('tier')}'"
        )

    # ------------------------------------------------------------------
    # 13. PACK-001 dependency version
    # ------------------------------------------------------------------
    def test_manifest_pack001_dependency(self, pack_yaml: Dict[str, Any]):
        """Verify PACK-001 dependency has a version constraint."""
        deps = pack_yaml["dependencies"]
        pack001 = next((d for d in deps if d["pack_id"] == "PACK-001"), None)
        assert pack001 is not None, "PACK-001 dependency not found"
        assert "version" in pack001, "PACK-001 dependency missing version constraint"
        assert ">=" in pack001["version"], (
            "PACK-001 dependency should use >= version constraint"
        )

    # ------------------------------------------------------------------
    # 14. All engines referenced in workflows
    # ------------------------------------------------------------------
    def test_manifest_all_engines_referenced(self, pack_yaml: Dict[str, Any]):
        """Verify all 7 professional engines are used in at least one workflow."""
        workflows = pack_yaml["workflows"]
        all_workflow_agents: Set[str] = set()

        for wf_name, wf in workflows.items():
            for phase in wf.get("phases", []):
                for agent_id in phase.get("agents", []):
                    all_workflow_agents.add(agent_id)

        for engine_id in PROFESSIONAL_ENGINES:
            assert engine_id in all_workflow_agents, (
                f"Professional engine '{engine_id}' is not referenced in any workflow"
            )

    # ------------------------------------------------------------------
    # 15. All cross-frameworks present
    # ------------------------------------------------------------------
    def test_manifest_all_cross_frameworks(self, pack_yaml: Dict[str, Any]):
        """Verify CDP, TCFD, SBTi, and EU Taxonomy app components exist."""
        components = pack_yaml["components"]
        pro_apps = components.get("professional_apps", [])
        app_ids = {a["id"] for a in pro_apps}

        expected_apps = {"GL-CDP-APP", "GL-TCFD-APP", "GL-SBTi-APP", "GL-TAXONOMY-APP"}
        for app_id in expected_apps:
            assert app_id in app_ids, f"Missing cross-framework app '{app_id}'"

        # Verify engines exist for each framework
        engine_groups = {
            "cdp_engines": "GL-CDP-",
            "tcfd_engines": "GL-TCFD-",
            "sbti_engines": "GL-SBTi-",
            "taxonomy_engines": "GL-TAX-",
        }
        for group_name, prefix in engine_groups.items():
            engines = components.get(group_name, [])
            assert len(engines) >= 3, (
                f"Expected at least 3 engines in '{group_name}', got {len(engines)}"
            )
            for engine in engines:
                assert engine["id"].startswith(prefix), (
                    f"Engine '{engine['id']}' in '{group_name}' should start with '{prefix}'"
                )
