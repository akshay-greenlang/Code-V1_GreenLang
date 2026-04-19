# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Demo Configuration Tests
==========================================================

Tests the demo_config.yaml file for validity, completeness, and
correctness of sample company profile data.

Test count target: ~15 tests
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import DEMO_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEMO_CONFIG_PATH = DEMO_DIR / "demo_config.yaml"


@pytest.fixture(scope="module")
def demo_config() -> Dict[str, Any]:
    """Load demo_config.yaml once for all demo tests."""
    assert DEMO_CONFIG_PATH.exists(), f"demo_config.yaml not found at {DEMO_CONFIG_PATH}"
    with open(DEMO_CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert data is not None, "demo_config.yaml parsed as empty"
    return data


# ---------------------------------------------------------------------------
# 1. YAML validity
# ---------------------------------------------------------------------------


class TestDemoYamlValidity:
    """Verify demo config YAML is valid."""

    def test_demo_file_exists(self):
        assert DEMO_CONFIG_PATH.exists()

    def test_demo_yaml_parses(self, demo_config):
        assert isinstance(demo_config, dict)

    def test_demo_not_empty(self, demo_config):
        assert len(demo_config) > 0


# ---------------------------------------------------------------------------
# 2. Pack config fields
# ---------------------------------------------------------------------------


class TestDemoPackFields:
    """Verify standard pack config fields are present."""

    def test_pack_name(self, demo_config):
        assert demo_config.get("pack_name") == "PACK-019-csddd-readiness"

    def test_version(self, demo_config):
        assert demo_config.get("version") == "1.0.0"

    def test_company_scope(self, demo_config):
        assert demo_config.get("company_scope") == "PHASE_1"

    def test_sector(self, demo_config):
        assert demo_config.get("sector") == "MANUFACTURING"

    def test_enabled_engines(self, demo_config):
        engines = demo_config.get("enabled_engines", [])
        assert isinstance(engines, list)
        assert len(engines) == 8
        assert "scope_assessment" in engines
        assert "climate_transition" in engines

    def test_enabled_workflows(self, demo_config):
        workflows = demo_config.get("enabled_workflows", [])
        assert isinstance(workflows, list)
        assert len(workflows) == 12


# ---------------------------------------------------------------------------
# 3. Demo company profile
# ---------------------------------------------------------------------------


class TestDemoCompanyProfile:
    """Verify demo company profile fields."""

    def test_demo_company_exists(self, demo_config):
        assert "demo_company" in demo_config

    def test_demo_company_name(self, demo_config):
        company = demo_config["demo_company"]
        assert company.get("legal_name") == "EuroManufacturing AG"

    def test_demo_company_country(self, demo_config):
        company = demo_config["demo_company"]
        assert company.get("incorporation_country") == "DE"

    def test_demo_company_revenue(self, demo_config):
        company = demo_config["demo_company"]
        revenue = company.get("revenue_eur", 0)
        assert revenue >= 1_500_000_000, "Phase 1 company must exceed EUR 1.5bn turnover"

    def test_demo_company_employees(self, demo_config):
        company = demo_config["demo_company"]
        employees = company.get("employee_count", 0)
        assert employees >= 5000, "Phase 1 company must exceed 5000 employees"

    def test_demo_company_subsidiaries(self, demo_config):
        company = demo_config["demo_company"]
        subs = company.get("subsidiaries", [])
        assert isinstance(subs, list)
        assert len(subs) >= 1


# ---------------------------------------------------------------------------
# 4. Demo impacts and grievances
# ---------------------------------------------------------------------------


class TestDemoData:
    """Verify demo impact and grievance data are present."""

    def test_demo_impacts_exist(self, demo_config):
        impacts = demo_config.get("demo_impacts", [])
        assert isinstance(impacts, list)
        assert len(impacts) >= 3

    def test_demo_impacts_have_ids(self, demo_config):
        for impact in demo_config.get("demo_impacts", []):
            assert "impact_id" in impact
            assert "type" in impact
            assert "severity" in impact

    def test_demo_grievances_exist(self, demo_config):
        grievances = demo_config.get("demo_grievances", [])
        assert isinstance(grievances, list)
        assert len(grievances) >= 2

    def test_demo_supply_chain(self, demo_config):
        sc = demo_config.get("demo_supply_chain", {})
        assert sc.get("tier_1_suppliers", 0) > 0
        assert sc.get("countries_of_origin", 0) > 0
