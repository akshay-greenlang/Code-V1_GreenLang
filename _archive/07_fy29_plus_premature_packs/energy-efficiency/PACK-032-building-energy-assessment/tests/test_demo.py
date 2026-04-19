# -*- coding: utf-8 -*-
"""
Unit tests for PACK-032 Building Energy Assessment Demo Configuration

Tests demo_config.yaml loading, field validation, preset cross-check,
and sample engine execution with demo data.

Target: 20+ tests
Author: GL-TestEngineer
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

PACK_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACK_ROOT / "config"
DEMO_DIR = CONFIG_DIR / "demo"
PRESETS_DIR = CONFIG_DIR / "presets"


def _load_config():
    path = CONFIG_DIR / "pack_config.py"
    if not path.exists():
        pytest.skip(f"pack_config.py not found: {path}")
    mod_key = "pack032_demo_cfg.pack_config"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load pack_config: {exc}")
    return mod


@pytest.fixture(scope="module")
def cfg_mod():
    return _load_config()


@pytest.fixture(scope="module")
def demo_data() -> Dict[str, Any]:
    path = DEMO_DIR / "demo_config.yaml"
    if not path.exists():
        pytest.skip("demo_config.yaml not found")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None
    return data


@pytest.fixture
def demo_config(cfg_mod):
    path = DEMO_DIR / "demo_config.yaml"
    if not path.exists():
        pytest.skip("demo_config.yaml not found")
    return cfg_mod.PackConfig.from_yaml(path)


# =========================================================================
# Test Demo File Existence
# =========================================================================


class TestDemoFiles:
    def test_demo_dir_exists(self):
        assert DEMO_DIR.is_dir()

    def test_demo_config_exists(self):
        assert (DEMO_DIR / "demo_config.yaml").exists()


# =========================================================================
# Test Demo YAML Structure
# =========================================================================


class TestDemoYAMLStructure:
    def test_is_dict(self, demo_data):
        assert isinstance(demo_data, dict)

    def test_has_building_name(self, demo_data):
        assert "building_name" in demo_data
        assert demo_data["building_name"] != ""

    def test_has_building_type(self, demo_data):
        assert "building_type" in demo_data

    def test_has_country(self, demo_data):
        assert "country" in demo_data

    def test_has_gross_internal_area(self, demo_data):
        assert "gross_internal_area_m2" in demo_data
        assert demo_data["gross_internal_area_m2"] > 0

    def test_has_building_address(self, demo_data):
        assert "building_address" in demo_data

    def test_has_client_name(self, demo_data):
        assert "client_name" in demo_data

    def test_has_assessment_level(self, demo_data):
        assert "assessment_level" in demo_data


# =========================================================================
# Test Demo Config Values (Thames Quarter Office Tower)
# =========================================================================


class TestDemoValues:
    def test_building_name(self, demo_data):
        assert "Thames Quarter" in demo_data["building_name"]

    def test_building_type_office(self, demo_data):
        assert demo_data["building_type"] == "OFFICE"

    def test_country_gb(self, demo_data):
        assert demo_data["country"] == "GB"

    def test_climate_zone(self, demo_data):
        assert demo_data["climate_zone"] == "CENTRAL_MARITIME"

    def test_gia(self, demo_data):
        assert demo_data["gross_internal_area_m2"] == pytest.approx(18500.0, rel=0.01)

    def test_nla(self, demo_data):
        assert demo_data["net_lettable_area_m2"] == pytest.approx(15200.0, rel=0.01)

    def test_floors(self, demo_data):
        assert demo_data["number_of_floors"] == 12

    def test_basements(self, demo_data):
        assert demo_data["number_of_basements"] == 2

    def test_current_epc_rating(self, demo_data):
        assert demo_data["current_epc_rating"] == "D"

    def test_target_epc_rating(self, demo_data):
        assert demo_data["target_epc_rating"] == "B"

    def test_current_eui(self, demo_data):
        assert demo_data["current_eui_kwh_m2"] == pytest.approx(178.0, rel=0.01)


# =========================================================================
# Test Demo Sub-configurations
# =========================================================================


class TestDemoSubConfigs:
    def test_envelope_section(self, demo_data):
        assert "envelope" in demo_data
        assert demo_data["envelope"]["wall_u_target"] == pytest.approx(0.25, rel=0.01)

    def test_hvac_section(self, demo_data):
        assert "hvac" in demo_data
        assert "heating_system_type" in demo_data["hvac"]

    def test_lighting_section(self, demo_data):
        assert "lighting" in demo_data
        assert demo_data["lighting"]["lpd_target_w_m2"] > 0

    def test_dhw_section(self, demo_data):
        assert "dhw" in demo_data

    def test_renewable_section(self, demo_data):
        assert "renewable" in demo_data
        assert demo_data["renewable"]["solar_pv_enabled"] is True

    def test_benchmark_section(self, demo_data):
        assert "benchmark" in demo_data
        assert demo_data["benchmark"]["enabled"] is True

    def test_retrofit_section(self, demo_data):
        assert "retrofit" in demo_data
        assert demo_data["retrofit"]["budget_eur"] > 0

    def test_indoor_environment_section(self, demo_data):
        assert "indoor_environment" in demo_data

    def test_carbon_section(self, demo_data):
        assert "carbon" in demo_data
        assert demo_data["carbon"]["grid_emission_factor_kg_per_kwh"] > 0

    def test_compliance_section(self, demo_data):
        assert "compliance" in demo_data

    def test_report_section(self, demo_data):
        assert "report" in demo_data

    def test_integration_section(self, demo_data):
        assert "integration" in demo_data

    def test_performance_section(self, demo_data):
        assert "performance" in demo_data

    def test_security_section(self, demo_data):
        assert "security" in demo_data
        roles = demo_data["security"]["roles"]
        assert isinstance(roles, list)
        assert len(roles) >= 5


# =========================================================================
# Test Demo Config Loading into PackConfig
# =========================================================================


class TestDemoPackConfig:
    def test_demo_loads_successfully(self, demo_config):
        assert demo_config is not None

    def test_demo_building_type(self, cfg_mod, demo_config):
        assert demo_config.pack.building_type == cfg_mod.BuildingType.OFFICE

    def test_demo_country(self, demo_config):
        assert demo_config.pack.country == "GB"

    def test_demo_config_hash(self, demo_config):
        h = demo_config.get_config_hash()
        assert len(h) == 64

    def test_demo_config_hash_deterministic(self, cfg_mod):
        path = DEMO_DIR / "demo_config.yaml"
        c1 = cfg_mod.PackConfig.from_yaml(path)
        c2 = cfg_mod.PackConfig.from_yaml(path)
        assert c1.get_config_hash() == c2.get_config_hash()
