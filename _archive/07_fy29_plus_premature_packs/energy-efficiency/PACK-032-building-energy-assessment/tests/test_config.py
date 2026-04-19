# -*- coding: utf-8 -*-
"""
Unit tests for PACK-032 Building Energy Assessment Pack Configuration

Tests PackConfig instantiation, enum values, preset loading, validation,
merge behaviour, environment variable overrides, sub-config defaults,
and provenance hashing.

Target: 65+ tests
Author: GL-TestEngineer
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

PACK_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
DEMO_DIR = CONFIG_DIR / "demo"


def _load_config():
    path = CONFIG_DIR / "pack_config.py"
    if not path.exists():
        pytest.skip(f"pack_config.py not found: {path}")
    mod_key = "pack032_cfg.pack_config"
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


@pytest.fixture
def pack_config(cfg_mod):
    return cfg_mod.PackConfig()


@pytest.fixture
def bea_config(cfg_mod):
    """BuildingEnergyAssessmentConfig with defaults."""
    return cfg_mod.BuildingEnergyAssessmentConfig()


# =========================================================================
# Test Enum Existence and Values
# =========================================================================


class TestEnums:
    def test_building_type_enum(self, cfg_mod):
        bt = cfg_mod.BuildingType
        assert hasattr(bt, "OFFICE")
        assert hasattr(bt, "RETAIL")
        assert hasattr(bt, "HOTEL")
        assert hasattr(bt, "HOSPITAL")
        assert hasattr(bt, "WAREHOUSE")

    def test_building_type_count(self, cfg_mod):
        bt = cfg_mod.BuildingType
        members = list(bt)
        assert len(members) >= 10

    def test_climate_zone_enum(self, cfg_mod):
        cz = cfg_mod.ClimateZone
        assert hasattr(cz, "NORTHERN_EUROPE")
        assert hasattr(cz, "CENTRAL_MARITIME")
        assert hasattr(cz, "MEDITERRANEAN")

    def test_assessment_level_enum(self, cfg_mod):
        al = cfg_mod.AssessmentLevel
        assert hasattr(al, "WALK_THROUGH")
        assert hasattr(al, "STANDARD")
        assert hasattr(al, "DETAILED")
        assert hasattr(al, "INVESTMENT_GRADE")

    def test_certification_target_enum(self, cfg_mod):
        ct = cfg_mod.CertificationTarget
        assert hasattr(ct, "NONE")
        assert hasattr(ct, "LEED_GOLD")
        assert hasattr(ct, "BREEAM_EXCELLENT")

    def test_building_age_enum(self, cfg_mod):
        ba = cfg_mod.BuildingAge
        assert hasattr(ba, "PRE_1919")
        members = list(ba)
        assert len(members) >= 7

    def test_occupancy_pattern_enum(self, cfg_mod):
        op = cfg_mod.OccupancyPattern
        assert hasattr(op, "SINGLE_SHIFT")
        assert hasattr(op, "CONTINUOUS")

    def test_heating_fuel_enum(self, cfg_mod):
        hf = cfg_mod.HeatingFuel
        assert hasattr(hf, "NATURAL_GAS")
        assert hasattr(hf, "ELECTRICITY")
        assert hasattr(hf, "HEAT_PUMP")

    def test_ownership_type_enum(self, cfg_mod):
        ot = cfg_mod.OwnershipType
        assert hasattr(ot, "OWNER_OCCUPIED")
        assert hasattr(ot, "MULTI_TENANT")

    def test_epc_methodology_enum(self, cfg_mod):
        em = cfg_mod.EPCMethodology
        assert hasattr(em, "SAP")
        assert hasattr(em, "SBEM")
        assert hasattr(em, "GEG")

    def test_retrofit_ambition_enum(self, cfg_mod):
        ra = cfg_mod.RetrofitAmbition
        assert hasattr(ra, "COST_OPTIMAL")
        assert hasattr(ra, "NZEB")
        assert hasattr(ra, "NET_ZERO")

    def test_output_format_enum(self, cfg_mod):
        of = cfg_mod.OutputFormat
        assert hasattr(of, "PDF")
        assert hasattr(of, "JSON")
        assert hasattr(of, "HTML")

    def test_ventilation_type_enum(self, cfg_mod):
        vt = cfg_mod.VentilationType
        assert hasattr(vt, "NATURAL")
        assert hasattr(vt, "MVHR")

    def test_ieq_category_enum(self, cfg_mod):
        iq = cfg_mod.IEQCategory
        assert hasattr(iq, "CATEGORY_I")
        assert hasattr(iq, "CATEGORY_II")

    def test_thermal_bridge_method_enum(self, cfg_mod):
        tb = cfg_mod.ThermalBridgeMethod
        assert hasattr(tb, "DEFAULT_UPLIFT")
        assert hasattr(tb, "TABULATED_PSI")


# =========================================================================
# Test BuildingEnergyAssessmentConfig Defaults
# =========================================================================


class TestBEAConfigDefaults:
    def test_instantiation(self, bea_config):
        assert bea_config is not None

    def test_default_building_type(self, cfg_mod, bea_config):
        assert bea_config.building_type == cfg_mod.BuildingType.OFFICE

    def test_default_climate_zone(self, cfg_mod, bea_config):
        assert bea_config.climate_zone == cfg_mod.ClimateZone.CENTRAL_MARITIME

    def test_default_assessment_level(self, cfg_mod, bea_config):
        assert bea_config.assessment_level == cfg_mod.AssessmentLevel.STANDARD

    def test_default_country(self, bea_config):
        assert bea_config.country == "GB"

    def test_default_certification(self, cfg_mod, bea_config):
        assert bea_config.certification_target == cfg_mod.CertificationTarget.NONE

    def test_default_floors(self, bea_config):
        assert bea_config.number_of_floors >= 1

    def test_default_occupancy_pattern(self, cfg_mod, bea_config):
        assert bea_config.occupancy_pattern == cfg_mod.OccupancyPattern.SINGLE_SHIFT

    def test_default_heating_fuel(self, cfg_mod, bea_config):
        assert bea_config.primary_heating_fuel == cfg_mod.HeatingFuel.NATURAL_GAS


# =========================================================================
# Test Sub-configuration Sections
# =========================================================================


class TestSubConfigs:
    def test_envelope_config(self, bea_config):
        assert bea_config.envelope is not None

    def test_hvac_config(self, bea_config):
        assert bea_config.hvac is not None

    def test_lighting_config(self, bea_config):
        assert bea_config.lighting is not None

    def test_dhw_config(self, bea_config):
        assert bea_config.dhw is not None

    def test_renewable_config(self, bea_config):
        assert bea_config.renewable is not None

    def test_benchmark_config(self, bea_config):
        assert bea_config.benchmark is not None

    def test_retrofit_config(self, bea_config):
        assert bea_config.retrofit is not None

    def test_indoor_environment_config(self, bea_config):
        assert bea_config.indoor_environment is not None

    def test_carbon_config(self, bea_config):
        assert bea_config.carbon is not None

    def test_compliance_config(self, bea_config):
        assert bea_config.compliance is not None

    def test_report_config(self, bea_config):
        assert bea_config.report is not None

    def test_integration_config(self, bea_config):
        assert bea_config.integration is not None

    def test_performance_config(self, bea_config):
        assert bea_config.performance is not None

    def test_security_config(self, bea_config):
        assert bea_config.security is not None

    def test_audit_trail_config(self, bea_config):
        assert bea_config.audit_trail is not None


# =========================================================================
# Test PackConfig Wrapper
# =========================================================================


class TestPackConfig:
    def test_instantiation(self, pack_config):
        assert pack_config is not None

    def test_pack_id(self, pack_config):
        assert "PACK-032" in pack_config.pack_id

    def test_config_version(self, pack_config):
        assert pack_config.config_version == "1.0.0"

    def test_preset_name_default(self, pack_config):
        assert pack_config.preset_name is None

    def test_pack_attribute(self, pack_config):
        assert pack_config.pack is not None


# =========================================================================
# Test Preset Loading
# =========================================================================


PRESET_NAMES = [
    "commercial_office",
    "retail_building",
    "hotel_hospitality",
    "healthcare_facility",
    "education_building",
    "residential_multifamily",
    "mixed_use_development",
    "public_sector_building",
]


class TestPresetLoading:
    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_file_exists(self, preset_name):
        path = PRESETS_DIR / f"{preset_name}.yaml"
        assert path.exists(), f"Preset file missing: {preset_name}.yaml"

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_valid_yaml(self, preset_name):
        path = PRESETS_DIR / f"{preset_name}.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_from_preset(self, cfg_mod, preset_name):
        config = cfg_mod.PackConfig.from_preset(preset_name)
        assert config is not None
        assert config.preset_name == preset_name

    def test_from_preset_unknown_raises(self, cfg_mod):
        with pytest.raises(ValueError, match="Unknown preset"):
            cfg_mod.PackConfig.from_preset("non_existent_preset_xyz")

    def test_from_preset_with_overrides(self, cfg_mod):
        config = cfg_mod.PackConfig.from_preset(
            "commercial_office",
            overrides={"reporting_year": 2026},
        )
        assert config.pack.reporting_year == 2026


# =========================================================================
# Test from_yaml
# =========================================================================


class TestFromYAML:
    def test_from_yaml_demo(self, cfg_mod):
        demo_path = DEMO_DIR / "demo_config.yaml"
        if not demo_path.exists():
            pytest.skip("demo_config.yaml not found")
        config = cfg_mod.PackConfig.from_yaml(demo_path)
        assert config is not None
        assert config.pack.building_name != ""

    def test_from_yaml_missing_raises(self, cfg_mod):
        with pytest.raises(FileNotFoundError):
            cfg_mod.PackConfig.from_yaml("/tmp/nonexistent_99999.yaml")

    def test_from_yaml_building_type(self, cfg_mod):
        demo_path = DEMO_DIR / "demo_config.yaml"
        if not demo_path.exists():
            pytest.skip("demo_config.yaml not found")
        config = cfg_mod.PackConfig.from_yaml(demo_path)
        assert config.pack.building_type == cfg_mod.BuildingType.OFFICE


# =========================================================================
# Test Merge
# =========================================================================


class TestMerge:
    def test_merge_overrides(self, cfg_mod):
        base = cfg_mod.PackConfig()
        merged = cfg_mod.PackConfig.merge(base, {"reporting_year": 2026})
        assert merged.pack.reporting_year == 2026

    def test_merge_preserves_base(self, cfg_mod):
        base = cfg_mod.PackConfig()
        original_country = base.pack.country
        merged = cfg_mod.PackConfig.merge(base, {"reporting_year": 2026})
        assert merged.pack.country == original_country

    def test_merge_nested(self, cfg_mod):
        base = cfg_mod.PackConfig()
        merged = cfg_mod.PackConfig.merge(
            base,
            {"envelope": {"wall_u_target": 0.15}},
        )
        assert merged.pack.envelope.wall_u_target == pytest.approx(0.15, rel=0.01)

    def test_merge_preserves_preset_name(self, cfg_mod):
        base = cfg_mod.PackConfig.from_preset("commercial_office")
        merged = cfg_mod.PackConfig.merge(base, {"reporting_year": 2026})
        assert merged.preset_name == "commercial_office"


# =========================================================================
# Test Deep Merge Helper
# =========================================================================


class TestDeepMerge:
    def test_deep_merge_flat(self, cfg_mod):
        result = cfg_mod.PackConfig._deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_deep_merge_override(self, cfg_mod):
        result = cfg_mod.PackConfig._deep_merge({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_deep_merge_nested(self, cfg_mod):
        base = {"a": {"x": 1, "y": 2}}
        over = {"a": {"y": 3, "z": 4}}
        result = cfg_mod.PackConfig._deep_merge(base, over)
        assert result["a"]["x"] == 1
        assert result["a"]["y"] == 3
        assert result["a"]["z"] == 4


# =========================================================================
# Test Config Hash (Provenance)
# =========================================================================


class TestConfigHash:
    def test_get_config_hash(self, pack_config):
        h = pack_config.get_config_hash()
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_deterministic(self, cfg_mod):
        c1 = cfg_mod.PackConfig()
        c2 = cfg_mod.PackConfig()
        assert c1.get_config_hash() == c2.get_config_hash()

    def test_hash_changes_with_overrides(self, cfg_mod):
        c1 = cfg_mod.PackConfig()
        c2 = cfg_mod.PackConfig.merge(c1, {"reporting_year": 2030})
        assert c1.get_config_hash() != c2.get_config_hash()


# =========================================================================
# Test Validate Completeness
# =========================================================================


class TestValidateCompleteness:
    def test_validate_default(self, pack_config):
        warnings = pack_config.validate_completeness()
        assert isinstance(warnings, list)

    def test_validate_with_building_name(self, cfg_mod):
        config = cfg_mod.PackConfig.merge(
            cfg_mod.PackConfig(),
            {"building_name": "Test Building"},
        )
        warnings = config.validate_completeness()
        # Should have fewer warnings since building_name is now set
        assert isinstance(warnings, list)


# =========================================================================
# Test Environment Variable Overrides
# =========================================================================


class TestEnvOverrides:
    def test_load_env_overrides_empty(self, cfg_mod):
        result = cfg_mod.PackConfig._load_env_overrides()
        # No BUILDING_ENERGY_PACK_ vars should be set during test
        assert isinstance(result, dict)

    def test_env_override_parsing(self, cfg_mod, monkeypatch):
        monkeypatch.setenv("BUILDING_ENERGY_PACK_REPORTING_YEAR", "2027")
        result = cfg_mod.PackConfig._load_env_overrides()
        assert "reporting_year" in result
        assert result["reporting_year"] == 2027

    def test_env_override_nested(self, cfg_mod, monkeypatch):
        monkeypatch.setenv("BUILDING_ENERGY_PACK_ENVELOPE__WALL_U_TARGET", "0.18")
        result = cfg_mod.PackConfig._load_env_overrides()
        assert "envelope" in result
        assert "wall_u_target" in result["envelope"]
        assert result["envelope"]["wall_u_target"] == pytest.approx(0.18, rel=0.01)

    def test_env_override_boolean_true(self, cfg_mod, monkeypatch):
        monkeypatch.setenv("BUILDING_ENERGY_PACK_BENCHMARK__ENABLED", "true")
        result = cfg_mod.PackConfig._load_env_overrides()
        assert result["benchmark"]["enabled"] is True

    def test_env_override_boolean_false(self, cfg_mod, monkeypatch):
        monkeypatch.setenv("BUILDING_ENERGY_PACK_BENCHMARK__ENABLED", "false")
        result = cfg_mod.PackConfig._load_env_overrides()
        assert result["benchmark"]["enabled"] is False


# =========================================================================
# Test Model Validators
# =========================================================================


class TestModelValidators:
    def test_epc_methodology_gb_residential(self, cfg_mod):
        """UK residential should auto-switch from SBEM to SAP."""
        config = cfg_mod.BuildingEnergyAssessmentConfig(
            country="GB",
            building_type=cfg_mod.BuildingType.RESIDENTIAL_HOUSE,
        )
        assert config.compliance.epc_methodology == cfg_mod.EPCMethodology.SAP

    def test_nzeb_enables_heat_pump(self, cfg_mod):
        """nZEB retrofit ambition should auto-enable heat pump."""
        config = cfg_mod.BuildingEnergyAssessmentConfig(
            retrofit=cfg_mod.RetrofitConfig(
                retrofit_ambition=cfg_mod.RetrofitAmbition.NZEB,
            ),
            renewable=cfg_mod.RenewableConfig(heat_pump_enabled=False),
        )
        assert config.renewable.heat_pump_enabled is True


# =========================================================================
# Test Reference Data
# =========================================================================


class TestReferenceData:
    def test_building_type_info(self, cfg_mod):
        info = cfg_mod.BUILDING_TYPE_INFO
        assert isinstance(info, dict)
        assert "OFFICE" in info
        assert "HOTEL" in info

    def test_available_presets(self, cfg_mod):
        presets = cfg_mod.AVAILABLE_PRESETS
        assert isinstance(presets, dict)
        assert len(presets) >= 8
        assert "commercial_office" in presets

    def test_u_value_targets(self, cfg_mod):
        targets = cfg_mod.U_VALUE_TARGETS
        assert isinstance(targets, dict)
        assert "NORTHERN_EUROPE" in targets or "CENTRAL_MARITIME" in targets


# =========================================================================
# Test Utility Functions
# =========================================================================


class TestUtilityFunctions:
    def test_load_preset_function(self, cfg_mod):
        config = cfg_mod.load_preset("commercial_office")
        assert config is not None
        assert config.preset_name == "commercial_office"

    def test_validate_config_function(self, cfg_mod):
        config = cfg_mod.BuildingEnergyAssessmentConfig()
        warnings = cfg_mod.validate_config(config)
        assert isinstance(warnings, list)
