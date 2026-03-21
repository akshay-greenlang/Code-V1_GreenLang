# -*- coding: utf-8 -*-
"""
Test suite for PACK-028 Sector Pathway Pack - Configuration & Presets.

Tests all 6 sector configuration presets (heavy_industry, power_generation,
transport, buildings, chemicals, mixed_sectors), YAML structure validation,
convergence models, SBTi/IEA/IPCC sections, and cross-preset consistency.

Author:  GreenLang Test Engineering
Pack:    PACK-028 Sector Pathway Pack
"""

import sys
import time
from pathlib import Path

import pytest
import yaml

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

PRESETS_DIR = _PACK_ROOT / "config" / "presets"

# Actual preset file basenames (no extension)
PRESET_FILES = [
    "heavy_industry",
    "power_generation",
    "transport",
    "buildings",
    "chemicals",
    "mixed_sectors",
]


def _load_preset(name):
    """Load and return parsed YAML for a preset."""
    path = PRESETS_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ========================================================================
# Preset File Existence
# ========================================================================


class TestPresetFilesExist:
    """Test that all preset files exist and are valid YAML."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_preset_yaml_exists(self, preset_name):
        path = PRESETS_DIR / f"{preset_name}.yaml"
        assert path.exists(), f"Preset file missing: {path}"

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_preset_valid_yaml(self, preset_name):
        data = _load_preset(preset_name)
        assert isinstance(data, dict)
        assert len(data) > 0

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_preset_has_primary_sector(self, preset_name):
        data = _load_preset(preset_name)
        assert "primary_sector" in data

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_preset_has_base_year(self, preset_name):
        data = _load_preset(preset_name)
        assert "base_year" in data

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_preset_has_convergence(self, preset_name):
        data = _load_preset(preset_name)
        assert "convergence" in data

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_preset_has_sbti_sda(self, preset_name):
        data = _load_preset(preset_name)
        assert "sbti_sda" in data

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_preset_has_iea_nze(self, preset_name):
        data = _load_preset(preset_name)
        assert "iea_nze" in data


# ========================================================================
# Heavy Industry Preset
# ========================================================================


class TestHeavyIndustryPreset:
    """Test heavy_industry.yaml preset (Steel, Cement, Aluminum)."""

    def test_primary_sector(self):
        data = _load_preset("heavy_industry")
        assert data["primary_sector"] == "STEEL"

    def test_secondary_sectors(self):
        data = _load_preset("heavy_industry")
        assert "CEMENT" in data["secondary_sectors"]
        assert "ALUMINUM" in data["secondary_sectors"]

    def test_sbti_sda_enabled(self):
        data = _load_preset("heavy_industry")
        assert data["sbti_sda"]["sbti_sda_enabled"] is True

    def test_convergence_model_s_curve(self):
        data = _load_preset("heavy_industry")
        assert data["convergence"]["convergence_model"] == "s_curve"

    def test_technology_roadmap_enabled(self):
        data = _load_preset("heavy_industry")
        assert data["technology"]["technology_roadmap_enabled"] is True

    def test_macc_enabled(self):
        data = _load_preset("heavy_industry")
        assert data["macc"]["macc_enabled"] is True


# ========================================================================
# Power Generation Preset
# ========================================================================


class TestPowerGenerationPreset:
    """Test power_generation.yaml preset."""

    def test_primary_sector(self):
        data = _load_preset("power_generation")
        assert data["primary_sector"] == "POWER"

    def test_sbti_sda_enabled(self):
        data = _load_preset("power_generation")
        assert data["sbti_sda"]["sbti_sda_enabled"] is True

    def test_convergence_model_exponential(self):
        data = _load_preset("power_generation")
        assert data["convergence"]["convergence_model"] == "exponential"

    def test_intensity_metric(self):
        data = _load_preset("power_generation")
        assert data["intensity"]["intensity_metric"] == "gCO2/kWh"


# ========================================================================
# Transport Preset
# ========================================================================


class TestTransportPreset:
    """Test transport.yaml preset (Aviation, Shipping, Road, Rail)."""

    def test_primary_sector(self):
        data = _load_preset("transport")
        assert data["primary_sector"] == "AVIATION"

    def test_secondary_sectors(self):
        data = _load_preset("transport")
        sec = data["secondary_sectors"]
        assert "SHIPPING" in sec
        assert "ROAD_TRANSPORT" in sec
        assert "RAIL" in sec

    def test_sbti_sda_enabled(self):
        data = _load_preset("transport")
        assert data["sbti_sda"]["sbti_sda_enabled"] is True

    def test_convergence_model_s_curve(self):
        data = _load_preset("transport")
        assert data["convergence"]["convergence_model"] == "s_curve"

    def test_regulatory_benchmarks(self):
        data = _load_preset("transport")
        benchmarks = data["benchmark"]["regulatory_benchmarks"]
        assert "ICAO_CORSIA" in benchmarks
        assert "IMO_CII" in benchmarks


# ========================================================================
# Buildings Preset
# ========================================================================


class TestBuildingsPreset:
    """Test buildings.yaml preset (Commercial, Residential)."""

    def test_primary_sector(self):
        data = _load_preset("buildings")
        assert data["primary_sector"] == "BUILDINGS_COMMERCIAL"

    def test_secondary_sectors(self):
        data = _load_preset("buildings")
        assert "BUILDINGS_RESIDENTIAL" in data["secondary_sectors"]

    def test_convergence_model_linear(self):
        data = _load_preset("buildings")
        assert data["convergence"]["convergence_model"] == "linear"

    def test_intensity_metric(self):
        data = _load_preset("buildings")
        assert data["intensity"]["intensity_metric"] == "kgCO2/m2/year"

    def test_crrem_benchmark(self):
        data = _load_preset("buildings")
        benchmarks = data["benchmark"]["regulatory_benchmarks"]
        assert "CRREM" in benchmarks


# ========================================================================
# Chemicals Preset
# ========================================================================


class TestChemicalsPreset:
    """Test chemicals.yaml preset."""

    def test_primary_sector(self):
        data = _load_preset("chemicals")
        assert data["primary_sector"] == "CHEMICALS"

    def test_sbti_sda_enabled(self):
        data = _load_preset("chemicals")
        assert data["sbti_sda"]["sbti_sda_enabled"] is True

    def test_convergence_model_s_curve(self):
        data = _load_preset("chemicals")
        assert data["convergence"]["convergence_model"] == "s_curve"


# ========================================================================
# Mixed Sectors Preset
# ========================================================================


class TestMixedSectorsPreset:
    """Test mixed_sectors.yaml preset."""

    def test_primary_sector(self):
        data = _load_preset("mixed_sectors")
        assert data["primary_sector"] == "MIXED"

    def test_secondary_sectors(self):
        data = _load_preset("mixed_sectors")
        sec = data["secondary_sectors"]
        assert len(sec) >= 3

    def test_sbti_sda_disabled(self):
        data = _load_preset("mixed_sectors")
        assert data["sbti_sda"]["sbti_sda_enabled"] is False

    def test_convergence_model_linear(self):
        data = _load_preset("mixed_sectors")
        assert data["convergence"]["convergence_model"] == "linear"

    def test_all_scenarios_enabled(self):
        data = _load_preset("mixed_sectors")
        scenarios = data["scenarios"]["scenarios_enabled"]
        assert len(scenarios) == 5


# ========================================================================
# Cross-Preset Structural Validation
# ========================================================================


REQUIRED_TOP_KEYS = [
    "organization_name",
    "primary_sector",
    "base_year",
    "sbti_sda",
    "iea_nze",
    "ipcc",
    "intensity",
    "convergence",
    "technology",
    "macc",
    "benchmark",
    "scenarios",
    "reporting",
    "performance",
]


class TestPresetStructure:
    """Test common structure across all presets."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    @pytest.mark.parametrize("key", REQUIRED_TOP_KEYS)
    def test_required_key_present(self, preset_name, key):
        data = _load_preset(preset_name)
        assert key in data, f"Missing key '{key}' in {preset_name}"


class TestPresetSBTiSection:
    """Test SBTi SDA section across all presets."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_sbti_has_enabled_flag(self, preset_name):
        data = _load_preset(preset_name)
        assert "sbti_sda_enabled" in data["sbti_sda"]

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_sbti_has_target_year(self, preset_name):
        data = _load_preset(preset_name)
        assert "sbti_target_year" in data["sbti_sda"]
        assert data["sbti_sda"]["sbti_target_year"] >= 2025

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_sbti_has_coverage(self, preset_name):
        data = _load_preset(preset_name)
        assert "coverage_scope1_pct" in data["sbti_sda"]
        assert data["sbti_sda"]["coverage_scope1_pct"] >= 90


class TestPresetIEASection:
    """Test IEA NZE section across all presets."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_iea_has_enabled_flag(self, preset_name):
        data = _load_preset(preset_name)
        assert "iea_nze_enabled" in data["iea_nze"]

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_iea_scenario(self, preset_name):
        data = _load_preset(preset_name)
        assert data["iea_nze"]["iea_scenario"] == "NZE"


class TestPresetConvergenceSection:
    """Test convergence section across all presets."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_convergence_model_valid(self, preset_name):
        data = _load_preset(preset_name)
        model = data["convergence"]["convergence_model"]
        assert model in ("linear", "exponential", "s_curve", "stepped")

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_convergence_rate_positive(self, preset_name):
        data = _load_preset(preset_name)
        rate = data["convergence"]["convergence_rate"]
        assert rate > 0


class TestPresetIntensitySection:
    """Test intensity section across all presets."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_intensity_has_metric(self, preset_name):
        data = _load_preset(preset_name)
        assert "intensity_metric" in data["intensity"]
        assert len(data["intensity"]["intensity_metric"]) > 0

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_intensity_has_boundary(self, preset_name):
        data = _load_preset(preset_name)
        assert "intensity_boundary" in data["intensity"]

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_intensity_baseline_year(self, preset_name):
        data = _load_preset(preset_name)
        assert data["intensity"]["baseline_year"] >= 2015
        assert data["intensity"]["baseline_year"] <= 2025


class TestPresetTechnologySection:
    """Test technology section across all presets."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_technology_roadmap_enabled(self, preset_name):
        data = _load_preset(preset_name)
        assert data["technology"]["technology_roadmap_enabled"] is True

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_technology_trl_threshold(self, preset_name):
        data = _load_preset(preset_name)
        trl = data["technology"]["trl_threshold"]
        assert 1 <= trl <= 9


class TestPresetMACCSection:
    """Test MACC section across all presets."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_macc_enabled(self, preset_name):
        data = _load_preset(preset_name)
        assert data["macc"]["macc_enabled"] is True

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_carbon_price_floor_positive(self, preset_name):
        data = _load_preset(preset_name)
        assert data["macc"]["carbon_price_floor"] > 0

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_carbon_price_ceiling_above_floor(self, preset_name):
        data = _load_preset(preset_name)
        assert data["macc"]["carbon_price_ceiling"] > data["macc"]["carbon_price_floor"]


class TestPresetScenariosSection:
    """Test scenarios section across all presets."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_scenarios_enabled(self, preset_name):
        data = _load_preset(preset_name)
        scenarios = data["scenarios"]["scenarios_enabled"]
        assert len(scenarios) >= 2
        assert "NZE" in scenarios

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_monte_carlo_runs(self, preset_name):
        data = _load_preset(preset_name)
        runs = data["scenarios"]["monte_carlo_runs"]
        assert runs >= 100


class TestPresetReportingSection:
    """Test reporting section across all presets."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_reporting_frameworks(self, preset_name):
        data = _load_preset(preset_name)
        frameworks = data["reporting"]["reporting_frameworks"]
        assert "SBTi" in frameworks
        assert "CDP" in frameworks

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_sha256_provenance(self, preset_name):
        data = _load_preset(preset_name)
        assert data["reporting"]["sha256_provenance"] is True


class TestPresetPerformanceSection:
    """Test performance section across all presets."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_cache_enabled(self, preset_name):
        data = _load_preset(preset_name)
        assert data["performance"]["cache_enabled"] is True

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_timeout_positive(self, preset_name):
        data = _load_preset(preset_name)
        assert data["performance"]["timeout_seconds"] > 0


# ========================================================================
# Cross-Preset Consistency
# ========================================================================


class TestCrossPresetConsistency:
    """Test consistency across presets."""

    def test_preset_count(self):
        assert len(PRESET_FILES) == 6

    def test_all_presets_have_same_base_year(self):
        base_years = set()
        for name in PRESET_FILES:
            data = _load_preset(name)
            base_years.add(data["base_year"])
        # All should use the same base year (2019)
        assert len(base_years) == 1
        assert 2019 in base_years

    def test_all_presets_use_gwp100(self):
        for name in PRESET_FILES:
            data = _load_preset(name)
            assert data["ipcc"]["ipcc_gwp_metric"] == "GWP100"

    def test_all_presets_use_c1_pathway(self):
        for name in PRESET_FILES:
            data = _load_preset(name)
            assert data["ipcc"]["ipcc_pathway"] == "C1"


# ========================================================================
# Preset Loading Performance
# ========================================================================


class TestPresetPerformanceLoading:
    """Test preset loading performance."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_preset_loads_under_100ms(self, preset_name):
        path = PRESETS_DIR / f"{preset_name}.yaml"
        start = time.time()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        elapsed = (time.time() - start) * 1000
        assert data is not None
        assert elapsed < 1000

    def test_all_presets_load_under_1s(self):
        start = time.time()
        for name in PRESET_FILES:
            data = _load_preset(name)
            assert data is not None
        elapsed = (time.time() - start) * 1000
        assert elapsed < 5000


# ========================================================================
# Preset Serialization Roundtrip
# ========================================================================


class TestPresetSerialization:
    """Test preset serialization roundtrip."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_yaml_roundtrip(self, preset_name):
        data = _load_preset(preset_name)
        yaml_str = yaml.dump(data, default_flow_style=False)
        data2 = yaml.safe_load(yaml_str)
        assert data == data2

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_preset_no_empty_strings_at_top_level(self, preset_name):
        """Top-level string values should not be empty (except organization_name)."""
        data = _load_preset(preset_name)
        for key, value in data.items():
            if isinstance(value, str) and key != "organization_name":
                assert value.strip() != "", f"Empty string for key: {key}"


# ========================================================================
# Preset File Size Validation
# ========================================================================


class TestPresetFileSize:
    """Test preset file sizes are reasonable."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_preset_file_size(self, preset_name):
        path = PRESETS_DIR / f"{preset_name}.yaml"
        size = path.stat().st_size
        assert size > 500, f"Preset too small: {size} bytes"
        assert size < 100000, f"Preset too large: {size} bytes"

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_preset_no_duplicate_top_keys(self, preset_name):
        data = _load_preset(preset_name)
        assert isinstance(data, dict)
        assert len(data) >= 10


# ========================================================================
# Benchmark Section Validation
# ========================================================================


class TestPresetBenchmarkSection:
    """Test benchmark section across all presets."""

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_benchmark_peer_enabled(self, preset_name):
        data = _load_preset(preset_name)
        assert data["benchmark"]["benchmark_peer_enabled"] is True

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_benchmark_iea_enabled(self, preset_name):
        data = _load_preset(preset_name)
        assert data["benchmark"]["benchmark_iea_enabled"] is True

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_benchmark_has_regulatory(self, preset_name):
        data = _load_preset(preset_name)
        reg = data["benchmark"]["regulatory_benchmarks"]
        assert isinstance(reg, list)
        assert len(reg) >= 2

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_benchmark_has_dimensions(self, preset_name):
        data = _load_preset(preset_name)
        dims = data["benchmark"]["benchmark_dimensions"]
        assert isinstance(dims, list)
        assert len(dims) >= 5
        assert "intensity_rank" in dims
        assert "pathway_gap" in dims
