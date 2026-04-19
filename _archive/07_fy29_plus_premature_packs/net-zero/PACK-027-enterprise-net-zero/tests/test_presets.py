# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - Presets.

Tests all 8 sector-specific enterprise presets: manufacturing, energy/utilities,
financial services, technology, retail/consumer goods, transport/logistics,
healthcare, and agriculture/food.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~50 tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

PRESETS_DIR = _PACK_ROOT / "config" / "presets"


# Actual preset filenames (without .yaml extension)
PRESET_FILES = [
    "manufacturing",
    "energy_utilities",
    "financial_services",
    "technology",
    "retail_consumer",
    "transport_logistics",
    "healthcare",
    "agriculture_food",
]


class TestPresetFilesExist:
    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_preset_yaml_exists(self, preset_name):
        path = PRESETS_DIR / f"{preset_name}.yaml"
        assert path.exists(), f"Preset file missing: {path}"

    @pytest.mark.parametrize("preset_name", PRESET_FILES)
    def test_preset_valid_yaml(self, preset_name):
        path = PRESETS_DIR / f"{preset_name}.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0


class TestManufacturingPreset:
    def test_sector_manufacturing(self):
        path = PRESETS_DIR / "manufacturing.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("organization", {}).get("sector") == "MANUFACTURING"

    def test_sbti_mixed_pathway(self):
        path = PRESETS_DIR / "manufacturing.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("target", {}).get("sbti_pathway") == "MIXED"

    def test_process_emissions_enabled(self):
        path = PRESETS_DIR / "manufacturing.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        scope1 = data.get("scope", {}).get("scope1_agents", [])
        assert "MRV-004" in scope1  # Process emissions

    def test_cbam_enabled(self):
        path = PRESETS_DIR / "manufacturing.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("carbon_pricing", {}).get("cbam_enabled") is True


class TestEnergyUtilitiesPreset:
    def test_sector_energy(self):
        path = PRESETS_DIR / "energy_utilities.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("organization", {}).get("sector") == "ENERGY_UTILITIES"

    def test_sda_pathway_mandatory(self):
        path = PRESETS_DIR / "energy_utilities.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("target", {}).get("sbti_pathway") == "SDA"

    def test_stranded_asset_analysis(self):
        path = PRESETS_DIR / "energy_utilities.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("scenarios", {}).get("stranded_asset_analysis") is True

    def test_scope4_enabled(self):
        path = PRESETS_DIR / "energy_utilities.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("scope4", {}).get("enabled") is True


class TestFinancialServicesPreset:
    def test_sector_financial(self):
        path = PRESETS_DIR / "financial_services.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("organization", {}).get("sector") == "FINANCIAL_SERVICES"

    def test_scope3_priority_includes_cat15(self):
        path = PRESETS_DIR / "financial_services.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        priority = data.get("scope", {}).get("scope3_priority_categories", [])
        assert 15 in priority


class TestTechnologyPreset:
    def test_sector_technology(self):
        path = PRESETS_DIR / "technology.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("organization", {}).get("sector") == "TECHNOLOGY"


class TestRetailConsumerPreset:
    def test_flag_enabled(self):
        path = PRESETS_DIR / "retail_consumer.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("target", {}).get("flag_enabled") is True

    def test_tier_depth_5(self):
        path = PRESETS_DIR / "retail_consumer.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("supply_chain", {}).get("tier_depth") == 5


class TestTransportLogisticsPreset:
    def test_sda_pathway(self):
        path = PRESETS_DIR / "transport_logistics.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("target", {}).get("sbti_pathway") == "SDA"

    def test_stranded_asset_analysis(self):
        path = PRESETS_DIR / "transport_logistics.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("scenarios", {}).get("stranded_asset_analysis") is True


class TestHealthcarePreset:
    def test_sector_healthcare_pharma(self):
        path = PRESETS_DIR / "healthcare.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("organization", {}).get("sector") == "HEALTHCARE_PHARMA"

    def test_scope1_includes_process_emissions(self):
        """Healthcare must include process emissions for API manufacturing."""
        path = PRESETS_DIR / "healthcare.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        scope1 = data.get("scope", {}).get("scope1_agents", [])
        assert "MRV-004" in scope1  # Process emissions
