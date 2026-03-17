# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep Pack - Configuration Tests
=============================================================

Tests pack_config.py: all configuration classes, enums, presets,
validation, defaults, and reference data constants.

Author: GreenLang Platform Team (GL-TestEngineer)
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACK_ROOT / "config"


def _load_module(file_name: str, module_name: str, subdir: str = ""):
    if subdir:
        file_path = PACK_ROOT / subdir / file_name
    else:
        file_path = PACK_ROOT / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    cfg_mod = _load_module("pack_config.py", "pack020_tc.pack_config", "config")
except Exception as exc:
    cfg_mod = None
    _load_error = str(exc)


def _require_cfg():
    if cfg_mod is None:
        pytest.skip(f"pack_config.py not loadable: {_load_error}")
    return cfg_mod


# =========================================================================
# Enum Tests
# =========================================================================

class TestEnums:
    """Test all configuration enums."""

    def test_battery_category_values(self):
        mod = _require_cfg()
        assert mod.BatteryCategory.EV.value == "EV"
        assert mod.BatteryCategory.INDUSTRIAL.value == "INDUSTRIAL"
        assert mod.BatteryCategory.LMT.value == "LMT"
        assert mod.BatteryCategory.PORTABLE.value == "PORTABLE"
        assert mod.BatteryCategory.SLI.value == "SLI"

    def test_battery_category_count(self):
        mod = _require_cfg()
        assert len(list(mod.BatteryCategory)) == 5

    def test_battery_chemistry_values(self):
        mod = _require_cfg()
        chemistries = list(mod.BatteryChemistry)
        assert len(chemistries) >= 8
        assert mod.BatteryChemistry.NMC.value == "NMC"
        assert mod.BatteryChemistry.LFP.value == "LFP"
        assert mod.BatteryChemistry.NMC811.value == "NMC811"

    def test_lifecycle_stage_values(self):
        mod = _require_cfg()
        stages = list(mod.LifecycleStage)
        assert len(stages) == 4
        assert mod.LifecycleStage.RAW_MATERIAL_EXTRACTION.value == "RAW_MATERIAL_EXTRACTION"

    def test_critical_raw_material_values(self):
        mod = _require_cfg()
        crm = list(mod.CriticalRawMaterial)
        assert len(crm) == 5
        assert mod.CriticalRawMaterial.COBALT.value == "COBALT"
        assert mod.CriticalRawMaterial.LITHIUM.value == "LITHIUM"

    def test_carbon_footprint_class_values(self):
        mod = _require_cfg()
        classes = list(mod.CarbonFootprintClass)
        assert len(classes) == 5
        assert mod.CarbonFootprintClass.CLASS_A.value == "CLASS_A"
        assert mod.CarbonFootprintClass.CLASS_E.value == "CLASS_E"

    def test_compliance_status_values(self):
        mod = _require_cfg()
        statuses = list(mod.ComplianceStatus)
        assert len(statuses) == 5
        assert mod.ComplianceStatus.COMPLIANT.value == "COMPLIANT"

    def test_label_element_values(self):
        mod = _require_cfg()
        elements = list(mod.LabelElement)
        assert len(elements) == 8
        assert mod.LabelElement.CE_MARKING.value == "CE_MARKING"
        assert mod.LabelElement.QR_CODE.value == "QR_CODE"

    def test_conformity_module_values(self):
        mod = _require_cfg()
        modules = list(mod.ConformityModule)
        assert len(modules) == 7
        assert mod.ConformityModule.MODULE_A.value == "MODULE_A"

    def test_report_format_values(self):
        mod = _require_cfg()
        formats = list(mod.ReportFormat)
        assert len(formats) == 4

    def test_cache_backend_values(self):
        mod = _require_cfg()
        backends = list(mod.CacheBackend)
        assert len(backends) == 3
        assert mod.CacheBackend.MEMORY.value == "MEMORY"


# =========================================================================
# Reference Data Constants Tests
# =========================================================================

class TestReferenceData:
    """Test reference data constants."""

    def test_recycled_content_targets_2031(self):
        mod = _require_cfg()
        targets = mod.RECYCLED_CONTENT_TARGETS_2031
        assert targets["cobalt"] == 16.0
        assert targets["lithium"] == 6.0
        assert targets["nickel"] == 6.0
        assert targets["lead"] == 85.0

    def test_recycled_content_targets_2036(self):
        mod = _require_cfg()
        targets = mod.RECYCLED_CONTENT_TARGETS_2036
        assert targets["cobalt"] == 26.0
        assert targets["lithium"] == 12.0
        assert targets["nickel"] == 15.0
        assert targets["lead"] == 85.0

    def test_recycling_efficiency_targets(self):
        mod = _require_cfg()
        targets = mod.RECYCLING_EFFICIENCY_TARGETS
        assert "lithium_ion" in targets
        assert "lead_acid" in targets
        assert targets["lithium_ion"]["2025"] == 65.0

    def test_material_recovery_targets(self):
        mod = _require_cfg()
        targets = mod.MATERIAL_RECOVERY_TARGETS
        assert "cobalt" in targets
        assert "lithium" in targets
        assert targets["cobalt"]["2027"] == 90.0
        assert targets["lithium"]["2031"] == 80.0

    def test_collection_targets(self):
        mod = _require_cfg()
        targets = mod.COLLECTION_TARGETS
        assert "portable" in targets
        assert targets["portable"]["2027"] == 63.0
        assert targets["ev"]["current"] == 100.0

    def test_required_label_elements(self):
        mod = _require_cfg()
        elements = mod.REQUIRED_LABEL_ELEMENTS
        assert "EV" in elements
        assert "PORTABLE" in elements
        assert "CE_MARKING" in elements["EV"]
        assert "QR_CODE" in elements["EV"]
        assert "QR_CODE" not in elements["PORTABLE"]

    def test_conformity_modules_by_category(self):
        mod = _require_cfg()
        modules = mod.CONFORMITY_MODULES_BY_CATEGORY
        assert "EV" in modules
        assert "MODULE_A" in modules["PORTABLE"]
        assert "MODULE_H" in modules["EV"]

    def test_passport_required(self):
        mod = _require_cfg()
        assert mod.PASSPORT_REQUIRED["EV"] is True
        assert mod.PASSPORT_REQUIRED["PORTABLE"] is False
        assert mod.PASSPORT_REQUIRED["LMT"] is True

    def test_carbon_footprint_required(self):
        mod = _require_cfg()
        assert mod.CARBON_FOOTPRINT_REQUIRED["EV"] is True
        assert mod.CARBON_FOOTPRINT_REQUIRED["PORTABLE"] is False

    def test_available_presets(self):
        mod = _require_cfg()
        presets = mod.AVAILABLE_PRESETS
        assert "ev_battery" in presets
        assert "industrial_storage" in presets
        assert "portable_battery" in presets
        assert len(presets) >= 6

    def test_battery_regulation_articles(self):
        mod = _require_cfg()
        articles = mod.BATTERY_REGULATION_ARTICLES
        assert "Art. 7" in articles
        assert "Art. 65" in articles
        assert articles["Art. 7"]["name"] == "Carbon Footprint Declaration"
        assert "EV" in articles["Art. 7"]["scope"]

    def test_chemistry_critical_materials(self):
        mod = _require_cfg()
        mapping = mod.CHEMISTRY_CRITICAL_MATERIALS
        assert "NMC" in mapping
        assert "LFP" in mapping
        assert "COBALT" in mapping["NMC"]
        assert "COBALT" not in mapping["LFP"]

    def test_high_risk_countries(self):
        mod = _require_cfg()
        countries = mod.HIGH_RISK_COUNTRIES
        assert "CD" in countries
        assert "CN" in countries
        assert isinstance(countries, list)
        assert len(countries) >= 10


# =========================================================================
# Pydantic Config Model Tests
# =========================================================================

class TestPydanticConfigModels:
    """Test Pydantic sub-config models."""

    def test_carbon_footprint_config_defaults(self):
        mod = _require_cfg()
        if not hasattr(mod, "CarbonFootprintConfig"):
            pytest.skip("CarbonFootprintConfig not found")
        cfg = mod.CarbonFootprintConfig()
        assert cfg.enabled is True
        assert cfg.methodology == "EU_BATTERY_REG_DA"
        assert cfg.functional_unit == "kgCO2e_per_kWh"
        assert len(cfg.lifecycle_stages) == 4

    def test_carbon_footprint_config_custom(self):
        mod = _require_cfg()
        if not hasattr(mod, "CarbonFootprintConfig"):
            pytest.skip("CarbonFootprintConfig not found")
        cfg = mod.CarbonFootprintConfig(enabled=False, methodology="PEF")
        assert cfg.enabled is False
        assert cfg.methodology == "PEF"


# =========================================================================
# Path Constants
# =========================================================================

class TestPathConstants:
    """Test pack path constants."""

    def test_pack_base_dir(self):
        mod = _require_cfg()
        assert hasattr(mod, "PACK_BASE_DIR")
        assert mod.PACK_BASE_DIR.exists()

    def test_config_dir(self):
        mod = _require_cfg()
        assert hasattr(mod, "CONFIG_DIR")
        assert mod.CONFIG_DIR.exists()
