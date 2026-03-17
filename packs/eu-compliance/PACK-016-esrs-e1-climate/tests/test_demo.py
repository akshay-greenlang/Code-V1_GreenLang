# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Demo Configuration Tests
==========================================================

Tests for demo configuration files, preset YAML files, and their
compatibility with the E1ClimateConfig Pydantic model.

Target: 22+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
Date:    March 2026
"""

from pathlib import Path

import pytest
import yaml

from .conftest import (
    PRESETS_DIR,
    DEMO_DIR,
    PRESET_NAMES,
    _load_config_module,
)


# ===========================================================================
# Demo Configuration File Tests
# ===========================================================================


class TestDemoConfigFile:
    """Tests for the demo_config.yaml file."""

    def test_demo_config_exists(self):
        """demo_config.yaml file exists."""
        path = DEMO_DIR / "demo_config.yaml"
        assert path.exists(), f"Demo config not found: {path}"

    def test_demo_config_parseable(self):
        """demo_config.yaml parses to a non-empty dictionary."""
        path = DEMO_DIR / "demo_config.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_demo_config_valid(self):
        """demo_config.yaml is valid as E1ClimateConfig input."""
        cfg = _load_config_module()
        path = DEMO_DIR / "demo_config.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        config = cfg.E1ClimateConfig(**data)
        assert config.company_name == "NordEnergy AG"

    def test_demo_config_loads_as_e1_config(self, demo_config):
        """Demo YAML loads as a valid E1ClimateConfig instance."""
        cfg = _load_config_module()
        assert isinstance(demo_config, cfg.E1ClimateConfig)

    def test_demo_config_company_name(self, demo_yaml_data):
        """Demo config has a company name."""
        assert "company_name" in demo_yaml_data
        assert len(demo_yaml_data["company_name"]) > 0

    def test_demo_config_reporting_year(self, demo_yaml_data):
        """Demo config has a valid reporting year."""
        year = demo_yaml_data.get("reporting_year", 0)
        assert 2024 <= year <= 2035


# ===========================================================================
# Demo Value Tests
# ===========================================================================


class TestDemoValues:
    """Tests for specific demo configuration values."""

    def test_demo_company_name_nordenergy(self, demo_yaml_data):
        """Demo company is NordEnergy AG."""
        assert demo_yaml_data["company_name"] == "NordEnergy AG"

    def test_demo_sector_energy(self, demo_yaml_data):
        """Demo sector is ENERGY."""
        assert demo_yaml_data.get("sector") == "ENERGY"

    def test_demo_reporting_year_2025(self, demo_yaml_data):
        """Demo reporting year is 2025."""
        assert demo_yaml_data.get("reporting_year") == 2025

    def test_demo_currency_eur(self, demo_yaml_data):
        """Demo currency is EUR."""
        assert demo_yaml_data.get("currency") == "EUR"

    def test_demo_fiscal_year_end(self, demo_yaml_data):
        """Demo fiscal year end is 12-31."""
        assert demo_yaml_data.get("fiscal_year_end") == "12-31"


# ===========================================================================
# Demo GHG Configuration Tests
# ===========================================================================


class TestDemoGHG:
    """Tests for GHG scope configuration in demo."""

    def test_demo_ghg_section_present(self, demo_yaml_data):
        """Demo has a ghg section."""
        assert "ghg" in demo_yaml_data

    def test_demo_ghg_enabled(self, demo_yaml_data):
        """GHG inventory is enabled in demo."""
        assert demo_yaml_data["ghg"]["enabled"] is True

    def test_demo_ghg_base_year_2020(self, demo_yaml_data):
        """Demo GHG base year is 2020."""
        assert demo_yaml_data["ghg"]["base_year"] == 2020

    def test_demo_ghg_scopes_enabled(self, demo_yaml_data):
        """Demo enables multiple GHG scopes."""
        scopes = demo_yaml_data["ghg"].get("scopes_enabled", [])
        assert len(scopes) >= 3


# ===========================================================================
# Demo Energy Configuration Tests
# ===========================================================================


class TestDemoEnergy:
    """Tests for energy source configuration in demo."""

    def test_demo_energy_section_present(self, demo_yaml_data):
        """Demo has an energy section."""
        assert "energy" in demo_yaml_data

    def test_demo_energy_enabled(self, demo_yaml_data):
        """Energy tracking is enabled in demo."""
        assert demo_yaml_data["energy"]["enabled"] is True

    def test_demo_energy_renewables_enabled(self, demo_yaml_data):
        """Renewable tracking is enabled in demo."""
        assert demo_yaml_data["energy"]["include_renewables"] is True


# ===========================================================================
# Demo Target Configuration Tests
# ===========================================================================


class TestDemoTargets:
    """Tests for target configuration in demo."""

    def test_demo_targets_section_present(self, demo_yaml_data):
        """Demo has a targets section."""
        assert "targets" in demo_yaml_data

    def test_demo_targets_enabled(self, demo_yaml_data):
        """Target tracking is enabled in demo."""
        assert demo_yaml_data["targets"]["enabled"] is True

    def test_demo_targets_sbti_1_5c(self, demo_yaml_data):
        """Demo targets are aligned to SBTi 1.5C."""
        assert demo_yaml_data["targets"]["sbti_commitment_level"] == "SBTi_1_5C"

    def test_demo_targets_year_2030(self, demo_yaml_data):
        """Demo target year is 2030."""
        assert demo_yaml_data["targets"]["target_year"] == 2030


# ===========================================================================
# Demo Risk Configuration Tests
# ===========================================================================


class TestDemoRisk:
    """Tests for risk configuration in demo."""

    def test_demo_risk_section_present(self, demo_yaml_data):
        """Demo has a climate_risk section."""
        assert "climate_risk" in demo_yaml_data

    def test_demo_risk_enabled(self, demo_yaml_data):
        """Climate risk assessment is enabled in demo."""
        assert demo_yaml_data["climate_risk"]["enabled"] is True

    def test_demo_risk_tcfd_aligned(self, demo_yaml_data):
        """Demo risk config has TCFD alignment enabled."""
        assert demo_yaml_data["climate_risk"]["tcfd_alignment"] is True

    def test_demo_risk_has_scenarios(self, demo_yaml_data):
        """Demo risk config has climate scenarios."""
        scenarios = demo_yaml_data["climate_risk"].get("scenarios", [])
        assert len(scenarios) >= 2


# ===========================================================================
# Preset File Tests
# ===========================================================================


class TestPresetFiles:
    """Tests for all preset YAML files."""

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_file_exists(self, preset_name):
        """Preset YAML file exists on disk."""
        path = PRESETS_DIR / f"{preset_name}.yaml"
        assert path.exists(), f"Preset file not found: {path}"

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_parseable(self, preset_name):
        """Preset YAML file parses to a non-empty dictionary."""
        path = PRESETS_DIR / f"{preset_name}.yaml"
        if not path.exists():
            pytest.skip(f"Preset not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_valid_as_e1_config(self, preset_name):
        """Preset YAML is valid as E1ClimateConfig input."""
        path = PRESETS_DIR / f"{preset_name}.yaml"
        if not path.exists():
            pytest.skip(f"Preset not found: {path}")
        cfg = _load_config_module()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        config = cfg.E1ClimateConfig(**data)
        assert config is not None
