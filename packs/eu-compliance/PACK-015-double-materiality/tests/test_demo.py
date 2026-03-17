# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Demo Configuration Tests
========================================================================

Tests for demo configuration files, preset YAML files, and their
compatibility with the DMAConfig Pydantic model.

Target: 20+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
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
        """demo_config.yaml is valid as DMAConfig input."""
        cfg = _load_config_module()
        path = DEMO_DIR / "demo_config.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        config = cfg.DMAConfig(**data)
        assert config.company_name == "NordTech Industries GmbH"

    def test_demo_config_loads_as_dma_config(self, demo_config):
        """Demo YAML loads as a valid DMAConfig instance."""
        cfg = _load_config_module()
        assert isinstance(demo_config, cfg.DMAConfig)

    def test_demo_config_company_name(self, demo_yaml_data):
        """Demo config has a company name."""
        assert "company_name" in demo_yaml_data
        assert len(demo_yaml_data["company_name"]) > 0

    def test_demo_config_reporting_year(self, demo_yaml_data):
        """Demo config has a valid reporting year."""
        year = demo_yaml_data.get("reporting_year", 0)
        assert 2024 <= year <= 2035

    def test_demo_config_sectors(self, demo_yaml_data):
        """Demo config lists at least one sector."""
        sectors = demo_yaml_data.get("sectors", [])
        assert len(sectors) >= 1

    def test_demo_thresholds_reasonable(self, demo_yaml_data):
        """Demo threshold values are within scoring scale."""
        threshold = demo_yaml_data.get("threshold", {})
        impact_thresh = threshold.get("impact_threshold", 5.0)
        financial_thresh = threshold.get("financial_threshold", 5.0)
        scoring_scale = demo_yaml_data.get("impact_materiality", {}).get("scoring_scale", 10)
        assert 1.0 <= impact_thresh <= scoring_scale
        assert 1.0 <= financial_thresh <= scoring_scale

    def test_demo_scoring_methodology(self, demo_yaml_data):
        """Demo config specifies a scoring methodology."""
        threshold = demo_yaml_data.get("threshold", {})
        methodology = threshold.get("methodology", "ABSOLUTE_CUTOFF")
        valid_methodologies = {
            "ABSOLUTE_CUTOFF", "PERCENTILE", "SECTOR_CALIBRATED",
            "EXPERT_JUDGMENT", "COMBINED",
        }
        assert methodology in valid_methodologies

    def test_demo_config_impact_materiality_enabled(self, demo_yaml_data):
        """Demo config has impact materiality enabled."""
        assert demo_yaml_data.get("impact_materiality", {}).get("enabled") is True

    def test_demo_config_financial_materiality_enabled(self, demo_yaml_data):
        """Demo config has financial materiality enabled."""
        assert demo_yaml_data.get("financial_materiality", {}).get("enabled") is True


# ===========================================================================
# Preset File Tests
# ===========================================================================


class TestPresetFiles:
    """Tests for preset YAML files."""

    def test_demo_preset_files_exist(self):
        """All 6 preset files exist on disk."""
        for name in PRESET_NAMES:
            path = PRESETS_DIR / f"{name}.yaml"
            assert path.exists(), f"Preset file missing: {path}"

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_parseable(self, preset_name):
        """Each preset YAML file parses to a dictionary."""
        path = PRESETS_DIR / f"{preset_name}.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_loads_as_pack_config(self, preset_name):
        """Each preset can be loaded via PackConfig.from_preset()."""
        cfg = _load_config_module()
        pc = cfg.PackConfig.from_preset(preset_name)
        assert pc.preset_name == preset_name
        assert isinstance(pc.pack, cfg.DMAConfig)

    def test_demo_large_enterprise_preset(self):
        """Large enterprise preset has 1-10 scale and multi-scorer."""
        cfg = _load_config_module()
        pc = cfg.PackConfig.from_preset("large_enterprise")
        assert pc.pack.impact_materiality.scoring_scale == 10
        assert pc.pack.impact_materiality.multi_scorer is True
        assert pc.pack.impact_materiality.sub_sub_topic_granularity is True

    def test_demo_mid_market_preset(self):
        """Mid-market preset has appropriate settings."""
        cfg = _load_config_module()
        pc = cfg.PackConfig.from_preset("mid_market")
        assert pc.pack.company_size.value == "MID_MARKET"

    def test_demo_sme_preset(self):
        """SME preset has simplified settings."""
        cfg = _load_config_module()
        pc = cfg.PackConfig.from_preset("sme")
        assert pc.pack.impact_materiality.scoring_scale == 5
        assert pc.pack.impact_materiality.multi_scorer is False

    def test_demo_financial_services_preset(self):
        """Financial services preset has sector-specific config."""
        cfg = _load_config_module()
        pc = cfg.PackConfig.from_preset("financial_services")
        assert cfg.SectorType.FINANCIAL_SERVICES in pc.pack.sectors
        assert pc.pack.financial_materiality.scenario_analysis is True
        assert pc.pack.financial_materiality.cost_of_capital_impact is True

    def test_demo_manufacturing_preset(self):
        """Manufacturing preset loads correctly."""
        cfg = _load_config_module()
        pc = cfg.PackConfig.from_preset("manufacturing")
        assert pc.preset_name == "manufacturing"

    def test_demo_multi_sector_preset(self):
        """Multi-sector preset loads correctly."""
        cfg = _load_config_module()
        pc = cfg.PackConfig.from_preset("multi_sector")
        assert pc.preset_name == "multi_sector"

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_impact_materiality_enabled(self, preset_name):
        """All presets have impact materiality enabled."""
        cfg = _load_config_module()
        pc = cfg.PackConfig.from_preset(preset_name)
        assert pc.pack.impact_materiality.enabled is True

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_all_esrs_topics_in_scope(self, preset_name):
        """All presets include all 10 ESRS topics in IRO scope."""
        cfg = _load_config_module()
        pc = cfg.PackConfig.from_preset(preset_name)
        topics = {t.value for t in pc.pack.iro_identification.esrs_topics_in_scope}
        expected = {"E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "G1"}
        assert topics == expected
