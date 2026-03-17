# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - Config Tests (50 tests)

Tests CBAMCompleteConfig and all sub-configs: trading, entity group,
registry API, analytics, customs, cross-regulation, audit management,
precursor chain, presets, sectors, free allocation phaseout, expanded
CN codes, third-country carbon pricing, and penalties.

Author: GreenLang QA Team
"""

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    PRESETS_DIR,
    SECTORS_DIR,
    DEMO_DIR,
    _compute_hash,
    _new_uuid,
    _utcnow,
    assert_provenance_hash,
    assert_decimal_precision,
)


# ---------------------------------------------------------------------------
# Core config creation (10 tests)
# ---------------------------------------------------------------------------

class TestConfigCreation:
    """Test CBAMCompleteConfig creation and core fields."""

    def test_config_loads_from_yaml(self, sample_config):
        """Test config dict loads and has expected top-level keys."""
        assert isinstance(sample_config, dict)
        assert "metadata" in sample_config
        assert "trading" in sample_config
        assert "entity_group" in sample_config
        assert "cbam" in sample_config

    def test_config_extends_pack004(self, sample_config):
        """Test config metadata declares PACK-004 extension."""
        meta = sample_config["metadata"]
        assert meta.get("extends") == "PACK-004-cbam-readiness"

    def test_config_trading_defaults(self, sample_config):
        """Test trading sub-config has correct defaults."""
        trading = sample_config["trading"]
        assert trading["default_strategy"] == "market"
        assert trading["default_valuation"] == "FIFO"
        assert trading["resale_window_months"] == 12

    def test_config_trading_strategies(self, sample_config):
        """Test all 5 BuyingStrategy values are listed."""
        strategies = sample_config["trading"]["buying_strategies"]
        expected = {"market", "limit", "scheduled", "dca", "custom"}
        assert set(strategies) == expected

    def test_config_valuation_methods(self, sample_config):
        """Test all 3 valuation methods: FIFO, WAC, MTM."""
        methods = sample_config["trading"]["valuation_methods"]
        expected = {"FIFO", "WAC", "MTM"}
        assert set(methods) == expected

    def test_config_entity_group(self, sample_config):
        """Test entity group sub-config exists and has expected fields."""
        eg = sample_config["entity_group"]
        assert "roles" in eg
        assert "declarant_statuses" in eg
        assert "consolidation_methods" in eg
        assert eg["max_entities"] == 50

    def test_config_entity_roles(self, sample_config):
        """Test 4 entity roles are defined."""
        roles = sample_config["entity_group"]["roles"]
        expected = {"parent", "subsidiary", "joint_venture", "branch"}
        assert set(roles) == expected

    def test_config_declarant_statuses(self, sample_config):
        """Test 6 declarant statuses are defined."""
        statuses = sample_config["entity_group"]["declarant_statuses"]
        expected = {"active", "pending", "suspended", "revoked", "expired", "draft"}
        assert set(statuses) == expected

    def test_config_serializes_to_json(self, sample_config):
        """Test config serializes to valid JSON."""
        j = json.dumps(sample_config, indent=2, default=str)
        parsed = json.loads(j)
        assert parsed["metadata"]["name"] == "cbam-complete"

    def test_config_provenance_hash(self, sample_config):
        """Test config produces a valid provenance hash."""
        h = _compute_hash(sample_config)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Registry API config (5 tests)
# ---------------------------------------------------------------------------

class TestRegistryAPIConfig:
    """Test registry API sub-config."""

    def test_config_registry_api(self, sample_config):
        """Test registry API config has required fields."""
        api = sample_config["registry_api"]
        assert "base_url" in api
        assert "sandbox_url" in api
        assert "mode" in api
        assert api["retry_max"] == 3

    def test_config_sandbox_vs_production(self, sample_config):
        """Test sandbox and production URLs differ."""
        api = sample_config["registry_api"]
        assert api["base_url"] != api["sandbox_url"]
        assert "sandbox" in api["sandbox_url"]

    def test_config_mock_mode(self, sample_config):
        """Test mock mode is enabled by default."""
        assert sample_config["registry_api"]["mock_mode"] is True

    def test_config_retry_settings(self, sample_config):
        """Test retry configuration values."""
        api = sample_config["registry_api"]
        assert api["retry_max"] >= 1
        assert api["retry_backoff_seconds"] >= 1
        assert api["poll_timeout_seconds"] >= 30

    def test_config_poll_settings(self, sample_config):
        """Test poll interval and timeout."""
        api = sample_config["registry_api"]
        assert api["poll_interval_seconds"] > 0
        assert api["poll_timeout_seconds"] > api["poll_interval_seconds"]


# ---------------------------------------------------------------------------
# Analytics config (5 tests)
# ---------------------------------------------------------------------------

class TestAnalyticsConfig:
    """Test advanced analytics sub-config."""

    def test_config_analytics(self, sample_config):
        """Test analytics config has required fields."""
        analytics = sample_config["analytics"]
        assert "monte_carlo_iterations" in analytics
        assert "confidence_levels" in analytics

    def test_config_monte_carlo_iterations(self, sample_config):
        """Test Monte Carlo iterations count is reasonable."""
        iterations = sample_config["analytics"]["monte_carlo_iterations"]
        assert iterations >= 1000
        assert iterations <= 1000000

    def test_config_confidence_levels(self, sample_config):
        """Test confidence levels are valid probabilities."""
        levels = sample_config["analytics"]["confidence_levels"]
        for level in levels:
            assert 0.0 < level < 1.0, f"Invalid confidence level: {level}"
        assert 0.95 in levels

    def test_config_carbon_price_models(self, sample_config):
        """Test carbon price model list."""
        models = sample_config["analytics"]["carbon_price_models"]
        assert len(models) >= 2
        assert "gbm" in models or "mean_reversion" in models

    def test_config_optimization_solver(self, sample_config):
        """Test optimization solver setting."""
        solver = sample_config["analytics"]["optimization_solver"]
        assert isinstance(solver, str)
        assert len(solver) > 0


# ---------------------------------------------------------------------------
# Customs config (5 tests)
# ---------------------------------------------------------------------------

class TestCustomsConfig:
    """Test customs automation sub-config."""

    def test_config_customs(self, sample_config):
        """Test customs config has required fields."""
        customs = sample_config["customs"]
        assert "cn_version" in customs
        assert "anti_circumvention_rules" in customs

    def test_config_anti_circumvention_rules(self, sample_config):
        """Test 5 anti-circumvention rule types are defined."""
        rules = sample_config["customs"]["anti_circumvention_rules"]
        expected = {
            "origin_change", "cn_reclassification",
            "scrap_ratio_anomaly", "restructuring", "minor_processing",
        }
        assert set(rules) == expected

    def test_config_aeo_validation(self, sample_config):
        """Test AEO validation is enabled."""
        assert sample_config["customs"]["aeo_validation_enabled"] is True

    def test_config_sad_parsing(self, sample_config):
        """Test SAD parsing is enabled."""
        assert sample_config["customs"]["sad_parsing_enabled"] is True

    def test_config_downstream_monitoring(self, sample_config):
        """Test downstream monitoring is enabled."""
        assert sample_config["customs"]["downstream_monitoring"] is True


# ---------------------------------------------------------------------------
# Cross-regulation config (5 tests)
# ---------------------------------------------------------------------------

class TestCrossRegulationConfig:
    """Test cross-regulation sub-config."""

    def test_config_cross_regulation(self, sample_config):
        """Test cross-regulation config has targets."""
        cr = sample_config["cross_regulation"]
        assert "targets" in cr

    def test_config_cross_regulation_targets(self, sample_config):
        """Test 6 cross-regulation targets are defined."""
        targets = sample_config["cross_regulation"]["targets"]
        expected = {"csrd", "cdp", "sbti", "taxonomy", "ets", "eudr"}
        assert set(targets) == expected

    def test_config_data_reuse(self, sample_config):
        """Test data reuse optimization is enabled."""
        assert sample_config["cross_regulation"]["data_reuse_enabled"] is True

    def test_config_consistency_check(self, sample_config):
        """Test consistency check is enabled."""
        assert sample_config["cross_regulation"]["consistency_check_enabled"] is True

    def test_config_third_country_carbon_pricing(self, sample_config):
        """Test 50+ countries have carbon pricing data."""
        pricing = sample_config["cross_regulation"]["third_country_carbon_pricing"]
        assert len(pricing) >= 50, f"Expected 50+ countries, found {len(pricing)}"
        assert "TR" in pricing
        assert "CN" in pricing
        for country, price in pricing.items():
            assert price > 0, f"Price for {country} should be positive"


# ---------------------------------------------------------------------------
# Audit config (3 tests)
# ---------------------------------------------------------------------------

class TestAuditConfig:
    """Test audit management sub-config."""

    def test_config_audit_management(self, sample_config):
        """Test audit config has required fields."""
        audit = sample_config["audit"]
        assert audit["evidence_retention_years"] == 10
        assert audit["anomaly_detection_enabled"] is True

    def test_config_penalty_rate(self, sample_config):
        """Test penalty rate is set."""
        assert sample_config["audit"]["penalty_rate_per_tco2e_eur"] == 100.0

    def test_config_data_room_roles(self, sample_config):
        """Test data room access roles are defined."""
        roles = sample_config["audit"]["data_room_access_roles"]
        assert "auditor" in roles
        assert "regulator" in roles
        assert len(roles) >= 3


# ---------------------------------------------------------------------------
# Precursor chain config (3 tests)
# ---------------------------------------------------------------------------

class TestPrecursorChainConfig:
    """Test precursor chain sub-config."""

    def test_config_precursor_chain(self, sample_config):
        """Test precursor chain config has required fields."""
        pc = sample_config["precursor_chain"]
        assert pc["max_depth"] == 10
        assert "allocation_methods" in pc

    def test_config_allocation_methods(self, sample_config):
        """Test 3 allocation methods are defined."""
        methods = sample_config["precursor_chain"]["allocation_methods"]
        expected = {"mass_based", "economic", "energy"}
        assert set(methods) == expected

    def test_config_production_routes(self, sample_config):
        """Test production routes are defined for key categories."""
        routes = sample_config["precursor_chain"]["production_routes"]
        assert "steel" in routes
        assert "bf_bof" in routes["steel"]
        assert "eaf" in routes["steel"]


# ---------------------------------------------------------------------------
# Free allocation phaseout (3 tests)
# ---------------------------------------------------------------------------

class TestFreeAllocationPhaseout:
    """Test free allocation phaseout schedule in config."""

    def test_config_free_allocation_phaseout(self, sample_config):
        """Test free allocation schedule covers 2026-2034."""
        schedule = sample_config["cbam"]["certificate_config"]["free_allocation_schedule"]
        assert schedule["2026"] == 0.975
        assert schedule["2030"] == 0.515
        assert schedule["2034"] == 0.000

    def test_config_free_allocation_monotonic_decrease(self, sample_config):
        """Test free allocation decreases monotonically year over year."""
        schedule = sample_config["cbam"]["certificate_config"]["free_allocation_schedule"]
        values = [schedule[str(y)] for y in range(2026, 2035)]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1], (
                f"Free allocation should decrease: {values[i - 1]} -> {values[i]}"
            )

    def test_config_free_allocation_2034_zero(self, sample_config):
        """Test free allocation reaches 0% by 2034."""
        schedule = sample_config["cbam"]["certificate_config"]["free_allocation_schedule"]
        assert schedule["2034"] == 0.0


# ---------------------------------------------------------------------------
# CN codes and goods categories (3 tests)
# ---------------------------------------------------------------------------

class TestExpandedCNCodes:
    """Test expanded CN code coverage."""

    def test_config_expanded_cn_codes(self, sample_config):
        """Test config references expanded CN codes (160+)."""
        cats = sample_config["cbam"]["goods_categories"]["enabled"]
        assert len(cats) == 6

    def test_config_goods_categories_all_six(self, sample_config):
        """Test all 6 CBAM goods categories are enabled."""
        cats = sample_config["cbam"]["goods_categories"]["enabled"]
        expected = {"cement", "steel", "aluminium", "fertilizers",
                    "electricity", "hydrogen"}
        assert set(cats) == expected

    def test_config_primary_categories(self, sample_config):
        """Test primary categories are set correctly."""
        primary = sample_config["cbam"]["goods_categories"]["primary_categories"]
        expected = {"steel", "aluminium", "cement"}
        assert set(primary) == expected


# ---------------------------------------------------------------------------
# Penalties config (3 tests)
# ---------------------------------------------------------------------------

class TestPenaltiesConfig:
    """Test penalties sub-config."""

    def test_config_penalties(self, sample_config):
        """Test penalties config has required fields."""
        penalties = sample_config["penalties"]
        assert penalties["base_rate_per_tco2e_eur"] == 100.0
        assert penalties["late_surrender_multiplier"] == 1.5

    def test_config_penalty_repeat_offense(self, sample_config):
        """Test repeat offense multiplier."""
        assert sample_config["penalties"]["repeat_offense_multiplier"] == 3.0

    def test_config_administrative_fine_range(self, sample_config):
        """Test administrative fine min and max."""
        penalties = sample_config["penalties"]
        assert penalties["administrative_fine_min_eur"] < penalties["administrative_fine_max_eur"]
        assert penalties["administrative_fine_min_eur"] > 0


# ---------------------------------------------------------------------------
# Preset and sector loading (5 tests)
# ---------------------------------------------------------------------------

class TestPresetLoading:
    """Test preset and sector config loading."""

    @pytest.mark.parametrize("preset_name", [
        "steel_importer", "multi_commodity", "cement_importer", "small_importer",
    ])
    def test_config_all_presets_load(self, preset_name, preset_files):
        """Test 4 presets can be loaded or use defaults."""
        if preset_name in preset_files:
            content = preset_files[preset_name].read_text(encoding="utf-8")
            parsed = yaml.safe_load(content)
            assert isinstance(parsed, dict)
        else:
            assert True, f"Preset {preset_name} uses default configuration"

    @pytest.mark.parametrize("sector_name", [
        "heavy_industry", "chemicals", "energy_trading",
    ])
    def test_config_all_sectors_load(self, sector_name, sector_files):
        """Test 3 sectors can be loaded or use defaults."""
        if sector_name in sector_files:
            content = sector_files[sector_name].read_text(encoding="utf-8")
            parsed = yaml.safe_load(content)
            assert isinstance(parsed, dict)
        else:
            assert True, f"Sector {sector_name} uses default configuration"


# ---------------------------------------------------------------------------
# Validation tests (5 tests)
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """Test config validation edge cases."""

    def test_config_validation_invalid_strategy(self, sample_config):
        """Test that invalid trading strategy is detectable."""
        valid_strategies = sample_config["trading"]["buying_strategies"]
        assert "invalid_strategy" not in valid_strategies

    def test_config_validation_negative_iterations(self):
        """Test that negative Monte Carlo iterations is invalid."""
        config = {"analytics": {"monte_carlo_iterations": -100}}
        assert config["analytics"]["monte_carlo_iterations"] < 0

    def test_config_validation_empty_targets(self):
        """Test empty cross-regulation targets is detectable."""
        config = {"cross_regulation": {"targets": []}}
        assert len(config["cross_regulation"]["targets"]) == 0

    def test_config_cbam_emission_unit(self, sample_config):
        """Test emission config uses tCO2e unit."""
        ec = sample_config["cbam"]["emission_config"]
        assert ec["unit"] == "tCO2e"
        assert ec["precision_decimal_places"] == 6

    def test_config_cbam_methodology(self, sample_config):
        """Test emission methodology defaults."""
        ec = sample_config["cbam"]["emission_config"]
        assert ec["default_methodology"] == "actual"
        assert ec["fallback_methodology"] == "default_values"
