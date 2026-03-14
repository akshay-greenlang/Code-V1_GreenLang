# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise Pack - Config Presets Tests (50 tests)

Tests EnterprisePackConfig creation, all sub-config models,
size presets, sector overrides, demo config, and validation.

Author: GreenLang QA Team
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


# ---------------------------------------------------------------------------
# EnterprisePackConfig creation and validation (10 tests)
# ---------------------------------------------------------------------------

class TestEnterprisePackConfigCreation:
    """Test EnterprisePackConfig instantiation and fields."""

    def test_config_creates_with_defaults(self, pack_config_module):
        """Test EnterprisePackConfig creates with all default values."""
        cfg = pack_config_module.EnterprisePackConfig()
        assert cfg.tenant.enabled is False
        assert cfg.sso.saml_enabled is False
        assert cfg.white_label.enabled is False
        assert cfg.iot.enabled is False
        assert cfg.carbon_credit.enabled is False

    def test_config_creates_with_all_fields(self, pack_config_module, sample_enterprise_config):
        """Test EnterprisePackConfig creates with all fields populated."""
        ent = sample_enterprise_config["enterprise"]
        cfg = pack_config_module.EnterprisePackConfig(**ent)
        assert cfg.tenant.enabled is True
        assert cfg.sso.saml_enabled is True
        assert cfg.white_label.enabled is True
        assert cfg.iot.enabled is True
        assert cfg.carbon_credit.enabled is True

    def test_config_serializes_to_dict(self, pack_config_module):
        """Test EnterprisePackConfig serializes to dict."""
        cfg = pack_config_module.EnterprisePackConfig()
        d = cfg.model_dump()
        assert isinstance(d, dict)
        assert "tenant" in d
        assert "sso" in d
        assert "white_label" in d

    def test_config_serializes_to_json(self, pack_config_module):
        """Test EnterprisePackConfig serializes to JSON."""
        cfg = pack_config_module.EnterprisePackConfig()
        j = cfg.model_dump_json()
        parsed = json.loads(j)
        assert "tenant" in parsed
        assert "predictive" in parsed

    def test_config_inherits_professional_fields(self, pack_config_module):
        """Test EnterprisePackConfig includes inherited professional configs."""
        cfg = pack_config_module.EnterprisePackConfig()
        assert hasattr(cfg, "consolidation")
        assert hasattr(cfg, "approval")
        assert hasattr(cfg, "quality_gates")
        assert hasattr(cfg, "cross_framework")
        assert hasattr(cfg, "scenarios")

    def test_config_metadata_creates(self, pack_config_module):
        """Test PackMetadata model creates correctly."""
        meta = pack_config_module.PackMetadata(
            name="csrd-enterprise",
            version="1.0.0",
            display_name="CSRD Enterprise Pack",
            category="eu-compliance",
        )
        assert meta.name == "csrd-enterprise"
        assert meta.tier == "enterprise"

    def test_config_components_creates(self, pack_config_module):
        """Test ComponentsConfig model creates with empty groups."""
        comp = pack_config_module.ComponentsConfig()
        assert len(comp.get_all_agent_ids()) == 0

    def test_config_performance_targets(self, pack_config_module):
        """Test PerformanceTargets with default enterprise values."""
        perf = pack_config_module.PerformanceTargets()
        assert perf.data_ingestion_rps == 50000
        assert perf.iot_events_per_second == 10000
        assert perf.forecast_max_seconds == 60
        assert perf.availability_percent == 99.99

    def test_config_requirements_defaults(self, pack_config_module):
        """Test RequirementsConfig default infrastructure values."""
        reqs = pack_config_module.RequirementsConfig()
        assert reqs.min_cpu_cores == 16
        assert reqs.min_memory_gb == 64
        assert "pgvector" in reqs.database_extensions

    def test_config_csrd_enterprise_pack_config(self, pack_config_module):
        """Test CSRDEnterprisePackConfig top-level model."""
        meta = pack_config_module.PackMetadata(
            name="csrd-enterprise",
            version="1.0.0",
            display_name="CSRD Enterprise Pack",
            category="eu-compliance",
        )
        cfg = pack_config_module.CSRDEnterprisePackConfig(metadata=meta)
        assert cfg.metadata.name == "csrd-enterprise"
        assert cfg.is_multi_tenant() is False
        assert cfg.is_iot_enabled() is False


# ---------------------------------------------------------------------------
# Sub-config model tests (24 tests - 2 per sub-config)
# ---------------------------------------------------------------------------

class TestMultiTenantConfig:
    def test_default_values(self, pack_config_module):
        cfg = pack_config_module.MultiTenantConfig()
        assert cfg.enabled is False
        assert cfg.max_tenants == 100
        assert cfg.data_residency_enforcement is True

    def test_custom_values(self, pack_config_module):
        cfg = pack_config_module.MultiTenantConfig(
            enabled=True, isolation_level="PHYSICAL", max_tenants=500,
        )
        assert cfg.enabled is True
        assert cfg.max_tenants == 500


class TestSSOConfig:
    def test_default_values(self, pack_config_module):
        cfg = pack_config_module.SSOConfig()
        assert cfg.saml_enabled is False
        assert cfg.session_timeout_minutes == 480

    def test_custom_values(self, pack_config_module):
        cfg = pack_config_module.SSOConfig(
            saml_enabled=True, oauth_enabled=True, mfa_required=True,
            allowed_domains=["example.com"],
        )
        assert cfg.saml_enabled is True
        assert cfg.mfa_required is True
        assert "example.com" in cfg.allowed_domains


class TestWhiteLabelConfig:
    def test_default_values(self, pack_config_module):
        cfg = pack_config_module.WhiteLabelConfig()
        assert cfg.enabled is False
        assert cfg.primary_color == "#1B5E20"

    def test_custom_values(self, pack_config_module):
        cfg = pack_config_module.WhiteLabelConfig(
            enabled=True, primary_color="#003366",
            custom_domain="sustainability.acme.com",
            powered_by_visible=False,
        )
        assert cfg.primary_color == "#003366"
        assert cfg.powered_by_visible is False


class TestPredictiveConfig:
    def test_default_values(self, pack_config_module):
        cfg = pack_config_module.PredictiveConfig()
        assert cfg.forecast_horizon_months == 12
        assert cfg.confidence_level == 0.95

    def test_custom_values(self, pack_config_module):
        cfg = pack_config_module.PredictiveConfig(
            forecast_horizon_months=24, anomaly_sensitivity=0.90,
            drift_psi_threshold=0.15,
        )
        assert cfg.forecast_horizon_months == 24
        assert cfg.anomaly_sensitivity == 0.90


class TestNarrativeConfig:
    def test_default_values(self, pack_config_module):
        cfg = pack_config_module.NarrativeConfig()
        assert "en" in cfg.languages
        assert cfg.fact_checking_enabled is True

    def test_custom_values(self, pack_config_module):
        cfg = pack_config_module.NarrativeConfig(
            languages=["en", "de", "fr"], tone="board",
            max_draft_tokens=16000,
        )
        assert len(cfg.languages) == 3
        assert cfg.max_draft_tokens == 16000


class TestWorkflowBuilderConfig:
    def test_default_values(self, pack_config_module):
        cfg = pack_config_module.WorkflowBuilderConfig()
        assert cfg.max_steps == 50
        assert cfg.template_sharing is True

    def test_custom_values(self, pack_config_module):
        cfg = pack_config_module.WorkflowBuilderConfig(
            max_steps=100, max_custom_workflows=200,
        )
        assert cfg.max_steps == 100
        assert cfg.max_custom_workflows == 200


class TestIoTConfig:
    def test_default_values(self, pack_config_module):
        cfg = pack_config_module.IoTConfig()
        assert cfg.enabled is False
        assert cfg.aggregation_window_minutes == 15

    def test_custom_values(self, pack_config_module):
        cfg = pack_config_module.IoTConfig(
            enabled=True, protocols=["MQTT", "OPCUA"],
            max_devices=5000,
        )
        assert cfg.enabled is True
        assert cfg.max_devices == 5000


class TestCarbonCreditConfig:
    def test_default_values(self, pack_config_module):
        cfg = pack_config_module.CarbonCreditConfig()
        assert cfg.enabled is False
        assert cfg.vintage_tracking is True

    def test_custom_values(self, pack_config_module):
        cfg = pack_config_module.CarbonCreditConfig(
            enabled=True, registries_enabled=["VCS", "ACR"],
            buffer_pool_percent=15.0,
        )
        assert "ACR" in cfg.registries_enabled
        assert cfg.buffer_pool_percent == 15.0


class TestSupplyChainConfig:
    def test_default_values(self, pack_config_module):
        cfg = pack_config_module.SupplyChainConfig()
        assert cfg.enabled is True
        assert cfg.max_tiers == 4

    def test_custom_values(self, pack_config_module):
        cfg = pack_config_module.SupplyChainConfig(
            max_tiers=6, risk_threshold=0.5,
            scoring_weights={"environmental": 0.50, "social": 0.30, "governance": 0.20},
        )
        assert cfg.max_tiers == 6
        assert cfg.scoring_weights["environmental"] == 0.50


class TestFilingConfig:
    def test_default_values(self, pack_config_module):
        cfg = pack_config_module.FilingConfig()
        assert cfg.enabled is True
        assert cfg.auto_submit is False

    def test_custom_values(self, pack_config_module):
        cfg = pack_config_module.FilingConfig(
            targets=["ESAP", "national_registries"],
            validation_strictness="relaxed",
            deadline_buffer_days=30,
        )
        assert len(cfg.targets) == 2
        assert cfg.deadline_buffer_days == 30


class TestAPIManagementConfig:
    def test_default_values(self, pack_config_module):
        cfg = pack_config_module.APIManagementConfig()
        assert cfg.rate_limit_per_minute == 600
        assert cfg.graphql_enabled is True

    def test_custom_values(self, pack_config_module):
        cfg = pack_config_module.APIManagementConfig(
            rate_limit_per_minute=1200, burst_limit=200,
            api_key_rotation_days=60,
        )
        assert cfg.rate_limit_per_minute == 1200
        assert cfg.api_key_rotation_days == 60


class TestMarketplaceConfig:
    def test_default_values(self, pack_config_module):
        cfg = pack_config_module.MarketplaceConfig()
        assert cfg.plugins_enabled is False
        assert cfg.sandbox_mode is True

    def test_custom_values(self, pack_config_module):
        cfg = pack_config_module.MarketplaceConfig(
            plugins_enabled=True, max_plugins=100,
            auto_update=True,
        )
        assert cfg.plugins_enabled is True
        assert cfg.max_plugins == 100


# ---------------------------------------------------------------------------
# Size Preset tests (4 tests)
# ---------------------------------------------------------------------------

class TestSizePresets:
    """Test 4 size presets load correctly."""

    @pytest.mark.parametrize("preset_name", [
        "global_enterprise", "saas_platform",
        "financial_enterprise", "consulting_firm",
    ])
    def test_preset_file_exists_and_valid(self, preset_name, preset_files):
        """Test preset YAML file exists and parses."""
        assert preset_name in preset_files, (
            f"Preset file not found for {preset_name}"
        )
        content = preset_files[preset_name].read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict), (
            f"Preset {preset_name} should parse to dict"
        )


# ---------------------------------------------------------------------------
# Sector Preset tests (5 tests)
# ---------------------------------------------------------------------------

class TestSectorPresets:
    """Test 5 sector overrides load correctly."""

    @pytest.mark.parametrize("sector_name", [
        "banking_enterprise", "oil_gas_enterprise",
        "automotive_enterprise", "pharma_enterprise", "conglomerate",
    ])
    def test_sector_file_exists_and_valid(self, sector_name, sector_files):
        """Test sector YAML file exists and parses."""
        assert sector_name in sector_files, (
            f"Sector file not found for {sector_name}"
        )
        content = sector_files[sector_name].read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict), (
            f"Sector {sector_name} should parse to dict"
        )


# ---------------------------------------------------------------------------
# Demo config and tenant profiles tests (3 tests)
# ---------------------------------------------------------------------------

class TestDemoConfig:
    """Test demo configuration and tenant profiles."""

    def test_demo_config_exists_and_valid(self, demo_config):
        """Test demo config YAML loads correctly."""
        assert isinstance(demo_config, dict)
        assert demo_config.get("enabled") is True
        assert demo_config.get("use_sample_data") is True
        assert demo_config.get("skip_external_apis") is True

    def test_demo_tenant_profiles_valid(self, demo_tenant_profiles):
        """Test demo tenant profiles JSON loads correctly."""
        assert isinstance(demo_tenant_profiles, (list, dict))

    def test_demo_iot_stream_exists(self, demo_iot_stream_path):
        """Test demo IoT stream CSV file exists."""
        assert demo_iot_stream_path.exists(), (
            f"Demo IoT stream not found at {demo_iot_stream_path}"
        )


# ---------------------------------------------------------------------------
# Config validation error tests (4 tests)
# ---------------------------------------------------------------------------

class TestConfigValidationErrors:
    """Test config validation rejects invalid values."""

    def test_supply_chain_weights_must_sum_to_one(self, pack_config_module):
        """Test supply chain scoring weights must sum to 1.0."""
        with pytest.raises(Exception):
            pack_config_module.SupplyChainConfig(
                scoring_weights={"environmental": 0.50, "social": 0.50, "governance": 0.50}
            )

    def test_forecast_horizon_too_small(self, pack_config_module):
        """Test forecast horizon below minimum raises error."""
        with pytest.raises(Exception):
            pack_config_module.PredictiveConfig(forecast_horizon_months=1)

    def test_max_tenants_too_large(self, pack_config_module):
        """Test max_tenants above 10000 raises error."""
        with pytest.raises(Exception):
            pack_config_module.MultiTenantConfig(max_tenants=99999)

    def test_invalid_assurance_level(self, pack_config_module):
        """Test invalid assurance level raises error."""
        with pytest.raises(Exception):
            pack_config_module.AssuranceConfig(assurance_level="invalid_level")
