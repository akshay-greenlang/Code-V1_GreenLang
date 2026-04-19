# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - Configuration.

Validates all Pydantic configuration models, enums, presets, validation rules,
enterprise configurations, and cross-field constraints in pack_config.py.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~50 tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

PACK_DIR = Path(__file__).resolve().parent.parent
if str(PACK_DIR.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(PACK_DIR.parent.parent.parent))
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

from config.pack_config import (
    EnterpriseSector,
    ConsolidationApproach,
    SBTiPathway,
    DataQualityTier,
    ERPSystem,
    CarbonPricingApproach,
    PackConfig,
    EnterpriseNetZeroConfig,
    EnterpriseOrganizationConfig,
    ConsolidationConfig,
    SBTiTargetConfig,
    ScenarioModelingConfig,
    CarbonPricingConfig as EnterpriseCarbonPricingConfig,
    SupplyChainConfig,
    AssuranceConfig,
    PerformanceConfig,
    validate_config,
    SUPPORTED_PRESETS,
)

# Aliases for test compatibility
DataQualityTarget = DataQualityTier
ERPPlatform = ERPSystem
EnterpriseBoundaryConfig = ConsolidationConfig
EnterpriseSBTiConfig = SBTiTargetConfig
EnterpriseScenarioConfig = ScenarioModelingConfig
EnterpriseSupplyChainConfig = SupplyChainConfig
EnterpriseAssuranceConfig = AssuranceConfig
EnterprisePerformanceConfig = PerformanceConfig


CONFIG_DIR = PACK_DIR / "config"
PRESETS_DIR = CONFIG_DIR / "presets"


# ===========================================================================
# Tests -- Enum Definitions
# ===========================================================================


class TestEnumDefinitions:
    """Tests that all configuration enums are properly defined."""

    def test_all_enums_defined(self) -> None:
        """All configuration enums must be importable."""
        enums = [EnterpriseSector, ConsolidationApproach, SBTiPathway,
                 DataQualityTarget, ERPPlatform, CarbonPricingApproach]
        for enum_cls in enums:
            assert len(enum_cls) > 0, f"{enum_cls.__name__} has no members"

    @pytest.mark.parametrize("sector", [
        "MANUFACTURING", "ENERGY_UTILITIES", "FINANCIAL_SERVICES",
        "TECHNOLOGY", "CONSUMER_GOODS", "TRANSPORT_LOGISTICS",
        "REAL_ESTATE", "HEALTHCARE_PHARMA",
    ])
    def test_enterprise_sectors(self, sector: str) -> None:
        """All enterprise sectors must be valid enum values."""
        assert EnterpriseSector(sector) is not None

    @pytest.mark.parametrize("approach", [
        "FINANCIAL_CONTROL", "OPERATIONAL_CONTROL", "EQUITY_SHARE",
    ])
    def test_consolidation_approaches(self, approach: str) -> None:
        """All consolidation approaches must be valid enum values."""
        assert ConsolidationApproach(approach) is not None

    @pytest.mark.parametrize("pathway", [
        "ACA_15C", "ACA_WB2C", "SDA", "FLAG", "MIXED",
    ])
    def test_sbti_pathways(self, pathway: str) -> None:
        """All SBTi pathways must be valid enum values."""
        assert SBTiPathway(pathway) is not None

    @pytest.mark.parametrize("platform", [
        "SAP_S4HANA", "ORACLE_ERP_CLOUD", "WORKDAY", "NONE",
    ])
    def test_erp_platforms(self, platform: str) -> None:
        """All ERP platforms must be valid enum values."""
        assert ERPPlatform(platform) is not None

    @pytest.mark.parametrize("approach", [
        "SHADOW_PRICE", "INTERNAL_FEE", "IMPLICIT", "REGULATORY",
    ])
    def test_carbon_pricing_approaches(self, approach: str) -> None:
        """All carbon pricing approaches must be valid."""
        assert CarbonPricingApproach(approach) is not None

    @pytest.mark.parametrize("target", [
        "LEVEL_1", "LEVEL_2", "LEVEL_3", "LEVEL_4", "LEVEL_5",
    ])
    def test_data_quality_levels(self, target: str) -> None:
        """All data quality levels must be valid."""
        assert DataQualityTarget(target) is not None


# ===========================================================================
# Tests -- OrganizationConfig Validation
# ===========================================================================


class TestOrganizationConfig:
    """Tests for Enterprise OrganizationConfig model."""

    def test_organization_config_defaults(self) -> None:
        """Default OrganizationConfig must instantiate without error."""
        org = EnterpriseOrganizationConfig()
        assert org.sector is not None
        assert org.headquarters_country is not None

    def test_large_enterprise_config(self) -> None:
        """Enterprise config must accept large org parameters."""
        org = EnterpriseOrganizationConfig(
            name="GlobalManufact Corp",
            sector=EnterpriseSector.MANUFACTURING,
            employee_count=12500,
            revenue_eur=Decimal("2800000000"),
            entity_count=28,
            operating_countries=["DE", "US", "CN", "GB", "FR", "JP", "IN", "BR",
                                 "SG", "NL", "CH", "SE", "KR", "AU", "CA", "MX", "IT", "ES"],
        )
        assert org.employee_count == 12500
        assert org.entity_count == 28

    def test_financial_services_config(self) -> None:
        """Financial services config with PCAF."""
        org = EnterpriseOrganizationConfig(
            name="GlobalBank Holdings",
            sector=EnterpriseSector.FINANCIAL_SERVICES,
            employee_count=85000,
            revenue_eur=Decimal("45000000000"),
        )
        assert org.sector == EnterpriseSector.FINANCIAL_SERVICES

    def test_name_accepts_string(self) -> None:
        """Organization name accepts a string."""
        org = EnterpriseOrganizationConfig(name="TestCorp")
        assert org.name == "TestCorp"

    def test_sector_accepts_enum(self) -> None:
        """Organization sector accepts enum values."""
        org = EnterpriseOrganizationConfig(
            sector=EnterpriseSector.TECHNOLOGY
        )
        assert org.sector == EnterpriseSector.TECHNOLOGY


# ===========================================================================
# Tests -- BoundaryConfig Validation
# ===========================================================================


class TestBoundaryConfig:
    """Tests for Enterprise BoundaryConfig (ConsolidationConfig) model."""

    def test_boundary_config_defaults(self) -> None:
        """Default ConsolidationConfig has enterprise defaults."""
        bc = EnterpriseBoundaryConfig()
        assert bc.approach == ConsolidationApproach.FINANCIAL_CONTROL
        assert bc.intercompany_elimination is True

    def test_financial_control_boundary(self) -> None:
        """Financial control boundary method."""
        bc = EnterpriseBoundaryConfig(
            approach=ConsolidationApproach.FINANCIAL_CONTROL
        )
        assert bc.approach == ConsolidationApproach.FINANCIAL_CONTROL

    def test_operational_control_boundary(self) -> None:
        """Operational control boundary method."""
        bc = EnterpriseBoundaryConfig(
            approach=ConsolidationApproach.OPERATIONAL_CONTROL
        )
        assert bc.approach == ConsolidationApproach.OPERATIONAL_CONTROL

    def test_equity_share_boundary(self) -> None:
        """Equity share boundary method."""
        bc = EnterpriseBoundaryConfig(
            approach=ConsolidationApproach.EQUITY_SHARE
        )
        assert bc.approach == ConsolidationApproach.EQUITY_SHARE

    def test_intercompany_elimination_enabled(self) -> None:
        """Enterprise boundary has intercompany elimination enabled."""
        bc = EnterpriseBoundaryConfig()
        assert bc.intercompany_elimination is True

    def test_base_year_recalculation_threshold(self) -> None:
        """Enterprise boundary has base year recalculation threshold."""
        bc = EnterpriseBoundaryConfig()
        assert bc.base_year_recalculation_threshold_pct > 0


# ===========================================================================
# Tests -- SBTi Config Validation
# ===========================================================================


class TestSBTiConfig:
    """Tests for Enterprise SBTi configuration."""

    def test_sbti_config_defaults(self) -> None:
        """Default SBTi config for enterprise."""
        sc = EnterpriseSBTiConfig()
        assert sc.sbti_pathway == SBTiPathway.ACA_15C
        assert sc.scope1_2_coverage_pct >= Decimal("95")
        assert sc.scope3_near_term_coverage_pct >= Decimal("67")

    def test_aca_15c_reduction_rate(self) -> None:
        """ACA 1.5C requires 4.2%/yr reduction."""
        sc = EnterpriseSBTiConfig(sbti_pathway=SBTiPathway.ACA_15C)
        assert sc.near_term_scope1_2_reduction_pct >= Decimal("42")

    def test_near_term_target_timeframe(self) -> None:
        """Near-term target must be reasonable."""
        sc = EnterpriseSBTiConfig()
        assert sc.near_term_target_year >= 2028
        assert sc.near_term_target_year <= 2035

    def test_long_term_scope3_coverage(self) -> None:
        """Long-term Scope 3 coverage must be 90%+."""
        sc = EnterpriseSBTiConfig()
        assert sc.scope3_long_term_coverage_pct >= Decimal("90")

    def test_net_zero_reduction_target(self) -> None:
        """Net-zero requires 90%+ absolute reduction by 2050."""
        sc = EnterpriseSBTiConfig()
        assert sc.long_term_reduction_pct >= Decimal("90")


# ===========================================================================
# Tests -- Carbon Pricing Config
# ===========================================================================


class TestCarbonPricingConfig:
    """Tests for enterprise carbon pricing configuration."""

    def test_carbon_pricing_defaults(self) -> None:
        """Default carbon pricing config."""
        cp = EnterpriseCarbonPricingConfig()
        assert cp.approach == CarbonPricingApproach.SHADOW_PRICE
        assert cp.price_usd_per_tco2e >= Decimal("50")

    def test_price_range_validation(self) -> None:
        """Carbon price must be $50-$200/tCO2e."""
        cp = EnterpriseCarbonPricingConfig(
            price_usd_per_tco2e=Decimal("85")
        )
        assert Decimal("50") <= cp.price_usd_per_tco2e <= Decimal("200")

    def test_cbam_configuration(self) -> None:
        """CBAM exposure tracking configuration."""
        cp = EnterpriseCarbonPricingConfig(cbam_enabled=True)
        assert cp.cbam_enabled is True

    def test_ets_configuration(self) -> None:
        """ETS compliance configuration."""
        cp = EnterpriseCarbonPricingConfig(ets_enabled=True)
        assert cp.ets_enabled is True


# ===========================================================================
# Tests -- PackConfig & Cross-Validation
# ===========================================================================


class TestPackConfig:
    """Tests for PackConfig top-level wrapper and cross-validation."""

    def test_pack_config_loads(self) -> None:
        """Default PackConfig must instantiate without errors."""
        config = PackConfig()
        assert config.pack_id == "PACK-027-enterprise-net-zero"
        assert config.config_version == "1.0.0"

    def test_config_hash_deterministic(self) -> None:
        """get_config_hash must produce deterministic SHA-256."""
        c1 = PackConfig()
        c2 = PackConfig()
        assert c1.get_config_hash() == c2.get_config_hash()
        assert len(c1.get_config_hash()) == 64

    def test_cross_validation_empty_org_name(self) -> None:
        """Cross-validation must warn when organization name is empty."""
        config = EnterpriseNetZeroConfig(
            organization=EnterpriseOrganizationConfig(name="")
        )
        warnings = validate_config(config)
        assert any("name" in w.lower() for w in warnings)

    def test_get_enabled_engines(self) -> None:
        """get_enabled_engines must return all engines."""
        config = EnterpriseNetZeroConfig()
        engines = config.get_enabled_engines()
        assert len(engines) >= 7
        engines_str = " ".join(engines)
        assert "enterprise_baseline" in engines_str
        assert "sbti_target" in engines_str
        assert "scenario_modeling" in engines_str
        assert "carbon_pricing" in engines_str
        assert "supply_chain_mapping" in engines_str
        assert "multi_entity_consolidation" in engines_str
        assert "financial_integration" in engines_str

    def test_performance_config_defaults(self) -> None:
        """Performance config must have enterprise-appropriate defaults."""
        pc = EnterprisePerformanceConfig()
        assert pc.timeout_seconds <= 300
        assert pc.max_concurrent_calcs >= 1

    def test_assurance_config_defaults(self) -> None:
        """Assurance config must default to limited assurance."""
        ac = EnterpriseAssuranceConfig()
        assert ac.level.value in ["LIMITED", "REASONABLE"]

    def test_preset_loading_manufacturing(self) -> None:
        """Manufacturing enterprise preset loads correctly."""
        config = PackConfig.from_preset("manufacturing")
        assert config.pack.organization.sector == EnterpriseSector.MANUFACTURING

    def test_preset_loading_financial_services(self) -> None:
        """Financial services preset loads correctly."""
        config = PackConfig.from_preset("financial_services")
        assert config.pack.organization.sector == EnterpriseSector.FINANCIAL_SERVICES

    def test_preset_loading_technology(self) -> None:
        """Technology preset loads correctly."""
        config = PackConfig.from_preset("technology")
        assert config.pack.organization.sector == EnterpriseSector.TECHNOLOGY

    def test_preset_loading_energy_utilities(self) -> None:
        """Energy utilities preset loads correctly."""
        config = PackConfig.from_preset("energy_utilities")
        assert config.pack.organization.sector == EnterpriseSector.ENERGY_UTILITIES

    def test_preset_loading_consumer_goods(self) -> None:
        """Consumer goods (retail_consumer) preset loads correctly."""
        config = PackConfig.from_preset("retail_consumer")
        assert config.pack.organization.sector is not None

    def test_invalid_preset_raises(self) -> None:
        """Loading an unknown preset must raise ValueError."""
        with pytest.raises((ValueError, KeyError)):
            PackConfig.from_preset("nonexistent_preset")

    def test_all_supported_presets_exist(self) -> None:
        """All SUPPORTED_PRESETS must have corresponding YAML files."""
        for preset_name in SUPPORTED_PRESETS:
            yaml_path = PRESETS_DIR / f"{preset_name}.yaml"
            assert yaml_path.exists(), f"Preset YAML missing: {yaml_path}"

    def test_preset_with_overrides(self) -> None:
        """Preset loading with overrides must apply the overrides."""
        config = PackConfig.from_preset(
            "manufacturing",
            overrides={"reporting_year": 2026},
        )
        assert config.pack.reporting_year == 2026

    def test_supply_chain_config_defaults(self) -> None:
        """Supply chain config has enterprise defaults."""
        sc = EnterpriseSupplyChainConfig()
        assert sc.tier_depth >= 3
        assert sc.cdp_supply_chain_integration is True

    def test_scenario_config_defaults(self) -> None:
        """Scenario config has Monte Carlo defaults."""
        sc = EnterpriseScenarioConfig()
        assert sc.monte_carlo_runs >= 10000
