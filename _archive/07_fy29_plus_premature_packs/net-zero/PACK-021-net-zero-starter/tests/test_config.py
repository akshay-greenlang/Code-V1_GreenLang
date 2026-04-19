# -*- coding: utf-8 -*-
"""
Test suite for PACK-021 Net Zero Starter Pack - Configuration.

Validates all Pydantic configuration models, enums, presets, validation rules,
and cross-field constraints in pack_config.py.

Author:  GreenLang Test Engineering
Pack:    PACK-021 Net Zero Starter
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

# Ensure the pack root is importable
PACK_DIR = Path(__file__).resolve().parent.parent
if str(PACK_DIR.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(PACK_DIR.parent.parent.parent))
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

from config.pack_config import (
    AmbitionLevel,
    BoundaryConfig,
    BoundaryMethod,
    DataSourceType,
    ERPType,
    MaturityAssessment,
    NetZeroStarterConfig,
    OffsetConfig,
    OffsetStrategy,
    OrganizationConfig,
    OrganizationSector,
    OrganizationSize,
    PackConfig,
    PathwayType,
    PerformanceConfig,
    ReductionConfig,
    ReportFormat,
    ReportingConfig,
    Scope3Method,
    ScopeConfig,
    ScorecardConfig,
    TargetConfig,
    TargetTimeframe,
    AuditTrailConfig,
    validate_config,
    SUPPORTED_PRESETS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_DIR = PACK_DIR / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
DEMO_DIR = CONFIG_DIR / "demo"


# ===========================================================================
# Tests -- Enum Definitions
# ===========================================================================


class TestEnumDefinitions:
    """Tests that all configuration enums are properly defined."""

    def test_all_enums_defined(self) -> None:
        """All 12 configuration enums must be importable."""
        enums = [
            OrganizationSector, BoundaryMethod, AmbitionLevel, PathwayType,
            TargetTimeframe, DataSourceType, ReportFormat, Scope3Method,
            OffsetStrategy, MaturityAssessment, ERPType, OrganizationSize,
        ]
        for enum_cls in enums:
            assert len(enum_cls) > 0, f"{enum_cls.__name__} has no members"

    @pytest.mark.parametrize("sector", [
        "MANUFACTURING", "SERVICES", "RETAIL", "ENERGY",
        "TECHNOLOGY", "AGRICULTURE", "OTHER",
    ])
    def test_organization_sectors(self, sector: str) -> None:
        """All expected organization sectors must be valid enum values."""
        assert OrganizationSector(sector) is not None

    @pytest.mark.parametrize("method", [
        "OPERATIONAL_CONTROL", "FINANCIAL_CONTROL", "EQUITY_SHARE",
    ])
    def test_boundary_methods(self, method: str) -> None:
        """All GHG Protocol boundary methods must be valid enum values."""
        assert BoundaryMethod(method) is not None

    @pytest.mark.parametrize("level", [
        "CELSIUS_1_5", "WELL_BELOW_2", "CELSIUS_2",
    ])
    def test_ambition_levels(self, level: str) -> None:
        """All SBTi ambition levels must be valid enum values."""
        assert AmbitionLevel(level) is not None

    @pytest.mark.parametrize("pathway", ["ACA", "SDA", "FLAG"])
    def test_pathway_types(self, pathway: str) -> None:
        """All SBTi pathway types must be valid enum values."""
        assert PathwayType(pathway) is not None

    @pytest.mark.parametrize("fmt", ["MARKDOWN", "HTML", "JSON", "PDF"])
    def test_report_formats(self, fmt: str) -> None:
        """All report format types must be valid enum values."""
        assert ReportFormat(fmt) is not None


# ===========================================================================
# Tests -- OrganizationConfig Validation
# ===========================================================================


class TestOrganizationConfig:
    """Tests for OrganizationConfig Pydantic model."""

    def test_organization_config_defaults(self) -> None:
        """Default OrganizationConfig must instantiate without error."""
        org = OrganizationConfig()
        assert org.sector == OrganizationSector.MANUFACTURING
        assert org.size == OrganizationSize.LARGE
        assert org.region == "EU"
        assert org.country == "DE"
        assert org.fiscal_year_end == "12-31"

    def test_organization_config_custom(self) -> None:
        """OrganizationConfig must accept custom values."""
        org = OrganizationConfig(
            name="TestCo",
            sector=OrganizationSector.TECHNOLOGY,
            size=OrganizationSize.SMALL,
            region="US",
            country="US",
            revenue_eur=100_000_000.0,
            employee_count=500,
        )
        assert org.name == "TestCo"
        assert org.sector == OrganizationSector.TECHNOLOGY
        assert org.employee_count == 500

    def test_fiscal_year_end_valid(self) -> None:
        """Valid fiscal year end dates must be accepted."""
        org = OrganizationConfig(fiscal_year_end="06-30")
        assert org.fiscal_year_end == "06-30"

    def test_fiscal_year_end_invalid_format(self) -> None:
        """Invalid fiscal year end format must raise ValueError."""
        with pytest.raises(Exception):
            OrganizationConfig(fiscal_year_end="December 31")

    def test_fiscal_year_end_invalid_month(self) -> None:
        """Month > 12 in fiscal_year_end must raise ValueError."""
        with pytest.raises(Exception):
            OrganizationConfig(fiscal_year_end="13-01")


# ===========================================================================
# Tests -- BoundaryConfig Validation
# ===========================================================================


class TestBoundaryConfig:
    """Tests for BoundaryConfig Pydantic model."""

    def test_boundary_config_defaults(self) -> None:
        """Default BoundaryConfig must instantiate correctly."""
        bc = BoundaryConfig()
        assert bc.method == BoundaryMethod.OPERATIONAL_CONTROL
        assert bc.reporting_currency == "EUR"
        assert bc.include_joint_ventures is False
        assert bc.base_year_recalculation_threshold_pct == 5.0

    def test_boundary_config_custom_method(self) -> None:
        """BoundaryConfig must accept EQUITY_SHARE method."""
        bc = BoundaryConfig(method=BoundaryMethod.EQUITY_SHARE)
        assert bc.method == BoundaryMethod.EQUITY_SHARE

    def test_currency_validation_valid(self) -> None:
        """Valid 3-letter currency codes must be accepted."""
        bc = BoundaryConfig(reporting_currency="usd")
        assert bc.reporting_currency == "USD"  # uppercased

    def test_currency_validation_invalid(self) -> None:
        """Invalid currency code must raise ValueError."""
        with pytest.raises(Exception):
            BoundaryConfig(reporting_currency="1234")


# ===========================================================================
# Tests -- ScopeConfig Defaults & Validation
# ===========================================================================


class TestScopeConfig:
    """Tests for ScopeConfig Pydantic model."""

    def test_scope_config_defaults(self) -> None:
        """Default ScopeConfig must include Scope 1, 2, and 3."""
        sc = ScopeConfig()
        assert sc.include_scope1 is True
        assert sc.include_scope2 is True
        assert sc.include_scope3 is True
        assert "location_based" in sc.scope2_methods
        assert "market_based" in sc.scope2_methods

    def test_scope3_categories_default(self) -> None:
        """Default Scope 3 categories should include common categories."""
        sc = ScopeConfig()
        # At minimum, categories 1, 4, 5 should be included
        for cat in [1, 4, 5]:
            assert cat in sc.scope3_categories, (
                f"Category {cat} missing from defaults"
            )

    def test_scope3_categories_sorted_deduplicated(self) -> None:
        """Scope 3 categories must be sorted and deduplicated."""
        sc = ScopeConfig(scope3_categories=[5, 1, 3, 1, 5])
        assert sc.scope3_categories == [1, 3, 5]

    def test_scope3_category_invalid(self) -> None:
        """Scope 3 category numbers outside 1-15 must raise ValueError."""
        with pytest.raises(Exception):
            ScopeConfig(scope3_categories=[0, 16])

    def test_scope2_methods_invalid(self) -> None:
        """Invalid Scope 2 method must raise ValueError."""
        with pytest.raises(Exception):
            ScopeConfig(scope2_methods=["invalid_method"])


# ===========================================================================
# Tests -- TargetConfig Validation
# ===========================================================================


class TestTargetConfig:
    """Tests for TargetConfig Pydantic model."""

    def test_target_config_defaults(self) -> None:
        """Default TargetConfig must have 1.5C ambition and ACA pathway."""
        tc = TargetConfig()
        assert tc.ambition_level == AmbitionLevel.CELSIUS_1_5
        assert tc.pathway_type == PathwayType.ACA
        assert tc.near_term_target_year == 2030
        assert tc.long_term_target_year == 2050

    def test_coverage_scope1_2_defaults(self) -> None:
        """Scope 1+2 coverage must default to 95% per SBTi."""
        tc = TargetConfig()
        assert tc.coverage_scope1_pct == 95.0
        assert tc.coverage_scope2_pct == 95.0

    def test_coverage_scope3_default(self) -> None:
        """Scope 3 coverage must default to 67% per SBTi."""
        tc = TargetConfig()
        assert tc.coverage_scope3_pct == 67.0

    def test_long_term_before_near_term_raises(self) -> None:
        """long_term_target_year before near_term_target_year must raise."""
        with pytest.raises(Exception):
            TargetConfig(
                near_term_target_year=2035,
                long_term_target_year=2035,
            )

    def test_flag_pathway_disabled_by_default(self) -> None:
        """FLAG pathway should be disabled by default."""
        tc = TargetConfig()
        assert tc.flag_pathway_enabled is False


# ===========================================================================
# Tests -- ReductionConfig Validation
# ===========================================================================


class TestReductionConfig:
    """Tests for ReductionConfig Pydantic model."""

    def test_reduction_config_defaults(self) -> None:
        """Default ReductionConfig must have sensible defaults."""
        rc = ReductionConfig()
        assert rc.max_actions == 100
        assert rc.planning_horizon_years == 10
        assert rc.discount_rate_pct == 8.0
        assert rc.include_renewable_procurement is True
        assert rc.include_energy_efficiency is True

    def test_discount_rate_bounds(self) -> None:
        """Discount rate outside 0-30% must raise validation error."""
        with pytest.raises(Exception):
            ReductionConfig(discount_rate_pct=35.0)

    def test_carbon_price_default(self) -> None:
        """Default carbon price must be 80 EUR/tCO2e."""
        rc = ReductionConfig()
        assert rc.carbon_price_eur_per_tco2e == 80.0


# ===========================================================================
# Tests -- OffsetConfig Validation
# ===========================================================================


class TestOffsetConfig:
    """Tests for OffsetConfig Pydantic model."""

    def test_offset_config_defaults(self) -> None:
        """Default OffsetConfig must have sensible defaults."""
        oc = OffsetConfig()
        assert oc.strategy == OffsetStrategy.BOTH
        assert oc.quality_minimum_score == 60
        assert oc.max_nature_based_pct == 50.0
        assert oc.vcmi_target_claim == "SILVER"

    def test_vcmi_claim_validation_valid(self) -> None:
        """Valid VCMI claim tiers must be accepted."""
        for tier in ["SILVER", "GOLD", "PLATINUM"]:
            oc = OffsetConfig(vcmi_target_claim=tier)
            assert oc.vcmi_target_claim == tier

    def test_vcmi_claim_validation_invalid(self) -> None:
        """Invalid VCMI claim tier must raise ValueError."""
        with pytest.raises(Exception):
            OffsetConfig(vcmi_target_claim="BRONZE")

    def test_vcmi_claim_case_insensitive(self) -> None:
        """VCMI claim tier should be uppercased."""
        oc = OffsetConfig(vcmi_target_claim="gold")
        assert oc.vcmi_target_claim == "GOLD"


# ===========================================================================
# Tests -- ReportingConfig Defaults
# ===========================================================================


class TestReportingConfig:
    """Tests for ReportingConfig Pydantic model."""

    def test_reporting_config_defaults(self) -> None:
        """Default ReportingConfig must enable CDP, TCFD, ESRS, SBTi mappings."""
        rc = ReportingConfig()
        assert rc.include_cdp_mapping is True
        assert rc.include_tcfd_mapping is True
        assert rc.include_esrs_mapping is True
        assert rc.include_sbti_mapping is True
        assert rc.language == "en"

    def test_default_formats(self) -> None:
        """Default report formats must include PDF and HTML."""
        rc = ReportingConfig()
        assert ReportFormat.PDF in rc.formats
        assert ReportFormat.HTML in rc.formats


# ===========================================================================
# Tests -- Preset Loading
# ===========================================================================


class TestPresetLoading:
    """Tests for preset configuration loading from YAML files."""

    def test_preset_manufacturing_loads(self) -> None:
        """Manufacturing preset must load without errors."""
        config = PackConfig.from_preset("manufacturing")
        assert config.pack.organization.sector == OrganizationSector.MANUFACTURING

    def test_preset_services_loads(self) -> None:
        """Services preset must load without errors."""
        config = PackConfig.from_preset("services")
        assert config.pack.organization.sector == OrganizationSector.SERVICES

    def test_preset_retail_loads(self) -> None:
        """Retail preset must load without errors."""
        config = PackConfig.from_preset("retail")
        assert config.pack.organization.sector == OrganizationSector.RETAIL

    def test_preset_energy_loads(self) -> None:
        """Energy preset must load without errors."""
        config = PackConfig.from_preset("energy")
        assert config.pack.organization.sector == OrganizationSector.ENERGY

    def test_preset_technology_loads(self) -> None:
        """Technology preset must load without errors."""
        config = PackConfig.from_preset("technology")
        assert config.pack.organization.sector == OrganizationSector.TECHNOLOGY

    def test_preset_sme_general_loads(self) -> None:
        """SME General preset must load without errors."""
        config = PackConfig.from_preset("sme_general")
        assert config.preset_name == "sme_general"

    def test_invalid_preset_raises(self) -> None:
        """Loading an unknown preset must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            PackConfig.from_preset("nonexistent_preset")

    def test_all_supported_presets_exist(self) -> None:
        """All SUPPORTED_PRESETS must have corresponding YAML files."""
        for preset_name in SUPPORTED_PRESETS:
            yaml_path = PRESETS_DIR / f"{preset_name}.yaml"
            assert yaml_path.exists(), (
                f"Preset YAML missing for '{preset_name}': {yaml_path}"
            )


# ===========================================================================
# Tests -- Demo Config
# ===========================================================================


class TestDemoConfig:
    """Tests for the demo configuration."""

    def test_demo_config_loads(self) -> None:
        """Demo config YAML must load without errors."""
        demo_path = DEMO_DIR / "demo_config.yaml"
        assert demo_path.exists(), f"Demo config not found: {demo_path}"

        config = PackConfig.from_yaml(demo_path)
        assert config.pack.organization.name == "CleanTech Industries GmbH"
        assert config.pack.organization.sector == OrganizationSector.MANUFACTURING


# ===========================================================================
# Tests -- PackConfig & Cross-Validation
# ===========================================================================


class TestPackConfig:
    """Tests for PackConfig top-level wrapper and cross-validation."""

    def test_pack_config_loads(self) -> None:
        """Default PackConfig must instantiate without errors."""
        config = PackConfig()
        assert config.pack_id == "PACK-021-net-zero-starter"
        assert config.config_version == "1.0.0"

    def test_config_hash_deterministic(self) -> None:
        """get_config_hash must produce deterministic SHA-256."""
        c1 = PackConfig()
        c2 = PackConfig()
        assert c1.get_config_hash() == c2.get_config_hash()
        assert len(c1.get_config_hash()) == 64

    def test_base_year_after_reporting_year_raises(self) -> None:
        """base_year after reporting_year must raise ValidationError."""
        with pytest.raises(Exception):
            NetZeroStarterConfig(
                base_year=2030,
                reporting_year=2025,
            )

    def test_config_cross_validation_empty_org_name(self) -> None:
        """Cross-validation must warn when organization name is empty."""
        config = NetZeroStarterConfig(organization=OrganizationConfig(name=""))
        warnings = validate_config(config)
        assert any("name" in w.lower() for w in warnings)

    def test_config_cross_validation_few_scope3_cats(self) -> None:
        """Cross-validation must warn if fewer than 3 Scope 3 categories."""
        config = NetZeroStarterConfig(
            scope=ScopeConfig(scope3_categories=[1, 2]),
        )
        warnings = validate_config(config)
        assert any("scope 3" in w.lower() for w in warnings)

    def test_config_cross_validation_no_frameworks(self) -> None:
        """Cross-validation must warn if no reporting frameworks enabled."""
        config = NetZeroStarterConfig(
            reporting=ReportingConfig(
                include_cdp_mapping=False,
                include_tcfd_mapping=False,
                include_esrs_mapping=False,
                include_sbti_mapping=False,
            ),
        )
        warnings = validate_config(config)
        assert any("framework" in w.lower() for w in warnings)

    def test_get_enabled_engines(self) -> None:
        """get_enabled_engines must return at least 5 engines."""
        config = NetZeroStarterConfig()
        engines = config.get_enabled_engines()
        assert len(engines) >= 5
        assert "baseline_inventory" in engines
        assert "target_setting" in engines

    def test_preset_with_overrides(self) -> None:
        """Preset loading with overrides must apply the overrides."""
        config = PackConfig.from_preset(
            "manufacturing",
            overrides={"reporting_year": 2026},
        )
        assert config.pack.reporting_year == 2026
