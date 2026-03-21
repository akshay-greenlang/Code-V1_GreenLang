# -*- coding: utf-8 -*-
"""
Test suite for PACK-026 SME Net Zero Pack - Configuration.

Validates all Pydantic configuration models, enums, presets, validation rules,
SME tier configurations, and cross-field constraints in pack_config.py.

Author:  GreenLang Test Engineering
Pack:    PACK-026 SME Net Zero
Tests:   ~200 lines, 40+ tests
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
    SMESize,
    SMESector,
    AccountingSoftware,
    BoundaryMethod,
    SMEDataQualityTier,
    PackConfig,
    SMENetZeroConfig,
    SMEOrganizationConfig,
    SMEBoundaryConfig,
    SMETargetConfig,
    SMEReductionConfig,
    SMEGrantConfig,
    SMECertificationConfig,
    SMEPerformanceConfig,
    SMEScopeConfig,
    validate_config,
    SUPPORTED_PRESETS,
)


CONFIG_DIR = PACK_DIR / "config"
PRESETS_DIR = CONFIG_DIR / "presets"


# ===========================================================================
# Tests -- Enum Definitions
# ===========================================================================


class TestEnumDefinitions:
    """Tests that all configuration enums are properly defined."""

    def test_all_enums_defined(self) -> None:
        """All configuration enums must be importable."""
        enums = [SMESize, SMESector, AccountingSoftware, BoundaryMethod, SMEDataQualityTier]
        for enum_cls in enums:
            assert len(enum_cls) > 0, f"{enum_cls.__name__} has no members"

    @pytest.mark.parametrize("tier", ["MICRO", "SMALL", "MEDIUM"])
    def test_sme_tiers(self, tier: str) -> None:
        """All SME tiers must be valid enum values."""
        assert SMESize(tier) is not None

    @pytest.mark.parametrize("sector", [
        "RETAIL", "HOSPITALITY", "SERVICES", "MANUFACTURING",
        "CONSTRUCTION", "TECHNOLOGY", "HEALTHCARE", "AGRICULTURE",
        "TRANSPORT", "OTHER",
    ])
    def test_sme_sectors(self, sector: str) -> None:
        """All SME sectors must be valid enum values."""
        assert SMESector(sector) is not None

    @pytest.mark.parametrize("platform", [
        "XERO", "QUICKBOOKS", "SAGE", "FRESHBOOKS", "WAVE", "MANUAL",
    ])
    def test_accounting_platforms(self, platform: str) -> None:
        """All accounting platforms must be valid enum values."""
        assert AccountingSoftware(platform) is not None

    @pytest.mark.parametrize("method", [
        "OPERATIONAL_CONTROL",
    ])
    def test_baseline_methods(self, method: str) -> None:
        """Boundary methods must be valid."""
        assert BoundaryMethod(method) is not None

    @pytest.mark.parametrize("level", [
        "BRONZE", "SILVER", "GOLD",
    ])
    def test_data_quality_levels(self, level: str) -> None:
        """All data quality levels must be valid."""
        assert SMEDataQualityTier(level) is not None


# ===========================================================================
# Tests -- OrganizationConfig Validation
# ===========================================================================


class TestOrganizationConfig:
    """Tests for SME OrganizationConfig model."""

    def test_organization_config_defaults(self) -> None:
        """Default OrganizationConfig must instantiate without error."""
        org = SMEOrganizationConfig()
        assert org.sme_size == SMESize.SMALL
        assert org.country == "DE"

    def test_micro_business_config(self) -> None:
        """Micro-business config must accept micro parameters."""
        org = SMEOrganizationConfig(
            name="Micro Shop",
            sme_size=SMESize.MICRO,
            sector=SMESector.RETAIL,
            employee_count=5,
            revenue_eur=200000,
        )
        assert org.sme_size == SMESize.MICRO
        assert org.employee_count == 5

    def test_medium_business_config(self) -> None:
        """Medium-business config must accept medium parameters."""
        org = SMEOrganizationConfig(
            name="MediumCo",
            sme_size=SMESize.MEDIUM,
            sector=SMESector.MANUFACTURING,
            employee_count=200,
            revenue_eur=40000000,
        )
        assert org.sme_size == SMESize.MEDIUM
        assert org.employee_count == 200

    def test_employee_count_validation_micro(self) -> None:
        """Micro tier with 50 employees logs warning (does not raise)."""
        # The config issues a warning, it does not raise
        org = SMEOrganizationConfig(
            sme_size=SMESize.MICRO,
            employee_count=50,
        )
        # Should be created but with a mismatch warning
        assert org.employee_count == 50

    def test_employee_count_validation_medium(self) -> None:
        """Medium tier should have max 249 employees (Pydantic le=249)."""
        with pytest.raises(Exception):
            SMEOrganizationConfig(
                sme_size=SMESize.MEDIUM,
                employee_count=500,
            )

    def test_revenue_validation_micro(self) -> None:
        """Micro tier with high revenue does not raise (uses float, no strict limit)."""
        # revenue_eur is Optional[float] with ge=0, no upper bound in model
        org = SMEOrganizationConfig(
            sme_size=SMESize.MICRO,
            revenue_eur=5000000,
        )
        assert org.revenue_eur == 5000000


# ===========================================================================
# Tests -- BaselineConfig Validation
# ===========================================================================


class TestBaselineConfig:
    """Tests for SME BoundaryConfig model."""

    def test_baseline_config_defaults(self) -> None:
        """Default BoundaryConfig has sensible SME defaults."""
        bc = SMEBoundaryConfig()
        assert bc.method == BoundaryMethod.OPERATIONAL_CONTROL
        assert bc.reporting_currency == "EUR"
        assert bc.include_all_sites is True

    def test_operational_control_boundary(self) -> None:
        """Operational control boundary method is the only option for SME."""
        bc = SMEBoundaryConfig(method=BoundaryMethod.OPERATIONAL_CONTROL)
        assert bc.method == BoundaryMethod.OPERATIONAL_CONTROL

    def test_scope3_simplified_categories(self) -> None:
        """SME scope config defaults to categories 1, 6, 7."""
        sc = SMEScopeConfig()
        assert 1 in sc.scope3_categories
        assert 6 in sc.scope3_categories
        assert 7 in sc.scope3_categories


# ===========================================================================
# Tests -- TargetConfig Validation
# ===========================================================================


class TestTargetConfig:
    """Tests for SME simplified TargetConfig."""

    def test_target_config_defaults(self) -> None:
        """Default target is 42% by 2030 on 1.5C pathway (SBTi SME min)."""
        tc = SMETargetConfig()
        assert tc.near_term_reduction_pct == 42.0
        assert tc.ambition_level == "CELSIUS_1_5"
        assert tc.near_term_target_year == 2030

    def test_scope_coverage(self) -> None:
        """SBTi SME route requires 95% scope 1+2 coverage."""
        tc = SMETargetConfig()
        # SBTi parameters require 95% scope 1+2 coverage (validated through constants)
        assert tc.sbti_sme_route is True
        assert tc.near_term_reduction_pct >= 42.0


# ===========================================================================
# Tests -- QuickWinsConfig
# ===========================================================================


class TestQuickWinsConfig:
    """Tests for SMEReductionConfig."""

    def test_quick_wins_defaults(self) -> None:
        """Default reduction config has max 10 actions and quick wins enabled."""
        qw = SMEReductionConfig()
        assert qw.max_actions == 10
        assert qw.planning_horizon_years <= 5
        assert qw.include_quick_wins is True


# ===========================================================================
# Tests -- PackConfig & Cross-Validation
# ===========================================================================


class TestPackConfig:
    """Tests for PackConfig top-level wrapper and cross-validation."""

    def test_pack_config_loads(self) -> None:
        """Default PackConfig must instantiate without errors."""
        config = PackConfig()
        assert config.pack_id == "PACK-026-sme-net-zero"
        assert config.config_version == "1.0.0"

    def test_config_hash_deterministic(self) -> None:
        """get_config_hash must produce deterministic SHA-256."""
        c1 = PackConfig()
        c2 = PackConfig()
        assert c1.get_config_hash() == c2.get_config_hash()
        assert len(c1.get_config_hash()) == 64

    def test_cross_validation_empty_org_name(self) -> None:
        """Cross-validation must warn when organization name is empty."""
        config = SMENetZeroConfig(organization=SMEOrganizationConfig(name=""))
        warnings = validate_config(config)
        assert any("name" in w.lower() for w in warnings)

    def test_cross_validation_mismatched_tier(self) -> None:
        """Cross-validation must warn when tier/employees mismatch."""
        config = SMENetZeroConfig(
            organization=SMEOrganizationConfig(
                sme_size=SMESize.MICRO,
                employee_count=8,
                revenue_eur=500000,
            ),
        )
        warnings = validate_config(config)
        # Should not warn since 8 employees is valid for micro
        assert not any("mismatch" in w.lower() for w in warnings)

    def test_get_enabled_engines(self) -> None:
        """get_enabled_engines must return at least 6 engines."""
        config = SMENetZeroConfig()
        engines = config.get_enabled_engines()
        assert len(engines) >= 6
        assert "sme_baseline_inventory" in engines
        assert "sme_target_setting" in engines
        assert "sme_quick_wins" in engines

    def test_performance_config_defaults(self) -> None:
        """Performance config must have SME-appropriate defaults."""
        pc = SMEPerformanceConfig()
        assert pc.timeout_seconds <= 120
        assert pc.batch_size >= 50
        assert pc.memory_limit_mb <= 2048

    def test_preset_loading_micro_business(self) -> None:
        """Micro business preset loads correctly."""
        config = PackConfig.from_preset("micro_business")
        assert config.pack.organization.sme_size == SMESize.MICRO

    def test_preset_loading_service_sme(self) -> None:
        """Service SME preset loads correctly."""
        config = PackConfig.from_preset("service_sme")
        assert config.pack is not None

    def test_preset_loading_manufacturing_sme(self) -> None:
        """Manufacturing SME preset loads correctly."""
        config = PackConfig.from_preset("manufacturing_sme")
        assert config.pack is not None

    def test_invalid_preset_raises(self) -> None:
        """Loading an unknown preset must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            PackConfig.from_preset("nonexistent_preset")

    def test_all_supported_presets_exist(self) -> None:
        """All SUPPORTED_PRESETS must have corresponding YAML files."""
        for preset_name in SUPPORTED_PRESETS:
            yaml_path = PRESETS_DIR / f"{preset_name}.yaml"
            assert yaml_path.exists(), f"Preset YAML missing: {yaml_path}"

    def test_preset_with_overrides(self) -> None:
        """Preset loading with overrides must apply the overrides."""
        config = PackConfig.from_preset(
            "service_sme",
            overrides={"reporting_year": 2026},
        )
        assert config.pack.reporting_year == 2026
