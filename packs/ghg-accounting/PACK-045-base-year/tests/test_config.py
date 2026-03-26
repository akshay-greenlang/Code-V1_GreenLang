# -*- coding: utf-8 -*-
"""
Tests for PACK-045 config/pack_config.py

Covers PackConfig, BaseYearManagementConfig, all enums, sub-configs,
presets, validation, and utility functions.
Target: ~150 tests.
"""

import os
import pytest
from decimal import Decimal
from pathlib import Path

import sys
PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from config.pack_config import (
    PackConfig,
    BaseYearManagementConfig,
    BaseYearSelectionConfig,
    RecalculationPolicyConfig,
    TriggerConfig,
    SignificanceConfig,
    AdjustmentConfig,
    TimeSeriesConfig,
    TargetTrackingConfig,
    AuditConfig,
    ReportingConfig,
    GWPConfig,
    ScopeConfig,
    NotificationConfig,
    PerformanceConfig,
    SecurityConfig,
    IntegrationConfig,
    BaseYearType,
    RecalculationTriggerType,
    SignificanceMethod,
    AdjustmentApproach,
    ConsolidationApproach,
    TargetType,
    SBTiAmbitionLevel,
    AuditLevel,
    ReportingFramework,
    GWPVersion,
    ScopeType,
    OutputFormat,
    NotificationChannel,
    SectorType,
    AVAILABLE_PRESETS,
    SECTOR_INFO,
    load_preset,
    validate_config,
    get_default_config,
    list_available_presets,
    get_sector_info,
)


# ============================================================================
# Enum Tests
# ============================================================================

class TestEnums:
    """Tests for all configuration enums."""

    def test_base_year_type_values(self):
        assert BaseYearType.FIXED == "FIXED"
        assert BaseYearType.ROLLING_3YR == "ROLLING_3YR"
        assert BaseYearType.ROLLING_5YR == "ROLLING_5YR"

    def test_base_year_type_count(self):
        assert len(BaseYearType) == 3

    def test_recalculation_trigger_type_values(self):
        assert RecalculationTriggerType.ACQUISITION == "ACQUISITION"
        assert RecalculationTriggerType.DIVESTITURE == "DIVESTITURE"
        assert RecalculationTriggerType.MERGER == "MERGER"
        assert RecalculationTriggerType.METHODOLOGY_CHANGE == "METHODOLOGY_CHANGE"
        assert RecalculationTriggerType.ERROR_CORRECTION == "ERROR_CORRECTION"
        assert RecalculationTriggerType.SOURCE_BOUNDARY_CHANGE == "SOURCE_BOUNDARY_CHANGE"
        assert RecalculationTriggerType.OUTSOURCING_INSOURCING == "OUTSOURCING_INSOURCING"

    def test_recalculation_trigger_type_count(self):
        assert len(RecalculationTriggerType) == 7

    def test_significance_method_values(self):
        assert SignificanceMethod.INDIVIDUAL == "INDIVIDUAL"
        assert SignificanceMethod.CUMULATIVE == "CUMULATIVE"
        assert SignificanceMethod.COMBINED == "COMBINED"

    def test_adjustment_approach_values(self):
        assert AdjustmentApproach.PRO_RATA == "PRO_RATA"
        assert AdjustmentApproach.FULL_YEAR == "FULL_YEAR"
        assert AdjustmentApproach.WEIGHTED_AVERAGE == "WEIGHTED_AVERAGE"

    def test_consolidation_approach_values(self):
        assert ConsolidationApproach.EQUITY_SHARE == "EQUITY_SHARE"
        assert ConsolidationApproach.OPERATIONAL_CONTROL == "OPERATIONAL_CONTROL"
        assert ConsolidationApproach.FINANCIAL_CONTROL == "FINANCIAL_CONTROL"

    def test_target_type_values(self):
        assert TargetType.ABSOLUTE == "ABSOLUTE"
        assert TargetType.INTENSITY == "INTENSITY"
        assert TargetType.BOTH == "BOTH"

    def test_sbti_ambition_level_values(self):
        assert SBTiAmbitionLevel.WELL_BELOW_2C == "WELL_BELOW_2C"
        assert SBTiAmbitionLevel.ONE_POINT_FIVE_C == "ONE_POINT_FIVE_C"
        assert SBTiAmbitionLevel.NET_ZERO == "NET_ZERO"

    def test_audit_level_values(self):
        assert AuditLevel.INTERNAL == "INTERNAL"
        assert AuditLevel.LIMITED_ASSURANCE == "LIMITED_ASSURANCE"
        assert AuditLevel.REASONABLE_ASSURANCE == "REASONABLE_ASSURANCE"

    def test_reporting_framework_values(self):
        expected = {"GHG_PROTOCOL", "ISO_14064", "ESRS_E1", "CDP", "SBTI", "SEC", "SB_253", "TCFD"}
        actual = {f.value for f in ReportingFramework}
        assert actual == expected

    def test_gwp_version_values(self):
        assert GWPVersion.AR4 == "AR4"
        assert GWPVersion.AR5 == "AR5"
        assert GWPVersion.AR6 == "AR6"

    def test_scope_type_values(self):
        assert ScopeType.SCOPE_1 == "SCOPE_1"
        assert ScopeType.SCOPE_2_LOCATION == "SCOPE_2_LOCATION"
        assert ScopeType.SCOPE_2_MARKET == "SCOPE_2_MARKET"
        assert ScopeType.SCOPE_3 == "SCOPE_3"

    def test_output_format_values(self):
        assert len(OutputFormat) == 5
        assert OutputFormat.MARKDOWN == "MARKDOWN"
        assert OutputFormat.PDF == "PDF"

    def test_notification_channel_values(self):
        assert len(NotificationChannel) == 4
        assert NotificationChannel.EMAIL == "EMAIL"

    def test_sector_type_values(self):
        assert len(SectorType) == 8
        assert SectorType.MANUFACTURING == "MANUFACTURING"
        assert SectorType.SME == "SME"


# ============================================================================
# Sub-Config Tests
# ============================================================================

class TestBaseYearSelectionConfig:
    """Tests for BaseYearSelectionConfig."""

    def test_defaults(self):
        cfg = BaseYearSelectionConfig()
        assert cfg.type == BaseYearType.FIXED
        assert cfg.base_year == 2022
        assert cfg.min_data_quality_score == 3.0
        assert cfg.min_completeness_pct == 90.0
        assert cfg.earliest_year == 2015
        assert cfg.latest_year == 2025

    def test_valid_custom(self):
        cfg = BaseYearSelectionConfig(base_year=2020, earliest_year=2018, latest_year=2025)
        assert cfg.base_year == 2020

    def test_base_year_before_earliest_raises(self):
        with pytest.raises((ValueError, Exception)):
            BaseYearSelectionConfig(base_year=2014, earliest_year=2015)

    def test_base_year_after_latest_raises(self):
        with pytest.raises(ValueError, match="cannot be later"):
            BaseYearSelectionConfig(base_year=2026, latest_year=2025)

    def test_rolling_type(self):
        cfg = BaseYearSelectionConfig(type=BaseYearType.ROLLING_3YR)
        assert cfg.type == BaseYearType.ROLLING_3YR


class TestRecalculationPolicyConfig:
    def test_defaults(self):
        cfg = RecalculationPolicyConfig()
        assert cfg.significance_threshold_pct == 5.0
        assert cfg.sbti_threshold_pct == 5.0
        assert cfg.cumulative_threshold_pct == 10.0
        assert cfg.auto_detect_triggers is True
        assert cfg.require_approval is True

    def test_custom_thresholds(self):
        cfg = RecalculationPolicyConfig(significance_threshold_pct=3.0)
        assert cfg.significance_threshold_pct == 3.0


class TestTriggerConfig:
    def test_defaults(self):
        cfg = TriggerConfig()
        assert len(cfg.enabled_triggers) == 7
        assert cfg.detection_frequency == "QUARTERLY"
        assert cfg.lookback_months == 12

    def test_detection_frequency_validation(self):
        cfg = TriggerConfig(detection_frequency="monthly")
        assert cfg.detection_frequency == "MONTHLY"

    def test_invalid_detection_frequency(self):
        with pytest.raises(ValueError):
            TriggerConfig(detection_frequency="WEEKLY")


class TestSignificanceConfig:
    def test_defaults(self):
        cfg = SignificanceConfig()
        assert cfg.method == SignificanceMethod.COMBINED
        assert cfg.individual_threshold_pct == 5.0
        assert cfg.cumulative_threshold_pct == 10.0

    def test_custom(self):
        cfg = SignificanceConfig(individual_threshold_pct=3.0, cumulative_threshold_pct=8.0)
        assert cfg.individual_threshold_pct == 3.0


class TestAdjustmentConfig:
    def test_defaults(self):
        cfg = AdjustmentConfig()
        assert cfg.approach == AdjustmentApproach.PRO_RATA
        assert cfg.pro_rata_method == "CALENDAR_DAYS"
        assert cfg.retain_original is True

    def test_pro_rata_method_validation(self):
        cfg = AdjustmentConfig(pro_rata_method="operating_days")
        assert cfg.pro_rata_method == "OPERATING_DAYS"

    def test_invalid_pro_rata_method(self):
        with pytest.raises(ValueError):
            AdjustmentConfig(pro_rata_method="INVALID")


class TestTimeSeriesConfig:
    def test_defaults(self):
        cfg = TimeSeriesConfig()
        assert cfg.min_years == 3
        assert cfg.max_gap_years == 1
        assert cfg.normalization_enabled is True
        assert cfg.interpolation_method == "LINEAR"

    def test_interpolation_validation(self):
        cfg = TimeSeriesConfig(interpolation_method="spline")
        assert cfg.interpolation_method == "SPLINE"

    def test_invalid_interpolation(self):
        with pytest.raises(ValueError):
            TimeSeriesConfig(interpolation_method="CUBIC")


class TestTargetTrackingConfig:
    def test_defaults(self):
        cfg = TargetTrackingConfig()
        assert cfg.target_type == TargetType.ABSOLUTE
        assert cfg.sbti_ambition == SBTiAmbitionLevel.ONE_POINT_FIVE_C
        assert cfg.near_term_target_year == 2030
        assert cfg.long_term_target_year == 2050
        assert cfg.annual_reduction_rate_pct == 4.2

    def test_near_term_after_long_term_raises(self):
        with pytest.raises((ValueError, Exception)):
            TargetTrackingConfig(near_term_target_year=2050, long_term_target_year=2050)


class TestAuditConfig:
    def test_defaults(self):
        cfg = AuditConfig()
        assert cfg.audit_level == AuditLevel.LIMITED_ASSURANCE
        assert cfg.sha256_provenance is True
        assert cfg.evidence_retention_years == 7


class TestReportingConfig:
    def test_defaults(self):
        cfg = ReportingConfig()
        assert ReportingFramework.GHG_PROTOCOL in cfg.frameworks
        assert ReportingFramework.CDP in cfg.frameworks
        assert cfg.output_format == OutputFormat.HTML
        assert cfg.output_language == "en"


class TestGWPConfig:
    def test_defaults(self):
        cfg = GWPConfig()
        assert cfg.version == GWPVersion.AR5
        assert cfg.include_seven_gases is True
        assert cfg.custom_gwp_overrides == {}

    def test_negative_gwp_override_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            GWPConfig(custom_gwp_overrides={"CH4": -1.0})

    def test_valid_gwp_override(self):
        cfg = GWPConfig(custom_gwp_overrides={"CH4": 30.0})
        assert cfg.custom_gwp_overrides["CH4"] == 30.0


class TestScopeConfig:
    def test_defaults(self):
        cfg = ScopeConfig()
        assert cfg.include_scope_1 is True
        assert cfg.include_scope_2 is True
        assert cfg.include_scope_3 is False
        assert cfg.scope_3_categories == []

    def test_scope3_auto_default_categories(self):
        cfg = ScopeConfig(include_scope_3=True)
        assert cfg.scope_3_categories == [1, 2, 3]

    def test_scope3_invalid_category_raises(self):
        with pytest.raises(ValueError, match="must be 1-15"):
            ScopeConfig(scope_3_categories=[0, 16])

    def test_scope3_categories_sorted_deduplicated(self):
        cfg = ScopeConfig(include_scope_3=True, scope_3_categories=[5, 3, 3, 1])
        assert cfg.scope_3_categories == [1, 3, 5]


class TestNotificationConfig:
    def test_defaults(self):
        cfg = NotificationConfig()
        assert NotificationChannel.EMAIL in cfg.channels
        assert cfg.notify_on_trigger is True


class TestPerformanceConfig:
    def test_defaults(self):
        cfg = PerformanceConfig()
        assert cfg.batch_size == 500
        assert cfg.cache_ttl_seconds == 3600


class TestSecurityConfig:
    def test_defaults(self):
        cfg = SecurityConfig()
        assert cfg.rbac_enabled is True
        assert "admin" in cfg.roles
        assert "viewer" in cfg.roles


class TestIntegrationConfig:
    def test_defaults(self):
        cfg = IntegrationConfig()
        assert cfg.pack041_enabled is True
        assert cfg.pack042_enabled is True
        assert cfg.pack043_enabled is False
        assert cfg.pack044_enabled is True


# ============================================================================
# BaseYearManagementConfig Tests
# ============================================================================

class TestBaseYearManagementConfig:
    def test_defaults(self):
        cfg = BaseYearManagementConfig()
        assert cfg.company_name == ""
        assert cfg.sector_type == SectorType.CORPORATE_OFFICE
        assert cfg.consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL
        assert cfg.country == "DE"
        assert cfg.reporting_year == 2026

    def test_sme_forces_threshold_up(self):
        cfg = BaseYearManagementConfig(
            sector_type=SectorType.SME,
            significance=SignificanceConfig(individual_threshold_pct=3.0, cumulative_threshold_pct=15.0),
        )
        assert cfg.significance.individual_threshold_pct == 10.0

    def test_sme_disables_scope3(self):
        cfg = BaseYearManagementConfig(
            sector_type=SectorType.SME,
            scope=ScopeConfig(include_scope_3=True),
        )
        assert cfg.scope.include_scope_3 is False

    def test_sme_downgrades_assurance(self):
        cfg = BaseYearManagementConfig(
            sector_type=SectorType.SME,
            audit=AuditConfig(audit_level=AuditLevel.REASONABLE_ASSURANCE),
        )
        assert cfg.audit.audit_level == AuditLevel.INTERNAL

    def test_threshold_consistency_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            BaseYearManagementConfig(
                significance=SignificanceConfig(
                    individual_threshold_pct=15.0,
                    cumulative_threshold_pct=10.0,
                )
            )

    def test_custom_company(self):
        cfg = BaseYearManagementConfig(company_name="Acme Corp", country="US")
        assert cfg.company_name == "Acme Corp"
        assert cfg.country == "US"

    def test_all_subconfigs_present(self):
        cfg = BaseYearManagementConfig()
        assert cfg.base_year_selection is not None
        assert cfg.recalculation_policy is not None
        assert cfg.trigger is not None
        assert cfg.significance is not None
        assert cfg.adjustment is not None
        assert cfg.time_series is not None
        assert cfg.target_tracking is not None
        assert cfg.audit is not None
        assert cfg.reporting is not None
        assert cfg.gwp is not None
        assert cfg.scope is not None
        assert cfg.notification is not None
        assert cfg.performance is not None
        assert cfg.security is not None
        assert cfg.integration is not None


# ============================================================================
# PackConfig Tests
# ============================================================================

class TestPackConfig:
    def test_defaults(self):
        pc = PackConfig()
        assert pc.pack_id == "PACK-045-base-year"
        assert pc.config_version == "1.0.0"
        assert pc.preset_name is None
        assert pc.pack is not None

    def test_config_hash_deterministic(self):
        pc1 = PackConfig()
        pc2 = PackConfig()
        assert pc1.get_config_hash() == pc2.get_config_hash()

    def test_config_hash_changes_with_override(self):
        pc1 = PackConfig()
        pc2 = PackConfig(pack=BaseYearManagementConfig(company_name="Different"))
        assert pc1.get_config_hash() != pc2.get_config_hash()

    def test_validate_completeness_empty_company(self):
        pc = PackConfig()
        warnings = pc.validate_completeness()
        assert any("company_name" in w.lower() for w in warnings)

    def test_validate_completeness_clean(self):
        cfg = BaseYearManagementConfig(
            company_name="Test Corp",
            base_year_selection=BaseYearSelectionConfig(base_year=2022),
            reporting_year=2026,
        )
        pc = PackConfig(pack=cfg)
        warnings = pc.validate_completeness()
        # Should have no company_name warning
        assert not any("no company_name" in w.lower() for w in warnings)

    def test_merge(self):
        base = PackConfig()
        merged = PackConfig.merge(base, {"company_name": "Merged Corp"})
        assert merged.pack.company_name == "Merged Corp"
        # Original not mutated
        assert base.pack.company_name == ""

    def test_deep_merge_nested(self):
        base = PackConfig()
        merged = PackConfig.merge(base, {
            "significance": {"individual_threshold_pct": 3.0}
        })
        assert merged.pack.significance.individual_threshold_pct == 3.0
        # Other fields preserved
        assert merged.pack.significance.cumulative_threshold_pct == 10.0

    def test_from_preset_corporate_office(self):
        pc = load_preset("corporate_office")
        assert pc.preset_name == "corporate_office"
        assert pc.pack is not None

    def test_from_preset_manufacturing(self):
        pc = load_preset("manufacturing")
        assert pc.preset_name == "manufacturing"

    def test_from_preset_energy_utility(self):
        pc = load_preset("energy_utility")
        assert pc.preset_name == "energy_utility"

    def test_from_preset_transport_logistics(self):
        pc = load_preset("transport_logistics")
        assert pc.preset_name == "transport_logistics"

    def test_from_preset_food_agriculture(self):
        pc = load_preset("food_agriculture")
        assert pc.preset_name == "food_agriculture"

    def test_from_preset_real_estate(self):
        pc = load_preset("real_estate")
        assert pc.preset_name == "real_estate"

    def test_from_preset_healthcare(self):
        pc = load_preset("healthcare")
        assert pc.preset_name == "healthcare"

    def test_from_preset_sme_simplified(self):
        pc = load_preset("sme_simplified")
        assert pc.preset_name == "sme_simplified"

    def test_from_preset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset("nonexistent_preset")

    def test_from_preset_with_overrides(self):
        pc = load_preset("manufacturing", overrides={"company_name": "Override Corp"})
        assert pc.pack.company_name == "Override Corp"

    def test_from_yaml_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            PackConfig.from_yaml("/nonexistent/path.yaml")

    def test_env_overrides_boolean_true(self, monkeypatch):
        monkeypatch.setenv("BASEYEAR_PACK_SCOPE__INCLUDE_SCOPE_3", "true")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("scope", {}).get("include_scope_3") is True

    def test_env_overrides_boolean_false(self, monkeypatch):
        monkeypatch.setenv("BASEYEAR_PACK_SCOPE__INCLUDE_SCOPE_3", "false")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("scope", {}).get("include_scope_3") is False

    def test_env_overrides_int(self, monkeypatch):
        monkeypatch.setenv("BASEYEAR_PACK_REPORTING_YEAR", "2025")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("reporting_year") == 2025

    def test_env_overrides_float(self, monkeypatch):
        monkeypatch.setenv("BASEYEAR_PACK_SIGNIFICANCE__INDIVIDUAL_THRESHOLD_PCT", "3.5")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("significance", {}).get("individual_threshold_pct") == 3.5

    def test_env_overrides_string(self, monkeypatch):
        monkeypatch.setenv("BASEYEAR_PACK_COMPANY_NAME", "Env Corp")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("company_name") == "Env Corp"


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    def test_validate_config_no_company(self):
        cfg = BaseYearManagementConfig()
        warnings = validate_config(cfg)
        assert any("company_name" in w.lower() for w in warnings)

    def test_validate_config_base_year_after_reporting(self):
        cfg = BaseYearManagementConfig(
            company_name="Test",
            base_year_selection=BaseYearSelectionConfig(base_year=2027, earliest_year=2020, latest_year=2030),
            reporting_year=2026,
        )
        warnings = validate_config(cfg)
        assert any("after" in w.lower() for w in warnings)

    def test_validate_config_scope3_without_pack042(self):
        cfg = BaseYearManagementConfig(
            company_name="Test",
            scope=ScopeConfig(include_scope_3=True, scope_3_categories=[1, 2]),
            integration=IntegrationConfig(pack042_enabled=False),
        )
        warnings = validate_config(cfg)
        assert any("pack042" in w.lower() for w in warnings)

    def test_validate_config_external_assurance_no_signature(self):
        cfg = BaseYearManagementConfig(
            company_name="Test",
            audit=AuditConfig(
                audit_level=AuditLevel.LIMITED_ASSURANCE,
                require_digital_signature=False,
            ),
        )
        warnings = validate_config(cfg)
        assert any("digital signature" in w.lower() for w in warnings)

    def test_validate_config_combined_equal_thresholds(self):
        cfg = BaseYearManagementConfig(
            company_name="Test",
            significance=SignificanceConfig(
                method=SignificanceMethod.COMBINED,
                individual_threshold_pct=5.0,
                cumulative_threshold_pct=5.0,
            ),
        )
        warnings = validate_config(cfg)
        assert any("redundant" in w.lower() for w in warnings)

    def test_validate_config_high_threshold_non_sme(self):
        cfg = BaseYearManagementConfig(
            company_name="Test",
            sector_type=SectorType.MANUFACTURING,
            significance=SignificanceConfig(
                individual_threshold_pct=12.0,
                cumulative_threshold_pct=15.0,
            ),
        )
        warnings = validate_config(cfg)
        assert any("unusually high" in w.lower() for w in warnings)

    def test_validate_config_adjustment_without_tracking(self):
        cfg = BaseYearManagementConfig(
            company_name="Test",
            reporting=ReportingConfig(include_adjustment_details=True),
            audit=AuditConfig(track_all_changes=False),
        )
        warnings = validate_config(cfg)
        assert any("track_all_changes" in w for w in warnings)

    def test_get_default_config(self):
        cfg = get_default_config()
        assert cfg.sector_type == SectorType.CORPORATE_OFFICE

    def test_get_default_config_manufacturing(self):
        cfg = get_default_config(SectorType.MANUFACTURING)
        assert cfg.sector_type == SectorType.MANUFACTURING

    def test_list_available_presets(self):
        presets = list_available_presets()
        assert isinstance(presets, dict)
        assert len(presets) == 8
        assert "manufacturing" in presets
        assert "sme_simplified" in presets

    def test_list_available_presets_returns_copy(self):
        p1 = list_available_presets()
        p1["test"] = "should not affect original"
        p2 = list_available_presets()
        assert "test" not in p2

    def test_get_sector_info_enum(self):
        info = get_sector_info(SectorType.MANUFACTURING)
        assert info["name"] == "Manufacturing Facility"

    def test_get_sector_info_string(self):
        info = get_sector_info("ENERGY_UTILITY")
        assert info["name"] == "Energy Utility"

    def test_get_sector_info_unknown(self):
        info = get_sector_info("UNKNOWN")
        assert "typical_base_year_type" in info

    def test_sector_info_keys(self):
        for sector_key in SECTOR_INFO:
            info = SECTOR_INFO[sector_key]
            assert "name" in info
            assert "typical_base_year_type" in info
            assert "significance_threshold_pct" in info

    def test_available_presets_count(self):
        assert len(AVAILABLE_PRESETS) == 8


# ============================================================================
# Validation edge cases
# ============================================================================

class TestValidationEdgeCases:
    def test_near_term_equals_base_year_warns(self):
        cfg = BaseYearManagementConfig(
            company_name="Test",
            base_year_selection=BaseYearSelectionConfig(base_year=2025, earliest_year=2020, latest_year=2025),
            target_tracking=TargetTrackingConfig(near_term_target_year=2025, long_term_target_year=2050),
        )
        warnings = validate_config(cfg)
        assert any("should be after base year" in w for w in warnings)

    def test_clean_config_minimal_warnings(self):
        cfg = BaseYearManagementConfig(
            company_name="Clean Corp",
            base_year_selection=BaseYearSelectionConfig(base_year=2022),
            reporting_year=2026,
            significance=SignificanceConfig(
                individual_threshold_pct=5.0,
                cumulative_threshold_pct=10.0,
            ),
            target_tracking=TargetTrackingConfig(near_term_target_year=2030, long_term_target_year=2050),
            audit=AuditConfig(
                audit_level=AuditLevel.LIMITED_ASSURANCE,
                require_digital_signature=True,
                track_all_changes=True,
            ),
            reporting=ReportingConfig(include_adjustment_details=True),
        )
        warnings = validate_config(cfg)
        assert len(warnings) == 0

    def test_performance_config_boundary_values(self):
        cfg = PerformanceConfig(
            max_recalculation_time_seconds=30,
            batch_size=50,
            cache_ttl_seconds=60,
        )
        assert cfg.max_recalculation_time_seconds == 30
        assert cfg.batch_size == 50

    def test_security_config_custom_roles(self):
        cfg = SecurityConfig(roles=["admin", "custom_role"])
        assert "custom_role" in cfg.roles

    def test_integration_config_all_enabled(self):
        cfg = IntegrationConfig(
            pack041_enabled=True,
            pack042_enabled=True,
            pack043_enabled=True,
            pack044_enabled=True,
            mrv_bridge_enabled=True,
            erp_connector_enabled=True,
        )
        assert cfg.pack043_enabled is True
        assert cfg.erp_connector_enabled is True
