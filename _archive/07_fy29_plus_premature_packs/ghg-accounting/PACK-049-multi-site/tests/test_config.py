# -*- coding: utf-8 -*-
"""
Tests for PACK-049 config/pack_config.py

Covers all 18 enums, 15 sub-configs, PackConfig (from_preset, from_yaml,
merge, validate, to_dict), environment overrides, and reference data.
Target: ~100 tests.
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
    MultiSitePackConfig,
    SiteRegistryConfig,
    DataCollectionConfig,
    BoundaryConfig,
    RegionalFactorConfig,
    ConsolidationConfig,
    AllocationConfig,
    ComparisonConfig,
    CompletionConfig,
    QualityConfig,
    ReportingConfig,
    SecurityConfig,
    PerformanceConfig,
    IntegrationConfig,
    AlertConfig,
    MigrationConfig,
    FacilityType,
    FacilityLifecycle,
    ConsolidationApproach,
    OwnershipType,
    CollectionPeriodType,
    SubmissionStatus,
    DataEntryMode,
    AllocationMethod,
    LandlordTenantSplit,
    CogenerationType,
    FactorTier,
    FactorSource,
    QualityDimension,
    QualityScore,
    ComparisonKPI,
    ReportType,
    ExportFormat,
    AlertType,
    AVAILABLE_PRESETS,
    DEFAULT_FACILITY_TYPES,
    DEFAULT_QUALITY_WEIGHTS,
    DEFAULT_ALLOCATION_PRIORITIES,
    CONSOLIDATION_APPROACH_GUIDANCE,
    REGIONAL_FACTOR_DATABASES,
    load_preset,
    validate_config,
    get_default_config,
    list_available_presets,
    get_facility_type_defaults,
    get_consolidation_guidance,
    get_regional_factor_database,
    get_quality_weights,
    get_allocation_priorities,
)


# ============================================================================
# Enum Tests (18 enums)
# ============================================================================

class TestEnums:

    # FacilityType
    def test_facility_type_manufacturing(self):
        assert FacilityType.MANUFACTURING == "MANUFACTURING"

    def test_facility_type_count(self):
        assert len(FacilityType) == 22

    def test_facility_type_all_values(self):
        expected = {
            "MANUFACTURING", "OFFICE", "WAREHOUSE", "RETAIL", "DATA_CENTER",
            "LABORATORY", "HOSPITAL", "HOTEL", "RESTAURANT", "SCHOOL",
            "UNIVERSITY", "GOVERNMENT", "MILITARY", "AIRPORT", "PORT",
            "MINE", "REFINERY", "POWER_PLANT", "FARM", "MIXED_USE",
            "DISTRIBUTION_CENTER", "OTHER",
        }
        actual = {ft.value for ft in FacilityType}
        assert actual == expected

    # FacilityLifecycle
    def test_facility_lifecycle_values(self):
        assert FacilityLifecycle.PLANNED == "PLANNED"
        assert FacilityLifecycle.OPERATIONAL == "OPERATIONAL"
        assert FacilityLifecycle.DECOMMISSIONED == "DECOMMISSIONED"

    def test_facility_lifecycle_count(self):
        assert len(FacilityLifecycle) == 7

    # ConsolidationApproach
    def test_consolidation_approach_values(self):
        assert ConsolidationApproach.EQUITY_SHARE == "EQUITY_SHARE"
        assert ConsolidationApproach.OPERATIONAL_CONTROL == "OPERATIONAL_CONTROL"
        assert ConsolidationApproach.FINANCIAL_CONTROL == "FINANCIAL_CONTROL"

    def test_consolidation_approach_count(self):
        assert len(ConsolidationApproach) == 3

    # OwnershipType
    def test_ownership_type_values(self):
        assert OwnershipType.WHOLLY_OWNED == "WHOLLY_OWNED"
        assert OwnershipType.JOINT_VENTURE == "JOINT_VENTURE"
        assert OwnershipType.FRANCHISE == "FRANCHISE"

    def test_ownership_type_count(self):
        assert len(OwnershipType) == 8

    # CollectionPeriodType
    def test_collection_period_values(self):
        assert CollectionPeriodType.MONTHLY == "MONTHLY"
        assert CollectionPeriodType.QUARTERLY == "QUARTERLY"
        assert CollectionPeriodType.ANNUAL == "ANNUAL"

    def test_collection_period_count(self):
        assert len(CollectionPeriodType) == 4

    # SubmissionStatus
    def test_submission_status_values(self):
        assert SubmissionStatus.NOT_STARTED == "NOT_STARTED"
        assert SubmissionStatus.SUBMITTED == "SUBMITTED"
        assert SubmissionStatus.APPROVED == "APPROVED"
        assert SubmissionStatus.REJECTED == "REJECTED"

    def test_submission_status_count(self):
        assert len(SubmissionStatus) == 8

    # DataEntryMode
    def test_data_entry_mode_values(self):
        assert DataEntryMode.MANUAL == "MANUAL"
        assert DataEntryMode.API_PUSH == "API_PUSH"
        assert DataEntryMode.IOT_FEED == "IOT_FEED"

    def test_data_entry_mode_count(self):
        assert len(DataEntryMode) == 5

    # AllocationMethod
    def test_allocation_method_values(self):
        assert AllocationMethod.FLOOR_AREA == "FLOOR_AREA"
        assert AllocationMethod.HEADCOUNT == "HEADCOUNT"
        assert AllocationMethod.REVENUE == "REVENUE"

    def test_allocation_method_count(self):
        assert len(AllocationMethod) == 7

    # LandlordTenantSplit
    def test_landlord_tenant_split_values(self):
        assert LandlordTenantSplit.WHOLE_BUILDING == "WHOLE_BUILDING"
        assert LandlordTenantSplit.SUB_METERED == "SUB_METERED"

    def test_landlord_tenant_split_count(self):
        assert len(LandlordTenantSplit) == 4

    # CogenerationType
    def test_cogeneration_type_values(self):
        assert CogenerationType.EFFICIENCY_METHOD == "EFFICIENCY_METHOD"
        assert CogenerationType.ENERGY_CONTENT_METHOD == "ENERGY_CONTENT_METHOD"
        assert CogenerationType.RESIDUAL_METHOD == "RESIDUAL_METHOD"

    def test_cogeneration_type_count(self):
        assert len(CogenerationType) == 3

    # FactorTier
    def test_factor_tier_values(self):
        assert FactorTier.TIER_3_FACILITY == "TIER_3_FACILITY"
        assert FactorTier.TIER_0_IPCC_DEFAULT == "TIER_0_IPCC_DEFAULT"

    def test_factor_tier_count(self):
        assert len(FactorTier) == 4

    # FactorSource
    def test_factor_source_values(self):
        assert FactorSource.IPCC_2019 == "IPCC_2019"
        assert FactorSource.DEFRA == "DEFRA"
        assert FactorSource.EPA_EGRID == "EPA_EGRID"

    def test_factor_source_count(self):
        assert len(FactorSource) == 10

    # QualityDimension
    def test_quality_dimension_values(self):
        assert QualityDimension.ACCURACY == "ACCURACY"
        assert QualityDimension.COMPLETENESS == "COMPLETENESS"

    def test_quality_dimension_count(self):
        assert len(QualityDimension) == 6

    # QualityScore
    def test_quality_score_values(self):
        assert QualityScore.SCORE_1_VERIFIED == "SCORE_1_VERIFIED"
        assert QualityScore.SCORE_5_PROXY == "SCORE_5_PROXY"

    def test_quality_score_count(self):
        assert len(QualityScore) == 5

    # ComparisonKPI
    def test_comparison_kpi_values(self):
        assert ComparisonKPI.EMISSIONS_PER_M2 == "EMISSIONS_PER_M2"
        assert ComparisonKPI.EMISSIONS_PER_FTE == "EMISSIONS_PER_FTE"

    def test_comparison_kpi_count(self):
        assert len(ComparisonKPI) == 8

    # ReportType
    def test_report_type_values(self):
        assert ReportType.PORTFOLIO_DASHBOARD == "PORTFOLIO_DASHBOARD"
        assert ReportType.SITE_DETAIL == "SITE_DETAIL"

    def test_report_type_count(self):
        assert len(ReportType) == 10

    # ExportFormat
    def test_export_format_values(self):
        assert ExportFormat.MARKDOWN == "MARKDOWN"
        assert ExportFormat.HTML == "HTML"
        assert ExportFormat.JSON == "JSON"
        assert ExportFormat.CSV == "CSV"
        assert ExportFormat.XBRL == "XBRL"

    def test_export_format_count(self):
        assert len(ExportFormat) == 5

    # AlertType
    def test_alert_type_values(self):
        assert AlertType.DEADLINE_APPROACHING == "DEADLINE_APPROACHING"
        assert AlertType.SUBMISSION_OVERDUE == "SUBMISSION_OVERDUE"

    def test_alert_type_count(self):
        assert len(AlertType) == 6


# ============================================================================
# Sub-Config Default Tests (15 sub-configs)
# ============================================================================

class TestSubConfigDefaults:

    def test_site_registry_defaults(self):
        cfg = SiteRegistryConfig()
        assert cfg.max_sites == 500
        assert cfg.lifecycle_tracking is True
        assert cfg.grouping_enabled is True

    def test_data_collection_defaults(self):
        cfg = DataCollectionConfig()
        assert cfg.collection_period == CollectionPeriodType.MONTHLY
        assert cfg.estimation_allowed is True

    def test_boundary_defaults(self):
        cfg = BoundaryConfig()
        assert cfg.consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL
        assert cfg.materiality_threshold == Decimal("0.05")
        assert cfg.de_minimis_threshold == Decimal("0.01")

    def test_regional_factor_defaults(self):
        cfg = RegionalFactorConfig()
        assert cfg.preferred_tier == FactorTier.TIER_2_NATIONAL
        assert cfg.default_source == FactorSource.DEFRA

    def test_consolidation_defaults(self):
        cfg = ConsolidationConfig()
        assert cfg.elimination_enabled is True
        assert cfg.completeness_threshold == Decimal("0.95")
        assert cfg.reconciliation_tolerance == Decimal("0.01")

    def test_allocation_defaults(self):
        cfg = AllocationConfig()
        assert cfg.default_method == AllocationMethod.FLOOR_AREA
        assert cfg.shared_services_enabled is True

    def test_comparison_defaults(self):
        cfg = ComparisonConfig()
        assert cfg.default_kpi == ComparisonKPI.EMISSIONS_PER_M2
        assert cfg.peer_group_min_size == 3
        assert cfg.trend_years == 3

    def test_completion_defaults(self):
        cfg = CompletionConfig()
        assert cfg.completeness_target == Decimal("0.95")
        assert cfg.escalation_enabled is True

    def test_quality_defaults(self):
        cfg = QualityConfig()
        assert cfg.minimum_quality_score == 3
        assert sum(cfg.quality_weights.values()) == Decimal("1.00")

    def test_reporting_defaults(self):
        cfg = ReportingConfig()
        assert cfg.default_format == ExportFormat.HTML
        assert cfg.drill_down_levels == 3

    def test_security_defaults(self):
        cfg = SecurityConfig()
        assert cfg.rls_enabled is True
        assert cfg.audit_enabled is True

    def test_performance_defaults(self):
        cfg = PerformanceConfig()
        assert cfg.max_concurrent_sites == 50
        assert cfg.batch_size == 100

    def test_integration_defaults(self):
        cfg = IntegrationConfig()
        assert cfg.mrv_agents_count == 30
        assert cfg.data_agents_count == 20

    def test_alert_defaults(self):
        cfg = AlertConfig()
        assert len(cfg.alert_types_enabled) >= 4

    def test_migration_defaults(self):
        cfg = MigrationConfig()
        assert cfg.schema_name == "ghg_multisite"
        assert cfg.table_prefix == "ms_"
        assert cfg.migration_start == "V376"


# ============================================================================
# Sub-Config Validation Tests
# ============================================================================

class TestSubConfigValidation:

    def test_data_collection_invalid_strictness_raises(self):
        with pytest.raises(ValueError, match="validation_strictness"):
            DataCollectionConfig(validation_strictness="INVALID")

    def test_regional_factor_invalid_frequency_raises(self):
        with pytest.raises(ValueError):
            RegionalFactorConfig(grid_factor_update_frequency="INVALID")

    def test_consolidation_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ConsolidationConfig(estimation_method="INVALID")

    def test_allocation_invalid_review_freq_raises(self):
        with pytest.raises(ValueError):
            AllocationConfig(allocation_review_frequency="INVALID")

    def test_completion_invalid_gap_freq_raises(self):
        with pytest.raises(ValueError):
            CompletionConfig(gap_report_frequency="INVALID")

    def test_quality_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            QualityConfig(quality_weights={
                "ACCURACY": Decimal("0.50"),
                "COMPLETENESS": Decimal("0.50"),
                "CONSISTENCY": Decimal("0.50"),
                "TIMELINESS": Decimal("0.15"),
                "METHODOLOGY": Decimal("0.10"),
                "DOCUMENTATION": Decimal("0.10"),
            })

    def test_site_registry_max_sites_min(self):
        with pytest.raises(ValueError):
            SiteRegistryConfig(max_sites=0)


# ============================================================================
# MultiSitePackConfig Tests
# ============================================================================

class TestMultiSitePackConfig:

    def test_default_creation(self):
        cfg = MultiSitePackConfig()
        assert cfg.reporting_year == 2026
        assert cfg.base_year == 2020

    def test_base_year_after_reporting_year_raises(self):
        with pytest.raises(ValueError, match="base_year"):
            MultiSitePackConfig(base_year=2027, reporting_year=2026)

    def test_total_sites_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            MultiSitePackConfig(
                total_sites=1000,
                site_registry=SiteRegistryConfig(max_sites=500),
            )

    def test_scopes_in_scope(self):
        cfg = MultiSitePackConfig()
        assert "SCOPE_1" in cfg.scopes_in_scope
        assert "SCOPE_2" in cfg.scopes_in_scope


# ============================================================================
# PackConfig Tests
# ============================================================================

class TestPackConfig:

    def test_default_pack_config(self):
        pc = PackConfig()
        assert pc.pack_id == "PACK-049-multi-site"
        assert pc.config_version == "1.0.0"

    def test_from_preset_corporate_general(self):
        pc = PackConfig.from_preset("corporate_general")
        assert pc.preset_name == "corporate_general"
        assert pc.pack is not None

    def test_from_preset_manufacturing(self):
        pc = PackConfig.from_preset("manufacturing")
        assert pc.preset_name == "manufacturing"

    def test_from_preset_retail_chain(self):
        pc = PackConfig.from_preset("retail_chain")
        assert pc.preset_name == "retail_chain"

    def test_from_preset_real_estate(self):
        pc = PackConfig.from_preset("real_estate")
        assert pc.preset_name == "real_estate"

    def test_from_preset_financial_services(self):
        pc = PackConfig.from_preset("financial_services")
        assert pc.preset_name == "financial_services"

    def test_from_preset_logistics(self):
        pc = PackConfig.from_preset("logistics")
        assert pc.preset_name == "logistics"

    def test_from_preset_healthcare(self):
        pc = PackConfig.from_preset("healthcare")
        assert pc.preset_name == "healthcare"

    def test_from_preset_public_sector(self):
        pc = PackConfig.from_preset("public_sector")
        assert pc.preset_name == "public_sector"

    def test_from_preset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            PackConfig.from_preset("nonexistent_preset")

    def test_from_yaml_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            PackConfig.from_yaml("/nonexistent/path.yaml")

    def test_merge_overrides(self):
        base = PackConfig()
        merged = PackConfig.merge(base, {"company_name": "Merged Corp"})
        assert merged.pack.company_name == "Merged Corp"

    def test_merge_nested_override(self):
        base = PackConfig()
        merged = PackConfig.merge(base, {
            "site_registry": {"max_sites": 1000},
        })
        assert merged.pack.site_registry.max_sites == 1000

    def test_validate_completeness(self):
        pc = PackConfig()
        warnings = pc.validate_completeness()
        assert isinstance(warnings, list)
        # Default config has no company_name
        assert any("company_name" in w for w in warnings)

    def test_to_dict(self):
        pc = PackConfig()
        d = pc.to_dict()
        assert isinstance(d, dict)
        assert "pack" in d
        assert "pack_id" in d

    def test_get_config_hash(self):
        pc = PackConfig()
        h = pc.get_config_hash()
        assert len(h) == 64

    def test_get_config_hash_deterministic(self):
        pc1 = PackConfig()
        pc2 = PackConfig()
        assert pc1.get_config_hash() == pc2.get_config_hash()

    def test_get_active_scopes(self):
        pc = PackConfig()
        scopes = pc.get_active_scopes()
        assert "SCOPE_1" in scopes
        assert "SCOPE_2" in scopes


# ============================================================================
# Environment Override Tests
# ============================================================================

class TestEnvironmentOverrides:

    def test_env_override_string(self, monkeypatch):
        monkeypatch.setenv("MULTISITE_PACK_COMPANY_NAME", "EnvCorp")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("company_name") == "EnvCorp"

    def test_env_override_bool_true(self, monkeypatch):
        monkeypatch.setenv("MULTISITE_PACK_SITE_REGISTRY__LIFECYCLE_TRACKING", "true")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("site_registry", {}).get("lifecycle_tracking") is True

    def test_env_override_int(self, monkeypatch):
        monkeypatch.setenv("MULTISITE_PACK_REPORTING_YEAR", "2027")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("reporting_year") == 2027

    def test_env_override_nested(self, monkeypatch):
        monkeypatch.setenv("MULTISITE_PACK_BOUNDARY__MATERIALITY_THRESHOLD", "0.10")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("boundary", {}).get("materiality_threshold") == 0.10


# ============================================================================
# Reference Data Tests
# ============================================================================

class TestReferenceData:

    def test_available_presets_count(self):
        assert len(AVAILABLE_PRESETS) == 8

    def test_available_presets_keys(self):
        expected = {
            "corporate_general", "manufacturing", "retail_chain",
            "real_estate", "financial_services", "logistics",
            "healthcare", "public_sector",
        }
        assert set(AVAILABLE_PRESETS.keys()) == expected

    def test_default_facility_types_count(self):
        assert len(DEFAULT_FACILITY_TYPES) >= 20

    def test_default_quality_weights_sum(self):
        total = sum(DEFAULT_QUALITY_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_default_allocation_priorities(self):
        assert len(DEFAULT_ALLOCATION_PRIORITIES) == 7

    def test_consolidation_guidance_equity(self):
        guidance = get_consolidation_guidance("EQUITY_SHARE")
        assert guidance is not None
        assert "name" in guidance

    def test_consolidation_guidance_operational(self):
        guidance = get_consolidation_guidance("OPERATIONAL_CONTROL")
        assert guidance is not None

    def test_consolidation_guidance_financial(self):
        guidance = get_consolidation_guidance("FINANCIAL_CONTROL")
        assert guidance is not None

    def test_regional_factor_databases(self):
        assert len(REGIONAL_FACTOR_DATABASES) >= 8

    def test_regional_factor_database_defra(self):
        db = get_regional_factor_database("DEFRA")
        assert db is not None
        assert db["coverage"] == "United Kingdom"

    def test_regional_factor_database_epa(self):
        db = get_regional_factor_database("EPA_EGRID")
        assert db is not None
        assert "United States" in db["coverage"]


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:

    def test_load_preset(self):
        pc = load_preset("corporate_general")
        assert pc.preset_name == "corporate_general"

    def test_validate_config_default(self):
        cfg = MultiSitePackConfig()
        warnings = validate_config(cfg)
        assert isinstance(warnings, list)

    def test_get_default_config(self):
        cfg = get_default_config()
        assert cfg.consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL

    def test_get_default_config_equity(self):
        cfg = get_default_config(ConsolidationApproach.EQUITY_SHARE)
        assert cfg.consolidation_approach == ConsolidationApproach.EQUITY_SHARE

    def test_list_available_presets(self):
        presets = list_available_presets()
        assert len(presets) == 8
        assert "corporate_general" in presets

    def test_get_facility_type_defaults_manufacturing(self):
        defaults = get_facility_type_defaults("MANUFACTURING")
        assert defaults is not None
        assert defaults["emission_profile"] == "energy_intensive"

    def test_get_facility_type_defaults_office(self):
        defaults = get_facility_type_defaults("OFFICE")
        assert defaults is not None
        assert defaults["emission_profile"] == "low_intensity"

    def test_get_facility_type_defaults_unknown(self):
        result = get_facility_type_defaults("NONEXISTENT")
        assert result is None

    def test_get_quality_weights(self):
        weights = get_quality_weights()
        assert sum(weights.values()) == Decimal("1.00")

    def test_get_allocation_priorities(self):
        priorities = get_allocation_priorities()
        assert priorities[0] == "FLOOR_AREA"
        assert len(priorities) == 7
