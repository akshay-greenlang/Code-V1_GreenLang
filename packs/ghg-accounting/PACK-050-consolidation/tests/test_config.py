# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Configuration Tests

Tests all enum values, config creation, preset loading, validation,
get_default_config(), list_available_presets(), and reference data helpers.

Target: 80-120 tests.
"""

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from config.pack_config import (
    PackConfig,
    ConsolidationPackConfig,
    EntityRegistryConfig,
    OwnershipConfig,
    BoundaryConfig,
    EquityShareConfig,
    ControlApproachConfig,
    EliminationConfig,
    MnAConfig,
    AdjustmentConfig,
    GroupReportingConfig,
    AuditConfig,
    SecurityConfig,
    PerformanceConfig,
    IntegrationConfig,
    AlertConfig,
    MigrationConfig,
    EntityType,
    EntityLifecycle,
    ConsolidationApproach,
    ControlType,
    OwnershipType,
    EliminationType,
    AdjustmentType,
    MnAEventType,
    ReportingFramework,
    DataQualityTier,
    CompletionStatus,
    ApprovalStatus,
    ReportType,
    ExportFormat,
    AlertType,
    ScopeCategory,
    MaterialityThreshold,
    ProRataMethod,
    AVAILABLE_PRESETS,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_OWNERSHIP_THRESHOLDS,
    DEFAULT_ELIMINATION_RULES,
    DEFAULT_MNA_RULES,
    DEFAULT_FRAMEWORK_REQUIREMENTS,
    load_preset,
    validate_config,
    get_default_config,
    list_available_presets,
    get_entity_type_defaults,
    get_ownership_threshold,
    get_elimination_rules,
    get_mna_rules,
    get_framework_requirements,
    _compute_hash,
)


# =========================================================================
# Enum Tests
# =========================================================================

class TestEnumValues:
    """Test all 18 enum types have expected members."""

    def test_entity_type_values(self):
        assert len(EntityType) == 8
        assert EntityType.SUBSIDIARY.value == "SUBSIDIARY"
        assert EntityType.JOINT_VENTURE.value == "JOINT_VENTURE"
        assert EntityType.ASSOCIATE.value == "ASSOCIATE"
        assert EntityType.FRANCHISE.value == "FRANCHISE"

    def test_entity_lifecycle_values(self):
        assert len(EntityLifecycle) == 6
        assert EntityLifecycle.ACTIVE.value == "ACTIVE"
        assert EntityLifecycle.DIVESTED.value == "DIVESTED"
        assert EntityLifecycle.LIQUIDATED.value == "LIQUIDATED"

    def test_consolidation_approach_values(self):
        assert len(ConsolidationApproach) == 3
        assert ConsolidationApproach.EQUITY_SHARE.value == "EQUITY_SHARE"
        assert ConsolidationApproach.OPERATIONAL_CONTROL.value == "OPERATIONAL_CONTROL"
        assert ConsolidationApproach.FINANCIAL_CONTROL.value == "FINANCIAL_CONTROL"

    def test_control_type_values(self):
        assert len(ControlType) == 3
        assert ControlType.OPERATIONAL.value == "OPERATIONAL"
        assert ControlType.FINANCIAL.value == "FINANCIAL"
        assert ControlType.NO_CONTROL.value == "NO_CONTROL"

    def test_ownership_type_values(self):
        assert len(OwnershipType) == 5
        assert OwnershipType.WHOLLY_OWNED.value == "WHOLLY_OWNED"
        assert OwnershipType.MINORITY.value == "MINORITY"
        assert OwnershipType.EQUAL_JV.value == "EQUAL_JV"

    def test_elimination_type_values(self):
        assert len(EliminationType) == 4
        assert EliminationType.ENERGY_TRANSFER.value == "ENERGY_TRANSFER"
        assert EliminationType.WASTE_TRANSFER.value == "WASTE_TRANSFER"
        assert EliminationType.PRODUCT_TRANSFER.value == "PRODUCT_TRANSFER"
        assert EliminationType.SERVICE_TRANSFER.value == "SERVICE_TRANSFER"

    def test_adjustment_type_values(self):
        assert len(AdjustmentType) == 5
        assert AdjustmentType.METHODOLOGY_CHANGE.value == "METHODOLOGY_CHANGE"
        assert AdjustmentType.ERROR_CORRECTION.value == "ERROR_CORRECTION"
        assert AdjustmentType.LATE_SUBMISSION.value == "LATE_SUBMISSION"

    def test_mna_event_type_values(self):
        assert len(MnAEventType) == 6
        assert MnAEventType.ACQUISITION.value == "ACQUISITION"
        assert MnAEventType.DIVESTITURE.value == "DIVESTITURE"
        assert MnAEventType.MERGER.value == "MERGER"
        assert MnAEventType.JV_FORMATION.value == "JV_FORMATION"
        assert MnAEventType.JV_DISSOLUTION.value == "JV_DISSOLUTION"

    def test_reporting_framework_values(self):
        assert len(ReportingFramework) == 9
        assert ReportingFramework.CSRD_ESRS_E1.value == "CSRD_ESRS_E1"
        assert ReportingFramework.CDP.value == "CDP"
        assert ReportingFramework.SEC_CLIMATE.value == "SEC_CLIMATE"
        assert ReportingFramework.NGER.value == "NGER"

    def test_data_quality_tier_values(self):
        assert len(DataQualityTier) == 5
        assert DataQualityTier.VERIFIED.value == "VERIFIED"
        assert DataQualityTier.EXTRAPOLATED.value == "EXTRAPOLATED"

    def test_completion_status_values(self):
        assert len(CompletionStatus) == 4
        assert CompletionStatus.COMPLETE.value == "COMPLETE"
        assert CompletionStatus.OVERDUE.value == "OVERDUE"

    def test_approval_status_values(self):
        assert len(ApprovalStatus) == 5
        assert ApprovalStatus.DRAFT.value == "DRAFT"
        assert ApprovalStatus.APPROVED.value == "APPROVED"
        assert ApprovalStatus.REJECTED.value == "REJECTED"

    def test_report_type_values(self):
        assert len(ReportType) == 10
        assert ReportType.CONSOLIDATED_GHG.value == "CONSOLIDATED_GHG"
        assert ReportType.ELIMINATION_LOG.value == "ELIMINATION_LOG"
        assert ReportType.DASHBOARD.value == "DASHBOARD"

    def test_export_format_values(self):
        assert len(ExportFormat) == 6
        assert ExportFormat.MARKDOWN.value == "MARKDOWN"
        assert ExportFormat.PDF.value == "PDF"
        assert ExportFormat.XLSX.value == "XLSX"

    def test_alert_type_values(self):
        assert len(AlertType) == 6
        assert AlertType.DEADLINE.value == "DEADLINE"
        assert AlertType.MNA_EVENT.value == "MNA_EVENT"

    def test_scope_category_values(self):
        assert len(ScopeCategory) == 4
        assert ScopeCategory.SCOPE_1.value == "SCOPE_1"
        assert ScopeCategory.SCOPE_2_LOCATION.value == "SCOPE_2_LOCATION"
        assert ScopeCategory.SCOPE_2_MARKET.value == "SCOPE_2_MARKET"
        assert ScopeCategory.SCOPE_3.value == "SCOPE_3"

    def test_materiality_threshold_values(self):
        assert len(MaterialityThreshold) == 4
        assert MaterialityThreshold.NONE.value == "NONE"
        assert MaterialityThreshold.TEN_PCT.value == "TEN_PCT"

    def test_pro_rata_method_values(self):
        assert len(ProRataMethod) == 3
        assert ProRataMethod.CALENDAR_DAYS.value == "CALENDAR_DAYS"
        assert ProRataMethod.REPORTING_MONTHS.value == "REPORTING_MONTHS"
        assert ProRataMethod.FINANCIAL_QUARTERS.value == "FINANCIAL_QUARTERS"


# =========================================================================
# Config Creation Tests
# =========================================================================

class TestConfigCreation:
    """Test config creation with defaults and custom values."""

    def test_default_consolidation_config(self):
        config = ConsolidationPackConfig()
        assert config.consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL
        assert config.reporting_year == 2026
        assert config.base_year == 2020
        assert config.country == "DE"
        assert config.currency == "EUR"

    def test_default_entity_registry_config(self):
        config = EntityRegistryConfig()
        assert config.max_entities == 500
        assert config.lifecycle_tracking is True
        assert config.hierarchy_depth_limit == 10
        assert len(config.entity_types_enabled) == 5

    def test_default_ownership_config(self):
        config = OwnershipConfig()
        assert config.multi_tier_enabled is True
        assert config.max_chain_depth == 10
        assert config.effective_equity_method == "MULTIPLICATIVE"
        assert config.circular_ownership_detection is True
        assert config.minority_interest_threshold_pct == Decimal("20.0")

    def test_default_boundary_config(self):
        config = BoundaryConfig()
        assert config.consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL
        assert config.materiality_threshold == MaterialityThreshold.FIVE_PCT
        assert config.materiality_threshold_pct == Decimal("0.05")
        assert config.dual_approach_enabled is False

    def test_default_equity_share_config(self):
        config = EquityShareConfig()
        assert config.default_equity_pct == Decimal("100.0")
        assert config.round_equity_to_dp == 2
        assert config.include_associates is True
        assert config.proportional_scope3 is False

    def test_default_control_approach_config(self):
        config = ControlApproachConfig()
        assert config.control_test_method == "POLICY_AUTHORITY"
        assert config.franchise_inclusion is True
        assert config.spv_consolidation is True

    def test_default_elimination_config(self):
        config = EliminationConfig()
        assert config.elimination_enabled is True
        assert len(config.elimination_types) == 4
        assert config.tolerance_pct == Decimal("5.0")

    def test_default_mna_config(self):
        config = MnAConfig()
        assert config.mna_tracking_enabled is True
        assert config.pro_rata_method == ProRataMethod.CALENDAR_DAYS
        assert config.auto_base_year_recalculation is True
        assert config.lookback_years == 5

    def test_default_adjustment_config(self):
        config = AdjustmentConfig()
        assert config.require_justification is True
        assert config.require_approval is True
        assert config.restatement_window_years == 3

    def test_default_reporting_config(self):
        config = GroupReportingConfig()
        assert config.default_format == ExportFormat.HTML
        assert config.include_entity_breakdown is True
        assert config.trend_years == 3

    def test_default_audit_config(self):
        config = AuditConfig()
        assert config.audit_trail_enabled is True
        assert config.sign_off_levels == 2
        assert config.evidence_retention_years == 7

    def test_default_security_config(self):
        config = SecurityConfig()
        assert config.rls_enabled is True
        assert config.audit_enabled is True
        assert len(config.permissions) == 10

    def test_default_performance_config(self):
        config = PerformanceConfig()
        assert config.max_concurrent_entities == 50
        assert config.batch_size == 100
        assert config.parallel_consolidation is True

    def test_default_integration_config(self):
        config = IntegrationConfig()
        assert config.mrv_agents_count == 30
        assert config.data_agents_count == 20
        assert len(config.pack_dependencies) == 9

    def test_default_alert_config(self):
        config = AlertConfig()
        assert len(config.alert_types_enabled) == 6
        assert config.mna_event_immediate is True

    def test_default_migration_config(self):
        config = MigrationConfig()
        assert config.schema_name == "ghg_consolidation"
        assert config.table_prefix == "gl_cons_"
        assert config.migration_start == "V416"
        assert config.migration_end == "V425"

    def test_custom_config_values(self):
        config = ConsolidationPackConfig(
            company_name="Test Corp",
            consolidation_approach=ConsolidationApproach.EQUITY_SHARE,
            reporting_year=2025,
            base_year=2019,
            country="US",
            currency="USD",
            total_entities=10,
        )
        assert config.company_name == "Test Corp"
        assert config.consolidation_approach == ConsolidationApproach.EQUITY_SHARE
        assert config.reporting_year == 2025
        assert config.base_year == 2019
        assert config.country == "US"
        assert config.total_entities == 10

    def test_pack_config_wrapper(self):
        pc = PackConfig()
        assert pc.pack_id == "PACK-050-consolidation"
        assert pc.config_version == "1.0.0"
        assert pc.preset_name is None


# =========================================================================
# Config Validation Tests
# =========================================================================

class TestConfigValidation:
    """Test config validation rules."""

    def test_base_year_after_reporting_year_raises(self):
        with pytest.raises(ValueError, match="base_year.*cannot be after"):
            ConsolidationPackConfig(base_year=2027, reporting_year=2026)

    def test_dual_approach_no_secondary_raises(self):
        with pytest.raises(ValueError, match="secondary_approach"):
            ConsolidationPackConfig(
                boundary=BoundaryConfig(
                    dual_approach_enabled=True,
                    secondary_approach=None,
                )
            )

    def test_dual_approach_same_as_primary_raises(self):
        with pytest.raises(ValueError, match="secondary_approach must differ"):
            ConsolidationPackConfig(
                consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
                boundary=BoundaryConfig(
                    consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
                    dual_approach_enabled=True,
                    secondary_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
                )
            )

    def test_total_entities_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="total_entities.*exceeds"):
            ConsolidationPackConfig(
                total_entities=1000,
                entity_registry=EntityRegistryConfig(max_entities=500),
            )

    def test_invalid_equity_method_raises(self):
        with pytest.raises(ValueError, match="effective_equity_method"):
            OwnershipConfig(effective_equity_method="INVALID")

    def test_invalid_control_test_method_raises(self):
        with pytest.raises(ValueError, match="control_test_method"):
            ControlApproachConfig(control_test_method="INVALID")

    def test_invalid_lease_treatment_raises(self):
        with pytest.raises(ValueError, match="leased_asset_treatment"):
            ControlApproachConfig(leased_asset_treatment="INVALID")

    def test_validate_config_no_company_name(self):
        config = ConsolidationPackConfig()
        warnings = validate_config(config)
        assert any("company_name" in w for w in warnings)

    def test_validate_config_no_total_entities(self):
        config = ConsolidationPackConfig()
        warnings = validate_config(config)
        assert any("total_entities" in w for w in warnings)

    def test_validate_config_equity_approach_warning(self):
        config = ConsolidationPackConfig(
            consolidation_approach=ConsolidationApproach.EQUITY_SHARE,
        )
        warnings = validate_config(config)
        assert any("equity percentages" in w.lower() for w in warnings)

    def test_validate_config_operational_control_franchise(self):
        config = ConsolidationPackConfig(
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            control_approach=ControlApproachConfig(franchise_inclusion=False),
        )
        warnings = validate_config(config)
        assert any("franchise" in w.lower() for w in warnings)

    def test_validate_config_csrd_without_scope3(self):
        config = ConsolidationPackConfig(
            scopes_in_scope=[ScopeCategory.SCOPE_1, ScopeCategory.SCOPE_2_LOCATION],
            reporting=GroupReportingConfig(
                reporting_frameworks=[ReportingFramework.CSRD_ESRS_E1],
            ),
        )
        warnings = validate_config(config)
        assert any("scope 3" in w.lower() for w in warnings)

    def test_validate_config_sec_without_financial_control(self):
        config = ConsolidationPackConfig(
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            reporting=GroupReportingConfig(
                reporting_frameworks=[ReportingFramework.SEC_CLIMATE],
            ),
        )
        warnings = validate_config(config)
        assert any("sec" in w.lower() for w in warnings)

    def test_validate_config_restatement_window_short(self):
        config = ConsolidationPackConfig(
            adjustment=AdjustmentConfig(restatement_window_years=1),
        )
        warnings = validate_config(config)
        assert any("restatement" in w.lower() for w in warnings)

    def test_validate_config_sign_off_one_level(self):
        config = ConsolidationPackConfig(
            audit=AuditConfig(require_sign_off=True, sign_off_levels=1),
        )
        warnings = validate_config(config)
        assert any("sign-off" in w.lower() or "sign_off" in w.lower() for w in warnings)

    def test_valid_config_from_dict(self, sample_config):
        config = ConsolidationPackConfig(**sample_config)
        assert config.company_name == "GreenTest Holdings AG"
        assert config.reporting_year == 2026


# =========================================================================
# Preset Tests
# =========================================================================

class TestPresets:
    """Test preset loading for all 8 presets."""

    def test_list_available_presets(self):
        presets = list_available_presets()
        assert isinstance(presets, dict)
        assert len(presets) == 8
        expected = [
            "corporate_conglomerate", "financial_holding", "jv_partnership",
            "multinational", "private_equity", "real_estate_fund",
            "public_company", "sme_group",
        ]
        for name in expected:
            assert name in presets

    def test_load_corporate_conglomerate_preset(self):
        pc = load_preset("corporate_conglomerate")
        assert pc.preset_name == "corporate_conglomerate"
        assert pc.pack.entity_registry.max_entities == 5000
        assert pc.pack.boundary.dual_approach_enabled is True

    def test_load_financial_holding_preset(self):
        pc = load_preset("financial_holding")
        assert pc.preset_name == "financial_holding"
        assert pc.pack.equity_share.round_equity_to_dp == 4

    def test_load_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset("nonexistent_preset")

    def test_preset_with_overrides(self):
        pc = load_preset(
            "corporate_conglomerate",
            overrides={"company_name": "Override Corp"},
        )
        assert pc.pack.company_name == "Override Corp"

    def test_preset_config_hash_deterministic(self):
        pc1 = load_preset("corporate_conglomerate")
        pc2 = load_preset("corporate_conglomerate")
        assert pc1.get_config_hash() == pc2.get_config_hash()

    def test_preset_config_hash_is_sha256(self):
        pc = load_preset("corporate_conglomerate")
        h = pc.get_config_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    @pytest.mark.parametrize("preset_name", list(AVAILABLE_PRESETS.keys()))
    def test_all_presets_have_description(self, preset_name):
        desc = AVAILABLE_PRESETS[preset_name]
        assert isinstance(desc, str)
        assert len(desc) > 10


# =========================================================================
# get_default_config Tests
# =========================================================================

class TestGetDefaultConfig:
    """Test get_default_config() factory."""

    def test_default_config_operational_control(self):
        config = get_default_config(ConsolidationApproach.OPERATIONAL_CONTROL)
        assert config.consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL

    def test_default_config_equity_share(self):
        config = get_default_config(ConsolidationApproach.EQUITY_SHARE)
        assert config.consolidation_approach == ConsolidationApproach.EQUITY_SHARE

    def test_default_config_financial_control(self):
        config = get_default_config(ConsolidationApproach.FINANCIAL_CONTROL)
        assert config.consolidation_approach == ConsolidationApproach.FINANCIAL_CONTROL

    def test_default_config_returns_valid_instance(self):
        config = get_default_config()
        assert isinstance(config, ConsolidationPackConfig)


# =========================================================================
# Reference Data Helper Tests
# =========================================================================

class TestReferenceDataHelpers:
    """Test reference data lookup functions."""

    @pytest.mark.parametrize("entity_type", [
        "SUBSIDIARY", "JOINT_VENTURE", "ASSOCIATE", "DIVISION",
        "BRANCH", "SPV", "FRANCHISE", "PARTNERSHIP",
    ])
    def test_get_entity_type_defaults(self, entity_type):
        defaults = get_entity_type_defaults(entity_type)
        assert defaults is not None
        assert "description" in defaults
        assert "typical_ownership_pct" in defaults

    def test_get_entity_type_defaults_unknown(self):
        result = get_entity_type_defaults("NONEXISTENT")
        assert result is None

    @pytest.mark.parametrize("ownership_type", [
        "WHOLLY_OWNED", "MAJORITY", "MINORITY", "EQUAL_JV", "ASSOCIATE",
    ])
    def test_get_ownership_threshold(self, ownership_type):
        threshold = get_ownership_threshold(ownership_type)
        assert threshold is not None
        assert "min_pct" in threshold
        assert "max_pct" in threshold

    def test_get_ownership_threshold_unknown(self):
        result = get_ownership_threshold("NONEXISTENT")
        assert result is None

    @pytest.mark.parametrize("elim_type", [
        "ENERGY_TRANSFER", "WASTE_TRANSFER", "PRODUCT_TRANSFER", "SERVICE_TRANSFER",
    ])
    def test_get_elimination_rules(self, elim_type):
        rules = get_elimination_rules(elim_type)
        assert rules is not None
        assert "description" in rules
        assert "scope_impact" in rules

    def test_get_elimination_rules_unknown(self):
        result = get_elimination_rules("NONEXISTENT")
        assert result is None

    @pytest.mark.parametrize("event_type", [
        "ACQUISITION", "DIVESTITURE", "MERGER", "DEMERGER",
        "JV_FORMATION", "JV_DISSOLUTION",
    ])
    def test_get_mna_rules(self, event_type):
        rules = get_mna_rules(event_type)
        assert rules is not None
        assert "boundary_impact" in rules
        assert "ghg_protocol_ref" in rules

    def test_get_mna_rules_unknown(self):
        result = get_mna_rules("NONEXISTENT")
        assert result is None

    @pytest.mark.parametrize("framework", [
        "CSRD_ESRS_E1", "CDP", "GRI_305", "SEC_CLIMATE",
        "SBTI", "IFRS_S2", "UK_SECR", "NGER",
    ])
    def test_get_framework_requirements(self, framework):
        reqs = get_framework_requirements(framework)
        assert reqs is not None
        assert "name" in reqs
        assert "scopes_required" in reqs

    def test_get_framework_requirements_unknown(self):
        result = get_framework_requirements("NONEXISTENT")
        assert result is None


# =========================================================================
# PackConfig Methods Tests
# =========================================================================

class TestPackConfigMethods:
    """Test PackConfig utility methods."""

    def test_to_dict(self):
        pc = PackConfig()
        d = pc.to_dict()
        assert isinstance(d, dict)
        assert "pack" in d
        assert "pack_id" in d

    def test_get_active_scopes(self):
        pc = PackConfig()
        scopes = pc.get_active_scopes()
        assert "SCOPE_1" in scopes
        assert "SCOPE_2_LOCATION" in scopes

    def test_get_active_frameworks(self):
        pc = PackConfig()
        frameworks = pc.get_active_frameworks()
        assert "CSRD_ESRS_E1" in frameworks

    def test_validate_completeness(self):
        pc = PackConfig()
        warnings = pc.validate_completeness()
        assert isinstance(warnings, list)

    def test_merge_configs(self):
        base = PackConfig()
        merged = PackConfig.merge(base, {"company_name": "Merged Corp"})
        assert merged.pack.company_name == "Merged Corp"

    def test_merge_preserves_preset_name(self):
        pc = load_preset("corporate_conglomerate")
        merged = PackConfig.merge(pc, {"company_name": "Merged"})
        assert merged.preset_name == "corporate_conglomerate"

    def test_compute_hash_deterministic(self):
        h1 = _compute_hash("test data")
        h2 = _compute_hash("test data")
        assert h1 == h2
        assert len(h1) == 64

    def test_compute_hash_different_for_different_input(self):
        h1 = _compute_hash("data1")
        h2 = _compute_hash("data2")
        assert h1 != h2

    def test_from_yaml_missing_file(self):
        with pytest.raises(FileNotFoundError):
            PackConfig.from_yaml("nonexistent_file.yaml")
