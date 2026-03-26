# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Configuration Tests (test_config.py)
============================================================

Tests PackConfig, InventoryManagementConfig, all sub-configs, enums,
validators, environment overrides, deep-merge logic, and config hashing.

Target: 80+ test cases.
"""

import os
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

import pytest

from conftest import _load_config_module, PRESETS_DIR, PRESET_NAMES


# ---------------------------------------------------------------------------
# Module-level dynamic import
# ---------------------------------------------------------------------------

_cfg = _load_config_module()

PackConfig = _cfg.PackConfig
InventoryManagementConfig = _cfg.InventoryManagementConfig

PeriodManagementConfig = _cfg.PeriodManagementConfig
DataCollectionConfig = _cfg.DataCollectionConfig
QualityManagementConfig = _cfg.QualityManagementConfig
ChangeManagementConfig = _cfg.ChangeManagementConfig
ReviewApprovalConfig = _cfg.ReviewApprovalConfig
VersioningConfig = _cfg.VersioningConfig
ConsolidationConfig = _cfg.ConsolidationConfig
GapAnalysisConfig = _cfg.GapAnalysisConfig
DocumentationConfig = _cfg.DocumentationConfig
BenchmarkingConfig = _cfg.BenchmarkingConfig
NotificationConfig = _cfg.NotificationConfig
SecurityConfig = _cfg.SecurityConfig
PerformanceConfig = _cfg.PerformanceConfig
AuditTrailConfig = _cfg.AuditTrailConfig
ReportingConfig = _cfg.ReportingConfig

# Enums
InventoryPeriodStatus = _cfg.InventoryPeriodStatus
DataCollectionStatus = _cfg.DataCollectionStatus
QualityLevel = _cfg.QualityLevel
ChangeType = _cfg.ChangeType
ReviewStage = _cfg.ReviewStage
VersionStatus = _cfg.VersionStatus
ConsolidationApproach = _cfg.ConsolidationApproach
GapPriority = _cfg.GapPriority
BenchmarkSource = _cfg.BenchmarkSource
SectorType = _cfg.SectorType
OutputFormat = _cfg.OutputFormat
ReportingFrequency = _cfg.ReportingFrequency
NotificationChannel = _cfg.NotificationChannel
FrameworkType = _cfg.FrameworkType

validate_config = _cfg.validate_config
get_default_config = _cfg.get_default_config
get_sector_info = _cfg.get_sector_info
list_available_presets = _cfg.list_available_presets
AVAILABLE_PRESETS = _cfg.AVAILABLE_PRESETS
SECTOR_INFO = _cfg.SECTOR_INFO


# ===================================================================
# Default Config Tests
# ===================================================================

class TestDefaultConfig:
    """Tests for default InventoryManagementConfig values."""

    def test_default_config_creates(self):
        cfg = InventoryManagementConfig()
        assert cfg is not None
        assert cfg.sector_type == SectorType.OFFICE

    def test_default_company_name_empty(self):
        cfg = InventoryManagementConfig()
        assert cfg.company_name == ""

    def test_default_country_de(self):
        cfg = InventoryManagementConfig()
        assert cfg.country == "DE"

    def test_default_reporting_year(self):
        cfg = InventoryManagementConfig()
        assert cfg.reporting_year == 2026

    def test_default_revenue_none(self):
        cfg = InventoryManagementConfig()
        assert cfg.revenue_meur is None

    def test_default_employees_none(self):
        cfg = InventoryManagementConfig()
        assert cfg.employees_fte is None


class TestSubConfigDefaults:
    """Tests for each sub-config default construction."""

    def test_period_management_defaults(self):
        cfg = PeriodManagementConfig()
        assert cfg.auto_create_periods is True
        assert cfg.lock_after_approval is True
        assert cfg.max_open_periods == 3
        assert cfg.retention_years == 7
        assert cfg.milestone_tracking is True

    def test_data_collection_defaults(self):
        cfg = DataCollectionConfig()
        assert cfg.auto_scheduling is True
        assert cfg.reminder_frequency_days == 7
        assert cfg.escalation_after_days == 21
        assert cfg.default_deadline_days == 30
        assert cfg.min_data_quality_score == 3.0
        assert cfg.collection_frequency == ReportingFrequency.QUARTERLY

    def test_quality_management_defaults(self):
        cfg = QualityManagementConfig()
        assert cfg.enabled is True
        assert cfg.auto_qaqc is True
        assert cfg.completeness_threshold_pct == 95.0
        assert cfg.review_levels == 2

    def test_change_management_defaults(self):
        cfg = ChangeManagementConfig()
        assert cfg.require_impact_assessment is True
        assert cfg.significance_threshold_pct == 5.0
        assert cfg.base_year_recalculation_threshold_pct == 5.0

    def test_review_approval_defaults(self):
        cfg = ReviewApprovalConfig()
        assert len(cfg.review_levels) == 3
        assert ReviewStage.PREPARER in cfg.review_levels

    def test_versioning_defaults(self):
        cfg = VersioningConfig()
        assert cfg.auto_version_on_changes is True
        assert cfg.max_draft_versions == 10
        assert cfg.allow_rollback is True
        assert cfg.immutable_after_finalization is True

    def test_consolidation_defaults(self):
        cfg = ConsolidationConfig()
        assert cfg.approach == ConsolidationApproach.OPERATIONAL_CONTROL
        assert cfg.equity_threshold_pct == 20.0

    def test_gap_analysis_defaults(self):
        cfg = GapAnalysisConfig()
        assert cfg.enabled is True
        assert cfg.methodology_tier_target == 2
        assert cfg.data_quality_target == 4.0

    def test_documentation_defaults(self):
        cfg = DocumentationConfig()
        assert cfg.require_methodology_docs is True
        assert cfg.verification_readiness is True

    def test_benchmarking_defaults(self):
        cfg = BenchmarkingConfig()
        assert cfg.enabled is True
        assert cfg.peer_group_size == 10
        assert BenchmarkSource.SECTOR_AVERAGE in cfg.benchmark_sources

    def test_notification_defaults(self):
        cfg = NotificationConfig()
        assert NotificationChannel.EMAIL in cfg.channels
        assert cfg.deadline_reminders is True

    def test_security_defaults(self):
        cfg = SecurityConfig()
        assert "admin" in cfg.roles
        assert cfg.audit_logging is True
        assert cfg.encryption_at_rest is True

    def test_performance_defaults(self):
        cfg = PerformanceConfig()
        assert cfg.max_entities == 500
        assert cfg.batch_size == 500
        assert cfg.cache_ttl_seconds == 3600

    def test_audit_trail_defaults(self):
        cfg = AuditTrailConfig()
        assert cfg.enabled is True
        assert cfg.sha256_provenance is True
        assert cfg.retention_years == 7

    def test_reporting_defaults(self):
        cfg = ReportingConfig()
        assert cfg.frequency == ReportingFrequency.ANNUAL
        assert OutputFormat.JSON in cfg.formats


# ===================================================================
# Enum Tests
# ===================================================================

class TestEnumValues:
    """Tests for all enum value definitions."""

    @pytest.mark.parametrize("member,value", [
        ("PLANNING", "PLANNING"),
        ("DATA_COLLECTION", "DATA_COLLECTION"),
        ("FINAL", "FINAL"),
        ("AMENDED", "AMENDED"),
        ("ARCHIVED", "ARCHIVED"),
    ])
    def test_inventory_period_status(self, member, value):
        assert InventoryPeriodStatus[member].value == value

    @pytest.mark.parametrize("member,value", [
        ("NOT_STARTED", "NOT_STARTED"),
        ("IN_PROGRESS", "IN_PROGRESS"),
        ("SUBMITTED", "SUBMITTED"),
        ("VALIDATED", "VALIDATED"),
        ("REJECTED", "REJECTED"),
    ])
    def test_data_collection_status(self, member, value):
        assert DataCollectionStatus[member].value == value

    @pytest.mark.parametrize("member", ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"])
    def test_quality_level(self, member):
        assert QualityLevel[member] is not None

    @pytest.mark.parametrize("member", [
        "ORGANIZATIONAL", "METHODOLOGY", "EMISSION_FACTOR",
        "ERROR_CORRECTION", "STRUCTURAL",
    ])
    def test_change_type(self, member):
        assert ChangeType[member] is not None

    @pytest.mark.parametrize("member", ["PREPARER", "REVIEWER", "APPROVER", "VERIFIER"])
    def test_review_stage(self, member):
        assert ReviewStage[member] is not None

    @pytest.mark.parametrize("member", ["DRAFT", "UNDER_REVIEW", "FINAL", "AMENDED", "SUPERSEDED"])
    def test_version_status(self, member):
        assert VersionStatus[member] is not None

    @pytest.mark.parametrize("member", ["EQUITY_SHARE", "OPERATIONAL_CONTROL", "FINANCIAL_CONTROL"])
    def test_consolidation_approach(self, member):
        assert ConsolidationApproach[member] is not None

    @pytest.mark.parametrize("member", ["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    def test_gap_priority(self, member):
        assert GapPriority[member] is not None

    @pytest.mark.parametrize("member", ["SECTOR_AVERAGE", "CDP_PEER", "INTERNAL_HISTORICAL", "CUSTOM"])
    def test_benchmark_source(self, member):
        assert BenchmarkSource[member] is not None

    @pytest.mark.parametrize("member", [
        "OFFICE", "MANUFACTURING", "ENERGY_UTILITY", "TRANSPORT_LOGISTICS",
        "FOOD_AGRICULTURE", "REAL_ESTATE", "HEALTHCARE", "SME",
    ])
    def test_sector_type(self, member):
        assert SectorType[member] is not None

    @pytest.mark.parametrize("member", ["MARKDOWN", "HTML", "JSON", "CSV", "XBRL"])
    def test_output_format(self, member):
        assert OutputFormat[member] is not None

    @pytest.mark.parametrize("member", ["MONTHLY", "QUARTERLY", "ANNUAL"])
    def test_reporting_frequency(self, member):
        assert ReportingFrequency[member] is not None

    @pytest.mark.parametrize("member", ["EMAIL", "SLACK", "TEAMS", "WEBHOOK", "IN_APP"])
    def test_notification_channel(self, member):
        assert NotificationChannel[member] is not None

    @pytest.mark.parametrize("member", [
        "GHG_PROTOCOL", "ESRS_E1", "CDP", "ISO_14064", "SBTI", "SEC", "SB_253",
    ])
    def test_framework_type(self, member):
        assert FrameworkType[member] is not None


# ===================================================================
# Field Validator Tests
# ===================================================================

class TestFieldValidators:
    """Tests for field boundaries and type enforcement."""

    def test_reporting_year_min(self):
        with pytest.raises(Exception):
            InventoryManagementConfig(reporting_year=2019)

    def test_reporting_year_max(self):
        with pytest.raises(Exception):
            InventoryManagementConfig(reporting_year=2036)

    def test_reporting_year_valid(self):
        cfg = InventoryManagementConfig(reporting_year=2025)
        assert cfg.reporting_year == 2025

    def test_max_open_periods_min(self):
        with pytest.raises(Exception):
            PeriodManagementConfig(max_open_periods=0)

    def test_max_open_periods_max(self):
        with pytest.raises(Exception):
            PeriodManagementConfig(max_open_periods=11)

    def test_retention_years_valid(self):
        cfg = PeriodManagementConfig(retention_years=10)
        assert cfg.retention_years == 10

    def test_reminder_frequency_min(self):
        with pytest.raises(Exception):
            DataCollectionConfig(reminder_frequency_days=0)

    def test_min_data_quality_score_bounds(self):
        with pytest.raises(Exception):
            DataCollectionConfig(min_data_quality_score=0.5)

    def test_completeness_threshold_min(self):
        with pytest.raises(Exception):
            QualityManagementConfig(completeness_threshold_pct=79.0)

    def test_review_levels_min(self):
        with pytest.raises(Exception):
            QualityManagementConfig(review_levels=0)

    def test_significance_threshold_min(self):
        with pytest.raises(Exception):
            ChangeManagementConfig(significance_threshold_pct=0.5)

    def test_peer_group_size_min(self):
        with pytest.raises(Exception):
            BenchmarkingConfig(peer_group_size=2)

    def test_equity_threshold_valid(self):
        cfg = ConsolidationConfig(equity_threshold_pct=50.0)
        assert cfg.equity_threshold_pct == 50.0

    def test_methodology_tier_target_bounds(self):
        with pytest.raises(Exception):
            GapAnalysisConfig(methodology_tier_target=4)

    def test_max_entities_bounds(self):
        with pytest.raises(Exception):
            PerformanceConfig(max_entities=5001)

    def test_cache_ttl_bounds(self):
        with pytest.raises(Exception):
            PerformanceConfig(cache_ttl_seconds=50)


# ===================================================================
# SME Simplified Validator Tests
# ===================================================================

class TestSMESimplifiedValidator:
    """Tests for the SME sector model validator."""

    def test_sme_caps_review_levels(self):
        cfg = InventoryManagementConfig(
            sector_type=SectorType.SME,
            quality_management=QualityManagementConfig(review_levels=3),
        )
        assert cfg.quality_management.review_levels == 1

    def test_sme_forces_annual_collection(self):
        cfg = InventoryManagementConfig(
            sector_type=SectorType.SME,
            data_collection=DataCollectionConfig(
                collection_frequency=ReportingFrequency.QUARTERLY
            ),
        )
        assert cfg.data_collection.collection_frequency == ReportingFrequency.ANNUAL

    def test_non_sme_preserves_review_levels(self):
        cfg = InventoryManagementConfig(
            sector_type=SectorType.MANUFACTURING,
            quality_management=QualityManagementConfig(review_levels=3),
        )
        assert cfg.quality_management.review_levels == 3


# ===================================================================
# Config Validation Warnings Tests
# ===================================================================

class TestConfigValidationWarnings:
    """Tests for validate_config returning appropriate warnings."""

    def test_no_company_name_warning(self):
        cfg = InventoryManagementConfig(company_name="")
        warnings = validate_config(cfg)
        assert any("company_name" in w.lower() for w in warnings)

    def test_low_review_levels_non_sme_warning(self):
        cfg = InventoryManagementConfig(
            sector_type=SectorType.MANUFACTURING,
            quality_management=QualityManagementConfig(review_levels=1),
        )
        warnings = validate_config(cfg)
        assert any("review levels" in w.lower() for w in warnings)

    def test_verification_without_evidence_warning(self):
        cfg = InventoryManagementConfig(
            documentation=DocumentationConfig(
                verification_readiness=True,
                require_evidence_links=False,
            ),
        )
        warnings = validate_config(cfg)
        assert any("evidence" in w.lower() for w in warnings)

    def test_benchmarking_no_metrics_warning(self):
        cfg = InventoryManagementConfig(
            benchmarking=BenchmarkingConfig(enabled=True, intensity_metrics=[]),
        )
        warnings = validate_config(cfg)
        assert any("intensity" in w.lower() for w in warnings)

    def test_no_warnings_for_valid_config(self):
        cfg = InventoryManagementConfig(
            company_name="Test Corp",
            sector_type=SectorType.OFFICE,
        )
        warnings = validate_config(cfg)
        # Should have no warnings with default config + company name
        assert len(warnings) == 0


# ===================================================================
# Config Hash Determinism Tests
# ===================================================================

class TestConfigHashDeterminism:
    """Tests for config hash reproducibility."""

    def test_same_config_same_hash(self):
        c1 = PackConfig()
        c2 = PackConfig()
        assert c1.get_config_hash() == c2.get_config_hash()

    def test_different_config_different_hash(self):
        c1 = PackConfig()
        c2 = PackConfig(
            pack=InventoryManagementConfig(company_name="Different Corp")
        )
        assert c1.get_config_hash() != c2.get_config_hash()

    def test_hash_is_sha256_length(self):
        c = PackConfig()
        h = c.get_config_hash()
        assert len(h) == 64

    def test_hash_is_hex(self):
        c = PackConfig()
        h = c.get_config_hash()
        int(h, 16)  # should not raise


# ===================================================================
# Environment Overrides Tests
# ===================================================================

class TestEnvironmentOverrides:
    """Tests for INVMGMT_PACK_* environment variable overrides."""

    def test_env_bool_true(self, monkeypatch):
        monkeypatch.setenv("INVMGMT_PACK_PERIOD_MANAGEMENT__AUTO_CREATE_PERIODS", "true")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("period_management", {}).get("auto_create_periods") is True

    def test_env_bool_false(self, monkeypatch):
        monkeypatch.setenv("INVMGMT_PACK_PERIOD_MANAGEMENT__AUTO_CREATE_PERIODS", "false")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("period_management", {}).get("auto_create_periods") is False

    def test_env_int_value(self, monkeypatch):
        monkeypatch.setenv("INVMGMT_PACK_PERIOD_MANAGEMENT__RETENTION_YEARS", "10")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("period_management", {}).get("retention_years") == 10

    def test_env_float_value(self, monkeypatch):
        monkeypatch.setenv("INVMGMT_PACK_QUALITY_MANAGEMENT__COMPLETENESS_THRESHOLD_PCT", "98.5")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("quality_management", {}).get("completeness_threshold_pct") == 98.5

    def test_env_string_value(self, monkeypatch):
        monkeypatch.setenv("INVMGMT_PACK_COMPANY_NAME", "TestCo")
        overrides = PackConfig._load_env_overrides()
        assert overrides.get("company_name") == "TestCo"

    def test_no_env_returns_empty(self):
        overrides = PackConfig._load_env_overrides()
        # May or may not be empty depending on environment, but should not error
        assert isinstance(overrides, dict)


# ===================================================================
# Deep Merge Tests
# ===================================================================

class TestDeepMerge:
    """Tests for the _deep_merge static method."""

    def test_shallow_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = PackConfig._deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = PackConfig._deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_override_replaces_non_dict(self):
        base = {"a": {"x": 1}}
        override = {"a": "replaced"}
        result = PackConfig._deep_merge(base, override)
        assert result["a"] == "replaced"

    def test_base_unmodified(self):
        base = {"a": 1}
        override = {"a": 2}
        PackConfig._deep_merge(base, override)
        assert base["a"] == 1  # base should not be mutated


# ===================================================================
# PackConfig Construction Tests
# ===================================================================

class TestPackConfigConstruction:
    """Tests for PackConfig creation and methods."""

    def test_default_pack_id(self):
        pc = PackConfig()
        assert pc.pack_id == "PACK-044-inventory-management"

    def test_default_config_version(self):
        pc = PackConfig()
        assert pc.config_version == "1.0.0"

    def test_from_preset_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            PackConfig.from_preset("nonexistent_preset")

    def test_merge_applies_overrides(self):
        base = PackConfig()
        merged = PackConfig.merge(base, {"company_name": "Override Corp"})
        assert merged.pack.company_name == "Override Corp"

    def test_validate_completeness_returns_list(self):
        pc = PackConfig()
        result = pc.validate_completeness()
        assert isinstance(result, list)


# ===================================================================
# Utility Function Tests
# ===================================================================

class TestUtilityFunctions:
    """Tests for module-level utility functions."""

    def test_get_default_config_office(self):
        cfg = get_default_config(SectorType.OFFICE)
        assert cfg.sector_type == SectorType.OFFICE

    def test_get_default_config_manufacturing(self):
        cfg = get_default_config(SectorType.MANUFACTURING)
        assert cfg.sector_type == SectorType.MANUFACTURING

    def test_list_available_presets_returns_dict(self):
        presets = list_available_presets()
        assert isinstance(presets, dict)
        assert "corporate_office" in presets

    def test_get_sector_info_known(self):
        info = get_sector_info(SectorType.MANUFACTURING)
        assert "name" in info
        assert info["name"] == "Manufacturing Facility"

    def test_get_sector_info_unknown_string(self):
        info = get_sector_info("UNKNOWN_SECTOR")
        assert "name" in info

    @pytest.mark.parametrize("sector", list(SectorType))
    def test_all_sectors_have_info(self, sector):
        info = get_sector_info(sector)
        assert info is not None
        assert "name" in info

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_yaml_files_exist(self, preset_name):
        path = PRESETS_DIR / f"{preset_name}.yaml"
        assert path.exists(), f"Preset file missing: {path}"
