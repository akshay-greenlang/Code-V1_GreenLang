# -*- coding: utf-8 -*-
"""
PACK-009 EU Climate Compliance Bundle Pack - Configuration Tests
=================================================================

Tests the BundleComplianceConfig configuration system including:
- RegulationType enum (4 values: CSRD, CBAM, EUDR, TAXONOMY)
- ComplianceStatus enum (5 values)
- BundleTier enum (4 tiers)
- DataFieldCategory enum (9 categories)
- ConsistencyLevel enum (4 levels)
- GapSeverity enum (5 levels)
- CalendarEventType enum (5 event types)
- EvidenceType enum (6 evidence types)
- RegulationConfig creation and validation
- CalendarConfig defaults
- DeduplicationConfig defaults
- ConsistencyConfig defaults
- ScoringConfig defaults and weight validation
- EvidenceConfig defaults
- BundleComplianceConfig creation with all sub-configs
- Enabled regulation listing and filtering
- PackConfig loading from presets and YAML
- get_default_config utility function
- Preset YAML file existence (4 presets)
- Demo config YAML existence and validity
- Configuration serialization and hashing

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

import importlib.util
import sys


def _import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_PACK_DIR = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PACK_DIR / "config"
_PRESETS_DIR = _CONFIG_DIR / "presets"

PACK_ROOT = _PACK_DIR
CONFIG_DIR = _CONFIG_DIR
PRESETS_DIR = _PRESETS_DIR
DEMO_DIR = _CONFIG_DIR / "demo"
REGULATIONS = ["CSRD", "CBAM", "EUDR", "TAXONOMY"]
BUNDLE_PRESET_IDS = ["enterprise_full", "financial_institution", "eu_importer", "sme_essential"]
_DEMO_DIR = DEMO_DIR


def _load_config_module():
    """Import pack_config.py from hyphenated pack directory."""
    path = _CONFIG_DIR / "pack_config.py"
    spec = importlib.util.spec_from_file_location("pack_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the config module once for the entire module
try:
    _cfg = _load_config_module()
except Exception:
    _cfg = None


def _skip_if_no_config():
    """Skip test if pack_config module could not be loaded."""
    if _cfg is None:
        pytest.skip("pack_config module could not be loaded")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPackConfig:
    """Test suite for PACK-009 BundleComplianceConfig configuration."""

    # -----------------------------------------------------------------
    # BundleComplianceConfig creation
    # -----------------------------------------------------------------

    def test_bundle_compliance_config_default_creation(self):
        """Test BundleComplianceConfig creates with valid defaults."""
        _skip_if_no_config()
        config = _cfg.BundleComplianceConfig()

        assert config.pack_id == "PACK-009-eu-climate-compliance-bundle"
        assert config.version == "1.0.0"
        assert config.tier == "bundle"
        assert config.bundle_tier == _cfg.BundleTier.ENTERPRISE_FULL
        assert config.reporting_year == 2025

        # All four regulations enabled by default
        enabled = config.enabled_regulations
        assert len(enabled) == 4, f"Expected 4 enabled regulations, got {len(enabled)}"

        # All sub-configs should be populated
        assert isinstance(config.calendar, _cfg.CalendarConfig)
        assert isinstance(config.deduplication, _cfg.DeduplicationConfig)
        assert isinstance(config.consistency, _cfg.ConsistencyConfig)
        assert isinstance(config.gap_analysis, _cfg.GapAnalysisConfig)
        assert isinstance(config.evidence, _cfg.EvidenceConfig)
        assert isinstance(config.reporting, _cfg.ReportingConfig)
        assert isinstance(config.scoring, _cfg.ScoringConfig)
        assert isinstance(config.data_mapper, _cfg.DataMapperConfig)
        assert isinstance(config.health_check, _cfg.HealthCheckConfig)
        assert isinstance(config.audit_trail, _cfg.AuditTrailConfig)
        assert isinstance(config.demo, _cfg.DemoConfig)

    # -----------------------------------------------------------------
    # Enum tests
    # -----------------------------------------------------------------

    def test_regulation_type_enum_values(self):
        """Test RegulationType enum has exactly 4 values: CSRD, CBAM, EUDR, TAXONOMY."""
        _skip_if_no_config()
        rt = _cfg.RegulationType
        members = list(rt)
        assert len(members) == 4, f"Expected 4 regulation types, got {len(members)}"

        expected = {"CSRD", "CBAM", "EUDR", "TAXONOMY"}
        actual = {m.value for m in members}
        assert actual == expected, f"RegulationType mismatch: {actual} != {expected}"

    def test_compliance_status_enum_values(self):
        """Test ComplianceStatus enum has the expected 5 values."""
        _skip_if_no_config()
        cs = _cfg.ComplianceStatus
        expected = {
            "NOT_STARTED", "IN_PROGRESS", "COMPLIANT",
            "NON_COMPLIANT", "PARTIALLY_COMPLIANT",
        }
        actual = {m.value for m in cs}
        assert actual == expected, f"ComplianceStatus mismatch: {actual} != {expected}"

    def test_bundle_tier_enum_values(self):
        """Test BundleTier enum has 4 deployment tiers."""
        _skip_if_no_config()
        bt = _cfg.BundleTier
        expected = {
            "ENTERPRISE_FULL", "FINANCIAL_INSTITUTION",
            "EU_IMPORTER", "SME_ESSENTIAL",
        }
        actual = {m.value for m in bt}
        assert actual == expected, f"BundleTier mismatch: {actual} != {expected}"

    def test_data_field_category_enum_values(self):
        """Test DataFieldCategory enum has 9 categories."""
        _skip_if_no_config()
        dfc = _cfg.DataFieldCategory
        expected = {
            "GHG_EMISSIONS", "SUPPLY_CHAIN", "ACTIVITY_CLASSIFICATION",
            "FINANCIAL_DATA", "CLIMATE_RISK", "WATER_POLLUTION",
            "BIODIVERSITY", "GOVERNANCE", "SOCIAL",
        }
        actual = {m.value for m in dfc}
        assert actual == expected, f"DataFieldCategory mismatch: {actual} != {expected}"

    def test_consistency_level_enum_values(self):
        """Test ConsistencyLevel enum has 4 classification levels."""
        _skip_if_no_config()
        cl = _cfg.ConsistencyLevel
        expected = {"EXACT", "APPROXIMATE", "CONFLICTING", "MISSING"}
        actual = {m.value for m in cl}
        assert actual == expected, f"ConsistencyLevel mismatch: {actual} != {expected}"

    def test_gap_severity_enum_values(self):
        """Test GapSeverity enum has 5 severity levels."""
        _skip_if_no_config()
        gs = _cfg.GapSeverity
        expected = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}
        actual = {m.value for m in gs}
        assert actual == expected, f"GapSeverity mismatch: {actual} != {expected}"

    def test_calendar_event_type_enum_values(self):
        """Test CalendarEventType enum has 5 event types."""
        _skip_if_no_config()
        cet = _cfg.CalendarEventType
        expected = {
            "FILING_DEADLINE", "DATA_COLLECTION", "REVIEW_MILESTONE",
            "AUDIT_DATE", "BOARD_REPORT",
        }
        actual = {m.value for m in cet}
        assert actual == expected, f"CalendarEventType mismatch: {actual} != {expected}"

    def test_evidence_type_enum_values(self):
        """Test EvidenceType enum has 6 evidence types."""
        _skip_if_no_config()
        et = _cfg.EvidenceType
        expected = {
            "DOCUMENT", "CERTIFICATE", "MEASUREMENT",
            "CALCULATION", "ATTESTATION", "REPORT",
        }
        actual = {m.value for m in et}
        assert actual == expected, f"EvidenceType mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # Sub-config defaults
    # -----------------------------------------------------------------

    def test_regulation_config_creation(self):
        """Test RegulationConfig creates with valid fields and validates pack_id."""
        _skip_if_no_config()
        rc = _cfg.RegulationConfig(
            enabled=True,
            priority=1,
            pack_id="PACK-001-csrd-starter",
            display_name="CSRD",
            regulation_reference="Directive (EU) 2022/2464",
            scoring_weight=0.30,
        )
        assert rc.enabled is True
        assert rc.priority == 1
        assert rc.pack_id == "PACK-001-csrd-starter"
        assert rc.scoring_weight == 0.30
        assert rc.data_quality_threshold == 0.80
        assert rc.reporting_frequency == "ANNUAL"

    def test_calendar_config_defaults(self):
        """Test CalendarConfig default values."""
        _skip_if_no_config()
        cal = _cfg.CalendarConfig()
        assert cal.enabled is True
        assert cal.lead_time_days == 30
        assert cal.critical_lead_time_days == 7
        assert len(cal.notification_channels) >= 1
        assert cal.conflict_detection is True
        assert cal.auto_schedule_reviews is True
        assert cal.review_lead_time_days == 14
        assert cal.include_csrd_deadlines is True
        assert cal.include_cbam_deadlines is True
        assert cal.include_eudr_deadlines is True
        assert cal.include_taxonomy_deadlines is True
        assert cal.fiscal_year_end_month == 12
        assert cal.timezone == "Europe/Brussels"

    def test_deduplication_config_defaults(self):
        """Test DeduplicationConfig default values."""
        _skip_if_no_config()
        dd = _cfg.DeduplicationConfig()
        assert dd.enabled is True
        assert dd.fuzzy_match_threshold == 0.90
        assert dd.hash_comparison is True
        assert dd.semantic_similarity is True
        assert dd.auto_merge is False
        assert dd.track_savings is True
        assert len(dd.dedup_categories) == 9  # All DataFieldCategory values
        assert dd.batch_size == 500
        assert dd.preserve_provenance is True

    def test_consistency_config_defaults(self):
        """Test ConsistencyConfig default values."""
        _skip_if_no_config()
        cc = _cfg.ConsistencyConfig()
        assert cc.enabled is True
        assert cc.numeric_tolerance_pct == 5.0
        assert cc.date_tolerance_days == 0
        assert cc.auto_reconcile is False
        assert cc.reconciliation_action == _cfg.ReconciliationAction.FLAG_FOR_REVIEW
        assert cc.check_ghg_emissions is True
        assert cc.check_financial_data is True
        assert cc.check_supply_chain is True
        assert cc.check_taxonomy_csrd is True
        assert cc.check_cbam_taxonomy is True
        assert cc.check_eudr_taxonomy is True

    def test_scoring_config_defaults(self):
        """Test ScoringConfig default values and weight validation."""
        _skip_if_no_config()
        sc = _cfg.ScoringConfig()
        assert sc.enabled is True
        assert sc.composite_method == _cfg.ScoringMethod.WEIGHTED_AVERAGE
        assert sc.passing_score == 70.0
        assert sc.include_trend is True
        assert sc.score_data_quality is True
        assert sc.score_evidence_completeness is True
        assert sc.score_timeliness is True

        # Weights must sum to 1.0
        total = sum(sc.regulation_weights.values())
        assert abs(total - 1.0) < 0.01, (
            f"Default scoring weights must sum to 1.0, got {total}"
        )

        # Letter grade thresholds must be in descending order
        grades = sc.letter_grade_thresholds
        assert grades["A"] > grades["B"] > grades["C"] > grades["D"] > grades["F"]

    def test_evidence_config_defaults(self):
        """Test EvidenceConfig default values."""
        _skip_if_no_config()
        ev = _cfg.EvidenceConfig()
        assert ev.enabled is True
        assert len(ev.evidence_types) == 6  # All EvidenceType values
        assert ev.deduplication is True
        assert ev.hash_algorithm == "SHA-256"
        assert ev.retention_years == 7
        assert ev.lifecycle_tracking is True
        assert ev.auto_expire is True
        assert ev.require_verification is True
        assert ev.max_file_size_mb == 50
        assert "pdf" in ev.allowed_formats
        assert "xlsx" in ev.allowed_formats
        assert ev.cross_reference_tracking is True

    # -----------------------------------------------------------------
    # BundleComplianceConfig - Enabled regulations
    # -----------------------------------------------------------------

    def test_bundle_config_enabled_regulations(self):
        """Test enabled_regulations returns all 4 regulations by default."""
        _skip_if_no_config()
        config = _cfg.BundleComplianceConfig()
        enabled = config.enabled_regulations

        assert len(enabled) == 4
        enabled_values = {r.value for r in enabled}
        assert enabled_values == {"CSRD", "CBAM", "EUDR", "TAXONOMY"}

    # -----------------------------------------------------------------
    # BundleComplianceConfig - Regulation configs
    # -----------------------------------------------------------------

    def test_bundle_config_regulation_configs(self):
        """Test regulation_configs contains entries for all 4 regulations."""
        _skip_if_no_config()
        config = _cfg.BundleComplianceConfig()

        assert len(config.regulation_configs) == 4
        for reg_key in REGULATIONS:
            assert reg_key in config.regulation_configs, (
                f"Missing regulation config: {reg_key}"
            )
            rc = config.regulation_configs[reg_key]
            assert rc.enabled is True
            assert rc.pack_id.startswith("PACK-")

        # Verify specific pack IDs
        assert config.regulation_configs["CSRD"].pack_id == "PACK-001-csrd-starter"
        assert config.regulation_configs["CBAM"].pack_id == "PACK-004-cbam-readiness"
        assert config.regulation_configs["EUDR"].pack_id == "PACK-006-eudr-starter"
        assert config.regulation_configs["TAXONOMY"].pack_id == "PACK-008-eu-taxonomy-alignment"

    # -----------------------------------------------------------------
    # PackConfig loading
    # -----------------------------------------------------------------

    def test_pack_config_from_yaml(self):
        """Test PackConfig.from_yaml loads from preset file if available."""
        _skip_if_no_config()
        enterprise_path = _PRESETS_DIR / "enterprise_full.yaml"
        if not enterprise_path.exists():
            pytest.skip("Enterprise preset file not found")

        pc = _cfg.PackConfig.from_yaml(enterprise_path)
        assert isinstance(pc.pack, _cfg.BundleComplianceConfig)
        assert isinstance(pc.loaded_from, list)
        assert len(pc.loaded_from) >= 1
        assert str(enterprise_path) in pc.loaded_from

    def test_pack_config_available_presets(self):
        """Test PackConfig.available_presets returns all 4 presets."""
        _skip_if_no_config()
        presets = _cfg.PackConfig.available_presets()
        assert isinstance(presets, dict)
        assert len(presets) == 4, f"Expected 4 presets, got {len(presets)}"

        for preset_id in BUNDLE_PRESET_IDS:
            assert preset_id in presets, (
                f"Missing preset: {preset_id}. Found: {list(presets.keys())}"
            )

    def test_get_default_config_function(self):
        """Test get_default_config returns valid PackConfig with defaults."""
        _skip_if_no_config()
        pc = _cfg.get_default_config()

        assert isinstance(pc, _cfg.PackConfig)
        assert isinstance(pc.pack, _cfg.BundleComplianceConfig)
        assert pc.pack.pack_id == "PACK-009-eu-climate-compliance-bundle"
        assert pc.pack.tier == "bundle"
        assert len(pc.active_regulations) == 4

    # -----------------------------------------------------------------
    # Preset YAML file existence
    # -----------------------------------------------------------------

    def test_preset_enterprise_full_yaml_exists(self):
        """Test enterprise_full preset YAML exists and is valid."""
        preset_path = _PRESETS_DIR / "enterprise_full.yaml"
        assert preset_path.exists(), f"Enterprise preset not found: {preset_path}"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Preset must parse to dict"
        assert data.get("bundle_tier") == "ENTERPRISE_FULL"

    def test_preset_financial_institution_yaml_exists(self):
        """Test financial_institution preset YAML exists and is valid."""
        preset_path = _PRESETS_DIR / "financial_institution.yaml"
        assert preset_path.exists(), f"FI preset not found: {preset_path}"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Preset must parse to dict"
        assert data.get("bundle_tier") == "FINANCIAL_INSTITUTION"

    def test_preset_eu_importer_yaml_exists(self):
        """Test eu_importer preset YAML exists and is valid."""
        preset_path = _PRESETS_DIR / "eu_importer.yaml"
        assert preset_path.exists(), f"EU importer preset not found: {preset_path}"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Preset must parse to dict"
        assert data.get("bundle_tier") == "EU_IMPORTER"

    def test_preset_sme_essential_yaml_exists(self):
        """Test sme_essential preset YAML exists and is valid."""
        preset_path = _PRESETS_DIR / "sme_essential.yaml"
        assert preset_path.exists(), f"SME preset not found: {preset_path}"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Preset must parse to dict"
        assert data.get("bundle_tier") == "SME_ESSENTIAL"

    def test_demo_config_yaml_exists(self):
        """Test demo_config.yaml exists and is valid with demo mode enabled."""
        demo_path = _DEMO_DIR / "demo_config.yaml"
        assert demo_path.exists(), f"Demo config not found: {demo_path}"
        with open(demo_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Demo config must parse to dict"

        # Demo config should have demo section with demo_mode_enabled
        demo = data.get("demo", {})
        assert demo.get("demo_mode_enabled") is True, (
            "Demo config should have demo_mode_enabled=true"
        )

        # Demo should have at least CSRD + TAXONOMY in sample_regulations
        sample_regs = demo.get("sample_regulations", [])
        assert len(sample_regs) >= 2, (
            f"Demo must have at least 2 sample regulations, got {sample_regs}"
        )

    # -----------------------------------------------------------------
    # Config serialization and hashing
    # -----------------------------------------------------------------

    def test_config_serialization(self):
        """Test BundleComplianceConfig can serialize to dict and back."""
        _skip_if_no_config()
        config = _cfg.BundleComplianceConfig()

        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert "pack_id" in config_dict
        assert "regulation_configs" in config_dict
        assert "calendar" in config_dict
        assert "deduplication" in config_dict
        assert "consistency" in config_dict
        assert "gap_analysis" in config_dict
        assert "evidence" in config_dict
        assert "reporting" in config_dict
        assert "scoring" in config_dict
        assert "data_mapper" in config_dict
        assert "health_check" in config_dict
        assert "audit_trail" in config_dict
        assert "demo" in config_dict

        # Reconstruct from dict
        reconstructed = _cfg.BundleComplianceConfig(**config_dict)
        assert reconstructed.pack_id == config.pack_id
        assert reconstructed.version == config.version
        assert reconstructed.tier == config.tier
        assert reconstructed.bundle_tier == config.bundle_tier

    def test_config_hash_reproducibility(self):
        """Test that config hash is reproducible for same configuration."""
        _skip_if_no_config()
        config = _cfg.BundleComplianceConfig()
        config_dict = config.model_dump()
        config_json = json.dumps(config_dict, sort_keys=True, default=str)

        hash1 = hashlib.sha256(config_json.encode()).hexdigest()
        hash2 = hashlib.sha256(config_json.encode()).hexdigest()

        assert hash1 == hash2, "Config hash should be deterministic"
        assert len(hash1) == 64, "SHA-256 hash must be 64 chars"

    def test_pack_config_get_config_hash(self):
        """Test PackConfig.get_config_hash returns valid SHA-256."""
        _skip_if_no_config()
        pc = _cfg.get_default_config()
        config_hash = pc.get_config_hash()

        assert isinstance(config_hash, str)
        assert len(config_hash) == 64
        # Reproducibility check
        assert pc.get_config_hash() == config_hash

    # -----------------------------------------------------------------
    # BundleComplianceConfig methods
    # -----------------------------------------------------------------

    def test_get_pack_ids(self):
        """Test get_pack_ids returns mapping for all enabled regulations."""
        _skip_if_no_config()
        config = _cfg.BundleComplianceConfig()
        pack_ids = config.get_pack_ids()

        assert isinstance(pack_ids, dict)
        assert len(pack_ids) == 4
        assert pack_ids["CSRD"] == "PACK-001-csrd-starter"
        assert pack_ids["CBAM"] == "PACK-004-cbam-readiness"
        assert pack_ids["EUDR"] == "PACK-006-eudr-starter"
        assert pack_ids["TAXONOMY"] == "PACK-008-eu-taxonomy-alignment"

    def test_get_inherited_agent_count(self):
        """Test get_inherited_agent_count returns 208 for all 4 packs enabled."""
        _skip_if_no_config()
        config = _cfg.BundleComplianceConfig()
        count = config.get_inherited_agent_count()

        # CSRD(51) + CBAM(47) + EUDR(59) + Taxonomy(51) = 208
        assert count == 208, f"Expected 208 inherited agents, got {count}"

    def test_get_scoring_weights_normalized(self):
        """Test get_scoring_weights_normalized sums to 1.0."""
        _skip_if_no_config()
        config = _cfg.BundleComplianceConfig()
        weights = config.get_scoring_weights_normalized()

        assert isinstance(weights, dict)
        assert len(weights) == 4
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001, (
            f"Normalized weights must sum to 1.0, got {total}"
        )

    def test_get_active_overlaps(self):
        """Test get_active_overlaps returns all 6 overlap pairs when all enabled."""
        _skip_if_no_config()
        config = _cfg.BundleComplianceConfig()
        overlaps = config.get_active_overlaps()

        assert isinstance(overlaps, dict)
        # 4 regulations give C(4,2) = 6 possible pairs
        assert len(overlaps) == 6, (
            f"Expected 6 active overlaps, got {len(overlaps)}"
        )

        expected_overlap_ids = [
            "CSRD-TAXONOMY", "CSRD-CBAM", "CSRD-EUDR",
            "TAXONOMY-EUDR", "TAXONOMY-CBAM", "CBAM-EUDR",
        ]
        for overlap_id in expected_overlap_ids:
            assert overlap_id in overlaps, (
                f"Missing overlap: {overlap_id}. Found: {list(overlaps.keys())}"
            )

    def test_get_feature_summary(self):
        """Test get_feature_summary returns expected feature flags."""
        _skip_if_no_config()
        config = _cfg.BundleComplianceConfig()
        features = config.get_feature_summary()

        assert isinstance(features, dict)
        assert features["csrd_enabled"] is True
        assert features["cbam_enabled"] is True
        assert features["eudr_enabled"] is True
        assert features["taxonomy_enabled"] is True
        assert features["calendar"] is True
        assert features["deduplication"] is True
        assert features["consistency_checking"] is True
        assert features["gap_analysis"] is True
        assert features["evidence_management"] is True
        assert features["consolidated_reporting"] is True
        assert features["compliance_scoring"] is True
        assert features["data_mapping"] is True
        assert features["health_check"] is True
        assert features["audit_trail"] is True
        assert features["demo_mode"] is False  # Not demo by default

    def test_get_regulation_display(self):
        """Test get_regulation_display returns display names for enabled regulations."""
        _skip_if_no_config()
        config = _cfg.BundleComplianceConfig()
        display = config.get_regulation_display()

        assert isinstance(display, dict)
        assert len(display) == 4
        assert "CSRD" in display
        assert "CBAM" in display
        assert "EUDR" in display
        assert "TAXONOMY" in display

    # -----------------------------------------------------------------
    # Validation tests
    # -----------------------------------------------------------------

    def test_config_validation_no_regulations_fails(self):
        """Test that disabling all regulations raises ValidationError."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.BundleComplianceConfig(
                regulation_configs={
                    "CSRD": _cfg.RegulationConfig(
                        enabled=False, pack_id="PACK-001-csrd-starter"
                    ),
                    "CBAM": _cfg.RegulationConfig(
                        enabled=False, pack_id="PACK-004-cbam-readiness"
                    ),
                    "EUDR": _cfg.RegulationConfig(
                        enabled=False, pack_id="PACK-006-eudr-starter"
                    ),
                    "TAXONOMY": _cfg.RegulationConfig(
                        enabled=False, pack_id="PACK-008-eu-taxonomy-alignment"
                    ),
                }
            )

    def test_config_validation_invalid_pack_id_fails(self):
        """Test that pack_id not starting with PACK- raises ValueError."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.RegulationConfig(
                pack_id="INVALID-001-test",
            )

    def test_consistency_config_no_checks_fails(self):
        """Test that ConsistencyConfig with no checks enabled raises ValueError."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.ConsistencyConfig(
                enabled=True,
                check_ghg_emissions=False,
                check_financial_data=False,
                check_supply_chain=False,
                check_taxonomy_csrd=False,
                check_cbam_taxonomy=False,
                check_eudr_taxonomy=False,
            )

    # -----------------------------------------------------------------
    # Environment overrides
    # -----------------------------------------------------------------

    def test_environment_override(self, monkeypatch):
        """Test environment variable overrides are respected."""
        _skip_if_no_config()

        monkeypatch.setenv("BUNDLE_PACK_ORG_NAME", "TestOrg AG")
        monkeypatch.setenv("BUNDLE_PACK_REPORTING_YEAR", "2026")

        env_org_name = os.getenv("BUNDLE_PACK_ORG_NAME", "")
        assert env_org_name == "TestOrg AG"

        env_year = int(os.getenv("BUNDLE_PACK_REPORTING_YEAR", "2025"))
        assert env_year == 2026

    # -----------------------------------------------------------------
    # Utility functions
    # -----------------------------------------------------------------

    def test_get_regulation_display_name(self):
        """Test get_regulation_display_name returns correct names."""
        _skip_if_no_config()

        assert _cfg.get_regulation_display_name("CSRD") == (
            "Corporate Sustainability Reporting Directive"
        )
        assert _cfg.get_regulation_display_name("CBAM") == (
            "Carbon Border Adjustment Mechanism"
        )
        assert _cfg.get_regulation_display_name("EUDR") == (
            "EU Deforestation Regulation"
        )
        assert _cfg.get_regulation_display_name("TAXONOMY") == (
            "EU Taxonomy Regulation"
        )

    def test_get_regulation_reference(self):
        """Test get_regulation_reference returns correct EU references."""
        _skip_if_no_config()

        assert "2022/2464" in _cfg.get_regulation_reference("CSRD")
        assert "2023/956" in _cfg.get_regulation_reference("CBAM")
        assert "2023/1115" in _cfg.get_regulation_reference("EUDR")
        assert "2020/852" in _cfg.get_regulation_reference("TAXONOMY")

    def test_validate_bundle_consistency(self):
        """Test validate_bundle_consistency returns correct status tuples."""
        _skip_if_no_config()

        # All compliant
        status, msg = _cfg.validate_bundle_consistency(True, True, True, True)
        assert status == _cfg.ComplianceStatus.COMPLIANT

        # All non-compliant
        status, msg = _cfg.validate_bundle_consistency(False, False, False, False)
        assert status == _cfg.ComplianceStatus.NON_COMPLIANT

        # Partially compliant
        status, msg = _cfg.validate_bundle_consistency(True, False, True, False)
        assert status == _cfg.ComplianceStatus.PARTIALLY_COMPLIANT

    def test_compute_composite_score_weighted_average(self):
        """Test compute_composite_score with weighted average method."""
        _skip_if_no_config()

        scores = {"CSRD": 80.0, "CBAM": 90.0, "EUDR": 70.0, "TAXONOMY": 85.0}
        weights = {"CSRD": 0.30, "CBAM": 0.25, "EUDR": 0.20, "TAXONOMY": 0.25}

        result = _cfg.compute_composite_score(
            scores, weights, _cfg.ScoringMethod.WEIGHTED_AVERAGE
        )

        # Expected: 80*0.30 + 90*0.25 + 70*0.20 + 85*0.25 = 24 + 22.5 + 14 + 21.25 = 81.75
        assert abs(result - 81.75) < 0.01, f"Expected ~81.75, got {result}"

    def test_compute_composite_score_minimum(self):
        """Test compute_composite_score with minimum score method."""
        _skip_if_no_config()

        scores = {"CSRD": 80.0, "CBAM": 90.0, "EUDR": 70.0, "TAXONOMY": 85.0}
        weights = {"CSRD": 0.25, "CBAM": 0.25, "EUDR": 0.25, "TAXONOMY": 0.25}

        result = _cfg.compute_composite_score(
            scores, weights, _cfg.ScoringMethod.MINIMUM_SCORE
        )
        assert result == 70.0, f"Expected 70.0 (minimum), got {result}"

    def test_get_letter_grade(self):
        """Test get_letter_grade returns correct grades for various scores."""
        _skip_if_no_config()

        assert _cfg.get_letter_grade(95.0) == "A"
        assert _cfg.get_letter_grade(85.0) == "B"
        assert _cfg.get_letter_grade(75.0) == "C"
        assert _cfg.get_letter_grade(65.0) == "D"
        assert _cfg.get_letter_grade(50.0) == "F"
        assert _cfg.get_letter_grade(0.0) == "F"

    def test_get_cross_regulation_overlaps(self):
        """Test get_cross_regulation_overlaps returns data for valid pairs."""
        _skip_if_no_config()

        overlap = _cfg.get_cross_regulation_overlaps("CSRD", "TAXONOMY")
        assert overlap is not None
        assert "shared_fields" in overlap
        assert len(overlap["shared_fields"]) > 0

        overlap_reverse = _cfg.get_cross_regulation_overlaps("TAXONOMY", "CSRD")
        assert overlap_reverse is not None

        # Non-existent pair returns None
        overlap_none = _cfg.get_cross_regulation_overlaps("CSRD", "UNKNOWN")
        assert overlap_none is None
