# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Configuration Tests
==================================================================

Tests for pack_config.py covering all enums, sub-config models,
the DMAConfig root model, PackConfig wrapper, preset loading,
environment variable overrides, and config hashing.

Target: 40+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
Date:    March 2026
"""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from .conftest import _load_config_module, PRESETS_DIR


# ---------------------------------------------------------------------------
# Module-level config module load (session-scoped via conftest fixture)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    """Load the pack_config module once per module."""
    return _load_config_module()


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestSectorTypeEnum:
    """Tests for SectorType enum values."""

    def test_sector_type_enum_values(self, cfg):
        """Verify SectorType has all 19 expected members."""
        expected = {
            "AGRICULTURE", "MINING", "MANUFACTURING", "ENERGY", "WATER",
            "CONSTRUCTION", "RETAIL", "TRANSPORT", "HOSPITALITY", "ICT",
            "FINANCIAL_SERVICES", "REAL_ESTATE", "PROFESSIONAL_SERVICES",
            "ADMINISTRATIVE", "PUBLIC_ADMINISTRATION", "EDUCATION",
            "HEALTHCARE", "OTHER_SERVICES", "GENERAL",
        }
        actual = {m.name for m in cfg.SectorType}
        assert actual == expected

    def test_sector_type_string_values_match_names(self, cfg):
        """Each SectorType value should equal its name."""
        for member in cfg.SectorType:
            assert member.value == member.name


class TestCompanySizeEnum:
    """Tests for CompanySize enum."""

    def test_company_size_enum_values(self, cfg):
        """Verify CompanySize has 6 members."""
        expected = {
            "LARGE_ENTERPRISE", "LARGE_NON_LISTED", "MID_MARKET",
            "SME_LISTED", "SME", "MICRO",
        }
        actual = {m.name for m in cfg.CompanySize}
        assert actual == expected

    def test_company_size_is_str_enum(self, cfg):
        """CompanySize members should be string instances."""
        for member in cfg.CompanySize:
            assert isinstance(member.value, str)


class TestStakeholderCategoryEnum:
    """Tests for StakeholderCategory enum."""

    def test_stakeholder_category_enum(self, cfg):
        """Verify all 16 stakeholder categories exist."""
        expected_count = 16
        assert len(cfg.StakeholderCategory) == expected_count

    def test_stakeholder_category_includes_affected(self, cfg):
        """Affected stakeholder categories are present."""
        names = {m.name for m in cfg.StakeholderCategory}
        for cat in ["EMPLOYEES", "VALUE_CHAIN_WORKERS", "LOCAL_COMMUNITIES",
                     "CONSUMERS", "INDIGENOUS_PEOPLES", "ECOSYSTEMS"]:
            assert cat in names

    def test_stakeholder_category_includes_users(self, cfg):
        """Users of sustainability statements are present."""
        names = {m.name for m in cfg.StakeholderCategory}
        for cat in ["INVESTORS", "LENDERS", "CREDITORS", "ASSET_MANAGERS",
                     "INSURERS", "RATING_AGENCIES", "REGULATORS", "NGOS",
                     "TRADE_UNIONS", "CUSTOMERS_B2B"]:
            assert cat in names


class TestMaterialityLevelEnum:
    """Tests for MaterialityLevel enum."""

    def test_materiality_level_enum(self, cfg):
        """Verify 4 materiality levels."""
        expected = {"MATERIAL", "NOT_MATERIAL", "BORDERLINE", "NOT_ASSESSED"}
        actual = {m.name for m in cfg.MaterialityLevel}
        assert actual == expected


class TestTimeHorizonEnum:
    """Tests for TimeHorizon enum."""

    def test_time_horizon_enum(self, cfg):
        """Verify 4 time horizons."""
        expected = {"SHORT_TERM", "MEDIUM_TERM", "LONG_TERM", "VERY_LONG_TERM"}
        actual = {m.name for m in cfg.TimeHorizon}
        assert actual == expected


class TestESRSTopicEnum:
    """Tests for ESRSTopic enum."""

    def test_esrs_topic_enum(self, cfg):
        """Verify all 10 ESRS topics."""
        expected = {"E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "G1"}
        actual = {m.value for m in cfg.ESRSTopic}
        assert actual == expected

    def test_esrs_topic_count(self, cfg):
        """Exactly 10 ESRS topics."""
        assert len(cfg.ESRSTopic) == 10


class TestScoringMethodologyEnum:
    """Tests for ScoringMethodology enum."""

    def test_scoring_methodology_enum(self, cfg):
        """Verify 5 scoring methodologies."""
        expected = {
            "ABSOLUTE_CUTOFF", "PERCENTILE", "SECTOR_CALIBRATED",
            "EXPERT_JUDGMENT", "COMBINED",
        }
        actual = {m.name for m in cfg.ScoringMethodology}
        assert actual == expected


class TestIROTypeEnum:
    """Tests for IROType enum."""

    def test_iro_type_enum(self, cfg):
        """Verify 3 IRO types: IMPACT, RISK, OPPORTUNITY."""
        expected = {"IMPACT", "RISK", "OPPORTUNITY"}
        actual = {m.name for m in cfg.IROType}
        assert actual == expected


# ===========================================================================
# Sub-Config Model Tests
# ===========================================================================


class TestImpactMaterialityConfigDefaults:
    """Tests for ImpactMaterialityConfig defaults."""

    def test_impact_materiality_config_defaults(self, cfg):
        """Default config has expected values."""
        imc = cfg.ImpactMaterialityConfig()
        assert imc.enabled is True
        assert imc.scoring_scale == 5
        assert imc.severity_dimensions == ["scale", "scope", "irremediable_character"]
        assert imc.severity_weighting == "EQUAL"
        assert imc.likelihood_enabled is True
        assert imc.multi_scorer is False

    def test_impact_materiality_scoring_scale_bounds(self, cfg):
        """Scoring scale must be between 3 and 10."""
        with pytest.raises(Exception):
            cfg.ImpactMaterialityConfig(scoring_scale=2)
        with pytest.raises(Exception):
            cfg.ImpactMaterialityConfig(scoring_scale=11)

    def test_impact_materiality_valid_severity_dimensions(self, cfg):
        """Only valid severity dimensions accepted."""
        with pytest.raises(Exception):
            cfg.ImpactMaterialityConfig(severity_dimensions=["invalid_dimension"])


class TestFinancialMaterialityConfigDefaults:
    """Tests for FinancialMaterialityConfig defaults."""

    def test_financial_materiality_config_defaults(self, cfg):
        """Default config has expected values."""
        fmc = cfg.FinancialMaterialityConfig()
        assert fmc.enabled is True
        assert fmc.scoring_scale == 5
        assert fmc.magnitude_type == "qualitative"
        assert fmc.magnitude_currency == "EUR"
        assert fmc.likelihood_enabled is True
        assert fmc.scenario_analysis is False
        assert fmc.multi_scorer is False

    def test_financial_materiality_time_horizons(self, cfg):
        """Default time horizons include SHORT, MEDIUM, LONG."""
        fmc = cfg.FinancialMaterialityConfig()
        horizon_values = [h.value for h in fmc.time_horizons]
        assert "SHORT_TERM" in horizon_values
        assert "MEDIUM_TERM" in horizon_values
        assert "LONG_TERM" in horizon_values


class TestStakeholderConfigDefaults:
    """Tests for StakeholderConfig defaults."""

    def test_stakeholder_config_defaults(self, cfg):
        """Default config has expected stakeholder categories."""
        sc = cfg.StakeholderConfig()
        assert sc.enabled is True
        assert len(sc.affected_stakeholders) >= 2
        assert len(sc.statement_users) >= 1
        assert sc.min_response_rate == pytest.approx(0.20)

    def test_stakeholder_dedup(self, cfg):
        """Duplicate affected stakeholders are removed."""
        sc = cfg.StakeholderConfig(
            affected_stakeholders=[
                cfg.StakeholderCategory.EMPLOYEES,
                cfg.StakeholderCategory.EMPLOYEES,
                cfg.StakeholderCategory.CONSUMERS,
            ]
        )
        assert len(sc.affected_stakeholders) == 2


class TestIROConfigDefaults:
    """Tests for IROConfig defaults."""

    def test_iro_config_defaults(self, cfg):
        """Default IRO config includes all ESRS topics."""
        iro = cfg.IROConfig()
        assert iro.enabled is True
        topic_values = {t.value for t in iro.esrs_topics_in_scope}
        assert len(topic_values) == 10
        assert iro.max_iros_per_topic == 20
        assert iro.value_chain_depth == 3


class TestMatrixConfigDefaults:
    """Tests for MatrixConfig defaults."""

    def test_matrix_config_defaults(self, cfg):
        """Default matrix config has correct axes."""
        mc = cfg.MatrixConfig()
        assert mc.enabled is True
        assert mc.x_axis == "financial_materiality"
        assert mc.y_axis == "impact_materiality"
        assert mc.color_by_pillar is True
        assert mc.quadrant_labels is True


class TestThresholdConfigDefaults:
    """Tests for ThresholdConfig defaults."""

    def test_threshold_config_defaults(self, cfg):
        """Default threshold config."""
        tc = cfg.ThresholdConfig()
        assert tc.enabled is True
        assert tc.methodology == cfg.ScoringMethodology.ABSOLUTE_CUTOFF
        assert tc.impact_threshold == pytest.approx(3.0)
        assert tc.financial_threshold == pytest.approx(3.0)
        assert tc.sensitivity_analysis is True

    def test_invalid_threshold_rejected(self, cfg):
        """Threshold out of range is rejected."""
        with pytest.raises(Exception):
            cfg.ThresholdConfig(impact_threshold=0.5)
        with pytest.raises(Exception):
            cfg.ThresholdConfig(impact_threshold=11.0)


class TestReportingConfigDefaults:
    """Tests for ReportingConfig defaults."""

    def test_reporting_config_defaults(self, cfg):
        """Default reporting config enables key disclosures."""
        rc = cfg.ReportingConfig()
        assert rc.enabled is True
        assert rc.iro_1_disclosure is True
        assert rc.iro_2_disclosure is True
        assert rc.sbm_3_disclosure is True
        assert rc.executive_summary is True
        assert rc.full_dma_report is True
        assert rc.audit_trail_report is True


# ===========================================================================
# DMAConfig Root Model Tests
# ===========================================================================


class TestDMAConfig:
    """Tests for the DMAConfig root configuration model."""

    def test_default_config_creation(self, cfg):
        """Default DMAConfig creates without errors."""
        config = cfg.DMAConfig()
        assert config.company_size == cfg.CompanySize.LARGE_ENTERPRISE
        assert config.sectors == [cfg.SectorType.GENERAL]
        assert config.impact_materiality.enabled is True
        assert config.financial_materiality.enabled is True
        assert config.stakeholder_engagement.enabled is True

    def test_config_validation(self, cfg):
        """validate_config returns warnings for issues."""
        config = cfg.DMAConfig(sectors=[])
        warnings = cfg.validate_config(config)
        assert len(warnings) > 0
        assert any("sector" in w.lower() for w in warnings)

    def test_config_with_manufacturing_sector(self, cfg):
        """Config with MANUFACTURING sector is valid."""
        config = cfg.DMAConfig(sectors=[cfg.SectorType.MANUFACTURING])
        assert cfg.SectorType.MANUFACTURING in config.sectors

    def test_config_merge_order(self, cfg):
        """Overrides take precedence in deep merge."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 5}
        result = cfg.PackConfig._deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3
        assert result["e"] == 5


# ===========================================================================
# PackConfig Wrapper Tests
# ===========================================================================


class TestPackConfig:
    """Tests for the PackConfig wrapper and preset loading."""

    def test_pack_config_wrapper(self, cfg):
        """PackConfig wraps DMAConfig correctly."""
        pc = cfg.PackConfig()
        assert pc.pack_id == "PACK-015-double-materiality"
        assert pc.config_version == "1.0.0"
        assert isinstance(pc.pack, cfg.DMAConfig)

    def test_config_from_preset_large_enterprise(self, cfg):
        """Load large_enterprise preset."""
        pc = cfg.PackConfig.from_preset("large_enterprise")
        assert pc.preset_name == "large_enterprise"
        assert pc.pack.company_size == cfg.CompanySize.LARGE_ENTERPRISE
        assert pc.pack.impact_materiality.scoring_scale == 10
        assert pc.pack.impact_materiality.multi_scorer is True

    def test_config_from_preset_mid_market(self, cfg):
        """Load mid_market preset."""
        pc = cfg.PackConfig.from_preset("mid_market")
        assert pc.preset_name == "mid_market"
        assert pc.pack.company_size == cfg.CompanySize.MID_MARKET

    def test_config_from_preset_sme(self, cfg):
        """Load sme preset."""
        pc = cfg.PackConfig.from_preset("sme")
        assert pc.preset_name == "sme"
        assert pc.pack.impact_materiality.scoring_scale == 5
        assert pc.pack.impact_materiality.multi_scorer is False

    def test_config_from_preset_financial_services(self, cfg):
        """Load financial_services preset."""
        pc = cfg.PackConfig.from_preset("financial_services")
        assert pc.preset_name == "financial_services"
        assert cfg.SectorType.FINANCIAL_SERVICES in pc.pack.sectors
        assert pc.pack.financial_materiality.scenario_analysis is True

    def test_config_from_preset_manufacturing(self, cfg):
        """Load manufacturing preset."""
        pc = cfg.PackConfig.from_preset("manufacturing")
        assert pc.preset_name == "manufacturing"

    def test_config_from_preset_multi_sector(self, cfg):
        """Load multi_sector preset."""
        pc = cfg.PackConfig.from_preset("multi_sector")
        assert pc.preset_name == "multi_sector"

    def test_config_from_preset_invalid(self, cfg):
        """Unknown preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            cfg.PackConfig.from_preset("nonexistent_preset")

    def test_config_serialization(self, cfg):
        """PackConfig can be serialized to JSON and back."""
        pc = cfg.PackConfig.from_preset("large_enterprise")
        json_str = pc.model_dump_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["pack_id"] == "PACK-015-double-materiality"

    def test_config_hash(self, cfg):
        """Config hash is a 64-char SHA-256 hex string."""
        pc = cfg.PackConfig.from_preset("large_enterprise")
        h = pc.get_config_hash()
        assert isinstance(h, str)
        assert len(h) == 64

    def test_config_hash_deterministic(self, cfg):
        """Same config produces same hash."""
        pc1 = cfg.PackConfig.from_preset("large_enterprise")
        pc2 = cfg.PackConfig.from_preset("large_enterprise")
        assert pc1.get_config_hash() == pc2.get_config_hash()

    def test_config_hash_differs_across_presets(self, cfg):
        """Different presets produce different hashes."""
        h1 = cfg.PackConfig.from_preset("large_enterprise").get_config_hash()
        h2 = cfg.PackConfig.from_preset("sme").get_config_hash()
        assert h1 != h2


# ===========================================================================
# Environment Variable Override Tests
# ===========================================================================


class TestEnvironmentVariableOverride:
    """Tests for DMA_PACK_* environment variable overrides."""

    def test_environment_variable_override(self, cfg):
        """DMA_PACK_ env vars are loaded as overrides."""
        env_vars = {
            "DMA_PACK_IMPACT_MATERIALITY__SCORING_SCALE": "7",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            overrides = cfg.PackConfig._load_env_overrides()
        assert overrides.get("impact_materiality", {}).get("scoring_scale") == 7

    def test_env_override_boolean_true(self, cfg):
        """Boolean true values parsed correctly."""
        env_vars = {"DMA_PACK_IMPACT_MATERIALITY__MULTI_SCORER": "true"}
        with patch.dict(os.environ, env_vars, clear=False):
            overrides = cfg.PackConfig._load_env_overrides()
        assert overrides["impact_materiality"]["multi_scorer"] is True

    def test_env_override_boolean_false(self, cfg):
        """Boolean false values parsed correctly."""
        env_vars = {"DMA_PACK_STAKEHOLDER_ENGAGEMENT__ENABLED": "false"}
        with patch.dict(os.environ, env_vars, clear=False):
            overrides = cfg.PackConfig._load_env_overrides()
        assert overrides["stakeholder_engagement"]["enabled"] is False

    def test_env_override_string_value(self, cfg):
        """String values passed through correctly."""
        env_vars = {"DMA_PACK_COMPANY_NAME": "TestCorp"}
        with patch.dict(os.environ, env_vars, clear=False):
            overrides = cfg.PackConfig._load_env_overrides()
        assert overrides["company_name"] == "TestCorp"


# ===========================================================================
# Utility Function Tests
# ===========================================================================


class TestUtilityFunctions:
    """Tests for utility functions in pack_config."""

    def test_get_default_config(self, cfg):
        """get_default_config returns a DMAConfig."""
        config = cfg.get_default_config()
        assert isinstance(config, cfg.DMAConfig)
        assert config.sectors == [cfg.SectorType.GENERAL]

    def test_get_default_config_with_sector(self, cfg):
        """get_default_config accepts sector parameter."""
        config = cfg.get_default_config(sector=cfg.SectorType.MANUFACTURING)
        assert cfg.SectorType.MANUFACTURING in config.sectors

    def test_get_sector_info(self, cfg):
        """get_sector_info returns materiality profile."""
        info = cfg.get_sector_info(cfg.SectorType.MANUFACTURING)
        assert "materiality_profile" in info
        profile = info["materiality_profile"]
        assert "E1" in profile
        assert profile["E1"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")

    def test_get_esrs_topic_info(self, cfg):
        """get_esrs_topic_info returns topic metadata."""
        info = cfg.get_esrs_topic_info("E1")
        assert info["name"] == "Climate Change"
        assert info["pillar"] == "Environmental"
        assert info["standard"] == "ESRS E1"

    def test_get_subtopics_for_topic(self, cfg):
        """get_subtopics_for_topic returns subtopic list."""
        subtopics = cfg.get_subtopics_for_topic("E1")
        assert len(subtopics) == 3
        assert "E1_CLIMATE_CHANGE_ADAPTATION" in subtopics

    def test_list_available_presets(self, cfg):
        """list_available_presets returns all 6 presets."""
        presets = cfg.list_available_presets()
        assert len(presets) == 6
        for name in ["large_enterprise", "mid_market", "sme",
                      "financial_services", "manufacturing", "multi_sector"]:
            assert name in presets
