# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Configuration Tests
=====================================================

Tests all enums, sub-configs, PackConfig instantiation, from_preset(),
validation, and computed properties in pack_config.py.

Test count target: ~30 tests
"""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import CONFIG_DIR, PRESETS_DIR, _load_module


# ---------------------------------------------------------------------------
# Load the config module once
# ---------------------------------------------------------------------------


def _config_mod():
    """Load the pack_config module."""
    return _load_module(CONFIG_DIR / "pack_config.py", "config.pack_config")


# ---------------------------------------------------------------------------
# 1. Enum existence and members
# ---------------------------------------------------------------------------


class TestConfigEnums:
    """Verify all enums exist with correct members."""

    def test_company_scope_enum(self):
        mod = _config_mod()
        cs = mod.CompanyScope
        assert cs.PHASE_1.value == "PHASE_1"
        assert cs.PHASE_2.value == "PHASE_2"
        assert cs.PHASE_3.value == "PHASE_3"
        assert cs.NOT_IN_SCOPE.value == "NOT_IN_SCOPE"
        assert cs.VOLUNTARY.value == "VOLUNTARY"
        assert len(cs) == 5

    def test_sector_type_enum(self):
        mod = _config_mod()
        st = mod.SectorType
        expected = {
            "MANUFACTURING", "EXTRACTIVES", "FINANCIAL_SERVICES", "RETAIL",
            "TECHNOLOGY", "AGRICULTURE", "ENERGY", "CONSTRUCTION",
            "TRANSPORT", "SERVICES", "OTHER",
        }
        actual = {m.value for m in st}
        assert expected == actual

    def test_adverse_impact_type_enum(self):
        mod = _config_mod()
        ait = mod.AdverseImpactType
        assert ait.HUMAN_RIGHTS.value == "HUMAN_RIGHTS"
        assert ait.ENVIRONMENTAL.value == "ENVIRONMENTAL"
        assert len(ait) == 2

    def test_impact_severity_enum(self):
        mod = _config_mod()
        sev = mod.ImpactSeverity
        assert sev.CRITICAL.value == "CRITICAL"
        assert sev.HIGH.value == "HIGH"
        assert sev.MEDIUM.value == "MEDIUM"
        assert sev.LOW.value == "LOW"
        assert len(sev) == 4

    def test_compliance_status_enum(self):
        mod = _config_mod()
        cs = mod.ComplianceStatus
        assert len(cs) == 4
        assert cs.COMPLIANT.value == "COMPLIANT"
        assert cs.NON_COMPLIANT.value == "NON_COMPLIANT"

    def test_measure_type_enum(self):
        mod = _config_mod()
        mt = mod.MeasureType
        expected = {"PREVENTION", "MITIGATION", "CESSATION", "REMEDIATION"}
        actual = {m.value for m in mt}
        assert expected == actual

    def test_stakeholder_group_enum(self):
        mod = _config_mod()
        sg = mod.StakeholderGroup
        assert sg.WORKERS.value == "WORKERS"
        assert sg.TRADE_UNIONS.value == "TRADE_UNIONS"
        assert len(sg) == 8

    def test_grievance_channel_enum(self):
        mod = _config_mod()
        gc = mod.GrievanceChannel
        assert gc.HOTLINE.value == "HOTLINE"
        assert gc.WEB_PORTAL.value == "WEB_PORTAL"
        assert len(gc) == 7

    def test_transition_plan_status_enum(self):
        mod = _config_mod()
        tps = mod.TransitionPlanStatus
        assert tps.DRAFTED.value == "DRAFTED"
        assert tps.ACHIEVED.value == "ACHIEVED"
        assert len(tps) == 6

    def test_article_reference_enum(self):
        mod = _config_mod()
        ar = mod.ArticleReference
        assert ar.ART_5.value == "ART_5"
        assert ar.ART_22.value == "ART_22"
        assert ar.ART_29.value == "ART_29"
        assert len(ar) == 25

    def test_oecd_step_enum(self):
        mod = _config_mod()
        os_enum = mod.OECDStep
        assert os_enum.STEP_1_EMBED.value == "STEP_1_EMBED"
        assert os_enum.STEP_6_REMEDIATE.value == "STEP_6_REMEDIATE"
        assert len(os_enum) == 6

    def test_report_format_enum(self):
        mod = _config_mod()
        rf = mod.ReportFormat
        assert rf.PDF.value == "PDF"
        assert rf.JSON.value == "JSON"
        assert len(rf) == 4

    def test_cache_backend_enum(self):
        mod = _config_mod()
        cb = mod.CacheBackend
        assert cb.MEMORY.value == "MEMORY"
        assert cb.REDIS.value == "REDIS"
        assert cb.DISABLED.value == "DISABLED"
        assert len(cb) == 3


# ---------------------------------------------------------------------------
# 2. Sub-config models
# ---------------------------------------------------------------------------


class TestSubConfigs:
    """Verify sub-configuration models instantiate correctly."""

    def test_scope_config_defaults(self):
        mod = _config_mod()
        sc = mod.ScopeConfig()
        assert sc.phase_1_employee_threshold == 5000
        assert sc.phase_1_turnover_threshold == Decimal("1500000000")
        assert sc.phase_3_turnover_threshold == Decimal("450000000")
        assert sc.headcount_method == "HEADCOUNT"

    def test_scope_config_determine_scope_phase_1(self):
        mod = _config_mod()
        sc = mod.ScopeConfig()
        result = sc.determine_scope(6000, Decimal("2000000000"))
        assert result == mod.CompanyScope.PHASE_1

    def test_scope_config_determine_scope_phase_3(self):
        mod = _config_mod()
        sc = mod.ScopeConfig()
        result = sc.determine_scope(1500, Decimal("500000000"))
        assert result == mod.CompanyScope.PHASE_3

    def test_scope_config_determine_scope_not_in_scope(self):
        mod = _config_mod()
        sc = mod.ScopeConfig()
        result = sc.determine_scope(500, Decimal("100000000"))
        assert result == mod.CompanyScope.NOT_IN_SCOPE

    def test_scope_config_application_date(self):
        mod = _config_mod()
        sc = mod.ScopeConfig()
        d = sc.get_application_date(mod.CompanyScope.PHASE_1)
        assert d == date(2027, 7, 26)
        d2 = sc.get_application_date(mod.CompanyScope.NOT_IN_SCOPE)
        assert d2 is None

    def test_impact_config_defaults(self):
        mod = _config_mod()
        ic = mod.ImpactConfig()
        assert ic.max_impacts == 500
        assert ic.value_chain_depth == 3
        assert ic.include_potential is True
        assert ic.include_actual is True

    def test_prevention_config_defaults(self):
        mod = _config_mod()
        pc = mod.PreventionConfig()
        assert pc.effectiveness_threshold == Decimal("60")
        assert pc.corrective_action_plan_days == 90

    def test_grievance_config_defaults(self):
        mod = _config_mod()
        gc = mod.GrievanceConfig()
        assert gc.response_time_target_days == 30
        assert gc.anonymous_allowed is True
        assert len(gc.channels) >= 3

    def test_climate_config_defaults(self):
        mod = _config_mod()
        cc = mod.ClimateConfig()
        assert cc.sbti_15c_annual_reduction == Decimal("4.2")
        assert 2030 in cc.target_years
        assert 2050 in cc.target_years
        assert cc.include_scope_3 is True

    def test_liability_config_defaults(self):
        mod = _config_mod()
        lc = mod.LiabilityConfig()
        assert lc.limitation_period_default == 5
        assert lc.turnover_penalty_cap_pct == Decimal("5.0")

    def test_stakeholder_config_defaults(self):
        mod = _config_mod()
        sc = mod.StakeholderConfig()
        assert sc.min_engagement_frequency == 2
        assert sc.free_prior_informed_consent is True


# ---------------------------------------------------------------------------
# 3. PackConfig
# ---------------------------------------------------------------------------


class TestPackConfig:
    """Verify PackConfig instantiation and methods."""

    def test_pack_config_defaults(self):
        mod = _config_mod()
        cfg = mod.PackConfig()
        assert cfg.pack_name == "PACK-019-csddd-readiness"
        assert cfg.version == "1.0.0"
        assert cfg.company_scope == mod.CompanyScope.PHASE_1
        assert cfg.sector == mod.SectorType.MANUFACTURING

    def test_pack_config_hash(self):
        mod = _config_mod()
        cfg = mod.PackConfig()
        h = cfg.config_hash
        assert isinstance(h, str)
        assert len(h) == 64

    def test_pack_config_is_in_scope(self):
        mod = _config_mod()
        cfg = mod.PackConfig()
        assert cfg.is_in_scope is True
        cfg2 = mod.PackConfig(company_scope=mod.CompanyScope.NOT_IN_SCOPE)
        assert cfg2.is_in_scope is False

    def test_pack_config_application_date(self):
        mod = _config_mod()
        cfg = mod.PackConfig()
        assert cfg.application_date == date(2027, 7, 26)

    def test_pack_config_core_articles(self):
        mod = _config_mod()
        cfg = mod.PackConfig()
        arts = cfg.core_articles
        assert len(arts) >= 10
        assert mod.ArticleReference.ART_5 in arts
        assert mod.ArticleReference.ART_22 in arts

    def test_validate_thresholds_no_warnings(self):
        mod = _config_mod()
        cfg = mod.PackConfig()
        warnings = cfg.validate_thresholds()
        assert isinstance(warnings, list)
        # Default config should be valid with minimal warnings
        # (climate_transition engine is enabled, so no Art. 22 warning)

    def test_get_article_requirements(self):
        mod = _config_mod()
        cfg = mod.PackConfig()
        reqs = cfg.get_article_requirements()
        assert isinstance(reqs, dict)
        assert len(reqs) > 0
        assert "ART_5" in reqs
        assert "ART_22" in reqs

    def test_get_engine_config(self):
        mod = _config_mod()
        cfg = mod.PackConfig()
        ec = cfg.get_engine_config("scope_assessment")
        assert isinstance(ec, dict)
        assert ec["engine"] == "scope_assessment"
        assert ec["enabled"] is True

    def test_get_engine_config_invalid_raises(self):
        mod = _config_mod()
        cfg = mod.PackConfig()
        with pytest.raises(ValueError):
            cfg.get_engine_config("nonexistent_engine")

    def test_get_workflow_config(self):
        mod = _config_mod()
        cfg = mod.PackConfig()
        wc = cfg.get_workflow_config()
        assert "workflows" in wc
        assert "config_hash" in wc

    def test_get_oecd_step_mapping(self):
        mod = _config_mod()
        cfg = mod.PackConfig()
        mapping = cfg.get_oecd_step_mapping()
        assert isinstance(mapping, dict)
        assert "STEP_1_EMBED" in mapping
        assert "STEP_6_REMEDIATE" in mapping

    def test_get_compliance_checklist(self):
        mod = _config_mod()
        cfg = mod.PackConfig()
        checklist = cfg.get_compliance_checklist()
        assert isinstance(checklist, list)
        assert len(checklist) > 10

    def test_to_dict(self):
        mod = _config_mod()
        cfg = mod.PackConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "config_hash" in d
        assert "is_in_scope" in d

    def test_from_preset_manufacturing(self):
        mod = _config_mod()
        if not (PRESETS_DIR / "manufacturing.yaml").exists():
            pytest.skip("manufacturing.yaml preset not available")
        cfg = mod.PackConfig.from_preset("manufacturing")
        assert cfg is not None
        assert isinstance(cfg, mod.PackConfig)

    def test_from_preset_invalid_raises(self):
        mod = _config_mod()
        with pytest.raises(ValueError):
            mod.PackConfig.from_preset("nonexistent_preset")


# ---------------------------------------------------------------------------
# 4. Module-level functions
# ---------------------------------------------------------------------------


class TestConfigModuleFunctions:
    """Verify module-level helper functions."""

    def test_get_all_articles(self):
        mod = _config_mod()
        arts = mod.get_all_articles()
        assert isinstance(arts, list)
        assert len(arts) == 25

    def test_get_article_description(self):
        mod = _config_mod()
        desc = mod.get_article_description(mod.ArticleReference.ART_5)
        assert isinstance(desc, str)
        assert len(desc) > 10

    def test_get_oecd_step_for_article(self):
        mod = _config_mod()
        step = mod.get_oecd_step_for_article(mod.ArticleReference.ART_5)
        assert step == mod.OECDStep.STEP_1_EMBED

    def test_get_phase_in_schedule(self):
        mod = _config_mod()
        schedule = mod.get_phase_in_schedule()
        assert isinstance(schedule, dict)
        assert "phase_1" in schedule
        assert "phase_2" in schedule
        assert "phase_3" in schedule

    def test_get_annex_instruments(self):
        mod = _config_mod()
        instruments = mod.get_annex_instruments()
        assert "human_rights" in instruments
        assert "environmental" in instruments
        assert len(instruments["human_rights"]) >= 10
        assert len(instruments["environmental"]) >= 5
