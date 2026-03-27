# -*- coding: utf-8 -*-
"""
Unit tests for Dual Reporting Reconciliation configuration.

AGENT-MRV-013: Dual Reporting Reconciliation Agent
Target: 30 tests covering DualReportingReconciliationConfig singleton.
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

from greenlang.agents.mrv.dual_reporting_reconciliation.config import (
    DualReportingReconciliationConfig,
    get_config,
)


# ===========================================================================
# 1. Singleton Tests
# ===========================================================================


class TestConfigSingleton:
    """Test DualReportingReconciliationConfig singleton pattern."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert cfg is not None

    def test_singleton_returns_same_instance(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_config_class_is_correct_type(self):
        cfg = get_config()
        assert isinstance(cfg, DualReportingReconciliationConfig)


# ===========================================================================
# 2. Default Values
# ===========================================================================


class TestConfigDefaults:
    """Test default configuration values."""

    def test_completeness_weight(self):
        cfg = get_config()
        assert hasattr(cfg, "completeness_weight")
        assert float(cfg.completeness_weight) == pytest.approx(0.30, abs=0.01)

    def test_consistency_weight(self):
        cfg = get_config()
        assert hasattr(cfg, "consistency_weight")
        assert float(cfg.consistency_weight) == pytest.approx(0.25, abs=0.01)

    def test_accuracy_weight(self):
        cfg = get_config()
        assert hasattr(cfg, "accuracy_weight")
        assert float(cfg.accuracy_weight) == pytest.approx(0.25, abs=0.01)

    def test_transparency_weight(self):
        cfg = get_config()
        assert hasattr(cfg, "transparency_weight")
        assert float(cfg.transparency_weight) == pytest.approx(0.20, abs=0.01)

    def test_weights_sum_to_one(self):
        cfg = get_config()
        total = (
            float(cfg.completeness_weight)
            + float(cfg.consistency_weight)
            + float(cfg.accuracy_weight)
            + float(cfg.transparency_weight)
        )
        assert total == pytest.approx(1.0, abs=0.01)

    def test_decimal_places(self):
        cfg = get_config()
        assert hasattr(cfg, "decimal_places")
        assert cfg.decimal_places == 8

    def test_enabled_frameworks(self):
        cfg = get_config()
        assert hasattr(cfg, "enabled_frameworks")
        frameworks = cfg.enabled_frameworks
        assert len(frameworks) == 7

    def test_strict_mode_default(self):
        cfg = get_config()
        assert hasattr(cfg, "strict_mode")
        assert cfg.strict_mode is False

    def test_fail_on_non_compliant_default(self):
        cfg = get_config()
        assert hasattr(cfg, "fail_on_non_compliant")
        assert cfg.fail_on_non_compliant is False

    def test_immaterial_threshold(self):
        cfg = get_config()
        assert hasattr(cfg, "immaterial_threshold")
        assert float(cfg.immaterial_threshold) == pytest.approx(5.0, abs=1.0)

    def test_minor_threshold(self):
        cfg = get_config()
        assert hasattr(cfg, "minor_threshold")
        assert float(cfg.minor_threshold) == pytest.approx(15.0, abs=1.0)

    def test_material_threshold(self):
        cfg = get_config()
        assert hasattr(cfg, "material_threshold")
        assert float(cfg.material_threshold) == pytest.approx(50.0, abs=1.0)

    def test_significant_threshold(self):
        cfg = get_config()
        assert hasattr(cfg, "significant_threshold")
        assert float(cfg.significant_threshold) == pytest.approx(100.0, abs=1.0)

    def test_assurance_threshold(self):
        cfg = get_config()
        assert hasattr(cfg, "assurance_threshold")
        assert float(cfg.assurance_threshold) == pytest.approx(0.90, abs=0.01)

    def test_stable_threshold(self):
        cfg = get_config()
        assert hasattr(cfg, "stable_threshold")
        assert float(cfg.stable_threshold) == pytest.approx(2.0, abs=0.5)

    def test_trend_min_periods(self):
        cfg = get_config()
        assert hasattr(cfg, "trend_min_periods")
        assert cfg.trend_min_periods == 2

    def test_trend_max_periods(self):
        cfg = get_config()
        assert hasattr(cfg, "trend_max_periods")
        assert cfg.trend_max_periods == 10


# ===========================================================================
# 3. Attribute Access
# ===========================================================================


class TestConfigAttributes:
    """Test that config attributes are accessible."""

    def test_has_to_dict(self):
        cfg = get_config()
        if hasattr(cfg, "to_dict"):
            d = cfg.to_dict()
            assert isinstance(d, dict)
            assert len(d) > 0

    def test_config_repr(self):
        cfg = get_config()
        r = repr(cfg)
        assert "DualReporting" in r or "Config" in r or "dual" in r.lower()
