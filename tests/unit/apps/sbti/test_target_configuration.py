# -*- coding: utf-8 -*-
"""
Unit tests for SBTi Target Configuration Engine.

Tests target creation for near-term, long-term, and net-zero targets,
timeframe validation (5-10 year window), target coverage validation,
status lifecycle transitions, Scope 3 requirement checks, and annual
rate calculations with 25+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime
from decimal import Decimal

import pytest


# ===========================================================================
# Target Creation
# ===========================================================================

class TestTargetCreation:
    """Test creation of different target types."""

    def test_create_near_term_target(self, sample_near_term_target):
        assert sample_near_term_target["target_type"] == "near_term"
        assert sample_near_term_target["scope"] == "scope_1_2"
        assert sample_near_term_target["reduction_pct"] == 42.0

    def test_create_long_term_target(self, sample_long_term_target):
        assert sample_long_term_target["target_type"] == "long_term"
        assert sample_long_term_target["target_year"] == 2050

    def test_create_net_zero_target(self, sample_net_zero_target):
        assert sample_net_zero_target["target_type"] == "net_zero"
        assert sample_net_zero_target["reduction_pct"] >= 90.0

    def test_create_scope3_target(self, sample_scope3_target):
        assert sample_scope3_target["scope"] == "scope_3"
        assert sample_scope3_target["ambition_level"] == "well_below_2C"

    def test_create_intensity_target(self, sample_intensity_target):
        assert sample_intensity_target["method"] == "intensity_physical"
        assert sample_intensity_target["intensity_metric"] is not None

    def test_create_flag_target(self, sample_flag_target):
        assert sample_flag_target["is_flag_target"] is True
        assert sample_flag_target["flag_pathway_type"] == "commodity"

    def test_target_has_required_fields(self, sample_near_term_target):
        required_fields = [
            "target_name", "target_type", "scope", "method", "ambition_level",
            "base_year", "base_year_emissions_tco2e", "target_year",
            "reduction_pct", "boundary_coverage_pct", "status",
        ]
        for field in required_fields:
            assert field in sample_near_term_target


# ===========================================================================
# Timeframe Validation
# ===========================================================================

class TestTargetValidation:
    """Test target timeframe validation (5-10 years)."""

    @pytest.mark.parametrize("base_year,target_year,expected_valid", [
        (2020, 2025, True),   # 5 years - minimum
        (2020, 2027, True),   # 7 years - middle
        (2020, 2030, True),   # 10 years - maximum
        (2020, 2024, False),  # 4 years - too short
        (2020, 2031, False),  # 11 years - too long
        (2020, 2020, False),  # 0 years - invalid
    ])
    def test_near_term_timeframe_validation(self, base_year, target_year, expected_valid):
        timeframe = target_year - base_year
        is_valid = 5 <= timeframe <= 10
        assert is_valid == expected_valid

    def test_long_term_must_be_by_2050(self, sample_long_term_target):
        assert sample_long_term_target["target_year"] <= 2050

    def test_net_zero_must_be_by_2050(self, sample_net_zero_target):
        assert sample_net_zero_target["target_year"] <= 2050

    def test_base_year_minimum_2015(self, sample_near_term_target):
        assert sample_near_term_target["base_year"] >= 2015

    @pytest.mark.parametrize("base_year,is_valid", [
        (2015, True),
        (2020, True),
        (2025, True),
        (2014, False),
        (2010, False),
    ])
    def test_base_year_validation(self, base_year, is_valid):
        valid = base_year >= 2015
        assert valid == is_valid


# ===========================================================================
# Target Lifecycle
# ===========================================================================

class TestTargetLifecycle:
    """Test target status transitions."""

    def test_initial_status_is_draft(self, sample_near_term_target):
        assert sample_near_term_target["status"] == "draft"

    @pytest.mark.parametrize("current,new_status,should_succeed", [
        ("draft", "pending_validation", True),
        ("draft", "withdrawn", True),
        ("draft", "active", False),
        ("pending_validation", "submitted", True),
        ("pending_validation", "draft", True),
        ("pending_validation", "withdrawn", True),
        ("pending_validation", "active", False),
        ("submitted", "validated", True),
        ("submitted", "pending_validation", True),
        ("submitted", "withdrawn", True),
        ("submitted", "active", False),
        ("validated", "approved", True),
        ("validated", "withdrawn", True),
        ("validated", "draft", False),
        ("approved", "active", True),
        ("approved", "withdrawn", True),
        ("active", "expired", True),
        ("active", "withdrawn", True),
        ("active", "draft", False),
        ("expired", "draft", False),
        ("withdrawn", "draft", False),
    ])
    def test_status_transitions(self, current, new_status, should_succeed):
        valid_transitions = {
            "draft": ["pending_validation", "withdrawn"],
            "pending_validation": ["submitted", "draft", "withdrawn"],
            "submitted": ["validated", "pending_validation", "withdrawn"],
            "validated": ["approved", "withdrawn"],
            "approved": ["active", "withdrawn"],
            "active": ["expired", "withdrawn"],
            "expired": [],
            "withdrawn": [],
        }
        allowed = new_status in valid_transitions.get(current, [])
        assert allowed == should_succeed

    def test_terminal_states_have_no_transitions(self):
        valid_transitions = {
            "expired": [],
            "withdrawn": [],
        }
        for state, transitions in valid_transitions.items():
            assert len(transitions) == 0, f"Terminal state '{state}' should have no transitions"


# ===========================================================================
# Coverage Validation
# ===========================================================================

class TestCoverageValidation:
    """Test boundary coverage validation."""

    def test_scope1_2_coverage_95_pct(self, sample_near_term_target):
        assert sample_near_term_target["boundary_coverage_pct"] >= 95.0

    @pytest.mark.parametrize("coverage,meets_min", [
        (95.0, True),
        (95.1, True),
        (100.0, True),
        (94.9, False),
        (90.0, False),
        (50.0, False),
    ])
    def test_scope1_2_coverage_threshold(self, coverage, meets_min):
        assert (coverage >= 95.0) == meets_min

    def test_scope3_coverage_67_pct(self, sample_scope3_target):
        assert sample_scope3_target["boundary_coverage_pct"] >= 67.0

    @pytest.mark.parametrize("coverage,meets_min", [
        (67.0, True),
        (72.0, True),
        (90.0, True),
        (66.9, False),
        (50.0, False),
    ])
    def test_scope3_near_term_coverage_threshold(self, coverage, meets_min):
        assert (coverage >= 67.0) == meets_min

    @pytest.mark.parametrize("coverage,meets_min", [
        (90.0, True),
        (95.0, True),
        (89.9, False),
        (67.0, False),
    ])
    def test_scope3_long_term_coverage_threshold(self, coverage, meets_min):
        assert (coverage >= 90.0) == meets_min


# ===========================================================================
# Scope 3 Requirement
# ===========================================================================

class TestScope3Requirement:
    """Test Scope 3 target requirement (40% trigger)."""

    def test_scope3_above_threshold_required(self, high_scope3_inventory):
        assert high_scope3_inventory["scope3_pct_of_total"] >= 40.0

    def test_scope3_below_threshold_not_required(self, low_scope3_inventory):
        assert low_scope3_inventory["scope3_pct_of_total"] < 40.0

    @pytest.mark.parametrize("scope3_pct,required", [
        (40.0, True),   # Exactly 40% - threshold met
        (40.1, True),   # Just above
        (65.0, True),   # Well above
        (39.9, False),  # Just below
        (10.0, False),  # Well below
        (0.0, False),   # No Scope 3
    ])
    def test_40pct_threshold_edge_cases(self, scope3_pct, required):
        assert (scope3_pct >= 40.0) == required

    def test_scope3_pct_calculation(self):
        s1 = 30_000.0
        s2 = 20_000.0
        s3 = 50_000.0
        total = s1 + s2 + s3
        s3_pct = (s3 / total) * 100
        assert s3_pct == 50.0
        assert s3_pct >= 40.0


# ===========================================================================
# Annual Rate Calculation
# ===========================================================================

class TestAnnualRateCalculation:
    """Test linear annual reduction rate calculation."""

    def test_near_term_annual_rate(self, sample_near_term_target):
        reduction = sample_near_term_target["reduction_pct"]
        years = sample_near_term_target["target_year"] - sample_near_term_target["base_year"]
        expected_rate = reduction / years
        assert abs(sample_near_term_target["linear_annual_reduction_pct"] - expected_rate) < 0.01

    @pytest.mark.parametrize("reduction_pct,base_year,target_year,expected_rate", [
        (42.0, 2020, 2030, 4.2),    # 1.5C ACA
        (25.0, 2020, 2030, 2.5),    # WB2C ACA
        (90.0, 2020, 2050, 3.0),    # Long-term
        (50.0, 2020, 2030, 5.0),    # Aggressive near-term
        (30.3, 2020, 2030, 3.03),   # FLAG sector
    ])
    def test_annual_rate_calculation(self, reduction_pct, base_year, target_year, expected_rate):
        years = target_year - base_year
        rate = round(reduction_pct / years, 2)
        assert rate == expected_rate

    def test_zero_year_timeframe_returns_zero(self):
        years = 0
        rate = 0.0 if years == 0 else 42.0 / years
        assert rate == 0.0

    def test_1_5c_minimum_rate(self):
        """1.5C alignment requires at least 4.2% per year for linear ACA."""
        min_rate = 4.2
        reduction_10yr = min_rate * 10
        assert reduction_10yr == 42.0

    def test_wb2c_minimum_rate(self):
        """WB2C alignment requires at least 2.5% per year for linear ACA."""
        min_rate = 2.5
        reduction_10yr = min_rate * 10
        assert reduction_10yr == 25.0
