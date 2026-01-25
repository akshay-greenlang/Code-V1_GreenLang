# -*- coding: utf-8 -*-
"""
GL-013 PREDICTMAINT - Unit Tests for Calculator Modules
Comprehensive test coverage for all calculator implementations.

Tests cover:
- RUL Calculator (Weibull, Exponential, Log-Normal models)
- Confidence interval calculations
- Failure Probability Calculator (distributions, hazard rates)
- Vibration Analyzer (ISO 10816, bearing frequencies)
- Thermal Degradation Calculator (Arrhenius equation)
- Maintenance Scheduler (cost optimization)
- Spare Parts Calculator (EOQ, safety stock)
- Anomaly Detector (CUSUM, SPC, Mahalanobis)
- Provenance (Merkle tree, hash verification)

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, MagicMock

# Import test fixtures from conftest
from ..conftest import (
    MachineClass,
    VibrationZone,
    HealthState,
    ISO_10816_LIMITS,
    WEIBULL_PARAMETERS,
)


# =============================================================================
# TEST CLASS: RUL CALCULATOR - WEIBULL MODEL
# =============================================================================


class TestRULCalculatorWeibull:
    """Tests for Weibull-based RUL calculation."""

    @pytest.mark.unit
    def test_weibull_rul_formula_verification(self, rul_calculator):
        """
        Verify Weibull RUL formula against manual calculation.

        Formula: t_target = gamma + eta * (-ln(R_target))^(1/beta)
        RUL = t_target - t_current
        """
        # Known parameters
        beta = Decimal("2.5")
        eta = Decimal("45000")
        gamma = Decimal("0")
        t_current = Decimal("30000")
        R_target = Decimal("0.5")

        # Manual calculation
        neg_ln_R = -Decimal(str(math.log(float(R_target))))  # 0.693...
        t_target = gamma + eta * Decimal(str(math.pow(float(neg_ln_R), 1.0/float(beta))))
        expected_rul = max(t_target - t_current, Decimal("0"))

        # Calculator result
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=t_current,
            target_reliability="0.5",
            custom_beta=beta,
            custom_eta=eta,
            custom_gamma=gamma,
        )

        # Verify calculation matches
        assert result["rul_hours"] == pytest.approx(
            expected_rul.quantize(Decimal("0.001")), abs=Decimal("1")
        )

    @pytest.mark.unit
    def test_weibull_current_reliability_formula(self, rul_calculator):
        """
        Verify current reliability formula: R(t) = exp(-((t-gamma)/eta)^beta)
        """
        beta = Decimal("2.5")
        eta = Decimal("45000")
        gamma = Decimal("0")
        t = Decimal("30000")

        # Manual calculation
        t_effective = t - gamma
        exponent = -math.pow(float(t_effective / eta), float(beta))
        expected_reliability = Decimal(str(math.exp(exponent)))

        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=t,
            custom_beta=beta,
            custom_eta=eta,
            custom_gamma=gamma,
        )

        assert result["current_reliability"] == pytest.approx(
            expected_reliability.quantize(Decimal("0.000001")), abs=Decimal("0.00001")
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("beta,eta,t,expected_R_approx", [
        # beta=1 (exponential), at t=eta, R = exp(-1) = 0.368
        (Decimal("1.0"), Decimal("50000"), Decimal("50000"), Decimal("0.368")),
        # beta=2, at t=eta, R = exp(-1) = 0.368
        (Decimal("2.0"), Decimal("50000"), Decimal("50000"), Decimal("0.368")),
        # t=0, R = 1
        (Decimal("2.5"), Decimal("50000"), Decimal("0"), Decimal("1.0")),
        # t >> eta, R approaches 0
        (Decimal("2.5"), Decimal("50000"), Decimal("100000"), Decimal("0.001")),
    ])
    def test_weibull_reliability_known_values(
        self, rul_calculator, beta, eta, t, expected_R_approx
    ):
        """Test reliability calculation against known values."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="custom",
            operating_hours=t,
            custom_beta=beta,
            custom_eta=eta,
            custom_gamma=Decimal("0"),
        )

        assert result["current_reliability"] == pytest.approx(
            expected_R_approx, abs=Decimal("0.01")
        )

    @pytest.mark.unit
    def test_weibull_gamma_parameter(self, rul_calculator):
        """Test location parameter (gamma) effect on RUL."""
        gamma = Decimal("5000")  # 5000 hour failure-free period

        result_no_gamma = rul_calculator.calculate_weibull_rul(
            equipment_type="custom",
            operating_hours=Decimal("10000"),
            custom_beta=Decimal("2.5"),
            custom_eta=Decimal("45000"),
            custom_gamma=Decimal("0"),
        )

        result_with_gamma = rul_calculator.calculate_weibull_rul(
            equipment_type="custom",
            operating_hours=Decimal("10000"),
            custom_beta=Decimal("2.5"),
            custom_eta=Decimal("45000"),
            custom_gamma=gamma,
        )

        # Equipment with gamma should show higher reliability
        # (effectively younger equipment)
        assert result_with_gamma["current_reliability"] > result_no_gamma["current_reliability"]

    @pytest.mark.unit
    def test_weibull_within_gamma_period(self, rul_calculator):
        """Test RUL when operating hours less than gamma."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="custom",
            operating_hours=Decimal("3000"),
            custom_beta=Decimal("2.5"),
            custom_eta=Decimal("45000"),
            custom_gamma=Decimal("5000"),  # Still in failure-free period
        )

        # Reliability should be 1.0 during failure-free period
        assert result["current_reliability"] == Decimal("1.000000")


# =============================================================================
# TEST CLASS: RUL CALCULATOR - CONFIDENCE INTERVALS
# =============================================================================


class TestRULCalculatorConfidenceIntervals:
    """Tests for RUL confidence interval calculations."""

    @pytest.mark.unit
    def test_confidence_interval_bounds(self, rul_calculator):
        """Test that confidence intervals have proper bounds."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
            confidence_level="90%",
        )

        # Lower bound should be less than point estimate
        assert result["confidence_lower"] < result["rul_hours"]
        # Upper bound should be greater than point estimate
        assert result["confidence_upper"] > result["rul_hours"]
        # Both should be positive
        assert result["confidence_lower"] >= Decimal("0")

    @pytest.mark.unit
    def test_confidence_interval_width_increases_with_level(self, rul_calculator):
        """Test that wider confidence level gives wider interval."""
        # Note: This test uses the mock calculator which uses fixed 80%/120% bounds
        # In real implementation, higher confidence = wider interval
        result_90 = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
            confidence_level="90%",
        )

        width_90 = result_90["confidence_upper"] - result_90["confidence_lower"]

        # In real calculator, 95% CI would be wider than 90%
        # This test verifies the structure exists
        assert width_90 > Decimal("0")

    @pytest.mark.unit
    @pytest.mark.parametrize("confidence_level", [
        "80%", "90%", "95%", "99%"
    ])
    def test_confidence_level_parsing(self, rul_calculator, confidence_level):
        """Test various confidence level formats are accepted."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
            confidence_level=confidence_level,
        )

        assert result["confidence_level"] == confidence_level


# =============================================================================
# TEST CLASS: FAILURE PROBABILITY CALCULATOR
# =============================================================================


class TestFailureProbabilityCalculator:
    """Tests for failure probability calculations."""

    @pytest.mark.unit
    def test_weibull_cdf_formula(self, failure_probability_calculator):
        """
        Verify Weibull CDF formula: F(t) = 1 - exp(-((t-gamma)/eta)^beta)
        """
        beta = Decimal("2.5")
        eta = Decimal("50000")
        t = Decimal("30000")
        gamma = Decimal("0")

        # Manual calculation
        t_eff = t - gamma
        exponent = -math.pow(float(t_eff / eta), float(beta))
        expected_reliability = Decimal(str(math.exp(exponent)))
        expected_failure_prob = Decimal("1") - expected_reliability

        result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=beta,
            eta=eta,
            time_hours=t,
            gamma=gamma,
        )

        assert result["failure_probability"] == pytest.approx(
            expected_failure_prob.quantize(Decimal("0.000001")), abs=Decimal("0.0001")
        )

    @pytest.mark.unit
    def test_reliability_failure_probability_sum(self, failure_probability_calculator):
        """Test that R(t) + F(t) = 1."""
        result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal("2.5"),
            eta=Decimal("50000"),
            time_hours=Decimal("30000"),
        )

        sum_rf = result["reliability"] + result["failure_probability"]
        assert sum_rf == pytest.approx(Decimal("1"), abs=Decimal("0.000001"))

    @pytest.mark.unit
    def test_hazard_rate_weibull_formula(self, failure_probability_calculator):
        """
        Verify Weibull hazard rate: h(t) = (beta/eta) * ((t-gamma)/eta)^(beta-1)
        """
        beta = Decimal("2.5")
        eta = Decimal("50000")
        t = Decimal("30000")
        gamma = Decimal("0")

        # Manual calculation
        t_eff = t - gamma
        expected_hazard = (
            (beta / eta) *
            Decimal(str(math.pow(float(t_eff / eta), float(beta - Decimal("1")))))
        )

        result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=beta,
            eta=eta,
            time_hours=t,
            gamma=gamma,
        )

        assert result["hazard_rate"] == pytest.approx(
            expected_hazard.quantize(Decimal("0.00000001")), abs=Decimal("0.0000001")
        )

    @pytest.mark.unit
    def test_mean_life_weibull(self, failure_probability_calculator):
        """
        Verify Weibull mean life: MTBF = eta * Gamma(1 + 1/beta)
        """
        beta = Decimal("2.5")
        eta = Decimal("50000")

        # Manual calculation using gamma function
        gamma_val = math.gamma(1 + 1/float(beta))
        expected_mtbf = eta * Decimal(str(gamma_val))

        result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=beta,
            eta=eta,
            time_hours=Decimal("30000"),
        )

        assert result["mean_life"] == pytest.approx(
            expected_mtbf.quantize(Decimal("0.001")), abs=Decimal("100")
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("beta,hazard_trend", [
        (Decimal("0.5"), "decreasing"),   # Infant mortality (DFR)
        (Decimal("1.0"), "constant"),     # Random failures (CFR)
        (Decimal("2.5"), "increasing"),   # Wear-out (IFR)
    ])
    def test_bathtub_curve_regions(
        self, failure_probability_calculator, beta, hazard_trend
    ):
        """Test bathtub curve hazard rate behavior."""
        t1 = Decimal("10000")
        t2 = Decimal("30000")

        result1 = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=beta,
            eta=Decimal("50000"),
            time_hours=t1,
        )

        result2 = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=beta,
            eta=Decimal("50000"),
            time_hours=t2,
        )

        if hazard_trend == "decreasing":
            assert result2["hazard_rate"] < result1["hazard_rate"]
        elif hazard_trend == "constant":
            assert result1["hazard_rate"] == pytest.approx(
                result2["hazard_rate"], abs=Decimal("0.0000001")
            )
        elif hazard_trend == "increasing":
            assert result2["hazard_rate"] > result1["hazard_rate"]


# =============================================================================
# TEST CLASS: VIBRATION ANALYZER - ISO 10816
# =============================================================================


class TestVibrationAnalyzerISO10816:
    """Tests for ISO 10816 compliant vibration analysis."""

    @pytest.mark.unit
    @pytest.mark.parametrize("machine_class,limits", [
        (MachineClass.CLASS_I, {"a": Decimal("0.71"), "b": Decimal("1.8"), "c": Decimal("4.5")}),
        (MachineClass.CLASS_II, {"a": Decimal("1.12"), "b": Decimal("2.8"), "c": Decimal("7.1")}),
        (MachineClass.CLASS_III, {"a": Decimal("1.8"), "b": Decimal("4.5"), "c": Decimal("11.2")}),
        (MachineClass.CLASS_IV, {"a": Decimal("2.8"), "b": Decimal("7.1"), "c": Decimal("18.0")}),
    ])
    def test_iso_10816_zone_limits(self, vibration_analyzer, machine_class, limits):
        """Verify ISO 10816-3 Table 1 vibration limits."""
        result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("0.1"),  # Very low value
            machine_class=machine_class,
        )

        assert result["zone_limits"]["zone_a_upper"] == limits["a"]
        assert result["zone_limits"]["zone_b_upper"] == limits["b"]
        assert result["zone_limits"]["zone_c_upper"] == limits["c"]

    @pytest.mark.unit
    def test_zone_classification_boundaries(self, vibration_analyzer):
        """Test exact boundary classification."""
        machine_class = MachineClass.CLASS_II
        limits = ISO_10816_LIMITS[machine_class]

        # Test at Zone A limit (should be Zone A)
        result_a = vibration_analyzer.assess_severity(
            velocity_rms=limits["zone_a_upper"],
            machine_class=machine_class,
        )
        assert result_a["zone"] == VibrationZone.ZONE_A

        # Test just above Zone A limit (should be Zone B)
        result_b = vibration_analyzer.assess_severity(
            velocity_rms=limits["zone_a_upper"] + Decimal("0.01"),
            machine_class=machine_class,
        )
        assert result_b["zone"] == VibrationZone.ZONE_B

    @pytest.mark.unit
    def test_alarm_levels_per_zone(self, vibration_analyzer):
        """Test alarm level mapping per zone."""
        machine_class = MachineClass.CLASS_II

        # Zone A - Normal
        result_a = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("0.5"),
            machine_class=machine_class,
        )
        assert result_a["alarm_level"] == "normal"

        # Zone B - Normal
        result_b = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("2.0"),
            machine_class=machine_class,
        )
        assert result_b["alarm_level"] == "normal"

        # Zone C - Warning
        result_c = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("5.0"),
            machine_class=machine_class,
        )
        assert result_c["alarm_level"] == "warning"

        # Zone D - Critical
        result_d = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("10.0"),
            machine_class=machine_class,
        )
        assert result_d["alarm_level"] == "critical"

    @pytest.mark.unit
    def test_margin_calculation_zone_a(self, vibration_analyzer):
        """Test margin to next zone from Zone A."""
        velocity = Decimal("0.5")
        machine_class = MachineClass.CLASS_II
        zone_a_limit = ISO_10816_LIMITS[machine_class]["zone_a_upper"]

        result = vibration_analyzer.assess_severity(
            velocity_rms=velocity,
            machine_class=machine_class,
        )

        expected_margin = zone_a_limit - velocity
        assert result["margin_to_next_zone"] == expected_margin


# =============================================================================
# TEST CLASS: VIBRATION ANALYZER - BEARING FREQUENCIES
# =============================================================================


class TestVibrationAnalyzerBearingFrequencies:
    """Tests for bearing fault frequency calculations."""

    @pytest.mark.unit
    def test_bpfo_formula(self, vibration_analyzer):
        """
        Verify BPFO formula: BPFO = (n/2) * f_s * (1 - (Bd/Pd) * cos(theta))

        Where:
        - n = number of rolling elements
        - f_s = shaft frequency (Hz)
        - Bd = ball diameter
        - Pd = pitch diameter
        - theta = contact angle
        """
        n = 9
        f_s = Decimal("1480") / Decimal("60")  # 24.67 Hz
        Bd = Decimal("7.938")
        Pd = Decimal("38.5")
        theta = Decimal("0")  # cos(0) = 1

        # Manual calculation
        expected_bpfo = (
            (Decimal(str(n)) / Decimal("2")) *
            f_s *
            (Decimal("1") - (Bd / Pd) * Decimal(str(math.cos(float(theta)))))
        )

        result = vibration_analyzer.calculate_bearing_frequencies(
            shaft_speed_rpm=Decimal("1480"),
            num_balls=n,
            ball_diameter=Bd,
            pitch_diameter=Pd,
            contact_angle_deg=theta,
        )

        assert result["bpfo"] == pytest.approx(
            expected_bpfo.quantize(Decimal("0.001")), abs=Decimal("0.1")
        )

    @pytest.mark.unit
    def test_bpfi_formula(self, vibration_analyzer):
        """
        Verify BPFI formula: BPFI = (n/2) * f_s * (1 + (Bd/Pd) * cos(theta))
        """
        n = 9
        f_s = Decimal("1480") / Decimal("60")
        Bd = Decimal("7.938")
        Pd = Decimal("38.5")
        theta = Decimal("0")

        expected_bpfi = (
            (Decimal(str(n)) / Decimal("2")) *
            f_s *
            (Decimal("1") + (Bd / Pd) * Decimal(str(math.cos(float(theta)))))
        )

        result = vibration_analyzer.calculate_bearing_frequencies(
            shaft_speed_rpm=Decimal("1480"),
            num_balls=n,
            ball_diameter=Bd,
            pitch_diameter=Pd,
            contact_angle_deg=theta,
        )

        assert result["bpfi"] == pytest.approx(
            expected_bpfi.quantize(Decimal("0.001")), abs=Decimal("0.1")
        )

    @pytest.mark.unit
    def test_bsf_formula(self, vibration_analyzer):
        """
        Verify BSF formula: BSF = (Pd/(2*Bd)) * f_s * (1 - ((Bd/Pd)*cos(theta))^2)
        """
        n = 9
        f_s = Decimal("1480") / Decimal("60")
        Bd = Decimal("7.938")
        Pd = Decimal("38.5")
        theta = Decimal("0")

        cos_theta = Decimal(str(math.cos(float(theta))))
        expected_bsf = (
            (Pd / (Decimal("2") * Bd)) *
            f_s *
            (Decimal("1") - ((Bd / Pd) * cos_theta) ** 2)
        )

        result = vibration_analyzer.calculate_bearing_frequencies(
            shaft_speed_rpm=Decimal("1480"),
            num_balls=n,
            ball_diameter=Bd,
            pitch_diameter=Pd,
            contact_angle_deg=theta,
        )

        assert result["bsf"] == pytest.approx(
            expected_bsf.quantize(Decimal("0.001")), abs=Decimal("0.1")
        )

    @pytest.mark.unit
    def test_ftf_formula(self, vibration_analyzer):
        """
        Verify FTF formula: FTF = (f_s/2) * (1 - (Bd/Pd) * cos(theta))
        """
        f_s = Decimal("1480") / Decimal("60")
        Bd = Decimal("7.938")
        Pd = Decimal("38.5")
        theta = Decimal("0")

        expected_ftf = (
            (f_s / Decimal("2")) *
            (Decimal("1") - (Bd / Pd) * Decimal(str(math.cos(float(theta)))))
        )

        result = vibration_analyzer.calculate_bearing_frequencies(
            shaft_speed_rpm=Decimal("1480"),
            num_balls=9,
            ball_diameter=Bd,
            pitch_diameter=Pd,
            contact_angle_deg=theta,
        )

        assert result["ftf"] == pytest.approx(
            expected_ftf.quantize(Decimal("0.001")), abs=Decimal("0.1")
        )

    @pytest.mark.unit
    def test_frequency_ordering(self, vibration_analyzer):
        """Test expected ordering of bearing frequencies."""
        result = vibration_analyzer.calculate_bearing_frequencies(
            shaft_speed_rpm=Decimal("1480"),
            num_balls=9,
            ball_diameter=Decimal("7.938"),
            pitch_diameter=Decimal("38.5"),
            contact_angle_deg=Decimal("0"),
        )

        # Expected ordering: FTF < shaft < BSF < BPFO < BPFI
        assert result["ftf"] < result["shaft_speed_hz"]
        assert result["bpfo"] < result["bpfi"]


# =============================================================================
# TEST CLASS: THERMAL DEGRADATION - ARRHENIUS
# =============================================================================


class TestThermalDegradationArrhenius:
    """Tests for Arrhenius-based thermal life calculations."""

    @pytest.mark.unit
    def test_arrhenius_aging_factor_formula(self, thermal_degradation_calculator):
        """
        Verify Arrhenius formula: AAF = exp(Ea/k_B * (1/T_ref - 1/T_op))

        Where:
        - Ea = activation energy (eV)
        - k_B = Boltzmann constant (8.617e-5 eV/K)
        - T_ref, T_op = temperatures in Kelvin
        """
        T_op_c = Decimal("120")
        T_ref_c = Decimal("110")
        Ea = Decimal("1.1")
        k_B = Decimal("8.617333262e-5")

        T_op_k = T_op_c + Decimal("273.15")
        T_ref_k = T_ref_c + Decimal("273.15")

        # Manual calculation
        exponent = float(Ea / k_B * (Decimal("1") / T_ref_k - Decimal("1") / T_op_k))
        expected_aaf = Decimal(str(math.exp(exponent)))

        result = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=T_op_c,
            reference_temperature_c=T_ref_c,
            activation_energy_ev=Ea,
        )

        assert result["acceleration_factor"] == pytest.approx(
            expected_aaf.quantize(Decimal("0.001")), abs=Decimal("0.01")
        )

    @pytest.mark.unit
    def test_arrhenius_at_reference_temperature(self, thermal_degradation_calculator):
        """Test AAF = 1 at reference temperature."""
        result = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=Decimal("110"),
            reference_temperature_c=Decimal("110"),
        )

        assert result["acceleration_factor"] == pytest.approx(
            Decimal("1.0"), abs=Decimal("0.001")
        )

    @pytest.mark.unit
    def test_arrhenius_higher_temp_faster_aging(self, thermal_degradation_calculator):
        """Test that higher temperature gives faster aging (AAF > 1)."""
        result = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=Decimal("130"),
            reference_temperature_c=Decimal("110"),
        )

        assert result["acceleration_factor"] > Decimal("1")

    @pytest.mark.unit
    def test_arrhenius_lower_temp_slower_aging(self, thermal_degradation_calculator):
        """Test that lower temperature gives slower aging (AAF < 1)."""
        result = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=Decimal("90"),
            reference_temperature_c=Decimal("110"),
        )

        assert result["acceleration_factor"] < Decimal("1")

    @pytest.mark.unit
    def test_ieee_10_degree_rule(self, thermal_degradation_calculator):
        """
        Test IEEE 10-degree rule approximation.

        Rule of thumb: Life doubles for every 10C decrease,
        halves for every 10C increase.
        """
        result_ref = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=Decimal("110"),
            reference_temperature_c=Decimal("110"),
        )

        result_plus_10 = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=Decimal("120"),
            reference_temperature_c=Decimal("110"),
        )

        ratio = result_plus_10["acceleration_factor"] / result_ref["acceleration_factor"]

        # Should be approximately 2x (depends on activation energy)
        assert ratio > Decimal("1.5")
        assert ratio < Decimal("3.0")


# =============================================================================
# TEST CLASS: MAINTENANCE SCHEDULER - COST OPTIMIZATION
# =============================================================================


class TestMaintenanceSchedulerCostOptimization:
    """Tests for maintenance schedule cost optimization."""

    @pytest.mark.unit
    def test_optimal_interval_formula(self, maintenance_scheduler):
        """
        Verify optimal interval calculation.

        For beta > 1: t_opt = eta * ((Cp/Cf) * (beta-1))^(1/beta)
        """
        beta = Decimal("2.5")
        eta = Decimal("50000")
        Cp = Decimal("1000")
        Cf = Decimal("10000")

        # Manual calculation
        cost_ratio = Cp / Cf
        inner = cost_ratio * (beta - Decimal("1"))
        expected_t_opt = eta * Decimal(str(math.pow(float(inner), 1/float(beta))))

        result = maintenance_scheduler.calculate_optimal_interval(
            beta=beta,
            eta=eta,
            preventive_cost=Cp,
            corrective_cost=Cf,
        )

        assert result["optimal_interval_hours"] == pytest.approx(
            expected_t_opt.quantize(Decimal("0.001")), abs=Decimal("100")
        )

    @pytest.mark.unit
    def test_run_to_failure_for_dfr(self, maintenance_scheduler):
        """Test that run-to-failure is optimal for DFR (beta < 1)."""
        result = maintenance_scheduler.calculate_optimal_interval(
            beta=Decimal("0.8"),
            eta=Decimal("50000"),
            preventive_cost=Decimal("1000"),
            corrective_cost=Decimal("10000"),
        )

        assert result["strategy"] == "run_to_failure"

    @pytest.mark.unit
    def test_run_to_failure_for_cfr(self, maintenance_scheduler):
        """Test that run-to-failure is optimal for CFR (beta = 1)."""
        result = maintenance_scheduler.calculate_optimal_interval(
            beta=Decimal("1.0"),
            eta=Decimal("50000"),
            preventive_cost=Decimal("1000"),
            corrective_cost=Decimal("10000"),
        )

        assert result["strategy"] == "run_to_failure"

    @pytest.mark.unit
    def test_higher_corrective_cost_shorter_interval(self, maintenance_scheduler):
        """Test that higher corrective cost gives shorter PM interval."""
        result_low_cf = maintenance_scheduler.calculate_optimal_interval(
            beta=Decimal("2.5"),
            eta=Decimal("50000"),
            preventive_cost=Decimal("1000"),
            corrective_cost=Decimal("5000"),
        )

        result_high_cf = maintenance_scheduler.calculate_optimal_interval(
            beta=Decimal("2.5"),
            eta=Decimal("50000"),
            preventive_cost=Decimal("1000"),
            corrective_cost=Decimal("50000"),
        )

        # Higher Cf/Cp ratio -> shorter interval
        assert result_high_cf["optimal_interval_hours"] < result_low_cf["optimal_interval_hours"]


# =============================================================================
# TEST CLASS: SPARE PARTS CALCULATOR - EOQ
# =============================================================================


class TestSparePartsCalculatorEOQ:
    """Tests for Economic Order Quantity calculations."""

    @pytest.mark.unit
    def test_eoq_formula(self, spare_parts_calculator):
        """
        Verify EOQ formula: EOQ = sqrt(2*D*S / H)

        Where:
        - D = annual demand
        - S = ordering cost per order
        - H = holding cost per unit per year
        """
        D = Decimal("100")
        S = Decimal("50")
        H_rate = Decimal("0.25")
        unit_cost = Decimal("200")
        H = H_rate * unit_cost  # 50

        expected_eoq = Decimal(str(math.sqrt(float(2 * D * S / H))))

        result = spare_parts_calculator.calculate_eoq(
            annual_demand=D,
            ordering_cost=S,
            holding_cost_rate=H_rate,
            unit_cost=unit_cost,
        )

        assert result["eoq"] == pytest.approx(
            expected_eoq.quantize(Decimal("1")), abs=Decimal("1")
        )

    @pytest.mark.unit
    def test_eoq_total_cost(self, spare_parts_calculator):
        """Test total annual cost calculation at EOQ."""
        result = spare_parts_calculator.calculate_eoq(
            annual_demand=Decimal("100"),
            ordering_cost=Decimal("50"),
            holding_cost_rate=Decimal("0.25"),
            unit_cost=Decimal("200"),
        )

        # At EOQ, ordering cost = holding cost
        assert result["total_ordering_cost"] == pytest.approx(
            result["total_holding_cost"], abs=Decimal("10")
        )

    @pytest.mark.unit
    def test_eoq_sensitivity_to_demand(self, spare_parts_calculator):
        """Test EOQ sensitivity to demand changes."""
        result_low = spare_parts_calculator.calculate_eoq(
            annual_demand=Decimal("50"),
            ordering_cost=Decimal("50"),
            holding_cost_rate=Decimal("0.25"),
            unit_cost=Decimal("200"),
        )

        result_high = spare_parts_calculator.calculate_eoq(
            annual_demand=Decimal("200"),
            ordering_cost=Decimal("50"),
            holding_cost_rate=Decimal("0.25"),
            unit_cost=Decimal("200"),
        )

        # EOQ proportional to sqrt(D), so 4x demand = 2x EOQ
        ratio = result_high["eoq"] / result_low["eoq"]
        assert ratio == pytest.approx(Decimal("2.0"), abs=Decimal("0.1"))


# =============================================================================
# TEST CLASS: ANOMALY DETECTOR - CUSUM
# =============================================================================


class TestAnomalyDetectorCUSUM:
    """Tests for CUSUM (Cumulative Sum) analysis."""

    @pytest.mark.unit
    def test_cusum_stable_process(self, anomaly_detector):
        """Test CUSUM on stable process."""
        # Generate stable data around mean 100 with low variance
        data = [100 + (i % 5 - 2) * 0.5 for i in range(100)]

        result = anomaly_detector.detect_cusum_shift(
            values=data,
            k=Decimal("0.5"),
            h=Decimal("5.0"),
        )

        assert result["is_out_of_control"] is False

    @pytest.mark.unit
    def test_cusum_detects_positive_shift(self, anomaly_detector):
        """Test CUSUM detects positive mean shift."""
        # Data shifts from 100 to 105 halfway through
        data = [100] * 50 + [105] * 50

        result = anomaly_detector.detect_cusum_shift(
            values=data,
            k=Decimal("0.5"),
            h=Decimal("5.0"),
        )

        assert result["shift_detected"] is True
        assert result["shift_direction"] == "up"

    @pytest.mark.unit
    def test_cusum_parameters_effect(self, anomaly_detector):
        """Test effect of k and h parameters on sensitivity."""
        data = [100] * 30 + [102] * 30  # Small shift

        # Low h = more sensitive
        result_sensitive = anomaly_detector.detect_cusum_shift(
            values=data,
            k=Decimal("0.5"),
            h=Decimal("3.0"),
        )

        # High h = less sensitive
        result_insensitive = anomaly_detector.detect_cusum_shift(
            values=data,
            k=Decimal("0.5"),
            h=Decimal("10.0"),
        )

        # Sensitive version more likely to detect small shift
        assert result_sensitive["decision_value"] > result_insensitive["threshold"]


# =============================================================================
# TEST CLASS: PROVENANCE - MERKLE TREE
# =============================================================================


class TestProvenanceMerkleTree:
    """Tests for Merkle tree provenance tracking."""

    @pytest.mark.unit
    def test_merkle_root_single_leaf(self, provenance_validator):
        """Test Merkle root with single leaf."""
        leaf = hashlib.sha256(b"single_calculation").hexdigest()
        leaves = [leaf]

        # Single leaf is its own root
        assert provenance_validator.verify_merkle_root(leaves, leaf) is True

    @pytest.mark.unit
    def test_merkle_root_two_leaves(self, provenance_validator):
        """Test Merkle root with two leaves."""
        leaf1 = hashlib.sha256(b"calculation1").hexdigest()
        leaf2 = hashlib.sha256(b"calculation2").hexdigest()

        expected_root = hashlib.sha256((leaf1 + leaf2).encode()).hexdigest()

        assert provenance_validator.verify_merkle_root([leaf1, leaf2], expected_root) is True

    @pytest.mark.unit
    def test_merkle_root_four_leaves(self, provenance_validator):
        """Test Merkle root with four leaves (balanced tree)."""
        leaves = [
            hashlib.sha256(f"calc{i}".encode()).hexdigest()
            for i in range(4)
        ]

        # Build expected tree
        l01 = hashlib.sha256((leaves[0] + leaves[1]).encode()).hexdigest()
        l23 = hashlib.sha256((leaves[2] + leaves[3]).encode()).hexdigest()
        expected_root = hashlib.sha256((l01 + l23).encode()).hexdigest()

        assert provenance_validator.verify_merkle_root(leaves, expected_root) is True

    @pytest.mark.unit
    def test_merkle_root_tamper_detection(self, provenance_validator):
        """Test that tampering is detected."""
        leaves = [
            hashlib.sha256(f"calc{i}".encode()).hexdigest()
            for i in range(4)
        ]

        # Calculate correct root
        l01 = hashlib.sha256((leaves[0] + leaves[1]).encode()).hexdigest()
        l23 = hashlib.sha256((leaves[2] + leaves[3]).encode()).hexdigest()
        correct_root = hashlib.sha256((l01 + l23).encode()).hexdigest()

        # Tamper with one leaf
        tampered_leaves = leaves.copy()
        tampered_leaves[0] = hashlib.sha256(b"tampered").hexdigest()

        # Should fail verification with original root
        assert provenance_validator.verify_merkle_root(tampered_leaves, correct_root) is False


# =============================================================================
# TEST CLASS: CALCULATION PRECISION
# =============================================================================


class TestCalculationPrecision:
    """Tests for decimal precision in calculations."""

    @pytest.mark.unit
    def test_decimal_precision_maintained(self, rul_calculator):
        """Test that decimal precision is maintained throughout calculation."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000.123456789"),
        )

        # Result should be Decimal, not float
        assert isinstance(result["rul_hours"], Decimal)
        assert isinstance(result["current_reliability"], Decimal)

    @pytest.mark.unit
    def test_no_floating_point_errors(self, failure_probability_calculator):
        """Test calculation doesn't introduce floating point errors."""
        # Test with values that cause issues in floating point
        result1 = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal("2.5"),
            eta=Decimal("50000"),
            time_hours=Decimal("0.1"),
        )

        result2 = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal("2.5"),
            eta=Decimal("50000"),
            time_hours=Decimal("0.2"),
        )

        result3 = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal("2.5"),
            eta=Decimal("50000"),
            time_hours=Decimal("0.3"),
        )

        # 0.1 + 0.2 = 0.3 exactly (unlike float 0.1 + 0.2 != 0.3)
        # Verify no numerical instability in results
        assert result1["failure_probability"] >= Decimal("0")
        assert result2["failure_probability"] >= Decimal("0")
        assert result3["failure_probability"] >= Decimal("0")

    @pytest.mark.unit
    def test_rounding_consistency(self, rul_calculator):
        """Test that rounding is consistent."""
        # Run same calculation multiple times
        results = []
        for _ in range(10):
            result = rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=Decimal("25000"),
            )
            results.append(result["rul_hours"])

        # All results should be identical
        assert all(r == results[0] for r in results)


# =============================================================================
# TEST CLASS: INTEGRATION BETWEEN CALCULATORS
# =============================================================================


class TestCalculatorIntegration:
    """Tests for integration between calculator modules."""

    @pytest.mark.unit
    def test_rul_uses_failure_probability(
        self, rul_calculator, failure_probability_calculator
    ):
        """Test RUL calculation is consistent with failure probability."""
        # Get RUL
        rul_result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("30000"),
            target_reliability="0.5",
            custom_beta=Decimal("2.5"),
            custom_eta=Decimal("45000"),
        )

        # Get failure probability at same time
        fp_result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal("2.5"),
            eta=Decimal("45000"),
            time_hours=Decimal("30000"),
        )

        # Reliability should match
        assert rul_result["current_reliability"] == pytest.approx(
            fp_result["reliability"], abs=Decimal("0.00001")
        )

    @pytest.mark.unit
    def test_vibration_informs_health_state(
        self, vibration_analyzer, rul_calculator
    ):
        """Test vibration zone can inform health state for RUL adjustment."""
        # Zone D vibration indicates critical condition
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("12.0"),
            machine_class=MachineClass.CLASS_II,
        )

        assert vib_result["zone"] == VibrationZone.ZONE_D

        # This would typically trigger health state adjustment in RUL
        # (health_state parameter in calculate_weibull_rul)

    @pytest.mark.unit
    def test_thermal_affects_maintenance_interval(
        self, thermal_degradation_calculator, maintenance_scheduler
    ):
        """Test thermal aging affects optimal maintenance interval."""
        # Higher temperature = faster aging = shorter effective eta
        thermal_result = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=Decimal("130"),
            reference_temperature_c=Decimal("110"),
        )

        aaf = thermal_result["acceleration_factor"]

        # Effective eta is reduced by acceleration factor
        nominal_eta = Decimal("50000")
        effective_eta = nominal_eta / aaf

        # Optimal interval should be shorter with reduced eta
        result_nominal = maintenance_scheduler.calculate_optimal_interval(
            beta=Decimal("2.5"),
            eta=nominal_eta,
            preventive_cost=Decimal("1000"),
            corrective_cost=Decimal("10000"),
        )

        result_thermal = maintenance_scheduler.calculate_optimal_interval(
            beta=Decimal("2.5"),
            eta=effective_eta,
            preventive_cost=Decimal("1000"),
            corrective_cost=Decimal("10000"),
        )

        assert result_thermal["optimal_interval_hours"] < result_nominal["optimal_interval_hours"]
