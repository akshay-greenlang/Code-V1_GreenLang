# -*- coding: utf-8 -*-
"""
GL-013 PREDICTMAINT - Unit Tests for Tools Module
Comprehensive test coverage for PredictiveMaintenanceTools.

Tests cover:
- RUL calculation (Weibull and Exponential models)
- Failure probability calculation
- Vibration spectrum analysis (ISO 10816 zones)
- Thermal degradation analysis
- Maintenance schedule optimization
- Spare parts requirement calculation
- Anomaly detection (Z-score method)
- Health index calculation
- Provenance hash generation

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from typing import Dict, List, Any
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
# TEST CLASS: RUL CALCULATION - WEIBULL MODEL
# =============================================================================


class TestCalculateRemainingUsefulLifeWeibull:
    """Tests for Weibull-based RUL calculation."""

    @pytest.mark.unit
    def test_weibull_rul_basic_calculation(self, rul_calculator):
        """Test basic Weibull RUL calculation with known parameters."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("30000"),
            target_reliability="0.5",
        )

        assert result is not None
        assert "rul_hours" in result
        assert "current_reliability" in result
        assert result["rul_hours"] > Decimal("0")
        assert Decimal("0") <= result["current_reliability"] <= Decimal("1")

    @pytest.mark.unit
    def test_weibull_rul_new_equipment(self, rul_calculator):
        """Test RUL for newly installed equipment (0 hours)."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("0"),
            target_reliability="0.5",
        )

        # New equipment should have maximum RUL
        assert result["rul_hours"] > Decimal("0")
        assert result["current_reliability"] == Decimal("1.000000")

    @pytest.mark.unit
    def test_weibull_rul_worn_equipment(self, rul_calculator):
        """Test RUL for heavily worn equipment."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("50000"),  # Beyond eta
            target_reliability="0.5",
        )

        # Worn equipment should have low RUL
        assert result["rul_hours"] >= Decimal("0")
        assert result["current_reliability"] < Decimal("0.5")

    @pytest.mark.unit
    @pytest.mark.parametrize("target_reliability", [
        "0.1", "0.3", "0.5", "0.7", "0.9"
    ])
    def test_weibull_rul_various_reliability_targets(
        self, rul_calculator, target_reliability
    ):
        """Test RUL calculation at various reliability targets."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("20000"),
            target_reliability=target_reliability,
        )

        # Higher reliability target means shorter RUL
        assert result["rul_hours"] >= Decimal("0")

    @pytest.mark.unit
    @pytest.mark.parametrize("equipment_type,expected_mtbf_min", [
        ("pump_centrifugal", 35000),
        ("motor_ac_induction_large", 50000),
        ("gearbox_helical", 60000),
    ])
    def test_weibull_rul_equipment_types(
        self, rul_calculator, equipment_type, expected_mtbf_min
    ):
        """Test RUL for different equipment types uses correct parameters."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type=equipment_type,
            operating_hours=Decimal("0"),
            target_reliability="0.5",
        )

        # RUL for new equipment should be around median life (R=0.5)
        # Median life is approximately 0.7 * eta for beta=2
        assert result["rul_hours"] > Decimal("0")

    @pytest.mark.unit
    def test_weibull_rul_custom_parameters(self, rul_calculator):
        """Test RUL with custom Weibull parameters."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="custom",
            operating_hours=Decimal("10000"),
            target_reliability="0.5",
            custom_beta=Decimal("3.0"),
            custom_eta=Decimal("40000"),
            custom_gamma=Decimal("0"),
        )

        assert result is not None
        assert result["model_used"] == "weibull"

    @pytest.mark.unit
    def test_weibull_rul_confidence_interval(self, rul_calculator):
        """Test that confidence intervals are calculated."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
            confidence_level="90%",
        )

        assert "confidence_lower" in result
        assert "confidence_upper" in result
        assert result["confidence_lower"] < result["rul_hours"]
        assert result["confidence_upper"] > result["rul_hours"]

    @pytest.mark.unit
    def test_weibull_rul_provenance_hash(self, rul_calculator):
        """Test that provenance hash is generated."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
        )

        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64  # SHA-256 hex length

    @pytest.mark.unit
    def test_weibull_rul_invalid_negative_hours(self, rul_calculator):
        """Test that negative operating hours raises error."""
        with pytest.raises(ValueError):
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=Decimal("-1000"),
            )

    @pytest.mark.unit
    def test_weibull_rul_invalid_reliability_range(self, rul_calculator):
        """Test that invalid reliability values raise error."""
        with pytest.raises(ValueError):
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=Decimal("25000"),
                target_reliability="1.5",  # Invalid: > 1
            )


# =============================================================================
# TEST CLASS: RUL CALCULATION - EXPONENTIAL MODEL
# =============================================================================


class TestCalculateRemainingUsefulLifeExponential:
    """Tests for Exponential-based RUL calculation."""

    @pytest.mark.unit
    def test_exponential_rul_basic(self, rul_calculator):
        """Test basic exponential RUL calculation."""
        result = rul_calculator.calculate_exponential_rul(
            failure_rate=Decimal("0.00002"),  # Per hour
            operating_hours=Decimal("10000"),
            target_reliability=Decimal("0.5"),
        )

        assert result is not None
        assert result["rul_hours"] > Decimal("0")
        assert result["model_used"] == "exponential"

    @pytest.mark.unit
    def test_exponential_rul_high_failure_rate(self, rul_calculator):
        """Test RUL with high failure rate."""
        result = rul_calculator.calculate_exponential_rul(
            failure_rate=Decimal("0.001"),
            operating_hours=Decimal("100"),
            target_reliability=Decimal("0.5"),
        )

        # High failure rate should give shorter RUL
        assert result["rul_hours"] < Decimal("1000")

    @pytest.mark.unit
    def test_exponential_rul_memoryless_property(self, rul_calculator):
        """Test exponential memoryless property: RUL independent of age."""
        failure_rate = Decimal("0.00002")

        result_new = rul_calculator.calculate_exponential_rul(
            failure_rate=failure_rate,
            operating_hours=Decimal("0"),
            target_reliability=Decimal("0.5"),
        )

        result_old = rul_calculator.calculate_exponential_rul(
            failure_rate=failure_rate,
            operating_hours=Decimal("50000"),
            target_reliability=Decimal("0.5"),
        )

        # For exponential, RUL should be same regardless of current age
        # (memoryless property)
        # Account for floating point and RUL is time to reach target from current
        assert abs(result_new["rul_hours"] - result_old["rul_hours"]) < Decimal("1")


# =============================================================================
# TEST CLASS: FAILURE PROBABILITY CALCULATION
# =============================================================================


class TestCalculateFailureProbability:
    """Tests for failure probability calculations."""

    @pytest.mark.unit
    def test_weibull_failure_probability_basic(self, failure_probability_calculator):
        """Test basic Weibull failure probability."""
        result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal("2.5"),
            eta=Decimal("50000"),
            time_hours=Decimal("30000"),
        )

        assert result is not None
        assert Decimal("0") <= result["failure_probability"] <= Decimal("1")
        assert result["reliability"] + result["failure_probability"] == pytest.approx(
            Decimal("1"), abs=Decimal("0.000001")
        )

    @pytest.mark.unit
    def test_failure_probability_at_zero(self, failure_probability_calculator):
        """Test failure probability at t=0 is 0."""
        result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal("2.5"),
            eta=Decimal("50000"),
            time_hours=Decimal("0"),
        )

        assert result["failure_probability"] == Decimal("0")
        assert result["reliability"] == Decimal("1")

    @pytest.mark.unit
    def test_failure_probability_at_eta(self, failure_probability_calculator):
        """Test failure probability at t=eta is approximately 0.632."""
        eta = Decimal("50000")
        result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal("1.0"),  # Exponential case
            eta=eta,
            time_hours=eta,
        )

        # At t=eta, F(eta) = 1 - exp(-1) = 0.632...
        expected = Decimal("1") - Decimal(str(math.exp(-1)))
        assert result["failure_probability"] == pytest.approx(
            expected, abs=Decimal("0.001")
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("beta,expected_behavior", [
        ("0.5", "decreasing"),  # Infant mortality
        ("1.0", "constant"),    # Random failures
        ("2.5", "increasing"),  # Wear-out
    ])
    def test_hazard_rate_behavior(
        self, failure_probability_calculator, beta, expected_behavior
    ):
        """Test hazard rate behavior for different beta values."""
        result1 = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal(beta),
            eta=Decimal("50000"),
            time_hours=Decimal("10000"),
        )

        result2 = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal(beta),
            eta=Decimal("50000"),
            time_hours=Decimal("30000"),
        )

        if expected_behavior == "decreasing":
            assert result1["hazard_rate"] > result2["hazard_rate"]
        elif expected_behavior == "constant":
            assert result1["hazard_rate"] == pytest.approx(
                result2["hazard_rate"], abs=Decimal("0.00000001")
            )
        elif expected_behavior == "increasing":
            assert result1["hazard_rate"] < result2["hazard_rate"]

    @pytest.mark.unit
    def test_failure_probability_provenance(self, failure_probability_calculator):
        """Test provenance hash for failure probability calculation."""
        result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal("2.5"),
            eta=Decimal("50000"),
            time_hours=Decimal("30000"),
        )

        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# =============================================================================
# TEST CLASS: VIBRATION SPECTRUM ANALYSIS
# =============================================================================


class TestAnalyzeVibrationSpectrum:
    """Tests for ISO 10816 vibration analysis."""

    @pytest.mark.unit
    def test_vibration_zone_a(self, vibration_analyzer):
        """Test classification in Zone A (Good)."""
        result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("0.8"),
            machine_class=MachineClass.CLASS_II,
        )

        assert result["zone"] == VibrationZone.ZONE_A
        assert result["alarm_level"] == "normal"
        assert "Good" in result["assessment"]

    @pytest.mark.unit
    def test_vibration_zone_b(self, vibration_analyzer):
        """Test classification in Zone B (Acceptable)."""
        result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("2.0"),
            machine_class=MachineClass.CLASS_II,
        )

        assert result["zone"] == VibrationZone.ZONE_B
        assert result["alarm_level"] == "normal"
        assert "Acceptable" in result["assessment"]

    @pytest.mark.unit
    def test_vibration_zone_c(self, vibration_analyzer):
        """Test classification in Zone C (Alert)."""
        result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("5.0"),
            machine_class=MachineClass.CLASS_II,
        )

        assert result["zone"] == VibrationZone.ZONE_C
        assert result["alarm_level"] == "warning"
        assert "Alert" in result["assessment"]

    @pytest.mark.unit
    def test_vibration_zone_d(self, vibration_analyzer):
        """Test classification in Zone D (Danger)."""
        result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("10.0"),
            machine_class=MachineClass.CLASS_II,
        )

        assert result["zone"] == VibrationZone.ZONE_D
        assert result["alarm_level"] == "critical"
        assert "Danger" in result["assessment"]

    @pytest.mark.unit
    @pytest.mark.parametrize("machine_class,zone_a_limit", [
        (MachineClass.CLASS_I, Decimal("0.71")),
        (MachineClass.CLASS_II, Decimal("1.12")),
        (MachineClass.CLASS_III, Decimal("1.8")),
        (MachineClass.CLASS_IV, Decimal("2.8")),
    ])
    def test_vibration_machine_class_limits(
        self, vibration_analyzer, machine_class, zone_a_limit
    ):
        """Test ISO 10816 limits per machine class."""
        # Test value just below Zone A limit
        result = vibration_analyzer.assess_severity(
            velocity_rms=zone_a_limit - Decimal("0.1"),
            machine_class=machine_class,
        )
        assert result["zone"] == VibrationZone.ZONE_A

        # Test value just above Zone A limit
        result = vibration_analyzer.assess_severity(
            velocity_rms=zone_a_limit + Decimal("0.1"),
            machine_class=machine_class,
        )
        assert result["zone"] == VibrationZone.ZONE_B

    @pytest.mark.unit
    def test_vibration_margin_calculation(self, vibration_analyzer):
        """Test margin to next zone calculation."""
        result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("0.5"),
            machine_class=MachineClass.CLASS_II,
        )

        # Margin should be positive and represent headroom to Zone B
        assert result["margin_to_next_zone"] > Decimal("0")
        expected_margin = Decimal("1.12") - Decimal("0.5")  # Zone A limit - value
        assert result["margin_to_next_zone"] == pytest.approx(
            expected_margin, abs=Decimal("0.001")
        )

    @pytest.mark.unit
    def test_vibration_provenance(self, vibration_analyzer):
        """Test provenance hash generation."""
        result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("2.5"),
            machine_class=MachineClass.CLASS_II,
        )

        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_vibration_negative_value_error(self, vibration_analyzer):
        """Test that negative vibration raises error."""
        with pytest.raises(ValueError):
            vibration_analyzer.assess_severity(
                velocity_rms=Decimal("-1.0"),
                machine_class=MachineClass.CLASS_II,
            )


# =============================================================================
# TEST CLASS: BEARING FAULT FREQUENCIES
# =============================================================================


class TestBearingFaultFrequencies:
    """Tests for bearing fault frequency calculations."""

    @pytest.mark.unit
    def test_bearing_frequencies_calculation(self, vibration_analyzer):
        """Test calculation of bearing fault frequencies."""
        result = vibration_analyzer.calculate_bearing_frequencies(
            shaft_speed_rpm=Decimal("1480"),
            num_balls=9,
            ball_diameter=Decimal("7.938"),
            pitch_diameter=Decimal("38.5"),
            contact_angle_deg=Decimal("0"),
        )

        assert "bpfo" in result
        assert "bpfi" in result
        assert "bsf" in result
        assert "ftf" in result

        # BPFI should always be greater than BPFO for ball bearings
        assert result["bpfi"] > result["bpfo"]

        # FTF should be less than shaft speed
        assert result["ftf"] < result["shaft_speed_hz"]

    @pytest.mark.unit
    def test_bearing_frequencies_6205(self, vibration_analyzer, bearing_equipment_data):
        """Test frequencies for SKF 6205 bearing."""
        result = vibration_analyzer.calculate_bearing_frequencies(
            shaft_speed_rpm=bearing_equipment_data["shaft_speed_rpm"],
            num_balls=bearing_equipment_data["number_of_balls"],
            ball_diameter=bearing_equipment_data["ball_diameter_mm"],
            pitch_diameter=bearing_equipment_data["pitch_diameter_mm"],
            contact_angle_deg=bearing_equipment_data["contact_angle_deg"],
        )

        # Known values for 6205 at 1480 RPM
        # BPFO approx 89 Hz, BPFI approx 132 Hz
        assert result["bpfo"] > Decimal("80")
        assert result["bpfo"] < Decimal("100")
        assert result["bpfi"] > Decimal("120")
        assert result["bpfi"] < Decimal("145")


# =============================================================================
# TEST CLASS: THERMAL DEGRADATION ANALYSIS
# =============================================================================


class TestAnalyzeThermalDegradation:
    """Tests for Arrhenius-based thermal degradation analysis."""

    @pytest.mark.unit
    def test_thermal_aging_factor_calculation(self, thermal_degradation_calculator):
        """Test Arrhenius aging acceleration factor."""
        result = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=Decimal("120"),
            reference_temperature_c=Decimal("110"),
        )

        # Higher temperature should accelerate aging (factor > 1)
        assert result["acceleration_factor"] > Decimal("1")

    @pytest.mark.unit
    def test_thermal_aging_at_reference(self, thermal_degradation_calculator):
        """Test aging factor at reference temperature is 1."""
        result = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=Decimal("110"),
            reference_temperature_c=Decimal("110"),
        )

        assert result["acceleration_factor"] == pytest.approx(
            Decimal("1.0"), abs=Decimal("0.001")
        )

    @pytest.mark.unit
    def test_thermal_aging_below_reference(self, thermal_degradation_calculator):
        """Test aging factor below reference temperature is < 1."""
        result = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=Decimal("100"),
            reference_temperature_c=Decimal("110"),
        )

        assert result["acceleration_factor"] < Decimal("1")

    @pytest.mark.unit
    def test_thermal_life_calculation(self, thermal_degradation_calculator):
        """Test thermal life calculation."""
        result = thermal_degradation_calculator.calculate_thermal_life(
            hot_spot_temperature_c=Decimal("120"),
            operating_hours=Decimal("50000"),
            reference_life_hours=Decimal("180000"),
        )

        assert "remaining_life_hours" in result
        assert "life_consumed_percent" in result
        assert result["remaining_life_hours"] >= Decimal("0")
        assert result["life_consumed_percent"] > Decimal("0")

    @pytest.mark.unit
    def test_thermal_life_10_degree_rule(self, thermal_degradation_calculator):
        """Test 10-degree rule of thumb: life halves for 10C increase."""
        result_low = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=Decimal("100"),
            reference_temperature_c=Decimal("100"),
        )

        result_high = thermal_degradation_calculator.calculate_arrhenius_aging_factor(
            operating_temperature_c=Decimal("110"),
            reference_temperature_c=Decimal("100"),
        )

        # Aging factor should approximately double for 10C increase
        # (actual depends on activation energy)
        ratio = result_high["acceleration_factor"] / result_low["acceleration_factor"]
        assert ratio > Decimal("1.5")
        assert ratio < Decimal("3.0")  # Approximately 2x


# =============================================================================
# TEST CLASS: MAINTENANCE SCHEDULE OPTIMIZATION
# =============================================================================


class TestOptimizeMaintenanceSchedule:
    """Tests for maintenance schedule optimization."""

    @pytest.mark.unit
    def test_optimal_interval_calculation(self, maintenance_scheduler):
        """Test optimal maintenance interval calculation."""
        result = maintenance_scheduler.calculate_optimal_interval(
            beta=Decimal("2.5"),
            eta=Decimal("50000"),
            preventive_cost=Decimal("1000"),
            corrective_cost=Decimal("10000"),
        )

        assert "optimal_interval_hours" in result
        assert result["optimal_interval_hours"] > Decimal("0")
        assert result["optimal_interval_hours"] < Decimal("50000")

    @pytest.mark.unit
    def test_run_to_failure_for_beta_le_1(self, maintenance_scheduler):
        """Test run-to-failure strategy when beta <= 1."""
        result = maintenance_scheduler.calculate_optimal_interval(
            beta=Decimal("0.8"),
            eta=Decimal("50000"),
            preventive_cost=Decimal("1000"),
            corrective_cost=Decimal("10000"),
        )

        # For beta <= 1 (decreasing or constant hazard), run-to-failure is optimal
        assert result["strategy"] == "run_to_failure"

    @pytest.mark.unit
    def test_higher_cost_ratio_shorter_interval(self, maintenance_scheduler):
        """Test that higher corrective/preventive cost ratio gives shorter interval."""
        result_low = maintenance_scheduler.calculate_optimal_interval(
            beta=Decimal("2.5"),
            eta=Decimal("50000"),
            preventive_cost=Decimal("1000"),
            corrective_cost=Decimal("2000"),
        )

        result_high = maintenance_scheduler.calculate_optimal_interval(
            beta=Decimal("2.5"),
            eta=Decimal("50000"),
            preventive_cost=Decimal("1000"),
            corrective_cost=Decimal("20000"),
        )

        # Higher cost ratio should give shorter interval
        assert result_high["optimal_interval_hours"] < result_low["optimal_interval_hours"]


# =============================================================================
# TEST CLASS: SPARE PARTS REQUIREMENT CALCULATION
# =============================================================================


class TestCalculateSparePartsRequirement:
    """Tests for spare parts inventory optimization."""

    @pytest.mark.unit
    def test_eoq_calculation(self, spare_parts_calculator):
        """Test Economic Order Quantity calculation."""
        result = spare_parts_calculator.calculate_eoq(
            annual_demand=Decimal("100"),
            ordering_cost=Decimal("50"),
            holding_cost_rate=Decimal("0.25"),
            unit_cost=Decimal("200"),
        )

        assert "eoq" in result
        assert result["eoq"] > Decimal("0")

        # Verify EOQ formula: sqrt(2*D*S / H)
        expected_eoq = Decimal(str(
            math.sqrt(2 * 100 * 50 / (0.25 * 200))
        ))
        assert result["eoq"] == pytest.approx(
            expected_eoq.quantize(Decimal("1")), abs=Decimal("1")
        )

    @pytest.mark.unit
    def test_eoq_higher_demand(self, spare_parts_calculator):
        """Test EOQ increases with higher demand."""
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

        assert result_high["eoq"] > result_low["eoq"]

    @pytest.mark.unit
    def test_safety_stock_calculation(self, spare_parts_calculator):
        """Test safety stock calculation."""
        try:
            result = spare_parts_calculator.calculate_safety_stock(
                demand_std_dev=Decimal("5"),
                lead_time_days=Decimal("14"),
                service_level=Decimal("0.95"),
            )

            assert "safety_stock" in result
            assert result["safety_stock"] > Decimal("0")
        except ImportError:
            pytest.skip("scipy not available for safety stock calculation")


# =============================================================================
# TEST CLASS: ANOMALY DETECTION - Z-SCORE
# =============================================================================


class TestDetectAnomaliesZScore:
    """Tests for Z-score anomaly detection."""

    @pytest.mark.unit
    def test_normal_value_not_anomaly(self, anomaly_detector):
        """Test that normal values are not flagged as anomalies."""
        historical = [100, 101, 99, 102, 100, 98, 101, 103, 99, 100]
        result = anomaly_detector.detect_univariate_anomaly(
            value=Decimal("101"),
            historical_data=historical,
            threshold_sigma="3.0",
        )

        assert result["is_anomaly"] is False
        assert result["z_score"] < Decimal("3.0")

    @pytest.mark.unit
    def test_outlier_detected(self, anomaly_detector):
        """Test that outliers are detected."""
        historical = [100, 101, 99, 102, 100, 98, 101, 103, 99, 100]
        result = anomaly_detector.detect_univariate_anomaly(
            value=Decimal("150"),  # Well outside normal range
            historical_data=historical,
            threshold_sigma="3.0",
        )

        assert result["is_anomaly"] is True
        assert result["z_score"] > Decimal("3.0")

    @pytest.mark.unit
    def test_anomaly_score_normalization(self, anomaly_detector):
        """Test anomaly score is normalized 0-1."""
        historical = [100, 101, 99, 102, 100]
        result = anomaly_detector.detect_univariate_anomaly(
            value=Decimal("200"),
            historical_data=historical,
            threshold_sigma="3.0",
        )

        assert Decimal("0") <= result["anomaly_score"] <= Decimal("1")

    @pytest.mark.unit
    @pytest.mark.parametrize("threshold", ["2.0", "3.0", "4.0"])
    def test_different_thresholds(self, anomaly_detector, threshold):
        """Test anomaly detection with different thresholds."""
        historical = [100] * 100  # Perfectly consistent data
        result = anomaly_detector.detect_univariate_anomaly(
            value=Decimal("100"),
            historical_data=historical,
            threshold_sigma=threshold,
        )

        # Same value as historical mean should not be anomaly
        assert result["is_anomaly"] is False

    @pytest.mark.unit
    def test_minimum_data_requirement(self, anomaly_detector):
        """Test error with insufficient historical data."""
        with pytest.raises(ValueError):
            anomaly_detector.detect_univariate_anomaly(
                value=Decimal("100"),
                historical_data=[100, 101],  # Only 2 points
                threshold_sigma="3.0",
            )

    @pytest.mark.unit
    def test_anomaly_provenance(self, anomaly_detector):
        """Test provenance hash for anomaly detection."""
        historical = [100, 101, 99, 102, 100]
        result = anomaly_detector.detect_univariate_anomaly(
            value=Decimal("105"),
            historical_data=historical,
        )

        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# =============================================================================
# TEST CLASS: CUSUM DETECTION
# =============================================================================


class TestCUSUMDetection:
    """Tests for CUSUM trend detection."""

    @pytest.mark.unit
    def test_cusum_stable_process(self, anomaly_detector):
        """Test CUSUM on stable process (no shift)."""
        # Generate stable data around mean 100
        data = [100 + i % 3 - 1 for i in range(50)]  # Minor variation

        result = anomaly_detector.detect_cusum_shift(
            values=data,
            k=Decimal("0.5"),
            h=Decimal("5.0"),
        )

        assert result["is_out_of_control"] is False
        assert result["shift_detected"] is False

    @pytest.mark.unit
    def test_cusum_detects_shift(self, anomaly_detector):
        """Test CUSUM detects mean shift."""
        # Generate data with shift: 100 for first half, 110 for second half
        data = [100] * 25 + [110] * 25

        result = anomaly_detector.detect_cusum_shift(
            values=data,
            k=Decimal("0.5"),
            h=Decimal("5.0"),
        )

        # Should detect the shift
        assert result["shift_detected"] is True


# =============================================================================
# TEST CLASS: HEALTH INDEX CALCULATION
# =============================================================================


class TestCalculateHealthIndex:
    """Tests for overall health index calculation."""

    @pytest.mark.unit
    def test_health_index_healthy_equipment(
        self, vibration_analyzer, pump_equipment_data
    ):
        """Test health index for healthy equipment."""
        # Simulate calculating overall health from multiple factors
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("1.0"),
            machine_class=MachineClass.CLASS_II,
        )

        # Zone A should give high health score
        assert vib_result["zone"] == VibrationZone.ZONE_A

    @pytest.mark.unit
    def test_health_index_degraded_equipment(
        self, vibration_analyzer, degraded_vibration_data
    ):
        """Test health index for degraded equipment."""
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=degraded_vibration_data["velocity_rms_mm_s"],
            machine_class=MachineClass.CLASS_II,
        )

        # Zone B-C should indicate degradation
        assert vib_result["zone"] in [VibrationZone.ZONE_B, VibrationZone.ZONE_C]


# =============================================================================
# TEST CLASS: PROVENANCE HASH GENERATION
# =============================================================================


class TestProvenanceHashGeneration:
    """Tests for provenance hash generation and verification."""

    @pytest.mark.unit
    def test_hash_is_sha256(self, rul_calculator):
        """Test that hash is valid SHA-256 format."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
        )

        # SHA-256 is 64 hex characters
        assert len(result["provenance_hash"]) == 64
        # Should be valid hex
        int(result["provenance_hash"], 16)

    @pytest.mark.unit
    def test_same_input_same_hash(self, rul_calculator):
        """Test deterministic hash generation."""
        result1 = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
        )

        result2 = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
        )

        assert result1["provenance_hash"] == result2["provenance_hash"]

    @pytest.mark.unit
    def test_different_input_different_hash(self, rul_calculator):
        """Test that different inputs produce different hashes."""
        result1 = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
        )

        result2 = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25001"),  # Slightly different
        )

        assert result1["provenance_hash"] != result2["provenance_hash"]

    @pytest.mark.unit
    def test_provenance_validator(self, provenance_validator):
        """Test provenance validation utility."""
        data = {"test": "value", "number": 42}
        hash1 = provenance_validator.compute_hash(data)

        assert provenance_validator.validate_hash(data, hash1) is True
        assert provenance_validator.validate_hash(data, "wrong_hash") is False

    @pytest.mark.unit
    def test_merkle_tree_verification(self, provenance_validator):
        """Test Merkle tree root verification."""
        leaves = [
            hashlib.sha256(b"leaf1").hexdigest(),
            hashlib.sha256(b"leaf2").hexdigest(),
            hashlib.sha256(b"leaf3").hexdigest(),
            hashlib.sha256(b"leaf4").hexdigest(),
        ]

        # Calculate expected root
        l01 = hashlib.sha256((leaves[0] + leaves[1]).encode()).hexdigest()
        l23 = hashlib.sha256((leaves[2] + leaves[3]).encode()).hexdigest()
        root = hashlib.sha256((l01 + l23).encode()).hexdigest()

        assert provenance_validator.verify_merkle_root(leaves, root) is True


# =============================================================================
# TEST CLASS: EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================


class TestEdgeCasesAndBoundaries:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_zero_operating_hours(self, rul_calculator):
        """Test calculation at exactly 0 operating hours."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("0"),
        )

        assert result["current_reliability"] == Decimal("1.000000")

    @pytest.mark.unit
    def test_very_large_operating_hours(self, rul_calculator):
        """Test calculation with very large operating hours."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("500000"),  # Way past design life
        )

        assert result["rul_hours"] >= Decimal("0")
        assert result["current_reliability"] < Decimal("0.001")

    @pytest.mark.unit
    def test_exact_zone_boundary(self, vibration_analyzer):
        """Test classification at exact zone boundary."""
        # At exactly Zone A limit
        zone_a_limit = Decimal("1.12")
        result = vibration_analyzer.assess_severity(
            velocity_rms=zone_a_limit,
            machine_class=MachineClass.CLASS_II,
        )

        # At boundary should be Zone A (using <=)
        assert result["zone"] == VibrationZone.ZONE_A

    @pytest.mark.unit
    def test_very_small_vibration(self, vibration_analyzer):
        """Test classification with very small vibration value."""
        result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("0.001"),
            machine_class=MachineClass.CLASS_II,
        )

        assert result["zone"] == VibrationZone.ZONE_A

    @pytest.mark.unit
    def test_decimal_precision_maintained(self, rul_calculator):
        """Test that decimal precision is maintained in calculations."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000.123456789"),
        )

        # Should not raise and should handle precision
        assert result is not None

    @pytest.mark.unit
    def test_string_to_decimal_conversion(self, rul_calculator):
        """Test that string inputs are converted to Decimal."""
        result_str = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours="25000",
        )

        result_decimal = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
        )

        assert result_str["rul_hours"] == result_decimal["rul_hours"]


# =============================================================================
# TEST CLASS: ERROR HANDLING
# =============================================================================


class TestErrorHandling:
    """Tests for proper error handling."""

    @pytest.mark.unit
    def test_invalid_equipment_type_handled(self, rul_calculator):
        """Test handling of unknown equipment type."""
        # Should use default parameters, not raise
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="unknown_equipment",
            operating_hours=Decimal("25000"),
        )

        assert result is not None

    @pytest.mark.unit
    def test_none_input_handling(self, rul_calculator):
        """Test handling of None inputs."""
        with pytest.raises((TypeError, ValueError)):
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=None,
            )

    @pytest.mark.unit
    def test_empty_historical_data(self, anomaly_detector):
        """Test handling of empty historical data."""
        with pytest.raises(ValueError):
            anomaly_detector.detect_univariate_anomaly(
                value=Decimal("100"),
                historical_data=[],
            )

    @pytest.mark.unit
    def test_invalid_beta_parameter(self, failure_probability_calculator):
        """Test handling of invalid beta (must be positive)."""
        with pytest.raises(ValueError):
            failure_probability_calculator.calculate_weibull_failure_probability(
                beta=Decimal("0"),
                eta=Decimal("50000"),
                time_hours=Decimal("25000"),
            )

    @pytest.mark.unit
    def test_invalid_eta_parameter(self, failure_probability_calculator):
        """Test handling of invalid eta (must be positive)."""
        with pytest.raises(ValueError):
            failure_probability_calculator.calculate_weibull_failure_probability(
                beta=Decimal("2.5"),
                eta=Decimal("-50000"),
                time_hours=Decimal("25000"),
            )
