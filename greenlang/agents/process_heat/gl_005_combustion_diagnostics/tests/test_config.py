# -*- coding: utf-8 -*-
"""
GL-005 Configuration Tests
==========================

Unit tests for GL-005 configuration module.
Tests all configuration schemas, validation, and factory functions.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    GL005Config,
    CQIConfig,
    CQIWeights,
    CQIThresholds,
    AnomalyDetectionConfig,
    SPCConfig,
    MLAnomalyConfig,
    FuelCharacterizationConfig,
    MaintenanceAdvisoryConfig,
    FoulingPredictionConfig,
    BurnerWearConfig,
    TrendingConfig,
    ComplianceConfig,
    DiagnosticMode,
    FuelCategory,
    ComplianceFramework,
    MaintenancePriority,
    AnomalyType,
    create_default_config,
    create_high_precision_config,
    create_compliance_focused_config,
)


class TestCQIWeights:
    """Tests for CQI weight configuration."""

    def test_default_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        weights = CQIWeights()
        total = (
            weights.oxygen +
            weights.carbon_monoxide +
            weights.carbon_dioxide +
            weights.nox +
            weights.combustibles
        )
        assert abs(total - 1.0) < 0.001

    def test_custom_weights_validation(self):
        """Test that custom weights must sum to 1.0."""
        # Valid weights
        valid_weights = CQIWeights(
            oxygen=0.20,
            carbon_monoxide=0.30,
            carbon_dioxide=0.20,
            nox=0.15,
            combustibles=0.15,
        )
        assert valid_weights.oxygen == 0.20

    def test_invalid_weights_raise_error(self):
        """Test that invalid weights raise validation error."""
        with pytest.raises(ValidationError):
            CQIWeights(
                oxygen=0.30,
                carbon_monoxide=0.30,
                carbon_dioxide=0.20,
                nox=0.15,
                combustibles=0.15,  # Sum = 1.1, should fail
            )

    def test_weight_bounds(self):
        """Test that weights are bounded between 0 and 1."""
        with pytest.raises(ValidationError):
            CQIWeights(oxygen=-0.1)

        with pytest.raises(ValidationError):
            CQIWeights(oxygen=1.5)


class TestCQIThresholds:
    """Tests for CQI threshold configuration."""

    def test_default_thresholds(self, default_cqi_thresholds):
        """Test default threshold values."""
        t = default_cqi_thresholds
        assert t.o2_optimal_min == 2.0
        assert t.o2_optimal_max == 4.0
        assert t.co_excellent == 50.0
        assert t.cqi_excellent == 90.0

    def test_threshold_ordering(self, default_cqi_thresholds):
        """Test that thresholds are properly ordered."""
        t = default_cqi_thresholds
        # O2 thresholds should increase
        assert t.o2_optimal_min < t.o2_optimal_max < t.o2_acceptable_max < t.o2_warning_max
        # CO thresholds should increase
        assert t.co_excellent < t.co_good < t.co_acceptable < t.co_warning
        # CQI thresholds should decrease
        assert t.cqi_excellent > t.cqi_good > t.cqi_acceptable > t.cqi_poor

    def test_threshold_bounds(self):
        """Test threshold value bounds."""
        # O2 must be between 0 and 21
        with pytest.raises(ValidationError):
            CQIThresholds(o2_optimal_min=-1.0)

        with pytest.raises(ValidationError):
            CQIThresholds(o2_optimal_max=25.0)


class TestCQIConfig:
    """Tests for complete CQI configuration."""

    def test_default_config(self, default_cqi_config):
        """Test default CQI configuration."""
        assert default_cqi_config.scoring_method == "weighted_linear"
        assert default_cqi_config.o2_reference_pct == 3.0
        assert default_cqi_config.calculation_interval_s == 60.0

    def test_custom_scoring_method(self):
        """Test custom scoring method configuration."""
        config = CQIConfig(scoring_method="weighted_sigmoid")
        assert config.scoring_method == "weighted_sigmoid"

    def test_calculation_interval_bounds(self):
        """Test calculation interval bounds."""
        # Minimum 1 second
        with pytest.raises(ValidationError):
            CQIConfig(calculation_interval_s=0.5)

        # Maximum 3600 seconds
        with pytest.raises(ValidationError):
            CQIConfig(calculation_interval_s=5000.0)


class TestSPCConfig:
    """Tests for SPC configuration."""

    def test_default_spc_config(self, default_spc_config):
        """Test default SPC configuration."""
        assert default_spc_config.sigma_warning == 2.0
        assert default_spc_config.sigma_control == 3.0
        assert default_spc_config.enable_run_rules is True

    def test_sigma_bounds(self):
        """Test sigma multiplier bounds."""
        # Warning sigma must be >= 1.0
        with pytest.raises(ValidationError):
            SPCConfig(sigma_warning=0.5)

        # Control sigma must be >= 2.0
        with pytest.raises(ValidationError):
            SPCConfig(sigma_control=1.5)

    def test_consecutive_point_rules(self, default_spc_config):
        """Test consecutive point rule settings."""
        assert default_spc_config.consecutive_one_side == 7
        assert default_spc_config.consecutive_trending == 6

    def test_window_sizes(self):
        """Test window size constraints."""
        # Baseline window minimum
        with pytest.raises(ValidationError):
            SPCConfig(baseline_window_size=10)

        # Moving window minimum
        with pytest.raises(ValidationError):
            SPCConfig(moving_window_size=2)


class TestMLAnomalyConfig:
    """Tests for ML anomaly detection configuration."""

    def test_default_ml_config(self, default_ml_config):
        """Test default ML configuration."""
        assert default_ml_config.enabled is True
        assert default_ml_config.contamination == 0.05
        assert default_ml_config.n_estimators == 100

    def test_contamination_bounds(self):
        """Test contamination parameter bounds."""
        with pytest.raises(ValidationError):
            MLAnomalyConfig(contamination=0.0)

        with pytest.raises(ValidationError):
            MLAnomalyConfig(contamination=0.6)

    def test_anomaly_threshold_bounds(self):
        """Test anomaly threshold bounds."""
        with pytest.raises(ValidationError):
            MLAnomalyConfig(anomaly_threshold=0.3)

        with pytest.raises(ValidationError):
            MLAnomalyConfig(anomaly_threshold=1.5)


class TestAnomalyDetectionConfig:
    """Tests for complete anomaly detection configuration."""

    def test_default_config(self, default_anomaly_config):
        """Test default anomaly detection configuration."""
        assert "spc" in default_anomaly_config.detection_modes
        assert "ml" in default_anomaly_config.detection_modes
        assert "rule_based" in default_anomaly_config.detection_modes

    def test_alert_cooldown(self, default_anomaly_config):
        """Test alert cooldown settings."""
        assert default_anomaly_config.alert_cooldown_s >= 60
        assert default_anomaly_config.alert_cooldown_s <= 3600

    def test_escalation_settings(self, default_anomaly_config):
        """Test escalation settings."""
        assert default_anomaly_config.escalation_window_s >= 300
        assert default_anomaly_config.escalation_threshold_count >= 2


class TestFuelCharacterizationConfig:
    """Tests for fuel characterization configuration."""

    def test_default_config(self, default_fuel_config):
        """Test default fuel characterization configuration."""
        assert default_fuel_config.enabled is True
        assert default_fuel_config.carbon_balance_method == "flue_gas"
        assert default_fuel_config.detect_fuel_blends is True

    def test_tolerance_bounds(self):
        """Test stoichiometric tolerance bounds."""
        with pytest.raises(ValidationError):
            FuelCharacterizationConfig(stoichiometric_tolerance=0.0001)

        with pytest.raises(ValidationError):
            FuelCharacterizationConfig(stoichiometric_tolerance=0.2)


class TestFoulingPredictionConfig:
    """Tests for fouling prediction configuration."""

    def test_default_config(self, default_fouling_config):
        """Test default fouling prediction configuration."""
        assert default_fouling_config.enabled is True
        assert default_fouling_config.prediction_horizon_days == 30
        assert default_fouling_config.fouling_warning_pct == 5.0
        assert default_fouling_config.fouling_critical_pct == 10.0

    def test_prediction_horizon_bounds(self):
        """Test prediction horizon bounds."""
        with pytest.raises(ValidationError):
            FoulingPredictionConfig(prediction_horizon_days=3)

        with pytest.raises(ValidationError):
            FoulingPredictionConfig(prediction_horizon_days=200)


class TestBurnerWearConfig:
    """Tests for burner wear configuration."""

    def test_default_config(self, default_burner_wear_config):
        """Test default burner wear configuration."""
        assert default_burner_wear_config.enabled is True
        assert default_burner_wear_config.prediction_horizon_days == 90
        assert default_burner_wear_config.expected_burner_life_hours == 20000

    def test_burner_life_bounds(self):
        """Test burner life bounds."""
        with pytest.raises(ValidationError):
            BurnerWearConfig(expected_burner_life_hours=1000)


class TestMaintenanceAdvisoryConfig:
    """Tests for maintenance advisory configuration."""

    def test_default_config(self, default_maintenance_config):
        """Test default maintenance advisory configuration."""
        assert default_maintenance_config.cmms_enabled is False
        assert default_maintenance_config.default_priority == MaintenancePriority.MEDIUM
        assert default_maintenance_config.auto_create_work_orders is False

    def test_cmms_configuration(self):
        """Test CMMS configuration options."""
        config = MaintenanceAdvisoryConfig(
            cmms_enabled=True,
            cmms_api_url="https://cmms.example.com/api",
            cmms_system="sap_pm",
        )
        assert config.cmms_enabled is True
        assert config.cmms_system == "sap_pm"


class TestTrendingConfig:
    """Tests for trending configuration."""

    def test_default_config(self, default_trending_config):
        """Test default trending configuration."""
        assert default_trending_config.enabled is True
        assert default_trending_config.raw_data_retention_days == 90
        assert default_trending_config.aggregated_data_retention_days == 730
        assert default_trending_config.detect_seasonality is True

    def test_aggregation_intervals(self, default_trending_config):
        """Test aggregation interval settings."""
        assert "1h" in default_trending_config.aggregation_intervals
        assert "1d" in default_trending_config.aggregation_intervals
        assert "1w" in default_trending_config.aggregation_intervals

    def test_retention_bounds(self):
        """Test data retention bounds."""
        with pytest.raises(ValidationError):
            TrendingConfig(raw_data_retention_days=3)


class TestComplianceConfig:
    """Tests for compliance configuration."""

    def test_default_config(self, default_compliance_config):
        """Test default compliance configuration."""
        assert ComplianceFramework.EPA_40CFR60 in default_compliance_config.frameworks
        assert default_compliance_config.track_exceedances is True

    def test_multiple_frameworks(self):
        """Test multiple compliance frameworks."""
        config = ComplianceConfig(
            frameworks=[
                ComplianceFramework.EPA_40CFR60,
                ComplianceFramework.EU_IED,
                ComplianceFramework.ISO_50001,
            ]
        )
        assert len(config.frameworks) == 3


class TestGL005Config:
    """Tests for complete GL-005 agent configuration."""

    def test_required_fields(self):
        """Test that required fields must be provided."""
        with pytest.raises(ValidationError):
            GL005Config()  # Missing agent_id and equipment_id

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = GL005Config(
            agent_id="GL005-TEST",
            equipment_id="BLR-001",
        )
        assert config.agent_id == "GL005-TEST"
        assert config.equipment_id == "BLR-001"

    def test_full_config(self, gl005_config):
        """Test full configuration."""
        assert gl005_config.agent_id == "GL005-TEST-001"
        assert gl005_config.equipment_id == "BLR-TEST-001"
        assert gl005_config.mode == DiagnosticMode.REAL_TIME
        assert gl005_config.primary_fuel == FuelCategory.NATURAL_GAS

    def test_default_subconfigs(self, gl005_config):
        """Test that sub-configurations have defaults."""
        assert gl005_config.cqi is not None
        assert gl005_config.anomaly_detection is not None
        assert gl005_config.fuel_characterization is not None
        assert gl005_config.maintenance is not None
        assert gl005_config.trending is not None
        assert gl005_config.compliance is not None

    def test_audit_settings(self, gl005_config):
        """Test audit and provenance settings."""
        assert gl005_config.enable_audit is True
        assert gl005_config.enable_provenance is True

    def test_performance_settings(self, gl005_config):
        """Test performance settings."""
        assert gl005_config.max_concurrent_analyses >= 1
        assert gl005_config.analysis_timeout_s >= 5.0


class TestFactoryFunctions:
    """Tests for configuration factory functions."""

    def test_create_default_config(self):
        """Test default configuration factory."""
        config = create_default_config(
            agent_id="GL005-FACTORY",
            equipment_id="BLR-FACTORY",
        )
        assert config.agent_id == "GL005-FACTORY"
        assert config.equipment_id == "BLR-FACTORY"
        assert config.primary_fuel == FuelCategory.NATURAL_GAS

    def test_create_default_config_with_fuel(self):
        """Test default configuration with custom fuel."""
        config = create_default_config(
            agent_id="GL005-FACTORY",
            equipment_id="BLR-FACTORY",
            fuel_type=FuelCategory.FUEL_OIL_2,
        )
        assert config.primary_fuel == FuelCategory.FUEL_OIL_2

    def test_create_high_precision_config(self):
        """Test high-precision configuration factory."""
        config = create_high_precision_config(
            agent_id="GL005-HP",
            equipment_id="BLR-HP",
        )
        # Verify tighter thresholds
        assert config.cqi.thresholds.co_excellent == 25.0
        assert config.cqi.calculation_interval_s == 30.0
        # Verify more sensitive SPC
        assert config.anomaly_detection.spc.sigma_warning == 1.5
        # Verify faster polling
        assert config.data_poll_interval_s == 2.0

    def test_create_compliance_focused_config(self):
        """Test compliance-focused configuration factory."""
        config = create_compliance_focused_config(
            agent_id="GL005-COMP",
            equipment_id="BLR-COMP",
            frameworks=[ComplianceFramework.EPA_40CFR60, ComplianceFramework.EU_IED],
        )
        assert len(config.compliance.frameworks) == 2
        assert config.compliance.track_exceedances is True
        assert config.trending.aggregated_data_retention_days == 3650


class TestEnums:
    """Tests for configuration enums."""

    def test_diagnostic_mode_values(self):
        """Test diagnostic mode enum values."""
        assert DiagnosticMode.REAL_TIME.value == "real_time"
        assert DiagnosticMode.BATCH.value == "batch"
        assert DiagnosticMode.ON_DEMAND.value == "on_demand"
        assert DiagnosticMode.SCHEDULED.value == "scheduled"

    def test_compliance_framework_values(self):
        """Test compliance framework enum values."""
        assert ComplianceFramework.EPA_40CFR60.value == "epa_40cfr60"
        assert ComplianceFramework.EU_IED.value == "eu_ied"
        assert ComplianceFramework.ISO_50001.value == "iso_50001"

    def test_fuel_category_values(self):
        """Test fuel category enum values."""
        assert FuelCategory.NATURAL_GAS.value == "natural_gas"
        assert FuelCategory.FUEL_OIL_2.value == "fuel_oil_2"
        assert FuelCategory.HYDROGEN.value == "hydrogen"

    def test_maintenance_priority_values(self):
        """Test maintenance priority enum values."""
        assert MaintenancePriority.CRITICAL.value == "critical"
        assert MaintenancePriority.HIGH.value == "high"
        assert MaintenancePriority.ROUTINE.value == "routine"

    def test_anomaly_type_values(self):
        """Test anomaly type enum values."""
        assert AnomalyType.EXCESS_OXYGEN.value == "excess_oxygen"
        assert AnomalyType.HIGH_CO.value == "high_co"
        assert AnomalyType.FOULING_DETECTED.value == "fouling_detected"
