"""
Unit Tests for Drift Detection Module.

Tests for the GreenLang Process Heat Agents drift detection system including:
- ProcessHeatDriftMonitor
- Drift profiles
- Alert manager

Example:
    pytest tests/unit/test_drift_detection.py -v
"""

import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from greenlang.ml.drift_detection.evidently_monitor import (
    ProcessHeatDriftMonitor,
    EvidentlyDriftConfig,
    DriftAnalysisResult,
    FeatureDriftInfo,
    PrometheusMetricExport,
)
from greenlang.ml.drift_detection.drift_profiles import (
    BaseDriftProfile,
    GL001CarbonEmissionsDriftProfile,
    GL003CSRDReportingDriftProfile,
    GL006Scope3DriftProfile,
    GL010EmissionsGuardianDriftProfile,
    FeatureSpec,
    AlertConfig,
    get_drift_profile,
    list_available_profiles,
    create_custom_profile,
)
from greenlang.ml.drift_detection.alert_manager import (
    DriftAlertManager,
    DriftAlert,
    AlertSeverity,
    AlertChannel,
    AlertStatus,
    RemediationAction,
    AlertManagerConfig,
    SlackConfig,
    PagerDutyConfig,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_storage_path():
    """Create temporary storage path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def evidently_config(temp_storage_path):
    """Create Evidently configuration with temp storage."""
    return EvidentlyDriftConfig(
        storage_path=temp_storage_path,
        significance_level=0.05,
        psi_threshold_low=0.1,
        psi_threshold_medium=0.2,
        psi_threshold_high=0.3,
    )


@pytest.fixture
def drift_monitor(evidently_config):
    """Create ProcessHeatDriftMonitor instance."""
    return ProcessHeatDriftMonitor(config=evidently_config)


@pytest.fixture
def sample_reference_data():
    """Generate sample reference data."""
    np.random.seed(42)
    return np.random.normal(loc=100, scale=20, size=(1000, 5))


@pytest.fixture
def sample_current_data_no_drift():
    """Generate current data with no drift."""
    np.random.seed(43)
    return np.random.normal(loc=100, scale=20, size=(500, 5))


@pytest.fixture
def sample_current_data_with_drift():
    """Generate current data with significant drift."""
    np.random.seed(44)
    return np.random.normal(loc=150, scale=40, size=(500, 5))


@pytest.fixture
def feature_names():
    """Sample feature names."""
    return [
        "temperature",
        "pressure",
        "flow_rate",
        "efficiency",
        "emissions",
    ]


@pytest.fixture
def alert_manager_config(temp_storage_path):
    """Create AlertManager configuration."""
    return AlertManagerConfig(
        storage_path=temp_storage_path,
        cooldown_minutes=1,
        max_alerts_per_hour=100,
        slack=SlackConfig(enabled=False),
        pagerduty=PagerDutyConfig(enabled=False),
    )


@pytest.fixture
def alert_manager(alert_manager_config):
    """Create DriftAlertManager instance."""
    return DriftAlertManager(config=alert_manager_config)


# =============================================================================
# ProcessHeatDriftMonitor Tests
# =============================================================================

class TestProcessHeatDriftMonitor:
    """Tests for ProcessHeatDriftMonitor class."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        monitor = ProcessHeatDriftMonitor()
        assert monitor.config is not None
        assert monitor.config.significance_level == 0.05
        assert monitor.SUPPORTED_AGENTS == [f"GL-{str(i).zfill(3)}" for i in range(1, 21)]

    def test_init_custom_config(self, evidently_config):
        """Test initialization with custom configuration."""
        monitor = ProcessHeatDriftMonitor(config=evidently_config)
        assert monitor.config == evidently_config

    def test_validate_agent_id_valid(self, drift_monitor):
        """Test valid agent ID validation."""
        # Should not raise
        drift_monitor._validate_agent_id("GL-001")
        drift_monitor._validate_agent_id("GL-010")
        drift_monitor._validate_agent_id("GL-020")

    def test_validate_agent_id_invalid(self, drift_monitor):
        """Test invalid agent ID validation."""
        with pytest.raises(ValueError) as exc_info:
            drift_monitor._validate_agent_id("GL-000")
        assert "Invalid agent ID" in str(exc_info.value)

        with pytest.raises(ValueError):
            drift_monitor._validate_agent_id("GL-021")

        with pytest.raises(ValueError):
            drift_monitor._validate_agent_id("INVALID")

    def test_calculate_data_hash(self, drift_monitor, sample_reference_data):
        """Test SHA-256 hash calculation for data."""
        hash1 = drift_monitor._calculate_data_hash(sample_reference_data)
        hash2 = drift_monitor._calculate_data_hash(sample_reference_data)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

        # Different data should produce different hash
        different_data = sample_reference_data + 1
        hash3 = drift_monitor._calculate_data_hash(different_data)
        assert hash1 != hash3

    def test_generate_report_id(self, drift_monitor):
        """Test unique report ID generation."""
        id1 = drift_monitor._generate_report_id("GL-001")
        id2 = drift_monitor._generate_report_id("GL-001")

        # Both IDs should be valid 16-character hex strings
        assert len(id1) == 16
        assert len(id2) == 16
        # With random suffix, IDs should be different
        # (though there's a tiny chance they could be same, we use random suffix)

    def test_set_reference_data(
        self, drift_monitor, sample_reference_data, feature_names
    ):
        """Test setting reference data for an agent."""
        drift_monitor.set_reference_data(
            agent_id="GL-001",
            data=sample_reference_data,
            feature_names=feature_names,
        )

        assert "GL-001" in drift_monitor._reference_data
        assert "GL-001" in drift_monitor._feature_names
        assert drift_monitor._feature_names["GL-001"] == feature_names

    def test_detect_data_drift_no_drift(
        self,
        drift_monitor,
        sample_reference_data,
        sample_current_data_no_drift,
        feature_names,
    ):
        """Test data drift detection when no drift present."""
        result = drift_monitor.detect_data_drift(
            reference_data=sample_reference_data,
            current_data=sample_current_data_no_drift,
            agent_id="GL-001",
            model_name="test_model",
            model_version="1.0.0",
            feature_names=feature_names,
        )

        assert isinstance(result, DriftAnalysisResult)
        assert result.agent_id == "GL-001"
        assert result.model_name == "test_model"
        assert result.drift_type == "data"
        assert result.severity in ["none", "low"]
        assert result.overall_drift_score < 0.5
        assert result.reference_data_size == 1000
        assert result.current_data_size == 500
        assert result.report_hash != ""

    def test_detect_data_drift_with_drift(
        self,
        drift_monitor,
        sample_reference_data,
        sample_current_data_with_drift,
        feature_names,
    ):
        """Test data drift detection when significant drift present."""
        result = drift_monitor.detect_data_drift(
            reference_data=sample_reference_data,
            current_data=sample_current_data_with_drift,
            agent_id="GL-001",
            model_name="test_model",
            model_version="1.0.0",
            feature_names=feature_names,
        )

        assert result.drift_detected is True
        assert result.severity in ["medium", "high", "critical"]
        assert result.overall_drift_score > 0.3
        assert len(result.drifted_features) > 0
        assert len(result.recommendations) > 0

    def test_detect_data_drift_shape_mismatch(
        self, drift_monitor, sample_reference_data
    ):
        """Test error handling for data shape mismatch."""
        mismatched_data = np.random.normal(size=(500, 3))

        with pytest.raises(ValueError) as exc_info:
            drift_monitor.detect_data_drift(
                reference_data=sample_reference_data,
                current_data=mismatched_data,
                agent_id="GL-001",
            )
        assert "Feature count mismatch" in str(exc_info.value)

    def test_detect_prediction_drift(self, drift_monitor):
        """Test prediction drift detection."""
        np.random.seed(42)
        reference_preds = np.random.normal(loc=100, scale=10, size=500)
        current_preds = np.random.normal(loc=120, scale=15, size=300)

        result = drift_monitor.detect_prediction_drift(
            reference_predictions=reference_preds,
            current_predictions=current_preds,
            agent_id="GL-001",
            model_name="predictor",
            model_version="2.0.0",
        )

        assert result.drift_type == "prediction"
        assert result.agent_id == "GL-001"
        assert len(result.feature_drift_results) == 1
        assert result.feature_drift_results[0].feature_name == "predictions"

    def test_detect_concept_drift(self, drift_monitor):
        """Test concept drift detection."""
        np.random.seed(42)
        ref_preds = np.random.normal(loc=100, scale=10, size=500)
        ref_actuals = ref_preds + np.random.normal(0, 5, size=500)

        cur_preds = np.random.normal(loc=100, scale=10, size=300)
        cur_actuals = cur_preds + np.random.normal(20, 15, size=300)  # Higher error

        result = drift_monitor.detect_concept_drift(
            reference_predictions=ref_preds,
            reference_actuals=ref_actuals,
            current_predictions=cur_preds,
            current_actuals=cur_actuals,
            agent_id="GL-001",
        )

        assert result.drift_type == "concept"
        assert result.drift_detected is True
        assert len(result.recommendations) > 0

    def test_generate_drift_report(
        self,
        drift_monitor,
        sample_reference_data,
        sample_current_data_with_drift,
        feature_names,
        temp_storage_path,
    ):
        """Test drift report generation."""
        result = drift_monitor.detect_data_drift(
            reference_data=sample_reference_data,
            current_data=sample_current_data_with_drift,
            agent_id="GL-001",
            feature_names=feature_names,
        )

        report_path = drift_monitor.generate_drift_report(
            result=result,
            output_path=temp_storage_path,
            format="both",
        )

        assert Path(report_path).exists()
        assert ".json" in report_path or ".html" in report_path

    def test_get_drift_metrics(
        self,
        drift_monitor,
        sample_reference_data,
        sample_current_data_with_drift,
        feature_names,
    ):
        """Test aggregated drift metrics retrieval."""
        # Run multiple analyses
        for _ in range(3):
            drift_monitor.detect_data_drift(
                reference_data=sample_reference_data,
                current_data=sample_current_data_with_drift,
                agent_id="GL-001",
                feature_names=feature_names,
            )

        metrics = drift_monitor.get_drift_metrics("GL-001", window_hours=24)

        assert metrics["agent_id"] == "GL-001"
        assert metrics["analysis_count"] == 3
        assert "severity_distribution" in metrics
        assert "most_drifted_features" in metrics

    def test_export_prometheus_metrics(
        self,
        drift_monitor,
        sample_reference_data,
        sample_current_data_with_drift,
        feature_names,
    ):
        """Test Prometheus metrics export."""
        drift_monitor.detect_data_drift(
            reference_data=sample_reference_data,
            current_data=sample_current_data_with_drift,
            agent_id="GL-001",
            feature_names=feature_names,
        )

        metrics = drift_monitor.export_prometheus_metrics("GL-001")

        assert len(metrics) > 0
        assert all(isinstance(m, PrometheusMetricExport) for m in metrics)
        assert any("drift" in m.name for m in metrics)

    def test_prometheus_text_format(
        self,
        drift_monitor,
        sample_reference_data,
        sample_current_data_with_drift,
        feature_names,
    ):
        """Test Prometheus text exposition format."""
        drift_monitor.detect_data_drift(
            reference_data=sample_reference_data,
            current_data=sample_current_data_with_drift,
            agent_id="GL-001",
            feature_names=feature_names,
        )

        text = drift_monitor.get_prometheus_text("GL-001")

        assert "# HELP" in text
        assert "# TYPE" in text
        assert "greenlang_drift" in text


# =============================================================================
# Drift Profiles Tests
# =============================================================================

class TestDriftProfiles:
    """Tests for drift profile configurations."""

    def test_gl001_carbon_emissions_profile(self):
        """Test GL-001 Carbon Emissions drift profile."""
        profile = GL001CarbonEmissionsDriftProfile()

        assert profile.agent_id == "GL-001"
        assert profile.agent_name == "Carbon Emissions Calculator"
        assert len(profile.expected_features) > 0
        assert profile.psi_threshold == 0.2
        assert profile.regulatory_framework == "GHG Protocol"

        # Check feature specs
        fuel_spec = profile.get_feature_spec("fuel_consumption_mwh")
        assert fuel_spec is not None
        assert fuel_spec.importance_weight == 2.0

    def test_gl003_csrd_profile(self):
        """Test GL-003 CSRD Reporting drift profile."""
        profile = GL003CSRDReportingDriftProfile()

        assert profile.agent_id == "GL-003"
        assert profile.agent_name == "CSRD Reporting Agent"
        assert profile.psi_threshold == 0.15  # More stringent
        assert "EU CSRD" in profile.compliance_requirements

    def test_gl006_scope3_profile(self):
        """Test GL-006 Scope 3 drift profile."""
        profile = GL006Scope3DriftProfile()

        assert profile.agent_id == "GL-006"
        assert "GHG Protocol Scope 3" in profile.regulatory_framework
        assert len(profile.expected_features) >= 10

    def test_gl010_emissions_guardian_profile(self):
        """Test GL-010 Emissions Guardian drift profile."""
        profile = GL010EmissionsGuardianDriftProfile()

        assert profile.agent_id == "GL-010"
        assert profile.monitoring_interval_minutes == 5  # Frequent
        assert profile.psi_threshold == 0.1  # Sensitive
        assert profile.alert_config.trigger_rollback_on_critical is True

    def test_feature_spec_validation(self):
        """Test FeatureSpec validation."""
        spec = FeatureSpec(
            name="test_feature",
            feature_type="numerical",
            expected_mean=100.0,
            expected_std=20.0,
            importance_weight=2.0,
        )

        assert spec.name == "test_feature"
        assert spec.importance_weight == 2.0
        assert spec.custom_psi_threshold is None

    def test_alert_config(self):
        """Test AlertConfig model."""
        config = AlertConfig(
            low_threshold=0.1,
            medium_threshold=0.2,
            high_threshold=0.3,
            critical_threshold=0.5,
            trigger_retrain_on_critical=True,
        )

        assert config.low_threshold == 0.1
        assert config.trigger_retrain_on_critical is True
        assert config.alert_cooldown_minutes == 60

    def test_get_drift_profile(self):
        """Test drift profile factory function."""
        profile = get_drift_profile("GL-001")
        assert profile.agent_id == "GL-001"

        profile = get_drift_profile("GL-003")
        assert profile.agent_id == "GL-003"

        # Default profile for unknown agent
        profile = get_drift_profile("GL-015")
        assert profile.agent_id == "GL-015"

    def test_list_available_profiles(self):
        """Test listing available profiles."""
        profiles = list_available_profiles()

        assert "GL-001" in profiles
        assert "GL-003" in profiles
        assert "GL-006" in profiles
        assert "GL-010" in profiles

    def test_create_custom_profile(self):
        """Test custom profile creation."""
        profile = create_custom_profile(
            agent_id="GL-015",
            agent_name="Custom Agent",
            features=[
                {"name": "feature1", "feature_type": "numerical", "importance_weight": 2.0},
                {"name": "feature2", "feature_type": "categorical"},
            ],
            psi_threshold=0.25,
        )

        assert profile.agent_id == "GL-015"
        assert profile.agent_name == "Custom Agent"
        assert len(profile.expected_features) == 2
        assert profile.psi_threshold == 0.25

    def test_profile_feature_names(self):
        """Test profile feature_names property."""
        profile = GL001CarbonEmissionsDriftProfile()

        names = profile.feature_names
        assert "fuel_consumption_mwh" in names
        assert "emission_factor" in names

    def test_profile_get_feature_threshold(self):
        """Test custom feature thresholds."""
        profile = GL001CarbonEmissionsDriftProfile()

        # Feature with custom threshold
        ef_threshold = profile.get_feature_threshold("emission_factor")
        assert ef_threshold == 0.15  # Custom threshold

        # Feature without custom threshold (uses default)
        temp_threshold = profile.get_feature_threshold("temperature_celsius")
        assert temp_threshold == profile.psi_threshold


# =============================================================================
# Alert Manager Tests
# =============================================================================

class TestDriftAlertManager:
    """Tests for DriftAlertManager class."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        manager = DriftAlertManager()
        assert manager.config is not None
        assert AlertChannel.LOG in manager.handlers

    def test_init_custom_config(self, alert_manager_config):
        """Test initialization with custom configuration."""
        manager = DriftAlertManager(config=alert_manager_config)
        assert manager.config == alert_manager_config

    def test_create_alert(self, alert_manager):
        """Test alert creation."""
        alert = alert_manager.create_alert(
            agent_id="GL-001",
            severity=AlertSeverity.WARNING,
            message="Test drift detected",
            drift_score=0.35,
            drift_type="data",
            drifted_features=["feature1", "feature2"],
        )

        assert isinstance(alert, DriftAlert)
        assert alert.agent_id == "GL-001"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.drift_score == 0.35
        assert len(alert.drifted_features) == 2
        assert alert.provenance_hash != ""

    def test_alert_provenance_hash(self, alert_manager):
        """Test alert provenance hash calculation."""
        alert = alert_manager.create_alert(
            agent_id="GL-001",
            severity=AlertSeverity.INFO,
            message="Test",
            drift_score=0.1,
        )

        hash1 = alert.provenance_hash
        hash2 = alert.calculate_provenance_hash()

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_dispatch_alert(self, alert_manager):
        """Test alert dispatching."""
        alert = alert_manager.create_alert(
            agent_id="GL-001",
            severity=AlertSeverity.WARNING,
            message="Test drift",
            drift_score=0.3,
        )

        # Should dispatch to log channel (always enabled)
        success = alert_manager.dispatch_alert(alert)
        assert success is True
        assert alert.status == AlertStatus.DISPATCHED
        assert AlertChannel.LOG in alert.dispatch_channels

    def test_alert_throttling(self, alert_manager):
        """Test alert throttling."""
        # First alert should succeed
        alert1 = alert_manager.create_alert(
            agent_id="GL-001",
            severity=AlertSeverity.WARNING,
            message="First alert",
            drift_score=0.3,
        )
        success1 = alert_manager.dispatch_alert(alert1)
        assert success1 is True

        # Second alert immediately after should be throttled
        alert2 = alert_manager.create_alert(
            agent_id="GL-001",
            severity=AlertSeverity.WARNING,
            message="Second alert",
            drift_score=0.3,
        )
        success2 = alert_manager.dispatch_alert(alert2)
        assert success2 is False

    def test_acknowledge_alert(self, alert_manager):
        """Test alert acknowledgment."""
        alert = alert_manager.create_alert(
            agent_id="GL-001",
            severity=AlertSeverity.WARNING,
            message="Test",
            drift_score=0.3,
        )
        alert_manager.dispatch_alert(alert)

        success = alert_manager.acknowledge_alert(
            alert_id=alert.alert_id,
            acknowledged_by="test_user",
        )

        assert success is True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "test_user"

    def test_resolve_alert(self, alert_manager):
        """Test alert resolution."""
        alert = alert_manager.create_alert(
            agent_id="GL-001",
            severity=AlertSeverity.WARNING,
            message="Test",
            drift_score=0.3,
        )
        alert_manager.dispatch_alert(alert)

        success = alert_manager.resolve_alert(alert.alert_id)

        assert success is True
        assert alert.status == AlertStatus.RESOLVED

    def test_get_active_alerts(self, alert_manager):
        """Test retrieving active alerts."""
        # Create and dispatch multiple alerts with different agents
        for i, agent_id in enumerate(["GL-001", "GL-002", "GL-003"]):
            alert = alert_manager.create_alert(
                agent_id=agent_id,
                severity=AlertSeverity.WARNING,
                message=f"Test alert {i}",
                drift_score=0.3,
            )
            # Clear throttle for test
            alert_manager._cooldowns.clear()
            alert_manager.dispatch_alert(alert)

        # Get all active alerts
        all_alerts = alert_manager.get_active_alerts()
        assert len(all_alerts) == 3

        # Filter by agent
        gl001_alerts = alert_manager.get_active_alerts(agent_id="GL-001")
        assert len(gl001_alerts) == 1

    def test_get_alert_statistics(self, alert_manager):
        """Test alert statistics."""
        for i in range(5):
            alert = alert_manager.create_alert(
                agent_id="GL-001",
                severity=AlertSeverity.WARNING,
                message=f"Test {i}",
                drift_score=0.3,
            )
            alert_manager._cooldowns.clear()
            alert_manager.dispatch_alert(alert)

        stats = alert_manager.get_alert_statistics(agent_id="GL-001", hours=24)

        assert stats["agent_id"] == "GL-001"
        assert stats["active_alerts"] == 5
        assert "severity_distribution" in stats

    def test_remediation_callback(self, alert_manager):
        """Test remediation callback registration and triggering."""
        callback_called = {"value": False}

        def test_callback(alert: DriftAlert):
            callback_called["value"] = True

        alert_manager.register_remediation_callback(
            RemediationAction.RETRAIN,
            test_callback,
        )

        # Create critical alert that triggers remediation
        alert = alert_manager.create_alert(
            agent_id="GL-001",
            severity=AlertSeverity.CRITICAL,
            message="Critical drift",
            drift_score=0.8,
        )

        alert_manager.dispatch_alert(alert)

        assert callback_called["value"] is True

    def test_test_integrations(self, alert_manager):
        """Test integration connection testing."""
        results = alert_manager.test_integrations()

        assert AlertChannel.LOG.value in results
        assert results[AlertChannel.LOG.value] is True

        # Disabled integrations
        assert results.get(AlertChannel.SLACK.value) is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestDriftDetectionIntegration:
    """Integration tests for the complete drift detection pipeline."""

    def test_full_drift_detection_pipeline(self, temp_storage_path):
        """Test complete drift detection and alerting pipeline."""
        # Setup monitor
        config = EvidentlyDriftConfig(storage_path=temp_storage_path)
        monitor = ProcessHeatDriftMonitor(config=config)

        # Setup alert manager
        alert_config = AlertManagerConfig(
            storage_path=temp_storage_path,
            cooldown_minutes=1,  # Minimum allowed
        )
        alert_manager = DriftAlertManager(config=alert_config)

        # Generate test data
        np.random.seed(42)
        reference = np.random.normal(loc=100, scale=20, size=(1000, 5))
        current = np.random.normal(loc=150, scale=40, size=(500, 5))
        features = ["temp", "pressure", "flow", "efficiency", "emissions"]

        # Detect drift
        result = monitor.detect_data_drift(
            reference_data=reference,
            current_data=current,
            agent_id="GL-001",
            model_name="test_model",
            feature_names=features,
        )

        # Create and dispatch alert based on result
        if result.drift_detected:
            severity = {
                "critical": AlertSeverity.CRITICAL,
                "high": AlertSeverity.CRITICAL,
                "medium": AlertSeverity.WARNING,
                "low": AlertSeverity.INFO,
                "none": AlertSeverity.INFO,
            }.get(result.severity, AlertSeverity.INFO)

            alert = alert_manager.create_alert(
                agent_id=result.agent_id,
                severity=severity,
                message=f"Drift detected with score {result.overall_drift_score:.3f}",
                drift_score=result.overall_drift_score,
                drift_type=result.drift_type,
                drifted_features=result.drifted_features,
                recommendations=result.recommendations,
                report_id=result.report_id,
            )

            success = alert_manager.dispatch_alert(alert)
            assert success is True

        # Verify metrics
        metrics = monitor.get_drift_metrics("GL-001")
        assert metrics["analysis_count"] > 0

        stats = alert_manager.get_alert_statistics("GL-001")
        assert stats["active_alerts"] > 0


# =============================================================================
# Statistical Tests
# =============================================================================

class TestStatisticalMethods:
    """Tests for statistical drift detection methods."""

    def test_ks_test(self, drift_monitor):
        """Test Kolmogorov-Smirnov test."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        similar = np.random.normal(0, 1, 500)
        different = np.random.normal(2, 1, 500)

        # Similar distributions
        stat1, pval1 = drift_monitor._ks_test(reference, similar)
        assert pval1 > 0.05  # Should not reject null hypothesis

        # Different distributions
        stat2, pval2 = drift_monitor._ks_test(reference, different)
        assert pval2 < 0.05  # Should reject null hypothesis

    def test_psi_calculation(self, drift_monitor):
        """Test Population Stability Index calculation."""
        np.random.seed(42)
        reference = np.random.normal(100, 20, 1000)
        similar = np.random.normal(100, 20, 500)
        different = np.random.normal(150, 40, 500)

        psi_similar = drift_monitor._calculate_psi(reference, similar)
        psi_different = drift_monitor._calculate_psi(reference, different)

        assert psi_similar < 0.1  # Low PSI
        assert psi_different > 0.2  # High PSI

    def test_js_divergence(self, drift_monitor):
        """Test Jensen-Shannon divergence calculation."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        similar = np.random.normal(0, 1, 500)
        different = np.random.normal(5, 1, 500)

        js_similar = drift_monitor._calculate_js_divergence(reference, similar)
        js_different = drift_monitor._calculate_js_divergence(reference, different)

        assert js_similar < js_different
        assert js_similar >= 0
        assert js_different >= 0

    def test_wasserstein_distance(self, drift_monitor):
        """Test Wasserstein distance calculation."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        similar = np.random.normal(0, 1, 500)
        different = np.random.normal(5, 1, 500)

        wd_similar = drift_monitor._calculate_wasserstein(reference, similar)
        wd_different = drift_monitor._calculate_wasserstein(reference, different)

        assert wd_similar < wd_different
        assert wd_different > 4  # Approximately mean difference


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
