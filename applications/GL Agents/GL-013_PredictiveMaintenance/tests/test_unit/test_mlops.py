# -*- coding: utf-8 -*-
import pytest
from datetime import datetime
import numpy as np
import sys
sys.path.insert(0, str(r"c:/Users/aksha/Code-V1_GreenLang/GL Agents/GL-013_PredictiveMaintenance"))


class TestModelRegistry:
    def test_register_model(self):
        from mlops.model_registry import ModelRegistry, ModelVersion, ModelStatus, ModelType
        registry = ModelRegistry()
        model = ModelVersion(
            version_id="v1.0.0",
            model_name="rul_estimator",
            model_type=ModelType.RUL_ESTIMATOR,
            version="1.0.0",
            status=ModelStatus.DEVELOPMENT,
            metrics={"mae": 5.2, "rmse": 7.8},
            hyperparameters={"n_estimators": 100},
            training_data_hash="abc123",
        )
        version_id = registry.register_model(model)
        assert version_id == "v1.0.0"
        assert model.provenance_hash is not None
    
    def test_get_model(self):
        from mlops.model_registry import ModelRegistry, ModelVersion, ModelStatus, ModelType
        registry = ModelRegistry()
        model = ModelVersion(
            version_id="v1.0.0",
            model_name="rul_estimator",
            model_type=ModelType.RUL_ESTIMATOR,
            version="1.0.0",
            status=ModelStatus.DEVELOPMENT,
            metrics={},
            hyperparameters={},
            training_data_hash="abc123",
        )
        registry.register_model(model)
        retrieved = registry.get_model("rul_estimator", "v1.0.0")
        assert retrieved is not None
        assert retrieved.version_id == "v1.0.0"
    
    def test_promote_to_production(self):
        from mlops.model_registry import ModelRegistry, ModelVersion, ModelStatus, ModelType
        registry = ModelRegistry()
        model = ModelVersion(
            version_id="v1.0.0",
            model_name="rul_estimator",
            model_type=ModelType.RUL_ESTIMATOR,
            version="1.0.0",
            status=ModelStatus.STAGING,
            metrics={},
            hyperparameters={},
            training_data_hash="abc123",
        )
        registry.register_model(model)
        result = registry.promote_to_production("rul_estimator", "v1.0.0")
        assert result is True
        
        prod_model = registry.get_model("rul_estimator")
        assert prod_model.status == ModelStatus.PRODUCTION
    
    def test_compare_versions(self):
        from mlops.model_registry import ModelRegistry, ModelVersion, ModelStatus, ModelType
        registry = ModelRegistry()
        
        for i, (mae, rmse) in enumerate([(5.0, 7.0), (4.5, 6.5)]):
            model = ModelVersion(
                version_id=f"v1.{i}.0",
                model_name="rul_estimator",
                model_type=ModelType.RUL_ESTIMATOR,
                version=f"1.{i}.0",
                status=ModelStatus.DEVELOPMENT,
                metrics={"mae": mae, "rmse": rmse},
                hyperparameters={},
                training_data_hash=f"hash{i}",
            )
            registry.register_model(model)
        
        comparison = registry.compare_versions("rul_estimator", "v1.0.0", "v1.1.0")
        assert "metrics" in comparison
        assert comparison["metrics"]["mae"]["improved"] is True


class TestDriftDetector:
    def test_set_reference(self):
        from mlops.drift_detection import DriftDetector
        detector = DriftDetector()
        reference = np.random.normal(0, 1, 1000)
        detector.set_reference("feature1", reference)
        assert "feature1" in detector._reference_data
    
    def test_detect_ks_drift_no_drift(self):
        from mlops.drift_detection import DriftDetector
        detector = DriftDetector()
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        detector.set_reference("feature1", reference)
        result = detector.detect_ks_drift("feature1", current)
        
        assert result.detected is False
        assert result.provenance_hash is not None
    
    def test_detect_ks_drift_with_drift(self):
        from mlops.drift_detection import DriftDetector
        detector = DriftDetector()
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(2, 1.5, 1000)  # Different distribution
        
        detector.set_reference("feature1", reference)
        result = detector.detect_ks_drift("feature1", current)
        
        assert result.detected is True
    
    def test_get_drift_summary(self):
        from mlops.drift_detection import DriftDetector
        detector = DriftDetector()
        np.random.seed(42)
        
        for name in ["f1", "f2", "f3"]:
            ref = np.random.normal(0, 1, 500)
            cur = np.random.normal(0.5, 1, 500)
            detector.set_reference(name, ref)
            detector.detect_ks_drift(name, cur)
        
        summary = detector.get_drift_summary()
        assert summary["total_checks"] == 3


class TestModelMonitor:
    def test_record_metric(self):
        from mlops.monitoring import ModelMonitor, MetricValue, MetricType
        monitor = ModelMonitor()
        metric = MetricValue(
            metric_type=MetricType.ACCURACY,
            value=0.92,
            model_name="rul_estimator",
            version_id="v1.0.0",
        )
        monitor.record_metric(metric)
        
        metrics = monitor.get_metrics("rul_estimator", MetricType.ACCURACY)
        assert len(metrics) == 1
    
    def test_threshold_violation_creates_alert(self):
        from mlops.monitoring import ModelMonitor, MetricValue, MetricType
        monitor = ModelMonitor()
        metric = MetricValue(
            metric_type=MetricType.ACCURACY,
            value=0.5,  # Below default threshold of 0.8
            model_name="rul_estimator",
            version_id="v1.0.0",
        )
        monitor.record_metric(metric)
        
        alerts = monitor.get_alerts()
        assert len(alerts) == 1
    
    def test_get_health_summary(self):
        from mlops.monitoring import ModelMonitor
        monitor = ModelMonitor()
        summary = monitor.get_health_summary()
        
        assert "total_metrics_tracked" in summary
        assert "health_status" in summary
        assert summary["health_status"] == "healthy"
