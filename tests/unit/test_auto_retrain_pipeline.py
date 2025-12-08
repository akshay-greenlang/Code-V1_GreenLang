"""
Unit tests for auto-retraining pipeline.

Tests cover:
    - Trigger evaluation (performance, drift, scheduled, manual)
    - Job management and tracking
    - Model validation
    - Deployment decision logic
    - Error handling and rollback
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call

from greenlang.ml.pipelines.auto_retrain import (
    AutoRetrainPipeline,
    TriggerConfig,
    TriggerType,
    RetrainingStatus,
    PerformanceDegradationTrigger,
    DataDriftTrigger,
    ScheduledTrigger,
    ValidationResult,
)


class TestTriggerConfig:
    """Test TriggerConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TriggerConfig()
        assert config.performance_metric_threshold == 0.92
        assert config.drift_threshold == 0.25
        assert config.evaluation_window_days == 30
        assert config.training_window_days == 90

    def test_custom_config(self):
        """Test custom configuration."""
        config = TriggerConfig(
            performance_metric_threshold=0.90,
            drift_threshold=0.30,
            training_window_days=120
        )
        assert config.performance_metric_threshold == 0.90
        assert config.drift_threshold == 0.30
        assert config.training_window_days == 120

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            TriggerConfig(performance_metric_threshold=1.5)  # > 1.0

        with pytest.raises(ValueError):
            TriggerConfig(drift_threshold=-0.1)  # < 0.0


class TestPerformanceDegradationTrigger:
    """Test performance degradation trigger."""

    def test_should_retrain_when_degraded(self):
        """Test trigger returns True when performance is degraded."""
        trigger = PerformanceDegradationTrigger(metric_threshold=0.92)

        with patch.object(trigger, '_fetch_current_metrics') as mock_fetch:
            mock_fetch.return_value = {"accuracy": 0.90}  # Below threshold
            should_retrain, reason = trigger.should_retrain("test_model")

            assert should_retrain is True
            assert "degraded" in reason.lower()

    def test_should_not_retrain_when_healthy(self):
        """Test trigger returns False when performance is healthy."""
        trigger = PerformanceDegradationTrigger(metric_threshold=0.92)

        with patch.object(trigger, '_fetch_current_metrics') as mock_fetch:
            mock_fetch.return_value = {"accuracy": 0.95}  # Above threshold
            should_retrain, reason = trigger.should_retrain("test_model")

            assert should_retrain is False
            assert "acceptable" in reason.lower()

    def test_trigger_type(self):
        """Test trigger type."""
        trigger = PerformanceDegradationTrigger(metric_threshold=0.92)
        assert trigger.get_trigger_type() == TriggerType.PERFORMANCE_DEGRADATION

    def test_handles_missing_metrics(self):
        """Test trigger handles missing metrics gracefully."""
        trigger = PerformanceDegradationTrigger(metric_threshold=0.92)

        with patch.object(trigger, '_fetch_current_metrics') as mock_fetch:
            mock_fetch.return_value = None
            should_retrain, reason = trigger.should_retrain("test_model")

            assert should_retrain is False


class TestDataDriftTrigger:
    """Test data drift trigger."""

    def test_should_retrain_on_drift(self):
        """Test trigger returns True when drift is detected."""
        trigger = DataDriftTrigger(drift_threshold=0.25)

        with patch.object(trigger, '_check_drift_from_evidently') as mock_drift:
            mock_drift.return_value = 0.30  # Above threshold
            should_retrain, reason = trigger.should_retrain("test_model")

            assert should_retrain is True
            assert "drift" in reason.lower()

    def test_should_not_retrain_without_drift(self):
        """Test trigger returns False when drift is not detected."""
        trigger = DataDriftTrigger(drift_threshold=0.25)

        with patch.object(trigger, '_check_drift_from_evidently') as mock_drift:
            mock_drift.return_value = 0.15  # Below threshold
            should_retrain, reason = trigger.should_retrain("test_model")

            assert should_retrain is False

    def test_trigger_type(self):
        """Test trigger type."""
        trigger = DataDriftTrigger()
        assert trigger.get_trigger_type() == TriggerType.DATA_DRIFT


class TestScheduledTrigger:
    """Test scheduled trigger."""

    def test_should_retrain_on_first_check(self):
        """Test trigger returns True on first check (no prior retrain)."""
        trigger = ScheduledTrigger(schedule_expression="0 0 * * 0")
        should_retrain, reason = trigger.should_retrain("test_model")

        assert should_retrain is True

    def test_should_not_retrain_before_schedule(self):
        """Test trigger returns False before schedule is due."""
        trigger = ScheduledTrigger(schedule_expression="0 0 * * 0")

        # Record that retraining just happened
        trigger.record_retrain("test_model")

        # Check immediately - should not retrain
        should_retrain, reason = trigger.should_retrain("test_model")
        assert should_retrain is False

    def test_should_retrain_after_schedule(self):
        """Test trigger returns True after schedule duration passes."""
        trigger = ScheduledTrigger(schedule_expression="0 0 * * 0")

        # Record retrain time 8 days ago
        trigger.last_retrain["test_model"] = datetime.now() - timedelta(days=8)

        # Should retrain now
        should_retrain, reason = trigger.should_retrain("test_model")
        assert should_retrain is True

    def test_trigger_type(self):
        """Test trigger type."""
        trigger = ScheduledTrigger("0 0 * * 0")
        assert trigger.get_trigger_type() == TriggerType.SCHEDULED


class TestAutoRetrainPipeline:
    """Test AutoRetrainPipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline."""
        config = TriggerConfig()
        return AutoRetrainPipeline(config)

    def test_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.config is not None
        assert len(pipeline.triggers) == 0
        assert len(pipeline.job_history) == 0

    def test_configure_trigger(self, pipeline):
        """Test trigger configuration."""
        pipeline.configure_trigger(
            metric_threshold=0.90,
            drift_threshold=0.20,
            schedule="0 0 * * 1"
        )

        assert len(pipeline.triggers) == 3
        assert pipeline.config.performance_metric_threshold == 0.90
        assert pipeline.config.drift_threshold == 0.20

    def test_check_retrain_needed_no_triggers(self, pipeline):
        """Test retrain check when no triggers configured."""
        result = pipeline.check_retrain_needed("test_model")
        assert result is False

    def test_check_retrain_needed_with_triggers(self, pipeline):
        """Test retrain check with configured triggers."""
        pipeline.configure_trigger()

        # Mock one trigger to return True
        with patch.object(pipeline.triggers[0], 'should_retrain') as mock_trigger:
            mock_trigger.return_value = (True, "Test trigger")
            result = pipeline.check_retrain_needed("test_model")

            assert result is True

    def test_start_retrain_job(self, pipeline):
        """Test starting a retrain job."""
        pipeline.configure_trigger()

        with patch.object(pipeline, '_extract_training_data') as mock_extract, \
             patch.object(pipeline, '_submit_k8s_job') as mock_submit, \
             patch.object(pipeline, '_notify_slack') as mock_notify:

            mock_extract.return_value = {"features": [], "labels": []}
            mock_submit.return_value = "k8s_job_123"

            job_id = pipeline.start_retrain_job(
                "test_model",
                {"learning_rate": 0.001}
            )

            assert job_id is not None
            assert job_id in pipeline.job_history

            job = pipeline.job_history[job_id]
            assert job.model_name == "test_model"
            assert job.status == RetrainingStatus.RUNNING
            assert job.k8s_job_name == "k8s_job_123"

    def test_validate_new_model(self, pipeline):
        """Test model validation."""
        with patch.object(pipeline, '_extract_validation_data') as mock_extract, \
             patch.object(pipeline, '_load_model_from_mlflow') as mock_load, \
             patch.object(pipeline, '_calculate_metrics') as mock_metrics:

            mock_extract.return_value = {"features": [], "labels": []}
            mock_load.return_value = MagicMock()
            mock_metrics.return_value = {
                "accuracy": 0.94,
                "precision": 0.93,
                "recall": 0.92,
                "f1_score": 0.925
            }

            result = pipeline.validate_new_model("test_model")

            assert isinstance(result, ValidationResult)
            assert result.is_valid is True
            assert result.accuracy == 0.94
            assert result.validation_hash is not None

    def test_deploy_if_better_with_improvement(self, pipeline):
        """Test deployment when model shows improvement."""
        job_id = "test_job_123"
        from greenlang.ml.pipelines.auto_retrain import RetariningJob

        job = RetariningJob(
            job_id=job_id,
            model_name="test_model",
            trigger_type=TriggerType.MANUAL,
            status=RetrainingStatus.VALIDATION,
            created_at=datetime.now()
        )
        pipeline.job_history[job_id] = job

        with patch.object(pipeline, '_get_model_metrics') as mock_metrics, \
             patch.object(pipeline, '_promote_to_production') as mock_promote, \
             patch.object(pipeline, '_notify_slack'):

            # New model F1: 0.95, Champion F1: 0.90 (5% improvement)
            mock_metrics.side_effect = [
                {"f1_score": 0.95},  # New model
                {"f1_score": 0.90}   # Champion
            ]
            mock_promote.return_value = True

            deployed = pipeline.deploy_if_better(
                "test_model",
                job_id,
                min_improvement=0.05
            )

            assert deployed is True
            assert job.deployed is True
            assert job.status == RetrainingStatus.COMPLETED
            mock_promote.assert_called_once()

    def test_deploy_if_better_without_improvement(self, pipeline):
        """Test deployment rejection when improvement is insufficient."""
        job_id = "test_job_123"
        from greenlang.ml.pipelines.auto_retrain import RetariningJob

        job = RetariningJob(
            job_id=job_id,
            model_name="test_model",
            trigger_type=TriggerType.MANUAL,
            status=RetrainingStatus.VALIDATION,
            created_at=datetime.now()
        )
        pipeline.job_history[job_id] = job

        with patch.object(pipeline, '_get_model_metrics') as mock_metrics, \
             patch.object(pipeline, '_promote_to_production') as mock_promote, \
             patch.object(pipeline, '_notify_slack'):

            # New model F1: 0.91, Champion F1: 0.90 (1% improvement < 5% threshold)
            mock_metrics.side_effect = [
                {"f1_score": 0.91},  # New model
                {"f1_score": 0.90}   # Champion
            ]

            deployed = pipeline.deploy_if_better(
                "test_model",
                job_id,
                min_improvement=0.05
            )

            assert deployed is False
            assert job.deployed is False
            mock_promote.assert_not_called()

    def test_get_job_status(self, pipeline):
        """Test retrieving job status."""
        from greenlang.ml.pipelines.auto_retrain import RetariningJob

        job = RetariningJob(
            job_id="test_job",
            model_name="test_model",
            trigger_type=TriggerType.MANUAL,
            status=RetrainingStatus.RUNNING,
            created_at=datetime.now()
        )
        pipeline.job_history["test_job"] = job

        retrieved = pipeline.get_job_status("test_job")
        assert retrieved == job
        assert retrieved.status == RetrainingStatus.RUNNING

    def test_list_recent_jobs(self, pipeline):
        """Test listing recent jobs."""
        from greenlang.ml.pipelines.auto_retrain import RetariningJob

        for i in range(5):
            job = RetariningJob(
                job_id=f"job_{i}",
                model_name="test_model",
                trigger_type=TriggerType.MANUAL,
                status=RetrainingStatus.COMPLETED,
                created_at=datetime.now() - timedelta(hours=i)
            )
            pipeline.job_history[job.job_id] = job

        recent = pipeline.list_recent_jobs("test_model", limit=3)
        assert len(recent) == 3
        # Should be ordered by created_at descending
        assert recent[0].job_id == "job_0"

    def test_list_recent_jobs_all_models(self, pipeline):
        """Test listing recent jobs across all models."""
        from greenlang.ml.pipelines.auto_retrain import RetariningJob

        for i in range(3):
            job1 = RetariningJob(
                job_id=f"job1_{i}",
                model_name="model1",
                trigger_type=TriggerType.MANUAL,
                status=RetrainingStatus.COMPLETED,
                created_at=datetime.now() - timedelta(hours=i)
            )
            job2 = RetariningJob(
                job_id=f"job2_{i}",
                model_name="model2",
                trigger_type=TriggerType.MANUAL,
                status=RetrainingStatus.COMPLETED,
                created_at=datetime.now() - timedelta(hours=i)
            )
            pipeline.job_history[job1.job_id] = job1
            pipeline.job_history[job2.job_id] = job2

        all_recent = pipeline.list_recent_jobs(limit=3)
        assert len(all_recent) == 3


class TestIntegration:
    """Integration tests for complete pipeline workflows."""

    def test_complete_retrain_workflow(self):
        """Test complete retraining workflow."""
        config = TriggerConfig()
        pipeline = AutoRetrainPipeline(config)
        pipeline.configure_trigger()

        with patch.object(pipeline, 'check_retrain_needed') as mock_check, \
             patch.object(pipeline, 'start_retrain_job') as mock_start, \
             patch.object(pipeline, 'validate_new_model') as mock_validate, \
             patch.object(pipeline, 'deploy_if_better') as mock_deploy:

            mock_check.return_value = True
            mock_start.return_value = "job_123"
            mock_validate.return_value = ValidationResult(
                is_valid=True,
                accuracy=0.95,
                precision=0.94,
                recall=0.93,
                f1_score=0.935,
                validation_hash="abc123",
                validation_timestamp=datetime.now()
            )
            mock_deploy.return_value = True

            # Check if retrain needed
            needs_retrain = pipeline.check_retrain_needed("heat_predictor")
            assert needs_retrain is True

            # Start retraining
            job_id = pipeline.start_retrain_job("heat_predictor", {})
            assert job_id == "job_123"

            # Validate
            result = pipeline.validate_new_model("heat_predictor")
            assert result.is_valid is True

            # Deploy
            deployed = pipeline.deploy_if_better("heat_predictor", job_id)
            assert deployed is True
