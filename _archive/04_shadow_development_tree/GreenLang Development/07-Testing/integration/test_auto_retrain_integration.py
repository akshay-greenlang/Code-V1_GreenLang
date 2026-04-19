"""
Integration tests for auto-retraining pipeline.

Tests complete workflows including:
    - End-to-end retraining pipeline
    - Trigger evaluation across multiple models
    - Job orchestration and tracking
    - Deployment safety mechanisms
    - Error handling and recovery
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

from greenlang.ml.pipelines.auto_retrain import (
    AutoRetrainPipeline,
    TriggerConfig,
    TriggerType,
    RetrainingStatus,
)


class TestEndToEndRetrainingWorkflow:
    """Test complete retraining workflows."""

    @pytest.fixture
    def pipeline_with_mocks(self):
        """Create a pipeline with necessary mocks."""
        config = TriggerConfig(
            performance_metric_threshold=0.92,
            drift_threshold=0.25,
            training_window_days=90,
            evaluation_window_days=30
        )
        pipeline = AutoRetrainPipeline(config)
        pipeline.configure_trigger()
        return pipeline

    def test_complete_retrain_and_deploy_workflow(self, pipeline_with_mocks):
        """
        Test complete workflow from trigger to deployment.

        Workflow:
            1. Check if retrain needed
            2. Start retrain job
            3. Validate new model
            4. Compare with champion
            5. Deploy if better
        """
        pipeline = pipeline_with_mocks
        model_name = "heat_predictor_v2"

        with patch.object(pipeline.triggers[0], 'should_retrain') as mock_trigger, \
             patch.object(pipeline, '_extract_training_data'), \
             patch.object(pipeline, '_submit_k8s_job') as mock_k8s, \
             patch.object(pipeline, '_load_model_from_mlflow') as mock_load, \
             patch.object(pipeline, '_calculate_metrics') as mock_metrics, \
             patch.object(pipeline, '_get_model_metrics') as mock_model_metrics, \
             patch.object(pipeline, '_promote_to_production') as mock_promote, \
             patch.object(pipeline, '_notify_slack'):

            # Setup mocks
            mock_trigger.return_value = (True, "Performance degraded")
            mock_k8s.return_value = "k8s-job-123"
            mock_load.return_value = MagicMock()
            mock_metrics.return_value = {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.93,
                "f1_score": 0.935
            }
            # New model: 0.95, Champion: 0.90 (5.26% improvement)
            mock_model_metrics.side_effect = [
                {"accuracy": 0.95, "f1_score": 0.95},
                {"accuracy": 0.90, "f1_score": 0.90}
            ]
            mock_promote.return_value = True

            # Step 1: Check retrain needed
            needs_retrain = pipeline.check_retrain_needed(model_name)
            assert needs_retrain is True

            # Step 2: Start retrain job
            job_id = pipeline.start_retrain_job(
                model_name,
                {"learning_rate": 0.001, "epochs": 50}
            )
            assert job_id is not None
            job = pipeline.get_job_status(job_id)
            assert job.status == RetrainingStatus.RUNNING

            # Step 3: Validate new model
            validation = pipeline.validate_new_model(model_name)
            assert validation.is_valid is True
            assert validation.accuracy == 0.95

            # Step 4: Deploy if better
            deployed = pipeline.deploy_if_better(model_name, job_id, min_improvement=0.05)
            assert deployed is True

            # Verify final state
            final_job = pipeline.get_job_status(job_id)
            assert final_job.deployed is True
            assert final_job.status == RetrainingStatus.COMPLETED
            assert final_job.improvement_pct is not None
            mock_promote.assert_called_once()

    def test_retrain_rejected_due_to_insufficient_improvement(self, pipeline_with_mocks):
        """Test that retraining is rejected if improvement is insufficient."""
        pipeline = pipeline_with_mocks
        model_name = "heat_predictor_v2"

        with patch.object(pipeline, '_extract_training_data'), \
             patch.object(pipeline, '_submit_k8s_job'), \
             patch.object(pipeline, '_get_model_metrics') as mock_metrics, \
             patch.object(pipeline, '_promote_to_production') as mock_promote, \
             patch.object(pipeline, '_notify_slack'):

            mock_metrics.side_effect = [
                {"f1_score": 0.91},  # New: 0.91
                {"f1_score": 0.90}   # Champion: 0.90 (1.1% improvement < 5%)
            ]

            job_id = pipeline.start_retrain_job(model_name, {})

            # Deploy should return False
            deployed = pipeline.deploy_if_better(model_name, job_id, min_improvement=0.05)
            assert deployed is False

            # Promotion should not be called
            mock_promote.assert_not_called()

            # Job should be marked as not deployed
            job = pipeline.get_job_status(job_id)
            assert job.deployed is False

    def test_retrain_with_degradation_prevents_deployment(self, pipeline_with_mocks):
        """Test that degraded model is not deployed."""
        pipeline = pipeline_with_mocks
        model_name = "heat_predictor_v2"

        with patch.object(pipeline, '_extract_training_data'), \
             patch.object(pipeline, '_submit_k8s_job'), \
             patch.object(pipeline, '_get_model_metrics') as mock_metrics, \
             patch.object(pipeline, '_promote_to_production') as mock_promote, \
             patch.object(pipeline, '_notify_slack'):

            # New model worse than champion
            mock_metrics.side_effect = [
                {"f1_score": 0.88},  # New: 0.88
                {"f1_score": 0.90}   # Champion: 0.90 (-2.2%)
            ]

            job_id = pipeline.start_retrain_job(model_name, {})
            deployed = pipeline.deploy_if_better(model_name, job_id, min_improvement=0.05)

            assert deployed is False
            mock_promote.assert_not_called()


class TestMultiModelScenarios:
    """Test scenarios with multiple models."""

    def test_monitor_multiple_models_simultaneously(self):
        """Test monitoring multiple models in parallel."""
        pipeline = AutoRetrainPipeline(TriggerConfig())
        pipeline.configure_trigger()

        models = [
            "GL-001-thermal",
            "GL-002-steam",
            "GL-003-waste-heat",
            "GL-004-efficiency",
            "GL-005-cost"
        ]

        with patch.object(pipeline.triggers[0], 'should_retrain') as mock_trigger:
            # Models 1, 3, 5 need retrain; 2, 4 don't
            mock_trigger.side_effect = [
                (True, "Trigger 1"),
                (False, "No trigger"),
                (True, "Trigger 3"),
                (False, "No trigger"),
                (True, "Trigger 5"),
            ]

            results = {}
            for model in models:
                needs_retrain = pipeline.check_retrain_needed(model)
                results[model] = needs_retrain

            # Verify expected triggers
            assert results["GL-001-thermal"] is True
            assert results["GL-002-steam"] is False
            assert results["GL-003-waste-heat"] is True
            assert results["GL-004-efficiency"] is False
            assert results["GL-005-cost"] is True

    def test_staggered_job_submission(self):
        """Test submitting jobs for multiple models with staggering."""
        pipeline = AutoRetrainPipeline(TriggerConfig())

        with patch.object(pipeline, '_extract_training_data'), \
             patch.object(pipeline, '_submit_k8s_job') as mock_k8s, \
             patch.object(pipeline, '_notify_slack'):

            models = ["model1", "model2", "model3"]
            mock_k8s.side_effect = [
                "k8s-job-1",
                "k8s-job-2",
                "k8s-job-3"
            ]

            job_ids = []
            for model in models:
                job_id = pipeline.start_retrain_job(model, {})
                job_ids.append(job_id)

            # Verify all jobs submitted
            assert len(job_ids) == 3
            assert all(job_id in pipeline.job_history for job_id in job_ids)

            # Verify each has unique K8s job
            jobs = [pipeline.get_job_status(jid) for jid in job_ids]
            k8s_jobs = [j.k8s_job_name for j in jobs]
            assert len(set(k8s_jobs)) == 3  # All unique


class TestTriggerIntegration:
    """Test trigger integration and evaluation."""

    def test_all_triggers_evaluate_independently(self):
        """Test that all trigger types are evaluated."""
        pipeline = AutoRetrainPipeline(TriggerConfig())
        pipeline.configure_trigger(
            metric_threshold=0.92,
            drift_threshold=0.25,
            schedule="0 0 * * 0"
        )

        assert len(pipeline.triggers) == 3
        trigger_types = {t.get_trigger_type() for t in pipeline.triggers}

        assert TriggerType.PERFORMANCE_DEGRADATION in trigger_types
        assert TriggerType.DATA_DRIFT in trigger_types
        assert TriggerType.SCHEDULED in trigger_types

    def test_any_trigger_initiates_retrain(self):
        """Test that any trigger can initiate retraining."""
        pipeline = AutoRetrainPipeline(TriggerConfig())
        pipeline.configure_trigger()

        with patch.object(pipeline.triggers[0], 'should_retrain') as perf_trigger, \
             patch.object(pipeline.triggers[1], 'should_retrain') as drift_trigger, \
             patch.object(pipeline.triggers[2], 'should_retrain') as sched_trigger:

            # Only scheduled trigger fires
            perf_trigger.return_value = (False, "OK")
            drift_trigger.return_value = (False, "OK")
            sched_trigger.return_value = (True, "Scheduled retraining due")

            result = pipeline.check_retrain_needed("test_model")
            assert result is True


class TestJobTracking:
    """Test job tracking and history."""

    def test_job_history_retention(self):
        """Test that job history is properly maintained."""
        pipeline = AutoRetrainPipeline(TriggerConfig())

        with patch.object(pipeline, '_extract_training_data'), \
             patch.object(pipeline, '_submit_k8s_job'), \
             patch.object(pipeline, '_notify_slack'):

            # Submit multiple jobs
            job_ids = []
            for i in range(5):
                job_id = pipeline.start_retrain_job(f"model_{i}", {})
                job_ids.append(job_id)

            # Verify all jobs are tracked
            assert len(pipeline.job_history) == 5

            # Verify we can retrieve each job
            for job_id in job_ids:
                job = pipeline.get_job_status(job_id)
                assert job is not None
                assert job.job_id == job_id

    def test_job_listing_with_filtering(self):
        """Test job listing with model filtering."""
        pipeline = AutoRetrainPipeline(TriggerConfig())

        with patch.object(pipeline, '_extract_training_data'), \
             patch.object(pipeline, '_submit_k8s_job'), \
             patch.object(pipeline, '_notify_slack'):

            # Submit jobs for different models
            for model in ["model_a", "model_b", "model_a"]:
                pipeline.start_retrain_job(model, {})

            # List all jobs
            all_jobs = pipeline.list_recent_jobs(limit=10)
            assert len(all_jobs) == 3

            # Filter by model
            model_a_jobs = pipeline.list_recent_jobs("model_a", limit=10)
            assert len(model_a_jobs) == 2
            assert all(j.model_name == "model_a" for j in model_a_jobs)

            model_b_jobs = pipeline.list_recent_jobs("model_b", limit=10)
            assert len(model_b_jobs) == 1


class TestErrorRecovery:
    """Test error handling and recovery."""

    def test_invalid_job_id_handling(self):
        """Test graceful handling of invalid job IDs."""
        pipeline = AutoRetrainPipeline(TriggerConfig())

        # Attempt to get non-existent job
        job = pipeline.get_job_status("non_existent_job_id")
        assert job is None

    def test_missing_model_metrics_handling(self):
        """Test handling when model metrics are unavailable."""
        pipeline = AutoRetrainPipeline(TriggerConfig())

        with patch.object(pipeline, '_get_model_metrics') as mock_metrics:
            mock_metrics.return_value = {}  # Empty metrics

            # Should handle gracefully
            improvement = pipeline._calculate_improvement({}, {})
            assert improvement == 0.0

    def test_deployment_failure_recorded(self):
        """Test that deployment failures are properly recorded."""
        pipeline = AutoRetrainPipeline(TriggerConfig())

        with patch.object(pipeline, '_extract_training_data'), \
             patch.object(pipeline, '_submit_k8s_job'), \
             patch.object(pipeline, '_get_model_metrics') as mock_metrics, \
             patch.object(pipeline, '_promote_to_production') as mock_promote, \
             patch.object(pipeline, '_notify_slack'):

            mock_metrics.side_effect = [
                {"f1_score": 0.95},
                {"f1_score": 0.90}
            ]
            mock_promote.return_value = False  # Promotion fails

            job_id = pipeline.start_retrain_job("test_model", {})
            deployed = pipeline.deploy_if_better("test_model", job_id)

            assert deployed is False
            job = pipeline.get_job_status(job_id)
            assert job.status == RetrainingStatus.FAILED
            assert job.error_message is not None


class TestConfigurationValidation:
    """Test configuration validation and constraints."""

    def test_metric_threshold_bounds(self):
        """Test metric threshold validation."""
        # Valid values
        config = TriggerConfig(performance_metric_threshold=0.5)
        assert config.performance_metric_threshold == 0.5

        config = TriggerConfig(performance_metric_threshold=1.0)
        assert config.performance_metric_threshold == 1.0

        # Invalid values
        with pytest.raises(ValueError):
            TriggerConfig(performance_metric_threshold=1.5)

        with pytest.raises(ValueError):
            TriggerConfig(performance_metric_threshold=-0.1)

    def test_drift_threshold_bounds(self):
        """Test drift threshold validation."""
        # Valid values
        config = TriggerConfig(drift_threshold=0.0)
        assert config.drift_threshold == 0.0

        config = TriggerConfig(drift_threshold=1.0)
        assert config.drift_threshold == 1.0

        # Invalid values
        with pytest.raises(ValueError):
            TriggerConfig(drift_threshold=-0.1)

        with pytest.raises(ValueError):
            TriggerConfig(drift_threshold=1.1)

    def test_window_bounds(self):
        """Test training/evaluation window bounds."""
        # Valid values
        config = TriggerConfig(training_window_days=7)
        assert config.training_window_days == 7

        config = TriggerConfig(evaluation_window_days=365)
        assert config.evaluation_window_days == 365

        # Invalid values
        with pytest.raises(ValueError):
            TriggerConfig(training_window_days=0)

        with pytest.raises(ValueError):
            TriggerConfig(evaluation_window_days=1000)
