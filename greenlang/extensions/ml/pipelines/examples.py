"""
Examples and usage patterns for the AutoRetrainPipeline.

This module demonstrates how to:
    1. Configure triggers for different scenarios
    2. Monitor and manage retraining jobs
    3. Handle deployment decisions
    4. Integrate with monitoring and alerting systems
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from greenlang.ml.pipelines.auto_retrain import (
    AutoRetrainPipeline,
    TriggerConfig,
    TriggerType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Basic Setup with Default Configuration
# =============================================================================

def example_basic_setup():
    """
    Basic setup with default trigger configuration.

    This example shows the simplest way to get started with auto-retraining.
    """
    print("Example 1: Basic Setup")
    print("-" * 60)

    # Initialize pipeline with default configuration
    config = TriggerConfig()
    pipeline = AutoRetrainPipeline(
        config,
        mlflow_tracking_uri="http://localhost:5000",
        k8s_namespace="ml-platform"
    )

    # Configure triggers with default settings
    pipeline.configure_trigger(
        metric_threshold=0.92,      # Retrain if accuracy < 92%
        drift_threshold=0.25,       # Retrain if PSI > 0.25
        schedule="0 0 * * 0"        # Weekly on Sundays
    )

    # Check if retraining is needed for heat_predictor model
    model_name = "heat_predictor_v2"
    needs_retrain = pipeline.check_retrain_needed(model_name)

    print(f"Model: {model_name}")
    print(f"Retraining needed: {needs_retrain}")

    if needs_retrain:
        # Start retraining job
        job_id = pipeline.start_retrain_job(
            model_name,
            training_config={
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
            }
        )
        print(f"Retraining job started: {job_id}")


# =============================================================================
# Example 2: Multi-Model Monitoring
# =============================================================================

def example_multi_model_monitoring():
    """
    Monitor multiple Process Heat agent models simultaneously.

    This example shows how to manage retraining for GL-001 through GL-005
    models with different configurations.
    """
    print("\nExample 2: Multi-Model Monitoring")
    print("-" * 60)

    # Models to monitor
    models = [
        "GL-001-thermal-command",
        "GL-002-steam-optimization",
        "GL-003-waste-heat-recovery",
        "GL-004-process-efficiency",
        "GL-005-energy-cost-prediction"
    ]

    # Different configuration for each model
    configs = {
        "GL-001-thermal-command": {
            "metric_threshold": 0.94,
            "drift_threshold": 0.20,
            "training_window_days": 120
        },
        "GL-002-steam-optimization": {
            "metric_threshold": 0.92,
            "drift_threshold": 0.25,
            "training_window_days": 90
        },
        # ... more models
    }

    pipeline = AutoRetrainPipeline(TriggerConfig())
    pipeline.configure_trigger()

    # Check each model
    for model_name in models:
        needs_retrain = pipeline.check_retrain_needed(model_name)
        status = "NEEDS RETRAIN" if needs_retrain else "OK"
        print(f"{model_name:40} {status}")

        if needs_retrain:
            job_id = pipeline.start_retrain_job(model_name, configs.get(model_name, {}))
            print(f"  -> Job ID: {job_id}")


# =============================================================================
# Example 3: Job Monitoring and Tracking
# =============================================================================

def example_job_monitoring():
    """
    Monitor retraining job progress and status.

    This example shows how to track job execution and wait for completion.
    """
    print("\nExample 3: Job Monitoring and Tracking")
    print("-" * 60)

    pipeline = AutoRetrainPipeline(TriggerConfig())

    # Start a retraining job
    job_id = pipeline.start_retrain_job(
        "heat_predictor_v2",
        {
            "learning_rate": 0.001,
            "batch_size": 64,
            "early_stopping_patience": 5,
        }
    )

    print(f"Job submitted: {job_id}")

    # Poll job status (in real scenario, this would be async)
    max_wait_minutes = 30
    poll_interval_seconds = 10

    start_time = datetime.now()
    while True:
        job = pipeline.get_job_status(job_id)

        if job is None:
            print("Job not found!")
            break

        elapsed = (datetime.now() - start_time).total_seconds()
        print(
            f"[{elapsed:.0f}s] Status: {job.status.value:15} "
            f"Started: {job.started_at}"
        )

        if job.status.value in ["completed", "failed", "rolled_back"]:
            print(f"Job finished with status: {job.status.value}")
            if job.error_message:
                print(f"Error: {job.error_message}")
            break

        if elapsed > max_wait_minutes * 60:
            print("Timeout waiting for job completion")
            break

        # In real scenario, use proper async/await or APScheduler
        # time.sleep(poll_interval_seconds)


# =============================================================================
# Example 4: Deployment Decision Logic
# =============================================================================

def example_deployment_decision():
    """
    Demonstrate the deployment decision process.

    Shows how the pipeline compares new model with champion and decides
    whether to promote based on minimum improvement threshold.
    """
    print("\nExample 4: Deployment Decision Logic")
    print("-" * 60)

    pipeline = AutoRetrainPipeline(TriggerConfig())

    # Scenarios with different improvement percentages
    scenarios = [
        {
            "name": "Significant Improvement",
            "new_f1": 0.96,
            "champion_f1": 0.93,
            "min_improvement": 0.05,  # 5%
        },
        {
            "name": "Marginal Improvement",
            "new_f1": 0.94,
            "champion_f1": 0.93,
            "min_improvement": 0.05,  # 5%
        },
        {
            "name": "Degradation",
            "new_f1": 0.91,
            "champion_f1": 0.93,
            "min_improvement": 0.05,  # 5%
        },
    ]

    for scenario in scenarios:
        new_f1 = scenario["new_f1"]
        champion_f1 = scenario["champion_f1"]
        min_improvement = scenario["min_improvement"]

        improvement = (new_f1 - champion_f1) / champion_f1 if champion_f1 > 0 else 0
        improvement_pct = improvement * 100

        would_deploy = improvement >= min_improvement

        print(f"\n{scenario['name']}:")
        print(f"  New Model F1:       {new_f1:.4f}")
        print(f"  Champion F1:        {champion_f1:.4f}")
        print(f"  Improvement:        {improvement_pct:+.2f}%")
        print(f"  Min Threshold:      {min_improvement*100:.2f}%")
        print(f"  Decision:           {'DEPLOY' if would_deploy else 'REJECT'}")


# =============================================================================
# Example 5: Handling Multiple Trigger Types
# =============================================================================

def example_trigger_types():
    """
    Show how different trigger types evaluate independently.

    Demonstrates that multiple triggers can cause retraining for different reasons.
    """
    print("\nExample 5: Multiple Trigger Types")
    print("-" * 60)

    pipeline = AutoRetrainPipeline(TriggerConfig())
    pipeline.configure_trigger(
        metric_threshold=0.92,
        drift_threshold=0.25,
        schedule="0 0 * * 0"
    )

    model_name = "heat_predictor_v2"

    print(f"Evaluating model: {model_name}\n")

    trigger_results = []
    for trigger in pipeline.triggers:
        should_retrain, reason = trigger.should_retrain(model_name)
        trigger_type = trigger.get_trigger_type().value

        trigger_results.append((trigger_type, should_retrain, reason))

        print(f"Trigger: {trigger_type:25} | Retrain: {str(should_retrain):5} | {reason}")

    any_trigger = any(result[1] for result in trigger_results)
    print(f"\nOverall: {'RETRAIN NEEDED' if any_trigger else 'NO ACTION REQUIRED'}")


# =============================================================================
# Example 6: Job History and Analytics
# =============================================================================

def example_job_history():
    """
    Analyze retraining job history for a model.

    Shows how to extract insights from job history.
    """
    print("\nExample 6: Job History and Analytics")
    print("-" * 60)

    pipeline = AutoRetrainPipeline(TriggerConfig())

    # (In a real scenario, these would be actual jobs from the system)
    print("Recent jobs for 'heat_predictor_v2':\n")

    recent_jobs = pipeline.list_recent_jobs("heat_predictor_v2", limit=10)

    if not recent_jobs:
        print("No jobs found")
    else:
        for job in recent_jobs:
            print(
                f"Job ID:       {job.job_id[:8]}...\n"
                f"  Status:     {job.status.value}\n"
                f"  Trigger:    {job.trigger_type.value}\n"
                f"  Created:    {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  Deployed:   {job.deployed}\n"
            )

            if job.improvement_pct is not None:
                print(f"  Improvement: {job.improvement_pct*100:+.2f}%\n")


# =============================================================================
# Example 7: Error Handling and Recovery
# =============================================================================

def example_error_handling():
    """
    Demonstrate error handling and recovery patterns.

    Shows how to gracefully handle failures in the pipeline.
    """
    print("\nExample 7: Error Handling and Recovery")
    print("-" * 60)

    pipeline = AutoRetrainPipeline(TriggerConfig())

    try:
        # Attempt to get status of non-existent job
        job = pipeline.get_job_status("non_existent_job_id")

        if job is None:
            print("Job not found - continuing with alternative action")
            # Could trigger a new job or alert operator
        else:
            print(f"Job status: {job.status.value}")

    except Exception as e:
        print(f"Error retrieving job: {e}")

    # Validate configuration
    try:
        invalid_config = TriggerConfig(
            performance_metric_threshold=1.5  # Invalid: > 1.0
        )
    except ValueError as e:
        print(f"Configuration validation error caught: {e}")
        print("Using default configuration instead")

        valid_config = TriggerConfig()
        print(f"Using metric threshold: {valid_config.performance_metric_threshold}")


# =============================================================================
# Main Example Runner
# =============================================================================

def run_all_examples():
    """Run all examples."""
    print("=" * 60)
    print("Auto-Retrain Pipeline Examples")
    print("=" * 60)

    example_basic_setup()
    example_multi_model_monitoring()
    example_job_monitoring()
    example_deployment_decision()
    example_trigger_types()
    example_job_history()
    example_error_handling()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_examples()
