"""
Example: Pipeline with Checkpointing and Recovery

This example demonstrates how to use pipeline checkpointing to:
1. Save pipeline state after each stage
2. Resume from checkpoint after failures
3. Visualize checkpoint history
4. Use different storage strategies
"""

import json
import logging
from pathlib import Path
from greenlang.sdk.pipeline import Pipeline
from greenlang.pipeline.checkpointing import CheckpointManager, CheckpointStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_1_file_checkpointing():
    """Example 1: Basic file-based checkpointing"""
    print("\n" + "="*60)
    print("Example 1: File-based Checkpointing")
    print("="*60)

    # Create pipeline with checkpointing enabled
    pipeline = Pipeline(
        name="data_processing_pipeline",
        version="1.0",
        description="ETL pipeline with checkpoint support",
        checkpoint_enabled=True,
        checkpoint_strategy="file",
        checkpoint_config={
            "base_path": "/tmp/greenlang/checkpoints"
        },
        inputs={
            "source_data": "customer_data.csv",
            "output_format": "parquet"
        },
        steps=[
            {
                "name": "data_ingestion",
                "agent": "DataIngestionAgent",
                "config": {"batch_size": 1000}
            },
            {
                "name": "data_validation",
                "agent": "ValidationAgent",
                "inputs": {
                    "data": "$data_ingestion.output"
                }
            },
            {
                "name": "data_transformation",
                "agent": "TransformationAgent",
                "inputs": {
                    "validated_data": "$data_validation.output"
                }
            },
            {
                "name": "data_export",
                "agent": "ExportAgent",
                "inputs": {
                    "transformed_data": "$data_transformation.output"
                }
            }
        ]
    )

    # Execute pipeline (will create checkpoints)
    try:
        results = pipeline.execute(dry_run=True)
        print(f"\nPipeline completed successfully!")
        print(f"Checkpoints created: {len(results['checkpoints_created'])}")

        # Visualize progress
        print("\n" + pipeline.visualize_progress())

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")

        # Show checkpoint history
        history = pipeline.get_checkpoint_history()
        print(f"\nCheckpoint history: {len(history)} checkpoints")
        for checkpoint in history:
            print(f"  - {checkpoint['stage_name']}: {checkpoint['status']}")


def example_2_resume_from_checkpoint():
    """Example 2: Resume pipeline from checkpoint after failure"""
    print("\n" + "="*60)
    print("Example 2: Resume from Checkpoint")
    print("="*60)

    # Create pipeline that simulates failure
    pipeline = Pipeline(
        name="resilient_pipeline",
        version="1.0",
        checkpoint_enabled=True,
        checkpoint_strategy="file",
        checkpoint_config={
            "base_path": "/tmp/greenlang/checkpoints"
        },
        auto_resume=True,  # Automatically resume from checkpoint
        steps=[
            {
                "name": "step_1_completed",
                "agent": "Agent1"
            },
            {
                "name": "step_2_completed",
                "agent": "Agent2"
            },
            {
                "name": "step_3_will_fail",
                "agent": "FailingAgent"  # This will fail
            },
            {
                "name": "step_4_pending",
                "agent": "Agent4"
            }
        ]
    )

    # First execution (will fail at step 3)
    print("\n1. First execution attempt (will fail at step 3):")
    try:
        pipeline.execute(dry_run=True)
    except:
        print("   Pipeline failed as expected")
        print(pipeline.visualize_progress())

    # Resume from checkpoint
    print("\n2. Resuming from checkpoint:")
    # In real scenario, fix the failing agent first
    # Then resume execution
    pipeline_resumed = Pipeline(
        name="resilient_pipeline",
        version="1.0",
        checkpoint_enabled=True,
        checkpoint_strategy="file",
        checkpoint_config={
            "base_path": "/tmp/greenlang/checkpoints"
        },
        steps=pipeline.steps  # Same steps
    )

    # Resume execution (will skip completed steps)
    results = pipeline_resumed.execute(resume=True, dry_run=True)
    print("   Pipeline resumed and completed!")
    print(f"   Resume count: {results.get('resume_count', 0)}")


def example_3_redis_checkpointing():
    """Example 3: Redis-based checkpointing for distributed pipelines"""
    print("\n" + "="*60)
    print("Example 3: Redis Checkpointing (Distributed)")
    print("="*60)

    # Create pipeline with Redis checkpointing
    pipeline = Pipeline(
        name="distributed_ml_pipeline",
        version="2.0",
        checkpoint_enabled=True,
        checkpoint_strategy="redis",
        checkpoint_config={
            "host": "localhost",
            "port": 6379,
            "ttl_seconds": 3600  # 1 hour TTL
        },
        inputs={
            "model_type": "random_forest",
            "dataset": "training_data.parquet"
        },
        steps=[
            {
                "name": "data_preprocessing",
                "agent": "PreprocessingAgent"
            },
            {
                "name": "feature_engineering",
                "agent": "FeatureAgent"
            },
            {
                "name": "model_training",
                "agent": "TrainingAgent",
                "config": {
                    "distributed": True,
                    "num_workers": 4
                }
            },
            {
                "name": "model_evaluation",
                "agent": "EvaluationAgent"
            },
            {
                "name": "model_deployment",
                "agent": "DeploymentAgent"
            }
        ]
    )

    print("Pipeline configured for Redis checkpointing")
    print("This enables:")
    print("  - Fast in-memory checkpoint access")
    print("  - Distributed pipeline coordination")
    print("  - Automatic TTL-based cleanup")
    print("  - Cross-node checkpoint sharing")


def example_4_checkpoint_management():
    """Example 4: Checkpoint management and visualization"""
    print("\n" + "="*60)
    print("Example 4: Checkpoint Management")
    print("="*60)

    # Create checkpoint manager directly
    manager = CheckpointManager(
        strategy=CheckpointStrategy.FILE,
        base_path="/tmp/greenlang/checkpoints"
    )

    # Simulate creating checkpoints
    pipeline_id = "analytics_pipeline_20240101_120000"

    # Create checkpoints for different stages
    checkpoint_ids = []
    stages = ["data_load", "data_clean", "analysis", "reporting"]

    for i, stage in enumerate(stages):
        checkpoint_id = manager.create_checkpoint(
            pipeline_id=pipeline_id,
            stage_name=stage,
            stage_index=i,
            state_data={"stage": stage, "progress": f"{(i+1)*25}%"},
            completed_stages=stages[:i],
            pending_stages=stages[i+1:],
            agent_outputs={f"{stage}_output": f"Result from {stage}"},
            provenance_hashes={stage: f"hash_{stage}_abc123"}
        )
        checkpoint_ids.append(checkpoint_id)
        print(f"Created checkpoint: {checkpoint_id}")

    # Visualize checkpoint chain
    print("\n" + manager.visualize_checkpoint_chain(pipeline_id))

    # Get statistics
    stats = manager.get_statistics()
    print("\nCheckpoint Statistics:")
    print(json.dumps(stats, indent=2, default=str))

    # Export checkpoint for backup
    if checkpoint_ids:
        export_path = Path("/tmp/checkpoint_backup.json")
        if manager.export_checkpoint(checkpoint_ids[0], export_path):
            print(f"\nExported checkpoint to {export_path}")

    # Cleanup old checkpoints
    deleted_count = manager.auto_cleanup(retention_days=7)
    print(f"\nCleaned up {deleted_count} old checkpoints")


def example_5_database_checkpointing():
    """Example 5: PostgreSQL database checkpointing"""
    print("\n" + "="*60)
    print("Example 5: Database Checkpointing")
    print("="*60)

    # Create pipeline with database checkpointing
    pipeline = Pipeline(
        name="regulatory_compliance_pipeline",
        version="3.0",
        checkpoint_enabled=True,
        checkpoint_strategy="database",
        checkpoint_config={
            "connection_string": "postgresql://user:password@localhost/greenlang"
        },
        steps=[
            {
                "name": "data_collection",
                "agent": "DataCollectorAgent"
            },
            {
                "name": "compliance_validation",
                "agent": "ComplianceAgent"
            },
            {
                "name": "report_generation",
                "agent": "ReportAgent"
            },
            {
                "name": "audit_logging",
                "agent": "AuditAgent"
            }
        ]
    )

    print("Database checkpointing provides:")
    print("  - Persistent storage with ACID guarantees")
    print("  - Complex queries on checkpoint history")
    print("  - Integration with existing databases")
    print("  - Scalable for enterprise deployments")


def example_6_pause_and_resume():
    """Example 6: Pause and resume pipeline execution"""
    print("\n" + "="*60)
    print("Example 6: Pause and Resume")
    print("="*60)

    # Create long-running pipeline
    pipeline = Pipeline(
        name="batch_processing_pipeline",
        version="1.0",
        checkpoint_enabled=True,
        checkpoint_strategy="file",
        checkpoint_config={
            "base_path": "/tmp/greenlang/checkpoints"
        },
        steps=[
            {"name": "batch_1", "agent": "BatchAgent"},
            {"name": "batch_2", "agent": "BatchAgent"},
            {"name": "batch_3", "agent": "BatchAgent"},
            {"name": "batch_4", "agent": "BatchAgent"},
            {"name": "batch_5", "agent": "BatchAgent"}
        ]
    )

    print("Simulating pausable pipeline execution:")

    # In practice, you would run this in a separate thread/process
    # and pause based on external conditions (time, resources, etc.)

    # Simulate pause
    print("1. Pipeline started...")
    print("2. Processing batch_1...")
    print("3. Processing batch_2...")
    print("4. PAUSE requested - creating checkpoint...")

    # Pipeline would call pause() method
    # pipeline.pause()

    print("5. Pipeline paused at batch_2")
    print("6. System maintenance in progress...")
    print("7. Maintenance complete - resuming pipeline...")

    # Resume from checkpoint
    print("8. Resumed from batch_3...")
    print("9. Pipeline completed!")


def example_7_advanced_checkpoint_features():
    """Example 7: Advanced checkpoint features"""
    print("\n" + "="*60)
    print("Example 7: Advanced Features")
    print("="*60)

    # Create pipeline with advanced configuration
    pipeline_config = {
        "name": "advanced_pipeline",
        "version": "4.0",
        "checkpoint_enabled": True,
        "checkpoint_strategy": "file",
        "checkpoint_config": {
            "base_path": "/tmp/greenlang/checkpoints"
        },

        # Advanced checkpoint settings
        "checkpoint_after_each_step": True,  # Checkpoint after every step
        "auto_resume": True,                 # Auto-resume on restart

        "steps": [
            {
                "name": "initialization",
                "agent": "InitAgent",
                "checkpoint_policy": "always"  # Always checkpoint
            },
            {
                "name": "processing",
                "agent": "ProcessAgent",
                "checkpoint_policy": "on_success"  # Only checkpoint on success
            },
            {
                "name": "validation",
                "agent": "ValidateAgent",
                "checkpoint_policy": "on_failure"  # Only checkpoint on failure
            },
            {
                "name": "finalization",
                "agent": "FinalizeAgent",
                "checkpoint_policy": "never"  # Never checkpoint
            }
        ]
    }

    print("Advanced checkpoint features:")
    print("  - Conditional checkpointing (always/on_success/on_failure/never)")
    print("  - Checkpoint compression for large states")
    print("  - Encrypted checkpoints for sensitive data")
    print("  - Checkpoint versioning and migration")
    print("  - Distributed checkpoint coordination")
    print("  - Checkpoint performance monitoring")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GreenLang Pipeline Checkpointing Examples")
    print("="*60)

    # Run examples
    example_1_file_checkpointing()
    example_2_resume_from_checkpoint()
    example_3_redis_checkpointing()
    example_4_checkpoint_management()
    example_5_database_checkpointing()
    example_6_pause_and_resume()
    example_7_advanced_checkpoint_features()

    print("\n" + "="*60)
    print("Checkpoint Best Practices:")
    print("="*60)
    print("1. Enable checkpointing for long-running pipelines")
    print("2. Use file storage for development, Redis/DB for production")
    print("3. Set appropriate retention policies (7-30 days typical)")
    print("4. Monitor checkpoint storage usage")
    print("5. Test resume functionality regularly")
    print("6. Include checkpoint validation in CI/CD")
    print("7. Document checkpoint requirements for each pipeline")
    print("8. Use provenance hashes for audit compliance")
    print("\n")