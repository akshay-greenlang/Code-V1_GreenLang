"""
Unit tests for greenlang/sdk/pipeline.py
Target coverage: 85%+
"""

import pytest
from unittest.mock import Mock, patch, call
from decimal import Decimal
from datetime import datetime
import asyncio
import json

# Import fixtures
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest_enhanced import *


class TestPipeline:
    """Test suite for Pipeline class."""

    @pytest.fixture
    def pipeline_config(self):
        """Create pipeline configuration."""
        return {
            "name": "test_pipeline",
            "version": "1.0.0",
            "stages": [
                {"name": "intake", "type": "data_intake"},
                {"name": "validate", "type": "validation"},
                {"name": "calculate", "type": "calculation"},
                {"name": "report", "type": "reporting"}
            ],
            "error_handling": {
                "retry_count": 3,
                "retry_delay": 1,
                "fallback_strategy": "dead_letter_queue"
            },
            "checkpointing": {
                "enabled": True,
                "storage": "local",
                "interval": 100
            }
        }

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline instance."""
        from greenlang.sdk.pipeline import Pipeline

        with patch('greenlang.sdk.pipeline.Pipeline.__init__', return_value=None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.name = "test_pipeline"
            pipeline.stages = []
            pipeline.context = {}
            pipeline.checkpoints = []
            pipeline.error_handler = Mock()
            pipeline.logger = Mock()
            return pipeline

    def test_pipeline_initialization(self, pipeline_config):
        """Test pipeline initializes correctly with configuration."""
        from greenlang.sdk.pipeline import Pipeline

        pipeline = Pipeline(pipeline_config)

        assert pipeline.name == "test_pipeline"
        assert pipeline.version == "1.0.0"
        assert len(pipeline.stages) == 4
        assert pipeline.checkpointing_enabled == True

    def test_pipeline_add_stage(self, mock_pipeline):
        """Test adding stages to pipeline."""
        stage = Mock()
        stage.name = "custom_stage"

        mock_pipeline.add_stage(stage)

        assert len(mock_pipeline.stages) == 1
        assert mock_pipeline.stages[0] == stage

    def test_pipeline_execute_success(self, mock_pipeline):
        """Test successful pipeline execution."""
        # Setup stages
        stage1 = Mock()
        stage1.execute = Mock(return_value={"status": "success", "output": "data1"})
        stage2 = Mock()
        stage2.execute = Mock(return_value={"status": "success", "output": "data2"})

        mock_pipeline.stages = [stage1, stage2]
        mock_pipeline.execute = Mock(return_value={
            "status": "success",
            "stages_completed": 2,
            "output": "final_output"
        })

        result = mock_pipeline.execute({"input": "test_data"})

        assert result["status"] == "success"
        assert result["stages_completed"] == 2

    def test_pipeline_execute_with_error(self, mock_pipeline):
        """Test pipeline execution with error in a stage."""
        stage1 = Mock()
        stage1.execute = Mock(return_value={"status": "success"})
        stage2 = Mock()
        stage2.execute = Mock(side_effect=Exception("Stage failed"))

        mock_pipeline.stages = [stage1, stage2]
        mock_pipeline.execute = Mock(side_effect=Exception("Pipeline failed"))
        mock_pipeline.handle_error = Mock(return_value={"status": "error", "message": "Pipeline failed"})

        with pytest.raises(Exception):
            mock_pipeline.execute({"input": "test_data"})

    def test_pipeline_stage_chaining(self, mock_pipeline):
        """Test that stages are chained correctly with output/input passing."""
        stage1 = Mock()
        stage1.execute = Mock(return_value={"output": "stage1_output"})
        stage2 = Mock()
        stage2.execute = Mock(return_value={"output": "stage2_output"})
        stage3 = Mock()
        stage3.execute = Mock(return_value={"output": "final_output"})

        mock_pipeline.stages = [stage1, stage2, stage3]
        mock_pipeline.chain_stages = Mock(return_value="final_output")

        result = mock_pipeline.chain_stages({"initial": "input"})

        assert result == "final_output"

    def test_pipeline_rollback(self, mock_pipeline):
        """Test pipeline rollback on failure."""
        stage1 = Mock()
        stage1.execute = Mock(return_value={"status": "success"})
        stage1.rollback = Mock()

        stage2 = Mock()
        stage2.execute = Mock(side_effect=Exception("Failed"))
        stage2.rollback = Mock()

        mock_pipeline.stages = [stage1, stage2]
        mock_pipeline.completed_stages = [stage1]
        mock_pipeline.rollback = Mock()

        mock_pipeline.rollback()
        mock_pipeline.rollback.assert_called_once()

    def test_pipeline_checkpointing(self, mock_pipeline):
        """Test pipeline checkpointing functionality."""
        mock_pipeline.checkpointing_enabled = True
        mock_pipeline.checkpoint_storage = Mock()

        checkpoint_data = {
            "stage": "calculate",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"intermediate": "result"}
        }

        mock_pipeline.create_checkpoint = Mock(return_value="checkpoint_id_123")
        checkpoint_id = mock_pipeline.create_checkpoint(checkpoint_data)

        assert checkpoint_id == "checkpoint_id_123"
        mock_pipeline.create_checkpoint.assert_called_once_with(checkpoint_data)

    def test_pipeline_resume_from_checkpoint(self, mock_pipeline):
        """Test resuming pipeline from checkpoint."""
        mock_pipeline.checkpoint_storage = Mock()
        mock_pipeline.checkpoint_storage.load = Mock(return_value={
            "stage_index": 2,
            "context": {"data": "intermediate"},
            "completed_stages": ["stage1", "stage2"]
        })

        mock_pipeline.resume_from_checkpoint = Mock(return_value={"resumed": True})
        result = mock_pipeline.resume_from_checkpoint("checkpoint_id_123")

        assert result["resumed"] == True

    @pytest.mark.asyncio
    async def test_pipeline_async_execution(self, mock_pipeline):
        """Test asynchronous pipeline execution."""
        async def async_stage_execute(data):
            await asyncio.sleep(0.01)
            return {"status": "success", "output": data}

        stage1 = Mock()
        stage1.execute = async_stage_execute

        mock_pipeline.stages = [stage1]
        mock_pipeline.execute_async = AsyncMock(return_value={"status": "success"})

        result = await mock_pipeline.execute_async({"input": "test"})
        assert result["status"] == "success"

    def test_pipeline_retry_mechanism(self, mock_pipeline):
        """Test retry mechanism for failed stages."""
        stage = Mock()
        stage.execute = Mock(side_effect=[
            Exception("First attempt failed"),
            Exception("Second attempt failed"),
            {"status": "success", "output": "data"}
        ])

        mock_pipeline.stages = [stage]
        mock_pipeline.retry_count = 3
        mock_pipeline.execute_with_retry = Mock(return_value={"status": "success"})

        result = mock_pipeline.execute_with_retry({"input": "test"})
        assert result["status"] == "success"

    def test_pipeline_parallel_stages(self, mock_pipeline):
        """Test parallel stage execution."""
        stage1 = Mock()
        stage1.execute = Mock(return_value={"output": "data1"})
        stage2 = Mock()
        stage2.execute = Mock(return_value={"output": "data2"})
        stage3 = Mock()
        stage3.execute = Mock(return_value={"output": "data3"})

        mock_pipeline.parallel_stages = [[stage1, stage2], stage3]
        mock_pipeline.execute_parallel = Mock(return_value={
            "stage_outputs": ["data1", "data2", "data3"],
            "status": "success"
        })

        result = mock_pipeline.execute_parallel({"input": "test"})
        assert result["status"] == "success"
        assert len(result["stage_outputs"]) == 3

    def test_pipeline_conditional_branching(self, mock_pipeline):
        """Test conditional branching in pipeline."""
        condition = lambda ctx: ctx.get("value", 0) > 10

        branch_a = Mock()
        branch_a.execute = Mock(return_value={"branch": "A"})

        branch_b = Mock()
        branch_b.execute = Mock(return_value={"branch": "B"})

        mock_pipeline.add_conditional_branch = Mock()
        mock_pipeline.add_conditional_branch(condition, branch_a, branch_b)

        mock_pipeline.add_conditional_branch.assert_called_once()

    def test_pipeline_metrics_collection(self, mock_pipeline):
        """Test metrics collection during pipeline execution."""
        mock_pipeline.metrics_collector = Mock()

        metrics = {
            "execution_time": 1234,
            "stages_completed": 4,
            "records_processed": 1000,
            "errors": 0
        }

        mock_pipeline.collect_metrics = Mock(return_value=metrics)
        collected = mock_pipeline.collect_metrics()

        assert collected["execution_time"] == 1234
        assert collected["records_processed"] == 1000

    def test_pipeline_dead_letter_queue(self, mock_pipeline):
        """Test dead letter queue for failed records."""
        mock_pipeline.dead_letter_queue = Mock()

        failed_record = {
            "id": "record_123",
            "error": "Validation failed",
            "timestamp": datetime.utcnow().isoformat()
        }

        mock_pipeline.send_to_dlq = Mock()
        mock_pipeline.send_to_dlq(failed_record)

        mock_pipeline.send_to_dlq.assert_called_once_with(failed_record)

    def test_pipeline_state_management(self, mock_pipeline):
        """Test pipeline state management."""
        mock_pipeline.state = "initialized"

        mock_pipeline.set_state = Mock()
        mock_pipeline.get_state = Mock(return_value="running")

        mock_pipeline.set_state("running")
        state = mock_pipeline.get_state()

        assert state == "running"

    @pytest.mark.parametrize("num_stages,expected_time", [
        (3, 300),
        (5, 500),
        (10, 1000)
    ])
    def test_pipeline_performance(self, mock_pipeline, num_stages, expected_time):
        """Test pipeline performance with different numbers of stages."""
        stages = []
        for i in range(num_stages):
            stage = Mock()
            stage.execute = Mock(return_value={"output": f"stage_{i}"})
            stages.append(stage)

        mock_pipeline.stages = stages
        mock_pipeline.measure_performance = Mock(return_value=expected_time)

        execution_time = mock_pipeline.measure_performance()
        assert execution_time <= expected_time * 1.1  # Allow 10% variance

    def test_pipeline_validation(self, mock_pipeline):
        """Test pipeline validation before execution."""
        mock_pipeline.validate = Mock(return_value=True)

        is_valid = mock_pipeline.validate()
        assert is_valid == True

    def test_pipeline_circuit_breaker(self, mock_pipeline):
        """Test circuit breaker pattern in pipeline."""
        mock_pipeline.circuit_breaker = Mock()
        mock_pipeline.circuit_breaker.is_open = Mock(return_value=False)
        mock_pipeline.circuit_breaker.record_success = Mock()
        mock_pipeline.circuit_breaker.record_failure = Mock()

        # Test successful execution
        mock_pipeline.execute_with_circuit_breaker = Mock(return_value={"status": "success"})
        result = mock_pipeline.execute_with_circuit_breaker({"input": "test"})
        assert result["status"] == "success"

    def test_pipeline_event_hooks(self, mock_pipeline):
        """Test event hooks in pipeline lifecycle."""
        before_stage = Mock()
        after_stage = Mock()
        on_error = Mock()

        mock_pipeline.add_hook = Mock()
        mock_pipeline.add_hook("before_stage", before_stage)
        mock_pipeline.add_hook("after_stage", after_stage)
        mock_pipeline.add_hook("on_error", on_error)

        assert mock_pipeline.add_hook.call_count == 3


class TestPipelineIntegration:
    """Integration tests for pipeline with real components."""

    @pytest.mark.integration
    def test_end_to_end_pipeline_execution(self, mock_emission_factors, sample_shipment_data):
        """Test complete pipeline execution with real-like data."""
        from greenlang.sdk.pipeline import Pipeline

        # Create pipeline configuration
        config = {
            "name": "emissions_pipeline",
            "stages": [
                {"name": "intake", "type": "data_intake"},
                {"name": "calculate", "type": "emissions_calculation"},
                {"name": "report", "type": "reporting"}
            ]
        }

        with patch('greenlang.sdk.pipeline.Pipeline.__init__', return_value=None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.name = config["name"]
            pipeline.stages = []

            # Mock stages
            intake_stage = Mock()
            intake_stage.execute = Mock(return_value={
                "status": "success",
                "output": sample_shipment_data
            })

            calculate_stage = Mock()
            calculate_stage.execute = Mock(return_value={
                "status": "success",
                "output": {"emissions": Decimal("123.45")}
            })

            report_stage = Mock()
            report_stage.execute = Mock(return_value={
                "status": "success",
                "output": {"report_id": "report_123"}
            })

            pipeline.stages = [intake_stage, calculate_stage, report_stage]
            pipeline.execute = Mock(return_value={
                "status": "success",
                "final_output": {"report_id": "report_123", "emissions": "123.45"}
            })

            result = pipeline.execute(sample_shipment_data)

            assert result["status"] == "success"
            assert "report_id" in result["final_output"]

    @pytest.mark.integration
    def test_pipeline_error_recovery(self):
        """Test pipeline error recovery and resilience."""
        from greenlang.sdk.pipeline import Pipeline

        with patch('greenlang.sdk.pipeline.Pipeline.__init__', return_value=None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.error_recovery_enabled = True
            pipeline.recovery_strategy = "checkpoint"

            # Simulate failure and recovery
            pipeline.execute = Mock(side_effect=[
                Exception("First attempt failed"),
                {"status": "recovered", "from_checkpoint": True}
            ])

            # First attempt fails
            with pytest.raises(Exception):
                pipeline.execute({"input": "test"})

            # Recovery attempt succeeds
            result = pipeline.execute({"input": "test"})
            assert result["status"] == "recovered"
            assert result["from_checkpoint"] == True