# -*- coding: utf-8 -*-
"""
Comprehensive tests for GreenLang BaseDataProcessor.

Tests cover:
- Data processor initialization and configuration
- Batch processing (sequential and parallel)
- Record validation and transformation
- Error collection and handling
- Progress tracking
- Batch size handling
- Max errors threshold
- Processing statistics
- Edge cases and performance
"""

import pytest
import time
from typing import Any, Dict, List
from unittest.mock import Mock, patch, MagicMock

from greenlang.agents.data_processor import (
    BaseDataProcessor,
    DataProcessorConfig,
    DataProcessorResult,
    ProcessingError,
)
from greenlang.agents.base import AgentResult


# Test Data Processor Implementations

class SimpleProcessor(BaseDataProcessor):
    """Simple processor that doubles a value."""

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Double the value field."""
        record["value"] = record.get("value", 0) * 2
        return record


class ValidationProcessor(BaseDataProcessor):
    """Processor with strict validation."""

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate record has required fields and positive value."""
        return "value" in record and record["value"] > 0

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process validated record."""
        record["validated"] = True
        return record


class FailingProcessor(BaseDataProcessor):
    """Processor that fails on specific conditions."""

    def __init__(self, fail_on_value=None, config=None):
        super().__init__(config)
        self.fail_on_value = fail_on_value

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Fail if record matches fail condition."""
        if self.fail_on_value is not None and record.get("value") == self.fail_on_value:
            raise ValueError(f"Failed on value {self.fail_on_value}")
        record["processed"] = True
        return record


class CountingProcessor(BaseDataProcessor):
    """Processor that counts operations."""

    def __init__(self, config=None):
        super().__init__(config)
        self.process_count = 0

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Count each processing operation."""
        self.process_count += 1
        record["count"] = self.process_count
        return record


class SlowProcessor(BaseDataProcessor):
    """Processor that simulates slow processing."""

    def __init__(self, delay_ms=10, config=None):
        super().__init__(config)
        self.delay_ms = delay_ms

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Add delay to simulate slow processing."""
        time.sleep(self.delay_ms / 1000.0)
        record["processed"] = True
        return record


# Test Classes

@pytest.mark.unit
class TestDataProcessorConfig:
    """Test DataProcessorConfig model."""

    def test_config_defaults(self):
        """Test config with default values."""
        config = DataProcessorConfig(
            name="TestProcessor",
            description="Test processor"
        )

        assert config.batch_size == 1000
        assert config.parallel_workers == 1
        assert config.enable_progress is True
        assert config.collect_errors is True
        assert config.max_errors == 100
        assert config.validate_records is True

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = DataProcessorConfig(
            name="CustomProcessor",
            description="Custom processor",
            batch_size=500,
            parallel_workers=4,
            enable_progress=False,
            collect_errors=False,
            max_errors=50,
            validate_records=False
        )

        assert config.batch_size == 500
        assert config.parallel_workers == 4
        assert config.enable_progress is False
        assert config.collect_errors is False
        assert config.max_errors == 50
        assert config.validate_records is False

    def test_config_validation_batch_size(self):
        """Test batch size validation."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DataProcessorConfig(
                name="Test",
                description="Test",
                batch_size=0
            )

        with pytest.raises(ValueError, match="batch_size must be positive"):
            DataProcessorConfig(
                name="Test",
                description="Test",
                batch_size=-10
            )

    def test_config_validation_workers(self):
        """Test parallel workers validation."""
        with pytest.raises(ValueError, match="parallel_workers must be positive"):
            DataProcessorConfig(
                name="Test",
                description="Test",
                parallel_workers=0
            )

        with pytest.raises(ValueError, match="cannot exceed 32"):
            DataProcessorConfig(
                name="Test",
                description="Test",
                parallel_workers=33
            )


@pytest.mark.unit
class TestProcessingError:
    """Test ProcessingError model."""

    def test_error_creation(self):
        """Test creating processing error."""
        error = ProcessingError(
            record_id=5,
            record_data={"value": 10},
            error_message="Test error"
        )

        assert error.record_id == 5
        assert error.record_data == {"value": 10}
        assert error.error_message == "Test error"
        assert error.timestamp is not None

    def test_error_with_string_id(self):
        """Test error with string record ID."""
        error = ProcessingError(
            record_id="rec-123",
            record_data={},
            error_message="Error"
        )

        assert error.record_id == "rec-123"


@pytest.mark.unit
class TestDataProcessorResult:
    """Test DataProcessorResult model."""

    def test_result_success(self):
        """Test successful processor result."""
        result = DataProcessorResult(
            success=True,
            data={"records": [{"value": 1}, {"value": 2}]},
            records_processed=2,
            records_failed=0
        )

        assert result.success is True
        assert result.records_processed == 2
        assert result.records_failed == 0
        assert result.errors == []
        assert result.batches_processed == 0

    def test_result_with_errors(self):
        """Test result with processing errors."""
        error1 = ProcessingError(
            record_id=1,
            record_data={"value": 1},
            error_message="Error 1"
        )
        error2 = ProcessingError(
            record_id=2,
            record_data={"value": 2},
            error_message="Error 2"
        )

        result = DataProcessorResult(
            success=False,
            data={},
            records_processed=5,
            records_failed=2,
            errors=[error1, error2]
        )

        assert result.success is False
        assert result.records_failed == 2
        assert len(result.errors) == 2


@pytest.mark.unit
class TestDataProcessorInitialization:
    """Test data processor initialization."""

    def test_initialization_defaults(self):
        """Test processor initializes with defaults."""
        processor = SimpleProcessor()

        assert processor.config is not None
        assert processor.config.batch_size == 1000
        assert processor.config.parallel_workers == 1
        assert isinstance(processor.config, DataProcessorConfig)

    def test_initialization_custom_config(self):
        """Test processor with custom config."""
        config = DataProcessorConfig(
            name="CustomProcessor",
            description="Custom",
            batch_size=250,
            parallel_workers=2
        )
        processor = SimpleProcessor(config)

        assert processor.config.batch_size == 250
        assert processor.config.parallel_workers == 2


@pytest.mark.unit
class TestRecordProcessing:
    """Test individual record processing."""

    def test_process_single_record(self):
        """Test processing a single record."""
        processor = SimpleProcessor()
        record = {"value": 5}
        result = processor.process_record(record)

        assert result["value"] == 10

    def test_transform_record_success(self):
        """Test transform_record with successful processing."""
        processor = SimpleProcessor()
        result = processor.transform_record({"value": 3}, 0)

        assert result["value"] == 6
        assert processor.stats.custom_counters["records_processed"] == 1

    def test_transform_record_with_error(self):
        """Test transform_record with error handling."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            collect_errors=True
        )
        processor = FailingProcessor(fail_on_value=5, config=config)

        result = processor.transform_record({"value": 5}, 0)

        assert "_error" in result
        assert result["_error"].error_message == "Failed on value 5"
        assert processor.stats.custom_counters["records_failed"] == 1

    def test_transform_record_raises_without_error_collection(self):
        """Test transform_record raises when error collection disabled."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            collect_errors=False
        )
        processor = FailingProcessor(fail_on_value=5, config=config)

        with pytest.raises(ValueError, match="Failed on value 5"):
            processor.transform_record({"value": 5}, 0)


@pytest.mark.unit
class TestRecordValidation:
    """Test record validation."""

    def test_validation_default_passes(self):
        """Test default validation passes all records."""
        processor = SimpleProcessor()
        assert processor.validate_record({"any": "data"}) is True
        assert processor.validate_record({}) is True

    def test_validation_custom_logic(self):
        """Test custom validation logic."""
        processor = ValidationProcessor()

        # Valid records
        assert processor.validate_record({"value": 10}) is True
        assert processor.validate_record({"value": 1}) is True

        # Invalid records
        assert processor.validate_record({}) is False
        assert processor.validate_record({"value": 0}) is False
        assert processor.validate_record({"value": -5}) is False

    def test_validation_failure_creates_error(self):
        """Test validation failure creates processing error."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            validate_records=True,
            collect_errors=True
        )
        processor = ValidationProcessor(config)

        result = processor.transform_record({"invalid": "data"}, 0)

        assert "_error" in result
        assert "validation failed" in result["_error"].error_message.lower()

    def test_validation_disabled(self):
        """Test processing with validation disabled."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            validate_records=False
        )
        processor = ValidationProcessor(config)

        # Should process even invalid records when validation disabled
        result = processor.transform_record({"invalid": "data"}, 0)
        assert "validated" in result


@pytest.mark.unit
class TestBatchProcessing:
    """Test batch processing functionality."""

    def test_create_batches(self):
        """Test creating batches from records."""
        processor = SimpleProcessor()
        records = [{"value": i} for i in range(10)]

        processor.config.batch_size = 3
        batches = processor.create_batches(records)

        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_create_batches_exact_fit(self):
        """Test batches when records divide evenly."""
        processor = SimpleProcessor()
        records = [{"value": i} for i in range(12)]

        processor.config.batch_size = 4
        batches = processor.create_batches(records)

        assert len(batches) == 3
        assert all(len(batch) == 4 for batch in batches)

    def test_process_batch(self):
        """Test processing a single batch."""
        processor = SimpleProcessor()
        batch = [{"value": 1}, {"value": 2}, {"value": 3}]

        results = processor.process_batch(batch, batch_index=0)

        assert len(results) == 3
        assert results[0]["value"] == 2
        assert results[1]["value"] == 4
        assert results[2]["value"] == 6
        assert processor.stats.custom_counters["batches_processed"] == 1

    def test_process_batch_with_errors(self):
        """Test batch processing with some errors."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            collect_errors=True
        )
        processor = FailingProcessor(fail_on_value=2, config=config)
        batch = [{"value": 1}, {"value": 2}, {"value": 3}]

        results = processor.process_batch(batch, batch_index=0)

        assert len(results) == 3
        assert results[0]["processed"] is True
        assert "_error" in results[1]
        assert results[2]["processed"] is True


@pytest.mark.unit
class TestSequentialProcessing:
    """Test sequential batch processing."""

    def test_sequential_processing(self):
        """Test processing batches sequentially."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=3,
            parallel_workers=1,
            enable_progress=False
        )
        processor = SimpleProcessor(config)

        records = [{"value": i} for i in range(10)]
        batches = processor.create_batches(records)

        results = processor.process_batches_sequential(batches)

        assert len(results) == 10
        assert all(r["value"] == i * 2 for i, r in enumerate(results))

    def test_sequential_processing_error_threshold(self):
        """Test sequential processing stops at error threshold."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=2,
            max_errors=3,
            enable_progress=False,
            collect_errors=True
        )
        processor = FailingProcessor(fail_on_value=1, config=config)

        # Create records that will fail
        records = [{"value": 1}] * 10  # All will fail
        batches = processor.create_batches(records)

        results = processor.process_batches_sequential(batches)

        # Should stop after max_errors reached
        errors = processor.stats.custom_counters.get("records_failed", 0)
        assert errors >= config.max_errors


@pytest.mark.unit
class TestParallelProcessing:
    """Test parallel batch processing."""

    def test_parallel_processing(self):
        """Test processing batches in parallel."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=5,
            parallel_workers=2,
            enable_progress=False
        )
        processor = SimpleProcessor(config)

        records = [{"value": i} for i in range(20)]
        batches = processor.create_batches(records)

        results = processor.process_batches_parallel(batches)

        assert len(results) == 20
        # Results may be out of order due to parallel processing
        values = sorted([r["value"] for r in results])
        assert values == [i * 2 for i in range(20)]

    def test_parallel_processing_error_threshold(self):
        """Test parallel processing stops at error threshold."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=2,
            parallel_workers=2,
            max_errors=5,
            enable_progress=False,
            collect_errors=True
        )
        processor = FailingProcessor(fail_on_value=1, config=config)

        # Create records that will fail
        records = [{"value": 1}] * 20
        batches = processor.create_batches(records)

        results = processor.process_batches_parallel(batches)

        # Should stop after max_errors reached
        errors = processor.stats.custom_counters.get("records_failed", 0)
        assert errors >= config.max_errors


@pytest.mark.unit
class TestDataProcessorExecution:
    """Test complete execution flow."""

    def test_execute_success(self):
        """Test successful execution."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=5,
            enable_progress=False
        )
        processor = SimpleProcessor(config)

        records = [{"value": i} for i in range(10)]
        result = processor.run({"records": records})

        assert result.success is True
        assert isinstance(result, DataProcessorResult)
        assert result.records_processed == 10
        assert result.records_failed == 0
        assert len(result.data["records"]) == 10

    def test_execute_with_validation_errors(self):
        """Test execution with validation errors."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=3,
            enable_progress=False,
            collect_errors=True,
            max_errors=100
        )
        processor = ValidationProcessor(config)

        records = [
            {"value": 1},   # Valid
            {"value": 0},   # Invalid
            {"value": 2},   # Valid
            {"value": -1},  # Invalid
            {"value": 3},   # Valid
        ]
        result = processor.run({"records": records})

        assert result.success is True
        assert result.records_processed == 3
        assert result.records_failed == 2
        assert len(result.errors) == 2
        assert len(result.data["records"]) == 3

    def test_execute_exceeds_max_errors(self):
        """Test execution fails when exceeding max errors."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=2,
            max_errors=2,
            enable_progress=False,
            collect_errors=True
        )
        processor = FailingProcessor(fail_on_value=1, config=config)

        records = [{"value": 1}] * 10  # All will fail
        result = processor.run({"records": records})

        assert result.success is False
        assert "max" in result.error.lower()
        assert result.records_failed >= config.max_errors

    def test_execute_no_records(self):
        """Test execution with empty records."""
        processor = SimpleProcessor()
        result = processor.run({"records": []})

        assert result.success is False
        assert "no records" in result.error.lower()

    def test_execute_missing_records_key(self):
        """Test execution without records key."""
        processor = SimpleProcessor()
        result = processor.run({"other": "data"})

        assert result.success is False


@pytest.mark.unit
class TestInputValidation:
    """Test input validation."""

    def test_validate_input_success(self):
        """Test input validation with valid input."""
        processor = SimpleProcessor()
        assert processor.validate_input({"records": []}) is True
        assert processor.validate_input({"records": [{"value": 1}]}) is True

    def test_validate_input_missing_records(self):
        """Test validation fails without records key."""
        processor = SimpleProcessor()
        assert processor.validate_input({}) is False
        assert processor.validate_input({"other": "data"}) is False

    def test_validate_input_wrong_type(self):
        """Test validation fails with wrong records type."""
        processor = SimpleProcessor()
        assert processor.validate_input({"records": "not a list"}) is False
        assert processor.validate_input({"records": 123}) is False
        assert processor.validate_input({"records": {"key": "value"}}) is False


@pytest.mark.unit
class TestProcessingStatistics:
    """Test processing statistics."""

    def test_get_processing_stats(self):
        """Test getting detailed processing statistics."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=3,
            enable_progress=False
        )
        processor = SimpleProcessor(config)

        records = [{"value": i} for i in range(10)]
        result = processor.run({"records": records})

        stats = processor.get_processing_stats()

        assert "processing" in stats
        assert stats["processing"]["records_processed"] == 10
        assert stats["processing"]["records_failed"] == 0
        assert stats["processing"]["batches_processed"] == 4
        assert stats["processing"]["success_rate"] == 100.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=2,
            enable_progress=False,
            collect_errors=True,
            max_errors=100
        )
        processor = ValidationProcessor(config)

        records = [
            {"value": 1},   # Valid
            {"value": 0},   # Invalid
            {"value": 2},   # Valid
            {"value": 0},   # Invalid
        ]
        result = processor.run({"records": records})

        stats = processor.get_processing_stats()

        # 2 processed, 2 failed = 50% success rate
        assert stats["processing"]["success_rate"] == 50.0

    def test_success_rate_zero_records(self):
        """Test success rate with no records processed."""
        processor = SimpleProcessor()
        # Don't run any processing

        success_rate = processor._calculate_success_rate()
        assert success_rate == 0.0


@pytest.mark.unit
class TestProgressTracking:
    """Test progress tracking functionality."""

    @patch('greenlang.agents.data_processor.tqdm')
    def test_progress_enabled(self, mock_tqdm):
        """Test progress bar is shown when enabled."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=2,
            enable_progress=True
        )
        processor = SimpleProcessor(config)

        records = [{"value": i} for i in range(6)]
        result = processor.run({"records": records})

        # tqdm should be called
        assert mock_tqdm.called

    def test_progress_disabled(self):
        """Test progress bar is not shown when disabled."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=2,
            enable_progress=False
        )
        processor = SimpleProcessor(config)

        records = [{"value": i} for i in range(6)]
        result = processor.run({"records": records})

        # No assertions needed - just verify it runs without progress bar


@pytest.mark.unit
class TestMetadata:
    """Test result metadata."""

    def test_result_metadata(self):
        """Test result contains correct metadata."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=5,
            parallel_workers=2,
            enable_progress=False
        )
        processor = SimpleProcessor(config)

        records = [{"value": i} for i in range(12)]
        result = processor.run({"records": records})

        assert result.metadata["total_input_records"] == 12
        assert result.metadata["total_output_records"] == 12
        assert result.metadata["batch_size"] == 5
        assert result.metadata["parallel_workers"] == 2


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_record(self):
        """Test processing a single record."""
        processor = SimpleProcessor()
        result = processor.run({"records": [{"value": 5}]})

        assert result.success is True
        assert result.records_processed == 1
        assert len(result.data["records"]) == 1

    def test_large_batch_size(self):
        """Test with batch size larger than records."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=1000,
            enable_progress=False
        )
        processor = SimpleProcessor(config)

        records = [{"value": i} for i in range(10)]
        result = processor.run({"records": records})

        assert result.success is True
        assert result.records_processed == 10
        assert result.batches_processed == 1

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=1,
            enable_progress=False
        )
        processor = SimpleProcessor(config)

        records = [{"value": i} for i in range(5)]
        result = processor.run({"records": records})

        assert result.success is True
        assert result.records_processed == 5
        assert result.batches_processed == 5

    def test_concurrent_executions(self):
        """Test multiple concurrent processor executions."""
        import threading

        processor = SimpleProcessor()
        results = []
        errors = []

        def run_processor(count):
            try:
                records = [{"value": i} for i in range(count)]
                result = processor.run({"records": records})
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_processor, args=(i+1,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert len(errors) == 0

    def test_processing_maintains_order(self):
        """Test sequential processing maintains record order."""
        config = DataProcessorConfig(
            name="Test",
            description="Test",
            batch_size=3,
            parallel_workers=1,  # Sequential
            enable_progress=False
        )
        processor = CountingProcessor(config)

        records = [{"id": i} for i in range(10)]
        result = processor.run({"records": records})

        # Check order is maintained
        assert result.success is True
        for i, record in enumerate(result.data["records"]):
            assert record["id"] == i
            assert record["count"] == i + 1

    def test_empty_record_fields(self):
        """Test processing records with empty fields."""
        processor = SimpleProcessor()
        records = [
            {},
            {"value": 0},
            {"other": "field"},
        ]
        result = processor.run({"records": records})

        assert result.success is True
        assert result.records_processed == 3

    def test_parallel_performance(self):
        """Test parallel processing is faster than sequential."""
        config_sequential = DataProcessorConfig(
            name="Sequential",
            description="Test",
            batch_size=5,
            parallel_workers=1,
            enable_progress=False
        )
        config_parallel = DataProcessorConfig(
            name="Parallel",
            description="Test",
            batch_size=5,
            parallel_workers=4,
            enable_progress=False
        )

        records = [{"value": i} for i in range(20)]

        # Sequential
        processor_seq = SlowProcessor(delay_ms=10, config=config_sequential)
        start_seq = time.time()
        result_seq = processor_seq.run({"records": records})
        time_seq = time.time() - start_seq

        # Parallel
        processor_par = SlowProcessor(delay_ms=10, config=config_parallel)
        start_par = time.time()
        result_par = processor_par.run({"records": records})
        time_par = time.time() - start_par

        assert result_seq.success is True
        assert result_par.success is True
        # Parallel should be noticeably faster (with some tolerance)
        assert time_par < time_seq * 0.8
