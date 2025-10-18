"""
Comprehensive tests for BaseDataProcessor class.
Tests batch processing, parallel execution, error handling, and progress tracking.
"""

import pytest
from greenlang.agents.data_processor import (
    BaseDataProcessor, DataProcessorConfig, DataProcessorResult, ProcessingError
)
from typing import Dict, Any


class SimpleProcessor(BaseDataProcessor):
    """Simple processor that doubles values."""

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Double the value field."""
        record["value"] = record["value"] * 2
        return record


class ValidatingProcessor(BaseDataProcessor):
    """Processor with custom validation."""

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate that record has required fields."""
        return "id" in record and "value" in record

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process validated record."""
        record["processed"] = True
        return record


class FailingProcessor(BaseDataProcessor):
    """Processor that fails on specific records."""

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Fail if value is negative."""
        if record["value"] < 0:
            raise ValueError(f"Negative value: {record['value']}")
        return record


class TestDataProcessorConfig:
    """Test DataProcessorConfig model."""

    def test_default_config(self):
        """Test default configuration."""
        config = DataProcessorConfig(
            name="TestProcessor",
            description="Test"
        )
        assert config.batch_size == 1000
        assert config.parallel_workers == 1
        assert config.enable_progress is True
        assert config.collect_errors is True
        assert config.max_errors == 100
        assert config.validate_records is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = DataProcessorConfig(
            name="CustomProcessor",
            description="Test",
            batch_size=500,
            parallel_workers=4,
            max_errors=50
        )
        assert config.batch_size == 500
        assert config.parallel_workers == 4
        assert config.max_errors == 50

    def test_batch_size_validation(self):
        """Test batch size validation."""
        with pytest.raises(ValueError):
            DataProcessorConfig(
                name="Bad",
                description="Test",
                batch_size=0
            )

    def test_workers_validation(self):
        """Test parallel workers validation."""
        with pytest.raises(ValueError):
            DataProcessorConfig(
                name="Bad",
                description="Test",
                parallel_workers=0
            )

        with pytest.raises(ValueError):
            DataProcessorConfig(
                name="Bad",
                description="Test",
                parallel_workers=100  # > 32
            )


class TestBaseDataProcessor:
    """Test BaseDataProcessor functionality."""

    def test_simple_processing(self):
        """Test basic record processing."""
        processor = SimpleProcessor()
        records = [
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
            {"id": 3, "value": 30}
        ]

        result = processor.run({"records": records})

        assert result.success is True
        assert len(result.data["records"]) == 3
        assert result.data["records"][0]["value"] == 20
        assert result.data["records"][1]["value"] == 40
        assert result.data["records"][2]["value"] == 60

    def test_batch_creation(self):
        """Test creating batches from records."""
        config = DataProcessorConfig(
            name="BatchProcessor",
            description="Test",
            batch_size=2
        )
        processor = SimpleProcessor(config=config)

        records = [{"id": i, "value": i} for i in range(5)]
        batches = processor.create_batches(records)

        assert len(batches) == 3  # 5 records / batch_size 2 = 3 batches
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_batch_processing(self):
        """Test batch processing."""
        config = DataProcessorConfig(
            name="BatchProcessor",
            description="Test",
            batch_size=10
        )
        processor = SimpleProcessor(config=config)

        records = [{"id": i, "value": i} for i in range(25)]
        result = processor.run({"records": records})

        assert result.success is True
        assert result.records_processed == 25
        assert result.batches_processed == 3  # 25 / 10 = 3 batches

    def test_validation(self):
        """Test record validation."""
        processor = ValidatingProcessor()

        records = [
            {"id": 1, "value": 10},  # Valid
            {"id": 2},  # Invalid - missing value
            {"value": 30}  # Invalid - missing id
        ]

        result = processor.run({"records": records})

        assert result.records_processed == 1  # Only 1 valid record
        assert result.records_failed == 2
        assert len(result.errors) == 2

    def test_error_collection(self):
        """Test that errors are collected."""
        processor = FailingProcessor()

        records = [
            {"id": 1, "value": 10},
            {"id": 2, "value": -5},  # Will fail
            {"id": 3, "value": 20},
            {"id": 4, "value": -3}   # Will fail
        ]

        result = processor.run({"records": records})

        assert result.records_processed == 2
        assert result.records_failed == 2
        assert len(result.errors) == 2

        error = result.errors[0]
        assert error.record_id == 1  # Index of first failed record
        assert "Negative value" in error.error_message

    def test_max_errors_threshold(self):
        """Test that processing stops at max errors."""
        config = DataProcessorConfig(
            name="ErrorProcessor",
            description="Test",
            max_errors=2  # Stop after 2 errors
        )
        processor = FailingProcessor(config=config)

        # All negative - will all fail
        records = [{"id": i, "value": -i} for i in range(1, 10)]

        result = processor.run({"records": records})

        assert result.success is False
        assert result.records_failed >= 2
        assert "max errors" in result.error.lower()

    def test_parallel_processing(self):
        """Test parallel processing with multiple workers."""
        config = DataProcessorConfig(
            name="ParallelProcessor",
            description="Test",
            batch_size=5,
            parallel_workers=2  # Use 2 workers
        )
        processor = SimpleProcessor(config=config)

        records = [{"id": i, "value": i} for i in range(20)]
        result = processor.run({"records": records})

        assert result.success is True
        assert result.records_processed == 20
        # Results should be correct regardless of parallel execution
        assert all(r["value"] == r["id"] * 2 for r in result.data["records"])

    def test_sequential_processing(self):
        """Test sequential processing (workers=1)."""
        config = DataProcessorConfig(
            name="SequentialProcessor",
            description="Test",
            batch_size=5,
            parallel_workers=1  # Sequential
        )
        processor = SimpleProcessor(config=config)

        records = [{"id": i, "value": i} for i in range(15)]
        result = processor.run({"records": records})

        assert result.success is True
        assert result.records_processed == 15

    def test_empty_records(self):
        """Test processing with no records."""
        processor = SimpleProcessor()

        result = processor.run({"records": []})

        assert result.success is False
        assert "No records" in result.error

    def test_missing_records_key(self):
        """Test with missing records key."""
        processor = SimpleProcessor()

        result = processor.run({"data": "test"})

        assert result.success is False
        assert "records" in result.error.lower()

    def test_processing_stats(self):
        """Test getting processing statistics."""
        processor = SimpleProcessor()

        records = [{"id": i, "value": i} for i in range(10)]
        processor.run({"records": records})

        stats = processor.get_processing_stats()

        assert "processing" in stats
        assert stats["processing"]["records_processed"] == 10
        assert stats["processing"]["records_failed"] == 0
        assert stats["processing"]["success_rate"] == 100.0

    def test_validation_disabled(self):
        """Test with record validation disabled."""
        config = DataProcessorConfig(
            name="NoValidationProcessor",
            description="Test",
            validate_records=False
        )
        processor = ValidatingProcessor(config=config)

        # Invalid records but validation is off
        records = [{"invalid": "data"}]

        result = processor.run({"records": records})

        # Should process without validation
        assert result.success is True

    def test_metadata(self):
        """Test result metadata."""
        config = DataProcessorConfig(
            name="MetadataProcessor",
            description="Test",
            batch_size=5,
            parallel_workers=2
        )
        processor = SimpleProcessor(config=config)

        records = [{"id": i, "value": i} for i in range(12)]
        result = processor.run({"records": records})

        assert result.metadata["total_input_records"] == 12
        assert result.metadata["total_output_records"] == 12
        assert result.metadata["batch_size"] == 5
        assert result.metadata["parallel_workers"] == 2

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        processor = FailingProcessor()

        records = [
            {"id": 1, "value": 10},  # Success
            {"id": 2, "value": -5},  # Fail
            {"id": 3, "value": 20},  # Success
            {"id": 4, "value": 30}   # Success
        ]

        processor.run({"records": records})

        success_rate = processor._calculate_success_rate()
        assert success_rate == 75.0  # 3 success out of 4 total


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
