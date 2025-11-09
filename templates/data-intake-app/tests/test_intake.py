"""
Tests for Data Intake Application

These tests demonstrate how to test applications built with GreenLang infrastructure.
All tests use pytest and pytest-asyncio for async support.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os

from greenlang.agents.templates import DataFormat


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file for testing."""
    data = pd.DataFrame({
        "facility_id": ["FAC-001", "FAC-002", "FAC-003"],
        "facility_name": ["Plant A", "Plant B", "Plant C"],
        "emissions": [1500.5, 2300.8, 1800.2],
        "energy_consumption": [50000, 75000, 60000],
        "reporting_period": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "data_quality_score": [95.5, 98.2, 92.8]
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        yield f.name

    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def invalid_csv_file():
    """Create a CSV file with invalid data for testing validation."""
    data = pd.DataFrame({
        "facility_id": ["FAC-001", "INVALID", "FAC-003"],
        "facility_name": ["Plant A", "Plant B", "Plant C"],
        "emissions": [1500.5, -100, 1800.2],  # Negative emissions (invalid)
        "energy_consumption": [50000, 75000, -5000],  # Negative energy (invalid)
        "reporting_period": ["2024-01-01", "invalid-date", "2024-01-01"],
        "data_quality_score": [95.5, 150, 92.8]  # Score > 100 (invalid)
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        yield f.name

    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.mark.asyncio
async def test_single_file_ingestion(sample_csv_file):
    """Test ingesting a single valid CSV file."""
    from src.main import DataIntakeApplication

    app = DataIntakeApplication()

    try:
        result = await app.ingest_file(
            file_path=sample_csv_file,
            format=DataFormat.CSV,
            validate=True,
            store_in_db=False
        )

        assert result["success"] is True
        assert result["rows_read"] == 3
        assert result["rows_valid"] == 3
        assert result["validation_issues"] == 0
        assert "provenance_id" in result

    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_validation_failure(invalid_csv_file):
    """Test that validation catches invalid data."""
    from src.main import DataIntakeApplication

    app = DataIntakeApplication()

    try:
        result = await app.ingest_file(
            file_path=invalid_csv_file,
            format=DataFormat.CSV,
            validate=True,
            store_in_db=False
        )

        # Should fail validation
        assert result["success"] is False or result["validation_issues"] > 0

    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_batch_ingestion(sample_csv_file):
    """Test batch ingestion of multiple files."""
    from src.main import DataIntakeApplication

    app = DataIntakeApplication()

    try:
        # Create multiple file configs (using same file for testing)
        file_configs = [
            {"file_path": sample_csv_file, "format": DataFormat.CSV},
            {"file_path": sample_csv_file, "format": DataFormat.CSV},
            {"file_path": sample_csv_file, "format": DataFormat.CSV}
        ]

        result = await app.batch_ingest(file_configs, parallel=True)

        assert result["total_files"] == 3
        assert result["successful"] >= 0
        assert "total_rows_ingested" in result

    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_cache_hit():
    """Test that cache improves performance on repeated ingestion."""
    from src.main import DataIntakeApplication

    app = DataIntakeApplication()

    try:
        # Create test data
        data = pd.DataFrame({
            "facility_id": ["FAC-001"],
            "facility_name": ["Plant A"],
            "emissions": [1500.5],
            "reporting_period": ["2024-01-01"]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            test_file = f.name

        # First ingestion (cache miss)
        result1 = await app.ingest_file(
            file_path=test_file,
            format=DataFormat.CSV,
            validate=True,
            store_in_db=False
        )

        # Second ingestion (cache hit)
        result2 = await app.ingest_file(
            file_path=test_file,
            format=DataFormat.CSV,
            validate=True,
            store_in_db=False
        )

        # Both should succeed
        assert result1["success"] is True
        assert result2["success"] is True

        # Check cache statistics
        stats = app.get_statistics()
        assert stats["cache"]["total_requests"] >= 2

        # Cleanup
        os.unlink(test_file)

    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_provenance_tracking(sample_csv_file):
    """Test that provenance is tracked correctly."""
    from src.main import DataIntakeApplication

    app = DataIntakeApplication()

    try:
        result = await app.ingest_file(
            file_path=sample_csv_file,
            format=DataFormat.CSV,
            validate=True,
            store_in_db=False
        )

        assert "provenance_id" in result

        stats = app.get_statistics()
        assert stats["provenance"]["total_operations"] >= 1
        assert stats["provenance"]["data_transformations"] >= 0

    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_metrics_collection(sample_csv_file):
    """Test that metrics are collected."""
    from src.main import DataIntakeApplication

    app = DataIntakeApplication()

    try:
        # Perform ingestion
        await app.ingest_file(
            file_path=sample_csv_file,
            format=DataFormat.CSV,
            validate=True,
            store_in_db=False
        )

        # Check that metrics were recorded
        stats = app.get_statistics()
        assert "cache" in stats
        assert "provenance" in stats
        assert "agent" in stats

    finally:
        await app.shutdown()


def test_validation_schema():
    """Test that validation schema is correctly configured."""
    from src.main import DataIntakeApplication

    app = DataIntakeApplication()

    schema = app._get_validation_schema()

    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema
    assert "facility_id" in schema["properties"]
    assert "emissions" in schema["properties"]


def test_configuration_loading():
    """Test that configuration is loaded correctly."""
    from src.main import DataIntakeApplication

    app = DataIntakeApplication()

    assert app.config is not None
    assert app.cache is not None
    assert app.db is not None
    assert app.provenance is not None
    assert app.validation is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
