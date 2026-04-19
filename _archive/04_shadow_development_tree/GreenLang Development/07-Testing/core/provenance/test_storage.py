# -*- coding: utf-8 -*-
"""
Tests for Provenance Storage

Tests the persistent storage of calculation provenance records for
audit trail queries and compliance reporting.

Author: GreenLang Team
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

from greenlang.core.provenance import (
    CalculationProvenance,
    OperationType,
)
from greenlang.core.provenance.storage import (
    SQLiteProvenanceStorage,
    create_storage,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def storage(temp_db):
    """Create a storage instance for testing."""
    return SQLiteProvenanceStorage(db_path=temp_db)


@pytest.fixture
def sample_provenance():
    """Create a sample provenance record for testing."""
    provenance = CalculationProvenance.create(
        agent_name="EmissionsCalculator",
        agent_version="1.0.0",
        calculation_type="scope1_emissions",
        input_data={"fuel_kg": 1000, "fuel_type": "natural_gas"},
        standards_applied=["GHG Protocol"],
        data_sources=["EPA eGRID 2023"],
    )

    provenance.add_step(
        operation=OperationType.LOOKUP,
        description="Lookup emission factor",
        inputs={"fuel_type": "natural_gas"},
        output=0.18414,
        data_source="EPA eGRID 2023",
    )

    provenance.add_step(
        operation=OperationType.MULTIPLY,
        description="Calculate emissions",
        inputs={"fuel_kg": 1000, "ef": 0.18414},
        output=184.14,
        formula="emissions = fuel_kg * ef",
    )

    provenance.finalize(output_data={"emissions_kg_co2": 184.14})

    return provenance


class TestSQLiteProvenanceStorage:
    """Tests for SQLite provenance storage."""

    def test_init_creates_database(self, temp_db):
        """Test that initializing storage creates database."""
        storage = SQLiteProvenanceStorage(db_path=temp_db)

        assert os.path.exists(temp_db)
        assert Path(temp_db).stat().st_size > 0

    def test_store_provenance(self, storage, sample_provenance):
        """Test storing a provenance record."""
        calc_id = storage.store(sample_provenance)

        assert calc_id == sample_provenance.calculation_id
        assert len(calc_id) == 16

    def test_retrieve_provenance(self, storage, sample_provenance):
        """Test retrieving a stored provenance record."""
        # Store
        calc_id = storage.store(sample_provenance)

        # Retrieve
        retrieved = storage.retrieve(calc_id)

        assert retrieved is not None
        assert retrieved.calculation_id == sample_provenance.calculation_id
        assert retrieved.metadata.agent_name == sample_provenance.metadata.agent_name
        assert retrieved.input_data == sample_provenance.input_data
        assert len(retrieved.steps) == len(sample_provenance.steps)

    def test_retrieve_nonexistent(self, storage):
        """Test retrieving a non-existent record."""
        result = storage.retrieve("nonexistent_id")

        assert result is None

    def test_delete_provenance(self, storage, sample_provenance):
        """Test deleting a provenance record."""
        # Store
        calc_id = storage.store(sample_provenance)

        # Delete
        deleted = storage.delete(calc_id)
        assert deleted is True

        # Verify deleted
        retrieved = storage.retrieve(calc_id)
        assert retrieved is None

    def test_delete_nonexistent(self, storage):
        """Test deleting a non-existent record."""
        deleted = storage.delete("nonexistent_id")

        assert deleted is False

    def test_query_all(self, storage):
        """Test querying all records."""
        # Store multiple records
        for i in range(5):
            prov = CalculationProvenance.create(
                agent_name=f"Agent{i}",
                agent_version="1.0.0",
                calculation_type="test",
                input_data={"x": i},
            )
            prov.finalize(output_data={"result": i * 2})
            storage.store(prov)

        # Query all
        results = storage.query(limit=100)

        assert len(results) == 5

    def test_query_by_agent_name(self, storage):
        """Test querying by agent name."""
        # Store records with different agents (unique inputs to avoid ID collision)
        for i, agent_name in enumerate(["AgentA", "AgentB", "AgentA"]):
            prov = CalculationProvenance.create(
                agent_name=agent_name,
                agent_version="1.0.0",
                calculation_type="test",
                input_data={"index": i},  # Make input unique
            )
            prov.finalize(output_data={})
            storage.store(prov)

        # Query by agent name
        results = storage.query(agent_name="AgentA")

        assert len(results) == 2
        assert all(r.metadata.agent_name == "AgentA" for r in results)

    def test_query_by_calculation_type(self, storage):
        """Test querying by calculation type."""
        # Store records with different types (unique inputs to avoid ID collision)
        for i, calc_type in enumerate(["type1", "type2", "type1"]):
            prov = CalculationProvenance.create(
                agent_name="TestAgent",
                agent_version="1.0.0",
                calculation_type=calc_type,
                input_data={"index": i},  # Make input unique
            )
            prov.finalize(output_data={})
            storage.store(prov)

        # Query by calculation type
        results = storage.query(calculation_type="type1")

        assert len(results) == 2
        assert all(r.metadata.calculation_type == "type1" for r in results)

    def test_query_with_limit_and_offset(self, storage):
        """Test pagination with limit and offset."""
        # Store 10 records
        for i in range(10):
            prov = CalculationProvenance.create(
                agent_name="TestAgent",
                agent_version="1.0.0",
                calculation_type="test",
                input_data={"index": i},
            )
            prov.finalize(output_data={})
            storage.store(prov)

        # Query first page
        page1 = storage.query(limit=3, offset=0)
        assert len(page1) == 3

        # Query second page
        page2 = storage.query(limit=3, offset=3)
        assert len(page2) == 3

        # Ensure different results
        assert page1[0].calculation_id != page2[0].calculation_id

    def test_query_by_errors(self, storage):
        """Test querying records with/without errors."""
        # Store record with error (unique input)
        prov1 = CalculationProvenance.create(
            agent_name="TestAgent",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={"with_error": True},  # Make input unique
        )
        prov1.add_error("Test error")
        prov1.finalize(output_data={})
        storage.store(prov1)

        # Store record without error (unique input)
        prov2 = CalculationProvenance.create(
            agent_name="TestAgent",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={"with_error": False},  # Make input unique
        )
        prov2.finalize(output_data={})
        storage.store(prov2)

        # Query records with errors
        with_errors = storage.query(has_errors=True)
        assert len(with_errors) == 1

        # Query records without errors
        without_errors = storage.query(has_errors=False)
        assert len(without_errors) == 1

    def test_find_by_input_hash(self, storage):
        """Test finding calculations with same inputs."""
        input_data = {"x": 100, "y": 200}

        # Store two calculations with same inputs
        for i in range(2):
            prov = CalculationProvenance.create(
                agent_name=f"Agent{i}",
                agent_version="1.0.0",
                calculation_type="test",
                input_data=input_data,
            )
            prov.finalize(output_data={"result": i})
            storage.store(prov)

        # Find by input hash
        duplicates = storage.find_by_input_hash(
            CalculationProvenance.create(
                agent_name="Temp",
                agent_version="1.0.0",
                calculation_type="test",
                input_data=input_data,
            ).input_hash
        )

        assert len(duplicates) == 2

    def test_find_by_data_source(self, storage):
        """Test finding calculations using specific data source."""
        # Store records with different data sources
        prov1 = CalculationProvenance.create(
            agent_name="Agent1",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={},
        )
        prov1.add_step(
            operation="lookup",
            description="Step 1",
            inputs={},
            output=1,
            data_source="EPA eGRID 2023",
        )
        prov1.finalize(output_data={})
        storage.store(prov1)

        prov2 = CalculationProvenance.create(
            agent_name="Agent2",
            agent_version="1.0.0",
            calculation_type="test",
            input_data={},
        )
        prov2.add_step(
            operation="lookup",
            description="Step 2",
            inputs={},
            output=2,
            data_source="DEFRA 2024",
        )
        prov2.finalize(output_data={})
        storage.store(prov2)

        # Find by data source
        results = storage.find_by_data_source("EPA eGRID 2023")

        assert len(results) == 1
        assert results[0].metadata.agent_name == "Agent1"

    def test_get_statistics(self, storage):
        """Test getting storage statistics."""
        # Store some records
        for i in range(5):
            prov = CalculationProvenance.create(
                agent_name=f"Agent{i % 2}",  # 2 unique agents
                agent_version="1.0.0",
                calculation_type=f"type{i % 3}",  # 3 types
                input_data={"x": i},
            )
            if i == 0:
                prov.add_error("Test error")
            if i == 1:
                prov.add_warning("Test warning")
            prov.finalize(output_data={})
            storage.store(prov)

        stats = storage.get_statistics()

        assert stats["total_calculations"] == 5
        assert stats["unique_agents"] == 2
        assert stats["records_with_errors"] == 1
        assert stats["records_with_warnings"] == 1
        assert "calculation_types" in stats
        assert "average_duration_ms" in stats

    def test_export_audit_report(self, storage, sample_provenance, temp_db):
        """Test exporting audit report."""
        # Store record
        storage.store(sample_provenance)

        # Export
        report_path = temp_db.replace(".db", "_report.json")
        result_path = storage.export_audit_report(report_path)

        assert os.path.exists(result_path)

        # Read and verify
        import json
        with open(result_path) as f:
            report = json.load(f)

        assert "generated_at" in report
        assert "statistics" in report
        assert "records" in report
        assert len(report["records"]) == 1

        # Cleanup
        os.unlink(result_path)

    def test_update_existing_record(self, storage, sample_provenance):
        """Test updating an existing provenance record."""
        # Store original
        calc_id = storage.store(sample_provenance)

        # Modify and store again
        sample_provenance.add_step(
            operation="add",
            description="Additional step",
            inputs={},
            output=100,
        )
        storage.store(sample_provenance)

        # Retrieve
        retrieved = storage.retrieve(calc_id)

        # Should have updated steps
        assert len(retrieved.steps) == 3  # Original 2 + 1 new

    def test_roundtrip_preserves_data(self, storage, sample_provenance):
        """Test that store/retrieve roundtrip preserves all data."""
        # Store
        storage.store(sample_provenance)

        # Retrieve
        retrieved = storage.retrieve(sample_provenance.calculation_id)

        # Compare all fields
        assert retrieved.calculation_id == sample_provenance.calculation_id
        assert retrieved.metadata.agent_name == sample_provenance.metadata.agent_name
        assert retrieved.metadata.agent_version == sample_provenance.metadata.agent_version
        assert retrieved.metadata.calculation_type == sample_provenance.metadata.calculation_type
        assert retrieved.input_data == sample_provenance.input_data
        assert retrieved.input_hash == sample_provenance.input_hash
        assert retrieved.output_data == sample_provenance.output_data
        assert retrieved.output_hash == sample_provenance.output_hash
        assert len(retrieved.steps) == len(sample_provenance.steps)

        # Verify integrity still valid
        assert retrieved.verify_integrity()["input_hash_valid"]
        assert retrieved.verify_integrity()["output_hash_valid"]


class TestStorageFactory:
    """Tests for storage factory function."""

    def test_create_sqlite_storage(self, temp_db):
        """Test creating SQLite storage via factory."""
        storage = create_storage(storage_type="sqlite", db_path=temp_db)

        assert isinstance(storage, SQLiteProvenanceStorage)
        assert os.path.exists(temp_db)

    def test_create_unsupported_storage(self):
        """Test creating unsupported storage type."""
        with pytest.raises(ValueError, match="Unknown storage type"):
            create_storage(storage_type="unsupported")


class TestStoragePerformance:
    """Performance tests for storage operations."""

    def test_bulk_store(self, storage):
        """Test storing many records."""
        # Store 100 records
        for i in range(100):
            prov = CalculationProvenance.create(
                agent_name=f"Agent{i}",
                agent_version="1.0.0",
                calculation_type="test",
                input_data={"index": i},
            )
            prov.finalize(output_data={"result": i * 2})
            storage.store(prov)

        # Verify all stored
        results = storage.query(limit=200)
        assert len(results) == 100

    def test_query_performance(self, storage):
        """Test query performance with many records."""
        # Store records
        for i in range(50):
            prov = CalculationProvenance.create(
                agent_name="TestAgent",
                agent_version="1.0.0",
                calculation_type="test",
                input_data={"x": i},
            )
            prov.finalize(output_data={})
            storage.store(prov)

        # Query should be fast even with many records
        results = storage.query(agent_name="TestAgent", limit=10)
        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
