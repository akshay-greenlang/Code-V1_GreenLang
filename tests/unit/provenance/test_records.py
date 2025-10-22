"""
Comprehensive tests for GreenLang Provenance Records Module.

Tests cover:
- ProvenanceRecord dataclass creation and validation
- ProvenanceContext for runtime tracking
- JSON serialization/deserialization
- Data lineage tracking
- Audit trail functionality for regulatory compliance
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

from greenlang.provenance.records import (
    ProvenanceRecord,
    ProvenanceContext
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_environment():
    """Sample environment data."""
    return {
        "timestamp": "2024-01-01T12:00:00Z",
        "python": {
            "version": "3.9.0",
            "version_info": {"major": 3, "minor": 9, "micro": 0}
        },
        "system": {
            "os": "Linux",
            "release": "5.10.0",
            "machine": "x86_64"
        }
    }


@pytest.fixture
def sample_dependencies():
    """Sample dependencies data."""
    return {
        "pandas": "2.0.3",
        "pydantic": "2.1.1",
        "pytest": "7.4.0"
    }


@pytest.fixture
def sample_configuration():
    """Sample configuration data."""
    return {
        "pipeline": "cbam_reporting",
        "version": "1.0.0",
        "settings": {
            "validate_inputs": True,
            "generate_reports": True
        }
    }


@pytest.fixture
def sample_provenance_record(sample_environment, sample_dependencies, sample_configuration):
    """Create a sample provenance record."""
    return ProvenanceRecord(
        record_id="test-record-001",
        generated_at="2024-01-01T12:00:00Z",
        environment=sample_environment,
        dependencies=sample_dependencies,
        configuration=sample_configuration,
        agent_execution=[
            {
                "agent_name": "ValidatorAgent",
                "duration_seconds": 1.5,
                "input_records": 100,
                "output_records": 98
            }
        ],
        data_lineage=[
            {
                "step": 0,
                "stage": "input",
                "description": "Loaded 100 records from CSV"
            }
        ],
        validation_results={
            "total_records": 100,
            "valid_records": 98,
            "invalid_records": 2
        }
    )


@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


# ============================================================================
# PROVENANCE RECORD TESTS
# ============================================================================

class TestProvenanceRecord:
    """Test suite for ProvenanceRecord dataclass."""

    def test_provenance_record_creation(self, sample_environment, sample_dependencies, sample_configuration):
        """Test creating a ProvenanceRecord."""
        record = ProvenanceRecord(
            record_id="test-001",
            generated_at="2024-01-01T12:00:00Z",
            environment=sample_environment,
            dependencies=sample_dependencies,
            configuration=sample_configuration
        )

        assert record.record_id == "test-001"
        assert record.generated_at == "2024-01-01T12:00:00Z"
        assert record.environment == sample_environment
        assert record.dependencies == sample_dependencies
        assert record.configuration == sample_configuration

    def test_provenance_record_defaults(self, sample_environment, sample_dependencies, sample_configuration):
        """Test default values in ProvenanceRecord."""
        record = ProvenanceRecord(
            record_id="test-001",
            generated_at="2024-01-01T12:00:00Z",
            environment=sample_environment,
            dependencies=sample_dependencies,
            configuration=sample_configuration
        )

        # Check defaults
        assert record.agent_execution == []
        assert record.data_lineage == []
        assert record.validation_results == {}
        assert record.input_file_hash is None
        assert record.metadata == {}

    def test_provenance_record_with_optional_fields(self, sample_environment, sample_dependencies, sample_configuration):
        """Test ProvenanceRecord with all optional fields."""
        record = ProvenanceRecord(
            record_id="test-001",
            generated_at="2024-01-01T12:00:00Z",
            environment=sample_environment,
            dependencies=sample_dependencies,
            configuration=sample_configuration,
            agent_execution=[{"agent": "test"}],
            data_lineage=[{"step": 0}],
            validation_results={"valid": True},
            input_file_hash={"hash": "abc123"},
            metadata={"author": "test"}
        )

        assert len(record.agent_execution) == 1
        assert len(record.data_lineage) == 1
        assert record.validation_results["valid"] is True
        assert record.input_file_hash["hash"] == "abc123"
        assert record.metadata["author"] == "test"

    def test_provenance_record_to_dict(self, sample_provenance_record):
        """Test converting ProvenanceRecord to dictionary."""
        record_dict = sample_provenance_record.to_dict()

        assert isinstance(record_dict, dict)
        assert record_dict["record_id"] == "test-record-001"
        assert "environment" in record_dict
        assert "dependencies" in record_dict
        assert "configuration" in record_dict
        assert "agent_execution" in record_dict
        assert "data_lineage" in record_dict

    def test_provenance_record_to_json(self, sample_provenance_record):
        """Test converting ProvenanceRecord to JSON."""
        json_str = sample_provenance_record.to_json()

        assert isinstance(json_str, str)
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["record_id"] == "test-record-001"

    def test_provenance_record_to_json_indent(self, sample_provenance_record):
        """Test JSON conversion with custom indent."""
        json_str = sample_provenance_record.to_json(indent=4)

        assert isinstance(json_str, str)
        # Should have indentation
        assert "\n" in json_str
        assert "    " in json_str  # 4 spaces

    def test_provenance_record_from_dict(self, sample_environment, sample_dependencies, sample_configuration):
        """Test creating ProvenanceRecord from dictionary."""
        data = {
            "record_id": "test-002",
            "generated_at": "2024-01-01T12:00:00Z",
            "environment": sample_environment,
            "dependencies": sample_dependencies,
            "configuration": sample_configuration
        }

        record = ProvenanceRecord.from_dict(data)

        assert record.record_id == "test-002"
        assert record.environment == sample_environment

    def test_provenance_record_from_json(self, sample_provenance_record):
        """Test creating ProvenanceRecord from JSON string."""
        json_str = sample_provenance_record.to_json()
        reconstructed = ProvenanceRecord.from_json(json_str)

        assert reconstructed.record_id == sample_provenance_record.record_id
        assert reconstructed.generated_at == sample_provenance_record.generated_at
        assert reconstructed.environment == sample_provenance_record.environment

    def test_provenance_record_save_load(self, sample_provenance_record, temp_json_file):
        """Test saving and loading ProvenanceRecord."""
        # Save
        sample_provenance_record.save(temp_json_file)

        # Verify file exists
        assert Path(temp_json_file).exists()

        # Load
        loaded = ProvenanceRecord.load(temp_json_file)

        # Verify
        assert loaded.record_id == sample_provenance_record.record_id
        assert loaded.environment == sample_provenance_record.environment
        assert loaded.dependencies == sample_provenance_record.dependencies

    def test_provenance_record_roundtrip(self, sample_provenance_record):
        """Test complete serialization/deserialization roundtrip."""
        # To JSON and back
        json_str = sample_provenance_record.to_json()
        reconstructed = ProvenanceRecord.from_json(json_str)

        # Should be equivalent
        assert reconstructed.record_id == sample_provenance_record.record_id
        assert reconstructed.agent_execution == sample_provenance_record.agent_execution
        assert reconstructed.data_lineage == sample_provenance_record.data_lineage


# ============================================================================
# PROVENANCE CONTEXT TESTS
# ============================================================================

class TestProvenanceContext:
    """Test suite for ProvenanceContext class."""

    def test_provenance_context_initialization(self):
        """Test initializing ProvenanceContext."""
        ctx = ProvenanceContext("test_pipeline")

        assert ctx.name == "test_pipeline"
        assert ctx.record_id is not None
        assert ctx.started_at is not None
        assert isinstance(ctx.inputs, list)
        assert isinstance(ctx.outputs, dict)
        assert isinstance(ctx.environment, dict)
        assert isinstance(ctx.dependencies, dict)

    def test_provenance_context_custom_record_id(self):
        """Test ProvenanceContext with custom record ID."""
        ctx = ProvenanceContext("test", record_id="custom-id-001")

        assert ctx.record_id == "custom-id-001"

    def test_provenance_context_record_id_generation(self):
        """Test automatic record ID generation."""
        ctx1 = ProvenanceContext("test")
        ctx2 = ProvenanceContext("test")

        # Should have different IDs
        assert ctx1.record_id != ctx2.record_id
        # Should contain name
        assert "test" in ctx1.record_id

    def test_provenance_context_record_input(self):
        """Test recording input sources."""
        ctx = ProvenanceContext("test")
        ctx.record_input("input.csv", {"rows": 1000})

        assert len(ctx.inputs) == 1
        assert ctx.inputs[0]["source"] == "input.csv"
        assert ctx.inputs[0]["metadata"]["rows"] == 1000

    def test_provenance_context_record_multiple_inputs(self):
        """Test recording multiple inputs."""
        ctx = ProvenanceContext("test")
        ctx.record_input("input1.csv")
        ctx.record_input("input2.csv")
        ctx.record_input("input3.csv")

        assert len(ctx.inputs) == 3

    def test_provenance_context_record_output(self):
        """Test recording outputs."""
        ctx = ProvenanceContext("test")
        ctx.record_output("output.csv", {"rows": 950})

        assert "output.csv" in ctx.outputs
        assert ctx.outputs["output.csv"]["metadata"]["rows"] == 950

    def test_provenance_context_record_agent_execution(self):
        """Test recording agent execution."""
        ctx = ProvenanceContext("test")
        ctx.record_agent_execution(
            agent_name="ValidatorAgent",
            start_time="2024-01-01T12:00:00Z",
            end_time="2024-01-01T12:00:05Z",
            duration_seconds=5.0,
            input_records=100,
            output_records=98
        )

        assert len(ctx.agent_executions) == 1
        execution = ctx.agent_executions[0]
        assert execution["agent_name"] == "ValidatorAgent"
        assert execution["duration_seconds"] == 5.0
        assert execution["input_records"] == 100
        assert execution["output_records"] == 98

    def test_provenance_context_agent_execution_creates_lineage(self):
        """Test that agent execution creates data lineage entry."""
        ctx = ProvenanceContext("test")
        ctx.record_agent_execution(
            agent_name="TransformAgent",
            start_time="2024-01-01T12:00:00Z",
            end_time="2024-01-01T12:00:05Z",
            duration_seconds=5.0,
            input_records=100,
            output_records=100
        )

        # Should create lineage entry
        assert len(ctx.data_lineage) == 1
        lineage = ctx.data_lineage[0]
        assert lineage["stage"] == "TransformAgent"
        assert lineage["input_records"] == 100
        assert lineage["output_records"] == 100

    def test_provenance_context_record_validation(self):
        """Test recording validation results."""
        ctx = ProvenanceContext("test")
        validation = {
            "total": 100,
            "valid": 95,
            "invalid": 5,
            "errors": ["Missing field: carbon_emissions"]
        }
        ctx.record_validation(validation)

        assert ctx.validation_results == validation

    def test_provenance_context_set_configuration(self):
        """Test setting configuration."""
        ctx = ProvenanceContext("test")
        config = {
            "pipeline": "cbam",
            "version": "1.0.0"
        }
        ctx.set_configuration(config)

        assert ctx.configuration == config

    def test_provenance_context_add_metadata(self):
        """Test adding metadata."""
        ctx = ProvenanceContext("test")
        ctx.add_metadata("author", "test_user")
        ctx.add_metadata("purpose", "testing")

        assert ctx.metadata["author"] == "test_user"
        assert ctx.metadata["purpose"] == "testing"

    def test_provenance_context_to_record(self):
        """Test converting context to ProvenanceRecord."""
        ctx = ProvenanceContext("test")
        ctx.record_input("input.csv")
        ctx.record_agent_execution(
            agent_name="TestAgent",
            start_time="2024-01-01T12:00:00Z",
            end_time="2024-01-01T12:00:01Z",
            duration_seconds=1.0,
            input_records=100,
            output_records=100
        )

        record = ctx.to_record()

        assert isinstance(record, ProvenanceRecord)
        assert record.record_id == ctx.record_id
        assert record.environment == ctx.environment
        assert record.dependencies == ctx.dependencies
        assert len(record.agent_execution) == 1

    def test_provenance_context_finalize(self, temp_json_file):
        """Test finalizing context."""
        ctx = ProvenanceContext("test")
        ctx.add_metadata("test", "value")

        record = ctx.finalize(output_path=temp_json_file)

        # Should return record
        assert isinstance(record, ProvenanceRecord)

        # Should save to file
        assert Path(temp_json_file).exists()

    def test_provenance_context_finalize_without_path(self):
        """Test finalizing without saving."""
        ctx = ProvenanceContext("test")
        record = ctx.finalize()

        assert isinstance(record, ProvenanceRecord)


# ============================================================================
# DATA LINEAGE TESTS
# ============================================================================

class TestDataLineage:
    """Test suite for data lineage tracking."""

    def test_lineage_basic_tracking(self):
        """Test basic lineage tracking."""
        ctx = ProvenanceContext("lineage_test")

        # Stage 1: Load
        ctx.record_agent_execution(
            agent_name="LoadAgent",
            start_time="2024-01-01T12:00:00Z",
            end_time="2024-01-01T12:00:01Z",
            duration_seconds=1.0,
            input_records=0,
            output_records=1000
        )

        # Stage 2: Validate
        ctx.record_agent_execution(
            agent_name="ValidateAgent",
            start_time="2024-01-01T12:00:01Z",
            end_time="2024-01-01T12:00:02Z",
            duration_seconds=1.0,
            input_records=1000,
            output_records=980
        )

        # Stage 3: Transform
        ctx.record_agent_execution(
            agent_name="TransformAgent",
            start_time="2024-01-01T12:00:02Z",
            end_time="2024-01-01T12:00:03Z",
            duration_seconds=1.0,
            input_records=980,
            output_records=980
        )

        # Should have 3 lineage entries
        assert len(ctx.data_lineage) == 3
        assert ctx.data_lineage[0]["stage"] == "LoadAgent"
        assert ctx.data_lineage[1]["stage"] == "ValidateAgent"
        assert ctx.data_lineage[2]["stage"] == "TransformAgent"

    def test_lineage_step_numbering(self):
        """Test that lineage steps are numbered correctly."""
        ctx = ProvenanceContext("lineage_test")

        for i in range(5):
            ctx.record_agent_execution(
                agent_name=f"Agent{i}",
                start_time="2024-01-01T12:00:00Z",
                end_time="2024-01-01T12:00:01Z",
                duration_seconds=1.0
            )

        # Steps should be 0, 1, 2, 3, 4
        for i in range(5):
            assert ctx.data_lineage[i]["step"] == i

    def test_lineage_record_counts(self):
        """Test tracking record counts through pipeline."""
        ctx = ProvenanceContext("lineage_test")

        stages = [
            ("Load", 0, 1000),
            ("Validate", 1000, 950),
            ("Transform", 950, 950),
            ("Aggregate", 950, 100)
        ]

        for stage_name, input_count, output_count in stages:
            ctx.record_agent_execution(
                agent_name=stage_name,
                start_time="2024-01-01T12:00:00Z",
                end_time="2024-01-01T12:00:01Z",
                duration_seconds=1.0,
                input_records=input_count,
                output_records=output_count
            )

        # Verify lineage
        assert ctx.data_lineage[0]["output_records"] == 1000
        assert ctx.data_lineage[1]["input_records"] == 1000
        assert ctx.data_lineage[1]["output_records"] == 950
        assert ctx.data_lineage[3]["output_records"] == 100


# ============================================================================
# INTEGRATION TESTS FOR AUDIT TRAIL
# ============================================================================

class TestAuditTrailIntegration:
    """Integration tests for audit trail functionality."""

    def test_complete_audit_trail_workflow(self, temp_json_file):
        """Test complete audit trail workflow."""
        # Initialize context
        ctx = ProvenanceContext("cbam_reporting")

        # Record configuration
        ctx.set_configuration({
            "pipeline": "cbam_reporting",
            "version": "1.0.0",
            "validate_inputs": True
        })

        # Record input
        ctx.record_input("shipments.csv", {
            "rows": 1000,
            "format": "csv"
        })

        # Simulate pipeline execution
        ctx.record_agent_execution(
            agent_name="ValidatorAgent",
            start_time="2024-01-01T12:00:00Z",
            end_time="2024-01-01T12:00:05Z",
            duration_seconds=5.0,
            input_records=1000,
            output_records=980
        )

        ctx.record_agent_execution(
            agent_name="CarbonCalculatorAgent",
            start_time="2024-01-01T12:00:05Z",
            end_time="2024-01-01T12:00:10Z",
            duration_seconds=5.0,
            input_records=980,
            output_records=980
        )

        # Record validation
        ctx.record_validation({
            "total": 1000,
            "valid": 980,
            "invalid": 20
        })

        # Record output
        ctx.record_output("cbam_report.xlsx", {
            "records": 980,
            "format": "xlsx"
        })

        # Add metadata
        ctx.add_metadata("author", "test_user")
        ctx.add_metadata("purpose", "CBAM_compliance")

        # Finalize and save
        record = ctx.finalize(output_path=temp_json_file)

        # Verify record
        assert record.record_id is not None
        assert len(record.agent_execution) == 2
        assert len(record.data_lineage) == 2
        assert record.validation_results["valid"] == 980

        # Load and verify
        loaded = ProvenanceRecord.load(temp_json_file)
        assert loaded.record_id == record.record_id

    def test_audit_trail_serialization(self):
        """Test audit trail serialization for long-term storage."""
        ctx = ProvenanceContext("audit_test")
        ctx.record_agent_execution(
            agent_name="TestAgent",
            start_time="2024-01-01T12:00:00Z",
            end_time="2024-01-01T12:00:01Z",
            duration_seconds=1.0,
            input_records=100,
            output_records=100
        )

        record = ctx.to_record()

        # Convert to JSON
        json_str = record.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "record_id" in parsed
        assert "environment" in parsed
        assert "dependencies" in parsed

        # Should be restorable
        restored = ProvenanceRecord.from_json(json_str)
        assert restored.record_id == record.record_id

    def test_regulatory_compliance_record(self):
        """Test creating record for regulatory compliance."""
        ctx = ProvenanceContext("cbam_compliance")

        # Required for CBAM compliance
        ctx.set_configuration({
            "regulation": "EU CBAM",
            "reporting_period": "2024-Q1",
            "facility_id": "FAC-001"
        })

        ctx.record_input("emissions_data.csv")

        ctx.add_metadata("compliance_officer", "Jane Doe")
        ctx.add_metadata("verification_date", "2024-01-15")
        ctx.add_metadata("approved", True)

        record = ctx.to_record()

        # Verify compliance fields
        assert "regulation" in record.configuration
        assert "compliance_officer" in record.metadata
        assert "verification_date" in record.metadata
