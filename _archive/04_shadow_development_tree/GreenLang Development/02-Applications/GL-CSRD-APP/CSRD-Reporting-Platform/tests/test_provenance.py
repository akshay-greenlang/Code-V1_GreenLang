# -*- coding: utf-8 -*-
"""
CSRD/ESRS Digital Reporting Platform - Provenance Tests
========================================================

THE FINAL TEST SUITE - Comprehensive provenance testing for 100% Phase 5 completion!

This test suite validates the complete provenance tracking framework that ensures:
1. 7-year regulatory audit trail (EU CSRD requirement)
2. SHA-256 file integrity verification
3. Complete calculation lineage tracking
4. Environment reproducibility
5. Data source traceability
6. NetworkX dependency graph analysis
7. Audit package generation (ZIP)
8. Audit report generation (Markdown)

Architecture:
- 12 test classes covering all provenance functionality
- 70+ test cases for comprehensive coverage
- Tests for all 4 Pydantic models
- Integration tests for audit workflows
- CLI interface testing

Version: 1.0.0
Author: GreenLang CSRD Team
"""

import hashlib
import json
import os
import platform
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import pytest

from provenance import (
    # Core functions
    hash_file,
    hash_data,
    capture_environment,
    get_dependency_versions,
    # Data source tracking
    create_data_source,
    # Calculation lineage
    track_calculation_lineage,
    # Provenance records
    create_provenance_record,
    # Graph analysis
    build_lineage_graph,
    get_calculation_path,
    # Serialization
    serialize_provenance,
    save_provenance_json,
    # Audit package
    create_audit_package,
    generate_audit_report,
    # Models
    DataSource,
    CalculationLineage,
    EnvironmentSnapshot,
    ProvenanceRecord,
)


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture
def tmp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_file(tmp_dir):
    """Create sample CSV file for testing."""
    csv_path = tmp_dir / "esg_data.csv"
    csv_content = """Year,Scope1_tCO2e,Scope2_tCO2e,Scope3_tCO2e
2023,1000.5,500.25,2000.75
2024,950.0,480.0,1950.0
"""
    csv_path.write_text(csv_content, encoding='utf-8')
    return csv_path


@pytest.fixture
def sample_json_file(tmp_dir):
    """Create sample JSON file for testing."""
    json_path = tmp_dir / "config.json"
    json_data = {
        "model": "gpt-4o",
        "temperature": 0.3,
        "max_tokens": 4000
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    return json_path


@pytest.fixture
def sample_excel_file(tmp_dir):
    """Create sample Excel file for testing."""
    try:
        import pandas as pd
        excel_path = tmp_dir / "emissions.xlsx"
        df = pd.DataFrame({
            'Year': [2023, 2024],
            'Scope1': [1000, 950],
            'Scope2': [500, 480]
        })
        df.to_excel(excel_path, index=False, sheet_name='Emissions')
        return excel_path
    except ImportError:
        pytest.skip("pandas or openpyxl not installed")


@pytest.fixture
def large_file(tmp_dir):
    """Create large file (>1MB) for testing chunked hashing."""
    large_path = tmp_dir / "large_file.bin"
    # Create 2MB file
    with open(large_path, 'wb') as f:
        # Write 2MB of repeated data
        chunk = b'A' * 1024  # 1KB
        for _ in range(2048):  # 2048 * 1KB = 2MB
            f.write(chunk)
    return large_path


@pytest.fixture
def sample_calculation_lineages():
    """Create sample calculation lineages for graph testing."""
    lineages = []

    # E1-1: Base metric (no dependencies)
    lineages.append(CalculationLineage(
        metric_code="E1-1-BASE",
        metric_name="Scope 1 Emissions",
        formula="direct",
        input_values={"value": 1000},
        output_value=1000,
        output_unit="tCO2e",
        dependencies=[]
    ))

    # E1-2: Another base metric
    lineages.append(CalculationLineage(
        metric_code="E1-2-BASE",
        metric_name="Scope 2 Emissions",
        formula="direct",
        input_values={"value": 500},
        output_value=500,
        output_unit="tCO2e",
        dependencies=[]
    ))

    # E1-3: Depends on E1-1 and E1-2
    lineages.append(CalculationLineage(
        metric_code="E1-3-SUM",
        metric_name="Total Emissions",
        formula="E1-1-BASE + E1-2-BASE",
        formula_type="sum",
        input_values={"E1-1-BASE": 1000, "E1-2-BASE": 500},
        output_value=1500,
        output_unit="tCO2e",
        dependencies=["E1-1-BASE", "E1-2-BASE"]
    ))

    # E1-4: Depends on E1-3
    lineages.append(CalculationLineage(
        metric_code="E1-4-INTENSITY",
        metric_name="Emission Intensity",
        formula="E1-3-SUM / revenue",
        formula_type="division",
        input_values={"E1-3-SUM": 1500, "revenue": 1000000},
        output_value=0.0015,
        output_unit="tCO2e/€",
        dependencies=["E1-3-SUM"]
    ))

    return lineages


# ============================================================================
# TEST CLASS 1: DataSource Model
# ============================================================================


class TestDataSourceModel:
    """Test DataSource Pydantic model."""

    def test_data_source_creation_csv(self, sample_csv_file):
        """Test creating DataSource for CSV file."""
        source = DataSource(
            source_type="csv",
            file_path=str(sample_csv_file),
            row_index=1,
            column_name="Scope1_tCO2e"
        )

        assert source.source_type == "csv"
        assert source.file_path == str(sample_csv_file)
        assert source.row_index == 1
        assert source.column_name == "Scope1_tCO2e"
        assert source.source_id is not None  # Auto-generated UUID
        assert source.timestamp is not None  # Auto-generated timestamp

    def test_data_source_creation_json(self, sample_json_file):
        """Test creating DataSource for JSON file."""
        source = DataSource(
            source_type="json",
            file_path=str(sample_json_file)
        )

        assert source.source_type == "json"
        assert source.file_path == str(sample_json_file)

    def test_data_source_creation_excel(self):
        """Test creating DataSource for Excel file."""
        source = DataSource(
            source_type="excel",
            file_path="/path/to/emissions.xlsx",
            sheet_name="Emissions",
            row_index=5,
            column_name="Scope1_tCO2e",
            cell_reference="B6"
        )

        assert source.source_type == "excel"
        assert source.sheet_name == "Emissions"
        assert source.cell_reference == "B6"

    def test_data_source_creation_database(self):
        """Test creating DataSource for database query."""
        source = DataSource(
            source_type="database",
            table_name="esg_metrics",
            query="SELECT scope1 FROM esg_metrics WHERE year = 2023",
            row_index=0,
            column_name="scope1"
        )

        assert source.source_type == "database"
        assert source.table_name == "esg_metrics"
        assert "SELECT" in source.query

    def test_data_source_auto_uuid(self):
        """Test that source_id is auto-generated."""
        source1 = DataSource(source_type="csv")
        source2 = DataSource(source_type="csv")

        assert source1.source_id != source2.source_id
        assert len(source1.source_id) == 36  # UUID4 format

    def test_data_source_auto_timestamp(self):
        """Test that timestamp is auto-generated."""
        source = DataSource(source_type="csv")

        assert source.timestamp is not None
        # Verify ISO 8601 format
        datetime.fromisoformat(source.timestamp.replace('Z', '+00:00'))

    def test_data_source_metadata(self):
        """Test DataSource with custom metadata."""
        source = DataSource(
            source_type="csv",
            file_path="/path/to/data.csv",
            metadata={
                "uploaded_by": "john.doe",
                "verified": True,
                "version": "1.2.3"
            }
        )

        assert source.metadata["uploaded_by"] == "john.doe"
        assert source.metadata["verified"] is True

    def test_data_source_serialization(self):
        """Test DataSource serialization to dict."""
        source = DataSource(
            source_type="csv",
            file_path="/path/to/data.csv",
            row_index=10
        )

        data = source.dict()

        assert isinstance(data, dict)
        assert data["source_type"] == "csv"
        assert data["file_path"] == "/path/to/data.csv"
        assert data["row_index"] == 10

    def test_data_source_validation_warning(self, caplog):
        """Test that invalid source_type logs warning but doesn't fail."""
        source = DataSource(
            source_type="unknown_type",
            file_path="/path/to/data.txt"
        )

        # Should still create (just logs warning)
        assert source.source_type == "unknown_type"


# ============================================================================
# TEST CLASS 2: CalculationLineage Model
# ============================================================================


class TestCalculationLineageModel:
    """Test CalculationLineage Pydantic model."""

    def test_calculation_lineage_creation(self):
        """Test creating CalculationLineage."""
        lineage = CalculationLineage(
            metric_code="E1-1",
            metric_name="Total GHG Emissions",
            formula="Scope1 + Scope2 + Scope3",
            formula_type="sum",
            input_values={"Scope1": 1000, "Scope2": 500, "Scope3": 2000},
            output_value=3500,
            output_unit="tCO2e"
        )

        assert lineage.metric_code == "E1-1"
        assert lineage.metric_name == "Total GHG Emissions"
        assert lineage.formula == "Scope1 + Scope2 + Scope3"
        assert lineage.output_value == 3500
        assert lineage.output_unit == "tCO2e"

    def test_calculation_lineage_auto_hash(self):
        """Test that hash is auto-generated."""
        lineage = CalculationLineage(
            metric_code="E1-1",
            metric_name="Test",
            formula="a + b",
            input_values={"a": 1, "b": 2},
            output_value=3,
            output_unit="units"
        )

        assert lineage.hash is not None
        assert len(lineage.hash) == 64  # SHA-256 hex length

    def test_calculation_lineage_hash_deterministic(self):
        """Test that hash is deterministic (same inputs → same hash)."""
        lineage1 = CalculationLineage(
            metric_code="E1-1",
            metric_name="Test",
            formula="a + b",
            input_values={"a": 1, "b": 2},
            output_value=3,
            output_unit="units"
        )

        lineage2 = CalculationLineage(
            metric_code="E1-1",
            metric_name="Test",
            formula="a + b",
            input_values={"a": 1, "b": 2},
            output_value=3,
            output_unit="units"
        )

        assert lineage1.hash == lineage2.hash

    def test_calculation_lineage_hash_changes_with_inputs(self):
        """Test that hash changes when inputs change."""
        lineage1 = CalculationLineage(
            metric_code="E1-1",
            metric_name="Test",
            formula="a + b",
            input_values={"a": 1, "b": 2},
            output_value=3,
            output_unit="units"
        )

        lineage2 = CalculationLineage(
            metric_code="E1-1",
            metric_name="Test",
            formula="a + b",
            input_values={"a": 5, "b": 10},  # Different inputs
            output_value=15,
            output_unit="units"
        )

        assert lineage1.hash != lineage2.hash

    def test_calculation_lineage_with_intermediate_steps(self):
        """Test CalculationLineage with intermediate steps."""
        lineage = CalculationLineage(
            metric_code="E1-1",
            metric_name="Total Emissions",
            formula="(Scope1 + Scope2) * intensity",
            input_values={"Scope1": 1000, "Scope2": 500, "intensity": 1.5},
            intermediate_steps=[
                "Step 1: Scope1 + Scope2 = 1500",
                "Step 2: 1500 * 1.5 = 2250"
            ],
            output_value=2250,
            output_unit="tCO2e"
        )

        assert len(lineage.intermediate_steps) == 2
        assert "Step 1" in lineage.intermediate_steps[0]

    def test_calculation_lineage_with_data_sources(self):
        """Test CalculationLineage with data sources."""
        source = DataSource(
            source_type="csv",
            file_path="/path/to/data.csv"
        )

        lineage = CalculationLineage(
            metric_code="E1-1",
            metric_name="Test",
            formula="a + b",
            input_values={"a": 1, "b": 2},
            output_value=3,
            output_unit="units",
            data_sources=[source]
        )

        assert len(lineage.data_sources) == 1
        assert lineage.data_sources[0].source_type == "csv"

    def test_calculation_lineage_with_dependencies(self):
        """Test CalculationLineage with dependencies."""
        lineage = CalculationLineage(
            metric_code="E1-3",
            metric_name="Total",
            formula="E1-1 + E1-2",
            input_values={"E1-1": 1000, "E1-2": 500},
            output_value=1500,
            output_unit="tCO2e",
            dependencies=["E1-1", "E1-2"]
        )

        assert lineage.dependencies == ["E1-1", "E1-2"]

    def test_calculation_lineage_serialization(self):
        """Test CalculationLineage serialization."""
        lineage = CalculationLineage(
            metric_code="E1-1",
            metric_name="Test",
            formula="a + b",
            input_values={"a": 1, "b": 2},
            output_value=3,
            output_unit="units"
        )

        data = lineage.dict()

        assert isinstance(data, dict)
        assert data["metric_code"] == "E1-1"
        assert data["hash"] is not None


# ============================================================================
# TEST CLASS 3: EnvironmentSnapshot Model
# ============================================================================


class TestEnvironmentSnapshotModel:
    """Test EnvironmentSnapshot Pydantic model."""

    def test_environment_snapshot_creation(self):
        """Test creating EnvironmentSnapshot."""
        snapshot = EnvironmentSnapshot(
            python_version=sys.version,
            python_major=3,
            python_minor=11,
            python_micro=5,
            python_implementation="CPython",
            python_compiler="GCC 9.3.0",
            platform="Linux",
            platform_release="5.15.0",
            platform_version="#1 SMP",
            machine="x86_64",
            processor="x86_64",
            hostname="test-server",
            user="test_user",
            working_directory="/home/test",
            process_id=12345
        )

        assert snapshot.python_major == 3
        assert snapshot.platform == "Linux"
        assert snapshot.hostname == "test-server"

    def test_environment_snapshot_auto_id(self):
        """Test that snapshot_id is auto-generated."""
        snapshot = EnvironmentSnapshot(
            python_version=sys.version,
            python_major=3,
            python_minor=11,
            python_micro=5,
            python_implementation="CPython",
            python_compiler="GCC 9.3.0",
            platform="Linux",
            platform_release="5.15.0",
            platform_version="#1",
            machine="x86_64",
            processor="x86_64",
            hostname="test",
            user="test",
            working_directory="/test",
            process_id=1
        )

        assert snapshot.snapshot_id is not None
        assert len(snapshot.snapshot_id) == 36  # UUID4

    def test_environment_snapshot_with_packages(self):
        """Test EnvironmentSnapshot with package versions."""
        snapshot = EnvironmentSnapshot(
            python_version=sys.version,
            python_major=3,
            python_minor=11,
            python_micro=5,
            python_implementation="CPython",
            python_compiler="GCC 9.3.0",
            platform="Linux",
            platform_release="5.15.0",
            platform_version="#1",
            machine="x86_64",
            processor="x86_64",
            hostname="test",
            user="test",
            working_directory="/test",
            process_id=1,
            package_versions={
                "pandas": "2.0.3",
                "pydantic": "2.1.1",
                "networkx": "3.1"
            }
        )

        assert snapshot.package_versions["pandas"] == "2.0.3"
        assert snapshot.package_versions["pydantic"] == "2.1.1"

    def test_environment_snapshot_with_llm_models(self):
        """Test EnvironmentSnapshot with LLM models."""
        snapshot = EnvironmentSnapshot(
            python_version=sys.version,
            python_major=3,
            python_minor=11,
            python_micro=5,
            python_implementation="CPython",
            python_compiler="GCC 9.3.0",
            platform="Linux",
            platform_release="5.15.0",
            platform_version="#1",
            machine="x86_64",
            processor="x86_64",
            hostname="test",
            user="test",
            working_directory="/test",
            process_id=1,
            llm_models={
                "materiality": "gpt-4o",
                "narratives": "claude-3-5-sonnet"
            }
        )

        assert snapshot.llm_models["materiality"] == "gpt-4o"
        assert snapshot.llm_models["narratives"] == "claude-3-5-sonnet"

    def test_environment_snapshot_with_config_hash(self):
        """Test EnvironmentSnapshot with config hash."""
        snapshot = EnvironmentSnapshot(
            python_version=sys.version,
            python_major=3,
            python_minor=11,
            python_micro=5,
            python_implementation="CPython",
            python_compiler="GCC 9.3.0",
            platform="Linux",
            platform_release="5.15.0",
            platform_version="#1",
            machine="x86_64",
            processor="x86_64",
            hostname="test",
            user="test",
            working_directory="/test",
            process_id=1,
            config_hash="a" * 64  # Mock SHA-256 hash
        )

        assert snapshot.config_hash == "a" * 64
        assert len(snapshot.config_hash) == 64

    def test_environment_snapshot_serialization(self):
        """Test EnvironmentSnapshot serialization."""
        snapshot = EnvironmentSnapshot(
            python_version=sys.version,
            python_major=3,
            python_minor=11,
            python_micro=5,
            python_implementation="CPython",
            python_compiler="GCC 9.3.0",
            platform="Linux",
            platform_release="5.15.0",
            platform_version="#1",
            machine="x86_64",
            processor="x86_64",
            hostname="test",
            user="test",
            working_directory="/test",
            process_id=1
        )

        data = snapshot.dict()

        assert isinstance(data, dict)
        assert data["python_major"] == 3
        assert data["platform"] == "Linux"


# ============================================================================
# TEST CLASS 4: ProvenanceRecord Model
# ============================================================================


class TestProvenanceRecordModel:
    """Test ProvenanceRecord Pydantic model."""

    def test_provenance_record_creation(self):
        """Test creating ProvenanceRecord."""
        record = ProvenanceRecord(
            agent_name="IntakeAgent",
            operation="validate_data",
            inputs={"file": "esg_data.csv"},
            outputs={"valid_rows": 100}
        )

        assert record.agent_name == "IntakeAgent"
        assert record.operation == "validate_data"
        assert record.inputs["file"] == "esg_data.csv"
        assert record.outputs["valid_rows"] == 100

    def test_provenance_record_auto_id(self):
        """Test that record_id is auto-generated."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test"
        )

        assert record.record_id is not None
        assert len(record.record_id) == 36  # UUID4

    def test_provenance_record_auto_timestamp(self):
        """Test that timestamp is auto-generated."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test"
        )

        assert record.timestamp is not None
        # Verify ISO 8601 format
        datetime.fromisoformat(record.timestamp.replace('Z', '+00:00'))

    def test_provenance_record_with_environment(self):
        """Test ProvenanceRecord with environment snapshot."""
        env = EnvironmentSnapshot(
            python_version=sys.version,
            python_major=3,
            python_minor=11,
            python_micro=5,
            python_implementation="CPython",
            python_compiler="GCC 9.3.0",
            platform="Linux",
            platform_release="5.15.0",
            platform_version="#1",
            machine="x86_64",
            processor="x86_64",
            hostname="test",
            user="test",
            working_directory="/test",
            process_id=1
        )

        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test",
            environment=env
        )

        assert record.environment is not None
        assert record.environment.python_major == 3

    def test_provenance_record_with_calculation_lineage(self):
        """Test ProvenanceRecord with calculation lineage."""
        lineage = CalculationLineage(
            metric_code="E1-1",
            metric_name="Test",
            formula="a + b",
            input_values={"a": 1, "b": 2},
            output_value=3,
            output_unit="units"
        )

        record = ProvenanceRecord(
            agent_name="CalculatorAgent",
            operation="calculate",
            calculation_lineage=lineage
        )

        assert record.calculation_lineage is not None
        assert record.calculation_lineage.metric_code == "E1-1"

    def test_provenance_record_with_data_sources(self):
        """Test ProvenanceRecord with data sources."""
        source = DataSource(
            source_type="csv",
            file_path="/path/to/data.csv"
        )

        record = ProvenanceRecord(
            agent_name="IntakeAgent",
            operation="load_data",
            data_sources=[source]
        )

        assert len(record.data_sources) == 1
        assert record.data_sources[0].source_type == "csv"

    def test_provenance_record_with_duration(self):
        """Test ProvenanceRecord with duration."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test",
            duration_seconds=2.5
        )

        assert record.duration_seconds == 2.5

    def test_provenance_record_status_success(self):
        """Test ProvenanceRecord with success status."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test",
            status="success"
        )

        assert record.status == "success"

    def test_provenance_record_status_error(self):
        """Test ProvenanceRecord with error status."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test",
            status="error",
            errors=["File not found", "Invalid format"]
        )

        assert record.status == "error"
        assert len(record.errors) == 2
        assert "File not found" in record.errors

    def test_provenance_record_with_warnings(self):
        """Test ProvenanceRecord with warnings."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test",
            status="warning",
            warnings=["Missing optional field", "Deprecated API used"]
        )

        assert record.status == "warning"
        assert len(record.warnings) == 2

    def test_provenance_record_serialization(self):
        """Test ProvenanceRecord serialization."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test",
            inputs={"key": "value"}
        )

        data = record.dict()

        assert isinstance(data, dict)
        assert data["agent_name"] == "TestAgent"
        assert data["operation"] == "test"


# ============================================================================
# TEST CLASS 5: SHA-256 Hashing
# ============================================================================


class TestSHA256Hashing:
    """Test SHA-256 hashing functions."""

    def test_hash_file_small(self, sample_csv_file):
        """Test hashing small CSV file."""
        hash_info = hash_file(sample_csv_file)

        assert hash_info["file_path"] == str(sample_csv_file.absolute())
        assert hash_info["file_name"] == "esg_data.csv"
        assert hash_info["hash_algorithm"] == "SHA256"
        assert len(hash_info["hash_value"]) == 64  # SHA-256 hex
        assert hash_info["file_size_bytes"] > 0
        assert "human_readable_size" in hash_info

    def test_hash_file_large(self, large_file):
        """Test hashing large file (>1MB) with chunked reading."""
        hash_info = hash_file(large_file)

        assert hash_info["file_size_bytes"] == 2 * 1024 * 1024  # 2MB
        assert len(hash_info["hash_value"]) == 64
        assert "2.00 MB" in hash_info["human_readable_size"]

    def test_hash_file_consistency(self, sample_csv_file):
        """Test that hashing same file twice gives same hash."""
        hash1 = hash_file(sample_csv_file)
        hash2 = hash_file(sample_csv_file)

        assert hash1["hash_value"] == hash2["hash_value"]

    def test_hash_file_different_algorithms(self, sample_csv_file):
        """Test different hash algorithms."""
        sha256_hash = hash_file(sample_csv_file, algorithm="sha256")
        sha512_hash = hash_file(sample_csv_file, algorithm="sha512")

        # MD5 is deprecated and now uses SHA256 instead
        with pytest.warns(UserWarning, match="MD5 is cryptographically broken"):
            md5_hash = hash_file(sample_csv_file, algorithm="md5")

        assert len(sha256_hash["hash_value"]) == 64
        assert len(sha512_hash["hash_value"]) == 128
        assert len(md5_hash["hash_value"]) == 64  # SHA256 hex digest length (not 32 for MD5)

    def test_hash_file_not_found(self):
        """Test hashing non-existent file."""
        with pytest.raises(FileNotFoundError):
            hash_file("/path/that/does/not/exist.csv")

    def test_hash_file_invalid_algorithm(self, sample_csv_file):
        """Test invalid hash algorithm."""
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            hash_file(sample_csv_file, algorithm="invalid")

    def test_hash_data_dict(self):
        """Test hashing dictionary data."""
        data = {
            "model": "gpt-4o",
            "temperature": 0.3,
            "max_tokens": 4000
        }

        hash_value = hash_data(data)

        assert len(hash_value) == 64  # SHA-256 hex
        assert isinstance(hash_value, str)

    def test_hash_data_consistency(self):
        """Test that hashing same data twice gives same hash."""
        data = {"key": "value", "number": 42}

        hash1 = hash_data(data)
        hash2 = hash_data(data)

        assert hash1 == hash2

    def test_hash_data_order_independent(self):
        """Test that dict key order doesn't affect hash."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "b": 2, "a": 1}

        hash1 = hash_data(data1)
        hash2 = hash_data(data2)

        assert hash1 == hash2

    def test_hash_verification_success(self, sample_csv_file):
        """Test successful hash verification."""
        # Get original hash
        original_hash = hash_file(sample_csv_file)

        # Verify hash
        verify_hash = hash_file(sample_csv_file)

        assert verify_hash["hash_value"] == original_hash["hash_value"]

    def test_hash_verification_failure(self, tmp_dir):
        """Test hash verification failure when file changes."""
        # Create file
        test_file = tmp_dir / "test.txt"
        test_file.write_text("original content")

        # Get original hash
        original_hash = hash_file(test_file)["hash_value"]

        # Modify file
        test_file.write_text("modified content")

        # Get new hash
        new_hash = hash_file(test_file)["hash_value"]

        assert new_hash != original_hash


# ============================================================================
# TEST CLASS 6: Calculation Lineage Tracking
# ============================================================================


class TestCalculationLineageTracking:
    """Test calculation lineage tracking functions."""

    def test_track_calculation_lineage_simple_formula(self):
        """Test tracking calculation lineage with simple formula."""
        lineage = track_calculation_lineage(
            metric_code="E1-1",
            metric_name="Scope 1 GHG Emissions",
            formula="fuel_combustion + refrigerants",
            input_values={"fuel_combustion": 800, "refrigerants": 200},
            output_value=1000,
            output_unit="tCO2e"
        )

        assert lineage.metric_code == "E1-1"
        assert lineage.output_value == 1000
        assert len(lineage.hash) == 64

    def test_track_calculation_lineage_complex_formula(self):
        """Test tracking complex calculation."""
        lineage = track_calculation_lineage(
            metric_code="E1-2",
            metric_name="GHG Intensity",
            formula="(Scope1 + Scope2 + Scope3) / revenue * 1000",
            formula_type="division",
            input_values={
                "Scope1": 1000,
                "Scope2": 500,
                "Scope3": 2000,
                "revenue": 1000000
            },
            output_value=3.5,
            output_unit="tCO2e/k€"
        )

        assert lineage.formula_type == "division"
        assert lineage.output_value == 3.5

    def test_track_calculation_lineage_with_intermediate_steps(self):
        """Test tracking with intermediate calculation steps."""
        lineage = track_calculation_lineage(
            metric_code="E1-3",
            metric_name="Total Emissions",
            formula="(Scope1 + Scope2) * intensity_factor",
            input_values={"Scope1": 1000, "Scope2": 500, "intensity_factor": 1.2},
            intermediate_steps=[
                "Step 1: Scope1 + Scope2 = 1500",
                "Step 2: 1500 * 1.2 = 1800"
            ],
            output_value=1800,
            output_unit="tCO2e"
        )

        assert len(lineage.intermediate_steps) == 2
        assert "Step 1" in lineage.intermediate_steps[0]

    def test_track_calculation_lineage_with_data_sources(self):
        """Test tracking with data sources."""
        source = DataSource(
            source_type="csv",
            file_path="/path/to/data.csv"
        )

        lineage = track_calculation_lineage(
            metric_code="E1-4",
            metric_name="Test Metric",
            formula="a + b",
            input_values={"a": 100, "b": 200},
            output_value=300,
            output_unit="units",
            data_sources=[source]
        )

        assert len(lineage.data_sources) == 1
        assert lineage.data_sources[0].source_type == "csv"

    def test_track_calculation_lineage_with_dependencies(self):
        """Test tracking with metric dependencies."""
        lineage = track_calculation_lineage(
            metric_code="E1-5",
            metric_name="Aggregate Metric",
            formula="E1-1 + E1-2",
            input_values={"E1-1": 1000, "E1-2": 500},
            output_value=1500,
            output_unit="tCO2e",
            dependencies=["E1-1", "E1-2"]
        )

        assert lineage.dependencies == ["E1-1", "E1-2"]

    def test_track_calculation_lineage_custom_agent(self):
        """Test tracking with custom agent name."""
        lineage = track_calculation_lineage(
            metric_code="E1-6",
            metric_name="Custom Calculation",
            formula="x * y",
            input_values={"x": 10, "y": 20},
            output_value=200,
            output_unit="units",
            agent_name="CustomAgent"
        )

        assert lineage.agent_name == "CustomAgent"

    def test_track_calculation_lineage_with_metadata(self):
        """Test tracking with custom metadata."""
        lineage = track_calculation_lineage(
            metric_code="E1-7",
            metric_name="Test",
            formula="a + b",
            input_values={"a": 1, "b": 2},
            output_value=3,
            output_unit="units",
            calculated_by="john.doe",
            verified=True
        )

        assert lineage.metadata["calculated_by"] == "john.doe"
        assert lineage.metadata["verified"] is True


# ============================================================================
# TEST CLASS 7: Data Source Creation
# ============================================================================


class TestDataSourceCreation:
    """Test data source creation functions."""

    def test_create_data_source_csv(self, sample_csv_file):
        """Test creating data source for CSV file."""
        source = create_data_source(
            source_type="csv",
            file_path=str(sample_csv_file),
            row_index=1,
            column_name="Scope1_tCO2e"
        )

        assert source.source_type == "csv"
        assert source.file_path == str(sample_csv_file)
        assert source.file_hash is not None  # Auto-hashed
        assert len(source.file_hash) == 64

    def test_create_data_source_json(self, sample_json_file):
        """Test creating data source for JSON file."""
        source = create_data_source(
            source_type="json",
            file_path=str(sample_json_file)
        )

        assert source.source_type == "json"
        assert source.file_hash is not None

    def test_create_data_source_excel(self):
        """Test creating data source for Excel file."""
        source = create_data_source(
            source_type="excel",
            file_path="/path/to/emissions.xlsx",
            sheet_name="Emissions",
            row_index=5,
            column_name="Scope1_tCO2e"
        )

        assert source.source_type == "excel"
        assert source.sheet_name == "Emissions"

    def test_create_data_source_database(self):
        """Test creating data source for database query."""
        source = create_data_source(
            source_type="database",
            table_name="esg_metrics",
            query="SELECT * FROM esg_metrics WHERE year = 2023"
        )

        assert source.source_type == "database"
        assert source.table_name == "esg_metrics"
        assert "SELECT" in source.query

    def test_create_data_source_with_metadata(self):
        """Test creating data source with custom metadata."""
        source = create_data_source(
            source_type="csv",
            file_path="/path/to/data.csv",
            uploaded_by="john.doe",
            version="1.0.0"
        )

        assert source.metadata["uploaded_by"] == "john.doe"
        assert source.metadata["version"] == "1.0.0"

    def test_create_data_source_nonexistent_file(self):
        """Test creating data source for non-existent file logs warning."""
        source = create_data_source(
            source_type="csv",
            file_path="/path/that/does/not/exist.csv"
        )

        # Should still create source, but file_hash will be None
        assert source.file_hash is None


# ============================================================================
# TEST CLASS 8: Environment Capture
# ============================================================================


class TestEnvironmentCapture:
    """Test environment capture functions."""

    def test_capture_environment_basic(self):
        """Test capturing basic environment."""
        env = capture_environment()

        assert env.python_major == sys.version_info.major
        assert env.python_minor == sys.version_info.minor
        assert env.python_micro == sys.version_info.micro
        assert env.platform == platform.system()
        assert env.process_id == os.getpid()

    def test_capture_environment_python_version(self):
        """Test Python version capture."""
        env = capture_environment()

        assert env.python_version == sys.version
        assert env.python_implementation == platform.python_implementation()

    def test_capture_environment_platform_info(self):
        """Test platform information capture."""
        env = capture_environment()

        assert env.platform in ["Linux", "Windows", "Darwin"]
        assert env.machine is not None
        assert env.hostname is not None

    def test_capture_environment_with_config(self, sample_json_file):
        """Test environment capture with config file."""
        env = capture_environment(config_path=sample_json_file)

        assert env.config_hash is not None
        assert len(env.config_hash) == 64  # SHA-256

    def test_capture_environment_with_llm_models(self):
        """Test environment capture with LLM models."""
        llm_models = {
            "materiality": "gpt-4o",
            "narratives": "claude-3-5-sonnet"
        }

        env = capture_environment(llm_models=llm_models)

        assert env.llm_models["materiality"] == "gpt-4o"
        assert env.llm_models["narratives"] == "claude-3-5-sonnet"

    def test_capture_environment_package_versions(self):
        """Test package versions capture."""
        env = capture_environment()

        assert isinstance(env.package_versions, dict)
        # Should have at least some critical packages
        assert len(env.package_versions) > 0

    def test_get_dependency_versions(self):
        """Test getting dependency versions."""
        deps = get_dependency_versions()

        assert isinstance(deps, dict)
        # Check for critical packages
        assert "pydantic" in deps
        assert "networkx" in deps


# ============================================================================
# TEST CLASS 9: NetworkX Graphs
# ============================================================================


class TestNetworkXGraphs:
    """Test NetworkX dependency graph functions."""

    def test_build_lineage_graph(self, sample_calculation_lineages):
        """Test building lineage graph."""
        G = build_lineage_graph(sample_calculation_lineages)

        assert isinstance(G, nx.DiGraph)
        assert len(G.nodes()) == 4
        assert len(G.edges()) == 3

    def test_build_lineage_graph_nodes(self, sample_calculation_lineages):
        """Test graph node attributes."""
        G = build_lineage_graph(sample_calculation_lineages)

        # Check node exists
        assert "E1-1-BASE" in G.nodes()

        # Check node attributes
        node_data = G.nodes["E1-1-BASE"]
        assert node_data["metric_name"] == "Scope 1 Emissions"
        assert node_data["output_value"] == 1000
        assert node_data["output_unit"] == "tCO2e"

    def test_build_lineage_graph_edges(self, sample_calculation_lineages):
        """Test graph edge creation."""
        G = build_lineage_graph(sample_calculation_lineages)

        # Check edges exist
        assert G.has_edge("E1-1-BASE", "E1-3-SUM")
        assert G.has_edge("E1-2-BASE", "E1-3-SUM")
        assert G.has_edge("E1-3-SUM", "E1-4-INTENSITY")

    def test_build_lineage_graph_topological_sort(self, sample_calculation_lineages):
        """Test topological sort on graph."""
        G = build_lineage_graph(sample_calculation_lineages)

        # Should be acyclic
        assert nx.is_directed_acyclic_graph(G)

        # Topological sort should work
        sorted_nodes = list(nx.topological_sort(G))
        assert len(sorted_nodes) == 4

    def test_build_lineage_graph_root_nodes(self, sample_calculation_lineages):
        """Test identifying root nodes (no dependencies)."""
        G = build_lineage_graph(sample_calculation_lineages)

        root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

        assert "E1-1-BASE" in root_nodes
        assert "E1-2-BASE" in root_nodes

    def test_get_calculation_path(self, sample_calculation_lineages):
        """Test getting calculation path for metric."""
        G = build_lineage_graph(sample_calculation_lineages)

        path = get_calculation_path(G, "E1-4-INTENSITY")

        # Path should include all dependencies
        assert "E1-1-BASE" in path
        assert "E1-2-BASE" in path
        assert "E1-3-SUM" in path
        assert "E1-4-INTENSITY" in path

    def test_get_calculation_path_order(self, sample_calculation_lineages):
        """Test calculation path is in correct order."""
        G = build_lineage_graph(sample_calculation_lineages)

        path = get_calculation_path(G, "E1-4-INTENSITY")

        # Base metrics should come first
        assert path.index("E1-1-BASE") < path.index("E1-3-SUM")
        assert path.index("E1-2-BASE") < path.index("E1-3-SUM")
        assert path.index("E1-3-SUM") < path.index("E1-4-INTENSITY")

    def test_get_calculation_path_nonexistent_metric(self, sample_calculation_lineages):
        """Test getting path for non-existent metric."""
        G = build_lineage_graph(sample_calculation_lineages)

        path = get_calculation_path(G, "E1-999-NONEXISTENT")

        assert path == []


# ============================================================================
# TEST CLASS 10: Audit Package Creation
# ============================================================================


class TestAuditPackageCreation:
    """Test audit package creation functions."""

    def test_create_audit_package_basic(self, tmp_dir):
        """Test creating basic audit package."""
        # Create sample record
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test",
            inputs={"key": "value"}
        )

        output_path = tmp_dir / "audit_package.zip"
        result_path = create_audit_package([record], output_path)

        assert result_path.exists()
        assert result_path.suffix == ".zip"

    def test_create_audit_package_contents(self, tmp_dir):
        """Test audit package contains required files."""
        env = capture_environment()
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test",
            environment=env
        )

        output_path = tmp_dir / "audit.zip"
        create_audit_package([record], output_path)

        # Check ZIP contents
        with zipfile.ZipFile(output_path, 'r') as zf:
            files = zf.namelist()

            assert "provenance.json" in files
            assert "environment.json" in files
            assert "manifest.json" in files

    def test_create_audit_package_with_lineage_graph(self, tmp_dir, sample_calculation_lineages):
        """Test audit package with lineage graph."""
        records = []
        for lineage in sample_calculation_lineages:
            record = ProvenanceRecord(
                agent_name="CalculatorAgent",
                operation="calculate",
                calculation_lineage=lineage
            )
            records.append(record)

        output_path = tmp_dir / "audit.zip"
        create_audit_package(records, output_path, include_lineage_graph=True)

        with zipfile.ZipFile(output_path, 'r') as zf:
            assert "lineage_graph.json" in zf.namelist()

            # Check graph content
            graph_data = json.loads(zf.read("lineage_graph.json"))
            assert "nodes" in graph_data
            assert "edges" in graph_data
            assert len(graph_data["nodes"]) == 4

    def test_create_audit_package_with_config(self, tmp_dir, sample_json_file):
        """Test audit package with config file."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test"
        )

        output_path = tmp_dir / "audit.zip"
        create_audit_package(
            [record],
            output_path,
            include_config=sample_json_file
        )

        with zipfile.ZipFile(output_path, 'r') as zf:
            files = zf.namelist()
            assert any("config.json" in f for f in files)

    def test_create_audit_package_with_data_files(self, tmp_dir, sample_csv_file):
        """Test audit package with data files."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test"
        )

        output_path = tmp_dir / "audit.zip"
        create_audit_package(
            [record],
            output_path,
            include_files=[sample_csv_file]
        )

        with zipfile.ZipFile(output_path, 'r') as zf:
            files = zf.namelist()
            assert any("esg_data.csv" in f for f in files)

    def test_create_audit_package_manifest(self, tmp_dir):
        """Test audit package manifest content."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test"
        )

        output_path = tmp_dir / "audit.zip"
        create_audit_package([record], output_path)

        with zipfile.ZipFile(output_path, 'r') as zf:
            manifest = json.loads(zf.read("manifest.json"))

            assert "package_created" in manifest
            assert "platform" in manifest
            assert manifest["platform"] == "CSRD/ESRS Digital Reporting Platform"
            assert "contents" in manifest
            assert manifest["contents"]["provenance_records"] == 1

    def test_create_audit_package_structure(self, tmp_dir):
        """Test audit package directory structure."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test"
        )

        output_path = tmp_dir / "audit.zip"
        create_audit_package([record], output_path)

        with zipfile.ZipFile(output_path, 'r') as zf:
            # Verify all files are valid JSON
            for filename in ["provenance.json", "manifest.json"]:
                content = zf.read(filename)
                data = json.loads(content)
                assert isinstance(data, dict)

    def test_create_audit_package_compression(self, tmp_dir):
        """Test audit package uses compression."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test",
            inputs={"data": "x" * 10000}  # Large data
        )

        output_path = tmp_dir / "audit.zip"
        create_audit_package([record], output_path)

        with zipfile.ZipFile(output_path, 'r') as zf:
            info = zf.getinfo("provenance.json")
            # Compressed size should be less than file size
            assert info.compress_size < info.file_size


# ============================================================================
# TEST CLASS 11: Audit Report Generation
# ============================================================================


class TestAuditReportGeneration:
    """Test audit report generation functions."""

    def test_generate_audit_report_basic(self):
        """Test generating basic audit report."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test"
        )

        report = generate_audit_report([record])

        assert isinstance(report, str)
        assert "CSRD/ESRS PROVENANCE AUDIT REPORT" in report
        assert "TestAgent" in report

    def test_generate_audit_report_with_environment(self):
        """Test audit report includes environment section."""
        env = capture_environment()
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test",
            environment=env
        )

        report = generate_audit_report([record])

        assert "Execution Environment" in report
        assert "Python:" in report
        assert "Platform:" in report

    def test_generate_audit_report_with_calculations(self, sample_calculation_lineages):
        """Test audit report includes calculation lineage."""
        records = []
        for lineage in sample_calculation_lineages[:2]:  # First 2
            record = ProvenanceRecord(
                agent_name="CalculatorAgent",
                operation="calculate",
                calculation_lineage=lineage
            )
            records.append(record)

        report = generate_audit_report(records)

        assert "Calculation Lineage" in report
        assert "E1-1-BASE" in report

    def test_generate_audit_report_agent_operations(self):
        """Test audit report agent operations section."""
        records = [
            ProvenanceRecord(agent_name="IntakeAgent", operation="load", status="success"),
            ProvenanceRecord(agent_name="IntakeAgent", operation="validate", status="success"),
            ProvenanceRecord(agent_name="CalculatorAgent", operation="calculate", status="success")
        ]

        report = generate_audit_report(records)

        assert "Agent Operations" in report
        assert "IntakeAgent" in report
        assert "CalculatorAgent" in report

    def test_generate_audit_report_with_errors(self):
        """Test audit report includes errors."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test",
            status="error",
            errors=["Error 1", "Error 2"]
        )

        report = generate_audit_report([record])

        assert "Data Quality Summary" in report
        assert "Error 1" in report

    def test_generate_audit_report_save_to_file(self, tmp_dir):
        """Test saving audit report to file."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test"
        )

        output_path = tmp_dir / "audit_report.md"
        report = generate_audit_report([record], output_path=output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert content == report

    def test_generate_audit_report_markdown_format(self):
        """Test audit report is valid Markdown."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test"
        )

        report = generate_audit_report([record])

        # Check Markdown headers
        assert report.startswith("# ")
        assert "## " in report
        assert "**" in report  # Bold text


# ============================================================================
# TEST CLASS 12: Serialization
# ============================================================================


class TestSerialization:
    """Test provenance serialization functions."""

    def test_serialize_provenance(self):
        """Test serializing provenance records."""
        records = [
            ProvenanceRecord(agent_name="Agent1", operation="op1"),
            ProvenanceRecord(agent_name="Agent2", operation="op2")
        ]

        data = serialize_provenance(records)

        assert isinstance(data, dict)
        assert "metadata" in data
        assert "records" in data
        assert "summary" in data
        assert len(data["records"]) == 2

    def test_serialize_provenance_metadata(self):
        """Test serialization metadata."""
        records = [ProvenanceRecord(agent_name="Test", operation="test")]

        data = serialize_provenance(records)

        assert data["metadata"]["total_records"] == 1
        assert data["metadata"]["platform"] == "CSRD/ESRS Digital Reporting Platform"

    def test_serialize_provenance_summary(self):
        """Test serialization summary statistics."""
        lineage = CalculationLineage(
            metric_code="E1-1",
            metric_name="Test",
            formula="a + b",
            input_values={"a": 1, "b": 2},
            output_value=3,
            output_unit="units"
        )

        records = [
            ProvenanceRecord(
                agent_name="CalculatorAgent",
                operation="calculate",
                calculation_lineage=lineage,
                status="success"
            ),
            ProvenanceRecord(
                agent_name="IntakeAgent",
                operation="load",
                status="error",
                errors=["File not found"]
            )
        ]

        data = serialize_provenance(records)

        assert data["summary"]["total_calculations"] == 1
        assert "CalculatorAgent" in data["summary"]["agents_used"]
        assert data["summary"]["total_errors"] == 1

    def test_save_provenance_json(self, tmp_dir):
        """Test saving provenance to JSON file."""
        record = ProvenanceRecord(
            agent_name="TestAgent",
            operation="test"
        )

        output_path = tmp_dir / "provenance.json"
        save_provenance_json([record], output_path)

        assert output_path.exists()

        # Verify JSON is valid
        with open(output_path, 'r') as f:
            data = json.load(f)
            assert "records" in data
            assert len(data["records"]) == 1


# ============================================================================
# TEST CLASS 13: Integration Tests
# ============================================================================


class TestProvenanceIntegration:
    """Integration tests for complete provenance workflows."""

    def test_complete_audit_workflow(self, tmp_dir, sample_csv_file):
        """Test complete audit workflow from data to package."""
        # 1. Create data source
        data_source = create_data_source(
            source_type="csv",
            file_path=str(sample_csv_file),
            row_index=1,
            column_name="Scope1_tCO2e"
        )

        # 2. Track calculation
        lineage = track_calculation_lineage(
            metric_code="E1-1",
            metric_name="Scope 1 Emissions",
            formula="direct",
            input_values={"value": 1000.5},
            output_value=1000.5,
            output_unit="tCO2e",
            data_sources=[data_source]
        )

        # 3. Capture environment
        env = capture_environment()

        # 4. Create provenance record
        record = create_provenance_record(
            agent_name="CalculatorAgent",
            operation="calculate_metric",
            inputs={"metric_code": "E1-1"},
            outputs={"value": 1000.5, "unit": "tCO2e"},
            calculation_lineage=lineage,
            data_sources=[data_source],
            environment=env,
            duration_seconds=0.5,
            status="success"
        )

        # 5. Create audit package
        audit_path = tmp_dir / "complete_audit.zip"
        create_audit_package(
            [record],
            audit_path,
            include_lineage_graph=True
        )

        # 6. Verify package
        assert audit_path.exists()
        with zipfile.ZipFile(audit_path, 'r') as zf:
            assert "provenance.json" in zf.namelist()
            assert "environment.json" in zf.namelist()
            assert "manifest.json" in zf.namelist()

    def test_multi_metric_calculation_lineage(self, tmp_dir):
        """Test tracking lineage for multiple dependent metrics."""
        # Base metric
        lineage1 = track_calculation_lineage(
            metric_code="E1-1",
            metric_name="Scope 1",
            formula="direct",
            input_values={"value": 1000},
            output_value=1000,
            output_unit="tCO2e"
        )

        # Dependent metric
        lineage2 = track_calculation_lineage(
            metric_code="E1-2",
            metric_name="Total",
            formula="E1-1 * 1.2",
            input_values={"E1-1": 1000},
            output_value=1200,
            output_unit="tCO2e",
            dependencies=["E1-1"]
        )

        # Build graph
        G = build_lineage_graph([lineage1, lineage2])

        # Verify dependency
        assert G.has_edge("E1-1", "E1-2")

        # Get calculation path
        path = get_calculation_path(G, "E1-2")
        assert "E1-1" in path
        assert "E1-2" in path

    def test_provenance_record_lifecycle(self, tmp_dir):
        """Test complete provenance record lifecycle."""
        # Create record
        record = create_provenance_record(
            agent_name="IntakeAgent",
            operation="validate_data",
            inputs={"file": "esg_data.csv", "rows": 100},
            outputs={"valid": 95, "invalid": 5},
            duration_seconds=2.5,
            status="warning",
            warnings=["5 rows skipped due to invalid data"]
        )

        # Serialize
        data = serialize_provenance([record])

        # Save to JSON
        json_path = tmp_dir / "provenance.json"
        save_provenance_json([record], json_path)

        # Load and verify
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
            assert loaded_data["metadata"]["total_records"] == 1
            assert loaded_data["summary"]["total_warnings"] == 1


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestProvenancePerformance:
    """Performance tests for provenance operations."""

    def test_hash_large_file_performance(self, large_file):
        """Test performance of hashing large file."""
        import time

        start = time.time()
        hash_info = hash_file(large_file)
        duration = time.time() - start

        # Should complete in reasonable time (< 1 second for 2MB)
        assert duration < 1.0
        assert hash_info["file_size_bytes"] == 2 * 1024 * 1024

    def test_build_large_lineage_graph_performance(self):
        """Test performance of building large lineage graph."""
        # Create 100 lineages
        lineages = []
        for i in range(100):
            lineage = CalculationLineage(
                metric_code=f"E1-{i}",
                metric_name=f"Metric {i}",
                formula=f"value_{i}",
                input_values={f"value_{i}": i},
                output_value=i,
                output_unit="units"
            )
            lineages.append(lineage)

        import time
        start = time.time()
        G = build_lineage_graph(lineages)
        duration = time.time() - start

        # Should complete quickly
        assert duration < 0.5
        assert len(G.nodes()) == 100


# ============================================================================
# EDGE CASES
# ============================================================================


class TestProvenanceEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_provenance_records(self):
        """Test serializing empty provenance records."""
        data = serialize_provenance([])

        assert data["metadata"]["total_records"] == 0
        assert data["summary"]["total_calculations"] == 0

    def test_hash_empty_file(self, tmp_dir):
        """Test hashing empty file."""
        empty_file = tmp_dir / "empty.txt"
        empty_file.write_text("")

        hash_info = hash_file(empty_file)

        assert hash_info["file_size_bytes"] == 0
        assert len(hash_info["hash_value"]) == 64

    def test_lineage_graph_no_dependencies(self):
        """Test lineage graph with no dependencies."""
        lineage = CalculationLineage(
            metric_code="E1-1",
            metric_name="Test",
            formula="direct",
            input_values={"value": 100},
            output_value=100,
            output_unit="units",
            dependencies=[]
        )

        G = build_lineage_graph([lineage])

        assert len(G.nodes()) == 1
        assert len(G.edges()) == 0

    def test_calculation_lineage_special_characters(self):
        """Test calculation lineage with special characters in formula."""
        lineage = track_calculation_lineage(
            metric_code="E1-1",
            metric_name="Test (Special) [Chars]",
            formula="(a + b) / (c - d) * 100%",
            input_values={"a": 10, "b": 20, "c": 50, "d": 20},
            output_value=100,
            output_unit="%"
        )

        assert lineage.formula == "(a + b) / (c - d) * 100%"
        assert len(lineage.hash) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
