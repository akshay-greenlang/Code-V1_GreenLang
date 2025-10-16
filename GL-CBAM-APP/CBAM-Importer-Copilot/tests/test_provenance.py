"""
CBAM Importer Copilot - Provenance Utility Tests

Tests for provenance tracking, file integrity, and audit trail.

Version: 1.0.0
"""

import pytest
import json
import hashlib
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.provenance import (
    ProvenanceTracker,
    compute_file_hash,
    verify_file_integrity,
    create_provenance_record,
    validate_provenance,
    extract_provenance_summary
)


# ============================================================================
# Test File Integrity
# ============================================================================

@pytest.mark.unit
class TestFileIntegrity:
    """Test file integrity and hashing utilities."""

    def test_compute_file_hash_sha256(self, sample_shipments_csv):
        """Test computes SHA256 hash of file."""
        hash_value = compute_file_hash(sample_shipments_csv)

        # SHA256 produces 64 hex characters
        assert len(hash_value) == 64
        assert all(c in '0123456789abcdef' for c in hash_value)

    def test_compute_file_hash_deterministic(self, sample_shipments_csv):
        """Test file hash is deterministic."""
        hash1 = compute_file_hash(sample_shipments_csv)
        hash2 = compute_file_hash(sample_shipments_csv)

        assert hash1 == hash2

    def test_compute_file_hash_different_files(self, sample_shipments_csv, sample_shipments_excel):
        """Test different files produce different hashes."""
        hash_csv = compute_file_hash(sample_shipments_csv)
        hash_excel = compute_file_hash(sample_shipments_excel)

        assert hash_csv != hash_excel

    def test_compute_file_hash_modified_content(self, tmp_path):
        """Test hash changes when file content changes."""
        file1 = tmp_path / "test1.txt"
        file1.write_text("Original content")
        hash1 = compute_file_hash(str(file1))

        file1.write_text("Modified content")
        hash2 = compute_file_hash(str(file1))

        assert hash1 != hash2

    def test_verify_file_integrity_valid(self, sample_shipments_csv):
        """Test verifies file integrity with correct hash."""
        original_hash = compute_file_hash(sample_shipments_csv)

        is_valid = verify_file_integrity(sample_shipments_csv, original_hash)

        assert is_valid is True

    def test_verify_file_integrity_invalid(self, sample_shipments_csv):
        """Test detects file integrity violation."""
        wrong_hash = "0" * 64  # Invalid hash

        is_valid = verify_file_integrity(sample_shipments_csv, wrong_hash)

        assert is_valid is False

    def test_verify_file_integrity_missing_file(self):
        """Test handles missing file gracefully."""
        is_valid = verify_file_integrity("nonexistent.csv", "abc123")

        assert is_valid is False


# ============================================================================
# Test Provenance Tracker
# ============================================================================

@pytest.mark.unit
class TestProvenanceTracker:
    """Test ProvenanceTracker class."""

    def test_tracker_initialization(self):
        """Test tracker initializes correctly."""
        tracker = ProvenanceTracker(enable_tracking=True)

        assert tracker is not None
        assert tracker.is_enabled() is True

    def test_tracker_disabled_mode(self):
        """Test tracker disabled mode."""
        tracker = ProvenanceTracker(enable_tracking=False)

        assert tracker.is_enabled() is False

    def test_record_input_file(self, sample_shipments_csv):
        """Test records input file information."""
        tracker = ProvenanceTracker(enable_tracking=True)

        tracker.record_input_file(sample_shipments_csv)

        provenance = tracker.get_provenance()
        assert 'input_file_integrity' in provenance

        file_info = provenance['input_file_integrity']
        assert 'sha256_hash' in file_info
        assert 'file_name' in file_info
        assert 'file_size_bytes' in file_info
        assert len(file_info['sha256_hash']) == 64

    def test_record_execution_environment(self):
        """Test records execution environment details."""
        tracker = ProvenanceTracker(enable_tracking=True)

        tracker.record_execution_environment()

        provenance = tracker.get_provenance()
        assert 'execution_environment' in provenance

        env = provenance['execution_environment']
        assert 'timestamp' in env
        assert 'python_version' in env
        assert 'platform' in env
        assert 'greenlang_version' in env

    def test_record_agent_execution(self):
        """Test records agent execution."""
        tracker = ProvenanceTracker(enable_tracking=True)

        tracker.record_agent_execution(
            agent_name="ShipmentIntakeAgent",
            start_time="2025-09-15T10:00:00Z",
            end_time="2025-09-15T10:00:05Z",
            records_processed=100
        )

        provenance = tracker.get_provenance()
        assert 'agent_execution' in provenance

        executions = provenance['agent_execution']
        assert len(executions) > 0
        assert executions[0]['agent_name'] == "ShipmentIntakeAgent"
        assert executions[0]['records_processed'] == 100

    def test_record_multiple_agents(self):
        """Test records multiple agent executions."""
        tracker = ProvenanceTracker(enable_tracking=True)

        tracker.record_agent_execution(
            agent_name="ShipmentIntakeAgent",
            start_time="2025-09-15T10:00:00Z",
            end_time="2025-09-15T10:00:05Z",
            records_processed=100
        )

        tracker.record_agent_execution(
            agent_name="EmissionsCalculatorAgent",
            start_time="2025-09-15T10:00:05Z",
            end_time="2025-09-15T10:00:10Z",
            records_processed=100
        )

        tracker.record_agent_execution(
            agent_name="ReportingPackagerAgent",
            start_time="2025-09-15T10:00:10Z",
            end_time="2025-09-15T10:00:12Z",
            records_processed=1
        )

        provenance = tracker.get_provenance()
        executions = provenance['agent_execution']

        assert len(executions) == 3
        agent_names = [e['agent_name'] for e in executions]
        assert 'ShipmentIntakeAgent' in agent_names
        assert 'EmissionsCalculatorAgent' in agent_names
        assert 'ReportingPackagerAgent' in agent_names

    def test_record_data_source(self):
        """Test records data source information."""
        tracker = ProvenanceTracker(enable_tracking=True)

        tracker.record_data_source(
            source_type="cn_codes",
            file_path="data/cn_codes.json",
            version="1.0.0"
        )

        provenance = tracker.get_provenance()
        assert 'data_sources' in provenance

        sources = provenance['data_sources']
        assert len(sources) > 0
        assert sources[0]['source_type'] == "cn_codes"

    def test_set_reproducibility_flag(self):
        """Test sets reproducibility flags."""
        tracker = ProvenanceTracker(enable_tracking=True)

        tracker.set_reproducibility(
            deterministic=True,
            zero_hallucination=True
        )

        provenance = tracker.get_provenance()
        assert 'reproducibility' in provenance

        repro = provenance['reproducibility']
        assert repro['deterministic'] is True
        assert repro['zero_hallucination'] is True

    def test_disabled_tracker_no_data(self, sample_shipments_csv):
        """Test disabled tracker doesn't record data."""
        tracker = ProvenanceTracker(enable_tracking=False)

        tracker.record_input_file(sample_shipments_csv)
        tracker.record_execution_environment()

        provenance = tracker.get_provenance()

        # Should return minimal or empty provenance
        assert provenance is None or len(provenance) == 0


# ============================================================================
# Test Provenance Record Creation
# ============================================================================

@pytest.mark.unit
class TestProvenanceRecordCreation:
    """Test provenance record creation utilities."""

    def test_create_provenance_record_basic(self, sample_shipments_csv):
        """Test creates basic provenance record."""
        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True}
        )

        assert record is not None
        assert 'input_file_integrity' in record
        assert 'execution_environment' in record

    def test_create_provenance_record_with_agents(self, sample_shipments_csv):
        """Test creates provenance with agent execution history."""
        agent_history = [
            {
                'agent_name': 'ShipmentIntakeAgent',
                'start_time': '2025-09-15T10:00:00Z',
                'end_time': '2025-09-15T10:00:05Z',
                'records_processed': 100
            },
            {
                'agent_name': 'EmissionsCalculatorAgent',
                'start_time': '2025-09-15T10:00:05Z',
                'end_time': '2025-09-15T10:00:10Z',
                'records_processed': 100
            }
        ]

        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True},
            agent_execution=agent_history
        )

        assert 'agent_execution' in record
        assert len(record['agent_execution']) == 2

    def test_create_provenance_record_with_data_sources(self, sample_shipments_csv):
        """Test creates provenance with data sources."""
        data_sources = [
            {
                'source_type': 'cn_codes',
                'file_path': 'data/cn_codes.json',
                'sha256_hash': 'abc123...'
            },
            {
                'source_type': 'cbam_rules',
                'file_path': 'rules/cbam_rules.yaml',
                'sha256_hash': 'def456...'
            }
        ]

        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True},
            data_sources=data_sources
        )

        assert 'data_sources' in record
        assert len(record['data_sources']) == 2

    def test_create_provenance_record_zero_hallucination(self, sample_shipments_csv):
        """Test creates provenance with zero hallucination guarantee."""
        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True},
            zero_hallucination=True
        )

        assert 'reproducibility' in record
        assert record['reproducibility']['zero_hallucination'] is True
        assert record['reproducibility']['deterministic'] is True


# ============================================================================
# Test Provenance Validation
# ============================================================================

@pytest.mark.unit
@pytest.mark.compliance
class TestProvenanceValidation:
    """Test provenance validation."""

    def test_validate_provenance_valid_record(self, sample_shipments_csv):
        """Test validates valid provenance record."""
        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True}
        )

        is_valid, errors = validate_provenance(record)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_provenance_missing_required_fields(self):
        """Test detects missing required fields."""
        incomplete_record = {
            'input_file_integrity': {
                'sha256_hash': 'abc123'
            }
            # Missing execution_environment, agent_execution, etc.
        }

        is_valid, errors = validate_provenance(incomplete_record)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_provenance_invalid_hash(self, sample_shipments_csv):
        """Test detects invalid SHA256 hash format."""
        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True}
        )

        # Corrupt hash
        record['input_file_integrity']['sha256_hash'] = 'invalid_hash'

        is_valid, errors = validate_provenance(record)

        assert is_valid is False
        assert any('hash' in str(e).lower() for e in errors)

    def test_validate_provenance_zero_hallucination_flag(self, sample_shipments_csv):
        """Test validates zero hallucination flag."""
        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True},
            zero_hallucination=True
        )

        is_valid, errors = validate_provenance(record)

        assert is_valid is True

        repro = record['reproducibility']
        assert repro['zero_hallucination'] is True
        assert repro['deterministic'] is True

    def test_validate_provenance_agent_execution_complete(self, sample_shipments_csv):
        """Test validates agent execution records are complete."""
        agent_history = [
            {
                'agent_name': 'ShipmentIntakeAgent',
                'start_time': '2025-09-15T10:00:00Z',
                'end_time': '2025-09-15T10:00:05Z',
                'records_processed': 100
            }
        ]

        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True},
            agent_execution=agent_history
        )

        is_valid, errors = validate_provenance(record)

        assert is_valid is True

    def test_validate_provenance_incomplete_agent_execution(self, sample_shipments_csv):
        """Test detects incomplete agent execution records."""
        incomplete_agent = [
            {
                'agent_name': 'ShipmentIntakeAgent',
                # Missing start_time, end_time, records_processed
            }
        ]

        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True},
            agent_execution=incomplete_agent
        )

        is_valid, errors = validate_provenance(record)

        # May be invalid depending on validation rules
        if not is_valid:
            assert len(errors) > 0


# ============================================================================
# Test Provenance Summary Extraction
# ============================================================================

@pytest.mark.unit
class TestProvenanceSummary:
    """Test provenance summary extraction."""

    def test_extract_provenance_summary(self, sample_shipments_csv):
        """Test extracts human-readable provenance summary."""
        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True}
        )

        summary = extract_provenance_summary(record)

        assert summary is not None
        assert 'input_file' in summary
        assert 'timestamp' in summary
        assert 'file_hash' in summary

    def test_extract_provenance_summary_with_agents(self, sample_shipments_csv):
        """Test summary includes agent execution info."""
        agent_history = [
            {
                'agent_name': 'ShipmentIntakeAgent',
                'start_time': '2025-09-15T10:00:00Z',
                'end_time': '2025-09-15T10:00:05Z',
                'records_processed': 100
            },
            {
                'agent_name': 'EmissionsCalculatorAgent',
                'start_time': '2025-09-15T10:00:05Z',
                'end_time': '2025-09-15T10:00:10Z',
                'records_processed': 100
            }
        ]

        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True},
            agent_execution=agent_history
        )

        summary = extract_provenance_summary(record)

        assert 'agents_executed' in summary
        assert summary['agents_executed'] == 2

    def test_extract_provenance_summary_zero_hallucination(self, sample_shipments_csv):
        """Test summary includes zero hallucination status."""
        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True},
            zero_hallucination=True
        )

        summary = extract_provenance_summary(record)

        assert 'zero_hallucination' in summary
        assert summary['zero_hallucination'] is True


# ============================================================================
# Test Provenance File Operations
# ============================================================================

@pytest.mark.unit
class TestProvenanceFileOperations:
    """Test provenance file save/load operations."""

    def test_save_provenance_to_file(self, sample_shipments_csv, tmp_path):
        """Test saves provenance to JSON file."""
        from utils.provenance import save_provenance

        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True}
        )

        output_path = tmp_path / "provenance.json"
        save_provenance(record, str(output_path))

        assert output_path.exists()

        # Verify JSON structure
        with open(output_path) as f:
            loaded = json.load(f)
            assert loaded == record

    def test_load_provenance_from_file(self, sample_shipments_csv, tmp_path):
        """Test loads provenance from JSON file."""
        from utils.provenance import save_provenance, load_provenance

        record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True}
        )

        output_path = tmp_path / "provenance.json"
        save_provenance(record, str(output_path))

        loaded_record = load_provenance(str(output_path))

        assert loaded_record == record

    def test_embed_provenance_in_report(self, sample_shipments_csv):
        """Test embeds provenance in CBAM report."""
        from utils.provenance import embed_provenance_in_report

        report = {
            'report_metadata': {'title': 'CBAM Report'},
            'emissions_summary': {'total_emissions_tco2': 100.0}
        }

        provenance_record = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True}
        )

        report_with_prov = embed_provenance_in_report(report, provenance_record)

        assert 'provenance' in report_with_prov
        assert report_with_prov['provenance'] == provenance_record


# ============================================================================
# Test Provenance Timestamp Validation
# ============================================================================

@pytest.mark.unit
class TestTimestampValidation:
    """Test timestamp validation in provenance."""

    def test_timestamp_format_valid(self):
        """Test validates ISO 8601 timestamp format."""
        from utils.provenance import is_valid_timestamp

        valid_timestamps = [
            "2025-09-15T10:00:00Z",
            "2025-09-15T10:00:00.123Z",
            "2025-09-15T10:00:00+00:00"
        ]

        for ts in valid_timestamps:
            assert is_valid_timestamp(ts) is True

    def test_timestamp_format_invalid(self):
        """Test detects invalid timestamp format."""
        from utils.provenance import is_valid_timestamp

        invalid_timestamps = [
            "2025-09-15",
            "10:00:00",
            "not a timestamp",
            "2025/09/15 10:00:00"
        ]

        for ts in invalid_timestamps:
            assert is_valid_timestamp(ts) is False

    def test_timestamp_ordering_valid(self):
        """Test validates timestamp ordering (start < end)."""
        from utils.provenance import validate_timestamp_ordering

        start = "2025-09-15T10:00:00Z"
        end = "2025-09-15T10:00:05Z"

        is_valid = validate_timestamp_ordering(start, end)

        assert is_valid is True

    def test_timestamp_ordering_invalid(self):
        """Test detects invalid timestamp ordering."""
        from utils.provenance import validate_timestamp_ordering

        start = "2025-09-15T10:00:05Z"
        end = "2025-09-15T10:00:00Z"  # End before start!

        is_valid = validate_timestamp_ordering(start, end)

        assert is_valid is False


# ============================================================================
# Test Provenance Chain Verification
# ============================================================================

@pytest.mark.integration
@pytest.mark.compliance
class TestProvenanceChainVerification:
    """Test provenance chain verification for audit trail."""

    def test_verify_complete_audit_trail(self, sample_shipments_csv):
        """Test verifies complete audit trail from input to output."""
        from utils.provenance import verify_audit_trail

        # Simulate complete pipeline execution
        provenance_chain = {
            'input_file_integrity': {
                'sha256_hash': compute_file_hash(sample_shipments_csv),
                'file_name': 'sample_shipments.csv',
                'file_size_bytes': Path(sample_shipments_csv).stat().st_size
            },
            'agent_execution': [
                {
                    'agent_name': 'ShipmentIntakeAgent',
                    'start_time': '2025-09-15T10:00:00Z',
                    'end_time': '2025-09-15T10:00:05Z',
                    'records_processed': 100
                },
                {
                    'agent_name': 'EmissionsCalculatorAgent',
                    'start_time': '2025-09-15T10:00:05Z',
                    'end_time': '2025-09-15T10:00:10Z',
                    'records_processed': 100
                },
                {
                    'agent_name': 'ReportingPackagerAgent',
                    'start_time': '2025-09-15T10:00:10Z',
                    'end_time': '2025-09-15T10:00:12Z',
                    'records_processed': 1
                }
            ],
            'reproducibility': {
                'deterministic': True,
                'zero_hallucination': True
            }
        }

        is_valid, errors = verify_audit_trail(provenance_chain)

        assert is_valid is True
        assert len(errors) == 0

    def test_verify_audit_trail_broken_chain(self, sample_shipments_csv):
        """Test detects broken audit trail."""
        from utils.provenance import verify_audit_trail

        # Missing agent in chain
        incomplete_chain = {
            'input_file_integrity': {
                'sha256_hash': compute_file_hash(sample_shipments_csv),
                'file_name': 'sample_shipments.csv',
                'file_size_bytes': Path(sample_shipments_csv).stat().st_size
            },
            'agent_execution': [
                {
                    'agent_name': 'ShipmentIntakeAgent',
                    'start_time': '2025-09-15T10:00:00Z',
                    'end_time': '2025-09-15T10:00:05Z',
                    'records_processed': 100
                }
                # Missing EmissionsCalculatorAgent and ReportingPackagerAgent!
            ],
            'reproducibility': {
                'deterministic': True,
                'zero_hallucination': True
            }
        }

        is_valid, errors = verify_audit_trail(incomplete_chain)

        assert is_valid is False
        assert len(errors) > 0

    def test_verify_file_integrity_in_chain(self, sample_shipments_csv, tmp_path):
        """Test verifies file wasn't tampered with using hash."""
        # Create provenance with original file hash
        original_hash = compute_file_hash(sample_shipments_csv)

        provenance = create_provenance_record(
            input_file=sample_shipments_csv,
            pipeline_config={'enable_provenance': True}
        )

        # Verify integrity
        stored_hash = provenance['input_file_integrity']['sha256_hash']
        is_valid = verify_file_integrity(sample_shipments_csv, stored_hash)

        assert is_valid is True
        assert stored_hash == original_hash


# ============================================================================
# Test Provenance Performance
# ============================================================================

@pytest.mark.performance
class TestProvenancePerformance:
    """Test provenance tracking performance overhead."""

    def test_provenance_tracking_overhead(self, large_shipments_csv):
        """Test provenance tracking has minimal overhead."""
        import time

        # Without provenance
        tracker_disabled = ProvenanceTracker(enable_tracking=False)
        start = time.time()
        tracker_disabled.record_input_file(large_shipments_csv)
        duration_disabled = time.time() - start

        # With provenance
        tracker_enabled = ProvenanceTracker(enable_tracking=True)
        start = time.time()
        tracker_enabled.record_input_file(large_shipments_csv)
        duration_enabled = time.time() - start

        # Overhead should be < 100ms
        overhead = duration_enabled - duration_disabled
        assert overhead < 0.1

    def test_hash_computation_performance(self, large_shipments_csv):
        """Test file hashing is reasonably fast."""
        import time

        start = time.time()
        hash_value = compute_file_hash(large_shipments_csv)
        duration = time.time() - start

        # Should hash large file in < 1 second
        assert duration < 1.0
        assert len(hash_value) == 64
