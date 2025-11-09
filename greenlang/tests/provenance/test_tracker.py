"""
Tests for ProvenanceTracker
"""

import pytest
import tempfile
from pathlib import Path
from greenlang.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    ProvenanceContext,
    track_with_provenance,
)


class TestProvenanceTracker:
    """Test ProvenanceTracker."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = ProvenanceTracker(name="test_tracker")
        assert tracker.name == "test_tracker"
        assert tracker.context is not None

    def test_track_operation(self):
        """Test operation tracking."""
        tracker = ProvenanceTracker()

        with tracker.track_operation("test_op"):
            tracker.context.add_metadata("test_key", "test_value")

        assert len(tracker.chain_of_custody) == 1
        assert tracker.chain_of_custody[0]["name"] == "test_op"
        assert tracker.chain_of_custody[0]["status"] == "success"

    def test_track_file_input(self):
        """Test file input tracking."""
        tracker = ProvenanceTracker(auto_hash_files=True)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test data")
            temp_path = f.name

        try:
            file_info = tracker.track_file_input(temp_path)

            assert file_info["path"] == temp_path
            assert "hash" in file_info
            assert file_info["size_bytes"] > 0

        finally:
            Path(temp_path).unlink()

    def test_track_file_output(self):
        """Test file output tracking."""
        tracker = ProvenanceTracker(auto_hash_files=True)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("output data")
            temp_path = f.name

        try:
            file_info = tracker.track_file_output(temp_path)

            assert file_info["path"] == temp_path
            assert "hash" in file_info

        finally:
            Path(temp_path).unlink()

    def test_track_data_transformation(self):
        """Test data transformation tracking."""
        tracker = ProvenanceTracker()

        tracker.track_data_transformation(
            source="input.csv",
            destination="output.csv",
            transformation="data cleaning",
            input_records=100,
            output_records=95
        )

        assert len(tracker.context.data_lineage) == 1
        lineage = tracker.context.data_lineage[0]
        assert lineage["source"] == "input.csv"
        assert lineage["destination"] == "output.csv"
        assert lineage["input_records"] == 100

    def test_get_record(self):
        """Test getting provenance record."""
        tracker = ProvenanceTracker(name="test")

        tracker.add_metadata("test", "value")

        record = tracker.get_record()

        assert isinstance(record, ProvenanceRecord)
        assert record.record_id is not None
        assert "test" in record.metadata

    def test_save_record(self):
        """Test saving provenance record."""
        tracker = ProvenanceTracker()

        tracker.add_metadata("test", "value")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            saved_path = tracker.save_record(temp_path)
            assert Path(saved_path).exists()

            # Verify can be loaded
            loaded_record = ProvenanceRecord.load(saved_path)
            assert loaded_record.metadata["test"] == "value"

        finally:
            Path(temp_path).unlink()


class TestProvenanceDecorator:
    """Test provenance decorator."""

    def test_decorator_basic(self):
        """Test basic decorator usage."""
        tracker = ProvenanceTracker()

        @track_with_provenance(tracker, "test_function")
        def test_func():
            return "result"

        result = test_func()

        assert result == "result"
        assert len(tracker.chain_of_custody) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
