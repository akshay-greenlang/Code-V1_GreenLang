# -*- coding: utf-8 -*-
"""
Provenance chain integration tests for AGENT-DATA-014 Time Series Gap Filler.

Tests SHA-256 provenance chain integrity across pipeline stages:
- Chain maintained across detect -> fill -> validate
- Chain verification after full pipeline run
- Multiple operations produce linked chain entries
- Deterministic hashing for identical inputs
- Chain export and verification
- Provenance entries contain required fields

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

import json
from typing import List, Optional

import pytest

from greenlang.time_series_gap_filler.gap_detector import GapDetectorEngine
from greenlang.time_series_gap_filler.interpolation_engine import InterpolationEngine
from greenlang.time_series_gap_filler.seasonal_filler import SeasonalFillerEngine
from greenlang.time_series_gap_filler.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    get_provenance_tracker,
)


# =========================================================================
# Test class: Provenance chain across operations
# =========================================================================


class TestProvenanceChainIntegration:
    """Integration tests for provenance chain integrity."""

    def test_provenance_chain_across_detect_and_fill(
        self, sample_series_with_gaps,
    ):
        """Provenance chain grows when detect and fill are called."""
        tracker = ProvenanceTracker()

        # Record a detect operation
        input_hash = tracker.hash_record({"values": sample_series_with_gaps})
        detect_hash = tracker.add_to_chain(
            operation="detect_gaps",
            input_hash=input_hash,
            output_hash=tracker.hash_record({"total_gaps": 3}),
            metadata={"engine": "gap_detector"},
        )
        assert len(detect_hash) == 64

        # Record a fill operation
        fill_hash = tracker.add_to_chain(
            operation="fill_gaps",
            input_hash=detect_hash,
            output_hash=tracker.hash_record({"gaps_filled": 3}),
            metadata={"engine": "interpolation", "method": "linear"},
        )
        assert len(fill_hash) == 64
        assert fill_hash != detect_hash

        # Verify chain integrity
        is_valid, chain = tracker.verify_chain()
        assert is_valid is True
        assert len(chain) >= 2

    def test_chain_verification_after_full_pipeline(
        self, sample_series_with_gaps,
    ):
        """Verify the chain after running detect + fill on real engines."""
        detector = GapDetectorEngine()
        interpolator = InterpolationEngine()

        # Run real operations which internally record provenance
        detection = detector.detect_gaps(sample_series_with_gaps)
        fill_result = interpolator.fill_gaps(
            sample_series_with_gaps, method="linear",
        )

        # Both should have valid SHA-256 hashes
        assert len(detection.provenance_hash) == 64
        assert len(fill_result.provenance_hash) == 64

        # Hashes should be different (different operations)
        assert detection.provenance_hash != fill_result.provenance_hash

    def test_multiple_operations_produce_linked_chain(self):
        """Multiple add_to_chain calls produce a valid linked chain."""
        tracker = ProvenanceTracker()

        hashes = []
        for i in range(5):
            h = tracker.add_to_chain(
                operation=f"step_{i}",
                input_hash=tracker.hash_record({"step": i, "input": True}),
                output_hash=tracker.hash_record({"step": i, "output": True}),
                metadata={"iteration": i},
            )
            hashes.append(h)

        # All hashes should be unique
        assert len(set(hashes)) == 5

        # Chain should be valid
        is_valid, chain = tracker.verify_chain()
        assert is_valid is True
        assert len(chain) == 5

    def test_deterministic_hashing_for_same_input(self):
        """Identical data produces the same hash every time."""
        tracker = ProvenanceTracker()

        data = {"series_id": "test_001", "values": [1.0, 2.0, 3.0], "method": "linear"}
        hash1 = tracker.hash_record(data)
        hash2 = tracker.hash_record(data)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_provenance_entry_contains_required_fields(self):
        """ProvenanceEntry objects contain all required audit fields."""
        tracker = ProvenanceTracker()

        entry = tracker.add_entry(
            operation="test_op",
            input_hash="a" * 64,
            output_hash="b" * 64,
            metadata={"test_key": "test_value"},
        )

        assert isinstance(entry, ProvenanceEntry)
        assert entry.entry_id != ""
        assert entry.operation == "test_op"
        assert entry.input_hash == "a" * 64
        assert entry.output_hash == "b" * 64
        assert entry.timestamp != ""
        assert entry.parent_hash != ""
        assert len(entry.chain_hash) == 64
        assert entry.metadata["test_key"] == "test_value"

    def test_chain_export_and_reimport(self):
        """Exported JSON can be parsed back and chain length matches."""
        tracker = ProvenanceTracker()

        for i in range(3):
            tracker.record(
                entity_type="gap_fill_job",
                entity_id=f"job_{i:03d}",
                action="fill",
                data_hash=tracker.hash_record({"job": i}),
            )

        json_str = tracker.export_json()
        parsed = json.loads(json_str)

        assert len(parsed) == 3
        for entry in parsed:
            assert "chain_hash" in entry
            assert entry["chain_hash"] != ""

    def test_entity_scoped_chain_verification(self):
        """Verify chain for a specific entity type and ID."""
        tracker = ProvenanceTracker()

        # Record entries for different entities
        tracker.record("gap_fill_job", "job_A", "detect", tracker.hash_record({"a": 1}))
        tracker.record("gap_fill_job", "job_A", "fill", tracker.hash_record({"a": 2}))
        tracker.record("gap_fill_job", "job_B", "detect", tracker.hash_record({"b": 1}))

        # Verify entity A chain
        is_valid_a, chain_a = tracker.verify_chain("gap_fill_job", "job_A")
        assert is_valid_a is True
        assert len(chain_a) == 2

        # Verify entity B chain
        is_valid_b, chain_b = tracker.verify_chain("gap_fill_job", "job_B")
        assert is_valid_b is True
        assert len(chain_b) == 1

        # Global chain has all entries
        is_valid_all, chain_all = tracker.verify_chain()
        assert is_valid_all is True
        assert len(chain_all) == 3

    def test_provenance_reset_clears_all_chains(self):
        """Resetting the tracker clears all stored chains."""
        tracker = ProvenanceTracker()

        tracker.record("test", "t1", "act", tracker.hash_record({"x": 1}))
        assert tracker.entry_count == 1

        tracker.reset()
        assert tracker.entry_count == 0

        _, chain = tracker.verify_chain()
        assert len(chain) == 0

    def test_fill_result_provenance_hash_is_sha256(
        self, sample_series_with_gaps,
    ):
        """Fill result provenance_hash is a valid 64-char hex SHA-256."""
        interpolator = InterpolationEngine()
        result = interpolator.fill_gaps(sample_series_with_gaps, method="linear")

        h = result.provenance_hash
        assert len(h) == 64
        # Should be valid hexadecimal
        int(h, 16)

    def test_detection_provenance_hash_is_sha256(
        self, sample_series_with_gaps,
    ):
        """Detection result provenance_hash is a valid 64-char hex SHA-256."""
        detector = GapDetectorEngine()
        detection = detector.detect_gaps(sample_series_with_gaps)

        h = detection.provenance_hash
        assert len(h) == 64
        int(h, 16)
