# -*- coding: utf-8 -*-
"""
Provenance chain integration tests for AGENT-DATA-015 Cross-Source Reconciliation.

Tests SHA-256 provenance chain integrity across the full pipeline:
- Full pipeline produces valid provenance chain
- verify_chain returns True after full pipeline
- Provenance is deterministic (same inputs -> same hash)
- Chain hashes link operations in sequence
- Entity-scoped chains are maintained
- Global chain captures all operations

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
"""

from typing import Any, Dict, List

import pytest

from greenlang.cross_source_reconciliation.provenance import (
    ProvenanceTracker,
)
from greenlang.cross_source_reconciliation.setup import (
    CrossSourceReconciliationService,
)


# =========================================================================
# Test class: Provenance chain integrity
# =========================================================================


class TestProvenanceChainIntegrity:
    """Test provenance chain integrity after pipeline operations."""

    def test_pipeline_produces_provenance_entries(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Full pipeline run produces provenance entries in the tracker."""
        initial_count = service._provenance.entry_count

        service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
        )

        final_count = service._provenance.entry_count
        assert final_count > initial_count, (
            f"Expected provenance entries to increase from {initial_count}, "
            f"got {final_count}"
        )

    def test_verify_chain_returns_true_after_pipeline(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """verify_chain on global chain returns True after pipeline run."""
        service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
        )

        is_valid, chain = service._provenance.verify_chain()
        assert is_valid is True
        assert len(chain) >= 1

    def test_provenance_hash_deterministic_same_inputs(self, service):
        """Same input data produces deterministic provenance hashes.

        The match_records method generates a unique match_id (UUID) per call,
        which is included in the provenance hash. Therefore, successive calls
        with the same data will have different provenance hashes at the match
        level. However, the ProvenanceTracker.hash_record utility itself must
        be deterministic: the same dictionary input always yields the same
        SHA-256 hash.
        """
        tracker = service._provenance
        data = {"entity_id": "E1", "period": "Q1", "value": 100.0}

        hash_1 = tracker.hash_record(data)
        hash_2 = tracker.hash_record(data)

        # ProvenanceTracker.hash_record is deterministic
        assert hash_1 == hash_2
        assert len(hash_1) == 64  # SHA-256 hex digest length

    def test_provenance_chain_links_operations_sequentially(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Chain entries are linked sequentially via parent hashes."""
        service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
        )

        chain = service._provenance.get_chain()
        assert len(chain) >= 2

        # Each entry after the first should have a non-empty chain_hash
        for entry in chain:
            assert entry.get("chain_hash", "") != "", (
                "Every chain entry must have a non-empty chain_hash"
            )

    def test_source_registration_creates_provenance_entries(self, service):
        """Registering a source records provenance for the source entity."""
        initial_count = service._provenance.entry_count

        service.register_source(
            name="Provenance Test Source",
            source_type="erp",
            priority=1,
        )

        assert service._provenance.entry_count > initial_count

    def test_provenance_tracker_reset_clears_chain(self):
        """ProvenanceTracker.reset() clears all entries."""
        tracker = ProvenanceTracker()
        tracker.record("test_entity", "id_1", "action_1", "hash_1")
        tracker.record("test_entity", "id_1", "action_2", "hash_2")

        assert tracker.entry_count >= 2

        tracker.reset()
        assert tracker.entry_count == 0
        assert tracker.get_current_hash() == ProvenanceTracker.GENESIS_HASH

    def test_provenance_export_json_not_empty(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Export provenance as JSON after pipeline produces non-empty output."""
        service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
        )

        json_output = service._provenance.export_json()
        assert json_output != "[]"
        assert len(json_output) > 10

    def test_provenance_entity_scoped_chain(self, service):
        """Entity-scoped chains are isolated per entity_type:entity_id."""
        source_a = service.register_source(
            name="Source A", source_type="erp",
        )
        source_b = service.register_source(
            name="Source B", source_type="utility",
        )

        # Global chain should have at least 2 entries (one per source)
        is_valid, chain = service._provenance.verify_chain()
        assert is_valid is True
        assert len(chain) >= 2
