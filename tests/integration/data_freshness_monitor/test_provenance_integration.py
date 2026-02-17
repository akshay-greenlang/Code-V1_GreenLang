# -*- coding: utf-8 -*-
"""
Provenance chain integration tests for AGENT-DATA-016 Data Freshness Monitor.

Tests provenance chain integrity across multiple operations: dataset
registration, SLA creation, freshness checks, breach detection, alert
generation, and full pipeline execution.

8+ tests covering:
- Provenance chain grows across operations
- Chain verification after full pipeline run
- Deterministic provenance hashes for identical inputs
- Chain integrity after mixed operations

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

from datetime import datetime, timedelta, timezone

import pytest

from greenlang.data_freshness_monitor.provenance import ProvenanceTracker


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ===================================================================
# Provenance Chain Across Operations
# ===================================================================


class TestProvenanceChainAcrossOperations:
    """Test that provenance entries accumulate correctly as service
    operations are performed."""

    def test_registration_creates_provenance_entry(self, service):
        """Verify that registering a dataset adds a provenance entry
        and the entry_count increases."""
        stats_before = service.get_stats()
        initial_entries = stats_before["provenance_entries"]

        service.register_dataset(name="Provenance DS", source="test")

        stats_after = service.get_stats()
        assert stats_after["provenance_entries"] > initial_entries

    def test_sla_creation_creates_provenance_entry(self, service):
        """Verify that creating an SLA adds a provenance entry."""
        stats_before = service.get_stats()
        initial_entries = stats_before["provenance_entries"]

        service.create_sla(
            name="Provenance SLA",
            warning_hours=12.0,
            critical_hours=48.0,
        )

        stats_after = service.get_stats()
        assert stats_after["provenance_entries"] > initial_entries

    def test_freshness_check_creates_provenance_entry(
        self, service
    ):
        """Verify that running a freshness check adds a provenance entry."""
        ds = service.register_dataset(name="Check Prov DS", source="test")
        dataset_id = ds["dataset_id"]

        stats_before = service.get_stats()
        initial_entries = stats_before["provenance_entries"]

        fresh_ts = (_utcnow() - timedelta(minutes=5)).isoformat()
        service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=fresh_ts,
        )

        stats_after = service.get_stats()
        assert stats_after["provenance_entries"] > initial_entries

    def test_chain_grows_across_multiple_operations(self, service):
        """Verify the provenance chain grows monotonically as operations
        are performed: register -> SLA -> check -> batch check."""
        entry_counts = []

        # Operation 1: Register
        ds = service.register_dataset(name="Chain Growth DS", source="test")
        entry_counts.append(service.get_stats()["provenance_entries"])

        # Operation 2: Create SLA
        service.create_sla(
            dataset_id=ds["dataset_id"],
            name="Chain SLA",
            warning_hours=6.0,
            critical_hours=24.0,
        )
        entry_counts.append(service.get_stats()["provenance_entries"])

        # Operation 3: Run check
        fresh_ts = (_utcnow() - timedelta(hours=2)).isoformat()
        service.run_check(
            dataset_id=ds["dataset_id"],
            last_refreshed_at=fresh_ts,
        )
        entry_counts.append(service.get_stats()["provenance_entries"])

        # Operation 4: Batch check
        service.run_batch_check(dataset_ids=[ds["dataset_id"]])
        entry_counts.append(service.get_stats()["provenance_entries"])

        # Verify monotonic increase
        for i in range(1, len(entry_counts)):
            assert entry_counts[i] > entry_counts[i - 1], (
                f"Provenance entries did not increase between operation "
                f"{i} and {i + 1}: {entry_counts}"
            )

    def test_breach_generates_provenance_entry(self, service):
        """Verify that detecting a breach through a stale check adds
        provenance entries for both the check and the breach."""
        ds = service.register_dataset(name="Breach Prov DS", source="test")
        dataset_id = ds["dataset_id"]

        stats_before = service.get_stats()

        stale_ts = (_utcnow() - timedelta(hours=100)).isoformat()
        service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=stale_ts,
        )

        stats_after = service.get_stats()
        # Should have at least +1 entry for the check (breach is tracked
        # through the check provenance)
        assert stats_after["provenance_entries"] > stats_before["provenance_entries"]


# ===================================================================
# Provenance Chain Verification After Pipeline
# ===================================================================


class TestProvenanceAfterPipeline:
    """Test provenance chain integrity after full pipeline execution."""

    def test_pipeline_creates_provenance_trail(self, service):
        """Run the full pipeline and verify that provenance entries are
        recorded for the pipeline operation itself."""
        ds = service.register_dataset(name="Pipeline Prov DS", source="test")
        dataset_id = ds["dataset_id"]

        stats_before = service.get_stats()
        initial_entries = stats_before["provenance_entries"]

        result = service.run_pipeline(
            dataset_ids=[dataset_id],
            run_predictions=False,
            generate_alerts=True,
        )

        assert result["provenance_hash"]
        assert len(result["provenance_hash"]) == 64

        stats_after = service.get_stats()
        # Pipeline should add entries for: batch check, individual checks,
        # and the pipeline summary
        assert stats_after["provenance_entries"] > initial_entries

    def test_standalone_provenance_tracker_verify_chain(
        self, provenance_tracker
    ):
        """Use the standalone ProvenanceTracker to add entries and verify
        the chain is valid."""
        tracker = provenance_tracker

        # Record a sequence of operations
        tracker.record(
            "dataset", "ds-001", "register",
            tracker.compute_hash({"name": "Test DS"}),
        )
        tracker.record(
            "sla_definition", "sla-001", "create",
            tracker.compute_hash({"warning_hours": 12}),
        )
        tracker.record(
            "freshness_check", "chk-001", "check",
            tracker.compute_hash({"age_hours": 5.0}),
        )

        # Verify global chain
        valid, chain = tracker.verify_chain()
        assert valid is True
        assert len(chain) == 3

    def test_provenance_tracker_entity_scoped_chain(
        self, provenance_tracker
    ):
        """Verify that entity-scoped chain retrieval returns only entries
        for the specified entity."""
        tracker = provenance_tracker

        # Record operations for two different entities
        tracker.record(
            "dataset", "ds-001", "register",
            tracker.compute_hash({"name": "DS 1"}),
        )
        tracker.record(
            "dataset", "ds-002", "register",
            tracker.compute_hash({"name": "DS 2"}),
        )
        tracker.record(
            "dataset", "ds-001", "update",
            tracker.compute_hash({"name": "DS 1 updated"}),
        )

        # Verify entity-scoped chain for ds-001
        valid, chain_ds1 = tracker.verify_chain("dataset", "ds-001")
        assert valid is True
        assert len(chain_ds1) == 2

        # Verify entity-scoped chain for ds-002
        valid, chain_ds2 = tracker.verify_chain("dataset", "ds-002")
        assert valid is True
        assert len(chain_ds2) == 1

    def test_provenance_hash_determinism(self, provenance_tracker):
        """Verify that the same input data produces the same hash
        regardless of insertion order (sorted keys)."""
        tracker = provenance_tracker

        data_a = {"name": "DS", "source": "API", "priority": 1}
        data_b = {"priority": 1, "source": "API", "name": "DS"}

        hash_a = tracker.compute_hash(data_a)
        hash_b = tracker.compute_hash(data_b)

        assert hash_a == hash_b
        assert len(hash_a) == 64

    def test_provenance_chain_export_json(self, provenance_tracker):
        """Verify the provenance tracker can export its chain as JSON."""
        tracker = provenance_tracker

        tracker.record(
            "dataset", "ds-export", "register",
            tracker.compute_hash({"name": "Export DS"}),
        )

        import json
        exported = tracker.export_json()
        parsed = json.loads(exported)
        assert isinstance(parsed, list)
        assert len(parsed) >= 1
        assert "chain_hash" in parsed[0]

    def test_provenance_genesis_hash(self, provenance_tracker):
        """Verify the tracker initializes with the genesis hash."""
        tracker = provenance_tracker
        genesis = tracker.get_latest_hash()
        assert genesis == ProvenanceTracker.GENESIS_HASH
        assert len(genesis) == 64

    def test_provenance_chain_hash_changes_after_each_entry(
        self, provenance_tracker
    ):
        """Verify the latest chain hash changes after each operation,
        demonstrating tamper-evident linking."""
        tracker = provenance_tracker

        hashes = [tracker.get_latest_hash()]

        for i in range(5):
            tracker.record(
                "test_entity", f"ent-{i}", "action",
                tracker.compute_hash({"step": i}),
            )
            hashes.append(tracker.get_latest_hash())

        # All hashes must be unique
        assert len(set(hashes)) == len(hashes)
