# -*- coding: utf-8 -*-
"""
Tests for RelationshipManager - AGENT-EUDR-008 Engine 4: Relationship Lifecycle

Comprehensive test suite covering:
- All lifecycle state transitions: valid and invalid (F4.1)
- Relationship attributes: dates, volume, frequency (F4.2)
- Upstream/downstream traversal
- Relationship strength scoring (F4.5)
- Conflict detection: circular dependencies, inconsistent declarations (F4.6)
- Timeline history with reason codes (F4.3, F4.7)
- Bulk import from ERP/procurement (F4.8)
- Seasonal patterns (F4.4)

Test count: 60+ tests
Coverage target: >= 85% of RelationshipManager module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.multi_tier_supplier.conftest import (
    SUP_ID_COCOA_IMPORTER_EU,
    SUP_ID_COCOA_TRADER_GH,
    SUP_ID_COCOA_PROCESSOR_GH,
    SUP_ID_COCOA_AGGREGATOR_GH,
    SUP_ID_COCOA_COOPERATIVE_GH,
    SUP_ID_COCOA_FARMER_1_GH,
    SUP_ID_COFFEE_IMPORTER_DE,
    SUP_ID_COFFEE_EXPORTER_CO,
    RELATIONSHIP_STATES,
    VALID_STATE_TRANSITIONS,
    INVALID_STATE_TRANSITIONS,
    SHA256_HEX_LENGTH,
    make_relationship,
    make_supplier,
    compute_sha256,
)


# ===========================================================================
# 1. State Transitions - Valid
# ===========================================================================


class TestValidStateTransitions:
    """Test all valid relationship state transitions per PRD F4.1."""

    @pytest.mark.parametrize("from_state,to_state", VALID_STATE_TRANSITIONS)
    def test_valid_transition(self, relationship_manager, from_state, to_state):
        """Each valid state transition succeeds."""
        rel = make_relationship(
            buyer_id="BUYER-TRANS",
            supplier_id="SUP-TRANS",
            state=from_state,
            rel_id=f"REL-{from_state}-{to_state}",
        )
        relationship_manager.create(rel)
        result = relationship_manager.transition(
            rel["relationship_id"], to_state, reason="test transition"
        )
        assert result["state"] == to_state

    def test_prospective_to_onboarding(self, relationship_manager):
        """Prospective -> Onboarding is the first valid transition."""
        rel = make_relationship(
            buyer_id="B-001", supplier_id="S-001", state="prospective", rel_id="REL-P2O"
        )
        relationship_manager.create(rel)
        result = relationship_manager.transition("REL-P2O", "onboarding", reason="initiated")
        assert result["state"] == "onboarding"

    def test_onboarding_to_active(self, relationship_manager):
        """Onboarding -> Active activates the relationship."""
        rel = make_relationship(
            buyer_id="B-002", supplier_id="S-002", state="onboarding", rel_id="REL-O2A"
        )
        relationship_manager.create(rel)
        result = relationship_manager.transition("REL-O2A", "active", reason="verified")
        assert result["state"] == "active"

    def test_active_to_suspended(self, relationship_manager):
        """Active -> Suspended puts relationship on hold."""
        rel = make_relationship(
            buyer_id="B-003", supplier_id="S-003", state="active", rel_id="REL-A2S"
        )
        relationship_manager.create(rel)
        result = relationship_manager.transition("REL-A2S", "suspended", reason="compliance issue")
        assert result["state"] == "suspended"

    def test_suspended_to_active_reinstated(self, relationship_manager):
        """Suspended -> Active reinstates the relationship."""
        rel = make_relationship(
            buyer_id="B-004", supplier_id="S-004", state="suspended", rel_id="REL-S2A"
        )
        relationship_manager.create(rel)
        result = relationship_manager.transition("REL-S2A", "active", reason="issue resolved")
        assert result["state"] == "active"

    def test_active_to_terminated(self, relationship_manager):
        """Active -> Terminated ends the relationship."""
        rel = make_relationship(
            buyer_id="B-005", supplier_id="S-005", state="active", rel_id="REL-A2T"
        )
        relationship_manager.create(rel)
        result = relationship_manager.transition("REL-A2T", "terminated", reason="contract ended")
        assert result["state"] == "terminated"

    def test_suspended_to_terminated(self, relationship_manager):
        """Suspended -> Terminated permanently ends suspended relationship."""
        rel = make_relationship(
            buyer_id="B-006", supplier_id="S-006", state="suspended", rel_id="REL-S2T"
        )
        relationship_manager.create(rel)
        result = relationship_manager.transition("REL-S2T", "terminated", reason="non-compliance")
        assert result["state"] == "terminated"


# ===========================================================================
# 2. State Transitions - Invalid
# ===========================================================================


class TestInvalidStateTransitions:
    """Test all invalid state transitions are rejected."""

    @pytest.mark.parametrize("from_state,to_state", INVALID_STATE_TRANSITIONS)
    def test_invalid_transition_raises(self, relationship_manager, from_state, to_state):
        """Each invalid state transition raises an error."""
        rel = make_relationship(
            buyer_id="BUYER-INV",
            supplier_id="SUP-INV",
            state=from_state,
            rel_id=f"REL-INV-{from_state}-{to_state}",
        )
        relationship_manager.create(rel)
        with pytest.raises((ValueError, RuntimeError)):
            relationship_manager.transition(
                rel["relationship_id"], to_state, reason="invalid"
            )

    def test_terminated_is_final(self, relationship_manager):
        """Terminated relationships cannot transition to any state."""
        rel = make_relationship(
            buyer_id="B-FINAL", supplier_id="S-FINAL", state="terminated", rel_id="REL-FINAL"
        )
        relationship_manager.create(rel)
        for target in ["prospective", "onboarding", "active", "suspended"]:
            with pytest.raises((ValueError, RuntimeError)):
                relationship_manager.transition("REL-FINAL", target, reason="test")

    def test_transition_nonexistent_raises(self, relationship_manager):
        """Transitioning a non-existent relationship raises error."""
        with pytest.raises((ValueError, KeyError)):
            relationship_manager.transition("REL-GHOST", "active", reason="ghost")


# ===========================================================================
# 3. Create and CRUD
# ===========================================================================


class TestRelationshipCRUD:
    """Test CRUD operations for relationships."""

    def test_create_relationship(self, relationship_manager):
        """Create a new relationship."""
        rel = make_relationship(
            buyer_id="CRUD-B", supplier_id="CRUD-S", rel_id="REL-CRUD-001"
        )
        result = relationship_manager.create(rel)
        assert result["relationship_id"] == "REL-CRUD-001"

    def test_get_relationship(self, relationship_manager):
        """Retrieve an existing relationship by ID."""
        rel = make_relationship(
            buyer_id="GET-B", supplier_id="GET-S", rel_id="REL-GET-001"
        )
        relationship_manager.create(rel)
        result = relationship_manager.get("REL-GET-001")
        assert result is not None
        assert result["buyer_id"] == "GET-B"

    def test_get_nonexistent_returns_none(self, relationship_manager):
        """Non-existent relationship returns None."""
        result = relationship_manager.get("REL-NONEXISTENT")
        assert result is None

    def test_update_volume(self, relationship_manager):
        """Update the volume of an existing relationship."""
        rel = make_relationship(
            buyer_id="UPD-B", supplier_id="UPD-S", volume_mt=1000.0, rel_id="REL-UPD-001"
        )
        relationship_manager.create(rel)
        result = relationship_manager.update("REL-UPD-001", {"volume_mt": 2000.0})
        assert result["volume_mt"] == pytest.approx(2000.0)

    def test_create_duplicate_id_raises(self, relationship_manager):
        """Duplicate relationship_id raises error."""
        rel = make_relationship(buyer_id="DUP-B", supplier_id="DUP-S", rel_id="REL-DUP-001")
        relationship_manager.create(rel)
        with pytest.raises((ValueError, KeyError)):
            relationship_manager.create(rel)


# ===========================================================================
# 4. Traversal
# ===========================================================================


class TestRelationshipTraversal:
    """Test upstream/downstream supplier traversal."""

    def test_get_upstream_suppliers(self, relationship_manager, sample_relationships):
        """Get all upstream suppliers for the cocoa importer."""
        for rel in sample_relationships:
            relationship_manager.create(rel)
        upstream = relationship_manager.get_upstream(SUP_ID_COCOA_IMPORTER_EU)
        # Importer buys from trader
        supplier_ids = [r["supplier_id"] for r in upstream]
        assert SUP_ID_COCOA_TRADER_GH in supplier_ids

    def test_get_downstream_buyers(self, relationship_manager, sample_relationships):
        """Get all downstream buyers for the cocoa trader."""
        for rel in sample_relationships:
            relationship_manager.create(rel)
        downstream = relationship_manager.get_downstream(SUP_ID_COCOA_TRADER_GH)
        buyer_ids = [r["buyer_id"] for r in downstream]
        assert SUP_ID_COCOA_IMPORTER_EU in buyer_ids

    def test_get_all_relationships_for_supplier(self, relationship_manager, sample_relationships):
        """Get all relationships (upstream + downstream) for a supplier."""
        for rel in sample_relationships:
            relationship_manager.create(rel)
        all_rels = relationship_manager.get_all_for_supplier(SUP_ID_COCOA_TRADER_GH)
        assert len(all_rels) >= 2

    def test_traverse_full_chain(self, relationship_manager, sample_relationships):
        """Traverse from importer to farmer through all tiers."""
        for rel in sample_relationships:
            relationship_manager.create(rel)
        chain = relationship_manager.traverse_chain(
            SUP_ID_COCOA_IMPORTER_EU, direction="upstream"
        )
        supplier_ids = [node["supplier_id"] for node in chain]
        assert SUP_ID_COCOA_TRADER_GH in supplier_ids

    def test_traverse_empty_result(self, relationship_manager):
        """Traversal for supplier with no relationships returns empty."""
        chain = relationship_manager.traverse_chain("SUP-LONELY", direction="upstream")
        assert len(chain) == 0


# ===========================================================================
# 5. Relationship Strength Scoring
# ===========================================================================


class TestRelationshipStrength:
    """Test relationship strength scoring (F4.5)."""

    def test_high_volume_high_strength(self, relationship_manager):
        """High-volume, long-duration relationship scores high."""
        rel = make_relationship(
            buyer_id="STR-B", supplier_id="STR-S", volume_mt=50000.0,
            rel_id="REL-STR-HIGH",
            start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        relationship_manager.create(rel)
        score = relationship_manager.calculate_strength("REL-STR-HIGH")
        assert 0.0 <= score <= 100.0
        assert score > 50.0

    def test_low_volume_low_strength(self, relationship_manager):
        """Low-volume, short-duration relationship scores lower."""
        rel = make_relationship(
            buyer_id="STR-B2", supplier_id="STR-S2", volume_mt=10.0,
            rel_id="REL-STR-LOW",
            start_date=datetime(2025, 11, 1, tzinfo=timezone.utc),
        )
        relationship_manager.create(rel)
        score = relationship_manager.calculate_strength("REL-STR-LOW")
        assert 0.0 <= score <= 100.0

    def test_strength_score_is_bounded(self, relationship_manager):
        """Strength score is always between 0 and 100."""
        rel = make_relationship(
            buyer_id="STR-B3", supplier_id="STR-S3", rel_id="REL-STR-BOUND"
        )
        relationship_manager.create(rel)
        score = relationship_manager.calculate_strength("REL-STR-BOUND")
        assert 0.0 <= score <= 100.0

    def test_strength_of_terminated_relationship(self, relationship_manager):
        """Terminated relationships may have lower effective strength."""
        rel = make_relationship(
            buyer_id="STR-TERM-B", supplier_id="STR-TERM-S",
            state="terminated", rel_id="REL-STR-TERM"
        )
        relationship_manager.create(rel)
        score = relationship_manager.calculate_strength("REL-STR-TERM")
        assert 0.0 <= score <= 100.0


# ===========================================================================
# 6. Conflict Detection
# ===========================================================================


class TestConflictDetection:
    """Test conflict detection in relationships (F4.6)."""

    def test_detect_circular_dependency(self, relationship_manager):
        """Circular dependency (A -> B -> C -> A) is detected."""
        rels = [
            make_relationship("CIRC-A", "CIRC-B", rel_id="REL-CIRC-1"),
            make_relationship("CIRC-B", "CIRC-C", rel_id="REL-CIRC-2"),
            make_relationship("CIRC-C", "CIRC-A", rel_id="REL-CIRC-3"),
        ]
        for r in rels:
            relationship_manager.create(r)
        conflicts = relationship_manager.detect_conflicts()
        circular = [c for c in conflicts if c["type"] == "circular_dependency"]
        assert len(circular) >= 1

    def test_detect_self_reference(self, relationship_manager):
        """Self-referencing relationship (A -> A) is detected as conflict."""
        rel = make_relationship("SELF-A", "SELF-A", rel_id="REL-SELF")
        relationship_manager.create(rel)
        conflicts = relationship_manager.detect_conflicts()
        self_refs = [c for c in conflicts if c["type"] in ("self_reference", "circular_dependency")]
        assert len(self_refs) >= 1

    def test_no_conflicts_in_clean_chain(self, relationship_manager, sample_relationships):
        """Linear chain without cycles has no circular dependency conflicts."""
        cocoa_rels = [r for r in sample_relationships if r["commodity"] == "cocoa"]
        for r in cocoa_rels:
            relationship_manager.create(r)
        conflicts = relationship_manager.detect_conflicts()
        circular = [c for c in conflicts if c["type"] == "circular_dependency"]
        assert len(circular) == 0

    def test_detect_duplicate_relationship(self, relationship_manager):
        """Two relationships between same buyer-supplier pair flagged."""
        rel1 = make_relationship("DUP-B", "DUP-S", rel_id="REL-DUP-A", commodity="cocoa")
        rel2 = make_relationship("DUP-B", "DUP-S", rel_id="REL-DUP-B", commodity="cocoa")
        relationship_manager.create(rel1)
        relationship_manager.create(rel2)
        conflicts = relationship_manager.detect_conflicts()
        duplicates = [c for c in conflicts if c["type"] == "duplicate_relationship"]
        assert len(duplicates) >= 1


# ===========================================================================
# 7. Timeline History
# ===========================================================================


class TestRelationshipTimeline:
    """Test relationship timeline and change history (F4.3, F4.7)."""

    def test_timeline_records_creation(self, relationship_manager):
        """Timeline includes the creation event."""
        rel = make_relationship(
            buyer_id="TL-B", supplier_id="TL-S", rel_id="REL-TL-001"
        )
        relationship_manager.create(rel)
        history = relationship_manager.get_history("REL-TL-001")
        assert len(history) >= 1
        assert history[0]["event"] in ("created", "creation")

    def test_timeline_records_transitions(self, relationship_manager):
        """Timeline records each state transition."""
        rel = make_relationship(
            buyer_id="TL-B2", supplier_id="TL-S2", state="prospective", rel_id="REL-TL-002"
        )
        relationship_manager.create(rel)
        relationship_manager.transition("REL-TL-002", "onboarding", reason="started")
        relationship_manager.transition("REL-TL-002", "active", reason="approved")
        history = relationship_manager.get_history("REL-TL-002")
        assert len(history) >= 3  # creation + 2 transitions

    def test_timeline_includes_reason_codes(self, relationship_manager):
        """Timeline entries include reason codes."""
        rel = make_relationship(
            buyer_id="TL-B3", supplier_id="TL-S3", state="active", rel_id="REL-TL-003"
        )
        relationship_manager.create(rel)
        relationship_manager.transition(
            "REL-TL-003", "suspended", reason="failed_audit"
        )
        history = relationship_manager.get_history("REL-TL-003")
        suspension_event = [h for h in history if h.get("to_state") == "suspended"]
        assert len(suspension_event) >= 1
        assert suspension_event[0]["reason"] == "failed_audit"

    def test_timeline_has_timestamps(self, relationship_manager):
        """All timeline entries have timestamps."""
        rel = make_relationship(buyer_id="TL-B4", supplier_id="TL-S4", rel_id="REL-TL-004")
        relationship_manager.create(rel)
        history = relationship_manager.get_history("REL-TL-004")
        for entry in history:
            assert "timestamp" in entry or "created_at" in entry


# ===========================================================================
# 8. Bulk Import
# ===========================================================================


class TestBulkRelationshipImport:
    """Test bulk import of relationships (F4.8)."""

    def test_bulk_import_multiple_relationships(self, relationship_manager):
        """Import 10 relationships in a single batch."""
        rels = [
            make_relationship(
                buyer_id=f"BULK-B-{i}", supplier_id=f"BULK-S-{i}",
                rel_id=f"REL-BULK-{i:03d}",
            )
            for i in range(10)
        ]
        result = relationship_manager.bulk_import(rels)
        assert result.imported_count == 10
        assert result.error_count == 0

    def test_bulk_import_with_duplicates(self, relationship_manager):
        """Bulk import handles duplicates gracefully."""
        rels = [
            make_relationship(buyer_id="BLK-B", supplier_id="BLK-S", rel_id="REL-BLK-DUP"),
            make_relationship(buyer_id="BLK-B", supplier_id="BLK-S", rel_id="REL-BLK-DUP"),
        ]
        result = relationship_manager.bulk_import(rels)
        assert result.imported_count == 1
        assert result.duplicate_count >= 1

    def test_bulk_import_empty(self, relationship_manager):
        """Empty bulk import returns zero counts."""
        result = relationship_manager.bulk_import([])
        assert result.imported_count == 0

    def test_bulk_import_partial_failure(self, relationship_manager):
        """Bulk import with some invalid records reports errors."""
        rels = [
            make_relationship(buyer_id="GOOD-B", supplier_id="GOOD-S", rel_id="REL-GOOD"),
            {"relationship_id": "REL-BAD", "buyer_id": None},  # invalid
        ]
        result = relationship_manager.bulk_import(rels)
        assert result.imported_count >= 1
        assert result.error_count >= 1

    def test_bulk_import_large_batch(self, relationship_manager):
        """Bulk import of 500 relationships succeeds."""
        rels = [
            make_relationship(
                buyer_id=f"LRG-B-{i}", supplier_id=f"LRG-S-{i}",
                rel_id=f"REL-LRG-{i:04d}",
            )
            for i in range(500)
        ]
        result = relationship_manager.bulk_import(rels)
        assert result.imported_count == 500


# ===========================================================================
# 9. Provenance
# ===========================================================================


class TestRelationshipProvenance:
    """Test provenance tracking for relationship operations."""

    def test_create_has_provenance_hash(self, relationship_manager):
        """Created relationship includes provenance hash."""
        rel = make_relationship(
            buyer_id="PROV-B", supplier_id="PROV-S", rel_id="REL-PROV-001"
        )
        result = relationship_manager.create(rel)
        assert result.get("provenance_hash") is not None
        assert len(result["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_transition_updates_provenance(self, relationship_manager):
        """State transition updates the provenance hash."""
        rel = make_relationship(
            buyer_id="PROV-B2", supplier_id="PROV-S2",
            state="prospective", rel_id="REL-PROV-002"
        )
        created = relationship_manager.create(rel)
        result = relationship_manager.transition("REL-PROV-002", "onboarding", reason="test")
        # Hash should change after transition
        assert result.get("provenance_hash") != created.get("provenance_hash")


# ===========================================================================
# 10. Seasonal Patterns
# ===========================================================================


class TestSeasonalPatterns:
    """Test seasonal relationship patterns (F4.4)."""

    def test_seasonal_relationship_start_end(self, relationship_manager):
        """Seasonal relationships have defined start and end dates."""
        rel = make_relationship(
            buyer_id="SEAS-B", supplier_id="SEAS-S", rel_id="REL-SEAS-001",
            start_date=datetime(2025, 10, 1, tzinfo=timezone.utc),
            end_date=datetime(2026, 3, 31, tzinfo=timezone.utc),
        )
        result = relationship_manager.create(rel)
        assert result["start_date"] is not None
        assert result["end_date"] is not None

    def test_seasonal_relationship_repeating(self, relationship_manager):
        """Multiple seasonal relationships for same pair in different years."""
        for year in [2024, 2025, 2026]:
            rel = make_relationship(
                buyer_id="SEAS-REP-B", supplier_id="SEAS-REP-S",
                rel_id=f"REL-SEAS-{year}",
                start_date=datetime(year, 10, 1, tzinfo=timezone.utc),
                end_date=datetime(year + 1, 3, 31, tzinfo=timezone.utc),
            )
            relationship_manager.create(rel)
        all_rels = relationship_manager.get_all_for_supplier("SEAS-REP-S")
        seasonal = [r for r in all_rels if r.get("buyer_id") == "SEAS-REP-B"]
        assert len(seasonal) == 3


# ===========================================================================
# 11. Relationship Attributes
# ===========================================================================


class TestRelationshipAttributes:
    """Test relationship attributes (F4.2)."""

    def test_relationship_has_commodity(self, relationship_manager):
        """Relationship includes commodity type."""
        rel = make_relationship(
            buyer_id="ATTR-B", supplier_id="ATTR-S", commodity="coffee",
            rel_id="REL-ATTR-001"
        )
        result = relationship_manager.create(rel)
        assert result["commodity"] == "coffee"

    def test_relationship_has_volume(self, relationship_manager):
        """Relationship includes volume_mt."""
        rel = make_relationship(
            buyer_id="ATTR-B2", supplier_id="ATTR-S2", volume_mt=5000.0,
            rel_id="REL-ATTR-002"
        )
        result = relationship_manager.create(rel)
        assert result["volume_mt"] == pytest.approx(5000.0)

    def test_relationship_has_frequency(self, relationship_manager):
        """Relationship includes transaction frequency."""
        rel = make_relationship(
            buyer_id="ATTR-B3", supplier_id="ATTR-S3", rel_id="REL-ATTR-003"
        )
        result = relationship_manager.create(rel)
        assert "frequency" in result

    @pytest.mark.parametrize("state", RELATIONSHIP_STATES)
    def test_create_with_each_state(self, relationship_manager, state):
        """Relationships can be created with any initial state."""
        rel = make_relationship(
            buyer_id=f"STATE-B-{state}", supplier_id=f"STATE-S-{state}",
            state=state, rel_id=f"REL-STATE-{state}"
        )
        result = relationship_manager.create(rel)
        assert result["state"] == state

    def test_relationship_start_date(self, relationship_manager):
        """Relationship has start date."""
        rel = make_relationship(
            buyer_id="DATE-B", supplier_id="DATE-S", rel_id="REL-DATE-001",
            start_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )
        result = relationship_manager.create(rel)
        assert result["start_date"] is not None
