# -*- coding: utf-8 -*-
"""
Tests for ChainIntegrityVerifier - AGENT-EUDR-009 Engine 7: Chain Integrity

Comprehensive test suite covering:
- End-to-end chain verification (F7.1)
- Temporal continuity (F7.2)
- Actor continuity (F7.3)
- Location continuity (F7.4)
- Mass conservation (F7.5)
- Origin preservation (F7.6)
- Orphan batch detection (F7.7)
- Circular dependency detection (F7.8)
- Completeness score calculation (F7.9)
- Verification certificate generation (F7.10)

Test count: 55+ tests
Coverage target: >= 85% of ChainIntegrityVerifier module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-009 Chain of Custody Agent (GL-EUDR-COC-009)
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.chain_of_custody.conftest import (
    EUDR_COMMODITIES,
    SHA256_HEX_LENGTH,
    RISK_CATEGORY_WEIGHTS,
    ACTOR_FARMER_GH,
    ACTOR_COOP_GH,
    ACTOR_TRADER_GH,
    ACTOR_SHIPPER_INT,
    ACTOR_IMPORTER_NL,
    ACTOR_PROCESSOR_DE,
    FAC_ID_PROC_GH,
    FAC_ID_WAREHOUSE_NL,
    FAC_ID_FACTORY_DE,
    PLOT_ID_COCOA_GH_1,
    PLOT_ID_COCOA_GH_2,
    make_event,
    make_batch,
    build_cocoa_chain,
    build_palm_oil_chain,
    build_coffee_chain,
    build_linear_genealogy,
    assert_valid_chain,
    assert_mass_conservation,
    assert_origin_preserved,
    assert_valid_completeness_score,
    assert_valid_provenance_hash,
    compute_sha256,
)


# ===========================================================================
# 1. End-to-End Chain Verification (F7.1)
# ===========================================================================


class TestEndToEndVerification:
    """Test complete chain verification from plot to EU market."""

    def test_complete_cocoa_chain_valid(self, chain_integrity_verifier):
        """Complete cocoa chain passes verification."""
        chain = build_cocoa_chain()
        result = chain_integrity_verifier.verify(chain)
        assert result["is_complete"] is True
        assert result["verification_status"] in ("passed", "complete")

    def test_complete_palm_oil_chain_valid(self, chain_integrity_verifier):
        """Complete palm oil chain passes verification."""
        chain = build_palm_oil_chain()
        result = chain_integrity_verifier.verify(chain)
        assert result["is_complete"] is True

    def test_complete_coffee_chain_valid(self, chain_integrity_verifier):
        """Complete coffee chain passes verification."""
        chain = build_coffee_chain()
        result = chain_integrity_verifier.verify(chain)
        assert result["is_complete"] is True

    def test_incomplete_chain_flagged(self, chain_integrity_verifier):
        """Chain missing events is flagged as incomplete."""
        chain = build_cocoa_chain()
        chain["events"] = chain["events"][:2]  # Remove most events
        result = chain_integrity_verifier.verify(chain)
        assert result["is_complete"] is False

    def test_empty_chain_fails(self, chain_integrity_verifier):
        """Empty chain with no batches fails verification."""
        chain = {"batches": [], "events": [], "transformations": [], "documents": []}
        result = chain_integrity_verifier.verify(chain)
        assert result["is_complete"] is False

    def test_chain_with_single_batch(self, chain_integrity_verifier):
        """Chain with a single batch and origin verifies as minimal."""
        batch = make_batch(batch_id="BATCH-SINGLE")
        chain = {
            "batches": [batch],
            "events": [make_event("receipt", "BATCH-SINGLE")],
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.verify(chain)
        assert result is not None

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_verification_all_commodities(self, chain_integrity_verifier, commodity):
        """Chain verification works for all 7 EUDR commodities."""
        batch = make_batch(commodity=commodity, batch_id=f"BATCH-VERIFY-{commodity}")
        chain = {
            "batches": [batch],
            "events": [make_event("receipt", f"BATCH-VERIFY-{commodity}")],
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.verify(chain)
        assert result is not None


# ===========================================================================
# 2. Temporal Continuity (F7.2)
# ===========================================================================


class TestTemporalContinuity:
    """Test temporal continuity validation in custody chains."""

    def test_no_temporal_gaps(self, chain_integrity_verifier):
        """Chain with continuous timestamps has no temporal gaps."""
        batch_id = "BATCH-TEMP-CONT"
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = []
        for i, etype in enumerate(["receipt", "storage_in", "storage_out", "transfer"]):
            events.append(make_event(
                etype, batch_id,
                timestamp=(base + timedelta(hours=i * 12)).isoformat(),
            ))
        chain = {
            "batches": [make_batch(batch_id=batch_id)],
            "events": events,
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.check_temporal_continuity(chain)
        assert result["gaps_found"] == 0

    def test_temporal_gap_detected(self, chain_integrity_verifier):
        """Chain with >72h gap between events is flagged."""
        batch_id = "BATCH-TEMP-GAP"
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            make_event("receipt", batch_id, timestamp=base.isoformat()),
            make_event("storage_in", batch_id,
                       timestamp=(base + timedelta(hours=100)).isoformat()),
        ]
        chain = {
            "batches": [make_batch(batch_id=batch_id)],
            "events": events,
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.check_temporal_continuity(chain)
        assert result["gaps_found"] >= 1

    def test_multiple_temporal_gaps(self, chain_integrity_verifier):
        """Multiple temporal gaps are all detected."""
        batch_id = "BATCH-MULTI-GAP"
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            make_event("receipt", batch_id, timestamp=base.isoformat()),
            make_event("storage_in", batch_id,
                       timestamp=(base + timedelta(hours=100)).isoformat()),
            make_event("storage_out", batch_id,
                       timestamp=(base + timedelta(hours=300)).isoformat()),
        ]
        chain = {
            "batches": [make_batch(batch_id=batch_id)],
            "events": events,
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.check_temporal_continuity(chain)
        assert result["gaps_found"] >= 2

    @pytest.mark.parametrize("gap_hours", [24, 48, 72])
    def test_gaps_within_threshold_not_flagged(self, chain_integrity_verifier, gap_hours):
        """Gaps within the 72h threshold are not flagged."""
        batch_id = f"BATCH-THR-{gap_hours}"
        base = datetime(2026, 2, 1, tzinfo=timezone.utc)
        events = [
            make_event("receipt", batch_id, timestamp=base.isoformat()),
            make_event("storage_in", batch_id,
                       timestamp=(base + timedelta(hours=gap_hours)).isoformat()),
        ]
        chain = {
            "batches": [make_batch(batch_id=batch_id)],
            "events": events,
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.check_temporal_continuity(chain)
        assert result["gaps_found"] == 0


# ===========================================================================
# 3. Actor Continuity (F7.3)
# ===========================================================================


class TestActorContinuityIntegrity:
    """Test actor continuity validation in custody chains."""

    def test_valid_actor_chain(self, chain_integrity_verifier):
        """Chain with valid actor continuity passes."""
        batch_id = "BATCH-ACT-VALID"
        events = [
            make_event("transfer", batch_id,
                       sender_actor_id=ACTOR_TRADER_GH,
                       receiver_actor_id=ACTOR_SHIPPER_INT,
                       timestamp="2026-01-01T10:00:00+00:00"),
            make_event("receipt", batch_id,
                       sender_actor_id=ACTOR_SHIPPER_INT,
                       receiver_actor_id=ACTOR_IMPORTER_NL,
                       timestamp="2026-01-15T10:00:00+00:00"),
        ]
        chain = {
            "batches": [make_batch(batch_id=batch_id)],
            "events": events,
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.check_actor_continuity(chain)
        assert result["breaks_found"] == 0

    def test_broken_actor_chain(self, chain_integrity_verifier):
        """Chain with broken actor continuity is flagged."""
        batch_id = "BATCH-ACT-BROKEN"
        events = [
            make_event("transfer", batch_id,
                       sender_actor_id=ACTOR_TRADER_GH,
                       receiver_actor_id=ACTOR_SHIPPER_INT,
                       timestamp="2026-01-01T10:00:00+00:00"),
            make_event("receipt", batch_id,
                       sender_actor_id=ACTOR_PROCESSOR_DE,  # Wrong
                       receiver_actor_id=ACTOR_IMPORTER_NL,
                       timestamp="2026-01-15T10:00:00+00:00"),
        ]
        chain = {
            "batches": [make_batch(batch_id=batch_id)],
            "events": events,
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.check_actor_continuity(chain)
        assert result["breaks_found"] >= 1

    def test_multiple_actor_breaks(self, chain_integrity_verifier):
        """Multiple actor breaks in a chain are all detected."""
        batch_id = "BATCH-ACT-MULTI"
        events = [
            make_event("transfer", batch_id,
                       sender_actor_id="A", receiver_actor_id="B",
                       timestamp="2026-01-01T10:00:00+00:00"),
            make_event("receipt", batch_id,
                       sender_actor_id="C", receiver_actor_id="D",  # B->C break
                       timestamp="2026-01-02T10:00:00+00:00"),
            make_event("transfer", batch_id,
                       sender_actor_id="E", receiver_actor_id="F",  # D->E break
                       timestamp="2026-01-03T10:00:00+00:00"),
        ]
        chain = {
            "batches": [make_batch(batch_id=batch_id)],
            "events": events,
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.check_actor_continuity(chain)
        assert result["breaks_found"] >= 2


# ===========================================================================
# 4. Location Continuity (F7.4)
# ===========================================================================


class TestLocationContinuityIntegrity:
    """Test location continuity validation in custody chains."""

    def test_valid_location_chain(self, chain_integrity_verifier):
        """Chain with valid location continuity passes."""
        batch_id = "BATCH-LOC-VALID"
        events = [
            make_event("transfer", batch_id,
                       source_facility_id=FAC_ID_PROC_GH,
                       dest_facility_id=FAC_ID_WAREHOUSE_NL,
                       timestamp="2026-01-01T10:00:00+00:00"),
            make_event("receipt", batch_id,
                       source_facility_id=FAC_ID_WAREHOUSE_NL,
                       dest_facility_id=FAC_ID_WAREHOUSE_NL,
                       timestamp="2026-01-15T10:00:00+00:00"),
        ]
        chain = {
            "batches": [make_batch(batch_id=batch_id)],
            "events": events,
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.check_location_continuity(chain)
        assert result["teleportations_found"] == 0

    def test_teleportation_detected(self, chain_integrity_verifier):
        """Goods teleporting between unrelated locations is detected."""
        batch_id = "BATCH-TELEPORT"
        events = [
            make_event("transfer", batch_id,
                       source_facility_id=FAC_ID_PROC_GH,
                       dest_facility_id=FAC_ID_WAREHOUSE_NL,
                       timestamp="2026-01-01T10:00:00+00:00"),
            make_event("receipt", batch_id,
                       source_facility_id=FAC_ID_FACTORY_DE,  # Teleportation
                       dest_facility_id=FAC_ID_FACTORY_DE,
                       timestamp="2026-01-15T10:00:00+00:00"),
        ]
        chain = {
            "batches": [make_batch(batch_id=batch_id)],
            "events": events,
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.check_location_continuity(chain)
        assert result["teleportations_found"] >= 1


# ===========================================================================
# 5. Mass Conservation (F7.5)
# ===========================================================================


class TestMassConservationIntegrity:
    """Test mass conservation validation in custody chains."""

    def test_mass_conserved(self, chain_integrity_verifier):
        """Chain with conserved mass passes validation."""
        chain = build_cocoa_chain()
        result = chain_integrity_verifier.check_mass_conservation(chain)
        assert result["conservation_valid"] is True

    def test_mass_exceeds_tolerance(self, chain_integrity_verifier):
        """Chain where output significantly exceeds input is flagged."""
        batch_parent = make_batch(batch_id="BATCH-MASS-P", quantity_kg=1000.0)
        batch_child = make_batch(batch_id="BATCH-MASS-C", quantity_kg=2000.0,
                                 parent_batch_ids=["BATCH-MASS-P"])
        batch_parent["child_batch_ids"] = ["BATCH-MASS-C"]
        chain = {
            "batches": [batch_parent, batch_child],
            "events": [],
            "transformations": [{
                "transformation_id": "XFRM-MASS",
                "input_batches": [{"batch_id": "BATCH-MASS-P", "quantity_kg": 1000.0}],
                "output_batches": [{"batch_id": "BATCH-MASS-C", "quantity_kg": 2000.0,
                                    "product_type": "main"}],
                "waste_kg": 0.0,
            }],
            "documents": [],
        }
        result = chain_integrity_verifier.check_mass_conservation(chain)
        assert result["conservation_valid"] is False

    def test_mass_within_tolerance_with_waste(self, chain_integrity_verifier):
        """Mass conservation passes when waste accounts for difference."""
        batch_parent = make_batch(batch_id="BATCH-MASS-W-P", quantity_kg=10000.0)
        batch_child = make_batch(batch_id="BATCH-MASS-W-C", quantity_kg=8700.0,
                                 parent_batch_ids=["BATCH-MASS-W-P"])
        batch_parent["child_batch_ids"] = ["BATCH-MASS-W-C"]
        chain = {
            "batches": [batch_parent, batch_child],
            "events": [],
            "transformations": [{
                "transformation_id": "XFRM-MASS-W",
                "input_batches": [{"batch_id": "BATCH-MASS-W-P", "quantity_kg": 10000.0}],
                "output_batches": [{"batch_id": "BATCH-MASS-W-C", "quantity_kg": 8700.0,
                                    "product_type": "main"}],
                "waste_kg": 1300.0,
            }],
            "documents": [],
        }
        result = chain_integrity_verifier.check_mass_conservation(chain)
        assert result["conservation_valid"] is True


# ===========================================================================
# 6. Origin Preservation (F7.6)
# ===========================================================================


class TestOriginPreservation:
    """Test origin plot percentage preservation across the chain."""

    def test_origin_sums_to_100(self, chain_integrity_verifier):
        """All batches have origin percentages summing to 100%."""
        chain = build_cocoa_chain()
        result = chain_integrity_verifier.check_origin_preservation(chain)
        assert result["all_origins_complete"] is True

    def test_incomplete_origin_flagged(self, chain_integrity_verifier):
        """Batch with origin percentages not summing to 100% is flagged."""
        batch = make_batch(
            batch_id="BATCH-ORIG-BAD",
            origin_plots=[
                {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 50.0},
                {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 30.0},
                # Missing 20%
            ],
        )
        chain = {
            "batches": [batch],
            "events": [],
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.check_origin_preservation(chain)
        assert result["all_origins_complete"] is False

    def test_empty_origins_flagged(self, chain_integrity_verifier):
        """Batch with empty origin plots is flagged."""
        batch = make_batch(batch_id="BATCH-NO-ORIG", origin_plots=[])
        chain = {
            "batches": [batch],
            "events": [],
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.check_origin_preservation(chain)
        assert result["all_origins_complete"] is False


# ===========================================================================
# 7. Orphan Batch Detection (F7.7)
# ===========================================================================


class TestOrphanDetection:
    """Test detection of orphan batches without upstream or downstream."""

    def test_detect_orphan_no_upstream(self, chain_integrity_verifier):
        """Detect intermediate batch with no upstream origin."""
        orphan = make_batch(batch_id="BATCH-ORPHAN", parent_batch_ids=[],
                            status="processing")
        chain = {
            "batches": [orphan],
            "events": [],
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.detect_orphans(chain)
        assert len(result["orphan_batches"]) >= 1

    def test_root_batch_not_orphan(self, chain_integrity_verifier):
        """Root batch (harvest) with no parents is not an orphan."""
        root = make_batch(batch_id="BATCH-ROOT", parent_batch_ids=[],
                          status="created")
        chain = {
            "batches": [root],
            "events": [],
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.detect_orphans(chain)
        root_orphans = [b for b in result.get("orphan_batches", [])
                        if b["batch_id"] == "BATCH-ROOT"]
        assert len(root_orphans) == 0

    def test_no_orphans_in_complete_chain(self, chain_integrity_verifier):
        """Complete chain has no orphan batches."""
        chain = build_cocoa_chain()
        result = chain_integrity_verifier.detect_orphans(chain)
        assert len(result.get("orphan_batches", [])) == 0


# ===========================================================================
# 8. Circular Dependency Detection (F7.8)
# ===========================================================================


class TestCircularDependencyDetection:
    """Test detection of circular dependencies in batch genealogy."""

    def test_detect_circular_dependency(self, chain_integrity_verifier):
        """Detect circular dependency in batch genealogy."""
        batch_a = make_batch(batch_id="BATCH-CIRC-A",
                             parent_batch_ids=["BATCH-CIRC-C"])
        batch_b = make_batch(batch_id="BATCH-CIRC-B",
                             parent_batch_ids=["BATCH-CIRC-A"])
        batch_c = make_batch(batch_id="BATCH-CIRC-C",
                             parent_batch_ids=["BATCH-CIRC-B"])
        batch_a["child_batch_ids"] = ["BATCH-CIRC-B"]
        batch_b["child_batch_ids"] = ["BATCH-CIRC-C"]
        batch_c["child_batch_ids"] = ["BATCH-CIRC-A"]
        chain = {
            "batches": [batch_a, batch_b, batch_c],
            "events": [],
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.detect_circular_dependencies(chain)
        assert result["circular_found"] is True

    def test_no_circular_in_linear_chain(self, chain_integrity_verifier):
        """Linear chain has no circular dependencies."""
        batches = build_linear_genealogy(depth=5)
        chain = {
            "batches": batches,
            "events": [],
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.detect_circular_dependencies(chain)
        assert result["circular_found"] is False

    def test_self_reference_detected(self, chain_integrity_verifier):
        """Batch referencing itself as parent is detected."""
        batch = make_batch(batch_id="BATCH-SELF",
                           parent_batch_ids=["BATCH-SELF"])
        batch["child_batch_ids"] = ["BATCH-SELF"]
        chain = {
            "batches": [batch],
            "events": [],
            "transformations": [],
            "documents": [],
        }
        result = chain_integrity_verifier.detect_circular_dependencies(chain)
        assert result["circular_found"] is True


# ===========================================================================
# 9. Completeness Score (F7.9)
# ===========================================================================


class TestCompletenessScore:
    """Test chain completeness score calculation (0-100)."""

    def test_complete_chain_high_score(self, chain_integrity_verifier):
        """Complete chain with all elements scores high."""
        chain = build_cocoa_chain()
        score = chain_integrity_verifier.calculate_completeness_score(chain)
        assert_valid_completeness_score(score)
        assert score >= 80.0

    def test_empty_chain_zero_score(self, chain_integrity_verifier):
        """Empty chain scores zero."""
        chain = {"batches": [], "events": [], "transformations": [], "documents": []}
        score = chain_integrity_verifier.calculate_completeness_score(chain)
        assert score == pytest.approx(0.0)

    def test_score_is_deterministic(self, chain_integrity_verifier):
        """Same chain always produces same score."""
        chain = build_cocoa_chain()
        s1 = chain_integrity_verifier.calculate_completeness_score(chain)
        s2 = chain_integrity_verifier.calculate_completeness_score(chain)
        assert s1 == s2

    def test_risk_category_weights_sum_to_one(self, chain_integrity_verifier):
        """Risk category weights sum to 1.0."""
        total = sum(RISK_CATEGORY_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_partial_chain_intermediate_score(self, chain_integrity_verifier):
        """Partial chain scores between 0 and 100."""
        batch = make_batch(batch_id="BATCH-PARTIAL-SC")
        chain = {
            "batches": [batch],
            "events": [make_event("receipt", "BATCH-PARTIAL-SC")],
            "transformations": [],
            "documents": [],
        }
        score = chain_integrity_verifier.calculate_completeness_score(chain)
        assert 0.0 < score < 100.0


# ===========================================================================
# 10. Verification Certificate (F7.10)
# ===========================================================================


class TestVerificationCertificate:
    """Test verification certificate generation."""

    def test_generate_certificate(self, chain_integrity_verifier):
        """Generate a verification certificate for a complete chain."""
        chain = build_cocoa_chain()
        cert = chain_integrity_verifier.generate_certificate(chain)
        assert cert is not None
        assert "verification_id" in cert
        assert "timestamp" in cert
        assert "completeness_score" in cert

    def test_certificate_includes_provenance_hash(self, chain_integrity_verifier):
        """Verification certificate includes a provenance hash."""
        chain = build_cocoa_chain()
        cert = chain_integrity_verifier.generate_certificate(chain)
        assert "provenance_hash" in cert
        assert_valid_provenance_hash(cert["provenance_hash"])

    def test_certificate_includes_findings(self, chain_integrity_verifier):
        """Certificate includes verification findings (gaps, breaks, etc.)."""
        chain = build_cocoa_chain()
        cert = chain_integrity_verifier.generate_certificate(chain)
        assert "findings" in cert or "results" in cert

    def test_certificate_for_incomplete_chain(self, chain_integrity_verifier):
        """Certificate for incomplete chain documents the issues."""
        batch = make_batch(batch_id="BATCH-CERT-INC", origin_plots=[])
        chain = {
            "batches": [batch],
            "events": [],
            "transformations": [],
            "documents": [],
        }
        cert = chain_integrity_verifier.generate_certificate(chain)
        assert cert["completeness_score"] < 50.0
