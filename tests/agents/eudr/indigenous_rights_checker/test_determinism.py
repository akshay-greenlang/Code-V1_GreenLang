# -*- coding: utf-8 -*-
"""
Determinism Tests for AGENT-EUDR-021 Indigenous Rights Checker

Validates bit-perfect reproducibility across all scoring engines,
hash computations, classification logic, and provenance tracking.
GreenLang's Zero-Hallucination principle mandates that the same inputs
always produce the exact same outputs (including Decimal precision
and SHA-256 provenance hashes), regardless of execution count, order,
or system state.

Test count: 35 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (Determinism & Reproducibility Validation)
"""

import hashlib
import json
import threading
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, ROUND_UP
from typing import Dict, List

import pytest

from tests.agents.eudr.indigenous_rights_checker.conftest import (
    compute_test_hash,
    compute_fpic_score,
    compute_overlap_risk_score,
    compute_violation_severity,
    classify_fpic_status,
    classify_risk_level,
    haversine_km,
    DeterministicUUID,
    SHA256_HEX_LENGTH,
    FPIC_ELEMENTS,
    DEFAULT_FPIC_WEIGHTS,
    DEFAULT_OVERLAP_RISK_WEIGHTS,
    DEFAULT_VIOLATION_SEVERITY_WEIGHTS,
    OVERLAP_TYPE_SCORES,
    LEGAL_STATUS_SCORES,
    VIOLATION_TYPE_SCORES,
    FPIC_OBTAINED_THRESHOLD,
    FPIC_PARTIAL_THRESHOLD,
    ALL_COMMODITIES,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    FPICStatus,
    RiskLevel,
    OverlapType,
    TerritoryLegalStatus,
)
from greenlang.agents.eudr.indigenous_rights_checker.provenance import (
    ProvenanceTracker,
)


# ===========================================================================
# 1. FPIC Scoring Determinism (8 tests)
# ===========================================================================


class TestFPICScoringDeterminism:
    """Validate FPIC scoring is bit-perfect reproducible."""

    def test_same_input_same_score_100_runs(self):
        """Test identical inputs produce identical FPIC score over 100 runs."""
        element_scores = {
            "community_identification": Decimal("90"),
            "information_disclosure": Decimal("85"),
            "prior_timing": Decimal("100"),
            "consultation_process": Decimal("80"),
            "community_representation": Decimal("85"),
            "consent_record": Decimal("90"),
            "absence_of_coercion": Decimal("95"),
            "agreement_documentation": Decimal("80"),
            "benefit_sharing": Decimal("75"),
            "monitoring_provisions": Decimal("70"),
        }
        results = set()
        for _ in range(100):
            score = compute_fpic_score(element_scores)
            results.add(score)

        assert len(results) == 1, (
            f"Expected exactly 1 unique result, got {len(results)}: {results}"
        )

    def test_decimal_precision_preserved(self):
        """Test Decimal precision is preserved in FPIC scoring."""
        element_scores = {
            elem: Decimal("33.33") for elem in FPIC_ELEMENTS
        }
        score = compute_fpic_score(element_scores)

        # Verify result is Decimal, not float
        assert isinstance(score, Decimal)
        # Verify 2 decimal places
        assert score == score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def test_fpic_weights_sum_to_one(self):
        """Test default FPIC weights sum to exactly 1.0."""
        total = sum(Decimal(str(w)) for w in DEFAULT_FPIC_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_max_score_is_100(self):
        """Test all elements at 100 produces exactly score 100.00."""
        element_scores = {elem: Decimal("100") for elem in FPIC_ELEMENTS}
        score = compute_fpic_score(element_scores)
        assert score == Decimal("100.00")

    def test_min_score_is_0(self):
        """Test all elements at 0 produces exactly score 0.00."""
        element_scores = {elem: Decimal("0") for elem in FPIC_ELEMENTS}
        score = compute_fpic_score(element_scores)
        assert score == Decimal("0.00")

    def test_classification_boundary_80_is_obtained(self):
        """Test score of exactly 80.00 classifies as CONSENT_OBTAINED."""
        status = classify_fpic_status(Decimal("80.00"))
        assert status == FPICStatus.CONSENT_OBTAINED.value

    def test_classification_boundary_50_is_partial(self):
        """Test score of exactly 50.00 classifies as CONSENT_PARTIAL."""
        status = classify_fpic_status(Decimal("50.00"))
        assert status == FPICStatus.CONSENT_PARTIAL.value

    def test_classification_boundary_49_99_is_missing(self):
        """Test score of 49.99 classifies as CONSENT_MISSING."""
        status = classify_fpic_status(Decimal("49.99"))
        assert status == FPICStatus.CONSENT_MISSING.value


# ===========================================================================
# 2. Overlap Risk Scoring Determinism (7 tests)
# ===========================================================================


class TestOverlapRiskScoringDeterminism:
    """Validate overlap risk scoring is bit-perfect reproducible."""

    def test_same_input_same_risk_score_100_runs(self):
        """Test identical inputs produce identical overlap risk score over 100 runs."""
        results = set()
        for _ in range(100):
            score = compute_overlap_risk_score(
                overlap_type="direct",
                legal_status="titled",
                community_population=26000,
                conflict_history_score=Decimal("70"),
                country_framework_score=Decimal("45"),
            )
            results.add(score)

        assert len(results) == 1

    def test_overlap_risk_weights_sum_to_one(self):
        """Test default overlap risk weights sum to exactly 1.0."""
        total = sum(Decimal(str(w)) for w in DEFAULT_OVERLAP_RISK_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_max_overlap_risk_score(self):
        """Test maximum possible overlap risk score is 100.00."""
        score = compute_overlap_risk_score(
            overlap_type="direct",
            legal_status="titled",
            community_population=50000,
            conflict_history_score=Decimal("100"),
            country_framework_score=Decimal("100"),
        )
        assert score == Decimal("100.00")

    def test_min_overlap_risk_score(self):
        """Test minimum possible overlap risk score for 'none' type."""
        score = compute_overlap_risk_score(
            overlap_type="none",
            legal_status="pending",
            community_population=0,
            conflict_history_score=Decimal("0"),
            country_framework_score=Decimal("0"),
        )
        # none(0)*0.40 + pending(40)*0.20 + pop(20)*0.10 + 0*0.15 + 0*0.15
        expected = (
            Decimal("0") * Decimal("0.40")
            + Decimal("40") * Decimal("0.20")
            + Decimal("20") * Decimal("0.10")
            + Decimal("0") * Decimal("0.15")
            + Decimal("0") * Decimal("0.15")
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        assert score == expected

    def test_risk_classification_boundary_80(self):
        """Test score of exactly 80.00 classifies as CRITICAL."""
        level = classify_risk_level(Decimal("80.00"))
        assert level == RiskLevel.CRITICAL.value

    def test_risk_classification_boundary_60(self):
        """Test score of exactly 60.00 classifies as HIGH."""
        level = classify_risk_level(Decimal("60.00"))
        assert level == RiskLevel.HIGH.value

    def test_risk_classification_boundary_40(self):
        """Test score of exactly 40.00 classifies as MEDIUM."""
        level = classify_risk_level(Decimal("40.00"))
        assert level == RiskLevel.MEDIUM.value


# ===========================================================================
# 3. Violation Severity Determinism (5 tests)
# ===========================================================================


class TestViolationSeverityDeterminism:
    """Validate violation severity scoring is bit-perfect reproducible."""

    def test_same_input_same_severity_100_runs(self):
        """Test identical inputs produce identical violation severity over 100 runs."""
        results = set()
        for _ in range(100):
            score = compute_violation_severity(
                violation_type="physical_violence",
                proximity_score=Decimal("90"),
                population_score=Decimal("80"),
                legal_gap_score=Decimal("60"),
                media_score=Decimal("70"),
            )
            results.add(score)

        assert len(results) == 1

    def test_violation_severity_weights_sum_to_one(self):
        """Test default violation severity weights sum to exactly 1.0."""
        total = sum(Decimal(str(w)) for w in DEFAULT_VIOLATION_SEVERITY_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_known_severity_value_physical_violence(self):
        """Test known severity value for physical_violence type."""
        score = compute_violation_severity(
            violation_type="physical_violence",
            proximity_score=Decimal("100"),
            population_score=Decimal("100"),
            legal_gap_score=Decimal("100"),
            media_score=Decimal("100"),
        )
        # 100*0.30 + 100*0.25 + 100*0.15 + 100*0.15 + 100*0.15 = 100.00
        assert score == Decimal("100.00")

    def test_violation_type_scores_are_deterministic(self):
        """Test all violation type base scores are deterministic constants."""
        expected_scores = {
            "physical_violence": Decimal("100"),
            "forced_displacement": Decimal("95"),
            "land_seizure": Decimal("90"),
            "cultural_destruction": Decimal("85"),
            "fpic_violation": Decimal("80"),
            "environmental_damage": Decimal("75"),
            "consultation_denial": Decimal("70"),
            "restriction_of_access": Decimal("65"),
            "benefit_sharing_breach": Decimal("60"),
            "discriminatory_policy": Decimal("55"),
        }
        for vtype, expected in expected_scores.items():
            assert VIOLATION_TYPE_SCORES[vtype] == expected

    def test_violation_severity_with_all_zeros(self):
        """Test violation severity with all zero factor scores."""
        score = compute_violation_severity(
            violation_type="discriminatory_policy",
            proximity_score=Decimal("0"),
            population_score=Decimal("0"),
            legal_gap_score=Decimal("0"),
            media_score=Decimal("0"),
        )
        # 55*0.30 + 0 + 0 + 0 + 0 = 16.50
        expected = (
            Decimal("55") * Decimal("0.30")
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        assert score == expected


# ===========================================================================
# 4. Hash Determinism (5 tests)
# ===========================================================================


class TestHashDeterminism:
    """Validate SHA-256 hashing is deterministic."""

    def test_same_data_same_hash_1000_runs(self):
        """Test identical data produces identical hash over 1000 runs."""
        data = {
            "territory_id": "t-001",
            "territory_name": "Terra Indigena Yanomami",
            "country_code": "BR",
            "fpic_score": "87.50",
        }
        hashes = set()
        for _ in range(1000):
            h = compute_test_hash(data)
            hashes.add(h)

        assert len(hashes) == 1

    def test_hash_length_is_64_hex_chars(self):
        """Test every hash is exactly 64 hex characters (SHA-256)."""
        for i in range(100):
            h = compute_test_hash({"index": i})
            assert len(h) == SHA256_HEX_LENGTH
            assert all(c in "0123456789abcdef" for c in h)

    def test_different_data_different_hash(self):
        """Test different data produces different hashes."""
        h1 = compute_test_hash({"id": "1"})
        h2 = compute_test_hash({"id": "2"})
        assert h1 != h2

    def test_key_order_does_not_affect_hash(self):
        """Test key ordering does not affect hash (sort_keys=True)."""
        h1 = compute_test_hash({"a": 1, "b": 2, "c": 3})
        h2 = compute_test_hash({"c": 3, "a": 1, "b": 2})
        assert h1 == h2

    def test_nested_data_hash_determinism(self):
        """Test nested data structures produce deterministic hashes."""
        data = {
            "territory": {
                "id": "t-001",
                "boundary": {
                    "type": "Polygon",
                    "coordinates": [
                        [[-60.0, -3.0], [-60.0, -2.0], [-59.0, -2.0],
                         [-59.0, -3.0], [-60.0, -3.0]]
                    ],
                },
            },
            "fpic": {"score": "87.50", "status": "consent_obtained"},
        }
        hashes = set()
        for _ in range(100):
            h = compute_test_hash(data)
            hashes.add(h)

        assert len(hashes) == 1


# ===========================================================================
# 5. Provenance Chain Determinism (5 tests)
# ===========================================================================


class TestProvenanceChainDeterminism:
    """Validate provenance chain operations are deterministic."""

    def test_provenance_chain_hash_integrity(self):
        """Test provenance chain hash chaining is tamper-evident."""
        tracker = ProvenanceTracker()
        tracker.record("territory", "create", "t-001")
        tracker.record("territory", "query", "t-001")
        tracker.record("overlap", "detect", "p-001")

        assert tracker.verify_chain() is True

    def test_provenance_compute_data_hash_deterministic(self):
        """Test compute_data_hash produces same hash for same data."""
        tracker = ProvenanceTracker()
        data = {"territory_id": "t-001", "score": "85.50"}
        hashes = set()
        for _ in range(100):
            h = tracker.compute_data_hash(data)
            hashes.add(h)

        assert len(hashes) == 1
        assert len(hashes.pop()) == SHA256_HEX_LENGTH

    def test_provenance_record_hash_value_deterministic(self):
        """Test that record hash_value is computed deterministically."""
        tracker = ProvenanceTracker()
        entry = tracker.record("territory", "query", "t-001")

        # Manually recompute what the hash should be
        canonical = json.dumps({
            "entity_type": entry.entity_type,
            "action": entry.action,
            "entity_id": entry.entity_id,
            "timestamp": entry.timestamp,
            "actor": entry.actor,
            "metadata": entry.metadata,
            "previous_hash": entry.previous_hash,
        }, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        assert entry.hash_value == expected

    def test_provenance_chain_verification_after_many_records(self):
        """Test chain verification succeeds with 100 records."""
        tracker = ProvenanceTracker()
        entity_types = [
            "territory", "fpic_assessment", "overlap", "community",
            "consultation", "grievance", "agreement", "workflow",
            "violation", "report", "country_score", "config_change",
        ]
        actions = [
            "query", "create", "update", "verify", "detect",
            "correlate", "advance", "generate", "classify", "export",
        ]
        for i in range(100):
            tracker.record(
                entity_types[i % len(entity_types)],
                actions[i % len(actions)],
                f"entity-{i:04d}",
            )

        assert tracker.verify_chain() is True
        chain = tracker.get_chain()
        assert len(chain) == 100

    def test_provenance_genesis_hash_is_constant(self):
        """Test genesis hash is the same across tracker instances."""
        t1 = ProvenanceTracker()
        t2 = ProvenanceTracker()
        assert t1._genesis_hash == t2._genesis_hash
        assert t1._genesis_hash == "GL-EUDR-IRC-021-INDIGENOUS-RIGHTS-CHECKER-GENESIS"


# ===========================================================================
# 6. Haversine Determinism (3 tests)
# ===========================================================================


class TestHaversineDeterminism:
    """Validate Haversine distance calculation determinism."""

    def test_same_coordinates_same_distance_1000_runs(self):
        """Test identical coordinates produce identical distance over 1000 runs."""
        results = set()
        for _ in range(1000):
            d = haversine_km(-3.0, -60.0, -3.5, -59.5)
            results.add(d)

        assert len(results) == 1

    def test_zero_distance_for_identical_points(self):
        """Test distance is 0 for identical points."""
        d = haversine_km(0.0, 0.0, 0.0, 0.0)
        assert d == 0.0

    def test_known_distance_value(self):
        """Test a known distance value (equator, 1 degree longitude ~ 111.32 km)."""
        d = haversine_km(0.0, 0.0, 0.0, 1.0)
        # At equator, 1 degree of longitude is approximately 111.32 km
        assert abs(d - 111.32) < 0.5  # Within 0.5 km tolerance


# ===========================================================================
# 7. DeterministicUUID Determinism (2 tests)
# ===========================================================================


class TestDeterministicUUIDDeterminism:
    """Validate DeterministicUUID sequence is reproducible."""

    def test_sequential_ids_are_predictable(self):
        """Test UUID generator produces predictable sequential IDs."""
        gen = DeterministicUUID(prefix="test")
        assert gen.next() == "test-00000001"
        assert gen.next() == "test-00000002"
        assert gen.next() == "test-00000003"

    def test_reset_restarts_sequence(self):
        """Test reset restarts the sequence from 1."""
        gen = DeterministicUUID(prefix="det")
        gen.next()
        gen.next()
        gen.reset()
        assert gen.next() == "det-00000001"
