# -*- coding: utf-8 -*-
"""
Tests for Determinism - AGENT-EUDR-023 Reproducibility Guarantees

Comprehensive test suite covering:
- Bit-perfect reproducibility of compliance scores
- Red flag scoring consistency across repeated runs
- Compliance determination consistency
- Provenance chain integrity and determinism
- Hash computation determinism
- Document verification determinism
- Certification validation determinism
- Category score aggregation determinism
- Cross-run consistency validation
- Ordering independence for set-based operations

Test count: 40+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 (Determinism & Reproducibility Tests)
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from tests.agents.eudr.legal_compliance_verifier.conftest import (
    compute_test_hash,
    compute_compliance_score,
    determine_compliance,
    classify_red_flag_severity,
    apply_country_multiplier,
    apply_commodity_multiplier,
    is_document_expired,
    days_until_expiry,
    SHA256_HEX_LENGTH,
    LEGISLATION_CATEGORIES,
    EUDR_COMMODITIES,
    EUDR_COUNTRIES_27,
    DOCUMENT_TYPES,
    RED_FLAG_INDICATORS,
    COUNTRY_MULTIPLIERS,
    COMMODITY_MULTIPLIERS,
    COMPLIANCE_BOUNDARIES,
    RED_FLAG_BOUNDARIES,
)


# ===========================================================================
# 1. Compliance Score Reproducibility (10 tests)
# ===========================================================================


class TestComplianceScoreReproducibility:
    """Test bit-perfect reproducibility of compliance scores."""

    def test_same_input_same_score_10_runs(self):
        """Test 10 runs with same input produce identical scores."""
        scores = {cat: Decimal("75") for cat in LEGISLATION_CATEGORIES}
        results = [compute_compliance_score(scores) for _ in range(10)]
        assert all(r == results[0] for r in results)

    def test_same_input_same_score_100_runs(self):
        """Test 100 runs with same input produce identical scores."""
        scores = {
            "land_use_rights": Decimal("85"),
            "environmental_protection": Decimal("72"),
            "forest_related_rules": Decimal("90"),
            "third_party_rights": Decimal("45"),
            "labour_rights": Decimal("60"),
            "tax_and_royalty": Decimal("88"),
            "trade_and_customs": Decimal("55"),
            "anti_corruption": Decimal("30"),
        }
        first = compute_compliance_score(scores)
        for _ in range(99):
            assert compute_compliance_score(scores) == first

    def test_score_exact_decimal_precision(self):
        """Test score precision uses exact Decimal arithmetic."""
        scores = {cat: Decimal("33") for cat in LEGISLATION_CATEGORIES}
        result = compute_compliance_score(scores)
        # 33 * 8 / 8 = 33.00
        assert result == Decimal("33.00") or result == Decimal("33")

    def test_score_unaffected_by_category_order(self):
        """Test score is unaffected by dictionary key ordering."""
        scores1 = {}
        scores2 = {}
        for i, cat in enumerate(LEGISLATION_CATEGORIES):
            scores1[cat] = Decimal(str(50 + i * 5))
        for cat in reversed(LEGISLATION_CATEGORIES):
            idx = LEGISLATION_CATEGORIES.index(cat)
            scores2[cat] = Decimal(str(50 + idx * 5))
        assert compute_compliance_score(scores1) == compute_compliance_score(scores2)

    @pytest.mark.parametrize("value", [0, 25, 50, 75, 100])
    def test_uniform_scores_produce_same_value(self, value):
        """Test uniform scores across all categories produce that value."""
        scores = {cat: Decimal(str(value)) for cat in LEGISLATION_CATEGORIES}
        result = compute_compliance_score(scores)
        assert result == Decimal(str(value))

    def test_empty_scores_produce_zero(self):
        """Test empty scores produce zero."""
        result = compute_compliance_score({})
        assert result == Decimal("0")

    def test_single_category_score(self):
        """Test single category score divides correctly."""
        scores = {"land_use_rights": Decimal("80")}
        result = compute_compliance_score(scores)
        assert result == Decimal("80")

    def test_decimal_precision_maintained(self):
        """Test Decimal precision is maintained through computation."""
        scores = {
            "land_use_rights": Decimal("77.77"),
            "environmental_protection": Decimal("88.88"),
        }
        result = compute_compliance_score(scores)
        # (77.77 + 88.88) / 2 = 83.325 -> 83.33
        expected = ((Decimal("77.77") + Decimal("88.88")) / Decimal("2")).quantize(Decimal("0.01"))
        assert result == expected

    def test_large_precision_values(self):
        """Test computation with many decimal places."""
        scores = {cat: Decimal("66.666666666") for cat in LEGISLATION_CATEGORIES}
        r1 = compute_compliance_score(scores)
        r2 = compute_compliance_score(scores)
        assert r1 == r2

    def test_boundary_values_deterministic(self):
        """Test boundary values produce consistent results."""
        for boundary in COMPLIANCE_BOUNDARIES:
            scores = {cat: Decimal(str(boundary)) for cat in LEGISLATION_CATEGORIES}
            results = [compute_compliance_score(scores) for _ in range(5)]
            assert all(r == results[0] for r in results)


# ===========================================================================
# 2. Red Flag Scoring Consistency (8 tests)
# ===========================================================================


class TestRedFlagScoringConsistency:
    """Test red flag scoring consistency across repeated runs."""

    def test_severity_deterministic_repeated(self):
        """Test severity classification is deterministic over 100 runs."""
        score = Decimal("65")
        results = [classify_red_flag_severity(score) for _ in range(100)]
        assert all(r == "high" for r in results)

    def test_country_multiplier_deterministic(self):
        """Test country multiplier application is deterministic."""
        results = [apply_country_multiplier(50, "BR") for _ in range(100)]
        assert all(r == results[0] for r in results)
        assert results[0] == Decimal("50") * Decimal("1.3")

    def test_commodity_multiplier_deterministic(self):
        """Test commodity multiplier application is deterministic."""
        results = [apply_commodity_multiplier(Decimal("65"), "oil_palm") for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_combined_multiplier_deterministic(self):
        """Test combined country + commodity multiplier is deterministic."""
        results = []
        for _ in range(100):
            score = apply_country_multiplier(60, "CD")
            score = apply_commodity_multiplier(score, "oil_palm")
            results.append(score)
        assert all(r == results[0] for r in results)

    @pytest.mark.parametrize("boundary", RED_FLAG_BOUNDARIES)
    def test_severity_at_boundary_deterministic(self, boundary):
        """Test severity at each boundary value is deterministic."""
        results = [classify_red_flag_severity(Decimal(str(boundary))) for _ in range(10)]
        assert all(r == results[0] for r in results)

    def test_all_indicators_score_deterministically(self):
        """Test all 40 indicators score deterministically."""
        for indicator in RED_FLAG_INDICATORS:
            base = indicator["base_score"]
            scores = [apply_country_multiplier(base, "BR") for _ in range(10)]
            assert all(s == scores[0] for s in scores)

    def test_severity_transitions_precise(self):
        """Test severity transitions happen at exact threshold values."""
        # Just below moderate threshold
        assert classify_red_flag_severity(Decimal("24.99")) == "low"
        # At moderate threshold
        assert classify_red_flag_severity(Decimal("25")) == "moderate"
        # Just below high threshold
        assert classify_red_flag_severity(Decimal("49.99")) == "moderate"
        # At high threshold
        assert classify_red_flag_severity(Decimal("50")) == "high"

    def test_multiplier_precision(self):
        """Test multiplier arithmetic maintains Decimal precision."""
        base = Decimal("73")
        country_mult = COUNTRY_MULTIPLIERS["BR"]
        commodity_mult = COMMODITY_MULTIPLIERS["soya"]
        expected = base * country_mult * commodity_mult
        for _ in range(50):
            result = apply_commodity_multiplier(
                apply_country_multiplier(73, "BR"), "soya",
            )
            assert result == expected


# ===========================================================================
# 3. Compliance Determination Consistency (6 tests)
# ===========================================================================


class TestComplianceDeterminationConsistency:
    """Test compliance determination consistency."""

    @pytest.mark.parametrize("score,expected", [
        (Decimal("0"), "NON_COMPLIANT"),
        (Decimal("49.99"), "NON_COMPLIANT"),
        (Decimal("50"), "PARTIALLY_COMPLIANT"),
        (Decimal("79.99"), "PARTIALLY_COMPLIANT"),
        (Decimal("80"), "COMPLIANT"),
        (Decimal("100"), "COMPLIANT"),
    ])
    def test_determination_boundary_consistency(self, score, expected):
        """Test determination at boundaries is consistent over 100 runs."""
        results = [determine_compliance(score) for _ in range(100)]
        assert all(r == expected for r in results)

    def test_determination_with_custom_thresholds_consistent(self):
        """Test determination with custom thresholds is consistent."""
        results = [
            determine_compliance(Decimal("75"), compliant=90, partial=40)
            for _ in range(100)
        ]
        assert all(r == "PARTIALLY_COMPLIANT" for r in results)

    def test_full_range_determination_consistency(self):
        """Test determination is consistent for all integer scores 0-100."""
        for score_int in range(101):
            score = Decimal(str(score_int))
            results = [determine_compliance(score) for _ in range(5)]
            assert all(r == results[0] for r in results), (
                f"Inconsistency at score {score_int}"
            )


# ===========================================================================
# 4. Provenance Chain Integrity (8 tests)
# ===========================================================================


class TestProvenanceChainIntegrity:
    """Test provenance chain integrity and determinism."""

    def test_hash_deterministic_simple(self):
        """Test hash of simple data is deterministic."""
        data = {"key": "value", "number": 42}
        h1 = compute_test_hash(data)
        h2 = compute_test_hash(data)
        assert h1 == h2

    def test_hash_deterministic_complex(self):
        """Test hash of complex nested data is deterministic."""
        data = {
            "supplier": "SUP-0001",
            "scores": {"cat1": "85", "cat2": "72"},
            "flags": [{"id": "RF-001", "score": "65"}],
            "timestamp": "2025-06-15T10:30:00Z",
        }
        results = [compute_test_hash(data) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_hash_changes_with_any_data_change(self):
        """Test hash changes when any data field changes."""
        base = {"a": "1", "b": "2", "c": "3"}
        base_hash = compute_test_hash(base)

        for key in base:
            modified = dict(base)
            modified[key] = "changed"
            assert compute_test_hash(modified) != base_hash

    def test_hash_key_order_independence(self):
        """Test hash is independent of key insertion order."""
        d1 = {"a": "1", "b": "2", "c": "3"}
        d2 = {"c": "3", "a": "1", "b": "2"}
        assert compute_test_hash(d1) == compute_test_hash(d2)

    def test_provenance_chain_builds_deterministically(self):
        """Test building a provenance chain produces same result each time."""
        def build_chain():
            chain = []
            for step in ["framework", "documents", "certifications", "red_flags", "compliance"]:
                h = compute_test_hash({"step": step, "data": "test"})
                chain.append(h)
            return compute_test_hash({"chain": chain})

        results = [build_chain() for _ in range(50)]
        assert all(r == results[0] for r in results)

    def test_provenance_chain_integrity_on_modification(self):
        """Test provenance chain hash changes when any step changes."""
        def build_chain(modified_step=None):
            chain = []
            for step in ["framework", "documents", "certifications"]:
                data = "modified" if step == modified_step else "original"
                chain.append(compute_test_hash({"step": step, "data": data}))
            return compute_test_hash({"chain": chain})

        original = build_chain()
        modified_fw = build_chain("framework")
        modified_doc = build_chain("documents")
        modified_cert = build_chain("certifications")

        assert original != modified_fw
        assert original != modified_doc
        assert original != modified_cert

    def test_hash_length_always_64(self):
        """Test hash is always 64 characters (SHA-256 hex)."""
        test_cases = [
            {},
            {"key": "value"},
            {"a": "b", "c": [1, 2, 3]},
            {"nested": {"deep": {"data": True}}},
            {"large": "x" * 10000},
        ]
        for data in test_cases:
            h = compute_test_hash(data)
            assert len(h) == SHA256_HEX_LENGTH

    def test_hash_contains_only_hex_chars(self):
        """Test hash contains only valid hexadecimal characters."""
        h = compute_test_hash({"test": "data"})
        assert all(c in "0123456789abcdef" for c in h)


# ===========================================================================
# 5. Document Verification Determinism (4 tests)
# ===========================================================================


class TestDocumentVerificationDeterminism:
    """Test document verification produces deterministic results."""

    def test_expiry_check_deterministic(self):
        """Test document expiry check is deterministic."""
        future = (date.today() + timedelta(days=100)).isoformat()
        past = (date.today() - timedelta(days=100)).isoformat()
        for _ in range(100):
            assert is_document_expired(future) is False
            assert is_document_expired(past) is True

    def test_days_until_expiry_deterministic(self):
        """Test days until expiry calculation is deterministic."""
        target = (date.today() + timedelta(days=42)).isoformat()
        results = [days_until_expiry(target) for _ in range(100)]
        assert all(r == 42 for r in results)

    def test_document_hash_deterministic(self):
        """Test document hash generation is deterministic."""
        for doc_type in DOCUMENT_TYPES:
            data = {"doc_id": f"DOC-{doc_type}", "type": doc_type}
            hashes = [compute_test_hash(data) for _ in range(10)]
            assert all(h == hashes[0] for h in hashes)

    def test_batch_verification_order_consistent(self):
        """Test batch verification returns results in consistent order."""
        today = date.today()
        docs = [
            {"document_id": f"DET-{i}", "expiry_date": (today + timedelta(days=i*30)).isoformat()}
            for i in range(10)
        ]
        for _ in range(10):
            results = [is_document_expired(d["expiry_date"]) for d in docs]
            assert results == [is_document_expired(d["expiry_date"]) for d in docs]


# ===========================================================================
# 6. Cross-Run Consistency (4 tests)
# ===========================================================================


class TestCrossRunConsistency:
    """Test consistency across simulated independent runs."""

    def test_full_pipeline_deterministic(self):
        """Test full compliance pipeline is deterministic end-to-end."""
        def run_pipeline():
            scores = {
                "land_use_rights": Decimal("85"),
                "environmental_protection": Decimal("72"),
                "forest_related_rules": Decimal("90"),
                "third_party_rights": Decimal("45"),
                "labour_rights": Decimal("60"),
                "tax_and_royalty": Decimal("88"),
                "trade_and_customs": Decimal("55"),
                "anti_corruption": Decimal("30"),
            }
            overall = compute_compliance_score(scores)
            determination = determine_compliance(overall)
            flag_score = apply_country_multiplier(65, "BR")
            flag_score = apply_commodity_multiplier(flag_score, "soya")
            severity = classify_red_flag_severity(flag_score)
            provenance = compute_test_hash({
                "score": str(overall),
                "determination": determination,
                "flag_severity": severity,
            })
            return {
                "overall": overall,
                "determination": determination,
                "flag_severity": severity,
                "provenance": provenance,
            }

        results = [run_pipeline() for _ in range(100)]
        first = results[0]
        for r in results[1:]:
            assert r == first

    def test_independent_assessments_same_input_same_output(self):
        """Test independent assessments with same input produce same output."""
        scores = {cat: Decimal(str(60 + i * 3)) for i, cat in enumerate(LEGISLATION_CATEGORIES)}
        for _ in range(50):
            s = compute_compliance_score(scores)
            d = determine_compliance(s)
            h = compute_test_hash({"score": str(s), "det": d})
            assert s == compute_compliance_score(scores)
            assert d == determine_compliance(s)
            assert h == compute_test_hash({"score": str(s), "det": d})

    def test_no_state_leaks_between_computations(self):
        """Test no state leaks between sequential computations."""
        # Run with different inputs and verify no cross-contamination
        for i in range(20):
            scores = {cat: Decimal(str(i * 5)) for cat in LEGISLATION_CATEGORIES}
            result = compute_compliance_score(scores)
            expected = Decimal(str(i * 5))
            assert result == expected

    def test_thread_safety_simulation(self):
        """Test computations are safe for concurrent access (simulated)."""
        # Simulate concurrent access by interleaving computations
        results_a = []
        results_b = []
        for i in range(50):
            # Interleave two different computations
            score_a = compute_compliance_score(
                {cat: Decimal("80") for cat in LEGISLATION_CATEGORIES}
            )
            score_b = compute_compliance_score(
                {cat: Decimal("40") for cat in LEGISLATION_CATEGORIES}
            )
            results_a.append(score_a)
            results_b.append(score_b)

        assert all(r == Decimal("80") for r in results_a)
        assert all(r == Decimal("40") for r in results_b)
