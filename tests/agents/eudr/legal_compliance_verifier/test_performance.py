# -*- coding: utf-8 -*-
"""
Tests for Performance - AGENT-EUDR-023 Performance Benchmarks

Comprehensive test suite covering:
- Single compliance check latency (< 500ms target)
- Full 8-category assessment latency (< 5s target)
- Batch 1000 suppliers throughput (< 60s target)
- Red flag scoring throughput
- Document verification throughput
- Certification validation throughput
- Report generation throughput
- Memory usage during batch processing
- Concurrent operation handling
- Score computation performance

Test count: 45+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 (Performance Tests)
"""

import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
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
)


# ---------------------------------------------------------------------------
# Performance test helpers
# ---------------------------------------------------------------------------


def _measure_time(func, *args, **kwargs):
    """Measure execution time of a function in milliseconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def _single_compliance_check(
    supplier_id: str,
    country_code: str,
    commodity: str,
) -> Dict[str, Any]:
    """Perform a single compliance check (all 8 categories)."""
    scores = {}
    for i, cat in enumerate(LEGISLATION_CATEGORIES):
        base = 60 + (i * 7) % 40
        scores[cat] = Decimal(str(base))

    overall = compute_compliance_score(scores)
    determination = determine_compliance(overall)

    return {
        "supplier_id": supplier_id,
        "overall_score": overall,
        "determination": determination,
        "provenance_hash": compute_test_hash({
            "supplier": supplier_id, "score": str(overall),
        }),
    }


def _verify_document_batch(documents: List[Dict]) -> List[Dict]:
    """Verify a batch of documents for performance testing."""
    results = []
    today = date.today()
    for doc in documents:
        expiry = doc.get("expiry_date", "")
        is_expired = is_document_expired(expiry) if expiry else True
        results.append({
            "document_id": doc.get("document_id"),
            "verified": not is_expired,
            "hash": compute_test_hash({"doc": doc.get("document_id")}),
        })
    return results


def _score_red_flags(
    indicators: List[Dict],
    country_code: str,
    commodity: str,
) -> List[Dict]:
    """Score red flag indicators for performance testing."""
    results = []
    for indicator in indicators:
        base = indicator["base_score"]
        adjusted = apply_country_multiplier(base, country_code)
        adjusted = apply_commodity_multiplier(adjusted, commodity)
        severity = classify_red_flag_severity(adjusted)
        results.append({
            "indicator_id": indicator["id"],
            "adjusted_score": adjusted,
            "severity": severity,
        })
    return results


def _generate_test_documents(count: int) -> List[Dict]:
    """Generate test documents for performance testing."""
    today = date.today()
    docs = []
    for i in range(count):
        docs.append({
            "document_id": f"PERF-DOC-{i:06d}",
            "document_type": DOCUMENT_TYPES[i % len(DOCUMENT_TYPES)],
            "expiry_date": (today + timedelta(days=365 - (i % 730))).isoformat(),
        })
    return docs


def _generate_test_suppliers(count: int) -> List[Dict]:
    """Generate test suppliers for batch performance testing."""
    return [
        {
            "supplier_id": f"PERF-SUP-{i:06d}",
            "country_code": EUDR_COUNTRIES_27[i % len(EUDR_COUNTRIES_27)],
            "commodity": EUDR_COMMODITIES[i % len(EUDR_COMMODITIES)],
        }
        for i in range(count)
    ]


# ===========================================================================
# 1. Single Compliance Check Performance (8 tests)
# ===========================================================================


class TestSingleCheckPerformance:
    """Test single compliance check meets latency targets."""

    def test_single_check_under_500ms(self):
        """Test single compliance check completes in under 500ms."""
        _, elapsed = _measure_time(
            _single_compliance_check, "SUP-0001", "BR", "soya",
        )
        assert elapsed < 500, f"Single check took {elapsed:.1f}ms (target: <500ms)"

    def test_single_check_under_100ms(self):
        """Test single compliance check typically completes in under 100ms."""
        _, elapsed = _measure_time(
            _single_compliance_check, "SUP-0001", "BR", "wood",
        )
        assert elapsed < 100, f"Single check took {elapsed:.1f}ms (target: <100ms)"

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_single_check_all_commodities(self, commodity):
        """Test single check performance for each commodity."""
        _, elapsed = _measure_time(
            _single_compliance_check, "SUP-PERF", "BR", commodity,
        )
        assert elapsed < 500

    def test_single_check_returns_valid_result(self):
        """Test single check returns valid determination."""
        result, _ = _measure_time(
            _single_compliance_check, "SUP-0001", "BR", "soya",
        )
        assert result["determination"] in ("COMPLIANT", "PARTIALLY_COMPLIANT", "NON_COMPLIANT")

    def test_single_check_includes_provenance(self):
        """Test single check generates provenance hash."""
        result, elapsed = _measure_time(
            _single_compliance_check, "SUP-0001", "BR", "soya",
        )
        assert len(result["provenance_hash"]) == SHA256_HEX_LENGTH
        assert elapsed < 500


# ===========================================================================
# 2. Full Assessment Performance (5 tests)
# ===========================================================================


class TestFullAssessmentPerformance:
    """Test full 8-category assessment meets latency targets."""

    def test_full_assessment_under_5s(self):
        """Test full assessment completes in under 5 seconds."""
        start = time.perf_counter()
        for _ in range(10):
            _single_compliance_check("SUP-FULL", "BR", "wood")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5000

    def test_full_assessment_with_documents(self):
        """Test assessment with document verification under 5s."""
        docs = _generate_test_documents(12)
        start = time.perf_counter()
        _verify_document_batch(docs)
        _single_compliance_check("SUP-DOCS", "BR", "soya")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5000

    def test_full_assessment_with_red_flags(self):
        """Test assessment with red flag scoring under 5s."""
        start = time.perf_counter()
        _score_red_flags(RED_FLAG_INDICATORS, "BR", "soya")
        _single_compliance_check("SUP-FLAGS", "BR", "soya")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5000

    def test_assessment_all_countries(self):
        """Test assessments for all 27 countries under 5s."""
        start = time.perf_counter()
        for country in EUDR_COUNTRIES_27:
            _single_compliance_check(f"SUP-{country}", country, "wood")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5000

    def test_repeated_assessment_consistent_time(self):
        """Test repeated assessments have consistent timing."""
        times = []
        for _ in range(20):
            _, elapsed = _measure_time(
                _single_compliance_check, "SUP-REPEAT", "BR", "soya",
            )
            times.append(elapsed)
        avg = sum(times) / len(times)
        max_time = max(times)
        assert max_time < avg * 5  # No outlier more than 5x average


# ===========================================================================
# 3. Batch Processing Performance (8 tests)
# ===========================================================================


class TestBatchPerformance:
    """Test batch processing meets throughput targets."""

    def test_batch_100_suppliers(self):
        """Test batch assessment of 100 suppliers."""
        suppliers = _generate_test_suppliers(100)
        start = time.perf_counter()
        for s in suppliers:
            _single_compliance_check(s["supplier_id"], s["country_code"], s["commodity"])
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 10, f"100 suppliers took {elapsed_s:.1f}s (target: <10s)"

    def test_batch_1000_suppliers_under_60s(self):
        """Test batch assessment of 1000 suppliers under 60 seconds."""
        suppliers = _generate_test_suppliers(1000)
        start = time.perf_counter()
        for s in suppliers:
            _single_compliance_check(s["supplier_id"], s["country_code"], s["commodity"])
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 60, f"1000 suppliers took {elapsed_s:.1f}s (target: <60s)"

    def test_batch_throughput_rate(self):
        """Test batch processing achieves minimum throughput rate."""
        count = 500
        suppliers = _generate_test_suppliers(count)
        start = time.perf_counter()
        for s in suppliers:
            _single_compliance_check(s["supplier_id"], s["country_code"], s["commodity"])
        elapsed_s = time.perf_counter() - start
        throughput = count / elapsed_s
        assert throughput >= 10, f"Throughput: {throughput:.0f}/s (target: >=10/s)"

    def test_batch_document_verification_1000(self):
        """Test batch verification of 1000 documents."""
        docs = _generate_test_documents(1000)
        start = time.perf_counter()
        results = _verify_document_batch(docs)
        elapsed_s = time.perf_counter() - start
        assert len(results) == 1000
        assert elapsed_s < 30

    def test_batch_document_verification_10000(self):
        """Test batch verification of 10000 documents."""
        docs = _generate_test_documents(10000)
        start = time.perf_counter()
        results = _verify_document_batch(docs)
        elapsed_s = time.perf_counter() - start
        assert len(results) == 10000
        assert elapsed_s < 60

    def test_batch_red_flag_scoring(self):
        """Test batch red flag scoring for 1000 suppliers."""
        start = time.perf_counter()
        for i in range(1000):
            country = EUDR_COUNTRIES_27[i % len(EUDR_COUNTRIES_27)]
            commodity = EUDR_COMMODITIES[i % len(EUDR_COMMODITIES)]
            _score_red_flags(RED_FLAG_INDICATORS, country, commodity)
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 60

    def test_batch_results_complete(self):
        """Test all batch results are returned."""
        suppliers = _generate_test_suppliers(50)
        results = [
            _single_compliance_check(s["supplier_id"], s["country_code"], s["commodity"])
            for s in suppliers
        ]
        assert len(results) == 50
        assert all(r["determination"] is not None for r in results)

    def test_batch_scales_linearly(self):
        """Test batch processing time scales approximately linearly."""
        suppliers_100 = _generate_test_suppliers(100)
        suppliers_200 = _generate_test_suppliers(200)

        start = time.perf_counter()
        for s in suppliers_100:
            _single_compliance_check(s["supplier_id"], s["country_code"], s["commodity"])
        time_100 = time.perf_counter() - start

        start = time.perf_counter()
        for s in suppliers_200:
            _single_compliance_check(s["supplier_id"], s["country_code"], s["commodity"])
        time_200 = time.perf_counter() - start

        # 200 should take no more than 3x of 100 (allowing overhead)
        assert time_200 < time_100 * 3


# ===========================================================================
# 4. Hash Computation Performance (5 tests)
# ===========================================================================


class TestHashPerformance:
    """Test SHA-256 hash computation performance."""

    def test_hash_1000_computations(self):
        """Test 1000 SHA-256 hash computations under 1 second."""
        start = time.perf_counter()
        for i in range(1000):
            compute_test_hash({"key": f"value-{i}", "number": i})
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 1.0

    def test_hash_10000_computations(self):
        """Test 10000 SHA-256 hash computations under 5 seconds."""
        start = time.perf_counter()
        for i in range(10000):
            compute_test_hash({"key": f"value-{i}", "number": i})
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 5.0

    def test_hash_large_payload(self):
        """Test hashing a large payload."""
        large_data = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        _, elapsed = _measure_time(compute_test_hash, large_data)
        assert elapsed < 100  # Should be under 100ms even for large payloads

    def test_hash_consistency_under_load(self):
        """Test hash consistency when computed under load."""
        data = {"test": "deterministic", "number": 42}
        hashes = set()
        for _ in range(1000):
            h = compute_test_hash(data)
            hashes.add(h)
        assert len(hashes) == 1  # All hashes should be identical

    def test_hash_throughput(self):
        """Test hash computation throughput."""
        count = 5000
        start = time.perf_counter()
        for i in range(count):
            compute_test_hash({"i": i})
        elapsed_s = time.perf_counter() - start
        throughput = count / elapsed_s
        assert throughput >= 1000  # At least 1000 hashes/second


# ===========================================================================
# 5. Score Computation Performance (8 tests)
# ===========================================================================


class TestScoreComputationPerformance:
    """Test score computation and classification performance."""

    def test_compliance_score_1000_computations(self):
        """Test 1000 compliance score computations."""
        scores = {cat: Decimal(str(50 + i * 5)) for i, cat in enumerate(LEGISLATION_CATEGORIES)}
        start = time.perf_counter()
        for _ in range(1000):
            compute_compliance_score(scores)
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 1.0

    def test_determination_1000_computations(self):
        """Test 1000 compliance determination computations."""
        start = time.perf_counter()
        for i in range(1000):
            determine_compliance(Decimal(str(i % 101)))
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 1.0

    def test_severity_classification_10000(self):
        """Test 10000 severity classifications."""
        start = time.perf_counter()
        for i in range(10000):
            classify_red_flag_severity(Decimal(str(i % 101)))
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 2.0

    def test_multiplier_application_10000(self):
        """Test 10000 multiplier applications."""
        start = time.perf_counter()
        for i in range(10000):
            country = EUDR_COUNTRIES_27[i % len(EUDR_COUNTRIES_27)]
            commodity = EUDR_COMMODITIES[i % len(EUDR_COMMODITIES)]
            score = apply_country_multiplier(50 + i % 50, country)
            apply_commodity_multiplier(score, commodity)
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 2.0

    def test_expiry_check_10000(self):
        """Test 10000 document expiry checks."""
        today = date.today()
        dates = [(today + timedelta(days=i - 5000)).isoformat() for i in range(10000)]
        start = time.perf_counter()
        for d in dates:
            is_document_expired(d)
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 2.0

    def test_days_until_expiry_10000(self):
        """Test 10000 days-until-expiry calculations."""
        today = date.today()
        dates = [(today + timedelta(days=i)).isoformat() for i in range(10000)]
        start = time.perf_counter()
        for d in dates:
            days_until_expiry(d)
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 2.0

    def test_combined_scoring_pipeline(self):
        """Test full scoring pipeline performance."""
        start = time.perf_counter()
        for i in range(500):
            # Full pipeline: score -> determine -> classify -> hash
            scores = {cat: Decimal(str(40 + (i + j) % 60))
                      for j, cat in enumerate(LEGISLATION_CATEGORIES)}
            overall = compute_compliance_score(scores)
            determine_compliance(overall)
            classify_red_flag_severity(Decimal(str(30 + i % 70)))
            compute_test_hash({"score": str(overall), "i": i})
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 5.0

    def test_provenance_chain_building(self):
        """Test provenance chain building performance."""
        start = time.perf_counter()
        for i in range(500):
            chain = []
            for step in range(7):  # 7 engines
                h = compute_test_hash({"step": step, "data": f"value-{i}"})
                chain.append(h)
            # Final hash
            compute_test_hash({"chain": chain})
        elapsed_s = time.perf_counter() - start
        assert elapsed_s < 5.0
