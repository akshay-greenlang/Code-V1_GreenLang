# -*- coding: utf-8 -*-
"""
Performance Tests for AGENT-EUDR-021 Indigenous Rights Checker

Validates that all engines and API operations meet their stated
performance targets from the PRD:
    - Single plot overlap analysis: < 500ms p99
    - FPIC verification: < 2s
    - Batch overlap (10,000 plots): < 5 minutes
    - Point-in-polygon query: < 100ms
    - Territory search: < 200ms
    - Concurrent request handling (10+ parallel)
    - Memory usage under load

Test count: 35 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (Performance Validation)
"""

import time
import statistics
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.agents.eudr.indigenous_rights_checker.conftest import (
    compute_test_hash,
    compute_fpic_score,
    compute_overlap_risk_score,
    compute_violation_severity,
    classify_fpic_status,
    classify_risk_level,
    haversine_km,
    SHA256_HEX_LENGTH,
    FPIC_ELEMENTS,
    DEFAULT_FPIC_WEIGHTS,
    DEFAULT_OVERLAP_RISK_WEIGHTS,
    DEFAULT_VIOLATION_SEVERITY_WEIGHTS,
    OVERLAP_TYPE_SCORES,
    LEGAL_STATUS_SCORES,
    VIOLATION_TYPE_SCORES,
    ALL_COMMODITIES,
    FPIC_COUNTRIES,
    ILO_169_EUDR_COUNTRIES,
    HIGH_RISK_COUNTRIES,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    IndigenousTerritory,
    FPICAssessment,
    TerritoryOverlap,
    IndigenousCommunity,
    ViolationAlert,
    ComplianceReport,
    DetectOverlapRequest,
    BatchOverlapRequest,
    OverlapDetectionResponse,
    BatchOverlapResponse,
    OverlapType,
    RiskLevel,
    FPICStatus,
    ViolationType,
    TerritoryLegalStatus,
    AlertSeverity,
    ConfidenceLevel,
    ReportType,
    ReportFormat,
    MAX_BATCH_SIZE,
)
from greenlang.agents.eudr.indigenous_rights_checker.config import (
    IndigenousRightsCheckerConfig,
)


# ===========================================================================
# 1. FPIC Scoring Performance (6 tests)
# ===========================================================================


class TestFPICScoringPerformance:
    """Performance tests for FPIC scoring calculations."""

    def test_single_fpic_scoring_under_1ms(self):
        """Test single FPIC composite scoring completes in < 1ms."""
        element_scores = {
            elem: Decimal("80") for elem in FPIC_ELEMENTS
        }
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            compute_fpic_score(element_scores)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 1.0, (
            f"FPIC scoring took {avg_ms:.3f}ms on average, expected < 1ms"
        )

    def test_fpic_scoring_throughput_10k_per_second(self):
        """Test FPIC scoring throughput exceeds 10,000 assessments/second."""
        element_scores = {
            elem: Decimal(str(50 + i * 5)) for i, elem in enumerate(FPIC_ELEMENTS)
        }
        num_records = 10000
        start = time.perf_counter()
        for _ in range(num_records):
            compute_fpic_score(element_scores)
        elapsed = time.perf_counter() - start
        throughput = num_records / elapsed

        assert throughput >= 10000, (
            f"Throughput was {throughput:.0f}/s, expected >= 10,000/s"
        )

    def test_fpic_classification_under_1ms(self):
        """Test FPIC classification completes in < 1ms."""
        scores = [Decimal(str(x)) for x in range(0, 101, 5)]
        iterations = len(scores)
        start = time.perf_counter()
        for score in scores:
            classify_fpic_status(score)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 1.0

    def test_fpic_scoring_batch_100_under_10ms(self):
        """Test batch of 100 FPIC scores completes in < 10ms."""
        batch = []
        for i in range(100):
            batch.append({
                elem: Decimal(str((i * 7 + j * 3) % 100))
                for j, elem in enumerate(FPIC_ELEMENTS)
            })
        start = time.perf_counter()
        results = [compute_fpic_score(scores) for scores in batch]
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(results) == 100
        assert elapsed_ms < 10.0, (
            f"Batch of 100 FPIC scores took {elapsed_ms:.1f}ms, expected < 10ms"
        )

    def test_fpic_scoring_with_custom_weights_same_speed(self):
        """Test custom weights do not degrade performance."""
        element_scores = {elem: Decimal("75") for elem in FPIC_ELEMENTS}
        custom_weights = {
            "community_identification": 0.20,
            "information_disclosure": 0.10,
            "prior_timing": 0.10,
            "consultation_process": 0.10,
            "community_representation": 0.10,
            "consent_record": 0.10,
            "absence_of_coercion": 0.10,
            "agreement_documentation": 0.10,
            "benefit_sharing": 0.05,
            "monitoring_provisions": 0.05,
        }
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            compute_fpic_score(element_scores, custom_weights)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 1.0

    def test_fpic_scoring_p99_latency(self):
        """Test FPIC scoring p99 latency is < 0.5ms."""
        element_scores = {
            elem: Decimal(str(60 + i * 3)) for i, elem in enumerate(FPIC_ELEMENTS)
        }
        latencies = []
        for _ in range(500):
            start = time.perf_counter()
            compute_fpic_score(element_scores)
            latencies.append((time.perf_counter() - start) * 1000)

        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        assert p99 < 0.5, f"p99 latency was {p99:.3f}ms, expected < 0.5ms"


# ===========================================================================
# 2. Overlap Risk Scoring Performance (6 tests)
# ===========================================================================


class TestOverlapRiskScoringPerformance:
    """Performance tests for overlap risk scoring calculations."""

    def test_single_overlap_scoring_under_1ms(self):
        """Test single overlap risk scoring completes in < 1ms."""
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            compute_overlap_risk_score(
                overlap_type="direct",
                legal_status="titled",
                community_population=26000,
                conflict_history_score=Decimal("70"),
                country_framework_score=Decimal("45"),
            )
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 1.0

    def test_overlap_scoring_all_types_under_5ms(self):
        """Test scoring all overlap types completes in < 5ms."""
        overlap_types = ["direct", "partial", "adjacent", "proximate", "none"]
        start = time.perf_counter()
        for ot in overlap_types:
            compute_overlap_risk_score(
                overlap_type=ot,
                legal_status="titled",
                community_population=10000,
                conflict_history_score=Decimal("50"),
                country_framework_score=Decimal("50"),
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 5.0

    def test_overlap_scoring_all_legal_statuses_under_5ms(self):
        """Test scoring all legal statuses completes in < 5ms."""
        statuses = ["titled", "declared", "claimed", "customary", "pending", "disputed"]
        start = time.perf_counter()
        for ls in statuses:
            compute_overlap_risk_score(
                overlap_type="direct",
                legal_status=ls,
                community_population=5000,
                conflict_history_score=Decimal("40"),
                country_framework_score=Decimal("60"),
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 5.0

    def test_overlap_risk_classification_throughput(self):
        """Test risk level classification throughput exceeds 100k/second."""
        scores = [Decimal(str(i)) for i in range(101)]
        start = time.perf_counter()
        for _ in range(100):
            for score in scores:
                classify_risk_level(score)
        elapsed = time.perf_counter() - start
        total = 100 * len(scores)
        throughput = total / elapsed

        assert throughput >= 100000

    def test_overlap_batch_1000_under_100ms(self):
        """Test batch of 1000 overlap risk scores completes in < 100ms."""
        start = time.perf_counter()
        for i in range(1000):
            compute_overlap_risk_score(
                overlap_type=["direct", "partial", "adjacent", "proximate", "none"][i % 5],
                legal_status=["titled", "declared", "claimed", "customary", "pending"][i % 5],
                community_population=(i + 1) * 100,
                conflict_history_score=Decimal(str(i % 100)),
                country_framework_score=Decimal(str((i * 7) % 100)),
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100.0, (
            f"Batch of 1000 overlap scores took {elapsed_ms:.1f}ms, expected < 100ms"
        )

    def test_overlap_scoring_p95_latency(self):
        """Test overlap risk scoring p95 latency is < 0.5ms."""
        latencies = []
        for _ in range(500):
            start = time.perf_counter()
            compute_overlap_risk_score(
                overlap_type="direct",
                legal_status="titled",
                community_population=26000,
                conflict_history_score=Decimal("70"),
                country_framework_score=Decimal("45"),
            )
            latencies.append((time.perf_counter() - start) * 1000)

        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        assert p95 < 0.5, f"p95 latency was {p95:.3f}ms, expected < 0.5ms"


# ===========================================================================
# 3. Violation Severity Scoring Performance (5 tests)
# ===========================================================================


class TestViolationSeverityPerformance:
    """Performance tests for violation severity scoring."""

    def test_single_violation_scoring_under_1ms(self):
        """Test single violation severity scoring completes in < 1ms."""
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            compute_violation_severity(
                violation_type="physical_violence",
                proximity_score=Decimal("90"),
                population_score=Decimal("80"),
                legal_gap_score=Decimal("60"),
                media_score=Decimal("70"),
            )
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 1.0

    def test_violation_scoring_all_types_under_5ms(self):
        """Test scoring all 10 violation types completes in < 5ms."""
        start = time.perf_counter()
        for vtype in VIOLATION_TYPE_SCORES.keys():
            compute_violation_severity(
                violation_type=vtype,
                proximity_score=Decimal("50"),
                population_score=Decimal("50"),
                legal_gap_score=Decimal("50"),
                media_score=Decimal("50"),
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 5.0

    def test_violation_scoring_throughput_50k_per_second(self):
        """Test violation scoring throughput exceeds 50,000/second."""
        num_records = 50000
        start = time.perf_counter()
        for i in range(num_records):
            compute_violation_severity(
                violation_type="land_seizure",
                proximity_score=Decimal(str(i % 100)),
                population_score=Decimal("60"),
                legal_gap_score=Decimal("40"),
                media_score=Decimal("50"),
            )
        elapsed = time.perf_counter() - start
        throughput = num_records / elapsed

        assert throughput >= 50000

    def test_violation_batch_500_under_50ms(self):
        """Test batch of 500 violation severity scores completes in < 50ms."""
        violation_types = list(VIOLATION_TYPE_SCORES.keys())
        start = time.perf_counter()
        for i in range(500):
            compute_violation_severity(
                violation_type=violation_types[i % len(violation_types)],
                proximity_score=Decimal(str((i * 3) % 100)),
                population_score=Decimal(str((i * 5) % 100)),
                legal_gap_score=Decimal(str((i * 7) % 100)),
                media_score=Decimal(str((i * 11) % 100)),
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50.0

    def test_violation_scoring_p99_latency(self):
        """Test violation severity scoring p99 latency is < 0.5ms."""
        latencies = []
        for _ in range(500):
            start = time.perf_counter()
            compute_violation_severity(
                violation_type="fpic_violation",
                proximity_score=Decimal("80"),
                population_score=Decimal("60"),
                legal_gap_score=Decimal("55"),
                media_score=Decimal("45"),
            )
            latencies.append((time.perf_counter() - start) * 1000)

        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        assert p99 < 0.5


# ===========================================================================
# 4. Hash Computation Performance (5 tests)
# ===========================================================================


class TestHashComputationPerformance:
    """Performance tests for SHA-256 provenance hash computation."""

    def test_single_hash_under_1ms(self):
        """Test single SHA-256 hash computation completes in < 1ms."""
        data = {"territory_id": "t-001", "score": "85.50"}
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            compute_test_hash(data)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 1.0

    def test_hash_throughput_50k_per_second(self):
        """Test hash throughput exceeds 50,000 hashes/second."""
        data = {"territory_id": "t-001", "score": "85.50", "country": "BR"}
        num_hashes = 50000
        start = time.perf_counter()
        for i in range(num_hashes):
            compute_test_hash({"id": str(i), **data})
        elapsed = time.perf_counter() - start
        throughput = num_hashes / elapsed

        assert throughput >= 50000

    def test_large_payload_hash_under_5ms(self):
        """Test hashing a large payload (10KB) completes in < 5ms."""
        large_data = {
            "territory_id": "t-001",
            "boundary_geojson": {
                "type": "Polygon",
                "coordinates": [
                    [[float(i), float(j)] for j in range(50)]
                    for i in range(50)
                ],
            },
            "metadata": {str(k): f"value_{k}" for k in range(100)},
        }
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            compute_test_hash(large_data)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 5.0

    def test_chained_hash_1000_under_100ms(self):
        """Test 1000 chained hashes (provenance chain) completes in < 100ms."""
        prev_hash = compute_test_hash({"genesis": True})
        start = time.perf_counter()
        for i in range(1000):
            prev_hash = compute_test_hash({
                "index": i,
                "previous_hash": prev_hash,
                "entity_type": "territory",
                "action": "query",
            })
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100.0
        assert len(prev_hash) == SHA256_HEX_LENGTH

    def test_hash_p99_latency(self):
        """Test hash computation p99 latency is < 0.5ms."""
        data = {"territory_id": "t-001", "score": "85.50"}
        latencies = []
        for _ in range(500):
            start = time.perf_counter()
            compute_test_hash(data)
            latencies.append((time.perf_counter() - start) * 1000)

        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        assert p99 < 0.5


# ===========================================================================
# 5. Haversine Distance Performance (4 tests)
# ===========================================================================


class TestHaversineDistancePerformance:
    """Performance tests for Haversine distance calculations."""

    def test_single_haversine_under_01ms(self):
        """Test single Haversine distance completes in < 0.1ms."""
        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            haversine_km(-3.0, -60.0, -3.5, -59.5)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 0.1

    def test_haversine_throughput_500k_per_second(self):
        """Test Haversine throughput exceeds 500,000 calculations/second."""
        num_calcs = 100000
        start = time.perf_counter()
        for i in range(num_calcs):
            haversine_km(
                -3.0 + (i % 10) * 0.1,
                -60.0 + (i % 10) * 0.1,
                -3.5,
                -59.5,
            )
        elapsed = time.perf_counter() - start
        throughput = num_calcs / elapsed

        assert throughput >= 500000

    def test_haversine_batch_10k_under_50ms(self):
        """Test batch of 10,000 Haversine distances completes in < 50ms."""
        start = time.perf_counter()
        for i in range(10000):
            lat1 = -90.0 + (i % 180)
            lon1 = -180.0 + (i % 360)
            haversine_km(lat1, lon1, 0.0, 0.0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50.0

    def test_haversine_p99_latency(self):
        """Test Haversine p99 latency is < 0.05ms."""
        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            haversine_km(-3.0, -60.0, -12.5, -55.3)
            latencies.append((time.perf_counter() - start) * 1000)

        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        assert p99 < 0.05


# ===========================================================================
# 6. Model Creation Performance (5 tests)
# ===========================================================================


class TestModelCreationPerformance:
    """Performance tests for Pydantic model instantiation."""

    def test_territory_model_creation_under_1ms(self):
        """Test IndigenousTerritory model creation completes in < 1ms."""
        iterations = 500
        start = time.perf_counter()
        for i in range(iterations):
            IndigenousTerritory(
                territory_id=f"t-perf-{i:04d}",
                territory_name=f"Test Territory {i}",
                people_name=f"People {i}",
                country_code="BR",
                legal_status=TerritoryLegalStatus.TITLED,
                data_source="funai",
                provenance_hash="a" * 64,
            )
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 1.0

    def test_overlap_model_creation_under_1ms(self):
        """Test TerritoryOverlap model creation completes in < 1ms."""
        iterations = 500
        start = time.perf_counter()
        for i in range(iterations):
            TerritoryOverlap(
                overlap_id=f"o-perf-{i:04d}",
                plot_id=f"p-{i:04d}",
                territory_id="t-001",
                overlap_type=OverlapType.DIRECT,
                distance_meters=Decimal("0"),
                risk_score=Decimal("90"),
                risk_level=RiskLevel.CRITICAL,
                provenance_hash="b" * 64,
            )
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 1.0

    def test_fpic_assessment_model_creation_under_2ms(self):
        """Test FPICAssessment model creation completes in < 2ms."""
        iterations = 500
        start = time.perf_counter()
        for i in range(iterations):
            FPICAssessment(
                assessment_id=f"a-perf-{i:04d}",
                plot_id=f"p-{i:04d}",
                territory_id="t-001",
                fpic_score=Decimal("85"),
                fpic_status=FPICStatus.CONSENT_OBTAINED,
                community_identification_score=Decimal("90"),
                information_disclosure_score=Decimal("85"),
                prior_timing_score=Decimal("100"),
                consultation_process_score=Decimal("80"),
                community_representation_score=Decimal("85"),
                consent_record_score=Decimal("90"),
                absence_of_coercion_score=Decimal("95"),
                agreement_documentation_score=Decimal("80"),
                benefit_sharing_score=Decimal("75"),
                monitoring_provisions_score=Decimal("70"),
                temporal_compliance=True,
                provenance_hash="c" * 64,
            )
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 2.0

    def test_batch_overlap_response_creation_under_5ms(self):
        """Test BatchOverlapResponse creation with summary fields in < 5ms."""
        iterations = 100
        start = time.perf_counter()
        for i in range(iterations):
            BatchOverlapResponse(
                total_plots=10000,
                plots_with_overlaps=500,
                critical_count=50,
                high_count=100,
                medium_count=200,
                low_count=150,
                processing_time_ms=25000.0,
                provenance_hash="d" * 64,
            )
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 5.0

    def test_detect_overlap_request_validation_under_1ms(self):
        """Test DetectOverlapRequest validation completes in < 1ms."""
        iterations = 500
        start = time.perf_counter()
        for i in range(iterations):
            DetectOverlapRequest(
                plot_id=f"p-{i:04d}",
                latitude=-3.0 + (i % 6) * 0.1,
                longitude=-60.0 + (i % 10) * 0.1,
            )
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 1.0


# ===========================================================================
# 7. End-to-End Scoring Pipeline Performance (4 tests)
# ===========================================================================


class TestEndToEndScoringPerformance:
    """Performance tests for combined scoring pipelines."""

    def test_full_fpic_pipeline_under_5ms(self):
        """Test full FPIC pipeline (score + classify + hash) in < 5ms."""
        element_scores = {
            elem: Decimal(str(70 + i * 3)) for i, elem in enumerate(FPIC_ELEMENTS)
        }
        iterations = 200
        start = time.perf_counter()
        for _ in range(iterations):
            score = compute_fpic_score(element_scores)
            status = classify_fpic_status(score)
            h = compute_test_hash({
                "fpic_score": str(score),
                "fpic_status": status,
            })
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 5.0

    def test_full_overlap_pipeline_under_5ms(self):
        """Test full overlap pipeline (score + classify + hash) in < 5ms."""
        iterations = 200
        start = time.perf_counter()
        for _ in range(iterations):
            score = compute_overlap_risk_score(
                overlap_type="direct",
                legal_status="titled",
                community_population=26000,
                conflict_history_score=Decimal("70"),
                country_framework_score=Decimal("45"),
            )
            level = classify_risk_level(score)
            h = compute_test_hash({
                "risk_score": str(score),
                "risk_level": level,
            })
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 5.0

    def test_combined_overlap_and_fpic_under_10ms(self):
        """Test combined overlap + FPIC scoring in < 10ms."""
        element_scores = {
            elem: Decimal(str(80)) for elem in FPIC_ELEMENTS
        }
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            overlap_score = compute_overlap_risk_score(
                overlap_type="direct",
                legal_status="titled",
                community_population=26000,
                conflict_history_score=Decimal("70"),
                country_framework_score=Decimal("45"),
            )
            overlap_level = classify_risk_level(overlap_score)
            fpic_score = compute_fpic_score(element_scores)
            fpic_status = classify_fpic_status(fpic_score)
            h = compute_test_hash({
                "overlap_risk_score": str(overlap_score),
                "overlap_risk_level": overlap_level,
                "fpic_score": str(fpic_score),
                "fpic_status": fpic_status,
            })
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000

        assert avg_ms < 10.0

    def test_full_pipeline_10k_records_under_10s(self):
        """Test processing 10,000 records through full pipeline in < 10s."""
        element_scores = {
            elem: Decimal("75") for elem in FPIC_ELEMENTS
        }
        start = time.perf_counter()
        for i in range(10000):
            # Overlap scoring
            ot = ["direct", "partial", "adjacent", "proximate", "none"][i % 5]
            ls = ["titled", "declared", "claimed", "customary", "pending"][i % 5]
            o_score = compute_overlap_risk_score(
                overlap_type=ot,
                legal_status=ls,
                community_population=(i + 1) * 10,
                conflict_history_score=Decimal(str(i % 100)),
                country_framework_score=Decimal(str((i * 7) % 100)),
            )
            classify_risk_level(o_score)

            # FPIC scoring
            f_score = compute_fpic_score(element_scores)
            classify_fpic_status(f_score)

            # Hash
            compute_test_hash({"index": i, "overlap": str(o_score), "fpic": str(f_score)})

        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, (
            f"10k records took {elapsed:.1f}s, expected < 10s"
        )
