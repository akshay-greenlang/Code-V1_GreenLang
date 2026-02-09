# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-DATA-011 Duplicate Detection Agent tests.

Provides reusable test fixtures for configuration, sample data, mock objects,
and pre-computed results used across all test modules in Batch 1.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Autouse fixture: clean GL_DD_ environment variables before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_dd_env(monkeypatch):
    """Remove all GL_DD_ environment variables before each test.

    This prevents leakage of environment state between tests that set
    GL_DD_ prefixed variables via monkeypatch or os.environ.
    """
    keys_to_remove = [k for k in os.environ if k.startswith("GL_DD_")]
    for key in keys_to_remove:
        monkeypatch.delenv(key, raising=False)

    # Also reset the singleton config so each test starts fresh
    from greenlang.duplicate_detector.config import reset_config
    reset_config()

    yield

    # Post-test cleanup: reset singleton again
    reset_config()


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Create a DuplicateDetectorConfig with test defaults."""
    from greenlang.duplicate_detector.config import DuplicateDetectorConfig

    return DuplicateDetectorConfig(
        database_url="postgresql://test:test@localhost:5432/testdb",
        redis_url="redis://localhost:6379/0",
        s3_bucket="test-dedup-bucket",
        max_records_per_job=100_000,
        default_batch_size=5_000,
        fingerprint_algorithm="sha256",
        fingerprint_normalize=True,
        blocking_strategy="sorted_neighborhood",
        blocking_window_size=10,
        blocking_key_size=3,
        canopy_tight_threshold=0.8,
        canopy_loose_threshold=0.4,
        default_similarity_algorithm="jaro_winkler",
        ngram_size=3,
        match_threshold=0.85,
        possible_threshold=0.65,
        non_match_threshold=0.40,
        use_fellegi_sunter=False,
        cluster_algorithm="union_find",
        cluster_min_quality=0.5,
        default_merge_strategy="keep_most_complete",
        merge_conflict_resolution="most_complete",
        pipeline_checkpoint_interval=1000,
        pipeline_timeout_seconds=3600,
        max_comparisons_per_block=50_000,
        cache_ttl_seconds=3600,
        cache_enabled=True,
        pool_min_size=2,
        pool_max_size=10,
        log_level="INFO",
        enable_metrics=True,
        max_field_weights=50,
        max_rules_per_job=100,
        comparison_sample_rate=1.0,
    )


# ---------------------------------------------------------------------------
# Sample records
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    """Return 20 sample records with realistic sustainability-domain data.

    Each record has: id, name, email, phone, address, city, state, zip,
    amount, date.
    """
    return [
        {
            "id": "rec-001",
            "name": "Alice Johnson",
            "email": "alice.johnson@greenco.com",
            "phone": "+1-555-0101",
            "address": "123 Oak Street",
            "city": "Portland",
            "state": "OR",
            "zip": "97201",
            "amount": 15200.50,
            "date": "2025-06-15",
        },
        {
            "id": "rec-002",
            "name": "Bob Martinez",
            "email": "bob.martinez@ecoworks.io",
            "phone": "+1-555-0102",
            "address": "456 Elm Avenue",
            "city": "Seattle",
            "state": "WA",
            "zip": "98101",
            "amount": 22300.75,
            "date": "2025-07-20",
        },
        {
            "id": "rec-003",
            "name": "Clara Lee",
            "email": "clara.lee@sustain.org",
            "phone": "+1-555-0103",
            "address": "789 Pine Road",
            "city": "San Francisco",
            "state": "CA",
            "zip": "94102",
            "amount": 8750.00,
            "date": "2025-08-01",
        },
        {
            "id": "rec-004",
            "name": "David Chen",
            "email": "david.chen@carbonzero.net",
            "phone": "+1-555-0104",
            "address": "321 Birch Lane",
            "city": "Denver",
            "state": "CO",
            "zip": "80201",
            "amount": 31400.25,
            "date": "2025-08-15",
        },
        {
            "id": "rec-005",
            "name": "Emma Williams",
            "email": "emma.w@greenfuture.com",
            "phone": "+1-555-0105",
            "address": "654 Maple Drive",
            "city": "Austin",
            "state": "TX",
            "zip": "73301",
            "amount": 19800.60,
            "date": "2025-09-10",
        },
        {
            "id": "rec-006",
            "name": "Frank Garcia",
            "email": "frank.garcia@ecocorp.com",
            "phone": "+1-555-0106",
            "address": "987 Cedar Court",
            "city": "Chicago",
            "state": "IL",
            "zip": "60601",
            "amount": 45100.00,
            "date": "2025-09-25",
        },
        {
            "id": "rec-007",
            "name": "Grace Kim",
            "email": "grace.kim@renewco.com",
            "phone": "+1-555-0107",
            "address": "147 Walnut Way",
            "city": "Boston",
            "state": "MA",
            "zip": "02101",
            "amount": 12600.30,
            "date": "2025-10-01",
        },
        {
            "id": "rec-008",
            "name": "Henry Patel",
            "email": "henry.patel@cleantech.io",
            "phone": "+1-555-0108",
            "address": "258 Spruce Boulevard",
            "city": "Minneapolis",
            "state": "MN",
            "zip": "55401",
            "amount": 28900.80,
            "date": "2025-10-15",
        },
        {
            "id": "rec-009",
            "name": "Isla Brown",
            "email": "isla.brown@greentrans.com",
            "phone": "+1-555-0109",
            "address": "369 Ash Circle",
            "city": "Miami",
            "state": "FL",
            "zip": "33101",
            "amount": 7250.40,
            "date": "2025-11-01",
        },
        {
            "id": "rec-010",
            "name": "Jack Thompson",
            "email": "jack.t@solarcity.com",
            "phone": "+1-555-0110",
            "address": "480 Cypress Lane",
            "city": "Phoenix",
            "state": "AZ",
            "zip": "85001",
            "amount": 33500.00,
            "date": "2025-11-15",
        },
        {
            "id": "rec-011",
            "name": "Karen Davis",
            "email": "karen.d@windpower.org",
            "phone": "+1-555-0111",
            "address": "591 Willow Street",
            "city": "Atlanta",
            "state": "GA",
            "zip": "30301",
            "amount": 16700.90,
            "date": "2025-12-01",
        },
        {
            "id": "rec-012",
            "name": "Liam Wilson",
            "email": "liam.w@bioenergy.com",
            "phone": "+1-555-0112",
            "address": "602 Poplar Avenue",
            "city": "Nashville",
            "state": "TN",
            "zip": "37201",
            "amount": 21300.55,
            "date": "2025-12-15",
        },
        {
            "id": "rec-013",
            "name": "Mia Rodriguez",
            "email": "mia.r@hydrogreen.com",
            "phone": "+1-555-0113",
            "address": "713 Chestnut Road",
            "city": "Raleigh",
            "state": "NC",
            "zip": "27601",
            "amount": 9400.20,
            "date": "2026-01-01",
        },
        {
            "id": "rec-014",
            "name": "Noah Anderson",
            "email": "noah.a@carboncap.com",
            "phone": "+1-555-0114",
            "address": "824 Hickory Place",
            "city": "Columbus",
            "state": "OH",
            "zip": "43201",
            "amount": 42700.30,
            "date": "2026-01-10",
        },
        {
            "id": "rec-015",
            "name": "Olivia Taylor",
            "email": "olivia.t@ecofund.org",
            "phone": "+1-555-0115",
            "address": "935 Magnolia Drive",
            "city": "Charlotte",
            "state": "NC",
            "zip": "28201",
            "amount": 18500.75,
            "date": "2026-01-20",
        },
        {
            "id": "rec-016",
            "name": "Patrick Moore",
            "email": "patrick.m@greengrid.com",
            "phone": "+1-555-0116",
            "address": "1046 Redwood Terrace",
            "city": "Salt Lake City",
            "state": "UT",
            "zip": "84101",
            "amount": 27800.10,
            "date": "2026-01-25",
        },
        {
            "id": "rec-017",
            "name": "Quinn Harris",
            "email": "quinn.h@sustainlogic.com",
            "phone": "+1-555-0117",
            "address": "1157 Sequoia Path",
            "city": "Pittsburgh",
            "state": "PA",
            "zip": "15201",
            "amount": 11200.45,
            "date": "2026-01-28",
        },
        {
            "id": "rec-018",
            "name": "Rachel Clark",
            "email": "rachel.c@cleanair.org",
            "phone": "+1-555-0118",
            "address": "1268 Juniper Court",
            "city": "Detroit",
            "state": "MI",
            "zip": "48201",
            "amount": 35600.90,
            "date": "2026-02-01",
        },
        {
            "id": "rec-019",
            "name": "Samuel Lewis",
            "email": "samuel.l@netzero.com",
            "phone": "+1-555-0119",
            "address": "1379 Dogwood Lane",
            "city": "Kansas City",
            "state": "MO",
            "zip": "64101",
            "amount": 14100.35,
            "date": "2026-02-05",
        },
        {
            "id": "rec-020",
            "name": "Tara White",
            "email": "tara.w@greenvolt.com",
            "phone": "+1-555-0120",
            "address": "1490 Holly Street",
            "city": "San Diego",
            "state": "CA",
            "zip": "92101",
            "amount": 20900.65,
            "date": "2026-02-08",
        },
    ]


@pytest.fixture
def sample_records_with_duplicates() -> List[Dict[str, Any]]:
    """Return 15 records where ~5 are near-duplicates of others.

    Near-duplicates have slight variations in name spelling, address
    formatting, phone formatting, or case differences.
    """
    return [
        # --- Original record group 1 ---
        {
            "id": "dup-001",
            "name": "Alice Johnson",
            "email": "alice.johnson@greenco.com",
            "phone": "+1-555-0101",
            "address": "123 Oak Street",
            "city": "Portland",
            "state": "OR",
            "zip": "97201",
            "amount": 15200.50,
            "date": "2025-06-15",
        },
        # Near-duplicate of dup-001: name case, address abbreviation
        {
            "id": "dup-002",
            "name": "alice johnson",
            "email": "alice.johnson@greenco.com",
            "phone": "15550101",
            "address": "123 Oak St",
            "city": "Portland",
            "state": "OR",
            "zip": "97201",
            "amount": 15200.50,
            "date": "2025-06-15",
        },
        # --- Original record group 2 ---
        {
            "id": "dup-003",
            "name": "Bob Martinez",
            "email": "bob.martinez@ecoworks.io",
            "phone": "+1-555-0102",
            "address": "456 Elm Avenue",
            "city": "Seattle",
            "state": "WA",
            "zip": "98101",
            "amount": 22300.75,
            "date": "2025-07-20",
        },
        # Near-duplicate of dup-003: typo in name, phone format
        {
            "id": "dup-004",
            "name": "Bob Martinex",
            "email": "bob.martinez@ecoworks.io",
            "phone": "+1 555 0102",
            "address": "456 Elm Avenue",
            "city": "Seattle",
            "state": "WA",
            "zip": "98101",
            "amount": 22300.75,
            "date": "2025-07-20",
        },
        # --- Original record group 3 ---
        {
            "id": "dup-005",
            "name": "Clara Lee",
            "email": "clara.lee@sustain.org",
            "phone": "+1-555-0103",
            "address": "789 Pine Road",
            "city": "San Francisco",
            "state": "CA",
            "zip": "94102",
            "amount": 8750.00,
            "date": "2025-08-01",
        },
        # Near-duplicate of dup-005: extra middle initial, address variation
        {
            "id": "dup-006",
            "name": "Clara M. Lee",
            "email": "clara.lee@sustain.org",
            "phone": "+1-555-0103",
            "address": "789 Pine Rd",
            "city": "San Francisco",
            "state": "CA",
            "zip": "94102",
            "amount": 8750.00,
            "date": "2025-08-01",
        },
        # --- Unique records ---
        {
            "id": "dup-007",
            "name": "David Chen",
            "email": "david.chen@carbonzero.net",
            "phone": "+1-555-0104",
            "address": "321 Birch Lane",
            "city": "Denver",
            "state": "CO",
            "zip": "80201",
            "amount": 31400.25,
            "date": "2025-08-15",
        },
        {
            "id": "dup-008",
            "name": "Emma Williams",
            "email": "emma.w@greenfuture.com",
            "phone": "+1-555-0105",
            "address": "654 Maple Drive",
            "city": "Austin",
            "state": "TX",
            "zip": "73301",
            "amount": 19800.60,
            "date": "2025-09-10",
        },
        {
            "id": "dup-009",
            "name": "Frank Garcia",
            "email": "frank.garcia@ecocorp.com",
            "phone": "+1-555-0106",
            "address": "987 Cedar Court",
            "city": "Chicago",
            "state": "IL",
            "zip": "60601",
            "amount": 45100.00,
            "date": "2025-09-25",
        },
        {
            "id": "dup-010",
            "name": "Grace Kim",
            "email": "grace.kim@renewco.com",
            "phone": "+1-555-0107",
            "address": "147 Walnut Way",
            "city": "Boston",
            "state": "MA",
            "zip": "02101",
            "amount": 12600.30,
            "date": "2025-10-01",
        },
        # --- Original record group 4 ---
        {
            "id": "dup-011",
            "name": "Henry Patel",
            "email": "henry.patel@cleantech.io",
            "phone": "+1-555-0108",
            "address": "258 Spruce Boulevard",
            "city": "Minneapolis",
            "state": "MN",
            "zip": "55401",
            "amount": 28900.80,
            "date": "2025-10-15",
        },
        # Near-duplicate of dup-011: address abbreviation, phone variation
        {
            "id": "dup-012",
            "name": "Henry Patel",
            "email": "henry.patel@cleantech.io",
            "phone": "(555) 010-8",
            "address": "258 Spruce Blvd",
            "city": "Minneapolis",
            "state": "MN",
            "zip": "55401",
            "amount": 28900.80,
            "date": "2025-10-15",
        },
        # --- Original record group 5 ---
        {
            "id": "dup-013",
            "name": "Isla Brown",
            "email": "isla.brown@greentrans.com",
            "phone": "+1-555-0109",
            "address": "369 Ash Circle",
            "city": "Miami",
            "state": "FL",
            "zip": "33101",
            "amount": 7250.40,
            "date": "2025-11-01",
        },
        # Near-duplicate of dup-013: email domain variation, zip typo
        {
            "id": "dup-014",
            "name": "Isla Brown",
            "email": "isla.b@greentrans.com",
            "phone": "+1-555-0109",
            "address": "369 Ash Circle",
            "city": "Miami",
            "state": "FL",
            "zip": "33102",
            "amount": 7250.40,
            "date": "2025-11-01",
        },
        # --- One more unique ---
        {
            "id": "dup-015",
            "name": "Jack Thompson",
            "email": "jack.t@solarcity.com",
            "phone": "+1-555-0110",
            "address": "480 Cypress Lane",
            "city": "Phoenix",
            "state": "AZ",
            "zip": "85001",
            "amount": 33500.00,
            "date": "2025-11-15",
        },
    ]


# ---------------------------------------------------------------------------
# Field comparison config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_field_configs():
    """Return a list of FieldComparisonConfig for common dedup fields."""
    from greenlang.duplicate_detector.models import (
        FieldComparisonConfig,
        FieldType,
        SimilarityAlgorithm,
    )

    return [
        FieldComparisonConfig(
            field_name="name",
            algorithm=SimilarityAlgorithm.JARO_WINKLER,
            weight=2.5,
            field_type=FieldType.STRING,
            case_sensitive=False,
            strip_whitespace=True,
            phonetic_encode=False,
        ),
        FieldComparisonConfig(
            field_name="email",
            algorithm=SimilarityAlgorithm.EXACT,
            weight=2.0,
            field_type=FieldType.STRING,
            case_sensitive=False,
            strip_whitespace=True,
        ),
        FieldComparisonConfig(
            field_name="phone",
            algorithm=SimilarityAlgorithm.LEVENSHTEIN,
            weight=1.5,
            field_type=FieldType.STRING,
            case_sensitive=False,
            strip_whitespace=True,
        ),
        FieldComparisonConfig(
            field_name="address",
            algorithm=SimilarityAlgorithm.NGRAM,
            weight=2.0,
            field_type=FieldType.STRING,
            case_sensitive=False,
            strip_whitespace=True,
        ),
        FieldComparisonConfig(
            field_name="city",
            algorithm=SimilarityAlgorithm.JARO_WINKLER,
            weight=1.0,
            field_type=FieldType.STRING,
        ),
        FieldComparisonConfig(
            field_name="state",
            algorithm=SimilarityAlgorithm.EXACT,
            weight=0.5,
            field_type=FieldType.CATEGORICAL,
        ),
        FieldComparisonConfig(
            field_name="zip",
            algorithm=SimilarityAlgorithm.EXACT,
            weight=1.0,
            field_type=FieldType.STRING,
        ),
        FieldComparisonConfig(
            field_name="amount",
            algorithm=SimilarityAlgorithm.NUMERIC,
            weight=1.0,
            field_type=FieldType.NUMERIC,
        ),
        FieldComparisonConfig(
            field_name="date",
            algorithm=SimilarityAlgorithm.DATE,
            weight=0.5,
            field_type=FieldType.DATE,
        ),
    ]


# ---------------------------------------------------------------------------
# DedupRule fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dedup_rule(sample_field_configs):
    """Return a complete DedupRule with field configs and thresholds."""
    from greenlang.duplicate_detector.models import (
        BlockingStrategy,
        DedupRule,
        MergeStrategy,
    )

    return DedupRule(
        name="sustainability-supplier-dedup",
        description="Deduplicate sustainability supplier records using weighted field comparison",
        field_configs=sample_field_configs,
        match_threshold=0.85,
        possible_threshold=0.65,
        blocking_strategy=BlockingStrategy.SORTED_NEIGHBORHOOD,
        blocking_fields=["state", "zip"],
        merge_strategy=MergeStrategy.KEEP_MOST_COMPLETE,
        active=True,
    )


# ---------------------------------------------------------------------------
# Pre-computed similarity results
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_similarity_results():
    """Return pre-computed SimilarityResult objects for testing classifier."""
    from greenlang.duplicate_detector.models import (
        SimilarityAlgorithm,
        SimilarityResult,
    )

    return [
        # High match: almost identical
        SimilarityResult(
            record_a_id="rec-001",
            record_b_id="rec-002",
            field_scores={"name": 0.95, "email": 1.0, "phone": 0.90, "address": 0.88},
            overall_score=0.93,
            algorithm_used=SimilarityAlgorithm.JARO_WINKLER,
            comparison_time_ms=1.2,
        ),
        # Possible match: moderate similarity
        SimilarityResult(
            record_a_id="rec-003",
            record_b_id="rec-004",
            field_scores={"name": 0.72, "email": 0.0, "phone": 0.65, "address": 0.80},
            overall_score=0.70,
            algorithm_used=SimilarityAlgorithm.JARO_WINKLER,
            comparison_time_ms=1.5,
        ),
        # Non match: low similarity
        SimilarityResult(
            record_a_id="rec-005",
            record_b_id="rec-006",
            field_scores={"name": 0.20, "email": 0.0, "phone": 0.10, "address": 0.15},
            overall_score=0.15,
            algorithm_used=SimilarityAlgorithm.JARO_WINKLER,
            comparison_time_ms=0.8,
        ),
        # Borderline match
        SimilarityResult(
            record_a_id="rec-007",
            record_b_id="rec-008",
            field_scores={"name": 0.85, "email": 0.90, "phone": 0.82, "address": 0.78},
            overall_score=0.85,
            algorithm_used=SimilarityAlgorithm.JARO_WINKLER,
            comparison_time_ms=1.1,
        ),
        # Zero similarity
        SimilarityResult(
            record_a_id="rec-009",
            record_b_id="rec-010",
            field_scores={"name": 0.0, "email": 0.0, "phone": 0.0, "address": 0.0},
            overall_score=0.0,
            algorithm_used=SimilarityAlgorithm.EXACT,
            comparison_time_ms=0.5,
        ),
    ]


# ---------------------------------------------------------------------------
# Pre-computed match results
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_match_results():
    """Return pre-computed MatchResult objects for testing clustering."""
    from greenlang.duplicate_detector.models import (
        MatchClassification,
        MatchResult,
    )

    return [
        MatchResult(
            record_a_id="rec-001",
            record_b_id="rec-002",
            classification=MatchClassification.MATCH,
            confidence=0.95,
            field_scores={"name": 0.95, "email": 1.0, "address": 0.88},
            overall_score=0.93,
            decision_reason="Above match threshold 0.85",
        ),
        MatchResult(
            record_a_id="rec-001",
            record_b_id="rec-003",
            classification=MatchClassification.MATCH,
            confidence=0.88,
            field_scores={"name": 0.90, "email": 0.85, "address": 0.82},
            overall_score=0.87,
            decision_reason="Above match threshold 0.85",
        ),
        MatchResult(
            record_a_id="rec-004",
            record_b_id="rec-005",
            classification=MatchClassification.MATCH,
            confidence=0.92,
            field_scores={"name": 0.92, "email": 0.95, "address": 0.90},
            overall_score=0.92,
            decision_reason="Above match threshold 0.85",
        ),
        MatchResult(
            record_a_id="rec-006",
            record_b_id="rec-007",
            classification=MatchClassification.POSSIBLE,
            confidence=0.72,
            field_scores={"name": 0.70, "email": 0.0, "address": 0.75},
            overall_score=0.70,
            decision_reason="Between possible and match thresholds",
        ),
        MatchResult(
            record_a_id="rec-008",
            record_b_id="rec-009",
            classification=MatchClassification.NON_MATCH,
            confidence=0.15,
            field_scores={"name": 0.10, "email": 0.0, "address": 0.20},
            overall_score=0.15,
            decision_reason="Below possible threshold 0.65",
        ),
    ]


# ---------------------------------------------------------------------------
# Pre-computed clusters
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_clusters():
    """Return pre-computed DuplicateCluster objects for testing merge."""
    from greenlang.duplicate_detector.models import DuplicateCluster

    return [
        DuplicateCluster(
            cluster_id="cluster-001",
            member_record_ids=["rec-001", "rec-002", "rec-003"],
            representative_id="rec-001",
            cluster_quality=0.91,
            density=0.85,
            diameter=0.12,
            member_count=3,
        ),
        DuplicateCluster(
            cluster_id="cluster-002",
            member_record_ids=["rec-004", "rec-005"],
            representative_id="rec-004",
            cluster_quality=0.92,
            density=1.0,
            diameter=0.08,
            member_count=2,
        ),
        DuplicateCluster(
            cluster_id="cluster-003",
            member_record_ids=["rec-006"],
            representative_id="rec-006",
            cluster_quality=1.0,
            density=1.0,
            diameter=0.0,
            member_count=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Mock prometheus
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_prometheus():
    """Mock prometheus_client Counter, Histogram, and Gauge classes."""
    mock_counter = MagicMock()
    mock_counter.labels.return_value.inc = MagicMock()

    mock_histogram = MagicMock()
    mock_histogram.labels.return_value.observe = MagicMock()

    mock_gauge = MagicMock()
    mock_gauge.set = MagicMock()

    return {
        "Counter": mock_counter,
        "Histogram": mock_histogram,
        "Gauge": mock_gauge,
    }
