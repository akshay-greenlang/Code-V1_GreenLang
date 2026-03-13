# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-021 Indigenous Rights Checker test suite.

Provides reusable fixtures for configuration objects, engine instances,
territory samples, FPIC assessment samples, overlap detection samples,
community consultation samples, violation alert samples, provenance
tracking helpers, mock PostGIS functions, mock Redis cache, mock
authentication, and shared constants used across all test modules.

Fixture count: 50+ fixtures
Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 Indigenous Rights Checker (GL-EUDR-IRC-021)
"""

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.indigenous_rights_checker.config import (
    IndigenousRightsCheckerConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    TerritoryLegalStatus,
    FPICStatus,
    OverlapType,
    ViolationType,
    FPICWorkflowStage,
    ConsultationStage,
    GrievanceStatus,
    AgreementStatus,
    ConfidenceLevel,
    RiskLevel,
    AlertSeverity,
    DataSource,
    ReportType,
    ReportFormat,
    CommunityRecognitionStatus,
    CountryRiskLevel,
    ViolationAlertStatus,
    IndigenousTerritory,
    FPICAssessment,
    TerritoryOverlap,
    IndigenousCommunity,
    ConsultationRecord,
    GrievanceRecord,
    BenefitSharingAgreement,
    FPICWorkflow,
    WorkflowTransition,
    ViolationAlert,
    ComplianceReport,
    CountryIndigenousRightsScore,
    AuditLogEntry,
    DetectOverlapRequest,
    BatchOverlapRequest,
    VerifyFPICRequest,
    GenerateReportRequest,
    EUDR_COMMODITIES,
    SUPPORTED_REGIONS,
    MAX_BATCH_SIZE,
    MAX_FPIC_SCORE,
    MIN_FPIC_SCORE,
    MAX_RISK_SCORE,
    MIN_RISK_SCORE,
)


# ---------------------------------------------------------------------------
# Deterministic UUID helper
# ---------------------------------------------------------------------------


class DeterministicUUID:
    """Generate sequential identifiers for deterministic testing."""

    def __init__(self, prefix: str = "test"):
        self._counter = 0
        self._prefix = prefix

    def next(self) -> str:
        self._counter += 1
        return f"{self._prefix}-{self._counter:08d}"

    def reset(self):
        self._counter = 0


# ---------------------------------------------------------------------------
# Provenance hash helper
# ---------------------------------------------------------------------------


def compute_test_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for test assertions."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Haversine distance helper
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance between two points in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ---------------------------------------------------------------------------
# Shared Constants
# ---------------------------------------------------------------------------

SHA256_HEX_LENGTH = 64

EUDR_CUTOFF_DATE_STR = "2020-12-31"
EUDR_CUTOFF_DATE_OBJ = date(2020, 12, 31)

ALL_COMMODITIES = [
    "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
]

# Overlap type ordering (severity descending)
OVERLAP_TYPES = ["direct", "partial", "adjacent", "proximate", "none"]

# Risk levels descending
RISK_LEVELS = ["critical", "high", "medium", "low", "none"]

# FPIC score thresholds
FPIC_OBTAINED_THRESHOLD = Decimal("80")
FPIC_PARTIAL_THRESHOLD = Decimal("50")

# 10 FPIC elements
FPIC_ELEMENTS = [
    "community_identification",
    "information_disclosure",
    "prior_timing",
    "consultation_process",
    "community_representation",
    "consent_record",
    "absence_of_coercion",
    "agreement_documentation",
    "benefit_sharing",
    "monitoring_provisions",
]

# Default FPIC weights
DEFAULT_FPIC_WEIGHTS = {
    "community_identification": 0.10,
    "information_disclosure": 0.15,
    "prior_timing": 0.10,
    "consultation_process": 0.15,
    "community_representation": 0.10,
    "consent_record": 0.15,
    "absence_of_coercion": 0.10,
    "agreement_documentation": 0.05,
    "benefit_sharing": 0.05,
    "monitoring_provisions": 0.05,
}

# Default overlap risk weights
DEFAULT_OVERLAP_RISK_WEIGHTS = {
    "overlap_type": 0.40,
    "territory_legal_status": 0.20,
    "community_population": 0.10,
    "conflict_history": 0.15,
    "country_rights_framework": 0.15,
}

# Overlap type scores (PRD Section 6.1)
OVERLAP_TYPE_SCORES = {
    "direct": Decimal("100"),
    "partial": Decimal("80"),
    "adjacent": Decimal("50"),
    "proximate": Decimal("25"),
    "none": Decimal("0"),
}

# Legal status scores
LEGAL_STATUS_SCORES = {
    "titled": Decimal("100"),
    "declared": Decimal("80"),
    "claimed": Decimal("60"),
    "customary": Decimal("50"),
    "pending": Decimal("40"),
    "disputed": Decimal("60"),
}

# Violation type severity scores
VIOLATION_TYPE_SCORES = {
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

# Violation severity weights
DEFAULT_VIOLATION_SEVERITY_WEIGHTS = {
    "violation_type": 0.30,
    "spatial_proximity": 0.25,
    "community_population": 0.15,
    "legal_framework_gap": 0.15,
    "media_coverage": 0.15,
}

# Countries with FPIC legal frameworks (8 per PRD)
FPIC_COUNTRIES = ["BR", "CO", "PE", "PY", "GT", "ID", "MY", "CM"]

# ILO 169 ratified countries (subset relevant to EUDR)
ILO_169_EUDR_COUNTRIES = [
    "AR", "BO", "BR", "CL", "CO", "CR", "DK", "EC", "GT",
    "HN", "MX", "NI", "NL", "NO", "PY", "PE", "ES",
]

# High-risk indigenous rights countries
HIGH_RISK_COUNTRIES = [
    "BR", "ID", "CO", "MY", "PY", "CM", "CI", "GH", "NG", "CD",
    "CG", "PE", "BO", "VN", "TH", "MM",
]

# 7 consultation stages in order
CONSULTATION_STAGES_ORDERED = [
    ConsultationStage.IDENTIFIED,
    ConsultationStage.NOTIFIED,
    ConsultationStage.INFORMATION_SHARED,
    ConsultationStage.CONSULTATION_HELD,
    ConsultationStage.RESPONSE_RECORDED,
    ConsultationStage.AGREEMENT_REACHED,
    ConsultationStage.MONITORING_ACTIVE,
]

# 7 FPIC workflow stages (main path)
WORKFLOW_STAGES_ORDERED = [
    FPICWorkflowStage.IDENTIFICATION,
    FPICWorkflowStage.INFORMATION_DISCLOSURE,
    FPICWorkflowStage.CONSULTATION,
    FPICWorkflowStage.CONSENT_DECISION,
    FPICWorkflowStage.AGREEMENT,
    FPICWorkflowStage.IMPLEMENTATION,
    FPICWorkflowStage.MONITORING,
]

# Grievance SLA defaults (days)
GRIEVANCE_SLA_DEFAULTS = {
    "acknowledge": 5,
    "investigate": 30,
    "resolve": 90,
}

# Report types
ALL_REPORT_TYPES = [
    ReportType.INDIGENOUS_RIGHTS_COMPLIANCE,
    ReportType.DDS_SECTION,
    ReportType.FSC_FPIC,
    ReportType.RSPO_FPIC,
    ReportType.SUPPLIER_SCORECARD,
    ReportType.TREND_REPORT,
    ReportType.EXECUTIVE_SUMMARY,
    ReportType.BI_EXPORT,
]

# Report formats
ALL_REPORT_FORMATS = [
    ReportFormat.PDF,
    ReportFormat.JSON,
    ReportFormat.HTML,
    ReportFormat.CSV,
    ReportFormat.XLSX,
]

# Report languages
ALL_REPORT_LANGUAGES = ["en", "fr", "de", "es", "pt"]


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Create an IndigenousRightsCheckerConfig with test defaults."""
    return IndigenousRightsCheckerConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        inner_buffer_km=5.0,
        outer_buffer_km=25.0,
        buffer_polygon_points=64,
        territory_staleness_months=12,
        fpic_validity_years=5,
        fpic_min_lead_time_days=90,
        fpic_coercion_min_days=30,
        violation_dedup_window_days=7,
        batch_max_size=10000,
        batch_concurrency=4,
        batch_timeout_s=300,
        retention_years=5,
        enable_provenance=True,
        genesis_hash="GL-EUDR-IRC-021-TEST-GENESIS",
        enable_metrics=False,
    )


@pytest.fixture
def strict_config():
    """Create an IndigenousRightsCheckerConfig with strict thresholds."""
    return IndigenousRightsCheckerConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        inner_buffer_km=2.0,
        outer_buffer_km=10.0,
        fpic_validity_years=3,
        fpic_min_lead_time_days=120,
        fpic_coercion_min_days=60,
        violation_dedup_window_days=3,
        enable_provenance=True,
        genesis_hash="GL-EUDR-IRC-021-TEST-STRICT-GENESIS",
        enable_metrics=False,
    )


@pytest.fixture(autouse=True)
def reset_singleton_config():
    """Reset the singleton config after each test to avoid cross-test leaks."""
    yield
    reset_config()


@pytest.fixture
def uuid_gen():
    """Create a deterministic UUID generator."""
    return DeterministicUUID()


# ---------------------------------------------------------------------------
# Mock Fixtures (Provenance, Metrics, Database, Redis)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provenance():
    """Create a mock ProvenanceTracker for testing."""
    tracker = MagicMock()
    tracker.genesis_hash = compute_test_hash({"genesis": "GL-EUDR-IRC-021"})
    tracker._entries = []
    tracker.entry_count = 0

    def record_side_effect(entity_type, action, entity_id, data=None, metadata=None):
        entry = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "hash_value": compute_test_hash({
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": action,
            }),
            "parent_hash": (
                tracker.genesis_hash
                if not tracker._entries
                else tracker._entries[-1]["hash_value"]
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        tracker._entries.append(entry)
        tracker.entry_count = len(tracker._entries)
        return entry

    tracker.record = MagicMock(side_effect=record_side_effect)
    tracker.verify_chain = MagicMock(return_value=True)
    tracker.get_entries = MagicMock(return_value=[])
    tracker.build_hash = MagicMock(side_effect=lambda d: compute_test_hash(d))
    tracker.clear = MagicMock()
    return tracker


@pytest.fixture
def mock_metrics():
    """Create a mock MetricsCollector for testing."""
    metrics = MagicMock()
    metrics.increment = MagicMock()
    metrics.observe = MagicMock()
    metrics.set_gauge = MagicMock()
    metrics.start_timer = MagicMock(return_value=MagicMock())
    metrics.labels = MagicMock(return_value=metrics)
    return metrics


@pytest.fixture
def mock_db_pool():
    """Create a mock database connection pool."""
    pool = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()

    cursor.fetchone = MagicMock(return_value=None)
    cursor.fetchall = MagicMock(return_value=[])
    cursor.rowcount = 0
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=False)

    conn.cursor = MagicMock(return_value=cursor)
    conn.execute = AsyncMock()
    conn.commit = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)

    pool.connection = MagicMock(return_value=conn)
    pool.getconn = AsyncMock(return_value=conn)
    pool.putconn = AsyncMock()
    return pool


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=0)
    redis.expire = AsyncMock(return_value=True)
    redis.keys = AsyncMock(return_value=[])
    redis.pipeline = MagicMock(return_value=redis)
    redis.execute = AsyncMock(return_value=[])
    return redis


@pytest.fixture
def mock_auth():
    """Create mock authentication middleware."""
    auth = MagicMock()
    auth.validate_token = MagicMock(return_value={
        "sub": "test-user-001",
        "role": "eudr_analyst",
        "permissions": [
            "eudr-irc:territories:read",
            "eudr-irc:territories:write",
            "eudr-irc:fpic:read",
            "eudr-irc:fpic:write",
            "eudr-irc:overlaps:read",
            "eudr-irc:consultations:read",
            "eudr-irc:consultations:write",
            "eudr-irc:violations:read",
            "eudr-irc:registry:read",
            "eudr-irc:reports:read",
            "eudr-irc:reports:generate",
        ],
    })
    auth.require_permission = MagicMock(return_value=True)
    return auth


# ---------------------------------------------------------------------------
# Sample Territory Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_territory():
    """Sample IndigenousTerritory: Yanomami, Brazil, titled."""
    return IndigenousTerritory(
        territory_id="t-001",
        territory_name="Terra Indigena Yanomami",
        indigenous_name="Watoriki",
        people_name="Yanomami",
        country_code="BR",
        region="amazon_basin",
        area_hectares=Decimal("9664975"),
        legal_status=TerritoryLegalStatus.TITLED,
        recognition_date=date(1992, 5, 25),
        governing_authority="FUNAI",
        boundary_geojson={
            "type": "Polygon",
            "coordinates": [[
                [-60.0, -3.0], [-60.0, -2.0],
                [-59.0, -2.0], [-59.0, -3.0],
                [-60.0, -3.0],
            ]],
        },
        data_source="funai",
        source_url="https://terrasindigenas.org.br/pt-br/terras-indigenas/3880",
        confidence=ConfidenceLevel.HIGH,
        version=1,
        provenance_hash=compute_test_hash({
            "territory_id": "t-001",
            "territory_name": "Terra Indigena Yanomami",
            "country_code": "BR",
        }),
    )


@pytest.fixture
def sample_territories():
    """List of 5 sample territories in different countries."""
    specs = [
        ("t-001", "Terra Indigena Yanomami", "Yanomami", "BR",
         Decimal("9664975"), TerritoryLegalStatus.TITLED, "funai",
         ConfidenceLevel.HIGH, Decimal("-3.0"), Decimal("-60.0")),
        ("t-002", "Adat Dayak Kalimantan", "Dayak", "ID",
         Decimal("500000"), TerritoryLegalStatus.CUSTOMARY, "bpn_aman",
         ConfidenceLevel.MEDIUM, Decimal("-1.5"), Decimal("116.0")),
        ("t-003", "Territoire Baka", "Baka", "CM",
         Decimal("200000"), TerritoryLegalStatus.CLAIMED, "achpr",
         ConfidenceLevel.LOW, Decimal("3.0"), Decimal("14.0")),
        ("t-004", "Resguardo Nukak", "Nukak Maku", "CO",
         Decimal("955000"), TerritoryLegalStatus.DECLARED, "national_registry",
         ConfidenceLevel.HIGH, Decimal("2.5"), Decimal("-71.0")),
        ("t-005", "Ashaninka Territory", "Ashaninka", "PE",
         Decimal("340000"), TerritoryLegalStatus.TITLED, "landmark",
         ConfidenceLevel.MEDIUM, Decimal("-11.0"), Decimal("-74.0")),
    ]
    territories = []
    for (tid, name, people, country, area, status, source,
         conf, lat, lon) in specs:
        territories.append(IndigenousTerritory(
            territory_id=tid,
            territory_name=name,
            people_name=people,
            country_code=country,
            area_hectares=area,
            legal_status=status,
            data_source=source,
            confidence=conf,
            boundary_geojson={
                "type": "Polygon",
                "coordinates": [[
                    [float(lon), float(lat)],
                    [float(lon), float(lat) + 1],
                    [float(lon) + 1, float(lat) + 1],
                    [float(lon) + 1, float(lat)],
                    [float(lon), float(lat)],
                ]],
            },
            provenance_hash=compute_test_hash({
                "territory_id": tid,
                "country_code": country,
            }),
        ))
    return territories


# ---------------------------------------------------------------------------
# Sample FPIC Assessment Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_fpic_obtained():
    """FPICAssessment with score >= 80: CONSENT_OBTAINED."""
    return FPICAssessment(
        assessment_id="a-001",
        plot_id="p-001",
        territory_id="t-001",
        community_id="c-001",
        fpic_score=Decimal("87.50"),
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
        country_specific_rules="BR",
        temporal_compliance=True,
        coercion_flags=[],
        validity_start=date(2024, 1, 1),
        validity_end=date(2029, 1, 1),
        provenance_hash=compute_test_hash({
            "assessment_id": "a-001",
            "fpic_score": "87.50",
        }),
    )


@pytest.fixture
def sample_fpic_partial():
    """FPICAssessment with 50 <= score < 80: CONSENT_PARTIAL."""
    return FPICAssessment(
        assessment_id="a-002",
        plot_id="p-002",
        territory_id="t-001",
        community_id="c-001",
        fpic_score=Decimal("62.00"),
        fpic_status=FPICStatus.CONSENT_PARTIAL,
        community_identification_score=Decimal("70"),
        information_disclosure_score=Decimal("60"),
        prior_timing_score=Decimal("50"),
        consultation_process_score=Decimal("65"),
        community_representation_score=Decimal("60"),
        consent_record_score=Decimal("55"),
        absence_of_coercion_score=Decimal("80"),
        agreement_documentation_score=Decimal("40"),
        benefit_sharing_score=Decimal("30"),
        monitoring_provisions_score=Decimal("20"),
        country_specific_rules="BR",
        temporal_compliance=False,
        coercion_flags=["rushed_timeline"],
        provenance_hash=compute_test_hash({
            "assessment_id": "a-002",
            "fpic_score": "62.00",
        }),
    )


@pytest.fixture
def sample_fpic_missing():
    """FPICAssessment with score < 50: CONSENT_MISSING."""
    return FPICAssessment(
        assessment_id="a-003",
        plot_id="p-003",
        territory_id="t-002",
        fpic_score=Decimal("25.00"),
        fpic_status=FPICStatus.CONSENT_MISSING,
        community_identification_score=Decimal("30"),
        information_disclosure_score=Decimal("20"),
        prior_timing_score=Decimal("10"),
        consultation_process_score=Decimal("25"),
        community_representation_score=Decimal("20"),
        consent_record_score=Decimal("0"),
        absence_of_coercion_score=Decimal("50"),
        agreement_documentation_score=Decimal("0"),
        benefit_sharing_score=Decimal("0"),
        monitoring_provisions_score=Decimal("0"),
        country_specific_rules="ID",
        temporal_compliance=False,
        coercion_flags=["economic_pressure", "information_withheld"],
        provenance_hash=compute_test_hash({
            "assessment_id": "a-003",
            "fpic_score": "25.00",
        }),
    )


# ---------------------------------------------------------------------------
# Sample Overlap Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_overlap_direct():
    """TerritoryOverlap: DIRECT overlap, CRITICAL risk."""
    return TerritoryOverlap(
        overlap_id="o-001",
        plot_id="p-001",
        territory_id="t-001",
        overlap_type=OverlapType.DIRECT,
        overlap_area_hectares=Decimal("150.5"),
        overlap_pct_of_plot=Decimal("100.0"),
        overlap_pct_of_territory=Decimal("0.002"),
        distance_meters=Decimal("0"),
        affected_communities=["c-001"],
        risk_score=Decimal("92.50"),
        risk_level=RiskLevel.CRITICAL,
        provenance_hash=compute_test_hash({
            "overlap_id": "o-001",
            "overlap_type": "direct",
        }),
    )


@pytest.fixture
def sample_overlap_adjacent():
    """TerritoryOverlap: ADJACENT overlap, MEDIUM risk."""
    return TerritoryOverlap(
        overlap_id="o-002",
        plot_id="p-002",
        territory_id="t-001",
        overlap_type=OverlapType.ADJACENT,
        distance_meters=Decimal("3500"),
        affected_communities=["c-001"],
        risk_score=Decimal("45.00"),
        risk_level=RiskLevel.MEDIUM,
        provenance_hash=compute_test_hash({
            "overlap_id": "o-002",
            "overlap_type": "adjacent",
        }),
    )


@pytest.fixture
def sample_overlap_none():
    """TerritoryOverlap: NONE overlap, NONE risk."""
    return TerritoryOverlap(
        overlap_id="o-003",
        plot_id="p-003",
        territory_id="t-001",
        overlap_type=OverlapType.NONE,
        distance_meters=Decimal("50000"),
        risk_score=Decimal("0"),
        risk_level=RiskLevel.NONE,
        provenance_hash=compute_test_hash({
            "overlap_id": "o-003",
            "overlap_type": "none",
        }),
    )


# ---------------------------------------------------------------------------
# Sample Community Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_community():
    """Sample IndigenousCommunity: Yanomami, Brazil."""
    return IndigenousCommunity(
        community_id="c-001",
        community_name="Yanomami do Rio Catrimani",
        indigenous_name="Watoriki",
        people_name="Yanomami",
        language="Yanomami",
        estimated_population=26000,
        country_code="BR",
        region="amazon_basin",
        territory_ids=["t-001"],
        legal_recognition_status=CommunityRecognitionStatus.CONSTITUTIONALLY_RECOGNIZED,
        applicable_legal_protections=[
            "Federal Constitution Art. 231",
            "ILO Convention 169",
        ],
        ilo_169_coverage=True,
        fpic_legal_requirement=True,
        representative_organizations=[
            {"name": "Hutukara Associacao Yanomami", "type": "community_org"},
        ],
        commodity_relevance=["cattle", "wood"],
        provenance_hash=compute_test_hash({
            "community_id": "c-001",
            "community_name": "Yanomami do Rio Catrimani",
        }),
    )


@pytest.fixture
def sample_communities():
    """List of 5 sample communities."""
    specs = [
        ("c-001", "Yanomami do Rio Catrimani", "Yanomami", "BR", 26000, True),
        ("c-002", "Dayak Meratus", "Dayak", "ID", 15000, False),
        ("c-003", "Baka de Dja", "Baka", "CM", 5000, False),
        ("c-004", "Nukak Maku", "Nukak Maku", "CO", 1200, True),
        ("c-005", "Ashaninka del Ene", "Ashaninka", "PE", 8000, True),
    ]
    communities = []
    for cid, name, people, country, pop, ilo in specs:
        communities.append(IndigenousCommunity(
            community_id=cid,
            community_name=name,
            people_name=people,
            country_code=country,
            estimated_population=pop,
            territory_ids=[f"t-{cid.split('-')[1]}"],
            ilo_169_coverage=ilo,
            fpic_legal_requirement=ilo,
            provenance_hash=compute_test_hash({
                "community_id": cid,
                "country_code": country,
            }),
        ))
    return communities


# ---------------------------------------------------------------------------
# Sample Consultation Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_consultation():
    """Sample ConsultationRecord: CONSULTATION_HELD stage."""
    return ConsultationRecord(
        consultation_id="con-001",
        community_id="c-001",
        plot_id="p-001",
        territory_id="t-001",
        consultation_stage=ConsultationStage.CONSULTATION_HELD,
        meeting_date=date(2026, 3, 1),
        meeting_location="Community house, Catrimani River",
        attendees=[
            {"name": "Chief Davi Kopenawa", "role": "community_leader"},
            {"name": "FUNAI representative", "role": "government"},
            {"name": "Operator representative", "role": "operator"},
        ],
        agenda="Discussion of soya farming near territory boundary",
        minutes="Community expressed concerns about river contamination.",
        outcomes="Agreed to 10km buffer zone and water monitoring",
        follow_up_actions=[
            {"action": "Water quality baseline", "deadline": "2026-04-01"},
        ],
        provenance_hash=compute_test_hash({
            "consultation_id": "con-001",
            "consultation_stage": "consultation_held",
        }),
    )


# ---------------------------------------------------------------------------
# Sample Grievance Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_grievance():
    """Sample GrievanceRecord: SUBMITTED status."""
    now = datetime.now(timezone.utc)
    return GrievanceRecord(
        grievance_id="g-001",
        community_id="c-001",
        territory_id="t-001",
        grievance_type="water_contamination",
        description="River water contaminated by upstream soya farming operations",
        severity=AlertSeverity.HIGH,
        status=GrievanceStatus.SUBMITTED,
        submitted_at=now,
        investigation_deadline=now + timedelta(days=30),
        resolution_deadline=now + timedelta(days=90),
        provenance_hash=compute_test_hash({
            "grievance_id": "g-001",
            "grievance_type": "water_contamination",
        }),
    )


# ---------------------------------------------------------------------------
# Sample Violation Alert Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_violation():
    """Sample ViolationAlert: FPIC_VIOLATION, HIGH severity."""
    return ViolationAlert(
        alert_id="v-001",
        source="iwgia",
        source_url="https://iwgia.org/report/2026/br-001",
        publication_date=date(2026, 2, 15),
        violation_type=ViolationType.FPIC_VIOLATION,
        country_code="BR",
        region="amazon_basin",
        location_lat=-3.46,
        location_lon=-62.21,
        affected_communities=["c-001"],
        severity_score=Decimal("78.50"),
        severity_level=AlertSeverity.HIGH,
        supply_chain_correlation=True,
        affected_plots=["p-001"],
        affected_suppliers=["sup-001"],
        provenance_hash=compute_test_hash({
            "alert_id": "v-001",
            "violation_type": "fpic_violation",
        }),
    )


@pytest.fixture
def sample_violations():
    """List of 5 violation alerts at different severity levels."""
    specs = [
        ("v-001", ViolationType.PHYSICAL_VIOLENCE, "BR", Decimal("95"),
         AlertSeverity.CRITICAL),
        ("v-002", ViolationType.LAND_SEIZURE, "ID", Decimal("82"),
         AlertSeverity.HIGH),
        ("v-003", ViolationType.FPIC_VIOLATION, "CM", Decimal("65"),
         AlertSeverity.MEDIUM),
        ("v-004", ViolationType.BENEFIT_SHARING_BREACH, "CO", Decimal("45"),
         AlertSeverity.LOW),
        ("v-005", ViolationType.DISCRIMINATORY_POLICY, "PE", Decimal("30"),
         AlertSeverity.LOW),
    ]
    violations = []
    for vid, vtype, country, score, severity in specs:
        violations.append(ViolationAlert(
            alert_id=vid,
            source="iwgia",
            publication_date=date(2026, 2, 15),
            violation_type=vtype,
            country_code=country,
            severity_score=score,
            severity_level=severity,
            provenance_hash=compute_test_hash({
                "alert_id": vid,
                "violation_type": vtype.value,
            }),
        ))
    return violations


# ---------------------------------------------------------------------------
# Sample Workflow Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_workflow():
    """Sample FPICWorkflow at CONSULTATION stage."""
    return FPICWorkflow(
        workflow_id="wf-001",
        plot_id="p-001",
        territory_id="t-001",
        community_id="c-001",
        current_stage=FPICWorkflowStage.CONSULTATION,
        stage_history=[
            {"stage": "identification", "entered": "2025-12-01T00:00:00Z"},
            {"stage": "information_disclosure", "entered": "2026-01-01T00:00:00Z"},
            {"stage": "consultation", "entered": "2026-02-01T00:00:00Z"},
        ],
        sla_status="on_track",
        next_deadline=datetime(2026, 4, 1, tzinfo=timezone.utc),
        escalation_level=0,
        provenance_hash=compute_test_hash({
            "workflow_id": "wf-001",
            "current_stage": "consultation",
        }),
    )


# ---------------------------------------------------------------------------
# Sample Report Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_report():
    """Sample ComplianceReport: JSON format, EN language."""
    return ComplianceReport(
        report_id="r-001",
        report_type=ReportType.INDIGENOUS_RIGHTS_COMPLIANCE,
        title="Indigenous Rights Compliance Report Q1 2026",
        format=ReportFormat.JSON,
        language="en",
        scope_type="operator",
        scope_ids=["op-001"],
        provenance_hash=compute_test_hash({
            "report_id": "r-001",
            "report_type": "indigenous_rights_compliance",
        }),
    )


# ---------------------------------------------------------------------------
# Sample Coordinates and Locations
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_coordinates():
    """Dictionary of test locations with (lat, lon) tuples."""
    return {
        "brazil_yanomami": (Decimal("-3.0"), Decimal("-60.0")),
        "brazil_mato_grosso": (Decimal("-12.5"), Decimal("-55.3")),
        "indonesia_kalimantan": (Decimal("-1.5"), Decimal("116.0")),
        "cameroon_dja": (Decimal("3.0"), Decimal("14.0")),
        "colombia_guaviare": (Decimal("2.5"), Decimal("-71.0")),
        "peru_ene": (Decimal("-11.0"), Decimal("-74.0")),
        "denmark_copenhagen": (Decimal("55.7"), Decimal("12.6")),
        "north_pole": (Decimal("90.0"), Decimal("0.0")),
        "south_pole": (Decimal("-90.0"), Decimal("0.0")),
        "antimeridian": (Decimal("0.0"), Decimal("180.0")),
    }


@pytest.fixture
def sample_plot_geojson():
    """Sample GeoJSON polygon for a production plot (approx 150 ha)."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [-60.05, -3.05],
            [-60.05, -2.95],
            [-59.95, -2.95],
            [-59.95, -3.05],
            [-60.05, -3.05],
        ]],
    }


# ---------------------------------------------------------------------------
# FPIC Documentation Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_fpic_documentation():
    """Complete FPIC documentation with all 10 elements present."""
    return {
        "community_identified": True,
        "community_identification_date": "2024-01-15",
        "community_leadership_verified": True,
        "information_disclosure_date": "2024-02-01",
        "information_materials_provided": True,
        "information_language_local": True,
        "prior_timing_met": True,
        "production_start_date": "2025-01-01",
        "consultation_held": True,
        "consultation_date": "2024-06-15",
        "consultation_minutes": True,
        "community_representation_verified": True,
        "representative_list_provided": True,
        "consent_recorded": True,
        "consent_date": "2024-08-01",
        "consent_form_signed": True,
        "absence_of_coercion_verified": True,
        "independent_observer_present": True,
        "agreement_documented": True,
        "agreement_terms_clear": True,
        "benefit_sharing_defined": True,
        "benefit_sharing_monetary": {"annual_payment": 50000, "currency": "BRL"},
        "monitoring_provisions_included": True,
        "monitoring_frequency": "quarterly",
    }


@pytest.fixture
def minimal_fpic_documentation():
    """Minimal FPIC documentation with only basic elements."""
    return {
        "community_identified": True,
        "information_disclosure_date": "2024-02-01",
        "consultation_held": True,
        "consultation_date": "2024-06-15",
    }


@pytest.fixture
def empty_fpic_documentation():
    """Empty FPIC documentation dict."""
    return {}


# ---------------------------------------------------------------------------
# Country Risk Score Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_country_scores():
    """Country indigenous rights scores for EUDR-relevant countries."""
    specs = [
        ("BR", True, True, Decimal("45"), Decimal("55"), Decimal("50"),
         Decimal("40"), 15, Decimal("48"), CountryRiskLevel.HIGH),
        ("CO", True, True, Decimal("40"), Decimal("50"), Decimal("45"),
         Decimal("35"), 20, Decimal("43"), CountryRiskLevel.HIGH),
        ("ID", False, False, Decimal("30"), Decimal("35"), Decimal("25"),
         Decimal("20"), 30, Decimal("28"), CountryRiskLevel.HIGH),
        ("DK", False, False, Decimal("90"), Decimal("95"), Decimal("85"),
         Decimal("95"), 0, Decimal("91"), CountryRiskLevel.LOW),
        ("CM", False, False, Decimal("25"), Decimal("30"), Decimal("20"),
         Decimal("15"), 25, Decimal("23"), CountryRiskLevel.HIGH),
    ]
    scores = []
    for (cc, ilo, fpic_req, tenure, recog, judicial, demarc,
         conflicts, composite, risk) in specs:
        scores.append(CountryIndigenousRightsScore(
            score_id=f"cs-{cc}",
            country_code=cc,
            ilo_169_ratified=ilo,
            fpic_legal_requirement=fpic_req,
            land_tenure_security_score=tenure,
            indigenous_rights_recognition_score=recog,
            judicial_protection_score=judicial,
            territory_demarcation_pct=demarc,
            active_land_conflicts=conflicts,
            composite_indigenous_rights_score=composite,
            risk_level=risk,
            provenance_hash=compute_test_hash({
                "score_id": f"cs-{cc}",
                "country_code": cc,
            }),
        ))
    return scores


# ---------------------------------------------------------------------------
# Computation Helpers for Tests
# ---------------------------------------------------------------------------


def compute_fpic_score(
    element_scores: Dict[str, Decimal],
    weights: Optional[Dict[str, float]] = None,
) -> Decimal:
    """Compute weighted FPIC composite score.

    Args:
        element_scores: Dictionary of element name to score (0-100).
        weights: Optional weight overrides.

    Returns:
        Weighted composite score as Decimal (0-100).
    """
    w = weights or DEFAULT_FPIC_WEIGHTS
    total = Decimal("0")
    for elem, weight in w.items():
        score = element_scores.get(elem, Decimal("0"))
        total += score * Decimal(str(weight))
    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def classify_fpic_status(score: Decimal) -> str:
    """Classify FPIC status from composite score."""
    if score >= FPIC_OBTAINED_THRESHOLD:
        return FPICStatus.CONSENT_OBTAINED.value
    elif score >= FPIC_PARTIAL_THRESHOLD:
        return FPICStatus.CONSENT_PARTIAL.value
    else:
        return FPICStatus.CONSENT_MISSING.value


def compute_overlap_risk_score(
    overlap_type: str,
    legal_status: str,
    community_population: int,
    conflict_history_score: Decimal,
    country_framework_score: Decimal,
    weights: Optional[Dict[str, float]] = None,
) -> Decimal:
    """Compute overlap risk score from 5 factors."""
    w = weights or DEFAULT_OVERLAP_RISK_WEIGHTS
    ot_score = OVERLAP_TYPE_SCORES.get(overlap_type, Decimal("0"))
    ls_score = LEGAL_STATUS_SCORES.get(legal_status, Decimal("50"))

    # Population factor: log scale
    if community_population >= 50000:
        pop_score = Decimal("100")
    elif community_population >= 10000:
        pop_score = Decimal("80")
    elif community_population >= 5000:
        pop_score = Decimal("60")
    elif community_population >= 1000:
        pop_score = Decimal("40")
    else:
        pop_score = Decimal("20")

    total = (
        ot_score * Decimal(str(w["overlap_type"]))
        + ls_score * Decimal(str(w["territory_legal_status"]))
        + pop_score * Decimal(str(w["community_population"]))
        + conflict_history_score * Decimal(str(w["conflict_history"]))
        + country_framework_score * Decimal(str(w["country_rights_framework"]))
    )
    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def classify_risk_level(score: Decimal) -> str:
    """Classify risk level from score."""
    if score >= Decimal("80"):
        return RiskLevel.CRITICAL.value
    elif score >= Decimal("60"):
        return RiskLevel.HIGH.value
    elif score >= Decimal("40"):
        return RiskLevel.MEDIUM.value
    elif score >= Decimal("20"):
        return RiskLevel.LOW.value
    else:
        return RiskLevel.NONE.value


def compute_violation_severity(
    violation_type: str,
    proximity_score: Decimal,
    population_score: Decimal,
    legal_gap_score: Decimal,
    media_score: Decimal,
    weights: Optional[Dict[str, float]] = None,
) -> Decimal:
    """Compute violation severity from 5 factors."""
    w = weights or DEFAULT_VIOLATION_SEVERITY_WEIGHTS
    vt_score = VIOLATION_TYPE_SCORES.get(violation_type, Decimal("50"))
    total = (
        vt_score * Decimal(str(w["violation_type"]))
        + proximity_score * Decimal(str(w["spatial_proximity"]))
        + population_score * Decimal(str(w["community_population"]))
        + legal_gap_score * Decimal(str(w["legal_framework_gap"]))
        + media_score * Decimal(str(w["media_coverage"]))
    )
    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
