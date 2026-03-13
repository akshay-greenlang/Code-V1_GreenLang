# -*- coding: utf-8 -*-
"""
AGENT-EUDR-019: Corruption Index Monitor - Institutional Quality Engine

Evaluates institutional frameworks, rule of law, judicial independence,
regulatory enforcement capacity, forest governance quality, and land
tenure security. Provides composite institutional quality scores across
8 dimensions with special focus on forest governance and land tenure --
two critical factors for EUDR compliance risk assessment.

Assessment Dimensions (8):
    1. JUDICIAL_INDEPENDENCE (JI): Independence of courts from political
       interference, judicial tenure security, case backlog metrics
    2. REGULATORY_ENFORCEMENT (RE): Capacity to implement and enforce
       environmental and forestry regulations
    3. PROPERTY_RIGHTS (PR): Strength of property rights protection,
       land registration systems, expropriation risk
    4. CONTRACT_ENFORCEMENT (CE): Reliability of contract enforcement
       through courts, arbitration availability
    5. TRANSPARENCY_LAWS (TL): Freedom of information legislation,
       open data initiatives, government disclosure requirements
    6. ANTI_CORRUPTION_FRAMEWORK (ACF): Anti-corruption institutional
       design, independent anti-corruption agencies, UNCAC compliance
    7. FOREST_GOVERNANCE (FG): Forest law enforcement capacity,
       concession management, community forestry rights, REDD+ readiness
    8. LAND_TENURE_SECURITY (LTS): Land title registration coverage,
       indigenous land rights recognition, tenure dispute resolution

Institutional Capacity Levels:
    STRONG:    composite >= 75  (robust institutions)
    ADEQUATE:  50 <= composite < 75  (functional with gaps)
    WEAK:      25 <= composite < 50  (significant deficiencies)
    VERY_WEAK: composite < 25  (institutional failure)

Forest Governance Profile:
    - legal_framework_score (0-100): Completeness of forest legislation
    - enforcement_capacity (0-1): Ratio of effective enforcement actions
    - monitoring_capability (0-1): Satellite/field monitoring coverage
    - indigenous_rights_protection (0-100): Recognition score
    - community_participation (0-100): Community forestry engagement
    - illegal_logging_prevalence: LOW/MEDIUM/HIGH/CRITICAL

EUDR Risk Mapping:
    institutional_composite 0   -> EUDR risk 1.0  (weakest institutions)
    institutional_composite 100 -> EUDR risk 0.0  (strongest institutions)
    Formula: eudr_risk = 1.0 - (composite / 100)

Zero-Hallucination Guarantees:
    - All scores from embedded institutional quality reference database
    - Composite scoring is deterministic weighted average
    - Forest governance assessment uses static reference data
    - All arithmetic uses ``decimal.Decimal``
    - SHA-256 provenance hashes on every result
    - No LLM/ML in any scoring or classification path

Prometheus Metrics (gl_eudr_cim_ prefix):
    - gl_eudr_cim_inst_assessments_total       (Counter)
    - gl_eudr_cim_inst_forest_queries_total    (Counter)
    - gl_eudr_cim_inst_query_duration_seconds  (Histogram)
    - gl_eudr_cim_inst_weak_countries          (Gauge)
    - gl_eudr_cim_inst_errors_total            (Counter, label: operation)

Performance Targets:
    - Single country assessment: <3ms
    - Forest governance query: <2ms
    - Cross-country comparison (10 countries): <20ms

Regulatory References:
    - EU 2023/1115 Article 10: Governance as risk factor
    - EU 2023/1115 Article 29: Country risk classification
    - EU 2023/1115 Article 13: Due diligence requirements
    - FAO Voluntary Guidelines on Forest Governance
    - VGGT (Voluntary Guidelines on Tenure of Land)

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-019 (Engine 4: Institutional Quality)
Agent ID: GL-EUDR-CIM-019
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-CIM-019"

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.corruption_index_monitor.config import get_config
except ImportError:
    get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.corruption_index_monitor.provenance import (
        ProvenanceTracker, get_tracker,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    get_tracker = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.corruption_index_monitor.metrics import (
        record_institutional_assessment, observe_query_duration, record_api_error,
    )
except ImportError:
    record_institutional_assessment = None  # type: ignore[assignment]
    observe_query_duration = None  # type: ignore[assignment]
    record_api_error = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Prometheus metrics (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(name: str, doc: str, labelnames: list = None):  # type: ignore[assignment]
        try:
            return Counter(name, doc, labelnames=labelnames or [])
        except ValueError:
            for c in _REGISTRY._names_to_collectors.values():
                if hasattr(c, "_name") and c._name == name:
                    return c
            from prometheus_client import CollectorRegistry
            return Counter(name, doc, labelnames=labelnames or [],
                           registry=CollectorRegistry())

    def _safe_histogram(name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
                        buckets: tuple = ()):
        try:
            kw: Dict[str, Any] = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(name, doc, labelnames=labelnames or [], **kw)
        except ValueError:
            for c in _REGISTRY._names_to_collectors.values():
                if hasattr(c, "_name") and c._name == name:
                    return c
            from prometheus_client import CollectorRegistry
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(name, doc, labelnames=labelnames or [],
                             registry=CollectorRegistry(), **kw)

    def _safe_gauge(name: str, doc: str, labelnames: list = None):  # type: ignore[assignment]
        try:
            return Gauge(name, doc, labelnames=labelnames or [])
        except ValueError:
            for c in _REGISTRY._names_to_collectors.values():
                if hasattr(c, "_name") and c._name == name:
                    return c
            from prometheus_client import CollectorRegistry
            return Gauge(name, doc, labelnames=labelnames or [],
                         registry=CollectorRegistry())

    _inst_assessments_total = _safe_counter(
        "gl_eudr_cim_inst_assessments_total",
        "Total institutional quality assessments performed",
    )
    _inst_forest_queries_total = _safe_counter(
        "gl_eudr_cim_inst_forest_queries_total",
        "Total forest governance queries performed",
    )
    _inst_query_duration = _safe_histogram(
        "gl_eudr_cim_inst_query_duration_seconds",
        "Duration of institutional quality queries in seconds",
        buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5),
    )
    _inst_weak_countries = _safe_gauge(
        "gl_eudr_cim_inst_weak_countries",
        "Countries with WEAK or VERY_WEAK institutional capacity",
    )
    _inst_errors_total = _safe_counter(
        "gl_eudr_cim_inst_errors_total",
        "Total errors in institutional quality engine operations",
        labelnames=["operation"],
    )
else:
    _inst_assessments_total = None  # type: ignore[assignment]
    _inst_forest_queries_total = None  # type: ignore[assignment]
    _inst_query_duration = None  # type: ignore[assignment]
    _inst_weak_countries = None  # type: ignore[assignment]
    _inst_errors_total = None  # type: ignore[assignment]


def _inc_inst_assessments() -> None:
    if PROMETHEUS_AVAILABLE and _inst_assessments_total is not None:
        _inst_assessments_total.inc()
    if record_institutional_assessment is not None:
        try:
            record_institutional_assessment()
        except Exception:
            pass


def _inc_forest_queries() -> None:
    if PROMETHEUS_AVAILABLE and _inst_forest_queries_total is not None:
        _inst_forest_queries_total.inc()


def _observe_inst_duration(seconds: float) -> None:
    if PROMETHEUS_AVAILABLE and _inst_query_duration is not None:
        _inst_query_duration.observe(seconds)
    if observe_query_duration is not None:
        try:
            observe_query_duration(seconds)
        except Exception:
            pass


def _inc_inst_error(operation: str) -> None:
    if PROMETHEUS_AVAILABLE and _inst_errors_total is not None:
        _inst_errors_total.labels(operation=operation).inc()
    if record_api_error is not None:
        try:
            record_api_error(operation)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _to_decimal(value: Any) -> Decimal:
    """Convert a numeric value to Decimal via string for determinism."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _compute_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class InstitutionalDimension(str, Enum):
    """Institutional quality assessment dimensions."""

    JUDICIAL_INDEPENDENCE = "JI"
    REGULATORY_ENFORCEMENT = "RE"
    PROPERTY_RIGHTS = "PR"
    CONTRACT_ENFORCEMENT = "CE"
    TRANSPARENCY_LAWS = "TL"
    ANTI_CORRUPTION_FRAMEWORK = "ACF"
    FOREST_GOVERNANCE = "FG"
    LAND_TENURE_SECURITY = "LTS"


class InstitutionalCapacityLevel(str, Enum):
    """Institutional capacity classification."""

    STRONG = "STRONG"
    ADEQUATE = "ADEQUATE"
    WEAK = "WEAK"
    VERY_WEAK = "VERY_WEAK"


class IllegalLoggingPrevalence(str, Enum):
    """Illegal logging prevalence classification."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InstitutionalAssessment:
    """Complete institutional quality assessment for a country.

    Attributes:
        country_code: ISO alpha-2 country code.
        year: Assessment year.
        dimension_scores: Scores per dimension (0-100).
        composite_score: Weighted composite (0-100).
        forest_governance_score: Forest-specific score (0-100).
        land_tenure_score: Land tenure security score (0-100).
        enforcement_effectiveness: Enforcement ratio (0-1).
        institutional_capacity_level: STRONG/ADEQUATE/WEAK/VERY_WEAK.
    """

    country_code: str
    year: int
    dimension_scores: Dict[str, Decimal]
    composite_score: Decimal
    forest_governance_score: Decimal
    land_tenure_score: Decimal
    enforcement_effectiveness: Decimal
    institutional_capacity_level: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "country_code": self.country_code,
            "year": self.year,
            "dimension_scores": {k: str(v) for k, v in self.dimension_scores.items()},
            "composite_score": str(self.composite_score),
            "forest_governance_score": str(self.forest_governance_score),
            "land_tenure_score": str(self.land_tenure_score),
            "enforcement_effectiveness": str(self.enforcement_effectiveness),
            "institutional_capacity_level": self.institutional_capacity_level,
        }


@dataclass
class ForestGovernanceProfile:
    """Forest governance profile for EUDR-specific assessment.

    Attributes:
        country_code: ISO alpha-2 country code.
        legal_framework_score: Forest legislation completeness (0-100).
        enforcement_capacity: Effective enforcement ratio (0-1).
        monitoring_capability: Monitoring coverage ratio (0-1).
        indigenous_rights_protection: Indigenous land rights score (0-100).
        community_participation: Community forestry engagement (0-100).
        illegal_logging_prevalence: Prevalence classification.
        concession_transparency: Concession transparency score (0-100).
        redd_plus_readiness: REDD+ programme readiness (0-100).
        protected_area_management: Protected area governance (0-100).
    """

    country_code: str
    legal_framework_score: Decimal
    enforcement_capacity: Decimal
    monitoring_capability: Decimal
    indigenous_rights_protection: Decimal
    community_participation: Decimal
    illegal_logging_prevalence: str
    concession_transparency: Decimal = field(default_factory=lambda: Decimal("0"))
    redd_plus_readiness: Decimal = field(default_factory=lambda: Decimal("0"))
    protected_area_management: Decimal = field(default_factory=lambda: Decimal("0"))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "country_code": self.country_code,
            "legal_framework_score": str(self.legal_framework_score),
            "enforcement_capacity": str(self.enforcement_capacity),
            "monitoring_capability": str(self.monitoring_capability),
            "indigenous_rights_protection": str(self.indigenous_rights_protection),
            "community_participation": str(self.community_participation),
            "illegal_logging_prevalence": self.illegal_logging_prevalence,
            "concession_transparency": str(self.concession_transparency),
            "redd_plus_readiness": str(self.redd_plus_readiness),
            "protected_area_management": str(self.protected_area_management),
        }


@dataclass
class InstitutionalQualityResult:
    """Result wrapper for institutional quality assessment."""

    success: bool
    data: Optional[InstitutionalAssessment] = None
    eudr_risk_factor: Decimal = field(default_factory=lambda: Decimal("1"))
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    calculation_timestamp: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class GovernanceProfileResult:
    """Result wrapper for governance profile query."""

    success: bool
    country_code: str = ""
    assessment: Optional[InstitutionalAssessment] = None
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    calculation_timestamp: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class StrengthResult:
    """Result wrapper for institutional strength assessment."""

    success: bool
    country_code: str = ""
    dimensions_assessed: List[str] = field(default_factory=list)
    dimension_scores: Dict[str, Decimal] = field(default_factory=dict)
    overall_strength: Decimal = field(default_factory=lambda: Decimal("0"))
    capacity_level: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    calculation_timestamp: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ForestGovernanceResult:
    """Result wrapper for forest governance assessment."""

    success: bool
    data: Optional[ForestGovernanceProfile] = None
    composite_forest_score: Decimal = field(default_factory=lambda: Decimal("0"))
    eudr_risk_factor: Decimal = field(default_factory=lambda: Decimal("1"))
    risk_factors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    calculation_timestamp: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result wrapper for cross-country institutional comparison."""

    success: bool
    countries: Dict[str, InstitutionalQualityResult] = field(default_factory=dict)
    dimension_rankings: Dict[str, List[Tuple[str, Decimal]]] = field(default_factory=dict)
    composite_rankings: List[Tuple[str, Decimal]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    calculation_timestamp: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Default dimension weights for composite scoring
# ---------------------------------------------------------------------------

DEFAULT_DIMENSION_WEIGHTS: Dict[str, Decimal] = {
    "JI": Decimal("0.125"),   # Judicial Independence
    "RE": Decimal("0.125"),   # Regulatory Enforcement
    "PR": Decimal("0.125"),   # Property Rights
    "CE": Decimal("0.100"),   # Contract Enforcement
    "TL": Decimal("0.100"),   # Transparency Laws
    "ACF": Decimal("0.125"),  # Anti-Corruption Framework
    "FG": Decimal("0.150"),   # Forest Governance (higher for EUDR)
    "LTS": Decimal("0.150"),  # Land Tenure Security (higher for EUDR)
}


# ---------------------------------------------------------------------------
# Institutional Quality Reference Data
# ---------------------------------------------------------------------------
# Format: country_code -> {dimension_code -> score (0-100)}
# Scores synthesized from World Bank Doing Business, Heritage Foundation
# Index of Economic Freedom, Bertelsmann Transformation Index, and
# country-specific governance assessments.
# ---------------------------------------------------------------------------

INSTITUTIONAL_DATA: Dict[str, Dict[str, int]] = {
    # Strong institutions (composite >= 75)
    "DK": {"JI": 95, "RE": 92, "PR": 95, "CE": 93, "TL": 94, "ACF": 96, "FG": 88, "LTS": 92},
    "FI": {"JI": 94, "RE": 91, "PR": 94, "CE": 92, "TL": 93, "ACF": 95, "FG": 90, "LTS": 91},
    "NZ": {"JI": 93, "RE": 90, "PR": 95, "CE": 91, "TL": 92, "ACF": 93, "FG": 85, "LTS": 93},
    "NO": {"JI": 92, "RE": 89, "PR": 93, "CE": 90, "TL": 91, "ACF": 92, "FG": 87, "LTS": 90},
    "SG": {"JI": 85, "RE": 95, "PR": 92, "CE": 95, "TL": 70, "ACF": 88, "FG": 75, "LTS": 90},
    "SE": {"JI": 93, "RE": 90, "PR": 93, "CE": 91, "TL": 92, "ACF": 93, "FG": 86, "LTS": 91},
    "CH": {"JI": 92, "RE": 88, "PR": 94, "CE": 92, "TL": 85, "ACF": 90, "FG": 80, "LTS": 92},
    "DE": {"JI": 88, "RE": 87, "PR": 90, "CE": 88, "TL": 82, "ACF": 85, "FG": 78, "LTS": 88},
    "UK": {"JI": 87, "RE": 84, "PR": 89, "CE": 87, "TL": 80, "ACF": 82, "FG": 75, "LTS": 87},
    "US": {"JI": 80, "RE": 82, "PR": 85, "CE": 86, "TL": 78, "ACF": 78, "FG": 72, "LTS": 85},
    "FR": {"JI": 82, "RE": 83, "PR": 85, "CE": 84, "TL": 75, "ACF": 80, "FG": 70, "LTS": 83},
    "CL": {"JI": 72, "RE": 70, "PR": 75, "CE": 73, "TL": 65, "ACF": 68, "FG": 58, "LTS": 70},
    # Adequate institutions (50 <= composite < 75)
    "CR": {"JI": 65, "RE": 62, "PR": 68, "CE": 65, "TL": 60, "ACF": 58, "FG": 55, "LTS": 60},
    "MY": {"JI": 55, "RE": 65, "PR": 68, "CE": 70, "TL": 50, "ACF": 55, "FG": 52, "LTS": 62},
    "BW": {"JI": 65, "RE": 60, "PR": 62, "CE": 58, "TL": 55, "ACF": 60, "FG": 48, "LTS": 55},
    "RW": {"JI": 52, "RE": 65, "PR": 55, "CE": 60, "TL": 42, "ACF": 58, "FG": 55, "LTS": 48},
    "GH": {"JI": 55, "RE": 48, "PR": 52, "CE": 50, "TL": 50, "ACF": 45, "FG": 42, "LTS": 45},
    "ZA": {"JI": 62, "RE": 55, "PR": 60, "CE": 58, "TL": 55, "ACF": 52, "FG": 50, "LTS": 55},
    "IN": {"JI": 55, "RE": 50, "PR": 55, "CE": 52, "TL": 50, "ACF": 45, "FG": 40, "LTS": 42},
    "BR": {"JI": 50, "RE": 48, "PR": 52, "CE": 55, "TL": 48, "ACF": 42, "FG": 38, "LTS": 40},
    "CO": {"JI": 48, "RE": 45, "PR": 50, "CE": 48, "TL": 45, "ACF": 40, "FG": 35, "LTS": 38},
    "ID": {"JI": 45, "RE": 48, "PR": 48, "CE": 50, "TL": 42, "ACF": 40, "FG": 38, "LTS": 35},
    "TH": {"JI": 42, "RE": 52, "PR": 50, "CE": 55, "TL": 38, "ACF": 40, "FG": 35, "LTS": 42},
    "PE": {"JI": 38, "RE": 40, "PR": 45, "CE": 42, "TL": 40, "ACF": 35, "FG": 32, "LTS": 35},
    "MX": {"JI": 35, "RE": 42, "PR": 48, "CE": 50, "TL": 38, "ACF": 32, "FG": 30, "LTS": 35},
    "AR": {"JI": 42, "RE": 40, "PR": 45, "CE": 48, "TL": 40, "ACF": 38, "FG": 32, "LTS": 38},
    "VN": {"JI": 30, "RE": 48, "PR": 42, "CE": 50, "TL": 25, "ACF": 35, "FG": 38, "LTS": 35},
    "EC": {"JI": 35, "RE": 38, "PR": 42, "CE": 40, "TL": 35, "ACF": 30, "FG": 28, "LTS": 30},
    "PH": {"JI": 42, "RE": 40, "PR": 45, "CE": 42, "TL": 38, "ACF": 35, "FG": 30, "LTS": 32},
    "CN": {"JI": 30, "RE": 60, "PR": 45, "CE": 62, "TL": 20, "ACF": 42, "FG": 45, "LTS": 38},
    "ET": {"JI": 32, "RE": 35, "PR": 35, "CE": 32, "TL": 28, "ACF": 35, "FG": 30, "LTS": 28},
    "TZ": {"JI": 38, "RE": 40, "PR": 38, "CE": 35, "TL": 32, "ACF": 35, "FG": 35, "LTS": 32},
    # Weak institutions (25 <= composite < 50)
    "CI": {"JI": 30, "RE": 28, "PR": 32, "CE": 30, "TL": 25, "ACF": 25, "FG": 22, "LTS": 20},
    "CM": {"JI": 22, "RE": 25, "PR": 28, "CE": 25, "TL": 20, "ACF": 20, "FG": 18, "LTS": 18},
    "NG": {"JI": 28, "RE": 25, "PR": 30, "CE": 28, "TL": 22, "ACF": 22, "FG": 15, "LTS": 18},
    "KH": {"JI": 18, "RE": 22, "PR": 25, "CE": 28, "TL": 15, "ACF": 18, "FG": 20, "LTS": 18},
    "MM": {"JI": 12, "RE": 15, "PR": 18, "CE": 15, "TL": 10, "ACF": 12, "FG": 10, "LTS": 12},
    "LA": {"JI": 15, "RE": 22, "PR": 20, "CE": 25, "TL": 12, "ACF": 18, "FG": 18, "LTS": 15},
    "PG": {"JI": 28, "RE": 22, "PR": 25, "CE": 22, "TL": 20, "ACF": 18, "FG": 15, "LTS": 18},
    "HN": {"JI": 22, "RE": 25, "PR": 28, "CE": 25, "TL": 20, "ACF": 18, "FG": 15, "LTS": 18},
    "GT": {"JI": 20, "RE": 22, "PR": 25, "CE": 22, "TL": 18, "ACF": 15, "FG": 12, "LTS": 15},
    "NI": {"JI": 15, "RE": 18, "PR": 20, "CE": 18, "TL": 12, "ACF": 12, "FG": 10, "LTS": 12},
    "PY": {"JI": 25, "RE": 28, "PR": 30, "CE": 28, "TL": 22, "ACF": 22, "FG": 18, "LTS": 22},
    "BO": {"JI": 22, "RE": 25, "PR": 28, "CE": 25, "TL": 20, "ACF": 20, "FG": 18, "LTS": 20},
    "UG": {"JI": 25, "RE": 28, "PR": 25, "CE": 22, "TL": 20, "ACF": 22, "FG": 18, "LTS": 15},
    "CD": {"JI": 10, "RE": 12, "PR": 12, "CE": 10, "TL": 8, "ACF": 10, "FG": 8, "LTS": 8},
    "MG": {"JI": 22, "RE": 20, "PR": 22, "CE": 20, "TL": 18, "ACF": 18, "FG": 15, "LTS": 15},
    "MZ": {"JI": 22, "RE": 20, "PR": 20, "CE": 18, "TL": 18, "ACF": 18, "FG": 15, "LTS": 12},
    "BD": {"JI": 25, "RE": 22, "PR": 25, "CE": 22, "TL": 18, "ACF": 20, "FG": 15, "LTS": 18},
    "PK": {"JI": 28, "RE": 25, "PR": 28, "CE": 25, "TL": 22, "ACF": 22, "FG": 15, "LTS": 18},
    "RU": {"JI": 22, "RE": 35, "PR": 30, "CE": 35, "TL": 18, "ACF": 22, "FG": 25, "LTS": 28},
    "SL": {"JI": 28, "RE": 25, "PR": 28, "CE": 25, "TL": 22, "ACF": 22, "FG": 18, "LTS": 20},
    "LR": {"JI": 22, "RE": 20, "PR": 22, "CE": 20, "TL": 18, "ACF": 18, "FG": 12, "LTS": 15},
    "GN": {"JI": 18, "RE": 18, "PR": 20, "CE": 18, "TL": 15, "ACF": 15, "FG": 12, "LTS": 12},
    "CF": {"JI": 10, "RE": 10, "PR": 12, "CE": 10, "TL": 8, "ACF": 8, "FG": 6, "LTS": 6},
    "CG": {"JI": 12, "RE": 15, "PR": 15, "CE": 12, "TL": 10, "ACF": 10, "FG": 8, "LTS": 10},
    "GA": {"JI": 25, "RE": 28, "PR": 28, "CE": 25, "TL": 22, "ACF": 22, "FG": 20, "LTS": 22},
    # Very Weak institutions (composite < 25)
    "VE": {"JI": 8, "RE": 10, "PR": 10, "CE": 12, "TL": 5, "ACF": 8, "FG": 5, "LTS": 8},
    "SO": {"JI": 5, "RE": 5, "PR": 8, "CE": 5, "TL": 3, "ACF": 5, "FG": 3, "LTS": 5},
    "SS": {"JI": 5, "RE": 5, "PR": 5, "CE": 5, "TL": 3, "ACF": 3, "FG": 3, "LTS": 3},
    "SY": {"JI": 8, "RE": 8, "PR": 10, "CE": 8, "TL": 5, "ACF": 5, "FG": 5, "LTS": 5},
    "YE": {"JI": 8, "RE": 8, "PR": 10, "CE": 8, "TL": 5, "ACF": 5, "FG": 5, "LTS": 5},
    "AF": {"JI": 8, "RE": 10, "PR": 10, "CE": 8, "TL": 5, "ACF": 8, "FG": 5, "LTS": 5},
    "SD": {"JI": 10, "RE": 12, "PR": 12, "CE": 10, "TL": 8, "ACF": 8, "FG": 5, "LTS": 8},
}


# ---------------------------------------------------------------------------
# Forest Governance Reference Data
# ---------------------------------------------------------------------------
# Format: country_code -> (legal_framework, enforcement_capacity,
#          monitoring_capability, indigenous_rights, community_participation,
#          illegal_logging_prevalence, concession_transparency,
#          redd_readiness, protected_area_mgmt)
# ---------------------------------------------------------------------------

FOREST_GOVERNANCE_DATA: Dict[str, Tuple[int, float, float, int, int, str, int, int, int]] = {
    "DK": (92, 0.90, 0.95, 85, 80, "LOW", 88, 75, 90),
    "FI": (95, 0.92, 0.95, 90, 85, "LOW", 90, 80, 92),
    "NZ": (88, 0.85, 0.90, 85, 75, "LOW", 82, 70, 88),
    "DE": (82, 0.80, 0.88, 75, 65, "LOW", 78, 68, 82),
    "BR": (65, 0.35, 0.55, 50, 45, "HIGH", 35, 55, 45),
    "CO": (55, 0.30, 0.40, 45, 40, "HIGH", 30, 45, 40),
    "PE": (50, 0.28, 0.38, 48, 42, "HIGH", 28, 42, 38),
    "ID": (55, 0.32, 0.45, 40, 38, "HIGH", 32, 50, 42),
    "MY": (58, 0.40, 0.50, 35, 30, "MEDIUM", 38, 45, 48),
    "GH": (48, 0.30, 0.35, 42, 40, "HIGH", 28, 35, 35),
    "CI": (35, 0.22, 0.28, 30, 28, "HIGH", 20, 25, 28),
    "CM": (30, 0.18, 0.22, 25, 22, "CRITICAL", 15, 22, 22),
    "NG": (28, 0.15, 0.20, 22, 20, "CRITICAL", 12, 18, 18),
    "CD": (15, 0.08, 0.12, 15, 12, "CRITICAL", 8, 12, 10),
    "CG": (18, 0.10, 0.15, 12, 10, "CRITICAL", 10, 15, 12),
    "GA": (35, 0.25, 0.30, 22, 20, "HIGH", 22, 28, 25),
    "ET": (40, 0.25, 0.30, 32, 28, "HIGH", 22, 30, 30),
    "TZ": (45, 0.30, 0.35, 38, 35, "MEDIUM", 28, 38, 35),
    "UG": (35, 0.20, 0.25, 28, 25, "HIGH", 18, 22, 22),
    "RW": (60, 0.45, 0.50, 35, 40, "MEDIUM", 42, 48, 50),
    "KH": (32, 0.18, 0.22, 20, 18, "CRITICAL", 12, 18, 18),
    "MM": (20, 0.10, 0.15, 15, 12, "CRITICAL", 8, 10, 12),
    "LA": (28, 0.15, 0.20, 18, 15, "HIGH", 12, 18, 18),
    "VN": (48, 0.35, 0.40, 22, 20, "MEDIUM", 30, 35, 38),
    "TH": (45, 0.32, 0.38, 28, 25, "MEDIUM", 30, 32, 35),
    "PH": (40, 0.25, 0.30, 35, 30, "HIGH", 22, 28, 28),
    "PG": (25, 0.12, 0.18, 30, 25, "CRITICAL", 10, 15, 15),
    "HN": (28, 0.15, 0.20, 22, 18, "HIGH", 12, 18, 15),
    "GT": (25, 0.12, 0.18, 30, 25, "HIGH", 10, 15, 15),
    "NI": (20, 0.10, 0.15, 18, 12, "HIGH", 8, 10, 10),
    "MX": (45, 0.28, 0.35, 42, 38, "HIGH", 25, 32, 32),
    "CR": (62, 0.48, 0.55, 55, 50, "LOW", 48, 52, 55),
    "EC": (42, 0.22, 0.30, 42, 35, "HIGH", 20, 28, 28),
    "BO": (35, 0.18, 0.25, 40, 35, "HIGH", 15, 22, 22),
    "PY": (30, 0.15, 0.20, 28, 22, "HIGH", 12, 18, 18),
    "VE": (12, 0.05, 0.08, 10, 8, "CRITICAL", 5, 5, 8),
    "SO": (5, 0.02, 0.03, 5, 3, "CRITICAL", 2, 2, 3),
    "SS": (5, 0.02, 0.03, 5, 3, "CRITICAL", 2, 2, 3),
    "CF": (10, 0.05, 0.08, 10, 8, "CRITICAL", 5, 5, 5),
    "MG": (28, 0.15, 0.20, 25, 22, "HIGH", 12, 18, 18),
    "MZ": (25, 0.12, 0.18, 22, 18, "HIGH", 10, 15, 15),
    "ZA": (58, 0.42, 0.48, 45, 40, "MEDIUM", 40, 42, 45),
    "BW": (55, 0.40, 0.45, 38, 35, "LOW", 38, 35, 42),
    "AR": (42, 0.25, 0.32, 35, 30, "HIGH", 22, 28, 28),
    "RU": (38, 0.22, 0.30, 18, 15, "HIGH", 18, 22, 25),
    "CN": (52, 0.40, 0.55, 15, 12, "MEDIUM", 28, 38, 42),
    "IN": (45, 0.28, 0.35, 40, 38, "HIGH", 22, 30, 32),
}


# ---------------------------------------------------------------------------
# Institutional Quality Engine
# ---------------------------------------------------------------------------


class InstitutionalQualityEngine:
    """Institutional quality assessment engine for EUDR compliance.

    Evaluates institutional frameworks across 8 dimensions with special
    emphasis on forest governance and land tenure security. Provides
    composite institutional quality scoring, forest governance profiling,
    cross-country comparison, and EUDR risk factor computation.

    All arithmetic uses Decimal for deterministic reproducibility and
    every result includes SHA-256 provenance hashes.

    EUDR risk mapping (inverse):
        composite 0   -> risk 1.0  (weakest institutions)
        composite 100 -> risk 0.0  (strongest institutions)
        Formula: eudr_risk = 1.0 - (composite / 100)

    Example::

        engine = InstitutionalQualityEngine()
        result = engine.assess_country_quality("BR")
        assert result.success
        assert result.data.institutional_capacity_level == "WEAK"
    """

    def __init__(
        self,
        weights: Optional[Dict[str, Decimal]] = None,
    ) -> None:
        """Initialize InstitutionalQualityEngine.

        Args:
            weights: Custom dimension weights. Keys must match
                InstitutionalDimension values and sum to ~1.0.
        """
        self._config = None
        self._tracker = None
        self._weights = weights if weights is not None else dict(DEFAULT_DIMENSION_WEIGHTS)

        try:
            if get_config is not None:
                self._config = get_config()
        except Exception:
            pass
        try:
            if get_tracker is not None:
                self._tracker = get_tracker()
        except Exception:
            pass

        logger.info(
            "InstitutionalQualityEngine initialized: version=%s, "
            "countries=%d, forest_profiles=%d",
            _MODULE_VERSION, len(INSTITUTIONAL_DATA),
            len(FOREST_GOVERNANCE_DATA),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_country_quality(
        self,
        country_code: str,
        year: Optional[int] = None,
    ) -> InstitutionalQualityResult:
        """Assess institutional quality for a country.

        Args:
            country_code: ISO alpha-2 country code.
            year: Assessment year. Defaults to 2024.

        Returns:
            InstitutionalQualityResult with all dimension scores,
            composite score, capacity level, and EUDR risk factor.
        """
        start = time.perf_counter()
        timestamp = _utcnow().isoformat()

        try:
            cc = country_code.upper().strip()
            if cc not in INSTITUTIONAL_DATA:
                return InstitutionalQualityResult(
                    success=False,
                    error=f"Country '{cc}' not in institutional quality database.",
                    calculation_timestamp=timestamp,
                )

            if year is None:
                year = 2024

            raw_scores = INSTITUTIONAL_DATA[cc]
            dim_scores: Dict[str, Decimal] = {}
            for dim, score in raw_scores.items():
                dim_scores[dim] = _to_decimal(score)

            composite = self._calculate_composite_institutional_score(dim_scores)
            capacity = self._classify_capacity(composite)

            fg_score = dim_scores.get("FG", Decimal("0"))
            lts_score = dim_scores.get("LTS", Decimal("0"))

            re_score = dim_scores.get("RE", Decimal("0"))
            enforcement = (re_score / Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            eudr_risk = self._map_institutional_to_eudr_risk(composite)

            assessment = InstitutionalAssessment(
                country_code=cc, year=year,
                dimension_scores=dim_scores,
                composite_score=composite,
                forest_governance_score=fg_score,
                land_tenure_score=lts_score,
                enforcement_effectiveness=enforcement,
                institutional_capacity_level=capacity,
            )

            prov_hash = _compute_hash({
                "operation": "assess_country_quality",
                "country_code": cc, "year": year,
                "composite": str(composite),
                "capacity": capacity,
                "eudr_risk": str(eudr_risk),
            })

            elapsed = (time.perf_counter() - start) * 1000
            _inc_inst_assessments()
            _observe_inst_duration(elapsed / 1000)

            self._record_provenance(
                "institutional_quality", "assess_quality", cc,
                {"year": year, "composite": str(composite)},
            )

            return InstitutionalQualityResult(
                success=True, data=assessment,
                eudr_risk_factor=eudr_risk,
                metadata={
                    "engine": "InstitutionalQualityEngine",
                    "engine_version": _MODULE_VERSION,
                    "agent_id": _AGENT_ID,
                    "operation": "assess_country_quality",
                    "processing_time_ms": round(elapsed, 3),
                    "weights": {k: str(v) for k, v in self._weights.items()},
                },
                provenance_hash=prov_hash,
                calculation_timestamp=timestamp,
            )

        except Exception as exc:
            logger.error("assess_country_quality failed: %s", exc, exc_info=True)
            _inc_inst_error("assess_country_quality")
            return InstitutionalQualityResult(
                success=False, error=str(exc),
                calculation_timestamp=timestamp,
            )

    def get_governance_profile(
        self,
        country_code: str,
    ) -> GovernanceProfileResult:
        """Get a comprehensive governance profile with strengths/weaknesses.

        Args:
            country_code: ISO alpha-2 country code.

        Returns:
            GovernanceProfileResult with assessment, identified strengths,
            weaknesses, and recommendations.
        """
        start = time.perf_counter()
        timestamp = _utcnow().isoformat()

        try:
            cc = country_code.upper().strip()
            quality_result = self.assess_country_quality(cc)

            if not quality_result.success:
                return GovernanceProfileResult(
                    success=False,
                    error=quality_result.error,
                    calculation_timestamp=timestamp,
                )

            assessment = quality_result.data
            strengths: List[str] = []
            weaknesses: List[str] = []
            recommendations: List[str] = []

            dim_labels = {
                "JI": "Judicial Independence",
                "RE": "Regulatory Enforcement",
                "PR": "Property Rights",
                "CE": "Contract Enforcement",
                "TL": "Transparency Laws",
                "ACF": "Anti-Corruption Framework",
                "FG": "Forest Governance",
                "LTS": "Land Tenure Security",
            }

            if assessment is not None:
                for dim, score in assessment.dimension_scores.items():
                    label = dim_labels.get(dim, dim)
                    if score >= Decimal("70"):
                        strengths.append(f"{label}: strong ({score}/100)")
                    elif score < Decimal("30"):
                        weaknesses.append(f"{label}: very weak ({score}/100)")
                        recommendations.append(
                            f"Priority: strengthen {label.lower()}"
                        )
                    elif score < Decimal("50"):
                        weaknesses.append(f"{label}: weak ({score}/100)")

                if assessment.forest_governance_score < Decimal("40"):
                    recommendations.append(
                        "Critical: improve forest governance capacity for EUDR compliance"
                    )
                if assessment.land_tenure_score < Decimal("40"):
                    recommendations.append(
                        "Critical: strengthen land tenure security and registration"
                    )

            prov_hash = _compute_hash({
                "operation": "get_governance_profile",
                "country_code": cc,
                "strengths": len(strengths),
                "weaknesses": len(weaknesses),
            })

            elapsed = (time.perf_counter() - start) * 1000
            _inc_inst_assessments()
            _observe_inst_duration(elapsed / 1000)

            return GovernanceProfileResult(
                success=True, country_code=cc,
                assessment=assessment,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                metadata={
                    "engine": "InstitutionalQualityEngine",
                    "engine_version": _MODULE_VERSION,
                    "agent_id": _AGENT_ID,
                    "operation": "get_governance_profile",
                    "processing_time_ms": round(elapsed, 3),
                },
                provenance_hash=prov_hash,
                calculation_timestamp=timestamp,
            )

        except Exception as exc:
            logger.error("get_governance_profile failed: %s", exc, exc_info=True)
            _inc_inst_error("get_governance_profile")
            return GovernanceProfileResult(
                success=False, error=str(exc),
                calculation_timestamp=timestamp,
            )

    def assess_institutional_strength(
        self,
        country_code: str,
        dimensions: Optional[List[str]] = None,
    ) -> StrengthResult:
        """Assess institutional strength for selected dimensions.

        Args:
            country_code: ISO alpha-2 country code.
            dimensions: Optional list of dimension codes to assess.
                Defaults to all 8 dimensions.

        Returns:
            StrengthResult with per-dimension scores and overall
            strength rating.
        """
        start = time.perf_counter()
        timestamp = _utcnow().isoformat()

        try:
            cc = country_code.upper().strip()
            if cc not in INSTITUTIONAL_DATA:
                return StrengthResult(
                    success=False,
                    error=f"Country '{cc}' not in database.",
                    calculation_timestamp=timestamp,
                )

            valid_dims = {d.value for d in InstitutionalDimension}
            if dimensions is None:
                dims = list(valid_dims)
            else:
                dims = [d.upper() for d in dimensions]
                invalid = set(dims) - valid_dims
                if invalid:
                    return StrengthResult(
                        success=False,
                        error=f"Invalid dimensions: {sorted(invalid)}. Valid: {sorted(valid_dims)}",
                        calculation_timestamp=timestamp,
                    )

            raw = INSTITUTIONAL_DATA[cc]
            scores: Dict[str, Decimal] = {}
            for dim in dims:
                if dim in raw:
                    scores[dim] = _to_decimal(raw[dim])

            if not scores:
                return StrengthResult(
                    success=False,
                    error=f"No data for requested dimensions.",
                    calculation_timestamp=timestamp,
                )

            avg = sum(scores.values()) / _to_decimal(len(scores))
            avg = avg.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            capacity = self._classify_capacity(avg)

            prov_hash = _compute_hash({
                "operation": "assess_institutional_strength",
                "country_code": cc, "dimensions": sorted(dims),
                "overall": str(avg),
            })

            elapsed = (time.perf_counter() - start) * 1000
            _inc_inst_assessments()
            _observe_inst_duration(elapsed / 1000)

            return StrengthResult(
                success=True, country_code=cc,
                dimensions_assessed=sorted(dims),
                dimension_scores=scores,
                overall_strength=avg,
                capacity_level=capacity,
                metadata={
                    "engine": "InstitutionalQualityEngine",
                    "engine_version": _MODULE_VERSION,
                    "agent_id": _AGENT_ID,
                    "operation": "assess_institutional_strength",
                    "processing_time_ms": round(elapsed, 3),
                },
                provenance_hash=prov_hash,
                calculation_timestamp=timestamp,
            )

        except Exception as exc:
            logger.error("assess_institutional_strength failed: %s", exc, exc_info=True)
            _inc_inst_error("assess_institutional_strength")
            return StrengthResult(
                success=False, error=str(exc),
                calculation_timestamp=timestamp,
            )

    def get_forest_governance(
        self,
        country_code: str,
    ) -> ForestGovernanceResult:
        """Get forest-specific governance assessment.

        Args:
            country_code: ISO alpha-2 country code.

        Returns:
            ForestGovernanceResult with detailed forest governance
            profile, composite forest score, and EUDR risk factors.
        """
        start = time.perf_counter()
        timestamp = _utcnow().isoformat()

        try:
            cc = country_code.upper().strip()
            if cc not in FOREST_GOVERNANCE_DATA:
                return ForestGovernanceResult(
                    success=False,
                    error=f"No forest governance data for '{cc}'.",
                    calculation_timestamp=timestamp,
                )

            (legal, enforce, monitor, indigenous, community,
             logging_prev, concession, redd, protected) = FOREST_GOVERNANCE_DATA[cc]

            profile = ForestGovernanceProfile(
                country_code=cc,
                legal_framework_score=_to_decimal(legal),
                enforcement_capacity=_to_decimal(enforce),
                monitoring_capability=_to_decimal(monitor),
                indigenous_rights_protection=_to_decimal(indigenous),
                community_participation=_to_decimal(community),
                illegal_logging_prevalence=logging_prev,
                concession_transparency=_to_decimal(concession),
                redd_plus_readiness=_to_decimal(redd),
                protected_area_management=_to_decimal(protected),
            )

            # Composite forest score: weighted average of key metrics
            # legal (25%), enforcement (20%), monitoring (20%),
            # indigenous (10%), community (10%), concession (15%)
            composite_forest = (
                _to_decimal(legal) * Decimal("0.25") +
                _to_decimal(enforce) * Decimal("100") * Decimal("0.20") +
                _to_decimal(monitor) * Decimal("100") * Decimal("0.20") +
                _to_decimal(indigenous) * Decimal("0.10") +
                _to_decimal(community) * Decimal("0.10") +
                _to_decimal(concession) * Decimal("0.15")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            composite_forest = max(Decimal("0"), min(Decimal("100"), composite_forest))
            eudr_risk = self._map_institutional_to_eudr_risk(composite_forest)

            # Identify risk factors
            risk_factors: List[str] = []
            if logging_prev in ("HIGH", "CRITICAL"):
                risk_factors.append(
                    f"Illegal logging prevalence: {logging_prev}"
                )
            if enforce < 0.3:
                risk_factors.append(
                    f"Low enforcement capacity: {enforce:.0%}"
                )
            if monitor < 0.3:
                risk_factors.append(
                    f"Weak monitoring capability: {monitor:.0%}"
                )
            if legal < 40:
                risk_factors.append(
                    f"Incomplete legal framework: {legal}/100"
                )
            if indigenous < 30:
                risk_factors.append(
                    f"Weak indigenous rights protection: {indigenous}/100"
                )
            if concession < 25:
                risk_factors.append(
                    f"Low concession transparency: {concession}/100"
                )

            prov_hash = _compute_hash({
                "operation": "get_forest_governance",
                "country_code": cc,
                "composite_forest": str(composite_forest),
                "eudr_risk": str(eudr_risk),
                "logging_prevalence": logging_prev,
            })

            elapsed = (time.perf_counter() - start) * 1000
            _inc_forest_queries()
            _observe_inst_duration(elapsed / 1000)

            self._record_provenance(
                "forest_governance", "query_governance", cc,
                {"composite": str(composite_forest), "logging": logging_prev},
            )

            return ForestGovernanceResult(
                success=True, data=profile,
                composite_forest_score=composite_forest,
                eudr_risk_factor=eudr_risk,
                risk_factors=risk_factors,
                metadata={
                    "engine": "InstitutionalQualityEngine",
                    "engine_version": _MODULE_VERSION,
                    "agent_id": _AGENT_ID,
                    "operation": "get_forest_governance",
                    "processing_time_ms": round(elapsed, 3),
                },
                provenance_hash=prov_hash,
                calculation_timestamp=timestamp,
            )

        except Exception as exc:
            logger.error("get_forest_governance failed: %s", exc, exc_info=True)
            _inc_inst_error("get_forest_governance")
            return ForestGovernanceResult(
                success=False, error=str(exc),
                calculation_timestamp=timestamp,
            )

    def compare_countries(
        self,
        country_codes: List[str],
        dimensions: Optional[List[str]] = None,
    ) -> ComparisonResult:
        """Compare institutional quality across multiple countries.

        Args:
            country_codes: List of ISO alpha-2 country codes.
            dimensions: Optional dimension filter. Defaults to all.

        Returns:
            ComparisonResult with per-country results, dimension
            rankings, and composite rankings.
        """
        start = time.perf_counter()
        timestamp = _utcnow().isoformat()

        try:
            if not country_codes:
                return ComparisonResult(
                    success=False,
                    error="country_codes must not be empty.",
                    calculation_timestamp=timestamp,
                )

            countries: Dict[str, InstitutionalQualityResult] = {}
            warnings: List[str] = []

            for cc in country_codes:
                result = self.assess_country_quality(cc)
                countries[cc.upper().strip()] = result
                if not result.success:
                    warnings.append(f"Failed for {cc}: {result.error}")

            # Build dimension rankings
            valid_dims = {d.value for d in InstitutionalDimension}
            dims_to_rank = (
                [d.upper() for d in dimensions] if dimensions
                else list(valid_dims)
            )

            dim_rankings: Dict[str, List[Tuple[str, Decimal]]] = {}
            for dim in dims_to_rank:
                entries: List[Tuple[str, Decimal]] = []
                for cc, res in countries.items():
                    if res.success and res.data and dim in res.data.dimension_scores:
                        entries.append((cc, res.data.dimension_scores[dim]))
                entries.sort(key=lambda x: x[1], reverse=True)
                dim_rankings[dim] = entries

            # Composite rankings
            composite_rankings: List[Tuple[str, Decimal]] = []
            for cc, res in countries.items():
                if res.success and res.data:
                    composite_rankings.append((cc, res.data.composite_score))
            composite_rankings.sort(key=lambda x: x[1], reverse=True)

            prov_hash = _compute_hash({
                "operation": "compare_countries",
                "country_codes": sorted([c.upper() for c in country_codes]),
                "dimensions": sorted(dims_to_rank),
                "successful": sum(1 for r in countries.values() if r.success),
            })

            elapsed = (time.perf_counter() - start) * 1000
            _inc_inst_assessments()
            _observe_inst_duration(elapsed / 1000)

            return ComparisonResult(
                success=True,
                countries=countries,
                dimension_rankings=dim_rankings,
                composite_rankings=composite_rankings,
                metadata={
                    "engine": "InstitutionalQualityEngine",
                    "engine_version": _MODULE_VERSION,
                    "agent_id": _AGENT_ID,
                    "operation": "compare_countries",
                    "processing_time_ms": round(elapsed, 3),
                },
                provenance_hash=prov_hash,
                calculation_timestamp=timestamp,
                warnings=warnings,
            )

        except Exception as exc:
            logger.error("compare_countries failed: %s", exc, exc_info=True)
            _inc_inst_error("compare_countries")
            return ComparisonResult(
                success=False, error=str(exc),
                calculation_timestamp=timestamp,
            )

    # ------------------------------------------------------------------
    # Core calculation methods
    # ------------------------------------------------------------------

    def _calculate_composite_institutional_score(
        self,
        dimension_scores: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate weighted composite institutional quality score.

        Args:
            dimension_scores: Dictionary of dimension code to score.

        Returns:
            Composite score as Decimal (0-100), rounded to 2 places.
        """
        weighted_sum = Decimal("0")
        weight_total = Decimal("0")

        for dim, score in dimension_scores.items():
            weight = self._weights.get(dim, Decimal("0"))
            weighted_sum += score * weight
            weight_total += weight

        if weight_total == Decimal("0"):
            return Decimal("0")

        composite = weighted_sum / weight_total
        composite = max(Decimal("0"), min(Decimal("100"), composite))
        return composite.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _map_institutional_to_eudr_risk(
        self,
        composite_score: Decimal,
    ) -> Decimal:
        """Map institutional composite score to EUDR risk factor.

        Args:
            composite_score: Institutional composite (0-100).

        Returns:
            EUDR risk factor [0, 1]. Lower quality = higher risk.
        """
        clamped = max(Decimal("0"), min(Decimal("100"), composite_score))
        risk = Decimal("1.0") - (clamped / Decimal("100"))
        return risk.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def get_eudr_risk_factor(
        self,
        country_code: str,
    ) -> Decimal:
        """Convenience: get EUDR risk from institutional quality.

        Args:
            country_code: ISO alpha-2 country code.

        Returns:
            EUDR risk as Decimal [0, 1].
            Returns Decimal('1.0') if data not found (precautionary).
        """
        result = self.assess_country_quality(country_code)
        if not result.success:
            return Decimal("1.0000")
        return result.eudr_risk_factor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_capacity(self, composite: Decimal) -> str:
        """Classify institutional capacity from composite score.

        Args:
            composite: Composite score (0-100).

        Returns:
            Capacity level string.
        """
        if composite >= Decimal("75"):
            return InstitutionalCapacityLevel.STRONG.value
        elif composite >= Decimal("50"):
            return InstitutionalCapacityLevel.ADEQUATE.value
        elif composite >= Decimal("25"):
            return InstitutionalCapacityLevel.WEAK.value
        else:
            return InstitutionalCapacityLevel.VERY_WEAK.value

    def _record_provenance(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record provenance entry if tracker available."""
        if self._tracker is None:
            return
        try:
            self._tracker.record(
                entity_type, action, entity_id, metadata=data,
            )
        except Exception as exc:
            logger.debug("Provenance recording failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "InstitutionalDimension",
    "InstitutionalCapacityLevel",
    "IllegalLoggingPrevalence",
    # Data classes
    "InstitutionalAssessment",
    "ForestGovernanceProfile",
    "InstitutionalQualityResult",
    "GovernanceProfileResult",
    "StrengthResult",
    "ForestGovernanceResult",
    "ComparisonResult",
    # Engine
    "InstitutionalQualityEngine",
    # Reference data
    "INSTITUTIONAL_DATA",
    "FOREST_GOVERNANCE_DATA",
    "DEFAULT_DIMENSION_WEIGHTS",
]
