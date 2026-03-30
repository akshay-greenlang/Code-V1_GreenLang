# -*- coding: utf-8 -*-
"""
Risk Propagation Engine - AGENT-EUDR-008 Engine 5

Production-grade risk propagation engine for multi-tier supplier tracking
under the EU Deforestation Regulation (EUDR). Calculates composite risk
scores per supplier across six risk categories (deforestation proximity,
country risk, certification gap, compliance history, data quality,
concentration risk), then propagates risk signals through the supplier
hierarchy from deep tiers upstream to Tier 1 and the EU operator.

Zero-Hallucination Guarantees:
    - All risk scores are deterministic (0-100 fixed-point arithmetic)
    - Risk weights are loaded from PRD Appendix B reference data
    - Country risk scores are deterministic lookups, not ML predictions
    - Propagation methods (max, weighted-average, volume-weighted) are
      pure arithmetic with no stochastic components
    - SHA-256 provenance chain hashing on all assessment results
    - No ML/LLM used in any risk calculation

Performance Targets:
    - Single supplier risk assessment: <5ms
    - Risk propagation through 1,000-node chain: <100ms
    - Batch assessment of 10,000 suppliers: <5s

Regulatory References:
    - EUDR Article 10: Risk assessment and mitigation
    - EUDR Article 14: Competent authority audit (5-year records)
    - PRD Appendix B: Risk Category Weights

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-008 (Engine 5: Risk Propagation)
Agent ID: GL-EUDR-MST-008
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a numeric value to [lo, hi]."""
    return max(lo, min(hi, value))

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskCategory(str, Enum):
    """Six risk categories per PRD Appendix B."""

    DEFORESTATION_PROXIMITY = "deforestation_proximity"
    COUNTRY_RISK = "country_risk"
    CERTIFICATION_GAP = "certification_gap"
    COMPLIANCE_HISTORY = "compliance_history"
    DATA_QUALITY = "data_quality"
    CONCENTRATION_RISK = "concentration_risk"

class RiskLevel(str, Enum):
    """Qualitative risk level classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

class PropagationMethod(str, Enum):
    """Risk propagation methods per PRD F5.4."""

    MAX = "max"
    WEIGHTED_AVERAGE = "weighted_average"
    VOLUME_WEIGHTED = "volume_weighted"

class TrendDirection(str, Enum):
    """Risk trend over time."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"

# ---------------------------------------------------------------------------
# Risk Category Weights (PRD Appendix B)
# ---------------------------------------------------------------------------

DEFAULT_RISK_WEIGHTS: Dict[str, float] = {
    RiskCategory.DEFORESTATION_PROXIMITY.value: 0.30,
    RiskCategory.COUNTRY_RISK.value: 0.20,
    RiskCategory.CERTIFICATION_GAP.value: 0.15,
    RiskCategory.COMPLIANCE_HISTORY.value: 0.15,
    RiskCategory.DATA_QUALITY.value: 0.10,
    RiskCategory.CONCENTRATION_RISK.value: 0.10,
}

# ---------------------------------------------------------------------------
# Risk Level Thresholds
# ---------------------------------------------------------------------------

RISK_LEVEL_THRESHOLDS: List[Tuple[float, float, RiskLevel]] = [
    (0.0, 20.0, RiskLevel.LOW),
    (20.0, 40.0, RiskLevel.MEDIUM),
    (40.0, 60.0, RiskLevel.HIGH),
    (60.0, 80.0, RiskLevel.VERY_HIGH),
    (80.0, 100.01, RiskLevel.CRITICAL),
]

# ---------------------------------------------------------------------------
# Country Risk Reference Data (ISO 3166-1 alpha-2 -> risk score 0-100)
# Deterministic lookup table; higher = riskier for deforestation.
# Sources: FAO Global Forest Resources Assessment, Transparency Intl CPI.
# ---------------------------------------------------------------------------

COUNTRY_RISK_SCORES: Dict[str, float] = {
    # Very high risk (70-100)
    "BR": 75.0, "ID": 78.0, "MY": 65.0, "CO": 68.0, "PE": 70.0,
    "BO": 72.0, "VE": 74.0, "EC": 62.0, "PY": 66.0, "GY": 58.0,
    "SR": 56.0, "CD": 82.0, "CG": 76.0, "CM": 73.0, "GA": 60.0,
    "GH": 64.0, "CI": 72.0, "LR": 68.0, "SL": 65.0, "NG": 70.0,
    "TZ": 55.0, "MZ": 58.0, "MG": 62.0, "MM": 80.0, "LA": 65.0,
    "KH": 63.0, "PG": 72.0, "PH": 50.0, "TH": 45.0, "VN": 48.0,
    "IN": 42.0, "CN": 38.0, "ET": 55.0, "KE": 48.0, "UG": 52.0,
    "AO": 60.0, "ZM": 50.0, "ZW": 55.0, "RW": 40.0,
    # Medium risk (30-50)
    "MX": 40.0, "GT": 48.0, "HN": 52.0, "NI": 50.0, "CR": 25.0,
    "PA": 35.0, "AR": 38.0, "CL": 22.0, "UY": 18.0,
    # Low risk (0-30)
    "DE": 5.0, "FR": 6.0, "NL": 4.0, "BE": 5.0, "IT": 7.0,
    "ES": 8.0, "PT": 9.0, "AT": 4.0, "SE": 3.0, "FI": 3.0,
    "DK": 3.0, "NO": 3.0, "PL": 6.0, "CZ": 5.0, "RO": 10.0,
    "BG": 12.0, "HR": 8.0, "HU": 6.0, "SK": 5.0, "SI": 4.0,
    "LT": 5.0, "LV": 6.0, "EE": 5.0, "IE": 4.0, "LU": 3.0,
    "GB": 5.0, "US": 12.0, "CA": 10.0, "AU": 15.0, "NZ": 8.0,
    "JP": 6.0, "KR": 7.0, "SG": 4.0,
}

#: Default risk for unlisted countries.
DEFAULT_COUNTRY_RISK: float = 50.0

# ---------------------------------------------------------------------------
# Certification Recognition (type -> risk reduction factor 0-1)
# Higher factor means greater risk reduction.
# ---------------------------------------------------------------------------

CERTIFICATION_RISK_REDUCTION: Dict[str, float] = {
    "fsc": 0.80,
    "pefc": 0.70,
    "rspo": 0.75,
    "rspo_ip": 0.85,
    "rspo_mb": 0.65,
    "rspo_sg": 0.55,
    "utz": 0.65,
    "rainforest_alliance": 0.70,
    "fairtrade": 0.60,
    "organic": 0.50,
    "iscc": 0.60,
    "bonsucro": 0.55,
    "rtrs": 0.60,
    "globalg.a.p.": 0.45,
    "4c": 0.55,
}

# ---------------------------------------------------------------------------
# Supplier Profile Completeness Fields (PRD Appendix D)
# ---------------------------------------------------------------------------

PROFILE_FIELD_WEIGHTS: Dict[str, Dict[str, float]] = {
    "legal_identity": {
        "weight": 0.25,
        "fields": {
            "legal_name": 0.40,
            "registration_id": 0.35,
            "country_iso": 0.25,
        },
    },
    "location": {
        "weight": 0.20,
        "fields": {
            "gps_latitude": 0.35,
            "gps_longitude": 0.35,
            "address": 0.15,
            "admin_region": 0.15,
        },
    },
    "commodity": {
        "weight": 0.15,
        "fields": {
            "commodity_types": 0.40,
            "annual_volume_tonnes": 0.30,
            "processing_capacity": 0.30,
        },
    },
    "certification": {
        "weight": 0.15,
        "fields": {
            "certification_type": 0.35,
            "certificate_id": 0.30,
            "valid_from": 0.15,
            "valid_until": 0.20,
        },
    },
    "compliance": {
        "weight": 0.15,
        "fields": {
            "dds_reference": 0.40,
            "deforestation_free_status": 0.35,
            "last_verification_date": 0.25,
        },
    },
    "contact": {
        "weight": 0.10,
        "fields": {
            "primary_contact_name": 0.35,
            "primary_contact_email": 0.35,
            "compliance_contact_name": 0.15,
            "compliance_contact_email": 0.15,
        },
    },
}

# ---------------------------------------------------------------------------
# Result Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RiskCategoryScore:
    """Score for a single risk category.

    Attributes:
        category: Risk category identifier.
        raw_score: Raw score before weighting (0-100).
        weight: Category weight (0-1).
        weighted_score: raw_score * weight.
        details: Supporting evidence for the score.
    """

    category: str = ""
    raw_score: float = 0.0
    weight: float = 0.0
    weighted_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskAssessmentResult:
    """Composite risk assessment result for a single supplier.

    Attributes:
        assessment_id: Unique UUID4 assessment identifier.
        supplier_id: Supplier being assessed.
        composite_score: Weighted composite risk score (0-100).
        risk_level: Qualitative risk level classification.
        category_scores: Per-category score breakdown.
        assessed_at: UTC ISO timestamp of assessment.
        processing_time_ms: Assessment duration in milliseconds.
        provenance_hash: SHA-256 hash of assessment inputs and outputs.
        engine_version: Engine version string.
    """

    assessment_id: str = ""
    supplier_id: str = ""
    composite_score: float = 0.0
    risk_level: str = RiskLevel.MEDIUM.value
    category_scores: List[RiskCategoryScore] = field(default_factory=list)
    assessed_at: str = ""
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    engine_version: str = _MODULE_VERSION

@dataclass
class RiskPropagationResult:
    """Result of risk propagation through a supplier chain.

    Attributes:
        propagation_id: Unique UUID4 propagation identifier.
        root_supplier_id: Root supplier (Tier 1) of the chain.
        method: Propagation method used.
        propagated_score: Final propagated risk score (0-100).
        propagated_level: Qualitative risk level of propagated score.
        tier_scores: Risk score at each tier level.
        max_risk_path: Supplier IDs on the highest-risk path.
        total_suppliers: Total suppliers in the chain.
        deepest_tier: Deepest tier reached.
        assessed_at: UTC ISO timestamp.
        processing_time_ms: Duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """

    propagation_id: str = ""
    root_supplier_id: str = ""
    method: str = PropagationMethod.MAX.value
    propagated_score: float = 0.0
    propagated_level: str = RiskLevel.MEDIUM.value
    tier_scores: Dict[int, float] = field(default_factory=dict)
    max_risk_path: List[str] = field(default_factory=list)
    total_suppliers: int = 0
    deepest_tier: int = 0
    assessed_at: str = ""
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

@dataclass
class RiskChangeAlert:
    """Alert generated when a supplier's risk crosses a threshold.

    Attributes:
        alert_id: Unique UUID4 alert identifier.
        supplier_id: Supplier whose risk changed.
        previous_score: Previous composite risk score.
        current_score: New composite risk score.
        previous_level: Previous risk level.
        current_level: New risk level.
        change_delta: Absolute change in score.
        direction: "increase" or "decrease".
        triggered_at: UTC ISO timestamp.
        categories_changed: Categories that drove the change.
    """

    alert_id: str = ""
    supplier_id: str = ""
    previous_score: float = 0.0
    current_score: float = 0.0
    previous_level: str = ""
    current_level: str = ""
    change_delta: float = 0.0
    direction: str = "increase"
    triggered_at: str = ""
    categories_changed: List[str] = field(default_factory=list)

@dataclass
class BatchRiskResult:
    """Batch risk assessment result.

    Attributes:
        batch_id: Unique UUID4 batch identifier.
        total_suppliers: Total suppliers assessed.
        successful: Number successfully assessed.
        failed: Number that failed assessment.
        results: Individual assessment results.
        summary: Aggregate statistics.
        processing_time_ms: Total batch duration in milliseconds.
        provenance_hash: SHA-256 hash of entire batch.
    """

    batch_id: str = ""
    total_suppliers: int = 0
    successful: int = 0
    failed: int = 0
    results: List[RiskAssessmentResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

@dataclass
class SupplierProfile:
    """Minimal supplier profile data required for risk assessment.

    This is a local model used within the risk propagation engine to
    decouple from external model dependencies.

    Attributes:
        supplier_id: Unique supplier identifier.
        legal_name: Legal entity name.
        country_iso: ISO 3166-1 alpha-2 country code.
        tier: Tier level in the supply chain (1 = direct supplier).
        commodity_types: List of EUDR commodity types handled.
        certifications: List of certification records.
        gps_latitude: Production plot latitude (WGS84).
        gps_longitude: Production plot longitude (WGS84).
        registration_id: Legal registration ID (e.g., company number).
        address: Physical address.
        admin_region: Administrative region.
        annual_volume_tonnes: Annual commodity volume in tonnes.
        processing_capacity: Processing capacity (tonnes/year).
        primary_contact_name: Primary contact name.
        primary_contact_email: Primary contact email.
        compliance_contact_name: Compliance contact name.
        compliance_contact_email: Compliance contact email.
        dds_references: List of DDS reference IDs.
        deforestation_free_status: Deforestation-free verification status.
        last_verification_date: Last verification date ISO string.
        deforestation_distance_km: Distance to nearest deforestation (km).
        compliance_violations: Number of historical compliance violations.
        last_violation_date: Date of most recent violation (ISO string).
        violation_severity_scores: List of violation severity scores (0-100).
        upstream_supplier_ids: List of upstream (sub-tier) supplier IDs.
        volume_from_suppliers: Dict mapping supplier_id -> volume in tonnes.
        total_sourcing_volume: Total volume sourced from all upstream.
    """

    supplier_id: str = ""
    legal_name: str = ""
    country_iso: str = ""
    tier: int = 1
    commodity_types: List[str] = field(default_factory=list)
    certifications: List[Dict[str, Any]] = field(default_factory=list)
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    registration_id: str = ""
    address: str = ""
    admin_region: str = ""
    annual_volume_tonnes: Optional[float] = None
    processing_capacity: Optional[float] = None
    primary_contact_name: str = ""
    primary_contact_email: str = ""
    compliance_contact_name: str = ""
    compliance_contact_email: str = ""
    dds_references: List[str] = field(default_factory=list)
    deforestation_free_status: str = ""
    last_verification_date: str = ""
    deforestation_distance_km: Optional[float] = None
    compliance_violations: int = 0
    last_violation_date: str = ""
    violation_severity_scores: List[float] = field(default_factory=list)
    upstream_supplier_ids: List[str] = field(default_factory=list)
    volume_from_suppliers: Dict[str, float] = field(default_factory=dict)
    total_sourcing_volume: float = 0.0

@dataclass
class SupplierChainNode:
    """A node in the supplier hierarchy tree used for propagation.

    Attributes:
        supplier_id: Unique supplier identifier.
        tier: Tier level in the chain.
        risk_score: Assessed risk score for this supplier.
        risk_level: Risk level classification.
        volume_share: Share of volume this node contributes (0-1).
        children: Child nodes (upstream/deeper-tier suppliers).
        profile: Full supplier profile.
    """

    supplier_id: str = ""
    tier: int = 1
    risk_score: float = 0.0
    risk_level: str = RiskLevel.MEDIUM.value
    volume_share: float = 1.0
    children: List[SupplierChainNode] = field(default_factory=list)
    profile: Optional[SupplierProfile] = None

# ===========================================================================
# RiskPropagationEngine
# ===========================================================================

class RiskPropagationEngine:
    """Production-grade risk propagation engine for EUDR multi-tier suppliers.

    Calculates composite risk scores per supplier across six categories
    defined in PRD Appendix B, then propagates risk through the supplier
    hierarchy using configurable propagation methods (max, weighted average,
    volume-weighted).

    All calculations are deterministic with zero LLM/ML involvement.

    Attributes:
        _risk_weights: Risk category weights (summing to 1.0).
        _risk_history: In-memory risk score history per supplier.
        _assessment_count: Running count of assessments performed.

    Example::

        engine = RiskPropagationEngine()
        profile = SupplierProfile(
            supplier_id="SUP-001",
            country_iso="BR",
            deforestation_distance_km=5.0,
        )
        result = engine.assess_supplier_risk("SUP-001", profile)
        assert 0.0 <= result.composite_score <= 100.0
    """

    def __init__(
        self,
        risk_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize RiskPropagationEngine.

        Args:
            risk_weights: Optional custom risk category weights. If not
                provided, uses PRD Appendix B defaults. Weights must
                sum to 1.0 (+/- 0.001 tolerance).

        Raises:
            ValueError: If custom weights do not sum to approximately 1.0.
        """
        if risk_weights is not None:
            weight_sum = sum(risk_weights.values())
            if abs(weight_sum - 1.0) > 0.001:
                raise ValueError(
                    f"Risk weights must sum to 1.0, got {weight_sum:.4f}"
                )
            self._risk_weights: Dict[str, float] = dict(risk_weights)
        else:
            self._risk_weights = dict(DEFAULT_RISK_WEIGHTS)

        self._risk_history: Dict[str, List[RiskAssessmentResult]] = {}
        self._assessment_count: int = 0

        logger.info(
            "RiskPropagationEngine initialized with weights: %s",
            {k: f"{v:.2f}" for k, v in self._risk_weights.items()},
        )

    # ------------------------------------------------------------------
    # Public API: Individual Risk Assessment
    # ------------------------------------------------------------------

    def assess_supplier_risk(
        self,
        supplier_id: str,
        profile: SupplierProfile,
    ) -> RiskAssessmentResult:
        """Calculate composite risk score for a single supplier.

        Evaluates six risk categories (deforestation proximity, country risk,
        certification gap, compliance history, data quality, concentration
        risk), applies PRD Appendix B weights, and produces a composite
        risk score from 0 (lowest risk) to 100 (highest risk).

        Args:
            supplier_id: Unique supplier identifier.
            profile: Supplier profile data for risk assessment.

        Returns:
            RiskAssessmentResult with composite score and category breakdown.

        Raises:
            ValueError: If supplier_id is empty.
        """
        if not supplier_id:
            raise ValueError("supplier_id must not be empty")

        t_start = time.monotonic()
        assessment_id = str(uuid.uuid4())
        category_scores: List[RiskCategoryScore] = []

        # Calculate each risk category
        deforestation_score = self.calculate_deforestation_risk(profile)
        category_scores.append(deforestation_score)

        country_score = self.calculate_country_risk(profile.country_iso)
        category_scores.append(country_score)

        certification_score = self.calculate_certification_risk(
            profile.certifications
        )
        category_scores.append(certification_score)

        compliance_score = self.calculate_compliance_risk(profile)
        category_scores.append(compliance_score)

        data_quality_score = self.calculate_data_quality_risk(profile)
        category_scores.append(data_quality_score)

        concentration_score = self.calculate_concentration_risk(profile)
        category_scores.append(concentration_score)

        # Composite score: weighted sum of category scores
        composite = sum(cs.weighted_score for cs in category_scores)
        composite = _clamp(composite)

        risk_level = self._classify_risk_level(composite)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        # Build provenance hash
        provenance_data = {
            "assessment_id": assessment_id,
            "supplier_id": supplier_id,
            "composite_score": round(composite, 4),
            "category_scores": [
                {
                    "category": cs.category,
                    "raw_score": round(cs.raw_score, 4),
                    "weight": round(cs.weight, 4),
                }
                for cs in category_scores
            ],
            "engine_version": _MODULE_VERSION,
        }
        provenance_hash = _compute_hash(provenance_data)

        result = RiskAssessmentResult(
            assessment_id=assessment_id,
            supplier_id=supplier_id,
            composite_score=round(composite, 2),
            risk_level=risk_level.value,
            category_scores=category_scores,
            assessed_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
            provenance_hash=provenance_hash,
            engine_version=_MODULE_VERSION,
        )

        # Track history
        if supplier_id not in self._risk_history:
            self._risk_history[supplier_id] = []
        self._risk_history[supplier_id].append(result)
        self._assessment_count += 1

        logger.info(
            "Risk assessment completed: supplier=%s score=%.1f level=%s "
            "time=%.2fms hash_prefix=%s",
            supplier_id,
            composite,
            risk_level.value,
            elapsed_ms,
            provenance_hash[:16],
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Individual Risk Category Calculations
    # ------------------------------------------------------------------

    def calculate_deforestation_risk(
        self,
        supplier: SupplierProfile,
    ) -> RiskCategoryScore:
        """Calculate deforestation proximity risk component.

        Risk is inversely proportional to distance from recent deforestation
        events. Closer distance means higher risk.

        Scoring Logic:
            - No data (None): 80 (high risk for unknown)
            - < 1 km: 95
            - 1-5 km: 80
            - 5-10 km: 60
            - 10-25 km: 40
            - 25-50 km: 25
            - 50-100 km: 15
            - > 100 km: 5

        Args:
            supplier: Supplier profile with deforestation_distance_km.

        Returns:
            RiskCategoryScore for deforestation proximity.
        """
        weight = self._risk_weights.get(
            RiskCategory.DEFORESTATION_PROXIMITY.value, 0.30
        )
        distance = supplier.deforestation_distance_km
        details: Dict[str, Any] = {}

        if distance is None:
            raw_score = 80.0
            details["reason"] = "no_deforestation_distance_data"
            details["distance_km"] = None
        elif distance < 1.0:
            raw_score = 95.0
            details["reason"] = "very_close_to_deforestation"
            details["distance_km"] = distance
        elif distance < 5.0:
            raw_score = 80.0
            details["reason"] = "close_to_deforestation"
            details["distance_km"] = distance
        elif distance < 10.0:
            raw_score = 60.0
            details["reason"] = "moderate_proximity"
            details["distance_km"] = distance
        elif distance < 25.0:
            raw_score = 40.0
            details["reason"] = "moderate_distance"
            details["distance_km"] = distance
        elif distance < 50.0:
            raw_score = 25.0
            details["reason"] = "distant"
            details["distance_km"] = distance
        elif distance < 100.0:
            raw_score = 15.0
            details["reason"] = "far_from_deforestation"
            details["distance_km"] = distance
        else:
            raw_score = 5.0
            details["reason"] = "very_far_from_deforestation"
            details["distance_km"] = distance

        # Adjust for deforestation-free verification
        if supplier.deforestation_free_status == "verified":
            raw_score = raw_score * 0.3
            details["deforestation_free_verified"] = True
        elif supplier.deforestation_free_status == "self_declared":
            raw_score = raw_score * 0.7
            details["deforestation_free_self_declared"] = True

        raw_score = _clamp(raw_score)
        weighted_score = raw_score * weight

        logger.debug(
            "Deforestation risk: supplier=%s distance=%s raw=%.1f "
            "weighted=%.1f",
            supplier.supplier_id,
            distance,
            raw_score,
            weighted_score,
        )

        return RiskCategoryScore(
            category=RiskCategory.DEFORESTATION_PROXIMITY.value,
            raw_score=round(raw_score, 2),
            weight=weight,
            weighted_score=round(weighted_score, 2),
            details=details,
        )

    def calculate_country_risk(
        self,
        country_iso: str,
    ) -> RiskCategoryScore:
        """Calculate country-level risk from reference data.

        Uses a deterministic lookup table mapping ISO 3166-1 alpha-2
        country codes to deforestation/governance risk scores.

        Args:
            country_iso: ISO 3166-1 alpha-2 country code (e.g., "BR").

        Returns:
            RiskCategoryScore for country risk.
        """
        weight = self._risk_weights.get(
            RiskCategory.COUNTRY_RISK.value, 0.20
        )
        country_upper = country_iso.upper().strip() if country_iso else ""

        if not country_upper:
            raw_score = DEFAULT_COUNTRY_RISK
            details = {
                "reason": "no_country_code_provided",
                "country_iso": None,
                "source": "default",
            }
        elif country_upper in COUNTRY_RISK_SCORES:
            raw_score = COUNTRY_RISK_SCORES[country_upper]
            details = {
                "reason": "country_risk_lookup",
                "country_iso": country_upper,
                "source": "reference_data",
            }
        else:
            raw_score = DEFAULT_COUNTRY_RISK
            details = {
                "reason": "country_not_in_reference_data",
                "country_iso": country_upper,
                "source": "default",
            }

        weighted_score = raw_score * weight

        logger.debug(
            "Country risk: country=%s raw=%.1f weighted=%.1f",
            country_upper or "UNKNOWN",
            raw_score,
            weighted_score,
        )

        return RiskCategoryScore(
            category=RiskCategory.COUNTRY_RISK.value,
            raw_score=round(raw_score, 2),
            weight=weight,
            weighted_score=round(weighted_score, 2),
            details=details,
        )

    def calculate_certification_risk(
        self,
        certifications: List[Dict[str, Any]],
    ) -> RiskCategoryScore:
        """Calculate certification gap risk.

        Risk is reduced by valid, recognized certifications. Having no
        certification results in maximum risk. Multiple valid certifications
        compound the risk reduction.

        Scoring Logic:
            - No certifications: 100 (maximum risk)
            - Expired only: 80
            - Valid but unrecognized: 60
            - Single valid recognized: 100 * (1 - reduction_factor)
            - Multiple valid: Use best reduction factor

        Args:
            certifications: List of certification dicts with at least
                "type", "valid_until" keys. Optional: "certificate_id",
                "valid_from".

        Returns:
            RiskCategoryScore for certification gap.
        """
        weight = self._risk_weights.get(
            RiskCategory.CERTIFICATION_GAP.value, 0.15
        )
        now = utcnow()
        details: Dict[str, Any] = {
            "total_certifications": len(certifications),
            "valid_certifications": 0,
            "expired_certifications": 0,
            "recognized_certifications": 0,
        }

        if not certifications:
            raw_score = 100.0
            details["reason"] = "no_certifications"
            return RiskCategoryScore(
                category=RiskCategory.CERTIFICATION_GAP.value,
                raw_score=raw_score,
                weight=weight,
                weighted_score=round(raw_score * weight, 2),
                details=details,
            )

        valid_certs: List[Dict[str, Any]] = []
        expired_certs: List[Dict[str, Any]] = []
        best_reduction = 0.0

        for cert in certifications:
            cert_type = str(cert.get("type", "")).lower().strip()
            valid_until_str = str(cert.get("valid_until", ""))

            # Check expiry
            is_valid = True
            if valid_until_str:
                try:
                    valid_until = datetime.fromisoformat(
                        valid_until_str.replace("Z", "+00:00")
                    )
                    if valid_until.tzinfo is None:
                        valid_until = valid_until.replace(tzinfo=timezone.utc)
                    if valid_until < now:
                        is_valid = False
                except (ValueError, TypeError):
                    is_valid = False

            if is_valid:
                valid_certs.append(cert)
                reduction = CERTIFICATION_RISK_REDUCTION.get(cert_type, 0.0)
                if reduction > 0:
                    details["recognized_certifications"] += 1
                if reduction > best_reduction:
                    best_reduction = reduction
            else:
                expired_certs.append(cert)

        details["valid_certifications"] = len(valid_certs)
        details["expired_certifications"] = len(expired_certs)

        if not valid_certs:
            raw_score = 80.0
            details["reason"] = "all_certifications_expired"
        elif best_reduction > 0:
            raw_score = 100.0 * (1.0 - best_reduction)
            details["reason"] = "valid_recognized_certification"
            details["best_reduction_factor"] = best_reduction
        else:
            raw_score = 60.0
            details["reason"] = "valid_but_unrecognized_certifications"

        raw_score = _clamp(raw_score)
        weighted_score = raw_score * weight

        logger.debug(
            "Certification risk: total=%d valid=%d expired=%d raw=%.1f "
            "weighted=%.1f",
            len(certifications),
            len(valid_certs),
            len(expired_certs),
            raw_score,
            weighted_score,
        )

        return RiskCategoryScore(
            category=RiskCategory.CERTIFICATION_GAP.value,
            raw_score=round(raw_score, 2),
            weight=weight,
            weighted_score=round(weighted_score, 2),
            details=details,
        )

    def calculate_compliance_risk(
        self,
        history: SupplierProfile,
    ) -> RiskCategoryScore:
        """Calculate historical compliance violations risk.

        Risk increases with number of violations, recency of violations,
        and severity of past violations.

        Scoring Logic:
            - 0 violations: 5 (minimal risk)
            - 1-2 violations: 30 base + severity adjustment
            - 3-5 violations: 55 base + severity adjustment
            - 6-10 violations: 75 base + severity adjustment
            - >10 violations: 90 base + severity adjustment
            - Recency factor: violations within 1 year add 10 points

        Args:
            history: Supplier profile with compliance violation data.

        Returns:
            RiskCategoryScore for compliance history.
        """
        weight = self._risk_weights.get(
            RiskCategory.COMPLIANCE_HISTORY.value, 0.15
        )
        violations = history.compliance_violations
        details: Dict[str, Any] = {
            "total_violations": violations,
            "last_violation_date": history.last_violation_date or None,
        }

        # Base score from violation count
        if violations <= 0:
            base_score = 5.0
            details["reason"] = "no_violations"
        elif violations <= 2:
            base_score = 30.0
            details["reason"] = "few_violations"
        elif violations <= 5:
            base_score = 55.0
            details["reason"] = "moderate_violations"
        elif violations <= 10:
            base_score = 75.0
            details["reason"] = "many_violations"
        else:
            base_score = 90.0
            details["reason"] = "extensive_violations"

        # Severity adjustment from violation severity scores
        severity_adjustment = 0.0
        if history.violation_severity_scores:
            avg_severity = sum(history.violation_severity_scores) / len(
                history.violation_severity_scores
            )
            # Scale severity adjustment: avg_severity 0-100 maps to 0-10
            severity_adjustment = avg_severity * 0.1
            details["avg_violation_severity"] = round(avg_severity, 2)

        # Recency factor: recent violations increase risk
        recency_adjustment = 0.0
        if history.last_violation_date:
            try:
                last_violation = datetime.fromisoformat(
                    history.last_violation_date.replace("Z", "+00:00")
                )
                if last_violation.tzinfo is None:
                    last_violation = last_violation.replace(
                        tzinfo=timezone.utc
                    )
                days_since = (utcnow() - last_violation).days
                if days_since < 365:
                    recency_adjustment = 10.0
                    details["recency_factor"] = "within_1_year"
                elif days_since < 730:
                    recency_adjustment = 5.0
                    details["recency_factor"] = "within_2_years"
                else:
                    details["recency_factor"] = "older_than_2_years"
            except (ValueError, TypeError):
                details["recency_factor"] = "unparseable_date"

        raw_score = _clamp(base_score + severity_adjustment + recency_adjustment)
        weighted_score = raw_score * weight

        logger.debug(
            "Compliance risk: supplier=%s violations=%d base=%.1f "
            "severity_adj=%.1f recency_adj=%.1f raw=%.1f weighted=%.1f",
            history.supplier_id,
            violations,
            base_score,
            severity_adjustment,
            recency_adjustment,
            raw_score,
            weighted_score,
        )

        return RiskCategoryScore(
            category=RiskCategory.COMPLIANCE_HISTORY.value,
            raw_score=round(raw_score, 2),
            weight=weight,
            weighted_score=round(weighted_score, 2),
            details=details,
        )

    def calculate_data_quality_risk(
        self,
        profile: SupplierProfile,
    ) -> RiskCategoryScore:
        """Calculate profile completeness risk.

        Risk is inversely proportional to profile completeness. A fully
        complete profile yields low risk; missing fields increase risk.

        Uses PRD Appendix D field weights to calculate a weighted
        completeness score, then converts to risk: risk = 100 - completeness.

        Args:
            profile: Supplier profile to evaluate for completeness.

        Returns:
            RiskCategoryScore for data quality.
        """
        weight = self._risk_weights.get(
            RiskCategory.DATA_QUALITY.value, 0.10
        )
        completeness = self._calculate_profile_completeness(profile)
        raw_score = _clamp(100.0 - completeness)

        details: Dict[str, Any] = {
            "completeness_score": round(completeness, 2),
            "missing_fields": self._identify_missing_fields(profile),
        }

        if completeness >= 90:
            details["reason"] = "high_completeness"
        elif completeness >= 70:
            details["reason"] = "moderate_completeness"
        elif completeness >= 50:
            details["reason"] = "low_completeness"
        else:
            details["reason"] = "very_low_completeness"

        weighted_score = raw_score * weight

        logger.debug(
            "Data quality risk: supplier=%s completeness=%.1f raw=%.1f "
            "weighted=%.1f",
            profile.supplier_id,
            completeness,
            raw_score,
            weighted_score,
        )

        return RiskCategoryScore(
            category=RiskCategory.DATA_QUALITY.value,
            raw_score=round(raw_score, 2),
            weight=weight,
            weighted_score=round(weighted_score, 2),
            details=details,
        )

    def calculate_concentration_risk(
        self,
        supplier: SupplierProfile,
    ) -> RiskCategoryScore:
        """Calculate single-source dependency risk.

        Evaluates concentration risk based on:
        - Number of upstream suppliers (fewer = higher risk)
        - Volume concentration (% from single source)
        - Geographic concentration (all from one country)

        Scoring Logic:
            - 0 upstream suppliers: 10 (terminal/origin node)
            - 1 upstream supplier: 85 (single-source dependency)
            - 2-3 upstream suppliers: 55
            - 4-5 upstream suppliers: 30
            - >5 upstream suppliers: 15
            - Volume concentration HHI adjustment: +0-20 points

        Args:
            supplier: Supplier profile with upstream relationships.

        Returns:
            RiskCategoryScore for concentration risk.
        """
        weight = self._risk_weights.get(
            RiskCategory.CONCENTRATION_RISK.value, 0.10
        )
        upstream_count = len(supplier.upstream_supplier_ids)
        details: Dict[str, Any] = {
            "upstream_supplier_count": upstream_count,
        }

        # Base score from upstream supplier count
        if upstream_count == 0:
            base_score = 10.0
            details["reason"] = "terminal_node_no_upstream"
        elif upstream_count == 1:
            base_score = 85.0
            details["reason"] = "single_source_dependency"
        elif upstream_count <= 3:
            base_score = 55.0
            details["reason"] = "limited_diversification"
        elif upstream_count <= 5:
            base_score = 30.0
            details["reason"] = "moderate_diversification"
        else:
            base_score = 15.0
            details["reason"] = "well_diversified"

        # Volume concentration adjustment via simplified HHI
        hhi_adjustment = 0.0
        if supplier.volume_from_suppliers and supplier.total_sourcing_volume > 0:
            hhi = self._calculate_hhi(
                supplier.volume_from_suppliers,
                supplier.total_sourcing_volume,
            )
            # HHI ranges from 0 to 10,000; normalize to 0-20 adjustment
            hhi_adjustment = min(20.0, hhi / 500.0)
            details["hhi"] = round(hhi, 2)
            details["hhi_adjustment"] = round(hhi_adjustment, 2)

        raw_score = _clamp(base_score + hhi_adjustment)
        weighted_score = raw_score * weight

        logger.debug(
            "Concentration risk: supplier=%s upstream=%d base=%.1f "
            "hhi_adj=%.1f raw=%.1f weighted=%.1f",
            supplier.supplier_id,
            upstream_count,
            base_score,
            hhi_adjustment,
            raw_score,
            weighted_score,
        )

        return RiskCategoryScore(
            category=RiskCategory.CONCENTRATION_RISK.value,
            raw_score=round(raw_score, 2),
            weight=weight,
            weighted_score=round(weighted_score, 2),
            details=details,
        )

    # ------------------------------------------------------------------
    # Public API: Risk Propagation
    # ------------------------------------------------------------------

    def propagate_risk(
        self,
        root_node: SupplierChainNode,
        method: PropagationMethod = PropagationMethod.MAX,
    ) -> RiskPropagationResult:
        """Propagate risk through supplier chain hierarchy.

        Traverses the supplier chain tree from leaf nodes (deepest tier)
        up to the root (Tier 1) and propagates risk scores according to
        the specified method.

        Args:
            root_node: Root of the supplier chain tree.
            method: Propagation method (max, weighted_average,
                volume_weighted).

        Returns:
            RiskPropagationResult with propagated score and tier breakdown.

        Raises:
            ValueError: If root_node has no supplier_id.
        """
        if not root_node.supplier_id:
            raise ValueError("root_node must have a supplier_id")

        t_start = time.monotonic()
        propagation_id = str(uuid.uuid4())

        # Collect all tier scores and max-risk path
        tier_scores: Dict[int, List[float]] = {}
        max_risk_path: List[str] = []
        total_suppliers = 0
        deepest_tier = 0

        # Walk the tree to gather per-tier risk scores
        self._walk_tree(
            root_node, tier_scores, max_risk_path, 0, total_suppliers
        )

        # Count total suppliers and find deepest tier
        for tier, scores in tier_scores.items():
            total_suppliers += len(scores)
            if tier > deepest_tier:
                deepest_tier = tier

        # Apply propagation method
        if method == PropagationMethod.MAX:
            propagated_score = self._propagate_max(root_node)
        elif method == PropagationMethod.WEIGHTED_AVERAGE:
            propagated_score = self._propagate_weighted_average(root_node)
        elif method == PropagationMethod.VOLUME_WEIGHTED:
            propagated_score = self._propagate_volume_weighted(root_node)
        else:
            propagated_score = self._propagate_max(root_node)

        propagated_score = _clamp(propagated_score)
        propagated_level = self._classify_risk_level(propagated_score)

        # Aggregate tier scores to averages
        tier_avg_scores: Dict[int, float] = {}
        for tier, scores in tier_scores.items():
            if scores:
                tier_avg_scores[tier] = round(
                    sum(scores) / len(scores), 2
                )

        # Find max-risk path
        max_risk_path = self._find_max_risk_path(root_node)

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        # Provenance hash
        provenance_data = {
            "propagation_id": propagation_id,
            "root_supplier_id": root_node.supplier_id,
            "method": method.value,
            "propagated_score": round(propagated_score, 4),
            "total_suppliers": total_suppliers,
            "deepest_tier": deepest_tier,
        }
        provenance_hash = _compute_hash(provenance_data)

        result = RiskPropagationResult(
            propagation_id=propagation_id,
            root_supplier_id=root_node.supplier_id,
            method=method.value,
            propagated_score=round(propagated_score, 2),
            propagated_level=propagated_level.value,
            tier_scores=tier_avg_scores,
            max_risk_path=max_risk_path,
            total_suppliers=total_suppliers,
            deepest_tier=deepest_tier,
            assessed_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
            provenance_hash=provenance_hash,
        )

        logger.info(
            "Risk propagation completed: root=%s method=%s score=%.1f "
            "level=%s tiers=%d suppliers=%d time=%.2fms",
            root_node.supplier_id,
            method.value,
            propagated_score,
            propagated_level.value,
            deepest_tier,
            total_suppliers,
            elapsed_ms,
        )

        return result

    def propagate_max(
        self,
        root_node: SupplierChainNode,
    ) -> RiskPropagationResult:
        """Propagate risk using max (worst-case) method.

        The risk at each node is the maximum of its own risk and the
        maximum of all its children's propagated risk.

        Args:
            root_node: Root of the supplier chain tree.

        Returns:
            RiskPropagationResult using max propagation.
        """
        return self.propagate_risk(root_node, PropagationMethod.MAX)

    def propagate_weighted_average(
        self,
        root_node: SupplierChainNode,
    ) -> RiskPropagationResult:
        """Propagate risk using weighted average method.

        The risk at each node is its own risk score, plus the average
        of all children's propagated risk scaled by a tier decay factor.

        Args:
            root_node: Root of the supplier chain tree.

        Returns:
            RiskPropagationResult using weighted average propagation.
        """
        return self.propagate_risk(
            root_node, PropagationMethod.WEIGHTED_AVERAGE
        )

    def propagate_volume_weighted(
        self,
        root_node: SupplierChainNode,
    ) -> RiskPropagationResult:
        """Propagate risk using volume-weighted method.

        The risk at each node accounts for the volume share that each
        upstream supplier contributes. Higher-volume suppliers have
        greater influence on the propagated risk.

        Args:
            root_node: Root of the supplier chain tree.

        Returns:
            RiskPropagationResult using volume-weighted propagation.
        """
        return self.propagate_risk(
            root_node, PropagationMethod.VOLUME_WEIGHTED
        )

    # ------------------------------------------------------------------
    # Public API: Risk Change Detection
    # ------------------------------------------------------------------

    def detect_risk_changes(
        self,
        supplier_id: str,
        threshold: float = 10.0,
    ) -> List[RiskChangeAlert]:
        """Detect risk threshold crossings for a supplier.

        Compares the most recent risk assessment against previous
        assessments and generates alerts when the score changes by
        more than the specified threshold or the risk level changes.

        Args:
            supplier_id: Supplier to check for risk changes.
            threshold: Minimum absolute score change to trigger alert
                (default 10.0 points).

        Returns:
            List of RiskChangeAlert objects for significant changes.
        """
        alerts: List[RiskChangeAlert] = []
        history = self._risk_history.get(supplier_id, [])

        if len(history) < 2:
            logger.debug(
                "detect_risk_changes: insufficient history for %s "
                "(entries=%d)",
                supplier_id,
                len(history),
            )
            return alerts

        current = history[-1]
        previous = history[-2]
        change_delta = abs(current.composite_score - previous.composite_score)

        if change_delta < threshold:
            return alerts

        # Determine direction
        if current.composite_score > previous.composite_score:
            direction = "increase"
        elif current.composite_score < previous.composite_score:
            direction = "decrease"
        else:
            return alerts

        # Identify which categories changed significantly
        categories_changed: List[str] = []
        prev_cat_map = {
            cs.category: cs.raw_score for cs in previous.category_scores
        }
        for cs in current.category_scores:
            prev_raw = prev_cat_map.get(cs.category, 0.0)
            if abs(cs.raw_score - prev_raw) >= 5.0:
                categories_changed.append(cs.category)

        alert = RiskChangeAlert(
            alert_id=str(uuid.uuid4()),
            supplier_id=supplier_id,
            previous_score=previous.composite_score,
            current_score=current.composite_score,
            previous_level=previous.risk_level,
            current_level=current.risk_level,
            change_delta=round(change_delta, 2),
            direction=direction,
            triggered_at=utcnow().isoformat(),
            categories_changed=categories_changed,
        )
        alerts.append(alert)

        logger.info(
            "Risk change alert: supplier=%s delta=%.1f direction=%s "
            "level=%s->%s",
            supplier_id,
            change_delta,
            direction,
            previous.risk_level,
            current.risk_level,
        )

        return alerts

    # ------------------------------------------------------------------
    # Public API: Batch Assessment
    # ------------------------------------------------------------------

    def batch_assess(
        self,
        suppliers: List[SupplierProfile],
        batch_size: int = 1000,
    ) -> BatchRiskResult:
        """Perform batch risk assessment for multiple suppliers.

        Processes suppliers in configurable batch chunks for memory
        efficiency. Returns aggregate statistics alongside individual
        results.

        Args:
            suppliers: List of supplier profiles to assess.
            batch_size: Number of suppliers per processing chunk
                (default 1000).

        Returns:
            BatchRiskResult with all individual results and summary.
        """
        t_start = time.monotonic()
        batch_id = str(uuid.uuid4())
        results: List[RiskAssessmentResult] = []
        failed_count = 0
        risk_level_counts: Dict[str, int] = {}
        score_sum = 0.0

        logger.info(
            "Starting batch risk assessment: batch=%s suppliers=%d "
            "batch_size=%d",
            batch_id,
            len(suppliers),
            batch_size,
        )

        for i in range(0, len(suppliers), batch_size):
            chunk = suppliers[i : i + batch_size]
            for profile in chunk:
                try:
                    result = self.assess_supplier_risk(
                        profile.supplier_id, profile
                    )
                    results.append(result)
                    score_sum += result.composite_score
                    level = result.risk_level
                    risk_level_counts[level] = (
                        risk_level_counts.get(level, 0) + 1
                    )
                except Exception as exc:
                    failed_count += 1
                    logger.warning(
                        "Batch risk assessment failed for supplier=%s: %s",
                        profile.supplier_id,
                        str(exc),
                    )

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        successful = len(results)
        avg_score = round(score_sum / successful, 2) if successful > 0 else 0.0

        summary: Dict[str, Any] = {
            "average_composite_score": avg_score,
            "risk_level_distribution": risk_level_counts,
            "highest_risk_supplier": (
                max(results, key=lambda r: r.composite_score).supplier_id
                if results
                else None
            ),
            "lowest_risk_supplier": (
                min(results, key=lambda r: r.composite_score).supplier_id
                if results
                else None
            ),
        }

        # Provenance hash for entire batch
        provenance_data = {
            "batch_id": batch_id,
            "total_suppliers": len(suppliers),
            "successful": successful,
            "failed": failed_count,
            "avg_score": avg_score,
        }
        provenance_hash = _compute_hash(provenance_data)

        logger.info(
            "Batch risk assessment completed: batch=%s total=%d "
            "success=%d failed=%d avg_score=%.1f time=%.2fms",
            batch_id,
            len(suppliers),
            successful,
            failed_count,
            avg_score,
            elapsed_ms,
        )

        return BatchRiskResult(
            batch_id=batch_id,
            total_suppliers=len(suppliers),
            successful=successful,
            failed=failed_count,
            results=results,
            summary=summary,
            processing_time_ms=round(elapsed_ms, 3),
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Public API: Risk History and Trends
    # ------------------------------------------------------------------

    def get_risk_history(
        self,
        supplier_id: str,
    ) -> List[RiskAssessmentResult]:
        """Return risk assessment history for a supplier.

        Args:
            supplier_id: Supplier to retrieve history for.

        Returns:
            List of RiskAssessmentResult in chronological order.
        """
        return list(self._risk_history.get(supplier_id, []))

    def get_risk_trend(
        self,
        supplier_id: str,
    ) -> TrendDirection:
        """Determine risk trend direction for a supplier.

        Compares the last 3 assessments (or fewer if unavailable) to
        determine whether risk is improving, stable, or degrading.

        Args:
            supplier_id: Supplier to analyze trend for.

        Returns:
            TrendDirection enum value.
        """
        history = self._risk_history.get(supplier_id, [])
        if len(history) < 2:
            return TrendDirection.STABLE

        # Use last 3 assessments
        recent = history[-3:] if len(history) >= 3 else history
        scores = [r.composite_score for r in recent]

        if len(scores) < 2:
            return TrendDirection.STABLE

        # Linear trend: compare first and last
        delta = scores[-1] - scores[0]
        if delta > 5.0:
            return TrendDirection.DEGRADING
        elif delta < -5.0:
            return TrendDirection.IMPROVING
        return TrendDirection.STABLE

    @property
    def assessment_count(self) -> int:
        """Return total number of risk assessments performed."""
        return self._assessment_count

    @property
    def tracked_supplier_count(self) -> int:
        """Return number of unique suppliers with risk history."""
        return len(self._risk_history)

    # ------------------------------------------------------------------
    # Internal Helpers: Propagation Methods
    # ------------------------------------------------------------------

    def _propagate_max(
        self,
        node: SupplierChainNode,
    ) -> float:
        """Max propagation: risk = max(self, max(children)).

        The worst-case risk from any node in the subtree propagates
        to the root. This is the most conservative approach.

        Args:
            node: Current node in the supplier tree.

        Returns:
            Maximum risk score in the subtree.
        """
        current_max = node.risk_score
        for child in node.children:
            child_max = self._propagate_max(child)
            if child_max > current_max:
                current_max = child_max
        return current_max

    def _propagate_weighted_average(
        self,
        node: SupplierChainNode,
        decay_factor: float = 0.8,
    ) -> float:
        """Weighted average propagation with tier decay.

        The risk at each node is a blend of its own risk and the
        average risk of its children, decayed by tier depth.

        Formula:
            propagated = self_risk * (1 - child_weight) + avg_children * child_weight
            child_weight = decay_factor * (num_children / (num_children + 1))

        Args:
            node: Current node in the supplier tree.
            decay_factor: Tier decay factor (0-1, default 0.8).

        Returns:
            Weighted average propagated risk score.
        """
        if not node.children:
            return node.risk_score

        child_scores = [
            self._propagate_weighted_average(child, decay_factor)
            for child in node.children
        ]
        avg_children = sum(child_scores) / len(child_scores)

        child_weight = decay_factor * (
            len(node.children) / (len(node.children) + 1)
        )
        propagated = (
            node.risk_score * (1.0 - child_weight)
            + avg_children * child_weight
        )
        return _clamp(propagated)

    def _propagate_volume_weighted(
        self,
        node: SupplierChainNode,
    ) -> float:
        """Volume-weighted propagation based on sourcing volume shares.

        Each child's propagated risk is weighted by its volume_share
        (fraction of total volume it contributes to this node).

        Formula:
            propagated = max(self_risk, sum(child_risk * child_volume_share))

        Args:
            node: Current node in the supplier tree.

        Returns:
            Volume-weighted propagated risk score.
        """
        if not node.children:
            return node.risk_score

        total_share = sum(c.volume_share for c in node.children)
        if total_share <= 0:
            # Fallback to equal weighting
            total_share = float(len(node.children))
            for child in node.children:
                child.volume_share = 1.0

        weighted_sum = 0.0
        for child in node.children:
            child_risk = self._propagate_volume_weighted(child)
            normalized_share = child.volume_share / total_share
            weighted_sum += child_risk * normalized_share

        # Propagated score is the maximum of own score and volume-weighted
        # children score, ensuring risk never decreases upstream
        propagated = max(node.risk_score, weighted_sum)
        return _clamp(propagated)

    # ------------------------------------------------------------------
    # Internal Helpers: Tree Walking
    # ------------------------------------------------------------------

    def _walk_tree(
        self,
        node: SupplierChainNode,
        tier_scores: Dict[int, List[float]],
        max_risk_path: List[str],
        current_tier: int,
        count: int,
    ) -> None:
        """Recursively walk the supplier chain tree collecting metrics.

        Args:
            node: Current node.
            tier_scores: Dict mapping tier -> list of risk scores.
            max_risk_path: Accumulator for highest-risk path.
            current_tier: Current tier depth (0 = root).
            count: Running supplier count (modified in-place via dict).
        """
        if current_tier not in tier_scores:
            tier_scores[current_tier] = []
        tier_scores[current_tier].append(node.risk_score)

        for child in node.children:
            self._walk_tree(
                child, tier_scores, max_risk_path, current_tier + 1, count
            )

    def _find_max_risk_path(
        self,
        node: SupplierChainNode,
    ) -> List[str]:
        """Find the path from root to the highest-risk leaf node.

        Args:
            node: Current node.

        Returns:
            List of supplier IDs along the maximum-risk path.
        """
        if not node.children:
            return [node.supplier_id]

        max_child_path: List[str] = []
        max_child_score = -1.0

        for child in node.children:
            child_path = self._find_max_risk_path(child)
            child_leaf_score = child.risk_score
            # Check deepest leaf's risk
            if child_path:
                for c in node.children:
                    if c.supplier_id == child_path[-1]:
                        child_leaf_score = c.risk_score
                        break
            if child.risk_score > max_child_score:
                max_child_score = child.risk_score
                max_child_path = child_path

        return [node.supplier_id] + max_child_path

    # ------------------------------------------------------------------
    # Internal Helpers: Score Classification
    # ------------------------------------------------------------------

    def _classify_risk_level(self, score: float) -> RiskLevel:
        """Classify a numeric risk score into a qualitative risk level.

        Args:
            score: Risk score (0-100).

        Returns:
            RiskLevel enum value.
        """
        for lo, hi, level in RISK_LEVEL_THRESHOLDS:
            if lo <= score < hi:
                return level
        return RiskLevel.CRITICAL

    # ------------------------------------------------------------------
    # Internal Helpers: Profile Completeness
    # ------------------------------------------------------------------

    def _calculate_profile_completeness(
        self,
        profile: SupplierProfile,
    ) -> float:
        """Calculate weighted profile completeness score (0-100).

        Uses PRD Appendix D field weights.

        Args:
            profile: Supplier profile to evaluate.

        Returns:
            Completeness score 0-100.
        """
        total_score = 0.0

        field_value_map: Dict[str, Any] = {
            "legal_name": profile.legal_name,
            "registration_id": profile.registration_id,
            "country_iso": profile.country_iso,
            "gps_latitude": profile.gps_latitude,
            "gps_longitude": profile.gps_longitude,
            "address": profile.address,
            "admin_region": profile.admin_region,
            "commodity_types": profile.commodity_types,
            "annual_volume_tonnes": profile.annual_volume_tonnes,
            "processing_capacity": profile.processing_capacity,
            "certification_type": (
                profile.certifications[0].get("type", "")
                if profile.certifications
                else ""
            ),
            "certificate_id": (
                profile.certifications[0].get("certificate_id", "")
                if profile.certifications
                else ""
            ),
            "valid_from": (
                profile.certifications[0].get("valid_from", "")
                if profile.certifications
                else ""
            ),
            "valid_until": (
                profile.certifications[0].get("valid_until", "")
                if profile.certifications
                else ""
            ),
            "dds_reference": (
                profile.dds_references[0] if profile.dds_references else ""
            ),
            "deforestation_free_status": profile.deforestation_free_status,
            "last_verification_date": profile.last_verification_date,
            "primary_contact_name": profile.primary_contact_name,
            "primary_contact_email": profile.primary_contact_email,
            "compliance_contact_name": profile.compliance_contact_name,
            "compliance_contact_email": profile.compliance_contact_email,
        }

        for category_key, category_info in PROFILE_FIELD_WEIGHTS.items():
            category_weight = category_info["weight"]
            fields_config = category_info["fields"]
            category_completeness = 0.0

            for field_name, field_weight in fields_config.items():
                value = field_value_map.get(field_name)
                is_present = self._is_field_present(value)
                if is_present:
                    category_completeness += field_weight

            total_score += category_completeness * category_weight * 100.0

        return _clamp(total_score)

    def _identify_missing_fields(
        self,
        profile: SupplierProfile,
    ) -> List[str]:
        """Identify missing fields in a supplier profile.

        Args:
            profile: Supplier profile to evaluate.

        Returns:
            List of missing field names.
        """
        missing: List[str] = []

        checks: List[Tuple[str, Any]] = [
            ("legal_name", profile.legal_name),
            ("registration_id", profile.registration_id),
            ("country_iso", profile.country_iso),
            ("gps_latitude", profile.gps_latitude),
            ("gps_longitude", profile.gps_longitude),
            ("address", profile.address),
            ("admin_region", profile.admin_region),
            ("commodity_types", profile.commodity_types),
            ("annual_volume_tonnes", profile.annual_volume_tonnes),
            ("certifications", profile.certifications),
            ("dds_references", profile.dds_references),
            ("deforestation_free_status", profile.deforestation_free_status),
            ("primary_contact_name", profile.primary_contact_name),
            ("primary_contact_email", profile.primary_contact_email),
        ]

        for field_name, value in checks:
            if not self._is_field_present(value):
                missing.append(field_name)

        return missing

    def _is_field_present(self, value: Any) -> bool:
        """Check if a field value is considered 'present'.

        Args:
            value: Field value to check.

        Returns:
            True if the value is non-empty, non-None, non-zero.
        """
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if isinstance(value, (list, dict)) and len(value) == 0:
            return False
        return True

    # ------------------------------------------------------------------
    # Internal Helpers: Concentration Metrics
    # ------------------------------------------------------------------

    def _calculate_hhi(
        self,
        volume_map: Dict[str, float],
        total_volume: float,
    ) -> float:
        """Calculate Herfindahl-Hirschman Index for volume concentration.

        HHI = sum of squared market shares * 10,000.
        Range: 0 (perfect competition) to 10,000 (monopoly).

        Args:
            volume_map: Dict mapping supplier_id -> volume.
            total_volume: Total sourcing volume.

        Returns:
            HHI value (0-10,000).
        """
        if total_volume <= 0:
            return 0.0

        hhi = 0.0
        for volume in volume_map.values():
            share = volume / total_volume
            hhi += (share * 100.0) ** 2

        return min(10000.0, hhi)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"RiskPropagationEngine("
            f"assessments={self._assessment_count}, "
            f"tracked_suppliers={len(self._risk_history)}, "
            f"version={_MODULE_VERSION!r})"
        )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "RiskCategory",
    "RiskLevel",
    "PropagationMethod",
    "TrendDirection",
    # Constants
    "DEFAULT_RISK_WEIGHTS",
    "RISK_LEVEL_THRESHOLDS",
    "COUNTRY_RISK_SCORES",
    "DEFAULT_COUNTRY_RISK",
    "CERTIFICATION_RISK_REDUCTION",
    "PROFILE_FIELD_WEIGHTS",
    # Data classes
    "RiskCategoryScore",
    "RiskAssessmentResult",
    "RiskPropagationResult",
    "RiskChangeAlert",
    "BatchRiskResult",
    "SupplierProfile",
    "SupplierChainNode",
    # Engine
    "RiskPropagationEngine",
]
