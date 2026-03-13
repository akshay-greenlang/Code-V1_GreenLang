# -*- coding: utf-8 -*-
"""
ComplianceImpactEngine - AGENT-EUDR-019 Engine 8: EUDR Compliance Impact Assessment

Evaluates the impact of corruption levels on EUDR compliance obligations.
Determines due diligence requirements based on corruption risk profiles,
maps corruption indices to EUDR Article 29 country benchmarking/classification,
and generates prioritized remediation recommendations.

Zero-Hallucination Guarantees:
    - All composite scoring uses deterministic weighted Decimal arithmetic.
    - Country classification uses explicit threshold comparisons.
    - Due diligence level determination uses static lookup rules.
    - Mitigation requirements are commodity-specific static lists.
    - Enhanced DD triggers are explicit boolean threshold checks.
    - SHA-256 provenance hashes on all output objects.

EUDR Classification Mapping (Article 29):
    - LOW_RISK:      Composite corruption score < 0.25 (simplified DD)
    - STANDARD_RISK: 0.25 <= composite score < 0.60 (standard DD)
    - HIGH_RISK:     Composite score >= 0.60 (enhanced DD)

Composite Score Weights:
    - CPI (Corruption Perceptions Index):      35%
    - WGI (Control of Corruption):             25%
    - Bribery Risk Score:                      20%
    - Institutional Quality Score:             20%

Enhanced Due Diligence Triggers:
    - CPI < 30
    - WGI Control of Corruption < -1.0
    - Recent significant decline (> 5 points/year)
    - Active FATF grey/blacklist membership
    - Known governance crisis or conflict

Due Diligence Levels:
    - SIMPLIFIED: Minimal requirements for LOW_RISK countries
    - STANDARD:   Full due diligence for STANDARD_RISK countries
    - ENHANCED:   Maximum scrutiny for HIGH_RISK countries

Performance Targets:
    - Single country assessment: <30ms
    - Batch assessment (180 countries): <3s
    - DD recommendation generation: <50ms
    - Country classification: <10ms

Regulatory References:
    - EUDR Article 29: Country benchmarking system
    - EUDR Article 10: Risk assessment obligations
    - EUDR Article 11: Risk mitigation measures
    - EUDR Article 12: Reporting obligations
    - EUDR Article 13: Record keeping (5 years)
    - EUDR Recital 31: Governance indicators
    - FATF Mutual Evaluation Reports

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-019, Engine 8 (Compliance Impact Engine)
Agent ID: GL-EUDR-CIM-019
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, dataclass with to_dict, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "impact") -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _to_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


def _clamp_decimal(value: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    """Clamp a Decimal value to [lo, hi] range.

    Args:
        value: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped Decimal.
    """
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EUDRCountryClassification(str, Enum):
    """EUDR Article 29 country risk classification.

    Values:
        LOW_RISK: Country with low corruption/deforestation risk.
        STANDARD_RISK: Country with moderate risk (default for most).
        HIGH_RISK: Country with high corruption/deforestation risk.
    """

    LOW_RISK = "LOW_RISK"
    STANDARD_RISK = "STANDARD_RISK"
    HIGH_RISK = "HIGH_RISK"


class DueDiligenceLevel(str, Enum):
    """Level of due diligence required under EUDR.

    Values:
        SIMPLIFIED: Minimal requirements for low-risk countries.
        STANDARD: Full due diligence for standard-risk countries.
        ENHANCED: Maximum scrutiny for high-risk countries.
    """

    SIMPLIFIED = "SIMPLIFIED"
    STANDARD = "STANDARD"
    ENHANCED = "ENHANCED"


class RecommendationPriority(str, Enum):
    """Priority level for due diligence recommendations.

    Values:
        CRITICAL: Must be addressed immediately.
        HIGH: Should be addressed within 30 days.
        MEDIUM: Should be addressed within 90 days.
        LOW: Address during next review cycle.
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class MonitoringFrequency(str, Enum):
    """Required monitoring frequency based on risk classification.

    Values:
        CONTINUOUS: Real-time monitoring with immediate alerts.
        MONTHLY: Monthly review cycle.
        QUARTERLY: Quarterly review cycle.
        SEMI_ANNUAL: Twice-yearly review cycle.
        ANNUAL: Annual review cycle.
    """

    CONTINUOUS = "continuous"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Composite score threshold for LOW_RISK classification.
LOW_RISK_THRESHOLD: Decimal = Decimal("0.25")

#: Composite score threshold for HIGH_RISK classification.
HIGH_RISK_THRESHOLD: Decimal = Decimal("0.60")

#: Composite score weights.
WEIGHT_CPI: Decimal = Decimal("0.35")
WEIGHT_WGI: Decimal = Decimal("0.25")
WEIGHT_BRIBERY: Decimal = Decimal("0.20")
WEIGHT_INSTITUTIONAL: Decimal = Decimal("0.20")

#: CPI threshold for enhanced DD trigger.
ENHANCED_DD_CPI_THRESHOLD: Decimal = Decimal("30")

#: WGI CC threshold for enhanced DD trigger.
ENHANCED_DD_WGI_THRESHOLD: Decimal = Decimal("-1.0")

#: Annual decline threshold for enhanced DD trigger (CPI points/year).
ENHANCED_DD_DECLINE_THRESHOLD: Decimal = Decimal("5")

#: CPI normalization range (0 = most corrupt, 100 = cleanest).
CPI_MIN: Decimal = Decimal("0")
CPI_MAX: Decimal = Decimal("100")

#: WGI normalization range (-2.5 to +2.5).
WGI_MIN: Decimal = Decimal("-2.5")
WGI_MAX: Decimal = Decimal("2.5")

#: Bribery risk normalization range (0-100).
BRIBERY_MIN: Decimal = Decimal("0")
BRIBERY_MAX: Decimal = Decimal("100")

#: Institutional quality normalization range (0-100).
INSTITUTIONAL_MIN: Decimal = Decimal("0")
INSTITUTIONAL_MAX: Decimal = Decimal("100")

#: EUDR-regulated commodity types.
EUDR_COMMODITIES: frozenset = frozenset({
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
})

#: FATF grey/blacklist countries (as of 2024 reference data).
FATF_GREY_LIST: frozenset = frozenset({
    "BF", "CM", "CD", "HT", "KE", "ML", "MZ", "MM", "NG", "PH",
    "SN", "ZA", "SS", "TZ", "VN", "YE",
})

FATF_BLACK_LIST: frozenset = frozenset({
    "KP", "IR", "MM",
})

#: Countries with known active governance crises.
GOVERNANCE_CRISIS_COUNTRIES: frozenset = frozenset({
    "MM", "VE", "HT", "SS", "SO", "YE", "AF", "LY", "SY",
})

# ---------------------------------------------------------------------------
# Reference Data: Country Corruption Indices
# ---------------------------------------------------------------------------
# Most recent available year. In production, these would be loaded from
# the database. This reference data enables offline operation.

REFERENCE_COUNTRY_DATA: Dict[str, Dict[str, Any]] = {
    "DK": {"cpi": Decimal("90"), "wgi_cc": Decimal("2.18"), "bribery": Decimal("5"), "institutional": Decimal("95"), "region": "europe"},
    "NZ": {"cpi": Decimal("85"), "wgi_cc": Decimal("2.10"), "bribery": Decimal("8"), "institutional": Decimal("92"), "region": "oceania"},
    "FI": {"cpi": Decimal("87"), "wgi_cc": Decimal("2.15"), "bribery": Decimal("6"), "institutional": Decimal("93"), "region": "europe"},
    "SG": {"cpi": Decimal("83"), "wgi_cc": Decimal("2.08"), "bribery": Decimal("10"), "institutional": Decimal("90"), "region": "asia"},
    "SE": {"cpi": Decimal("82"), "wgi_cc": Decimal("2.05"), "bribery": Decimal("7"), "institutional": Decimal("91"), "region": "europe"},
    "DE": {"cpi": Decimal("78"), "wgi_cc": Decimal("1.82"), "bribery": Decimal("12"), "institutional": Decimal("88"), "region": "europe"},
    "US": {"cpi": Decimal("69"), "wgi_cc": Decimal("1.20"), "bribery": Decimal("18"), "institutional": Decimal("80"), "region": "americas"},
    "MY": {"cpi": Decimal("50"), "wgi_cc": Decimal("-0.08"), "bribery": Decimal("42"), "institutional": Decimal("55"), "region": "asia"},
    "GH": {"cpi": Decimal("42"), "wgi_cc": Decimal("-0.08"), "bribery": Decimal("48"), "institutional": Decimal("45"), "region": "africa"},
    "CO": {"cpi": Decimal("40"), "wgi_cc": Decimal("-0.22"), "bribery": Decimal("50"), "institutional": Decimal("42"), "region": "americas"},
    "IN": {"cpi": Decimal("39"), "wgi_cc": Decimal("-0.30"), "bribery": Decimal("52"), "institutional": Decimal("40"), "region": "asia"},
    "BR": {"cpi": Decimal("36"), "wgi_cc": Decimal("-0.42"), "bribery": Decimal("55"), "institutional": Decimal("38"), "region": "americas"},
    "CI": {"cpi": Decimal("37"), "wgi_cc": Decimal("-0.45"), "bribery": Decimal("55"), "institutional": Decimal("35"), "region": "africa"},
    "TH": {"cpi": Decimal("35"), "wgi_cc": Decimal("-0.40"), "bribery": Decimal("55"), "institutional": Decimal("37"), "region": "asia"},
    "ID": {"cpi": Decimal("34"), "wgi_cc": Decimal("-0.52"), "bribery": Decimal("58"), "institutional": Decimal("35"), "region": "asia"},
    "PH": {"cpi": Decimal("34"), "wgi_cc": Decimal("-0.50"), "bribery": Decimal("56"), "institutional": Decimal("34"), "region": "asia"},
    "EC": {"cpi": Decimal("33"), "wgi_cc": Decimal("-0.55"), "bribery": Decimal("60"), "institutional": Decimal("30"), "region": "americas"},
    "PE": {"cpi": Decimal("33"), "wgi_cc": Decimal("-0.48"), "bribery": Decimal("58"), "institutional": Decimal("32"), "region": "americas"},
    "BO": {"cpi": Decimal("29"), "wgi_cc": Decimal("-0.70"), "bribery": Decimal("65"), "institutional": Decimal("28"), "region": "americas"},
    "PY": {"cpi": Decimal("28"), "wgi_cc": Decimal("-0.82"), "bribery": Decimal("68"), "institutional": Decimal("25"), "region": "americas"},
    "CM": {"cpi": Decimal("26"), "wgi_cc": Decimal("-1.05"), "bribery": Decimal("72"), "institutional": Decimal("22"), "region": "africa"},
    "MZ": {"cpi": Decimal("25"), "wgi_cc": Decimal("-0.90"), "bribery": Decimal("70"), "institutional": Decimal("23"), "region": "africa"},
    "LR": {"cpi": Decimal("25"), "wgi_cc": Decimal("-0.88"), "bribery": Decimal("68"), "institutional": Decimal("24"), "region": "africa"},
    "NG": {"cpi": Decimal("24"), "wgi_cc": Decimal("-1.10"), "bribery": Decimal("75"), "institutional": Decimal("20"), "region": "africa"},
    "GT": {"cpi": Decimal("23"), "wgi_cc": Decimal("-0.95"), "bribery": Decimal("72"), "institutional": Decimal("21"), "region": "americas"},
    "HN": {"cpi": Decimal("23"), "wgi_cc": Decimal("-0.92"), "bribery": Decimal("70"), "institutional": Decimal("22"), "region": "americas"},
    "KH": {"cpi": Decimal("22"), "wgi_cc": Decimal("-1.15"), "bribery": Decimal("78"), "institutional": Decimal("18"), "region": "asia"},
    "CG": {"cpi": Decimal("20"), "wgi_cc": Decimal("-1.20"), "bribery": Decimal("80"), "institutional": Decimal("15"), "region": "africa"},
    "CD": {"cpi": Decimal("20"), "wgi_cc": Decimal("-1.45"), "bribery": Decimal("82"), "institutional": Decimal("12"), "region": "africa"},
    "MM": {"cpi": Decimal("20"), "wgi_cc": Decimal("-1.50"), "bribery": Decimal("85"), "institutional": Decimal("10"), "region": "asia"},
    "VE": {"cpi": Decimal("13"), "wgi_cc": Decimal("-1.74"), "bribery": Decimal("90"), "institutional": Decimal("5"), "region": "americas"},
}

# ---------------------------------------------------------------------------
# Commodity-Specific DD Requirements
# ---------------------------------------------------------------------------

COMMODITY_DD_REQUIREMENTS: Dict[str, Dict[str, List[str]]] = {
    "cattle": {
        "SIMPLIFIED": [
            "Supplier self-declaration of deforestation-free compliance",
            "GPS coordinates of farm/ranch",
            "Basic supply chain documentation",
        ],
        "STANDARD": [
            "Farm GPS boundary verification",
            "Animal health and movement records",
            "Grazing area satellite monitoring",
            "Supplier audit within 24 months",
            "Deforestation-free certification or equivalent",
        ],
        "ENHANCED": [
            "On-site farm inspection and GPS boundary survey",
            "Full animal traceability from birth to slaughter",
            "Monthly satellite deforestation monitoring",
            "Third-party verification of deforestation-free status",
            "Government land-use permit verification",
            "Community consent documentation (FPIC)",
            "Enhanced supply chain mapping (all tiers)",
            "Quarterly compliance reporting",
        ],
    },
    "cocoa": {
        "SIMPLIFIED": [
            "Cooperative-level compliance declaration",
            "GPS coordinates of cooperatives",
            "Basic traceability records",
        ],
        "STANDARD": [
            "Farm-level GPS polygon mapping",
            "Cooperative membership verification",
            "Certification status check (UTZ, Rainforest Alliance)",
            "Annual satellite monitoring",
            "Fermentation/drying facility documentation",
        ],
        "ENHANCED": [
            "Individual farmer GPS polygon verification",
            "Monthly satellite deforestation monitoring",
            "Third-party certification audit",
            "Community land rights verification",
            "Child labor risk assessment",
            "Government forest boundary cross-reference",
            "Full supply chain traceability to plot level",
            "Quarterly third-party audits",
        ],
    },
    "coffee": {
        "SIMPLIFIED": [
            "Washing station level compliance declaration",
            "GPS coordinates of processing facilities",
            "Basic origin documentation",
        ],
        "STANDARD": [
            "Farm-level GPS mapping",
            "Washing station records and traceability",
            "Quality grade certificates with origin",
            "Annual satellite monitoring",
            "Certification verification (Rainforest Alliance, Fairtrade)",
        ],
        "ENHANCED": [
            "Individual farm GPS polygon verification",
            "Monthly satellite deforestation monitoring",
            "Third-party certification audit",
            "Shade-grown verification",
            "Water source and watershed assessment",
            "Full supply chain mapping (farm to export)",
            "Quarterly compliance reporting",
            "Community engagement documentation",
        ],
    },
    "oil_palm": {
        "SIMPLIFIED": [
            "Mill-level compliance declaration",
            "GPS coordinates of mills",
            "Basic NDPE (No Deforestation, Peat, Exploitation) commitment",
        ],
        "STANDARD": [
            "Plantation boundary GPS polygon mapping",
            "Mill GPS and supply base mapping",
            "RSPO certification status",
            "Peatland assessment",
            "Annual satellite monitoring",
            "NDPE compliance documentation",
        ],
        "ENHANCED": [
            "Individual plantation GPS polygon verification",
            "Monthly satellite deforestation and fire monitoring",
            "RSPO/ISCC full certification audit",
            "HCV/HCS assessment (High Conservation Value/High Carbon Stock)",
            "Peatland drainage and conversion assessment",
            "Third-party traceability to plantation (TTP)",
            "Community FPIC documentation",
            "Quarterly independent monitoring reports",
            "Smallholder inclusion program verification",
        ],
    },
    "rubber": {
        "SIMPLIFIED": [
            "Processing facility compliance declaration",
            "GPS coordinates of processing facilities",
            "Basic supply chain documentation",
        ],
        "STANDARD": [
            "Plantation/smallholder GPS mapping",
            "Processing facility chain of custody",
            "FSC certification verification (where applicable)",
            "Annual satellite monitoring",
            "Species identification documentation",
        ],
        "ENHANCED": [
            "Individual plot GPS polygon verification",
            "Monthly satellite deforestation monitoring",
            "Full chain of custody audit (FSC/PEFC)",
            "Land-use permit verification",
            "Community land rights assessment",
            "Labor conditions audit",
            "Full supply chain mapping to plot level",
            "Quarterly independent monitoring",
        ],
    },
    "soya": {
        "SIMPLIFIED": [
            "Storage facility level compliance declaration",
            "GPS coordinates of storage facilities",
            "Basic GMO status documentation",
        ],
        "STANDARD": [
            "Farm-level GPS polygon mapping",
            "Storage facility and silo documentation",
            "GMO/non-GMO status verification",
            "Annual satellite monitoring",
            "RTRS certification (where applicable)",
        ],
        "ENHANCED": [
            "Individual farm GPS polygon verification",
            "Monthly satellite deforestation and Cerrado monitoring",
            "Full chain of custody audit",
            "Moratorium compliance verification (Soy Moratorium)",
            "Third-party environmental impact assessment",
            "Government environmental license verification",
            "Full supply chain mapping (farm to port)",
            "Quarterly compliance and traceability reports",
        ],
    },
    "wood": {
        "SIMPLIFIED": [
            "Species identification (genus/species)",
            "GPS coordinates of forest management unit",
            "FLEGT/CITES compliance documentation",
        ],
        "STANDARD": [
            "Forest management unit boundary mapping",
            "Species identification with DNA verification available",
            "Felling license and harvest plan",
            "FSC/PEFC chain of custody certification",
            "Annual satellite monitoring",
            "Legal compliance documentation (EUTR-equivalent)",
        ],
        "ENHANCED": [
            "Individual felling site GPS polygon verification",
            "Species identification with DNA timber tracking",
            "Monthly satellite deforestation monitoring",
            "Full chain of custody audit (FSC/PEFC/VLC)",
            "Government felling permit verification",
            "Indigenous and community rights assessment",
            "CITES species check and documentation",
            "Third-party legal compliance verification",
            "Quarterly supply chain audit and traceability report",
        ],
    },
}

# ---------------------------------------------------------------------------
# Risk Factor Definitions
# ---------------------------------------------------------------------------

RISK_FACTORS: Dict[str, Dict[str, Any]] = {
    "high_corruption": {
        "name": "High Corruption Level",
        "description": "Country has a CPI score below 30, indicating systemic corruption",
        "trigger": "CPI < 30",
        "weight": Decimal("0.30"),
    },
    "weak_governance": {
        "name": "Weak Governance Indicators",
        "description": "WGI Control of Corruption below -1.0, indicating severe governance weakness",
        "trigger": "WGI CC < -1.0",
        "weight": Decimal("0.20"),
    },
    "rapid_decline": {
        "name": "Rapid Governance Decline",
        "description": "CPI score declining more than 5 points per year",
        "trigger": "CPI decline > 5 pts/yr",
        "weight": Decimal("0.15"),
    },
    "fatf_listed": {
        "name": "FATF Grey/Blacklist",
        "description": "Country is on FATF grey or blacklist for AML/CFT deficiencies",
        "trigger": "FATF grey/blacklist",
        "weight": Decimal("0.15"),
    },
    "governance_crisis": {
        "name": "Active Governance Crisis",
        "description": "Country experiencing active governance crisis, conflict, or collapse",
        "trigger": "governance crisis flag",
        "weight": Decimal("0.10"),
    },
    "high_bribery": {
        "name": "High Bribery Risk",
        "description": "Sector-specific bribery risk score above 70",
        "trigger": "Bribery score > 70",
        "weight": Decimal("0.10"),
    },
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ComplianceImpact:
    """Full compliance impact assessment for a country.

    Attributes:
        impact_id: Unique identifier.
        country_code: ISO 3166-1 alpha-2 country code.
        corruption_composite_score: Weighted composite corruption score (0-1).
        eudr_classification: EUDR Article 29 risk classification.
        due_diligence_level: Required due diligence level.
        risk_factors: List of active risk factors.
        enhanced_dd_triggers: List of enhanced DD triggers that fired.
        monitoring_frequency: Required monitoring frequency.
        cpi_score: Raw CPI score.
        wgi_cc_score: Raw WGI CC score.
        bribery_score: Raw bribery risk score.
        institutional_score: Raw institutional quality score.
        cpi_normalized: CPI normalized to 0-1 corruption scale.
        wgi_normalized: WGI normalized to 0-1 corruption scale.
        bribery_normalized: Bribery normalized to 0-1 corruption scale.
        institutional_normalized: Institutional normalized to 0-1 corruption scale.
        warnings: Any assessment warnings.
        provenance_hash: SHA-256 hash.
    """

    impact_id: str = ""
    country_code: str = ""
    corruption_composite_score: Decimal = Decimal("0")
    eudr_classification: str = "STANDARD_RISK"
    due_diligence_level: str = "STANDARD"
    risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    enhanced_dd_triggers: List[str] = field(default_factory=list)
    monitoring_frequency: str = "quarterly"
    cpi_score: Decimal = Decimal("0")
    wgi_cc_score: Decimal = Decimal("0")
    bribery_score: Decimal = Decimal("0")
    institutional_score: Decimal = Decimal("0")
    cpi_normalized: Decimal = Decimal("0")
    wgi_normalized: Decimal = Decimal("0")
    bribery_normalized: Decimal = Decimal("0")
    institutional_normalized: Decimal = Decimal("0")
    warnings: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "impact_id": self.impact_id,
            "country_code": self.country_code,
            "corruption_composite_score": str(self.corruption_composite_score),
            "eudr_classification": self.eudr_classification,
            "due_diligence_level": self.due_diligence_level,
            "risk_factors": self.risk_factors,
            "enhanced_dd_triggers": self.enhanced_dd_triggers,
            "monitoring_frequency": self.monitoring_frequency,
            "cpi_score": str(self.cpi_score),
            "wgi_cc_score": str(self.wgi_cc_score),
            "bribery_score": str(self.bribery_score),
            "institutional_score": str(self.institutional_score),
            "cpi_normalized": str(self.cpi_normalized),
            "wgi_normalized": str(self.wgi_normalized),
            "bribery_normalized": str(self.bribery_normalized),
            "institutional_normalized": str(self.institutional_normalized),
            "warnings": self.warnings,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class DDRecommendation:
    """A due diligence recommendation for a specific country/commodity pair.

    Attributes:
        recommendation_id: Unique identifier.
        country_code: ISO country code.
        commodity: EUDR commodity type.
        category: Recommendation category.
        priority: Priority level.
        description: Detailed description.
        action_items: List of specific action items.
        regulatory_reference: Relevant EUDR article reference.
        estimated_effort: Estimated effort to implement.
        deadline_guidance: Suggested timeline for completion.
        provenance_hash: SHA-256 hash.
    """

    recommendation_id: str = ""
    country_code: str = ""
    commodity: str = ""
    category: str = ""
    priority: str = "MEDIUM"
    description: str = ""
    action_items: List[str] = field(default_factory=list)
    regulatory_reference: str = ""
    estimated_effort: str = ""
    deadline_guidance: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "recommendation_id": self.recommendation_id,
            "country_code": self.country_code,
            "commodity": self.commodity,
            "category": self.category,
            "priority": self.priority,
            "description": self.description,
            "action_items": self.action_items,
            "regulatory_reference": self.regulatory_reference,
            "estimated_effort": self.estimated_effort,
            "deadline_guidance": self.deadline_guidance,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class CountryImpactProfile:
    """Detailed impact profile for a country covering all commodities.

    Attributes:
        profile_id: Unique identifier.
        country_code: ISO country code.
        impact: Core compliance impact assessment.
        commodity_requirements: Per-commodity DD requirements.
        recommendations: Prioritized recommendations.
        total_recommendations: Count of recommendations.
        critical_recommendations: Count of critical-priority recommendations.
        provenance_hash: SHA-256 hash.
    """

    profile_id: str = ""
    country_code: str = ""
    impact: Optional[Dict[str, Any]] = None
    commodity_requirements: Dict[str, List[str]] = field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    total_recommendations: int = 0
    critical_recommendations: int = 0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "profile_id": self.profile_id,
            "country_code": self.country_code,
            "impact": self.impact,
            "commodity_requirements": self.commodity_requirements,
            "recommendations": self.recommendations,
            "total_recommendations": self.total_recommendations,
            "critical_recommendations": self.critical_recommendations,
            "provenance_hash": self.provenance_hash,
        }


# ---------------------------------------------------------------------------
# ComplianceImpactEngine
# ---------------------------------------------------------------------------


class ComplianceImpactEngine:
    """Production-grade EUDR compliance impact assessment engine.

    Evaluates how corruption levels impact EUDR compliance obligations
    for specific countries and commodities. Maps corruption indices to
    EUDR Article 29 classifications, determines due diligence levels,
    and generates prioritized remediation recommendations.

    Thread Safety:
        All mutable state is protected by a reentrant lock.

    Zero-Hallucination:
        All scoring uses deterministic weighted Decimal arithmetic.
        Classification uses explicit threshold comparisons. Requirements
        are static commodity-specific lists. No ML/LLM in any path.

    Attributes:
        _custom_data: User-supplied country data overrides.
        _annual_decline_data: Country CPI annual decline data.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> engine = ComplianceImpactEngine()
        >>> result = engine.assess_compliance_impact("BR")
        >>> assert result["eudr_classification"] in ("LOW_RISK", "STANDARD_RISK", "HIGH_RISK")
        >>> assert "due_diligence_level" in result
    """

    def __init__(self) -> None:
        """Initialize ComplianceImpactEngine with reference data."""
        self._custom_data: Dict[str, Dict[str, Any]] = {}
        self._annual_decline_data: Dict[str, Decimal] = {}
        self._lock: threading.RLock = threading.RLock()
        logger.info(
            "ComplianceImpactEngine initialized (version=%s, "
            "reference_countries=%d, commodities=%d)",
            _MODULE_VERSION,
            len(REFERENCE_COUNTRY_DATA),
            len(EUDR_COMMODITIES),
        )

    # ------------------------------------------------------------------
    # Data Access
    # ------------------------------------------------------------------

    def load_custom_data(
        self,
        country_code: str,
        data: Dict[str, Any],
    ) -> None:
        """Load custom country data for impact assessment.

        Args:
            country_code: ISO country code.
            data: Dictionary with keys: cpi, wgi_cc, bribery, institutional.

        Raises:
            ValueError: If country_code or data is invalid.
        """
        if not country_code or not isinstance(country_code, str):
            raise ValueError("country_code must be a non-empty string")
        if not data:
            raise ValueError("data must be non-empty")

        country_code = country_code.upper()
        with self._lock:
            self._custom_data[country_code] = dict(data)

        logger.info("Loaded custom data for %s", country_code)

    def set_annual_decline(
        self,
        country_code: str,
        decline_per_year: Decimal,
    ) -> None:
        """Set the annual CPI decline rate for a country.

        Used to evaluate the 'rapid decline' enhanced DD trigger.

        Args:
            country_code: ISO country code.
            decline_per_year: CPI points declined per year (positive = decline).
        """
        country_code = country_code.upper()
        with self._lock:
            self._annual_decline_data[country_code] = decline_per_year
        logger.debug("Set annual decline for %s: %s pts/yr", country_code, decline_per_year)

    def _get_country_data(self, country_code: str) -> Dict[str, Any]:
        """Retrieve country data from custom or reference sources.

        Args:
            country_code: ISO country code (uppercase).

        Returns:
            Dictionary with cpi, wgi_cc, bribery, institutional values.
        """
        with self._lock:
            custom = self._custom_data.get(country_code)
            if custom:
                return dict(custom)

        ref = REFERENCE_COUNTRY_DATA.get(country_code)
        if ref:
            return dict(ref)

        return {}

    # ------------------------------------------------------------------
    # Core Calculation Methods (Zero-Hallucination)
    # ------------------------------------------------------------------

    def _normalize_cpi(self, cpi: Decimal) -> Decimal:
        """Normalize CPI score to 0-1 corruption scale.

        CPI: 0 = most corrupt, 100 = cleanest.
        Normalized: 0 = cleanest, 1 = most corrupt.

        Args:
            cpi: Raw CPI score (0-100).

        Returns:
            Normalized corruption score (0-1).
        """
        cpi_clamped = _clamp_decimal(cpi, CPI_MIN, CPI_MAX)
        normalized = (CPI_MAX - cpi_clamped) / CPI_MAX
        return normalized.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _normalize_wgi(self, wgi_cc: Decimal) -> Decimal:
        """Normalize WGI Control of Corruption to 0-1 corruption scale.

        WGI CC: -2.5 = most corrupt, +2.5 = cleanest.
        Normalized: 0 = cleanest, 1 = most corrupt.

        Args:
            wgi_cc: Raw WGI CC estimate (-2.5 to +2.5).

        Returns:
            Normalized corruption score (0-1).
        """
        wgi_clamped = _clamp_decimal(wgi_cc, WGI_MIN, WGI_MAX)
        wgi_range = WGI_MAX - WGI_MIN  # 5.0
        normalized = (WGI_MAX - wgi_clamped) / wgi_range
        return normalized.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _normalize_bribery(self, bribery: Decimal) -> Decimal:
        """Normalize bribery risk score to 0-1 corruption scale.

        Bribery: 0 = low risk, 100 = high risk.
        Normalized: 0 = low risk, 1 = high risk (same direction).

        Args:
            bribery: Raw bribery risk score (0-100).

        Returns:
            Normalized corruption score (0-1).
        """
        bribery_clamped = _clamp_decimal(bribery, BRIBERY_MIN, BRIBERY_MAX)
        normalized = bribery_clamped / BRIBERY_MAX
        return normalized.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _normalize_institutional(self, institutional: Decimal) -> Decimal:
        """Normalize institutional quality score to 0-1 corruption scale.

        Institutional: 0 = weakest, 100 = strongest.
        Normalized: 0 = strongest (low corruption), 1 = weakest (high corruption).

        Args:
            institutional: Raw institutional quality score (0-100).

        Returns:
            Normalized corruption score (0-1).
        """
        inst_clamped = _clamp_decimal(institutional, INSTITUTIONAL_MIN, INSTITUTIONAL_MAX)
        normalized = (INSTITUTIONAL_MAX - inst_clamped) / INSTITUTIONAL_MAX
        return normalized.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _calculate_composite_corruption_score(
        self,
        cpi: Decimal,
        wgi_cc: Decimal,
        bribery: Decimal,
        institutional: Decimal,
    ) -> Tuple[Decimal, Dict[str, Decimal]]:
        """Calculate weighted composite corruption score.

        Normalizes all four indices to a 0-1 corruption scale and
        computes the weighted average. Higher score = more corrupt.

        Weights:
            CPI: 35%, WGI CC: 25%, Bribery: 20%, Institutional: 20%

        Args:
            cpi: Raw CPI score (0-100).
            wgi_cc: Raw WGI CC estimate (-2.5 to +2.5).
            bribery: Raw bribery risk score (0-100).
            institutional: Raw institutional quality score (0-100).

        Returns:
            Tuple of (composite_score, normalized_components dict).
        """
        cpi_norm = self._normalize_cpi(cpi)
        wgi_norm = self._normalize_wgi(wgi_cc)
        bribery_norm = self._normalize_bribery(bribery)
        inst_norm = self._normalize_institutional(institutional)

        composite = (
            WEIGHT_CPI * cpi_norm
            + WEIGHT_WGI * wgi_norm
            + WEIGHT_BRIBERY * bribery_norm
            + WEIGHT_INSTITUTIONAL * inst_norm
        )
        composite = _clamp_decimal(
            composite.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
            Decimal("0"),
            Decimal("1"),
        )

        normalized = {
            "cpi_normalized": cpi_norm,
            "wgi_normalized": wgi_norm,
            "bribery_normalized": bribery_norm,
            "institutional_normalized": inst_norm,
        }

        return composite, normalized

    def _map_to_eudr_classification(
        self,
        composite_score: Decimal,
    ) -> str:
        """Map composite corruption score to EUDR Article 29 classification.

        Classification rules:
            - LOW_RISK:      composite < 0.25
            - STANDARD_RISK: 0.25 <= composite < 0.60
            - HIGH_RISK:     composite >= 0.60

        Args:
            composite_score: Weighted composite corruption score (0-1).

        Returns:
            EUDRCountryClassification value string.
        """
        if composite_score < LOW_RISK_THRESHOLD:
            return EUDRCountryClassification.LOW_RISK.value
        elif composite_score < HIGH_RISK_THRESHOLD:
            return EUDRCountryClassification.STANDARD_RISK.value
        else:
            return EUDRCountryClassification.HIGH_RISK.value

    def _determine_due_diligence_level(
        self,
        composite_score: Decimal,
        enhanced_triggers: List[str],
    ) -> str:
        """Determine the required due diligence level.

        Base level is determined by classification. Any enhanced DD
        trigger overrides to ENHANCED regardless of classification.

        Args:
            composite_score: Composite corruption score.
            enhanced_triggers: List of enhanced DD triggers that fired.

        Returns:
            DueDiligenceLevel value string.
        """
        # Any enhanced trigger forces ENHANCED
        if enhanced_triggers:
            return DueDiligenceLevel.ENHANCED.value

        classification = self._map_to_eudr_classification(composite_score)

        if classification == EUDRCountryClassification.LOW_RISK.value:
            return DueDiligenceLevel.SIMPLIFIED.value
        elif classification == EUDRCountryClassification.HIGH_RISK.value:
            return DueDiligenceLevel.ENHANCED.value
        else:
            return DueDiligenceLevel.STANDARD.value

    def _check_enhanced_dd_triggers(
        self,
        country_code: str,
        cpi: Decimal,
        wgi_cc: Decimal,
        bribery: Decimal,
    ) -> List[str]:
        """Check all enhanced due diligence trigger conditions.

        Enhanced DD Triggers:
            1. CPI < 30
            2. WGI CC < -1.0
            3. Annual CPI decline > 5 points
            4. FATF grey/blacklist membership
            5. Active governance crisis

        Args:
            country_code: ISO country code.
            cpi: Raw CPI score.
            wgi_cc: Raw WGI CC estimate.
            bribery: Raw bribery risk score.

        Returns:
            List of trigger description strings that fired.
        """
        triggers: List[str] = []

        # Trigger 1: CPI below 30
        if cpi < ENHANCED_DD_CPI_THRESHOLD:
            triggers.append(
                f"CPI score ({cpi}) is below enhanced DD threshold "
                f"({ENHANCED_DD_CPI_THRESHOLD})"
            )

        # Trigger 2: WGI CC below -1.0
        if wgi_cc < ENHANCED_DD_WGI_THRESHOLD:
            triggers.append(
                f"WGI Control of Corruption ({wgi_cc}) is below enhanced "
                f"DD threshold ({ENHANCED_DD_WGI_THRESHOLD})"
            )

        # Trigger 3: Rapid decline
        with self._lock:
            annual_decline = self._annual_decline_data.get(country_code)
        if annual_decline is not None and annual_decline > ENHANCED_DD_DECLINE_THRESHOLD:
            triggers.append(
                f"CPI declining at {annual_decline} points/year, exceeding "
                f"threshold of {ENHANCED_DD_DECLINE_THRESHOLD} points/year"
            )

        # Trigger 4: FATF grey/blacklist
        if country_code in FATF_BLACK_LIST:
            triggers.append(
                f"{country_code} is on the FATF blacklist "
                "(highest risk for money laundering/terrorist financing)"
            )
        elif country_code in FATF_GREY_LIST:
            triggers.append(
                f"{country_code} is on the FATF grey list "
                "(strategic deficiencies in AML/CFT)"
            )

        # Trigger 5: Governance crisis
        if country_code in GOVERNANCE_CRISIS_COUNTRIES:
            triggers.append(
                f"{country_code} is experiencing an active governance "
                "crisis, conflict, or institutional collapse"
            )

        return triggers

    def _identify_risk_factors(
        self,
        country_code: str,
        cpi: Decimal,
        wgi_cc: Decimal,
        bribery: Decimal,
    ) -> List[Dict[str, Any]]:
        """Identify all active risk factors for a country.

        Args:
            country_code: ISO country code.
            cpi: Raw CPI score.
            wgi_cc: Raw WGI CC estimate.
            bribery: Raw bribery risk score.

        Returns:
            List of active risk factor dictionaries.
        """
        active_factors: List[Dict[str, Any]] = []

        if cpi < Decimal("30"):
            factor = dict(RISK_FACTORS["high_corruption"])
            factor["current_value"] = str(cpi)
            active_factors.append(factor)

        if wgi_cc < Decimal("-1.0"):
            factor = dict(RISK_FACTORS["weak_governance"])
            factor["current_value"] = str(wgi_cc)
            active_factors.append(factor)

        with self._lock:
            decline = self._annual_decline_data.get(country_code)
        if decline is not None and decline > Decimal("5"):
            factor = dict(RISK_FACTORS["rapid_decline"])
            factor["current_value"] = str(decline)
            active_factors.append(factor)

        if country_code in FATF_GREY_LIST or country_code in FATF_BLACK_LIST:
            factor = dict(RISK_FACTORS["fatf_listed"])
            factor["current_value"] = "blacklist" if country_code in FATF_BLACK_LIST else "greylist"
            active_factors.append(factor)

        if country_code in GOVERNANCE_CRISIS_COUNTRIES:
            factor = dict(RISK_FACTORS["governance_crisis"])
            factor["current_value"] = "active"
            active_factors.append(factor)

        if bribery > Decimal("70"):
            factor = dict(RISK_FACTORS["high_bribery"])
            factor["current_value"] = str(bribery)
            active_factors.append(factor)

        # Convert Decimal weights to strings for serialization
        for factor in active_factors:
            if isinstance(factor.get("weight"), Decimal):
                factor["weight"] = str(factor["weight"])

        return active_factors

    def _determine_monitoring_frequency(
        self,
        classification: str,
        enhanced_triggers: List[str],
    ) -> str:
        """Determine required monitoring frequency.

        Args:
            classification: EUDR country classification.
            enhanced_triggers: Active enhanced DD triggers.

        Returns:
            MonitoringFrequency value string.
        """
        if enhanced_triggers and len(enhanced_triggers) >= 3:
            return MonitoringFrequency.CONTINUOUS.value
        elif classification == EUDRCountryClassification.HIGH_RISK.value:
            return MonitoringFrequency.MONTHLY.value
        elif enhanced_triggers:
            return MonitoringFrequency.MONTHLY.value
        elif classification == EUDRCountryClassification.STANDARD_RISK.value:
            return MonitoringFrequency.QUARTERLY.value
        else:
            return MonitoringFrequency.ANNUAL.value

    def _generate_mitigation_requirements(
        self,
        classification: str,
        commodity: Optional[str] = None,
    ) -> List[str]:
        """Generate mitigation requirements based on classification and commodity.

        Args:
            classification: EUDR classification.
            commodity: Optional EUDR commodity type.

        Returns:
            List of mitigation requirement strings.
        """
        general_mitigations: Dict[str, List[str]] = {
            EUDRCountryClassification.LOW_RISK.value: [
                "Maintain standard record-keeping per Article 13 (5 years)",
                "Annual compliance review of corruption indices",
                "Document simplified due diligence decisions",
            ],
            EUDRCountryClassification.STANDARD_RISK.value: [
                "Implement full due diligence system per Article 4",
                "Quarterly corruption index monitoring",
                "Supplier audit program (every 24 months minimum)",
                "Risk mitigation measures per Article 11",
                "Due diligence statements per Article 9",
                "Maintain complete documentation per Article 13 (5 years)",
            ],
            EUDRCountryClassification.HIGH_RISK.value: [
                "Implement enhanced due diligence system with additional controls",
                "Monthly corruption index monitoring with real-time alerts",
                "Annual supplier on-site audits",
                "Third-party verification of deforestation-free status",
                "Enhanced risk mitigation per Article 11",
                "Detailed due diligence statements per Article 9",
                "Full supply chain mapping and traceability",
                "Community engagement and FPIC documentation",
                "Government permit and license verification",
                "Quarterly compliance reporting to management",
                "Maintain complete documentation per Article 13 (5 years)",
            ],
        }

        requirements = list(general_mitigations.get(classification, []))

        # Add commodity-specific requirements
        if commodity and commodity in COMMODITY_DD_REQUIREMENTS:
            dd_level = DueDiligenceLevel.ENHANCED.value if classification == EUDRCountryClassification.HIGH_RISK.value else DueDiligenceLevel.STANDARD.value
            commodity_reqs = COMMODITY_DD_REQUIREMENTS[commodity].get(dd_level, [])
            for req in commodity_reqs:
                if req not in requirements:
                    requirements.append(f"[{commodity}] {req}")

        return requirements

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_compliance_impact(
        self,
        country_code: str,
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform full compliance impact assessment for a country.

        Calculates the composite corruption score, determines EUDR
        classification, checks enhanced DD triggers, identifies risk
        factors, and determines the required due diligence level.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            commodity: Optional EUDR commodity for commodity-specific requirements.

        Returns:
            Dictionary containing ComplianceImpact data plus
            mitigation_requirements, processing_time_ms, and provenance.

        Raises:
            ValueError: If country_code is empty or commodity is invalid.
        """
        start_time = time.monotonic()

        if not country_code or not isinstance(country_code, str):
            raise ValueError("country_code must be a non-empty string")
        country_code = country_code.upper()

        if commodity and commodity not in EUDR_COMMODITIES:
            raise ValueError(
                f"commodity must be one of {sorted(EUDR_COMMODITIES)}, "
                f"got '{commodity}'"
            )

        # Retrieve country data
        data = self._get_country_data(country_code)

        impact = ComplianceImpact(
            impact_id=_generate_id("impact"),
            country_code=country_code,
        )

        if not data:
            impact.warnings.append(
                f"No corruption index data available for {country_code}. "
                "Defaulting to STANDARD_RISK classification."
            )
            impact.eudr_classification = EUDRCountryClassification.STANDARD_RISK.value
            impact.due_diligence_level = DueDiligenceLevel.STANDARD.value
            impact.monitoring_frequency = MonitoringFrequency.QUARTERLY.value
            impact.provenance_hash = _compute_hash(impact)

            processing_time_ms = (time.monotonic() - start_time) * 1000.0
            out = impact.to_dict()
            out["mitigation_requirements"] = self._generate_mitigation_requirements(
                impact.eudr_classification, commodity,
            )
            out["processing_time_ms"] = round(processing_time_ms, 3)
            out["calculation_timestamp"] = _utcnow().isoformat()
            return out

        # Extract raw scores
        cpi = _to_decimal(data.get("cpi", 50))
        wgi_cc = _to_decimal(data.get("wgi_cc", 0))
        bribery = _to_decimal(data.get("bribery", 50))
        institutional = _to_decimal(data.get("institutional", 50))

        # Calculate composite score
        composite, normalized = self._calculate_composite_corruption_score(
            cpi, wgi_cc, bribery, institutional,
        )

        # Check enhanced DD triggers
        enhanced_triggers = self._check_enhanced_dd_triggers(
            country_code, cpi, wgi_cc, bribery,
        )

        # Determine classification and DD level
        classification = self._map_to_eudr_classification(composite)
        dd_level = self._determine_due_diligence_level(composite, enhanced_triggers)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            country_code, cpi, wgi_cc, bribery,
        )

        # Determine monitoring frequency
        monitoring_freq = self._determine_monitoring_frequency(
            classification, enhanced_triggers,
        )

        # Populate impact
        impact.corruption_composite_score = composite
        impact.eudr_classification = classification
        impact.due_diligence_level = dd_level
        impact.risk_factors = risk_factors
        impact.enhanced_dd_triggers = enhanced_triggers
        impact.monitoring_frequency = monitoring_freq
        impact.cpi_score = cpi
        impact.wgi_cc_score = wgi_cc
        impact.bribery_score = bribery
        impact.institutional_score = institutional
        impact.cpi_normalized = normalized["cpi_normalized"]
        impact.wgi_normalized = normalized["wgi_normalized"]
        impact.bribery_normalized = normalized["bribery_normalized"]
        impact.institutional_normalized = normalized["institutional_normalized"]
        impact.provenance_hash = _compute_hash(impact)

        # Generate mitigation requirements
        mitigation_reqs = self._generate_mitigation_requirements(
            classification, commodity,
        )

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        out = impact.to_dict()
        out["mitigation_requirements"] = mitigation_reqs
        out["commodity"] = commodity
        out["processing_time_ms"] = round(processing_time_ms, 3)
        out["calculation_timestamp"] = _utcnow().isoformat()

        logger.info(
            "Compliance impact for %s: composite=%s classification=%s "
            "dd_level=%s triggers=%d factors=%d time_ms=%.1f",
            country_code, composite, classification, dd_level,
            len(enhanced_triggers), len(risk_factors), processing_time_ms,
        )
        return out

    def get_country_impact_profile(
        self,
        country_code: str,
    ) -> Dict[str, Any]:
        """Get detailed impact profile covering all commodities.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Dictionary with CountryImpactProfile data.

        Raises:
            ValueError: If country_code is empty.
        """
        start_time = time.monotonic()

        if not country_code or not isinstance(country_code, str):
            raise ValueError("country_code must be a non-empty string")
        country_code = country_code.upper()

        # Get base impact assessment
        base_impact = self.assess_compliance_impact(country_code)
        dd_level = base_impact.get("due_diligence_level", "STANDARD")

        # Generate per-commodity requirements
        commodity_requirements: Dict[str, List[str]] = {}
        for commodity in sorted(EUDR_COMMODITIES):
            reqs = COMMODITY_DD_REQUIREMENTS.get(commodity, {}).get(dd_level, [])
            commodity_requirements[commodity] = reqs

        # Generate recommendations
        recommendations = self._generate_recommendations(
            country_code, base_impact,
        )

        profile = CountryImpactProfile(
            profile_id=_generate_id("profile"),
            country_code=country_code,
            impact=base_impact,
            commodity_requirements=commodity_requirements,
            recommendations=recommendations,
            total_recommendations=len(recommendations),
            critical_recommendations=sum(
                1 for r in recommendations
                if r.get("priority") == "CRITICAL"
            ),
        )
        profile.provenance_hash = _compute_hash(profile)

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        out = profile.to_dict()
        out["processing_time_ms"] = round(processing_time_ms, 3)
        out["calculation_timestamp"] = _utcnow().isoformat()

        logger.info(
            "Impact profile for %s: %d recommendations (%d critical) "
            "time_ms=%.1f",
            country_code, len(recommendations),
            profile.critical_recommendations, processing_time_ms,
        )
        return out

    def get_dd_recommendations(
        self,
        country_code: str,
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate due diligence recommendations for a country/commodity.

        Provides prioritized, actionable recommendations based on the
        country's corruption profile and the specific commodity.

        Args:
            country_code: ISO country code.
            commodity: Optional EUDR commodity type.

        Returns:
            Dictionary with prioritized DDRecommendation list.

        Raises:
            ValueError: If country_code is empty or commodity is invalid.
        """
        start_time = time.monotonic()

        if not country_code or not isinstance(country_code, str):
            raise ValueError("country_code must be a non-empty string")
        country_code = country_code.upper()

        if commodity and commodity not in EUDR_COMMODITIES:
            raise ValueError(
                f"commodity must be one of {sorted(EUDR_COMMODITIES)}"
            )

        # Get base impact
        base_impact = self.assess_compliance_impact(country_code, commodity)
        recommendations = self._generate_recommendations(
            country_code, base_impact, commodity,
        )

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "country_code": country_code,
            "commodity": commodity,
            "eudr_classification": base_impact.get("eudr_classification"),
            "due_diligence_level": base_impact.get("due_diligence_level"),
            "total_recommendations": len(recommendations),
            "critical_count": sum(1 for r in recommendations if r.get("priority") == "CRITICAL"),
            "high_count": sum(1 for r in recommendations if r.get("priority") == "HIGH"),
            "recommendations": recommendations,
            "processing_time_ms": round(processing_time_ms, 3),
            "calculation_timestamp": _utcnow().isoformat(),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "DD recommendations for %s/%s: %d recommendations time_ms=%.1f",
            country_code, commodity, len(recommendations), processing_time_ms,
        )
        return result

    def classify_country(
        self,
        country_code: str,
    ) -> Dict[str, Any]:
        """Classify a country under EUDR Article 29 benchmarking.

        Simplified method that returns just the classification and
        DD level without full impact assessment details.

        Args:
            country_code: ISO country code.

        Returns:
            Dictionary with classification, dd_level, and composite_score.

        Raises:
            ValueError: If country_code is empty.
        """
        start_time = time.monotonic()

        if not country_code or not isinstance(country_code, str):
            raise ValueError("country_code must be a non-empty string")
        country_code = country_code.upper()

        data = self._get_country_data(country_code)
        warnings_list: List[str] = []

        if not data:
            warnings_list.append(f"No data for {country_code}, defaulting to STANDARD_RISK")
            processing_time_ms = (time.monotonic() - start_time) * 1000.0
            result = {
                "country_code": country_code,
                "eudr_classification": EUDRCountryClassification.STANDARD_RISK.value,
                "due_diligence_level": DueDiligenceLevel.STANDARD.value,
                "composite_score": "N/A",
                "warnings": warnings_list,
                "processing_time_ms": round(processing_time_ms, 3),
                "provenance_hash": "",
            }
            result["provenance_hash"] = _compute_hash(result)
            return result

        cpi = _to_decimal(data.get("cpi", 50))
        wgi_cc = _to_decimal(data.get("wgi_cc", 0))
        bribery = _to_decimal(data.get("bribery", 50))
        institutional = _to_decimal(data.get("institutional", 50))

        composite, _ = self._calculate_composite_corruption_score(
            cpi, wgi_cc, bribery, institutional,
        )
        enhanced_triggers = self._check_enhanced_dd_triggers(
            country_code, cpi, wgi_cc, bribery,
        )
        classification = self._map_to_eudr_classification(composite)
        dd_level = self._determine_due_diligence_level(composite, enhanced_triggers)

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "country_code": country_code,
            "eudr_classification": classification,
            "due_diligence_level": dd_level,
            "composite_score": str(composite),
            "enhanced_dd_triggered": len(enhanced_triggers) > 0,
            "enhanced_dd_trigger_count": len(enhanced_triggers),
            "warnings": warnings_list,
            "processing_time_ms": round(processing_time_ms, 3),
            "calculation_timestamp": _utcnow().isoformat(),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Classification for %s: %s (composite=%s, dd=%s) time_ms=%.1f",
            country_code, classification, composite, dd_level,
            processing_time_ms,
        )
        return result

    def batch_classify_countries(
        self,
        country_codes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Classify multiple countries in batch.

        Args:
            country_codes: List of country codes (None = all reference countries).

        Returns:
            Dictionary with per-country classifications and summary.
        """
        start_time = time.monotonic()

        if country_codes is None:
            country_codes = sorted(REFERENCE_COUNTRY_DATA.keys())
            with self._lock:
                for cc in self._custom_data:
                    if cc not in country_codes:
                        country_codes.append(cc)

        classifications: List[Dict[str, Any]] = []
        class_counts = {
            EUDRCountryClassification.LOW_RISK.value: 0,
            EUDRCountryClassification.STANDARD_RISK.value: 0,
            EUDRCountryClassification.HIGH_RISK.value: 0,
        }

        for cc in country_codes:
            try:
                result = self.classify_country(cc)
                classifications.append(result)
                cls = result.get("eudr_classification", "STANDARD_RISK")
                class_counts[cls] = class_counts.get(cls, 0) + 1
            except ValueError as exc:
                classifications.append({
                    "country_code": cc,
                    "error": str(exc),
                    "eudr_classification": "STANDARD_RISK",
                })

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "countries_classified": len(classifications),
            "classification_summary": class_counts,
            "classifications": classifications,
            "processing_time_ms": round(processing_time_ms, 3),
            "calculation_timestamp": _utcnow().isoformat(),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Batch classification: %d countries classified: %s time_ms=%.1f",
            len(classifications), class_counts, processing_time_ms,
        )
        return result

    def get_commodity_requirements(
        self,
        commodity: str,
        dd_level: str = "STANDARD",
    ) -> Dict[str, Any]:
        """Get DD requirements for a specific commodity and DD level.

        Args:
            commodity: EUDR commodity type.
            dd_level: Due diligence level (SIMPLIFIED, STANDARD, ENHANCED).

        Returns:
            Dictionary with commodity-specific requirements.

        Raises:
            ValueError: If commodity or dd_level is invalid.
        """
        if commodity not in EUDR_COMMODITIES:
            raise ValueError(f"commodity must be one of {sorted(EUDR_COMMODITIES)}")
        valid_levels = {"SIMPLIFIED", "STANDARD", "ENHANCED"}
        if dd_level not in valid_levels:
            raise ValueError(f"dd_level must be one of {sorted(valid_levels)}")

        requirements = COMMODITY_DD_REQUIREMENTS.get(commodity, {}).get(dd_level, [])

        return {
            "commodity": commodity,
            "dd_level": dd_level,
            "requirement_count": len(requirements),
            "requirements": requirements,
            "provenance_hash": _compute_hash(
                {"commodity": commodity, "dd_level": dd_level}
            ),
        }

    # ------------------------------------------------------------------
    # Recommendation Generation
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        country_code: str,
        impact: Dict[str, Any],
        commodity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate prioritized DD recommendations based on impact assessment.

        Args:
            country_code: ISO country code.
            impact: Impact assessment dictionary.
            commodity: Optional commodity for specific recommendations.

        Returns:
            List of DDRecommendation dictionaries sorted by priority.
        """
        recommendations: List[Dict[str, Any]] = []
        classification = impact.get("eudr_classification", "STANDARD_RISK")
        dd_level = impact.get("due_diligence_level", "STANDARD")
        triggers = impact.get("enhanced_dd_triggers", [])

        # Recommendation 1: Baseline DD system
        if classification == EUDRCountryClassification.HIGH_RISK.value:
            rec = DDRecommendation(
                recommendation_id=_generate_id("rec"),
                country_code=country_code,
                commodity=commodity or "all",
                category="Due Diligence System",
                priority=RecommendationPriority.CRITICAL.value,
                description=(
                    f"Implement enhanced due diligence system for all "
                    f"sourcing from {country_code}. Country is classified "
                    f"as HIGH_RISK under EUDR Article 29."
                ),
                action_items=[
                    "Establish dedicated compliance team for this source country",
                    "Implement real-time corruption index monitoring",
                    "Set up enhanced supplier verification workflows",
                    "Commission third-party compliance audits",
                ],
                regulatory_reference="EUDR Articles 4, 10, 11",
                estimated_effort="High (40-80 person-hours)",
                deadline_guidance="Within 30 days",
            )
            rec.provenance_hash = _compute_hash(rec)
            recommendations.append(rec.to_dict())
        elif classification == EUDRCountryClassification.STANDARD_RISK.value:
            rec = DDRecommendation(
                recommendation_id=_generate_id("rec"),
                country_code=country_code,
                commodity=commodity or "all",
                category="Due Diligence System",
                priority=RecommendationPriority.HIGH.value,
                description=(
                    f"Maintain standard due diligence system for sourcing "
                    f"from {country_code}. Country is classified as "
                    f"STANDARD_RISK under EUDR Article 29."
                ),
                action_items=[
                    "Ensure due diligence system meets Article 4 requirements",
                    "Implement quarterly corruption index monitoring",
                    "Schedule biennial supplier audits",
                ],
                regulatory_reference="EUDR Articles 4, 10",
                estimated_effort="Medium (20-40 person-hours)",
                deadline_guidance="Within 60 days",
            )
            rec.provenance_hash = _compute_hash(rec)
            recommendations.append(rec.to_dict())

        # Recommendation 2: Enhanced DD triggers
        if triggers:
            rec = DDRecommendation(
                recommendation_id=_generate_id("rec"),
                country_code=country_code,
                commodity=commodity or "all",
                category="Enhanced Due Diligence",
                priority=RecommendationPriority.CRITICAL.value,
                description=(
                    f"{len(triggers)} enhanced DD triggers active for "
                    f"{country_code}. Enhanced due diligence is mandatory."
                ),
                action_items=[
                    f"Address trigger: {trigger}" for trigger in triggers
                ] + [
                    "Escalate to senior compliance management",
                    "Consider source country diversification",
                ],
                regulatory_reference="EUDR Article 10, 11",
                estimated_effort="High (60-120 person-hours)",
                deadline_guidance="Immediate (within 7 days)",
            )
            rec.provenance_hash = _compute_hash(rec)
            recommendations.append(rec.to_dict())

        # Recommendation 3: Supplier verification
        if dd_level in ("STANDARD", "ENHANCED"):
            priority = (
                RecommendationPriority.CRITICAL.value
                if dd_level == "ENHANCED"
                else RecommendationPriority.HIGH.value
            )
            rec = DDRecommendation(
                recommendation_id=_generate_id("rec"),
                country_code=country_code,
                commodity=commodity or "all",
                category="Supplier Verification",
                priority=priority,
                description=(
                    f"Verify all suppliers in {country_code} meet EUDR "
                    f"deforestation-free requirements."
                ),
                action_items=[
                    "Collect GPS coordinates for all supply points",
                    "Verify supplier deforestation-free declarations",
                    "Cross-reference with satellite monitoring data",
                    "Check supplier certification status",
                ],
                regulatory_reference="EUDR Articles 9, 10",
                estimated_effort="Medium-High (30-60 person-hours per supplier)",
                deadline_guidance="Within 30 days for enhanced, 90 days for standard",
            )
            rec.provenance_hash = _compute_hash(rec)
            recommendations.append(rec.to_dict())

        # Recommendation 4: Record keeping
        rec = DDRecommendation(
            recommendation_id=_generate_id("rec"),
            country_code=country_code,
            commodity=commodity or "all",
            category="Record Keeping",
            priority=RecommendationPriority.MEDIUM.value,
            description=(
                "Ensure all due diligence records are maintained for "
                "minimum 5 years per EUDR Article 13."
            ),
            action_items=[
                "Verify document management system meets 5-year retention",
                "Ensure corruption index monitoring history is archived",
                "Maintain audit trail of all compliance decisions",
                "Store provenance hashes for regulatory audit readiness",
            ],
            regulatory_reference="EUDR Article 13",
            estimated_effort="Low (5-10 person-hours)",
            deadline_guidance="Within 90 days",
        )
        rec.provenance_hash = _compute_hash(rec)
        recommendations.append(rec.to_dict())

        # Recommendation 5: Commodity-specific (if applicable)
        if commodity and commodity in COMMODITY_DD_REQUIREMENTS:
            commodity_reqs = COMMODITY_DD_REQUIREMENTS[commodity].get(dd_level, [])
            if commodity_reqs:
                rec = DDRecommendation(
                    recommendation_id=_generate_id("rec"),
                    country_code=country_code,
                    commodity=commodity,
                    category=f"Commodity-Specific ({commodity})",
                    priority=RecommendationPriority.HIGH.value,
                    description=(
                        f"Implement {commodity}-specific due diligence "
                        f"requirements for {dd_level} level."
                    ),
                    action_items=list(commodity_reqs),
                    regulatory_reference="EUDR Articles 10, 11, Annex I",
                    estimated_effort="Medium (20-40 person-hours)",
                    deadline_guidance="Within 60 days",
                )
                rec.provenance_hash = _compute_hash(rec)
                recommendations.append(rec.to_dict())

        # Sort by priority (CRITICAL first, then HIGH, MEDIUM, LOW)
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recommendations.sort(
            key=lambda r: priority_order.get(r.get("priority", "LOW"), 3)
        )

        return recommendations
