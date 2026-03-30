# -*- coding: utf-8 -*-
"""
SubstitutionRiskAnalyzer - AGENT-EUDR-018 Engine 5: Commodity Substitution Detection

Detects commodity switching patterns and substitution fraud across EUDR-regulated
supply chains. Identifies when suppliers change declared commodities without
proper disclosure, detects greenwashing through certification misrepresentation,
and quantifies the risk impact of commodity substitution events.

Zero-Hallucination Guarantees:
    - All risk scoring uses deterministic arithmetic (Decimal, no ML/LLM).
    - Statistical anomaly detection uses explicit threshold comparisons.
    - Greenwashing detection uses certification registry lookups.
    - Cross-commodity risk matrix uses static regulatory lookup tables.
    - SHA-256 provenance hashes on all output objects.

Detection Methods:
    1. Volume Anomaly Detection: Z-score analysis of shipment volumes to detect
       sudden changes indicative of commodity substitution.
    2. Declaration Consistency: Cross-referencing declared commodities against
       satellite imagery, certifications, and trade records.
    3. Seasonal Pattern Analysis: Detecting commodity switching that follows
       seasonal harvesting/planting cycles.
    4. Greenwashing Detection: Verifying sustainability claims against
       recognized certification bodies (RSPO, FSC, Rainforest Alliance, UTZ).
    5. Commodity Pair Risk Matrix: Static risk table for switching between
       specific commodity pairs (e.g., soya-to-cattle = HIGH).

Alert Severity Levels:
    - INFO:     Low-confidence anomaly, no immediate action required.
    - WARNING:  Moderate-confidence anomaly, investigation recommended.
    - HIGH:     High-confidence substitution pattern detected.
    - CRITICAL: Confirmed substitution or greenwashing with evidence.

Performance Targets:
    - Single supplier substitution check: <100ms
    - Batch screening (100 suppliers): <5s
    - Risk score calculation: <20ms
    - Greenwashing detection: <50ms

Regulatory References:
    - EUDR Article 3: Prohibition on non-compliant commodities
    - EUDR Article 4: Operator due diligence obligations
    - EUDR Article 9: Due diligence statements (declaration accuracy)
    - EUDR Article 10: Information requirements (commodity declaration)
    - EUDR Article 24: Penalties for false declarations

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-018, Engine 5 (Substitution Risk Analyzer)
Agent ID: GL-EUDR-CRA-018
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def _generate_id(prefix: str = "sub") -> str:
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: EUDR-regulated commodity types.
EUDR_COMMODITIES: frozenset = frozenset({
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
})

#: Minimum risk score (Decimal).
MIN_RISK_SCORE: Decimal = Decimal("0")

#: Maximum risk score (Decimal).
MAX_RISK_SCORE: Decimal = Decimal("100")

#: Default time window for switching pattern analysis (days).
DEFAULT_TIME_WINDOW_DAYS: int = 365

#: Z-score threshold for volume anomaly detection.
VOLUME_ANOMALY_Z_THRESHOLD: Decimal = Decimal("2.5")

#: Minimum number of historical records needed for statistical analysis.
MIN_HISTORY_RECORDS: int = 3

#: Alert severity levels in ascending order.
ALERT_SEVERITIES: Tuple[str, ...] = ("INFO", "WARNING", "HIGH", "CRITICAL")

#: Risk score thresholds for severity classification.
SEVERITY_THRESHOLDS: Dict[str, Decimal] = {
    "INFO": Decimal("20"),
    "WARNING": Decimal("40"),
    "HIGH": Decimal("65"),
    "CRITICAL": Decimal("85"),
}

#: Recognized sustainability certification bodies.
RECOGNIZED_CERTIFICATIONS: Dict[str, Dict[str, Any]] = {
    "RSPO": {
        "full_name": "Roundtable on Sustainable Palm Oil",
        "commodities": ["oil_palm"],
        "url": "https://rspo.org",
        "verification_method": "certificate_number",
    },
    "FSC": {
        "full_name": "Forest Stewardship Council",
        "commodities": ["wood", "rubber"],
        "url": "https://fsc.org",
        "verification_method": "chain_of_custody_code",
    },
    "PEFC": {
        "full_name": "Programme for Endorsement of Forest Certification",
        "commodities": ["wood"],
        "url": "https://pefc.org",
        "verification_method": "certificate_number",
    },
    "RAINFOREST_ALLIANCE": {
        "full_name": "Rainforest Alliance Certified",
        "commodities": ["cocoa", "coffee"],
        "url": "https://rainforest-alliance.org",
        "verification_method": "certificate_number",
    },
    "UTZ": {
        "full_name": "UTZ Certified (now Rainforest Alliance)",
        "commodities": ["cocoa", "coffee"],
        "url": "https://utz.org",
        "verification_method": "certificate_number",
    },
    "FAIRTRADE": {
        "full_name": "Fairtrade International",
        "commodities": ["cocoa", "coffee", "soya"],
        "url": "https://fairtrade.net",
        "verification_method": "flo_id",
    },
    "ISCC": {
        "full_name": "International Sustainability & Carbon Certification",
        "commodities": ["oil_palm", "soya"],
        "url": "https://iscc-system.org",
        "verification_method": "certificate_number",
    },
    "RTRS": {
        "full_name": "Round Table on Responsible Soy",
        "commodities": ["soya"],
        "url": "https://responsiblesoy.org",
        "verification_method": "certificate_number",
    },
}

# ---------------------------------------------------------------------------
# Cross-Commodity Substitution Risk Matrix
# ---------------------------------------------------------------------------
# Each pair has a risk_weight (Decimal) and a reason string.
# Higher weight = higher risk when switching from commodity A to B.

SUBSTITUTION_RISK_MATRIX: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("soya", "cattle"): {
        "risk_weight": Decimal("85"),
        "reason": "Soya-to-cattle switch often indicates deforestation for "
                  "pasture followed by feed production claim",
        "regulatory_concern": "HIGH",
    },
    ("cattle", "soya"): {
        "risk_weight": Decimal("80"),
        "reason": "Cattle-to-soya switch may mask pasture expansion as "
                  "crop production",
        "regulatory_concern": "HIGH",
    },
    ("oil_palm", "rubber"): {
        "risk_weight": Decimal("70"),
        "reason": "Both plantation crops; switching may obscure "
                  "deforestation timeline",
        "regulatory_concern": "MEDIUM",
    },
    ("rubber", "oil_palm"): {
        "risk_weight": Decimal("72"),
        "reason": "Rubber-to-palm switch in SE Asia associated with "
                  "high deforestation risk",
        "regulatory_concern": "HIGH",
    },
    ("wood", "oil_palm"): {
        "risk_weight": Decimal("90"),
        "reason": "Wood harvesting followed by palm plantation is classic "
                  "deforestation pathway in Indonesia/Malaysia",
        "regulatory_concern": "CRITICAL",
    },
    ("oil_palm", "wood"): {
        "risk_weight": Decimal("65"),
        "reason": "Palm-to-wood switch may indicate salvage logging",
        "regulatory_concern": "MEDIUM",
    },
    ("wood", "soya"): {
        "risk_weight": Decimal("88"),
        "reason": "Forest clearing for soya expansion is a major "
                  "deforestation driver in Brazil",
        "regulatory_concern": "CRITICAL",
    },
    ("soya", "wood"): {
        "risk_weight": Decimal("55"),
        "reason": "Soya-to-wood switch less common but may indicate "
                  "land use change reporting issues",
        "regulatory_concern": "MEDIUM",
    },
    ("cocoa", "coffee"): {
        "risk_weight": Decimal("30"),
        "reason": "Both shade-grown crops in similar regions; switch is "
                  "lower risk but still requires disclosure",
        "regulatory_concern": "LOW",
    },
    ("coffee", "cocoa"): {
        "risk_weight": Decimal("30"),
        "reason": "Similar risk profile to cocoa-to-coffee switch",
        "regulatory_concern": "LOW",
    },
    ("wood", "cattle"): {
        "risk_weight": Decimal("92"),
        "reason": "Forest-to-pasture conversion is the highest "
                  "deforestation driver globally",
        "regulatory_concern": "CRITICAL",
    },
    ("cattle", "wood"): {
        "risk_weight": Decimal("50"),
        "reason": "Cattle-to-wood may indicate reforestation or "
                  "silvopastoral system adoption",
        "regulatory_concern": "LOW",
    },
    ("cocoa", "oil_palm"): {
        "risk_weight": Decimal("75"),
        "reason": "Cocoa-to-palm switch in West Africa associated with "
                  "encroachment into cocoa belt forests",
        "regulatory_concern": "HIGH",
    },
    ("oil_palm", "cocoa"): {
        "risk_weight": Decimal("60"),
        "reason": "Palm-to-cocoa switch may indicate crop diversification "
                  "or land use change",
        "regulatory_concern": "MEDIUM",
    },
    ("rubber", "wood"): {
        "risk_weight": Decimal("45"),
        "reason": "Rubber-to-wood switch may indicate plantation end-of-life "
                  "timber harvesting",
        "regulatory_concern": "LOW",
    },
    ("wood", "rubber"): {
        "risk_weight": Decimal("78"),
        "reason": "Forest-to-rubber conversion widespread in Southeast Asia",
        "regulatory_concern": "HIGH",
    },
}

#: Seasonal commodity switching risk factors by quarter.
SEASONAL_RISK_FACTORS: Dict[str, Dict[int, Decimal]] = {
    "cattle": {1: Decimal("1.0"), 2: Decimal("1.1"), 3: Decimal("1.2"), 4: Decimal("1.0")},
    "cocoa": {1: Decimal("0.9"), 2: Decimal("1.0"), 3: Decimal("1.3"), 4: Decimal("1.1")},
    "coffee": {1: Decimal("1.1"), 2: Decimal("0.9"), 3: Decimal("1.0"), 4: Decimal("1.2")},
    "oil_palm": {1: Decimal("1.0"), 2: Decimal("1.2"), 3: Decimal("1.1"), 4: Decimal("1.0")},
    "rubber": {1: Decimal("1.0"), 2: Decimal("1.0"), 3: Decimal("1.1"), 4: Decimal("0.9")},
    "soya": {1: Decimal("0.8"), 2: Decimal("1.3"), 3: Decimal("1.2"), 4: Decimal("0.9")},
    "wood": {1: Decimal("1.0"), 2: Decimal("1.0"), 3: Decimal("0.9"), 4: Decimal("1.1")},
}

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SubstitutionAlert:
    """A substitution risk alert record.

    Attributes:
        alert_id: Unique alert identifier.
        supplier_id: Supplier under investigation.
        severity: Alert severity (INFO, WARNING, HIGH, CRITICAL).
        declared_commodity: Commodity declared by the supplier.
        suspected_commodity: Commodity suspected based on evidence.
        risk_score: Risk score (Decimal 0-100).
        description: Human-readable alert description.
        evidence_summary: Supporting evidence items.
        detection_method: Method that triggered the alert.
        created_at: UTC timestamp of alert creation.
        acknowledged: Whether the alert has been acknowledged.
        acknowledged_at: UTC timestamp of acknowledgement.
        provenance_hash: SHA-256 hash for audit trail.
    """

    alert_id: str = ""
    supplier_id: str = ""
    severity: str = "INFO"
    declared_commodity: str = ""
    suspected_commodity: str = ""
    risk_score: Decimal = Decimal("0")
    description: str = ""
    evidence_summary: List[str] = field(default_factory=list)
    detection_method: str = ""
    created_at: str = ""
    acknowledged: bool = False
    acknowledged_at: Optional[str] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON/provenance.

        Returns:
            Dictionary representation.
        """
        return {
            "alert_id": self.alert_id,
            "supplier_id": self.supplier_id,
            "severity": self.severity,
            "declared_commodity": self.declared_commodity,
            "suspected_commodity": self.suspected_commodity,
            "risk_score": str(self.risk_score),
            "description": self.description,
            "evidence_summary": self.evidence_summary,
            "detection_method": self.detection_method,
            "created_at": self.created_at,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at,
            "provenance_hash": self.provenance_hash,
        }

@dataclass
class SwitchingPattern:
    """Detected commodity switching pattern for a supplier.

    Attributes:
        pattern_id: Unique pattern identifier.
        supplier_id: Supplier under analysis.
        time_window_days: Analysis window in days.
        commodity_transitions: Ordered list of (from, to, date) transitions.
        transition_count: Total transitions detected.
        dominant_commodity: Most frequently declared commodity.
        anomaly_score: Statistical anomaly score (Decimal 0-100).
        seasonal_correlation: Whether pattern correlates with seasons.
        risk_assessment: Overall risk assessment string.
        provenance_hash: SHA-256 hash.
    """

    pattern_id: str = ""
    supplier_id: str = ""
    time_window_days: int = 365
    commodity_transitions: List[Dict[str, str]] = field(default_factory=list)
    transition_count: int = 0
    dominant_commodity: str = ""
    anomaly_score: Decimal = Decimal("0")
    seasonal_correlation: bool = False
    risk_assessment: str = "LOW"
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "pattern_id": self.pattern_id,
            "supplier_id": self.supplier_id,
            "time_window_days": self.time_window_days,
            "commodity_transitions": self.commodity_transitions,
            "transition_count": self.transition_count,
            "dominant_commodity": self.dominant_commodity,
            "anomaly_score": str(self.anomaly_score),
            "seasonal_correlation": self.seasonal_correlation,
            "risk_assessment": self.risk_assessment,
            "provenance_hash": self.provenance_hash,
        }

@dataclass
class GreenwashingResult:
    """Result of a greenwashing detection analysis.

    Attributes:
        result_id: Unique result identifier.
        declared_commodity: Commodity as declared.
        declared_sustainability: Sustainability claim made.
        certifications_claimed: List of certifications claimed.
        certifications_verified: Certifications that passed verification.
        certifications_failed: Certifications that failed verification.
        greenwashing_detected: Whether greenwashing was detected.
        confidence: Confidence in the detection (Decimal 0.0-1.0).
        risk_factors: Identified risk factors.
        recommendations: Recommended actions.
        provenance_hash: SHA-256 hash.
    """

    result_id: str = ""
    declared_commodity: str = ""
    declared_sustainability: str = ""
    certifications_claimed: List[str] = field(default_factory=list)
    certifications_verified: List[str] = field(default_factory=list)
    certifications_failed: List[str] = field(default_factory=list)
    greenwashing_detected: bool = False
    confidence: Decimal = Decimal("0.0")
    risk_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "result_id": self.result_id,
            "declared_commodity": self.declared_commodity,
            "declared_sustainability": self.declared_sustainability,
            "certifications_claimed": self.certifications_claimed,
            "certifications_verified": self.certifications_verified,
            "certifications_failed": self.certifications_failed,
            "greenwashing_detected": self.greenwashing_detected,
            "confidence": str(self.confidence),
            "risk_factors": self.risk_factors,
            "recommendations": self.recommendations,
            "provenance_hash": self.provenance_hash,
        }

# ---------------------------------------------------------------------------
# SubstitutionRiskAnalyzer
# ---------------------------------------------------------------------------

class SubstitutionRiskAnalyzer:
    """Production-grade commodity substitution risk detection for EUDR compliance.

    Detects commodity switching patterns, substitution fraud, and greenwashing
    across EUDR-regulated supply chains. Uses statistical anomaly detection,
    certification verification, cross-commodity risk matrices, and seasonal
    pattern analysis to quantify substitution risk per supplier.

    Thread Safety:
        All mutable state is protected by a reentrant lock. Multiple threads
        can safely call any public method concurrently.

    Zero-Hallucination:
        All risk scores are computed via Decimal arithmetic with deterministic
        formulas. No ML/LLM models are used in any calculation path.

    Attributes:
        _supplier_history: In-memory store of supplier commodity declarations.
        _alerts: In-memory alert store keyed by alert_id.
        _lock: Reentrant lock for thread-safe state access.

    Example:
        >>> analyzer = SubstitutionRiskAnalyzer()
        >>> result = analyzer.detect_substitution(
        ...     supplier_id="SUP-001",
        ...     commodity_history=[
        ...         {"commodity": "soya", "date": "2025-01-15", "volume": 100},
        ...         {"commodity": "cattle", "date": "2025-06-20", "volume": 80},
        ...     ],
        ...     current_declaration={"commodity": "cattle", "volume": 90},
        ... )
        >>> assert result["substitution_detected"] in (True, False)
    """

    def __init__(self) -> None:
        """Initialize SubstitutionRiskAnalyzer with empty state stores."""
        self._supplier_history: Dict[str, List[Dict[str, Any]]] = {}
        self._alerts: Dict[str, SubstitutionAlert] = {}
        self._lock: threading.RLock = threading.RLock()
        logger.info(
            "SubstitutionRiskAnalyzer initialized (version=%s)",
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_substitution(
        self,
        supplier_id: str,
        commodity_history: List[Dict[str, Any]],
        current_declaration: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Detect if a supplier switched commodities without proper disclosure.

        Analyzes the supplier's commodity declaration history against the
        current declaration to identify undisclosed commodity substitution.
        Uses volume anomaly detection and commodity transition analysis.

        Args:
            supplier_id: Unique supplier identifier.
            commodity_history: List of historical commodity declarations, each
                containing at minimum ``commodity`` (str), ``date`` (str ISO),
                and ``volume`` (numeric).
            current_declaration: Current declaration with ``commodity`` (str)
                and ``volume`` (numeric).

        Returns:
            Dictionary with keys:
                - substitution_detected (bool)
                - risk_score (str, Decimal 0-100)
                - previous_commodity (str or None)
                - current_commodity (str)
                - confidence (str, Decimal 0.0-1.0)
                - volume_anomaly (bool)
                - evidence (list of str)
                - alert (dict or None)
                - processing_time_ms (float)
                - provenance_hash (str)

        Raises:
            ValueError: If supplier_id is empty, commodity_history is not a list,
                or current_declaration is missing required fields.
        """
        start_time = time.monotonic()

        # -- Input validation -----------------------------------------------
        if not supplier_id or not isinstance(supplier_id, str):
            raise ValueError("supplier_id must be a non-empty string")
        if not isinstance(commodity_history, list):
            raise ValueError("commodity_history must be a list")
        if not isinstance(current_declaration, dict):
            raise ValueError("current_declaration must be a dict")
        current_commodity = current_declaration.get("commodity", "")
        if not current_commodity:
            raise ValueError("current_declaration must include 'commodity'")
        if current_commodity not in EUDR_COMMODITIES:
            raise ValueError(
                f"current_declaration commodity '{current_commodity}' is not "
                f"a valid EUDR commodity. Valid: {sorted(EUDR_COMMODITIES)}"
            )

        # -- Store history --------------------------------------------------
        with self._lock:
            self._supplier_history[supplier_id] = list(commodity_history)

        # -- Analyze transitions --------------------------------------------
        evidence: List[str] = []
        substitution_detected = False
        previous_commodity: Optional[str] = None
        volume_anomaly = False
        risk_score = Decimal("0")
        confidence = Decimal("0.0")

        # Sort history by date
        sorted_history = sorted(
            commodity_history,
            key=lambda r: r.get("date", ""),
        )

        if sorted_history:
            last_record = sorted_history[-1]
            previous_commodity = last_record.get("commodity")

            # Detect commodity switch
            if previous_commodity and previous_commodity != current_commodity:
                substitution_detected = True
                evidence.append(
                    f"Commodity switch detected: {previous_commodity} -> "
                    f"{current_commodity}"
                )

                # Cross-commodity risk from static matrix
                pair_key = (previous_commodity, current_commodity)
                pair_risk = SUBSTITUTION_RISK_MATRIX.get(pair_key)
                if pair_risk is not None:
                    risk_score = pair_risk["risk_weight"]
                    evidence.append(
                        f"Cross-commodity risk: {pair_risk['reason']}"
                    )
                else:
                    # Default risk for unknown pair
                    risk_score = Decimal("50")
                    evidence.append(
                        "Cross-commodity risk: pair not in risk matrix, "
                        "default risk applied"
                    )

                confidence = Decimal("0.7")

            # Volume anomaly detection
            volume_anomaly_result = self._detect_volume_anomaly(
                sorted_history, current_declaration,
            )
            if volume_anomaly_result["anomaly_detected"]:
                volume_anomaly = True
                risk_score = min(
                    MAX_RISK_SCORE,
                    risk_score + Decimal("15"),
                )
                evidence.append(
                    f"Volume anomaly: z-score={volume_anomaly_result['z_score']}"
                )
                confidence = min(Decimal("1.0"), confidence + Decimal("0.15"))

        # -- Adjust for declaration count -----------------------------------
        declaration_count = len(sorted_history)
        if declaration_count < MIN_HISTORY_RECORDS:
            confidence = confidence * Decimal("0.6")
            evidence.append(
                f"Low confidence adjustment: only {declaration_count} "
                f"historical records (minimum {MIN_HISTORY_RECORDS})"
            )

        # -- Generate alert if needed ---------------------------------------
        alert_dict: Optional[Dict[str, Any]] = None
        if substitution_detected:
            severity = self._classify_severity(risk_score)
            alert = self._create_alert(
                supplier_id=supplier_id,
                severity=severity,
                declared_commodity=current_commodity,
                suspected_commodity=previous_commodity or "",
                risk_score=risk_score,
                description=f"Commodity substitution detected for supplier "
                            f"{supplier_id}: {previous_commodity} -> "
                            f"{current_commodity}",
                evidence_summary=evidence,
                detection_method="declaration_analysis",
            )
            alert_dict = alert.to_dict()

        # -- Build result ---------------------------------------------------
        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "substitution_detected": substitution_detected,
            "risk_score": str(risk_score.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )),
            "previous_commodity": previous_commodity,
            "current_commodity": current_commodity,
            "confidence": str(confidence.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP,
            )),
            "volume_anomaly": volume_anomaly,
            "evidence": evidence,
            "alert": alert_dict,
            "processing_time_ms": round(processing_time_ms, 3),
            "provenance_hash": "",
        }
        result["provenance_hash"] = self._compute_provenance_hash(result)

        logger.info(
            "Substitution detection completed for supplier=%s detected=%s "
            "risk=%s time_ms=%.1f",
            supplier_id, substitution_detected, result["risk_score"],
            processing_time_ms,
        )
        return result

    def analyze_switching_pattern(
        self,
        supplier_id: str,
        time_window_days: int = DEFAULT_TIME_WINDOW_DAYS,
    ) -> Dict[str, Any]:
        """Analyze commodity switching patterns for a supplier over time.

        Examines the supplier's commodity declaration history within the
        specified time window to identify switching patterns, transition
        frequency, and dominant commodity.

        Args:
            supplier_id: Unique supplier identifier.
            time_window_days: Number of days to analyze (default 365).

        Returns:
            Dictionary containing the SwitchingPattern data plus
            processing_time_ms and provenance_hash.

        Raises:
            ValueError: If supplier_id is empty or time_window_days < 1.
        """
        start_time = time.monotonic()

        if not supplier_id or not isinstance(supplier_id, str):
            raise ValueError("supplier_id must be a non-empty string")
        if time_window_days < 1:
            raise ValueError("time_window_days must be >= 1")

        with self._lock:
            history = list(self._supplier_history.get(supplier_id, []))

        # Filter by time window
        cutoff = utcnow() - timedelta(days=time_window_days)
        cutoff_str = cutoff.isoformat()
        filtered = [
            r for r in history
            if r.get("date", "") >= cutoff_str
        ]

        # Sort by date
        filtered.sort(key=lambda r: r.get("date", ""))

        # Detect transitions
        transitions: List[Dict[str, str]] = []
        commodity_counts: Dict[str, int] = {}
        for i, record in enumerate(filtered):
            commodity = record.get("commodity", "unknown")
            commodity_counts[commodity] = commodity_counts.get(commodity, 0) + 1
            if i > 0:
                prev_commodity = filtered[i - 1].get("commodity", "unknown")
                if commodity != prev_commodity:
                    transitions.append({
                        "from": prev_commodity,
                        "to": commodity,
                        "date": record.get("date", ""),
                    })

        # Dominant commodity
        dominant = ""
        max_count = 0
        for comm, count in commodity_counts.items():
            if count > max_count:
                max_count = count
                dominant = comm

        # Anomaly score based on transition frequency
        total_records = len(filtered)
        transition_count = len(transitions)
        if total_records > 1:
            transition_rate = Decimal(str(transition_count)) / Decimal(str(total_records - 1))
            anomaly_score = min(
                MAX_RISK_SCORE,
                (transition_rate * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                ),
            )
        else:
            anomaly_score = Decimal("0")

        # Risk assessment
        if anomaly_score >= Decimal("60"):
            risk_assessment = "CRITICAL"
        elif anomaly_score >= Decimal("40"):
            risk_assessment = "HIGH"
        elif anomaly_score >= Decimal("20"):
            risk_assessment = "MEDIUM"
        else:
            risk_assessment = "LOW"

        pattern = SwitchingPattern(
            pattern_id=_generate_id("pat"),
            supplier_id=supplier_id,
            time_window_days=time_window_days,
            commodity_transitions=transitions,
            transition_count=transition_count,
            dominant_commodity=dominant,
            anomaly_score=anomaly_score,
            seasonal_correlation=False,
            risk_assessment=risk_assessment,
        )
        pattern.provenance_hash = _compute_hash(pattern)

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = pattern.to_dict()
        result["processing_time_ms"] = round(processing_time_ms, 3)
        result["record_count"] = total_records

        logger.info(
            "Switching pattern analysis for supplier=%s transitions=%d "
            "anomaly_score=%s time_ms=%.1f",
            supplier_id, transition_count, anomaly_score, processing_time_ms,
        )
        return result

    def calculate_substitution_risk(
        self,
        supplier_id: str,
        declared_commodity: str,
    ) -> Decimal:
        """Calculate a substitution risk score (0-100) for a supplier.

        Combines historical switching frequency, volume anomaly patterns,
        cross-commodity risk factors, and recency of last switch to produce
        a composite risk score.

        Args:
            supplier_id: Unique supplier identifier.
            declared_commodity: Currently declared commodity type.

        Returns:
            Decimal risk score between 0 and 100.

        Raises:
            ValueError: If declared_commodity is not a valid EUDR commodity.
        """
        if declared_commodity not in EUDR_COMMODITIES:
            raise ValueError(
                f"'{declared_commodity}' is not a valid EUDR commodity. "
                f"Valid: {sorted(EUDR_COMMODITIES)}"
            )

        with self._lock:
            history = list(self._supplier_history.get(supplier_id, []))

        if not history:
            logger.debug(
                "No history for supplier=%s, returning base risk",
                supplier_id,
            )
            return Decimal("10")

        # Component 1: Transition frequency (weight 0.35)
        sorted_history = sorted(history, key=lambda r: r.get("date", ""))
        transitions = 0
        for i in range(1, len(sorted_history)):
            if sorted_history[i].get("commodity") != sorted_history[i - 1].get("commodity"):
                transitions += 1

        freq_score = Decimal("0")
        if len(sorted_history) > 1:
            freq_rate = Decimal(str(transitions)) / Decimal(str(len(sorted_history) - 1))
            freq_score = min(MAX_RISK_SCORE, freq_rate * Decimal("100"))

        # Component 2: Volume anomaly score (weight 0.25)
        vol_score = self._compute_volume_anomaly_score(sorted_history)

        # Component 3: Cross-commodity pair risk (weight 0.25)
        pair_score = Decimal("0")
        if sorted_history:
            last_commodity = sorted_history[-1].get("commodity", "")
            if last_commodity != declared_commodity:
                pair_key = (last_commodity, declared_commodity)
                pair_data = SUBSTITUTION_RISK_MATRIX.get(pair_key)
                if pair_data is not None:
                    pair_score = pair_data["risk_weight"]
                else:
                    pair_score = Decimal("40")

        # Component 4: Recency of last switch (weight 0.15)
        recency_score = self._compute_recency_score(sorted_history)

        # Weighted composite
        composite = (
            freq_score * Decimal("0.35")
            + vol_score * Decimal("0.25")
            + pair_score * Decimal("0.25")
            + recency_score * Decimal("0.15")
        )
        composite = min(MAX_RISK_SCORE, max(MIN_RISK_SCORE, composite))
        final = composite.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        logger.debug(
            "Substitution risk for supplier=%s commodity=%s: freq=%.1f "
            "vol=%.1f pair=%.1f recency=%.1f -> composite=%.2f",
            supplier_id, declared_commodity,
            float(freq_score), float(vol_score),
            float(pair_score), float(recency_score), float(final),
        )
        return final

    def detect_greenwashing(
        self,
        declared_commodity: str,
        actual_characteristics: Dict[str, Any],
        certifications: List[str],
    ) -> Dict[str, Any]:
        """Detect commodity misclassification for regulatory advantage.

        Cross-references claimed sustainability certifications against
        recognized certification bodies and verifies that certifications
        are applicable to the declared commodity.

        Args:
            declared_commodity: Commodity as declared by the supplier.
            actual_characteristics: Dictionary of commodity characteristics
                (e.g., ``{"species": "Elaeis guineensis", "origin": "ID"}``).
            certifications: List of certification identifiers claimed
                (e.g., ``["RSPO", "ISCC"]``).

        Returns:
            GreenwashingResult serialized as a dictionary.

        Raises:
            ValueError: If declared_commodity is not a valid EUDR commodity.
        """
        start_time = time.monotonic()

        if declared_commodity not in EUDR_COMMODITIES:
            raise ValueError(
                f"'{declared_commodity}' is not a valid EUDR commodity"
            )

        verified: List[str] = []
        failed: List[str] = []
        risk_factors: List[str] = []
        recommendations: List[str] = []

        # Verify each certification
        for cert in certifications:
            cert_upper = cert.upper().replace(" ", "_")
            cert_info = RECOGNIZED_CERTIFICATIONS.get(cert_upper)

            if cert_info is None:
                failed.append(cert)
                risk_factors.append(
                    f"Unrecognized certification: '{cert}'"
                )
                recommendations.append(
                    f"Verify certification '{cert}' against recognized bodies"
                )
                continue

            # Check commodity applicability
            applicable_commodities = cert_info.get("commodities", [])
            if declared_commodity not in applicable_commodities:
                failed.append(cert)
                risk_factors.append(
                    f"Certification '{cert}' ({cert_info['full_name']}) is not "
                    f"applicable to {declared_commodity}. Applicable: "
                    f"{applicable_commodities}"
                )
                recommendations.append(
                    f"Request correct certification for {declared_commodity}"
                )
            else:
                verified.append(cert)

        # Check for missing expected certifications
        expected_certs = self._get_expected_certifications(declared_commodity)
        if not certifications and expected_certs:
            risk_factors.append(
                f"No certifications provided for {declared_commodity}. "
                f"Expected at least one of: {expected_certs}"
            )
            recommendations.append(
                f"Obtain recognized certification: {', '.join(expected_certs)}"
            )

        # Check characteristics consistency
        origin = actual_characteristics.get("origin", "")
        species = actual_characteristics.get("species", "")
        characteristic_issues = self._check_characteristic_consistency(
            declared_commodity, origin, species,
        )
        risk_factors.extend(characteristic_issues)

        # Determine greenwashing detected
        greenwashing_detected = len(failed) > 0 or len(risk_factors) >= 2
        confidence = self._compute_greenwashing_confidence(
            verified, failed, risk_factors, certifications,
        )

        result = GreenwashingResult(
            result_id=_generate_id("gw"),
            declared_commodity=declared_commodity,
            declared_sustainability=(
                actual_characteristics.get("sustainability_claim", "")
            ),
            certifications_claimed=list(certifications),
            certifications_verified=verified,
            certifications_failed=failed,
            greenwashing_detected=greenwashing_detected,
            confidence=confidence,
            risk_factors=risk_factors,
            recommendations=recommendations,
        )
        result.provenance_hash = _compute_hash(result)

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        output = result.to_dict()
        output["processing_time_ms"] = round(processing_time_ms, 3)

        logger.info(
            "Greenwashing detection for commodity=%s detected=%s "
            "confidence=%s time_ms=%.1f",
            declared_commodity, greenwashing_detected,
            confidence, processing_time_ms,
        )
        return output

    def verify_commodity_declaration(
        self,
        declaration: Dict[str, Any],
        supporting_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify that a declared commodity matches supporting evidence.

        Cross-references the commodity declaration against trade records,
        satellite observations, laboratory analyses, and certification
        documents to confirm or flag inconsistencies.

        Args:
            declaration: Commodity declaration with keys:
                ``commodity`` (str), ``volume`` (numeric),
                ``origin_country`` (str), ``supplier_id`` (str).
            supporting_evidence: List of evidence items, each with keys:
                ``type`` (str), ``source`` (str), ``data`` (dict).

        Returns:
            Dictionary with verification result, consistency checks, and
            provenance hash.

        Raises:
            ValueError: If declaration is missing required fields.
        """
        start_time = time.monotonic()

        # Validate declaration
        required_fields = {"commodity", "volume", "origin_country", "supplier_id"}
        missing = required_fields - set(declaration.keys())
        if missing:
            raise ValueError(f"Declaration missing required fields: {missing}")

        declared_commodity = declaration["commodity"]
        if declared_commodity not in EUDR_COMMODITIES:
            raise ValueError(
                f"Declaration commodity '{declared_commodity}' is not valid"
            )

        consistency_checks: List[Dict[str, Any]] = []
        overall_consistent = True
        confidence = Decimal("1.0")

        for evidence_item in supporting_evidence:
            evidence_type = evidence_item.get("type", "unknown")
            evidence_data = evidence_item.get("data", {})

            check_result = self._verify_evidence_item(
                declared_commodity, declaration, evidence_type, evidence_data,
            )
            consistency_checks.append(check_result)

            if not check_result.get("consistent", True):
                overall_consistent = False
                confidence = max(
                    Decimal("0.0"),
                    confidence - Decimal("0.2"),
                )

        # Generate alert for inconsistencies
        alert_dict: Optional[Dict[str, Any]] = None
        if not overall_consistent:
            risk_score = Decimal("100") - (confidence * Decimal("100"))
            severity = self._classify_severity(risk_score)
            alert = self._create_alert(
                supplier_id=declaration.get("supplier_id", ""),
                severity=severity,
                declared_commodity=declared_commodity,
                suspected_commodity="",
                risk_score=risk_score,
                description=(
                    f"Declaration inconsistency for {declared_commodity} from "
                    f"supplier {declaration.get('supplier_id', '')}"
                ),
                evidence_summary=[
                    c.get("reason", "") for c in consistency_checks
                    if not c.get("consistent", True)
                ],
                detection_method="evidence_verification",
            )
            alert_dict = alert.to_dict()

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "declaration_verified": overall_consistent,
            "consistency_checks": consistency_checks,
            "confidence": str(confidence.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP,
            )),
            "evidence_count": len(supporting_evidence),
            "alert": alert_dict,
            "processing_time_ms": round(processing_time_ms, 3),
            "provenance_hash": "",
        }
        result["provenance_hash"] = self._compute_provenance_hash(result)

        logger.info(
            "Declaration verification completed: consistent=%s confidence=%s "
            "evidence_items=%d time_ms=%.1f",
            overall_consistent, result["confidence"],
            len(supporting_evidence), processing_time_ms,
        )
        return result

    def get_substitution_alerts(
        self,
        severity_threshold: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get active substitution alerts, optionally filtered by severity.

        Args:
            severity_threshold: Minimum severity to include. If None, all
                alerts are returned. Valid values: INFO, WARNING, HIGH, CRITICAL.

        Returns:
            List of alert dictionaries sorted by risk_score descending.

        Raises:
            ValueError: If severity_threshold is not a valid severity level.
        """
        if severity_threshold is not None:
            severity_threshold = severity_threshold.upper()
            if severity_threshold not in ALERT_SEVERITIES:
                raise ValueError(
                    f"Invalid severity_threshold '{severity_threshold}'. "
                    f"Valid: {ALERT_SEVERITIES}"
                )

        with self._lock:
            alerts = list(self._alerts.values())

        # Filter by severity threshold
        if severity_threshold is not None:
            min_index = ALERT_SEVERITIES.index(severity_threshold)
            alerts = [
                a for a in alerts
                if ALERT_SEVERITIES.index(a.severity) >= min_index
            ]

        # Sort by risk_score descending
        alerts.sort(key=lambda a: a.risk_score, reverse=True)

        result = [a.to_dict() for a in alerts]

        logger.debug(
            "Retrieved %d substitution alerts (threshold=%s)",
            len(result), severity_threshold,
        )
        return result

    def calculate_cross_commodity_risk(
        self,
        from_commodity: str,
        to_commodity: str,
    ) -> Dict[str, Any]:
        """Calculate risk impact when switching between specific commodity pairs.

        Uses the static cross-commodity substitution risk matrix to return
        the risk weight, regulatory concern level, and reason for the
        specified commodity pair transition.

        Args:
            from_commodity: Source commodity being switched from.
            to_commodity: Target commodity being switched to.

        Returns:
            Dictionary with risk_weight, reason, regulatory_concern,
            reverse_risk (the opposite direction's risk), and provenance_hash.

        Raises:
            ValueError: If either commodity is not a valid EUDR commodity
                or if from_commodity equals to_commodity.
        """
        if from_commodity not in EUDR_COMMODITIES:
            raise ValueError(
                f"'{from_commodity}' is not a valid EUDR commodity"
            )
        if to_commodity not in EUDR_COMMODITIES:
            raise ValueError(
                f"'{to_commodity}' is not a valid EUDR commodity"
            )
        if from_commodity == to_commodity:
            raise ValueError(
                "from_commodity and to_commodity must be different"
            )

        pair_key = (from_commodity, to_commodity)
        pair_data = SUBSTITUTION_RISK_MATRIX.get(pair_key)

        if pair_data is not None:
            risk_weight = pair_data["risk_weight"]
            reason = pair_data["reason"]
            regulatory_concern = pair_data["regulatory_concern"]
        else:
            risk_weight = Decimal("40")
            reason = (
                f"No specific risk data for {from_commodity} -> "
                f"{to_commodity}; default moderate risk applied"
            )
            regulatory_concern = "MEDIUM"

        # Reverse direction
        reverse_key = (to_commodity, from_commodity)
        reverse_data = SUBSTITUTION_RISK_MATRIX.get(reverse_key)
        reverse_risk = (
            str(reverse_data["risk_weight"]) if reverse_data
            else str(Decimal("40"))
        )

        result = {
            "from_commodity": from_commodity,
            "to_commodity": to_commodity,
            "risk_weight": str(risk_weight),
            "reason": reason,
            "regulatory_concern": regulatory_concern,
            "reverse_risk_weight": reverse_risk,
            "asymmetric": str(risk_weight) != reverse_risk,
            "provenance_hash": "",
        }
        result["provenance_hash"] = self._compute_provenance_hash(result)

        logger.debug(
            "Cross-commodity risk: %s -> %s = %s (%s)",
            from_commodity, to_commodity, risk_weight, regulatory_concern,
        )
        return result

    def analyze_seasonal_switching(
        self,
        supplier_id: str,
    ) -> Dict[str, Any]:
        """Detect seasonal commodity switching patterns for a supplier.

        Analyzes commodity declarations across calendar quarters to identify
        whether the supplier systematically switches commodities based on
        seasonal patterns (e.g., declaring soya in Q2-Q3 and cattle in Q4-Q1).

        Args:
            supplier_id: Unique supplier identifier.

        Returns:
            Dictionary with seasonal_pattern_detected, quarter_analysis,
            risk_adjustment, and provenance_hash.

        Raises:
            ValueError: If supplier_id is empty.
        """
        start_time = time.monotonic()

        if not supplier_id:
            raise ValueError("supplier_id must be a non-empty string")

        with self._lock:
            history = list(self._supplier_history.get(supplier_id, []))

        # Group declarations by quarter
        quarter_commodities: Dict[int, Dict[str, int]] = {
            1: {}, 2: {}, 3: {}, 4: {},
        }
        for record in history:
            date_str = record.get("date", "")
            commodity = record.get("commodity", "unknown")
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                quarter = (dt.month - 1) // 3 + 1
            except (ValueError, AttributeError):
                continue

            quarter_commodities[quarter][commodity] = (
                quarter_commodities[quarter].get(commodity, 0) + 1
            )

        # Analyze per-quarter dominant commodity
        quarter_analysis: Dict[str, Dict[str, Any]] = {}
        dominant_per_quarter: Dict[int, str] = {}
        for q in range(1, 5):
            commodities = quarter_commodities[q]
            if commodities:
                dominant = max(commodities, key=lambda k: commodities[k])
                dominant_per_quarter[q] = dominant
                quarter_analysis[f"Q{q}"] = {
                    "dominant_commodity": dominant,
                    "commodity_distribution": commodities,
                    "declaration_count": sum(commodities.values()),
                }
            else:
                quarter_analysis[f"Q{q}"] = {
                    "dominant_commodity": None,
                    "commodity_distribution": {},
                    "declaration_count": 0,
                }

        # Detect seasonal switching
        unique_dominants = set(dominant_per_quarter.values())
        seasonal_pattern_detected = len(unique_dominants) > 1

        # Compute risk adjustment
        if seasonal_pattern_detected:
            switch_count = len(unique_dominants) - 1
            risk_adjustment = Decimal(str(switch_count * 15))
        else:
            risk_adjustment = Decimal("0")

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "supplier_id": supplier_id,
            "seasonal_pattern_detected": seasonal_pattern_detected,
            "quarter_analysis": quarter_analysis,
            "unique_commodities_across_quarters": len(unique_dominants),
            "dominant_per_quarter": {
                f"Q{q}": c for q, c in dominant_per_quarter.items()
            },
            "risk_adjustment": str(risk_adjustment),
            "total_records_analyzed": len(history),
            "processing_time_ms": round(processing_time_ms, 3),
            "provenance_hash": "",
        }
        result["provenance_hash"] = self._compute_provenance_hash(result)

        logger.info(
            "Seasonal switching analysis for supplier=%s detected=%s "
            "unique_commodities=%d time_ms=%.1f",
            supplier_id, seasonal_pattern_detected,
            len(unique_dominants), processing_time_ms,
        )
        return result

    def batch_screen(
        self,
        suppliers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Batch screening of multiple suppliers for substitution risk.

        Processes each supplier record and returns a risk assessment for
        each. Each supplier dict must contain ``supplier_id`` (str) and
        ``declared_commodity`` (str).

        Args:
            suppliers: List of supplier dicts with ``supplier_id`` and
                ``declared_commodity``.

        Returns:
            List of result dicts, one per supplier, each containing
            supplier_id, risk_score, severity, and provenance_hash.

        Raises:
            ValueError: If suppliers is not a list or exceeds 10000 items.
        """
        start_time = time.monotonic()

        if not isinstance(suppliers, list):
            raise ValueError("suppliers must be a list")
        if len(suppliers) > 10_000:
            raise ValueError(
                f"Batch size {len(suppliers)} exceeds maximum 10000"
            )

        results: List[Dict[str, Any]] = []
        for supplier in suppliers:
            supplier_id = supplier.get("supplier_id", "")
            declared_commodity = supplier.get("declared_commodity", "")

            if not supplier_id or not declared_commodity:
                results.append({
                    "supplier_id": supplier_id,
                    "risk_score": str(Decimal("0")),
                    "severity": "INFO",
                    "error": "Missing supplier_id or declared_commodity",
                    "provenance_hash": "",
                })
                continue

            try:
                risk_score = self.calculate_substitution_risk(
                    supplier_id, declared_commodity,
                )
                severity = self._classify_severity(risk_score)
                entry = {
                    "supplier_id": supplier_id,
                    "declared_commodity": declared_commodity,
                    "risk_score": str(risk_score),
                    "severity": severity,
                    "provenance_hash": "",
                }
                entry["provenance_hash"] = self._compute_provenance_hash(entry)
                results.append(entry)
            except ValueError as exc:
                results.append({
                    "supplier_id": supplier_id,
                    "risk_score": str(Decimal("0")),
                    "severity": "INFO",
                    "error": str(exc),
                    "provenance_hash": "",
                })

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        logger.info(
            "Batch screening completed: suppliers=%d time_ms=%.1f",
            len(suppliers), processing_time_ms,
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_provenance_hash(self, data: Any) -> str:
        """Compute SHA-256 provenance hash for audit trail.

        Args:
            data: Data to hash.

        Returns:
            SHA-256 hex digest (64 characters).
        """
        return _compute_hash(data)

    def _detect_volume_anomaly(
        self,
        history: List[Dict[str, Any]],
        current: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Detect volume anomalies using z-score analysis.

        Args:
            history: Sorted historical records with 'volume' field.
            current: Current declaration with 'volume' field.

        Returns:
            Dict with anomaly_detected (bool), z_score (str), mean (str),
            std_dev (str).
        """
        volumes = []
        for record in history:
            vol = record.get("volume")
            if vol is not None:
                try:
                    volumes.append(_to_decimal(vol))
                except ValueError:
                    continue

        current_vol_raw = current.get("volume")
        if current_vol_raw is None or len(volumes) < MIN_HISTORY_RECORDS:
            return {
                "anomaly_detected": False,
                "z_score": "0.0",
                "mean": "0.0",
                "std_dev": "0.0",
            }

        current_vol = _to_decimal(current_vol_raw)
        n = len(volumes)
        mean = sum(volumes) / n
        variance = sum((v - mean) ** 2 for v in volumes) / n

        if variance == 0:
            return {
                "anomaly_detected": False,
                "z_score": "0.0",
                "mean": str(mean.quantize(Decimal("0.01"))),
                "std_dev": "0.0",
            }

        std_dev = variance.sqrt()
        z_score = abs(current_vol - mean) / std_dev

        anomaly_detected = z_score >= VOLUME_ANOMALY_Z_THRESHOLD

        return {
            "anomaly_detected": anomaly_detected,
            "z_score": str(z_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
            "mean": str(mean.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
            "std_dev": str(std_dev.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
        }

    def _compute_volume_anomaly_score(
        self,
        history: List[Dict[str, Any]],
    ) -> Decimal:
        """Compute an aggregate volume anomaly score from history.

        Args:
            history: Sorted history records.

        Returns:
            Decimal score 0-100.
        """
        if len(history) < MIN_HISTORY_RECORDS:
            return Decimal("10")

        volumes = []
        for record in history:
            vol = record.get("volume")
            if vol is not None:
                try:
                    volumes.append(_to_decimal(vol))
                except ValueError:
                    continue

        if len(volumes) < MIN_HISTORY_RECORDS:
            return Decimal("10")

        n = len(volumes)
        mean = sum(volumes) / n
        if mean == 0:
            return Decimal("0")

        # Coefficient of variation as anomaly indicator
        variance = sum((v - mean) ** 2 for v in volumes) / n
        if variance == 0:
            return Decimal("0")

        std_dev = variance.sqrt()
        cv = std_dev / mean
        score = min(MAX_RISK_SCORE, cv * Decimal("100"))
        return score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _compute_recency_score(
        self,
        history: List[Dict[str, Any]],
    ) -> Decimal:
        """Compute recency score based on how recently a switch occurred.

        Args:
            history: Sorted history records.

        Returns:
            Decimal score 0-100. Higher = more recent switch.
        """
        if len(history) < 2:
            return Decimal("0")

        # Find last switch
        last_switch_idx = -1
        for i in range(len(history) - 1, 0, -1):
            if history[i].get("commodity") != history[i - 1].get("commodity"):
                last_switch_idx = i
                break

        if last_switch_idx < 0:
            return Decimal("0")

        last_switch_date_str = history[last_switch_idx].get("date", "")
        try:
            last_switch_dt = datetime.fromisoformat(
                last_switch_date_str.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            return Decimal("30")

        now = utcnow()
        if last_switch_dt.tzinfo is None:
            last_switch_dt = last_switch_dt.replace(tzinfo=timezone.utc)

        days_since = max(0, (now - last_switch_dt).days)

        # Decay: 100 for today, ~50 at 90 days, ~10 at 365 days
        if days_since == 0:
            return MAX_RISK_SCORE
        decay_factor = Decimal(str(math.exp(-days_since / 130.0)))
        score = (MAX_RISK_SCORE * decay_factor).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        return max(MIN_RISK_SCORE, min(MAX_RISK_SCORE, score))

    def _classify_severity(self, risk_score: Decimal) -> str:
        """Classify alert severity based on risk score.

        Args:
            risk_score: Risk score (Decimal 0-100).

        Returns:
            Severity string: INFO, WARNING, HIGH, or CRITICAL.
        """
        if risk_score >= SEVERITY_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        if risk_score >= SEVERITY_THRESHOLDS["HIGH"]:
            return "HIGH"
        if risk_score >= SEVERITY_THRESHOLDS["WARNING"]:
            return "WARNING"
        return "INFO"

    def _create_alert(
        self,
        supplier_id: str,
        severity: str,
        declared_commodity: str,
        suspected_commodity: str,
        risk_score: Decimal,
        description: str,
        evidence_summary: List[str],
        detection_method: str,
    ) -> SubstitutionAlert:
        """Create and store a substitution alert.

        Args:
            supplier_id: Supplier identifier.
            severity: Alert severity level.
            declared_commodity: Declared commodity.
            suspected_commodity: Suspected actual commodity.
            risk_score: Computed risk score.
            description: Alert description.
            evidence_summary: List of evidence strings.
            detection_method: Detection method that triggered the alert.

        Returns:
            Created SubstitutionAlert instance.
        """
        alert = SubstitutionAlert(
            alert_id=_generate_id("alert"),
            supplier_id=supplier_id,
            severity=severity,
            declared_commodity=declared_commodity,
            suspected_commodity=suspected_commodity,
            risk_score=risk_score,
            description=description,
            evidence_summary=evidence_summary,
            detection_method=detection_method,
            created_at=utcnow().isoformat(),
        )
        alert.provenance_hash = _compute_hash(alert)

        with self._lock:
            self._alerts[alert.alert_id] = alert

        logger.info(
            "Substitution alert created: id=%s severity=%s supplier=%s "
            "risk=%.1f",
            alert.alert_id, severity, supplier_id, float(risk_score),
        )
        return alert

    def _get_expected_certifications(
        self,
        commodity: str,
    ) -> List[str]:
        """Get expected certifications for a commodity.

        Args:
            commodity: EUDR commodity type.

        Returns:
            List of certification names applicable to the commodity.
        """
        expected: List[str] = []
        for cert_name, cert_info in RECOGNIZED_CERTIFICATIONS.items():
            if commodity in cert_info.get("commodities", []):
                expected.append(cert_name)
        return expected

    def _check_characteristic_consistency(
        self,
        declared_commodity: str,
        origin: str,
        species: str,
    ) -> List[str]:
        """Check if commodity characteristics are consistent with declaration.

        Args:
            declared_commodity: Declared commodity type.
            origin: Origin country code.
            species: Species identification.

        Returns:
            List of inconsistency descriptions (empty if consistent).
        """
        issues: List[str] = []

        # Known species-commodity mappings
        species_map: Dict[str, List[str]] = {
            "oil_palm": [
                "elaeis guineensis", "elaeis oleifera", "elaeis",
            ],
            "cocoa": [
                "theobroma cacao", "theobroma",
            ],
            "coffee": [
                "coffea arabica", "coffea canephora", "coffea robusta",
                "coffea liberica", "coffea",
            ],
            "rubber": [
                "hevea brasiliensis", "hevea",
            ],
            "soya": [
                "glycine max", "glycine",
            ],
        }

        if species and declared_commodity in species_map:
            expected_species = species_map[declared_commodity]
            species_lower = species.lower().strip()
            if not any(s in species_lower for s in expected_species):
                issues.append(
                    f"Species '{species}' does not match expected species "
                    f"for {declared_commodity}: {expected_species}"
                )

        # Known high-risk origin-commodity combinations
        high_risk_origins: Dict[str, List[str]] = {
            "oil_palm": ["ID", "MY"],
            "soya": ["BR", "AR", "PY"],
            "cattle": ["BR", "AR", "CO"],
            "cocoa": ["CI", "GH", "CM"],
            "coffee": ["BR", "VN", "CO", "ET"],
            "wood": ["BR", "ID", "MY", "CG", "CD"],
            "rubber": ["TH", "ID", "MY", "VN"],
        }

        if origin and declared_commodity in high_risk_origins:
            if origin in high_risk_origins[declared_commodity]:
                issues.append(
                    f"Origin {origin} is a high-risk country for "
                    f"{declared_commodity} under EUDR benchmarking"
                )

        return issues

    def _compute_greenwashing_confidence(
        self,
        verified: List[str],
        failed: List[str],
        risk_factors: List[str],
        certifications: List[str],
    ) -> Decimal:
        """Compute confidence score for greenwashing detection.

        Args:
            verified: Verified certifications.
            failed: Failed certifications.
            risk_factors: Identified risk factors.
            certifications: Original certification claims.

        Returns:
            Decimal confidence score 0.0-1.0.
        """
        if not certifications and not risk_factors:
            return Decimal("0.3")

        total_certs = len(certifications) if certifications else 1
        failed_ratio = Decimal(str(len(failed))) / Decimal(str(total_certs))
        risk_factor_weight = min(
            Decimal("0.4"),
            Decimal(str(len(risk_factors))) * Decimal("0.1"),
        )

        confidence = min(
            Decimal("1.0"),
            failed_ratio * Decimal("0.6") + risk_factor_weight,
        )
        return confidence.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _verify_evidence_item(
        self,
        declared_commodity: str,
        declaration: Dict[str, Any],
        evidence_type: str,
        evidence_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify a single evidence item against the declaration.

        Args:
            declared_commodity: Declared commodity.
            declaration: Full declaration dict.
            evidence_type: Type of evidence (trade_record, satellite,
                lab_analysis, certification).
            evidence_data: Evidence payload.

        Returns:
            Dict with consistent (bool), evidence_type, reason.
        """
        consistent = True
        reason = ""

        if evidence_type == "trade_record":
            trade_commodity = evidence_data.get("commodity", "")
            if trade_commodity and trade_commodity != declared_commodity:
                consistent = False
                reason = (
                    f"Trade record shows commodity '{trade_commodity}', "
                    f"declared '{declared_commodity}'"
                )

            trade_volume = evidence_data.get("volume")
            declared_volume = declaration.get("volume")
            if trade_volume is not None and declared_volume is not None:
                try:
                    tv = _to_decimal(trade_volume)
                    dv = _to_decimal(declared_volume)
                    if dv > 0:
                        deviation = abs(tv - dv) / dv
                        if deviation > Decimal("0.20"):
                            consistent = False
                            reason += (
                                f" Volume deviation: trade={tv}, "
                                f"declared={dv} (deviation={deviation:.1%})"
                            )
                except ValueError:
                    pass

        elif evidence_type == "satellite":
            satellite_land_use = evidence_data.get("land_use", "")
            commodity_land_use_map: Dict[str, List[str]] = {
                "cattle": ["pasture", "grassland", "rangeland"],
                "cocoa": ["agroforestry", "plantation", "cocoa_farm"],
                "coffee": ["agroforestry", "plantation", "coffee_farm"],
                "oil_palm": ["plantation", "palm_plantation"],
                "rubber": ["plantation", "rubber_plantation"],
                "soya": ["cropland", "agriculture", "soya_field"],
                "wood": ["forest", "managed_forest", "plantation"],
            }
            expected = commodity_land_use_map.get(declared_commodity, [])
            if satellite_land_use and expected:
                if satellite_land_use.lower() not in expected:
                    consistent = False
                    reason = (
                        f"Satellite land use '{satellite_land_use}' "
                        f"inconsistent with {declared_commodity} "
                        f"(expected: {expected})"
                    )

        elif evidence_type == "lab_analysis":
            lab_commodity = evidence_data.get("identified_commodity", "")
            if lab_commodity and lab_commodity != declared_commodity:
                consistent = False
                reason = (
                    f"Lab analysis identified '{lab_commodity}', "
                    f"declared '{declared_commodity}'"
                )

        elif evidence_type == "certification":
            cert_commodity = evidence_data.get("commodity_scope", "")
            if cert_commodity and cert_commodity != declared_commodity:
                consistent = False
                reason = (
                    f"Certification scope '{cert_commodity}' does not match "
                    f"declared '{declared_commodity}'"
                )

        if not reason:
            reason = f"{evidence_type}: consistent with declaration"

        return {
            "evidence_type": evidence_type,
            "consistent": consistent,
            "reason": reason,
        }
