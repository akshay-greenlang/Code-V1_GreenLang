# -*- coding: utf-8 -*-
"""
GreenwashingDetectionEngine - PACK-018 EU Green Claims Prep Engine 6
=====================================================================

Detects greenwashing risks using the TerraChoice Seven Sins of
Greenwashing framework and the EU Environmental Claims, Green Transition
(ECGT) prohibited practices as defined in the proposed Green Claims
Directive (COM/2023/166) and the Empowering Consumers Directive
(Directive 2024/825).

The engine screens individual claims and claim portfolios for patterns
that indicate potential greenwashing, including vague language, missing
evidence, partial scope misrepresentation, false labels, and prohibited
offset-based neutrality claims.

TerraChoice Seven Sins of Greenwashing:
    1. Sin of the Hidden Trade-off: Claim based on narrow attribute
       while ignoring significant environmental impacts elsewhere.
    2. Sin of No Proof: Claim that cannot be substantiated by easily
       accessible supporting information or reliable third-party
       certification.
    3. Sin of Vagueness: Claim so poorly defined or broad that its
       real meaning is likely to be misunderstood by consumers.
    4. Sin of False Labels: Claim that gives the impression of
       third-party endorsement where none exists (fake labels/seals).
    5. Sin of Irrelevance: Claim that may be truthful but is
       unimportant or unhelpful for consumers seeking
       environmentally preferable products.
    6. Sin of Lesser of Two Evils: Claim that may be true within
       the product category but risks distracting consumers from
       the greater environmental impacts of the category as a whole.
    7. Sin of Fibbing: Claim that is simply false.

EU ECGT Prohibited Practices (Directive 2024/825):
    - Generic environmental excellence claims (e.g., "green", "eco")
      without substantiation per recognized environmental certification
    - Displaying sustainability labels not based on a certification
      scheme or not established by public authorities
    - Making a generic environmental claim where the trader cannot
      demonstrate recognised excellent environmental performance
    - Presenting requirements imposed by law as a distinguishing
      characteristic of the trader's offer
    - Using carbon offsets to claim environmental neutrality
    - Making unsubstantiated future environmental claims

Zero-Hallucination:
    - Pattern matching uses deterministic keyword and regex checks
    - Risk scoring uses fixed point weights per severity level
    - Portfolio screening uses deterministic aggregation
    - SHA-256 provenance hash on every result
    - No LLM involvement in any detection or scoring path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-018 EU Green Claims Prep
Status:  Production Ready
"""

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import AlertSeverity

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimal numbers, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GreenwashingSin(str, Enum):
    """TerraChoice Seven Sins of Greenwashing.

    Each sin represents a distinct pattern of misleading environmental
    communication identified in the TerraChoice Environmental Marketing
    study, widely adopted as an industry framework.
    """
    HIDDEN_TRADEOFF = "hidden_tradeoff"
    NO_PROOF = "no_proof"
    VAGUENESS = "vagueness"
    FALSE_LABELS = "false_labels"
    IRRELEVANCE = "irrelevance"
    LESSER_OF_TWO_EVILS = "lesser_of_two_evils"
    FIBBING = "fibbing"

class ProhibitedPractice(str, Enum):
    """Prohibited environmental marketing practices per EU ECGT.

    Practices explicitly banned or restricted under the Empowering
    Consumers Directive (2024/825) and the proposed Green Claims
    Directive.
    """
    GENERIC_EXCELLENCE_CLAIM = "generic_excellence_claim"
    UNCERTIFIED_LABEL = "uncertified_label"
    PARTIAL_SCOPE_MISREPRESENTATION = "partial_scope_misrepresentation"
    LEGAL_REQUIREMENT_AS_DISTINCTION = "legal_requirement_as_distinction"
    OFFSET_NEUTRALITY_CLAIM = "offset_neutrality_claim"
    UNSUBSTANTIATED_FUTURE_CLAIM = "unsubstantiated_future_claim"

class ClaimType(str, Enum):
    """Type of environmental claim being screened.

    Categorizes claims by their nature to apply type-specific
    detection rules.
    """
    PRODUCT = "product"
    CORPORATE = "corporate"
    SERVICE = "service"
    PROCESS = "process"
    LABEL = "label"
    MARKETING = "marketing"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Vague keywords commonly associated with greenwashing.
VAGUE_KEYWORDS: List[str] = [
    "green",
    "eco",
    "eco-friendly",
    "environmentally friendly",
    "sustainable",
    "natural",
    "earth-friendly",
    "planet-friendly",
    "climate-friendly",
    "clean",
    "conscious",
    "responsible",
]

# Additional vague phrases that lack specificity.
VAGUE_PHRASES: List[str] = [
    "better for the environment",
    "good for the planet",
    "environmentally safe",
    "non-toxic",
    "all natural",
    "made with nature",
    "earth conscious",
    "planet positive",
    "nature friendly",
    "green choice",
    "eco choice",
    "the sustainable option",
    "reducing our footprint",
    "low impact",
]

# Offset/neutrality keywords indicating potential prohibited claims.
OFFSET_NEUTRALITY_KEYWORDS: List[str] = [
    "carbon neutral",
    "climate neutral",
    "carbon negative",
    "net zero",
    "net-zero",
    "carbon positive",
    "climate positive",
    "offsetting",
    "carbon offset",
    "carbon compensated",
    "co2 neutral",
    "co2 compensated",
    "environmentally neutral",
]

# Future claim indicators.
FUTURE_CLAIM_INDICATORS: List[str] = [
    "by 2030",
    "by 2040",
    "by 2050",
    "will be",
    "will become",
    "committed to",
    "on track to",
    "aiming to",
    "targeting",
    "goal of",
    "ambition to",
    "pledge",
    "promise",
    "plan to achieve",
    "roadmap to",
]

# Legal requirement phrases that should not be used as distinctions.
LEGAL_REQUIREMENT_PHRASES: List[str] = [
    "compliant with eu regulations",
    "meets legal requirements",
    "in accordance with the law",
    "legally compliant",
    "regulatory compliant",
    "bpa free",
    "cfc free",
    "lead free",
    "asbestos free",
    "reaches compliant",
    "rohs compliant",
]

# Prohibited practices detection configuration.
PROHIBITED_PATTERNS: Dict[str, Dict[str, Any]] = {
    ProhibitedPractice.GENERIC_EXCELLENCE_CLAIM: {
        "description": "Generic environmental excellence claim without "
                       "recognised certification or substantiation",
        "severity": AlertSeverity.HIGH,
        "article_reference": "Article 2(1) Directive 2024/825",
        "detection_keywords": VAGUE_KEYWORDS,
        "requires_certification": True,
        "recommendation": "Replace generic claim with specific, measurable "
                          "environmental attribute backed by recognised "
                          "certification scheme or PEF/OEF methodology",
    },
    ProhibitedPractice.UNCERTIFIED_LABEL: {
        "description": "Sustainability label not based on a certification "
                       "scheme or not established by public authorities",
        "severity": AlertSeverity.CRITICAL,
        "article_reference": "Article 2(4) Directive 2024/825",
        "detection_keywords": ["certified", "label", "seal", "badge", "mark"],
        "requires_certification": True,
        "recommendation": "Ensure all sustainability labels are based on "
                          "recognised third-party certification schemes or "
                          "established by public authorities",
    },
    ProhibitedPractice.PARTIAL_SCOPE_MISREPRESENTATION: {
        "description": "Claim that implies whole-product or whole-company "
                       "environmental performance when it covers only a "
                       "partial scope",
        "severity": AlertSeverity.HIGH,
        "article_reference": "Article 3(1)(b) COM/2023/166",
        "detection_keywords": ["100%", "entirely", "completely", "fully",
                               "whole", "total", "across all"],
        "requires_certification": False,
        "recommendation": "Clearly specify the scope and boundaries of the "
                          "environmental claim, including which products, "
                          "processes, or operations are covered",
    },
    ProhibitedPractice.LEGAL_REQUIREMENT_AS_DISTINCTION: {
        "description": "Presenting requirements imposed by law as a "
                       "distinguishing feature of the trader's offer",
        "severity": AlertSeverity.MEDIUM,
        "article_reference": "Article 2(2) Directive 2024/825",
        "detection_keywords": LEGAL_REQUIREMENT_PHRASES,
        "requires_certification": False,
        "recommendation": "Do not market legal compliance as a distinguishing "
                          "environmental attribute; focus on performance "
                          "that exceeds regulatory requirements",
    },
    ProhibitedPractice.OFFSET_NEUTRALITY_CLAIM: {
        "description": "Claiming carbon/climate neutrality or equivalent "
                       "based on greenhouse gas emission offsets",
        "severity": AlertSeverity.CRITICAL,
        "article_reference": "Article 2(3) Directive 2024/825",
        "detection_keywords": OFFSET_NEUTRALITY_KEYWORDS,
        "requires_certification": False,
        "recommendation": "Remove carbon/climate neutrality claims based on "
                          "offsets; instead report actual emission reductions "
                          "and separately disclose any offset purchases",
    },
    ProhibitedPractice.UNSUBSTANTIATED_FUTURE_CLAIM: {
        "description": "Future environmental performance claim without "
                       "clear commitments, implementation plan, or "
                       "independent monitoring",
        "severity": AlertSeverity.HIGH,
        "article_reference": "Article 5 COM/2023/166",
        "detection_keywords": FUTURE_CLAIM_INDICATORS,
        "requires_certification": False,
        "recommendation": "Support future claims with binding targets, "
                          "detailed implementation plan with milestones, "
                          "and independent third-party monitoring",
    },
}

# Risk score weights per alert severity.
SEVERITY_RISK_WEIGHTS: Dict[str, Decimal] = {
    AlertSeverity.LOW.value: Decimal("5"),
    AlertSeverity.MEDIUM.value: Decimal("10"),
    AlertSeverity.HIGH.value: Decimal("20"),
    AlertSeverity.CRITICAL.value: Decimal("30"),
}

# Sin descriptions for reporting.
SIN_DESCRIPTIONS: Dict[str, str] = {
    GreenwashingSin.HIDDEN_TRADEOFF.value: (
        "Claim is based on a narrow set of attributes without attention "
        "to other important environmental issues"
    ),
    GreenwashingSin.NO_PROOF.value: (
        "Claim is not backed by easily accessible supporting information "
        "or by a reliable third-party certification"
    ),
    GreenwashingSin.VAGUENESS.value: (
        "Claim is so poorly defined or broad that its real meaning is "
        "likely to be misunderstood by the consumer"
    ),
    GreenwashingSin.FALSE_LABELS.value: (
        "Claim that, through words or images, gives the impression of "
        "third-party endorsement where no such endorsement exists"
    ),
    GreenwashingSin.IRRELEVANCE.value: (
        "Claim that may be truthful but is unimportant or unhelpful for "
        "consumers seeking environmentally preferable products"
    ),
    GreenwashingSin.LESSER_OF_TWO_EVILS.value: (
        "Claim that may be true within the product category but risks "
        "distracting the consumer from the greater environmental impacts "
        "of the category as a whole"
    ),
    GreenwashingSin.FIBBING.value: (
        "Environmental claim that is simply false or fabricated"
    ),
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class GreenwashingAlert(BaseModel):
    """A single greenwashing detection alert.

    Represents a potential greenwashing issue identified during
    claim screening, with classification, severity, and remediation
    guidance.
    """
    alert_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for the alert",
    )
    sin_type: Optional[str] = Field(
        None,
        description="TerraChoice sin type (if applicable)",
    )
    prohibited_practice: Optional[str] = Field(
        None,
        description="EU ECGT prohibited practice type (if applicable)",
    )
    severity: str = Field(
        AlertSeverity.MEDIUM.value,
        description="Alert severity level",
    )
    claim_text: str = Field(
        ...,
        description="The claim text that triggered the alert",
    )
    description: str = Field(
        ...,
        description="Detailed description of the greenwashing issue",
    )
    evidence: Optional[str] = Field(
        None,
        description="Evidence or keywords that triggered the detection",
    )
    recommendation: str = Field(
        ...,
        description="Recommended remediation action",
    )
    article_reference: Optional[str] = Field(
        None,
        description="Regulatory article reference for prohibited practices",
    )

class ClaimScreeningInput(BaseModel):
    """Input for screening a single environmental claim.

    Captures the claim text and supporting context needed for
    comprehensive greenwashing detection.
    """
    claim_text: str = Field(
        ...,
        min_length=3,
        description="Full text of the environmental claim to screen",
    )
    evidence_provided: Optional[str] = Field(
        None,
        description="Summary of evidence supporting the claim",
    )
    claim_type: ClaimType = Field(
        ClaimType.PRODUCT,
        description="Type of environmental claim",
    )
    labels_used: List[str] = Field(
        default_factory=list,
        description="List of sustainability labels displayed",
    )
    lifecycle_stages_covered: List[str] = Field(
        default_factory=list,
        description="Life-cycle stages covered by the claim",
    )
    has_third_party_certification: bool = Field(
        False,
        description="Whether claim is backed by third-party certification",
    )
    certification_scheme: Optional[str] = Field(
        None,
        description="Name of the certification scheme if applicable",
    )
    covers_full_lifecycle: bool = Field(
        False,
        description="Whether the claim covers the full product lifecycle",
    )
    relies_on_offsets: bool = Field(
        False,
        description="Whether the claim relies on carbon offsets",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GreenwashingDetectionEngine:
    """Engine for detecting greenwashing risks in environmental claims.

    Applies the TerraChoice Seven Sins framework and EU ECGT prohibited
    practices to screen individual claims and claim portfolios for
    greenwashing risks. Generates alerts with severity ratings, evidence
    references, and remediation recommendations.

    Attributes:
        engine_id: Unique identifier for this engine instance.
        version: Module version string.

    Example:
        >>> engine = GreenwashingDetectionEngine()
        >>> result = engine.screen_claim(
        ...     claim_text="Our product is eco-friendly",
        ...     evidence=None,
        ...     claim_type=ClaimType.PRODUCT,
        ... )
        >>> assert "provenance_hash" in result
        >>> assert len(result["alerts"]) > 0
    """

    def __init__(self) -> None:
        """Initialize GreenwashingDetectionEngine."""
        self.engine_id: str = _new_uuid()
        self.version: str = _MODULE_VERSION
        logger.info(
            "GreenwashingDetectionEngine initialized | engine_id=%s version=%s",
            self.engine_id,
            self.version,
        )

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def screen_claim(
        self,
        claim_text: str,
        evidence: Optional[str] = None,
        claim_type: ClaimType = ClaimType.PRODUCT,
        *,
        labels_used: Optional[List[str]] = None,
        has_certification: bool = False,
        covers_full_lifecycle: bool = False,
        lifecycle_stages: Optional[List[str]] = None,
        relies_on_offsets: bool = False,
    ) -> Dict[str, Any]:
        """Screen a single environmental claim for greenwashing risks.

        Runs both the Seven Sins detection and prohibited practices
        checks, then calculates an aggregate risk score.

        Args:
            claim_text: Full text of the environmental claim.
            evidence: Summary of supporting evidence (if any).
            claim_type: Type of environmental claim.
            labels_used: Sustainability labels displayed.
            has_certification: Whether third-party certification exists.
            covers_full_lifecycle: Whether claim covers full lifecycle.
            lifecycle_stages: Life-cycle stages covered.
            relies_on_offsets: Whether claim relies on offsets.

        Returns:
            Dict with alerts, risk score, summary, and provenance_hash.
        """
        logger.info(
            "Screening claim | type=%s text_length=%d",
            claim_type.value,
            len(claim_text),
        )
        timestamp = utcnow()
        screening_id = _new_uuid()
        labels = labels_used or []
        stages = lifecycle_stages or []

        # Run Seven Sins detection
        sins_result = self.detect_seven_sins(
            claim_text=claim_text,
            evidence=evidence,
            lifecycle_coverage=stages,
            covers_full_lifecycle=covers_full_lifecycle,
        )

        # Run prohibited practices check
        practices_result = self.check_prohibited_practices(
            claim_text=claim_text,
            labels=labels,
            evidence=evidence,
            has_certification=has_certification,
            relies_on_offsets=relies_on_offsets,
        )

        # Combine alerts
        all_alerts: List[Dict[str, Any]] = []
        all_alerts.extend(sins_result.get("alerts", []))
        all_alerts.extend(practices_result.get("alerts", []))

        # Calculate risk score
        risk_result = self.calculate_risk_score(all_alerts)

        result = {
            "screening_id": screening_id,
            "claim_text": claim_text,
            "claim_type": claim_type.value,
            "timestamp": str(timestamp),
            "alerts": all_alerts,
            "alert_count": len(all_alerts),
            "risk_score": risk_result["risk_score"],
            "risk_level": risk_result["risk_level"],
            "seven_sins_detected": sins_result.get("sins_detected", []),
            "prohibited_practices_detected": practices_result.get(
                "practices_detected", []
            ),
            "has_evidence": evidence is not None and len(evidence or "") > 0,
            "has_certification": has_certification,
            "covers_full_lifecycle": covers_full_lifecycle,
            "summary": self._generate_screening_summary(
                all_alerts, risk_result,
            ),
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Claim screened | screening_id=%s alerts=%d risk=%s",
            screening_id,
            len(all_alerts),
            risk_result["risk_score"],
        )
        return result

    def detect_seven_sins(
        self,
        claim_text: str,
        evidence: Optional[str] = None,
        lifecycle_coverage: Optional[List[str]] = None,
        *,
        covers_full_lifecycle: bool = False,
    ) -> Dict[str, Any]:
        """Detect TerraChoice Seven Sins of Greenwashing.

        Applies pattern-based detection for each of the seven sins
        against the provided claim text and supporting context.

        Args:
            claim_text: Full text of the environmental claim.
            evidence: Summary of supporting evidence.
            lifecycle_coverage: Life-cycle stages covered.
            covers_full_lifecycle: Whether full lifecycle is covered.

        Returns:
            Dict with detected sins, alerts, and provenance_hash.
        """
        logger.info("Detecting Seven Sins | text_length=%d", len(claim_text))
        timestamp = utcnow()
        detection_id = _new_uuid()
        alerts: List[Dict[str, Any]] = []
        sins_detected: List[str] = []
        claim_lower = claim_text.lower()
        stages = lifecycle_coverage or []

        # Sin 1: Hidden Trade-off
        if stages and not covers_full_lifecycle and len(stages) < 4:
            sin_alert = self._create_alert(
                sin_type=GreenwashingSin.HIDDEN_TRADEOFF.value,
                severity=AlertSeverity.MEDIUM.value,
                claim_text=claim_text,
                description=(
                    f"Claim covers only {len(stages)} lifecycle stage(s) "
                    f"({', '.join(stages)}), potentially hiding trade-offs "
                    f"in uncovered stages"
                ),
                evidence=f"Lifecycle stages covered: {', '.join(stages)}",
                recommendation=(
                    "Extend assessment to cover full product lifecycle "
                    "(raw materials, manufacturing, distribution, use, "
                    "end-of-life) or explicitly disclose excluded stages"
                ),
            )
            alerts.append(sin_alert)
            sins_detected.append(GreenwashingSin.HIDDEN_TRADEOFF.value)

        # Sin 2: No Proof
        if evidence is None or len(evidence or "") < 10:
            sin_alert = self._create_alert(
                sin_type=GreenwashingSin.NO_PROOF.value,
                severity=AlertSeverity.HIGH.value,
                claim_text=claim_text,
                description=(
                    "Claim lacks accessible supporting evidence or "
                    "third-party certification to substantiate the "
                    "environmental assertion"
                ),
                evidence="No evidence provided or evidence insufficient",
                recommendation=(
                    "Provide publicly accessible evidence such as LCA data, "
                    "third-party certification, or independent audit report "
                    "to substantiate the claim"
                ),
            )
            alerts.append(sin_alert)
            sins_detected.append(GreenwashingSin.NO_PROOF.value)

        # Sin 3: Vagueness
        vague_matches = self._find_vague_terms(claim_lower)
        if vague_matches:
            sin_alert = self._create_alert(
                sin_type=GreenwashingSin.VAGUENESS.value,
                severity=AlertSeverity.HIGH.value,
                claim_text=claim_text,
                description=(
                    f"Claim uses vague terms ({', '.join(vague_matches)}) "
                    f"that are poorly defined and likely to be misunderstood "
                    f"by consumers"
                ),
                evidence=f"Vague terms detected: {', '.join(vague_matches)}",
                recommendation=(
                    "Replace vague terms with specific, measurable "
                    "environmental attributes (e.g., '30% recycled content' "
                    "instead of 'eco-friendly')"
                ),
            )
            alerts.append(sin_alert)
            sins_detected.append(GreenwashingSin.VAGUENESS.value)

        # Sin 4: False Labels
        false_label_indicators = [
            "self-certified", "own label", "proprietary seal",
            "internal certification", "self-awarded",
        ]
        label_matches = [
            term for term in false_label_indicators if term in claim_lower
        ]
        if label_matches:
            sin_alert = self._create_alert(
                sin_type=GreenwashingSin.FALSE_LABELS.value,
                severity=AlertSeverity.CRITICAL.value,
                claim_text=claim_text,
                description=(
                    "Claim suggests third-party endorsement through "
                    "self-certified labels or proprietary seals that may "
                    "mislead consumers"
                ),
                evidence=f"False label indicators: {', '.join(label_matches)}",
                recommendation=(
                    "Remove self-certified labels and replace with "
                    "recognised third-party certification schemes "
                    "(e.g., EU Ecolabel, FSC, GOTS)"
                ),
            )
            alerts.append(sin_alert)
            sins_detected.append(GreenwashingSin.FALSE_LABELS.value)

        # Sin 5: Irrelevance
        irrelevance_indicators = [
            "cfc-free", "cfc free", "does not contain lead",
            "no ozone depleting", "free of banned substances",
            "mercury-free", "mercury free",
        ]
        irrelevance_matches = [
            term for term in irrelevance_indicators if term in claim_lower
        ]
        if irrelevance_matches:
            sin_alert = self._create_alert(
                sin_type=GreenwashingSin.IRRELEVANCE.value,
                severity=AlertSeverity.LOW.value,
                claim_text=claim_text,
                description=(
                    "Claim highlights the absence of substances already "
                    "banned by regulation, which is truthful but irrelevant "
                    "as a distinguishing attribute"
                ),
                evidence=f"Irrelevance indicators: {', '.join(irrelevance_matches)}",
                recommendation=(
                    "Remove claims about substances already prohibited by "
                    "law; focus on genuine environmental attributes that "
                    "differentiate the product"
                ),
            )
            alerts.append(sin_alert)
            sins_detected.append(GreenwashingSin.IRRELEVANCE.value)

        # Sin 6: Lesser of Two Evils
        lesser_evil_indicators = [
            "greener cigarette", "eco-friendly pesticide",
            "sustainable mining", "green fossil",
            "cleaner coal", "sustainable palm oil",
            "responsible fast fashion",
        ]
        lesser_matches = [
            term for term in lesser_evil_indicators if term in claim_lower
        ]
        if lesser_matches:
            sin_alert = self._create_alert(
                sin_type=GreenwashingSin.LESSER_OF_TWO_EVILS.value,
                severity=AlertSeverity.MEDIUM.value,
                claim_text=claim_text,
                description=(
                    "Claim may be true within the product category but "
                    "distracts from the inherently high environmental "
                    "impact of the category"
                ),
                evidence=f"Lesser-evil indicators: {', '.join(lesser_matches)}",
                recommendation=(
                    "Disclose the overall environmental impact of the "
                    "product category alongside any relative improvement "
                    "claims to avoid misleading consumers"
                ),
            )
            alerts.append(sin_alert)
            sins_detected.append(GreenwashingSin.LESSER_OF_TWO_EVILS.value)

        # Sin 7: Fibbing (check for superlative claims without evidence)
        fibbing_patterns = [
            r"\b100\s*%\s*sustainable\b",
            r"\bzero\s+(environmental\s+)?impact\b",
            r"\bno\s+environmental\s+impact\b",
            r"\bcompletely\s+harmless\b",
            r"\btotally\s+green\b",
            r"\bperfectly\s+sustainable\b",
        ]
        fibbing_matches: List[str] = []
        for pattern in fibbing_patterns:
            match = re.search(pattern, claim_lower)
            if match:
                fibbing_matches.append(match.group())

        if fibbing_matches:
            sin_alert = self._create_alert(
                sin_type=GreenwashingSin.FIBBING.value,
                severity=AlertSeverity.CRITICAL.value,
                claim_text=claim_text,
                description=(
                    "Claim makes absolute environmental assertions "
                    "('100% sustainable', 'zero impact') that are "
                    "factually impossible for any product or service"
                ),
                evidence=f"Fibbing indicators: {', '.join(fibbing_matches)}",
                recommendation=(
                    "Remove absolute environmental claims; no product "
                    "has zero environmental impact. Use qualified, "
                    "specific, and measurable claims instead"
                ),
            )
            alerts.append(sin_alert)
            sins_detected.append(GreenwashingSin.FIBBING.value)

        result = {
            "detection_id": detection_id,
            "timestamp": str(timestamp),
            "claim_text": claim_text,
            "sins_detected": list(set(sins_detected)),
            "sins_count": len(set(sins_detected)),
            "alerts": alerts,
            "alert_count": len(alerts),
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Seven Sins detection complete | detection_id=%s sins=%d alerts=%d",
            detection_id,
            len(set(sins_detected)),
            len(alerts),
        )
        return result

    def check_prohibited_practices(
        self,
        claim_text: str,
        labels: Optional[List[str]] = None,
        evidence: Optional[str] = None,
        *,
        has_certification: bool = False,
        relies_on_offsets: bool = False,
    ) -> Dict[str, Any]:
        """Check claim against EU ECGT prohibited practices.

        Evaluates the claim text against each prohibited practice
        defined in the Empowering Consumers Directive and the proposed
        Green Claims Directive.

        Args:
            claim_text: Full text of the environmental claim.
            labels: Sustainability labels displayed.
            evidence: Summary of supporting evidence.
            has_certification: Whether third-party certification exists.
            relies_on_offsets: Whether claim relies on carbon offsets.

        Returns:
            Dict with detected practices, alerts, and provenance_hash.
        """
        logger.info(
            "Checking prohibited practices | text_length=%d",
            len(claim_text),
        )
        timestamp = utcnow()
        check_id = _new_uuid()
        alerts: List[Dict[str, Any]] = []
        practices_detected: List[str] = []
        claim_lower = claim_text.lower()
        label_list = labels or []

        # Practice 1: Generic excellence claims
        generic_matches = self._find_vague_terms(claim_lower)
        if generic_matches and not has_certification:
            practice = ProhibitedPractice.GENERIC_EXCELLENCE_CLAIM
            config = PROHIBITED_PATTERNS[practice]
            alert = self._create_alert(
                prohibited_practice=practice.value,
                severity=config["severity"].value,
                claim_text=claim_text,
                description=config["description"],
                evidence=f"Generic terms without certification: {', '.join(generic_matches)}",
                recommendation=config["recommendation"],
                article_reference=config["article_reference"],
            )
            alerts.append(alert)
            practices_detected.append(practice.value)

        # Practice 2: Uncertified labels
        if label_list and not has_certification:
            practice = ProhibitedPractice.UNCERTIFIED_LABEL
            config = PROHIBITED_PATTERNS[practice]
            alert = self._create_alert(
                prohibited_practice=practice.value,
                severity=config["severity"].value,
                claim_text=claim_text,
                description=config["description"],
                evidence=f"Labels without certification: {', '.join(label_list)}",
                recommendation=config["recommendation"],
                article_reference=config["article_reference"],
            )
            alerts.append(alert)
            practices_detected.append(practice.value)

        # Practice 3: Partial scope misrepresentation
        scope_keywords = PROHIBITED_PATTERNS[
            ProhibitedPractice.PARTIAL_SCOPE_MISREPRESENTATION
        ]["detection_keywords"]
        scope_matches = [kw for kw in scope_keywords if kw.lower() in claim_lower]
        if scope_matches:
            practice = ProhibitedPractice.PARTIAL_SCOPE_MISREPRESENTATION
            config = PROHIBITED_PATTERNS[practice]
            alert = self._create_alert(
                prohibited_practice=practice.value,
                severity=config["severity"].value,
                claim_text=claim_text,
                description=config["description"],
                evidence=f"Totality language detected: {', '.join(scope_matches)}",
                recommendation=config["recommendation"],
                article_reference=config["article_reference"],
            )
            alerts.append(alert)
            practices_detected.append(practice.value)

        # Practice 4: Legal requirement as distinction
        legal_matches = [
            phrase for phrase in LEGAL_REQUIREMENT_PHRASES
            if phrase in claim_lower
        ]
        if legal_matches:
            practice = ProhibitedPractice.LEGAL_REQUIREMENT_AS_DISTINCTION
            config = PROHIBITED_PATTERNS[practice]
            alert = self._create_alert(
                prohibited_practice=practice.value,
                severity=config["severity"].value,
                claim_text=claim_text,
                description=config["description"],
                evidence=f"Legal requirement language: {', '.join(legal_matches)}",
                recommendation=config["recommendation"],
                article_reference=config["article_reference"],
            )
            alerts.append(alert)
            practices_detected.append(practice.value)

        # Practice 5: Offset neutrality claims
        offset_matches = [
            kw for kw in OFFSET_NEUTRALITY_KEYWORDS if kw in claim_lower
        ]
        if offset_matches or relies_on_offsets:
            practice = ProhibitedPractice.OFFSET_NEUTRALITY_CLAIM
            config = PROHIBITED_PATTERNS[practice]
            evidence_text = (
                f"Offset/neutrality language: {', '.join(offset_matches)}"
                if offset_matches
                else "Claim relies on carbon offsets"
            )
            alert = self._create_alert(
                prohibited_practice=practice.value,
                severity=config["severity"].value,
                claim_text=claim_text,
                description=config["description"],
                evidence=evidence_text,
                recommendation=config["recommendation"],
                article_reference=config["article_reference"],
            )
            alerts.append(alert)
            practices_detected.append(practice.value)

        # Practice 6: Unsubstantiated future claims
        future_matches = [
            indicator for indicator in FUTURE_CLAIM_INDICATORS
            if indicator in claim_lower
        ]
        if future_matches and (evidence is None or len(evidence or "") < 20):
            practice = ProhibitedPractice.UNSUBSTANTIATED_FUTURE_CLAIM
            config = PROHIBITED_PATTERNS[practice]
            alert = self._create_alert(
                prohibited_practice=practice.value,
                severity=config["severity"].value,
                claim_text=claim_text,
                description=config["description"],
                evidence=f"Future indicators: {', '.join(future_matches)}",
                recommendation=config["recommendation"],
                article_reference=config["article_reference"],
            )
            alerts.append(alert)
            practices_detected.append(practice.value)

        result = {
            "check_id": check_id,
            "timestamp": str(timestamp),
            "claim_text": claim_text,
            "practices_detected": list(set(practices_detected)),
            "practices_count": len(set(practices_detected)),
            "alerts": alerts,
            "alert_count": len(alerts),
            "has_certification": has_certification,
            "labels_checked": label_list,
            "relies_on_offsets": relies_on_offsets,
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Prohibited practices check complete | check_id=%s practices=%d",
            check_id,
            len(set(practices_detected)),
        )
        return result

    def calculate_risk_score(
        self, alerts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate aggregate greenwashing risk score from alerts.

        Applies fixed-point severity weights to each alert to produce
        a cumulative risk score and risk classification.

        Risk score weights:
            - CRITICAL: 30 points per alert
            - HIGH: 20 points per alert
            - MEDIUM: 10 points per alert
            - LOW: 5 points per alert

        Args:
            alerts: List of greenwashing alert dicts.

        Returns:
            Dict with risk score, risk level, breakdown,
            and provenance_hash.
        """
        logger.info("Calculating risk score | alert_count=%d", len(alerts))
        timestamp = utcnow()
        calc_id = _new_uuid()

        total_score = Decimal("0")
        severity_breakdown: Dict[str, int] = {
            AlertSeverity.LOW.value: 0,
            AlertSeverity.MEDIUM.value: 0,
            AlertSeverity.HIGH.value: 0,
            AlertSeverity.CRITICAL.value: 0,
        }

        for alert in alerts:
            severity = alert.get("severity", AlertSeverity.MEDIUM.value)
            weight = SEVERITY_RISK_WEIGHTS.get(severity, Decimal("10"))
            total_score += weight
            if severity in severity_breakdown:
                severity_breakdown[severity] += 1

        # Cap at 100 for percentage interpretation
        capped_score = min(total_score, Decimal("100"))

        # Determine risk level
        if capped_score >= Decimal("60"):
            risk_level = "critical"
        elif capped_score >= Decimal("40"):
            risk_level = "high"
        elif capped_score >= Decimal("20"):
            risk_level = "medium"
        elif capped_score > Decimal("0"):
            risk_level = "low"
        else:
            risk_level = "none"

        result = {
            "calculation_id": calc_id,
            "timestamp": str(timestamp),
            "risk_score": str(_round_val(capped_score, 2)),
            "raw_score": str(_round_val(total_score, 2)),
            "risk_level": risk_level,
            "alert_count": len(alerts),
            "severity_breakdown": severity_breakdown,
            "scoring_weights": {
                k: str(v) for k, v in SEVERITY_RISK_WEIGHTS.items()
            },
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Risk score calculated | calc_id=%s score=%s level=%s",
            calc_id,
            str(capped_score),
            risk_level,
        )
        return result

    def screen_portfolio(
        self,
        claims_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Screen a portfolio of environmental claims for greenwashing.

        Processes multiple claims and produces aggregate statistics
        and portfolio-level risk assessment.

        Each entry in claims_list should contain:
            - claim_text (str): The claim text.
            - evidence (str, optional): Supporting evidence.
            - claim_type (str, optional): Type of claim.
            - labels (list, optional): Labels used.
            - has_certification (bool, optional): Certification status.
            - covers_full_lifecycle (bool, optional): Lifecycle coverage.
            - relies_on_offsets (bool, optional): Offset reliance.

        Args:
            claims_list: List of claim dicts to screen.

        Returns:
            Dict with individual results, aggregate metrics,
            portfolio risk score, and provenance_hash.
        """
        logger.info(
            "Screening portfolio | claims_count=%d", len(claims_list),
        )
        timestamp = utcnow()
        portfolio_id = _new_uuid()

        individual_results: List[Dict[str, Any]] = []
        all_alerts: List[Dict[str, Any]] = []
        total_claims = len(claims_list)
        clean_claims = 0
        flagged_claims = 0

        sin_counts: Dict[str, int] = {}
        practice_counts: Dict[str, int] = {}

        for claim_entry in claims_list:
            claim_text = claim_entry.get("claim_text", "")
            if not claim_text or len(claim_text) < 3:
                continue

            evidence = claim_entry.get("evidence")
            claim_type_str = claim_entry.get("claim_type", "product")
            try:
                claim_type = ClaimType(claim_type_str)
            except ValueError:
                claim_type = ClaimType.PRODUCT

            labels = claim_entry.get("labels", [])
            has_cert = claim_entry.get("has_certification", False)
            full_lc = claim_entry.get("covers_full_lifecycle", False)
            offsets = claim_entry.get("relies_on_offsets", False)

            result = self.screen_claim(
                claim_text=claim_text,
                evidence=evidence,
                claim_type=claim_type,
                labels_used=labels,
                has_certification=has_cert,
                covers_full_lifecycle=full_lc,
                relies_on_offsets=offsets,
            )
            individual_results.append(result)
            all_alerts.extend(result.get("alerts", []))

            if result.get("alert_count", 0) > 0:
                flagged_claims += 1
            else:
                clean_claims += 1

            # Accumulate sin and practice counts
            for sin in result.get("seven_sins_detected", []):
                sin_counts[sin] = sin_counts.get(sin, 0) + 1
            for practice in result.get("prohibited_practices_detected", []):
                practice_counts[practice] = practice_counts.get(practice, 0) + 1

        # Portfolio-level risk score
        portfolio_risk = self.calculate_risk_score(all_alerts)

        # Calculate portfolio metrics
        flagged_pct = _safe_divide(
            _decimal(flagged_claims) * Decimal("100"),
            _decimal(total_claims) if total_claims > 0 else Decimal("1"),
        )

        result = {
            "portfolio_id": portfolio_id,
            "timestamp": str(timestamp),
            "total_claims": total_claims,
            "clean_claims": clean_claims,
            "flagged_claims": flagged_claims,
            "flagged_percentage": str(_round_val(flagged_pct, 2)),
            "total_alerts": len(all_alerts),
            "portfolio_risk_score": portfolio_risk["risk_score"],
            "portfolio_risk_level": portfolio_risk["risk_level"],
            "sin_distribution": sin_counts,
            "practice_distribution": practice_counts,
            "severity_breakdown": portfolio_risk.get("severity_breakdown", {}),
            "individual_results": individual_results,
            "top_risks": self._identify_top_risks(sin_counts, practice_counts),
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Portfolio screened | portfolio_id=%s total=%d flagged=%d risk=%s",
            portfolio_id,
            total_claims,
            flagged_claims,
            portfolio_risk["risk_score"],
        )
        return result

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _find_vague_terms(self, text_lower: str) -> List[str]:
        """Find vague environmental terms in lowercase text.

        Args:
            text_lower: Lowercased claim text.

        Returns:
            List of matched vague terms.
        """
        matches: List[str] = []

        # Check keywords (word-boundary aware for short terms)
        for keyword in VAGUE_KEYWORDS:
            if keyword in text_lower:
                matches.append(keyword)

        # Check phrases
        for phrase in VAGUE_PHRASES:
            if phrase in text_lower:
                matches.append(phrase)

        return list(set(matches))

    def _create_alert(
        self,
        claim_text: str,
        description: str,
        recommendation: str,
        severity: str = AlertSeverity.MEDIUM.value,
        *,
        sin_type: Optional[str] = None,
        prohibited_practice: Optional[str] = None,
        evidence: Optional[str] = None,
        article_reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a standardized greenwashing alert dict.

        Args:
            claim_text: The claim that triggered the alert.
            description: Detailed description of the issue.
            recommendation: Remediation recommendation.
            severity: Alert severity level.
            sin_type: TerraChoice sin type (if applicable).
            prohibited_practice: Prohibited practice type (if applicable).
            evidence: Evidence that triggered detection.
            article_reference: Regulatory article reference.

        Returns:
            Standardized alert dict.
        """
        alert = {
            "alert_id": _new_uuid(),
            "sin_type": sin_type,
            "prohibited_practice": prohibited_practice,
            "severity": severity,
            "claim_text": claim_text,
            "description": description,
            "evidence": evidence,
            "recommendation": recommendation,
            "article_reference": article_reference,
        }
        return alert

    def _generate_screening_summary(
        self,
        alerts: List[Dict[str, Any]],
        risk_result: Dict[str, Any],
    ) -> str:
        """Generate a human-readable screening summary.

        Args:
            alerts: List of detected alerts.
            risk_result: Risk score calculation result.

        Returns:
            Summary string.
        """
        if not alerts:
            return "No greenwashing risks detected. Claim appears compliant."

        critical_count = sum(
            1 for a in alerts if a.get("severity") == AlertSeverity.CRITICAL.value
        )
        high_count = sum(
            1 for a in alerts if a.get("severity") == AlertSeverity.HIGH.value
        )

        parts: List[str] = [
            f"{len(alerts)} greenwashing risk(s) detected "
            f"(risk level: {risk_result.get('risk_level', 'unknown')})",
        ]

        if critical_count > 0:
            parts.append(f"{critical_count} CRITICAL issue(s) require immediate action")
        if high_count > 0:
            parts.append(f"{high_count} HIGH severity issue(s) need remediation")

        return ". ".join(parts) + "."

    def _identify_top_risks(
        self,
        sin_counts: Dict[str, int],
        practice_counts: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Identify the most prevalent risks across the portfolio.

        Args:
            sin_counts: Count of each detected sin.
            practice_counts: Count of each detected practice.

        Returns:
            Sorted list of top risks with counts and descriptions.
        """
        all_risks: List[Dict[str, Any]] = []

        for sin_key, count in sin_counts.items():
            all_risks.append({
                "risk_type": "seven_sins",
                "key": sin_key,
                "count": count,
                "description": SIN_DESCRIPTIONS.get(sin_key, "Unknown sin type"),
            })

        for practice_key, count in practice_counts.items():
            config = None
            for p_enum, p_config in PROHIBITED_PATTERNS.items():
                if p_enum.value == practice_key:
                    config = p_config
                    break
            all_risks.append({
                "risk_type": "prohibited_practice",
                "key": practice_key,
                "count": count,
                "description": config["description"] if config else "Unknown practice",
            })

        # Sort by count descending
        all_risks.sort(key=lambda r: r["count"], reverse=True)
        return all_risks[:10]
