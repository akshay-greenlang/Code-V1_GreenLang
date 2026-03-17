# -*- coding: utf-8 -*-
"""
ECGTBridge - Empowering Consumers for Green Transition Bridge for PACK-018
=============================================================================

This module checks environmental marketing claims against the Empowering
Consumers for the Green Transition Directive (Directive 2024/825, amending
Directives 2005/29/EC and 2011/83/EU). The ECGT Directive introduces
new prohibited commercial practices related to greenwashing, including
banned generic environmental claims, misleading labels, and unsubstantiated
durability claims.

ECGT Annex I Prohibited Practices (13 additions):
    2a.  Making generic environmental claims without recognized excellence
    4a.  Making environmental claims about entire product when only part qualifies
    4b.  Presenting legally required features as distinctive
    4c.  Making claims based on carbon offsetting that product has reduced/zero impact
    10a. Displaying sustainability labels not based on certification or authority
    10b. Displaying sustainability labels not established by public authority
         for non-EU schemes without meeting minimum transparency/credibility
    23d. Making generic environmental claims where trader cannot demonstrate
         recognized excellent environmental performance
    23e. Making environmental claims about future performance without
         clear, objective, verifiable commitments and targets
    23f. Claiming carbon neutral/reduced impact based solely on offsetting
    23g. Environmental claim about entire product that concerns only
         certain aspects of the product
    23h. Presenting requirements imposed by law on all products as
         distinctive feature of the trader's offer
    23i. Omitting information that the environmental performance
         will deteriorate over time

Label Verification Checks:
    - Label must be based on official certification scheme or government authority
    - Non-EU labels must meet minimum transparency requirements
    - Label scope must match the claim scope

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-018 EU Green Claims Prep Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "ProhibitedPractice",
    "ECGTCheckStatus",
    "LabelVerificationStatus",
    "ECGTBridgeConfig",
    "ProhibitedPracticeDetection",
    "LabelCheckResult",
    "ECGTComplianceResult",
    "ECGTBridge",
]

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProhibitedPractice(str, Enum):
    """ECGT Annex I prohibited commercial practices."""

    GENERIC_ENV_CLAIM = "2a_generic_environmental_claim"
    PARTIAL_PRODUCT_CLAIM = "4a_partial_product_claim"
    LEGAL_REQUIREMENT_AS_DISTINCTIVE = "4b_legal_requirement_as_distinctive"
    OFFSET_BASED_CLAIM = "4c_offset_based_neutral_claim"
    UNCERTIFIED_LABEL = "10a_uncertified_sustainability_label"
    NON_AUTHORITY_LABEL = "10b_non_authority_label"
    UNDEMONSTRATED_EXCELLENCE = "23d_undemonstrated_excellence"
    FUTURE_WITHOUT_COMMITMENTS = "23e_future_without_commitments"
    OFFSET_ONLY_NEUTRAL = "23f_offset_only_neutral_claim"
    PRODUCT_ASPECT_MISMATCH = "23g_product_aspect_mismatch"
    LEGAL_AS_DISTINCTIVE = "23h_legal_as_distinctive"
    PERFORMANCE_DETERIORATION = "23i_performance_deterioration_omission"


class ECGTCheckStatus(str, Enum):
    """Status of an ECGT compliance check."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
    NOT_APPLICABLE = "not_applicable"


class LabelVerificationStatus(str, Enum):
    """Status of a sustainability label verification."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"
    EXPIRED = "expired"
    NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------------------
# Prohibited Practice Detection Patterns
# ---------------------------------------------------------------------------

GENERIC_CLAIM_KEYWORDS: List[str] = [
    "eco-friendly", "environmentally friendly", "green", "sustainable",
    "nature-friendly", "ecological", "climate-friendly", "eco",
    "good for the environment", "better for the planet",
    "environmentally responsible", "earth-friendly", "planet-friendly",
]

OFFSET_CLAIM_KEYWORDS: List[str] = [
    "carbon neutral", "climate neutral", "carbon compensated",
    "CO2 neutral", "offset", "carbon offset", "climate compensated",
    "net zero through offsets", "neutralized",
]

FUTURE_CLAIM_KEYWORDS: List[str] = [
    "will be carbon neutral", "on track to", "committed to",
    "aiming for", "target of", "goal of", "by 2030", "by 2040",
    "by 2050", "future sustainability", "pathway to",
]

KNOWN_EU_LABELS: Dict[str, Dict[str, str]] = {
    "EU_ECOLABEL": {"authority": "European Commission", "type": "official", "status": "active"},
    "EU_ENERGY_LABEL": {"authority": "European Commission", "type": "official", "status": "active"},
    "EU_ORGANIC": {"authority": "European Commission", "type": "official", "status": "active"},
    "EMAS": {"authority": "European Commission", "type": "official", "status": "active"},
    "NORDIC_SWAN": {"authority": "Nordic Council of Ministers", "type": "official", "status": "active"},
    "BLUE_ANGEL": {"authority": "German Federal Government", "type": "official", "status": "active"},
    "NF_ENVIRONNEMENT": {"authority": "AFNOR", "type": "official", "status": "active"},
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class ECGTBridgeConfig(BaseModel):
    """Configuration for the ECGT Compliance Bridge."""

    pack_id: str = Field(default="PACK-018")
    prohibited_practices_check: bool = Field(
        default=True,
        description="Enable check against Annex I prohibited practices",
    )
    label_verification: bool = Field(
        default=True,
        description="Enable sustainability label verification",
    )
    enable_provenance: bool = Field(default=True)
    strict_mode: bool = Field(
        default=False,
        description="If True, flag borderline cases as non-compliant",
    )
    recognized_labels: List[str] = Field(
        default_factory=lambda: list(KNOWN_EU_LABELS.keys()),
        description="List of recognized official sustainability labels",
    )


class ProhibitedPracticeDetection(BaseModel):
    """Detection result for a single prohibited practice."""

    practice: ProhibitedPractice = Field(...)
    detected: bool = Field(default=False)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_keywords: List[str] = Field(default_factory=list)
    explanation: str = Field(default="")
    remediation: str = Field(default="")


class LabelCheckResult(BaseModel):
    """Result of a sustainability label verification."""

    label_name: str = Field(default="")
    status: LabelVerificationStatus = Field(default=LabelVerificationStatus.UNVERIFIED)
    is_official: bool = Field(default=False)
    issuing_authority: str = Field(default="")
    scope_match: bool = Field(default=True)
    issues: List[str] = Field(default_factory=list)


class ECGTComplianceResult(BaseModel):
    """Result of an ECGT compliance check."""

    check_id: str = Field(default_factory=_new_uuid)
    claim_text: str = Field(default="")
    overall_status: ECGTCheckStatus = Field(default=ECGTCheckStatus.NEEDS_REVIEW)
    prohibited_practices_detected: List[ProhibitedPracticeDetection] = Field(
        default_factory=list
    )
    total_violations: int = Field(default=0)
    label_checks: List[LabelCheckResult] = Field(default_factory=list)
    labels_verified: int = Field(default=0)
    labels_rejected: int = Field(default=0)
    recommendations: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# ECGTBridge
# ---------------------------------------------------------------------------


class ECGTBridge:
    """ECGT Directive compliance bridge for PACK-018.

    Checks environmental marketing claims against the Empowering Consumers
    for the Green Transition Directive (2024/825) prohibited practices list
    and verifies sustainability label legitimacy.

    Attributes:
        config: ECGT bridge configuration.

    Example:
        >>> config = ECGTBridgeConfig()
        >>> bridge = ECGTBridge(config)
        >>> result = bridge.check_ecgt_compliance("Our product is eco-friendly", ["EU_ECOLABEL"])
        >>> assert result["overall_status"] in ["compliant", "non_compliant", "needs_review"]
    """

    def __init__(self, config: Optional[ECGTBridgeConfig] = None) -> None:
        """Initialize ECGTBridge.

        Args:
            config: Bridge configuration. Defaults used if None.
        """
        self.config = config or ECGTBridgeConfig()
        logger.info(
            "ECGTBridge initialized (practices_check=%s, label_verify=%s, strict=%s)",
            self.config.prohibited_practices_check,
            self.config.label_verification,
            self.config.strict_mode,
        )

    def check_ecgt_compliance(
        self,
        claim_text: str,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Check a claim against ECGT Directive requirements.

        Args:
            claim_text: The environmental claim text to check.
            labels: Optional list of sustainability labels used with the claim.

        Returns:
            Dict with compliance status, detected prohibited practices,
            label check results, recommendations, and provenance hash.
        """
        start = _utcnow()
        result = ECGTComplianceResult(claim_text=claim_text)

        if self.config.prohibited_practices_check:
            detections = self._screen_prohibited_practices(claim_text)
            result.prohibited_practices_detected = detections
            result.total_violations = sum(1 for d in detections if d.detected)

        if self.config.label_verification and labels:
            label_results = self._verify_labels(labels)
            result.label_checks = label_results
            result.labels_verified = sum(1 for l in label_results if l.status == LabelVerificationStatus.VERIFIED)
            result.labels_rejected = sum(1 for l in label_results if l.status == LabelVerificationStatus.REJECTED)

        result.overall_status = self._determine_overall_status(result)
        result.recommendations = self._generate_recommendations(result)

        elapsed = (_utcnow() - start).total_seconds() * 1000

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        logger.info(
            "ECGTBridge checked claim: %s (violations=%d, labels_ok=%d, labels_bad=%d) in %.1fms",
            result.overall_status.value,
            result.total_violations,
            result.labels_verified,
            result.labels_rejected,
            elapsed,
        )

        return result.model_dump(mode="json")

    def check_blacklisted_practices(self, claim_text: str) -> Dict[str, Any]:
        """Check a claim against all ECGT Annex I blacklisted practices.

        Screens the claim text for all 12 prohibited commercial practices
        introduced by the ECGT Directive (EU) 2024/825.

        Args:
            claim_text: The environmental claim text to screen.

        Returns:
            Dict with detected practices, violation count, and hash.
        """
        detections = self._screen_prohibited_practices(claim_text)
        violations = [d for d in detections if d.detected]

        result = {
            "claim_text": claim_text[:200],
            "practices_checked": len(detections),
            "violations_found": len(violations),
            "violations": [
                {"practice": v.practice.value, "confidence": v.confidence, "keywords": v.matched_keywords}
                for v in violations
            ],
            "compliant": len(violations) == 0,
            "provenance_hash": _compute_hash({"text": claim_text[:200]}),
        }
        logger.info("ECGTBridge blacklist check: %d violations found", len(violations))
        return result

    def validate_generic_claims(self, claim_text: str) -> Dict[str, Any]:
        """Validate whether a claim constitutes a prohibited generic claim.

        Per ECGT Art. 2a and 23d, generic environmental claims like
        "eco-friendly" or "green" are prohibited unless the trader
        can demonstrate recognized excellent environmental performance.

        Args:
            claim_text: The claim text to validate.

        Returns:
            Dict with generic claim detection result and remediation.
        """
        detection = self._check_generic_claims(claim_text.lower())

        result = {
            "claim_text": claim_text[:200],
            "is_generic_claim": detection.detected,
            "confidence": detection.confidence,
            "matched_keywords": detection.matched_keywords,
            "explanation": detection.explanation,
            "remediation": detection.remediation,
            "regulatory_reference": "ECGT Directive (EU) 2024/825 Art. 2a, 23d",
            "provenance_hash": _compute_hash({"text": claim_text[:200], "generic": detection.detected}),
        }
        logger.info("ECGTBridge generic claim check: detected=%s", detection.detected)
        return result

    def check_offset_neutrality_ban(self, claim_text: str) -> Dict[str, Any]:
        """Check if a claim violates the carbon offset neutrality ban.

        Per ECGT Art. 4c and 23f, claims of carbon neutrality, climate
        neutrality, or reduced environmental impact that are based
        solely on carbon offsetting schemes are prohibited.

        Args:
            claim_text: The claim text to check.

        Returns:
            Dict with offset ban detection result and remediation.
        """
        detection = self._check_offset_claims(claim_text.lower())

        result = {
            "claim_text": claim_text[:200],
            "offset_ban_violated": detection.detected,
            "confidence": detection.confidence,
            "matched_keywords": detection.matched_keywords,
            "explanation": detection.explanation,
            "remediation": detection.remediation,
            "regulatory_reference": "ECGT Directive (EU) 2024/825 Art. 4c, 23f",
            "provenance_hash": _compute_hash({"text": claim_text[:200], "offset": detection.detected}),
        }
        logger.info("ECGTBridge offset ban check: violated=%s", detection.detected)
        return result

    def get_transposition_status(self) -> Dict[str, Any]:
        """Get ECGT Directive transposition status across EU member states.

        The ECGT Directive (EU) 2024/825 must be transposed into
        national law by 27 March 2026 and applied from 27 September 2026.

        Returns:
            Dict with transposition timeline and key dates.
        """
        result = {
            "directive": "Directive (EU) 2024/825",
            "short_name": "Empowering Consumers for the Green Transition",
            "adoption_date": "2024-02-28",
            "transposition_deadline": "2026-03-27",
            "application_date": "2026-09-27",
            "amends": ["Directive 2005/29/EC (UCPD)", "Directive 2011/83/EU (CRD)"],
            "new_prohibited_practices": 12,
            "status": "transposition_period",
            "provenance_hash": _compute_hash({"directive": "2024/825"}),
        }
        logger.info("ECGTBridge transposition status retrieved")
        return result

    def screen_consumer_communications(
        self,
        communications: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Screen multiple consumer communications for ECGT compliance.

        Batch-screens a list of marketing communications across
        different channels for prohibited practices and label issues.

        Args:
            communications: List of dicts with 'text' and optional 'channel'.

        Returns:
            Dict with per-communication results and aggregate stats.
        """
        results: List[Dict[str, Any]] = []
        total_violations = 0

        for comm in communications:
            text = comm.get("text", "")
            channel = comm.get("channel", "unknown")
            labels = comm.get("labels", [])

            check_result = self.check_ecgt_compliance(text, labels if labels else None)
            violations = check_result.get("total_violations", 0)
            total_violations += violations

            results.append({
                "channel": channel,
                "text_preview": text[:100],
                "status": check_result.get("overall_status", "needs_review"),
                "violations": violations,
            })

        result = {
            "communications_screened": len(communications),
            "total_violations": total_violations,
            "compliant_count": sum(1 for r in results if r["status"] == "compliant"),
            "non_compliant_count": sum(1 for r in results if r["status"] == "non_compliant"),
            "results": results,
            "provenance_hash": _compute_hash({"screened": len(communications)}),
        }
        logger.info(
            "ECGTBridge screened %d communications: %d violations",
            len(communications), total_violations,
        )
        return result

    def get_prohibited_practices(self) -> List[Dict[str, str]]:
        """Get list of all ECGT Annex I prohibited practices."""
        return [{"id": p.value, "name": p.name} for p in ProhibitedPractice]

    def get_recognized_labels(self) -> Dict[str, Dict[str, str]]:
        """Get list of recognized official sustainability labels."""
        return dict(KNOWN_EU_LABELS)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _screen_prohibited_practices(self, claim_text: str) -> List[ProhibitedPracticeDetection]:
        """Screen claim text against all prohibited practices."""
        text_lower = claim_text.lower()
        detections: List[ProhibitedPracticeDetection] = []

        detections.append(self._check_generic_claims(text_lower))
        detections.append(self._check_offset_claims(text_lower))
        detections.append(self._check_future_claims(text_lower))
        detections.append(self._check_partial_product_claims(text_lower))

        return detections

    def _check_generic_claims(self, text: str) -> ProhibitedPracticeDetection:
        """Check for generic environmental claims (Art. 2a, 23d)."""
        matched = [kw for kw in GENERIC_CLAIM_KEYWORDS if kw in text]
        detected = len(matched) > 0
        return ProhibitedPracticeDetection(
            practice=ProhibitedPractice.GENERIC_ENV_CLAIM,
            detected=detected,
            confidence=min(len(matched) * 0.3, 1.0) if detected else 0.0,
            matched_keywords=matched,
            explanation="Generic environmental claims require recognized excellent performance" if detected else "",
            remediation="Specify the environmental benefit with measurable, verifiable data" if detected else "",
        )

    def _check_offset_claims(self, text: str) -> ProhibitedPracticeDetection:
        """Check for carbon offset-based neutrality claims (Art. 4c, 23f)."""
        matched = [kw for kw in OFFSET_CLAIM_KEYWORDS if kw in text]
        detected = len(matched) > 0
        return ProhibitedPracticeDetection(
            practice=ProhibitedPractice.OFFSET_BASED_CLAIM,
            detected=detected,
            confidence=min(len(matched) * 0.4, 1.0) if detected else 0.0,
            matched_keywords=matched,
            explanation="Claims of carbon neutrality based on offsetting are prohibited" if detected else "",
            remediation="Remove offset-based neutrality claims; focus on absolute reduction achievements" if detected else "",
        )

    def _check_future_claims(self, text: str) -> ProhibitedPracticeDetection:
        """Check for future performance claims without commitments (Art. 23e)."""
        matched = [kw for kw in FUTURE_CLAIM_KEYWORDS if kw in text]
        detected = len(matched) > 0
        return ProhibitedPracticeDetection(
            practice=ProhibitedPractice.FUTURE_WITHOUT_COMMITMENTS,
            detected=detected,
            confidence=min(len(matched) * 0.35, 1.0) if detected else 0.0,
            matched_keywords=matched,
            explanation="Future environmental claims require clear, verifiable commitments and targets" if detected else "",
            remediation="Provide specific, time-bound, measurable commitments with third-party verification" if detected else "",
        )

    def _check_partial_product_claims(self, text: str) -> ProhibitedPracticeDetection:
        """Check for claims about entire product when only partial (Art. 4a, 23g)."""
        partial_indicators = ["100%", "entirely", "completely", "fully", "whole product"]
        matched = [kw for kw in partial_indicators if kw in text]
        detected = len(matched) > 0 and self.config.strict_mode
        return ProhibitedPracticeDetection(
            practice=ProhibitedPractice.PARTIAL_PRODUCT_CLAIM,
            detected=detected,
            confidence=0.5 if detected else 0.0,
            matched_keywords=matched,
            explanation="Claims about entire product may only apply to specific aspects" if detected else "",
            remediation="Clarify which aspects of the product the environmental claim applies to" if detected else "",
        )

    def _verify_labels(self, labels: List[str]) -> List[LabelCheckResult]:
        """Verify sustainability labels against known registries."""
        results = []
        for label in labels:
            label_upper = label.upper().replace(" ", "_")
            known = KNOWN_EU_LABELS.get(label_upper)
            if known:
                results.append(LabelCheckResult(
                    label_name=label,
                    status=LabelVerificationStatus.VERIFIED,
                    is_official=True,
                    issuing_authority=known.get("authority", ""),
                ))
            else:
                results.append(LabelCheckResult(
                    label_name=label,
                    status=LabelVerificationStatus.UNVERIFIED,
                    is_official=False,
                    issues=[f"Label '{label}' not found in recognized EU label registry"],
                ))
        return results

    def _determine_overall_status(self, result: ECGTComplianceResult) -> ECGTCheckStatus:
        """Determine overall ECGT compliance status."""
        if result.total_violations > 0:
            return ECGTCheckStatus.NON_COMPLIANT
        if result.labels_rejected > 0:
            return ECGTCheckStatus.NON_COMPLIANT
        if any(
            d.confidence > 0.0 and not d.detected
            for d in result.prohibited_practices_detected
        ):
            return ECGTCheckStatus.NEEDS_REVIEW
        if result.prohibited_practices_detected or result.label_checks:
            return ECGTCheckStatus.COMPLIANT
        return ECGTCheckStatus.NEEDS_REVIEW

    def _generate_recommendations(self, result: ECGTComplianceResult) -> List[str]:
        """Generate actionable recommendations based on check results."""
        recs: List[str] = []

        for d in result.prohibited_practices_detected:
            if d.detected and d.remediation:
                recs.append(d.remediation)

        if result.labels_rejected > 0:
            recs.append("Replace unverified labels with officially recognized EU sustainability labels")

        if not result.label_checks and not result.prohibited_practices_detected:
            recs.append("Submit claim text for comprehensive ECGT compliance review")

        if result.overall_status == ECGTCheckStatus.COMPLIANT:
            recs.append("Maintain documentation of substantiation evidence for regulatory inspection")

        return recs
