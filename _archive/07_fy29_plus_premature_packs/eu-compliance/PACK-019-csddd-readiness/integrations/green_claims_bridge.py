# -*- coding: utf-8 -*-
"""
GreenClaimsBridge - CSDDD Remediation / Green Claims Cross-Validation for PACK-019
=====================================================================================

This module cross-validates CSDDD remediation claims and environmental statements
against the EU Green Claims Directive requirements. It checks that any remediation
or sustainability claims made under CSDDD due diligence are substantiated,
verifiable, and not misleading (greenwashing risk detection).

Legal References:
    - Directive (EU) 2024/1760 (CSDDD), Art 10 - Remediation
    - Directive (EU) 2024/825 (Green Claims Directive / Empowering Consumers)
    - Art 3 GCD: Substantiation of environmental claims
    - Art 5 GCD: Communication requirements for environmental claims
    - Art 10 GCD: Verification by third-party conformity bodies
    - UN Guiding Principles on Business and Human Rights (remedy pillar)

Greenwashing Risk Categories:
    - Vagueness: Claims too broad or undefined ("eco-friendly", "sustainable")
    - Selectivity: Highlighting positive while hiding negative impacts
    - False labels: Misleading certification or label imagery
    - Irrelevance: True claims that are irrelevant to the product/context
    - Hidden trade-offs: Environmental benefit offset by greater harm

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-019 CSDDD Readiness Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
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

class ClaimType(str, Enum):
    """Type of environmental or remediation claim."""

    REMEDIATION = "remediation"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    CLIMATE = "climate"
    CIRCULAR_ECONOMY = "circular_economy"
    BIODIVERSITY = "biodiversity"

class SubstantiationStatus(str, Enum):
    """Status of claim substantiation per Green Claims Directive."""

    SUBSTANTIATED = "substantiated"
    PARTIALLY_SUBSTANTIATED = "partially_substantiated"
    UNSUBSTANTIATED = "unsubstantiated"
    UNDER_REVIEW = "under_review"
    NOT_ASSESSED = "not_assessed"

class GreenwashingRiskLevel(str, Enum):
    """Greenwashing risk classification."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

class GreenwashingCategory(str, Enum):
    """Category of greenwashing risk."""

    VAGUENESS = "vagueness"
    SELECTIVITY = "selectivity"
    FALSE_LABELS = "false_labels"
    IRRELEVANCE = "irrelevance"
    HIDDEN_TRADE_OFFS = "hidden_trade_offs"
    NO_PROOF = "no_proof"
    MISLEADING_COMPARISONS = "misleading_comparisons"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class GreenClaimsBridgeConfig(BaseModel):
    """Configuration for the Green Claims Bridge."""

    pack_id: str = Field(default="PACK-019")
    enable_provenance: bool = Field(default=True)
    vagueness_keyword_threshold: int = Field(
        default=2, ge=0,
        description="Max vague keywords before flagging",
    )
    require_third_party_verification: bool = Field(
        default=True,
        description="Require third-party verification per GCD Art 10",
    )

class EnvironmentalClaim(BaseModel):
    """An environmental or remediation claim to validate."""

    claim_id: str = Field(default_factory=_new_uuid)
    claim_type: ClaimType = Field(default=ClaimType.ENVIRONMENTAL)
    statement: str = Field(default="")
    scope: str = Field(default="", description="Product, company, or activity scope")
    evidence_provided: bool = Field(default=False)
    evidence_description: str = Field(default="")
    third_party_verified: bool = Field(default=False)
    verification_body: str = Field(default="")
    publication_date: Optional[datetime] = Field(None)
    is_comparative: bool = Field(default=False)

class ClaimValidationResult(BaseModel):
    """Result of validating a single claim."""

    claim_id: str = Field(default="")
    substantiation_status: SubstantiationStatus = Field(
        default=SubstantiationStatus.NOT_ASSESSED
    )
    greenwashing_risk: GreenwashingRiskLevel = Field(default=GreenwashingRiskLevel.NONE)
    greenwashing_categories: List[GreenwashingCategory] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    gcd_articles_triggered: List[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=100.0)

class GreenwashingAssessment(BaseModel):
    """Greenwashing risk assessment for a set of statements."""

    assessment_id: str = Field(default_factory=_new_uuid)
    total_statements: int = Field(default=0)
    flagged_statements: int = Field(default=0)
    risk_level: GreenwashingRiskLevel = Field(default=GreenwashingRiskLevel.NONE)
    categories_found: List[GreenwashingCategory] = Field(default_factory=list)
    flagged_details: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class BridgeResult(BaseModel):
    """Result of a Green Claims bridge operation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    claims_validated: int = Field(default=0)
    claims_substantiated: int = Field(default=0)
    claims_flagged: int = Field(default=0)
    validations: List[ClaimValidationResult] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class CSDDDGreenClaimsMapping(BaseModel):
    """Mapping between CSDDD data and Green Claims Directive requirements."""

    mapping_id: str = Field(default_factory=_new_uuid)
    csddd_article: str = Field(default="")
    csddd_requirement: str = Field(default="")
    gcd_article: str = Field(default="")
    gcd_requirement: str = Field(default="")
    overlap_description: str = Field(default="")
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Vague / Greenwashing Keyword Lists (deterministic detection)
# ---------------------------------------------------------------------------

VAGUE_KEYWORDS: List[str] = [
    "eco-friendly", "green", "sustainable", "natural",
    "environmentally friendly", "planet-friendly", "clean",
    "carbon neutral", "net zero", "climate positive",
    "responsibly sourced", "ethical", "conscious",
    "biodegradable", "recyclable", "eco",
]

MISLEADING_PATTERNS: Dict[str, GreenwashingCategory] = {
    "100% sustainable": GreenwashingCategory.VAGUENESS,
    "completely green": GreenwashingCategory.VAGUENESS,
    "zero impact": GreenwashingCategory.VAGUENESS,
    "no environmental footprint": GreenwashingCategory.VAGUENESS,
    "better than": GreenwashingCategory.MISLEADING_COMPARISONS,
    "more sustainable than": GreenwashingCategory.MISLEADING_COMPARISONS,
    "best in class": GreenwashingCategory.MISLEADING_COMPARISONS,
    "certified green": GreenwashingCategory.FALSE_LABELS,
    "eco-certified": GreenwashingCategory.FALSE_LABELS,
    "approved sustainable": GreenwashingCategory.FALSE_LABELS,
}

CSDDD_TO_GCD_MAPPING: List[Dict[str, str]] = [
    {
        "csddd_article": "Art_10",
        "csddd_requirement": "Remediation of actual adverse impacts",
        "gcd_article": "Art_3",
        "gcd_requirement": "Substantiation of environmental claims",
        "overlap": "Remediation claims must be substantiated with evidence",
    },
    {
        "csddd_article": "Art_14",
        "csddd_requirement": "Public communication and reporting",
        "gcd_article": "Art_5",
        "gcd_requirement": "Communication of environmental claims",
        "overlap": "Public sustainability communications must comply with GCD",
    },
    {
        "csddd_article": "Art_22",
        "csddd_requirement": "Climate transition plan",
        "gcd_article": "Art_3",
        "gcd_requirement": "Substantiation of environmental claims",
        "overlap": "Climate transition claims require scientific substantiation",
    },
    {
        "csddd_article": "Art_8",
        "csddd_requirement": "Preventing potential adverse impacts",
        "gcd_article": "Art_10",
        "gcd_requirement": "Third-party verification",
        "overlap": "Prevention effectiveness claims may require verification",
    },
]

# ---------------------------------------------------------------------------
# GreenClaimsBridge
# ---------------------------------------------------------------------------

class GreenClaimsBridge:
    """CSDDD remediation and Green Claims Directive cross-validation bridge.

    Cross-validates CSDDD remediation claims and environmental statements
    against the Green Claims Directive. Uses deterministic keyword detection
    and rule-based validation (zero-hallucination).

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = GreenClaimsBridge(GreenClaimsBridgeConfig())
        >>> claims = [{"statement": "We are 100% sustainable", ...}]
        >>> result = bridge.validate_remediation_claims(claims)
        >>> assert result.claims_flagged > 0
    """

    def __init__(self, config: Optional[GreenClaimsBridgeConfig] = None) -> None:
        """Initialize GreenClaimsBridge."""
        self.config = config or GreenClaimsBridgeConfig()
        logger.info("GreenClaimsBridge initialized (pack=%s)", self.config.pack_id)

    def validate_remediation_claims(
        self,
        claims: List[Dict[str, Any]],
    ) -> BridgeResult:
        """Validate CSDDD remediation and environmental claims.

        Args:
            claims: List of claim dicts with keys:
                statement, claim_type, evidence_provided, evidence_description,
                third_party_verified, verification_body, scope, is_comparative.

        Returns:
            BridgeResult with validation results for each claim.
        """
        result = BridgeResult(started_at=utcnow())

        try:
            validations: List[ClaimValidationResult] = []

            for claim_data in claims:
                claim = EnvironmentalClaim(
                    claim_id=claim_data.get("claim_id", _new_uuid()),
                    claim_type=ClaimType(claim_data.get("claim_type", "environmental")),
                    statement=claim_data.get("statement", ""),
                    scope=claim_data.get("scope", ""),
                    evidence_provided=claim_data.get("evidence_provided", False),
                    evidence_description=claim_data.get("evidence_description", ""),
                    third_party_verified=claim_data.get("third_party_verified", False),
                    verification_body=claim_data.get("verification_body", ""),
                    is_comparative=claim_data.get("is_comparative", False),
                )
                validation = self._validate_single_claim(claim)
                validations.append(validation)

            result.validations = validations
            result.claims_validated = len(validations)
            result.claims_substantiated = sum(
                1 for v in validations
                if v.substantiation_status == SubstantiationStatus.SUBSTANTIATED
            )
            result.claims_flagged = sum(
                1 for v in validations
                if v.greenwashing_risk in (
                    GreenwashingRiskLevel.HIGH,
                    GreenwashingRiskLevel.CRITICAL,
                )
            )
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result)

            logger.info(
                "Validated %d claims: %d substantiated, %d flagged",
                result.claims_validated,
                result.claims_substantiated,
                result.claims_flagged,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Claim validation failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def check_greenwashing_risk(
        self,
        statements: List[str],
    ) -> GreenwashingAssessment:
        """Check a list of statements for greenwashing risk.

        Uses deterministic keyword matching (zero-hallucination).

        Args:
            statements: List of environmental/sustainability statement strings.

        Returns:
            GreenwashingAssessment with flagged statements and risk level.
        """
        flagged_details: List[Dict[str, Any]] = []
        categories_found: set = set()

        for stmt in statements:
            stmt_lower = stmt.lower()
            found_issues: List[Dict[str, str]] = []

            # Check vague keywords
            vague_count = sum(
                1 for kw in VAGUE_KEYWORDS if kw in stmt_lower
            )
            if vague_count > self.config.vagueness_keyword_threshold:
                found_issues.append({
                    "category": GreenwashingCategory.VAGUENESS.value,
                    "detail": f"Contains {vague_count} vague environmental keywords",
                })
                categories_found.add(GreenwashingCategory.VAGUENESS)

            # Check misleading patterns
            for pattern, category in MISLEADING_PATTERNS.items():
                if pattern.lower() in stmt_lower:
                    found_issues.append({
                        "category": category.value,
                        "detail": f"Misleading pattern detected: '{pattern}'",
                    })
                    categories_found.add(category)

            if found_issues:
                flagged_details.append({
                    "statement": stmt,
                    "issues": found_issues,
                })

        # Determine overall risk level
        if len(flagged_details) == 0:
            risk_level = GreenwashingRiskLevel.NONE
        elif len(flagged_details) == 1:
            risk_level = GreenwashingRiskLevel.LOW
        elif len(flagged_details) <= len(statements) // 2:
            risk_level = GreenwashingRiskLevel.MEDIUM
        elif len(flagged_details) < len(statements):
            risk_level = GreenwashingRiskLevel.HIGH
        else:
            risk_level = GreenwashingRiskLevel.CRITICAL

        recommendations = self._generate_greenwashing_recommendations(
            list(categories_found)
        )

        assessment = GreenwashingAssessment(
            total_statements=len(statements),
            flagged_statements=len(flagged_details),
            risk_level=risk_level,
            categories_found=list(categories_found),
            flagged_details=flagged_details,
            recommendations=recommendations,
        )
        assessment.provenance_hash = _compute_hash(assessment)

        logger.info(
            "Greenwashing check: %d/%d flagged (risk=%s)",
            len(flagged_details),
            len(statements),
            risk_level.value,
        )
        return assessment

    def get_substantiation_status(
        self,
        claim: Dict[str, Any],
    ) -> ClaimValidationResult:
        """Get substantiation status for a single claim.

        Args:
            claim: Claim dict with statement, evidence, and verification data.

        Returns:
            ClaimValidationResult with substantiation and risk assessment.
        """
        env_claim = EnvironmentalClaim(
            claim_id=claim.get("claim_id", _new_uuid()),
            claim_type=ClaimType(claim.get("claim_type", "environmental")),
            statement=claim.get("statement", ""),
            scope=claim.get("scope", ""),
            evidence_provided=claim.get("evidence_provided", False),
            evidence_description=claim.get("evidence_description", ""),
            third_party_verified=claim.get("third_party_verified", False),
            verification_body=claim.get("verification_body", ""),
            is_comparative=claim.get("is_comparative", False),
        )
        return self._validate_single_claim(env_claim)

    def map_csddd_to_green_claims(
        self,
        csddd_data: Optional[Dict[str, Any]] = None,
    ) -> List[CSDDDGreenClaimsMapping]:
        """Map CSDDD requirements to Green Claims Directive obligations.

        Args:
            csddd_data: Optional CSDDD data for contextual mapping.

        Returns:
            List of CSDDDGreenClaimsMapping objects.
        """
        mappings: List[CSDDDGreenClaimsMapping] = []

        for m in CSDDD_TO_GCD_MAPPING:
            mapping = CSDDDGreenClaimsMapping(
                csddd_article=m["csddd_article"],
                csddd_requirement=m["csddd_requirement"],
                gcd_article=m["gcd_article"],
                gcd_requirement=m["gcd_requirement"],
                overlap_description=m["overlap"],
            )
            mapping.provenance_hash = _compute_hash(mapping)
            mappings.append(mapping)

        logger.info("CSDDD-to-GCD mapping: %d linkages", len(mappings))
        return mappings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_single_claim(self, claim: EnvironmentalClaim) -> ClaimValidationResult:
        """Validate a single environmental or remediation claim."""
        issues: List[str] = []
        recommendations: List[str] = []
        gcd_articles: List[str] = []
        greenwashing_cats: List[GreenwashingCategory] = []
        score = 100.0

        stmt_lower = claim.statement.lower()

        # Check 1: Evidence provided (GCD Art 3)
        if not claim.evidence_provided:
            issues.append("No evidence provided to substantiate claim (GCD Art 3)")
            recommendations.append("Provide scientific evidence or lifecycle data")
            gcd_articles.append("Art_3")
            score -= 30.0
            greenwashing_cats.append(GreenwashingCategory.NO_PROOF)

        # Check 2: Third-party verification (GCD Art 10)
        if self.config.require_third_party_verification and not claim.third_party_verified:
            issues.append("Claim not verified by third-party body (GCD Art 10)")
            recommendations.append("Engage accredited conformity assessment body")
            gcd_articles.append("Art_10")
            score -= 20.0

        # Check 3: Vague keywords
        vague_count = sum(1 for kw in VAGUE_KEYWORDS if kw in stmt_lower)
        if vague_count > self.config.vagueness_keyword_threshold:
            issues.append(
                f"Claim contains {vague_count} vague terms (GCD Art 5)"
            )
            recommendations.append("Replace vague terms with specific, measurable claims")
            gcd_articles.append("Art_5")
            score -= 15.0
            greenwashing_cats.append(GreenwashingCategory.VAGUENESS)

        # Check 4: Misleading patterns
        for pattern, cat in MISLEADING_PATTERNS.items():
            if pattern.lower() in stmt_lower:
                issues.append(f"Potentially misleading pattern: '{pattern}'")
                score -= 10.0
                greenwashing_cats.append(cat)

        # Check 5: Comparative claims require comparison data
        if claim.is_comparative and not claim.evidence_provided:
            issues.append(
                "Comparative claim requires comparison methodology (GCD Art 3)"
            )
            gcd_articles.append("Art_3")
            score -= 15.0
            greenwashing_cats.append(GreenwashingCategory.MISLEADING_COMPARISONS)

        score = max(score, 0.0)

        # Determine substantiation status
        if score >= 80.0:
            substantiation = SubstantiationStatus.SUBSTANTIATED
        elif score >= 50.0:
            substantiation = SubstantiationStatus.PARTIALLY_SUBSTANTIATED
        else:
            substantiation = SubstantiationStatus.UNSUBSTANTIATED

        # Determine greenwashing risk
        if not greenwashing_cats:
            gw_risk = GreenwashingRiskLevel.NONE
        elif len(greenwashing_cats) == 1:
            gw_risk = GreenwashingRiskLevel.LOW
        elif len(greenwashing_cats) <= 3:
            gw_risk = GreenwashingRiskLevel.MEDIUM
        else:
            gw_risk = GreenwashingRiskLevel.HIGH

        return ClaimValidationResult(
            claim_id=claim.claim_id,
            substantiation_status=substantiation,
            greenwashing_risk=gw_risk,
            greenwashing_categories=list(set(greenwashing_cats)),
            issues=issues,
            recommendations=recommendations,
            gcd_articles_triggered=list(set(gcd_articles)),
            score=round(score, 1),
        )

    def _generate_greenwashing_recommendations(
        self,
        categories: List[GreenwashingCategory],
    ) -> List[str]:
        """Generate recommendations based on greenwashing categories found."""
        recs: List[str] = []
        category_recs = {
            GreenwashingCategory.VAGUENESS: (
                "Replace vague environmental terms with specific, measurable, "
                "and verifiable claims (GCD Art 5)"
            ),
            GreenwashingCategory.SELECTIVITY: (
                "Ensure environmental claims cover the full lifecycle "
                "and do not selectively omit negative impacts"
            ),
            GreenwashingCategory.FALSE_LABELS: (
                "Remove unofficial or misleading sustainability labels; "
                "use only recognised EU Ecolabel or accredited certifications"
            ),
            GreenwashingCategory.IRRELEVANCE: (
                "Ensure claims are relevant to the product or activity context"
            ),
            GreenwashingCategory.HIDDEN_TRADE_OFFS: (
                "Disclose all environmental trade-offs associated with "
                "the claimed benefit"
            ),
            GreenwashingCategory.NO_PROOF: (
                "Provide scientific evidence, LCA data, or recognised "
                "methodology to support all environmental claims (GCD Art 3)"
            ),
            GreenwashingCategory.MISLEADING_COMPARISONS: (
                "Base comparative claims on equivalent scope, methodology, "
                "and recent data; disclose comparison parameters (GCD Art 3)"
            ),
        }
        for cat in categories:
            rec = category_recs.get(cat)
            if rec and rec not in recs:
                recs.append(rec)
        return recs
