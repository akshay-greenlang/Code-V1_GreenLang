# -*- coding: utf-8 -*-
"""
Risk Scoring Engine - SEC-010 Phase 2

Calculates risk scores for threats using likelihood, impact, and
CVSS 3.1 methodology. Provides threat prioritization and severity
classification.

Example:
    >>> from greenlang.infrastructure.threat_modeling import RiskScorer, Threat
    >>> scorer = RiskScorer()
    >>> likelihood = scorer.calculate_likelihood(threat)
    >>> impact = scorer.calculate_impact(threat)
    >>> risk_score = scorer.calculate_risk_score(threat, likelihood, impact)
    >>> prioritized = scorer.prioritize_threats(threats)

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.threat_modeling.models import (
    Threat,
    ThreatCategory,
    ThreatStatus,
)
from greenlang.infrastructure.threat_modeling.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk Factor Definitions
# ---------------------------------------------------------------------------


class LikelihoodFactor(str, Enum):
    """Factors affecting threat likelihood."""

    SKILL_LEVEL = "skill_level"
    """Attacker skill level required."""

    MOTIVE = "motive"
    """Attacker motivation."""

    OPPORTUNITY = "opportunity"
    """Opportunity to exploit."""

    SIZE = "size"
    """Size of attacker population."""

    EASE_OF_DISCOVERY = "ease_of_discovery"
    """How easy to discover the vulnerability."""

    EASE_OF_EXPLOIT = "ease_of_exploit"
    """How easy to exploit once discovered."""

    AWARENESS = "awareness"
    """How well known is the vulnerability."""

    INTRUSION_DETECTION = "intrusion_detection"
    """Likelihood of detection."""


class ImpactFactor(str, Enum):
    """Factors affecting threat impact."""

    CONFIDENTIALITY = "confidentiality"
    """Impact on confidentiality."""

    INTEGRITY = "integrity"
    """Impact on integrity."""

    AVAILABILITY = "availability"
    """Impact on availability."""

    FINANCIAL = "financial"
    """Financial impact."""

    REPUTATION = "reputation"
    """Reputational impact."""

    COMPLIANCE = "compliance"
    """Regulatory compliance impact."""

    PRIVACY = "privacy"
    """Privacy impact (GDPR, etc.)."""


# ---------------------------------------------------------------------------
# CVSS 3.1 Constants
# ---------------------------------------------------------------------------


class CVSSAttackVector(str, Enum):
    """CVSS 3.1 Attack Vector."""

    NETWORK = "N"  # 0.85
    ADJACENT = "A"  # 0.62
    LOCAL = "L"  # 0.55
    PHYSICAL = "P"  # 0.2


class CVSSAttackComplexity(str, Enum):
    """CVSS 3.1 Attack Complexity."""

    LOW = "L"  # 0.77
    HIGH = "H"  # 0.44


class CVSSPrivilegesRequired(str, Enum):
    """CVSS 3.1 Privileges Required."""

    NONE = "N"  # 0.85
    LOW = "L"  # 0.62 (0.68 if scope changed)
    HIGH = "H"  # 0.27 (0.50 if scope changed)


class CVSSUserInteraction(str, Enum):
    """CVSS 3.1 User Interaction."""

    NONE = "N"  # 0.85
    REQUIRED = "R"  # 0.62


class CVSSScope(str, Enum):
    """CVSS 3.1 Scope."""

    UNCHANGED = "U"
    CHANGED = "C"


class CVSSImpact(str, Enum):
    """CVSS 3.1 Impact values for C/I/A."""

    HIGH = "H"  # 0.56
    LOW = "L"  # 0.22
    NONE = "N"  # 0


@dataclass
class CVSSVector:
    """CVSS 3.1 vector components."""

    attack_vector: CVSSAttackVector = CVSSAttackVector.NETWORK
    attack_complexity: CVSSAttackComplexity = CVSSAttackComplexity.LOW
    privileges_required: CVSSPrivilegesRequired = CVSSPrivilegesRequired.NONE
    user_interaction: CVSSUserInteraction = CVSSUserInteraction.NONE
    scope: CVSSScope = CVSSScope.UNCHANGED
    confidentiality: CVSSImpact = CVSSImpact.HIGH
    integrity: CVSSImpact = CVSSImpact.HIGH
    availability: CVSSImpact = CVSSImpact.HIGH

    def to_string(self) -> str:
        """Convert to CVSS 3.1 vector string."""
        return (
            f"CVSS:3.1/AV:{self.attack_vector.value}/AC:{self.attack_complexity.value}"
            f"/PR:{self.privileges_required.value}/UI:{self.user_interaction.value}"
            f"/S:{self.scope.value}/C:{self.confidentiality.value}"
            f"/I:{self.integrity.value}/A:{self.availability.value}"
        )


@dataclass
class RiskAssessment:
    """Complete risk assessment for a threat."""

    threat_id: str
    likelihood: int
    impact: int
    risk_score: float
    severity: str
    cvss_score: Optional[float]
    cvss_vector: Optional[str]
    likelihood_factors: Dict[str, int]
    impact_factors: Dict[str, int]
    business_impact: float
    assessed_at: datetime


# ---------------------------------------------------------------------------
# Risk Scorer Implementation
# ---------------------------------------------------------------------------


class RiskScorer:
    """Risk scoring engine for threats.

    Calculates risk scores based on likelihood, impact, and CVSS 3.1
    methodology. Provides threat prioritization and severity classification.

    Attributes:
        config: Threat modeling configuration.

    Example:
        >>> scorer = RiskScorer()
        >>> score = scorer.calculate_risk_score(threat, 4, 5)
        >>> cvss = scorer.calculate_cvss(threat)
    """

    # CVSS 3.1 metric values
    CVSS_AV_VALUES = {
        CVSSAttackVector.NETWORK: 0.85,
        CVSSAttackVector.ADJACENT: 0.62,
        CVSSAttackVector.LOCAL: 0.55,
        CVSSAttackVector.PHYSICAL: 0.20,
    }

    CVSS_AC_VALUES = {
        CVSSAttackComplexity.LOW: 0.77,
        CVSSAttackComplexity.HIGH: 0.44,
    }

    CVSS_PR_VALUES_UNCHANGED = {
        CVSSPrivilegesRequired.NONE: 0.85,
        CVSSPrivilegesRequired.LOW: 0.62,
        CVSSPrivilegesRequired.HIGH: 0.27,
    }

    CVSS_PR_VALUES_CHANGED = {
        CVSSPrivilegesRequired.NONE: 0.85,
        CVSSPrivilegesRequired.LOW: 0.68,
        CVSSPrivilegesRequired.HIGH: 0.50,
    }

    CVSS_UI_VALUES = {
        CVSSUserInteraction.NONE: 0.85,
        CVSSUserInteraction.REQUIRED: 0.62,
    }

    CVSS_IMPACT_VALUES = {
        CVSSImpact.HIGH: 0.56,
        CVSSImpact.LOW: 0.22,
        CVSSImpact.NONE: 0.0,
    }

    # Default likelihood factors by category
    CATEGORY_LIKELIHOOD_DEFAULTS: Dict[str, Dict[str, int]] = {
        "S": {"skill_level": 3, "opportunity": 4, "ease_of_exploit": 3},
        "T": {"skill_level": 3, "opportunity": 3, "ease_of_exploit": 3},
        "R": {"skill_level": 2, "opportunity": 4, "ease_of_exploit": 4},
        "I": {"skill_level": 2, "opportunity": 3, "ease_of_exploit": 3},
        "D": {"skill_level": 2, "opportunity": 5, "ease_of_exploit": 4},
        "E": {"skill_level": 4, "opportunity": 2, "ease_of_exploit": 2},
    }

    # Default impact factors by category
    CATEGORY_IMPACT_DEFAULTS: Dict[str, Dict[str, int]] = {
        "S": {"confidentiality": 4, "integrity": 3, "financial": 3},
        "T": {"confidentiality": 2, "integrity": 5, "financial": 4},
        "R": {"confidentiality": 1, "integrity": 2, "compliance": 4},
        "I": {"confidentiality": 5, "privacy": 5, "compliance": 4},
        "D": {"availability": 5, "financial": 4, "reputation": 3},
        "E": {"confidentiality": 4, "integrity": 4, "compliance": 4},
    }

    def __init__(self) -> None:
        """Initialize the risk scorer."""
        self.config = get_config()
        logger.debug("Risk scorer initialized")

    def calculate_likelihood(
        self,
        threat: Threat,
        factors: Optional[Dict[str, int]] = None,
    ) -> int:
        """Calculate likelihood score for a threat.

        Considers attacker skill, opportunity, resources, and ease of exploit.

        Args:
            threat: The threat to score.
            factors: Optional custom likelihood factors (each 1-5).

        Returns:
            Likelihood score (1-5).
        """
        # Get default factors for threat category
        category_factors = self.CATEGORY_LIKELIHOOD_DEFAULTS.get(
            threat.category.value, {}
        )

        # Merge with provided factors
        if factors:
            category_factors.update(factors)

        if not category_factors:
            return threat.likelihood

        # Calculate average of factors
        total = sum(category_factors.values())
        count = len(category_factors)
        avg_likelihood = total / count

        # Round to integer 1-5
        return max(1, min(5, round(avg_likelihood)))

    def calculate_impact(
        self,
        threat: Threat,
        factors: Optional[Dict[str, int]] = None,
    ) -> int:
        """Calculate impact score for a threat.

        Considers confidentiality, integrity, availability, financial,
        and compliance impacts.

        Args:
            threat: The threat to score.
            factors: Optional custom impact factors (each 1-5).

        Returns:
            Impact score (1-5).
        """
        # Get default factors for threat category
        category_factors = self.CATEGORY_IMPACT_DEFAULTS.get(
            threat.category.value, {}
        )

        # Merge with provided factors
        if factors:
            category_factors.update(factors)

        if not category_factors:
            return threat.impact

        # Calculate weighted average (CIA triad weighted higher)
        weights = {
            "confidentiality": 1.5,
            "integrity": 1.5,
            "availability": 1.5,
            "financial": 1.0,
            "reputation": 0.8,
            "compliance": 1.2,
            "privacy": 1.3,
        }

        weighted_sum = sum(
            category_factors.get(factor, 3) * weights.get(factor, 1.0)
            for factor in category_factors
        )
        total_weight = sum(
            weights.get(factor, 1.0) for factor in category_factors
        )

        avg_impact = weighted_sum / total_weight if total_weight > 0 else 3

        return max(1, min(5, round(avg_impact)))

    def calculate_risk_score(
        self,
        threat: Threat,
        likelihood: Optional[int] = None,
        impact: Optional[int] = None,
        business_context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate composite risk score for a threat.

        Combines likelihood, impact, and business context into a
        single risk score on a 0-10 scale.

        Args:
            threat: The threat to score.
            likelihood: Override likelihood score (1-5).
            impact: Override impact score (1-5).
            business_context: Optional business context factors.

        Returns:
            Risk score (0.0-10.0).
        """
        # Use provided or threat's values
        _likelihood = likelihood if likelihood is not None else threat.likelihood
        _impact = impact if impact is not None else threat.impact

        # Get STRIDE category weight
        category_weight = self.config.stride_weights.get(threat.category.value, 1.0)

        # Base risk calculation: likelihood * impact / 2.5 (to scale to 10)
        base_risk = (_likelihood * _impact) / 2.5

        # Apply category weight
        weighted_risk = base_risk * category_weight

        # Apply business context adjustments
        context_multiplier = 1.0
        if business_context:
            # High-value asset
            if business_context.get("is_critical_asset"):
                context_multiplier *= 1.3
            # Customer-facing
            if business_context.get("customer_facing"):
                context_multiplier *= 1.2
            # Regulated data
            if business_context.get("regulated_data"):
                context_multiplier *= 1.25
            # External exposure
            if business_context.get("internet_exposed"):
                context_multiplier *= 1.15

        final_score = weighted_risk * context_multiplier

        logger.debug(
            "Risk score calculated: threat=%s, likelihood=%d, impact=%d, score=%.2f",
            threat.title[:30],
            _likelihood,
            _impact,
            final_score,
        )

        return min(10.0, max(0.0, final_score))

    def prioritize_threats(
        self,
        threats: List[Threat],
        recalculate_scores: bool = True,
    ) -> List[Threat]:
        """Sort threats by risk score descending.

        Prioritizes threats for remediation based on risk score.

        Args:
            threats: List of threats to prioritize.
            recalculate_scores: Whether to recalculate risk scores.

        Returns:
            List of threats sorted by risk score (highest first).
        """
        if recalculate_scores:
            for threat in threats:
                likelihood = self.calculate_likelihood(threat)
                impact = self.calculate_impact(threat)
                threat.risk_score = self.calculate_risk_score(threat, likelihood, impact)
                threat.severity = self.config.get_severity_for_score(threat.risk_score)

        # Sort by risk score descending
        sorted_threats = sorted(threats, key=lambda t: t.risk_score, reverse=True)

        logger.info(
            "Prioritized %d threats, top risk score: %.2f",
            len(sorted_threats),
            sorted_threats[0].risk_score if sorted_threats else 0,
        )

        return sorted_threats

    def calculate_cvss(
        self,
        threat: Threat,
        vector: Optional[CVSSVector] = None,
    ) -> float:
        """Calculate CVSS 3.1 base score for a threat.

        Implements the CVSS 3.1 base score calculation algorithm.

        Args:
            threat: The threat to score.
            vector: Optional CVSS vector (will be inferred if not provided).

        Returns:
            CVSS 3.1 base score (0.0-10.0).
        """
        # Use provided vector or infer from threat
        if vector is None:
            vector = self._infer_cvss_vector(threat)

        # Get metric values
        av = self.CVSS_AV_VALUES[vector.attack_vector]
        ac = self.CVSS_AC_VALUES[vector.attack_complexity]

        # Privileges required depends on scope
        if vector.scope == CVSSScope.CHANGED:
            pr = self.CVSS_PR_VALUES_CHANGED[vector.privileges_required]
        else:
            pr = self.CVSS_PR_VALUES_UNCHANGED[vector.privileges_required]

        ui = self.CVSS_UI_VALUES[vector.user_interaction]

        # Impact values
        c = self.CVSS_IMPACT_VALUES[vector.confidentiality]
        i = self.CVSS_IMPACT_VALUES[vector.integrity]
        a = self.CVSS_IMPACT_VALUES[vector.availability]

        # Calculate Impact Sub Score (ISS)
        iss = 1 - ((1 - c) * (1 - i) * (1 - a))

        # Calculate Impact
        if vector.scope == CVSSScope.UNCHANGED:
            impact = 6.42 * iss
        else:
            impact = 7.52 * (iss - 0.029) - 3.25 * pow(iss - 0.02, 15)

        # Calculate Exploitability
        exploitability = 8.22 * av * ac * pr * ui

        # Calculate Base Score
        if impact <= 0:
            base_score = 0.0
        else:
            if vector.scope == CVSSScope.UNCHANGED:
                base_score = min(impact + exploitability, 10)
            else:
                base_score = min(1.08 * (impact + exploitability), 10)

        # Round up to 1 decimal place
        base_score = math.ceil(base_score * 10) / 10

        logger.debug(
            "CVSS calculated: threat=%s, score=%.1f, vector=%s",
            threat.title[:30],
            base_score,
            vector.to_string(),
        )

        return base_score

    def assess_threat(
        self,
        threat: Threat,
        likelihood_factors: Optional[Dict[str, int]] = None,
        impact_factors: Optional[Dict[str, int]] = None,
        business_context: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessment:
        """Perform complete risk assessment for a threat.

        Calculates all risk metrics and returns a comprehensive assessment.

        Args:
            threat: The threat to assess.
            likelihood_factors: Custom likelihood factors.
            impact_factors: Custom impact factors.
            business_context: Business context factors.

        Returns:
            Complete RiskAssessment.
        """
        # Calculate likelihood
        likelihood = self.calculate_likelihood(threat, likelihood_factors)

        # Calculate impact
        impact = self.calculate_impact(threat, impact_factors)

        # Calculate risk score
        risk_score = self.calculate_risk_score(threat, likelihood, impact, business_context)

        # Determine severity
        severity = self.config.get_severity_for_score(risk_score)

        # Calculate CVSS
        cvss_vector = self._infer_cvss_vector(threat)
        cvss_score = self.calculate_cvss(threat, cvss_vector)

        # Calculate business impact score
        business_impact = self._calculate_business_impact(threat, business_context)

        # Build factor dictionaries
        _likelihood_factors = self.CATEGORY_LIKELIHOOD_DEFAULTS.get(
            threat.category.value, {}
        ).copy()
        if likelihood_factors:
            _likelihood_factors.update(likelihood_factors)

        _impact_factors = self.CATEGORY_IMPACT_DEFAULTS.get(
            threat.category.value, {}
        ).copy()
        if impact_factors:
            _impact_factors.update(impact_factors)

        assessment = RiskAssessment(
            threat_id=threat.id,
            likelihood=likelihood,
            impact=impact,
            risk_score=risk_score,
            severity=severity,
            cvss_score=cvss_score,
            cvss_vector=cvss_vector.to_string(),
            likelihood_factors=_likelihood_factors,
            impact_factors=_impact_factors,
            business_impact=business_impact,
            assessed_at=datetime.now(timezone.utc),
        )

        return assessment

    def get_severity_distribution(self, threats: List[Threat]) -> Dict[str, int]:
        """Get distribution of threats by severity.

        Args:
            threats: List of threats.

        Returns:
            Dictionary mapping severity to count.
        """
        distribution = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for threat in threats:
            severity = threat.severity.lower()
            if severity in distribution:
                distribution[severity] += 1

        return distribution

    def get_category_risk_summary(self, threats: List[Threat]) -> Dict[str, float]:
        """Get average risk score by STRIDE category.

        Args:
            threats: List of threats.

        Returns:
            Dictionary mapping category to average risk score.
        """
        category_scores: Dict[str, List[float]] = {
            cat.value: [] for cat in ThreatCategory
        }

        for threat in threats:
            category_scores[threat.category.value].append(threat.risk_score)

        return {
            cat: (sum(scores) / len(scores) if scores else 0.0)
            for cat, scores in category_scores.items()
        }

    def _infer_cvss_vector(self, threat: Threat) -> CVSSVector:
        """Infer CVSS vector from threat properties.

        Args:
            threat: The threat to analyze.

        Returns:
            Inferred CVSSVector.
        """
        vector = CVSSVector()

        # Infer attack vector from threat description/category
        description_lower = threat.description.lower()
        if any(word in description_lower for word in ["remote", "network", "internet", "api"]):
            vector.attack_vector = CVSSAttackVector.NETWORK
        elif any(word in description_lower for word in ["local", "file", "process"]):
            vector.attack_vector = CVSSAttackVector.LOCAL
        elif any(word in description_lower for word in ["physical", "hardware"]):
            vector.attack_vector = CVSSAttackVector.PHYSICAL
        else:
            vector.attack_vector = CVSSAttackVector.NETWORK

        # Infer complexity
        if threat.likelihood >= 4:
            vector.attack_complexity = CVSSAttackComplexity.LOW
        else:
            vector.attack_complexity = CVSSAttackComplexity.HIGH

        # Infer privileges required
        if "unauthenticated" in description_lower or "anonymous" in description_lower:
            vector.privileges_required = CVSSPrivilegesRequired.NONE
        elif "admin" in description_lower or "elevated" in description_lower:
            vector.privileges_required = CVSSPrivilegesRequired.HIGH
        else:
            vector.privileges_required = CVSSPrivilegesRequired.LOW

        # Infer user interaction
        if any(word in description_lower for word in ["click", "user", "social", "phishing"]):
            vector.user_interaction = CVSSUserInteraction.REQUIRED
        else:
            vector.user_interaction = CVSSUserInteraction.NONE

        # Infer scope
        if any(word in description_lower for word in ["escape", "cross", "lateral", "propagate"]):
            vector.scope = CVSSScope.CHANGED
        else:
            vector.scope = CVSSScope.UNCHANGED

        # Infer CIA impacts from category
        category_cia = {
            ThreatCategory.SPOOFING: (CVSSImpact.HIGH, CVSSImpact.LOW, CVSSImpact.NONE),
            ThreatCategory.TAMPERING: (CVSSImpact.LOW, CVSSImpact.HIGH, CVSSImpact.NONE),
            ThreatCategory.REPUDIATION: (CVSSImpact.NONE, CVSSImpact.LOW, CVSSImpact.NONE),
            ThreatCategory.INFORMATION_DISCLOSURE: (CVSSImpact.HIGH, CVSSImpact.NONE, CVSSImpact.NONE),
            ThreatCategory.DENIAL_OF_SERVICE: (CVSSImpact.NONE, CVSSImpact.NONE, CVSSImpact.HIGH),
            ThreatCategory.ELEVATION_OF_PRIVILEGE: (CVSSImpact.HIGH, CVSSImpact.HIGH, CVSSImpact.LOW),
        }

        c, i, a = category_cia.get(
            threat.category,
            (CVSSImpact.HIGH, CVSSImpact.HIGH, CVSSImpact.HIGH),
        )
        vector.confidentiality = c
        vector.integrity = i
        vector.availability = a

        return vector

    def _calculate_business_impact(
        self,
        threat: Threat,
        business_context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate business impact score.

        Args:
            threat: The threat to assess.
            business_context: Business context factors.

        Returns:
            Business impact score (0.0-10.0).
        """
        base_impact = threat.impact * 2  # Scale 1-5 to 2-10

        if not business_context:
            return float(base_impact)

        multiplier = 1.0

        # Critical asset increases impact
        if business_context.get("is_critical_asset"):
            multiplier *= 1.5

        # Customer-facing increases impact
        if business_context.get("customer_facing"):
            multiplier *= 1.3

        # Regulated data increases impact
        if business_context.get("regulated_data"):
            multiplier *= 1.4

        # Revenue-generating increases impact
        if business_context.get("revenue_generating"):
            multiplier *= 1.2

        return min(10.0, base_impact * multiplier)


__all__ = [
    "RiskScorer",
    "RiskAssessment",
    "CVSSVector",
    "CVSSAttackVector",
    "CVSSAttackComplexity",
    "CVSSPrivilegesRequired",
    "CVSSUserInteraction",
    "CVSSScope",
    "CVSSImpact",
    "LikelihoodFactor",
    "ImpactFactor",
]
