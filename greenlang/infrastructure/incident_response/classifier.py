# -*- coding: utf-8 -*-
"""
Incident Classifier - SEC-010

Classifies incidents by severity (P0-P3) based on type, scope, and business
impact. Determines appropriate escalation policies and SLA requirements.

Example:
    >>> from greenlang.infrastructure.incident_response.classifier import (
    ...     IncidentClassifier,
    ... )
    >>> classifier = IncidentClassifier(config)
    >>> severity = classifier.classify(incident)
    >>> impact = classifier.calculate_business_impact(incident)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from greenlang.infrastructure.incident_response.config import (
    IncidentResponseConfig,
    get_config,
)
from greenlang.infrastructure.incident_response.models import (
    Incident,
    IncidentType,
    EscalationLevel,
    Alert,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity Level Definitions
# ---------------------------------------------------------------------------


@dataclass
class SeverityLevel:
    """Definition of a severity level.

    Attributes:
        level: Escalation level (P0-P3).
        name: Human-readable name.
        description: Level description.
        response_time_minutes: Required response time.
        resolution_time_minutes: Target resolution time.
        escalation_policy: When to escalate.
        notification_channels: Required notification channels.
        requires_incident_commander: Whether IC is required.
    """

    level: EscalationLevel
    name: str
    description: str
    response_time_minutes: int
    resolution_time_minutes: int
    escalation_policy: str
    notification_channels: List[str]
    requires_incident_commander: bool = False


SEVERITY_LEVELS: Dict[EscalationLevel, SeverityLevel] = {
    EscalationLevel.P0: SeverityLevel(
        level=EscalationLevel.P0,
        name="Critical",
        description="Production down, active attack, data breach",
        response_time_minutes=5,
        resolution_time_minutes=60,
        escalation_policy="immediate",
        notification_channels=["pagerduty", "slack", "sms", "email"],
        requires_incident_commander=True,
    ),
    EscalationLevel.P1: SeverityLevel(
        level=EscalationLevel.P1,
        name="High",
        description="Major degradation, potential data exposure",
        response_time_minutes=15,
        resolution_time_minutes=240,
        escalation_policy="1h",
        notification_channels=["pagerduty", "slack", "email"],
        requires_incident_commander=True,
    ),
    EscalationLevel.P2: SeverityLevel(
        level=EscalationLevel.P2,
        name="Medium",
        description="Limited impact, isolated issue",
        response_time_minutes=60,
        resolution_time_minutes=1440,
        escalation_policy="4h",
        notification_channels=["slack", "email"],
        requires_incident_commander=False,
    ),
    EscalationLevel.P3: SeverityLevel(
        level=EscalationLevel.P3,
        name="Low",
        description="Minor issue, informational",
        response_time_minutes=240,
        resolution_time_minutes=10080,
        escalation_policy="24h",
        notification_channels=["email"],
        requires_incident_commander=False,
    ),
}


# ---------------------------------------------------------------------------
# Incident Type Severity Mapping
# ---------------------------------------------------------------------------

# Base severity for incident types (can be elevated based on scope/impact)
INCIDENT_TYPE_BASE_SEVERITY: Dict[IncidentType, EscalationLevel] = {
    IncidentType.DATA_BREACH: EscalationLevel.P0,
    IncidentType.RANSOMWARE: EscalationLevel.P0,
    IncidentType.DDOS_ATTACK: EscalationLevel.P0,
    IncidentType.CREDENTIAL_COMPROMISE: EscalationLevel.P0,
    IncidentType.MALWARE: EscalationLevel.P1,
    IncidentType.UNAUTHORIZED_ACCESS: EscalationLevel.P1,
    IncidentType.DATA_EXFILTRATION: EscalationLevel.P1,
    IncidentType.PRIVILEGE_ESCALATION: EscalationLevel.P1,
    IncidentType.SESSION_HIJACK: EscalationLevel.P1,
    IncidentType.INSIDER_THREAT: EscalationLevel.P1,
    IncidentType.SQL_INJECTION: EscalationLevel.P1,
    IncidentType.SUPPLY_CHAIN: EscalationLevel.P1,
    IncidentType.XSS_ATTACK: EscalationLevel.P2,
    IncidentType.BRUTE_FORCE: EscalationLevel.P2,
    IncidentType.API_ABUSE: EscalationLevel.P2,
    IncidentType.PHISHING: EscalationLevel.P2,
    IncidentType.CONFIGURATION_DRIFT: EscalationLevel.P2,
    IncidentType.COMPLIANCE_VIOLATION: EscalationLevel.P2,
    IncidentType.AVAILABILITY: EscalationLevel.P2,
    IncidentType.UNKNOWN: EscalationLevel.P3,
}


# ---------------------------------------------------------------------------
# Business Impact Scoring
# ---------------------------------------------------------------------------


@dataclass
class BusinessImpact:
    """Business impact assessment.

    Attributes:
        score: Overall impact score (0-100).
        revenue_impact: Estimated revenue impact.
        users_affected: Number of affected users.
        data_classification: Highest data classification involved.
        regulatory_impact: Regulatory/compliance impact.
        reputation_impact: Reputation/brand impact.
        operational_impact: Operational impact level.
        factors: Individual impact factors.
    """

    score: float
    revenue_impact: str = "none"
    users_affected: int = 0
    data_classification: str = "public"
    regulatory_impact: str = "none"
    reputation_impact: str = "low"
    operational_impact: str = "low"
    factors: Dict[str, float] = field(default_factory=dict)


# Impact factor weights
IMPACT_WEIGHTS: Dict[str, float] = {
    "data_classification": 25.0,  # Highest weight for data sensitivity
    "users_affected": 20.0,
    "regulatory_impact": 20.0,
    "revenue_impact": 15.0,
    "reputation_impact": 10.0,
    "operational_impact": 10.0,
}

# Data classification scores
DATA_CLASSIFICATION_SCORES: Dict[str, float] = {
    "public": 0.0,
    "internal": 0.25,
    "confidential": 0.5,
    "restricted": 0.75,
    "pii": 0.9,
    "phi": 1.0,
    "pci": 1.0,
}

# Impact level scores
IMPACT_LEVEL_SCORES: Dict[str, float] = {
    "none": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "critical": 1.0,
}


# ---------------------------------------------------------------------------
# Incident Classifier
# ---------------------------------------------------------------------------


class IncidentClassifier:
    """Classifies incident severity and calculates business impact.

    Determines the appropriate severity level (P0-P3) for incidents based
    on their type, affected scope, and potential business impact. Also
    calculates a composite business impact score.

    Attributes:
        config: Incident response configuration.

    Example:
        >>> classifier = IncidentClassifier(config)
        >>> severity = classifier.classify(incident)
        >>> impact = classifier.calculate_business_impact(incident)
    """

    def __init__(
        self,
        config: Optional[IncidentResponseConfig] = None,
    ) -> None:
        """Initialize the incident classifier.

        Args:
            config: Incident response configuration.
        """
        self.config = config or get_config()

        # Customizable scoring thresholds
        self._elevation_thresholds = {
            "users_affected_p0": 10000,
            "users_affected_p1": 1000,
            "users_affected_p2": 100,
            "systems_affected_p0": 10,
            "systems_affected_p1": 5,
            "impact_score_p0": 80,
            "impact_score_p1": 60,
        }

        logger.info("IncidentClassifier initialized")

    def classify(self, incident: Incident) -> EscalationLevel:
        """Classify incident severity.

        Determines the appropriate severity level based on:
        - Incident type (base severity)
        - Number of affected users
        - Number of affected systems
        - Business impact score

        Args:
            incident: Incident to classify.

        Returns:
            Recommended escalation level.
        """
        # Get base severity from incident type
        base_severity = INCIDENT_TYPE_BASE_SEVERITY.get(
            incident.incident_type,
            EscalationLevel.P2,
        )

        logger.debug(
            "Base severity for %s: %s",
            incident.incident_type.value,
            base_severity.value,
        )

        # Calculate business impact
        impact = self.calculate_business_impact(incident)

        # Determine if severity should be elevated
        final_severity = self._determine_final_severity(
            base_severity,
            incident,
            impact,
        )

        logger.info(
            "Classified incident %s as %s (base: %s, impact_score: %.1f)",
            incident.incident_number,
            final_severity.value,
            base_severity.value,
            impact.score,
        )

        return final_severity

    def _determine_final_severity(
        self,
        base_severity: EscalationLevel,
        incident: Incident,
        impact: BusinessImpact,
    ) -> EscalationLevel:
        """Determine final severity, potentially elevated from base.

        Args:
            base_severity: Base severity from incident type.
            incident: Incident being classified.
            impact: Calculated business impact.

        Returns:
            Final escalation level.
        """
        severity_order = [
            EscalationLevel.P3,
            EscalationLevel.P2,
            EscalationLevel.P1,
            EscalationLevel.P0,
        ]

        current_idx = severity_order.index(base_severity)

        # Check for elevation conditions
        elevations = 0

        # High impact score elevation
        if impact.score >= self._elevation_thresholds["impact_score_p0"]:
            elevations = max(elevations, 3 - current_idx)  # Elevate to P0
        elif impact.score >= self._elevation_thresholds["impact_score_p1"]:
            elevations = max(elevations, 2 - current_idx)  # Elevate to P1

        # Many affected users elevation
        users_affected = incident.affected_users or 0
        if users_affected >= self._elevation_thresholds["users_affected_p0"]:
            elevations = max(elevations, 3 - current_idx)
        elif users_affected >= self._elevation_thresholds["users_affected_p1"]:
            elevations = max(elevations, 2 - current_idx)
        elif users_affected >= self._elevation_thresholds["users_affected_p2"]:
            elevations = max(elevations, 1)

        # Many affected systems elevation
        systems_count = len(incident.affected_systems)
        if systems_count >= self._elevation_thresholds["systems_affected_p0"]:
            elevations = max(elevations, 3 - current_idx)
        elif systems_count >= self._elevation_thresholds["systems_affected_p1"]:
            elevations = max(elevations, 2 - current_idx)

        # Apply elevation
        final_idx = min(current_idx + elevations, 3)

        return severity_order[final_idx]

    def calculate_business_impact(self, incident: Incident) -> BusinessImpact:
        """Calculate business impact score.

        Assesses the potential business impact based on multiple factors
        including data sensitivity, affected users, regulatory implications,
        and operational impact.

        Args:
            incident: Incident to assess.

        Returns:
            BusinessImpact assessment.
        """
        factors: Dict[str, float] = {}

        # Data classification factor
        data_class = self._infer_data_classification(incident)
        factors["data_classification"] = DATA_CLASSIFICATION_SCORES.get(
            data_class, 0.25
        )

        # Users affected factor
        users_affected = incident.affected_users or 0
        if users_affected >= 100000:
            factors["users_affected"] = 1.0
        elif users_affected >= 10000:
            factors["users_affected"] = 0.8
        elif users_affected >= 1000:
            factors["users_affected"] = 0.6
        elif users_affected >= 100:
            factors["users_affected"] = 0.4
        elif users_affected >= 10:
            factors["users_affected"] = 0.2
        else:
            factors["users_affected"] = 0.1

        # Regulatory impact factor
        regulatory_impact = self._assess_regulatory_impact(incident)
        factors["regulatory_impact"] = IMPACT_LEVEL_SCORES.get(
            regulatory_impact, 0.0
        )

        # Revenue impact factor
        revenue_impact = self._assess_revenue_impact(incident)
        factors["revenue_impact"] = IMPACT_LEVEL_SCORES.get(
            revenue_impact, 0.0
        )

        # Reputation impact factor
        reputation_impact = self._assess_reputation_impact(incident)
        factors["reputation_impact"] = IMPACT_LEVEL_SCORES.get(
            reputation_impact, 0.25
        )

        # Operational impact factor
        operational_impact = self._assess_operational_impact(incident)
        factors["operational_impact"] = IMPACT_LEVEL_SCORES.get(
            operational_impact, 0.25
        )

        # Calculate weighted score
        total_score = 0.0
        for factor_name, factor_value in factors.items():
            weight = IMPACT_WEIGHTS.get(factor_name, 0.0)
            total_score += factor_value * weight

        return BusinessImpact(
            score=total_score,
            revenue_impact=revenue_impact,
            users_affected=users_affected,
            data_classification=data_class,
            regulatory_impact=regulatory_impact,
            reputation_impact=reputation_impact,
            operational_impact=operational_impact,
            factors=factors,
        )

    def _infer_data_classification(self, incident: Incident) -> str:
        """Infer data classification from incident type.

        Args:
            incident: Incident to analyze.

        Returns:
            Data classification level.
        """
        # High-sensitivity incident types
        if incident.incident_type in (
            IncidentType.DATA_BREACH,
            IncidentType.DATA_EXFILTRATION,
            IncidentType.CREDENTIAL_COMPROMISE,
        ):
            return "restricted"

        if incident.incident_type in (
            IncidentType.SQL_INJECTION,
            IncidentType.UNAUTHORIZED_ACCESS,
            IncidentType.INSIDER_THREAT,
        ):
            return "confidential"

        if incident.incident_type in (
            IncidentType.MALWARE,
            IncidentType.RANSOMWARE,
            IncidentType.PRIVILEGE_ESCALATION,
        ):
            return "confidential"

        # Check metadata for explicit classification
        if "data_classification" in incident.metadata:
            return str(incident.metadata["data_classification"])

        # Check tags for hints
        tags_lower = [t.lower() for t in incident.tags]
        if any(t in tags_lower for t in ("pii", "personal", "gdpr")):
            return "pii"
        if any(t in tags_lower for t in ("phi", "health", "hipaa")):
            return "phi"
        if any(t in tags_lower for t in ("pci", "payment", "card")):
            return "pci"

        return "internal"

    def _assess_regulatory_impact(self, incident: Incident) -> str:
        """Assess potential regulatory impact.

        Args:
            incident: Incident to analyze.

        Returns:
            Impact level string.
        """
        # Data breach incidents have high regulatory impact
        if incident.incident_type == IncidentType.DATA_BREACH:
            return "critical"

        # Data exposure incidents
        if incident.incident_type in (
            IncidentType.DATA_EXFILTRATION,
            IncidentType.CREDENTIAL_COMPROMISE,
        ):
            return "high"

        # Compliance violations
        if incident.incident_type == IncidentType.COMPLIANCE_VIOLATION:
            return "high"

        # Check for regulated data involvement
        data_class = self._infer_data_classification(incident)
        if data_class in ("pii", "phi", "pci"):
            return "high"

        return "low"

    def _assess_revenue_impact(self, incident: Incident) -> str:
        """Assess potential revenue impact.

        Args:
            incident: Incident to analyze.

        Returns:
            Impact level string.
        """
        # DDoS and availability issues directly impact revenue
        if incident.incident_type in (
            IncidentType.DDOS_ATTACK,
            IncidentType.AVAILABILITY,
            IncidentType.RANSOMWARE,
        ):
            return "high"

        # Data breaches can lead to significant financial impact
        if incident.incident_type == IncidentType.DATA_BREACH:
            return "critical"

        # Check affected systems for revenue-critical services
        critical_services = {"api", "payment", "checkout", "billing"}
        for system in incident.affected_systems:
            if any(svc in system.lower() for svc in critical_services):
                return "high"

        return "low"

    def _assess_reputation_impact(self, incident: Incident) -> str:
        """Assess potential reputation impact.

        Args:
            incident: Incident to analyze.

        Returns:
            Impact level string.
        """
        # Public-facing incidents have high reputation impact
        if incident.incident_type in (
            IncidentType.DATA_BREACH,
            IncidentType.RANSOMWARE,
            IncidentType.DDOS_ATTACK,
        ):
            return "high"

        # Customer-affecting incidents
        users_affected = incident.affected_users or 0
        if users_affected >= 1000:
            return "high"
        elif users_affected >= 100:
            return "medium"

        return "low"

    def _assess_operational_impact(self, incident: Incident) -> str:
        """Assess operational impact.

        Args:
            incident: Incident to analyze.

        Returns:
            Impact level string.
        """
        # Full service outage
        if incident.incident_type == IncidentType.AVAILABILITY:
            systems_count = len(incident.affected_systems)
            if systems_count >= 5:
                return "critical"
            elif systems_count >= 2:
                return "high"
            return "medium"

        # Ransomware requires full incident response
        if incident.incident_type == IncidentType.RANSOMWARE:
            return "critical"

        # Multiple systems affected
        if len(incident.affected_systems) >= 3:
            return "high"

        return "low"

    def get_severity_level_info(
        self,
        severity: EscalationLevel,
    ) -> SeverityLevel:
        """Get detailed information about a severity level.

        Args:
            severity: Escalation level.

        Returns:
            SeverityLevel details.
        """
        return SEVERITY_LEVELS.get(
            severity,
            SEVERITY_LEVELS[EscalationLevel.P2],
        )

    def classify_alert(self, alert: Alert) -> EscalationLevel:
        """Classify a single alert's severity.

        Args:
            alert: Alert to classify.

        Returns:
            Recommended escalation level.
        """
        # Use alert's existing severity as base
        return alert.severity

    def get_required_notification_channels(
        self,
        severity: EscalationLevel,
    ) -> List[str]:
        """Get required notification channels for a severity level.

        Args:
            severity: Escalation level.

        Returns:
            List of notification channel names.
        """
        level_info = self.get_severity_level_info(severity)
        return level_info.notification_channels

    def requires_incident_commander(
        self,
        severity: EscalationLevel,
    ) -> bool:
        """Check if severity level requires incident commander.

        Args:
            severity: Escalation level.

        Returns:
            True if IC is required.
        """
        level_info = self.get_severity_level_info(severity)
        return level_info.requires_incident_commander


# ---------------------------------------------------------------------------
# Global Classifier Instance
# ---------------------------------------------------------------------------

_global_classifier: Optional[IncidentClassifier] = None


def get_classifier(
    config: Optional[IncidentResponseConfig] = None,
) -> IncidentClassifier:
    """Get or create the global incident classifier.

    Args:
        config: Optional configuration override.

    Returns:
        The global IncidentClassifier instance.
    """
    global _global_classifier

    if _global_classifier is None:
        _global_classifier = IncidentClassifier(config)

    return _global_classifier


def reset_classifier() -> None:
    """Reset the global classifier."""
    global _global_classifier
    _global_classifier = None


__all__ = [
    "IncidentClassifier",
    "SeverityLevel",
    "BusinessImpact",
    "SEVERITY_LEVELS",
    "INCIDENT_TYPE_BASE_SEVERITY",
    "get_classifier",
    "reset_classifier",
]
