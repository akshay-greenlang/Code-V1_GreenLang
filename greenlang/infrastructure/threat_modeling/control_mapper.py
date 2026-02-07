# -*- coding: utf-8 -*-
"""
Security Control Mapper - SEC-010 Phase 2

Maps threats to security controls from the GreenLang security track
(SEC-001 through SEC-009). Assesses control effectiveness and identifies
gaps in security coverage.

Example:
    >>> from greenlang.infrastructure.threat_modeling import ControlMapper, Threat
    >>> mapper = ControlMapper()
    >>> controls = mapper.map_threat_to_controls(threat)
    >>> effectiveness = mapper.assess_control_effectiveness(threat, controls)
    >>> gaps = mapper.identify_gaps(threats, controls)

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from greenlang.infrastructure.threat_modeling.models import (
    Mitigation,
    MitigationStatus,
    Threat,
    ThreatCategory,
    ThreatStatus,
)
from greenlang.infrastructure.threat_modeling.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Control Definitions
# ---------------------------------------------------------------------------


class ControlStatus(str, Enum):
    """Implementation status of a security control."""

    NOT_IMPLEMENTED = "not_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"


@dataclass
class SecurityControl:
    """A security control from the GreenLang security track."""

    id: str
    name: str
    description: str
    sec_module: str  # SEC-001 through SEC-009
    category: str  # Authentication, Authorization, Encryption, etc.
    mitigates_categories: List[str]  # STRIDE categories this control mitigates
    effectiveness: float  # Base effectiveness (0.0-1.0)
    status: ControlStatus = ControlStatus.IMPLEMENTED
    implementation_notes: str = ""


# ---------------------------------------------------------------------------
# Control Catalog
# ---------------------------------------------------------------------------

# Complete control catalog from SEC-001 through SEC-009
CONTROL_CATALOG: Dict[str, SecurityControl] = {
    # SEC-001: JWT Authentication
    "SEC-001-JWT": SecurityControl(
        id="SEC-001-JWT",
        name="JWT Token Authentication",
        description="JSON Web Token authentication with RS256 signing and token rotation.",
        sec_module="SEC-001",
        category="Authentication",
        mitigates_categories=["S", "E"],
        effectiveness=0.85,
    ),
    "SEC-001-MFA": SecurityControl(
        id="SEC-001-MFA",
        name="Multi-Factor Authentication",
        description="TOTP-based MFA for enhanced authentication security.",
        sec_module="SEC-001",
        category="Authentication",
        mitigates_categories=["S"],
        effectiveness=0.95,
    ),
    "SEC-001-OAUTH": SecurityControl(
        id="SEC-001-OAUTH",
        name="OAuth 2.0 / OIDC Integration",
        description="OAuth 2.0 and OpenID Connect for federated authentication.",
        sec_module="SEC-001",
        category="Authentication",
        mitigates_categories=["S", "R"],
        effectiveness=0.80,
    ),
    "SEC-001-SESSION": SecurityControl(
        id="SEC-001-SESSION",
        name="Session Management",
        description="Secure session handling with rotation and timeout.",
        sec_module="SEC-001",
        category="Authentication",
        mitigates_categories=["S", "E"],
        effectiveness=0.75,
    ),
    "SEC-001-BRUTE": SecurityControl(
        id="SEC-001-BRUTE",
        name="Brute Force Protection",
        description="Rate limiting and account lockout for authentication endpoints.",
        sec_module="SEC-001",
        category="Authentication",
        mitigates_categories=["S", "D"],
        effectiveness=0.80,
    ),

    # SEC-002: RBAC Authorization
    "SEC-002-RBAC": SecurityControl(
        id="SEC-002-RBAC",
        name="Role-Based Access Control",
        description="Hierarchical RBAC with role inheritance and delegation.",
        sec_module="SEC-002",
        category="Authorization",
        mitigates_categories=["E", "I"],
        effectiveness=0.85,
    ),
    "SEC-002-PERMS": SecurityControl(
        id="SEC-002-PERMS",
        name="Fine-Grained Permissions",
        description="Resource-level permissions with action granularity.",
        sec_module="SEC-002",
        category="Authorization",
        mitigates_categories=["E", "I"],
        effectiveness=0.90,
    ),
    "SEC-002-TENANT": SecurityControl(
        id="SEC-002-TENANT",
        name="Tenant Isolation",
        description="Multi-tenant data isolation with tenant context enforcement.",
        sec_module="SEC-002",
        category="Authorization",
        mitigates_categories=["E", "I"],
        effectiveness=0.95,
    ),
    "SEC-002-POLICY": SecurityControl(
        id="SEC-002-POLICY",
        name="Policy Engine",
        description="Centralized policy decision point for authorization.",
        sec_module="SEC-002",
        category="Authorization",
        mitigates_categories=["E"],
        effectiveness=0.85,
    ),

    # SEC-003: Data Encryption
    "SEC-003-AES": SecurityControl(
        id="SEC-003-AES",
        name="AES-256 Encryption at Rest",
        description="AES-256-GCM encryption for data at rest.",
        sec_module="SEC-003",
        category="Encryption",
        mitigates_categories=["I", "T"],
        effectiveness=0.95,
    ),
    "SEC-003-KMS": SecurityControl(
        id="SEC-003-KMS",
        name="Key Management Service",
        description="AWS KMS integration for cryptographic key management.",
        sec_module="SEC-003",
        category="Encryption",
        mitigates_categories=["I", "T", "S"],
        effectiveness=0.90,
    ),
    "SEC-003-FIELD": SecurityControl(
        id="SEC-003-FIELD",
        name="Field-Level Encryption",
        description="Application-layer encryption for sensitive fields.",
        sec_module="SEC-003",
        category="Encryption",
        mitigates_categories=["I"],
        effectiveness=0.85,
    ),
    "SEC-003-HASH": SecurityControl(
        id="SEC-003-HASH",
        name="Secure Hashing",
        description="Argon2id password hashing and SHA-256 for integrity.",
        sec_module="SEC-003",
        category="Encryption",
        mitigates_categories=["T", "S"],
        effectiveness=0.90,
    ),

    # SEC-004: TLS Configuration
    "SEC-004-TLS": SecurityControl(
        id="SEC-004-TLS",
        name="TLS 1.3 Encryption",
        description="TLS 1.3 for all data in transit with strong cipher suites.",
        sec_module="SEC-004",
        category="Transport Security",
        mitigates_categories=["I", "T"],
        effectiveness=0.95,
    ),
    "SEC-004-MTLS": SecurityControl(
        id="SEC-004-MTLS",
        name="Mutual TLS",
        description="mTLS for service-to-service authentication.",
        sec_module="SEC-004",
        category="Transport Security",
        mitigates_categories=["S", "I", "T"],
        effectiveness=0.95,
    ),
    "SEC-004-CERT": SecurityControl(
        id="SEC-004-CERT",
        name="Certificate Management",
        description="Automated certificate rotation and validation.",
        sec_module="SEC-004",
        category="Transport Security",
        mitigates_categories=["S", "I"],
        effectiveness=0.85,
    ),

    # SEC-005: Audit Logging
    "SEC-005-AUDIT": SecurityControl(
        id="SEC-005-AUDIT",
        name="Comprehensive Audit Logging",
        description="Immutable audit logs for all security-relevant events.",
        sec_module="SEC-005",
        category="Audit & Monitoring",
        mitigates_categories=["R"],
        effectiveness=0.95,
    ),
    "SEC-005-TAMPER": SecurityControl(
        id="SEC-005-TAMPER",
        name="Tamper-Evident Logs",
        description="Cryptographic chaining for log integrity verification.",
        sec_module="SEC-005",
        category="Audit & Monitoring",
        mitigates_categories=["R", "T"],
        effectiveness=0.90,
    ),
    "SEC-005-SIEM": SecurityControl(
        id="SEC-005-SIEM",
        name="SIEM Integration",
        description="Real-time log streaming to SIEM for threat detection.",
        sec_module="SEC-005",
        category="Audit & Monitoring",
        mitigates_categories=["R"],
        effectiveness=0.80,
    ),
    "SEC-005-ALERT": SecurityControl(
        id="SEC-005-ALERT",
        name="Security Alerting",
        description="Real-time alerts for security events and anomalies.",
        sec_module="SEC-005",
        category="Audit & Monitoring",
        mitigates_categories=["D"],
        effectiveness=0.75,
    ),

    # SEC-006: Secrets Management
    "SEC-006-VAULT": SecurityControl(
        id="SEC-006-VAULT",
        name="Secrets Vault",
        description="HashiCorp Vault / AWS Secrets Manager for secret storage.",
        sec_module="SEC-006",
        category="Secrets Management",
        mitigates_categories=["I", "S"],
        effectiveness=0.95,
    ),
    "SEC-006-ROTATE": SecurityControl(
        id="SEC-006-ROTATE",
        name="Secret Rotation",
        description="Automated secret rotation with zero-downtime updates.",
        sec_module="SEC-006",
        category="Secrets Management",
        mitigates_categories=["I", "S"],
        effectiveness=0.85,
    ),
    "SEC-006-INJECT": SecurityControl(
        id="SEC-006-INJECT",
        name="Secret Injection",
        description="Kubernetes secret injection without environment variables.",
        sec_module="SEC-006",
        category="Secrets Management",
        mitigates_categories=["I"],
        effectiveness=0.80,
    ),

    # SEC-007: Security Scanning
    "SEC-007-SAST": SecurityControl(
        id="SEC-007-SAST",
        name="Static Application Security Testing",
        description="Automated code scanning for vulnerabilities in CI/CD.",
        sec_module="SEC-007",
        category="Security Scanning",
        mitigates_categories=["T", "I", "E"],
        effectiveness=0.75,
    ),
    "SEC-007-DAST": SecurityControl(
        id="SEC-007-DAST",
        name="Dynamic Application Security Testing",
        description="Runtime vulnerability scanning of deployed applications.",
        sec_module="SEC-007",
        category="Security Scanning",
        mitigates_categories=["T", "I", "E"],
        effectiveness=0.70,
    ),
    "SEC-007-SCA": SecurityControl(
        id="SEC-007-SCA",
        name="Software Composition Analysis",
        description="Dependency vulnerability scanning and SBOM generation.",
        sec_module="SEC-007",
        category="Security Scanning",
        mitigates_categories=["T", "I", "E"],
        effectiveness=0.80,
    ),
    "SEC-007-CONTAINER": SecurityControl(
        id="SEC-007-CONTAINER",
        name="Container Image Scanning",
        description="Vulnerability scanning of container images in registry.",
        sec_module="SEC-007",
        category="Security Scanning",
        mitigates_categories=["T", "E"],
        effectiveness=0.75,
    ),
    "SEC-007-IACSS": SecurityControl(
        id="SEC-007-IACSS",
        name="Infrastructure as Code Security Scanning",
        description="Security scanning of Terraform/Kubernetes configurations.",
        sec_module="SEC-007",
        category="Security Scanning",
        mitigates_categories=["E", "I"],
        effectiveness=0.70,
    ),

    # SEC-008: Security Policies (OPA/Rego)
    "SEC-008-OPA": SecurityControl(
        id="SEC-008-OPA",
        name="Policy as Code (OPA)",
        description="Open Policy Agent for declarative policy enforcement.",
        sec_module="SEC-008",
        category="Policy Enforcement",
        mitigates_categories=["E", "T"],
        effectiveness=0.85,
    ),
    "SEC-008-ADMISSION": SecurityControl(
        id="SEC-008-ADMISSION",
        name="Kubernetes Admission Control",
        description="Policy-based admission control for K8s resources.",
        sec_module="SEC-008",
        category="Policy Enforcement",
        mitigates_categories=["E", "T"],
        effectiveness=0.80,
    ),
    "SEC-008-NETWORK": SecurityControl(
        id="SEC-008-NETWORK",
        name="Network Policies",
        description="Kubernetes network policies for microsegmentation.",
        sec_module="SEC-008",
        category="Policy Enforcement",
        mitigates_categories=["E", "D"],
        effectiveness=0.75,
    ),

    # SEC-009: SOC 2 Preparation
    "SEC-009-SOC2": SecurityControl(
        id="SEC-009-SOC2",
        name="SOC 2 Controls",
        description="SOC 2 Type II control implementation and evidence collection.",
        sec_module="SEC-009",
        category="Compliance",
        mitigates_categories=["R"],
        effectiveness=0.90,
    ),
    "SEC-009-EVIDENCE": SecurityControl(
        id="SEC-009-EVIDENCE",
        name="Evidence Collection",
        description="Automated evidence collection for compliance audits.",
        sec_module="SEC-009",
        category="Compliance",
        mitigates_categories=["R"],
        effectiveness=0.85,
    ),

    # General Controls
    "GEN-VALIDATION": SecurityControl(
        id="GEN-VALIDATION",
        name="Input Validation",
        description="Comprehensive input validation and sanitization.",
        sec_module="General",
        category="Application Security",
        mitigates_categories=["T", "I"],
        effectiveness=0.85,
    ),
    "GEN-RATELIMIT": SecurityControl(
        id="GEN-RATELIMIT",
        name="Rate Limiting",
        description="API rate limiting to prevent abuse and DoS.",
        sec_module="General",
        category="Application Security",
        mitigates_categories=["D"],
        effectiveness=0.80,
    ),
    "GEN-WAF": SecurityControl(
        id="GEN-WAF",
        name="Web Application Firewall",
        description="WAF protection against common web attacks.",
        sec_module="General",
        category="Application Security",
        mitigates_categories=["T", "D", "I"],
        effectiveness=0.75,
    ),
    "GEN-DDOS": SecurityControl(
        id="GEN-DDOS",
        name="DDoS Protection",
        description="AWS Shield / CloudFlare DDoS mitigation.",
        sec_module="General",
        category="Infrastructure Security",
        mitigates_categories=["D"],
        effectiveness=0.90,
    ),
}


# ---------------------------------------------------------------------------
# Gap Analysis Types
# ---------------------------------------------------------------------------


@dataclass
class ControlGap:
    """Identified gap in security control coverage."""

    threat_id: str
    threat_category: ThreatCategory
    threat_title: str
    missing_control_type: str
    severity: str
    recommendation: str
    suggested_controls: List[str]


@dataclass
class ControlEffectivenessReport:
    """Report on control effectiveness for a set of threats."""

    total_threats: int
    mitigated_threats: int
    partially_mitigated: int
    unmitigated_threats: int
    overall_coverage: float
    category_coverage: Dict[str, float]
    control_usage: Dict[str, int]
    gaps: List[ControlGap]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Control Mapper Implementation
# ---------------------------------------------------------------------------


class ControlMapper:
    """Maps threats to security controls and identifies gaps.

    Uses the GreenLang security control catalog (SEC-001 through SEC-009)
    to find applicable controls for threats and assess coverage.

    Attributes:
        config: Threat modeling configuration.
        control_catalog: Dictionary of available security controls.

    Example:
        >>> mapper = ControlMapper()
        >>> controls = mapper.map_threat_to_controls(threat)
        >>> effectiveness = mapper.assess_control_effectiveness(threat, controls)
    """

    def __init__(self) -> None:
        """Initialize the control mapper."""
        self.config = get_config()
        self.control_catalog = CONTROL_CATALOG.copy()
        logger.debug("Control mapper initialized with %d controls", len(self.control_catalog))

    def map_threat_to_controls(self, threat: Threat) -> List[SecurityControl]:
        """Find applicable controls for a threat.

        Maps a threat to security controls based on its STRIDE category
        and specific characteristics.

        Args:
            threat: The threat to map.

        Returns:
            List of applicable SecurityControl instances.
        """
        applicable_controls: List[SecurityControl] = []
        category_code = threat.category.value

        logger.debug("Mapping controls for threat: %s (%s)", threat.title[:30], category_code)

        # Find controls that mitigate this category
        for control in self.control_catalog.values():
            if category_code in control.mitigates_categories:
                applicable_controls.append(control)

        # Sort by effectiveness (highest first)
        applicable_controls.sort(key=lambda c: c.effectiveness, reverse=True)

        logger.debug("Found %d applicable controls", len(applicable_controls))
        return applicable_controls

    def assess_control_effectiveness(
        self,
        threat: Threat,
        controls: List[SecurityControl],
    ) -> float:
        """Assess coverage percentage for controls against a threat.

        Calculates how effectively the given controls mitigate the threat.

        Args:
            threat: The threat being mitigated.
            controls: List of controls applied to the threat.

        Returns:
            Effectiveness percentage (0.0-1.0).
        """
        if not controls:
            return 0.0

        category_code = threat.category.value

        # Calculate combined effectiveness
        # Using diminishing returns formula: 1 - (1-e1)(1-e2)...
        combined_ineffectiveness = 1.0

        for control in controls:
            if category_code in control.mitigates_categories:
                # Only count controls that actually mitigate this category
                control_effectiveness = control.effectiveness

                # Adjust for control status
                status_multiplier = {
                    ControlStatus.NOT_IMPLEMENTED: 0.0,
                    ControlStatus.PARTIALLY_IMPLEMENTED: 0.5,
                    ControlStatus.IMPLEMENTED: 0.9,
                    ControlStatus.VERIFIED: 1.0,
                }
                adjusted_effectiveness = control_effectiveness * status_multiplier.get(
                    control.status, 0.9
                )

                combined_ineffectiveness *= (1 - adjusted_effectiveness)

        effectiveness = 1 - combined_ineffectiveness

        logger.debug(
            "Control effectiveness for %s: %.2f%% (%d controls)",
            threat.title[:30],
            effectiveness * 100,
            len(controls),
        )

        return effectiveness

    def identify_gaps(
        self,
        threats: List[Threat],
        applied_controls: Optional[Dict[str, List[str]]] = None,
    ) -> List[ControlGap]:
        """Identify missing controls for unmitigated threats.

        Analyzes threats and identifies gaps where controls are missing
        or insufficient.

        Args:
            threats: List of threats to analyze.
            applied_controls: Dict mapping threat_id to list of control_ids.

        Returns:
            List of identified ControlGap instances.
        """
        gaps: List[ControlGap] = []

        if applied_controls is None:
            applied_controls = {}

        for threat in threats:
            # Skip already mitigated threats
            if threat.status in (ThreatStatus.MITIGATED, ThreatStatus.ACCEPTED):
                continue

            # Get applied controls for this threat
            threat_control_ids = set(applied_controls.get(threat.id, []))

            # Get all applicable controls
            applicable_controls = self.map_threat_to_controls(threat)

            # Check effectiveness
            applied_control_objs = [
                c for c in applicable_controls if c.id in threat_control_ids
            ]
            effectiveness = self.assess_control_effectiveness(threat, applied_control_objs)

            # If effectiveness is below threshold, identify gap
            if effectiveness < 0.7:  # 70% threshold for adequate coverage
                missing_controls = [
                    c for c in applicable_controls if c.id not in threat_control_ids
                ]

                # Determine severity based on threat risk
                if threat.risk_score >= 8.0:
                    severity = "critical"
                elif threat.risk_score >= 6.0:
                    severity = "high"
                elif threat.risk_score >= 4.0:
                    severity = "medium"
                else:
                    severity = "low"

                gap = ControlGap(
                    threat_id=threat.id,
                    threat_category=threat.category,
                    threat_title=threat.title,
                    missing_control_type=self._get_control_type_recommendation(threat.category),
                    severity=severity,
                    recommendation=self._generate_gap_recommendation(threat, missing_controls),
                    suggested_controls=[c.id for c in missing_controls[:5]],  # Top 5
                )
                gaps.append(gap)

        logger.info("Identified %d control gaps for %d threats", len(gaps), len(threats))
        return gaps

    def recommend_controls(self, threat: Threat) -> List[Mitigation]:
        """Suggest controls for unmitigated threats.

        Generates mitigation recommendations based on the threat's
        category and characteristics.

        Args:
            threat: The threat to mitigate.

        Returns:
            List of recommended Mitigation instances.
        """
        mitigations: List[Mitigation] = []

        # Get applicable controls
        controls = self.map_threat_to_controls(threat)

        # Sort by effectiveness and filter top recommendations
        top_controls = controls[:5]

        for control in top_controls:
            mitigation = Mitigation(
                threat_id=threat.id,
                control_id=control.id,
                title=f"Implement {control.name}",
                description=control.description,
                status=MitigationStatus.PROPOSED,
                effectiveness=control.effectiveness * 100,
                priority=self._calculate_priority(threat, control),
                effort_estimate=self._estimate_effort(control),
            )
            mitigations.append(mitigation)

        logger.debug(
            "Generated %d mitigation recommendations for threat: %s",
            len(mitigations),
            threat.title[:30],
        )

        return mitigations

    def generate_effectiveness_report(
        self,
        threats: List[Threat],
        applied_controls: Optional[Dict[str, List[str]]] = None,
    ) -> ControlEffectivenessReport:
        """Generate a comprehensive control effectiveness report.

        Analyzes all threats and controls to produce coverage metrics,
        gap analysis, and recommendations.

        Args:
            threats: List of threats to analyze.
            applied_controls: Dict mapping threat_id to list of control_ids.

        Returns:
            Complete ControlEffectivenessReport.
        """
        logger.info("Generating control effectiveness report for %d threats", len(threats))

        if applied_controls is None:
            applied_controls = {}

        mitigated_count = 0
        partial_count = 0
        unmitigated_count = 0
        category_coverage: Dict[str, List[float]] = {cat.value: [] for cat in ThreatCategory}
        control_usage: Dict[str, int] = {}

        for threat in threats:
            # Get applied controls
            threat_control_ids = set(applied_controls.get(threat.id, []))
            applicable_controls = self.map_threat_to_controls(threat)
            applied_control_objs = [
                c for c in applicable_controls if c.id in threat_control_ids
            ]

            # Calculate effectiveness
            effectiveness = self.assess_control_effectiveness(threat, applied_control_objs)
            category_coverage[threat.category.value].append(effectiveness)

            # Count control usage
            for control_id in threat_control_ids:
                control_usage[control_id] = control_usage.get(control_id, 0) + 1

            # Categorize mitigation status
            if effectiveness >= 0.8:
                mitigated_count += 1
            elif effectiveness >= 0.4:
                partial_count += 1
            else:
                unmitigated_count += 1

        # Calculate overall coverage
        all_effectiveness = [
            e for cat_list in category_coverage.values() for e in cat_list
        ]
        overall_coverage = sum(all_effectiveness) / len(all_effectiveness) if all_effectiveness else 0.0

        # Calculate category averages
        category_avg = {
            cat: (sum(scores) / len(scores) if scores else 0.0)
            for cat, scores in category_coverage.items()
        }

        # Identify gaps
        gaps = self.identify_gaps(threats, applied_controls)

        # Generate recommendations
        recommendations = self._generate_report_recommendations(
            overall_coverage, category_avg, gaps
        )

        report = ControlEffectivenessReport(
            total_threats=len(threats),
            mitigated_threats=mitigated_count,
            partially_mitigated=partial_count,
            unmitigated_threats=unmitigated_count,
            overall_coverage=overall_coverage,
            category_coverage=category_avg,
            control_usage=control_usage,
            gaps=gaps,
            recommendations=recommendations,
        )

        logger.info(
            "Report generated: coverage=%.1f%%, mitigated=%d, gaps=%d",
            overall_coverage * 100,
            mitigated_count,
            len(gaps),
        )

        return report

    def get_control(self, control_id: str) -> Optional[SecurityControl]:
        """Get a control by ID.

        Args:
            control_id: The control ID.

        Returns:
            SecurityControl if found, None otherwise.
        """
        return self.control_catalog.get(control_id)

    def get_controls_by_module(self, sec_module: str) -> List[SecurityControl]:
        """Get all controls from a specific SEC module.

        Args:
            sec_module: Module ID (e.g., "SEC-001").

        Returns:
            List of controls from that module.
        """
        return [
            c for c in self.control_catalog.values()
            if c.sec_module == sec_module
        ]

    def get_controls_by_category(self, category: str) -> List[SecurityControl]:
        """Get all controls in a specific category.

        Args:
            category: Control category (e.g., "Authentication").

        Returns:
            List of controls in that category.
        """
        return [
            c for c in self.control_catalog.values()
            if c.category == category
        ]

    def _get_control_type_recommendation(self, category: ThreatCategory) -> str:
        """Get recommended control type for a threat category.

        Args:
            category: The STRIDE category.

        Returns:
            Recommended control type string.
        """
        recommendations = {
            ThreatCategory.SPOOFING: "Authentication controls",
            ThreatCategory.TAMPERING: "Input validation and integrity controls",
            ThreatCategory.REPUDIATION: "Audit logging controls",
            ThreatCategory.INFORMATION_DISCLOSURE: "Encryption and access controls",
            ThreatCategory.DENIAL_OF_SERVICE: "Rate limiting and availability controls",
            ThreatCategory.ELEVATION_OF_PRIVILEGE: "Authorization controls",
        }
        return recommendations.get(category, "Security controls")

    def _generate_gap_recommendation(
        self,
        threat: Threat,
        missing_controls: List[SecurityControl],
    ) -> str:
        """Generate a recommendation for a control gap.

        Args:
            threat: The unmitigated threat.
            missing_controls: List of missing controls.

        Returns:
            Recommendation string.
        """
        if not missing_controls:
            return f"Review threat '{threat.title}' for additional mitigation options."

        top_control = missing_controls[0]
        return f"Implement '{top_control.name}' to mitigate {threat.category.full_name} threats."

    def _calculate_priority(self, threat: Threat, control: SecurityControl) -> int:
        """Calculate implementation priority for a control.

        Args:
            threat: The threat being mitigated.
            control: The control to implement.

        Returns:
            Priority (1=highest, 5=lowest).
        """
        # Higher risk threats get higher priority controls
        if threat.risk_score >= 8.0:
            base_priority = 1
        elif threat.risk_score >= 6.0:
            base_priority = 2
        elif threat.risk_score >= 4.0:
            base_priority = 3
        else:
            base_priority = 4

        # More effective controls get higher priority
        if control.effectiveness >= 0.9:
            return base_priority
        elif control.effectiveness >= 0.8:
            return min(5, base_priority + 1)
        else:
            return min(5, base_priority + 2)

    def _estimate_effort(self, control: SecurityControl) -> str:
        """Estimate implementation effort for a control.

        Args:
            control: The control to estimate.

        Returns:
            Effort estimate string.
        """
        # Simple heuristic based on control type
        high_effort_categories = {"Encryption", "Infrastructure Security"}
        low_effort_categories = {"Policy Enforcement", "Security Scanning"}

        if control.category in high_effort_categories:
            return "high"
        elif control.category in low_effort_categories:
            return "low"
        else:
            return "medium"

    def _generate_report_recommendations(
        self,
        overall_coverage: float,
        category_coverage: Dict[str, float],
        gaps: List[ControlGap],
    ) -> List[str]:
        """Generate recommendations for the effectiveness report.

        Args:
            overall_coverage: Overall control coverage.
            category_coverage: Coverage by STRIDE category.
            gaps: Identified control gaps.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # Overall coverage recommendation
        if overall_coverage < 0.5:
            recommendations.append(
                "Critical: Overall control coverage is below 50%. Immediate action required."
            )
        elif overall_coverage < 0.7:
            recommendations.append(
                "High priority: Improve control coverage to at least 70% before production deployment."
            )

        # Category-specific recommendations
        weak_categories = [
            cat for cat, cov in category_coverage.items() if cov < 0.5
        ]
        if weak_categories:
            cat_names = [ThreatCategory(c).full_name for c in weak_categories]
            recommendations.append(
                f"Focus on improving coverage for: {', '.join(cat_names)}"
            )

        # Critical gap recommendations
        critical_gaps = [g for g in gaps if g.severity == "critical"]
        if critical_gaps:
            recommendations.append(
                f"Address {len(critical_gaps)} critical control gaps immediately"
            )

        # Suggest specific controls
        suggested = set()
        for gap in gaps[:5]:
            suggested.update(gap.suggested_controls[:2])
        if suggested:
            recommendations.append(
                f"Priority controls to implement: {', '.join(list(suggested)[:5])}"
            )

        if not recommendations:
            recommendations.append("Control coverage is adequate. Continue monitoring.")

        return recommendations


__all__ = [
    "ControlMapper",
    "SecurityControl",
    "ControlGap",
    "ControlEffectivenessReport",
    "ControlStatus",
    "CONTROL_CATALOG",
]
