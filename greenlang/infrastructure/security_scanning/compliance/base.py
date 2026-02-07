# -*- coding: utf-8 -*-
"""
Compliance Framework Base Classes - SEC-007

Abstract base classes and data models for compliance framework implementations.
Provides the common interface that SOC 2, ISO 27001, and GDPR compliance
classes implement.

Example:
    >>> class MyFramework(ComplianceFramework):
    ...     async def check_controls(self, scan_results):
    ...         # Implementation
    ...         pass
    ...
    ...     def generate_report(self, results):
    ...         # Implementation
    ...         pass

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FrameworkType(str, Enum):
    """Supported compliance framework types."""

    SOC2 = "SOC2"
    ISO27001 = "ISO27001"
    GDPR = "GDPR"
    PCI_DSS = "PCI_DSS"
    HIPAA = "HIPAA"


class ControlStatus(str, Enum):
    """Status of a compliance control check."""

    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_TESTED = "NOT_TESTED"


class ControlCategory(str, Enum):
    """Categories of compliance controls."""

    ACCESS_CONTROL = "access_control"
    VULNERABILITY_MANAGEMENT = "vulnerability_management"
    CHANGE_MANAGEMENT = "change_management"
    ENCRYPTION = "encryption"
    LOGGING_MONITORING = "logging_monitoring"
    INCIDENT_RESPONSE = "incident_response"
    DATA_PROTECTION = "data_protection"
    SECURE_DEVELOPMENT = "secure_development"
    NETWORK_SECURITY = "network_security"
    PHYSICAL_SECURITY = "physical_security"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ControlDefinition:
    """Definition of a compliance control.

    Attributes:
        control_id: Unique identifier (e.g., "CC6.1", "A.12.6.1").
        name: Human-readable control name.
        description: Detailed description of the control.
        category: Category of the control.
        requirements: List of specific requirements for the control.
        automated_checks: List of automated checks that verify this control.
        evidence_types: Types of evidence required for this control.
    """

    control_id: str
    name: str
    description: str
    category: ControlCategory
    requirements: List[str] = field(default_factory=list)
    automated_checks: List[str] = field(default_factory=list)
    evidence_types: List[str] = field(default_factory=list)


@dataclass
class ControlResult:
    """Result of checking a single compliance control.

    Attributes:
        control_id: The control that was checked.
        status: Pass/Fail/Partial status.
        score: Numeric score (0.0 to 1.0).
        findings: List of specific findings related to this control.
        evidence: Evidence collected for this control.
        recommendations: Suggested remediation actions.
        checked_at: Timestamp of the check.
        details: Additional details about the check.
    """

    control_id: str
    status: ControlStatus
    score: float = 0.0
    findings: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if the control passed."""
        return self.status == ControlStatus.PASS

    @property
    def failed(self) -> bool:
        """Check if the control failed."""
        return self.status == ControlStatus.FAIL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "control_id": self.control_id,
            "status": self.status.value,
            "score": self.score,
            "findings_count": len(self.findings),
            "evidence_count": len(self.evidence),
            "recommendations": self.recommendations,
            "checked_at": self.checked_at.isoformat(),
            "details": self.details,
        }


@dataclass
class ComplianceReport:
    """Complete compliance report for a framework.

    Attributes:
        framework: The compliance framework.
        report_id: Unique report identifier.
        generated_at: Report generation timestamp.
        period_start: Start of the assessment period.
        period_end: End of the assessment period.
        overall_status: Overall compliance status.
        overall_score: Overall compliance score (0.0 to 1.0).
        control_results: Results for each control checked.
        summary: Executive summary text.
        findings_by_severity: Count of findings by severity.
        gaps: Identified compliance gaps.
        recommendations: Priority recommendations.
        metadata: Additional report metadata.
    """

    framework: FrameworkType
    report_id: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    overall_status: ControlStatus = ControlStatus.NOT_TESTED
    overall_score: float = 0.0
    control_results: List[ControlResult] = field(default_factory=list)
    summary: str = ""
    findings_by_severity: Dict[str, int] = field(default_factory=dict)
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate report ID if not provided."""
        if not self.report_id:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            content = f"{self.framework.value}:{timestamp}"
            self.report_id = hashlib.sha256(content.encode()).hexdigest()[:12]

    @property
    def passed_controls(self) -> List[ControlResult]:
        """Get all passed controls."""
        return [r for r in self.control_results if r.status == ControlStatus.PASS]

    @property
    def failed_controls(self) -> List[ControlResult]:
        """Get all failed controls."""
        return [r for r in self.control_results if r.status == ControlStatus.FAIL]

    @property
    def partial_controls(self) -> List[ControlResult]:
        """Get all partially compliant controls."""
        return [r for r in self.control_results if r.status == ControlStatus.PARTIAL]

    def calculate_score(self) -> float:
        """Calculate overall compliance score."""
        if not self.control_results:
            return 0.0

        total_score = sum(r.score for r in self.control_results)
        self.overall_score = total_score / len(self.control_results)
        return self.overall_score

    def calculate_status(self) -> ControlStatus:
        """Determine overall compliance status based on control results."""
        if not self.control_results:
            self.overall_status = ControlStatus.NOT_TESTED
            return self.overall_status

        failed_count = len(self.failed_controls)
        partial_count = len(self.partial_controls)
        passed_count = len(self.passed_controls)
        total = len(self.control_results)

        if failed_count == 0 and partial_count == 0:
            self.overall_status = ControlStatus.PASS
        elif failed_count > total * 0.2:  # More than 20% failed
            self.overall_status = ControlStatus.FAIL
        else:
            self.overall_status = ControlStatus.PARTIAL

        return self.overall_status

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "framework": self.framework.value,
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "overall_status": self.overall_status.value,
            "overall_score": self.overall_score,
            "summary": self.summary,
            "statistics": {
                "total_controls": len(self.control_results),
                "passed": len(self.passed_controls),
                "failed": len(self.failed_controls),
                "partial": len(self.partial_controls),
            },
            "findings_by_severity": self.findings_by_severity,
            "gaps": self.gaps,
            "recommendations": self.recommendations,
            "control_results": [r.to_dict() for r in self.control_results],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Finding to Control Mapping
# ---------------------------------------------------------------------------


@dataclass
class FindingControlMapping:
    """Maps a security finding type to compliance controls.

    Attributes:
        finding_type: Type of security finding (e.g., "vulnerability", "secret").
        scanner: Scanner that produced the finding.
        control_ids: List of control IDs this finding relates to.
        severity_weight: How much this finding type affects the control score.
    """

    finding_type: str
    scanner: str
    control_ids: List[str]
    severity_weight: float = 1.0


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------


class ComplianceFramework(ABC):
    """Abstract base class for compliance framework implementations.

    Subclasses must implement:
        - check_controls(): Evaluate controls against scan results
        - generate_report(): Generate compliance report from results

    Attributes:
        framework_type: The type of compliance framework.
        controls: Dictionary of control definitions.
        finding_mappings: Mappings from findings to controls.
    """

    def __init__(self, framework_type: FrameworkType) -> None:
        """Initialize the compliance framework.

        Args:
            framework_type: The type of framework this implements.
        """
        self.framework_type = framework_type
        self.controls: Dict[str, ControlDefinition] = {}
        self.finding_mappings: List[FindingControlMapping] = []
        self._initialize_controls()
        self._initialize_mappings()

    @abstractmethod
    def _initialize_controls(self) -> None:
        """Initialize the control definitions for this framework.

        Subclasses must populate self.controls with ControlDefinition objects.
        """
        pass

    @abstractmethod
    def _initialize_mappings(self) -> None:
        """Initialize finding-to-control mappings.

        Subclasses must populate self.finding_mappings with mapping rules.
        """
        pass

    @abstractmethod
    async def check_controls(
        self,
        scan_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ControlResult]:
        """Check compliance controls against scan results.

        Args:
            scan_results: List of security scan findings.
            context: Additional context (e.g., asset inventory, configs).

        Returns:
            List of ControlResult objects for each control checked.
        """
        pass

    @abstractmethod
    def generate_report(
        self,
        results: List[ControlResult],
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> ComplianceReport:
        """Generate a compliance report from control check results.

        Args:
            results: List of control check results.
            period_start: Start of the assessment period.
            period_end: End of the assessment period.

        Returns:
            Complete ComplianceReport object.
        """
        pass

    def get_control(self, control_id: str) -> Optional[ControlDefinition]:
        """Get a specific control definition.

        Args:
            control_id: The control identifier.

        Returns:
            ControlDefinition or None if not found.
        """
        return self.controls.get(control_id)

    def get_controls_for_finding(
        self, finding_type: str, scanner: str
    ) -> List[str]:
        """Get control IDs that a finding type relates to.

        Args:
            finding_type: Type of the finding.
            scanner: Scanner that produced the finding.

        Returns:
            List of related control IDs.
        """
        control_ids: Set[str] = set()
        for mapping in self.finding_mappings:
            if mapping.finding_type == finding_type or mapping.scanner == scanner:
                control_ids.update(mapping.control_ids)
        return list(control_ids)

    def get_controls_by_category(
        self, category: ControlCategory
    ) -> List[ControlDefinition]:
        """Get all controls in a category.

        Args:
            category: The control category.

        Returns:
            List of control definitions in the category.
        """
        return [
            ctrl for ctrl in self.controls.values()
            if ctrl.category == category
        ]

    def _calculate_control_score(
        self,
        control_id: str,
        findings: List[Dict[str, Any]],
    ) -> float:
        """Calculate compliance score for a control based on findings.

        Args:
            control_id: The control being scored.
            findings: Findings related to this control.

        Returns:
            Score from 0.0 (non-compliant) to 1.0 (fully compliant).
        """
        if not findings:
            return 1.0  # No findings = compliant

        # Weight findings by severity
        severity_weights = {
            "CRITICAL": 1.0,
            "HIGH": 0.8,
            "MEDIUM": 0.5,
            "LOW": 0.2,
            "INFO": 0.1,
        }

        total_weight = sum(
            severity_weights.get(f.get("severity", "LOW").upper(), 0.2)
            for f in findings
        )

        # Score decreases with more/severe findings
        # 10+ weighted findings = 0 score
        score = max(0.0, 1.0 - (total_weight / 10.0))
        return round(score, 2)

    def _determine_control_status(
        self,
        score: float,
        findings: List[Dict[str, Any]],
    ) -> ControlStatus:
        """Determine control status based on score and findings.

        Args:
            score: The control's compliance score.
            findings: Findings related to the control.

        Returns:
            ControlStatus based on the score.
        """
        # Check for critical findings
        critical_count = sum(
            1 for f in findings
            if f.get("severity", "").upper() == "CRITICAL"
        )
        if critical_count > 0:
            return ControlStatus.FAIL

        if score >= 0.9:
            return ControlStatus.PASS
        elif score >= 0.6:
            return ControlStatus.PARTIAL
        else:
            return ControlStatus.FAIL

    def _generate_recommendations(
        self,
        control_id: str,
        findings: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate recommendations based on findings.

        Args:
            control_id: The control being evaluated.
            findings: Findings related to the control.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []
        control = self.get_control(control_id)

        if not control:
            return recommendations

        # Group findings by severity
        critical = [f for f in findings if f.get("severity", "").upper() == "CRITICAL"]
        high = [f for f in findings if f.get("severity", "").upper() == "HIGH"]

        if critical:
            recommendations.append(
                f"URGENT: Address {len(critical)} critical findings "
                f"affecting {control.name} within 24 hours"
            )

        if high:
            recommendations.append(
                f"HIGH PRIORITY: Remediate {len(high)} high-severity findings "
                f"affecting {control.name} within 7 days"
            )

        if findings:
            recommendations.append(
                f"Review and address all {len(findings)} findings related to "
                f"{control.name} ({control_id})"
            )

        return recommendations
