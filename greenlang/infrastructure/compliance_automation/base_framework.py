# -*- coding: utf-8 -*-
"""
Base Compliance Framework - SEC-010 Phase 5

Abstract base class for compliance framework implementations. All framework-specific
implementations (ISO 27001, GDPR, PCI-DSS, etc.) inherit from this class to ensure
consistent interfaces for control assessment, evidence collection, and reporting.

Classes:
    - BaseComplianceFramework: Abstract base class for all compliance frameworks.

Example:
    >>> class ISO27001Framework(BaseComplianceFramework):
    ...     @property
    ...     def framework_id(self) -> ComplianceFramework:
    ...         return ComplianceFramework.ISO27001
    ...
    ...     async def get_controls(self) -> List[ControlMapping]:
    ...         return [...]
    ...
    >>> framework = ISO27001Framework(config)
    >>> status = await framework.calculate_compliance_score()

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

from greenlang.infrastructure.compliance_automation.models import (
    ComplianceFramework,
    ComplianceGap,
    ComplianceReport,
    ComplianceStatus,
    ControlMapping,
    ControlStatus,
    EvidenceSource,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Assessment Result
# ---------------------------------------------------------------------------


class ControlAssessmentResult:
    """Result of assessing a single control.

    Attributes:
        control_id: The control identifier.
        status: The assessed status.
        score: Score from 0-100.
        evidence_collected: List of evidence items collected.
        gaps: List of identified gaps.
        notes: Assessment notes.
        assessed_at: When the assessment was performed.
    """

    def __init__(
        self,
        control_id: str,
        status: ControlStatus = ControlStatus.NOT_IMPLEMENTED,
        score: float = 0.0,
        evidence_collected: Optional[List[Dict[str, Any]]] = None,
        gaps: Optional[List[ComplianceGap]] = None,
        notes: str = "",
    ) -> None:
        """Initialize the assessment result.

        Args:
            control_id: The control identifier.
            status: The assessed status.
            score: Score from 0-100.
            evidence_collected: List of evidence items collected.
            gaps: List of identified gaps.
            notes: Assessment notes.
        """
        self.control_id = control_id
        self.status = status
        self.score = max(0.0, min(100.0, score))
        self.evidence_collected = evidence_collected or []
        self.gaps = gaps or []
        self.notes = notes
        self.assessed_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "control_id": self.control_id,
            "status": self.status.value,
            "score": self.score,
            "evidence_collected": self.evidence_collected,
            "gaps": [g.model_dump() for g in self.gaps],
            "notes": self.notes,
            "assessed_at": self.assessed_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Base Compliance Framework
# ---------------------------------------------------------------------------


class BaseComplianceFramework(ABC):
    """Abstract base class for compliance framework implementations.

    All compliance frameworks (ISO 27001, GDPR, PCI-DSS, CCPA, LGPD) must
    inherit from this class and implement the required abstract methods.

    This class provides common functionality for:
    - Control management and assessment
    - Evidence collection and validation
    - Compliance score calculation
    - Report generation
    - Gap identification and tracking

    Attributes:
        config: Configuration settings for the framework.
        _controls: Cached list of controls.
        _last_assessment: Timestamp of last assessment.

    Example:
        >>> class ISO27001Framework(BaseComplianceFramework):
        ...     @property
        ...     def framework_id(self) -> ComplianceFramework:
        ...         return ComplianceFramework.ISO27001
        ...
        >>> framework = ISO27001Framework(config)
        >>> controls = await framework.get_controls()
        >>> status = await framework.calculate_compliance_score()
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the compliance framework.

        Args:
            config: Optional configuration settings. If None, uses defaults.
        """
        self.config = config
        self._controls: Optional[List[ControlMapping]] = None
        self._last_assessment: Optional[datetime] = None
        self._assessment_results: Dict[str, ControlAssessmentResult] = {}
        logger.info(
            "Initialized %s compliance framework",
            self.framework_id.value,
        )

    # -------------------------------------------------------------------------
    # Abstract Properties
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def framework_id(self) -> ComplianceFramework:
        """Return the unique identifier for this framework.

        Returns:
            The ComplianceFramework enum value for this implementation.
        """
        pass

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Return the human-readable name of the framework.

        Returns:
            Human-readable framework name.
        """
        pass

    @property
    @abstractmethod
    def framework_version(self) -> str:
        """Return the version of the framework.

        Returns:
            Framework version string (e.g., "2022" for ISO 27001:2022).
        """
        pass

    # -------------------------------------------------------------------------
    # Abstract Methods - Must be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_controls(self) -> List[ControlMapping]:
        """Get all controls for this framework.

        Returns:
            List of ControlMapping objects representing all controls
            in this compliance framework.
        """
        pass

    @abstractmethod
    async def assess_control(
        self,
        control_id: str,
        collect_evidence: bool = True,
    ) -> ControlAssessmentResult:
        """Assess a single control.

        Args:
            control_id: The identifier of the control to assess.
            collect_evidence: Whether to collect evidence during assessment.

        Returns:
            ControlAssessmentResult with status, score, and gaps.

        Raises:
            ValueError: If the control_id is not valid for this framework.
        """
        pass

    @abstractmethod
    async def collect_evidence(
        self,
        control_id: str,
    ) -> List[Dict[str, Any]]:
        """Collect evidence for a specific control.

        Args:
            control_id: The identifier of the control.

        Returns:
            List of evidence items (dictionaries with evidence data).

        Raises:
            ValueError: If the control_id is not valid for this framework.
        """
        pass

    # -------------------------------------------------------------------------
    # Implemented Methods - Common functionality
    # -------------------------------------------------------------------------

    async def calculate_compliance_score(self) -> ComplianceStatus:
        """Calculate the overall compliance score for this framework.

        Assesses all controls and calculates weighted scores to determine
        the overall compliance status.

        Returns:
            ComplianceStatus with overall score and breakdown.
        """
        logger.info("Calculating compliance score for %s", self.framework_id.value)

        controls = await self.get_controls()
        total_controls = len(controls)
        compliant_count = 0
        non_compliant_count = 0
        not_applicable_count = 0
        total_score = 0.0
        all_gaps: List[str] = []

        # Assess each control
        for control in controls:
            try:
                result = await self.assess_control(
                    control.framework_control,
                    collect_evidence=True,
                )
                self._assessment_results[control.framework_control] = result

                # Count by status
                if result.status == ControlStatus.NOT_APPLICABLE:
                    not_applicable_count += 1
                elif result.status in (
                    ControlStatus.VERIFIED,
                    ControlStatus.IMPLEMENTED,
                ):
                    compliant_count += 1
                    total_score += result.score
                elif result.status == ControlStatus.PARTIALLY_IMPLEMENTED:
                    compliant_count += 0.5  # Partial credit
                    total_score += result.score * 0.5
                else:
                    non_compliant_count += 1

                # Collect gaps
                for gap in result.gaps:
                    all_gaps.append(gap.id)

            except Exception as e:
                logger.error(
                    "Failed to assess control %s: %s",
                    control.framework_control,
                    str(e),
                )
                non_compliant_count += 1

        # Calculate overall score
        applicable_controls = total_controls - not_applicable_count
        if applicable_controls > 0:
            overall_score = total_score / applicable_controls
        else:
            overall_score = 100.0

        # Determine status
        if overall_score >= 95.0:
            status = "compliant"
        elif overall_score >= 70.0:
            status = "partial"
        else:
            status = "non_compliant"

        self._last_assessment = datetime.now(timezone.utc)

        return ComplianceStatus(
            framework=self.framework_id,
            score=overall_score,
            status=status,
            gaps=all_gaps,
            controls_total=total_controls,
            controls_compliant=int(compliant_count),
            controls_non_compliant=non_compliant_count,
            controls_not_applicable=not_applicable_count,
            last_assessed=self._last_assessment,
        )

    async def generate_report(
        self,
        include_evidence: bool = True,
        include_recommendations: bool = True,
    ) -> ComplianceReport:
        """Generate a compliance report for this framework.

        Args:
            include_evidence: Whether to include evidence in the report.
            include_recommendations: Whether to include recommendations.

        Returns:
            ComplianceReport with detailed compliance information.
        """
        logger.info("Generating compliance report for %s", self.framework_id.value)

        # Calculate current compliance
        status = await self.calculate_compliance_score()

        # Compile findings
        findings: List[Dict[str, Any]] = []
        for control_id, result in self._assessment_results.items():
            finding = {
                "control_id": control_id,
                "status": result.status.value,
                "score": result.score,
                "notes": result.notes,
            }
            if include_evidence:
                finding["evidence"] = result.evidence_collected
            findings.append(finding)

        # Generate recommendations
        recommendations: List[str] = []
        if include_recommendations:
            recommendations = self._generate_recommendations()

        # Build executive summary
        executive_summary = self._generate_executive_summary(status)

        return ComplianceReport(
            report_type="assessment",
            frameworks=[self.framework_id],
            overall_score=status.score,
            framework_scores={self.framework_id.value: status.score},
            executive_summary=executive_summary,
            findings=findings,
            recommendations=recommendations,
        )

    async def identify_gaps(self) -> List[ComplianceGap]:
        """Identify compliance gaps for this framework.

        Returns:
            List of ComplianceGap objects representing identified gaps.
        """
        logger.info("Identifying compliance gaps for %s", self.framework_id.value)

        # Ensure we have assessment results
        if not self._assessment_results:
            await self.calculate_compliance_score()

        gaps: List[ComplianceGap] = []
        for result in self._assessment_results.values():
            gaps.extend(result.gaps)

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps.sort(key=lambda g: severity_order.get(g.severity, 4))

        return gaps

    async def get_control_status(self, control_id: str) -> Optional[ControlStatus]:
        """Get the current status of a specific control.

        Args:
            control_id: The control identifier.

        Returns:
            The ControlStatus if assessed, None otherwise.
        """
        result = self._assessment_results.get(control_id)
        return result.status if result else None

    async def get_evidence_sources(
        self,
        control_id: str,
    ) -> List[EvidenceSource]:
        """Get evidence sources for a specific control.

        Args:
            control_id: The control identifier.

        Returns:
            List of EvidenceSource objects for the control.

        Raises:
            ValueError: If the control_id is not valid.
        """
        controls = await self.get_controls()
        for control in controls:
            if control.framework_control == control_id:
                sources = []
                for source_name in control.evidence_sources:
                    sources.append(
                        EvidenceSource(
                            name=source_name,
                            source_type="automated",
                            collection_method="automated",
                        )
                    )
                return sources

        raise ValueError(f"Control {control_id} not found in {self.framework_id.value}")

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on assessment results.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # Analyze gaps and generate recommendations
        critical_gaps = sum(
            1 for r in self._assessment_results.values()
            for g in r.gaps if g.severity == "critical"
        )
        high_gaps = sum(
            1 for r in self._assessment_results.values()
            for g in r.gaps if g.severity == "high"
        )

        if critical_gaps > 0:
            recommendations.append(
                f"URGENT: Address {critical_gaps} critical compliance gaps immediately"
            )
        if high_gaps > 0:
            recommendations.append(
                f"HIGH PRIORITY: Remediate {high_gaps} high-severity gaps within 30 days"
            )

        # Check for not-implemented controls
        not_implemented = sum(
            1 for r in self._assessment_results.values()
            if r.status == ControlStatus.NOT_IMPLEMENTED
        )
        if not_implemented > 0:
            recommendations.append(
                f"Implement {not_implemented} missing controls to improve compliance posture"
            )

        # Check for unverified controls
        unverified = sum(
            1 for r in self._assessment_results.values()
            if r.status == ControlStatus.IMPLEMENTED
        )
        if unverified > 0:
            recommendations.append(
                f"Verify {unverified} implemented controls through testing"
            )

        return recommendations

    def _generate_executive_summary(self, status: ComplianceStatus) -> str:
        """Generate an executive summary based on compliance status.

        Args:
            status: Current compliance status.

        Returns:
            Executive summary string.
        """
        summary_parts = [
            f"## {self.framework_name} Compliance Assessment",
            f"**Assessment Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            f"**Framework Version:** {self.framework_version}",
            f"",
            f"### Overall Status: {status.status.upper()}",
            f"**Compliance Score:** {status.score:.1f}%",
            f"",
            f"### Control Summary",
            f"- **Total Controls:** {status.controls_total}",
            f"- **Compliant:** {status.controls_compliant}",
            f"- **Non-Compliant:** {status.controls_non_compliant}",
            f"- **Not Applicable:** {status.controls_not_applicable}",
            f"",
            f"### Identified Gaps: {len(status.gaps)}",
        ]

        if status.score >= 95.0:
            summary_parts.append(
                "\nThe organization demonstrates strong compliance with "
                f"{self.framework_name} requirements."
            )
        elif status.score >= 70.0:
            summary_parts.append(
                f"\nThe organization shows partial compliance with {self.framework_name}. "
                "Remediation of identified gaps is recommended to achieve full compliance."
            )
        else:
            summary_parts.append(
                f"\nSignificant gaps exist in {self.framework_name} compliance. "
                "Immediate action is required to address critical deficiencies."
            )

        return "\n".join(summary_parts)


# ---------------------------------------------------------------------------
# Framework Registry
# ---------------------------------------------------------------------------


class ComplianceFrameworkRegistry:
    """Registry of available compliance framework implementations.

    Provides factory methods to instantiate framework implementations
    by their framework ID.
    """

    _frameworks: Dict[ComplianceFramework, Type[BaseComplianceFramework]] = {}

    @classmethod
    def register(
        cls,
        framework_id: ComplianceFramework,
        framework_class: Type[BaseComplianceFramework],
    ) -> None:
        """Register a framework implementation.

        Args:
            framework_id: The framework identifier.
            framework_class: The framework implementation class.
        """
        cls._frameworks[framework_id] = framework_class
        logger.info(
            "Registered compliance framework: %s -> %s",
            framework_id.value,
            framework_class.__name__,
        )

    @classmethod
    def get(
        cls,
        framework_id: ComplianceFramework,
        config: Optional[Any] = None,
    ) -> BaseComplianceFramework:
        """Get an instance of a registered framework.

        Args:
            framework_id: The framework identifier.
            config: Optional configuration for the framework.

        Returns:
            An instance of the framework implementation.

        Raises:
            ValueError: If the framework is not registered.
        """
        framework_class = cls._frameworks.get(framework_id)
        if framework_class is None:
            raise ValueError(
                f"Framework {framework_id.value} is not registered. "
                f"Available: {[f.value for f in cls._frameworks.keys()]}"
            )
        return framework_class(config)

    @classmethod
    def list_available(cls) -> List[ComplianceFramework]:
        """List all registered frameworks.

        Returns:
            List of registered framework IDs.
        """
        return list(cls._frameworks.keys())


__all__ = [
    "BaseComplianceFramework",
    "ControlAssessmentResult",
    "ComplianceFrameworkRegistry",
]
