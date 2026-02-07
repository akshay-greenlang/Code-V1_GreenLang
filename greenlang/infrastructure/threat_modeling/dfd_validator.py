# -*- coding: utf-8 -*-
"""
Data Flow Diagram Validator - SEC-010 Phase 2

Validates data flow diagrams for completeness, consistency, and
security requirements. Checks trust boundaries, data classification,
and identifies missing controls.

Example:
    >>> from greenlang.infrastructure.threat_modeling import DataFlowValidator
    >>> validator = DataFlowValidator()
    >>> result = validator.validate_dfd(threat_model)
    >>> if not result.is_valid:
    ...     for error in result.errors:
    ...         print(error)

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
    Component,
    ComponentType,
    DataClassification,
    DataFlow,
    ThreatModel,
    TrustBoundary,
)
from greenlang.infrastructure.threat_modeling.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation Result Types
# ---------------------------------------------------------------------------


class ValidationSeverity(str, Enum):
    """Severity level for validation findings."""

    ERROR = "error"
    """Must be fixed before approval."""

    WARNING = "warning"
    """Should be reviewed but not blocking."""

    INFO = "info"
    """Informational finding."""


@dataclass
class ValidationFinding:
    """A single validation finding."""

    code: str
    severity: ValidationSeverity
    message: str
    component_id: Optional[str] = None
    data_flow_id: Optional[str] = None
    trust_boundary_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of DFD validation."""

    is_valid: bool
    error_count: int
    warning_count: int
    info_count: int
    findings: List[ValidationFinding]
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def errors(self) -> List[ValidationFinding]:
        """Return only error-level findings."""
        return [f for f in self.findings if f.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationFinding]:
        """Return only warning-level findings."""
        return [f for f in self.findings if f.severity == ValidationSeverity.WARNING]


@dataclass
class DFDReport:
    """Complete DFD compliance report."""

    service_name: str
    generated_at: datetime
    validation_result: ValidationResult
    trust_boundary_coverage: float
    data_classification_coverage: float
    control_coverage: float
    recommendations: List[str]
    summary: str


# ---------------------------------------------------------------------------
# Data Flow Validator Implementation
# ---------------------------------------------------------------------------


class DataFlowValidator:
    """Validates data flow diagrams for completeness and security.

    Checks trust boundaries, data classification, encryption requirements,
    and identifies missing controls in the threat model.

    Attributes:
        config: Threat modeling configuration.

    Example:
        >>> validator = DataFlowValidator()
        >>> result = validator.validate_dfd(threat_model)
        >>> print(f"Valid: {result.is_valid}, Errors: {result.error_count}")
    """

    def __init__(self) -> None:
        """Initialize the data flow validator."""
        self.config = get_config()
        logger.debug("Data flow validator initialized")

    def validate_dfd(self, threat_model: ThreatModel) -> ValidationResult:
        """Validate the data flow diagram for completeness.

        Performs comprehensive validation including:
        - Component connectivity
        - Data flow consistency
        - Trust boundary coverage
        - Data classification
        - Security controls

        Args:
            threat_model: The threat model to validate.

        Returns:
            ValidationResult with all findings.
        """
        logger.info("Validating DFD for service: %s", threat_model.service_name)
        findings: List[ValidationFinding] = []

        # Run all validation checks
        findings.extend(self._validate_components(threat_model))
        findings.extend(self._validate_data_flows(threat_model))
        findings.extend(self.check_trust_boundaries(threat_model))
        findings.extend(self.check_data_classification(threat_model))
        findings.extend(self.detect_missing_controls(threat_model))

        # Count by severity
        error_count = sum(1 for f in findings if f.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for f in findings if f.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for f in findings if f.severity == ValidationSeverity.INFO)

        # Model is valid if no errors
        is_valid = error_count == 0

        result = ValidationResult(
            is_valid=is_valid,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            findings=findings,
        )

        logger.info(
            "DFD validation complete: valid=%s, errors=%d, warnings=%d",
            is_valid,
            error_count,
            warning_count,
        )

        return result

    def check_trust_boundaries(self, threat_model: ThreatModel) -> List[ValidationFinding]:
        """Ensure all cross-boundary flows are identified.

        Validates that trust boundaries are properly defined and that
        all flows crossing boundaries are marked.

        Args:
            threat_model: The threat model to validate.

        Returns:
            List of validation findings.
        """
        findings: List[ValidationFinding] = []

        logger.debug("Checking trust boundaries")

        # Get all component IDs in boundaries
        components_in_boundaries: Set[str] = set()
        for boundary in threat_model.trust_boundaries:
            components_in_boundaries.update(boundary.component_ids)

        # Check for components not in any boundary
        orphan_components = [
            c for c in threat_model.components
            if c.id not in components_in_boundaries
            and c.component_type not in (ComponentType.USER, ComponentType.EXTERNAL_SERVICE)
        ]

        for component in orphan_components:
            findings.append(
                ValidationFinding(
                    code="TB001",
                    severity=ValidationSeverity.WARNING,
                    message=f"Component '{component.name}' is not within any trust boundary",
                    component_id=component.id,
                )
            )

        # Check that flows crossing boundaries are marked
        for flow in threat_model.data_flows:
            source_boundary = self._find_component_boundary(
                flow.source_component_id, threat_model.trust_boundaries
            )
            dest_boundary = self._find_component_boundary(
                flow.destination_component_id, threat_model.trust_boundaries
            )

            # Check if crossing boundary
            crosses_boundary = source_boundary != dest_boundary
            if crosses_boundary and not flow.crosses_trust_boundary:
                findings.append(
                    ValidationFinding(
                        code="TB002",
                        severity=ValidationSeverity.ERROR,
                        message=f"Data flow crosses trust boundary but is not marked as such",
                        data_flow_id=flow.id,
                        details={
                            "source_boundary": source_boundary,
                            "destination_boundary": dest_boundary,
                        },
                    )
                )

            # Flows crossing boundaries should have enhanced security
            if crosses_boundary or flow.crosses_trust_boundary:
                if not flow.encryption:
                    findings.append(
                        ValidationFinding(
                            code="TB003",
                            severity=ValidationSeverity.ERROR,
                            message="Cross-boundary flow must be encrypted",
                            data_flow_id=flow.id,
                        )
                    )
                if not flow.authentication_required:
                    findings.append(
                        ValidationFinding(
                            code="TB004",
                            severity=ValidationSeverity.WARNING,
                            message="Cross-boundary flow should require authentication",
                            data_flow_id=flow.id,
                        )
                    )

        # Check for empty trust boundaries
        for boundary in threat_model.trust_boundaries:
            if not boundary.component_ids:
                findings.append(
                    ValidationFinding(
                        code="TB005",
                        severity=ValidationSeverity.WARNING,
                        message=f"Trust boundary '{boundary.name}' has no components",
                        trust_boundary_id=boundary.id,
                    )
                )

        return findings

    def check_data_classification(self, threat_model: ThreatModel) -> List[ValidationFinding]:
        """Verify sensitive data is properly labeled.

        Checks that data classification is consistent across components
        and data flows, and that sensitive data has appropriate controls.

        Args:
            threat_model: The threat model to validate.

        Returns:
            List of validation findings.
        """
        findings: List[ValidationFinding] = []

        logger.debug("Checking data classification")

        # Check that all components have data classification
        for component in threat_model.components:
            # Skip user and external components
            if component.component_type in (ComponentType.USER, ComponentType.EXTERNAL_SERVICE):
                continue

            # Verify classification is not default for data stores
            if component.component_type in (ComponentType.DATABASE, ComponentType.CACHE, ComponentType.FILE_STORAGE):
                if component.data_classification == DataClassification.INTERNAL:
                    findings.append(
                        ValidationFinding(
                            code="DC001",
                            severity=ValidationSeverity.INFO,
                            message=f"Data store '{component.name}' has default classification - please verify",
                            component_id=component.id,
                        )
                    )

        # Check data flow classification consistency
        for flow in threat_model.data_flows:
            # Find destination component
            dest = next(
                (c for c in threat_model.components if c.id == flow.destination_component_id),
                None,
            )
            if dest and dest.data_classification != DataClassification.PUBLIC:
                # Flow should not downgrade classification
                if self._classification_level(flow.data_classification) > self._classification_level(dest.data_classification):
                    findings.append(
                        ValidationFinding(
                            code="DC002",
                            severity=ValidationSeverity.WARNING,
                            message=f"Data flow classification ({flow.data_classification.value}) exceeds destination component classification ({dest.data_classification.value})",
                            data_flow_id=flow.id,
                            component_id=dest.id,
                        )
                    )

        # Check that sensitive data has appropriate controls
        for flow in threat_model.data_flows:
            if flow.data_classification in (DataClassification.RESTRICTED, DataClassification.SECRET):
                if not flow.encryption:
                    findings.append(
                        ValidationFinding(
                            code="DC003",
                            severity=ValidationSeverity.ERROR,
                            message=f"Sensitive data ({flow.data_classification.value}) must be encrypted in transit",
                            data_flow_id=flow.id,
                        )
                    )
                if not flow.authentication_required:
                    findings.append(
                        ValidationFinding(
                            code="DC004",
                            severity=ValidationSeverity.ERROR,
                            message=f"Sensitive data ({flow.data_classification.value}) requires authentication",
                            data_flow_id=flow.id,
                        )
                    )
                if not flow.authorization_required:
                    findings.append(
                        ValidationFinding(
                            code="DC005",
                            severity=ValidationSeverity.ERROR,
                            message=f"Sensitive data ({flow.data_classification.value}) requires authorization",
                            data_flow_id=flow.id,
                        )
                    )

        return findings

    def detect_missing_controls(self, threat_model: ThreatModel) -> List[ValidationFinding]:
        """Identify gaps in security controls.

        Checks for missing security controls such as encryption,
        authentication, logging, and input validation.

        Args:
            threat_model: The threat model to validate.

        Returns:
            List of validation findings.
        """
        findings: List[ValidationFinding] = []

        logger.debug("Detecting missing controls")

        # Check for unencrypted PII flows
        for flow in threat_model.data_flows:
            data_type_lower = flow.data_type.lower()
            is_pii_flow = any(
                keyword in data_type_lower
                for keyword in ("user", "customer", "personal", "email", "phone", "address", "ssn", "credit")
            )

            if is_pii_flow and not flow.encryption:
                findings.append(
                    ValidationFinding(
                        code="MC001",
                        severity=ValidationSeverity.ERROR,
                        message=f"PII data flow '{flow.data_type}' is not encrypted",
                        data_flow_id=flow.id,
                    )
                )

        # Check for external-facing components without authentication
        for component in threat_model.components:
            if component.is_external or component.network_zone in ("dmz", "external"):
                # Find inbound flows
                inbound_flows = [
                    f for f in threat_model.data_flows
                    if f.destination_component_id == component.id
                ]
                unauthenticated_flows = [f for f in inbound_flows if not f.authentication_required]
                if unauthenticated_flows:
                    findings.append(
                        ValidationFinding(
                            code="MC002",
                            severity=ValidationSeverity.WARNING,
                            message=f"External-facing component '{component.name}' has {len(unauthenticated_flows)} unauthenticated inbound flows",
                            component_id=component.id,
                        )
                    )

        # Check for databases without encryption at rest (inferred from classification)
        for component in threat_model.components:
            if component.component_type == ComponentType.DATABASE:
                if component.data_classification in (DataClassification.RESTRICTED, DataClassification.SECRET):
                    # This is a placeholder - actual check would verify encryption config
                    findings.append(
                        ValidationFinding(
                            code="MC003",
                            severity=ValidationSeverity.INFO,
                            message=f"Verify that database '{component.name}' has encryption at rest enabled for {component.data_classification.value} data",
                            component_id=component.id,
                        )
                    )

        # Check for missing audit logging on sensitive operations
        sensitive_flows = [
            f for f in threat_model.data_flows
            if f.data_classification in (DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED, DataClassification.SECRET)
        ]
        if sensitive_flows:
            findings.append(
                ValidationFinding(
                    code="MC004",
                    severity=ValidationSeverity.INFO,
                    message=f"Verify audit logging is enabled for {len(sensitive_flows)} sensitive data flows",
                )
            )

        # Check for API components without rate limiting
        api_components = [
            c for c in threat_model.components
            if c.component_type in (ComponentType.API, ComponentType.WEB_APP)
        ]
        if api_components:
            findings.append(
                ValidationFinding(
                    code="MC005",
                    severity=ValidationSeverity.INFO,
                    message=f"Verify rate limiting is configured for {len(api_components)} API/web components",
                )
            )

        return findings

    def generate_dfd_report(self, threat_model: ThreatModel) -> DFDReport:
        """Generate a DFD compliance report.

        Creates a comprehensive report on DFD completeness and security.

        Args:
            threat_model: The threat model to analyze.

        Returns:
            Complete DFDReport.
        """
        logger.info("Generating DFD report for service: %s", threat_model.service_name)
        start_time = datetime.now(timezone.utc)

        # Run validation
        validation_result = self.validate_dfd(threat_model)

        # Calculate coverage metrics
        trust_boundary_coverage = self._calculate_trust_boundary_coverage(threat_model)
        data_classification_coverage = self._calculate_data_classification_coverage(threat_model)
        control_coverage = self._calculate_control_coverage(threat_model)

        # Generate recommendations
        recommendations = self._generate_recommendations(validation_result)

        # Generate summary
        if validation_result.is_valid:
            summary = f"DFD validation passed with {validation_result.warning_count} warnings."
        else:
            summary = f"DFD validation failed with {validation_result.error_count} errors and {validation_result.warning_count} warnings."

        report = DFDReport(
            service_name=threat_model.service_name,
            generated_at=start_time,
            validation_result=validation_result,
            trust_boundary_coverage=trust_boundary_coverage,
            data_classification_coverage=data_classification_coverage,
            control_coverage=control_coverage,
            recommendations=recommendations,
            summary=summary,
        )

        logger.info(
            "DFD report generated: valid=%s, coverage=(tb=%.1f%%, dc=%.1f%%, ctrl=%.1f%%)",
            validation_result.is_valid,
            trust_boundary_coverage * 100,
            data_classification_coverage * 100,
            control_coverage * 100,
        )

        return report

    def _validate_components(self, threat_model: ThreatModel) -> List[ValidationFinding]:
        """Validate component definitions.

        Args:
            threat_model: The threat model to validate.

        Returns:
            List of validation findings.
        """
        findings: List[ValidationFinding] = []

        # Check for minimum components
        if len(threat_model.components) < 2:
            findings.append(
                ValidationFinding(
                    code="CP001",
                    severity=ValidationSeverity.ERROR,
                    message="Threat model must have at least 2 components",
                )
            )

        # Check for duplicate component IDs
        component_ids = [c.id for c in threat_model.components]
        if len(component_ids) != len(set(component_ids)):
            findings.append(
                ValidationFinding(
                    code="CP002",
                    severity=ValidationSeverity.ERROR,
                    message="Duplicate component IDs detected",
                )
            )

        # Check for isolated components (no data flows)
        connected_components: Set[str] = set()
        for flow in threat_model.data_flows:
            connected_components.add(flow.source_component_id)
            connected_components.add(flow.destination_component_id)

        for component in threat_model.components:
            if component.id not in connected_components:
                # Users and external services might not have flows
                if component.component_type not in (ComponentType.USER, ComponentType.EXTERNAL_SERVICE, ComponentType.LOGGING):
                    findings.append(
                        ValidationFinding(
                            code="CP003",
                            severity=ValidationSeverity.WARNING,
                            message=f"Component '{component.name}' has no data flows",
                            component_id=component.id,
                        )
                    )

        return findings

    def _validate_data_flows(self, threat_model: ThreatModel) -> List[ValidationFinding]:
        """Validate data flow definitions.

        Args:
            threat_model: The threat model to validate.

        Returns:
            List of validation findings.
        """
        findings: List[ValidationFinding] = []

        component_ids = {c.id for c in threat_model.components}

        for flow in threat_model.data_flows:
            # Check that source component exists
            if flow.source_component_id not in component_ids:
                findings.append(
                    ValidationFinding(
                        code="DF001",
                        severity=ValidationSeverity.ERROR,
                        message=f"Data flow references non-existent source component",
                        data_flow_id=flow.id,
                        details={"source_component_id": flow.source_component_id},
                    )
                )

            # Check that destination component exists
            if flow.destination_component_id not in component_ids:
                findings.append(
                    ValidationFinding(
                        code="DF002",
                        severity=ValidationSeverity.ERROR,
                        message=f"Data flow references non-existent destination component",
                        data_flow_id=flow.id,
                        details={"destination_component_id": flow.destination_component_id},
                    )
                )

            # Check that data type is specified
            if not flow.data_type or flow.data_type == "generic":
                findings.append(
                    ValidationFinding(
                        code="DF003",
                        severity=ValidationSeverity.INFO,
                        message="Data flow has generic data type - consider specifying",
                        data_flow_id=flow.id,
                    )
                )

        return findings

    def _find_component_boundary(
        self,
        component_id: str,
        boundaries: List[TrustBoundary],
    ) -> Optional[str]:
        """Find which trust boundary contains a component.

        Args:
            component_id: The component ID to find.
            boundaries: List of trust boundaries.

        Returns:
            Boundary ID if found, None otherwise.
        """
        for boundary in boundaries:
            if component_id in boundary.component_ids:
                return boundary.id
        return None

    def _classification_level(self, classification: DataClassification) -> int:
        """Get numeric level for data classification.

        Args:
            classification: The data classification.

        Returns:
            Numeric level (higher = more sensitive).
        """
        levels = {
            DataClassification.PUBLIC: 1,
            DataClassification.INTERNAL: 2,
            DataClassification.CONFIDENTIAL: 3,
            DataClassification.RESTRICTED: 4,
            DataClassification.SECRET: 5,
        }
        return levels.get(classification, 0)

    def _calculate_trust_boundary_coverage(self, threat_model: ThreatModel) -> float:
        """Calculate percentage of components in trust boundaries.

        Args:
            threat_model: The threat model to analyze.

        Returns:
            Coverage percentage (0.0 to 1.0).
        """
        if not threat_model.components:
            return 1.0

        components_in_boundaries: Set[str] = set()
        for boundary in threat_model.trust_boundaries:
            components_in_boundaries.update(boundary.component_ids)

        # Exclude users and external services
        relevant_components = [
            c for c in threat_model.components
            if c.component_type not in (ComponentType.USER, ComponentType.EXTERNAL_SERVICE)
        ]

        if not relevant_components:
            return 1.0

        covered = sum(1 for c in relevant_components if c.id in components_in_boundaries)
        return covered / len(relevant_components)

    def _calculate_data_classification_coverage(self, threat_model: ThreatModel) -> float:
        """Calculate percentage of flows with non-default classification.

        Args:
            threat_model: The threat model to analyze.

        Returns:
            Coverage percentage (0.0 to 1.0).
        """
        if not threat_model.data_flows:
            return 1.0

        classified = sum(
            1 for f in threat_model.data_flows
            if f.data_classification != DataClassification.INTERNAL
        )
        return classified / len(threat_model.data_flows)

    def _calculate_control_coverage(self, threat_model: ThreatModel) -> float:
        """Calculate percentage of flows with security controls.

        Args:
            threat_model: The threat model to analyze.

        Returns:
            Coverage percentage (0.0 to 1.0).
        """
        if not threat_model.data_flows:
            return 1.0

        controlled = sum(
            1 for f in threat_model.data_flows
            if f.encryption and f.authentication_required
        )
        return controlled / len(threat_model.data_flows)

    def _generate_recommendations(self, validation_result: ValidationResult) -> List[str]:
        """Generate recommendations from validation findings.

        Args:
            validation_result: The validation result.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # Group findings by code prefix
        code_counts: Dict[str, int] = {}
        for finding in validation_result.findings:
            prefix = finding.code[:2]
            code_counts[prefix] = code_counts.get(prefix, 0) + 1

        if code_counts.get("TB", 0) > 0:
            recommendations.append("Review and update trust boundary definitions")

        if code_counts.get("DC", 0) > 0:
            recommendations.append("Complete data classification for all components and flows")

        if code_counts.get("MC", 0) > 0:
            recommendations.append("Implement missing security controls (encryption, authentication, logging)")

        if code_counts.get("DF", 0) > 0:
            recommendations.append("Fix data flow references and add missing data types")

        if code_counts.get("CP", 0) > 0:
            recommendations.append("Ensure all components are connected and properly defined")

        if not recommendations:
            recommendations.append("DFD is well-structured. Continue regular reviews.")

        return recommendations


__all__ = [
    "DataFlowValidator",
    "ValidationResult",
    "ValidationFinding",
    "ValidationSeverity",
    "DFDReport",
]
