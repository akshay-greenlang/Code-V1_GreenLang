"""
GreenLang Compliance Exceptions - Regulatory and Audit Trail Errors

This module provides exception classes for compliance-related errors
across supported regulatory frameworks including EUDR, CSRD, CBAM,
and general audit trail and provenance tracking.

Features:
- EUDR deforestation regulation violations
- CSRD sustainability reporting violations
- CBAM carbon border adjustment violations
- Regulatory deadline breaches
- Audit trail integrity errors
- Data provenance failures

Author: GreenLang Team
Date: 2026-04-02
"""

from typing import Any, Dict, List, Optional

from greenlang.exceptions.base import GreenLangException


class ComplianceException(GreenLangException):
    """Base exception for compliance-related errors.

    Raised when regulatory compliance checks, audit trail operations,
    or provenance tracking fails.
    """
    ERROR_PREFIX = "GL_COMPLIANCE"


class EUDRViolationError(ComplianceException):
    """EUDR (EU Deforestation Regulation) compliance violation.

    Raised when due diligence, traceability, or risk assessment
    requirements of the EUDR are not met.

    Example:
        >>> raise EUDRViolationError(
        ...     message="Geolocation data missing for commodity shipment",
        ...     commodity="palm_oil",
        ...     requirement="Article 9(1)(d)",
        ...     context={"shipment_id": "SHP-2026-001"}
        ... )
    """

    def __init__(
        self,
        message: str,
        commodity: Optional[str] = None,
        requirement: Optional[str] = None,
        risk_level: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize EUDR violation error.

        Args:
            message: Error message
            commodity: Commodity type (palm_oil, soy, coffee, cocoa, etc.)
            requirement: EUDR article or requirement reference
            risk_level: Risk assessment level (negligible, low, standard, high)
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if commodity:
            context["commodity"] = commodity
        if requirement:
            context["requirement"] = requirement
        if risk_level:
            context["risk_level"] = risk_level
        super().__init__(message, agent_name=agent_name, context=context)


class CSRDViolationError(ComplianceException):
    """CSRD (Corporate Sustainability Reporting Directive) compliance violation.

    Raised when CSRD reporting requirements, ESRS standards, or
    double materiality assessment criteria are not met.

    Example:
        >>> raise CSRDViolationError(
        ...     message="ESRS E1 Climate Change disclosure incomplete",
        ...     esrs_standard="E1",
        ...     disclosure_requirement="E1-6",
        ...     context={"missing_datapoints": ["Scope 3 Category 1"]}
        ... )
    """

    def __init__(
        self,
        message: str,
        esrs_standard: Optional[str] = None,
        disclosure_requirement: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize CSRD violation error.

        Args:
            message: Error message
            esrs_standard: ESRS standard reference (E1, E2, S1, G1, etc.)
            disclosure_requirement: Specific disclosure requirement ID
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if esrs_standard:
            context["esrs_standard"] = esrs_standard
        if disclosure_requirement:
            context["disclosure_requirement"] = disclosure_requirement
        super().__init__(message, agent_name=agent_name, context=context)


class CBAMViolationError(ComplianceException):
    """CBAM (Carbon Border Adjustment Mechanism) compliance violation.

    Raised when CBAM reporting, embedded emissions calculations, or
    certificate requirements are not met.

    Example:
        >>> raise CBAMViolationError(
        ...     message="Embedded emissions not verified for steel import",
        ...     cn_code="7208",
        ...     reporting_period="2026-Q1",
        ...     context={"importer": "Steel Corp EU"}
        ... )
    """

    def __init__(
        self,
        message: str,
        cn_code: Optional[str] = None,
        reporting_period: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize CBAM violation error.

        Args:
            message: Error message
            cn_code: Combined Nomenclature code for the product
            reporting_period: CBAM reporting period (e.g., "2026-Q1")
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if cn_code:
            context["cn_code"] = cn_code
        if reporting_period:
            context["reporting_period"] = reporting_period
        super().__init__(message, agent_name=agent_name, context=context)


class RegulatoryDeadlineError(ComplianceException):
    """Regulatory deadline has been missed or is at risk.

    Raised when a compliance reporting deadline is breached or when
    processing will not complete before a mandatory deadline.

    Example:
        >>> raise RegulatoryDeadlineError(
        ...     message="CBAM quarterly report deadline missed",
        ...     regulation="CBAM",
        ...     deadline="2026-04-30",
        ...     context={"days_overdue": 5}
        ... )
    """

    def __init__(
        self,
        message: str,
        regulation: Optional[str] = None,
        deadline: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize regulatory deadline error.

        Args:
            message: Error message
            regulation: Name of the regulation (CBAM, CSRD, EUDR)
            deadline: Deadline date (ISO 8601)
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if regulation:
            context["regulation"] = regulation
        if deadline:
            context["deadline"] = deadline
        super().__init__(message, agent_name=agent_name, context=context)


class AuditTrailError(ComplianceException):
    """Audit trail integrity or completeness failure.

    Raised when audit trail records are missing, tampered with, or
    cannot be written.

    Example:
        >>> raise AuditTrailError(
        ...     message="Audit trail gap detected",
        ...     audit_event="emission_calculation",
        ...     context={"gap_start": "2026-03-15", "gap_end": "2026-03-17"}
        ... )
    """

    def __init__(
        self,
        message: str,
        audit_event: Optional[str] = None,
        record_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize audit trail error.

        Args:
            message: Error message
            audit_event: Type of audit event
            record_id: ID of the affected audit record
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if audit_event:
            context["audit_event"] = audit_event
        if record_id:
            context["record_id"] = record_id
        super().__init__(message, agent_name=agent_name, context=context)


class ProvenanceError(ComplianceException):
    """Data provenance verification failed.

    Raised when SHA-256 provenance hash verification fails or when
    data lineage cannot be established.

    Example:
        >>> raise ProvenanceError(
        ...     message="Provenance hash mismatch",
        ...     expected_hash="abc123...",
        ...     actual_hash="def456...",
        ...     data_source="erp/sap"
        ... )
    """

    def __init__(
        self,
        message: str,
        expected_hash: Optional[str] = None,
        actual_hash: Optional[str] = None,
        data_source: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize provenance error.

        Args:
            message: Error message
            expected_hash: Expected SHA-256 hash
            actual_hash: Actual SHA-256 hash computed
            data_source: Source of the data being verified
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if expected_hash:
            context["expected_hash"] = expected_hash
        if actual_hash:
            context["actual_hash"] = actual_hash
        if data_source:
            context["data_source"] = data_source
        super().__init__(message, agent_name=agent_name, context=context)


__all__ = [
    'ComplianceException',
    'EUDRViolationError',
    'CSRDViolationError',
    'CBAMViolationError',
    'RegulatoryDeadlineError',
    'AuditTrailError',
    'ProvenanceError',
]
