# -*- coding: utf-8 -*-
"""
GreenLang Shared Enumerations
==============================

Centralized enumerations that are duplicated across 50+ agent model files.
Agent-specific enums (e.g. ``FuelType``, ``NodeType``) remain in their
respective agent modules. Only commonly-reused enums live here.

Usage::

    from greenlang.schemas.enums import CalculationStatus, JobStatus

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

from enum import Enum


# =============================================================================
# Job / Calculation Lifecycle
# =============================================================================


class CalculationStatus(str, Enum):
    """Lifecycle status of a calculation or computation task.

    Duplicated in 30+ MRV agents, 10+ pack engines.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(str, Enum):
    """Lifecycle status of an asynchronous processing job.

    Duplicated in 20+ data agents, 10+ EUDR agents.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingStatus(str, Enum):
    """Generic processing pipeline status.

    Use when neither CalculationStatus nor JobStatus fits exactly.
    """

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


# =============================================================================
# Severity / Priority
# =============================================================================


class Severity(str, Enum):
    """Severity level for validation findings, alerts, and issues."""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationSeverity(str, Enum):
    """Severity specifically for validation results.

    Duplicated in 15+ data/MRV agents.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Priority(str, Enum):
    """Priority levels for task scheduling and queue ordering."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Data Quality
# =============================================================================


class DataQualityLevel(str, Enum):
    """Data quality tier classification per GHG Protocol guidance.

    Duplicated across MRV agents and pack engines.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"
    DEFAULT = "default"


class MatchStatus(str, Enum):
    """Status of a data matching/reconciliation operation.

    Duplicated in duplicate_detector, cross_source_reconciliation,
    data_quality_profiler, and EUDR agents.
    """

    MATCHED = "matched"
    UNMATCHED = "unmatched"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    PENDING = "pending"


class ResolutionStatus(str, Enum):
    """Status of conflict/issue resolution.

    Duplicated in cross_source_reconciliation, validation_rule_engine.
    """

    PENDING = "pending"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


# =============================================================================
# Reporting
# =============================================================================


class ReportingPeriod(str, Enum):
    """Temporal granularity for emission/metric reporting aggregation.

    Duplicated in 30+ MRV agents.
    """

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class RegulatoryFramework(str, Enum):
    """Regulatory framework governing calculation methodology.

    Duplicated in 20+ MRV agents and 10+ pack engines.
    """

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS_E1 = "csrd_esrs_e1"
    EPA_40CFR98 = "epa_40cfr98"
    UK_SECR = "uk_secr"
    EU_ETS = "eu_ets"
    EUDR = "eudr"
    CBAM = "cbam"
    SBTi = "sbti"
    CDP = "cdp"
    TCFD = "tcfd"


# =============================================================================
# Risk / Compliance
# =============================================================================


class RiskLevel(str, Enum):
    """Risk assessment level.

    Duplicated in EUDR agents and compliance packs.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class ComplianceStatus(str, Enum):
    """Compliance assessment status.

    Duplicated in EUDR agents and compliance packs.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    NOT_ASSESSED = "not_assessed"


# =============================================================================
# Organizational Boundaries (GHG Protocol)
# =============================================================================


class ControlApproach(str, Enum):
    """Organizational boundary approach for emission ownership.

    Duplicated in MRV agents and GHG accounting packs.
    """

    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    EQUITY_SHARE = "equity_share"


# =============================================================================
# Units (commonly duplicated subset)
# =============================================================================


class EmissionUnit(str, Enum):
    """Standard emission reporting units.

    Duplicated in 30+ MRV agents.
    """

    KG_CO2E = "kg_co2e"
    TONNES_CO2E = "tonnes_co2e"
    MT_CO2E = "mt_co2e"
    KG = "kg"
    TONNES = "tonnes"
