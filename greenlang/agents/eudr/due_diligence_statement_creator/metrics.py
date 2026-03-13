# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-037: Due Diligence Statement Creator

40+ Prometheus metrics for DDS creation service monitoring with graceful
fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_ddsc_`` prefix for consistent
identification in Prometheus queries, Grafana dashboards, and alerting
rules across the GreenLang platform.

Metrics (40):
    Counters (14):
        1.  gl_eudr_ddsc_statements_created_total          - DDS created [type]
        2.  gl_eudr_ddsc_statements_submitted_total         - DDS submitted [status]
        3.  gl_eudr_ddsc_amendments_created_total           - Amendments created [reason]
        4.  gl_eudr_ddsc_validations_passed_total           - Validations passed
        5.  gl_eudr_ddsc_validations_failed_total           - Validations failed [field]
        6.  gl_eudr_ddsc_documents_packaged_total           - Documents packaged [type]
        7.  gl_eudr_ddsc_signatures_applied_total           - Signatures applied [type]
        8.  gl_eudr_ddsc_geolocations_formatted_total       - Geolocations formatted [method]
        9.  gl_eudr_ddsc_risk_integrations_total            - Risk data integrations [source]
        10. gl_eudr_ddsc_supply_chain_compilations_total    - Supply chain compilations
        11. gl_eudr_ddsc_versions_created_total             - Statement versions created
        12. gl_eudr_ddsc_withdrawals_total                  - Statements withdrawn
        13. gl_eudr_ddsc_translations_total                 - Translations created [language]
        14. gl_eudr_ddsc_batch_operations_total             - Batch operations completed

    Histograms (11):
        15. gl_eudr_ddsc_statement_generation_duration_seconds
        16. gl_eudr_ddsc_validation_duration_seconds
        17. gl_eudr_ddsc_geolocation_formatting_duration_seconds
        18. gl_eudr_ddsc_risk_integration_duration_seconds
        19. gl_eudr_ddsc_supply_chain_compilation_duration_seconds
        20. gl_eudr_ddsc_document_packaging_duration_seconds
        21. gl_eudr_ddsc_signing_duration_seconds
        22. gl_eudr_ddsc_submission_duration_seconds
        23. gl_eudr_ddsc_amendment_duration_seconds
        24. gl_eudr_ddsc_translation_duration_seconds [language]
        25. gl_eudr_ddsc_version_creation_duration_seconds

    Gauges (15):
        26. gl_eudr_ddsc_active_statements
        27. gl_eudr_ddsc_pending_submissions
        28. gl_eudr_ddsc_failed_validations
        29. gl_eudr_ddsc_total_commodity_volume
        30. gl_eudr_ddsc_draft_statements
        31. gl_eudr_ddsc_validated_statements
        32. gl_eudr_ddsc_signed_statements
        33. gl_eudr_ddsc_submitted_statements
        34. gl_eudr_ddsc_accepted_statements
        35. gl_eudr_ddsc_rejected_statements
        36. gl_eudr_ddsc_amended_statements
        37. gl_eudr_ddsc_withdrawn_statements
        38. gl_eudr_ddsc_total_documents
        39. gl_eudr_ddsc_total_geolocations
        40. gl_eudr_ddsc_average_risk_score

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-037 (GL-EUDR-DDSC-037)
Status: Production Ready
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus_client not available; metrics disabled")


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

if _PROMETHEUS_AVAILABLE:
    # Counters (14)
    _STATEMENTS_CREATED = Counter(
        "gl_eudr_ddsc_statements_created_total",
        "Due Diligence Statements created",
        ["type"],
    )
    _STATEMENTS_SUBMITTED = Counter(
        "gl_eudr_ddsc_statements_submitted_total",
        "Due Diligence Statements submitted to EU IS",
        ["status"],
    )
    _AMENDMENTS_CREATED = Counter(
        "gl_eudr_ddsc_amendments_created_total",
        "DDS amendments created",
        ["reason"],
    )
    _VALIDATIONS_PASSED = Counter(
        "gl_eudr_ddsc_validations_passed_total",
        "DDS validations that passed all checks",
    )
    _VALIDATIONS_FAILED = Counter(
        "gl_eudr_ddsc_validations_failed_total",
        "DDS validations that failed",
        ["field"],
    )
    _DOCUMENTS_PACKAGED = Counter(
        "gl_eudr_ddsc_documents_packaged_total",
        "Documents packaged into DDS evidence bundles",
        ["type"],
    )
    _SIGNATURES_APPLIED = Counter(
        "gl_eudr_ddsc_signatures_applied_total",
        "Digital signatures applied to DDS",
        ["type"],
    )
    _GEOLOCATIONS_FORMATTED = Counter(
        "gl_eudr_ddsc_geolocations_formatted_total",
        "Geolocation records formatted per Article 9",
        ["method"],
    )
    _RISK_INTEGRATIONS = Counter(
        "gl_eudr_ddsc_risk_integrations_total",
        "Risk data integrations from upstream agents",
        ["source"],
    )
    _SUPPLY_CHAIN_COMPILATIONS = Counter(
        "gl_eudr_ddsc_supply_chain_compilations_total",
        "Supply chain data compilations completed",
    )
    _VERSIONS_CREATED = Counter(
        "gl_eudr_ddsc_versions_created_total",
        "DDS version records created",
    )
    _WITHDRAWALS = Counter(
        "gl_eudr_ddsc_withdrawals_total",
        "DDS withdrawals processed",
    )
    _TRANSLATIONS = Counter(
        "gl_eudr_ddsc_translations_total",
        "DDS translations created",
        ["language"],
    )
    _BATCH_OPERATIONS = Counter(
        "gl_eudr_ddsc_batch_operations_total",
        "Batch operations completed",
    )

    # Histograms (11)
    _STATEMENT_GENERATION_DURATION = Histogram(
        "gl_eudr_ddsc_statement_generation_duration_seconds",
        "Full DDS generation latency from start to completion",
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    _VALIDATION_DURATION = Histogram(
        "gl_eudr_ddsc_validation_duration_seconds",
        "DDS compliance validation latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _GEOLOCATION_FORMATTING_DURATION = Histogram(
        "gl_eudr_ddsc_geolocation_formatting_duration_seconds",
        "Geolocation data formatting latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _RISK_INTEGRATION_DURATION = Histogram(
        "gl_eudr_ddsc_risk_integration_duration_seconds",
        "Risk data integration latency from upstream agents",
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    _SUPPLY_CHAIN_COMPILATION_DURATION = Histogram(
        "gl_eudr_ddsc_supply_chain_compilation_duration_seconds",
        "Supply chain data compilation latency",
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    _DOCUMENT_PACKAGING_DURATION = Histogram(
        "gl_eudr_ddsc_document_packaging_duration_seconds",
        "Document packaging latency for evidence bundles",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _SIGNING_DURATION = Histogram(
        "gl_eudr_ddsc_signing_duration_seconds",
        "Digital signing operation latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _SUBMISSION_DURATION = Histogram(
        "gl_eudr_ddsc_submission_duration_seconds",
        "EU IS submission latency",
        buckets=(1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    _AMENDMENT_DURATION = Histogram(
        "gl_eudr_ddsc_amendment_duration_seconds",
        "Amendment processing latency",
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _TRANSLATION_DURATION = Histogram(
        "gl_eudr_ddsc_translation_duration_seconds",
        "DDS translation latency",
        ["language"],
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _VERSION_CREATION_DURATION = Histogram(
        "gl_eudr_ddsc_version_creation_duration_seconds",
        "DDS version creation latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )

    # Gauges (15)
    _ACTIVE_STATEMENTS = Gauge(
        "gl_eudr_ddsc_active_statements",
        "Number of active DDS records",
    )
    _PENDING_SUBMISSIONS = Gauge(
        "gl_eudr_ddsc_pending_submissions",
        "Number of DDS pending submission to EU IS",
    )
    _FAILED_VALIDATIONS = Gauge(
        "gl_eudr_ddsc_failed_validations",
        "Number of DDS that failed validation",
    )
    _TOTAL_COMMODITY_VOLUME = Gauge(
        "gl_eudr_ddsc_total_commodity_volume",
        "Total commodity volume across all active DDS (metric tonnes)",
    )
    _DRAFT_STATEMENTS = Gauge(
        "gl_eudr_ddsc_draft_statements",
        "Number of DDS in draft status",
    )
    _VALIDATED_STATEMENTS = Gauge(
        "gl_eudr_ddsc_validated_statements",
        "Number of validated DDS",
    )
    _SIGNED_STATEMENTS = Gauge(
        "gl_eudr_ddsc_signed_statements",
        "Number of signed DDS",
    )
    _SUBMITTED_STATEMENTS = Gauge(
        "gl_eudr_ddsc_submitted_statements",
        "Number of submitted DDS",
    )
    _ACCEPTED_STATEMENTS = Gauge(
        "gl_eudr_ddsc_accepted_statements",
        "Number of accepted DDS",
    )
    _REJECTED_STATEMENTS = Gauge(
        "gl_eudr_ddsc_rejected_statements",
        "Number of rejected DDS",
    )
    _AMENDED_STATEMENTS = Gauge(
        "gl_eudr_ddsc_amended_statements",
        "Number of amended DDS",
    )
    _WITHDRAWN_STATEMENTS = Gauge(
        "gl_eudr_ddsc_withdrawn_statements",
        "Number of withdrawn DDS",
    )
    _TOTAL_DOCUMENTS = Gauge(
        "gl_eudr_ddsc_total_documents",
        "Total supporting documents across all DDS",
    )
    _TOTAL_GEOLOCATIONS = Gauge(
        "gl_eudr_ddsc_total_geolocations",
        "Total geolocation plot records across all DDS",
    )
    _AVERAGE_RISK_SCORE = Gauge(
        "gl_eudr_ddsc_average_risk_score",
        "Average risk score across all active DDS",
    )


# ---------------------------------------------------------------------------
# Helper Functions - Counters
# ---------------------------------------------------------------------------


def record_statement_created(statement_type: str) -> None:
    """Record a DDS creation metric."""
    if _PROMETHEUS_AVAILABLE:
        _STATEMENTS_CREATED.labels(type=statement_type).inc()


def record_statement_submitted(status: str) -> None:
    """Record a DDS submission metric."""
    if _PROMETHEUS_AVAILABLE:
        _STATEMENTS_SUBMITTED.labels(status=status).inc()


def record_amendment_created(reason: str) -> None:
    """Record an amendment creation metric."""
    if _PROMETHEUS_AVAILABLE:
        _AMENDMENTS_CREATED.labels(reason=reason).inc()


def record_validation_passed() -> None:
    """Record a validation pass metric."""
    if _PROMETHEUS_AVAILABLE:
        _VALIDATIONS_PASSED.inc()


def record_validation_failed(field: str) -> None:
    """Record a validation failure metric."""
    if _PROMETHEUS_AVAILABLE:
        _VALIDATIONS_FAILED.labels(field=field).inc()


def record_document_packaged(document_type: str) -> None:
    """Record a document packaging metric."""
    if _PROMETHEUS_AVAILABLE:
        _DOCUMENTS_PACKAGED.labels(type=document_type).inc()


def record_signature_applied(signature_type: str) -> None:
    """Record a signature application metric."""
    if _PROMETHEUS_AVAILABLE:
        _SIGNATURES_APPLIED.labels(type=signature_type).inc()


def record_geolocation_formatted(method: str) -> None:
    """Record a geolocation formatting metric."""
    if _PROMETHEUS_AVAILABLE:
        _GEOLOCATIONS_FORMATTED.labels(method=method).inc()


def record_risk_integration(source: str) -> None:
    """Record a risk data integration metric."""
    if _PROMETHEUS_AVAILABLE:
        _RISK_INTEGRATIONS.labels(source=source).inc()


def record_supply_chain_compilation() -> None:
    """Record a supply chain compilation metric."""
    if _PROMETHEUS_AVAILABLE:
        _SUPPLY_CHAIN_COMPILATIONS.inc()


def record_version_created() -> None:
    """Record a version creation metric."""
    if _PROMETHEUS_AVAILABLE:
        _VERSIONS_CREATED.inc()


def record_withdrawal() -> None:
    """Record a DDS withdrawal metric."""
    if _PROMETHEUS_AVAILABLE:
        _WITHDRAWALS.inc()


def record_translation(language: str) -> None:
    """Record a translation creation metric."""
    if _PROMETHEUS_AVAILABLE:
        _TRANSLATIONS.labels(language=language).inc()


def record_batch_operation() -> None:
    """Record a batch operation metric."""
    if _PROMETHEUS_AVAILABLE:
        _BATCH_OPERATIONS.inc()


# ---------------------------------------------------------------------------
# Helper Functions - Histograms
# ---------------------------------------------------------------------------


def observe_statement_generation_duration(duration: float) -> None:
    """Observe full DDS generation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _STATEMENT_GENERATION_DURATION.observe(duration)


def observe_validation_duration(duration: float) -> None:
    """Observe DDS validation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _VALIDATION_DURATION.observe(duration)


def observe_geolocation_formatting_duration(duration: float) -> None:
    """Observe geolocation formatting duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _GEOLOCATION_FORMATTING_DURATION.observe(duration)


def observe_risk_integration_duration(duration: float) -> None:
    """Observe risk data integration duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _RISK_INTEGRATION_DURATION.observe(duration)


def observe_supply_chain_compilation_duration(duration: float) -> None:
    """Observe supply chain compilation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _SUPPLY_CHAIN_COMPILATION_DURATION.observe(duration)


def observe_document_packaging_duration(duration: float) -> None:
    """Observe document packaging duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _DOCUMENT_PACKAGING_DURATION.observe(duration)


def observe_signing_duration(duration: float) -> None:
    """Observe digital signing duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _SIGNING_DURATION.observe(duration)


def observe_submission_duration(duration: float) -> None:
    """Observe EU IS submission duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _SUBMISSION_DURATION.observe(duration)


def observe_amendment_duration(duration: float) -> None:
    """Observe amendment processing duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _AMENDMENT_DURATION.observe(duration)


def observe_translation_duration(language: str, duration: float) -> None:
    """Observe DDS translation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _TRANSLATION_DURATION.labels(language=language).observe(duration)


def observe_version_creation_duration(duration: float) -> None:
    """Observe version creation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _VERSION_CREATION_DURATION.observe(duration)


# ---------------------------------------------------------------------------
# Helper Functions - Gauges
# ---------------------------------------------------------------------------


def set_active_statements(count: int) -> None:
    """Set gauge of active DDS records."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_STATEMENTS.set(count)


def set_pending_submissions(count: int) -> None:
    """Set gauge of DDS pending submission."""
    if _PROMETHEUS_AVAILABLE:
        _PENDING_SUBMISSIONS.set(count)


def set_failed_validations(count: int) -> None:
    """Set gauge of failed validations."""
    if _PROMETHEUS_AVAILABLE:
        _FAILED_VALIDATIONS.set(count)


def set_total_commodity_volume(volume: float) -> None:
    """Set gauge of total commodity volume."""
    if _PROMETHEUS_AVAILABLE:
        _TOTAL_COMMODITY_VOLUME.set(volume)


def set_draft_statements(count: int) -> None:
    """Set gauge of draft DDS count."""
    if _PROMETHEUS_AVAILABLE:
        _DRAFT_STATEMENTS.set(count)


def set_validated_statements(count: int) -> None:
    """Set gauge of validated DDS count."""
    if _PROMETHEUS_AVAILABLE:
        _VALIDATED_STATEMENTS.set(count)


def set_signed_statements(count: int) -> None:
    """Set gauge of signed DDS count."""
    if _PROMETHEUS_AVAILABLE:
        _SIGNED_STATEMENTS.set(count)


def set_submitted_statements(count: int) -> None:
    """Set gauge of submitted DDS count."""
    if _PROMETHEUS_AVAILABLE:
        _SUBMITTED_STATEMENTS.set(count)


def set_accepted_statements(count: int) -> None:
    """Set gauge of accepted DDS count."""
    if _PROMETHEUS_AVAILABLE:
        _ACCEPTED_STATEMENTS.set(count)


def set_rejected_statements(count: int) -> None:
    """Set gauge of rejected DDS count."""
    if _PROMETHEUS_AVAILABLE:
        _REJECTED_STATEMENTS.set(count)


def set_amended_statements(count: int) -> None:
    """Set gauge of amended DDS count."""
    if _PROMETHEUS_AVAILABLE:
        _AMENDED_STATEMENTS.set(count)


def set_withdrawn_statements(count: int) -> None:
    """Set gauge of withdrawn DDS count."""
    if _PROMETHEUS_AVAILABLE:
        _WITHDRAWN_STATEMENTS.set(count)


def set_total_documents(count: int) -> None:
    """Set gauge of total documents in packages."""
    if _PROMETHEUS_AVAILABLE:
        _TOTAL_DOCUMENTS.set(count)


def set_total_geolocations(count: int) -> None:
    """Set gauge of total geolocation plots."""
    if _PROMETHEUS_AVAILABLE:
        _TOTAL_GEOLOCATIONS.set(count)


def set_average_risk_score(score: float) -> None:
    """Set gauge of average risk score."""
    if _PROMETHEUS_AVAILABLE:
        _AVERAGE_RISK_SCORE.set(score)
