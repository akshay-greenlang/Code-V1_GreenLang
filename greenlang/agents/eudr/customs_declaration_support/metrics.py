# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-039: Customs Declaration Support

40+ Prometheus metrics for customs declaration service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_cds_`` prefix for consistent
identification in Prometheus queries, Grafana dashboards, and alerting
rules across the GreenLang platform.

Metrics (42 per PRD Section 7.6):
    Counters (16):
        1.  gl_eudr_cds_declarations_created_total           - Declarations created [type]
        2.  gl_eudr_cds_declarations_submitted_total         - Declarations submitted [system]
        3.  gl_eudr_cds_declarations_cleared_total           - Declarations cleared [type]
        4.  gl_eudr_cds_declarations_rejected_total          - Declarations rejected [reason]
        5.  gl_eudr_cds_compliance_checks_passed_total       - Compliance checks passed [check_type]
        6.  gl_eudr_cds_compliance_checks_failed_total       - Compliance checks failed [check_type]
        7.  gl_eudr_cds_tariffs_calculated_total             - Tariff calculations completed [tariff_type]
        8.  gl_eudr_cds_cn_codes_mapped_total                - CN code mappings performed [commodity]
        9.  gl_eudr_cds_hs_codes_validated_total             - HS code validations performed
        10. gl_eudr_cds_origin_verifications_total           - Origin verifications performed [status]
        11. gl_eudr_cds_value_calculations_total             - Value calculations performed [incoterms]
        12. gl_eudr_cds_submissions_retried_total            - Submission retries [system]
        13. gl_eudr_cds_mrn_assigned_total                   - MRNs assigned by customs
        14. gl_eudr_cds_sad_forms_generated_total            - SAD forms generated [form_type]
        15. gl_eudr_cds_amendments_total                     - Declaration amendments [type]
        16. gl_eudr_cds_currency_conversions_total           - Currency conversions performed [currency]

    Histograms (12):
        17. gl_eudr_cds_declaration_generation_duration_seconds - Declaration generation latency
        18. gl_eudr_cds_submission_duration_seconds           - Customs submission latency [system]
        19. gl_eudr_cds_compliance_check_duration_seconds     - Compliance check latency [check_type]
        20. gl_eudr_cds_tariff_calculation_duration_seconds   - Tariff calculation latency
        21. gl_eudr_cds_value_calculation_duration_seconds    - Value calculation latency
        22. gl_eudr_cds_cn_code_mapping_duration_seconds      - CN code mapping latency
        23. gl_eudr_cds_hs_code_validation_duration_seconds   - HS code validation latency
        24. gl_eudr_cds_origin_verification_duration_seconds  - Origin verification latency
        25. gl_eudr_cds_sad_generation_duration_seconds        - SAD form generation latency
        26. gl_eudr_cds_clearance_duration_seconds             - Clearance processing latency
        27. gl_eudr_cds_currency_conversion_duration_seconds   - Currency conversion latency
        28. gl_eudr_cds_customs_response_duration_seconds      - Customs system response latency [system]

    Gauges (14):
        29. gl_eudr_cds_pending_declarations                  - Pending declarations count
        30. gl_eudr_cds_declarations_awaiting_clearance       - Declarations awaiting clearance
        31. gl_eudr_cds_average_tariff_rate                   - Average tariff rate applied
        32. gl_eudr_cds_total_customs_value_eur               - Total customs value in EUR
        33. gl_eudr_cds_total_duty_amount_eur                 - Total duty amount in EUR
        34. gl_eudr_cds_active_submissions                    - Active submission count
        35. gl_eudr_cds_compliance_pass_rate                  - EUDR compliance pass rate
        36. gl_eudr_cds_declarations_by_status                - Declarations by status [status]
        37. gl_eudr_cds_declarations_by_commodity             - Declarations by commodity [commodity]
        38. gl_eudr_cds_exchange_rate_eur_usd                 - EUR/USD exchange rate
        39. gl_eudr_cds_exchange_rate_eur_gbp                 - EUR/GBP exchange rate
        40. gl_eudr_cds_exchange_rate_eur_jpy                 - EUR/JPY exchange rate
        41. gl_eudr_cds_submission_error_rate                 - Submission error rate
        42. gl_eudr_cds_average_clearance_time_hours          - Average clearance time in hours

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-039 (GL-EUDR-CDS-039)
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
    # Counters (16)
    _DECLARATIONS_CREATED = Counter(
        "gl_eudr_cds_declarations_created_total",
        "Customs declarations created",
        ["type"],
    )
    _DECLARATIONS_SUBMITTED = Counter(
        "gl_eudr_cds_declarations_submitted_total",
        "Customs declarations submitted to authorities",
        ["system"],
    )
    _DECLARATIONS_CLEARED = Counter(
        "gl_eudr_cds_declarations_cleared_total",
        "Customs declarations cleared",
        ["type"],
    )
    _DECLARATIONS_REJECTED = Counter(
        "gl_eudr_cds_declarations_rejected_total",
        "Customs declarations rejected",
        ["reason"],
    )
    _COMPLIANCE_CHECKS_PASSED = Counter(
        "gl_eudr_cds_compliance_checks_passed_total",
        "EUDR compliance checks that passed",
        ["check_type"],
    )
    _COMPLIANCE_CHECKS_FAILED = Counter(
        "gl_eudr_cds_compliance_checks_failed_total",
        "EUDR compliance checks that failed",
        ["check_type"],
    )
    _TARIFFS_CALCULATED = Counter(
        "gl_eudr_cds_tariffs_calculated_total",
        "Tariff calculations completed",
        ["tariff_type"],
    )
    _CN_CODES_MAPPED = Counter(
        "gl_eudr_cds_cn_codes_mapped_total",
        "CN code mappings performed",
        ["commodity"],
    )
    _HS_CODES_VALIDATED = Counter(
        "gl_eudr_cds_hs_codes_validated_total",
        "HS code validations performed",
    )
    _ORIGIN_VERIFICATIONS = Counter(
        "gl_eudr_cds_origin_verifications_total",
        "Country of origin verifications",
        ["status"],
    )
    _VALUE_CALCULATIONS = Counter(
        "gl_eudr_cds_value_calculations_total",
        "Customs value calculations performed",
        ["incoterms"],
    )
    _SUBMISSIONS_RETRIED = Counter(
        "gl_eudr_cds_submissions_retried_total",
        "Customs submission retry attempts",
        ["system"],
    )
    _MRN_ASSIGNED = Counter(
        "gl_eudr_cds_mrn_assigned_total",
        "Movement Reference Numbers assigned by customs",
    )
    _SAD_FORMS_GENERATED = Counter(
        "gl_eudr_cds_sad_forms_generated_total",
        "SAD forms generated",
        ["form_type"],
    )
    _AMENDMENTS = Counter(
        "gl_eudr_cds_amendments_total",
        "Declaration amendments processed",
        ["type"],
    )
    _CURRENCY_CONVERSIONS = Counter(
        "gl_eudr_cds_currency_conversions_total",
        "Currency conversions performed",
        ["currency"],
    )

    # Histograms (12)
    _DECLARATION_GENERATION_DURATION = Histogram(
        "gl_eudr_cds_declaration_generation_duration_seconds",
        "Customs declaration generation latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _SUBMISSION_DURATION = Histogram(
        "gl_eudr_cds_submission_duration_seconds",
        "Customs system submission latency",
        ["system"],
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0),
    )
    _COMPLIANCE_CHECK_DURATION = Histogram(
        "gl_eudr_cds_compliance_check_duration_seconds",
        "EUDR compliance check latency",
        ["check_type"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _TARIFF_CALCULATION_DURATION = Histogram(
        "gl_eudr_cds_tariff_calculation_duration_seconds",
        "Tariff calculation latency",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _VALUE_CALCULATION_DURATION = Histogram(
        "gl_eudr_cds_value_calculation_duration_seconds",
        "Customs value calculation latency",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _CN_CODE_MAPPING_DURATION = Histogram(
        "gl_eudr_cds_cn_code_mapping_duration_seconds",
        "CN code mapping latency",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
    )
    _HS_CODE_VALIDATION_DURATION = Histogram(
        "gl_eudr_cds_hs_code_validation_duration_seconds",
        "HS code validation latency",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
    )
    _ORIGIN_VERIFICATION_DURATION = Histogram(
        "gl_eudr_cds_origin_verification_duration_seconds",
        "Origin verification latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _SAD_GENERATION_DURATION = Histogram(
        "gl_eudr_cds_sad_generation_duration_seconds",
        "SAD form generation latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _CLEARANCE_DURATION = Histogram(
        "gl_eudr_cds_clearance_duration_seconds",
        "Customs clearance processing latency",
        buckets=(
            3600.0, 14400.0, 43200.0, 86400.0, 172800.0,
            604800.0,
        ),
    )
    _CURRENCY_CONVERSION_DURATION = Histogram(
        "gl_eudr_cds_currency_conversion_duration_seconds",
        "Currency conversion latency",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
    )
    _CUSTOMS_RESPONSE_DURATION = Histogram(
        "gl_eudr_cds_customs_response_duration_seconds",
        "Customs system response latency",
        ["system"],
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0),
    )

    # Gauges (14)
    _PENDING_DECLARATIONS = Gauge(
        "gl_eudr_cds_pending_declarations",
        "Number of pending customs declarations",
    )
    _DECLARATIONS_AWAITING_CLEARANCE = Gauge(
        "gl_eudr_cds_declarations_awaiting_clearance",
        "Declarations awaiting customs clearance",
    )
    _AVERAGE_TARIFF_RATE = Gauge(
        "gl_eudr_cds_average_tariff_rate",
        "Average tariff rate applied (percent)",
    )
    _TOTAL_CUSTOMS_VALUE_EUR = Gauge(
        "gl_eudr_cds_total_customs_value_eur",
        "Total customs value in EUR",
    )
    _TOTAL_DUTY_AMOUNT_EUR = Gauge(
        "gl_eudr_cds_total_duty_amount_eur",
        "Total duty amount in EUR",
    )
    _ACTIVE_SUBMISSIONS = Gauge(
        "gl_eudr_cds_active_submissions",
        "Number of active submission processes",
    )
    _COMPLIANCE_PASS_RATE = Gauge(
        "gl_eudr_cds_compliance_pass_rate",
        "EUDR compliance pass rate (0-1)",
    )
    _DECLARATIONS_BY_STATUS = Gauge(
        "gl_eudr_cds_declarations_by_status",
        "Declarations count by status",
        ["status"],
    )
    _DECLARATIONS_BY_COMMODITY = Gauge(
        "gl_eudr_cds_declarations_by_commodity",
        "Declarations count by EUDR commodity",
        ["commodity"],
    )
    _EXCHANGE_RATE_EUR_USD = Gauge(
        "gl_eudr_cds_exchange_rate_eur_usd",
        "EUR/USD exchange rate",
    )
    _EXCHANGE_RATE_EUR_GBP = Gauge(
        "gl_eudr_cds_exchange_rate_eur_gbp",
        "EUR/GBP exchange rate",
    )
    _EXCHANGE_RATE_EUR_JPY = Gauge(
        "gl_eudr_cds_exchange_rate_eur_jpy",
        "EUR/JPY exchange rate",
    )
    _SUBMISSION_ERROR_RATE = Gauge(
        "gl_eudr_cds_submission_error_rate",
        "Customs submission error rate (0-1)",
    )
    _AVERAGE_CLEARANCE_TIME_HOURS = Gauge(
        "gl_eudr_cds_average_clearance_time_hours",
        "Average customs clearance time in hours",
    )


# ---------------------------------------------------------------------------
# Helper Functions - Counters
# ---------------------------------------------------------------------------


def record_declaration_created(declaration_type: str) -> None:
    """Record a declaration creation metric."""
    if _PROMETHEUS_AVAILABLE:
        _DECLARATIONS_CREATED.labels(type=declaration_type).inc()


def record_declaration_submitted(customs_system: str) -> None:
    """Record a declaration submission metric."""
    if _PROMETHEUS_AVAILABLE:
        _DECLARATIONS_SUBMITTED.labels(system=customs_system).inc()


def record_declaration_cleared(declaration_type: str = "import") -> None:
    """Record a declaration clearance metric."""
    if _PROMETHEUS_AVAILABLE:
        _DECLARATIONS_CLEARED.labels(type=declaration_type).inc()


def record_declaration_rejected(reason: str) -> None:
    """Record a declaration rejection metric."""
    if _PROMETHEUS_AVAILABLE:
        _DECLARATIONS_REJECTED.labels(reason=reason).inc()


def record_compliance_check_passed(check_type: str) -> None:
    """Record a compliance check pass metric."""
    if _PROMETHEUS_AVAILABLE:
        _COMPLIANCE_CHECKS_PASSED.labels(check_type=check_type).inc()


def record_compliance_check_failed(check_type: str) -> None:
    """Record a compliance check failure metric."""
    if _PROMETHEUS_AVAILABLE:
        _COMPLIANCE_CHECKS_FAILED.labels(check_type=check_type).inc()


def record_tariff_calculated(tariff_type: str = "standard") -> None:
    """Record a tariff calculation metric."""
    if _PROMETHEUS_AVAILABLE:
        _TARIFFS_CALCULATED.labels(tariff_type=tariff_type).inc()


def record_cn_code_mapped(commodity: str) -> None:
    """Record a CN code mapping metric."""
    if _PROMETHEUS_AVAILABLE:
        _CN_CODES_MAPPED.labels(commodity=commodity).inc()


def record_hs_code_validated(hs_code: str = "") -> None:
    """Record an HS code validation metric."""
    if _PROMETHEUS_AVAILABLE:
        _HS_CODES_VALIDATED.inc()


def record_origin_verification(status: str) -> None:
    """Record an origin verification metric."""
    if _PROMETHEUS_AVAILABLE:
        _ORIGIN_VERIFICATIONS.labels(status=status).inc()


def record_value_calculation(incoterms: str) -> None:
    """Record a value calculation metric."""
    if _PROMETHEUS_AVAILABLE:
        _VALUE_CALCULATIONS.labels(incoterms=incoterms).inc()


def record_submission_retried(customs_system: str) -> None:
    """Record a submission retry metric."""
    if _PROMETHEUS_AVAILABLE:
        _SUBMISSIONS_RETRIED.labels(system=customs_system).inc()


def record_mrn_assigned() -> None:
    """Record an MRN assignment metric."""
    if _PROMETHEUS_AVAILABLE:
        _MRN_ASSIGNED.inc()


def record_sad_form_generated(form_type: str = "IM") -> None:
    """Record a SAD form generation metric."""
    if _PROMETHEUS_AVAILABLE:
        _SAD_FORMS_GENERATED.labels(form_type=form_type).inc()


def record_amendment(amendment_type: str) -> None:
    """Record a declaration amendment metric."""
    if _PROMETHEUS_AVAILABLE:
        _AMENDMENTS.labels(type=amendment_type).inc()


def record_currency_conversion(from_currency: str, to_currency: str = "EUR") -> None:
    """Record a currency conversion metric."""
    if _PROMETHEUS_AVAILABLE:
        _CURRENCY_CONVERSIONS.labels(currency=from_currency).inc()


# ---------------------------------------------------------------------------
# Helper Functions - Histograms
# ---------------------------------------------------------------------------


def observe_declaration_generation_duration(duration: float) -> None:
    """Observe declaration generation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _DECLARATION_GENERATION_DURATION.observe(duration)


def observe_submission_duration(customs_system: str, duration: float) -> None:
    """Observe customs submission duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _SUBMISSION_DURATION.labels(system=customs_system).observe(duration)


def observe_compliance_check_duration(
    duration_or_check_type: float | str = 0.0,
    duration: float = 0.0,
) -> None:
    """Observe compliance check duration in seconds.

    Can be called with:
        observe_compliance_check_duration(2.5)  # duration only
        observe_compliance_check_duration("dds", 2.5)  # check_type + duration
    """
    if isinstance(duration_or_check_type, (int, float)):
        actual_duration = float(duration_or_check_type)
        check_type = "general"
    else:
        check_type = duration_or_check_type
        actual_duration = duration
    if _PROMETHEUS_AVAILABLE:
        _COMPLIANCE_CHECK_DURATION.labels(check_type=check_type).observe(
            actual_duration
        )


def observe_tariff_calculation_duration(duration: float) -> None:
    """Observe tariff calculation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _TARIFF_CALCULATION_DURATION.observe(duration)


def observe_value_calculation_duration(duration: float) -> None:
    """Observe value calculation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _VALUE_CALCULATION_DURATION.observe(duration)


def observe_cn_code_mapping_duration(duration: float) -> None:
    """Observe CN code mapping duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _CN_CODE_MAPPING_DURATION.observe(duration)


def observe_hs_code_validation_duration(duration: float) -> None:
    """Observe HS code validation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _HS_CODE_VALIDATION_DURATION.observe(duration)


def observe_origin_verification_duration(duration: float) -> None:
    """Observe origin verification duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _ORIGIN_VERIFICATION_DURATION.observe(duration)


def observe_sad_generation_duration(duration: float) -> None:
    """Observe SAD form generation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _SAD_GENERATION_DURATION.observe(duration)


def observe_clearance_duration(duration: float) -> None:
    """Observe customs clearance processing duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _CLEARANCE_DURATION.observe(duration)


def observe_currency_conversion_duration(duration: float) -> None:
    """Observe currency conversion duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _CURRENCY_CONVERSION_DURATION.observe(duration)


def observe_customs_response_duration(
    customs_system: str, duration: float,
) -> None:
    """Observe customs system response duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _CUSTOMS_RESPONSE_DURATION.labels(system=customs_system).observe(
            duration
        )


# ---------------------------------------------------------------------------
# Helper Functions - Gauges
# ---------------------------------------------------------------------------


def set_pending_declarations(count: int) -> None:
    """Set gauge of pending declarations."""
    if _PROMETHEUS_AVAILABLE:
        _PENDING_DECLARATIONS.set(count)


def set_declarations_awaiting_clearance(count: int) -> None:
    """Set gauge of declarations awaiting clearance."""
    if _PROMETHEUS_AVAILABLE:
        _DECLARATIONS_AWAITING_CLEARANCE.set(count)


def set_average_tariff_rate(rate: float) -> None:
    """Set gauge of average tariff rate."""
    if _PROMETHEUS_AVAILABLE:
        _AVERAGE_TARIFF_RATE.set(rate)


def set_total_customs_value_eur(value: float) -> None:
    """Set gauge of total customs value in EUR."""
    if _PROMETHEUS_AVAILABLE:
        _TOTAL_CUSTOMS_VALUE_EUR.set(value)


def set_total_duty_amount_eur(value: float) -> None:
    """Set gauge of total duty amount in EUR."""
    if _PROMETHEUS_AVAILABLE:
        _TOTAL_DUTY_AMOUNT_EUR.set(value)


def set_active_submissions(count: int) -> None:
    """Set gauge of active submissions."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_SUBMISSIONS.set(count)


def set_compliance_pass_rate(rate: float) -> None:
    """Set gauge of EUDR compliance pass rate."""
    if _PROMETHEUS_AVAILABLE:
        _COMPLIANCE_PASS_RATE.set(rate)


def set_declarations_by_status(status: str, count: int) -> None:
    """Set gauge of declarations by status."""
    if _PROMETHEUS_AVAILABLE:
        _DECLARATIONS_BY_STATUS.labels(status=status).set(count)


def set_declarations_by_commodity(commodity: str, count: int) -> None:
    """Set gauge of declarations by commodity."""
    if _PROMETHEUS_AVAILABLE:
        _DECLARATIONS_BY_COMMODITY.labels(commodity=commodity).set(count)


def set_exchange_rate_eur_usd(rate: float) -> None:
    """Set gauge of EUR/USD exchange rate."""
    if _PROMETHEUS_AVAILABLE:
        _EXCHANGE_RATE_EUR_USD.set(rate)


def set_exchange_rate_eur_gbp(rate: float) -> None:
    """Set gauge of EUR/GBP exchange rate."""
    if _PROMETHEUS_AVAILABLE:
        _EXCHANGE_RATE_EUR_GBP.set(rate)


def set_exchange_rate_eur_jpy(rate: float) -> None:
    """Set gauge of EUR/JPY exchange rate."""
    if _PROMETHEUS_AVAILABLE:
        _EXCHANGE_RATE_EUR_JPY.set(rate)


def set_submission_error_rate(rate: float) -> None:
    """Set gauge of submission error rate."""
    if _PROMETHEUS_AVAILABLE:
        _SUBMISSION_ERROR_RATE.set(rate)


def set_average_clearance_time_hours(hours: float) -> None:
    """Set gauge of average clearance time in hours."""
    if _PROMETHEUS_AVAILABLE:
        _AVERAGE_CLEARANCE_TIME_HOURS.set(hours)


# ---------------------------------------------------------------------------
# Alias / Convenience Functions
# ---------------------------------------------------------------------------


def record_origin_verified(status: str) -> None:
    """Alias for record_origin_verification."""
    record_origin_verification(status)


def record_compliance_check(result: str) -> None:
    """Record a compliance check result (pass/fail/warning)."""
    if result in ("pass", "PASS"):
        record_compliance_check_passed(result)
    else:
        record_compliance_check_failed(result)


def record_mrn_generated() -> None:
    """Alias for record_mrn_assigned."""
    record_mrn_assigned()


def record_customs_submission(system: str) -> None:
    """Record a customs submission to a specific system."""
    record_declaration_submitted(system)


def record_batch_operation() -> None:
    """Record a batch operation."""
    if _PROMETHEUS_AVAILABLE:
        _DECLARATIONS_CREATED.labels(type="batch").inc()


def observe_customs_submission_duration(duration: float) -> None:
    """Observe customs submission duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _SUBMISSION_DURATION.labels(system="general").observe(duration)


def observe_sad_form_generation_duration(duration: float) -> None:
    """Alias for observe_sad_generation_duration."""
    observe_sad_generation_duration(duration)


def observe_mrn_generation_duration(duration: float) -> None:
    """Observe MRN generation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _DECLARATION_GENERATION_DURATION.observe(duration)


def observe_batch_processing_duration(duration: float) -> None:
    """Observe batch processing duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _DECLARATION_GENERATION_DURATION.observe(duration)


def set_active_declarations(count: int) -> None:
    """Set gauge of active declarations."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_SUBMISSIONS.set(count)


def set_submitted_declarations(count: int) -> None:
    """Set gauge of submitted declarations."""
    set_declarations_by_status("submitted", count)


def set_cleared_declarations(count: int) -> None:
    """Set gauge of cleared declarations."""
    set_declarations_by_status("cleared", count)


def set_rejected_declarations(count: int) -> None:
    """Set gauge of rejected declarations."""
    set_declarations_by_status("rejected", count)


def set_total_customs_value(value: float) -> None:
    """Alias for set_total_customs_value_eur."""
    set_total_customs_value_eur(value)


def set_total_duty_collected(value: float) -> None:
    """Alias for set_total_duty_amount_eur."""
    set_total_duty_amount_eur(value)


def set_total_vat_collected(value: float) -> None:
    """Set gauge of total VAT collected."""
    if _PROMETHEUS_AVAILABLE:
        _TOTAL_DUTY_AMOUNT_EUR.set(value)


def set_average_processing_time(time_seconds: float) -> None:
    """Set gauge of average processing time."""
    if _PROMETHEUS_AVAILABLE:
        _AVERAGE_TARIFF_RATE.set(time_seconds)


def set_origin_verification_rate(rate: float) -> None:
    """Set gauge of origin verification rate."""
    if _PROMETHEUS_AVAILABLE:
        _COMPLIANCE_PASS_RATE.set(rate)


def set_ncts_queue_depth(depth: int) -> None:
    """Set gauge of NCTS queue depth."""
    set_declarations_by_status("ncts_queued", depth)


def set_ais_queue_depth(depth: int) -> None:
    """Set gauge of AIS queue depth."""
    set_declarations_by_status("ais_queued", depth)
