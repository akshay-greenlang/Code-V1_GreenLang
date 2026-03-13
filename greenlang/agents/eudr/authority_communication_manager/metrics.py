# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-040: Authority Communication Manager

45 Prometheus metrics for authority communication manager service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_acm_`` prefix for consistent
identification in Prometheus queries, Grafana dashboards, and alerting
rules across the GreenLang platform.

Metrics (45 per PRD Section 7.6):
    Counters (15):
        1.  gl_eudr_acm_communications_created_total         - Communications created [type, member_state]
        2.  gl_eudr_acm_communications_sent_total             - Communications sent [type, channel]
        3.  gl_eudr_acm_communications_responded_total        - Communications responded [type]
        4.  gl_eudr_acm_information_requests_received_total   - Info requests received [request_type]
        5.  gl_eudr_acm_information_requests_fulfilled_total  - Info requests fulfilled [request_type]
        6.  gl_eudr_acm_inspections_scheduled_total           - Inspections scheduled [inspection_type]
        7.  gl_eudr_acm_inspections_completed_total           - Inspections completed [inspection_type]
        8.  gl_eudr_acm_non_compliance_issued_total           - Non-compliance notices [violation_type, severity]
        9.  gl_eudr_acm_appeals_filed_total                   - Appeals filed [member_state]
        10. gl_eudr_acm_appeals_resolved_total                - Appeals resolved [decision]
        11. gl_eudr_acm_documents_exchanged_total             - Documents exchanged [doc_type, direction]
        12. gl_eudr_acm_notifications_sent_total              - Notifications sent [channel]
        13. gl_eudr_acm_notifications_failed_total            - Notifications failed [channel]
        14. gl_eudr_acm_deadline_reminders_sent_total         - Deadline reminders sent
        15. gl_eudr_acm_api_errors_total                      - API errors [operation]

    Histograms (15):
        16. gl_eudr_acm_response_time_hours                   - Response time in hours [type]
        17. gl_eudr_acm_processing_duration_seconds            - Processing duration [operation]
        18. gl_eudr_acm_inspection_duration_hours              - Inspection duration [inspection_type]
        19. gl_eudr_acm_appeal_resolution_days                 - Appeal resolution time [decision]
        20. gl_eudr_acm_document_upload_duration_seconds       - Document upload latency
        21. gl_eudr_acm_notification_delivery_seconds          - Notification delivery latency [channel]
        22. gl_eudr_acm_template_render_duration_seconds       - Template rendering latency [language]
        23. gl_eudr_acm_encryption_duration_seconds            - Document encryption latency
        24. gl_eudr_acm_request_handling_duration_seconds      - Request handling latency [request_type]
        25. gl_eudr_acm_non_compliance_processing_seconds      - Non-compliance processing latency
        26. gl_eudr_acm_appeal_processing_seconds              - Appeal processing latency
        27. gl_eudr_acm_communication_creation_seconds         - Communication creation latency
        28. gl_eudr_acm_authority_routing_seconds              - Authority routing latency
        29. gl_eudr_acm_deadline_check_duration_seconds        - Deadline check scan duration
        30. gl_eudr_acm_batch_processing_duration_seconds      - Batch processing duration

    Gauges (15):
        31. gl_eudr_acm_pending_communications                 - Pending communications count
        32. gl_eudr_acm_overdue_responses                      - Overdue responses count
        33. gl_eudr_acm_active_appeals                         - Active appeals count
        34. gl_eudr_acm_pending_inspections                    - Pending inspections count
        35. gl_eudr_acm_open_non_compliance_cases              - Open non-compliance cases
        36. gl_eudr_acm_active_threads                         - Active communication threads
        37. gl_eudr_acm_pending_approvals                      - Pending approval workflows
        38. gl_eudr_acm_template_count                         - Loaded templates count
        39. gl_eudr_acm_authority_count                        - Configured authorities count
        40. gl_eudr_acm_documents_stored                       - Total documents in storage
        41. gl_eudr_acm_encrypted_documents                    - Encrypted documents count
        42. gl_eudr_acm_notification_queue_depth                - Notification queue depth
        43. gl_eudr_acm_deadline_reminders_pending              - Pending deadline reminders
        44. gl_eudr_acm_average_response_time_hours             - Average response time gauge
        45. gl_eudr_acm_member_states_active                    - Active member states count

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-040 (GL-EUDR-ACM-040)
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
    # Counters (15)
    _COMMUNICATIONS_CREATED = Counter(
        "gl_eudr_acm_communications_created_total",
        "Communications created",
        ["type", "member_state"],
    )
    _COMMUNICATIONS_SENT = Counter(
        "gl_eudr_acm_communications_sent_total",
        "Communications sent",
        ["type", "channel"],
    )
    _COMMUNICATIONS_RESPONDED = Counter(
        "gl_eudr_acm_communications_responded_total",
        "Communications responded to",
        ["type"],
    )
    _INFORMATION_REQUESTS_RECEIVED = Counter(
        "gl_eudr_acm_information_requests_received_total",
        "Information requests received from authorities",
        ["request_type"],
    )
    _INFORMATION_REQUESTS_FULFILLED = Counter(
        "gl_eudr_acm_information_requests_fulfilled_total",
        "Information requests fulfilled",
        ["request_type"],
    )
    _INSPECTIONS_SCHEDULED = Counter(
        "gl_eudr_acm_inspections_scheduled_total",
        "Inspections scheduled",
        ["inspection_type"],
    )
    _INSPECTIONS_COMPLETED = Counter(
        "gl_eudr_acm_inspections_completed_total",
        "Inspections completed",
        ["inspection_type"],
    )
    _NON_COMPLIANCE_ISSUED = Counter(
        "gl_eudr_acm_non_compliance_issued_total",
        "Non-compliance notices issued",
        ["violation_type", "severity"],
    )
    _APPEALS_FILED = Counter(
        "gl_eudr_acm_appeals_filed_total",
        "Appeals filed",
        ["member_state"],
    )
    _APPEALS_RESOLVED = Counter(
        "gl_eudr_acm_appeals_resolved_total",
        "Appeals resolved",
        ["decision"],
    )
    _DOCUMENTS_EXCHANGED = Counter(
        "gl_eudr_acm_documents_exchanged_total",
        "Documents exchanged with authorities",
        ["doc_type", "direction"],
    )
    _NOTIFICATIONS_SENT = Counter(
        "gl_eudr_acm_notifications_sent_total",
        "Notifications sent",
        ["channel"],
    )
    _NOTIFICATIONS_FAILED = Counter(
        "gl_eudr_acm_notifications_failed_total",
        "Notifications that failed delivery",
        ["channel"],
    )
    _DEADLINE_REMINDERS_SENT = Counter(
        "gl_eudr_acm_deadline_reminders_sent_total",
        "Deadline reminders sent",
    )
    _API_ERRORS = Counter(
        "gl_eudr_acm_api_errors_total",
        "API errors by operation type",
        ["operation"],
    )

    # Histograms (15)
    _RESPONSE_TIME_HOURS = Histogram(
        "gl_eudr_acm_response_time_hours",
        "Time to respond to authority communications in hours",
        ["type"],
        buckets=(1.0, 4.0, 8.0, 24.0, 48.0, 72.0, 120.0, 240.0, 360.0),
    )
    _PROCESSING_DURATION = Histogram(
        "gl_eudr_acm_processing_duration_seconds",
        "Processing duration in seconds",
        ["operation"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _INSPECTION_DURATION_HOURS = Histogram(
        "gl_eudr_acm_inspection_duration_hours",
        "Inspection duration in hours",
        ["inspection_type"],
        buckets=(1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 48.0),
    )
    _APPEAL_RESOLUTION_DAYS = Histogram(
        "gl_eudr_acm_appeal_resolution_days",
        "Appeal resolution time in days",
        ["decision"],
        buckets=(7.0, 14.0, 30.0, 60.0, 90.0, 120.0, 180.0),
    )
    _DOCUMENT_UPLOAD_DURATION = Histogram(
        "gl_eudr_acm_document_upload_duration_seconds",
        "Document upload latency",
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _NOTIFICATION_DELIVERY = Histogram(
        "gl_eudr_acm_notification_delivery_seconds",
        "Notification delivery latency",
        ["channel"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    _TEMPLATE_RENDER_DURATION = Histogram(
        "gl_eudr_acm_template_render_duration_seconds",
        "Template rendering latency",
        ["language"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
    )
    _ENCRYPTION_DURATION = Histogram(
        "gl_eudr_acm_encryption_duration_seconds",
        "Document encryption latency",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )
    _REQUEST_HANDLING_DURATION = Histogram(
        "gl_eudr_acm_request_handling_duration_seconds",
        "Information request handling latency",
        ["request_type"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _NON_COMPLIANCE_PROCESSING = Histogram(
        "gl_eudr_acm_non_compliance_processing_seconds",
        "Non-compliance processing latency",
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _APPEAL_PROCESSING = Histogram(
        "gl_eudr_acm_appeal_processing_seconds",
        "Appeal processing latency",
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _COMMUNICATION_CREATION = Histogram(
        "gl_eudr_acm_communication_creation_seconds",
        "Communication creation latency",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _AUTHORITY_ROUTING = Histogram(
        "gl_eudr_acm_authority_routing_seconds",
        "Authority routing latency",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
    )
    _DEADLINE_CHECK_DURATION = Histogram(
        "gl_eudr_acm_deadline_check_duration_seconds",
        "Deadline check scan duration",
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _BATCH_PROCESSING_DURATION = Histogram(
        "gl_eudr_acm_batch_processing_duration_seconds",
        "Batch processing duration",
        buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    )

    # Gauges (15)
    _PENDING_COMMUNICATIONS = Gauge(
        "gl_eudr_acm_pending_communications",
        "Number of pending communications",
    )
    _OVERDUE_RESPONSES = Gauge(
        "gl_eudr_acm_overdue_responses",
        "Number of overdue responses",
    )
    _ACTIVE_APPEALS = Gauge(
        "gl_eudr_acm_active_appeals",
        "Number of active appeals",
    )
    _PENDING_INSPECTIONS = Gauge(
        "gl_eudr_acm_pending_inspections",
        "Number of pending inspections",
    )
    _OPEN_NON_COMPLIANCE = Gauge(
        "gl_eudr_acm_open_non_compliance_cases",
        "Number of open non-compliance cases",
    )
    _ACTIVE_THREADS = Gauge(
        "gl_eudr_acm_active_threads",
        "Number of active communication threads",
    )
    _PENDING_APPROVALS = Gauge(
        "gl_eudr_acm_pending_approvals",
        "Number of pending approval workflows",
    )
    _TEMPLATE_COUNT = Gauge(
        "gl_eudr_acm_template_count",
        "Number of loaded communication templates",
    )
    _AUTHORITY_COUNT = Gauge(
        "gl_eudr_acm_authority_count",
        "Number of configured authorities",
    )
    _DOCUMENTS_STORED = Gauge(
        "gl_eudr_acm_documents_stored",
        "Total documents in storage",
    )
    _ENCRYPTED_DOCUMENTS = Gauge(
        "gl_eudr_acm_encrypted_documents",
        "Number of encrypted documents",
    )
    _NOTIFICATION_QUEUE_DEPTH = Gauge(
        "gl_eudr_acm_notification_queue_depth",
        "Current notification queue depth",
    )
    _DEADLINE_REMINDERS_PENDING = Gauge(
        "gl_eudr_acm_deadline_reminders_pending",
        "Number of pending deadline reminders",
    )
    _AVERAGE_RESPONSE_TIME = Gauge(
        "gl_eudr_acm_average_response_time_hours",
        "Average response time in hours",
    )
    _MEMBER_STATES_ACTIVE = Gauge(
        "gl_eudr_acm_member_states_active",
        "Number of active member state configurations",
    )


# ---------------------------------------------------------------------------
# Helper Functions - Counters
# ---------------------------------------------------------------------------


def record_communication_created(comm_type: str, member_state: str) -> None:
    """Record a communication creation metric."""
    if _PROMETHEUS_AVAILABLE:
        _COMMUNICATIONS_CREATED.labels(
            type=comm_type, member_state=member_state
        ).inc()


def record_communication_sent(comm_type: str, channel: str) -> None:
    """Record a communication sent metric."""
    if _PROMETHEUS_AVAILABLE:
        _COMMUNICATIONS_SENT.labels(type=comm_type, channel=channel).inc()


def record_communication_responded(comm_type: str) -> None:
    """Record a communication response metric."""
    if _PROMETHEUS_AVAILABLE:
        _COMMUNICATIONS_RESPONDED.labels(type=comm_type).inc()


def record_information_request_received(request_type: str) -> None:
    """Record an information request received metric."""
    if _PROMETHEUS_AVAILABLE:
        _INFORMATION_REQUESTS_RECEIVED.labels(request_type=request_type).inc()


def record_information_request_fulfilled(request_type: str) -> None:
    """Record an information request fulfilled metric."""
    if _PROMETHEUS_AVAILABLE:
        _INFORMATION_REQUESTS_FULFILLED.labels(request_type=request_type).inc()


def record_inspection_scheduled(inspection_type: str) -> None:
    """Record an inspection scheduled metric."""
    if _PROMETHEUS_AVAILABLE:
        _INSPECTIONS_SCHEDULED.labels(inspection_type=inspection_type).inc()


def record_inspection_completed(inspection_type: str) -> None:
    """Record an inspection completed metric."""
    if _PROMETHEUS_AVAILABLE:
        _INSPECTIONS_COMPLETED.labels(inspection_type=inspection_type).inc()


def record_non_compliance_issued(violation_type: str, severity: str) -> None:
    """Record a non-compliance issuance metric."""
    if _PROMETHEUS_AVAILABLE:
        _NON_COMPLIANCE_ISSUED.labels(
            violation_type=violation_type, severity=severity
        ).inc()


def record_appeal_filed(member_state: str) -> None:
    """Record an appeal filing metric."""
    if _PROMETHEUS_AVAILABLE:
        _APPEALS_FILED.labels(member_state=member_state).inc()


def record_appeal_resolved(decision: str) -> None:
    """Record an appeal resolution metric."""
    if _PROMETHEUS_AVAILABLE:
        _APPEALS_RESOLVED.labels(decision=decision).inc()


def record_document_exchanged(doc_type: str, direction: str) -> None:
    """Record a document exchange metric."""
    if _PROMETHEUS_AVAILABLE:
        _DOCUMENTS_EXCHANGED.labels(
            doc_type=doc_type, direction=direction
        ).inc()


def record_notification_sent(channel: str) -> None:
    """Record a notification sent metric."""
    if _PROMETHEUS_AVAILABLE:
        _NOTIFICATIONS_SENT.labels(channel=channel).inc()


def record_notification_failed(channel: str) -> None:
    """Record a notification failure metric."""
    if _PROMETHEUS_AVAILABLE:
        _NOTIFICATIONS_FAILED.labels(channel=channel).inc()


def record_deadline_reminder_sent() -> None:
    """Record a deadline reminder sent metric."""
    if _PROMETHEUS_AVAILABLE:
        _DEADLINE_REMINDERS_SENT.inc()


def record_api_error(operation: str) -> None:
    """Record an API error metric."""
    if _PROMETHEUS_AVAILABLE:
        _API_ERRORS.labels(operation=operation).inc()


# ---------------------------------------------------------------------------
# Helper Functions - Histograms
# ---------------------------------------------------------------------------


def observe_response_time_hours(comm_type: str, hours: float) -> None:
    """Observe response time in hours."""
    if _PROMETHEUS_AVAILABLE:
        _RESPONSE_TIME_HOURS.labels(type=comm_type).observe(hours)


def observe_processing_duration(operation: str, duration: float) -> None:
    """Observe processing duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _PROCESSING_DURATION.labels(operation=operation).observe(duration)


def observe_inspection_duration_hours(
    inspection_type: str, hours: float
) -> None:
    """Observe inspection duration in hours."""
    if _PROMETHEUS_AVAILABLE:
        _INSPECTION_DURATION_HOURS.labels(
            inspection_type=inspection_type
        ).observe(hours)


def observe_appeal_resolution_days(decision: str, days: float) -> None:
    """Observe appeal resolution time in days."""
    if _PROMETHEUS_AVAILABLE:
        _APPEAL_RESOLUTION_DAYS.labels(decision=decision).observe(days)


def observe_document_upload_duration(duration: float) -> None:
    """Observe document upload duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _DOCUMENT_UPLOAD_DURATION.observe(duration)


def observe_notification_delivery(channel: str, duration: float) -> None:
    """Observe notification delivery latency in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _NOTIFICATION_DELIVERY.labels(channel=channel).observe(duration)


def observe_template_render_duration(language: str, duration: float) -> None:
    """Observe template rendering duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _TEMPLATE_RENDER_DURATION.labels(language=language).observe(duration)


def observe_encryption_duration(duration: float) -> None:
    """Observe document encryption duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _ENCRYPTION_DURATION.observe(duration)


def observe_request_handling_duration(
    request_type: str, duration: float
) -> None:
    """Observe information request handling duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _REQUEST_HANDLING_DURATION.labels(
            request_type=request_type
        ).observe(duration)


def observe_non_compliance_processing(duration: float) -> None:
    """Observe non-compliance processing duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _NON_COMPLIANCE_PROCESSING.observe(duration)


def observe_appeal_processing(duration: float) -> None:
    """Observe appeal processing duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _APPEAL_PROCESSING.observe(duration)


def observe_communication_creation(duration: float) -> None:
    """Observe communication creation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _COMMUNICATION_CREATION.observe(duration)


def observe_authority_routing(duration: float) -> None:
    """Observe authority routing duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _AUTHORITY_ROUTING.observe(duration)


def observe_deadline_check_duration(duration: float) -> None:
    """Observe deadline check scan duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _DEADLINE_CHECK_DURATION.observe(duration)


def observe_batch_processing_duration(duration: float) -> None:
    """Observe batch processing duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _BATCH_PROCESSING_DURATION.observe(duration)


# ---------------------------------------------------------------------------
# Helper Functions - Gauges
# ---------------------------------------------------------------------------


def set_pending_communications(count: int) -> None:
    """Set gauge of pending communications."""
    if _PROMETHEUS_AVAILABLE:
        _PENDING_COMMUNICATIONS.set(count)


def set_overdue_responses(count: int) -> None:
    """Set gauge of overdue responses."""
    if _PROMETHEUS_AVAILABLE:
        _OVERDUE_RESPONSES.set(count)


def set_active_appeals(count: int) -> None:
    """Set gauge of active appeals."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_APPEALS.set(count)


def set_pending_inspections(count: int) -> None:
    """Set gauge of pending inspections."""
    if _PROMETHEUS_AVAILABLE:
        _PENDING_INSPECTIONS.set(count)


def set_open_non_compliance(count: int) -> None:
    """Set gauge of open non-compliance cases."""
    if _PROMETHEUS_AVAILABLE:
        _OPEN_NON_COMPLIANCE.set(count)


def set_active_threads(count: int) -> None:
    """Set gauge of active communication threads."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_THREADS.set(count)


def set_pending_approvals(count: int) -> None:
    """Set gauge of pending approval workflows."""
    if _PROMETHEUS_AVAILABLE:
        _PENDING_APPROVALS.set(count)


def set_template_count(count: int) -> None:
    """Set gauge of loaded templates."""
    if _PROMETHEUS_AVAILABLE:
        _TEMPLATE_COUNT.set(count)


def set_authority_count(count: int) -> None:
    """Set gauge of configured authorities."""
    if _PROMETHEUS_AVAILABLE:
        _AUTHORITY_COUNT.set(count)


def set_documents_stored(count: int) -> None:
    """Set gauge of total stored documents."""
    if _PROMETHEUS_AVAILABLE:
        _DOCUMENTS_STORED.set(count)


def set_encrypted_documents(count: int) -> None:
    """Set gauge of encrypted documents."""
    if _PROMETHEUS_AVAILABLE:
        _ENCRYPTED_DOCUMENTS.set(count)


def set_notification_queue_depth(count: int) -> None:
    """Set gauge of notification queue depth."""
    if _PROMETHEUS_AVAILABLE:
        _NOTIFICATION_QUEUE_DEPTH.set(count)


def set_deadline_reminders_pending(count: int) -> None:
    """Set gauge of pending deadline reminders."""
    if _PROMETHEUS_AVAILABLE:
        _DEADLINE_REMINDERS_PENDING.set(count)


def set_average_response_time(hours: float) -> None:
    """Set gauge of average response time in hours."""
    if _PROMETHEUS_AVAILABLE:
        _AVERAGE_RESPONSE_TIME.set(hours)


def set_member_states_active(count: int) -> None:
    """Set gauge of active member state configurations."""
    if _PROMETHEUS_AVAILABLE:
        _MEMBER_STATES_ACTIVE.set(count)
