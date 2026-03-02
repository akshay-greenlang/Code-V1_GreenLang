# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Quarterly Automation Engine v1.1

Automated quarterly CBAM report generation with deadline tracking,
amendment management, and EU CBAM Registry format output.

Per EU CBAM Regulation 2023/956 and Implementing Regulation 2023/1773:
- Articles 6-8: Quarterly reporting obligation for EU importers
- Articles 3-5: Report content requirements (CN codes, emissions, origins)
- Article 9: Amendment/correction provisions (T+60 day window)
- Article 6(1): Submission deadline (T+30 after quarter end)

Architecture:
    QuarterlySchedulerEngine   - Period calculation, deadline management, lifecycle
    ReportAssemblerEngine      - Shipment aggregation, XML/Markdown generation
    AmendmentManagerEngine     - Versioned amendments, diff tracking, rollback
    DeadlineTrackerEngine      - Proactive alerts at T-30/14/7/3/1 thresholds
    NotificationService        - Multi-channel delivery (email, webhook)

Design Principles:
    - ZERO HALLUCINATION: All calculations are deterministic Python arithmetic
    - Decimal precision: ROUND_HALF_UP for all monetary/emissions values
    - Thread safety: RLock-based singletons for scheduler and assembler
    - Provenance: SHA-256 hashing on every report and amendment
    - Audit trail: Full version history with structured diffs

Usage:
    >>> from quarterly_engine import (
    ...     QuarterlySchedulerEngine,
    ...     ReportAssemblerEngine,
    ...     AmendmentManagerEngine,
    ...     DeadlineTrackerEngine,
    ...     NotificationService,
    ... )
    >>>
    >>> scheduler = QuarterlySchedulerEngine()
    >>> period = scheduler.get_current_quarter()
    >>> print(f"Current period: {period.period_label}")
    >>>
    >>> assembler = ReportAssemblerEngine()
    >>> report = assembler.assemble_quarterly_report(period, importer_id, shipments)
    >>> print(f"Total emissions: {report.total_embedded_emissions} tCO2e")

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

# Version metadata
__version__ = "1.1.0"
__author__ = "GreenLang CBAM Team"
__license__ = "Proprietary"

# ---------------------------------------------------------------------------
# Public API imports
# ---------------------------------------------------------------------------

from .models import (
    # Enums
    QuarterlyPeriod,
    ReportStatus,
    AmendmentReason,
    NotificationType,
    AlertLevel,
    CBAMSector,
    CalculationMethod,
    # Constants
    QUARTER_MONTHS,
    SUBMISSION_DEADLINE_DAYS,
    AMENDMENT_WINDOW_DAYS,
    ALERT_THRESHOLDS,
    TRANSITIONAL_PERIOD_END,
    DEFINITIVE_PERIOD_START,
    DEFAULT_VALUE_MARKUP_PCT,
    COMPLEX_GOODS_THRESHOLD_PCT,
    VALID_STATUS_TRANSITIONS,
    # Models
    QuarterlyReportPeriod,
    ShipmentAggregation,
    QuarterlyReport,
    ReportAmendment,
    DeadlineAlert,
    NotificationConfig,
    NotificationLogEntry,
    # Helpers
    compute_sha256,
    quantize_decimal,
    validate_status_transition,
)

from .quarterly_scheduler import QuarterlySchedulerEngine
from .report_assembler import ReportAssemblerEngine
from .amendment_manager import AmendmentManagerEngine
from .deadline_tracker import DeadlineTrackerEngine
from .notification_service import NotificationService

# ---------------------------------------------------------------------------
# Module-level __all__ for explicit public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Enums
    "QuarterlyPeriod",
    "ReportStatus",
    "AmendmentReason",
    "NotificationType",
    "AlertLevel",
    "CBAMSector",
    "CalculationMethod",
    # Constants
    "QUARTER_MONTHS",
    "SUBMISSION_DEADLINE_DAYS",
    "AMENDMENT_WINDOW_DAYS",
    "ALERT_THRESHOLDS",
    "TRANSITIONAL_PERIOD_END",
    "DEFINITIVE_PERIOD_START",
    "DEFAULT_VALUE_MARKUP_PCT",
    "COMPLEX_GOODS_THRESHOLD_PCT",
    "VALID_STATUS_TRANSITIONS",
    # Models
    "QuarterlyReportPeriod",
    "ShipmentAggregation",
    "QuarterlyReport",
    "ReportAmendment",
    "DeadlineAlert",
    "NotificationConfig",
    "NotificationLogEntry",
    # Helpers
    "compute_sha256",
    "quantize_decimal",
    "validate_status_transition",
    # Engines
    "QuarterlySchedulerEngine",
    "ReportAssemblerEngine",
    "AmendmentManagerEngine",
    "DeadlineTrackerEngine",
    "NotificationService",
]
