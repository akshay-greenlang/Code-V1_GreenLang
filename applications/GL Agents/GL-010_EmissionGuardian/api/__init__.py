# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - REST API Module

Production-grade REST API for emissions compliance monitoring.

Modules:
    - main: FastAPI application with middleware and configuration
    - schemas: Pydantic models for request/response validation
    - routes_cems: CEMS data endpoints
    - routes_compliance: Compliance monitoring endpoints
    - routes_rata: RATA test management endpoints
    - routes_fugitive: Fugitive detection endpoints
    - routes_trading: Carbon trading endpoints
    - routes_reports: Reporting endpoints

Example:
    >>> from api import app
    >>> # Run with uvicorn
    >>> import uvicorn
    >>> uvicorn.run(app, host="0.0.0.0", port=8010)
"""

from .main import app
from .schemas import (
    # Enumerations
    Pollutant,
    DataQuality,
    AveragingPeriod,
    OperatingState,
    ComplianceStatusType,
    ExceedanceSeverity,
    CorrectiveActionState,
    RATATestType,
    RATAStatus,
    AlertSeverity,
    AlertStatus,
    SensorStatus,
    ReviewDecision,
    TradeType,
    TradeStatus,
    CarbonMarket,
    OffsetStandard,
    ReportType,
    SortOrder,
    # Base models
    ProvenanceBase,
    TimestampMixin,
    # Pagination
    PaginationParams,
    PaginationMeta,
    PaginatedResponse,
    # Filters
    DateTimeRange,
    DateRange,
    NumericRange,
    # Error models
    ErrorDetail,
    ErrorResponse,
    # CEMS models
    CEMSReadingBase,
    CEMSReadingResponse,
    CEMSReadingCreate,
    CEMSReadingFilter,
    HourlyAverageResponse,
    DailySummaryResponse,
    DataAvailabilityResponse,
    # Compliance models
    PermitLimitResponse,
    ComplianceStatusResponse,
    ExceedanceResponse,
    ExceedanceFilter,
    ComplianceScheduleResponse,
    CorrectiveActionCreate,
    CorrectiveActionResponse,
    CorrectiveActionUpdate,
    # RATA models
    RATAScheduleResponse,
    RATARunData,
    RATATestCreate,
    RATAResultResponse,
    RATAReportResponse,
    # Fugitive models
    FugitiveAlertResponse,
    FugitiveAlertFilter,
    SensorStatusResponse,
    ReviewDecisionCreate,
    ReviewDecisionResponse,
    LDARStatusResponse,
    # Trading models
    TradingPositionResponse,
    TradingRecommendationResponse,
    TradeApprovalCreate,
    TradeApprovalResponse,
    OffsetCertificateResponse,
    # Reporting models
    QuarterlyReportRequest,
    QuarterlyReportResponse,
    AnnualReportResponse,
    DeviationReportResponse,
    AuditTrailFilter,
    AuditTrailResponse,
    # Auth models
    TokenResponse,
    UserInfo,
    # Health models
    HealthCheckResponse,
    ReadinessCheckResponse,
)

__version__ = "1.0.0"
__author__ = "GreenLang GL-010 EmissionsGuardian"

__all__ = [
    "app",
    # Enumerations
    "Pollutant",
    "DataQuality",
    "AveragingPeriod",
    "OperatingState",
    "ComplianceStatusType",
    "ExceedanceSeverity",
    "CorrectiveActionState",
    "RATATestType",
    "RATAStatus",
    "AlertSeverity",
    "AlertStatus",
    "SensorStatus",
    "ReviewDecision",
    "TradeType",
    "TradeStatus",
    "CarbonMarket",
    "OffsetStandard",
    "ReportType",
    "SortOrder",
    # Models
    "ProvenanceBase",
    "TimestampMixin",
    "PaginationParams",
    "PaginationMeta",
    "PaginatedResponse",
    "ErrorResponse",
    "HealthCheckResponse",
    "ReadinessCheckResponse",
]
