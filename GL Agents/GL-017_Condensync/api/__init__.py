# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC API Module

FastAPI REST API for Condenser Optimization Agent.

This module provides production-grade REST endpoints for:
- Condenser diagnostic analysis
- Vacuum optimization recommendations
- Fouling prediction and trending
- Cleaning schedule recommendations
- Health monitoring and metrics
- KPI tracking and reporting

Features:
- Full Pydantic v2 request/response validation
- Correlation ID tracking for distributed tracing
- SHA-256 provenance hashing for audit trails
- Comprehensive error handling
- Rate limiting support
- OpenAPI/Swagger documentation

Usage:
    from api import create_app, router

    # Create standalone application
    app = create_app()

    # Or include router in existing application
    existing_app.include_router(router)

Example:
    # Start API server
    uvicorn.run("api:create_app", host="0.0.0.0", port=8017, factory=True)

Author: GL-APIDeveloper
Date: December 2025
Version: 1.0.0
"""

from .routes import (
    # Application factory
    create_app,
    # Router for integration
    router,
    # Constants
    AGENT_ID,
    AGENT_NAME,
    AGENT_VERSION,
    # Utility functions
    compute_provenance_hash,
    get_uptime_seconds,
)

from .schemas import (
    # Enumerations
    CondenserType,
    ConditionStatus,
    SeverityLevel,
    AlertLevel,
    FoulingType,
    CleaningMethod,
    OptimizationMode,
    MetricTimeRange,
    HealthStatus,
    # Base models
    BaseRequest,
    BaseResponse,
    ProvenanceMetadata,
    # Condenser data models
    CondenserOperatingData,
    CondenserHistoricalData,
    # Diagnostic schemas
    DiagnosticRequest,
    DiagnosticResponse,
    PerformanceMetrics,
    EnergyImpact,
    DiagnosticIssue,
    # Vacuum optimization schemas
    VacuumOptimizationRequest,
    VacuumOptimizationResponse,
    OptimizationSetpoint,
    OptimizationBenefit,
    SensitivityResult,
    # Fouling prediction schemas
    FoulingPredictionRequest,
    FoulingPredictionResponse,
    FoulingTrendPoint,
    # Cleaning recommendation schemas
    CleaningRecommendationRequest,
    CleaningRecommendationResponse,
    CleaningMethodRecommendation,
    # Health and status schemas
    HealthResponse,
    StatusResponse,
    ComponentStatus,
    # Metrics schemas
    MetricsRequest,
    MetricsResponse,
    MetricValue,
    # KPI schemas
    CurrentKPIsResponse,
    KPIValue,
    HistoricalKPIsRequest,
    HistoricalKPIsResponse,
    HistoricalKPI,
    HistoricalKPIPoint,
    # Error schemas
    ErrorResponse,
    ErrorDetail,
)

__version__ = "1.0.0"
__agent_id__ = "GL-017"
__agent_name__ = "CONDENSYNC"

__all__ = [
    # Module metadata
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # Application factory and router
    "create_app",
    "router",
    # Constants
    "AGENT_ID",
    "AGENT_NAME",
    "AGENT_VERSION",
    # Utility functions
    "compute_provenance_hash",
    "get_uptime_seconds",
    # Enumerations
    "CondenserType",
    "ConditionStatus",
    "SeverityLevel",
    "AlertLevel",
    "FoulingType",
    "CleaningMethod",
    "OptimizationMode",
    "MetricTimeRange",
    "HealthStatus",
    # Base models
    "BaseRequest",
    "BaseResponse",
    "ProvenanceMetadata",
    # Condenser data models
    "CondenserOperatingData",
    "CondenserHistoricalData",
    # Diagnostic schemas
    "DiagnosticRequest",
    "DiagnosticResponse",
    "PerformanceMetrics",
    "EnergyImpact",
    "DiagnosticIssue",
    # Vacuum optimization schemas
    "VacuumOptimizationRequest",
    "VacuumOptimizationResponse",
    "OptimizationSetpoint",
    "OptimizationBenefit",
    "SensitivityResult",
    # Fouling prediction schemas
    "FoulingPredictionRequest",
    "FoulingPredictionResponse",
    "FoulingTrendPoint",
    # Cleaning recommendation schemas
    "CleaningRecommendationRequest",
    "CleaningRecommendationResponse",
    "CleaningMethodRecommendation",
    # Health and status schemas
    "HealthResponse",
    "StatusResponse",
    "ComponentStatus",
    # Metrics schemas
    "MetricsRequest",
    "MetricsResponse",
    "MetricValue",
    # KPI schemas
    "CurrentKPIsResponse",
    "KPIValue",
    "HistoricalKPIsRequest",
    "HistoricalKPIsResponse",
    "HistoricalKPI",
    "HistoricalKPIPoint",
    # Error schemas
    "ErrorResponse",
    "ErrorDetail",
]
