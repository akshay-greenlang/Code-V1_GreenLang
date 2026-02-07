# -*- coding: utf-8 -*-
"""
Threat Modeling API - SEC-010 Phase 2

FastAPI router, schemas, and middleware for the threat modeling REST API.

Provides:
    - REST API endpoints for threat model management
    - Request/response schemas with full validation
    - STRIDE analysis and risk scoring endpoints
"""

from greenlang.infrastructure.threat_modeling.api.threat_routes import (
    router as threat_router,
    # Request models
    CreateThreatModelRequest,
    UpdateThreatModelRequest,
    AddComponentRequest,
    UpdateComponentRequest,
    AddDataFlowRequest,
    AddMitigationRequest,
    RunAnalysisRequest,
    ApproveRequest,
    # Response models
    ThreatModelResponse,
    ThreatModelListResponse,
    ThreatModelDetailResponse,
    ComponentResponse,
    ThreatResponse,
    AnalysisResultResponse,
    ReportResponse,
)

__all__ = [
    "threat_router",
    # Request models
    "CreateThreatModelRequest",
    "UpdateThreatModelRequest",
    "AddComponentRequest",
    "UpdateComponentRequest",
    "AddDataFlowRequest",
    "AddMitigationRequest",
    "RunAnalysisRequest",
    "ApproveRequest",
    # Response models
    "ThreatModelResponse",
    "ThreatModelListResponse",
    "ThreatModelDetailResponse",
    "ComponentResponse",
    "ThreatResponse",
    "AnalysisResultResponse",
    "ReportResponse",
]
