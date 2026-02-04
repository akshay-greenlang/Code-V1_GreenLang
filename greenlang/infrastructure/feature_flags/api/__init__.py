# -*- coding: utf-8 -*-
"""
Feature Flags API - INFRA-008

FastAPI router, schemas, and middleware for the feature flag REST API.

Provides:
    - REST API endpoints for flag management and evaluation
    - Request/response schemas with full validation
    - Middleware for automatic flag injection into ExecutionContext
"""

from greenlang.infrastructure.feature_flags.api.schemas import (
    BatchEvaluateRequest,
    BatchEvaluateResponse,
    CreateFlagRequest,
    EvaluateRequest,
    EvaluateResponse,
    FlagListResponse,
    FlagResponse,
    KillSwitchRequest,
    OverrideRequest,
    RolloutRequest,
    UpdateFlagRequest,
    VariantRequest,
)

__all__ = [
    "BatchEvaluateRequest",
    "BatchEvaluateResponse",
    "CreateFlagRequest",
    "EvaluateRequest",
    "EvaluateResponse",
    "FlagListResponse",
    "FlagResponse",
    "KillSwitchRequest",
    "OverrideRequest",
    "RolloutRequest",
    "UpdateFlagRequest",
    "VariantRequest",
]
