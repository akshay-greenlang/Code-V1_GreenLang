# -*- coding: utf-8 -*-
"""
Feature Flags REST API Router - INFRA-008

FastAPI APIRouter providing REST endpoints for the GreenLang feature flag
system. Supports flag CRUD, evaluation, rollout control, kill switch,
overrides, variants, and audit trail access.

All endpoints use dependency injection for the FeatureFlagService and
include proper error handling with standard HTTP status codes.

Endpoints:
    GET    /api/v1/flags              - List flags (paginated, filterable)
    POST   /api/v1/flags              - Create flag
    GET    /api/v1/flags/stale        - List stale flags
    GET    /api/v1/flags/{key}        - Get flag details
    PUT    /api/v1/flags/{key}        - Update flag
    DELETE /api/v1/flags/{key}        - Archive flag (soft delete)
    POST   /api/v1/flags/{key}/evaluate   - Evaluate flag
    POST   /api/v1/flags/evaluate-batch   - Batch evaluate
    PUT    /api/v1/flags/{key}/rollout    - Set rollout percentage
    POST   /api/v1/flags/{key}/kill       - Activate kill switch
    POST   /api/v1/flags/{key}/restore    - Deactivate kill switch
    GET    /api/v1/flags/{key}/audit      - Get audit trail
    GET    /api/v1/flags/{key}/metrics    - Get flag metrics
    POST   /api/v1/flags/{key}/variants   - Add/update variant
    POST   /api/v1/flags/{key}/overrides  - Set override
    DELETE /api/v1/flags/{key}/overrides  - Clear override

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.feature_flags.api.router import flags_router
    >>> app = FastAPI()
    >>> app.include_router(flags_router)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.feature_flags.models import (
    EvaluationContext,
    FeatureFlag,
    FlagStatus,
    FlagType,
    FlagVariant,
)
from greenlang.infrastructure.feature_flags.service import (
    FeatureFlagService,
    get_feature_flag_service,
)

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, status
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    Query = None  # type: ignore[assignment]
    status = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]

from greenlang.infrastructure.feature_flags.api.schemas import (
    AuditLogEntryResponse,
    AuditLogResponse,
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
    StatisticsResponse,
    UpdateFlagRequest,
    VariantRequest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency Injection
# ---------------------------------------------------------------------------


def _get_service() -> FeatureFlagService:
    """FastAPI dependency that provides the FeatureFlagService singleton.

    Returns:
        The global FeatureFlagService instance.
    """
    return get_feature_flag_service()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _flag_to_response(flag: FeatureFlag) -> FlagResponse:
    """Convert a FeatureFlag model to a FlagResponse API schema.

    Args:
        flag: The internal FeatureFlag model.

    Returns:
        API-facing FlagResponse.
    """
    return FlagResponse(
        key=flag.key,
        name=flag.name,
        description=flag.description,
        flag_type=flag.flag_type.value,
        status=flag.status.value,
        default_value=flag.default_value,
        rollout_percentage=flag.rollout_percentage,
        environments=flag.environments,
        tags=flag.tags,
        owner=flag.owner,
        metadata=flag.metadata,
        start_time=flag.start_time,
        end_time=flag.end_time,
        created_at=flag.created_at,
        updated_at=flag.updated_at,
        version=flag.version,
    )


def _build_evaluation_context(req: EvaluateRequest) -> EvaluationContext:
    """Convert an EvaluateRequest to an internal EvaluationContext.

    Args:
        req: API-layer evaluation request.

    Returns:
        Internal EvaluationContext for the engine.
    """
    return EvaluationContext(
        user_id=req.user_id,
        tenant_id=req.tenant_id,
        environment=req.environment,
        user_segments=req.user_segments,
        user_attributes=req.user_attributes,
    )


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    flags_router = APIRouter(
        prefix="/api/v1/flags",
        tags=["Feature Flags"],
        responses={
            400: {"description": "Bad Request"},
            404: {"description": "Flag Not Found"},
            409: {"description": "Conflict"},
            422: {"description": "Validation Error"},
            500: {"description": "Internal Server Error"},
        },
    )

    # -- List Flags --------------------------------------------------------

    @flags_router.get(
        "",
        response_model=FlagListResponse,
        summary="List feature flags",
        description="Retrieve a paginated, filterable list of feature flags.",
        operation_id="list_flags",
    )
    async def list_flags(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        flag_status: Optional[str] = Query(
            None, alias="status", description="Filter by status"
        ),
        flag_type: Optional[str] = Query(None, description="Filter by flag type"),
        tag: Optional[str] = Query(None, description="Filter by tag"),
        owner: Optional[str] = Query(None, description="Filter by owner"),
        service: FeatureFlagService = Depends(_get_service),
    ) -> FlagListResponse:
        """List flags with pagination and optional filters.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.
            flag_status: Optional status filter.
            flag_type: Optional type filter.
            tag: Optional tag filter.
            owner: Optional owner filter.
            service: Injected FeatureFlagService.

        Returns:
            Paginated list of flags.
        """
        # Convert string status to enum if provided
        status_enum: Optional[FlagStatus] = None
        if flag_status:
            try:
                status_enum = FlagStatus(flag_status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid status '{flag_status}'. "
                           f"Allowed: {[s.value for s in FlagStatus]}",
                )

        offset = (page - 1) * page_size
        flags = await service.list_flags(
            status=status_enum,
            flag_type=flag_type,
            tag=tag,
            owner=owner,
            offset=offset,
            limit=page_size,
        )
        total = await service.count_flags(
            status=status_enum,
            flag_type=flag_type,
            tag=tag,
            owner=owner,
        )

        total_pages = (total + page_size - 1) // page_size if total > 0 else 0

        return FlagListResponse(
            items=[_flag_to_response(f) for f in flags],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )

    # -- Create Flag -------------------------------------------------------

    @flags_router.post(
        "",
        response_model=FlagResponse,
        status_code=201,
        summary="Create a feature flag",
        description="Create a new feature flag definition.",
        operation_id="create_flag",
    )
    async def create_flag(
        request: CreateFlagRequest,
        service: FeatureFlagService = Depends(_get_service),
    ) -> FlagResponse:
        """Create a new feature flag.

        Args:
            request: Flag creation request.
            service: Injected FeatureFlagService.

        Returns:
            The created flag.

        Raises:
            HTTPException 409: If a flag with the same key already exists.
        """
        try:
            flag_type_enum = FlagType(request.flag_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid flag_type '{request.flag_type}'.",
            )

        flag = FeatureFlag(
            key=request.key,
            name=request.name,
            description=request.description,
            flag_type=flag_type_enum,
            status=FlagStatus.DRAFT,
            default_value=request.default_value,
            rollout_percentage=request.rollout_percentage,
            environments=request.environments,
            tags=request.tags,
            owner=request.owner,
            metadata=request.metadata,
            start_time=request.start_time,
            end_time=request.end_time,
        )

        try:
            created = await service.create_flag(flag)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            )

        logger.info("API: Created flag %s", created.key)
        return _flag_to_response(created)

    # -- Stale Flags (before {key} route to avoid path conflict) -----------

    @flags_router.get(
        "/stale",
        response_model=List[FlagResponse],
        summary="List stale flags",
        description="List flags that have not been evaluated recently.",
        operation_id="list_stale_flags",
    )
    async def list_stale_flags(
        days: int = Query(30, ge=1, le=365, description="Stale threshold in days"),
        service: FeatureFlagService = Depends(_get_service),
    ) -> List[FlagResponse]:
        """List flags older than the threshold that may be stale.

        This is a heuristic based on flag updated_at. For precise stale
        detection based on evaluation metrics, use the lifecycle APIs.

        Args:
            days: Number of days without update to consider stale.
            service: Injected FeatureFlagService.

        Returns:
            List of potentially stale flags.
        """
        all_flags = await service.list_flags(offset=0, limit=10000)
        threshold = datetime.now(timezone.utc).timestamp() - (days * 86400)
        stale = [
            f for f in all_flags
            if f.updated_at.timestamp() < threshold
            and f.status not in (FlagStatus.ARCHIVED, FlagStatus.PERMANENT)
        ]
        return [_flag_to_response(f) for f in stale]

    # -- Batch Evaluate (before {key} route to avoid path conflict) --------

    @flags_router.post(
        "/evaluate-batch",
        response_model=BatchEvaluateResponse,
        summary="Batch evaluate flags",
        description="Evaluate multiple flags in a single request.",
        operation_id="batch_evaluate_flags",
    )
    async def batch_evaluate_flags(
        request: BatchEvaluateRequest,
        service: FeatureFlagService = Depends(_get_service),
    ) -> BatchEvaluateResponse:
        """Evaluate multiple flags at once.

        Args:
            request: Batch evaluation request with flag keys and context.
            service: Injected FeatureFlagService.

        Returns:
            Batch evaluation results.
        """
        context = _build_evaluation_context(request.context)
        results = await service.evaluate_batch(request.flag_keys, context)

        response_results: Dict[str, EvaluateResponse] = {}
        total_duration = 0
        for key, result in results.items():
            response_results[key] = EvaluateResponse(
                flag_key=result.flag_key,
                enabled=result.enabled,
                reason=result.reason,
                variant_key=result.variant_key,
                metadata=result.metadata,
                duration_us=result.duration_us,
            )
            total_duration += result.duration_us

        return BatchEvaluateResponse(
            results=response_results,
            total_duration_us=total_duration,
        )

    # -- Get Flag ----------------------------------------------------------

    @flags_router.get(
        "/{key}",
        response_model=FlagResponse,
        summary="Get flag details",
        description="Retrieve a specific feature flag by its key.",
        operation_id="get_flag",
    )
    async def get_flag(
        key: str,
        service: FeatureFlagService = Depends(_get_service),
    ) -> FlagResponse:
        """Get a single flag by key.

        Args:
            key: Flag key.
            service: Injected FeatureFlagService.

        Returns:
            Flag details.

        Raises:
            HTTPException 404: If the flag does not exist.
        """
        flag = await service.get_flag(key)
        if flag is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Flag '{key}' not found.",
            )
        return _flag_to_response(flag)

    # -- Update Flag -------------------------------------------------------

    @flags_router.put(
        "/{key}",
        response_model=FlagResponse,
        summary="Update flag",
        description="Update an existing feature flag.",
        operation_id="update_flag",
    )
    async def update_flag(
        key: str,
        request: UpdateFlagRequest,
        service: FeatureFlagService = Depends(_get_service),
    ) -> FlagResponse:
        """Update an existing flag.

        Args:
            key: Flag key.
            request: Update request with fields to change.
            service: Injected FeatureFlagService.

        Returns:
            Updated flag.

        Raises:
            HTTPException 404: If the flag does not exist.
        """
        updates = request.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No fields provided for update.",
            )

        # Convert flag_type string to enum if provided
        if "flag_type" in updates:
            try:
                updates["flag_type"] = FlagType(updates["flag_type"])
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid flag_type '{updates['flag_type']}'.",
                )

        try:
            updated = await service.update_flag(key, updates)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )

        return _flag_to_response(updated)

    # -- Delete (Archive) Flag ---------------------------------------------

    @flags_router.delete(
        "/{key}",
        status_code=204,
        summary="Archive flag",
        description="Soft-delete (archive) a feature flag.",
        operation_id="delete_flag",
    )
    async def delete_flag(
        key: str,
        service: FeatureFlagService = Depends(_get_service),
    ) -> None:
        """Archive a flag (soft delete).

        Args:
            key: Flag key.
            service: Injected FeatureFlagService.

        Raises:
            HTTPException 404: If the flag does not exist.
        """
        result = await service.delete_flag(key)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Flag '{key}' not found.",
            )

    # -- Evaluate Flag -----------------------------------------------------

    @flags_router.post(
        "/{key}/evaluate",
        response_model=EvaluateResponse,
        summary="Evaluate flag",
        description="Evaluate a flag for the given context.",
        operation_id="evaluate_flag",
    )
    async def evaluate_flag(
        key: str,
        request: EvaluateRequest,
        service: FeatureFlagService = Depends(_get_service),
    ) -> EvaluateResponse:
        """Evaluate a flag for the given context.

        Args:
            key: Flag key.
            request: Evaluation context.
            service: Injected FeatureFlagService.

        Returns:
            Evaluation result.
        """
        context = _build_evaluation_context(request)
        result = await service.evaluate(key, context)
        return EvaluateResponse(
            flag_key=result.flag_key,
            enabled=result.enabled,
            reason=result.reason,
            variant_key=result.variant_key,
            metadata=result.metadata,
            duration_us=result.duration_us,
        )

    # -- Rollout -----------------------------------------------------------

    @flags_router.put(
        "/{key}/rollout",
        response_model=FlagResponse,
        summary="Set rollout percentage",
        description="Set the rollout percentage for a flag.",
        operation_id="set_rollout",
    )
    async def set_rollout(
        key: str,
        request: RolloutRequest,
        service: FeatureFlagService = Depends(_get_service),
    ) -> FlagResponse:
        """Set rollout percentage for a flag.

        Args:
            key: Flag key.
            request: Rollout request with percentage.
            service: Injected FeatureFlagService.

        Returns:
            Updated flag.
        """
        try:
            updated = await service.set_rollout_percentage(
                key, request.percentage, request.updated_by
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )
        return _flag_to_response(updated)

    # -- Kill Switch -------------------------------------------------------

    @flags_router.post(
        "/{key}/kill",
        status_code=200,
        summary="Activate kill switch",
        description="Emergency: immediately disable a flag via kill switch.",
        operation_id="kill_flag",
    )
    async def kill_flag(
        key: str,
        request: KillSwitchRequest,
        service: FeatureFlagService = Depends(_get_service),
    ) -> Dict[str, Any]:
        """Activate the kill switch for a flag.

        Args:
            key: Flag key.
            request: Kill switch request.
            service: Injected FeatureFlagService.

        Returns:
            Kill switch result.
        """
        result = await service.kill_flag(key, request.actor, request.reason)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Flag '{key}' not found.",
            )
        return {"flag_key": key, "killed": True, "actor": request.actor}

    @flags_router.post(
        "/{key}/restore",
        status_code=200,
        summary="Deactivate kill switch",
        description="Restore a killed flag to active status.",
        operation_id="restore_flag",
    )
    async def restore_flag(
        key: str,
        request: KillSwitchRequest,
        service: FeatureFlagService = Depends(_get_service),
    ) -> Dict[str, Any]:
        """Deactivate the kill switch and restore a flag.

        Args:
            key: Flag key.
            request: Restore request.
            service: Injected FeatureFlagService.

        Returns:
            Restore result.
        """
        result = await service.restore_flag(key, request.actor, request.reason)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Flag '{key}' not found or not currently killed.",
            )
        return {"flag_key": key, "restored": True, "actor": request.actor}

    # -- Audit Log ---------------------------------------------------------

    @flags_router.get(
        "/{key}/audit",
        response_model=AuditLogResponse,
        summary="Get audit trail",
        description="Retrieve the audit log for a flag.",
        operation_id="get_flag_audit",
    )
    async def get_flag_audit(
        key: str,
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        service: FeatureFlagService = Depends(_get_service),
    ) -> AuditLogResponse:
        """Get paginated audit log for a flag.

        Args:
            key: Flag key.
            page: Page number.
            page_size: Items per page.
            service: Injected FeatureFlagService.

        Returns:
            Paginated audit log.
        """
        flag = await service.get_flag(key)
        if flag is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Flag '{key}' not found.",
            )

        offset = (page - 1) * page_size
        entries = await service.get_audit_log(key, offset=offset, limit=page_size)
        # Get total count (fetch all to count - acceptable for audit log)
        all_entries = await service.get_audit_log(key, offset=0, limit=10000)
        total = len(all_entries)

        items = [
            AuditLogEntryResponse(
                flag_key=e.flag_key,
                action=e.action,
                old_value=e.old_value,
                new_value=e.new_value,
                changed_by=e.changed_by,
                change_reason=e.change_reason,
                ip_address=e.ip_address,
                created_at=e.created_at,
            )
            for e in entries
        ]

        return AuditLogResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )

    # -- Flag Metrics (Statistics) -----------------------------------------

    @flags_router.get(
        "/{key}/metrics",
        response_model=Dict[str, Any],
        summary="Get flag metrics",
        description="Retrieve evaluation metrics for a specific flag.",
        operation_id="get_flag_metrics",
    )
    async def get_flag_metrics(
        key: str,
        service: FeatureFlagService = Depends(_get_service),
    ) -> Dict[str, Any]:
        """Get metrics for a specific flag.

        Args:
            key: Flag key.
            service: Injected FeatureFlagService.

        Returns:
            Flag metrics including evaluation counts and status.
        """
        flag = await service.get_flag(key)
        if flag is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Flag '{key}' not found.",
            )

        audit_entries = await service.get_audit_log(key, offset=0, limit=10000)
        return {
            "flag_key": key,
            "status": flag.status.value,
            "flag_type": flag.flag_type.value,
            "rollout_percentage": flag.rollout_percentage,
            "version": flag.version,
            "total_changes": len(audit_entries),
            "created_at": flag.created_at.isoformat(),
            "updated_at": flag.updated_at.isoformat(),
            "age_days": (
                datetime.now(timezone.utc) - flag.created_at
            ).days,
        }

    # -- Variants ----------------------------------------------------------

    @flags_router.post(
        "/{key}/variants",
        response_model=Dict[str, Any],
        status_code=201,
        summary="Add/update variant",
        description="Add or update a variant for a multivariate flag.",
        operation_id="add_variant",
    )
    async def add_variant(
        key: str,
        request: VariantRequest,
        service: FeatureFlagService = Depends(_get_service),
    ) -> Dict[str, Any]:
        """Add or update a variant for a flag.

        Args:
            key: Flag key.
            request: Variant definition.
            service: Injected FeatureFlagService.

        Returns:
            Variant creation result.
        """
        variant = FlagVariant(
            variant_key=request.variant_key,
            flag_key=key,
            variant_value=request.variant_value,
            weight=request.weight,
            description=request.description,
        )
        try:
            await service.add_variant(variant)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )
        return {
            "flag_key": key,
            "variant_key": request.variant_key,
            "weight": request.weight,
            "created": True,
        }

    # -- Overrides ---------------------------------------------------------

    @flags_router.post(
        "/{key}/overrides",
        response_model=Dict[str, Any],
        status_code=201,
        summary="Set override",
        description="Set a scoped override for a flag.",
        operation_id="set_override",
    )
    async def set_override(
        key: str,
        request: OverrideRequest,
        service: FeatureFlagService = Depends(_get_service),
    ) -> Dict[str, Any]:
        """Set an override for a flag.

        Args:
            key: Flag key.
            request: Override definition.
            service: Injected FeatureFlagService.

        Returns:
            Override creation result.
        """
        try:
            await service.set_override(
                flag_key=key,
                scope_type=request.scope_type,
                scope_value=request.scope_value,
                enabled=request.enabled,
                variant_key=request.variant_key,
                expires_at=request.expires_at,
                created_by=request.created_by,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )
        return {
            "flag_key": key,
            "scope_type": request.scope_type,
            "scope_value": request.scope_value,
            "enabled": request.enabled,
            "created": True,
        }

    @flags_router.delete(
        "/{key}/overrides",
        status_code=204,
        summary="Clear override",
        description="Remove a scoped override from a flag.",
        operation_id="clear_override",
    )
    async def clear_override(
        key: str,
        scope_type: str = Query(..., description="Override scope type"),
        scope_value: str = Query(..., description="Override scope value"),
        service: FeatureFlagService = Depends(_get_service),
    ) -> None:
        """Clear an override from a flag.

        Args:
            key: Flag key.
            scope_type: Override scope type.
            scope_value: Override scope value.
            service: Injected FeatureFlagService.
        """
        result = await service.clear_override(key, scope_type, scope_value)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Override not found for flag '{key}' "
                       f"scope={scope_type}:{scope_value}.",
            )

    # -- System Statistics -------------------------------------------------

    @flags_router.get(
        "/system/statistics",
        response_model=StatisticsResponse,
        summary="Get system statistics",
        description="Retrieve overall feature flag system statistics.",
        operation_id="get_statistics",
    )
    async def get_system_statistics(
        service: FeatureFlagService = Depends(_get_service),
    ) -> StatisticsResponse:
        """Get overall system statistics.

        Args:
            service: Injected FeatureFlagService.

        Returns:
            System-wide statistics.
        """
        stats = await service.get_statistics()
        return StatisticsResponse(**stats)

else:
    flags_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - flags_router is None")


__all__ = ["flags_router"]
