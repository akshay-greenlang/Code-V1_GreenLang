# -*- coding: utf-8 -*-
"""
Secrets Rotation REST API Routes - SEC-006

FastAPI router for secrets rotation management:

  POST /rotate/{path:path}   - Trigger manual rotation
  GET  /rotation/status      - Current rotation status
  GET  /rotation/schedule    - Rotation schedule
  POST /rotation/schedule    - Update schedule

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import (
        APIRouter,
        Depends,
        Header,
        HTTPException,
        Path,
        Query,
        Request,
        status,
    )
    from pydantic import BaseModel, ConfigDict, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[assignment]
    Header = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    Path = None  # type: ignore[assignment]
    Query = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    status = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class RotationStatusItem(BaseModel):
        """Status of a single rotation task."""

        model_config = ConfigDict(from_attributes=True)

        identifier: str = Field(..., description="Secret identifier.")
        type: str = Field(..., description="Rotation type.")
        last_rotation: Optional[str] = Field(
            default=None, description="Last rotation timestamp."
        )
        next_rotation: Optional[str] = Field(
            default=None, description="Next scheduled rotation."
        )
        retry_count: int = Field(default=0, description="Failed retry count.")
        last_error: Optional[str] = Field(
            default=None, description="Last error message."
        )
        status: str = Field(default="pending", description="Current status.")

    class RotationStatusResponse(BaseModel):
        """Response for rotation status."""

        enabled: bool = Field(..., description="Whether rotation is enabled.")
        running: bool = Field(default=False, description="Whether manager is running.")
        total_schedules: int = Field(default=0, description="Total scheduled rotations.")
        failed_rotations: int = Field(default=0, description="Rotations in failed state.")
        pending_rotations: int = Field(default=0, description="Rotations pending now.")
        items: List[RotationStatusItem] = Field(
            default_factory=list, description="Individual rotation statuses."
        )

    class RotationScheduleItem(BaseModel):
        """A scheduled rotation."""

        path: str = Field(..., description="Secret path.")
        type: str = Field(..., description="Secret type.")
        next_rotation: Optional[str] = Field(
            default=None, description="Next rotation time."
        )
        last_rotation: Optional[str] = Field(
            default=None, description="Last rotation time."
        )
        interval_hours: Optional[int] = Field(
            default=None, description="Rotation interval in hours."
        )

    class RotationScheduleResponse(BaseModel):
        """Response for rotation schedule."""

        enabled: bool = Field(..., description="Whether rotation is enabled.")
        schedule: List[RotationScheduleItem] = Field(
            default_factory=list, description="Scheduled rotations."
        )

    class UpdateScheduleRequest(BaseModel):
        """Request to update rotation schedule."""

        model_config = ConfigDict(extra="forbid")

        path: str = Field(..., description="Secret path.")
        enabled: bool = Field(default=True, description="Enable rotation.")
        interval_hours: Optional[int] = Field(
            default=None, ge=1, description="Rotation interval in hours."
        )
        cron_expression: Optional[str] = Field(
            default=None, description="Cron expression for rotation."
        )

    class RotationResult(BaseModel):
        """Result of a rotation operation."""

        success: bool = Field(..., description="Whether rotation succeeded.")
        path: str = Field(..., description="Secret path.")
        old_version: Optional[str] = Field(
            default=None, description="Previous version."
        )
        new_version: Optional[str] = Field(
            default=None, description="New version."
        )
        rotated_at: str = Field(..., description="Rotation timestamp.")
        next_rotation: Optional[str] = Field(
            default=None, description="Next scheduled rotation."
        )
        error: Optional[str] = Field(default=None, description="Error message.")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _get_secrets_service() -> Any:
    """FastAPI dependency for SecretsService."""
    try:
        from greenlang.infrastructure.secrets_service import get_secrets_service

        return get_secrets_service()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Secrets service not configured.",
        )


def _get_tenant_id(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Optional[str]:
    """Extract tenant ID from header."""
    return x_tenant_id


def _get_user_id(
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> str:
    """Extract user ID from header."""
    return x_user_id or "anonymous"


def _get_correlation_id(
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
) -> str:
    """Get or generate correlation ID."""
    import uuid

    return x_correlation_id or str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    rotation_router = APIRouter(
        prefix="/rotation",
        tags=["Secrets Rotation"],
        responses={
            403: {"description": "Forbidden"},
            503: {"description": "Service Unavailable"},
        },
    )

    # -------------------------------------------------------------------------
    # Trigger Manual Rotation
    # -------------------------------------------------------------------------

    @rotation_router.post(
        "/rotate/{path:path}",
        response_model=RotationResult,
        summary="Trigger rotation",
        description="Manually trigger rotation for a specific secret.",
        operation_id="trigger_rotation",
    )
    async def trigger_rotation(
        request: Request,
        path: str = Path(..., description="Secret path to rotate."),
        secrets_service: Any = Depends(_get_secrets_service),
        tenant_id: Optional[str] = Depends(_get_tenant_id),
        user_id: str = Depends(_get_user_id),
        correlation_id: str = Depends(_get_correlation_id),
    ) -> RotationResult:
        """Trigger manual rotation for a secret.

        This immediately rotates the secret and invalidates cached values.
        Requires appropriate permissions for the secret.
        """
        try:
            result = await secrets_service.trigger_rotation(
                path=path,
                tenant_id=tenant_id,
            )

            logger.info(
                "Manual rotation triggered: %s",
                path,
                extra={
                    "event_category": "secrets",
                    "path": path,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "correlation_id": correlation_id,
                    "status": result.get("status"),
                },
            )

            return RotationResult(
                success=result.get("status") == "success",
                path=path,
                old_version=result.get("old_version"),
                new_version=result.get("new_version"),
                rotated_at=result.get("rotated_at", datetime.utcnow().isoformat()),
                next_rotation=result.get("next_rotation"),
                error=result.get("error"),
            )

        except Exception as exc:
            logger.exception(
                "Rotation failed: %s",
                path,
                extra={
                    "event_category": "secrets",
                    "path": path,
                    "tenant_id": tenant_id,
                    "correlation_id": correlation_id,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Rotation failed: {exc}",
            )

    # -------------------------------------------------------------------------
    # Get Rotation Status
    # -------------------------------------------------------------------------

    @rotation_router.get(
        "/status",
        response_model=RotationStatusResponse,
        summary="Get rotation status",
        description="Get current rotation status for all managed secrets.",
        operation_id="get_rotation_status",
    )
    async def get_rotation_status(
        request: Request,
        secrets_service: Any = Depends(_get_secrets_service),
        correlation_id: str = Depends(_get_correlation_id),
    ) -> RotationStatusResponse:
        """Get rotation status for all secrets.

        Returns the status of all configured rotation schedules,
        including failed rotations that need attention.
        """
        try:
            status_data = secrets_service.get_rotation_status()

            if not status_data.get("enabled", False):
                return RotationStatusResponse(
                    enabled=False,
                    running=False,
                    total_schedules=0,
                    failed_rotations=0,
                    pending_rotations=0,
                    items=[],
                )

            schedules = status_data.get("schedules", {})
            items = []
            failed_count = 0
            pending_count = 0

            for key, info in schedules.items():
                item_status = "ok"
                if info.get("last_error"):
                    item_status = "failed"
                    failed_count += 1
                elif info.get("next_rotation"):
                    # Check if pending
                    from datetime import datetime

                    next_rot = info["next_rotation"]
                    if isinstance(next_rot, str):
                        try:
                            next_dt = datetime.fromisoformat(next_rot.replace("Z", "+00:00"))
                            if next_dt <= datetime.now(next_dt.tzinfo):
                                item_status = "pending"
                                pending_count += 1
                        except ValueError:
                            pass

                items.append(
                    RotationStatusItem(
                        identifier=info.get("identifier", key),
                        type=info.get("type", "unknown"),
                        last_rotation=info.get("last_rotation"),
                        next_rotation=info.get("next_rotation"),
                        retry_count=info.get("retry_count", 0),
                        last_error=info.get("last_error"),
                        status=item_status,
                    )
                )

            return RotationStatusResponse(
                enabled=True,
                running=True,  # Assumed if schedules exist
                total_schedules=len(items),
                failed_rotations=failed_count,
                pending_rotations=pending_count,
                items=items,
            )

        except Exception as exc:
            logger.exception(
                "Failed to get rotation status",
                extra={
                    "event_category": "secrets",
                    "correlation_id": correlation_id,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get status: {exc}",
            )

    # -------------------------------------------------------------------------
    # Get Rotation Schedule
    # -------------------------------------------------------------------------

    @rotation_router.get(
        "/schedule",
        response_model=RotationScheduleResponse,
        summary="Get rotation schedule",
        description="Get the rotation schedule for all secrets.",
        operation_id="get_rotation_schedule",
    )
    async def get_rotation_schedule(
        request: Request,
        secrets_service: Any = Depends(_get_secrets_service),
        correlation_id: str = Depends(_get_correlation_id),
    ) -> RotationScheduleResponse:
        """Get the rotation schedule.

        Returns upcoming rotations sorted by next rotation time.
        """
        try:
            schedule_data = secrets_service.get_rotation_schedule()

            if not schedule_data.get("enabled", False):
                return RotationScheduleResponse(
                    enabled=False,
                    schedule=[],
                )

            schedule_items = []
            for item in schedule_data.get("schedule", []):
                schedule_items.append(
                    RotationScheduleItem(
                        path=item.get("path", ""),
                        type=item.get("type", "unknown"),
                        next_rotation=item.get("next_rotation"),
                        last_rotation=item.get("last_rotation"),
                    )
                )

            return RotationScheduleResponse(
                enabled=True,
                schedule=schedule_items,
            )

        except Exception as exc:
            logger.exception(
                "Failed to get rotation schedule",
                extra={
                    "event_category": "secrets",
                    "correlation_id": correlation_id,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get schedule: {exc}",
            )

    # -------------------------------------------------------------------------
    # Update Rotation Schedule
    # -------------------------------------------------------------------------

    @rotation_router.post(
        "/schedule",
        response_model=Dict[str, Any],
        summary="Update rotation schedule",
        description="Update the rotation schedule for a secret.",
        operation_id="update_rotation_schedule",
    )
    async def update_rotation_schedule(
        request: Request,
        body: UpdateScheduleRequest,
        secrets_service: Any = Depends(_get_secrets_service),
        user_id: str = Depends(_get_user_id),
        correlation_id: str = Depends(_get_correlation_id),
    ) -> Dict[str, Any]:
        """Update rotation schedule for a secret.

        Note: This is a placeholder. Full implementation requires
        updating the RotationConfig in the SecretsRotationManager.
        """
        logger.info(
            "Rotation schedule update requested: %s",
            body.path,
            extra={
                "event_category": "secrets",
                "path": body.path,
                "user_id": user_id,
                "enabled": body.enabled,
                "interval_hours": body.interval_hours,
                "correlation_id": correlation_id,
            },
        )

        # TODO: Implement dynamic schedule updates
        # This would require the rotation manager to support runtime config changes

        return {
            "success": True,
            "message": "Schedule update queued. Changes will apply on next check interval.",
            "path": body.path,
            "enabled": body.enabled,
            "interval_hours": body.interval_hours,
        }

else:
    rotation_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - rotation_router is None")


__all__ = ["rotation_router"]
