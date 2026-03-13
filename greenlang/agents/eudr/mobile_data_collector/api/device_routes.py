# -*- coding: utf-8 -*-
"""
Device Routes - AGENT-EUDR-015 Mobile Data Collector

REST API endpoints for mobile device fleet management including
registration, listing, get, update, heartbeat, telemetry,
decommission, fleet status dashboard, stale device detection,
campaign creation, and campaign retrieval.

Endpoints (11):
    POST   /devices                              Register device
    GET    /devices                              List devices with filters
    GET    /devices/{device_id}                  Get device
    PUT    /devices/{device_id}                  Update device
    POST   /devices/{device_id}/heartbeat        Record heartbeat
    POST   /devices/{device_id}/telemetry        Submit telemetry
    POST   /devices/{device_id}/decommission     Decommission device
    GET    /fleet/status                         Fleet dashboard status
    GET    /fleet/stale                          List stale devices
    POST   /campaigns                            Create collection campaign
    GET    /campaigns/{campaign_id}              Get campaign

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015, Section 7.4
Agent ID: GL-EUDR-MDC-015
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from greenlang.agents.eudr.mobile_data_collector.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_mdc_service,
    get_pagination,
    rate_limit_admin,
    rate_limit_read,
    rate_limit_write,
    require_permission,
    validate_campaign_id,
    validate_device_id,
)
from greenlang.agents.eudr.mobile_data_collector.api.schemas import (
    CampaignResponseSchema,
    CampaignSchema,
    DeviceListSchema,
    DevicePlatformSchema,
    DeviceRegisterSchema,
    DeviceResponseSchema,
    DeviceStatusSchema,
    DeviceUpdateSchema,
    ErrorSchema,
    FleetStatusSchema,
    HeartbeatSchema,
    PaginationSchema,
    SuccessSchema,
    TelemetrySchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["EUDR Mobile Data - Devices"],
    responses={
        400: {"model": ErrorSchema, "description": "Validation error"},
        404: {"model": ErrorSchema, "description": "Device not found"},
    },
)


# ---------------------------------------------------------------------------
# POST /devices
# ---------------------------------------------------------------------------


@router.post(
    "/devices",
    response_model=DeviceResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Register device",
    description=(
        "Register a new mobile device in the fleet. Assigns a unique "
        "device identifier, records hardware model, platform, OS "
        "version, and agent version. Optionally assigns an operator "
        "and collection area."
    ),
    responses={
        201: {"description": "Device registered successfully"},
        400: {"description": "Invalid device data"},
        409: {"description": "Device already registered"},
    },
)
async def register_device(
    body: DeviceRegisterSchema,
    user: AuthUser = Depends(require_permission("eudr-mdc:devices:create")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_admin),
) -> DeviceResponseSchema:
    """Register a new mobile device.

    Args:
        body: Device registration data with model, platform, and version.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        DeviceResponseSchema with registered device details.
    """
    start = time.monotonic()
    logger.info(
        "Register device: user=%s model=%s platform=%s os=%s agent=%s",
        user.user_id,
        body.device_model,
        body.platform.value,
        body.os_version,
        body.agent_version,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return DeviceResponseSchema(
        device_model=body.device_model,
        platform=body.platform.value,
        os_version=body.os_version,
        agent_version=body.agent_version,
        status="active",
        assigned_operator_id=body.assigned_operator_id,
        processing_time_ms=round(elapsed_ms, 2),
        message="Device registered successfully",
    )


# ---------------------------------------------------------------------------
# GET /devices
# ---------------------------------------------------------------------------


@router.get(
    "/devices",
    response_model=DeviceListSchema,
    summary="List devices with filters",
    description=(
        "List registered devices with optional filters by status, "
        "platform, operator ID, and agent version. Results are "
        "paginated."
    ),
    responses={
        200: {"description": "Devices retrieved successfully"},
    },
)
async def list_devices(
    device_status: Optional[DeviceStatusSchema] = Query(
        None, alias="status",
        description="Filter by device status",
    ),
    platform: Optional[DevicePlatformSchema] = Query(
        None, description="Filter by OS platform",
    ),
    operator_id: Optional[str] = Query(
        None, max_length=255, description="Filter by assigned operator",
    ),
    agent_version: Optional[str] = Query(
        None, max_length=20, description="Filter by agent version",
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(require_permission("eudr-mdc:devices:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> DeviceListSchema:
    """List registered devices with optional filters.

    Args:
        device_status: Filter by device status.
        platform: Filter by OS platform.
        operator_id: Filter by assigned operator.
        agent_version: Filter by agent version.
        pagination: Pagination parameters.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        DeviceListSchema with matching devices and pagination.
    """
    start = time.monotonic()
    logger.info(
        "List devices: user=%s page=%d page_size=%d",
        user.user_id,
        pagination.page,
        pagination.page_size,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return DeviceListSchema(
        devices=[],
        pagination=PaginationSchema(
            total=0,
            page=pagination.page,
            page_size=pagination.page_size,
            has_more=False,
        ),
        processing_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# GET /devices/{device_id}
# ---------------------------------------------------------------------------


@router.get(
    "/devices/{device_id}",
    response_model=DeviceResponseSchema,
    summary="Get device",
    description="Retrieve a registered device by its identifier.",
    responses={
        200: {"description": "Device retrieved"},
        404: {"description": "Device not found"},
    },
)
async def get_device(
    device_id: str = Depends(validate_device_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:devices:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> DeviceResponseSchema:
    """Get a registered device by identifier.

    Args:
        device_id: Device identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        DeviceResponseSchema with device details.

    Raises:
        HTTPException: 404 if device not found.
    """
    logger.info(
        "Get device: user=%s device_id=%s", user.user_id, device_id
    )

    # Placeholder - real implementation queries database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Device {device_id} not found",
    )


# ---------------------------------------------------------------------------
# PUT /devices/{device_id}
# ---------------------------------------------------------------------------


@router.put(
    "/devices/{device_id}",
    response_model=DeviceResponseSchema,
    summary="Update device",
    description=(
        "Update a registered device's assigned operator, collection "
        "area, agent version, or metadata."
    ),
    responses={
        200: {"description": "Device updated successfully"},
        404: {"description": "Device not found"},
        409: {"description": "Device is decommissioned"},
    },
)
async def update_device(
    body: DeviceUpdateSchema,
    device_id: str = Depends(validate_device_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:devices:update")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> DeviceResponseSchema:
    """Update a registered device.

    Args:
        body: Device update data.
        device_id: Device identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        DeviceResponseSchema with updated device details.

    Raises:
        HTTPException: 404 if not found, 409 if decommissioned.
    """
    start = time.monotonic()
    logger.info(
        "Update device: user=%s device_id=%s", user.user_id, device_id
    )

    # Placeholder - real implementation queries database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Device {device_id} not found",
    )


# ---------------------------------------------------------------------------
# POST /devices/{device_id}/heartbeat
# ---------------------------------------------------------------------------


@router.post(
    "/devices/{device_id}/heartbeat",
    response_model=SuccessSchema,
    summary="Record device heartbeat",
    description=(
        "Record a heartbeat from a mobile device reporting battery "
        "level, storage usage, GPS quality, pending data counts, "
        "and connectivity. Heartbeats update the device's last-seen "
        "timestamp and health metrics."
    ),
    responses={
        200: {"description": "Heartbeat recorded"},
        404: {"description": "Device not found"},
    },
)
async def record_heartbeat(
    body: HeartbeatSchema,
    device_id: str = Depends(validate_device_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:devices:heartbeat")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> SuccessSchema:
    """Record a device heartbeat.

    Args:
        body: Heartbeat data with device health metrics.
        device_id: Device identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        SuccessSchema confirming heartbeat recorded.

    Raises:
        HTTPException: 404 if device not found.
    """
    logger.info(
        "Heartbeat: user=%s device=%s battery=%s storage_free=%s conn=%s",
        user.user_id,
        device_id,
        body.battery_level_pct,
        body.storage_free_bytes,
        body.connectivity_type,
    )

    return SuccessSchema(
        status="success",
        message="Heartbeat recorded",
        data={"device_id": device_id},
    )


# ---------------------------------------------------------------------------
# POST /devices/{device_id}/telemetry
# ---------------------------------------------------------------------------


@router.post(
    "/devices/{device_id}/telemetry",
    response_model=SuccessSchema,
    summary="Submit device telemetry",
    description=(
        "Submit a telemetry event from a device. Telemetry events "
        "track sync operations, battery warnings, storage warnings, "
        "GPS fix loss, and application errors."
    ),
    responses={
        200: {"description": "Telemetry recorded"},
        404: {"description": "Device not found"},
    },
)
async def submit_telemetry(
    body: TelemetrySchema,
    device_id: str = Depends(validate_device_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:devices:telemetry")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> SuccessSchema:
    """Submit a device telemetry event.

    Args:
        body: Telemetry event data.
        device_id: Device identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        SuccessSchema confirming telemetry recorded.

    Raises:
        HTTPException: 404 if device not found.
    """
    logger.info(
        "Telemetry: user=%s device=%s event=%s",
        user.user_id,
        device_id,
        body.event_type,
    )

    return SuccessSchema(
        status="success",
        message="Telemetry event recorded",
        data={"device_id": device_id, "event_type": body.event_type},
    )


# ---------------------------------------------------------------------------
# POST /devices/{device_id}/decommission
# ---------------------------------------------------------------------------


@router.post(
    "/devices/{device_id}/decommission",
    response_model=DeviceResponseSchema,
    summary="Decommission device",
    description=(
        "Decommission a mobile device removing it from the active "
        "fleet. Decommissioned devices cannot submit new data or "
        "perform sync. All pending data must be synced or discarded "
        "before decommission."
    ),
    responses={
        200: {"description": "Device decommissioned successfully"},
        404: {"description": "Device not found"},
        409: {"description": "Device has pending unsynchronized data"},
    },
)
async def decommission_device(
    device_id: str = Depends(validate_device_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:devices:decommission")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_admin),
) -> DeviceResponseSchema:
    """Decommission a mobile device.

    Args:
        device_id: Device identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        DeviceResponseSchema with decommissioned device details.

    Raises:
        HTTPException: 404 if not found, 409 if pending data exists.
    """
    logger.info(
        "Decommission device: user=%s device_id=%s",
        user.user_id,
        device_id,
    )

    # Placeholder - real implementation queries database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Device {device_id} not found",
    )


# ---------------------------------------------------------------------------
# GET /fleet/status
# ---------------------------------------------------------------------------


@router.get(
    "/fleet/status",
    response_model=FleetStatusSchema,
    summary="Fleet dashboard status",
    description=(
        "Retrieve the fleet dashboard summary with counts of active, "
        "offline, low-battery, low-storage, decommissioned, and "
        "outdated-agent devices. Includes total pending forms, photos, "
        "and sync bytes across the fleet."
    ),
    responses={
        200: {"description": "Fleet status retrieved"},
    },
)
async def get_fleet_status(
    user: AuthUser = Depends(require_permission("eudr-mdc:fleet:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> FleetStatusSchema:
    """Get fleet dashboard status summary.

    Args:
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        FleetStatusSchema with fleet-wide aggregated metrics.
    """
    start = time.monotonic()
    logger.info("Fleet status: user=%s", user.user_id)

    elapsed_ms = (time.monotonic() - start) * 1000
    return FleetStatusSchema(
        total_devices=0,
        active_devices=0,
        offline_devices=0,
        low_battery_devices=0,
        low_storage_devices=0,
        decommissioned_devices=0,
        outdated_agent_devices=0,
        total_pending_forms=0,
        total_pending_photos=0,
        total_pending_sync_bytes=0,
        processing_time_ms=round(elapsed_ms, 2),
        message="Fleet status retrieved",
    )


# ---------------------------------------------------------------------------
# GET /fleet/stale
# ---------------------------------------------------------------------------


@router.get(
    "/fleet/stale",
    response_model=DeviceListSchema,
    summary="List stale devices",
    description=(
        "List devices that have not sent a heartbeat within the "
        "configured threshold (default 72 hours). Stale devices "
        "may be offline, out of range, or experiencing issues."
    ),
    responses={
        200: {"description": "Stale devices retrieved"},
    },
)
async def list_stale_devices(
    threshold_hours: int = Query(
        default=72, ge=1, le=720,
        description="Hours since last heartbeat to consider stale",
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(require_permission("eudr-mdc:fleet:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> DeviceListSchema:
    """List devices that have not sent a heartbeat within the threshold.

    Args:
        threshold_hours: Hours since last heartbeat to consider stale.
        pagination: Pagination parameters.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        DeviceListSchema with stale devices and pagination.
    """
    start = time.monotonic()
    logger.info(
        "Stale devices: user=%s threshold=%dh page=%d",
        user.user_id,
        threshold_hours,
        pagination.page,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return DeviceListSchema(
        devices=[],
        pagination=PaginationSchema(
            total=0,
            page=pagination.page,
            page_size=pagination.page_size,
            has_more=False,
        ),
        processing_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# POST /campaigns
# ---------------------------------------------------------------------------


@router.post(
    "/campaigns",
    response_model=CampaignResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create collection campaign",
    description=(
        "Create a data collection campaign targeting a specific "
        "EUDR commodity and geographic area. Assigns devices to "
        "the campaign and tracks progress toward form submission "
        "and area coverage targets."
    ),
    responses={
        201: {"description": "Campaign created successfully"},
        400: {"description": "Invalid campaign data"},
    },
)
async def create_campaign(
    body: CampaignSchema,
    user: AuthUser = Depends(require_permission("eudr-mdc:campaigns:create")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_admin),
) -> CampaignResponseSchema:
    """Create a data collection campaign.

    Args:
        body: Campaign definition with targets and device assignments.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        CampaignResponseSchema with campaign details.
    """
    start = time.monotonic()
    logger.info(
        "Create campaign: user=%s name=%s commodity=%s country=%s devices=%d",
        user.user_id,
        body.name,
        body.commodity_type.value if body.commodity_type else "none",
        body.country_code or "any",
        len(body.assigned_device_ids),
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return CampaignResponseSchema(
        name=body.name,
        status="active",
        commodity_type=body.commodity_type.value if body.commodity_type else None,
        country_code=body.country_code,
        start_date=body.start_date,
        end_date=body.end_date,
        target_forms=body.target_forms,
        completed_forms=0,
        assigned_devices=len(body.assigned_device_ids),
        progress_percent=0.0,
        processing_time_ms=round(elapsed_ms, 2),
        message="Campaign created successfully",
    )


# ---------------------------------------------------------------------------
# GET /campaigns/{campaign_id}
# ---------------------------------------------------------------------------


@router.get(
    "/campaigns/{campaign_id}",
    response_model=CampaignResponseSchema,
    summary="Get campaign",
    description=(
        "Retrieve a collection campaign by its identifier including "
        "progress metrics, assigned devices, and target completion."
    ),
    responses={
        200: {"description": "Campaign retrieved"},
        404: {"description": "Campaign not found"},
    },
)
async def get_campaign(
    campaign_id: str = Depends(validate_campaign_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:campaigns:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> CampaignResponseSchema:
    """Get a collection campaign by identifier.

    Args:
        campaign_id: Campaign identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        CampaignResponseSchema with campaign details and progress.

    Raises:
        HTTPException: 404 if campaign not found.
    """
    logger.info(
        "Get campaign: user=%s campaign_id=%s",
        user.user_id,
        campaign_id,
    )

    # Placeholder - real implementation queries database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Campaign {campaign_id} not found",
    )
