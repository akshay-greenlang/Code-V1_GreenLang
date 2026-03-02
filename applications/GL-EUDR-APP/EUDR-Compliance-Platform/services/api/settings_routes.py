"""
Settings Management API Routes for GL-EUDR-APP v1.0

Manages application-level settings for the EUDR compliance platform.
Settings include risk thresholds, notification preferences, default
commodities, API configuration, and compliance parameters.

Prefix: /api/v1/settings
Tags: Settings
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/settings", tags=["Settings"])

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class RiskThresholds(BaseModel):
    """Risk score thresholds for level classification."""

    low_max: float = Field(39.9, description="Maximum score for 'low' risk")
    medium_max: float = Field(59.9, description="Maximum score for 'medium' risk")
    high_max: float = Field(79.9, description="Maximum score for 'high' risk")
    critical_min: float = Field(80.0, description="Minimum score for 'critical' risk")


class NotificationSettings(BaseModel):
    """Notification delivery preferences."""

    email_enabled: bool = Field(True, description="Enable email notifications")
    email_recipients: List[str] = Field(
        default_factory=list, description="Email recipients for alerts"
    )
    slack_enabled: bool = Field(False, description="Enable Slack notifications")
    slack_webhook_url: Optional[str] = Field(None, description="Slack webhook URL")
    alert_on_critical: bool = Field(True, description="Alert on critical risk events")
    alert_on_deforestation: bool = Field(True, description="Alert on deforestation detections")
    alert_on_dds_deadline: bool = Field(True, description="Alert on DDS submission deadlines")
    digest_frequency: str = Field(
        "daily", description="Digest frequency: daily | weekly | none"
    )


class ComplianceSettings(BaseModel):
    """EUDR compliance configuration."""

    default_cutoff_date: str = Field(
        "2020-12-31", description="EUDR deforestation cutoff date (YYYY-MM-DD)"
    )
    dds_auto_validate: bool = Field(
        True, description="Automatically validate DDS on generation"
    )
    required_document_types: List[str] = Field(
        default_factory=lambda: [
            "CERTIFICATE", "PERMIT", "LAND_TITLE", "INVOICE", "TRANSPORT"
        ],
        description="Document types required for compliance",
    )
    assessment_validity_days: int = Field(
        90, ge=30, le=365, description="Risk assessment validity period in days"
    )
    commodities: List[str] = Field(
        default_factory=lambda: [
            "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"
        ],
        description="Enabled EUDR-regulated commodities",
    )


class PipelineSettings(BaseModel):
    """Pipeline execution configuration."""

    max_concurrent_runs: int = Field(5, ge=1, le=50, description="Max concurrent pipeline runs")
    default_priority: str = Field("normal", description="Default pipeline priority")
    stage_timeout_seconds: int = Field(
        300, ge=60, le=3600, description="Per-stage timeout in seconds"
    )
    retry_max_attempts: int = Field(3, ge=1, le=10, description="Max retry attempts")
    retry_backoff_seconds: int = Field(
        30, ge=5, le=300, description="Backoff between retries in seconds"
    )


class ApplicationSettings(BaseModel):
    """Complete application settings.

    Example response::

        {
            "app_name": "GL-EUDR-APP",
            "app_version": "1.0.0",
            "environment": "development",
            "risk_thresholds": {...},
            "notifications": {...},
            "compliance": {...},
            "pipeline": {...},
            "data_retention_days": 2555,
            "timezone": "UTC",
            "last_modified": "2025-11-09T10:30:00Z"
        }
    """

    app_name: str = Field("GL-EUDR-APP", description="Application name")
    app_version: str = Field("1.0.0", description="Application version")
    environment: str = Field(
        "development", description="Environment: development | staging | production"
    )
    risk_thresholds: RiskThresholds = Field(default_factory=RiskThresholds)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    compliance: ComplianceSettings = Field(default_factory=ComplianceSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    data_retention_days: int = Field(
        2555, ge=365, le=3650, description="Data retention period (7 years default)"
    )
    timezone: str = Field("UTC", description="Application timezone")
    last_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last settings modification timestamp",
    )


class SettingsUpdateRequest(BaseModel):
    """Request to update application settings.

    Only provided fields are updated; omitted fields retain current values.
    """

    environment: Optional[str] = None
    risk_thresholds: Optional[RiskThresholds] = None
    notifications: Optional[NotificationSettings] = None
    compliance: Optional[ComplianceSettings] = None
    pipeline: Optional[PipelineSettings] = None
    data_retention_days: Optional[int] = Field(None, ge=365, le=3650)
    timezone: Optional[str] = None


# ---------------------------------------------------------------------------
# In-Memory Storage (v1.0)
# ---------------------------------------------------------------------------

_settings: Optional[dict] = None


def _get_settings() -> ApplicationSettings:
    """Retrieve current settings, initializing with defaults if needed."""
    global _settings
    if _settings is None:
        _settings = ApplicationSettings().model_dump(mode="json")
        # Ensure datetime is proper
        _settings["last_modified"] = datetime.now(timezone.utc).isoformat()
    return ApplicationSettings(**_settings)


def _get_default_settings() -> ApplicationSettings:
    """Return factory-default settings."""
    return ApplicationSettings()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=ApplicationSettings,
    summary="Get settings",
    description="Retrieve current application settings.",
)
async def get_settings() -> ApplicationSettings:
    """
    Retrieve the current application settings.

    Returns default values on first access. Settings persist in memory
    for the lifetime of the application process.

    Returns:
        200 with current settings.
    """
    return _get_settings()


@router.put(
    "/",
    response_model=ApplicationSettings,
    summary="Update settings",
    description="Update application settings. Only provided fields are modified.",
)
async def update_settings(body: SettingsUpdateRequest) -> ApplicationSettings:
    """
    Partially update application settings.

    Only fields present in the request body are updated. Omitted fields
    retain their current values.

    Returns:
        200 with updated settings.
    """
    global _settings
    current = _get_settings()
    current_dict = current.model_dump(mode="json")

    update_data = body.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        if value is not None:
            if isinstance(value, dict):
                # Merge nested settings
                if key in current_dict and isinstance(current_dict[key], dict):
                    current_dict[key].update(value)
                else:
                    current_dict[key] = value
            else:
                current_dict[key] = value

    current_dict["last_modified"] = datetime.now(timezone.utc).isoformat()
    _settings = current_dict

    logger.info("Settings updated: %s", list(update_data.keys()))
    return ApplicationSettings(**_settings)


@router.get(
    "/defaults",
    response_model=ApplicationSettings,
    summary="Get default settings",
    description="Retrieve factory-default application settings.",
)
async def get_default_settings() -> ApplicationSettings:
    """
    Return the factory-default settings.

    This does not modify the current active settings -- it provides a
    reference for what the defaults are.

    Returns:
        200 with default settings.
    """
    return _get_default_settings()
