# -*- coding: utf-8 -*-
"""
User Settings API Routes - GL-VCCI Scope 3 Platform v1.1.0

FastAPI router providing endpoints for user-level application settings.
Manages preferences for reporting, display, notifications, and defaults.

Endpoints:
    GET  /api/v1/settings/          - Get current user settings
    PUT  /api/v1/settings/          - Update user settings
    GET  /api/v1/settings/defaults  - Get default settings

Version: 1.1.0
Date: 2026-03-01
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================================================
# ROUTER SETUP
# ============================================================================

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserProfile(BaseModel):
    """User profile information."""
    display_name: str = Field(default="", description="User display name")
    email: str = Field(default="", description="User email address")
    role: str = Field(default="analyst", description="User role (admin, manager, analyst, viewer)")
    organization: str = Field(default="", description="Organization name")
    timezone: str = Field(default="UTC", description="User timezone (IANA format)")
    locale: str = Field(default="en-US", description="User locale for number/date formatting")


class ReportingPreferences(BaseModel):
    """Preferences for reporting and disclosure generation."""
    default_standard: str = Field(default="ghg_protocol", description="Default reporting standard")
    default_export_format: str = Field(default="pdf", description="Default export format (pdf, excel, json)")
    include_charts: bool = Field(default=True, description="Include charts in reports")
    include_provenance: bool = Field(default=True, description="Include provenance hashes")
    consolidation_approach: str = Field(default="operational_control", description="GHG inventory boundary approach")
    base_year: Optional[int] = Field(None, description="Base year for emissions tracking")
    currency: str = Field(default="USD", description="Reporting currency")
    emission_unit: str = Field(default="tCO2e", description="Emission unit (tCO2e, kgCO2e, MtCO2e)")
    gwp_source: str = Field(default="IPCC_AR5", description="GWP values source (IPCC_AR4, IPCC_AR5, IPCC_AR6)")
    scope3_categories_reported: List[int] = Field(
        default_factory=lambda: list(range(1, 16)),
        description="Scope 3 categories to include (1-15)",
    )


class DisplayPreferences(BaseModel):
    """UI display preferences."""
    theme: str = Field(default="light", description="UI theme (light, dark, system)")
    sidebar_collapsed: bool = Field(default=False, description="Whether sidebar is collapsed")
    dashboard_layout: str = Field(default="standard", description="Dashboard layout (standard, compact, detailed)")
    decimal_places: int = Field(default=1, ge=0, le=6, description="Decimal places for numbers")
    chart_color_scheme: str = Field(default="default", description="Chart color scheme")
    date_format: str = Field(default="YYYY-MM-DD", description="Date display format")
    number_format: str = Field(default="1,234.56", description="Number display format")
    items_per_page: int = Field(default=25, ge=10, le=100, description="Table items per page")


class NotificationPreferences(BaseModel):
    """Notification preferences."""
    email_notifications: bool = Field(default=True, description="Enable email notifications")
    report_completion: bool = Field(default=True, description="Notify when reports complete")
    validation_warnings: bool = Field(default=True, description="Notify on validation warnings")
    compliance_alerts: bool = Field(default=True, description="Notify on compliance score changes")
    data_quality_alerts: bool = Field(default=True, description="Notify on data quality issues")
    weekly_digest: bool = Field(default=False, description="Send weekly summary digest")
    notification_channels: List[str] = Field(
        default_factory=lambda: ["in_app"],
        description="Notification channels (in_app, email, slack)",
    )


class CDPPreferences(BaseModel):
    """CDP-specific preferences."""
    auto_populate_on_data_change: bool = Field(default=True, description="Auto-populate when data changes")
    show_score_prediction: bool = Field(default=True, description="Display score prediction widget")
    highlight_data_gaps: bool = Field(default=True, description="Highlight unanswered required questions")
    default_export_format: str = Field(default="excel", description="Default CDP export format")
    compare_with_previous_year: bool = Field(default=True, description="Show year-over-year comparison")


class CompliancePreferences(BaseModel):
    """Compliance scorecard preferences."""
    standards_to_assess: List[str] = Field(
        default_factory=lambda: ["ghg_protocol", "esrs_e1", "cdp", "ifrs_s2", "iso_14083"],
        description="Standards to include in scorecard",
    )
    show_action_items: bool = Field(default=True, description="Display action items")
    gap_severity_filter: str = Field(default="all", description="Minimum gap severity to show (all, critical, high, medium)")
    auto_refresh_scorecard: bool = Field(default=False, description="Regenerate scorecard when data changes")


class Settings(BaseModel):
    """Complete user settings."""
    profile: UserProfile = Field(default_factory=UserProfile, description="User profile")
    reporting: ReportingPreferences = Field(default_factory=ReportingPreferences, description="Reporting preferences")
    display: DisplayPreferences = Field(default_factory=DisplayPreferences, description="Display preferences")
    notifications: NotificationPreferences = Field(default_factory=NotificationPreferences, description="Notification preferences")
    cdp: CDPPreferences = Field(default_factory=CDPPreferences, description="CDP-specific preferences")
    compliance: CompliancePreferences = Field(default_factory=CompliancePreferences, description="Compliance preferences")
    last_updated: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Last settings update timestamp",
    )
    version: str = Field(default="1.1.0", description="Settings schema version")


class SettingsUpdateRequest(BaseModel):
    """Partial settings update request. Only provided fields are updated."""
    profile: Optional[UserProfile] = Field(None, description="User profile updates")
    reporting: Optional[ReportingPreferences] = Field(None, description="Reporting preference updates")
    display: Optional[DisplayPreferences] = Field(None, description="Display preference updates")
    notifications: Optional[NotificationPreferences] = Field(None, description="Notification preference updates")
    cdp: Optional[CDPPreferences] = Field(None, description="CDP preference updates")
    compliance: Optional[CompliancePreferences] = Field(None, description="Compliance preference updates")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")


# ============================================================================
# IN-MEMORY STORAGE
# ============================================================================

# Default settings (read-only)
_default_settings = Settings()

# Current user settings (mutable)
_current_settings = Settings()


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get(
    "/",
    response_model=Settings,
    summary="Get current user settings",
)
async def get_settings() -> Settings:
    """
    Retrieve the current user settings.

    Returns the complete settings object including profile, reporting,
    display, notification, CDP, and compliance preferences.

    Returns:
        Settings with all current preferences.
    """
    logger.info("Retrieved user settings (last updated: %s)", _current_settings.last_updated)
    return _current_settings


@router.put(
    "/",
    response_model=Settings,
    summary="Update user settings",
)
async def update_settings(body: SettingsUpdateRequest) -> Settings:
    """
    Update user settings with partial or full updates.

    Only the fields provided in the request body are updated. Fields
    not included remain unchanged. The last_updated timestamp is
    automatically updated.

    Args:
        body: Partial settings update request.

    Returns:
        Updated Settings object.
    """
    global _current_settings

    if body.profile is not None:
        _current_settings.profile = body.profile
    if body.reporting is not None:
        _current_settings.reporting = body.reporting
    if body.display is not None:
        _current_settings.display = body.display
    if body.notifications is not None:
        _current_settings.notifications = body.notifications
    if body.cdp is not None:
        _current_settings.cdp = body.cdp
    if body.compliance is not None:
        _current_settings.compliance = body.compliance

    _current_settings.last_updated = datetime.utcnow().isoformat()

    logger.info("Updated user settings at %s", _current_settings.last_updated)
    return _current_settings


@router.get(
    "/defaults",
    response_model=Settings,
    summary="Get default settings",
)
async def get_defaults() -> Settings:
    """
    Retrieve the default settings.

    Returns the factory-default settings object. Useful for resetting
    individual preference categories to their default values.

    Returns:
        Settings with all default preferences.
    """
    logger.info("Retrieved default settings")
    return _default_settings


__all__ = ["router"]
