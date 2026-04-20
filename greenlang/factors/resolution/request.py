# -*- coding: utf-8 -*-
"""Resolution request — enforces non-negotiable #6 (method_profile required)."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Optional

from pydantic import Field, model_validator

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.schemas.base import GreenLangBase


class ResolutionRequest(GreenLangBase):
    """A single resolution request.

    ``method_profile`` is **required** — callers cannot ask the engine for
    a raw factor without first choosing a methodology.  This is CTO
    non-negotiable #6.
    """

    activity: str = Field(
        ...,
        description=(
            "Free-text activity description or canonical activity_id. "
            "Examples: 'diesel combustion stationary', 'purchased electricity', "
            "'road freight 40 t truck'."
        ),
    )
    method_profile: MethodProfile = Field(
        ...,
        description=(
            "Required methodology profile — enforces non-negotiable #6. "
            "Bind callers to a profile before they see a factor."
        ),
    )
    jurisdiction: Optional[str] = Field(
        default=None,
        description="ISO country/region code (e.g. 'IN', 'US-CA', 'EU')",
    )
    reporting_date: Optional[str] = Field(
        default=None,
        description="Reporting period end (ISO-8601 date). Defaults to 'today'.",
    )
    activity_id: Optional[str] = Field(
        default=None,
        description="Customer-stable activity identifier, used for tenant-overlay lookup.",
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for the customer-override step.",
    )
    supplier_id: Optional[str] = Field(
        default=None, description="Supplier / manufacturer id for step 2."
    )
    facility_id: Optional[str] = Field(
        default=None, description="Facility / asset id for step 3."
    )
    utility_or_grid_region: Optional[str] = Field(
        default=None,
        description=(
            "Utility tariff id or grid sub-region (e.g. 'eGRID-SERC', 'CEA-S') "
            "for step 4."
        ),
    )
    preferred_sources: Optional[list[str]] = Field(
        default=None,
        description=(
            "Prefer these source_ids over others when all else is equal "
            "(e.g. ['epa_hub', 'desnz_ghg_conversion'])."
        ),
    )
    include_preview: bool = Field(
        default=False,
        description="Allow preview-status factors when no certified match exists.",
    )
    extras: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form extra context (e.g. fuel_type, activity_unit hints).",
    )

    # ----- Convenience accessors -----

    def resolved_date(self) -> date:
        if not self.reporting_date:
            return date.today()
        if isinstance(self.reporting_date, (date, datetime)):
            return self.reporting_date if isinstance(self.reporting_date, date) else self.reporting_date.date()
        return date.fromisoformat(str(self.reporting_date))

    @model_validator(mode="after")
    def _activity_must_not_be_blank(self) -> "ResolutionRequest":
        if not str(self.activity).strip():
            raise ValueError("activity must be a non-empty string")
        return self


__all__ = ["ResolutionRequest"]
