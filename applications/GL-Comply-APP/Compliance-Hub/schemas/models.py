# -*- coding: utf-8 -*-
"""Unified compliance data model.

Reuses greenlang.schemas.base for GreenLangBase / GreenLangRecord / GreenLangRequest
and greenlang.schemas.enums for shared enums. Framework enum is specific to
Comply-Hub and mirrors the 10 app pack names.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import Field

from greenlang.schemas.base import (
    GreenLangBase,
    GreenLangRecord,
    GreenLangRequest,
    GreenLangResponse,
)
from greenlang.schemas.enums import (
    ComplianceStatus,
    ReportFormat,
    ReportingPeriod,
)


class FrameworkEnum(str, Enum):
    CSRD = "csrd"
    CBAM = "cbam"
    EUDR = "eudr"
    GHG_PROTOCOL = "ghg"
    ISO_14064 = "iso14064"
    SB253 = "sb253"
    SBTI = "sbti"
    EU_TAXONOMY = "taxonomy"
    TCFD = "tcfd"
    CDP = "cdp"


class EntitySnapshot(GreenLangBase):
    entity_id: str
    legal_name: str
    jurisdiction: str = Field(
        ..., description="ISO 3166 country code (e.g. 'DE', 'US', 'US-CA')"
    )
    revenue_eur: Optional[float] = None
    employees: Optional[int] = None
    sector_nace: Optional[str] = None
    sector_gics: Optional[str] = None
    imports_cbam_goods: bool = False
    handles_eudr_commodities: bool = False
    operates_in_us_ca: bool = False


class ApplicabilityRequest(GreenLangRequest):
    entity: EntitySnapshot
    reporting_year: int = Field(..., ge=2020, le=2100)


class ApplicabilityResult(GreenLangResponse):
    applicable_frameworks: list[FrameworkEnum]
    rationale: dict[FrameworkEnum, str]


class ComplianceRequest(GreenLangRequest):
    entity: EntitySnapshot
    reporting_period_start: datetime
    reporting_period_end: datetime
    period: ReportingPeriod = ReportingPeriod.ANNUAL
    frameworks: list[FrameworkEnum]
    data_sources: dict[str, Any] = Field(
        default_factory=dict,
        description="Pointers to input artifacts (S3 URIs, DB refs, etc.)",
    )


class FrameworkResult(GreenLangRecord):
    framework: FrameworkEnum
    compliance_status: ComplianceStatus
    report_uri: Optional[str] = None
    findings_summary: Optional[str] = None
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 of inputs+outputs for this framework run",
    )
    metrics: dict[str, Any] = Field(default_factory=dict)
    duration_ms: Optional[int] = None


class UnifiedComplianceReport(GreenLangResponse):
    job_id: str
    entity_id: str
    reporting_period_start: datetime
    reporting_period_end: datetime
    frameworks: list[FrameworkEnum]
    results: dict[FrameworkEnum, FrameworkResult]
    overall_status: ComplianceStatus
    gap_analysis: list[dict[str, Any]] = Field(default_factory=list)
    unified_report_uri: Optional[str] = None
    report_format: ReportFormat = ReportFormat.PDF
    aggregate_provenance_hash: str
