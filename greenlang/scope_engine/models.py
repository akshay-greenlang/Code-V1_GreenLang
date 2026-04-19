# -*- coding: utf-8 -*-
"""Scope Engine Pydantic models.

Reuses greenlang.schemas.base (GreenLangBase, GreenLangRecord, GreenLangRequest,
GreenLangResponse) and greenlang.data.emission_factor_record.Scope for scope enum.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from pydantic import Field

from greenlang.data.emission_factor_record import Scope
from greenlang.schemas.base import (
    GreenLangBase,
    GreenLangRecord,
    GreenLangRequest,
    GreenLangResponse,
)
from greenlang.schemas.enums import GeographicRegion, ReportingPeriod


class GHGGas(str, Enum):
    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    HFC = "HFC"
    PFC = "PFC"
    SF6 = "SF6"
    NF3 = "NF3"
    CO2E = "CO2e"


class GWPBasis(str, Enum):
    AR4_100YR = "AR4-100yr"
    AR5_100YR = "AR5-100yr"
    AR5_20YR = "AR5-20yr"
    AR6_100YR = "AR6-100yr"
    AR6_20YR = "AR6-20yr"


class ConsolidationApproach(str, Enum):
    EQUITY_SHARE = "equity_share"
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"


class Framework(str, Enum):
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    SBTI = "sbti"
    CSRD_E1 = "csrd_e1"
    CBAM = "cbam"


class ActivityRecord(GreenLangRecord):
    activity_id: str = Field(..., description="Stable activity identifier")
    activity_type: str = Field(
        ...,
        description="Canonical activity taxonomy key (e.g. stationary_combustion, "
        "mobile_combustion, purchased_electricity_location). Drives MRV routing.",
    )
    fuel_type: Optional[str] = Field(
        default=None,
        description="Fuel/commodity type for factor catalog lookup "
        "(e.g. diesel, natural_gas, electricity). Falls back to activity_type.",
    )
    scope_hint: Optional[Scope] = Field(default=None, description="Optional scope override")
    scope3_category: Optional[int] = Field(default=None, ge=1, le=15)
    quantity: Decimal
    unit: str
    region: Optional[GeographicRegion] = None
    region_code: Optional[str] = Field(
        default=None, description="ISO 3166 country code (e.g. 'US'); preferred over region"
    )
    year: int = Field(..., ge=1990, le=2100)
    period: Optional[ReportingPeriod] = None
    entity_id: Optional[str] = Field(default=None, description="Entity graph node id")
    methodology: Optional[str] = None
    factor_override_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmissionResult(GreenLangBase):
    activity_id: str
    scope: Scope
    gas: GHGGas
    gas_amount: Decimal
    gas_unit: str = "kg"
    gwp_basis: GWPBasis
    co2e_kg: Decimal
    factor_id: str
    factor_source: str
    factor_vintage: int
    formula_hash: str = Field(..., description="SHA-256 of (activity, factor, formula)")
    cached: bool = False


class ScopeBreakdown(GreenLangBase):
    scope_1_co2e_kg: Decimal = Decimal(0)
    scope_2_location_co2e_kg: Decimal = Decimal(0)
    scope_2_market_co2e_kg: Decimal = Decimal(0)
    scope_3_co2e_kg: Decimal = Decimal(0)
    scope_3_by_category: dict[int, Decimal] = Field(default_factory=dict)


class ScopeComputation(GreenLangResponse):
    computation_id: str
    entity_id: Optional[str]
    reporting_period_start: datetime
    reporting_period_end: datetime
    gwp_basis: GWPBasis
    consolidation: ConsolidationApproach
    breakdown: ScopeBreakdown
    results: list[EmissionResult]
    total_co2e_kg: Decimal
    computation_hash: str = Field(..., description="Aggregate SHA-256 over inputs+results")


class ComputationRequest(GreenLangRequest):
    activities: list[ActivityRecord]
    gwp_basis: GWPBasis = GWPBasis.AR6_100YR
    consolidation: ConsolidationApproach = ConsolidationApproach.OPERATIONAL_CONTROL
    entity_id: Optional[str] = None
    reporting_period_start: datetime
    reporting_period_end: datetime
    frameworks: list[Framework] = Field(default_factory=list)


class ComputationResponse(GreenLangResponse):
    computation: ScopeComputation
    framework_views: dict[Framework, "FrameworkView"] = Field(default_factory=dict)


class FrameworkView(GreenLangBase):
    framework: Framework
    rows: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)


ComputationResponse.model_rebuild()
