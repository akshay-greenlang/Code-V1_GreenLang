# -*- coding: utf-8 -*-
"""Comply request/response models."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import Field

from greenlang.schemas.base import (
    GreenLangBase,
    GreenLangRequest,
    GreenLangResponse,
)


class EntityProfile(GreenLangBase):
    """Minimal entity attributes the orchestrator feeds to Policy Graph."""

    entity_id: str
    legal_name: str
    hq_country: str = Field(
        ..., description="ISO 3166 country code (e.g. 'DE', 'US', 'IN')"
    )
    operates_in: list[str] = Field(
        default_factory=list,
        description="Jurisdictions where the entity conducts business (e.g. 'EU', 'US-CA')",
    )
    employees: Optional[int] = None
    turnover_m_eur: Optional[float] = None
    balance_sheet_m_eur: Optional[float] = None
    revenue_usd: Optional[float] = None
    type: str = Field(
        default="corporation",
        description="Entity form (corporation, partnership, sole_prop, …)",
    )


class ComplianceRunRequest(GreenLangRequest):
    """Single entrypoint for a Comply run."""

    entity: EntityProfile
    reporting_period_start: datetime
    reporting_period_end: datetime
    jurisdiction: str = Field(..., description="Primary jurisdiction under test")
    # Activities conform to scope_engine.models.ActivityRecord; typed as dict
    # here so Comply can be called from non-Python callers without importing
    # the Scope Engine schema directly.
    activities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="ActivityRecord-shaped dicts (passed through to scope_engine).",
    )
    # Optional overrides for the run infrastructure.
    case_id: Optional[str] = Field(
        default=None, description="Evidence Vault case identifier for this run"
    )
    vault_id: str = Field(default="comply-default")
    ledger_sqlite: Optional[str] = None
    vault_sqlite: Optional[str] = None
    force_frameworks: Optional[list[str]] = Field(
        default=None,
        description="If set, skip Policy Graph and use these frameworks verbatim",
    )


class FrameworkResult(GreenLangBase):
    """One per applicable regulation after Scope Engine projection."""

    regulation: str = Field(..., description="Policy-Graph regulation name (CBAM, CSRD, SB-253…)")
    framework_id: Optional[str] = Field(
        default=None,
        description="scope_engine Framework enum value (e.g. 'cbam', 'csrd_e1')",
    )
    jurisdiction: str
    deadline: Optional[str] = Field(default=None, description="ISO deadline if set")
    required_factor_classes: list[str] = Field(default_factory=list)
    total_co2e_kg: Optional[float] = Field(
        default=None, description="Total CO2e in kg (None if no Scope Engine match)"
    )
    scope_1_co2e_kg: Optional[float] = None
    scope_2_co2e_kg: Optional[float] = None
    scope_3_co2e_kg: Optional[float] = None
    computation_id: Optional[str] = None
    computation_hash: Optional[str] = None
    evidence_count: int = 0
    ledger_chain_hash: Optional[str] = None


class ComplianceRunResult(GreenLangResponse):
    entity_id: str
    reporting_period_start: datetime
    reporting_period_end: datetime
    jurisdiction: str
    applicable_regulations: list[str] = Field(default_factory=list)
    framework_results: list[FrameworkResult] = Field(default_factory=list)
    case_id: str
    evidence_bundle_path: Optional[str] = Field(
        default=None, description="Path to the signed ZIP bundle when bundling was requested"
    )
    ledger_global_chain_head: Optional[str] = None
    applicability_rationale: dict[str, str] = Field(default_factory=dict)
    evaluated_at: datetime
