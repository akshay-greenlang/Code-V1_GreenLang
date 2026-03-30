# -*- coding: utf-8 -*-
"""
MultiEntityConsolidationEngine - PACK-027 Enterprise Net Zero Pack Engine 7
============================================================================

Consolidates GHG data across 100+ subsidiaries, joint ventures, and
associates using financial control, operational control, or equity share
approaches per GHG Protocol Corporate Standard Chapter 3.

Calculation Methodology:
    Financial Control:
        entity_emissions = 100% if parent has financial control, else 0%
        (ability to direct financial and operating policies)

    Operational Control:
        entity_emissions = 100% if parent has operational control, else 0%
        (authority to introduce and implement operating policies)

    Equity Share:
        entity_emissions = ownership_pct% of entity total emissions

    Intercompany Elimination:
        consolidated = sum(entity_emissions) - sum(intercompany_overlap)
        Overlap: Entity A Scope 3 Cat 1 = Entity B Scope 1

    Pro-Rata for M&A:
        acquisition:  entity_emissions * (days_in_reporting_year / 365)
        divestiture:  entity_emissions * (days_in_reporting_year / 365)

    Base Year Recalculation Triggers:
        Structural change (M&A) > 5% significance threshold
        Methodology change > 5% impact
        Discovery of significant error > 5%
        Change in organizational boundary
        Outsourcing/insourcing > 5% threshold

Regulatory References:
    - GHG Protocol Corporate Standard Chapter 3 (Organizational Boundaries)
    - GHG Protocol Corporate Standard Chapter 5 (Base Year)
    - IPCC AR6 WG1 (2021) - GWP-100 values
    - IFRS/IAS 27/28/31 - Financial reporting consolidation
    - SBTi Corporate Manual V5.3 - Boundary requirements

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Consolidation rules from GHG Protocol
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-027 Enterprise Net Zero Pack
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone, date
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {k: v for k, v in serializable.items()
                        if k not in ("calculated_at", "processing_time_ms", "provenance_hash")}
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConsolidationApproach(str, Enum):
    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"
    EQUITY_SHARE = "equity_share"

class EntityType(str, Enum):
    SUBSIDIARY = "subsidiary"
    JOINT_VENTURE = "joint_venture"
    ASSOCIATE = "associate"
    SPECIAL_PURPOSE_VEHICLE = "spv"
    PARENT = "parent"

class RecalculationTrigger(str, Enum):
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    METHODOLOGY_CHANGE = "methodology_change"
    ERROR_CORRECTION = "error_correction"
    BOUNDARY_CHANGE = "boundary_change"
    OUTSOURCING = "outsourcing"
    INSOURCING = "insourcing"

class CurrencyCode(str, Enum):
    USD = "usd"
    EUR = "eur"
    GBP = "gbp"
    JPY = "jpy"
    CNY = "cny"
    CHF = "chf"
    CAD = "cad"
    AUD = "aud"

# Significance threshold for base year recalculation.
SIGNIFICANCE_THRESHOLD_PCT: Decimal = Decimal("5.0")
DAYS_IN_YEAR: int = 365

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class EntityEmissions(BaseModel):
    """Emission data for a single entity."""
    entity_id: str = Field(..., min_length=1, max_length=100)
    entity_name: str = Field(..., min_length=1, max_length=300)
    entity_type: EntityType = Field(default=EntityType.SUBSIDIARY)
    parent_entity_id: str = Field(default="", max_length=100)
    ownership_pct: Decimal = Field(default=Decimal("100"), ge=Decimal("0"), le=Decimal("100"))
    has_financial_control: bool = Field(default=True)
    has_operational_control: bool = Field(default=True)
    country: str = Field(default="", max_length=2)
    reporting_currency: str = Field(default="usd", max_length=3)
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_location_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_market_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    acquisition_date: Optional[str] = Field(None, max_length=10)
    divestiture_date: Optional[str] = Field(None, max_length=10)
    revenue_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    employees: int = Field(default=0, ge=0)

class IntercompanyEntry(BaseModel):
    """Intercompany transaction for elimination."""
    selling_entity_id: str = Field(..., max_length=100)
    buying_entity_id: str = Field(..., max_length=100)
    transaction_type: str = Field(..., max_length=100)
    tco2e: Decimal = Field(..., ge=Decimal("0"))
    description: str = Field(default="", max_length=500)

class BaseYearData(BaseModel):
    """Base year emission data for recalculation assessment."""
    base_year: int = Field(default=2019, ge=2015, le=2025)
    base_year_total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    base_year_scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    base_year_scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    base_year_scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))

class RecalculationEvent(BaseModel):
    """Base year recalculation trigger event."""
    trigger: RecalculationTrigger = Field(...)
    event_description: str = Field(..., max_length=500)
    affected_entity_id: str = Field(default="", max_length=100)
    impact_tco2e: Decimal = Field(default=Decimal("0"))
    event_date: str = Field(default="", max_length=10)

class ConsolidationInput(BaseModel):
    """Complete input for multi-entity consolidation."""
    organization_name: str = Field(default="Enterprise", min_length=1, max_length=500)
    reporting_year: int = Field(default=2026, ge=2015, le=2100)
    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
    )
    entities: List[EntityEmissions] = Field(default_factory=list)
    intercompany_transactions: List[IntercompanyEntry] = Field(default_factory=list)
    base_year_data: Optional[BaseYearData] = Field(None)
    recalculation_events: List[RecalculationEvent] = Field(default_factory=list)
    reporting_currency: str = Field(default="usd", max_length=3)

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class EntityContribution(BaseModel):
    """Contribution of a single entity to consolidated total."""
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    entity_type: str = Field(default="")
    ownership_pct: Decimal = Field(default=Decimal("100"))
    inclusion_method: str = Field(default="full")
    consolidation_factor: Decimal = Field(default=Decimal("1.0"))
    pro_rata_factor: Decimal = Field(default=Decimal("1.0"))
    gross_scope1_tco2e: Decimal = Field(default=Decimal("0"))
    gross_scope2_location_tco2e: Decimal = Field(default=Decimal("0"))
    gross_scope2_market_tco2e: Decimal = Field(default=Decimal("0"))
    gross_scope3_tco2e: Decimal = Field(default=Decimal("0"))
    consolidated_scope1_tco2e: Decimal = Field(default=Decimal("0"))
    consolidated_scope2_location_tco2e: Decimal = Field(default=Decimal("0"))
    consolidated_scope2_market_tco2e: Decimal = Field(default=Decimal("0"))
    consolidated_scope3_tco2e: Decimal = Field(default=Decimal("0"))
    consolidated_total_tco2e: Decimal = Field(default=Decimal("0"))
    pct_of_group_total: Decimal = Field(default=Decimal("0"))

class EliminationEntry(BaseModel):
    """Detail of an intercompany elimination."""
    selling_entity: str = Field(default="")
    buying_entity: str = Field(default="")
    transaction_type: str = Field(default="")
    tco2e_eliminated: Decimal = Field(default=Decimal("0"))
    description: str = Field(default="")

class BaseYearRecalculation(BaseModel):
    """Base year recalculation assessment."""
    recalculation_required: bool = Field(default=False)
    total_trigger_impact_tco2e: Decimal = Field(default=Decimal("0"))
    significance_pct: Decimal = Field(default=Decimal("0"))
    exceeds_threshold: bool = Field(default=False)
    original_base_year_tco2e: Decimal = Field(default=Decimal("0"))
    restated_base_year_tco2e: Decimal = Field(default=Decimal("0"))
    trigger_events: List[Dict[str, Any]] = Field(default_factory=list)

class ConsolidationResult(BaseModel):
    """Complete consolidation result."""
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    organization_name: str = Field(default="")
    reporting_year: int = Field(default=0)
    consolidation_approach: str = Field(default="")

    consolidated_scope1_tco2e: Decimal = Field(default=Decimal("0"))
    consolidated_scope2_location_tco2e: Decimal = Field(default=Decimal("0"))
    consolidated_scope2_market_tco2e: Decimal = Field(default=Decimal("0"))
    consolidated_scope3_tco2e: Decimal = Field(default=Decimal("0"))
    consolidated_total_location_tco2e: Decimal = Field(default=Decimal("0"))
    consolidated_total_market_tco2e: Decimal = Field(default=Decimal("0"))

    sum_of_entity_totals_tco2e: Decimal = Field(default=Decimal("0"))
    total_eliminations_tco2e: Decimal = Field(default=Decimal("0"))

    entity_contributions: List[EntityContribution] = Field(default_factory=list)
    elimination_entries: List[EliminationEntry] = Field(default_factory=list)
    base_year_recalculation: BaseYearRecalculation = Field(default_factory=BaseYearRecalculation)

    entity_count: int = Field(default=0)
    entities_included: int = Field(default=0)
    entities_excluded: int = Field(default=0)

    regulatory_citations: List[str] = Field(default_factory=lambda: [
        "GHG Protocol Corporate Standard Chapter 3 (Organizational Boundaries)",
        "GHG Protocol Corporate Standard Chapter 5 (Base Year)",
        "SBTi Corporate Manual V5.3 - Boundary Requirements",
    ])
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MultiEntityConsolidationEngine:
    """Multi-entity GHG consolidation engine.

    Consolidates 100+ entities using financial control, operational control,
    or equity share with intercompany elimination and base year recalculation.

    Usage::

        engine = MultiEntityConsolidationEngine()
        result = engine.calculate(consolidation_input)
        assert result.provenance_hash
    """

    engine_version: str = _MODULE_VERSION

    def calculate(self, data: ConsolidationInput) -> ConsolidationResult:
        """Run multi-entity consolidation."""
        t0 = time.perf_counter()
        logger.info(
            "Consolidation: org=%s, year=%d, approach=%s, entities=%d",
            data.organization_name, data.reporting_year,
            data.consolidation_approach.value, len(data.entities),
        )

        # Step 1: Determine inclusion and consolidation factors
        contributions: List[EntityContribution] = []
        included_count = 0
        excluded_count = 0

        for entity in data.entities:
            factor, method = self._get_consolidation_factor(
                entity, data.consolidation_approach,
            )
            pro_rata = self._get_pro_rata_factor(entity, data.reporting_year)

            if factor > Decimal("0"):
                included_count += 1
            else:
                excluded_count += 1

            effective = _round_val(factor * pro_rata, 4)

            cons_s1 = _round_val(entity.scope1_tco2e * effective)
            cons_s2_loc = _round_val(entity.scope2_location_tco2e * effective)
            cons_s2_mkt = _round_val(entity.scope2_market_tco2e * effective)
            cons_s3 = _round_val(entity.scope3_tco2e * effective)
            cons_total = _round_val(cons_s1 + cons_s2_loc + cons_s3)

            contributions.append(EntityContribution(
                entity_id=entity.entity_id,
                entity_name=entity.entity_name,
                entity_type=entity.entity_type.value,
                ownership_pct=entity.ownership_pct,
                inclusion_method=method,
                consolidation_factor=factor,
                pro_rata_factor=pro_rata,
                gross_scope1_tco2e=entity.scope1_tco2e,
                gross_scope2_location_tco2e=entity.scope2_location_tco2e,
                gross_scope2_market_tco2e=entity.scope2_market_tco2e,
                gross_scope3_tco2e=entity.scope3_tco2e,
                consolidated_scope1_tco2e=cons_s1,
                consolidated_scope2_location_tco2e=cons_s2_loc,
                consolidated_scope2_market_tco2e=cons_s2_mkt,
                consolidated_scope3_tco2e=cons_s3,
                consolidated_total_tco2e=cons_total,
            ))

        # Step 2: Sum consolidated totals
        total_s1 = sum(c.consolidated_scope1_tco2e for c in contributions)
        total_s2_loc = sum(c.consolidated_scope2_location_tco2e for c in contributions)
        total_s2_mkt = sum(c.consolidated_scope2_market_tco2e for c in contributions)
        total_s3 = sum(c.consolidated_scope3_tco2e for c in contributions)
        entity_sum = _round_val(total_s1 + total_s2_loc + total_s3)

        # Step 3: Intercompany eliminations
        elim_entries: List[EliminationEntry] = []
        total_elim = Decimal("0")
        entity_ids = {e.entity_id for e in data.entities}

        for txn in data.intercompany_transactions:
            if txn.selling_entity_id in entity_ids and txn.buying_entity_id in entity_ids:
                elim_entries.append(EliminationEntry(
                    selling_entity=txn.selling_entity_id,
                    buying_entity=txn.buying_entity_id,
                    transaction_type=txn.transaction_type,
                    tco2e_eliminated=txn.tco2e,
                    description=txn.description,
                ))
                total_elim += txn.tco2e

        total_elim = _round_val(total_elim)

        # Step 4: Final consolidated totals
        final_s1 = _round_val(total_s1)
        final_s2_loc = _round_val(total_s2_loc)
        final_s2_mkt = _round_val(total_s2_mkt)
        final_s3 = _round_val(total_s3 - total_elim)
        final_total_loc = _round_val(final_s1 + final_s2_loc + final_s3)
        final_total_mkt = _round_val(final_s1 + final_s2_mkt + final_s3)

        # Update entity contribution percentages
        for c in contributions:
            c.pct_of_group_total = _round_val(
                _safe_pct(c.consolidated_total_tco2e, final_total_loc), 2
            )

        # Step 5: Base year recalculation assessment
        by_recalc = self._assess_base_year_recalculation(data)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ConsolidationResult(
            organization_name=data.organization_name,
            reporting_year=data.reporting_year,
            consolidation_approach=data.consolidation_approach.value,
            consolidated_scope1_tco2e=final_s1,
            consolidated_scope2_location_tco2e=final_s2_loc,
            consolidated_scope2_market_tco2e=final_s2_mkt,
            consolidated_scope3_tco2e=final_s3,
            consolidated_total_location_tco2e=final_total_loc,
            consolidated_total_market_tco2e=final_total_mkt,
            sum_of_entity_totals_tco2e=entity_sum,
            total_eliminations_tco2e=total_elim,
            entity_contributions=contributions,
            elimination_entries=elim_entries,
            base_year_recalculation=by_recalc,
            entity_count=len(data.entities),
            entities_included=included_count,
            entities_excluded=excluded_count,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Consolidation complete: total_loc=%.2f, total_mkt=%.2f, "
            "entities=%d (incl=%d, excl=%d), eliminations=%.2f, hash=%s",
            float(final_total_loc), float(final_total_mkt),
            len(data.entities), included_count, excluded_count,
            float(total_elim), result.provenance_hash[:16],
        )
        return result

    async def calculate_async(self, data: ConsolidationInput) -> ConsolidationResult:
        """Async wrapper for calculate()."""
        return self.calculate(data)

    def _get_consolidation_factor(
        self, entity: EntityEmissions, approach: ConsolidationApproach,
    ) -> tuple[Decimal, str]:
        """Determine consolidation factor and method for an entity."""
        if approach == ConsolidationApproach.FINANCIAL_CONTROL:
            if entity.has_financial_control:
                return Decimal("1.0"), "full_consolidation"
            return Decimal("0"), "excluded_no_financial_control"
        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
            if entity.has_operational_control:
                return Decimal("1.0"), "full_consolidation"
            return Decimal("0"), "excluded_no_operational_control"
        else:  # Equity share
            return entity.ownership_pct / Decimal("100"), "equity_share"

    def _get_pro_rata_factor(
        self, entity: EntityEmissions, reporting_year: int,
    ) -> Decimal:
        """Calculate pro-rata factor for mid-year acquisitions/divestitures."""
        start_day = 1
        end_day = DAYS_IN_YEAR

        if entity.acquisition_date:
            try:
                acq = datetime.strptime(entity.acquisition_date, "%Y-%m-%d").date()
                if acq.year == reporting_year:
                    start_day = acq.timetuple().tm_yday
            except (ValueError, AttributeError):
                pass

        if entity.divestiture_date:
            try:
                div = datetime.strptime(entity.divestiture_date, "%Y-%m-%d").date()
                if div.year == reporting_year:
                    end_day = div.timetuple().tm_yday
            except (ValueError, AttributeError):
                pass

        days_included = max(0, end_day - start_day + 1)
        return _round_val(_decimal(days_included) / _decimal(DAYS_IN_YEAR), 4)

    def _assess_base_year_recalculation(
        self, data: ConsolidationInput,
    ) -> BaseYearRecalculation:
        """Assess whether base year recalculation is required."""
        if not data.base_year_data or not data.recalculation_events:
            return BaseYearRecalculation()

        total_impact = sum(abs(e.impact_tco2e) for e in data.recalculation_events)
        significance = _safe_pct(total_impact, data.base_year_data.base_year_total_tco2e)
        exceeds = significance >= SIGNIFICANCE_THRESHOLD_PCT
        required = exceeds

        restated = data.base_year_data.base_year_total_tco2e
        for event in data.recalculation_events:
            if event.trigger in (RecalculationTrigger.ACQUISITION, RecalculationTrigger.INSOURCING):
                restated += abs(event.impact_tco2e)
            elif event.trigger in (RecalculationTrigger.DIVESTITURE, RecalculationTrigger.OUTSOURCING):
                restated -= abs(event.impact_tco2e)
            else:
                restated += event.impact_tco2e

        triggers: List[Dict[str, Any]] = []
        for e in data.recalculation_events:
            triggers.append({
                "trigger": e.trigger.value,
                "description": e.event_description,
                "entity": e.affected_entity_id,
                "impact_tco2e": str(e.impact_tco2e),
                "date": e.event_date,
            })

        return BaseYearRecalculation(
            recalculation_required=required,
            total_trigger_impact_tco2e=_round_val(total_impact),
            significance_pct=_round_val(significance, 2),
            exceeds_threshold=exceeds,
            original_base_year_tco2e=data.base_year_data.base_year_total_tco2e,
            restated_base_year_tco2e=_round_val(restated),
            trigger_events=triggers,
        )
