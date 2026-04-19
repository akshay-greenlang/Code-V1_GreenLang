# -*- coding: utf-8 -*-
"""
InitiativeTrackerBridge - Initiative Deployment Tracking for PACK-029
=======================================================================

Enterprise bridge for importing initiative deployment status, tracking
effectiveness (actual tCO2e reduction achieved), linking initiatives to
variance attribution, forecasting future initiative impact, budget
tracking (CapEx/OpEx spent vs planned), and risk status (RAG per initiative).

Integration Points:
    - Initiative Registry: Import all registered reduction initiatives
    - Deployment Status: planned, in-progress, completed, deferred
    - Effectiveness Tracking: Actual vs projected tCO2e reduction
    - Variance Attribution: Link initiatives to emission variance drivers
    - Forecast Impact: Project future initiative abatement potential
    - Budget Tracking: CapEx/OpEx spent vs planned per initiative
    - Risk Status: Red/Amber/Green per initiative

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-029 Interim Targets Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InitiativeStatus(str, Enum):
    PLANNED = "planned"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"

class InitiativeCategory(str, Enum):
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_ENERGY = "renewable_energy"
    ELECTRIFICATION = "electrification"
    FUEL_SWITCHING = "fuel_switching"
    PROCESS_CHANGE = "process_change"
    SUPPLY_CHAIN = "supply_chain"
    BEHAVIORAL = "behavioral"
    TECHNOLOGY = "technology"
    CCS_CCUS = "ccs_ccus"
    NATURE_BASED = "nature_based"

class RAGStatus(str, Enum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"

class BudgetType(str, Enum):
    CAPEX = "capex"
    OPEX = "opex"
    MIXED = "mixed"

class VarianceType(str, Enum):
    FAVORABLE = "favorable"     # Lower emissions than expected
    UNFAVORABLE = "unfavorable"  # Higher emissions than expected
    NEUTRAL = "neutral"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class InitiativeTrackerConfig(BaseModel):
    """Configuration for the initiative tracker bridge."""
    pack_id: str = Field(default="PACK-029")
    organization_id: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    enable_provenance: bool = Field(default=True)
    db_connection_string: str = Field(default="")
    refresh_interval_minutes: int = Field(default=60, ge=5, le=1440)
    rag_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "green_min_delivery_pct": 80.0,
        "amber_min_delivery_pct": 50.0,
        "green_max_budget_overrun_pct": 10.0,
        "amber_max_budget_overrun_pct": 25.0,
    })

class Initiative(BaseModel):
    """Single emission reduction initiative."""
    initiative_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="")
    description: str = Field(default="")
    category: InitiativeCategory = Field(default=InitiativeCategory.ENERGY_EFFICIENCY)
    status: InitiativeStatus = Field(default=InitiativeStatus.PLANNED)
    scope_coverage: str = Field(default="Scope 1+2")
    scope3_categories: List[int] = Field(default_factory=list)
    start_date: str = Field(default="")
    end_date: str = Field(default="")
    target_year: int = Field(default=2030)
    projected_reduction_tco2e: float = Field(default=0.0)
    actual_reduction_tco2e: float = Field(default=0.0)
    delivery_pct: float = Field(default=0.0)
    capex_planned_usd: float = Field(default=0.0)
    capex_spent_usd: float = Field(default=0.0)
    opex_annual_usd: float = Field(default=0.0)
    opex_spent_usd: float = Field(default=0.0)
    budget_type: BudgetType = Field(default=BudgetType.MIXED)
    rag_status: RAGStatus = Field(default=RAGStatus.GREEN)
    risk_description: str = Field(default="")
    owner: str = Field(default="")
    department: str = Field(default="")
    linked_target_year: int = Field(default=2030)
    linked_target_ref: str = Field(default="")
    provenance_hash: str = Field(default="")

class InitiativePortfolio(BaseModel):
    """Portfolio of all initiatives."""
    portfolio_id: str = Field(default_factory=_new_uuid)
    organization_id: str = Field(default="")
    initiatives: List[Initiative] = Field(default_factory=list)
    total_count: int = Field(default=0)
    by_status: Dict[str, int] = Field(default_factory=dict)
    by_category: Dict[str, int] = Field(default_factory=dict)
    by_rag: Dict[str, int] = Field(default_factory=dict)
    total_projected_tco2e: float = Field(default=0.0)
    total_actual_tco2e: float = Field(default=0.0)
    total_capex_planned_usd: float = Field(default=0.0)
    total_capex_spent_usd: float = Field(default=0.0)
    overall_delivery_pct: float = Field(default=0.0)
    overall_rag: RAGStatus = Field(default=RAGStatus.GREEN)
    provenance_hash: str = Field(default="")

class VarianceAttribution(BaseModel):
    """Link between initiative and emission variance."""
    attribution_id: str = Field(default_factory=_new_uuid)
    initiative_id: str = Field(default="")
    initiative_name: str = Field(default="")
    variance_tco2e: float = Field(default=0.0)
    variance_type: VarianceType = Field(default=VarianceType.NEUTRAL)
    variance_share_pct: float = Field(default=0.0)
    explanation: str = Field(default="")
    scope: str = Field(default="")
    reporting_period: str = Field(default="")
    provenance_hash: str = Field(default="")

class ForecastResult(BaseModel):
    """Forecast of future initiative impact."""
    forecast_id: str = Field(default_factory=_new_uuid)
    forecast_year: int = Field(default=2030)
    total_projected_reduction_tco2e: float = Field(default=0.0)
    by_initiative: List[Dict[str, Any]] = Field(default_factory=list)
    by_category: Dict[str, float] = Field(default_factory=dict)
    cumulative_to_year_tco2e: float = Field(default=0.0)
    confidence_level: str = Field(default="medium")
    provenance_hash: str = Field(default="")

class InitiativeTrackerResult(BaseModel):
    """Complete initiative tracker result."""
    result_id: str = Field(default_factory=_new_uuid)
    portfolio: Optional[InitiativePortfolio] = Field(None)
    variance_attributions: List[VarianceAttribution] = Field(default_factory=list)
    forecast: Optional[ForecastResult] = Field(None)
    budget_summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# InitiativeTrackerBridge
# ---------------------------------------------------------------------------

class InitiativeTrackerBridge:
    """Initiative deployment tracking bridge for PACK-029.

    Tracks emission reduction initiatives, monitors deployment
    progress, attributes variance, forecasts future impact,
    and provides RAG status for each initiative.

    Example:
        >>> bridge = InitiativeTrackerBridge(InitiativeTrackerConfig())
        >>> portfolio = await bridge.import_initiatives(initiative_data)
        >>> variance = await bridge.attribute_variance(current_vs_target)
        >>> forecast = await bridge.forecast_impact(2030)
    """

    def __init__(self, config: Optional[InitiativeTrackerConfig] = None) -> None:
        self.config = config or InitiativeTrackerConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._portfolio_cache: Optional[InitiativePortfolio] = None

        self.logger.info(
            "InitiativeTrackerBridge (PACK-029) initialized: year=%d",
            self.config.reporting_year,
        )

    async def import_initiatives(
        self,
        initiative_data: Optional[List[Dict[str, Any]]] = None,
    ) -> InitiativePortfolio:
        """Import initiative registry and deployment status."""
        data = initiative_data or self._generate_default_initiatives()

        initiatives: List[Initiative] = []
        for item in data:
            projected = item.get("projected_reduction_tco2e", 0.0)
            actual = item.get("actual_reduction_tco2e", 0.0)
            delivery = (actual / max(projected, 1.0)) * 100.0 if projected > 0 else 0.0

            rag = self._calculate_rag(
                delivery,
                item.get("capex_spent_usd", 0.0),
                item.get("capex_planned_usd", 0.0),
            )

            init = Initiative(
                initiative_id=item.get("initiative_id", _new_uuid()),
                name=item.get("name", ""),
                description=item.get("description", ""),
                category=InitiativeCategory(item.get("category", "energy_efficiency")),
                status=InitiativeStatus(item.get("status", "planned")),
                scope_coverage=item.get("scope_coverage", "Scope 1+2"),
                scope3_categories=item.get("scope3_categories", []),
                start_date=item.get("start_date", ""),
                end_date=item.get("end_date", ""),
                target_year=item.get("target_year", 2030),
                projected_reduction_tco2e=projected,
                actual_reduction_tco2e=actual,
                delivery_pct=round(delivery, 2),
                capex_planned_usd=item.get("capex_planned_usd", 0.0),
                capex_spent_usd=item.get("capex_spent_usd", 0.0),
                opex_annual_usd=item.get("opex_annual_usd", 0.0),
                opex_spent_usd=item.get("opex_spent_usd", 0.0),
                budget_type=BudgetType(item.get("budget_type", "mixed")),
                rag_status=rag,
                risk_description=item.get("risk_description", ""),
                owner=item.get("owner", ""),
                department=item.get("department", ""),
                linked_target_year=item.get("linked_target_year", 2030),
                linked_target_ref=item.get("linked_target_ref", ""),
            )
            if self.config.enable_provenance:
                init.provenance_hash = _compute_hash(init)
            initiatives.append(init)

        # Build portfolio summary
        by_status: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        by_rag: Dict[str, int] = {}
        total_proj = total_actual = total_capex_plan = total_capex_spent = 0.0

        for init in initiatives:
            by_status[init.status.value] = by_status.get(init.status.value, 0) + 1
            by_category[init.category.value] = by_category.get(init.category.value, 0) + 1
            by_rag[init.rag_status.value] = by_rag.get(init.rag_status.value, 0) + 1
            total_proj += init.projected_reduction_tco2e
            total_actual += init.actual_reduction_tco2e
            total_capex_plan += init.capex_planned_usd
            total_capex_spent += init.capex_spent_usd

        overall_delivery = (total_actual / max(total_proj, 1.0)) * 100.0
        overall_rag = self._calculate_rag(overall_delivery, total_capex_spent, total_capex_plan)

        portfolio = InitiativePortfolio(
            organization_id=self.config.organization_id,
            initiatives=initiatives,
            total_count=len(initiatives),
            by_status=by_status,
            by_category=by_category,
            by_rag=by_rag,
            total_projected_tco2e=round(total_proj, 2),
            total_actual_tco2e=round(total_actual, 2),
            total_capex_planned_usd=round(total_capex_plan, 2),
            total_capex_spent_usd=round(total_capex_spent, 2),
            overall_delivery_pct=round(overall_delivery, 2),
            overall_rag=overall_rag,
        )

        if self.config.enable_provenance:
            portfolio.provenance_hash = _compute_hash(portfolio)

        self._portfolio_cache = portfolio
        self.logger.info(
            "Initiatives imported: %d total, delivery=%.1f%%, RAG=%s, "
            "projected=%.0f tCO2e, actual=%.0f tCO2e",
            len(initiatives), overall_delivery, overall_rag.value,
            total_proj, total_actual,
        )
        return portfolio

    async def attribute_variance(
        self,
        total_variance_tco2e: float,
        scope: str = "Scope 1+2",
        reporting_period: str = "",
    ) -> List[VarianceAttribution]:
        """Attribute emission variance to specific initiatives."""
        portfolio = self._portfolio_cache
        if not portfolio:
            return []

        active = [i for i in portfolio.initiatives if i.status in (InitiativeStatus.IN_PROGRESS, InitiativeStatus.COMPLETED)]
        if not active:
            return []

        attributions: List[VarianceAttribution] = []
        total_actual = sum(i.actual_reduction_tco2e for i in active)

        for init in active:
            share = (init.actual_reduction_tco2e / max(total_actual, 1.0)) * 100.0
            attributed_variance = total_variance_tco2e * (share / 100.0)
            v_type = VarianceType.FAVORABLE if attributed_variance < 0 else (
                VarianceType.UNFAVORABLE if attributed_variance > 0 else VarianceType.NEUTRAL
            )

            attr = VarianceAttribution(
                initiative_id=init.initiative_id,
                initiative_name=init.name,
                variance_tco2e=round(attributed_variance, 2),
                variance_type=v_type,
                variance_share_pct=round(share, 2),
                explanation=f"{init.name}: {init.actual_reduction_tco2e:.0f} tCO2e reduced ({init.delivery_pct:.0f}% delivery)",
                scope=scope,
                reporting_period=reporting_period or f"FY{self.config.reporting_year}",
            )
            if self.config.enable_provenance:
                attr.provenance_hash = _compute_hash(attr)
            attributions.append(attr)

        self.logger.info(
            "Variance attributed: %.0f tCO2e across %d initiatives",
            total_variance_tco2e, len(attributions),
        )
        return attributions

    async def forecast_impact(
        self, forecast_year: int,
    ) -> ForecastResult:
        """Forecast future initiative impact for a target year."""
        portfolio = self._portfolio_cache
        if not portfolio:
            return ForecastResult(forecast_year=forecast_year)

        by_initiative: List[Dict[str, Any]] = []
        by_category: Dict[str, float] = {}
        total_projected = 0.0

        for init in portfolio.initiatives:
            if init.status == InitiativeStatus.CANCELLED:
                continue

            # Scale projection based on delivery trend
            scale_factor = min(init.delivery_pct / 100.0, 1.2) if init.delivery_pct > 0 else 0.8
            years_remaining = max(forecast_year - self.config.reporting_year, 1)
            years_to_target = max(init.target_year - self.config.reporting_year, 1)
            time_factor = min(years_remaining / years_to_target, 1.0)

            forecast_reduction = init.projected_reduction_tco2e * scale_factor * time_factor

            by_initiative.append({
                "initiative_id": init.initiative_id,
                "name": init.name,
                "category": init.category.value,
                "projected_reduction_tco2e": round(forecast_reduction, 2),
                "confidence": "high" if init.delivery_pct >= 80 else ("medium" if init.delivery_pct >= 50 else "low"),
                "status": init.status.value,
            })

            cat = init.category.value
            by_category[cat] = by_category.get(cat, 0.0) + forecast_reduction
            total_projected += forecast_reduction

        # Cumulative from base year
        cumulative = total_projected * max(forecast_year - self.config.reporting_year, 1) * 0.5

        confidence = "high" if portfolio.overall_delivery_pct >= 80 else (
            "medium" if portfolio.overall_delivery_pct >= 50 else "low"
        )

        forecast = ForecastResult(
            forecast_year=forecast_year,
            total_projected_reduction_tco2e=round(total_projected, 2),
            by_initiative=by_initiative,
            by_category={k: round(v, 2) for k, v in by_category.items()},
            cumulative_to_year_tco2e=round(cumulative, 2),
            confidence_level=confidence,
        )

        if self.config.enable_provenance:
            forecast.provenance_hash = _compute_hash(forecast)

        self.logger.info(
            "Forecast %d: %.0f tCO2e total, %d initiatives, confidence=%s",
            forecast_year, total_projected, len(by_initiative), confidence,
        )
        return forecast

    async def get_budget_summary(self) -> Dict[str, Any]:
        """Get budget summary across all initiatives."""
        portfolio = self._portfolio_cache
        if not portfolio:
            return {"available": False}

        total_capex_plan = portfolio.total_capex_planned_usd
        total_capex_spent = portfolio.total_capex_spent_usd
        total_opex_plan = sum(i.opex_annual_usd for i in portfolio.initiatives)
        total_opex_spent = sum(i.opex_spent_usd for i in portfolio.initiatives)

        capex_utilization = (total_capex_spent / max(total_capex_plan, 1.0)) * 100.0
        opex_utilization = (total_opex_spent / max(total_opex_plan, 1.0)) * 100.0

        return {
            "capex_planned_usd": round(total_capex_plan, 2),
            "capex_spent_usd": round(total_capex_spent, 2),
            "capex_utilization_pct": round(capex_utilization, 2),
            "opex_planned_usd": round(total_opex_plan, 2),
            "opex_spent_usd": round(total_opex_spent, 2),
            "opex_utilization_pct": round(opex_utilization, 2),
            "total_planned_usd": round(total_capex_plan + total_opex_plan, 2),
            "total_spent_usd": round(total_capex_spent + total_opex_spent, 2),
            "cost_per_tco2e_usd": round(
                (total_capex_spent + total_opex_spent) / max(portfolio.total_actual_tco2e, 1.0), 2
            ),
        }

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "pack_id": self.config.pack_id,
            "reporting_year": self.config.reporting_year,
            "portfolio_loaded": self._portfolio_cache is not None,
            "initiative_count": self._portfolio_cache.total_count if self._portfolio_cache else 0,
            "overall_rag": self._portfolio_cache.overall_rag.value if self._portfolio_cache else "N/A",
        }

    def _calculate_rag(
        self, delivery_pct: float, spent: float, planned: float,
    ) -> RAGStatus:
        """Calculate RAG status based on delivery and budget."""
        thresholds = self.config.rag_thresholds
        overrun = ((spent - planned) / max(planned, 1.0)) * 100.0 if planned > 0 else 0.0

        if delivery_pct >= thresholds["green_min_delivery_pct"] and overrun <= thresholds["green_max_budget_overrun_pct"]:
            return RAGStatus.GREEN
        elif delivery_pct >= thresholds["amber_min_delivery_pct"] and overrun <= thresholds["amber_max_budget_overrun_pct"]:
            return RAGStatus.AMBER
        else:
            return RAGStatus.RED

    def _generate_default_initiatives(self) -> List[Dict[str, Any]]:
        """Generate default initiative portfolio."""
        return [
            {"name": "LED lighting retrofit", "category": "energy_efficiency", "status": "completed", "projected_reduction_tco2e": 500, "actual_reduction_tco2e": 520, "capex_planned_usd": 200000, "capex_spent_usd": 195000, "opex_annual_usd": 5000, "opex_spent_usd": 4800},
            {"name": "Renewable electricity PPA", "category": "renewable_energy", "status": "in_progress", "projected_reduction_tco2e": 8000, "actual_reduction_tco2e": 6500, "capex_planned_usd": 50000, "capex_spent_usd": 48000, "opex_annual_usd": 120000, "opex_spent_usd": 95000},
            {"name": "Fleet electrification Phase 1", "category": "electrification", "status": "in_progress", "projected_reduction_tco2e": 3000, "actual_reduction_tco2e": 1800, "capex_planned_usd": 2500000, "capex_spent_usd": 1800000, "opex_annual_usd": 30000, "opex_spent_usd": 25000},
            {"name": "HVAC system upgrade", "category": "energy_efficiency", "status": "in_progress", "projected_reduction_tco2e": 1200, "actual_reduction_tco2e": 900, "capex_planned_usd": 800000, "capex_spent_usd": 750000, "opex_annual_usd": 15000, "opex_spent_usd": 12000},
            {"name": "Supplier engagement program", "category": "supply_chain", "status": "planned", "projected_reduction_tco2e": 5000, "actual_reduction_tco2e": 0, "capex_planned_usd": 100000, "capex_spent_usd": 20000, "scope_coverage": "Scope 3", "scope3_categories": [1, 4]},
            {"name": "On-site solar PV", "category": "renewable_energy", "status": "planned", "projected_reduction_tco2e": 2500, "actual_reduction_tco2e": 0, "capex_planned_usd": 1500000, "capex_spent_usd": 0},
            {"name": "Heat pump installation", "category": "electrification", "status": "planned", "projected_reduction_tco2e": 2000, "actual_reduction_tco2e": 0, "capex_planned_usd": 600000, "capex_spent_usd": 0},
            {"name": "Travel policy reduction", "category": "behavioral", "status": "completed", "projected_reduction_tco2e": 400, "actual_reduction_tco2e": 350, "capex_planned_usd": 10000, "capex_spent_usd": 8000, "scope_coverage": "Scope 3", "scope3_categories": [6]},
        ]
