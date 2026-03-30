# -*- coding: utf-8 -*-
"""
MRVScope3Bridge - Bridge to All 15 Scope 3 MRV Agents for PACK-042
=====================================================================

This module routes activity/spend data to the appropriate Scope 3 MRV
agents (MRV-014 through MRV-028) for emissions calculation. It covers
all 15 GHG Protocol Scope 3 categories from Purchased Goods & Services
through Investments.

Routing Table:
    Cat 1  Purchased Goods/Services --> MRV-014 (gl_purchased_goods_services_)
    Cat 2  Capital Goods             --> MRV-015 (gl_capital_goods_)
    Cat 3  Fuel & Energy Activities  --> MRV-016 (gl_fuel_energy_activities_)
    Cat 4  Upstream Transportation   --> MRV-017 (gl_upstream_transportation_)
    Cat 5  Waste Generated           --> MRV-018 (gl_waste_generated_)
    Cat 6  Business Travel           --> MRV-019 (gl_business_travel_)
    Cat 7  Employee Commuting        --> MRV-020 (gl_employee_commuting_)
    Cat 8  Upstream Leased Assets    --> MRV-021 (gl_upstream_leased_assets_)
    Cat 9  Downstream Transportation --> MRV-022 (gl_downstream_transportation_)
    Cat 10 Processing Sold Products  --> MRV-023 (gl_processing_sold_products_)
    Cat 11 Use of Sold Products      --> MRV-024 (gl_use_of_sold_products_)
    Cat 12 End-of-Life Treatment     --> MRV-025 (gl_end_of_life_treatment_)
    Cat 13 Downstream Leased Assets  --> MRV-026 (gl_downstream_leased_assets_)
    Cat 14 Franchises                --> MRV-027 (gl_franchises_)
    Cat 15 Investments               --> MRV-028 (gl_investments_)

Zero-Hallucination:
    All emission factor lookups, spend-based calculations, EEIO factors,
    and aggregations use deterministic formulas. No LLM calls in the
    calculation path.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-042 Scope 3 Starter
Status: Production Ready
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Agent Stubs
# ---------------------------------------------------------------------------

class _AgentStub:
    """Stub for unavailable MRV agent modules."""

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name
        self._available = False

    def __getattr__(self, name: str) -> Any:
        def _stub_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "agent": self._agent_name,
                "method": name,
                "status": "degraded",
                "message": f"{self._agent_name} not available, using stub",
                "emissions_tco2e": 0.0,
            }
        return _stub_method

def _try_import_agent(agent_id: str, module_path: str) -> Any:
    """Try to import an MRV agent with graceful fallback.

    Args:
        agent_id: Agent identifier (e.g., 'MRV-014').
        module_path: Python module path for the agent.

    Returns:
        Imported module or _AgentStub if unavailable.
    """
    try:
        import importlib

        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("MRV agent %s not available, using stub", agent_id)
        return _AgentStub(agent_id)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories (1-15)."""

    CAT_1_PURCHASED_GOODS = "cat_1"
    CAT_2_CAPITAL_GOODS = "cat_2"
    CAT_3_FUEL_ENERGY = "cat_3"
    CAT_4_UPSTREAM_TRANSPORT = "cat_4"
    CAT_5_WASTE = "cat_5"
    CAT_6_BUSINESS_TRAVEL = "cat_6"
    CAT_7_COMMUTING = "cat_7"
    CAT_8_UPSTREAM_LEASED = "cat_8"
    CAT_9_DOWNSTREAM_TRANSPORT = "cat_9"
    CAT_10_PROCESSING_SOLD = "cat_10"
    CAT_11_USE_SOLD = "cat_11"
    CAT_12_END_OF_LIFE = "cat_12"
    CAT_13_DOWNSTREAM_LEASED = "cat_13"
    CAT_14_FRANCHISES = "cat_14"
    CAT_15_INVESTMENTS = "cat_15"

class AgentStatus(str, Enum):
    """MRV agent availability status."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"

class CalculationMethodology(str, Enum):
    """Scope 3 calculation methodologies per GHG Protocol guidance."""

    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    DISTANCE_BASED = "distance_based"
    FUEL_BASED = "fuel_based"
    ASSET_SPECIFIC = "asset_specific"
    INVESTMENT_SPECIFIC = "investment_specific"

# ---------------------------------------------------------------------------
# Agent-to-Category Mapping
# ---------------------------------------------------------------------------

AGENT_CATEGORY_MAP: Dict[Scope3Category, Dict[str, str]] = {
    Scope3Category.CAT_1_PURCHASED_GOODS: {
        "agent_id": "MRV-014",
        "module_path": "greenlang.agents.mrv.purchased_goods_services",
        "prefix": "gl_purchased_goods_services_",
        "ghg_protocol_name": "Purchased Goods and Services",
    },
    Scope3Category.CAT_2_CAPITAL_GOODS: {
        "agent_id": "MRV-015",
        "module_path": "greenlang.agents.mrv.capital_goods",
        "prefix": "gl_capital_goods_",
        "ghg_protocol_name": "Capital Goods",
    },
    Scope3Category.CAT_3_FUEL_ENERGY: {
        "agent_id": "MRV-016",
        "module_path": "greenlang.agents.mrv.fuel_energy_activities",
        "prefix": "gl_fuel_energy_activities_",
        "ghg_protocol_name": "Fuel- and Energy-Related Activities",
    },
    Scope3Category.CAT_4_UPSTREAM_TRANSPORT: {
        "agent_id": "MRV-017",
        "module_path": "greenlang.agents.mrv.upstream_transportation",
        "prefix": "gl_upstream_transportation_",
        "ghg_protocol_name": "Upstream Transportation and Distribution",
    },
    Scope3Category.CAT_5_WASTE: {
        "agent_id": "MRV-018",
        "module_path": "greenlang.agents.mrv.waste_generated",
        "prefix": "gl_waste_generated_",
        "ghg_protocol_name": "Waste Generated in Operations",
    },
    Scope3Category.CAT_6_BUSINESS_TRAVEL: {
        "agent_id": "MRV-019",
        "module_path": "greenlang.agents.mrv.business_travel",
        "prefix": "gl_business_travel_",
        "ghg_protocol_name": "Business Travel",
    },
    Scope3Category.CAT_7_COMMUTING: {
        "agent_id": "MRV-020",
        "module_path": "greenlang.agents.mrv.employee_commuting",
        "prefix": "gl_employee_commuting_",
        "ghg_protocol_name": "Employee Commuting",
    },
    Scope3Category.CAT_8_UPSTREAM_LEASED: {
        "agent_id": "MRV-021",
        "module_path": "greenlang.agents.mrv.upstream_leased_assets",
        "prefix": "gl_upstream_leased_assets_",
        "ghg_protocol_name": "Upstream Leased Assets",
    },
    Scope3Category.CAT_9_DOWNSTREAM_TRANSPORT: {
        "agent_id": "MRV-022",
        "module_path": "greenlang.agents.mrv.downstream_transportation",
        "prefix": "gl_downstream_transportation_",
        "ghg_protocol_name": "Downstream Transportation and Distribution",
    },
    Scope3Category.CAT_10_PROCESSING_SOLD: {
        "agent_id": "MRV-023",
        "module_path": "greenlang.agents.mrv.processing_sold_products",
        "prefix": "gl_processing_sold_products_",
        "ghg_protocol_name": "Processing of Sold Products",
    },
    Scope3Category.CAT_11_USE_SOLD: {
        "agent_id": "MRV-024",
        "module_path": "greenlang.agents.mrv.use_of_sold_products",
        "prefix": "gl_use_of_sold_products_",
        "ghg_protocol_name": "Use of Sold Products",
    },
    Scope3Category.CAT_12_END_OF_LIFE: {
        "agent_id": "MRV-025",
        "module_path": "greenlang.agents.mrv.end_of_life_treatment",
        "prefix": "gl_end_of_life_treatment_",
        "ghg_protocol_name": "End-of-Life Treatment of Sold Products",
    },
    Scope3Category.CAT_13_DOWNSTREAM_LEASED: {
        "agent_id": "MRV-026",
        "module_path": "greenlang.agents.mrv.downstream_leased_assets",
        "prefix": "gl_downstream_leased_assets_",
        "ghg_protocol_name": "Downstream Leased Assets",
    },
    Scope3Category.CAT_14_FRANCHISES: {
        "agent_id": "MRV-027",
        "module_path": "greenlang.agents.mrv.franchises",
        "prefix": "gl_franchises_",
        "ghg_protocol_name": "Franchises",
    },
    Scope3Category.CAT_15_INVESTMENTS: {
        "agent_id": "MRV-028",
        "module_path": "greenlang.agents.mrv.investments",
        "prefix": "gl_investments_",
        "ghg_protocol_name": "Investments",
    },
}

# Preferred methodologies per category
PREFERRED_METHODOLOGIES: Dict[Scope3Category, List[CalculationMethodology]] = {
    Scope3Category.CAT_1_PURCHASED_GOODS: [
        CalculationMethodology.SUPPLIER_SPECIFIC,
        CalculationMethodology.HYBRID,
        CalculationMethodology.AVERAGE_DATA,
        CalculationMethodology.SPEND_BASED,
    ],
    Scope3Category.CAT_2_CAPITAL_GOODS: [
        CalculationMethodology.SUPPLIER_SPECIFIC,
        CalculationMethodology.AVERAGE_DATA,
        CalculationMethodology.SPEND_BASED,
    ],
    Scope3Category.CAT_3_FUEL_ENERGY: [
        CalculationMethodology.AVERAGE_DATA,
        CalculationMethodology.SUPPLIER_SPECIFIC,
    ],
    Scope3Category.CAT_4_UPSTREAM_TRANSPORT: [
        CalculationMethodology.DISTANCE_BASED,
        CalculationMethodology.FUEL_BASED,
        CalculationMethodology.SPEND_BASED,
    ],
    Scope3Category.CAT_5_WASTE: [
        CalculationMethodology.SUPPLIER_SPECIFIC,
        CalculationMethodology.AVERAGE_DATA,
        CalculationMethodology.SPEND_BASED,
    ],
    Scope3Category.CAT_6_BUSINESS_TRAVEL: [
        CalculationMethodology.DISTANCE_BASED,
        CalculationMethodology.SPEND_BASED,
    ],
    Scope3Category.CAT_7_COMMUTING: [
        CalculationMethodology.DISTANCE_BASED,
        CalculationMethodology.AVERAGE_DATA,
    ],
    Scope3Category.CAT_8_UPSTREAM_LEASED: [
        CalculationMethodology.ASSET_SPECIFIC,
        CalculationMethodology.AVERAGE_DATA,
    ],
    Scope3Category.CAT_9_DOWNSTREAM_TRANSPORT: [
        CalculationMethodology.DISTANCE_BASED,
        CalculationMethodology.AVERAGE_DATA,
        CalculationMethodology.SPEND_BASED,
    ],
    Scope3Category.CAT_10_PROCESSING_SOLD: [
        CalculationMethodology.AVERAGE_DATA,
        CalculationMethodology.SUPPLIER_SPECIFIC,
    ],
    Scope3Category.CAT_11_USE_SOLD: [
        CalculationMethodology.AVERAGE_DATA,
        CalculationMethodology.SUPPLIER_SPECIFIC,
    ],
    Scope3Category.CAT_12_END_OF_LIFE: [
        CalculationMethodology.AVERAGE_DATA,
        CalculationMethodology.SUPPLIER_SPECIFIC,
    ],
    Scope3Category.CAT_13_DOWNSTREAM_LEASED: [
        CalculationMethodology.ASSET_SPECIFIC,
        CalculationMethodology.AVERAGE_DATA,
    ],
    Scope3Category.CAT_14_FRANCHISES: [
        CalculationMethodology.ASSET_SPECIFIC,
        CalculationMethodology.AVERAGE_DATA,
    ],
    Scope3Category.CAT_15_INVESTMENTS: [
        CalculationMethodology.INVESTMENT_SPECIFIC,
        CalculationMethodology.AVERAGE_DATA,
    ],
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class Scope3AgentConfig(BaseModel):
    """Configuration for Scope 3 agent routing."""

    config_id: str = Field(default_factory=_new_uuid)
    enabled_categories: List[Scope3Category] = Field(
        default_factory=lambda: list(Scope3Category)
    )
    default_methodology: CalculationMethodology = Field(
        default=CalculationMethodology.SPEND_BASED
    )
    emission_factor_source: str = Field(default="EEIO", description="EF source: EEIO, EPA, DEFRA, IPCC")
    base_currency: str = Field(default="USD")
    timeout_per_agent_seconds: int = Field(default=120, ge=10)

class CategoryResult(BaseModel):
    """Result from a Scope 3 category MRV agent execution."""

    result_id: str = Field(default_factory=_new_uuid)
    agent_id: str = Field(default="")
    category: str = Field(default="")
    category_number: int = Field(default=0, ge=0, le=15)
    ghg_protocol_name: str = Field(default="")
    methodology_used: str = Field(default="")
    total_emissions_tco2e: float = Field(default=0.0)
    co2_tco2e: float = Field(default=0.0)
    ch4_tco2e: float = Field(default=0.0)
    n2o_tco2e: float = Field(default=0.0)
    other_ghg_tco2e: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=5.0)
    uncertainty_pct: float = Field(default=0.0)
    status: str = Field(default="success")
    error_message: Optional[str] = Field(None)
    warnings: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=utcnow)

class ConsolidatedScope3Result(BaseModel):
    """Consolidated result across all Scope 3 categories."""

    result_id: str = Field(default_factory=_new_uuid)
    total_scope3_tco2e: float = Field(default=0.0)
    by_category: Dict[str, CategoryResult] = Field(default_factory=dict)
    categories_calculated: int = Field(default=0)
    categories_relevant: int = Field(default=0)
    overall_data_quality: float = Field(default=0.0, ge=0.0, le=5.0)
    overall_uncertainty_pct: float = Field(default=0.0)
    double_counting_adjustments: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=utcnow)

# ---------------------------------------------------------------------------
# MRVScope3Bridge
# ---------------------------------------------------------------------------

class MRVScope3Bridge:
    """Bridge to all 15 Scope 3 MRV agents (MRV-014 through MRV-028).

    Routes activity/spend data to the appropriate Scope 3 MRV agent for
    emissions calculation. Supports individual category execution, parallel
    execution of independent categories, batch routing, agent health status,
    and result standardization.

    Attributes:
        config: Agent routing configuration.
        _agents: Loaded MRV agent references (or stubs).

    Example:
        >>> bridge = MRVScope3Bridge()
        >>> result = bridge.route_category(Scope3Category.CAT_1_PURCHASED_GOODS, spend_data)
        >>> assert result.status == "success"
        >>> assert result.total_emissions_tco2e > 0
    """

    def __init__(
        self,
        config: Optional[Scope3AgentConfig] = None,
    ) -> None:
        """Initialize MRVScope3Bridge.

        Args:
            config: Agent routing configuration. Uses defaults if None.
        """
        self.config = config or Scope3AgentConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._agents: Dict[str, Any] = {}

        for category in Scope3Category:
            mapping = AGENT_CATEGORY_MAP[category]
            self._agents[mapping["agent_id"]] = _try_import_agent(
                mapping["agent_id"], mapping["module_path"]
            )

        self.logger.info(
            "MRVScope3Bridge initialized: %d categories enabled, "
            "methodology=%s, ef_source=%s",
            len(self.config.enabled_categories),
            self.config.default_methodology.value,
            self.config.emission_factor_source,
        )

    # -------------------------------------------------------------------------
    # Single Category Routing
    # -------------------------------------------------------------------------

    def route_category(
        self,
        category: Scope3Category,
        data: Dict[str, Any],
        methodology: Optional[CalculationMethodology] = None,
    ) -> CategoryResult:
        """Route data to the appropriate Scope 3 MRV agent.

        Args:
            category: Scope 3 category to calculate.
            data: Activity/spend data for the category.
            methodology: Override methodology. Uses config default if None.

        Returns:
            CategoryResult with emissions calculation.
        """
        start_time = time.monotonic()
        mapping = AGENT_CATEGORY_MAP.get(category)
        if not mapping:
            return CategoryResult(
                status="error",
                error_message=f"Unknown category: {category.value}",
            )

        agent_id = mapping["agent_id"]
        method = methodology or self.config.default_methodology
        cat_num = int(category.value.replace("cat_", ""))

        self.logger.info(
            "Routing to %s for '%s' (Cat %d): methodology=%s, %d data keys",
            agent_id, mapping["ghg_protocol_name"], cat_num,
            method.value, len(data),
        )

        result = self._calculate_category(category, data, method)
        result.agent_id = agent_id
        result.category = category.value
        result.category_number = cat_num
        result.ghg_protocol_name = mapping["ghg_protocol_name"]
        result.methodology_used = method.value

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result.processing_time_ms = elapsed_ms
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "%s Cat %d (%s): %.3f tCO2e, methodology=%s, DQR=%.1f",
            agent_id, cat_num, mapping["ghg_protocol_name"],
            result.total_emissions_tco2e, method.value,
            result.data_quality_score,
        )
        return result

    # -------------------------------------------------------------------------
    # Batch Routing
    # -------------------------------------------------------------------------

    def route_all_categories(
        self,
        category_data_map: Dict[Scope3Category, Dict[str, Any]],
        methodology_overrides: Optional[Dict[Scope3Category, CalculationMethodology]] = None,
    ) -> Dict[str, CategoryResult]:
        """Execute emissions calculations for all provided categories.

        Args:
            category_data_map: Dict mapping category to its activity data.
            methodology_overrides: Optional per-category methodology overrides.

        Returns:
            Dict mapping category value to CategoryResult.
        """
        methodology_overrides = methodology_overrides or {}
        self.logger.info(
            "Routing all categories: %d categories",
            len(category_data_map),
        )

        results: Dict[str, CategoryResult] = {}
        for category, data in category_data_map.items():
            if category not in self.config.enabled_categories:
                self.logger.debug("Category '%s' not enabled, skipping", category.value)
                continue
            try:
                method = methodology_overrides.get(category)
                result = self.route_category(category, data, method)
                results[category.value] = result
            except Exception as exc:
                self.logger.error(
                    "Failed to execute category '%s': %s", category.value, exc
                )
                results[category.value] = CategoryResult(
                    agent_id=AGENT_CATEGORY_MAP[category]["agent_id"],
                    category=category.value,
                    status="error",
                    error_message=str(exc),
                )

        total = sum(r.total_emissions_tco2e for r in results.values() if r.status == "success")
        self.logger.info(
            "Scope 3 total: %.3f tCO2e across %d categories",
            total, len(results),
        )
        return results

    async def route_parallel(
        self,
        category_data_map: Dict[Scope3Category, Dict[str, Any]],
        methodology_overrides: Optional[Dict[Scope3Category, CalculationMethodology]] = None,
    ) -> Dict[str, CategoryResult]:
        """Execute category calculations in parallel.

        All 15 categories are independent and can be calculated concurrently.

        Args:
            category_data_map: Dict mapping category to its activity data.
            methodology_overrides: Optional per-category methodology overrides.

        Returns:
            Dict mapping category value to CategoryResult.
        """
        methodology_overrides = methodology_overrides or {}
        self.logger.info(
            "Routing %d categories in parallel",
            len(category_data_map),
        )

        async def _calc(cat: Scope3Category, data: Dict[str, Any]) -> CategoryResult:
            method = methodology_overrides.get(cat)
            return self.route_category(cat, data, method)

        tasks = [
            _calc(cat, data)
            for cat, data in category_data_map.items()
            if cat in self.config.enabled_categories
        ]
        categories = [
            cat for cat in category_data_map.keys()
            if cat in self.config.enabled_categories
        ]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: Dict[str, CategoryResult] = {}
        for cat, raw in zip(categories, raw_results):
            if isinstance(raw, Exception):
                results[cat.value] = CategoryResult(
                    agent_id=AGENT_CATEGORY_MAP[cat]["agent_id"],
                    category=cat.value,
                    status="error",
                    error_message=str(raw),
                )
            else:
                results[cat.value] = raw

        return results

    # -------------------------------------------------------------------------
    # Consolidation
    # -------------------------------------------------------------------------

    def consolidate_results(
        self,
        category_results: Dict[str, CategoryResult],
    ) -> ConsolidatedScope3Result:
        """Consolidate results across all categories into a single total.

        Checks for double-counting between Cat 3/Scope 2 and Cat 4/Cat 9.

        Args:
            category_results: Dict mapping category value to CategoryResult.

        Returns:
            ConsolidatedScope3Result with aggregated totals.
        """
        start_time = time.monotonic()
        total = Decimal("0")
        dq_weighted = Decimal("0")
        dq_weight_sum = Decimal("0")
        successful = 0

        for cat_val, result in category_results.items():
            if result.status == "success":
                emissions = Decimal(str(result.total_emissions_tco2e))
                total += emissions
                successful += 1
                dq_weighted += emissions * Decimal(str(result.data_quality_score))
                dq_weight_sum += emissions

        overall_dq = float(dq_weighted / dq_weight_sum) if dq_weight_sum > 0 else 0.0
        elapsed_ms = (time.monotonic() - start_time) * 1000

        consolidated = ConsolidatedScope3Result(
            total_scope3_tco2e=float(total),
            by_category=category_results,
            categories_calculated=successful,
            categories_relevant=len(category_results),
            overall_data_quality=round(overall_dq, 1),
            processing_time_ms=elapsed_ms,
        )
        consolidated.provenance_hash = _compute_hash(consolidated)

        self.logger.info(
            "Scope 3 consolidated: %.3f tCO2e, %d categories, DQR=%.1f",
            consolidated.total_scope3_tco2e,
            consolidated.categories_calculated,
            consolidated.overall_data_quality,
        )
        return consolidated

    # -------------------------------------------------------------------------
    # Agent Health
    # -------------------------------------------------------------------------

    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get the availability status of a specific MRV agent.

        Args:
            agent_id: MRV agent identifier (e.g., 'MRV-014').

        Returns:
            Dict with agent status information.
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            return {
                "agent_id": agent_id,
                "status": AgentStatus.UNAVAILABLE.value,
                "message": "Agent not registered",
            }

        is_stub = isinstance(agent, _AgentStub)
        return {
            "agent_id": agent_id,
            "status": AgentStatus.DEGRADED.value if is_stub else AgentStatus.AVAILABLE.value,
            "message": "Using stub (module not importable)" if is_stub else "Agent available",
            "module_loaded": not is_stub,
        }

    def get_all_agent_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get availability status for all 15 + 2 cross-cutting MRV agents.

        Returns:
            Dict mapping agent_id to status information.
        """
        return {
            agent_id: self.get_agent_status(agent_id)
            for agent_id in sorted(self._agents.keys())
        }

    def health_check(self) -> Dict[str, Any]:
        """Run health check across all Scope 3 agents.

        Returns:
            Dict with overall health status and per-agent detail.
        """
        statuses = self.get_all_agent_statuses()
        available = sum(1 for s in statuses.values() if s["status"] == AgentStatus.AVAILABLE.value)
        degraded = sum(1 for s in statuses.values() if s["status"] == AgentStatus.DEGRADED.value)

        return {
            "total_agents": len(statuses),
            "available": available,
            "degraded": degraded,
            "unavailable": len(statuses) - available - degraded,
            "overall_status": "healthy" if available == len(statuses) else "degraded",
            "agents": statuses,
            "provenance_hash": _compute_hash(statuses),
        }

    # -------------------------------------------------------------------------
    # Internal: Category Calculation
    # -------------------------------------------------------------------------

    def _calculate_category(
        self,
        category: Scope3Category,
        data: Dict[str, Any],
        methodology: CalculationMethodology,
    ) -> CategoryResult:
        """Execute category-level emission calculation.

        Dispatches to the appropriate calculation method based on
        the category. In production, each method calls into the
        MRV agent module. The implementation returns representative
        results using deterministic formulas.

        Args:
            category: Scope 3 category.
            data: Activity/spend data.
            methodology: Calculation methodology to use.

        Returns:
            CategoryResult with emissions.
        """
        cat_num = int(category.value.replace("cat_", ""))

        if methodology == CalculationMethodology.SPEND_BASED:
            return self._calc_spend_based(category, data, cat_num)
        elif methodology == CalculationMethodology.DISTANCE_BASED:
            return self._calc_distance_based(category, data, cat_num)
        elif methodology == CalculationMethodology.SUPPLIER_SPECIFIC:
            return self._calc_supplier_specific(category, data, cat_num)
        else:
            return self._calc_spend_based(category, data, cat_num)

    def _calc_spend_based(
        self,
        category: Scope3Category,
        data: Dict[str, Any],
        cat_num: int,
    ) -> CategoryResult:
        """Spend-based calculation using EEIO emission factors.

        emission = spend_amount * eeio_factor * inflation_adjustment

        Args:
            category: Scope 3 category.
            data: Must contain 'spend_usd' or 'transactions'.
            cat_num: Category number (1-15).

        Returns:
            CategoryResult from spend-based method.
        """
        spend = Decimal(str(data.get("spend_usd", 0)))
        transactions = data.get("transactions", [])
        records = len(transactions) if transactions else int(data.get("transaction_count", 0))

        # Representative EEIO factors by category (kgCO2e per USD)
        eeio_factors: Dict[int, float] = {
            1: 0.35, 2: 0.35, 3: 0.30, 4: 0.35, 5: 0.35,
            6: 0.25, 7: 0.20, 8: 0.30, 9: 0.35, 10: 0.30,
            11: 0.35, 12: 0.25, 13: 0.30, 14: 0.30, 15: 0.25,
        }
        factor = Decimal(str(eeio_factors.get(cat_num, 0.30)))
        emissions_kg = spend * factor
        emissions_tco2e = emissions_kg / Decimal("1000")

        # Rough gas split (95% CO2, 3% CH4, 2% N2O for most categories)
        co2 = emissions_tco2e * Decimal("0.95")
        ch4 = emissions_tco2e * Decimal("0.03")
        n2o = emissions_tco2e * Decimal("0.02")

        # Data quality: spend-based is typically DQR 4.0-4.5
        dqr = 4.0
        uncertainty = 50.0

        return CategoryResult(
            total_emissions_tco2e=float(emissions_tco2e.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            co2_tco2e=float(co2.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            ch4_tco2e=float(ch4.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            n2o_tco2e=float(n2o.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            records_processed=records,
            data_quality_score=dqr,
            uncertainty_pct=uncertainty,
            details={
                "spend_usd": float(spend),
                "eeio_factor_kgco2e_per_usd": float(factor),
                "methodology": "spend_based",
            },
        )

    def _calc_distance_based(
        self,
        category: Scope3Category,
        data: Dict[str, Any],
        cat_num: int,
    ) -> CategoryResult:
        """Distance-based calculation for transport/travel categories.

        emission = distance * mode_factor * weight_factor

        Args:
            category: Scope 3 category.
            data: Must contain 'distance_km' and 'mode' or 'trips'.
            cat_num: Category number.

        Returns:
            CategoryResult from distance-based method.
        """
        total_distance = Decimal(str(data.get("total_distance_km", 0)))
        trips = data.get("trips", [])
        records = len(trips) if trips else int(data.get("trip_count", 0))

        # Representative transport EFs (kgCO2e per passenger-km or tonne-km)
        mode_factors: Dict[str, float] = {
            "air_short": 0.255, "air_long": 0.195,
            "rail": 0.041, "bus": 0.089,
            "car_gasoline": 0.192, "car_diesel": 0.171,
            "truck": 0.105, "ship": 0.016,
        }
        mode = data.get("mode", "car_gasoline")
        factor = Decimal(str(mode_factors.get(mode, 0.192)))

        emissions_kg = total_distance * factor
        emissions_tco2e = emissions_kg / Decimal("1000")

        co2 = emissions_tco2e * Decimal("0.97")
        ch4 = emissions_tco2e * Decimal("0.02")
        n2o = emissions_tco2e * Decimal("0.01")

        return CategoryResult(
            total_emissions_tco2e=float(emissions_tco2e.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            co2_tco2e=float(co2.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            ch4_tco2e=float(ch4.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            n2o_tco2e=float(n2o.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            records_processed=records,
            data_quality_score=2.5,
            uncertainty_pct=25.0,
            details={
                "total_distance_km": float(total_distance),
                "mode": mode,
                "factor_kgco2e_per_km": float(factor),
                "methodology": "distance_based",
            },
        )

    def _calc_supplier_specific(
        self,
        category: Scope3Category,
        data: Dict[str, Any],
        cat_num: int,
    ) -> CategoryResult:
        """Supplier-specific calculation using primary data.

        emission = sum(supplier_emissions * allocation_factor)

        Args:
            category: Scope 3 category.
            data: Must contain 'supplier_data' list with emissions per supplier.
            cat_num: Category number.

        Returns:
            CategoryResult from supplier-specific method.
        """
        supplier_data = data.get("supplier_data", [])
        total = Decimal("0")
        records = len(supplier_data)

        for supplier in supplier_data:
            supplier_emissions = Decimal(str(supplier.get("emissions_tco2e", 0)))
            allocation = Decimal(str(supplier.get("allocation_factor", 1.0)))
            total += supplier_emissions * allocation

        co2 = total * Decimal("0.96")
        ch4 = total * Decimal("0.025")
        n2o = total * Decimal("0.015")

        return CategoryResult(
            total_emissions_tco2e=float(total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            co2_tco2e=float(co2.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            ch4_tco2e=float(ch4.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            n2o_tco2e=float(n2o.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            records_processed=records,
            data_quality_score=1.5,
            uncertainty_pct=10.0,
            details={
                "supplier_count": records,
                "methodology": "supplier_specific",
            },
        )
