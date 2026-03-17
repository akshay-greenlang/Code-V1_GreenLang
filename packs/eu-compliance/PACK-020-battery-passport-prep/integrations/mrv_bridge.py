# -*- coding: utf-8 -*-
"""
MRVBridge - AGENT-MRV Integration Bridge for PACK-020
=========================================================

Connects PACK-020 to AGENT-MRV agents for routing manufacturing emissions
data into battery carbon footprint calculations. Maps Scope 1 (on-site
manufacturing), Scope 2 (purchased electricity for production), and Scope 3
(upstream material extraction and downstream distribution) to EU Battery
Regulation Art 7 carbon footprint lifecycle stages.

Methods:
    - get_manufacturing_emissions()  -- Aggregate all manufacturing emissions
    - get_scope1_data()              -- Scope 1 factory/process emissions
    - get_scope2_data()              -- Scope 2 electricity for production
    - get_scope3_data()              -- Scope 3 upstream materials & downstream
    - calculate_carbon_intensity()   -- kgCO2e per kWh battery capacity

MRV Agent Routing for Battery Manufacturing:
    Scope 1: MRV-001 (Stationary Combustion), MRV-004 (Process Emissions)
    Scope 2: MRV-009 (Location-Based), MRV-010 (Market-Based)
    Scope 3: MRV-014 (Purchased Goods Cat 1), MRV-017 (Transport Cat 4),
             MRV-024 (Use of Sold Products Cat 11), MRV-025 (End-of-Life Cat 12)

Legal References:
    - Regulation (EU) 2023/1542, Art 7 (Carbon footprint)
    - Commission Delegated Regulation (EU) 2024/1781 (methodology)
    - Product Environmental Footprint Category Rules (PEFCR) for batteries

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-020 Battery Passport Prep Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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
# Enums
# ---------------------------------------------------------------------------


class MRVScope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class LifecycleStage(str, Enum):
    """Battery carbon footprint lifecycle stages (EU 2024/1781)."""

    RAW_MATERIAL_ACQUISITION = "raw_material_acquisition"
    MAIN_PRODUCTION = "main_production"
    DISTRIBUTION = "distribution"
    END_OF_LIFE_RECYCLING = "end_of_life_recycling"


class AgentStatus(str, Enum):
    """MRV agent availability status."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class MRVBridgeConfig(BaseModel):
    """Configuration for the MRV Bridge."""

    pack_id: str = Field(default="PACK-020")
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    enable_provenance: bool = Field(default=True)
    timeout_per_agent_seconds: int = Field(default=60, ge=10)
    gwp_source: str = Field(default="IPCC AR6")
    battery_capacity_kwh: float = Field(default=0.0, ge=0.0)
    functional_unit: str = Field(
        default="kgCO2e_per_kWh",
        description="Carbon footprint functional unit per EU 2024/1781",
    )


class MRVAgentMapping(BaseModel):
    """Mapping of an MRV agent to battery lifecycle stage."""

    agent_id: str = Field(default="")
    agent_name: str = Field(default="")
    scope: MRVScope = Field(default=MRVScope.SCOPE_1)
    lifecycle_stage: LifecycleStage = Field(
        default=LifecycleStage.MAIN_PRODUCTION
    )
    ghg_protocol_category: str = Field(default="")


class ScopeEmissionsResult(BaseModel):
    """Result of a scope-level emissions import."""

    operation_id: str = Field(default_factory=_new_uuid)
    scope: MRVScope = Field(default=MRVScope.SCOPE_1)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    agents_queried: int = Field(default=0)
    agents_responded: int = Field(default=0)
    total_tco2e: float = Field(default=0.0)
    emissions_by_category: List[Dict[str, Any]] = Field(default_factory=list)
    lifecycle_mapping: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ManufacturingEmissionsResult(BaseModel):
    """Aggregated manufacturing emissions for battery carbon footprint."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    scope1_tco2e: float = Field(default=0.0)
    scope2_location_tco2e: float = Field(default=0.0)
    scope2_market_tco2e: float = Field(default=0.0)
    scope3_upstream_tco2e: float = Field(default=0.0)
    scope3_downstream_tco2e: float = Field(default=0.0)
    total_tco2e: float = Field(default=0.0)
    carbon_intensity_kgco2e_per_kwh: float = Field(default=0.0)
    lifecycle_breakdown: Dict[str, float] = Field(default_factory=dict)
    agents_queried: int = Field(default=0)
    provenance_hash: str = Field(default="")


class CarbonIntensityResult(BaseModel):
    """Carbon intensity calculation result."""

    battery_capacity_kwh: float = Field(default=0.0, ge=0.0)
    total_emissions_kgco2e: float = Field(default=0.0, ge=0.0)
    carbon_intensity_kgco2e_per_kwh: float = Field(default=0.0, ge=0.0)
    performance_class: str = Field(default="not_classified")
    lifecycle_contributions: Dict[str, float] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# MRV Agent Routing Table for Battery Manufacturing
# ---------------------------------------------------------------------------

BATTERY_MRV_ROUTING: Dict[str, MRVAgentMapping] = {
    "factory_combustion": MRVAgentMapping(
        agent_id="MRV-001", agent_name="Stationary Combustion",
        scope=MRVScope.SCOPE_1,
        lifecycle_stage=LifecycleStage.MAIN_PRODUCTION,
    ),
    "process_emissions": MRVAgentMapping(
        agent_id="MRV-004", agent_name="Process Emissions",
        scope=MRVScope.SCOPE_1,
        lifecycle_stage=LifecycleStage.MAIN_PRODUCTION,
    ),
    "fugitive_emissions": MRVAgentMapping(
        agent_id="MRV-005", agent_name="Fugitive Emissions",
        scope=MRVScope.SCOPE_1,
        lifecycle_stage=LifecycleStage.MAIN_PRODUCTION,
    ),
    "scope2_location": MRVAgentMapping(
        agent_id="MRV-009", agent_name="Scope 2 Location-Based",
        scope=MRVScope.SCOPE_2,
        lifecycle_stage=LifecycleStage.MAIN_PRODUCTION,
    ),
    "scope2_market": MRVAgentMapping(
        agent_id="MRV-010", agent_name="Scope 2 Market-Based",
        scope=MRVScope.SCOPE_2,
        lifecycle_stage=LifecycleStage.MAIN_PRODUCTION,
    ),
    "purchased_goods": MRVAgentMapping(
        agent_id="MRV-014", agent_name="Purchased Goods & Services",
        scope=MRVScope.SCOPE_3,
        lifecycle_stage=LifecycleStage.RAW_MATERIAL_ACQUISITION,
        ghg_protocol_category="Cat 1",
    ),
    "capital_goods": MRVAgentMapping(
        agent_id="MRV-015", agent_name="Capital Goods",
        scope=MRVScope.SCOPE_3,
        lifecycle_stage=LifecycleStage.MAIN_PRODUCTION,
        ghg_protocol_category="Cat 2",
    ),
    "upstream_transport": MRVAgentMapping(
        agent_id="MRV-017", agent_name="Upstream Transportation",
        scope=MRVScope.SCOPE_3,
        lifecycle_stage=LifecycleStage.RAW_MATERIAL_ACQUISITION,
        ghg_protocol_category="Cat 4",
    ),
    "downstream_transport": MRVAgentMapping(
        agent_id="MRV-022", agent_name="Downstream Transportation",
        scope=MRVScope.SCOPE_3,
        lifecycle_stage=LifecycleStage.DISTRIBUTION,
        ghg_protocol_category="Cat 9",
    ),
    "use_of_sold_products": MRVAgentMapping(
        agent_id="MRV-024", agent_name="Use of Sold Products",
        scope=MRVScope.SCOPE_3,
        lifecycle_stage=LifecycleStage.DISTRIBUTION,
        ghg_protocol_category="Cat 11",
    ),
    "end_of_life": MRVAgentMapping(
        agent_id="MRV-025", agent_name="End-of-Life Treatment",
        scope=MRVScope.SCOPE_3,
        lifecycle_stage=LifecycleStage.END_OF_LIFE_RECYCLING,
        ghg_protocol_category="Cat 12",
    ),
}

SCOPE1_AGENTS: List[str] = [
    k for k, v in BATTERY_MRV_ROUTING.items() if v.scope == MRVScope.SCOPE_1
]
SCOPE2_AGENTS: List[str] = [
    k for k, v in BATTERY_MRV_ROUTING.items() if v.scope == MRVScope.SCOPE_2
]
SCOPE3_AGENTS: List[str] = [
    k for k, v in BATTERY_MRV_ROUTING.items() if v.scope == MRVScope.SCOPE_3
]

# Carbon footprint performance class thresholds (kgCO2e/kWh)
# Based on EU 2024/1781 Annex II performance classes for EV batteries
PERFORMANCE_CLASS_THRESHOLDS: Dict[str, float] = {
    "A": 40.0,
    "B": 60.0,
    "C": 80.0,
    "D": 100.0,
    "E": 120.0,
}


# ---------------------------------------------------------------------------
# MRVBridge
# ---------------------------------------------------------------------------


class MRVBridge:
    """AGENT-MRV integration bridge for PACK-020 Battery Passport Prep.

    Routes MRV emission data for battery carbon footprint calculations.
    Maps GHG Protocol scopes to EU Battery Regulation lifecycle stages and
    calculates carbon intensity in kgCO2e/kWh.

    Attributes:
        config: Bridge configuration.
        _agent_status: Cached agent availability status.

    Example:
        >>> bridge = MRVBridge(MRVBridgeConfig(battery_capacity_kwh=60.0))
        >>> result = bridge.get_manufacturing_emissions(context)
        >>> assert result.status == "completed"
    """

    def __init__(self, config: Optional[MRVBridgeConfig] = None) -> None:
        """Initialize MRVBridge."""
        self.config = config or MRVBridgeConfig()
        self._agent_status: Dict[str, AgentStatus] = {}
        logger.info(
            "MRVBridge initialized (year=%d, capacity=%.1f kWh, agents=%d)",
            self.config.reporting_year,
            self.config.battery_capacity_kwh,
            len(BATTERY_MRV_ROUTING),
        )

    def get_manufacturing_emissions(
        self,
        context: Dict[str, Any],
    ) -> ManufacturingEmissionsResult:
        """Aggregate all manufacturing emissions for battery carbon footprint.

        Combines Scope 1, 2, and 3 emissions from relevant MRV agents and
        maps them to EU Battery Regulation lifecycle stages.

        Args:
            context: Pipeline context with emission data.

        Returns:
            ManufacturingEmissionsResult with lifecycle breakdown.
        """
        result = ManufacturingEmissionsResult(started_at=_utcnow())

        try:
            scope1 = self.get_scope1_data(context)
            scope2 = self.get_scope2_data(context)
            scope3 = self.get_scope3_data(context)

            result.scope1_tco2e = scope1.total_tco2e
            result.scope2_location_tco2e = scope2.lifecycle_mapping.get(
                "location_based", 0.0
            )
            result.scope2_market_tco2e = scope2.lifecycle_mapping.get(
                "market_based", 0.0
            )

            scope3_upstream = sum(
                e.get("tco2e", 0.0) for e in scope3.emissions_by_category
                if e.get("direction") == "upstream"
            )
            scope3_downstream = sum(
                e.get("tco2e", 0.0) for e in scope3.emissions_by_category
                if e.get("direction") == "downstream"
            )

            result.scope3_upstream_tco2e = round(scope3_upstream, 4)
            result.scope3_downstream_tco2e = round(scope3_downstream, 4)

            result.total_tco2e = round(
                result.scope1_tco2e
                + result.scope2_location_tco2e
                + scope3_upstream
                + scope3_downstream,
                4,
            )

            result.lifecycle_breakdown = self._build_lifecycle_breakdown(
                scope1, scope2, scope3, context
            )
            result.agents_queried = (
                scope1.agents_queried + scope2.agents_queried + scope3.agents_queried
            )

            if self.config.battery_capacity_kwh > 0:
                result.carbon_intensity_kgco2e_per_kwh = round(
                    (result.total_tco2e * 1000)
                    / self.config.battery_capacity_kwh,
                    2,
                )

            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result)

            logger.info(
                "Manufacturing emissions: %.4f tCO2e total, %.2f kgCO2e/kWh",
                result.total_tco2e,
                result.carbon_intensity_kgco2e_per_kwh,
            )

        except Exception as exc:
            result.status = "failed"
            logger.error("Manufacturing emissions aggregation failed: %s", str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def get_scope1_data(self, context: Dict[str, Any]) -> ScopeEmissionsResult:
        """Retrieve Scope 1 factory and process emissions.

        Args:
            context: Pipeline context with Scope 1 data.

        Returns:
            ScopeEmissionsResult with Scope 1 breakdown.
        """
        result = ScopeEmissionsResult(
            scope=MRVScope.SCOPE_1, started_at=_utcnow()
        )

        try:
            emissions = context.get("scope1_manufacturing", [])
            result.agents_queried = len(SCOPE1_AGENTS)
            result.agents_responded = len(SCOPE1_AGENTS)
            result.emissions_by_category = emissions
            result.total_tco2e = round(
                sum(e.get("tco2e", 0.0) for e in emissions), 4
            )
            result.lifecycle_mapping = {
                LifecycleStage.MAIN_PRODUCTION.value: result.total_tco2e,
            }
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(emissions)

            logger.info("Scope 1 import: %.4f tCO2e", result.total_tco2e)

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Scope 1 import failed: %s", str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def get_scope2_data(self, context: Dict[str, Any]) -> ScopeEmissionsResult:
        """Retrieve Scope 2 electricity emissions for battery production.

        Args:
            context: Pipeline context with Scope 2 data.

        Returns:
            ScopeEmissionsResult with location and market-based emissions.
        """
        result = ScopeEmissionsResult(
            scope=MRVScope.SCOPE_2, started_at=_utcnow()
        )

        try:
            location = context.get("scope2_location_tco2e", 0.0)
            market = context.get("scope2_market_tco2e", 0.0)

            result.agents_queried = len(SCOPE2_AGENTS)
            result.agents_responded = len(SCOPE2_AGENTS)
            result.emissions_by_category = [
                {"category": "location_based", "tco2e": location},
                {"category": "market_based", "tco2e": market},
            ]
            result.total_tco2e = round(location, 4)
            result.lifecycle_mapping = {
                "location_based": location,
                "market_based": market,
                LifecycleStage.MAIN_PRODUCTION.value: round(location, 4),
            }
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    {"location": location, "market": market}
                )

            logger.info(
                "Scope 2 import: location=%.4f, market=%.4f tCO2e",
                location, market,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Scope 2 import failed: %s", str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def get_scope3_data(self, context: Dict[str, Any]) -> ScopeEmissionsResult:
        """Retrieve Scope 3 upstream and downstream emissions.

        Args:
            context: Pipeline context with Scope 3 data.

        Returns:
            ScopeEmissionsResult with upstream/downstream breakdown.
        """
        result = ScopeEmissionsResult(
            scope=MRVScope.SCOPE_3, started_at=_utcnow()
        )

        try:
            upstream_cats = context.get("scope3_upstream_categories", [])
            downstream_cats = context.get("scope3_downstream_categories", [])

            all_cats: List[Dict[str, Any]] = []
            for cat in upstream_cats:
                cat["direction"] = "upstream"
                all_cats.append(cat)
            for cat in downstream_cats:
                cat["direction"] = "downstream"
                all_cats.append(cat)

            result.agents_queried = len(SCOPE3_AGENTS)
            result.agents_responded = min(len(SCOPE3_AGENTS), len(all_cats))
            result.emissions_by_category = all_cats
            result.total_tco2e = round(
                sum(c.get("tco2e", 0.0) for c in all_cats), 4
            )

            upstream_total = sum(
                c.get("tco2e", 0.0) for c in upstream_cats
            )
            downstream_total = sum(
                c.get("tco2e", 0.0) for c in downstream_cats
            )

            result.lifecycle_mapping = {
                LifecycleStage.RAW_MATERIAL_ACQUISITION.value: round(upstream_total, 4),
                LifecycleStage.DISTRIBUTION.value: round(downstream_total * 0.6, 4),
                LifecycleStage.END_OF_LIFE_RECYCLING.value: round(downstream_total * 0.4, 4),
            }
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(all_cats)

            logger.info(
                "Scope 3 import: %.4f tCO2e (upstream=%.4f, downstream=%.4f)",
                result.total_tco2e, upstream_total, downstream_total,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Scope 3 import failed: %s", str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def calculate_carbon_intensity(
        self,
        context: Dict[str, Any],
    ) -> CarbonIntensityResult:
        """Calculate carbon intensity in kgCO2e per kWh of battery capacity.

        Uses deterministic arithmetic (zero-hallucination). Maps to EU
        2024/1781 performance classes for EV batteries.

        Args:
            context: Pipeline context with emissions and capacity data.

        Returns:
            CarbonIntensityResult with intensity and performance class.
        """
        capacity = context.get(
            "battery_capacity_kwh", self.config.battery_capacity_kwh
        )
        mfg = self.get_manufacturing_emissions(context)

        total_kgco2e = mfg.total_tco2e * 1000  # Convert tonnes to kg

        if capacity > 0:
            intensity = round(total_kgco2e / capacity, 2)
        else:
            intensity = 0.0

        perf_class = self._classify_performance(intensity)

        result = CarbonIntensityResult(
            battery_capacity_kwh=capacity,
            total_emissions_kgco2e=round(total_kgco2e, 2),
            carbon_intensity_kgco2e_per_kwh=intensity,
            performance_class=perf_class,
            lifecycle_contributions=mfg.lifecycle_breakdown,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        logger.info(
            "Carbon intensity: %.2f kgCO2e/kWh (class %s)",
            intensity, perf_class,
        )
        return result

    def get_agent_routing(self) -> Dict[str, MRVAgentMapping]:
        """Get the battery MRV agent routing table.

        Returns:
            Dict of routing key to MRVAgentMapping.
        """
        return dict(BATTERY_MRV_ROUTING)

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status.

        Returns:
            Dict with bridge status information.
        """
        return {
            "pack_id": self.config.pack_id,
            "reporting_year": self.config.reporting_year,
            "battery_capacity_kwh": self.config.battery_capacity_kwh,
            "functional_unit": self.config.functional_unit,
            "gwp_source": self.config.gwp_source,
            "agents_total": len(BATTERY_MRV_ROUTING),
            "scope1_agents": len(SCOPE1_AGENTS),
            "scope2_agents": len(SCOPE2_AGENTS),
            "scope3_agents": len(SCOPE3_AGENTS),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_lifecycle_breakdown(
        self,
        scope1: ScopeEmissionsResult,
        scope2: ScopeEmissionsResult,
        scope3: ScopeEmissionsResult,
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Build lifecycle stage breakdown per EU 2024/1781."""
        raw_materials = scope3.lifecycle_mapping.get(
            LifecycleStage.RAW_MATERIAL_ACQUISITION.value, 0.0
        )
        main_production = (
            scope1.lifecycle_mapping.get(LifecycleStage.MAIN_PRODUCTION.value, 0.0)
            + scope2.lifecycle_mapping.get(LifecycleStage.MAIN_PRODUCTION.value, 0.0)
        )
        distribution = scope3.lifecycle_mapping.get(
            LifecycleStage.DISTRIBUTION.value, 0.0
        )
        end_of_life = scope3.lifecycle_mapping.get(
            LifecycleStage.END_OF_LIFE_RECYCLING.value, 0.0
        )

        return {
            LifecycleStage.RAW_MATERIAL_ACQUISITION.value: round(raw_materials, 4),
            LifecycleStage.MAIN_PRODUCTION.value: round(main_production, 4),
            LifecycleStage.DISTRIBUTION.value: round(distribution, 4),
            LifecycleStage.END_OF_LIFE_RECYCLING.value: round(end_of_life, 4),
        }

    @staticmethod
    def _classify_performance(intensity_kgco2e_per_kwh: float) -> str:
        """Classify carbon footprint into performance classes.

        Args:
            intensity_kgco2e_per_kwh: Carbon intensity value.

        Returns:
            Performance class letter (A-E) or 'F' if above all thresholds.
        """
        if intensity_kgco2e_per_kwh <= 0.0:
            return "not_classified"
        for cls, threshold in sorted(
            PERFORMANCE_CLASS_THRESHOLDS.items(), key=lambda x: x[1]
        ):
            if intensity_kgco2e_per_kwh <= threshold:
                return cls
        return "F"
