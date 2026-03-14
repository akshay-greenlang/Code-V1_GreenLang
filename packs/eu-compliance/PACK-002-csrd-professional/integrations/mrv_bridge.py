# -*- coding: utf-8 -*-
"""
ProfessionalMRVBridge - Enhanced MRV Bridge for CSRD Professional Pack
=======================================================================

This module extends PACK-001's MRVBridge with professional-grade features
for enterprise CSRD reporting. It adds intensity metrics calculation,
biogenic carbon tracking, base year recalculation, multi-entity routing,
Scope 3 screening, and enhanced provenance chains.

Enhanced Features over PACK-001:
    - Intensity metrics: per-revenue, per-employee, per-unit, per-m2, per-pkm
    - Biogenic carbon: separate tracking of biogenic CO2 and removals
    - Base year recalculation: handles structural changes per GHG Protocol
    - Multi-entity routing: routes calculations per subsidiary entity
    - Scope 3 screening: automated significance assessment (>40% = significant)
    - Enhanced provenance: entity_id and intensity fields in chain entries

Routing Architecture:
    ESRS E1 Code --> ProfessionalMRVBridge --> MRV Agent Instance
                                                      |
                                                      v
                                              CalculationResult
                                                      |
                            +-----------+-------------+--------+
                            v           v             v        v
                        Intensity   Biogenic     BaseYear   Entity
                        Metrics     Carbon       Recalc     Routing

Zero-Hallucination Guarantee:
    All routing is deterministic via a static lookup table. Intensity
    metrics use simple division. Base year recalculation uses explicit
    threshold comparison. No LLM is used in any calculation path.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-002 CSRD Professional
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash.

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _serialize_data(data: Any) -> str:
    """Serialize data to a stable string for hashing.

    Args:
        data: Data to serialize.

    Returns:
        Deterministic string representation.
    """
    if isinstance(data, dict):
        return json.dumps(data, sort_keys=True, default=str)
    return str(data)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ScopeType(str, Enum):
    """GHG Protocol emission scope classification."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class CalculationStatus(str, Enum):
    """Status of a calculation result."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class IntensityMetricType(str, Enum):
    """Types of emission intensity metrics."""
    PER_REVENUE = "per_revenue"
    PER_EMPLOYEE = "per_employee"
    PER_UNIT = "per_unit"
    PER_M2 = "per_m2"
    PER_PASSENGER_KM = "per_passenger_km"


class ScreeningSignificance(str, Enum):
    """Significance level for Scope 3 category screening."""
    SIGNIFICANT = "significant"
    MATERIAL = "material"
    NOT_SIGNIFICANT = "not_significant"
    NOT_RELEVANT = "not_relevant"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class ProfessionalMRVBridgeConfig(BaseModel):
    """Configuration for the Professional MRV Bridge."""

    # Inherited PACK-001 config fields
    enable_provenance: bool = Field(
        default=True, description="Enable provenance tracking"
    )
    enable_scope1: bool = Field(default=True, description="Enable Scope 1")
    enable_scope2: bool = Field(default=True, description="Enable Scope 2")
    enable_scope3: bool = Field(default=True, description="Enable Scope 3")
    enabled_scope3_categories: List[int] = Field(
        default_factory=lambda: list(range(1, 16)),
        description="Enabled Scope 3 categories (1-15)",
    )
    gwp_source: str = Field(default="AR6", description="GWP source")
    reporting_unit: str = Field(default="tCO2e", description="Reporting unit")
    timeout_per_calculation_seconds: int = Field(default=60)

    # Professional features
    enable_intensity_metrics: bool = Field(
        default=True, description="Enable intensity metric calculations"
    )
    enable_biogenic_carbon: bool = Field(
        default=True, description="Enable biogenic carbon tracking"
    )
    enable_base_year_recalculation: bool = Field(
        default=True, description="Enable base year recalculation"
    )
    enable_multi_entity: bool = Field(
        default=True, description="Enable multi-entity routing"
    )
    enable_scope3_screening: bool = Field(
        default=True, description="Enable Scope 3 screening"
    )
    scope3_significance_threshold_pct: float = Field(
        default=40.0, ge=0.0, le=100.0,
        description="Percentage of total above which a category is significant",
    )
    base_year_recalculation_threshold_pct: float = Field(
        default=5.0, ge=0.0, le=100.0,
        description="Percentage change threshold triggering base year recalculation",
    )


class ProvenanceChainEntry(BaseModel):
    """A single entry in the provenance chain with professional fields."""

    step_index: int = Field(..., description="Step index in the chain")
    agent_id: str = Field(..., description="Agent that performed this step")
    metric_code: str = Field(..., description="ESRS metric code")
    input_hash: str = Field(..., description="SHA-256 hash of input data")
    output_hash: str = Field(..., description="SHA-256 hash of output data")
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Step timestamp"
    )
    execution_time_ms: float = Field(default=0.0, description="Step duration")
    entity_id: Optional[str] = Field(
        None, description="Entity ID for multi-entity routing"
    )
    intensity_metrics_included: bool = Field(
        default=False, description="Whether intensity metrics were calculated"
    )
    biogenic_flag: bool = Field(
        default=False, description="Whether biogenic carbon was tracked"
    )


class MRVRoutingEntry(BaseModel):
    """A single entry in the MRV routing table."""

    metric_code: str = Field(..., description="ESRS E1 metric code")
    agent_id: str = Field(..., description="Target MRV agent ID")
    scope: ScopeType = Field(..., description="Emission scope")
    description: str = Field(default="", description="Human-readable description")
    scope3_category: Optional[int] = Field(None, description="Scope 3 cat 1-15")
    is_reconciliation: bool = Field(default=False)
    is_professional: bool = Field(
        default=False, description="Whether this is a PACK-002 professional entry"
    )


class CalculationResult(BaseModel):
    """Result from a single MRV calculation."""

    metric_code: str = Field(..., description="ESRS E1 metric code")
    agent_id: str = Field(..., description="MRV agent that performed calculation")
    scope: ScopeType = Field(..., description="Emission scope")
    status: CalculationStatus = Field(..., description="Calculation status")
    emissions_value: float = Field(default=0.0, description="Total in tCO2e")
    emissions_unit: str = Field(default="tCO2e")
    co2_value: float = Field(default=0.0, description="CO2 component")
    ch4_value: float = Field(default=0.0, description="CH4 in CO2e")
    n2o_value: float = Field(default=0.0, description="N2O in CO2e")
    other_ghg_value: float = Field(default=0.0, description="Other GHG in CO2e")
    methodology: str = Field(default="")
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty_pct: float = Field(default=0.0, ge=0.0)
    provenance_hash: str = Field(default="")
    provenance_chain: List[ProvenanceChainEntry] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)
    error_message: Optional[str] = Field(None)
    raw_output: Dict[str, Any] = Field(default_factory=dict)
    entity_id: Optional[str] = Field(None, description="Entity that owns this calc")


class IntensityMetrics(BaseModel):
    """Emission intensity metrics for ESRS E1 reporting."""

    per_revenue: Optional[float] = Field(
        None, description="tCO2e per EUR million revenue"
    )
    per_employee: Optional[float] = Field(
        None, description="tCO2e per employee"
    )
    per_unit: Optional[float] = Field(
        None, description="tCO2e per production unit"
    )
    per_m2: Optional[float] = Field(
        None, description="tCO2e per square meter"
    )
    per_passenger_km: Optional[float] = Field(
        None, description="tCO2e per passenger-km"
    )
    denominators: Dict[str, float] = Field(
        default_factory=dict,
        description="Denominator values used for intensity calculation",
    )
    provenance_hash: str = Field(default="")


class BiogenicCarbonResult(BaseModel):
    """Biogenic carbon tracking result per ESRS E1-6."""

    biogenic_co2_emissions: float = Field(
        default=0.0, description="Gross biogenic CO2 emissions (tCO2)"
    )
    biogenic_removals: float = Field(
        default=0.0, description="Biogenic CO2 removals (tCO2)"
    )
    net_biogenic: float = Field(
        default=0.0, description="Net biogenic CO2 (emissions - removals)"
    )
    biogenic_sources: List[str] = Field(
        default_factory=list,
        description="Sources of biogenic emissions",
    )
    methodology: str = Field(default="")
    provenance_hash: str = Field(default="")


class BaseYearConfig(BaseModel):
    """Configuration for base year recalculation."""

    base_year: int = Field(..., ge=2015, le=2030, description="Base year")
    original_emissions: float = Field(
        ..., ge=0.0, description="Original base year emissions (tCO2e)"
    )
    structural_changes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of structural changes since base year",
    )
    recalculation_threshold_pct: float = Field(
        default=5.0, ge=0.0, le=100.0,
        description="Threshold triggering recalculation (%)",
    )


class BaseYearResult(BaseModel):
    """Result of a base year recalculation."""

    base_year: int = Field(..., description="Base year")
    original_emissions: float = Field(
        ..., description="Original base year emissions"
    )
    recalculated_emissions: float = Field(
        ..., description="Recalculated base year emissions"
    )
    adjustment: float = Field(
        ..., description="Adjustment amount (recalculated - original)"
    )
    adjustment_pct: float = Field(
        ..., description="Adjustment as percentage of original"
    )
    recalculation_triggered: bool = Field(
        ..., description="Whether threshold was exceeded"
    )
    structural_changes_applied: List[Dict[str, Any]] = Field(
        default_factory=list
    )
    methodology: str = Field(default="GHG Protocol Base Year Recalculation")
    provenance_hash: str = Field(default="")


class CategoryScreening(BaseModel):
    """Scope 3 category screening result."""

    category_number: int = Field(..., ge=1, le=15)
    category_name: str = Field(default="")
    estimated_emissions: float = Field(default=0.0, description="Estimated tCO2e")
    pct_of_total: float = Field(default=0.0, description="% of total Scope 3")
    significance: ScreeningSignificance = Field(
        ..., description="Significance assessment"
    )
    data_availability: str = Field(
        default="low", description="Data availability (low, medium, high)"
    )
    recommended_approach: str = Field(
        default="", description="Recommended calculation approach"
    )
    rationale: str = Field(default="", description="Screening rationale")


class EntityCalculationResult(BaseModel):
    """Calculation result scoped to a specific entity."""

    entity_id: str = Field(..., description="Entity identifier")
    entity_name: str = Field(default="", description="Entity name")
    scope1_total: float = Field(default=0.0)
    scope2_location_total: float = Field(default=0.0)
    scope2_market_total: float = Field(default=0.0)
    scope3_total: float = Field(default=0.0)
    total_emissions: float = Field(default=0.0)
    calculation_results: List[CalculationResult] = Field(default_factory=list)
    intensity_metrics: Optional[IntensityMetrics] = Field(None)
    biogenic_carbon: Optional[BiogenicCarbonResult] = Field(None)
    provenance_hash: str = Field(default="")


class AggregatedEmissions(BaseModel):
    """Aggregated emissions across multiple calculation results."""

    total_emissions: float = Field(default=0.0)
    co2_total: float = Field(default=0.0)
    ch4_total: float = Field(default=0.0)
    n2o_total: float = Field(default=0.0)
    other_ghg_total: float = Field(default=0.0)
    calculation_count: int = Field(default=0)
    successful_count: int = Field(default=0)
    failed_count: int = Field(default=0)
    avg_data_quality: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Static Routing Table (extends PACK-001)
# ---------------------------------------------------------------------------

# Scope 3 category names
SCOPE3_CATEGORY_NAMES: Dict[int, str] = {
    1: "Purchased Goods & Services",
    2: "Capital Goods",
    3: "Fuel & Energy Activities",
    4: "Upstream Transportation",
    5: "Waste Generated in Operations",
    6: "Business Travel",
    7: "Employee Commuting",
    8: "Upstream Leased Assets",
    9: "Downstream Transportation",
    10: "Processing of Sold Products",
    11: "Use of Sold Products",
    12: "End-of-Life Treatment",
    13: "Downstream Leased Assets",
    14: "Franchises",
    15: "Investments",
}

MRV_ROUTING_TABLE: Dict[str, MRVRoutingEntry] = {
    # -----------------------------------------------------------------
    # Scope 1: Direct emissions (E1-1-*)
    # -----------------------------------------------------------------
    "E1-1-1": MRVRoutingEntry(
        metric_code="E1-1-1", agent_id="GL-MRV-X-001",
        scope=ScopeType.SCOPE_1,
        description="Stationary combustion emissions from fixed sources",
    ),
    "E1-1-2": MRVRoutingEntry(
        metric_code="E1-1-2", agent_id="GL-MRV-X-003",
        scope=ScopeType.SCOPE_1,
        description="Mobile combustion emissions from transport fleet",
    ),
    "E1-1-3": MRVRoutingEntry(
        metric_code="E1-1-3", agent_id="GL-MRV-X-004",
        scope=ScopeType.SCOPE_1,
        description="Process emissions from industrial processes",
    ),
    "E1-1-4": MRVRoutingEntry(
        metric_code="E1-1-4", agent_id="GL-MRV-X-005",
        scope=ScopeType.SCOPE_1,
        description="Fugitive emissions from leaks and venting",
    ),
    "E1-1-5": MRVRoutingEntry(
        metric_code="E1-1-5", agent_id="GL-MRV-X-002",
        scope=ScopeType.SCOPE_1,
        description="Refrigerant and fluorinated gas emissions",
    ),
    "E1-1-6": MRVRoutingEntry(
        metric_code="E1-1-6", agent_id="GL-MRV-X-006",
        scope=ScopeType.SCOPE_1,
        description="Land use change emissions",
    ),
    "E1-1-7": MRVRoutingEntry(
        metric_code="E1-1-7", agent_id="GL-MRV-X-007",
        scope=ScopeType.SCOPE_1,
        description="Waste treatment emissions (on-site)",
    ),
    "E1-1-8": MRVRoutingEntry(
        metric_code="E1-1-8", agent_id="GL-MRV-X-008",
        scope=ScopeType.SCOPE_1,
        description="Agricultural emissions",
    ),
    # -----------------------------------------------------------------
    # Scope 2: Indirect energy emissions (E1-2-*)
    # -----------------------------------------------------------------
    "E1-2-1": MRVRoutingEntry(
        metric_code="E1-2-1", agent_id="GL-MRV-X-009",
        scope=ScopeType.SCOPE_2,
        description="Location-based electricity emissions",
    ),
    "E1-2-2": MRVRoutingEntry(
        metric_code="E1-2-2", agent_id="GL-MRV-X-010",
        scope=ScopeType.SCOPE_2,
        description="Market-based electricity emissions",
    ),
    "E1-2-3": MRVRoutingEntry(
        metric_code="E1-2-3", agent_id="GL-MRV-X-011",
        scope=ScopeType.SCOPE_2,
        description="Steam and heat purchase emissions",
    ),
    "E1-2-4": MRVRoutingEntry(
        metric_code="E1-2-4", agent_id="GL-MRV-X-012",
        scope=ScopeType.SCOPE_2,
        description="Cooling purchase emissions",
    ),
    "E1-2-R": MRVRoutingEntry(
        metric_code="E1-2-R", agent_id="GL-MRV-X-013",
        scope=ScopeType.SCOPE_2,
        description="Scope 2 dual reporting reconciliation",
        is_reconciliation=True,
    ),
    # -----------------------------------------------------------------
    # Scope 3: Value chain emissions (E1-3-Cat*)
    # -----------------------------------------------------------------
    **{
        f"E1-3-Cat{cat:02d}": MRVRoutingEntry(
            metric_code=f"E1-3-Cat{cat:02d}",
            agent_id=f"GL-MRV-X-{13 + cat:03d}",
            scope=ScopeType.SCOPE_3,
            scope3_category=cat,
            description=SCOPE3_CATEGORY_NAMES.get(cat, f"Category {cat}"),
        )
        for cat in range(1, 16)
    },
    # -----------------------------------------------------------------
    # Cross-cutting
    # -----------------------------------------------------------------
    "E1-3-MAP": MRVRoutingEntry(
        metric_code="E1-3-MAP", agent_id="GL-MRV-X-029",
        scope=ScopeType.SCOPE_3,
        description="Scope 3 category mapping and classification",
    ),
    "E1-AUDIT": MRVRoutingEntry(
        metric_code="E1-AUDIT", agent_id="GL-MRV-X-030",
        scope=ScopeType.SCOPE_1,
        description="Audit trail and lineage tracking",
    ),
    # -----------------------------------------------------------------
    # PACK-002 Professional routing entries
    # -----------------------------------------------------------------
    "E1-INT-REV": MRVRoutingEntry(
        metric_code="E1-INT-REV", agent_id="GL-PRO-INTENSITY",
        scope=ScopeType.SCOPE_1,
        description="Intensity metric: emissions per EUR million revenue",
        is_professional=True,
    ),
    "E1-INT-EMP": MRVRoutingEntry(
        metric_code="E1-INT-EMP", agent_id="GL-PRO-INTENSITY",
        scope=ScopeType.SCOPE_1,
        description="Intensity metric: emissions per employee",
        is_professional=True,
    ),
    "E1-INT-UNIT": MRVRoutingEntry(
        metric_code="E1-INT-UNIT", agent_id="GL-PRO-INTENSITY",
        scope=ScopeType.SCOPE_1,
        description="Intensity metric: emissions per production unit",
        is_professional=True,
    ),
    "E1-BIO": MRVRoutingEntry(
        metric_code="E1-BIO", agent_id="GL-PRO-BIOGENIC",
        scope=ScopeType.SCOPE_1,
        description="Biogenic carbon emissions and removals tracking",
        is_professional=True,
    ),
    "E1-BASE": MRVRoutingEntry(
        metric_code="E1-BASE", agent_id="GL-PRO-BASEYEAR",
        scope=ScopeType.SCOPE_1,
        description="Base year recalculation per GHG Protocol",
        is_professional=True,
    ),
    "E1-SCR": MRVRoutingEntry(
        metric_code="E1-SCR", agent_id="GL-MRV-X-029",
        scope=ScopeType.SCOPE_3,
        description="Scope 3 category screening and significance assessment",
        is_professional=True,
    ),
}


# ---------------------------------------------------------------------------
# ProfessionalMRVBridge Implementation
# ---------------------------------------------------------------------------


class ProfessionalMRVBridge:
    """Enhanced MRV Bridge for CSRD Professional Pack.

    Extends PACK-001's MRVBridge with intensity metrics, biogenic carbon
    tracking, base year recalculation, multi-entity routing, Scope 3
    screening, and enhanced provenance chains.

    Attributes:
        config: Bridge configuration
        _routing_table: Static ESRS metric to MRV agent routing table
        _agents: Registry of MRV agent instances
        _provenance_chains: Accumulated provenance entries
        _step_counter: Global step counter for provenance ordering

    Example:
        >>> bridge = ProfessionalMRVBridge()
        >>> result = await bridge.route_calculation("E1-1-1", data)
        >>> intensity = bridge.calculate_intensity_metrics(
        ...     result.emissions_value,
        ...     {"revenue_eur_m": 500, "employee_count": 1200},
        ... )
    """

    def __init__(
        self,
        config: Optional[ProfessionalMRVBridgeConfig] = None,
    ) -> None:
        """Initialize the Professional MRV Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or ProfessionalMRVBridgeConfig()
        self._routing_table: Dict[str, MRVRoutingEntry] = dict(MRV_ROUTING_TABLE)
        self._agents: Dict[str, Any] = {}
        self._provenance_chains: List[ProvenanceChainEntry] = []
        self._step_counter: int = 0

        logger.info(
            "ProfessionalMRVBridge initialized: %d routing entries, "
            "intensity=%s, biogenic=%s, base_year=%s, multi_entity=%s",
            len(self._routing_table),
            self.config.enable_intensity_metrics,
            self.config.enable_biogenic_carbon,
            self.config.enable_base_year_recalculation,
            self.config.enable_multi_entity,
        )

    # -------------------------------------------------------------------------
    # Routing
    # -------------------------------------------------------------------------

    def get_routing_entry(self, metric_code: str) -> Optional[MRVRoutingEntry]:
        """Look up the routing entry for an ESRS metric code.

        Args:
            metric_code: ESRS E1 metric code.

        Returns:
            MRVRoutingEntry if found, None otherwise.
        """
        return self._routing_table.get(metric_code)

    def get_all_routing_entries(self) -> Dict[str, MRVRoutingEntry]:
        """Return the complete routing table.

        Returns:
            Dictionary of metric_code to MRVRoutingEntry.
        """
        return dict(self._routing_table)

    def get_professional_routing_entries(self) -> Dict[str, MRVRoutingEntry]:
        """Return only PACK-002 professional routing entries.

        Returns:
            Dictionary of professional-only routing entries.
        """
        return {
            k: v for k, v in self._routing_table.items()
            if v.is_professional
        }

    async def route_calculation(
        self,
        metric_code: str,
        data: Dict[str, Any],
        entity_id: Optional[str] = None,
    ) -> CalculationResult:
        """Route a calculation request to the appropriate MRV agent.

        Args:
            metric_code: ESRS E1 metric code.
            data: Activity data for the calculation.
            entity_id: Optional entity ID for multi-entity routing.

        Returns:
            CalculationResult with emissions and provenance.

        Raises:
            ValueError: If the metric code is not found.
        """
        entry = self.get_routing_entry(metric_code)
        if entry is None:
            raise ValueError(
                f"Unknown metric code '{metric_code}'. "
                f"Valid codes: {sorted(self._routing_table.keys())}"
            )

        if not self._is_scope_enabled(entry.scope):
            return CalculationResult(
                metric_code=metric_code,
                agent_id=entry.agent_id,
                scope=entry.scope,
                status=CalculationStatus.SKIPPED,
                entity_id=entity_id,
            )

        if (
            entry.scope3_category is not None
            and entry.scope3_category not in self.config.enabled_scope3_categories
        ):
            return CalculationResult(
                metric_code=metric_code,
                agent_id=entry.agent_id,
                scope=entry.scope,
                status=CalculationStatus.SKIPPED,
                entity_id=entity_id,
            )

        return await self._execute_mrv_calculation(entry, data, entity_id)

    async def _execute_mrv_calculation(
        self,
        entry: MRVRoutingEntry,
        data: Dict[str, Any],
        entity_id: Optional[str] = None,
    ) -> CalculationResult:
        """Execute a single MRV calculation via the routed agent.

        Args:
            entry: The routing entry.
            data: Activity data.
            entity_id: Optional entity ID.

        Returns:
            CalculationResult with computed values and provenance.
        """
        start_time = time.monotonic()
        input_hash = _compute_hash(
            f"{entry.metric_code}:{entity_id or 'default'}:{_serialize_data(data)}"
        )

        try:
            agent = self._agents.get(entry.agent_id)
            if agent is not None:
                raw_result = await self._invoke_agent(agent, data)
            else:
                raw_result = self._calculate_deterministic(entry, data)

            elapsed_ms = (time.monotonic() - start_time) * 1000
            output_hash = _compute_hash(_serialize_data(raw_result))

            emissions = raw_result.get("total_emissions", 0.0)

            provenance_entry = ProvenanceChainEntry(
                step_index=self._step_counter,
                agent_id=entry.agent_id,
                metric_code=entry.metric_code,
                input_hash=input_hash,
                output_hash=output_hash,
                execution_time_ms=elapsed_ms,
                entity_id=entity_id,
            )
            self._step_counter += 1
            self._provenance_chains.append(provenance_entry)

            provenance_hash = _compute_hash(
                f"{input_hash}:{output_hash}:{entry.agent_id}:{entity_id or ''}"
            )

            result = CalculationResult(
                metric_code=entry.metric_code,
                agent_id=entry.agent_id,
                scope=entry.scope,
                status=CalculationStatus.SUCCESS,
                emissions_value=emissions,
                emissions_unit=self.config.reporting_unit,
                co2_value=raw_result.get("co2", 0.0),
                ch4_value=raw_result.get("ch4", 0.0),
                n2o_value=raw_result.get("n2o", 0.0),
                other_ghg_value=raw_result.get("other_ghg", 0.0),
                methodology=raw_result.get("methodology", ""),
                data_quality_score=raw_result.get("data_quality_score", 0.0),
                uncertainty_pct=raw_result.get("uncertainty_pct", 0.0),
                provenance_hash=provenance_hash,
                provenance_chain=[provenance_entry],
                execution_time_ms=elapsed_ms,
                raw_output=raw_result,
                entity_id=entity_id,
            )

            logger.info(
                "Metric %s calculated by %s: %.4f %s in %.1fms (entity=%s)",
                entry.metric_code, entry.agent_id,
                emissions, self.config.reporting_unit, elapsed_ms,
                entity_id or "default",
            )
            return result

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Calculation failed for metric %s via %s: %s",
                entry.metric_code, entry.agent_id, exc, exc_info=True,
            )
            return CalculationResult(
                metric_code=entry.metric_code,
                agent_id=entry.agent_id,
                scope=entry.scope,
                status=CalculationStatus.FAILED,
                execution_time_ms=elapsed_ms,
                error_message=str(exc),
                entity_id=entity_id,
            )

    # -------------------------------------------------------------------------
    # Intensity Metrics
    # -------------------------------------------------------------------------

    def calculate_intensity_metrics(
        self,
        total_emissions: float,
        denominators: Dict[str, float],
    ) -> IntensityMetrics:
        """Calculate emission intensity metrics (zero-hallucination: division only).

        Args:
            total_emissions: Total emissions in tCO2e.
            denominators: Dictionary of denominator values:
                - revenue_eur_m: Revenue in EUR millions
                - employee_count: Number of employees
                - production_units: Number of production units
                - floor_area_m2: Floor area in square meters
                - passenger_km: Passenger-kilometers

        Returns:
            IntensityMetrics with calculated intensity values.
        """
        metrics = IntensityMetrics(denominators=denominators)

        revenue = denominators.get("revenue_eur_m", 0.0)
        if revenue > 0:
            metrics.per_revenue = round(total_emissions / revenue, 6)

        employees = denominators.get("employee_count", 0.0)
        if employees > 0:
            metrics.per_employee = round(total_emissions / employees, 6)

        units = denominators.get("production_units", 0.0)
        if units > 0:
            metrics.per_unit = round(total_emissions / units, 6)

        area = denominators.get("floor_area_m2", 0.0)
        if area > 0:
            metrics.per_m2 = round(total_emissions / area, 6)

        pkm = denominators.get("passenger_km", 0.0)
        if pkm > 0:
            metrics.per_passenger_km = round(total_emissions / pkm, 6)

        metrics.provenance_hash = _compute_hash(
            f"intensity:{total_emissions}:{_serialize_data(denominators)}"
        )

        logger.info(
            "Intensity metrics calculated: per_rev=%s, per_emp=%s, "
            "per_unit=%s, per_m2=%s",
            metrics.per_revenue, metrics.per_employee,
            metrics.per_unit, metrics.per_m2,
        )
        return metrics

    # -------------------------------------------------------------------------
    # Biogenic Carbon
    # -------------------------------------------------------------------------

    def calculate_biogenic_carbon(
        self,
        biogenic_emissions: float,
        biogenic_removals: float,
        sources: Optional[List[str]] = None,
        methodology: str = "ESRS E1-6 biogenic carbon",
    ) -> BiogenicCarbonResult:
        """Calculate biogenic carbon balance (zero-hallucination: subtraction only).

        Args:
            biogenic_emissions: Gross biogenic CO2 emissions (tCO2).
            biogenic_removals: Biogenic CO2 removals (tCO2).
            sources: List of biogenic emission source descriptions.
            methodology: Calculation methodology name.

        Returns:
            BiogenicCarbonResult with net biogenic balance.
        """
        net = biogenic_emissions - biogenic_removals

        result = BiogenicCarbonResult(
            biogenic_co2_emissions=biogenic_emissions,
            biogenic_removals=biogenic_removals,
            net_biogenic=net,
            biogenic_sources=sources or [],
            methodology=methodology,
        )
        result.provenance_hash = _compute_hash(
            f"biogenic:{biogenic_emissions}:{biogenic_removals}:{net}"
        )

        logger.info(
            "Biogenic carbon: emissions=%.2f, removals=%.2f, net=%.2f tCO2",
            biogenic_emissions, biogenic_removals, net,
        )
        return result

    # -------------------------------------------------------------------------
    # Base Year Recalculation
    # -------------------------------------------------------------------------

    def recalculate_base_year(
        self,
        config: BaseYearConfig,
        current_data: Optional[Dict[str, Any]] = None,
    ) -> BaseYearResult:
        """Recalculate base year emissions after structural changes.

        Per GHG Protocol guidance, base year emissions are recalculated
        when structural changes (acquisitions, divestments, mergers,
        methodology changes) exceed the significance threshold.

        Args:
            config: Base year configuration with structural changes.
            current_data: Optional current period data for reference.

        Returns:
            BaseYearResult with recalculated emissions and adjustment.
        """
        total_adjustment = 0.0
        applied_changes: List[Dict[str, Any]] = []

        for change in config.structural_changes:
            change_type = change.get("type", "unknown")
            emissions_impact = float(change.get("emissions_impact", 0.0))
            total_adjustment += emissions_impact
            applied_changes.append({
                "type": change_type,
                "emissions_impact": emissions_impact,
                "description": change.get("description", ""),
                "date": change.get("date", ""),
            })

        recalculated = config.original_emissions + total_adjustment
        adjustment_pct = 0.0
        if config.original_emissions > 0:
            adjustment_pct = abs(total_adjustment) / config.original_emissions * 100.0

        triggered = adjustment_pct >= config.recalculation_threshold_pct

        result = BaseYearResult(
            base_year=config.base_year,
            original_emissions=config.original_emissions,
            recalculated_emissions=recalculated,
            adjustment=total_adjustment,
            adjustment_pct=round(adjustment_pct, 4),
            recalculation_triggered=triggered,
            structural_changes_applied=applied_changes,
        )
        result.provenance_hash = _compute_hash(
            f"baseyear:{config.base_year}:{config.original_emissions}:"
            f"{recalculated}:{adjustment_pct}"
        )

        logger.info(
            "Base year recalculation: year=%d, original=%.2f, "
            "recalculated=%.2f, adjustment=%.4f%%, triggered=%s",
            config.base_year, config.original_emissions,
            recalculated, adjustment_pct, triggered,
        )
        return result

    # -------------------------------------------------------------------------
    # Multi-Entity Routing
    # -------------------------------------------------------------------------

    async def route_entity_calculations(
        self,
        entity_id: str,
        entity_name: str,
        data: Dict[str, Any],
        denominators: Optional[Dict[str, float]] = None,
    ) -> EntityCalculationResult:
        """Route all calculations for a single entity.

        Args:
            entity_id: Unique entity identifier.
            entity_name: Human-readable entity name.
            data: Activity data keyed by metric code.
            denominators: Optional intensity metric denominators.

        Returns:
            EntityCalculationResult with per-entity emissions.
        """
        logger.info(
            "Routing calculations for entity: %s (%s)",
            entity_id, entity_name,
        )
        start_time = time.monotonic()

        calculation_results: List[CalculationResult] = []
        scope1_total = 0.0
        scope2_loc_total = 0.0
        scope2_mkt_total = 0.0
        scope3_total = 0.0

        for metric_code, metric_data in data.items():
            if not isinstance(metric_data, dict):
                continue
            entry = self.get_routing_entry(metric_code)
            if entry is None:
                continue

            result = await self.route_calculation(
                metric_code, metric_data, entity_id=entity_id
            )
            calculation_results.append(result)

            if result.status == CalculationStatus.SUCCESS:
                if entry.scope == ScopeType.SCOPE_1:
                    scope1_total += result.emissions_value
                elif entry.scope == ScopeType.SCOPE_2:
                    if metric_code == "E1-2-1":
                        scope2_loc_total += result.emissions_value
                    elif metric_code == "E1-2-2":
                        scope2_mkt_total += result.emissions_value
                    else:
                        scope2_loc_total += result.emissions_value
                        scope2_mkt_total += result.emissions_value
                elif entry.scope == ScopeType.SCOPE_3:
                    scope3_total += result.emissions_value

        total_emissions = scope1_total + scope2_loc_total + scope3_total

        # Calculate intensity metrics if denominators provided
        intensity = None
        if denominators and self.config.enable_intensity_metrics:
            intensity = self.calculate_intensity_metrics(
                total_emissions, denominators
            )

        entity_result = EntityCalculationResult(
            entity_id=entity_id,
            entity_name=entity_name,
            scope1_total=scope1_total,
            scope2_location_total=scope2_loc_total,
            scope2_market_total=scope2_mkt_total,
            scope3_total=scope3_total,
            total_emissions=total_emissions,
            calculation_results=calculation_results,
            intensity_metrics=intensity,
        )
        entity_result.provenance_hash = _compute_hash(
            f"entity:{entity_id}:{total_emissions}:"
            f"{scope1_total}:{scope2_loc_total}:{scope3_total}"
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Entity %s calculations complete: total=%.4f tCO2e "
            "(S1=%.4f, S2-loc=%.4f, S2-mkt=%.4f, S3=%.4f) in %.1fms",
            entity_id, total_emissions, scope1_total,
            scope2_loc_total, scope2_mkt_total, scope3_total,
            elapsed_ms,
        )
        return entity_result

    # -------------------------------------------------------------------------
    # Scope 3 Screening
    # -------------------------------------------------------------------------

    def screen_scope3_categories(
        self,
        data: Dict[int, Dict[str, Any]],
    ) -> List[CategoryScreening]:
        """Screen Scope 3 categories for significance.

        Per GHG Protocol guidance, categories exceeding the significance
        threshold (default 40% of total Scope 3) are flagged as significant.

        Args:
            data: Dictionary mapping category number to screening data
                  with keys: estimated_emissions, data_availability.

        Returns:
            List of CategoryScreening results sorted by emissions desc.
        """
        screenings: List[CategoryScreening] = []
        total_estimated = 0.0

        # First pass: sum total estimated emissions
        for cat_num, cat_data in data.items():
            est = float(cat_data.get("estimated_emissions", 0.0))
            total_estimated += est

        # Second pass: assess significance
        for cat_num in sorted(data.keys()):
            cat_data = data[cat_num]
            est = float(cat_data.get("estimated_emissions", 0.0))
            availability = cat_data.get("data_availability", "low")

            pct_of_total = 0.0
            if total_estimated > 0:
                pct_of_total = est / total_estimated * 100.0

            if pct_of_total >= self.config.scope3_significance_threshold_pct:
                significance = ScreeningSignificance.SIGNIFICANT
                approach = "primary_data"
                rationale = (
                    f"Category represents {pct_of_total:.1f}% of total Scope 3 "
                    f"(above {self.config.scope3_significance_threshold_pct}% threshold)"
                )
            elif pct_of_total >= 5.0:
                significance = ScreeningSignificance.MATERIAL
                approach = "hybrid_method"
                rationale = (
                    f"Category represents {pct_of_total:.1f}% of total Scope 3 "
                    f"(material but below significance threshold)"
                )
            elif est > 0:
                significance = ScreeningSignificance.NOT_SIGNIFICANT
                approach = "spend_based"
                rationale = (
                    f"Category represents {pct_of_total:.1f}% of total Scope 3 "
                    f"(below materiality threshold)"
                )
            else:
                significance = ScreeningSignificance.NOT_RELEVANT
                approach = "excluded"
                rationale = "No estimated emissions for this category"

            screenings.append(CategoryScreening(
                category_number=cat_num,
                category_name=SCOPE3_CATEGORY_NAMES.get(cat_num, f"Category {cat_num}"),
                estimated_emissions=est,
                pct_of_total=round(pct_of_total, 2),
                significance=significance,
                data_availability=availability,
                recommended_approach=approach,
                rationale=rationale,
            ))

        # Sort by emissions descending
        screenings.sort(key=lambda s: s.estimated_emissions, reverse=True)

        significant_count = sum(
            1 for s in screenings
            if s.significance == ScreeningSignificance.SIGNIFICANT
        )
        logger.info(
            "Scope 3 screening: %d categories screened, %d significant, "
            "total estimated=%.2f tCO2e",
            len(screenings), significant_count, total_estimated,
        )
        return screenings

    # -------------------------------------------------------------------------
    # Aggregation (same as PACK-001)
    # -------------------------------------------------------------------------

    def _aggregate_results(
        self, results: List[CalculationResult]
    ) -> AggregatedEmissions:
        """Aggregate multiple CalculationResults.

        Args:
            results: List of calculation results.

        Returns:
            AggregatedEmissions with summed values.
        """
        if not results:
            return AggregatedEmissions()

        successful = [r for r in results if r.status == CalculationStatus.SUCCESS]
        failed = [r for r in results if r.status == CalculationStatus.FAILED]

        total = sum(r.emissions_value for r in successful)
        co2 = sum(r.co2_value for r in successful)
        ch4 = sum(r.ch4_value for r in successful)
        n2o = sum(r.n2o_value for r in successful)
        other = sum(r.other_ghg_value for r in successful)

        avg_quality = 0.0
        if successful:
            avg_quality = sum(r.data_quality_score for r in successful) / len(successful)

        hashes = sorted(r.provenance_hash for r in results if r.provenance_hash)
        combined_hash = _compute_hash("|".join(hashes)) if hashes else ""

        return AggregatedEmissions(
            total_emissions=total,
            co2_total=co2,
            ch4_total=ch4,
            n2o_total=n2o,
            other_ghg_total=other,
            calculation_count=len(results),
            successful_count=len(successful),
            failed_count=len(failed),
            avg_data_quality=round(avg_quality, 4),
            provenance_hash=combined_hash,
        )

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _is_scope_enabled(self, scope: ScopeType) -> bool:
        """Check whether a scope is enabled."""
        return {
            ScopeType.SCOPE_1: self.config.enable_scope1,
            ScopeType.SCOPE_2: self.config.enable_scope2,
            ScopeType.SCOPE_3: self.config.enable_scope3,
        }.get(scope, False)

    async def _invoke_agent(
        self, agent: Any, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invoke an MRV agent instance.

        Args:
            agent: The MRV agent object.
            data: Input data.

        Returns:
            Dictionary with calculation output.
        """
        import asyncio

        execute_fn = getattr(agent, "execute", None)
        if execute_fn is None:
            return {"total_emissions": 0.0}

        if asyncio.iscoroutinefunction(execute_fn):
            result = await execute_fn(data)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, execute_fn, data)

        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "dict"):
            return result.dict()
        if isinstance(result, dict):
            return result
        return {"total_emissions": 0.0, "raw": str(result)}

    def _calculate_deterministic(
        self, entry: MRVRoutingEntry, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deterministic fallback calculation: activity * factor.

        Args:
            entry: The routing entry.
            data: Activity data with quantity/factor fields.

        Returns:
            Dictionary with calculated values.
        """
        quantity = float(data.get("quantity", data.get("activity_data", 0.0)))
        factor = float(data.get("emission_factor", data.get("factor", 0.0)))
        emissions = quantity * factor

        return {
            "total_emissions": emissions,
            "co2": emissions * 0.95,
            "ch4": emissions * 0.03,
            "n2o": emissions * 0.015,
            "other_ghg": emissions * 0.005,
            "methodology": f"deterministic_fallback:{entry.agent_id}",
            "data_quality_score": 0.5,
            "uncertainty_pct": 25.0,
        }

    def get_provenance_chain(self) -> List[ProvenanceChainEntry]:
        """Return the complete provenance chain.

        Returns:
            List of ProvenanceChainEntry in execution order.
        """
        return list(self._provenance_chains)

    def reset_provenance(self) -> None:
        """Clear the provenance chain for a fresh cycle."""
        self._provenance_chains.clear()
        self._step_counter = 0
        logger.debug("Professional MRV provenance chain reset")
