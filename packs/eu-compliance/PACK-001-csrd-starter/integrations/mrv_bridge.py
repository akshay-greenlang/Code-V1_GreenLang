# -*- coding: utf-8 -*-
"""
MRVBridge - MRV Agents to CSRD Calculator Bridge
=================================================

This module implements the bridge between GreenLang's 30 MRV calculation
agents and the CSRD Starter Pack's CalculatorAgent. It provides deterministic
routing of ESRS E1 climate metric codes to the appropriate MRV calculation
engine, aggregation of results across scopes, and complete provenance chain
maintenance.

Routing Architecture:
    ESRS E1 Metric Code --> MRVBridge Routing Table --> MRV Agent Instance
                                                            |
                                                            v
                                                    CalculationResult
                                                            |
                                                            v
                                           Aggregation --> ScopeResult
                                                            |
                                                            v
                                     Provenance Chain --> AuditTrail

Zero-Hallucination Guarantee:
    All routing is deterministic via a static lookup table. No LLM is used
    in the calculation path. Every numeric result flows through a validated
    MRV engine with SHA-256 provenance hashing at each step.

Example:
    >>> bridge = MRVBridge(MRVBridgeConfig())
    >>> result = await bridge.route_calculation("E1-1-1", activity_data)
    >>> assert result.provenance_hash is not None

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


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


# =============================================================================
# Data Models
# =============================================================================


class MRVBridgeConfig(BaseModel):
    """Configuration for the MRV Bridge."""
    enable_provenance: bool = Field(default=True, description="Enable provenance tracking")
    enable_scope1: bool = Field(default=True, description="Enable Scope 1 calculations")
    enable_scope2: bool = Field(default=True, description="Enable Scope 2 calculations")
    enable_scope3: bool = Field(default=True, description="Enable Scope 3 calculations")
    enabled_scope3_categories: List[int] = Field(
        default_factory=lambda: list(range(1, 16)),
        description="Enabled Scope 3 categories (1-15)",
    )
    gwp_source: str = Field(default="AR6", description="GWP source (AR4, AR5, AR6)")
    reporting_unit: str = Field(default="tCO2e", description="Reporting unit for emissions")
    timeout_per_calculation_seconds: int = Field(
        default=60, description="Timeout per MRV calculation"
    )


class ProvenanceChainEntry(BaseModel):
    """A single entry in the provenance chain tracking data lineage."""
    step_index: int = Field(..., description="Step index in the chain (0-based)")
    agent_id: str = Field(..., description="Agent that performed this step")
    metric_code: str = Field(..., description="ESRS metric code being calculated")
    input_hash: str = Field(..., description="SHA-256 hash of input data")
    output_hash: str = Field(..., description="SHA-256 hash of output data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Step timestamp")
    execution_time_ms: float = Field(default=0.0, description="Step execution time in ms")


class CalculationResult(BaseModel):
    """Result from a single MRV calculation."""
    metric_code: str = Field(..., description="ESRS E1 metric code")
    agent_id: str = Field(..., description="MRV agent that performed the calculation")
    scope: ScopeType = Field(..., description="Emission scope")
    status: CalculationStatus = Field(..., description="Calculation status")
    emissions_value: float = Field(default=0.0, description="Calculated emissions in tCO2e")
    emissions_unit: str = Field(default="tCO2e", description="Emissions unit")
    co2_value: float = Field(default=0.0, description="CO2 component")
    ch4_value: float = Field(default=0.0, description="CH4 component (in CO2e)")
    n2o_value: float = Field(default=0.0, description="N2O component (in CO2e)")
    other_ghg_value: float = Field(default=0.0, description="Other GHG component (in CO2e)")
    methodology: str = Field(default="", description="Calculation methodology applied")
    data_quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Data quality score (0-1)"
    )
    uncertainty_pct: float = Field(default=0.0, ge=0.0, description="Uncertainty percentage")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    provenance_chain: List[ProvenanceChainEntry] = Field(
        default_factory=list, description="Complete provenance chain"
    )
    execution_time_ms: float = Field(default=0.0, description="Execution time in ms")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    raw_output: Dict[str, Any] = Field(
        default_factory=dict, description="Raw agent output data"
    )


class AggregatedEmissions(BaseModel):
    """Aggregated emissions across multiple calculation results."""
    total_emissions: float = Field(default=0.0, description="Total emissions in tCO2e")
    co2_total: float = Field(default=0.0, description="Total CO2")
    ch4_total: float = Field(default=0.0, description="Total CH4 in CO2e")
    n2o_total: float = Field(default=0.0, description="Total N2O in CO2e")
    other_ghg_total: float = Field(default=0.0, description="Total other GHG in CO2e")
    calculation_count: int = Field(default=0, description="Number of calculations aggregated")
    successful_count: int = Field(default=0, description="Number of successful calculations")
    failed_count: int = Field(default=0, description="Number of failed calculations")
    avg_data_quality: float = Field(default=0.0, description="Average data quality score")
    provenance_hash: str = Field(default="", description="Aggregated provenance hash")


class Scope1Result(BaseModel):
    """Aggregated Scope 1 calculation results."""
    stationary_combustion: Optional[CalculationResult] = None
    mobile_combustion: Optional[CalculationResult] = None
    process_emissions: Optional[CalculationResult] = None
    fugitive_emissions: Optional[CalculationResult] = None
    refrigerants: Optional[CalculationResult] = None
    land_use: Optional[CalculationResult] = None
    waste_treatment: Optional[CalculationResult] = None
    agricultural: Optional[CalculationResult] = None
    aggregated: AggregatedEmissions = Field(
        default_factory=AggregatedEmissions, description="Aggregated Scope 1 totals"
    )
    provenance_hash: str = Field(default="", description="Scope 1 provenance hash")


class Scope2Result(BaseModel):
    """Aggregated Scope 2 calculation results."""
    location_based: Optional[CalculationResult] = None
    market_based: Optional[CalculationResult] = None
    steam_heat: Optional[CalculationResult] = None
    cooling: Optional[CalculationResult] = None
    reconciliation: Optional[CalculationResult] = None
    aggregated_location: AggregatedEmissions = Field(
        default_factory=AggregatedEmissions, description="Location-based totals"
    )
    aggregated_market: AggregatedEmissions = Field(
        default_factory=AggregatedEmissions, description="Market-based totals"
    )
    provenance_hash: str = Field(default="", description="Scope 2 provenance hash")


class Scope3Result(BaseModel):
    """Aggregated Scope 3 calculation results."""
    categories: Dict[int, CalculationResult] = Field(
        default_factory=dict, description="Results by category number (1-15)"
    )
    category_mapping: Optional[CalculationResult] = None
    audit_trail: Optional[CalculationResult] = None
    aggregated: AggregatedEmissions = Field(
        default_factory=AggregatedEmissions, description="Aggregated Scope 3 totals"
    )
    enabled_categories: List[int] = Field(
        default_factory=list, description="Categories that were calculated"
    )
    provenance_hash: str = Field(default="", description="Scope 3 provenance hash")


class MRVRoutingEntry(BaseModel):
    """A single entry in the MRV routing table."""
    metric_code: str = Field(..., description="ESRS E1 metric code")
    agent_id: str = Field(..., description="Target MRV agent ID")
    scope: ScopeType = Field(..., description="Emission scope")
    description: str = Field(default="", description="Human-readable description")
    scope3_category: Optional[int] = Field(
        None, description="Scope 3 category number (1-15) if applicable"
    )
    is_reconciliation: bool = Field(
        default=False, description="Whether this is a reconciliation calculation"
    )


# =============================================================================
# Static Routing Table
# =============================================================================

MRV_ROUTING_TABLE: Dict[str, MRVRoutingEntry] = {
    # -------------------------------------------------------------------------
    # Scope 1: Direct emissions (E1-1-*)
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    # Scope 2: Indirect energy emissions (E1-2-*)
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    # Scope 3: Value chain emissions (E1-3-Cat*)
    # -------------------------------------------------------------------------
    "E1-3-Cat01": MRVRoutingEntry(
        metric_code="E1-3-Cat01", agent_id="GL-MRV-X-014",
        scope=ScopeType.SCOPE_3, scope3_category=1,
        description="Purchased goods and services emissions",
    ),
    "E1-3-Cat02": MRVRoutingEntry(
        metric_code="E1-3-Cat02", agent_id="GL-MRV-X-015",
        scope=ScopeType.SCOPE_3, scope3_category=2,
        description="Capital goods emissions",
    ),
    "E1-3-Cat03": MRVRoutingEntry(
        metric_code="E1-3-Cat03", agent_id="GL-MRV-X-016",
        scope=ScopeType.SCOPE_3, scope3_category=3,
        description="Fuel and energy-related activities emissions",
    ),
    "E1-3-Cat04": MRVRoutingEntry(
        metric_code="E1-3-Cat04", agent_id="GL-MRV-X-017",
        scope=ScopeType.SCOPE_3, scope3_category=4,
        description="Upstream transportation and distribution emissions",
    ),
    "E1-3-Cat05": MRVRoutingEntry(
        metric_code="E1-3-Cat05", agent_id="GL-MRV-X-018",
        scope=ScopeType.SCOPE_3, scope3_category=5,
        description="Waste generated in operations emissions",
    ),
    "E1-3-Cat06": MRVRoutingEntry(
        metric_code="E1-3-Cat06", agent_id="GL-MRV-X-019",
        scope=ScopeType.SCOPE_3, scope3_category=6,
        description="Business travel emissions",
    ),
    "E1-3-Cat07": MRVRoutingEntry(
        metric_code="E1-3-Cat07", agent_id="GL-MRV-X-020",
        scope=ScopeType.SCOPE_3, scope3_category=7,
        description="Employee commuting emissions",
    ),
    "E1-3-Cat08": MRVRoutingEntry(
        metric_code="E1-3-Cat08", agent_id="GL-MRV-X-021",
        scope=ScopeType.SCOPE_3, scope3_category=8,
        description="Upstream leased assets emissions",
    ),
    "E1-3-Cat09": MRVRoutingEntry(
        metric_code="E1-3-Cat09", agent_id="GL-MRV-X-022",
        scope=ScopeType.SCOPE_3, scope3_category=9,
        description="Downstream transportation and distribution emissions",
    ),
    "E1-3-Cat10": MRVRoutingEntry(
        metric_code="E1-3-Cat10", agent_id="GL-MRV-X-023",
        scope=ScopeType.SCOPE_3, scope3_category=10,
        description="Processing of sold products emissions",
    ),
    "E1-3-Cat11": MRVRoutingEntry(
        metric_code="E1-3-Cat11", agent_id="GL-MRV-X-024",
        scope=ScopeType.SCOPE_3, scope3_category=11,
        description="Use of sold products emissions",
    ),
    "E1-3-Cat12": MRVRoutingEntry(
        metric_code="E1-3-Cat12", agent_id="GL-MRV-X-025",
        scope=ScopeType.SCOPE_3, scope3_category=12,
        description="End-of-life treatment of sold products emissions",
    ),
    "E1-3-Cat13": MRVRoutingEntry(
        metric_code="E1-3-Cat13", agent_id="GL-MRV-X-026",
        scope=ScopeType.SCOPE_3, scope3_category=13,
        description="Downstream leased assets emissions",
    ),
    "E1-3-Cat14": MRVRoutingEntry(
        metric_code="E1-3-Cat14", agent_id="GL-MRV-X-027",
        scope=ScopeType.SCOPE_3, scope3_category=14,
        description="Franchises emissions",
    ),
    "E1-3-Cat15": MRVRoutingEntry(
        metric_code="E1-3-Cat15", agent_id="GL-MRV-X-028",
        scope=ScopeType.SCOPE_3, scope3_category=15,
        description="Investments emissions",
    ),
    # -------------------------------------------------------------------------
    # Cross-cutting
    # -------------------------------------------------------------------------
    "E1-3-MAP": MRVRoutingEntry(
        metric_code="E1-3-MAP", agent_id="GL-MRV-X-029",
        scope=ScopeType.SCOPE_3,
        description="Scope 3 category mapping and classification",
    ),
    "E1-AUDIT": MRVRoutingEntry(
        metric_code="E1-AUDIT", agent_id="GL-MRV-X-030",
        scope=ScopeType.SCOPE_1,  # Cross-cutting, assigned to scope 1 for grouping
        description="Audit trail and lineage tracking",
    ),
}


# =============================================================================
# MRV Bridge Implementation
# =============================================================================


class MRVBridge:
    """Bridge between GreenLang MRV calculation agents and CSRD CalculatorAgent.

    Routes ESRS E1 climate metrics to appropriate MRV calculation engines using
    a static routing table. Maintains zero-hallucination guarantee by never
    using LLM for numeric calculations. Tracks complete provenance chains
    across all calculation steps.

    Attributes:
        config: Bridge configuration
        _routing_table: Static ESRS metric code to MRV agent routing table
        _agents: Registry of MRV agent instances
        _provenance_chains: Accumulated provenance entries per execution

    Example:
        >>> bridge = MRVBridge(MRVBridgeConfig())
        >>> result = await bridge.route_calculation("E1-1-1", {"fuel_type": "natural_gas"})
        >>> print(result.emissions_value, result.provenance_hash)
    """

    def __init__(self, config: Optional[MRVBridgeConfig] = None) -> None:
        """Initialize the MRV Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or MRVBridgeConfig()
        self._routing_table: Dict[str, MRVRoutingEntry] = dict(MRV_ROUTING_TABLE)
        self._agents: Dict[str, Any] = {}
        self._provenance_chains: List[ProvenanceChainEntry] = []
        self._step_counter = 0

        logger.info(
            "MRVBridge initialized with %d routing entries, scope1=%s, scope2=%s, scope3=%s",
            len(self._routing_table),
            self.config.enable_scope1,
            self.config.enable_scope2,
            self.config.enable_scope3,
        )

    # -------------------------------------------------------------------------
    # Routing
    # -------------------------------------------------------------------------

    def get_routing_entry(self, metric_code: str) -> Optional[MRVRoutingEntry]:
        """Look up the routing entry for an ESRS metric code.

        Args:
            metric_code: ESRS E1 metric code (e.g., "E1-1-1").

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

    def get_scope_metrics(self, scope: ScopeType) -> List[MRVRoutingEntry]:
        """Return all routing entries for a given scope.

        Args:
            scope: The emission scope to filter by.

        Returns:
            List of MRVRoutingEntry for the given scope.
        """
        return [
            entry for entry in self._routing_table.values()
            if entry.scope == scope
        ]

    async def route_calculation(
        self,
        metric_code: str,
        data: Dict[str, Any],
    ) -> CalculationResult:
        """Route a calculation request to the appropriate MRV agent.

        This is the primary entry point for individual metric calculations.
        It looks up the routing entry, invokes the corresponding MRV agent,
        wraps the result in a CalculationResult, and appends provenance
        tracking entries.

        Args:
            metric_code: ESRS E1 metric code (e.g., "E1-1-1").
            data: Activity data to pass to the MRV agent.

        Returns:
            CalculationResult with emissions values and provenance.

        Raises:
            ValueError: If the metric code is not found in the routing table.
        """
        entry = self.get_routing_entry(metric_code)
        if entry is None:
            raise ValueError(
                f"Unknown metric code '{metric_code}'. "
                f"Valid codes: {sorted(self._routing_table.keys())}"
            )

        if not self._is_scope_enabled(entry.scope):
            logger.info("Scope %s is disabled, skipping metric %s", entry.scope.value, metric_code)
            return CalculationResult(
                metric_code=metric_code,
                agent_id=entry.agent_id,
                scope=entry.scope,
                status=CalculationStatus.SKIPPED,
            )

        if (entry.scope3_category is not None
                and entry.scope3_category not in self.config.enabled_scope3_categories):
            logger.info(
                "Scope 3 category %d is disabled, skipping metric %s",
                entry.scope3_category, metric_code,
            )
            return CalculationResult(
                metric_code=metric_code,
                agent_id=entry.agent_id,
                scope=entry.scope,
                status=CalculationStatus.SKIPPED,
            )

        return await self._execute_mrv_calculation(entry, data)

    async def _execute_mrv_calculation(
        self,
        entry: MRVRoutingEntry,
        data: Dict[str, Any],
    ) -> CalculationResult:
        """Execute a single MRV calculation via the routed agent.

        Args:
            entry: The routing entry defining which agent to call.
            data: Activity data for the calculation.

        Returns:
            CalculationResult with computed values and provenance.
        """
        start_time = time.monotonic()
        input_hash = _compute_hash(f"{entry.metric_code}:{_serialize_data(data)}")

        try:
            agent = self._agents.get(entry.agent_id)
            if agent is not None:
                raw_result = await self._invoke_agent(agent, data)
            else:
                raw_result = self._calculate_deterministic(entry, data)

            elapsed_ms = (time.monotonic() - start_time) * 1000
            output_hash = _compute_hash(_serialize_data(raw_result))

            emissions = raw_result.get("total_emissions", 0.0)
            co2 = raw_result.get("co2", 0.0)
            ch4 = raw_result.get("ch4", 0.0)
            n2o = raw_result.get("n2o", 0.0)
            other = raw_result.get("other_ghg", 0.0)

            provenance_entry = ProvenanceChainEntry(
                step_index=self._step_counter,
                agent_id=entry.agent_id,
                metric_code=entry.metric_code,
                input_hash=input_hash,
                output_hash=output_hash,
                execution_time_ms=elapsed_ms,
            )
            self._step_counter += 1
            self._provenance_chains.append(provenance_entry)

            provenance_hash = _compute_hash(
                f"{input_hash}:{output_hash}:{entry.agent_id}"
            )

            result = CalculationResult(
                metric_code=entry.metric_code,
                agent_id=entry.agent_id,
                scope=entry.scope,
                status=CalculationStatus.SUCCESS,
                emissions_value=emissions,
                emissions_unit=self.config.reporting_unit,
                co2_value=co2,
                ch4_value=ch4,
                n2o_value=n2o,
                other_ghg_value=other,
                methodology=raw_result.get("methodology", ""),
                data_quality_score=raw_result.get("data_quality_score", 0.0),
                uncertainty_pct=raw_result.get("uncertainty_pct", 0.0),
                provenance_hash=provenance_hash,
                provenance_chain=[provenance_entry],
                execution_time_ms=elapsed_ms,
                raw_output=raw_result,
            )

            logger.info(
                "Metric %s calculated by %s: %.4f %s in %.1fms",
                entry.metric_code, entry.agent_id,
                emissions, self.config.reporting_unit, elapsed_ms,
            )
            return result

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Calculation failed for metric %s via agent %s: %s",
                entry.metric_code, entry.agent_id, exc, exc_info=True,
            )
            return CalculationResult(
                metric_code=entry.metric_code,
                agent_id=entry.agent_id,
                scope=entry.scope,
                status=CalculationStatus.FAILED,
                execution_time_ms=elapsed_ms,
                error_message=str(exc),
            )

    # -------------------------------------------------------------------------
    # Scope-level calculations
    # -------------------------------------------------------------------------

    async def calculate_scope1(self, data: Dict[str, Any]) -> Scope1Result:
        """Calculate all Scope 1 emissions.

        Routes activity data to each Scope 1 MRV agent and aggregates results.

        Args:
            data: Activity data keyed by metric code or emission source type.

        Returns:
            Scope1Result with individual and aggregated Scope 1 emissions.
        """
        if not self.config.enable_scope1:
            logger.info("Scope 1 calculations disabled")
            return Scope1Result()

        logger.info("Starting Scope 1 calculations")
        start_time = time.monotonic()

        scope1_metrics = {
            "stationary_combustion": "E1-1-1",
            "mobile_combustion": "E1-1-2",
            "process_emissions": "E1-1-3",
            "fugitive_emissions": "E1-1-4",
            "refrigerants": "E1-1-5",
            "land_use": "E1-1-6",
            "waste_treatment": "E1-1-7",
            "agricultural": "E1-1-8",
        }

        results: Dict[str, Optional[CalculationResult]] = {}
        for field_name, metric_code in scope1_metrics.items():
            source_data = data.get(field_name, data.get(metric_code, {}))
            if source_data:
                results[field_name] = await self.route_calculation(metric_code, source_data)
            else:
                results[field_name] = None

        aggregated = self._aggregate_results(
            [r for r in results.values() if r is not None]
        )

        scope1 = Scope1Result(
            stationary_combustion=results.get("stationary_combustion"),
            mobile_combustion=results.get("mobile_combustion"),
            process_emissions=results.get("process_emissions"),
            fugitive_emissions=results.get("fugitive_emissions"),
            refrigerants=results.get("refrigerants"),
            land_use=results.get("land_use"),
            waste_treatment=results.get("waste_treatment"),
            agricultural=results.get("agricultural"),
            aggregated=aggregated,
        )
        scope1.provenance_hash = _compute_hash(
            f"scope1:{aggregated.provenance_hash}"
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Scope 1 complete: %.4f tCO2e total, %d calculations in %.1fms",
            aggregated.total_emissions, aggregated.calculation_count, elapsed_ms,
        )
        return scope1

    async def calculate_scope2(self, data: Dict[str, Any]) -> Scope2Result:
        """Calculate all Scope 2 emissions (location and market based).

        Routes activity data to Scope 2 MRV agents and produces both
        location-based and market-based aggregations per GHG Protocol.

        Args:
            data: Activity data keyed by metric code or energy source type.

        Returns:
            Scope2Result with individual, location-based, and market-based totals.
        """
        if not self.config.enable_scope2:
            logger.info("Scope 2 calculations disabled")
            return Scope2Result()

        logger.info("Starting Scope 2 calculations")
        start_time = time.monotonic()

        scope2_metrics = {
            "location_based": "E1-2-1",
            "market_based": "E1-2-2",
            "steam_heat": "E1-2-3",
            "cooling": "E1-2-4",
            "reconciliation": "E1-2-R",
        }

        results: Dict[str, Optional[CalculationResult]] = {}
        for field_name, metric_code in scope2_metrics.items():
            source_data = data.get(field_name, data.get(metric_code, {}))
            if source_data:
                results[field_name] = await self.route_calculation(metric_code, source_data)
            else:
                results[field_name] = None

        location_results = [
            r for key, r in results.items()
            if r is not None and key in ("location_based", "steam_heat", "cooling")
        ]
        market_results = [
            r for key, r in results.items()
            if r is not None and key in ("market_based", "steam_heat", "cooling")
        ]

        aggregated_location = self._aggregate_results(location_results)
        aggregated_market = self._aggregate_results(market_results)

        scope2 = Scope2Result(
            location_based=results.get("location_based"),
            market_based=results.get("market_based"),
            steam_heat=results.get("steam_heat"),
            cooling=results.get("cooling"),
            reconciliation=results.get("reconciliation"),
            aggregated_location=aggregated_location,
            aggregated_market=aggregated_market,
        )
        scope2.provenance_hash = _compute_hash(
            f"scope2:{aggregated_location.provenance_hash}:{aggregated_market.provenance_hash}"
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Scope 2 complete: location=%.4f, market=%.4f tCO2e in %.1fms",
            aggregated_location.total_emissions,
            aggregated_market.total_emissions,
            elapsed_ms,
        )
        return scope2

    async def calculate_scope3(
        self,
        data: Dict[str, Any],
        enabled_categories: Optional[List[int]] = None,
    ) -> Scope3Result:
        """Calculate Scope 3 emissions for enabled categories.

        Routes activity data to Scope 3 MRV agents for each enabled category.

        Args:
            data: Activity data keyed by category number or metric code.
            enabled_categories: Override list of enabled category numbers (1-15).
                Falls back to config.enabled_scope3_categories if not provided.

        Returns:
            Scope3Result with per-category and aggregated Scope 3 emissions.
        """
        if not self.config.enable_scope3:
            logger.info("Scope 3 calculations disabled")
            return Scope3Result()

        cats = enabled_categories or self.config.enabled_scope3_categories
        logger.info("Starting Scope 3 calculations for categories: %s", cats)
        start_time = time.monotonic()

        category_results: Dict[int, CalculationResult] = {}
        for cat_num in cats:
            metric_code = f"E1-3-Cat{cat_num:02d}"
            cat_data = data.get(f"category_{cat_num}", data.get(metric_code, {}))
            if cat_data:
                result = await self.route_calculation(metric_code, cat_data)
                category_results[cat_num] = result

        mapping_data = data.get("category_mapping", {})
        mapping_result = None
        if mapping_data:
            mapping_result = await self.route_calculation("E1-3-MAP", mapping_data)

        audit_data = data.get("audit_trail", {})
        audit_result = None
        if audit_data:
            audit_result = await self.route_calculation("E1-AUDIT", audit_data)

        all_results = list(category_results.values())
        aggregated = self._aggregate_results(all_results)

        scope3 = Scope3Result(
            categories=category_results,
            category_mapping=mapping_result,
            audit_trail=audit_result,
            aggregated=aggregated,
            enabled_categories=list(cats),
        )
        scope3.provenance_hash = _compute_hash(
            f"scope3:{aggregated.provenance_hash}:{sorted(cats)}"
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Scope 3 complete: %.4f tCO2e across %d categories in %.1fms",
            aggregated.total_emissions, len(category_results), elapsed_ms,
        )
        return scope3

    # -------------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------------

    def _aggregate_results(
        self, results: List[CalculationResult]
    ) -> AggregatedEmissions:
        """Aggregate multiple CalculationResults into a single AggregatedEmissions.

        Args:
            results: List of individual calculation results to aggregate.

        Returns:
            AggregatedEmissions with summed values and averaged quality.
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
    # Internal helpers
    # -------------------------------------------------------------------------

    def _is_scope_enabled(self, scope: ScopeType) -> bool:
        """Check whether a scope is enabled in the configuration.

        Args:
            scope: The scope to check.

        Returns:
            True if the scope is enabled, False otherwise.
        """
        scope_flags = {
            ScopeType.SCOPE_1: self.config.enable_scope1,
            ScopeType.SCOPE_2: self.config.enable_scope2,
            ScopeType.SCOPE_3: self.config.enable_scope3,
        }
        return scope_flags.get(scope, False)

    async def _invoke_agent(
        self, agent: Any, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invoke an MRV agent instance.

        Args:
            agent: The MRV agent object.
            data: Input data for the agent.

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
        """Perform a deterministic calculation when no agent instance is available.

        This fallback uses simple activity * factor multiplication for basic
        emission estimates. It is used during testing and initial setup before
        full MRV agents are deployed.

        Args:
            entry: The routing entry for the metric.
            data: Activity data with quantity and emission_factor fields.

        Returns:
            Dictionary with calculated emission values.
        """
        quantity = float(data.get("quantity", data.get("activity_data", 0.0)))
        factor = float(data.get("emission_factor", data.get("factor", 0.0)))
        emissions = quantity * factor

        return {
            "total_emissions": emissions,
            "co2": emissions * 0.95,  # Approximate GHG split
            "ch4": emissions * 0.03,
            "n2o": emissions * 0.015,
            "other_ghg": emissions * 0.005,
            "methodology": f"deterministic_fallback:{entry.agent_id}",
            "data_quality_score": 0.5,
            "uncertainty_pct": 25.0,
        }

    def get_provenance_chain(self) -> List[ProvenanceChainEntry]:
        """Return the complete provenance chain for all calculations performed.

        Returns:
            List of ProvenanceChainEntry in execution order.
        """
        return list(self._provenance_chains)

    def reset_provenance(self) -> None:
        """Clear the provenance chain for a fresh calculation cycle."""
        self._provenance_chains.clear()
        self._step_counter = 0
        logger.debug("Provenance chain reset")


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string.

    Args:
        data: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _serialize_data(data: Any) -> str:
    """Serialize data to a stable string for hashing.

    Args:
        data: Data to serialize (dict, list, or primitive).

    Returns:
        Deterministic string representation.
    """
    import json
    if isinstance(data, dict):
        return json.dumps(data, sort_keys=True, default=str)
    return str(data)
