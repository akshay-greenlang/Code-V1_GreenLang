# -*- coding: utf-8 -*-
"""
Supply Chain Deep Mapping Workflow
====================================

Five-phase supply chain mapping workflow for multi-tier traceability and
origin identification.

This workflow enables comprehensive supply chain visibility through:
- Tier 1 supplier data collection
- Tier 2+ expansion via supplier questionnaires
- Origin tracing to farm/forest level
- Network analysis for bottlenecks and risks
- Diversification planning for resilience

Phases:
    1. Tier 1 Collection - Gather direct supplier relationships
    2. Tier Expansion - Extend mapping to Tier 2, 3, N suppliers
    3. Origin Tracing - Trace commodities to production origins
    4. Network Analysis - Analyze supply chain topology and dependencies
    5. Diversification Planning - Identify single points of failure

Regulatory Context:
    EUDR Article 9(1) requires "information on geolocation of all plots of land"
    where relevant commodities were produced. Deep mapping enables operators to
    fulfill this requirement even for complex, multi-tier supply chains.

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import random
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class Phase(str, Enum):
    """Workflow phases."""
    TIER1_COLLECTION = "tier1_collection"
    TIER_EXPANSION = "tier_expansion"
    ORIGIN_TRACING = "origin_tracing"
    NETWORK_ANALYSIS = "network_analysis"
    DIVERSIFICATION_PLANNING = "diversification_planning"


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class NodeType(str, Enum):
    """Supply chain node types."""
    OPERATOR = "operator"
    TIER1_SUPPLIER = "tier1_supplier"
    TIER2_SUPPLIER = "tier2_supplier"
    TIER3_SUPPLIER = "tier3_supplier"
    PRODUCER = "producer"
    PLOT = "plot"


# =============================================================================
# DATA MODELS
# =============================================================================


class SupplyChainDeepMappingConfig(BaseModel):
    """Configuration for supply chain deep mapping workflow."""
    max_tier_depth: int = Field(default=3, ge=1, le=5, description="Maximum tier depth")
    trace_to_origin: bool = Field(default=True, description="Trace to farm/plot level")
    min_network_completeness: float = Field(default=0.80, ge=0.0, le=1.0, description="Target completeness")
    identify_single_points_of_failure: bool = Field(default=True, description="SPOF analysis")
    operator_id: Optional[str] = Field(None, description="Operator context")


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase: Phase = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Execution duration")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


class WorkflowContext(BaseModel):
    """Shared context passed between workflow phases."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique execution ID")
    config: SupplyChainDeepMappingConfig = Field(default_factory=SupplyChainDeepMappingConfig)
    phase_results: List[PhaseResult] = Field(default_factory=list, description="Completed phase results")
    state: Dict[str, Any] = Field(default_factory=dict, description="Shared state data")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Workflow start time")

    class Config:
        arbitrary_types_allowed = True


class WorkflowResult(BaseModel):
    """Complete result from the supply chain deep mapping workflow."""
    workflow_name: str = Field(default="supply_chain_deep_mapping", description="Workflow identifier")
    phases: List[PhaseResult] = Field(default_factory=list, description="All phase results")
    overall_status: PhaseStatus = Field(..., description="Overall workflow status")
    total_duration_seconds: float = Field(default=0.0, ge=0.0, description="Total execution time")
    provenance_hash: str = Field(default="", description="Workflow-level provenance hash")
    execution_id: str = Field(..., description="Execution identifier")
    total_nodes: int = Field(default=0, ge=0, description="Total supply chain nodes")
    total_edges: int = Field(default=0, ge=0, description="Total relationships")
    tier_depth_reached: int = Field(default=0, ge=0, description="Maximum tier depth")
    network_completeness: float = Field(default=0.0, ge=0.0, le=1.0, description="Mapping completeness")
    single_points_of_failure: List[str] = Field(default_factory=list, description="Critical nodes")
    diversification_recommendations: List[str] = Field(default_factory=list, description="Actions")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


# =============================================================================
# SUPPLY CHAIN DEEP MAPPING WORKFLOW
# =============================================================================


class SupplyChainDeepMappingWorkflow:
    """
    Five-phase supply chain deep mapping workflow.

    Provides comprehensive multi-tier supply chain visibility through:
    - Automated tier expansion via supplier questionnaires
    - Origin tracing to farm/plot level
    - Network topology analysis
    - Risk concentration identification
    - Diversification strategy generation

    Example:
        >>> config = SupplyChainDeepMappingConfig(
        ...     max_tier_depth=3,
        ...     trace_to_origin=True,
        ... )
        >>> workflow = SupplyChainDeepMappingWorkflow(config)
        >>> result = await workflow.run(WorkflowContext(config=config))
        >>> assert result.network_completeness >= 0.8
    """

    def __init__(self, config: Optional[SupplyChainDeepMappingConfig] = None) -> None:
        """Initialize the supply chain deep mapping workflow."""
        self.config = config or SupplyChainDeepMappingConfig()
        self.logger = logging.getLogger(f"{__name__}.SupplyChainDeepMappingWorkflow")

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """
        Execute the full 5-phase supply chain deep mapping workflow.

        Args:
            context: Workflow context with configuration and initial state.

        Returns:
            WorkflowResult with network map, completeness, and diversification plan.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting supply chain deep mapping workflow execution_id=%s max_depth=%d",
            context.execution_id,
            self.config.max_tier_depth,
        )

        context.config = self.config

        phase_handlers = [
            (Phase.TIER1_COLLECTION, self._phase_1_tier1_collection),
            (Phase.TIER_EXPANSION, self._phase_2_tier_expansion),
            (Phase.ORIGIN_TRACING, self._phase_3_origin_tracing),
            (Phase.NETWORK_ANALYSIS, self._phase_4_network_analysis),
            (Phase.DIVERSIFICATION_PLANNING, self._phase_5_diversification_planning),
        ]

        overall_status = PhaseStatus.COMPLETED

        for phase, handler in phase_handlers:
            phase_start = datetime.utcnow()
            self.logger.info("Starting phase: %s", phase.value)

            try:
                phase_result = await handler(context)
                phase_result.duration_seconds = (datetime.utcnow() - phase_start).total_seconds()
                phase_result.timestamp = datetime.utcnow()
            except Exception as exc:
                self.logger.error("Phase '%s' failed: %s", phase.value, exc, exc_info=True)
                phase_result = PhaseResult(
                    phase=phase,
                    status=PhaseStatus.FAILED,
                    data={"error": str(exc)},
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    provenance_hash=self._hash({"error": str(exc)}),
                    timestamp=datetime.utcnow(),
                )

            context.phase_results.append(phase_result)

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                self.logger.error("Phase '%s' failed; halting workflow.", phase.value)
                break

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        # Extract final outputs
        network = context.state.get("network", {})
        nodes = network.get("nodes", [])
        edges = network.get("edges", [])
        tier_depth = context.state.get("tier_depth_reached", 0)
        completeness = context.state.get("network_completeness", 0.0)
        spof = context.state.get("single_points_of_failure", [])
        recommendations = context.state.get("diversification_recommendations", [])

        provenance = self._hash({
            "execution_id": context.execution_id,
            "phases": [p.provenance_hash for p in context.phase_results],
            "max_depth": self.config.max_tier_depth,
        })

        self.logger.info(
            "Supply chain deep mapping finished execution_id=%s status=%s "
            "nodes=%d edges=%d depth=%d completeness=%.1f%%",
            context.execution_id,
            overall_status.value,
            len(nodes),
            len(edges),
            tier_depth,
            completeness * 100,
        )

        return WorkflowResult(
            phases=context.phase_results,
            overall_status=overall_status,
            total_duration_seconds=total_duration,
            provenance_hash=provenance,
            execution_id=context.execution_id,
            total_nodes=len(nodes),
            total_edges=len(edges),
            tier_depth_reached=tier_depth,
            network_completeness=completeness,
            single_points_of_failure=spof,
            diversification_recommendations=recommendations,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Tier 1 Collection
    # -------------------------------------------------------------------------

    async def _phase_1_tier1_collection(self, context: WorkflowContext) -> PhaseResult:
        """
        Gather direct (Tier 1) supplier relationships.

        Collects:
        - Supplier profiles (name, country, commodity)
        - Contract/purchase order data
        - Volume/value of transactions
        - Delivery/sourcing locations
        """
        phase = Phase.TIER1_COLLECTION
        self.logger.info("Collecting Tier 1 supplier data")

        await asyncio.sleep(0.05)

        # Simulate Tier 1 collection
        tier1_count = random.randint(10, 50)
        tier1_suppliers = []

        for i in range(tier1_count):
            supplier = {
                "node_id": f"T1-{uuid.uuid4().hex[:8]}",
                "node_type": NodeType.TIER1_SUPPLIER.value,
                "supplier_name": f"Tier1 Supplier {i+1}",
                "country": random.choice(["BR", "ID", "CO", "PE", "MY"]),
                "commodity": random.choice(["cocoa", "coffee", "oil_palm", "soya"]),
                "annual_volume_mt": random.randint(100, 10000),
            }
            tier1_suppliers.append(supplier)

        # Initialize network
        network = {
            "nodes": [
                {
                    "node_id": "OPERATOR-001",
                    "node_type": NodeType.OPERATOR.value,
                    "name": "Primary Operator",
                }
            ] + tier1_suppliers,
            "edges": [
                {
                    "edge_id": f"E-{i}",
                    "source": "OPERATOR-001",
                    "target": tier1_suppliers[i]["node_id"],
                    "relationship": "purchases_from",
                }
                for i in range(len(tier1_suppliers))
            ],
        }

        context.state["network"] = network
        context.state["tier_depth_reached"] = 1

        provenance = self._hash({
            "phase": phase.value,
            "tier1_count": len(tier1_suppliers),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "tier1_suppliers": len(tier1_suppliers),
                "nodes_created": len(tier1_suppliers) + 1,
                "edges_created": len(tier1_suppliers),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Tier Expansion
    # -------------------------------------------------------------------------

    async def _phase_2_tier_expansion(self, context: WorkflowContext) -> PhaseResult:
        """
        Extend mapping to Tier 2, 3, N suppliers.

        For each tier N supplier, request:
        - List of tier N+1 suppliers
        - Volume/value breakdowns
        - Geographic sources

        Continue until max_tier_depth reached or no new suppliers.
        """
        phase = Phase.TIER_EXPANSION
        network = context.state.get("network", {})
        max_depth = self.config.max_tier_depth

        self.logger.info("Expanding supply chain to tier depth %d", max_depth)

        current_depth = 1
        nodes = list(network["nodes"])
        edges = list(network["edges"])

        while current_depth < max_depth:
            current_tier_nodes = [
                n for n in nodes
                if n["node_type"] == f"tier{current_depth}_supplier"
            ]

            if not current_tier_nodes:
                break

            # Expand each current tier node
            new_tier_nodes = []
            new_edges = []

            for parent_node in current_tier_nodes:
                # Simulate upstream expansion (some suppliers have upstream, some don't)
                if random.random() < 0.7:
                    upstream_count = random.randint(1, 5)
                    for j in range(upstream_count):
                        upstream_node = {
                            "node_id": f"T{current_depth+1}-{uuid.uuid4().hex[:8]}",
                            "node_type": f"tier{current_depth+1}_supplier",
                            "supplier_name": f"Tier{current_depth+1} Supplier",
                            "country": random.choice(["BR", "ID", "CO", "PE", "MY"]),
                            "commodity": parent_node.get("commodity", "unknown"),
                        }
                        new_tier_nodes.append(upstream_node)

                        edge = {
                            "edge_id": f"E-{len(edges) + len(new_edges)}",
                            "source": parent_node["node_id"],
                            "target": upstream_node["node_id"],
                            "relationship": "sources_from",
                        }
                        new_edges.append(edge)

            if not new_tier_nodes:
                break

            nodes.extend(new_tier_nodes)
            edges.extend(new_edges)
            current_depth += 1

        network["nodes"] = nodes
        network["edges"] = edges
        context.state["network"] = network
        context.state["tier_depth_reached"] = current_depth

        provenance = self._hash({
            "phase": phase.value,
            "tier_depth": current_depth,
            "total_nodes": len(nodes),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "tier_depth_reached": current_depth,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Origin Tracing
    # -------------------------------------------------------------------------

    async def _phase_3_origin_tracing(self, context: WorkflowContext) -> PhaseResult:
        """
        Trace commodities to production origins (farms/plots).

        For deepest tier suppliers:
        - Request producer/farm information
        - Collect plot-level geolocation
        - Link plots to commodities
        """
        phase = Phase.ORIGIN_TRACING
        network = context.state.get("network", {})

        if not self.config.trace_to_origin:
            self.logger.info("Origin tracing disabled; skipping")
            return PhaseResult(
                phase=phase,
                status=PhaseStatus.COMPLETED,
                data={"origin_tracing": False},
                provenance_hash=self._hash({"phase": phase.value, "skipped": True}),
            )

        self.logger.info("Tracing supply chain to production origins")

        nodes = list(network["nodes"])
        edges = list(network["edges"])

        # Find deepest tier suppliers
        tier_depth = context.state.get("tier_depth_reached", 1)
        deepest_suppliers = [
            n for n in nodes
            if n["node_type"] == f"tier{tier_depth}_supplier"
        ]

        # Create producer and plot nodes
        plot_count = 0
        for supplier in deepest_suppliers:
            # Each supplier sources from 1-3 producers
            producer_count = random.randint(1, 3)
            for i in range(producer_count):
                producer_node = {
                    "node_id": f"PROD-{uuid.uuid4().hex[:8]}",
                    "node_type": NodeType.PRODUCER.value,
                    "producer_name": f"Producer {uuid.uuid4().hex[:4]}",
                    "country": supplier.get("country", "XX"),
                }
                nodes.append(producer_node)

                edges.append({
                    "edge_id": f"E-{len(edges)}",
                    "source": supplier["node_id"],
                    "target": producer_node["node_id"],
                    "relationship": "sources_from",
                })

                # Each producer has 1-5 plots
                plots_per_producer = random.randint(1, 5)
                for j in range(plots_per_producer):
                    plot_node = {
                        "node_id": f"PLOT-{uuid.uuid4().hex[:8]}",
                        "node_type": NodeType.PLOT.value,
                        "area_hectares": random.uniform(1, 100),
                        "latitude": random.uniform(-30, 10),
                        "longitude": random.uniform(-80, 120),
                    }
                    nodes.append(plot_node)
                    plot_count += 1

                    edges.append({
                        "edge_id": f"E-{len(edges)}",
                        "source": producer_node["node_id"],
                        "target": plot_node["node_id"],
                        "relationship": "owns",
                    })

        network["nodes"] = nodes
        network["edges"] = edges
        context.state["network"] = network

        provenance = self._hash({
            "phase": phase.value,
            "plot_count": plot_count,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "origin_tracing": True,
                "plots_identified": plot_count,
                "total_nodes": len(nodes),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Network Analysis
    # -------------------------------------------------------------------------

    async def _phase_4_network_analysis(self, context: WorkflowContext) -> PhaseResult:
        """
        Analyze supply chain topology and dependencies.

        Analysis:
        - Network completeness (% of expected relationships mapped)
        - Centrality metrics (identify critical nodes)
        - Cluster analysis (geographic/commodity concentrations)
        - Path analysis (average chain length)
        """
        phase = Phase.NETWORK_ANALYSIS
        network = context.state.get("network", {})
        nodes = network.get("nodes", [])
        edges = network.get("edges", [])

        self.logger.info("Analyzing network topology (%d nodes, %d edges)", len(nodes), len(edges))

        # Calculate network completeness
        expected_edges = len(nodes) * 2  # Heuristic: expect ~2 edges per node
        completeness = min(1.0, len(edges) / max(expected_edges, 1))

        # Identify high-degree nodes (potential single points of failure)
        node_degrees = {}
        for edge in edges:
            source = edge["source"]
            node_degrees[source] = node_degrees.get(source, 0) + 1

        # Nodes with degree > 10 are potential bottlenecks
        critical_nodes = [
            node_id for node_id, degree in node_degrees.items()
            if degree > 10
        ]

        # Calculate average path length (simplified: tier depth)
        tier_depth = context.state.get("tier_depth_reached", 1)
        avg_path_length = tier_depth + 1  # Operator -> Tier1 -> ... -> Plot

        context.state["network_completeness"] = round(completeness, 3)
        context.state["critical_nodes"] = critical_nodes
        context.state["avg_path_length"] = avg_path_length

        provenance = self._hash({
            "phase": phase.value,
            "completeness": completeness,
            "critical_nodes": len(critical_nodes),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "network_completeness": round(completeness, 3),
                "critical_nodes": len(critical_nodes),
                "avg_path_length": avg_path_length,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 5: Diversification Planning
    # -------------------------------------------------------------------------

    async def _phase_5_diversification_planning(self, context: WorkflowContext) -> PhaseResult:
        """
        Identify single points of failure and generate diversification plan.

        Actions:
        - Flag suppliers with >X% of volume (concentration risk)
        - Identify geographies with >Y% of sourcing (country risk)
        - Recommend alternative suppliers/regions
        - Calculate diversification targets
        """
        phase = Phase.DIVERSIFICATION_PLANNING
        critical_nodes = context.state.get("critical_nodes", [])
        network = context.state.get("network", {})
        nodes = network.get("nodes", [])

        self.logger.info("Planning supply chain diversification")

        recommendations = []
        spof = []

        # Identify single points of failure
        if self.config.identify_single_points_of_failure and critical_nodes:
            spof = critical_nodes[:5]  # Top 5 critical nodes
            recommendations.append(
                f"Identified {len(spof)} single points of failure. "
                "Develop backup supplier relationships to reduce dependency."
            )

        # Geographic concentration analysis
        country_counts = {}
        for node in nodes:
            if "country" in node:
                country = node["country"]
                country_counts[country] = country_counts.get(country, 0) + 1

        total_suppliers = len([n for n in nodes if "supplier" in n.get("node_type", "")])
        for country, count in country_counts.items():
            if total_suppliers > 0 and count / total_suppliers > 0.4:
                recommendations.append(
                    f"High concentration in {country} ({count}/{total_suppliers} suppliers, "
                    f"{count/total_suppliers*100:.0f}%). Target: reduce to <30% within 12 months."
                )

        # Commodity concentration
        commodity_counts = {}
        for node in nodes:
            if "commodity" in node:
                commodity = node["commodity"]
                commodity_counts[commodity] = commodity_counts.get(commodity, 0) + 1

        if commodity_counts:
            top_commodity = max(commodity_counts, key=commodity_counts.get)
            if commodity_counts[top_commodity] / total_suppliers > 0.5:
                recommendations.append(
                    f"High commodity concentration in {top_commodity}. "
                    "Diversify commodity portfolio to reduce market risk."
                )

        # Network completeness recommendations
        completeness = context.state.get("network_completeness", 0.0)
        if completeness < self.config.min_network_completeness:
            recommendations.append(
                f"Network completeness is {completeness*100:.0f}% "
                f"(target: {self.config.min_network_completeness*100:.0f}%). "
                "Expand mapping to improve supply chain visibility."
            )

        if not recommendations:
            recommendations.append(
                "Supply chain diversity is adequate. Continue monitoring for changes."
            )

        context.state["single_points_of_failure"] = spof
        context.state["diversification_recommendations"] = recommendations

        provenance = self._hash({
            "phase": phase.value,
            "spof_count": len(spof),
            "recommendations": len(recommendations),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "single_points_of_failure": len(spof),
                "recommendations": len(recommendations),
            },
            provenance_hash=provenance,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
