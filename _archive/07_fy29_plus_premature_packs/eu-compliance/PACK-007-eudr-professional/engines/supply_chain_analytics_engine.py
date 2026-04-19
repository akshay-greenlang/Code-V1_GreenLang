"""
SupplyChainAnalyticsEngine - Deep supply chain analysis with multi-tier mapping for EUDR

This module implements advanced supply chain analytics for PACK-007 EUDR Professional Pack.
Provides multi-tier supply chain mapping, critical node identification, concentration risk
analysis, mass balance tracking, and scenario planning per EU Regulation 2023/1115.

Example:
    >>> config = SupplyChainConfig(max_tier_depth=5)
    >>> engine = SupplyChainAnalyticsEngine(config)
    >>> network = engine.map_supply_chain("SUPPLIER-001", depth=3)
    >>> critical_nodes = engine.identify_critical_nodes(network)
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field, validator
import hashlib
import logging
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ChainOfCustodyModel(str, Enum):
    """Chain of custody models for EUDR."""
    IDENTITY_PRESERVED = "IDENTITY_PRESERVED"
    SEGREGATED = "SEGREGATED"
    MASS_BALANCE = "MASS_BALANCE"
    BOOK_AND_CLAIM = "BOOK_AND_CLAIM"


class CommodityType(str, Enum):
    """EUDR commodity types."""
    TIMBER = "TIMBER"
    PALM_OIL = "PALM_OIL"
    CATTLE = "CATTLE"
    SOY = "SOY"
    COCOA = "COCOA"
    COFFEE = "COFFEE"
    RUBBER = "RUBBER"


class SupplyChainConfig(BaseModel):
    """Configuration for supply chain analytics engine."""

    max_tier_depth: int = Field(5, ge=1, le=10, description="Maximum supply chain tier depth")
    coc_models: List[ChainOfCustodyModel] = Field(
        default=[ChainOfCustodyModel.SEGREGATED, ChainOfCustodyModel.MASS_BALANCE],
        description="Accepted chain of custody models"
    )
    network_analysis: bool = Field(True, description="Enable network analysis")
    concentration_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Concentration risk threshold")
    critical_node_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Critical node importance threshold")
    mass_balance_tolerance: float = Field(0.05, ge=0.0, le=0.2, description="Mass balance tolerance (5%)")


class SupplyChainNode(BaseModel):
    """Node in supply chain network."""

    node_id: str = Field(..., description="Unique node identifier")
    name: str = Field(..., description="Supplier/entity name")
    tier: int = Field(..., ge=0, description="Supply chain tier (0=operator, 1=tier-1, etc.)")
    country: str = Field(..., description="ISO3 country code")
    commodity: CommodityType = Field(..., description="Commodity type")
    annual_volume_tonnes: float = Field(..., ge=0.0, description="Annual volume in tonnes")
    suppliers: List[str] = Field(default_factory=list, description="Supplier node IDs (upstream)")
    customers: List[str] = Field(default_factory=list, description="Customer node IDs (downstream)")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="EUDR risk score")
    coc_model: ChainOfCustodyModel = Field(..., description="Chain of custody model used")
    verification_status: str = Field(..., description="Verification status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional node metadata")


class SupplyChainEdge(BaseModel):
    """Edge (connection) in supply chain network."""

    edge_id: str = Field(..., description="Unique edge identifier")
    source_node: str = Field(..., description="Source node ID (supplier)")
    target_node: str = Field(..., description="Target node ID (customer)")
    annual_flow_tonnes: float = Field(..., ge=0.0, description="Annual commodity flow in tonnes")
    coc_model: ChainOfCustodyModel = Field(..., description="Chain of custody model")
    traceability_score: float = Field(..., ge=0.0, le=1.0, description="Traceability score")


class SupplyChainNetwork(BaseModel):
    """Complete supply chain network."""

    network_id: str = Field(..., description="Network identifier")
    root_node_id: str = Field(..., description="Root node (operator)")
    nodes: List[SupplyChainNode] = Field(..., description="All nodes in network")
    edges: List[SupplyChainEdge] = Field(..., description="All edges in network")
    total_tiers: int = Field(..., ge=1, description="Total number of tiers")
    total_volume_tonnes: float = Field(..., ge=0.0, description="Total network volume")
    avg_risk_score: float = Field(..., ge=0.0, le=1.0, description="Network average risk score")
    traceability_completeness: float = Field(..., ge=0.0, le=1.0, description="Network traceability completeness")
    creation_timestamp: datetime = Field(..., description="Network mapping timestamp")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class CriticalNode(BaseModel):
    """Critical node in supply chain."""

    node_id: str = Field(..., description="Node identifier")
    node_name: str = Field(..., description="Node name")
    criticality_score: float = Field(..., ge=0.0, le=1.0, description="Criticality score")
    importance_type: str = Field(..., description="Type of criticality (VOLUME, BOTTLENECK, RISK)")
    volume_percentage: float = Field(..., ge=0.0, le=1.0, description="Percentage of total volume")
    downstream_dependents: int = Field(..., ge=0, description="Number of downstream dependents")
    alternative_suppliers: int = Field(..., ge=0, description="Number of alternative suppliers available")
    mitigation_priority: str = Field(..., description="Mitigation priority (CRITICAL, HIGH, MEDIUM, LOW)")


class ConcentrationRisk(BaseModel):
    """Supply chain concentration risk analysis."""

    herfindahl_index: float = Field(..., ge=0.0, le=1.0, description="Herfindahl-Hirschman Index")
    top_3_concentration: float = Field(..., ge=0.0, le=1.0, description="Top 3 suppliers concentration")
    top_5_concentration: float = Field(..., ge=0.0, le=1.0, description="Top 5 suppliers concentration")
    geographic_concentration: Dict[str, float] = Field(..., description="Concentration by country")
    risk_level: str = Field(..., description="Concentration risk level (LOW, MEDIUM, HIGH, CRITICAL)")
    diversification_needed: bool = Field(..., description="Whether diversification is needed")
    recommendation: str = Field(..., description="Diversification recommendation")


class OriginTrace(BaseModel):
    """Trace of product origin through supply chain."""

    product_id: str = Field(..., description="Product identifier")
    origin_plots: List[str] = Field(..., description="Origin plot IDs")
    supply_chain_path: List[str] = Field(..., description="Node IDs from origin to operator")
    total_hops: int = Field(..., ge=0, description="Number of supply chain hops")
    traceability_confidence: float = Field(..., ge=0.0, le=1.0, description="Trace confidence score")
    coc_model: ChainOfCustodyModel = Field(..., description="Chain of custody model")
    mass_balance_verified: bool = Field(False, description="Whether mass balance is verified")


class MassBalanceResult(BaseModel):
    """Mass balance verification result."""

    node_id: str = Field(..., description="Node identifier")
    input_volume_tonnes: float = Field(..., ge=0.0, description="Total input volume")
    output_volume_tonnes: float = Field(..., ge=0.0, description="Total output volume")
    conversion_rate: float = Field(..., ge=0.0, le=1.0, description="Conversion/yield rate")
    expected_output: float = Field(..., ge=0.0, description="Expected output based on conversion")
    actual_output: float = Field(..., ge=0.0, description="Actual output")
    variance_percentage: float = Field(..., description="Variance percentage")
    is_balanced: bool = Field(..., description="Whether mass balance passes tolerance")
    discrepancies: List[str] = Field(default_factory=list, description="Identified discrepancies")


class DiversificationScore(BaseModel):
    """Supply chain diversification assessment."""

    supplier_diversity_index: float = Field(..., ge=0.0, le=1.0, description="Supplier diversity index")
    geographic_diversity_index: float = Field(..., ge=0.0, le=1.0, description="Geographic diversity index")
    commodity_diversity_index: float = Field(..., ge=0.0, le=1.0, description="Commodity diversity index")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall diversification score")
    risk_level: str = Field(..., description="Diversification risk level")
    improvement_opportunities: List[str] = Field(..., description="Diversification opportunities")


class PropagationResult(BaseModel):
    """Risk propagation analysis result."""

    shock_node_id: str = Field(..., description="Node where shock originates")
    shock_type: str = Field(..., description="Type of shock (SUPPLY_DISRUPTION, QUALITY_ISSUE, etc.)")
    affected_nodes: List[str] = Field(..., description="Node IDs affected by shock")
    propagation_depth: int = Field(..., ge=0, description="Number of tiers affected")
    volume_impact_tonnes: float = Field(..., ge=0.0, description="Total volume impacted")
    volume_impact_percentage: float = Field(..., ge=0.0, le=1.0, description="Percentage of network impacted")
    recovery_time_days: int = Field(..., ge=0, description="Estimated recovery time")
    mitigation_actions: List[str] = Field(..., description="Recommended mitigation actions")


class AlternativeSupplier(BaseModel):
    """Alternative supplier recommendation."""

    supplier_id: str = Field(..., description="Alternative supplier identifier")
    supplier_name: str = Field(..., description="Supplier name")
    country: str = Field(..., description="ISO3 country code")
    capacity_tonnes: float = Field(..., ge=0.0, description="Available capacity")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score")
    compatibility_score: float = Field(..., ge=0.0, le=1.0, description="Compatibility with requirements")
    cost_differential: float = Field(..., description="Cost differential vs current supplier (%)")
    lead_time_days: int = Field(..., ge=0, description="Lead time in days")


class ScenarioChange(BaseModel):
    """Change to apply in scenario planning."""

    change_type: str = Field(..., description="Type of change (ADD_SUPPLIER, REMOVE_SUPPLIER, etc.)")
    target_node_id: Optional[str] = Field(None, description="Target node ID")
    new_volume_tonnes: Optional[float] = Field(None, ge=0.0, description="New volume for node")
    new_supplier_id: Optional[str] = Field(None, description="New supplier to add")


class ScenarioResult(BaseModel):
    """Scenario planning result."""

    scenario_id: str = Field(..., description="Scenario identifier")
    description: str = Field(..., description="Scenario description")
    changes_applied: List[ScenarioChange] = Field(..., description="Changes applied")
    baseline_risk: float = Field(..., ge=0.0, le=1.0, description="Baseline network risk")
    scenario_risk: float = Field(..., ge=0.0, le=1.0, description="Scenario network risk")
    risk_change: float = Field(..., description="Risk change (positive = increase)")
    baseline_concentration: float = Field(..., ge=0.0, le=1.0, description="Baseline concentration")
    scenario_concentration: float = Field(..., ge=0.0, le=1.0, description="Scenario concentration")
    feasibility_score: float = Field(..., ge=0.0, le=1.0, description="Scenario feasibility")
    recommendation: str = Field(..., description="Scenario recommendation")


class NetworkGraph(BaseModel):
    """Network graph visualization data."""

    graph_id: str = Field(..., description="Graph identifier")
    graph_type: str = Field(..., description="Graph type (FULL, TIER, CRITICAL_PATH)")
    nodes_data: List[Dict[str, Any]] = Field(..., description="Node visualization data")
    edges_data: List[Dict[str, Any]] = Field(..., description="Edge visualization data")
    layout_algorithm: str = Field("HIERARCHICAL", description="Layout algorithm used")
    color_scheme: str = Field("RISK_BASED", description="Node coloring scheme")


class SupplyChainAnalyticsEngine:
    """
    Advanced supply chain analytics engine for EUDR compliance.

    Implements multi-tier supply chain mapping, network analysis, concentration risk
    assessment, mass balance verification, and scenario planning.

    Attributes:
        config: Engine configuration
        network_cache: Cached supply chain networks

    Example:
        >>> config = SupplyChainConfig(max_tier_depth=5)
        >>> engine = SupplyChainAnalyticsEngine(config)
        >>> network = engine.map_supply_chain("ROOT-001", depth=3)
        >>> print(f"Mapped {len(network.nodes)} nodes across {network.total_tiers} tiers")
    """

    def __init__(self, config: SupplyChainConfig):
        """Initialize supply chain analytics engine."""
        self.config = config
        self.network_cache: Dict[str, SupplyChainNetwork] = {}
        self._initialize_sample_data()
        logger.info(f"SupplyChainAnalyticsEngine initialized with max_depth={config.max_tier_depth}")

    def _initialize_sample_data(self):
        """Initialize sample supply chain data for demonstration."""
        # Sample nodes (in production, query from database)
        self.sample_nodes = {
            "OP-001": SupplyChainNode(
                node_id="OP-001", name="EU Timber Importer", tier=0, country="DEU",
                commodity=CommodityType.TIMBER, annual_volume_tonnes=50000,
                suppliers=["T1-001", "T1-002"], customers=[], risk_score=0.15,
                coc_model=ChainOfCustodyModel.SEGREGATED, verification_status="VERIFIED"
            ),
            "T1-001": SupplyChainNode(
                node_id="T1-001", name="Primary Processor A", tier=1, country="MYS",
                commodity=CommodityType.TIMBER, annual_volume_tonnes=30000,
                suppliers=["T2-001", "T2-002"], customers=["OP-001"], risk_score=0.35,
                coc_model=ChainOfCustodyModel.SEGREGATED, verification_status="VERIFIED"
            ),
            "T1-002": SupplyChainNode(
                node_id="T1-002", name="Primary Processor B", tier=1, country="IDN",
                commodity=CommodityType.TIMBER, annual_volume_tonnes=20000,
                suppliers=["T2-003"], customers=["OP-001"], risk_score=0.52,
                coc_model=ChainOfCustodyModel.MASS_BALANCE, verification_status="PENDING"
            ),
            "T2-001": SupplyChainNode(
                node_id="T2-001", name="Logging Company A", tier=2, country="MYS",
                commodity=CommodityType.TIMBER, annual_volume_tonnes=18000,
                suppliers=["T3-001", "T3-002"], customers=["T1-001"], risk_score=0.42,
                coc_model=ChainOfCustodyModel.SEGREGATED, verification_status="VERIFIED"
            ),
            "T2-002": SupplyChainNode(
                node_id="T2-002", name="Logging Company B", tier=2, country="MYS",
                commodity=CommodityType.TIMBER, annual_volume_tonnes=12000,
                suppliers=["T3-003"], customers=["T1-001"], risk_score=0.28,
                coc_model=ChainOfCustodyModel.SEGREGATED, verification_status="VERIFIED"
            ),
            "T2-003": SupplyChainNode(
                node_id="T2-003", name="Logging Company C", tier=2, country="IDN",
                commodity=CommodityType.TIMBER, annual_volume_tonnes=20000,
                suppliers=["T3-004"], customers=["T1-002"], risk_score=0.68,
                coc_model=ChainOfCustodyModel.MASS_BALANCE, verification_status="PENDING"
            ),
            "T3-001": SupplyChainNode(
                node_id="T3-001", name="Forest Concession A", tier=3, country="MYS",
                commodity=CommodityType.TIMBER, annual_volume_tonnes=10000,
                suppliers=[], customers=["T2-001"], risk_score=0.38,
                coc_model=ChainOfCustodyModel.IDENTITY_PRESERVED, verification_status="VERIFIED"
            ),
            "T3-002": SupplyChainNode(
                node_id="T3-002", name="Forest Concession B", tier=3, country="MYS",
                commodity=CommodityType.TIMBER, annual_volume_tonnes=8000,
                suppliers=[], customers=["T2-001"], risk_score=0.45,
                coc_model=ChainOfCustodyModel.SEGREGATED, verification_status="VERIFIED"
            ),
            "T3-003": SupplyChainNode(
                node_id="T3-003", name="Forest Concession C", tier=3, country="MYS",
                commodity=CommodityType.TIMBER, annual_volume_tonnes=12000,
                suppliers=[], customers=["T2-002"], risk_score=0.22,
                coc_model=ChainOfCustodyModel.SEGREGATED, verification_status="VERIFIED"
            ),
            "T3-004": SupplyChainNode(
                node_id="T3-004", name="Forest Concession D", tier=3, country="IDN",
                commodity=CommodityType.TIMBER, annual_volume_tonnes=20000,
                suppliers=[], customers=["T2-003"], risk_score=0.75,
                coc_model=ChainOfCustodyModel.MASS_BALANCE, verification_status="UNVERIFIED"
            ),
        }

        logger.info(f"Initialized {len(self.sample_nodes)} sample supply chain nodes")

    def map_supply_chain(self, root_supplier: str, depth: int) -> SupplyChainNetwork:
        """
        Map multi-tier supply chain network.

        Args:
            root_supplier: Root node (operator) identifier
            depth: Depth of tiers to map

        Returns:
            Complete supply chain network

        Raises:
            ValueError: If depth exceeds max_tier_depth or root not found
        """
        try:
            if depth > self.config.max_tier_depth:
                raise ValueError(f"Depth {depth} exceeds max_tier_depth {self.config.max_tier_depth}")

            if root_supplier not in self.sample_nodes:
                raise ValueError(f"Root supplier {root_supplier} not found")

            # Build network by traversing suppliers
            nodes = []
            edges = []
            visited = set()

            def traverse(node_id: str, current_depth: int):
                if current_depth > depth or node_id in visited:
                    return
                visited.add(node_id)

                node = self.sample_nodes.get(node_id)
                if node:
                    nodes.append(node)

                    # Create edges to suppliers
                    for supplier_id in node.suppliers:
                        edge = SupplyChainEdge(
                            edge_id=f"EDGE-{node_id}-{supplier_id}",
                            source_node=supplier_id,
                            target_node=node_id,
                            annual_flow_tonnes=node.annual_volume_tonnes / len(node.suppliers),
                            coc_model=node.coc_model,
                            traceability_score=0.85 if node.verification_status == "VERIFIED" else 0.5
                        )
                        edges.append(edge)

                        # Traverse upstream
                        traverse(supplier_id, current_depth + 1)

            traverse(root_supplier, 0)

            # Calculate network metrics
            total_volume = sum(n.annual_volume_tonnes for n in nodes)
            avg_risk = sum(n.risk_score for n in nodes) / len(nodes) if nodes else 0.0
            verified_nodes = sum(1 for n in nodes if n.verification_status == "VERIFIED")
            traceability_completeness = verified_nodes / len(nodes) if nodes else 0.0
            total_tiers = max((n.tier for n in nodes), default=0) + 1

            # Calculate provenance hash
            provenance_data = {
                "root_supplier": root_supplier,
                "depth": depth,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "timestamp": datetime.utcnow().isoformat()
            }
            provenance_hash = self._calculate_hash(provenance_data)

            network = SupplyChainNetwork(
                network_id=f"NETWORK-{root_supplier}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                root_node_id=root_supplier,
                nodes=nodes,
                edges=edges,
                total_tiers=total_tiers,
                total_volume_tonnes=total_volume,
                avg_risk_score=avg_risk,
                traceability_completeness=traceability_completeness,
                creation_timestamp=datetime.utcnow(),
                provenance_hash=provenance_hash
            )

            # Cache network
            self.network_cache[network.network_id] = network

            logger.info(f"Mapped supply chain: {len(nodes)} nodes, {len(edges)} edges, {total_tiers} tiers")
            return network

        except Exception as e:
            logger.error(f"Failed to map supply chain: {str(e)}", exc_info=True)
            raise

    def identify_critical_nodes(self, network: SupplyChainNetwork) -> List[CriticalNode]:
        """
        Identify critical nodes in supply chain network.

        Args:
            network: Supply chain network

        Returns:
            List of critical nodes sorted by criticality score
        """
        try:
            critical_nodes = []

            # Build dependency map
            downstream_deps = {}
            for node in network.nodes:
                downstream_deps[node.node_id] = len(node.customers)

            for node in network.nodes:
                # Calculate volume percentage
                volume_pct = node.annual_volume_tonnes / network.total_volume_tonnes if network.total_volume_tonnes > 0 else 0

                # Calculate criticality score (volume + downstream deps + risk)
                volume_factor = volume_pct
                dependency_factor = downstream_deps.get(node.node_id, 0) / len(network.nodes)
                risk_factor = node.risk_score

                criticality_score = (
                    0.5 * volume_factor +
                    0.3 * dependency_factor +
                    0.2 * risk_factor
                )

                # Determine importance type
                if volume_pct >= 0.3:
                    importance_type = "VOLUME"
                elif downstream_deps.get(node.node_id, 0) >= 3:
                    importance_type = "BOTTLENECK"
                elif risk_factor >= 0.6:
                    importance_type = "RISK"
                else:
                    importance_type = "NORMAL"

                # Count alternative suppliers (same tier, same country, lower risk)
                alternatives = sum(
                    1 for n in network.nodes
                    if n.tier == node.tier and n.country == node.country and
                    n.risk_score < node.risk_score and n.node_id != node.node_id
                )

                # Determine mitigation priority
                if criticality_score >= 0.7:
                    priority = "CRITICAL"
                elif criticality_score >= 0.5:
                    priority = "HIGH"
                elif criticality_score >= 0.3:
                    priority = "MEDIUM"
                else:
                    priority = "LOW"

                if criticality_score >= self.config.critical_node_threshold:
                    critical_nodes.append(CriticalNode(
                        node_id=node.node_id,
                        node_name=node.name,
                        criticality_score=criticality_score,
                        importance_type=importance_type,
                        volume_percentage=volume_pct,
                        downstream_dependents=downstream_deps.get(node.node_id, 0),
                        alternative_suppliers=alternatives,
                        mitigation_priority=priority
                    ))

            # Sort by criticality score
            critical_nodes.sort(key=lambda x: x.criticality_score, reverse=True)

            logger.info(f"Identified {len(critical_nodes)} critical nodes")
            return critical_nodes

        except Exception as e:
            logger.error(f"Failed to identify critical nodes: {str(e)}", exc_info=True)
            return []

    def calculate_concentration_risk(self, network: SupplyChainNetwork) -> ConcentrationRisk:
        """
        Calculate supply chain concentration risk.

        Args:
            network: Supply chain network

        Returns:
            Concentration risk analysis
        """
        try:
            # Calculate Herfindahl-Hirschman Index
            tier1_nodes = [n for n in network.nodes if n.tier == 1]
            if not tier1_nodes:
                tier1_nodes = network.nodes

            total_volume = sum(n.annual_volume_tonnes for n in tier1_nodes)
            market_shares = [n.annual_volume_tonnes / total_volume for n in tier1_nodes] if total_volume > 0 else []
            hhi = sum(share ** 2 for share in market_shares)

            # Calculate top-N concentration
            sorted_shares = sorted(market_shares, reverse=True)
            top3_concentration = sum(sorted_shares[:3]) if len(sorted_shares) >= 3 else sum(sorted_shares)
            top5_concentration = sum(sorted_shares[:5]) if len(sorted_shares) >= 5 else sum(sorted_shares)

            # Geographic concentration
            country_volumes = {}
            for node in tier1_nodes:
                country_volumes[node.country] = country_volumes.get(node.country, 0) + node.annual_volume_tonnes

            geographic_concentration = {
                country: volume / total_volume for country, volume in country_volumes.items()
            } if total_volume > 0 else {}

            # Determine risk level
            if hhi >= 0.7 or top3_concentration >= 0.8:
                risk_level = "CRITICAL"
                diversification_needed = True
                recommendation = "Immediate diversification required - single supplier dependency risk"
            elif hhi >= 0.5 or top3_concentration >= 0.7:
                risk_level = "HIGH"
                diversification_needed = True
                recommendation = "Diversification recommended - reduce top-3 concentration below 60%"
            elif hhi >= 0.3 or top3_concentration >= 0.6:
                risk_level = "MEDIUM"
                diversification_needed = False
                recommendation = "Monitor concentration levels - consider adding 2-3 alternative suppliers"
            else:
                risk_level = "LOW"
                diversification_needed = False
                recommendation = "Concentration levels acceptable - maintain current supplier diversity"

            result = ConcentrationRisk(
                herfindahl_index=hhi,
                top_3_concentration=top3_concentration,
                top_5_concentration=top5_concentration,
                geographic_concentration=geographic_concentration,
                risk_level=risk_level,
                diversification_needed=diversification_needed,
                recommendation=recommendation
            )

            logger.info(f"Concentration risk: HHI={hhi:.3f}, Top3={top3_concentration:.1%}, level={risk_level}")
            return result

        except Exception as e:
            logger.error(f"Failed to calculate concentration risk: {str(e)}", exc_info=True)
            raise

    def trace_origin(self, product_id: str) -> OriginTrace:
        """
        Trace product origin through supply chain.

        Args:
            product_id: Product identifier

        Returns:
            Origin trace with full supply chain path
        """
        try:
            # Mock tracing (in production, query actual product records)
            origin_plots = [f"PLOT-{product_id}-001", f"PLOT-{product_id}-002"]
            supply_chain_path = ["T3-001", "T2-001", "T1-001", "OP-001"]
            total_hops = len(supply_chain_path) - 1

            # Calculate traceability confidence based on verification status
            verified_hops = 3  # Mock: 3 out of 4 nodes verified
            traceability_confidence = verified_hops / len(supply_chain_path)

            result = OriginTrace(
                product_id=product_id,
                origin_plots=origin_plots,
                supply_chain_path=supply_chain_path,
                total_hops=total_hops,
                traceability_confidence=traceability_confidence,
                coc_model=ChainOfCustodyModel.SEGREGATED,
                mass_balance_verified=True
            )

            logger.info(f"Traced product {product_id}: {total_hops} hops, confidence={traceability_confidence:.1%}")
            return result

        except Exception as e:
            logger.error(f"Failed to trace origin: {str(e)}", exc_info=True)
            raise

    def track_mass_balance(self, commodity_flow: Dict[str, Any]) -> MassBalanceResult:
        """
        Track mass balance for commodity flow verification.

        Args:
            commodity_flow: Commodity flow data with inputs/outputs

        Returns:
            Mass balance verification result
        """
        try:
            node_id = commodity_flow.get('node_id', 'UNKNOWN')
            input_volume = commodity_flow.get('input_volume_tonnes', 0.0)
            output_volume = commodity_flow.get('output_volume_tonnes', 0.0)
            conversion_rate = commodity_flow.get('conversion_rate', 0.85)  # Default 85% yield

            expected_output = input_volume * conversion_rate
            actual_output = output_volume
            variance = actual_output - expected_output
            variance_pct = (variance / expected_output * 100) if expected_output > 0 else 0

            # Check if within tolerance
            is_balanced = abs(variance_pct) <= (self.config.mass_balance_tolerance * 100)

            discrepancies = []
            if not is_balanced:
                discrepancies.append(f"Variance {variance_pct:.1f}% exceeds tolerance {self.config.mass_balance_tolerance * 100:.1f}%")

            result = MassBalanceResult(
                node_id=node_id,
                input_volume_tonnes=input_volume,
                output_volume_tonnes=output_volume,
                conversion_rate=conversion_rate,
                expected_output=expected_output,
                actual_output=actual_output,
                variance_percentage=variance_pct,
                is_balanced=is_balanced,
                discrepancies=discrepancies
            )

            logger.info(f"Mass balance for {node_id}: balanced={is_balanced}, variance={variance_pct:.1f}%")
            return result

        except Exception as e:
            logger.error(f"Failed to track mass balance: {str(e)}", exc_info=True)
            raise

    def analyze_diversification(self, network: SupplyChainNetwork) -> DiversificationScore:
        """
        Analyze supply chain diversification.

        Args:
            network: Supply chain network

        Returns:
            Diversification assessment
        """
        try:
            tier1_nodes = [n for n in network.nodes if n.tier == 1]

            # Supplier diversity (inverse HHI)
            total_volume = sum(n.annual_volume_tonnes for n in tier1_nodes) if tier1_nodes else 1
            shares = [n.annual_volume_tonnes / total_volume for n in tier1_nodes]
            hhi = sum(s ** 2 for s in shares)
            supplier_diversity = 1 - hhi

            # Geographic diversity (number of countries / max possible)
            countries = set(n.country for n in tier1_nodes)
            geographic_diversity = min(len(countries) / 5, 1.0)  # Normalize to max 5 countries

            # Commodity diversity (always 1.0 for single commodity, or calculate if multi-commodity)
            commodity_diversity = 1.0  # Single commodity in this implementation

            # Overall score
            overall_score = (
                0.5 * supplier_diversity +
                0.4 * geographic_diversity +
                0.1 * commodity_diversity
            )

            # Determine risk level
            if overall_score >= 0.7:
                risk_level = "LOW"
            elif overall_score >= 0.5:
                risk_level = "MEDIUM"
            elif overall_score >= 0.3:
                risk_level = "HIGH"
            else:
                risk_level = "CRITICAL"

            # Improvement opportunities
            opportunities = []
            if supplier_diversity < 0.6:
                opportunities.append("Add 2-3 new tier-1 suppliers to reduce concentration")
            if geographic_diversity < 0.6:
                opportunities.append("Diversify sourcing across additional countries")
            if len(tier1_nodes) < 3:
                opportunities.append("Increase total number of tier-1 suppliers to minimum of 3")

            result = DiversificationScore(
                supplier_diversity_index=supplier_diversity,
                geographic_diversity_index=geographic_diversity,
                commodity_diversity_index=commodity_diversity,
                overall_score=overall_score,
                risk_level=risk_level,
                improvement_opportunities=opportunities
            )

            logger.info(f"Diversification score: {overall_score:.3f}, risk={risk_level}")
            return result

        except Exception as e:
            logger.error(f"Failed to analyze diversification: {str(e)}", exc_info=True)
            raise

    def model_risk_propagation(self, network: SupplyChainNetwork, shock_node: str) -> PropagationResult:
        """
        Model risk propagation from shock node.

        Args:
            network: Supply chain network
            shock_node: Node ID where shock originates

        Returns:
            Risk propagation analysis
        """
        try:
            shock_type = "SUPPLY_DISRUPTION"  # Default shock type
            affected_nodes = [shock_node]
            visited = set([shock_node])

            # Find shock node
            shock_node_obj = next((n for n in network.nodes if n.node_id == shock_node), None)
            if not shock_node_obj:
                raise ValueError(f"Shock node {shock_node} not found in network")

            # Propagate downstream
            def propagate_downstream(node_id: str):
                for node in network.nodes:
                    if node_id in node.suppliers and node.node_id not in visited:
                        affected_nodes.append(node.node_id)
                        visited.add(node.node_id)
                        propagate_downstream(node.node_id)

            propagate_downstream(shock_node)

            # Calculate impacts
            affected_volume = sum(
                n.annual_volume_tonnes for n in network.nodes if n.node_id in affected_nodes
            )
            volume_impact_pct = affected_volume / network.total_volume_tonnes if network.total_volume_tonnes > 0 else 0

            # Calculate propagation depth
            max_tier = max((n.tier for n in network.nodes if n.node_id in affected_nodes), default=0)
            min_tier = shock_node_obj.tier
            propagation_depth = max_tier - min_tier + 1

            # Estimate recovery time based on impact
            if volume_impact_pct >= 0.5:
                recovery_time_days = 90
            elif volume_impact_pct >= 0.3:
                recovery_time_days = 60
            elif volume_impact_pct >= 0.15:
                recovery_time_days = 30
            else:
                recovery_time_days = 14

            # Mitigation actions
            mitigation_actions = [
                f"Activate alternative suppliers in tier {shock_node_obj.tier}",
                f"Increase capacity at unaffected tier-{shock_node_obj.tier} nodes by 20-30%",
                "Expedite verification of pending suppliers for emergency activation",
                f"Implement {recovery_time_days}-day contingency plan"
            ]

            result = PropagationResult(
                shock_node_id=shock_node,
                shock_type=shock_type,
                affected_nodes=affected_nodes,
                propagation_depth=propagation_depth,
                volume_impact_tonnes=affected_volume,
                volume_impact_percentage=volume_impact_pct,
                recovery_time_days=recovery_time_days,
                mitigation_actions=mitigation_actions
            )

            logger.info(f"Risk propagation from {shock_node}: {len(affected_nodes)} nodes affected, "
                       f"{volume_impact_pct:.1%} volume impacted")
            return result

        except Exception as e:
            logger.error(f"Failed to model risk propagation: {str(e)}", exc_info=True)
            raise

    def suggest_alternatives(self, node_id: str) -> List[AlternativeSupplier]:
        """
        Suggest alternative suppliers for a node.

        Args:
            node_id: Node identifier to find alternatives for

        Returns:
            List of alternative suppliers
        """
        try:
            # Find target node
            target_node = self.sample_nodes.get(node_id)
            if not target_node:
                raise ValueError(f"Node {node_id} not found")

            alternatives = []

            # Find similar nodes (same tier, lower risk)
            for candidate_id, candidate in self.sample_nodes.items():
                if (candidate.tier == target_node.tier and
                    candidate.commodity == target_node.commodity and
                    candidate.node_id != node_id and
                    candidate.risk_score < target_node.risk_score):

                    # Calculate compatibility score
                    risk_improvement = target_node.risk_score - candidate.risk_score
                    country_match = 1.0 if candidate.country == target_node.country else 0.7
                    coc_match = 1.0 if candidate.coc_model == target_node.coc_model else 0.8
                    compatibility = (risk_improvement * 0.5 + country_match * 0.3 + coc_match * 0.2)

                    # Mock cost and lead time
                    cost_differential = -15.0 if candidate.risk_score < 0.3 else 5.0
                    lead_time_days = 30 if candidate.country == target_node.country else 60

                    alternatives.append(AlternativeSupplier(
                        supplier_id=candidate.node_id,
                        supplier_name=candidate.name,
                        country=candidate.country,
                        capacity_tonnes=candidate.annual_volume_tonnes,
                        risk_score=candidate.risk_score,
                        compatibility_score=compatibility,
                        cost_differential=cost_differential,
                        lead_time_days=lead_time_days
                    ))

            # Sort by compatibility
            alternatives.sort(key=lambda x: x.compatibility_score, reverse=True)

            logger.info(f"Found {len(alternatives)} alternative suppliers for {node_id}")
            return alternatives[:5]  # Return top 5

        except Exception as e:
            logger.error(f"Failed to suggest alternatives: {str(e)}", exc_info=True)
            return []

    def scenario_planning(self, network: SupplyChainNetwork, changes: List[ScenarioChange]) -> ScenarioResult:
        """
        Run scenario planning with proposed changes.

        Args:
            network: Baseline supply chain network
            changes: List of changes to apply

        Returns:
            Scenario analysis result
        """
        try:
            scenario_id = f"SCENARIO-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

            # Calculate baseline metrics
            baseline_risk = network.avg_risk_score
            concentration = self.calculate_concentration_risk(network)
            baseline_concentration = concentration.herfindahl_index

            # Simulate changes (simplified - in production, rebuild network)
            scenario_risk = baseline_risk
            scenario_concentration = baseline_concentration

            for change in changes:
                if change.change_type == "ADD_SUPPLIER":
                    # Adding supplier reduces concentration and may reduce risk
                    scenario_concentration *= 0.85
                    scenario_risk *= 0.95
                elif change.change_type == "REMOVE_SUPPLIER":
                    # Removing supplier increases concentration and may increase risk
                    scenario_concentration *= 1.15
                    scenario_risk *= 1.05

            risk_change = scenario_risk - baseline_risk
            concentration_change = scenario_concentration - baseline_concentration

            # Feasibility score (higher = more feasible)
            if len(changes) <= 2:
                feasibility = 0.9
            elif len(changes) <= 4:
                feasibility = 0.7
            else:
                feasibility = 0.5

            # Recommendation
            if risk_change < 0 and concentration_change < 0:
                recommendation = "RECOMMENDED - Scenario improves both risk and concentration"
            elif risk_change < 0 or concentration_change < 0:
                recommendation = "CONSIDER - Scenario shows partial improvement"
            else:
                recommendation = "NOT RECOMMENDED - Scenario increases risk or concentration"

            result = ScenarioResult(
                scenario_id=scenario_id,
                description=f"Scenario with {len(changes)} changes",
                changes_applied=changes,
                baseline_risk=baseline_risk,
                scenario_risk=scenario_risk,
                risk_change=risk_change,
                baseline_concentration=baseline_concentration,
                scenario_concentration=scenario_concentration,
                feasibility_score=feasibility,
                recommendation=recommendation
            )

            logger.info(f"Scenario {scenario_id}: risk_change={risk_change:+.3f}, "
                       f"concentration_change={concentration_change:+.3f}")
            return result

        except Exception as e:
            logger.error(f"Failed to run scenario planning: {str(e)}", exc_info=True)
            raise

    def generate_network_graph(self, network: SupplyChainNetwork) -> NetworkGraph:
        """
        Generate network graph visualization data.

        Args:
            network: Supply chain network

        Returns:
            Network graph visualization data
        """
        try:
            nodes_data = []
            for node in network.nodes:
                nodes_data.append({
                    "id": node.node_id,
                    "label": node.name,
                    "tier": node.tier,
                    "risk_score": node.risk_score,
                    "volume": node.annual_volume_tonnes,
                    "color": self._risk_to_color(node.risk_score),
                    "size": node.annual_volume_tonnes / 1000  # Scale for visualization
                })

            edges_data = []
            for edge in network.edges:
                edges_data.append({
                    "source": edge.source_node,
                    "target": edge.target_node,
                    "flow": edge.annual_flow_tonnes,
                    "width": edge.annual_flow_tonnes / 5000,  # Scale for visualization
                    "coc_model": edge.coc_model.value
                })

            graph = NetworkGraph(
                graph_id=f"GRAPH-{network.network_id}",
                graph_type="FULL",
                nodes_data=nodes_data,
                edges_data=edges_data,
                layout_algorithm="HIERARCHICAL",
                color_scheme="RISK_BASED"
            )

            logger.info(f"Generated network graph with {len(nodes_data)} nodes, {len(edges_data)} edges")
            return graph

        except Exception as e:
            logger.error(f"Failed to generate network graph: {str(e)}", exc_info=True)
            raise

    def _risk_to_color(self, risk_score: float) -> str:
        """Convert risk score to color code."""
        if risk_score >= 0.7:
            return "#d32f2f"  # Red
        elif risk_score >= 0.5:
            return "#f57c00"  # Orange
        elif risk_score >= 0.3:
            return "#fbc02d"  # Yellow
        else:
            return "#388e3c"  # Green

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 hash for provenance tracking.

        Args:
            data: Data to hash

        Returns:
            Hexadecimal hash string
        """
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
