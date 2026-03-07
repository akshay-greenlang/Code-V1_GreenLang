# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-001 Supply Chain Mapper test suite.

Provides reusable fixtures for graph construction, node/edge factories,
risk propagation inputs, commodity-specific supply chain builders,
and mock services used across all test modules.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 Supply Chain Mapping Master (GL-EUDR-SCM-001)
"""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

from greenlang.agents.eudr.supply_chain_mapper.graph_engine import (
    GraphEngineConfig,
    SupplyChainGraphEngine,
    SupplyChainNode as GENode,
    SupplyChainEdge as GEEdge,
    NodeType as GENodeType,
    CustodyModel as GECustodyModel,
    RiskLevel as GERiskLevel,
    ComplianceStatus as GEComplianceStatus,
)
from greenlang.agents.eudr.supply_chain_mapper.risk_propagation import (
    RiskPropagationConfig,
    RiskPropagationEngine,
    NodeRiskInput,
    RiskLevel as RPRiskLevel,
)
from greenlang.agents.eudr.supply_chain_mapper.gap_analyzer import (
    GapAnalyzer,
    GapAnalyzerConfig,
)
from greenlang.agents.eudr.supply_chain_mapper.visualization_engine import (
    VisualizationEngine,
)
from greenlang.agents.eudr.supply_chain_mapper.provenance import (
    ProvenanceTracker,
)
from greenlang.agents.eudr.supply_chain_mapper.models import (
    EUDRCommodity,
    NodeType,
    CustodyModel,
    RiskLevel,
    ComplianceStatus,
    GapType,
    GapSeverity,
    SupplyChainNode,
    SupplyChainEdge,
    SupplyChainGraph,
    SupplyChainGap,
)


# ---------------------------------------------------------------------------
# Helper: deterministic UUID generator
# ---------------------------------------------------------------------------

class DeterministicUUID:
    """Generate sequential UUIDs for deterministic testing."""

    def __init__(self, prefix: str = "test"):
        self._counter = 0
        self._prefix = prefix

    def next(self) -> str:
        self._counter += 1
        return f"{self._prefix}-{self._counter:08d}"

    def reset(self):
        self._counter = 0


# ---------------------------------------------------------------------------
# Graph Engine Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def graph_config():
    """Create a test-mode graph engine configuration (no persistence)."""
    return GraphEngineConfig(
        enable_persistence=False,
        enable_audit_trail=True,
        enable_snapshots=False,
        max_graph_nodes=10_000,
    )


@pytest.fixture
def graph_engine(graph_config):
    """Create a SupplyChainGraphEngine in memory-only mode."""
    eng = SupplyChainGraphEngine(config=graph_config)
    _run(eng.initialize())
    return eng


@pytest.fixture
def uuid_gen():
    """Create a deterministic UUID generator."""
    return DeterministicUUID()


# ---------------------------------------------------------------------------
# Risk Propagation Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def risk_config():
    """Create default risk propagation config."""
    return RiskPropagationConfig()


@pytest.fixture
def risk_engine(risk_config):
    """Create a RiskPropagationEngine."""
    return RiskPropagationEngine(config=risk_config)


# ---------------------------------------------------------------------------
# Gap Analyzer Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gap_config():
    """Create default gap analyzer config."""
    return GapAnalyzerConfig()


@pytest.fixture
def gap_analyzer(gap_config):
    """Create a GapAnalyzer."""
    return GapAnalyzer(config=gap_config)


# ---------------------------------------------------------------------------
# Visualization Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def viz_engine():
    """Create a VisualizationEngine."""
    return VisualizationEngine()


# ---------------------------------------------------------------------------
# Provenance Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance_tracker():
    """Create a ProvenanceTracker."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Node/Edge Factory Helpers
# ---------------------------------------------------------------------------

def make_ge_node(
    node_type: str = "producer",
    operator_name: str = "Test Operator",
    country_code: str = "BR",
    node_id: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    commodities: Optional[List[str]] = None,
    tier_depth: int = 0,
    risk_score: float = 0.0,
    risk_level: str = "standard",
    compliance_status: str = "pending_verification",
    certifications: Optional[List[str]] = None,
    plot_ids: Optional[List[str]] = None,
    **kwargs,
) -> GENode:
    """Create a graph engine SupplyChainNode for testing."""
    return GENode(
        node_id=node_id or str(uuid.uuid4()),
        node_type=GENodeType(node_type),
        operator_name=operator_name,
        country_code=country_code,
        latitude=latitude,
        longitude=longitude,
        commodities=commodities or [],
        tier_depth=tier_depth,
        risk_score=risk_score,
        risk_level=GERiskLevel(risk_level),
        compliance_status=GEComplianceStatus(compliance_status),
        certifications=certifications or [],
        plot_ids=plot_ids or [],
        **kwargs,
    )


def make_ge_edge(
    source_node_id: str,
    target_node_id: str,
    commodity: str = "cocoa",
    quantity: Decimal = Decimal("1000"),
    edge_id: Optional[str] = None,
    custody_model: str = "segregated",
    batch_number: Optional[str] = None,
    **kwargs,
) -> GEEdge:
    """Create a graph engine SupplyChainEdge for testing."""
    return GEEdge(
        edge_id=edge_id or str(uuid.uuid4()),
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        commodity=commodity,
        quantity=quantity,
        custody_model=GECustodyModel(custody_model),
        batch_number=batch_number,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# EUDR Commodity Supply Chain Builders
# ---------------------------------------------------------------------------

# Country risk benchmarking data (simplified for testing)
COUNTRY_RISK_DB = {
    "BR": 65.0,   # Brazil - standard/high
    "ID": 70.0,   # Indonesia - high
    "GH": 55.0,   # Ghana - standard
    "CI": 60.0,   # Cote d'Ivoire - standard
    "CO": 50.0,   # Colombia - standard
    "MY": 60.0,   # Malaysia - standard
    "PY": 55.0,   # Paraguay - standard
    "AR": 45.0,   # Argentina - standard
    "TH": 45.0,   # Thailand - standard
    "CM": 65.0,   # Cameroon - standard/high
    "NG": 70.0,   # Nigeria - high
    "DE": 10.0,   # Germany - low
    "NL": 10.0,   # Netherlands - low
    "FR": 10.0,   # France - low
    "BE": 10.0,   # Belgium - low
}

# Commodity-specific supply chain archetypes for building test chains
COMMODITY_CHAINS = {
    "cattle": {
        "origin_countries": ["BR", "AR", "PY"],
        "tiers": ["producer", "collector", "processor", "trader", "importer"],
        "derived_products": ["beef", "leather"],
    },
    "cocoa": {
        "origin_countries": ["GH", "CI", "CM"],
        "tiers": ["producer", "collector", "processor", "trader", "importer"],
        "derived_products": ["chocolate"],
    },
    "coffee": {
        "origin_countries": ["BR", "CO", "CI"],
        "tiers": ["producer", "collector", "processor", "trader", "importer"],
        "derived_products": [],
    },
    "oil_palm": {
        "origin_countries": ["ID", "MY"],
        "tiers": ["producer", "collector", "processor", "trader", "importer"],
        "derived_products": ["palm_oil"],
    },
    "rubber": {
        "origin_countries": ["ID", "TH", "MY"],
        "tiers": ["producer", "collector", "processor", "trader", "importer"],
        "derived_products": ["natural_rubber", "tyres"],
    },
    "soya": {
        "origin_countries": ["BR", "AR", "PY"],
        "tiers": ["producer", "collector", "processor", "trader", "importer"],
        "derived_products": ["soybean_oil", "soybean_meal"],
    },
    "wood": {
        "origin_countries": ["BR", "ID", "CM"],
        "tiers": ["producer", "collector", "processor", "trader", "importer"],
        "derived_products": ["timber", "furniture", "paper", "charcoal"],
    },
}


def build_complete_chain(
    engine: SupplyChainGraphEngine,
    commodity: str,
    graph_name: Optional[str] = None,
) -> Tuple[str, List[str], List[str]]:
    """Build a complete EUDR supply chain (producer -> importer) in the graph engine.

    Returns:
        Tuple of (graph_id, node_ids, edge_ids)
    """
    chain_info = COMMODITY_CHAINS[commodity]
    origin = chain_info["origin_countries"][0]

    graph_id = _run(engine.create_graph(
        operator_id="test-operator",
        commodity=commodity,
        graph_name=graph_name or f"Test {commodity} chain",
    ))

    node_ids = []
    for i, tier_type in enumerate(chain_info["tiers"]):
        cc = origin if tier_type == "producer" else ("NL" if tier_type == "importer" else origin)
        node_id = _run(engine.add_node(
            graph_id=graph_id,
            node_type=GENodeType(tier_type),
            operator_name=f"Test {tier_type.title()} {i}",
            country_code=cc,
            commodities=[commodity],
            tier_depth=len(chain_info["tiers"]) - 1 - i,
        ))
        node_ids.append(node_id)

    edge_ids = []
    for i in range(len(node_ids) - 1):
        edge_id = _run(engine.add_edge(
            graph_id=graph_id,
            source_node_id=node_ids[i],
            target_node_id=node_ids[i + 1],
            commodity=commodity,
            quantity=Decimal("1000"),
            product_description=f"{commodity} transfer {i}",
        ))
        edge_ids.append(edge_id)

    return graph_id, node_ids, edge_ids


# ---------------------------------------------------------------------------
# Shared constants for test assertions
# ---------------------------------------------------------------------------

ALL_COMMODITIES = ["cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]

ALL_NODE_TYPES = [
    "producer", "collector", "processor", "trader", "importer",
    "certifier", "warehouse", "port",
]

ALL_GAP_TYPES = [
    "missing_geolocation", "missing_polygon", "broken_custody_chain",
    "unverified_actor", "missing_tier", "mass_balance_discrepancy",
    "missing_certification", "stale_data", "orphan_node",
    "missing_documentation",
]
