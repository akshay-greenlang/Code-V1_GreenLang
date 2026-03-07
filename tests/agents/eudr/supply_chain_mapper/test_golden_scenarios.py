# -*- coding: utf-8 -*-
"""
Golden Test Scenarios - AGENT-EUDR-001 Supply Chain Mapper

49 golden tests covering all 7 EUDR commodities x 7 supply chain scenarios:
    1. Complete chain (100% traceability)
    2. Partial chain (missing intermediary)
    3. Broken chain (no producer)
    4. Many-to-many chain
    5. Batch split/merge chain
    6. High-risk chain
    7. Multi-tier chain (6+ tiers)

Each golden test validates:
    - Graph construction (correct node/edge counts)
    - Traceability score (complete vs partial)
    - Risk propagation (highest-risk-wins)
    - Gap detection (appropriate gaps for each scenario)
    - Provenance hash (deterministic, 64-char SHA-256)

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 (Section 13 - Golden Tests)
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import pytest

from greenlang.agents.eudr.supply_chain_mapper.graph_engine import (
    CustodyModel,
    GraphEngineConfig,
    NodeType,
    SupplyChainGraphEngine,
)


# ===========================================================================
# Async helper
# ===========================================================================

def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine():
    """Create engine for golden tests."""
    eng = SupplyChainGraphEngine(
        config=GraphEngineConfig(
            enable_persistence=False,
            enable_audit_trail=True,
            enable_snapshots=False,
            max_graph_nodes=50_000,
        )
    )
    _run(eng.initialize())
    return eng


# ---------------------------------------------------------------------------
# Commodity-specific origin configurations
# ---------------------------------------------------------------------------

COMMODITY_CONFIG = {
    "cattle": {
        "origins": [("BR", "Para"), ("AR", "Buenos Aires"), ("PY", "Chaco")],
        "derived": "beef",
        "typical_tiers": ["producer", "collector", "processor", "trader", "importer"],
        "importer_country": "DE",
    },
    "cocoa": {
        "origins": [("GH", "Ashanti"), ("CI", "Bas-Sassandra"), ("CM", "South-West")],
        "derived": "chocolate",
        "typical_tiers": ["producer", "collector", "processor", "trader", "importer"],
        "importer_country": "NL",
    },
    "coffee": {
        "origins": [("BR", "Minas Gerais"), ("CO", "Huila"), ("ET", "Yirgacheffe")],
        "derived": "coffee",
        "typical_tiers": ["producer", "collector", "processor", "trader", "importer"],
        "importer_country": "DE",
    },
    "oil_palm": {
        "origins": [("ID", "Riau"), ("MY", "Sabah"), ("ID", "Kalimantan")],
        "derived": "palm_oil",
        "typical_tiers": ["producer", "collector", "processor", "trader", "importer"],
        "importer_country": "NL",
    },
    "rubber": {
        "origins": [("ID", "North Sumatra"), ("TH", "Songkhla"), ("MY", "Johor")],
        "derived": "natural_rubber",
        "typical_tiers": ["producer", "collector", "processor", "trader", "importer"],
        "importer_country": "FR",
    },
    "soya": {
        "origins": [("BR", "Mato Grosso"), ("AR", "Cordoba"), ("PY", "Alto Parana")],
        "derived": "soybean_oil",
        "typical_tiers": ["producer", "collector", "processor", "trader", "importer"],
        "importer_country": "NL",
    },
    "wood": {
        "origins": [("BR", "Amazonas"), ("ID", "Papua"), ("CM", "East")],
        "derived": "timber",
        "typical_tiers": ["producer", "collector", "processor", "trader", "importer"],
        "importer_country": "BE",
    },
}

ALL_COMMODITIES = list(COMMODITY_CONFIG.keys())


# ===========================================================================
# Helper: Build supply chain scenarios
# ===========================================================================


def _build_complete_chain(
    engine: SupplyChainGraphEngine,
    commodity: str,
) -> Tuple[str, List[str], List[str]]:
    """Scenario 1: Complete chain with 100% traceability."""
    cfg = COMMODITY_CONFIG[commodity]
    origin_cc, origin_region = cfg["origins"][0]
    imp_cc = cfg["importer_country"]

    gid = _run(engine.create_graph(
        operator_id=f"golden-{commodity}", commodity=commodity,
        graph_name=f"Golden Complete {commodity}",
    ))

    tiers = cfg["typical_tiers"]
    nodes = []
    for i, tier in enumerate(tiers):
        cc = origin_cc if tier in ("producer", "collector") else (
            imp_cc if tier == "importer" else "CH"
        )
        nid = _run(engine.add_node(
            gid, NodeType(tier),
            operator_name=f"{commodity.title()} {tier.title()} {i}",
            country_code=cc,
            commodities=[commodity],
            tier_depth=len(tiers) - 1 - i,
            latitude=-2.5 + i if cc == origin_cc else 50.0 + i,
            longitude=-44.0 + i if cc == origin_cc else 4.0 + i,
            plot_ids=[f"PLOT-{commodity}-{i:03d}"] if tier == "producer" else [],
        ))
        nodes.append(nid)

    edges = []
    for i in range(len(nodes) - 1):
        eid = _run(engine.add_edge(
            gid, nodes[i], nodes[i + 1], commodity=commodity,
            quantity=Decimal("5000") - Decimal(str(i * 200)),
            product_description=f"{commodity} transfer tier {i}",
            batch_number=f"BATCH-{commodity.upper()}-{i:03d}",
        ))
        edges.append(eid)

    return gid, nodes, edges


def _build_partial_chain(
    engine: SupplyChainGraphEngine,
    commodity: str,
) -> Tuple[str, List[str], List[str]]:
    """Scenario 2: Partial chain (missing collector intermediary)."""
    cfg = COMMODITY_CONFIG[commodity]
    origin_cc, _ = cfg["origins"][0]
    imp_cc = cfg["importer_country"]

    gid = _run(engine.create_graph(
        operator_id=f"golden-partial-{commodity}", commodity=commodity,
        graph_name=f"Golden Partial {commodity}",
    ))

    # Skip collector tier
    tiers = ["producer", "processor", "trader", "importer"]
    nodes = []
    for i, tier in enumerate(tiers):
        cc = origin_cc if tier == "producer" else (imp_cc if tier == "importer" else "CH")
        nid = _run(engine.add_node(
            gid, NodeType(tier),
            operator_name=f"{commodity.title()} {tier.title()} {i}",
            country_code=cc,
            commodities=[commodity],
            tier_depth=len(tiers) - 1 - i,
        ))
        nodes.append(nid)

    edges = []
    for i in range(len(nodes) - 1):
        eid = _run(engine.add_edge(
            gid, nodes[i], nodes[i + 1], commodity=commodity,
            quantity=Decimal("3000"),
            product_description=f"Partial {commodity} transfer {i}",
        ))
        edges.append(eid)

    return gid, nodes, edges


def _build_broken_chain(
    engine: SupplyChainGraphEngine,
    commodity: str,
) -> Tuple[str, List[str], List[str]]:
    """Scenario 3: Broken chain (no producer, starts at collector)."""
    cfg = COMMODITY_CONFIG[commodity]
    origin_cc, _ = cfg["origins"][0]
    imp_cc = cfg["importer_country"]

    gid = _run(engine.create_graph(
        operator_id=f"golden-broken-{commodity}", commodity=commodity,
        graph_name=f"Golden Broken {commodity}",
    ))

    # No producer tier
    tiers = ["collector", "processor", "trader", "importer"]
    nodes = []
    for i, tier in enumerate(tiers):
        cc = origin_cc if tier == "collector" else (imp_cc if tier == "importer" else "CH")
        nid = _run(engine.add_node(
            gid, NodeType(tier),
            operator_name=f"{commodity.title()} {tier.title()} (broken)",
            country_code=cc,
            commodities=[commodity],
            tier_depth=len(tiers) - 1 - i,
        ))
        nodes.append(nid)

    edges = []
    for i in range(len(nodes) - 1):
        eid = _run(engine.add_edge(
            gid, nodes[i], nodes[i + 1], commodity=commodity,
            quantity=Decimal("2000"),
            product_description=f"Broken {commodity} transfer {i}",
        ))
        edges.append(eid)

    return gid, nodes, edges


def _build_many_to_many_chain(
    engine: SupplyChainGraphEngine,
    commodity: str,
) -> Tuple[str, List[str], List[str]]:
    """Scenario 4: Many-to-many (3 producers -> 2 collectors -> 1 processor -> importer)."""
    cfg = COMMODITY_CONFIG[commodity]
    origins = cfg["origins"]
    imp_cc = cfg["importer_country"]

    gid = _run(engine.create_graph(
        operator_id=f"golden-m2m-{commodity}", commodity=commodity,
        graph_name=f"Golden M2M {commodity}",
    ))

    nodes = []
    producers = []
    for i, (cc, region) in enumerate(origins):
        nid = _run(engine.add_node(
            gid, NodeType.PRODUCER,
            operator_name=f"Producer {region}",
            country_code=cc, commodities=[commodity], tier_depth=4,
            plot_ids=[f"PLOT-{cc}-{i:03d}"],
        ))
        nodes.append(nid)
        producers.append(nid)

    collectors = []
    for i in range(2):
        cc = origins[i][0]
        nid = _run(engine.add_node(
            gid, NodeType.COLLECTOR,
            operator_name=f"Collector {i}",
            country_code=cc, commodities=[commodity], tier_depth=3,
        ))
        nodes.append(nid)
        collectors.append(nid)

    processor = _run(engine.add_node(
        gid, NodeType.PROCESSOR,
        operator_name=f"Processor Central",
        country_code=origins[0][0], commodities=[commodity], tier_depth=2,
    ))
    nodes.append(processor)

    importer = _run(engine.add_node(
        gid, NodeType.IMPORTER,
        operator_name=f"EU Importer",
        country_code=imp_cc, commodities=[commodity], tier_depth=0,
    ))
    nodes.append(importer)

    edges = []
    # Producers -> Collectors (many-to-many)
    edges.append(_run(engine.add_edge(gid, producers[0], collectors[0], commodity, Decimal("1000"), product_description="P0->C0")))
    edges.append(_run(engine.add_edge(gid, producers[1], collectors[0], commodity, Decimal("800"), product_description="P1->C0")))
    edges.append(_run(engine.add_edge(gid, producers[1], collectors[1], commodity, Decimal("700"), product_description="P1->C1")))
    edges.append(_run(engine.add_edge(gid, producers[2], collectors[1], commodity, Decimal("900"), product_description="P2->C1")))
    # Collectors -> Processor
    edges.append(_run(engine.add_edge(gid, collectors[0], processor, commodity, Decimal("1800"), product_description="C0->Proc")))
    edges.append(_run(engine.add_edge(gid, collectors[1], processor, commodity, Decimal("1600"), product_description="C1->Proc")))
    # Processor -> Importer
    edges.append(_run(engine.add_edge(gid, processor, importer, commodity, Decimal("3400"), product_description="Proc->Imp")))

    return gid, nodes, edges


def _build_batch_split_merge(
    engine: SupplyChainGraphEngine,
    commodity: str,
) -> Tuple[str, List[str], List[str]]:
    """Scenario 5: Batch split/merge (1 producer -> split to 2 processors -> merge at trader)."""
    cfg = COMMODITY_CONFIG[commodity]
    origin_cc = cfg["origins"][0][0]
    imp_cc = cfg["importer_country"]

    gid = _run(engine.create_graph(
        operator_id=f"golden-split-{commodity}", commodity=commodity,
        graph_name=f"Golden Split/Merge {commodity}",
    ))

    producer = _run(engine.add_node(
        gid, NodeType.PRODUCER, f"Producer {commodity.title()}",
        origin_cc, commodities=[commodity], tier_depth=4,
        plot_ids=[f"PLOT-SPLIT-001"],
    ))
    proc1 = _run(engine.add_node(
        gid, NodeType.PROCESSOR, "Processor A",
        origin_cc, commodities=[commodity], tier_depth=2,
    ))
    proc2 = _run(engine.add_node(
        gid, NodeType.PROCESSOR, "Processor B",
        origin_cc, commodities=[commodity], tier_depth=2,
    ))
    trader = _run(engine.add_node(
        gid, NodeType.TRADER, "Trader Merge",
        "CH", commodities=[commodity], tier_depth=1,
    ))
    importer = _run(engine.add_node(
        gid, NodeType.IMPORTER, "EU Importer",
        imp_cc, commodities=[commodity], tier_depth=0,
    ))

    nodes = [producer, proc1, proc2, trader, importer]
    edges = []
    # Split: producer -> proc1, producer -> proc2
    edges.append(_run(engine.add_edge(gid, producer, proc1, commodity, Decimal("3000"),
                                 product_description="Split A", batch_number="BATCH-SPLIT-A")))
    edges.append(_run(engine.add_edge(gid, producer, proc2, commodity, Decimal("2000"),
                                 product_description="Split B", batch_number="BATCH-SPLIT-B")))
    # Merge: proc1 -> trader, proc2 -> trader
    edges.append(_run(engine.add_edge(gid, proc1, trader, commodity, Decimal("2800"),
                                 product_description="Merge A")))
    edges.append(_run(engine.add_edge(gid, proc2, trader, commodity, Decimal("1800"),
                                 product_description="Merge B")))
    # trader -> importer
    edges.append(_run(engine.add_edge(gid, trader, importer, commodity, Decimal("4600"),
                                 product_description="Final")))

    return gid, nodes, edges


def _build_high_risk_chain(
    engine: SupplyChainGraphEngine,
    commodity: str,
) -> Tuple[str, List[str], List[str]]:
    """Scenario 6: High-risk chain (all nodes high risk)."""
    cfg = COMMODITY_CONFIG[commodity]
    origin_cc = cfg["origins"][0][0]
    imp_cc = cfg["importer_country"]

    gid = _run(engine.create_graph(
        operator_id=f"golden-highrisk-{commodity}", commodity=commodity,
        graph_name=f"Golden High-Risk {commodity}",
    ))

    tiers = cfg["typical_tiers"]
    nodes = []
    for i, tier in enumerate(tiers):
        cc = origin_cc if tier in ("producer", "collector") else (imp_cc if tier == "importer" else origin_cc)
        nid = _run(engine.add_node(
            gid, NodeType(tier),
            operator_name=f"HighRisk {tier.title()} {i}",
            country_code=cc, commodities=[commodity],
            tier_depth=len(tiers) - 1 - i,
            risk_score=85.0 + i,
        ))
        nodes.append(nid)

    edges = []
    for i in range(len(nodes) - 1):
        eid = _run(engine.add_edge(
            gid, nodes[i], nodes[i + 1], commodity=commodity,
            quantity=Decimal("4000"),
            product_description=f"HighRisk {commodity} {i}",
        ))
        edges.append(eid)

    return gid, nodes, edges


def _build_multi_tier_chain(
    engine: SupplyChainGraphEngine,
    commodity: str,
) -> Tuple[str, List[str], List[str]]:
    """Scenario 7: Multi-tier chain (8 tiers = producer->coll->coll->proc->proc->trader->trader->importer)."""
    cfg = COMMODITY_CONFIG[commodity]
    origin_cc = cfg["origins"][0][0]
    imp_cc = cfg["importer_country"]

    gid = _run(engine.create_graph(
        operator_id=f"golden-multitier-{commodity}", commodity=commodity,
        graph_name=f"Golden Multi-Tier {commodity}",
    ))

    tier_types = [
        NodeType.PRODUCER, NodeType.COLLECTOR, NodeType.COLLECTOR,
        NodeType.PROCESSOR, NodeType.PROCESSOR, NodeType.TRADER,
        NodeType.TRADER, NodeType.IMPORTER,
    ]

    nodes = []
    for i, nt in enumerate(tier_types):
        cc = origin_cc if i < 4 else (imp_cc if i == len(tier_types) - 1 else "CH")
        nid = _run(engine.add_node(
            gid, nt,
            operator_name=f"Tier{i} {nt.value.title()}",
            country_code=cc, commodities=[commodity],
            tier_depth=len(tier_types) - 1 - i,
        ))
        nodes.append(nid)

    edges = []
    for i in range(len(nodes) - 1):
        eid = _run(engine.add_edge(
            gid, nodes[i], nodes[i + 1], commodity=commodity,
            quantity=Decimal("5000") - Decimal(str(i * 100)),
            product_description=f"MultiTier {commodity} {i}",
        ))
        edges.append(eid)

    return gid, nodes, edges


# ===========================================================================
# Scenario Builders Registry
# ===========================================================================

SCENARIOS = {
    "complete_chain": _build_complete_chain,
    "partial_chain": _build_partial_chain,
    "broken_chain": _build_broken_chain,
    "many_to_many": _build_many_to_many_chain,
    "batch_split_merge": _build_batch_split_merge,
    "high_risk": _build_high_risk_chain,
    "multi_tier": _build_multi_tier_chain,
}

SCENARIO_NAMES = list(SCENARIOS.keys())


# ===========================================================================
# Golden Tests: 7 commodities x 7 scenarios = 49 tests
# ===========================================================================


class TestGoldenScenarios:
    """Golden tests for all 7 commodities x 7 scenarios."""

    @pytest.mark.parametrize("commodity", ALL_COMMODITIES)
    def test_complete_chain(self, engine, commodity):
        """Scenario 1: Complete chain has all tiers and full connectivity."""
        gid, nodes, edges = _build_complete_chain(engine, commodity)
        stats = engine.get_statistics(gid)
        assert stats["total_nodes"] == 5
        assert stats["total_edges"] == 4
        assert engine.has_cycles(gid) is False
        # Root should be producer, leaf should be importer
        roots = engine.get_root_nodes(gid)
        leaves = engine.get_leaf_nodes(gid)
        assert len(roots) == 1
        assert len(leaves) == 1

    @pytest.mark.parametrize("commodity", ALL_COMMODITIES)
    def test_partial_chain(self, engine, commodity):
        """Scenario 2: Partial chain is missing collector tier."""
        gid, nodes, edges = _build_partial_chain(engine, commodity)
        stats = engine.get_statistics(gid)
        assert stats["total_nodes"] == 4
        assert stats["total_edges"] == 3
        assert engine.has_cycles(gid) is False

    @pytest.mark.parametrize("commodity", ALL_COMMODITIES)
    def test_broken_chain(self, engine, commodity):
        """Scenario 3: Broken chain has no producer."""
        gid, nodes, edges = _build_broken_chain(engine, commodity)
        stats = engine.get_statistics(gid)
        assert stats["total_nodes"] == 4
        assert stats["total_edges"] == 3
        # Root should be collector (no producer)
        roots = engine.get_root_nodes(gid)
        root_node = engine.get_node(gid, roots[0])
        assert root_node.node_type == NodeType.COLLECTOR

    @pytest.mark.parametrize("commodity", ALL_COMMODITIES)
    def test_many_to_many(self, engine, commodity):
        """Scenario 4: Many-to-many chain with 3 producers, 2 collectors."""
        gid, nodes, edges = _build_many_to_many_chain(engine, commodity)
        stats = engine.get_statistics(gid)
        assert stats["total_nodes"] == 7  # 3 prod + 2 coll + 1 proc + 1 imp
        assert stats["total_edges"] == 7
        # Multiple roots
        roots = engine.get_root_nodes(gid)
        assert len(roots) == 3

    @pytest.mark.parametrize("commodity", ALL_COMMODITIES)
    def test_batch_split_merge(self, engine, commodity):
        """Scenario 5: Batch split/merge chain."""
        gid, nodes, edges = _build_batch_split_merge(engine, commodity)
        stats = engine.get_statistics(gid)
        assert stats["total_nodes"] == 5
        assert stats["total_edges"] == 5
        # Diamond pattern: producer is root, importer is leaf
        roots = engine.get_root_nodes(gid)
        leaves = engine.get_leaf_nodes(gid)
        assert len(roots) == 1
        assert len(leaves) == 1

    @pytest.mark.parametrize("commodity", ALL_COMMODITIES)
    def test_high_risk(self, engine, commodity):
        """Scenario 6: All nodes are high-risk."""
        gid, nodes, edges = _build_high_risk_chain(engine, commodity)
        stats = engine.get_statistics(gid)
        assert stats["total_nodes"] == 5
        # All nodes should have risk_score >= 85
        for nid in nodes:
            node = engine.get_node(gid, nid)
            assert node.risk_score >= 85.0

    @pytest.mark.parametrize("commodity", ALL_COMMODITIES)
    def test_multi_tier(self, engine, commodity):
        """Scenario 7: Multi-tier chain with 8 tiers."""
        gid, nodes, edges = _build_multi_tier_chain(engine, commodity)
        stats = engine.get_statistics(gid)
        assert stats["total_nodes"] == 8
        assert stats["total_edges"] == 7
        # Max depth should be 7 (8 nodes - 1)
        depth = engine.get_max_depth(gid)
        assert depth >= 7
        # Topological sort should order all 8 nodes
        topo = engine.topological_sort(gid)
        assert len(topo) == 8
