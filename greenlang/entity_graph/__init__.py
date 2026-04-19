# -*- coding: utf-8 -*-
"""
GreenLang Entity Graph -- v3 Product Module (L1 Data Foundation)
=================================================================

The Entity Graph models the relationships between organizational entities
(facilities, suppliers, products, activities, emission sources) as a
directed graph.  It is the foundational data layer that other v3 product
modules -- Climate Ledger, Comply Engine, and the MRV pipeline -- build
upon.

Quick Start::

    from greenlang.entity_graph import EntityGraph, EntityNode, EntityEdge
    from greenlang.entity_graph.types import NodeType, EdgeType

    g = EntityGraph(graph_id="acme_corp")
    g.add_node(EntityNode(
        node_id="org_1",
        node_type=NodeType.ORGANIZATION,
        name="Acme Corp",
    ))
    g.add_node(EntityNode(
        node_id="fac_1",
        node_type=NodeType.FACILITY,
        name="Berlin Plant",
        geography="DE",
    ))
    g.add_edge(EntityEdge(
        edge_id="e_1",
        source_id="org_1",
        target_id="fac_1",
        edge_type=EdgeType.OWNS,
    ))

    print(g.stats())
    # {'graph_id': 'acme_corp', 'node_count': 2, 'edge_count': 1, ...}

Author: GreenLang Platform Team
Date: April 2026
Status: v3 Stub (0.1.0)
"""

from __future__ import annotations

from greenlang.entity_graph.graph import EntityGraph
from greenlang.entity_graph.models import EntityEdge, EntityNode
from greenlang.entity_graph.types import EdgeType, NodeType

__version__ = "0.1.0"

__all__ = [
    "EntityGraph",
    "EntityNode",
    "EntityEdge",
    "NodeType",
    "EdgeType",
]
