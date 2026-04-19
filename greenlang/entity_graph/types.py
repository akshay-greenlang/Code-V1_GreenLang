# -*- coding: utf-8 -*-
"""
Entity Graph Type Constants
============================

Canonical string constants for node types and edge types used throughout
the v3 Entity Graph product module (L1 Data Foundation).

Using class-level constants rather than enums keeps the stub lightweight
while still providing IDE autocompletion and a single source of truth.

Usage::

    from greenlang.entity_graph.types import NodeType, EdgeType

    node = EntityNode(node_type=NodeType.FACILITY, ...)
    edge = EntityEdge(edge_type=EdgeType.OWNS, ...)

Author: GreenLang Platform Team
Date: April 2026
Status: v3 Stub
"""

from __future__ import annotations


class NodeType:
    """Allowed node types for the Entity Graph.

    Each constant represents a category of organizational entity that
    can participate in the v3 Entity Graph.
    """

    ORGANIZATION = "organization"
    FACILITY = "facility"
    SUPPLIER = "supplier"
    PRODUCT = "product"
    ACTIVITY = "activity"
    EMISSION_SOURCE = "emission_source"
    GEOGRAPHY = "geography"

    ALL: list[str] = [
        ORGANIZATION,
        FACILITY,
        SUPPLIER,
        PRODUCT,
        ACTIVITY,
        EMISSION_SOURCE,
        GEOGRAPHY,
    ]


class EdgeType:
    """Allowed edge types for the Entity Graph.

    Each constant represents a directed relationship between two
    ``EntityNode`` instances in the v3 Entity Graph.
    """

    OWNS = "owns"
    SUPPLIES_TO = "supplies_to"
    PRODUCES = "produces"
    EMITS = "emits"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    CONSUMES = "consumes"
    TRANSPORTS = "transports"

    ALL: list[str] = [
        OWNS,
        SUPPLIES_TO,
        PRODUCES,
        EMITS,
        LOCATED_IN,
        PART_OF,
        CONSUMES,
        TRANSPORTS,
    ]
