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

    The canonical v3 hierarchy is ``organization → facility → asset → meter``.
    Additional categories cover supply-chain, product, activity, emission,
    and geographic anchoring.
    """

    # Core organizational hierarchy (v3 spec).
    ORGANIZATION = "organization"
    FACILITY = "facility"
    ASSET = "asset"
    METER = "meter"

    # Extended categories.
    SUPPLIER = "supplier"
    PRODUCT = "product"
    ACTIVITY = "activity"
    EMISSION_SOURCE = "emission_source"
    GEOGRAPHY = "geography"

    # Canonical allowed parent for each node type.  Empty tuple == no parent constraint.
    # Used by EntityGraph.validate_hierarchy() to detect misrooted graphs.
    ALLOWED_PARENTS: dict[str, tuple[str, ...]] = {
        ORGANIZATION: (),
        FACILITY: (ORGANIZATION,),
        ASSET: (FACILITY,),
        METER: (ASSET, FACILITY),
        SUPPLIER: (ORGANIZATION,),
        PRODUCT: (ORGANIZATION, FACILITY),
        ACTIVITY: (FACILITY, ASSET),
        EMISSION_SOURCE: (FACILITY, ASSET, ACTIVITY),
        GEOGRAPHY: (),
    }

    ALL: list[str] = [
        ORGANIZATION,
        FACILITY,
        ASSET,
        METER,
        SUPPLIER,
        PRODUCT,
        ACTIVITY,
        EMISSION_SOURCE,
        GEOGRAPHY,
    ]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Return True if ``value`` is a recognised node type."""
        return value in cls.ALL


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
