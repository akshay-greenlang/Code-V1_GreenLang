# -*- coding: utf-8 -*-
"""
Entity Graph Data Models
=========================

Pydantic models for nodes and edges in the v3 Entity Graph product module
(L1 Data Foundation).  All models inherit from ``GreenLangBase`` so they
share the platform-wide ``model_config`` (extra="forbid", ORM mode, etc.).

Usage::

    from greenlang.entity_graph.models import EntityNode, EntityEdge
    from greenlang.entity_graph.types import NodeType, EdgeType

    node = EntityNode(
        node_id="fac_001",
        node_type=NodeType.FACILITY,
        name="Berlin Manufacturing Plant",
        geography="DE",
    )

    edge = EntityEdge(
        edge_id="e_001",
        source_id="org_001",
        target_id="fac_001",
        edge_type=EdgeType.OWNS,
    )

Author: GreenLang Platform Team
Date: April 2026
Status: v3 Stub
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from greenlang.schemas import GreenLangBase


class EntityNode(GreenLangBase):
    """A node in the v3 Entity Graph.

    Represents an organizational entity such as a facility, supplier,
    product, activity, or emission source.  The ``attributes`` dict
    provides schema-free extensibility for domain-specific metadata
    without violating ``extra="forbid"`` on the model itself.

    Attributes:
        node_id: Unique identifier for this node.
        node_type: Category of entity (see ``NodeType`` constants).
        name: Human-readable display name.
        attributes: Arbitrary key-value metadata for the node.
        geography: ISO 3166-1 alpha-2 country code or region string.
        created_at: Timestamp when the node was created (UTC).
        updated_at: Timestamp when the node was last modified (UTC).
    """

    node_id: str = Field(..., description="Unique identifier for this node")
    node_type: str = Field(
        ...,
        description="Category of entity (organization, facility, supplier, etc.)",
    )
    name: str = Field(..., description="Human-readable display name")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata for the node",
    )
    geography: Optional[str] = Field(
        default=None,
        description="ISO 3166-1 alpha-2 country code or region string",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the node was created (UTC)",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the node was last modified (UTC)",
    )


class EntityEdge(GreenLangBase):
    """A directed edge between two nodes in the v3 Entity Graph.

    Represents a typed relationship (e.g. "owns", "supplies_to")
    from a source node to a target node.  Edges may carry a numeric
    ``weight`` and optional temporal validity bounds.

    Attributes:
        edge_id: Unique identifier for this edge.
        source_id: Node ID of the relationship origin.
        target_id: Node ID of the relationship destination.
        edge_type: Relationship category (see ``EdgeType`` constants).
        attributes: Arbitrary key-value metadata for the edge.
        weight: Numeric weight / strength of the relationship.
        valid_from: Start of the period this edge is valid.
        valid_to: End of the period this edge is valid (None = open-ended).
    """

    edge_id: str = Field(..., description="Unique identifier for this edge")
    source_id: str = Field(
        ..., description="Node ID of the relationship origin"
    )
    target_id: str = Field(
        ..., description="Node ID of the relationship destination"
    )
    edge_type: str = Field(
        ...,
        description="Relationship category (owns, supplies_to, produces, etc.)",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata for the edge",
    )
    weight: float = Field(
        default=1.0,
        description="Numeric weight / strength of the relationship",
    )
    valid_from: Optional[datetime] = Field(
        default=None,
        description="Start of the period this edge is valid",
    )
    valid_to: Optional[datetime] = Field(
        default=None,
        description="End of the period this edge is valid (None = open-ended)",
    )
