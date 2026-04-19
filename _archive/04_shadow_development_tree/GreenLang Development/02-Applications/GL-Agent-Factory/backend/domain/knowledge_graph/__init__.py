# -*- coding: utf-8 -*-
"""
GreenLang Knowledge Graph Service
==================================

Comprehensive Knowledge Graph service for industrial process heat systems.
Provides Neo4j-based graph storage, SPARQL query execution, entity extraction,
and semantic reasoning over equipment, processes, safety, and standards data.

Key Components:
- KnowledgeGraphService: Main service for graph operations
- EntityExtractor: Extract entities from equipment tags and descriptions
- GraphQueryBuilder: Pre-built queries for common operations
- SeedDataGenerator: Generate seed data for the knowledge graph

Example:
    >>> from backend.domain.knowledge_graph import KnowledgeGraphService
    >>> kg = KnowledgeGraphService()
    >>> equipment = kg.get_equipment_by_tag("B-101")
    >>> connected = kg.get_connected_equipment("B-101")
    >>> standards = kg.get_applicable_standards("boiler")
"""

from .knowledge_graph_service import (
    KnowledgeGraphService,
    KnowledgeGraphConfig,
    Triple,
    Node,
    Relationship,
    GraphQueryResult,
    TraversalResult,
    ProvenanceInfo,
)
from .entity_extractors import (
    EntityExtractor,
    ExtractedEntity,
    EntityType,
    TagPattern,
    EquipmentTagExtractor,
    ProcessParameterExtractor,
    SafetyInterlockExtractor,
    StandardsReferenceExtractor,
)
from .graph_queries import (
    GraphQueryBuilder,
    EquipmentQuery,
    StandardsQuery,
    SafetyQuery,
    ProcessFlowQuery,
    QueryTemplate,
)
from .seed_data import (
    SeedDataGenerator,
    EquipmentInstance,
    ProcessConnection,
    SafetyInterlockInstance,
    StandardReference,
    MeasurementInstance,
    SeedStatistics,
)

__all__ = [
    # Main Service
    "KnowledgeGraphService",
    "KnowledgeGraphConfig",
    "Triple",
    "Node",
    "Relationship",
    "GraphQueryResult",
    "TraversalResult",
    "ProvenanceInfo",
    # Entity Extractors
    "EntityExtractor",
    "ExtractedEntity",
    "EntityType",
    "TagPattern",
    "EquipmentTagExtractor",
    "ProcessParameterExtractor",
    "SafetyInterlockExtractor",
    "StandardsReferenceExtractor",
    # Graph Queries
    "GraphQueryBuilder",
    "EquipmentQuery",
    "StandardsQuery",
    "SafetyQuery",
    "ProcessFlowQuery",
    "QueryTemplate",
    # Seed Data
    "SeedDataGenerator",
    "EquipmentInstance",
    "ProcessConnection",
    "SafetyInterlockInstance",
    "StandardReference",
    "MeasurementInstance",
    "SeedStatistics",
]

__version__ = "1.0.0"
