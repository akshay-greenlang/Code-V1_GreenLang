# -*- coding: utf-8 -*-
"""
Pre-built Graph Queries for Knowledge Graph
============================================

Provides pre-built, optimized queries for common knowledge graph operations:
- Equipment relationships and hierarchy
- Standards applicability lookup
- Safety requirements discovery
- Process flow tracing

This module follows GreenLang's zero-hallucination principle by using
deterministic query templates with parameterized execution.

Example:
    >>> builder = GraphQueryBuilder()
    >>> query = builder.equipment.get_by_tag("B-101")
    >>> result = kg_service.execute_cypher(query.cypher, query.parameters)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Query Models
# =============================================================================

class QueryLanguage(str, Enum):
    """Supported query languages."""
    CYPHER = "cypher"
    SPARQL = "sparql"
    GREMLIN = "gremlin"


class QueryType(str, Enum):
    """Types of pre-built queries."""
    EQUIPMENT_LOOKUP = "equipment_lookup"
    EQUIPMENT_CONNECTIONS = "equipment_connections"
    EQUIPMENT_HIERARCHY = "equipment_hierarchy"
    STANDARDS_LOOKUP = "standards_lookup"
    STANDARDS_APPLICABILITY = "standards_applicability"
    SAFETY_REQUIREMENTS = "safety_requirements"
    SAFETY_INTERLOCKS = "safety_interlocks"
    PROCESS_FLOW = "process_flow"
    PROCESS_TRACE = "process_trace"
    SIMILARITY_SEARCH = "similarity_search"
    CROSS_REFERENCE = "cross_reference"


class QueryTemplate(BaseModel):
    """Query template with parameterized query string."""

    id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(default="", description="Template description")
    query_type: QueryType = Field(..., description="Type of query")
    language: QueryLanguage = Field(
        default=QueryLanguage.CYPHER,
        description="Query language"
    )
    template: str = Field(..., description="Query template string")
    parameters: Dict[str, str] = Field(
        default_factory=dict,
        description="Parameter descriptions"
    )
    required_parameters: List[str] = Field(
        default_factory=list,
        description="Required parameters"
    )
    default_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default parameter values"
    )

    @property
    def provenance_hash(self) -> str:
        """Generate provenance hash for this template."""
        data = f"{self.id}:{self.template}"
        return hashlib.sha256(data.encode()).hexdigest()


class ExecutableQuery(BaseModel):
    """Query ready for execution with bound parameters."""

    template_id: str = Field(..., description="Source template ID")
    query_type: QueryType = Field(..., description="Type of query")
    cypher: str = Field(..., description="Cypher query string")
    sparql: Optional[str] = Field(None, description="SPARQL query string")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Bound parameters"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def provenance_hash(self) -> str:
        """Generate provenance hash for this query execution."""
        import json
        data = f"{self.template_id}:{self.cypher}:{json.dumps(self.parameters, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()


# =============================================================================
# Equipment Queries
# =============================================================================

class EquipmentQuery:
    """
    Pre-built queries for equipment-related operations.

    Provides optimized queries for:
    - Equipment lookup by tag, type, or properties
    - Equipment connections and relationships
    - Equipment hierarchy traversal
    - Similar equipment discovery

    Example:
        >>> eq_query = EquipmentQuery()
        >>> query = eq_query.get_by_tag("B-101")
        >>> result = kg.execute_cypher(query.cypher, query.parameters)
    """

    # Query templates
    TEMPLATES = {
        "get_by_tag": QueryTemplate(
            id="eq_get_by_tag",
            name="Get Equipment by Tag",
            description="Retrieve equipment node by its tag identifier",
            query_type=QueryType.EQUIPMENT_LOOKUP,
            template="""
                MATCH (e:Equipment)
                WHERE e.id = $tag OR e.tag = $tag
                RETURN e
                LIMIT 1
            """,
            parameters={"tag": "Equipment tag (e.g., 'B-101')"},
            required_parameters=["tag"],
        ),
        "get_by_type": QueryTemplate(
            id="eq_get_by_type",
            name="Get Equipment by Type",
            description="Retrieve all equipment of a specific type",
            query_type=QueryType.EQUIPMENT_LOOKUP,
            template="""
                MATCH (e:Equipment)
                WHERE e.equipment_type = $equipment_type
                RETURN e
                ORDER BY e.id
                LIMIT $limit
            """,
            parameters={
                "equipment_type": "Equipment type (e.g., 'boiler')",
                "limit": "Maximum results"
            },
            required_parameters=["equipment_type"],
            default_values={"limit": 100},
        ),
        "get_connected": QueryTemplate(
            id="eq_get_connected",
            name="Get Connected Equipment",
            description="Get equipment connected to a given equipment",
            query_type=QueryType.EQUIPMENT_CONNECTIONS,
            template="""
                MATCH (e:Equipment {id: $equipment_id})-[r]-(connected:Equipment)
                WHERE type(r) IN $relationship_types OR size($relationship_types) = 0
                RETURN e, r, connected
                LIMIT $limit
            """,
            parameters={
                "equipment_id": "Source equipment ID",
                "relationship_types": "Filter by relationship types",
                "limit": "Maximum results"
            },
            required_parameters=["equipment_id"],
            default_values={"relationship_types": [], "limit": 100},
        ),
        "get_upstream": QueryTemplate(
            id="eq_get_upstream",
            name="Get Upstream Equipment",
            description="Get equipment feeding into a given equipment",
            query_type=QueryType.EQUIPMENT_CONNECTIONS,
            template="""
                MATCH path = (upstream:Equipment)-[:FEEDS|CONNECTS_TO*1..{depth}]->(e:Equipment {{id: $equipment_id}})
                RETURN upstream, relationships(path) as rels, length(path) as distance
                ORDER BY distance
                LIMIT $limit
            """,
            parameters={
                "equipment_id": "Target equipment ID",
                "depth": "Maximum traversal depth",
                "limit": "Maximum results"
            },
            required_parameters=["equipment_id"],
            default_values={"depth": 5, "limit": 50},
        ),
        "get_downstream": QueryTemplate(
            id="eq_get_downstream",
            name="Get Downstream Equipment",
            description="Get equipment receiving from a given equipment",
            query_type=QueryType.EQUIPMENT_CONNECTIONS,
            template="""
                MATCH path = (e:Equipment {{id: $equipment_id}})-[:FEEDS|CONNECTS_TO*1..{depth}]->(downstream:Equipment)
                RETURN downstream, relationships(path) as rels, length(path) as distance
                ORDER BY distance
                LIMIT $limit
            """,
            parameters={
                "equipment_id": "Source equipment ID",
                "depth": "Maximum traversal depth",
                "limit": "Maximum results"
            },
            required_parameters=["equipment_id"],
            default_values={"depth": 5, "limit": 50},
        ),
        "get_by_location": QueryTemplate(
            id="eq_get_by_location",
            name="Get Equipment by Location",
            description="Get all equipment in a specific location/area",
            query_type=QueryType.EQUIPMENT_LOOKUP,
            template="""
                MATCH (e:Equipment)-[:LOCATED_IN]->(loc:Location)
                WHERE loc.id = $location_id OR loc.name CONTAINS $location_name
                RETURN e, loc
                ORDER BY e.id
                LIMIT $limit
            """,
            parameters={
                "location_id": "Location ID",
                "location_name": "Location name (partial match)",
                "limit": "Maximum results"
            },
            required_parameters=[],
            default_values={"location_id": "", "location_name": "", "limit": 100},
        ),
        "find_similar": QueryTemplate(
            id="eq_find_similar",
            name="Find Similar Equipment",
            description="Find equipment similar to a given equipment",
            query_type=QueryType.SIMILARITY_SEARCH,
            template="""
                MATCH (e:Equipment {id: $equipment_id})
                MATCH (similar:Equipment)
                WHERE similar.id <> e.id
                  AND similar.equipment_type = e.equipment_type
                WITH e, similar,
                     CASE WHEN similar.capacity IS NOT NULL AND e.capacity IS NOT NULL
                          THEN 1.0 - abs(similar.capacity - e.capacity) / (e.capacity + 0.001)
                          ELSE 0.5 END as capacity_sim
                WHERE capacity_sim >= $similarity_threshold
                RETURN similar, capacity_sim as similarity
                ORDER BY similarity DESC
                LIMIT $limit
            """,
            parameters={
                "equipment_id": "Reference equipment ID",
                "similarity_threshold": "Minimum similarity score (0-1)",
                "limit": "Maximum results"
            },
            required_parameters=["equipment_id"],
            default_values={"similarity_threshold": 0.7, "limit": 10},
        ),
        "get_hierarchy": QueryTemplate(
            id="eq_get_hierarchy",
            name="Get Equipment Hierarchy",
            description="Get equipment hierarchy (parent/child relationships)",
            query_type=QueryType.EQUIPMENT_HIERARCHY,
            template="""
                MATCH path = (root:Equipment)-[:HAS_COMPONENT*0..{depth}]->(child:Equipment)
                WHERE root.id = $equipment_id OR child.id = $equipment_id
                RETURN nodes(path) as hierarchy, length(path) as depth
                ORDER BY depth
                LIMIT $limit
            """,
            parameters={
                "equipment_id": "Equipment ID",
                "depth": "Maximum hierarchy depth",
                "limit": "Maximum results"
            },
            required_parameters=["equipment_id"],
            default_values={"depth": 5, "limit": 100},
        ),
    }

    def __init__(self):
        """Initialize equipment query builder."""
        self.templates = self.TEMPLATES.copy()

    def get_by_tag(self, tag: str) -> ExecutableQuery:
        """
        Build query to get equipment by tag.

        Args:
            tag: Equipment tag identifier

        Returns:
            ExecutableQuery ready for execution

        Example:
            >>> query = eq_query.get_by_tag("B-101")
        """
        template = self.templates["get_by_tag"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={"tag": tag},
        )

    def get_by_type(
        self,
        equipment_type: str,
        limit: int = 100,
    ) -> ExecutableQuery:
        """
        Build query to get equipment by type.

        Args:
            equipment_type: Type of equipment (e.g., 'boiler')
            limit: Maximum results

        Returns:
            ExecutableQuery ready for execution
        """
        template = self.templates["get_by_type"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={"equipment_type": equipment_type, "limit": limit},
        )

    def get_connected(
        self,
        equipment_id: str,
        relationship_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> ExecutableQuery:
        """
        Build query to get connected equipment.

        Args:
            equipment_id: Source equipment ID
            relationship_types: Filter by relationship types
            limit: Maximum results

        Returns:
            ExecutableQuery ready for execution
        """
        template = self.templates["get_connected"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={
                "equipment_id": equipment_id,
                "relationship_types": relationship_types or [],
                "limit": limit,
            },
        )

    def get_upstream(
        self,
        equipment_id: str,
        depth: int = 5,
        limit: int = 50,
    ) -> ExecutableQuery:
        """
        Build query to get upstream equipment.

        Args:
            equipment_id: Target equipment ID
            depth: Maximum traversal depth
            limit: Maximum results

        Returns:
            ExecutableQuery ready for execution
        """
        template = self.templates["get_upstream"]
        cypher = template.template.format(depth=depth).strip()
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=cypher,
            parameters={"equipment_id": equipment_id, "limit": limit},
        )

    def get_downstream(
        self,
        equipment_id: str,
        depth: int = 5,
        limit: int = 50,
    ) -> ExecutableQuery:
        """
        Build query to get downstream equipment.

        Args:
            equipment_id: Source equipment ID
            depth: Maximum traversal depth
            limit: Maximum results

        Returns:
            ExecutableQuery ready for execution
        """
        template = self.templates["get_downstream"]
        cypher = template.template.format(depth=depth).strip()
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=cypher,
            parameters={"equipment_id": equipment_id, "limit": limit},
        )

    def find_similar(
        self,
        equipment_id: str,
        similarity_threshold: float = 0.7,
        limit: int = 10,
    ) -> ExecutableQuery:
        """
        Build query to find similar equipment.

        Args:
            equipment_id: Reference equipment ID
            similarity_threshold: Minimum similarity (0-1)
            limit: Maximum results

        Returns:
            ExecutableQuery ready for execution
        """
        template = self.templates["find_similar"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={
                "equipment_id": equipment_id,
                "similarity_threshold": similarity_threshold,
                "limit": limit,
            },
        )


# =============================================================================
# Standards Queries
# =============================================================================

class StandardsQuery:
    """
    Pre-built queries for standards-related operations.

    Provides optimized queries for:
    - Standards lookup by code or body
    - Standards applicability for equipment
    - Section and requirement lookup
    - Cross-references between standards

    Example:
        >>> std_query = StandardsQuery()
        >>> query = std_query.get_applicable("boiler")
        >>> result = kg.execute_cypher(query.cypher, query.parameters)
    """

    TEMPLATES = {
        "get_by_code": QueryTemplate(
            id="std_get_by_code",
            name="Get Standard by Code",
            description="Retrieve standard by its code",
            query_type=QueryType.STANDARDS_LOOKUP,
            template="""
                MATCH (s:Standard)
                WHERE s.code = $code OR s.id = $code
                RETURN s
                LIMIT 1
            """,
            parameters={"code": "Standard code (e.g., 'NFPA 85')"},
            required_parameters=["code"],
        ),
        "get_by_body": QueryTemplate(
            id="std_get_by_body",
            name="Get Standards by Body",
            description="Retrieve all standards from a standards body",
            query_type=QueryType.STANDARDS_LOOKUP,
            template="""
                MATCH (s:Standard)
                WHERE s.body = $body
                RETURN s
                ORDER BY s.code
                LIMIT $limit
            """,
            parameters={
                "body": "Standards body (e.g., 'ASME', 'API')",
                "limit": "Maximum results"
            },
            required_parameters=["body"],
            default_values={"limit": 100},
        ),
        "get_applicable": QueryTemplate(
            id="std_get_applicable",
            name="Get Applicable Standards",
            description="Get standards applicable to an equipment type",
            query_type=QueryType.STANDARDS_APPLICABILITY,
            template="""
                MATCH (s:Standard)-[:APPLIES_TO]->(eq_type)
                WHERE eq_type.id = $equipment_type OR eq_type.name = $equipment_type
                   OR $equipment_type IN s.equipment_types
                RETURN s
                ORDER BY s.code
                LIMIT $limit
            """,
            parameters={
                "equipment_type": "Equipment type",
                "limit": "Maximum results"
            },
            required_parameters=["equipment_type"],
            default_values={"limit": 50},
        ),
        "get_for_equipment": QueryTemplate(
            id="std_get_for_equipment",
            name="Get Standards for Equipment",
            description="Get standards applicable to a specific equipment",
            query_type=QueryType.STANDARDS_APPLICABILITY,
            template="""
                MATCH (e:Equipment {id: $equipment_id})
                MATCH (s:Standard)
                WHERE e.equipment_type IN s.equipment_types
                   OR (e)-[:COMPLIES_WITH]->(s)
                RETURN s
                ORDER BY s.code
                LIMIT $limit
            """,
            parameters={
                "equipment_id": "Equipment ID",
                "limit": "Maximum results"
            },
            required_parameters=["equipment_id"],
            default_values={"limit": 50},
        ),
        "get_cross_references": QueryTemplate(
            id="std_get_cross_refs",
            name="Get Cross-References",
            description="Get standards that reference or are referenced by a standard",
            query_type=QueryType.CROSS_REFERENCE,
            template="""
                MATCH (s:Standard {code: $code})-[r:REFERENCES|SUPERSEDES]-(other:Standard)
                RETURN s, type(r) as relationship, other
                LIMIT $limit
            """,
            parameters={
                "code": "Standard code",
                "limit": "Maximum results"
            },
            required_parameters=["code"],
            default_values={"limit": 50},
        ),
        "get_mandatory_sections": QueryTemplate(
            id="std_get_mandatory",
            name="Get Mandatory Sections",
            description="Get mandatory sections for equipment type",
            query_type=QueryType.STANDARDS_APPLICABILITY,
            template="""
                MATCH (s:Standard)-[:HAS_SECTION]->(sect:Section)
                WHERE s.code = $code AND sect.mandatory = true
                RETURN s.code as standard, sect
                ORDER BY sect.id
                LIMIT $limit
            """,
            parameters={
                "code": "Standard code",
                "limit": "Maximum results"
            },
            required_parameters=["code"],
            default_values={"limit": 100},
        ),
    }

    def __init__(self):
        """Initialize standards query builder."""
        self.templates = self.TEMPLATES.copy()

    def get_by_code(self, code: str) -> ExecutableQuery:
        """Build query to get standard by code."""
        template = self.templates["get_by_code"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={"code": code},
        )

    def get_by_body(self, body: str, limit: int = 100) -> ExecutableQuery:
        """Build query to get standards by body."""
        template = self.templates["get_by_body"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={"body": body, "limit": limit},
        )

    def get_applicable(
        self,
        equipment_type: str,
        limit: int = 50,
    ) -> ExecutableQuery:
        """Build query to get applicable standards for equipment type."""
        template = self.templates["get_applicable"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={"equipment_type": equipment_type, "limit": limit},
        )

    def get_for_equipment(
        self,
        equipment_id: str,
        limit: int = 50,
    ) -> ExecutableQuery:
        """Build query to get standards for specific equipment."""
        template = self.templates["get_for_equipment"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={"equipment_id": equipment_id, "limit": limit},
        )

    def get_cross_references(
        self,
        code: str,
        limit: int = 50,
    ) -> ExecutableQuery:
        """Build query to get cross-references for a standard."""
        template = self.templates["get_cross_references"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={"code": code, "limit": limit},
        )


# =============================================================================
# Safety Queries
# =============================================================================

class SafetyQuery:
    """
    Pre-built queries for safety-related operations.

    Provides optimized queries for:
    - Safety interlock lookup
    - Hazard identification
    - Protection layer analysis
    - Safety requirements for equipment

    Example:
        >>> safety_query = SafetyQuery()
        >>> query = safety_query.get_interlocks_for_equipment("B-101")
        >>> result = kg.execute_cypher(query.cypher, query.parameters)
    """

    TEMPLATES = {
        "get_interlocks_for_equipment": QueryTemplate(
            id="safety_get_interlocks",
            name="Get Interlocks for Equipment",
            description="Get safety interlocks for a specific equipment",
            query_type=QueryType.SAFETY_INTERLOCKS,
            template="""
                MATCH (e:Equipment {id: $equipment_id})-[:HAS_INTERLOCK]->(i:SafetyInterlock)
                RETURN e, i
                ORDER BY i.sil_level DESC
                LIMIT $limit
            """,
            parameters={
                "equipment_id": "Equipment ID",
                "limit": "Maximum results"
            },
            required_parameters=["equipment_id"],
            default_values={"limit": 50},
        ),
        "get_interlocks_by_type": QueryTemplate(
            id="safety_get_by_type",
            name="Get Interlocks by Type",
            description="Get safety interlocks of a specific type",
            query_type=QueryType.SAFETY_INTERLOCKS,
            template="""
                MATCH (i:SafetyInterlock)
                WHERE i.interlock_type = $interlock_type
                RETURN i
                ORDER BY i.sil_level DESC
                LIMIT $limit
            """,
            parameters={
                "interlock_type": "Interlock type (e.g., 'level', 'pressure')",
                "limit": "Maximum results"
            },
            required_parameters=["interlock_type"],
            default_values={"limit": 100},
        ),
        "get_interlocks_by_sil": QueryTemplate(
            id="safety_get_by_sil",
            name="Get Interlocks by SIL Level",
            description="Get safety interlocks requiring specific SIL level",
            query_type=QueryType.SAFETY_INTERLOCKS,
            template="""
                MATCH (i:SafetyInterlock)
                WHERE i.sil_level >= $min_sil_level
                RETURN i
                ORDER BY i.sil_level DESC
                LIMIT $limit
            """,
            parameters={
                "min_sil_level": "Minimum SIL level (1-4)",
                "limit": "Maximum results"
            },
            required_parameters=["min_sil_level"],
            default_values={"limit": 100},
        ),
        "get_hazards_for_equipment": QueryTemplate(
            id="safety_get_hazards",
            name="Get Hazards for Equipment",
            description="Get hazards associated with equipment",
            query_type=QueryType.SAFETY_REQUIREMENTS,
            template="""
                MATCH (e:Equipment {id: $equipment_id})-[:PROTECTS_AGAINST|HAS_HAZARD]->(h:Hazard)
                RETURN e, h
                ORDER BY h.severity DESC
                LIMIT $limit
            """,
            parameters={
                "equipment_id": "Equipment ID",
                "limit": "Maximum results"
            },
            required_parameters=["equipment_id"],
            default_values={"limit": 50},
        ),
        "get_protection_layers": QueryTemplate(
            id="safety_get_protection",
            name="Get Protection Layers",
            description="Get independent protection layers for equipment",
            query_type=QueryType.SAFETY_REQUIREMENTS,
            template="""
                MATCH (e:Equipment {id: $equipment_id})-[:HAS_PROTECTION]->(p:ProtectionLayer)
                RETURN e, p
                ORDER BY p.pfd ASC
                LIMIT $limit
            """,
            parameters={
                "equipment_id": "Equipment ID",
                "limit": "Maximum results"
            },
            required_parameters=["equipment_id"],
            default_values={"limit": 50},
        ),
        "get_safety_requirements": QueryTemplate(
            id="safety_get_requirements",
            name="Get Safety Requirements",
            description="Get all safety requirements for equipment",
            query_type=QueryType.SAFETY_REQUIREMENTS,
            template="""
                MATCH (e:Equipment {id: $equipment_id})
                OPTIONAL MATCH (e)-[:HAS_INTERLOCK]->(i:SafetyInterlock)
                OPTIONAL MATCH (e)-[:PROTECTS_AGAINST]->(h:Hazard)
                OPTIONAL MATCH (e)-[:HAS_PROTECTION]->(p:ProtectionLayer)
                RETURN e, collect(DISTINCT i) as interlocks,
                       collect(DISTINCT h) as hazards,
                       collect(DISTINCT p) as protection_layers
            """,
            parameters={"equipment_id": "Equipment ID"},
            required_parameters=["equipment_id"],
        ),
    }

    def __init__(self):
        """Initialize safety query builder."""
        self.templates = self.TEMPLATES.copy()

    def get_interlocks_for_equipment(
        self,
        equipment_id: str,
        limit: int = 50,
    ) -> ExecutableQuery:
        """Build query to get interlocks for equipment."""
        template = self.templates["get_interlocks_for_equipment"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={"equipment_id": equipment_id, "limit": limit},
        )

    def get_interlocks_by_type(
        self,
        interlock_type: str,
        limit: int = 100,
    ) -> ExecutableQuery:
        """Build query to get interlocks by type."""
        template = self.templates["get_interlocks_by_type"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={"interlock_type": interlock_type, "limit": limit},
        )

    def get_interlocks_by_sil(
        self,
        min_sil_level: int,
        limit: int = 100,
    ) -> ExecutableQuery:
        """Build query to get interlocks by SIL level."""
        template = self.templates["get_interlocks_by_sil"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={"min_sil_level": min_sil_level, "limit": limit},
        )

    def get_hazards_for_equipment(
        self,
        equipment_id: str,
        limit: int = 50,
    ) -> ExecutableQuery:
        """Build query to get hazards for equipment."""
        template = self.templates["get_hazards_for_equipment"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={"equipment_id": equipment_id, "limit": limit},
        )

    def get_safety_requirements(
        self,
        equipment_id: str,
    ) -> ExecutableQuery:
        """Build query to get all safety requirements for equipment."""
        template = self.templates["get_safety_requirements"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={"equipment_id": equipment_id},
        )


# =============================================================================
# Process Flow Queries
# =============================================================================

class ProcessFlowQuery:
    """
    Pre-built queries for process flow tracing.

    Provides optimized queries for:
    - Process flow paths
    - Material flow tracing
    - Energy flow analysis
    - Process unit operations

    Example:
        >>> flow_query = ProcessFlowQuery()
        >>> query = flow_query.trace_flow("B-101", depth=5)
        >>> result = kg.execute_cypher(query.cypher, query.parameters)
    """

    TEMPLATES = {
        "trace_flow": QueryTemplate(
            id="flow_trace",
            name="Trace Process Flow",
            description="Trace process flow from starting equipment",
            query_type=QueryType.PROCESS_FLOW,
            template="""
                MATCH path = (start:Equipment {{id: $start_id}})-[:FEEDS|CONNECTS_TO*1..{depth}]->(end:Equipment)
                RETURN nodes(path) as equipment,
                       relationships(path) as connections,
                       length(path) as path_length
                ORDER BY path_length
                LIMIT $limit
            """,
            parameters={
                "start_id": "Starting equipment ID",
                "depth": "Maximum traversal depth",
                "limit": "Maximum results"
            },
            required_parameters=["start_id"],
            default_values={"depth": 10, "limit": 100},
        ),
        "trace_reverse_flow": QueryTemplate(
            id="flow_trace_reverse",
            name="Trace Reverse Process Flow",
            description="Trace process flow to ending equipment (reverse)",
            query_type=QueryType.PROCESS_FLOW,
            template="""
                MATCH path = (start:Equipment)-[:FEEDS|CONNECTS_TO*1..{depth}]->(end:Equipment {{id: $end_id}})
                RETURN nodes(path) as equipment,
                       relationships(path) as connections,
                       length(path) as path_length
                ORDER BY path_length
                LIMIT $limit
            """,
            parameters={
                "end_id": "Ending equipment ID",
                "depth": "Maximum traversal depth",
                "limit": "Maximum results"
            },
            required_parameters=["end_id"],
            default_values={"depth": 10, "limit": 100},
        ),
        "find_path": QueryTemplate(
            id="flow_find_path",
            name="Find Path Between Equipment",
            description="Find shortest path between two equipment",
            query_type=QueryType.PROCESS_TRACE,
            template="""
                MATCH path = shortestPath(
                    (start:Equipment {id: $start_id})-[:FEEDS|CONNECTS_TO*..{max_depth}]-(end:Equipment {id: $end_id})
                )
                RETURN nodes(path) as equipment,
                       relationships(path) as connections,
                       length(path) as path_length
            """,
            parameters={
                "start_id": "Starting equipment ID",
                "end_id": "Ending equipment ID",
                "max_depth": "Maximum path length"
            },
            required_parameters=["start_id", "end_id"],
            default_values={"max_depth": 20},
        ),
        "get_process_units": QueryTemplate(
            id="flow_get_units",
            name="Get Process Units",
            description="Get all equipment in a process unit/area",
            query_type=QueryType.PROCESS_FLOW,
            template="""
                MATCH (e:Equipment)-[:PART_OF]->(u:ProcessUnit {id: $unit_id})
                OPTIONAL MATCH (e)-[r:FEEDS|CONNECTS_TO]-(other:Equipment)
                WHERE (other)-[:PART_OF]->(u)
                RETURN e, collect(DISTINCT {rel: r, other: other}) as connections
                LIMIT $limit
            """,
            parameters={
                "unit_id": "Process unit ID",
                "limit": "Maximum results"
            },
            required_parameters=["unit_id"],
            default_values={"limit": 100},
        ),
        "get_material_flow": QueryTemplate(
            id="flow_get_material",
            name="Get Material Flow",
            description="Trace flow of a specific material through process",
            query_type=QueryType.PROCESS_FLOW,
            template="""
                MATCH path = (e:Equipment)-[:PROCESSES]->(m:Material {{id: $material_id}})
                OPTIONAL MATCH (e)-[:FEEDS]->(next:Equipment)
                WHERE (next)-[:PROCESSES]->(m)
                RETURN e, next, relationships(path) as flow
                LIMIT $limit
            """,
            parameters={
                "material_id": "Material ID",
                "limit": "Maximum results"
            },
            required_parameters=["material_id"],
            default_values={"limit": 100},
        ),
    }

    def __init__(self):
        """Initialize process flow query builder."""
        self.templates = self.TEMPLATES.copy()

    def trace_flow(
        self,
        start_id: str,
        depth: int = 10,
        limit: int = 100,
    ) -> ExecutableQuery:
        """Build query to trace process flow from equipment."""
        template = self.templates["trace_flow"]
        cypher = template.template.format(depth=depth).strip()
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=cypher,
            parameters={"start_id": start_id, "limit": limit},
        )

    def trace_reverse_flow(
        self,
        end_id: str,
        depth: int = 10,
        limit: int = 100,
    ) -> ExecutableQuery:
        """Build query to trace reverse process flow to equipment."""
        template = self.templates["trace_reverse_flow"]
        cypher = template.template.format(depth=depth).strip()
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=cypher,
            parameters={"end_id": end_id, "limit": limit},
        )

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 20,
    ) -> ExecutableQuery:
        """Build query to find path between two equipment."""
        template = self.templates["find_path"]
        cypher = template.template.format(max_depth=max_depth).strip()
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=cypher,
            parameters={"start_id": start_id, "end_id": end_id},
        )

    def get_process_units(
        self,
        unit_id: str,
        limit: int = 100,
    ) -> ExecutableQuery:
        """Build query to get equipment in a process unit."""
        template = self.templates["get_process_units"]
        return ExecutableQuery(
            template_id=template.id,
            query_type=template.query_type,
            cypher=template.template.strip(),
            parameters={"unit_id": unit_id, "limit": limit},
        )


# =============================================================================
# Composite Query Builder
# =============================================================================

class GraphQueryBuilder:
    """
    Composite query builder providing access to all query types.

    Provides a unified interface for building queries across all domains:
    - Equipment queries
    - Standards queries
    - Safety queries
    - Process flow queries

    Example:
        >>> builder = GraphQueryBuilder()
        >>> eq_query = builder.equipment.get_by_tag("B-101")
        >>> std_query = builder.standards.get_applicable("boiler")
        >>> safety_query = builder.safety.get_interlocks_for_equipment("B-101")
    """

    def __init__(self):
        """Initialize composite query builder."""
        self.equipment = EquipmentQuery()
        self.standards = StandardsQuery()
        self.safety = SafetyQuery()
        self.process_flow = ProcessFlowQuery()

        logger.info("GraphQueryBuilder initialized")

    def get_all_templates(self) -> Dict[str, QueryTemplate]:
        """Get all available query templates."""
        templates = {}
        templates.update({f"equipment.{k}": v for k, v in self.equipment.templates.items()})
        templates.update({f"standards.{k}": v for k, v in self.standards.templates.items()})
        templates.update({f"safety.{k}": v for k, v in self.safety.templates.items()})
        templates.update({f"process_flow.{k}": v for k, v in self.process_flow.templates.items()})
        return templates

    def get_template(self, template_path: str) -> Optional[QueryTemplate]:
        """
        Get a specific template by path.

        Args:
            template_path: Template path (e.g., 'equipment.get_by_tag')

        Returns:
            QueryTemplate or None if not found
        """
        parts = template_path.split(".")
        if len(parts) != 2:
            return None

        domain, name = parts
        domain_builder = getattr(self, domain, None)
        if domain_builder and hasattr(domain_builder, "templates"):
            return domain_builder.templates.get(name)
        return None


# =============================================================================
# Module-level singleton
# =============================================================================

_query_builder_instance: Optional[GraphQueryBuilder] = None


def get_query_builder() -> GraphQueryBuilder:
    """Get or create the global query builder instance."""
    global _query_builder_instance
    if _query_builder_instance is None:
        _query_builder_instance = GraphQueryBuilder()
    return _query_builder_instance
