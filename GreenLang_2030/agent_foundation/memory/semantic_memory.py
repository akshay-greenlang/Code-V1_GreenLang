"""
Semantic Memory implementation for GreenLang Agent Foundation.

This module implements knowledge storage for facts, concepts, procedures,
and relationships with knowledge graph integration and hierarchical organization.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field, validator
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import defaultdict
import networkx as nx
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge in semantic memory."""
    FACT = "fact"          # Concrete facts
    CONCEPT = "concept"    # Abstract concepts
    PROCEDURE = "procedure"  # How-to knowledge
    RELATIONSHIP = "relationship"  # Entity relationships


class StorageType(Enum):
    """Storage backend types."""
    KNOWLEDGE_GRAPH = "knowledge_graph"
    VECTOR_DATABASE = "vector_database"
    DOCUMENT_STORE = "document_store"
    GRAPH_DATABASE = "graph_database"


@dataclass
class Triple:
    """RDF-style triple for knowledge graph."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_string(self) -> str:
        """Convert to string representation."""
        return f"{self.subject} -> {self.predicate} -> {self.object}"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'confidence': self.confidence,
            'source': self.source,
            'timestamp': self.timestamp.isoformat()
        }


class KnowledgeItem(BaseModel):
    """Individual knowledge item in semantic memory."""

    knowledge_id: str = Field(..., description="Unique knowledge identifier")
    knowledge_type: KnowledgeType = Field(..., description="Type of knowledge")
    content: Any = Field(..., description="Knowledge content")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score")
    version: int = Field(1, ge=1, description="Version number")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    access_count: int = Field(0, ge=0, description="Number of accesses")
    provenance_hash: str = Field("", description="SHA-256 hash for audit")

    @validator('provenance_hash', always=True)
    def calculate_provenance(cls, v, values):
        """Calculate SHA-256 hash if not provided."""
        if not v and 'knowledge_id' in values and 'content' in values:
            content_str = json.dumps(values['content'], sort_keys=True, default=str)
            id_str = values['knowledge_id']
            provenance_str = f"{id_str}{content_str}{values.get('version', 1)}"
            return hashlib.sha256(provenance_str.encode()).hexdigest()
        return v


class KnowledgeGraph:
    """
    Knowledge graph for semantic relationships.

    Uses NetworkX for graph operations and supports RDF-style triples.
    """

    def __init__(self):
        """Initialize knowledge graph."""
        self.graph = nx.DiGraph()
        self.triples: List[Triple] = []
        self.entity_index: Dict[str, List[Triple]] = defaultdict(list)
        self.predicate_index: Dict[str, List[Triple]] = defaultdict(list)

    def add_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        source: Optional[str] = None
    ) -> None:
        """
        Add a triple to the knowledge graph.

        Args:
            subject: Subject entity
            predicate: Relationship/predicate
            object: Object entity
            confidence: Confidence score
            source: Source of the knowledge
        """
        triple = Triple(subject, predicate, object, confidence, source)
        self.triples.append(triple)

        # Add to graph
        self.graph.add_edge(
            subject,
            object,
            predicate=predicate,
            confidence=confidence,
            timestamp=triple.timestamp
        )

        # Update indices
        self.entity_index[subject].append(triple)
        self.entity_index[object].append(triple)
        self.predicate_index[predicate].append(triple)

        logger.debug(f"Added triple: {triple.to_string()}")

    def query_by_subject(self, subject: str) -> List[Triple]:
        """Query all triples with given subject."""
        return [t for t in self.triples if t.subject == subject]

    def query_by_predicate(self, predicate: str) -> List[Triple]:
        """Query all triples with given predicate."""
        return self.predicate_index.get(predicate, [])

    def query_by_pattern(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None
    ) -> List[Triple]:
        """
        Query triples matching pattern.

        Args:
            subject: Subject pattern (None = wildcard)
            predicate: Predicate pattern
            object: Object pattern

        Returns:
            Matching triples
        """
        results = []
        for triple in self.triples:
            if subject and triple.subject != subject:
                continue
            if predicate and triple.predicate != predicate:
                continue
            if object and triple.object != object:
                continue
            results.append(triple)
        return results

    def find_path(self, start: str, end: str) -> Optional[List[str]]:
        """
        Find path between two entities.

        Args:
            start: Starting entity
            end: Target entity

        Returns:
            Path if exists, None otherwise
        """
        try:
            path = nx.shortest_path(self.graph, start, end)
            return path
        except nx.NetworkXNoPath:
            return None

    def get_neighbors(self, entity: str, depth: int = 1) -> Dict[int, List[str]]:
        """
        Get neighbors up to specified depth.

        Args:
            entity: Central entity
            depth: Maximum depth

        Returns:
            Dict of depth -> list of entities
        """
        neighbors = {}
        current_level = {entity}

        for d in range(1, depth + 1):
            next_level = set()
            for node in current_level:
                if node in self.graph:
                    next_level.update(self.graph.neighbors(node))
            neighbors[d] = list(next_level)
            current_level = next_level

        return neighbors

    def compute_similarity(self, entity1: str, entity2: str) -> float:
        """
        Compute similarity between two entities based on graph structure.

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            Similarity score 0-1
        """
        # Get neighbors for both entities
        neighbors1 = set(self.graph.neighbors(entity1)) if entity1 in self.graph else set()
        neighbors2 = set(self.graph.neighbors(entity2)) if entity2 in self.graph else set()

        if not neighbors1 and not neighbors2:
            return 0.0

        # Jaccard similarity
        intersection = neighbors1.intersection(neighbors2)
        union = neighbors1.union(neighbors2)

        if union:
            return len(intersection) / len(union)
        return 0.0


class SemanticMemory:
    """
    Semantic memory system for facts, concepts, procedures, and relationships.

    Implements hierarchical organization with domain categorization and
    multiple storage backends including knowledge graph, vector database,
    and document store.
    """

    def __init__(self, agent_id: str = "default"):
        """
        Initialize semantic memory.

        Args:
            agent_id: Agent identifier
        """
        self.agent_id = agent_id
        self.knowledge_graph = KnowledgeGraph()

        # Knowledge storage by type
        self.facts: Dict[str, KnowledgeItem] = {}
        self.concepts: Dict[str, KnowledgeItem] = {}
        self.procedures: Dict[str, KnowledgeItem] = {}
        self.relationships: Dict[str, KnowledgeItem] = {}

        # Hierarchical organization
        self.domain_hierarchy = {
            "Environmental": {
                "GHG Emissions": ["Scope 1", "Scope 2", "Scope 3"],
                "Energy": ["Renewable", "Non-renewable", "Efficiency"],
                "Waste": ["Hazardous", "Non-hazardous", "Recycling"],
                "Water": ["Consumption", "Discharge", "Quality"]
            },
            "Social": {
                "Labor": ["Health & Safety", "Diversity", "Training"],
                "Community": ["Impact", "Engagement", "Development"],
                "Human Rights": ["Supply Chain", "Operations", "Remediation"]
            },
            "Governance": {
                "Ethics": ["Anti-corruption", "Compliance", "Transparency"],
                "Risk": ["Climate", "Operational", "Regulatory"],
                "Board": ["Composition", "Oversight", "Compensation"]
            }
        }

        # Semantic similarity clusters
        self.concept_clusters: Dict[str, List[str]] = {}

        # Co-occurrence networks
        self.co_occurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Version control
        self.version_history: Dict[str, List[KnowledgeItem]] = defaultdict(list)

        # Statistics
        self.stats = {
            "total_facts": 0,
            "total_concepts": 0,
            "total_procedures": 0,
            "total_relationships": 0,
            "total_queries": 0
        }

    async def store_fact(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> str:
        """
        Store a fact in semantic memory.

        Args:
            content: Fact content (should be verifiable truth)
            metadata: Optional metadata
            embedding: Optional vector embedding

        Returns:
            Knowledge ID
        """
        knowledge_id = hashlib.sha256(
            f"fact_{json.dumps(content, sort_keys=True, default=str)}".encode()
        ).hexdigest()[:16]

        # Check if fact already exists
        if knowledge_id in self.facts:
            existing = self.facts[knowledge_id]
            existing.access_count += 1
            existing.updated_at = datetime.now()
            logger.debug(f"Fact already exists: {knowledge_id[:8]}")
            return knowledge_id

        # Create new fact
        fact = KnowledgeItem(
            knowledge_id=knowledge_id,
            knowledge_type=KnowledgeType.FACT,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )

        self.facts[knowledge_id] = fact
        self.stats["total_facts"] += 1

        # Extract and store as triple if applicable
        if isinstance(content, dict):
            if all(k in content for k in ['subject', 'predicate', 'object']):
                self.knowledge_graph.add_triple(
                    content['subject'],
                    content['predicate'],
                    content['object'],
                    confidence=fact.confidence,
                    source=knowledge_id
                )

        logger.info(f"Stored fact: {knowledge_id[:8]}")
        return knowledge_id

    async def store_concept(
        self,
        name: str,
        definition: str,
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        related_concepts: Optional[List[str]] = None
    ) -> str:
        """
        Store a concept in semantic memory.

        Args:
            name: Concept name
            definition: Concept definition
            properties: Optional properties
            embedding: Optional vector embedding
            related_concepts: Optional related concepts

        Returns:
            Knowledge ID
        """
        knowledge_id = hashlib.sha256(f"concept_{name}".encode()).hexdigest()[:16]

        # Check for existing concept
        if knowledge_id in self.concepts:
            # Update version
            existing = self.concepts[knowledge_id]
            self.version_history[knowledge_id].append(existing.copy())
            existing.version += 1
            existing.content = {
                'name': name,
                'definition': definition,
                'properties': properties or {}
            }
            existing.updated_at = datetime.now()
            logger.debug(f"Updated concept: {knowledge_id[:8]} to version {existing.version}")
            return knowledge_id

        # Create new concept
        concept = KnowledgeItem(
            knowledge_id=knowledge_id,
            knowledge_type=KnowledgeType.CONCEPT,
            content={
                'name': name,
                'definition': definition,
                'properties': properties or {}
            },
            embedding=embedding
        )

        self.concepts[knowledge_id] = concept
        self.stats["total_concepts"] += 1

        # Add to knowledge graph
        self.knowledge_graph.add_triple(name, "is_a", "concept")

        # Link related concepts
        if related_concepts:
            for related in related_concepts:
                self.knowledge_graph.add_triple(name, "related_to", related)
                self.co_occurrence[name][related] += 1

        logger.info(f"Stored concept: {name} ({knowledge_id[:8]})")
        return knowledge_id

    async def store_procedure(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        prerequisites: Optional[List[str]] = None,
        expected_outcomes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a procedure in semantic memory.

        Args:
            name: Procedure name
            steps: List of procedure steps
            prerequisites: Optional prerequisites
            expected_outcomes: Optional expected outcomes
            metadata: Optional metadata

        Returns:
            Knowledge ID
        """
        knowledge_id = hashlib.sha256(f"procedure_{name}".encode()).hexdigest()[:16]

        procedure = KnowledgeItem(
            knowledge_id=knowledge_id,
            knowledge_type=KnowledgeType.PROCEDURE,
            content={
                'name': name,
                'steps': steps,
                'prerequisites': prerequisites or [],
                'expected_outcomes': expected_outcomes or {}
            },
            metadata=metadata or {}
        )

        self.procedures[knowledge_id] = procedure
        self.stats["total_procedures"] += 1

        # Add to knowledge graph
        self.knowledge_graph.add_triple(name, "is_a", "procedure")
        for i, step in enumerate(steps):
            step_name = step.get('name', f"step_{i+1}")
            self.knowledge_graph.add_triple(name, "has_step", step_name)

        logger.info(f"Stored procedure: {name} with {len(steps)} steps")
        return knowledge_id

    async def store_relationship(
        self,
        subject: str,
        predicate: str,
        object: str,
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0
    ) -> str:
        """
        Store a relationship in semantic memory.

        Args:
            subject: Subject entity
            predicate: Relationship type
            object: Object entity
            properties: Optional relationship properties
            confidence: Confidence score

        Returns:
            Knowledge ID
        """
        relationship_str = f"{subject}_{predicate}_{object}"
        knowledge_id = hashlib.sha256(f"rel_{relationship_str}".encode()).hexdigest()[:16]

        relationship = KnowledgeItem(
            knowledge_id=knowledge_id,
            knowledge_type=KnowledgeType.RELATIONSHIP,
            content={
                'subject': subject,
                'predicate': predicate,
                'object': object,
                'properties': properties or {}
            },
            confidence=confidence
        )

        self.relationships[knowledge_id] = relationship
        self.stats["total_relationships"] += 1

        # Add to knowledge graph
        self.knowledge_graph.add_triple(
            subject, predicate, object,
            confidence=confidence,
            source=knowledge_id
        )

        # Update co-occurrence
        self.co_occurrence[subject][object] += 1

        logger.info(f"Stored relationship: {relationship_str}")
        return knowledge_id

    async def query(
        self,
        query_type: KnowledgeType,
        pattern: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[KnowledgeItem]:
        """
        Query semantic memory.

        Args:
            query_type: Type of knowledge to query
            pattern: Query pattern
            limit: Maximum results

        Returns:
            List of matching knowledge items
        """
        self.stats["total_queries"] += 1
        results = []

        # Select appropriate storage
        if query_type == KnowledgeType.FACT:
            storage = self.facts
        elif query_type == KnowledgeType.CONCEPT:
            storage = self.concepts
        elif query_type == KnowledgeType.PROCEDURE:
            storage = self.procedures
        elif query_type == KnowledgeType.RELATIONSHIP:
            storage = self.relationships
        else:
            return []

        # Apply pattern matching
        for item in storage.values():
            if self._matches_pattern(item, pattern):
                results.append(item)
                item.access_count += 1

        # Sort by relevance (access count and confidence)
        results.sort(
            key=lambda x: x.confidence * (1 + np.log1p(x.access_count)),
            reverse=True
        )

        return results[:limit]

    def _matches_pattern(
        self,
        item: KnowledgeItem,
        pattern: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if item matches query pattern."""
        if not pattern:
            return True

        content = item.content if isinstance(item.content, dict) else {'content': item.content}

        for key, value in pattern.items():
            if key in content:
                if isinstance(value, str) and isinstance(content[key], str):
                    # Substring match for strings
                    if value.lower() not in content[key].lower():
                        return False
                elif content[key] != value:
                    return False
            elif key in item.metadata:
                if item.metadata[key] != value:
                    return False
            else:
                return False

        return True

    async def find_related(
        self,
        entity: str,
        max_depth: int = 2,
        limit: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Find entities related to given entity.

        Args:
            entity: Entity to find relations for
            max_depth: Maximum graph traversal depth
            limit: Maximum results

        Returns:
            List of (entity, relevance_score) tuples
        """
        related = []

        # Get neighbors from knowledge graph
        neighbors = self.knowledge_graph.get_neighbors(entity, max_depth)

        for depth, entities in neighbors.items():
            # Score decreases with depth
            relevance = 1.0 / depth

            for related_entity in entities:
                # Boost score based on co-occurrence
                co_occur_boost = min(
                    self.co_occurrence[entity].get(related_entity, 0) / 10,
                    0.5
                )
                score = relevance + co_occur_boost
                related.append((related_entity, score))

        # Sort by relevance
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:limit]

    async def get_domain_knowledge(
        self,
        domain: str,
        category: Optional[str] = None,
        subcategory: Optional[str] = None
    ) -> List[KnowledgeItem]:
        """
        Get knowledge items for a specific domain/category.

        Args:
            domain: Top-level domain
            category: Optional category
            subcategory: Optional subcategory

        Returns:
            List of relevant knowledge items
        """
        results = []

        # Build search pattern
        pattern = {'domain': domain}
        if category:
            pattern['category'] = category
        if subcategory:
            pattern['subcategory'] = subcategory

        # Search all knowledge types
        for storage in [self.facts, self.concepts, self.procedures, self.relationships]:
            for item in storage.values():
                if 'domain' in item.metadata:
                    if self._matches_pattern(item, pattern):
                        results.append(item)

        return results

    async def reason_about(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Reason about a question using semantic knowledge.

        Args:
            question: Question to reason about
            context: Optional context

        Returns:
            Reasoning result with evidence
        """
        # Extract entities from question (simplified)
        entities = [word for word in question.split() if word[0].isupper()]

        evidence = []
        reasoning_path = []

        # Find relevant facts
        for entity in entities:
            # Query knowledge graph
            triples = self.knowledge_graph.query_by_subject(entity)
            triples.extend(self.knowledge_graph.query_by_pattern(object=entity))

            for triple in triples:
                evidence.append({
                    'type': 'triple',
                    'content': triple.to_dict(),
                    'confidence': triple.confidence
                })

            # Find related concepts
            if entity in self.concepts:
                concept = self.concepts[entity]
                evidence.append({
                    'type': 'concept',
                    'content': concept.content,
                    'confidence': concept.confidence
                })

        # Build reasoning path
        if len(entities) >= 2:
            # Try to find path between entities
            for i in range(len(entities) - 1):
                path = self.knowledge_graph.find_path(entities[i], entities[i + 1])
                if path:
                    reasoning_path.append({
                        'from': entities[i],
                        'to': entities[i + 1],
                        'path': path
                    })

        result = {
            'question': question,
            'entities_identified': entities,
            'evidence': evidence,
            'reasoning_path': reasoning_path,
            'confidence': np.mean([e['confidence'] for e in evidence]) if evidence else 0,
            'answer': self._generate_answer(evidence, reasoning_path)
        }

        return result

    def _generate_answer(
        self,
        evidence: List[Dict],
        reasoning_path: List[Dict]
    ) -> str:
        """Generate answer from evidence and reasoning."""
        if not evidence:
            return "No relevant knowledge found."

        # Simplified answer generation
        facts = [e for e in evidence if e['type'] == 'triple']
        if facts:
            # Return most confident fact
            best_fact = max(facts, key=lambda x: x['confidence'])
            triple = best_fact['content']
            return f"{triple['subject']} {triple['predicate']} {triple['object']}"

        concepts = [e for e in evidence if e['type'] == 'concept']
        if concepts:
            concept = concepts[0]['content']
            return concept.get('definition', 'Concept found but no definition available.')

        return "Knowledge found but unable to generate specific answer."

    async def consolidate_knowledge(self) -> Dict[str, int]:
        """
        Consolidate and organize knowledge.

        Returns:
            Consolidation statistics
        """
        stats = {
            'duplicates_removed': 0,
            'conflicts_resolved': 0,
            'clusters_formed': 0
        }

        # Remove duplicate facts
        fact_contents = {}
        for fact_id, fact in list(self.facts.items()):
            content_str = json.dumps(fact.content, sort_keys=True, default=str)
            if content_str in fact_contents:
                # Merge metadata and remove duplicate
                existing_id = fact_contents[content_str]
                self.facts[existing_id].metadata.update(fact.metadata)
                del self.facts[fact_id]
                stats['duplicates_removed'] += 1
            else:
                fact_contents[content_str] = fact_id

        # Cluster related concepts
        if self.concepts:
            concept_names = list(self.concepts.keys())
            for i, concept1 in enumerate(concept_names):
                cluster = [concept1]
                for concept2 in concept_names[i + 1:]:
                    similarity = self.knowledge_graph.compute_similarity(
                        self.concepts[concept1].content.get('name', ''),
                        self.concepts[concept2].content.get('name', '')
                    )
                    if similarity > 0.7:
                        cluster.append(concept2)

                if len(cluster) > 1:
                    cluster_id = f"cluster_{stats['clusters_formed']}"
                    self.concept_clusters[cluster_id] = cluster
                    stats['clusters_formed'] += 1

        logger.info(f"Knowledge consolidation complete: {stats}")
        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get semantic memory statistics.

        Returns:
            Memory statistics
        """
        return {
            **self.stats,
            'graph_nodes': self.knowledge_graph.graph.number_of_nodes(),
            'graph_edges': self.knowledge_graph.graph.number_of_edges(),
            'total_triples': len(self.knowledge_graph.triples),
            'concept_clusters': len(self.concept_clusters),
            'version_history_items': sum(len(v) for v in self.version_history.values()),
            'avg_confidence': np.mean([
                item.confidence
                for storage in [self.facts, self.concepts, self.procedures, self.relationships]
                for item in storage.values()
            ]) if any([self.facts, self.concepts, self.procedures, self.relationships]) else 0
        }