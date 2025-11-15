"""
Knowledge Graph Integration for GreenLang RAG System

Neo4j-based knowledge graph for enhanced retrieval and reasoning.
Supports entity extraction, relationship mapping, and graph-based retrieval.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    id: str
    type: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: Optional[str] = None
    provenance_hash: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
        if not self.provenance_hash:
            self._calculate_provenance()

    def _generate_id(self) -> str:
        """Generate unique ID for entity"""
        id_str = f"{self.type}:{self.name}"
        return hashlib.md5(id_str.encode()).hexdigest()[:16]

    def _calculate_provenance(self):
        """Calculate SHA-256 hash for audit trail"""
        prov_str = f"{self.type}:{self.name}:{json.dumps(self.properties, sort_keys=True)}"
        self.provenance_hash = hashlib.sha256(prov_str.encode()).hexdigest()


@dataclass
class Relationship:
    """Represents a relationship between entities"""
    id: str
    type: str
    source_id: str
    target_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    provenance_hash: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
        if not self.provenance_hash:
            self._calculate_provenance()

    def _generate_id(self) -> str:
        """Generate unique ID for relationship"""
        id_str = f"{self.source_id}-{self.type}-{self.target_id}"
        return hashlib.md5(id_str.encode()).hexdigest()[:16]

    def _calculate_provenance(self):
        """Calculate SHA-256 hash for audit trail"""
        prov_str = f"{self.type}:{self.source_id}:{self.target_id}:{json.dumps(self.properties, sort_keys=True)}"
        self.provenance_hash = hashlib.sha256(prov_str.encode()).hexdigest()


class EntityExtractor:
    """
    Extract entities from text using NLP techniques
    Supports multiple entity types relevant to GreenLang
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        custom_patterns: Optional[Dict[str, List[str]]] = None,
        confidence_threshold: float = 0.8
    ):
        self.model_name = model_name
        self.custom_patterns = custom_patterns or {}
        self.confidence_threshold = confidence_threshold
        self.nlp = None
        self._load_model()

    def _load_model(self):
        """Load NLP model for entity extraction"""
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except ImportError:
            logger.warning("spaCy not installed, using pattern-based extraction")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")

    def extract(self, text: str, source: Optional[str] = None) -> List[Entity]:
        """
        Extract entities from text

        Args:
            text: Text to extract entities from
            source: Source identifier for provenance

        Returns:
            List of extracted entities
        """
        entities = []

        # Use spaCy if available
        if self.nlp:
            entities.extend(self._extract_with_spacy(text, source))

        # Add custom pattern extraction
        entities.extend(self._extract_with_patterns(text, source))

        # GreenLang-specific entity extraction
        entities.extend(self._extract_greenlang_entities(text, source))

        # Deduplicate entities
        unique_entities = {}
        for entity in entities:
            if entity.confidence >= self.confidence_threshold:
                key = f"{entity.type}:{entity.name}"
                if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
                    unique_entities[key] = entity

        return list(unique_entities.values())

    def _extract_with_spacy(self, text: str, source: Optional[str]) -> List[Entity]:
        """Extract entities using spaCy NER"""
        entities = []
        doc = self.nlp(text)

        for ent in doc.ents:
            entity = Entity(
                id="",
                type=ent.label_,
                name=ent.text,
                properties={
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "spacy_label": ent.label_
                },
                confidence=0.9,  # spaCy confidence estimate
                source=source
            )
            entities.append(entity)

        return entities

    def _extract_with_patterns(self, text: str, source: Optional[str]) -> List[Entity]:
        """Extract entities using custom patterns"""
        entities = []

        for entity_type, patterns in self.custom_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = Entity(
                        id="",
                        type=entity_type,
                        name=match.group(0),
                        properties={
                            "pattern": pattern,
                            "start": match.start(),
                            "end": match.end()
                        },
                        confidence=0.8,  # Pattern match confidence
                        source=source
                    )
                    entities.append(entity)

        return entities

    def _extract_greenlang_entities(self, text: str, source: Optional[str]) -> List[Entity]:
        """Extract GreenLang-specific entities"""
        entities = []

        # Carbon emission values
        carbon_pattern = r'\b(\d+(?:\.\d+)?)\s*(kg|ton|tonne|t|mt)?\s*CO2(?:e)?\b'
        matches = re.finditer(carbon_pattern, text, re.IGNORECASE)
        for match in matches:
            value = match.group(1)
            unit = match.group(2) or "kg"
            entity = Entity(
                id="",
                type="EMISSION",
                name=f"{value} {unit} CO2e",
                properties={
                    "value": float(value),
                    "unit": unit,
                    "type": "carbon_emission"
                },
                confidence=0.95,
                source=source
            )
            entities.append(entity)

        # Regulatory frameworks
        frameworks = [
            "CSRD", "ESRS", "TCFD", "GRI", "CDP", "SASB", "ISSB",
            "EU Taxonomy", "SFDR", "SEC Climate", "SBTi"
        ]
        for framework in frameworks:
            if framework.lower() in text.lower():
                entity = Entity(
                    id="",
                    type="FRAMEWORK",
                    name=framework,
                    properties={"category": "regulatory"},
                    confidence=1.0,
                    source=source
                )
                entities.append(entity)

        # Scope categories
        scope_pattern = r'\b(Scope\s*[123])\b'
        matches = re.finditer(scope_pattern, text, re.IGNORECASE)
        for match in matches:
            entity = Entity(
                id="",
                type="SCOPE",
                name=match.group(1),
                properties={"scope_number": match.group(1)[-1]},
                confidence=1.0,
                source=source
            )
            entities.append(entity)

        # Organizations/Companies
        org_pattern = r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:Inc|Corp|Ltd|LLC|GmbH|AG|SA|NV|PLC)\b'
        matches = re.finditer(org_pattern, text)
        for match in matches:
            entity = Entity(
                id="",
                type="ORGANIZATION",
                name=match.group(0),
                properties={"company_name": match.group(1)},
                confidence=0.85,
                source=source
            )
            entities.append(entity)

        return entities


class RelationshipExtractor:
    """
    Extract relationships between entities
    Uses dependency parsing and pattern matching
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self._load_model()

    def _load_model(self):
        """Load NLP model for relationship extraction"""
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model for relationships: {self.model_name}")
        except:
            logger.warning("spaCy not available for relationship extraction")

    def extract(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Extract relationships between entities

        Args:
            text: Source text
            entities: List of entities found in text

        Returns:
            List of relationships
        """
        relationships = []

        # Create entity lookup
        entity_lookup = {entity.name.lower(): entity for entity in entities}

        # Use spaCy if available
        if self.nlp:
            relationships.extend(
                self._extract_with_dependency(text, entity_lookup)
            )

        # Pattern-based extraction
        relationships.extend(
            self._extract_with_patterns(text, entity_lookup)
        )

        # GreenLang-specific relationships
        relationships.extend(
            self._extract_greenlang_relationships(text, entity_lookup)
        )

        # Deduplicate
        unique_rels = {}
        for rel in relationships:
            key = f"{rel.source_id}-{rel.type}-{rel.target_id}"
            if key not in unique_rels or rel.confidence > unique_rels[key].confidence:
                unique_rels[key] = rel

        return list(unique_rels.values())

    def _extract_with_dependency(
        self,
        text: str,
        entity_lookup: Dict[str, Entity]
    ) -> List[Relationship]:
        """Extract relationships using dependency parsing"""
        relationships = []
        doc = self.nlp(text)

        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj"]:
                # Check if token and its head are entities
                if (token.text.lower() in entity_lookup and
                    token.head.text.lower() in entity_lookup):

                    source = entity_lookup[token.text.lower()]
                    target = entity_lookup[token.head.text.lower()]

                    rel = Relationship(
                        id="",
                        type=token.dep_,
                        source_id=source.id,
                        target_id=target.id,
                        properties={
                            "dependency": token.dep_,
                            "verb": token.head.text
                        },
                        confidence=0.7
                    )
                    relationships.append(rel)

        return relationships

    def _extract_with_patterns(
        self,
        text: str,
        entity_lookup: Dict[str, Entity]
    ) -> List[Relationship]:
        """Extract relationships using patterns"""
        relationships = []

        # Pattern: Entity1 "verb" Entity2
        patterns = [
            (r'({entity1})\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*({entity2})', 'IS_A'),
            (r'({entity1})\s+(?:has|have|had)\s+({entity2})', 'HAS'),
            (r'({entity1})\s+(?:emits?|produced?|generates?)\s+({entity2})', 'EMITS'),
            (r'({entity1})\s+(?:reports?|discloses?)\s+(?:to|under)\s+({entity2})', 'REPORTS_TO'),
            (r'({entity1})\s+(?:complies?|aligns?)\s+(?:with)\s+({entity2})', 'COMPLIES_WITH'),
        ]

        for entity1 in entity_lookup.values():
            for entity2 in entity_lookup.values():
                if entity1.id == entity2.id:
                    continue

                for pattern_template, rel_type in patterns:
                    pattern = pattern_template.format(
                        entity1=re.escape(entity1.name),
                        entity2=re.escape(entity2.name)
                    )

                    if re.search(pattern, text, re.IGNORECASE):
                        rel = Relationship(
                            id="",
                            type=rel_type,
                            source_id=entity1.id,
                            target_id=entity2.id,
                            properties={"pattern_matched": pattern_template},
                            confidence=0.8
                        )
                        relationships.append(rel)

        return relationships

    def _extract_greenlang_relationships(
        self,
        text: str,
        entity_lookup: Dict[str, Entity]
    ) -> List[Relationship]:
        """Extract GreenLang-specific relationships"""
        relationships = []

        # Organization -> Emission relationships
        for entity1 in entity_lookup.values():
            if entity1.type == "ORGANIZATION":
                for entity2 in entity_lookup.values():
                    if entity2.type == "EMISSION":
                        # Check proximity
                        if entity1.name in text and entity2.name in text:
                            idx1 = text.find(entity1.name)
                            idx2 = text.find(entity2.name)
                            if abs(idx1 - idx2) < 100:  # Within 100 chars
                                rel = Relationship(
                                    id="",
                                    type="EMITS",
                                    source_id=entity1.id,
                                    target_id=entity2.id,
                                    properties={"proximity": abs(idx1 - idx2)},
                                    confidence=0.75
                                )
                                relationships.append(rel)

        # Framework compliance relationships
        for entity1 in entity_lookup.values():
            if entity1.type == "ORGANIZATION":
                for entity2 in entity_lookup.values():
                    if entity2.type == "FRAMEWORK":
                        pattern = f"{entity1.name}.*(?:complies?|reports?|aligns?).*{entity2.name}"
                        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                            rel = Relationship(
                                id="",
                                type="COMPLIES_WITH",
                                source_id=entity1.id,
                                target_id=entity2.id,
                                properties={"compliance_type": "regulatory"},
                                confidence=0.85
                            )
                            relationships.append(rel)

        return relationships


class Neo4jConnector:
    """
    Neo4j database connector for knowledge graph operations
    Handles connection, queries, and transactions
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        encrypted: bool = False
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.encrypted = encrypted
        self.driver = None
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                encrypted=self.encrypted
            )
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.uri}")
        except ImportError:
            raise ImportError("neo4j not installed. Install with: pip install neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

    def create_entity(self, entity: Entity) -> bool:
        """Create or update entity in Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MERGE (e:Entity {id: $id})
                SET e.type = $type,
                    e.name = $name,
                    e.confidence = $confidence,
                    e.source = $source,
                    e.provenance_hash = $provenance_hash,
                    e.properties = $properties,
                    e.updated_at = timestamp()
                RETURN e
                """
                session.run(
                    query,
                    id=entity.id,
                    type=entity.type,
                    name=entity.name,
                    confidence=entity.confidence,
                    source=entity.source,
                    provenance_hash=entity.provenance_hash,
                    properties=json.dumps(entity.properties)
                )
                return True
        except Exception as e:
            logger.error(f"Failed to create entity: {e}")
            return False

    def create_relationship(self, relationship: Relationship) -> bool:
        """Create or update relationship in Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (s:Entity {id: $source_id})
                MATCH (t:Entity {id: $target_id})
                MERGE (s)-[r:RELATIONSHIP {id: $id}]->(t)
                SET r.type = $type,
                    r.confidence = $confidence,
                    r.provenance_hash = $provenance_hash,
                    r.properties = $properties,
                    r.updated_at = timestamp()
                RETURN r
                """
                session.run(
                    query,
                    id=relationship.id,
                    source_id=relationship.source_id,
                    target_id=relationship.target_id,
                    type=relationship.type,
                    confidence=relationship.confidence,
                    provenance_hash=relationship.provenance_hash,
                    properties=json.dumps(relationship.properties)
                )
                return True
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False

    def query(self, cypher: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute Cypher query"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []


class KnowledgeGraphStore:
    """
    Main knowledge graph store for RAG system
    Integrates entity/relationship extraction with Neo4j storage
    """

    def __init__(
        self,
        neo4j_config: Dict[str, Any],
        entity_extractor: Optional[EntityExtractor] = None,
        relationship_extractor: Optional[RelationshipExtractor] = None
    ):
        self.neo4j = Neo4jConnector(**neo4j_config)
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.relationship_extractor = relationship_extractor or RelationshipExtractor()

        # Create indices for performance
        self._create_indices()

    def _create_indices(self):
        """Create Neo4j indices for performance"""
        indices = [
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX relationship_id IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.id)",
            "CREATE INDEX relationship_type IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.type)"
        ]

        for index_query in indices:
            try:
                self.neo4j.query(index_query)
            except:
                pass  # Index might already exist

    def add_document(
        self,
        text: str,
        source: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process document and add to knowledge graph

        Args:
            text: Document text
            source: Source identifier
            metadata: Additional metadata

        Returns:
            Processing statistics
        """
        # Extract entities
        entities = self.entity_extractor.extract(text, source)

        # Extract relationships
        relationships = self.relationship_extractor.extract(text, entities)

        # Store in Neo4j
        entities_created = 0
        for entity in entities:
            if self.neo4j.create_entity(entity):
                entities_created += 1

        relationships_created = 0
        for relationship in relationships:
            if self.neo4j.create_relationship(relationship):
                relationships_created += 1

        # Store document metadata
        doc_query = """
        CREATE (d:Document {
            source: $source,
            metadata: $metadata,
            entity_count: $entity_count,
            relationship_count: $relationship_count,
            processed_at: timestamp()
        })
        RETURN d
        """
        self.neo4j.query(
            doc_query,
            {
                "source": source,
                "metadata": json.dumps(metadata or {}),
                "entity_count": len(entities),
                "relationship_count": len(relationships)
            }
        )

        return {
            "source": source,
            "entities_extracted": len(entities),
            "entities_created": entities_created,
            "relationships_extracted": len(relationships),
            "relationships_created": relationships_created
        }

    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Entity]:
        """Search for entities in the knowledge graph"""
        cypher = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $query
        """

        if entity_type:
            cypher += " AND e.type = $entity_type"

        cypher += """
        RETURN e
        ORDER BY e.confidence DESC
        LIMIT $limit
        """

        results = self.neo4j.query(
            cypher,
            {
                "query": query,
                "entity_type": entity_type,
                "limit": limit
            }
        )

        entities = []
        for result in results:
            entity_data = result['e']
            entity = Entity(
                id=entity_data['id'],
                type=entity_data['type'],
                name=entity_data['name'],
                properties=json.loads(entity_data.get('properties', '{}')),
                confidence=entity_data.get('confidence', 1.0),
                source=entity_data.get('source'),
                provenance_hash=entity_data.get('provenance_hash')
            )
            entities.append(entity)

        return entities

    def get_entity_context(
        self,
        entity_id: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get context around an entity (neighboring entities and relationships)

        Args:
            entity_id: Entity ID to get context for
            max_depth: Maximum graph traversal depth

        Returns:
            Dictionary with entity, related entities, and relationships
        """
        # Get entity and its relationships
        cypher = """
        MATCH (e:Entity {id: $entity_id})
        OPTIONAL MATCH path = (e)-[r:RELATIONSHIP*1..""" + str(max_depth) + """]->(related:Entity)
        RETURN e, collect(DISTINCT related) as related_entities,
               collect(DISTINCT relationships(path)) as relationships
        """

        results = self.neo4j.query(cypher, {"entity_id": entity_id})

        if not results:
            return {}

        result = results[0]

        # Parse entity
        entity_data = result['e']
        entity = Entity(
            id=entity_data['id'],
            type=entity_data['type'],
            name=entity_data['name'],
            properties=json.loads(entity_data.get('properties', '{}')),
            confidence=entity_data.get('confidence', 1.0)
        )

        # Parse related entities
        related = []
        for rel_data in result['related_entities']:
            if rel_data:
                related.append(Entity(
                    id=rel_data['id'],
                    type=rel_data['type'],
                    name=rel_data['name'],
                    properties=json.loads(rel_data.get('properties', '{}')),
                    confidence=rel_data.get('confidence', 1.0)
                ))

        return {
            "entity": entity,
            "related_entities": related,
            "relationship_count": len(result['relationships'])
        }

    def close(self):
        """Close knowledge graph connections"""
        self.neo4j.close()


class GraphRetrieval:
    """
    Graph-based retrieval strategy for RAG
    Combines graph traversal with vector search
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraphStore,
        vector_store: Any,
        embedding_generator: Any
    ):
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_graph: bool = True,
        use_vector: bool = True,
        graph_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Retrieve information using both graph and vector search

        Args:
            query: Search query
            top_k: Number of results
            use_graph: Whether to use graph search
            use_vector: Whether to use vector search
            graph_depth: Graph traversal depth

        Returns:
            Combined retrieval results
        """
        results = {
            "query": query,
            "graph_results": [],
            "vector_results": [],
            "combined_results": [],
            "provenance_hash": None
        }

        # Graph-based retrieval
        if use_graph:
            # Extract entities from query
            query_entities = self.knowledge_graph.entity_extractor.extract(query)

            graph_contexts = []
            for entity in query_entities:
                # Search for matching entities
                matching = self.knowledge_graph.search_entities(
                    entity.name,
                    entity.type,
                    limit=5
                )

                # Get context for each match
                for match in matching:
                    context = self.knowledge_graph.get_entity_context(
                        match.id,
                        graph_depth
                    )
                    if context:
                        graph_contexts.append(context)

            results["graph_results"] = graph_contexts

        # Vector-based retrieval
        if use_vector:
            query_embedding = self.embedding_generator.embed_query(query)
            documents, scores = self.vector_store.similarity_search(
                query_embedding,
                top_k
            )

            results["vector_results"] = [
                {"document": doc, "score": score}
                for doc, score in zip(documents, scores)
            ]

        # Combine results
        combined = self._combine_results(
            results["graph_results"],
            results["vector_results"],
            top_k
        )
        results["combined_results"] = combined

        # Calculate provenance
        prov_str = f"{query}:{json.dumps(combined, sort_keys=True, default=str)}"
        results["provenance_hash"] = hashlib.sha256(prov_str.encode()).hexdigest()

        return results

    def _combine_results(
        self,
        graph_results: List[Dict],
        vector_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Combine graph and vector results"""
        combined = []

        # Add graph results with boosted scores
        for graph_result in graph_results:
            combined.append({
                "type": "graph",
                "content": graph_result,
                "score": 0.9,  # High confidence for graph results
                "source": "knowledge_graph"
            })

        # Add vector results
        for vector_result in vector_results:
            combined.append({
                "type": "vector",
                "content": vector_result["document"],
                "score": vector_result["score"],
                "source": "vector_store"
            })

        # Sort by score and return top k
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]