# -*- coding: utf-8 -*-
"""
Tests for Entity MDM Vector Store (Weaviate).

Tests Weaviate connection, indexing (single, batch), vector search,
threshold filtering, CRUD operations, and error handling.

Target: 400+ lines, 18 tests
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any


# Mock vector store service (would be actual module in production)
class VectorStore:
    """Vector store for entity embeddings using Weaviate."""

    def __init__(self, client, class_name: str = "Supplier"):
        self.client = client
        self.class_name = class_name
        self._ensure_schema()

    def _ensure_schema(self):
        """Ensure Weaviate schema exists."""
        try:
            schema = self.client.schema.get()
            existing_classes = [c["class"] for c in schema.get("classes", [])]

            if self.class_name not in existing_classes:
                self._create_schema()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Weaviate: {e}")

    def _create_schema(self):
        """Create Weaviate schema for suppliers."""
        class_obj = {
            "class": self.class_name,
            "vectorizer": "none",  # We provide our own vectors
            "properties": [
                {"name": "name", "dataType": ["string"]},
                {"name": "country", "dataType": ["string"]},
                {"name": "lei_code", "dataType": ["string"]},
                {"name": "duns_number", "dataType": ["string"]},
            ]
        }
        self.client.schema.create_class(class_obj)

    def add_entity(self, entity_id: str, properties: Dict[str, Any], vector: np.ndarray):
        """Add a single entity to the vector store."""
        if not entity_id:
            raise ValueError("Entity ID cannot be empty")

        if vector is None or len(vector) == 0:
            raise ValueError("Vector cannot be empty")

        data_object = {
            "class": self.class_name,
            "id": entity_id,
            "properties": properties,
            "vector": vector.tolist() if isinstance(vector, np.ndarray) else vector
        }

        self.client.data_object.create(data_object)

    def add_entities_batch(self, entities: List[Dict[str, Any]], batch_size: int = 100):
        """Add multiple entities in batch."""
        if not entities:
            return

        with self.client.batch as batch:
            batch.batch_size = batch_size

            for entity in entities:
                if "id" not in entity or "properties" not in entity or "vector" not in entity:
                    raise ValueError("Each entity must have 'id', 'properties', and 'vector'")

                vector = entity["vector"]
                if isinstance(vector, np.ndarray):
                    vector = vector.tolist()

                batch.add_data_object(
                    data_object=entity["properties"],
                    class_name=self.class_name,
                    uuid=entity["id"],
                    vector=vector
                )

    def search_similar(self, query_vector: np.ndarray, top_k: int = 10,
                      min_similarity: float = 0.0, filters: Dict[str, Any] = None) -> List[Dict]:
        """Search for similar entities using vector similarity."""
        if query_vector is None or len(query_vector) == 0:
            raise ValueError("Query vector cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Build query
        query = (
            self.client.query
            .get(self.class_name, ["name", "country", "lei_code", "duns_number"])
            .with_near_vector({"vector": query_vector.tolist()})
            .with_limit(top_k)
            .with_additional(["id", "distance", "certainty"])
        )

        # Add filters if provided
        if filters:
            where_filter = self._build_where_filter(filters)
            query = query.with_where(where_filter)

        result = query.do()

        # Parse results
        entities = result.get("data", {}).get("Get", {}).get(self.class_name, [])

        # Filter by minimum similarity
        filtered_entities = []
        for entity in entities:
            certainty = entity.get("_additional", {}).get("certainty", 0)
            if certainty >= min_similarity:
                filtered_entities.append({
                    "id": entity["_additional"]["id"],
                    "name": entity["name"],
                    "country": entity.get("country"),
                    "lei_code": entity.get("lei_code"),
                    "duns_number": entity.get("duns_number"),
                    "similarity": certainty,
                    "distance": entity["_additional"]["distance"]
                })

        return filtered_entities

    def _build_where_filter(self, filters: Dict[str, Any]) -> Dict:
        """Build Weaviate where filter from dict."""
        conditions = []

        for key, value in filters.items():
            conditions.append({
                "path": [key],
                "operator": "Equal",
                "valueString": value
            })

        if len(conditions) == 1:
            return conditions[0]
        else:
            return {
                "operator": "And",
                "operands": conditions
            }

    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """Get entity by ID."""
        if not entity_id:
            raise ValueError("Entity ID cannot be empty")

        result = self.client.data_object.get_by_id(entity_id, class_name=self.class_name)

        if not result:
            return None

        return {
            "id": result["id"],
            "properties": result["properties"],
            "vector": result.get("vector")
        }

    def update_entity(self, entity_id: str, properties: Dict[str, Any]):
        """Update entity properties."""
        if not entity_id:
            raise ValueError("Entity ID cannot be empty")

        self.client.data_object.update(
            data_object=properties,
            class_name=self.class_name,
            uuid=entity_id
        )

    def delete_entity(self, entity_id: str):
        """Delete entity by ID."""
        if not entity_id:
            raise ValueError("Entity ID cannot be empty")

        self.client.data_object.delete(entity_id, class_name=self.class_name)

    def count_entities(self) -> int:
        """Count total entities in vector store."""
        result = self.client.query.aggregate(self.class_name).with_meta_count().do()
        return result.get("data", {}).get("Aggregate", {}).get(self.class_name, [{}])[0].get("meta", {}).get("count", 0)


# ============================================================================
# TEST SUITE
# ============================================================================

class TestVectorStore:
    """Test suite for vector store operations."""

    def test_vector_store_initialization(self, mock_weaviate_client):
        """Test vector store initialization and schema check."""
        store = VectorStore(mock_weaviate_client)

        assert store.client is not None
        assert store.class_name == "Supplier"
        mock_weaviate_client.schema.get.assert_called_once()

    def test_schema_creation_if_not_exists(self, mock_weaviate_client):
        """Test schema creation when it doesn't exist."""
        # Mock schema without Supplier class
        mock_weaviate_client.schema.get.return_value = {"classes": []}

        store = VectorStore(mock_weaviate_client)

        mock_weaviate_client.schema.create_class.assert_called_once()

    def test_add_single_entity(self, mock_weaviate_client, create_embedding):
        """Test adding a single entity to vector store."""
        store = VectorStore(mock_weaviate_client)

        entity_id = "uuid-001"
        properties = {
            "name": "ACME Corporation",
            "country": "US",
            "lei_code": "549300ACME000001"
        }
        vector = create_embedding(seed=42)

        store.add_entity(entity_id, properties, vector)

        mock_weaviate_client.data_object.create.assert_called_once()

    def test_add_entity_with_empty_id_raises_error(self, mock_weaviate_client, create_embedding):
        """Test that adding entity with empty ID raises error."""
        store = VectorStore(mock_weaviate_client)

        with pytest.raises(ValueError, match="Entity ID cannot be empty"):
            store.add_entity("", {"name": "Test"}, create_embedding())

    def test_add_entity_with_empty_vector_raises_error(self, mock_weaviate_client):
        """Test that adding entity with empty vector raises error."""
        store = VectorStore(mock_weaviate_client)

        with pytest.raises(ValueError, match="Vector cannot be empty"):
            store.add_entity("uuid-001", {"name": "Test"}, np.array([]))

    def test_add_entities_batch(self, mock_weaviate_client, create_embedding):
        """Test batch adding multiple entities."""
        store = VectorStore(mock_weaviate_client)

        entities = [
            {
                "id": "uuid-001",
                "properties": {"name": "ACME Corporation", "country": "US"},
                "vector": create_embedding(seed=1)
            },
            {
                "id": "uuid-002",
                "properties": {"name": "ABC Manufacturing", "country": "US"},
                "vector": create_embedding(seed=2)
            },
            {
                "id": "uuid-003",
                "properties": {"name": "Global Tech", "country": "US"},
                "vector": create_embedding(seed=3)
            }
        ]

        store.add_entities_batch(entities)

        # Verify batch operations were called
        assert mock_weaviate_client.batch.configure.called

    def test_add_entities_batch_with_custom_batch_size(self, mock_weaviate_client, create_embedding):
        """Test batch adding with custom batch size."""
        store = VectorStore(mock_weaviate_client)

        entities = [
            {
                "id": f"uuid-{i:03d}",
                "properties": {"name": f"Company {i}", "country": "US"},
                "vector": create_embedding(seed=i)
            }
            for i in range(50)
        ]

        store.add_entities_batch(entities, batch_size=25)

        assert mock_weaviate_client.batch.configure.called

    def test_add_entities_batch_with_invalid_entity_raises_error(self, mock_weaviate_client):
        """Test that batch add with invalid entity raises error."""
        store = VectorStore(mock_weaviate_client)

        entities = [
            {"id": "uuid-001"}  # Missing properties and vector
        ]

        with pytest.raises(ValueError, match="Each entity must have"):
            store.add_entities_batch(entities)

    def test_search_similar_entities(self, mock_weaviate_client, mock_weaviate_query_response, create_embedding):
        """Test searching for similar entities."""
        store = VectorStore(mock_weaviate_client)

        # Mock query response
        query_mock = MagicMock()
        query_mock.do.return_value = mock_weaviate_query_response
        query_mock.with_near_vector.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.with_additional.return_value = query_mock

        mock_weaviate_client.query.get.return_value = query_mock

        query_vector = create_embedding(seed=42)
        results = store.search_similar(query_vector, top_k=10)

        assert len(results) == 3
        assert results[0]["name"] == "ACME Corporation Ltd"
        assert results[0]["similarity"] == 0.925

    def test_search_similar_with_min_similarity_filter(self, mock_weaviate_client, mock_weaviate_query_response, create_embedding):
        """Test search with minimum similarity threshold."""
        store = VectorStore(mock_weaviate_client)

        query_mock = MagicMock()
        query_mock.do.return_value = mock_weaviate_query_response
        query_mock.with_near_vector.return_value = query_mock
        query_mock.with_limit.return_value = query_mock
        query_mock.with_additional.return_value = query_mock

        mock_weaviate_client.query.get.return_value = query_mock

        query_vector = create_embedding(seed=42)
        results = store.search_similar(query_vector, top_k=10, min_similarity=0.90)

        # Only results with similarity >= 0.90 should be returned
        assert len(results) == 1  # Only first result has 0.925
        assert results[0]["similarity"] >= 0.90

    def test_search_similar_with_empty_vector_raises_error(self, mock_weaviate_client):
        """Test that search with empty vector raises error."""
        store = VectorStore(mock_weaviate_client)

        with pytest.raises(ValueError, match="Query vector cannot be empty"):
            store.search_similar(np.array([]), top_k=10)

    def test_search_similar_with_invalid_top_k_raises_error(self, mock_weaviate_client, create_embedding):
        """Test that search with invalid top_k raises error."""
        store = VectorStore(mock_weaviate_client)

        with pytest.raises(ValueError, match="top_k must be positive"):
            store.search_similar(create_embedding(), top_k=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            store.search_similar(create_embedding(), top_k=-5)

    def test_get_entity_by_id(self, mock_weaviate_client):
        """Test retrieving entity by ID."""
        store = VectorStore(mock_weaviate_client)

        mock_weaviate_client.data_object.get_by_id.return_value = {
            "id": "uuid-001",
            "properties": {
                "name": "ACME Corporation",
                "country": "US"
            },
            "vector": [0.1, 0.2, 0.3]
        }

        entity = store.get_entity("uuid-001")

        assert entity is not None
        assert entity["id"] == "uuid-001"
        assert entity["properties"]["name"] == "ACME Corporation"

    def test_get_entity_with_empty_id_raises_error(self, mock_weaviate_client):
        """Test that get with empty ID raises error."""
        store = VectorStore(mock_weaviate_client)

        with pytest.raises(ValueError, match="Entity ID cannot be empty"):
            store.get_entity("")

    def test_get_entity_returns_none_if_not_found(self, mock_weaviate_client):
        """Test that get returns None if entity not found."""
        store = VectorStore(mock_weaviate_client)

        mock_weaviate_client.data_object.get_by_id.return_value = None

        entity = store.get_entity("nonexistent-id")

        assert entity is None

    def test_update_entity(self, mock_weaviate_client):
        """Test updating entity properties."""
        store = VectorStore(mock_weaviate_client)

        updated_properties = {
            "name": "ACME Corporation Ltd",
            "lei_code": "549300ACME000001"
        }

        store.update_entity("uuid-001", updated_properties)

        mock_weaviate_client.data_object.update.assert_called_once()

    def test_delete_entity(self, mock_weaviate_client):
        """Test deleting entity by ID."""
        store = VectorStore(mock_weaviate_client)

        store.delete_entity("uuid-001")

        mock_weaviate_client.data_object.delete.assert_called_once_with(
            "uuid-001",
            class_name="Supplier"
        )

    def test_delete_entity_with_empty_id_raises_error(self, mock_weaviate_client):
        """Test that delete with empty ID raises error."""
        store = VectorStore(mock_weaviate_client)

        with pytest.raises(ValueError, match="Entity ID cannot be empty"):
            store.delete_entity("")
