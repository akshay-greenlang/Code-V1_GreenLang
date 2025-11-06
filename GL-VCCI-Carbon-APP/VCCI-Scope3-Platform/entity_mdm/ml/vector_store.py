"""
Weaviate vector store for entity resolution.

This module implements vector storage and retrieval using Weaviate,
including batch indexing, similarity search, and entity management.

Author: GreenLang AI
Phase: 5 - Entity Resolution ML
"""

from typing import List, Optional, Dict, Any, Tuple
import logging
import time
from datetime import datetime
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from weaviate.util import generate_uuid5
import numpy as np

from entity_mdm.ml.config import MLConfig, WeaviateConfig
from entity_mdm.ml.embeddings import EmbeddingPipeline
from entity_mdm.ml.exceptions import VectorStoreException

logger = logging.getLogger(__name__)


class SupplierEntity:
    """Data model for supplier entities in vector store."""

    def __init__(
        self,
        entity_id: str,
        name: str,
        normalized_name: str,
        address: Optional[str] = None,
        country: Optional[str] = None,
        tax_id: Optional[str] = None,
        website: Optional[str] = None,
        industry: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize supplier entity.

        Args:
            entity_id: Unique entity identifier
            name: Original supplier name
            normalized_name: Normalized supplier name
            address: Supplier address
            country: Country code
            tax_id: Tax identification number
            website: Company website
            industry: Industry classification
            metadata: Additional metadata
        """
        self.entity_id = entity_id
        self.name = name
        self.normalized_name = normalized_name
        self.address = address
        self.country = country
        self.tax_id = tax_id
        self.website = website
        self.industry = industry
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "normalized_name": self.normalized_name,
            "address": self.address,
            "country": self.country,
            "tax_id": self.tax_id,
            "website": self.website,
            "industry": self.industry,
            "metadata": self.metadata,
            "indexed_at": datetime.utcnow().isoformat(),
        }

    def get_search_text(self) -> str:
        """
        Get combined text for embedding.

        Returns:
            Combined text representation for similarity search
        """
        parts = [self.normalized_name]
        if self.address:
            parts.append(self.address)
        if self.country:
            parts.append(self.country)
        if self.industry:
            parts.append(self.industry)
        return " | ".join(parts)


class VectorStore:
    """
    Weaviate-based vector store for entity resolution.

    This class handles:
    - Schema creation and management
    - Batch indexing of entities
    - Vector similarity search
    - CRUD operations for entities
    """

    COLLECTION_NAME = "SupplierEntity"
    BATCH_SIZE = 1000

    def __init__(
        self,
        config: Optional[MLConfig] = None,
        embedding_pipeline: Optional[EmbeddingPipeline] = None,
    ) -> None:
        """
        Initialize vector store.

        Args:
            config: ML configuration object
            embedding_pipeline: Embedding pipeline instance. If None, creates new one.
        """
        self.config = config or MLConfig()
        self.weaviate_config: WeaviateConfig = self.config.weaviate
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline(self.config)

        # Initialize client
        self._client: Optional[weaviate.WeaviateClient] = None
        self._connect()

        # Ensure schema exists
        self._ensure_schema()

        logger.info(f"Initialized VectorStore with collection={self.COLLECTION_NAME}")

    def _connect(self) -> None:
        """
        Connect to Weaviate instance.

        Raises:
            VectorStoreException: If connection fails
        """
        try:
            if self.weaviate_config.use_embedded:
                # Embedded Weaviate for development
                self._client = weaviate.WeaviateClient(
                    embedded_options=weaviate.embedded.EmbeddedOptions(
                        persistence_data_path=str(
                            self.weaviate_config.persistence_data_path
                        )
                        if self.weaviate_config.persistence_data_path
                        else None,
                    )
                )
            else:
                # Connect to external Weaviate instance
                auth_config = None
                if self.weaviate_config.auth_client_secret:
                    auth_config = weaviate.auth.AuthApiKey(
                        self.weaviate_config.auth_client_secret
                    )

                self._client = weaviate.connect_to_custom(
                    http_host=self.weaviate_config.host,
                    http_port=self.weaviate_config.port,
                    http_secure=(self.weaviate_config.scheme == "https"),
                    grpc_host=self.weaviate_config.host,
                    grpc_port=self.weaviate_config.grpc_port,
                    grpc_secure=(self.weaviate_config.scheme == "https"),
                    auth_credentials=auth_config,
                )

            # Wait for Weaviate to be ready
            timeout = self.weaviate_config.startup_period
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self._client.is_ready():
                    logger.info("Connected to Weaviate")
                    return
                time.sleep(0.5)

            raise VectorStoreException(
                operation="connect",
                message="Weaviate not ready within timeout period",
            )

        except Exception as e:
            raise VectorStoreException(
                operation="connect",
                message=f"Failed to connect to Weaviate: {e}",
            )

    def _ensure_schema(self) -> None:
        """
        Ensure the schema exists in Weaviate.

        Creates the collection if it doesn't exist.

        Raises:
            VectorStoreException: If schema creation fails
        """
        try:
            # Check if collection exists
            if self._client.collections.exists(self.COLLECTION_NAME):
                logger.info(f"Collection {self.COLLECTION_NAME} already exists")
                return

            # Create collection
            self._client.collections.create(
                name=self.COLLECTION_NAME,
                description="Supplier entities for entity resolution",
                vectorizer_config=Configure.Vectorizer.none(),  # We provide vectors
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric="cosine",
                    ef_construction=128,
                    max_connections=64,
                ),
                properties=[
                    Property(
                        name="entity_id",
                        data_type=DataType.TEXT,
                        description="Unique entity identifier",
                        index_filterable=True,
                        index_searchable=False,
                    ),
                    Property(
                        name="name",
                        data_type=DataType.TEXT,
                        description="Original supplier name",
                        index_searchable=True,
                    ),
                    Property(
                        name="normalized_name",
                        data_type=DataType.TEXT,
                        description="Normalized supplier name",
                        index_searchable=True,
                    ),
                    Property(
                        name="address",
                        data_type=DataType.TEXT,
                        description="Supplier address",
                        index_searchable=True,
                    ),
                    Property(
                        name="country",
                        data_type=DataType.TEXT,
                        description="Country code",
                        index_filterable=True,
                    ),
                    Property(
                        name="tax_id",
                        data_type=DataType.TEXT,
                        description="Tax identification number",
                        index_filterable=True,
                    ),
                    Property(
                        name="website",
                        data_type=DataType.TEXT,
                        description="Company website",
                    ),
                    Property(
                        name="industry",
                        data_type=DataType.TEXT,
                        description="Industry classification",
                        index_filterable=True,
                    ),
                    Property(
                        name="metadata",
                        data_type=DataType.OBJECT,
                        description="Additional metadata",
                    ),
                    Property(
                        name="indexed_at",
                        data_type=DataType.DATE,
                        description="Indexing timestamp",
                    ),
                ],
            )

            logger.info(f"Created collection {self.COLLECTION_NAME}")

        except Exception as e:
            raise VectorStoreException(
                operation="create_schema",
                message=f"Failed to create schema: {e}",
            )

    def index_entity(self, entity: SupplierEntity) -> str:
        """
        Index a single entity.

        Args:
            entity: Supplier entity to index

        Returns:
            UUID of indexed entity

        Raises:
            VectorStoreException: If indexing fails
        """
        return self.index_entities([entity])[0]

    def index_entities(
        self,
        entities: List[SupplierEntity],
        show_progress: bool = True,
    ) -> List[str]:
        """
        Index multiple entities in batches.

        Args:
            entities: List of supplier entities to index
            show_progress: Show progress logging

        Returns:
            List of UUIDs for indexed entities

        Raises:
            VectorStoreException: If indexing fails
        """
        try:
            collection = self._client.collections.get(self.COLLECTION_NAME)
            uuids = []

            # Generate embeddings for all entities
            search_texts = [e.get_search_text() for e in entities]
            embeddings = self.embedding_pipeline.embed_batch(
                search_texts,
                show_progress=show_progress,
            )

            # Index in batches
            total_batches = (len(entities) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

            with collection.batch.dynamic() as batch:
                for i, (entity, embedding) in enumerate(zip(entities, embeddings)):
                    # Generate deterministic UUID from entity_id
                    uuid = generate_uuid5(entity.entity_id)

                    # Add to batch
                    batch.add_object(
                        properties=entity.to_dict(),
                        vector=embedding.tolist(),
                        uuid=uuid,
                    )
                    uuids.append(str(uuid))

                    # Progress logging
                    if show_progress and (i + 1) % self.BATCH_SIZE == 0:
                        batch_num = (i + 1) // self.BATCH_SIZE
                        logger.info(
                            f"Indexed batch {batch_num}/{total_batches} "
                            f"({i + 1}/{len(entities)} entities)"
                        )

            if show_progress:
                logger.info(f"Successfully indexed {len(entities)} entities")

            return uuids

        except Exception as e:
            raise VectorStoreException(
                operation="index",
                message=f"Failed to index entities: {e}",
                details={"num_entities": len(entities)},
            )

    def search(
        self,
        query_entity: SupplierEntity,
        limit: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[SupplierEntity, float]]:
        """
        Search for similar entities.

        Args:
            query_entity: Entity to search for
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0.0 to 1.0)
            filters: Additional filters (e.g., {'country': 'US'})

        Returns:
            List of (entity, similarity_score) tuples, sorted by similarity

        Raises:
            VectorStoreException: If search fails
        """
        try:
            # Generate embedding for query
            query_text = query_entity.get_search_text()
            query_embedding = self.embedding_pipeline.embed(query_text)

            # Build search query
            collection = self._client.collections.get(self.COLLECTION_NAME)
            search_query = collection.query.near_vector(
                near_vector=query_embedding.tolist(),
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
            )

            # Apply filters if provided
            if filters:
                # Note: Filter syntax depends on Weaviate version
                # This is a simplified example
                pass  # TODO: Implement filtering when needed

            # Execute search
            response = search_query

            # Convert results
            results = []
            for obj in response.objects:
                # Convert distance to similarity (cosine distance -> similarity)
                distance = obj.metadata.distance
                similarity = 1.0 - distance

                # Apply threshold
                if threshold is not None and similarity < threshold:
                    continue

                # Create entity from properties
                entity = SupplierEntity(
                    entity_id=obj.properties["entity_id"],
                    name=obj.properties["name"],
                    normalized_name=obj.properties["normalized_name"],
                    address=obj.properties.get("address"),
                    country=obj.properties.get("country"),
                    tax_id=obj.properties.get("tax_id"),
                    website=obj.properties.get("website"),
                    industry=obj.properties.get("industry"),
                    metadata=obj.properties.get("metadata", {}),
                )

                results.append((entity, similarity))

            return results

        except Exception as e:
            raise VectorStoreException(
                operation="search",
                message=f"Failed to search entities: {e}",
                details={"limit": limit, "threshold": threshold},
            )

    def get_entity(self, entity_id: str) -> Optional[SupplierEntity]:
        """
        Retrieve an entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Supplier entity or None if not found

        Raises:
            VectorStoreException: If retrieval fails
        """
        try:
            collection = self._client.collections.get(self.COLLECTION_NAME)
            uuid = generate_uuid5(entity_id)

            obj = collection.query.fetch_object_by_id(uuid)
            if not obj:
                return None

            return SupplierEntity(
                entity_id=obj.properties["entity_id"],
                name=obj.properties["name"],
                normalized_name=obj.properties["normalized_name"],
                address=obj.properties.get("address"),
                country=obj.properties.get("country"),
                tax_id=obj.properties.get("tax_id"),
                website=obj.properties.get("website"),
                industry=obj.properties.get("industry"),
                metadata=obj.properties.get("metadata", {}),
            )

        except Exception as e:
            raise VectorStoreException(
                operation="get",
                message=f"Failed to retrieve entity: {e}",
                details={"entity_id": entity_id},
            )

    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            True if deleted, False if not found

        Raises:
            VectorStoreException: If deletion fails
        """
        try:
            collection = self._client.collections.get(self.COLLECTION_NAME)
            uuid = generate_uuid5(entity_id)

            collection.data.delete_by_id(uuid)
            logger.info(f"Deleted entity {entity_id}")
            return True

        except Exception as e:
            raise VectorStoreException(
                operation="delete",
                message=f"Failed to delete entity: {e}",
                details={"entity_id": entity_id},
            )

    def count_entities(self) -> int:
        """
        Get total number of indexed entities.

        Returns:
            Total entity count

        Raises:
            VectorStoreException: If count fails
        """
        try:
            collection = self._client.collections.get(self.COLLECTION_NAME)
            response = collection.aggregate.over_all(total_count=True)
            return response.total_count

        except Exception as e:
            raise VectorStoreException(
                operation="count",
                message=f"Failed to count entities: {e}",
            )

    def clear_all(self) -> None:
        """
        Delete all entities (use with caution).

        Raises:
            VectorStoreException: If clearing fails
        """
        try:
            if self._client.collections.exists(self.COLLECTION_NAME):
                self._client.collections.delete(self.COLLECTION_NAME)
                logger.warning(f"Deleted collection {self.COLLECTION_NAME}")
                self._ensure_schema()

        except Exception as e:
            raise VectorStoreException(
                operation="clear",
                message=f"Failed to clear entities: {e}",
            )

    def close(self) -> None:
        """Close connection to Weaviate."""
        if self._client:
            self._client.close()
            logger.info("Closed Weaviate connection")

    def __enter__(self) -> "VectorStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
