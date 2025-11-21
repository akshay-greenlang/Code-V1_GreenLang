# -*- coding: utf-8 -*-
"""
Weaviate client for GreenLang RAG system.

Handles:
- Connection management with retries and health checks
- Schema creation and validation (idempotent)
- Batch operations with dynamic sizing
- Query optimization

CRITICAL: Schema must match CTO spec Section 6 exactly.
"""

import weaviate
from weaviate.util import generate_uuid5
from typing import List, Dict, Optional, Any
import os
import time
import logging

logger = logging.getLogger(__name__)


class WeaviateClient:
    """
    Weaviate client with connection management and schema handling.

    Features:
    - Automatic retry with exponential backoff
    - Health check validation
    - Idempotent schema creation
    - Batch operations with dynamic sizing
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout_config: int = 30000,
        startup_period: int = 30,
    ):
        """
        Initialize Weaviate client with connection management.

        Args:
            endpoint: Weaviate endpoint URL
            api_key: API key for authentication (None for anonymous)
            timeout_config: Request timeout in milliseconds
            startup_period: Time to wait for Weaviate startup (seconds)

        Raises:
            ConnectionError: If Weaviate is not reachable after retries
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_config = timeout_config
        self.startup_period = startup_period

        logger.info(f"Initializing Weaviate client for {endpoint}")

        # Create client with proper configuration
        try:
            if api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=api_key)
                self.client = weaviate.Client(
                    url=endpoint,
                    auth_client_secret=auth_config,
                    timeout_config=(5, timeout_config / 1000),  # (connect, read) in seconds
                )
            else:
                self.client = weaviate.Client(
                    url=endpoint,
                    timeout_config=(5, timeout_config / 1000),
                )
        except Exception as e:
            raise ConnectionError(f"Failed to create Weaviate client: {e}")

        # Wait for Weaviate to be ready with retry logic
        self._wait_for_ready(max_retries=10, initial_delay=1.0)

        logger.info("Weaviate client initialized successfully")

    def _wait_for_ready(
        self,
        max_retries: int = 10,
        initial_delay: float = 1.0,
    ) -> None:
        """
        Wait for Weaviate to be ready with exponential backoff.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries (seconds)

        Raises:
            ConnectionError: If Weaviate is not ready after max_retries
        """
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                if self.health_check():
                    logger.info(f"Weaviate is ready (attempt {attempt + 1})")
                    return
            except Exception as e:
                logger.warning(f"Weaviate not ready (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                logger.info(f"Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
                delay = min(delay * 2, 30.0)  # Exponential backoff, max 30s

        raise ConnectionError(
            f"Weaviate not ready after {max_retries} attempts. "
            f"Please check if Weaviate is running at {self.endpoint}"
        )

    def ensure_schema(self) -> None:
        """
        Create Chunk class schema if not exists (idempotent).

        Schema matches CTO spec (Section 6):
        - chunk_id (string, primary)
        - collection (string, filterable)
        - doc_id, title, publisher, year, version
        - section_path, section_hash
        - page_start, page_end, para_index
        - text (text, searchable)
        - vector (float[], provided externally)

        This method is idempotent - safe to call multiple times.
        """
        class_name = "Chunk"

        # Check if class already exists
        try:
            schema = self.client.schema.get()
            class_names = [c["class"] for c in schema.get("classes", [])]

            if class_name in class_names:
                logger.info(f"Schema class '{class_name}' already exists")
                return
        except Exception as e:
            logger.warning(f"Failed to check existing schema: {e}")

        # Define schema for Chunk class
        chunk_schema = {
            "class": class_name,
            "description": "Document chunk for GreenLang RAG system",
            "vectorizer": "none",  # We supply embeddings externally
            "properties": [
                {
                    "name": "chunk_id",
                    "dataType": ["string"],
                    "description": "Deterministic UUID v5 chunk identifier",
                    "indexInverted": True,
                },
                {
                    "name": "collection",
                    "dataType": ["string"],
                    "description": "Collection name for filtering",
                    "indexInverted": True,
                },
                {
                    "name": "doc_id",
                    "dataType": ["string"],
                    "description": "Document identifier",
                    "indexInverted": True,
                },
                {
                    "name": "title",
                    "dataType": ["string"],
                    "description": "Document title",
                    "indexInverted": True,
                },
                {
                    "name": "publisher",
                    "dataType": ["string"],
                    "description": "Publisher/standards body",
                    "indexInverted": True,
                },
                {
                    "name": "year",
                    "dataType": ["string"],
                    "description": "Publication year",
                    "indexInverted": True,
                },
                {
                    "name": "version",
                    "dataType": ["string"],
                    "description": "Document version",
                    "indexInverted": True,
                },
                {
                    "name": "section_path",
                    "dataType": ["string"],
                    "description": "Hierarchical section path",
                    "indexInverted": True,
                },
                {
                    "name": "section_hash",
                    "dataType": ["string"],
                    "description": "Section hash for verification",
                    "indexInverted": False,
                },
                {
                    "name": "page_start",
                    "dataType": ["int"],
                    "description": "Starting page number",
                    "indexInverted": False,
                },
                {
                    "name": "page_end",
                    "dataType": ["int"],
                    "description": "Ending page number",
                    "indexInverted": False,
                },
                {
                    "name": "para_index",
                    "dataType": ["int"],
                    "description": "Paragraph index within section",
                    "indexInverted": False,
                },
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "Chunk text content",
                    "indexInverted": True,
                },
            ],
        }

        # Create schema
        try:
            self.client.schema.create_class(chunk_schema)
            logger.info(f"Created schema class '{class_name}'")
        except Exception as e:
            # Check if it already exists (race condition)
            try:
                schema = self.client.schema.get()
                class_names = [c["class"] for c in schema.get("classes", [])]
                if class_name in class_names:
                    logger.info(f"Schema class '{class_name}' already exists (created by another process)")
                    return
            except:
                pass

            raise RuntimeError(f"Failed to create schema: {e}")

    def delete_schema(self) -> None:
        """
        Delete Chunk class (for testing/reset).

        WARNING: This will delete all data in the Chunk class.
        """
        class_name = "Chunk"

        try:
            self.client.schema.delete_class(class_name)
            logger.info(f"Deleted schema class '{class_name}'")
        except Exception as e:
            logger.warning(f"Failed to delete schema (may not exist): {e}")

    def health_check(self) -> bool:
        """
        Check if Weaviate is healthy.

        Returns:
            True if Weaviate is ready, False otherwise
        """
        try:
            # Check if client is ready
            return self.client.is_ready()
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Chunk class statistics.

        Returns:
            Dictionary with stats (total_objects, schema_info)
        """
        class_name = "Chunk"

        try:
            # Get object count using aggregation
            result = (
                self.client.query
                .aggregate(class_name)
                .with_meta_count()
                .do()
            )

            total_objects = 0
            if "data" in result:
                agg_data = result["data"].get("Aggregate", {}).get(class_name, [])
                if agg_data and len(agg_data) > 0:
                    total_objects = agg_data[0].get("meta", {}).get("count", 0)

            # Get schema info
            schema = self.client.schema.get(class_name)

            return {
                "class_name": class_name,
                "total_objects": total_objects,
                "schema": schema,
                "endpoint": self.endpoint,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "class_name": class_name,
                "error": str(e),
                "endpoint": self.endpoint,
            }

    def batch_add_objects(
        self,
        objects: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Add objects in batches with error handling.

        Args:
            objects: List of objects to add (each with 'properties' and 'vector')
            batch_size: Batch size (dynamic sizing recommended)

        Returns:
            Dictionary with stats (added, failed, errors)
        """
        class_name = "Chunk"

        total_added = 0
        total_failed = 0
        errors = []

        # Configure batch
        self.client.batch.configure(
            batch_size=batch_size,
            dynamic=True,  # Dynamic batch sizing for performance
            timeout_retries=3,
            callback=None,
        )

        try:
            with self.client.batch as batch:
                for i, obj in enumerate(objects):
                    try:
                        # Extract properties and vector
                        properties = obj.get("properties", {})
                        vector = obj.get("vector")

                        if not vector:
                            raise ValueError(f"Object {i} missing vector")

                        # Generate UUID from chunk_id for determinism
                        chunk_id = properties.get("chunk_id")
                        if chunk_id:
                            uuid = generate_uuid5(chunk_id)
                        else:
                            uuid = None  # Let Weaviate generate

                        # Add to batch
                        batch.add_data_object(
                            data_object=properties,
                            class_name=class_name,
                            vector=vector,
                            uuid=uuid,
                        )

                        total_added += 1
                    except Exception as e:
                        total_failed += 1
                        error_msg = f"Failed to add object {i}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
        except Exception as e:
            error_msg = f"Batch operation failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

        logger.info(f"Batch add complete: {total_added} added, {total_failed} failed")

        return {
            "added": total_added,
            "failed": total_failed,
            "errors": errors,
        }

    def similarity_search(
        self,
        query_vector: List[float],
        k: int = 10,
        collection_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        KNN search in Weaviate with collection filtering.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            collection_filter: Filter by collections (None = all)

        Returns:
            List of results with properties and distance
        """
        class_name = "Chunk"

        try:
            # Build query
            query = (
                self.client.query
                .get(class_name, [
                    "chunk_id",
                    "collection",
                    "doc_id",
                    "title",
                    "publisher",
                    "year",
                    "version",
                    "section_path",
                    "section_hash",
                    "page_start",
                    "page_end",
                    "para_index",
                    "text",
                ])
                .with_near_vector({
                    "vector": query_vector,
                    "certainty": 0.0,  # No certainty threshold
                })
                .with_limit(k)
                .with_additional(["distance", "vector"])
            )

            # Add collection filter if specified
            if collection_filter:
                # Build where filter for collections
                # Format: {"operator": "Or", "operands": [{"path": ["collection"], "operator": "Equal", "valueString": "col1"}, ...]}
                if len(collection_filter) == 1:
                    where_filter = {
                        "path": ["collection"],
                        "operator": "Equal",
                        "valueString": collection_filter[0],
                    }
                else:
                    where_filter = {
                        "operator": "Or",
                        "operands": [
                            {
                                "path": ["collection"],
                                "operator": "Equal",
                                "valueString": coll,
                            }
                            for coll in collection_filter
                        ],
                    }

                query = query.with_where(where_filter)

            # Execute query
            result = query.do()

            # Parse results
            objects = result.get("data", {}).get("Get", {}).get(class_name, [])

            return objects
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RuntimeError(f"Weaviate search failed: {e}")
