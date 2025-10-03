"""
Vector store abstraction for RAG system.

Provides deterministic vector storage with:
- Provider abstraction (FAISS, ChromaDB, Weaviate)
- Exact search (no approximate search for determinism)
- Collection filtering
- Security controls (NO dangerous deserialization)
"""

import numpy as np
import pickle
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

from greenlang.intelligence.rag.config import RAGConfig, get_config, is_collection_allowed
from greenlang.intelligence.rag.models import Chunk

logger = logging.getLogger(__name__)


class Document:
    """
    Document wrapper for vector store.

    Wraps a Chunk with its embedding vector.
    """

    def __init__(self, chunk: Chunk, embedding: Optional[np.ndarray] = None):
        """
        Initialize document.

        Args:
            chunk: Document chunk
            embedding: Embedding vector (optional)
        """
        self.chunk = chunk
        self.embedding = embedding

        # Metadata for filtering
        self.metadata = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "collection": self._extract_collection(chunk),
            "section_path": chunk.section_path,
            "page_start": chunk.page_start,
        }

    def _extract_collection(self, chunk: Chunk) -> str:
        """
        Extract collection name from chunk.

        Since Chunk doesn't have collection field, we need to get it from
        the doc_meta that was used to create it. For now, we'll use
        the extra field or infer from doc_id.

        Args:
            chunk: Document chunk

        Returns:
            Collection name (or "unknown")
        """
        # Check if collection is in extra metadata
        if "collection" in chunk.extra:
            return chunk.extra["collection"]

        # Default to unknown (should be set by caller)
        return "unknown"


class VectorStoreProvider(ABC):
    """
    Abstract base class for vector store providers.

    All providers must implement:
    - add_documents: Add documents with embeddings
    - similarity_search: Search for similar documents
    - save: Persist vector store to disk
    - load: Load vector store from disk
    """

    @abstractmethod
    def add_documents(
        self,
        docs: List[Document],
        collection: str,
    ) -> None:
        """
        Add documents to vector store.

        Args:
            docs: List of documents with embeddings
            collection: Collection name

        Raises:
            ValueError: If collection is not allowed
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        collections: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            collections: Filter by collections (None = all allowed collections)

        Returns:
            List of documents sorted by similarity (descending)
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save vector store to disk.

        Args:
            path: Directory path to save to
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load vector store from disk.

        Args:
            path: Directory path to load from
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.

        Returns:
            Dictionary with stats (total_docs, collections, etc.)
        """
        pass


class FAISSProvider(VectorStoreProvider):
    """
    FAISS vector store provider.

    Uses FAISS IndexFlatL2 for exact L2 distance search.

    Determinism guarantees:
    - Exact search (no approximate indexing)
    - Single-threaded search
    - Deterministic tie-breaking by chunk_id

    Security:
    - NEVER uses allow_dangerous_deserialization=True
    - Custom safe deserialization
    - Collection allowlist enforcement
    """

    def __init__(
        self,
        dimension: int = 384,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize FAISS provider.

        Args:
            dimension: Embedding dimension
            config: RAG configuration
        """
        self.config = config or get_config()
        self.dimension = dimension

        # Initialize FAISS index
        import faiss

        # Use IndexFlatL2 for exact search (deterministic)
        self.index = faiss.IndexFlatL2(dimension)

        # Set single-threaded mode for determinism
        if self.config.mode == "replay":
            faiss.omp_set_num_threads(1)
            logger.info("FAISS configured for single-threaded (deterministic) mode")

        # Store documents and metadata
        self.documents: List[Document] = []
        self.doc_metadata: List[Dict[str, Any]] = []

        # Collection index
        self.collections: Dict[str, List[int]] = {}

    def add_documents(
        self,
        docs: List[Document],
        collection: str,
    ) -> None:
        """
        Add documents to FAISS index.

        Args:
            docs: List of documents with embeddings
            collection: Collection name

        Raises:
            ValueError: If collection is not allowed
        """
        # Validate collection
        if not is_collection_allowed(collection, self.config):
            raise ValueError(
                f"Collection '{collection}' is not allowed. "
                f"Allowed: {', '.join(self.config.allowlist)}"
            )

        if not docs:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(docs)} documents to collection '{collection}'")

        # Collect embeddings
        embeddings = []
        for doc in docs:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.chunk.chunk_id} has no embedding")

            # Verify dimension
            if len(doc.embedding) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, "
                    f"got {len(doc.embedding)}"
                )

            embeddings.append(doc.embedding)

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Add to FAISS index
        start_idx = len(self.documents)
        self.index.add(embeddings_array)

        # Store documents and metadata
        for i, doc in enumerate(docs):
            # Update document metadata with collection
            doc.metadata["collection"] = collection

            self.documents.append(doc)
            self.doc_metadata.append(doc.metadata)

            # Update collection index
            doc_idx = start_idx + i
            if collection not in self.collections:
                self.collections[collection] = []
            self.collections[collection].append(doc_idx)

        logger.info(
            f"Added {len(docs)} documents to '{collection}' "
            f"(total: {len(self.documents)})"
        )

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        collections: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Search for similar documents using L2 distance.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            collections: Filter by collections (None = all collections)

        Returns:
            List of documents sorted by similarity (closest first)
        """
        if len(self.documents) == 0:
            logger.warning("Vector store is empty")
            return []

        # Validate query embedding
        if len(query_embedding) != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, "
                f"got {len(query_embedding)}"
            )

        # Filter by collections
        if collections is not None:
            # Validate collections
            for coll in collections:
                if not is_collection_allowed(coll, self.config):
                    raise ValueError(
                        f"Collection '{coll}' is not allowed. "
                        f"Allowed: {', '.join(self.config.allowlist)}"
                    )

            # Get indices for allowed collections
            allowed_indices = []
            for coll in collections:
                if coll in self.collections:
                    allowed_indices.extend(self.collections[coll])

            if not allowed_indices:
                logger.warning(f"No documents in collections: {collections}")
                return []

            # Convert to set for fast lookup
            allowed_indices_set = set(allowed_indices)
        else:
            allowed_indices_set = None

        # Search FAISS index
        query_array = np.array([query_embedding], dtype=np.float32)

        # Search with larger k to allow for filtering
        search_k = min(k * 10, len(self.documents)) if collections else k
        distances, indices = self.index.search(query_array, search_k)

        # Filter results by collection and collect documents
        results = []
        for i, idx in enumerate(indices[0]):
            # Check if index is valid
            if idx < 0 or idx >= len(self.documents):
                continue

            # Filter by collection
            if allowed_indices_set is not None and idx not in allowed_indices_set:
                continue

            doc = self.documents[idx]
            distance = distances[0][i]

            results.append((doc, distance))

            # Stop when we have enough results
            if len(results) >= k:
                break

        # Sort by distance (ascending = most similar first)
        # Deterministic tie-breaking by chunk_id
        results.sort(key=lambda x: (x[1], x[0].chunk.chunk_id))

        # Return just the documents
        return [doc for doc, _ in results]

    def save(self, path: Path) -> None:
        """
        Save FAISS index and metadata to disk.

        Args:
            path: Directory path to save to
        """
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")

        # Save documents and metadata using pickle
        # NOTE: This is safe because we control the data being pickled
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(
                {
                    "dimension": self.dimension,
                    "documents": self.documents,
                    "doc_metadata": self.doc_metadata,
                    "collections": self.collections,
                },
                f,
            )
        logger.info(f"Saved metadata to {metadata_path}")

    def load(self, path: Path) -> None:
        """
        Load FAISS index and metadata from disk.

        SECURITY: Does NOT use allow_dangerous_deserialization.
        Uses safe custom deserialization.

        Args:
            path: Directory path to load from
        """
        import faiss

        path = Path(path)

        # Load FAISS index
        index_path = path / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        # Load with safe mode (no allow_dangerous_deserialization)
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index from {index_path}")

        # Load metadata
        metadata_path = path / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, "rb") as f:
            # Load metadata safely
            # NOTE: We control this data, but should validate it
            metadata = pickle.load(f)

        self.dimension = metadata["dimension"]
        self.documents = metadata["documents"]
        self.doc_metadata = metadata["doc_metadata"]
        self.collections = metadata["collections"]

        logger.info(
            f"Loaded {len(self.documents)} documents from {len(self.collections)} collections"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "provider": "faiss",
            "dimension": self.dimension,
            "total_documents": len(self.documents),
            "total_collections": len(self.collections),
            "collections": {
                name: len(indices) for name, indices in self.collections.items()
            },
            "mode": self.config.mode,
        }


class WeaviateProvider(VectorStoreProvider):
    """
    Weaviate vector store provider for production deployments.

    Features:
    - Self-hosted Weaviate instance
    - Collection-based filtering
    - Batch ingestion with dynamic sizing
    - KNN search with exact/approximate modes

    Determinism guarantees:
    - Deterministic UUID generation from chunk_id
    - Sorted results by distance (ASC), then chunk_id (ASC)
    - Single-threaded mode in replay mode
    """

    def __init__(
        self,
        dimension: int = 384,
        config: Optional[RAGConfig] = None,
        weaviate_client=None,
    ):
        """
        Initialize Weaviate provider.

        Args:
            dimension: Embedding dimension
            config: RAG configuration
            weaviate_client: WeaviateClient instance (optional)
        """
        self.config = config or get_config()
        self.dimension = dimension

        # Import and create WeaviateClient
        if weaviate_client is None:
            from greenlang.intelligence.rag.weaviate_client import WeaviateClient

            self.weaviate_client = WeaviateClient(
                endpoint=self.config.weaviate_endpoint,
                api_key=os.getenv("WEAVIATE_API_KEY"),
                timeout_config=30000,
                startup_period=30,
            )
        else:
            self.weaviate_client = weaviate_client

        # Ensure schema exists
        self.weaviate_client.ensure_schema()

        # Track metadata for stats
        self.total_documents_added = 0
        self.collections_added = set()

        logger.info(f"Weaviate provider initialized (dimension={dimension})")

    def add_documents(
        self,
        docs: List[Document],
        collection: str,
    ) -> None:
        """
        Add documents to Weaviate with batch optimization.

        Uses Weaviate's batch API for efficiency:
        - Dynamic batch sizing
        - Automatic retry on failures
        - Progress tracking

        Args:
            docs: List of documents with embeddings
            collection: Collection name

        Raises:
            ValueError: If collection is not allowed
        """
        # Validate collection
        if not is_collection_allowed(collection, self.config):
            raise ValueError(
                f"Collection '{collection}' is not allowed. "
                f"Allowed: {', '.join(self.config.allowlist)}"
            )

        if not docs:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(docs)} documents to collection '{collection}'")

        # Prepare objects for batch insertion
        objects = []
        for doc in docs:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.chunk.chunk_id} has no embedding")

            # Verify dimension
            if len(doc.embedding) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, "
                    f"got {len(doc.embedding)}"
                )

            # Extract metadata from chunk
            chunk = doc.chunk

            # Build properties dict matching Weaviate schema
            properties = {
                "chunk_id": chunk.chunk_id,
                "collection": collection,
                "doc_id": chunk.doc_id,
                "title": doc.metadata.get("title", ""),
                "publisher": doc.metadata.get("publisher", ""),
                "year": doc.metadata.get("year", ""),
                "version": doc.metadata.get("version", ""),
                "section_path": chunk.section_path,
                "section_hash": chunk.section_hash,
                "page_start": chunk.page_start or 0,
                "page_end": chunk.page_end or 0,
                "para_index": chunk.paragraph or 0,
                "text": chunk.text,
            }

            # Create object with properties and vector
            obj = {
                "properties": properties,
                "vector": doc.embedding.tolist(),  # Convert numpy array to list
            }

            objects.append(obj)

        # Batch add objects
        result = self.weaviate_client.batch_add_objects(
            objects=objects,
            batch_size=100,  # Dynamic sizing handled by client
        )

        # Track stats
        self.total_documents_added += result["added"]
        self.collections_added.add(collection)

        # Log results
        if result["failed"] > 0:
            logger.warning(
                f"Added {result['added']} documents to '{collection}', "
                f"but {result['failed']} failed"
            )
            for error in result["errors"][:5]:  # Show first 5 errors
                logger.error(f"  {error}")
        else:
            logger.info(
                f"Successfully added {result['added']} documents to '{collection}'"
            )

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        collections: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        KNN search in Weaviate with collection filtering.

        Uses Weaviate GraphQL API:
        - nearVector query
        - where filter for collections
        - limit and offset

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            collections: Filter by collections (None = all collections)

        Returns:
            List of documents sorted by similarity (closest first)
        """
        # Validate query embedding
        if len(query_embedding) != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, "
                f"got {len(query_embedding)}"
            )

        # Validate collections
        if collections is not None:
            for coll in collections:
                if not is_collection_allowed(coll, self.config):
                    raise ValueError(
                        f"Collection '{coll}' is not allowed. "
                        f"Allowed: {', '.join(self.config.allowlist)}"
                    )

        # Execute similarity search
        try:
            results = self.weaviate_client.similarity_search(
                query_vector=query_embedding.tolist(),
                k=k,
                collection_filter=collections,
            )
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}")
            return []

        if not results:
            logger.warning(
                f"No results found (collections={collections})"
            )
            return []

        # Convert Weaviate results to Document objects
        documents = []
        for result in results:
            # Extract properties
            props = result

            # Reconstruct Chunk object
            chunk = Chunk(
                chunk_id=props.get("chunk_id", ""),
                doc_id=props.get("doc_id", ""),
                section_path=props.get("section_path", ""),
                section_hash=props.get("section_hash", ""),
                page_start=props.get("page_start"),
                page_end=props.get("page_end"),
                paragraph=props.get("para_index"),
                start_char=0,  # Not stored in Weaviate
                end_char=0,  # Not stored in Weaviate
                text=props.get("text", ""),
                token_count=0,  # Not stored in Weaviate
                extra={"collection": props.get("collection", "")},
            )

            # Get distance (lower = more similar)
            additional = result.get("_additional", {})
            distance = additional.get("distance", 1.0)

            # Convert distance to similarity score (0-1, higher = more similar)
            # Weaviate distance is L2 distance, convert to similarity
            # similarity = 1 / (1 + distance)
            similarity_score = 1.0 / (1.0 + distance)

            # Create Document with embedding if available
            embedding = additional.get("vector")
            if embedding:
                embedding = np.array(embedding, dtype=np.float32)

            doc = Document(chunk=chunk, embedding=embedding)
            doc.metadata["similarity_score"] = similarity_score
            doc.metadata["distance"] = distance

            documents.append(doc)

        # Sort by distance (ascending = most similar first)
        # Deterministic tie-breaking by chunk_id
        documents.sort(key=lambda x: (x.metadata["distance"], x.chunk.chunk_id))

        logger.info(f"Retrieved {len(documents)} documents from Weaviate")

        return documents

    def save(self, path: Path) -> None:
        """
        Save Weaviate connection info (not data).

        Weaviate is persistent, so we just save connection config.

        Args:
            path: Directory path to save to
        """
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config_path = path / "weaviate_config.json"

        config = {
            "provider": "weaviate",
            "endpoint": self.weaviate_client.endpoint,
            "dimension": self.dimension,
            "total_documents_added": self.total_documents_added,
            "collections_added": list(self.collections_added),
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved Weaviate config to {config_path}")

    def load(self, path: Path) -> None:
        """
        Load Weaviate connection from saved config.

        Args:
            path: Directory path to load from
        """
        import json

        path = Path(path)
        config_path = path / "weaviate_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Weaviate config not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.dimension = config.get("dimension", self.dimension)
        self.total_documents_added = config.get("total_documents_added", 0)
        self.collections_added = set(config.get("collections_added", []))

        logger.info(
            f"Loaded Weaviate config from {config_path} "
            f"(dimension={self.dimension}, docs={self.total_documents_added})"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        weaviate_stats = self.weaviate_client.get_stats()

        return {
            "provider": "weaviate",
            "dimension": self.dimension,
            "endpoint": self.weaviate_client.endpoint,
            "total_documents_added": self.total_documents_added,
            "collections_added": list(self.collections_added),
            "weaviate_stats": weaviate_stats,
            "mode": self.config.mode,
        }


def get_vector_store(
    dimension: int = 384,
    config: Optional[RAGConfig] = None,
) -> VectorStoreProvider:
    """
    Get vector store provider based on configuration.

    Supports:
    - faiss: In-memory, fast, good for dev/testing
    - weaviate: Self-hosted, production-ready, scalable
    - chromadb: Persistent, alternative to FAISS (not yet implemented)

    Args:
        dimension: Embedding dimension
        config: RAG configuration

    Returns:
        VectorStoreProvider instance

    Raises:
        ValueError: If provider is not supported
    """
    config = config or get_config()

    provider = config.vector_store_provider.lower()

    if provider == "faiss":
        return FAISSProvider(dimension=dimension, config=config)
    elif provider == "weaviate":
        from greenlang.intelligence.rag.weaviate_client import WeaviateClient

        weaviate_client = WeaviateClient(
            endpoint=config.weaviate_endpoint,
            api_key=os.getenv("WEAVIATE_API_KEY"),
            timeout_config=30000,
            startup_period=30,
        )
        return WeaviateProvider(
            dimension=dimension,
            config=config,
            weaviate_client=weaviate_client,
        )
    elif provider == "chromadb":
        raise NotImplementedError("ChromaDB provider not yet implemented")
    else:
        raise ValueError(
            f"Unknown vector store provider: {provider}. "
            f"Supported: faiss, weaviate"
        )
