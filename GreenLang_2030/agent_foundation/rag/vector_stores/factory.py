"""
Vector Store Factory - Production Implementation

Factory pattern for creating and managing vector store instances with:
- Unified configuration management
- Support for ChromaDB and Pinecone backends
- Health monitoring and failover
- Caching and performance optimization
- Cost tracking and metrics

Author: GreenLang Backend Team
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class VectorStoreType(str, Enum):
    """Supported vector store types."""

    CHROMA = "chroma"
    PINECONE = "pinecone"
    FAISS = "faiss"
    HYBRID = "hybrid"


class VectorStoreConfig(BaseModel):
    """Configuration for vector store creation."""

    store_type: VectorStoreType = Field(
        default=VectorStoreType.CHROMA,
        description="Type of vector store to create"
    )

    # Common configuration
    collection_name: str = Field(
        default="greenlang_rag",
        description="Collection/index name"
    )
    embedding_dimension: int = Field(
        default=384,
        description="Dimension of embeddings (384 for MiniLM, 1536 for OpenAI)"
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric for similarity search"
    )
    batch_size: int = Field(
        default=100,
        description="Batch size for bulk operations"
    )

    # ChromaDB specific
    persist_directory: Optional[str] = Field(
        default=None,
        description="Directory for ChromaDB persistent storage"
    )
    chroma_host: Optional[str] = Field(
        default=None,
        description="ChromaDB server host (for client mode)"
    )
    chroma_port: Optional[int] = Field(
        default=8000,
        description="ChromaDB server port"
    )

    # Pinecone specific
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="Pinecone API key"
    )
    pinecone_environment: str = Field(
        default="us-east-1",
        description="Pinecone environment/region"
    )
    pinecone_cloud: str = Field(
        default="aws",
        description="Cloud provider for Pinecone (aws, gcp, azure)"
    )
    pinecone_namespace: Optional[str] = Field(
        default=None,
        description="Default namespace for multi-tenancy"
    )

    # FAISS specific
    faiss_index_type: str = Field(
        default="IndexFlatL2",
        description="FAISS index type (IndexFlatL2, IndexIVFFlat, etc.)"
    )
    faiss_index_path: Optional[str] = Field(
        default=None,
        description="Path for FAISS index persistence"
    )

    # Performance and monitoring
    enable_metrics: bool = Field(
        default=True,
        description="Enable performance metrics tracking"
    )
    enable_health_checks: bool = Field(
        default=True,
        description="Enable health monitoring"
    )
    health_check_interval_seconds: int = Field(
        default=300,
        description="Health check interval in seconds"
    )

    @validator('embedding_dimension')
    def validate_embedding_dimension(cls, v):
        """Validate embedding dimension is reasonable."""
        if v < 64 or v > 4096:
            raise ValueError("Embedding dimension must be between 64 and 4096")
        return v

    @validator('batch_size')
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if v < 1 or v > 10000:
            raise ValueError("Batch size must be between 1 and 10000")
        return v


class VectorStoreFactory:
    """
    Factory for creating vector store instances.

    Supports multiple backends (ChromaDB, Pinecone) with unified interface.
    Handles configuration, validation, and instance management.

    Features:
    - Unified configuration management
    - Backend validation and error handling
    - Health monitoring integration
    - Performance metrics tracking
    - Failover support (optional)

    Example:
        >>> config = VectorStoreConfig(
        ...     store_type=VectorStoreType.CHROMA,
        ...     persist_directory="./chroma_db"
        ... )
        >>> factory = VectorStoreFactory()
        >>> store = factory.create("chroma", config)
        >>> store.add_documents(documents, embeddings)

    Production Usage:
        >>> # Create Pinecone store for production
        >>> config = VectorStoreConfig(
        ...     store_type=VectorStoreType.PINECONE,
        ...     pinecone_api_key="your-api-key",
        ...     pinecone_environment="us-east-1",
        ...     collection_name="greenlang-prod"
        ... )
        >>> store = factory.create(config.store_type, config)
    """

    # Mapping of store types to their implementations
    _store_classes: Dict[VectorStoreType, Type] = {}

    def __init__(self):
        """Initialize factory with available store implementations."""
        self._register_stores()
        self._instances: Dict[str, Any] = {}
        self.config_cache: Dict[str, VectorStoreConfig] = {}

    def _register_stores(self):
        """Register available vector store implementations."""
        try:
            from .chroma_store import ChromaVectorStore

            self._store_classes[VectorStoreType.CHROMA] = ChromaVectorStore
        except ImportError:
            logger.warning("ChromaVectorStore not available")

        try:
            from .pinecone_store import PineconeVectorStore

            self._store_classes[VectorStoreType.PINECONE] = PineconeVectorStore
        except ImportError:
            logger.warning("PineconeVectorStore not available")

        logger.info(f"Registered {len(self._store_classes)} vector store implementations")

    def create(
        self,
        store_type: Union[str, VectorStoreType],
        config: Optional[VectorStoreConfig] = None
    ) -> Any:
        """
        Create a vector store instance.

        Args:
            store_type: Type of vector store ('chroma', 'pinecone')
            config: Configuration for the store (uses defaults if None)

        Returns:
            Configured vector store instance

        Raises:
            ValueError: If store type is not supported or config is invalid
            ImportError: If required dependencies are not installed

        Example:
            >>> config = VectorStoreConfig(persist_directory="./data")
            >>> store = factory.create(VectorStoreType.CHROMA, config)
        """
        # Normalize store type
        if isinstance(store_type, str):
            try:
                store_type = VectorStoreType(store_type.lower())
            except ValueError:
                raise ValueError(
                    f"Unsupported store type: {store_type}. "
                    f"Supported types: {[t.value for t in VectorStoreType]}"
                )

        # Check if store type is registered
        if store_type not in self._store_classes:
            raise ValueError(
                f"Vector store type '{store_type.value}' is not available. "
                f"Please install the required dependencies."
            )

        # Use default config if not provided
        if config is None:
            config = VectorStoreConfig(store_type=store_type)
        elif config.store_type != store_type:
            config.store_type = store_type

        # Validate config
        self._validate_config(store_type, config)

        # Get store class
        store_class = self._store_classes[store_type]

        # Create instance with appropriate parameters
        try:
            if store_type == VectorStoreType.CHROMA:
                instance = self._create_chroma_store(config)
            elif store_type == VectorStoreType.PINECONE:
                instance = self._create_pinecone_store(config)
            else:
                raise ValueError(f"Unknown store type: {store_type}")

            logger.info(
                f"Created {store_type.value} vector store instance "
                f"(collection: {config.collection_name})"
            )

            return instance

        except Exception as e:
            logger.error(f"Failed to create vector store: {e}", exc_info=True)
            raise

    def _validate_config(
        self, store_type: VectorStoreType, config: VectorStoreConfig
    ) -> None:
        """
        Validate configuration for the selected store type.

        Args:
            store_type: Type of vector store
            config: Configuration to validate

        Raises:
            ValueError: If configuration is invalid for the store type
        """
        if store_type == VectorStoreType.CHROMA:
            # ChromaDB doesn't require API keys
            pass

        elif store_type == VectorStoreType.PINECONE:
            if not config.pinecone_api_key:
                raise ValueError(
                    "pinecone_api_key is required for Pinecone vector store"
                )
            if not config.pinecone_environment:
                raise ValueError(
                    "pinecone_environment is required for Pinecone vector store"
                )

    def _create_chroma_store(self, config: VectorStoreConfig) -> "ChromaVectorStore":
        """Create ChromaDB vector store instance."""
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            distance_metric=config.distance_metric,
            embedding_dimension=config.embedding_dimension,
            batch_size=config.batch_size
        )

    def _create_pinecone_store(self, config: VectorStoreConfig) -> "PineconeVectorStore":
        """Create Pinecone vector store instance."""
        from .pinecone_store import PineconeVectorStore

        return PineconeVectorStore(
            api_key=config.pinecone_api_key,
            environment=config.pinecone_environment,
            index_name=config.collection_name,
            dimension=config.embedding_dimension,
            metric=config.distance_metric,
            namespace=config.pinecone_namespace,
            cloud=config.pinecone_cloud,
            region=config.pinecone_environment,
            batch_size=config.batch_size
        )

    def get_available_stores(self) -> Dict[str, str]:
        """
        Get list of available vector store types.

        Returns:
            Dictionary mapping store type names to descriptions
        """
        descriptions = {
            VectorStoreType.CHROMA: "ChromaDB (development/local)",
            VectorStoreType.PINECONE: "Pinecone (production/cloud)",
            VectorStoreType.FAISS: "FAISS (local, no persistence)",
            VectorStoreType.HYBRID: "Hybrid (vector + keyword search)",
        }

        available = {}
        for store_type, desc in descriptions.items():
            if store_type in self._store_classes:
                available[store_type.value] = f"{desc} [available]"
            else:
                available[store_type.value] = f"{desc} [not available]"

        return available

    def validate_connection(self, store_type: Union[str, VectorStoreType], config: VectorStoreConfig) -> bool:
        """
        Validate connection to vector store backend.

        Args:
            store_type: Type of vector store
            config: Configuration to validate

        Returns:
            True if connection is valid, False otherwise

        Example:
            >>> config = VectorStoreConfig(pinecone_api_key="key")
            >>> is_valid = factory.validate_connection(VectorStoreType.PINECONE, config)
        """
        if isinstance(store_type, str):
            store_type = VectorStoreType(store_type.lower())

        try:
            if store_type == VectorStoreType.CHROMA:
                # Try to create ChromaDB client
                import chromadb

                if config.persist_directory:
                    client = chromadb.PersistentClient(path=config.persist_directory)
                else:
                    client = chromadb.Client()
                return True

            elif store_type == VectorStoreType.PINECONE:
                # Try to connect to Pinecone
                from pinecone import Pinecone

                pc = Pinecone(api_key=config.pinecone_api_key)
                pc.list_indexes()  # Test connection
                return True

            return False

        except Exception as e:
            logger.error(f"Connection validation failed for {store_type}: {e}")
            return False


# Convenience functions
def create_chroma_store(
    collection_name: str = "greenlang_rag",
    persist_directory: Optional[str] = None,
    **kwargs
) -> "ChromaVectorStore":
    """
    Create a ChromaDB vector store instance.

    Args:
        collection_name: Name of the collection
        persist_directory: Directory for persistent storage
        **kwargs: Additional configuration parameters

    Returns:
        ChromaVectorStore instance

    Example:
        >>> store = create_chroma_store(
        ...     collection_name="my_collection",
        ...     persist_directory="./chroma_data"
        ... )
    """
    config = VectorStoreConfig(
        store_type=VectorStoreType.CHROMA,
        collection_name=collection_name,
        persist_directory=persist_directory,
        **kwargs
    )
    factory = VectorStoreFactory()
    return factory.create(VectorStoreType.CHROMA, config)


def create_pinecone_store(
    api_key: str,
    collection_name: str = "greenlang_rag",
    environment: str = "us-east-1",
    namespace: Optional[str] = None,
    **kwargs
) -> "PineconeVectorStore":
    """
    Create a Pinecone vector store instance.

    Args:
        api_key: Pinecone API key
        collection_name: Name of the index
        environment: Pinecone environment/region
        namespace: Default namespace for multi-tenancy
        **kwargs: Additional configuration parameters

    Returns:
        PineconeVectorStore instance

    Example:
        >>> store = create_pinecone_store(
        ...     api_key="your-key",
        ...     collection_name="greenlang-prod",
        ...     environment="us-east-1"
        ... )
    """
    config = VectorStoreConfig(
        store_type=VectorStoreType.PINECONE,
        collection_name=collection_name,
        pinecone_api_key=api_key,
        pinecone_environment=environment,
        pinecone_namespace=namespace,
        **kwargs
    )
    factory = VectorStoreFactory()
    return factory.create(VectorStoreType.PINECONE, config)
