# -*- coding: utf-8 -*-
"""
Embedding provider abstraction for RAG system.

Provides deterministic embedding generation with:
- Provider abstraction (MiniLM, OpenAI, Anthropic)
- CPU-only execution for reproducibility
- Fixed seeds for deterministic mode
- Batch processing for efficiency
- Cost tracking
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Optional
import logging

from greenlang.agents.intelligence.rag.config import RAGConfig, get_config
from greenlang.agents.intelligence.rag.hashing import embedding_hash

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    All embedding providers must implement:
    - name: Model name identifier
    - dim: Embedding dimension
    - embed: Async embedding generation
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get model name."""
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Get embedding dimension."""
        pass

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (numpy arrays)

        Raises:
            ValueError: If texts is empty or contains invalid input
        """
        pass


class MiniLMProvider(EmbeddingProvider):
    """
    MiniLM embedding provider using sentence-transformers.

    Uses sentence-transformers/all-MiniLM-L6-v2 model with:
    - CPU-only execution (for determinism)
    - Fixed random seeds
    - L2 normalization
    - Batch processing

    Determinism guarantees:
    - torch.use_deterministic_algorithms(True)
    - Single-threaded execution
    - Fixed seeds (42 for all RNG)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize MiniLM provider.

        Args:
            model_name: Model name (default: all-MiniLM-L6-v2)
            config: RAG configuration (defaults to global config)
        """
        self.config = config or get_config()
        self._model_name = model_name
        self._dimension = 384  # MiniLM-L6-v2 produces 384-dim embeddings
        self._model = None
        self._total_embeddings = 0

        # Set deterministic mode
        self._setup_determinism()

    def _setup_determinism(self):
        """Configure deterministic execution."""
        if self.config.mode == "replay":
            # Fix all random seeds
            import random
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

            # Force CPU execution
            torch.set_num_threads(1)

            # Enable deterministic algorithms
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as e:
                logger.warning(f"Could not enable deterministic algorithms: {e}")

            # Disable CUDA (force CPU)
            torch.cuda.is_available = lambda: False

            logger.info("MiniLM provider configured for deterministic (replay) mode")

    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading {self._model_name}...")

                # Load model with CPU-only device
                self._model = SentenceTransformer(
                    self._model_name,
                    device="cpu"
                )

                # Verify dimension
                test_embedding = self._model.encode(["test"], convert_to_numpy=True)
                actual_dim = test_embedding.shape[1]

                if actual_dim != self._dimension:
                    logger.warning(
                        f"Expected dimension {self._dimension}, got {actual_dim}"
                    )
                    self._dimension = actual_dim

                logger.info(f"Loaded {self._model_name} (dim={self._dimension})")

            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

    @property
    def name(self) -> str:
        """Get model name."""
        return self._model_name

    @property
    def dim(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            List of L2-normalized embedding vectors

        Raises:
            ValueError: If texts is empty
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")

        # Load model lazily
        self._load_model()

        # Generate embeddings
        logger.debug(f"Embedding {len(texts)} texts...")

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalization
        )

        # Track cost
        self._total_embeddings += len(texts)

        # Convert to list of arrays
        result = [embeddings[i] for i in range(len(texts))]

        # Log embedding hashes in replay mode (for verification)
        if self.config.mode == "replay" and logger.isEnabledFor(logging.DEBUG):
            for i, emb in enumerate(result):
                emb_hash = embedding_hash(emb.tolist())
                logger.debug(f"Embedding {i}: hash={emb_hash[:8]}")

        return result

    def embed_sync(self, texts: List[str]) -> List[np.ndarray]:
        """
        Synchronous embedding generation (no async/await needed).

        Args:
            texts: List of text strings

        Returns:
            List of L2-normalized embedding vectors

        Raises:
            ValueError: If texts is empty
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")

        # Load model lazily
        self._load_model()

        # Generate embeddings synchronously
        logger.debug(f"Embedding {len(texts)} texts...")

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalization
        )

        # Track cost
        self._total_embeddings += len(texts)

        # Convert to list of arrays
        result = [embeddings[i] for i in range(len(texts))]

        # Log embedding hashes in replay mode (for verification)
        if self.config.mode == "replay" and logger.isEnabledFor(logging.DEBUG):
            for i, emb in enumerate(result):
                emb_hash = embedding_hash(emb.tolist())
                logger.debug(f"Embedding {i}: hash={emb_hash[:8]}")

        return result

    def get_stats(self) -> dict:
        """
        Get provider statistics.

        Returns:
            Dictionary with embedding stats
        """
        return {
            "model_name": self._model_name,
            "dimension": self._dimension,
            "total_embeddings": self._total_embeddings,
            "mode": self.config.mode,
        }


class OpenAIProvider(EmbeddingProvider):
    """
    OpenAI embedding provider.

    Uses OpenAI's text-embedding-ada-002 or text-embedding-3-small.

    Note: Not deterministic in replay mode. Use MiniLM for determinism.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            config: RAG configuration
        """
        self.config = config or get_config()
        self._model_name = model_name
        self._api_key = api_key

        # Model dimensions
        self._dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        self._dimension = self._dimensions.get(model_name, 1536)
        self._total_embeddings = 0

        if self.config.mode == "replay":
            logger.warning(
                "OpenAI embeddings are NOT deterministic in replay mode. "
                "Use MiniLM for determinism."
            )

    @property
    def name(self) -> str:
        """Get model name."""
        return self._model_name

    @property
    def dim(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings using OpenAI API.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")

        try:
            import openai

            if self._api_key:
                openai.api_key = self._api_key

            # Call OpenAI API
            response = await openai.Embedding.acreate(
                input=texts,
                model=self._model_name,
            )

            # Extract embeddings
            embeddings = [
                np.array(item["embedding"], dtype=np.float32)
                for item in response["data"]
            ]

            # L2 normalize
            for i, emb in enumerate(embeddings):
                norm = np.linalg.norm(emb)
                if norm > 0:
                    embeddings[i] = emb / norm

            self._total_embeddings += len(texts)

            return embeddings

        except ImportError:
            raise ImportError(
                "openai not installed. Install with: pip install openai"
            )

    def get_stats(self) -> dict:
        """Get provider statistics."""
        return {
            "model_name": self._model_name,
            "dimension": self._dimension,
            "total_embeddings": self._total_embeddings,
            "mode": self.config.mode,
        }


def get_embedding_provider(config: Optional[RAGConfig] = None) -> EmbeddingProvider:
    """
    Get embedding provider based on configuration.

    Args:
        config: RAG configuration (defaults to global config)

    Returns:
        EmbeddingProvider instance

    Raises:
        ValueError: If provider is not supported
    """
    config = config or get_config()

    provider = config.embedding_provider.lower()

    if provider == "minilm":
        return MiniLMProvider(
            model_name=config.embedding_model,
            config=config,
        )
    elif provider == "openai":
        return OpenAIProvider(
            model_name=config.embedding_model,
            config=config,
        )
    elif provider == "anthropic":
        # Anthropic doesn't have official embeddings yet
        raise ValueError(
            "Anthropic embedding provider not yet available. "
            "Use 'minilm' or 'openai'."
        )
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported: minilm, openai"
        )
