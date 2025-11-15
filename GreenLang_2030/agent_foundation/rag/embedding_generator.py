"""
Embedding Generator for GreenLang RAG System

Multi-model embedding generation with support for:
- Sentence Transformers (all-mpnet-base-v2, all-MiniLM-L6-v2, etc.)
- OpenAI Embeddings (text-embedding-ada-002, text-embedding-3-small)
- Cohere Embeddings (embed-english-v3.0, embed-multilingual-v3.0)
- Custom embeddings via transformers

Includes caching for 66% cost reduction as per GreenLang requirements.
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Supported embedding models"""
    # Sentence Transformers
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    MPNET = "sentence-transformers/all-mpnet-base-v2"
    E5_SMALL = "intfloat/e5-small-v2"
    E5_BASE = "intfloat/e5-base-v2"
    E5_LARGE = "intfloat/e5-large-v2"
    BGE_SMALL = "BAAI/bge-small-en-v1.5"
    BGE_BASE = "BAAI/bge-base-en-v1.5"
    BGE_LARGE = "BAAI/bge-large-en-v1.5"

    # OpenAI
    ADA_002 = "text-embedding-ada-002"
    EMBEDDING_3_SMALL = "text-embedding-3-small"
    EMBEDDING_3_LARGE = "text-embedding-3-large"

    # Cohere
    COHERE_ENGLISH = "embed-english-v3.0"
    COHERE_MULTILINGUAL = "embed-multilingual-v3.0"
    COHERE_LIGHT = "embed-english-light-v3.0"

    # Custom
    CUSTOM = "custom"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model: EmbeddingModel
    dimension: int
    batch_size: int = 32
    normalize: bool = True
    device: str = "cpu"  # cpu, cuda, mps
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30


class EmbeddingCache:
    """
    Cache for embeddings to reduce costs by 66%
    Implements LRU cache with TTL
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size: int = 10000,
        ttl_hours: int = 24
    ):
        self.cache_dir = cache_dir or Path("./cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)

        # In-memory cache
        self.memory_cache: Dict[str, Tuple[np.ndarray, datetime]] = {}

        # Load persistent cache index
        self.index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_index()

    def _load_index(self) -> Dict[str, Dict]:
        """Load cache index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_index(self):
        """Save cache index to disk"""
        with open(self.index_file, 'w') as f:
            json.dump(self.cache_index, f)

    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model"""
        key_string = f"{model}:{text}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache

        Args:
            text: Text that was embedded
            model: Model used for embedding

        Returns:
            Cached embedding or None if not found/expired
        """
        cache_key = self._get_cache_key(text, model)

        # Check memory cache first
        if cache_key in self.memory_cache:
            embedding, timestamp = self.memory_cache[cache_key]
            if datetime.now() - timestamp < self.ttl:
                logger.debug(f"Cache hit (memory): {cache_key[:8]}...")
                return embedding
            else:
                del self.memory_cache[cache_key]

        # Check disk cache
        if cache_key in self.cache_index:
            cache_info = self.cache_index[cache_key]
            timestamp = datetime.fromisoformat(cache_info['timestamp'])

            if datetime.now() - timestamp < self.ttl:
                cache_file = self.cache_dir / f"{cache_key}.npy"
                if cache_file.exists():
                    try:
                        embedding = np.load(cache_file)
                        # Add to memory cache
                        self.memory_cache[cache_key] = (embedding, timestamp)
                        logger.debug(f"Cache hit (disk): {cache_key[:8]}...")
                        return embedding
                    except:
                        # Corrupted cache file
                        del self.cache_index[cache_key]
                        cache_file.unlink(missing_ok=True)

        return None

    def set(self, text: str, model: str, embedding: np.ndarray):
        """
        Store embedding in cache

        Args:
            text: Text that was embedded
            model: Model used for embedding
            embedding: The embedding vector
        """
        cache_key = self._get_cache_key(text, model)
        timestamp = datetime.now()

        # Add to memory cache
        self.memory_cache[cache_key] = (embedding, timestamp)

        # Enforce memory cache size limit
        if len(self.memory_cache) > self.max_size:
            # Remove oldest entries
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1][1]
            )
            for key, _ in sorted_items[:len(self.memory_cache) - self.max_size]:
                del self.memory_cache[key]

        # Save to disk
        cache_file = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_file, embedding)

        # Update index
        self.cache_index[cache_key] = {
            'timestamp': timestamp.isoformat(),
            'model': model,
            'dimension': embedding.shape[0],
            'text_preview': text[:100]
        }
        self._save_index()

        logger.debug(f"Cache set: {cache_key[:8]}...")

    def clear_expired(self):
        """Remove expired entries from cache"""
        now = datetime.now()
        expired_keys = []

        for key, info in self.cache_index.items():
            timestamp = datetime.fromisoformat(info['timestamp'])
            if now - timestamp >= self.ttl:
                expired_keys.append(key)

        for key in expired_keys:
            # Remove from index
            del self.cache_index[key]
            # Remove file
            cache_file = self.cache_dir / f"{key}.npy"
            cache_file.unlink(missing_ok=True)
            # Remove from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]

        if expired_keys:
            self._save_index()
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_size': len(self.cache_index),
            'cache_directory': str(self.cache_dir),
            'max_size': self.max_size,
            'ttl_hours': self.ttl.total_seconds() / 3600
        }


class EmbeddingGenerator:
    """
    Generate embeddings for documents using various models
    Supports batch processing and caching for efficiency
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding generator

        Args:
            config: Embedding configuration
        """
        self.config = config
        self.model = None
        self.cache = None

        # Initialize cache if enabled
        if config.cache_enabled:
            self.cache = EmbeddingCache(ttl_hours=config.cache_ttl_hours)

        # Load model
        self._load_model()

    def _load_model(self):
        """Load embedding model based on configuration"""
        model_name = self.config.model.value

        if "sentence-transformers" in model_name or model_name.startswith("BAAI/") or model_name.startswith("intfloat/"):
            self._load_sentence_transformer()
        elif model_name.startswith("text-embedding"):
            self._load_openai()
        elif model_name.startswith("embed-"):
            self._load_cohere()
        elif self.config.model == EmbeddingModel.CUSTOM:
            self._load_custom()
        else:
            raise ValueError(f"Unknown embedding model: {model_name}")

    def _load_sentence_transformer(self):
        """Load Sentence Transformer model"""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.config.model.value,
                device=self.config.device
            )

            # Update dimension from model
            self.config.dimension = self.model.get_sentence_embedding_dimension()

            logger.info(f"Loaded Sentence Transformer: {self.config.model.value}")
            logger.info(f"Embedding dimension: {self.config.dimension}")

        except ImportError:
            raise ImportError("sentence-transformers not installed: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Error loading Sentence Transformer: {e}")
            raise

    def _load_openai(self):
        """Load OpenAI embedding configuration"""
        if not self.config.api_key:
            raise ValueError("OpenAI API key required for OpenAI embeddings")

        # Set dimensions based on model
        if self.config.model == EmbeddingModel.ADA_002:
            self.config.dimension = 1536
        elif self.config.model == EmbeddingModel.EMBEDDING_3_SMALL:
            self.config.dimension = 1536
        elif self.config.model == EmbeddingModel.EMBEDDING_3_LARGE:
            self.config.dimension = 3072

        logger.info(f"Configured OpenAI embeddings: {self.config.model.value}")

    def _load_cohere(self):
        """Load Cohere embedding configuration"""
        if not self.config.api_key:
            raise ValueError("Cohere API key required for Cohere embeddings")

        # Set dimensions based on model
        if self.config.model == EmbeddingModel.COHERE_ENGLISH:
            self.config.dimension = 1024
        elif self.config.model == EmbeddingModel.COHERE_MULTILINGUAL:
            self.config.dimension = 1024
        elif self.config.model == EmbeddingModel.COHERE_LIGHT:
            self.config.dimension = 384

        logger.info(f"Configured Cohere embeddings: {self.config.model.value}")

    def _load_custom(self):
        """Load custom embedding model"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            model_name = self.config.api_base or "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

            if self.config.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            elif self.config.device == "mps" and torch.backends.mps.is_available():
                self.model = self.model.to("mps")

            self.model.eval()

            # Get dimension from model
            with torch.no_grad():
                dummy_input = self.tokenizer("test", return_tensors="pt")
                if self.config.device in ["cuda", "mps"]:
                    dummy_input = {k: v.to(self.config.device) for k, v in dummy_input.items()}
                output = self.model(**dummy_input)
                self.config.dimension = output.last_hidden_state.shape[-1]

            logger.info(f"Loaded custom model: {model_name}")
            logger.info(f"Embedding dimension: {self.config.dimension}")

        except ImportError:
            raise ImportError("transformers not installed: pip install transformers torch")
        except Exception as e:
            logger.error(f"Error loading custom model: {e}")
            raise

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed
            show_progress: Show progress bar

        Returns:
            Array of embeddings (num_texts, dimension)
        """
        if not texts:
            return np.array([])

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache if enabled
        if self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.config.model.value)
                if cached is not None:
                    embeddings.append(cached)
                else:
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            embeddings = [None] * len(texts)

        # Generate embeddings for uncached texts
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} texts (cached: {len(texts) - len(uncached_texts)})")

            if self.model and hasattr(self.model, 'encode'):
                # Sentence Transformer
                new_embeddings = self._embed_sentence_transformer(uncached_texts, show_progress)
            elif self.config.model.value.startswith("text-embedding"):
                # OpenAI
                new_embeddings = self._embed_openai(uncached_texts)
            elif self.config.model.value.startswith("embed-"):
                # Cohere
                new_embeddings = self._embed_cohere(uncached_texts)
            else:
                # Custom
                new_embeddings = self._embed_custom(uncached_texts)

            # Add to results and cache
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                if self.cache:
                    self.cache.set(texts[idx], self.config.model.value, embedding)

        return np.array(embeddings)

    def _embed_sentence_transformer(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[np.ndarray]:
        """Embed using Sentence Transformer"""
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True
        )
        return list(embeddings)

    def _embed_openai(self, texts: List[str]) -> List[np.ndarray]:
        """Embed using OpenAI API"""
        try:
            import openai
            from tenacity import retry, stop_after_attempt, wait_exponential

            openai.api_key = self.config.api_key
            if self.config.api_base:
                openai.api_base = self.config.api_base

            @retry(
                stop=stop_after_attempt(self.config.max_retries),
                wait=wait_exponential(multiplier=1, min=4, max=10)
            )
            def get_embeddings(batch):
                response = openai.Embedding.create(
                    model=self.config.model.value,
                    input=batch
                )
                return [np.array(data['embedding']) for data in response['data']]

            embeddings = []
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                batch_embeddings = get_embeddings(batch)
                embeddings.extend(batch_embeddings)

            if self.config.normalize:
                embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]

            return embeddings

        except ImportError:
            raise ImportError("openai not installed: pip install openai tenacity")
        except Exception as e:
            logger.error(f"Error with OpenAI embeddings: {e}")
            raise

    def _embed_cohere(self, texts: List[str]) -> List[np.ndarray]:
        """Embed using Cohere API"""
        try:
            import cohere
            from tenacity import retry, stop_after_attempt, wait_exponential

            co = cohere.Client(self.config.api_key)

            @retry(
                stop=stop_after_attempt(self.config.max_retries),
                wait=wait_exponential(multiplier=1, min=4, max=10)
            )
            def get_embeddings(batch):
                response = co.embed(
                    texts=batch,
                    model=self.config.model.value,
                    input_type="search_document"
                )
                return [np.array(emb) for emb in response.embeddings]

            embeddings = []
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                batch_embeddings = get_embeddings(batch)
                embeddings.extend(batch_embeddings)

            if self.config.normalize:
                embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]

            return embeddings

        except ImportError:
            raise ImportError("cohere not installed: pip install cohere tenacity")
        except Exception as e:
            logger.error(f"Error with Cohere embeddings: {e}")
            raise

    def _embed_custom(self, texts: List[str]) -> List[np.ndarray]:
        """Embed using custom transformer model"""
        import torch

        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                )

                if self.config.device in ["cuda", "mps"]:
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

                # Get embeddings
                outputs = self.model(**inputs)

                # Pool embeddings (mean pooling)
                attention_mask = inputs['attention_mask']
                embeddings_batch = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings_batch.size()).float()
                sum_embeddings = torch.sum(embeddings_batch * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask

                # Convert to numpy
                batch_embeddings = mean_embeddings.cpu().numpy()

                if self.config.normalize:
                    batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)

                embeddings.extend(list(batch_embeddings))

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        # For some models, queries need special formatting
        if self.config.model.value.startswith("intfloat/e5"):
            query = f"query: {query}"
        elif self.config.model.value.startswith("BAAI/bge"):
            query = f"Represent this sentence for searching relevant passages: {query}"

        embeddings = self.embed_texts([query])
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.config.dimension

    def clear_cache(self):
        """Clear embedding cache"""
        if self.cache:
            self.cache.clear_expired()

    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics"""
        stats = {
            'model': self.config.model.value,
            'dimension': self.config.dimension,
            'batch_size': self.config.batch_size,
            'normalize': self.config.normalize,
            'device': self.config.device
        }

        if self.cache:
            stats['cache'] = self.cache.get_stats()

        return stats


class MultiModelEmbedding:
    """
    Generate embeddings using multiple models and combine them
    Useful for improving retrieval quality
    """

    def __init__(self, configs: List[EmbeddingConfig], weights: Optional[List[float]] = None):
        """
        Initialize multi-model embedding generator

        Args:
            configs: List of embedding configurations
            weights: Weights for combining embeddings (normalized internally)
        """
        self.generators = [EmbeddingGenerator(config) for config in configs]

        if weights:
            if len(weights) != len(configs):
                raise ValueError("Number of weights must match number of configs")
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            # Equal weights
            self.weights = [1.0 / len(configs)] * len(configs)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate combined embeddings for texts

        Args:
            texts: List of texts to embed

        Returns:
            Combined embeddings
        """
        all_embeddings = []

        for generator, weight in zip(self.generators, self.weights):
            embeddings = generator.embed_texts(texts)
            # Normalize to same scale
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Apply weight
            embeddings = embeddings * weight
            all_embeddings.append(embeddings)

        # Concatenate embeddings
        combined = np.concatenate(all_embeddings, axis=1)

        # Final normalization
        combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)

        return combined

    def embed_query(self, query: str) -> np.ndarray:
        """Generate combined embedding for query"""
        return self.embed_texts([query])[0]

    def get_dimension(self) -> int:
        """Get total dimension of combined embeddings"""
        return sum(gen.get_dimension() for gen in self.generators)