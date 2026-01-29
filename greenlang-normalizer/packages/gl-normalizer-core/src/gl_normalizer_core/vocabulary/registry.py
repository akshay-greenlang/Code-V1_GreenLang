"""
Vocabulary Registry singleton for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides a centralized registry for managing vocabulary loaders
and providing lazy access to vocabularies. The registry follows the singleton
pattern to ensure consistent vocabulary access across the application.

Key Design Principles:
    - Singleton pattern for global access
    - Lazy loading of vocabularies
    - Built-in support for core vocabularies (fuels, materials, processes, units)
    - Thread-safe operations

Example:
    >>> from gl_normalizer_core.vocabulary.registry import VocabularyRegistry
    >>> registry = VocabularyRegistry.get_instance()
    >>> registry.register_vocabulary("custom", yaml_loader)
    >>> vocab = registry.get_vocabulary("fuels")
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import hashlib
import json
import logging
import os
import threading

from gl_normalizer_core.errors.codes import GLNORMErrorCode
from gl_normalizer_core.errors import VocabularyError
from gl_normalizer_core.vocabulary.models import (
    Vocabulary,
    Entity,
    EntityType,
)
from gl_normalizer_core.vocabulary.loader import (
    VocabularyLoader,
    YAMLVocabularyLoader,
    CachedLoader,
    LoaderConfig,
)

logger = logging.getLogger(__name__)


# Built-in vocabulary IDs
BUILTIN_VOCABULARIES: Set[str] = frozenset({
    "fuels",
    "materials",
    "processes",
    "units",
})


class VocabularyRegistry:
    """
    Singleton registry for vocabulary management.

    Provides centralized access to vocabularies with lazy loading,
    caching, and support for multiple loader backends.

    Attributes:
        _loaders: Dictionary of vocabulary ID to loader mappings.
        _vocabularies: Cache of loaded vocabularies.
        _default_loader: Default loader for unregistered vocabularies.

    Example:
        >>> registry = VocabularyRegistry.get_instance()
        >>> registry.register_vocabulary("custom", my_loader)
        >>> vocab = registry.get_vocabulary("fuels")
        >>> entities = vocab.search_aliases("natural gas", EntityType.FUEL)
    """

    _instance: Optional["VocabularyRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "VocabularyRegistry":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry (only runs once due to singleton)."""
        if getattr(self, "_initialized", False):
            return

        self._loaders: Dict[str, VocabularyLoader] = {}
        self._vocabularies: Dict[str, Vocabulary] = {}
        self._default_loader: Optional[VocabularyLoader] = None
        self._vocab_lock = threading.RLock()
        self._snapshot_hash: Optional[str] = None

        # Initialize with default configuration
        self._initialize_defaults()
        self._initialized = True

        logger.info("VocabularyRegistry initialized")

    @classmethod
    def get_instance(cls) -> "VocabularyRegistry":
        """
        Get the singleton instance of VocabularyRegistry.

        Returns:
            The singleton VocabularyRegistry instance.

        Example:
            >>> registry = VocabularyRegistry.get_instance()
        """
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (primarily for testing).

        This clears the singleton state and allows a new instance
        to be created on the next access.
        """
        with cls._lock:
            cls._instance = None
        logger.info("VocabularyRegistry instance reset")

    def _initialize_defaults(self) -> None:
        """Initialize default loaders and configuration."""
        # Try to find vocabulary directory from environment or default locations
        vocab_dir = self._find_vocabulary_directory()

        if vocab_dir:
            logger.info("Using vocabulary directory", vocab_dir=str(vocab_dir))
            base_loader = YAMLVocabularyLoader(
                vocab_dir=str(vocab_dir),
                git_enabled=True,
                config=LoaderConfig(
                    validate_on_load=True,
                    verify_signature=False,  # Default to off, enable in production
                ),
            )
            self._default_loader = CachedLoader(
                base_loader,
                ttl_seconds=3600,  # 1 hour cache
                max_entries=50,
            )

            # Register built-in vocabularies with the default loader
            for vocab_id in BUILTIN_VOCABULARIES:
                self._loaders[vocab_id] = self._default_loader

    def _find_vocabulary_directory(self) -> Optional[Path]:
        """Find the vocabulary directory from environment or defaults."""
        # Check environment variable first
        env_dir = os.environ.get("GL_VOCABULARY_DIR")
        if env_dir:
            path = Path(env_dir)
            if path.exists():
                return path

        # Check common locations relative to package
        package_dir = Path(__file__).parent
        common_paths = [
            package_dir / "data" / "vocabularies",
            package_dir.parent / "data" / "vocabularies",
            package_dir.parent.parent / "data" / "vocabularies",
            Path.cwd() / "vocabularies",
            Path.cwd() / "data" / "vocabularies",
        ]

        for path in common_paths:
            if path.exists():
                return path

        logger.warning(
            "Vocabulary directory not found. Set GL_VOCABULARY_DIR environment "
            "variable or place vocabularies in a standard location."
        )
        return None

    def register_vocabulary(
        self,
        vocab_id: str,
        loader: VocabularyLoader,
        replace: bool = False,
    ) -> None:
        """
        Register a vocabulary loader.

        Args:
            vocab_id: The vocabulary identifier.
            loader: The loader to use for this vocabulary.
            replace: Whether to replace an existing registration.

        Raises:
            VocabularyError: If vocabulary is already registered and replace=False.

        Example:
            >>> loader = YAMLVocabularyLoader("/path/to/vocab")
            >>> registry.register_vocabulary("custom", loader)
        """
        with self._vocab_lock:
            if vocab_id in self._loaders and not replace:
                raise VocabularyError(
                    f"Vocabulary '{vocab_id}' is already registered. "
                    "Use replace=True to override.",
                    vocabulary_id=vocab_id,
                    code=GLNORMErrorCode.E500_VOCABULARY_VERSION_MISMATCH.value,
                )

            self._loaders[vocab_id] = loader

            # Invalidate cached vocabulary if it exists
            if vocab_id in self._vocabularies:
                del self._vocabularies[vocab_id]
                self._invalidate_snapshot()

            logger.info(
                "Registered vocabulary loader",
                vocab_id=vocab_id,
                loader_type=type(loader).__name__,
            )

    def unregister_vocabulary(self, vocab_id: str) -> bool:
        """
        Unregister a vocabulary loader.

        Args:
            vocab_id: The vocabulary identifier to unregister.

        Returns:
            True if unregistered, False if not found.

        Example:
            >>> registry.unregister_vocabulary("custom")
        """
        with self._vocab_lock:
            if vocab_id not in self._loaders:
                return False

            del self._loaders[vocab_id]

            # Remove cached vocabulary
            if vocab_id in self._vocabularies:
                del self._vocabularies[vocab_id]
                self._invalidate_snapshot()

            logger.info("Unregistered vocabulary", vocab_id=vocab_id)
            return True

    def get_vocabulary(
        self,
        vocab_id: str,
        version: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Vocabulary:
        """
        Get a vocabulary by ID with lazy loading.

        Args:
            vocab_id: The vocabulary identifier.
            version: Optional specific version to load.
            force_refresh: Force reload even if cached.

        Returns:
            The loaded Vocabulary object.

        Raises:
            VocabularyError: If vocabulary not found or loading fails.

        Example:
            >>> vocab = registry.get_vocabulary("fuels")
            >>> entity = vocab.get_entity("GL-FUEL-NATGAS")
        """
        cache_key = f"{vocab_id}:{version or 'latest'}"

        with self._vocab_lock:
            # Check cache unless force refresh
            if not force_refresh and cache_key in self._vocabularies:
                logger.debug("Returning cached vocabulary", vocab_id=vocab_id)
                return self._vocabularies[cache_key]

            # Find loader
            loader = self._loaders.get(vocab_id)
            if not loader and self._default_loader:
                loader = self._default_loader
            if not loader:
                raise VocabularyError(
                    f"No loader registered for vocabulary '{vocab_id}'",
                    vocabulary_id=vocab_id,
                    code=GLNORMErrorCode.E404_VOCABULARY_NOT_FOUND.value,
                )

            # Load vocabulary
            logger.info(
                "Loading vocabulary",
                vocab_id=vocab_id,
                version=version,
            )
            vocabulary = loader.load(vocab_id, version)

            # Cache the loaded vocabulary
            self._vocabularies[cache_key] = vocabulary
            self._invalidate_snapshot()

            return vocabulary

    def list_vocabularies(self) -> List[str]:
        """
        List all registered vocabulary IDs.

        Returns:
            Sorted list of vocabulary identifiers.

        Example:
            >>> vocabs = registry.list_vocabularies()
            >>> print(vocabs)
            ['fuels', 'materials', 'processes', 'units']
        """
        with self._vocab_lock:
            vocab_ids: Set[str] = set(self._loaders.keys())

            # Also include any from default loader
            if self._default_loader:
                try:
                    vocab_ids.update(self._default_loader.list_vocabularies())
                except Exception as e:
                    logger.warning(f"Failed to list from default loader: {e}")

            return sorted(vocab_ids)

    def get_entity(
        self,
        vocab_id: str,
        entity_id: str,
    ) -> Optional[Entity]:
        """
        Get an entity from a specific vocabulary.

        Args:
            vocab_id: The vocabulary identifier.
            entity_id: The entity identifier.

        Returns:
            The Entity if found, None otherwise.

        Raises:
            VocabularyError: If vocabulary loading fails.

        Example:
            >>> entity = registry.get_entity("fuels", "GL-FUEL-NATGAS")
            >>> print(entity.canonical_name)
            'Natural gas'
        """
        vocabulary = self.get_vocabulary(vocab_id)
        return vocabulary.get_entity(entity_id)

    def search_aliases(
        self,
        alias: str,
        entity_type: Optional[EntityType] = None,
        vocab_ids: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Entity]:
        """
        Search for entities matching an alias across vocabularies.

        Args:
            alias: The alias to search for.
            entity_type: Optional filter by entity type.
            vocab_ids: Optional list of vocabularies to search.
                      Searches all if not specified.
            limit: Maximum number of results.

        Returns:
            List of matching entities with their vocabulary context.

        Example:
            >>> entities = registry.search_aliases("nat gas", EntityType.FUEL)
            >>> for entity in entities:
            ...     print(f"{entity.id}: {entity.canonical_name}")
        """
        results: List[Entity] = []
        search_vocabs = vocab_ids or self.list_vocabularies()

        for vocab_id in search_vocabs:
            try:
                vocabulary = self.get_vocabulary(vocab_id)
                matches = vocabulary.search_aliases(
                    alias,
                    entity_type=entity_type,
                    limit=limit,
                )
                results.extend(matches)
            except VocabularyError as e:
                logger.warning(
                    f"Failed to search vocabulary '{vocab_id}': {e}"
                )
                continue

            if len(results) >= limit:
                break

        return results[:limit]

    def get_snapshot_hash(self) -> str:
        """
        Get a hash representing the current state of all vocabularies.

        This hash can be used for determinism verification - if the hash
        matches between runs, the vocabulary state is identical.

        Returns:
            SHA-256 hash of vocabulary state.

        Example:
            >>> hash1 = registry.get_snapshot_hash()
            >>> # ... later ...
            >>> hash2 = registry.get_snapshot_hash()
            >>> assert hash1 == hash2  # Vocabularies unchanged
        """
        with self._vocab_lock:
            if self._snapshot_hash is not None:
                return self._snapshot_hash

            # Compute hash from all loaded vocabularies
            hash_components: List[str] = []

            for vocab_id in sorted(self._vocabularies.keys()):
                vocab = self._vocabularies[vocab_id]
                vocab_hash = vocab.compute_signature()
                hash_components.append(f"{vocab_id}:{vocab.version}:{vocab_hash}")

            combined = "|".join(hash_components)
            self._snapshot_hash = hashlib.sha256(combined.encode()).hexdigest()

            logger.debug(
                "Computed snapshot hash",
                vocab_count=len(self._vocabularies),
                hash=self._snapshot_hash[:16],
            )

            return self._snapshot_hash

    def refresh(self, vocab_id: Optional[str] = None) -> None:
        """
        Refresh vocabularies by clearing cache and reloading.

        Args:
            vocab_id: Optional specific vocabulary to refresh.
                     Refreshes all if not specified.

        Example:
            >>> registry.refresh()  # Refresh all
            >>> registry.refresh("fuels")  # Refresh specific
        """
        with self._vocab_lock:
            if vocab_id:
                # Refresh specific vocabulary
                keys_to_remove = [
                    key for key in self._vocabularies.keys()
                    if key.startswith(f"{vocab_id}:")
                ]
                for key in keys_to_remove:
                    del self._vocabularies[key]

                # Invalidate loader cache if it's a CachedLoader
                loader = self._loaders.get(vocab_id, self._default_loader)
                if loader and isinstance(loader, CachedLoader):
                    loader.invalidate(vocab_id)

                logger.info("Refreshed vocabulary", vocab_id=vocab_id)
            else:
                # Refresh all
                self._vocabularies.clear()

                # Invalidate all loader caches
                for loader in self._loaders.values():
                    if isinstance(loader, CachedLoader):
                        loader.invalidate()
                if self._default_loader and isinstance(self._default_loader, CachedLoader):
                    self._default_loader.invalidate()

                logger.info("Refreshed all vocabularies")

            self._invalidate_snapshot()

    def _invalidate_snapshot(self) -> None:
        """Invalidate the cached snapshot hash."""
        self._snapshot_hash = None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry statistics.
        """
        with self._vocab_lock:
            loader_stats = {}
            for vocab_id, loader in self._loaders.items():
                if isinstance(loader, CachedLoader):
                    loader_stats[vocab_id] = loader.get_stats()

            return {
                "registered_vocabularies": list(self._loaders.keys()),
                "cached_vocabularies": list(self._vocabularies.keys()),
                "loader_stats": loader_stats,
                "snapshot_hash": self._snapshot_hash,
            }

    def set_default_loader(self, loader: VocabularyLoader) -> None:
        """
        Set the default loader for unregistered vocabularies.

        Args:
            loader: The loader to use as default.

        Example:
            >>> loader = RemoteVocabularyLoader("https://api.example.com")
            >>> registry.set_default_loader(CachedLoader(loader))
        """
        with self._vocab_lock:
            self._default_loader = loader
            logger.info(
                "Set default loader",
                loader_type=type(loader).__name__,
            )


# Convenience function for quick access
def get_registry() -> VocabularyRegistry:
    """
    Get the global vocabulary registry instance.

    Returns:
        The singleton VocabularyRegistry.

    Example:
        >>> from gl_normalizer_core.vocabulary.registry import get_registry
        >>> registry = get_registry()
        >>> vocab = registry.get_vocabulary("fuels")
    """
    return VocabularyRegistry.get_instance()


__all__ = [
    "VocabularyRegistry",
    "get_registry",
    "BUILTIN_VOCABULARIES",
]
