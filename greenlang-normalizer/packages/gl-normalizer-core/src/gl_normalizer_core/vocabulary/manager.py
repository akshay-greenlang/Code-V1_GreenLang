"""
Vocabulary Manager for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides the main VocabularyManager class that serves as the
primary interface for vocabulary operations. It combines the registry,
loaders, and validators into a cohesive API for vocabulary management.

Key Design Principles:
    - Lazy loading: Vocabularies are loaded on first access
    - Deterministic: Snapshot hashes enable reproducibility verification
    - Thread-safe: All operations are thread-safe
    - Extensible: Support for custom loaders and validators

Example:
    >>> from gl_normalizer_core.vocabulary.manager import VocabularyManager
    >>> manager = VocabularyManager()
    >>> vocab = manager.load_vocabulary("fuels")
    >>> entity = manager.get_entity("fuels", "GL-FUEL-NATGAS")
    >>> matches = manager.search_aliases("natural gas", "fuel")
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import json
import logging
import threading

from gl_normalizer_core.errors.codes import GLNORMErrorCode
from gl_normalizer_core.errors import VocabularyError
from gl_normalizer_core.vocabulary.models import (
    Vocabulary,
    Entity,
    Alias,
    EntityType,
    VocabularyMetadata,
)
from gl_normalizer_core.vocabulary.loader import (
    VocabularyLoader,
    YAMLVocabularyLoader,
    RemoteVocabularyLoader,
    CachedLoader,
    LoaderConfig,
)
from gl_normalizer_core.vocabulary.registry import (
    VocabularyRegistry,
    get_registry,
    BUILTIN_VOCABULARIES,
)
from gl_normalizer_core.vocabulary.validators import (
    validate_vocabulary,
    validate_signature,
    validate_schema,
    check_deprecations,
    ValidationResult,
    DeprecationWarning,
)

logger = logging.getLogger(__name__)


class VocabularyManager:
    """
    Main interface for vocabulary management.

    Provides high-level operations for loading, searching, and validating
    vocabularies. Wraps the VocabularyRegistry singleton with additional
    convenience methods and validation.

    Attributes:
        registry: The underlying vocabulary registry.
        validate_on_load: Whether to validate vocabularies when loaded.
        strict_validation: Whether to use strict validation mode.

    Example:
        >>> manager = VocabularyManager()
        >>> vocab = manager.load_vocabulary("fuels")
        >>> entity = manager.get_entity("fuels", "GL-FUEL-NATGAS")
        >>> print(entity.canonical_name)
        'Natural gas'
    """

    def __init__(
        self,
        registry: Optional[VocabularyRegistry] = None,
        validate_on_load: bool = True,
        strict_validation: bool = False,
        vocab_dir: Optional[str] = None,
        remote_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        """
        Initialize the VocabularyManager.

        Args:
            registry: Optional custom registry. Uses global singleton if not provided.
            validate_on_load: Whether to validate vocabularies when loaded.
            strict_validation: Whether to use strict validation mode.
            vocab_dir: Optional path to vocabulary directory.
            remote_url: Optional URL for remote vocabulary service.
            api_key: Optional API key for remote service.
            cache_ttl_seconds: Cache TTL in seconds.

        Example:
            >>> manager = VocabularyManager(
            ...     vocab_dir="/path/to/vocab",
            ...     validate_on_load=True,
            ... )
        """
        self.registry = registry or get_registry()
        self.validate_on_load = validate_on_load
        self.strict_validation = strict_validation
        self._lock = threading.RLock()

        # Configure loaders if paths provided
        if vocab_dir:
            self._setup_yaml_loader(vocab_dir, cache_ttl_seconds)

        if remote_url:
            self._setup_remote_loader(remote_url, api_key, cache_ttl_seconds)

        logger.info(
            "VocabularyManager initialized",
            validate_on_load=validate_on_load,
            strict_validation=strict_validation,
        )

    def _setup_yaml_loader(self, vocab_dir: str, cache_ttl: int) -> None:
        """Configure YAML loader for local files."""
        base_loader = YAMLVocabularyLoader(
            vocab_dir=vocab_dir,
            git_enabled=True,
            config=LoaderConfig(validate_on_load=False),  # We validate separately
        )
        cached_loader = CachedLoader(base_loader, ttl_seconds=cache_ttl)
        self.registry.set_default_loader(cached_loader)

        logger.info("Configured YAML loader", vocab_dir=vocab_dir)

    def _setup_remote_loader(
        self,
        base_url: str,
        api_key: Optional[str],
        cache_ttl: int,
    ) -> None:
        """Configure remote loader for API access."""
        base_loader = RemoteVocabularyLoader(
            base_url=base_url,
            api_key=api_key,
            config=LoaderConfig(validate_on_load=False),
        )
        cached_loader = CachedLoader(base_loader, ttl_seconds=cache_ttl)
        self.registry.set_default_loader(cached_loader)

        logger.info("Configured remote loader", base_url=base_url)

    def load_vocabulary(
        self,
        vocab_id: str,
        version: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Vocabulary:
        """
        Load a vocabulary by ID.

        Args:
            vocab_id: The vocabulary identifier (e.g., "fuels", "materials").
            version: Optional specific version to load.
            force_refresh: Force reload even if cached.

        Returns:
            The loaded Vocabulary object.

        Raises:
            VocabularyError: If vocabulary not found or validation fails.

        Example:
            >>> vocab = manager.load_vocabulary("fuels")
            >>> print(f"Loaded {vocab.id} version {vocab.version}")
            >>> print(f"Contains {vocab.entity_count()} entities")
        """
        start_time = datetime.utcnow()

        logger.info(
            "Loading vocabulary",
            vocab_id=vocab_id,
            version=version,
            force_refresh=force_refresh,
        )

        vocabulary = self.registry.get_vocabulary(
            vocab_id,
            version=version,
            force_refresh=force_refresh,
        )

        # Validate if configured
        if self.validate_on_load:
            result = validate_vocabulary(
                vocabulary,
                verify_signature=True,
                strict=self.strict_validation,
            )

            if not result.is_valid:
                errors_str = "; ".join(e.message for e in result.errors[:3])
                raise VocabularyError(
                    f"Vocabulary validation failed: {errors_str}",
                    vocabulary_id=vocab_id,
                    version=vocabulary.version,
                    code=GLNORMErrorCode.E502_VOCABULARY_CORRUPTED.value,
                )

            if result.warnings:
                for warning in result.warnings[:5]:
                    logger.warning(
                        "Vocabulary validation warning",
                        vocab_id=vocab_id,
                        warning=warning.message,
                    )

        load_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        logger.info(
            "Vocabulary loaded",
            vocab_id=vocab_id,
            version=vocabulary.version,
            entity_count=vocabulary.entity_count(),
            alias_count=vocabulary.alias_count(),
            load_time_ms=round(load_time_ms, 2),
        )

        return vocabulary

    def get_entity(
        self,
        vocab_id: str,
        entity_id: str,
    ) -> Optional[Entity]:
        """
        Get an entity from a vocabulary.

        Args:
            vocab_id: The vocabulary identifier.
            entity_id: The entity identifier.

        Returns:
            The Entity if found, None otherwise.

        Example:
            >>> entity = manager.get_entity("fuels", "GL-FUEL-NATGAS")
            >>> if entity:
            ...     print(f"Found: {entity.canonical_name}")
            ...     print(f"Aliases: {entity.aliases}")
        """
        vocabulary = self.load_vocabulary(vocab_id)
        entity = vocabulary.get_entity(entity_id)

        if entity:
            logger.debug(
                "Entity retrieved",
                vocab_id=vocab_id,
                entity_id=entity_id,
            )
        else:
            logger.debug(
                "Entity not found",
                vocab_id=vocab_id,
                entity_id=entity_id,
            )

        return entity

    def search_aliases(
        self,
        alias: str,
        entity_type: Optional[Union[str, EntityType]] = None,
        vocab_ids: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Entity]:
        """
        Search for entities matching an alias.

        Searches across one or more vocabularies for entities that
        match the given alias string.

        Args:
            alias: The alias to search for.
            entity_type: Optional filter by entity type (string or EntityType).
            vocab_ids: Optional list of vocabularies to search.
                      Searches all if not specified.
            limit: Maximum number of results.

        Returns:
            List of matching Entity objects, ordered by relevance.

        Example:
            >>> matches = manager.search_aliases("nat gas", "fuel")
            >>> for entity in matches:
            ...     print(f"{entity.id}: {entity.canonical_name}")
        """
        # Convert string to EntityType if needed
        entity_type_enum: Optional[EntityType] = None
        if entity_type:
            if isinstance(entity_type, str):
                try:
                    entity_type_enum = EntityType(entity_type.lower())
                except ValueError:
                    logger.warning(
                        f"Unknown entity type: {entity_type}, ignoring filter"
                    )
            else:
                entity_type_enum = entity_type

        results = self.registry.search_aliases(
            alias,
            entity_type=entity_type_enum,
            vocab_ids=vocab_ids,
            limit=limit,
        )

        logger.debug(
            "Alias search complete",
            alias=alias,
            entity_type=entity_type,
            result_count=len(results),
        )

        return results

    def get_snapshot_hash(self) -> str:
        """
        Get a hash representing the current vocabulary state.

        This hash can be used for determinism verification. If the hash
        matches between runs, the vocabulary state is identical.

        Returns:
            SHA-256 hash string representing vocabulary state.

        Example:
            >>> hash1 = manager.get_snapshot_hash()
            >>> # ... perform operations ...
            >>> hash2 = manager.get_snapshot_hash()
            >>> if hash1 != hash2:
            ...     print("Vocabulary state changed!")
        """
        return self.registry.get_snapshot_hash()

    def refresh(self, vocab_id: Optional[str] = None) -> None:
        """
        Refresh vocabularies by clearing cache and reloading.

        Args:
            vocab_id: Optional specific vocabulary to refresh.
                     Refreshes all if not specified.

        Example:
            >>> manager.refresh()  # Refresh all vocabularies
            >>> manager.refresh("fuels")  # Refresh only fuels
        """
        self.registry.refresh(vocab_id)

        logger.info(
            "Vocabularies refreshed",
            vocab_id=vocab_id or "all",
        )

    def list_vocabularies(self) -> List[str]:
        """
        List all available vocabulary IDs.

        Returns:
            Sorted list of vocabulary identifiers.

        Example:
            >>> vocabs = manager.list_vocabularies()
            >>> print(vocabs)
            ['fuels', 'materials', 'processes', 'units']
        """
        return self.registry.list_vocabularies()

    def register_vocabulary(
        self,
        vocab_id: str,
        loader: VocabularyLoader,
        replace: bool = False,
    ) -> None:
        """
        Register a custom vocabulary loader.

        Args:
            vocab_id: The vocabulary identifier.
            loader: The loader to use for this vocabulary.
            replace: Whether to replace an existing registration.

        Example:
            >>> loader = YAMLVocabularyLoader("/custom/path")
            >>> manager.register_vocabulary("custom", loader)
        """
        self.registry.register_vocabulary(vocab_id, loader, replace=replace)

        logger.info(
            "Vocabulary registered",
            vocab_id=vocab_id,
            loader_type=type(loader).__name__,
        )

    def validate(
        self,
        vocab_id: str,
        verify_signature: bool = True,
    ) -> ValidationResult:
        """
        Validate a vocabulary.

        Performs comprehensive validation including signature verification,
        schema validation, and deprecation checking.

        Args:
            vocab_id: The vocabulary identifier.
            verify_signature: Whether to verify the vocabulary signature.

        Returns:
            ValidationResult with all validation findings.

        Example:
            >>> result = manager.validate("fuels")
            >>> if not result.is_valid:
            ...     for error in result.errors:
            ...         print(f"Error: {error.message}")
        """
        vocabulary = self.registry.get_vocabulary(vocab_id)

        return validate_vocabulary(
            vocabulary,
            verify_signature=verify_signature,
            strict=self.strict_validation,
        )

    def get_deprecations(self, vocab_id: str) -> List[DeprecationWarning]:
        """
        Get deprecation warnings for a vocabulary.

        Args:
            vocab_id: The vocabulary identifier.

        Returns:
            List of DeprecationWarning objects.

        Example:
            >>> warnings = manager.get_deprecations("fuels")
            >>> for w in warnings:
            ...     print(f"Deprecated: {w.entity_name}")
            ...     if w.replacement_id:
            ...         print(f"  Use: {w.replacement_id}")
        """
        vocabulary = self.load_vocabulary(vocab_id)
        return check_deprecations(vocabulary)

    def get_entity_by_alias(
        self,
        alias: str,
        vocab_id: Optional[str] = None,
        entity_type: Optional[Union[str, EntityType]] = None,
    ) -> Optional[Entity]:
        """
        Get an entity by its alias.

        This is a convenience method that returns the best matching
        entity for a given alias.

        Args:
            alias: The alias to look up.
            vocab_id: Optional vocabulary to search in.
            entity_type: Optional filter by entity type.

        Returns:
            The best matching Entity, or None if not found.

        Example:
            >>> entity = manager.get_entity_by_alias("nat gas", "fuels")
            >>> if entity:
            ...     print(entity.canonical_name)  # "Natural gas"
        """
        if vocab_id:
            vocabulary = self.load_vocabulary(vocab_id)
            entity_type_enum = None
            if entity_type:
                if isinstance(entity_type, str):
                    try:
                        entity_type_enum = EntityType(entity_type.lower())
                    except ValueError:
                        pass
                else:
                    entity_type_enum = entity_type

            return vocabulary.get_entity_by_alias(
                alias,
                entity_type=entity_type_enum,
            )

        # Search across all vocabularies
        matches = self.search_aliases(alias, entity_type=entity_type, limit=1)
        return matches[0] if matches else None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get manager and registry statistics.

        Returns:
            Dictionary with statistics about loaded vocabularies and cache.

        Example:
            >>> stats = manager.get_stats()
            >>> print(f"Loaded vocabularies: {len(stats['cached_vocabularies'])}")
        """
        registry_stats = self.registry.get_stats()

        return {
            **registry_stats,
            "validate_on_load": self.validate_on_load,
            "strict_validation": self.strict_validation,
            "snapshot_hash": self.get_snapshot_hash(),
        }

    def resolve_entity(
        self,
        raw_name: str,
        entity_type: Optional[Union[str, EntityType]] = None,
        vocab_id: Optional[str] = None,
        min_confidence: float = 0.8,
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a raw name to a canonical entity with confidence score.

        This method provides entity resolution with match confidence,
        suitable for audit trails.

        Args:
            raw_name: The raw name to resolve.
            entity_type: Optional filter by entity type.
            vocab_id: Optional vocabulary to search in.
            min_confidence: Minimum confidence threshold.

        Returns:
            Dictionary with entity and match metadata, or None if no match.

        Example:
            >>> result = manager.resolve_entity("Nat Gas", "fuel")
            >>> if result:
            ...     print(f"Resolved: {result['entity'].canonical_name}")
            ...     print(f"Confidence: {result['confidence']}")
            ...     print(f"Match method: {result['match_method']}")
        """
        # Determine search vocabularies
        vocab_ids = [vocab_id] if vocab_id else None

        # Search for matches
        matches = self.search_aliases(
            raw_name,
            entity_type=entity_type,
            vocab_ids=vocab_ids,
            limit=5,
        )

        if not matches:
            return None

        # Calculate confidence based on match quality
        best_match = matches[0]
        raw_lower = raw_name.lower().strip()
        canonical_lower = best_match.canonical_name.lower()

        # Determine match method and confidence
        if raw_lower == canonical_lower:
            match_method = "exact_name"
            confidence = 1.0
        elif raw_lower in best_match.get_all_aliases():
            match_method = "alias"
            confidence = 1.0
        elif canonical_lower.startswith(raw_lower) or raw_lower.startswith(canonical_lower):
            match_method = "rule"
            confidence = 0.95
        elif raw_lower in canonical_lower or canonical_lower in raw_lower:
            match_method = "rule"
            confidence = 0.90
        else:
            match_method = "fuzzy"
            # Calculate simple similarity
            common = set(raw_lower.split()) & set(canonical_lower.split())
            total = set(raw_lower.split()) | set(canonical_lower.split())
            confidence = len(common) / len(total) if total else 0

        if confidence < min_confidence:
            return None

        return {
            "entity": best_match,
            "confidence": confidence,
            "match_method": match_method,
            "raw_name": raw_name,
            "candidates": matches[:3],
        }


# Create a default manager instance for convenience
_default_manager: Optional[VocabularyManager] = None
_manager_lock = threading.Lock()


def get_manager() -> VocabularyManager:
    """
    Get the default VocabularyManager instance.

    Returns:
        The default VocabularyManager singleton.

    Example:
        >>> from gl_normalizer_core.vocabulary.manager import get_manager
        >>> manager = get_manager()
        >>> vocab = manager.load_vocabulary("fuels")
    """
    global _default_manager

    if _default_manager is None:
        with _manager_lock:
            if _default_manager is None:
                _default_manager = VocabularyManager()

    return _default_manager


__all__ = [
    "VocabularyManager",
    "get_manager",
]
