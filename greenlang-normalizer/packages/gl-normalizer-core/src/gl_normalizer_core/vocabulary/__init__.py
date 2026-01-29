"""
Vocabulary Management module for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides comprehensive vocabulary management capabilities for
the GreenLang Normalizer, including:

- Vocabulary data models (Entity, Alias, Vocabulary)
- Multiple loader backends (YAML, Remote API)
- Caching with TTL-based invalidation
- Singleton registry for global access
- Comprehensive validation and deprecation checking

Key Design Principles:
    - Lazy loading: Vocabularies are loaded on first access
    - Deterministic: Snapshot hashes enable reproducibility verification
    - Thread-safe: All operations are thread-safe
    - Extensible: Support for custom loaders and validators
    - Git integration: Support for Git-versioned vocabulary files

Example:
    >>> from gl_normalizer_core.vocabulary import VocabularyManager
    >>> manager = VocabularyManager()
    >>> vocab = manager.load_vocabulary("fuels")
    >>> entity = manager.get_entity("fuels", "GL-FUEL-NATGAS")
    >>> print(entity.canonical_name)
    'Natural gas'

    # Search for entities by alias
    >>> matches = manager.search_aliases("nat gas", "fuel")
    >>> for entity in matches:
    ...     print(f"{entity.id}: {entity.canonical_name}")

    # Verify determinism with snapshot hash
    >>> hash1 = manager.get_snapshot_hash()
    >>> # ... perform operations ...
    >>> hash2 = manager.get_snapshot_hash()
    >>> assert hash1 == hash2  # Vocabularies unchanged

    # Check for deprecations
    >>> warnings = manager.get_deprecations("fuels")
    >>> for w in warnings:
    ...     print(f"Deprecated: {w.entity_name} - {w.reason}")

Built-in Vocabularies:
    - fuels: Fuel type entities (natural gas, diesel, etc.)
    - materials: Material entities (Portland cement, steel, etc.)
    - processes: Process entities (electric arc furnace, etc.)
    - units: Unit of measurement entities

See Also:
    - VocabularyManager: Main interface for vocabulary operations
    - VocabularyRegistry: Singleton registry for loader management
    - YAMLVocabularyLoader: Load vocabularies from YAML files
    - RemoteVocabularyLoader: Load vocabularies from API
"""

from gl_normalizer_core.vocabulary.models import (
    EntityType,
    DeprecationInfo,
    Alias,
    Entity,
    VocabularyMetadata,
    Vocabulary,
)

from gl_normalizer_core.vocabulary.loader import (
    LoaderConfig,
    VocabularyLoader,
    YAMLVocabularyLoader,
    RemoteVocabularyLoader,
    CachedLoader,
)

from gl_normalizer_core.vocabulary.registry import (
    VocabularyRegistry,
    get_registry,
    BUILTIN_VOCABULARIES,
)

from gl_normalizer_core.vocabulary.validators import (
    ValidationSeverity,
    ValidationError,
    DeprecationWarning,
    ValidationResult,
    validate_signature,
    validate_schema,
    check_deprecations,
    validate_vocabulary,
)

from gl_normalizer_core.vocabulary.manager import (
    VocabularyManager,
    get_manager,
)


__all__ = [
    # Models
    "EntityType",
    "DeprecationInfo",
    "Alias",
    "Entity",
    "VocabularyMetadata",
    "Vocabulary",
    # Loaders
    "LoaderConfig",
    "VocabularyLoader",
    "YAMLVocabularyLoader",
    "RemoteVocabularyLoader",
    "CachedLoader",
    # Registry
    "VocabularyRegistry",
    "get_registry",
    "BUILTIN_VOCABULARIES",
    # Validators
    "ValidationSeverity",
    "ValidationError",
    "DeprecationWarning",
    "ValidationResult",
    "validate_signature",
    "validate_schema",
    "check_deprecations",
    "validate_vocabulary",
    # Manager
    "VocabularyManager",
    "get_manager",
]
