"""
Vocabulary data models for GL-FOUND-X-003 Unit & Reference Normalizer.

This module defines the Pydantic models for vocabulary management, including
vocabularies, entities, aliases, and metadata. These models support
deterministic entity resolution with full provenance tracking.

Key Design Principles:
    - Immutable vocabulary records with version tracking
    - SHA-256 signature verification for integrity
    - Git-compatible versioning for vocabulary files
    - Support for deprecated entities with replacement mapping

Example:
    >>> from gl_normalizer_core.vocabulary.models import Vocabulary, Entity, Alias
    >>> entity = Entity(
    ...     id="GL-FUEL-NATGAS",
    ...     canonical_name="Natural gas",
    ...     aliases=["Nat Gas", "NG", "Methane"],
    ...     properties={"density": 0.8, "carbon_content": 0.75}
    ... )
    >>> vocab = Vocabulary(
    ...     id="fuels",
    ...     version="2026.01.0",
    ...     entities={"GL-FUEL-NATGAS": entity}
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import hashlib
import json

from pydantic import BaseModel, Field, field_validator, model_validator


class EntityType(str, Enum):
    """
    Type of entity in the vocabulary.

    Attributes:
        FUEL: Fuel type entity (e.g., natural gas, diesel).
        MATERIAL: Material entity (e.g., Portland cement, steel).
        PROCESS: Process entity (e.g., electric arc furnace).
        UNIT: Unit of measurement entity.
        EMISSION_FACTOR: Emission factor entity.
        CUSTOM: Custom entity type for extensibility.
    """

    FUEL = "fuel"
    MATERIAL = "material"
    PROCESS = "process"
    UNIT = "unit"
    EMISSION_FACTOR = "emission_factor"
    CUSTOM = "custom"


class DeprecationInfo(BaseModel):
    """
    Information about a deprecated entity.

    Attributes:
        deprecated_at: Date when the entity was deprecated.
        reason: Reason for deprecation.
        replacement_id: ID of the replacement entity, if available.
        removal_date: Planned removal date, if known.

    Example:
        >>> info = DeprecationInfo(
        ...     deprecated_at=datetime(2025, 1, 1),
        ...     reason="Merged with GL-FUEL-NATGAS-PIPELINE",
        ...     replacement_id="GL-FUEL-NATGAS-PIPELINE"
        ... )
    """

    deprecated_at: datetime = Field(
        ...,
        description="Date when the entity was deprecated",
    )
    reason: str = Field(
        ...,
        max_length=500,
        description="Reason for deprecation",
    )
    replacement_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="ID of the replacement entity, if available",
    )
    removal_date: Optional[datetime] = Field(
        default=None,
        description="Planned removal date, if known",
    )


class Alias(BaseModel):
    """
    Alias mapping for entity resolution.

    Aliases provide alternative names for entities with priority-based
    matching when multiple vocabularies contain the same alias.

    Attributes:
        alias: The alias string (case-normalized during matching).
        canonical_id: ID of the canonical entity this alias maps to.
        priority: Priority for conflict resolution (higher = preferred).
        locale: Optional locale for locale-specific aliases.
        source: Source of this alias (e.g., "official", "community").

    Example:
        >>> alias = Alias(
        ...     alias="Nat Gas",
        ...     canonical_id="GL-FUEL-NATGAS",
        ...     priority=100
        ... )
    """

    alias: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The alias string",
    )
    canonical_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="ID of the canonical entity this alias maps to",
    )
    priority: int = Field(
        default=0,
        ge=-1000,
        le=1000,
        description="Priority for conflict resolution (higher = preferred)",
    )
    locale: Optional[str] = Field(
        default=None,
        pattern=r"^[a-z]{2}(-[A-Z]{2})?$",
        description="Optional locale code (e.g., 'en', 'en-US')",
    )
    source: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Source of this alias",
    )

    @field_validator("alias")
    @classmethod
    def normalize_alias(cls, v: str) -> str:
        """Normalize alias by stripping whitespace."""
        return v.strip()

    def matches(self, query: str, case_sensitive: bool = False) -> bool:
        """
        Check if this alias matches a query string.

        Args:
            query: The query string to match against.
            case_sensitive: Whether to perform case-sensitive matching.

        Returns:
            True if the alias matches the query.
        """
        if case_sensitive:
            return self.alias == query.strip()
        return self.alias.lower() == query.strip().lower()


class Entity(BaseModel):
    """
    Entity in a vocabulary.

    Entities represent canonical reference data items such as fuels,
    materials, or processes. Each entity has a unique ID, canonical name,
    and optional aliases for fuzzy matching.

    Attributes:
        id: Unique identifier for this entity (e.g., "GL-FUEL-NATGAS").
        canonical_name: Official canonical name for this entity.
        entity_type: Type of entity (fuel, material, process, etc.).
        aliases: List of alternative names for this entity.
        properties: Key-value properties for this entity.
        deprecated: Whether this entity is deprecated.
        deprecation_info: Deprecation details if deprecated.
        effective_date: Date when this entity became effective.
        expiration_date: Date when this entity expires (if applicable).
        metadata: Additional metadata for extensibility.

    Example:
        >>> entity = Entity(
        ...     id="GL-FUEL-NATGAS",
        ...     canonical_name="Natural gas",
        ...     entity_type=EntityType.FUEL,
        ...     aliases=["Nat Gas", "NG", "Methane"],
        ...     properties={"density_kg_m3": 0.8, "carbon_content": 0.75}
        ... )
    """

    id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Unique identifier for this entity",
    )
    canonical_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Official canonical name for this entity",
    )
    entity_type: Optional[EntityType] = Field(
        default=None,
        description="Type of entity (fuel, material, process, etc.)",
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="List of alternative names for this entity",
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value properties for this entity",
    )
    deprecated: bool = Field(
        default=False,
        description="Whether this entity is deprecated",
    )
    deprecation_info: Optional[DeprecationInfo] = Field(
        default=None,
        description="Deprecation details if deprecated",
    )
    effective_date: Optional[datetime] = Field(
        default=None,
        description="Date when this entity became effective",
    )
    expiration_date: Optional[datetime] = Field(
        default=None,
        description="Date when this entity expires (if applicable)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for extensibility",
    )

    @model_validator(mode="after")
    def validate_deprecation_consistency(self) -> "Entity":
        """Validate that deprecation info is provided when deprecated."""
        if self.deprecated and not self.deprecation_info:
            # Allow deprecated without info for backwards compatibility
            pass
        if not self.deprecated and self.deprecation_info:
            raise ValueError(
                "deprecation_info provided but deprecated is False. "
                "Set deprecated=True when providing deprecation info."
            )
        return self

    def is_active(self, as_of: Optional[datetime] = None) -> bool:
        """
        Check if this entity is active (not deprecated and within validity period).

        Args:
            as_of: Date to check against. Defaults to now.

        Returns:
            True if the entity is active.
        """
        if self.deprecated:
            return False

        check_date = as_of or datetime.utcnow()

        if self.effective_date and check_date < self.effective_date:
            return False

        if self.expiration_date and check_date > self.expiration_date:
            return False

        return True

    def get_all_aliases(self) -> Set[str]:
        """
        Get all aliases including canonical name (lowercased for matching).

        Returns:
            Set of all aliases including the canonical name.
        """
        aliases = {alias.lower() for alias in self.aliases}
        aliases.add(self.canonical_name.lower())
        return aliases

    def to_hash_string(self) -> str:
        """
        Generate a deterministic string representation for hashing.

        Returns:
            JSON string with sorted keys for reproducible hashing.
        """
        data = {
            "id": self.id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type.value if self.entity_type else None,
            "aliases": sorted(self.aliases),
            "properties": self.properties,
            "deprecated": self.deprecated,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"))


class VocabularyMetadata(BaseModel):
    """
    Metadata for a vocabulary.

    Contains version information, signatures for integrity verification,
    and validity period for the vocabulary.

    Attributes:
        version: Semantic version of the vocabulary (e.g., "2026.01.0").
        signature: SHA-256 signature of the vocabulary content.
        created_at: Timestamp when this vocabulary version was created.
        created_by: User or system that created this version.
        expires_at: Expiration timestamp for this vocabulary version.
        git_commit: Git commit hash for version-controlled vocabularies.
        git_tag: Git tag for this vocabulary version.
        description: Human-readable description of this version.
        changelog: List of changes in this version.

    Example:
        >>> metadata = VocabularyMetadata(
        ...     version="2026.01.0",
        ...     signature="sha256:abc123...",
        ...     created_at=datetime.utcnow(),
        ...     git_commit="a1b2c3d4"
        ... )
    """

    version: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$",
        description="Semantic version of the vocabulary",
    )
    signature: Optional[str] = Field(
        default=None,
        pattern=r"^sha256:[a-f0-9]{64}$",
        description="SHA-256 signature of the vocabulary content",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when this vocabulary version was created",
    )
    created_by: Optional[str] = Field(
        default=None,
        max_length=100,
        description="User or system that created this version",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Expiration timestamp for this vocabulary version",
    )
    git_commit: Optional[str] = Field(
        default=None,
        pattern=r"^[a-f0-9]{7,40}$",
        description="Git commit hash for version-controlled vocabularies",
    )
    git_tag: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Git tag for this vocabulary version",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Human-readable description of this version",
    )
    changelog: List[str] = Field(
        default_factory=list,
        description="List of changes in this version",
    )

    def is_expired(self, as_of: Optional[datetime] = None) -> bool:
        """
        Check if this vocabulary version has expired.

        Args:
            as_of: Date to check against. Defaults to now.

        Returns:
            True if the vocabulary has expired.
        """
        if not self.expires_at:
            return False
        check_date = as_of or datetime.utcnow()
        return check_date > self.expires_at


class Vocabulary(BaseModel):
    """
    Complete vocabulary containing entities and aliases.

    A vocabulary is a collection of canonical entities with their aliases,
    versioned and signed for integrity verification. Vocabularies support
    Git-based version control and lazy loading.

    Attributes:
        id: Unique identifier for this vocabulary (e.g., "fuels", "materials").
        version: Semantic version string.
        entities: Dictionary of entity ID to Entity objects.
        aliases: List of explicit alias mappings.
        metadata: Vocabulary metadata including signature and timestamps.
        source_path: Path to the vocabulary source file (for YAML loaders).
        parent_id: ID of parent vocabulary for inheritance.

    Example:
        >>> from gl_normalizer_core.vocabulary.models import Vocabulary, Entity
        >>> vocab = Vocabulary(
        ...     id="fuels",
        ...     version="2026.01.0",
        ...     entities={
        ...         "GL-FUEL-NATGAS": Entity(
        ...             id="GL-FUEL-NATGAS",
        ...             canonical_name="Natural gas"
        ...         )
        ...     }
        ... )
    """

    id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_-]*$",
        description="Unique identifier for this vocabulary",
    )
    version: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Semantic version string",
    )
    entities: Dict[str, Entity] = Field(
        default_factory=dict,
        description="Dictionary of entity ID to Entity objects",
    )
    aliases: List[Alias] = Field(
        default_factory=list,
        description="List of explicit alias mappings",
    )
    metadata: Optional[VocabularyMetadata] = Field(
        default=None,
        description="Vocabulary metadata including signature and timestamps",
    )
    source_path: Optional[str] = Field(
        default=None,
        description="Path to the vocabulary source file",
    )
    parent_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="ID of parent vocabulary for inheritance",
    )

    @model_validator(mode="after")
    def build_alias_index(self) -> "Vocabulary":
        """Build the internal alias index after initialization."""
        # Index is built lazily on first access via get_entity_by_alias
        return self

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by its ID.

        Args:
            entity_id: The entity ID to look up.

        Returns:
            The Entity if found, None otherwise.
        """
        return self.entities.get(entity_id)

    def get_entity_by_alias(
        self,
        alias: str,
        entity_type: Optional[EntityType] = None,
        case_sensitive: bool = False,
    ) -> Optional[Entity]:
        """
        Get an entity by an alias.

        Args:
            alias: The alias to search for.
            entity_type: Optional filter by entity type.
            case_sensitive: Whether to perform case-sensitive matching.

        Returns:
            The Entity if found, None otherwise.
        """
        search_alias = alias if case_sensitive else alias.lower()

        # Check explicit alias mappings first (higher priority)
        for alias_obj in sorted(self.aliases, key=lambda a: -a.priority):
            if alias_obj.matches(alias, case_sensitive):
                entity = self.entities.get(alias_obj.canonical_id)
                if entity and (entity_type is None or entity.entity_type == entity_type):
                    return entity

        # Check entity aliases
        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue

            if case_sensitive:
                if alias in entity.aliases or alias == entity.canonical_name:
                    return entity
            else:
                if search_alias in entity.get_all_aliases():
                    return entity

        return None

    def search_aliases(
        self,
        query: str,
        entity_type: Optional[EntityType] = None,
        limit: int = 10,
    ) -> List[Entity]:
        """
        Search for entities matching a query string.

        Args:
            query: The search query.
            entity_type: Optional filter by entity type.
            limit: Maximum number of results to return.

        Returns:
            List of matching entities, ordered by relevance.
        """
        results: List[tuple[int, Entity]] = []
        query_lower = query.lower()

        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue

            # Calculate match score
            score = 0

            # Exact canonical name match (highest priority)
            if entity.canonical_name.lower() == query_lower:
                score = 1000
            # Exact alias match
            elif query_lower in entity.get_all_aliases():
                score = 900
            # Canonical name starts with query
            elif entity.canonical_name.lower().startswith(query_lower):
                score = 800
            # Canonical name contains query
            elif query_lower in entity.canonical_name.lower():
                score = 700
            # Any alias starts with query
            elif any(a.startswith(query_lower) for a in entity.get_all_aliases()):
                score = 600
            # Any alias contains query
            elif any(query_lower in a for a in entity.get_all_aliases()):
                score = 500

            if score > 0:
                results.append((score, entity))

        # Sort by score descending, then by canonical name
        results.sort(key=lambda x: (-x[0], x[1].canonical_name))

        return [entity for _, entity in results[:limit]]

    def get_active_entities(
        self,
        as_of: Optional[datetime] = None,
        entity_type: Optional[EntityType] = None,
    ) -> List[Entity]:
        """
        Get all active (non-deprecated) entities.

        Args:
            as_of: Date to check against. Defaults to now.
            entity_type: Optional filter by entity type.

        Returns:
            List of active entities.
        """
        entities = []
        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            if entity.is_active(as_of):
                entities.append(entity)
        return entities

    def get_deprecated_entities(self) -> List[Entity]:
        """
        Get all deprecated entities.

        Returns:
            List of deprecated entities.
        """
        return [e for e in self.entities.values() if e.deprecated]

    def compute_signature(self) -> str:
        """
        Compute SHA-256 signature of the vocabulary content.

        Returns:
            Signature string in format "sha256:<hash>".
        """
        # Create deterministic representation
        entity_hashes = []
        for entity_id in sorted(self.entities.keys()):
            entity_hashes.append(self.entities[entity_id].to_hash_string())

        alias_strs = []
        for alias in sorted(self.aliases, key=lambda a: (a.alias, a.canonical_id)):
            alias_strs.append(f"{alias.alias}:{alias.canonical_id}:{alias.priority}")

        content = {
            "id": self.id,
            "version": self.version,
            "entities": entity_hashes,
            "aliases": alias_strs,
        }

        content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
        hash_value = hashlib.sha256(content_str.encode()).hexdigest()

        return f"sha256:{hash_value}"

    def verify_signature(self) -> bool:
        """
        Verify the vocabulary signature matches its content.

        Returns:
            True if signature is valid, False otherwise.
        """
        if not self.metadata or not self.metadata.signature:
            return False

        computed = self.compute_signature()
        return computed == self.metadata.signature

    def entity_count(self) -> int:
        """Return the number of entities in this vocabulary."""
        return len(self.entities)

    def alias_count(self) -> int:
        """Return the total number of aliases (explicit + entity aliases)."""
        explicit = len(self.aliases)
        entity_aliases = sum(len(e.aliases) for e in self.entities.values())
        return explicit + entity_aliases


# Type aliases for convenience
EntityDict = Dict[str, Entity]
AliasList = List[Alias]


__all__ = [
    "EntityType",
    "DeprecationInfo",
    "Alias",
    "Entity",
    "VocabularyMetadata",
    "Vocabulary",
    "EntityDict",
    "AliasList",
]
