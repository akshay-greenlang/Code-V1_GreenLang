"""
Vocabulary Management module for the GreenLang Normalizer.

This module provides vocabulary loading, versioning, and management
capabilities for reference data used in normalization operations.

Example:
    >>> from gl_normalizer_core.vocab import VocabularyManager
    >>> manager = VocabularyManager("/path/to/vocab")
    >>> fuels = manager.get_vocabulary("fuels")
    >>> print(fuels.version)
    1.2.0
"""

from typing import Any, Dict, List, Optional, Set
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

from pydantic import BaseModel, Field
import structlog
import yaml

from gl_normalizer_core.errors import VocabularyError

logger = structlog.get_logger(__name__)


class VocabStatus(str, Enum):
    """Status of a vocabulary."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DRAFT = "draft"
    ARCHIVED = "archived"


class VocabEntry(BaseModel):
    """
    A single entry in a vocabulary.

    Attributes:
        id: Unique identifier within vocabulary
        name: Canonical name
        aliases: Alternative names/spellings
        category: Category within vocabulary
        description: Entry description
        metadata: Additional metadata
        deprecated: Whether entry is deprecated
        deprecated_by: ID of replacement entry if deprecated
    """

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Canonical name")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    category: Optional[str] = Field(None, description="Category")
    description: Optional[str] = Field(None, description="Description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    deprecated: bool = Field(default=False, description="Whether deprecated")
    deprecated_by: Optional[str] = Field(None, description="Replacement entry ID")


class VocabVersion(BaseModel):
    """
    Version information for a vocabulary.

    Attributes:
        version: Semantic version string
        released_at: Release timestamp
        changelog: Change description
        previous_version: Previous version string
        hash: Content hash for integrity
    """

    version: str = Field(..., description="Semantic version (e.g., 1.2.0)")
    released_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Release timestamp"
    )
    changelog: Optional[str] = Field(None, description="Change description")
    previous_version: Optional[str] = Field(None, description="Previous version")
    hash: str = Field(..., description="Content hash for integrity")


class Vocabulary(BaseModel):
    """
    A complete vocabulary with entries and metadata.

    Attributes:
        id: Vocabulary identifier
        name: Human-readable name
        description: Vocabulary description
        version: Current version info
        entries: Vocabulary entries
        status: Vocabulary status
        owner: Owner/maintainer
        tags: Classification tags
    """

    id: str = Field(..., description="Vocabulary identifier")
    name: str = Field(..., description="Vocabulary name")
    description: Optional[str] = Field(None, description="Description")
    version: VocabVersion = Field(..., description="Version information")
    entries: List[VocabEntry] = Field(default_factory=list, description="Entries")
    status: VocabStatus = Field(default=VocabStatus.ACTIVE, description="Status")
    owner: Optional[str] = Field(None, description="Owner/maintainer")
    tags: List[str] = Field(default_factory=list, description="Tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")

    def get_entry(self, entry_id: str) -> Optional[VocabEntry]:
        """Get an entry by ID."""
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def get_entry_by_name(self, name: str) -> Optional[VocabEntry]:
        """Get an entry by name (case-insensitive)."""
        name_lower = name.lower()
        for entry in self.entries:
            if entry.name.lower() == name_lower:
                return entry
            for alias in entry.aliases:
                if alias.lower() == name_lower:
                    return entry
        return None

    def get_active_entries(self) -> List[VocabEntry]:
        """Get all non-deprecated entries."""
        return [e for e in self.entries if not e.deprecated]

    @property
    def entry_count(self) -> int:
        """Get number of entries."""
        return len(self.entries)

    def calculate_hash(self) -> str:
        """Calculate content hash for integrity."""
        content = json.dumps(
            [e.model_dump() for e in self.entries],
            sort_keys=True,
            default=str
        )
        return hashlib.sha256(content.encode()).hexdigest()


class VocabularyManager:
    """
    Manager for vocabulary loading and versioning.

    This class provides capabilities for loading vocabularies from
    files, managing versions, and caching for performance.

    Attributes:
        vocab_path: Base path for vocabulary files
        cache: Loaded vocabulary cache
        strict_mode: Fail on validation errors

    Example:
        >>> manager = VocabularyManager("/path/to/vocab")
        >>> fuels = manager.load_vocabulary("fuels")
        >>> entry = fuels.get_entry_by_name("Natural Gas")
    """

    SUPPORTED_FORMATS = {".yaml", ".yml", ".json"}

    def __init__(
        self,
        vocab_path: Optional[str] = None,
        strict_mode: bool = False,
        cache_enabled: bool = True,
    ) -> None:
        """
        Initialize VocabularyManager.

        Args:
            vocab_path: Base path for vocabulary files
            strict_mode: Fail on validation errors
            cache_enabled: Enable vocabulary caching
        """
        self.vocab_path = Path(vocab_path) if vocab_path else None
        self.strict_mode = strict_mode
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Vocabulary] = {}
        self._version_history: Dict[str, List[VocabVersion]] = {}

        logger.info(
            "VocabularyManager initialized",
            vocab_path=str(vocab_path),
            strict_mode=strict_mode,
            cache_enabled=cache_enabled,
        )

    def load_vocabulary(
        self,
        vocab_id: str,
        version: Optional[str] = None,
        force_reload: bool = False,
    ) -> Vocabulary:
        """
        Load a vocabulary by ID.

        Args:
            vocab_id: Vocabulary identifier
            version: Specific version to load (latest if None)
            force_reload: Force reload from file

        Returns:
            Loaded Vocabulary

        Raises:
            VocabularyError: If vocabulary not found or invalid
        """
        cache_key = f"{vocab_id}:{version or 'latest'}"

        # Check cache
        if self.cache_enabled and not force_reload and cache_key in self._cache:
            logger.debug("Returning cached vocabulary", vocab_id=vocab_id)
            return self._cache[cache_key]

        # Load from file
        if self.vocab_path:
            vocab = self._load_from_file(vocab_id, version)
        else:
            raise VocabularyError(
                f"Vocabulary '{vocab_id}' not found - no vocab path configured",
                vocabulary_id=vocab_id,
                version=version,
            )

        # Cache if enabled
        if self.cache_enabled:
            self._cache[cache_key] = vocab

        return vocab

    def _load_from_file(
        self,
        vocab_id: str,
        version: Optional[str],
    ) -> Vocabulary:
        """Load vocabulary from file system."""
        if not self.vocab_path or not self.vocab_path.exists():
            raise VocabularyError(
                f"Vocabulary path does not exist: {self.vocab_path}",
                vocabulary_id=vocab_id,
            )

        # Try different file formats
        for ext in self.SUPPORTED_FORMATS:
            file_path = self.vocab_path / f"{vocab_id}{ext}"
            if file_path.exists():
                return self._parse_vocab_file(file_path, vocab_id, version)

        # Try subdirectory
        vocab_dir = self.vocab_path / vocab_id
        if vocab_dir.exists():
            for ext in self.SUPPORTED_FORMATS:
                file_path = vocab_dir / f"vocabulary{ext}"
                if file_path.exists():
                    return self._parse_vocab_file(file_path, vocab_id, version)

        raise VocabularyError(
            f"Vocabulary file not found for '{vocab_id}'",
            vocabulary_id=vocab_id,
            hint=f"Looked in {self.vocab_path}",
        )

    def _parse_vocab_file(
        self,
        file_path: Path,
        vocab_id: str,
        version: Optional[str],
    ) -> Vocabulary:
        """Parse a vocabulary file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix in {".yaml", ".yml"}:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            # Parse entries
            entries = []
            for entry_data in data.get("entries", []):
                entries.append(VocabEntry(**entry_data))

            # Create version info
            version_data = data.get("version", {})
            content_hash = hashlib.sha256(
                json.dumps(data.get("entries", []), sort_keys=True).encode()
            ).hexdigest()

            vocab_version = VocabVersion(
                version=version_data.get("version", "1.0.0"),
                changelog=version_data.get("changelog"),
                hash=content_hash,
            )

            # Create vocabulary
            vocab = Vocabulary(
                id=vocab_id,
                name=data.get("name", vocab_id),
                description=data.get("description"),
                version=vocab_version,
                entries=entries,
                status=VocabStatus(data.get("status", "active")),
                owner=data.get("owner"),
                tags=data.get("tags", []),
                metadata=data.get("metadata", {}),
            )

            logger.info(
                "Loaded vocabulary from file",
                vocab_id=vocab_id,
                entry_count=len(entries),
                version=vocab_version.version,
            )

            return vocab

        except Exception as e:
            raise VocabularyError(
                f"Failed to parse vocabulary file: {str(e)}",
                vocabulary_id=vocab_id,
            ) from e

    def create_vocabulary(
        self,
        vocab_id: str,
        name: str,
        entries: List[VocabEntry],
        description: Optional[str] = None,
        version: str = "1.0.0",
    ) -> Vocabulary:
        """
        Create a new vocabulary.

        Args:
            vocab_id: Vocabulary identifier
            name: Human-readable name
            entries: List of entries
            description: Vocabulary description
            version: Initial version

        Returns:
            Created Vocabulary
        """
        # Calculate content hash
        content_hash = hashlib.sha256(
            json.dumps([e.model_dump() for e in entries], sort_keys=True).encode()
        ).hexdigest()

        vocab = Vocabulary(
            id=vocab_id,
            name=name,
            description=description,
            version=VocabVersion(version=version, hash=content_hash),
            entries=entries,
        )

        # Cache it
        if self.cache_enabled:
            self._cache[f"{vocab_id}:latest"] = vocab
            self._cache[f"{vocab_id}:{version}"] = vocab

        logger.info(
            "Created vocabulary",
            vocab_id=vocab_id,
            entry_count=len(entries),
            version=version,
        )

        return vocab

    def get_vocabulary(self, vocab_id: str) -> Optional[Vocabulary]:
        """Get a vocabulary from cache (without file loading)."""
        cache_key = f"{vocab_id}:latest"
        return self._cache.get(cache_key)

    def list_vocabularies(self) -> List[str]:
        """List all cached vocabulary IDs."""
        vocab_ids = set()
        for key in self._cache.keys():
            vocab_id = key.split(":")[0]
            vocab_ids.add(vocab_id)
        return sorted(vocab_ids)

    def clear_cache(self) -> None:
        """Clear the vocabulary cache."""
        self._cache.clear()
        logger.info("Vocabulary cache cleared")

    def validate_vocabulary(self, vocab: Vocabulary) -> List[str]:
        """
        Validate a vocabulary for issues.

        Args:
            vocab: Vocabulary to validate

        Returns:
            List of validation error messages
        """
        errors = []

        # Check for duplicate IDs
        ids = [e.id for e in vocab.entries]
        if len(ids) != len(set(ids)):
            errors.append("Duplicate entry IDs found")

        # Check for empty entries
        for entry in vocab.entries:
            if not entry.name.strip():
                errors.append(f"Entry {entry.id} has empty name")

        # Check deprecated entries
        for entry in vocab.entries:
            if entry.deprecated and entry.deprecated_by:
                if not vocab.get_entry(entry.deprecated_by):
                    errors.append(
                        f"Entry {entry.id} deprecated by non-existent entry {entry.deprecated_by}"
                    )

        return errors


__all__ = [
    "VocabularyManager",
    "Vocabulary",
    "VocabEntry",
    "VocabVersion",
    "VocabStatus",
]
