"""
Vocabulary loading implementations for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides various vocabulary loader implementations:
- YAMLVocabularyLoader: Load vocabularies from local YAML files
- RemoteVocabularyLoader: Fetch vocabularies from gl-normalizer-service API
- CachedLoader: Wrapper providing TTL-based caching for any loader

Key Design Principles:
    - Lazy loading: Vocabularies are loaded on first access
    - Caching: TTL-based caching reduces repeated loads
    - Git integration: Support for Git-versioned vocabulary files
    - Extensibility: Abstract base class for custom loaders

Example:
    >>> from gl_normalizer_core.vocabulary.loader import YAMLVocabularyLoader, CachedLoader
    >>> base_loader = YAMLVocabularyLoader(vocab_dir="/path/to/vocab")
    >>> loader = CachedLoader(base_loader, ttl_seconds=3600)
    >>> vocab = loader.load("fuels")
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import hashlib
import json
import logging
import subprocess
import threading
import time

from pydantic import BaseModel, Field

from gl_normalizer_core.errors.codes import GLNORMErrorCode
from gl_normalizer_core.errors import VocabularyError
from gl_normalizer_core.vocabulary.models import (
    Vocabulary,
    Entity,
    Alias,
    VocabularyMetadata,
    EntityType,
    DeprecationInfo,
)

logger = logging.getLogger(__name__)


class LoaderConfig(BaseModel):
    """
    Configuration for vocabulary loaders.

    Attributes:
        timeout_seconds: Timeout for load operations.
        retry_count: Number of retries on failure.
        retry_delay_seconds: Delay between retries.
        validate_on_load: Whether to validate vocabulary after loading.
        verify_signature: Whether to verify vocabulary signatures.
    """

    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout for load operations in seconds",
    )
    retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retries on failure",
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Delay between retries in seconds",
    )
    validate_on_load: bool = Field(
        default=True,
        description="Whether to validate vocabulary after loading",
    )
    verify_signature: bool = Field(
        default=True,
        description="Whether to verify vocabulary signatures",
    )


class VocabularyLoader(ABC):
    """
    Abstract base class for vocabulary loaders.

    Subclasses must implement the load() method to fetch vocabulary
    data from their respective sources.

    Attributes:
        config: Loader configuration.

    Example:
        >>> class CustomLoader(VocabularyLoader):
        ...     def load(self, vocab_id: str) -> Vocabulary:
        ...         # Custom loading logic
        ...         pass
    """

    def __init__(self, config: Optional[LoaderConfig] = None) -> None:
        """
        Initialize the vocabulary loader.

        Args:
            config: Loader configuration. Uses defaults if not provided.
        """
        self.config = config or LoaderConfig()

    @abstractmethod
    def load(self, vocab_id: str, version: Optional[str] = None) -> Vocabulary:
        """
        Load a vocabulary by ID.

        Args:
            vocab_id: The vocabulary identifier (e.g., "fuels", "materials").
            version: Optional specific version to load.

        Returns:
            The loaded Vocabulary object.

        Raises:
            VocabularyError: If loading fails.
        """
        pass

    @abstractmethod
    def list_vocabularies(self) -> List[str]:
        """
        List available vocabulary IDs.

        Returns:
            List of vocabulary identifiers.
        """
        pass

    @abstractmethod
    def get_latest_version(self, vocab_id: str) -> str:
        """
        Get the latest version of a vocabulary.

        Args:
            vocab_id: The vocabulary identifier.

        Returns:
            Version string of the latest vocabulary.

        Raises:
            VocabularyError: If vocabulary not found.
        """
        pass


class YAMLVocabularyLoader(VocabularyLoader):
    """
    Load vocabularies from local YAML files.

    Supports Git-versioned vocabulary directories with automatic
    version detection from Git tags.

    Attributes:
        vocab_dir: Path to the vocabulary directory.
        git_enabled: Whether to use Git for version information.

    Example:
        >>> loader = YAMLVocabularyLoader(vocab_dir="/path/to/vocab")
        >>> vocab = loader.load("fuels")
        >>> print(vocab.version)
        '2026.01.0'
    """

    def __init__(
        self,
        vocab_dir: str,
        git_enabled: bool = True,
        config: Optional[LoaderConfig] = None,
    ) -> None:
        """
        Initialize the YAML vocabulary loader.

        Args:
            vocab_dir: Path to the vocabulary directory.
            git_enabled: Whether to use Git for version information.
            config: Loader configuration.
        """
        super().__init__(config)
        self.vocab_dir = Path(vocab_dir)
        self.git_enabled = git_enabled

        if not self.vocab_dir.exists():
            logger.warning(
                "Vocabulary directory does not exist",
                vocab_dir=str(self.vocab_dir),
            )

    def load(self, vocab_id: str, version: Optional[str] = None) -> Vocabulary:
        """
        Load a vocabulary from YAML file.

        Args:
            vocab_id: The vocabulary identifier.
            version: Optional specific version (uses Git tag if git_enabled).

        Returns:
            The loaded Vocabulary object.

        Raises:
            VocabularyError: If loading fails.
        """
        try:
            import yaml
        except ImportError:
            raise VocabularyError(
                "PyYAML is required for YAML vocabulary loading. "
                "Install with: pip install pyyaml",
                vocabulary_id=vocab_id,
                code=GLNORMErrorCode.E501_VOCABULARY_LOAD_FAILED.value,
            )

        vocab_file = self._find_vocab_file(vocab_id)
        if not vocab_file:
            raise VocabularyError(
                f"Vocabulary '{vocab_id}' not found in {self.vocab_dir}",
                vocabulary_id=vocab_id,
                code=GLNORMErrorCode.E404_VOCABULARY_NOT_FOUND.value,
            )

        logger.info(
            "Loading vocabulary from YAML",
            vocab_id=vocab_id,
            path=str(vocab_file),
        )

        try:
            with open(vocab_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise VocabularyError(
                f"Failed to parse YAML file: {str(e)}",
                vocabulary_id=vocab_id,
                code=GLNORMErrorCode.E501_VOCABULARY_LOAD_FAILED.value,
            )

        # Get version from Git or file
        vocab_version = version or self._get_version(vocab_file)

        # Parse vocabulary data
        vocabulary = self._parse_vocabulary_data(vocab_id, vocab_version, data)
        vocabulary.source_path = str(vocab_file)

        # Add Git metadata if available
        if self.git_enabled:
            git_info = self._get_git_info(vocab_file)
            if vocabulary.metadata and git_info:
                vocabulary.metadata.git_commit = git_info.get("commit")
                vocabulary.metadata.git_tag = git_info.get("tag")

        # Validate if configured
        if self.config.validate_on_load:
            self._validate_vocabulary(vocabulary)

        # Verify signature if configured
        if self.config.verify_signature and vocabulary.metadata:
            if vocabulary.metadata.signature and not vocabulary.verify_signature():
                raise VocabularyError(
                    "Vocabulary signature verification failed",
                    vocabulary_id=vocab_id,
                    version=vocab_version,
                    code=GLNORMErrorCode.E502_VOCABULARY_CORRUPTED.value,
                )

        logger.info(
            "Loaded vocabulary",
            vocab_id=vocab_id,
            version=vocabulary.version,
            entity_count=vocabulary.entity_count(),
            alias_count=vocabulary.alias_count(),
        )

        return vocabulary

    def list_vocabularies(self) -> List[str]:
        """
        List available vocabulary IDs by scanning the directory.

        Returns:
            List of vocabulary identifiers found in the directory.
        """
        if not self.vocab_dir.exists():
            return []

        vocab_ids: Set[str] = set()

        # Look for .yaml and .yml files
        for pattern in ["*.yaml", "*.yml"]:
            for path in self.vocab_dir.glob(pattern):
                # Skip hidden files and test files
                if path.name.startswith(".") or path.name.startswith("test_"):
                    continue
                vocab_id = path.stem
                vocab_ids.add(vocab_id)

        # Also check subdirectories (vocab_id/vocab.yaml structure)
        for subdir in self.vocab_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                vocab_file = subdir / "vocab.yaml"
                if vocab_file.exists():
                    vocab_ids.add(subdir.name)
                vocab_file = subdir / f"{subdir.name}.yaml"
                if vocab_file.exists():
                    vocab_ids.add(subdir.name)

        return sorted(vocab_ids)

    def get_latest_version(self, vocab_id: str) -> str:
        """
        Get the latest version of a vocabulary.

        Uses Git tags if git_enabled, otherwise extracts from file.

        Args:
            vocab_id: The vocabulary identifier.

        Returns:
            Version string.
        """
        vocab_file = self._find_vocab_file(vocab_id)
        if not vocab_file:
            raise VocabularyError(
                f"Vocabulary '{vocab_id}' not found",
                vocabulary_id=vocab_id,
                code=GLNORMErrorCode.E404_VOCABULARY_NOT_FOUND.value,
            )

        return self._get_version(vocab_file)

    def _find_vocab_file(self, vocab_id: str) -> Optional[Path]:
        """Find the vocabulary file for a given ID."""
        # Check direct file: vocab_id.yaml or vocab_id.yml
        for ext in [".yaml", ".yml"]:
            path = self.vocab_dir / f"{vocab_id}{ext}"
            if path.exists():
                return path

        # Check subdirectory: vocab_id/vocab.yaml or vocab_id/vocab_id.yaml
        subdir = self.vocab_dir / vocab_id
        if subdir.is_dir():
            for filename in ["vocab.yaml", "vocab.yml", f"{vocab_id}.yaml", f"{vocab_id}.yml"]:
                path = subdir / filename
                if path.exists():
                    return path

        return None

    def _get_version(self, vocab_file: Path) -> str:
        """Get version from Git tag or file metadata."""
        if self.git_enabled:
            git_info = self._get_git_info(vocab_file)
            if git_info and git_info.get("tag"):
                return git_info["tag"]

        # Try to extract version from file content
        try:
            import yaml
            with open(vocab_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict):
                if "version" in data:
                    return str(data["version"])
                if "metadata" in data and isinstance(data["metadata"], dict):
                    if "version" in data["metadata"]:
                        return str(data["metadata"]["version"])
        except Exception:
            pass

        # Default version based on file modification time
        mtime = vocab_file.stat().st_mtime
        dt = datetime.fromtimestamp(mtime)
        return dt.strftime("%Y.%m.%d")

    def _get_git_info(self, vocab_file: Path) -> Optional[Dict[str, str]]:
        """Get Git information for a file."""
        try:
            # Get the latest commit hash for this file
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H", "--", str(vocab_file)],
                cwd=str(vocab_file.parent),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None

            commit = result.stdout.strip()[:40]

            # Try to get the tag for this commit
            tag_result = subprocess.run(
                ["git", "describe", "--tags", "--exact-match", commit],
                cwd=str(vocab_file.parent),
                capture_output=True,
                text=True,
                timeout=5,
            )
            tag = tag_result.stdout.strip() if tag_result.returncode == 0 else None

            return {"commit": commit, "tag": tag}

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"Git info retrieval failed: {e}")
            return None

    def _parse_vocabulary_data(
        self,
        vocab_id: str,
        version: str,
        data: Dict[str, Any],
    ) -> Vocabulary:
        """Parse vocabulary data from YAML structure."""
        entities: Dict[str, Entity] = {}
        aliases: List[Alias] = []

        # Parse entities
        entities_data = data.get("entities", data.get("items", []))
        if isinstance(entities_data, list):
            for entity_data in entities_data:
                entity = self._parse_entity(entity_data)
                entities[entity.id] = entity
        elif isinstance(entities_data, dict):
            for entity_id, entity_data in entities_data.items():
                if isinstance(entity_data, dict):
                    entity_data["id"] = entity_data.get("id", entity_id)
                    entity = self._parse_entity(entity_data)
                    entities[entity.id] = entity

        # Parse explicit aliases
        aliases_data = data.get("aliases", [])
        for alias_data in aliases_data:
            if isinstance(alias_data, dict):
                aliases.append(Alias(**alias_data))

        # Parse metadata
        metadata = None
        meta_data = data.get("metadata", {})
        if meta_data:
            metadata = VocabularyMetadata(
                version=meta_data.get("version", version),
                signature=meta_data.get("signature"),
                created_at=meta_data.get("created_at", datetime.utcnow()),
                created_by=meta_data.get("created_by"),
                expires_at=meta_data.get("expires_at"),
                description=meta_data.get("description"),
                changelog=meta_data.get("changelog", []),
            )
        else:
            metadata = VocabularyMetadata(version=version)

        return Vocabulary(
            id=vocab_id,
            version=version,
            entities=entities,
            aliases=aliases,
            metadata=metadata,
            parent_id=data.get("parent_id"),
        )

    def _parse_entity(self, data: Dict[str, Any]) -> Entity:
        """Parse a single entity from data."""
        # Parse entity type
        entity_type = None
        type_str = data.get("type", data.get("entity_type"))
        if type_str:
            try:
                entity_type = EntityType(type_str.lower())
            except ValueError:
                entity_type = EntityType.CUSTOM

        # Parse deprecation info
        deprecation_info = None
        if data.get("deprecated"):
            dep_data = data.get("deprecation_info", data.get("deprecation", {}))
            if dep_data:
                deprecation_info = DeprecationInfo(
                    deprecated_at=dep_data.get("deprecated_at", datetime.utcnow()),
                    reason=dep_data.get("reason", "Deprecated"),
                    replacement_id=dep_data.get("replacement_id"),
                    removal_date=dep_data.get("removal_date"),
                )

        # Parse dates
        effective_date = data.get("effective_date")
        if effective_date and isinstance(effective_date, str):
            effective_date = datetime.fromisoformat(effective_date.replace("Z", "+00:00"))

        expiration_date = data.get("expiration_date")
        if expiration_date and isinstance(expiration_date, str):
            expiration_date = datetime.fromisoformat(expiration_date.replace("Z", "+00:00"))

        return Entity(
            id=data["id"],
            canonical_name=data.get("canonical_name", data.get("name", data["id"])),
            entity_type=entity_type,
            aliases=data.get("aliases", []),
            properties=data.get("properties", {}),
            deprecated=data.get("deprecated", False),
            deprecation_info=deprecation_info,
            effective_date=effective_date,
            expiration_date=expiration_date,
            metadata=data.get("metadata", {}),
        )

    def _validate_vocabulary(self, vocabulary: Vocabulary) -> None:
        """Validate a loaded vocabulary."""
        errors: List[str] = []

        # Check for duplicate entity IDs (shouldn't happen with dict, but check aliases)
        entity_ids = set(vocabulary.entities.keys())

        # Check all alias references point to valid entities
        for alias in vocabulary.aliases:
            if alias.canonical_id not in entity_ids:
                errors.append(
                    f"Alias '{alias.alias}' references non-existent entity "
                    f"'{alias.canonical_id}'"
                )

        # Check deprecated entities have valid replacement references
        for entity in vocabulary.entities.values():
            if entity.deprecated and entity.deprecation_info:
                replacement_id = entity.deprecation_info.replacement_id
                if replacement_id and replacement_id not in entity_ids:
                    errors.append(
                        f"Deprecated entity '{entity.id}' references non-existent "
                        f"replacement '{replacement_id}'"
                    )

        if errors:
            raise VocabularyError(
                f"Vocabulary validation failed: {'; '.join(errors)}",
                vocabulary_id=vocabulary.id,
                version=vocabulary.version,
                code=GLNORMErrorCode.E502_VOCABULARY_CORRUPTED.value,
            )


class RemoteVocabularyLoader(VocabularyLoader):
    """
    Load vocabularies from gl-normalizer-service API.

    Fetches vocabularies from a remote HTTP API endpoint with
    support for authentication and caching headers.

    Attributes:
        base_url: Base URL of the normalizer service API.
        api_key: Optional API key for authentication.
        headers: Additional HTTP headers to send with requests.

    Example:
        >>> loader = RemoteVocabularyLoader(
        ...     base_url="https://api.greenlang.io/normalizer",
        ...     api_key="your-api-key"
        ... )
        >>> vocab = loader.load("fuels")
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        config: Optional[LoaderConfig] = None,
    ) -> None:
        """
        Initialize the remote vocabulary loader.

        Args:
            base_url: Base URL of the normalizer service API.
            api_key: Optional API key for authentication.
            headers: Additional HTTP headers.
            config: Loader configuration.
        """
        super().__init__(config)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = headers or {}

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def load(self, vocab_id: str, version: Optional[str] = None) -> Vocabulary:
        """
        Load a vocabulary from the remote API.

        Args:
            vocab_id: The vocabulary identifier.
            version: Optional specific version to load.

        Returns:
            The loaded Vocabulary object.

        Raises:
            VocabularyError: If loading fails.
        """
        try:
            import httpx
        except ImportError:
            raise VocabularyError(
                "httpx is required for remote vocabulary loading. "
                "Install with: pip install httpx",
                vocabulary_id=vocab_id,
                code=GLNORMErrorCode.E501_VOCABULARY_LOAD_FAILED.value,
            )

        url = f"{self.base_url}/vocabularies/{vocab_id}"
        if version:
            url = f"{url}/versions/{version}"

        logger.info(
            "Loading vocabulary from remote API",
            vocab_id=vocab_id,
            url=url,
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.config.retry_count + 1):
            try:
                with httpx.Client(timeout=self.config.timeout_seconds) as client:
                    response = client.get(url, headers=self.headers)

                    if response.status_code == 404:
                        raise VocabularyError(
                            f"Vocabulary '{vocab_id}' not found at {url}",
                            vocabulary_id=vocab_id,
                            code=GLNORMErrorCode.E404_VOCABULARY_NOT_FOUND.value,
                        )

                    response.raise_for_status()
                    data = response.json()

                    # Parse response into Vocabulary
                    vocabulary = self._parse_response(vocab_id, data)

                    logger.info(
                        "Loaded vocabulary from remote",
                        vocab_id=vocab_id,
                        version=vocabulary.version,
                        entity_count=vocabulary.entity_count(),
                    )

                    return vocabulary

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code >= 500:
                    # Retry on server errors
                    if attempt < self.config.retry_count:
                        logger.warning(
                            f"Remote load failed (attempt {attempt + 1}), retrying...",
                            error=str(e),
                        )
                        time.sleep(self.config.retry_delay_seconds)
                        continue
                raise VocabularyError(
                    f"HTTP error loading vocabulary: {str(e)}",
                    vocabulary_id=vocab_id,
                    code=GLNORMErrorCode.E501_VOCABULARY_LOAD_FAILED.value,
                )
            except httpx.RequestError as e:
                last_error = e
                if attempt < self.config.retry_count:
                    logger.warning(
                        f"Remote load failed (attempt {attempt + 1}), retrying...",
                        error=str(e),
                    )
                    time.sleep(self.config.retry_delay_seconds)
                    continue

        raise VocabularyError(
            f"Failed to load vocabulary after {self.config.retry_count + 1} attempts: "
            f"{str(last_error)}",
            vocabulary_id=vocab_id,
            code=GLNORMErrorCode.E501_VOCABULARY_LOAD_FAILED.value,
        )

    def list_vocabularies(self) -> List[str]:
        """
        List available vocabularies from the remote API.

        Returns:
            List of vocabulary identifiers.
        """
        try:
            import httpx
        except ImportError:
            return []

        url = f"{self.base_url}/vocabularies"

        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                response = client.get(url, headers=self.headers)
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "vocabularies" in data:
                    return data["vocabularies"]
                return []
        except Exception as e:
            logger.warning(f"Failed to list vocabularies from remote: {e}")
            return []

    def get_latest_version(self, vocab_id: str) -> str:
        """
        Get the latest version of a vocabulary from the API.

        Args:
            vocab_id: The vocabulary identifier.

        Returns:
            Version string.
        """
        try:
            import httpx
        except ImportError:
            raise VocabularyError(
                "httpx is required",
                vocabulary_id=vocab_id,
                code=GLNORMErrorCode.E501_VOCABULARY_LOAD_FAILED.value,
            )

        url = f"{self.base_url}/vocabularies/{vocab_id}/versions/latest"

        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                response = client.get(url, headers=self.headers)
                response.raise_for_status()
                data = response.json()

                if isinstance(data, str):
                    return data
                if isinstance(data, dict):
                    return data.get("version", data.get("latest_version", "1.0.0"))
                return "1.0.0"
        except Exception as e:
            logger.warning(f"Failed to get latest version: {e}")
            return "1.0.0"

    def _parse_response(self, vocab_id: str, data: Dict[str, Any]) -> Vocabulary:
        """Parse API response into Vocabulary object."""
        entities: Dict[str, Entity] = {}
        aliases: List[Alias] = []

        # Parse entities from response
        entities_data = data.get("entities", data.get("items", {}))
        if isinstance(entities_data, list):
            for entity_data in entities_data:
                entity = Entity(**entity_data)
                entities[entity.id] = entity
        elif isinstance(entities_data, dict):
            for entity_id, entity_data in entities_data.items():
                if isinstance(entity_data, dict):
                    entity_data["id"] = entity_data.get("id", entity_id)
                    entity = Entity(**entity_data)
                    entities[entity.id] = entity

        # Parse aliases
        for alias_data in data.get("aliases", []):
            aliases.append(Alias(**alias_data))

        # Parse metadata
        metadata = None
        if "metadata" in data:
            metadata = VocabularyMetadata(**data["metadata"])

        return Vocabulary(
            id=data.get("id", vocab_id),
            version=data.get("version", "1.0.0"),
            entities=entities,
            aliases=aliases,
            metadata=metadata,
        )


class CacheEntry(BaseModel):
    """Cache entry for vocabulary caching."""

    vocabulary: Vocabulary
    loaded_at: datetime
    expires_at: datetime
    hit_count: int = 0


class CachedLoader(VocabularyLoader):
    """
    Caching wrapper for vocabulary loaders.

    Provides TTL-based caching to reduce repeated loads and improve
    performance. Supports both in-memory caching and optional
    persistent caching.

    Attributes:
        base_loader: The underlying loader to wrap.
        ttl_seconds: Time-to-live for cache entries.
        max_entries: Maximum number of cached vocabularies.

    Example:
        >>> base_loader = YAMLVocabularyLoader(vocab_dir="/path/to/vocab")
        >>> loader = CachedLoader(base_loader, ttl_seconds=3600)
        >>> vocab = loader.load("fuels")  # Loads from disk
        >>> vocab = loader.load("fuels")  # Returns cached version
    """

    def __init__(
        self,
        base_loader: VocabularyLoader,
        ttl_seconds: int = 3600,
        max_entries: int = 100,
        config: Optional[LoaderConfig] = None,
    ) -> None:
        """
        Initialize the cached loader.

        Args:
            base_loader: The underlying loader to wrap.
            ttl_seconds: Time-to-live for cache entries in seconds.
            max_entries: Maximum number of cached vocabularies.
            config: Loader configuration.
        """
        super().__init__(config)
        self.base_loader = base_loader
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def load(self, vocab_id: str, version: Optional[str] = None) -> Vocabulary:
        """
        Load a vocabulary with caching.

        Args:
            vocab_id: The vocabulary identifier.
            version: Optional specific version to load.

        Returns:
            The loaded Vocabulary object (cached if available).
        """
        cache_key = f"{vocab_id}:{version or 'latest'}"

        with self._lock:
            # Check cache
            entry = self._cache.get(cache_key)
            if entry and datetime.utcnow() < entry.expires_at:
                entry.hit_count += 1
                self._stats["hits"] += 1
                logger.debug(
                    "Cache hit",
                    vocab_id=vocab_id,
                    version=entry.vocabulary.version,
                    hit_count=entry.hit_count,
                )
                return entry.vocabulary

            # Cache miss - load from base loader
            self._stats["misses"] += 1
            logger.debug("Cache miss", vocab_id=vocab_id, cache_key=cache_key)

        # Load outside the lock to avoid blocking other threads
        vocabulary = self.base_loader.load(vocab_id, version)

        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_entries:
                self._evict_oldest()

            # Store in cache
            self._cache[cache_key] = CacheEntry(
                vocabulary=vocabulary,
                loaded_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=self.ttl_seconds),
            )

        return vocabulary

    def list_vocabularies(self) -> List[str]:
        """List available vocabularies (delegates to base loader)."""
        return self.base_loader.list_vocabularies()

    def get_latest_version(self, vocab_id: str) -> str:
        """Get latest version (delegates to base loader)."""
        return self.base_loader.get_latest_version(vocab_id)

    def invalidate(self, vocab_id: Optional[str] = None) -> int:
        """
        Invalidate cache entries.

        Args:
            vocab_id: Optional vocabulary ID to invalidate.
                     If None, invalidates all entries.

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            if vocab_id is None:
                count = len(self._cache)
                self._cache.clear()
                logger.info("Invalidated all cache entries", count=count)
                return count

            # Invalidate entries for specific vocab_id
            keys_to_remove = [
                key for key in self._cache.keys()
                if key.startswith(f"{vocab_id}:")
            ]
            for key in keys_to_remove:
                del self._cache[key]

            logger.info(
                "Invalidated cache entries",
                vocab_id=vocab_id,
                count=len(keys_to_remove),
            )
            return len(keys_to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with hit/miss counts and cache size.
        """
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0

            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "max_size": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
            }

    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry."""
        if not self._cache:
            return

        # Find oldest entry
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].loaded_at,
        )
        del self._cache[oldest_key]
        self._stats["evictions"] += 1
        logger.debug("Evicted cache entry", key=oldest_key)


__all__ = [
    "LoaderConfig",
    "VocabularyLoader",
    "YAMLVocabularyLoader",
    "RemoteVocabularyLoader",
    "CachedLoader",
    "CacheEntry",
]
