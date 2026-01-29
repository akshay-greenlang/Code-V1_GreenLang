"""
Vocabulary provider for the GreenLang Normalizer SDK.

This module provides vocabulary loading and management capabilities
for local and remote vocabulary sources.

Example:
    >>> from gl_normalizer_sdk import LocalVocabProvider
    >>> provider = LocalVocabProvider("/path/to/vocab")
    >>> fuels = provider.get_vocabulary("fuels")
    >>> print(fuels.version)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

import httpx
from pydantic import BaseModel, Field
import yaml


class VocabEntry(BaseModel):
    """A vocabulary entry."""

    id: str
    name: str
    aliases: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Vocabulary(BaseModel):
    """A vocabulary with entries."""

    id: str
    name: str
    version: str
    entries: List[VocabEntry] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_entry(self, entry_id: str) -> Optional[VocabEntry]:
        """Get entry by ID."""
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def search(self, query: str) -> List[VocabEntry]:
        """Search entries by name or alias."""
        query_lower = query.lower()
        results = []
        for entry in self.entries:
            if query_lower in entry.name.lower():
                results.append(entry)
                continue
            for alias in entry.aliases:
                if query_lower in alias.lower():
                    results.append(entry)
                    break
        return results


class VocabProvider(ABC):
    """
    Abstract base class for vocabulary providers.

    Vocabulary providers are responsible for loading and caching
    vocabulary data from various sources.
    """

    @abstractmethod
    def get_vocabulary(self, vocab_id: str) -> Vocabulary:
        """
        Get a vocabulary by ID.

        Args:
            vocab_id: Vocabulary identifier

        Returns:
            Vocabulary object

        Raises:
            ValueError: If vocabulary not found
        """
        pass

    @abstractmethod
    def list_vocabularies(self) -> List[str]:
        """
        List available vocabulary IDs.

        Returns:
            List of vocabulary IDs
        """
        pass

    @abstractmethod
    def refresh(self) -> None:
        """Refresh vocabulary cache."""
        pass


class LocalVocabProvider(VocabProvider):
    """
    Vocabulary provider for local file-based vocabularies.

    This provider loads vocabularies from YAML or JSON files
    in a specified directory.

    Example:
        >>> provider = LocalVocabProvider("/path/to/vocab")
        >>> fuels = provider.get_vocabulary("fuels")
    """

    SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json"}

    def __init__(self, vocab_path: str) -> None:
        """
        Initialize local vocabulary provider.

        Args:
            vocab_path: Path to vocabulary directory
        """
        self.vocab_path = Path(vocab_path)
        self._cache: Dict[str, Vocabulary] = {}
        self._scan_vocabularies()

    def _scan_vocabularies(self) -> None:
        """Scan vocabulary directory for files."""
        if not self.vocab_path.exists():
            return

        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in self.vocab_path.glob(f"*{ext}"):
                vocab_id = file_path.stem
                self._cache[vocab_id] = self._load_vocabulary(file_path, vocab_id)

    def _load_vocabulary(self, file_path: Path, vocab_id: str) -> Vocabulary:
        """Load vocabulary from file."""
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        entries = [VocabEntry(**e) for e in data.get("entries", [])]

        return Vocabulary(
            id=vocab_id,
            name=data.get("name", vocab_id),
            version=data.get("version", "1.0.0"),
            entries=entries,
            metadata=data.get("metadata", {}),
        )

    def get_vocabulary(self, vocab_id: str) -> Vocabulary:
        """Get vocabulary by ID."""
        if vocab_id not in self._cache:
            raise ValueError(f"Vocabulary '{vocab_id}' not found")
        return self._cache[vocab_id]

    def list_vocabularies(self) -> List[str]:
        """List available vocabularies."""
        return list(self._cache.keys())

    def refresh(self) -> None:
        """Refresh vocabulary cache."""
        self._cache.clear()
        self._scan_vocabularies()


class RemoteVocabProvider(VocabProvider):
    """
    Vocabulary provider for remote API-based vocabularies.

    This provider fetches vocabularies from a GreenLang Normalizer
    Service instance.

    Example:
        >>> provider = RemoteVocabProvider("http://localhost:8000")
        >>> fuels = provider.get_vocabulary("fuels")
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize remote vocabulary provider.

        Args:
            base_url: Base URL of normalizer service
            api_key: Optional API key
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self._cache: Dict[str, Vocabulary] = {}

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_vocabulary(self, vocab_id: str) -> Vocabulary:
        """Get vocabulary from remote service."""
        if vocab_id in self._cache:
            return self._cache[vocab_id]

        with httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._get_headers(),
        ) as client:
            response = client.get(f"/api/v1/vocabularies/{vocab_id}")
            response.raise_for_status()
            data = response.json()

        entries = [VocabEntry(**e) for e in data.get("entries", [])]
        vocab = Vocabulary(
            id=vocab_id,
            name=data.get("name", vocab_id),
            version=data.get("version", "1.0.0"),
            entries=entries,
            metadata=data.get("metadata", {}),
        )

        self._cache[vocab_id] = vocab
        return vocab

    def list_vocabularies(self) -> List[str]:
        """List available vocabularies from remote."""
        with httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._get_headers(),
        ) as client:
            response = client.get("/api/v1/vocabularies")
            response.raise_for_status()
            data = response.json()

        return [v["id"] for v in data.get("vocabularies", [])]

    def refresh(self) -> None:
        """Clear vocabulary cache."""
        self._cache.clear()


__all__ = [
    "VocabProvider",
    "LocalVocabProvider",
    "RemoteVocabProvider",
    "Vocabulary",
    "VocabEntry",
]
