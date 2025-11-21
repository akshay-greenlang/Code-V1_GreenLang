# -*- coding: utf-8 -*-
"""
GreenLang Resource Loader
Intelligent resource loading with caching and validation.
"""

from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import logging
import hashlib
from datetime import datetime, timedelta

from .readers import DataReader
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class ResourceCache:
    """Simple cache for loaded resources."""

    def __init__(self, max_size: int = 100, ttl_minutes: int = 60):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached items
            ttl_minutes: Time-to-live in minutes
        """
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get cached item if not expired."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if DeterministicClock.now() - entry["timestamp"] > self.ttl:
            # Expired
            del self._cache[key]
            return None

        return entry["data"]

    def set(self, key: str, data: Any):
        """Cache an item."""
        # Implement simple LRU by removing oldest if full
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

        self._cache[key] = {
            "data": data,
            "timestamp": DeterministicClock.now()
        }

    def clear(self):
        """Clear all cached items."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_minutes": self.ttl.total_seconds() / 60
        }


class ResourceLoader:
    """
    Intelligent resource loader with caching and validation.

    Features:
    - Multi-format support (JSON, CSV, YAML, Excel, etc.)
    - Automatic caching with TTL
    - Hash-based cache invalidation
    - Resource path searching
    - Validation on load

    Example:
        loader = ResourceLoader(search_paths=["./config", "./data"])

        # Load with caching
        config = loader.load("settings.yaml")

        # Load and validate
        schema = {...}
        data = loader.load("data.json", schema=schema)

        # Force reload (bypass cache)
        fresh_data = loader.load("data.json", use_cache=False)
    """

    def __init__(
        self,
        search_paths: Optional[List[Union[str, Path]]] = None,
        cache_enabled: bool = True,
        cache_size: int = 100,
        cache_ttl_minutes: int = 60
    ):
        """
        Initialize resource loader.

        Args:
            search_paths: List of directories to search for resources
            cache_enabled: Enable resource caching
            cache_size: Maximum cache size
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.search_paths = [Path(p) for p in (search_paths or [])]
        self.cache_enabled = cache_enabled
        self.reader = DataReader()

        if cache_enabled:
            self.cache = ResourceCache(max_size=cache_size, ttl_minutes=cache_ttl_minutes)
        else:
            self.cache = None

        self._load_count = 0
        self._cache_hits = 0

    def add_search_path(self, path: Union[str, Path]):
        """Add a directory to search paths."""
        path = Path(path)
        if path not in self.search_paths:
            self.search_paths.append(path)
            logger.debug(f"Added search path: {path}")

    def find_resource(self, resource_name: str) -> Optional[Path]:
        """
        Find resource file in search paths.

        Args:
            resource_name: Name or relative path of resource

        Returns:
            Full path if found, None otherwise
        """
        # Try as absolute path first
        path = Path(resource_name)
        if path.is_absolute() and path.exists():
            return path

        # Search in search paths
        for search_path in self.search_paths:
            full_path = search_path / resource_name
            if full_path.exists():
                return full_path

        # Try relative to current working directory
        if path.exists():
            return path

        return None

    def get_file_hash(self, path: Path) -> str:
        """
        Calculate hash of file for cache invalidation.

        Args:
            path: Path to file

        Returns:
            SHA256 hash as hex string
        """
        sha256 = hashlib.sha256()

        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    def load(
        self,
        resource_name: str,
        use_cache: bool = True,
        validate: Optional[callable] = None,
        **kwargs
    ) -> Any:
        """
        Load a resource with caching and validation.

        Args:
            resource_name: Name or path of resource
            use_cache: Use cached version if available
            validate: Optional validation function
            **kwargs: Format-specific options for reader

        Returns:
            Loaded resource data

        Raises:
            FileNotFoundError: If resource not found
            ValueError: If validation fails
        """
        self._load_count += 1

        # Find resource
        path = self.find_resource(resource_name)
        if path is None:
            raise FileNotFoundError(f"Resource not found: {resource_name} (searched: {self.search_paths})")

        # Generate cache key
        file_hash = self.get_file_hash(path)
        cache_key = f"{path}:{file_hash}"

        # Check cache
        if self.cache_enabled and use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self._cache_hits += 1
                logger.debug(f"Cache hit for {resource_name}")
                return cached_data

        # Load resource
        logger.info(f"Loading resource: {resource_name} from {path}")
        data = self.reader.read(path, **kwargs)

        # Validate if validator provided
        if validate:
            try:
                is_valid = validate(data)
                if not is_valid:
                    raise ValueError(f"Resource validation failed for {resource_name}")
            except Exception as e:
                logger.error(f"Validation error for {resource_name}: {str(e)}")
                raise

        # Cache if enabled
        if self.cache_enabled:
            self.cache.set(cache_key, data)

        return data

    def preload(self, resource_names: List[str], **kwargs):
        """
        Preload multiple resources into cache.

        Args:
            resource_names: List of resource names
            **kwargs: Options passed to load()
        """
        for name in resource_names:
            try:
                self.load(name, **kwargs)
            except Exception as e:
                logger.error(f"Failed to preload {name}: {str(e)}")

    def clear_cache(self):
        """Clear resource cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Resource cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        stats = {
            "loads": self._load_count,
            "cache_hits": self._cache_hits,
            "hit_rate": round((self._cache_hits / self._load_count * 100), 2) if self._load_count > 0 else 0,
            "search_paths": [str(p) for p in self.search_paths]
        }

        if self.cache:
            stats["cache"] = self.cache.get_stats()

        return stats
