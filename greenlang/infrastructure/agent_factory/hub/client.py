# -*- coding: utf-8 -*-
"""
Hub Client - Async REST client for the remote Agent Hub registry.

Provides authenticated HTTP access to a remote Agent Hub with retry logic,
exponential backoff, progress callbacks for uploads/downloads, and local
caching of downloaded packages.

Example:
    >>> config = HubClientConfig(base_url="https://hub.greenlang.io", api_key="sk-...")
    >>> client = HubClient(config)
    >>> async with client:
    ...     results = await client.search("emissions")
    ...     path = await client.download("emissions-calc", "1.0.0")

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 1.0
DEFAULT_CACHE_DIR = ".greenlang/hub_cache"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HubClientConfig:
    """Configuration for the Hub REST client.

    Attributes:
        base_url: Hub registry base URL.
        api_key: API key for authentication.
        jwt_token: JWT token for authentication (alternative to api_key).
        timeout_seconds: HTTP request timeout.
        max_retries: Maximum retry attempts.
        backoff_base: Base delay for exponential backoff (seconds).
        cache_dir: Local cache directory for downloaded packages.
    """

    base_url: str = "https://hub.greenlang.io"
    api_key: Optional[str] = None
    jwt_token: Optional[str] = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE
    cache_dir: str = DEFAULT_CACHE_DIR


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int], None]
"""Callback(bytes_transferred, total_bytes) for upload/download progress."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class HubClient:
    """Async REST client for the GreenLang Agent Hub.

    Handles authentication, retries, exponential backoff, and local
    caching of downloaded packages. Uses httpx for async HTTP.

    Attributes:
        config: Client configuration.
    """

    def __init__(self, config: HubClientConfig) -> None:
        """Initialize the Hub client.

        Args:
            config: Client configuration with URL and auth.
        """
        self.config = config
        self._cache_dir = Path(config.cache_dir).resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: Any = None

    async def __aenter__(self) -> HubClient:
        """Enter async context and create HTTP client."""
        try:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                headers=self._build_headers(),
            )
        except ImportError:
            logger.warning("httpx not installed; HubClient will use stub mode")
            self._client = None
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context and close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def search(
        self,
        query: str = "",
        tags: Optional[List[str]] = None,
        agent_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for packages on the remote hub.

        Args:
            query: Text search query.
            tags: Filter by tags.
            agent_type: Filter by agent type.

        Returns:
            List of package record dicts.
        """
        params: Dict[str, Any] = {}
        if query:
            params["q"] = query
        if tags:
            params["tags"] = ",".join(tags)
        if agent_type:
            params["type"] = agent_type

        data = await self._request("GET", "/api/v1/agents", params=params)
        return data if isinstance(data, list) else []

    async def publish(
        self,
        package_path: str | Path,
        progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Upload and publish a package to the hub.

        Args:
            package_path: Path to the .glpack file.
            progress: Optional progress callback.

        Returns:
            Published package record dict.
        """
        path = Path(package_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Package not found: {path}")

        file_size = path.stat().st_size
        content = path.read_bytes()

        if progress:
            progress(0, file_size)

        data = await self._request(
            "POST",
            "/api/v1/agents/publish",
            files={"package": (path.name, content, "application/gzip")},
        )

        if progress:
            progress(file_size, file_size)

        return data if isinstance(data, dict) else {}

    async def download(
        self,
        agent_key: str,
        version: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> str:
        """Download a package from the hub.

        Downloads are cached locally to avoid redundant transfers.

        Args:
            agent_key: Agent identifier.
            version: Specific version (latest if None).
            progress: Optional progress callback.

        Returns:
            Local filesystem path to the downloaded .glpack file.
        """
        # Check cache first
        cached = self._check_cache(agent_key, version or "latest")
        if cached:
            logger.info("Cache hit: %s v%s", agent_key, version or "latest")
            return str(cached)

        endpoint = f"/api/v1/agents/{agent_key}/download"
        params: Dict[str, Any] = {}
        if version:
            params["version"] = version

        data = await self._request("GET", endpoint, params=params, raw=True)

        # Save to cache
        ver_dir = self._cache_dir / agent_key / (version or "latest")
        ver_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{agent_key}-{version or 'latest'}.glpack"
        dest = ver_dir / filename

        if isinstance(data, bytes):
            dest.write_bytes(data)
            if progress:
                progress(len(data), len(data))
        else:
            dest.write_text(json.dumps(data), encoding="utf-8")

        logger.info("Downloaded %s v%s to %s", agent_key, version, dest)
        return str(dest)

    async def list_versions(self, agent_key: str) -> List[str]:
        """List all versions of an agent on the hub.

        Args:
            agent_key: Agent identifier.

        Returns:
            List of version strings.
        """
        data = await self._request("GET", f"/api/v1/agents/{agent_key}/versions")
        return data if isinstance(data, list) else []

    async def unpublish(self, agent_key: str, version: str) -> bool:
        """Unpublish a version from the hub.

        Args:
            agent_key: Agent identifier.
            version: Version to unpublish.

        Returns:
            True if successful.
        """
        data = await self._request(
            "DELETE",
            f"/api/v1/agents/{agent_key}/versions/{version}",
        )
        return data.get("success", False) if isinstance(data, dict) else False

    def clear_cache(self) -> int:
        """Clear the local download cache.

        Returns:
            Number of cached files removed.
        """
        count = 0
        if self._cache_dir.exists():
            for item in self._cache_dir.rglob("*.glpack"):
                item.unlink()
                count += 1
        logger.info("Cleared %d cached packages", count)
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        """Build authentication headers."""
        headers: Dict[str, str] = {
            "User-Agent": "GreenLang-HubClient/1.0",
            "Accept": "application/json",
        }
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        elif self.config.jwt_token:
            headers["Authorization"] = f"Bearer {self.config.jwt_token}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        raw: bool = False,
    ) -> Any:
        """Make an HTTP request with retry and exponential backoff.

        Args:
            method: HTTP method.
            path: URL path.
            params: Query parameters.
            json_data: JSON body.
            files: Multipart file upload.
            raw: If True, return raw bytes.

        Returns:
            Parsed JSON response or raw bytes.
        """
        if self._client is None:
            logger.warning("HTTP client not available, returning empty response")
            return b"" if raw else {}

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._client.request(
                    method,
                    path,
                    params=params,
                    json=json_data,
                    files=files,
                )
                response.raise_for_status()
                if raw:
                    return response.content
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    delay = self.config.backoff_base * (2 ** attempt)
                    logger.warning(
                        "Request %s %s failed (attempt %d/%d): %s. Retrying in %.1fs",
                        method,
                        path,
                        attempt + 1,
                        self.config.max_retries + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

        logger.error("Request %s %s failed after %d attempts", method, path, self.config.max_retries + 1)
        raise ConnectionError(
            f"Hub request failed: {method} {path}: {last_error}"
        ) from last_error

    def _check_cache(
        self, agent_key: str, version: str
    ) -> Optional[Path]:
        """Check if a package is in the local cache.

        Returns:
            Cached file path if found, None otherwise.
        """
        filename = f"{agent_key}-{version}.glpack"
        cached_path = self._cache_dir / agent_key / version / filename
        if cached_path.exists():
            return cached_path
        return None
