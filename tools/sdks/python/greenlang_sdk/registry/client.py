"""
GreenLang Registry Client
HTTP client for interacting with the Agent Registry API
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import httpx
from httpx import Timeout


class RegistryError(Exception):
    """Base exception for registry operations"""
    pass


class AgentNotFoundError(RegistryError):
    """Agent not found in registry"""
    pass


class AgentAlreadyExistsError(RegistryError):
    """Agent already exists in registry"""
    pass


class VersionAlreadyExistsError(RegistryError):
    """Version already exists for agent"""
    pass


class RegistryClient:
    """
    Client for interacting with the GreenLang Agent Registry API

    Example:
        client = RegistryClient(base_url="http://localhost:8000")

        # Register new agent
        agent = client.register(
            name="thermosync",
            namespace="greenlang",
            description="Temperature monitoring agent",
            spec_hash="abc123..."
        )

        # Publish version
        version = client.publish_version(
            agent_id=agent["id"],
            version="1.0.0",
            pack_path="/path/to/pack.glpack"
        )

        # List agents
        agents = client.list_agents(namespace="greenlang")

        # Submit certification
        cert = client.certify(
            agent_id=agent["id"],
            version="1.0.0",
            dimension="security",
            status="passed",
            score=95.0
        )
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        retries: int = 3
    ):
        """
        Initialize Registry Client

        Args:
            base_url: Base URL of the registry API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            retries: Number of retry attempts for failed requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = Timeout(timeout)
        self.retries = retries

        # Configure HTTP client with retries
        transport = httpx.HTTPTransport(retries=retries)
        self.client = httpx.Client(transport=transport, timeout=self.timeout)

        # Set headers
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "GreenLang-SDK/1.0.0"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def close(self):
        """Close HTTP client"""
        self.client.close()

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request with error handling

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional request parameters

        Returns:
            Response JSON data

        Raises:
            RegistryError: On API errors
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.client.request(
                method=method,
                url=url,
                headers=self.headers,
                **kwargs
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code

            if status_code == 404:
                raise AgentNotFoundError(f"Resource not found: {endpoint}")
            elif status_code == 409:
                raise AgentAlreadyExistsError(f"Resource already exists: {endpoint}")
            else:
                error_detail = e.response.json().get("detail", str(e))
                raise RegistryError(f"API error ({status_code}): {error_detail}")

        except httpx.RequestError as e:
            raise RegistryError(f"Request failed: {str(e)}")

    # ========================================================================
    # Agent Operations
    # ========================================================================

    def register(
        self,
        name: str,
        namespace: str = "default",
        description: Optional[str] = None,
        author: Optional[str] = None,
        repository_url: Optional[str] = None,
        homepage_url: Optional[str] = None,
        spec_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register a new agent in the registry

        Args:
            name: Agent name (alphanumeric with hyphens/underscores)
            namespace: Agent namespace (default: "default")
            description: Agent description
            author: Author name
            repository_url: Source code repository URL
            homepage_url: Agent homepage URL
            spec_hash: SHA-256 hash of agent specification (auto-generated if not provided)

        Returns:
            Agent metadata

        Raises:
            AgentAlreadyExistsError: If agent already exists
            RegistryError: On other errors
        """
        # Generate spec hash if not provided
        if not spec_hash:
            spec_content = json.dumps({
                "name": name,
                "namespace": namespace,
                "description": description
            }, sort_keys=True)
            spec_hash = hashlib.sha256(spec_content.encode()).hexdigest()

        payload = {
            "name": name,
            "namespace": namespace,
            "description": description,
            "author": author,
            "repository_url": repository_url,
            "homepage_url": homepage_url,
            "spec_hash": spec_hash
        }

        return self._request("POST", "/api/v1/agents", json=payload)

    def get(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent details by ID

        Args:
            agent_id: Agent UUID

        Returns:
            Agent metadata

        Raises:
            AgentNotFoundError: If agent not found
        """
        return self._request("GET", f"/api/v1/agents/{agent_id}")

    def list_agents(
        self,
        namespace: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        List agents with filtering and pagination

        Args:
            namespace: Filter by namespace
            status: Filter by status (active, deprecated, archived)
            search: Search in name and description
            page: Page number (1-indexed)
            page_size: Number of results per page

        Returns:
            {
                "agents": [...],
                "total": int,
                "page": int,
                "page_size": int
            }
        """
        params = {"page": page, "page_size": page_size}
        if namespace:
            params["namespace"] = namespace
        if status:
            params["status"] = status
        if search:
            params["search"] = search

        return self._request("GET", "/api/v1/agents", params=params)

    # ========================================================================
    # Version Operations
    # ========================================================================

    def publish_version(
        self,
        agent_id: str,
        version: str,
        pack_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        capabilities: Optional[List[str]] = None,
        dependencies: Optional[List[Dict[str, str]]] = None,
        published_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Publish a new version of an agent

        Args:
            agent_id: Agent UUID
            version: Semantic version (e.g., "1.0.0")
            pack_path: Path to .glpack file (local or remote)
            metadata: Additional metadata
            capabilities: List of agent capabilities
            dependencies: List of dependencies
            published_by: Publisher identifier

        Returns:
            Version metadata

        Raises:
            VersionAlreadyExistsError: If version already exists
            AgentNotFoundError: If agent not found
        """
        # Calculate pack hash if local file
        pack_hash = None
        size_bytes = None

        if os.path.exists(pack_path):
            with open(pack_path, "rb") as f:
                content = f.read()
                pack_hash = hashlib.sha256(content).hexdigest()
                size_bytes = len(content)
        else:
            # For remote paths, hash must be provided or calculated separately
            pack_hash = hashlib.sha256(pack_path.encode()).hexdigest()

        payload = {
            "version": version,
            "pack_path": pack_path,
            "pack_hash": pack_hash,
            "metadata": metadata or {},
            "capabilities": capabilities or [],
            "dependencies": dependencies or [],
            "size_bytes": size_bytes,
            "published_by": published_by
        }

        return self._request("POST", f"/api/v1/agents/{agent_id}/versions", json=payload)

    def get_version(self, agent_id: str, version: str) -> Dict[str, Any]:
        """
        Get specific version details

        Args:
            agent_id: Agent UUID
            version: Version string

        Returns:
            Version metadata
        """
        return self._request("GET", f"/api/v1/agents/{agent_id}/versions/{version}")

    def list_versions(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        List all versions of an agent

        Args:
            agent_id: Agent UUID

        Returns:
            List of version metadata
        """
        return self._request("GET", f"/api/v1/agents/{agent_id}/versions")

    # ========================================================================
    # Certification Operations
    # ========================================================================

    def certify(
        self,
        agent_id: str,
        version: str,
        dimension: str,
        status: str,
        score: Optional[float] = None,
        evidence: Optional[Dict[str, Any]] = None,
        certified_by: str = "GL-CERT"
    ) -> Dict[str, Any]:
        """
        Submit certification for an agent version

        Args:
            agent_id: Agent UUID
            version: Version string
            dimension: Certification dimension (security, performance, reliability, etc.)
            status: Certification status (passed, failed, pending)
            score: Certification score (0-100)
            evidence: Supporting evidence and test results
            certified_by: Certifying authority

        Returns:
            Certification metadata
        """
        payload = {
            "version": version,
            "dimension": dimension,
            "status": status,
            "score": score,
            "evidence": evidence or {},
            "certified_by": certified_by
        }

        return self._request("POST", f"/api/v1/agents/{agent_id}/certify", json=payload)

    def list_certifications(
        self,
        agent_id: str,
        version: Optional[str] = None,
        dimension: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List certifications for an agent

        Args:
            agent_id: Agent UUID
            version: Filter by version
            dimension: Filter by dimension

        Returns:
            List of certification metadata
        """
        params = {}
        if version:
            params["version"] = version
        if dimension:
            params["dimension"] = dimension

        return self._request("GET", f"/api/v1/agents/{agent_id}/certifications", params=params)

    # ========================================================================
    # Download Tracking
    # ========================================================================

    def track_download(
        self,
        agent_id: str,
        version: str,
        downloaded_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track agent download for analytics

        Args:
            agent_id: Agent UUID
            version: Version string
            downloaded_by: Downloader identifier

        Returns:
            Status response
        """
        params = {}
        if downloaded_by:
            params["downloaded_by"] = downloaded_by

        return self._request(
            "POST",
            f"/api/v1/agents/{agent_id}/versions/{version}/download",
            params=params
        )

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        return self._request("GET", "/health")

    def ready_check(self) -> Dict[str, Any]:
        """Check API readiness"""
        return self._request("GET", "/api/v1/ready")


# ============================================================================
# Async Registry Client
# ============================================================================

class AsyncRegistryClient:
    """
    Async version of RegistryClient for high-performance applications

    Example:
        async with AsyncRegistryClient(base_url="http://localhost:8000") as client:
            agent = await client.register(name="thermosync", namespace="greenlang")
            version = await client.publish_version(agent["id"], "1.0.0", "/path/to/pack.glpack")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = Timeout(timeout)

        self.client = httpx.AsyncClient(timeout=self.timeout)

        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "GreenLang-SDK/1.0.0"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self.client.aclose()

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make async HTTP request"""
        url = f"{self.base_url}{endpoint}"

        try:
            response = await self.client.request(
                method=method,
                url=url,
                headers=self.headers,
                **kwargs
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code

            if status_code == 404:
                raise AgentNotFoundError(f"Resource not found: {endpoint}")
            elif status_code == 409:
                raise AgentAlreadyExistsError(f"Resource already exists: {endpoint}")
            else:
                error_detail = e.response.json().get("detail", str(e))
                raise RegistryError(f"API error ({status_code}): {error_detail}")

        except httpx.RequestError as e:
            raise RegistryError(f"Request failed: {str(e)}")

    # All methods similar to RegistryClient but with async/await
    async def register(self, name: str, namespace: str = "default", **kwargs) -> Dict[str, Any]:
        spec_hash = kwargs.get("spec_hash")
        if not spec_hash:
            spec_content = json.dumps({"name": name, "namespace": namespace}, sort_keys=True)
            spec_hash = hashlib.sha256(spec_content.encode()).hexdigest()

        payload = {
            "name": name,
            "namespace": namespace,
            "spec_hash": spec_hash,
            **kwargs
        }
        return await self._request("POST", "/api/v1/agents", json=payload)

    async def get(self, agent_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/v1/agents/{agent_id}")

    async def list_agents(self, **params) -> Dict[str, Any]:
        return await self._request("GET", "/api/v1/agents", params=params)

    async def publish_version(self, agent_id: str, version: str, pack_path: str, **kwargs) -> Dict[str, Any]:
        pack_hash = None
        size_bytes = None

        if os.path.exists(pack_path):
            with open(pack_path, "rb") as f:
                content = f.read()
                pack_hash = hashlib.sha256(content).hexdigest()
                size_bytes = len(content)
        else:
            pack_hash = hashlib.sha256(pack_path.encode()).hexdigest()

        payload = {
            "version": version,
            "pack_path": pack_path,
            "pack_hash": pack_hash,
            "size_bytes": size_bytes,
            "metadata": kwargs.get("metadata", {}),
            "capabilities": kwargs.get("capabilities", []),
            "dependencies": kwargs.get("dependencies", []),
            "published_by": kwargs.get("published_by")
        }

        return await self._request("POST", f"/api/v1/agents/{agent_id}/versions", json=payload)

    async def certify(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        return await self._request("POST", f"/api/v1/agents/{agent_id}/certify", json=kwargs)

    async def health_check(self) -> Dict[str, Any]:
        return await self._request("GET", "/health")
