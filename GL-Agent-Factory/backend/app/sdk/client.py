"""
GreenLang SDK Client

Pre-built Python client for the GreenLang API with:
- Sync and async support
- Automatic retry with exponential backoff
- Type-safe methods
- Comprehensive error handling

Example:
    >>> from app.sdk import GreenLangClient
    >>> client = GreenLangClient(api_key="gl_your_key")
    >>> agents = client.list_agents()
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, Field


# =============================================================================
# Exceptions
# =============================================================================


class APIError(Exception):
    """Base exception for GreenLang API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthenticationError(APIError):
    """Authentication failed - invalid or missing API key."""

    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(message, status_code=401, error_code="UNAUTHORIZED")


class RateLimitError(APIError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(
            message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
        )
        self.retry_after = retry_after


class ValidationError(APIError):
    """Request validation failed."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class NotFoundError(APIError):
    """Resource not found."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404, error_code="NOT_FOUND")


# =============================================================================
# Response Models
# =============================================================================


class Agent(BaseModel):
    """Agent model."""

    id: str
    agent_id: str
    name: str
    version: str
    state: str
    category: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    tenant_id: str
    created_at: str
    updated_at: str


class AgentListResponse(BaseModel):
    """Paginated list of agents."""

    data: List[Agent]
    meta: Dict[str, Any]


class Execution(BaseModel):
    """Execution model."""

    id: str
    execution_id: str
    agent_id: str
    version: str
    status: str
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None


class ExecutionListResponse(BaseModel):
    """Paginated list of executions."""

    data: List[Execution]
    meta: Dict[str, Any]


class BatchJob(BaseModel):
    """Batch job model."""

    job_id: str
    status: str
    progress: Dict[str, Any]
    created_at: str
    completed_at: Optional[str] = None


# =============================================================================
# Configuration
# =============================================================================


DEFAULT_BASE_URL = "https://api.greenlang.io"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 5, 10, 30]  # Exponential backoff delays


# =============================================================================
# Base Client
# =============================================================================


class BaseClient:
    """Base client with shared functionality."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Initialize the client.

        Args:
            api_key: GreenLang API key (gl_xxx format)
            base_url: API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for rate limits
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "GreenLang-Python-SDK/1.0.0",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _handle_response(self, response: httpx.Response) -> Any:
        """Handle API response and raise appropriate exceptions."""
        # Rate limit
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise RateLimitError(
                f"Rate limit exceeded. Retry after {retry_after} seconds.",
                retry_after=retry_after,
            )

        # Authentication
        if response.status_code == 401:
            raise AuthenticationError()

        # Not found
        if response.status_code == 404:
            raise NotFoundError()

        # Validation error
        if response.status_code == 400:
            try:
                error_data = response.json()
                error = error_data.get("error", {})
                raise ValidationError(
                    error.get("message", "Validation error"),
                    details=error.get("details"),
                )
            except ValueError:
                raise ValidationError("Invalid request")

        # Other errors
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error = error_data.get("error", {})
                raise APIError(
                    error.get("message", "API error"),
                    status_code=response.status_code,
                    error_code=error.get("code"),
                    details=error.get("details"),
                )
            except ValueError:
                raise APIError(
                    response.text or "Unknown error",
                    status_code=response.status_code,
                )

        # No content
        if response.status_code == 204:
            return None

        # Parse JSON response
        return response.json()


# =============================================================================
# Synchronous Client
# =============================================================================


class GreenLangClient(BaseClient):
    """
    Synchronous GreenLang API client.

    Example:
        >>> client = GreenLangClient(api_key="gl_your_api_key")
        >>>
        >>> # List agents
        >>> agents = client.list_agents()
        >>> for agent in agents.data:
        ...     print(f"{agent.agent_id}: {agent.name}")
        >>>
        >>> # Execute an agent
        >>> result = client.execute_agent(
        ...     agent_id="carbon/calculator",
        ...     inputs={"activity_type": "electricity", "quantity": 1000}
        ... )
        >>> print(f"Carbon: {result.outputs['carbon_footprint']} tCO2e")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._get_headers(),
            timeout=self.timeout,
        )

    def __enter__(self) -> "GreenLangClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make HTTP request with retry logic."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    path,
                    params=params,
                    json=json,
                )
                return self._handle_response(response)

            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = min(e.retry_after, RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    time.sleep(delay)
                    continue
                raise

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_error = APIError(f"Network error: {e}")
                if attempt < self.max_retries:
                    time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                raise last_error

        raise last_error or APIError("Unknown error")

    # =========================================================================
    # Agent Methods
    # =========================================================================

    def list_agents(
        self,
        category: Optional[str] = None,
        state: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> AgentListResponse:
        """
        List agents with optional filtering.

        Args:
            category: Filter by category
            state: Filter by state (DRAFT, CERTIFIED, etc.)
            tags: Filter by tags
            search: Text search
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Paginated list of agents
        """
        params = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if state:
            params["state"] = state
        if tags:
            params["tags"] = ",".join(tags)
        if search:
            params["search"] = search

        data = self._request("GET", "/v1/agents", params=params)
        return AgentListResponse(**data)

    def get_agent(self, agent_id: str) -> Agent:
        """
        Get agent by ID.

        Args:
            agent_id: Agent ID

        Returns:
            Agent details
        """
        data = self._request("GET", f"/v1/agents/{agent_id}")
        return Agent(**data)

    def create_agent(
        self,
        agent_id: str,
        name: str,
        category: str,
        entrypoint: str,
        version: str = "1.0.0",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Agent:
        """
        Create a new agent.

        Args:
            agent_id: Unique agent ID (category/name format)
            name: Human-readable name
            category: Agent category
            entrypoint: Python entrypoint
            version: Initial version
            description: Optional description
            tags: Optional tags

        Returns:
            Created agent
        """
        body = {
            "agent_id": agent_id,
            "name": name,
            "category": category,
            "entrypoint": entrypoint,
            "version": version,
        }
        if description:
            body["description"] = description
        if tags:
            body["tags"] = tags

        data = self._request("POST", "/v1/agents", json=body)
        return Agent(**data)

    # =========================================================================
    # Execution Methods
    # =========================================================================

    def execute_agent(
        self,
        agent_id: str,
        inputs: Dict[str, Any],
        version: Optional[str] = None,
        async_mode: bool = False,
    ) -> Execution:
        """
        Execute an agent.

        Args:
            agent_id: Agent to execute
            inputs: Execution inputs
            version: Optional version (latest if not specified)
            async_mode: Run asynchronously

        Returns:
            Execution result
        """
        body = {"inputs": inputs}
        if version:
            body["version"] = version
        if async_mode:
            body["async"] = True

        data = self._request("POST", f"/v1/agents/{agent_id}/execute", json=body)
        return Execution(**data)

    def get_execution(self, execution_id: str) -> Execution:
        """
        Get execution by ID.

        Args:
            execution_id: Execution ID

        Returns:
            Execution details
        """
        data = self._request("GET", f"/v1/executions/{execution_id}")
        return Execution(**data)

    def list_executions(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ExecutionListResponse:
        """
        List executions.

        Args:
            agent_id: Filter by agent
            status: Filter by status
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Paginated list of executions
        """
        params = {"limit": limit, "offset": offset}
        if agent_id:
            params["agent_id"] = agent_id
        if status:
            params["status"] = status

        data = self._request("GET", "/v1/executions", params=params)
        return ExecutionListResponse(**data)

    # =========================================================================
    # Batch Methods
    # =========================================================================

    def create_batch_job(
        self,
        file_path: str,
        agent_id: str,
        input_mapping: Dict[str, str],
        output_format: str = "csv",
    ) -> BatchJob:
        """
        Create a batch processing job.

        Args:
            file_path: Path to input file
            agent_id: Agent to execute
            input_mapping: Column to input field mapping
            output_format: Output format (csv, excel, json)

        Returns:
            Created batch job
        """
        import json as json_lib

        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {
                "config": json_lib.dumps({
                    "agent_id": agent_id,
                    "input_mapping": input_mapping,
                    "output_format": output_format,
                })
            }

            # Use a fresh client for multipart
            with httpx.Client(
                base_url=self.base_url,
                headers={"X-API-Key": self.api_key},
                timeout=self.timeout,
            ) as client:
                response = client.post("/v1/batch/jobs", files=files, data=data)
                result = self._handle_response(response)
                return BatchJob(**result)

    def get_batch_job(self, job_id: str) -> BatchJob:
        """
        Get batch job status.

        Args:
            job_id: Job ID

        Returns:
            Batch job details
        """
        data = self._request("GET", f"/v1/batch/jobs/{job_id}")
        return BatchJob(**data)


# =============================================================================
# Asynchronous Client
# =============================================================================


class GreenLangAsyncClient(BaseClient):
    """
    Asynchronous GreenLang API client.

    Example:
        >>> async with GreenLangAsyncClient(api_key="gl_your_key") as client:
        ...     agents = await client.list_agents()
        ...     for agent in agents.data:
        ...         print(f"{agent.agent_id}: {agent.name}")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "GreenLangAsyncClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make HTTP request with retry logic."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(
                    method,
                    path,
                    params=params,
                    json=json,
                )
                return self._handle_response(response)

            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = min(e.retry_after, RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    await asyncio.sleep(delay)
                    continue
                raise

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_error = APIError(f"Network error: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                    continue
                raise last_error

        raise last_error or APIError("Unknown error")

    # =========================================================================
    # Agent Methods
    # =========================================================================

    async def list_agents(
        self,
        category: Optional[str] = None,
        state: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> AgentListResponse:
        """List agents with optional filtering."""
        params = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if state:
            params["state"] = state
        if tags:
            params["tags"] = ",".join(tags)
        if search:
            params["search"] = search

        data = await self._request("GET", "/v1/agents", params=params)
        return AgentListResponse(**data)

    async def get_agent(self, agent_id: str) -> Agent:
        """Get agent by ID."""
        data = await self._request("GET", f"/v1/agents/{agent_id}")
        return Agent(**data)

    async def create_agent(
        self,
        agent_id: str,
        name: str,
        category: str,
        entrypoint: str,
        version: str = "1.0.0",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Agent:
        """Create a new agent."""
        body = {
            "agent_id": agent_id,
            "name": name,
            "category": category,
            "entrypoint": entrypoint,
            "version": version,
        }
        if description:
            body["description"] = description
        if tags:
            body["tags"] = tags

        data = await self._request("POST", "/v1/agents", json=body)
        return Agent(**data)

    # =========================================================================
    # Execution Methods
    # =========================================================================

    async def execute_agent(
        self,
        agent_id: str,
        inputs: Dict[str, Any],
        version: Optional[str] = None,
        async_mode: bool = False,
    ) -> Execution:
        """Execute an agent."""
        body = {"inputs": inputs}
        if version:
            body["version"] = version
        if async_mode:
            body["async"] = True

        data = await self._request("POST", f"/v1/agents/{agent_id}/execute", json=body)
        return Execution(**data)

    async def get_execution(self, execution_id: str) -> Execution:
        """Get execution by ID."""
        data = await self._request("GET", f"/v1/executions/{execution_id}")
        return Execution(**data)

    async def list_executions(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ExecutionListResponse:
        """List executions."""
        params = {"limit": limit, "offset": offset}
        if agent_id:
            params["agent_id"] = agent_id
        if status:
            params["status"] = status

        data = await self._request("GET", "/v1/executions", params=params)
        return ExecutionListResponse(**data)

    # =========================================================================
    # Batch Methods
    # =========================================================================

    async def get_batch_job(self, job_id: str) -> BatchJob:
        """Get batch job status."""
        data = await self._request("GET", f"/v1/batch/jobs/{job_id}")
        return BatchJob(**data)
