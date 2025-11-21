# -*- coding: utf-8 -*-
"""
GreenLang SDK Client

Main client class for interacting with the GreenLang API.
"""

import time
from typing import Dict, List, Optional, Iterator, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import (
    Workflow,
    WorkflowDefinition,
    Agent,
    ExecutionResult,
    Citation,
    PaginatedResponse,
)
from .exceptions import (
    APIException,
    AuthenticationException,
    RateLimitException,
    NotFoundException,
    ValidationException,
    GreenLangException,
)


class GreenLangClient:
    """
    GreenLang API Client

    The main entry point for interacting with the GreenLang API.

    Args:
        api_key: Your GreenLang API key
        base_url: Base URL for the API (default: https://api.greenlang.com)
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum number of retries for failed requests (default: 3)

    Example:
        >>> client = GreenLangClient(api_key="gl_your_api_key_here")
        >>> workflows = client.list_workflows()
        >>> for workflow in workflows:
        ...     print(workflow.name)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.greenlang.com",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Create session with retry logic
        self.session = requests.Session()

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # 1s, 2s, 4s, etc.
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "greenlang-python-sdk/1.0.0",
        })

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
    ) -> Dict:
        """
        Make HTTP request to API

        Args:
            method: HTTP method
            endpoint: API endpoint (without base URL)
            params: Query parameters
            json: JSON body

        Returns:
            Response JSON

        Raises:
            AuthenticationException: If authentication fails
            RateLimitException: If rate limit is exceeded
            NotFoundException: If resource not found
            ValidationException: If request validation fails
            APIException: For other API errors
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                timeout=self.timeout,
            )

            # Handle error responses
            if response.status_code >= 400:
                self._handle_error_response(response)

            return response.json()

        except requests.exceptions.Timeout:
            raise APIException("Request timeout")
        except requests.exceptions.ConnectionError:
            raise APIException("Connection error")
        except requests.exceptions.RequestException as e:
            raise APIException(f"Request failed: {str(e)}")

    def _handle_error_response(self, response: requests.Response):
        """Handle error responses from API"""
        try:
            error_data = response.json()
            message = error_data.get("detail", response.text)
        except Exception:
            message = response.text

        if response.status_code == 401:
            raise AuthenticationException(message)
        elif response.status_code == 404:
            raise NotFoundException(message)
        elif response.status_code == 422:
            raise ValidationException(message)
        elif response.status_code == 429:
            # Extract rate limit info from headers
            reset_time = response.headers.get("X-RateLimit-Reset", "unknown")
            raise RateLimitException(f"{message}. Reset at: {reset_time}")
        else:
            raise APIException(f"API error ({response.status_code}): {message}")

    # Workflow Methods

    def create_workflow(self, workflow_def: Dict) -> Workflow:
        """
        Create a new workflow

        Args:
            workflow_def: Workflow definition dictionary

        Returns:
            Created Workflow object

        Example:
            >>> workflow_def = {
            ...     "name": "Carbon Analysis",
            ...     "description": "Analyze carbon emissions",
            ...     "agents": [
            ...         {"agent_id": "carbon_analyzer", "config": {}}
            ...     ]
            ... }
            >>> workflow = client.create_workflow(workflow_def)
            >>> print(workflow.id)
        """
        response = self._request("POST", "/api/workflows", json=workflow_def)
        return Workflow(**response)

    def get_workflow(self, workflow_id: str) -> Workflow:
        """
        Get workflow by ID

        Args:
            workflow_id: Workflow ID

        Returns:
            Workflow object

        Example:
            >>> workflow = client.get_workflow("wf_123")
            >>> print(workflow.name)
        """
        response = self._request("GET", f"/api/workflows/{workflow_id}")
        return Workflow(**response)

    def list_workflows(
        self,
        limit: int = 20,
        offset: int = 0,
        category: Optional[str] = None,
    ) -> List[Workflow]:
        """
        List workflows

        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            category: Filter by category

        Returns:
            List of Workflow objects

        Example:
            >>> workflows = client.list_workflows(limit=10, category="carbon")
            >>> for wf in workflows:
            ...     print(wf.name)
        """
        params = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category

        response = self._request("GET", "/api/workflows", params=params)
        return [Workflow(**wf) for wf in response.get("items", [])]

    def list_workflows_iter(
        self,
        page_size: int = 20,
        category: Optional[str] = None,
    ) -> Iterator[Workflow]:
        """
        Iterate over all workflows with automatic pagination

        Args:
            page_size: Number of items per page
            category: Filter by category

        Yields:
            Workflow objects

        Example:
            >>> for workflow in client.list_workflows_iter():
            ...     print(workflow.name)
        """
        offset = 0
        while True:
            workflows = self.list_workflows(
                limit=page_size,
                offset=offset,
                category=category,
            )
            if not workflows:
                break

            for workflow in workflows:
                yield workflow

            if len(workflows) < page_size:
                break

            offset += page_size

    def update_workflow(self, workflow_id: str, updates: Dict) -> Workflow:
        """
        Update workflow

        Args:
            workflow_id: Workflow ID
            updates: Fields to update

        Returns:
            Updated Workflow object

        Example:
            >>> workflow = client.update_workflow(
            ...     "wf_123",
            ...     {"name": "Updated Name"}
            ... )
        """
        response = self._request("PUT", f"/api/workflows/{workflow_id}", json=updates)
        return Workflow(**response)

    def delete_workflow(self, workflow_id: str) -> None:
        """
        Delete workflow

        Args:
            workflow_id: Workflow ID

        Example:
            >>> client.delete_workflow("wf_123")
        """
        self._request("DELETE", f"/api/workflows/{workflow_id}")

    def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict,
        stream: bool = False,
    ) -> ExecutionResult:
        """
        Execute a workflow

        Args:
            workflow_id: Workflow ID
            input_data: Input data for workflow
            stream: Whether to stream results

        Returns:
            ExecutionResult object

        Example:
            >>> result = client.execute_workflow(
            ...     "wf_123",
            ...     {"query": "What is carbon footprint?"}
            ... )
            >>> print(result.data)
        """
        response = self._request(
            "POST",
            f"/api/workflows/{workflow_id}/execute",
            json={"input_data": input_data, "stream": stream},
        )
        return ExecutionResult(**response)

    # Agent Methods

    def get_agent(self, agent_id: str) -> Agent:
        """
        Get agent by ID

        Args:
            agent_id: Agent ID

        Returns:
            Agent object

        Example:
            >>> agent = client.get_agent("carbon_analyzer")
            >>> print(agent.description)
        """
        response = self._request("GET", f"/api/agents/{agent_id}")
        return Agent(**response)

    def list_agents(
        self,
        category: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Agent]:
        """
        List agents

        Args:
            category: Filter by category
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Agent objects

        Example:
            >>> agents = client.list_agents(category="carbon")
            >>> for agent in agents:
            ...     print(agent.name)
        """
        params = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category

        response = self._request("GET", "/api/agents", params=params)
        return [Agent(**agent) for agent in response.get("items", [])]

    def execute_agent(
        self,
        agent_id: str,
        input_data: Dict,
        config: Optional[Dict] = None,
    ) -> ExecutionResult:
        """
        Execute an agent directly

        Args:
            agent_id: Agent ID
            input_data: Input data for agent
            config: Optional agent configuration

        Returns:
            ExecutionResult object

        Example:
            >>> result = client.execute_agent(
            ...     "carbon_analyzer",
            ...     {"query": "Calculate emissions"}
            ... )
            >>> print(result.data)
        """
        payload = {"input_data": input_data}
        if config:
            payload["config"] = config

        response = self._request(
            "POST",
            f"/api/agents/{agent_id}/execute",
            json=payload,
        )
        return ExecutionResult(**response)

    # Execution Methods

    def get_execution(self, execution_id: str) -> ExecutionResult:
        """
        Get execution result by ID

        Args:
            execution_id: Execution ID

        Returns:
            ExecutionResult object

        Example:
            >>> result = client.get_execution("exec_123")
            >>> print(result.status)
        """
        response = self._request("GET", f"/api/executions/{execution_id}")
        return ExecutionResult(**response)

    def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[ExecutionResult]:
        """
        List execution results

        Args:
            workflow_id: Filter by workflow ID
            status: Filter by status (pending, running, completed, failed)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of ExecutionResult objects

        Example:
            >>> results = client.list_executions(status="completed")
            >>> for result in results:
            ...     print(result.id, result.status)
        """
        params = {"limit": limit, "offset": offset}
        if workflow_id:
            params["workflow_id"] = workflow_id
        if status:
            params["status"] = status

        response = self._request("GET", "/api/executions", params=params)
        return [ExecutionResult(**exec) for exec in response.get("items", [])]

    # Streaming Methods

    def stream_execution(
        self,
        workflow_id: str,
        input_data: Dict,
    ) -> Iterator[Dict]:
        """
        Stream workflow execution results

        Args:
            workflow_id: Workflow ID
            input_data: Input data for workflow

        Yields:
            Streaming result chunks

        Example:
            >>> for chunk in client.stream_execution("wf_123", {"query": "test"}):
            ...     print(chunk)
        """
        url = f"{self.base_url}/api/workflows/{workflow_id}/execute"

        with self.session.post(
            url,
            json={"input_data": input_data, "stream": True},
            stream=True,
            timeout=None,  # No timeout for streaming
        ) as response:
            if response.status_code >= 400:
                self._handle_error_response(response)

            for line in response.iter_lines():
                if line:
                    # Assume SSE format: "data: {...}"
                    if line.startswith(b"data: "):
                        import json
                        data = json.loads(line[6:])
                        yield data

    # Citation Methods

    def get_citations(self, execution_id: str) -> List[Citation]:
        """
        Get citations for an execution

        Args:
            execution_id: Execution ID

        Returns:
            List of Citation objects

        Example:
            >>> citations = client.get_citations("exec_123")
            >>> for citation in citations:
            ...     print(citation.source_url)
        """
        response = self._request("GET", f"/api/executions/{execution_id}/citations")
        return [Citation(**cite) for cite in response.get("citations", [])]

    # Utility Methods

    def health_check(self) -> Dict:
        """
        Check API health status

        Returns:
            Health status dictionary

        Example:
            >>> status = client.health_check()
            >>> print(status["status"])
        """
        return self._request("GET", "/health")

    def close(self):
        """Close the HTTP session"""
        self.session.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Convenience function
def create_client(api_key: str, **kwargs) -> GreenLangClient:
    """
    Create a GreenLang client

    Args:
        api_key: Your GreenLang API key
        **kwargs: Additional arguments for GreenLangClient

    Returns:
        GreenLangClient instance

    Example:
        >>> client = create_client(api_key="gl_your_api_key")
    """
    return GreenLangClient(api_key=api_key, **kwargs)
