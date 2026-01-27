# -*- coding: utf-8 -*-
"""
Legacy HTTP Agent Adapter
==========================

GLIP v1 wrapper for existing HTTP-based agents.

This adapter allows legacy agents that use HTTP/REST invocation
to work within the GLIP v1 execution model without modification.

The adapter:
1. Reads RunContext from GL_INPUT_URI
2. Translates to HTTP request
3. Invokes the legacy agent endpoint
4. Translates response to result.json
5. Writes outputs to GL_OUTPUT_URI

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, Field

from greenlang.orchestrator.executors.base import (
    ArtifactReference,
    RunContext,
    StepMetadata,
    StepResult,
)
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class HttpMethod(str, Enum):
    """HTTP methods supported for legacy agents."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"


class AuthType(str, Enum):
    """Authentication types for legacy agents."""
    NONE = "none"
    BEARER_TOKEN = "bearer_token"
    API_KEY = "api_key"
    BASIC = "basic"
    OAUTH2 = "oauth2"


class AdapterConfig(BaseModel):
    """Configuration for a legacy HTTP agent adapter."""
    # Identity
    agent_id: str = Field(..., description="Agent ID being adapted")
    agent_version: str = Field(..., description="Agent version")

    # Endpoint
    base_url: str = Field(..., description="Base URL of legacy agent")
    endpoint_path: str = Field(default="/execute", description="Endpoint path")
    method: HttpMethod = Field(default=HttpMethod.POST, description="HTTP method")

    # Authentication
    auth_type: AuthType = Field(default=AuthType.NONE, description="Auth type")
    auth_config: Dict[str, Any] = Field(default_factory=dict, description="Auth config")

    # Request mapping
    param_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Map GLIP params to HTTP request fields"
    )
    input_artifact_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Map input artifacts to HTTP request fields"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers"
    )

    # Response mapping
    output_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Map HTTP response to GLIP outputs"
    )
    artifact_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Map HTTP response fields to artifacts"
    )
    success_field: str = Field(
        default="success",
        description="Response field indicating success"
    )
    error_field: str = Field(
        default="error",
        description="Response field containing error message"
    )

    # Behavior
    timeout_seconds: int = Field(default=300, description="HTTP timeout")
    retry_on_5xx: bool = Field(default=True, description="Retry on 5xx errors")
    max_retries: int = Field(default=3, description="Max retries")
    retry_delay_seconds: float = Field(default=1.0, description="Retry delay")

    # Validation
    validate_response_schema: bool = Field(
        default=False,
        description="Validate response against schema"
    )
    response_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="JSON schema for response validation"
    )


class HttpLegacyAdapter:
    """
    Adapter that wraps legacy HTTP agents for GLIP v1 compatibility.

    This is designed to be used as a container entrypoint that:
    1. Reads GLIP v1 RunContext from environment
    2. Makes HTTP call to legacy agent
    3. Writes GLIP v1 outputs to artifact store

    Usage (as standalone script in container):
        adapter = HttpLegacyAdapter(config)
        await adapter.run()

    Usage (programmatically):
        adapter = HttpLegacyAdapter(config)
        result = await adapter.execute(run_context)
    """

    def __init__(self, config: AdapterConfig):
        """
        Initialize the adapter.

        Args:
            config: Adapter configuration
        """
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        logger.info(f"Initialized HttpLegacyAdapter for {config.agent_id}@{config.agent_version}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                follow_redirects=True,
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_auth_headers(self) -> Dict[str, str]:
        """Build authentication headers."""
        headers = {}

        if self.config.auth_type == AuthType.BEARER_TOKEN:
            token = self.config.auth_config.get("token", "")
            headers["Authorization"] = f"Bearer {token}"

        elif self.config.auth_type == AuthType.API_KEY:
            key_name = self.config.auth_config.get("header_name", "X-API-Key")
            key_value = self.config.auth_config.get("api_key", "")
            headers[key_name] = key_value

        elif self.config.auth_type == AuthType.BASIC:
            import base64
            username = self.config.auth_config.get("username", "")
            password = self.config.auth_config.get("password", "")
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"

        return headers

    def _build_request_body(
        self,
        context: RunContext,
    ) -> Dict[str, Any]:
        """
        Build HTTP request body from RunContext.

        Args:
            context: GLIP v1 RunContext

        Returns:
            Request body dictionary
        """
        body = {}

        # Map params
        for glip_param, http_field in self.config.param_mapping.items():
            if glip_param in context.params:
                body[http_field] = context.params[glip_param]

        # If no explicit mapping, pass all params
        if not self.config.param_mapping:
            body.update(context.params)

        # Map input artifacts
        for glip_input, http_field in self.config.input_artifact_mapping.items():
            if glip_input in context.inputs:
                artifact = context.inputs[glip_input]
                body[http_field] = artifact.uri

        # Add standard GLIP fields for context
        body["_glip_context"] = {
            "run_id": context.run_id,
            "step_id": context.step_id,
            "idempotency_key": context.idempotency_key,
            "trace_id": context.trace_id,
        }

        return body

    def _parse_response(
        self,
        response_data: Dict[str, Any],
        http_status: int,
    ) -> StepResult:
        """
        Parse HTTP response into StepResult.

        Args:
            response_data: HTTP response body as dict
            http_status: HTTP status code

        Returns:
            StepResult
        """
        # Determine success
        if self.config.success_field in response_data:
            success = bool(response_data[self.config.success_field])
        else:
            success = 200 <= http_status < 300

        # Extract error
        error = None
        if not success and self.config.error_field in response_data:
            error = str(response_data[self.config.error_field])

        # Map outputs
        data = {}
        if self.config.output_mapping:
            for http_field, glip_field in self.config.output_mapping.items():
                if http_field in response_data:
                    data[glip_field] = response_data[http_field]
        else:
            # No mapping, pass through all data
            data = {k: v for k, v in response_data.items()
                    if k not in [self.config.success_field, self.config.error_field, "_glip_context"]}

        # Map artifacts
        artifacts = {}
        for http_field, artifact_name in self.config.artifact_mapping.items():
            if http_field in response_data:
                uri = response_data[http_field]
                artifacts[artifact_name] = ArtifactReference(
                    uri=uri,
                    checksum="",  # Would need to fetch and compute
                )

        return StepResult(
            success=success,
            data=data,
            artifacts=artifacts,
            error=error,
        )

    async def execute(
        self,
        context: RunContext,
    ) -> tuple[StepResult, StepMetadata]:
        """
        Execute the adapted agent.

        Args:
            context: GLIP v1 RunContext

        Returns:
            Tuple of (StepResult, StepMetadata)
        """
        started_at = DeterministicClock.now()
        client = await self._get_client()

        # Build request
        url = f"{self.config.base_url.rstrip('/')}{self.config.endpoint_path}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-GreenLang-Run-ID": context.run_id,
            "X-GreenLang-Step-ID": context.step_id,
            "X-GreenLang-Trace-ID": context.trace_id,
            "X-GreenLang-Idempotency-Key": context.idempotency_key,
            **self.config.headers,
            **self._build_auth_headers(),
        }
        body = self._build_request_body(context)

        logger.info(f"Calling legacy agent: {self.config.method.value} {url}")
        logger.debug(f"Request body: {json.dumps(body, default=str)}")

        # Execute with retry
        last_error = None
        response_data = None
        http_status = 0

        for attempt in range(self.config.max_retries + 1):
            try:
                if self.config.method == HttpMethod.GET:
                    response = await client.get(url, headers=headers, params=body)
                elif self.config.method == HttpMethod.POST:
                    response = await client.post(url, headers=headers, json=body)
                elif self.config.method == HttpMethod.PUT:
                    response = await client.put(url, headers=headers, json=body)
                elif self.config.method == HttpMethod.PATCH:
                    response = await client.patch(url, headers=headers, json=body)
                else:
                    raise ValueError(f"Unsupported HTTP method: {self.config.method}")

                http_status = response.status_code

                # Check for retryable errors
                if self.config.retry_on_5xx and 500 <= http_status < 600:
                    if attempt < self.config.max_retries:
                        logger.warning(f"Got {http_status}, retrying ({attempt + 1}/{self.config.max_retries})")
                        await self._sleep(self.config.retry_delay_seconds * (attempt + 1))
                        continue

                # Parse response
                try:
                    response_data = response.json()
                except json.JSONDecodeError:
                    response_data = {"_raw_response": response.text}

                break

            except httpx.TimeoutException as e:
                last_error = f"Request timeout: {e}"
                if attempt < self.config.max_retries:
                    logger.warning(f"Timeout, retrying ({attempt + 1}/{self.config.max_retries})")
                    await self._sleep(self.config.retry_delay_seconds * (attempt + 1))
                    continue

            except httpx.RequestError as e:
                last_error = f"Request error: {e}"
                if attempt < self.config.max_retries:
                    logger.warning(f"Request error, retrying ({attempt + 1}/{self.config.max_retries})")
                    await self._sleep(self.config.retry_delay_seconds * (attempt + 1))
                    continue

        completed_at = DeterministicClock.now()
        duration_ms = (completed_at - started_at).total_seconds() * 1000

        # Build result
        if response_data is not None:
            result = self._parse_response(response_data, http_status)
        else:
            result = StepResult(
                success=False,
                error=last_error or "Unknown error",
                error_code="GL-E-HTTP-ADAPTER-ERROR",
            )

        # Build metadata
        result_json = json.dumps(result.model_dump(), sort_keys=True, default=str)
        result_checksum = hashlib.sha256(result_json.encode()).hexdigest()

        metadata = StepMetadata(
            agent_id=self.config.agent_id,
            agent_version=self.config.agent_version,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            result_checksum=result_checksum,
            artifacts_checksums={},
            input_context_hash=context.compute_hash(),
            idempotency_key=context.idempotency_key,
            exit_code=0 if result.success else 1,
            status="success" if result.success else "failed",
        )

        logger.info(f"Legacy agent completed: success={result.success}, duration={duration_ms:.2f}ms")

        return result, metadata

    async def _sleep(self, seconds: float):
        """Async sleep wrapper."""
        import asyncio
        await asyncio.sleep(seconds)

    @classmethod
    def from_agent_registry(
        cls,
        agent_id: str,
        agent_version: str,
        endpoint_config: Dict[str, Any],
    ) -> "HttpLegacyAdapter":
        """
        Create adapter from agent registry configuration.

        Args:
            agent_id: Agent ID
            agent_version: Agent version
            endpoint_config: Endpoint configuration from registry

        Returns:
            Configured HttpLegacyAdapter
        """
        config = AdapterConfig(
            agent_id=agent_id,
            agent_version=agent_version,
            **endpoint_config,
        )
        return cls(config)


class AdapterContainerEntrypoint:
    """
    Container entrypoint for running HTTP adapter.

    This is the `gl-adapter-http` command that gets run in adapter containers.

    Usage:
        GL_INPUT_URI=s3://... GL_OUTPUT_URI=s3://... gl-adapter-http
    """

    def __init__(
        self,
        adapter_config: AdapterConfig,
        artifact_store: Any,  # ArtifactStore
    ):
        """
        Initialize entrypoint.

        Args:
            adapter_config: Adapter configuration
            artifact_store: Artifact store for I/O
        """
        self.config = adapter_config
        self.artifact_store = artifact_store
        self.adapter = HttpLegacyAdapter(adapter_config)

    async def run(self) -> int:
        """
        Run the adapter entrypoint.

        Reads from GL_INPUT_URI, executes adapter, writes to GL_OUTPUT_URI.

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        import os

        input_uri = os.environ.get("GL_INPUT_URI")
        output_uri = os.environ.get("GL_OUTPUT_URI")
        run_id = os.environ.get("GL_RUN_ID")
        step_id = os.environ.get("GL_STEP_ID")
        tenant_id = os.environ.get("GL_TENANT_ID")

        if not all([input_uri, output_uri, run_id, step_id, tenant_id]):
            logger.error("Missing required environment variables")
            return 1

        try:
            # Read input context
            context_data = await self.artifact_store.read_artifact(
                run_id=run_id,
                step_id=step_id,
                name="input.json",
                tenant_id=tenant_id,
            )

            if not context_data:
                logger.error("Failed to read input context")
                return 1

            context = RunContext(**json.loads(context_data))

            # Execute adapter
            result, metadata = await self.adapter.execute(context)

            # Write outputs
            await self._write_outputs(result, metadata, run_id, step_id, tenant_id)

            return 0 if result.success else 1

        except Exception as e:
            logger.error(f"Adapter execution failed: {e}", exc_info=True)

            # Write error
            error_result = StepResult(
                success=False,
                error=str(e),
                error_code="GL-E-HTTP-ADAPTER-CRASH",
            )
            await self._write_error(error_result, run_id, step_id, tenant_id)

            return 1

        finally:
            await self.adapter.close()

    async def _write_outputs(
        self,
        result: StepResult,
        metadata: StepMetadata,
        run_id: str,
        step_id: str,
        tenant_id: str,
    ):
        """Write result and metadata to artifact store."""
        # Write result.json
        result_data = json.dumps(result.model_dump(), sort_keys=True, default=str, indent=2)
        await self.artifact_store.write_artifact(
            run_id=run_id,
            step_id=step_id,
            name="result.json",
            data=result_data.encode(),
            media_type="application/json",
            tenant_id=tenant_id,
        )

        # Write step_metadata.json
        metadata_data = json.dumps(metadata.model_dump(), sort_keys=True, default=str, indent=2)
        await self.artifact_store.write_artifact(
            run_id=run_id,
            step_id=step_id,
            name="step_metadata.json",
            data=metadata_data.encode(),
            media_type="application/json",
            tenant_id=tenant_id,
        )

    async def _write_error(
        self,
        error: StepResult,
        run_id: str,
        step_id: str,
        tenant_id: str,
    ):
        """Write error.json to artifact store."""
        error_data = json.dumps(error.model_dump(), sort_keys=True, default=str, indent=2)
        await self.artifact_store.write_artifact(
            run_id=run_id,
            step_id=step_id,
            name="error.json",
            data=error_data.encode(),
            media_type="application/json",
            tenant_id=tenant_id,
        )


def create_adapter_config(
    agent_id: str,
    agent_version: str,
    base_url: str,
    endpoint_path: str = "/execute",
    method: str = "POST",
    auth_type: str = "none",
    auth_config: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> AdapterConfig:
    """
    Factory function to create adapter configuration.

    Args:
        agent_id: Agent ID being adapted
        agent_version: Agent version
        base_url: Base URL of legacy agent
        endpoint_path: Endpoint path
        method: HTTP method
        auth_type: Authentication type
        auth_config: Authentication configuration
        headers: Additional headers

    Returns:
        AdapterConfig instance
    """
    return AdapterConfig(
        agent_id=agent_id,
        agent_version=agent_version,
        base_url=base_url,
        endpoint_path=endpoint_path,
        method=HttpMethod(method.upper()),
        auth_type=AuthType(auth_type.lower()),
        auth_config=auth_config or {},
        headers=headers or {},
    )
