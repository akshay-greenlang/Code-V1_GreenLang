# -*- coding: utf-8 -*-
"""
GreenLang Python SDK

Official Python SDK for the GreenLang API.

Usage:
    from greenlang_sdk import GreenLangClient

    client = GreenLangClient(api_key="your-api-key")

    # Execute a workflow
    result = client.execute_workflow(
        workflow_id="wf_123",
        input_data={"query": "What is carbon footprint?"}
    )

    print(result.data)

Features:
- Type-safe API with Pydantic models
- Automatic retry logic with exponential backoff
- Pagination support
- Streaming results
- Comprehensive error handling
"""

from .client import GreenLangClient
from .models import (
    Workflow,
    WorkflowDefinition,
    Agent,
    ExecutionResult,
    Citation,
    APIError,
    AuthenticationError,
    RateLimitError,
    ValidationError as SDKValidationError,
)
from .exceptions import (
    GreenLangException,
    APIException,
    AuthenticationException,
    RateLimitException,
    NotFoundException,
    ValidationException,
)

__version__ = "1.0.0"
__all__ = [
    "GreenLangClient",
    "Workflow",
    "WorkflowDefinition",
    "Agent",
    "ExecutionResult",
    "Citation",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "SDKValidationError",
    "GreenLangException",
    "APIException",
    "AuthenticationException",
    "RateLimitException",
    "NotFoundException",
    "ValidationException",
]
