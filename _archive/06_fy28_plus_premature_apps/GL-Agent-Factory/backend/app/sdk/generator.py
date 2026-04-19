"""
SDK Generator for GreenLang

Auto-generates Python SDK from OpenAPI specification:
- Parse OpenAPI spec
- Generate Pydantic models
- Generate sync/async client methods
- Type annotations

Example:
    >>> from app.sdk import SDKGenerator, SDKConfig
    >>> generator = SDKGenerator(SDKConfig())
    >>> generator.generate_from_url("https://api.greenlang.io/openapi.json")
"""

import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SDKConfig:
    """Configuration for SDK generation."""

    package_name: str = "greenlang"
    version: str = "1.0.0"
    author: str = "GreenLang"
    author_email: str = "sdk@greenlang.io"
    description: str = "Python SDK for GreenLang API"
    output_dir: str = "./sdk_output"
    include_async: bool = True
    include_sync: bool = True
    generate_models: bool = True
    generate_tests: bool = True


# =============================================================================
# Type Mapping
# =============================================================================


OPENAPI_TO_PYTHON_TYPES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "List",
    "object": "Dict[str, Any]",
}

OPENAPI_FORMAT_TO_PYTHON = {
    "date-time": "datetime",
    "date": "date",
    "email": "str",
    "uri": "str",
    "uuid": "str",
    "binary": "bytes",
}


# =============================================================================
# SDK Generator
# =============================================================================


class SDKGenerator:
    """
    Generates Python SDK from OpenAPI specification.

    Features:
    - Pydantic model generation
    - Sync and async client methods
    - Type annotations
    - Error handling
    - Documentation

    Example:
        >>> generator = SDKGenerator(SDKConfig())
        >>> generator.generate_from_spec(openapi_spec)
    """

    def __init__(self, config: SDKConfig):
        """
        Initialize the generator.

        Args:
            config: SDK configuration
        """
        self.config = config
        self._generated_models: Set[str] = set()
        self._model_imports: Set[str] = set()

    def generate_from_spec(self, spec: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate SDK from OpenAPI specification.

        Args:
            spec: OpenAPI specification dictionary

        Returns:
            Dictionary mapping file paths to content
        """
        files = {}

        # Generate models
        if self.config.generate_models:
            models_content = self._generate_models(spec)
            files["models.py"] = models_content

        # Generate client
        client_content = self._generate_client(spec)
        files["client.py"] = client_content

        # Generate exceptions
        exceptions_content = self._generate_exceptions()
        files["exceptions.py"] = exceptions_content

        # Generate __init__.py
        init_content = self._generate_init()
        files["__init__.py"] = init_content

        # Generate setup.py
        setup_content = self._generate_setup()
        files["setup.py"] = setup_content

        # Generate README
        readme_content = self._generate_readme(spec)
        files["README.md"] = readme_content

        # Generate tests
        if self.config.generate_tests:
            tests_content = self._generate_tests(spec)
            files["tests/test_client.py"] = tests_content
            files["tests/__init__.py"] = ""

        return files

    def generate_to_directory(self, spec: Dict[str, Any]) -> None:
        """
        Generate SDK to output directory.

        Args:
            spec: OpenAPI specification
        """
        files = self.generate_from_spec(spec)
        output_dir = Path(self.config.output_dir) / self.config.package_name

        for file_path, content in files.items():
            full_path = output_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

        logger.info(f"SDK generated at {output_dir}")

    # =========================================================================
    # Model Generation
    # =========================================================================

    def _generate_models(self, spec: Dict[str, Any]) -> str:
        """Generate Pydantic models from OpenAPI schemas."""
        lines = [
            '"""',
            f'{self.config.package_name.title()} SDK Models',
            '',
            'Auto-generated from OpenAPI specification.',
            'Do not edit manually.',
            '"""',
            '',
            'from datetime import datetime, date',
            'from typing import Any, Dict, List, Optional, Union',
            'from enum import Enum',
            '',
            'from pydantic import BaseModel, Field',
            '',
            '',
        ]

        schemas = spec.get("components", {}).get("schemas", {})

        # Generate enums first
        for name, schema in schemas.items():
            if "enum" in schema:
                lines.append(self._generate_enum(name, schema))
                lines.append('')

        # Generate models
        for name, schema in schemas.items():
            if "enum" not in schema and schema.get("type") == "object":
                lines.append(self._generate_model(name, schema, schemas))
                lines.append('')

        return "\n".join(lines)

    def _generate_enum(self, name: str, schema: Dict[str, Any]) -> str:
        """Generate Python enum from OpenAPI enum schema."""
        lines = [
            f'class {name}(str, Enum):',
            f'    """Auto-generated enum."""',
        ]

        for value in schema.get("enum", []):
            # Make valid Python identifier
            key = re.sub(r'[^a-zA-Z0-9_]', '_', str(value).upper())
            if key[0].isdigit():
                key = f"_{key}"
            lines.append(f'    {key} = "{value}"')

        return "\n".join(lines)

    def _generate_model(
        self,
        name: str,
        schema: Dict[str, Any],
        all_schemas: Dict[str, Any],
    ) -> str:
        """Generate Pydantic model from OpenAPI object schema."""
        lines = [
            f'class {name}(BaseModel):',
            f'    """{schema.get("description", "Auto-generated model.")}"""',
        ]

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        if not properties:
            lines.append('    pass')
            return "\n".join(lines)

        for prop_name, prop_schema in properties.items():
            python_type = self._get_python_type(prop_schema, all_schemas)
            description = prop_schema.get("description", "")

            # Handle optional vs required
            if prop_name not in required:
                if "Optional" not in python_type:
                    python_type = f"Optional[{python_type}]"
                default = " = None"
            else:
                default = ""

            # Use Field for description
            if description:
                lines.append(
                    f'    {prop_name}: {python_type} = Field({default.strip(" = ") or "..."}, '
                    f'description="{description}")'
                )
            else:
                lines.append(f'    {prop_name}: {python_type}{default}')

        # Add Config for snake_case
        lines.append('')
        lines.append('    class Config:')
        lines.append('        populate_by_name = True')

        return "\n".join(lines)

    def _get_python_type(
        self,
        schema: Dict[str, Any],
        all_schemas: Dict[str, Any],
    ) -> str:
        """Convert OpenAPI type to Python type annotation."""
        # Handle $ref
        if "$ref" in schema:
            ref = schema["$ref"]
            type_name = ref.split("/")[-1]
            return type_name

        # Handle allOf, anyOf, oneOf
        if "allOf" in schema:
            types = [self._get_python_type(s, all_schemas) for s in schema["allOf"]]
            return types[0] if len(types) == 1 else f"Union[{', '.join(types)}]"

        if "anyOf" in schema or "oneOf" in schema:
            items = schema.get("anyOf") or schema.get("oneOf")
            types = [self._get_python_type(s, all_schemas) for s in items]
            # Filter out None
            types = [t for t in types if t != "None"]
            if len(types) == 1:
                return f"Optional[{types[0]}]"
            return f"Optional[Union[{', '.join(types)}]]"

        schema_type = schema.get("type", "any")
        schema_format = schema.get("format")

        # Check format first
        if schema_format in OPENAPI_FORMAT_TO_PYTHON:
            return OPENAPI_FORMAT_TO_PYTHON[schema_format]

        # Handle arrays
        if schema_type == "array":
            items_schema = schema.get("items", {})
            items_type = self._get_python_type(items_schema, all_schemas)
            return f"List[{items_type}]"

        # Handle objects with additionalProperties
        if schema_type == "object":
            if "additionalProperties" in schema:
                value_type = self._get_python_type(
                    schema["additionalProperties"], all_schemas
                )
                return f"Dict[str, {value_type}]"
            return "Dict[str, Any]"

        # Basic types
        return OPENAPI_TO_PYTHON_TYPES.get(schema_type, "Any")

    # =========================================================================
    # Client Generation
    # =========================================================================

    def _generate_client(self, spec: Dict[str, Any]) -> str:
        """Generate the SDK client."""
        lines = [
            '"""',
            f'{self.config.package_name.title()} SDK Client',
            '',
            'Auto-generated from OpenAPI specification.',
            '"""',
            '',
            'import asyncio',
            'import time',
            'from typing import Any, Dict, List, Optional, Union',
            '',
            'import httpx',
            '',
            'from .exceptions import (',
            '    APIError,',
            '    AuthenticationError,',
            '    NotFoundError,',
            '    RateLimitError,',
            '    ValidationError,',
            ')',
            'from .models import *',
            '',
            '',
            '# Default settings',
            'DEFAULT_BASE_URL = "https://api.greenlang.io"',
            'DEFAULT_TIMEOUT = 30.0',
            'DEFAULT_MAX_RETRIES = 3',
            '',
            '',
        ]

        # Generate base client
        lines.extend(self._generate_base_client())

        # Generate sync client
        if self.config.include_sync:
            lines.extend(self._generate_sync_client(spec))

        # Generate async client
        if self.config.include_async:
            lines.extend(self._generate_async_client(spec))

        return "\n".join(lines)

    def _generate_base_client(self) -> List[str]:
        """Generate base client class."""
        return [
            'class BaseClient:',
            '    """Base client with shared functionality."""',
            '',
            '    def __init__(',
            '        self,',
            '        api_key: Optional[str] = None,',
            '        base_url: str = DEFAULT_BASE_URL,',
            '        timeout: float = DEFAULT_TIMEOUT,',
            '        max_retries: int = DEFAULT_MAX_RETRIES,',
            '    ):',
            '        """',
            '        Initialize the client.',
            '',
            '        Args:',
            '            api_key: GreenLang API key',
            '            base_url: API base URL',
            '            timeout: Request timeout in seconds',
            '            max_retries: Maximum retry attempts for rate limits',
            '        """',
            '        self.api_key = api_key',
            '        self.base_url = base_url.rstrip("/")',
            '        self.timeout = timeout',
            '        self.max_retries = max_retries',
            '',
            '    def _get_headers(self) -> Dict[str, str]:',
            '        """Get request headers."""',
            '        headers = {',
            '            "Content-Type": "application/json",',
            '            "Accept": "application/json",',
            '            "User-Agent": "GreenLang-Python-SDK/1.0.0",',
            '        }',
            '        if self.api_key:',
            '            headers["X-API-Key"] = self.api_key',
            '        return headers',
            '',
            '    def _handle_response(self, response: httpx.Response) -> Any:',
            '        """Handle API response."""',
            '        if response.status_code == 429:',
            '            retry_after = int(response.headers.get("Retry-After", 60))',
            '            raise RateLimitError(',
            '                f"Rate limit exceeded. Retry after {retry_after}s",',
            '                retry_after=retry_after,',
            '            )',
            '',
            '        if response.status_code == 401:',
            '            raise AuthenticationError("Invalid or missing API key")',
            '',
            '        if response.status_code == 404:',
            '            raise NotFoundError("Resource not found")',
            '',
            '        if response.status_code == 400:',
            '            try:',
            '                error_data = response.json()',
            '                raise ValidationError(',
            '                    error_data.get("error", {}).get("message", "Validation error"),',
            '                    details=error_data.get("error", {}).get("details"),',
            '                )',
            '            except ValueError:',
            '                raise ValidationError("Validation error")',
            '',
            '        if response.status_code >= 400:',
            '            try:',
            '                error_data = response.json()',
            '                raise APIError(',
            '                    error_data.get("error", {}).get("message", "API error"),',
            '                    status_code=response.status_code,',
            '                )',
            '            except ValueError:',
            '                raise APIError(response.text, status_code=response.status_code)',
            '',
            '        if response.status_code == 204:',
            '            return None',
            '',
            '        return response.json()',
            '',
            '',
        ]

    def _generate_sync_client(self, spec: Dict[str, Any]) -> List[str]:
        """Generate synchronous client."""
        lines = [
            'class GreenLangClient(BaseClient):',
            '    """',
            '    Synchronous GreenLang API client.',
            '',
            '    Example:',
            '        >>> client = GreenLangClient(api_key="gl_your_key")',
            '        >>> agents = client.list_agents()',
            '        >>> for agent in agents.data:',
            '        ...     print(agent.name)',
            '    """',
            '',
            '    def __init__(self, *args, **kwargs):',
            '        super().__init__(*args, **kwargs)',
            '        self._client = httpx.Client(',
            '            base_url=self.base_url,',
            '            headers=self._get_headers(),',
            '            timeout=self.timeout,',
            '        )',
            '',
            '    def __enter__(self):',
            '        return self',
            '',
            '    def __exit__(self, *args):',
            '        self.close()',
            '',
            '    def close(self):',
            '        """Close the client."""',
            '        self._client.close()',
            '',
            '    def _request_with_retry(',
            '        self,',
            '        method: str,',
            '        path: str,',
            '        **kwargs,',
            '    ) -> Any:',
            '        """Make request with retry logic."""',
            '        for attempt in range(self.max_retries):',
            '            try:',
            '                response = self._client.request(method, path, **kwargs)',
            '                return self._handle_response(response)',
            '            except RateLimitError as e:',
            '                if attempt < self.max_retries - 1:',
            '                    time.sleep(e.retry_after)',
            '                else:',
            '                    raise',
            '',
        ]

        # Generate methods for each endpoint
        lines.extend(self._generate_client_methods(spec, is_async=False))

        lines.append('')
        return lines

    def _generate_async_client(self, spec: Dict[str, Any]) -> List[str]:
        """Generate asynchronous client."""
        lines = [
            'class GreenLangAsyncClient(BaseClient):',
            '    """',
            '    Asynchronous GreenLang API client.',
            '',
            '    Example:',
            '        >>> async with GreenLangAsyncClient(api_key="gl_your_key") as client:',
            '        ...     agents = await client.list_agents()',
            '        ...     for agent in agents.data:',
            '        ...         print(agent.name)',
            '    """',
            '',
            '    def __init__(self, *args, **kwargs):',
            '        super().__init__(*args, **kwargs)',
            '        self._client: Optional[httpx.AsyncClient] = None',
            '',
            '    async def __aenter__(self):',
            '        self._client = httpx.AsyncClient(',
            '            base_url=self.base_url,',
            '            headers=self._get_headers(),',
            '            timeout=self.timeout,',
            '        )',
            '        return self',
            '',
            '    async def __aexit__(self, *args):',
            '        await self.close()',
            '',
            '    async def close(self):',
            '        """Close the client."""',
            '        if self._client:',
            '            await self._client.aclose()',
            '',
            '    async def _request_with_retry(',
            '        self,',
            '        method: str,',
            '        path: str,',
            '        **kwargs,',
            '    ) -> Any:',
            '        """Make request with retry logic."""',
            '        if not self._client:',
            '            raise RuntimeError("Client not initialized. Use async with.")',
            '',
            '        for attempt in range(self.max_retries):',
            '            try:',
            '                response = await self._client.request(method, path, **kwargs)',
            '                return self._handle_response(response)',
            '            except RateLimitError as e:',
            '                if attempt < self.max_retries - 1:',
            '                    await asyncio.sleep(e.retry_after)',
            '                else:',
            '                    raise',
            '',
        ]

        # Generate methods for each endpoint
        lines.extend(self._generate_client_methods(spec, is_async=True))

        lines.append('')
        return lines

    def _generate_client_methods(
        self,
        spec: Dict[str, Any],
        is_async: bool,
    ) -> List[str]:
        """Generate client methods for each endpoint."""
        lines = []
        paths = spec.get("paths", {})

        for path, operations in paths.items():
            for method, operation in operations.items():
                if method not in ["get", "post", "put", "patch", "delete"]:
                    continue

                if not isinstance(operation, dict):
                    continue

                method_lines = self._generate_method(
                    path, method, operation, is_async
                )
                lines.extend(method_lines)
                lines.append('')

        return lines

    def _generate_method(
        self,
        path: str,
        http_method: str,
        operation: Dict[str, Any],
        is_async: bool,
    ) -> List[str]:
        """Generate a single client method."""
        # Generate method name from operationId or path
        operation_id = operation.get("operationId")
        if operation_id:
            method_name = self._to_snake_case(operation_id)
        else:
            method_name = self._path_to_method_name(path, http_method)

        # Get parameters
        parameters = operation.get("parameters", [])
        path_params = [p for p in parameters if p.get("in") == "path"]
        query_params = [p for p in parameters if p.get("in") == "query"]

        # Build method signature
        prefix = "async " if is_async else ""
        params = ["self"]

        for param in path_params:
            param_name = param["name"]
            param_type = self._get_python_type(param.get("schema", {}), {})
            params.append(f"{param_name}: {param_type}")

        # Request body
        has_body = "requestBody" in operation
        if has_body:
            body_schema = operation["requestBody"].get("content", {}).get(
                "application/json", {}
            ).get("schema", {})
            body_type = self._get_python_type(body_schema, {})
            if "$ref" in body_schema:
                body_type = body_schema["$ref"].split("/")[-1]
            params.append(f"body: {body_type}")

        # Query parameters
        for param in query_params:
            param_name = param["name"]
            param_type = self._get_python_type(param.get("schema", {}), {})
            default = " = None" if not param.get("required") else ""
            if not param.get("required") and "Optional" not in param_type:
                param_type = f"Optional[{param_type}]"
            params.append(f"{param_name}: {param_type}{default}")

        signature = ", ".join(params)

        # Get return type
        responses = operation.get("responses", {})
        success_response = responses.get("200") or responses.get("201") or responses.get("202")
        return_type = "Any"
        if success_response:
            content = success_response.get("content", {}).get("application/json", {})
            if "schema" in content:
                return_type = self._get_python_type(content["schema"], {})
                if "$ref" in content["schema"]:
                    return_type = content["schema"]["$ref"].split("/")[-1]

        # Build method
        await_prefix = "await " if is_async else ""
        description = operation.get("summary", operation.get("description", ""))

        lines = [
            f'    {prefix}def {method_name}({signature}) -> {return_type}:',
            f'        """',
            f'        {description}',
            f'        """',
        ]

        # Build path with parameters
        path_code = f'"{path}"'
        for param in path_params:
            path_code = path_code.replace(
                "{" + param["name"] + "}",
                f'{{{param["name"]}}}'
            )
        if path_params:
            path_code = f'f{path_code}'

        # Build query params
        if query_params:
            lines.append('        params = {}')
            for param in query_params:
                param_name = param["name"]
                lines.append(f'        if {param_name} is not None:')
                lines.append(f'            params["{param_name}"] = {param_name}')
        else:
            lines.append('        params = None')

        # Make request
        if has_body:
            json_arg = 'json=body.dict() if hasattr(body, "dict") else body'
        else:
            json_arg = 'json=None'

        lines.append(
            f'        return {await_prefix}self._request_with_retry('
        )
        lines.append(f'            "{http_method.upper()}",')
        lines.append(f'            {path_code},')
        lines.append(f'            params=params,')
        lines.append(f'            {json_arg},')
        lines.append('        )')

        return lines

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _to_snake_case(self, name: str) -> str:
        """Convert camelCase to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _path_to_method_name(self, path: str, method: str) -> str:
        """Convert path to method name."""
        # Remove path parameters
        clean = re.sub(r'\{[^}]+\}', '', path)
        # Remove leading/trailing slashes
        clean = clean.strip('/')
        # Replace slashes with underscores
        clean = clean.replace('/', '_')
        # Add method prefix
        prefix_map = {
            'get': 'get' if '{' in path else 'list',
            'post': 'create',
            'put': 'update',
            'patch': 'update',
            'delete': 'delete',
        }
        prefix = prefix_map.get(method, method)
        return f"{prefix}_{clean}"

    # =========================================================================
    # Other File Generation
    # =========================================================================

    def _generate_exceptions(self) -> str:
        """Generate exception classes."""
        return '''"""
GreenLang SDK Exceptions

Custom exception classes for error handling.
"""


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AuthenticationError(APIError):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class RateLimitError(APIError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ValidationError(APIError):
    """Request validation failed."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=400)
        self.details = details


class NotFoundError(APIError):
    """Resource not found."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)
'''

    def _generate_init(self) -> str:
        """Generate __init__.py."""
        return f'''"""
{self.config.package_name.title()} SDK

Python SDK for the GreenLang API.

Example:
    >>> from {self.config.package_name} import GreenLangClient
    >>> client = GreenLangClient(api_key="gl_your_key")
    >>> agents = client.list_agents()
"""

from .client import GreenLangClient, GreenLangAsyncClient
from .exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NotFoundError,
)
from .models import *

__version__ = "{self.config.version}"
__author__ = "{self.config.author}"

__all__ = [
    "GreenLangClient",
    "GreenLangAsyncClient",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "NotFoundError",
]
'''

    def _generate_setup(self) -> str:
        """Generate setup.py."""
        return f'''"""
Setup script for {self.config.package_name} SDK.
"""

from setuptools import setup, find_packages

setup(
    name="{self.config.package_name}",
    version="{self.config.version}",
    author="{self.config.author}",
    author_email="{self.config.author_email}",
    description="{self.config.description}",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/greenlang/greenlang-python",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
    ],
    extras_require={{
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "respx>=0.20.0",
        ],
    }},
)
'''

    def _generate_readme(self, spec: Dict[str, Any]) -> str:
        """Generate README.md."""
        info = spec.get("info", {})
        title = info.get("title", "GreenLang SDK")

        return f'''# {title}

{self.config.description}

## Installation

```bash
pip install {self.config.package_name}
```

## Quick Start

### Synchronous Client

```python
from {self.config.package_name} import GreenLangClient

# Initialize client
client = GreenLangClient(api_key="gl_your_api_key")

# List agents
agents = client.list_agents()
for agent in agents.data:
    print(f"{{agent.agent_id}}: {{agent.name}}")

# Execute an agent
result = client.execute_agent(
    agent_id="carbon/calculator",
    body={{
        "inputs": {{
            "activity_type": "electricity",
            "quantity": 1000,
            "unit": "kWh"
        }}
    }}
)
print(f"Carbon footprint: {{result.outputs['carbon_footprint']}} tCO2e")
```

### Async Client

```python
import asyncio
from {self.config.package_name} import GreenLangAsyncClient

async def main():
    async with GreenLangAsyncClient(api_key="gl_your_api_key") as client:
        agents = await client.list_agents()
        for agent in agents.data:
            print(f"{{agent.agent_id}}: {{agent.name}}")

asyncio.run(main())
```

## Error Handling

```python
from {self.config.package_name} import GreenLangClient, RateLimitError, APIError

client = GreenLangClient(api_key="gl_your_api_key")

try:
    result = client.execute_agent("carbon/calculator", body={{...}})
except RateLimitError as e:
    print(f"Rate limited. Retry after {{e.retry_after}} seconds")
except APIError as e:
    print(f"API error: {{e.message}}")
```

## License

MIT License - see LICENSE file for details.
'''

    def _generate_tests(self, spec: Dict[str, Any]) -> str:
        """Generate test file."""
        return f'''"""
Tests for {self.config.package_name} SDK.
"""

import pytest
import httpx
import respx
from {self.config.package_name} import (
    GreenLangClient,
    GreenLangAsyncClient,
    APIError,
    RateLimitError,
)


@pytest.fixture
def client():
    """Create test client."""
    return GreenLangClient(
        api_key="test_key",
        base_url="https://api.test.greenlang.io"
    )


@respx.mock
def test_list_agents(client):
    """Test listing agents."""
    respx.get("https://api.test.greenlang.io/v1/agents").mock(
        return_value=httpx.Response(200, json={{
            "data": [
                {{"id": "1", "agent_id": "carbon/calc", "name": "Calculator"}}
            ],
            "meta": {{"total": 1}}
        }})
    )

    result = client.list_agents()
    assert len(result["data"]) == 1
    assert result["data"][0]["agent_id"] == "carbon/calc"


@respx.mock
def test_rate_limit_handling(client):
    """Test rate limit handling."""
    respx.get("https://api.test.greenlang.io/v1/agents").mock(
        return_value=httpx.Response(
            429,
            json={{"error": {{"message": "Rate limited"}}}},
            headers={{"Retry-After": "60"}}
        )
    )

    with pytest.raises(RateLimitError) as exc_info:
        client.list_agents()

    assert exc_info.value.retry_after == 60


@pytest.mark.asyncio
async def test_async_client():
    """Test async client."""
    async with GreenLangAsyncClient(
        api_key="test_key",
        base_url="https://api.test.greenlang.io"
    ) as client:
        assert client._client is not None
'''


# =============================================================================
# Factory Function
# =============================================================================


def generate_python_sdk(
    openapi_spec: Dict[str, Any],
    config: Optional[SDKConfig] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate Python SDK from OpenAPI specification.

    Args:
        openapi_spec: OpenAPI specification dictionary
        config: Optional SDK configuration
        output_dir: Optional output directory (overrides config)

    Returns:
        Dictionary mapping file paths to content

    Example:
        >>> import json
        >>> with open("openapi.json") as f:
        ...     spec = json.load(f)
        >>> files = generate_python_sdk(spec)
        >>> for path, content in files.items():
        ...     print(f"Generated: {path}")
    """
    config = config or SDKConfig()
    if output_dir:
        config.output_dir = output_dir

    generator = SDKGenerator(config)
    return generator.generate_from_spec(openapi_spec)
