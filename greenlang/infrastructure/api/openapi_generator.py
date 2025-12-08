"""
OpenAPI 3.0 Specification Generator for GreenLang

This module provides automatic OpenAPI specification generation
from GreenLang agent schemas and service definitions with full
FastAPI integration for route auto-discovery.

Features:
- OpenAPI 3.0/3.1 support
- Automatic schema generation from Pydantic models
- FastAPI route auto-discovery
- Security scheme definitions (API Key, OAuth2, JWT)
- Tag grouping with descriptions
- Server configuration
- Extension support
- Export to JSON and YAML formats

Example:
    >>> # Manual schema registration
    >>> generator = OpenAPIGenerator(config)
    >>> generator.add_schema(EmissionModel)
    >>> spec = generator.generate()
    >>>
    >>> # FastAPI auto-discovery
    >>> from fastapi import FastAPI
    >>> app = FastAPI()
    >>> generator = OpenAPIGenerator.from_fastapi_app(app)
    >>> spec = generator.generate()
"""

import hashlib
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, get_type_hints
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from fastapi import FastAPI, APIRouter
    from fastapi.routing import APIRoute
    from starlette.routing import Route
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    APIRouter = None
    APIRoute = None
    Route = None

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


class OpenAPIVersion(str, Enum):
    """OpenAPI specification versions."""
    V3_0_3 = "3.0.3"
    V3_1_0 = "3.1.0"


class SecuritySchemeType(str, Enum):
    """Security scheme types."""
    API_KEY = "apiKey"
    HTTP = "http"
    OAUTH2 = "oauth2"
    OPENID_CONNECT = "openIdConnect"


class HTTPAuthScheme(str, Enum):
    """HTTP authentication schemes."""
    BASIC = "basic"
    BEARER = "bearer"
    DIGEST = "digest"


class ParameterLocation(str, Enum):
    """API parameter locations."""
    QUERY = "query"
    HEADER = "header"
    PATH = "path"
    COOKIE = "cookie"


@dataclass
class OpenAPIGeneratorConfig:
    """Configuration for OpenAPI generator."""
    title: str = "GreenLang API"
    description: str = "GreenLang Enterprise Sustainability Platform API"
    version: str = "1.0.0"
    openapi_version: OpenAPIVersion = OpenAPIVersion.V3_0_3
    terms_of_service: Optional[str] = None
    contact_name: Optional[str] = "GreenLang Support"
    contact_email: Optional[str] = "support@greenlang.io"
    contact_url: Optional[str] = "https://greenlang.io/support"
    license_name: str = "Proprietary"
    license_url: Optional[str] = None
    servers: List[Dict[str, str]] = field(default_factory=list)
    external_docs_url: Optional[str] = None
    external_docs_description: Optional[str] = None


class SchemaDefinition(BaseModel):
    """Schema definition for OpenAPI."""
    type: str = Field(default="object", description="Schema type")
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    description: Optional[str] = Field(default=None)
    example: Optional[Any] = Field(default=None)
    enum: Optional[List[Any]] = Field(default=None)
    items: Optional[Dict[str, Any]] = Field(default=None)
    format: Optional[str] = Field(default=None)
    minimum: Optional[float] = Field(default=None)
    maximum: Optional[float] = Field(default=None)
    pattern: Optional[str] = Field(default=None)


class OperationDefinition(BaseModel):
    """API operation definition."""
    operation_id: str = Field(..., description="Unique operation ID")
    summary: str = Field(..., description="Operation summary")
    description: Optional[str] = Field(default=None)
    tags: List[str] = Field(default_factory=list)
    parameters: List[Dict[str, Any]] = Field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = Field(default=None)
    responses: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    security: List[Dict[str, List[str]]] = Field(default_factory=list)
    deprecated: bool = Field(default=False)


class PathDefinition(BaseModel):
    """API path definition."""
    path: str = Field(..., description="Path template")
    operations: Dict[str, OperationDefinition] = Field(default_factory=dict)


class SecurityScheme(BaseModel):
    """Security scheme definition."""
    type: SecuritySchemeType = Field(..., description="Scheme type")
    name: str = Field(..., description="Scheme name")
    description: Optional[str] = Field(default=None)
    # API Key specific
    api_key_name: Optional[str] = Field(default=None)
    api_key_in: Optional[ParameterLocation] = Field(default=None)
    # HTTP specific
    http_scheme: Optional[HTTPAuthScheme] = Field(default=None)
    bearer_format: Optional[str] = Field(default=None)
    # OAuth2 specific
    oauth2_flows: Optional[Dict[str, Any]] = Field(default=None)
    # OpenID Connect specific
    openid_connect_url: Optional[str] = Field(default=None)


class OpenAPIGenerator:
    """
    OpenAPI specification generator.

    Generates OpenAPI 3.0/3.1 specifications from GreenLang
    service definitions and Pydantic models.

    Attributes:
        config: Generator configuration
        schemas: Registered schemas
        paths: Registered paths

    Example:
        >>> config = OpenAPIGeneratorConfig(
        ...     title="Emissions API",
        ...     version="2.0.0"
        ... )
        >>> generator = OpenAPIGenerator(config)
        >>> generator.add_pydantic_schema(EmissionReport)
        >>> generator.add_path("/emissions/{id}", get_emission)
        >>> spec = generator.generate()
    """

    def __init__(self, config: OpenAPIGeneratorConfig):
        """
        Initialize OpenAPI generator.

        Args:
            config: Generator configuration
        """
        self.config = config
        self._schemas: Dict[str, SchemaDefinition] = {}
        self._paths: Dict[str, PathDefinition] = {}
        self._security_schemes: Dict[str, SecurityScheme] = {}
        self._tags: List[Dict[str, str]] = []

        # Add default servers if not configured
        if not config.servers:
            config.servers = [
                {"url": "https://api.greenlang.io/v1", "description": "Production"},
                {"url": "https://staging-api.greenlang.io/v1", "description": "Staging"},
            ]

        logger.info(f"OpenAPIGenerator initialized for {config.title}")

    def add_pydantic_schema(
        self,
        model: Type[BaseModel],
        name: Optional[str] = None
    ) -> str:
        """
        Add a Pydantic model as a schema.

        Args:
            model: Pydantic model class
            name: Optional schema name

        Returns:
            Schema name
        """
        schema_name = name or model.__name__

        # Generate schema from Pydantic model
        schema_dict = model.schema()

        # Convert to SchemaDefinition
        schema_def = self._convert_pydantic_schema(schema_dict)
        self._schemas[schema_name] = schema_def

        # Handle nested definitions
        if "definitions" in schema_dict:
            for def_name, def_schema in schema_dict["definitions"].items():
                nested_def = self._convert_pydantic_schema(def_schema)
                self._schemas[def_name] = nested_def

        logger.debug(f"Added schema: {schema_name}")
        return schema_name

    def _convert_pydantic_schema(self, schema: Dict[str, Any]) -> SchemaDefinition:
        """Convert Pydantic schema to OpenAPI schema."""
        properties = {}
        for prop_name, prop_def in schema.get("properties", {}).items():
            properties[prop_name] = self._convert_property(prop_def)

        return SchemaDefinition(
            type=schema.get("type", "object"),
            properties=properties,
            required=schema.get("required", []),
            description=schema.get("description"),
            example=schema.get("example"),
        )

    def _convert_property(self, prop: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a property definition."""
        result = {}

        # Handle $ref
        if "$ref" in prop:
            ref = prop["$ref"]
            if ref.startswith("#/definitions/"):
                ref = ref.replace("#/definitions/", "#/components/schemas/")
            result["$ref"] = ref
            return result

        # Handle allOf
        if "allOf" in prop:
            return {"allOf": [self._convert_property(p) for p in prop["allOf"]]}

        # Handle anyOf
        if "anyOf" in prop:
            return {"anyOf": [self._convert_property(p) for p in prop["anyOf"]]}

        # Basic type mapping
        prop_type = prop.get("type")
        if prop_type:
            result["type"] = prop_type

        # Format
        if "format" in prop:
            result["format"] = prop["format"]

        # Description
        if "description" in prop:
            result["description"] = prop["description"]

        # Enum
        if "enum" in prop:
            result["enum"] = prop["enum"]

        # Array items
        if prop_type == "array" and "items" in prop:
            result["items"] = self._convert_property(prop["items"])

        # Numeric constraints
        for constraint in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]:
            if constraint in prop:
                result[constraint] = prop[constraint]

        # String constraints
        for constraint in ["minLength", "maxLength", "pattern"]:
            if constraint in prop:
                result[constraint] = prop[constraint]

        # Default value
        if "default" in prop:
            result["default"] = prop["default"]

        # Example
        if "example" in prop:
            result["example"] = prop["example"]

        return result

    def add_schema(
        self,
        name: str,
        schema: SchemaDefinition
    ) -> None:
        """
        Add a schema definition directly.

        Args:
            name: Schema name
            schema: Schema definition
        """
        self._schemas[name] = schema
        logger.debug(f"Added schema: {name}")

    def add_path(
        self,
        path: str,
        method: str,
        operation: OperationDefinition
    ) -> None:
        """
        Add a path operation.

        Args:
            path: Path template
            method: HTTP method
            operation: Operation definition
        """
        if path not in self._paths:
            self._paths[path] = PathDefinition(path=path)

        self._paths[path].operations[method.lower()] = operation
        logger.debug(f"Added path: {method.upper()} {path}")

    def add_crud_paths(
        self,
        resource: str,
        schema_name: str,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Add standard CRUD paths for a resource.

        Args:
            resource: Resource name (plural)
            schema_name: Schema name for the resource
            tags: Operation tags
        """
        resource_tags = tags or [resource.title()]

        # List
        self.add_path(
            f"/{resource}",
            "get",
            OperationDefinition(
                operation_id=f"list_{resource}",
                summary=f"List {resource}",
                tags=resource_tags,
                parameters=[
                    {
                        "name": "page",
                        "in": "query",
                        "schema": {"type": "integer", "default": 1}
                    },
                    {
                        "name": "page_size",
                        "in": "query",
                        "schema": {"type": "integer", "default": 20}
                    },
                ],
                responses={
                    "200": {
                        "description": f"List of {resource}",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": f"#/components/schemas/{schema_name}"}
                                }
                            }
                        }
                    }
                }
            )
        )

        # Create
        self.add_path(
            f"/{resource}",
            "post",
            OperationDefinition(
                operation_id=f"create_{resource[:-1]}",
                summary=f"Create {resource[:-1]}",
                tags=resource_tags,
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": f"#/components/schemas/{schema_name}"}
                        }
                    }
                },
                responses={
                    "201": {
                        "description": f"Created {resource[:-1]}",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": f"#/components/schemas/{schema_name}"}
                            }
                        }
                    }
                }
            )
        )

        # Get by ID
        self.add_path(
            f"/{resource}/{{id}}",
            "get",
            OperationDefinition(
                operation_id=f"get_{resource[:-1]}",
                summary=f"Get {resource[:-1]} by ID",
                tags=resource_tags,
                parameters=[
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                responses={
                    "200": {
                        "description": f"{resource[:-1]} details",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": f"#/components/schemas/{schema_name}"}
                            }
                        }
                    },
                    "404": {"description": "Not found"}
                }
            )
        )

        # Update
        self.add_path(
            f"/{resource}/{{id}}",
            "put",
            OperationDefinition(
                operation_id=f"update_{resource[:-1]}",
                summary=f"Update {resource[:-1]}",
                tags=resource_tags,
                parameters=[
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": f"#/components/schemas/{schema_name}"}
                        }
                    }
                },
                responses={
                    "200": {
                        "description": f"Updated {resource[:-1]}",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": f"#/components/schemas/{schema_name}"}
                            }
                        }
                    }
                }
            )
        )

        # Delete
        self.add_path(
            f"/{resource}/{{id}}",
            "delete",
            OperationDefinition(
                operation_id=f"delete_{resource[:-1]}",
                summary=f"Delete {resource[:-1]}",
                tags=resource_tags,
                parameters=[
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }
                ],
                responses={
                    "204": {"description": "Deleted successfully"},
                    "404": {"description": "Not found"}
                }
            )
        )

        logger.info(f"Added CRUD paths for resource: {resource}")

    def add_security_scheme(self, scheme: SecurityScheme) -> None:
        """
        Add a security scheme.

        Args:
            scheme: Security scheme definition
        """
        self._security_schemes[scheme.name] = scheme
        logger.debug(f"Added security scheme: {scheme.name}")

    def add_bearer_auth(
        self,
        name: str = "bearerAuth",
        description: str = "JWT Bearer token authentication"
    ) -> None:
        """Add standard Bearer token authentication."""
        self.add_security_scheme(SecurityScheme(
            type=SecuritySchemeType.HTTP,
            name=name,
            description=description,
            http_scheme=HTTPAuthScheme.BEARER,
            bearer_format="JWT",
        ))

    def add_api_key_auth(
        self,
        name: str = "apiKeyAuth",
        key_name: str = "X-API-Key",
        location: ParameterLocation = ParameterLocation.HEADER
    ) -> None:
        """Add API key authentication."""
        self.add_security_scheme(SecurityScheme(
            type=SecuritySchemeType.API_KEY,
            name=name,
            description="API key authentication",
            api_key_name=key_name,
            api_key_in=location,
        ))

    def add_oauth2(
        self,
        name: str = "oauth2",
        authorization_url: str = "",
        token_url: str = "",
        scopes: Optional[Dict[str, str]] = None
    ) -> None:
        """Add OAuth2 authentication."""
        self.add_security_scheme(SecurityScheme(
            type=SecuritySchemeType.OAUTH2,
            name=name,
            description="OAuth2 authentication",
            oauth2_flows={
                "authorizationCode": {
                    "authorizationUrl": authorization_url,
                    "tokenUrl": token_url,
                    "scopes": scopes or {}
                }
            },
        ))

    def add_tag(
        self,
        name: str,
        description: Optional[str] = None,
        external_docs_url: Optional[str] = None
    ) -> None:
        """
        Add a tag definition.

        Args:
            name: Tag name
            description: Tag description
            external_docs_url: External documentation URL
        """
        tag = {"name": name}
        if description:
            tag["description"] = description
        if external_docs_url:
            tag["externalDocs"] = {"url": external_docs_url}

        self._tags.append(tag)

    def generate(self) -> Dict[str, Any]:
        """
        Generate the OpenAPI specification.

        Returns:
            Complete OpenAPI specification as dictionary
        """
        spec = {
            "openapi": self.config.openapi_version.value,
            "info": self._generate_info(),
            "servers": self.config.servers,
        }

        # External docs
        if self.config.external_docs_url:
            spec["externalDocs"] = {
                "url": self.config.external_docs_url,
                "description": self.config.external_docs_description or "Documentation"
            }

        # Tags
        if self._tags:
            spec["tags"] = self._tags

        # Paths
        spec["paths"] = self._generate_paths()

        # Components
        spec["components"] = self._generate_components()

        # Security
        if self._security_schemes:
            spec["security"] = [
                {name: []} for name in self._security_schemes.keys()
            ]

        logger.info(
            f"Generated OpenAPI spec: {len(self._paths)} paths, "
            f"{len(self._schemas)} schemas"
        )

        return spec

    def _generate_info(self) -> Dict[str, Any]:
        """Generate info section."""
        info = {
            "title": self.config.title,
            "description": self.config.description,
            "version": self.config.version,
        }

        if self.config.terms_of_service:
            info["termsOfService"] = self.config.terms_of_service

        if self.config.contact_name or self.config.contact_email:
            info["contact"] = {}
            if self.config.contact_name:
                info["contact"]["name"] = self.config.contact_name
            if self.config.contact_email:
                info["contact"]["email"] = self.config.contact_email
            if self.config.contact_url:
                info["contact"]["url"] = self.config.contact_url

        info["license"] = {"name": self.config.license_name}
        if self.config.license_url:
            info["license"]["url"] = self.config.license_url

        return info

    def _generate_paths(self) -> Dict[str, Any]:
        """Generate paths section."""
        paths = {}

        for path, path_def in self._paths.items():
            paths[path] = {}
            for method, operation in path_def.operations.items():
                op_dict = {
                    "operationId": operation.operation_id,
                    "summary": operation.summary,
                    "responses": operation.responses,
                }

                if operation.description:
                    op_dict["description"] = operation.description

                if operation.tags:
                    op_dict["tags"] = operation.tags

                if operation.parameters:
                    op_dict["parameters"] = operation.parameters

                if operation.request_body:
                    op_dict["requestBody"] = operation.request_body

                if operation.security:
                    op_dict["security"] = operation.security

                if operation.deprecated:
                    op_dict["deprecated"] = True

                paths[path][method] = op_dict

        return paths

    def _generate_components(self) -> Dict[str, Any]:
        """Generate components section."""
        components = {}

        # Schemas
        if self._schemas:
            components["schemas"] = {}
            for name, schema in self._schemas.items():
                components["schemas"][name] = schema.dict(exclude_none=True)

        # Security schemes
        if self._security_schemes:
            components["securitySchemes"] = {}
            for name, scheme in self._security_schemes.items():
                scheme_dict = {"type": scheme.type.value}

                if scheme.description:
                    scheme_dict["description"] = scheme.description

                if scheme.type == SecuritySchemeType.API_KEY:
                    scheme_dict["name"] = scheme.api_key_name
                    scheme_dict["in"] = scheme.api_key_in.value

                elif scheme.type == SecuritySchemeType.HTTP:
                    scheme_dict["scheme"] = scheme.http_scheme.value
                    if scheme.bearer_format:
                        scheme_dict["bearerFormat"] = scheme.bearer_format

                elif scheme.type == SecuritySchemeType.OAUTH2:
                    scheme_dict["flows"] = scheme.oauth2_flows

                elif scheme.type == SecuritySchemeType.OPENID_CONNECT:
                    scheme_dict["openIdConnectUrl"] = scheme.openid_connect_url

                components["securitySchemes"][name] = scheme_dict

        return components

    def to_json(self, indent: int = 2) -> str:
        """
        Generate specification as JSON string.

        Args:
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(self.generate(), indent=indent)

    def to_yaml(self) -> str:
        """
        Generate specification as YAML string.

        Returns:
            YAML string

        Raises:
            ImportError: If PyYAML is not installed
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML output. Install with: pip install pyyaml")
        return yaml.dump(self.generate(), default_flow_style=False, sort_keys=False)

    def save_to_file(
        self,
        file_path: str,
        format: str = "auto"
    ) -> str:
        """
        Save the OpenAPI specification to a file.

        Args:
            file_path: Path to save the specification
            format: Output format ('json', 'yaml', or 'auto' to detect from extension)

        Returns:
            Absolute path to the saved file

        Raises:
            ValueError: If format cannot be determined
            ImportError: If YAML format requested but PyYAML not installed
        """
        path = Path(file_path)

        # Determine format
        if format == "auto":
            ext = path.suffix.lower()
            if ext in [".json"]:
                format = "json"
            elif ext in [".yaml", ".yml"]:
                format = "yaml"
            else:
                raise ValueError(
                    f"Cannot determine format from extension '{ext}'. "
                    "Use format='json' or format='yaml'"
                )

        # Generate content
        if format == "json":
            content = self.to_json(indent=2)
        elif format == "yaml":
            content = self.to_yaml()
        else:
            raise ValueError(f"Unknown format: {format}")

        # Write file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        logger.info(f"OpenAPI specification saved to: {path.absolute()}")
        return str(path.absolute())

    # =========================================================================
    # FASTAPI AUTO-DISCOVERY
    # =========================================================================

    @classmethod
    def from_fastapi_app(
        cls,
        app: "FastAPI",
        config: Optional[OpenAPIGeneratorConfig] = None,
        include_security_schemes: bool = True
    ) -> "OpenAPIGenerator":
        """
        Create an OpenAPIGenerator from a FastAPI application.

        Auto-discovers all routes, schemas, and security schemes from
        the FastAPI app and its included routers.

        Args:
            app: FastAPI application instance
            config: Optional generator configuration (defaults to app settings)
            include_security_schemes: Whether to include security schemes

        Returns:
            Configured OpenAPIGenerator instance

        Raises:
            ImportError: If FastAPI is not available

        Example:
            >>> from fastapi import FastAPI
            >>> app = FastAPI(title="My API", version="1.0.0")
            >>> generator = OpenAPIGenerator.from_fastapi_app(app)
            >>> spec = generator.generate()
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for auto-discovery. "
                "Install with: pip install fastapi"
            )

        # Create config from app settings if not provided
        if config is None:
            config = OpenAPIGeneratorConfig(
                title=app.title or "GreenLang API",
                description=app.description or "",
                version=app.version or "1.0.0",
            )

        generator = cls(config)

        # Extract routes from FastAPI app
        generator._extract_fastapi_routes(app)

        # Extract schemas from route models
        generator._extract_fastapi_schemas(app)

        # Add default security schemes
        if include_security_schemes:
            generator._add_default_security_schemes()

        # Extract tags from app
        generator._extract_fastapi_tags(app)

        logger.info(
            f"OpenAPIGenerator created from FastAPI app: "
            f"{len(generator._paths)} paths, {len(generator._schemas)} schemas"
        )

        return generator

    def _extract_fastapi_routes(self, app: "FastAPI") -> None:
        """
        Extract all routes from a FastAPI application.

        Args:
            app: FastAPI application instance
        """
        for route in app.routes:
            if isinstance(route, APIRoute):
                self._process_api_route(route)

    def _process_api_route(self, route: "APIRoute") -> None:
        """
        Process a single APIRoute and add it to the generator.

        Args:
            route: FastAPI APIRoute instance
        """
        path = route.path
        methods = route.methods or {"GET"}

        for method in methods:
            method_lower = method.lower()
            if method_lower in ["head", "options"]:
                continue  # Skip HEAD and OPTIONS

            # Build operation definition
            operation = self._build_operation_from_route(route, method_lower)
            self.add_path(path, method_lower, operation)

    def _build_operation_from_route(
        self,
        route: "APIRoute",
        method: str
    ) -> OperationDefinition:
        """
        Build an OperationDefinition from an APIRoute.

        Args:
            route: FastAPI APIRoute instance
            method: HTTP method

        Returns:
            OperationDefinition for the route
        """
        # Get operation ID
        operation_id = route.operation_id
        if not operation_id:
            # Generate operation ID from path and method
            path_parts = route.path.strip("/").replace("/", "_").replace("{", "").replace("}", "")
            operation_id = f"{method}_{path_parts}" if path_parts else method

        # Get summary and description
        summary = route.summary or route.name or operation_id
        description = route.description

        # Get tags
        tags = list(route.tags) if route.tags else []

        # Build parameters
        parameters = self._extract_route_parameters(route)

        # Build request body
        request_body = self._extract_request_body(route) if method in ["post", "put", "patch"] else None

        # Build responses
        responses = self._extract_responses(route)

        # Check if deprecated
        deprecated = getattr(route, "deprecated", False)

        return OperationDefinition(
            operation_id=operation_id,
            summary=summary,
            description=description,
            tags=tags,
            parameters=parameters,
            request_body=request_body,
            responses=responses,
            deprecated=deprecated
        )

    def _extract_route_parameters(self, route: "APIRoute") -> List[Dict[str, Any]]:
        """
        Extract parameters from a route.

        Args:
            route: FastAPI APIRoute instance

        Returns:
            List of parameter definitions
        """
        parameters = []

        # Extract path parameters
        if route.dependant:
            for param in route.dependant.path_params:
                param_def = {
                    "name": param.name,
                    "in": "path",
                    "required": True,
                    "schema": self._get_param_schema(param)
                }
                if param.field_info and param.field_info.description:
                    param_def["description"] = param.field_info.description
                parameters.append(param_def)

            # Extract query parameters
            for param in route.dependant.query_params:
                param_def = {
                    "name": param.name,
                    "in": "query",
                    "required": param.required,
                    "schema": self._get_param_schema(param)
                }
                if param.field_info and param.field_info.description:
                    param_def["description"] = param.field_info.description
                if param.field_info and param.field_info.default is not None:
                    param_def["schema"]["default"] = param.field_info.default
                parameters.append(param_def)

            # Extract header parameters
            for param in route.dependant.header_params:
                param_def = {
                    "name": param.name,
                    "in": "header",
                    "required": param.required,
                    "schema": self._get_param_schema(param)
                }
                if param.field_info and param.field_info.description:
                    param_def["description"] = param.field_info.description
                parameters.append(param_def)

        return parameters

    def _get_param_schema(self, param: Any) -> Dict[str, Any]:
        """
        Get JSON schema for a parameter.

        Args:
            param: FastAPI parameter model

        Returns:
            JSON schema dictionary
        """
        type_annotation = param.type_

        # Map Python types to JSON schema types
        type_mapping = {
            int: {"type": "integer"},
            float: {"type": "number"},
            str: {"type": "string"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }

        # Check for basic types
        if type_annotation in type_mapping:
            return type_mapping[type_annotation]

        # Check for Optional types
        origin = getattr(type_annotation, "__origin__", None)
        if origin is Union:
            args = getattr(type_annotation, "__args__", ())
            non_none_args = [a for a in args if a is not type(None)]
            if non_none_args:
                return self._get_param_schema_from_type(non_none_args[0])

        # Check for Enum types
        if isinstance(type_annotation, type) and issubclass(type_annotation, Enum):
            return {
                "type": "string",
                "enum": [e.value for e in type_annotation]
            }

        return {"type": "string"}

    def _get_param_schema_from_type(self, type_annotation: Any) -> Dict[str, Any]:
        """Get schema from a type annotation."""
        type_mapping = {
            int: {"type": "integer"},
            float: {"type": "number"},
            str: {"type": "string"},
            bool: {"type": "boolean"},
        }
        return type_mapping.get(type_annotation, {"type": "string"})

    def _extract_request_body(self, route: "APIRoute") -> Optional[Dict[str, Any]]:
        """
        Extract request body schema from a route.

        Args:
            route: FastAPI APIRoute instance

        Returns:
            Request body definition or None
        """
        if not route.dependant or not route.dependant.body_params:
            return None

        # Get the body parameter (usually there's just one)
        body_params = route.dependant.body_params
        if not body_params:
            return None

        body_param = body_params[0]
        type_annotation = body_param.type_

        # Check if it's a Pydantic model
        if isinstance(type_annotation, type) and issubclass(type_annotation, BaseModel):
            schema_name = type_annotation.__name__
            # Add schema if not already added
            if schema_name not in self._schemas:
                self.add_pydantic_schema(type_annotation)

            return {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": f"#/components/schemas/{schema_name}"}
                    }
                }
            }

        return None

    def _extract_responses(self, route: "APIRoute") -> Dict[str, Dict[str, Any]]:
        """
        Extract response definitions from a route.

        Args:
            route: FastAPI APIRoute instance

        Returns:
            Response definitions dictionary
        """
        responses = {}

        # Get response model
        response_model = route.response_model
        status_code = route.status_code or 200

        if response_model:
            if isinstance(response_model, type) and issubclass(response_model, BaseModel):
                schema_name = response_model.__name__
                # Add schema if not already added
                if schema_name not in self._schemas:
                    self.add_pydantic_schema(response_model)

                responses[str(status_code)] = {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": f"#/components/schemas/{schema_name}"}
                        }
                    }
                }
            else:
                responses[str(status_code)] = {
                    "description": "Successful response"
                }
        else:
            responses[str(status_code)] = {
                "description": "Successful response"
            }

        # Add common error responses
        if route.responses:
            for code, response_info in route.responses.items():
                if isinstance(response_info, dict):
                    model = response_info.get("model")
                    if model and isinstance(model, type) and issubclass(model, BaseModel):
                        schema_name = model.__name__
                        if schema_name not in self._schemas:
                            self.add_pydantic_schema(model)
                        responses[str(code)] = {
                            "description": response_info.get("description", "Error response"),
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": f"#/components/schemas/{schema_name}"}
                                }
                            }
                        }
                    else:
                        responses[str(code)] = {
                            "description": response_info.get("description", "Error response")
                        }

        return responses

    def _extract_fastapi_schemas(self, app: "FastAPI") -> None:
        """
        Extract all Pydantic model schemas from FastAPI routes.

        Args:
            app: FastAPI application instance
        """
        # Schemas are automatically extracted during route processing
        # This method can be used for additional schema discovery
        pass

    def _extract_fastapi_tags(self, app: "FastAPI") -> None:
        """
        Extract tag definitions from FastAPI app.

        Args:
            app: FastAPI application instance
        """
        if hasattr(app, "openapi_tags") and app.openapi_tags:
            for tag_info in app.openapi_tags:
                self.add_tag(
                    name=tag_info.get("name", ""),
                    description=tag_info.get("description"),
                    external_docs_url=tag_info.get("externalDocs", {}).get("url")
                )

    def _add_default_security_schemes(self) -> None:
        """Add default GreenLang security schemes."""
        # JWT Bearer authentication
        self.add_bearer_auth(
            name="bearerAuth",
            description="JWT Bearer token authentication. Obtain tokens via /api/v1/auth/token"
        )

        # API Key authentication
        self.add_api_key_auth(
            name="apiKeyAuth",
            key_name="X-API-Key",
            location=ParameterLocation.HEADER
        )

        # OAuth2 authentication
        self.add_oauth2(
            name="oauth2",
            authorization_url="https://auth.greenlang.io/oauth/authorize",
            token_url="https://auth.greenlang.io/oauth/token",
            scopes={
                "emissions:read": "Read emission data",
                "emissions:write": "Create and modify emission calculations",
                "agents:read": "Read agent information",
                "agents:execute": "Execute agents",
                "compliance:read": "Read compliance reports",
                "compliance:write": "Generate compliance reports",
                "admin": "Full administrative access"
            }
        )

    # =========================================================================
    # ADDITIONAL UTILITY METHODS
    # =========================================================================

    def add_examples_for_endpoint(
        self,
        path: str,
        method: str,
        request_example: Optional[Dict[str, Any]] = None,
        response_examples: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """
        Add request/response examples to an endpoint.

        Args:
            path: API path
            method: HTTP method
            request_example: Example request body
            response_examples: Dictionary of status code to example response
        """
        if path not in self._paths:
            logger.warning(f"Path {path} not found, cannot add examples")
            return

        path_def = self._paths[path]
        if method.lower() not in path_def.operations:
            logger.warning(f"Method {method} not found for path {path}")
            return

        operation = path_def.operations[method.lower()]

        # Add request example
        if request_example and operation.request_body:
            if "content" in operation.request_body:
                for content_type in operation.request_body["content"]:
                    operation.request_body["content"][content_type]["example"] = request_example

        # Add response examples
        if response_examples:
            for status_code, example in response_examples.items():
                if str(status_code) in operation.responses:
                    if "content" in operation.responses[str(status_code)]:
                        for content_type in operation.responses[str(status_code)]["content"]:
                            operation.responses[str(status_code)]["content"][content_type]["example"] = example

    def merge_spec(self, other_spec: Dict[str, Any]) -> None:
        """
        Merge another OpenAPI specification into this generator.

        Useful for combining multiple API specifications.

        Args:
            other_spec: OpenAPI specification dictionary to merge
        """
        # Merge paths
        if "paths" in other_spec:
            for path, path_def in other_spec["paths"].items():
                if path not in self._paths:
                    self._paths[path] = PathDefinition(path=path)
                for method, operation in path_def.items():
                    self._paths[path].operations[method] = OperationDefinition(
                        operation_id=operation.get("operationId", ""),
                        summary=operation.get("summary", ""),
                        description=operation.get("description"),
                        tags=operation.get("tags", []),
                        parameters=operation.get("parameters", []),
                        request_body=operation.get("requestBody"),
                        responses=operation.get("responses", {}),
                        deprecated=operation.get("deprecated", False)
                    )

        # Merge schemas
        if "components" in other_spec and "schemas" in other_spec["components"]:
            for name, schema in other_spec["components"]["schemas"].items():
                if name not in self._schemas:
                    self._schemas[name] = SchemaDefinition(**schema)

        # Merge tags
        if "tags" in other_spec:
            existing_tag_names = {t["name"] for t in self._tags}
            for tag in other_spec["tags"]:
                if tag["name"] not in existing_tag_names:
                    self._tags.append(tag)

        logger.info(f"Merged specification with {len(other_spec.get('paths', {}))} paths")

    def validate_spec(self) -> List[str]:
        """
        Validate the generated OpenAPI specification.

        Returns:
            List of validation warnings/errors (empty if valid)
        """
        issues = []
        spec = self.generate()

        # Check required fields
        if not spec.get("info", {}).get("title"):
            issues.append("Missing info.title")

        if not spec.get("info", {}).get("version"):
            issues.append("Missing info.version")

        # Check paths have at least one operation
        for path, path_def in spec.get("paths", {}).items():
            if not any(k in path_def for k in ["get", "post", "put", "patch", "delete"]):
                issues.append(f"Path {path} has no operations defined")

        # Check all schema references exist
        schema_refs = set()
        paths_str = json.dumps(spec.get("paths", {}))

        import re
        for ref in re.findall(r'"\$ref":\s*"#/components/schemas/([^"]+)"', paths_str):
            schema_refs.add(ref)

        defined_schemas = set(spec.get("components", {}).get("schemas", {}).keys())
        missing_schemas = schema_refs - defined_schemas
        for schema in missing_schemas:
            issues.append(f"Referenced schema '{schema}' is not defined")

        # Check operation IDs are unique
        operation_ids = []
        for path_def in spec.get("paths", {}).values():
            for method, operation in path_def.items():
                if isinstance(operation, dict) and "operationId" in operation:
                    op_id = operation["operationId"]
                    if op_id in operation_ids:
                        issues.append(f"Duplicate operationId: {op_id}")
                    operation_ids.append(op_id)

        if issues:
            logger.warning(f"OpenAPI validation found {len(issues)} issues")
        else:
            logger.info("OpenAPI specification is valid")

        return issues


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def create_greenlang_openapi_generator(
    title: str = "GreenLang API",
    version: str = "1.0.0",
    description: Optional[str] = None
) -> OpenAPIGenerator:
    """
    Create a pre-configured OpenAPIGenerator for GreenLang.

    Includes default security schemes, servers, and tags.

    Args:
        title: API title
        version: API version
        description: API description

    Returns:
        Configured OpenAPIGenerator instance
    """
    config = OpenAPIGeneratorConfig(
        title=title,
        version=version,
        description=description or "GreenLang Enterprise Sustainability Platform API",
        servers=[
            {"url": "https://api.greenlang.io/v1", "description": "Production"},
            {"url": "https://staging-api.greenlang.io/v1", "description": "Staging"},
            {"url": "http://localhost:8000/api/v1", "description": "Development"},
        ],
        contact_name="GreenLang Support",
        contact_email="support@greenlang.io",
        contact_url="https://greenlang.io/support",
        external_docs_url="https://docs.greenlang.io",
        external_docs_description="GreenLang Documentation"
    )

    generator = OpenAPIGenerator(config)

    # Add default security schemes
    generator._add_default_security_schemes()

    # Add standard tags
    generator.add_tag("Emissions", "Emission calculation and tracking endpoints")
    generator.add_tag("Agents", "Agent management and execution endpoints")
    generator.add_tag("Jobs", "Job queue and status endpoints")
    generator.add_tag("Compliance", "Compliance reporting endpoints")
    generator.add_tag("Health", "Health check and monitoring endpoints")

    return generator


def generate_openapi_from_app(
    app: "FastAPI",
    output_path: Optional[str] = None,
    format: str = "json"
) -> Union[str, Dict[str, Any]]:
    """
    Generate OpenAPI specification from a FastAPI application.

    Convenience function for quick spec generation.

    Args:
        app: FastAPI application instance
        output_path: Optional path to save the specification
        format: Output format ('json' or 'yaml')

    Returns:
        Specification string if output_path provided, dict otherwise

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> spec = generate_openapi_from_app(app)
        >>> generate_openapi_from_app(app, "openapi.json")
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Install with: pip install fastapi")

    generator = OpenAPIGenerator.from_fastapi_app(app)

    if output_path:
        generator.save_to_file(output_path, format=format)
        return output_path

    return generator.generate()
