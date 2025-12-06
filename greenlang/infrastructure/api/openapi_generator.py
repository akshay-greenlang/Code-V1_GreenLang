"""
OpenAPI 3.0 Specification Generator for GreenLang

This module provides automatic OpenAPI specification generation
from GreenLang agent schemas and service definitions.

Features:
- OpenAPI 3.0/3.1 support
- Automatic schema generation from Pydantic models
- Security scheme definitions
- Tag grouping
- Server configuration
- Extension support

Example:
    >>> generator = OpenAPIGenerator(config)
    >>> generator.add_schema(EmissionModel)
    >>> spec = generator.generate()
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
        """
        try:
            import yaml
            return yaml.dump(self.generate(), default_flow_style=False)
        except ImportError:
            raise ImportError("PyYAML is required for YAML output")
