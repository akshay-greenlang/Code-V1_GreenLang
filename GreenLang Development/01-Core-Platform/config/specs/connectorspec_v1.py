# -*- coding: utf-8 -*-
"""
GreenLang ConnectorSpec v1 - Pydantic Models

This module defines the authoritative Pydantic models for ConnectorSpec v1.
These models provide type-safe connector pack manifests with comprehensive validation.

Schema Sections:
- connector: Core config (type, capabilities, endpoints, auth)
- data: Data models (query schema, payload schema)
- security: Egress control, TLS, domain allowlisting
- cache: Snapshot config, TTL, replay modes
- provenance: Query hashing, schema hashing, seed tracking

Design Principles:
- Pydantic v2 as source of truth (JSON Schema is generated from these models)
- Strict validation (extra='forbid' catches typos)
- Determinism by default (connectors must be reproducible)
- Security-first (default-deny egress, TLS enforcement)
- Byte-exact snapshots (canonical JSON, SHA-256 addressing)

Author: GreenLang Framework Team
Date: October 2025
Spec: DATA-301 (Connector SDK + Mock Grid-Intensity)
CTO Approval: Required for all changes
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import jsonschema
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    ValidationError,
)

from .errors import GLVErr, GLValidationError, raise_validation_error


# ============================================================================
# REGEX PATTERNS (Production-Ready)
# ============================================================================

# Semantic Versioning 2.0.0: https://semver.org/
SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

# Connector ID slug: lowercase alphanumeric with separators (/, -, _)
# Example: "grid/intensity/mock", "weather/nws/api"
SLUG_RE = re.compile(
    r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+$"
)

# Python URI: python://module.path:ClassName
# Example: "python://greenlang.connectors.grid.mock:GridIntensityMockConnector"
PYTHON_URI_RE = re.compile(
    r"^python://([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*):([A-Z][a-zA-Z0-9_]*)$",
    re.IGNORECASE
)

# Domain pattern: hostname or hostname:port
# Example: "api.electricitymaps.com", "api.watttime.org:443"
DOMAIN_RE = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?::\d{1,5})?$"
)


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_python_uri(uri: str, field_path: List[str]) -> str:
    """
    Validate Python URI format for connector class reference.

    Security checks:
    1. Format: python://module.path:ClassName
    2. No path traversal (../)
    3. No absolute paths (/etc/passwd)
    4. Valid Python identifiers

    Args:
        uri: Python URI to validate
        field_path: Field path for error reporting

    Returns:
        Validated URI string

    Raises:
        GLValidationError: If URI is malformed or insecure
    """
    if not PYTHON_URI_RE.match(uri):
        raise GLValidationError(
            GLVErr.INVALID_URI,
            f"Invalid python:// URI: '{uri}'. "
            f"Expected format: 'python://module.path:ClassName'. "
            f"Example: 'python://greenlang.connectors.grid.mock:GridIntensityMockConnector'",
            field_path
        )

    # Security: Check for path traversal
    if ".." in uri or uri.startswith("/"):
        raise GLValidationError(
            GLVErr.INVALID_URI,
            f"Security violation: URI contains path traversal or absolute path: '{uri}'",
            field_path
        )

    return uri


def validate_domain(domain: str, field_path: List[str]) -> str:
    """
    Validate domain format for egress allowlist.

    Args:
        domain: Domain to validate (e.g., "api.example.com", "api.example.com:443")
        field_path: Field path for error reporting

    Returns:
        Validated domain string

    Raises:
        GLValidationError: If domain is malformed
    """
    if not DOMAIN_RE.match(domain):
        raise GLValidationError(
            GLVErr.INVALID_URI,
            f"Invalid domain format: '{domain}'. "
            f"Expected format: 'hostname' or 'hostname:port'. "
            f"Example: 'api.example.com' or 'api.example.com:443'",
            field_path
        )

    return domain


# ============================================================================
# PYDANTIC MODELS - Leaf Nodes
# ============================================================================

class ConnectorCapabilities(BaseModel):
    """
    Connector capabilities specification.

    Defines what the connector supports:
    - Time-series data or single-point data
    - Authentication requirements
    - Minimum time resolution (hourly, daily, etc.)
    - Rate limiting information

    Example:
        >>> ConnectorCapabilities(
        ...     supports_time_series=True,
        ...     requires_auth=True,
        ...     min_resolution="hour",
        ...     rate_limit_per_hour=100
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    supports_time_series: bool = Field(
        default=False,
        description="Whether connector supports time-series data queries"
    )
    requires_auth: bool = Field(
        default=False,
        description="Whether connector requires authentication (API key, OAuth, etc.)"
    )
    min_resolution: Optional[Literal["minute", "hour", "day", "month"]] = Field(
        default=None,
        description="Minimum time resolution supported (if time-series)"
    )
    max_resolution: Optional[Literal["minute", "hour", "day", "month", "year"]] = Field(
        default=None,
        description="Maximum time resolution supported (if time-series)"
    )
    rate_limit_per_hour: Optional[int] = Field(
        default=None,
        ge=1,
        description="API rate limit (requests per hour)"
    )
    supports_regions: List[str] = Field(
        default_factory=list,
        description="Supported region codes (ISO 3166-2 format, e.g., ['CA-ON', 'US-CAISO'])"
    )


class AuthConfig(BaseModel):
    """
    Authentication configuration for connector.

    Supports multiple auth types:
    - api_key: API key in header or query param
    - oauth2: OAuth 2.0 flow
    - basic: HTTP Basic Auth
    - none: No authentication

    Example:
        >>> AuthConfig(
        ...     type="api_key",
        ...     header_name="X-API-Key",
        ...     env_var="ELECTRICITYMAPS_API_KEY"
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["api_key", "oauth2", "basic", "none"] = Field(
        default="none",
        description="Authentication type"
    )
    header_name: Optional[str] = Field(
        default=None,
        description="HTTP header name for API key (if type=api_key)"
    )
    query_param: Optional[str] = Field(
        default=None,
        description="Query parameter name for API key (if type=api_key)"
    )
    env_var: Optional[str] = Field(
        default=None,
        description="Environment variable name containing credentials"
    )
    oauth_flow: Optional[Literal["client_credentials", "authorization_code"]] = Field(
        default=None,
        description="OAuth 2.0 flow type (if type=oauth2)"
    )
    token_endpoint: Optional[str] = Field(
        default=None,
        description="OAuth 2.0 token endpoint URL (if type=oauth2)"
    )


class EndpointConfig(BaseModel):
    """
    API endpoint configuration.

    Example:
        >>> EndpointConfig(
        ...     base_url="https://api.electricitymaps.com/v3",
        ...     query_path="/carbon-intensity/history",
        ...     method="GET",
        ...     timeout_s=30
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(
        ...,
        description="Base URL for API endpoint (must use HTTPS)"
    )
    query_path: str = Field(
        ...,
        description="Path for query endpoint (appended to base_url)"
    )
    method: Literal["GET", "POST"] = Field(
        default="GET",
        description="HTTP method"
    )
    timeout_s: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout in seconds (default: 30s, max: 5 minutes)"
    )

    @field_validator("base_url")
    @classmethod
    def validate_https(cls, v: str, info) -> str:
        """Enforce HTTPS for security."""
        if not v.startswith("https://"):
            raise GLValidationError(
                GLVErr.INVALID_URI,
                f"Endpoint base_url must use HTTPS (got: {v})",
                ["connector", "endpoint", "base_url"]
            )
        return v


# ============================================================================
# PYDANTIC MODELS - Section Specs
# ============================================================================

class ConnectorSpec(BaseModel):
    """
    Connector section: core configuration.

    Defines the connector's identity, implementation, capabilities, and endpoints.

    Example:
        >>> ConnectorSpec(
        ...     type="grid_intensity",
        ...     impl="python://greenlang.connectors.grid.mock:GridIntensityMockConnector",
        ...     capabilities=ConnectorCapabilities(
        ...         supports_time_series=True,
        ...         requires_auth=False,
        ...         min_resolution="hour"
        ...     ),
        ...     mock=True
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "grid_intensity",
        "weather",
        "commodity_price",
        "emission_factor",
        "custom"
    ] = Field(
        ...,
        description="Connector type category"
    )
    impl: str = Field(
        ...,
        description="Python URI for connector implementation (python://module:ClassName)"
    )
    capabilities: ConnectorCapabilities = Field(
        ...,
        description="Connector capabilities"
    )
    endpoint: Optional[EndpointConfig] = Field(
        default=None,
        description="API endpoint configuration (omit for mock connectors)"
    )
    auth: Optional[AuthConfig] = Field(
        default=None,
        description="Authentication configuration (omit for mock connectors)"
    )
    mock: bool = Field(
        default=False,
        description="Whether this is a mock connector (no network access)"
    )

    @field_validator("impl")
    @classmethod
    def validate_impl_uri(cls, v: str, info) -> str:
        """Validate python:// URI format."""
        field_path = ["connector", "impl"]
        return validate_python_uri(v, field_path)

    @model_validator(mode="after")
    def validate_mock_no_endpoint(self):
        """Mock connectors must not have endpoint config."""
        if self.mock and self.endpoint:
            raise GLValidationError(
                GLVErr.CONSTRAINT,
                "Mock connectors must not have endpoint configuration",
                ["connector", "endpoint"]
            )
        return self

    @model_validator(mode="after")
    def validate_real_has_endpoint(self):
        """Real (non-mock) connectors must have endpoint config."""
        if not self.mock and not self.endpoint:
            raise GLValidationError(
                GLVErr.CONSTRAINT,
                "Non-mock connectors must have endpoint configuration",
                ["connector", "endpoint"]
            )
        return self


class DataSpec(BaseModel):
    """
    Data section: query and payload schemas.

    Defines JSON Schemas for query inputs and payload outputs.
    Uses JSON Schema draft-2020-12 for maximum compatibility.

    Example:
        >>> DataSpec(
        ...     query_schema={
        ...         "type": "object",
        ...         "properties": {
        ...             "region": {"type": "string", "pattern": "^[A-Z]{2}(-[A-Z0-9]+)?$"},
        ...             "start": {"type": "string", "format": "date-time"},
        ...             "end": {"type": "string", "format": "date-time"}
        ...         },
        ...         "required": ["region", "start", "end"]
        ...     },
        ...     payload_schema={
        ...         "type": "object",
        ...         "properties": {
        ...             "series": {"type": "array", "items": {...}},
        ...             "region": {"type": "string"}
        ...         },
        ...         "required": ["series", "region"]
        ...     }
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    query_schema: Dict[str, Any] = Field(
        ...,
        description="JSON Schema (draft-2020-12) for query input"
    )
    payload_schema: Dict[str, Any] = Field(
        ...,
        description="JSON Schema (draft-2020-12) for payload output"
    )
    query_model: Optional[str] = Field(
        default=None,
        description="Pydantic model name for query (e.g., 'GridIntensityQuery')"
    )
    payload_model: Optional[str] = Field(
        default=None,
        description="Pydantic model name for payload (e.g., 'GridIntensityPayload')"
    )

    @model_validator(mode="after")
    def validate_json_schemas(self):
        """Validate query_schema and payload_schema are valid JSON Schema draft-2020-12."""
        # Validate query_schema
        try:
            jsonschema.Draft202012Validator.check_schema(self.query_schema)
        except jsonschema.SchemaError as e:
            raise GLValidationError(
                GLVErr.AI_SCHEMA_INVALID,
                f"Invalid query_schema: {e.message}",
                ["data", "query_schema"]
            )

        # Validate payload_schema
        try:
            jsonschema.Draft202012Validator.check_schema(self.payload_schema)
        except jsonschema.SchemaError as e:
            raise GLValidationError(
                GLVErr.AI_SCHEMA_INVALID,
                f"Invalid payload_schema: {e.message}",
                ["data", "payload_schema"]
            )

        return self


class SecuritySpec(BaseModel):
    """
    Security section: egress control, TLS enforcement, domain allowlisting.

    Implements default-deny security model:
    - Network egress disabled by default
    - Explicit domain allowlisting required
    - TLS enforcement (HTTPS only)
    - Metadata endpoint blocking (169.254.169.254)

    Example:
        >>> SecuritySpec(
        ...     allow_egress=True,
        ...     egress_allowlist=["api.electricitymaps.com:443", "api.watttime.org:443"],
        ...     require_tls=True,
        ...     block_metadata_endpoints=True
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    allow_egress: bool = Field(
        default=False,
        description="Whether to allow network egress (default: false, default-deny)"
    )
    egress_allowlist: List[str] = Field(
        default_factory=list,
        description="Allowed domains for egress (hostname or hostname:port)"
    )
    require_tls: bool = Field(
        default=True,
        description="Require TLS/HTTPS for all network requests (default: true)"
    )
    block_metadata_endpoints: bool = Field(
        default=True,
        description="Block access to cloud metadata endpoints (169.254.169.254, etc.)"
    )
    max_redirect_hops: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Maximum HTTP redirect hops allowed (default: 0, no redirects)"
    )

    @field_validator("egress_allowlist")
    @classmethod
    def validate_domains(cls, v: List[str], info) -> List[str]:
        """Validate all domains in allowlist."""
        for domain in v:
            validate_domain(domain, ["security", "egress_allowlist"])
        return v

    @model_validator(mode="after")
    def validate_allowlist_if_egress(self):
        """If egress enabled, require non-empty allowlist."""
        if self.allow_egress and not self.egress_allowlist:
            raise GLValidationError(
                GLVErr.CONSTRAINT,
                "allow_egress=true requires non-empty egress_allowlist",
                ["security", "egress_allowlist"]
            )
        return self


class CacheSpec(BaseModel):
    """
    Cache section: snapshot configuration, TTL, replay modes.

    Controls how connector data is cached and replayed:
    - Snapshot format (canonical JSON)
    - TTL (time-to-live) for cached data
    - Supported modes (record, replay, golden)

    Example:
        >>> CacheSpec(
        ...     snapshot_format="canonical_json",
        ...     default_ttl_hours=6,
        ...     snapshot_dir=".greenlang/snapshots/connectors",
        ...     supported_modes=["record", "replay", "golden"]
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    snapshot_format: Literal["canonical_json"] = Field(
        default="canonical_json",
        description="Snapshot serialization format (canonical_json for byte-exact reproducibility)"
    )
    default_ttl_hours: Optional[int] = Field(
        default=None,
        ge=1,
        le=8760,
        description="Default TTL for snapshots in hours (None = no expiry)"
    )
    snapshot_dir: str = Field(
        default=".greenlang/snapshots/connectors",
        description="Directory for storing snapshots (relative to project root)"
    )
    supported_modes: List[Literal["record", "replay", "golden"]] = Field(
        default_factory=lambda: ["record", "replay"],
        description="Supported execution modes"
    )

    @field_validator("supported_modes")
    @classmethod
    def validate_modes_unique(cls, v: List[str], info) -> List[str]:
        """Ensure modes are unique."""
        if len(v) != len(set(v)):
            duplicates = [mode for mode in v if v.count(mode) > 1]
            raise GLValidationError(
                GLVErr.DUPLICATE_NAME,
                f"Duplicate cache modes: {duplicates}",
                ["cache", "supported_modes"]
            )
        return v


class ProvenanceSpec(BaseModel):
    """
    Provenance section: query hashing, schema hashing, seed tracking.

    Ensures connector fetches are reproducible and auditable:
    - Query hash (SHA-256 of canonical query)
    - Schema hash (SHA-256 of payload schema)
    - Seed tracking (for deterministic mock data)
    - Snapshot ID (SHA-256 of snapshot bytes)

    Example:
        >>> ProvenanceSpec(
        ...     compute_query_hash=True,
        ...     compute_schema_hash=True,
        ...     track_seed=True,
        ...     record_fields=["connector_id", "connector_version", "query_hash", "schema_hash", "seed", "snapshot_id", "timestamp"]
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    compute_query_hash: bool = Field(
        default=True,
        description="Compute SHA-256 hash of canonical query (default: true)"
    )
    compute_schema_hash: bool = Field(
        default=True,
        description="Compute SHA-256 hash of payload schema (default: true)"
    )
    track_seed: bool = Field(
        default=False,
        description="Track deterministic seed for mock connectors (default: false)"
    )
    record_fields: List[str] = Field(
        default_factory=lambda: [
            "connector_id",
            "connector_version",
            "mode",
            "query_hash",
            "schema_hash",
            "timestamp"
        ],
        description="Fields to include in provenance record"
    )

    @field_validator("record_fields")
    @classmethod
    def validate_record_unique(cls, v: List[str], info) -> List[str]:
        """Ensure record fields are unique."""
        if len(v) != len(set(v)):
            duplicates = [field for field in v if v.count(field) > 1]
            raise GLValidationError(
                GLVErr.DUPLICATE_NAME,
                f"Duplicate provenance record fields: {duplicates}",
                ["provenance", "record_fields"]
            )
        return v


# ============================================================================
# PYDANTIC MODELS - Top-Level ConnectorSpec v1
# ============================================================================

class ConnectorSpecV1(BaseModel):
    """
    GreenLang ConnectorSpec v1 - Top-Level Schema

    This is the authoritative specification for GreenLang connector packs.
    Every connector pack MUST conform to this schema.

    Schema Sections:
    - Metadata: schema_version, id, name, version, summary, tags, owners, license
    - connector: Core config (type, impl, capabilities, endpoints, auth)
    - data: Data models (query schema, payload schema)
    - security: Egress control, TLS enforcement, domain allowlisting
    - cache: Snapshot config, TTL, replay modes
    - provenance: Query hashing, schema hashing, seed tracking
    - tests: Golden tests and property-based tests (optional)

    Example:
        >>> spec = ConnectorSpecV1(
        ...     schema_version="1.0.0",
        ...     id="grid/intensity/mock",
        ...     name="Grid Intensity Mock Connector",
        ...     version="0.1.0",
        ...     summary="Mock connector for grid carbon intensity data (deterministic, no network)",
        ...     connector=ConnectorSpec(...),
        ...     data=DataSpec(...),
        ...     security=SecuritySpec(...),
        ...     cache=CacheSpec(...),
        ...     provenance=ProvenanceSpec(...)
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    # Metadata
    schema_version: Literal["1.0.0"] = Field(
        ...,
        description="ConnectorSpec schema version (MUST be '1.0.0')"
    )
    id: str = Field(
        ...,
        description="Connector ID slug (e.g., 'grid/intensity/mock'). Format: segment/segment/..."
    )
    name: str = Field(
        ...,
        min_length=3,
        description="Human-readable connector name"
    )
    version: str = Field(
        ...,
        description="Connector version (semantic versioning 2.0.0)"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Short description of connector purpose"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization and search"
    )
    owners: Optional[List[str]] = Field(
        default=None,
        description="Connector owners (e.g., ['@gl/data-connectors'])"
    )
    license: Optional[str] = Field(
        default=None,
        description="License identifier (e.g., 'Apache-2.0', 'MIT')"
    )

    # Core sections (required)
    connector: ConnectorSpec = Field(
        ...,
        description="Connector specification (type, impl, capabilities, endpoints)"
    )
    data: DataSpec = Field(
        ...,
        description="Data specification (query schema, payload schema)"
    )
    security: SecuritySpec = Field(
        ...,
        description="Security specification (egress control, TLS, allowlisting)"
    )
    cache: CacheSpec = Field(
        ...,
        description="Cache specification (snapshot format, TTL, modes)"
    )
    provenance: ProvenanceSpec = Field(
        ...,
        description="Provenance specification (query hash, schema hash, seed tracking)"
    )

    # Optional sections
    tests: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Test configuration (golden tests, property-based tests)"
    )

    @field_validator("id")
    @classmethod
    def validate_id_slug(cls, v: str, info) -> str:
        """Validate connector ID slug format."""
        if not SLUG_RE.match(v):
            raise GLValidationError(
                GLVErr.INVALID_SLUG,
                f"Invalid connector ID slug: '{v}'. "
                f"Expected format: 'segment/segment/...' with lowercase alphanumeric and separators (-, _). "
                f"Example: 'grid/intensity/mock'",
                ["id"]
            )
        return v

    @field_validator("version")
    @classmethod
    def validate_version_semver(cls, v: str, info) -> str:
        """Validate version conforms to Semantic Versioning 2.0.0."""
        if not SEMVER_RE.match(v):
            raise GLValidationError(
                GLVErr.INVALID_SEMVER,
                f"Invalid semantic version: '{v}'. "
                f"Expected format: MAJOR.MINOR.PATCH (e.g., '0.1.0'). "
                f"See https://semver.org/ for specification.",
                ["version"]
            )
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags_unique(cls, v: List[str], info) -> List[str]:
        """Ensure tags are unique."""
        if len(v) != len(set(v)):
            duplicates = [tag for tag in v if v.count(tag) > 1]
            raise GLValidationError(
                GLVErr.DUPLICATE_NAME,
                f"Duplicate tags: {duplicates}",
                ["tags"]
            )
        return v


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def from_yaml(path: Union[str, Path]) -> ConnectorSpecV1:
    """
    Load ConnectorSpec v1 from YAML file.

    Args:
        path: Path to connector.yaml file

    Returns:
        Validated ConnectorSpecV1 instance

    Raises:
        GLValidationError: If spec is invalid
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ConnectorSpec file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        return ConnectorSpecV1.model_validate(data)
    except ValidationError as e:
        gl_errors = GLValidationError.from_pydantic(e, context=str(path))
        # Raise first error (caller can catch and iterate through all if needed)
        raise gl_errors[0] if gl_errors else e


def from_json(path: Union[str, Path]) -> ConnectorSpecV1:
    """
    Load ConnectorSpec v1 from JSON file.

    Args:
        path: Path to connector.json file

    Returns:
        Validated ConnectorSpecV1 instance

    Raises:
        GLValidationError: If spec is invalid
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ConnectorSpec file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        return ConnectorSpecV1.model_validate(data)
    except ValidationError as e:
        gl_errors = GLValidationError.from_pydantic(e, context=str(path))
        raise gl_errors[0] if gl_errors else e


def validate_spec(data: dict) -> ConnectorSpecV1:
    """
    Validate ConnectorSpec v1 from dictionary.

    Args:
        data: ConnectorSpec data as dictionary

    Returns:
        Validated ConnectorSpecV1 instance

    Raises:
        GLValidationError: If spec is invalid
    """
    try:
        return ConnectorSpecV1.model_validate(data)
    except ValidationError as e:
        gl_errors = GLValidationError.from_pydantic(e)
        raise gl_errors[0] if gl_errors else e


def to_json_schema() -> dict:
    """
    Export ConnectorSpec v1 as JSON Schema (draft-2020-12).

    This is generated from the Pydantic models (Pydantic is source of truth).
    Used by:
    - CLI validator (gl spec validate)
    - Documentation generation
    - External tooling (VS Code, CI checks)

    Returns:
        JSON Schema dictionary
    """
    schema = ConnectorSpecV1.model_json_schema(mode="serialization")

    # Add JSON Schema metadata
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://greenlang.io/specs/connectorspec_v1.json"
    schema["title"] = "GreenLang ConnectorSpec v1"

    return schema
