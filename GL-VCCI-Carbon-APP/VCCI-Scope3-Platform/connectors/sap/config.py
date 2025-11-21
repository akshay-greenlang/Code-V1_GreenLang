# -*- coding: utf-8 -*-
"""
SAP S/4HANA Connector Configuration Management
GL-VCCI Scope 3 Platform

Configuration management for SAP connector including environment variables,
OAuth settings, OData endpoints, and service settings.

Version: 1.0.0
Phase: 4 (Weeks 19-22)
Date: 2025-11-06
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field, validator
from greenlang.determinism import FinancialDecimal


class SAPEnvironment(str, Enum):
    """SAP environment types."""
    SANDBOX = "sandbox"
    DEVELOPMENT = "development"
    QUALITY = "quality"
    PRODUCTION = "production"


class SAPModule(str, Enum):
    """SAP S/4HANA modules supported."""
    MM = "MM"  # Materials Management
    SD = "SD"  # Sales & Distribution
    FI = "FI"  # Financial Accounting


class ODataEndpoint(BaseModel):
    """
    Configuration for a single OData endpoint.

    Attributes:
        name: Endpoint name (e.g., "purchase_orders")
        service_path: OData service path
        entity_set: Entity set name in OData service
        module: SAP module (MM, SD, FI)
        enabled: Whether endpoint is enabled
        batch_size: Default batch size for pagination
    """
    name: str
    service_path: str
    entity_set: str
    module: SAPModule
    enabled: bool = True
    batch_size: int = Field(default=1000, ge=1, le=10000)

    class Config:
        use_enum_values = True


class RetryConfig(BaseModel):
    """
    Configuration for retry logic with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay in seconds
        backoff_multiplier: Multiplier for exponential backoff
        retry_on_status_codes: HTTP status codes to retry on
    """
    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    max_delay: float = Field(default=8.0, ge=1.0, le=300.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    retry_on_status_codes: List[int] = Field(
        default_factory=lambda: [408, 429, 500, 502, 503, 504]
    )

    @validator('max_delay')
    def max_delay_must_be_greater_than_base(cls, v, values):
        """Validate max_delay is greater than base_delay."""
        if 'base_delay' in values and v < values['base_delay']:
            raise ValueError('max_delay must be greater than base_delay')
        return v


class RateLimitConfig(BaseModel):
    """
    Configuration for rate limiting.

    Attributes:
        requests_per_minute: Maximum requests per minute
        enabled: Whether rate limiting is enabled
        burst_size: Maximum burst size for token bucket algorithm
    """
    requests_per_minute: int = Field(default=10, ge=1, le=1000)
    enabled: bool = True
    burst_size: int = Field(default=5, ge=1, le=100)


class OAuth2Config(BaseModel):
    """
    Configuration for OAuth 2.0 authentication.

    Attributes:
        client_id: OAuth client ID
        client_secret: OAuth client secret
        token_url: OAuth token endpoint URL
        scope: OAuth scopes (space-separated)
        grant_type: OAuth grant type
        token_cache_ttl: Token cache TTL in seconds
    """
    client_id: str
    client_secret: str
    token_url: str
    scope: str = "API_BUSINESS_PARTNER"
    grant_type: str = "client_credentials"
    token_cache_ttl: int = Field(default=3300, ge=60, le=86400)  # 55 minutes (token valid 60min)

    class Config:
        # Don't expose secrets in string representation
        json_encoders = {
            str: lambda v: '***' if 'secret' in str(v).lower() else v
        }


class TimeoutConfig(BaseModel):
    """
    Configuration for request timeouts.

    Attributes:
        connect_timeout: Connection timeout in seconds
        read_timeout: Read timeout in seconds
        total_timeout: Total timeout in seconds
    """
    connect_timeout: float = Field(default=10.0, ge=1.0, le=60.0)
    read_timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    total_timeout: float = Field(default=60.0, ge=1.0, le=600.0)


class SAPConnectorConfig(BaseModel):
    """
    Main configuration class for SAP S/4HANA connector.

    This class consolidates all configuration settings and provides
    methods to load from environment variables.
    """
    # Environment
    environment: SAPEnvironment = SAPEnvironment.SANDBOX

    # Base URL
    base_url: str

    # OAuth configuration
    oauth: OAuth2Config

    # Rate limiting
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    # Retry configuration
    retry: RetryConfig = Field(default_factory=RetryConfig)

    # Timeout configuration
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)

    # OData endpoints
    endpoints: Dict[str, ODataEndpoint] = Field(default_factory=dict)

    # Default batch size
    default_batch_size: int = Field(default=1000, ge=1, le=10000)

    # Connection pool settings
    pool_connections: int = Field(default=10, ge=1, le=100)
    pool_maxsize: int = Field(default=20, ge=1, le=200)

    # Enable detailed logging
    debug_mode: bool = False

    class Config:
        use_enum_values = True

    def __init__(self, **data):
        """Initialize and set up default endpoints."""
        super().__init__(**data)
        if not self.endpoints:
            self._init_default_endpoints()

    def _init_default_endpoints(self):
        """Initialize default OData endpoint configurations."""
        self.endpoints = {
            # MM (Materials Management) endpoints
            "purchase_orders": ODataEndpoint(
                name="purchase_orders",
                service_path="/sap/opu/odata/sap/MM_PUR_PO_MAINT_V2_SRV",
                entity_set="C_PurchaseOrderTP",
                module=SAPModule.MM,
                enabled=True,
                batch_size=self.default_batch_size
            ),
            "goods_receipts": ODataEndpoint(
                name="goods_receipts",
                service_path="/sap/opu/odata/sap/API_MATERIAL_DOCUMENT_SRV",
                entity_set="A_MaterialDocumentHeader",
                module=SAPModule.MM,
                enabled=True,
                batch_size=self.default_batch_size
            ),
            "vendor_master": ODataEndpoint(
                name="vendor_master",
                service_path="/sap/opu/odata/sap/MD_SUPPLIER_MASTER_SRV",
                entity_set="A_Supplier",
                module=SAPModule.MM,
                enabled=True,
                batch_size=self.default_batch_size
            ),
            "material_master": ODataEndpoint(
                name="material_master",
                service_path="/sap/opu/odata/sap/API_MATERIAL_STOCK_SRV",
                entity_set="A_MaterialStock",
                module=SAPModule.MM,
                enabled=True,
                batch_size=self.default_batch_size
            ),
            # SD (Sales & Distribution) endpoints
            "outbound_deliveries": ODataEndpoint(
                name="outbound_deliveries",
                service_path="/sap/opu/odata/sap/API_OUTBOUND_DELIVERY_SRV",
                entity_set="A_OutbDeliveryHeader",
                module=SAPModule.SD,
                enabled=True,
                batch_size=self.default_batch_size
            ),
            "transportation_orders": ODataEndpoint(
                name="transportation_orders",
                service_path="/sap/opu/odata/sap/API_TRANSPORTATION_ORDER_SRV",
                entity_set="A_TransportationOrder",
                module=SAPModule.SD,
                enabled=True,
                batch_size=self.default_batch_size
            ),
            # FI (Financial Accounting) endpoints
            "fixed_assets": ODataEndpoint(
                name="fixed_assets",
                service_path="/sap/opu/odata/sap/API_FIXEDASSET_SRV",
                entity_set="A_FixedAsset",
                module=SAPModule.FI,
                enabled=True,
                batch_size=self.default_batch_size
            ),
        }

    @classmethod
    def from_env(cls, environment: Optional[str] = None) -> "SAPConnectorConfig":
        """
        Load configuration from environment variables.

        Environment variables:
            SAP_ENVIRONMENT: Environment (sandbox, development, quality, production)
            SAP_BASE_URL: SAP S/4HANA base URL
            SAP_CLIENT_ID: OAuth client ID
            SAP_CLIENT_SECRET: OAuth client secret
            SAP_TOKEN_URL: OAuth token URL
            SAP_OAUTH_SCOPE: OAuth scopes
            SAP_RATE_LIMIT_RPM: Rate limit (requests per minute)
            SAP_BATCH_SIZE: Default batch size for pagination
            SAP_DEBUG_MODE: Enable debug logging

        Args:
            environment: Override environment from env var

        Returns:
            SAPConnectorConfig instance

        Raises:
            ValueError: If required environment variables are missing
        """
        # Determine environment
        env = environment or os.getenv("SAP_ENVIRONMENT", "sandbox")
        sap_environment = SAPEnvironment(env)

        # Load base URL (required)
        base_url = os.getenv("SAP_BASE_URL")
        if not base_url:
            raise ValueError("SAP_BASE_URL environment variable is required")

        # Load OAuth config (required)
        client_id = os.getenv("SAP_CLIENT_ID")
        client_secret = os.getenv("SAP_CLIENT_SECRET")
        token_url = os.getenv("SAP_TOKEN_URL")

        if not all([client_id, client_secret, token_url]):
            raise ValueError(
                "SAP_CLIENT_ID, SAP_CLIENT_SECRET, and SAP_TOKEN_URL "
                "environment variables are required"
            )

        oauth_config = OAuth2Config(
            client_id=client_id,
            client_secret=client_secret,
            token_url=token_url,
            scope=os.getenv("SAP_OAUTH_SCOPE", "API_BUSINESS_PARTNER")
        )

        # Rate limit configuration
        rate_limit_config = RateLimitConfig(
            requests_per_minute=int(os.getenv("SAP_RATE_LIMIT_RPM", "10")),
            enabled=os.getenv("SAP_RATE_LIMIT_ENABLED", "true").lower() == "true"
        )

        # Retry configuration
        retry_config = RetryConfig(
            max_retries=int(os.getenv("SAP_MAX_RETRIES", "3")),
            base_delay=float(os.getenv("SAP_RETRY_BASE_DELAY", "1.0")),
            max_delay=float(os.getenv("SAP_RETRY_MAX_DELAY", "8.0"))
        )

        # Timeout configuration
        timeout_config = TimeoutConfig(
            connect_timeout=float(os.getenv("SAP_CONNECT_TIMEOUT", "10.0")),
            read_timeout=float(os.getenv("SAP_READ_TIMEOUT", "30.0")),
            total_timeout=FinancialDecimal.from_string(os.getenv("SAP_TOTAL_TIMEOUT", "60.0"))
        )

        # Default batch size
        default_batch_size = int(os.getenv("SAP_BATCH_SIZE", "1000"))

        # Debug mode
        debug_mode = os.getenv("SAP_DEBUG_MODE", "false").lower() == "true"

        return cls(
            environment=sap_environment,
            base_url=base_url,
            oauth=oauth_config,
            rate_limit=rate_limit_config,
            retry=retry_config,
            timeout=timeout_config,
            default_batch_size=default_batch_size,
            debug_mode=debug_mode
        )

    def get_endpoint_config(self, endpoint_name: str) -> Optional[ODataEndpoint]:
        """
        Get configuration for a specific endpoint.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            ODataEndpoint if found, None otherwise
        """
        return self.endpoints.get(endpoint_name)

    def is_endpoint_enabled(self, endpoint_name: str) -> bool:
        """
        Check if an endpoint is enabled.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            True if endpoint is enabled, False otherwise
        """
        endpoint = self.get_endpoint_config(endpoint_name)
        return endpoint.enabled if endpoint else False

    def get_full_endpoint_url(self, endpoint_name: str) -> Optional[str]:
        """
        Get full URL for an endpoint.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            Full URL if endpoint exists, None otherwise
        """
        endpoint = self.get_endpoint_config(endpoint_name)
        if not endpoint:
            return None

        # Remove trailing slash from base_url if present
        base = self.base_url.rstrip('/')
        # Ensure service_path starts with /
        path = endpoint.service_path if endpoint.service_path.startswith('/') else f'/{endpoint.service_path}'

        return f"{base}{path}"

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate base URL
        if not self.base_url:
            errors.append("base_url is required")
        elif not (self.base_url.startswith('http://') or self.base_url.startswith('https://')):
            errors.append("base_url must start with http:// or https://")

        # Validate OAuth config
        if not self.oauth.client_id:
            errors.append("OAuth client_id is required")
        if not self.oauth.client_secret:
            errors.append("OAuth client_secret is required")
        if not self.oauth.token_url:
            errors.append("OAuth token_url is required")

        # Validate at least one endpoint is enabled
        enabled_endpoints = [e for e in self.endpoints.values() if e.enabled]
        if not enabled_endpoints:
            errors.append("At least one endpoint must be enabled")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary (excluding secrets).

        Returns:
            Dictionary representation of configuration
        """
        return {
            "environment": self.environment,
            "base_url": self.base_url,
            "oauth": {
                "client_id": self.oauth.client_id,
                "token_url": self.oauth.token_url,
                "scope": self.oauth.scope,
                # Exclude client_secret
            },
            "rate_limit": self.rate_limit.dict(),
            "retry": self.retry.dict(),
            "timeout": self.timeout.dict(),
            "endpoints": {
                name: {
                    "service_path": endpoint.service_path,
                    "entity_set": endpoint.entity_set,
                    "module": endpoint.module,
                    "enabled": endpoint.enabled,
                    "batch_size": endpoint.batch_size
                }
                for name, endpoint in self.endpoints.items()
            },
            "default_batch_size": self.default_batch_size,
            "debug_mode": self.debug_mode
        }


# Global configuration instance
_config: Optional[SAPConnectorConfig] = None


def get_config() -> SAPConnectorConfig:
    """
    Get global configuration instance.

    Loads configuration from environment on first call,
    then returns cached instance.

    Returns:
        SAPConnectorConfig instance
    """
    global _config

    if _config is None:
        _config = SAPConnectorConfig.from_env()

    return _config


def reset_config():
    """
    Reset global configuration instance.

    Useful for testing or when configuration needs to be reloaded.
    """
    global _config
    _config = None
