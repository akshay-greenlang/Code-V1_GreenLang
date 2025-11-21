# -*- coding: utf-8 -*-
"""
Multi-tenancy Support for GreenLang
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import secrets
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock

try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logging.warning("PyJWT not available. Install with: pip install pyjwt")

logger = logging.getLogger(__name__)


class TenantTier(Enum):
    """Tenant subscription tiers"""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class IsolationLevel(Enum):
    """Data isolation levels"""

    SHARED = "shared"  # Shared resources, logical isolation
    NAMESPACE = "namespace"  # Kubernetes namespace isolation
    CLUSTER = "cluster"  # Dedicated cluster
    PHYSICAL = "physical"  # Physical isolation


@dataclass
class TenantQuota:
    """Resource quotas for tenant"""

    # Compute resources
    max_cpu: str = "10"  # CPU cores
    max_memory: str = "20Gi"  # Memory
    max_storage: str = "100Gi"  # Storage
    max_gpu: int = 0  # GPU count

    # Execution limits
    max_concurrent_pipelines: int = 5  # Concurrent pipeline executions
    max_pipeline_duration: int = 3600  # Max pipeline duration (seconds)
    max_steps_per_pipeline: int = 20  # Max steps per pipeline
    max_retries: int = 3  # Max retry attempts

    # Pack limits
    max_packs: int = 100  # Max number of packs
    max_pack_size: str = "1Gi"  # Max pack size
    allowed_pack_categories: List[str] = field(default_factory=list)

    # API limits
    max_api_calls_per_hour: int = 1000  # API rate limit
    max_api_calls_per_day: int = 10000  # Daily API limit

    # Data limits
    max_datasets: int = 50  # Max datasets
    max_dataset_size: str = "10Gi"  # Max dataset size
    data_retention_days: int = 90  # Data retention period

    # Network limits
    max_bandwidth_mbps: int = 100  # Max network bandwidth
    allowed_endpoints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "compute": {
                "cpu": self.max_cpu,
                "memory": self.max_memory,
                "storage": self.max_storage,
                "gpu": self.max_gpu,
            },
            "execution": {
                "concurrent_pipelines": self.max_concurrent_pipelines,
                "pipeline_duration": self.max_pipeline_duration,
                "steps_per_pipeline": self.max_steps_per_pipeline,
                "retries": self.max_retries,
            },
            "packs": {
                "max_packs": self.max_packs,
                "max_size": self.max_pack_size,
                "categories": self.allowed_pack_categories,
            },
            "api": {
                "calls_per_hour": self.max_api_calls_per_hour,
                "calls_per_day": self.max_api_calls_per_day,
            },
            "data": {
                "max_datasets": self.max_datasets,
                "max_size": self.max_dataset_size,
                "retention_days": self.data_retention_days,
            },
            "network": {
                "bandwidth_mbps": self.max_bandwidth_mbps,
                "endpoints": self.allowed_endpoints,
            },
        }

    @classmethod
    def from_tier(cls, tier: TenantTier) -> "TenantQuota":
        """Create quota from tier"""
        if tier == TenantTier.FREE:
            return cls(
                max_cpu="1",
                max_memory="2Gi",
                max_storage="10Gi",
                max_concurrent_pipelines=1,
                max_packs=10,
                max_api_calls_per_hour=100,
            )
        elif tier == TenantTier.STARTER:
            return cls(
                max_cpu="4",
                max_memory="8Gi",
                max_storage="50Gi",
                max_concurrent_pipelines=3,
                max_packs=50,
                max_api_calls_per_hour=500,
            )
        elif tier == TenantTier.PROFESSIONAL:
            return cls(
                max_cpu="16",
                max_memory="32Gi",
                max_storage="200Gi",
                max_concurrent_pipelines=10,
                max_packs=200,
                max_api_calls_per_hour=2000,
                max_gpu=1,
            )
        elif tier == TenantTier.ENTERPRISE:
            return cls(
                max_cpu="64",
                max_memory="128Gi",
                max_storage="1Ti",
                max_concurrent_pipelines=50,
                max_packs=1000,
                max_api_calls_per_hour=10000,
                max_gpu=4,
            )
        else:  # CUSTOM
            return cls()


@dataclass
class TenantSettings:
    """Tenant configuration settings"""

    # Feature flags
    features: Dict[str, bool] = field(default_factory=dict)

    # Regional settings
    region: str = "us-west-2"
    availability_zones: List[str] = field(default_factory=list)

    # Security settings
    require_mfa: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    encryption_key_id: Optional[str] = None

    # Compliance settings
    compliance_standards: List[str] = field(
        default_factory=list
    )  # GDPR, HIPAA, SOC2, etc.
    data_residency: Optional[str] = None

    # Integration settings
    allowed_integrations: List[str] = field(default_factory=list)
    webhook_urls: List[str] = field(default_factory=list)

    # Notification settings
    notification_channels: List[str] = field(default_factory=list)
    alert_thresholds: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tenant:
    """Tenant information"""

    tenant_id: str = field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    name: str = ""
    organization: str = ""
    tier: TenantTier = TenantTier.FREE

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    # Status
    active: bool = True
    suspended: bool = False
    suspension_reason: Optional[str] = None

    # Configuration
    quota: TenantQuota = field(default_factory=TenantQuota)
    settings: TenantSettings = field(default_factory=TenantSettings)
    isolation_level: IsolationLevel = IsolationLevel.SHARED

    # Contact information
    admin_email: Optional[str] = None
    admin_name: Optional[str] = None
    billing_email: Optional[str] = None

    # Usage tracking
    current_usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "organization": self.organization,
            "tier": self.tier.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "active": self.active,
            "suspended": self.suspended,
            "quota": self.quota.to_dict(),
            "settings": {
                "features": self.settings.features,
                "region": self.settings.region,
                "compliance": self.settings.compliance_standards,
            },
            "isolation_level": self.isolation_level.value,
            "admin_email": self.admin_email,
        }


@dataclass
class TenantContext:
    """Runtime context for tenant operations"""

    tenant_id: str
    tenant: Optional[Tenant] = None

    # Runtime limits
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    allowed_packs: List[str] = field(default_factory=list)
    allowed_actions: Set[str] = field(default_factory=set)

    # Regional context
    region: str = "us-west-2"
    availability_zone: Optional[str] = None

    # Security context
    user_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: Set[str] = field(default_factory=set)

    # Session info
    session_id: str = field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    # Request context
    request_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if context is valid"""
        if self.expires_at and DeterministicClock.utcnow() > self.expires_at:
            return False

        if self.tenant and not self.tenant.active:
            return False

        return True

    def has_permission(self, permission: str) -> bool:
        """Check if context has permission"""
        return permission in self.permissions or "*" in self.permissions

    def check_quota(self, resource: str, amount: Any) -> bool:
        """Check if operation is within quota"""
        if not self.tenant:
            return True

        # Check against tenant quota
        # This is simplified - real implementation would track usage
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "tenant_id": self.tenant_id,
            "region": self.region,
            "user_id": self.user_id,
            "roles": self.roles,
            "permissions": list(self.permissions),
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class TenantIsolation:
    """Tenant isolation enforcement"""

    @staticmethod
    def get_namespace(tenant_id: str, isolation_level: IsolationLevel) -> str:
        """Get Kubernetes namespace for tenant"""
        if isolation_level == IsolationLevel.SHARED:
            return "greenlang-shared"
        elif isolation_level == IsolationLevel.NAMESPACE:
            return f"tenant-{tenant_id[:8]}"
        else:
            return f"tenant-{tenant_id}"

    @staticmethod
    def get_storage_path(tenant_id: str) -> Path:
        """Get isolated storage path for tenant"""
        base_path = Path("/data/tenants")
        return base_path / tenant_id

    @staticmethod
    def get_database_schema(tenant_id: str) -> str:
        """Get database schema for tenant"""
        return f"tenant_{tenant_id.replace('-', '_')}"

    @staticmethod
    def apply_network_policies(tenant_id: str, policies: List[Dict[str, Any]]):
        """Apply network isolation policies"""
        # This would integrate with Kubernetes NetworkPolicies
        logger.info(f"Applying network policies for tenant {tenant_id}")

    @staticmethod
    def create_resource_quota(tenant_id: str, quota: TenantQuota) -> Dict[str, Any]:
        """Create Kubernetes ResourceQuota for tenant"""
        return {
            "apiVersion": "v1",
            "kind": "ResourceQuota",
            "metadata": {
                "name": f"tenant-{tenant_id[:8]}-quota",
                "namespace": TenantIsolation.get_namespace(
                    tenant_id, IsolationLevel.NAMESPACE
                ),
            },
            "spec": {
                "hard": {
                    "requests.cpu": quota.max_cpu,
                    "requests.memory": quota.max_memory,
                    "persistentvolumeclaims": str(quota.max_datasets),
                    "pods": str(
                        quota.max_concurrent_pipelines * 5
                    ),  # Assuming 5 pods per pipeline
                }
            },
        }


class TenantManager:
    """Manager for multi-tenant operations"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize tenant manager

        Args:
            config: Configuration including:
                - jwt_secret: Secret for JWT tokens
                - jwt_algorithm: JWT algorithm (default: HS256)
                - token_expiry: Token expiry in seconds
                - storage_backend: Storage backend for tenant data
        """
        self.config = config or {}
        self.jwt_secret = self.config.get("jwt_secret", secrets.token_urlsafe(32))
        self.jwt_algorithm = self.config.get("jwt_algorithm", "HS256")
        self.token_expiry = self.config.get("token_expiry", 3600)

        # Tenant storage (in production, use database)
        self.tenants: Dict[str, Tenant] = {}
        self.contexts: Dict[str, TenantContext] = {}

        # Usage tracking
        self.usage_tracker: Dict[str, Dict[str, Any]] = {}

        logger.info("TenantManager initialized")

    def create_tenant(
        self, name: str, organization: str, tier: TenantTier = TenantTier.FREE, **kwargs
    ) -> Tenant:
        """
        Create a new tenant

        Args:
            name: Tenant name
            organization: Organization name
            tier: Subscription tier
            **kwargs: Additional tenant properties

        Returns:
            Created tenant
        """
        tenant = Tenant(
            name=name,
            organization=organization,
            tier=tier,
            quota=TenantQuota.from_tier(tier),
            **kwargs,
        )

        # Store tenant
        self.tenants[tenant.tenant_id] = tenant

        # Initialize usage tracking
        self.usage_tracker[tenant.tenant_id] = {
            "api_calls": 0,
            "pipelines_executed": 0,
            "storage_used": 0,
            "compute_hours": 0,
        }

        # Create isolation resources
        if tenant.isolation_level != IsolationLevel.SHARED:
            self._create_isolation_resources(tenant)

        logger.info(f"Created tenant: {tenant.tenant_id} ({tenant.name})")
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID"""
        return self.tenants.get(tenant_id)

    def update_tenant(
        self, tenant_id: str, updates: Dict[str, Any]
    ) -> Optional[Tenant]:
        """Update tenant properties"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return None

        # Update allowed fields
        allowed_fields = ["name", "tier", "quota", "settings", "active", "suspended"]
        for field in allowed_fields:
            if field in updates:
                setattr(tenant, field, updates[field])

        tenant.updated_at = DeterministicClock.utcnow()

        logger.info(f"Updated tenant: {tenant_id}")
        return tenant

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete tenant"""
        if tenant_id not in self.tenants:
            return False

        tenant = self.tenants[tenant_id]

        # Cleanup isolation resources
        if tenant.isolation_level != IsolationLevel.SHARED:
            self._cleanup_isolation_resources(tenant)

        # Remove tenant
        del self.tenants[tenant_id]

        # Remove usage tracking
        if tenant_id in self.usage_tracker:
            del self.usage_tracker[tenant_id]

        logger.info(f"Deleted tenant: {tenant_id}")
        return True

    def get_tenant_context(self, token: str) -> TenantContext:
        """
        Extract tenant context from auth token

        Args:
            token: JWT token

        Returns:
            TenantContext
        """
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT is required for token processing")

        try:
            # Decode JWT (in production, verify signature)
            claims = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                options={"verify_signature": True},
            )
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            # For development, decode without verification
            # SECURITY FIX: Verify JWT signature
            import os
            if os.getenv("GREENLANG_DEBUG_MODE") == "true":
                logger.warning("JWT verification disabled - DEBUG only!")
                claims = jwt.decode(token, options={"verify_signature": False})
            else:
                secret_key = os.getenv("JWT_SECRET_KEY", "default-secret-key-change-in-production")
                claims = jwt.decode(token, secret_key, algorithms=["HS256"])

        tenant_id = claims.get("tenant_id")
        if not tenant_id:
            raise ValueError("Token missing tenant_id")

        # Get tenant
        tenant = self.get_tenant(tenant_id)

        # Create context
        context = TenantContext(
            tenant_id=tenant_id,
            tenant=tenant,
            resource_limits=claims.get("limits", {}),
            allowed_packs=claims.get("packs", []),
            region=claims.get("region", "us-west-2"),
            user_id=claims.get("sub"),
            roles=claims.get("roles", []),
            permissions=set(claims.get("permissions", [])),
        )

        # Set expiry
        if "exp" in claims:
            context.expires_at = datetime.fromtimestamp(claims["exp"])

        # Cache context
        self.contexts[context.session_id] = context

        return context

    def create_token(
        self,
        tenant_id: str,
        user_id: str,
        roles: List[str] = None,
        permissions: List[str] = None,
    ) -> str:
        """
        Create JWT token for tenant

        Args:
            tenant_id: Tenant ID
            user_id: User ID
            roles: User roles
            permissions: User permissions

        Returns:
            JWT token
        """
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT is required for token creation")

        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant not found: {tenant_id}")

        # Prepare claims
        now = DeterministicClock.utcnow()
        claims = {
            "tenant_id": tenant_id,
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(seconds=self.token_expiry),
            "roles": roles or [],
            "permissions": permissions or [],
            "region": tenant.settings.region,
            "tier": tenant.tier.value,
            "limits": {"cpu": tenant.quota.max_cpu, "memory": tenant.quota.max_memory},
            "packs": tenant.quota.allowed_pack_categories,
        }

        # Create token
        token = jwt.encode(claims, self.jwt_secret, algorithm=self.jwt_algorithm)

        return token

    def validate_context(self, context: TenantContext) -> bool:
        """Validate tenant context"""
        if not context.is_valid():
            return False

        # Check if tenant exists and is active
        tenant = self.get_tenant(context.tenant_id)
        if not tenant or not tenant.active or tenant.suspended:
            return False

        return True

    def check_quota(self, tenant_id: str, resource: str, amount: Any) -> bool:
        """
        Check if operation is within tenant quota

        Args:
            tenant_id: Tenant ID
            resource: Resource type
            amount: Amount to check

        Returns:
            True if within quota
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False

        usage = self.usage_tracker.get(tenant_id, {})

        # Check different resource types
        if resource == "api_calls":
            current = usage.get("api_calls_today", 0)
            return current + amount <= tenant.quota.max_api_calls_per_day
        elif resource == "pipelines":
            current = usage.get("concurrent_pipelines", 0)
            return current + amount <= tenant.quota.max_concurrent_pipelines
        elif resource == "storage":
            current = usage.get("storage_used", 0)
            max_storage = self._parse_size(tenant.quota.max_storage)
            return current + amount <= max_storage

        return True

    def track_usage(self, tenant_id: str, resource: str, amount: Any):
        """Track resource usage for tenant"""
        if tenant_id not in self.usage_tracker:
            self.usage_tracker[tenant_id] = {}

        usage = self.usage_tracker[tenant_id]

        if resource == "api_calls":
            usage["api_calls"] = usage.get("api_calls", 0) + amount
            usage["api_calls_today"] = usage.get("api_calls_today", 0) + amount
        elif resource == "pipelines":
            usage["pipelines_executed"] = usage.get("pipelines_executed", 0) + amount
        elif resource == "storage":
            usage["storage_used"] = usage.get("storage_used", 0) + amount
        elif resource == "compute":
            usage["compute_hours"] = usage.get("compute_hours", 0) + amount

    def get_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get usage statistics for tenant"""
        return self.usage_tracker.get(tenant_id, {})

    def _create_isolation_resources(self, tenant: Tenant):
        """Create isolation resources for tenant"""
        # Create namespace
        namespace = TenantIsolation.get_namespace(
            tenant.tenant_id, tenant.isolation_level
        )
        logger.info(f"Creating namespace: {namespace}")

        # Create resource quota
        quota_manifest = TenantIsolation.create_resource_quota(
            tenant.tenant_id, tenant.quota
        )
        logger.info(f"Creating resource quota for tenant: {tenant.tenant_id}")

        # Create storage directories
        storage_path = TenantIsolation.get_storage_path(tenant.tenant_id)
        storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created storage path: {storage_path}")

    def _cleanup_isolation_resources(self, tenant: Tenant):
        """Cleanup isolation resources for tenant"""
        # Remove namespace
        namespace = TenantIsolation.get_namespace(
            tenant.tenant_id, tenant.isolation_level
        )
        logger.info(f"Removing namespace: {namespace}")

        # Remove storage
        storage_path = TenantIsolation.get_storage_path(tenant.tenant_id)
        if storage_path.exists():
            import shutil

            shutil.rmtree(storage_path)
            logger.info(f"Removed storage path: {storage_path}")

    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes"""
        units = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4}

        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                return int(size_str[: -len(unit)]) * multiplier

        return int(size_str)
