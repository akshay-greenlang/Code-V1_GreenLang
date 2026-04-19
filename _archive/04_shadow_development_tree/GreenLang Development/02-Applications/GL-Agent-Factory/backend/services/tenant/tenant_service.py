"""
Tenant Service - Multi-Tenancy Management

This module provides comprehensive tenant management services including:
- Tenant CRUD operations
- Tenant onboarding workflow
- Usage tracking and billing metrics
- Feature flag management
- Quota enforcement

The TenantService is the central service for all tenant-related operations
and ensures data isolation, proper validation, and audit logging.

Example:
    >>> service = TenantService(session_manager, cache, event_bus)
    >>> tenant = await service.create_tenant(TenantCreateInput(
    ...     name="Acme Corporation",
    ...     slug="acme-corp",
    ...     admin_email="admin@acme.com",
    ...     subscription_tier=SubscriptionTier.PRO
    ... ))
"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field, validator

from models.tenant import (
    Tenant,
    TenantStatus,
    TenantInvitation,
    TenantUsageLog,
    SubscriptionTier,
    DEFAULT_TIER_QUOTAS,
    DEFAULT_TIER_FEATURES,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================


class TenantServiceError(Exception):
    """Base exception for tenant service errors."""

    def __init__(self, message: str, code: str = "TENANT_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class TenantNotFoundError(TenantServiceError):
    """Raised when tenant is not found."""

    def __init__(self, tenant_id: str):
        super().__init__(
            message=f"Tenant '{tenant_id}' not found",
            code="TENANT_NOT_FOUND",
        )
        self.tenant_id = tenant_id


class TenantAlreadyExistsError(TenantServiceError):
    """Raised when tenant already exists."""

    def __init__(self, slug: str):
        super().__init__(
            message=f"Tenant with slug '{slug}' already exists",
            code="TENANT_ALREADY_EXISTS",
        )
        self.slug = slug


class QuotaExceededError(TenantServiceError):
    """Raised when quota is exceeded."""

    def __init__(self, quota_name: str, limit: int, current: int):
        super().__init__(
            message=f"Quota '{quota_name}' exceeded. Limit: {limit}, Current: {current}",
            code="QUOTA_EXCEEDED",
        )
        self.quota_name = quota_name
        self.limit = limit
        self.current = current


class TenantSuspendedError(TenantServiceError):
    """Raised when tenant is suspended."""

    def __init__(self, tenant_id: str, reason: Optional[str] = None):
        message = f"Tenant '{tenant_id}' is suspended"
        if reason:
            message += f": {reason}"
        super().__init__(message=message, code="TENANT_SUSPENDED")
        self.tenant_id = tenant_id
        self.reason = reason


class FeatureNotEnabledError(TenantServiceError):
    """Raised when feature is not enabled."""

    def __init__(self, feature_name: str, tier: SubscriptionTier):
        super().__init__(
            message=f"Feature '{feature_name}' is not available for {tier.value} tier",
            code="FEATURE_NOT_ENABLED",
        )
        self.feature_name = feature_name
        self.tier = tier


# =============================================================================
# Input/Output Models
# =============================================================================


class TenantCreateInput(BaseModel):
    """Input for creating a new tenant."""

    name: str = Field(..., min_length=2, max_length=255, description="Organization name")
    slug: str = Field(..., min_length=2, max_length=100, description="URL-safe slug")
    admin_email: EmailStr = Field(..., description="Admin user email")
    admin_name: Optional[str] = Field(None, description="Admin user name")
    subscription_tier: SubscriptionTier = Field(
        SubscriptionTier.FREE,
        description="Subscription tier",
    )
    domain: Optional[str] = Field(None, description="Custom domain")
    data_residency_region: str = Field(
        "us-east-1",
        description="Data residency region",
    )
    settings: Optional[Dict[str, Any]] = Field(None, description="Initial settings")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    start_trial: bool = Field(True, description="Start with trial period")
    trial_days: int = Field(14, ge=0, le=90, description="Trial period in days")

    @validator("slug")
    def validate_slug(cls, v: str) -> str:
        """Validate slug format."""
        import re

        if not re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$", v):
            raise ValueError(
                "Slug must be lowercase alphanumeric with hyphens, "
                "starting and ending with alphanumeric"
            )
        return v


class TenantUpdateInput(BaseModel):
    """Input for updating a tenant."""

    name: Optional[str] = Field(None, min_length=2, max_length=255)
    domain: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    billing_email: Optional[EmailStr] = None
    primary_contact_name: Optional[str] = None
    primary_contact_email: Optional[EmailStr] = None


class TenantOnboardingInput(BaseModel):
    """Input for tenant onboarding workflow."""

    name: str = Field(..., min_length=2, max_length=255)
    slug: str = Field(..., min_length=2, max_length=100)
    admin_email: EmailStr
    admin_name: str
    admin_password: Optional[str] = Field(None, min_length=8)
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    company_size: Optional[str] = None
    industry: Optional[str] = None
    use_case: Optional[str] = None
    referral_source: Optional[str] = None
    accept_terms: bool = Field(..., description="Must accept terms of service")
    terms_version: str = Field("1.0", description="Version of terms accepted")


class QuotaUpdateInput(BaseModel):
    """Input for updating tenant quotas."""

    agents: Optional[int] = Field(None, ge=0)
    executions_per_month: Optional[int] = Field(None, ge=0)
    storage_gb: Optional[int] = Field(None, ge=0)
    concurrent_executions: Optional[int] = Field(None, ge=0)
    api_calls_per_minute: Optional[int] = Field(None, ge=0)
    team_members: Optional[int] = Field(None, ge=-1)  # -1 for unlimited


class FeatureFlagUpdate(BaseModel):
    """Input for updating feature flags."""

    feature_name: str
    enabled: bool


class UsageMetric(BaseModel):
    """Usage metric data."""

    metric_name: str
    value: int
    period: str  # "daily" or "monthly"
    timestamp: datetime


class TenantListFilters(BaseModel):
    """Filters for listing tenants."""

    status: Optional[TenantStatus] = None
    subscription_tier: Optional[SubscriptionTier] = None
    search_query: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


class PaginationParams(BaseModel):
    """Pagination parameters."""

    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)
    sort_by: str = Field("created_at")
    sort_order: str = Field("desc")


# =============================================================================
# Tenant Service
# =============================================================================


class TenantService:
    """
    Tenant Service for multi-tenancy management.

    This service provides:
    - Tenant CRUD operations
    - Onboarding workflow
    - Usage tracking
    - Quota enforcement
    - Feature flag management
    - Billing integration

    Attributes:
        session_manager: Database session manager
        cache: Cache client (Redis)
        event_bus: Event bus for notifications

    Example:
        >>> service = TenantService(session_manager, cache, event_bus)
        >>> tenant = await service.create_tenant(input_data)
    """

    CACHE_TTL_SECONDS = 300  # 5 minutes
    INVITATION_EXPIRY_DAYS = 7

    def __init__(
        self,
        session_manager: Optional[Any] = None,
        cache: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        """
        Initialize the tenant service.

        Args:
            session_manager: TenantSessionManager for database access
            cache: Redis cache client
            event_bus: Event bus for publishing events
        """
        self.session_manager = session_manager
        self.cache = cache
        self.event_bus = event_bus

        # In-memory storage for development/testing
        self._tenants: Dict[str, Tenant] = {}
        self._users: Dict[str, Dict] = {}
        self._invitations: Dict[str, TenantInvitation] = {}

        logger.info("TenantService initialized")

    # =========================================================================
    # Tenant CRUD Operations
    # =========================================================================

    async def create_tenant(
        self,
        input_data: TenantCreateInput,
        created_by: Optional[str] = None,
    ) -> Tenant:
        """
        Create a new tenant.

        Args:
            input_data: Tenant creation input
            created_by: User ID who created the tenant

        Returns:
            Created Tenant object

        Raises:
            TenantAlreadyExistsError: If slug already exists
            TenantServiceError: If creation fails
        """
        start_time = datetime.utcnow()

        try:
            # Check for duplicate slug
            existing = await self.get_tenant_by_slug(input_data.slug)
            if existing:
                raise TenantAlreadyExistsError(input_data.slug)

            # Generate tenant ID
            tenant_id = f"t-{input_data.slug}"

            # Create tenant object
            tenant = Tenant()
            tenant.id = uuid4()
            tenant.tenant_id = tenant_id
            tenant.name = input_data.name
            tenant.slug = input_data.slug
            tenant.domain = input_data.domain
            tenant.subscription_tier = input_data.subscription_tier
            tenant.status = TenantStatus.PENDING
            tenant.is_active = False
            tenant.data_residency_region = input_data.data_residency_region
            tenant.settings = input_data.settings or {}
            tenant.metadata = input_data.metadata or {}
            tenant.quotas = {}  # Will use tier defaults
            tenant.current_usage = {
                "agents": 0,
                "executions_per_month": 0,
                "storage_gb": 0,
                "team_members": 1,  # Start with admin
            }
            tenant.feature_flags = {}  # Will use tier defaults
            tenant.created_at = datetime.utcnow()
            tenant.updated_at = datetime.utcnow()

            # Set trial period if applicable
            if input_data.start_trial and input_data.subscription_tier != SubscriptionTier.ENTERPRISE:
                tenant.is_trial = True
                tenant.trial_ends_at = datetime.utcnow() + timedelta(days=input_data.trial_days)

            # Persist tenant
            await self._persist_tenant(tenant)

            # Create admin user invitation
            await self._create_admin_invitation(
                tenant=tenant,
                email=input_data.admin_email,
                name=input_data.admin_name,
            )

            # Publish event
            await self._publish_event(
                "tenant.created",
                {
                    "tenant_id": tenant.tenant_id,
                    "name": tenant.name,
                    "tier": tenant.subscription_tier.value,
                    "created_by": created_by,
                },
            )

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.info(
                f"Created tenant {tenant.tenant_id} in {processing_time:.2f}ms",
                extra={
                    "tenant_id": tenant.tenant_id,
                    "tier": tenant.subscription_tier.value,
                    "processing_time_ms": processing_time,
                },
            )

            return tenant

        except TenantAlreadyExistsError:
            raise
        except Exception as e:
            logger.error(f"Failed to create tenant: {e}", exc_info=True)
            raise TenantServiceError(f"Failed to create tenant: {str(e)}")

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """
        Get tenant by tenant_id.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant object or None if not found
        """
        # Check cache first
        cached = await self._get_cached_tenant(tenant_id)
        if cached:
            return cached

        # Load from database
        tenant = await self._load_tenant(tenant_id)

        if tenant:
            await self._cache_tenant(tenant)

        return tenant

    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """
        Get tenant by slug.

        Args:
            slug: Tenant slug

        Returns:
            Tenant object or None if not found
        """
        tenant_id = f"t-{slug}"
        return await self.get_tenant(tenant_id)

    async def get_tenant_by_id(self, id: UUID) -> Optional[Tenant]:
        """
        Get tenant by UUID.

        Args:
            id: Tenant UUID

        Returns:
            Tenant object or None if not found
        """
        # In a real implementation, this would query by UUID
        for tenant in self._tenants.values():
            if tenant.id == id:
                return tenant
        return None

    async def update_tenant(
        self,
        tenant_id: str,
        input_data: TenantUpdateInput,
        updated_by: Optional[str] = None,
    ) -> Tenant:
        """
        Update tenant information.

        Args:
            tenant_id: Tenant identifier
            input_data: Update data
            updated_by: User ID who made the update

        Returns:
            Updated Tenant object

        Raises:
            TenantNotFoundError: If tenant not found
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        # Apply updates
        if input_data.name is not None:
            tenant.name = input_data.name
        if input_data.domain is not None:
            tenant.domain = input_data.domain
        if input_data.settings is not None:
            tenant.settings = {**tenant.settings, **input_data.settings}
        if input_data.metadata is not None:
            tenant.metadata = {**tenant.metadata, **input_data.metadata}
        if input_data.billing_email is not None:
            tenant.billing_email = input_data.billing_email
        if input_data.primary_contact_name is not None:
            tenant.primary_contact_name = input_data.primary_contact_name
        if input_data.primary_contact_email is not None:
            tenant.primary_contact_email = input_data.primary_contact_email

        tenant.updated_at = datetime.utcnow()

        # Persist changes
        await self._persist_tenant(tenant)

        # Invalidate cache
        await self._invalidate_tenant_cache(tenant_id)

        # Publish event
        await self._publish_event(
            "tenant.updated",
            {
                "tenant_id": tenant_id,
                "updated_by": updated_by,
                "fields_updated": list(input_data.dict(exclude_unset=True).keys()),
            },
        )

        logger.info(f"Updated tenant {tenant_id}")
        return tenant

    async def delete_tenant(
        self,
        tenant_id: str,
        deleted_by: Optional[str] = None,
        hard_delete: bool = False,
    ) -> bool:
        """
        Delete a tenant (soft or hard delete).

        Args:
            tenant_id: Tenant identifier
            deleted_by: User ID who deleted
            hard_delete: If True, permanently delete

        Returns:
            True if deleted successfully

        Raises:
            TenantNotFoundError: If tenant not found
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        if hard_delete:
            # Permanent deletion
            await self._delete_tenant(tenant_id)
            logger.warning(f"Hard deleted tenant {tenant_id} by {deleted_by}")
        else:
            # Soft delete - deactivate
            tenant.status = TenantStatus.DEACTIVATED
            tenant.is_active = False
            tenant.updated_at = datetime.utcnow()
            await self._persist_tenant(tenant)
            logger.info(f"Soft deleted tenant {tenant_id} by {deleted_by}")

        # Invalidate cache
        await self._invalidate_tenant_cache(tenant_id)

        # Publish event
        await self._publish_event(
            "tenant.deleted",
            {
                "tenant_id": tenant_id,
                "deleted_by": deleted_by,
                "hard_delete": hard_delete,
            },
        )

        return True

    async def list_tenants(
        self,
        filters: Optional[TenantListFilters] = None,
        pagination: Optional[PaginationParams] = None,
    ) -> Tuple[List[Tenant], int]:
        """
        List tenants with filtering and pagination.

        Args:
            filters: Filter criteria
            pagination: Pagination parameters

        Returns:
            Tuple of (list of tenants, total count)
        """
        filters = filters or TenantListFilters()
        pagination = pagination or PaginationParams()

        # Get all tenants
        tenants = list(self._tenants.values())

        # Apply filters
        if filters.status:
            tenants = [t for t in tenants if t.status == filters.status]
        if filters.subscription_tier:
            tenants = [t for t in tenants if t.subscription_tier == filters.subscription_tier]
        if filters.search_query:
            query = filters.search_query.lower()
            tenants = [
                t for t in tenants
                if query in t.name.lower() or query in t.slug.lower()
            ]
        if filters.created_after:
            tenants = [t for t in tenants if t.created_at >= filters.created_after]
        if filters.created_before:
            tenants = [t for t in tenants if t.created_at <= filters.created_before]

        # Get total count before pagination
        total_count = len(tenants)

        # Sort
        reverse = pagination.sort_order == "desc"
        tenants.sort(
            key=lambda t: getattr(t, pagination.sort_by, t.created_at),
            reverse=reverse,
        )

        # Paginate
        start = pagination.offset
        end = start + pagination.limit
        tenants = tenants[start:end]

        return tenants, total_count

    # =========================================================================
    # Tenant Status Management
    # =========================================================================

    async def activate_tenant(
        self,
        tenant_id: str,
        activated_by: Optional[str] = None,
    ) -> Tenant:
        """
        Activate a pending tenant.

        Args:
            tenant_id: Tenant identifier
            activated_by: User ID who activated

        Returns:
            Activated tenant

        Raises:
            TenantNotFoundError: If tenant not found
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        tenant.status = TenantStatus.ACTIVE
        tenant.is_active = True
        tenant.activated_at = datetime.utcnow()
        tenant.updated_at = datetime.utcnow()

        await self._persist_tenant(tenant)
        await self._invalidate_tenant_cache(tenant_id)

        await self._publish_event(
            "tenant.activated",
            {
                "tenant_id": tenant_id,
                "activated_by": activated_by,
            },
        )

        logger.info(f"Activated tenant {tenant_id}")
        return tenant

    async def suspend_tenant(
        self,
        tenant_id: str,
        reason: str,
        suspended_by: Optional[str] = None,
    ) -> Tenant:
        """
        Suspend a tenant.

        Args:
            tenant_id: Tenant identifier
            reason: Suspension reason
            suspended_by: User ID who suspended

        Returns:
            Suspended tenant

        Raises:
            TenantNotFoundError: If tenant not found
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        tenant.status = TenantStatus.SUSPENDED
        tenant.is_active = False
        tenant.suspended_at = datetime.utcnow()
        tenant.suspension_reason = reason
        tenant.updated_at = datetime.utcnow()

        await self._persist_tenant(tenant)
        await self._invalidate_tenant_cache(tenant_id)

        await self._publish_event(
            "tenant.suspended",
            {
                "tenant_id": tenant_id,
                "reason": reason,
                "suspended_by": suspended_by,
            },
        )

        logger.warning(f"Suspended tenant {tenant_id}: {reason}")
        return tenant

    async def reactivate_tenant(
        self,
        tenant_id: str,
        reactivated_by: Optional[str] = None,
    ) -> Tenant:
        """
        Reactivate a suspended tenant.

        Args:
            tenant_id: Tenant identifier
            reactivated_by: User ID who reactivated

        Returns:
            Reactivated tenant

        Raises:
            TenantNotFoundError: If tenant not found
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        tenant.status = TenantStatus.ACTIVE
        tenant.is_active = True
        tenant.suspended_at = None
        tenant.suspension_reason = None
        tenant.updated_at = datetime.utcnow()

        await self._persist_tenant(tenant)
        await self._invalidate_tenant_cache(tenant_id)

        await self._publish_event(
            "tenant.reactivated",
            {
                "tenant_id": tenant_id,
                "reactivated_by": reactivated_by,
            },
        )

        logger.info(f"Reactivated tenant {tenant_id}")
        return tenant

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def upgrade_subscription(
        self,
        tenant_id: str,
        new_tier: SubscriptionTier,
        upgraded_by: Optional[str] = None,
    ) -> Tenant:
        """
        Upgrade tenant subscription tier.

        Args:
            tenant_id: Tenant identifier
            new_tier: New subscription tier
            upgraded_by: User ID who upgraded

        Returns:
            Updated tenant

        Raises:
            TenantNotFoundError: If tenant not found
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        old_tier = tenant.subscription_tier
        tenant.subscription_tier = new_tier
        tenant.updated_at = datetime.utcnow()

        # End trial if upgrading from trial
        if tenant.is_trial:
            tenant.is_trial = False
            tenant.trial_ends_at = None

        await self._persist_tenant(tenant)
        await self._invalidate_tenant_cache(tenant_id)

        await self._publish_event(
            "tenant.subscription_upgraded",
            {
                "tenant_id": tenant_id,
                "old_tier": old_tier.value,
                "new_tier": new_tier.value,
                "upgraded_by": upgraded_by,
            },
        )

        logger.info(f"Upgraded tenant {tenant_id} from {old_tier.value} to {new_tier.value}")
        return tenant

    async def downgrade_subscription(
        self,
        tenant_id: str,
        new_tier: SubscriptionTier,
        downgraded_by: Optional[str] = None,
    ) -> Tenant:
        """
        Downgrade tenant subscription tier.

        Args:
            tenant_id: Tenant identifier
            new_tier: New subscription tier
            downgraded_by: User ID who downgraded

        Returns:
            Updated tenant

        Raises:
            TenantNotFoundError: If tenant not found
            TenantServiceError: If current usage exceeds new tier limits
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        # Check if current usage exceeds new tier limits
        new_tier_quotas = DEFAULT_TIER_QUOTAS[new_tier]
        current_usage = tenant.current_usage or {}

        for quota_name, limit in new_tier_quotas.items():
            if limit == -1:  # Unlimited
                continue
            current = current_usage.get(quota_name, 0)
            if current > limit:
                raise TenantServiceError(
                    f"Cannot downgrade: {quota_name} usage ({current}) exceeds "
                    f"{new_tier.value} tier limit ({limit})"
                )

        old_tier = tenant.subscription_tier
        tenant.subscription_tier = new_tier
        tenant.updated_at = datetime.utcnow()

        await self._persist_tenant(tenant)
        await self._invalidate_tenant_cache(tenant_id)

        await self._publish_event(
            "tenant.subscription_downgraded",
            {
                "tenant_id": tenant_id,
                "old_tier": old_tier.value,
                "new_tier": new_tier.value,
                "downgraded_by": downgraded_by,
            },
        )

        logger.info(f"Downgraded tenant {tenant_id} from {old_tier.value} to {new_tier.value}")
        return tenant

    # =========================================================================
    # Quota Management
    # =========================================================================

    async def update_quotas(
        self,
        tenant_id: str,
        quota_updates: QuotaUpdateInput,
        updated_by: Optional[str] = None,
    ) -> Tenant:
        """
        Update tenant quotas (custom overrides).

        Args:
            tenant_id: Tenant identifier
            quota_updates: Quota updates
            updated_by: User ID who updated

        Returns:
            Updated tenant

        Raises:
            TenantNotFoundError: If tenant not found
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        # Update quotas
        update_dict = quota_updates.dict(exclude_unset=True)
        if tenant.quotas is None:
            tenant.quotas = {}
        tenant.quotas.update(update_dict)
        tenant.updated_at = datetime.utcnow()

        await self._persist_tenant(tenant)
        await self._invalidate_tenant_cache(tenant_id)

        await self._publish_event(
            "tenant.quotas_updated",
            {
                "tenant_id": tenant_id,
                "updates": update_dict,
                "updated_by": updated_by,
            },
        )

        logger.info(f"Updated quotas for tenant {tenant_id}: {update_dict}")
        return tenant

    async def check_quota(
        self,
        tenant_id: str,
        quota_name: str,
        increment: int = 1,
    ) -> Tuple[bool, int, int]:
        """
        Check if operation would exceed quota.

        Args:
            tenant_id: Tenant identifier
            quota_name: Quota to check
            increment: Amount to add

        Returns:
            Tuple of (allowed, limit, current_usage)

        Raises:
            TenantNotFoundError: If tenant not found
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        limit = tenant.get_quota(quota_name)
        current = tenant.get_current_usage(quota_name)

        # -1 means unlimited
        if limit == -1:
            return True, limit, current

        allowed = (current + increment) <= limit
        return allowed, limit, current

    async def increment_usage(
        self,
        tenant_id: str,
        metric_name: str,
        increment: int = 1,
        check_quota: bool = True,
    ) -> int:
        """
        Increment usage counter.

        Args:
            tenant_id: Tenant identifier
            metric_name: Metric to increment
            increment: Amount to add
            check_quota: If True, check quota before incrementing

        Returns:
            New usage value

        Raises:
            TenantNotFoundError: If tenant not found
            QuotaExceededError: If quota would be exceeded
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        # Check quota if required
        if check_quota:
            allowed, limit, current = await self.check_quota(
                tenant_id, metric_name, increment
            )
            if not allowed:
                raise QuotaExceededError(metric_name, limit, current)

        # Increment usage
        if tenant.current_usage is None:
            tenant.current_usage = {}
        current = tenant.current_usage.get(metric_name, 0)
        new_value = current + increment
        tenant.current_usage[metric_name] = new_value
        tenant.updated_at = datetime.utcnow()

        await self._persist_tenant(tenant)

        # Log usage for billing
        await self._log_usage(
            tenant_id=tenant.id,
            metric_name=metric_name,
            value=increment,
        )

        return new_value

    async def reset_monthly_usage(
        self,
        tenant_id: Optional[str] = None,
    ) -> int:
        """
        Reset monthly usage counters.

        Args:
            tenant_id: Optional specific tenant (None for all)

        Returns:
            Number of tenants reset
        """
        tenants = []
        if tenant_id:
            tenant = await self.get_tenant(tenant_id)
            if tenant:
                tenants = [tenant]
        else:
            tenants = list(self._tenants.values())

        monthly_metrics = ["executions_per_month", "api_calls_per_minute"]

        count = 0
        for tenant in tenants:
            for metric in monthly_metrics:
                if tenant.current_usage and metric in tenant.current_usage:
                    tenant.current_usage[metric] = 0
            tenant.usage_reset_at = datetime.utcnow()
            tenant.updated_at = datetime.utcnow()
            await self._persist_tenant(tenant)
            await self._invalidate_tenant_cache(tenant.tenant_id)
            count += 1

        logger.info(f"Reset monthly usage for {count} tenants")
        return count

    # =========================================================================
    # Feature Flag Management
    # =========================================================================

    async def update_feature_flag(
        self,
        tenant_id: str,
        feature_name: str,
        enabled: bool,
        updated_by: Optional[str] = None,
    ) -> Tenant:
        """
        Update a feature flag for a tenant.

        Args:
            tenant_id: Tenant identifier
            feature_name: Feature to update
            enabled: Enable or disable
            updated_by: User ID who updated

        Returns:
            Updated tenant

        Raises:
            TenantNotFoundError: If tenant not found
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        if tenant.feature_flags is None:
            tenant.feature_flags = {}
        tenant.feature_flags[feature_name] = enabled
        tenant.updated_at = datetime.utcnow()

        await self._persist_tenant(tenant)
        await self._invalidate_tenant_cache(tenant_id)

        await self._publish_event(
            "tenant.feature_flag_updated",
            {
                "tenant_id": tenant_id,
                "feature_name": feature_name,
                "enabled": enabled,
                "updated_by": updated_by,
            },
        )

        logger.info(f"Updated feature flag {feature_name}={enabled} for tenant {tenant_id}")
        return tenant

    async def check_feature(
        self,
        tenant_id: str,
        feature_name: str,
    ) -> bool:
        """
        Check if a feature is enabled for a tenant.

        Args:
            tenant_id: Tenant identifier
            feature_name: Feature to check

        Returns:
            True if feature is enabled

        Raises:
            TenantNotFoundError: If tenant not found
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        return tenant.is_feature_enabled(feature_name)

    # =========================================================================
    # Onboarding Workflow
    # =========================================================================

    async def onboard_tenant(
        self,
        input_data: TenantOnboardingInput,
    ) -> Tuple[Tenant, str]:
        """
        Complete tenant onboarding workflow.

        This method:
        1. Validates input and terms acceptance
        2. Creates tenant
        3. Creates admin user
        4. Sends welcome email
        5. Activates tenant

        Args:
            input_data: Onboarding input data

        Returns:
            Tuple of (tenant, admin_user_id)

        Raises:
            TenantServiceError: If onboarding fails
        """
        if not input_data.accept_terms:
            raise TenantServiceError("Must accept terms of service", "TERMS_NOT_ACCEPTED")

        try:
            # Create tenant
            tenant = await self.create_tenant(
                TenantCreateInput(
                    name=input_data.name,
                    slug=input_data.slug,
                    admin_email=input_data.admin_email,
                    admin_name=input_data.admin_name,
                    subscription_tier=input_data.subscription_tier,
                    metadata={
                        "company_size": input_data.company_size,
                        "industry": input_data.industry,
                        "use_case": input_data.use_case,
                        "referral_source": input_data.referral_source,
                    },
                )
            )

            # Record terms acceptance
            tenant.accepted_terms_version = input_data.terms_version
            tenant.accepted_terms_at = datetime.utcnow()

            # Create admin user
            admin_user_id = await self._create_admin_user(
                tenant=tenant,
                email=input_data.admin_email,
                name=input_data.admin_name,
                password=input_data.admin_password,
            )

            # Activate tenant
            await self.activate_tenant(tenant.tenant_id)

            # Send welcome email
            await self._send_welcome_email(
                tenant=tenant,
                admin_email=input_data.admin_email,
            )

            await self._publish_event(
                "tenant.onboarded",
                {
                    "tenant_id": tenant.tenant_id,
                    "admin_user_id": admin_user_id,
                },
            )

            logger.info(f"Completed onboarding for tenant {tenant.tenant_id}")
            return tenant, admin_user_id

        except TenantAlreadyExistsError:
            raise
        except Exception as e:
            logger.error(f"Onboarding failed: {e}", exc_info=True)
            raise TenantServiceError(f"Onboarding failed: {str(e)}")

    # =========================================================================
    # Usage and Billing
    # =========================================================================

    async def get_usage_summary(
        self,
        tenant_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get usage summary for billing.

        Args:
            tenant_id: Tenant identifier
            start_date: Start of period (default: start of current month)
            end_date: End of period (default: now)

        Returns:
            Usage summary dictionary
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        # Default to current month
        if not start_date:
            now = datetime.utcnow()
            start_date = datetime(now.year, now.month, 1)
        if not end_date:
            end_date = datetime.utcnow()

        # Get current usage
        current_usage = tenant.current_usage or {}
        quotas = tenant.get_effective_quotas()

        # Calculate usage percentages
        usage_summary = {}
        for metric, limit in quotas.items():
            current = current_usage.get(metric, 0)
            usage_summary[metric] = {
                "current": current,
                "limit": limit,
                "percentage": (current / limit * 100) if limit > 0 else 0,
                "remaining": limit - current if limit >= 0 else -1,
            }

        return {
            "tenant_id": tenant_id,
            "subscription_tier": tenant.subscription_tier.value,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "usage": usage_summary,
            "is_trial": tenant.is_trial,
            "trial_ends_at": tenant.trial_ends_at.isoformat() if tenant.trial_ends_at else None,
        }

    async def get_billing_metrics(
        self,
        tenant_id: str,
    ) -> Dict[str, Any]:
        """
        Get billing-related metrics.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Billing metrics dictionary
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise TenantNotFoundError(tenant_id)

        current_usage = tenant.current_usage or {}

        return {
            "tenant_id": tenant_id,
            "subscription_tier": tenant.subscription_tier.value,
            "stripe_customer_id": tenant.stripe_customer_id,
            "stripe_subscription_id": tenant.stripe_subscription_id,
            "billing_email": tenant.billing_email,
            "is_trial": tenant.is_trial,
            "trial_ends_at": tenant.trial_ends_at.isoformat() if tenant.trial_ends_at else None,
            "usage_this_month": {
                "executions": current_usage.get("executions_per_month", 0),
                "storage_gb": current_usage.get("storage_gb", 0),
                "team_members": current_usage.get("team_members", 0),
            },
        }

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    async def _persist_tenant(self, tenant: Tenant) -> None:
        """Persist tenant to database."""
        if self.session_manager:
            async with self.session_manager.system_session() as session:
                session.add(tenant)
                await session.commit()
        else:
            # In-memory storage for development
            self._tenants[tenant.tenant_id] = tenant

    async def _load_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Load tenant from database."""
        if self.session_manager:
            async with self.session_manager.system_session() as session:
                from sqlalchemy import select
                result = await session.execute(
                    select(Tenant).where(Tenant.tenant_id == tenant_id)
                )
                return result.scalar_one_or_none()
        else:
            return self._tenants.get(tenant_id)

    async def _delete_tenant(self, tenant_id: str) -> None:
        """Delete tenant from database."""
        if self.session_manager:
            async with self.session_manager.system_session() as session:
                from sqlalchemy import delete
                await session.execute(
                    delete(Tenant).where(Tenant.tenant_id == tenant_id)
                )
                await session.commit()
        else:
            if tenant_id in self._tenants:
                del self._tenants[tenant_id]

    async def _get_cached_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant from cache."""
        if not self.cache:
            return None

        try:
            cache_key = f"tenant:{tenant_id}"
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                # Deserialize tenant
                return self._deserialize_tenant(cached_data)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return None

    async def _cache_tenant(self, tenant: Tenant) -> None:
        """Cache tenant."""
        if not self.cache:
            return

        try:
            cache_key = f"tenant:{tenant.tenant_id}"
            await self.cache.setex(
                cache_key,
                self.CACHE_TTL_SECONDS,
                self._serialize_tenant(tenant),
            )
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    async def _invalidate_tenant_cache(self, tenant_id: str) -> None:
        """Invalidate tenant cache."""
        if not self.cache:
            return

        try:
            cache_key = f"tenant:{tenant_id}"
            await self.cache.delete(cache_key)
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")

    def _serialize_tenant(self, tenant: Tenant) -> str:
        """Serialize tenant for caching."""
        import json
        return json.dumps(tenant.to_dict())

    def _deserialize_tenant(self, data: str) -> Optional[Tenant]:
        """Deserialize tenant from cache."""
        import json
        try:
            tenant_dict = json.loads(data)
            tenant = Tenant()
            tenant.id = UUID(tenant_dict["id"])
            tenant.tenant_id = tenant_dict["tenant_id"]
            tenant.name = tenant_dict["name"]
            tenant.slug = tenant_dict["slug"]
            tenant.status = TenantStatus(tenant_dict["status"])
            tenant.subscription_tier = SubscriptionTier(tenant_dict["subscription_tier"])
            tenant.is_active = tenant_dict["is_active"]
            tenant.settings = tenant_dict.get("settings", {})
            tenant.quotas = tenant_dict.get("quotas", {})
            tenant.current_usage = tenant_dict.get("current_usage", {})
            tenant.feature_flags = tenant_dict.get("feature_flags", {})
            tenant.created_at = datetime.fromisoformat(tenant_dict["created_at"])
            tenant.updated_at = datetime.fromisoformat(tenant_dict["updated_at"])
            return tenant
        except Exception as e:
            logger.warning(f"Tenant deserialization error: {e}")
            return None

    async def _publish_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Publish event to event bus."""
        if not self.event_bus:
            logger.debug(f"Event (no bus): {event_type} - {payload}")
            return

        try:
            await self.event_bus.publish(event_type, payload)
        except Exception as e:
            logger.warning(f"Event publish error: {e}")

    async def _log_usage(
        self,
        tenant_id: UUID,
        metric_name: str,
        value: int,
    ) -> None:
        """Log usage for billing and analytics."""
        now = datetime.utcnow()

        log_entry = TenantUsageLog()
        log_entry.id = uuid4()
        log_entry.tenant_id = tenant_id
        log_entry.metric_name = metric_name
        log_entry.metric_value = value
        log_entry.period_start = datetime(now.year, now.month, 1)
        log_entry.period_end = now
        log_entry.recorded_at = now

        # In production, persist to database
        logger.debug(f"Usage logged: {tenant_id}/{metric_name}={value}")

    async def _create_admin_invitation(
        self,
        tenant: Tenant,
        email: str,
        name: Optional[str] = None,
    ) -> TenantInvitation:
        """Create admin user invitation."""
        invitation = TenantInvitation()
        invitation.id = uuid4()
        invitation.tenant_id = tenant.id
        invitation.email = email
        invitation.role = "admin"
        invitation.token = secrets.token_urlsafe(32)
        invitation.invited_by = tenant.id  # Self-invited during creation
        invitation.expires_at = datetime.utcnow() + timedelta(days=self.INVITATION_EXPIRY_DAYS)
        invitation.created_at = datetime.utcnow()

        self._invitations[invitation.token] = invitation
        logger.info(f"Created admin invitation for {email}")
        return invitation

    async def _create_admin_user(
        self,
        tenant: Tenant,
        email: str,
        name: str,
        password: Optional[str] = None,
    ) -> str:
        """Create admin user for tenant."""
        user_id = f"user-{uuid4()}"

        # In production, this would create a User record
        self._users[user_id] = {
            "id": user_id,
            "tenant_id": str(tenant.id),
            "email": email,
            "name": name,
            "roles": ["admin"],
            "created_at": datetime.utcnow().isoformat(),
        }

        logger.info(f"Created admin user {user_id} for tenant {tenant.tenant_id}")
        return user_id

    async def _send_welcome_email(
        self,
        tenant: Tenant,
        admin_email: str,
    ) -> None:
        """Send welcome email to admin."""
        # In production, integrate with email service
        logger.info(f"Welcome email sent to {admin_email} for tenant {tenant.tenant_id}")

    def calculate_provenance_hash(
        self,
        tenant: Tenant,
        operation: str,
    ) -> str:
        """
        Calculate SHA-256 hash for audit trail.

        Args:
            tenant: Tenant object
            operation: Operation performed

        Returns:
            Provenance hash string
        """
        data_str = (
            f"{tenant.tenant_id}:{operation}:{tenant.updated_at.isoformat()}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()
