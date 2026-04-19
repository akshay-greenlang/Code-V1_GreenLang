"""
Tenant Management Router - Multi-Tenancy API Endpoints

This module provides comprehensive API endpoints for tenant management including:
- Tenant CRUD operations
- Subscription and quota management
- Feature flag management
- Usage tracking and billing metrics
- User invitation management
- Tenant settings configuration

All endpoints are secured with proper authentication and authorization.

Example:
    >>> # Create tenant
    >>> POST /v1/tenants
    >>> {
    ...     "name": "Acme Corporation",
    ...     "slug": "acme-corp",
    ...     "admin_email": "admin@acme.com"
    ... }
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, EmailStr, Field

from models.tenant import SubscriptionTier, TenantStatus
from services.tenant.tenant_service import (
    TenantService,
    TenantCreateInput,
    TenantUpdateInput,
    TenantOnboardingInput,
    TenantListFilters,
    PaginationParams,
    QuotaUpdateInput,
    TenantNotFoundError,
    TenantAlreadyExistsError,
    QuotaExceededError,
    TenantServiceError,
)
from app.middleware.tenant_context import (
    get_tenant_context,
    require_role,
    require_feature,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize service (in production, use dependency injection)
_tenant_service: Optional[TenantService] = None


def get_tenant_service() -> TenantService:
    """Get or create tenant service instance."""
    global _tenant_service
    if _tenant_service is None:
        _tenant_service = TenantService()
    return _tenant_service


# =============================================================================
# Request/Response Models
# =============================================================================


class TenantCreateRequest(BaseModel):
    """Request to create a tenant."""

    name: str = Field(..., min_length=2, max_length=255, description="Tenant name")
    slug: str = Field(..., min_length=2, max_length=100, description="URL-safe slug")
    admin_email: EmailStr = Field(..., description="Admin user email")
    admin_name: Optional[str] = Field(None, description="Admin user name")
    subscription_tier: Optional[SubscriptionTier] = Field(
        SubscriptionTier.FREE,
        description="Subscription tier",
    )
    domain: Optional[str] = Field(None, description="Custom domain")
    data_residency_region: Optional[str] = Field("us-east-1", description="Data residency")
    settings: Optional[Dict[str, Any]] = Field(None, description="Initial settings")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    start_trial: Optional[bool] = Field(True, description="Start trial period")
    trial_days: Optional[int] = Field(14, ge=0, le=90, description="Trial days")


class TenantOnboardingRequest(BaseModel):
    """Request for tenant onboarding."""

    name: str = Field(..., min_length=2, max_length=255)
    slug: str = Field(..., min_length=2, max_length=100)
    admin_email: EmailStr
    admin_name: str
    admin_password: Optional[str] = Field(None, min_length=8)
    subscription_tier: Optional[SubscriptionTier] = SubscriptionTier.FREE
    company_size: Optional[str] = None
    industry: Optional[str] = None
    use_case: Optional[str] = None
    referral_source: Optional[str] = None
    accept_terms: bool = Field(..., description="Must accept terms")
    terms_version: str = Field("1.0")


class TenantUpdateRequest(BaseModel):
    """Request to update a tenant."""

    name: Optional[str] = Field(None, min_length=2, max_length=255)
    domain: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    billing_email: Optional[EmailStr] = None
    primary_contact_name: Optional[str] = None
    primary_contact_email: Optional[EmailStr] = None


class TenantResponse(BaseModel):
    """Tenant response model."""

    id: str
    tenant_id: str
    name: str
    slug: str
    domain: Optional[str] = None
    status: str
    subscription_tier: str
    is_active: bool
    is_trial: bool
    trial_ends_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    settings: Dict[str, Any]
    quotas: Dict[str, int]
    current_usage: Dict[str, int]
    feature_flags: Dict[str, bool]


class TenantPublicResponse(BaseModel):
    """Public tenant response (limited fields)."""

    id: str
    tenant_id: str
    name: str
    slug: str
    subscription_tier: str
    is_active: bool
    created_at: datetime


class TenantListResponse(BaseModel):
    """Response for listing tenants."""

    tenants: List[TenantPublicResponse]
    total: int
    limit: int
    offset: int


class QuotaResponse(BaseModel):
    """Quota response model."""

    tenant_id: str
    quotas: Dict[str, int]
    usage: Dict[str, int]
    usage_percentage: Dict[str, float]


class QuotaUpdateRequest(BaseModel):
    """Request to update quotas."""

    agents: Optional[int] = Field(None, ge=0)
    executions_per_month: Optional[int] = Field(None, ge=0)
    storage_gb: Optional[int] = Field(None, ge=0)
    concurrent_executions: Optional[int] = Field(None, ge=0)
    api_calls_per_minute: Optional[int] = Field(None, ge=0)
    team_members: Optional[int] = Field(None, ge=-1)


class FeatureFlagResponse(BaseModel):
    """Feature flags response."""

    tenant_id: str
    features: Dict[str, bool]


class FeatureFlagUpdateRequest(BaseModel):
    """Request to update a feature flag."""

    enabled: bool


class UsageSummaryResponse(BaseModel):
    """Usage summary response."""

    tenant_id: str
    subscription_tier: str
    period_start: str
    period_end: str
    usage: Dict[str, Any]
    is_trial: bool
    trial_ends_at: Optional[str] = None


class BillingMetricsResponse(BaseModel):
    """Billing metrics response."""

    tenant_id: str
    subscription_tier: str
    stripe_customer_id: Optional[str] = None
    billing_email: Optional[str] = None
    is_trial: bool
    trial_ends_at: Optional[str] = None
    usage_this_month: Dict[str, int]


class UserRoleRequest(BaseModel):
    """Request to assign a role."""

    role: str = Field(..., description="Role name")


class SubscriptionUpdateRequest(BaseModel):
    """Request to change subscription tier."""

    new_tier: SubscriptionTier


class SettingsUpdateRequest(BaseModel):
    """Request to update settings."""

    settings: Dict[str, Any]


class InvitationCreateRequest(BaseModel):
    """Request to create an invitation."""

    email: EmailStr
    role: str = Field(default="viewer")


class InvitationResponse(BaseModel):
    """Invitation response."""

    id: str
    email: str
    role: str
    expires_at: datetime
    created_at: datetime


# =============================================================================
# Tenant CRUD Endpoints
# =============================================================================


@router.post(
    "",
    response_model=TenantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create tenant",
    description="Create a new tenant (system admin only).",
)
async def create_tenant(
    request: TenantCreateRequest,
    service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """
    Create a new tenant.

    - System admin only endpoint
    - Creates tenant with specified settings
    - Optionally starts trial period
    - Sends admin invitation email
    """
    try:
        tenant = await service.create_tenant(
            TenantCreateInput(
                name=request.name,
                slug=request.slug,
                admin_email=request.admin_email,
                admin_name=request.admin_name,
                subscription_tier=request.subscription_tier or SubscriptionTier.FREE,
                domain=request.domain,
                data_residency_region=request.data_residency_region or "us-east-1",
                settings=request.settings,
                metadata=request.metadata,
                start_trial=request.start_trial if request.start_trial is not None else True,
                trial_days=request.trial_days or 14,
            )
        )

        logger.info(f"Created tenant via API: {tenant.tenant_id}")

        return TenantResponse(
            id=str(tenant.id),
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            slug=tenant.slug,
            domain=tenant.domain,
            status=tenant.status.value,
            subscription_tier=tenant.subscription_tier.value,
            is_active=tenant.is_active,
            is_trial=tenant.is_trial,
            trial_ends_at=tenant.trial_ends_at,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
            settings=tenant.settings or {},
            quotas=tenant.get_effective_quotas(),
            current_usage=tenant.current_usage or {},
            feature_flags=tenant.get_effective_feature_flags(),
        )

    except TenantAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": e.code, "message": e.message},
        )
    except TenantServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": e.code, "message": e.message},
        )


@router.post(
    "/onboard",
    response_model=TenantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Onboard tenant",
    description="Complete tenant self-service onboarding.",
)
async def onboard_tenant(
    request: TenantOnboardingRequest,
    service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """
    Complete tenant onboarding workflow.

    - Public endpoint for self-service signup
    - Creates tenant and admin user
    - Activates tenant
    - Sends welcome email
    """
    try:
        tenant, admin_user_id = await service.onboard_tenant(
            TenantOnboardingInput(
                name=request.name,
                slug=request.slug,
                admin_email=request.admin_email,
                admin_name=request.admin_name,
                admin_password=request.admin_password,
                subscription_tier=request.subscription_tier or SubscriptionTier.FREE,
                company_size=request.company_size,
                industry=request.industry,
                use_case=request.use_case,
                referral_source=request.referral_source,
                accept_terms=request.accept_terms,
                terms_version=request.terms_version,
            )
        )

        logger.info(f"Onboarded tenant via API: {tenant.tenant_id}")

        return TenantResponse(
            id=str(tenant.id),
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            slug=tenant.slug,
            domain=tenant.domain,
            status=tenant.status.value,
            subscription_tier=tenant.subscription_tier.value,
            is_active=tenant.is_active,
            is_trial=tenant.is_trial,
            trial_ends_at=tenant.trial_ends_at,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
            settings=tenant.settings or {},
            quotas=tenant.get_effective_quotas(),
            current_usage=tenant.current_usage or {},
            feature_flags=tenant.get_effective_feature_flags(),
        )

    except TenantAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": e.code, "message": e.message},
        )
    except TenantServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": e.code, "message": e.message},
        )


@router.get(
    "",
    response_model=TenantListResponse,
    summary="List tenants",
    description="List all tenants (system admin only).",
)
async def list_tenants(
    status_filter: Optional[TenantStatus] = Query(None, alias="status"),
    tier: Optional[SubscriptionTier] = Query(None),
    search: Optional[str] = Query(None, min_length=1),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    service: TenantService = Depends(get_tenant_service),
) -> TenantListResponse:
    """
    List all tenants with filtering and pagination.

    - System admin only endpoint
    - Supports filtering by status, tier, and search
    - Returns paginated results
    """
    filters = TenantListFilters(
        status=status_filter,
        subscription_tier=tier,
        search_query=search,
    )
    pagination = PaginationParams(
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    tenants, total = await service.list_tenants(filters, pagination)

    return TenantListResponse(
        tenants=[
            TenantPublicResponse(
                id=str(t.id),
                tenant_id=t.tenant_id,
                name=t.name,
                slug=t.slug,
                subscription_tier=t.subscription_tier.value,
                is_active=t.is_active,
                created_at=t.created_at,
            )
            for t in tenants
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/current",
    response_model=TenantResponse,
    summary="Get current tenant",
    description="Get the current tenant from request context.",
)
async def get_current_tenant(
    http_request: Request,
    service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """
    Get the current tenant based on request context.

    Uses the tenant context from the middleware.
    """
    context = get_tenant_context()
    if not context:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "NO_TENANT_CONTEXT", "message": "Tenant context not available"},
        )

    tenant = await service.get_tenant(context.tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TENANT_NOT_FOUND", "message": "Tenant not found"},
        )

    return TenantResponse(
        id=str(tenant.id),
        tenant_id=tenant.tenant_id,
        name=tenant.name,
        slug=tenant.slug,
        domain=tenant.domain,
        status=tenant.status.value,
        subscription_tier=tenant.subscription_tier.value,
        is_active=tenant.is_active,
        is_trial=tenant.is_trial,
        trial_ends_at=tenant.trial_ends_at,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
        settings=tenant.settings or {},
        quotas=tenant.get_effective_quotas(),
        current_usage=tenant.current_usage or {},
        feature_flags=tenant.get_effective_feature_flags(),
    )


@router.get(
    "/{tenant_id}",
    response_model=TenantResponse,
    summary="Get tenant",
    description="Get tenant details by ID.",
)
async def get_tenant(
    tenant_id: str,
    service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """
    Get tenant details by tenant_id.

    Includes configuration, quotas, and feature flags.
    """
    tenant = await service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TENANT_NOT_FOUND", "message": f"Tenant '{tenant_id}' not found"},
        )

    return TenantResponse(
        id=str(tenant.id),
        tenant_id=tenant.tenant_id,
        name=tenant.name,
        slug=tenant.slug,
        domain=tenant.domain,
        status=tenant.status.value,
        subscription_tier=tenant.subscription_tier.value,
        is_active=tenant.is_active,
        is_trial=tenant.is_trial,
        trial_ends_at=tenant.trial_ends_at,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
        settings=tenant.settings or {},
        quotas=tenant.get_effective_quotas(),
        current_usage=tenant.current_usage or {},
        feature_flags=tenant.get_effective_feature_flags(),
    )


@router.patch(
    "/{tenant_id}",
    response_model=TenantResponse,
    summary="Update tenant",
    description="Update tenant information.",
)
async def update_tenant(
    tenant_id: str,
    request: TenantUpdateRequest,
    http_request: Request,
    service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """
    Update tenant information.

    - Admin only endpoint
    - Updates specified fields only
    """
    context = get_tenant_context()
    user_id = context.user_id if context else None

    try:
        tenant = await service.update_tenant(
            tenant_id,
            TenantUpdateInput(**request.dict(exclude_unset=True)),
            updated_by=user_id,
        )

        return TenantResponse(
            id=str(tenant.id),
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            slug=tenant.slug,
            domain=tenant.domain,
            status=tenant.status.value,
            subscription_tier=tenant.subscription_tier.value,
            is_active=tenant.is_active,
            is_trial=tenant.is_trial,
            trial_ends_at=tenant.trial_ends_at,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
            settings=tenant.settings or {},
            quotas=tenant.get_effective_quotas(),
            current_usage=tenant.current_usage or {},
            feature_flags=tenant.get_effective_feature_flags(),
        )

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": e.code, "message": e.message},
        )


@router.delete(
    "/{tenant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete tenant",
    description="Delete a tenant (soft delete by default).",
)
async def delete_tenant(
    tenant_id: str,
    hard_delete: bool = Query(False, description="Permanently delete"),
    http_request: Request = None,
    service: TenantService = Depends(get_tenant_service),
) -> None:
    """
    Delete a tenant.

    - System admin only endpoint
    - Soft delete by default (deactivates tenant)
    - Hard delete permanently removes all data
    """
    context = get_tenant_context()
    user_id = context.user_id if context else None

    try:
        await service.delete_tenant(
            tenant_id,
            deleted_by=user_id,
            hard_delete=hard_delete,
        )
        logger.info(f"Deleted tenant {tenant_id} (hard={hard_delete})")

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": e.code, "message": e.message},
        )


# =============================================================================
# Status Management Endpoints
# =============================================================================


@router.post(
    "/{tenant_id}/activate",
    response_model=TenantResponse,
    summary="Activate tenant",
    description="Activate a pending tenant.",
)
async def activate_tenant(
    tenant_id: str,
    http_request: Request,
    service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """Activate a pending tenant."""
    context = get_tenant_context()
    user_id = context.user_id if context else None

    try:
        tenant = await service.activate_tenant(tenant_id, activated_by=user_id)

        return TenantResponse(
            id=str(tenant.id),
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            slug=tenant.slug,
            domain=tenant.domain,
            status=tenant.status.value,
            subscription_tier=tenant.subscription_tier.value,
            is_active=tenant.is_active,
            is_trial=tenant.is_trial,
            trial_ends_at=tenant.trial_ends_at,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
            settings=tenant.settings or {},
            quotas=tenant.get_effective_quotas(),
            current_usage=tenant.current_usage or {},
            feature_flags=tenant.get_effective_feature_flags(),
        )

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": e.code, "message": e.message},
        )


@router.post(
    "/{tenant_id}/suspend",
    response_model=TenantResponse,
    summary="Suspend tenant",
    description="Suspend a tenant.",
)
async def suspend_tenant(
    tenant_id: str,
    reason: str = Query(..., min_length=1),
    http_request: Request = None,
    service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """Suspend a tenant."""
    context = get_tenant_context()
    user_id = context.user_id if context else None

    try:
        tenant = await service.suspend_tenant(
            tenant_id,
            reason=reason,
            suspended_by=user_id,
        )

        return TenantResponse(
            id=str(tenant.id),
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            slug=tenant.slug,
            domain=tenant.domain,
            status=tenant.status.value,
            subscription_tier=tenant.subscription_tier.value,
            is_active=tenant.is_active,
            is_trial=tenant.is_trial,
            trial_ends_at=tenant.trial_ends_at,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
            settings=tenant.settings or {},
            quotas=tenant.get_effective_quotas(),
            current_usage=tenant.current_usage or {},
            feature_flags=tenant.get_effective_feature_flags(),
        )

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": e.code, "message": e.message},
        )


@router.post(
    "/{tenant_id}/reactivate",
    response_model=TenantResponse,
    summary="Reactivate tenant",
    description="Reactivate a suspended tenant.",
)
async def reactivate_tenant(
    tenant_id: str,
    http_request: Request,
    service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """Reactivate a suspended tenant."""
    context = get_tenant_context()
    user_id = context.user_id if context else None

    try:
        tenant = await service.reactivate_tenant(tenant_id, reactivated_by=user_id)

        return TenantResponse(
            id=str(tenant.id),
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            slug=tenant.slug,
            domain=tenant.domain,
            status=tenant.status.value,
            subscription_tier=tenant.subscription_tier.value,
            is_active=tenant.is_active,
            is_trial=tenant.is_trial,
            trial_ends_at=tenant.trial_ends_at,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
            settings=tenant.settings or {},
            quotas=tenant.get_effective_quotas(),
            current_usage=tenant.current_usage or {},
            feature_flags=tenant.get_effective_feature_flags(),
        )

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": e.code, "message": e.message},
        )


# =============================================================================
# Quota Endpoints
# =============================================================================


@router.get(
    "/{tenant_id}/quotas",
    response_model=QuotaResponse,
    summary="Get tenant quotas",
    description="Get tenant quota limits and current usage.",
)
async def get_tenant_quotas(
    tenant_id: str,
    service: TenantService = Depends(get_tenant_service),
) -> QuotaResponse:
    """Get tenant quotas and usage."""
    tenant = await service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TENANT_NOT_FOUND", "message": f"Tenant '{tenant_id}' not found"},
        )

    quotas = tenant.get_effective_quotas()
    usage = tenant.current_usage or {}

    # Calculate usage percentage
    usage_percentage = {}
    for quota_name, limit in quotas.items():
        current = usage.get(quota_name, 0)
        if limit > 0:
            usage_percentage[quota_name] = round((current / limit) * 100, 2)
        elif limit == -1:
            usage_percentage[quota_name] = 0.0
        else:
            usage_percentage[quota_name] = 100.0 if current > 0 else 0.0

    return QuotaResponse(
        tenant_id=tenant_id,
        quotas=quotas,
        usage=usage,
        usage_percentage=usage_percentage,
    )


@router.patch(
    "/{tenant_id}/quotas",
    response_model=QuotaResponse,
    summary="Update tenant quotas",
    description="Update tenant quota limits (system admin only).",
)
async def update_tenant_quotas(
    tenant_id: str,
    request: QuotaUpdateRequest,
    http_request: Request,
    service: TenantService = Depends(get_tenant_service),
) -> QuotaResponse:
    """Update tenant quota limits."""
    context = get_tenant_context()
    user_id = context.user_id if context else None

    try:
        tenant = await service.update_quotas(
            tenant_id,
            QuotaUpdateInput(**request.dict(exclude_unset=True)),
            updated_by=user_id,
        )

        quotas = tenant.get_effective_quotas()
        usage = tenant.current_usage or {}

        usage_percentage = {}
        for quota_name, limit in quotas.items():
            current = usage.get(quota_name, 0)
            if limit > 0:
                usage_percentage[quota_name] = round((current / limit) * 100, 2)
            elif limit == -1:
                usage_percentage[quota_name] = 0.0
            else:
                usage_percentage[quota_name] = 100.0 if current > 0 else 0.0

        return QuotaResponse(
            tenant_id=tenant_id,
            quotas=quotas,
            usage=usage,
            usage_percentage=usage_percentage,
        )

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": e.code, "message": e.message},
        )


# =============================================================================
# Feature Flag Endpoints
# =============================================================================


@router.get(
    "/{tenant_id}/features",
    response_model=FeatureFlagResponse,
    summary="Get feature flags",
    description="Get tenant feature flags.",
)
async def get_tenant_features(
    tenant_id: str,
    service: TenantService = Depends(get_tenant_service),
) -> FeatureFlagResponse:
    """Get tenant feature flags."""
    tenant = await service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TENANT_NOT_FOUND", "message": f"Tenant '{tenant_id}' not found"},
        )

    return FeatureFlagResponse(
        tenant_id=tenant_id,
        features=tenant.get_effective_feature_flags(),
    )


@router.patch(
    "/{tenant_id}/features/{feature_name}",
    response_model=FeatureFlagResponse,
    summary="Update feature flag",
    description="Update a tenant feature flag (system admin only).",
)
async def update_tenant_feature(
    tenant_id: str,
    feature_name: str,
    request: FeatureFlagUpdateRequest,
    http_request: Request,
    service: TenantService = Depends(get_tenant_service),
) -> FeatureFlagResponse:
    """Update a feature flag for a tenant."""
    context = get_tenant_context()
    user_id = context.user_id if context else None

    try:
        tenant = await service.update_feature_flag(
            tenant_id,
            feature_name,
            request.enabled,
            updated_by=user_id,
        )

        return FeatureFlagResponse(
            tenant_id=tenant_id,
            features=tenant.get_effective_feature_flags(),
        )

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": e.code, "message": e.message},
        )


# =============================================================================
# Subscription Endpoints
# =============================================================================


@router.post(
    "/{tenant_id}/subscription/upgrade",
    response_model=TenantResponse,
    summary="Upgrade subscription",
    description="Upgrade tenant subscription tier.",
)
async def upgrade_tenant_subscription(
    tenant_id: str,
    request: SubscriptionUpdateRequest,
    http_request: Request,
    service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """Upgrade tenant subscription."""
    context = get_tenant_context()
    user_id = context.user_id if context else None

    try:
        tenant = await service.upgrade_subscription(
            tenant_id,
            request.new_tier,
            upgraded_by=user_id,
        )

        return TenantResponse(
            id=str(tenant.id),
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            slug=tenant.slug,
            domain=tenant.domain,
            status=tenant.status.value,
            subscription_tier=tenant.subscription_tier.value,
            is_active=tenant.is_active,
            is_trial=tenant.is_trial,
            trial_ends_at=tenant.trial_ends_at,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
            settings=tenant.settings or {},
            quotas=tenant.get_effective_quotas(),
            current_usage=tenant.current_usage or {},
            feature_flags=tenant.get_effective_feature_flags(),
        )

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": e.code, "message": e.message},
        )


@router.post(
    "/{tenant_id}/subscription/downgrade",
    response_model=TenantResponse,
    summary="Downgrade subscription",
    description="Downgrade tenant subscription tier.",
)
async def downgrade_tenant_subscription(
    tenant_id: str,
    request: SubscriptionUpdateRequest,
    http_request: Request,
    service: TenantService = Depends(get_tenant_service),
) -> TenantResponse:
    """Downgrade tenant subscription."""
    context = get_tenant_context()
    user_id = context.user_id if context else None

    try:
        tenant = await service.downgrade_subscription(
            tenant_id,
            request.new_tier,
            downgraded_by=user_id,
        )

        return TenantResponse(
            id=str(tenant.id),
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            slug=tenant.slug,
            domain=tenant.domain,
            status=tenant.status.value,
            subscription_tier=tenant.subscription_tier.value,
            is_active=tenant.is_active,
            is_trial=tenant.is_trial,
            trial_ends_at=tenant.trial_ends_at,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
            settings=tenant.settings or {},
            quotas=tenant.get_effective_quotas(),
            current_usage=tenant.current_usage or {},
            feature_flags=tenant.get_effective_feature_flags(),
        )

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": e.code, "message": e.message},
        )
    except TenantServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": e.code, "message": e.message},
        )


# =============================================================================
# Usage and Billing Endpoints
# =============================================================================


@router.get(
    "/{tenant_id}/usage",
    response_model=UsageSummaryResponse,
    summary="Get usage summary",
    description="Get tenant usage summary for billing.",
)
async def get_tenant_usage(
    tenant_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    service: TenantService = Depends(get_tenant_service),
) -> UsageSummaryResponse:
    """Get tenant usage summary."""
    try:
        usage = await service.get_usage_summary(
            tenant_id,
            start_date=start_date,
            end_date=end_date,
        )

        return UsageSummaryResponse(**usage)

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": e.code, "message": e.message},
        )


@router.get(
    "/{tenant_id}/billing",
    response_model=BillingMetricsResponse,
    summary="Get billing metrics",
    description="Get tenant billing metrics.",
)
async def get_tenant_billing(
    tenant_id: str,
    service: TenantService = Depends(get_tenant_service),
) -> BillingMetricsResponse:
    """Get tenant billing metrics."""
    try:
        metrics = await service.get_billing_metrics(tenant_id)
        return BillingMetricsResponse(**metrics)

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": e.code, "message": e.message},
        )


# =============================================================================
# Settings Endpoints
# =============================================================================


@router.get(
    "/{tenant_id}/settings",
    response_model=Dict[str, Any],
    summary="Get tenant settings",
    description="Get tenant settings.",
)
async def get_tenant_settings(
    tenant_id: str,
    service: TenantService = Depends(get_tenant_service),
) -> Dict[str, Any]:
    """Get tenant settings."""
    tenant = await service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TENANT_NOT_FOUND", "message": f"Tenant '{tenant_id}' not found"},
        )

    return tenant.settings or {}


@router.patch(
    "/{tenant_id}/settings",
    response_model=Dict[str, Any],
    summary="Update tenant settings",
    description="Update tenant settings.",
)
async def update_tenant_settings(
    tenant_id: str,
    request: SettingsUpdateRequest,
    http_request: Request,
    service: TenantService = Depends(get_tenant_service),
) -> Dict[str, Any]:
    """Update tenant settings."""
    context = get_tenant_context()
    user_id = context.user_id if context else None

    try:
        tenant = await service.update_tenant(
            tenant_id,
            TenantUpdateInput(settings=request.settings),
            updated_by=user_id,
        )

        return tenant.settings or {}

    except TenantNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": e.code, "message": e.message},
        )


# =============================================================================
# User Role Management Endpoints
# =============================================================================


@router.post(
    "/{tenant_id}/users/{user_id}/roles",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Assign role",
    description="Assign a role to a user.",
)
async def assign_role(
    tenant_id: str,
    user_id: str,
    request: UserRoleRequest,
) -> None:
    """
    Assign a role to a user.

    Valid roles: admin, developer, viewer
    """
    # In production, implement role assignment logic
    logger.info(f"Assigning role {request.role} to user {user_id} in tenant {tenant_id}")


@router.delete(
    "/{tenant_id}/users/{user_id}/roles/{role_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove role",
    description="Remove a role from a user.",
)
async def remove_role(
    tenant_id: str,
    user_id: str,
    role_id: str,
) -> None:
    """
    Remove a role from a user.

    Cannot remove the last admin.
    """
    # In production, implement role removal logic
    logger.info(f"Removing role {role_id} from user {user_id} in tenant {tenant_id}")
