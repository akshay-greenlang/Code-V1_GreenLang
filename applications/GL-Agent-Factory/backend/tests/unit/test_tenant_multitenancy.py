"""
Unit Tests for Multi-Tenancy Components

This module provides comprehensive unit tests for:
- Tenant Model with subscription tiers and feature flags
- Tenant Context Middleware
- Row-Level Security helpers
- Tenant Service CRUD operations

Test coverage target: 85%+
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

# Import components under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.tenant import (
    Tenant,
    TenantStatus,
    SubscriptionTier,
    TenantUsageLog,
    TenantInvitation,
    DEFAULT_TIER_QUOTAS,
    DEFAULT_TIER_FEATURES,
)
from app.middleware.tenant_context import (
    TenantContext,
    get_tenant_context,
    set_tenant_context,
    clear_tenant_context,
)
from services.tenant.tenant_service import (
    TenantService,
    TenantCreateInput,
    TenantUpdateInput,
    TenantOnboardingInput,
    TenantNotFoundError,
    TenantAlreadyExistsError,
    QuotaExceededError,
    TenantServiceError,
)


# =============================================================================
# Tenant Model Tests
# =============================================================================


class TestTenantModel:
    """Test cases for Tenant model."""

    def test_tenant_creation_with_defaults(self):
        """Test creating a tenant with default values."""
        tenant = Tenant()
        tenant.id = uuid4()
        tenant.tenant_id = "t-test-corp"
        tenant.name = "Test Corporation"
        tenant.slug = "test-corp"
        tenant.status = TenantStatus.PENDING
        tenant.subscription_tier = SubscriptionTier.FREE
        tenant.is_active = True
        tenant.settings = {}
        tenant.quotas = {}
        tenant.current_usage = {}
        tenant.feature_flags = {}
        tenant.created_at = datetime.utcnow()
        tenant.updated_at = datetime.utcnow()

        assert tenant.tenant_id == "t-test-corp"
        assert tenant.name == "Test Corporation"
        assert tenant.status == TenantStatus.PENDING
        assert tenant.subscription_tier == SubscriptionTier.FREE

    def test_tenant_status_enum(self):
        """Test tenant status enum values."""
        assert TenantStatus.PENDING.value == "pending"
        assert TenantStatus.ACTIVE.value == "active"
        assert TenantStatus.SUSPENDED.value == "suspended"
        assert TenantStatus.DEACTIVATED.value == "deactivated"

    def test_subscription_tier_enum(self):
        """Test subscription tier enum values."""
        assert SubscriptionTier.FREE.value == "free"
        assert SubscriptionTier.PRO.value == "pro"
        assert SubscriptionTier.ENTERPRISE.value == "enterprise"

    def test_get_effective_quotas_free_tier(self):
        """Test effective quotas for free tier."""
        tenant = self._create_tenant(SubscriptionTier.FREE)
        quotas = tenant.get_effective_quotas()

        assert quotas["agents"] == 5
        assert quotas["executions_per_month"] == 100
        assert quotas["storage_gb"] == 1
        assert quotas["concurrent_executions"] == 2

    def test_get_effective_quotas_pro_tier(self):
        """Test effective quotas for pro tier."""
        tenant = self._create_tenant(SubscriptionTier.PRO)
        quotas = tenant.get_effective_quotas()

        assert quotas["agents"] == 50
        assert quotas["executions_per_month"] == 5000
        assert quotas["storage_gb"] == 25

    def test_get_effective_quotas_enterprise_tier(self):
        """Test effective quotas for enterprise tier."""
        tenant = self._create_tenant(SubscriptionTier.ENTERPRISE)
        quotas = tenant.get_effective_quotas()

        assert quotas["agents"] == 500
        assert quotas["team_members"] == -1  # Unlimited

    def test_get_effective_quotas_with_custom_override(self):
        """Test effective quotas with custom overrides."""
        tenant = self._create_tenant(SubscriptionTier.FREE)
        tenant.quotas = {"agents": 20}  # Custom override

        quotas = tenant.get_effective_quotas()
        assert quotas["agents"] == 20  # Custom value
        assert quotas["executions_per_month"] == 100  # Default

    def test_check_quota_within_limit(self):
        """Test quota check within limit."""
        tenant = self._create_tenant(SubscriptionTier.FREE)
        tenant.current_usage = {"agents": 3}

        assert tenant.check_quota("agents", 1) is True
        assert tenant.check_quota("agents", 2) is True

    def test_check_quota_exceeds_limit(self):
        """Test quota check exceeding limit."""
        tenant = self._create_tenant(SubscriptionTier.FREE)
        tenant.current_usage = {"agents": 5}

        assert tenant.check_quota("agents", 1) is False

    def test_check_quota_unlimited(self):
        """Test quota check for unlimited quota."""
        tenant = self._create_tenant(SubscriptionTier.ENTERPRISE)
        tenant.current_usage = {"team_members": 1000}

        assert tenant.check_quota("team_members", 100) is True

    def test_get_quota_remaining(self):
        """Test getting remaining quota."""
        tenant = self._create_tenant(SubscriptionTier.FREE)
        tenant.current_usage = {"agents": 3}

        remaining = tenant.get_quota_remaining("agents")
        assert remaining == 2  # 5 - 3

    def test_get_effective_feature_flags_free_tier(self):
        """Test feature flags for free tier."""
        tenant = self._create_tenant(SubscriptionTier.FREE)
        features = tenant.get_effective_feature_flags()

        assert features["basic_agents"] is True
        assert features["carbon_emissions"] is True
        assert features["cbam_compliance"] is False
        assert features["csrd_reporting"] is False
        assert features["sso_authentication"] is False

    def test_get_effective_feature_flags_pro_tier(self):
        """Test feature flags for pro tier."""
        tenant = self._create_tenant(SubscriptionTier.PRO)
        features = tenant.get_effective_feature_flags()

        assert features["cbam_compliance"] is True
        assert features["csrd_reporting"] is True
        assert features["audit_logs"] is True
        assert features["sso_authentication"] is False

    def test_get_effective_feature_flags_enterprise_tier(self):
        """Test feature flags for enterprise tier."""
        tenant = self._create_tenant(SubscriptionTier.ENTERPRISE)
        features = tenant.get_effective_feature_flags()

        assert features["eudr_compliance"] is True
        assert features["sso_authentication"] is True
        assert features["dedicated_infrastructure"] is True
        assert features["white_labeling"] is True

    def test_is_feature_enabled(self):
        """Test feature enabled check."""
        tenant = self._create_tenant(SubscriptionTier.PRO)

        assert tenant.is_feature_enabled("cbam_compliance") is True
        assert tenant.is_feature_enabled("sso_authentication") is False
        assert tenant.is_feature_enabled("nonexistent_feature") is False

    def test_feature_flag_override(self):
        """Test custom feature flag override."""
        tenant = self._create_tenant(SubscriptionTier.FREE)
        tenant.feature_flags = {"cbam_compliance": True}  # Enable for free tier

        assert tenant.is_feature_enabled("cbam_compliance") is True

    def test_is_operational_active_tenant(self):
        """Test is_operational for active tenant."""
        tenant = self._create_tenant(SubscriptionTier.FREE)
        tenant.status = TenantStatus.ACTIVE
        tenant.is_active = True

        assert tenant.is_operational() is True

    def test_is_operational_suspended_tenant(self):
        """Test is_operational for suspended tenant."""
        tenant = self._create_tenant(SubscriptionTier.FREE)
        tenant.status = TenantStatus.SUSPENDED
        tenant.is_active = False

        assert tenant.is_operational() is False

    def test_is_in_trial(self):
        """Test trial period check."""
        tenant = self._create_tenant(SubscriptionTier.FREE)
        tenant.is_trial = True
        tenant.trial_ends_at = datetime.utcnow() + timedelta(days=7)

        assert tenant.is_in_trial() is True

    def test_is_in_trial_expired(self):
        """Test expired trial period."""
        tenant = self._create_tenant(SubscriptionTier.FREE)
        tenant.is_trial = True
        tenant.trial_ends_at = datetime.utcnow() - timedelta(days=1)

        assert tenant.is_in_trial() is False

    def test_to_dict(self):
        """Test converting tenant to dictionary."""
        tenant = self._create_tenant(SubscriptionTier.PRO)
        tenant_dict = tenant.to_dict()

        assert tenant_dict["tenant_id"] == tenant.tenant_id
        assert tenant_dict["name"] == tenant.name
        assert tenant_dict["subscription_tier"] == "pro"
        assert "quotas" in tenant_dict
        assert "feature_flags" in tenant_dict

    def test_provenance_hash(self):
        """Test provenance hash calculation."""
        tenant = self._create_tenant(SubscriptionTier.FREE)
        hash1 = tenant.calculate_provenance_hash()

        # Same tenant should have same hash
        hash2 = tenant.calculate_provenance_hash()
        assert hash1 == hash2

        # After update, hash should change
        tenant.name = "Updated Name"
        tenant.updated_at = datetime.utcnow()
        hash3 = tenant.calculate_provenance_hash()
        assert hash1 != hash3

    def _create_tenant(self, tier: SubscriptionTier) -> Tenant:
        """Helper to create a tenant for testing."""
        tenant = Tenant()
        tenant.id = uuid4()
        tenant.tenant_id = f"t-test-{uuid4().hex[:8]}"
        tenant.name = "Test Tenant"
        tenant.slug = "test-tenant"
        tenant.status = TenantStatus.ACTIVE
        tenant.subscription_tier = tier
        tenant.is_active = True
        tenant.is_trial = False
        tenant.settings = {}
        tenant.quotas = {}
        tenant.current_usage = {}
        tenant.feature_flags = {}
        tenant.created_at = datetime.utcnow()
        tenant.updated_at = datetime.utcnow()
        return tenant


class TestTenantInvitation:
    """Test cases for TenantInvitation model."""

    def test_invitation_is_valid(self):
        """Test valid invitation."""
        invitation = TenantInvitation()
        invitation.id = uuid4()
        invitation.tenant_id = uuid4()
        invitation.email = "test@example.com"
        invitation.role = "admin"
        invitation.token = "test-token-123"
        invitation.invited_by = uuid4()
        invitation.expires_at = datetime.utcnow() + timedelta(days=7)
        invitation.accepted_at = None
        invitation.created_at = datetime.utcnow()

        assert invitation.is_valid() is True
        assert invitation.is_expired() is False

    def test_invitation_expired(self):
        """Test expired invitation."""
        invitation = TenantInvitation()
        invitation.expires_at = datetime.utcnow() - timedelta(days=1)
        invitation.accepted_at = None

        assert invitation.is_expired() is True
        assert invitation.is_valid() is False

    def test_invitation_already_accepted(self):
        """Test already accepted invitation."""
        invitation = TenantInvitation()
        invitation.expires_at = datetime.utcnow() + timedelta(days=7)
        invitation.accepted_at = datetime.utcnow()

        assert invitation.is_valid() is False


# =============================================================================
# Tenant Context Tests
# =============================================================================


class TestTenantContext:
    """Test cases for TenantContext."""

    def test_context_creation(self):
        """Test creating tenant context."""
        context = TenantContext(
            tenant_id="t-test",
            tenant_uuid=uuid4(),
            name="Test Tenant",
            slug="test",
            subscription_tier=SubscriptionTier.PRO,
            status=TenantStatus.ACTIVE,
            feature_flags={"api_access": True},
            quotas={"agents": 50},
            current_usage={"agents": 10},
            user_id="user-123",
            user_roles=["admin", "developer"],
        )

        assert context.tenant_id == "t-test"
        assert context.subscription_tier == SubscriptionTier.PRO
        assert context.user_id == "user-123"

    def test_context_is_feature_enabled(self):
        """Test feature check in context."""
        context = TenantContext(
            tenant_id="t-test",
            tenant_uuid=uuid4(),
            name="Test",
            slug="test",
            subscription_tier=SubscriptionTier.FREE,
            status=TenantStatus.ACTIVE,
            feature_flags={"api_access": True, "custom_agents": False},
        )

        assert context.is_feature_enabled("api_access") is True
        assert context.is_feature_enabled("custom_agents") is False
        assert context.is_feature_enabled("nonexistent") is False

    def test_context_check_quota(self):
        """Test quota check in context."""
        context = TenantContext(
            tenant_id="t-test",
            tenant_uuid=uuid4(),
            name="Test",
            slug="test",
            subscription_tier=SubscriptionTier.FREE,
            status=TenantStatus.ACTIVE,
            quotas={"agents": 10},
            current_usage={"agents": 8},
        )

        assert context.check_quota("agents", 1) is True
        assert context.check_quota("agents", 2) is True
        assert context.check_quota("agents", 3) is False

    def test_context_has_role(self):
        """Test role check in context."""
        context = TenantContext(
            tenant_id="t-test",
            tenant_uuid=uuid4(),
            name="Test",
            slug="test",
            subscription_tier=SubscriptionTier.FREE,
            status=TenantStatus.ACTIVE,
            user_roles=["admin", "developer"],
        )

        assert context.has_role("admin") is True
        assert context.has_role("developer") is True
        assert context.has_role("viewer") is False

    def test_context_is_admin(self):
        """Test admin check in context."""
        context = TenantContext(
            tenant_id="t-test",
            tenant_uuid=uuid4(),
            name="Test",
            slug="test",
            subscription_tier=SubscriptionTier.FREE,
            status=TenantStatus.ACTIVE,
            user_roles=["admin"],
        )

        assert context.is_admin() is True

        context.user_roles = ["developer"]
        assert context.is_admin() is False

    def test_context_from_tenant(self):
        """Test creating context from Tenant model."""
        tenant = Tenant()
        tenant.id = uuid4()
        tenant.tenant_id = "t-test"
        tenant.name = "Test Tenant"
        tenant.slug = "test"
        tenant.status = TenantStatus.ACTIVE
        tenant.subscription_tier = SubscriptionTier.PRO
        tenant.is_active = True
        tenant.settings = {}
        tenant.quotas = {}
        tenant.current_usage = {"agents": 5}
        tenant.feature_flags = {}
        tenant.created_at = datetime.utcnow()
        tenant.updated_at = datetime.utcnow()

        context = TenantContext.from_tenant(
            tenant=tenant,
            user_id="user-123",
            user_roles=["admin"],
            request_id="req-abc",
        )

        assert context.tenant_id == "t-test"
        assert context.name == "Test Tenant"
        assert context.user_id == "user-123"
        assert context.request_id == "req-abc"

    def test_context_var_set_and_get(self):
        """Test context variable set and get."""
        context = TenantContext(
            tenant_id="t-test",
            tenant_uuid=uuid4(),
            name="Test",
            slug="test",
            subscription_tier=SubscriptionTier.FREE,
            status=TenantStatus.ACTIVE,
        )

        # Set context
        set_tenant_context(context)

        # Get context
        retrieved = get_tenant_context()
        assert retrieved is not None
        assert retrieved.tenant_id == "t-test"

        # Clear context
        clear_tenant_context()
        assert get_tenant_context() is None


# =============================================================================
# Tenant Service Tests
# =============================================================================


class TestTenantService:
    """Test cases for TenantService."""

    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return TenantService()

    @pytest.mark.asyncio
    async def test_create_tenant(self, service):
        """Test creating a tenant."""
        input_data = TenantCreateInput(
            name="Acme Corporation",
            slug="acme-corp",
            admin_email="admin@acme.com",
            admin_name="Admin User",
            subscription_tier=SubscriptionTier.PRO,
        )

        tenant = await service.create_tenant(input_data)

        assert tenant.name == "Acme Corporation"
        assert tenant.slug == "acme-corp"
        assert tenant.tenant_id == "t-acme-corp"
        assert tenant.subscription_tier == SubscriptionTier.PRO
        assert tenant.status == TenantStatus.PENDING

    @pytest.mark.asyncio
    async def test_create_tenant_with_trial(self, service):
        """Test creating tenant with trial period."""
        input_data = TenantCreateInput(
            name="Trial Corp",
            slug="trial-corp",
            admin_email="admin@trial.com",
            subscription_tier=SubscriptionTier.FREE,
            start_trial=True,
            trial_days=14,
        )

        tenant = await service.create_tenant(input_data)

        assert tenant.is_trial is True
        assert tenant.trial_ends_at is not None
        assert tenant.trial_ends_at > datetime.utcnow()

    @pytest.mark.asyncio
    async def test_create_tenant_duplicate_slug(self, service):
        """Test creating tenant with duplicate slug."""
        input_data = TenantCreateInput(
            name="First Corp",
            slug="unique-slug",
            admin_email="admin@first.com",
        )

        await service.create_tenant(input_data)

        # Try to create with same slug
        with pytest.raises(TenantAlreadyExistsError):
            await service.create_tenant(
                TenantCreateInput(
                    name="Second Corp",
                    slug="unique-slug",
                    admin_email="admin@second.com",
                )
            )

    @pytest.mark.asyncio
    async def test_get_tenant(self, service):
        """Test getting a tenant."""
        # Create tenant first
        input_data = TenantCreateInput(
            name="Get Test Corp",
            slug="get-test-corp",
            admin_email="admin@gettest.com",
        )
        created = await service.create_tenant(input_data)

        # Get tenant
        tenant = await service.get_tenant(created.tenant_id)

        assert tenant is not None
        assert tenant.name == "Get Test Corp"
        assert tenant.tenant_id == created.tenant_id

    @pytest.mark.asyncio
    async def test_get_tenant_not_found(self, service):
        """Test getting non-existent tenant."""
        tenant = await service.get_tenant("t-nonexistent")
        assert tenant is None

    @pytest.mark.asyncio
    async def test_update_tenant(self, service):
        """Test updating a tenant."""
        # Create tenant
        input_data = TenantCreateInput(
            name="Update Test Corp",
            slug="update-test",
            admin_email="admin@update.com",
        )
        created = await service.create_tenant(input_data)

        # Update tenant
        update_data = TenantUpdateInput(
            name="Updated Corp Name",
            settings={"theme": "dark"},
        )
        updated = await service.update_tenant(created.tenant_id, update_data)

        assert updated.name == "Updated Corp Name"
        assert updated.settings.get("theme") == "dark"

    @pytest.mark.asyncio
    async def test_update_tenant_not_found(self, service):
        """Test updating non-existent tenant."""
        with pytest.raises(TenantNotFoundError):
            await service.update_tenant(
                "t-nonexistent",
                TenantUpdateInput(name="New Name"),
            )

    @pytest.mark.asyncio
    async def test_delete_tenant_soft(self, service):
        """Test soft deleting a tenant."""
        # Create tenant
        input_data = TenantCreateInput(
            name="Delete Test Corp",
            slug="delete-test",
            admin_email="admin@delete.com",
        )
        created = await service.create_tenant(input_data)

        # Soft delete
        result = await service.delete_tenant(created.tenant_id, hard_delete=False)
        assert result is True

        # Tenant should still exist but be deactivated
        tenant = await service.get_tenant(created.tenant_id)
        assert tenant.status == TenantStatus.DEACTIVATED
        assert tenant.is_active is False

    @pytest.mark.asyncio
    async def test_delete_tenant_hard(self, service):
        """Test hard deleting a tenant."""
        # Create tenant
        input_data = TenantCreateInput(
            name="Hard Delete Corp",
            slug="hard-delete",
            admin_email="admin@harddelete.com",
        )
        created = await service.create_tenant(input_data)

        # Hard delete
        result = await service.delete_tenant(created.tenant_id, hard_delete=True)
        assert result is True

        # Tenant should be gone
        tenant = await service.get_tenant(created.tenant_id)
        assert tenant is None

    @pytest.mark.asyncio
    async def test_activate_tenant(self, service):
        """Test activating a tenant."""
        # Create tenant (starts as PENDING)
        input_data = TenantCreateInput(
            name="Activate Test Corp",
            slug="activate-test",
            admin_email="admin@activate.com",
        )
        created = await service.create_tenant(input_data)
        assert created.status == TenantStatus.PENDING

        # Activate
        activated = await service.activate_tenant(created.tenant_id)

        assert activated.status == TenantStatus.ACTIVE
        assert activated.is_active is True
        assert activated.activated_at is not None

    @pytest.mark.asyncio
    async def test_suspend_tenant(self, service):
        """Test suspending a tenant."""
        # Create and activate tenant
        input_data = TenantCreateInput(
            name="Suspend Test Corp",
            slug="suspend-test",
            admin_email="admin@suspend.com",
        )
        created = await service.create_tenant(input_data)
        await service.activate_tenant(created.tenant_id)

        # Suspend
        suspended = await service.suspend_tenant(
            created.tenant_id,
            reason="Payment overdue",
        )

        assert suspended.status == TenantStatus.SUSPENDED
        assert suspended.is_active is False
        assert suspended.suspension_reason == "Payment overdue"

    @pytest.mark.asyncio
    async def test_reactivate_tenant(self, service):
        """Test reactivating a suspended tenant."""
        # Create, activate, then suspend
        input_data = TenantCreateInput(
            name="Reactivate Test Corp",
            slug="reactivate-test",
            admin_email="admin@reactivate.com",
        )
        created = await service.create_tenant(input_data)
        await service.activate_tenant(created.tenant_id)
        await service.suspend_tenant(created.tenant_id, reason="Test")

        # Reactivate
        reactivated = await service.reactivate_tenant(created.tenant_id)

        assert reactivated.status == TenantStatus.ACTIVE
        assert reactivated.is_active is True
        assert reactivated.suspension_reason is None

    @pytest.mark.asyncio
    async def test_upgrade_subscription(self, service):
        """Test upgrading subscription tier."""
        # Create free tier tenant
        input_data = TenantCreateInput(
            name="Upgrade Test Corp",
            slug="upgrade-test",
            admin_email="admin@upgrade.com",
            subscription_tier=SubscriptionTier.FREE,
        )
        created = await service.create_tenant(input_data)

        # Upgrade to PRO
        upgraded = await service.upgrade_subscription(
            created.tenant_id,
            SubscriptionTier.PRO,
        )

        assert upgraded.subscription_tier == SubscriptionTier.PRO

    @pytest.mark.asyncio
    async def test_downgrade_subscription(self, service):
        """Test downgrading subscription tier."""
        # Create PRO tier tenant
        input_data = TenantCreateInput(
            name="Downgrade Test Corp",
            slug="downgrade-test",
            admin_email="admin@downgrade.com",
            subscription_tier=SubscriptionTier.PRO,
        )
        created = await service.create_tenant(input_data)

        # Downgrade to FREE
        downgraded = await service.downgrade_subscription(
            created.tenant_id,
            SubscriptionTier.FREE,
        )

        assert downgraded.subscription_tier == SubscriptionTier.FREE

    @pytest.mark.asyncio
    async def test_check_quota(self, service):
        """Test checking quota."""
        # Create tenant
        input_data = TenantCreateInput(
            name="Quota Check Corp",
            slug="quota-check",
            admin_email="admin@quota.com",
            subscription_tier=SubscriptionTier.FREE,
        )
        created = await service.create_tenant(input_data)

        # Check quota
        allowed, limit, current = await service.check_quota(
            created.tenant_id,
            "agents",
            increment=1,
        )

        assert allowed is True
        assert limit == 5  # Free tier limit
        assert current == 0

    @pytest.mark.asyncio
    async def test_increment_usage(self, service):
        """Test incrementing usage."""
        # Create tenant
        input_data = TenantCreateInput(
            name="Usage Test Corp",
            slug="usage-test",
            admin_email="admin@usage.com",
        )
        created = await service.create_tenant(input_data)

        # Increment usage
        new_value = await service.increment_usage(
            created.tenant_id,
            "executions_per_month",
            increment=5,
        )

        assert new_value == 5

        # Increment again
        new_value = await service.increment_usage(
            created.tenant_id,
            "executions_per_month",
            increment=3,
        )

        assert new_value == 8

    @pytest.mark.asyncio
    async def test_increment_usage_exceeds_quota(self, service):
        """Test incrementing usage beyond quota."""
        # Create free tier tenant
        input_data = TenantCreateInput(
            name="Quota Exceed Corp",
            slug="quota-exceed",
            admin_email="admin@exceed.com",
            subscription_tier=SubscriptionTier.FREE,
        )
        created = await service.create_tenant(input_data)

        # Try to exceed quota (free tier has 5 agents limit)
        created.current_usage = {"agents": 5}
        await service._persist_tenant(created)

        with pytest.raises(QuotaExceededError):
            await service.increment_usage(
                created.tenant_id,
                "agents",
                increment=1,
                check_quota=True,
            )

    @pytest.mark.asyncio
    async def test_update_feature_flag(self, service):
        """Test updating feature flag."""
        # Create tenant
        input_data = TenantCreateInput(
            name="Feature Flag Corp",
            slug="feature-flag",
            admin_email="admin@feature.com",
            subscription_tier=SubscriptionTier.FREE,
        )
        created = await service.create_tenant(input_data)

        # Enable a feature
        updated = await service.update_feature_flag(
            created.tenant_id,
            "custom_feature",
            enabled=True,
        )

        assert updated.is_feature_enabled("custom_feature") is True

    @pytest.mark.asyncio
    async def test_check_feature(self, service):
        """Test checking feature."""
        # Create PRO tenant
        input_data = TenantCreateInput(
            name="Feature Check Corp",
            slug="feature-check",
            admin_email="admin@featurecheck.com",
            subscription_tier=SubscriptionTier.PRO,
        )
        created = await service.create_tenant(input_data)

        # Check features
        assert await service.check_feature(created.tenant_id, "cbam_compliance") is True
        assert await service.check_feature(created.tenant_id, "sso_authentication") is False

    @pytest.mark.asyncio
    async def test_list_tenants(self, service):
        """Test listing tenants."""
        # Create multiple tenants
        for i in range(3):
            await service.create_tenant(
                TenantCreateInput(
                    name=f"List Test Corp {i}",
                    slug=f"list-test-{i}",
                    admin_email=f"admin{i}@list.com",
                )
            )

        # List all
        tenants, total = await service.list_tenants()

        assert total >= 3
        assert len(tenants) >= 3

    @pytest.mark.asyncio
    async def test_onboard_tenant(self, service):
        """Test tenant onboarding workflow."""
        input_data = TenantOnboardingInput(
            name="Onboard Test Corp",
            slug="onboard-test",
            admin_email="admin@onboard.com",
            admin_name="Admin User",
            accept_terms=True,
            terms_version="1.0",
        )

        tenant, admin_user_id = await service.onboard_tenant(input_data)

        assert tenant.name == "Onboard Test Corp"
        assert tenant.status == TenantStatus.ACTIVE  # Onboarding activates
        assert tenant.is_active is True
        assert admin_user_id is not None

    @pytest.mark.asyncio
    async def test_onboard_tenant_terms_not_accepted(self, service):
        """Test onboarding fails without accepting terms."""
        input_data = TenantOnboardingInput(
            name="Terms Test Corp",
            slug="terms-test",
            admin_email="admin@terms.com",
            admin_name="Admin",
            accept_terms=False,  # Not accepted
            terms_version="1.0",
        )

        with pytest.raises(TenantServiceError) as exc_info:
            await service.onboard_tenant(input_data)

        assert "TERMS_NOT_ACCEPTED" in str(exc_info.value.code)

    @pytest.mark.asyncio
    async def test_get_usage_summary(self, service):
        """Test getting usage summary."""
        # Create tenant with usage
        input_data = TenantCreateInput(
            name="Usage Summary Corp",
            slug="usage-summary",
            admin_email="admin@usagesummary.com",
        )
        created = await service.create_tenant(input_data)

        # Add some usage
        await service.increment_usage(created.tenant_id, "executions_per_month", 50)

        # Get summary
        summary = await service.get_usage_summary(created.tenant_id)

        assert summary["tenant_id"] == created.tenant_id
        assert "usage" in summary
        assert "period_start" in summary


# =============================================================================
# Default Quota and Feature Tests
# =============================================================================


class TestDefaultTierConfigurations:
    """Test default tier configurations."""

    def test_default_quotas_defined_for_all_tiers(self):
        """Test that all tiers have default quotas defined."""
        for tier in SubscriptionTier:
            assert tier in DEFAULT_TIER_QUOTAS
            quotas = DEFAULT_TIER_QUOTAS[tier]
            assert "agents" in quotas
            assert "executions_per_month" in quotas
            assert "storage_gb" in quotas

    def test_default_features_defined_for_all_tiers(self):
        """Test that all tiers have default features defined."""
        for tier in SubscriptionTier:
            assert tier in DEFAULT_TIER_FEATURES
            features = DEFAULT_TIER_FEATURES[tier]
            assert "basic_agents" in features
            assert "api_access" in features

    def test_quotas_increase_with_tier(self):
        """Test that quotas increase with tier level."""
        free_quotas = DEFAULT_TIER_QUOTAS[SubscriptionTier.FREE]
        pro_quotas = DEFAULT_TIER_QUOTAS[SubscriptionTier.PRO]
        enterprise_quotas = DEFAULT_TIER_QUOTAS[SubscriptionTier.ENTERPRISE]

        assert pro_quotas["agents"] > free_quotas["agents"]
        assert enterprise_quotas["agents"] > pro_quotas["agents"]

    def test_features_increase_with_tier(self):
        """Test that features increase with tier level."""
        free_features = DEFAULT_TIER_FEATURES[SubscriptionTier.FREE]
        pro_features = DEFAULT_TIER_FEATURES[SubscriptionTier.PRO]
        enterprise_features = DEFAULT_TIER_FEATURES[SubscriptionTier.ENTERPRISE]

        # Count enabled features
        free_count = sum(1 for v in free_features.values() if v)
        pro_count = sum(1 for v in pro_features.values() if v)
        enterprise_count = sum(1 for v in enterprise_features.values() if v)

        assert pro_count > free_count
        assert enterprise_count > pro_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
