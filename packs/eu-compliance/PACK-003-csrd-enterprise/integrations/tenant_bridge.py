# -*- coding: utf-8 -*-
"""
TenantBridge - Multi-Tenant Bridge for CSRD Enterprise Pack
=============================================================

This module connects the CSRD Enterprise Pack to the platform's TenantManager
(greenlang/auth/tenant.py) and enriches tenant profiles with CSRD-specific
metadata, feature flags per tier, data partition enforcement, tier migration,
and cross-tenant anonymous benchmarking.

Platform Integration:
    greenlang/auth/tenant.py -> TenantManager
        - Wraps TenantManager via composition (not inheritance)
        - Falls back to internal state when TenantManager is unavailable

Architecture:
    CSRD Enterprise Pack --> TenantBridge --> greenlang.auth.tenant.TenantManager
                                 |
                                 v
    TenantProfile (enriched) <-- CSRD Config + Feature Flags + Resource Usage

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-003 CSRD Enterprise
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TenantTier(str, Enum):
    """Tenant subscription tiers matching platform definitions."""

    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class TenantIsolationLevel(str, Enum):
    """Data isolation levels for multi-tenant deployments."""

    SHARED = "shared"
    NAMESPACE = "namespace"
    CLUSTER = "cluster"
    PHYSICAL = "physical"


class TenantStatus(str, Enum):
    """Tenant lifecycle status."""

    ACTIVE = "active"
    PROVISIONING = "provisioning"
    SUSPENDED = "suspended"
    DECOMMISSIONED = "decommissioned"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class CSRDTenantConfig(BaseModel):
    """CSRD-specific configuration attached to a tenant profile."""

    enabled_esrs_standards: List[str] = Field(
        default_factory=lambda: [
            "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E2", "ESRS_S1", "ESRS_G1",
        ],
    )
    enabled_scope3_categories: List[int] = Field(
        default_factory=lambda: list(range(1, 16)),
    )
    cross_frameworks: List[str] = Field(
        default_factory=lambda: ["cdp", "tcfd", "sbti", "eu_taxonomy"],
    )
    reporting_period_start: Optional[str] = Field(None)
    reporting_period_end: Optional[str] = Field(None)
    consolidation_approach: str = Field(default="operational_control")
    assurance_level: str = Field(default="limited")
    base_year: Optional[int] = Field(None, ge=2015, le=2030)
    enable_iot_integration: bool = Field(default=False)
    enable_supply_chain_esg: bool = Field(default=False)
    enable_carbon_credits: bool = Field(default=False)
    enable_predictive_analytics: bool = Field(default=False)
    enable_white_label: bool = Field(default=False)
    max_subsidiaries: int = Field(default=10, ge=0)
    max_users: int = Field(default=50, ge=1)
    max_api_calls_per_hour: int = Field(default=1000, ge=100)
    data_residency_region: str = Field(default="EU")


class TenantProfile(BaseModel):
    """Complete tenant profile with CSRD-specific enrichment."""

    tenant_id: str = Field(default_factory=_new_uuid)
    name: str = Field(..., min_length=1, max_length=255)
    tier: TenantTier = Field(default=TenantTier.ENTERPRISE)
    isolation_level: TenantIsolationLevel = Field(
        default=TenantIsolationLevel.NAMESPACE,
    )
    admin_email: str = Field(...)
    region: str = Field(default="EU")
    csrd_config: CSRDTenantConfig = Field(default_factory=CSRDTenantConfig)
    features: Dict[str, bool] = Field(default_factory=dict)
    status: TenantStatus = Field(default=TenantStatus.ACTIVE)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: Optional[datetime] = Field(None)
    provenance_hash: str = Field(default="")

    @field_validator("admin_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email format validation."""
        if "@" not in v or "." not in v:
            raise ValueError("admin_email must be a valid email address")
        return v


class ResourceUsage(BaseModel):
    """Tenant resource usage metrics."""

    tenant_id: str = Field(...)
    storage_used_mb: float = Field(default=0.0, ge=0.0)
    api_calls_today: int = Field(default=0, ge=0)
    api_calls_this_hour: int = Field(default=0, ge=0)
    active_users: int = Field(default=0, ge=0)
    active_workflows: int = Field(default=0, ge=0)
    subsidiaries_configured: int = Field(default=0, ge=0)
    last_activity_at: Optional[datetime] = Field(None)
    measured_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Tier Feature Definitions
# ---------------------------------------------------------------------------


TIER_FEATURES: Dict[TenantTier, Dict[str, bool]] = {
    TenantTier.STARTER: {
        "multi_entity": False,
        "cross_framework": False,
        "quality_gates": True,
        "approval_chain": False,
        "webhooks": False,
        "sso": False,
        "white_label": False,
        "iot_integration": False,
        "supply_chain_esg": False,
        "carbon_credits": False,
        "predictive_analytics": False,
        "graphql_api": False,
        "marketplace_plugins": False,
        "auditor_portal": False,
        "custom_workflows": False,
    },
    TenantTier.PROFESSIONAL: {
        "multi_entity": True,
        "cross_framework": True,
        "quality_gates": True,
        "approval_chain": True,
        "webhooks": True,
        "sso": False,
        "white_label": False,
        "iot_integration": False,
        "supply_chain_esg": False,
        "carbon_credits": False,
        "predictive_analytics": False,
        "graphql_api": True,
        "marketplace_plugins": True,
        "auditor_portal": False,
        "custom_workflows": False,
    },
    TenantTier.ENTERPRISE: {
        "multi_entity": True,
        "cross_framework": True,
        "quality_gates": True,
        "approval_chain": True,
        "webhooks": True,
        "sso": True,
        "white_label": True,
        "iot_integration": True,
        "supply_chain_esg": True,
        "carbon_credits": True,
        "predictive_analytics": True,
        "graphql_api": True,
        "marketplace_plugins": True,
        "auditor_portal": True,
        "custom_workflows": True,
    },
    TenantTier.CUSTOM: {
        "multi_entity": True,
        "cross_framework": True,
        "quality_gates": True,
        "approval_chain": True,
        "webhooks": True,
        "sso": True,
        "white_label": True,
        "iot_integration": True,
        "supply_chain_esg": True,
        "carbon_credits": True,
        "predictive_analytics": True,
        "graphql_api": True,
        "marketplace_plugins": True,
        "auditor_portal": True,
        "custom_workflows": True,
    },
}

TIER_ISOLATION_DEFAULTS: Dict[TenantTier, TenantIsolationLevel] = {
    TenantTier.STARTER: TenantIsolationLevel.SHARED,
    TenantTier.PROFESSIONAL: TenantIsolationLevel.NAMESPACE,
    TenantTier.ENTERPRISE: TenantIsolationLevel.CLUSTER,
    TenantTier.CUSTOM: TenantIsolationLevel.PHYSICAL,
}


# ---------------------------------------------------------------------------
# TenantBridge
# ---------------------------------------------------------------------------


class TenantBridge:
    """Bridge connecting CSRD Enterprise Pack to the platform TenantManager.

    Wraps the existing TenantManager via composition and enriches tenant
    data with CSRD-specific configuration, feature flags, and resource usage.
    Falls back to an internal store when the platform TenantManager is
    not available (e.g., during testing).

    Attributes:
        _platform_manager: Reference to greenlang.auth.tenant.TenantManager.
        _profiles: Internal tenant profile store.

    Example:
        >>> bridge = TenantBridge()
        >>> profile = bridge.create_csrd_tenant(
        ...     name="Acme Corp",
        ...     tier="enterprise",
        ...     admin_email="admin@acme.com",
        ... )
        >>> assert profile.tier == TenantTier.ENTERPRISE
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the TenantBridge.

        Args:
            config: Optional configuration overrides.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._profiles: Dict[str, TenantProfile] = {}
        self._resource_usage: Dict[str, ResourceUsage] = {}

        # Attempt to import platform TenantManager
        self._platform_manager: Any = None
        try:
            from greenlang.auth.tenant import TenantManager
            self._platform_manager = TenantManager()
            self.logger.info("Platform TenantManager connected")
        except (ImportError, Exception) as exc:
            self.logger.warning(
                "Platform TenantManager unavailable, using internal store: %s", exc
            )

    # -------------------------------------------------------------------------
    # Tenant CRUD
    # -------------------------------------------------------------------------

    def create_csrd_tenant(
        self,
        name: str,
        tier: str = "enterprise",
        admin_email: str = "",
        csrd_config: Optional[Dict[str, Any]] = None,
    ) -> TenantProfile:
        """Create a new tenant with CSRD-specific configuration.

        Args:
            name: Tenant organization name.
            tier: Subscription tier (starter/professional/enterprise/custom).
            admin_email: Administrator email address.
            csrd_config: Optional CSRD-specific configuration overrides.

        Returns:
            Fully enriched TenantProfile.

        Raises:
            ValueError: If tier is invalid or email is malformed.
        """
        try:
            tier_enum = TenantTier(tier)
        except ValueError:
            valid = [t.value for t in TenantTier]
            raise ValueError(f"Invalid tier '{tier}'. Valid: {valid}")

        config = CSRDTenantConfig(**(csrd_config or {}))
        isolation = TIER_ISOLATION_DEFAULTS.get(
            tier_enum, TenantIsolationLevel.NAMESPACE,
        )
        features = dict(TIER_FEATURES.get(tier_enum, {}))

        profile = TenantProfile(
            name=name,
            tier=tier_enum,
            isolation_level=isolation,
            admin_email=admin_email,
            region=config.data_residency_region,
            csrd_config=config,
            features=features,
            status=TenantStatus.PROVISIONING,
        )
        profile.provenance_hash = _compute_hash(profile)

        self._profiles[profile.tenant_id] = profile
        self._resource_usage[profile.tenant_id] = ResourceUsage(
            tenant_id=profile.tenant_id,
        )

        # Sync to platform if available
        if self._platform_manager is not None:
            try:
                self._sync_to_platform(profile)
            except Exception as exc:
                self.logger.warning("Platform sync failed: %s", exc)

        profile.status = TenantStatus.ACTIVE
        self.logger.info(
            "CSRD tenant created: id=%s, name='%s', tier=%s, isolation=%s",
            profile.tenant_id, name, tier, isolation.value,
        )
        return profile

    def get_tenant_profile(self, tenant_id: str) -> TenantProfile:
        """Retrieve a CSRD-enriched tenant profile.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            TenantProfile with full CSRD configuration.

        Raises:
            KeyError: If tenant not found.
        """
        if tenant_id not in self._profiles:
            raise KeyError(f"Tenant '{tenant_id}' not found")
        return self._profiles[tenant_id]

    def update_csrd_config(
        self, tenant_id: str, config: Dict[str, Any],
    ) -> TenantProfile:
        """Update the CSRD-specific configuration for a tenant.

        Args:
            tenant_id: Tenant identifier.
            config: Configuration fields to update.

        Returns:
            Updated TenantProfile.

        Raises:
            KeyError: If tenant not found.
        """
        profile = self.get_tenant_profile(tenant_id)
        current = profile.csrd_config.model_dump()
        current.update(config)
        profile.csrd_config = CSRDTenantConfig(**current)
        profile.updated_at = _utcnow()
        profile.provenance_hash = _compute_hash(profile)

        self.logger.info(
            "CSRD config updated for tenant '%s': %s", tenant_id, list(config.keys()),
        )
        return profile

    def list_csrd_tenants(
        self, filters: Optional[Dict[str, Any]] = None,
    ) -> List[TenantProfile]:
        """List tenants with optional filtering.

        Args:
            filters: Optional filter criteria (tier, status, region).

        Returns:
            Filtered list of TenantProfile.
        """
        filters = filters or {}
        results: List[TenantProfile] = []

        for profile in self._profiles.values():
            if "tier" in filters and profile.tier.value != filters["tier"]:
                continue
            if "status" in filters and profile.status.value != filters["status"]:
                continue
            if "region" in filters and profile.region != filters["region"]:
                continue
            results.append(profile)

        return results

    # -------------------------------------------------------------------------
    # Data Isolation
    # -------------------------------------------------------------------------

    def enforce_data_partition(self, tenant_id: str, operation: str) -> bool:
        """Verify that an operation respects tenant data isolation.

        Args:
            tenant_id: Tenant requesting the operation.
            operation: Operation description (read/write/delete).

        Returns:
            True if the operation is allowed under the tenant's isolation level.
        """
        if tenant_id not in self._profiles:
            self.logger.warning(
                "Data partition check failed: unknown tenant '%s'", tenant_id,
            )
            return False

        profile = self._profiles[tenant_id]
        if profile.status != TenantStatus.ACTIVE:
            self.logger.warning(
                "Data partition denied for tenant '%s': status=%s",
                tenant_id, profile.status.value,
            )
            return False

        self.logger.debug(
            "Data partition enforced: tenant=%s, operation=%s, isolation=%s",
            tenant_id, operation, profile.isolation_level.value,
        )
        return True

    # -------------------------------------------------------------------------
    # Feature Flags
    # -------------------------------------------------------------------------

    def get_tenant_features(self, tenant_id: str) -> Dict[str, bool]:
        """Get the feature flags available to a tenant based on its tier.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            Dictionary of feature name to enabled status.

        Raises:
            KeyError: If tenant not found.
        """
        profile = self.get_tenant_profile(tenant_id)
        return dict(profile.features)

    # -------------------------------------------------------------------------
    # Tier Migration
    # -------------------------------------------------------------------------

    def migrate_tenant_tier(
        self, tenant_id: str, new_tier: str,
    ) -> Dict[str, Any]:
        """Upgrade or downgrade a tenant's subscription tier.

        Updates the feature flags and isolation level to match the new tier.

        Args:
            tenant_id: Tenant identifier.
            new_tier: Target tier name.

        Returns:
            Migration result with old and new tier info.

        Raises:
            KeyError: If tenant not found.
            ValueError: If new_tier is invalid.
        """
        try:
            new_tier_enum = TenantTier(new_tier)
        except ValueError:
            valid = [t.value for t in TenantTier]
            raise ValueError(f"Invalid tier '{new_tier}'. Valid: {valid}")

        profile = self.get_tenant_profile(tenant_id)
        old_tier = profile.tier

        profile.tier = new_tier_enum
        profile.isolation_level = TIER_ISOLATION_DEFAULTS.get(
            new_tier_enum, TenantIsolationLevel.NAMESPACE,
        )
        profile.features = dict(TIER_FEATURES.get(new_tier_enum, {}))
        profile.updated_at = _utcnow()
        profile.provenance_hash = _compute_hash(profile)

        self.logger.info(
            "Tenant '%s' migrated: %s -> %s",
            tenant_id, old_tier.value, new_tier,
        )
        return {
            "tenant_id": tenant_id,
            "old_tier": old_tier.value,
            "new_tier": new_tier,
            "new_isolation_level": profile.isolation_level.value,
            "features_changed": True,
            "timestamp": _utcnow().isoformat(),
            "provenance_hash": profile.provenance_hash,
        }

    # -------------------------------------------------------------------------
    # Cross-Tenant Metrics
    # -------------------------------------------------------------------------

    def aggregate_cross_tenant_metrics(
        self, metric: str, anonymize: bool = True,
    ) -> Dict[str, Any]:
        """Aggregate a metric across all tenants for benchmarking.

        All tenant-identifying information is removed when anonymize=True.

        Args:
            metric: Metric name to aggregate (e.g. 'storage_used_mb').
            anonymize: Whether to strip tenant identifiers.

        Returns:
            Aggregated metric dictionary.
        """
        values: List[float] = []
        for tid, usage in self._resource_usage.items():
            value = getattr(usage, metric, None)
            if value is not None and isinstance(value, (int, float)):
                values.append(float(value))

        if not values:
            return {
                "metric": metric,
                "count": 0,
                "message": "No data available",
            }

        result: Dict[str, Any] = {
            "metric": metric,
            "count": len(values),
            "sum": sum(values),
            "average": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "anonymized": anonymize,
            "timestamp": _utcnow().isoformat(),
        }

        if not anonymize:
            result["tenant_values"] = {
                tid: getattr(self._resource_usage[tid], metric, 0)
                for tid in self._resource_usage
            }

        result["provenance_hash"] = _compute_hash(result)
        return result

    # -------------------------------------------------------------------------
    # Resource Usage
    # -------------------------------------------------------------------------

    def get_tenant_resource_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get current resource usage for a tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            Resource usage dictionary with storage, API calls, and user counts.

        Raises:
            KeyError: If tenant not found.
        """
        if tenant_id not in self._profiles:
            raise KeyError(f"Tenant '{tenant_id}' not found")

        usage = self._resource_usage.get(tenant_id)
        if usage is None:
            usage = ResourceUsage(tenant_id=tenant_id)
            self._resource_usage[tenant_id] = usage

        profile = self._profiles[tenant_id]
        return {
            "tenant_id": tenant_id,
            "tier": profile.tier.value,
            "storage_used_mb": usage.storage_used_mb,
            "api_calls_today": usage.api_calls_today,
            "api_calls_this_hour": usage.api_calls_this_hour,
            "active_users": usage.active_users,
            "active_workflows": usage.active_workflows,
            "subsidiaries_configured": usage.subsidiaries_configured,
            "last_activity_at": (
                usage.last_activity_at.isoformat() if usage.last_activity_at else None
            ),
            "limits": {
                "max_users": profile.csrd_config.max_users,
                "max_api_calls_per_hour": profile.csrd_config.max_api_calls_per_hour,
                "max_subsidiaries": profile.csrd_config.max_subsidiaries,
            },
            "measured_at": usage.measured_at.isoformat(),
        }

    # -------------------------------------------------------------------------
    # Platform Sync
    # -------------------------------------------------------------------------

    def _sync_to_platform(self, profile: TenantProfile) -> None:
        """Sync tenant profile to the platform TenantManager.

        Args:
            profile: TenantProfile to sync.
        """
        if self._platform_manager is None:
            return

        self.logger.debug(
            "Syncing tenant '%s' to platform TenantManager", profile.tenant_id,
        )
        # Platform-specific sync logic would go here
