# -*- coding: utf-8 -*-
"""
MarketplaceBridge - Plugin Marketplace Bridge for CSRD Enterprise Pack
========================================================================

This module connects the CSRD Enterprise Pack to the platform's plugin
marketplace (greenlang/ecosystem/marketplace/) for plugin discovery,
tenant-scoped installation, compatibility checking, usage tracking,
and quota enforcement.

Platform Integration:
    greenlang/ecosystem/marketplace/ -> Search, Categories, Versioning
    greenlang/ecosystem/marketplace/validator.py -> Plugin Validator
    greenlang/ecosystem/marketplace/dependency_resolver.py -> Dependency Resolver

Architecture:
    Plugin Registry --> MarketplaceBridge --> Compatibility Check
                             |                      |
                             v                      v
    Tenant Quota <-- Install/Uninstall <-- Dependency Resolver
                             |
                             v
    Usage Metrics --> Plugin Configuration --> Feature Flags

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

from pydantic import BaseModel, Field

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


class PluginCategory(str, Enum):
    """Plugin marketplace categories."""

    DATA_CONNECTOR = "data_connector"
    REPORTING = "reporting"
    ANALYTICS = "analytics"
    COMPLIANCE = "compliance"
    INTEGRATION = "integration"
    VISUALIZATION = "visualization"
    AUTOMATION = "automation"
    INDUSTRY = "industry"


class PluginStatus(str, Enum):
    """Plugin installation status."""

    AVAILABLE = "available"
    INSTALLED = "installed"
    UPDATING = "updating"
    DISABLED = "disabled"
    DEPRECATED = "deprecated"


class CompatibilityLevel(str, Enum):
    """Plugin compatibility levels."""

    FULL = "full"
    PARTIAL = "partial"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class PluginInfo(BaseModel):
    """Plugin information from the marketplace catalog."""

    plugin_id: str = Field(default_factory=_new_uuid)
    name: str = Field(...)
    description: str = Field(default="")
    version: str = Field(default="1.0.0")
    category: PluginCategory = Field(default=PluginCategory.INTEGRATION)
    author: str = Field(default="")
    license: str = Field(default="MIT")
    min_pack_version: str = Field(default="3.0.0")
    max_pack_version: Optional[str] = Field(None)
    dependencies: List[str] = Field(default_factory=list)
    rating: float = Field(default=0.0, ge=0.0, le=5.0)
    downloads: int = Field(default=0, ge=0)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: Optional[datetime] = Field(None)
    provenance_hash: str = Field(default="")


class InstallResult(BaseModel):
    """Result of a plugin installation or update."""

    install_id: str = Field(default_factory=_new_uuid)
    plugin_id: str = Field(...)
    tenant_id: str = Field(...)
    version: str = Field(...)
    status: str = Field(default="success")
    installed_at: datetime = Field(default_factory=_utcnow)
    dependencies_resolved: List[str] = Field(default_factory=list)
    error_message: Optional[str] = Field(None)
    provenance_hash: str = Field(default="")


class InstalledPlugin(BaseModel):
    """Plugin installed for a specific tenant."""

    plugin_id: str = Field(...)
    tenant_id: str = Field(...)
    name: str = Field(...)
    version: str = Field(...)
    category: PluginCategory = Field(default=PluginCategory.INTEGRATION)
    status: PluginStatus = Field(default=PluginStatus.INSTALLED)
    config: Dict[str, Any] = Field(default_factory=dict)
    installed_at: datetime = Field(default_factory=_utcnow)
    updated_at: Optional[datetime] = Field(None)
    api_calls_total: int = Field(default=0)
    last_used_at: Optional[datetime] = Field(None)


class PluginUsageMetrics(BaseModel):
    """Usage metrics for an installed plugin."""

    plugin_id: str = Field(...)
    tenant_id: str = Field(...)
    api_calls_total: int = Field(default=0)
    api_calls_today: int = Field(default=0)
    api_calls_this_month: int = Field(default=0)
    errors_total: int = Field(default=0)
    avg_latency_ms: float = Field(default=0.0)
    last_used_at: Optional[datetime] = Field(None)
    data_processed_mb: float = Field(default=0.0)
    measured_at: datetime = Field(default_factory=_utcnow)


class CompatibilityResult(BaseModel):
    """Plugin compatibility check result."""

    plugin_id: str = Field(...)
    pack_version: str = Field(...)
    compatibility: CompatibilityLevel = Field(default=CompatibilityLevel.UNKNOWN)
    issues: List[str] = Field(default_factory=list)
    missing_dependencies: List[str] = Field(default_factory=list)
    recommendation: str = Field(default="")


# ---------------------------------------------------------------------------
# Tier Plugin Limits
# ---------------------------------------------------------------------------


TIER_PLUGIN_LIMITS: Dict[str, int] = {
    "starter": 5,
    "professional": 20,
    "enterprise": 100,
    "custom": 500,
}


# ---------------------------------------------------------------------------
# Catalog (built-in CSRD plugins)
# ---------------------------------------------------------------------------

BUILTIN_PLUGINS: List[Dict[str, Any]] = [
    {
        "name": "GRI Standards Mapper",
        "description": "Maps CSRD/ESRS disclosures to GRI Standards",
        "category": "compliance",
        "tags": ["gri", "mapping", "cross-framework"],
    },
    {
        "name": "SASB Industry Metrics",
        "description": "Industry-specific SASB metrics integration",
        "category": "compliance",
        "tags": ["sasb", "industry", "metrics"],
    },
    {
        "name": "CDP Questionnaire Sync",
        "description": "Synchronize CSRD data with CDP questionnaire",
        "category": "integration",
        "tags": ["cdp", "questionnaire", "sync"],
    },
    {
        "name": "EcoVadis Connector",
        "description": "Import supplier ESG scores from EcoVadis",
        "category": "data_connector",
        "tags": ["ecovadis", "supplier", "esg"],
    },
    {
        "name": "Power BI Export",
        "description": "Export CSRD dashboards to Power BI",
        "category": "visualization",
        "tags": ["powerbi", "export", "dashboard"],
    },
    {
        "name": "SAP S/4HANA Connector",
        "description": "Real-time data integration with SAP S/4HANA",
        "category": "data_connector",
        "tags": ["sap", "erp", "real-time"],
    },
    {
        "name": "Science-Based Targets Tracker",
        "description": "Track progress against SBTi validated targets",
        "category": "analytics",
        "tags": ["sbti", "targets", "tracking"],
    },
    {
        "name": "XBRL iXBRL Generator",
        "description": "Generate iXBRL-tagged ESRS disclosures for filing",
        "category": "reporting",
        "tags": ["xbrl", "ixbrl", "filing"],
    },
]


# ---------------------------------------------------------------------------
# MarketplaceBridge
# ---------------------------------------------------------------------------


class MarketplaceBridge:
    """Plugin marketplace bridge for CSRD Enterprise Pack.

    Manages plugin discovery, tenant-scoped installation, compatibility
    checking, usage tracking, and quota enforcement per tenant tier.

    Attributes:
        _catalog: Plugin catalog.
        _installed: Plugins installed per tenant.
        _usage: Usage metrics per tenant/plugin.

    Example:
        >>> bridge = MarketplaceBridge()
        >>> plugins = bridge.discover_plugins("compliance")
        >>> result = bridge.install_plugin("t-1", plugins[0].plugin_id, "1.0.0")
        >>> assert result.status == "success"
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Marketplace Bridge.

        Args:
            config: Optional configuration overrides.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}

        self._catalog: Dict[str, PluginInfo] = {}
        self._installed: Dict[str, Dict[str, InstalledPlugin]] = {}
        self._usage: Dict[str, Dict[str, PluginUsageMetrics]] = {}

        # Attempt to connect platform marketplace
        self._platform_marketplace: Any = None
        try:
            from greenlang.ecosystem.marketplace.search import MarketplaceSearch
            self._platform_marketplace = MarketplaceSearch
            self.logger.info("Platform marketplace connected")
        except (ImportError, Exception) as exc:
            self.logger.warning("Platform marketplace unavailable: %s", exc)

        # Initialize built-in catalog
        self._initialize_catalog()
        self.logger.info("MarketplaceBridge initialized with %d plugins", len(self._catalog))

    def _initialize_catalog(self) -> None:
        """Initialize the built-in plugin catalog."""
        for plugin_data in BUILTIN_PLUGINS:
            try:
                category = PluginCategory(plugin_data.get("category", "integration"))
            except ValueError:
                category = PluginCategory.INTEGRATION

            plugin = PluginInfo(
                name=plugin_data["name"],
                description=plugin_data.get("description", ""),
                category=category,
                author="GreenLang Platform Team",
                tags=plugin_data.get("tags", []),
                rating=4.5,
            )
            plugin.provenance_hash = _compute_hash(plugin)
            self._catalog[plugin.plugin_id] = plugin

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    def discover_plugins(
        self,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> List[PluginInfo]:
        """Discover plugins from the marketplace catalog.

        Args:
            category: Optional category filter.
            search_query: Optional text search query.

        Returns:
            Filtered list of PluginInfo.
        """
        results: List[PluginInfo] = []
        for plugin in self._catalog.values():
            if category:
                try:
                    cat_enum = PluginCategory(category)
                    if plugin.category != cat_enum:
                        continue
                except ValueError:
                    pass

            if search_query:
                query_lower = search_query.lower()
                searchable = f"{plugin.name} {plugin.description} {' '.join(plugin.tags)}".lower()
                if query_lower not in searchable:
                    continue

            results.append(plugin)

        self.logger.info(
            "Plugin discovery: category=%s, query=%s, results=%d",
            category, search_query, len(results),
        )
        return results

    # -------------------------------------------------------------------------
    # Installation
    # -------------------------------------------------------------------------

    def install_plugin(
        self, tenant_id: str, plugin_id: str, version: str = "1.0.0",
    ) -> InstallResult:
        """Install a plugin for a specific tenant.

        Args:
            tenant_id: Tenant identifier.
            plugin_id: Plugin identifier from catalog.
            version: Version to install.

        Returns:
            InstallResult with installation status.
        """
        if plugin_id not in self._catalog:
            result = InstallResult(
                plugin_id=plugin_id, tenant_id=tenant_id, version=version,
                status="failed", error_message="Plugin not found in catalog",
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Check quota
        quota_check = self.enforce_plugin_quotas(tenant_id)
        if not quota_check.get("within_quota", True):
            result = InstallResult(
                plugin_id=plugin_id, tenant_id=tenant_id, version=version,
                status="failed", error_message="Plugin quota exceeded",
            )
            result.provenance_hash = _compute_hash(result)
            return result

        plugin_info = self._catalog[plugin_id]

        if tenant_id not in self._installed:
            self._installed[tenant_id] = {}

        installed = InstalledPlugin(
            plugin_id=plugin_id,
            tenant_id=tenant_id,
            name=plugin_info.name,
            version=version,
            category=plugin_info.category,
        )
        self._installed[tenant_id][plugin_id] = installed

        result = InstallResult(
            plugin_id=plugin_id,
            tenant_id=tenant_id,
            version=version,
            status="success",
            dependencies_resolved=plugin_info.dependencies,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Plugin installed: tenant=%s, plugin='%s', version=%s",
            tenant_id, plugin_info.name, version,
        )
        return result

    def uninstall_plugin(
        self, tenant_id: str, plugin_id: str,
    ) -> Dict[str, Any]:
        """Uninstall a plugin from a tenant.

        Args:
            tenant_id: Tenant identifier.
            plugin_id: Plugin identifier.

        Returns:
            Uninstall result.
        """
        tenant_plugins = self._installed.get(tenant_id, {})
        if plugin_id not in tenant_plugins:
            return {
                "tenant_id": tenant_id,
                "plugin_id": plugin_id,
                "uninstalled": False,
                "reason": "Plugin not installed",
            }

        del tenant_plugins[plugin_id]

        # Clean up usage
        if tenant_id in self._usage and plugin_id in self._usage[tenant_id]:
            del self._usage[tenant_id][plugin_id]

        self.logger.info(
            "Plugin uninstalled: tenant=%s, plugin=%s", tenant_id, plugin_id,
        )
        return {
            "tenant_id": tenant_id,
            "plugin_id": plugin_id,
            "uninstalled": True,
            "timestamp": _utcnow().isoformat(),
        }

    def update_plugin(
        self, tenant_id: str, plugin_id: str, target_version: str,
    ) -> InstallResult:
        """Update an installed plugin to a new version.

        Args:
            tenant_id: Tenant identifier.
            plugin_id: Plugin identifier.
            target_version: Target version to update to.

        Returns:
            InstallResult with update status.
        """
        tenant_plugins = self._installed.get(tenant_id, {})
        if plugin_id not in tenant_plugins:
            result = InstallResult(
                plugin_id=plugin_id, tenant_id=tenant_id, version=target_version,
                status="failed", error_message="Plugin not installed",
            )
            result.provenance_hash = _compute_hash(result)
            return result

        installed = tenant_plugins[plugin_id]
        installed.version = target_version
        installed.updated_at = _utcnow()
        installed.status = PluginStatus.INSTALLED

        result = InstallResult(
            plugin_id=plugin_id,
            tenant_id=tenant_id,
            version=target_version,
            status="success",
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Plugin updated: tenant=%s, plugin=%s, version=%s",
            tenant_id, plugin_id, target_version,
        )
        return result

    # -------------------------------------------------------------------------
    # Compatibility
    # -------------------------------------------------------------------------

    def check_compatibility(
        self, plugin_id: str, pack_version: str,
    ) -> CompatibilityResult:
        """Check plugin compatibility with a pack version.

        Args:
            plugin_id: Plugin identifier.
            pack_version: Pack version string.

        Returns:
            CompatibilityResult.
        """
        if plugin_id not in self._catalog:
            return CompatibilityResult(
                plugin_id=plugin_id,
                pack_version=pack_version,
                compatibility=CompatibilityLevel.UNKNOWN,
                issues=["Plugin not found in catalog"],
            )

        plugin = self._catalog[plugin_id]

        # Simple version comparison
        if pack_version >= plugin.min_pack_version:
            if plugin.max_pack_version is None or pack_version <= plugin.max_pack_version:
                return CompatibilityResult(
                    plugin_id=plugin_id,
                    pack_version=pack_version,
                    compatibility=CompatibilityLevel.FULL,
                    recommendation="Plugin is fully compatible",
                )

        return CompatibilityResult(
            plugin_id=plugin_id,
            pack_version=pack_version,
            compatibility=CompatibilityLevel.PARTIAL,
            issues=[f"Requires pack version >= {plugin.min_pack_version}"],
            recommendation="Consider upgrading pack version",
        )

    # -------------------------------------------------------------------------
    # Tenant Plugin Queries
    # -------------------------------------------------------------------------

    def get_installed_plugins(self, tenant_id: str) -> List[InstalledPlugin]:
        """Get all plugins installed for a tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            List of InstalledPlugin.
        """
        return list(self._installed.get(tenant_id, {}).values())

    def get_plugin_usage(
        self, tenant_id: str, plugin_id: str,
    ) -> PluginUsageMetrics:
        """Get usage metrics for an installed plugin.

        Args:
            tenant_id: Tenant identifier.
            plugin_id: Plugin identifier.

        Returns:
            PluginUsageMetrics.
        """
        if tenant_id not in self._usage:
            self._usage[tenant_id] = {}

        if plugin_id not in self._usage[tenant_id]:
            self._usage[tenant_id][plugin_id] = PluginUsageMetrics(
                plugin_id=plugin_id, tenant_id=tenant_id,
            )

        return self._usage[tenant_id][plugin_id]

    # -------------------------------------------------------------------------
    # Quota Enforcement
    # -------------------------------------------------------------------------

    def enforce_plugin_quotas(self, tenant_id: str) -> Dict[str, Any]:
        """Check and enforce plugin installation quotas per tenant tier.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            Quota enforcement result.
        """
        installed_count = len(self._installed.get(tenant_id, {}))
        tier = self._config.get("tenant_tier", "enterprise")
        max_plugins = TIER_PLUGIN_LIMITS.get(tier, 100)

        within_quota = installed_count < max_plugins

        return {
            "tenant_id": tenant_id,
            "tier": tier,
            "installed_count": installed_count,
            "max_plugins": max_plugins,
            "within_quota": within_quota,
            "remaining": max_plugins - installed_count,
            "timestamp": _utcnow().isoformat(),
        }

    # -------------------------------------------------------------------------
    # Plugin Configuration
    # -------------------------------------------------------------------------

    def configure_plugin(
        self,
        tenant_id: str,
        plugin_id: str,
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Configure an installed plugin for a tenant.

        Args:
            tenant_id: Tenant identifier.
            plugin_id: Plugin identifier.
            settings: Plugin-specific configuration settings.

        Returns:
            Configuration result.
        """
        tenant_plugins = self._installed.get(tenant_id, {})
        if plugin_id not in tenant_plugins:
            return {
                "tenant_id": tenant_id,
                "plugin_id": plugin_id,
                "configured": False,
                "reason": "Plugin not installed",
            }

        installed = tenant_plugins[plugin_id]
        installed.config.update(settings)
        installed.updated_at = _utcnow()

        self.logger.info(
            "Plugin configured: tenant=%s, plugin=%s, keys=%s",
            tenant_id, plugin_id, list(settings.keys()),
        )
        return {
            "tenant_id": tenant_id,
            "plugin_id": plugin_id,
            "configured": True,
            "settings_applied": list(settings.keys()),
            "timestamp": _utcnow().isoformat(),
        }
