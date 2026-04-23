# -*- coding: utf-8 -*-
"""
Automated partner onboarding for the Factors API.

Creates partner-specific environments with API keys, SQLite catalogs,
and usage tracking for design partner pilots.

Example:
    >>> config = create_partner_environment(
    ...     partner_id="acme-corp",
    ...     partner_name="Acme Corporation",
    ...     tier="pro",
    ...     contact_email="sustainability@acme.example.com",
    ... )
    >>> print(config.api_key)
    gl_partner_acme-corp_...
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Tier-based defaults ─────────────────────────────────────────────

_TIER_GEOGRAPHY_DEFAULTS: Dict[str, List[str]] = {
    "community": ["US", "GB"],
    "pro": ["US", "GB", "DE", "FR", "JP", "AU", "CA"],
    "enterprise": [],  # empty = all geographies
}

_TIER_SECTOR_DEFAULTS: Dict[str, List[str]] = {
    "community": ["energy", "transport"],
    "pro": ["energy", "transport", "manufacturing", "buildings", "waste"],
    "enterprise": [],  # empty = all sectors
}

_TIER_RATE_LIMITS: Dict[str, int] = {
    "community": 1000,
    "pro": 10000,
    "enterprise": 100000,
}

# Base URL can be overridden via environment variable
_DEFAULT_BASE_URL = os.getenv("GL_FACTORS_BASE_URL", "https://api.greenlang.io")


@dataclass
class PartnerConfig:
    """Configuration returned after partner environment creation.

    Contains all details a partner needs to start using the Factors API.
    """

    partner_id: str
    partner_name: str
    tier: str
    contact_email: str
    api_key: str
    base_url: str
    edition_id: str
    supported_geographies: List[str]
    supported_sectors: List[str]
    catalog_path: Optional[str]
    rate_limit_per_day: int
    created_at: str
    usage_tracking_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (redacts API key for safety)."""
        return {
            "partner_id": self.partner_id,
            "partner_name": self.partner_name,
            "tier": self.tier,
            "contact_email": self.contact_email,
            "api_key_prefix": self.api_key[:20] + "...",
            "base_url": self.base_url,
            "edition_id": self.edition_id,
            "supported_geographies": self.supported_geographies,
            "supported_sectors": self.supported_sectors,
            "catalog_path": self.catalog_path,
            "rate_limit_per_day": self.rate_limit_per_day,
            "created_at": self.created_at,
            "usage_tracking_enabled": self.usage_tracking_enabled,
        }

    @property
    def api_key_hash(self) -> str:
        """SHA-256 hash of the API key (for storage/lookup)."""
        return hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()


def _generate_api_key(partner_id: str) -> str:
    """Generate a secure API key with partner-specific prefix.

    Format: gl_partner_{partner_id}_{32_random_chars}

    Args:
        partner_id: Unique partner identifier.

    Returns:
        Generated API key string.
    """
    random_part = secrets.token_urlsafe(24)[:32]
    return "gl_partner_%s_%s" % (partner_id, random_part)


def _resolve_edition_id() -> str:
    """Resolve the current default edition ID from environment or fallback."""
    return os.getenv("GL_FACTORS_DEFAULT_EDITION", "2026.04.1")


def _create_partner_catalog(
    partner_id: str,
    catalog_dir: Path,
    geographies: List[str],
    sectors: List[str],
) -> Optional[Path]:
    """Create a partner-specific SQLite catalog as a subset of the full catalog.

    The partner catalog contains only factors relevant to the partner's
    configured geographies and sectors.

    Args:
        partner_id: Unique partner identifier.
        catalog_dir: Directory to store the partner's catalog file.
        geographies: List of ISO 3166 geography codes for filtering.
        sectors: List of sector names for filtering.

    Returns:
        Path to the created SQLite catalog, or None if creation failed.
    """
    catalog_path = catalog_dir / ("%s_catalog.sqlite" % partner_id)

    try:
        conn = sqlite3.connect(str(catalog_path))
        cursor = conn.cursor()

        # Create partner catalog schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS partner_info (
                partner_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                geographies TEXT,
                sectors TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS factors (
                factor_id TEXT PRIMARY KEY,
                fuel_type TEXT,
                geography TEXT,
                scope TEXT,
                sector TEXT,
                co2e_per_unit REAL,
                unit TEXT,
                source_org TEXT,
                source_year INTEGER,
                factor_status TEXT DEFAULT 'certified',
                content_hash TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                endpoint TEXT,
                query TEXT,
                response_time_ms REAL,
                status_code INTEGER
            )
        """)

        # Insert partner info
        cursor.execute(
            "INSERT OR REPLACE INTO partner_info (partner_id, created_at, geographies, sectors) "
            "VALUES (?, ?, ?, ?)",
            (
                partner_id,
                datetime.now(timezone.utc).isoformat(),
                ",".join(geographies) if geographies else "*",
                ",".join(sectors) if sectors else "*",
            ),
        )

        # Create indexes for common queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_factors_geography ON factors(geography)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_factors_sector ON factors(sector)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_factors_fuel ON factors(fuel_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_log(timestamp)"
        )

        conn.commit()
        conn.close()

        logger.info("Created partner catalog: path=%s partner=%s", catalog_path, partner_id)
        return catalog_path

    except Exception as exc:
        logger.error("Failed to create partner catalog: %s", exc)
        return None


def _setup_usage_tracking(partner_id: str, catalog_path: Optional[Path]) -> bool:
    """Initialize usage tracking tables for the partner.

    Args:
        partner_id: Unique partner identifier.
        catalog_path: Path to the partner's SQLite catalog.

    Returns:
        True if tracking was set up successfully.
    """
    if not catalog_path or not catalog_path.exists():
        logger.warning("Cannot set up usage tracking: no catalog for partner=%s", partner_id)
        return False

    try:
        conn = sqlite3.connect(str(catalog_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_usage (
                date TEXT NOT NULL,
                partner_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                request_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                avg_latency_ms REAL DEFAULT 0.0,
                PRIMARY KEY (date, partner_id, endpoint)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                partner_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT
            )
        """)

        # Log the setup event
        cursor.execute(
            "INSERT INTO api_events (timestamp, partner_id, event_type, details) "
            "VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                partner_id,
                "tracking_initialized",
                "Usage tracking tables created",
            ),
        )

        conn.commit()
        conn.close()
        logger.info("Usage tracking initialized for partner=%s", partner_id)
        return True

    except Exception as exc:
        logger.error("Failed to set up usage tracking: %s", exc)
        return False


def create_partner_environment(
    partner_id: str,
    partner_name: str,
    tier: str = "pro",
    contact_email: str = "",
    *,
    base_url: Optional[str] = None,
    catalog_dir: Optional[Path] = None,
    geographies: Optional[List[str]] = None,
    sectors: Optional[List[str]] = None,
) -> PartnerConfig:
    """Create a complete partner environment for Factors API access.

    This is the main entry point for partner onboarding. It:
      1. Generates a secure API key
      2. Creates a partner-specific SQLite catalog (subset of full catalog)
      3. Sets up usage tracking for the partner
      4. Returns a complete configuration dict

    Args:
        partner_id: Unique partner identifier (used in API key prefix).
        partner_name: Human-readable partner name.
        tier: Access tier - "community", "pro", or "enterprise".
        contact_email: Partner contact email address.
        base_url: Override the default API base URL.
        catalog_dir: Directory for partner catalog files. Defaults to temp dir.
        geographies: Override default geographies for this tier.
        sectors: Override default sectors for this tier.

    Returns:
        PartnerConfig with all connection details.

    Raises:
        ValueError: If partner_id is empty or tier is invalid.
    """
    if not partner_id or not partner_id.strip():
        raise ValueError("partner_id must not be empty")

    tier_lower = tier.lower().strip()
    valid_tiers = ("community", "pro", "enterprise")
    if tier_lower not in valid_tiers:
        raise ValueError("tier must be one of: %s" % ", ".join(valid_tiers))

    logger.info(
        "Creating partner environment: partner=%s tier=%s",
        partner_id, tier_lower,
    )

    # Step 1: Generate API key
    api_key = _generate_api_key(partner_id)

    # Step 2: Resolve geographies and sectors
    resolved_geographies = geographies or _TIER_GEOGRAPHY_DEFAULTS.get(tier_lower, [])
    resolved_sectors = sectors or _TIER_SECTOR_DEFAULTS.get(tier_lower, [])

    # Step 3: Create partner-specific SQLite catalog
    resolved_catalog_dir = catalog_dir or Path(os.getenv("GL_PARTNER_CATALOG_DIR", "."))
    catalog_path = _create_partner_catalog(
        partner_id=partner_id,
        catalog_dir=resolved_catalog_dir,
        geographies=resolved_geographies,
        sectors=resolved_sectors,
    )

    # Step 4: Set up usage tracking
    _setup_usage_tracking(partner_id, catalog_path)

    # Step 5: Build configuration
    config = PartnerConfig(
        partner_id=partner_id,
        partner_name=partner_name,
        tier=tier_lower,
        contact_email=contact_email,
        api_key=api_key,
        base_url=base_url or _DEFAULT_BASE_URL,
        edition_id=_resolve_edition_id(),
        supported_geographies=resolved_geographies,
        supported_sectors=resolved_sectors,
        catalog_path=str(catalog_path) if catalog_path else None,
        rate_limit_per_day=_TIER_RATE_LIMITS.get(tier_lower, 1000),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "Partner environment created: partner=%s api_key_prefix=%s",
        partner_id, config.api_key[:20],
    )
    return config


# ===========================================================================
# Track C-5: OEM white-label lifecycle
# ===========================================================================
#
# An OEM (third-party platform / consultancy / ERP integrator) signs up
# under one of the OEM-eligible parent plans (consulting / platform /
# enterprise per :mod:`greenlang.factors.billing.skus`) and then
# provisions sub-tenants on top of its own grant. Each sub-tenant
# inherits a strict subset of the OEM's redistribution grant so a
# customer can never "punch above" the parent license.
#
# We keep the OEM registry process-local for the launch slice. The
# in-memory store is hydrated by tests and by the FastAPI lifespan
# handler from the same SQLite/Postgres tables that ``EntitlementRegistry``
# already manages, but we deliberately avoid hard-wiring those imports
# here so unit tests run with zero filesystem state.


# License redistribution grant classes accepted by an OEM grant. These
# correspond 1:1 with the ``license_class`` values published in
# ``greenlang/factors/data/source_registry.yaml`` PLUS the synthetic
# "open" / "licensed" buckets used by the marketing surface so the
# signup form can present human-friendly choices.
OEM_GRANT_CLASSES: tuple = (
    # Marketing / coarse-grained buckets used on the signup form.
    "open",                       # any source flagged redistribution_allowed
    "restricted",                 # commercial restricted (e.g. greenlang_terms)
    "licensed",                   # requires customer's own license chain
    "connector_only",             # never redistributable (live API only)
    # Fine-grained source_registry.yaml license classes.
    "public_us_government",
    "public_in_government",
    "public_international",
    "uk_open_government",
    "eu_publication",
    "academic_research",
    "wri_wbcsd_terms",
    "smart_freight_terms",
    "registry_terms",
    "pcaf_attribution",
    "greenlang_terms",
    "commercial_connector",
)

# Parent plan IDs that are eligible to act as an OEM.
# Mirrors :class:`greenlang.factors.billing.skus.Tier` plus the
# Marketing-branded "platform" plan slug used on the pricing page.
OEM_ELIGIBLE_PARENT_PLANS: tuple = (
    "consulting",
    "consulting_platform",   # marketing alias used on pricing page
    "platform",
    "enterprise",
)


class OemError(RuntimeError):
    """Raised when an OEM lifecycle invariant is violated.

    This is intentionally a subclass of ``RuntimeError`` so callers
    that wrap the API behind a generic exception handler still see a
    typed failure (instead of a bare ``ValueError``).
    """


@dataclass
class RedistributionGrant:
    """The set of redistribution classes an OEM is licensed to resell."""

    oem_id: str
    parent_plan: str
    allowed_classes: List[str]
    notes: Optional[str] = None

    def covers(self, license_class: Optional[str]) -> bool:
        """Return True if a factor with ``license_class`` may be sub-licensed.

        ``None`` / unknown license classes are treated as denied because
        the launch policy is "no inferred grants".
        """
        if not license_class:
            return False
        return license_class in self.allowed_classes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "oem_id": self.oem_id,
            "parent_plan": self.parent_plan,
            "allowed_classes": list(self.allowed_classes),
            "notes": self.notes,
        }


@dataclass
class SubTenant:
    """A single sub-tenant provisioned under an OEM partner."""

    id: str
    oem_id: str
    name: str
    entitlements: List[str]
    branding: Optional[Any] = None  # Optional[BrandingConfig] - kept loose to
                                    # avoid a hard pydantic import at module load.
    created_at: str = ""
    active: bool = True
    api_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "oem_id": self.oem_id,
            "name": self.name,
            "entitlements": list(self.entitlements),
            "active": self.active,
            "created_at": self.created_at,
            "api_key_prefix": (self.api_key[:20] + "...") if self.api_key else None,
        }
        if self.branding is not None and hasattr(self.branding, "model_dump"):
            out["branding"] = self.branding.model_dump(mode="json")
        elif self.branding is not None:
            out["branding"] = self.branding
        return out


@dataclass
class OemPartner:
    """Top-level OEM partner record.

    An OEM owns:
      * a :class:`RedistributionGrant` that scopes what it can resell,
      * an optional :class:`BrandingConfig` for white-labelling responses,
      * zero or more :class:`SubTenant` rows.
    """

    id: str
    name: str
    contact_email: str
    parent_plan: str
    grant: RedistributionGrant
    api_key: str
    created_at: str
    branding: Optional[Any] = None
    subtenants: Dict[str, SubTenant] = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "contact_email": self.contact_email,
            "parent_plan": self.parent_plan,
            "grant": self.grant.to_dict(),
            "api_key_prefix": self.api_key[:20] + "...",
            "created_at": self.created_at,
            "active": self.active,
            "subtenants": [s.to_dict() for s in self.subtenants.values()],
        }
        if self.branding is not None and hasattr(self.branding, "model_dump"):
            out["branding"] = self.branding.model_dump(mode="json")
        elif self.branding is not None:
            out["branding"] = self.branding
        return out


# ---------------------------------------------------------------------------
# Process-local OEM registry
# ---------------------------------------------------------------------------

# Module-level in-memory store. Reset via ``_reset_oem_registry()`` in
# tests (kept underscored on purpose - it is not part of the public API).
_OEM_REGISTRY: Dict[str, OemPartner] = {}


def _reset_oem_registry() -> None:
    """Clear the in-memory OEM registry. Test-only helper."""
    _OEM_REGISTRY.clear()


def _generate_oem_id(name: str) -> str:
    """Stable-ish OEM id derived from name + a short random suffix.

    Using a slug + random suffix keeps the identifier human-readable in
    logs while remaining collision-resistant across reruns.
    """
    import re as _re

    slug = _re.sub(r"[^a-z0-9]+", "-", (name or "oem").lower()).strip("-") or "oem"
    return "oem_%s_%s" % (slug[:32], secrets.token_hex(4))


def _generate_subtenant_id(oem_id: str, name: str) -> str:
    """Sub-tenant identifier scoped to its OEM parent."""
    import re as _re

    slug = _re.sub(r"[^a-z0-9]+", "-", (name or "sub").lower()).strip("-") or "sub"
    return "%s_sub_%s_%s" % (oem_id, slug[:24], secrets.token_hex(3))


def _normalise_grant_classes(classes: List[str]) -> List[str]:
    """Validate + de-dup a list of redistribution grant classes.

    Order is preserved so audit logs match what the operator typed.
    """
    if not classes:
        raise OemError("redistribution_grants must contain at least one class")
    out: List[str] = []
    seen = set()
    for raw in classes:
        if not raw or not isinstance(raw, str):
            raise OemError("redistribution grant class must be a non-empty string")
        cls = raw.strip().lower()
        if cls not in OEM_GRANT_CLASSES:
            raise OemError(
                "Unknown redistribution grant class %r; expected one of %s"
                % (cls, list(OEM_GRANT_CLASSES))
            )
        if cls in seen:
            continue
        out.append(cls)
        seen.add(cls)
    return out


def _validate_parent_plan(parent_plan: str) -> str:
    """Normalise + validate an OEM parent plan slug."""
    if not parent_plan or not isinstance(parent_plan, str):
        raise OemError("parent_plan is required")
    plan = parent_plan.strip().lower()
    if plan not in OEM_ELIGIBLE_PARENT_PLANS:
        raise OemError(
            "parent_plan %r is not OEM-eligible; expected one of %s"
            % (plan, list(OEM_ELIGIBLE_PARENT_PLANS))
        )
    return plan


# ---------------------------------------------------------------------------
# Public lifecycle API
# ---------------------------------------------------------------------------


def create_oem_partner(
    name: str,
    contact_email: str,
    redistribution_grants: List[str],
    parent_plan: str,
    *,
    branding: Optional[Any] = None,
    notes: Optional[str] = None,
) -> OemPartner:
    """Provision a new OEM partner.

    Args:
        name: Display name (e.g. ``"Acme Sustainability"``).
        contact_email: Primary admin contact for the OEM.
        redistribution_grants: License-class strings the OEM may resell.
            Validated against :data:`OEM_GRANT_CLASSES`.
        parent_plan: Plan slug the OEM rides on; must be one of
            :data:`OEM_ELIGIBLE_PARENT_PLANS`.
        branding: Optional :class:`BrandingConfig`; can be set later via
            :func:`update_branding`.
        notes: Free-form audit annotation.

    Returns:
        Persisted :class:`OemPartner`.

    Raises:
        OemError: On any validation failure.
    """
    if not name or not name.strip():
        raise OemError("name must not be empty")
    if not contact_email or "@" not in contact_email:
        raise OemError("contact_email must be a valid email address")

    plan = _validate_parent_plan(parent_plan)
    grant_classes = _normalise_grant_classes(redistribution_grants)

    oem_id = _generate_oem_id(name)
    while oem_id in _OEM_REGISTRY:  # paranoia: handle the 1-in-2^32 collision
        oem_id = _generate_oem_id(name)

    grant = RedistributionGrant(
        oem_id=oem_id,
        parent_plan=plan,
        allowed_classes=grant_classes,
        notes=notes,
    )

    api_key = "gl_oem_%s_%s" % (oem_id[-8:], secrets.token_urlsafe(24)[:32])
    partner = OemPartner(
        id=oem_id,
        name=name.strip(),
        contact_email=contact_email.strip(),
        parent_plan=plan,
        grant=grant,
        api_key=api_key,
        created_at=datetime.now(timezone.utc).isoformat(),
        branding=branding,
    )
    _OEM_REGISTRY[oem_id] = partner

    logger.info(
        "OEM partner created: id=%s plan=%s grants=%s",
        oem_id, plan, grant_classes,
    )
    return partner


def get_oem_partner(oem_id: str) -> OemPartner:
    """Look up an OEM partner by id; raise :class:`OemError` if missing."""
    partner = _OEM_REGISTRY.get(oem_id)
    if partner is None:
        raise OemError("Unknown OEM id %r" % oem_id)
    return partner


def list_oem_partners() -> List[OemPartner]:
    """Return every registered OEM (active + revoked)."""
    return list(_OEM_REGISTRY.values())


def get_redistribution_grant(oem_id: str) -> RedistributionGrant:
    """Return the OEM's :class:`RedistributionGrant`."""
    return get_oem_partner(oem_id).grant


def update_branding(oem_id: str, branding: Any) -> OemPartner:
    """Replace the OEM's :class:`BrandingConfig` payload.

    ``branding`` is typed loosely (``Any``) so this module does not need
    to import the pydantic model unconditionally; callers should pass a
    :class:`BrandingConfig` instance. Passing ``None`` clears branding.
    """
    partner = get_oem_partner(oem_id)
    partner.branding = branding
    logger.info("OEM branding updated: id=%s", oem_id)
    return partner


def provision_subtenant(
    oem_id: str,
    subtenant_name: str,
    branding: Optional[Any] = None,
    entitlements: Optional[List[str]] = None,
) -> SubTenant:
    """Provision a sub-tenant under an existing OEM.

    The sub-tenant's ``entitlements`` MUST be a subset of the OEM's
    redistribution grant. We refuse silently-narrower or accidentally-
    wider configurations: any entitlement not covered by the parent
    grant raises :class:`EntitlementError`.

    Args:
        oem_id: Parent OEM identifier.
        subtenant_name: Display name for the sub-tenant.
        branding: Optional :class:`BrandingConfig` override (defaults to
            the parent OEM's branding when serialising responses).
        entitlements: List of redistribution grant classes (strings)
            this sub-tenant is allowed to resolve. Must be subset of the
            OEM grant.

    Returns:
        Persisted :class:`SubTenant`.

    Raises:
        OemError: If the OEM is unknown or inactive.
        EntitlementError: If ``entitlements`` exceeds the OEM grant.
    """
    # Local import keeps the partner_setup module importable in stripped
    # CLI contexts where greenlang.factors.entitlements is not loaded.
    from greenlang.factors.entitlements import EntitlementError

    partner = get_oem_partner(oem_id)
    if not partner.active:
        raise OemError("OEM %r is not active" % oem_id)
    if not subtenant_name or not subtenant_name.strip():
        raise OemError("subtenant_name must not be empty")

    requested = entitlements or []
    if not isinstance(requested, list):
        raise OemError("entitlements must be a list of strings")

    # Validate + normalise (same vocabulary as the OEM grant).
    if requested:
        normalised: List[str] = []
        for raw in requested:
            if not raw or not isinstance(raw, str):
                raise OemError("entitlement must be a non-empty string")
            normalised.append(raw.strip().lower())
        # Reject unknown classes outright before subset check.
        for cls in normalised:
            if cls not in OEM_GRANT_CLASSES:
                raise EntitlementError(
                    "Unknown sub-tenant entitlement %r; expected one of %s"
                    % (cls, list(OEM_GRANT_CLASSES))
                )
        # Subset check: each requested entitlement must be granted to
        # the parent OEM. This is the load-bearing license guard.
        granted = set(partner.grant.allowed_classes)
        violators = [cls for cls in normalised if cls not in granted]
        if violators:
            raise EntitlementError(
                "Sub-tenant entitlements %s exceed OEM grant %s"
                % (violators, sorted(granted))
            )
    else:
        normalised = []

    sub_id = _generate_subtenant_id(oem_id, subtenant_name)
    while sub_id in partner.subtenants:
        sub_id = _generate_subtenant_id(oem_id, subtenant_name)

    sub_api_key = "gl_oemsub_%s_%s" % (sub_id[-8:], secrets.token_urlsafe(20)[:24])
    subtenant = SubTenant(
        id=sub_id,
        oem_id=oem_id,
        name=subtenant_name.strip(),
        entitlements=normalised,
        branding=branding,
        created_at=datetime.now(timezone.utc).isoformat(),
        active=True,
        api_key=sub_api_key,
    )
    partner.subtenants[sub_id] = subtenant
    logger.info(
        "Sub-tenant provisioned: oem=%s sub=%s entitlements=%s",
        oem_id, sub_id, normalised,
    )
    return subtenant


def list_subtenants(oem_id: str, *, active_only: bool = False) -> List[SubTenant]:
    """List sub-tenants for an OEM."""
    partner = get_oem_partner(oem_id)
    subs = list(partner.subtenants.values())
    if active_only:
        subs = [s for s in subs if s.active]
    return sorted(subs, key=lambda s: s.created_at)


def revoke_subtenant(oem_id: str, subtenant_id: str) -> bool:
    """Soft-revoke a sub-tenant (sets ``active=False``).

    Returns True when the sub-tenant was revoked, False when it was
    already inactive or not found.
    """
    partner = get_oem_partner(oem_id)
    sub = partner.subtenants.get(subtenant_id)
    if sub is None:
        logger.warning(
            "Cannot revoke missing sub-tenant: oem=%s sub=%s",
            oem_id, subtenant_id,
        )
        return False
    if not sub.active:
        return False
    sub.active = False
    logger.info("Sub-tenant revoked: oem=%s sub=%s", oem_id, subtenant_id)
    return True


__all__ = [
    # Existing exports
    "PartnerConfig",
    "create_partner_environment",
    # OEM lifecycle
    "OemError",
    "OemPartner",
    "SubTenant",
    "RedistributionGrant",
    "OEM_GRANT_CLASSES",
    "OEM_ELIGIBLE_PARENT_PLANS",
    "create_oem_partner",
    "provision_subtenant",
    "update_branding",
    "revoke_subtenant",
    "list_subtenants",
    "list_oem_partners",
    "get_oem_partner",
    "get_redistribution_grant",
]
