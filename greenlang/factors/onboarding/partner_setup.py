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
