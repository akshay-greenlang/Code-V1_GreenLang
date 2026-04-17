# -*- coding: utf-8 -*-
"""
Pilot environment provisioner (F091).

Sets up isolated environments for design partners: tenant namespace,
API key, rate limits, tier configuration, and initial factor dataset.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.factors.pilot.registry import PilotPartner, PilotRegistry, PilotTier

logger = logging.getLogger(__name__)


@dataclass
class PilotConfig:
    """Configuration for a pilot partner environment."""

    partner_id: str
    tenant_id: str
    api_key: str
    tier: str
    rate_limit_per_day: int
    rate_limit_per_minute: int
    allowed_endpoints: List[str]
    allowed_sources: List[str]
    max_results_per_query: int
    connector_access: List[str]
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partner_id": self.partner_id,
            "tenant_id": self.tenant_id,
            "api_key_prefix": self.api_key[:8] + "...",
            "tier": self.tier,
            "rate_limit_per_day": self.rate_limit_per_day,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "allowed_endpoints": self.allowed_endpoints,
            "allowed_sources": self.allowed_sources,
            "max_results_per_query": self.max_results_per_query,
            "connector_access": self.connector_access,
            "created_at": self.created_at,
        }


# Tier-based defaults
_TIER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "community": {
        "rate_limit_per_day": 1000,
        "rate_limit_per_minute": 10,
        "max_results_per_query": 25,
        "allowed_endpoints": ["/search", "/detail", "/health"],
        "allowed_sources": ["defra_*", "epa_*", "ipcc_*"],
        "connector_access": [],
    },
    "pro": {
        "rate_limit_per_day": 10000,
        "rate_limit_per_minute": 60,
        "max_results_per_query": 100,
        "allowed_endpoints": ["/search", "/detail", "/match", "/batch", "/health"],
        "allowed_sources": ["*"],
        "connector_access": ["electricity_maps"],
    },
    "enterprise": {
        "rate_limit_per_day": 100000,
        "rate_limit_per_minute": 300,
        "max_results_per_query": 500,
        "allowed_endpoints": ["*"],
        "allowed_sources": ["*"],
        "connector_access": ["iea_statistics", "ecoinvent", "electricity_maps"],
    },
}


class PilotProvisioner:
    """
    Provisions pilot partner environments with tier-appropriate configuration.

    Steps:
      1. Generate API key
      2. Create tenant-scoped config
      3. Register in pilot registry
      4. Return provisioning details
    """

    def __init__(self, registry: PilotRegistry) -> None:
        self._registry = registry
        self._provisions: Dict[str, PilotConfig] = {}

    @staticmethod
    def _generate_api_key(prefix: str = "glf") -> str:
        """Generate a secure API key with prefix."""
        token = secrets.token_urlsafe(32)
        return f"{prefix}_{token}"

    def provision(
        self,
        name: str,
        contact_email: str,
        organization: str,
        tier: PilotTier = PilotTier.PRO,
        use_cases: Optional[List[str]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> PilotConfig:
        """
        Provision a complete pilot environment for a design partner.

        Returns PilotConfig with all access details.
        """
        # Enroll in registry
        partner = self._registry.enroll(
            name=name,
            contact_email=contact_email,
            organization=organization,
            tier=tier,
            use_cases=use_cases,
        )

        # Generate API key
        api_key = self._generate_api_key()

        # Activate partner
        self._registry.activate(partner.partner_id, api_key)

        # Build config from tier defaults
        defaults = _TIER_DEFAULTS.get(tier.value, _TIER_DEFAULTS["community"])
        if overrides:
            defaults = {**defaults, **overrides}

        config = PilotConfig(
            partner_id=partner.partner_id,
            tenant_id=partner.tenant_id,
            api_key=api_key,
            tier=tier.value,
            rate_limit_per_day=defaults["rate_limit_per_day"],
            rate_limit_per_minute=defaults["rate_limit_per_minute"],
            max_results_per_query=defaults["max_results_per_query"],
            allowed_endpoints=defaults["allowed_endpoints"],
            allowed_sources=defaults["allowed_sources"],
            connector_access=defaults["connector_access"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self._provisions[partner.partner_id] = config
        logger.info(
            "Provisioned pilot: partner=%s tier=%s tenant=%s",
            partner.partner_id, tier.value, partner.tenant_id,
        )
        return config

    def get_config(self, partner_id: str) -> Optional[PilotConfig]:
        return self._provisions.get(partner_id)

    def deprovision(self, partner_id: str) -> bool:
        """Deprovision a partner environment."""
        if partner_id in self._provisions:
            del self._provisions[partner_id]
            self._registry.complete(partner_id)
            logger.info("Deprovisioned pilot: %s", partner_id)
            return True
        return False

    def list_provisions(self) -> List[PilotConfig]:
        return list(self._provisions.values())
