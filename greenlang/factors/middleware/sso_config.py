# -*- coding: utf-8 -*-
"""
Per-tenant Single Sign-On (SSO) configuration store (SEC-5).

The Factors API exposes two SSO flavors — SAML 2.0 and OpenID Connect.
Both share the same tenant-scoped configuration shape, stored here:

    SSOTenantConfig
        * tenant_id
        * protocol                 : "saml" | "oidc"
        * enabled                  : bool
        * idp_metadata_url         : discovery URL (OIDC) or metadata URL (SAML)
        * idp_entity_id            : SAML Entity ID / OIDC issuer
        * idp_sso_url              : SSO / authorize endpoint
        * idp_slo_url              : SLO / end-session endpoint (optional)
        * idp_public_cert_pem      : SAML IdP signing cert (PEM, optional for SAML)
        * idp_jwks_uri             : OIDC JWKS (optional for OIDC)
        * sp_entity_id             : our SP entity ID per tenant
        * client_id                : OIDC client id (OIDC only)
        * client_secret            : OIDC client secret (OIDC only)
        * attribute_mappings       : IdP -> GreenLang claim map
        * jit_provisioning         : bool — auto-create user on first login
        * default_role             : default RBAC role for JIT users
        * default_tier             : JIT tier (defaults to "enterprise")
        * allowed_email_domains    : whitelist of email domains
        * created_at / updated_at

The loader resolves configs from three sources, in priority order:

    1. In-memory registry (tests / runtime overrides)
    2. SQL database table ``factors.sso_tenant_config``     (Pro/Enterprise)
    3. Static JSON file via ``GL_FACTORS_SSO_CONFIG_FILE``  (dev fallback)

The file format is JSON keyed by ``tenant_id``::

    {
      "acme-corp": {
        "protocol": "saml",
        "enabled": true,
        "idp_metadata_url": "https://acme.okta.com/app/xxx/sso/saml/metadata",
        "attribute_mappings": {
          "email": "NameID",
          "first_name": "urn:oid:2.5.4.42",
          "last_name":  "urn:oid:2.5.4.4",
          "groups":     "http://schemas.xmlsoap.org/claims/Group"
        },
        "jit_provisioning": true,
        "default_role": "analyst",
        "default_tier": "enterprise",
        "allowed_email_domains": ["acme.com"]
      },
      "globex": {
        "protocol": "oidc",
        "enabled": true,
        "idp_metadata_url": "https://globex.auth0.com/.well-known/openid-configuration",
        "client_id": "abc123",
        "client_secret": "...",
        "attribute_mappings": {
          "email": "email",
          "first_name": "given_name",
          "last_name":  "family_name",
          "groups": "https://globex.example/groups"
        }
      }
    }

Secrets (``client_secret``, ``idp_public_cert_pem``) are kept out of version
control and are injected via Vault -> ExternalSecret at runtime. The file
store is a **developer convenience**, not a production deployment method.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Attribute mappings that apply when a tenant config omits the field.
#: Keys are canonical GreenLang claim names; values are the assertion
#: attribute names coming back from the IdP. These defaults match Okta,
#: Azure AD, and Auth0 "standard profile" SAML/OIDC outputs.
DEFAULT_SAML_ATTRIBUTE_MAP: Dict[str, str] = {
    "email": "NameID",
    "first_name": "urn:oid:2.5.4.42",
    "last_name": "urn:oid:2.5.4.4",
    "groups": "http://schemas.xmlsoap.org/claims/Group",
}

DEFAULT_OIDC_ATTRIBUTE_MAP: Dict[str, str] = {
    "email": "email",
    "first_name": "given_name",
    "last_name": "family_name",
    "groups": "groups",
}


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------

@dataclass
class SSOTenantConfig:
    """Per-tenant SSO configuration.

    See module docstring for field semantics.
    """

    tenant_id: str
    protocol: str  # "saml" | "oidc"
    enabled: bool = True

    idp_metadata_url: Optional[str] = None
    idp_entity_id: Optional[str] = None
    idp_sso_url: Optional[str] = None
    idp_slo_url: Optional[str] = None
    idp_public_cert_pem: Optional[str] = None
    idp_jwks_uri: Optional[str] = None

    sp_entity_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None

    attribute_mappings: Dict[str, str] = field(default_factory=dict)
    jit_provisioning: bool = True
    default_role: str = "viewer"
    default_tier: str = "enterprise"
    allowed_email_domains: List[str] = field(default_factory=list)

    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        proto = (self.protocol or "").lower()
        if proto not in {"saml", "oidc"}:
            raise ValueError(f"Unsupported SSO protocol: {self.protocol!r}")
        self.protocol = proto

        # Fill defaults for attribute mappings.
        defaults = (
            DEFAULT_SAML_ATTRIBUTE_MAP
            if proto == "saml"
            else DEFAULT_OIDC_ATTRIBUTE_MAP
        )
        for k, v in defaults.items():
            self.attribute_mappings.setdefault(k, v)

        if self.sp_entity_id is None:
            self.sp_entity_id = f"urn:greenlang:factors:sp:{self.tenant_id}"

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses. Secrets are **not** redacted here;
        callers must redact before returning to end users."""
        return {
            "tenant_id": self.tenant_id,
            "protocol": self.protocol,
            "enabled": self.enabled,
            "idp_metadata_url": self.idp_metadata_url,
            "idp_entity_id": self.idp_entity_id,
            "idp_sso_url": self.idp_sso_url,
            "idp_slo_url": self.idp_slo_url,
            "sp_entity_id": self.sp_entity_id,
            "client_id": self.client_id,
            "attribute_mappings": dict(self.attribute_mappings),
            "jit_provisioning": self.jit_provisioning,
            "default_role": self.default_role,
            "default_tier": self.default_tier,
            "allowed_email_domains": list(self.allowed_email_domains),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def redacted(self) -> Dict[str, Any]:
        """Safe representation for admin UIs / audit logs."""
        out = self.to_dict()
        if self.client_secret:
            out["client_secret"] = "***REDACTED***"
        if self.idp_public_cert_pem:
            out["idp_public_cert_pem"] = "***REDACTED***"
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SSOConfigRegistry:
    """Thread-safe registry of :class:`SSOTenantConfig` records.

    The registry is the single source of truth for SSO config lookups at
    request time. It is populated once at boot by :func:`load_from_env` and
    mutated by admin operations (``PUT /v1/admin/sso/tenants/{id}``).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._configs: Dict[str, SSOTenantConfig] = {}

    def get(self, tenant_id: str) -> Optional[SSOTenantConfig]:
        with self._lock:
            return self._configs.get(tenant_id)

    def get_by_entity(self, entity_id: str) -> Optional[SSOTenantConfig]:
        """Reverse lookup: given an SP Entity ID (SAML) or client_id (OIDC)
        return the matching tenant config, or ``None``."""
        with self._lock:
            for cfg in self._configs.values():
                if cfg.sp_entity_id == entity_id:
                    return cfg
                if cfg.client_id and cfg.client_id == entity_id:
                    return cfg
            return None

    def put(self, cfg: SSOTenantConfig) -> SSOTenantConfig:
        with self._lock:
            cfg.updated_at = datetime.now(timezone.utc).isoformat()
            if cfg.tenant_id not in self._configs:
                cfg.created_at = cfg.updated_at
            self._configs[cfg.tenant_id] = cfg
            return cfg

    def delete(self, tenant_id: str) -> bool:
        with self._lock:
            return self._configs.pop(tenant_id, None) is not None

    def list(self) -> List[SSOTenantConfig]:
        with self._lock:
            return list(self._configs.values())

    def clear(self) -> None:
        with self._lock:
            self._configs.clear()


_REGISTRY: Optional[SSOConfigRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def default_registry() -> SSOConfigRegistry:
    """Return the process-wide registry, loading from env on first call."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = SSOConfigRegistry()
            try:
                load_from_env(_REGISTRY)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SSO config load failed: %s", exc)
        return _REGISTRY


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_from_env(registry: SSOConfigRegistry) -> int:
    """Populate *registry* from the JSON file referenced by
    ``GL_FACTORS_SSO_CONFIG_FILE`` (if set).

    Returns:
        Number of tenant configs loaded.
    """
    path = os.getenv("GL_FACTORS_SSO_CONFIG_FILE")
    if not path:
        return 0
    return load_from_file(registry, path)


def load_from_file(registry: SSOConfigRegistry, path: str) -> int:
    """Populate *registry* from a JSON file. Missing file is treated as
    zero configs loaded (not an error)."""
    if not os.path.exists(path):
        logger.info("SSO config file %s not found; skipping load", path)
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    n = 0
    for tenant_id, blob in data.items():
        blob = dict(blob)
        blob.setdefault("tenant_id", tenant_id)
        registry.put(SSOTenantConfig(**blob))
        n += 1
    logger.info("Loaded %d SSO tenant config(s) from %s", n, path)
    return n


def parse_tenant_blob(tenant_id: str, blob: Dict[str, Any]) -> SSOTenantConfig:
    """Build an :class:`SSOTenantConfig` from an arbitrary dict, applying
    type coercion for booleans and lists so admin-UI form POSTs round-trip
    cleanly."""
    clean = dict(blob)
    clean["tenant_id"] = tenant_id
    for b in ("enabled", "jit_provisioning"):
        if b in clean and isinstance(clean[b], str):
            clean[b] = clean[b].lower() in {"1", "true", "yes", "on"}
    if "allowed_email_domains" in clean and isinstance(
        clean["allowed_email_domains"], str
    ):
        clean["allowed_email_domains"] = [
            s.strip() for s in clean["allowed_email_domains"].split(",") if s.strip()
        ]
    return SSOTenantConfig(**clean)


__all__ = [
    "DEFAULT_SAML_ATTRIBUTE_MAP",
    "DEFAULT_OIDC_ATTRIBUTE_MAP",
    "SSOTenantConfig",
    "SSOConfigRegistry",
    "default_registry",
    "load_from_env",
    "load_from_file",
    "parse_tenant_blob",
]
