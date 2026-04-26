# -*- coding: utf-8 -*-
"""SourceRightsService — the Phase 1 enforcement core.

Decisions are derived from the source registry
(`greenlang/factors/data/source_registry.yaml`) plus an
EntitlementStore (`config/entitlements/alpha_v0_1.yaml`). Every
gate returns a :class:`Decision` (allow / deny + reason) so the
caller can either filter quietly (route layer) or raise
:class:`IngestionBlocked` / :class:`RightsDenied` (publish layer).
"""
from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple as _Tuple

from .entitlements import EntitlementStore, load_alpha_entitlements
from .errors import IngestionBlocked, RightsDenied

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class Outcome(str, enum.Enum):
    ALLOW = "allow"
    DENY = "deny"
    METADATA_ONLY = "metadata_only"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    reason: str
    licence_class: Optional[str] = None
    redistribution_class: Optional[str] = None
    entitlement_model: Optional[str] = None

    @property
    def allowed(self) -> bool:
        return self.outcome == Outcome.ALLOW

    @property
    def metadata_only(self) -> bool:
        return self.outcome == Outcome.METADATA_ONLY

    @property
    def denied(self) -> bool:
        return self.outcome == Outcome.DENY


# Sources whose ``legal_signoff.status`` is not in this set may not be
# served / ingested in production.
_INGESTION_LEGAL_OK = frozenset({"approved"})

# Licence classes that always block both ingestion and serving.
_BLOCKED_LICENCE_CLASSES = frozenset({"blocked"})


# ---------------------------------------------------------------------------
# SourceRightsService
# ---------------------------------------------------------------------------


@dataclass
class SourceRightsService:
    """Phase 1 enforcement core.

    Construct via :func:`default_service` to use the default registry
    + alpha entitlements file. Tests construct directly with custom
    `registry_index` / `entitlements` for isolation.
    """

    registry_index: Dict[str, Dict[str, Any]]  # source_urn -> source dict
    entitlements: EntitlementStore = field(default_factory=EntitlementStore)
    release_profile: str = "alpha-v0.1"

    # ---------------- Lookup ----------------

    def get_source(self, source_urn: str) -> Optional[Dict[str, Any]]:
        return self.registry_index.get(source_urn)

    def licence_class_for(self, source_urn: str) -> Optional[str]:
        s = self.get_source(source_urn)
        return s.get("licence_class") if s else None

    def redistribution_class_for(self, source_urn: str) -> Optional[str]:
        s = self.get_source(source_urn)
        return s.get("redistribution_class") if s else None

    def entitlement_model_for(self, source_urn: str) -> Optional[str]:
        s = self.get_source(source_urn)
        if not s:
            return None
        rules = s.get("entitlement_rules") or {}
        return rules.get("model")

    def release_milestone_for(self, source_urn: str) -> Optional[str]:
        s = self.get_source(source_urn)
        return s.get("release_milestone") if s else None

    def legal_signoff_status_for(self, source_urn: str) -> Optional[str]:
        s = self.get_source(source_urn)
        if not s:
            return None
        ls = s.get("legal_signoff") or {}
        return ls.get("status")

    # ---------------- Ingestion gate ----------------

    def check_ingestion_allowed(
        self, source_urn: str, *, strict_unknown: bool = False
    ) -> Decision:
        """Decide whether a record from ``source_urn`` may be published.

        Reasons for denial: licence_class is `blocked`; legal_signoff
        status is not `approved`; release milestone is later than the
        current release_profile.

        Unknown sources fail OPEN by default (the provenance gate is
        the canonical "is this a registered source" check). Pass
        ``strict_unknown=True`` to flip a missing source to DENY (used
        by the dedicated regression test).
        """
        s = self.get_source(source_urn)
        if not s:
            if strict_unknown:
                return Decision(
                    Outcome.DENY,
                    f"source {source_urn!r} not in registry",
                )
            return Decision(
                Outcome.ALLOW,
                f"source {source_urn!r} not in rights registry; provenance gate is the source-of-record",
            )
        lc = s.get("licence_class")
        if lc in _BLOCKED_LICENCE_CLASSES:
            return Decision(
                Outcome.DENY,
                f"source {source_urn!r} is blocked",
                licence_class=lc,
            )
        ls = (s.get("legal_signoff") or {}).get("status")
        if ls not in _INGESTION_LEGAL_OK:
            return Decision(
                Outcome.DENY,
                f"source {source_urn!r} legal_signoff.status={ls!r} (not approved)",
                licence_class=lc,
            )
        rm = s.get("release_milestone")
        if rm and not self._milestone_in_profile(rm):
            return Decision(
                Outcome.DENY,
                f"source {source_urn!r} release_milestone={rm!r} not yet in profile {self.release_profile!r}",
                licence_class=lc,
            )
        return Decision(
            Outcome.ALLOW,
            "ingestion allowed",
            licence_class=lc,
            redistribution_class=s.get("redistribution_class"),
            entitlement_model=(s.get("entitlement_rules") or {}).get("model"),
        )

    def assert_ingestion_allowed(self, source_urn: str) -> None:
        """Like :meth:`check_ingestion_allowed` but raises on deny."""
        d = self.check_ingestion_allowed(source_urn)
        if d.denied:
            raise IngestionBlocked(d.reason)

    # ---------------- Query gate ----------------

    def check_factor_read_allowed(
        self,
        tenant_id: Optional[str],
        source_urn: str,
        *,
        action: str = "read",
    ) -> Decision:
        """Decide whether ``tenant_id`` may receive a factor's value from
        ``source_urn``.

        ``community_open`` sources allow everyone unconditionally.
        ``connector_only`` sources allow metadata only by default
        (``Outcome.METADATA_ONLY``); a tenant-side connector can be
        granted full read via an entitlement record.
        ``commercial_licensed`` requires an active SOURCE_ACCESS or
        PACK_ACCESS entitlement.
        ``private_tenant_scoped`` requires PRIVATE_OWNER entitlement.
        ``blocked`` denies everyone.
        """
        s = self.get_source(source_urn)
        if not s:
            # Unknown source falls open. The provenance / schema gate
            # is the canonical "is this a registered source" check;
            # the rights gate only enforces on KNOWN sources that are
            # blocked / pending / wrong-milestone. Tests that
            # explicitly want a hard deny on unknown sources should
            # check the registry separately.
            return Decision(
                Outcome.ALLOW,
                f"source {source_urn!r} not in rights registry; provenance gate is the source-of-record",
            )
        lc = s.get("licence_class")
        rc = s.get("redistribution_class")
        em = (s.get("entitlement_rules") or {}).get("model")
        ls = (s.get("legal_signoff") or {}).get("status")
        if ls != "approved":
            return Decision(
                Outcome.DENY,
                f"source {source_urn!r} legal_signoff.status={ls!r} (not approved)",
                licence_class=lc, redistribution_class=rc, entitlement_model=em,
            )

        if lc == "blocked":
            return Decision(
                Outcome.DENY,
                "source is blocked",
                licence_class=lc, redistribution_class=rc, entitlement_model=em,
            )

        if lc == "community_open":
            return Decision(
                Outcome.ALLOW, "community_open: public access",
                licence_class=lc, redistribution_class=rc, entitlement_model=em,
            )

        if lc == "method_only":
            # method_only sources publish framework / method text + may
            # publish a small set of derived reference values. Treat as
            # public read at v0.1 alpha; v0.5+ revisits per
            # `tenant_entitled_only` redistribution_class.
            if rc == "tenant_entitled_only":
                if tenant_id and self.entitlements.has_active_source_access(
                    tenant_id, source_urn
                ):
                    return Decision(
                        Outcome.ALLOW, "method_only with active entitlement",
                        licence_class=lc, redistribution_class=rc, entitlement_model=em,
                    )
                return Decision(
                    Outcome.DENY,
                    "method_only source requires active entitlement",
                    licence_class=lc, redistribution_class=rc, entitlement_model=em,
                )
            return Decision(
                Outcome.ALLOW, "method_only: public method reference",
                licence_class=lc, redistribution_class=rc, entitlement_model=em,
            )

        if lc == "commercial_licensed":
            if not tenant_id:
                return Decision(
                    Outcome.DENY,
                    "commercial_licensed requires authenticated tenant",
                    licence_class=lc, redistribution_class=rc, entitlement_model=em,
                )
            if self.entitlements.has_active_source_access(tenant_id, source_urn):
                return Decision(
                    Outcome.ALLOW, "active entitlement",
                    licence_class=lc, redistribution_class=rc, entitlement_model=em,
                )
            return Decision(
                Outcome.DENY,
                "commercial_licensed: tenant has no active entitlement",
                licence_class=lc, redistribution_class=rc, entitlement_model=em,
            )

        if lc == "private_tenant_scoped":
            if not tenant_id:
                return Decision(
                    Outcome.DENY,
                    "private_tenant_scoped requires authenticated tenant",
                    licence_class=lc, redistribution_class=rc, entitlement_model=em,
                )
            if self.entitlements.is_private_owner(tenant_id, source_urn):
                return Decision(
                    Outcome.ALLOW, "tenant is the private owner",
                    licence_class=lc, redistribution_class=rc, entitlement_model=em,
                )
            return Decision(
                Outcome.DENY,
                "private_tenant_scoped: tenant is not the owner",
                licence_class=lc, redistribution_class=rc, entitlement_model=em,
            )

        if lc == "connector_only":
            # connector_only sources expose metadata to everyone but
            # values only via the connector path (out of scope for
            # bulk REST). Bulk read returns METADATA_ONLY by default;
            # if the tenant has SOURCE_ACCESS we treat it as full read.
            if tenant_id and self.entitlements.has_active_source_access(
                tenant_id, source_urn
            ):
                return Decision(
                    Outcome.ALLOW, "connector_only with active entitlement",
                    licence_class=lc, redistribution_class=rc, entitlement_model=em,
                )
            return Decision(
                Outcome.METADATA_ONLY,
                "connector_only: metadata only (use connector for values)",
                licence_class=lc, redistribution_class=rc, entitlement_model=em,
            )

        return Decision(
            Outcome.DENY,
            f"unknown licence_class {lc!r}",
            licence_class=lc, redistribution_class=rc, entitlement_model=em,
        )

    def assert_factor_read_allowed(
        self, tenant_id: Optional[str], source_urn: str
    ) -> None:
        d = self.check_factor_read_allowed(tenant_id, source_urn)
        if d.denied:
            raise RightsDenied(d.reason)

    # ---------------- Pack download gate ----------------

    def check_pack_download_allowed(
        self,
        tenant_id: Optional[str],
        pack_urn: str,
        source_urn: str,
    ) -> Decision:
        """Decide whether ``tenant_id`` may download a pack archive.

        Pack download is the bulk-redistribution path; it inherits the
        source's redistribution_class. ``redistribution_allowed`` and
        ``attribution_required`` are downloadable; everything else
        requires an entitlement OR is denied.
        """
        s = self.get_source(source_urn)
        if not s:
            # Unknown source falls open for pack download too — same
            # rationale as the read path. Strict callers can verify
            # source registration separately via SourceRegistry.
            return Decision(
                Outcome.ALLOW,
                f"source {source_urn!r} not in rights registry (pack {pack_urn!r})",
            )
        lc = s.get("licence_class")
        rc = s.get("redistribution_class")
        em = (s.get("entitlement_rules") or {}).get("model")
        ls = (s.get("legal_signoff") or {}).get("status")
        if ls != "approved":
            return Decision(
                Outcome.DENY,
                f"source {source_urn!r} legal_signoff.status={ls!r}",
                licence_class=lc, redistribution_class=rc, entitlement_model=em,
            )
        if lc == "blocked":
            return Decision(Outcome.DENY, "blocked source",
                            licence_class=lc, redistribution_class=rc, entitlement_model=em)
        if rc in ("redistribution_allowed", "attribution_required"):
            return Decision(
                Outcome.ALLOW, "pack download permitted",
                licence_class=lc, redistribution_class=rc, entitlement_model=em,
            )
        if rc == "metadata_only":
            return Decision(
                Outcome.METADATA_ONLY,
                "pack values not redistributable; metadata only",
                licence_class=lc, redistribution_class=rc, entitlement_model=em,
            )
        if rc == "tenant_entitled_only":
            if tenant_id and (
                self.entitlements.has_active_source_access(tenant_id, source_urn)
                or self.entitlements.has_active_pack_access(tenant_id, pack_urn)
            ):
                return Decision(
                    Outcome.ALLOW, "active entitlement",
                    licence_class=lc, redistribution_class=rc, entitlement_model=em,
                )
            return Decision(
                Outcome.DENY,
                "pack download requires active entitlement",
                licence_class=lc, redistribution_class=rc, entitlement_model=em,
            )
        if rc == "no_redistribution":
            return Decision(
                Outcome.DENY, "no_redistribution",
                licence_class=lc, redistribution_class=rc, entitlement_model=em,
            )
        if rc == "blocked":
            return Decision(
                Outcome.DENY, "redistribution_class=blocked",
                licence_class=lc, redistribution_class=rc, entitlement_model=em,
            )
        return Decision(
            Outcome.DENY,
            f"unknown redistribution_class {rc!r}",
            licence_class=lc, redistribution_class=rc, entitlement_model=em,
        )

    # ---------------- Licence-tag mismatch ----------------

    def check_record_licence_matches_registry(
        self, source_urn: str, record_licence: Optional[str]
    ) -> Decision:
        """Verify a published record's ``licence`` field matches the registry.

        Semantic mirrors the other gates: unknown sources fall OPEN
        (the provenance gate is the canonical "is this a registered
        source" check). Known sources without a pinned ``licence`` in
        the registry also fall open. Strict matching only fires when:
        (a) the source IS in the registry; AND (b) the registry pins
        a specific ``licence`` value; AND (c) the record carries a
        DIFFERENT ``licence`` value. A record without any ``licence``
        field at all falls open here too — the schema/provenance gate
        is responsible for asserting the record HAS a licence; this
        gate only catches MISMATCH.
        """
        s = self.get_source(source_urn)
        if not s:
            return Decision(
                Outcome.ALLOW,
                f"source {source_urn!r} not in rights registry; provenance gate is the source-of-record",
            )
        registry_licence = s.get("licence")
        if not registry_licence:
            return Decision(
                Outcome.ALLOW,
                "registry does not pin a licence; nothing to mismatch",
                licence_class=s.get("licence_class"),
            )
        if not record_licence:
            return Decision(
                Outcome.ALLOW,
                "record carries no `licence` field; provenance gate enforces presence",
                licence_class=s.get("licence_class"),
            )
        if record_licence != registry_licence:
            return Decision(
                Outcome.DENY,
                f"record licence {record_licence!r} does not match registry pin {registry_licence!r}",
                licence_class=s.get("licence_class"),
            )
        return Decision(
            Outcome.ALLOW, "licence tag matches",
            licence_class=s.get("licence_class"),
        )

    # ---------------- Helpers ----------------

    _PROFILE_MILESTONE_FLOOR: Dict[str, _Tuple[str, ...]] = field(
        default_factory=lambda: {
            "alpha-v0.1": ("v0.1",),
            "beta-v0.5": ("v0.1", "v0.5"),
            "rc-v0.9": ("v0.1", "v0.5", "v0.9"),
            "ga-v1.0": ("v0.1", "v0.5", "v0.9", "v1.0"),
            "dev": ("v0.1", "v0.5", "v0.9", "v1.0", "v1.5", "v2.0", "v2.5", "v3.0"),
        }
    )

    def _milestone_in_profile(self, milestone: str) -> bool:
        # ``release_profile`` drives the floor of milestones whose sources
        # are eligible to ingest in production. Unknown profiles fall back
        # to alpha (most restrictive).
        allowed = self._PROFILE_MILESTONE_FLOOR.get(
            self.release_profile, ("v0.1",)
        )
        return milestone in allowed


# ---------------------------------------------------------------------------
# Default factory + module-level convenience wrappers
# ---------------------------------------------------------------------------


_DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "source_registry.yaml"
)


def _load_registry_index(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    p = path or _DEFAULT_REGISTRY_PATH
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover
        raise RuntimeError("PyYAML required to load source_registry.yaml")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    sources = raw.get("sources") or []
    out: Dict[str, Dict[str, Any]] = {}
    for s in sources:
        if not isinstance(s, dict):
            continue
        urn = s.get("urn")
        if isinstance(urn, str) and urn:
            out[urn] = s
    return out


_DEFAULT_LOCK = threading.Lock()
_DEFAULT_INSTANCE: Optional[SourceRightsService] = None


def default_service(
    *,
    release_profile: Optional[str] = None,
    refresh: bool = False,
) -> SourceRightsService:
    """Lazy-built process-wide SourceRightsService.

    Tests should construct their own SourceRightsService directly to
    avoid sharing state.
    """
    global _DEFAULT_INSTANCE
    with _DEFAULT_LOCK:
        if _DEFAULT_INSTANCE is None or refresh:
            try:
                from greenlang.factors.release_profile import current_profile
                profile = release_profile or current_profile().value
            except Exception:
                profile = release_profile or "alpha-v0.1"
            _DEFAULT_INSTANCE = SourceRightsService(
                registry_index=_load_registry_index(),
                entitlements=load_alpha_entitlements(),
                release_profile=profile,
            )
        return _DEFAULT_INSTANCE


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (used by call sites that do NOT want
# to pass a service instance around).
# ---------------------------------------------------------------------------


def check_ingestion_allowed(source_urn: str) -> Decision:
    return default_service().check_ingestion_allowed(source_urn)


def check_factor_read_allowed(
    tenant_id: Optional[str], source_urn: str, *, action: str = "read"
) -> Decision:
    return default_service().check_factor_read_allowed(
        tenant_id, source_urn, action=action
    )


def check_pack_download_allowed(
    tenant_id: Optional[str], pack_urn: str, source_urn: str
) -> Decision:
    return default_service().check_pack_download_allowed(
        tenant_id, pack_urn, source_urn
    )
