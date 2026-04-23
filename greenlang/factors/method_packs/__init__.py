# -*- coding: utf-8 -*-
"""
GreenLang Factors - Method Pack Library (Phase F2, GAP-8 + GAP-9 closed).

Method packs are the commercial layer of GreenLang Factors.  The same
activity ("12,500 kWh, India, FY2027") resolves to *different* factors
depending on whether the caller is doing corporate inventory, product
carbon, freight, financed-emissions, or land-sector reporting.

Each :class:`MethodPack` wraps:

- factor-selection rules (which factor families apply, which statuses allowed)
- boundary rules (combustion / WTT / WTW / cradle-to-gate / cradle-to-grave)
- inclusion / exclusion logic (biogenic treatment, market instruments)
- gas-to-CO2e conversion basis (AR4 / AR5 / AR6, 100-yr or 20-yr)
- region hierarchy for fallback (facility > utility > country > GLOBAL)
- reporting labels (which framework(s) this satisfies)
- audit text templates (used by the Explain endpoint in Phase F3)
- deprecation policy (how long after a source update we keep an old version)

Quickstart::

    from greenlang.factors.method_packs import get_pack
    from greenlang.data.canonical_v2 import MethodProfile

    # Profile-level lookup (legacy):
    pack = get_pack(MethodProfile.CORPORATE_SCOPE2_LOCATION)

    # Variant-level lookup (GAP-8 / GAP-9):
    pcaf_mortgage = get_pack("pcaf_mortgages")
    lsr_removals = get_pack("lsr_removals")

Public surface re-exported below.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.method_packs.base import (
    BoundaryRule,
    CannotResolveAction,
    DeprecationRule,
    FallbackStep,
    MethodPack,
    SelectionRule,
)
from greenlang.factors.method_packs.exceptions import (
    FactorCannotResolveSafelyError,
)
from greenlang.factors.method_packs.registry import (
    MethodPackNotFound,
    get_pack as _get_pack_by_profile,
    list_packs,
    register_pack,
    registered_profiles,
)

# Import + register all built-in packs on module load.  Each module
# self-registers via ``register_pack(...)`` at import time.
from greenlang.factors.method_packs import (  # noqa: F401
    corporate,
    electricity,
    eu_policy,
    product_carbon,
    product_lca_variants,
    freight,
    land_removals,
    finance_proxy,
)

# Re-export the GAP-8 + GAP-9 variant-level helpers so callers can reach
# them without importing the submodules directly.
from greenlang.factors.method_packs.finance_proxy import (  # noqa: F401
    PCAFAssetClass,
    PCAFDataQualityScore,
    PCAFAttributionMethod,
    PCAFIntensityMode,
    PCAFPackMetadata,
    PCAFListedEquityPack,
    PCAFCorporateBondsPack,
    PCAFBusinessLoansPack,
    PCAFProjectFinancePack,
    PCAFCommercialRealEstatePack,
    PCAFMortgagesPack,
    PCAFMotorVehicleLoansPack,
    PCAF_DQS_RUBRIC,
    PCAF_ATTRIBUTION_HIERARCHY,
    get_pcaf_variant,
    get_pcaf_metadata,
    list_pcaf_variants,
)
from greenlang.factors.method_packs.product_lca_variants import (  # noqa: F401
    PAS_2050,
    PEF,
    OEF,
    get_product_lca_variant,
    list_product_lca_variants,
)
from greenlang.factors.method_packs.land_removals import (  # noqa: F401
    PermanenceClass,
    ReversalRiskLevel,
    BiogenicAccountingTreatment,
    RemovalCategory,
    RemovalType,
    VerificationStandard,
    ReportingFrequency,
    LSRPackMetadata,
    GHGLSRLandUseEmissionsPack,
    GHGLSRLandManagementPack,
    GHGLSRRemovalsPack,
    GHGLSRStoragePack,
    DEFAULT_BUFFER_POOL,
    RISK_BUFFER_MULTIPLIER,
    LSR_FALLBACK_HIERARCHY,
    get_lsr_variant,
    get_lsr_metadata,
    list_lsr_variants,
    compute_buffer_pool_pct,
)


def get_pack(key: Union[MethodProfile, str]) -> MethodPack:
    """Retrieve a registered method pack.

    Accepts either:
      * a :class:`MethodProfile` enum value (legacy behaviour — returns
        the umbrella pack registered under that profile), OR
      * a string variant name (GAP-8 / GAP-9 — returns the specific
        PCAF or LSR variant, e.g. ``"pcaf_listed_equity"`` or
        ``"lsr_removals"``).

    Raises :class:`MethodPackNotFound` if the key is unknown.
    """
    if isinstance(key, MethodProfile):
        return _get_pack_by_profile(key)

    if isinstance(key, str):
        # Try PCAF named variants.
        try:
            return get_pcaf_variant(key)
        except KeyError:
            pass
        # Try LSR named variants.
        try:
            return get_lsr_variant(key)
        except KeyError:
            pass
        # Try product-LCA variants (PAS 2050, PEF, OEF).
        try:
            return get_product_lca_variant(key)
        except KeyError:
            pass
        # Try as MethodProfile enum value string.
        try:
            return _get_pack_by_profile(MethodProfile(key))
        except (ValueError, MethodPackNotFound):
            pass

        raise MethodPackNotFound(
            "no method pack registered for key %r; "
            "try a MethodProfile or a registered variant name" % key
        )

    raise TypeError(
        "get_pack() expected MethodProfile or str, got %s" % type(key).__name__
    )


# ---------------------------------------------------------------------------
# Audit-text renderer (MP3/MP4/MP5 wave 1)
# ---------------------------------------------------------------------------
# Each pack ships a Jinja template under ``audit_texts/<pack_id>.j2`` that
# describes the factor selection rationale for ``/explain``. The templates
# have a YAML-ish frontmatter block that captures methodology-lead approval
# state:
#
#     {# ---
#     approved: false
#     approved_by: null
#     approved_at: null
#     methodology_lead: null
#     --- #}
#
# When ``approved: false`` the renderer prepends a draft banner so every
# consumer (UI, SDK, auditor export) is aware that the wording has not
# passed methodology review yet. This lets us ship template bodies TODAY
# without waiting for approval. Methodology lead flips the frontmatter to
# ``approved: true`` in a separate PR, banner disappears.

#: Banner prepended to every audit text produced by a draft (unapproved) template.
AUDIT_DRAFT_BANNER: str = (
    "[Draft — Methodology Review Required — "
    "do not rely on for regulatory filing]"
)

#: Resolution directory for the built-in audit-text templates.
_AUDIT_TEMPLATE_DIR: Path = Path(__file__).resolve().parent / "audit_texts"

#: Compiled frontmatter matcher. Matches the opening ``{# ---`` marker,
#: captures the YAML-ish body up to the trailing ``--- #}`` marker.
_FRONTMATTER_RE = re.compile(
    r"^\s*\{#\s*---\s*(?P<fm>.*?)\s*---\s*#\}\s*",
    re.DOTALL,
)


def _load_template_source(pack_id: str) -> str:
    """Return the raw Jinja source for ``pack_id``.

    Raises :class:`FileNotFoundError` when no ``<pack_id>.j2`` file exists.
    Callers should catch and either fall back to the inline
    ``MethodPack.audit_text_template`` string or re-raise.
    """
    template_path = _AUDIT_TEMPLATE_DIR / f"{pack_id}.j2"
    return template_path.read_text(encoding="utf-8")


def parse_frontmatter(template_source: str) -> Tuple[Dict[str, Any], str]:
    """Split a template into its frontmatter dict + body string.

    Recognised frontmatter keys (string values or YAML ``null``):

    * ``approved`` — ``true`` / ``false`` / ``null``
    * ``approved_by`` — email / handle
    * ``approved_at`` — ISO date
    * ``methodology_lead`` — email / handle

    When no frontmatter is found both values default sensibly (empty dict
    + unmodified source). Any parse issue downgrades to ``approved=False``
    so an unreadable template NEVER bypasses the draft banner.
    """
    match = _FRONTMATTER_RE.match(template_source)
    if match is None:
        return {}, template_source
    fm_body = match.group("fm")
    body = template_source[match.end():]
    data: Dict[str, Any] = {}
    for raw_line in fm_body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, raw_value = line.partition(":")
        key = key.strip()
        value_str = raw_value.strip()
        value: Any
        if value_str.lower() in ("true", "yes"):
            value = True
        elif value_str.lower() in ("false", "no"):
            value = False
        elif value_str.lower() in ("null", "none", "~", ""):
            value = None
        else:
            # Strip surrounding quotes if present.
            if len(value_str) >= 2 and value_str[0] == value_str[-1] and value_str[0] in ('"', "'"):
                value_str = value_str[1:-1]
            value = value_str
        data[key] = value
    return data, body


def _render_jinja(template_body: str, *, factor: Any, extra_context: Optional[Dict[str, Any]] = None) -> str:
    """Render ``template_body`` against ``factor``.

    When Jinja2 is installed we use it; otherwise we fall back to a
    minimal ``{{ dotted.path }}`` substitution that covers the templates
    we ship. The fallback is deterministic and deliberately narrow so
    tests do not need Jinja2 at collection time.
    """
    context: Dict[str, Any] = {"factor": factor}
    if extra_context:
        context.update(extra_context)
    # Expose commonly-referenced shortcuts so templates can say
    # ``{{ factor.source.authority }}`` etc. without callers having to
    # guess the exact attribute path.
    try:
        import jinja2  # type: ignore

        env = jinja2.Environment(
            autoescape=False,
            keep_trailing_newline=True,
            undefined=jinja2.ChainableUndefined,
        )
        return env.from_string(template_body).render(**context)
    except Exception:
        return _fallback_substitute(template_body, context)


def _fallback_substitute(body: str, context: Dict[str, Any]) -> str:
    """Fallback `{{ dotted.path }}` renderer when Jinja2 is absent."""
    pattern = re.compile(r"\{\{\s*([\w\.]+)\s*\}\}")

    def _lookup(dotted: str) -> Any:
        current: Any = context
        for part in dotted.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
            if current is None:
                return ""
        return current

    def _replace(match: "re.Match[str]") -> str:
        value = _lookup(match.group(1))
        return "" if value is None else str(value)

    return pattern.sub(_replace, body)


def render_audit_text(
    pack_id: str,
    factor: Any,
    *,
    extra_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the ``/explain`` audit text for ``pack_id``.

    Applies the SAFE-DRAFT policy: any template whose frontmatter is
    missing or has ``approved: false`` (the default until methodology
    review signs off) is prefixed with
    :data:`AUDIT_DRAFT_BANNER` so regulators / auditors / tenants see a
    clear warning that the wording is not yet approved for filing.

    Parameters
    ----------
    pack_id:
        Pack identifier (e.g. ``"corporate"``, ``"electricity"``,
        ``"eu_policy"``).
    factor:
        A :class:`ResolvedFactor` (or any object with ``chosen_factor``,
        ``source``, ``method_pack``, ``fallback_rank`` attributes). The
        template may also access the dotted attributes directly.
    extra_context:
        Optional mapping of additional placeholders (e.g. ``pack_name``,
        ``geography``, ``scope3_category``) merged into the Jinja context.

    Returns
    -------
    str
        Rendered audit text. When the template is unapproved the string
        starts with the draft banner separated by a blank line.
    """
    source = _load_template_source(pack_id)
    frontmatter, body = parse_frontmatter(source)
    rendered = _render_jinja(body, factor=factor, extra_context=extra_context)
    if not frontmatter.get("approved", False):
        return f"{AUDIT_DRAFT_BANNER}\n\n{rendered}"
    return rendered


def load_template(pack_id: str) -> str:
    """Public wrapper around :func:`_load_template_source`.

    Exposed so tests (and the MP-wave fixture harness) can introspect
    the raw template without going through :func:`render_audit_text`.
    """
    return _load_template_source(pack_id)


__all__ = [
    # Base types
    "BoundaryRule",
    "CannotResolveAction",
    "DeprecationRule",
    "FactorCannotResolveSafelyError",
    "FallbackStep",
    "MethodPack",
    "MethodPackNotFound",
    "SelectionRule",
    # Audit-text renderer (MP3/MP4/MP5 wave 1)
    "AUDIT_DRAFT_BANNER",
    "load_template",
    "parse_frontmatter",
    "render_audit_text",
    # Registry primitives
    "get_pack",
    "list_packs",
    "register_pack",
    "registered_profiles",
    # PCAF (GAP-8)
    "PCAFAssetClass",
    "PCAFDataQualityScore",
    "PCAFAttributionMethod",
    "PCAFIntensityMode",
    "PCAFPackMetadata",
    "PCAFListedEquityPack",
    "PCAFCorporateBondsPack",
    "PCAFBusinessLoansPack",
    "PCAFProjectFinancePack",
    "PCAFCommercialRealEstatePack",
    "PCAFMortgagesPack",
    "PCAFMotorVehicleLoansPack",
    "PCAF_DQS_RUBRIC",
    "PCAF_ATTRIBUTION_HIERARCHY",
    "get_pcaf_variant",
    "get_pcaf_metadata",
    "list_pcaf_variants",
    # LSR (GAP-9)
    "PermanenceClass",
    "ReversalRiskLevel",
    "BiogenicAccountingTreatment",
    "RemovalCategory",
    "RemovalType",
    "VerificationStandard",
    "ReportingFrequency",
    "LSRPackMetadata",
    "GHGLSRLandUseEmissionsPack",
    "GHGLSRLandManagementPack",
    "GHGLSRRemovalsPack",
    "GHGLSRStoragePack",
    "DEFAULT_BUFFER_POOL",
    "RISK_BUFFER_MULTIPLIER",
    "LSR_FALLBACK_HIERARCHY",
    "get_lsr_variant",
    "get_lsr_metadata",
    "list_lsr_variants",
    "compute_buffer_pool_pct",
    # Product-LCA variants (PAS 2050, PEF, OEF)
    "PAS_2050",
    "PEF",
    "OEF",
    "get_product_lca_variant",
    "list_product_lca_variants",
]
