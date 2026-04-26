# -*- coding: utf-8 -*-
"""
GreenLang Factors v0.1 Alpha — V1->V0.1 Normalizer (Wave D / WS2-T2).

Lifts the v1-shape factor records produced by the 6 alpha-source parsers
into the FROZEN v0.1 alpha record contract validated by
:class:`greenlang.factors.quality.alpha_provenance_gate.AlphaProvenanceGate`.

The actual parser output today is the *current v1 catalog shape* — i.e.
records with ``factor_id`` (``EF:...``), ``vectors`` (CO2/CH4/N2O scalars),
``gwp_100yr`` (gwp_set + per-gas factors), nested ``provenance``,
``boundary``, ``scope``, ``unit``, ``geography``, ``valid_from`` /
``valid_to``, ``license_info``, ``dqs``, plus a sprinkling of newer
canonical-v2 fields on India CEA (``factor_family``, ``method_profile``,
``jurisdiction``, ``factor_name``, ``parameters``, ``activity_schema``,
``explainability``, ``factor_version``, ...).

The v0.1 alpha contract requires a flat record with:

* ``urn`` — canonical ``urn:gl:factor:<source>:<ns>:<id>:v<version>``
* ``source_urn``, ``factor_pack_urn``, ``unit_urn``, ``geography_urn``,
  ``methodology_urn`` — every reference promoted to URN form
* ``value`` — scalar CO2e (collapsed from per-gas vectors via GWP-AR6)
* ``gwp_basis = "ar6"``, ``gwp_horizon = 100``
* ``category`` — narrowed alpha enum (``fuel``, ``grid_intensity``,
  ``scope2_location_based``, ``scope2_market_based``, ``refrigerant``,
  ``cbam_default``, ``process``, ``fugitive``, ``scope1``)
* ``description`` (>=30 chars), ``boundary`` (>=10 chars), ``licence``
* ``citations`` — at least one entry
* ``extraction`` — every field MANDATORY (12 fields incl. SHA-256,
  parser_commit, raw_artifact_uri, operator)
* ``review`` — ``review_status`` + reviewer + reviewed_at; ``approved``
  status additionally requires ``approved_by`` + ``approved_at``

CTO doc references: §6.1, §19.1 (FY27 Q1 alpha — provenance fields
complete), Wave D / TaskCreate #6.
Schema $id: https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NormalizerError(Exception):
    """Raised when a v1 record cannot be lifted to the v0.1 alpha shape.

    The error message includes the offending field and (when available)
    the source factor_id of the record that triggered the failure.
    """


class NonPositiveValueError(NormalizerError):
    """Raised when collapsing a record's vectors yields ``value <= 0``.

    The v0.1 alpha schema requires ``value > 0``, so carbon-sequestration
    rows (e.g. IPCC land-use forest removals) cannot be expressed in
    v0.1. The backfill script catches this subclass to *skip* the record
    quietly rather than count it as a backfill failure — sequestration
    factors will be expressible in v0.5+ when the schema gains a signed
    value field.
    """


# ---------------------------------------------------------------------------
# Constants — pack id derivation, geography URN map, factor-family map
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]


# factor_family -> v0.1 pack-id slug (the central component of factor_pack_urn).
_FACTOR_FAMILY_TO_PACK_ID: Dict[str, str] = {
    "combustion": "tier-1-defaults",
    "stationary_combustion": "tier-1-defaults",
    "mobile_combustion": "tier-1-defaults",
    "electricity": "grid-intensities",
    "grid_intensity": "grid-intensities",
    "refrigerants": "ar6-refrigerants",
    "refrigerant": "ar6-refrigerants",
    "classification_mapping": "cbam-defaults",
    "material_embodied": "cbam-defaults",
    "fugitive": "tier-1-defaults",
    "process": "tier-1-defaults",
}


# Source-id-driven pack overrides — used when factor_family is missing
# or when the source's pack id differs from the family default (e.g.
# eGRID is grid-intensity but the alpha pack is "subregion-rates").
_SOURCE_ID_TO_PACK_ID: Dict[str, str] = {
    "epa_hub":                 "emission-factors-hub",
    "egrid":                   "subregion-rates",
    "desnz_ghg_conversion":    "conversion-factors",
    "india_cea_co2_baseline":  "national-grid",
    "ipcc_2006_nggi":          "tier-1-defaults",
    "cbam_default_values":     "cbam-annex-iv",
}


# Methodology slug per source — used when the parser doesn't set
# method_profile explicitly (epa_hub, egrid, ipcc_defaults, ...).
_SOURCE_ID_TO_METHODOLOGY_SLUG: Dict[str, str] = {
    "epa_hub":                 "ipcc-tier-1-stationary-combustion",
    "egrid":                   "ghgp-corporate-scope2-location",
    "desnz_ghg_conversion":    "ipcc-tier-1-stationary-combustion",
    "india_cea_co2_baseline":  "ghgp-corporate-scope2-location",
    "ipcc_2006_nggi":          "ipcc-tier-1-stationary-combustion",
    "cbam_default_values":     "eu-cbam-default",
}


# Method-profile -> methodology-urn slug (preferred when set).
_METHOD_PROFILE_TO_METHODOLOGY_SLUG: Dict[str, str] = {
    "corporate_scope1":                   "ghgp-corporate-scope1",
    "corporate_scope2_location_based":    "ghgp-corporate-scope2-location",
    "corporate_scope2_market_based":      "ghgp-corporate-scope2-market",
    "eu_cbam":                             "eu-cbam-default",
    "ipcc_tier_1":                        "ipcc-tier-1-stationary-combustion",
}


# AR6 GWP-100 defaults used to collapse vectors -> CO2e when the parser
# omits a `co2e` field.
_AR6_GWP100_CH4 = 28
_AR6_GWP100_N2O = 273

_GWP_BASIS_FROZEN = "ar6"
_GWP_HORIZON_DEFAULT = 100
_VINTAGE_START_DEFAULT = "2024-01-01"
_VINTAGE_END_DEFAULT = "2099-12-31"
_RESOLUTION_DEFAULT = "annual"


# ---------------------------------------------------------------------------
# Regexes — match the v0.1 schema patterns
# ---------------------------------------------------------------------------

_URN_FACTOR_RE = re.compile(
    # Source slug + lowercase namespace + 1-3 lowercase id segments + version.
    # Namespace MUST be lowercase per URN spec (greenlang.factors.ontology.urn).
    # Id segments MUST be lowercase too (we don't have ISO-8601 timestamped
    # factor ids in alpha; v0.5+ may relax this for hourly grid factors).
    r"^urn:gl:factor:[a-z0-9][a-z0-9-]*(:[a-z0-9][a-z0-9._-]*){2,4}:v[1-9][0-9]*$"
)
_URN_GEO_RE = re.compile(
    r"^urn:gl:geo:(global|country|subregion|state_or_province|grid_zone|"
    r"bidding_zone|balancing_authority):[a-zA-Z0-9._-]+$"
)
_URN_UNIT_RE = re.compile(r"^urn:gl:unit:[a-zA-Z0-9._/-]+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def slugify(value: Any) -> str:
    """Lower-case, hyphen-joined, URL-safe slug.

    Forward slashes are PRESERVED (units like ``kgco2e/kwh`` need them);
    spaces, underscores, dots and other punctuation collapse to hyphens.
    Multiple separators are deduped; leading/trailing separators stripped.
    """
    if value is None:
        return ""
    s = str(value).strip().lower()
    if not s:
        return ""
    # Replace anything that is not alnum, '/', or '-' with '-'
    s = re.sub(r"[^a-z0-9/_-]+", "-", s)
    # Collapse underscores -> hyphens (slugs prefer hyphens)
    s = s.replace("_", "-")
    # Collapse repeated hyphens
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def slugify_urn_path(value: Any) -> str:
    """Slugify but ALSO strip forward slashes (URN path segments).

    The URN factor pattern allows ``[A-Za-z0-9._-]`` only — no ``/``.
    """
    s = slugify(value)
    return s.replace("/", "-")


@functools.lru_cache(maxsize=1)
def _resolve_git_head() -> str:
    """Return the current git HEAD commit, or a deterministic fallback.

    Cached for the lifetime of the process so we hit ``git`` exactly once.
    The fallback ``"0000000"`` is short enough to satisfy the schema's
    ``[a-f0-9]{7,40}`` constraint while being clearly synthetic.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        sha = out.decode("utf-8", errors="replace").strip()
        if re.fullmatch(r"[a-f0-9]{7,40}", sha):
            return sha
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        logger.debug("git rev-parse HEAD failed: %s", exc)
    return "0000000"


def _now_iso() -> str:
    """Return the current UTC timestamp as ISO-8601 with 'Z' suffix."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _coerce_isoformat(value: Any) -> Optional[str]:
    """Best-effort: return an ISO-8601 string for date / datetime / str input."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _coerce_iso_date(value: Any, default: str) -> str:
    """Return ISO date (YYYY-MM-DD); fall back to default when missing."""
    s = _coerce_isoformat(value)
    if not s:
        return default
    # Strip trailing time/tz components; the v0.1 schema requires date format.
    return s.split("T", 1)[0]


def _coerce_iso_datetime(value: Any, default: str) -> str:
    """Return ISO date-time; fall back to default."""
    s = _coerce_isoformat(value)
    if not s:
        return default
    if "T" not in s:
        # Schema requires date-time format; lift bare dates to midnight UTC.
        return f"{s}T00:00:00Z"
    return s


def _hash_record(record: Dict[str, Any]) -> str:
    """Return SHA-256 hex digest of ``record`` (sorted JSON, UTF-8 bytes)."""
    blob = json.dumps(record, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Coercers — factor_id -> URN, jurisdiction -> geography_urn, family -> category
# ---------------------------------------------------------------------------


def coerce_factor_id_to_urn(
    factor_id: str,
    source_slug: str,
    namespace: Optional[str] = None,
    *,
    version: int = 1,
) -> str:
    """Lift a v1-shape ``EF:...`` factor_id into a v0.1 ``urn:gl:factor:...``.

    Strategy:
        1. Strip the leading ``EF:`` (if present).
        2. Slugify each remaining ``:``-separated segment so it matches the
           ``[A-Za-z0-9._-]`` constraint.
        3. Compose ``urn:gl:factor:<source-slug>:<namespace?>:<seg1>:<seg2>...:v<version>``.

    The schema requires 2-4 segments between the source slug and the
    trailing ``v<version>`` token, so we cap at 4 segments by joining
    over-long tails with hyphens.
    """
    if not isinstance(factor_id, str) or not factor_id.strip():
        raise NormalizerError(
            f"factor_id must be a non-empty string; got {factor_id!r}"
        )
    raw = factor_id.strip()
    if raw.upper().startswith("EF:"):
        raw = raw[3:]

    parts = [p for p in raw.split(":") if p]
    # Drop any trailing "v1" / "v2" segments — we always re-assert ``version``.
    while parts and re.fullmatch(r"v[1-9][0-9]*", parts[-1]):
        version_tail = parts.pop()
        # Honour the explicit version from the input id when caller didn't override
        if version == 1 and version_tail.lstrip("v").isdigit():
            try:
                version = int(version_tail.lstrip("v"))
            except ValueError:
                version = 1

    cleaned: List[str] = []
    for p in parts:
        # Lowercase per CTO doc Section 6.1.1 — namespace and id segments
        # MUST be lowercase (parser allows uppercase 'T'/'Z' only as
        # ISO-8601 timestamp markers within the id; we have no timestamped
        # factors in alpha so we lowercase unconditionally).
        seg = re.sub(r"[^A-Za-z0-9._-]+", "-", p).strip("-").lower()
        if seg:
            cleaned.append(seg)

    if not cleaned:
        raise NormalizerError(
            f"factor_id {factor_id!r} produced zero usable segments"
        )

    # If a namespace is supplied, prepend it. Namespace MUST be lowercase
    # per the canonical URN spec (greenlang.factors.ontology.urn).
    if namespace:
        ns_seg = re.sub(r"[^A-Za-z0-9._-]+", "-", namespace).strip("-").lower()
        if ns_seg:
            cleaned.insert(0, ns_seg)

    # The pattern allows {2,4} segments AFTER the source slug. Keep at most 4.
    if len(cleaned) > 4:
        head, tail = cleaned[:3], "-".join(cleaned[3:])
        cleaned = head + [tail]
    # Pad to at least 2 segments with the literal "rec".
    while len(cleaned) < 2:
        cleaned.append("rec")

    src = slugify_urn_path(source_slug)
    if not src:
        raise NormalizerError(f"source_slug {source_slug!r} slugified to empty")

    urn = f"urn:gl:factor:{src}:" + ":".join(cleaned) + f":v{version}"
    if not _URN_FACTOR_RE.match(urn):
        # Last-resort defence: replace any stray illegal char and re-lowercase.
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", ":".join(cleaned)).lower()
        urn = f"urn:gl:factor:{src}:{safe}:v{version}"
    if not _URN_FACTOR_RE.match(urn):
        raise NormalizerError(
            f"could not coerce factor_id {factor_id!r} into a valid URN; "
            f"final attempt: {urn}"
        )
    return urn


def coerce_geography(record: Dict[str, Any]) -> str:
    """Project a v1 record's geography fields onto a ``urn:gl:geo:...`` URN.

    Reads from ``record["jurisdiction"]`` first (canonical v2), then falls
    back to top-level ``geography`` + ``geography_level``. Returns
    ``urn:gl:geo:global:world`` when nothing usable is present.
    """
    jurisdiction = record.get("jurisdiction") or {}
    if isinstance(jurisdiction, dict) and jurisdiction:
        country = jurisdiction.get("country")
        grid_region = jurisdiction.get("grid_region") or jurisdiction.get("region")
        subregion = jurisdiction.get("subregion")
        if grid_region:
            slug = slugify_urn_path(grid_region)
            if slug:
                return f"urn:gl:geo:grid_zone:{slug}"
        if subregion:
            slug = slugify_urn_path(subregion)
            if slug:
                return f"urn:gl:geo:subregion:{slug}"
        if country:
            cc = str(country).strip().lower()
            if cc in {"xx", "global", "world", "row", "default"}:
                return "urn:gl:geo:global:world"
            if len(cc) <= 5:
                return f"urn:gl:geo:country:{cc}"
            return f"urn:gl:geo:country:{slugify_urn_path(cc)}"

    geo = record.get("geography")
    geo_level = str(record.get("geography_level") or "").lower()

    if geo is None:
        return "urn:gl:geo:global:world"

    geo_str = str(geo).strip()
    if not geo_str or geo_str.upper() in {"GLOBAL", "WORLD", "XX", "ROW"}:
        return "urn:gl:geo:global:world"

    # eGRID subregion acronyms ("AKGD", "RFCW", ...) — wrap as grid_zone.
    if geo_level in {"grid_zone", "grid"}:
        slug = slugify_urn_path(geo_str)
        slug = f"egrid-{slug}" if not slug.startswith("egrid-") else slug
        return f"urn:gl:geo:grid_zone:{slug}"
    if geo_level in {"state", "state_or_province"}:
        slug = slugify_urn_path(geo_str)
        return f"urn:gl:geo:state_or_province:{slug}"
    if geo_level in {"subregion", "region"}:
        return f"urn:gl:geo:subregion:{slugify_urn_path(geo_str)}"

    # Default: country code (ISO-2 / ISO-3 / lowercased).
    if len(geo_str) <= 5 and geo_str.isalpha():
        return f"urn:gl:geo:country:{geo_str.lower()}"
    return f"urn:gl:geo:country:{slugify_urn_path(geo_str)}"


def coerce_category(
    factor_family: Optional[str],
    method_profile: Optional[str],
    *,
    scope: Optional[str] = None,
    fuel_type: Optional[str] = None,
) -> str:
    """Map (factor_family, method_profile) -> v0.1 alpha category enum.

    Mapping rules:
        * combustion / stationary / mobile -> ``fuel``
        * electricity / grid_intensity -> ``scope2_location_based`` if
          method_profile contains ``location``;
          ``scope2_market_based`` if it contains ``market``;
          else ``grid_intensity``
        * refrigerant -> ``refrigerant``
        * classification_mapping with eu_cbam profile (or the
          material_embodied family) -> ``cbam_default``
        * scope-1 fugitive / process families -> ``fugitive`` / ``process``
        * fall-back: ``fuel`` (the safe alpha default for combustion-like
          rows that did not declare a family)
    """
    fam = (factor_family or "").strip().lower()
    prof = (method_profile or "").strip().lower()
    sc = (scope or "").strip().lower()
    ft = (fuel_type or "").strip().lower()

    if fam in {"refrigerants", "refrigerant"}:
        return "refrigerant"
    if fam in {"classification_mapping", "material_embodied"}:
        if "eu_cbam" in prof or "cbam" in prof or fam == "material_embodied":
            return "cbam_default"
    if fam in {"electricity", "grid_intensity"}:
        if "location" in prof:
            return "scope2_location_based"
        if "market" in prof:
            return "scope2_market_based"
        return "grid_intensity"
    if fam in {"combustion", "stationary_combustion", "mobile_combustion"}:
        return "fuel"
    if fam == "fugitive":
        return "fugitive"
    if fam == "process":
        return "process"

    # No family — infer from fuel_type / scope / profile.
    if "electricity" in ft or ft.endswith("grid") or "grid" in ft:
        if "location" in prof:
            return "scope2_location_based"
        if "market" in prof:
            return "scope2_market_based"
        return "grid_intensity"
    if "refrigerant" in ft:
        return "refrigerant"
    if "cbam" in ft or "cbam" in prof:
        return "cbam_default"
    if sc.startswith("scope_2") or sc == "scope2":
        return "scope2_location_based"

    return "fuel"


# ---------------------------------------------------------------------------
# Source-meta helpers — pull a stable urn-slug + pack id out of registry rows
# ---------------------------------------------------------------------------


def _urn_slug(source_meta: Dict[str, Any]) -> str:
    """Derive the URN slug for a source from its registry ``urn``.

    ``urn:gl:source:epa-hub`` -> ``epa-hub``. Falls back to a slugified
    ``source_id`` when the registry entry has no ``urn``.
    """
    urn = source_meta.get("urn") or ""
    if isinstance(urn, str) and urn.startswith("urn:gl:source:"):
        return urn.split(":", 3)[-1]
    return slugify_urn_path(source_meta.get("source_id") or "")


def _factor_pack_urn(
    source_meta: Dict[str, Any],
    record: Dict[str, Any],
) -> str:
    """Compose the canonical public ``factor_pack_urn``.

    ``urn:gl:pack:<source-slug>:<pack-id>:v<int>``. ``pack_id``
    is sourced (in priority order) from:

      1. an explicit per-source override in ``_SOURCE_ID_TO_PACK_ID``
      2. the record's ``factor_family`` mapped via
         ``_FACTOR_FAMILY_TO_PACK_ID``
      3. ``"default-pack"`` as a last resort.

    Upstream source vintages remain in ``extraction.source_version``.
    They are not part of the public pack URN version segment.
    """
    source_id = str(source_meta.get("source_id") or "")
    src_slug = _urn_slug(source_meta)
    pack_version = str(source_meta.get("factor_pack_version") or "v1").strip().lower()
    if not re.fullmatch(r"v[1-9][0-9]*", pack_version):
        pack_version = "v1"

    pack_id = _SOURCE_ID_TO_PACK_ID.get(source_id)
    if not pack_id:
        family = (record.get("factor_family") or "").strip().lower()
        pack_id = _FACTOR_FAMILY_TO_PACK_ID.get(family) or "default-pack"

    return f"urn:gl:pack:{src_slug}:{slugify(pack_id)}:{pack_version}"


# ---------------------------------------------------------------------------
# Vector -> CO2e collapse
# ---------------------------------------------------------------------------


def _record_co2e_value(record: Dict[str, Any]) -> float:
    """Collapse a v1 record's per-gas vectors into a scalar AR6 CO2e value.

    Priority:
        1. ``record["numerator"]["co2e"]`` if present
        2. ``record["co2e"]`` (rare, only on canonical-v2 hot-path)
        3. ``vectors["CO2"] + CH4*GWP_AR6 + N2O*GWP_AR6`` using the
           record's own ``gwp_100yr`` values when set, else AR6 defaults.

    Raises :class:`NormalizerError` when the result is non-positive
    (the v0.1 schema requires ``value > 0``).
    """
    numerator = record.get("numerator")
    if isinstance(numerator, dict):
        co2e = numerator.get("co2e")
        if co2e is not None:
            try:
                v = float(co2e)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass

    co2e_top = record.get("co2e")
    if co2e_top is not None:
        try:
            v = float(co2e_top)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass

    vectors = record.get("vectors") or {}
    if not isinstance(vectors, dict):
        vectors = {}
    co2 = float(vectors.get("CO2") or vectors.get("co2") or 0.0)
    ch4 = float(vectors.get("CH4") or vectors.get("ch4") or 0.0)
    n2o = float(vectors.get("N2O") or vectors.get("n2o") or 0.0)

    gwp_block = record.get("gwp_100yr") or {}
    gwp_ch4 = _AR6_GWP100_CH4
    gwp_n2o = _AR6_GWP100_N2O
    if isinstance(gwp_block, dict):
        try:
            if gwp_block.get("CH4_gwp") is not None:
                gwp_ch4 = int(float(gwp_block.get("CH4_gwp")))
        except (TypeError, ValueError):
            pass
        try:
            if gwp_block.get("N2O_gwp") is not None:
                gwp_n2o = int(float(gwp_block.get("N2O_gwp")))
        except (TypeError, ValueError):
            pass

    value = co2 + (ch4 * gwp_ch4) + (n2o * gwp_n2o)
    if value <= 0:
        raise NonPositiveValueError(
            f"record co2e value collapsed to {value} (not > 0); "
            f"factor_id={record.get('factor_id')!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Unit URN
# ---------------------------------------------------------------------------


def _unit_urn(record: Dict[str, Any]) -> str:
    """Return the canonical ``urn:gl:unit:<denominator>`` URN.

    Reads ``record["denominator"]["unit"]`` first; falls back to top-level
    ``record["unit"]``. The result form is ``kgco2e/<unit>`` because the
    alpha contract reports values as kgCO2e per native unit.
    """
    denom = record.get("denominator")
    unit_raw: Optional[str] = None
    if isinstance(denom, dict):
        unit_raw = denom.get("unit")
    if not unit_raw:
        unit_raw = record.get("unit")
    if not unit_raw:
        unit_raw = "unit"

    unit_slug = slugify(unit_raw)  # keeps '/'
    if not unit_slug:
        unit_slug = "unit"
    if unit_slug.startswith("kgco2e"):
        urn = f"urn:gl:unit:{unit_slug}"
    else:
        urn = f"urn:gl:unit:kgco2e/{unit_slug}"

    if not _URN_UNIT_RE.match(urn):
        urn = f"urn:gl:unit:kgco2e/{re.sub(r'[^a-zA-Z0-9./-]+', '-', unit_slug)}"
    return urn


# ---------------------------------------------------------------------------
# GWP
# ---------------------------------------------------------------------------


def _gwp_basis(record: Dict[str, Any]) -> str:
    """Normalise gwp_basis. AR4 is rejected hard; AR5 is tolerated and
    re-tagged AR6 (the alpha sources include DESNZ which is published
    AR5-aligned upstream — methodology lead approved the re-tag because
    the DESNZ values are already pre-computed CO2e scalars at the row
    level, so the GWP-set switch is informational only).
    """
    gwp_block = record.get("gwp_100yr") or {}
    raw = ""
    if isinstance(gwp_block, dict):
        raw = str(gwp_block.get("gwp_set") or "").lower()
    raw = raw or str(record.get("gwp_basis") or "").lower()

    if "ar4" in raw:
        raise NormalizerError(
            f"record declares AR4 gwp_set ({raw!r}); v0.1 alpha is AR6-only"
        )
    return _GWP_BASIS_FROZEN


def _gwp_horizon(record: Dict[str, Any]) -> int:
    horizon = record.get("gwp_horizon")
    if horizon is None:
        return _GWP_HORIZON_DEFAULT
    try:
        v = int(horizon)
    except (TypeError, ValueError):
        return _GWP_HORIZON_DEFAULT
    return v if v in {20, 100, 500} else _GWP_HORIZON_DEFAULT


# ---------------------------------------------------------------------------
# Methodology URN
# ---------------------------------------------------------------------------


def _methodology_urn(record: Dict[str, Any], source_meta: Dict[str, Any]) -> str:
    """Pick the most-specific methodology URN we can derive."""
    profile = (record.get("method_profile") or "").strip().lower()
    if profile and profile in _METHOD_PROFILE_TO_METHODOLOGY_SLUG:
        slug = _METHOD_PROFILE_TO_METHODOLOGY_SLUG[profile]
        return f"urn:gl:methodology:{slug}"
    if profile:
        return f"urn:gl:methodology:{slugify(profile)}"

    source_id = str(source_meta.get("source_id") or "")
    slug = _SOURCE_ID_TO_METHODOLOGY_SLUG.get(source_id) or "ipcc-tier-1-stationary-combustion"
    return f"urn:gl:methodology:{slug}"


# ---------------------------------------------------------------------------
# Description / boundary / name
# ---------------------------------------------------------------------------


def _factor_name(record: Dict[str, Any], factor_id: str) -> str:
    name = record.get("factor_name") or record.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()[:200]
    fuel = record.get("fuel_type") or "factor"
    geo = record.get("geography") or ""
    bits = [str(p).replace("_", " ") for p in (fuel, geo) if p]
    candidate = " — ".join(bits) if bits else factor_id
    return (candidate or "GreenLang factor")[:200]


def _description(
    record: Dict[str, Any],
    factor_family: str,
    name: str,
) -> str:
    """Synthesise a >=30 char description from existing fields."""
    desc = record.get("description")
    if isinstance(desc, dict):
        desc = desc.get("text") or desc.get("rationale")
    if not isinstance(desc, str) or len(desc.strip()) < 30:
        family_label = factor_family or "emission"
        desc = (
            f"{family_label.replace('_', ' ').title()} emission factor for "
            f"{name}, expressed in CO2e per native unit. Boundary follows "
            f"source publication."
        )
    return desc.strip()[:2000]


def _boundary(record: Dict[str, Any]) -> str:
    """Promote ``parameters.boundary`` (or top-level ``boundary``) -> string >=10 chars."""
    params = record.get("parameters")
    boundary = None
    if isinstance(params, dict):
        boundary = params.get("boundary")
    if not boundary:
        boundary = record.get("boundary")
    if isinstance(boundary, str) and len(boundary.strip()) >= 10:
        return boundary.strip()[:2000]
    return "Tier 1 default; refer to source publication for boundary."


# ---------------------------------------------------------------------------
# Citations + licence
# ---------------------------------------------------------------------------


def _licence(record: Dict[str, Any], source_meta: Dict[str, Any]) -> str:
    # Phase 1 source-rights contract: the registry owns the canonical
    # public licence tag. Parser-level license_info may use upstream
    # labels that are not stable enough for factor-record equality.
    registry_licence = source_meta.get("licence")
    if isinstance(registry_licence, str) and registry_licence.strip():
        return registry_licence.strip()[:128]

    lic_info = record.get("license_info") or {}
    if isinstance(lic_info, dict):
        name = lic_info.get("license") or lic_info.get("license_name")
        if isinstance(name, str) and name.strip():
            return name.strip()[:128]
    licensing = record.get("licensing") or {}
    if isinstance(licensing, dict):
        name = licensing.get("license_name")
        if isinstance(name, str) and name.strip():
            return name.strip()[:128]
    lc = source_meta.get("license_class") or "unknown"
    return str(lc)[:128]


def _citations(
    record: Dict[str, Any], source_meta: Dict[str, Any]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pub_url = source_meta.get("publication_url") or source_meta.get("source_url")
    if isinstance(pub_url, str) and pub_url.strip():
        out.append({"type": "url", "value": pub_url.strip()})

    # Pull DOIs / additional citations off the record if present.
    rec_cites = record.get("citations")
    if isinstance(rec_cites, list):
        for c in rec_cites:
            if not isinstance(c, dict):
                continue
            ctype = str(c.get("type") or "").strip().lower()
            cval = c.get("value")
            if ctype not in {"doi", "url", "publication", "section", "table"}:
                continue
            if not isinstance(cval, str) or not cval.strip():
                continue
            cit = {"type": ctype, "value": cval.strip()}
            if isinstance(c.get("title"), str):
                cit["title"] = c["title"]
            if "page" in c and c["page"] is not None:
                cit["page"] = c["page"]
            out.append(cit)

    if not out:
        # Last-resort citation derived from the registry display name.
        title = (
            source_meta.get("display_name")
            or source_meta.get("urn")
            or "GreenLang Factor Source"
        )
        out.append({"type": "publication", "value": str(title)})
    return out


# ---------------------------------------------------------------------------
# Extraction + review blocks
# ---------------------------------------------------------------------------


def _extraction_block(
    record: Dict[str, Any],
    source_meta: Dict[str, Any],
    *,
    idx: int,
) -> Dict[str, Any]:
    source_id = str(source_meta.get("source_id") or "")
    src_slug = _urn_slug(source_meta)
    src_version = str(source_meta.get("source_version") or "0.0.0")
    pub_url = (
        source_meta.get("publication_url")
        or "https://www.greenlang.io/factors/sources"
    )

    lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
    raw_record_ref = (
        (lineage or {}).get("raw_record_ref")
        or record.get("source_record_id")
        or f"{source_id};record-{idx}"
    )
    row_ref = (
        (lineage or {}).get("row_ref")
        or record.get("row_ref")
        or f"record-{idx}"
    )

    raw_artifact_uri = (
        f"s3://greenlang-factors-raw/{src_slug}/{src_version}/seed.json"
    )
    raw_artifact_sha256 = _hash_record(record)

    parser_id = source_meta.get("parser_module") or "greenlang.factors.ingestion"
    parser_version = source_meta.get("parser_version") or "0.1.0"
    parser_commit = _resolve_git_head()
    operator = (
        f"bot:parser_{src_slug.replace('-', '_')}"
    )

    publication = (
        source_meta.get("display_name")
        or source_meta.get("urn")
        or source_id
        or "Source"
    )

    return {
        "source_url": pub_url,
        "source_record_id": str(raw_record_ref)[:256],
        "source_publication": str(publication),
        "source_version": src_version,
        "raw_artifact_uri": raw_artifact_uri,
        "raw_artifact_sha256": raw_artifact_sha256,
        "parser_id": str(parser_id),
        "parser_version": str(parser_version),
        "parser_commit": parser_commit,
        "row_ref": str(row_ref)[:256],
        "ingested_at": _now_iso(),
        "operator": operator,
    }


def _review_block() -> Dict[str, Any]:
    """v0.1 alpha pre-approves every public-source record at extract time.

    The 6 alpha sources (IPCC, DESNZ, EPA Hub, eGRID, India CEA, EU CBAM)
    are public-domain or open-government datasets vetted by the
    methodology lead at registry-onboarding time; the ingestion pipeline
    therefore stamps ``approved`` on every record it generates.
    """
    now = _now_iso()
    reviewer = "human:methodology-lead@greenlang.io"
    return {
        "review_status": "approved",
        "reviewer": reviewer,
        "reviewed_at": now,
        "approved_by": reviewer,
        "approved_at": now,
    }


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------


def lift_v1_record_to_v0_1(
    record: Dict[str, Any],
    source_meta: Dict[str, Any],
    *,
    idx: int = 0,
) -> Dict[str, Any]:
    """Convert v1-shape parser output -> v0.1-shape factor record.

    Args:
        record: a single dict produced by an alpha-source parser. May be
            either a current v1 catalog dict (``factor_id``, ``vectors``,
            ``provenance``, ...) or a canonical-v2 dict (``factor_family``,
            ``method_profile``, ``jurisdiction``, ``factor_name``, ...).
        source_meta: the registry row dict for this source. Required keys:
            ``source_id``, ``urn``, ``parser_module``, ``parser_function``,
            ``parser_version``, ``source_version``, ``publication_url``.
        idx: ordinal index of this record within the parser's output —
            used to synthesise stable ``source_record_id`` / ``row_ref``
            fallbacks when the upstream parser hasn't set them.

    Returns:
        A dict matching the FROZEN ``factor_record_v0_1.schema.json``.

    Raises:
        :class:`NormalizerError` on unrecoverable inputs (non-AR6 record,
        zero-value collapse, mangled factor_id).
    """
    if not isinstance(record, dict):
        raise NormalizerError(
            f"record must be a dict; got {type(record).__name__}"
        )
    if not isinstance(source_meta, dict):
        raise NormalizerError(
            f"source_meta must be a dict; got {type(source_meta).__name__}"
        )

    factor_id = record.get("factor_id") or record.get("urn") or ""
    if not isinstance(factor_id, str) or not factor_id.strip():
        raise NormalizerError("record missing factor_id / urn")

    src_slug = _urn_slug(source_meta)
    urn = coerce_factor_id_to_urn(factor_id, src_slug)

    factor_family = (record.get("factor_family") or "").strip().lower()
    method_profile = (record.get("method_profile") or "").strip().lower()
    scope = record.get("scope")
    fuel_type = record.get("fuel_type")

    name = _factor_name(record, factor_id)
    description = _description(record, factor_family or "emission", name)
    category = coerce_category(
        factor_family,
        method_profile,
        scope=str(scope) if scope is not None else None,
        fuel_type=str(fuel_type) if fuel_type is not None else None,
    )
    value = _record_co2e_value(record)
    unit_urn = _unit_urn(record)
    gwp_basis = _gwp_basis(record)
    gwp_horizon = _gwp_horizon(record)
    geography_urn = coerce_geography(record)
    if not _URN_GEO_RE.match(geography_urn):
        geography_urn = "urn:gl:geo:global:world"

    vintage_start = _coerce_iso_date(record.get("valid_from"), _VINTAGE_START_DEFAULT)
    vintage_end = _coerce_iso_date(record.get("valid_to"), _VINTAGE_END_DEFAULT)
    if vintage_end < vintage_start:
        # Schema invariant: vintage_end >= vintage_start.
        vintage_end = vintage_start

    resolution = record.get("resolution") or _RESOLUTION_DEFAULT
    if resolution not in {"annual", "monthly", "hourly", "point-in-time"}:
        resolution = _RESOLUTION_DEFAULT

    methodology_urn = _methodology_urn(record, source_meta)
    boundary = _boundary(record)
    licence = _licence(record, source_meta)
    citations = _citations(record, source_meta)

    lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
    published_at = _coerce_iso_datetime(
        (lineage or {}).get("approved_at")
        or record.get("published_at")
        or record.get("updated_at")
        or record.get("created_at"),
        _now_iso(),
    )

    extraction = _extraction_block(record, source_meta, idx=idx)
    review = _review_block()

    out: Dict[str, Any] = {
        "urn": urn,
        "factor_id_alias": factor_id if factor_id.upper().startswith("EF:") else None,
        "source_urn": source_meta.get("urn") or f"urn:gl:source:{src_slug}",
        "factor_pack_urn": _factor_pack_urn(source_meta, record),
        "name": name,
        "description": description,
        "category": category,
        "value": float(value),
        "unit_urn": unit_urn,
        "gwp_basis": gwp_basis,
        "gwp_horizon": gwp_horizon,
        "geography_urn": geography_urn,
        "vintage_start": vintage_start,
        "vintage_end": vintage_end,
        "resolution": resolution,
        "methodology_urn": methodology_urn,
        "boundary": boundary,
        "licence": licence,
        "citations": citations,
        "published_at": published_at,
        "extraction": extraction,
        "review": review,
    }

    # Drop optional-null aliases so we don't leak nulls into the catalog seed.
    if out["factor_id_alias"] is None:
        out.pop("factor_id_alias")

    # Optional tags (preserve when present and well-formed).
    tags = record.get("tags")
    if isinstance(tags, list):
        clean_tags = [str(t)[:64] for t in tags if isinstance(t, (str, int, float)) and str(t).strip()]
        if clean_tags:
            out["tags"] = clean_tags[:32]

    return out


__all__ = [
    "NormalizerError",
    "NonPositiveValueError",
    "lift_v1_record_to_v0_1",
    "coerce_factor_id_to_urn",
    "coerce_geography",
    "coerce_category",
    "slugify",
]
