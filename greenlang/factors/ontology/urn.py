# -*- coding: utf-8 -*-
"""GreenLang Factors v0.1 Alpha - Canonical URN scheme.

Public IDs in the GreenLang Factors catalog are URNs (RFC-2141-like opaque
identifiers under the ``urn:gl:`` namespace). They are immutable, opaque to
clients, and never change once published. Integer primary keys remain
internal implementation details. The legacy ``EF:...`` identifier is
retained ONLY as a non-canonical alias and can be lifted into a URN via
:func:`coerce_factor_id_to_urn`.

CTO doc Section 6.1.1 / Section 19 defines the kinds and component
contracts.

Kind contract (kind -> required components -> output form)

* ``factor``       : ``source``, ``namespace``, ``id``, ``version`` (int>=1)
                     ``urn:gl:factor:<source>:<namespace>:<id>:v<version>``
* ``source``       : ``slug``
                     ``urn:gl:source:<slug>``
* ``pack``         : ``source``, ``pack_id``, ``version`` (int>=1)
                     ``urn:gl:pack:<source>:<pack-id>:v<version>``
* ``methodology``  : ``slug``
                     ``urn:gl:methodology:<slug>``
* ``geo``          : ``geo_type``, ``id``
                     ``urn:gl:geo:<type>:<id>``
* ``unit``         : ``symbol`` (may contain '/' and '.')
                     ``urn:gl:unit:<symbol>``
* ``activity``     : ``slug``
                     ``urn:gl:activity:<slug>``
                     OR (Phase 2 taxonomy form): ``taxonomy``, ``code``
                     ``urn:gl:activity:<taxonomy>:<code-slug>``
* ``community``    : ``slug``
                     ``urn:gl:community:<slug>``
* ``partner``      : ``tenant``, ``pack_id``, ``version`` (int>=1)
                     ``urn:gl:partner:<tenant>:<pack-id>:v<version>``
* ``enterprise``   : ``tenant``, ``pack_id``, ``version`` (int>=1)
                     ``urn:gl:enterprise:<tenant>:<pack-id>:v<version>``

Slug grammar:
    * ``<source>``, ``<slug>``, ``<tenant>``, ``<pack-id>``, ``<geo-id>``:
      lowercase ``[a-z0-9]`` plus hyphen ``-`` and dot ``.``.
    * ``<namespace>``: same plus underscore ``_``.
    * ``<id>`` (factor): same plus underscore ``_`` and colon ``:`` and
      uppercase ``T`` / ``Z`` for ISO-8601 timestamp markers.
    * ``<symbol>`` (unit): lowercase, digits, hyphen, dot, slash,
      underscore, caret, percent.
    * ``<version>``: positive integer printed as ``v<n>``.

Public surface:
    * :class:`GLUrn`           - pydantic v2 model with parsed components
    * :class:`InvalidUrnError` - raised on malformed input
    * :func:`parse`            - strict parser
    * :func:`build`            - typed builder
    * :func:`validate`         - boolean check
    * :func:`coerce_factor_id_to_urn` - migration helper for ``EF:`` ids
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "GLUrn",
    "InvalidUrnError",
    "parse",
    "build",
    "validate",
    "validate_urn",
    "coerce_factor_id_to_urn",
    "URN_PREFIX",
    "ALLOWED_KINDS",
    "ALLOWED_GEO_TYPES",
    "ALLOWED_ACTIVITY_TAXONOMIES",
]


URN_PREFIX = "urn:gl"

KindLiteral = Literal[
    "factor",
    "source",
    "pack",
    "methodology",
    "geo",
    "unit",
    "activity",
    "community",
    "partner",
    "enterprise",
]

ALLOWED_KINDS: Tuple[str, ...] = (
    "factor",
    "source",
    "pack",
    "methodology",
    "geo",
    "unit",
    "activity",
    "community",
    "partner",
    "enterprise",
)

ALLOWED_GEO_TYPES: Tuple[str, ...] = (
    "country",
    "subregion",
    "grid_zone",
    "bidding_zone",
    "balancing_authority",
    "state",
    "province",
    # ``state_or_province`` is the canonical type used by the FROZEN
    # ``factor_record_v0_1.schema.json`` (geography_urn regex) and by the
    # V500 Postgres CHECK on ``factors_v0_1.geography.type``. Phase 2
    # ontology seed (``geography_seed_v0_1.yaml``) emits URNs of this
    # form for sub-national administrative regions (e.g.,
    # ``urn:gl:geo:state_or_province:us-tx``). Additive — does not
    # remove the legacy ``state`` / ``province`` types.
    "state_or_province",
    "region",
    "basin",
    "tenant",
    "global",
)

# Phase 2 (WS5) — allowed taxonomy slugs for the optional taxonomy/code
# segmentation of an activity URN. Lower-case-only. CTO list.
ALLOWED_ACTIVITY_TAXONOMIES: Tuple[str, ...] = (
    "ipcc",
    "ghgp",
    "hs-cn",
    "cpc",
    "nace",
    "naics",
    "sic",
    "pact",
    "freight",
    "cbam",
    "pcf",
    "refrigerants",
    "agriculture",
    "waste",
    "land-use",
)

_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.\-]*[a-z0-9])?$")
# Activity taxonomy code: allows dot, underscore, hyphen in interior. Must
# start with [a-z0-9]; trailing char is unrestricted within the allowed
# class, so single-char codes are permitted (e.g. cpc:0). Phase 2 (WS5).
_ACTIVITY_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9._\-]*$")
_NAMESPACE_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._\-]*[a-z0-9])?$")
_FACTOR_ID_RE = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9._:\-]*[a-zA-Z0-9])?$")
_GEO_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.\-]*[a-z0-9])?$")
_UNIT_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.\-_/^%]*[a-z0-9%])?$")
_VERSION_RE = re.compile(r"^v([1-9][0-9]*)$")
_MAX_URN_LEN = 512


class InvalidUrnError(ValueError):
    """Raised when a string is not a valid GreenLang URN."""


class GLUrn(BaseModel):
    """Parsed GreenLang URN. Only ``kind``-relevant fields are populated."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", str_strip_whitespace=False
    )

    kind: KindLiteral = Field(..., description="URN kind discriminator.")
    source: Optional[str] = None
    namespace: Optional[str] = None
    id: Optional[str] = None
    version: Optional[int] = Field(default=None, ge=1)
    slug: Optional[str] = None
    pack_id: Optional[str] = None
    geo_type: Optional[str] = None
    symbol: Optional[str] = None
    tenant: Optional[str] = None
    # Phase 2 (WS5) — optional activity taxonomy + code segmentation.
    # Populated only for `urn:gl:activity:<taxonomy>:<code-slug>` form.
    # The legacy `urn:gl:activity:<slug>` form continues to populate
    # ``slug`` and leaves these as None.
    taxonomy: Optional[str] = None
    code: Optional[str] = None

    def render(self) -> str:
        """Render this URN to its canonical string form."""
        return _render(self)

    def __str__(self) -> str:  # pragma: no cover
        return self.render()

    @property
    def is_factor(self) -> bool:
        return self.kind == "factor"

    @property
    def is_pack(self) -> bool:
        return self.kind in ("pack", "partner", "enterprise")


def _split_prefix(s: str) -> Tuple[str, str]:
    if not isinstance(s, str):
        raise InvalidUrnError(
            f"URN must be a string, got {type(s).__name__}"
        )
    if not s:
        raise InvalidUrnError("URN is empty")
    if len(s) > _MAX_URN_LEN:
        raise InvalidUrnError(
            f"URN exceeds max length {_MAX_URN_LEN} (got {len(s)})"
        )
    if not s.startswith("urn:gl:"):
        raise InvalidUrnError(
            f"URN must start with 'urn:gl:' (got prefix {s[:7]!r})"
        )
    rest = s[len("urn:gl:"):]
    if not rest:
        raise InvalidUrnError("URN has no kind segment after 'urn:gl:'")
    colon_idx = rest.find(":")
    if colon_idx <= 0:
        raise InvalidUrnError(
            f"URN missing kind/body separator after 'urn:gl:': {s!r}"
        )
    kind = rest[:colon_idx]
    body = rest[colon_idx + 1:]
    if kind not in ALLOWED_KINDS:
        raise InvalidUrnError(
            f"Unknown URN kind {kind!r}; expected one of {ALLOWED_KINDS}"
        )
    if not body:
        raise InvalidUrnError(f"URN {kind!r} has empty body: {s!r}")
    return kind, body


def _check_no_empty_segments(
    parts: Iterable[str], kind: str, original: str
) -> None:
    for p in parts:
        if p == "":
            raise InvalidUrnError(
                f"URN {kind!r} has empty segment: {original!r}"
            )


def _parse_version(seg: str, original: str) -> int:
    m = _VERSION_RE.match(seg)
    if not m:
        raise InvalidUrnError(
            f"URN version segment must match 'v<positive-int>', got "
            f"{seg!r} in {original!r}"
        )
    return int(m.group(1))


def _validate_slug(value: str, field: str, original: str) -> None:
    if not _SLUG_RE.match(value):
        raise InvalidUrnError(
            f"URN field {field!r}={value!r} is not a valid slug "
            f"(lowercase a-z 0-9 hyphen dot only): {original!r}"
        )


def _validate_namespace(value: str, original: str) -> None:
    if not _NAMESPACE_RE.match(value):
        raise InvalidUrnError(
            f"URN namespace={value!r} is invalid "
            f"(lowercase a-z 0-9 hyphen dot underscore only): {original!r}"
        )


def _validate_factor_id(value: str, original: str) -> None:
    if not _FACTOR_ID_RE.match(value):
        raise InvalidUrnError(
            f"URN factor id={value!r} is invalid: {original!r}"
        )
    for ch in value:
        if ch.isupper() and ch not in ("T", "Z"):
            raise InvalidUrnError(
                f"URN factor id={value!r} contains disallowed uppercase "
                f"character {ch!r} (only 'T' and 'Z' permitted for "
                f"ISO-8601 timestamps): {original!r}"
            )


def _validate_geo_id(value: str, original: str) -> None:
    if not _GEO_ID_RE.match(value):
        raise InvalidUrnError(
            f"URN geo id={value!r} is invalid: {original!r}"
        )


def _validate_unit_symbol(value: str, original: str) -> None:
    if not _UNIT_RE.match(value):
        raise InvalidUrnError(
            f"URN unit symbol={value!r} is invalid: {original!r}"
        )


def parse(s: str) -> GLUrn:
    """Strictly parse a GreenLang URN string into a :class:`GLUrn` model.

    Raises:
        InvalidUrnError: if the string is not a well-formed GreenLang URN.
    """
    kind, body = _split_prefix(s)

    if kind == "factor":
        last_colon = body.rfind(":")
        if last_colon < 0:
            raise InvalidUrnError(
                f"URN factor missing ':v<version>' suffix: {s!r}"
            )
        version_seg = body[last_colon + 1:]
        head = body[:last_colon]
        version = _parse_version(version_seg, s)
        parts = head.split(":", 2)
        if len(parts) != 3:
            raise InvalidUrnError(
                f"URN factor must have form "
                f"'urn:gl:factor:<source>:<namespace>:<id>:v<version>': "
                f"{s!r}"
            )
        source, namespace, fid = parts
        _check_no_empty_segments((source, namespace, fid), kind, s)
        _validate_slug(source, "source", s)
        _validate_namespace(namespace, s)
        _validate_factor_id(fid, s)
        return GLUrn(
            kind="factor",
            source=source,
            namespace=namespace,
            id=fid,
            version=version,
        )

    if kind == "source":
        slug = body
        _check_no_empty_segments((slug,), kind, s)
        if ":" in slug:
            raise InvalidUrnError(
                f"URN source must have no further ':' after slug: {s!r}"
            )
        _validate_slug(slug, "slug", s)
        return GLUrn(kind="source", slug=slug)

    if kind == "pack":
        parts = body.split(":")
        if len(parts) != 3:
            raise InvalidUrnError(
                f"URN pack must have form "
                f"'urn:gl:pack:<source>:<pack-id>:v<version>': {s!r}"
            )
        source, pack_id, version_seg = parts
        _check_no_empty_segments(parts, kind, s)
        _validate_slug(source, "source", s)
        _validate_slug(pack_id, "pack_id", s)
        version = _parse_version(version_seg, s)
        return GLUrn(
            kind="pack", source=source, pack_id=pack_id, version=version
        )

    if kind == "methodology":
        slug = body
        if ":" in slug:
            raise InvalidUrnError(
                f"URN methodology must have no further ':' after slug: "
                f"{s!r}"
            )
        _check_no_empty_segments((slug,), kind, s)
        _validate_slug(slug, "slug", s)
        return GLUrn(kind="methodology", slug=slug)

    if kind == "geo":
        parts = body.split(":", 1)
        if len(parts) != 2:
            raise InvalidUrnError(
                f"URN geo must have form 'urn:gl:geo:<type>:<id>': {s!r}"
            )
        geo_type, geo_id = parts
        _check_no_empty_segments(parts, kind, s)
        if geo_type not in ALLOWED_GEO_TYPES:
            raise InvalidUrnError(
                f"URN geo type={geo_type!r} is not one of "
                f"{ALLOWED_GEO_TYPES}: {s!r}"
            )
        if ":" in geo_id:
            raise InvalidUrnError(
                f"URN geo id must not contain ':' (got {geo_id!r}): {s!r}"
            )
        _validate_geo_id(geo_id, s)
        return GLUrn(kind="geo", geo_type=geo_type, id=geo_id)

    if kind == "unit":
        symbol = body
        if ":" in symbol:
            raise InvalidUrnError(
                f"URN unit symbol must not contain ':' (got "
                f"{symbol!r}): {s!r}"
            )
        _check_no_empty_segments((symbol,), kind, s)
        _validate_unit_symbol(symbol, s)
        return GLUrn(kind="unit", symbol=symbol)

    if kind == "activity":
        # Phase 2 (WS5): activity URN may have either form:
        #   1. legacy: urn:gl:activity:<slug>            -> slug
        #   2. taxonomy form: urn:gl:activity:<taxonomy>:<code-slug>
        #                                                 -> taxonomy + code
        # We disambiguate by counting colons in the body. The taxonomy
        # form has exactly ONE additional colon. More than one colon is
        # invalid for either form.
        colon_count = body.count(":")
        if colon_count == 0:
            slug = body
            _check_no_empty_segments((slug,), kind, s)
            _validate_slug(slug, "slug", s)
            return GLUrn(kind="activity", slug=slug)
        if colon_count == 1:
            taxonomy, code = body.split(":", 1)
            _check_no_empty_segments((taxonomy, code), kind, s)
            if taxonomy not in ALLOWED_ACTIVITY_TAXONOMIES:
                raise InvalidUrnError(
                    f"URN activity taxonomy={taxonomy!r} is not one of "
                    f"{ALLOWED_ACTIVITY_TAXONOMIES}: {s!r}"
                )
            if not _ACTIVITY_CODE_RE.match(code):
                raise InvalidUrnError(
                    f"URN activity code={code!r} is not a valid taxonomy "
                    f"code (lowercase a-z 0-9 hyphen dot underscore "
                    f"only): {s!r}"
                )
            return GLUrn(kind="activity", taxonomy=taxonomy, code=code)
        raise InvalidUrnError(
            f"URN activity must have form 'urn:gl:activity:<slug>' or "
            f"'urn:gl:activity:<taxonomy>:<code>': {s!r}"
        )

    if kind == "community":
        slug = body
        if ":" in slug:
            raise InvalidUrnError(
                f"URN community must have no further ':' after slug: {s!r}"
            )
        _check_no_empty_segments((slug,), kind, s)
        _validate_slug(slug, "slug", s)
        return GLUrn(kind="community", slug=slug)

    if kind in ("partner", "enterprise"):
        parts = body.split(":")
        if len(parts) != 3:
            raise InvalidUrnError(
                f"URN {kind!r} must have form "
                f"'urn:gl:{kind}:<tenant>:<pack-id>:v<version>': {s!r}"
            )
        tenant, pack_id, version_seg = parts
        _check_no_empty_segments(parts, kind, s)
        _validate_slug(tenant, "tenant", s)
        _validate_slug(pack_id, "pack_id", s)
        version = _parse_version(version_seg, s)
        return GLUrn(
            kind=kind,  # type: ignore[arg-type]
            tenant=tenant,
            pack_id=pack_id,
            version=version,
        )

    raise InvalidUrnError(  # pragma: no cover
        f"Unhandled URN kind {kind!r}: {s!r}"
    )


_KIND_REQUIRED: Dict[str, Tuple[str, ...]] = {
    "factor": ("source", "namespace", "id", "version"),
    "source": ("slug",),
    "pack": ("source", "pack_id", "version"),
    "methodology": ("slug",),
    "geo": ("geo_type", "id"),
    "unit": ("symbol",),
    # ``activity`` accepts EITHER ``slug`` (legacy) OR (``taxonomy``,
    # ``code``) (Phase 2 WS5). Build() special-cases the validation; this
    # tuple lists the legacy form's fields.
    "activity": ("slug",),
    "community": ("slug",),
    "partner": ("tenant", "pack_id", "version"),
    "enterprise": ("tenant", "pack_id", "version"),
}

# Phase 2 (WS5) — accepted component sets for the activity kind.
_ACTIVITY_BUILD_FORMS: Tuple[Tuple[str, ...], ...] = (
    ("slug",),
    ("taxonomy", "code"),
)


def _coerce_version(value: Any, original_kind: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidUrnError(
            f"URN {original_kind!r} version must be a positive int, got "
            f"{type(value).__name__}={value!r}"
        )
    if value < 1:
        raise InvalidUrnError(
            f"URN {original_kind!r} version must be >= 1, got {value!r}"
        )
    return value


def build(kind: str, **components: Any) -> str:
    """Build a canonical GreenLang URN string from typed components.

    Args:
        kind: One of :data:`ALLOWED_KINDS`.
        **components: Kind-specific components (see module docstring).

    Returns:
        The canonical URN string.

    Raises:
        InvalidUrnError: if ``kind`` is unknown, required components are
            missing, or any component fails validation.
    """
    if kind not in ALLOWED_KINDS:
        raise InvalidUrnError(
            f"Unknown URN kind {kind!r}; expected one of {ALLOWED_KINDS}"
        )

    # ------------------------------------------------------------------
    # Phase 2 (WS5): activity is the only kind that accepts more than one
    # set of required components (``slug`` OR ``(taxonomy, code)``). We
    # resolve the active form from the supplied component keys before
    # the generic missing/extra checks below.
    # ------------------------------------------------------------------
    if kind == "activity":
        provided = {
            k for k, v in components.items() if v not in (None, "")
        }
        # Pick the form whose required keys are entirely present.
        active_form: Optional[Tuple[str, ...]] = None
        for form in _ACTIVITY_BUILD_FORMS:
            if set(form).issubset(provided):
                active_form = form
                break
        if active_form is None:
            raise InvalidUrnError(
                f"URN 'activity' missing required components: must "
                f"supply either 'slug' OR ('taxonomy', 'code') "
                f"(got {sorted(components.keys())})"
            )
        extra = sorted(set(components.keys()) - set(active_form))
        if extra:
            raise InvalidUrnError(
                f"URN 'activity' got unexpected components: {extra} "
                f"(allowed: {list(active_form)})"
            )
        required = active_form
    else:
        required = _KIND_REQUIRED[kind]
        missing = [r for r in required if components.get(r) in (None, "")]
        if missing:
            raise InvalidUrnError(
                f"URN {kind!r} missing required components: {missing} "
                f"(got {sorted(components.keys())})"
            )
        extra = sorted(set(components.keys()) - set(required))
        if extra:
            raise InvalidUrnError(
                f"URN {kind!r} got unexpected components: {extra} "
                f"(allowed: {list(required)})"
            )

    if kind == "factor":
        version = _coerce_version(components["version"], kind)
        candidate = (
            f"urn:gl:factor:{components['source']}:"
            f"{components['namespace']}:{components['id']}:v{version}"
        )
    elif kind == "source":
        candidate = f"urn:gl:source:{components['slug']}"
    elif kind == "pack":
        version = _coerce_version(components["version"], kind)
        candidate = (
            f"urn:gl:pack:{components['source']}:"
            f"{components['pack_id']}:v{version}"
        )
    elif kind == "methodology":
        candidate = f"urn:gl:methodology:{components['slug']}"
    elif kind == "geo":
        candidate = (
            f"urn:gl:geo:{components['geo_type']}:{components['id']}"
        )
    elif kind == "unit":
        candidate = f"urn:gl:unit:{components['symbol']}"
    elif kind == "activity":
        if "taxonomy" in components and "code" in components:
            candidate = (
                f"urn:gl:activity:{components['taxonomy']}:"
                f"{components['code']}"
            )
        else:
            candidate = f"urn:gl:activity:{components['slug']}"
    elif kind == "community":
        candidate = f"urn:gl:community:{components['slug']}"
    elif kind in ("partner", "enterprise"):
        version = _coerce_version(components["version"], kind)
        candidate = (
            f"urn:gl:{kind}:{components['tenant']}:"
            f"{components['pack_id']}:v{version}"
        )
    else:  # pragma: no cover
        raise InvalidUrnError(f"Unhandled URN kind {kind!r}")

    parse(candidate)
    return candidate


def _render(u: GLUrn) -> str:
    """Render a :class:`GLUrn` model back to its canonical string form."""
    if u.kind == "factor":
        return (
            f"urn:gl:factor:{u.source}:{u.namespace}:{u.id}:v{u.version}"
        )
    if u.kind == "source":
        return f"urn:gl:source:{u.slug}"
    if u.kind == "pack":
        return f"urn:gl:pack:{u.source}:{u.pack_id}:v{u.version}"
    if u.kind == "methodology":
        return f"urn:gl:methodology:{u.slug}"
    if u.kind == "geo":
        return f"urn:gl:geo:{u.geo_type}:{u.id}"
    if u.kind == "unit":
        return f"urn:gl:unit:{u.symbol}"
    if u.kind == "activity":
        # Phase 2 (WS5): two render forms.
        if u.taxonomy is not None and u.code is not None:
            return f"urn:gl:activity:{u.taxonomy}:{u.code}"
        return f"urn:gl:activity:{u.slug}"
    if u.kind == "community":
        return f"urn:gl:community:{u.slug}"
    if u.kind in ("partner", "enterprise"):
        return (
            f"urn:gl:{u.kind}:{u.tenant}:{u.pack_id}:v{u.version}"
        )
    raise InvalidUrnError(  # pragma: no cover
        f"Unhandled URN kind on render: {u.kind!r}"
    )


def validate(s: str) -> bool:
    """Return ``True`` if ``s`` is a valid GreenLang URN, else ``False``.

    Never raises.
    """
    try:
        parse(s)
        return True
    except InvalidUrnError:
        return False
    except Exception:  # pragma: no cover
        return False


# Public alias used by the v0.1 Alpha SDK surface — validates that the
# argument is a well-formed GreenLang URN. Raises ``ValueError`` (via
# :class:`InvalidUrnError`) on malformed input. This is the entry point
# the SDK calls before issuing a network request that uses URN as the
# canonical primary id.
def validate_urn(s: str) -> str:
    """Validate ``s`` is a well-formed GreenLang URN and return it unchanged.

    Args:
        s: Candidate URN string.

    Returns:
        The same string ``s`` when valid (allows fluent use in call sites).

    Raises:
        ValueError: when ``s`` is not a well-formed GreenLang URN. The
            raised exception is :class:`InvalidUrnError`, which subclasses
            :class:`ValueError` so callers can ``except ValueError`` cleanly.
    """
    parse(s)  # raises InvalidUrnError (a ValueError) on failure.
    return s


_LEGACY_EF_RE = re.compile(r"^EF:[A-Za-z0-9_.:\-]+$")


def coerce_factor_id_to_urn(
    factor_id: str,
    *,
    source: str,
    namespace: str,
    version: int = 1,
) -> str:
    """Lift a legacy ``EF:...`` factor id into a canonical factor URN.

    The ``EF:...`` id is treated as an opaque alias; only the trailing
    leaf token after the ``EF:`` prefix is reused to derive the URN
    ``<id>`` component. ``source``, ``namespace`` and ``version`` are
    supplied by the caller.

    Mapping rules:
        1. Strip the ``EF:`` prefix.
        2. Drop trailing ``vNN`` legacy version suffixes.
        3. Lowercase the last surviving segment and replace any
           disallowed characters with ``-``.
        4. Round-trip through :func:`parse` to guarantee well-formedness.

    Args:
        factor_id: Legacy ``EF:...`` id (must match
            ``^EF:[A-Za-z0-9_.:-]+$``).
        source: URN source slug (e.g. ``"epa-egrid"``).
        namespace: URN namespace (e.g. ``"subregion-serc"``).
        version: Positive integer URN version. Defaults to 1.

    Returns:
        The canonical factor URN string.

    Raises:
        InvalidUrnError: if any input is malformed.

    Example:
        >>> coerce_factor_id_to_urn(
        ...     "EF:US:grid:eGRID-SERC:2024:v1",
        ...     source="epa-egrid",
        ...     namespace="subregion-serc",
        ...     version=1,
        ... )
        'urn:gl:factor:epa-egrid:subregion-serc:2024:v1'
    """
    if not isinstance(factor_id, str) or not _LEGACY_EF_RE.match(factor_id):
        raise InvalidUrnError(
            f"factor_id must match '^EF:[A-Za-z0-9_.:-]+$', got "
            f"{factor_id!r}"
        )
    body = factor_id[len("EF:"):]
    segments = [seg for seg in body.split(":") if seg]
    if not segments:
        raise InvalidUrnError(
            f"factor_id {factor_id!r} has no segments after 'EF:'"
        )
    while segments and _VERSION_RE.match(segments[-1].lower()):
        segments.pop()
    if not segments:
        raise InvalidUrnError(
            f"factor_id {factor_id!r} has no non-version segments"
        )
    leaf = segments[-1].lower()
    leaf = re.sub(r"[^a-z0-9._\-]", "-", leaf)
    leaf = re.sub(r"-{2,}", "-", leaf).strip("-")
    if not leaf:
        raise InvalidUrnError(
            f"factor_id {factor_id!r} produced empty URN id after "
            f"normalisation"
        )
    return build(
        "factor",
        source=source,
        namespace=namespace,
        id=leaf,
        version=version,
    )
