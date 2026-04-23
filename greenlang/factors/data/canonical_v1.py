# -*- coding: utf-8 -*-
"""GreenLang Factor Record v1 — the canonical frozen model (W4-A).

This module implements the 10 migrations enumerated in
``docs/specs/schema_v1_gap_report.md``:

* M01  — strict factor_id regex.
* M02  — 15-value ``factor_family`` enum.
* M03  — 14-value ``method_profile`` enum (expanded).
* M04  — ``source_version`` consolidated top-level field.
* M05  — flat ``geography`` → ``jurisdiction{country,region,grid_region}``.
* M06  — ``valid_to`` required, 9999-12-31 sentinel.
* M07  — ``activity_schema{category,sub_category,classification_codes[]}``.
* M08  — ``numerator{co2,ch4,n2o,f_gases{...},biogenic_co2,co2e,unit}``.
* M09  — ``denominator{value,unit}``.
* M10  — ``gwp_set`` enum with Kyoto_SAR_100 rename.
* M11  — ``formula_type`` enum.
* M12  — DQS rescale (1-5 → 0-100 composite).
* M13  — :class:`Lineage` consolidation.
* M14/M15/M16/T02 — Licensing consolidation + redistribution_class enum.
* M17  — Explainability: audit_text + replacement_pointer.
* M19  — 7 per-family parameter models (see :mod:`categorical_parameters`).
* M20  — GWP coefficients externalised (see :mod:`gwp_registry`).

All CTO-reversible enum choices are surfaced as **module-level tuple constants**
at the top of this file. Changing any of them is a 1-line edit per the
W4-A brief. CTO decision references live in ``docs/specs/enum_decisions_v1.md``.
"""
from __future__ import annotations

import re
import warnings
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field, validator as field_validator  # type: ignore

    ConfigDict = dict  # type: ignore
    _PYDANTIC_V2 = False

from greenlang.factors.data import gwp_registry
from greenlang.factors.data.categorical_parameters import (
    ParametersUnion,
    dump_parameters,
    parse_parameters,
)


# ---------------------------------------------------------------------------
# CTO-reversible enum constants (see docs/specs/enum_decisions_v1.md)
# ---------------------------------------------------------------------------
# Change any value here to override; a future CTO sign-off is a 1-line edit.

STATUS_ENUM: Tuple[str, ...] = (
    "draft",
    "preview",
    "certified",
    "connector_only",
    "deprecated",
    "superseded",
    "retired",
)
"""Factor-record lifecycle states (W4-A brief, 7 values)."""

REDISTRIBUTION_CLASS_ENUM: Tuple[str, ...] = (
    "open",
    "licensed_embedded",
    "customer_private",
    "oem_redistributable",
)
"""Record-level redistribution classes (W4-A brief, 4 values)."""

GWP_SET_ENUM: Tuple[str, ...] = (
    "IPCC_AR4_100",
    "IPCC_AR5_100",
    "IPCC_AR6_100",
    "IPCC_AR5_20",
    "IPCC_AR6_20",
    "Kyoto_SAR_100",
)
"""Supported GWP reference sets (W4-A brief, 6 values)."""

DEFAULT_GWP_SET: str = "IPCC_AR6_100"
"""Default GWP set for new factor records."""

METHOD_PROFILE_ENUM: Tuple[str, ...] = (
    "corporate_scope1",
    "corporate_scope2_location_based",
    "corporate_scope2_market_based",
    "corporate_scope3_upstream",
    "corporate_scope3_downstream",
    "product_carbon_iso_14067",
    "product_carbon_ghgp",
    "product_carbon_pact",
    "freight_iso_14083",
    "freight_glec",
    "land_removals_lsr",
    "finance_proxy_pcaf",
    "eu_cbam",
    "eu_dpp",
)
"""14-value MethodProfile enum (W4-A brief)."""

FACTOR_FAMILY_ENUM: Tuple[str, ...] = (
    "combustion",
    "electricity",
    "transport",
    "materials_products",
    "refrigerants",
    "land_removals",
    "finance_proxies",
    "waste",
    "energy_conversion",
    "carbon_content",
    "oxidation",
    "heating_value",
    "density",
    "residual_mix",
    "classification_mapping",
)

FORMULA_TYPE_ENUM: Tuple[str, ...] = (
    "direct_factor",
    "combustion",
    "lca",
    "spend_proxy",
    "transport_chain",
    "residual_mix",
    "carbon_budget",
    "custom",
)

# Map legacy status / redistribution / method-profile values -> canonical v1.
_LEGACY_STATUS_MAP: Dict[str, str] = {
    "active": "certified",          # frozen schema -> brief
    "under_review": "preview",       # frozen schema -> brief
    "private": "certified",          # superseded by redistribution_class
}

_LEGACY_REDISTRIBUTION_MAP: Dict[str, str] = {
    "licensed": "licensed_embedded",
    "restricted": "licensed_embedded",
    "oem": "oem_redistributable",
}

_LEGACY_METHOD_PROFILE_MAP: Dict[str, str] = {
    "corporate_scope3": "corporate_scope3_upstream",
    "product_carbon": "product_carbon_ghgp",
    "freight_iso_14083_wtw": "freight_iso_14083",
    "freight_iso_14083_ttw": "freight_iso_14083",
    "freight_iso14083_glec_wtw": "freight_iso_14083",
    "freight_iso14083_glec_ttw": "freight_iso_14083",
    "freight_glec_wtw": "freight_glec",
    "freight_glec_ttw": "freight_glec",
    "land_removals": "land_removals_lsr",
    "land_removals_ghgp_lsr": "land_removals_lsr",
    "finance_proxy": "finance_proxy_pcaf",
    "finance_proxies_pcaf": "finance_proxy_pcaf",
    "india_ccts": "eu_cbam",  # placeholder; India-CCTS handled at pack layer
    "eu_dpp_battery": "eu_dpp",
    "eu_dpp_textile": "eu_dpp",
    "product_iso14067": "product_carbon_iso_14067",
    "product_pact": "product_carbon_pact",
}

FACTOR_ID_REGEX = re.compile(r"^EF:[A-Za-z0-9_.:-]+$")

_DEPRECATION_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Emit a DeprecationWarning once per (process, key)."""
    if key in _DEPRECATION_WARNED:
        return
    _DEPRECATION_WARNED.add(key)
    warnings.warn(message, DeprecationWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# DQS helpers (M12 / T04)
# ---------------------------------------------------------------------------


def compute_fqs(
    temporal: int,
    geographic: int,
    technology: int,
    verification: int,
    completeness: int,
) -> float:
    """Compute the 0-100 composite FQS from five 1-5 dimension scores.

    Weights per ``factor_record_v1.schema.json``:

    =============  =======
    Dimension       Weight
    =============  =======
    temporal         0.25
    geographic       0.25
    technology       0.20
    verification     0.15
    completeness     0.15
    =============  =======

    Scaled to 0-100 by multiplying by 20 (max score = 5 × 20 = 100).
    Result is rounded to 2 decimals to stay inside the ±0.5 tolerance the
    schema prescribes without accumulating float drift.
    """
    for name, value in (
        ("temporal", temporal),
        ("geographic", geographic),
        ("technology", technology),
        ("verification", verification),
        ("completeness", completeness),
    ):
        if not (1 <= int(value) <= 5):
            raise ValueError(f"{name} score must be 1-5, got {value}")

    raw = (
        0.25 * temporal
        + 0.25 * geographic
        + 0.20 * technology
        + 0.15 * verification
        + 0.15 * completeness
    )
    return round(20.0 * raw, 2)


# ---------------------------------------------------------------------------
# Sub-objects
# ---------------------------------------------------------------------------


class JurisdictionV1(BaseModel):
    """Flattened jurisdiction (M05)."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    country: str = Field(min_length=2, max_length=2, pattern=r"^[A-Z]{2}$")
    region: Optional[str] = Field(default=None, max_length=64)
    grid_region: Optional[str] = Field(default=None, max_length=64)


class ActivitySchemaV1(BaseModel):
    """Activity-schema restructure (M07)."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1, max_length=128)
    sub_category: Optional[str] = Field(default=None, max_length=128)
    classification_codes: List[str] = Field(default_factory=list)

    if _PYDANTIC_V2:

        @field_validator("classification_codes")
        @classmethod
        def _validate_codes(cls, v: List[str]) -> List[str]:
            pattern = re.compile(r"^[A-Z][A-Z0-9_-]*:[A-Za-z0-9._-]+$")
            for code in v:
                if not pattern.match(code):
                    raise ValueError(
                        f"classification_code {code!r} must match SCHEME:CODE"
                    )
            return v


class NumeratorV1(BaseModel):
    """Numerator shape (M08) — HFCs/PFCs/SF6/NF3 under f_gases dict.

    Negative values ARE permitted here because the ``land_removals`` factor
    family carries sequestration (negative-emission) records. The frozen
    JSON Schema uses ``minimum: 0`` on the record-level emission families;
    regulatory CI will enforce that at serialisation time, not here.
    """

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    co2: Optional[float] = None
    ch4: Optional[float] = None
    n2o: Optional[float] = None
    co2e: Optional[float] = None
    f_gases: Dict[str, float] = Field(default_factory=dict)
    biogenic_co2: Optional[float] = None
    unit: str = Field(default="kg")

    if _PYDANTIC_V2:

        @field_validator("unit")
        @classmethod
        def _validate_unit(cls, v: str) -> str:
            if v not in {"kg", "g", "t", "lb"}:
                raise ValueError(f"numerator.unit must be kg/g/t/lb, got {v}")
            return v


class DenominatorV1(BaseModel):
    """Denominator shape (M09)."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    value: float = Field(default=1.0, gt=0.0)
    unit: str = Field(min_length=1, max_length=32)


class QualityV1(BaseModel):
    """Rescaled data-quality block (M12 / T04)."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    temporal_score: int = Field(ge=1, le=5)
    geographic_score: int = Field(ge=1, le=5)
    technology_score: int = Field(ge=1, le=5)
    verification_score: int = Field(ge=1, le=5)
    completeness_score: int = Field(ge=1, le=5)
    composite_fqs: float = Field(ge=0.0, le=100.0)

    @classmethod
    def from_dimensions(
        cls,
        temporal: int,
        geographic: int,
        technology: int,
        verification: int,
        completeness: int,
    ) -> "QualityV1":
        composite = compute_fqs(
            temporal, geographic, technology, verification, completeness
        )
        return cls(
            temporal_score=temporal,
            geographic_score=geographic,
            technology_score=technology,
            verification_score=verification,
            completeness_score=completeness,
            composite_fqs=composite,
        )


class RawRecordRefV1(BaseModel):
    """Pointer to the pre-normalisation raw source record."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    raw_record_id: str
    raw_payload_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_format: Literal["csv", "xml", "json", "pdf_ocr", "yaml", "xlsx", "api"]
    storage_uri: Optional[str] = None


class LineageV1(BaseModel):
    """Consolidated lineage sub-object (M13)."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    ingested_at: datetime
    ingested_by: str = Field(min_length=1, max_length=256)
    approved_by: Optional[str] = Field(default=None, max_length=256)
    approved_at: Optional[datetime] = None
    change_reason: str = Field(default="Initial ingest", min_length=1, max_length=2048)
    previous_factor_version: Optional[str] = None
    raw_record_ref: Optional[RawRecordRefV1] = None


class LicensingV1(BaseModel):
    """Consolidated licensing block (M14/M15/M16/T02)."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    redistribution_class: str
    customer_entitlement_required: bool
    license_name: Optional[str] = Field(default=None, max_length=128)
    license_url: Optional[str] = None
    attribution_required: bool = False
    attribution_text: Optional[str] = Field(default=None, max_length=1024)
    restrictions: List[str] = Field(default_factory=list)

    if _PYDANTIC_V2:

        @field_validator("redistribution_class")
        @classmethod
        def _validate_class(cls, v: str) -> str:
            if v not in REDISTRIBUTION_CLASS_ENUM:
                raise ValueError(
                    f"redistribution_class {v!r} not in {REDISTRIBUTION_CLASS_ENUM}"
                )
            return v

    # Legacy boolean aliases (read-only). Derived from redistribution_class.
    @property
    def redistribution_allowed(self) -> bool:
        return self.redistribution_class in {"open", "oem_redistributable"}

    @property
    def commercial_use_allowed(self) -> bool:
        return self.redistribution_class in {
            "open",
            "licensed_embedded",
            "oem_redistributable",
        }


class ExplainabilityV1(BaseModel):
    """Explainability block (M17)."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

    assumptions: List[str] = Field(default_factory=list)
    fallback_rank: int = Field(default=7, ge=1, le=7)
    audit_text: Optional[str] = Field(default=None, max_length=2048)
    replacement_pointer: Optional[str] = None
    # Legacy alias (input-compat)
    rationale: Optional[str] = Field(default=None, max_length=2048)

    if _PYDANTIC_V2:

        @field_validator("audit_text", mode="before")
        @classmethod
        def _rationale_alias(cls, v, info):
            if v is not None:
                return v
            values = getattr(info, "data", {})
            return values.get("rationale")


# ---------------------------------------------------------------------------
# Main record
# ---------------------------------------------------------------------------


class FactorRecordV1(BaseModel):
    """Canonical frozen factor record (v1).

    See ``config/schemas/factor_record_v1.schema.json`` for the spec-of-record.
    """

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid", populate_by_name=True)

    factor_id: str
    factor_family: str
    factor_name: str = Field(min_length=1, max_length=512)
    method_profile: str
    source_id: str = Field(min_length=1, max_length=256)
    source_version: str = Field(min_length=1, max_length=64)
    factor_version: str
    status: str
    jurisdiction: JurisdictionV1
    valid_from: date
    valid_to: date
    activity_schema: ActivitySchemaV1
    numerator: NumeratorV1
    denominator: DenominatorV1
    gwp_set: str
    formula_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    quality: QualityV1
    lineage: LineageV1
    licensing: LicensingV1
    explainability: ExplainabilityV1

    # Non-schema fields — kept on the model for in-memory convenience and
    # excluded from the serialised payload.
    connector_only_flag: bool = Field(default=False, exclude=True)

    if _PYDANTIC_V2:

        @field_validator("factor_id")
        @classmethod
        def _validate_factor_id(cls, v: str) -> str:
            if not FACTOR_ID_REGEX.match(v):
                raise ValueError(
                    f"factor_id {v!r} must match {FACTOR_ID_REGEX.pattern}"
                )
            return v

        @field_validator("factor_family")
        @classmethod
        def _validate_factor_family(cls, v: str) -> str:
            if v not in FACTOR_FAMILY_ENUM:
                raise ValueError(
                    f"factor_family {v!r} not in {FACTOR_FAMILY_ENUM}"
                )
            return v

        @field_validator("status")
        @classmethod
        def _validate_status(cls, v: str) -> str:
            if v not in STATUS_ENUM:
                raise ValueError(f"status {v!r} not in {STATUS_ENUM}")
            return v

        @field_validator("method_profile")
        @classmethod
        def _validate_method_profile(cls, v: str) -> str:
            if v not in METHOD_PROFILE_ENUM:
                raise ValueError(
                    f"method_profile {v!r} not in {METHOD_PROFILE_ENUM}"
                )
            return v

        @field_validator("formula_type")
        @classmethod
        def _validate_formula_type(cls, v: str) -> str:
            if v not in FORMULA_TYPE_ENUM:
                raise ValueError(f"formula_type {v!r} not in {FORMULA_TYPE_ENUM}")
            return v

        @field_validator("gwp_set")
        @classmethod
        def _validate_gwp_set(cls, v: str) -> str:
            if v not in GWP_SET_ENUM:
                raise ValueError(f"gwp_set {v!r} not in {GWP_SET_ENUM}")
            return v

        @field_validator("valid_to")
        @classmethod
        def _validate_valid_to(cls, v: date, info) -> date:
            valid_from = info.data.get("valid_from")
            if valid_from is not None and v <= valid_from:
                raise ValueError(
                    f"valid_to ({v}) must be strictly greater than "
                    f"valid_from ({valid_from})"
                )
            return v

        @field_validator("factor_version", mode="before")
        @classmethod
        def _validate_factor_version(cls, v: Any) -> str:
            if v is None:
                return "1.0.0"
            s = str(v).strip()
            # Normalise legacy short versions:  "v1" -> "1.0.0"
            #                                   "v2.0" -> "2.0.0"
            #                                   "1.2" -> "1.2.0"
            m = re.match(r"^v?(\d+)(?:\.(\d+))?(?:\.(\d+))?([-+].*)?$", s)
            if m:
                major = m.group(1) or "0"
                minor = m.group(2) or "0"
                patch = m.group(3) or "0"
                suffix = m.group(4) or ""
                return f"{major}.{minor}.{patch}{suffix}"
            # Fallback: attempt strict semver check.
            if not re.match(
                r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
                r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$",
                s,
            ):
                _warn_once(
                    f"factor_version:unrecognised:{s}",
                    f"factor_version {s!r} not semver-like; coercing to 1.0.0."
                )
                return "1.0.0"
            return s

    def parsed_parameters(self) -> ParametersUnion:
        """Return ``self.parameters`` parsed into the family-specific model."""
        return parse_parameters(self.factor_family, self.parameters)

    def compute_co2e(self, gwp_set: Optional[str] = None) -> Decimal:
        """Derive CO2e from numerator gas masses + gwp_registry (N1)."""
        set_name = gwp_set or self.gwp_set
        gases: Dict[str, float] = {}
        if self.numerator.co2 is not None:
            gases["CO2"] = self.numerator.co2
        if self.numerator.ch4 is not None:
            gases["CH4"] = self.numerator.ch4
        if self.numerator.n2o is not None:
            gases["N2O"] = self.numerator.n2o
        return gwp_registry.co2e(gases, set_name, f_gases=self.numerator.f_gases)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-Schema-compatible dict."""
        if _PYDANTIC_V2:
            payload = self.model_dump(mode="json", exclude_none=False)
        else:  # pragma: no cover
            payload = self.dict()
        payload.pop("connector_only_flag", None)
        return payload


# ---------------------------------------------------------------------------
# Back-compat readers (old-shape input → new-shape output)
# ---------------------------------------------------------------------------


_VALID_TO_SENTINEL = date(9999, 12, 31)


def _map_status(raw: Any) -> str:
    if raw is None or raw == "":
        return "preview"
    v = str(raw)
    if v in STATUS_ENUM:
        return v
    mapped = _LEGACY_STATUS_MAP.get(v)
    if mapped is not None:
        _warn_once(
            f"status:{v}",
            f"factor status {v!r} is a legacy value; mapped to {mapped!r}."
        )
        return mapped
    _warn_once(
        f"status:unknown:{v}",
        f"unknown status {v!r}; falling back to 'preview'."
    )
    return "preview"


def _map_method_profile(raw: Any) -> str:
    if raw is None:
        return "corporate_scope1"
    v = str(raw)
    if v in METHOD_PROFILE_ENUM:
        return v
    mapped = _LEGACY_METHOD_PROFILE_MAP.get(v)
    if mapped is not None:
        _warn_once(
            f"method_profile:{v}",
            f"method_profile {v!r} is a legacy value; mapped to {mapped!r}."
        )
        return mapped
    _warn_once(
        f"method_profile:unknown:{v}",
        f"unknown method_profile {v!r}; defaulting to corporate_scope1."
    )
    return "corporate_scope1"


def _map_redistribution_class(raw: Any, legacy_flags: Dict[str, Any]) -> str:
    if raw is not None:
        v = str(raw)
        if v in REDISTRIBUTION_CLASS_ENUM:
            return v
        mapped = _LEGACY_REDISTRIBUTION_MAP.get(v)
        if mapped is not None:
            _warn_once(
                f"redistribution_class:{v}",
                f"redistribution_class {v!r} is legacy; mapped to {mapped!r}."
            )
            return mapped
    # Derive from boolean flags (legacy shape)
    redist = legacy_flags.get("redistribution_allowed")
    commercial = legacy_flags.get("commercial_use_allowed")
    if redist is True and commercial is True:
        return "open"
    if redist is False and commercial is True:
        return "licensed_embedded"
    if redist is False and commercial is False:
        return "customer_private"
    # Unknown: default to licensed_embedded (conservative).
    return "licensed_embedded"


def _map_gwp_set(raw: Any) -> str:
    if raw is None:
        return DEFAULT_GWP_SET
    v = str(raw)
    if v in GWP_SET_ENUM:
        return v
    try:
        return gwp_registry.normalize_gwp_set(v)
    except ValueError:
        _warn_once(
            f"gwp_set:unknown:{v}",
            f"unknown gwp_set {v!r}; falling back to {DEFAULT_GWP_SET}."
        )
        return DEFAULT_GWP_SET


def _map_factor_family(raw: Any, fuel_type: Optional[str], unit: Optional[str]) -> str:
    if raw and raw in FACTOR_FAMILY_ENUM:
        return raw
    # Derive from taxonomy (best-effort).
    # Electricity heuristic
    if fuel_type and "elec" in str(fuel_type).lower():
        return "electricity"
    if unit and str(unit).lower() in {"kwh", "mwh", "gwh"}:
        return "electricity"
    # Default: combustion (the dominant family).
    return "combustion"


def _map_formula_type(raw: Any, factor_family: str) -> str:
    if raw and raw in FORMULA_TYPE_ENUM:
        return raw
    default_per_family = {
        "combustion": "combustion",
        "electricity": "direct_factor",
        "transport": "transport_chain",
        "materials_products": "lca",
        "refrigerants": "direct_factor",
        "land_removals": "carbon_budget",
        "finance_proxies": "spend_proxy",
        "waste": "direct_factor",
    }
    return default_per_family.get(factor_family, "direct_factor")


def _map_jurisdiction(data: Dict[str, Any]) -> JurisdictionV1:
    if isinstance(data.get("jurisdiction"), dict):
        jdata = data["jurisdiction"]
        return JurisdictionV1(
            country=str(jdata.get("country", "XX"))[:2].upper() or "XX",
            region=jdata.get("region"),
            grid_region=jdata.get("grid_region"),
        )
    geography = data.get("geography") or "XX"
    country = str(geography).strip().upper()
    # Map common non-ISO names to ISO-3166-1 alpha-2.
    _ISO_ALIASES = {
        "UK": "GB",
        "USA": "US",
        "IND": "IN",
        "DEU": "DE",
        "FRA": "FR",
        "EUROPE": "XX",
        "EU": "XX",
        "GLOBAL": "XX",
    }
    if country in _ISO_ALIASES:
        country = _ISO_ALIASES[country]
    elif len(country) != 2:
        country = "XX"
    region = data.get("region_hint")
    grid_region = None
    # electricity heuristic
    if data.get("subregion_code"):
        grid_region = data["subregion_code"]
    return JurisdictionV1(country=country, region=region, grid_region=grid_region)


def _map_activity_schema(data: Dict[str, Any]) -> ActivitySchemaV1:
    if isinstance(data.get("activity_schema"), dict):
        raw = data["activity_schema"]
        return ActivitySchemaV1(
            category=raw.get("category") or data.get("fuel_type") or "unknown",
            sub_category=raw.get("sub_category"),
            classification_codes=list(raw.get("classification_codes") or []),
        )
    fuel = data.get("fuel_type") or data.get("activity") or "unknown"
    tags = data.get("activity_tags") or []
    sector_tags = data.get("sector_tags") or []
    codes: List[str] = []
    for t in sector_tags:
        if ":" in t and re.match(r"^[A-Z]", t):
            codes.append(t)
    return ActivitySchemaV1(
        category=str(fuel),
        sub_category=tags[0] if tags else None,
        classification_codes=codes,
    )


def _map_numerator(data: Dict[str, Any]) -> NumeratorV1:
    if isinstance(data.get("numerator"), dict):
        return NumeratorV1(**data["numerator"])
    vec = data.get("vectors") or {}
    f_gases: Dict[str, float] = {}
    for legacy_key in ("HFCs", "PFCs", "SF6", "NF3"):
        val = vec.get(legacy_key)
        if val is not None and float(val) > 0:
            f_gases[legacy_key] = float(val)
    return NumeratorV1(
        co2=vec.get("CO2"),
        ch4=vec.get("CH4"),
        n2o=vec.get("N2O"),
        biogenic_co2=vec.get("biogenic_CO2"),
        f_gases=f_gases,
        unit="kg",
    )


def _map_denominator(data: Dict[str, Any]) -> DenominatorV1:
    if isinstance(data.get("denominator"), dict):
        raw = data["denominator"]
        return DenominatorV1(
            value=float(raw.get("value", 1.0)),
            unit=str(raw.get("unit") or data.get("unit") or "unit"),
        )
    return DenominatorV1(value=1.0, unit=str(data.get("unit") or "unit"))


def _map_quality(data: Dict[str, Any]) -> QualityV1:
    if isinstance(data.get("quality"), dict):
        q = data["quality"]
        if "composite_fqs" in q:
            return QualityV1(**q)
    dqs = data.get("dqs") or {}
    temporal = int(dqs.get("temporal", 3))
    # Rename legacy dimension keys -> v1 names.
    geographic = int(dqs.get("geographic", dqs.get("geographical", 3)))
    technology = int(dqs.get("technology", dqs.get("technological", 3)))
    verification = int(dqs.get("verification", dqs.get("methodological", 3)))
    completeness = int(dqs.get("completeness", dqs.get("representativeness", 3)))
    return QualityV1.from_dimensions(
        temporal=temporal,
        geographic=geographic,
        technology=technology,
        verification=verification,
        completeness=completeness,
    )


def _map_lineage(data: Dict[str, Any]) -> LineageV1:
    if isinstance(data.get("lineage"), dict):
        raw = dict(data["lineage"])
        raw_rref = raw.pop("raw_record_ref", None)
        if raw_rref is not None and not isinstance(raw_rref, RawRecordRefV1):
            raw_rref = RawRecordRefV1(**raw_rref) if isinstance(raw_rref, dict) else None
        return LineageV1(raw_record_ref=raw_rref, **raw)
    # Build from legacy fields.
    ingested_at = data.get("ingested_at") or data.get("created_at") or datetime.now(
        timezone.utc
    )
    if isinstance(ingested_at, str):
        ingested_at = datetime.fromisoformat(ingested_at.replace("Z", "+00:00"))
    # Ensure tz-aware
    if isinstance(ingested_at, datetime) and ingested_at.tzinfo is None:
        ingested_at = ingested_at.replace(tzinfo=timezone.utc)
    ingested_by = (
        data.get("ingested_by")
        or data.get("created_by")
        or "greenlang_system"
    )
    change_log = data.get("change_log") or []
    change_reason = "Initial ingest"
    previous_version = None
    if change_log:
        last = change_log[-1]
        if isinstance(last, dict):
            change_reason = last.get("change_reason", change_reason)
            previous_version = last.get("previous_factor_version")
    return LineageV1(
        ingested_at=ingested_at,
        ingested_by=str(ingested_by),
        change_reason=change_reason,
        previous_factor_version=previous_version,
    )


def _map_licensing(data: Dict[str, Any]) -> LicensingV1:
    if isinstance(data.get("licensing"), dict):
        raw = data["licensing"]
        rclass = _map_redistribution_class(
            raw.get("redistribution_class"), raw
        )
        return LicensingV1(
            redistribution_class=rclass,
            customer_entitlement_required=bool(
                raw.get(
                    "customer_entitlement_required",
                    rclass in {"licensed_embedded", "customer_private"},
                )
            ),
            license_name=raw.get("license_name"),
            license_url=raw.get("license_url"),
            attribution_required=bool(raw.get("attribution_required", False)),
            attribution_text=raw.get("attribution_text"),
            restrictions=list(raw.get("restrictions") or []),
        )
    legacy = data.get("license_info") or {}
    rclass = _map_redistribution_class(
        data.get("redistribution_class"), legacy
    )
    return LicensingV1(
        redistribution_class=rclass,
        customer_entitlement_required=rclass in {"licensed_embedded", "customer_private"},
        license_name=legacy.get("license"),
        license_url=legacy.get("license_url"),
        attribution_required=bool(legacy.get("attribution_required", False)),
        attribution_text=data.get("attribution_text"),
        restrictions=list(legacy.get("restrictions") or []),
    )


def _map_explainability(data: Dict[str, Any]) -> ExplainabilityV1:
    if isinstance(data.get("explainability"), dict):
        raw = data["explainability"]
        return ExplainabilityV1(
            assumptions=list(raw.get("assumptions") or []),
            fallback_rank=int(raw.get("fallback_rank", 7)),
            audit_text=raw.get("audit_text") or raw.get("rationale"),
            replacement_pointer=raw.get("replacement_pointer"),
            rationale=raw.get("rationale"),
        )
    return ExplainabilityV1(
        assumptions=[],
        fallback_rank=7,
    )


def _map_parameters(
    data: Dict[str, Any], factor_family: str
) -> Dict[str, Any]:
    """Project legacy combustion-ish flags into a canonical parameters dict."""
    raw = data.get("parameters")
    if isinstance(raw, dict):
        try:
            parsed = parse_parameters(factor_family, raw)
            return dump_parameters(parsed)
        except Exception:
            # Fall through to family-specific derivation.
            pass
    # Derive based on family.
    if factor_family == "combustion":
        params = {
            "fuel_code": str(data.get("fuel_type") or "unknown").lower(),
            "moisture_share": _pct_to_share(data.get("moisture_content_pct")),
            "ash_share": _pct_to_share(data.get("ash_content_pct")),
            "biogenic_carbon_share": 1.0 if data.get("biogenic_flag") else None,
        }
        return dump_parameters(parse_parameters("combustion", params))
    if factor_family == "electricity":
        params = {
            "electricity_basis": "location",
        }
        return dump_parameters(parse_parameters("electricity", params))
    # Everything else: empty parameters (generic bucket).
    return {}


def _pct_to_share(value: Any) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    return v / 100.0 if v > 1.0 else v


def _map_valid_dates(data: Dict[str, Any]) -> Tuple[date, date]:
    vf = data.get("valid_from")
    if isinstance(vf, str):
        vf = date.fromisoformat(vf)
    if vf is None:
        vf = date(2020, 1, 1)
    vt = data.get("valid_to")
    if isinstance(vt, str):
        vt = date.fromisoformat(vt)
    if vt is None:
        vt = _VALID_TO_SENTINEL
    if vt <= vf:
        vt = _VALID_TO_SENTINEL
    return vf, vt


def _resolve_source_version(data: Dict[str, Any]) -> str:
    # M04: first non-null of source_release / release_version / provenance.version.
    candidates = [
        data.get("source_version"),
        data.get("source_release"),
        data.get("release_version"),
    ]
    prov = data.get("provenance") or {}
    if isinstance(prov, dict):
        candidates.append(prov.get("version"))
    vals = [c for c in candidates if c]
    if not vals:
        return "unspecified"
    if len(set(vals)) > 1:
        _warn_once(
            f"source_version_disagreement:{vals[0]}",
            f"multiple source_version values found: {vals}; using {vals[0]!r}.",
        )
    return str(vals[0])


def _derive_factor_name(data: Dict[str, Any]) -> str:
    name = data.get("factor_name")
    if name:
        return str(name)[:512]
    bits = [
        data.get("fuel_type") or data.get("activity"),
        data.get("unit"),
        data.get("geography"),
    ]
    return " | ".join(str(b) for b in bits if b) or data.get("factor_id", "factor")


# ---------------------------------------------------------------------------
# Public conversion surface
# ---------------------------------------------------------------------------


def from_legacy_dict(data: Dict[str, Any]) -> FactorRecordV1:
    """Construct a :class:`FactorRecordV1` from any old-shape dict.

    This is the bidirectional back-compat reader used by
    :mod:`catalog_repository` and the migration script.  All numeric values
    are preserved bit-identically; only field *shapes* / *names* change.
    """
    data = dict(data or {})
    factor_family = _map_factor_family(
        data.get("factor_family"), data.get("fuel_type"), data.get("unit")
    )
    valid_from, valid_to = _map_valid_dates(data)
    return FactorRecordV1(
        factor_id=data["factor_id"],
        factor_family=factor_family,
        factor_name=_derive_factor_name(data),
        method_profile=_map_method_profile(data.get("method_profile")),
        source_id=str(data.get("source_id") or "unknown_source"),
        source_version=_resolve_source_version(data),
        factor_version=str(
            data.get("factor_version")
            or data.get("release_version")
            or "1.0.0"
        ),
        status=_map_status(data.get("factor_status") or data.get("status")),
        jurisdiction=_map_jurisdiction(data),
        valid_from=valid_from,
        valid_to=valid_to,
        activity_schema=_map_activity_schema(data),
        numerator=_map_numerator(data),
        denominator=_map_denominator(data),
        gwp_set=_map_gwp_set(
            data.get("gwp_set")
            or (data.get("gwp_100yr") or {}).get("gwp_set")
        ),
        formula_type=_map_formula_type(data.get("formula_type"), factor_family),
        parameters=_map_parameters(data, factor_family),
        quality=_map_quality(data),
        lineage=_map_lineage(data),
        licensing=_map_licensing(data),
        explainability=_map_explainability(data),
        connector_only_flag=(
            data.get("connector_only") is True
            or str(data.get("factor_status") or "").lower() == "connector_only"
        ),
    )


def to_legacy_dict(record: FactorRecordV1) -> Dict[str, Any]:
    """Project a v1 record back into the legacy EmissionFactorRecord shape.

    Used by :mod:`catalog_repository` to keep downstream agents (which still
    consume the legacy dataclass) working while the migration is staged.
    """
    vectors = {}
    if record.numerator.co2 is not None:
        vectors["CO2"] = record.numerator.co2
    if record.numerator.ch4 is not None:
        vectors["CH4"] = record.numerator.ch4
    if record.numerator.n2o is not None:
        vectors["N2O"] = record.numerator.n2o
    for gas_key, field_name in (
        ("HFCs", "HFCs"),
        ("PFCs", "PFCs"),
        ("SF6", "SF6"),
        ("NF3", "NF3"),
    ):
        if gas_key in record.numerator.f_gases:
            vectors[field_name] = record.numerator.f_gases[gas_key]
    if record.numerator.biogenic_co2 is not None:
        vectors["biogenic_CO2"] = record.numerator.biogenic_co2

    # GWPs from registry (for legacy callers).
    ch4_gwp = float(gwp_registry.lookup(record.gwp_set, "CH4"))
    n2o_gwp = float(gwp_registry.lookup(record.gwp_set, "N2O"))

    return {
        "factor_id": record.factor_id,
        "fuel_type": str(record.activity_schema.category),
        "unit": record.denominator.unit,
        "geography": record.jurisdiction.country,
        "geography_level": "country",
        "region_hint": record.jurisdiction.region,
        "vectors": vectors,
        "gwp_100yr": {
            "gwp_set": record.gwp_set,
            "CH4_gwp": ch4_gwp,
            "N2O_gwp": n2o_gwp,
        },
        "scope": "1",  # legacy — not faithfully rehydrated
        "boundary": "combustion",
        "provenance": {
            "source_org": record.source_id,
            "source_publication": record.factor_name,
            "source_year": record.valid_from.year,
            "methodology": "IPCC_Tier_1",
            "version": record.source_version,
        },
        "valid_from": record.valid_from.isoformat(),
        "valid_to": (
            None
            if record.valid_to == _VALID_TO_SENTINEL
            else record.valid_to.isoformat()
        ),
        "uncertainty_95ci": 0.05,
        "dqs": {
            "temporal": record.quality.temporal_score,
            "geographical": record.quality.geographic_score,
            "technological": record.quality.technology_score,
            "representativeness": record.quality.completeness_score,
            "methodological": record.quality.verification_score,
        },
        "license_info": {
            "license": record.licensing.license_name or record.licensing.redistribution_class,
            "redistribution_allowed": record.licensing.redistribution_allowed,
            "commercial_use_allowed": record.licensing.commercial_use_allowed,
            "attribution_required": record.licensing.attribution_required,
        },
        "created_at": record.lineage.ingested_at.isoformat(),
        "updated_at": record.lineage.ingested_at.isoformat(),
        "created_by": record.lineage.ingested_by,
        "factor_status": record.status,
        "factor_family": record.factor_family,
        "factor_name": record.factor_name,
        "method_profile": record.method_profile,
        "factor_version": record.factor_version,
        "source_id": record.source_id,
        "source_release": record.source_version,
        "redistribution_class": record.licensing.redistribution_class,
        "validation_flags": {},
        "activity_tags": [],
        "sector_tags": list(record.activity_schema.classification_codes),
    }


def migrate_record(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a single old-shape dict to the v1 serialised shape.

    Idempotent: re-running on a v1 dict returns the same dict (with any
    missing defaults filled in).
    """
    record = from_legacy_dict(data)
    return record.to_dict()


__all__ = [
    # CTO-reversible constants
    "STATUS_ENUM",
    "REDISTRIBUTION_CLASS_ENUM",
    "GWP_SET_ENUM",
    "DEFAULT_GWP_SET",
    "METHOD_PROFILE_ENUM",
    "FACTOR_FAMILY_ENUM",
    "FORMULA_TYPE_ENUM",
    # Helpers
    "FACTOR_ID_REGEX",
    "compute_fqs",
    # Sub-objects
    "JurisdictionV1",
    "ActivitySchemaV1",
    "NumeratorV1",
    "DenominatorV1",
    "QualityV1",
    "LineageV1",
    "LicensingV1",
    "ExplainabilityV1",
    "RawRecordRefV1",
    # Main model
    "FactorRecordV1",
    # Migration surface
    "from_legacy_dict",
    "to_legacy_dict",
    "migrate_record",
]
