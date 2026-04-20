# -*- coding: utf-8 -*-
"""
CTO-spec canonical-record extensions (Phase F1).

This module introduces the CTO-spec enumerations and structured
sub-objects that the current :class:`~greenlang.data.emission_factor_record.EmissionFactorRecord`
is missing.  The existing record class is extended **backward-compatibly**:
every new field is optional so the ~25 600 lines of YAML factor data
continue to load without migration.

CTO reference: "GreenLang Factors — Canonical factor record: required parameters"
(2026-04-20).

Non-negotiables this module encodes:

1. Never store only CO2e — stored in gas vectors, CO2e derived (already in v2).
2. Never overwrite a factor — see :class:`ChangeLog` + ``factor_version``.
3. Never hide fallback logic — see :class:`Explainability.fallback_rank` +
   ``assumptions`` list.
4. Never mix licensing classes — see :class:`RedistributionClass` +
   :func:`enforce_license_class_homogeneity`.
5. Never ship without validity + source version — see
   :func:`validate_non_negotiables` (enforced at repository write time).
6. Never let policy workflows call raw factors — :class:`MethodProfile` guards
   the resolution path so callers must pass a profile.

Adding more fields later is fine as long as they land optional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FactorFamily(str, Enum):
    """The 15 factor families listed in the CTO spec."""

    EMISSIONS = "emissions"
    ENERGY_CONVERSION = "energy_conversion"
    CARBON_CONTENT = "carbon_content"
    OXIDATION = "oxidation"
    HEATING_VALUE = "heating_value"
    DENSITY = "density"
    REFRIGERANT_GWP = "refrigerant_gwp"
    GRID_INTENSITY = "grid_intensity"
    RESIDUAL_MIX = "residual_mix"
    TRANSPORT_LANE = "transport_lane"
    MATERIAL_EMBODIED = "material_embodied"
    WASTE_TREATMENT = "waste_treatment"
    LAND_USE_REMOVALS = "land_use_removals"
    FINANCE_PROXY = "finance_proxy"
    CLASSIFICATION_MAPPING = "classification_mapping"


class FormulaType(str, Enum):
    """How the factor is applied to activity data."""

    DIRECT_FACTOR = "direct_factor"            # activity × factor = emissions
    COMBUSTION = "combustion"                  # mass × LHV × oxidation × factor
    LCA = "lca"                                # cradle-to-gate LCA
    SPEND_PROXY = "spend_proxy"                # spend × intensity
    TRANSPORT_CHAIN = "transport_chain"        # ISO 14083 chain calc
    RESIDUAL_MIX = "residual_mix"              # AIB residual mix adjustment
    CARBON_BUDGET = "carbon_budget"            # LSR-style budget logic
    CUSTOM = "custom"                          # pack-defined


class SourceType(str, Enum):
    """Nature of the upstream data source."""

    GOVERNMENT = "government"
    STANDARD_SETTER = "standard_setter"
    INDUSTRY_BODY = "industry_body"
    LICENSED_COMMERCIAL = "licensed_commercial"
    CUSTOMER_PROVIDED = "customer_provided"
    ACADEMIC = "academic"


class RedistributionClass(str, Enum):
    """License class for the factor data itself.

    The CTO non-negotiable is: "never mix licensing classes" — a single
    API response must not mingle OPEN + LICENSED + CUSTOMER_PRIVATE
    records, so the redistribution class is surfaced on every record.
    """

    OPEN = "open"                              # public, redistributable
    RESTRICTED = "restricted"                  # attribution / field-limits
    LICENSED = "licensed"                      # commercial license required
    CUSTOMER_PRIVATE = "customer_private"      # tenant overlay
    OEM_REDISTRIBUTABLE = "oem_redistributable"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    INTERNAL_REVIEW = "internal_review"
    EXTERNAL_VERIFIED = "external_verified"   # third-party audit
    REGULATOR_APPROVED = "regulator_approved"


class PrimaryDataFlag(str, Enum):
    """Primary vs secondary data tagging."""

    PRIMARY = "primary"                        # directly measured
    PRIMARY_MODELED = "primary_modeled"        # measured inputs + model
    SECONDARY = "secondary"                    # database default
    PROXY = "proxy"                            # stand-in estimate


class UncertaintyDistribution(str, Enum):
    """Distribution the ``uncertainty_95ci`` range describes."""

    UNKNOWN = "unknown"
    NORMAL = "normal"
    LOG_NORMAL = "log_normal"
    TRIANGULAR = "triangular"
    UNIFORM = "uniform"
    BETA_PERT = "beta_pert"


class ElectricityBasis(str, Enum):
    """Scope 2 electricity basis (GHG Protocol Scope 2 Guidance)."""

    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"
    SUPPLIER_SPECIFIC = "supplier_specific"
    RESIDUAL_MIX = "residual_mix"


class MethodProfile(str, Enum):
    """Top-level methodology profile the factor is valid for.

    Non-negotiable #6: every resolution call passes a ``method_profile``
    so policy workflows cannot bypass methodology constraints by calling
    the raw catalog.
    """

    CORPORATE_SCOPE1 = "corporate_scope1"
    CORPORATE_SCOPE2_LOCATION = "corporate_scope2_location_based"
    CORPORATE_SCOPE2_MARKET = "corporate_scope2_market_based"
    CORPORATE_SCOPE3 = "corporate_scope3"
    PRODUCT_CARBON = "product_carbon"          # ISO 14067 / GHG PS / PACT
    FREIGHT_ISO_14083 = "freight_iso_14083"    # GLEC-aligned
    LAND_REMOVALS = "land_removals"            # GHG LSR
    FINANCE_PROXY = "finance_proxy"            # PCAF
    EU_CBAM = "eu_cbam"
    EU_DPP = "eu_dpp"


# ---------------------------------------------------------------------------
# Structured sub-objects
# ---------------------------------------------------------------------------


@dataclass
class Jurisdiction:
    """Structured jurisdiction replacing the flat ``geography`` string."""

    country: Optional[str] = None          # ISO 3166-1 alpha-2 (e.g., 'IN', 'US')
    region: Optional[str] = None           # sub-national (e.g., 'US-CA')
    grid_region: Optional[str] = None      # grid sub-region (e.g., 'eGRID-SERC')


@dataclass
class ActivitySchema:
    """Structured activity categorisation."""

    category: str                          # e.g., 'purchased_electricity'
    sub_category: Optional[str] = None     # e.g., 'grid_average'
    classification_codes: List[str] = field(default_factory=list)
    # Each code is a qualified identifier: "NAICS:221112", "CN:7208", "ISIC:D351".


@dataclass
class FactorParameters:
    """Electricity + combustion + general adjustment flags (CTO spec)."""

    scope_applicability: List[str] = field(default_factory=list)
    electricity_basis: Optional[ElectricityBasis] = None
    residual_mix_applicable: bool = False
    supplier_specific: bool = False
    transmission_loss_included: bool = False
    biogenic_share: Optional[float] = None          # 0.0–1.0
    uncertainty_low: Optional[float] = None
    uncertainty_high: Optional[float] = None


@dataclass
class Verification:
    """Third-party / internal verification status of a factor."""

    status: VerificationStatus = VerificationStatus.UNVERIFIED
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    verification_reference: Optional[str] = None


@dataclass
class Explainability:
    """Human-readable explanation shipped with every resolved factor."""

    assumptions: List[str] = field(default_factory=list)
    fallback_rank: int = 7              # 1 = customer override, 7 = global default
    rationale: Optional[str] = None     # short natural-language summary


@dataclass
class ChangeLogEntry:
    """Immutable change-log row (CTO non-negotiable #2)."""

    changed_at: datetime
    changed_by: str
    change_reason: str                  # e.g., "annual source refresh"
    field_changes: List[str] = field(default_factory=list)
    previous_factor_version: Optional[str] = None


@dataclass
class RawRecordRef:
    """Pointer to the *pre-normalization* source record.

    CTO spec: "store both raw source records AND GreenLang-normalized
    records".  The raw form is needed for licensing audits and for
    re-normalization if the matcher vocabulary evolves.
    """

    raw_record_id: str
    raw_payload_hash: str               # SHA-256 of canonical JSON
    raw_format: str                     # "csv" | "xml" | "json" | "pdf_ocr" | ...
    storage_uri: Optional[str] = None


# ---------------------------------------------------------------------------
# Enforcement helpers
# ---------------------------------------------------------------------------


class NonNegotiableViolation(ValueError):
    """Raised when a record / response violates a CTO non-negotiable."""


def enforce_license_class_homogeneity(
    records: Iterable[Any],
) -> None:
    """Non-negotiable #4: a single response must not mix licensing classes.

    Accepts any iterable whose items expose either ``redistribution_class``
    or the legacy ``license_class`` string.  Raises on mixed classes.
    """
    classes: set[str] = set()
    for rec in records:
        klass = getattr(rec, "redistribution_class", None) or getattr(
            rec, "license_class", None
        )
        if klass is None:
            continue
        classes.add(str(klass))
    if len(classes) > 1:
        raise NonNegotiableViolation(
            "License-class mixing detected in single response: %s. "
            "Per CTO non-negotiable #4, the API must not mingle "
            "redistribution classes. Split into separate responses." % sorted(classes)
        )


def validate_non_negotiables(record: Any) -> None:
    """Run the per-record non-negotiables (1, 2, 5, 6).

    Applied at repository write time + CI linter.  This is the single
    point callers invoke to reject malformed records.
    """
    # #1: gas vectors present (derived CO2e OK, but raw gas must be there).
    if not getattr(record, "vectors", None):
        raise NonNegotiableViolation(
            "non-negotiable #1: record %r has no gas vectors; "
            "CO2e must be derived from CO2/CH4/N2O — never store only CO2e."
            % getattr(record, "factor_id", "<unknown>")
        )

    # #5: validity + source version required.
    if getattr(record, "valid_from", None) is None:
        raise NonNegotiableViolation(
            "non-negotiable #5: record %r missing valid_from."
            % getattr(record, "factor_id", "<unknown>")
        )
    prov = getattr(record, "provenance", None)
    source_version = getattr(record, "source_release", None) or getattr(
        record, "release_version", None
    )
    if prov is None and not source_version:
        raise NonNegotiableViolation(
            "non-negotiable #5: record %r missing source version + provenance."
            % getattr(record, "factor_id", "<unknown>")
        )

    # #2: factor_version must be set if the record is in a non-preview status.
    status = str(getattr(record, "factor_status", "certified")).lower()
    factor_version = getattr(record, "factor_version", None)
    if status in {"certified", "deprecated"} and not factor_version and not getattr(
        record, "release_version", None
    ):
        raise NonNegotiableViolation(
            "non-negotiable #2: %r in status=%s has no factor_version + no release_version; "
            "every certified/deprecated record must pin its version." % (
                getattr(record, "factor_id", "<unknown>"), status
            )
        )

    # #6: policy-flavoured profiles must not be None for certified records.
    # (Soft check: if a method_profile field exists, it should be populated
    # when status is 'certified' and the compliance_frameworks list includes
    # a regulated framework like 'CBAM' or 'CSRD'.)
    if status == "certified":
        method_profile = getattr(record, "method_profile", None)
        frameworks = set(str(f).upper() for f in (getattr(record, "compliance_frameworks", []) or []))
        regulated = {"CBAM", "CSRD", "CSRD_E1", "SB253", "SB-253", "PACT"}
        if regulated & frameworks and not method_profile:
            raise NonNegotiableViolation(
                "non-negotiable #6: certified record %r covers a regulated framework "
                "(%s) but has no method_profile. Policy workflows must not call raw "
                "factors — bind to a MethodProfile." % (
                    getattr(record, "factor_id", "<unknown>"), sorted(regulated & frameworks)
                )
            )


# Public surface.
__all__ = [
    # Enums
    "FactorFamily",
    "FormulaType",
    "SourceType",
    "RedistributionClass",
    "VerificationStatus",
    "PrimaryDataFlag",
    "UncertaintyDistribution",
    "ElectricityBasis",
    "MethodProfile",
    # Sub-objects
    "Jurisdiction",
    "ActivitySchema",
    "FactorParameters",
    "Verification",
    "Explainability",
    "ChangeLogEntry",
    "RawRecordRef",
    # Enforcement
    "NonNegotiableViolation",
    "enforce_license_class_homogeneity",
    "validate_non_negotiables",
]
