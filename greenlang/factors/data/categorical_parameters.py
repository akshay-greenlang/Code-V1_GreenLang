# -*- coding: utf-8 -*-
"""Per-factor-family parameter models (W4-A / M19 / §4 of gap report).

Seven Pydantic discriminated-union models mirror
``config/schemas/categories/*.schema.json`` field-for-field:

* :class:`CombustionParameters`       — combustion.schema.json
* :class:`ElectricityParameters`      — electricity.schema.json
* :class:`TransportParameters`        — transport.schema.json
* :class:`MaterialsProductsParameters` — materials.schema.json (incl. Cat 11)
* :class:`RefrigerantsParameters`     — refrigerants.schema.json
* :class:`LandRemovalsParameters`     — land_removals.schema.json
* :class:`FinanceProxiesParameters`   — finance_proxy.schema.json
* :class:`WasteParameters`            — (inline v1 spec only)
* :class:`GenericParameters`          — the 7 auxiliary families

All models share a common :class:`_CommonParameters` base for
``scope_applicability``, ``uncertainty_low``, ``uncertainty_high``.

Discrimination is on the ``factor_family`` tag supplied by the
containing record; the classes themselves carry a ``family`` literal
string so :func:`parse_parameters` can dispatch.
"""
from __future__ import annotations

from typing import List, Literal, Optional, Union

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover — dev shim for pydantic v1
    from pydantic import BaseModel, Field, validator as field_validator  # type: ignore

    ConfigDict = dict  # type: ignore
    _PYDANTIC_V2 = False


# ---------------------------------------------------------------------------
# Scope enum (string-valued; spec: scope_1 / scope_2 / scope_3 / scope_4)
# ---------------------------------------------------------------------------

ScopeApplicability = Literal["scope_1", "scope_2", "scope_3", "scope_4"]


class _CommonParameters(BaseModel):
    """Fields shared by every parameters sub-schema."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid", frozen=False)

    scope_applicability: List[ScopeApplicability] = Field(default_factory=list)
    uncertainty_low: Optional[float] = None
    uncertainty_high: Optional[float] = None


# ---------------------------------------------------------------------------
# 1. Combustion
# ---------------------------------------------------------------------------


class CombustionParameters(_CommonParameters):
    """Stationary / mobile combustion factor parameters."""

    family: Literal["combustion"] = "combustion"
    fuel_code: str = Field(min_length=1, max_length=64)
    LHV: Optional[float] = Field(default=None, ge=0.0)
    HHV: Optional[float] = Field(default=None, ge=0.0)
    density: Optional[float] = Field(default=None, gt=0.0)
    oxidation_factor: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fossil_carbon_share: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    biogenic_carbon_share: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sulfur_share: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    moisture_share: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ash_share: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    if _PYDANTIC_V2:

        @field_validator("HHV")
        @classmethod
        def _hhv_gte_lhv(cls, v: Optional[float], info) -> Optional[float]:
            if v is not None:
                data = info.data
                lhv = data.get("LHV")
                if lhv is not None and v < lhv:
                    raise ValueError("HHV must be >= LHV when both present")
            return v
    else:  # pragma: no cover

        @field_validator("HHV")
        def _hhv_gte_lhv_v1(cls, v, values):  # type: ignore
            lhv = values.get("LHV") if isinstance(values, dict) else None
            if v is not None and lhv is not None and v < lhv:
                raise ValueError("HHV must be >= LHV when both present")
            return v


# ---------------------------------------------------------------------------
# 2. Electricity
# ---------------------------------------------------------------------------


ElectricityBasisLit = Literal["location", "market", "supplier", "residual"]
CertificateHandlingLit = Optional[Literal["GO", "REC", "I-REC"]]


class ElectricityParameters(_CommonParameters):
    """Scope-2 electricity factor parameters."""

    family: Literal["electricity"] = "electricity"
    electricity_basis: ElectricityBasisLit
    supplier_specific: bool = False
    residual_mix_applicable: bool = False
    certificate_handling: CertificateHandlingLit = None
    td_loss_included: bool = False
    subregion_code: Optional[str] = Field(default=None, max_length=64)


# ---------------------------------------------------------------------------
# 3. Transport
# ---------------------------------------------------------------------------


TransportMode = Literal["road", "rail", "sea", "air", "inland_waterway"]
PayloadBasis = Literal["t-km", "v-km", "TEU-km", "pax-km"]
DistanceBasis = Literal["great_circle", "route", "route_with_empty"]
EnergyBasis = Literal["WTW", "TTW", "TTT", "WTT"]


class TransportParameters(_CommonParameters):
    """ISO 14083 / GLEC-aligned transport factor parameters."""

    family: Literal["transport"] = "transport"
    mode: TransportMode
    vehicle_class: Optional[str] = Field(default=None, max_length=128)
    payload_basis: Optional[PayloadBasis] = None
    distance_basis: Optional[DistanceBasis] = None
    empty_running_assumption: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    utilization_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    refrigerated: bool = False
    energy_basis: Optional[EnergyBasis] = None


# ---------------------------------------------------------------------------
# 4. Materials / Products (incl. Cat 11 use-phase)
# ---------------------------------------------------------------------------


BoundaryLit = Literal["cradle_to_gate", "gate_to_gate", "cradle_to_grave"]
AllocationMethodLit = Literal["mass", "economic", "system_expansion"]
EoLAllocationLit = Optional[Literal["100_1", "50_50", "avoided_burden", "none"]]


class MaterialsProductsParameters(_CommonParameters):
    """Material / product LCA factor parameters (PACT-compatible)."""

    family: Literal["materials_products"] = "materials_products"
    boundary: BoundaryLit
    allocation_method: AllocationMethodLit
    recycled_content_assumption: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    supplier_primary_data_share: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    pcr_reference: Optional[str] = Field(default=None, max_length=256)
    epd_reference: Optional[str] = Field(default=None, max_length=256)
    pact_compatible: bool = False
    # GHG Protocol Scope 3 Cat 11 fields (may be left null)
    product_lifetime_years: Optional[float] = Field(default=None, ge=0.0)
    use_phase_energy_kwh: Optional[float] = Field(default=None, ge=0.0)
    use_phase_frequency_per_year: Optional[float] = Field(default=None, ge=0.0)
    end_of_life_allocation_method: EoLAllocationLit = None


# ---------------------------------------------------------------------------
# 5. Refrigerants
# ---------------------------------------------------------------------------


LeakageBasisLit = Literal["annual", "charge", "disposal"]
RecoveryTreatmentLit = Optional[
    Literal["none", "partial_recovery", "full_recovery", "destroyed"]
]
GwpSetLit = Optional[
    Literal[
        "IPCC_AR4_100",
        "IPCC_AR5_100",
        "IPCC_AR5_20",
        "IPCC_AR6_100",
        "IPCC_AR6_20",
        "Kyoto_SAR_100",
    ]
]


class RefrigerantsParameters(_CommonParameters):
    """Refrigerant (F-gas + ammonia/CO2) factor parameters."""

    family: Literal["refrigerants"] = "refrigerants"
    gas_code: str = Field(max_length=64)
    leakage_basis: LeakageBasisLit
    recharge_assumption: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recovery_destruction_treatment: RecoveryTreatmentLit = None
    gwp_set_mapping: GwpSetLit = None


# ---------------------------------------------------------------------------
# 6. Land removals
# ---------------------------------------------------------------------------


SequestrationBasis = Literal[
    "stock_change", "flux", "gain_loss", "tier1_default", "tier3_model"
]
PermanenceClass = Optional[
    Literal["short_term", "medium_term", "long_term", "permanent"]
]
BiogenicTreatment = Optional[
    Literal["zero_rated", "separate_reporting", "included_in_co2e"]
]


class LandRemovalsParameters(_CommonParameters):
    """GHG-LSR land-use and removals factor parameters."""

    family: Literal["land_removals"] = "land_removals"
    land_use_category: str = Field(max_length=128)
    sequestration_basis: SequestrationBasis
    permanence_class: PermanenceClass = None
    reversal_risk_flag: bool = False
    biogenic_accounting_treatment: BiogenicTreatment = None


# ---------------------------------------------------------------------------
# 7. Finance proxies (PCAF)
# ---------------------------------------------------------------------------


IntensityBasisLit = Literal["revenue", "asset", "ebitda", "employee"]
PCAFScoreLit = Optional[
    Literal["score_1", "score_2", "score_3", "score_4", "score_5"]
]


class FinanceProxiesParameters(_CommonParameters):
    """PCAF asset-class financed-emissions factor parameters."""

    family: Literal["finance_proxies"] = "finance_proxies"
    asset_class: str = Field(max_length=128)
    sector_code: Optional[str] = Field(default=None, max_length=64)
    intensity_basis: IntensityBasisLit
    geography: Optional[str] = Field(default=None, max_length=64)
    proxy_confidence_class: PCAFScoreLit = None


# ---------------------------------------------------------------------------
# 8. Waste
# ---------------------------------------------------------------------------


TreatmentRouteLit = Literal[
    "landfill", "incineration", "compost", "recycle", "anaerobic_digestion"
]


class WasteParameters(_CommonParameters):
    """Waste treatment factor parameters."""

    family: Literal["waste"] = "waste"
    treatment_route: TreatmentRouteLit
    methane_recovery_factor: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    net_calorific_value: Optional[float] = Field(default=None, ge=0.0)


# ---------------------------------------------------------------------------
# 9. Generic (the 7 auxiliary families without a schema of their own)
# ---------------------------------------------------------------------------


class GenericParameters(_CommonParameters):
    """Catch-all parameters for generic factor families.

    Applies to: energy_conversion, carbon_content, oxidation, heating_value,
    density, residual_mix, classification_mapping.  Extra attributes allowed.
    """

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="allow", frozen=False)

    family: Literal["generic"] = "generic"


# ---------------------------------------------------------------------------
# Discriminated-union type alias + factory
# ---------------------------------------------------------------------------


ParametersUnion = Union[
    CombustionParameters,
    ElectricityParameters,
    TransportParameters,
    MaterialsProductsParameters,
    RefrigerantsParameters,
    LandRemovalsParameters,
    FinanceProxiesParameters,
    WasteParameters,
    GenericParameters,
]


_FAMILY_TO_MODEL = {
    "combustion": CombustionParameters,
    "electricity": ElectricityParameters,
    "transport": TransportParameters,
    "materials_products": MaterialsProductsParameters,
    "refrigerants": RefrigerantsParameters,
    "land_removals": LandRemovalsParameters,
    "finance_proxies": FinanceProxiesParameters,
    "waste": WasteParameters,
    # auxiliary -> generic
    "energy_conversion": GenericParameters,
    "carbon_content": GenericParameters,
    "oxidation": GenericParameters,
    "heating_value": GenericParameters,
    "density": GenericParameters,
    "residual_mix": GenericParameters,
    "classification_mapping": GenericParameters,
}


def parse_parameters(
    factor_family: str, payload: Optional[dict]
) -> ParametersUnion:
    """Parse a ``parameters`` payload into the correct model for the family.

    Parameters
    ----------
    factor_family
        The record's ``factor_family`` (15-value enum string).
    payload
        The raw ``parameters`` dict from the record (may be ``None``).

    Returns
    -------
    ParametersUnion
        A validated parameters instance of the family-specific class.

    Raises
    ------
    ValueError
        If ``factor_family`` is unknown.
    pydantic.ValidationError
        If the payload does not match the schema.
    """
    if factor_family not in _FAMILY_TO_MODEL:
        raise ValueError(
            f"Unknown factor_family {factor_family!r}. "
            f"Valid: {sorted(_FAMILY_TO_MODEL)}"
        )
    model_cls = _FAMILY_TO_MODEL[factor_family]
    data = dict(payload or {})
    # Never let the inbound payload override the family tag.
    data.pop("family", None)
    if _PYDANTIC_V2:
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)  # pragma: no cover


def dump_parameters(params: ParametersUnion) -> dict:
    """Serialise a parameters instance to a plain dict (schema-compatible)."""
    if _PYDANTIC_V2:
        data = params.model_dump(exclude_none=False)
    else:  # pragma: no cover
        data = params.dict(exclude_none=False)
    data.pop("family", None)
    return data


__all__ = [
    "_CommonParameters",
    "CombustionParameters",
    "ElectricityParameters",
    "TransportParameters",
    "MaterialsProductsParameters",
    "RefrigerantsParameters",
    "LandRemovalsParameters",
    "FinanceProxiesParameters",
    "WasteParameters",
    "GenericParameters",
    "ParametersUnion",
    "parse_parameters",
    "dump_parameters",
]
