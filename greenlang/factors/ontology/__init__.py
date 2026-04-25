# -*- coding: utf-8 -*-
"""Unit, geography, methodology, chemistry, GWP, and heating-value helpers (S2-S4)."""

from greenlang.factors.ontology.units import (
    is_known_activity_unit,
    suggest_si_base,
    convert_energy,
    convert_energy_to_kwh,
    fuel_energy_content,
    convert_fuel_to_kwh,
    gas_volume_to_mass_kg,
    gas_volume_to_energy_mj,
    steam_enthalpy_kj_per_kg,
    steam_energy_mj,
)
from greenlang.factors.ontology.geography import is_iso3166_alpha2, normalize_grid_token
from greenlang.factors.ontology.methodology import methodology_tags_for_record
from greenlang.factors.ontology.heating_values import (
    FuelHeatingValue,
    get_fuel,
    get_heating_value,
    list_fuels,
    convert_mass_to_energy,
    convert_volume_to_energy,
    apply_moisture_correction,
    apply_temperature_correction,
)
from greenlang.factors.ontology.gwp_sets import (
    GWPSet,
    DEFAULT_GWP_SET,
    get_gwp,
    convert_co2e,
    normalize_gas_code,
)
from greenlang.factors.ontology.chemistry import (
    C_TO_CO2_RATIO,
    CO2Split,
    BiogenicFate,
    RefrigerantLeakageResult,
    carbon_to_co2,
    apply_oxidation_factor,
    get_default_oxidation_factor,
    split_fossil_biogenic_co2,
    biogenic_fate,
    model_refrigerant_leakage,
    build_combustion_gas_vector,
    aggregate_co2e,
)
from greenlang.factors.ontology.urn import (
    GLUrn,
    InvalidUrnError,
    parse as parse_urn,
    build as build_urn,
    validate as validate_urn,
    coerce_factor_id_to_urn,
    URN_PREFIX,
    ALLOWED_KINDS as URN_ALLOWED_KINDS,
    ALLOWED_GEO_TYPES as URN_ALLOWED_GEO_TYPES,
)

__all__ = [
    # units
    "is_known_activity_unit",
    "suggest_si_base",
    "convert_energy",
    "convert_energy_to_kwh",
    "fuel_energy_content",
    "convert_fuel_to_kwh",
    "gas_volume_to_mass_kg",
    "gas_volume_to_energy_mj",
    "steam_enthalpy_kj_per_kg",
    "steam_energy_mj",
    # geography / methodology
    "is_iso3166_alpha2",
    "normalize_grid_token",
    "methodology_tags_for_record",
    # heating values
    "FuelHeatingValue",
    "get_fuel",
    "get_heating_value",
    "list_fuels",
    "convert_mass_to_energy",
    "convert_volume_to_energy",
    "apply_moisture_correction",
    "apply_temperature_correction",
    # GWP
    "GWPSet",
    "DEFAULT_GWP_SET",
    "get_gwp",
    "convert_co2e",
    "normalize_gas_code",
    # chemistry
    "C_TO_CO2_RATIO",
    "CO2Split",
    "BiogenicFate",
    "RefrigerantLeakageResult",
    "carbon_to_co2",
    "apply_oxidation_factor",
    "get_default_oxidation_factor",
    "split_fossil_biogenic_co2",
    "biogenic_fate",
    "model_refrigerant_leakage",
    "build_combustion_gas_vector",
    "aggregate_co2e",
    # URN scheme (v0.1 Alpha)
    "GLUrn",
    "InvalidUrnError",
    "parse_urn",
    "build_urn",
    "validate_urn",
    "coerce_factor_id_to_urn",
    "URN_PREFIX",
    "URN_ALLOWED_KINDS",
    "URN_ALLOWED_GEO_TYPES",
]
