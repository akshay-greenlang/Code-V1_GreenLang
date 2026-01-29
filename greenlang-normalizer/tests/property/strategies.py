"""
Custom Hypothesis Strategies for GL-FOUND-X-003 Property-Based Tests.

This module provides custom Hypothesis strategies that generate valid inputs
for testing the GreenLang Normalizer components. These strategies ensure
that generated values are within valid ranges and follow the expected formats.

Strategies:
    - valid_unit_strings: Generate valid unit expression strings
    - valid_values: Generate valid numeric values (avoiding overflow)
    - valid_contexts: Generate valid conversion contexts
    - valid_entity_names: Generate valid entity identifiers
    - valid_dimensions: Generate valid dimension objects
    - valid_quantities: Generate valid quantity objects

Usage:
    >>> from strategies import valid_values, valid_unit_strings
    >>> from hypothesis import given
    >>>
    >>> @given(value=valid_values(), unit=valid_unit_strings())
    >>> def test_something(value, unit):
    ...     # Test implementation
    ...     pass
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy


# =============================================================================
# Constants for Generation
# =============================================================================

# Valid SI prefixes
SI_PREFIXES = ["Y", "Z", "E", "P", "T", "G", "M", "k", "h", "da", "d", "c", "m", "u", "n", "p", "f", "a", "z", "y"]

# Base units that can be prefixed
PREFIXABLE_BASE_UNITS = ["g", "m", "s", "W", "J", "L", "Wh", "t", "bar", "Pa"]

# Non-prefixable units
NON_PREFIXABLE_UNITS = [
    "ft", "in", "lb", "oz", "gal", "bbl", "BTU", "therm",
    "mile", "yard", "acre", "psi", "atm", "degC", "degF", "kelvin",
]

# Common sustainability units
SUSTAINABILITY_UNITS = [
    "kg", "g", "t", "lb", "oz",  # Mass
    "m", "km", "mi", "ft",  # Length
    "L", "gal", "bbl", "m3",  # Volume
    "kWh", "MWh", "GWh", "MJ", "GJ", "TJ", "BTU", "therm",  # Energy
    "kW", "MW", "GW", "W",  # Power
    "kgCO2e", "tCO2e", "lbCO2e",  # Emissions
]

# Unit conversion pairs (source, target) that are known to be valid
VALID_CONVERSION_PAIRS = [
    # Mass conversions
    ("kilogram", "gram"),
    ("kilogram", "metric_ton"),
    ("gram", "kilogram"),
    ("metric_ton", "kilogram"),
    ("kilogram", "pound"),
    ("pound", "kilogram"),
    # Energy conversions
    ("kilowatt_hour", "megajoule"),
    ("megajoule", "kilowatt_hour"),
    ("megawatt_hour", "kilowatt_hour"),
    ("kilowatt_hour", "megawatt_hour"),
    ("gigajoule", "megajoule"),
    ("megajoule", "gigajoule"),
    # Volume conversions
    ("liter", "gallon"),
    ("gallon", "liter"),
    ("cubic_meter", "liter"),
    ("liter", "cubic_meter"),
    # Distance conversions
    ("kilometer", "mile"),
    ("mile", "kilometer"),
]

# Dimension groups (units within same group are compatible)
DIMENSION_GROUPS = {
    "mass": ["kilogram", "gram", "metric_ton", "pound"],
    "length": ["meter", "kilometer", "mile", "foot"],
    "time": ["second", "minute", "hour", "day"],
    "energy": ["joule", "kilojoule", "megajoule", "gigajoule", "kilowatt_hour", "megawatt_hour"],
    "volume": ["liter", "cubic_meter", "gallon"],
    "power": ["watt", "kilowatt", "megawatt"],
}

# Valid entity types for reference resolution
ENTITY_TYPES = ["fuel", "material", "process"]

# Sample fuel names for entity resolution tests
FUEL_NAMES = [
    "Natural Gas", "Diesel", "Petrol", "Gasoline", "Coal",
    "Nat Gas", "Heavy Fuel Oil", "LPG", "Propane", "Kerosene",
    "Biomass", "Wood Pellets", "Biodiesel", "Ethanol",
]

# Sample material names
MATERIAL_NAMES = [
    "Steel", "Aluminum", "Concrete", "Cement", "Glass",
    "Copper", "Iron", "Plastic", "Wood", "Paper",
]

# Sample process names
PROCESS_NAMES = [
    "Combustion", "Electric Arc Furnace", "Blast Furnace",
    "Electrolysis", "Steam Reforming", "Fermentation",
]


# =============================================================================
# Numeric Value Strategies
# =============================================================================

@st.composite
def valid_values(
    draw: st.DrawFn,
    min_value: float = 1e-15,
    max_value: float = 1e15,
    allow_negative: bool = False,
) -> float:
    """
    Generate valid numeric values for unit conversions.

    Avoids values that could cause overflow or precision issues:
    - Very small values close to zero
    - Very large values approaching infinity
    - NaN and infinity

    Args:
        draw: Hypothesis draw function
        min_value: Minimum absolute value (default 1e-15)
        max_value: Maximum absolute value (default 1e15)
        allow_negative: Whether to allow negative values

    Returns:
        A valid float value for conversion testing
    """
    # Generate value with controlled magnitude
    value = draw(
        st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )

    if allow_negative and draw(st.booleans()):
        value = -value

    return value


@st.composite
def valid_positive_values(draw: st.DrawFn) -> float:
    """
    Generate strictly positive numeric values.

    Returns:
        A positive float value
    """
    return draw(valid_values(min_value=1e-10, max_value=1e10, allow_negative=False))


@st.composite
def valid_scaling_factors(draw: st.DrawFn) -> float:
    """
    Generate valid scaling factors for testing scaling property.

    Scaling factors should be reasonable (not too large or small)
    to avoid numerical precision issues.

    Returns:
        A valid scaling factor
    """
    return draw(
        st.floats(
            min_value=0.001,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )


@st.composite
def valid_decimal_values(draw: st.DrawFn) -> Decimal:
    """
    Generate valid Decimal values for high-precision testing.

    Returns:
        A valid Decimal value
    """
    # Generate as string to maintain precision
    mantissa = draw(st.integers(min_value=1, max_value=999999999))
    exponent = draw(st.integers(min_value=-10, max_value=10))
    return Decimal(f"{mantissa}E{exponent}")


# =============================================================================
# Unit String Strategies
# =============================================================================

@st.composite
def valid_unit_strings(draw: st.DrawFn) -> str:
    """
    Generate valid unit expression strings.

    Generates various unit formats:
    - Simple units (kg, m, s)
    - Prefixed units (km, mg, MW)
    - Units with exponents (m2, s**-1)
    - Compound units (kg/m3)

    Returns:
        A valid unit string
    """
    format_type = draw(st.sampled_from(["simple", "prefixed", "compound"]))

    if format_type == "simple":
        return draw(st.sampled_from(SUSTAINABILITY_UNITS))

    elif format_type == "prefixed":
        prefix = draw(st.sampled_from(["k", "M", "G", "m", "c"]))
        base = draw(st.sampled_from(["g", "m", "W", "J", "L", "Wh"]))
        return f"{prefix}{base}"

    else:  # compound
        num_unit = draw(st.sampled_from(["kg", "MJ", "kWh", "t"]))
        den_unit = draw(st.sampled_from(["m3", "km", "h", "t", "L"]))
        return f"{num_unit}/{den_unit}"


@st.composite
def valid_simple_unit_strings(draw: st.DrawFn) -> str:
    """
    Generate simple (non-compound) unit strings.

    Returns:
        A simple unit string without division
    """
    return draw(st.sampled_from(SUSTAINABILITY_UNITS))


@st.composite
def valid_prefixed_unit_strings(draw: st.DrawFn) -> str:
    """
    Generate unit strings with SI prefixes.

    Returns:
        A prefixed unit string
    """
    prefix = draw(st.sampled_from(SI_PREFIXES[:10]))  # Common prefixes
    base = draw(st.sampled_from(PREFIXABLE_BASE_UNITS))
    return f"{prefix}{base}"


@st.composite
def valid_compound_unit_strings(draw: st.DrawFn) -> str:
    """
    Generate compound unit strings (with division).

    Returns:
        A compound unit string
    """
    numerator = draw(st.sampled_from(["kg", "MJ", "kWh", "kgCO2e", "t"]))
    denominator = draw(st.sampled_from(["m3", "km", "h", "year", "t", "L"]))
    return f"{numerator}/{denominator}"


@st.composite
def valid_unit_with_exponent(draw: st.DrawFn) -> str:
    """
    Generate unit strings with exponents.

    Returns:
        A unit string with exponent notation
    """
    base = draw(st.sampled_from(["m", "s", "kg", "A", "K"]))
    exponent = draw(st.integers(min_value=1, max_value=3))
    return f"{base}**{exponent}"


# =============================================================================
# Conversion Context Strategies
# =============================================================================

@st.composite
def valid_conversion_pairs_strategy(draw: st.DrawFn) -> Tuple[str, str]:
    """
    Generate valid (source_unit, target_unit) pairs for conversion testing.

    Only generates pairs that are known to have valid conversion paths.

    Returns:
        Tuple of (source_unit, target_unit)
    """
    return draw(st.sampled_from(VALID_CONVERSION_PAIRS))


@st.composite
def valid_same_dimension_units(draw: st.DrawFn) -> Tuple[str, str]:
    """
    Generate two units from the same dimension group.

    Returns:
        Tuple of (unit_a, unit_b) from the same dimension
    """
    dimension = draw(st.sampled_from(list(DIMENSION_GROUPS.keys())))
    units = DIMENSION_GROUPS[dimension]
    unit_a = draw(st.sampled_from(units))
    unit_b = draw(st.sampled_from(units))
    return (unit_a, unit_b)


@st.composite
def valid_contexts(draw: st.DrawFn) -> Dict[str, Any]:
    """
    Generate valid conversion contexts.

    A context includes metadata required for certain conversions:
    - Reference conditions (temperature, pressure)
    - GWP version for emissions
    - Locale settings

    Returns:
        A valid context dictionary
    """
    context: Dict[str, Any] = {}

    # Optionally add reference conditions
    if draw(st.booleans()):
        context["reference_conditions"] = {
            "temperature_C": draw(st.floats(min_value=-40, max_value=60)),
            "pressure_kPa": draw(st.floats(min_value=80, max_value=120)),
        }

    # Optionally add GWP version
    if draw(st.booleans()):
        context["gwp_version"] = draw(st.sampled_from(["AR4", "AR5", "AR6"]))

    # Optionally add locale
    if draw(st.booleans()):
        context["locale"] = draw(st.sampled_from(["en_US", "en_GB", "de_DE", "fr_FR"]))

    return context


# =============================================================================
# Entity Name Strategies
# =============================================================================

@st.composite
def valid_entity_names(
    draw: st.DrawFn,
    entity_type: Optional[str] = None,
) -> str:
    """
    Generate valid entity identifier strings.

    Args:
        draw: Hypothesis draw function
        entity_type: Optional entity type filter (fuel, material, process)

    Returns:
        A valid entity name string
    """
    if entity_type is None:
        entity_type = draw(st.sampled_from(ENTITY_TYPES))

    if entity_type == "fuel":
        return draw(st.sampled_from(FUEL_NAMES))
    elif entity_type == "material":
        return draw(st.sampled_from(MATERIAL_NAMES))
    else:
        return draw(st.sampled_from(PROCESS_NAMES))


@st.composite
def valid_fuel_names(draw: st.DrawFn) -> str:
    """
    Generate valid fuel names for resolution testing.

    Returns:
        A valid fuel name
    """
    return draw(valid_entity_names(entity_type="fuel"))


@st.composite
def valid_material_names(draw: st.DrawFn) -> str:
    """
    Generate valid material names for resolution testing.

    Returns:
        A valid material name
    """
    return draw(valid_entity_names(entity_type="material"))


@st.composite
def valid_entity_with_variations(draw: st.DrawFn) -> Tuple[str, str]:
    """
    Generate an entity name with its canonical form and a variation.

    Returns:
        Tuple of (canonical_name, variation)
    """
    canonical = draw(st.sampled_from(FUEL_NAMES))

    # Generate a variation
    variation_type = draw(st.sampled_from(["lowercase", "uppercase", "spaces"]))

    if variation_type == "lowercase":
        variation = canonical.lower()
    elif variation_type == "uppercase":
        variation = canonical.upper()
    else:
        variation = f"  {canonical}  "  # Extra spaces

    return (canonical, variation)


# =============================================================================
# Dimension Strategies
# =============================================================================

@st.composite
def valid_dimension_exponents(draw: st.DrawFn) -> Dict[str, int]:
    """
    Generate valid dimension exponent dictionaries.

    Returns:
        Dictionary mapping dimension names to exponents
    """
    dimensions = ["mass", "length", "time", "temperature", "amount", "current", "luminosity"]

    # Select 1-3 non-zero dimensions
    num_dims = draw(st.integers(min_value=1, max_value=3))
    selected_dims = draw(st.permutations(dimensions))[:num_dims]

    result = {}
    for dim in selected_dims:
        # Exponents typically -3 to 3 for physical quantities
        exp = draw(st.integers(min_value=-3, max_value=3).filter(lambda x: x != 0))
        result[dim] = exp

    return result


@st.composite
def valid_dimension_pairs(draw: st.DrawFn) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Generate two dimension dictionaries for testing operations.

    Returns:
        Tuple of two dimension exponent dictionaries
    """
    dim_a = draw(valid_dimension_exponents())
    dim_b = draw(valid_dimension_exponents())
    return (dim_a, dim_b)


@st.composite
def valid_dimension_for_unit(draw: st.DrawFn, unit: str) -> Dict[str, int]:
    """
    Generate the correct dimension for a given unit.

    Args:
        draw: Hypothesis draw function
        unit: Unit string

    Returns:
        Dimension exponent dictionary for the unit
    """
    # Map units to their dimensions
    unit_dimensions = {
        "kilogram": {"mass": 1},
        "meter": {"length": 1},
        "second": {"time": 1},
        "joule": {"mass": 1, "length": 2, "time": -2},
        "watt": {"mass": 1, "length": 2, "time": -3},
        "kilowatt_hour": {"mass": 1, "length": 2, "time": -2},
    }

    if unit in unit_dimensions:
        return unit_dimensions[unit]

    # Default to mass for unknown units
    return {"mass": 1}


# =============================================================================
# Quantity Strategies
# =============================================================================

@st.composite
def valid_quantity_strings(draw: st.DrawFn) -> str:
    """
    Generate valid quantity strings (value + unit).

    Returns:
        A valid quantity string like "100 kg"
    """
    value = draw(valid_positive_values())
    unit = draw(valid_simple_unit_strings())

    # Format with appropriate precision
    if value >= 1000 or value < 0.001:
        value_str = f"{value:.6e}"
    else:
        value_str = f"{value:.6f}"

    return f"{value_str} {unit}"


@st.composite
def valid_quantity_dict(draw: st.DrawFn) -> Dict[str, Any]:
    """
    Generate a valid quantity as a dictionary.

    Returns:
        Dictionary with magnitude, unit, and optional metadata
    """
    return {
        "magnitude": draw(valid_positive_values()),
        "unit": draw(valid_simple_unit_strings()),
        "original_unit": None,
        "uncertainty": None,
    }


# =============================================================================
# Audit Record Strategies
# =============================================================================

@st.composite
def valid_measurement_audit(draw: st.DrawFn) -> Dict[str, Any]:
    """
    Generate a valid measurement audit record.

    Returns:
        A measurement audit dictionary
    """
    raw_value = draw(valid_positive_values())
    raw_unit = draw(valid_simple_unit_strings())

    return {
        "field": draw(st.sampled_from(["energy", "mass", "volume", "emissions"])),
        "raw_value": raw_value,
        "raw_unit": raw_unit,
        "expected_dimension": draw(st.sampled_from(list(DIMENSION_GROUPS.keys()))),
        "canonical_value": raw_value,  # Simplified for testing
        "canonical_unit": raw_unit,
        "conversion_steps": [],
    }


@st.composite
def valid_entity_audit(draw: st.DrawFn) -> Dict[str, Any]:
    """
    Generate a valid entity audit record.

    Returns:
        An entity audit dictionary
    """
    entity_type = draw(st.sampled_from(ENTITY_TYPES))

    return {
        "field": f"{entity_type}_type",
        "entity_type": entity_type,
        "raw_name": draw(valid_entity_names(entity_type=entity_type)),
        "reference_id": f"GL-{entity_type.upper()}-001",
        "canonical_name": draw(valid_entity_names(entity_type=entity_type)),
        "match_method": draw(st.sampled_from(["exact", "alias", "fuzzy"])),
        "confidence": draw(st.floats(min_value=0.5, max_value=1.0)),
    }


@st.composite
def valid_audit_event(draw: st.DrawFn) -> Dict[str, Any]:
    """
    Generate a valid complete audit event.

    Returns:
        A complete audit event dictionary
    """
    import uuid
    from datetime import datetime

    num_measurements = draw(st.integers(min_value=1, max_value=5))
    num_entities = draw(st.integers(min_value=0, max_value=3))

    measurements = [draw(valid_measurement_audit()) for _ in range(num_measurements)]
    entities = [draw(valid_entity_audit()) for _ in range(num_entities)]

    return {
        "event_id": f"norm-evt-{uuid.uuid4().hex[:12]}",
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "source_record_id": f"rec-{uuid.uuid4().hex[:8]}",
        "policy_mode": draw(st.sampled_from(["STRICT", "LENIENT"])),
        "status": draw(st.sampled_from(["success", "warning"])),
        "measurements": measurements,
        "entities": entities,
    }


# =============================================================================
# Reference Data Strategies
# =============================================================================

@st.composite
def valid_vocab_entry(draw: st.DrawFn) -> Dict[str, Any]:
    """
    Generate a valid vocabulary entry.

    Returns:
        A vocabulary entry dictionary
    """
    entity_type = draw(st.sampled_from(ENTITY_TYPES))
    name = draw(valid_entity_names(entity_type=entity_type))

    # Generate some aliases
    num_aliases = draw(st.integers(min_value=0, max_value=3))
    aliases = [name.lower()]
    for _ in range(num_aliases):
        aliases.append(f"{name} variation")

    return {
        "id": f"GL-{entity_type.upper()}-{draw(st.integers(min_value=1, max_value=999)):03d}",
        "name": name,
        "aliases": aliases,
        "metadata": {
            "category": draw(st.sampled_from(["primary", "secondary", "tertiary"])),
        },
    }


@st.composite
def valid_vocabulary(draw: st.DrawFn, size: int = 10) -> List[Dict[str, Any]]:
    """
    Generate a valid vocabulary with multiple entries.

    Args:
        draw: Hypothesis draw function
        size: Number of entries to generate

    Returns:
        List of vocabulary entry dictionaries
    """
    return [draw(valid_vocab_entry()) for _ in range(size)]


# =============================================================================
# Tolerance and Precision Strategies
# =============================================================================

@st.composite
def valid_tolerance(draw: st.DrawFn) -> float:
    """
    Generate a valid numerical tolerance value.

    Returns:
        A valid tolerance (relative error threshold)
    """
    return draw(st.floats(min_value=1e-15, max_value=1e-6))


@st.composite
def valid_precision_config(draw: st.DrawFn) -> Dict[str, Any]:
    """
    Generate a valid precision configuration.

    Returns:
        Precision configuration dictionary
    """
    return {
        "significant_digits": draw(st.integers(min_value=6, max_value=15)),
        "rounding_mode": draw(st.sampled_from(["ROUND_HALF_UP", "ROUND_HALF_EVEN", "ROUND_DOWN"])),
        "relative_tolerance": draw(valid_tolerance()),
    }


# =============================================================================
# Export all strategies
# =============================================================================

__all__ = [
    # Numeric strategies
    "valid_values",
    "valid_positive_values",
    "valid_scaling_factors",
    "valid_decimal_values",
    # Unit string strategies
    "valid_unit_strings",
    "valid_simple_unit_strings",
    "valid_prefixed_unit_strings",
    "valid_compound_unit_strings",
    "valid_unit_with_exponent",
    # Conversion strategies
    "valid_conversion_pairs_strategy",
    "valid_same_dimension_units",
    "valid_contexts",
    # Entity strategies
    "valid_entity_names",
    "valid_fuel_names",
    "valid_material_names",
    "valid_entity_with_variations",
    # Dimension strategies
    "valid_dimension_exponents",
    "valid_dimension_pairs",
    "valid_dimension_for_unit",
    # Quantity strategies
    "valid_quantity_strings",
    "valid_quantity_dict",
    # Audit strategies
    "valid_measurement_audit",
    "valid_entity_audit",
    "valid_audit_event",
    # Reference data strategies
    "valid_vocab_entry",
    "valid_vocabulary",
    # Precision strategies
    "valid_tolerance",
    "valid_precision_config",
    # Constants
    "VALID_CONVERSION_PAIRS",
    "DIMENSION_GROUPS",
    "SUSTAINABILITY_UNITS",
    "FUEL_NAMES",
    "MATERIAL_NAMES",
    "PROCESS_NAMES",
]
