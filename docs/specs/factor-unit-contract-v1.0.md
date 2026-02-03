# GreenLang Factor & Unit Contract v1.0

**Version:** 1.0.0
**Status:** Draft
**Last Updated:** 2026-02-03

## Overview

The Factor & Unit Contract specifies how emission factors are versioned, cited, and applied in GreenLang calculations. It also defines the unit normalization rules to ensure consistent, auditable calculations.

**Core Principle:**
> Every calculation must trace back to a versioned, citable emission factor with documented methodology.

## Emission Factor Model

### Factor Structure

```python
@dataclass
class EmissionFactor:
    """Represents a versioned emission factor."""

    # Identification
    factor_id: str              # Unique identifier (e.g., "DEFRA-2024-diesel-road")
    source: str                 # Data source (DEFRA, EPA, IPCC, ecoinvent)
    vintage: int                # Publication year

    # Value
    value: Decimal              # Factor value (exact precision)
    unit: str                   # Factor unit (e.g., "kg_CO2e/liter")

    # Methodology
    methodology: str            # Calculation methodology
    scope: str                  # GHG Protocol scope (1, 2, 3)
    category: str               # Activity category

    # Uncertainty
    uncertainty_pct: Decimal    # Uncertainty percentage
    uncertainty_type: str       # Type (normal, uniform, triangular)

    # Citation
    citation: str               # Full citation
    url: Optional[str]          # Source URL

    # Metadata
    valid_from: date            # Start of validity period
    valid_until: Optional[date] # End of validity (None = current)
    superseded_by: Optional[str]# Replacement factor ID
```

### Factor Identifier Format

```
{SOURCE}-{YEAR}-{FUEL_TYPE}-{CATEGORY}-{REGION}

Examples:
- DEFRA-2024-diesel-road-UK
- EPA-2024-natural_gas-combustion-US
- IPCC-2021-coal-power_generation-GLOBAL
- ecoinvent-3.9-electricity-grid_mix-DE
```

## Factor Versioning

### Version Rules

1. **Annual Updates**: Factors are versioned by publication year
2. **Immutability**: Once published, a factor version never changes
3. **Supersession**: New versions supersede old ones but don't delete them
4. **Backward Compatibility**: Old factors remain available for audit

### Version Resolution

```python
def resolve_factor(
    source: str,
    fuel_type: str,
    category: str,
    year: Optional[int] = None,
    region: Optional[str] = None
) -> EmissionFactor:
    """
    Resolve emission factor with optional version constraints.

    Args:
        source: Data source (DEFRA, EPA, etc.)
        fuel_type: Fuel type (diesel, natural_gas, etc.)
        category: Activity category
        year: Specific vintage (None = latest)
        region: Geographic region (None = global)

    Returns:
        Matching EmissionFactor
    """
    if year:
        # Exact version match
        return get_factor_exact(source, fuel_type, category, year, region)
    else:
        # Latest available version
        return get_factor_latest(source, fuel_type, category, region)
```

### Vintage Constraints

```yaml
# In pack.yaml
policy:
  ef_vintage_min: 2024  # Minimum acceptable vintage

# In pipeline
steps:
  - name: calculate
    config:
      factor_vintage: 2024  # Specific vintage
      # OR
      factor_vintage_min: 2023  # Minimum vintage
```

## Factor Application

### Application Formula

```
emissions = activity_value × factor_value × [unit_conversion]
```

### Application Rules

1. **Units Must Match**: Activity unit must be convertible to factor unit
2. **Precision Preserved**: Use Decimal arithmetic throughout
3. **Citation Required**: Every application records factor citation
4. **Uncertainty Propagated**: Uncertainty carries through calculations

### Implementation

```python
def apply_emission_factor(
    activity: ActivityData,
    factor: EmissionFactor
) -> EmissionResult:
    """
    Apply emission factor to activity data.

    Args:
        activity: Activity data with value and unit
        factor: Emission factor to apply

    Returns:
        Calculated emissions with provenance
    """
    # Unit conversion if needed
    converted_value = convert_units(
        activity.value,
        activity.unit,
        factor.get_activity_unit()
    )

    # Calculate emissions
    emissions = converted_value * factor.value

    # Calculate uncertainty
    uncertainty = calculate_uncertainty(
        activity.uncertainty,
        factor.uncertainty_pct,
        factor.uncertainty_type
    )

    return EmissionResult(
        value=emissions,
        unit=factor.get_emissions_unit(),  # kg_CO2e
        uncertainty=uncertainty,
        factor_citation=create_citation(factor),
        methodology=factor.methodology
    )
```

## Factor Sources

### Supported Sources

| Source | Coverage | Update Frequency | URL |
|--------|----------|------------------|-----|
| DEFRA | UK, comprehensive | Annual | gov.uk/defra |
| EPA | US, comprehensive | Annual | epa.gov |
| IPCC | Global, guidelines | 5-10 years | ipcc.ch |
| ecoinvent | Global, LCA | Continuous | ecoinvent.org |
| GHG Protocol | Global, methodology | As needed | ghgprotocol.org |
| IEA | Global, energy | Annual | iea.org |

### Source Priority

When multiple sources are available:

1. **Regulatory requirement** takes precedence (e.g., CSRD requires EFRAG factors)
2. **Geographic match** (e.g., use DEFRA for UK, EPA for US)
3. **Latest vintage** within acceptable range
4. **Most specific** category match

## Citation Requirements

### Citation Format

```
{Source} {Publication Title} {Year}, {Table/Section Reference}

Examples:
- DEFRA UK Government GHG Conversion Factors 2024, Table 1: Fuels
- EPA Emission Factors for Greenhouse Gas Inventories 2024, Table 1
- IPCC Guidelines for National Greenhouse Gas Inventories 2006, Volume 2, Chapter 1
```

### Citation Record

```json
{
  "factor_id": "DEFRA-2024-diesel-road",
  "citation": {
    "source": "DEFRA",
    "title": "UK Government GHG Conversion Factors for Company Reporting",
    "year": 2024,
    "reference": "Table 1: Fuels",
    "url": "https://gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
    "accessed_at": "2026-02-03"
  }
}
```

## Unit System

### Supported Units

#### Mass Units
| Unit | Symbol | Base Conversion |
|------|--------|-----------------|
| Kilogram | kg | 1 |
| Gram | g | 0.001 |
| Metric ton (tonne) | t | 1000 |
| Pound | lb | 0.453592 |
| Short ton | US_ton | 907.185 |

#### Volume Units
| Unit | Symbol | Base Conversion (liters) |
|------|--------|--------------------------|
| Liter | L | 1 |
| Milliliter | mL | 0.001 |
| Cubic meter | m3 | 1000 |
| Gallon (US) | gal_US | 3.78541 |
| Gallon (UK) | gal_UK | 4.54609 |
| Barrel (oil) | bbl | 158.987 |

#### Energy Units
| Unit | Symbol | Base Conversion (MJ) |
|------|--------|---------------------|
| Megajoule | MJ | 1 |
| Kilojoule | kJ | 0.001 |
| Gigajoule | GJ | 1000 |
| Kilowatt-hour | kWh | 3.6 |
| Megawatt-hour | MWh | 3600 |
| Therm | therm | 105.506 |
| BTU | BTU | 0.001055 |
| MMBTU | MMBTU | 1055.06 |

### Unit Conversion

```python
def convert_units(
    value: Decimal,
    from_unit: str,
    to_unit: str
) -> Decimal:
    """
    Convert value between compatible units.

    Args:
        value: Numeric value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted value with full precision

    Raises:
        IncompatibleUnitsError: If units cannot be converted
    """
    # Get unit definitions
    from_def = get_unit_definition(from_unit)
    to_def = get_unit_definition(to_unit)

    # Check compatibility
    if from_def.dimension != to_def.dimension:
        raise IncompatibleUnitsError(from_unit, to_unit)

    # Convert via base unit
    base_value = value * from_def.to_base
    result = base_value / to_def.to_base

    return result
```

### Heating Values

#### High vs Low Heating Value

| Fuel | HHV (MJ/kg) | LHV (MJ/kg) | HHV/LHV Ratio |
|------|-------------|-------------|---------------|
| Natural Gas | 55.5 | 50.0 | 1.11 |
| Diesel | 45.6 | 42.8 | 1.07 |
| Coal (bituminous) | 32.5 | 31.0 | 1.05 |
| Wood (dry) | 20.0 | 18.5 | 1.08 |

#### Heating Value Conversion

```python
def convert_heating_value(
    value: Decimal,
    from_basis: str,  # "HHV" or "LHV"
    to_basis: str,
    fuel_type: str
) -> Decimal:
    """Convert between HHV and LHV."""
    ratio = get_hhv_lhv_ratio(fuel_type)

    if from_basis == "HHV" and to_basis == "LHV":
        return value / ratio
    elif from_basis == "LHV" and to_basis == "HHV":
        return value * ratio
    else:
        return value
```

## GWP Timeframes

### Global Warming Potential Values

| Gas | AR4 (100yr) | AR5 (100yr) | AR6 (100yr) |
|-----|-------------|-------------|-------------|
| CO2 | 1 | 1 | 1 |
| CH4 | 25 | 28 | 27.9 |
| N2O | 298 | 265 | 273 |
| HFC-134a | 1430 | 1300 | 1526 |
| SF6 | 22800 | 23500 | 25200 |

### GWP Conversion

```python
def convert_gwp_timeframe(
    emissions: dict,  # {"CO2": x, "CH4": y, "N2O": z}
    from_gwp: str,    # "AR4", "AR5", "AR6"
    to_gwp: str
) -> Decimal:
    """
    Convert CO2e using different GWP timeframes.

    Args:
        emissions: Gas-specific emissions
        from_gwp: Source GWP timeframe
        to_gwp: Target GWP timeframe

    Returns:
        Total CO2e under target GWP
    """
    total = Decimal(0)
    for gas, value in emissions.items():
        from_factor = get_gwp_factor(gas, from_gwp)
        to_factor = get_gwp_factor(gas, to_gwp)

        # Remove original GWP, apply new
        base_value = value / from_factor
        converted = base_value * to_factor
        total += converted

    return total
```

## Uncertainty Quantification

### Uncertainty Types

| Type | Description | Use Case |
|------|-------------|----------|
| Normal | Gaussian distribution | Most measurements |
| Uniform | Equal probability | Range estimates |
| Triangular | Min/mode/max | Expert judgment |
| Lognormal | Right-skewed | Positive values |

### Uncertainty Propagation

```python
def propagate_uncertainty(
    activity_uncertainty: UncertaintyEstimate,
    factor_uncertainty: UncertaintyEstimate,
    operation: str = "multiply"
) -> UncertaintyEstimate:
    """
    Propagate uncertainty through calculation.

    For multiplication: σ_result = √(σ_a² + σ_f²)
    """
    if operation == "multiply":
        combined_pct = (
            activity_uncertainty.pct ** 2 +
            factor_uncertainty.pct ** 2
        ) ** Decimal("0.5")

        return UncertaintyEstimate(
            pct=combined_pct,
            type="normal",
            confidence=0.95
        )
```

### Monte Carlo Simulation

```python
def monte_carlo_uncertainty(
    calculation_fn: Callable,
    inputs: dict,
    uncertainties: dict,
    n_samples: int = 10000,
    seed: int = 42
) -> UncertaintyResult:
    """
    Estimate uncertainty via Monte Carlo simulation.

    Args:
        calculation_fn: Function to evaluate
        inputs: Central value inputs
        uncertainties: Uncertainty for each input
        n_samples: Number of samples
        seed: Random seed for reproducibility

    Returns:
        Uncertainty estimate with percentiles
    """
    rng = np.random.default_rng(seed)
    results = []

    for _ in range(n_samples):
        sampled_inputs = sample_inputs(inputs, uncertainties, rng)
        result = calculation_fn(**sampled_inputs)
        results.append(result)

    return UncertaintyResult(
        mean=np.mean(results),
        std=np.std(results),
        p5=np.percentile(results, 5),
        p95=np.percentile(results, 95),
        samples=results
    )
```

## Factor Database Schema

```sql
CREATE TABLE emission_factors (
    factor_id VARCHAR(100) PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    vintage INTEGER NOT NULL,
    fuel_type VARCHAR(100) NOT NULL,
    category VARCHAR(100) NOT NULL,
    region VARCHAR(50),

    -- Value
    value DECIMAL(20, 10) NOT NULL,
    unit VARCHAR(50) NOT NULL,

    -- Methodology
    methodology VARCHAR(200),
    scope VARCHAR(10),

    -- Uncertainty
    uncertainty_pct DECIMAL(10, 4),
    uncertainty_type VARCHAR(20),

    -- Citation
    citation TEXT NOT NULL,
    url VARCHAR(500),

    -- Validity
    valid_from DATE NOT NULL,
    valid_until DATE,
    superseded_by VARCHAR(100),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(source, vintage, fuel_type, category, region)
);

CREATE INDEX idx_factors_source_vintage ON emission_factors(source, vintage);
CREATE INDEX idx_factors_fuel_type ON emission_factors(fuel_type);
CREATE INDEX idx_factors_validity ON emission_factors(valid_from, valid_until);
```

## Commands

```bash
# List available factors
gl factors list --source DEFRA --vintage 2024

# Get specific factor
gl factors get DEFRA-2024-diesel-road

# Validate factor usage
gl factors validate run.json

# Check for updates
gl factors check-updates --current 2023

# Export factor database
gl factors export --format json > factors.json
```

## Appendix A: Unit Aliases

| Alias | Canonical |
|-------|-----------|
| kg | kilogram |
| g | gram |
| t | tonne |
| mt | metric_ton |
| L | liter |
| l | liter |
| m³ | cubic_meter |
| kWh | kilowatt_hour |
| MWh | megawatt_hour |

## Appendix B: Factor Quality Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| 1 - Measured | Direct measurement | Tier 3 reporting |
| 2 - Calculated | Derived from measurements | Tier 2 reporting |
| 3 - Default | Published default values | Tier 1 reporting |
| 4 - Estimated | Expert estimates | Screening |
| 5 - Proxy | Similar activity proxy | Gap filling |
