"""
Test Data Generator for Emission Factors

This module provides utilities to generate realistic test data for:
- 500+ emission factors across multiple categories
- Multiple geographies and temporal ranges
- Gas vectors (CO2, CH4, N2O)
- Data quality tiers
- Calculation test scenarios

Usage:
    from tests.test_data_generator import EmissionFactorGenerator

    generator = EmissionFactorGenerator()
    factors = generator.generate_realistic_factors(count=500)
"""

import random
import string
from typing import List, Dict, Any, Optional
from datetime import date, datetime, timedelta
from dataclasses import dataclass, asdict
import json


@dataclass
class GeneratedFactor:
    """Generated emission factor data structure."""
    factor_id: str
    name: str
    category: str
    subcategory: str
    emission_factor_kg_co2e: float
    unit: str
    scope: str
    source_org: str
    source_publication: Optional[str]
    source_uri: str
    standard: str
    last_updated: str
    year_applicable: int
    geographic_scope: str
    geography_level: str
    country_code: str
    state_province: Optional[str]
    region: str
    data_quality_tier: str
    uncertainty_percent: float
    confidence_95ci: Optional[float]
    completeness_score: Optional[float]
    renewable_share: Optional[float]
    notes: str
    metadata_json: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class GasVector:
    """Gas vector data structure."""
    factor_id: str
    gas_type: str
    kg_per_unit: float
    gwp: int


class EmissionFactorGenerator:
    """Generate realistic emission factor test data."""

    # Fuel type definitions with typical ranges
    FUEL_TYPES = {
        'diesel': {
            'category': 'fuels',
            'units': ['gallon', 'liter', 'kg'],
            'ef_range_gallon': (9.5, 10.5),
            'ef_range_liter': (2.5, 2.8),
            'ef_range_kg': (3.1, 3.3),
            'scope': 'Scope 1',
            'gas_composition': {
                'CO2': 0.993,  # 99.3% of emissions
                'CH4': 0.004,  # 0.4%
                'N2O': 0.003   # 0.3%
            }
        },
        'gasoline': {
            'category': 'fuels',
            'units': ['gallon', 'liter', 'kg'],
            'ef_range_gallon': (8.5, 9.0),
            'ef_range_liter': (2.2, 2.4),
            'ef_range_kg': (3.1, 3.2),
            'scope': 'Scope 1',
            'gas_composition': {
                'CO2': 0.992,
                'CH4': 0.005,
                'N2O': 0.003
            }
        },
        'natural_gas': {
            'category': 'fuels',
            'units': ['therms', 'ccf', 'mcf', 'm3', 'MMBtu'],
            'ef_range_therms': (5.2, 5.4),
            'ef_range_m3': (1.85, 1.95),
            'ef_range_MMBtu': (52, 54),
            'scope': 'Scope 1',
            'gas_composition': {
                'CO2': 0.995,
                'CH4': 0.003,
                'N2O': 0.002
            }
        },
        'propane': {
            'category': 'fuels',
            'units': ['gallon', 'liter', 'kg'],
            'ef_range_gallon': (5.6, 5.9),
            'ef_range_liter': (1.4, 1.6),
            'ef_range_kg': (2.9, 3.1),
            'scope': 'Scope 1',
            'gas_composition': {
                'CO2': 0.994,
                'CH4': 0.004,
                'N2O': 0.002
            }
        },
        'coal': {
            'category': 'fuels',
            'units': ['tons', 'kg', 'lbs'],
            'ef_range_tons': (2000, 2200),
            'ef_range_kg': (2.0, 2.2),
            'ef_range_lbs': (0.9, 1.0),
            'scope': 'Scope 1',
            'gas_composition': {
                'CO2': 0.996,
                'CH4': 0.002,
                'N2O': 0.002
            }
        },
        'electricity': {
            'category': 'grids',
            'units': ['kwh', 'mwh'],
            'ef_range_kwh': (0.2, 0.6),
            'ef_range_mwh': (200, 600),
            'scope': 'Scope 2 - Location-Based',
            'gas_composition': {
                'CO2': 0.998,
                'CH4': 0.001,
                'N2O': 0.001
            }
        }
    }

    # Geographic regions
    GEOGRAPHIES = {
        'United States': {'code': 'US', 'region': 'North America', 'level': 'Country'},
        'Canada': {'code': 'CA', 'region': 'North America', 'level': 'Country'},
        'Mexico': {'code': 'MX', 'region': 'North America', 'level': 'Country'},
        'United Kingdom': {'code': 'UK', 'region': 'Europe', 'level': 'Country'},
        'Germany': {'code': 'DE', 'region': 'Europe', 'level': 'Country'},
        'France': {'code': 'FR', 'region': 'Europe', 'level': 'Country'},
        'Spain': {'code': 'ES', 'region': 'Europe', 'level': 'Country'},
        'Italy': {'code': 'IT', 'region': 'Europe', 'level': 'Country'},
        'China': {'code': 'CN', 'region': 'Asia', 'level': 'Country'},
        'Japan': {'code': 'JP', 'region': 'Asia', 'level': 'Country'},
        'India': {'code': 'IN', 'region': 'Asia', 'level': 'Country'},
        'Australia': {'code': 'AU', 'region': 'Oceania', 'level': 'Country'},
        'Brazil': {'code': 'BR', 'region': 'South America', 'level': 'Country'},
    }

    # Data sources
    DATA_SOURCES = {
        'EPA': {
            'name': 'Environmental Protection Agency',
            'uri_base': 'https://epa.gov',
            'standard': 'GHG Protocol',
            'typical_tier': 'Tier 1',
            'typical_uncertainty': 5.0
        },
        'DEFRA': {
            'name': 'UK Department for Environment, Food & Rural Affairs',
            'uri_base': 'https://defra.gov.uk',
            'standard': 'GHG Protocol',
            'typical_tier': 'Tier 1',
            'typical_uncertainty': 6.0
        },
        'IEA': {
            'name': 'International Energy Agency',
            'uri_base': 'https://iea.org',
            'standard': 'GHG Protocol',
            'typical_tier': 'Tier 2',
            'typical_uncertainty': 8.0
        },
        'IPCC': {
            'name': 'Intergovernmental Panel on Climate Change',
            'uri_base': 'https://ipcc.ch',
            'standard': 'IPCC Guidelines',
            'typical_tier': 'Tier 3',
            'typical_uncertainty': 12.0
        },
        'EU Commission': {
            'name': 'European Commission',
            'uri_base': 'https://ec.europa.eu',
            'standard': 'EU CBAM',
            'typical_tier': 'Tier 1',
            'typical_uncertainty': 5.5
        }
    }

    def __init__(self, seed: int = 42):
        """Initialize generator with seed for reproducibility."""
        random.seed(seed)

    def generate_realistic_factors(self, count: int = 500) -> List[GeneratedFactor]:
        """Generate realistic emission factors."""
        factors = []

        # Distribute across fuel types
        fuel_types = list(self.FUEL_TYPES.keys())

        for i in range(count):
            fuel_type = fuel_types[i % len(fuel_types)]
            factors.append(self._generate_single_factor(i, fuel_type))

        return factors

    def _generate_single_factor(self, index: int, fuel_type: str) -> GeneratedFactor:
        """Generate single emission factor."""
        fuel_config = self.FUEL_TYPES[fuel_type]

        # Select geography
        geography_name = random.choice(list(self.GEOGRAPHIES.keys()))
        geography = self.GEOGRAPHIES[geography_name]

        # Select data source
        source_name = random.choice(list(self.DATA_SOURCES.keys()))
        source = self.DATA_SOURCES[source_name]

        # Select unit
        unit = random.choice(fuel_config['units'])

        # Calculate emission factor value
        ef_range_key = f"ef_range_{unit.replace('-', '_')}"
        if ef_range_key in fuel_config:
            ef_value = random.uniform(*fuel_config[ef_range_key])
        else:
            # Fallback to first unit range
            first_range_key = [k for k in fuel_config.keys() if k.startswith('ef_range')][0]
            ef_value = random.uniform(*fuel_config[first_range_key])

        # Generate factor ID
        factor_id = f"{fuel_type}_{geography['code'].lower()}_{2020 + (index % 5)}_{index:04d}"

        # Generate temporal data
        year = 2020 + (index % 5)
        last_updated = date(year, random.randint(1, 12), 1).isoformat()

        # Generate data quality
        tier = source['typical_tier']
        uncertainty = source['typical_uncertainty'] + random.uniform(-2.0, 2.0)

        # Renewable share (only for electricity)
        renewable_share = None
        if fuel_type == 'electricity':
            renewable_share = random.uniform(0.1, 0.4)

        # Create factor
        factor = GeneratedFactor(
            factor_id=factor_id,
            name=f"{fuel_type.replace('_', ' ').title()} - {geography_name} {year}",
            category=fuel_config['category'],
            subcategory=fuel_type,
            emission_factor_kg_co2e=round(ef_value, 4),
            unit=unit,
            scope=fuel_config['scope'],
            source_org=source_name,
            source_publication=f"{source['name']} {year} Report",
            source_uri=f"{source['uri_base']}/factors/{factor_id}",
            standard=source['standard'],
            last_updated=last_updated,
            year_applicable=year,
            geographic_scope=geography_name,
            geography_level=geography['level'],
            country_code=geography['code'],
            state_province=None,
            region=geography['region'],
            data_quality_tier=tier,
            uncertainty_percent=round(uncertainty, 2),
            confidence_95ci=round(ef_value * (uncertainty / 100) * 1.96, 4) if random.random() > 0.5 else None,
            completeness_score=round(random.uniform(0.8, 1.0), 3) if random.random() > 0.3 else None,
            renewable_share=renewable_share,
            notes=f"Generated test factor for {fuel_type} in {geography_name}",
            metadata_json=json.dumps({
                'generated': True,
                'index': index,
                'version': '1.0'
            })
        )

        return factor

    def generate_gas_vectors(self, factor: GeneratedFactor) -> List[GasVector]:
        """Generate gas vectors for a factor."""
        fuel_type = factor.subcategory
        if fuel_type not in self.FUEL_TYPES:
            return []

        fuel_config = self.FUEL_TYPES[fuel_type]
        gas_composition = fuel_config.get('gas_composition', {})

        vectors = []

        # GWP values (AR5)
        gwp_values = {
            'CO2': 1,
            'CH4': 28,
            'N2O': 265
        }

        # Calculate gas breakdown
        total_ef = factor.emission_factor_kg_co2e

        for gas_type, fraction in gas_composition.items():
            # CO2e contribution from this gas
            co2e_contribution = total_ef * fraction

            # Direct emissions (kg of gas per unit)
            gwp = gwp_values[gas_type]
            kg_per_unit = co2e_contribution / gwp

            vectors.append(GasVector(
                factor_id=factor.factor_id,
                gas_type=gas_type,
                kg_per_unit=round(kg_per_unit, 6),
                gwp=gwp
            ))

        return vectors

    def generate_calculation_scenarios(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate calculation test scenarios."""
        scenarios = []

        for i in range(count):
            fuel_type = random.choice(list(self.FUEL_TYPES.keys()))
            fuel_config = self.FUEL_TYPES[fuel_type]

            unit = random.choice(fuel_config['units'])
            amount = random.uniform(10.0, 10000.0)

            # Expected emissions (rough estimate)
            ef_range_key = f"ef_range_{unit.replace('-', '_')}"
            if ef_range_key in fuel_config:
                ef_value = sum(fuel_config[ef_range_key]) / 2  # Use average
            else:
                first_range_key = [k for k in fuel_config.keys() if k.startswith('ef_range')][0]
                ef_value = sum(fuel_config[first_range_key]) / 2

            expected_emissions = amount * ef_value

            scenarios.append({
                'scenario_id': f"scenario_{i:04d}",
                'fuel_type': fuel_type,
                'activity_amount': round(amount, 2),
                'activity_unit': unit,
                'expected_emissions_kg_co2e': round(expected_emissions, 2),
                'emission_factor_used': round(ef_value, 4)
            })

        return scenarios


# ==================== CONVENIENCE FUNCTIONS ====================

def generate_test_database_data(count: int = 500) -> Dict[str, Any]:
    """Generate complete test database data."""
    generator = EmissionFactorGenerator()

    factors = generator.generate_realistic_factors(count)

    # Generate gas vectors for all factors
    all_gas_vectors = []
    for factor in factors:
        vectors = generator.generate_gas_vectors(factor)
        all_gas_vectors.extend(vectors)

    return {
        'factors': [f.to_dict() for f in factors],
        'gas_vectors': [asdict(v) for v in all_gas_vectors],
        'count': len(factors),
        'gas_vector_count': len(all_gas_vectors)
    }


def save_test_data_to_json(filename: str, count: int = 500):
    """Save test data to JSON file."""
    data = generate_test_database_data(count)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Generated {count} factors with {data['gas_vector_count']} gas vectors")
    print(f"Saved to: {filename}")


# ==================== EXAMPLE USAGE ====================

if __name__ == '__main__':
    # Generate 500 factors
    generator = EmissionFactorGenerator()

    factors = generator.generate_realistic_factors(500)

    print(f"Generated {len(factors)} emission factors")
    print("\nSample factors:")

    for factor in factors[:5]:
        print(f"\n{factor.factor_id}:")
        print(f"  Name: {factor.name}")
        print(f"  EF: {factor.emission_factor_kg_co2e} kg CO2e per {factor.unit}")
        print(f"  Geography: {factor.geographic_scope}")
        print(f"  Source: {factor.source_org}")
        print(f"  Quality: {factor.data_quality_tier} (±{factor.uncertainty_percent}%)")

        # Show gas vectors
        vectors = generator.generate_gas_vectors(factor)
        if vectors:
            print(f"  Gas Vectors:")
            for vector in vectors:
                print(f"    {vector.gas_type}: {vector.kg_per_unit} kg per {factor.unit}")

    # Generate calculation scenarios
    scenarios = generator.generate_calculation_scenarios(10)

    print(f"\n\nGenerated {len(scenarios)} calculation scenarios")
    print("\nSample scenarios:")

    for scenario in scenarios[:3]:
        print(f"\n{scenario['scenario_id']}:")
        print(f"  Fuel: {scenario['fuel_type']}")
        print(f"  Amount: {scenario['activity_amount']} {scenario['activity_unit']}")
        print(f"  Expected: {scenario['expected_emissions_kg_co2e']:.2f} kg CO2e")

    # Save to file
    save_test_data_to_json('test_emission_factors_500.json', 500)
