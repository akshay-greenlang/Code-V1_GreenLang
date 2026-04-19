# -*- coding: utf-8 -*-
"""
GreenLang Sample Data Generator

Generates realistic synthetic data for testing, demos, and development.
Produces CBAM declarations, emissions records, energy consumption, and activity data.
"""

from typing import List, Optional
from decimal import Decimal
from datetime import datetime, date, timedelta
import random
from uuid import uuid4

from contracts import (
    CBAMDataContract,
    EmissionsDataContract,
    EnergyDataContract,
    ActivityDataContract,
    CBAMProductCategory,
    GHGScope,
    EmissionFactorSource,
    DataQualityLevel,
    EnergyType,
    ActivityType
)


# ============================================================================
# SAMPLE DATA GENERATOR
# ============================================================================

class SampleDataGenerator:
    """
    Generate synthetic data for GreenLang testing and demos.

    All data is randomly generated but realistic and internally consistent.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed:
            random.seed(seed)

    # ========================================================================
    # CBAM SAMPLE DATA
    # ========================================================================

    def generate_cbam_samples(
        self,
        count: int = 10,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[CBAMDataContract]:
        """
        Generate synthetic CBAM import declarations.

        Args:
            count: Number of samples to generate
            start_date: Start date for import dates (default: 90 days ago)
            end_date: End date for import dates (default: today)

        Returns:
            List of CBAM data contracts
        """
        if not start_date:
            start_date = date.today() - timedelta(days=90)
        if not end_date:
            end_date = date.today()

        samples = []

        for _ in range(count):
            import_date = self._random_date(start_date, end_date)
            quarter = (import_date.month - 1) // 3 + 1
            declaration_period = f"{import_date.year}-Q{quarter}"

            # Random product
            product_category = random.choice(list(CBAMProductCategory))

            # Product-specific parameters
            product_configs = {
                CBAMProductCategory.CEMENT: {
                    'cn_codes': ['25231000', '25232100', '25232900'],
                    'quantity_range': (100, 5000),
                    'specific_emissions_range': (0.5, 0.9),  # tCO2e/tonne
                    'description': 'Portland cement'
                },
                CBAMProductCategory.IRON_STEEL: {
                    'cn_codes': ['72071100', '72071900', '72072000'],
                    'quantity_range': (500, 10000),
                    'specific_emissions_range': (1.5, 2.5),
                    'description': 'Semi-finished products of iron or non-alloy steel'
                },
                CBAMProductCategory.ALUMINIUM: {
                    'cn_codes': ['76011000', '76012000', '76020000'],
                    'quantity_range': (100, 3000),
                    'specific_emissions_range': (8.0, 12.0),
                    'description': 'Unwrought aluminium'
                },
                CBAMProductCategory.FERTILIZERS: {
                    'cn_codes': ['31021000', '31022100', '31051000'],
                    'quantity_range': (200, 8000),
                    'specific_emissions_range': (2.0, 4.0),
                    'description': 'Urea fertilizers'
                },
                CBAMProductCategory.ELECTRICITY: {
                    'cn_codes': ['27160000', '27160000', '27160000'],
                    'quantity_range': (10000, 100000),
                    'specific_emissions_range': (0.3, 0.8),
                    'description': 'Electrical energy (MWh)'
                },
                CBAMProductCategory.HYDROGEN: {
                    'cn_codes': ['28041000', '28041000', '28041000'],
                    'quantity_range': (50, 2000),
                    'specific_emissions_range': (8.0, 15.0),
                    'description': 'Hydrogen gas'
                }
            }

            config = product_configs[product_category]

            # Random values within product-specific ranges
            quantity = Decimal(str(random.uniform(*config['quantity_range']))).quantize(Decimal('0.001'))
            specific_emissions = Decimal(str(random.uniform(*config['specific_emissions_range']))).quantize(Decimal('0.0001'))

            total_embedded_emissions = (quantity * specific_emissions).quantize(Decimal('0.001'))

            # Split into direct and indirect (typically 80-90% direct for manufacturing)
            direct_ratio = Decimal(str(random.uniform(0.75, 0.92)))
            direct_emissions = (total_embedded_emissions * direct_ratio).quantize(Decimal('0.001'))
            indirect_emissions = (total_embedded_emissions - direct_emissions).quantize(Decimal('0.001'))

            # Random origin country (common CBAM import sources)
            origin_countries = ['CN', 'IN', 'RU', 'TR', 'UA', 'KR', 'BR', 'SA']
            country_of_origin = random.choice(origin_countries)

            # Quality and verification
            quality_levels = [DataQualityLevel.EXCELLENT, DataQualityLevel.GOOD, DataQualityLevel.FAIR]
            data_quality = random.choice(quality_levels)

            is_verified = random.choice([True, False])
            verifier_name = random.choice([
                'Bureau Veritas', 'DNV', 'SGS', 'TUV SUD', 'Lloyd\'s Register'
            ]) if is_verified else None

            verification_date = import_date + timedelta(days=random.randint(10, 30)) if is_verified else None

            sample = CBAMDataContract(
                importer_id=f"GB{random.randint(100000000000, 999999999999)}",
                import_date=import_date,
                declaration_period=declaration_period,
                product_category=product_category,
                cn_code=random.choice(config['cn_codes']),
                product_description=config['description'],
                quantity=quantity,
                quantity_unit='tonnes' if product_category != CBAMProductCategory.ELECTRICITY else 'MWh',
                country_of_origin=country_of_origin,
                installation_id=f"INST-{random.randint(1000, 9999)}" if random.random() > 0.3 else None,
                direct_emissions_co2e=direct_emissions,
                indirect_emissions_co2e=indirect_emissions,
                total_embedded_emissions=total_embedded_emissions,
                specific_emissions=specific_emissions,
                emission_factor_source=random.choice(list(EmissionFactorSource)),
                methodology=random.choice(['ISO 14064-1', 'GHG Protocol', 'EN 19694-1']),
                is_verified=is_verified,
                verifier_name=verifier_name,
                verification_date=verification_date,
                data_quality_level=data_quality,
                uncertainty_percentage=Decimal(str(random.uniform(5, 20))).quantize(Decimal('0.1')) if random.random() > 0.5 else None
            )

            samples.append(sample)

        return samples

    # ========================================================================
    # EMISSIONS SAMPLE DATA
    # ========================================================================

    def generate_emissions_samples(
        self,
        count: int = 20,
        organization_id: str = "ORG-DEMO-001",
        year: int = 2024
    ) -> List[EmissionsDataContract]:
        """
        Generate synthetic emissions records.

        Args:
            count: Number of samples
            organization_id: Organization identifier
            year: Reporting year

        Returns:
            List of emissions data contracts
        """
        samples = []

        # Emission sources by scope
        scope_sources = {
            GHGScope.SCOPE_1: [
                ('Natural gas combustion', ActivityType.FUEL_COMBUSTION, 'natural_gas', 0.184, 'kWh'),
                ('Diesel combustion (generators)', ActivityType.FUEL_COMBUSTION, 'diesel', 0.267, 'liter'),
                ('Gasoline (fleet vehicles)', ActivityType.FUEL_COMBUSTION, 'gasoline', 0.242, 'liter'),
                ('Refrigerant leakage (R-134a)', ActivityType.REFRIGERANT_LEAKAGE, 'refrigerant', 1.430, 'kg')
            ],
            GHGScope.SCOPE_2: [
                ('Purchased electricity', ActivityType.ELECTRICITY_CONSUMPTION, 'grid_electricity', 0.475, 'kWh'),
                ('Purchased steam', ActivityType.ELECTRICITY_CONSUMPTION, 'district_steam', 0.123, 'kg')
            ],
            GHGScope.SCOPE_3: [
                ('Business travel - air', ActivityType.TRANSPORT, 'air_travel', 0.255, 'km'),
                ('Employee commuting', ActivityType.TRANSPORT, 'car_travel', 0.171, 'km'),
                ('Waste disposal', ActivityType.WASTE_DISPOSAL, 'mixed_waste', 0.021, 'kg'),
                ('Purchased goods', ActivityType.MATERIAL_PROCESSING, 'materials', 1.5, 'kg')
            ]
        }

        for _ in range(count):
            # Random scope and source
            scope = random.choice(list(GHGScope))
            source_info = random.choice(scope_sources[scope])
            emission_source, activity_type, fuel_type, base_ef, activity_unit = source_info

            # Random facility
            facilities = ['PLANT-A', 'PLANT-B', 'HQ-OFFICE', 'WAREHOUSE-1', 'WAREHOUSE-2']
            facility_id = random.choice(facilities)

            # Random month in year
            month = random.randint(1, 12)
            period_start = date(year, month, 1)
            if month == 12:
                period_end = date(year, 12, 31)
            else:
                period_end = date(year, month + 1, 1) - timedelta(days=1)

            # Activity amount
            activity_ranges = {
                'kWh': (10000, 100000),
                'liter': (500, 5000),
                'km': (1000, 50000),
                'kg': (100, 10000)
            }
            activity_range = activity_ranges.get(activity_unit, (100, 10000))
            activity_amount = Decimal(str(random.uniform(*activity_range))).quantize(Decimal('0.001'))

            # Emission factor (add some variation)
            emission_factor = Decimal(str(base_ef * random.uniform(0.9, 1.1))).quantize(Decimal('0.000001'))

            # Calculate total emissions
            total_co2e = (activity_amount * emission_factor).quantize(Decimal('0.001'))

            # Break down by gas (simplified)
            co2_tonnes = (total_co2e * Decimal('0.95')).quantize(Decimal('0.001'))
            ch4_tonnes = (total_co2e * Decimal('0.03')).quantize(Decimal('0.000001'))
            n2o_tonnes = (total_co2e * Decimal('0.02')).quantize(Decimal('0.000001'))

            # Location
            countries = ['US', 'GB', 'DE', 'FR', 'JP', 'CA']
            location_country = random.choice(countries)

            sample = EmissionsDataContract(
                organization_id=organization_id,
                facility_id=facility_id,
                reporting_period_start=period_start,
                reporting_period_end=period_end,
                ghg_scope=scope,
                emission_source=emission_source,
                activity_type=activity_type,
                co2_tonnes=co2_tonnes,
                ch4_tonnes=ch4_tonnes,
                n2o_tonnes=n2o_tonnes,
                total_co2e_tonnes=total_co2e,
                activity_amount=activity_amount,
                activity_unit=activity_unit,
                emission_factor_value=emission_factor,
                emission_factor_unit=f"kgCO2e/{activity_unit}",
                emission_factor_source=random.choice([EmissionFactorSource.DEFRA_2024, EmissionFactorSource.EPA_EGRID_2023]),
                location_country=location_country,
                data_quality_level=random.choice([DataQualityLevel.EXCELLENT, DataQualityLevel.GOOD]),
                uncertainty_percentage=Decimal(str(random.uniform(5, 15))).quantize(Decimal('0.1')),
                is_assured=random.choice([True, False]),
                assurance_level=random.choice(['limited', 'reasonable']) if random.random() > 0.5 else None,
                calculation_method=random.choice(['GHG Protocol', 'ISO 14064-1', 'DEFRA Guidelines'])
            )

            samples.append(sample)

        return samples

    # ========================================================================
    # ENERGY SAMPLE DATA
    # ========================================================================

    def generate_energy_samples(
        self,
        count: int = 15,
        organization_id: str = "ORG-DEMO-001",
        year: int = 2024
    ) -> List[EnergyDataContract]:
        """
        Generate synthetic energy consumption records.

        Args:
            count: Number of samples
            organization_id: Organization identifier
            year: Consumption year

        Returns:
            List of energy data contracts
        """
        samples = []

        energy_configs = {
            EnergyType.ELECTRICITY: {
                'amount_range': (10000, 150000),
                'unit': 'kWh',
                'cost_per_unit': 0.12,
                'ef_location': 0.475,
                'ef_market': 0.500
            },
            EnergyType.NATURAL_GAS: {
                'amount_range': (5000, 80000),
                'unit': 'kWh',
                'cost_per_unit': 0.05,
                'ef_location': 0.184,
                'ef_market': 0.184
            },
            EnergyType.DIESEL: {
                'amount_range': (500, 5000),
                'unit': 'liters',
                'cost_per_unit': 1.50,
                'ef_location': 2.67,
                'ef_market': 2.67
            }
        }

        for _ in range(count):
            energy_type = random.choice([EnergyType.ELECTRICITY, EnergyType.NATURAL_GAS, EnergyType.DIESEL])
            config = energy_configs[energy_type]

            # Random month
            month = random.randint(1, 12)
            period_start = datetime(year, month, 1)
            if month == 12:
                period_end = datetime(year, 12, 31, 23, 59, 59)
            else:
                next_month = datetime(year, month + 1, 1)
                period_end = next_month - timedelta(seconds=1)

            # Consumption
            consumption = Decimal(str(random.uniform(*config['amount_range']))).quantize(Decimal('0.001'))

            # Cost
            cost = (consumption * Decimal(str(config['cost_per_unit']))).quantize(Decimal('0.01'))

            # Renewable
            is_renewable = random.random() < 0.2  # 20% chance
            renewable_pct = Decimal(str(random.uniform(50, 100))).quantize(Decimal('0.1')) if is_renewable else None

            # Emissions
            scope_2_location = (consumption * Decimal(str(config['ef_location'])) / 1000).quantize(Decimal('0.001'))
            scope_2_market = (consumption * Decimal(str(config['ef_market'])) / 1000).quantize(Decimal('0.001'))

            if is_renewable and renewable_pct:
                scope_2_market = (scope_2_market * (100 - renewable_pct) / 100).quantize(Decimal('0.001'))

            facilities = ['PLANT-A', 'PLANT-B', 'HQ-OFFICE']
            sample = EnergyDataContract(
                organization_id=organization_id,
                facility_id=random.choice(facilities),
                meter_id=f"MTR-{random.randint(1000, 9999)}",
                consumption_period_start=period_start,
                consumption_period_end=period_end,
                energy_type=energy_type,
                consumption_amount=consumption,
                consumption_unit=config['unit'],
                energy_cost=cost,
                currency='USD',
                is_renewable=is_renewable,
                renewable_percentage=renewable_pct,
                has_green_certificate=is_renewable and random.random() > 0.5,
                grid_region=random.choice(['WECC', 'ERCOT', 'NPCC']) if energy_type == EnergyType.ELECTRICITY else None,
                supplier_name=random.choice(['City Power', 'Green Energy Co', 'National Grid']),
                scope_2_location_based_co2e=scope_2_location,
                scope_2_market_based_co2e=scope_2_market,
                data_source=random.choice(['utility_bill', 'meter_reading']),
                data_quality_level=random.choice([DataQualityLevel.EXCELLENT, DataQualityLevel.GOOD]),
                location_country=random.choice(['US', 'GB', 'DE']),
                location_region=random.choice(['CA', 'TX', 'NY'])
            )

            samples.append(sample)

        return samples

    # ========================================================================
    # ACTIVITY SAMPLE DATA
    # ========================================================================

    def generate_activity_samples(
        self,
        count: int = 25,
        organization_id: str = "ORG-DEMO-001",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[ActivityDataContract]:
        """
        Generate synthetic activity data records.

        Args:
            count: Number of samples
            organization_id: Organization identifier
            start_date: Start date (default: 30 days ago)
            end_date: End date (default: today)

        Returns:
            List of activity data contracts
        """
        if not start_date:
            start_date = date.today() - timedelta(days=30)
        if not end_date:
            end_date = date.today()

        samples = []

        activity_configs = {
            ActivityType.FUEL_COMBUSTION: {
                'descriptions': [
                    'Natural gas combustion in boiler',
                    'Diesel combustion in backup generator',
                    'Propane use in forklift'
                ],
                'amount_range': (100, 5000),
                'units': ['m3', 'liters', 'kg']
            },
            ActivityType.ELECTRICITY_CONSUMPTION: {
                'descriptions': [
                    'Grid electricity consumption - main meter',
                    'Electricity consumption - production line',
                    'Office building electricity use'
                ],
                'amount_range': (1000, 50000),
                'units': ['kWh']
            },
            ActivityType.TRANSPORT: {
                'descriptions': [
                    'Freight transport by truck',
                    'Employee business travel by car',
                    'Courier delivery services'
                ],
                'amount_range': (50, 2000),
                'units': ['km']
            },
            ActivityType.MATERIAL_PROCESSING: {
                'descriptions': [
                    'Steel processing in production',
                    'Plastic molding operations',
                    'Aluminum extrusion'
                ],
                'amount_range': (500, 10000),
                'units': ['kg', 'tonnes']
            }
        }

        for _ in range(count):
            activity_type = random.choice(list(ActivityType))
            config = activity_configs.get(activity_type, {
                'descriptions': ['General activity'],
                'amount_range': (100, 1000),
                'units': ['units']
            })

            activity_date = self._random_date(start_date, end_date)
            description = random.choice(config['descriptions'])
            amount = Decimal(str(random.uniform(*config['amount_range']))).quantize(Decimal('0.001'))
            unit = random.choice(config['units'])

            sample = ActivityDataContract(
                organization_id=organization_id,
                facility_id=random.choice(['PLANT-A', 'PLANT-B', 'WAREHOUSE-1']),
                activity_date=activity_date,
                activity_type=activity_type,
                activity_description=description,
                activity_amount=amount,
                activity_unit=unit,
                asset_id=f"ASSET-{random.randint(100, 999)}",
                data_source=random.choice(['direct_measurement', 'utility_bill', 'invoice']),
                data_quality_level=random.choice([DataQualityLevel.EXCELLENT, DataQualityLevel.GOOD, DataQualityLevel.FAIR]),
                is_estimated=random.choice([True, False]),
                location_country=random.choice(['US', 'GB', 'DE'])
            )

            samples.append(sample)

        return samples

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _random_date(self, start: date, end: date) -> date:
        """Generate random date between start and end."""
        delta = end - start
        random_days = random.randint(0, delta.days)
        return start + timedelta(days=random_days)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_cbam_sample(count: int = 10, seed: Optional[int] = None) -> List[CBAMDataContract]:
    """
    Generate CBAM sample data.

    Args:
        count: Number of samples
        seed: Random seed

    Returns:
        List of CBAM contracts
    """
    generator = SampleDataGenerator(seed=seed)
    return generator.generate_cbam_samples(count=count)


def generate_emissions_sample(count: int = 20, seed: Optional[int] = None) -> List[EmissionsDataContract]:
    """
    Generate emissions sample data.

    Args:
        count: Number of samples
        seed: Random seed

    Returns:
        List of emissions contracts
    """
    generator = SampleDataGenerator(seed=seed)
    return generator.generate_emissions_samples(count=count)


def generate_energy_sample(count: int = 15, seed: Optional[int] = None) -> List[EnergyDataContract]:
    """
    Generate energy sample data.

    Args:
        count: Number of samples
        seed: Random seed

    Returns:
        List of energy contracts
    """
    generator = SampleDataGenerator(seed=seed)
    return generator.generate_energy_samples(count=count)
