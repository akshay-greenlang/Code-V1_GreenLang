# GreenLang Data Engineering Module

Enterprise-grade data contracts, ETL pipelines, and quality assurance for carbon accounting and sustainability data.

## Overview

This module provides:

1. **Data Contracts** - Pydantic models with strict validation
2. **Emission Factor Loader** - Load DEFRA, EPA eGRID, and custom factors
3. **Data Quality Framework** - Comprehensive quality scoring (0-100%)
4. **Sample Data Generator** - Synthetic data for testing and demos

## Components

### 1. Data Contracts (`contracts.py`)

Four core data contracts with complete field validation:

#### CBAMDataContract
EU Carbon Border Adjustment Mechanism import declarations.

```python
from greenlang.data import CBAMDataContract
from decimal import Decimal
from datetime import date

cbam_data = CBAMDataContract(
    importer_id="GB123456789000",
    import_date=date(2024, 3, 15),
    declaration_period="2024-Q1",
    product_category="iron_steel",
    cn_code="72071100",
    product_description="Semi-finished steel products",
    quantity=Decimal("1000.000"),
    quantity_unit="tonnes",
    country_of_origin="CN",
    direct_emissions_co2e=Decimal("1800.000"),
    indirect_emissions_co2e=Decimal("200.000"),
    total_embedded_emissions=Decimal("2000.000"),
    specific_emissions=Decimal("2.0000"),
    emission_factor_source="defra_2024",
    methodology="ISO 14064-1",
    is_verified=True,
    data_quality_level="excellent"
)
```

**Key Features:**
- Automatic validation of emissions calculations
- CN code format validation
- Country code validation (ISO 3166-1 alpha-2)
- Verification workflow tracking

#### EmissionsDataContract
GHG Protocol compliant emissions tracking (Scope 1, 2, 3).

```python
from greenlang.data import EmissionsDataContract, GHGScope
from decimal import Decimal
from datetime import date

emissions = EmissionsDataContract(
    organization_id="ORG-12345",
    facility_id="PLANT-A",
    reporting_period_start=date(2024, 1, 1),
    reporting_period_end=date(2024, 12, 31),
    ghg_scope=GHGScope.SCOPE_1,
    emission_source="Natural gas combustion",
    activity_type="fuel_combustion",
    co2_tonnes=Decimal("1500.000"),
    ch4_tonnes=Decimal("0.150"),
    n2o_tonnes=Decimal("0.030"),
    total_co2e_tonnes=Decimal("1520.000"),
    activity_amount=Decimal("800000.000"),
    activity_unit="kWh",
    emission_factor_value=Decimal("0.001900"),
    emission_factor_unit="tCO2e/kWh",
    emission_factor_source="defra_2024",
    location_country="GB",
    data_quality_level="good",
    calculation_method="GHG Protocol"
)
```

**Key Features:**
- Breakdown by 7 GHG gases (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)
- Automatic calculation validation
- GHG scope tracking
- External assurance tracking

#### EnergyDataContract
Energy consumption tracking with Scope 2 emissions.

```python
from greenlang.data import EnergyDataContract, EnergyType
from decimal import Decimal
from datetime import datetime

energy = EnergyDataContract(
    organization_id="ORG-12345",
    facility_id="PLANT-A",
    consumption_period_start=datetime(2024, 1, 1),
    consumption_period_end=datetime(2024, 1, 31, 23, 59, 59),
    energy_type=EnergyType.ELECTRICITY,
    consumption_amount=Decimal("50000.000"),
    consumption_unit="kWh",
    energy_cost=Decimal("8500.00"),
    currency="USD",
    is_renewable=False,
    scope_2_location_based_co2e=Decimal("22.500"),
    scope_2_market_based_co2e=Decimal("25.000"),
    data_source="utility_bill",
    data_quality_level="excellent",
    location_country="US",
    location_region="CA"
)
```

**Key Features:**
- Renewable energy tracking with certificates (RECs/GOs)
- Dual Scope 2 reporting (location-based and market-based)
- Cost tracking
- Grid region support

#### ActivityDataContract
Raw activity data (fuel, electricity, transport, materials).

```python
from greenlang.data import ActivityDataContract, ActivityType
from decimal import Decimal
from datetime import date

activity = ActivityDataContract(
    organization_id="ORG-12345",
    facility_id="PLANT-A",
    activity_date=date(2024, 1, 15),
    activity_type=ActivityType.FUEL_COMBUSTION,
    activity_description="Natural gas combustion in boiler",
    activity_amount=Decimal("15000.000"),
    activity_unit="m3",
    fuel_type="natural_gas",
    data_source="direct_measurement",
    data_quality_level="excellent",
    location_country="US"
)
```

### 2. Emission Factor Loader (`emission_factors.py`)

Load emission factors from authoritative sources into PostgreSQL.

```python
from greenlang.data import EmissionFactorLoader
from pathlib import Path

# Initialize loader
loader = EmissionFactorLoader(
    db_connection_string="postgresql://user:pass@localhost/greenlang"
)
await loader.initialize()

# Load DEFRA 2024 factors
defra_count = await loader.load_defra_2024(
    csv_path=Path("data/DEFRA_2024_Conversion_Factors.csv")
)
print(f"Loaded {defra_count} DEFRA emission factors")

# Load EPA eGRID 2023 factors
egrid_count = await loader.load_epa_egrid_2023(
    excel_path=Path("data/eGRID2023_Data.xlsx")
)
print(f"Loaded {egrid_count} EPA eGRID factors")

# Search factors
natural_gas_factors = await loader.search_factors(
    activity_name="natural gas",
    country_code="GB",
    limit=10
)

for factor in natural_gas_factors:
    print(f"{factor.activity_name}: {factor.co2e_factor} {factor.unit_numerator}/{factor.unit_denominator}")

await loader.close()
```

**Supported Sources:**
- DEFRA 2024 (UK Government GHG Conversion Factors)
- EPA eGRID 2023 (US Electricity Grid Emissions)
- Custom CSV uploads

**Database Schema:**
```sql
CREATE TABLE emission_factors (
    id UUID PRIMARY KEY,
    source VARCHAR(100),
    source_year INTEGER,
    category VARCHAR(200),
    activity_name VARCHAR(500),
    geographic_scope VARCHAR(50),
    country_code VARCHAR(2),
    grid_region VARCHAR(100),
    co2_factor NUMERIC(12, 6),
    ch4_factor NUMERIC(12, 9),
    n2o_factor NUMERIC(12, 9),
    co2e_factor NUMERIC(12, 6),
    unit_numerator VARCHAR(50),
    unit_denominator VARCHAR(50),
    quality_rating VARCHAR(20),
    valid_from DATE,
    valid_to DATE,
    ...
);
```

### 3. Data Quality Framework (`quality.py`)

Comprehensive data quality scoring across 5 dimensions.

```python
from greenlang.data import check_data_quality, DataQualityChecker

# Quick check (convenience function)
report = check_data_quality(cbam_data)

print(f"Overall Score: {report.overall_score}/100")
print(f"Quality Level: {report.quality_level}")
print(f"Checks Passed: {report.checks_passed}/{report.total_checks}")

if report.critical_issues:
    print("CRITICAL ISSUES:")
    for issue in report.critical_issues:
        print(f"  - {issue.message}")

if report.recommendations:
    print("\nRECOMMENDATIONS:")
    for rec in report.recommendations:
        print(f"  - {rec}")

# Advanced usage
checker = DataQualityChecker(config={
    'strict_mode': True,
    'custom_thresholds': {
        'completeness': 95.0,
        'accuracy': 90.0
    }
})

report = checker.check_cbam_data(cbam_data)
```

**Quality Dimensions:**

1. **Completeness** (30% weight)
   - All required fields populated
   - Optional fields filled

2. **Accuracy** (30% weight)
   - Values within expected ranges
   - Format validation (CN codes, country codes)
   - Reasonable emission factors

3. **Consistency** (25% weight)
   - Calculated fields match (total = direct + indirect)
   - Specific emissions = total / quantity
   - Verification details complete

4. **Timeliness** (10% weight)
   - Dates not in future
   - Reporting periods match activities
   - Verification dates logical

5. **Uniqueness** (5% weight)
   - No duplicate records

**Quality Levels:**
- **Excellent**: 90-100% (High confidence, suitable for reporting)
- **Good**: 70-89% (Acceptable, minor improvements needed)
- **Fair**: 50-69% (Usable but significant issues)
- **Poor**: <50% (Data quality unacceptable)

**Report Structure:**
```python
{
    "record_id": "uuid",
    "data_type": "cbam",
    "overall_score": 92.5,
    "quality_level": "excellent",
    "completeness_score": 95.0,
    "accuracy_score": 90.0,
    "consistency_score": 100.0,
    "timeliness_score": 85.0,
    "uniqueness_score": 100.0,
    "checks_passed": 18,
    "checks_failed": 2,
    "total_checks": 20,
    "critical_issues": [],
    "high_issues": [
        {
            "dimension": "accuracy",
            "check_name": "specific_emissions_range",
            "severity": "high",
            "message": "Specific emissions outside typical range"
        }
    ],
    "recommendations": [
        "Complete all required fields",
        "Consider third-party verification"
    ]
}
```

### 4. Sample Data Generator (`sample_data.py`)

Generate realistic synthetic data for testing and demos.

```python
from greenlang.data import SampleDataGenerator

# Initialize with seed for reproducibility
generator = SampleDataGenerator(seed=42)

# Generate CBAM data
cbam_samples = generator.generate_cbam_samples(count=100)
print(f"Generated {len(cbam_samples)} CBAM declarations")

# Generate emissions data
emissions_samples = generator.generate_emissions_samples(
    count=200,
    organization_id="DEMO-ORG",
    year=2024
)

# Generate energy data
energy_samples = generator.generate_energy_samples(
    count=150,
    organization_id="DEMO-ORG"
)

# Generate activity data
from datetime import date, timedelta

activity_samples = generator.generate_activity_samples(
    count=500,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

# Convenience functions (no seed control)
from greenlang.data import (
    generate_cbam_sample,
    generate_emissions_sample,
    generate_energy_sample
)

cbam_data = generate_cbam_sample(count=50)
emissions_data = generate_emissions_sample(count=100)
energy_data = generate_energy_sample(count=75)
```

**Sample Data Characteristics:**
- Internally consistent (calculations match)
- Realistic value ranges by product/activity type
- Geographic diversity
- Temporal distribution
- Quality variation (excellent to fair)
- Verification status variation

## Usage Examples

### Complete Workflow Example

```python
import asyncio
from greenlang.data import (
    SampleDataGenerator,
    EmissionFactorLoader,
    DataQualityChecker,
    check_data_quality
)
from pathlib import Path

async def main():
    # 1. Load emission factors
    loader = EmissionFactorLoader("postgresql://localhost/greenlang")
    await loader.initialize()

    defra_count = await loader.load_defra_2024(
        Path("data/DEFRA_2024.csv")
    )
    print(f"Loaded {defra_count} emission factors")

    # 2. Generate sample data
    generator = SampleDataGenerator(seed=42)
    cbam_samples = generator.generate_cbam_samples(count=100)

    # 3. Quality check all samples
    checker = DataQualityChecker()
    acceptable_data = []

    for sample in cbam_samples:
        report = checker.check_cbam_data(sample)

        if report.overall_score >= 70.0:
            acceptable_data.append(sample)
        else:
            print(f"Low quality: {sample.id} - Score: {report.overall_score}")

    print(f"Acceptable data: {len(acceptable_data)}/{len(cbam_samples)}")

    # 4. Lookup emission factors
    factors = await loader.search_factors(
        activity_name="steel",
        country_code="GB"
    )

    for factor in factors[:5]:
        print(f"{factor.activity_name}: {factor.co2e_factor}")

    await loader.close()

asyncio.run(main())
```

### Batch Quality Assessment

```python
from greenlang.data import generate_cbam_sample, DataQualityChecker
import pandas as pd

# Generate samples
samples = generate_cbam_sample(count=1000)

# Batch quality check
checker = DataQualityChecker()
reports = [checker.check_cbam_data(s) for s in samples]

# Analyze quality distribution
df = pd.DataFrame([
    {
        'record_id': r.record_id,
        'overall_score': r.overall_score,
        'quality_level': r.quality_level,
        'critical_issues': len(r.critical_issues),
        'is_acceptable': r.is_acceptable
    }
    for r in reports
])

print("\nQuality Distribution:")
print(df['quality_level'].value_counts())

print("\nAverage Scores:")
print(f"Overall: {df['overall_score'].mean():.2f}")
print(f"Acceptable rate: {df['is_acceptable'].sum() / len(df) * 100:.1f}%")

# Export low-quality records for review
low_quality = df[df['overall_score'] < 70]
low_quality.to_csv('low_quality_records.csv', index=False)
```

## Installation

```bash
pip install pydantic asyncpg pandas openpyxl httpx
```

## Database Setup

```sql
-- Create database
CREATE DATABASE greenlang;

-- The emission_factors table is created automatically
-- when EmissionFactorLoader.initialize() is called
```

## Data Sources

### DEFRA GHG Conversion Factors
- **Source**: UK Government DEFRA
- **URL**: https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024
- **Coverage**: Fuels, electricity, transport, refrigerants, waste
- **Geographic Scope**: UK (with some international factors)
- **Update Frequency**: Annual

### EPA eGRID
- **Source**: US Environmental Protection Agency
- **URL**: https://www.epa.gov/egrid
- **Coverage**: US electricity grid emissions
- **Geographic Scope**: US (by subregion)
- **Update Frequency**: Annual

## Validation Rules

### CBAM Data
- Total emissions = Direct + Indirect (tolerance: 0.001 tCO2e)
- Specific emissions = Total / Quantity (tolerance: 0.0001 tCO2e/unit)
- CN code: 8 digits
- Country code: ISO 3166-1 alpha-2 (2 letters)
- Import date: Not in future
- Declaration period: Matches import date quarter

### Emissions Data
- Calculated emissions = Activity × Emission Factor (tolerance: 5%)
- Reporting period end > start
- All dates not in future
- At least one GHG value > 0

### Energy Data
- Consumption amount > 0
- Period end > start
- If renewable, renewable_percentage must be set
- Scope 2 emissions calculated from consumption

### Activity Data
- Activity amount > 0
- Activity date not in future

## Performance

- **Emission factor lookup**: <10ms (indexed queries)
- **Quality check**: <50ms per record
- **Sample generation**: 1000 records/second
- **Bulk factor loading**: 10,000 factors/minute

## Best Practices

1. **Always validate data** before storing in database
2. **Use quality scores** to filter unreliable data
3. **Track data lineage** via metadata fields
4. **Version emission factors** via valid_from/valid_to
5. **Regular quality audits** of incoming data
6. **Document custom emission factors** thoroughly

## License

Copyright GreenLang 2024. All rights reserved.
