# GreenLang Emission Factor Data Migration Guide

## Overview

This guide covers the migration from DEFRA 2023 to DEFRA 2024 emission factors and the integration of EPA eGRID 2023 data for US electricity grid intensity.

## Data Sources

### DEFRA 2024 (UK Government GHG Conversion Factors)

- **Version:** 2024
- **Publication Date:** June 2024
- **Source:** UK Department for Environment, Food & Rural Affairs
- **URL:** https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024
- **GWP Basis:** IPCC AR6 100-year GWP
- **Total Factors:** 4,127+
- **Categories:** Fuels, Electricity, Heat & Steam, Transport, Waste, Water, Materials, Refrigerants, Hotel Stays, WTT Fuels

### EPA eGRID 2023 (US Electricity Grid)

- **Version:** eGRID2023
- **Data Year:** 2022
- **Publication Date:** January 2024
- **Source:** US Environmental Protection Agency
- **URL:** https://www.epa.gov/egrid
- **Coverage:** 26 eGRID subregions, 50+ US states
- **Total Power Plants:** 10,247

## Migration Steps

### 1. Pre-Migration Validation

Run the validation script before migration:

```bash
python scripts/migrate_defra_2024.py --validate-only
```

This checks:
- Source file existence (defra_2023.json)
- Target file existence (defra_2024.json)
- JSON structure validity
- Factor count comparison
- Significant value changes (>10%)

### 2. Backup Creation

The migration script automatically creates backups:

```bash
# Backups are stored in: backups/defra_migration/
# Format: defra_backup_YYYYMMDD_HHMMSS/
```

### 3. Execute Migration

```bash
# Dry run (no changes)
python scripts/migrate_defra_2024.py --dry-run

# Full migration
python scripts/migrate_defra_2024.py --migrate
```

### 4. Rollback (if needed)

```bash
# Rollback to latest backup
python scripts/migrate_defra_2024.py --rollback

# Rollback to specific backup
python scripts/migrate_defra_2024.py --rollback --backup-path backups/defra_migration/defra_backup_20240101_120000
```

## Version Selection in Code

### Default Version (DEFRA 2024)

```python
from greenlang.data.emission_factor_db import get_database, DataVersion

# Default is DEFRA 2024
db = get_database()
factor = db.lookup("natural_gas", "GB", 2024)
```

### Specify Version Explicitly

```python
from greenlang.data.emission_factor_db import get_database, DataVersion

# Use DEFRA 2023 explicitly
db = get_database(version=DataVersion.DEFRA_2023)
factor = db.lookup("diesel", "GB", 2023, version=DataVersion.DEFRA_2023)

# Use EPA eGRID for US grid
factor = db.lookup_grid_intensity("CA")  # California
factor = db.lookup_grid_intensity("CAMX")  # eGRID subregion
```

### Version Comparison

```python
from greenlang.data.emission_factor_db import get_database, DataVersion

db = get_database()
comparison = db.compare_versions(
    fuel_type="diesel",
    region="GB",
    version1=DataVersion.DEFRA_2023,
    version2=DataVersion.DEFRA_2024
)

print(f"2023: {comparison['record1'].ef_value}")
print(f"2024: {comparison['record2'].ef_value}")
print(f"Change: {comparison['percent_change']}%")
```

## Key Changes: DEFRA 2023 vs 2024

### Updated GWP Values (AR6)

| Gas | AR5 (2023) | AR6 (2024) |
|-----|------------|------------|
| CO2 | 1 | 1 |
| CH4 | 28 | 27.9 |
| N2O | 265 | 273 |

### Significant Factor Changes

Notable changes in emission factors between 2023 and 2024:

1. **UK Grid Electricity:** Decreased due to higher renewable penetration
   - 2023: ~225 kgCO2e/MWh
   - 2024: ~190 kgCO2e/MWh

2. **Diesel (UK Average):** Slight increase due to biofuel blend adjustments
   - 2023: 2.67 kgCO2e/L
   - 2024: 2.52 kgCO2e/L

3. **Petrol (UK Average):** Decreased with E10 introduction
   - 2023: 2.31 kgCO2e/L
   - 2024: 2.09 kgCO2e/L

### New Categories in 2024

- Well-to-Tank (WTT) factors expanded
- Electric vehicle factors added
- Sustainable Aviation Fuel (SAF) factors
- Enhanced transport subcategories
- Updated refrigerant factors (HFC phasedown)

## EPA eGRID Integration

### Loading eGRID Data

```python
from greenlang.data.emission_factor_db import get_database

db = get_database()

# Lookup by US state
ca_grid = db.lookup_grid_intensity("CA")
print(f"California: {ca_grid.co2e_kg_per_mwh} kgCO2e/MWh")

# Lookup by eGRID subregion
camx = db.get_egrid_subregion("CAMX")
print(f"WECC California: {camx.co2e_kg_per_mwh} kgCO2e/MWh")

# Access generation mix
print(f"Solar: {camx.generation_mix['solar']}%")
print(f"Natural Gas: {camx.generation_mix['natural_gas']}%")
```

### eGRID Subregion Codes

| Code | Region | States |
|------|--------|--------|
| CAMX | WECC California | CA |
| ERCT | ERCOT (Texas) | TX |
| FRCC | Florida | FL |
| MROE | MRO East | WI, MI |
| MROW | MRO West | MN, IA, ND, SD, NE, MT |
| NEWE | NPCC New England | CT, MA, ME, NH, RI, VT |
| NWPP | WECC Northwest | WA, OR, ID, NV, UT, WY, CO |
| NYUP | NPCC Upstate NY | NY |
| RFCE | RFC East | PA, NJ, DE, MD, DC |
| RFCW | RFC West | OH, IN, WV, KY |
| RMPA | WECC Rockies | CO, WY |
| SRSO | SERC South | AL, GA, FL |
| SRTV | SERC Tennessee Valley | TN, KY, AL, MS, GA |
| SRVC | SERC Virginia/Carolina | VA, NC, SC |

### Command Line Tools

```bash
# Validate eGRID data
python scripts/load_egrid_2023.py --validate

# Get statistics
python scripts/load_egrid_2023.py --stats

# Lookup by state
python scripts/load_egrid_2023.py --lookup CA

# Lookup by subregion
python scripts/load_egrid_2023.py --subregion CAMX

# Export to CSV
python scripts/load_egrid_2023.py --export-csv

# Full report
python scripts/load_egrid_2023.py --report
```

## Data Quality Validation

### Validating Emission Factors

```python
from greenlang.data.quality import check_emission_factor_quality

ef_record = {
    "co2": 56.04,
    "ch4": 0.00102,
    "n2o": 0.00011,
    "co2e_ar6": 56.34,
    "unit": "kgCO2e/GJ",
    "uncertainty": 0.02,
    "quality": "1",
    "citation": "DEFRA 2024 Conversion Factors"
}

report = check_emission_factor_quality(ef_record, "natural_gas", "defra_2024")
print(f"Quality Score: {report.overall_score}")
print(f"Level: {report.quality_level}")
```

### Batch Validation

```python
from greenlang.data.quality import check_emission_factor_batch

records = [...]  # List of emission factor records
metrics = check_emission_factor_batch(records, version="defra_2024")

print(f"Pass Rate: {metrics.pass_rate * 100}%")
print(f"Average Score: {metrics.average_score}")
```

## Redis Cache Configuration

### Enable Caching

```python
from greenlang.cache import get_cache, RedisCacheConfig

config = RedisCacheConfig(
    host="localhost",
    port=6379,
    emission_factor_ttl_seconds=86400,  # 24 hours
    enable_fallback=True  # Fall back to local cache if Redis unavailable
)

cache = get_cache(config)
```

### Cache Invalidation

```python
from greenlang.cache import invalidate_emission_factor

# Invalidate specific factor
invalidate_emission_factor(fuel_type="diesel", region="GB")

# Invalidate by version
invalidate_emission_factor(version="defra_2024")

# Invalidate all
cache.clear()
```

## Troubleshooting

### Common Issues

1. **Factor Not Found**
   - Check fuel type spelling (natural_gas, not naturalGas)
   - Check region code (GB, not UK)
   - Try fallback to GLOBAL region

2. **Version Mismatch**
   - Ensure correct DataVersion enum is used
   - Check that data files exist in factors/ directory

3. **Cache Issues**
   - Clear cache after data updates
   - Check Redis connection if using Redis
   - Verify TTL settings

### Logging

Enable debug logging:

```python
import logging
logging.getLogger('greenlang.data').setLevel(logging.DEBUG)
```

## Support

For issues with the emission factor data:
- DEFRA issues: [UK Government GHG Conversion Factors](https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024)
- EPA eGRID issues: [EPA eGRID](https://www.epa.gov/egrid)
- GreenLang issues: Open an issue on the repository
