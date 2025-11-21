# -*- coding: utf-8 -*-
# GL-VCCI Data Module
# Emission factor databases and data management

"""
VCCI Data Module
================

Emission factor databases and reference data for Scope 3 calculations.

Databases:
----------
1. DEFRA 2024 (free, public)
   - UK Government emission factors
   - 5,000+ factors
   - Updated annually

2. EPA 2024 (free, public)
   - US EPA emission factors
   - 8,000+ factors
   - Updated annually

3. Ecoinvent v3.10 (licensed, $60K/year)
   - LCA database
   - 20,000+ processes
   - Gold standard for LCA

4. Custom Emission Factors
   - Company-specific data
   - Supplier-specific data
   - Industry averages

Data Structure:
--------------
```python
{
  "factor_id": "defra_2024_electricity_uk_grid",
  "category": 3,  # Fuel & Energy-Related Activities
  "scope": "scope3",
  "name": "Electricity - UK Grid Average",
  "value": 0.193,  # kgCO2e/kWh
  "unit": "kgCO2e/kWh",
  "uncertainty": 0.10,  # ±10%
  "source": "DEFRA",
  "year": 2024,
  "region": "GB",
  "data_quality": 95,  # 0-100 score
  "last_updated": "2024-06-01"
}
```

Usage:
------
```python
from data import EmissionFactorDB

# Initialize database
efdb = EmissionFactorDB(sources=["defra", "epa", "ecoinvent"])

# Query emission factor
ef = efdb.get_factor(
    category=1,
    product="Steel",
    region="EU",
    year=2024
)

# Search by keyword
results = efdb.search("electricity grid")
```
"""

__version__ = "1.0.0"

__all__ = [
    # "EmissionFactorDB",
    # "load_defra_factors",
    # "load_epa_factors",
    # "load_ecoinvent_factors",
]
