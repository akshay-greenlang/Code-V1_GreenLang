# -*- coding: utf-8 -*-
# GL-VCCI SDK Module
# Python SDK for programmatic access to VCCI Scope 3 Platform

"""
VCCI Python SDK
===============

High-level Python SDK for programmatic access to the VCCI Scope 3 Platform.

Main Classes:
------------
- Scope3Pipeline: End-to-end pipeline orchestration
- VCCIClient: Low-level API client
- Scope3Calculator: Direct access to calculation engine
- ReportGenerator: Programmatic report generation

Usage:
------
```python
from sdk import Scope3Pipeline

# Initialize pipeline
pipeline = Scope3Pipeline(config_path="config/vcci_config.yaml")

# Run complete Scope 3 analysis
results = pipeline.run(
    procurement_data="procurement.csv",
    logistics_data="logistics.csv",
    supplier_data="suppliers.json"
)

# Access results
print(f"Total Scope 3 emissions: {results.total_emissions:.2f} tCO2e")
print(f"Data coverage (Tier 1/2): {results.data_coverage:.1%}")
print(f"Top hotspot: {results.top_supplier} ({results.top_emissions:.2f} tCO2e)")

# Generate reports
pipeline.generate_report(
    results=results,
    format="ghg-protocol",
    output="scope3_report.pdf"
)
```

Advanced Usage:
--------------
```python
from sdk import VCCIClient, Scope3Calculator

# Low-level API access
client = VCCIClient(api_key="your_api_key")
data = client.intake.upload_file("procurement.csv")

# Direct calculation
calculator = Scope3Calculator()
emissions = calculator.calculate_category_1(
    product="Steel",
    quantity=1000,
    unit="kg"
)
```
"""

__version__ = "1.0.0"

# SDK classes will be exported here when implemented
__all__ = [
    # "Scope3Pipeline",
    # "VCCIClient",
    # "Scope3Calculator",
    # "ReportGenerator",
]
