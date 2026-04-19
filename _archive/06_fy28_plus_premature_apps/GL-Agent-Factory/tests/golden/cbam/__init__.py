"""
CBAM Compliance Agent Golden Tests

This package contains 80+ golden tests for the GL-002 CBAM Compliance Agent.
Tests validate embedded emissions calculations for imported goods per EU Regulation 2023/956.

Test Files:
- test_steel_imports.yaml: 30 tests for iron/steel CN codes
- test_aluminum_imports.yaml: 30 tests for aluminum products
- test_cement_imports.yaml: 20 tests for cement products

Default Emission Factors (EU Implementing Regulation 2023/1773):
- Steel (Global): Direct 1.85 tCO2e/t, Indirect 0.32 tCO2e/t
- Steel (China): Direct 2.10 tCO2e/t, Indirect 0.45 tCO2e/t
- Aluminum (Global): Direct 1.60 tCO2e/t, Indirect 6.50 tCO2e/t
- Cement (Global): Direct 0.83 tCO2e/t, Indirect 0.05 tCO2e/t

EU ETS Carbon Price: 85.0 EUR/tCO2
"""
