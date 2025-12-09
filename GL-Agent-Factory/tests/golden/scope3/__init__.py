"""
Scope 3 Emissions Agent Golden Tests

This package contains 125+ golden tests for the GL-006 Scope 3 Emissions Agent.
Tests validate value chain emissions calculations across all 15 GHG Protocol categories.

Test Files:
- test_category_1_purchased_goods.yaml: 30 tests for Category 1
- test_category_4_transportation.yaml: 30 tests for Category 4
- test_category_6_business_travel.yaml: 20 tests for Category 6
- test_all_15_categories.yaml: 45 tests covering all categories

Emission Factor Sources:
- EPA EEIO (spend-based)
- GLEC Framework (transport)
- DEFRA (travel and waste)

Key Calculation Formulas:
- Spend-based: emissions = spend_usd * eeio_factor
- Transport: emissions = distance_km * weight_tonnes * transport_factor
- Travel: emissions = distance_km * travel_factor
"""
