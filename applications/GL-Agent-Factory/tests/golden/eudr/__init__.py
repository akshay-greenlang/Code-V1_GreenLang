"""
EUDR Compliance Agent Golden Tests

This package contains 100+ golden tests for the GL-004 EUDR Compliance Agent.
Tests validate deforestation-free compliance per EU Regulation 2023/1115.

Test Files:
- test_geojson_validation.yaml: 40 tests for geolocation validation
- test_commodity_risk.yaml: 35 tests for commodity risk assessment
- test_deforestation_detection.yaml: 25 tests for compliance determination

Country Risk Scores (EU provisional assessment):
- Brazil (BR): High risk, score 75.0
- Indonesia (ID): High risk, score 72.0
- Malaysia (MY): High risk, score 68.0
- Colombia (CO): Standard risk, score 55.0
- Peru (PE): Standard risk, score 50.0
- Ghana (GH): Standard risk, score 45.0
- Cote d'Ivoire (CI): Standard risk, score 48.0

EUDR Cutoff Date: December 31, 2020
"""
