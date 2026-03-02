"""
Unit tests for AGENT-MRV-019 (Business Travel Agent).

Test modules:
- conftest.py: Shared pytest fixtures (~100 lines)
- test_models.py: Enums, constants, input/result models, helpers (55 tests)
- test_config.py: Configuration dataclasses, env loading, singleton (40 tests)
- test_metrics.py: Prometheus metrics, graceful fallback (25 tests)
- test_provenance.py: SHA-256 chain, validation, Merkle tree (30 tests)
- test_business_travel_database.py: EF lookups, airport search, fallback (45 tests)
- test_air_travel_calculator.py: Flight distance-based calculations (45 tests)
- test_ground_transport_calculator.py: Rail/road/bus/taxi/ferry (45 tests)
- test_hotel_stay_calculator.py: Hotel room-night calculations (35 tests)
- test_spend_based_calculator.py: EEIO spend-based calculations (35 tests)
- test_compliance_checker.py: Regulatory compliance checks (40 tests)
- test_business_travel_pipeline.py: End-to-end pipeline (35 tests)
- test_setup.py: Service facade and factory (35 tests)
- test_api.py: API endpoints and routes (40 tests)

Total: 575+ tests
"""
