"""
GL-001 ThermalCommand Test Suite

Comprehensive testing following the Test Pyramid:
- Unit tests: Schema validation, unit conversion, constraint enforcement
- Simulation tests: Digital twin, dispatch behavior, MILP optimization
- Integration tests: OPC-UA, Kafka, GraphQL/gRPC, CMMS
- Safety tests: Boundary violations, SIS permissives, emergency stop
- Performance tests: Cycle time, throughput, memory usage
- Determinism tests: Reproducibility, SHA-256 provenance

Target: 85%+ coverage

Test Categories:
    test_unit/: Unit tests for individual components
    test_simulation/: Plant surrogate model tests
    test_integration/: External system integration tests
    test_safety/: Safety boundary and SIS tests
    test_performance/: Performance and load tests
    test_determinism/: Reproducibility tests
    test_acceptance/: Acceptance criteria validation
"""

import pytest

# Test configuration
TEST_CONFIG = {
    "coverage_target": 0.85,
    "performance_targets": {
        "optimization_cycle_ms": 5000,
        "data_throughput_pts_per_sec": 10000,
        "memory_mb": 512,
        "api_response_ms": 200,
    },
    "safety_targets": {
        "boundary_violations": 0,
        "sis_override_attempts": 0,
    },
}
