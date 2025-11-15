"""
GL-002 BoilerEfficiencyOptimizer Test Suite

Comprehensive test suite for GL-002 with ≥85% coverage target.
Tests all components including orchestrator, calculators, integrations,
tools, performance, determinism, compliance, and security.

Test Coverage Goals:
- Unit Test Coverage: ≥85%
- Integration Test Coverage: ≥80%
- Performance Test Coverage: All critical paths
- Security Test Coverage: 100% for critical security aspects

Total Tests Target: 180+ tests minimum
"""

# Test configuration
TEST_CONFIG = {
    "coverage_target": 85.0,  # Minimum coverage percentage
    "performance_targets": {
        "max_latency_ms": 3000,  # 3 seconds max
        "min_throughput_rps": 100,  # 100 requests per second minimum
        "max_memory_mb": 500,  # 500MB max memory usage
    },
    "determinism": {
        "max_variance": 1e-9,  # Maximum allowed variance for floating point
        "hash_algorithm": "sha256",  # Algorithm for provenance hashing
    },
    "compliance": {
        "standards": ["ISO 50001", "ASME PTC 4", "EN 12952", "EN 12953"],
        "emissions_precision": 6,  # Decimal places for emissions calculations
    },
}

# Test data paths
TEST_DATA_DIR = "test_data"
FIXTURES_DIR = "fixtures"
MOCK_DATA_DIR = "mock_data"

__all__ = [
    "TEST_CONFIG",
    "TEST_DATA_DIR",
    "FIXTURES_DIR",
    "MOCK_DATA_DIR",
]