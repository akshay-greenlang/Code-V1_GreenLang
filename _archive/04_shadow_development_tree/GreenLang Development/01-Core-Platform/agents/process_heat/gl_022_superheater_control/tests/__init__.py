"""
GL-022 SuperheaterControlAgent Test Suite

Comprehensive test coverage for the Superheater Control Agent including:
- Configuration validation tests
- Thermodynamic calculation tests
- PID controller behavior tests
- Safety and ASME compliance tests
- Provenance hash determinism tests

Target: 85%+ code coverage

Test Categories:
- Unit tests: Individual function/method testing
- Integration tests: Component interaction testing
- Performance tests: Throughput and latency validation
- Compliance tests: ASME PTC 4 and IAPWS-IF97 compliance

Standards Tested:
- ASME PTC 4 (Performance Test Code for Fired Steam Generators)
- IAPWS-IF97 (International Association for Properties of Water and Steam)
- IEC 61511 (Safety Instrumented Systems)
"""

__version__ = "1.0.0"
__author__ = "GreenLang Test Engineering"

# Test markers for selective execution
MARKERS = {
    "unit": "Unit tests for individual functions",
    "integration": "Integration tests for component interaction",
    "performance": "Performance benchmarks and load tests",
    "compliance": "Regulatory compliance validation tests",
    "safety": "Safety-critical functionality tests",
    "slow": "Tests that take longer than 1 second",
}
