# -*- coding: utf-8 -*-
"""
GreenLang Agent Foundation Testing Framework

Comprehensive testing suite for AI agents with:
- 90%+ test coverage target
- 12-dimension quality validation
- Zero-hallucination guarantee testing
- Deterministic mock LLM providers
- Performance and load testing
- Compliance and regulatory testing
"""

from .agent_test_framework import (
    AgentTestCase,
    TestConfig,
    MockLLMProvider,
    DeterministicLLMProvider,
    PerformanceTestRunner,
    ProvenanceValidator,
    TestDataGenerator,
    AgentTestFixtures,
    TestMetrics,
    CoverageAnalyzer
)

from .quality_validators import (
    QualityValidator,
    FunctionalQualityValidator,
    PerformanceValidator,
    ReliabilityValidator,
    SecurityValidator,
    ComplianceValidator,
    ScalabilityValidator,
    MaintainabilityValidator,
    CompatibilityValidator,
    UsabilityValidator,
    PortabilityValidator,
    InteroperabilityValidator,
    TestabilityValidator,
    QualityReport,
    QualityMetrics
)

__version__ = "1.0.0"

__all__ = [
    # Test Framework
    "AgentTestCase",
    "TestConfig",
    "MockLLMProvider",
    "DeterministicLLMProvider",
    "PerformanceTestRunner",
    "ProvenanceValidator",
    "TestDataGenerator",
    "AgentTestFixtures",
    "TestMetrics",
    "CoverageAnalyzer",

    # Quality Validators
    "QualityValidator",
    "FunctionalQualityValidator",
    "PerformanceValidator",
    "ReliabilityValidator",
    "SecurityValidator",
    "ComplianceValidator",
    "ScalabilityValidator",
    "MaintainabilityValidator",
    "CompatibilityValidator",
    "UsabilityValidator",
    "PortabilityValidator",
    "InteroperabilityValidator",
    "TestabilityValidator",
    "QualityReport",
    "QualityMetrics"
]

# Test Configuration Defaults
DEFAULT_TEST_CONFIG = {
    "coverage_target": 0.90,  # 90% coverage target
    "performance_p99_ms": 5000,  # 5 second P99 latency
    "memory_limit_mb": 4096,  # 4GB per agent
    "concurrent_agents": 10000,  # Support 10k agents
    "zero_hallucination": True,  # Guarantee no hallucinations
    "provenance_tracking": True,  # Full audit trail
    "deterministic_testing": True,  # Reproducible tests
}

# Quality Dimensions (ISO 25010)
QUALITY_DIMENSIONS = [
    "functional_quality",
    "performance_efficiency",
    "compatibility",
    "usability",
    "reliability",
    "security",
    "maintainability",
    "portability",
    "scalability",
    "interoperability",
    "reusability",
    "testability"
]

# Test Categories
TEST_CATEGORIES = {
    "unit": "Unit tests for individual components",
    "integration": "Integration tests for component interactions",
    "e2e": "End-to-end tests for complete workflows",
    "performance": "Performance and load tests",
    "security": "Security and vulnerability tests",
    "compliance": "Regulatory compliance tests",
    "chaos": "Chaos engineering tests",
    "regression": "Regression test suite"
}