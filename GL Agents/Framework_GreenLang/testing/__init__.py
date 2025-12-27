"""
GreenLang Framework - Testing Infrastructure

Comprehensive testing utilities for GreenLang agents.
Provides tools for unit testing, integration testing, performance testing,
chaos testing, and compliance verification.

Modules:
- test_utils: Golden value fixtures, property-based testing, data generators
- conftest_template: Common pytest fixtures for agents
- chaos: Chaos engineering utilities for resilience testing

Target Coverage: 85%+
Author: GreenLang QA Team
Version: 1.0.0
"""

from .test_utils import (
    GoldenValueFixtures,
    ThermodynamicTestData,
    PropertyTestHelpers,
    CoverageReporter,
    NISTReferenceData,
    IAPWSReferenceData,
    EPAReferenceData,
    GoldenValue,
    ToleranceType,
    ReferenceDataSource,
    TestDataFactory,
    AssertionHelpers,
)

from .chaos import (
    ChaosController,
    ChaosConfig,
    ChaosEvent,
    ChaosType,
    SeverityLevel,
    NetworkFailureSimulator,
    NetworkLatencyInjector,
    TimeoutInjector,
    ResourceExhaustionSimulator,
    ExceptionInjector,
    DataCorruptionSimulator,
    ClockSkewSimulator,
)

__all__ = [
    # Test utilities - Reference data
    "GoldenValueFixtures",
    "ThermodynamicTestData",
    "PropertyTestHelpers",
    "CoverageReporter",
    "NISTReferenceData",
    "IAPWSReferenceData",
    "EPAReferenceData",
    # Test utilities - Core classes
    "GoldenValue",
    "ToleranceType",
    "ReferenceDataSource",
    "TestDataFactory",
    "AssertionHelpers",
    # Chaos testing - Configuration
    "ChaosController",
    "ChaosConfig",
    "ChaosEvent",
    "ChaosType",
    "SeverityLevel",
    # Chaos testing - Injectors
    "NetworkFailureSimulator",
    "NetworkLatencyInjector",
    "TimeoutInjector",
    "ResourceExhaustionSimulator",
    "ExceptionInjector",
    "DataCorruptionSimulator",
    "ClockSkewSimulator",
]
