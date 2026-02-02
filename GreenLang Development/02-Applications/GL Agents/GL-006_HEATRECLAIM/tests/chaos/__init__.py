# -*- coding: utf-8 -*-
"""
GL-006 HEATRECLAIM - Chaos Engineering Tests

This package contains chaos engineering tests for validating system resilience
under failure conditions. Tests verify that the Heat Exchanger Network optimizer
gracefully handles:

1. Network partitions and timeouts
2. Service degradation and slow responses
3. Random failures and transient errors
4. Resource exhaustion scenarios
5. Circuit breaker behavior under load

Reference Standards:
    - Netflix Chaos Engineering Principles
    - AWS Fault Injection Simulator patterns
    - IEC 61508 Fault Tolerance Validation
    - ASME PTC 4.3/4.4 Safety Requirements

Author: GL-TestEngineer
Date: December 2025
Version: 1.0.0
"""

from .test_resilience import *

__all__ = [
    "TestCircuitBreakerChaos",
    "TestNetworkFailures",
    "TestServiceDegradation",
    "TestResourceExhaustion",
    "TestRecoveryBehavior",
]
