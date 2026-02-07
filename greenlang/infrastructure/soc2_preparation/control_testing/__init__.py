# -*- coding: utf-8 -*-
"""
Control Testing Framework - SEC-009 Phase 4

Automated control testing for SOC 2 Type II compliance. Provides a comprehensive
test framework covering all SOC 2 Trust Services Criteria (CC6, CC7, CC8, A1, C1).

Components:
    - ControlTestFramework: Core test execution engine with suite management
    - TestAutomation: System-level automated tests querying live infrastructure
    - TestReporter: Report generation in multiple formats (text, PDF, JSON)
    - BaseControlTest: Abstract base class for implementing control tests

Example:
    >>> from greenlang.infrastructure.soc2_preparation.control_testing import (
    ...     ControlTestFramework,
    ...     TestAutomation,
    ... )
    >>> framework = ControlTestFramework()
    >>> automation = TestAutomation(framework)
    >>> results = await framework.execute_suite(["CC6", "CC7"])
"""

from greenlang.infrastructure.soc2_preparation.control_testing.test_framework import (
    ControlTestFramework,
    ControlTest,
    TestResult,
    TestRun,
    TestStatus,
    TestType,
)
from greenlang.infrastructure.soc2_preparation.control_testing.test_cases import (
    BaseControlTest,
    CC6Tests,
    CC7Tests,
    CC8Tests,
    A1Tests,
    C1Tests,
)
from greenlang.infrastructure.soc2_preparation.control_testing.automation import (
    TestAutomation,
)
from greenlang.infrastructure.soc2_preparation.control_testing.reporter import (
    TestReporter,
)

__all__ = [
    # Framework
    "ControlTestFramework",
    "ControlTest",
    "TestResult",
    "TestRun",
    "TestStatus",
    "TestType",
    # Test Cases
    "BaseControlTest",
    "CC6Tests",
    "CC7Tests",
    "CC8Tests",
    "A1Tests",
    "C1Tests",
    # Automation
    "TestAutomation",
    # Reporter
    "TestReporter",
]
