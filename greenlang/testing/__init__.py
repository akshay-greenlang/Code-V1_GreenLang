"""
GreenLang Testing Framework
===========================

A comprehensive testing framework for GreenLang infrastructure-based applications.

This module provides specialized test cases, mocks, fixtures, and assertions
for testing agents, pipelines, LLM integrations, caching, databases, and
full integration scenarios.

Basic Usage
-----------

```python
from greenlang.testing import (
    AgentTestCase,
    PipelineTestCase,
    LLMTestCase,
    CacheTestCase,
    DatabaseTestCase,
    IntegrationTestCase
)

class TestMyAgent(AgentTestCase):
    def test_agent_execution(self):
        result = self.run_agent(MyAgent, input_data)
        self.assert_success(result)
        self.assert_output_schema(result, expected_schema)
        self.assert_performance(result, max_time=1.0)
```

Main Components
---------------

Test Cases:
- AgentTestCase: For testing GreenLang agents
- PipelineTestCase: For testing agent pipelines
- LLMTestCase: For testing LLM integrations
- CacheTestCase: For testing caching mechanisms
- DatabaseTestCase: For testing database operations
- IntegrationTestCase: For end-to-end testing

Mocks:
- MockChatSession: Mock LLM chat sessions
- MockCacheManager: Mock cache operations
- MockDatabaseManager: Mock database operations
- MockValidationFramework: Mock validation
- MockTelemetryManager: Mock telemetry

Assertions:
- assert_agent_result_valid()
- assert_schema_valid()
- assert_performance()
- assert_cache_hit_rate()
- assert_no_hallucination()
- assert_deterministic()

Fixtures:
- Sample emissions data
- Sample supplier data
- Sample configurations
- Mock LLM responses
- Test database schemas
"""

from .agent_test import AgentTestCase, PipelineTestCase
from .llm_test import LLMTestCase
from .cache_test import CacheTestCase
from .database_test import DatabaseTestCase
from .integration_test import IntegrationTestCase
from .mocks import (
    MockChatSession,
    MockCacheManager,
    MockDatabaseManager,
    MockValidationFramework,
    MockTelemetryManager,
)
from .assertions import (
    assert_agent_result_valid,
    assert_schema_valid,
    assert_performance,
    assert_cache_hit_rate,
    assert_no_hallucination,
    assert_deterministic,
    assert_cost_within_budget,
    assert_token_count,
)

__version__ = "1.0.0"

__all__ = [
    # Test Cases
    "AgentTestCase",
    "PipelineTestCase",
    "LLMTestCase",
    "CacheTestCase",
    "DatabaseTestCase",
    "IntegrationTestCase",

    # Mocks
    "MockChatSession",
    "MockCacheManager",
    "MockDatabaseManager",
    "MockValidationFramework",
    "MockTelemetryManager",

    # Assertions
    "assert_agent_result_valid",
    "assert_schema_valid",
    "assert_performance",
    "assert_cache_hit_rate",
    "assert_no_hallucination",
    "assert_deterministic",
    "assert_cost_within_budget",
    "assert_token_count",
]
