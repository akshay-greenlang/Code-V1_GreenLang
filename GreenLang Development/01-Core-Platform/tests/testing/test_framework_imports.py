# -*- coding: utf-8 -*-
"""
Framework Import Verification
==============================

Quick test to verify all framework components can be imported.
"""

def test_imports():
    """Test that all framework components can be imported."""

    # Test main module imports
    from greenlang.testing import (
        AgentTestCase,
        PipelineTestCase,
        LLMTestCase,
        CacheTestCase,
        DatabaseTestCase,
        IntegrationTestCase,
    )

    # Test mock imports
    from greenlang.testing import (
        MockChatSession,
        MockCacheManager,
        MockDatabaseManager,
        MockValidationFramework,
        MockTelemetryManager,
    )

    # Test assertion imports
    from greenlang.testing import (
        assert_agent_result_valid,
        assert_schema_valid,
        assert_performance,
        assert_cache_hit_rate,
        assert_no_hallucination,
        assert_deterministic,
        assert_cost_within_budget,
        assert_token_count,
    )

    # Test individual modules
    from greenlang.testing import agent_test
    from greenlang.testing import llm_test
    from greenlang.testing import cache_test
    from greenlang.testing import database_test
    from greenlang.testing import integration_test
    from greenlang.testing import mocks
    from greenlang.testing import assertions

    print("✅ All imports successful!")
    print("\nAvailable Test Cases:")
    print("  - AgentTestCase")
    print("  - PipelineTestCase")
    print("  - LLMTestCase")
    print("  - CacheTestCase")
    print("  - DatabaseTestCase")
    print("  - IntegrationTestCase")

    print("\nAvailable Mocks:")
    print("  - MockChatSession")
    print("  - MockCacheManager")
    print("  - MockDatabaseManager")
    print("  - MockValidationFramework")
    print("  - MockTelemetryManager")

    print("\nAvailable Assertions:")
    print("  - assert_agent_result_valid")
    print("  - assert_schema_valid")
    print("  - assert_performance")
    print("  - assert_cache_hit_rate")
    print("  - assert_no_hallucination")
    print("  - assert_deterministic")
    print("  - assert_cost_within_budget")
    print("  - assert_token_count")

    return True


if __name__ == '__main__':
    try:
        test_imports()
        print("\n✅ GreenLang Testing Framework is ready to use!")
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
