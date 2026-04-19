# -*- coding: utf-8 -*-
"""
Agent Testing Framework
=======================

Test cases and utilities for testing GreenLang agents and pipelines.

This module provides specialized test cases for testing individual agents
and complete pipelines with mocking, performance monitoring, and assertion helpers.
"""

import unittest
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Type, Callable
from unittest.mock import Mock, patch
import json
from contextlib import contextmanager

from .mocks import MockChatSession, MockCacheManager, MockDatabaseManager
from .assertions import (
    assert_agent_result_valid,
    assert_schema_valid,
    assert_performance,
)


class AgentTestCase(unittest.TestCase):
    """
    Base test case for testing GreenLang agents.

    Provides setUp/tearDown helpers, mock infrastructure components,
    assertion helpers for agent results, and performance testing.

    Example:
    --------
    ```python
    class TestEmissionsAgent(AgentTestCase):
        def test_calculate_emissions(self):
            agent = EmissionsCalculatorAgent()
            result = self.run_agent(agent, self.sample_input)

            self.assert_success(result)
            self.assert_output_schema(result, EmissionsSchema)
            self.assert_performance(result, max_time=2.0)
    ```
    """

    def setUp(self):
        """Set up test fixtures and mock infrastructure."""
        # Start performance monitoring
        self.start_time = time.time()
        tracemalloc.start()

        # Create mock infrastructure
        self.mock_chat = MockChatSession()
        self.mock_cache = MockCacheManager()
        self.mock_db = MockDatabaseManager()

        # Track agent executions
        self.agent_executions = []
        self.performance_metrics = {}

        # Set up test data directory
        import os
        self.test_data_dir = os.path.join(
            os.path.dirname(__file__), 'fixtures'
        )

    def tearDown(self):
        """Clean up after tests."""
        # Stop performance monitoring
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.performance_metrics['memory_current'] = current
        self.performance_metrics['memory_peak'] = peak
        self.performance_metrics['total_time'] = time.time() - self.start_time

        # Clean up mocks
        self.mock_chat.reset()
        self.mock_cache.clear()
        self.mock_db.reset()

    def run_agent(
        self,
        agent: Any,
        input_data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run an agent with input data and track performance.

        Args:
            agent: Agent instance or class to test
            input_data: Input data for the agent
            **kwargs: Additional arguments for agent execution

        Returns:
            Agent execution result with metadata
        """
        start = time.time()
        start_memory = tracemalloc.get_traced_memory()[0]

        try:
            # Initialize agent if it's a class
            if isinstance(agent, type):
                agent = agent()

            # Run the agent
            result = agent.run(input_data, **kwargs)

            # Track execution
            execution_time = time.time() - start
            memory_used = tracemalloc.get_traced_memory()[0] - start_memory

            execution_record = {
                'agent': agent.__class__.__name__,
                'input': input_data,
                'result': result,
                'execution_time': execution_time,
                'memory_used': memory_used,
                'success': True,
                'error': None,
            }

            self.agent_executions.append(execution_record)

            return {
                'result': result,
                'execution_time': execution_time,
                'memory_used': memory_used,
                'success': True,
            }

        except Exception as e:
            execution_time = time.time() - start
            memory_used = tracemalloc.get_traced_memory()[0] - start_memory

            execution_record = {
                'agent': agent.__class__.__name__,
                'input': input_data,
                'result': None,
                'execution_time': execution_time,
                'memory_used': memory_used,
                'success': False,
                'error': str(e),
            }

            self.agent_executions.append(execution_record)
            raise

    def run_agent_batch(
        self,
        agent: Any,
        input_batch: List[Any],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run an agent with batch input and track performance.

        Args:
            agent: Agent instance or class to test
            input_batch: List of input data items
            **kwargs: Additional arguments for agent execution

        Returns:
            List of agent execution results with metadata
        """
        results = []
        for input_data in input_batch:
            result = self.run_agent(agent, input_data, **kwargs)
            results.append(result)
        return results

    def assert_success(self, result: Dict[str, Any]):
        """Assert that agent execution was successful."""
        self.assertTrue(
            result.get('success', False),
            f"Agent execution failed: {result.get('error')}"
        )
        self.assertIsNotNone(result.get('result'))

    def assert_output_schema(self, result: Dict[str, Any], schema: Any):
        """Assert that agent output matches expected schema."""
        assert_schema_valid(result.get('result'), schema)

    def assert_performance(
        self,
        result: Dict[str, Any],
        max_time: Optional[float] = None,
        max_memory: Optional[int] = None
    ):
        """Assert that agent performance is within bounds."""
        if max_time is not None:
            self.assertLessEqual(
                result.get('execution_time', float('inf')),
                max_time,
                f"Execution time {result.get('execution_time')}s exceeded max {max_time}s"
            )

        if max_memory is not None:
            self.assertLessEqual(
                result.get('memory_used', float('inf')),
                max_memory,
                f"Memory usage {result.get('memory_used')} exceeded max {max_memory}"
            )

    def assert_deterministic(
        self,
        agent: Any,
        input_data: Any,
        runs: int = 3
    ):
        """Assert that agent produces deterministic results."""
        results = []
        for _ in range(runs):
            result = self.run_agent(agent, input_data)
            results.append(result['result'])

        # All results should be identical
        first_result = json.dumps(results[0], sort_keys=True)
        for result in results[1:]:
            self.assertEqual(
                first_result,
                json.dumps(result, sort_keys=True),
                "Agent produced non-deterministic results"
            )

    @contextmanager
    def mock_infrastructure(
        self,
        chat: Optional[Mock] = None,
        cache: Optional[Mock] = None,
        db: Optional[Mock] = None
    ):
        """Context manager for mocking infrastructure components."""
        with patch('greenlang.core.chat.ChatSession', chat or self.mock_chat), \
             patch('greenlang.core.cache.CacheManager', cache or self.mock_cache), \
             patch('greenlang.core.database.DatabaseManager', db or self.mock_db):
            yield

    def load_fixture(self, filename: str) -> Any:
        """Load a test fixture file."""
        import os
        import json
        import yaml

        filepath = os.path.join(self.test_data_dir, filename)

        if filename.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filename.endswith(('.yaml', '.yml')):
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        else:
            with open(filepath, 'r') as f:
                return f.read()


class PipelineTestCase(AgentTestCase):
    """
    Test case for testing agent pipelines.

    Extends AgentTestCase with pipeline-specific functionality.

    Example:
    --------
    ```python
    class TestEmissionsPipeline(PipelineTestCase):
        def test_full_pipeline(self):
            pipeline = EmissionsCalculationPipeline()
            result = self.run_pipeline(pipeline, self.sample_data)

            self.assert_pipeline_success(result)
            self.assert_all_stages_completed(result)
            self.assert_pipeline_performance(result, max_time=10.0)
    ```
    """

    def run_pipeline(
        self,
        pipeline: Any,
        input_data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a pipeline with input data and track stage performance.

        Args:
            pipeline: Pipeline instance or class to test
            input_data: Input data for the pipeline
            **kwargs: Additional arguments for pipeline execution

        Returns:
            Pipeline execution result with stage metadata
        """
        start = time.time()

        try:
            # Initialize pipeline if it's a class
            if isinstance(pipeline, type):
                pipeline = pipeline()

            # Run the pipeline
            result = pipeline.run(input_data, **kwargs)

            execution_time = time.time() - start

            return {
                'result': result,
                'execution_time': execution_time,
                'stages': getattr(result, 'stages', []),
                'success': True,
            }

        except Exception as e:
            execution_time = time.time() - start

            return {
                'result': None,
                'execution_time': execution_time,
                'stages': [],
                'success': False,
                'error': str(e),
            }

    def assert_pipeline_success(self, result: Dict[str, Any]):
        """Assert that pipeline execution was successful."""
        self.assertTrue(
            result.get('success', False),
            f"Pipeline execution failed: {result.get('error')}"
        )

    def assert_all_stages_completed(self, result: Dict[str, Any]):
        """Assert that all pipeline stages completed successfully."""
        stages = result.get('stages', [])
        self.assertTrue(len(stages) > 0, "No stages executed")

        for stage in stages:
            self.assertTrue(
                stage.get('success', False),
                f"Stage {stage.get('name')} failed: {stage.get('error')}"
            )

    def assert_stage_output(
        self,
        result: Dict[str, Any],
        stage_name: str,
        expected_schema: Any
    ):
        """Assert that a specific stage output matches schema."""
        stages = result.get('stages', [])
        stage = next((s for s in stages if s.get('name') == stage_name), None)

        self.assertIsNotNone(stage, f"Stage {stage_name} not found")
        assert_schema_valid(stage.get('output'), expected_schema)

    def assert_pipeline_performance(
        self,
        result: Dict[str, Any],
        max_time: Optional[float] = None
    ):
        """Assert that pipeline performance is within bounds."""
        if max_time is not None:
            self.assertLessEqual(
                result.get('execution_time', float('inf')),
                max_time,
                f"Pipeline execution time {result.get('execution_time')}s exceeded max {max_time}s"
            )
