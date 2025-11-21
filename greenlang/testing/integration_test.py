# -*- coding: utf-8 -*-
"""
Integration Testing Framework
============================

Test cases and utilities for integration and end-to-end testing.

This module provides specialized test cases for full stack testing,
Docker Compose integration, and end-to-end pipeline testing.
"""

import unittest
import subprocess
import time
import os
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from .agent_test import AgentTestCase
from .llm_test import LLMTestCase
from .cache_test import CacheTestCase
from .database_test import DatabaseTestCase


class IntegrationTestCase(unittest.TestCase):
    """
    Base test case for integration and end-to-end testing.

    Provides full stack testing, Docker Compose integration,
    and end-to-end pipeline testing capabilities.

    Example:
    --------
    ```python
    class TestFullStack(IntegrationTestCase):
        def test_end_to_end_workflow(self):
            # Start all services
            with self.docker_services():
                # Test complete workflow
                result = self.run_end_to_end_test()
                self.assert_integration_success(result)
    ```
    """

    def setUp(self):
        """Set up integration test environment."""
        self.services = []
        self.cleanup_tasks = []
        self.integration_results = []

        # Check for Docker availability
        self.docker_available = self._check_docker_available()

    def tearDown(self):
        """Clean up integration test environment."""
        # Run cleanup tasks
        for cleanup_task in reversed(self.cleanup_tasks):
            try:
                cleanup_task()
            except Exception as e:
                print(f"Cleanup task failed: {e}")

        # Stop all services
        self.stop_all_services()

    def _check_docker_available(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @contextmanager
    def docker_services(
        self,
        compose_file: str = "docker-compose.yml",
        services: Optional[List[str]] = None
    ):
        """
        Context manager for starting Docker Compose services.

        Args:
            compose_file: Path to docker-compose.yml
            services: List of specific services to start (None = all)

        Example:
        --------
        ```python
        with self.docker_services("docker-compose.test.yml", ["postgres", "redis"]):
            # Services are running
            self.run_tests()
            # Services stopped automatically
        ```
        """
        if not self.docker_available:
            self.skipTest("Docker not available")

        # Start services
        cmd = ['docker-compose', '-f', compose_file, 'up', '-d']
        if services:
            cmd.extend(services)

        subprocess.run(cmd, check=True)

        # Wait for services to be ready
        self.wait_for_services_ready()

        try:
            yield
        finally:
            # Stop services
            subprocess.run(
                ['docker-compose', '-f', compose_file, 'down'],
                check=True
            )

    def wait_for_services_ready(
        self,
        timeout: int = 60,
        check_interval: float = 1.0
    ):
        """
        Wait for Docker services to be ready.

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self._check_services_health():
                return
            time.sleep(check_interval)

        raise TimeoutError("Services did not become ready in time")

    def _check_services_health(self) -> bool:
        """Check if all services are healthy."""
        # This is a hook for subclasses to implement
        # Default implementation just waits
        return True

    def start_service(
        self,
        name: str,
        command: List[str],
        env: Optional[Dict[str, str]] = None
    ):
        """
        Start a service process.

        Args:
            name: Service name
            command: Command to start service
            env: Environment variables
        """
        process = subprocess.Popen(
            command,
            env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        self.services.append({
            'name': name,
            'process': process,
            'command': command,
        })

        return process

    def stop_service(self, name: str):
        """Stop a service by name."""
        for service in self.services:
            if service['name'] == name:
                service['process'].terminate()
                service['process'].wait(timeout=10)
                self.services.remove(service)
                return

    def stop_all_services(self):
        """Stop all running services."""
        for service in self.services:
            try:
                service['process'].terminate()
                service['process'].wait(timeout=10)
            except Exception as e:
                print(f"Failed to stop service {service['name']}: {e}")

        self.services.clear()

    def run_end_to_end_test(
        self,
        test_data: Any,
        expected_output: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run an end-to-end test with input data.

        Args:
            test_data: Input data for the test
            expected_output: Expected output (for validation)

        Returns:
            Test result with metadata
        """
        start_time = time.time()

        try:
            # This is a hook for subclasses to implement
            result = self._execute_end_to_end(test_data)

            execution_time = time.time() - start_time

            test_result = {
                'input': test_data,
                'output': result,
                'expected': expected_output,
                'execution_time': execution_time,
                'success': True,
                'error': None,
            }

            self.integration_results.append(test_result)

            # Validate output if expected provided
            if expected_output is not None:
                self._validate_output(result, expected_output)

            return test_result

        except Exception as e:
            execution_time = time.time() - start_time

            test_result = {
                'input': test_data,
                'output': None,
                'expected': expected_output,
                'execution_time': execution_time,
                'success': False,
                'error': str(e),
            }

            self.integration_results.append(test_result)
            raise

    def _execute_end_to_end(self, test_data: Any) -> Any:
        """
        Execute end-to-end test logic.

        This should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _execute_end_to_end")

    def _validate_output(self, actual: Any, expected: Any):
        """Validate actual output against expected output."""
        self.assertEqual(actual, expected)

    def assert_integration_success(self, result: Dict[str, Any]):
        """Assert that integration test was successful."""
        self.assertTrue(
            result.get('success', False),
            f"Integration test failed: {result.get('error')}"
        )

    def assert_all_services_running(self):
        """Assert that all services are running."""
        for service in self.services:
            self.assertIsNone(
                service['process'].poll(),
                f"Service {service['name']} is not running"
            )

    def mock_external_service(
        self,
        service_url: str,
        mock_responses: Dict[str, Any]
    ):
        """
        Mock an external service for testing.

        Args:
            service_url: URL of the service to mock
            mock_responses: Dictionary mapping endpoints to responses
        """
        # This would typically use a library like responses or httpretty
        # For now, just record the mock configuration
        self.cleanup_tasks.append(lambda: None)  # Placeholder for cleanup

    @contextmanager
    def temporary_environment(self, env_vars: Dict[str, str]):
        """
        Context manager for temporarily setting environment variables.

        Args:
            env_vars: Dictionary of environment variables to set

        Example:
        --------
        ```python
        with self.temporary_environment({'API_KEY': 'test_key'}):
            # API_KEY is set
            result = my_function()
            # API_KEY restored to original value
        ```
        """
        original_env = {}

        # Save original values and set new ones
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            yield
        finally:
            # Restore original values
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get aggregated integration test statistics."""
        total_tests = len(self.integration_results)
        successful_tests = len([r for r in self.integration_results if r['success']])

        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'success_rate': successful_tests / max(total_tests, 1),
            'total_time': sum(r['execution_time'] for r in self.integration_results),
            'avg_time': sum(r['execution_time'] for r in self.integration_results) / max(total_tests, 1),
        }
