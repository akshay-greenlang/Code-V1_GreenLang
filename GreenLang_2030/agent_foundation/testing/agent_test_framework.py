"""
GreenLang Agent Test Framework
Comprehensive testing infrastructure for AI agents with full lifecycle testing.
"""

import unittest
import pytest
import asyncio
import hashlib
import json
import time
import tracemalloc
import psutil
import threading
from typing import Any, Dict, List, Optional, Callable, Tuple, Type, Union
from dataclasses import dataclass, asdict, field
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from contextlib import contextmanager
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import numpy as np
from decimal import Decimal
import logging
import sys
import os
import tempfile
import uuid
import concurrent.futures
from faker import Faker

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Agent Lifecycle States (from Architecture doc lines 58-66)
class AgentState(Enum):
    """Agent lifecycle states."""
    CREATED = "Agent instantiated, not initialized"
    INITIALIZING = "Loading configuration and resources"
    READY = "Agent ready to receive messages"
    RUNNING = "Actively processing tasks"
    PAUSED = "Temporarily suspended"
    STOPPING = "Graceful shutdown in progress"
    TERMINATED = "Agent fully stopped"
    ERROR = "Fatal error occurred"


# Test Configuration
@dataclass
class TestConfig:
    """Comprehensive test configuration."""
    # Coverage targets
    coverage_target: float = 0.90  # 90% coverage target

    # Performance targets (from Architecture doc line 22-28)
    agent_creation_ms: float = 100.0  # <100ms
    message_passing_ms: float = 10.0  # <10ms
    memory_retrieval_ms: float = 50.0  # <50ms for recent
    llm_call_avg_ms: float = 2000.0  # <2s average
    llm_call_p99_ms: float = 5000.0  # <5s P99
    concurrent_agents: int = 10000  # 10,000+ per cluster

    # Test settings
    enable_performance_tests: bool = True
    enable_llm_mocking: bool = True
    enable_determinism_checks: bool = True
    enable_provenance_tracking: bool = True
    enable_zero_hallucination: bool = True

    # Memory limits
    memory_threshold_mb: float = 4096.0  # 4GB per agent

    # Test execution
    test_iterations: int = 10
    random_seed: int = 42
    async_timeout: float = 30.0


# Base Test Case with Full Lifecycle Support
class AgentTestCase(unittest.TestCase):
    """Base test case for GreenLang agents with comprehensive lifecycle testing."""

    def setUp(self):
        """Set up test environment with lifecycle tracking."""
        self.test_config = TestConfig()
        self.start_time = time.time()
        self.memory_tracker = None

        # Set random seeds for reproducibility
        np.random.seed(self.test_config.random_seed)
        Faker.seed(self.test_config.random_seed)

        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Performance tracking
        self.performance_metrics = []
        self.lifecycle_events = []

        # Start memory tracking if enabled
        if self.test_config.enable_performance_tests:
            tracemalloc.start()
            self.memory_tracker = tracemalloc.get_traced_memory()

        # Process monitoring
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def tearDown(self):
        """Clean up test environment with metrics collection."""
        # Stop memory tracking
        if self.memory_tracker:
            tracemalloc.stop()

        # Log test duration
        duration = time.time() - self.start_time
        self.logger.info(f"Test {self._testMethodName} completed in {duration:.3f}s")

        # Collect final metrics
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - self.initial_memory

        self.performance_metrics.append({
            "test": self._testMethodName,
            "duration_s": duration,
            "memory_increase_mb": memory_increase,
            "lifecycle_events": len(self.lifecycle_events)
        })

    def create_mock_agent(
        self,
        agent_class: Type,
        config: Optional[Dict] = None,
        initial_state: AgentState = AgentState.CREATED
    ):
        """Create a mock agent with lifecycle management."""
        if config is None:
            config = {
                "name": f"test_agent_{uuid.uuid4().hex[:8]}",
                "version": "1.0.0",
                "environment": "test",
                "debug": True
            }

        # Create mock agent with lifecycle tracking
        agent = Mock(spec=agent_class)
        agent.state = initial_state
        agent.config = config
        agent.lifecycle_history = []
        agent.provenance_chain = []

        # Mock state transitions
        def transition_to(new_state: AgentState):
            old_state = agent.state
            agent.state = new_state
            agent.lifecycle_history.append({
                "from": old_state,
                "to": new_state,
                "timestamp": datetime.now().isoformat()
            })
            self.lifecycle_events.append({
                "agent": config["name"],
                "transition": f"{old_state.name} -> {new_state.name}",
                "time": time.time() - self.start_time
            })

        agent.transition_to = transition_to

        # Mock LLM if enabled
        if self.test_config.enable_llm_mocking:
            agent.llm_client = self.create_deterministic_llm()

        # Mock memory systems
        agent.short_term_memory = deque(maxlen=100)
        agent.long_term_memory = {}
        agent.episodic_memory = []
        agent.semantic_memory = {}

        return agent

    def create_deterministic_llm(self) -> "DeterministicLLMProvider":
        """Create deterministic LLM for reproducible testing."""
        return DeterministicLLMProvider(seed=self.test_config.random_seed)

    def assert_lifecycle_transition(
        self,
        agent: Mock,
        from_state: AgentState,
        to_state: AgentState,
        max_duration_ms: Optional[float] = None
    ):
        """Assert valid lifecycle state transition."""
        # Verify current state
        self.assertEqual(agent.state, from_state,
                        f"Agent not in expected state {from_state.name}")

        # Perform transition with timing
        start_time = time.perf_counter()
        agent.transition_to(to_state)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Verify new state
        self.assertEqual(agent.state, to_state,
                        f"Agent failed to transition to {to_state.name}")

        # Check duration if specified
        if max_duration_ms:
            self.assertLessEqual(duration_ms, max_duration_ms,
                               f"Transition took {duration_ms:.2f}ms > {max_duration_ms}ms")

        # Verify transition is recorded
        last_transition = agent.lifecycle_history[-1]
        self.assertEqual(last_transition["from"], from_state)
        self.assertEqual(last_transition["to"], to_state)

    def test_full_lifecycle(self, agent: Mock):
        """Test complete agent lifecycle from creation to termination."""
        # CREATED -> INITIALIZING
        self.assert_lifecycle_transition(
            agent, AgentState.CREATED, AgentState.INITIALIZING,
            max_duration_ms=self.test_config.agent_creation_ms
        )

        # INITIALIZING -> READY
        self.assert_lifecycle_transition(
            agent, AgentState.INITIALIZING, AgentState.READY,
            max_duration_ms=100
        )

        # READY -> RUNNING
        self.assert_lifecycle_transition(
            agent, AgentState.READY, AgentState.RUNNING,
            max_duration_ms=10
        )

        # RUNNING -> PAUSED
        self.assert_lifecycle_transition(
            agent, AgentState.RUNNING, AgentState.PAUSED,
            max_duration_ms=10
        )

        # PAUSED -> READY
        self.assert_lifecycle_transition(
            agent, AgentState.PAUSED, AgentState.READY,
            max_duration_ms=10
        )

        # READY -> STOPPING
        self.assert_lifecycle_transition(
            agent, AgentState.READY, AgentState.STOPPING,
            max_duration_ms=50
        )

        # STOPPING -> TERMINATED
        self.assert_lifecycle_transition(
            agent, AgentState.STOPPING, AgentState.TERMINATED,
            max_duration_ms=100
        )

    def assert_provenance_tracking(
        self,
        result: Any,
        input_data: Any,
        expected_chain_length: Optional[int] = None
    ):
        """Assert provenance tracking for reproducibility."""
        if not self.test_config.enable_provenance_tracking:
            return

        # Verify provenance hash exists
        self.assertIsNotNone(getattr(result, 'provenance_hash', None),
                           "Result missing provenance hash")

        # Verify hash format (SHA-256)
        provenance_hash = result.provenance_hash
        self.assertEqual(len(provenance_hash), 64,
                        "Invalid SHA-256 hash length")
        self.assertTrue(all(c in '0123456789abcdef' for c in provenance_hash),
                       "Invalid SHA-256 hash characters")

        # Verify deterministic hashing
        input_json = json.dumps(input_data, sort_keys=True)
        expected_hash = hashlib.sha256(input_json.encode()).hexdigest()

        # Check chain length if specified
        if expected_chain_length and hasattr(result, 'provenance_chain'):
            self.assertEqual(len(result.provenance_chain), expected_chain_length,
                           f"Provenance chain length mismatch")

    def assert_zero_hallucination(
        self,
        result: Any,
        ground_truth: Any,
        tolerance: float = 1e-6
    ):
        """Assert zero hallucination for critical calculations."""
        if not self.test_config.enable_zero_hallucination:
            return

        # For numeric results
        if isinstance(result, (int, float, Decimal)):
            if isinstance(ground_truth, (int, float, Decimal)):
                diff = abs(float(result) - float(ground_truth))
                self.assertLessEqual(diff, tolerance,
                                   f"Calculation error: {diff} > {tolerance}")

        # For structured results
        elif hasattr(result, 'calculations'):
            for key, value in result.calculations.items():
                if key in ground_truth:
                    self.assert_zero_hallucination(value, ground_truth[key], tolerance)

    @contextmanager
    def assert_performance(
        self,
        max_duration_ms: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
        max_cpu_percent: Optional[float] = None
    ):
        """Enhanced performance assertion with CPU tracking."""
        max_duration_ms = max_duration_ms or self.test_config.llm_call_avg_ms
        max_memory_mb = max_memory_mb or self.test_config.memory_threshold_mb

        start_time = time.perf_counter()
        start_memory = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0

        # Start CPU monitoring
        cpu_monitor = CPUMonitor() if max_cpu_percent else None
        if cpu_monitor:
            cpu_monitor.start()

        try:
            yield
        finally:
            # Check duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.assertLessEqual(duration_ms, max_duration_ms,
                               f"Duration {duration_ms:.2f}ms > {max_duration_ms}ms")

            # Check memory
            if tracemalloc.is_tracing():
                end_memory = tracemalloc.get_traced_memory()[0]
                memory_increase_mb = (end_memory - start_memory) / 1024 / 1024
                self.assertLessEqual(memory_increase_mb, max_memory_mb,
                                   f"Memory {memory_increase_mb:.2f}MB > {max_memory_mb}MB")

            # Check CPU
            if cpu_monitor:
                cpu_monitor.stop()
                max_cpu = cpu_monitor.max_cpu_percent
                if max_cpu_percent:
                    self.assertLessEqual(max_cpu, max_cpu_percent,
                                       f"CPU usage {max_cpu:.1f}% > {max_cpu_percent}%")

            # Track metrics
            self.performance_metrics.append({
                "context": "performance_assertion",
                "duration_ms": duration_ms,
                "memory_mb": memory_increase_mb if tracemalloc.is_tracing() else None,
                "cpu_percent": max_cpu if cpu_monitor else None
            })

    def assert_concurrent_agents(
        self,
        agent_class: Type,
        num_agents: int = 100,
        max_total_duration_s: float = 10.0
    ):
        """Test concurrent agent execution."""
        agents = []
        results = []

        # Create agents
        for i in range(num_agents):
            config = {"name": f"concurrent_agent_{i}", "version": "1.0.0"}
            agent = self.create_mock_agent(agent_class, config)
            agents.append(agent)

        # Execute concurrently
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            # Simulate agent processing
            def process_agent(agent):
                agent.transition_to(AgentState.RUNNING)
                time.sleep(0.01)  # Simulate work
                agent.transition_to(AgentState.READY)
                return {"agent": agent.config["name"], "success": True}

            futures = [executor.submit(process_agent, agent) for agent in agents]
            results = [f.result(timeout=max_total_duration_s) for f in futures]

        duration = time.time() - start_time

        # Verify results
        self.assertEqual(len(results), num_agents,
                        f"Not all agents completed: {len(results)}/{num_agents}")
        self.assertLessEqual(duration, max_total_duration_s,
                           f"Concurrent execution took {duration:.2f}s > {max_total_duration_s}s")

        # Check all agents succeeded
        success_count = sum(1 for r in results if r["success"])
        self.assertEqual(success_count, num_agents,
                        f"Some agents failed: {success_count}/{num_agents}")


# Deterministic LLM Provider
class DeterministicLLMProvider:
    """Deterministic LLM provider for reproducible testing."""

    def __init__(self, seed: int = 42):
        """Initialize with seed for determinism."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.call_count = 0
        self.call_history = []

        # Deterministic response templates
        self.templates = {
            "classification": ["Category A", "Category B", "Category C"],
            "extraction": {"field1": "value1", "field2": "value2"},
            "generation": "Deterministic generated text for testing.",
            "reasoning": "Step 1: Analyze input. Step 2: Process. Step 3: Conclude.",
            "calculation": 42.0
        }

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate deterministic response based on prompt hash."""
        self.call_count += 1

        # Record call
        call_record = {
            "prompt": prompt,
            "kwargs": kwargs,
            "call_number": self.call_count,
            "timestamp": datetime.now().isoformat()
        }
        self.call_history.append(call_record)

        # Generate deterministic response based on prompt hash
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        hash_int = int(prompt_hash[:8], 16)

        # Select template based on hash
        template_keys = list(self.templates.keys())
        selected_template = template_keys[hash_int % len(template_keys)]
        response = self.templates[selected_template]

        # Calculate deterministic metrics
        tokens = len(prompt.split()) + 20  # Fixed token count

        return {
            "response": response,
            "model": "deterministic-test-model",
            "tokens_used": tokens,
            "cost": tokens * 0.0001,
            "latency_ms": 50.0,  # Fixed latency
            "prompt_hash": prompt_hash[:8],
            "deterministic": True
        }

    async def generate_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Async version with deterministic delay."""
        await asyncio.sleep(0.05)  # Fixed 50ms delay
        return self.generate(prompt, **kwargs)

    def reset(self):
        """Reset provider state."""
        self.call_count = 0
        self.call_history = []


# Mock LLM Provider
class MockLLMProvider:
    """Mock LLM provider with configurable responses."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        """Initialize with custom responses."""
        self.responses = responses or {}
        self.default_response = "Mock response"
        self.call_history = []
        self.error_rate = 0.0  # Configurable error rate for testing
        self.latency_ms = 100  # Configurable latency

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate mock response."""
        # Record call
        self.call_history.append({
            "prompt": prompt,
            "kwargs": kwargs,
            "timestamp": datetime.now().isoformat()
        })

        # Simulate errors if configured
        if np.random.random() < self.error_rate:
            raise Exception("Simulated LLM error")

        # Find matching response
        response = self.default_response
        for pattern, pattern_response in self.responses.items():
            if pattern in prompt:
                response = pattern_response
                break

        # Simulate latency
        time.sleep(self.latency_ms / 1000)

        return {
            "response": response,
            "model": "mock-model",
            "tokens_used": 100,
            "cost": 0.01,
            "latency_ms": self.latency_ms
        }


# Performance Test Runner
class PerformanceTestRunner:
    """Run comprehensive performance tests on agents."""

    def __init__(self, config: TestConfig = None):
        """Initialize performance test runner."""
        self.config = config or TestConfig()
        self.results = []

    def test_agent_creation_performance(
        self,
        agent_class: Type,
        iterations: int = 100
    ) -> Dict[str, float]:
        """Test agent creation performance."""
        durations = []

        for i in range(iterations):
            start_time = time.perf_counter()

            config = {"name": f"perf_agent_{i}", "version": "1.0.0"}
            agent = Mock(spec=agent_class)
            agent.config = config
            agent.state = AgentState.CREATED

            duration_ms = (time.perf_counter() - start_time) * 1000
            durations.append(duration_ms)

        result = {
            "mean_ms": np.mean(durations),
            "p50_ms": np.percentile(durations, 50),
            "p95_ms": np.percentile(durations, 95),
            "p99_ms": np.percentile(durations, 99),
            "max_ms": np.max(durations),
            "target_ms": self.config.agent_creation_ms,
            "passed": np.percentile(durations, 99) < self.config.agent_creation_ms
        }

        self.results.append({"test": "agent_creation", **result})
        return result

    def test_message_passing_performance(
        self,
        num_agents: int = 10,
        messages_per_agent: int = 100
    ) -> Dict[str, float]:
        """Test inter-agent message passing performance."""
        agents = []
        message_times = []

        # Create agents
        for i in range(num_agents):
            agent = Mock()
            agent.id = f"agent_{i}"
            agent.inbox = deque()
            agents.append(agent)

        # Send messages between agents
        for _ in range(messages_per_agent):
            for sender in agents:
                receiver = agents[(agents.index(sender) + 1) % len(agents)]

                start_time = time.perf_counter()

                message = {
                    "from": sender.id,
                    "to": receiver.id,
                    "content": "test message",
                    "timestamp": datetime.now().isoformat()
                }
                receiver.inbox.append(message)

                duration_ms = (time.perf_counter() - start_time) * 1000
                message_times.append(duration_ms)

        result = {
            "mean_ms": np.mean(message_times),
            "p50_ms": np.percentile(message_times, 50),
            "p95_ms": np.percentile(message_times, 95),
            "p99_ms": np.percentile(message_times, 99),
            "target_ms": self.config.message_passing_ms,
            "passed": np.percentile(message_times, 99) < self.config.message_passing_ms
        }

        self.results.append({"test": "message_passing", **result})
        return result

    def test_memory_retrieval_performance(
        self,
        memory_size: int = 10000,
        num_queries: int = 1000
    ) -> Dict[str, float]:
        """Test memory retrieval performance."""
        # Create memory store
        memory_store = {}
        for i in range(memory_size):
            key = f"memory_{i}"
            memory_store[key] = {
                "content": f"Memory content {i}",
                "timestamp": datetime.now().isoformat(),
                "importance": np.random.random()
            }

        # Test retrieval
        retrieval_times = []
        for _ in range(num_queries):
            key = f"memory_{np.random.randint(0, memory_size)}"

            start_time = time.perf_counter()
            _ = memory_store.get(key)
            duration_ms = (time.perf_counter() - start_time) * 1000

            retrieval_times.append(duration_ms)

        result = {
            "mean_ms": np.mean(retrieval_times),
            "p50_ms": np.percentile(retrieval_times, 50),
            "p95_ms": np.percentile(retrieval_times, 95),
            "p99_ms": np.percentile(retrieval_times, 99),
            "target_ms": self.config.memory_retrieval_ms,
            "passed": np.percentile(retrieval_times, 99) < self.config.memory_retrieval_ms
        }

        self.results.append({"test": "memory_retrieval", **result})
        return result

    def generate_report(self) -> Dict[str, Any]:
        """Generate performance test report."""
        passed_tests = sum(1 for r in self.results if r.get("passed", False))
        total_tests = len(self.results)

        return {
            "summary": {
                "passed": passed_tests,
                "total": total_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "tests": self.results,
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat()
        }


# Provenance Validator
class ProvenanceValidator:
    """Validate provenance tracking for reproducibility."""

    @staticmethod
    def calculate_hash(data: Any) -> str:
        """Calculate SHA-256 hash of data."""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode()).hexdigest()

    @staticmethod
    def validate_chain(chain: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate provenance chain integrity."""
        errors = []

        if not chain:
            errors.append("Empty provenance chain")
            return False, errors

        # Check each link
        for i, link in enumerate(chain):
            # Verify required fields
            required_fields = ["hash", "timestamp", "operation", "input_hash"]
            for field in required_fields:
                if field not in link:
                    errors.append(f"Link {i} missing field: {field}")

            # Verify hash format
            if "hash" in link and len(link["hash"]) != 64:
                errors.append(f"Link {i} has invalid hash length")

            # Verify timestamp
            if "timestamp" in link:
                try:
                    datetime.fromisoformat(link["timestamp"])
                except ValueError:
                    errors.append(f"Link {i} has invalid timestamp format")

        # Verify chain continuity
        for i in range(1, len(chain)):
            if chain[i].get("parent_hash") != chain[i-1].get("hash"):
                errors.append(f"Chain broken between links {i-1} and {i}")

        return len(errors) == 0, errors


# Test Data Generator
class TestDataGenerator:
    """Generate comprehensive test data for agents."""

    def __init__(self, seed: int = 42):
        """Initialize with seed for reproducibility."""
        self.faker = Faker()
        Faker.seed(seed)
        np.random.seed(seed)

    def generate_agent_configs(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate agent configurations."""
        configs = []

        for i in range(count):
            config = {
                "name": f"agent_{self.faker.word()}_{i}",
                "version": f"{np.random.randint(1, 3)}.{np.random.randint(0, 10)}.0",
                "environment": np.random.choice(["dev", "test", "staging", "prod"]),
                "debug": np.random.choice([True, False]),
                "max_memory_mb": np.random.choice([1024, 2048, 4096]),
                "timeout_s": np.random.choice([30, 60, 120]),
                "retry_count": np.random.randint(1, 5)
            }
            configs.append(config)

        return configs

    def generate_test_messages(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate test messages for agent communication."""
        messages = []
        message_types = ["REQUEST", "RESPONSE", "EVENT", "COMMAND"]
        priorities = ["CRITICAL", "HIGH", "NORMAL", "LOW"]

        for i in range(count):
            message = {
                "message_id": str(uuid.uuid4()),
                "sender_id": f"agent_{np.random.randint(0, 10)}",
                "recipient_id": f"agent_{np.random.randint(0, 10)}",
                "message_type": np.random.choice(message_types),
                "priority": np.random.choice(priorities),
                "payload": {
                    "action": self.faker.word(),
                    "data": {
                        "value": np.random.random(),
                        "text": self.faker.sentence()
                    }
                },
                "timestamp": datetime.now().isoformat(),
                "correlation_id": str(uuid.uuid4()) if np.random.random() > 0.5 else None
            }
            messages.append(message)

        return messages

    def generate_memory_entries(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Generate memory entries for testing memory systems."""
        entries = []
        memory_types = ["short_term", "long_term", "episodic", "semantic"]

        for i in range(count):
            entry = {
                "id": str(uuid.uuid4()),
                "type": np.random.choice(memory_types),
                "content": self.faker.text(max_nb_chars=200),
                "importance": np.random.random(),
                "timestamp": self.faker.date_time_between(
                    start_date="-30d", end_date="now"
                ).isoformat(),
                "access_count": np.random.randint(0, 100),
                "last_accessed": datetime.now().isoformat(),
                "embeddings": np.random.randn(768).tolist(),  # 768-dim vector
                "metadata": {
                    "source": self.faker.word(),
                    "confidence": np.random.random(),
                    "tags": [self.faker.word() for _ in range(np.random.randint(1, 5))]
                }
            }
            entries.append(entry)

        return entries

    def generate_carbon_data(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate carbon emissions test data."""
        data = []
        fuel_types = ["diesel", "natural_gas", "electricity", "coal", "gasoline"]
        regions = ["US", "EU", "CN", "IN", "BR"]

        for i in range(count):
            record = {
                "id": f"EMISSION_{i:04d}",
                "fuel_type": np.random.choice(fuel_types),
                "quantity": np.random.uniform(10, 1000),
                "unit": np.random.choice(["liters", "kWh", "kg", "cubic_meters"]),
                "region": np.random.choice(regions),
                "date": self.faker.date_between(start_date="-1y", end_date="today"),
                "emission_factor": np.random.uniform(0.5, 3.5),
                "calculated_emissions": None  # To be calculated
            }
            # Calculate emissions
            record["calculated_emissions"] = record["quantity"] * record["emission_factor"]
            data.append(record)

        return data


# Agent Test Fixtures
class AgentTestFixtures:
    """Comprehensive test fixtures for agent testing."""

    @staticmethod
    def create_state_machine_mock():
        """Create mock state machine for lifecycle testing."""
        state_machine = Mock()
        state_machine.current_state = AgentState.CREATED
        state_machine.transitions = []

        def transition(to_state: AgentState):
            old_state = state_machine.current_state
            state_machine.current_state = to_state
            state_machine.transitions.append({
                "from": old_state,
                "to": to_state,
                "timestamp": datetime.now().isoformat()
            })

        state_machine.transition = transition
        return state_machine

    @staticmethod
    def create_rag_system_mock():
        """Create mock RAG system."""
        rag = Mock()
        rag.index = Mock()
        rag.retriever = Mock()
        rag.reranker = Mock()

        # Mock document indexing
        rag.index.add = Mock(return_value={"success": True, "doc_id": str(uuid.uuid4())})

        # Mock retrieval
        rag.retriever.search = Mock(return_value=[
            {
                "content": "Relevant document content",
                "score": 0.95,
                "metadata": {"source": "test_doc.pdf"}
            }
        ])

        # Mock reranking
        rag.reranker.rerank = Mock(side_effect=lambda docs: sorted(
            docs, key=lambda x: x["score"], reverse=True
        ))

        return rag

    @staticmethod
    def create_vector_store_mock():
        """Create mock vector store."""
        store = Mock()
        store.vectors = {}

        def add_vector(id: str, vector: List[float], metadata: Dict = None):
            store.vectors[id] = {
                "vector": vector,
                "metadata": metadata or {}
            }
            return {"success": True, "id": id}

        def search_similar(query_vector: List[float], k: int = 10):
            # Simple mock similarity search
            results = []
            for id, data in list(store.vectors.items())[:k]:
                results.append({
                    "id": id,
                    "score": np.random.uniform(0.7, 1.0),
                    "metadata": data["metadata"]
                })
            return sorted(results, key=lambda x: x["score"], reverse=True)

        store.add = add_vector
        store.search = search_similar
        return store


# Test Metrics Collector
class TestMetrics:
    """Collect and analyze test metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = defaultdict(list)
        self.start_time = time.time()

    def record(self, category: str, metric: str, value: float):
        """Record a metric."""
        self.metrics[category].append({
            "metric": metric,
            "value": value,
            "timestamp": time.time() - self.start_time
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {}

        for category, values in self.metrics.items():
            if values:
                metric_values = [v["value"] for v in values]
                summary[category] = {
                    "count": len(values),
                    "mean": np.mean(metric_values),
                    "std": np.std(metric_values),
                    "min": np.min(metric_values),
                    "max": np.max(metric_values),
                    "p50": np.percentile(metric_values, 50),
                    "p95": np.percentile(metric_values, 95),
                    "p99": np.percentile(metric_values, 99)
                }

        return summary


# Coverage Analyzer
class CoverageAnalyzer:
    """Analyze test coverage for agents."""

    def __init__(self, target_coverage: float = 0.90):
        """Initialize coverage analyzer."""
        self.target_coverage = target_coverage
        self.coverage_data = {}

    def analyze_module(self, module_name: str, coverage_percent: float):
        """Record module coverage."""
        self.coverage_data[module_name] = {
            "coverage": coverage_percent,
            "target": self.target_coverage,
            "passed": coverage_percent >= self.target_coverage
        }

    def get_report(self) -> Dict[str, Any]:
        """Generate coverage report."""
        total_modules = len(self.coverage_data)
        passed_modules = sum(1 for d in self.coverage_data.values() if d["passed"])

        overall_coverage = np.mean([d["coverage"] for d in self.coverage_data.values()]) \
                          if self.coverage_data else 0

        return {
            "overall_coverage": overall_coverage,
            "target_coverage": self.target_coverage,
            "passed": overall_coverage >= self.target_coverage,
            "modules": self.coverage_data,
            "summary": {
                "total_modules": total_modules,
                "passed_modules": passed_modules,
                "failed_modules": total_modules - passed_modules
            }
        }


# CPU Monitor
class CPUMonitor:
    """Monitor CPU usage during tests."""

    def __init__(self, interval: float = 0.1):
        """Initialize CPU monitor."""
        self.interval = interval
        self.monitoring = False
        self.thread = None
        self.cpu_samples = []
        self.max_cpu_percent = 0

    def start(self):
        """Start monitoring CPU."""
        self.monitoring = True
        self.cpu_samples = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop monitoring CPU."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1)

        if self.cpu_samples:
            self.max_cpu_percent = max(self.cpu_samples)

    def _monitor(self):
        """Monitor CPU in background thread."""
        process = psutil.Process()

        while self.monitoring:
            cpu_percent = process.cpu_percent()
            self.cpu_samples.append(cpu_percent)
            time.sleep(self.interval)


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=html", "--cov-report=term"])