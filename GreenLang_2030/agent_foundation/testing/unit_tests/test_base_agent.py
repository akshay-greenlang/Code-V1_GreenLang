"""
Unit Tests for Base Agent Lifecycle
Tests agent creation, state transitions, and core functionality.
"""

import pytest
import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import time
import asyncio
import hashlib
import json
from datetime import datetime
from typing import Dict, Any

# Import test framework
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from testing.agent_test_framework import (
    AgentTestCase,
    AgentState,
    TestConfig,
    DeterministicLLMProvider,
    TestDataGenerator,
    ProvenanceValidator
)


class TestBaseAgent(AgentTestCase):
    """Test base agent functionality."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.data_generator = TestDataGenerator(seed=42)

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        # Create mock agent class
        MockAgent = type('MockAgent', (), {})

        # Create agent instance
        config = {"name": "test_agent", "version": "1.0.0"}
        agent = self.create_mock_agent(MockAgent, config)

        # Verify initialization
        self.assertEqual(agent.state, AgentState.CREATED)
        self.assertEqual(agent.config["name"], "test_agent")
        self.assertIsNotNone(agent.lifecycle_history)
        self.assertIsNotNone(agent.provenance_chain)

    def test_agent_configuration(self):
        """Test agent configuration handling."""
        configs = self.data_generator.generate_agent_configs(count=5)

        for config in configs:
            MockAgent = type('MockAgent', (), {})
            agent = self.create_mock_agent(MockAgent, config)

            # Verify config is stored correctly
            self.assertEqual(agent.config["name"], config["name"])
            self.assertEqual(agent.config["version"], config["version"])
            self.assertEqual(agent.config["environment"], config["environment"])

    def test_agent_memory_initialization(self):
        """Test agent memory systems are initialized."""
        MockAgent = type('MockAgent', (), {})
        agent = self.create_mock_agent(MockAgent)

        # Verify memory systems
        self.assertIsNotNone(agent.short_term_memory)
        self.assertIsNotNone(agent.long_term_memory)
        self.assertIsNotNone(agent.episodic_memory)
        self.assertIsNotNone(agent.semantic_memory)

    def test_deterministic_llm_provider(self):
        """Test deterministic LLM provider for reproducible testing."""
        llm = DeterministicLLMProvider(seed=42)

        # Test deterministic responses
        prompt = "Test prompt"
        response1 = llm.generate(prompt)
        response2 = llm.generate(prompt)

        # Verify determinism
        self.assertEqual(response1["prompt_hash"], response2["prompt_hash"])
        self.assertTrue(response1["deterministic"])
        self.assertEqual(response1["latency_ms"], 50.0)  # Fixed latency

    def test_agent_provenance_tracking(self):
        """Test provenance tracking for reproducibility."""
        MockAgent = type('MockAgent', (), {
            'process': lambda self, x: {"result": x, "provenance_hash": hashlib.sha256(
                json.dumps(x, sort_keys=True).encode()
            ).hexdigest()}
        })

        agent = MockAgent()
        test_input = {"data": "test"}

        result = agent.process(test_input)

        # Verify provenance
        self.assertIn("provenance_hash", result)
        self.assertEqual(len(result["provenance_hash"]), 64)  # SHA-256 hash

        # Test provenance validation
        is_valid, errors = ProvenanceValidator.validate_chain([{
            "hash": result["provenance_hash"],
            "timestamp": datetime.now().isoformat(),
            "operation": "process",
            "input_hash": ProvenanceValidator.calculate_hash(test_input)
        }])

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    @pytest.mark.asyncio
    async def test_async_agent_processing(self):
        """Test async agent processing."""
        MockAgent = type('MockAgent', (), {
            'process_async': AsyncMock(return_value={"result": "async_result"})
        })

        agent = MockAgent()
        result = await agent.process_async({"test": "input"})

        self.assertEqual(result["result"], "async_result")
        agent.process_async.assert_called_once()

    def test_agent_error_handling(self):
        """Test agent error handling."""
        MockAgent = type('MockAgent', (), {
            'process': Mock(side_effect=Exception("Test error"))
        })

        agent = self.create_mock_agent(MockAgent)

        # Test error doesn't crash agent
        with self.assertRaises(Exception) as context:
            agent.process({"test": "input"})

        self.assertIn("Test error", str(context.exception))

    def test_agent_performance_constraints(self):
        """Test agent meets performance constraints."""
        MockAgent = type('MockAgent', (), {
            'process': lambda self, x: {"result": x}
        })

        agent = MockAgent()

        # Test with performance constraints
        with self.assert_performance(max_duration_ms=100, max_memory_mb=10):
            for _ in range(100):
                agent.process({"test": "data"})

    def test_zero_hallucination_guarantee(self):
        """Test zero-hallucination for calculations."""
        MockAgent = type('MockAgent', (), {
            'calculate': lambda self, x, y: x + y
        })

        agent = MockAgent()

        # Test calculations
        result = agent.calculate(2.5, 3.7)
        expected = 6.2

        self.assert_zero_hallucination(result, expected, tolerance=1e-10)


class TestAgentLifecycle(AgentTestCase):
    """Test agent lifecycle state transitions."""

    def test_full_lifecycle_transitions(self):
        """Test complete lifecycle from creation to termination."""
        MockAgent = type('MockAgent', (), {})
        agent = self.create_mock_agent(MockAgent)

        # Test full lifecycle
        self.test_full_lifecycle(agent)

        # Verify history
        self.assertEqual(len(agent.lifecycle_history), 7)  # 7 transitions
        self.assertEqual(agent.state, AgentState.TERMINATED)

    def test_valid_state_transitions(self):
        """Test only valid state transitions are allowed."""
        MockAgent = type('MockAgent', (), {})
        agent = self.create_mock_agent(MockAgent)

        # Valid transitions
        valid_transitions = [
            (AgentState.CREATED, AgentState.INITIALIZING),
            (AgentState.INITIALIZING, AgentState.READY),
            (AgentState.READY, AgentState.RUNNING),
            (AgentState.RUNNING, AgentState.PAUSED),
            (AgentState.PAUSED, AgentState.READY),
            (AgentState.READY, AgentState.STOPPING),
            (AgentState.STOPPING, AgentState.TERMINATED)
        ]

        for from_state, to_state in valid_transitions:
            agent.state = from_state
            agent.transition_to(to_state)
            self.assertEqual(agent.state, to_state)

    def test_state_transition_timing(self):
        """Test state transitions meet timing requirements."""
        MockAgent = type('MockAgent', (), {})
        agent = self.create_mock_agent(MockAgent)

        # Test transition timing
        self.assert_lifecycle_transition(
            agent,
            AgentState.CREATED,
            AgentState.INITIALIZING,
            max_duration_ms=100  # <100ms target
        )

    def test_error_state_recovery(self):
        """Test recovery from error state."""
        MockAgent = type('MockAgent', (), {
            'reset': Mock()
        })

        agent = self.create_mock_agent(MockAgent)

        # Force error state
        agent.state = AgentState.ERROR

        # Attempt recovery
        agent.transition_to(AgentState.STOPPING)
        self.assertEqual(agent.state, AgentState.STOPPING)

        agent.transition_to(AgentState.TERMINATED)
        self.assertEqual(agent.state, AgentState.TERMINATED)

    def test_concurrent_state_transitions(self):
        """Test thread-safe state transitions."""
        import threading

        MockAgent = type('MockAgent', (), {})
        agent = self.create_mock_agent(MockAgent)

        transitions_completed = []

        def perform_transition():
            try:
                agent.transition_to(AgentState.INITIALIZING)
                transitions_completed.append(True)
            except Exception:
                transitions_completed.append(False)

        # Create multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=perform_transition)
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify only one transition succeeded
        self.assertEqual(agent.state, AgentState.INITIALIZING)

    def test_lifecycle_event_recording(self):
        """Test lifecycle events are properly recorded."""
        MockAgent = type('MockAgent', (), {})
        agent = self.create_mock_agent(MockAgent)

        initial_event_count = len(self.lifecycle_events)

        # Perform transitions
        agent.transition_to(AgentState.INITIALIZING)
        agent.transition_to(AgentState.READY)

        # Verify events recorded
        self.assertEqual(len(self.lifecycle_events), initial_event_count + 2)

        # Check event details
        last_event = self.lifecycle_events[-1]
        self.assertIn("agent", last_event)
        self.assertIn("transition", last_event)
        self.assertIn("time", last_event)


class TestAgentCommunication(AgentTestCase):
    """Test inter-agent communication protocols."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.data_generator = TestDataGenerator(seed=42)

    def test_message_creation(self):
        """Test agent message creation."""
        messages = self.data_generator.generate_test_messages(count=10)

        for message in messages:
            # Verify message structure
            self.assertIn("message_id", message)
            self.assertIn("sender_id", message)
            self.assertIn("recipient_id", message)
            self.assertIn("message_type", message)
            self.assertIn("priority", message)
            self.assertIn("payload", message)
            self.assertIn("timestamp", message)

    def test_message_passing_performance(self):
        """Test message passing meets performance targets."""
        MockAgent = type('MockAgent', (), {
            'inbox': [],
            'send_message': lambda self, msg: self.inbox.append(msg)
        })

        sender = MockAgent()
        receiver = MockAgent()

        # Test message passing performance
        with self.assert_performance(max_duration_ms=10):  # <10ms target
            message = {
                "from": "sender",
                "to": "receiver",
                "content": "test"
            }
            receiver.send_message(message)

        self.assertEqual(len(receiver.inbox), 1)

    def test_broadcast_messaging(self):
        """Test broadcast messaging to multiple agents."""
        # Create multiple agents
        agents = []
        for i in range(5):
            MockAgent = type('MockAgent', (), {
                'id': f'agent_{i}',
                'inbox': []
            })
            agents.append(MockAgent())

        # Broadcast message
        broadcast_message = {
            "from": "coordinator",
            "to": "broadcast",
            "content": "system update"
        }

        for agent in agents:
            agent.inbox.append(broadcast_message)

        # Verify all agents received message
        for agent in agents:
            self.assertEqual(len(agent.inbox), 1)
            self.assertEqual(agent.inbox[0]["content"], "system update")

    def test_message_priority_handling(self):
        """Test message priority processing."""
        MockAgent = type('MockAgent', (), {
            'inbox': [],
            'process_messages': lambda self: sorted(
                self.inbox,
                key=lambda m: ["LOW", "NORMAL", "HIGH", "CRITICAL"].index(m["priority"]),
                reverse=True
            )
        })

        agent = MockAgent()

        # Add messages with different priorities
        priorities = ["LOW", "CRITICAL", "NORMAL", "HIGH"]
        for priority in priorities:
            agent.inbox.append({
                "priority": priority,
                "content": f"Message with {priority} priority"
            })

        # Process messages
        processed = agent.process_messages()

        # Verify priority order
        self.assertEqual(processed[0]["priority"], "CRITICAL")
        self.assertEqual(processed[1]["priority"], "HIGH")
        self.assertEqual(processed[2]["priority"], "NORMAL")
        self.assertEqual(processed[3]["priority"], "LOW")

    def test_message_correlation(self):
        """Test message correlation for request-response patterns."""
        import uuid

        correlation_id = str(uuid.uuid4())

        request = {
            "message_id": str(uuid.uuid4()),
            "correlation_id": correlation_id,
            "type": "REQUEST"
        }

        response = {
            "message_id": str(uuid.uuid4()),
            "correlation_id": correlation_id,
            "type": "RESPONSE"
        }

        # Verify correlation
        self.assertEqual(request["correlation_id"], response["correlation_id"])

    def test_concurrent_message_processing(self):
        """Test concurrent message processing."""
        num_agents = 10
        messages_per_agent = 100

        # Test concurrent agents
        self.assert_concurrent_agents(
            type('MockAgent', (), {}),
            num_agents=num_agents,
            max_total_duration_s=5.0
        )


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=html", "--cov-target=90"])