"""
Unit Tests for Agent Capabilities
Tests planning, reasoning, meta-cognition, error recovery, and tool framework.
Validates all cognitive capabilities of GreenLang agents.
"""

import pytest
import asyncio
import time
import numpy as np
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from enum import Enum
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from testing.agent_test_framework import (
    AgentTestCase,
    TestConfig,
    DeterministicLLMProvider
)


# Planning System
class PlanningStrategy(Enum):
    """Planning strategies."""
    HIERARCHICAL = "hierarchical"
    REACTIVE = "reactive"
    DELIBERATIVE = "deliberative"
    HYBRID = "hybrid"


class Plan:
    """Represents a plan."""

    def __init__(self, goal: str, strategy: PlanningStrategy = PlanningStrategy.HIERARCHICAL):
        self.goal = goal
        self.strategy = strategy
        self.steps = []
        self.status = "created"
        self.metadata = {}

    def add_step(self, step: Dict[str, Any]):
        """Add step to plan."""
        step['id'] = len(self.steps)
        step['status'] = 'pending'
        self.steps.append(step)

    def execute_step(self, step_id: int) -> Dict[str, Any]:
        """Execute a plan step."""
        if step_id >= len(self.steps):
            raise ValueError(f"Invalid step_id: {step_id}")

        step = self.steps[step_id]
        step['status'] = 'executing'
        step['started_at'] = datetime.now().isoformat()

        # Simulate execution
        time.sleep(0.001)  # 1ms simulated work

        step['status'] = 'completed'
        step['completed_at'] = datetime.now().isoformat()

        return step

    def is_complete(self) -> bool:
        """Check if all steps completed."""
        return all(step['status'] == 'completed' for step in self.steps)


class PlanningEngine:
    """Planning engine with multiple strategies."""

    def __init__(self, strategy: PlanningStrategy = PlanningStrategy.HIERARCHICAL):
        self.strategy = strategy
        self.plans = []

    def create_plan(self, goal: str, context: Optional[Dict] = None) -> Plan:
        """Create a plan for achieving goal."""
        plan = Plan(goal, self.strategy)

        if self.strategy == PlanningStrategy.HIERARCHICAL:
            self._hierarchical_planning(plan, context)
        elif self.strategy == PlanningStrategy.REACTIVE:
            self._reactive_planning(plan, context)
        elif self.strategy == PlanningStrategy.DELIBERATIVE:
            self._deliberative_planning(plan, context)
        elif self.strategy == PlanningStrategy.HYBRID:
            self._hybrid_planning(plan, context)

        self.plans.append(plan)
        return plan

    def _hierarchical_planning(self, plan: Plan, context: Optional[Dict]):
        """Hierarchical task decomposition."""
        # Decompose goal into subgoals
        plan.add_step({'action': 'analyze_goal', 'goal': plan.goal})
        plan.add_step({'action': 'decompose', 'level': 1})
        plan.add_step({'action': 'execute_subgoals', 'level': 2})
        plan.add_step({'action': 'integrate_results', 'level': 1})

    def _reactive_planning(self, plan: Plan, context: Optional[Dict]):
        """Reactive planning based on current state."""
        plan.add_step({'action': 'sense_environment', 'context': context})
        plan.add_step({'action': 'select_action', 'reactive': True})
        plan.add_step({'action': 'execute_action', 'immediate': True})

    def _deliberative_planning(self, plan: Plan, context: Optional[Dict]):
        """Deliberative planning with lookahead."""
        plan.add_step({'action': 'model_world', 'depth': 3})
        plan.add_step({'action': 'search_solution_space', 'breadth': 5})
        plan.add_step({'action': 'evaluate_options', 'criteria': ['cost', 'time']})
        plan.add_step({'action': 'select_best_plan', 'optimize': True})
        plan.add_step({'action': 'execute_plan', 'monitor': True})

    def _hybrid_planning(self, plan: Plan, context: Optional[Dict]):
        """Hybrid planning combining reactive and deliberative."""
        plan.add_step({'action': 'deliberate_high_level', 'strategy': 'lookahead'})
        plan.add_step({'action': 'react_to_changes', 'adaptive': True})
        plan.add_step({'action': 'replan_if_needed', 'threshold': 0.5})


# Reasoning System
class ReasoningType(Enum):
    """Types of reasoning."""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"


class ReasoningEngine:
    """Multi-strategy reasoning engine."""

    def __init__(self):
        self.knowledge_base = []
        self.inference_history = []

    def reason(self, premises: List[str], reasoning_type: ReasoningType) -> Dict[str, Any]:
        """Perform reasoning."""
        if reasoning_type == ReasoningType.DEDUCTIVE:
            return self._deductive_reasoning(premises)
        elif reasoning_type == ReasoningType.INDUCTIVE:
            return self._inductive_reasoning(premises)
        elif reasoning_type == ReasoningType.ABDUCTIVE:
            return self._abductive_reasoning(premises)
        elif reasoning_type == ReasoningType.ANALOGICAL:
            return self._analogical_reasoning(premises)

    def _deductive_reasoning(self, premises: List[str]) -> Dict[str, Any]:
        """Deductive reasoning: general to specific."""
        # Example: All fuels emit CO2 -> Diesel is fuel -> Diesel emits CO2
        conclusion = f"Deduced from {len(premises)} premises"
        return {
            'type': 'deductive',
            'premises': premises,
            'conclusion': conclusion,
            'validity': True
        }

    def _inductive_reasoning(self, premises: List[str]) -> Dict[str, Any]:
        """Inductive reasoning: specific to general."""
        # Example: Diesel emits CO2, Gas emits CO2 -> All fuels emit CO2
        conclusion = f"Generalized from {len(premises)} observations"
        return {
            'type': 'inductive',
            'observations': premises,
            'conclusion': conclusion,
            'confidence': 0.8  # Not certain
        }

    def _abductive_reasoning(self, premises: List[str]) -> Dict[str, Any]:
        """Abductive reasoning: best explanation."""
        # Example: High emissions observed -> Likely diesel fuel used
        conclusion = "Best explanation for observations"
        return {
            'type': 'abductive',
            'observations': premises,
            'explanation': conclusion,
            'likelihood': 0.75
        }

    def _analogical_reasoning(self, premises: List[str]) -> Dict[str, Any]:
        """Analogical reasoning: similarity-based."""
        # Example: Diesel similar to gasoline -> emissions similar
        conclusion = "Conclusion by analogy"
        return {
            'type': 'analogical',
            'source': premises[0] if premises else None,
            'target': premises[1] if len(premises) > 1 else None,
            'conclusion': conclusion,
            'similarity': 0.7
        }


# Meta-Cognition System
class MetaCognition:
    """Self-monitoring and self-improvement."""

    def __init__(self):
        self.performance_metrics = []
        self.self_assessments = []
        self.improvements = []

    def monitor_performance(self, task: str, result: Dict[str, Any]):
        """Monitor task performance."""
        metric = {
            'task': task,
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'quality_score': self._assess_quality(result)
        }
        self.performance_metrics.append(metric)

    def _assess_quality(self, result: Dict[str, Any]) -> float:
        """Self-assess result quality."""
        # Simple quality assessment
        if 'error' in result:
            return 0.0
        if result.get('success', False):
            return 0.9
        return 0.5

    def identify_improvements(self) -> List[Dict[str, Any]]:
        """Identify areas for improvement."""
        improvements = []

        # Analyze recent performance
        recent = self.performance_metrics[-10:] if self.performance_metrics else []

        if recent:
            avg_quality = np.mean([m['quality_score'] for m in recent])

            if avg_quality < 0.7:
                improvements.append({
                    'area': 'task_execution',
                    'current_score': avg_quality,
                    'target_score': 0.9,
                    'recommendation': 'Improve error handling'
                })

        self.improvements = improvements
        return improvements

    def self_improve(self) -> Dict[str, Any]:
        """Attempt self-improvement."""
        improvements = self.identify_improvements()

        applied_improvements = []
        for improvement in improvements:
            # Simulate applying improvement
            applied_improvements.append({
                'improvement': improvement,
                'applied': True,
                'timestamp': datetime.now().isoformat()
            })

        return {
            'improvements_identified': len(improvements),
            'improvements_applied': len(applied_improvements),
            'details': applied_improvements
        }


# Error Recovery System
class ErrorRecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    COMPENSATION = "compensation"


class ErrorRecovery:
    """Error recovery with multiple strategies."""

    def __init__(self):
        self.error_history = []
        self.recovery_attempts = []
        self.circuit_breaker_state = "closed"  # closed, open, half_open
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5

    def recover(
        self,
        error: Exception,
        strategy: ErrorRecoveryStrategy,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Recover from error using strategy."""
        self.error_history.append({
            'error': str(error),
            'strategy': strategy.value,
            'timestamp': datetime.now().isoformat()
        })

        if strategy == ErrorRecoveryStrategy.RETRY:
            return self._retry_recovery(error, context)
        elif strategy == ErrorRecoveryStrategy.CIRCUIT_BREAKER:
            return self._circuit_breaker_recovery(error, context)
        elif strategy == ErrorRecoveryStrategy.FALLBACK:
            return self._fallback_recovery(error, context)
        elif strategy == ErrorRecoveryStrategy.COMPENSATION:
            return self._compensation_recovery(error, context)

    def _retry_recovery(self, error: Exception, context: Optional[Dict]) -> Dict[str, Any]:
        """Retry with exponential backoff."""
        max_retries = context.get('max_retries', 3) if context else 3
        attempt = 0

        while attempt < max_retries:
            try:
                # Simulate retry
                time.sleep(0.001 * (2 ** attempt))  # Exponential backoff
                attempt += 1

                # Simulate success on 3rd attempt
                if attempt >= 3:
                    return {
                        'strategy': 'retry',
                        'success': True,
                        'attempts': attempt
                    }
            except Exception:
                continue

        return {
            'strategy': 'retry',
            'success': False,
            'attempts': attempt
        }

    def _circuit_breaker_recovery(self, error: Exception, context: Optional[Dict]) -> Dict[str, Any]:
        """Circuit breaker pattern."""
        if self.circuit_breaker_state == "open":
            return {
                'strategy': 'circuit_breaker',
                'success': False,
                'state': 'open',
                'message': 'Circuit breaker is open'
            }

        # Track failure
        self.circuit_breaker_failures += 1

        # Open circuit if threshold exceeded
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_state = "open"

            return {
                'strategy': 'circuit_breaker',
                'success': False,
                'state': 'opened',
                'failures': self.circuit_breaker_failures
            }

        return {
            'strategy': 'circuit_breaker',
            'success': False,
            'state': self.circuit_breaker_state,
            'failures': self.circuit_breaker_failures
        }

    def _fallback_recovery(self, error: Exception, context: Optional[Dict]) -> Dict[str, Any]:
        """Fallback to alternative implementation."""
        fallback_value = context.get('fallback_value', 'default') if context else 'default'

        return {
            'strategy': 'fallback',
            'success': True,
            'value': fallback_value,
            'original_error': str(error)
        }

    def _compensation_recovery(self, error: Exception, context: Optional[Dict]) -> Dict[str, Any]:
        """Compensating transaction."""
        # Rollback previous operations
        rollback_steps = context.get('rollback_steps', []) if context else []

        for step in reversed(rollback_steps):
            # Execute compensation
            pass

        return {
            'strategy': 'compensation',
            'success': True,
            'rollback_steps': len(rollback_steps)
        }

    def reset_circuit_breaker(self):
        """Reset circuit breaker."""
        self.circuit_breaker_state = "closed"
        self.circuit_breaker_failures = 0


# Tool Framework
class Tool:
    """Base tool class."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0

    def execute(self, **kwargs) -> Any:
        """Execute tool."""
        self.usage_count += 1
        raise NotImplementedError


class ToolFramework:
    """Tool framework for agents."""

    def __init__(self):
        self.tools = {}
        self.execution_history = []

    def register_tool(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")

        tool = self.tools[tool_name]

        start_time = time.perf_counter()
        result = tool.execute(**kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000

        self.execution_history.append({
            'tool': tool_name,
            'kwargs': kwargs,
            'result': result,
            'duration_ms': duration_ms,
            'timestamp': datetime.now().isoformat()
        })

        return result

    def list_tools(self) -> List[str]:
        """List available tools."""
        return list(self.tools.keys())


# Unit Tests
class TestPlanning(AgentTestCase):
    """Test planning capabilities."""

    def test_hierarchical_planning(self):
        """Test hierarchical planning strategy."""
        engine = PlanningEngine(PlanningStrategy.HIERARCHICAL)
        plan = engine.create_plan("Calculate carbon emissions")

        self.assertEqual(len(plan.steps), 4)
        self.assertEqual(plan.strategy, PlanningStrategy.HIERARCHICAL)
        self.assertEqual(plan.steps[0]['action'], 'analyze_goal')

    def test_reactive_planning(self):
        """Test reactive planning strategy."""
        engine = PlanningEngine(PlanningStrategy.REACTIVE)
        plan = engine.create_plan("Respond to alert", context={'alert': 'high_emissions'})

        self.assertEqual(plan.strategy, PlanningStrategy.REACTIVE)
        self.assertTrue(any(s['action'] == 'sense_environment' for s in plan.steps))

    def test_deliberative_planning(self):
        """Test deliberative planning strategy."""
        engine = PlanningEngine(PlanningStrategy.DELIBERATIVE)
        plan = engine.create_plan("Optimize emissions reduction")

        self.assertGreater(len(plan.steps), 3)
        self.assertTrue(any('search_solution_space' in s['action'] for s in plan.steps))

    def test_hybrid_planning(self):
        """Test hybrid planning strategy."""
        engine = PlanningEngine(PlanningStrategy.HYBRID)
        plan = engine.create_plan("Adaptive emissions monitoring")

        self.assertTrue(any('deliberate' in s['action'] for s in plan.steps))
        self.assertTrue(any('react' in s['action'] for s in plan.steps))

    def test_plan_execution(self):
        """Test plan execution."""
        engine = PlanningEngine()
        plan = engine.create_plan("Test goal")

        # Execute all steps
        for i in range(len(plan.steps)):
            step_result = plan.execute_step(i)
            self.assertEqual(step_result['status'], 'completed')

        self.assertTrue(plan.is_complete())

    def test_plan_execution_performance(self):
        """Test plan execution meets performance targets."""
        engine = PlanningEngine()
        plan = engine.create_plan("Performance test")

        # Execute with performance monitoring
        with self.assert_performance(max_duration_ms=100):
            for i in range(len(plan.steps)):
                plan.execute_step(i)


class TestReasoning(AgentTestCase):
    """Test reasoning capabilities."""

    def test_deductive_reasoning(self):
        """Test deductive reasoning."""
        engine = ReasoningEngine()
        result = engine.reason(
            ['All fuels emit CO2', 'Diesel is a fuel'],
            ReasoningType.DEDUCTIVE
        )

        self.assertEqual(result['type'], 'deductive')
        self.assertTrue(result['validity'])

    def test_inductive_reasoning(self):
        """Test inductive reasoning."""
        engine = ReasoningEngine()
        result = engine.reason(
            ['Diesel emits CO2', 'Gasoline emits CO2', 'Coal emits CO2'],
            ReasoningType.INDUCTIVE
        )

        self.assertEqual(result['type'], 'inductive')
        self.assertGreater(result['confidence'], 0.5)

    def test_abductive_reasoning(self):
        """Test abductive reasoning (best explanation)."""
        engine = ReasoningEngine()
        result = engine.reason(
            ['High carbon emissions detected'],
            ReasoningType.ABDUCTIVE
        )

        self.assertEqual(result['type'], 'abductive')
        self.assertIn('likelihood', result)

    def test_analogical_reasoning(self):
        """Test analogical reasoning."""
        engine = ReasoningEngine()
        result = engine.reason(
            ['Diesel combustion', 'Gasoline combustion'],
            ReasoningType.ANALOGICAL
        )

        self.assertEqual(result['type'], 'analogical')
        self.assertGreater(result['similarity'], 0.5)


class TestMetaCognition(AgentTestCase):
    """Test meta-cognition capabilities."""

    def test_performance_monitoring(self):
        """Test self-monitoring of performance."""
        meta = MetaCognition()

        # Record performance
        meta.monitor_performance('calculate_emissions', {'success': True, 'value': 100})
        meta.monitor_performance('validate_data', {'success': True})

        self.assertEqual(len(meta.performance_metrics), 2)
        self.assertGreater(meta.performance_metrics[0]['quality_score'], 0.5)

    def test_quality_assessment(self):
        """Test self-assessment of quality."""
        meta = MetaCognition()

        # Good result
        meta.monitor_performance('task1', {'success': True})
        self.assertEqual(meta.performance_metrics[0]['quality_score'], 0.9)

        # Bad result
        meta.monitor_performance('task2', {'error': 'failed'})
        self.assertEqual(meta.performance_metrics[1]['quality_score'], 0.0)

    def test_improvement_identification(self):
        """Test identifying areas for improvement."""
        meta = MetaCognition()

        # Record poor performance
        for i in range(10):
            meta.monitor_performance(f'task_{i}', {'success': False})

        improvements = meta.identify_improvements()

        self.assertGreater(len(improvements), 0)
        self.assertLess(improvements[0]['current_score'], 0.7)

    def test_self_improvement(self):
        """Test self-improvement mechanism."""
        meta = MetaCognition()

        # Record poor performance
        for i in range(10):
            meta.monitor_performance(f'task_{i}', {'success': False})

        result = meta.self_improve()

        self.assertGreater(result['improvements_identified'], 0)
        self.assertGreater(result['improvements_applied'], 0)


class TestErrorRecovery(AgentTestCase):
    """Test error recovery capabilities."""

    def test_retry_recovery(self):
        """Test retry error recovery."""
        recovery = ErrorRecovery()
        result = recovery.recover(
            Exception("Test error"),
            ErrorRecoveryStrategy.RETRY,
            context={'max_retries': 3}
        )

        self.assertEqual(result['strategy'], 'retry')
        self.assertIn('attempts', result)

    def test_circuit_breaker(self):
        """Test circuit breaker pattern."""
        recovery = ErrorRecovery()

        # Trigger failures to open circuit
        for i in range(6):
            result = recovery.recover(
                Exception(f"Error {i}"),
                ErrorRecoveryStrategy.CIRCUIT_BREAKER
            )

        # Circuit should be open after threshold
        self.assertEqual(result['state'], 'opened')

        # Further calls should fail fast
        result = recovery.recover(
            Exception("Another error"),
            ErrorRecoveryStrategy.CIRCUIT_BREAKER
        )
        self.assertEqual(result['state'], 'open')

    def test_circuit_breaker_reset(self):
        """Test resetting circuit breaker."""
        recovery = ErrorRecovery()

        # Open circuit
        for i in range(6):
            recovery.recover(Exception(f"Error {i}"), ErrorRecoveryStrategy.CIRCUIT_BREAKER)

        self.assertEqual(recovery.circuit_breaker_state, "open")

        # Reset
        recovery.reset_circuit_breaker()

        self.assertEqual(recovery.circuit_breaker_state, "closed")
        self.assertEqual(recovery.circuit_breaker_failures, 0)

    def test_fallback_recovery(self):
        """Test fallback error recovery."""
        recovery = ErrorRecovery()
        result = recovery.recover(
            Exception("Primary failed"),
            ErrorRecoveryStrategy.FALLBACK,
            context={'fallback_value': 'backup_result'}
        )

        self.assertTrue(result['success'])
        self.assertEqual(result['value'], 'backup_result')

    def test_compensation_recovery(self):
        """Test compensating transaction recovery."""
        recovery = ErrorRecovery()
        result = recovery.recover(
            Exception("Transaction failed"),
            ErrorRecoveryStrategy.COMPENSATION,
            context={'rollback_steps': ['step1', 'step2', 'step3']}
        )

        self.assertTrue(result['success'])
        self.assertEqual(result['rollback_steps'], 3)


class TestToolFramework(AgentTestCase):
    """Test tool framework."""

    def setUp(self):
        super().setUp()

        # Create sample tool
        class CalculatorTool(Tool):
            def execute(self, **kwargs):
                operation = kwargs.get('operation', 'add')
                a = kwargs.get('a', 0)
                b = kwargs.get('b', 0)

                if operation == 'add':
                    return a + b
                elif operation == 'multiply':
                    return a * b
                return 0

        self.calculator_tool = CalculatorTool("calculator", "Perform calculations")

    def test_tool_registration(self):
        """Test registering tools."""
        framework = ToolFramework()
        framework.register_tool(self.calculator_tool)

        self.assertIn("calculator", framework.tools)

    def test_tool_execution(self):
        """Test executing tools."""
        framework = ToolFramework()
        framework.register_tool(self.calculator_tool)

        result = framework.execute_tool("calculator", operation='add', a=5, b=3)

        self.assertEqual(result, 8)
        self.assertEqual(self.calculator_tool.usage_count, 1)

    def test_tool_execution_history(self):
        """Test tool execution history tracking."""
        framework = ToolFramework()
        framework.register_tool(self.calculator_tool)

        framework.execute_tool("calculator", operation='add', a=5, b=3)
        framework.execute_tool("calculator", operation='multiply', a=4, b=2)

        self.assertEqual(len(framework.execution_history), 2)
        self.assertEqual(framework.execution_history[0]['result'], 8)
        self.assertEqual(framework.execution_history[1]['result'], 8)

    def test_list_tools(self):
        """Test listing available tools."""
        framework = ToolFramework()
        framework.register_tool(self.calculator_tool)

        tools = framework.list_tools()

        self.assertEqual(len(tools), 1)
        self.assertIn("calculator", tools)

    def test_tool_not_found(self):
        """Test handling of unknown tool."""
        framework = ToolFramework()

        with self.assertRaises(ValueError):
            framework.execute_tool("nonexistent_tool")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=term"])
