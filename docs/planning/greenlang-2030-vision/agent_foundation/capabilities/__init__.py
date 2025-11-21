# -*- coding: utf-8 -*-
"""
Agent Capabilities Framework for GreenLang Agent Foundation.

This package provides comprehensive capabilities for GreenLang agents including:
- Tool use and function calling with sandboxing
- Planning algorithms (hierarchical, reactive, deliberative, hybrid)
- Reasoning engines (deductive, inductive, abductive, analogical)
- Meta-cognition and self-reflection
- Multi-step task execution
- Error recovery and resilience patterns

Example:
    >>> from capabilities import ToolFramework, PlanningFramework
    >>> tool_framework = ToolFramework(config)
    >>> result = await tool_framework.execute_tool("carbon_calculator", params)
"""

from capabilities.tool_framework import (
    ToolFramework,
    ToolExecutor,
    ToolRegistry,
    ToolPermission,
    ToolSandbox,
    FunctionCallingProtocol
)

from capabilities.planning import (
    PlanningFramework,
    HierarchicalPlanner,
    ReactivePlanner,
    DeliberativePlanner,
    HybridPlanner,
    PlanExecutor,
    PlanningAlgorithm
)

from capabilities.reasoning import (
    ReasoningFramework,
    DeductiveReasoner,
    InductiveReasoner,
    AbductiveReasoner,
    AnalogicalReasoner,
    ReasoningEngine,
    InferenceResult
)

from capabilities.meta_cognition import (
    MetaCognition,
    SelfMonitor,
    SelfImprovement,
    ConfidenceEstimator,
    MetaReasoner,
    PerformanceTracker,
    ExperienceDatabase
)

from capabilities.task_executor import (
    TaskExecutor,
    TaskDecomposer,
    ExecutionStrategy,
    StateManager,
    CheckpointManager,
    ProgressTracker,
    TaskResult
)

from capabilities.error_recovery import (
    ResilienceFramework,
    ErrorHandler,
    RecoveryStrategy,
    CircuitBreaker,
    RetryPolicy,
    FallbackHandler,
    CompensationHandler
)

__all__ = [
    # Tool Framework
    'ToolFramework',
    'ToolExecutor',
    'ToolRegistry',
    'ToolPermission',
    'ToolSandbox',
    'FunctionCallingProtocol',

    # Planning
    'PlanningFramework',
    'HierarchicalPlanner',
    'ReactivePlanner',
    'DeliberativePlanner',
    'HybridPlanner',
    'PlanExecutor',
    'PlanningAlgorithm',

    # Reasoning
    'ReasoningFramework',
    'DeductiveReasoner',
    'InductiveReasoner',
    'AbductiveReasoner',
    'AnalogicalReasoner',
    'ReasoningEngine',
    'InferenceResult',

    # Meta-Cognition
    'MetaCognition',
    'SelfMonitor',
    'SelfImprovement',
    'ConfidenceEstimator',
    'MetaReasoner',
    'PerformanceTracker',
    'ExperienceDatabase',

    # Task Execution
    'TaskExecutor',
    'TaskDecomposer',
    'ExecutionStrategy',
    'StateManager',
    'CheckpointManager',
    'ProgressTracker',
    'TaskResult',

    # Error Recovery
    'ResilienceFramework',
    'ErrorHandler',
    'RecoveryStrategy',
    'CircuitBreaker',
    'RetryPolicy',
    'FallbackHandler',
    'CompensationHandler',
]

__version__ = '1.0.0'