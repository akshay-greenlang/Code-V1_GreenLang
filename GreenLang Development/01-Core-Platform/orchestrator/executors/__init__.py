# -*- coding: utf-8 -*-
"""
Orchestrator Execution Backends
================================

Provides pluggable execution backends for GLIP v1 agents.

Available Backends:
    - K8sExecutor: Kubernetes Jobs (primary)
    - LocalExecutor: Local subprocess (development)
    - LegacyHttpExecutor: HTTP wrapper for legacy agents

Author: GreenLang Team
"""

from greenlang.orchestrator.executors.base import (
    ExecutorBackend,
    RunContext,
    StepResult,
    ExecutionStatus,
    StepMetadata,
    ResourceProfile,
    ExecutionResult,
)
# Alias for compatibility
StepStatus = ExecutionStatus

# K8s executor is optional (requires kubernetes_asyncio)
try:
    from greenlang.orchestrator.executors.k8s_executor import K8sExecutor, K8sExecutorConfig
    KUBERNETES_AVAILABLE = True
except ImportError:
    K8sExecutor = None
    K8sExecutorConfig = None
    KUBERNETES_AVAILABLE = False

__all__ = [
    "ExecutorBackend",
    "RunContext",
    "StepResult",
    "ExecutionStatus",
    "StepStatus",  # Alias for ExecutionStatus
    "StepMetadata",
    "ResourceProfile",
    "ExecutionResult",
    "K8sExecutor",
    "K8sExecutorConfig",
    "KUBERNETES_AVAILABLE",
]
