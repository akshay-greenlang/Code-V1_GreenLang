"""
GreenLang Workflow Exceptions - Workflow Orchestration Errors

This module provides exception classes for workflow-related errors.

Features:
- DAG validation errors
- Policy violation tracking
- Resource constraint errors
- Orchestration failures

Author: GreenLang Team
Date: 2025-11-21
"""

from typing import Any, Dict, Optional

from greenlang.exceptions.base import GreenLangException


class WorkflowException(GreenLangException):
    """Base exception for workflow-related errors.

    Raised when workflow orchestration encounters an error.
    """
    ERROR_PREFIX = "GL_WORKFLOW"


class DAGError(WorkflowException):
    """Workflow DAG (Directed Acyclic Graph) is invalid.

    Raised when workflow DAG contains cycles, missing dependencies, or other
    structural issues.

    Example:
        >>> raise DAGError(
        ...     message="Cycle detected in workflow DAG",
        ...     context={"cycle": ["step1", "step2", "step3", "step1"]}
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        invalid_nodes: Optional[list] = None,
    ):
        """Initialize DAG error.

        Args:
            message: Error message
            context: Error context
            workflow_id: ID of workflow with invalid DAG
            invalid_nodes: List of problematic nodes
        """
        context = context or {}
        if workflow_id:
            context["workflow_id"] = workflow_id
        if invalid_nodes:
            context["invalid_nodes"] = invalid_nodes
        super().__init__(message, context=context)


class PolicyViolation(WorkflowException):
    """Workflow violates policy constraints.

    Raised when workflow execution violates security, compliance, or resource
    policies.

    Example:
        >>> raise PolicyViolation(
        ...     message="Workflow exceeds maximum execution time policy",
        ...     context={"policy": "max_execution_time", "limit": 300, "actual": 350}
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        policy_name: Optional[str] = None,
        violation_details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize policy violation.

        Args:
            message: Error message
            context: Error context
            policy_name: Name of violated policy
            violation_details: Details about the violation
        """
        context = context or {}
        if policy_name:
            context["policy_name"] = policy_name
        if violation_details:
            context["violation_details"] = violation_details
        super().__init__(message, context=context)


class ResourceError(WorkflowException):
    """Workflow resource constraint exceeded.

    Raised when workflow exceeds resource limits (CPU, memory, disk, network).

    Example:
        >>> raise ResourceError(
        ...     message="Memory limit exceeded",
        ...     context={"resource": "memory", "limit_mb": 1024, "used_mb": 1500}
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        resource_type: Optional[str] = None,
        limit: Optional[float] = None,
        used: Optional[float] = None,
    ):
        """Initialize resource error.

        Args:
            message: Error message
            context: Error context
            resource_type: Type of resource (cpu, memory, disk, network)
            limit: Resource limit
            used: Actual resource usage
        """
        context = context or {}
        if resource_type:
            context["resource_type"] = resource_type
        if limit:
            context["limit"] = limit
        if used:
            context["used"] = used
        super().__init__(message, context=context)


class OrchestrationError(WorkflowException):
    """Workflow orchestration failed.

    Raised when workflow orchestrator encounters an error coordinating agent
    execution.

    Example:
        >>> raise OrchestrationError(
        ...     message="Failed to coordinate parallel agent execution",
        ...     context={"failed_agents": ["agent1", "agent2"], "reason": "Deadlock"}
        ... )
    """
    pass


__all__ = [
    'WorkflowException',
    'DAGError',
    'PolicyViolation',
    'ResourceError',
    'OrchestrationError',
]
