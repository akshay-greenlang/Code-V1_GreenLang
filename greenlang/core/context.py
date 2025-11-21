# -*- coding: utf-8 -*-
"""
GreenLang Execution Context
============================

Provides execution context management for agents and pipelines.
Manages request-scoped data, correlation IDs, and execution metadata.

Author: GreenLang Framework Team
"""

from typing import Dict, Any, Optional, List
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
import threading
import logging

from greenlang.determinism import deterministic_uuid

logger = logging.getLogger(__name__)

# Thread-local context storage
_context_var: ContextVar[Optional['ExecutionContext']] = ContextVar('execution_context', default=None)


@dataclass
class ExecutionContext:
    """
    Represents the execution context for an agent or pipeline.

    Tracks request-scoped data, correlation IDs, and metadata
    throughout the execution lifecycle.
    """

    # Core identifiers (using deterministic UUIDs for reproducibility)
    request_id: str = field(default_factory=lambda: deterministic_uuid(f"request:{datetime.now().isoformat()}"))
    correlation_id: str = field(default_factory=lambda: deterministic_uuid(f"correlation:{datetime.now().isoformat()}"))
    session_id: Optional[str] = None

    # Execution metadata
    start_time: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    environment: str = "production"

    # Request data
    input_data: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)

    # Execution state
    variables: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Feature flags
    features: Dict[str, bool] = field(default_factory=dict)

    def set_variable(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self.variables[key] = value
        logger.debug(f"Set context variable: {key}")

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.variables.get(key, default)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the context."""
        if tag not in self.tags:
            self.tags.append(tag)

    def record_metric(self, name: str, value: Any) -> None:
        """Record a metric."""
        self.metrics[name] = value

    def add_error(self, error: Exception, details: Optional[Dict] = None) -> None:
        """Record an error in the context."""
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.errors.append(error_info)
        logger.warning(f"Recorded error in context: {error_info['type']}")

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self.features.get(feature, False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "environment": self.environment,
            "tags": self.tags,
            "metrics": self.metrics,
            "error_count": len(self.errors),
            "features": self.features
        }

    def create_child_context(self) -> 'ExecutionContext':
        """
        Create a child context that inherits from this one.

        Returns:
            New ExecutionContext with inherited values
        """
        child = ExecutionContext(
            correlation_id=self.correlation_id,  # Preserve correlation
            session_id=self.session_id,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            environment=self.environment
        )

        # Copy feature flags and tags
        child.features = self.features.copy()
        child.tags = self.tags.copy()

        return child


def get_current_context() -> Optional[ExecutionContext]:
    """Get the current execution context."""
    return _context_var.get()


def set_current_context(context: ExecutionContext) -> None:
    """Set the current execution context."""
    _context_var.set(context)
    logger.debug(f"Set execution context: {context.request_id}")


def create_context(**kwargs) -> ExecutionContext:
    """
    Create and set a new execution context.

    Args:
        **kwargs: Arguments to pass to ExecutionContext

    Returns:
        The newly created context
    """
    context = ExecutionContext(**kwargs)
    set_current_context(context)
    return context


def clear_context() -> None:
    """Clear the current execution context."""
    _context_var.set(None)
    logger.debug("Cleared execution context")


class ContextManager:
    """Context manager for execution context."""

    def __init__(self, context: Optional[ExecutionContext] = None, **kwargs):
        """
        Initialize context manager.

        Args:
            context: Existing context to use, or None to create new
            **kwargs: Arguments for creating new context
        """
        self.context = context or ExecutionContext(**kwargs)
        self.previous_context: Optional[ExecutionContext] = None

    def __enter__(self) -> ExecutionContext:
        """Enter the context."""
        self.previous_context = get_current_context()
        set_current_context(self.context)
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context."""
        if self.previous_context:
            set_current_context(self.previous_context)
        else:
            clear_context()