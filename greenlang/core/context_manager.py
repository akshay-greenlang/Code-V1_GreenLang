"""
Unified Context Manager for GreenLang

This module provides a centralized context management system that coordinates
between different context types across the GreenLang framework.
"""

import os
import json
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum


class ContextType(Enum):
    """Types of context available in the system"""

    EXECUTION = "execution"
    CLI = "cli"
    TENANT = "tenant"
    TRACING = "tracing"
    SDK = "sdk"
    CONNECTOR = "connector"  # For connector execution contexts


@dataclass
class BaseContext:
    """Base context class that all context types inherit from"""

    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_type: ContextType = ContextType.EXECUTION
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_context_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        data = asdict(self)
        data["context_type"] = self.context_type.value
        data["created_at"] = self.created_at.isoformat()
        return data

    def to_env(self) -> Dict[str, str]:
        """Convert context to environment variables"""
        env = {}
        env["GREENLANG_CONTEXT_ID"] = self.context_id
        env["GREENLANG_CONTEXT_TYPE"] = self.context_type.value

        # Add metadata as env vars
        for key, value in self.metadata.items():
            env_key = f"GREENLANG_{key.upper()}"
            env[env_key] = str(value)

        return env


@dataclass
class ExecutionContext(BaseContext):
    """Context for pipeline and workflow execution"""

    context_type: ContextType = field(default=ContextType.EXECUTION)
    pipeline_id: Optional[str] = None
    run_id: Optional[str] = None
    user: Optional[str] = None
    project: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def add_artifact(self, artifact_path: str, artifact_type: str = "output") -> None:
        """Add an artifact to the context"""
        self.artifacts.append(
            {
                "path": artifact_path,
                "type": artifact_type,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def get_step_result(self, step_name: str) -> Optional[Any]:
        """Get result from a previous step"""
        return self.step_results.get(step_name)

    def set_step_result(self, step_name: str, result: Any) -> None:
        """Set result for a step"""
        self.step_results[step_name] = result


@dataclass
class CLIContext(BaseContext):
    """Context for CLI operations"""

    context_type: ContextType = field(default=ContextType.CLI)
    verbose: bool = False
    dry_run: bool = False
    output_format: str = "json"
    config_path: Optional[Path] = None
    working_directory: Path = field(default_factory=Path.cwd)
    command: Optional[str] = None
    subcommand: Optional[str] = None

    def is_interactive(self) -> bool:
        """Check if running in interactive mode"""
        return os.isatty(0)


@dataclass
class TenantContext(BaseContext):
    """Context for multi-tenant operations"""

    context_type: ContextType = field(default=ContextType.TENANT)
    tenant_id: Optional[str] = None
    tenant_name: Optional[str] = None
    user_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    quotas: Dict[str, int] = field(default_factory=dict)
    region: Optional[str] = None
    session_id: Optional[str] = None
    session_expires: Optional[datetime] = None

    def has_permission(self, permission: str) -> bool:
        """Check if tenant has a specific permission"""
        return permission in self.permissions or "*" in self.permissions

    def check_quota(self, resource: str, amount: int = 1) -> bool:
        """Check if quota allows resource usage"""
        if resource not in self.quotas:
            return True
        return self.quotas[resource] >= amount


@dataclass
class TracingContext(BaseContext):
    """Context for distributed tracing"""

    context_type: ContextType = field(default=ContextType.TRACING)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def create_child_span(self) -> "TracingContext":
        """Create a child span context"""
        return TracingContext(
            trace_id=self.trace_id or str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            parent_span_id=self.span_id,
            baggage=self.baggage.copy(),
            parent_context_id=self.context_id,
        )


class UnifiedContextManager:
    """
    Unified context manager that coordinates between different context types
    and provides a centralized interface for context management.
    """

    def __init__(self):
        self._contexts: Dict[str, BaseContext] = {}
        self._active_contexts: Dict[ContextType, str] = {}
        self._context_stack: List[str] = []

    def create_context(self, context_type: ContextType, **kwargs) -> BaseContext:
        """Create a new context of the specified type"""
        context_classes = {
            ContextType.EXECUTION: ExecutionContext,
            ContextType.CLI: CLIContext,
            ContextType.TENANT: TenantContext,
            ContextType.TRACING: TracingContext,
            ContextType.SDK: BaseContext,
        }

        context_class = context_classes.get(context_type, BaseContext)
        context = context_class(**kwargs)

        # Store the context
        self._contexts[context.context_id] = context
        self._active_contexts[context_type] = context.context_id

        return context

    def get_context(self, context_id: str) -> Optional[BaseContext]:
        """Get a context by ID"""
        return self._contexts.get(context_id)

    def get_active_context(self, context_type: ContextType) -> Optional[BaseContext]:
        """Get the currently active context of a specific type"""
        context_id = self._active_contexts.get(context_type)
        if context_id:
            return self._contexts.get(context_id)
        return None

    def set_active_context(self, context: BaseContext) -> None:
        """Set a context as the active context for its type"""
        self._contexts[context.context_id] = context
        self._active_contexts[context.context_type] = context.context_id

    def push_context(self, context: BaseContext) -> None:
        """Push a context onto the stack"""
        self._contexts[context.context_id] = context
        self._context_stack.append(context.context_id)
        self._active_contexts[context.context_type] = context.context_id

    def pop_context(self) -> Optional[BaseContext]:
        """Pop a context from the stack"""
        if self._context_stack:
            context_id = self._context_stack.pop()
            context = self._contexts.get(context_id)

            # Update active context for this type
            if context:
                # Find the previous context of this type in the stack
                for cid in reversed(self._context_stack):
                    c = self._contexts.get(cid)
                    if c and c.context_type == context.context_type:
                        self._active_contexts[context.context_type] = cid
                        break
                else:
                    # No more contexts of this type in stack
                    self._active_contexts.pop(context.context_type, None)

            return context
        return None

    def create_child_context(
        self,
        parent_context_id: str,
        context_type: Optional[ContextType] = None,
        **kwargs,
    ) -> Optional[BaseContext]:
        """Create a child context inheriting from parent"""
        parent = self.get_context(parent_context_id)
        if not parent:
            return None

        # Use parent's type if not specified
        if context_type is None:
            context_type = parent.context_type

        # Create child context with parent reference
        kwargs["parent_context_id"] = parent_context_id

        # Inherit certain attributes from parent
        if context_type == parent.context_type:
            # Same type - inherit more attributes
            if isinstance(parent, ExecutionContext):
                kwargs.setdefault("pipeline_id", parent.pipeline_id)
                kwargs.setdefault("user", parent.user)
                kwargs.setdefault("project", parent.project)
                kwargs.setdefault("labels", parent.labels.copy())
            elif isinstance(parent, TenantContext):
                kwargs.setdefault("tenant_id", parent.tenant_id)
                kwargs.setdefault("user_id", parent.user_id)
                kwargs.setdefault("roles", parent.roles.copy())
                kwargs.setdefault("permissions", parent.permissions.copy())

        return self.create_context(context_type, **kwargs)

    def merge_contexts(self, *context_ids: str) -> BaseContext:
        """Merge multiple contexts into a unified context"""
        contexts = [
            self.get_context(cid) for cid in context_ids if self.get_context(cid)
        ]

        if not contexts:
            return self.create_context(ContextType.EXECUTION)

        # Create a new execution context as the merged result
        merged = ExecutionContext()

        for context in contexts:
            # Merge metadata
            merged.metadata.update(context.metadata)

            # Type-specific merging
            if isinstance(context, ExecutionContext):
                merged.parameters.update(context.parameters)
                merged.environment.update(context.environment)
                merged.artifacts.extend(context.artifacts)
                merged.step_results.update(context.step_results)
                merged.labels.update(context.labels)
                merged.annotations.update(context.annotations)
            elif isinstance(context, CLIContext):
                merged.metadata["cli_verbose"] = context.verbose
                merged.metadata["cli_dry_run"] = context.dry_run
            elif isinstance(context, TenantContext):
                merged.metadata["tenant_id"] = context.tenant_id
                merged.metadata["user_id"] = context.user_id
                merged.metadata["roles"] = context.roles
            elif isinstance(context, TracingContext):
                merged.metadata["trace_id"] = context.trace_id
                merged.metadata["span_id"] = context.span_id

        self._contexts[merged.context_id] = merged
        return merged

    def persist_context(self, context_id: str, file_path: Path) -> bool:
        """Persist a context to disk"""
        context = self.get_context(context_id)
        if not context:
            return False

        try:
            with open(file_path, "w") as f:
                json.dump(context.to_dict(), f, indent=2)
            return True
        except Exception:
            return False

    def restore_context(self, file_path: Path) -> Optional[BaseContext]:
        """Restore a context from disk"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            context_type = ContextType(data.get("context_type", "execution"))

            # Remove fields that will be set by the class
            data.pop("context_type", None)
            data.pop("created_at", None)

            return self.create_context(context_type, **data)
        except Exception:
            return None

    def cleanup_context(self, context_id: str) -> bool:
        """Clean up and remove a context"""
        if context_id in self._contexts:
            context = self._contexts[context_id]

            # Remove from active if it's active
            if self._active_contexts.get(context.context_type) == context_id:
                self._active_contexts.pop(context.context_type, None)

            # Remove from stack
            self._context_stack = [
                cid for cid in self._context_stack if cid != context_id
            ]

            # Remove from storage
            del self._contexts[context_id]
            return True

        return False

    def get_all_contexts(
        self, context_type: Optional[ContextType] = None
    ) -> List[BaseContext]:
        """Get all contexts, optionally filtered by type"""
        contexts = list(self._contexts.values())
        if context_type:
            contexts = [c for c in contexts if c.context_type == context_type]
        return contexts

    def clear_all(self) -> None:
        """Clear all contexts"""
        self._contexts.clear()
        self._active_contexts.clear()
        self._context_stack.clear()


# Global context manager instance
_context_manager = UnifiedContextManager()


def get_context_manager() -> UnifiedContextManager:
    """Get the global context manager instance"""
    return _context_manager


def create_execution_context(**kwargs) -> ExecutionContext:
    """Convenience function to create an execution context"""
    manager = get_context_manager()
    return manager.create_context(ContextType.EXECUTION, **kwargs)


def create_cli_context(**kwargs) -> CLIContext:
    """Convenience function to create a CLI context"""
    manager = get_context_manager()
    return manager.create_context(ContextType.CLI, **kwargs)


def create_tenant_context(**kwargs) -> TenantContext:
    """Convenience function to create a tenant context"""
    manager = get_context_manager()
    return manager.create_context(ContextType.TENANT, **kwargs)


def create_tracing_context(**kwargs) -> TracingContext:
    """Convenience function to create a tracing context"""
    manager = get_context_manager()
    return manager.create_context(ContextType.TRACING, **kwargs)


def get_current_context(
    context_type: ContextType = ContextType.EXECUTION,
) -> Optional[BaseContext]:
    """Get the current active context of a specific type"""
    manager = get_context_manager()
    return manager.get_active_context(context_type)
