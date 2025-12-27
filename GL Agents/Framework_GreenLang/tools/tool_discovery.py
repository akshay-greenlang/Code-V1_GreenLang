"""
GreenLang Framework - Tool Discovery Service

This module provides automatic tool discovery, registration, and lifecycle
management for MCP-compliant tools in the GreenLang ecosystem.

Features:
- Automatic tool registration from agent capabilities
- Capability matching for intelligent tool selection
- Tool health monitoring and circuit breaker patterns
- Tool metrics collection and aggregation
- Dynamic tool routing based on context

All operations are fully auditable with provenance tracking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Tuple
import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from collections import defaultdict
from functools import wraps

# Import from MCP protocol module
import sys
from pathlib import Path

# Add parent path for imports
_framework_path = Path(__file__).parent.parent
if str(_framework_path) not in sys.path:
    sys.path.insert(0, str(_framework_path))

from advanced.mcp_protocol import (
    MCPTool,
    MCPToolRegistry,
    ToolDefinition,
    ToolParameter,
    ToolCategory,
    SecurityLevel,
    ExecutionMode,
    ToolCallRequest,
    ToolCallResponse,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ToolHealthStatus(Enum):
    """Health status of a tool."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class CapabilityType(Enum):
    """Types of agent capabilities."""
    CALCULATION = "calculation"
    INTEGRATION = "integration"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ToolCapability:
    """Represents a capability that a tool provides."""
    name: str
    capability_type: CapabilityType
    description: str
    input_types: List[str]
    output_types: List[str]
    keywords: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    version: str = "1.0.0"

    def matches(self, query: str, min_confidence: float = 0.5) -> float:
        """
        Check if this capability matches a query.

        Returns confidence score (0-1) if match, 0 if no match.
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        # Check name match
        if self.name.lower() in query_lower:
            return 0.9 * self.confidence_score

        # Check description match
        if any(word in self.description.lower() for word in query_words):
            return 0.7 * self.confidence_score

        # Check keyword match
        keyword_matches = sum(1 for kw in self.keywords if kw.lower() in query_lower)
        if keyword_matches > 0:
            keyword_score = min(0.8, 0.3 + 0.15 * keyword_matches)
            return keyword_score * self.confidence_score

        return 0.0


@dataclass
class ToolMetrics:
    """Metrics collected for a tool."""
    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time_ms: float = 0.0
    min_execution_time_ms: float = float('inf')
    max_execution_time_ms: float = 0.0
    last_call_time: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls

    @property
    def avg_execution_time_ms(self) -> float:
        """Calculate average execution time."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_execution_time_ms / self.successful_calls

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    def record_call(
        self,
        success: bool,
        execution_time_ms: float,
        error: Optional[str] = None
    ) -> None:
        """Record a tool call."""
        self.total_calls += 1
        self.last_call_time = datetime.now(timezone.utc)

        if success:
            self.successful_calls += 1
            self.total_execution_time_ms += execution_time_ms
            self.min_execution_time_ms = min(self.min_execution_time_ms, execution_time_ms)
            self.max_execution_time_ms = max(self.max_execution_time_ms, execution_time_ms)
        else:
            self.failed_calls += 1
            self.last_error = error
            self.last_error_time = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": round(self.success_rate, 4),
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "min_execution_time_ms": round(self.min_execution_time_ms, 2) if self.min_execution_time_ms != float('inf') else None,
            "max_execution_time_ms": round(self.max_execution_time_ms, 2),
            "last_call_time": self.last_call_time.isoformat() if self.last_call_time else None,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
        }


@dataclass
class ToolHealthCheck:
    """Health check result for a tool."""
    tool_name: str
    status: ToolHealthStatus
    latency_ms: float
    last_check_time: datetime
    consecutive_failures: int = 0
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "last_check_time": self.last_check_time.isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "error_message": self.error_message,
            "details": self.details,
        }


@dataclass
class CircuitBreaker:
    """Circuit breaker for tool invocation."""
    tool_name: str
    state: CircuitState = CircuitState.CLOSED
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 30
    half_open_max_calls: int = 3

    _failure_count: int = 0
    _success_count_half_open: int = 0
    _last_failure_time: Optional[datetime] = None
    _state_changed_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._last_failure_time:
                elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
            return False
        else:  # HALF_OPEN
            return self._success_count_half_open < self.half_open_max_calls

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self._success_count_half_open += 1
            if self._success_count_half_open >= self.half_open_max_calls:
                self._transition_to(CircuitState.CLOSED)
        else:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)

        if self.state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif self._failure_count >= self.failure_threshold:
            self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self._state_changed_time = datetime.now(timezone.utc)

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count_half_open = 0

        logger.info(f"Circuit breaker {self.tool_name}: {old_state.value} -> {new_state.value}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "state_changed_time": self._state_changed_time.isoformat(),
        }


@dataclass
class ToolRegistration:
    """Complete registration information for a tool."""
    tool: MCPTool
    capabilities: List[ToolCapability]
    metrics: ToolMetrics
    health: ToolHealthCheck
    circuit_breaker: CircuitBreaker
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_agent: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def definition(self) -> ToolDefinition:
        """Get tool definition."""
        return self.tool.definition


# =============================================================================
# TOOL DISCOVERY SERVICE
# =============================================================================

class ToolDiscoveryService:
    """
    Central service for tool discovery, registration, and lifecycle management.

    Features:
    - Automatic tool registration from agent capabilities
    - Capability-based tool matching
    - Health monitoring with circuit breakers
    - Metrics collection and aggregation
    - Dynamic tool routing

    Example:
        >>> service = ToolDiscoveryService()
        >>> service.register_tool(my_tool, capabilities=[...])
        >>> matching_tools = service.find_tools_by_capability("calculate efficiency")
        >>> result = service.invoke_with_routing("calculate_efficiency", {...})
    """

    def __init__(
        self,
        health_check_interval_seconds: int = 60,
        metrics_retention_hours: int = 24,
    ):
        """
        Initialize the tool discovery service.

        Args:
            health_check_interval_seconds: Interval between health checks
            metrics_retention_hours: How long to retain detailed metrics
        """
        self._registrations: Dict[str, ToolRegistration] = {}
        self._capability_index: Dict[str, List[str]] = defaultdict(list)
        self._category_index: Dict[ToolCategory, List[str]] = defaultdict(list)
        self._tag_index: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()
        self._health_check_interval = health_check_interval_seconds
        self._metrics_retention_hours = metrics_retention_hours
        self._health_check_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start background health monitoring."""
        if self._running:
            return
        self._running = True
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()
        logger.info("Tool discovery service started")

    def stop(self) -> None:
        """Stop background health monitoring."""
        self._running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
        logger.info("Tool discovery service stopped")

    def register_tool(
        self,
        tool: MCPTool,
        capabilities: Optional[List[ToolCapability]] = None,
        source_agent: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Register a tool with the discovery service.

        Args:
            tool: The MCP tool to register
            capabilities: Tool capabilities for matching
            source_agent: Agent ID that provides this tool
            tags: Tags for categorization
        """
        with self._lock:
            name = tool.definition.name

            # Create default capabilities from definition if not provided
            if capabilities is None:
                capabilities = self._extract_capabilities(tool.definition)

            # Create registration
            registration = ToolRegistration(
                tool=tool,
                capabilities=capabilities,
                metrics=ToolMetrics(tool_name=name),
                health=ToolHealthCheck(
                    tool_name=name,
                    status=ToolHealthStatus.UNKNOWN,
                    latency_ms=0,
                    last_check_time=datetime.now(timezone.utc),
                ),
                circuit_breaker=CircuitBreaker(tool_name=name),
                source_agent=source_agent,
                tags=tags or [],
            )

            self._registrations[name] = registration

            # Update indices
            for cap in capabilities:
                self._capability_index[cap.name].append(name)
                for keyword in cap.keywords:
                    self._capability_index[keyword.lower()].append(name)

            self._category_index[tool.definition.category].append(name)

            for tag in registration.tags:
                self._tag_index[tag.lower()].append(name)

            logger.info(f"Registered tool: {name} with {len(capabilities)} capabilities")

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool from the discovery service.

        Args:
            name: Tool name to unregister

        Returns:
            True if tool was unregistered, False if not found
        """
        with self._lock:
            if name not in self._registrations:
                return False

            registration = self._registrations.pop(name)

            # Clean up indices
            for cap in registration.capabilities:
                if name in self._capability_index[cap.name]:
                    self._capability_index[cap.name].remove(name)
                for keyword in cap.keywords:
                    if name in self._capability_index[keyword.lower()]:
                        self._capability_index[keyword.lower()].remove(name)

            if name in self._category_index[registration.definition.category]:
                self._category_index[registration.definition.category].remove(name)

            for tag in registration.tags:
                if name in self._tag_index[tag.lower()]:
                    self._tag_index[tag.lower()].remove(name)

            logger.info(f"Unregistered tool: {name}")
            return True

    def get_tool(self, name: str) -> Optional[ToolRegistration]:
        """Get a tool registration by name."""
        return self._registrations.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tool definitions."""
        return [reg.definition for reg in self._registrations.values()]

    def find_tools_by_capability(
        self,
        query: str,
        min_confidence: float = 0.5,
        max_results: int = 10,
    ) -> List[Tuple[ToolRegistration, float]]:
        """
        Find tools that match a capability query.

        Args:
            query: Natural language query describing desired capability
            min_confidence: Minimum confidence score (0-1)
            max_results: Maximum number of results

        Returns:
            List of (ToolRegistration, confidence_score) tuples, sorted by confidence
        """
        results: List[Tuple[ToolRegistration, float]] = []

        for name, registration in self._registrations.items():
            best_score = 0.0
            for cap in registration.capabilities:
                score = cap.matches(query, min_confidence)
                best_score = max(best_score, score)

            # Also check tool name and description
            if registration.definition.name.lower() in query.lower():
                best_score = max(best_score, 0.8)
            elif any(word in registration.definition.description.lower()
                    for word in query.lower().split()):
                best_score = max(best_score, 0.6)

            if best_score >= min_confidence:
                results.append((registration, best_score))

        # Sort by confidence descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def find_tools_by_category(self, category: ToolCategory) -> List[ToolRegistration]:
        """Find tools by category."""
        names = self._category_index.get(category, [])
        return [self._registrations[name] for name in names if name in self._registrations]

    def find_tools_by_tag(self, tag: str) -> List[ToolRegistration]:
        """Find tools by tag."""
        names = self._tag_index.get(tag.lower(), [])
        return [self._registrations[name] for name in names if name in self._registrations]

    def find_healthy_tools(self, category: Optional[ToolCategory] = None) -> List[ToolRegistration]:
        """Find all healthy tools, optionally filtered by category."""
        results = []
        for name, registration in self._registrations.items():
            if registration.health.status == ToolHealthStatus.HEALTHY:
                if category is None or registration.definition.category == category:
                    results.append(registration)
        return results

    def invoke_with_routing(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        caller_agent_id: str = "",
        fallback_tool: Optional[str] = None,
    ) -> ToolCallResponse:
        """
        Invoke a tool with circuit breaker protection and optional fallback.

        Args:
            tool_name: Tool to invoke
            arguments: Tool arguments
            caller_agent_id: ID of calling agent
            fallback_tool: Alternative tool to use if primary fails

        Returns:
            ToolCallResponse from tool execution
        """
        registration = self._registrations.get(tool_name)
        if not registration:
            return ToolCallResponse(
                request_id="",
                tool_name=tool_name,
                success=False,
                error=f"Tool not found: {tool_name}",
            )

        # Check circuit breaker
        if not registration.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker OPEN for tool: {tool_name}")
            if fallback_tool:
                return self.invoke_with_routing(fallback_tool, arguments, caller_agent_id)
            return ToolCallResponse(
                request_id="",
                tool_name=tool_name,
                success=False,
                error="Circuit breaker is open - tool temporarily unavailable",
            )

        # Execute tool
        request = ToolCallRequest(
            tool_name=tool_name,
            arguments=arguments,
            caller_agent_id=caller_agent_id,
        )

        start_time = time.time()
        try:
            response = registration.tool.execute(request)
            execution_time = (time.time() - start_time) * 1000

            # Record metrics
            registration.metrics.record_call(
                success=response.success,
                execution_time_ms=execution_time,
                error=response.error,
            )

            # Update circuit breaker
            if response.success:
                registration.circuit_breaker.record_success()
            else:
                registration.circuit_breaker.record_failure()

            response.execution_time_ms = execution_time
            return response

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            registration.metrics.record_call(
                success=False,
                execution_time_ms=execution_time,
                error=str(e),
            )
            registration.circuit_breaker.record_failure()

            logger.error(f"Tool invocation failed: {tool_name}", exc_info=True)

            # Try fallback
            if fallback_tool:
                return self.invoke_with_routing(fallback_tool, arguments, caller_agent_id)

            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=tool_name,
                success=False,
                error=str(e),
            )

    def check_tool_health(self, tool_name: str) -> ToolHealthCheck:
        """
        Perform health check on a specific tool.

        Args:
            tool_name: Tool to check

        Returns:
            ToolHealthCheck result
        """
        registration = self._registrations.get(tool_name)
        if not registration:
            return ToolHealthCheck(
                tool_name=tool_name,
                status=ToolHealthStatus.UNKNOWN,
                latency_ms=0,
                last_check_time=datetime.now(timezone.utc),
                error_message="Tool not found",
            )

        # Perform a simple validation call
        start_time = time.time()
        try:
            # Check if tool can validate arguments (lightweight health check)
            errors = registration.tool.validate_arguments({})
            latency_ms = (time.time() - start_time) * 1000

            # Determine status based on metrics
            metrics = registration.metrics
            if metrics.error_rate > 0.5:
                status = ToolHealthStatus.UNHEALTHY
            elif metrics.error_rate > 0.2 or registration.circuit_breaker.state != CircuitState.CLOSED:
                status = ToolHealthStatus.DEGRADED
            else:
                status = ToolHealthStatus.HEALTHY

            health = ToolHealthCheck(
                tool_name=tool_name,
                status=status,
                latency_ms=latency_ms,
                last_check_time=datetime.now(timezone.utc),
                consecutive_failures=0 if status == ToolHealthStatus.HEALTHY else registration.health.consecutive_failures,
                details={
                    "success_rate": metrics.success_rate,
                    "avg_latency_ms": metrics.avg_execution_time_ms,
                    "circuit_state": registration.circuit_breaker.state.value,
                },
            )
            registration.health = health
            return health

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            health = ToolHealthCheck(
                tool_name=tool_name,
                status=ToolHealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                last_check_time=datetime.now(timezone.utc),
                consecutive_failures=registration.health.consecutive_failures + 1,
                error_message=str(e),
            )
            registration.health = health
            return health

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all registered tools."""
        return {
            name: reg.metrics.to_dict()
            for name, reg in self._registrations.items()
        }

    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all registered tools."""
        return {
            name: reg.health.to_dict()
            for name, reg in self._registrations.items()
        }

    def get_service_status(self) -> Dict[str, Any]:
        """Get overall service status."""
        total_tools = len(self._registrations)
        healthy_count = sum(
            1 for reg in self._registrations.values()
            if reg.health.status == ToolHealthStatus.HEALTHY
        )
        degraded_count = sum(
            1 for reg in self._registrations.values()
            if reg.health.status == ToolHealthStatus.DEGRADED
        )
        unhealthy_count = sum(
            1 for reg in self._registrations.values()
            if reg.health.status == ToolHealthStatus.UNHEALTHY
        )

        total_calls = sum(reg.metrics.total_calls for reg in self._registrations.values())
        total_errors = sum(reg.metrics.failed_calls for reg in self._registrations.values())

        return {
            "status": "running" if self._running else "stopped",
            "total_tools": total_tools,
            "healthy_tools": healthy_count,
            "degraded_tools": degraded_count,
            "unhealthy_tools": unhealthy_count,
            "total_calls": total_calls,
            "total_errors": total_errors,
            "overall_error_rate": total_errors / total_calls if total_calls > 0 else 0,
            "categories": {
                cat.value: len(tools)
                for cat, tools in self._category_index.items()
            },
        }

    def _extract_capabilities(self, definition: ToolDefinition) -> List[ToolCapability]:
        """Extract capabilities from tool definition."""
        # Map category to capability type
        category_to_type = {
            ToolCategory.CALCULATOR: CapabilityType.CALCULATION,
            ToolCategory.CONNECTOR: CapabilityType.INTEGRATION,
            ToolCategory.ANALYZER: CapabilityType.ANALYSIS,
            ToolCategory.OPTIMIZER: CapabilityType.OPTIMIZATION,
            ToolCategory.MONITOR: CapabilityType.MONITORING,
            ToolCategory.VALIDATOR: CapabilityType.VALIDATION,
            ToolCategory.TRANSFORMER: CapabilityType.TRANSFORMATION,
        }

        capability_type = category_to_type.get(definition.category, CapabilityType.CALCULATION)

        # Extract keywords from name and description
        keywords = []
        name_parts = re.findall(r'[A-Za-z]+', definition.name)
        keywords.extend([p.lower() for p in name_parts])

        desc_words = re.findall(r'\b[A-Za-z]{4,}\b', definition.description)
        keywords.extend([w.lower() for w in desc_words[:10]])

        # Deduplicate
        keywords = list(set(keywords))

        return [
            ToolCapability(
                name=definition.name,
                capability_type=capability_type,
                description=definition.description,
                input_types=[p.type for p in definition.parameters],
                output_types=["object"],
                keywords=keywords,
            )
        ]

    def _health_check_loop(self) -> None:
        """Background loop for health checks."""
        while self._running:
            try:
                for name in list(self._registrations.keys()):
                    if not self._running:
                        break
                    self.check_tool_health(name)
            except Exception as e:
                logger.error(f"Health check loop error: {e}", exc_info=True)

            # Wait for next interval
            for _ in range(self._health_check_interval):
                if not self._running:
                    break
                time.sleep(1)


# =============================================================================
# AUTO-REGISTRATION DECORATORS
# =============================================================================

def discoverable_tool(
    capabilities: Optional[List[ToolCapability]] = None,
    tags: Optional[List[str]] = None,
    auto_register: bool = True,
):
    """
    Decorator to make a tool automatically discoverable.

    Args:
        capabilities: Tool capabilities
        tags: Tool tags
        auto_register: Whether to auto-register with global service

    Example:
        @discoverable_tool(
            capabilities=[ToolCapability(...)],
            tags=["thermal", "calculator"]
        )
        class MyCalculatorTool(MCPTool):
            ...
    """
    def decorator(cls: Type[MCPTool]) -> Type[MCPTool]:
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if auto_register and GLOBAL_DISCOVERY_SERVICE is not None:
                GLOBAL_DISCOVERY_SERVICE.register_tool(
                    self,
                    capabilities=capabilities,
                    tags=tags,
                )

        cls.__init__ = new_init
        cls._discoverable = True
        cls._capabilities = capabilities
        cls._tags = tags
        return cls

    return decorator


# =============================================================================
# CAPABILITY BUILDERS
# =============================================================================

class CapabilityBuilder:
    """Builder for creating tool capabilities."""

    def __init__(self, name: str):
        """Initialize builder with capability name."""
        self._name = name
        self._type = CapabilityType.CALCULATION
        self._description = ""
        self._input_types: List[str] = []
        self._output_types: List[str] = []
        self._keywords: List[str] = []
        self._confidence = 1.0
        self._version = "1.0.0"

    def with_type(self, capability_type: CapabilityType) -> "CapabilityBuilder":
        """Set capability type."""
        self._type = capability_type
        return self

    def with_description(self, description: str) -> "CapabilityBuilder":
        """Set description."""
        self._description = description
        return self

    def with_inputs(self, *input_types: str) -> "CapabilityBuilder":
        """Set input types."""
        self._input_types = list(input_types)
        return self

    def with_outputs(self, *output_types: str) -> "CapabilityBuilder":
        """Set output types."""
        self._output_types = list(output_types)
        return self

    def with_keywords(self, *keywords: str) -> "CapabilityBuilder":
        """Set keywords."""
        self._keywords = list(keywords)
        return self

    def with_confidence(self, confidence: float) -> "CapabilityBuilder":
        """Set confidence score."""
        self._confidence = confidence
        return self

    def build(self) -> ToolCapability:
        """Build the capability."""
        return ToolCapability(
            name=self._name,
            capability_type=self._type,
            description=self._description,
            input_types=self._input_types,
            output_types=self._output_types,
            keywords=self._keywords,
            confidence_score=self._confidence,
            version=self._version,
        )


def capability(name: str) -> CapabilityBuilder:
    """Create a capability builder."""
    return CapabilityBuilder(name)


# =============================================================================
# GLOBAL SERVICE INSTANCE
# =============================================================================

GLOBAL_DISCOVERY_SERVICE: Optional[ToolDiscoveryService] = None


def get_discovery_service() -> ToolDiscoveryService:
    """Get the global discovery service, creating if needed."""
    global GLOBAL_DISCOVERY_SERVICE
    if GLOBAL_DISCOVERY_SERVICE is None:
        GLOBAL_DISCOVERY_SERVICE = ToolDiscoveryService()
    return GLOBAL_DISCOVERY_SERVICE


def register_tool_globally(
    tool: MCPTool,
    capabilities: Optional[List[ToolCapability]] = None,
    tags: Optional[List[str]] = None,
) -> None:
    """Register a tool with the global discovery service."""
    service = get_discovery_service()
    service.register_tool(tool, capabilities=capabilities, tags=tags)


# Export list
__all__ = [
    # Enums
    "ToolHealthStatus",
    "CircuitState",
    "CapabilityType",
    # Data models
    "ToolCapability",
    "ToolMetrics",
    "ToolHealthCheck",
    "CircuitBreaker",
    "ToolRegistration",
    # Main service
    "ToolDiscoveryService",
    # Decorators and builders
    "discoverable_tool",
    "CapabilityBuilder",
    "capability",
    # Global functions
    "get_discovery_service",
    "register_tool_globally",
    "GLOBAL_DISCOVERY_SERVICE",
]
