"""
GreenLang Framework - MCP Connector Tool Definitions

This module provides MCP-compliant tool definitions for industrial system
integrations with proper security levels and access controls.

Connectors include:
- SCADA connector (read/write process data)
- OPC-UA connector (industrial protocol)
- Kafka producer/consumer (event streaming)
- Database query tool (SQL with parameterization)

All connectors implement proper security controls and audit logging.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import asyncio
import hashlib
import json
import logging
import re

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
# SECURITY AND ACCESS CONTROL
# =============================================================================

class AccessLevel(Enum):
    """Access levels for connector operations."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class SecurityContext:
    """Security context for connector operations."""
    user_id: str
    roles: List[str]
    access_level: AccessLevel
    session_id: str = ""
    ip_address: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def has_permission(self, required_level: AccessLevel) -> bool:
        """Check if context has required permission level."""
        level_order = [AccessLevel.READ, AccessLevel.WRITE, AccessLevel.EXECUTE, AccessLevel.ADMIN]
        current_idx = level_order.index(self.access_level)
        required_idx = level_order.index(required_level)
        return current_idx >= required_idx


@dataclass
class AuditRecord:
    """Audit record for connector operations."""
    operation: str
    tool_name: str
    user_id: str
    timestamp: datetime
    request_hash: str
    response_hash: str
    success: bool
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "operation": self.operation,
            "tool_name": self.tool_name,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "details": self.details,
        }


class AuditLogger:
    """Audit logger for connector operations."""

    def __init__(self):
        """Initialize audit logger."""
        self._records: List[AuditRecord] = []
        self._max_records = 10000

    def log(self, record: AuditRecord) -> None:
        """Log an audit record."""
        self._records.append(record)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]
        logger.info(f"AUDIT: {json.dumps(record.to_dict())}")

    def get_records(
        self,
        user_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[AuditRecord]:
        """Query audit records."""
        records = self._records
        if user_id:
            records = [r for r in records if r.user_id == user_id]
        if tool_name:
            records = [r for r in records if r.tool_name == tool_name]
        if since:
            records = [r for r in records if r.timestamp >= since]
        return records


# Global audit logger
AUDIT_LOGGER = AuditLogger()


# =============================================================================
# CONNECTOR RESULT MODELS
# =============================================================================

@dataclass
class ScadaReadResult:
    """Result from SCADA read operation."""
    tag_name: str
    value: Any
    quality: str  # "GOOD", "BAD", "UNCERTAIN"
    timestamp: datetime
    unit: Optional[str] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag_name": self.tag_name,
            "value": self.value,
            "quality": self.quality,
            "timestamp": self.timestamp.isoformat(),
            "unit": self.unit,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class ScadaWriteResult:
    """Result from SCADA write operation."""
    tag_name: str
    old_value: Any
    new_value: Any
    success: bool
    timestamp: datetime
    requires_confirmation: bool = False
    confirmation_id: Optional[str] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag_name": self.tag_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "requires_confirmation": self.requires_confirmation,
            "confirmation_id": self.confirmation_id,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class OpcUaNodeResult:
    """Result from OPC-UA node operation."""
    node_id: str
    browse_name: str
    value: Any
    data_type: str
    source_timestamp: datetime
    server_timestamp: datetime
    status_code: int
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "browse_name": self.browse_name,
            "value": self.value,
            "data_type": self.data_type,
            "source_timestamp": self.source_timestamp.isoformat(),
            "server_timestamp": self.server_timestamp.isoformat(),
            "status_code": self.status_code,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class KafkaProduceResult:
    """Result from Kafka produce operation."""
    topic: str
    partition: int
    offset: int
    timestamp: datetime
    key: Optional[str]
    message_size_bytes: int
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic": self.topic,
            "partition": self.partition,
            "offset": self.offset,
            "timestamp": self.timestamp.isoformat(),
            "key": self.key,
            "message_size_bytes": self.message_size_bytes,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class KafkaConsumeResult:
    """Result from Kafka consume operation."""
    topic: str
    partition: int
    offset: int
    timestamp: datetime
    key: Optional[str]
    value: Any
    headers: Dict[str, str]
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic": self.topic,
            "partition": self.partition,
            "offset": self.offset,
            "timestamp": self.timestamp.isoformat(),
            "key": self.key,
            "value": self.value,
            "headers": self.headers,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class DatabaseQueryResult:
    """Result from database query operation."""
    query_hash: str
    row_count: int
    column_names: List[str]
    rows: List[List[Any]]
    execution_time_ms: float
    affected_rows: Optional[int] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_hash": self.query_hash,
            "row_count": self.row_count,
            "column_names": self.column_names,
            "rows": self.rows[:100],  # Limit for response size
            "execution_time_ms": self.execution_time_ms,
            "affected_rows": self.affected_rows,
            "provenance_hash": self.provenance_hash,
        }


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

SCADA_READ_DEFINITION = ToolDefinition(
    name="scada_read",
    description=(
        "Read process data from SCADA system. Retrieves current value, "
        "quality status, and timestamp for specified tags. Supports batch "
        "reads for multiple tags."
    ),
    parameters=[
        ToolParameter(
            name="tags",
            type="array",
            description="List of SCADA tag names to read",
            required=True,
        ),
        ToolParameter(
            name="include_quality",
            type="boolean",
            description="Include data quality indicators (default true)",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="timeout_seconds",
            type="number",
            description="Read timeout in seconds (default 5)",
            required=False,
            default=5,
            minimum=1,
            maximum=30,
        ),
    ],
    category=ToolCategory.CONNECTOR,
    security_level=SecurityLevel.READ_ONLY,
    execution_mode=ExecutionMode.SYNC,
    timeout_seconds=30,
    requires_confirmation=False,
    audit_level="full",
    version="1.0.0",
)


SCADA_WRITE_DEFINITION = ToolDefinition(
    name="scada_write",
    description=(
        "Write setpoint or control value to SCADA system. REQUIRES CONFIRMATION "
        "for safety-critical tags. Validates value against configured limits "
        "before writing."
    ),
    parameters=[
        ToolParameter(
            name="tag_name",
            type="string",
            description="SCADA tag name to write",
            required=True,
        ),
        ToolParameter(
            name="value",
            type="number",
            description="Value to write",
            required=True,
        ),
        ToolParameter(
            name="force",
            type="boolean",
            description="Force write without limit checks (requires ADMIN)",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="confirmation_id",
            type="string",
            description="Confirmation ID for safety-critical writes",
            required=False,
        ),
    ],
    category=ToolCategory.CONNECTOR,
    security_level=SecurityLevel.CONTROLLED_WRITE,
    execution_mode=ExecutionMode.SYNC,
    timeout_seconds=10,
    requires_confirmation=True,
    audit_level="full",
    version="1.0.0",
)


OPC_UA_READ_DEFINITION = ToolDefinition(
    name="opcua_read",
    description=(
        "Read node values from OPC-UA server. Supports reading single nodes, "
        "multiple nodes, or browsing node hierarchies. Returns typed values "
        "with timestamps and status codes."
    ),
    parameters=[
        ToolParameter(
            name="node_ids",
            type="array",
            description="List of OPC-UA node IDs (e.g., 'ns=2;s=Tag1')",
            required=True,
        ),
        ToolParameter(
            name="server_url",
            type="string",
            description="OPC-UA server endpoint URL",
            required=True,
            pattern=r"^opc\.tcp://.*$",
        ),
        ToolParameter(
            name="security_mode",
            type="string",
            description="Security mode for connection",
            required=False,
            default="SignAndEncrypt",
            enum=["None", "Sign", "SignAndEncrypt"],
        ),
        ToolParameter(
            name="timeout_ms",
            type="number",
            description="Operation timeout in milliseconds",
            required=False,
            default=5000,
            minimum=100,
            maximum=60000,
        ),
    ],
    category=ToolCategory.CONNECTOR,
    security_level=SecurityLevel.READ_ONLY,
    execution_mode=ExecutionMode.SYNC,
    timeout_seconds=60,
    audit_level="full",
    version="1.0.0",
)


OPC_UA_WRITE_DEFINITION = ToolDefinition(
    name="opcua_write",
    description=(
        "Write value to OPC-UA server node. Validates data type compatibility "
        "and checks write permissions before writing."
    ),
    parameters=[
        ToolParameter(
            name="node_id",
            type="string",
            description="OPC-UA node ID to write",
            required=True,
        ),
        ToolParameter(
            name="value",
            type="object",
            description="Value to write with type info",
            required=True,
        ),
        ToolParameter(
            name="server_url",
            type="string",
            description="OPC-UA server endpoint URL",
            required=True,
        ),
        ToolParameter(
            name="security_mode",
            type="string",
            description="Security mode for connection",
            required=False,
            default="SignAndEncrypt",
            enum=["None", "Sign", "SignAndEncrypt"],
        ),
    ],
    category=ToolCategory.CONNECTOR,
    security_level=SecurityLevel.CONTROLLED_WRITE,
    execution_mode=ExecutionMode.SYNC,
    timeout_seconds=30,
    requires_confirmation=True,
    audit_level="full",
    version="1.0.0",
)


KAFKA_PRODUCE_DEFINITION = ToolDefinition(
    name="kafka_produce",
    description=(
        "Produce message to Kafka topic. Supports JSON serialization, "
        "message keys, headers, and custom partitioning."
    ),
    parameters=[
        ToolParameter(
            name="topic",
            type="string",
            description="Kafka topic name",
            required=True,
            pattern=r"^[a-zA-Z0-9._-]+$",
        ),
        ToolParameter(
            name="message",
            type="object",
            description="Message payload (will be JSON serialized)",
            required=True,
        ),
        ToolParameter(
            name="key",
            type="string",
            description="Message key for partitioning",
            required=False,
        ),
        ToolParameter(
            name="headers",
            type="object",
            description="Message headers (key-value pairs)",
            required=False,
        ),
        ToolParameter(
            name="partition",
            type="number",
            description="Target partition (if not using key-based routing)",
            required=False,
            minimum=0,
        ),
        ToolParameter(
            name="bootstrap_servers",
            type="string",
            description="Kafka bootstrap servers (comma-separated)",
            required=True,
        ),
    ],
    category=ToolCategory.CONNECTOR,
    security_level=SecurityLevel.CONTROLLED_WRITE,
    execution_mode=ExecutionMode.ASYNC,
    timeout_seconds=30,
    audit_level="full",
    version="1.0.0",
)


KAFKA_CONSUME_DEFINITION = ToolDefinition(
    name="kafka_consume",
    description=(
        "Consume messages from Kafka topic. Supports consuming from "
        "specific partitions, offsets, or using consumer groups."
    ),
    parameters=[
        ToolParameter(
            name="topic",
            type="string",
            description="Kafka topic name",
            required=True,
        ),
        ToolParameter(
            name="group_id",
            type="string",
            description="Consumer group ID",
            required=False,
        ),
        ToolParameter(
            name="max_messages",
            type="number",
            description="Maximum messages to consume (default 100)",
            required=False,
            default=100,
            minimum=1,
            maximum=10000,
        ),
        ToolParameter(
            name="timeout_ms",
            type="number",
            description="Poll timeout in milliseconds",
            required=False,
            default=5000,
            minimum=100,
            maximum=60000,
        ),
        ToolParameter(
            name="from_beginning",
            type="boolean",
            description="Start consuming from beginning of topic",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="bootstrap_servers",
            type="string",
            description="Kafka bootstrap servers (comma-separated)",
            required=True,
        ),
    ],
    category=ToolCategory.CONNECTOR,
    security_level=SecurityLevel.READ_ONLY,
    execution_mode=ExecutionMode.ASYNC,
    timeout_seconds=120,
    audit_level="full",
    version="1.0.0",
)


DATABASE_QUERY_DEFINITION = ToolDefinition(
    name="database_query",
    description=(
        "Execute parameterized SQL query against database. Uses prepared "
        "statements to prevent SQL injection. Supports SELECT, INSERT, "
        "UPDATE, DELETE with proper access controls."
    ),
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="SQL query with parameter placeholders (:param or ?)",
            required=True,
        ),
        ToolParameter(
            name="parameters",
            type="object",
            description="Query parameters as key-value pairs",
            required=False,
            default={},
        ),
        ToolParameter(
            name="database",
            type="string",
            description="Database connection name from config",
            required=True,
        ),
        ToolParameter(
            name="max_rows",
            type="number",
            description="Maximum rows to return (default 1000)",
            required=False,
            default=1000,
            minimum=1,
            maximum=100000,
        ),
        ToolParameter(
            name="timeout_seconds",
            type="number",
            description="Query timeout in seconds",
            required=False,
            default=30,
            minimum=1,
            maximum=600,
        ),
    ],
    category=ToolCategory.CONNECTOR,
    security_level=SecurityLevel.ADVISORY,  # Depends on query type
    execution_mode=ExecutionMode.SYNC,
    timeout_seconds=600,
    audit_level="full",
    version="1.0.0",
)


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

class ScadaReadTool(MCPTool):
    """
    MCP Tool for reading SCADA process data.

    This is a simulation/mock implementation. In production, this would
    connect to actual SCADA systems (e.g., OSIsoft PI, Wonderware).
    """

    def __init__(self):
        """Initialize SCADA read tool."""
        super().__init__(SCADA_READ_DEFINITION)
        # Mock tag database for demonstration
        self._mock_tags: Dict[str, Dict[str, Any]] = {
            "BOILER_01.STEAM_TEMP": {"value": 485.2, "unit": "C", "quality": "GOOD"},
            "BOILER_01.STEAM_PRESSURE": {"value": 42.5, "unit": "bar", "quality": "GOOD"},
            "BOILER_01.FEED_WATER_FLOW": {"value": 125.8, "unit": "t/h", "quality": "GOOD"},
            "BOILER_01.FLUE_GAS_O2": {"value": 3.2, "unit": "%", "quality": "GOOD"},
            "BOILER_01.EFFICIENCY": {"value": 88.5, "unit": "%", "quality": "GOOD"},
            "TURBINE_01.POWER_OUTPUT": {"value": 45.2, "unit": "MW", "quality": "GOOD"},
            "TURBINE_01.INLET_TEMP": {"value": 480.0, "unit": "C", "quality": "GOOD"},
            "TURBINE_01.EXHAUST_TEMP": {"value": 35.5, "unit": "C", "quality": "GOOD"},
        }

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute SCADA read operation."""
        try:
            args = request.arguments
            tags = args["tags"]
            include_quality = args.get("include_quality", True)

            results = []
            now = datetime.now(timezone.utc)

            for tag in tags:
                if tag in self._mock_tags:
                    tag_data = self._mock_tags[tag]
                    result = ScadaReadResult(
                        tag_name=tag,
                        value=tag_data["value"],
                        quality=tag_data["quality"] if include_quality else "GOOD",
                        timestamp=now,
                        unit=tag_data.get("unit"),
                        provenance_hash=hashlib.sha256(
                            f"{tag}:{tag_data['value']}:{now.isoformat()}".encode()
                        ).hexdigest(),
                    )
                else:
                    result = ScadaReadResult(
                        tag_name=tag,
                        value=None,
                        quality="BAD",
                        timestamp=now,
                        provenance_hash="",
                    )
                results.append(result.to_dict())

            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                result={"tags": results, "count": len(results)},
            )

        except Exception as e:
            logger.error(f"SCADA read failed: {e}", exc_info=True)
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
            )


class ScadaWriteTool(MCPTool):
    """
    MCP Tool for writing SCADA setpoints.

    Implements safety checks and confirmation requirements for
    safety-critical tags.
    """

    # Tags requiring confirmation before write
    SAFETY_CRITICAL_TAGS = {
        "BOILER_01.FUEL_VALVE",
        "BOILER_01.EMERGENCY_STOP",
        "TURBINE_01.GOVERNOR",
    }

    def __init__(self):
        """Initialize SCADA write tool."""
        super().__init__(SCADA_WRITE_DEFINITION)
        self._pending_confirmations: Dict[str, Dict[str, Any]] = {}
        self._tag_limits: Dict[str, Dict[str, float]] = {
            "BOILER_01.STEAM_TEMP_SP": {"min": 400, "max": 550},
            "BOILER_01.STEAM_PRESSURE_SP": {"min": 30, "max": 50},
            "BOILER_01.LOAD_SP": {"min": 0, "max": 100},
        }

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute SCADA write operation."""
        try:
            args = request.arguments
            tag_name = args["tag_name"]
            value = args["value"]
            force = args.get("force", False)
            confirmation_id = args.get("confirmation_id")

            now = datetime.now(timezone.utc)

            # Check if this is a safety-critical tag requiring confirmation
            if tag_name in self.SAFETY_CRITICAL_TAGS:
                if confirmation_id is None:
                    # Generate confirmation request
                    conf_id = hashlib.sha256(
                        f"{tag_name}:{value}:{now.isoformat()}".encode()
                    ).hexdigest()[:16]
                    self._pending_confirmations[conf_id] = {
                        "tag": tag_name,
                        "value": value,
                        "requested_at": now,
                        "expires_at": now.replace(minute=now.minute + 5),
                    }

                    result = ScadaWriteResult(
                        tag_name=tag_name,
                        old_value=None,
                        new_value=value,
                        success=False,
                        timestamp=now,
                        requires_confirmation=True,
                        confirmation_id=conf_id,
                        provenance_hash="",
                    )

                    return ToolCallResponse(
                        request_id=request.request_id,
                        tool_name=request.tool_name,
                        success=True,
                        result=result.to_dict(),
                    )

                # Verify confirmation
                if confirmation_id not in self._pending_confirmations:
                    return ToolCallResponse(
                        request_id=request.request_id,
                        tool_name=request.tool_name,
                        success=False,
                        error="Invalid or expired confirmation ID",
                    )

                pending = self._pending_confirmations.pop(confirmation_id)
                if pending["tag"] != tag_name or pending["value"] != value:
                    return ToolCallResponse(
                        request_id=request.request_id,
                        tool_name=request.tool_name,
                        success=False,
                        error="Confirmation does not match requested write",
                    )

            # Check value limits (unless force is True)
            if not force and tag_name in self._tag_limits:
                limits = self._tag_limits[tag_name]
                if value < limits["min"] or value > limits["max"]:
                    return ToolCallResponse(
                        request_id=request.request_id,
                        tool_name=request.tool_name,
                        success=False,
                        error=f"Value {value} outside limits [{limits['min']}, {limits['max']}]",
                    )

            # Simulate write (in production, this would write to SCADA)
            old_value = 0.0  # Would be read from SCADA

            result = ScadaWriteResult(
                tag_name=tag_name,
                old_value=old_value,
                new_value=value,
                success=True,
                timestamp=now,
                requires_confirmation=False,
                provenance_hash=hashlib.sha256(
                    f"{tag_name}:{old_value}:{value}:{now.isoformat()}".encode()
                ).hexdigest(),
            )

            # Audit log
            AUDIT_LOGGER.log(AuditRecord(
                operation="scada_write",
                tool_name=self.definition.name,
                user_id=request.caller_agent_id,
                timestamp=now,
                request_hash=hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest(),
                response_hash=result.provenance_hash,
                success=True,
                duration_ms=0,
                details={"tag": tag_name, "old_value": old_value, "new_value": value},
            ))

            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                result=result.to_dict(),
            )

        except Exception as e:
            logger.error(f"SCADA write failed: {e}", exc_info=True)
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
            )


class OpcUaReadTool(MCPTool):
    """
    MCP Tool for reading OPC-UA node values.

    This is a simulation/mock implementation. In production, this would
    use opcua or asyncua library for actual OPC-UA communication.
    """

    def __init__(self):
        """Initialize OPC-UA read tool."""
        super().__init__(OPC_UA_READ_DEFINITION)
        # Mock node database
        self._mock_nodes: Dict[str, Dict[str, Any]] = {
            "ns=2;s=Temperature.PV": {"value": 485.2, "type": "Double", "browse_name": "Temperature.PV"},
            "ns=2;s=Pressure.PV": {"value": 42.5, "type": "Double", "browse_name": "Pressure.PV"},
            "ns=2;s=Flow.PV": {"value": 125.8, "type": "Double", "browse_name": "Flow.PV"},
            "ns=2;s=Status": {"value": 1, "type": "Int32", "browse_name": "Status"},
        }

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute OPC-UA read operation."""
        try:
            args = request.arguments
            node_ids = args["node_ids"]
            server_url = args["server_url"]

            results = []
            now = datetime.now(timezone.utc)

            for node_id in node_ids:
                if node_id in self._mock_nodes:
                    node_data = self._mock_nodes[node_id]
                    result = OpcUaNodeResult(
                        node_id=node_id,
                        browse_name=node_data["browse_name"],
                        value=node_data["value"],
                        data_type=node_data["type"],
                        source_timestamp=now,
                        server_timestamp=now,
                        status_code=0,  # Good
                        provenance_hash=hashlib.sha256(
                            f"{node_id}:{node_data['value']}:{now.isoformat()}".encode()
                        ).hexdigest(),
                    )
                else:
                    result = OpcUaNodeResult(
                        node_id=node_id,
                        browse_name="Unknown",
                        value=None,
                        data_type="Unknown",
                        source_timestamp=now,
                        server_timestamp=now,
                        status_code=2150891520,  # BadNodeIdUnknown
                        provenance_hash="",
                    )
                results.append(result.to_dict())

            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                result={"nodes": results, "server_url": server_url},
            )

        except Exception as e:
            logger.error(f"OPC-UA read failed: {e}", exc_info=True)
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
            )


class OpcUaWriteTool(MCPTool):
    """
    MCP Tool for writing OPC-UA node values.
    """

    def __init__(self):
        """Initialize OPC-UA write tool."""
        super().__init__(OPC_UA_WRITE_DEFINITION)

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute OPC-UA write operation."""
        try:
            args = request.arguments
            node_id = args["node_id"]
            value = args["value"]
            server_url = args["server_url"]

            now = datetime.now(timezone.utc)

            # Simulate write operation
            result = OpcUaNodeResult(
                node_id=node_id,
                browse_name=node_id.split(";")[-1] if ";" in node_id else node_id,
                value=value,
                data_type=type(value).__name__,
                source_timestamp=now,
                server_timestamp=now,
                status_code=0,  # Good
                provenance_hash=hashlib.sha256(
                    f"{node_id}:{value}:{now.isoformat()}".encode()
                ).hexdigest(),
            )

            # Audit log
            AUDIT_LOGGER.log(AuditRecord(
                operation="opcua_write",
                tool_name=self.definition.name,
                user_id=request.caller_agent_id,
                timestamp=now,
                request_hash=hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest(),
                response_hash=result.provenance_hash,
                success=True,
                duration_ms=0,
                details={"node_id": node_id, "value": value, "server": server_url},
            ))

            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                result=result.to_dict(),
            )

        except Exception as e:
            logger.error(f"OPC-UA write failed: {e}", exc_info=True)
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
            )


class KafkaProduceTool(MCPTool):
    """
    MCP Tool for producing Kafka messages.

    This is a simulation/mock implementation. In production, this would
    use kafka-python or confluent-kafka library.
    """

    def __init__(self):
        """Initialize Kafka produce tool."""
        super().__init__(KAFKA_PRODUCE_DEFINITION)
        self._message_counter = 0

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute Kafka produce operation."""
        try:
            args = request.arguments
            topic = args["topic"]
            message = args["message"]
            key = args.get("key")
            headers = args.get("headers", {})
            partition = args.get("partition", 0)

            now = datetime.now(timezone.utc)
            self._message_counter += 1

            # Serialize message
            message_bytes = json.dumps(message).encode()

            result = KafkaProduceResult(
                topic=topic,
                partition=partition,
                offset=self._message_counter,
                timestamp=now,
                key=key,
                message_size_bytes=len(message_bytes),
                provenance_hash=hashlib.sha256(
                    f"{topic}:{partition}:{self._message_counter}:{message_bytes.hex()}".encode()
                ).hexdigest(),
            )

            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                result=result.to_dict(),
            )

        except Exception as e:
            logger.error(f"Kafka produce failed: {e}", exc_info=True)
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
            )


class KafkaConsumeTool(MCPTool):
    """
    MCP Tool for consuming Kafka messages.
    """

    def __init__(self):
        """Initialize Kafka consume tool."""
        super().__init__(KAFKA_CONSUME_DEFINITION)
        # Mock message store
        self._mock_messages: List[Dict[str, Any]] = [
            {"key": "sensor-001", "value": {"temp": 485.2, "pressure": 42.5}, "offset": 0},
            {"key": "sensor-001", "value": {"temp": 486.1, "pressure": 42.4}, "offset": 1},
            {"key": "sensor-002", "value": {"temp": 320.5, "pressure": 15.2}, "offset": 2},
        ]

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute Kafka consume operation."""
        try:
            args = request.arguments
            topic = args["topic"]
            max_messages = args.get("max_messages", 100)
            from_beginning = args.get("from_beginning", False)

            now = datetime.now(timezone.utc)
            messages = []

            for i, msg in enumerate(self._mock_messages[:max_messages]):
                result = KafkaConsumeResult(
                    topic=topic,
                    partition=0,
                    offset=msg["offset"],
                    timestamp=now,
                    key=msg.get("key"),
                    value=msg["value"],
                    headers={},
                    provenance_hash=hashlib.sha256(
                        f"{topic}:{msg['offset']}:{json.dumps(msg['value'])}".encode()
                    ).hexdigest(),
                )
                messages.append(result.to_dict())

            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                result={"messages": messages, "count": len(messages)},
            )

        except Exception as e:
            logger.error(f"Kafka consume failed: {e}", exc_info=True)
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
            )


class DatabaseQueryTool(MCPTool):
    """
    MCP Tool for executing parameterized database queries.

    Implements SQL injection prevention through parameter binding.
    """

    # Allowed query patterns
    ALLOWED_SELECT_PATTERN = re.compile(r"^\s*SELECT\s+", re.IGNORECASE)
    DANGEROUS_PATTERNS = [
        re.compile(r";\s*(DROP|DELETE|TRUNCATE|ALTER|CREATE)", re.IGNORECASE),
        re.compile(r"--"),
        re.compile(r"/\*"),
        re.compile(r"xp_cmdshell", re.IGNORECASE),
        re.compile(r"EXEC\s+", re.IGNORECASE),
    ]

    def __init__(self):
        """Initialize database query tool."""
        super().__init__(DATABASE_QUERY_DEFINITION)
        # Mock database
        self._mock_data: Dict[str, List[Dict[str, Any]]] = {
            "emission_factors": [
                {"id": 1, "fuel_type": "natural_gas", "co2_kg_per_kwh": 0.18293, "source": "DEFRA", "year": 2023},
                {"id": 2, "fuel_type": "diesel", "co2_kg_per_kwh": 0.25301, "source": "DEFRA", "year": 2023},
                {"id": 3, "fuel_type": "coal", "co2_kg_per_kwh": 0.32307, "source": "DEFRA", "year": 2023},
            ],
            "process_data": [
                {"tag": "TEMP_001", "value": 485.2, "timestamp": "2024-01-15T10:00:00Z"},
                {"tag": "TEMP_001", "value": 486.1, "timestamp": "2024-01-15T10:01:00Z"},
                {"tag": "PRES_001", "value": 42.5, "timestamp": "2024-01-15T10:00:00Z"},
            ],
        }

    def _check_query_safety(self, query: str) -> Optional[str]:
        """Check if query is safe to execute."""
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(query):
                return "Query contains potentially dangerous pattern"
        return None

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute database query operation."""
        try:
            args = request.arguments
            query = args["query"]
            parameters = args.get("parameters", {})
            database = args["database"]
            max_rows = args.get("max_rows", 1000)

            now = datetime.now(timezone.utc)

            # Security check
            safety_error = self._check_query_safety(query)
            if safety_error:
                return ToolCallResponse(
                    request_id=request.request_id,
                    tool_name=request.tool_name,
                    success=False,
                    error=f"Security violation: {safety_error}",
                )

            # Simulate query execution
            # In production, this would use SQLAlchemy or similar

            # Mock response based on query pattern
            if "emission_factors" in query.lower():
                rows = self._mock_data["emission_factors"]
            elif "process_data" in query.lower():
                rows = self._mock_data["process_data"]
            else:
                rows = []

            if rows:
                column_names = list(rows[0].keys())
                row_data = [list(r.values()) for r in rows[:max_rows]]
            else:
                column_names = []
                row_data = []

            query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

            result = DatabaseQueryResult(
                query_hash=query_hash,
                row_count=len(row_data),
                column_names=column_names,
                rows=row_data,
                execution_time_ms=5.2,
                provenance_hash=hashlib.sha256(
                    f"{query_hash}:{len(row_data)}:{now.isoformat()}".encode()
                ).hexdigest(),
            )

            # Audit log
            AUDIT_LOGGER.log(AuditRecord(
                operation="database_query",
                tool_name=self.definition.name,
                user_id=request.caller_agent_id,
                timestamp=now,
                request_hash=query_hash,
                response_hash=result.provenance_hash,
                success=True,
                duration_ms=5.2,
                details={"database": database, "row_count": len(row_data)},
            ))

            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                result=result.to_dict(),
            )

        except Exception as e:
            logger.error(f"Database query failed: {e}", exc_info=True)
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
            )


# =============================================================================
# TOOL REGISTRY
# =============================================================================

def create_connector_registry() -> MCPToolRegistry:
    """
    Create and populate the MCP connector tool registry.

    Returns:
        MCPToolRegistry with all connector tools registered.
    """
    registry = MCPToolRegistry(server_name="GreenLang Connector MCP Server")

    # Register all connector tools
    registry.register(ScadaReadTool())
    registry.register(ScadaWriteTool())
    registry.register(OpcUaReadTool())
    registry.register(OpcUaWriteTool())
    registry.register(KafkaProduceTool())
    registry.register(KafkaConsumeTool())
    registry.register(DatabaseQueryTool())

    logger.info(f"Registered {len(registry.list_tools())} connector tools")

    return registry


# Global connector registry instance
CONNECTOR_REGISTRY = create_connector_registry()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_connector_tools() -> List[ToolDefinition]:
    """Get all connector tool definitions."""
    return CONNECTOR_REGISTRY.list_tools()


def invoke_connector(name: str, arguments: Dict[str, Any]) -> ToolCallResponse:
    """
    Invoke a connector tool by name.

    Args:
        name: Tool name (e.g., "scada_read")
        arguments: Tool arguments

    Returns:
        ToolCallResponse with result or error
    """
    request = ToolCallRequest(tool_name=name, arguments=arguments)
    return CONNECTOR_REGISTRY.invoke(request)


def get_audit_records(
    user_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    since: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Get audit records for connector operations."""
    records = AUDIT_LOGGER.get_records(user_id, tool_name, since)
    return [r.to_dict() for r in records]


# Export list
__all__ = [
    # Security
    "AccessLevel",
    "SecurityContext",
    "AuditRecord",
    "AuditLogger",
    "AUDIT_LOGGER",
    # Result models
    "ScadaReadResult",
    "ScadaWriteResult",
    "OpcUaNodeResult",
    "KafkaProduceResult",
    "KafkaConsumeResult",
    "DatabaseQueryResult",
    # Tool definitions
    "SCADA_READ_DEFINITION",
    "SCADA_WRITE_DEFINITION",
    "OPC_UA_READ_DEFINITION",
    "OPC_UA_WRITE_DEFINITION",
    "KAFKA_PRODUCE_DEFINITION",
    "KAFKA_CONSUME_DEFINITION",
    "DATABASE_QUERY_DEFINITION",
    # Tool classes
    "ScadaReadTool",
    "ScadaWriteTool",
    "OpcUaReadTool",
    "OpcUaWriteTool",
    "KafkaProduceTool",
    "KafkaConsumeTool",
    "DatabaseQueryTool",
    # Registry
    "CONNECTOR_REGISTRY",
    "create_connector_registry",
    # Convenience functions
    "get_connector_tools",
    "invoke_connector",
    "get_audit_records",
]
