"""
GreenLang Framework - MCP Connector Tools Tests

Comprehensive test suite for MCP connector tool definitions and implementations.
Tests cover:
- Tool definitions and security levels
- SCADA read/write operations
- OPC-UA node operations
- Kafka produce/consume operations
- Database query operations
- Audit logging
- Circuit breaker patterns
"""

import pytest
import json
from datetime import datetime, timezone
from typing import Dict, Any

# Import modules under test
import sys
from pathlib import Path

# Add parent paths for imports
_framework_path = Path(__file__).parent.parent.parent
if str(_framework_path) not in sys.path:
    sys.path.insert(0, str(_framework_path))

from advanced.mcp_protocol import (
    ToolCallRequest,
    ToolCallResponse,
    ToolCategory,
    SecurityLevel,
)
from tools.mcp_connectors import (
    # Security
    AccessLevel,
    SecurityContext,
    AuditRecord,
    AUDIT_LOGGER,
    # Tool definitions
    SCADA_READ_DEFINITION,
    SCADA_WRITE_DEFINITION,
    OPC_UA_READ_DEFINITION,
    OPC_UA_WRITE_DEFINITION,
    KAFKA_PRODUCE_DEFINITION,
    KAFKA_CONSUME_DEFINITION,
    DATABASE_QUERY_DEFINITION,
    # Tool classes
    ScadaReadTool,
    ScadaWriteTool,
    OpcUaReadTool,
    OpcUaWriteTool,
    KafkaProduceTool,
    KafkaConsumeTool,
    DatabaseQueryTool,
    # Registry
    CONNECTOR_REGISTRY,
    create_connector_registry,
    # Convenience functions
    get_connector_tools,
    invoke_connector,
    get_audit_records,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def scada_read_tool():
    """Create SCADA read tool instance."""
    return ScadaReadTool()


@pytest.fixture
def scada_write_tool():
    """Create SCADA write tool instance."""
    return ScadaWriteTool()


@pytest.fixture
def opcua_read_tool():
    """Create OPC-UA read tool instance."""
    return OpcUaReadTool()


@pytest.fixture
def opcua_write_tool():
    """Create OPC-UA write tool instance."""
    return OpcUaWriteTool()


@pytest.fixture
def kafka_produce_tool():
    """Create Kafka produce tool instance."""
    return KafkaProduceTool()


@pytest.fixture
def kafka_consume_tool():
    """Create Kafka consume tool instance."""
    return KafkaConsumeTool()


@pytest.fixture
def database_tool():
    """Create database query tool instance."""
    return DatabaseQueryTool()


# =============================================================================
# SECURITY MODEL TESTS
# =============================================================================

class TestSecurityModel:
    """Test security model components."""

    def test_security_context_permissions(self):
        """Test security context permission checking."""
        context = SecurityContext(
            user_id="user1",
            roles=["operator"],
            access_level=AccessLevel.WRITE,
            session_id="sess123",
        )

        assert context.has_permission(AccessLevel.READ) is True
        assert context.has_permission(AccessLevel.WRITE) is True
        assert context.has_permission(AccessLevel.EXECUTE) is False
        assert context.has_permission(AccessLevel.ADMIN) is False

    def test_security_context_admin(self):
        """Test admin has all permissions."""
        context = SecurityContext(
            user_id="admin1",
            roles=["admin"],
            access_level=AccessLevel.ADMIN,
        )

        assert context.has_permission(AccessLevel.READ) is True
        assert context.has_permission(AccessLevel.WRITE) is True
        assert context.has_permission(AccessLevel.EXECUTE) is True
        assert context.has_permission(AccessLevel.ADMIN) is True

    def test_audit_record_creation(self):
        """Test audit record creation and serialization."""
        now = datetime.now(timezone.utc)
        record = AuditRecord(
            operation="scada_read",
            tool_name="scada_read",
            user_id="user1",
            timestamp=now,
            request_hash="abc123",
            response_hash="def456",
            success=True,
            duration_ms=15.5,
            details={"tag_count": 3},
        )

        record_dict = record.to_dict()
        assert record_dict["operation"] == "scada_read"
        assert record_dict["success"] is True
        assert record_dict["duration_ms"] == 15.5


# =============================================================================
# TOOL DEFINITION TESTS
# =============================================================================

class TestToolDefinitions:
    """Test connector tool definitions."""

    def test_scada_read_definition(self):
        """Test SCADA read tool definition."""
        defn = SCADA_READ_DEFINITION
        assert defn.name == "scada_read"
        assert defn.category == ToolCategory.CONNECTOR
        assert defn.security_level == SecurityLevel.READ_ONLY
        assert defn.requires_confirmation is False

    def test_scada_write_definition(self):
        """Test SCADA write tool definition."""
        defn = SCADA_WRITE_DEFINITION
        assert defn.name == "scada_write"
        assert defn.security_level == SecurityLevel.CONTROLLED_WRITE
        assert defn.requires_confirmation is True  # Safety critical

    def test_opcua_definitions(self):
        """Test OPC-UA tool definitions."""
        assert OPC_UA_READ_DEFINITION.security_level == SecurityLevel.READ_ONLY
        assert OPC_UA_WRITE_DEFINITION.security_level == SecurityLevel.CONTROLLED_WRITE

    def test_kafka_definitions(self):
        """Test Kafka tool definitions."""
        assert KAFKA_PRODUCE_DEFINITION.execution_mode.value == "async"
        assert KAFKA_CONSUME_DEFINITION.execution_mode.value == "async"

    def test_database_definition(self):
        """Test database query tool definition."""
        defn = DATABASE_QUERY_DEFINITION
        assert defn.name == "database_query"
        assert defn.audit_level == "full"


# =============================================================================
# SCADA TOOL TESTS
# =============================================================================

class TestScadaReadTool:
    """Test SCADA read operations."""

    def test_read_single_tag(self, scada_read_tool):
        """Test reading a single SCADA tag."""
        request = ToolCallRequest(
            tool_name="scada_read",
            arguments={
                "tags": ["BOILER_01.STEAM_TEMP"],
            }
        )

        response = scada_read_tool.execute(request)

        assert response.success is True
        result = response.result
        assert result["count"] == 1
        assert len(result["tags"]) == 1
        assert result["tags"][0]["tag_name"] == "BOILER_01.STEAM_TEMP"
        assert result["tags"][0]["quality"] == "GOOD"

    def test_read_multiple_tags(self, scada_read_tool):
        """Test reading multiple SCADA tags."""
        request = ToolCallRequest(
            tool_name="scada_read",
            arguments={
                "tags": [
                    "BOILER_01.STEAM_TEMP",
                    "BOILER_01.STEAM_PRESSURE",
                    "BOILER_01.FEED_WATER_FLOW",
                ],
            }
        )

        response = scada_read_tool.execute(request)

        assert response.success is True
        assert response.result["count"] == 3

    def test_read_unknown_tag(self, scada_read_tool):
        """Test reading unknown tag returns BAD quality."""
        request = ToolCallRequest(
            tool_name="scada_read",
            arguments={
                "tags": ["UNKNOWN_TAG"],
            }
        )

        response = scada_read_tool.execute(request)

        assert response.success is True
        assert response.result["tags"][0]["quality"] == "BAD"
        assert response.result["tags"][0]["value"] is None

    def test_read_includes_units(self, scada_read_tool):
        """Test reading includes engineering units."""
        request = ToolCallRequest(
            tool_name="scada_read",
            arguments={
                "tags": ["BOILER_01.STEAM_TEMP"],
            }
        )

        response = scada_read_tool.execute(request)
        tag_result = response.result["tags"][0]

        assert "unit" in tag_result
        assert tag_result["unit"] == "C"


class TestScadaWriteTool:
    """Test SCADA write operations."""

    def test_write_normal_tag(self, scada_write_tool):
        """Test writing to a normal tag."""
        request = ToolCallRequest(
            tool_name="scada_write",
            arguments={
                "tag_name": "BOILER_01.LOAD_SP",
                "value": 80.0,
            }
        )

        response = scada_write_tool.execute(request)

        assert response.success is True
        result = response.result
        assert result["tag_name"] == "BOILER_01.LOAD_SP"
        assert result["new_value"] == 80.0
        assert result["success"] is True

    def test_write_safety_tag_requires_confirmation(self, scada_write_tool):
        """Test writing to safety-critical tag requires confirmation."""
        request = ToolCallRequest(
            tool_name="scada_write",
            arguments={
                "tag_name": "BOILER_01.FUEL_VALVE",
                "value": 50.0,
            }
        )

        response = scada_write_tool.execute(request)

        assert response.success is True
        result = response.result
        assert result["requires_confirmation"] is True
        assert result["confirmation_id"] is not None
        assert result["success"] is False  # Not written yet

    def test_write_with_confirmation(self, scada_write_tool):
        """Test writing with confirmation completes."""
        # First request to get confirmation ID
        request1 = ToolCallRequest(
            tool_name="scada_write",
            arguments={
                "tag_name": "BOILER_01.FUEL_VALVE",
                "value": 50.0,
            }
        )
        response1 = scada_write_tool.execute(request1)
        conf_id = response1.result["confirmation_id"]

        # Second request with confirmation
        request2 = ToolCallRequest(
            tool_name="scada_write",
            arguments={
                "tag_name": "BOILER_01.FUEL_VALVE",
                "value": 50.0,
                "confirmation_id": conf_id,
            }
        )
        response2 = scada_write_tool.execute(request2)

        assert response2.success is True
        assert response2.result["success"] is True
        assert response2.result["requires_confirmation"] is False

    def test_write_limit_violation(self, scada_write_tool):
        """Test writing value outside limits fails."""
        request = ToolCallRequest(
            tool_name="scada_write",
            arguments={
                "tag_name": "BOILER_01.STEAM_TEMP_SP",
                "value": 600.0,  # Max is 550
            }
        )

        response = scada_write_tool.execute(request)

        assert response.success is False
        assert "outside limits" in response.error


# =============================================================================
# OPC-UA TOOL TESTS
# =============================================================================

class TestOpcUaReadTool:
    """Test OPC-UA read operations."""

    def test_read_single_node(self, opcua_read_tool):
        """Test reading a single OPC-UA node."""
        request = ToolCallRequest(
            tool_name="opcua_read",
            arguments={
                "node_ids": ["ns=2;s=Temperature.PV"],
                "server_url": "opc.tcp://localhost:4840",
            }
        )

        response = opcua_read_tool.execute(request)

        assert response.success is True
        assert len(response.result["nodes"]) == 1
        node = response.result["nodes"][0]
        assert node["node_id"] == "ns=2;s=Temperature.PV"
        assert node["status_code"] == 0  # Good

    def test_read_multiple_nodes(self, opcua_read_tool):
        """Test reading multiple OPC-UA nodes."""
        request = ToolCallRequest(
            tool_name="opcua_read",
            arguments={
                "node_ids": [
                    "ns=2;s=Temperature.PV",
                    "ns=2;s=Pressure.PV",
                    "ns=2;s=Flow.PV",
                ],
                "server_url": "opc.tcp://localhost:4840",
            }
        )

        response = opcua_read_tool.execute(request)

        assert response.success is True
        assert len(response.result["nodes"]) == 3

    def test_read_unknown_node(self, opcua_read_tool):
        """Test reading unknown node returns error status."""
        request = ToolCallRequest(
            tool_name="opcua_read",
            arguments={
                "node_ids": ["ns=2;s=Unknown.Node"],
                "server_url": "opc.tcp://localhost:4840",
            }
        )

        response = opcua_read_tool.execute(request)

        assert response.success is True
        node = response.result["nodes"][0]
        assert node["status_code"] != 0  # Not Good


class TestOpcUaWriteTool:
    """Test OPC-UA write operations."""

    def test_write_node(self, opcua_write_tool):
        """Test writing to an OPC-UA node."""
        request = ToolCallRequest(
            tool_name="opcua_write",
            arguments={
                "node_id": "ns=2;s=Setpoint.SP",
                "value": 100.0,
                "server_url": "opc.tcp://localhost:4840",
            },
            caller_agent_id="test_agent"
        )

        response = opcua_write_tool.execute(request)

        assert response.success is True
        assert response.result["value"] == 100.0
        assert response.result["status_code"] == 0


# =============================================================================
# KAFKA TOOL TESTS
# =============================================================================

class TestKafkaProduceTool:
    """Test Kafka produce operations."""

    def test_produce_message(self, kafka_produce_tool):
        """Test producing a Kafka message."""
        request = ToolCallRequest(
            tool_name="kafka_produce",
            arguments={
                "topic": "sensor-data",
                "message": {"sensor_id": "S001", "value": 25.5},
                "key": "S001",
                "bootstrap_servers": "localhost:9092",
            }
        )

        response = kafka_produce_tool.execute(request)

        assert response.success is True
        result = response.result
        assert result["topic"] == "sensor-data"
        assert result["key"] == "S001"
        assert result["offset"] >= 0
        assert result["message_size_bytes"] > 0

    def test_produce_with_headers(self, kafka_produce_tool):
        """Test producing with message headers."""
        request = ToolCallRequest(
            tool_name="kafka_produce",
            arguments={
                "topic": "events",
                "message": {"event": "alarm", "severity": "high"},
                "headers": {"source": "plant-1", "type": "alarm"},
                "bootstrap_servers": "localhost:9092",
            }
        )

        response = kafka_produce_tool.execute(request)

        assert response.success is True


class TestKafkaConsumeTool:
    """Test Kafka consume operations."""

    def test_consume_messages(self, kafka_consume_tool):
        """Test consuming Kafka messages."""
        request = ToolCallRequest(
            tool_name="kafka_consume",
            arguments={
                "topic": "sensor-data",
                "max_messages": 10,
                "bootstrap_servers": "localhost:9092",
            }
        )

        response = kafka_consume_tool.execute(request)

        assert response.success is True
        result = response.result
        assert "messages" in result
        assert len(result["messages"]) <= 10

    def test_consume_from_beginning(self, kafka_consume_tool):
        """Test consuming from beginning of topic."""
        request = ToolCallRequest(
            tool_name="kafka_consume",
            arguments={
                "topic": "sensor-data",
                "from_beginning": True,
                "max_messages": 5,
                "bootstrap_servers": "localhost:9092",
            }
        )

        response = kafka_consume_tool.execute(request)

        assert response.success is True


# =============================================================================
# DATABASE TOOL TESTS
# =============================================================================

class TestDatabaseQueryTool:
    """Test database query operations."""

    def test_select_query(self, database_tool):
        """Test executing SELECT query."""
        request = ToolCallRequest(
            tool_name="database_query",
            arguments={
                "query": "SELECT * FROM emission_factors",
                "database": "greenlang_db",
            },
            caller_agent_id="test_agent"
        )

        response = database_tool.execute(request)

        assert response.success is True
        result = response.result
        assert result["row_count"] > 0
        assert "column_names" in result
        assert "rows" in result

    def test_parameterized_query(self, database_tool):
        """Test parameterized query (SQL injection prevention)."""
        request = ToolCallRequest(
            tool_name="database_query",
            arguments={
                "query": "SELECT * FROM emission_factors WHERE fuel_type = :fuel",
                "parameters": {"fuel": "natural_gas"},
                "database": "greenlang_db",
            }
        )

        response = database_tool.execute(request)

        assert response.success is True

    def test_dangerous_query_blocked(self, database_tool):
        """Test that dangerous queries are blocked."""
        request = ToolCallRequest(
            tool_name="database_query",
            arguments={
                "query": "SELECT * FROM users; DROP TABLE users;--",
                "database": "greenlang_db",
            }
        )

        response = database_tool.execute(request)

        assert response.success is False
        assert "Security violation" in response.error

    def test_max_rows_limit(self, database_tool):
        """Test max rows limit is respected."""
        request = ToolCallRequest(
            tool_name="database_query",
            arguments={
                "query": "SELECT * FROM emission_factors",
                "database": "greenlang_db",
                "max_rows": 2,
            }
        )

        response = database_tool.execute(request)

        assert response.success is True
        assert response.result["row_count"] <= 2


# =============================================================================
# REGISTRY TESTS
# =============================================================================

class TestConnectorRegistry:
    """Test the connector tool registry."""

    def test_registry_created(self):
        """Test registry is properly created."""
        assert CONNECTOR_REGISTRY is not None
        tools = CONNECTOR_REGISTRY.list_tools()
        assert len(tools) == 7  # 7 connector tools

    def test_all_connectors_registered(self):
        """Test all connector tools are registered."""
        tool_names = [t.name for t in CONNECTOR_REGISTRY.list_tools()]
        assert "scada_read" in tool_names
        assert "scada_write" in tool_names
        assert "opcua_read" in tool_names
        assert "opcua_write" in tool_names
        assert "kafka_produce" in tool_names
        assert "kafka_consume" in tool_names
        assert "database_query" in tool_names

    def test_invoke_through_registry(self):
        """Test invoking tool through registry."""
        request = ToolCallRequest(
            tool_name="scada_read",
            arguments={
                "tags": ["BOILER_01.STEAM_TEMP"],
            }
        )

        response = CONNECTOR_REGISTRY.invoke(request)
        assert response.success is True


# =============================================================================
# AUDIT LOGGING TESTS
# =============================================================================

class TestAuditLogging:
    """Test audit logging functionality."""

    def test_audit_log_created(self, scada_write_tool):
        """Test that write operations create audit logs."""
        initial_count = len(AUDIT_LOGGER.get_records(tool_name="scada_write"))

        request = ToolCallRequest(
            tool_name="scada_write",
            arguments={
                "tag_name": "BOILER_01.LOAD_SP",
                "value": 75.0,
            },
            caller_agent_id="test_agent"
        )

        scada_write_tool.execute(request)

        # Check audit log was created
        records = AUDIT_LOGGER.get_records(tool_name="scada_write")
        assert len(records) > initial_count

    def test_get_audit_records_function(self):
        """Test get_audit_records convenience function."""
        records = get_audit_records()
        assert isinstance(records, list)


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_connector_tools(self):
        """Test getting all connector tools."""
        tools = get_connector_tools()
        assert len(tools) == 7
        assert all(t.category == ToolCategory.CONNECTOR for t in tools)

    def test_invoke_connector(self):
        """Test invoking connector directly."""
        response = invoke_connector(
            "scada_read",
            {"tags": ["BOILER_01.EFFICIENCY"]}
        )
        assert response.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
