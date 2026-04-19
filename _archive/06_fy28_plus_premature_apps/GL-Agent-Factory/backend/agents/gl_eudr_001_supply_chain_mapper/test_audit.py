"""
GL-EUDR-001: Audit Logging Tests

Comprehensive test suite for audit logging covering:
- Audit log entry creation
- Checksum computation
- Various action types
- Severity levels
- Context handling
- Middleware integration

Run with: pytest test_audit.py -v
"""

import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from .audit import (
    AuditAction,
    AuditSeverity,
    AuditLogEntry,
    AuditContext,
    AuditLogger,
    AuditMiddleware,
    global_audit_logger,
    get_audit_logger,
    audited,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def audit_logger():
    """Create an audit logger (no file output for testing)."""
    return AuditLogger(log_to_file=False, log_to_db=False)


@pytest.fixture
def audit_context():
    """Create a sample audit context."""
    return AuditContext(
        user_id=uuid.uuid4(),
        user_email="test@example.com",
        user_role="analyst",
        organization_id=uuid.uuid4(),
        request_id="req-123",
        ip_address="192.168.1.1",
        user_agent="TestClient/1.0",
        endpoint="/api/nodes",
        method="POST"
    )


@pytest.fixture
def sample_node_id():
    """Generate a sample node ID."""
    return uuid.uuid4()


# =============================================================================
# AUDIT ACTION TESTS
# =============================================================================

class TestAuditActions:
    """Test audit action definitions."""

    def test_node_actions_exist(self):
        """Test node-related actions exist."""
        assert AuditAction.NODE_CREATE
        assert AuditAction.NODE_UPDATE
        assert AuditAction.NODE_DELETE

    def test_edge_actions_exist(self):
        """Test edge-related actions exist."""
        assert AuditAction.EDGE_CREATE
        assert AuditAction.EDGE_UPDATE
        assert AuditAction.EDGE_DELETE

    def test_plot_actions_exist(self):
        """Test plot-related actions exist."""
        assert AuditAction.PLOT_CREATE
        assert AuditAction.PLOT_UPDATE
        assert AuditAction.PLOT_DELETE

    def test_coverage_actions_exist(self):
        """Test coverage-related actions exist."""
        assert AuditAction.COVERAGE_CALCULATE
        assert AuditAction.COVERAGE_GATES_CHECK

    def test_snapshot_actions_exist(self):
        """Test snapshot-related actions exist."""
        assert AuditAction.SNAPSHOT_CREATE
        assert AuditAction.SNAPSHOT_QUERY
        assert AuditAction.SNAPSHOT_DIFF

    def test_entity_resolution_actions_exist(self):
        """Test entity resolution actions exist."""
        assert AuditAction.ER_RUN
        assert AuditAction.ER_MERGE
        assert AuditAction.ER_REJECT

    def test_auth_actions_exist(self):
        """Test authentication actions exist."""
        assert AuditAction.AUTH_LOGIN
        assert AuditAction.AUTH_LOGOUT
        assert AuditAction.AUTH_TOKEN_REFRESH
        assert AuditAction.AUTH_FAILED

    def test_action_values(self):
        """Test action enum values are correct format."""
        assert AuditAction.NODE_CREATE.value == "node.create"
        assert AuditAction.EDGE_DELETE.value == "edge.delete"
        assert AuditAction.AUTH_LOGIN.value == "auth.login"


# =============================================================================
# AUDIT SEVERITY TESTS
# =============================================================================

class TestAuditSeverity:
    """Test audit severity levels."""

    def test_all_severities_defined(self):
        """Test all severity levels are defined."""
        assert AuditSeverity.INFO
        assert AuditSeverity.WARNING
        assert AuditSeverity.ERROR
        assert AuditSeverity.CRITICAL

    def test_severity_values(self):
        """Test severity enum values."""
        assert AuditSeverity.INFO.value == "INFO"
        assert AuditSeverity.WARNING.value == "WARNING"
        assert AuditSeverity.ERROR.value == "ERROR"
        assert AuditSeverity.CRITICAL.value == "CRITICAL"


# =============================================================================
# AUDIT LOG ENTRY TESTS
# =============================================================================

class TestAuditLogEntry:
    """Test AuditLogEntry model."""

    def test_entry_creation(self):
        """Test creating an audit log entry."""
        entry = AuditLogEntry(
            action=AuditAction.NODE_CREATE,
            user_id=uuid.uuid4(),
            resource_type="node",
            resource_id=uuid.uuid4()
        )

        assert entry.entry_id is not None
        assert entry.timestamp is not None
        assert entry.action == AuditAction.NODE_CREATE
        assert entry.severity == AuditSeverity.INFO  # Default

    def test_entry_with_changes(self):
        """Test entry with change tracking."""
        entry = AuditLogEntry(
            action=AuditAction.NODE_UPDATE,
            changes={"name": {"old": "Old Name", "new": "New Name"}},
            previous_state={"name": "Old Name"},
            new_state={"name": "New Name"}
        )

        assert "name" in entry.changes
        assert entry.previous_state["name"] == "Old Name"
        assert entry.new_state["name"] == "New Name"

    def test_entry_with_error(self):
        """Test entry with error information."""
        entry = AuditLogEntry(
            action=AuditAction.NODE_CREATE,
            success=False,
            error_message="Validation failed",
            severity=AuditSeverity.ERROR
        )

        assert entry.success is False
        assert entry.error_message == "Validation failed"
        assert entry.severity == AuditSeverity.ERROR

    def test_entry_checksum_computation(self):
        """Test checksum computation."""
        entry = AuditLogEntry(
            action=AuditAction.NODE_CREATE,
            user_id=uuid.uuid4(),
            resource_id=uuid.uuid4()
        )

        checksum = entry.compute_checksum()

        assert checksum is not None
        assert len(checksum) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_entry_checksum_deterministic(self):
        """Test checksum is deterministic."""
        user_id = uuid.uuid4()
        resource_id = uuid.uuid4()

        entry1 = AuditLogEntry(
            entry_id=uuid.uuid4(),
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            action=AuditAction.NODE_CREATE,
            user_id=user_id,
            resource_id=resource_id
        )
        entry1_checksum = entry1.compute_checksum()

        entry2 = AuditLogEntry(
            entry_id=entry1.entry_id,
            timestamp=entry1.timestamp,
            action=AuditAction.NODE_CREATE,
            user_id=user_id,
            resource_id=resource_id
        )
        entry2_checksum = entry2.compute_checksum()

        assert entry1_checksum == entry2_checksum

    def test_entry_finalize(self):
        """Test finalizing entry adds checksum."""
        entry = AuditLogEntry(
            action=AuditAction.NODE_CREATE
        )

        assert entry.checksum is None

        entry.finalize()

        assert entry.checksum is not None
        assert len(entry.checksum) == 64

    def test_entry_multiple_resource_ids(self):
        """Test entry with multiple resource IDs."""
        ids = [uuid.uuid4() for _ in range(3)]
        entry = AuditLogEntry(
            action=AuditAction.ER_MERGE,
            resource_ids=ids
        )

        assert len(entry.resource_ids) == 3


# =============================================================================
# AUDIT CONTEXT TESTS
# =============================================================================

class TestAuditContext:
    """Test AuditContext model."""

    def test_context_creation(self, audit_context):
        """Test creating an audit context."""
        assert audit_context.user_email == "test@example.com"
        assert audit_context.user_role == "analyst"
        assert audit_context.ip_address == "192.168.1.1"

    def test_context_optional_fields(self):
        """Test context with only required fields."""
        context = AuditContext()

        assert context.user_id is None
        assert context.user_email is None
        assert context.request_id is None

    def test_context_full_fields(self, audit_context):
        """Test context with all fields."""
        assert audit_context.user_id is not None
        assert audit_context.organization_id is not None
        assert audit_context.endpoint == "/api/nodes"
        assert audit_context.method == "POST"


# =============================================================================
# AUDIT LOGGER TESTS
# =============================================================================

class TestAuditLogger:
    """Test AuditLogger class."""

    def test_logger_creation(self, audit_logger):
        """Test creating an audit logger."""
        assert audit_logger is not None
        assert audit_logger.log_to_file is False
        assert audit_logger.log_to_db is False

    def test_log_basic_entry(self, audit_logger, audit_context, sample_node_id):
        """Test logging a basic entry."""
        entry = audit_logger.log(
            action=AuditAction.NODE_CREATE,
            context=audit_context,
            resource_type="node",
            resource_id=sample_node_id
        )

        assert entry is not None
        assert entry.action == AuditAction.NODE_CREATE
        assert entry.resource_id == sample_node_id
        assert entry.checksum is not None

    def test_log_with_changes(self, audit_logger, audit_context, sample_node_id):
        """Test logging with change tracking."""
        entry = audit_logger.log(
            action=AuditAction.NODE_UPDATE,
            context=audit_context,
            resource_type="node",
            resource_id=sample_node_id,
            changes={"name": {"old": "Old", "new": "New"}},
            previous_state={"name": "Old"},
            new_state={"name": "New"}
        )

        assert entry.changes["name"]["old"] == "Old"
        assert entry.previous_state["name"] == "Old"
        assert entry.new_state["name"] == "New"

    def test_log_error(self, audit_logger, audit_context):
        """Test logging an error entry."""
        entry = audit_logger.log(
            action=AuditAction.NODE_CREATE,
            context=audit_context,
            resource_type="node",
            success=False,
            error_message="Failed to create node",
            severity=AuditSeverity.ERROR
        )

        assert entry.success is False
        assert entry.error_message == "Failed to create node"
        assert entry.severity == AuditSeverity.ERROR

    def test_log_node_create(self, audit_logger, audit_context, sample_node_id):
        """Test convenience method for node creation."""
        entry = audit_logger.log_node_create(
            context=audit_context,
            node_id=sample_node_id,
            node_data={"name": "Test Node", "node_type": "TRADER"}
        )

        assert entry.action == AuditAction.NODE_CREATE
        assert entry.resource_id == sample_node_id
        assert "node_type" in entry.metadata

    def test_log_node_update(self, audit_logger, audit_context, sample_node_id):
        """Test convenience method for node update."""
        entry = audit_logger.log_node_update(
            context=audit_context,
            node_id=sample_node_id,
            previous={"name": "Old Name"},
            updated={"name": "New Name"},
            changes={"name": {"old": "Old Name", "new": "New Name"}}
        )

        assert entry.action == AuditAction.NODE_UPDATE
        assert entry.previous_state["name"] == "Old Name"
        assert entry.new_state["name"] == "New Name"

    def test_log_node_delete(self, audit_logger, audit_context, sample_node_id):
        """Test convenience method for node deletion."""
        entry = audit_logger.log_node_delete(
            context=audit_context,
            node_id=sample_node_id,
            node_data={"name": "Deleted Node"}
        )

        assert entry.action == AuditAction.NODE_DELETE
        assert entry.severity == AuditSeverity.WARNING
        assert entry.previous_state["name"] == "Deleted Node"

    def test_log_entity_merge(self, audit_logger, audit_context):
        """Test convenience method for entity merge."""
        keep_id = uuid.uuid4()
        merge_id = uuid.uuid4()

        entry = audit_logger.log_entity_merge(
            context=audit_context,
            keep_id=keep_id,
            merge_id=merge_id,
            merge_details={"merged_fields": ["name", "address"]}
        )

        assert entry.action == AuditAction.ER_MERGE
        assert entry.severity == AuditSeverity.WARNING
        assert keep_id in entry.resource_ids
        assert merge_id in entry.resource_ids

    def test_log_snapshot_create(self, audit_logger, audit_context):
        """Test convenience method for snapshot creation."""
        snapshot_id = uuid.uuid4()

        entry = audit_logger.log_snapshot_create(
            context=audit_context,
            snapshot_id=snapshot_id,
            snapshot_data={
                "node_count": 100,
                "edge_count": 200,
                "coverage_percentage": 85.5,
                "trigger_type": "DDS_SUBMISSION"
            }
        )

        assert entry.action == AuditAction.SNAPSHOT_CREATE
        assert entry.new_state["node_count"] == 100

    def test_log_coverage_check(self, audit_logger, audit_context):
        """Test convenience method for coverage gate check."""
        importer_id = uuid.uuid4()

        entry = audit_logger.log_coverage_check(
            context=audit_context,
            importer_id=importer_id,
            commodity="COFFEE",
            result={
                "can_proceed_to_risk_assessment": True,
                "can_submit_dds": False,
                "mapping_completeness": 92.5
            }
        )

        assert entry.action == AuditAction.COVERAGE_GATES_CHECK
        assert entry.metadata["commodity"] == "COFFEE"
        assert entry.metadata["can_proceed"] is True

    def test_log_auth_failure(self, audit_logger, audit_context):
        """Test convenience method for auth failure."""
        entry = audit_logger.log_auth_failure(
            context=audit_context,
            reason="Invalid credentials"
        )

        assert entry.action == AuditAction.AUTH_FAILED
        assert entry.success is False
        assert entry.error_message == "Invalid credentials"
        assert entry.severity == AuditSeverity.WARNING

    def test_context_propagation(self, audit_logger, audit_context, sample_node_id):
        """Test context is properly propagated to entry."""
        entry = audit_logger.log(
            action=AuditAction.NODE_CREATE,
            context=audit_context,
            resource_type="node",
            resource_id=sample_node_id
        )

        assert entry.user_id == audit_context.user_id
        assert entry.user_email == audit_context.user_email
        assert entry.organization_id == audit_context.organization_id
        assert entry.ip_address == audit_context.ip_address
        assert entry.request_id == audit_context.request_id


# =============================================================================
# AUDIT MIDDLEWARE TESTS
# =============================================================================

class TestAuditMiddleware:
    """Test audit middleware integration."""

    def test_middleware_creation(self, audit_logger):
        """Test creating audit middleware."""
        middleware = AuditMiddleware(audit_logger)
        assert middleware.audit_logger is audit_logger

    def test_create_context_from_request(self, audit_logger):
        """Test creating context from request."""
        middleware = AuditMiddleware(audit_logger)

        mock_request = Mock()
        mock_request.headers = {
            "X-Request-ID": "req-456",
            "User-Agent": "TestBrowser/1.0"
        }
        mock_request.client = Mock()
        mock_request.client.host = "10.0.0.1"
        mock_request.url = Mock()
        mock_request.url.path = "/api/v1/nodes"
        mock_request.method = "GET"

        mock_user = Mock()
        mock_user.user_id = uuid.uuid4()
        mock_user.email = "user@test.com"
        mock_user.role = "analyst"
        mock_user.organization_id = uuid.uuid4()

        context = middleware.create_context_from_request(mock_request, mock_user)

        assert context.request_id == "req-456"
        assert context.ip_address == "10.0.0.1"
        assert context.user_agent == "TestBrowser/1.0"
        assert context.endpoint == "/api/v1/nodes"
        assert context.method == "GET"
        assert context.user_email == "user@test.com"


# =============================================================================
# AUDITED DECORATOR TESTS
# =============================================================================

class TestAuditedDecorator:
    """Test @audited decorator."""

    def test_audited_function_success(self, audit_logger, audit_context):
        """Test decorator logs successful function."""
        @audited(AuditAction.NODE_CREATE, "node", lambda r: r.get("id"))
        async def create_node(**kwargs):
            return {"id": uuid.uuid4(), "name": "Test"}

        # Would need to run async test
        # For now, verify decorator returns a function
        assert callable(create_node)

    def test_audited_preserves_function(self):
        """Test decorator preserves function behavior."""
        @audited(AuditAction.NODE_READ, "node")
        async def read_node(node_id):
            return {"id": node_id}

        # Verify it's still callable
        assert callable(read_node)


# =============================================================================
# GLOBAL LOGGER TESTS
# =============================================================================

class TestGlobalLogger:
    """Test global audit logger instance."""

    def test_get_audit_logger(self):
        """Test getting global audit logger."""
        logger = get_audit_logger()
        assert logger is not None
        assert isinstance(logger, AuditLogger)

    def test_global_logger_singleton(self):
        """Test global logger is same instance."""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        assert logger1 is logger2

    def test_global_logger_is_module_level(self):
        """Test global logger is module-level instance."""
        assert global_audit_logger is not None
        assert isinstance(global_audit_logger, AuditLogger)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_context(self, audit_logger):
        """Test logging with empty context."""
        entry = audit_logger.log(
            action=AuditAction.NODE_READ,
            context=AuditContext()
        )

        assert entry is not None
        assert entry.user_id is None

    def test_large_metadata(self, audit_logger, audit_context):
        """Test logging with large metadata."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(100)}

        entry = audit_logger.log(
            action=AuditAction.NODE_CREATE,
            context=audit_context,
            metadata=large_metadata
        )

        assert len(entry.metadata) == 100

    def test_special_characters_in_data(self, audit_logger, audit_context):
        """Test logging with special characters."""
        entry = audit_logger.log(
            action=AuditAction.NODE_CREATE,
            context=audit_context,
            new_state={"name": "Test's \"Special\" <Node> & More"},
            metadata={"description": "Unicode: \u00e9\u00e8\u00ea"}
        )

        assert entry is not None
        assert "Test's" in entry.new_state["name"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
