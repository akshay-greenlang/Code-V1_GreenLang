"""
Pydantic Model Tests for GreenLang Agent Registry.

This module tests the Pydantic models used for request/response validation.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from greenlang_registry.models import (
    AgentMetadata,
    AgentVersion,
    PublishRequest,
    PromoteRequest,
    SemanticVersion,
    LifecycleState,
    RuntimeRequirements,
    AgentCapability,
)


# =============================================================================
# SemanticVersion Tests
# =============================================================================

class TestSemanticVersion:
    """Tests for SemanticVersion model."""

    def test_valid_semantic_version(self):
        """Test creating a valid semantic version."""
        version = SemanticVersion(major=2, minor=3, patch=1)
        assert version.major == 2
        assert version.minor == 3
        assert version.patch == 1

    def test_semantic_version_with_prerelease(self):
        """Test semantic version with prerelease tag."""
        version = SemanticVersion(major=1, minor=0, patch=0, prerelease="alpha.1")
        assert version.prerelease == "alpha.1"

    def test_semantic_version_to_string(self):
        """Test converting semantic version to string."""
        version = SemanticVersion(major=2, minor=3, patch=1)
        assert version.to_string() == "2.3.1"

    def test_semantic_version_to_string_with_prerelease(self):
        """Test converting semantic version with prerelease to string."""
        version = SemanticVersion(major=1, minor=0, patch=0, prerelease="beta")
        assert version.to_string() == "1.0.0-beta"

    def test_negative_version_number_fails(self):
        """Test that negative version numbers fail validation."""
        with pytest.raises(ValidationError):
            SemanticVersion(major=-1, minor=0, patch=0)


# =============================================================================
# AgentMetadata Tests
# =============================================================================

class TestAgentMetadata:
    """Tests for AgentMetadata model."""

    def test_valid_agent_metadata(self):
        """Test creating valid agent metadata."""
        metadata = AgentMetadata(
            agent_id="gl-test-agent",
            name="Test Agent",
            description="A test agent",
            domain="sustainability.test",
        )
        assert metadata.agent_id == "gl-test-agent"
        assert metadata.name == "Test Agent"

    def test_invalid_agent_id_pattern(self):
        """Test that invalid agent_id pattern fails."""
        with pytest.raises(ValidationError):
            AgentMetadata(
                agent_id="Invalid Agent ID!",  # Contains spaces and special chars
                name="Test",
            )

    def test_agent_id_must_start_with_letter(self):
        """Test that agent_id must start with a letter."""
        with pytest.raises(ValidationError):
            AgentMetadata(
                agent_id="123-test-agent",  # Starts with number
                name="Test",
            )

    def test_agent_metadata_with_tags(self):
        """Test agent metadata with tags."""
        metadata = AgentMetadata(
            agent_id="gl-test-agent",
            name="Test Agent",
            tags=["carbon", "emissions", "test"],
        )
        assert metadata.tags == ["carbon", "emissions", "test"]


# =============================================================================
# PublishRequest Tests
# =============================================================================

class TestPublishRequest:
    """Tests for PublishRequest model."""

    def test_valid_publish_request(self):
        """Test creating a valid publish request."""
        request = PublishRequest(
            agent_id="gl-new-agent",
            name="New Agent",
            version="1.0.0",
        )
        assert request.agent_id == "gl-new-agent"
        assert request.version == "1.0.0"

    def test_publish_request_with_full_data(self):
        """Test publish request with all optional fields."""
        request = PublishRequest(
            agent_id="gl-new-agent",
            name="New Agent",
            description="Full description",
            version="2.3.1-beta",
            domain="sustainability.carbon",
            type="calculator",
            category="emissions",
            tags=["carbon", "test"],
            team="test-team",
            tenant_id="test-tenant",
            container_image="gcr.io/test/agent:1.0.0",
            runtime_requirements=RuntimeRequirements(
                cpu_request="500m",
                memory_request="512Mi",
            ),
            capabilities=[
                AgentCapability(
                    name="calculate",
                    description="Calculate emissions",
                )
            ],
        )
        assert request.domain == "sustainability.carbon"
        assert len(request.capabilities) == 1

    def test_invalid_version_format(self):
        """Test that invalid version format fails."""
        with pytest.raises(ValidationError):
            PublishRequest(
                agent_id="gl-test-agent",
                name="Test",
                version="invalid-version",
            )

    def test_parse_semantic_version(self):
        """Test parsing version string to SemanticVersion."""
        request = PublishRequest(
            agent_id="gl-test-agent",
            name="Test",
            version="2.3.1-alpha",
        )
        semantic = request.parse_semantic_version()
        assert semantic.major == 2
        assert semantic.minor == 3
        assert semantic.patch == 1
        assert semantic.prerelease == "alpha"

    def test_version_with_build_metadata(self):
        """Test version with build metadata."""
        request = PublishRequest(
            agent_id="gl-test-agent",
            name="Test",
            version="1.0.0+build.123",
        )
        semantic = request.parse_semantic_version()
        assert semantic.build == "build.123"


# =============================================================================
# PromoteRequest Tests
# =============================================================================

class TestPromoteRequest:
    """Tests for PromoteRequest model."""

    def test_valid_promote_request(self):
        """Test creating a valid promote request."""
        request = PromoteRequest(
            target_state=LifecycleState.EXPERIMENTAL,
            reason="Ready for testing",
        )
        assert request.target_state == LifecycleState.EXPERIMENTAL
        assert request.reason == "Ready for testing"

    def test_promote_request_all_fields(self):
        """Test promote request with all fields."""
        request = PromoteRequest(
            target_state=LifecycleState.CERTIFIED,
            reason="All tests passed",
            promoted_by="qa-team",
            metadata={"approval_ticket": "JIRA-123"},
        )
        assert request.promoted_by == "qa-team"
        assert request.metadata["approval_ticket"] == "JIRA-123"

    def test_invalid_target_state(self):
        """Test that invalid target state fails."""
        with pytest.raises(ValidationError):
            PromoteRequest(
                target_state="invalid_state",
            )


# =============================================================================
# RuntimeRequirements Tests
# =============================================================================

class TestRuntimeRequirements:
    """Tests for RuntimeRequirements model."""

    def test_valid_runtime_requirements(self):
        """Test creating valid runtime requirements."""
        requirements = RuntimeRequirements(
            cpu_request="500m",
            cpu_limit="2000m",
            memory_request="512Mi",
            memory_limit="2Gi",
        )
        assert requirements.cpu_request == "500m"
        assert requirements.memory_limit == "2Gi"

    def test_runtime_requirements_with_dependencies(self):
        """Test runtime requirements with package dependencies."""
        requirements = RuntimeRequirements(
            cpu_request="500m",
            dependencies=["greenlang-sdk>=2.0.0", "pandas>=2.0"],
        )
        assert len(requirements.dependencies) == 2

    def test_runtime_requirements_with_services(self):
        """Test runtime requirements with service dependencies."""
        requirements = RuntimeRequirements(
            services=[
                {"name": "greenlang.db", "version": ">=14.0"},
                {"name": "redis", "version": ">=7.0"},
            ]
        )
        assert len(requirements.services) == 2


# =============================================================================
# AgentVersion Tests
# =============================================================================

class TestAgentVersion:
    """Tests for AgentVersion model."""

    def test_valid_agent_version(self):
        """Test creating a valid agent version."""
        version = AgentVersion(
            version_id="gl-test-agent:1.0.0",
            agent_id="gl-test-agent",
            version="1.0.0",
            lifecycle_state=LifecycleState.DRAFT,
        )
        assert version.version_id == "gl-test-agent:1.0.0"
        assert version.lifecycle_state == LifecycleState.DRAFT

    def test_agent_version_with_timestamps(self):
        """Test agent version with timestamp fields."""
        now = datetime.utcnow()
        version = AgentVersion(
            version_id="gl-test-agent:1.0.0",
            agent_id="gl-test-agent",
            version="1.0.0",
            lifecycle_state=LifecycleState.CERTIFIED,
            created_at=now,
            published_at=now,
        )
        assert version.created_at == now
        assert version.published_at == now

    def test_agent_version_deprecated(self):
        """Test deprecated agent version."""
        now = datetime.utcnow()
        version = AgentVersion(
            version_id="gl-test-agent:1.0.0",
            agent_id="gl-test-agent",
            version="1.0.0",
            lifecycle_state=LifecycleState.DEPRECATED,
            deprecated_at=now,
        )
        assert version.lifecycle_state == LifecycleState.DEPRECATED
        assert version.deprecated_at == now
