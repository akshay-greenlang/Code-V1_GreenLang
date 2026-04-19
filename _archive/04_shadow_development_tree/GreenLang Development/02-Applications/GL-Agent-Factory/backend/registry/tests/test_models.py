"""
Test Suite for Agent Registry Pydantic Models

This module contains comprehensive tests for:
- AgentRecord validation
- AgentVersion validation
- Request/Response model validation
- Checksum computation
- Status transitions

Run with: pytest backend/registry/tests/test_models.py -v
"""

import pytest
from datetime import datetime
from uuid import UUID, uuid4

from backend.registry.models import (
    AgentRecord,
    AgentVersion,
    AgentStatus,
    CertificationStatus,
    AgentCreateRequest,
    AgentUpdateRequest,
    AgentResponse,
    AgentSearchRequest,
    VersionCreateRequest,
    PublishRequest,
)


class TestAgentRecord:
    """Tests for AgentRecord model."""

    def test_create_valid_agent(self):
        """Test creating a valid agent record."""
        agent = AgentRecord(
            name="test-agent",
            version="1.0.0",
            description="Test agent description",
            category="emissions",
            author="test-user",
        )

        assert agent.name == "test-agent"
        assert agent.version == "1.0.0"
        assert agent.category == "emissions"
        assert agent.status == AgentStatus.DRAFT
        assert agent.downloads == 0
        assert agent.checksum.startswith("sha256:")

    def test_name_validation_lowercase(self):
        """Test that name is normalized to lowercase."""
        agent = AgentRecord(
            name="Test-Agent",
            version="1.0.0",
            category="test",
            author="user",
        )
        assert agent.name == "test-agent"

    def test_name_validation_invalid_start(self):
        """Test that name must start with a letter."""
        with pytest.raises(ValueError, match="must start with a letter"):
            AgentRecord(
                name="123-agent",
                version="1.0.0",
                category="test",
                author="user",
            )

    def test_name_validation_invalid_chars(self):
        """Test that name can only contain valid characters."""
        with pytest.raises(ValueError, match="alphanumeric"):
            AgentRecord(
                name="test@agent",
                version="1.0.0",
                category="test",
                author="user",
            )

    def test_version_validation_valid(self):
        """Test valid semantic versions."""
        valid_versions = [
            "1.0.0",
            "0.1.0",
            "10.20.30",
            "1.0.0-alpha",
            "1.0.0-beta.1",
            "1.0.0+build.123",
        ]

        for version in valid_versions:
            agent = AgentRecord(
                name="test",
                version=version,
                category="test",
                author="user",
            )
            assert agent.version == version

    def test_version_validation_invalid(self):
        """Test invalid semantic versions."""
        invalid_versions = [
            "1.0",
            "1",
            "1.0.0.0",
            "v1.0.0",
            "1.a.0",
        ]

        for version in invalid_versions:
            with pytest.raises(ValueError):
                AgentRecord(
                    name="test",
                    version=version,
                    category="test",
                    author="user",
                )

    def test_category_normalized(self):
        """Test that category is normalized to lowercase."""
        agent = AgentRecord(
            name="test",
            version="1.0.0",
            category="  EMISSIONS  ",
            author="user",
        )
        assert agent.category == "emissions"

    def test_checksum_computed(self):
        """Test that checksum is automatically computed."""
        agent = AgentRecord(
            name="test",
            version="1.0.0",
            category="test",
            author="user",
        )
        assert agent.checksum
        assert agent.checksum.startswith("sha256:")
        assert len(agent.checksum) == 71  # "sha256:" + 64 hex chars

    def test_checksum_deterministic(self):
        """Test that checksum is deterministic."""
        agent1 = AgentRecord(
            name="test",
            version="1.0.0",
            category="test",
            author="user",
            pack_yaml={"key": "value"},
        )
        agent2 = AgentRecord(
            name="test",
            version="1.0.0",
            category="test",
            author="user",
            pack_yaml={"key": "value"},
        )
        assert agent1.checksum == agent2.checksum

    def test_checksum_changes_with_content(self):
        """Test that checksum changes when content changes."""
        agent1 = AgentRecord(
            name="test",
            version="1.0.0",
            category="test",
            author="user",
            pack_yaml={"key": "value1"},
        )
        agent2 = AgentRecord(
            name="test",
            version="1.0.0",
            category="test",
            author="user",
            pack_yaml={"key": "value2"},
        )
        assert agent1.checksum != agent2.checksum

    def test_increment_downloads(self):
        """Test increment_downloads method."""
        agent = AgentRecord(
            name="test",
            version="1.0.0",
            category="test",
            author="user",
            downloads=5,
        )
        updated = agent.increment_downloads()
        assert updated.downloads == 6
        assert agent.downloads == 5  # Original unchanged

    def test_deprecate(self):
        """Test deprecate method."""
        agent = AgentRecord(
            name="test",
            version="1.0.0",
            category="test",
            author="user",
            status=AgentStatus.PUBLISHED,
        )
        deprecated = agent.deprecate()
        assert deprecated.status == AgentStatus.DEPRECATED
        assert agent.status == AgentStatus.PUBLISHED  # Original unchanged

    def test_publish_from_draft(self):
        """Test publish method from draft status."""
        agent = AgentRecord(
            name="test",
            version="1.0.0",
            category="test",
            author="user",
            status=AgentStatus.DRAFT,
        )
        published = agent.publish()
        assert published.status == AgentStatus.PUBLISHED

    def test_publish_from_deprecated_fails(self):
        """Test that publishing deprecated agent fails."""
        agent = AgentRecord(
            name="test",
            version="1.0.0",
            category="test",
            author="user",
            status=AgentStatus.DEPRECATED,
        )
        with pytest.raises(ValueError, match="Cannot publish a deprecated agent"):
            agent.publish()

    def test_certification_status(self):
        """Test certification status field."""
        agent = AgentRecord(
            name="test",
            version="1.0.0",
            category="test",
            author="user",
            certification_status=[
                CertificationStatus(
                    framework="CBAM",
                    certified=True,
                    certified_at=datetime.utcnow(),
                    certified_by="auditor",
                )
            ],
        )
        assert len(agent.certification_status) == 1
        assert agent.certification_status[0].framework == "CBAM"
        assert agent.certification_status[0].certified is True

    def test_uuid_auto_generated(self):
        """Test that UUID is auto-generated."""
        agent = AgentRecord(
            name="test",
            version="1.0.0",
            category="test",
            author="user",
        )
        assert isinstance(agent.id, UUID)

    def test_timestamps_auto_generated(self):
        """Test that timestamps are auto-generated."""
        agent = AgentRecord(
            name="test",
            version="1.0.0",
            category="test",
            author="user",
        )
        assert isinstance(agent.created_at, datetime)
        assert isinstance(agent.updated_at, datetime)


class TestAgentVersion:
    """Tests for AgentVersion model."""

    def test_create_valid_version(self):
        """Test creating a valid version record."""
        agent_id = uuid4()
        version = AgentVersion(
            agent_id=agent_id,
            version="1.0.0",
            changelog="Initial release",
            breaking_changes=False,
        )

        assert version.agent_id == agent_id
        assert version.version == "1.0.0"
        assert version.changelog == "Initial release"
        assert version.breaking_changes is False
        assert version.is_latest is False

    def test_version_validation(self):
        """Test version string validation."""
        with pytest.raises(ValueError):
            AgentVersion(
                agent_id=uuid4(),
                version="invalid",
            )

    def test_compare_version_higher(self):
        """Test version comparison - higher."""
        v1 = AgentVersion(agent_id=uuid4(), version="2.0.0")
        v2 = AgentVersion(agent_id=uuid4(), version="1.0.0")
        assert v1.compare_version(v2) == 1

    def test_compare_version_lower(self):
        """Test version comparison - lower."""
        v1 = AgentVersion(agent_id=uuid4(), version="1.0.0")
        v2 = AgentVersion(agent_id=uuid4(), version="2.0.0")
        assert v1.compare_version(v2) == -1

    def test_compare_version_equal(self):
        """Test version comparison - equal."""
        v1 = AgentVersion(agent_id=uuid4(), version="1.0.0")
        v2 = AgentVersion(agent_id=uuid4(), version="1.0.0")
        assert v1.compare_version(v2) == 0

    def test_compare_version_patch(self):
        """Test version comparison - patch difference."""
        v1 = AgentVersion(agent_id=uuid4(), version="1.0.1")
        v2 = AgentVersion(agent_id=uuid4(), version="1.0.0")
        assert v1.compare_version(v2) == 1


class TestCertificationStatus:
    """Tests for CertificationStatus model."""

    def test_create_certification(self):
        """Test creating certification status."""
        cert = CertificationStatus(
            framework="CSRD",
            certified=True,
            certified_at=datetime.utcnow(),
            certified_by="auditor@company.com",
            notes="Passed compliance audit",
        )

        assert cert.framework == "CSRD"
        assert cert.certified is True
        assert cert.certified_by == "auditor@company.com"

    def test_optional_fields(self):
        """Test optional fields are None by default."""
        cert = CertificationStatus(framework="TEST")

        assert cert.certified is False
        assert cert.certified_at is None
        assert cert.certified_by is None
        assert cert.expiry_date is None
        assert cert.notes is None


class TestAgentCreateRequest:
    """Tests for AgentCreateRequest model."""

    def test_valid_request(self):
        """Test valid create request."""
        request = AgentCreateRequest(
            name="new-agent",
            version="1.0.0",
            description="Test description",
            category="emissions",
            author="test-user",
        )

        assert request.name == "new-agent"
        assert request.version == "1.0.0"

    def test_name_normalized(self):
        """Test name normalization."""
        request = AgentCreateRequest(
            name="NEW-Agent",
            category="test",
            author="user",
        )
        assert request.name == "new-agent"

    def test_default_values(self):
        """Test default values."""
        request = AgentCreateRequest(
            name="test",
            category="test",
            author="user",
        )

        assert request.version == "1.0.0"
        assert request.description == ""
        assert request.tags == []
        assert request.license == "Apache-2.0"

    def test_name_min_length(self):
        """Test name minimum length."""
        with pytest.raises(ValueError):
            AgentCreateRequest(
                name="ab",  # Too short
                category="test",
                author="user",
            )


class TestAgentUpdateRequest:
    """Tests for AgentUpdateRequest model."""

    def test_partial_update(self):
        """Test partial update - only some fields."""
        request = AgentUpdateRequest(description="Updated description")

        assert request.description == "Updated description"
        assert request.tags is None
        assert request.pack_yaml is None

    def test_all_fields_optional(self):
        """Test all fields are optional."""
        request = AgentUpdateRequest()

        assert request.description is None
        assert request.pack_yaml is None
        assert request.tags is None


class TestAgentSearchRequest:
    """Tests for AgentSearchRequest model."""

    def test_valid_search(self):
        """Test valid search request."""
        request = AgentSearchRequest(
            query="carbon emissions",
            category="regulatory",
            limit=10,
        )

        assert request.query == "carbon emissions"
        assert request.category == "regulatory"
        assert request.limit == 10

    def test_default_pagination(self):
        """Test default pagination values."""
        request = AgentSearchRequest(query="test")

        assert request.limit == 20
        assert request.offset == 0
        assert request.sort_by == "downloads"
        assert request.sort_order == "desc"

    def test_limit_bounds(self):
        """Test limit boundaries."""
        with pytest.raises(ValueError):
            AgentSearchRequest(query="test", limit=0)

        with pytest.raises(ValueError):
            AgentSearchRequest(query="test", limit=101)


class TestVersionCreateRequest:
    """Tests for VersionCreateRequest model."""

    def test_valid_request(self):
        """Test valid version create request."""
        request = VersionCreateRequest(
            version="2.0.0",
            changelog="Major update",
            breaking_changes=True,
        )

        assert request.version == "2.0.0"
        assert request.changelog == "Major update"
        assert request.breaking_changes is True

    def test_invalid_version(self):
        """Test invalid version format."""
        with pytest.raises(ValueError):
            VersionCreateRequest(version="invalid")


class TestPublishRequest:
    """Tests for PublishRequest model."""

    def test_valid_request(self):
        """Test valid publish request."""
        request = PublishRequest(
            version="1.0.0",
            release_notes="Initial public release",
            certifications=["CBAM", "CSRD"],
        )

        assert request.version == "1.0.0"
        assert request.release_notes == "Initial public release"
        assert len(request.certifications) == 2

    def test_optional_certifications(self):
        """Test optional certifications."""
        request = PublishRequest(version="1.0.0")

        assert request.certifications is None
        assert request.release_notes is None


class TestAgentResponse:
    """Tests for AgentResponse model."""

    def test_from_dict(self):
        """Test creating response from dictionary."""
        data = {
            "id": uuid4(),
            "name": "test-agent",
            "version": "1.0.0",
            "description": "Test",
            "category": "test",
            "status": AgentStatus.PUBLISHED,
            "author": "user",
            "checksum": "sha256:abc",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "downloads": 100,
            "tags": ["tag1"],
            "regulatory_frameworks": ["CBAM"],
            "certification_status": [],
            "documentation_url": "https://docs.example.com",
            "repository_url": "https://github.com/example/repo",
            "license": "MIT",
        }

        response = AgentResponse(**data)
        assert response.name == "test-agent"
        assert response.downloads == 100

    def test_computed_fields(self):
        """Test computed fields have defaults."""
        response = AgentResponse(
            id=uuid4(),
            name="test",
            version="1.0.0",
            description="",
            category="test",
            status=AgentStatus.DRAFT,
            author="user",
            checksum="sha256:abc",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            downloads=0,
            tags=[],
            regulatory_frameworks=[],
            certification_status=[],
            documentation_url=None,
            repository_url=None,
            license="Apache-2.0",
        )

        assert response.version_count == 0
        assert response.latest_version is None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
