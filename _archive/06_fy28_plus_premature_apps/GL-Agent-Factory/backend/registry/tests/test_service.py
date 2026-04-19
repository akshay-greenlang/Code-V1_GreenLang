"""
Test Suite for Agent Registry Service

This module contains comprehensive tests for AgentRegistryService:
- Agent CRUD operations
- Version management
- Search functionality
- Publishing workflow
- Statistics

Run with: pytest backend/registry/tests/test_service.py -v
"""

import pytest
from datetime import datetime
from uuid import uuid4

from backend.registry.service import AgentRegistryService


class TestAgentRegistryServiceCreate:
    """Tests for agent creation operations."""

    @pytest.fixture
    def service(self):
        """Create service instance with in-memory storage."""
        return AgentRegistryService(session=None)

    @pytest.mark.asyncio
    async def test_create_agent_basic(self, service):
        """Test basic agent creation."""
        agent = await service.create_agent(
            name="test-agent",
            version="1.0.0",
            category="emissions",
            author="test-user",
        )

        assert agent is not None
        assert agent.name == "test-agent"
        assert agent.version == "1.0.0"
        assert agent.category == "emissions"
        assert agent.status == "draft"
        assert agent.author == "test-user"

    @pytest.mark.asyncio
    async def test_create_agent_with_metadata(self, service):
        """Test agent creation with full metadata."""
        agent = await service.create_agent(
            name="carbon-calculator",
            version="1.0.0",
            description="Calculate carbon emissions",
            category="regulatory",
            author="greenlang",
            pack_yaml={"key": "value"},
            tags=["carbon", "emissions"],
            regulatory_frameworks=["CBAM", "CSRD"],
            documentation_url="https://docs.example.com",
            repository_url="https://github.com/example/repo",
            license="MIT",
        )

        assert agent.description == "Calculate carbon emissions"
        assert agent.tags == ["carbon", "emissions"]
        assert agent.regulatory_frameworks == ["CBAM", "CSRD"]
        assert agent.documentation_url == "https://docs.example.com"
        assert agent.license == "MIT"

    @pytest.mark.asyncio
    async def test_create_agent_name_normalized(self, service):
        """Test that agent name is normalized."""
        agent = await service.create_agent(
            name="  Test-Agent  ",
            version="1.0.0",
            category="test",
            author="user",
        )

        assert agent.name == "test-agent"

    @pytest.mark.asyncio
    async def test_create_agent_duplicate_name_fails(self, service):
        """Test that duplicate names are rejected."""
        await service.create_agent(
            name="unique-agent",
            version="1.0.0",
            category="test",
            author="user",
        )

        with pytest.raises(ValueError, match="already exists"):
            await service.create_agent(
                name="unique-agent",
                version="2.0.0",
                category="other",
                author="other-user",
            )

    @pytest.mark.asyncio
    async def test_create_agent_generates_checksum(self, service):
        """Test that checksum is generated on creation."""
        agent = await service.create_agent(
            name="checksum-test",
            version="1.0.0",
            category="test",
            author="user",
        )

        assert agent.checksum
        assert agent.checksum.startswith("sha256:")

    @pytest.mark.asyncio
    async def test_create_agent_creates_initial_version(self, service):
        """Test that initial version is created."""
        agent = await service.create_agent(
            name="version-test",
            version="1.0.0",
            category="test",
            author="user",
        )

        versions = await service.list_versions(agent.id)
        assert len(versions) == 1
        assert versions[0].version == "1.0.0"
        assert versions[0].is_latest is True


class TestAgentRegistryServiceRead:
    """Tests for agent read operations."""

    @pytest.fixture
    async def service_with_agents(self):
        """Create service with pre-populated agents."""
        service = AgentRegistryService(session=None)

        await service.create_agent(
            name="agent-one",
            version="1.0.0",
            category="emissions",
            author="user1",
            tags=["carbon"],
        )
        await service.create_agent(
            name="agent-two",
            version="2.0.0",
            category="regulatory",
            author="user2",
            tags=["compliance"],
        )
        await service.create_agent(
            name="agent-three",
            version="1.5.0",
            category="emissions",
            author="user1",
            tags=["carbon", "scope3"],
        )

        return service

    @pytest.mark.asyncio
    async def test_get_agent_by_id(self, service_with_agents):
        """Test getting agent by ID."""
        agents, _ = await service_with_agents.list_agents(limit=1)
        agent_id = agents[0].id

        agent = await service_with_agents.get_agent(agent_id)
        assert agent is not None
        assert agent.id == agent_id

    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, service_with_agents):
        """Test getting non-existent agent."""
        agent = await service_with_agents.get_agent(uuid4())
        assert agent is None

    @pytest.mark.asyncio
    async def test_get_agent_by_name(self, service_with_agents):
        """Test getting agent by name."""
        agent = await service_with_agents.get_agent_by_name("agent-one")
        assert agent is not None
        assert agent.name == "agent-one"

    @pytest.mark.asyncio
    async def test_list_agents_all(self, service_with_agents):
        """Test listing all agents."""
        agents, total = await service_with_agents.list_agents()

        assert len(agents) == 3
        assert total == 3

    @pytest.mark.asyncio
    async def test_list_agents_filter_category(self, service_with_agents):
        """Test filtering by category."""
        agents, total = await service_with_agents.list_agents(category="emissions")

        assert len(agents) == 2
        assert total == 2
        assert all(a.category == "emissions" for a in agents)

    @pytest.mark.asyncio
    async def test_list_agents_filter_author(self, service_with_agents):
        """Test filtering by author."""
        agents, total = await service_with_agents.list_agents(author="user1")

        assert len(agents) == 2
        assert total == 2
        assert all(a.author == "user1" for a in agents)

    @pytest.mark.asyncio
    async def test_list_agents_pagination(self, service_with_agents):
        """Test pagination."""
        page1, total = await service_with_agents.list_agents(limit=2, offset=0)
        page2, _ = await service_with_agents.list_agents(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 1
        assert total == 3


class TestAgentRegistryServiceUpdate:
    """Tests for agent update operations."""

    @pytest.fixture
    async def service_with_agent(self):
        """Create service with one agent."""
        service = AgentRegistryService(session=None)
        agent = await service.create_agent(
            name="update-test",
            version="1.0.0",
            category="test",
            author="user",
            description="Original description",
        )
        return service, agent

    @pytest.mark.asyncio
    async def test_update_description(self, service_with_agent):
        """Test updating description."""
        service, agent = service_with_agent

        updated = await service.update_agent(
            agent.id,
            updates={"description": "New description"},
        )

        assert updated.description == "New description"

    @pytest.mark.asyncio
    async def test_update_tags(self, service_with_agent):
        """Test updating tags."""
        service, agent = service_with_agent

        updated = await service.update_agent(
            agent.id,
            updates={"tags": ["new", "tags"]},
        )

        assert updated.tags == ["new", "tags"]

    @pytest.mark.asyncio
    async def test_update_updates_timestamp(self, service_with_agent):
        """Test that update modifies updated_at."""
        service, agent = service_with_agent
        original_updated = agent.updated_at

        updated = await service.update_agent(
            agent.id,
            updates={"description": "Changed"},
        )

        assert updated.updated_at >= original_updated

    @pytest.mark.asyncio
    async def test_update_not_found(self, service_with_agent):
        """Test updating non-existent agent."""
        service, _ = service_with_agent

        result = await service.update_agent(
            uuid4(),
            updates={"description": "New"},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_update_recomputes_checksum(self, service_with_agent):
        """Test that checksum is recomputed on pack_yaml change."""
        service, agent = service_with_agent
        original_checksum = agent.checksum

        updated = await service.update_agent(
            agent.id,
            updates={"pack_yaml": {"new": "config"}},
        )

        assert updated.checksum != original_checksum


class TestAgentRegistryServiceDelete:
    """Tests for agent delete operations."""

    @pytest.fixture
    async def service_with_agents(self):
        """Create service with agents in different states."""
        service = AgentRegistryService(session=None)

        draft_agent = await service.create_agent(
            name="draft-agent",
            version="1.0.0",
            category="test",
            author="user",
        )

        published_agent = await service.create_agent(
            name="published-agent",
            version="1.0.0",
            category="test",
            author="user",
        )
        published_agent.status = "published"

        return service, draft_agent, published_agent

    @pytest.mark.asyncio
    async def test_delete_draft_agent(self, service_with_agents):
        """Test deleting draft agent."""
        service, draft_agent, _ = service_with_agents

        result = await service.delete_agent(draft_agent.id)

        assert result is True
        agent = await service.get_agent(draft_agent.id)
        assert agent is None

    @pytest.mark.asyncio
    async def test_delete_published_deprecates(self, service_with_agents):
        """Test that deleting published agent deprecates it."""
        service, _, published_agent = service_with_agents

        result = await service.delete_agent(published_agent.id)

        assert result is True
        agent = await service.get_agent(published_agent.id)
        assert agent is not None
        assert agent.status == "deprecated"

    @pytest.mark.asyncio
    async def test_delete_not_found(self, service_with_agents):
        """Test deleting non-existent agent."""
        service, _, _ = service_with_agents

        result = await service.delete_agent(uuid4())
        assert result is False


class TestAgentRegistryServiceSearch:
    """Tests for agent search operations."""

    @pytest.fixture
    async def service_with_searchable_agents(self):
        """Create service with searchable agents."""
        service = AgentRegistryService(session=None)

        await service.create_agent(
            name="carbon-emissions-calculator",
            version="1.0.0",
            description="Calculate carbon emissions for scope 1, 2, and 3",
            category="emissions",
            author="greenlang",
            tags=["carbon", "ghg", "scope1", "scope2", "scope3"],
        )
        await service.create_agent(
            name="cbam-compliance-checker",
            version="1.0.0",
            description="Check CBAM compliance for EU imports",
            category="regulatory",
            author="greenlang",
            tags=["cbam", "eu", "compliance"],
        )
        await service.create_agent(
            name="building-energy-monitor",
            version="1.0.0",
            description="Monitor building energy consumption",
            category="energy",
            author="user",
            tags=["energy", "building", "consumption"],
        )

        return service

    @pytest.mark.asyncio
    async def test_search_by_name(self, service_with_searchable_agents):
        """Test searching by agent name."""
        service = service_with_searchable_agents

        agents, total = await service.search_agents("carbon")

        assert len(agents) >= 1
        assert any("carbon" in a.name for a in agents)

    @pytest.mark.asyncio
    async def test_search_by_description(self, service_with_searchable_agents):
        """Test searching in description."""
        service = service_with_searchable_agents

        agents, total = await service.search_agents("scope")

        assert len(agents) >= 1

    @pytest.mark.asyncio
    async def test_search_by_tag(self, service_with_searchable_agents):
        """Test searching by tag."""
        service = service_with_searchable_agents

        agents, total = await service.search_agents("cbam")

        assert len(agents) >= 1

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self, service_with_searchable_agents):
        """Test searching with category filter."""
        service = service_with_searchable_agents

        agents, _ = await service.search_agents("", category="regulatory")

        assert all(a.category == "regulatory" for a in agents)

    @pytest.mark.asyncio
    async def test_search_no_results(self, service_with_searchable_agents):
        """Test search with no results."""
        service = service_with_searchable_agents

        agents, total = await service.search_agents("nonexistent-query-xyz")

        assert len(agents) == 0
        assert total == 0


class TestAgentRegistryServiceVersions:
    """Tests for version management operations."""

    @pytest.fixture
    async def service_with_agent(self):
        """Create service with one agent."""
        service = AgentRegistryService(session=None)
        agent = await service.create_agent(
            name="versioned-agent",
            version="1.0.0",
            category="test",
            author="user",
        )
        return service, agent

    @pytest.mark.asyncio
    async def test_create_version(self, service_with_agent):
        """Test creating a new version."""
        service, agent = service_with_agent

        version = await service.create_version(
            agent_id=agent.id,
            version="1.1.0",
            changelog="Added new features",
            breaking_changes=False,
        )

        assert version.version == "1.1.0"
        assert version.changelog == "Added new features"
        assert version.is_latest is True

    @pytest.mark.asyncio
    async def test_create_version_updates_latest_flag(self, service_with_agent):
        """Test that creating version updates latest flag."""
        service, agent = service_with_agent

        # Initial version should be latest
        versions = await service.list_versions(agent.id)
        assert versions[0].is_latest is True

        # Create new version
        await service.create_version(
            agent_id=agent.id,
            version="1.1.0",
            changelog="New version",
        )

        # Check latest flags
        versions = await service.list_versions(agent.id)
        latest_versions = [v for v in versions if v.is_latest]
        assert len(latest_versions) == 1
        assert latest_versions[0].version == "1.1.0"

    @pytest.mark.asyncio
    async def test_create_version_lower_fails(self, service_with_agent):
        """Test that creating lower version fails."""
        service, agent = service_with_agent

        # Create 2.0.0 first
        await service.create_version(
            agent_id=agent.id,
            version="2.0.0",
            changelog="Major update",
        )

        # Try to create 1.5.0 (lower than 2.0.0)
        with pytest.raises(ValueError, match="must be higher"):
            await service.create_version(
                agent_id=agent.id,
                version="1.5.0",
                changelog="Should fail",
            )

    @pytest.mark.asyncio
    async def test_create_version_duplicate_fails(self, service_with_agent):
        """Test that duplicate version fails."""
        service, agent = service_with_agent

        with pytest.raises(ValueError, match="already exists"):
            await service.create_version(
                agent_id=agent.id,
                version="1.0.0",  # Same as initial
                changelog="Duplicate",
            )

    @pytest.mark.asyncio
    async def test_list_versions(self, service_with_agent):
        """Test listing versions."""
        service, agent = service_with_agent

        await service.create_version(agent.id, "1.1.0", "v1.1")
        await service.create_version(agent.id, "1.2.0", "v1.2")

        versions = await service.list_versions(agent.id)

        assert len(versions) == 3
        # Should be ordered by created_at desc
        assert versions[0].version == "1.2.0"

    @pytest.mark.asyncio
    async def test_get_version_specific(self, service_with_agent):
        """Test getting specific version."""
        service, agent = service_with_agent

        await service.create_version(agent.id, "1.1.0", "v1.1")

        version = await service.get_version(agent.id, "1.0.0")

        assert version is not None
        assert version.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_version_latest(self, service_with_agent):
        """Test getting latest version."""
        service, agent = service_with_agent

        await service.create_version(agent.id, "1.1.0", "v1.1")
        await service.create_version(agent.id, "1.2.0", "v1.2")

        version = await service.get_version(agent.id, "latest")

        assert version is not None
        assert version.version == "1.2.0"
        assert version.is_latest is True


class TestAgentRegistryServicePublish:
    """Tests for publishing workflow."""

    @pytest.fixture
    async def service_with_unpublished_agent(self):
        """Create service with unpublished agent."""
        service = AgentRegistryService(session=None)
        agent = await service.create_agent(
            name="publish-test",
            version="1.0.0",
            category="test",
            author="user",
        )
        return service, agent

    @pytest.mark.asyncio
    async def test_publish_agent(self, service_with_unpublished_agent):
        """Test publishing an agent."""
        service, agent = service_with_unpublished_agent

        version = await service.publish_agent(
            agent_id=agent.id,
            version="1.0.0",
            release_notes="Initial release",
        )

        assert version.published_at is not None
        assert version.release_notes == "Initial release"

        # Check agent status
        updated_agent = await service.get_agent(agent.id)
        assert updated_agent.status == "published"

    @pytest.mark.asyncio
    async def test_publish_with_certifications(self, service_with_unpublished_agent):
        """Test publishing with certifications."""
        service, agent = service_with_unpublished_agent

        await service.publish_agent(
            agent_id=agent.id,
            version="1.0.0",
            certifications=["CBAM", "CSRD"],
            user_id="auditor",
        )

        updated_agent = await service.get_agent(agent.id)
        assert len(updated_agent.certification_status) == 2

    @pytest.mark.asyncio
    async def test_publish_already_published_fails(self, service_with_unpublished_agent):
        """Test that re-publishing fails."""
        service, agent = service_with_unpublished_agent

        await service.publish_agent(agent.id, "1.0.0")

        with pytest.raises(ValueError, match="already published"):
            await service.publish_agent(agent.id, "1.0.0")

    @pytest.mark.asyncio
    async def test_deprecate_agent(self, service_with_unpublished_agent):
        """Test deprecating an agent."""
        service, agent = service_with_unpublished_agent

        result = await service.deprecate_agent(agent.id)

        assert result is not None
        assert result.status == "deprecated"


class TestAgentRegistryServiceDownloads:
    """Tests for download operations."""

    @pytest.fixture
    async def service_with_published_agent(self):
        """Create service with published agent."""
        service = AgentRegistryService(session=None)
        agent = await service.create_agent(
            name="download-test",
            version="1.0.0",
            category="test",
            author="user",
        )
        await service.publish_agent(agent.id, "1.0.0")
        return service, agent

    @pytest.mark.asyncio
    async def test_get_download_published(self, service_with_published_agent):
        """Test getting download info for published agent."""
        service, agent = service_with_published_agent

        download = await service.get_download(agent.id)

        assert download is not None
        assert download["version"] == "1.0.0"
        assert download["artifact_path"] is not None

    @pytest.mark.asyncio
    async def test_get_download_unpublished_fails(self):
        """Test that unpublished agents can't be downloaded."""
        service = AgentRegistryService(session=None)
        agent = await service.create_agent(
            name="unpublished",
            version="1.0.0",
            category="test",
            author="user",
        )

        download = await service.get_download(agent.id)

        assert download is None

    @pytest.mark.asyncio
    async def test_increment_download(self, service_with_published_agent):
        """Test incrementing download counter."""
        service, agent = service_with_published_agent
        original_downloads = agent.downloads

        await service.increment_download(agent.id, "1.0.0")

        updated = await service.get_agent(agent.id)
        assert updated.downloads == original_downloads + 1


class TestAgentRegistryServiceStatistics:
    """Tests for statistics operations."""

    @pytest.fixture
    async def service_with_mixed_agents(self):
        """Create service with agents in various states."""
        service = AgentRegistryService(session=None)

        # Create agents in different categories and states
        agent1 = await service.create_agent(
            name="stats-agent-1",
            version="1.0.0",
            category="emissions",
            author="user",
        )
        await service.publish_agent(agent1.id, "1.0.0")

        agent2 = await service.create_agent(
            name="stats-agent-2",
            version="1.0.0",
            category="emissions",
            author="user",
        )

        agent3 = await service.create_agent(
            name="stats-agent-3",
            version="1.0.0",
            category="regulatory",
            author="user",
        )
        await service.publish_agent(agent3.id, "1.0.0")
        await service.deprecate_agent(agent3.id)

        return service

    @pytest.mark.asyncio
    async def test_get_statistics(self, service_with_mixed_agents):
        """Test getting registry statistics."""
        service = service_with_mixed_agents

        stats = await service.get_statistics()

        assert stats["total_agents"] == 3
        assert stats["total_versions"] >= 3


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
