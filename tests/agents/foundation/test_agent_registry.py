# -*- coding: utf-8 -*-
"""
Tests for GL-FOUND-X-007: Versioned Agent Registry

Comprehensive tests covering:
    - Agent Registration with metadata validation
    - Agent Discovery with filtering
    - Version Management with semantic versioning
    - Capability Matching
    - Dependency Resolution with cycle detection
    - Hot Reload Support
    - Health Status Tracking
    - Export/Import functionality

Test Coverage Target: 85%+
"""

import pytest
import threading
import time
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent

# Direct import to bypass foundation __init__.py which has broken imports
# This imports the agent_registry module directly from its file path
_spec = importlib.util.spec_from_file_location(
    "agent_registry",
    Path(__file__).parent.parent.parent.parent / "greenlang" / "agents" / "foundation" / "agent_registry.py"
)
_agent_registry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_agent_registry)

# Extract all needed exports from the module
VersionedAgentRegistry = _agent_registry.VersionedAgentRegistry
AgentMetadataEntry = _agent_registry.AgentMetadataEntry
AgentCapability = _agent_registry.AgentCapability
AgentVariant = _agent_registry.AgentVariant
AgentDependency = _agent_registry.AgentDependency
AgentLayer = _agent_registry.AgentLayer
SectorClassification = _agent_registry.SectorClassification
AgentHealthStatus = _agent_registry.AgentHealthStatus
CapabilityCategory = _agent_registry.CapabilityCategory
SemanticVersion = _agent_registry.SemanticVersion
RegistryQueryInput = _agent_registry.RegistryQueryInput
RegistryQueryOutput = _agent_registry.RegistryQueryOutput
DependencyResolutionInput = _agent_registry.DependencyResolutionInput
DependencyResolutionOutput = _agent_registry.DependencyResolutionOutput
create_agent_registry = _agent_registry.create_agent_registry
create_agent_metadata = _agent_registry.create_agent_metadata


# =============================================================================
# Test Agent Implementations
# =============================================================================

class MockCalculatorAgent(BaseAgent):
    """Mock agent for testing - always succeeds."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        return AgentResult(
            success=True,
            data={"result": "calculated", "input_keys": list(input_data.keys())}
        )


class MockValidatorAgent(BaseAgent):
    """Mock validation agent for testing."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        return AgentResult(
            success=True,
            data={"validated": True, "errors": []}
        )


class FailingAgent(BaseAgent):
    """Agent that fails on instantiation."""

    def __init__(self, config=None):
        raise RuntimeError("Intentional failure for testing")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        return AgentResult(success=False, error="Should not reach here")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return VersionedAgentRegistry()


@pytest.fixture
def sample_metadata():
    """Create sample agent metadata."""
    return AgentMetadataEntry(
        agent_id="GL-MRV-X-001",
        name="Emissions Calculator",
        description="Calculates GHG emissions from activity data",
        version="1.0.0",
        layer=AgentLayer.MRV,
        sectors=[SectorClassification.ENERGY, SectorClassification.INDUSTRIALS],
        capabilities=[
            AgentCapability(
                name="emissions_calculation",
                category=CapabilityCategory.CALCULATION,
                description="Calculate emissions from activity data",
                input_types=["activity_data", "emission_factors"],
                output_types=["emissions_result"]
            ),
            AgentCapability(
                name="scope1_calculation",
                category=CapabilityCategory.CALCULATION,
                description="Calculate Scope 1 emissions",
                input_types=["fuel_consumption"],
                output_types=["scope1_emissions"]
            )
        ],
        variants=[
            AgentVariant(variant_type="geography", variant_value="US"),
            AgentVariant(variant_type="protocol", variant_value="GHG_Protocol")
        ],
        tags=["emissions", "scope1", "calculation"],
        author="GreenLang Team"
    )


@pytest.fixture
def sample_metadata_v2():
    """Create sample agent metadata version 2.0.0."""
    return AgentMetadataEntry(
        agent_id="GL-MRV-X-001",
        name="Emissions Calculator",
        description="Calculates GHG emissions from activity data - Enhanced",
        version="2.0.0",
        layer=AgentLayer.MRV,
        sectors=[SectorClassification.ENERGY, SectorClassification.INDUSTRIALS],
        capabilities=[
            AgentCapability(
                name="emissions_calculation",
                category=CapabilityCategory.CALCULATION,
                description="Calculate emissions from activity data",
                input_types=["activity_data", "emission_factors"],
                output_types=["emissions_result"]
            )
        ],
        tags=["emissions", "scope1", "calculation", "v2"]
    )


@pytest.fixture
def dependent_metadata():
    """Create agent metadata with dependencies."""
    return AgentMetadataEntry(
        agent_id="GL-MRV-X-002",
        name="Emissions Aggregator",
        description="Aggregates emissions from multiple sources",
        version="1.0.0",
        layer=AgentLayer.MRV,
        capabilities=[
            AgentCapability(
                name="emissions_aggregation",
                category=CapabilityCategory.AGGREGATION,
                description="Aggregate emissions data",
                input_types=["emissions_result"],
                output_types=["aggregated_emissions"]
            )
        ],
        dependencies=[
            AgentDependency(
                agent_id="GL-MRV-X-001",
                version_constraint=">=1.0.0",
                reason="Needs emissions calculator for input"
            )
        ]
    )


# =============================================================================
# SemanticVersion Tests
# =============================================================================

class TestSemanticVersion:
    """Tests for SemanticVersion parsing and comparison."""

    def test_parse_simple_version(self):
        """Test parsing simple version string."""
        version = SemanticVersion.parse("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
        assert version.build is None

    def test_parse_prerelease_version(self):
        """Test parsing version with prerelease."""
        version = SemanticVersion.parse("1.0.0-alpha.1")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.prerelease == "alpha.1"

    def test_parse_build_metadata(self):
        """Test parsing version with build metadata."""
        version = SemanticVersion.parse("1.0.0+build.123")
        assert version.build == "build.123"

    def test_parse_full_version(self):
        """Test parsing complete version string."""
        version = SemanticVersion.parse("2.1.3-beta.2+build.456")
        assert version.major == 2
        assert version.minor == 1
        assert version.patch == 3
        assert version.prerelease == "beta.2"
        assert version.build == "build.456"

    def test_parse_invalid_version(self):
        """Test parsing invalid version string."""
        with pytest.raises(ValueError, match="Invalid semantic version"):
            SemanticVersion.parse("invalid")

    def test_version_comparison(self):
        """Test version comparison."""
        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("2.0.0")
        v3 = SemanticVersion.parse("1.1.0")
        v4 = SemanticVersion.parse("1.0.1")

        assert v1 < v2
        assert v1 < v3
        assert v1 < v4
        assert v3 < v2
        assert v4 < v3

    def test_prerelease_comparison(self):
        """Test that prerelease is less than release."""
        release = SemanticVersion.parse("1.0.0")
        prerelease = SemanticVersion.parse("1.0.0-alpha")

        assert prerelease < release

    def test_version_to_string(self):
        """Test converting version back to string."""
        version_str = "1.2.3-alpha+build"
        version = SemanticVersion.parse(version_str)
        assert str(version) == version_str

    def test_version_compatibility(self):
        """Test version compatibility check."""
        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("1.5.0")
        v3 = SemanticVersion.parse("2.0.0")

        assert v1.is_compatible_with(v2)
        assert not v1.is_compatible_with(v3)


# =============================================================================
# Agent Registration Tests
# =============================================================================

class TestAgentRegistration:
    """Tests for agent registration functionality."""

    def test_register_agent_success(self, registry, sample_metadata):
        """Test successful agent registration."""
        result = registry.register_agent(sample_metadata)

        assert result is True
        assert registry.get_agent("GL-MRV-X-001") is not None

    def test_register_agent_with_class(self, registry, sample_metadata):
        """Test registering agent with class."""
        result = registry.register_agent(sample_metadata, MockCalculatorAgent)

        assert result is True
        agent_class = registry.get_agent_class("GL-MRV-X-001")
        assert agent_class is MockCalculatorAgent

    def test_register_multiple_versions(self, registry, sample_metadata, sample_metadata_v2):
        """Test registering multiple versions of same agent."""
        registry.register_agent(sample_metadata)
        registry.register_agent(sample_metadata_v2)

        versions = registry.list_versions("GL-MRV-X-001")
        assert len(versions) == 2
        assert "2.0.0" in versions
        assert "1.0.0" in versions

    def test_get_latest_version(self, registry, sample_metadata, sample_metadata_v2):
        """Test getting latest version when no version specified."""
        registry.register_agent(sample_metadata)
        registry.register_agent(sample_metadata_v2)

        latest = registry.get_agent("GL-MRV-X-001")
        assert latest.version == "2.0.0"

    def test_get_specific_version(self, registry, sample_metadata, sample_metadata_v2):
        """Test getting specific version."""
        registry.register_agent(sample_metadata)
        registry.register_agent(sample_metadata_v2)

        v1 = registry.get_agent("GL-MRV-X-001", "1.0.0")
        assert v1.version == "1.0.0"

    def test_overwrite_existing_version(self, registry, sample_metadata):
        """Test overwriting existing version logs warning."""
        registry.register_agent(sample_metadata)

        # Modify and re-register
        sample_metadata.description = "Updated description"
        result = registry.register_agent(sample_metadata)

        assert result is True
        agent = registry.get_agent("GL-MRV-X-001", "1.0.0")
        assert agent.description == "Updated description"

    def test_invalid_agent_id_format(self, registry):
        """Test registration with invalid agent ID format."""
        with pytest.raises(ValueError, match="Invalid agent ID format"):
            AgentMetadataEntry(
                agent_id="invalid-id",
                name="Test",
                description="Test",
                version="1.0.0",
                layer=AgentLayer.MRV
            )

    def test_unregister_specific_version(self, registry, sample_metadata, sample_metadata_v2):
        """Test unregistering specific version."""
        registry.register_agent(sample_metadata)
        registry.register_agent(sample_metadata_v2)

        result = registry.unregister_agent("GL-MRV-X-001", "1.0.0")

        assert result is True
        versions = registry.list_versions("GL-MRV-X-001")
        assert len(versions) == 1
        assert "2.0.0" in versions

    def test_unregister_all_versions(self, registry, sample_metadata, sample_metadata_v2):
        """Test unregistering all versions."""
        registry.register_agent(sample_metadata)
        registry.register_agent(sample_metadata_v2)

        result = registry.unregister_agent("GL-MRV-X-001")

        assert result is True
        assert registry.get_agent("GL-MRV-X-001") is None

    def test_unregister_nonexistent_agent(self, registry):
        """Test unregistering agent that doesn't exist."""
        result = registry.unregister_agent("GL-MRV-X-999")
        assert result is False


# =============================================================================
# Agent Discovery Tests
# =============================================================================

class TestAgentDiscovery:
    """Tests for agent discovery and querying."""

    def test_query_by_layer(self, registry, sample_metadata):
        """Test querying agents by layer."""
        registry.register_agent(sample_metadata)

        # Add agent in different layer
        foundation_agent = create_agent_metadata(
            agent_id="GL-FOUND-X-001",
            name="Orchestrator",
            description="Test orchestrator",
            version="1.0.0",
            layer=AgentLayer.FOUNDATION
        )
        registry.register_agent(foundation_agent)

        result = registry.query_agents(RegistryQueryInput(layer=AgentLayer.MRV))

        assert result.total_count == 1
        assert result.agents[0].agent_id == "GL-MRV-X-001"

    def test_query_by_sector(self, registry, sample_metadata):
        """Test querying agents by sector."""
        registry.register_agent(sample_metadata)

        result = registry.query_agents(
            RegistryQueryInput(sector=SectorClassification.ENERGY)
        )

        assert result.total_count == 1
        assert SectorClassification.ENERGY in result.agents[0].sectors

    def test_query_by_capability(self, registry, sample_metadata):
        """Test querying agents by capability."""
        registry.register_agent(sample_metadata)

        result = registry.query_agents(
            RegistryQueryInput(capability="emissions_calculation")
        )

        assert result.total_count == 1
        assert result.agents[0].has_capability("emissions_calculation")

    def test_query_by_capability_category(self, registry, sample_metadata):
        """Test querying agents by capability category."""
        registry.register_agent(sample_metadata)

        # Add agent with different capability category
        validation_agent = create_agent_metadata(
            agent_id="GL-DATA-X-001",
            name="Validator",
            description="Data validator",
            version="1.0.0",
            layer=AgentLayer.DATA,
            capabilities=[{
                "name": "data_validation",
                "category": "validation",
                "description": "Validate data"
            }]
        )
        registry.register_agent(validation_agent)

        result = registry.query_agents(
            RegistryQueryInput(capability_category=CapabilityCategory.CALCULATION)
        )

        assert result.total_count == 1
        assert result.agents[0].agent_id == "GL-MRV-X-001"

    def test_query_by_tags(self, registry, sample_metadata):
        """Test querying agents by tags."""
        registry.register_agent(sample_metadata)

        result = registry.query_agents(
            RegistryQueryInput(tags=["emissions", "scope1"])
        )

        assert result.total_count == 1

    def test_query_by_variant(self, registry, sample_metadata):
        """Test querying agents by variant."""
        registry.register_agent(sample_metadata)

        result = registry.query_agents(
            RegistryQueryInput(variant_type="geography", variant_value="US")
        )

        assert result.total_count == 1

    def test_query_by_health_status(self, registry, sample_metadata):
        """Test querying agents by health status."""
        sample_metadata.health_status = AgentHealthStatus.HEALTHY
        registry.register_agent(sample_metadata)

        result = registry.query_agents(
            RegistryQueryInput(health_status=AgentHealthStatus.HEALTHY)
        )

        assert result.total_count == 1

    def test_query_with_text_search(self, registry, sample_metadata):
        """Test querying with text search."""
        registry.register_agent(sample_metadata)

        result = registry.query_agents(
            RegistryQueryInput(search_text="GHG emissions")
        )

        assert result.total_count == 1

    def test_query_with_version_constraints(self, registry, sample_metadata, sample_metadata_v2):
        """Test querying with version constraints."""
        registry.register_agent(sample_metadata)
        registry.register_agent(sample_metadata_v2)

        # Min version
        result = registry.query_agents(
            RegistryQueryInput(min_version="1.5.0")
        )
        assert result.total_count == 1
        assert result.agents[0].version == "2.0.0"

        # Max version
        result = registry.query_agents(
            RegistryQueryInput(max_version="1.5.0")
        )
        assert result.total_count == 1
        assert result.agents[0].version == "1.0.0"

    def test_query_pagination(self, registry):
        """Test query pagination."""
        # Register multiple agents
        for i in range(5):
            metadata = create_agent_metadata(
                agent_id=f"GL-MRV-X-00{i+1}",
                name=f"Agent {i+1}",
                description=f"Test agent {i+1}",
                version="1.0.0",
                layer=AgentLayer.MRV
            )
            registry.register_agent(metadata)

        # Page 1
        result = registry.query_agents(
            RegistryQueryInput(limit=2, offset=0)
        )
        assert len(result.agents) == 2
        assert result.total_count == 5

        # Page 2
        result = registry.query_agents(
            RegistryQueryInput(limit=2, offset=2)
        )
        assert len(result.agents) == 2

    def test_query_combined_filters(self, registry, sample_metadata):
        """Test querying with multiple combined filters."""
        registry.register_agent(sample_metadata)

        result = registry.query_agents(
            RegistryQueryInput(
                layer=AgentLayer.MRV,
                sector=SectorClassification.ENERGY,
                capability="emissions_calculation",
                tags=["emissions"]
            )
        )

        assert result.total_count == 1

    def test_query_returns_provenance_hash(self, registry, sample_metadata):
        """Test that query returns provenance hash."""
        registry.register_agent(sample_metadata)

        result = registry.query_agents(RegistryQueryInput())

        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 16


# =============================================================================
# Capability Matching Tests
# =============================================================================

class TestCapabilityMatching:
    """Tests for capability matching functionality."""

    def test_find_agents_by_single_capability(self, registry, sample_metadata):
        """Test finding agents by single capability."""
        registry.register_agent(sample_metadata)

        required = [
            AgentCapability(
                name="emissions_calculation",
                category=CapabilityCategory.CALCULATION,
                description="Required capability",
                input_types=["activity_data"]
            )
        ]

        result = registry.find_agents_by_capabilities(required)

        assert len(result) >= 1
        assert result[0].agent_id == "GL-MRV-X-001"

    def test_find_agents_by_multiple_capabilities(self, registry, sample_metadata):
        """Test finding agents with multiple capabilities."""
        registry.register_agent(sample_metadata)

        required = [
            AgentCapability(
                name="emissions_calculation",
                category=CapabilityCategory.CALCULATION,
                description="Required capability",
                input_types=["activity_data"]
            ),
            AgentCapability(
                name="scope1_calculation",
                category=CapabilityCategory.CALCULATION,
                description="Required capability",
                input_types=["fuel_consumption"]
            )
        ]

        result = registry.find_agents_by_capabilities(required)

        assert len(result) == 1

    def test_capability_input_type_matching(self, registry, sample_metadata):
        """Test that capability input types are matched."""
        registry.register_agent(sample_metadata)

        # This should NOT match - wrong input type
        required = [
            AgentCapability(
                name="emissions_calculation",
                category=CapabilityCategory.CALCULATION,
                description="Required capability",
                input_types=["wrong_type"]
            )
        ]

        result = registry.find_agents_by_capabilities(required)

        assert len(result) == 0


# =============================================================================
# Dependency Resolution Tests
# =============================================================================

class TestDependencyResolution:
    """Tests for dependency resolution functionality."""

    def test_simple_dependency_resolution(self, registry, sample_metadata, dependent_metadata):
        """Test resolving simple dependencies."""
        registry.register_agent(sample_metadata)
        registry.register_agent(dependent_metadata)

        result = registry.resolve_dependencies(
            DependencyResolutionInput(agent_ids=["GL-MRV-X-002"])
        )

        assert result.success is True
        assert "GL-MRV-X-001" in result.resolved_order
        assert "GL-MRV-X-002" in result.resolved_order
        # Dependency should come before dependent
        assert result.resolved_order.index("GL-MRV-X-001") < result.resolved_order.index("GL-MRV-X-002")

    def test_missing_dependency_detection(self, registry, dependent_metadata):
        """Test detection of missing dependencies."""
        registry.register_agent(dependent_metadata)

        result = registry.resolve_dependencies(
            DependencyResolutionInput(agent_ids=["GL-MRV-X-002"])
        )

        assert result.success is False
        assert "GL-MRV-X-001" in result.missing_dependencies

    def test_circular_dependency_detection(self, registry):
        """Test detection of circular dependencies."""
        # Create circular dependency: A -> B -> C -> A
        agent_a = create_agent_metadata(
            agent_id="GL-MRV-X-001",
            name="Agent A",
            description="Test agent A",
            version="1.0.0",
            layer=AgentLayer.MRV,
            dependencies=[{"agent_id": "GL-MRV-X-003"}]
        )
        agent_b = create_agent_metadata(
            agent_id="GL-MRV-X-002",
            name="Agent B",
            description="Test agent B",
            version="1.0.0",
            layer=AgentLayer.MRV,
            dependencies=[{"agent_id": "GL-MRV-X-001"}]
        )
        agent_c = create_agent_metadata(
            agent_id="GL-MRV-X-003",
            name="Agent C",
            description="Test agent C",
            version="1.0.0",
            layer=AgentLayer.MRV,
            dependencies=[{"agent_id": "GL-MRV-X-002"}]
        )

        registry.register_agent(agent_a)
        registry.register_agent(agent_b)
        registry.register_agent(agent_c)

        result = registry.resolve_dependencies(
            DependencyResolutionInput(agent_ids=["GL-MRV-X-001"])
        )

        assert result.success is False
        assert len(result.circular_dependencies) > 0

    def test_optional_dependencies(self, registry, sample_metadata):
        """Test handling of optional dependencies."""
        agent_with_optional = create_agent_metadata(
            agent_id="GL-MRV-X-003",
            name="Agent with Optional",
            description="Has optional dependency",
            version="1.0.0",
            layer=AgentLayer.MRV,
            dependencies=[
                {"agent_id": "GL-MRV-X-001", "optional": False},
                {"agent_id": "GL-MRV-X-999", "optional": True}
            ]
        )

        registry.register_agent(sample_metadata)
        registry.register_agent(agent_with_optional)

        # Without optional
        result = registry.resolve_dependencies(
            DependencyResolutionInput(
                agent_ids=["GL-MRV-X-003"],
                include_optional=False
            )
        )
        assert result.success is True
        assert "GL-MRV-X-999" not in result.dependency_graph.get("GL-MRV-X-003", [])

        # With optional (will have missing)
        result = registry.resolve_dependencies(
            DependencyResolutionInput(
                agent_ids=["GL-MRV-X-003"],
                include_optional=True,
                fail_on_missing=False
            )
        )
        assert "GL-MRV-X-999" in result.missing_dependencies

    def test_version_constraint_satisfaction(self):
        """Test version constraint checking."""
        dep = AgentDependency(
            agent_id="test",
            version_constraint=">=1.0.0"
        )

        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("2.0.0")
        v0 = SemanticVersion.parse("0.9.0")

        assert dep.version_satisfies(v1) is True
        assert dep.version_satisfies(v2) is True
        assert dep.version_satisfies(v0) is False

    def test_caret_version_constraint(self):
        """Test caret (^) version constraint."""
        dep = AgentDependency(
            agent_id="test",
            version_constraint="^1.2.0"
        )

        assert dep.version_satisfies(SemanticVersion.parse("1.2.0")) is True
        assert dep.version_satisfies(SemanticVersion.parse("1.5.0")) is True
        assert dep.version_satisfies(SemanticVersion.parse("2.0.0")) is False
        assert dep.version_satisfies(SemanticVersion.parse("1.1.0")) is False


# =============================================================================
# Health Tracking Tests
# =============================================================================

class TestHealthTracking:
    """Tests for health status tracking."""

    def test_initial_health_unknown(self, registry, sample_metadata):
        """Test that initial health status is unknown."""
        registry.register_agent(sample_metadata)

        agent = registry.get_agent("GL-MRV-X-001")
        assert agent.health_status == AgentHealthStatus.UNKNOWN

    def test_check_healthy_agent(self, registry, sample_metadata):
        """Test health check for healthy agent."""
        registry.register_agent(sample_metadata, MockCalculatorAgent)

        status = registry.check_agent_health("GL-MRV-X-001")

        assert status == AgentHealthStatus.HEALTHY

    def test_check_unhealthy_agent(self, registry, sample_metadata):
        """Test health check for unhealthy agent."""
        registry.register_agent(sample_metadata, FailingAgent)

        status = registry.check_agent_health("GL-MRV-X-001")

        assert status == AgentHealthStatus.UNHEALTHY

    def test_manual_health_set(self, registry, sample_metadata):
        """Test manually setting health status."""
        registry.register_agent(sample_metadata)

        result = registry.set_agent_health("GL-MRV-X-001", AgentHealthStatus.DEGRADED)

        assert result is True
        agent = registry.get_agent("GL-MRV-X-001")
        assert agent.health_status == AgentHealthStatus.DEGRADED

    def test_health_check_nonexistent_agent(self, registry):
        """Test health check for nonexistent agent."""
        status = registry.check_agent_health("GL-MRV-X-999")
        assert status is None


# =============================================================================
# Hot Reload Tests
# =============================================================================

class TestHotReload:
    """Tests for hot reload functionality."""

    def test_register_reload_callback(self, registry, sample_metadata):
        """Test registering reload callback."""
        callback_called = []

        def on_reload(agent_id, version):
            callback_called.append((agent_id, version))

        registry.register_reload_callback(on_reload)
        registry.register_agent(sample_metadata)

        assert len(callback_called) == 1
        assert callback_called[0] == ("GL-MRV-X-001", "1.0.0")

    def test_unregister_reload_callback(self, registry, sample_metadata):
        """Test unregistering reload callback."""
        callback_called = []

        def on_reload(agent_id, version):
            callback_called.append((agent_id, version))

        registry.register_reload_callback(on_reload)
        registry.unregister_reload_callback(on_reload)
        registry.register_agent(sample_metadata)

        assert len(callback_called) == 0

    def test_hot_reload_agent(self, registry, sample_metadata, sample_metadata_v2):
        """Test hot-reloading an agent."""
        registry.register_agent(sample_metadata)

        result = registry.hot_reload_agent(
            "GL-MRV-X-001",
            sample_metadata_v2,
            MockCalculatorAgent
        )

        assert result is True
        versions = registry.list_versions("GL-MRV-X-001")
        assert "2.0.0" in versions


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_registration(self, registry):
        """Test concurrent agent registration."""
        errors = []

        def register_agent(agent_num):
            try:
                metadata = create_agent_metadata(
                    agent_id=f"GL-MRV-X-{agent_num:03d}",
                    name=f"Agent {agent_num}",
                    description=f"Test agent {agent_num}",
                    version="1.0.0",
                    layer=AgentLayer.MRV
                )
                registry.register_agent(metadata)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_agent, args=(i,))
            for i in range(1, 11)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.get_all_agent_ids()) == 10

    def test_concurrent_query(self, registry, sample_metadata):
        """Test concurrent querying."""
        registry.register_agent(sample_metadata)

        results = []
        errors = []

        def query_registry():
            try:
                result = registry.query_agents(RegistryQueryInput())
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=query_registry)
            for _ in range(10)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r.total_count == 1 for r in results)


# =============================================================================
# Export/Import Tests
# =============================================================================

class TestExportImport:
    """Tests for export/import functionality."""

    def test_export_registry(self, registry, sample_metadata):
        """Test exporting registry."""
        registry.register_agent(sample_metadata)

        export_data = registry.export_registry()

        assert "version" in export_data
        assert "exported_at" in export_data
        assert "agents" in export_data
        assert "GL-MRV-X-001" in export_data["agents"]

    def test_import_registry_merge(self, registry, sample_metadata, sample_metadata_v2):
        """Test importing with merge."""
        registry.register_agent(sample_metadata)

        export_data = {
            "version": "1.0",
            "agents": {
                "GL-MRV-X-001": {
                    "2.0.0": sample_metadata_v2.model_dump()
                }
            }
        }

        count = registry.import_registry(export_data, merge=True)

        assert count == 1
        versions = registry.list_versions("GL-MRV-X-001")
        assert len(versions) == 2

    def test_import_registry_replace(self, registry, sample_metadata, sample_metadata_v2):
        """Test importing with replace."""
        registry.register_agent(sample_metadata)

        export_data = {
            "version": "1.0",
            "agents": {
                "GL-MRV-X-001": {
                    "2.0.0": sample_metadata_v2.model_dump()
                }
            }
        }

        count = registry.import_registry(export_data, merge=False)

        assert count == 1
        versions = registry.list_versions("GL-MRV-X-001")
        assert len(versions) == 1
        assert "2.0.0" in versions


# =============================================================================
# Execute Method Tests
# =============================================================================

class TestExecuteMethod:
    """Tests for the execute method (BaseAgent interface)."""

    def test_execute_register_operation(self, registry, sample_metadata):
        """Test execute with register operation."""
        result = registry.run({
            "operation": "register",
            "metadata": sample_metadata.model_dump()
        })

        assert result.success is True
        assert result.data["registered"] is True

    def test_execute_query_operation(self, registry, sample_metadata):
        """Test execute with query operation."""
        registry.register_agent(sample_metadata)

        result = registry.run({
            "operation": "query",
            "query": {"layer": "mrv"}
        })

        assert result.success is True
        assert result.data["total_count"] == 1

    def test_execute_get_agent_operation(self, registry, sample_metadata):
        """Test execute with get_agent operation."""
        registry.register_agent(sample_metadata)

        result = registry.run({
            "operation": "get_agent",
            "agent_id": "GL-MRV-X-001"
        })

        assert result.success is True
        assert result.data["metadata"]["agent_id"] == "GL-MRV-X-001"

    def test_execute_resolve_dependencies_operation(self, registry, sample_metadata, dependent_metadata):
        """Test execute with resolve_dependencies operation."""
        registry.register_agent(sample_metadata)
        registry.register_agent(dependent_metadata)

        result = registry.run({
            "operation": "resolve_dependencies",
            "resolution": {"agent_ids": ["GL-MRV-X-002"]}
        })

        assert result.success is True
        assert "resolved_order" in result.data

    def test_execute_list_versions_operation(self, registry, sample_metadata, sample_metadata_v2):
        """Test execute with list_versions operation."""
        registry.register_agent(sample_metadata)
        registry.register_agent(sample_metadata_v2)

        result = registry.run({
            "operation": "list_versions",
            "agent_id": "GL-MRV-X-001"
        })

        assert result.success is True
        assert len(result.data["versions"]) == 2

    def test_execute_get_statistics_operation(self, registry, sample_metadata):
        """Test execute with get_statistics operation."""
        registry.register_agent(sample_metadata)

        result = registry.run({
            "operation": "get_statistics"
        })

        assert result.success is True
        assert "total_agents" in result.data
        assert result.data["total_agents"] == 1

    def test_execute_unknown_operation(self, registry):
        """Test execute with unknown operation."""
        result = registry.run({
            "operation": "unknown_operation"
        })

        assert result.success is False
        assert "Unknown operation" in result.error


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_agent_registry(self):
        """Test creating registry with factory function."""
        registry = create_agent_registry()

        assert registry is not None
        assert registry.AGENT_ID == "GL-FOUND-X-007"

    def test_create_agent_metadata(self):
        """Test creating metadata with factory function."""
        metadata = create_agent_metadata(
            agent_id="GL-MRV-X-001",
            name="Test Agent",
            description="Test description",
            version="1.0.0",
            layer=AgentLayer.MRV,
            capabilities=[{
                "name": "test_cap",
                "category": "calculation",
                "description": "Test capability"
            }],
            sectors=[SectorClassification.ENERGY],
            variants=[{"variant_type": "geo", "variant_value": "US"}],
            tags=["test"]
        )

        assert metadata.agent_id == "GL-MRV-X-001"
        assert len(metadata.capabilities) == 1
        assert len(metadata.variants) == 1


# =============================================================================
# Statistics Tests
# =============================================================================

class TestStatistics:
    """Tests for registry statistics."""

    def test_statistics_after_registration(self, registry, sample_metadata):
        """Test statistics after agent registration."""
        registry.register_agent(sample_metadata)

        stats = registry.get_statistics()

        assert stats["total_agents"] == 1
        assert stats["total_versions"] == 1
        assert stats["registration_count"] == 1

    def test_statistics_by_layer(self, registry, sample_metadata):
        """Test statistics by layer."""
        registry.register_agent(sample_metadata)

        stats = registry.get_statistics()

        assert stats["agents_by_layer"]["mrv"] == 1

    def test_statistics_query_count(self, registry, sample_metadata):
        """Test query count in statistics."""
        registry.register_agent(sample_metadata)
        registry.query_agents(RegistryQueryInput())
        registry.query_agents(RegistryQueryInput())

        stats = registry.get_statistics()

        assert stats["query_count"] == 2


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_get_nonexistent_agent(self, registry):
        """Test getting agent that doesn't exist."""
        agent = registry.get_agent("GL-MRV-X-999")
        assert agent is None

    def test_list_versions_nonexistent_agent(self, registry):
        """Test listing versions for nonexistent agent."""
        versions = registry.list_versions("GL-MRV-X-999")
        assert versions == []

    def test_get_agent_class_nonexistent(self, registry):
        """Test getting class for nonexistent agent."""
        agent_class = registry.get_agent_class("GL-MRV-X-999")
        assert agent_class is None

    def test_empty_query_returns_all(self, registry, sample_metadata):
        """Test that empty query returns all agents."""
        registry.register_agent(sample_metadata)

        result = registry.query_agents(RegistryQueryInput())

        assert result.total_count >= 1

    def test_provenance_hash_consistency(self, registry, sample_metadata):
        """Test that provenance hash is consistent."""
        registry.register_agent(sample_metadata)

        hash1 = sample_metadata.provenance_hash
        hash2 = sample_metadata.provenance_hash

        assert hash1 == hash2
        assert len(hash1) == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
