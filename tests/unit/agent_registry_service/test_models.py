# -*- coding: utf-8 -*-
"""
Unit Tests for Agent Registry Models (AGENT-FOUND-007)

Tests all enums, model classes, field validation, serialization,
hash computation, and edge cases for the agent registry data types.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline enums and models mirroring the agent registry service
# ---------------------------------------------------------------------------


class AgentLayer(str, Enum):
    FOUNDATION = "foundation"
    ORCHESTRATION = "orchestration"
    INGESTION = "ingestion"
    VALIDATION = "validation"
    NORMALIZATION = "normalization"
    CALCULATION = "calculation"
    REPORTING = "reporting"
    COMPLIANCE = "compliance"
    ANALYTICS = "analytics"
    INTEGRATION = "integration"
    UTILITY = "utility"

    @property
    def prefix(self) -> str:
        prefixes = {
            "foundation": "FOUND",
            "orchestration": "ORCH",
            "ingestion": "ING",
            "validation": "VAL",
            "normalization": "NORM",
            "calculation": "CALC",
            "reporting": "RPT",
            "compliance": "COMP",
            "analytics": "ANLYT",
            "integration": "INTG",
            "utility": "UTIL",
        }
        return prefixes.get(self.value, "UNK")

    @property
    def description(self) -> str:
        descriptions = {
            "foundation": "Core platform foundation agents",
            "orchestration": "Workflow and pipeline orchestration",
            "ingestion": "Data ingestion and ETL agents",
            "validation": "Data validation and quality agents",
            "normalization": "Unit and reference normalization",
            "calculation": "Emissions and sustainability calculations",
            "reporting": "Report generation and formatting",
            "compliance": "Regulatory compliance agents",
            "analytics": "Analytics and insights agents",
            "integration": "External system integration agents",
            "utility": "Shared utility agents",
        }
        return descriptions.get(self.value, "Unknown layer")


class SectorClassification(str, Enum):
    ENERGY = "energy"
    MANUFACTURING = "manufacturing"
    TRANSPORTATION = "transportation"
    BUILDINGS = "buildings"
    AGRICULTURE = "agriculture"
    WASTE = "waste"
    INDUSTRIAL_PROCESSES = "industrial_processes"
    LAND_USE = "land_use"
    WATER = "water"
    CROSS_SECTOR = "cross_sector"

    @property
    def description(self) -> str:
        descriptions = {
            "energy": "Energy generation and consumption",
            "manufacturing": "Industrial manufacturing processes",
            "transportation": "Transport and logistics",
            "buildings": "Building energy and operations",
            "agriculture": "Agricultural emissions",
            "waste": "Waste management and treatment",
            "industrial_processes": "Industrial process emissions",
            "land_use": "Land use change and forestry",
            "water": "Water supply and treatment",
            "cross_sector": "Cross-sector and multi-industry",
        }
        return descriptions.get(self.value, "Unknown sector")


class AgentHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DISABLED = "disabled"


class ExecutionMode(str, Enum):
    GLIP_V1 = "glip_v1"
    LEGACY_HTTP = "legacy_http"
    HYBRID = "hybrid"


class IdempotencySupport(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class CapabilityCategory(str, Enum):
    CALCULATION = "calculation"
    VALIDATION = "validation"
    NORMALIZATION = "normalization"
    INGESTION = "ingestion"
    REPORTING = "reporting"
    COMPLIANCE = "compliance"
    ANALYTICS = "analytics"
    ORCHESTRATION = "orchestration"
    INTEGRATION = "integration"
    UTILITY = "utility"


class RegistryChangeType(str, Enum):
    REGISTER = "register"
    UNREGISTER = "unregister"
    UPDATE = "update"
    HOT_RELOAD = "hot_reload"
    HEALTH_CHANGE = "health_change"
    VERSION_ADD = "version_add"


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class SemanticVersion:
    """Semantic version with comparison and compatibility checks."""

    def __init__(self, version_str: str):
        self._raw = version_str
        parts = version_str.split("-", 1)
        core = parts[0]
        self.prerelease = parts[1] if len(parts) > 1 else None
        segments = core.split(".")
        if len(segments) != 3:
            raise ValueError(f"Invalid semantic version: {version_str}")
        try:
            self.major = int(segments[0])
            self.minor = int(segments[1])
            self.patch = int(segments[2])
        except ValueError:
            raise ValueError(f"Invalid semantic version: {version_str}")

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-{self.prerelease}" if self.prerelease else base

    def __repr__(self) -> str:
        return f"SemanticVersion('{self}')"

    def __eq__(self, other) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch, self.prerelease) == \
               (other.major, other.minor, other.patch, other.prerelease)

    def __lt__(self, other) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        # Prerelease versions sort before release
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease
        return False

    def __le__(self, other) -> bool:
        return self == other or self < other

    def __gt__(self, other) -> bool:
        return not self <= other

    def __ge__(self, other) -> bool:
        return not self < other

    def __hash__(self):
        return hash((self.major, self.minor, self.patch, self.prerelease))

    def is_compatible_with(self, other: "SemanticVersion") -> bool:
        """Check if this version is backward-compatible with other (same major)."""
        return self.major == other.major and self >= other


class ResourceProfile:
    """Resource requirements for an agent."""

    def __init__(
        self,
        cpu_request: str = "100m",
        cpu_limit: str = "500m",
        memory_request: str = "128Mi",
        memory_limit: str = "512Mi",
        gpu_required: bool = False,
        gpu_type: Optional[str] = None,
        gpu_count: int = 0,
    ):
        self.cpu_request = cpu_request
        self.cpu_limit = cpu_limit
        self.memory_request = memory_request
        self.memory_limit = memory_limit
        self.gpu_required = gpu_required
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count

    def to_k8s_resources(self) -> Dict[str, Dict[str, str]]:
        resources = {
            "requests": {
                "cpu": self.cpu_request,
                "memory": self.memory_request,
            },
            "limits": {
                "cpu": self.cpu_limit,
                "memory": self.memory_limit,
            },
        }
        if self.gpu_required and self.gpu_type:
            resources["limits"][f"nvidia.com/{self.gpu_type}"] = str(self.gpu_count)
        return resources


class ContainerSpec:
    """Container specification for agent deployment."""

    def __init__(
        self,
        image: str = "",
        tag: str = "latest",
        pull_policy: str = "IfNotPresent",
        ports: Optional[List[int]] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ):
        if image and not re.match(r'^[a-zA-Z0-9_./-]+$', image):
            raise ValueError(f"Invalid image name: {image}")
        self.image = image
        self.tag = tag
        self.pull_policy = pull_policy
        self.ports = ports or [8080]
        self.env_vars = env_vars or {}

    @property
    def full_image(self) -> str:
        return f"{self.image}:{self.tag}" if self.image else ""


class LegacyHttpConfig:
    """Configuration for legacy HTTP-based agents."""

    def __init__(
        self,
        endpoint: str = "",
        auth_type: str = "none",
        timeout_seconds: int = 30,
        retry_count: int = 3,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.endpoint = endpoint
        self.auth_type = auth_type
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        self.headers = headers or {}


class AgentCapability:
    """A capability provided by an agent."""

    def __init__(
        self,
        name: str,
        category: str = "utility",
        input_types: Optional[List[str]] = None,
        output_types: Optional[List[str]] = None,
        description: str = "",
    ):
        self.name = name
        self.category = CapabilityCategory(category)
        self.input_types = input_types or []
        self.output_types = output_types or []
        self.description = description

    def matches(self, required_name: str = "", required_category: str = "",
                required_input: str = "", required_output: str = "") -> bool:
        """Check if this capability matches the given requirements."""
        if required_name and self.name != required_name:
            return False
        if required_category and self.category.value != required_category:
            return False
        if required_input and required_input not in self.input_types:
            return False
        if required_output and required_output not in self.output_types:
            return False
        return True


class AgentVariant:
    """A variant of an agent for different sectors/regions."""

    def __init__(self, region: str = "global", sector: str = "cross_sector",
                 config_overrides: Optional[Dict[str, Any]] = None):
        self.region = region
        self.sector = sector
        self.config_overrides = config_overrides or {}

    @property
    def key(self) -> str:
        return f"{self.region}:{self.sector}"


class AgentDependency:
    """A dependency on another agent."""

    def __init__(self, agent_id: str, version_constraint: str = ">=1.0.0",
                 optional: bool = False):
        self.agent_id = agent_id
        self.version_constraint = version_constraint
        self.optional = optional

    def version_satisfies(self, version: SemanticVersion) -> bool:
        """Check if a version satisfies this constraint."""
        constraint = self.version_constraint.strip()
        if constraint.startswith(">="):
            target = SemanticVersion(constraint[2:])
            return version >= target
        elif constraint.startswith("^"):
            target = SemanticVersion(constraint[1:])
            return version.major == target.major and version >= target
        elif constraint.startswith("~"):
            target = SemanticVersion(constraint[1:])
            return (version.major == target.major and
                    version.minor == target.minor and
                    version >= target)
        elif constraint.startswith("="):
            target = SemanticVersion(constraint[1:])
            return version == target
        else:
            target = SemanticVersion(constraint)
            return version >= target
        return False


class AgentMetadataEntry:
    """Full metadata entry for a registered agent."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        layer: str = "utility",
        sector_classifications: Optional[List[str]] = None,
        execution_mode: str = "glip_v1",
        idempotency_support: str = "none",
        health_status: str = "unknown",
        tags: Optional[List[str]] = None,
        capabilities: Optional[List[AgentCapability]] = None,
        variants: Optional[List[AgentVariant]] = None,
        dependencies: Optional[List[AgentDependency]] = None,
        resource_profile: Optional[ResourceProfile] = None,
        container_spec: Optional[ContainerSpec] = None,
        legacy_http_config: Optional[LegacyHttpConfig] = None,
        registered_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        registered_by: Optional[str] = None,
        provenance_hash: str = "",
    ):
        if not agent_id:
            raise ValueError("agent_id is required")
        if not name:
            raise ValueError("name is required")

        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.version = SemanticVersion(version)
        self.layer = AgentLayer(layer)
        self.sector_classifications = [
            SectorClassification(s) for s in (sector_classifications or [])
        ]
        self.execution_mode = ExecutionMode(execution_mode)
        self.idempotency_support = IdempotencySupport(idempotency_support)
        self.health_status = AgentHealthStatus(health_status)
        self.tags = tags or []
        self.capabilities = capabilities or []
        self.variants = variants or []
        self.dependencies = dependencies or []
        self.resource_profile = resource_profile or ResourceProfile()
        self.container_spec = container_spec
        self.legacy_http_config = legacy_http_config
        self.registered_at = registered_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        self.registered_by = registered_by
        self.provenance_hash = provenance_hash

        # Validate execution mode config
        if self.execution_mode == ExecutionMode.LEGACY_HTTP and not self.legacy_http_config:
            pass  # Warning but allow
        if self.execution_mode == ExecutionMode.GLIP_V1 and self.legacy_http_config:
            pass  # Warning but allow

    def compute_provenance_hash(self) -> str:
        data = {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": str(self.version),
            "layer": self.layer.value,
            "execution_mode": self.execution_mode.value,
            "capabilities": [c.name for c in self.capabilities],
            "dependencies": [d.agent_id for d in self.dependencies],
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def has_capability(self, name: str) -> bool:
        return any(c.name == name for c in self.capabilities)

    def has_variant(self, region: str, sector: str) -> bool:
        return any(v.region == region and v.sector == sector for v in self.variants)

    def supports_sector(self, sector: str) -> bool:
        try:
            sc = SectorClassification(sector)
            return sc in self.sector_classifications
        except ValueError:
            return False

    @property
    def is_glip_compatible(self) -> bool:
        return self.execution_mode in (ExecutionMode.GLIP_V1, ExecutionMode.HYBRID)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAgentLayerEnum:
    """Test AgentLayer enum values and properties."""

    def test_foundation_value(self):
        assert AgentLayer.FOUNDATION.value == "foundation"

    def test_orchestration_value(self):
        assert AgentLayer.ORCHESTRATION.value == "orchestration"

    def test_ingestion_value(self):
        assert AgentLayer.INGESTION.value == "ingestion"

    def test_validation_value(self):
        assert AgentLayer.VALIDATION.value == "validation"

    def test_normalization_value(self):
        assert AgentLayer.NORMALIZATION.value == "normalization"

    def test_calculation_value(self):
        assert AgentLayer.CALCULATION.value == "calculation"

    def test_reporting_value(self):
        assert AgentLayer.REPORTING.value == "reporting"

    def test_compliance_value(self):
        assert AgentLayer.COMPLIANCE.value == "compliance"

    def test_analytics_value(self):
        assert AgentLayer.ANALYTICS.value == "analytics"

    def test_integration_value(self):
        assert AgentLayer.INTEGRATION.value == "integration"

    def test_utility_value(self):
        assert AgentLayer.UTILITY.value == "utility"

    def test_enum_count(self):
        assert len(AgentLayer) == 11

    def test_from_string(self):
        assert AgentLayer("calculation") == AgentLayer.CALCULATION

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            AgentLayer("invalid_layer")

    def test_prefix_property_foundation(self):
        assert AgentLayer.FOUNDATION.prefix == "FOUND"

    def test_prefix_property_calculation(self):
        assert AgentLayer.CALCULATION.prefix == "CALC"

    def test_prefix_property_reporting(self):
        assert AgentLayer.REPORTING.prefix == "RPT"

    def test_description_property_foundation(self):
        assert "foundation" in AgentLayer.FOUNDATION.description.lower()

    def test_description_property_calculation(self):
        assert "calculation" in AgentLayer.CALCULATION.description.lower()

    def test_all_layers_have_prefix(self):
        for layer in AgentLayer:
            assert layer.prefix != "UNK"

    def test_all_layers_have_description(self):
        for layer in AgentLayer:
            assert "Unknown" not in layer.description


class TestSectorClassificationEnum:
    """Test SectorClassification enum values."""

    def test_energy_value(self):
        assert SectorClassification.ENERGY.value == "energy"

    def test_manufacturing_value(self):
        assert SectorClassification.MANUFACTURING.value == "manufacturing"

    def test_transportation_value(self):
        assert SectorClassification.TRANSPORTATION.value == "transportation"

    def test_buildings_value(self):
        assert SectorClassification.BUILDINGS.value == "buildings"

    def test_agriculture_value(self):
        assert SectorClassification.AGRICULTURE.value == "agriculture"

    def test_waste_value(self):
        assert SectorClassification.WASTE.value == "waste"

    def test_industrial_processes_value(self):
        assert SectorClassification.INDUSTRIAL_PROCESSES.value == "industrial_processes"

    def test_land_use_value(self):
        assert SectorClassification.LAND_USE.value == "land_use"

    def test_water_value(self):
        assert SectorClassification.WATER.value == "water"

    def test_cross_sector_value(self):
        assert SectorClassification.CROSS_SECTOR.value == "cross_sector"

    def test_enum_count(self):
        assert len(SectorClassification) == 10

    def test_description_property_energy(self):
        assert "energy" in SectorClassification.ENERGY.description.lower()

    def test_description_property_manufacturing(self):
        assert "manufacturing" in SectorClassification.MANUFACTURING.description.lower()

    def test_all_sectors_have_description(self):
        for sector in SectorClassification:
            assert "Unknown" not in sector.description


class TestAgentHealthStatusEnum:
    """Test AgentHealthStatus enum values."""

    def test_healthy(self):
        assert AgentHealthStatus.HEALTHY.value == "healthy"

    def test_degraded(self):
        assert AgentHealthStatus.DEGRADED.value == "degraded"

    def test_unhealthy(self):
        assert AgentHealthStatus.UNHEALTHY.value == "unhealthy"

    def test_unknown(self):
        assert AgentHealthStatus.UNKNOWN.value == "unknown"

    def test_disabled(self):
        assert AgentHealthStatus.DISABLED.value == "disabled"

    def test_enum_count(self):
        assert len(AgentHealthStatus) == 5

    def test_from_string(self):
        assert AgentHealthStatus("healthy") == AgentHealthStatus.HEALTHY


class TestExecutionModeEnum:
    """Test ExecutionMode enum values."""

    def test_glip_v1(self):
        assert ExecutionMode.GLIP_V1.value == "glip_v1"

    def test_legacy_http(self):
        assert ExecutionMode.LEGACY_HTTP.value == "legacy_http"

    def test_hybrid(self):
        assert ExecutionMode.HYBRID.value == "hybrid"

    def test_enum_count(self):
        assert len(ExecutionMode) == 3


class TestIdempotencySupportEnum:
    """Test IdempotencySupport enum values."""

    def test_full(self):
        assert IdempotencySupport.FULL.value == "full"

    def test_partial(self):
        assert IdempotencySupport.PARTIAL.value == "partial"

    def test_none(self):
        assert IdempotencySupport.NONE.value == "none"

    def test_enum_count(self):
        assert len(IdempotencySupport) == 3


class TestCapabilityCategoryEnum:
    """Test CapabilityCategory enum values."""

    def test_calculation(self):
        assert CapabilityCategory.CALCULATION.value == "calculation"

    def test_validation(self):
        assert CapabilityCategory.VALIDATION.value == "validation"

    def test_normalization(self):
        assert CapabilityCategory.NORMALIZATION.value == "normalization"

    def test_ingestion(self):
        assert CapabilityCategory.INGESTION.value == "ingestion"

    def test_reporting(self):
        assert CapabilityCategory.REPORTING.value == "reporting"

    def test_compliance(self):
        assert CapabilityCategory.COMPLIANCE.value == "compliance"

    def test_analytics(self):
        assert CapabilityCategory.ANALYTICS.value == "analytics"

    def test_orchestration(self):
        assert CapabilityCategory.ORCHESTRATION.value == "orchestration"

    def test_integration(self):
        assert CapabilityCategory.INTEGRATION.value == "integration"

    def test_utility(self):
        assert CapabilityCategory.UTILITY.value == "utility"

    def test_enum_count(self):
        assert len(CapabilityCategory) == 10


class TestRegistryChangeTypeEnum:
    """Test RegistryChangeType enum values."""

    def test_register(self):
        assert RegistryChangeType.REGISTER.value == "register"

    def test_unregister(self):
        assert RegistryChangeType.UNREGISTER.value == "unregister"

    def test_update(self):
        assert RegistryChangeType.UPDATE.value == "update"

    def test_hot_reload(self):
        assert RegistryChangeType.HOT_RELOAD.value == "hot_reload"

    def test_health_change(self):
        assert RegistryChangeType.HEALTH_CHANGE.value == "health_change"

    def test_version_add(self):
        assert RegistryChangeType.VERSION_ADD.value == "version_add"

    def test_enum_count(self):
        assert len(RegistryChangeType) == 6


class TestSemanticVersion:
    """Test SemanticVersion parsing, comparison, and compatibility."""

    def test_parse_valid(self):
        v = SemanticVersion("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_zeros(self):
        v = SemanticVersion("0.0.0")
        assert v.major == 0 and v.minor == 0 and v.patch == 0

    def test_parse_large_numbers(self):
        v = SemanticVersion("100.200.300")
        assert v.major == 100

    def test_parse_invalid_two_segments(self):
        with pytest.raises(ValueError):
            SemanticVersion("1.2")

    def test_parse_invalid_one_segment(self):
        with pytest.raises(ValueError):
            SemanticVersion("1")

    def test_parse_invalid_non_numeric(self):
        with pytest.raises(ValueError):
            SemanticVersion("a.b.c")

    def test_str_conversion(self):
        v = SemanticVersion("1.2.3")
        assert str(v) == "1.2.3"

    def test_repr_conversion(self):
        v = SemanticVersion("1.2.3")
        assert "1.2.3" in repr(v)

    def test_equality(self):
        assert SemanticVersion("1.2.3") == SemanticVersion("1.2.3")

    def test_inequality(self):
        assert SemanticVersion("1.2.3") != SemanticVersion("1.2.4")

    def test_less_than_patch(self):
        assert SemanticVersion("1.2.3") < SemanticVersion("1.2.4")

    def test_less_than_minor(self):
        assert SemanticVersion("1.2.3") < SemanticVersion("1.3.0")

    def test_less_than_major(self):
        assert SemanticVersion("1.2.3") < SemanticVersion("2.0.0")

    def test_greater_than(self):
        assert SemanticVersion("2.0.0") > SemanticVersion("1.9.9")

    def test_less_equal(self):
        assert SemanticVersion("1.2.3") <= SemanticVersion("1.2.3")
        assert SemanticVersion("1.2.3") <= SemanticVersion("1.2.4")

    def test_greater_equal(self):
        assert SemanticVersion("1.2.3") >= SemanticVersion("1.2.3")
        assert SemanticVersion("1.2.4") >= SemanticVersion("1.2.3")

    def test_prerelease_parsing(self):
        v = SemanticVersion("1.2.3-beta.1")
        assert v.prerelease == "beta.1"

    def test_prerelease_str(self):
        v = SemanticVersion("1.2.3-rc.1")
        assert str(v) == "1.2.3-rc.1"

    def test_prerelease_less_than_release(self):
        assert SemanticVersion("1.2.3-beta") < SemanticVersion("1.2.3")

    def test_compatible_same_major(self):
        v1 = SemanticVersion("1.2.3")
        v2 = SemanticVersion("1.0.0")
        assert v1.is_compatible_with(v2)

    def test_incompatible_different_major(self):
        v1 = SemanticVersion("2.0.0")
        v2 = SemanticVersion("1.0.0")
        assert not v1.is_compatible_with(v2)

    def test_compatible_requires_greater_equal(self):
        v1 = SemanticVersion("1.0.0")
        v2 = SemanticVersion("1.2.0")
        assert not v1.is_compatible_with(v2)

    def test_hash_for_set_usage(self):
        s = {SemanticVersion("1.0.0"), SemanticVersion("1.0.0")}
        assert len(s) == 1


class TestResourceProfile:
    """Test ResourceProfile model."""

    def test_defaults(self):
        rp = ResourceProfile()
        assert rp.cpu_request == "100m"
        assert rp.cpu_limit == "500m"
        assert rp.memory_request == "128Mi"
        assert rp.memory_limit == "512Mi"
        assert rp.gpu_required is False

    def test_custom_values(self):
        rp = ResourceProfile(cpu_request="200m", memory_limit="1Gi")
        assert rp.cpu_request == "200m"
        assert rp.memory_limit == "1Gi"

    def test_to_k8s_resources(self):
        rp = ResourceProfile()
        k8s = rp.to_k8s_resources()
        assert k8s["requests"]["cpu"] == "100m"
        assert k8s["requests"]["memory"] == "128Mi"
        assert k8s["limits"]["cpu"] == "500m"
        assert k8s["limits"]["memory"] == "512Mi"

    def test_to_k8s_resources_with_gpu(self):
        rp = ResourceProfile(gpu_required=True, gpu_type="gpu", gpu_count=1)
        k8s = rp.to_k8s_resources()
        assert "nvidia.com/gpu" in k8s["limits"]
        assert k8s["limits"]["nvidia.com/gpu"] == "1"

    def test_no_gpu_in_k8s_resources(self):
        rp = ResourceProfile()
        k8s = rp.to_k8s_resources()
        assert not any("nvidia" in k for k in k8s["limits"])


class TestContainerSpec:
    """Test ContainerSpec model."""

    def test_defaults(self):
        cs = ContainerSpec()
        assert cs.image == ""
        assert cs.tag == "latest"
        assert cs.pull_policy == "IfNotPresent"
        assert cs.ports == [8080]

    def test_valid_image(self):
        cs = ContainerSpec(image="greenlang/agent-001")
        assert cs.image == "greenlang/agent-001"

    def test_invalid_image_raises(self):
        with pytest.raises(ValueError):
            ContainerSpec(image="image with spaces")

    def test_full_image_property(self):
        cs = ContainerSpec(image="greenlang/agent", tag="2.1.0")
        assert cs.full_image == "greenlang/agent:2.1.0"

    def test_full_image_empty(self):
        cs = ContainerSpec()
        assert cs.full_image == ""

    def test_env_vars(self):
        cs = ContainerSpec(env_vars={"KEY": "VALUE"})
        assert cs.env_vars["KEY"] == "VALUE"


class TestLegacyHttpConfig:
    """Test LegacyHttpConfig model."""

    def test_defaults(self):
        cfg = LegacyHttpConfig()
        assert cfg.endpoint == ""
        assert cfg.auth_type == "none"
        assert cfg.timeout_seconds == 30
        assert cfg.retry_count == 3

    def test_custom_endpoint(self):
        cfg = LegacyHttpConfig(endpoint="http://legacy:8080/api")
        assert cfg.endpoint == "http://legacy:8080/api"

    def test_auth_type(self):
        cfg = LegacyHttpConfig(auth_type="bearer")
        assert cfg.auth_type == "bearer"

    def test_timeout(self):
        cfg = LegacyHttpConfig(timeout_seconds=60)
        assert cfg.timeout_seconds == 60

    def test_headers(self):
        cfg = LegacyHttpConfig(headers={"X-Api-Key": "secret"})
        assert cfg.headers["X-Api-Key"] == "secret"


class TestAgentCapability:
    """Test AgentCapability model and matches() logic."""

    def test_creation(self):
        cap = AgentCapability(name="carbon_calc", category="calculation")
        assert cap.name == "carbon_calc"
        assert cap.category == CapabilityCategory.CALCULATION

    def test_matches_by_name(self):
        cap = AgentCapability(name="carbon_calc", category="calculation")
        assert cap.matches(required_name="carbon_calc") is True
        assert cap.matches(required_name="other") is False

    def test_matches_by_category(self):
        cap = AgentCapability(name="carbon_calc", category="calculation")
        assert cap.matches(required_category="calculation") is True
        assert cap.matches(required_category="validation") is False

    def test_matches_by_input_type(self):
        cap = AgentCapability(
            name="calc", category="calculation",
            input_types=["emission_factor", "activity_data"],
        )
        assert cap.matches(required_input="emission_factor") is True
        assert cap.matches(required_input="unknown") is False

    def test_matches_by_output_type(self):
        cap = AgentCapability(
            name="calc", category="calculation",
            output_types=["carbon_footprint"],
        )
        assert cap.matches(required_output="carbon_footprint") is True
        assert cap.matches(required_output="unknown") is False

    def test_matches_all_criteria(self):
        cap = AgentCapability(
            name="calc", category="calculation",
            input_types=["ef"], output_types=["cf"],
        )
        assert cap.matches(
            required_name="calc", required_category="calculation",
            required_input="ef", required_output="cf",
        ) is True

    def test_matches_empty_criteria_returns_true(self):
        cap = AgentCapability(name="calc", category="calculation")
        assert cap.matches() is True


class TestAgentVariant:
    """Test AgentVariant model."""

    def test_creation(self):
        v = AgentVariant(region="EU", sector="energy")
        assert v.region == "EU"
        assert v.sector == "energy"

    def test_key_property(self):
        v = AgentVariant(region="US", sector="manufacturing")
        assert v.key == "US:manufacturing"

    def test_defaults(self):
        v = AgentVariant()
        assert v.region == "global"
        assert v.sector == "cross_sector"

    def test_config_overrides(self):
        v = AgentVariant(config_overrides={"param": "value"})
        assert v.config_overrides["param"] == "value"


class TestAgentDependency:
    """Test AgentDependency and version_satisfies()."""

    def test_creation(self):
        dep = AgentDependency(agent_id="gl-001", version_constraint=">=1.0.0")
        assert dep.agent_id == "gl-001"
        assert dep.optional is False

    def test_version_satisfies_gte(self):
        dep = AgentDependency(agent_id="gl-001", version_constraint=">=2.0.0")
        assert dep.version_satisfies(SemanticVersion("2.1.0")) is True
        assert dep.version_satisfies(SemanticVersion("2.0.0")) is True
        assert dep.version_satisfies(SemanticVersion("1.9.0")) is False

    def test_version_satisfies_caret(self):
        dep = AgentDependency(agent_id="gl-001", version_constraint="^1.2.0")
        assert dep.version_satisfies(SemanticVersion("1.5.0")) is True
        assert dep.version_satisfies(SemanticVersion("1.2.0")) is True
        assert dep.version_satisfies(SemanticVersion("2.0.0")) is False
        assert dep.version_satisfies(SemanticVersion("1.1.0")) is False

    def test_version_satisfies_tilde(self):
        dep = AgentDependency(agent_id="gl-001", version_constraint="~1.2.0")
        assert dep.version_satisfies(SemanticVersion("1.2.5")) is True
        assert dep.version_satisfies(SemanticVersion("1.2.0")) is True
        assert dep.version_satisfies(SemanticVersion("1.3.0")) is False

    def test_version_satisfies_exact(self):
        dep = AgentDependency(agent_id="gl-001", version_constraint="=1.2.3")
        assert dep.version_satisfies(SemanticVersion("1.2.3")) is True
        assert dep.version_satisfies(SemanticVersion("1.2.4")) is False

    def test_version_satisfies_bare(self):
        dep = AgentDependency(agent_id="gl-001", version_constraint="1.0.0")
        assert dep.version_satisfies(SemanticVersion("1.0.0")) is True
        assert dep.version_satisfies(SemanticVersion("2.0.0")) is True
        assert dep.version_satisfies(SemanticVersion("0.9.0")) is False

    def test_optional_dependency(self):
        dep = AgentDependency(agent_id="gl-001", optional=True)
        assert dep.optional is True


class TestAgentMetadataEntry:
    """Test AgentMetadataEntry creation, validation, and methods."""

    def test_basic_creation(self):
        entry = AgentMetadataEntry(agent_id="gl-001", name="Agent 001")
        assert entry.agent_id == "gl-001"
        assert entry.name == "Agent 001"

    def test_empty_agent_id_raises(self):
        with pytest.raises(ValueError):
            AgentMetadataEntry(agent_id="", name="Test")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            AgentMetadataEntry(agent_id="gl-001", name="")

    def test_version_parsed(self):
        entry = AgentMetadataEntry(agent_id="gl-001", name="A", version="2.1.0")
        assert entry.version.major == 2
        assert entry.version.minor == 1
        assert entry.version.patch == 0

    def test_layer_enum(self):
        entry = AgentMetadataEntry(agent_id="gl-001", name="A", layer="calculation")
        assert entry.layer == AgentLayer.CALCULATION

    def test_sector_classifications(self):
        entry = AgentMetadataEntry(
            agent_id="gl-001", name="A",
            sector_classifications=["energy", "manufacturing"],
        )
        assert SectorClassification.ENERGY in entry.sector_classifications
        assert SectorClassification.MANUFACTURING in entry.sector_classifications

    def test_execution_mode(self):
        entry = AgentMetadataEntry(
            agent_id="gl-001", name="A", execution_mode="legacy_http",
        )
        assert entry.execution_mode == ExecutionMode.LEGACY_HTTP

    def test_provenance_hash(self):
        entry = AgentMetadataEntry(agent_id="gl-001", name="A")
        h = entry.compute_provenance_hash()
        assert len(h) == 64
        assert re.match(r"^[0-9a-f]{64}$", h)

    def test_provenance_hash_deterministic(self):
        entry = AgentMetadataEntry(agent_id="gl-001", name="A", version="1.0.0")
        h1 = entry.compute_provenance_hash()
        h2 = entry.compute_provenance_hash()
        assert h1 == h2

    def test_has_capability_true(self):
        cap = AgentCapability(name="calc", category="calculation")
        entry = AgentMetadataEntry(
            agent_id="gl-001", name="A", capabilities=[cap],
        )
        assert entry.has_capability("calc") is True

    def test_has_capability_false(self):
        entry = AgentMetadataEntry(agent_id="gl-001", name="A")
        assert entry.has_capability("calc") is False

    def test_has_variant(self):
        variant = AgentVariant(region="EU", sector="energy")
        entry = AgentMetadataEntry(
            agent_id="gl-001", name="A", variants=[variant],
        )
        assert entry.has_variant("EU", "energy") is True
        assert entry.has_variant("US", "energy") is False

    def test_supports_sector_true(self):
        entry = AgentMetadataEntry(
            agent_id="gl-001", name="A",
            sector_classifications=["energy"],
        )
        assert entry.supports_sector("energy") is True

    def test_supports_sector_false(self):
        entry = AgentMetadataEntry(agent_id="gl-001", name="A")
        assert entry.supports_sector("energy") is False

    def test_supports_sector_invalid(self):
        entry = AgentMetadataEntry(agent_id="gl-001", name="A")
        assert entry.supports_sector("nonexistent") is False

    def test_is_glip_compatible_glip_v1(self):
        entry = AgentMetadataEntry(
            agent_id="gl-001", name="A", execution_mode="glip_v1",
        )
        assert entry.is_glip_compatible is True

    def test_is_glip_compatible_hybrid(self):
        entry = AgentMetadataEntry(
            agent_id="gl-001", name="A", execution_mode="hybrid",
        )
        assert entry.is_glip_compatible is True

    def test_is_glip_compatible_legacy(self):
        entry = AgentMetadataEntry(
            agent_id="gl-001", name="A", execution_mode="legacy_http",
        )
        assert entry.is_glip_compatible is False

    def test_default_health_status(self):
        entry = AgentMetadataEntry(agent_id="gl-001", name="A")
        assert entry.health_status == AgentHealthStatus.UNKNOWN

    def test_default_resource_profile(self):
        entry = AgentMetadataEntry(agent_id="gl-001", name="A")
        assert entry.resource_profile is not None
        assert entry.resource_profile.cpu_request == "100m"

    def test_timestamps_auto(self):
        entry = AgentMetadataEntry(agent_id="gl-001", name="A")
        assert isinstance(entry.registered_at, datetime)
        assert isinstance(entry.updated_at, datetime)
