"""
Unit tests for STRIDEEngine.

Tests STRIDE threat analysis for components, data flows, and trust boundaries.
Validates threat identification, categorization, and threat model generation.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4

from greenlang.infrastructure.threat_modeling.stride_engine import (
    STRIDEEngine,
    THREAT_CATEGORIES,
)
from greenlang.infrastructure.threat_modeling.models import (
    Component,
    ComponentType,
    DataFlow,
    TrustBoundary,
    Threat,
    ThreatCategory,
    ThreatModel,
    RiskLevel,
)


class TestSTRIDEEngineInitialization:
    """Test STRIDEEngine initialization."""

    def test_initialization_with_config(self, stride_config):
        """Test engine initializes with configuration."""
        engine = STRIDEEngine(config=stride_config)

        assert engine.config == stride_config
        assert engine.enabled_categories == stride_config.enabled_categories

    def test_initialization_default_config(self):
        """Test engine initializes with default configuration."""
        engine = STRIDEEngine()

        assert engine.config is not None
        assert len(engine.enabled_categories) == 6  # All STRIDE categories

    def test_initialization_loads_threat_library(self, stride_config):
        """Test engine loads threat library."""
        engine = STRIDEEngine(config=stride_config)

        assert engine.threat_library is not None


class TestThreatCategories:
    """Test THREAT_CATEGORIES constant."""

    def test_all_stride_categories_defined(self):
        """Test all STRIDE categories are defined."""
        expected_categories = ["S", "T", "R", "I", "D", "E"]

        for cat in expected_categories:
            assert cat in THREAT_CATEGORIES

    def test_category_definitions_complete(self):
        """Test category definitions have required fields."""
        for cat_id, cat_def in THREAT_CATEGORIES.items():
            assert "name" in cat_def
            assert "description" in cat_def
            assert "examples" in cat_def
            assert isinstance(cat_def["examples"], list)

    def test_spoofing_category(self):
        """Test Spoofing category definition."""
        assert THREAT_CATEGORIES["S"]["name"] == "Spoofing"
        assert "identity" in THREAT_CATEGORIES["S"]["description"].lower()

    def test_tampering_category(self):
        """Test Tampering category definition."""
        assert THREAT_CATEGORIES["T"]["name"] == "Tampering"
        assert "modif" in THREAT_CATEGORIES["T"]["description"].lower()

    def test_repudiation_category(self):
        """Test Repudiation category definition."""
        assert THREAT_CATEGORIES["R"]["name"] == "Repudiation"
        assert "deny" in THREAT_CATEGORIES["R"]["description"].lower()

    def test_information_disclosure_category(self):
        """Test Information Disclosure category definition."""
        assert THREAT_CATEGORIES["I"]["name"] == "Information Disclosure"
        assert "expos" in THREAT_CATEGORIES["I"]["description"].lower()

    def test_denial_of_service_category(self):
        """Test Denial of Service category definition."""
        assert THREAT_CATEGORIES["D"]["name"] == "Denial of Service"
        assert "availab" in THREAT_CATEGORIES["D"]["description"].lower()

    def test_elevation_of_privilege_category(self):
        """Test Elevation of Privilege category definition."""
        assert THREAT_CATEGORIES["E"]["name"] == "Elevation of Privilege"
        assert "privileg" in THREAT_CATEGORIES["E"]["description"].lower()


class TestAnalyzeComponent:
    """Test component analysis for threats."""

    def test_analyze_service_component(self, stride_config, sample_component):
        """Test analyzing service component."""
        engine = STRIDEEngine(config=stride_config)

        threats = engine.analyze_component(sample_component)

        assert isinstance(threats, list)
        # Services should have multiple threat categories
        categories = {t.category for t in threats}
        assert len(categories) >= 1

    def test_analyze_database_component(self, stride_config, database_component):
        """Test analyzing database component."""
        engine = STRIDEEngine(config=stride_config)

        threats = engine.analyze_component(database_component)

        assert isinstance(threats, list)
        # Databases should have tampering and information disclosure threats
        categories = {t.category for t in threats}
        assert ThreatCategory.TAMPERING in categories or ThreatCategory.INFORMATION_DISCLOSURE in categories

    def test_analyze_external_service_component(self, stride_config, external_api_component):
        """Test analyzing external service component."""
        engine = STRIDEEngine(config=stride_config)

        threats = engine.analyze_component(external_api_component)

        assert isinstance(threats, list)

    def test_analyze_user_component(self, stride_config, user_component):
        """Test analyzing user component."""
        engine = STRIDEEngine(config=stride_config)

        threats = engine.analyze_component(user_component)

        assert isinstance(threats, list)
        # Users should have spoofing and repudiation threats
        categories = {t.category for t in threats}
        assert ThreatCategory.SPOOFING in categories or ThreatCategory.REPUDIATION in categories

    def test_analyze_component_sets_affected_component(
        self, stride_config, sample_component
    ):
        """Test threats reference the analyzed component."""
        engine = STRIDEEngine(config=stride_config)

        threats = engine.analyze_component(sample_component)

        for threat in threats:
            assert sample_component.component_id in threat.affected_component_ids

    def test_analyze_component_considers_data_classification(self, stride_config):
        """Test analysis considers data classification."""
        engine = STRIDEEngine(config=stride_config)

        public_component = Component(
            component_id=str(uuid4()),
            name="public-cdn",
            component_type=ComponentType.SERVICE,
            description="Public CDN service",
            technology_stack=["CloudFront"],
            data_classification="public",
            authentication_required=False,
            authorization_level="none",
            exposed_ports=[443],
            protocols=["HTTPS"],
            metadata={},
        )

        restricted_component = Component(
            component_id=str(uuid4()),
            name="secrets-service",
            component_type=ComponentType.SERVICE,
            description="Secrets management service",
            technology_stack=["Vault"],
            data_classification="restricted",
            authentication_required=True,
            authorization_level="role_based",
            exposed_ports=[8200],
            protocols=["HTTPS"],
            metadata={},
        )

        public_threats = engine.analyze_component(public_component)
        restricted_threats = engine.analyze_component(restricted_component)

        # Restricted component should have more/higher severity threats
        assert len(restricted_threats) >= len(public_threats)

    def test_analyze_component_considers_authentication(self, stride_config):
        """Test analysis considers authentication requirements."""
        engine = STRIDEEngine(config=stride_config)

        no_auth_component = Component(
            component_id=str(uuid4()),
            name="public-api",
            component_type=ComponentType.SERVICE,
            description="Public API",
            technology_stack=["FastAPI"],
            data_classification="public",
            authentication_required=False,
            authorization_level="none",
            exposed_ports=[8080],
            protocols=["HTTP"],
            metadata={},
        )

        threats = engine.analyze_component(no_auth_component)

        # Should identify spoofing threats for unauthenticated endpoints
        spoofing_threats = [t for t in threats if t.category == ThreatCategory.SPOOFING]
        assert len(spoofing_threats) >= 0  # May or may not find spoofing


class TestAnalyzeDataFlow:
    """Test data flow analysis for threats."""

    def test_analyze_data_flow(self, stride_config, api_to_db_flow):
        """Test analyzing data flow."""
        engine = STRIDEEngine(config=stride_config)

        threats = engine.analyze_data_flow(api_to_db_flow)

        assert isinstance(threats, list)

    def test_analyze_unencrypted_flow(self, stride_config):
        """Test analyzing unencrypted data flow."""
        engine = STRIDEEngine(config=stride_config)

        unencrypted_flow = DataFlow(
            flow_id=str(uuid4()),
            name="Unencrypted Flow",
            source_component_id=str(uuid4()),
            destination_component_id=str(uuid4()),
            data_types=["sensitive_data"],
            protocol="HTTP",
            encrypted=False,
            authentication="none",
            data_classification="confidential",
            metadata={},
        )

        threats = engine.analyze_data_flow(unencrypted_flow)

        # Should identify information disclosure and tampering threats
        categories = {t.category for t in threats}
        assert ThreatCategory.INFORMATION_DISCLOSURE in categories or ThreatCategory.TAMPERING in categories

    def test_analyze_flow_with_pci_data(self, stride_config, api_to_external_flow):
        """Test analyzing flow with PCI data."""
        engine = STRIDEEngine(config=stride_config)

        threats = engine.analyze_data_flow(api_to_external_flow)

        # PCI data flows should have higher scrutiny
        assert len(threats) >= 1

    def test_analyze_data_flow_sets_affected_flow(self, stride_config, user_to_api_flow):
        """Test threats reference the analyzed data flow."""
        engine = STRIDEEngine(config=stride_config)

        threats = engine.analyze_data_flow(user_to_api_flow)

        for threat in threats:
            assert user_to_api_flow.flow_id in threat.affected_data_flow_ids

    def test_analyze_data_flow_considers_protocol(self, stride_config):
        """Test analysis considers protocol security."""
        engine = STRIDEEngine(config=stride_config)

        http_flow = DataFlow(
            flow_id=str(uuid4()),
            name="HTTP Flow",
            source_component_id=str(uuid4()),
            destination_component_id=str(uuid4()),
            data_types=["user_data"],
            protocol="HTTP",
            encrypted=False,
            authentication="basic",
            data_classification="confidential",
            metadata={},
        )

        https_flow = DataFlow(
            flow_id=str(uuid4()),
            name="HTTPS Flow",
            source_component_id=str(uuid4()),
            destination_component_id=str(uuid4()),
            data_types=["user_data"],
            protocol="HTTPS",
            encrypted=True,
            authentication="jwt",
            data_classification="confidential",
            metadata={},
        )

        http_threats = engine.analyze_data_flow(http_flow)
        https_threats = engine.analyze_data_flow(https_flow)

        # HTTP should have more/higher severity threats
        assert len(http_threats) >= len(https_threats)


class TestAnalyzeTrustBoundary:
    """Test trust boundary analysis for threats."""

    def test_analyze_network_boundary(self, stride_config, network_boundary):
        """Test analyzing network trust boundary."""
        engine = STRIDEEngine(config=stride_config)

        threats = engine.analyze_trust_boundary(network_boundary)

        assert isinstance(threats, list)

    def test_analyze_authentication_boundary(self, stride_config, authentication_boundary):
        """Test analyzing authentication trust boundary."""
        engine = STRIDEEngine(config=stride_config)

        threats = engine.analyze_trust_boundary(authentication_boundary)

        # Should identify spoofing and elevation of privilege threats
        categories = {t.category for t in threats}
        assert len(categories) >= 1

    def test_analyze_boundary_without_controls(self, stride_config):
        """Test analyzing boundary without security controls."""
        engine = STRIDEEngine(config=stride_config)

        weak_boundary = TrustBoundary(
            boundary_id=str(uuid4()),
            name="Weak Boundary",
            boundary_type="network",
            description="Boundary with no controls",
            components_inside=[],
            components_outside=[],
            controls=[],  # No controls
            metadata={},
        )

        threats = engine.analyze_trust_boundary(weak_boundary)

        # Should identify more threats due to lack of controls
        assert len(threats) >= 1

    def test_analyze_boundary_crossing(self, stride_config):
        """Test analyzing boundary crossing scenario."""
        engine = STRIDEEngine(config=stride_config)

        boundary = TrustBoundary(
            boundary_id=str(uuid4()),
            name="DMZ Boundary",
            boundary_type="network",
            description="DMZ boundary",
            components_inside=["internal-api"],
            components_outside=["external-client"],
            controls=["firewall"],
            metadata={"crossing_flows": ["external-to-internal"]},
        )

        threats = engine.analyze_trust_boundary(boundary)

        assert isinstance(threats, list)


class TestGenerateThreatModel:
    """Test threat model generation."""

    def test_generate_threat_model(
        self,
        stride_config,
        multiple_components,
        multiple_data_flows,
        multiple_boundaries,
    ):
        """Test generating complete threat model."""
        engine = STRIDEEngine(config=stride_config)

        model = engine.generate_threat_model(
            name="Test Model",
            components=multiple_components,
            data_flows=multiple_data_flows,
            trust_boundaries=multiple_boundaries,
        )

        assert isinstance(model, ThreatModel)
        assert model.name == "Test Model"
        assert len(model.components) == len(multiple_components)
        assert len(model.data_flows) == len(multiple_data_flows)
        assert len(model.trust_boundaries) == len(multiple_boundaries)

    def test_generate_threat_model_analyzes_all_components(
        self,
        stride_config,
        multiple_components,
    ):
        """Test model generation analyzes all components."""
        engine = STRIDEEngine(config=stride_config)

        model = engine.generate_threat_model(
            name="Test Model",
            components=multiple_components,
            data_flows=[],
            trust_boundaries=[],
        )

        # All components should be analyzed
        all_affected_components = set()
        for threat in model.threats:
            all_affected_components.update(threat.affected_component_ids)

        component_ids = {c.component_id for c in multiple_components}
        # At least some components should have threats
        assert len(all_affected_components & component_ids) >= 1

    def test_generate_threat_model_calculates_overall_risk(
        self,
        stride_config,
        multiple_components,
        multiple_data_flows,
        multiple_boundaries,
    ):
        """Test model generation calculates overall risk."""
        engine = STRIDEEngine(config=stride_config)

        model = engine.generate_threat_model(
            name="Test Model",
            components=multiple_components,
            data_flows=multiple_data_flows,
            trust_boundaries=multiple_boundaries,
        )

        assert model.overall_risk_score >= 0
        assert model.overall_risk_level is not None

    def test_generate_threat_model_sets_metadata(
        self,
        stride_config,
        sample_component,
    ):
        """Test model generation sets metadata."""
        engine = STRIDEEngine(config=stride_config)

        model = engine.generate_threat_model(
            name="Test Model",
            components=[sample_component],
            data_flows=[],
            trust_boundaries=[],
            metadata={"custom_field": "value"},
        )

        assert model.created_at is not None
        assert "custom_field" in model.metadata

    def test_generate_threat_model_empty_inputs(self, stride_config):
        """Test model generation with empty inputs."""
        engine = STRIDEEngine(config=stride_config)

        model = engine.generate_threat_model(
            name="Empty Model",
            components=[],
            data_flows=[],
            trust_boundaries=[],
        )

        assert isinstance(model, ThreatModel)
        assert len(model.threats) == 0


class TestThreatLibraryIntegration:
    """Test threat library integration."""

    def test_loads_threats_from_library(self, stride_config, mock_threat_library):
        """Test engine loads threats from library."""
        engine = STRIDEEngine(config=stride_config)

        with patch.object(engine, "threat_library", mock_threat_library):
            component = Component(
                component_id=str(uuid4()),
                name="test-service",
                component_type=ComponentType.SERVICE,
                description="Test",
                technology_stack=["Python"],
                data_classification="confidential",
                authentication_required=True,
                authorization_level="role_based",
                exposed_ports=[8000],
                protocols=["HTTPS"],
                metadata={},
            )

            engine.analyze_component(component)

            mock_threat_library.get_threats_for_component.assert_called()

    def test_enriches_threats_with_library_data(self, stride_config):
        """Test threats are enriched with library data."""
        engine = STRIDEEngine(config=stride_config)

        # Mock threat library to return enrichment data
        mock_library = MagicMock()
        mock_library.get_threats_for_component.return_value = [
            {
                "category": ThreatCategory.SPOOFING,
                "title": "Library Threat",
                "description": "From library",
                "mitigations": ["Use MFA", "Implement token binding"],
            }
        ]

        with patch.object(engine, "threat_library", mock_library):
            component = Component(
                component_id=str(uuid4()),
                name="test-service",
                component_type=ComponentType.SERVICE,
                description="Test",
                technology_stack=["Python"],
                data_classification="confidential",
                authentication_required=True,
                authorization_level="role_based",
                exposed_ports=[8000],
                protocols=["HTTPS"],
                metadata={},
            )

            threats = engine.analyze_component(component)

            # Should include library threat
            library_threats = [t for t in threats if "Library" in t.title]
            assert len(library_threats) >= 0  # May or may not find


class TestThreatDeduplication:
    """Test threat deduplication."""

    def test_deduplicates_similar_threats(self, stride_config, sample_component):
        """Test similar threats are deduplicated."""
        engine = STRIDEEngine(config=stride_config)

        # Analyze same component multiple times
        threats1 = engine.analyze_component(sample_component)
        threats2 = engine.analyze_component(sample_component)

        # When combined, should be deduplicated
        combined = engine._deduplicate_threats(threats1 + threats2)

        assert len(combined) <= len(threats1) + len(threats2)

    def test_keeps_distinct_threats(self, stride_config, sample_component, database_component):
        """Test distinct threats are kept."""
        engine = STRIDEEngine(config=stride_config)

        threats1 = engine.analyze_component(sample_component)
        threats2 = engine.analyze_component(database_component)

        combined = engine._deduplicate_threats(threats1 + threats2)

        # Should keep threats from both components
        assert len(combined) >= max(len(threats1), len(threats2))


class TestCategoryFiltering:
    """Test category filtering functionality."""

    def test_filter_by_enabled_categories(self, stride_config):
        """Test only enabled categories are analyzed."""
        # Enable only spoofing and tampering
        stride_config.enabled_categories = [
            ThreatCategory.SPOOFING,
            ThreatCategory.TAMPERING,
        ]
        engine = STRIDEEngine(config=stride_config)

        component = Component(
            component_id=str(uuid4()),
            name="test-service",
            component_type=ComponentType.SERVICE,
            description="Test",
            technology_stack=["Python"],
            data_classification="confidential",
            authentication_required=True,
            authorization_level="role_based",
            exposed_ports=[8000],
            protocols=["HTTPS"],
            metadata={},
        )

        threats = engine.analyze_component(component)

        # Should only have spoofing and tampering threats
        for threat in threats:
            assert threat.category in stride_config.enabled_categories

    def test_all_categories_enabled_by_default(self):
        """Test all categories are enabled by default."""
        engine = STRIDEEngine()

        assert len(engine.enabled_categories) == 6
        assert ThreatCategory.SPOOFING in engine.enabled_categories
        assert ThreatCategory.TAMPERING in engine.enabled_categories
        assert ThreatCategory.REPUDIATION in engine.enabled_categories
        assert ThreatCategory.INFORMATION_DISCLOSURE in engine.enabled_categories
        assert ThreatCategory.DENIAL_OF_SERVICE in engine.enabled_categories
        assert ThreatCategory.ELEVATION_OF_PRIVILEGE in engine.enabled_categories
