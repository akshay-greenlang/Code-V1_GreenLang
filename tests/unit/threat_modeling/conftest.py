"""
Test fixtures for threat_modeling module.

Provides mock components, data flows, trust boundaries, and threat data
for comprehensive unit testing of threat modeling components.
"""

import pytest
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from greenlang.infrastructure.threat_modeling.models import (
    Component,
    ComponentType,
    DataFlow,
    TrustBoundary,
    Threat,
    ThreatCategory,
    RiskLevel,
    CVSSVector,
    ThreatModel,
    Mitigation,
    MitigationStatus,
)
from greenlang.infrastructure.threat_modeling.config import (
    ThreatModelingConfig,
    STRIDEConfig,
    RiskScorerConfig,
)


# -----------------------------------------------------------------------------
# Component Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_component() -> Component:
    """Create a sample component for testing."""
    return Component(
        component_id=str(uuid4()),
        name="greenlang-api",
        component_type=ComponentType.SERVICE,
        description="Main API service for GreenLang platform",
        technology_stack=["Python", "FastAPI", "PostgreSQL"],
        data_classification="confidential",
        authentication_required=True,
        authorization_level="role_based",
        exposed_ports=[8000],
        protocols=["HTTPS", "gRPC"],
        metadata={
            "team": "platform",
            "environment": "production",
            "criticality": "high",
        },
    )


@pytest.fixture
def database_component() -> Component:
    """Create a database component."""
    return Component(
        component_id=str(uuid4()),
        name="greenlang-db",
        component_type=ComponentType.DATABASE,
        description="PostgreSQL database for GreenLang",
        technology_stack=["PostgreSQL", "TimescaleDB"],
        data_classification="restricted",
        authentication_required=True,
        authorization_level="connection_string",
        exposed_ports=[5432],
        protocols=["TCP"],
        metadata={
            "encryption_at_rest": True,
            "backup_frequency": "daily",
        },
    )


@pytest.fixture
def external_api_component() -> Component:
    """Create an external API component."""
    return Component(
        component_id=str(uuid4()),
        name="payment-gateway",
        component_type=ComponentType.EXTERNAL_SERVICE,
        description="External payment processing service",
        technology_stack=["REST API"],
        data_classification="pci",
        authentication_required=True,
        authorization_level="api_key",
        exposed_ports=[443],
        protocols=["HTTPS"],
        metadata={
            "vendor": "stripe",
            "pci_compliant": True,
        },
    )


@pytest.fixture
def user_component() -> Component:
    """Create a user/actor component."""
    return Component(
        component_id=str(uuid4()),
        name="authenticated-user",
        component_type=ComponentType.USER,
        description="Authenticated user accessing the system",
        technology_stack=["Browser", "Mobile App"],
        data_classification="public",
        authentication_required=True,
        authorization_level="jwt",
        exposed_ports=[],
        protocols=["HTTPS"],
        metadata={
            "user_types": ["admin", "standard", "viewer"],
        },
    )


@pytest.fixture
def multiple_components(
    sample_component,
    database_component,
    external_api_component,
    user_component,
) -> List[Component]:
    """Create a list of multiple components."""
    return [
        sample_component,
        database_component,
        external_api_component,
        user_component,
    ]


# -----------------------------------------------------------------------------
# Data Flow Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def api_to_db_flow(sample_component, database_component) -> DataFlow:
    """Create a data flow from API to database."""
    return DataFlow(
        flow_id=str(uuid4()),
        name="API to Database",
        source_component_id=sample_component.component_id,
        destination_component_id=database_component.component_id,
        data_types=["user_data", "emission_records", "audit_logs"],
        protocol="TCP",
        encrypted=True,
        authentication="connection_pool",
        data_classification="confidential",
        metadata={
            "connection_pooling": True,
            "max_connections": 100,
        },
    )


@pytest.fixture
def user_to_api_flow(user_component, sample_component) -> DataFlow:
    """Create a data flow from user to API."""
    return DataFlow(
        flow_id=str(uuid4()),
        name="User to API",
        source_component_id=user_component.component_id,
        destination_component_id=sample_component.component_id,
        data_types=["api_requests", "authentication_tokens"],
        protocol="HTTPS",
        encrypted=True,
        authentication="jwt",
        data_classification="confidential",
        metadata={
            "rate_limited": True,
            "cors_enabled": True,
        },
    )


@pytest.fixture
def api_to_external_flow(sample_component, external_api_component) -> DataFlow:
    """Create a data flow from API to external service."""
    return DataFlow(
        flow_id=str(uuid4()),
        name="API to Payment Gateway",
        source_component_id=sample_component.component_id,
        destination_component_id=external_api_component.component_id,
        data_types=["payment_data", "transaction_info"],
        protocol="HTTPS",
        encrypted=True,
        authentication="api_key",
        data_classification="pci",
        metadata={
            "retry_policy": "exponential_backoff",
            "timeout_seconds": 30,
        },
    )


@pytest.fixture
def multiple_data_flows(
    api_to_db_flow,
    user_to_api_flow,
    api_to_external_flow,
) -> List[DataFlow]:
    """Create a list of multiple data flows."""
    return [api_to_db_flow, user_to_api_flow, api_to_external_flow]


# -----------------------------------------------------------------------------
# Trust Boundary Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def network_boundary() -> TrustBoundary:
    """Create a network trust boundary."""
    return TrustBoundary(
        boundary_id=str(uuid4()),
        name="Internal Network Boundary",
        boundary_type="network",
        description="Boundary between public internet and internal network",
        components_inside=[],  # Will be populated with component IDs
        components_outside=[],
        controls=[
            "firewall",
            "waf",
            "ids",
            "load_balancer",
        ],
        metadata={
            "zone": "dmz",
            "compliance": ["soc2", "pci"],
        },
    )


@pytest.fixture
def authentication_boundary() -> TrustBoundary:
    """Create an authentication trust boundary."""
    return TrustBoundary(
        boundary_id=str(uuid4()),
        name="Authentication Boundary",
        boundary_type="authentication",
        description="Boundary requiring user authentication",
        components_inside=[],
        components_outside=[],
        controls=[
            "mfa",
            "session_management",
            "jwt_validation",
        ],
        metadata={
            "session_timeout_minutes": 30,
        },
    )


@pytest.fixture
def data_classification_boundary() -> TrustBoundary:
    """Create a data classification trust boundary."""
    return TrustBoundary(
        boundary_id=str(uuid4()),
        name="PCI Data Zone",
        boundary_type="data_classification",
        description="Boundary for PCI-sensitive data",
        components_inside=[],
        components_outside=[],
        controls=[
            "encryption",
            "access_logging",
            "data_masking",
            "key_rotation",
        ],
        metadata={
            "compliance_framework": "pci-dss",
            "audit_frequency": "quarterly",
        },
    )


@pytest.fixture
def multiple_boundaries(
    network_boundary,
    authentication_boundary,
    data_classification_boundary,
) -> List[TrustBoundary]:
    """Create a list of multiple trust boundaries."""
    return [
        network_boundary,
        authentication_boundary,
        data_classification_boundary,
    ]


# -----------------------------------------------------------------------------
# Threat Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def spoofing_threat() -> Threat:
    """Create a spoofing threat."""
    return Threat(
        threat_id=str(uuid4()),
        title="Identity Spoofing via Token Theft",
        description="Attacker steals JWT tokens to impersonate legitimate users",
        category=ThreatCategory.SPOOFING,
        affected_component_ids=[],
        affected_data_flow_ids=[],
        attack_vector="Network-based token interception",
        prerequisites=["Access to network traffic", "Weak token security"],
        potential_impact="Unauthorized access to user data and actions",
        likelihood=0.6,
        impact=0.8,
        risk_score=0.0,  # Will be calculated
        risk_level=RiskLevel.HIGH,
        mitigations=[],
        status="identified",
        identified_at=datetime.utcnow(),
        identified_by="threat_modeling_system",
    )


@pytest.fixture
def tampering_threat() -> Threat:
    """Create a tampering threat."""
    return Threat(
        threat_id=str(uuid4()),
        title="Data Tampering in Transit",
        description="Attacker modifies data during transmission between components",
        category=ThreatCategory.TAMPERING,
        affected_component_ids=[],
        affected_data_flow_ids=[],
        attack_vector="Man-in-the-middle attack",
        prerequisites=["Network access", "Weak encryption"],
        potential_impact="Data integrity compromise",
        likelihood=0.4,
        impact=0.9,
        risk_score=0.0,
        risk_level=RiskLevel.HIGH,
        mitigations=[],
        status="identified",
        identified_at=datetime.utcnow(),
        identified_by="threat_modeling_system",
    )


@pytest.fixture
def repudiation_threat() -> Threat:
    """Create a repudiation threat."""
    return Threat(
        threat_id=str(uuid4()),
        title="Transaction Repudiation",
        description="User denies performing a transaction due to insufficient logging",
        category=ThreatCategory.REPUDIATION,
        affected_component_ids=[],
        affected_data_flow_ids=[],
        attack_vector="Exploitation of logging gaps",
        prerequisites=["Insufficient audit trails"],
        potential_impact="Inability to prove transaction authenticity",
        likelihood=0.3,
        impact=0.6,
        risk_score=0.0,
        risk_level=RiskLevel.MEDIUM,
        mitigations=[],
        status="identified",
        identified_at=datetime.utcnow(),
        identified_by="threat_modeling_system",
    )


@pytest.fixture
def information_disclosure_threat() -> Threat:
    """Create an information disclosure threat."""
    return Threat(
        threat_id=str(uuid4()),
        title="Sensitive Data Exposure",
        description="Sensitive data exposed through error messages or logs",
        category=ThreatCategory.INFORMATION_DISCLOSURE,
        affected_component_ids=[],
        affected_data_flow_ids=[],
        attack_vector="Error message analysis, log scraping",
        prerequisites=["Access to logs or error responses"],
        potential_impact="Exposure of PII, credentials, or business data",
        likelihood=0.5,
        impact=0.7,
        risk_score=0.0,
        risk_level=RiskLevel.HIGH,
        mitigations=[],
        status="identified",
        identified_at=datetime.utcnow(),
        identified_by="threat_modeling_system",
    )


@pytest.fixture
def dos_threat() -> Threat:
    """Create a denial of service threat."""
    return Threat(
        threat_id=str(uuid4()),
        title="API Rate Limiting Bypass",
        description="Attacker bypasses rate limiting to cause service degradation",
        category=ThreatCategory.DENIAL_OF_SERVICE,
        affected_component_ids=[],
        affected_data_flow_ids=[],
        attack_vector="Distributed requests, IP rotation",
        prerequisites=["Botnet access", "Rate limiting weaknesses"],
        potential_impact="Service unavailability, increased costs",
        likelihood=0.7,
        impact=0.6,
        risk_score=0.0,
        risk_level=RiskLevel.HIGH,
        mitigations=[],
        status="identified",
        identified_at=datetime.utcnow(),
        identified_by="threat_modeling_system",
    )


@pytest.fixture
def elevation_of_privilege_threat() -> Threat:
    """Create an elevation of privilege threat."""
    return Threat(
        threat_id=str(uuid4()),
        title="Privilege Escalation via RBAC Bypass",
        description="User gains elevated privileges through RBAC configuration flaws",
        category=ThreatCategory.ELEVATION_OF_PRIVILEGE,
        affected_component_ids=[],
        affected_data_flow_ids=[],
        attack_vector="Exploiting role assignment logic",
        prerequisites=["Valid user account", "RBAC misconfiguration"],
        potential_impact="Unauthorized administrative access",
        likelihood=0.4,
        impact=0.95,
        risk_score=0.0,
        risk_level=RiskLevel.CRITICAL,
        mitigations=[],
        status="identified",
        identified_at=datetime.utcnow(),
        identified_by="threat_modeling_system",
    )


@pytest.fixture
def all_stride_threats(
    spoofing_threat,
    tampering_threat,
    repudiation_threat,
    information_disclosure_threat,
    dos_threat,
    elevation_of_privilege_threat,
) -> List[Threat]:
    """Create all STRIDE category threats."""
    return [
        spoofing_threat,
        tampering_threat,
        repudiation_threat,
        information_disclosure_threat,
        dos_threat,
        elevation_of_privilege_threat,
    ]


# -----------------------------------------------------------------------------
# CVSS Vector Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def high_cvss_vector() -> CVSSVector:
    """Create a high severity CVSS vector."""
    return CVSSVector(
        attack_vector="NETWORK",
        attack_complexity="LOW",
        privileges_required="NONE",
        user_interaction="NONE",
        scope="CHANGED",
        confidentiality_impact="HIGH",
        integrity_impact="HIGH",
        availability_impact="HIGH",
    )


@pytest.fixture
def medium_cvss_vector() -> CVSSVector:
    """Create a medium severity CVSS vector."""
    return CVSSVector(
        attack_vector="NETWORK",
        attack_complexity="HIGH",
        privileges_required="LOW",
        user_interaction="REQUIRED",
        scope="UNCHANGED",
        confidentiality_impact="LOW",
        integrity_impact="LOW",
        availability_impact="NONE",
    )


@pytest.fixture
def low_cvss_vector() -> CVSSVector:
    """Create a low severity CVSS vector."""
    return CVSSVector(
        attack_vector="LOCAL",
        attack_complexity="HIGH",
        privileges_required="HIGH",
        user_interaction="REQUIRED",
        scope="UNCHANGED",
        confidentiality_impact="NONE",
        integrity_impact="LOW",
        availability_impact="NONE",
    )


# -----------------------------------------------------------------------------
# Mitigation Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_mitigation() -> Mitigation:
    """Create a sample mitigation."""
    return Mitigation(
        mitigation_id=str(uuid4()),
        title="Implement Token Binding",
        description="Bind JWT tokens to client certificates to prevent theft",
        threat_ids=[],
        control_type="preventive",
        implementation_effort="medium",
        effectiveness=0.85,
        status=MitigationStatus.PLANNED,
        owner="security-team",
        due_date=datetime.utcnow(),
        metadata={
            "related_controls": ["AC-17", "IA-5"],
        },
    )


@pytest.fixture
def implemented_mitigation() -> Mitigation:
    """Create an implemented mitigation."""
    return Mitigation(
        mitigation_id=str(uuid4()),
        title="TLS 1.3 Encryption",
        description="Enforce TLS 1.3 for all data in transit",
        threat_ids=[],
        control_type="preventive",
        implementation_effort="low",
        effectiveness=0.95,
        status=MitigationStatus.IMPLEMENTED,
        owner="platform-team",
        due_date=None,
        implemented_at=datetime.utcnow(),
        metadata={
            "cipher_suites": ["TLS_AES_256_GCM_SHA384"],
        },
    )


# -----------------------------------------------------------------------------
# Threat Model Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_threat_model(
    multiple_components,
    multiple_data_flows,
    multiple_boundaries,
    all_stride_threats,
) -> ThreatModel:
    """Create a complete threat model."""
    return ThreatModel(
        model_id=str(uuid4()),
        name="GreenLang Platform Threat Model",
        version="1.0.0",
        description="Comprehensive threat model for GreenLang platform",
        scope="Production environment",
        components=multiple_components,
        data_flows=multiple_data_flows,
        trust_boundaries=multiple_boundaries,
        threats=all_stride_threats,
        overall_risk_score=0.65,
        overall_risk_level=RiskLevel.HIGH,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        created_by="security-team",
        status="active",
        metadata={
            "review_frequency": "quarterly",
            "last_review": datetime.utcnow().isoformat(),
        },
    )


# -----------------------------------------------------------------------------
# Configuration Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def stride_config() -> STRIDEConfig:
    """Create STRIDE engine configuration."""
    return STRIDEConfig(
        enabled_categories=[
            ThreatCategory.SPOOFING,
            ThreatCategory.TAMPERING,
            ThreatCategory.REPUDIATION,
            ThreatCategory.INFORMATION_DISCLOSURE,
            ThreatCategory.DENIAL_OF_SERVICE,
            ThreatCategory.ELEVATION_OF_PRIVILEGE,
        ],
        component_type_mappings={
            ComponentType.SERVICE: [
                ThreatCategory.SPOOFING,
                ThreatCategory.TAMPERING,
                ThreatCategory.DENIAL_OF_SERVICE,
            ],
            ComponentType.DATABASE: [
                ThreatCategory.TAMPERING,
                ThreatCategory.INFORMATION_DISCLOSURE,
            ],
            ComponentType.USER: [
                ThreatCategory.SPOOFING,
                ThreatCategory.REPUDIATION,
            ],
        },
        threat_library_path="/etc/greenlang/threat_library.yaml",
    )


@pytest.fixture
def risk_scorer_config() -> RiskScorerConfig:
    """Create risk scorer configuration."""
    return RiskScorerConfig(
        likelihood_weights={
            "attack_complexity": 0.3,
            "privileges_required": 0.2,
            "user_interaction": 0.2,
            "prerequisites": 0.3,
        },
        impact_weights={
            "confidentiality": 0.35,
            "integrity": 0.35,
            "availability": 0.30,
        },
        risk_thresholds={
            RiskLevel.CRITICAL: 0.9,
            RiskLevel.HIGH: 0.7,
            RiskLevel.MEDIUM: 0.4,
            RiskLevel.LOW: 0.0,
        },
        business_context_multipliers={
            "production": 1.5,
            "staging": 1.0,
            "development": 0.5,
        },
    )


@pytest.fixture
def threat_modeling_config(
    stride_config,
    risk_scorer_config,
) -> ThreatModelingConfig:
    """Create full threat modeling configuration."""
    return ThreatModelingConfig(
        stride=stride_config,
        risk_scorer=risk_scorer_config,
        auto_scan_enabled=True,
        scan_interval_hours=24,
        notification_threshold=RiskLevel.HIGH,
        retention_days=365,
    )


# -----------------------------------------------------------------------------
# Mock Service Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_threat_library():
    """Create mock threat library."""
    library = MagicMock()
    library.get_threats_for_component.return_value = []
    library.get_threats_for_data_flow.return_value = []
    library.get_mitigations_for_threat.return_value = []
    return library


@pytest.fixture
def mock_database():
    """Create mock database for threat model persistence."""
    db = AsyncMock()
    db.save_threat_model.return_value = True
    db.get_threat_model.return_value = None
    db.list_threat_models.return_value = []
    db.save_threat.return_value = True
    return db


# -----------------------------------------------------------------------------
# API Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def security_analyst_headers() -> Dict[str, str]:
    """Create headers for security analyst role."""
    return {
        "Authorization": "Bearer test-security-analyst-token",
        "X-User-Id": "analyst-user",
        "X-User-Roles": "security-analyst,viewer",
    }


@pytest.fixture
def threat_model_admin_headers() -> Dict[str, str]:
    """Create headers for threat model admin role."""
    return {
        "Authorization": "Bearer test-threat-model-admin-token",
        "X-User-Id": "tm-admin",
        "X-User-Roles": "threat-model-admin,security-analyst,viewer",
    }
