# -*- coding: utf-8 -*-
"""
Threat Modeling Data Models - SEC-010 Phase 2

Pydantic v2 models for the GreenLang threat modeling system. Provides strongly-typed
data structures for threat models, components, data flows, trust boundaries,
threats, and mitigations.

All datetime fields use UTC. All models enforce strict validation via Pydantic v2
field validators and model configuration.

Models:
    - ThreatCategory: STRIDE threat category enumeration
    - ThreatStatus: Threat lifecycle status
    - ThreatModelStatus: Threat model lifecycle status
    - ComponentType: System component type enumeration
    - MitigationStatus: Mitigation implementation status
    - Component: System component in a threat model
    - DataFlow: Data flow between components
    - TrustBoundary: Trust boundary definition
    - Threat: Individual threat instance
    - Mitigation: Control mitigating a threat
    - ThreatModel: Complete threat model for a service

Example:
    >>> from greenlang.infrastructure.threat_modeling.models import (
    ...     ThreatModel, Component, ComponentType, ThreatCategory,
    ... )
    >>> component = Component(
    ...     name="API Gateway",
    ...     component_type=ComponentType.API,
    ...     trust_level=2,
    ...     data_classification="confidential",
    ... )
    >>> model = ThreatModel(
    ...     service_name="payment-service",
    ...     version="1.0.0",
    ...     components=[component],
    ... )

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SERVICE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
"""Valid service name: lowercase alphanumeric, dots, hyphens, underscores. 1-128 chars."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ThreatCategory(str, Enum):
    """STRIDE threat category enumeration.

    Each category represents a distinct type of security threat:
    - S: Spoofing identity
    - T: Tampering with data
    - R: Repudiation of actions
    - I: Information disclosure
    - D: Denial of service
    - E: Elevation of privilege
    """

    SPOOFING = "S"
    """Spoofing - Pretending to be someone or something else."""

    TAMPERING = "T"
    """Tampering - Modifying data or code without authorization."""

    REPUDIATION = "R"
    """Repudiation - Denying having performed an action."""

    INFORMATION_DISCLOSURE = "I"
    """Information Disclosure - Exposing information to unauthorized parties."""

    DENIAL_OF_SERVICE = "D"
    """Denial of Service - Preventing legitimate access to services."""

    ELEVATION_OF_PRIVILEGE = "E"
    """Elevation of Privilege - Gaining elevated access or capabilities."""

    @property
    def full_name(self) -> str:
        """Return the full name of the threat category."""
        names = {
            "S": "Spoofing",
            "T": "Tampering",
            "R": "Repudiation",
            "I": "Information Disclosure",
            "D": "Denial of Service",
            "E": "Elevation of Privilege",
        }
        return names.get(self.value, "Unknown")

    @property
    def description(self) -> str:
        """Return a detailed description of the threat category."""
        descriptions = {
            "S": "Attacks that involve impersonating a user, component, or system to gain unauthorized access.",
            "T": "Attacks that modify data, code, or configurations without proper authorization.",
            "R": "Ability to deny having performed an action when there is no proof otherwise.",
            "I": "Unauthorized access to or disclosure of sensitive information.",
            "D": "Attacks that prevent legitimate users from accessing the system or its resources.",
            "E": "Attacks that allow an attacker to gain elevated access rights or capabilities.",
        }
        return descriptions.get(self.value, "Unknown threat category.")


class ThreatStatus(str, Enum):
    """Threat lifecycle status enumeration.

    Tracks the progression of a threat from identification to resolution.
    """

    IDENTIFIED = "identified"
    """Threat has been identified but not yet analyzed."""

    ANALYZING = "analyzing"
    """Threat is being analyzed for likelihood and impact."""

    MITIGATED = "mitigated"
    """Threat has been mitigated through controls."""

    ACCEPTED = "accepted"
    """Residual risk has been accepted by stakeholders."""

    TRANSFERRED = "transferred"
    """Risk has been transferred (e.g., via insurance or contract)."""

    FALSE_POSITIVE = "false_positive"
    """Threat was incorrectly identified and is not applicable."""


class ThreatModelStatus(str, Enum):
    """Threat model lifecycle status enumeration."""

    DRAFT = "draft"
    """Model is in draft state, not yet complete."""

    IN_REVIEW = "in_review"
    """Model is under security team review."""

    APPROVED = "approved"
    """Model has been approved by security team."""

    REJECTED = "rejected"
    """Model was rejected and needs revision."""

    ARCHIVED = "archived"
    """Model has been superseded and archived."""


class ComponentType(str, Enum):
    """System component type enumeration.

    Defines the types of components that can be included in a threat model.
    Each type has associated default threat patterns.
    """

    WEB_APP = "web_app"
    """Web application (frontend, SPA, etc.)."""

    API = "api"
    """API service (REST, GraphQL, gRPC)."""

    DATABASE = "database"
    """Database (PostgreSQL, MySQL, etc.)."""

    MESSAGE_QUEUE = "message_queue"
    """Message queue (RabbitMQ, Kafka, SQS)."""

    EXTERNAL_SERVICE = "external_service"
    """External third-party service."""

    CACHE = "cache"
    """Caching layer (Redis, Memcached)."""

    FILE_STORAGE = "file_storage"
    """File storage (S3, local filesystem)."""

    LOAD_BALANCER = "load_balancer"
    """Load balancer or reverse proxy."""

    CONTAINER = "container"
    """Container or orchestration (Docker, K8s)."""

    SERVERLESS = "serverless"
    """Serverless function (Lambda, Cloud Functions)."""

    IDENTITY_PROVIDER = "identity_provider"
    """Identity provider (OAuth, SAML, OIDC)."""

    CDN = "cdn"
    """Content delivery network."""

    LOGGING = "logging"
    """Logging and monitoring service."""

    SECRET_MANAGER = "secret_manager"
    """Secrets management service."""

    NETWORK = "network"
    """Network component (VPC, firewall, VPN)."""

    USER = "user"
    """Human user or operator."""

    UNKNOWN = "unknown"
    """Unknown component type."""


class MitigationStatus(str, Enum):
    """Mitigation implementation status enumeration."""

    PROPOSED = "proposed"
    """Mitigation has been proposed but not started."""

    IN_PROGRESS = "in_progress"
    """Mitigation implementation is in progress."""

    IMPLEMENTED = "implemented"
    """Mitigation has been implemented."""

    VERIFIED = "verified"
    """Mitigation has been verified as effective."""

    REJECTED = "rejected"
    """Mitigation was rejected (not applicable or not feasible)."""


class DataClassification(str, Enum):
    """Data classification levels for components and flows."""

    PUBLIC = "public"
    """Publicly available data."""

    INTERNAL = "internal"
    """Internal use only."""

    CONFIDENTIAL = "confidential"
    """Confidential business data."""

    RESTRICTED = "restricted"
    """Highly restricted (PII, PHI, PCI)."""

    SECRET = "secret"
    """Trade secrets and critical security data."""


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------


class Component(BaseModel):
    """System component in a threat model.

    Represents a distinct element of the system architecture that
    can be analyzed for threats. Components have a type, trust level,
    and data classification that inform the threat analysis.

    Attributes:
        id: Unique identifier for the component.
        name: Human-readable component name.
        component_type: Type of the component (web_app, api, database, etc.).
        description: Detailed description of the component's purpose.
        trust_level: Trust level (1=untrusted, 2=semi-trusted, 3=trusted, 4=highly-trusted).
        data_classification: Classification of data handled by this component.
        owner: Team or individual responsible for this component.
        technology_stack: Technologies used (e.g., ["Python", "FastAPI", "PostgreSQL"]).
        metadata: Additional metadata for the component.
        created_at: Component creation timestamp.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique component identifier.",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable component name.",
    )
    component_type: ComponentType = Field(
        default=ComponentType.UNKNOWN,
        description="Type of the component.",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Detailed description of the component.",
    )
    trust_level: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Trust level (1=untrusted, 2=semi-trusted, 3=trusted, 4=highly-trusted).",
    )
    data_classification: DataClassification = Field(
        default=DataClassification.INTERNAL,
        description="Classification of data handled by this component.",
    )
    owner: str = Field(
        default="",
        max_length=256,
        description="Team or individual responsible for this component.",
    )
    technology_stack: List[str] = Field(
        default_factory=list,
        description="Technologies used by this component.",
    )
    is_external: bool = Field(
        default=False,
        description="Whether this is an external/third-party component.",
    )
    network_zone: str = Field(
        default="internal",
        max_length=64,
        description="Network zone (e.g., 'dmz', 'internal', 'external').",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the component.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Component creation timestamp (UTC).",
    )

    @field_validator("technology_stack")
    @classmethod
    def validate_technology_stack(cls, v: List[str]) -> List[str]:
        """Clean and deduplicate technology stack entries."""
        cleaned = [t.strip() for t in v if t.strip()]
        seen: set[str] = set()
        deduped: List[str] = []
        for tech in cleaned:
            if tech.lower() not in seen:
                seen.add(tech.lower())
                deduped.append(tech)
        return deduped

    @field_validator("network_zone")
    @classmethod
    def validate_network_zone(cls, v: str) -> str:
        """Normalize network zone to lowercase."""
        return v.strip().lower()


class DataFlow(BaseModel):
    """Data flow between components.

    Represents the movement of data between two components in the system.
    Data flows are analyzed for threats, especially when crossing trust boundaries.

    Attributes:
        id: Unique identifier for the data flow.
        source_component_id: ID of the source component.
        destination_component_id: ID of the destination component.
        data_type: Type of data being transferred.
        protocol: Communication protocol (e.g., HTTPS, gRPC, TCP).
        encryption: Whether the flow is encrypted in transit.
        authentication_required: Whether authentication is required.
        description: Description of the data flow.
        data_classification: Classification of data in this flow.
        bidirectional: Whether data flows in both directions.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique data flow identifier.",
    )
    source_component_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="ID of the source component.",
    )
    destination_component_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="ID of the destination component.",
    )
    data_type: str = Field(
        default="generic",
        max_length=256,
        description="Type of data being transferred (e.g., 'user_data', 'credentials').",
    )
    protocol: str = Field(
        default="https",
        max_length=64,
        description="Communication protocol (e.g., 'https', 'grpc', 'tcp').",
    )
    encryption: bool = Field(
        default=True,
        description="Whether the data flow is encrypted in transit.",
    )
    encryption_algorithm: str = Field(
        default="TLS 1.3",
        max_length=64,
        description="Encryption algorithm/protocol used.",
    )
    authentication_required: bool = Field(
        default=True,
        description="Whether authentication is required for this flow.",
    )
    authentication_method: str = Field(
        default="jwt",
        max_length=64,
        description="Authentication method (e.g., 'jwt', 'mtls', 'api_key').",
    )
    authorization_required: bool = Field(
        default=True,
        description="Whether authorization checks are performed.",
    )
    description: str = Field(
        default="",
        max_length=2048,
        description="Description of the data flow.",
    )
    data_classification: DataClassification = Field(
        default=DataClassification.INTERNAL,
        description="Classification of data in this flow.",
    )
    bidirectional: bool = Field(
        default=False,
        description="Whether data flows in both directions.",
    )
    crosses_trust_boundary: bool = Field(
        default=False,
        description="Whether this flow crosses a trust boundary.",
    )
    trust_boundary_id: Optional[str] = Field(
        default=None,
        description="ID of the trust boundary this flow crosses.",
    )
    port: Optional[int] = Field(
        default=None,
        ge=1,
        le=65535,
        description="Network port used for this flow.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the data flow.",
    )

    @field_validator("protocol")
    @classmethod
    def normalize_protocol(cls, v: str) -> str:
        """Normalize protocol to lowercase."""
        return v.strip().lower()

    @model_validator(mode="after")
    def validate_flow(self) -> "DataFlow":
        """Validate data flow consistency."""
        if self.source_component_id == self.destination_component_id:
            raise ValueError("Source and destination components must be different.")
        return self


class TrustBoundary(BaseModel):
    """Trust boundary definition.

    Represents a boundary where the trust level changes. Data flows
    crossing trust boundaries require additional security analysis.

    Attributes:
        id: Unique identifier for the trust boundary.
        name: Human-readable boundary name.
        description: Description of the trust boundary.
        component_ids: IDs of components within this boundary.
        trust_level: Trust level within this boundary.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique trust boundary identifier.",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable boundary name.",
    )
    description: str = Field(
        default="",
        max_length=2048,
        description="Description of the trust boundary.",
    )
    component_ids: List[str] = Field(
        default_factory=list,
        description="IDs of components within this boundary.",
    )
    trust_level: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Trust level within this boundary.",
    )
    boundary_type: str = Field(
        default="network",
        max_length=64,
        description="Type of boundary (e.g., 'network', 'process', 'privilege').",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the trust boundary.",
    )


class Threat(BaseModel):
    """Individual threat instance.

    Represents a specific threat identified during STRIDE analysis.
    Includes likelihood, impact, and risk scoring.

    Attributes:
        id: Unique identifier for the threat.
        category: STRIDE threat category.
        title: Brief title describing the threat.
        description: Detailed description of the threat.
        component_id: ID of the affected component.
        data_flow_id: ID of the affected data flow (if applicable).
        likelihood: Likelihood score (1-5).
        impact: Impact score (1-5).
        risk_score: Calculated risk score (0-10).
        severity: Severity level derived from risk score.
        status: Current threat status.
        mitigation_ids: IDs of mitigations addressing this threat.
        cvss_score: CVSS 3.1 base score (if applicable).
        cwe_ids: Related CWE identifiers.
        attack_vector: Attack vector description.
        prerequisites: Prerequisites for the attack.
        countermeasures: Recommended countermeasures.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique threat identifier.",
    )
    category: ThreatCategory = Field(
        ...,
        description="STRIDE threat category.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Brief title describing the threat.",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Detailed description of the threat.",
    )
    component_id: Optional[str] = Field(
        default=None,
        description="ID of the affected component.",
    )
    data_flow_id: Optional[str] = Field(
        default=None,
        description="ID of the affected data flow (if applicable).",
    )
    likelihood: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Likelihood score (1=rare, 2=unlikely, 3=possible, 4=likely, 5=almost certain).",
    )
    impact: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Impact score (1=insignificant, 2=minor, 3=moderate, 4=major, 5=catastrophic).",
    )
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Calculated risk score (0-10).",
    )
    severity: str = Field(
        default="medium",
        description="Severity level: critical, high, medium, low.",
    )
    status: ThreatStatus = Field(
        default=ThreatStatus.IDENTIFIED,
        description="Current threat status.",
    )
    mitigation_ids: List[str] = Field(
        default_factory=list,
        description="IDs of mitigations addressing this threat.",
    )
    cvss_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="CVSS 3.1 base score (if applicable).",
    )
    cvss_vector: Optional[str] = Field(
        default=None,
        max_length=256,
        description="CVSS 3.1 vector string.",
    )
    cwe_ids: List[str] = Field(
        default_factory=list,
        description="Related CWE identifiers (e.g., ['CWE-89', 'CWE-79']).",
    )
    attack_vector: str = Field(
        default="",
        max_length=1024,
        description="Description of the attack vector.",
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Prerequisites for the attack to succeed.",
    )
    countermeasures: List[str] = Field(
        default_factory=list,
        description="Recommended countermeasures.",
    )
    affected_assets: List[str] = Field(
        default_factory=list,
        description="Assets affected by this threat.",
    )
    threat_source: str = Field(
        default="stride_analysis",
        max_length=64,
        description="Source of threat identification.",
    )
    notes: str = Field(
        default="",
        max_length=4096,
        description="Additional notes about the threat.",
    )
    identified_by: str = Field(
        default="",
        max_length=256,
        description="Who identified this threat.",
    )
    identified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the threat was identified.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the threat.",
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate and normalize severity level."""
        allowed = {"critical", "high", "medium", "low"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(f"Invalid severity '{v}'. Allowed: {sorted(allowed)}")
        return v_lower

    @field_validator("cwe_ids")
    @classmethod
    def validate_cwe_ids(cls, v: List[str]) -> List[str]:
        """Validate CWE ID format."""
        validated = []
        for cwe in v:
            cwe_upper = cwe.strip().upper()
            if not cwe_upper.startswith("CWE-"):
                cwe_upper = f"CWE-{cwe_upper}"
            validated.append(cwe_upper)
        return validated

    @model_validator(mode="after")
    def calculate_severity_from_score(self) -> "Threat":
        """Update severity based on risk score if not explicitly set."""
        if self.risk_score > 0:
            if self.risk_score >= 8.0:
                object.__setattr__(self, "severity", "critical")
            elif self.risk_score >= 6.0:
                object.__setattr__(self, "severity", "high")
            elif self.risk_score >= 4.0:
                object.__setattr__(self, "severity", "medium")
            else:
                object.__setattr__(self, "severity", "low")
        return self


class Mitigation(BaseModel):
    """Control mitigating a threat.

    Represents a security control or measure that mitigates one or more threats.
    Mitigations are linked to security controls from SEC-001 to SEC-009.

    Attributes:
        id: Unique identifier for the mitigation.
        threat_id: ID of the threat this mitigates.
        control_id: ID of the security control implementing this mitigation.
        title: Brief title for the mitigation.
        description: Detailed description of the mitigation.
        status: Implementation status.
        effectiveness: Estimated effectiveness (0-100%).
        owner: Team or individual responsible.
        implemented_at: When the mitigation was implemented.
        verified_at: When the mitigation was verified.
        verified_by: Who verified the mitigation.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique mitigation identifier.",
    )
    threat_id: str = Field(
        ...,
        min_length=1,
        description="ID of the threat this mitigates.",
    )
    control_id: str = Field(
        default="",
        max_length=64,
        description="ID of the security control (e.g., 'SEC-001-JWT').",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Brief title for the mitigation.",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Detailed description of the mitigation.",
    )
    status: MitigationStatus = Field(
        default=MitigationStatus.PROPOSED,
        description="Implementation status.",
    )
    effectiveness: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Estimated effectiveness percentage (0-100).",
    )
    residual_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Residual risk after mitigation (0-10).",
    )
    owner: str = Field(
        default="",
        max_length=256,
        description="Team or individual responsible.",
    )
    priority: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Implementation priority (1=highest, 5=lowest).",
    )
    effort_estimate: str = Field(
        default="medium",
        max_length=64,
        description="Effort estimate: low, medium, high, very_high.",
    )
    implementation_notes: str = Field(
        default="",
        max_length=4096,
        description="Notes on implementation approach.",
    )
    verification_method: str = Field(
        default="",
        max_length=1024,
        description="How the mitigation will be verified.",
    )
    due_date: Optional[datetime] = Field(
        default=None,
        description="Target implementation date.",
    )
    implemented_at: Optional[datetime] = Field(
        default=None,
        description="When the mitigation was implemented.",
    )
    verified_at: Optional[datetime] = Field(
        default=None,
        description="When the mitigation was verified.",
    )
    verified_by: str = Field(
        default="",
        max_length=256,
        description="Who verified the mitigation.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Mitigation creation timestamp.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata.",
    )

    @field_validator("effort_estimate")
    @classmethod
    def validate_effort_estimate(cls, v: str) -> str:
        """Validate effort estimate value."""
        allowed = {"low", "medium", "high", "very_high"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(f"Invalid effort estimate '{v}'. Allowed: {sorted(allowed)}")
        return v_lower


class ThreatModel(BaseModel):
    """Complete threat model for a service.

    Represents a comprehensive threat model including components,
    data flows, trust boundaries, identified threats, and mitigations.

    Attributes:
        id: Unique identifier for the threat model.
        service_name: Name of the service being modeled.
        version: Version of the threat model.
        status: Lifecycle status of the model.
        description: Description of the system being modeled.
        components: List of system components.
        data_flows: List of data flows between components.
        trust_boundaries: List of trust boundaries.
        threats: List of identified threats.
        mitigations: List of mitigations.
        overall_risk_score: Aggregate risk score for the service.
        category_scores: Risk scores by STRIDE category.
        created_by: Who created the threat model.
        approved_by: Who approved the threat model.
        approved_at: When the threat model was approved.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique threat model identifier.",
    )
    service_name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Name of the service being modeled.",
    )
    version: str = Field(
        default="1.0.0",
        max_length=32,
        description="Version of the threat model.",
    )
    status: ThreatModelStatus = Field(
        default=ThreatModelStatus.DRAFT,
        description="Lifecycle status of the model.",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Description of the system being modeled.",
    )
    scope: str = Field(
        default="",
        max_length=2048,
        description="Scope and boundaries of the threat model.",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions made during modeling.",
    )
    out_of_scope: List[str] = Field(
        default_factory=list,
        description="Items explicitly out of scope.",
    )
    components: List[Component] = Field(
        default_factory=list,
        description="List of system components.",
    )
    data_flows: List[DataFlow] = Field(
        default_factory=list,
        description="List of data flows between components.",
    )
    trust_boundaries: List[TrustBoundary] = Field(
        default_factory=list,
        description="List of trust boundaries.",
    )
    threats: List[Threat] = Field(
        default_factory=list,
        description="List of identified threats.",
    )
    mitigations: List[Mitigation] = Field(
        default_factory=list,
        description="List of mitigations.",
    )
    overall_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Aggregate risk score for the service.",
    )
    category_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Risk scores by STRIDE category.",
    )
    threat_count: int = Field(
        default=0,
        ge=0,
        description="Total number of threats identified.",
    )
    mitigated_count: int = Field(
        default=0,
        ge=0,
        description="Number of threats with mitigations.",
    )
    critical_count: int = Field(
        default=0,
        ge=0,
        description="Number of critical severity threats.",
    )
    high_count: int = Field(
        default=0,
        ge=0,
        description="Number of high severity threats.",
    )
    owner: str = Field(
        default="",
        max_length=256,
        description="Team responsible for this threat model.",
    )
    reviewers: List[str] = Field(
        default_factory=list,
        description="People who reviewed this threat model.",
    )
    created_by: str = Field(
        default="",
        max_length=256,
        description="Who created the threat model.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Threat model creation timestamp.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp.",
    )
    approved_by: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Who approved the threat model.",
    )
    approved_at: Optional[datetime] = Field(
        default=None,
        description="When the threat model was approved.",
    )
    next_review_date: Optional[datetime] = Field(
        default=None,
        description="Scheduled date for next review.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata.",
    )

    @field_validator("service_name")
    @classmethod
    def validate_service_name(cls, v: str) -> str:
        """Validate service name format."""
        v_lower = v.strip().lower()
        if not _SERVICE_NAME_PATTERN.match(v_lower):
            raise ValueError(
                f"Service name '{v}' is invalid. Must be lowercase alphanumeric "
                "with dots, hyphens, underscores. 1-128 chars."
            )
        return v_lower

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Clean and deduplicate tags."""
        cleaned = [t.strip().lower() for t in v if t.strip()]
        seen: set[str] = set()
        deduped: List[str] = []
        for tag in cleaned:
            if tag not in seen:
                seen.add(tag)
                deduped.append(tag)
        return deduped

    @model_validator(mode="after")
    def update_counts(self) -> "ThreatModel":
        """Update threat counts from the threats list."""
        if self.threats:
            object.__setattr__(self, "threat_count", len(self.threats))
            mitigated = sum(1 for t in self.threats if t.status == ThreatStatus.MITIGATED)
            object.__setattr__(self, "mitigated_count", mitigated)
            critical = sum(1 for t in self.threats if t.severity == "critical")
            object.__setattr__(self, "critical_count", critical)
            high = sum(1 for t in self.threats if t.severity == "high")
            object.__setattr__(self, "high_count", high)
        return self

    def get_component_by_id(self, component_id: str) -> Optional[Component]:
        """Find a component by its ID.

        Args:
            component_id: The component ID to find.

        Returns:
            The Component if found, None otherwise.
        """
        for comp in self.components:
            if comp.id == component_id:
                return comp
        return None

    def get_threats_for_component(self, component_id: str) -> List[Threat]:
        """Get all threats affecting a specific component.

        Args:
            component_id: The component ID to filter by.

        Returns:
            List of threats affecting the component.
        """
        return [t for t in self.threats if t.component_id == component_id]

    def get_unmitigated_threats(self) -> List[Threat]:
        """Get all threats that are not yet mitigated.

        Returns:
            List of unmitigated threats.
        """
        return [
            t for t in self.threats
            if t.status not in (ThreatStatus.MITIGATED, ThreatStatus.ACCEPTED, ThreatStatus.TRANSFERRED)
        ]

    def get_critical_threats(self) -> List[Threat]:
        """Get all critical severity threats.

        Returns:
            List of critical threats.
        """
        return [t for t in self.threats if t.severity == "critical"]


__all__ = [
    "ThreatCategory",
    "ThreatStatus",
    "ThreatModelStatus",
    "ComponentType",
    "MitigationStatus",
    "DataClassification",
    "Component",
    "DataFlow",
    "TrustBoundary",
    "Threat",
    "Mitigation",
    "ThreatModel",
]
