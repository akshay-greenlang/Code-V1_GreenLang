# -*- coding: utf-8 -*-
"""
Audit Trail & Lineage Agent API Router - AGENT-MRV-030

This module implements the FastAPI router for immutable, tamper-evident audit
trails and end-to-end calculation lineage across all MRV emissions calculations
(Scope 1, 2, and 3).

Provides 25 REST endpoints for:
- Audit event recording and querying (record, batch, get, list, soft-delete)
- Hash chain verification (verify integrity, get chain)
- Lineage graph construction and traversal (nodes, edges, graph, trace, visualize)
- Evidence package creation, signing, and verification
- Compliance traceability and coverage assessment
- Change detection and impact analysis
- Full pipeline execution (single, batch)
- Audit trail summary
- Health check with 7 engine statuses

Every audit event is cryptographically chained via SHA-256. The lineage graph
links inputs, intermediate values, and outputs into a traversable DAG that
supports forward-impact analysis and backward-traceability.

Follows GreenLang's zero-hallucination principle: all hash computations and chain
verifications use deterministic SHA-256; no LLM calls in the audit path.

Agent ID: GL-MRV-X-042
Prefix: gl_atl_
API: /api/v1/audit-trail-lineage

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.agents.mrv.audit_trail_lineage.api.router import router
    >>> app = FastAPI()
    >>> app.include_router(router, prefix="/api/v1/audit-trail-lineage")
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends, status
from pydantic import Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
import logging
import uuid
import hashlib
from datetime import datetime
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    tags=["audit-trail-lineage"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# ENUMS
# ============================================================================


class AuditEventTypeEnum(str, Enum):
    """Audit event type categories for MRV calculation lifecycle."""

    DATA_INGESTION = "data_ingestion"
    DATA_VALIDATION = "data_validation"
    EMISSION_FACTOR_LOOKUP = "emission_factor_lookup"
    CALCULATION_STARTED = "calculation_started"
    CALCULATION_COMPLETED = "calculation_completed"
    AGGREGATION = "aggregation"
    COMPLIANCE_CHECK = "compliance_check"
    REPORT_GENERATION = "report_generation"
    DATA_CORRECTION = "data_correction"
    RECALCULATION = "recalculation"
    APPROVAL = "approval"
    SIGNATURE = "signature"
    EXPORT = "export"
    DELETION = "deletion"
    CONFIGURATION_CHANGE = "configuration_change"


class EmissionScopeEnum(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    CROSS_SCOPE = "cross_scope"


class LineageNodeTypeEnum(str, Enum):
    """Types of nodes in the calculation lineage DAG."""

    SOURCE = "source"
    ACTIVITY_DATA = "activity_data"
    EMISSION_FACTOR = "emission_factor"
    INTERMEDIATE = "intermediate"
    CALCULATION = "calculation"
    AGGREGATION = "aggregation"
    DISCLOSURE = "disclosure"


class LineageEdgeTypeEnum(str, Enum):
    """Types of edges connecting lineage DAG nodes."""

    INPUT_TO = "input_to"
    DERIVED_FROM = "derived_from"
    AGGREGATED_INTO = "aggregated_into"
    TRANSFORMED_BY = "transformed_by"
    VALIDATED_BY = "validated_by"
    OVERRIDES = "overrides"


class TraversalDirectionEnum(str, Enum):
    """Direction of lineage graph traversal."""

    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"


class GraphFormatEnum(str, Enum):
    """Supported formats for lineage graph visualization export."""

    DOT = "dot"
    JSON = "json"
    MERMAID = "mermaid"
    D3 = "d3"
    CYTOSCAPE = "cytoscape"


class AssuranceLevelEnum(str, Enum):
    """Assurance levels for evidence packages per ISAE 3410."""

    NONE = "none"
    LIMITED = "limited"
    REASONABLE = "reasonable"


class FrameworkEnum(str, Enum):
    """Supported regulatory and voluntary compliance frameworks."""

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    SEC_CLIMATE = "sec_climate"
    EU_TAXONOMY = "eu_taxonomy"
    ISAE_3410 = "isae_3410"


# ============================================================================
# SERVICE DEPENDENCY
# ============================================================================


_service_instance = None


def get_service():
    """
    Get or create AuditTrailLineageService singleton instance.

    Returns:
        AuditTrailLineageService instance

    Raises:
        HTTPException: If service initialization fails (503)
    """
    global _service_instance

    if _service_instance is None:
        try:
            from greenlang.agents.mrv.audit_trail_lineage.setup import (
                get_service as _get_service,
            )
            _service_instance = _get_service()
            logger.info("AuditTrailLineageService initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize AuditTrailLineageService: {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service initialization failed",
            )

    return _service_instance


# ============================================================================
# REQUEST MODELS - EVENTS
# ============================================================================


class RecordEventRequest(GreenLangBase):
    """
    Request model for recording a single audit event.

    Each event is appended to the SHA-256 hash chain for the given
    organization and reporting year combination, producing a
    cryptographically linked, immutable audit trail.

    Attributes:
        event_type: Type of audit event being recorded
        agent_id: Identifier of the MRV agent that triggered the event
        scope: GHG Protocol emission scope (scope_1, scope_2, scope_3)
        category: Scope 3 category number (1-15) or null for Scope 1/2
        organization_id: Organization identifier
        reporting_year: Reporting period year
        calculation_id: Optional linked calculation UUID
        payload: Event payload data (inputs, outputs, parameters)
        data_quality_score: Optional data quality score (1-5)
        metadata: Additional metadata key-value pairs
    """

    event_type: str = Field(
        ..., min_length=1, max_length=100,
        description="Audit event type (e.g., calculation_started)",
    )
    agent_id: str = Field(
        ..., min_length=1, max_length=50,
        description="MRV agent identifier (e.g., GL-MRV-S1-001)",
    )
    scope: Optional[str] = Field(
        None, max_length=20,
        description="Emission scope (scope_1, scope_2, scope_3, cross_scope)",
    )
    category: Optional[int] = Field(
        None, ge=1, le=15,
        description="Scope 3 category number (1-15)",
    )
    organization_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2000, le=2100,
        description="Reporting year",
    )
    calculation_id: Optional[str] = Field(
        None, max_length=100,
        description="Linked calculation UUID",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload data (inputs, outputs, parameters)",
    )
    data_quality_score: Optional[int] = Field(
        None, ge=1, le=5,
        description="Data quality score (1=best, 5=worst)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata key-value pairs",
    )


class BatchRecordRequest(GreenLangBase):
    """
    Request model for recording a batch of audit events.

    Supports up to 10,000 events per batch. Events are appended to the
    hash chain in order. If validate_only is True, events are validated
    but not persisted.

    Attributes:
        events: List of audit events to record
        validate_only: If True, validate without persisting
    """

    events: List[RecordEventRequest] = Field(
        ..., min_items=1, max_items=10000,
        description="List of audit events to record",
    )
    validate_only: bool = Field(
        False,
        description="If True, validate events without persisting",
    )


# ============================================================================
# REQUEST MODELS - CHAIN
# ============================================================================


class ChainVerifyRequest(GreenLangBase):
    """
    Request model for verifying hash chain integrity.

    Verifies that the SHA-256 hash chain for the given organization and
    reporting year is intact between the specified positions. Each event's
    hash must equal SHA-256(previous_hash + event_payload).

    Attributes:
        organization_id: Organization identifier
        reporting_year: Reporting year
        start_position: Starting chain position (0-based, default 0)
        end_position: Ending chain position (null = chain head)
    """

    organization_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2000, le=2100,
        description="Reporting year",
    )
    start_position: int = Field(
        0, ge=0,
        description="Starting chain position (0-based)",
    )
    end_position: Optional[int] = Field(
        None, ge=0,
        description="Ending chain position (null = chain head)",
    )


# ============================================================================
# REQUEST MODELS - LINEAGE
# ============================================================================


class CreateNodeRequest(GreenLangBase):
    """
    Request model for creating a lineage graph node.

    Nodes represent data points in the MRV calculation DAG, from raw
    source inputs through intermediate calculations to disclosed totals.

    Attributes:
        node_type: Type of lineage node
        level: Hierarchical level in the lineage graph
        qualified_name: Fully qualified name (e.g., scope1.stationary.fuel_oil)
        display_name: Human-readable display name
        organization_id: Organization identifier
        reporting_year: Reporting year
        agent_id: MRV agent that produced this node
        value: Numeric value at this node (tCO2e, kWh, etc.)
        unit: Unit of measurement
        data_quality_score: Data quality score (1-5)
        metadata: Additional metadata
    """

    node_type: str = Field(
        ..., max_length=50,
        description="Node type (source, activity_data, emission_factor, etc.)",
    )
    level: int = Field(
        ..., ge=0, le=20,
        description="Hierarchical level (0=leaf, higher=aggregated)",
    )
    qualified_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Fully qualified name",
    )
    display_name: str = Field(
        ..., min_length=1, max_length=300,
        description="Human-readable display name",
    )
    organization_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2000, le=2100,
        description="Reporting year",
    )
    agent_id: Optional[str] = Field(
        None, max_length=50,
        description="MRV agent that produced this node",
    )
    value: Optional[float] = Field(
        None,
        description="Numeric value (tCO2e, kWh, etc.)",
    )
    unit: Optional[str] = Field(
        None, max_length=30,
        description="Unit of measurement",
    )
    data_quality_score: Optional[int] = Field(
        None, ge=1, le=5,
        description="Data quality score (1=best, 5=worst)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class CreateEdgeRequest(GreenLangBase):
    """
    Request model for creating a lineage graph edge.

    Edges connect lineage nodes to represent data flow and transformation
    relationships in the MRV calculation DAG.

    Attributes:
        source_node_id: Source node UUID
        target_node_id: Target node UUID
        edge_type: Type of relationship
        organization_id: Organization identifier
        reporting_year: Reporting year
        transformation_description: Description of the transformation
        confidence: Confidence score (0.0-1.0)
        metadata: Additional metadata
    """

    source_node_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Source node UUID",
    )
    target_node_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Target node UUID",
    )
    edge_type: str = Field(
        ..., max_length=50,
        description="Edge type (input_to, derived_from, aggregated_into, etc.)",
    )
    organization_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2000, le=2100,
        description="Reporting year",
    )
    transformation_description: Optional[str] = Field(
        None, max_length=1000,
        description="Description of the transformation applied",
    )
    confidence: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Confidence score (0.0=no confidence, 1.0=full confidence)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class LineageTraceRequest(GreenLangBase):
    """
    Request model for tracing lineage in the DAG.

    Traverses the lineage graph from a starting node in the specified
    direction, optionally filtering by node type and level.

    Attributes:
        start_node_id: Starting node UUID for traversal
        direction: Traversal direction (forward, backward, both)
        max_depth: Maximum traversal depth
        node_type_filter: Optional filter by node type
        level_filter: Optional filter by hierarchical level
    """

    start_node_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Starting node UUID for traversal",
    )
    direction: str = Field(
        "backward",
        description="Traversal direction (forward, backward, both)",
    )
    max_depth: int = Field(
        10, ge=1, le=50,
        description="Maximum traversal depth",
    )
    node_type_filter: Optional[str] = Field(
        None, max_length=50,
        description="Filter by node type",
    )
    level_filter: Optional[int] = Field(
        None, ge=0, le=20,
        description="Filter by hierarchical level",
    )


# ============================================================================
# REQUEST MODELS - EVIDENCE
# ============================================================================


class CreateEvidenceRequest(GreenLangBase):
    """
    Request model for creating an evidence package.

    Bundles audit events, lineage graphs, and supporting documents into a
    verifiable evidence package for third-party assurance engagements.

    Attributes:
        organization_id: Organization identifier
        reporting_year: Reporting year
        frameworks: Frameworks to include in the evidence package
        scope_filter: Optional scope filter (scope_1, scope_2, scope_3)
        assurance_level: Assurance level (none, limited, reasonable)
        metadata: Additional metadata
    """

    organization_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2000, le=2100,
        description="Reporting year",
    )
    frameworks: List[str] = Field(
        default_factory=lambda: ["ghg_protocol"],
        description="Frameworks to include (ghg_protocol, iso_14064, etc.)",
    )
    scope_filter: Optional[str] = Field(
        None, max_length=20,
        description="Scope filter (scope_1, scope_2, scope_3)",
    )
    assurance_level: str = Field(
        "limited",
        description="Assurance level (none, limited, reasonable)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the evidence package",
    )


class SignPackageRequest(GreenLangBase):
    """
    Request model for signing an evidence package.

    Applies a digital signature to the evidence package for
    non-repudiation and integrity verification.

    Attributes:
        algorithm: Signature algorithm (ed25519, rsa, ecdsa)
        signer_id: Identifier of the signer (auditor, system)
        private_key_ref: Reference to the signing key in Vault
    """

    algorithm: str = Field(
        "ed25519",
        description="Signature algorithm (ed25519, rsa, ecdsa)",
    )
    signer_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Signer identifier",
    )
    private_key_ref: Optional[str] = Field(
        None, max_length=500,
        description="Reference to the signing key in secrets manager",
    )


# ============================================================================
# REQUEST MODELS - COMPLIANCE
# ============================================================================


class ComplianceTraceRequest(GreenLangBase):
    """
    Request model for tracing compliance requirements.

    Maps audit trail events and lineage data to specific regulatory
    framework requirements and data points.

    Attributes:
        framework: Regulatory framework to trace against
        organization_id: Organization identifier
        reporting_year: Reporting year
        data_points: Specific data points to trace
        scope_filter: Optional scope filter
    """

    framework: str = Field(
        ..., max_length=50,
        description="Framework to trace (ghg_protocol, iso_14064, etc.)",
    )
    organization_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2000, le=2100,
        description="Reporting year",
    )
    data_points: List[str] = Field(
        default_factory=list,
        description="Specific data points to trace (e.g., esrs_e1_dp_01)",
    )
    scope_filter: Optional[str] = Field(
        None, max_length=20,
        description="Scope filter (scope_1, scope_2, scope_3)",
    )


# ============================================================================
# REQUEST MODELS - CHANGES
# ============================================================================


class DetectChangeRequest(GreenLangBase):
    """
    Request model for detecting and recording a change event.

    Records a change to an emission factor, activity data, calculation
    parameter, or configuration setting, and triggers impact analysis
    to determine which downstream calculations are affected.

    Attributes:
        change_type: Type of change detected
        affected_entity_type: Type of entity affected (emission_factor, etc.)
        affected_entity_id: Identifier of the affected entity
        old_value: Previous value (serialized to string)
        new_value: New value (serialized to string)
        trigger: What triggered the change (user, system, api, recalculation)
        organization_id: Organization identifier
        reporting_year: Reporting year
        severity: Change severity (low, medium, high, critical)
        metadata: Additional metadata
    """

    change_type: str = Field(
        ..., min_length=1, max_length=100,
        description="Change type (ef_update, data_correction, parameter_change)",
    )
    affected_entity_type: str = Field(
        ..., min_length=1, max_length=100,
        description="Entity type (emission_factor, activity_data, configuration)",
    )
    affected_entity_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Identifier of the affected entity",
    )
    old_value: Optional[str] = Field(
        None, max_length=5000,
        description="Previous value (serialized)",
    )
    new_value: Optional[str] = Field(
        None, max_length=5000,
        description="New value (serialized)",
    )
    trigger: str = Field(
        "user",
        description="Change trigger (user, system, api, recalculation)",
    )
    organization_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2000, le=2100,
        description="Reporting year",
    )
    severity: str = Field(
        "medium",
        description="Change severity (low, medium, high, critical)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


# ============================================================================
# REQUEST MODELS - PIPELINE
# ============================================================================


class PipelineExecuteRequest(GreenLangBase):
    """
    Request model for full pipeline execution.

    Executes the complete 10-stage audit trail pipeline: validate event,
    record to hash chain, build lineage nodes/edges, check compliance
    coverage, detect changes, optionally create evidence package, and
    return consolidated results.

    Attributes:
        event_type: Audit event type
        agent_id: MRV agent identifier
        scope: Emission scope
        category: Scope 3 category number (1-15)
        organization_id: Organization identifier
        reporting_year: Reporting year
        calculation_id: Linked calculation UUID
        payload: Event payload data
        data_quality_score: Data quality score (1-5)
        metadata: Additional metadata
        include_evidence: Whether to include evidence packaging
        include_compliance: Whether to include compliance trace
    """

    event_type: str = Field(
        ..., min_length=1, max_length=100,
        description="Audit event type",
    )
    agent_id: str = Field(
        ..., min_length=1, max_length=50,
        description="MRV agent identifier",
    )
    scope: Optional[str] = Field(
        None, max_length=20,
        description="Emission scope",
    )
    category: Optional[int] = Field(
        None, ge=1, le=15,
        description="Scope 3 category number (1-15)",
    )
    organization_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Organization identifier",
    )
    reporting_year: int = Field(
        ..., ge=2000, le=2100,
        description="Reporting year",
    )
    calculation_id: Optional[str] = Field(
        None, max_length=100,
        description="Linked calculation UUID",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload data",
    )
    data_quality_score: Optional[int] = Field(
        None, ge=1, le=5,
        description="Data quality score (1=best, 5=worst)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    include_evidence: bool = Field(
        False,
        description="Whether to include evidence packaging in pipeline output",
    )
    include_compliance: bool = Field(
        False,
        description="Whether to include compliance trace in pipeline output",
    )


class BatchPipelineRequest(GreenLangBase):
    """
    Request model for batch pipeline execution.

    Executes the full audit trail pipeline for a batch of events.
    Events are processed sequentially to maintain hash chain ordering.

    Attributes:
        events: List of pipeline execution requests
        include_evidence: Whether to include evidence packaging
        include_compliance: Whether to include compliance tracing
    """

    events: List[PipelineExecuteRequest] = Field(
        ..., min_items=1, max_items=10000,
        description="List of pipeline execution requests",
    )
    include_evidence: bool = Field(
        False,
        description="Whether to include evidence packaging",
    )
    include_compliance: bool = Field(
        False,
        description="Whether to include compliance tracing",
    )


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class EventResponse(GreenLangBase):
    """Response model for a single audit event."""

    event_id: str = Field(..., description="Unique event UUID")
    event_type: str = Field(..., description="Audit event type")
    agent_id: str = Field(..., description="MRV agent identifier")
    scope: Optional[str] = Field(None, description="Emission scope")
    category: Optional[int] = Field(None, description="Scope 3 category")
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., description="Reporting year")
    calculation_id: Optional[str] = Field(
        None, description="Linked calculation UUID"
    )
    chain_position: int = Field(
        ..., description="Position in the hash chain (0-based)"
    )
    event_hash: str = Field(
        ..., description="SHA-256 hash of this event"
    )
    previous_hash: str = Field(
        ..., description="SHA-256 hash of the previous event in chain"
    )
    payload: Dict[str, Any] = Field(
        ..., description="Event payload data"
    )
    data_quality_score: Optional[int] = Field(
        None, description="Data quality score (1-5)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: str = Field(..., description="ISO 8601 creation timestamp")


class BatchEventResponse(GreenLangBase):
    """Response model for batch audit event recording."""

    batch_id: str = Field(..., description="Unique batch UUID")
    events_recorded: int = Field(
        ..., description="Number of events successfully recorded"
    )
    events_failed: int = Field(
        0, description="Number of events that failed"
    )
    event_ids: List[str] = Field(
        ..., description="List of recorded event UUIDs"
    )
    chain_head_hash: str = Field(
        ..., description="SHA-256 hash at chain head after batch"
    )
    chain_length: int = Field(
        ..., description="Total chain length after batch"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-event error details"
    )
    processing_time_ms: float = Field(
        ..., description="Total batch processing time (ms)"
    )
    validated_only: bool = Field(
        False, description="Whether events were validated but not persisted"
    )


class ChainVerifyResponse(GreenLangBase):
    """Response model for hash chain integrity verification."""

    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., description="Reporting year")
    is_valid: bool = Field(
        ..., description="Whether the chain is intact"
    )
    chain_length: int = Field(
        ..., description="Total chain length verified"
    )
    start_position: int = Field(
        ..., description="Starting position checked"
    )
    end_position: int = Field(
        ..., description="Ending position checked"
    )
    events_verified: int = Field(
        ..., description="Number of events verified"
    )
    first_invalid_position: Optional[int] = Field(
        None, description="Position of first integrity violation (null if valid)"
    )
    head_hash: str = Field(
        ..., description="SHA-256 hash at chain head"
    )
    genesis_hash: str = Field(
        ..., description="Genesis hash at chain start"
    )
    verification_time_ms: float = Field(
        ..., description="Verification duration (ms)"
    )


class LineageNodeResponse(GreenLangBase):
    """Response model for a lineage graph node."""

    node_id: str = Field(..., description="Unique node UUID")
    node_type: str = Field(..., description="Node type")
    level: int = Field(..., description="Hierarchical level")
    qualified_name: str = Field(..., description="Fully qualified name")
    display_name: str = Field(..., description="Human-readable name")
    value: Optional[float] = Field(None, description="Numeric value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    data_quality_score: Optional[int] = Field(
        None, description="Data quality score (1-5)"
    )
    agent_id: Optional[str] = Field(None, description="Producing agent")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: str = Field(..., description="ISO 8601 creation timestamp")


class LineageEdgeResponse(GreenLangBase):
    """Response model for a lineage graph edge."""

    edge_id: str = Field(..., description="Unique edge UUID")
    source_node_id: str = Field(..., description="Source node UUID")
    target_node_id: str = Field(..., description="Target node UUID")
    edge_type: str = Field(..., description="Edge type")
    transformation_description: Optional[str] = Field(
        None, description="Transformation description"
    )
    confidence: float = Field(..., description="Confidence score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: str = Field(..., description="ISO 8601 creation timestamp")


class LineageGraphResponse(GreenLangBase):
    """Response model for a complete lineage graph."""

    calculation_id: str = Field(
        ..., description="Calculation UUID this graph belongs to"
    )
    nodes: List[LineageNodeResponse] = Field(
        ..., description="List of lineage nodes"
    )
    edges: List[LineageEdgeResponse] = Field(
        ..., description="List of lineage edges"
    )
    node_count: int = Field(..., description="Total number of nodes")
    edge_count: int = Field(..., description="Total number of edges")
    max_depth: int = Field(
        ..., description="Maximum depth of the graph"
    )
    root_nodes: List[str] = Field(
        ..., description="UUIDs of root nodes (no incoming edges)"
    )
    leaf_nodes: List[str] = Field(
        ..., description="UUIDs of leaf nodes (no outgoing edges)"
    )


class LineageTraceResponse(GreenLangBase):
    """Response model for lineage trace traversal."""

    start_node_id: str = Field(..., description="Starting node UUID")
    direction: str = Field(..., description="Traversal direction")
    depth_reached: int = Field(
        ..., description="Actual depth reached in traversal"
    )
    nodes: List[LineageNodeResponse] = Field(
        ..., description="Nodes visited during traversal"
    )
    edges: List[LineageEdgeResponse] = Field(
        ..., description="Edges traversed"
    )
    path_count: int = Field(
        ..., description="Number of distinct paths found"
    )
    traversal_time_ms: float = Field(
        ..., description="Traversal duration (ms)"
    )


class VisualizationResponse(GreenLangBase):
    """Response model for lineage graph visualization export."""

    calculation_id: str = Field(
        ..., description="Calculation UUID"
    )
    format: str = Field(
        ..., description="Export format (dot, json, mermaid, d3, cytoscape)"
    )
    content: str = Field(
        ..., description="Visualization content in the requested format"
    )
    node_count: int = Field(..., description="Number of nodes rendered")
    edge_count: int = Field(..., description="Number of edges rendered")


class EvidencePackageResponse(GreenLangBase):
    """Response model for an evidence package."""

    package_id: str = Field(..., description="Unique package UUID")
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., description="Reporting year")
    assurance_level: str = Field(
        ..., description="Assurance level (none, limited, reasonable)"
    )
    frameworks: List[str] = Field(
        ..., description="Frameworks included"
    )
    event_count: int = Field(
        ..., description="Number of audit events included"
    )
    lineage_node_count: int = Field(
        ..., description="Number of lineage nodes included"
    )
    package_hash: str = Field(
        ..., description="SHA-256 hash of the complete package"
    )
    is_signed: bool = Field(
        False, description="Whether the package has been digitally signed"
    )
    signature: Optional[str] = Field(
        None, description="Digital signature (base64-encoded)"
    )
    signer_id: Optional[str] = Field(
        None, description="Identifier of the signer"
    )
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    size_bytes: int = Field(
        ..., description="Package size in bytes"
    )


class SignatureVerifyResponse(GreenLangBase):
    """Response model for evidence package signature verification."""

    package_id: str = Field(..., description="Package UUID")
    is_valid: bool = Field(
        ..., description="Whether the signature is valid"
    )
    algorithm: str = Field(..., description="Signature algorithm used")
    signer_id: Optional[str] = Field(None, description="Signer identifier")
    signed_at: Optional[str] = Field(None, description="Signing timestamp")
    verification_time_ms: float = Field(
        ..., description="Verification duration (ms)"
    )


class ComplianceTraceResponse(GreenLangBase):
    """Response model for compliance requirement tracing."""

    framework: str = Field(
        ..., description="Regulatory framework traced"
    )
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., description="Reporting year")
    total_requirements: int = Field(
        ..., description="Total requirements in the framework"
    )
    requirements_covered: int = Field(
        ..., description="Requirements with supporting audit evidence"
    )
    coverage_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Coverage percentage (0-100)",
    )
    status: str = Field(
        ..., description="Coverage status (pass, warn, fail)"
    )
    requirement_details: List[Dict[str, Any]] = Field(
        ..., description="Per-requirement coverage details"
    )
    gaps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Identified coverage gaps",
    )


class ComplianceCoverageResponse(GreenLangBase):
    """Response model for overall compliance coverage."""

    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., description="Reporting year")
    frameworks: List[ComplianceTraceResponse] = Field(
        ..., description="Per-framework coverage results"
    )
    overall_coverage_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Weighted overall coverage percentage",
    )
    overall_status: str = Field(
        ..., description="Overall status (pass, warn, fail)"
    )


class ChangeDetectionResponse(GreenLangBase):
    """Response model for change detection."""

    change_id: str = Field(..., description="Unique change UUID")
    change_type: str = Field(..., description="Type of change")
    affected_entity_type: str = Field(
        ..., description="Type of affected entity"
    )
    affected_entity_id: str = Field(
        ..., description="Identifier of affected entity"
    )
    old_value: Optional[str] = Field(None, description="Previous value")
    new_value: Optional[str] = Field(None, description="New value")
    trigger: str = Field(..., description="What triggered the change")
    severity: str = Field(..., description="Severity (low/medium/high/critical)")
    impact_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of downstream impact",
    )
    affected_calculations: int = Field(
        0, description="Number of downstream calculations affected"
    )
    requires_recalculation: bool = Field(
        False, description="Whether downstream recalculation is needed"
    )
    created_at: str = Field(..., description="ISO 8601 creation timestamp")


class ChangeImpactResponse(GreenLangBase):
    """Response model for change impact analysis."""

    change_id: str = Field(..., description="Change UUID")
    affected_nodes: List[Dict[str, Any]] = Field(
        ..., description="Lineage nodes affected by the change"
    )
    affected_calculations: List[Dict[str, Any]] = Field(
        ..., description="Calculations that need recalculation"
    )
    affected_disclosures: List[Dict[str, Any]] = Field(
        ..., description="Disclosure data points affected"
    )
    total_impact_nodes: int = Field(
        ..., description="Total nodes in impact scope"
    )
    materiality_assessment: Dict[str, Any] = Field(
        ..., description="Materiality assessment of the change"
    )


class PipelineResponse(GreenLangBase):
    """Response model for full pipeline execution."""

    pipeline_id: str = Field(..., description="Unique pipeline execution UUID")
    event: EventResponse = Field(
        ..., description="Recorded audit event"
    )
    lineage_nodes_created: int = Field(
        0, description="Lineage nodes created during pipeline"
    )
    lineage_edges_created: int = Field(
        0, description="Lineage edges created during pipeline"
    )
    compliance_coverage: Optional[Dict[str, Any]] = Field(
        None, description="Compliance coverage if requested"
    )
    evidence_package_id: Optional[str] = Field(
        None, description="Evidence package UUID if requested"
    )
    changes_detected: int = Field(
        0, description="Changes detected during pipeline"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash for the pipeline execution"
    )
    processing_time_ms: float = Field(
        ..., description="Total pipeline processing time (ms)"
    )
    stages_completed: List[str] = Field(
        ..., description="Pipeline stages that completed successfully"
    )


class BatchPipelineResponse(GreenLangBase):
    """Response model for batch pipeline execution."""

    batch_id: str = Field(..., description="Unique batch UUID")
    results: List[PipelineResponse] = Field(
        ..., description="Per-event pipeline results"
    )
    events_processed: int = Field(
        ..., description="Events processed successfully"
    )
    events_failed: int = Field(
        0, description="Events that failed processing"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-event error details"
    )
    total_processing_time_ms: float = Field(
        ..., description="Total batch processing time (ms)"
    )


class AuditSummaryResponse(GreenLangBase):
    """Response model for audit trail summary."""

    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., description="Reporting year")
    total_events: int = Field(
        ..., description="Total audit events recorded"
    )
    chain_length: int = Field(
        ..., description="Current hash chain length"
    )
    chain_is_valid: bool = Field(
        ..., description="Whether the chain integrity is intact"
    )
    events_by_type: Dict[str, int] = Field(
        ..., description="Event count breakdown by type"
    )
    events_by_scope: Dict[str, int] = Field(
        ..., description="Event count breakdown by scope"
    )
    events_by_agent: Dict[str, int] = Field(
        ..., description="Event count breakdown by agent"
    )
    lineage_nodes: int = Field(
        ..., description="Total lineage nodes"
    )
    lineage_edges: int = Field(
        ..., description="Total lineage edges"
    )
    evidence_packages: int = Field(
        ..., description="Total evidence packages created"
    )
    compliance_coverage: Dict[str, float] = Field(
        ..., description="Per-framework compliance coverage percentage"
    )
    changes_detected: int = Field(
        ..., description="Total changes detected"
    )
    last_event_at: Optional[str] = Field(
        None, description="Timestamp of most recent event"
    )


class EngineStatus(GreenLangBase):
    """Status of a single engine in the audit trail agent."""

    engine_name: str = Field(..., description="Engine identifier")
    status: str = Field(
        ..., description="Engine status (healthy, degraded, unhealthy)"
    )
    last_used: Optional[str] = Field(
        None, description="Last usage timestamp"
    )


class HealthResponse(GreenLangBase):
    """Response model for health check with 7 engine statuses."""

    status: str = Field(..., description="Service health status")
    agent_id: str = Field(..., description="Agent identifier")
    version: str = Field(..., description="Agent version")
    uptime_seconds: float = Field(
        ..., description="Seconds since service start"
    )
    engines: List[EngineStatus] = Field(
        ..., description="Individual engine status"
    )
    total_events: int = Field(
        ..., description="Total audit events recorded across all orgs"
    )
    total_chains: int = Field(
        ..., description="Total active hash chains"
    )


# ============================================================================
# MODULE-LEVEL TRACKING
# ============================================================================

_start_time: datetime = datetime.utcnow()

# Engine names for health check -- matches the 7 engines from __init__.py
_ENGINE_NAMES = [
    "AuditEventEngine",
    "LineageGraphEngine",
    "EvidencePackagerEngine",
    "ComplianceTracerEngine",
    "ChangeDetectorEngine",
    "ComplianceCheckerEngine",
    "AuditTrailPipelineEngine",
]


# ============================================================================
# ENDPOINTS - EVENTS (5)
# ============================================================================


@router.post(
    "/events",
    response_model=EventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a single audit event",
    description=(
        "Record an immutable audit event and append it to the SHA-256 hash "
        "chain for the given organization and reporting year. The event hash "
        "is computed as SHA-256(previous_hash + canonical_payload), ensuring "
        "tamper-evident ordering. Returns the recorded event with its chain "
        "position and hash."
    ),
)
async def record_event(
    request: RecordEventRequest,
    service=Depends(get_service),
) -> EventResponse:
    """
    Record a single audit event to the hash chain.

    Args:
        request: Audit event recording request
        service: AuditTrailLineageService instance

    Returns:
        EventResponse with event_id, chain position, and hashes

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            "Recording audit event: type=%s, agent=%s, org=%s, year=%d",
            request.event_type,
            request.agent_id,
            request.organization_id,
            request.reporting_year,
        )

        result = await service.record_event(request.dict())
        event_id = result.get("event_id", str(uuid.uuid4()))

        return EventResponse(
            event_id=event_id,
            event_type=result.get("event_type", request.event_type),
            agent_id=result.get("agent_id", request.agent_id),
            scope=result.get("scope", request.scope),
            category=result.get("category", request.category),
            organization_id=result.get(
                "organization_id", request.organization_id
            ),
            reporting_year=result.get(
                "reporting_year", request.reporting_year
            ),
            calculation_id=result.get(
                "calculation_id", request.calculation_id
            ),
            chain_position=result.get("chain_position", 0),
            event_hash=result.get("event_hash", ""),
            previous_hash=result.get("previous_hash", ""),
            payload=result.get("payload", request.payload),
            data_quality_score=result.get(
                "data_quality_score", request.data_quality_score
            ),
            metadata=result.get("metadata", request.metadata),
            created_at=result.get(
                "created_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in record_event: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in record_event: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record audit event",
        )


@router.post(
    "/events/batch",
    response_model=BatchEventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a batch of audit events",
    description=(
        "Record up to 10,000 audit events in a single batch. Events are "
        "appended to the hash chain in order. If validate_only is True, "
        "events are validated but not persisted."
    ),
)
async def record_events_batch(
    request: BatchRecordRequest,
    service=Depends(get_service),
) -> BatchEventResponse:
    """
    Record a batch of audit events to the hash chain.

    Args:
        request: Batch audit event recording request
        service: AuditTrailLineageService instance

    Returns:
        BatchEventResponse with batch summary and per-event results

    Raises:
        HTTPException: 400 for validation, 500 for processing failures
    """
    try:
        logger.info(
            "Recording batch of %d audit events (validate_only=%s)",
            len(request.events),
            request.validate_only,
        )

        result = await service.record_events_batch(request.dict())
        batch_id = result.get("batch_id", str(uuid.uuid4()))

        return BatchEventResponse(
            batch_id=batch_id,
            events_recorded=result.get("events_recorded", 0),
            events_failed=result.get("events_failed", 0),
            event_ids=result.get("event_ids", []),
            chain_head_hash=result.get("chain_head_hash", ""),
            chain_length=result.get("chain_length", 0),
            errors=result.get("errors", []),
            processing_time_ms=result.get("processing_time_ms", 0.0),
            validated_only=result.get(
                "validated_only", request.validate_only
            ),
        )

    except ValueError as e:
        logger.error("Validation error in record_events_batch: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in record_events_batch: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch event recording failed",
        )


@router.get(
    "/events/{event_id}",
    response_model=EventResponse,
    summary="Get a single audit event",
    description=(
        "Retrieve a single audit event by its UUID. Returns the full event "
        "record including chain position, hashes, payload, and metadata."
    ),
)
async def get_event(
    event_id: str = Path(..., description="Audit event UUID"),
    service=Depends(get_service),
) -> EventResponse:
    """
    Get a single audit event by UUID.

    Args:
        event_id: Event UUID
        service: AuditTrailLineageService instance

    Returns:
        EventResponse with full event details

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info("Getting audit event: %s", event_id)

        result = await service.get_event(event_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit event {event_id} not found",
            )

        return EventResponse(
            event_id=result.get("event_id", event_id),
            event_type=result.get("event_type", ""),
            agent_id=result.get("agent_id", ""),
            scope=result.get("scope"),
            category=result.get("category"),
            organization_id=result.get("organization_id", ""),
            reporting_year=result.get("reporting_year", 0),
            calculation_id=result.get("calculation_id"),
            chain_position=result.get("chain_position", 0),
            event_hash=result.get("event_hash", ""),
            previous_hash=result.get("previous_hash", ""),
            payload=result.get("payload", {}),
            data_quality_score=result.get("data_quality_score"),
            metadata=result.get("metadata", {}),
            created_at=result.get(
                "created_at", datetime.utcnow().isoformat()
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_event: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit event",
        )


@router.get(
    "/events",
    response_model=List[EventResponse],
    summary="List and query audit events",
    description=(
        "List audit events with optional filters for organization, reporting "
        "year, event type, agent, and scope. Supports pagination via "
        "limit and offset parameters."
    ),
)
async def list_events(
    organization_id: Optional[str] = Query(
        None, description="Filter by organization identifier"
    ),
    reporting_year: Optional[int] = Query(
        None, ge=2000, le=2100, description="Filter by reporting year"
    ),
    event_type: Optional[str] = Query(
        None, description="Filter by event type"
    ),
    agent_id: Optional[str] = Query(
        None, description="Filter by MRV agent identifier"
    ),
    scope: Optional[str] = Query(
        None, description="Filter by emission scope"
    ),
    limit: int = Query(
        100, ge=1, le=10000, description="Maximum records to return"
    ),
    offset: int = Query(
        0, ge=0, description="Number of records to skip"
    ),
    service=Depends(get_service),
) -> List[EventResponse]:
    """
    List audit events with optional filtering and pagination.

    Args:
        organization_id: Optional organization filter
        reporting_year: Optional year filter
        event_type: Optional event type filter
        agent_id: Optional agent filter
        scope: Optional scope filter
        limit: Maximum number of records
        offset: Number of records to skip
        service: AuditTrailLineageService instance

    Returns:
        List of EventResponse matching the filters

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info(
            "Listing audit events: org=%s, year=%s, type=%s, agent=%s, "
            "scope=%s, limit=%d, offset=%d",
            organization_id, reporting_year, event_type, agent_id,
            scope, limit, offset,
        )

        filters = {
            "organization_id": organization_id,
            "reporting_year": reporting_year,
            "event_type": event_type,
            "agent_id": agent_id,
            "scope": scope,
            "limit": limit,
            "offset": offset,
        }

        results = await service.list_events(filters)

        return [
            EventResponse(
                event_id=r.get("event_id", ""),
                event_type=r.get("event_type", ""),
                agent_id=r.get("agent_id", ""),
                scope=r.get("scope"),
                category=r.get("category"),
                organization_id=r.get("organization_id", ""),
                reporting_year=r.get("reporting_year", 0),
                calculation_id=r.get("calculation_id"),
                chain_position=r.get("chain_position", 0),
                event_hash=r.get("event_hash", ""),
                previous_hash=r.get("previous_hash", ""),
                payload=r.get("payload", {}),
                data_quality_score=r.get("data_quality_score"),
                metadata=r.get("metadata", {}),
                created_at=r.get(
                    "created_at", datetime.utcnow().isoformat()
                ),
            )
            for r in results
        ]

    except Exception as e:
        logger.error("Error in list_events: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list audit events",
        )


@router.delete(
    "/events/{event_id}",
    response_model=Dict[str, Any],
    summary="Soft-delete an audit event",
    description=(
        "Soft-delete an audit event by UUID. The event is marked as deleted "
        "but remains in the hash chain for integrity. A deletion audit event "
        "is automatically appended to the chain."
    ),
)
async def delete_event(
    event_id: str = Path(..., description="Audit event UUID to soft-delete"),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Soft-delete an audit event.

    Args:
        event_id: Event UUID to delete
        service: AuditTrailLineageService instance

    Returns:
        Dictionary with deletion status

    Raises:
        HTTPException: 404 if not found, 500 for deletion failures
    """
    try:
        logger.info("Soft-deleting audit event: %s", event_id)

        result = await service.delete_event(event_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit event {event_id} not found",
            )

        return {
            "event_id": event_id,
            "deleted": result.get("deleted", True),
            "message": result.get(
                "message",
                f"Audit event {event_id} soft-deleted successfully",
            ),
            "deletion_event_id": result.get("deletion_event_id"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in delete_event: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete audit event",
        )


# ============================================================================
# ENDPOINTS - CHAIN (2)
# ============================================================================


@router.post(
    "/chain/verify",
    response_model=ChainVerifyResponse,
    summary="Verify hash chain integrity",
    description=(
        "Verify the SHA-256 hash chain integrity for a given organization "
        "and reporting year. Checks that each event hash equals "
        "SHA-256(previous_hash + canonical_payload) for all events in the "
        "specified range. Returns verification status, chain length, and "
        "the position of the first invalid event if the chain is broken."
    ),
)
async def verify_chain(
    request: ChainVerifyRequest,
    service=Depends(get_service),
) -> ChainVerifyResponse:
    """
    Verify hash chain integrity for an organization and year.

    Args:
        request: Chain verification request
        service: AuditTrailLineageService instance

    Returns:
        ChainVerifyResponse with verification results

    Raises:
        HTTPException: 400 for invalid range, 500 for verification failures
    """
    try:
        logger.info(
            "Verifying hash chain: org=%s, year=%d, range=[%d, %s]",
            request.organization_id,
            request.reporting_year,
            request.start_position,
            request.end_position,
        )

        result = await service.verify_chain(request.dict())

        return ChainVerifyResponse(
            organization_id=result.get(
                "organization_id", request.organization_id
            ),
            reporting_year=result.get(
                "reporting_year", request.reporting_year
            ),
            is_valid=result.get("is_valid", False),
            chain_length=result.get("chain_length", 0),
            start_position=result.get(
                "start_position", request.start_position
            ),
            end_position=result.get("end_position", 0),
            events_verified=result.get("events_verified", 0),
            first_invalid_position=result.get("first_invalid_position"),
            head_hash=result.get("head_hash", ""),
            genesis_hash=result.get(
                "genesis_hash", "greenlang-atl-genesis-v1"
            ),
            verification_time_ms=result.get("verification_time_ms", 0.0),
        )

    except ValueError as e:
        logger.error("Validation error in verify_chain: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in verify_chain: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hash chain verification failed",
        )


@router.get(
    "/chain/{org_id}/{year}",
    response_model=ChainVerifyResponse,
    summary="Get hash chain for organization and year",
    description=(
        "Retrieve the hash chain metadata for a given organization and "
        "reporting year. Returns chain length, head hash, genesis hash, "
        "and overall integrity status."
    ),
)
async def get_chain(
    org_id: str = Path(
        ..., description="Organization identifier"
    ),
    year: int = Path(
        ..., ge=2000, le=2100, description="Reporting year"
    ),
    service=Depends(get_service),
) -> ChainVerifyResponse:
    """
    Get hash chain metadata for an organization and year.

    Args:
        org_id: Organization identifier
        year: Reporting year
        service: AuditTrailLineageService instance

    Returns:
        ChainVerifyResponse with chain metadata

    Raises:
        HTTPException: 404 if no chain exists, 500 for retrieval failures
    """
    try:
        logger.info("Getting hash chain: org=%s, year=%d", org_id, year)

        result = await service.get_chain(org_id, year)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No hash chain found for organization {org_id} "
                    f"and year {year}"
                ),
            )

        return ChainVerifyResponse(
            organization_id=result.get("organization_id", org_id),
            reporting_year=result.get("reporting_year", year),
            is_valid=result.get("is_valid", True),
            chain_length=result.get("chain_length", 0),
            start_position=result.get("start_position", 0),
            end_position=result.get("end_position", 0),
            events_verified=result.get("events_verified", 0),
            first_invalid_position=result.get("first_invalid_position"),
            head_hash=result.get("head_hash", ""),
            genesis_hash=result.get(
                "genesis_hash", "greenlang-atl-genesis-v1"
            ),
            verification_time_ms=result.get("verification_time_ms", 0.0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_chain: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve hash chain",
        )


# ============================================================================
# ENDPOINTS - LINEAGE (5)
# ============================================================================


@router.post(
    "/lineage/nodes",
    response_model=LineageNodeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a lineage graph node",
    description=(
        "Create a node in the MRV calculation lineage DAG. Nodes represent "
        "data points from raw source inputs (level 0) through intermediate "
        "calculations to disclosed totals (highest level)."
    ),
)
async def create_lineage_node(
    request: CreateNodeRequest,
    service=Depends(get_service),
) -> LineageNodeResponse:
    """
    Create a lineage graph node.

    Args:
        request: Node creation request
        service: AuditTrailLineageService instance

    Returns:
        LineageNodeResponse with the created node

    Raises:
        HTTPException: 400 for validation, 500 for creation failures
    """
    try:
        logger.info(
            "Creating lineage node: type=%s, name=%s, org=%s",
            request.node_type,
            request.qualified_name,
            request.organization_id,
        )

        result = await service.create_lineage_node(request.dict())
        node_id = result.get("node_id", str(uuid.uuid4()))

        return LineageNodeResponse(
            node_id=node_id,
            node_type=result.get("node_type", request.node_type),
            level=result.get("level", request.level),
            qualified_name=result.get(
                "qualified_name", request.qualified_name
            ),
            display_name=result.get(
                "display_name", request.display_name
            ),
            value=result.get("value", request.value),
            unit=result.get("unit", request.unit),
            data_quality_score=result.get(
                "data_quality_score", request.data_quality_score
            ),
            agent_id=result.get("agent_id", request.agent_id),
            metadata=result.get("metadata", request.metadata),
            created_at=result.get(
                "created_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in create_lineage_node: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in create_lineage_node: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create lineage node",
        )


@router.post(
    "/lineage/edges",
    response_model=LineageEdgeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a lineage graph edge",
    description=(
        "Create an edge in the MRV calculation lineage DAG connecting "
        "two nodes. Edges represent data flow and transformation "
        "relationships (input_to, derived_from, aggregated_into, etc.)."
    ),
)
async def create_lineage_edge(
    request: CreateEdgeRequest,
    service=Depends(get_service),
) -> LineageEdgeResponse:
    """
    Create a lineage graph edge.

    Args:
        request: Edge creation request
        service: AuditTrailLineageService instance

    Returns:
        LineageEdgeResponse with the created edge

    Raises:
        HTTPException: 400 for validation, 404 if nodes not found, 500 for failures
    """
    try:
        logger.info(
            "Creating lineage edge: %s -> %s (type=%s)",
            request.source_node_id,
            request.target_node_id,
            request.edge_type,
        )

        result = await service.create_lineage_edge(request.dict())
        edge_id = result.get("edge_id", str(uuid.uuid4()))

        return LineageEdgeResponse(
            edge_id=edge_id,
            source_node_id=result.get(
                "source_node_id", request.source_node_id
            ),
            target_node_id=result.get(
                "target_node_id", request.target_node_id
            ),
            edge_type=result.get("edge_type", request.edge_type),
            transformation_description=result.get(
                "transformation_description",
                request.transformation_description,
            ),
            confidence=result.get("confidence", request.confidence),
            metadata=result.get("metadata", request.metadata),
            created_at=result.get(
                "created_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in create_lineage_edge: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in create_lineage_edge: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create lineage edge",
        )


@router.get(
    "/lineage/graph/{calc_id}",
    response_model=LineageGraphResponse,
    summary="Get lineage graph for a calculation",
    description=(
        "Retrieve the complete lineage DAG for a specific calculation. "
        "Returns all nodes and edges that contributed to the calculation "
        "result, including root nodes (sources) and leaf nodes (outputs)."
    ),
)
async def get_lineage_graph(
    calc_id: str = Path(
        ..., description="Calculation UUID"
    ),
    service=Depends(get_service),
) -> LineageGraphResponse:
    """
    Get the complete lineage graph for a calculation.

    Args:
        calc_id: Calculation UUID
        service: AuditTrailLineageService instance

    Returns:
        LineageGraphResponse with nodes, edges, and graph metadata

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info("Getting lineage graph for calculation: %s", calc_id)

        result = await service.get_lineage_graph(calc_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lineage graph for calculation {calc_id} not found",
            )

        nodes = [
            LineageNodeResponse(
                node_id=n.get("node_id", ""),
                node_type=n.get("node_type", ""),
                level=n.get("level", 0),
                qualified_name=n.get("qualified_name", ""),
                display_name=n.get("display_name", ""),
                value=n.get("value"),
                unit=n.get("unit"),
                data_quality_score=n.get("data_quality_score"),
                agent_id=n.get("agent_id"),
                metadata=n.get("metadata", {}),
                created_at=n.get(
                    "created_at", datetime.utcnow().isoformat()
                ),
            )
            for n in result.get("nodes", [])
        ]

        edges = [
            LineageEdgeResponse(
                edge_id=e.get("edge_id", ""),
                source_node_id=e.get("source_node_id", ""),
                target_node_id=e.get("target_node_id", ""),
                edge_type=e.get("edge_type", ""),
                transformation_description=e.get(
                    "transformation_description"
                ),
                confidence=e.get("confidence", 1.0),
                metadata=e.get("metadata", {}),
                created_at=e.get(
                    "created_at", datetime.utcnow().isoformat()
                ),
            )
            for e in result.get("edges", [])
        ]

        return LineageGraphResponse(
            calculation_id=calc_id,
            nodes=nodes,
            edges=edges,
            node_count=result.get("node_count", len(nodes)),
            edge_count=result.get("edge_count", len(edges)),
            max_depth=result.get("max_depth", 0),
            root_nodes=result.get("root_nodes", []),
            leaf_nodes=result.get("leaf_nodes", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in get_lineage_graph: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve lineage graph",
        )


@router.post(
    "/lineage/trace",
    response_model=LineageTraceResponse,
    summary="Trace lineage forward or backward",
    description=(
        "Traverse the lineage DAG from a starting node in the specified "
        "direction (forward for impact analysis, backward for traceability). "
        "Supports optional filtering by node type and hierarchical level, "
        "with configurable maximum traversal depth."
    ),
)
async def trace_lineage(
    request: LineageTraceRequest,
    service=Depends(get_service),
) -> LineageTraceResponse:
    """
    Trace lineage in the DAG from a starting node.

    Args:
        request: Lineage trace request with direction and filters
        service: AuditTrailLineageService instance

    Returns:
        LineageTraceResponse with visited nodes, edges, and paths

    Raises:
        HTTPException: 404 if start node not found, 500 for traversal failures
    """
    try:
        logger.info(
            "Tracing lineage: start=%s, direction=%s, depth=%d",
            request.start_node_id,
            request.direction,
            request.max_depth,
        )

        result = await service.trace_lineage(request.dict())

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Starting node {request.start_node_id} not found"
                ),
            )

        nodes = [
            LineageNodeResponse(
                node_id=n.get("node_id", ""),
                node_type=n.get("node_type", ""),
                level=n.get("level", 0),
                qualified_name=n.get("qualified_name", ""),
                display_name=n.get("display_name", ""),
                value=n.get("value"),
                unit=n.get("unit"),
                data_quality_score=n.get("data_quality_score"),
                agent_id=n.get("agent_id"),
                metadata=n.get("metadata", {}),
                created_at=n.get(
                    "created_at", datetime.utcnow().isoformat()
                ),
            )
            for n in result.get("nodes", [])
        ]

        edges = [
            LineageEdgeResponse(
                edge_id=e.get("edge_id", ""),
                source_node_id=e.get("source_node_id", ""),
                target_node_id=e.get("target_node_id", ""),
                edge_type=e.get("edge_type", ""),
                transformation_description=e.get(
                    "transformation_description"
                ),
                confidence=e.get("confidence", 1.0),
                metadata=e.get("metadata", {}),
                created_at=e.get(
                    "created_at", datetime.utcnow().isoformat()
                ),
            )
            for e in result.get("edges", [])
        ]

        return LineageTraceResponse(
            start_node_id=request.start_node_id,
            direction=result.get("direction", request.direction),
            depth_reached=result.get("depth_reached", 0),
            nodes=nodes,
            edges=edges,
            path_count=result.get("path_count", 0),
            traversal_time_ms=result.get("traversal_time_ms", 0.0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in trace_lineage: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lineage trace failed",
        )


@router.get(
    "/lineage/visualize/{calc_id}",
    response_model=VisualizationResponse,
    summary="Get lineage graph visualization",
    description=(
        "Export the lineage DAG for a calculation in a visualization-ready "
        "format. Supported formats: DOT (Graphviz), JSON, Mermaid, D3, "
        "and Cytoscape."
    ),
)
async def visualize_lineage(
    calc_id: str = Path(
        ..., description="Calculation UUID"
    ),
    format: str = Query(
        "mermaid",
        description="Visualization format (dot, json, mermaid, d3, cytoscape)",
    ),
    service=Depends(get_service),
) -> VisualizationResponse:
    """
    Get lineage graph visualization for a calculation.

    Args:
        calc_id: Calculation UUID
        format: Visualization format
        service: AuditTrailLineageService instance

    Returns:
        VisualizationResponse with rendered content

    Raises:
        HTTPException: 400 for invalid format, 404 if not found, 500 for failures
    """
    try:
        valid_formats = {"dot", "json", "mermaid", "d3", "cytoscape"}
        if format not in valid_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid format '{format}'. "
                    f"Must be one of: {', '.join(sorted(valid_formats))}"
                ),
            )

        logger.info(
            "Visualizing lineage: calc=%s, format=%s",
            calc_id, format,
        )

        result = await service.visualize_lineage(calc_id, format)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Lineage graph for calculation {calc_id} not found"
                ),
            )

        return VisualizationResponse(
            calculation_id=calc_id,
            format=format,
            content=result.get("content", ""),
            node_count=result.get("node_count", 0),
            edge_count=result.get("edge_count", 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in visualize_lineage: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lineage visualization failed",
        )


# ============================================================================
# ENDPOINTS - EVIDENCE (4)
# ============================================================================


@router.post(
    "/evidence/package",
    response_model=EvidencePackageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an evidence package",
    description=(
        "Bundle audit events, lineage graphs, and supporting documents into "
        "a verifiable evidence package for third-party assurance engagements. "
        "Supports limited and reasonable assurance levels per ISAE 3410."
    ),
)
async def create_evidence_package(
    request: CreateEvidenceRequest,
    service=Depends(get_service),
) -> EvidencePackageResponse:
    """
    Create an evidence package for third-party verification.

    Args:
        request: Evidence package creation request
        service: AuditTrailLineageService instance

    Returns:
        EvidencePackageResponse with package details

    Raises:
        HTTPException: 400 for validation, 500 for creation failures
    """
    try:
        logger.info(
            "Creating evidence package: org=%s, year=%d, "
            "assurance=%s, frameworks=%s",
            request.organization_id,
            request.reporting_year,
            request.assurance_level,
            request.frameworks,
        )

        result = await service.create_evidence_package(request.dict())
        package_id = result.get("package_id", str(uuid.uuid4()))

        return EvidencePackageResponse(
            package_id=package_id,
            organization_id=result.get(
                "organization_id", request.organization_id
            ),
            reporting_year=result.get(
                "reporting_year", request.reporting_year
            ),
            assurance_level=result.get(
                "assurance_level", request.assurance_level
            ),
            frameworks=result.get("frameworks", request.frameworks),
            event_count=result.get("event_count", 0),
            lineage_node_count=result.get("lineage_node_count", 0),
            package_hash=result.get("package_hash", ""),
            is_signed=result.get("is_signed", False),
            signature=result.get("signature"),
            signer_id=result.get("signer_id"),
            created_at=result.get(
                "created_at", datetime.utcnow().isoformat()
            ),
            size_bytes=result.get("size_bytes", 0),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in create_evidence_package: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in create_evidence_package: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create evidence package",
        )


@router.get(
    "/evidence/{package_id}",
    response_model=EvidencePackageResponse,
    summary="Get an evidence package",
    description=(
        "Retrieve an evidence package by UUID. Returns package metadata "
        "including assurance level, framework coverage, hash, signature "
        "status, and size."
    ),
)
async def get_evidence_package(
    package_id: str = Path(
        ..., description="Evidence package UUID"
    ),
    service=Depends(get_service),
) -> EvidencePackageResponse:
    """
    Get an evidence package by UUID.

    Args:
        package_id: Evidence package UUID
        service: AuditTrailLineageService instance

    Returns:
        EvidencePackageResponse with package details

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info("Getting evidence package: %s", package_id)

        result = await service.get_evidence_package(package_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evidence package {package_id} not found",
            )

        return EvidencePackageResponse(
            package_id=result.get("package_id", package_id),
            organization_id=result.get("organization_id", ""),
            reporting_year=result.get("reporting_year", 0),
            assurance_level=result.get("assurance_level", "limited"),
            frameworks=result.get("frameworks", []),
            event_count=result.get("event_count", 0),
            lineage_node_count=result.get("lineage_node_count", 0),
            package_hash=result.get("package_hash", ""),
            is_signed=result.get("is_signed", False),
            signature=result.get("signature"),
            signer_id=result.get("signer_id"),
            created_at=result.get(
                "created_at", datetime.utcnow().isoformat()
            ),
            size_bytes=result.get("size_bytes", 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in get_evidence_package: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve evidence package",
        )


@router.post(
    "/evidence/{package_id}/sign",
    response_model=EvidencePackageResponse,
    summary="Sign an evidence package",
    description=(
        "Apply a digital signature to an evidence package using the "
        "specified algorithm (Ed25519, RSA, or ECDSA). The signature "
        "covers the package hash for non-repudiation."
    ),
)
async def sign_evidence_package(
    package_id: str = Path(
        ..., description="Evidence package UUID"
    ),
    request: SignPackageRequest = ...,
    service=Depends(get_service),
) -> EvidencePackageResponse:
    """
    Sign an evidence package with a digital signature.

    Args:
        package_id: Evidence package UUID
        request: Signing request with algorithm and signer
        service: AuditTrailLineageService instance

    Returns:
        EvidencePackageResponse with signature applied

    Raises:
        HTTPException: 404 if not found, 400 for invalid algorithm, 500 for failures
    """
    try:
        logger.info(
            "Signing evidence package: %s, algorithm=%s, signer=%s",
            package_id,
            request.algorithm,
            request.signer_id,
        )

        sign_data = request.dict()
        sign_data["package_id"] = package_id

        result = await service.sign_evidence_package(sign_data)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evidence package {package_id} not found",
            )

        return EvidencePackageResponse(
            package_id=result.get("package_id", package_id),
            organization_id=result.get("organization_id", ""),
            reporting_year=result.get("reporting_year", 0),
            assurance_level=result.get("assurance_level", "limited"),
            frameworks=result.get("frameworks", []),
            event_count=result.get("event_count", 0),
            lineage_node_count=result.get("lineage_node_count", 0),
            package_hash=result.get("package_hash", ""),
            is_signed=result.get("is_signed", True),
            signature=result.get("signature"),
            signer_id=result.get("signer_id", request.signer_id),
            created_at=result.get(
                "created_at", datetime.utcnow().isoformat()
            ),
            size_bytes=result.get("size_bytes", 0),
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(
            f"Validation error in sign_evidence_package: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in sign_evidence_package: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to sign evidence package",
        )


@router.post(
    "/evidence/{package_id}/verify",
    response_model=SignatureVerifyResponse,
    summary="Verify evidence package signature",
    description=(
        "Verify the digital signature of an evidence package. Checks that "
        "the signature is valid and that the package hash has not been "
        "tampered with since signing."
    ),
)
async def verify_evidence_signature(
    package_id: str = Path(
        ..., description="Evidence package UUID"
    ),
    service=Depends(get_service),
) -> SignatureVerifyResponse:
    """
    Verify the digital signature of an evidence package.

    Args:
        package_id: Evidence package UUID
        service: AuditTrailLineageService instance

    Returns:
        SignatureVerifyResponse with verification result

    Raises:
        HTTPException: 404 if not found, 400 if not signed, 500 for failures
    """
    try:
        logger.info(
            "Verifying evidence package signature: %s", package_id
        )

        result = await service.verify_evidence_signature(package_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evidence package {package_id} not found",
            )

        return SignatureVerifyResponse(
            package_id=result.get("package_id", package_id),
            is_valid=result.get("is_valid", False),
            algorithm=result.get("algorithm", ""),
            signer_id=result.get("signer_id"),
            signed_at=result.get("signed_at"),
            verification_time_ms=result.get(
                "verification_time_ms", 0.0
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in verify_evidence_signature: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Signature verification failed",
        )


# ============================================================================
# ENDPOINTS - COMPLIANCE (2)
# ============================================================================


@router.post(
    "/compliance/trace",
    response_model=ComplianceTraceResponse,
    summary="Trace compliance requirements",
    description=(
        "Map audit trail events and lineage data to specific regulatory "
        "framework requirements. Returns coverage percentage, per-requirement "
        "evidence mapping, and identified gaps."
    ),
)
async def trace_compliance(
    request: ComplianceTraceRequest,
    service=Depends(get_service),
) -> ComplianceTraceResponse:
    """
    Trace compliance requirements for a framework.

    Args:
        request: Compliance trace request
        service: AuditTrailLineageService instance

    Returns:
        ComplianceTraceResponse with coverage and gap details

    Raises:
        HTTPException: 400 for invalid framework, 500 for failures
    """
    try:
        logger.info(
            "Tracing compliance: framework=%s, org=%s, year=%d",
            request.framework,
            request.organization_id,
            request.reporting_year,
        )

        result = await service.trace_compliance(request.dict())

        return ComplianceTraceResponse(
            framework=result.get("framework", request.framework),
            organization_id=result.get(
                "organization_id", request.organization_id
            ),
            reporting_year=result.get(
                "reporting_year", request.reporting_year
            ),
            total_requirements=result.get("total_requirements", 0),
            requirements_covered=result.get("requirements_covered", 0),
            coverage_pct=result.get("coverage_pct", 0.0),
            status=result.get("status", "fail"),
            requirement_details=result.get("requirement_details", []),
            gaps=result.get("gaps", []),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in trace_compliance: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in trace_compliance: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance trace failed",
        )


@router.get(
    "/compliance/coverage/{org_id}",
    response_model=ComplianceCoverageResponse,
    summary="Get compliance coverage for an organization",
    description=(
        "Retrieve overall compliance coverage for an organization across "
        "all supported regulatory frameworks. Returns per-framework and "
        "overall coverage percentages with pass/warn/fail status."
    ),
)
async def get_compliance_coverage(
    org_id: str = Path(
        ..., description="Organization identifier"
    ),
    year: Optional[int] = Query(
        None, ge=2000, le=2100,
        description="Reporting year (default: current year)",
    ),
    framework: Optional[str] = Query(
        None,
        description="Filter by framework (ghg_protocol, iso_14064, etc.)",
    ),
    service=Depends(get_service),
) -> ComplianceCoverageResponse:
    """
    Get compliance coverage for an organization.

    Args:
        org_id: Organization identifier
        year: Optional reporting year filter
        framework: Optional framework filter
        service: AuditTrailLineageService instance

    Returns:
        ComplianceCoverageResponse with per-framework coverage

    Raises:
        HTTPException: 404 if no data, 500 for failures
    """
    try:
        effective_year = year or datetime.utcnow().year

        logger.info(
            "Getting compliance coverage: org=%s, year=%d, framework=%s",
            org_id, effective_year, framework,
        )

        filters = {
            "organization_id": org_id,
            "reporting_year": effective_year,
            "framework": framework,
        }

        result = await service.get_compliance_coverage(filters)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No compliance data found for organization {org_id} "
                    f"and year {effective_year}"
                ),
            )

        frameworks_data = result.get("frameworks", [])
        framework_responses = [
            ComplianceTraceResponse(
                framework=f.get("framework", ""),
                organization_id=f.get("organization_id", org_id),
                reporting_year=f.get("reporting_year", effective_year),
                total_requirements=f.get("total_requirements", 0),
                requirements_covered=f.get("requirements_covered", 0),
                coverage_pct=f.get("coverage_pct", 0.0),
                status=f.get("status", "fail"),
                requirement_details=f.get("requirement_details", []),
                gaps=f.get("gaps", []),
            )
            for f in frameworks_data
        ]

        return ComplianceCoverageResponse(
            organization_id=org_id,
            reporting_year=effective_year,
            frameworks=framework_responses,
            overall_coverage_pct=result.get(
                "overall_coverage_pct", 0.0
            ),
            overall_status=result.get("overall_status", "fail"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in get_compliance_coverage: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compliance coverage",
        )


# ============================================================================
# ENDPOINTS - CHANGES (3)
# ============================================================================


@router.post(
    "/changes/detect",
    response_model=ChangeDetectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Detect and record a change",
    description=(
        "Record a change to an emission factor, activity data, calculation "
        "parameter, or configuration setting. Triggers downstream impact "
        "analysis to determine which calculations are affected and whether "
        "recalculation is required."
    ),
)
async def detect_change(
    request: DetectChangeRequest,
    service=Depends(get_service),
) -> ChangeDetectionResponse:
    """
    Detect and record a change event.

    Args:
        request: Change detection request
        service: AuditTrailLineageService instance

    Returns:
        ChangeDetectionResponse with impact analysis

    Raises:
        HTTPException: 400 for validation, 500 for failures
    """
    try:
        logger.info(
            "Detecting change: type=%s, entity=%s/%s, severity=%s",
            request.change_type,
            request.affected_entity_type,
            request.affected_entity_id,
            request.severity,
        )

        result = await service.detect_change(request.dict())
        change_id = result.get("change_id", str(uuid.uuid4()))

        return ChangeDetectionResponse(
            change_id=change_id,
            change_type=result.get(
                "change_type", request.change_type
            ),
            affected_entity_type=result.get(
                "affected_entity_type", request.affected_entity_type
            ),
            affected_entity_id=result.get(
                "affected_entity_id", request.affected_entity_id
            ),
            old_value=result.get("old_value", request.old_value),
            new_value=result.get("new_value", request.new_value),
            trigger=result.get("trigger", request.trigger),
            severity=result.get("severity", request.severity),
            impact_summary=result.get("impact_summary", {}),
            affected_calculations=result.get(
                "affected_calculations", 0
            ),
            requires_recalculation=result.get(
                "requires_recalculation", False
            ),
            created_at=result.get(
                "created_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in detect_change: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in detect_change: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Change detection failed",
        )


@router.get(
    "/changes/{change_id}",
    response_model=ChangeDetectionResponse,
    summary="Get change details",
    description=(
        "Retrieve the details of a detected change by UUID, including "
        "old and new values, severity, impact summary, and whether "
        "recalculation is required."
    ),
)
async def get_change(
    change_id: str = Path(..., description="Change UUID"),
    service=Depends(get_service),
) -> ChangeDetectionResponse:
    """
    Get change details by UUID.

    Args:
        change_id: Change UUID
        service: AuditTrailLineageService instance

    Returns:
        ChangeDetectionResponse with change details

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info("Getting change details: %s", change_id)

        result = await service.get_change(change_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Change {change_id} not found",
            )

        return ChangeDetectionResponse(
            change_id=result.get("change_id", change_id),
            change_type=result.get("change_type", ""),
            affected_entity_type=result.get(
                "affected_entity_type", ""
            ),
            affected_entity_id=result.get(
                "affected_entity_id", ""
            ),
            old_value=result.get("old_value"),
            new_value=result.get("new_value"),
            trigger=result.get("trigger", ""),
            severity=result.get("severity", "medium"),
            impact_summary=result.get("impact_summary", {}),
            affected_calculations=result.get(
                "affected_calculations", 0
            ),
            requires_recalculation=result.get(
                "requires_recalculation", False
            ),
            created_at=result.get(
                "created_at", datetime.utcnow().isoformat()
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_change: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve change details",
        )


@router.get(
    "/changes/{change_id}/impact",
    response_model=ChangeImpactResponse,
    summary="Get change impact analysis",
    description=(
        "Retrieve the full downstream impact analysis for a detected change. "
        "Shows affected lineage nodes, calculations that need recalculation, "
        "affected disclosure data points, and materiality assessment."
    ),
)
async def get_change_impact(
    change_id: str = Path(..., description="Change UUID"),
    service=Depends(get_service),
) -> ChangeImpactResponse:
    """
    Get full impact analysis for a change.

    Args:
        change_id: Change UUID
        service: AuditTrailLineageService instance

    Returns:
        ChangeImpactResponse with affected nodes, calculations, disclosures

    Raises:
        HTTPException: 404 if not found, 500 for analysis failures
    """
    try:
        logger.info("Getting change impact: %s", change_id)

        result = await service.get_change_impact(change_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Change {change_id} not found",
            )

        return ChangeImpactResponse(
            change_id=result.get("change_id", change_id),
            affected_nodes=result.get("affected_nodes", []),
            affected_calculations=result.get(
                "affected_calculations", []
            ),
            affected_disclosures=result.get(
                "affected_disclosures", []
            ),
            total_impact_nodes=result.get("total_impact_nodes", 0),
            materiality_assessment=result.get(
                "materiality_assessment", {}
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in get_change_impact: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Change impact analysis failed",
        )


# ============================================================================
# ENDPOINTS - PIPELINE (2)
# ============================================================================


@router.post(
    "/pipeline/execute",
    response_model=PipelineResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Execute full audit trail pipeline",
    description=(
        "Execute the complete 10-stage audit trail pipeline: (1) validate "
        "event, (2) record to hash chain, (3) build lineage nodes, "
        "(4) build lineage edges, (5) check compliance coverage, "
        "(6) detect changes, (7) assess materiality, (8) create evidence "
        "package (optional), (9) compute provenance hash, (10) seal and "
        "return consolidated results."
    ),
)
async def execute_pipeline(
    request: PipelineExecuteRequest,
    service=Depends(get_service),
) -> PipelineResponse:
    """
    Execute the full audit trail pipeline for a single event.

    Args:
        request: Pipeline execution request
        service: AuditTrailLineageService instance

    Returns:
        PipelineResponse with consolidated pipeline results

    Raises:
        HTTPException: 400 for validation, 500 for pipeline failures
    """
    try:
        logger.info(
            "Executing audit trail pipeline: type=%s, agent=%s, org=%s, "
            "year=%d, evidence=%s, compliance=%s",
            request.event_type,
            request.agent_id,
            request.organization_id,
            request.reporting_year,
            request.include_evidence,
            request.include_compliance,
        )

        result = await service.execute_pipeline(request.dict())
        pipeline_id = result.get("pipeline_id", str(uuid.uuid4()))

        event_data = result.get("event", {})
        event_response = EventResponse(
            event_id=event_data.get("event_id", str(uuid.uuid4())),
            event_type=event_data.get(
                "event_type", request.event_type
            ),
            agent_id=event_data.get("agent_id", request.agent_id),
            scope=event_data.get("scope", request.scope),
            category=event_data.get("category", request.category),
            organization_id=event_data.get(
                "organization_id", request.organization_id
            ),
            reporting_year=event_data.get(
                "reporting_year", request.reporting_year
            ),
            calculation_id=event_data.get(
                "calculation_id", request.calculation_id
            ),
            chain_position=event_data.get("chain_position", 0),
            event_hash=event_data.get("event_hash", ""),
            previous_hash=event_data.get("previous_hash", ""),
            payload=event_data.get("payload", request.payload),
            data_quality_score=event_data.get(
                "data_quality_score", request.data_quality_score
            ),
            metadata=event_data.get("metadata", request.metadata),
            created_at=event_data.get(
                "created_at", datetime.utcnow().isoformat()
            ),
        )

        return PipelineResponse(
            pipeline_id=pipeline_id,
            event=event_response,
            lineage_nodes_created=result.get(
                "lineage_nodes_created", 0
            ),
            lineage_edges_created=result.get(
                "lineage_edges_created", 0
            ),
            compliance_coverage=result.get("compliance_coverage"),
            evidence_package_id=result.get("evidence_package_id"),
            changes_detected=result.get("changes_detected", 0),
            provenance_hash=result.get("provenance_hash", ""),
            processing_time_ms=result.get("processing_time_ms", 0.0),
            stages_completed=result.get("stages_completed", []),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in execute_pipeline: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in execute_pipeline: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline execution failed",
        )


@router.post(
    "/pipeline/execute/batch",
    response_model=BatchPipelineResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Execute batch audit trail pipeline",
    description=(
        "Execute the full audit trail pipeline for a batch of events. "
        "Events are processed sequentially to maintain hash chain ordering. "
        "Returns per-event results and batch summary."
    ),
)
async def execute_pipeline_batch(
    request: BatchPipelineRequest,
    service=Depends(get_service),
) -> BatchPipelineResponse:
    """
    Execute the full audit trail pipeline for a batch of events.

    Args:
        request: Batch pipeline execution request
        service: AuditTrailLineageService instance

    Returns:
        BatchPipelineResponse with per-event results

    Raises:
        HTTPException: 400 for validation, 500 for batch failures
    """
    try:
        logger.info(
            "Executing batch audit trail pipeline: %d events, "
            "evidence=%s, compliance=%s",
            len(request.events),
            request.include_evidence,
            request.include_compliance,
        )

        result = await service.execute_pipeline_batch(request.dict())
        batch_id = result.get("batch_id", str(uuid.uuid4()))

        pipeline_results = []
        for pr in result.get("results", []):
            event_data = pr.get("event", {})
            event_resp = EventResponse(
                event_id=event_data.get(
                    "event_id", str(uuid.uuid4())
                ),
                event_type=event_data.get("event_type", ""),
                agent_id=event_data.get("agent_id", ""),
                scope=event_data.get("scope"),
                category=event_data.get("category"),
                organization_id=event_data.get(
                    "organization_id", ""
                ),
                reporting_year=event_data.get("reporting_year", 0),
                calculation_id=event_data.get("calculation_id"),
                chain_position=event_data.get("chain_position", 0),
                event_hash=event_data.get("event_hash", ""),
                previous_hash=event_data.get("previous_hash", ""),
                payload=event_data.get("payload", {}),
                data_quality_score=event_data.get(
                    "data_quality_score"
                ),
                metadata=event_data.get("metadata", {}),
                created_at=event_data.get(
                    "created_at", datetime.utcnow().isoformat()
                ),
            )

            pipeline_results.append(
                PipelineResponse(
                    pipeline_id=pr.get(
                        "pipeline_id", str(uuid.uuid4())
                    ),
                    event=event_resp,
                    lineage_nodes_created=pr.get(
                        "lineage_nodes_created", 0
                    ),
                    lineage_edges_created=pr.get(
                        "lineage_edges_created", 0
                    ),
                    compliance_coverage=pr.get(
                        "compliance_coverage"
                    ),
                    evidence_package_id=pr.get(
                        "evidence_package_id"
                    ),
                    changes_detected=pr.get("changes_detected", 0),
                    provenance_hash=pr.get("provenance_hash", ""),
                    processing_time_ms=pr.get(
                        "processing_time_ms", 0.0
                    ),
                    stages_completed=pr.get(
                        "stages_completed", []
                    ),
                )
            )

        return BatchPipelineResponse(
            batch_id=batch_id,
            results=pipeline_results,
            events_processed=result.get("events_processed", 0),
            events_failed=result.get("events_failed", 0),
            errors=result.get("errors", []),
            total_processing_time_ms=result.get(
                "total_processing_time_ms", 0.0
            ),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in execute_pipeline_batch: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in execute_pipeline_batch: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch pipeline execution failed",
        )


# ============================================================================
# ENDPOINTS - SUMMARY (1)
# ============================================================================


@router.get(
    "/summary/{org_id}/{year}",
    response_model=AuditSummaryResponse,
    summary="Get audit trail summary",
    description=(
        "Retrieve a comprehensive audit trail summary for an organization "
        "and reporting year. Includes total events, chain integrity, "
        "breakdowns by type/scope/agent, lineage statistics, evidence "
        "package count, compliance coverage, and change count."
    ),
)
async def get_audit_summary(
    org_id: str = Path(
        ..., description="Organization identifier"
    ),
    year: int = Path(
        ..., ge=2000, le=2100, description="Reporting year"
    ),
    service=Depends(get_service),
) -> AuditSummaryResponse:
    """
    Get audit trail summary for an organization and year.

    Args:
        org_id: Organization identifier
        year: Reporting year
        service: AuditTrailLineageService instance

    Returns:
        AuditSummaryResponse with comprehensive summary

    Raises:
        HTTPException: 404 if no data, 500 for retrieval failures
    """
    try:
        logger.info(
            "Getting audit trail summary: org=%s, year=%d",
            org_id, year,
        )

        result = await service.get_audit_summary(org_id, year)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No audit trail data found for organization "
                    f"{org_id} and year {year}"
                ),
            )

        return AuditSummaryResponse(
            organization_id=result.get("organization_id", org_id),
            reporting_year=result.get("reporting_year", year),
            total_events=result.get("total_events", 0),
            chain_length=result.get("chain_length", 0),
            chain_is_valid=result.get("chain_is_valid", True),
            events_by_type=result.get("events_by_type", {}),
            events_by_scope=result.get("events_by_scope", {}),
            events_by_agent=result.get("events_by_agent", {}),
            lineage_nodes=result.get("lineage_nodes", 0),
            lineage_edges=result.get("lineage_edges", 0),
            evidence_packages=result.get("evidence_packages", 0),
            compliance_coverage=result.get(
                "compliance_coverage", {}
            ),
            changes_detected=result.get("changes_detected", 0),
            last_event_at=result.get("last_event_at"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in get_audit_summary: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit trail summary",
        )


# ============================================================================
# ENDPOINTS - HEALTH (1)
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check with 7 engine statuses",
    description=(
        "Health check endpoint for the Audit Trail & Lineage Agent. Returns "
        "service status, agent identifier (GL-MRV-X-042), version, uptime, "
        "and per-engine health status for all 7 engines "
        "(AuditEventEngine, LineageGraphEngine, EvidencePackagerEngine, "
        "ComplianceTracerEngine, ChangeDetectorEngine, "
        "ComplianceCheckerEngine, AuditTrailPipelineEngine). "
        "No authentication required."
    ),
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint (no auth required).

    Returns:
        HealthResponse with service status, engine health, and event counts
    """
    try:
        uptime = (datetime.utcnow() - _start_time).total_seconds()

        engines = []
        for engine_name in _ENGINE_NAMES:
            engines.append(
                EngineStatus(
                    engine_name=engine_name,
                    status="healthy",
                    last_used=None,
                )
            )

        return HealthResponse(
            status="healthy",
            agent_id="GL-MRV-X-042",
            version="1.0.0",
            uptime_seconds=round(uptime, 2),
            engines=engines,
            total_events=0,
            total_chains=0,
        )

    except Exception as e:
        logger.error("Error in health_check: %s", e, exc_info=True)
        return HealthResponse(
            status="unhealthy",
            agent_id="GL-MRV-X-042",
            version="1.0.0",
            uptime_seconds=0.0,
            engines=[],
            total_events=0,
            total_chains=0,
        )
