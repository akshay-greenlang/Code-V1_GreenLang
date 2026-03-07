# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-001 Supply Chain Mapper

Additional Pydantic v2 request/response models specific to the REST API layer.
Core domain models are imported from the main models module; this file defines
API-level wrappers, paginated list responses, and onboarding models.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001 Supply Chain Mapping Master (GL-EUDR-SCM-001)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from greenlang.agents.eudr.supply_chain_mapper.models import (
    ComplianceStatus,
    CustodyModel,
    DDSExportData,
    EUDRCommodity,
    GapAnalysisResult,
    GapSeverity,
    GapType,
    GraphLayoutData,
    NodeType,
    RiskLevel,
    RiskPropagationResult,
    RiskSummary,
    SankeyData,
    SupplyChainEdge,
    SupplyChainGap,
    SupplyChainGraph,
    SupplyChainNode,
    TierDistribution,
    TraceResult,
    TransportMode,
)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# Paginated List Responses
# =============================================================================


class PaginatedMeta(BaseModel):
    """Pagination metadata for list responses."""

    total: int = Field(..., ge=0, description="Total number of results")
    limit: int = Field(..., ge=1, description="Maximum results returned")
    offset: int = Field(..., ge=0, description="Results skipped")
    has_more: bool = Field(..., description="Whether more results exist")


class GraphListResponse(BaseModel):
    """Paginated response for listing supply chain graphs."""

    graphs: List[GraphSummary] = Field(
        default_factory=list, description="List of graph summaries"
    )
    meta: PaginatedMeta = Field(..., description="Pagination metadata")

    model_config = ConfigDict(from_attributes=True)


class GraphSummary(BaseModel):
    """Summary view of a supply chain graph for list endpoints."""

    graph_id: str = Field(..., description="Unique graph identifier")
    operator_id: str = Field(..., description="Owner operator ID")
    commodity: EUDRCommodity = Field(..., description="Primary commodity")
    graph_name: Optional[str] = Field(None, description="Human-readable name")
    total_nodes: int = Field(default=0, ge=0)
    total_edges: int = Field(default=0, ge=0)
    max_tier_depth: int = Field(default=0, ge=0)
    traceability_score: float = Field(default=0.0, ge=0.0, le=100.0)
    compliance_readiness: float = Field(default=0.0, ge=0.0, le=100.0)
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = ConfigDict(from_attributes=True)


# Fix forward reference
GraphListResponse.model_rebuild()


class GapListResponse(BaseModel):
    """Paginated response for listing gaps."""

    gaps: List[SupplyChainGap] = Field(
        default_factory=list, description="List of compliance gaps"
    )
    meta: PaginatedMeta = Field(..., description="Pagination metadata")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Graph CRUD API Models
# =============================================================================


class GraphCreateRequest(BaseModel):
    """Request to create a new supply chain graph."""

    commodity: EUDRCommodity = Field(
        ..., description="Primary EUDR commodity for this graph"
    )
    graph_name: Optional[str] = Field(
        None, max_length=500, description="Human-readable name"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "commodity": "cocoa",
                    "graph_name": "Ghana Cocoa Supply Chain Q1 2026",
                }
            ]
        },
    )


class GraphCreateResponse(BaseModel):
    """Response after creating a supply chain graph."""

    graph_id: str = Field(..., description="Unique graph identifier")
    operator_id: str = Field(..., description="Owner operator ID")
    commodity: EUDRCommodity = Field(..., description="Primary commodity")
    graph_name: Optional[str] = Field(None)
    status: str = Field(default="created")
    created_at: datetime = Field(default_factory=_utcnow)

    model_config = ConfigDict(from_attributes=True)


class GraphDeleteResponse(BaseModel):
    """Response after deleting a supply chain graph."""

    graph_id: str = Field(..., description="Deleted graph identifier")
    status: str = Field(default="deleted")
    deleted_at: datetime = Field(default_factory=_utcnow)


# =============================================================================
# Mapping API Models
# =============================================================================


class DiscoverRequest(BaseModel):
    """Request to trigger multi-tier recursive discovery."""

    max_depth: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum tier depth to discover",
    )
    include_certifications: bool = Field(
        default=True,
        description="Include certification body nodes",
    )
    commodity_filter: Optional[EUDRCommodity] = Field(
        None,
        description="Filter discovery to specific commodity",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "max_depth": 5,
                    "include_certifications": True,
                    "commodity_filter": "cocoa",
                }
            ]
        },
    )


class DiscoverResponse(BaseModel):
    """Response from multi-tier discovery operation."""

    graph_id: str = Field(..., description="Graph ID")
    tiers_discovered: int = Field(default=0, ge=0)
    new_nodes_added: int = Field(default=0, ge=0)
    new_edges_added: int = Field(default=0, ge=0)
    opaque_segments: int = Field(
        default=0, ge=0, description="Sub-tier segments without visibility"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    status: str = Field(default="completed")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Risk API Models
# =============================================================================


class RiskPropagateRequest(BaseModel):
    """Request to trigger risk propagation across the graph."""

    risk_weights: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Custom risk weights. Keys: country, commodity, supplier, "
            "deforestation. Must sum to 1.0. If None, uses config defaults."
        ),
    )
    propagation_source: str = Field(
        default="api_request",
        description="Source identifier for audit trail",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "risk_weights": {
                        "country": 0.30,
                        "commodity": 0.20,
                        "supplier": 0.25,
                        "deforestation": 0.25,
                    },
                    "propagation_source": "quarterly_review",
                }
            ]
        },
    )

    @field_validator("risk_weights")
    @classmethod
    def validate_risk_weights(
        cls, v: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        """Validate risk weights sum to 1.0 if provided."""
        if v is None:
            return v
        required_keys = {"country", "commodity", "supplier", "deforestation"}
        if set(v.keys()) != required_keys:
            raise ValueError(
                f"risk_weights must contain exactly: {required_keys}"
            )
        weight_sum = sum(v.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(
                f"risk_weights must sum to 1.0, got {weight_sum:.4f}"
            )
        return v


class RiskPropagateResponse(BaseModel):
    """Response from risk propagation operation."""

    graph_id: str = Field(..., description="Graph ID")
    nodes_updated: int = Field(default=0, ge=0)
    propagation_results: List[RiskPropagationResult] = Field(
        default_factory=list
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    status: str = Field(default="completed")

    model_config = ConfigDict(from_attributes=True)


class RiskHeatmapResponse(BaseModel):
    """Risk heatmap data for visualization."""

    graph_id: str = Field(..., description="Graph ID")
    heatmap_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {node_id, risk_score, risk_level, lat, lon} entries",
    )
    risk_distribution: Dict[str, int] = Field(
        default_factory=lambda: {"low": 0, "standard": 0, "high": 0}
    )
    generated_at: datetime = Field(default_factory=_utcnow)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Gap Analysis API Models
# =============================================================================


class GapAnalyzeRequest(BaseModel):
    """Request to trigger gap analysis on a graph."""

    include_resolved: bool = Field(
        default=False,
        description="Include previously resolved gaps in analysis",
    )
    severity_filter: Optional[GapSeverity] = Field(
        None,
        description="Only analyze gaps of this severity or higher",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "include_resolved": False,
                    "severity_filter": "high",
                }
            ]
        },
    )


class GapResolveRequest(BaseModel):
    """Request to mark a gap as resolved."""

    resolution_notes: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Explanation of how the gap was resolved",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of supporting evidence documents",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "resolution_notes": (
                        "GPS coordinates added for all producer plots "
                        "via satellite imagery verification"
                    ),
                    "evidence_ids": ["ev-001", "ev-002"],
                }
            ]
        },
    )

    @field_validator("resolution_notes")
    @classmethod
    def validate_resolution_notes(cls, v: str) -> str:
        """Validate resolution notes are non-empty."""
        if not v or not v.strip():
            raise ValueError("resolution_notes must be non-empty")
        return v


class GapResolveResponse(BaseModel):
    """Response after resolving a gap."""

    gap_id: str = Field(..., description="Resolved gap ID")
    graph_id: str = Field(..., description="Parent graph ID")
    status: str = Field(default="resolved")
    resolved_at: datetime = Field(default_factory=_utcnow)
    compliance_readiness: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Updated compliance readiness score",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Visualization API Models
# =============================================================================


class LayoutRequest(BaseModel):
    """Optional query params for layout generation."""

    algorithm: str = Field(
        default="force_directed",
        description="Layout algorithm: force_directed, hierarchical, radial",
    )


# =============================================================================
# Onboarding API Models
# =============================================================================


class OnboardingInviteRequest(BaseModel):
    """Request to invite a supplier for onboarding."""

    supplier_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Legal name of the supplier to invite",
    )
    supplier_email: str = Field(
        ...,
        min_length=5,
        max_length=320,
        description="Email address of the supplier contact",
    )
    supplier_country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR commodity the supplier provides",
    )
    graph_id: Optional[str] = Field(
        None,
        description="Graph ID to associate the supplier with",
    )
    message: Optional[str] = Field(
        None,
        max_length=2000,
        description="Custom invitation message",
    )
    expires_in_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days until the invitation expires",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "supplier_name": "Cooperative Alpha GH",
                    "supplier_email": "contact@cooperative-alpha.gh",
                    "supplier_country": "GH",
                    "commodity": "cocoa",
                    "graph_id": "graph-001",
                    "message": "Please complete your EUDR supply chain profile.",
                    "expires_in_days": 30,
                }
            ]
        },
    )

    @field_validator("supplier_country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "supplier_country must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @field_validator("supplier_name")
    @classmethod
    def validate_supplier_name(cls, v: str) -> str:
        """Validate supplier name is non-empty."""
        if not v or not v.strip():
            raise ValueError("supplier_name must be non-empty")
        return v


class OnboardingInviteResponse(BaseModel):
    """Response after creating a supplier onboarding invitation."""

    invitation_id: str = Field(..., description="Unique invitation ID")
    token: str = Field(..., description="Secure token for the supplier link")
    supplier_name: str = Field(...)
    supplier_email: str = Field(...)
    status: str = Field(default="pending")
    expires_at: datetime = Field(default_factory=_utcnow)
    onboarding_url: str = Field(
        default="", description="URL for the supplier to complete onboarding"
    )

    model_config = ConfigDict(from_attributes=True)


class OnboardingStatusResponse(BaseModel):
    """Response for checking onboarding invitation status."""

    invitation_id: str = Field(...)
    supplier_name: str = Field(...)
    supplier_email: str = Field(...)
    status: str = Field(default="pending")
    commodity: EUDRCommodity = Field(...)
    supplier_country: str = Field(...)
    graph_id: Optional[str] = Field(None)
    expires_at: datetime = Field(default_factory=_utcnow)
    submitted_at: Optional[datetime] = Field(None)

    model_config = ConfigDict(from_attributes=True)


class OnboardingSubmitRequest(BaseModel):
    """Request from a supplier submitting their onboarding data."""

    operator_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Legal name of the operator",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    region: Optional[str] = Field(None, max_length=200)
    coordinates: Optional[Tuple[float, float]] = Field(
        None,
        description="GPS coordinates (lat, lon) in WGS84",
    )
    commodities: List[EUDRCommodity] = Field(
        default_factory=list,
        description="Commodities handled",
    )
    certifications: List[str] = Field(
        default_factory=list,
        description="Certifications held (FSC, RSPO, RA, etc.)",
    )
    plot_ids: List[str] = Field(
        default_factory=list,
        description="Production plot IDs (for producers)",
    )
    node_type: NodeType = Field(
        default=NodeType.PRODUCER,
        description="Role in the supply chain",
    )
    sub_suppliers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of sub-tier suppliers for recursive mapping",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "operator_name": "Cooperative Alpha GH",
                    "country_code": "GH",
                    "region": "Ashanti",
                    "coordinates": [6.6885, -1.6244],
                    "commodities": ["cocoa"],
                    "certifications": ["RA-2024-GH-001"],
                    "plot_ids": ["plot-gh-001", "plot-gh-002"],
                    "node_type": "producer",
                    "sub_suppliers": [],
                }
            ]
        },
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v


class OnboardingSubmitResponse(BaseModel):
    """Response after a supplier submits their onboarding data."""

    invitation_id: str = Field(...)
    node_id: Optional[str] = Field(
        None, description="Node ID if automatically added to graph"
    )
    status: str = Field(default="submitted")
    submitted_at: datetime = Field(default_factory=_utcnow)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Batch Trace API Models
# =============================================================================


class BatchTraceResponse(BaseModel):
    """Response for batch-level traceability query."""

    batch_id: str = Field(..., description="Batch/lot identifier queried")
    graph_id: str = Field(..., description="Graph ID searched")
    edges: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Edges associated with this batch",
    )
    origin_nodes: List[str] = Field(
        default_factory=list,
        description="Origin node IDs for this batch",
    )
    destination_nodes: List[str] = Field(
        default_factory=list,
        description="Destination node IDs for this batch",
    )
    total_quantity: Optional[str] = Field(
        None, description="Total quantity traced"
    )
    custody_model: Optional[str] = Field(None)
    is_complete: bool = Field(
        default=True, description="Whether full chain traced"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Health Check
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    agent_id: str = Field(default="GL-EUDR-SCM-001")
    agent_name: str = Field(default="EUDR Supply Chain Mapping Master")
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=_utcnow)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "BatchTraceResponse",
    "DiscoverRequest",
    "DiscoverResponse",
    "GapAnalyzeRequest",
    "GapListResponse",
    "GapResolveRequest",
    "GapResolveResponse",
    "GraphCreateRequest",
    "GraphCreateResponse",
    "GraphDeleteResponse",
    "GraphListResponse",
    "GraphSummary",
    "HealthResponse",
    "LayoutRequest",
    "OnboardingInviteRequest",
    "OnboardingInviteResponse",
    "OnboardingStatusResponse",
    "OnboardingSubmitRequest",
    "OnboardingSubmitResponse",
    "PaginatedMeta",
    "RiskHeatmapResponse",
    "RiskPropagateRequest",
    "RiskPropagateResponse",
]
