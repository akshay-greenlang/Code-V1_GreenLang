# -*- coding: utf-8 -*-
"""
Threat Modeling API Routes - SEC-010 Phase 2

FastAPI routes for threat modeling functionality including:
- GET /threats - List threat models (pagination, filters)
- POST /threats - Create new threat model
- GET /threats/{id} - Get threat model details
- PUT /threats/{id} - Update threat model
- DELETE /threats/{id} - Delete (draft only)
- POST /threats/{id}/analyze - Run STRIDE analysis
- POST /threats/{id}/components - Add component
- PUT /threats/{id}/components/{cid} - Update component
- POST /threats/{id}/data-flows - Add data flow
- POST /threats/{id}/mitigations - Add mitigation
- PUT /threats/{id}/approve - Approve threat model
- GET /threats/{id}/report - Generate PDF/JSON report

Requires secops:threats:read or secops:threats:write permissions.

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator

from greenlang.infrastructure.threat_modeling.models import (
    Component,
    ComponentType,
    DataClassification,
    DataFlow,
    Mitigation,
    MitigationStatus,
    Threat,
    ThreatCategory,
    ThreatModel,
    ThreatModelStatus,
    ThreatStatus,
    TrustBoundary,
)
from greenlang.infrastructure.threat_modeling.stride_engine import STRIDEEngine
from greenlang.infrastructure.threat_modeling.attack_surface import AttackSurfaceMapper
from greenlang.infrastructure.threat_modeling.dfd_validator import DataFlowValidator
from greenlang.infrastructure.threat_modeling.risk_scorer import RiskScorer
from greenlang.infrastructure.threat_modeling.control_mapper import ControlMapper
from greenlang.infrastructure.threat_modeling.metrics import (
    record_threat_model_created,
    update_threat_counts,
    update_severity_counts,
    update_risk_score,
    record_analysis_duration,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/secops/threats",
    tags=["Threat Modeling"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"},
    },
)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class CreateThreatModelRequest(BaseModel):
    """Request to create a new threat model."""

    service_name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Name of the service being modeled",
    )
    version: str = Field(
        default="1.0.0",
        max_length=32,
        description="Version of the threat model",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Description of the system being modeled",
    )
    scope: str = Field(
        default="",
        max_length=2048,
        description="Scope and boundaries of the threat model",
    )
    owner: str = Field(
        default="",
        max_length=256,
        description="Team responsible for this threat model",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )

    @field_validator("service_name")
    @classmethod
    def validate_service_name(cls, v: str) -> str:
        """Normalize service name to lowercase."""
        return v.strip().lower().replace(" ", "-")


class UpdateThreatModelRequest(BaseModel):
    """Request to update a threat model."""

    version: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Updated version",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=4096,
        description="Updated description",
    )
    scope: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="Updated scope",
    )
    owner: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Updated owner",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Updated tags",
    )
    status: Optional[ThreatModelStatus] = Field(
        default=None,
        description="Updated status",
    )


class AddComponentRequest(BaseModel):
    """Request to add a component to a threat model."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Component name",
    )
    component_type: ComponentType = Field(
        default=ComponentType.UNKNOWN,
        description="Type of the component",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Component description",
    )
    trust_level: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Trust level (1-4)",
    )
    data_classification: DataClassification = Field(
        default=DataClassification.INTERNAL,
        description="Data classification",
    )
    owner: str = Field(
        default="",
        max_length=256,
        description="Component owner",
    )
    technology_stack: List[str] = Field(
        default_factory=list,
        description="Technologies used",
    )
    is_external: bool = Field(
        default=False,
        description="Whether this is an external component",
    )
    network_zone: str = Field(
        default="internal",
        max_length=64,
        description="Network zone",
    )


class UpdateComponentRequest(BaseModel):
    """Request to update a component."""

    name: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Updated name",
    )
    component_type: Optional[ComponentType] = Field(
        default=None,
        description="Updated component type",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=4096,
        description="Updated description",
    )
    trust_level: Optional[int] = Field(
        default=None,
        ge=1,
        le=4,
        description="Updated trust level",
    )
    data_classification: Optional[DataClassification] = Field(
        default=None,
        description="Updated data classification",
    )


class AddDataFlowRequest(BaseModel):
    """Request to add a data flow."""

    source_component_id: str = Field(
        ...,
        min_length=1,
        description="Source component ID",
    )
    destination_component_id: str = Field(
        ...,
        min_length=1,
        description="Destination component ID",
    )
    data_type: str = Field(
        default="generic",
        max_length=256,
        description="Type of data being transferred",
    )
    protocol: str = Field(
        default="https",
        max_length=64,
        description="Communication protocol",
    )
    encryption: bool = Field(
        default=True,
        description="Whether the flow is encrypted",
    )
    authentication_required: bool = Field(
        default=True,
        description="Whether authentication is required",
    )
    authentication_method: str = Field(
        default="jwt",
        max_length=64,
        description="Authentication method",
    )
    authorization_required: bool = Field(
        default=True,
        description="Whether authorization is required",
    )
    description: str = Field(
        default="",
        max_length=2048,
        description="Flow description",
    )
    data_classification: DataClassification = Field(
        default=DataClassification.INTERNAL,
        description="Data classification",
    )
    bidirectional: bool = Field(
        default=False,
        description="Whether data flows both ways",
    )


class AddMitigationRequest(BaseModel):
    """Request to add a mitigation."""

    threat_id: str = Field(
        ...,
        min_length=1,
        description="ID of the threat to mitigate",
    )
    control_id: str = Field(
        default="",
        max_length=64,
        description="Security control ID",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Mitigation title",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Mitigation description",
    )
    effectiveness: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Estimated effectiveness percentage",
    )
    owner: str = Field(
        default="",
        max_length=256,
        description="Mitigation owner",
    )
    priority: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Implementation priority",
    )
    due_date: Optional[datetime] = Field(
        default=None,
        description="Target implementation date",
    )


class RunAnalysisRequest(BaseModel):
    """Request to run STRIDE analysis."""

    include_generic_threats: bool = Field(
        default=True,
        description="Whether to include generic STRIDE threats",
    )
    recalculate_scores: bool = Field(
        default=True,
        description="Whether to recalculate risk scores",
    )
    validate_dfd: bool = Field(
        default=True,
        description="Whether to validate the DFD",
    )


class ApproveRequest(BaseModel):
    """Request to approve a threat model."""

    approved_by: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Who is approving the model",
    )
    comments: str = Field(
        default="",
        max_length=4096,
        description="Approval comments",
    )


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class ComponentResponse(BaseModel):
    """Response for a component."""

    id: str
    name: str
    component_type: ComponentType
    description: str
    trust_level: int
    data_classification: DataClassification
    owner: str
    is_external: bool
    network_zone: str


class DataFlowResponse(BaseModel):
    """Response for a data flow."""

    id: str
    source_component_id: str
    destination_component_id: str
    data_type: str
    protocol: str
    encryption: bool
    authentication_required: bool
    data_classification: DataClassification
    crosses_trust_boundary: bool


class ThreatResponse(BaseModel):
    """Response for a threat."""

    id: str
    category: ThreatCategory
    title: str
    description: str
    component_id: Optional[str]
    data_flow_id: Optional[str]
    likelihood: int
    impact: int
    risk_score: float
    severity: str
    status: ThreatStatus
    cvss_score: Optional[float]
    cwe_ids: List[str]
    countermeasures: List[str]


class MitigationResponse(BaseModel):
    """Response for a mitigation."""

    id: str
    threat_id: str
    control_id: str
    title: str
    description: str
    status: MitigationStatus
    effectiveness: float
    owner: str
    priority: int


class ThreatModelResponse(BaseModel):
    """Summary response for a threat model."""

    id: str
    service_name: str
    version: str
    status: ThreatModelStatus
    overall_risk_score: float
    threat_count: int
    mitigated_count: int
    critical_count: int
    high_count: int
    owner: str
    created_at: datetime
    updated_at: datetime
    approved_by: Optional[str]
    approved_at: Optional[datetime]


class ThreatModelListResponse(BaseModel):
    """Response for listing threat models."""

    total: int
    page: int
    page_size: int
    items: List[ThreatModelResponse]


class ThreatModelDetailResponse(BaseModel):
    """Detailed response for a threat model."""

    id: str
    service_name: str
    version: str
    status: ThreatModelStatus
    description: str
    scope: str
    owner: str
    overall_risk_score: float
    category_scores: Dict[str, float]
    threat_count: int
    mitigated_count: int
    critical_count: int
    high_count: int
    components: List[ComponentResponse]
    data_flows: List[DataFlowResponse]
    threats: List[ThreatResponse]
    mitigations: List[MitigationResponse]
    tags: List[str]
    created_by: str
    created_at: datetime
    updated_at: datetime
    approved_by: Optional[str]
    approved_at: Optional[datetime]


class AnalysisResultResponse(BaseModel):
    """Response for STRIDE analysis."""

    threat_model_id: str
    threats_identified: int
    threats_by_category: Dict[str, int]
    threats_by_severity: Dict[str, int]
    overall_risk_score: float
    validation_passed: bool
    validation_errors: int
    validation_warnings: int
    analysis_duration_ms: float


class ReportResponse(BaseModel):
    """Response for report generation."""

    threat_model_id: str
    service_name: str
    format: str
    content: Optional[str] = None
    download_url: Optional[str] = None
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Storage (would be database in production)
# ---------------------------------------------------------------------------

_threat_models: Dict[str, ThreatModel] = {}


def _get_threat_model(threat_model_id: str) -> ThreatModel:
    """Get a threat model by ID or raise 404."""
    model = _threat_models.get(threat_model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Threat model {threat_model_id} not found",
        )
    return model


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=ThreatModelListResponse)
async def list_threat_models(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[ThreatModelStatus] = Query(None, description="Filter by status"),
    service_name: Optional[str] = Query(None, description="Filter by service name"),
) -> ThreatModelListResponse:
    """List all threat models with pagination and filtering.

    Requires: secops:threats:read permission.
    """
    logger.info("Listing threat models: page=%d, size=%d", page, page_size)

    # Filter models
    models = list(_threat_models.values())

    if status_filter:
        models = [m for m in models if m.status == status_filter]

    if service_name:
        models = [m for m in models if service_name.lower() in m.service_name.lower()]

    # Sort by updated_at descending
    models.sort(key=lambda m: m.updated_at, reverse=True)

    # Paginate
    total = len(models)
    start = (page - 1) * page_size
    end = start + page_size
    page_models = models[start:end]

    # Convert to response
    items = [
        ThreatModelResponse(
            id=m.id,
            service_name=m.service_name,
            version=m.version,
            status=m.status,
            overall_risk_score=m.overall_risk_score,
            threat_count=m.threat_count,
            mitigated_count=m.mitigated_count,
            critical_count=m.critical_count,
            high_count=m.high_count,
            owner=m.owner,
            created_at=m.created_at,
            updated_at=m.updated_at,
            approved_by=m.approved_by,
            approved_at=m.approved_at,
        )
        for m in page_models
    ]

    return ThreatModelListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=items,
    )


@router.post("", response_model=ThreatModelResponse, status_code=status.HTTP_201_CREATED)
async def create_threat_model(
    request: CreateThreatModelRequest,
) -> ThreatModelResponse:
    """Create a new threat model.

    Requires: secops:threats:write permission.
    """
    logger.info("Creating threat model for service: %s", request.service_name)

    # Check for duplicate service name
    for existing in _threat_models.values():
        if existing.service_name == request.service_name and existing.status != ThreatModelStatus.ARCHIVED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Threat model for service '{request.service_name}' already exists",
            )

    # Create threat model
    model = ThreatModel(
        id=str(uuid4()),
        service_name=request.service_name,
        version=request.version,
        description=request.description,
        scope=request.scope,
        owner=request.owner,
        tags=request.tags,
        status=ThreatModelStatus.DRAFT,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # Store
    _threat_models[model.id] = model

    # Record metrics
    record_threat_model_created(model.service_name)

    logger.info("Created threat model: id=%s, service=%s", model.id, model.service_name)

    return ThreatModelResponse(
        id=model.id,
        service_name=model.service_name,
        version=model.version,
        status=model.status,
        overall_risk_score=model.overall_risk_score,
        threat_count=model.threat_count,
        mitigated_count=model.mitigated_count,
        critical_count=model.critical_count,
        high_count=model.high_count,
        owner=model.owner,
        created_at=model.created_at,
        updated_at=model.updated_at,
        approved_by=model.approved_by,
        approved_at=model.approved_at,
    )


@router.get("/{threat_model_id}", response_model=ThreatModelDetailResponse)
async def get_threat_model(
    threat_model_id: str,
) -> ThreatModelDetailResponse:
    """Get threat model details.

    Requires: secops:threats:read permission.
    """
    logger.info("Getting threat model: %s", threat_model_id)

    model = _get_threat_model(threat_model_id)

    # Convert components
    components = [
        ComponentResponse(
            id=c.id,
            name=c.name,
            component_type=c.component_type,
            description=c.description,
            trust_level=c.trust_level,
            data_classification=c.data_classification,
            owner=c.owner,
            is_external=c.is_external,
            network_zone=c.network_zone,
        )
        for c in model.components
    ]

    # Convert data flows
    data_flows = [
        DataFlowResponse(
            id=f.id,
            source_component_id=f.source_component_id,
            destination_component_id=f.destination_component_id,
            data_type=f.data_type,
            protocol=f.protocol,
            encryption=f.encryption,
            authentication_required=f.authentication_required,
            data_classification=f.data_classification,
            crosses_trust_boundary=f.crosses_trust_boundary,
        )
        for f in model.data_flows
    ]

    # Convert threats
    threats = [
        ThreatResponse(
            id=t.id,
            category=t.category,
            title=t.title,
            description=t.description,
            component_id=t.component_id,
            data_flow_id=t.data_flow_id,
            likelihood=t.likelihood,
            impact=t.impact,
            risk_score=t.risk_score,
            severity=t.severity,
            status=t.status,
            cvss_score=t.cvss_score,
            cwe_ids=t.cwe_ids,
            countermeasures=t.countermeasures,
        )
        for t in model.threats
    ]

    # Convert mitigations
    mitigations = [
        MitigationResponse(
            id=m.id,
            threat_id=m.threat_id,
            control_id=m.control_id,
            title=m.title,
            description=m.description,
            status=m.status,
            effectiveness=m.effectiveness,
            owner=m.owner,
            priority=m.priority,
        )
        for m in model.mitigations
    ]

    return ThreatModelDetailResponse(
        id=model.id,
        service_name=model.service_name,
        version=model.version,
        status=model.status,
        description=model.description,
        scope=model.scope,
        owner=model.owner,
        overall_risk_score=model.overall_risk_score,
        category_scores=model.category_scores,
        threat_count=model.threat_count,
        mitigated_count=model.mitigated_count,
        critical_count=model.critical_count,
        high_count=model.high_count,
        components=components,
        data_flows=data_flows,
        threats=threats,
        mitigations=mitigations,
        tags=model.tags,
        created_by=model.created_by,
        created_at=model.created_at,
        updated_at=model.updated_at,
        approved_by=model.approved_by,
        approved_at=model.approved_at,
    )


@router.put("/{threat_model_id}", response_model=ThreatModelResponse)
async def update_threat_model(
    threat_model_id: str,
    request: UpdateThreatModelRequest,
) -> ThreatModelResponse:
    """Update a threat model.

    Requires: secops:threats:write permission.
    """
    logger.info("Updating threat model: %s", threat_model_id)

    model = _get_threat_model(threat_model_id)

    # Check if approved models can be updated
    if model.status == ThreatModelStatus.APPROVED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot update approved threat model. Create a new version instead.",
        )

    # Update fields
    if request.version is not None:
        model.version = request.version
    if request.description is not None:
        model.description = request.description
    if request.scope is not None:
        model.scope = request.scope
    if request.owner is not None:
        model.owner = request.owner
    if request.tags is not None:
        model.tags = request.tags
    if request.status is not None:
        model.status = request.status

    model.updated_at = datetime.now(timezone.utc)

    # Store
    _threat_models[model.id] = model

    return ThreatModelResponse(
        id=model.id,
        service_name=model.service_name,
        version=model.version,
        status=model.status,
        overall_risk_score=model.overall_risk_score,
        threat_count=model.threat_count,
        mitigated_count=model.mitigated_count,
        critical_count=model.critical_count,
        high_count=model.high_count,
        owner=model.owner,
        created_at=model.created_at,
        updated_at=model.updated_at,
        approved_by=model.approved_by,
        approved_at=model.approved_at,
    )


@router.delete("/{threat_model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_threat_model(
    threat_model_id: str,
) -> None:
    """Delete a threat model (draft only).

    Requires: secops:threats:write permission.
    """
    logger.info("Deleting threat model: %s", threat_model_id)

    model = _get_threat_model(threat_model_id)

    if model.status != ThreatModelStatus.DRAFT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only draft threat models can be deleted. Archive instead.",
        )

    del _threat_models[threat_model_id]
    logger.info("Deleted threat model: %s", threat_model_id)


@router.post("/{threat_model_id}/analyze", response_model=AnalysisResultResponse)
async def run_stride_analysis(
    threat_model_id: str,
    request: RunAnalysisRequest,
) -> AnalysisResultResponse:
    """Run STRIDE analysis on the threat model.

    Requires: secops:threats:write permission.
    """
    logger.info("Running STRIDE analysis for: %s", threat_model_id)
    start_time = datetime.now(timezone.utc)

    model = _get_threat_model(threat_model_id)

    # Initialize engines
    stride_engine = STRIDEEngine()
    risk_scorer = RiskScorer()
    dfd_validator = DataFlowValidator()

    # Run analysis
    all_threats: List[Threat] = []

    # Analyze components
    for component in model.components:
        component_threats = stride_engine.analyze_component(
            component,
            include_generic=request.include_generic_threats,
        )
        all_threats.extend(component_threats)

    # Analyze data flows
    for flow in model.data_flows:
        flow_threats = stride_engine.analyze_data_flow(flow)
        all_threats.extend(flow_threats)

    # Recalculate risk scores if requested
    if request.recalculate_scores:
        for threat in all_threats:
            likelihood = risk_scorer.calculate_likelihood(threat)
            impact = risk_scorer.calculate_impact(threat)
            threat.risk_score = risk_scorer.calculate_risk_score(threat, likelihood, impact)

    # Store threats in model
    model.threats = all_threats
    model.updated_at = datetime.now(timezone.utc)

    # Calculate category scores
    category_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for threat in all_threats:
        cat = threat.category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1
        severity_counts[threat.severity] = severity_counts.get(threat.severity, 0) + 1

    # Calculate overall risk score
    if all_threats:
        model.overall_risk_score = sum(t.risk_score for t in all_threats) / len(all_threats)
    else:
        model.overall_risk_score = 0.0

    # Validate DFD if requested
    validation_passed = True
    validation_errors = 0
    validation_warnings = 0

    if request.validate_dfd:
        validation_result = dfd_validator.validate_dfd(model)
        validation_passed = validation_result.is_valid
        validation_errors = validation_result.error_count
        validation_warnings = validation_result.warning_count

    # Store updated model
    _threat_models[model.id] = model

    # Record metrics
    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    record_analysis_duration("stride_analysis", duration_ms / 1000)
    update_threat_counts(model.service_name, category_counts)
    update_severity_counts(model.service_name, severity_counts)
    update_risk_score(model.service_name, model.overall_risk_score)

    logger.info(
        "STRIDE analysis complete: threats=%d, risk=%.2f, duration_ms=%.2f",
        len(all_threats),
        model.overall_risk_score,
        duration_ms,
    )

    return AnalysisResultResponse(
        threat_model_id=model.id,
        threats_identified=len(all_threats),
        threats_by_category=category_counts,
        threats_by_severity=severity_counts,
        overall_risk_score=model.overall_risk_score,
        validation_passed=validation_passed,
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
        analysis_duration_ms=duration_ms,
    )


@router.post("/{threat_model_id}/components", response_model=ComponentResponse, status_code=status.HTTP_201_CREATED)
async def add_component(
    threat_model_id: str,
    request: AddComponentRequest,
) -> ComponentResponse:
    """Add a component to the threat model.

    Requires: secops:threats:write permission.
    """
    logger.info("Adding component to threat model: %s", threat_model_id)

    model = _get_threat_model(threat_model_id)

    # Create component
    component = Component(
        id=str(uuid4()),
        name=request.name,
        component_type=request.component_type,
        description=request.description,
        trust_level=request.trust_level,
        data_classification=request.data_classification,
        owner=request.owner,
        technology_stack=request.technology_stack,
        is_external=request.is_external,
        network_zone=request.network_zone,
    )

    # Add to model
    model.components.append(component)
    model.updated_at = datetime.now(timezone.utc)
    _threat_models[model.id] = model

    logger.info("Added component: id=%s, name=%s", component.id, component.name)

    return ComponentResponse(
        id=component.id,
        name=component.name,
        component_type=component.component_type,
        description=component.description,
        trust_level=component.trust_level,
        data_classification=component.data_classification,
        owner=component.owner,
        is_external=component.is_external,
        network_zone=component.network_zone,
    )


@router.put("/{threat_model_id}/components/{component_id}", response_model=ComponentResponse)
async def update_component(
    threat_model_id: str,
    component_id: str,
    request: UpdateComponentRequest,
) -> ComponentResponse:
    """Update a component in the threat model.

    Requires: secops:threats:write permission.
    """
    logger.info("Updating component %s in threat model %s", component_id, threat_model_id)

    model = _get_threat_model(threat_model_id)

    # Find component
    component = next((c for c in model.components if c.id == component_id), None)
    if not component:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Component {component_id} not found",
        )

    # Update fields
    if request.name is not None:
        component.name = request.name
    if request.component_type is not None:
        component.component_type = request.component_type
    if request.description is not None:
        component.description = request.description
    if request.trust_level is not None:
        component.trust_level = request.trust_level
    if request.data_classification is not None:
        component.data_classification = request.data_classification

    model.updated_at = datetime.now(timezone.utc)
    _threat_models[model.id] = model

    return ComponentResponse(
        id=component.id,
        name=component.name,
        component_type=component.component_type,
        description=component.description,
        trust_level=component.trust_level,
        data_classification=component.data_classification,
        owner=component.owner,
        is_external=component.is_external,
        network_zone=component.network_zone,
    )


@router.post("/{threat_model_id}/data-flows", response_model=DataFlowResponse, status_code=status.HTTP_201_CREATED)
async def add_data_flow(
    threat_model_id: str,
    request: AddDataFlowRequest,
) -> DataFlowResponse:
    """Add a data flow to the threat model.

    Requires: secops:threats:write permission.
    """
    logger.info("Adding data flow to threat model: %s", threat_model_id)

    model = _get_threat_model(threat_model_id)

    # Validate component IDs
    component_ids = {c.id for c in model.components}
    if request.source_component_id not in component_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Source component {request.source_component_id} not found",
        )
    if request.destination_component_id not in component_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Destination component {request.destination_component_id} not found",
        )

    # Create data flow
    flow = DataFlow(
        id=str(uuid4()),
        source_component_id=request.source_component_id,
        destination_component_id=request.destination_component_id,
        data_type=request.data_type,
        protocol=request.protocol,
        encryption=request.encryption,
        authentication_required=request.authentication_required,
        authentication_method=request.authentication_method,
        authorization_required=request.authorization_required,
        description=request.description,
        data_classification=request.data_classification,
        bidirectional=request.bidirectional,
    )

    # Add to model
    model.data_flows.append(flow)
    model.updated_at = datetime.now(timezone.utc)
    _threat_models[model.id] = model

    logger.info("Added data flow: id=%s", flow.id)

    return DataFlowResponse(
        id=flow.id,
        source_component_id=flow.source_component_id,
        destination_component_id=flow.destination_component_id,
        data_type=flow.data_type,
        protocol=flow.protocol,
        encryption=flow.encryption,
        authentication_required=flow.authentication_required,
        data_classification=flow.data_classification,
        crosses_trust_boundary=flow.crosses_trust_boundary,
    )


@router.post("/{threat_model_id}/mitigations", response_model=MitigationResponse, status_code=status.HTTP_201_CREATED)
async def add_mitigation(
    threat_model_id: str,
    request: AddMitigationRequest,
) -> MitigationResponse:
    """Add a mitigation to the threat model.

    Requires: secops:threats:write permission.
    """
    logger.info("Adding mitigation to threat model: %s", threat_model_id)

    model = _get_threat_model(threat_model_id)

    # Validate threat ID
    threat_ids = {t.id for t in model.threats}
    if request.threat_id not in threat_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Threat {request.threat_id} not found",
        )

    # Create mitigation
    mitigation = Mitigation(
        id=str(uuid4()),
        threat_id=request.threat_id,
        control_id=request.control_id,
        title=request.title,
        description=request.description,
        effectiveness=request.effectiveness,
        owner=request.owner,
        priority=request.priority,
        due_date=request.due_date,
        status=MitigationStatus.PROPOSED,
    )

    # Add to model
    model.mitigations.append(mitigation)
    model.updated_at = datetime.now(timezone.utc)
    _threat_models[model.id] = model

    logger.info("Added mitigation: id=%s, threat=%s", mitigation.id, mitigation.threat_id)

    return MitigationResponse(
        id=mitigation.id,
        threat_id=mitigation.threat_id,
        control_id=mitigation.control_id,
        title=mitigation.title,
        description=mitigation.description,
        status=mitigation.status,
        effectiveness=mitigation.effectiveness,
        owner=mitigation.owner,
        priority=mitigation.priority,
    )


@router.put("/{threat_model_id}/approve", response_model=ThreatModelResponse)
async def approve_threat_model(
    threat_model_id: str,
    request: ApproveRequest,
) -> ThreatModelResponse:
    """Approve a threat model.

    Requires: secops:threats:write permission.
    """
    logger.info("Approving threat model: %s", threat_model_id)

    model = _get_threat_model(threat_model_id)

    # Validate status
    if model.status not in (ThreatModelStatus.DRAFT, ThreatModelStatus.IN_REVIEW):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot approve threat model in {model.status.value} status",
        )

    # Check for critical unmitigated threats
    critical_unmitigated = [
        t for t in model.threats
        if t.severity == "critical" and t.status == ThreatStatus.IDENTIFIED
    ]
    if critical_unmitigated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot approve: {len(critical_unmitigated)} critical threats are unmitigated",
        )

    # Approve
    model.status = ThreatModelStatus.APPROVED
    model.approved_by = request.approved_by
    model.approved_at = datetime.now(timezone.utc)
    model.updated_at = datetime.now(timezone.utc)

    _threat_models[model.id] = model

    logger.info("Threat model approved: %s by %s", model.id, request.approved_by)

    return ThreatModelResponse(
        id=model.id,
        service_name=model.service_name,
        version=model.version,
        status=model.status,
        overall_risk_score=model.overall_risk_score,
        threat_count=model.threat_count,
        mitigated_count=model.mitigated_count,
        critical_count=model.critical_count,
        high_count=model.high_count,
        owner=model.owner,
        created_at=model.created_at,
        updated_at=model.updated_at,
        approved_by=model.approved_by,
        approved_at=model.approved_at,
    )


@router.get("/{threat_model_id}/report", response_model=ReportResponse)
async def generate_report(
    threat_model_id: str,
    format: str = Query("json", description="Report format: json, pdf, html"),
) -> ReportResponse:
    """Generate a threat model report.

    Requires: secops:threats:read permission.
    """
    logger.info("Generating report for threat model: %s, format=%s", threat_model_id, format)

    model = _get_threat_model(threat_model_id)

    # Validate format
    allowed_formats = {"json", "pdf", "html", "markdown"}
    if format.lower() not in allowed_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid format. Allowed: {sorted(allowed_formats)}",
        )

    # Generate report content
    if format.lower() == "json":
        # Return model as JSON
        content = model.model_dump_json(indent=2)
    else:
        # For PDF/HTML, would generate actual document
        # For now, return placeholder
        content = None

    return ReportResponse(
        threat_model_id=model.id,
        service_name=model.service_name,
        format=format.lower(),
        content=content,
        download_url=f"/api/v1/secops/threats/{model.id}/report/download?format={format}" if content is None else None,
        generated_at=datetime.now(timezone.utc),
    )


__all__ = [
    "router",
    # Request models
    "CreateThreatModelRequest",
    "UpdateThreatModelRequest",
    "AddComponentRequest",
    "UpdateComponentRequest",
    "AddDataFlowRequest",
    "AddMitigationRequest",
    "RunAnalysisRequest",
    "ApproveRequest",
    # Response models
    "ThreatModelResponse",
    "ThreatModelListResponse",
    "ThreatModelDetailResponse",
    "ComponentResponse",
    "ThreatResponse",
    "AnalysisResultResponse",
    "ReportResponse",
]
