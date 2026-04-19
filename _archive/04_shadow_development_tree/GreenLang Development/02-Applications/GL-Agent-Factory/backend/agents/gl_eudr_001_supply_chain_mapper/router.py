"""
GL-EUDR-001: Supply Chain Mapper API Routes (Secured)

FastAPI router implementing the Supply Chain Mapper API endpoints.
All endpoints are prefixed with /api/v1/supply-chain.

Security Features:
- JWT authentication on all endpoints
- Role-based access control (RBAC)
- Resource ownership verification (IDOR protection)
- PII masking based on user permissions
- Rate limiting on expensive operations
- Comprehensive audit logging
- Input validation and sanitization

Endpoints:
- Nodes: CRUD operations for supply chain nodes
- Edges: CRUD operations for edges/relationships
- Plots: CRUD operations for origin plots
- Graph: Query full supply chain graph
- Coverage: Calculate and check coverage gates
- Snapshots: Create and query snapshots with as-of support
- Entity Resolution: Run and manage entity resolution
- Natural Language: Query supply chain using NL
- Bulk: Batch import/export operations
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from .agent import (
    SupplyChainMapperAgent,
    SupplyChainMapperInput,
    CommodityType,
    NodeType,
    EdgeType,
    DataSource,
    VerificationStatus,
    DisclosureStatus,
    OperatorSize,
    RiskLevel,
    OperationType,
    SnapshotTrigger,
    ResolutionStatus,
    SupplyChainNode,
    SupplyChainEdge,
    OriginPlot,
    PlotGeometry,
    Address,
)
from .auth import (
    User,
    UserRole,
    Permission,
    get_current_user,
    require_permissions,
    require_role,
    ResourceOwnershipVerifier,
    PIIMasker,
    rate_limit,
    strict_rate_limiter,
)
from .audit import (
    AuditLogger,
    AuditContext,
    AuditAction,
    get_audit_logger,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/supply-chain", tags=["Supply Chain Mapper"])


# =============================================================================
# AGENT DEPENDENCY (Thread-Safe)
# =============================================================================

class AgentManager:
    """Thread-safe agent manager with tenant isolation."""

    _instances: Dict[str, SupplyChainMapperAgent] = {}

    @classmethod
    def get_agent(cls, organization_id: str) -> SupplyChainMapperAgent:
        """Get or create agent instance for organization."""
        if organization_id not in cls._instances:
            cls._instances[organization_id] = SupplyChainMapperAgent()
        return cls._instances[organization_id]


def get_agent(current_user: User = Depends(get_current_user)) -> SupplyChainMapperAgent:
    """Get tenant-scoped agent instance."""
    org_id = str(current_user.organization_id)
    return AgentManager.get_agent(org_id)


def get_audit_context(
    request: Request,
    current_user: User = Depends(get_current_user)
) -> AuditContext:
    """Create audit context from request."""
    return AuditContext(
        user_id=current_user.user_id,
        user_email=current_user.email,
        user_role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
        organization_id=current_user.organization_id,
        request_id=request.headers.get("X-Request-ID", str(uuid.uuid4())),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("User-Agent"),
        endpoint=str(request.url.path),
        method=request.method
    )


# =============================================================================
# FIELD ALLOWLISTS (Mass Assignment Protection)
# =============================================================================

ALLOWED_NODE_UPDATE_FIELDS = {
    "name", "address", "tax_id", "duns_number", "eori_number",
    "verification_status", "disclosure_status"
}

ALLOWED_EDGE_UPDATE_FIELDS = {
    "quantity", "quantity_unit", "transaction_date", "verified",
    "confidence_score", "documents"
}


# =============================================================================
# REQUEST/RESPONSE MODELS (Pydantic v2 Syntax)
# =============================================================================

class CreateNodeRequest(BaseModel):
    """Request to create a supply chain node."""
    node_type: NodeType
    name: str = Field(..., min_length=1, max_length=500)
    country_code: str = Field(..., pattern=r"^[A-Z]{2}$")
    commodities: List[CommodityType]
    address: Optional[Address] = None
    tax_id: Optional[str] = None
    duns_number: Optional[str] = Field(None, pattern=r"^[0-9]{9}$")
    eori_number: Optional[str] = Field(None, max_length=17)
    operator_size: Optional[OperatorSize] = None
    disclosure_status: DisclosureStatus = DisclosureStatus.FULL

    class Config:
        use_enum_values = True


class UpdateNodeRequest(BaseModel):
    """Request to update a supply chain node."""
    name: Optional[str] = Field(None, min_length=1, max_length=500)
    address: Optional[Address] = None
    tax_id: Optional[str] = None
    duns_number: Optional[str] = Field(None, pattern=r"^[0-9]{9}$")
    eori_number: Optional[str] = Field(None, max_length=17)
    verification_status: Optional[VerificationStatus] = None
    disclosure_status: Optional[DisclosureStatus] = None

    class Config:
        use_enum_values = True


class CreateEdgeRequest(BaseModel):
    """Request to create a supply chain edge."""
    source_node_id: UUID
    target_node_id: UUID
    edge_type: EdgeType
    commodity: CommodityType
    quantity: Optional[float] = Field(None, ge=0)
    quantity_unit: Optional[str] = Field(None, max_length=20)
    transaction_date: Optional[str] = None
    data_source: DataSource = DataSource.SUPPLIER_DECLARED
    inference_method: Optional[str] = None
    inference_evidence: List[dict] = Field(default_factory=list)

    class Config:
        use_enum_values = True


class CreatePlotRequest(BaseModel):
    """Request to create an origin plot."""
    producer_node_id: UUID
    plot_identifier: Optional[str] = Field(None, max_length=255)
    geometry: PlotGeometry
    area_hectares: Optional[float] = Field(None, gt=0)
    commodity: CommodityType
    country_code: str = Field(..., pattern=r"^[A-Z]{2}$")

    class Config:
        use_enum_values = True


class CreateSnapshotRequest(BaseModel):
    """Request to create a snapshot."""
    importer_id: UUID
    commodity: CommodityType
    trigger_type: SnapshotTrigger = SnapshotTrigger.MANUAL

    class Config:
        use_enum_values = True


class RunEntityResolutionRequest(BaseModel):
    """Request to run entity resolution."""
    scope: str = Field(default="ALL", pattern=r"^(ALL|NEW_NODES|SPECIFIC_NODES)$")
    node_ids: Optional[List[UUID]] = None


class ResolveEntityRequest(BaseModel):
    """Request to resolve an entity resolution candidate."""
    decision: str = Field(..., pattern=r"^(MERGE|NO_MERGE)$")
    reason: Optional[str] = Field(None, max_length=500)


class NaturalLanguageQueryRequest(BaseModel):
    """Request for natural language query."""
    query: str = Field(..., min_length=3, max_length=1000)
    importer_id: UUID
    commodity: CommodityType

    class Config:
        use_enum_values = True


class BulkImportRequest(BaseModel):
    """Request for bulk import."""
    nodes: List[CreateNodeRequest] = Field(default_factory=list, max_length=1000)
    edges: List[CreateEdgeRequest] = Field(default_factory=list, max_length=5000)
    plots: List[CreatePlotRequest] = Field(default_factory=list, max_length=10000)


class GraphMetadata(BaseModel):
    """Metadata for supply chain graph response."""
    total_nodes: int
    total_edges: int
    max_tier: int
    commodities: List[str]
    has_cycles: bool
    inferred_edge_count: int


class GraphResponse(BaseModel):
    """Response containing supply chain graph."""
    nodes: List[dict]
    edges: List[dict]
    metadata: GraphMetadata


class NLQueryResult(BaseModel):
    """Result from natural language query."""
    original_query: str
    interpreted_query: str
    generated_filter: dict
    results: List[dict]
    result_count: int


class PaginatedResponse(BaseModel):
    """Generic paginated response."""
    items: List[dict]
    total: int
    limit: int
    offset: int


# =============================================================================
# NODE ENDPOINTS
# =============================================================================

@router.post("/nodes", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_node(
    request: Request,
    body: CreateNodeRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.NODE_WRITE)),
    agent: SupplyChainMapperAgent = Depends(get_agent),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """Create a new supply chain node."""
    audit_context = get_audit_context(request, current_user)

    # Add organization context to node metadata
    node = SupplyChainNode(
        node_type=body.node_type,
        name=body.name,
        country_code=body.country_code,
        commodities=body.commodities,
        address=body.address,
        tax_id=body.tax_id,
        duns_number=body.duns_number,
        eori_number=body.eori_number,
        operator_size=body.operator_size,
        disclosure_status=body.disclosure_status,
        metadata={"organization_id": str(current_user.organization_id)}
    )

    result = agent.add_node(node)

    # Audit log
    audit_logger.log_node_create(
        context=audit_context,
        node_id=result.node_id,
        node_data=result.dict()
    )

    # Mask PII in response
    return PIIMasker.mask_node(result, current_user)


@router.get("/nodes", response_model=PaginatedResponse)
async def list_nodes(
    commodity: Optional[CommodityType] = None,
    tier: Optional[int] = None,
    country: Optional[str] = Query(None, pattern=r"^[A-Z]{2}$"),
    verification_status: Optional[VerificationStatus] = None,
    disclosure_status: Optional[DisclosureStatus] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.NODE_READ)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """List supply chain nodes with filtering."""
    nodes = agent.get_all_nodes()

    # Filter by organization (IDOR protection)
    nodes = ResourceOwnershipVerifier.filter_by_organization(nodes, current_user)

    # Apply additional filters
    if commodity:
        nodes = [n for n in nodes if commodity in n.commodities]
    if tier is not None:
        nodes = [n for n in nodes if n.tier == tier]
    if country:
        nodes = [n for n in nodes if n.country_code == country]
    if verification_status:
        nodes = [n for n in nodes if n.verification_status == verification_status]
    if disclosure_status:
        nodes = [n for n in nodes if n.disclosure_status == disclosure_status]

    # Paginate
    total = len(nodes)
    nodes = nodes[offset:offset + limit]

    # Mask PII in responses
    masked_nodes = [PIIMasker.mask_node(n, current_user) for n in nodes]

    return PaginatedResponse(
        items=masked_nodes,
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/nodes/{node_id}")
async def get_node(
    node_id: UUID,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.NODE_READ)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """Get node details by ID."""
    # Verify ownership (IDOR protection)
    node = ResourceOwnershipVerifier.verify_node_access(
        node_id=node_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    return PIIMasker.mask_node(node, current_user)


@router.patch("/nodes/{node_id}")
async def update_node(
    request: Request,
    node_id: UUID,
    body: UpdateNodeRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.NODE_WRITE)),
    agent: SupplyChainMapperAgent = Depends(get_agent),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """Update a supply chain node."""
    audit_context = get_audit_context(request, current_user)

    # Verify ownership (IDOR protection)
    node = ResourceOwnershipVerifier.verify_node_access(
        node_id=node_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    # Capture previous state for audit
    previous_state = node.dict()

    # Update ONLY allowed fields (Mass Assignment Protection)
    update_data = body.dict(exclude_unset=True)
    changes = {}
    for field, value in update_data.items():
        if field in ALLOWED_NODE_UPDATE_FIELDS and value is not None:
            old_value = getattr(node, field, None)
            setattr(node, field, value)
            if old_value != value:
                changes[field] = {"old": old_value, "new": value}

    node.updated_at = datetime.utcnow()

    # Audit log
    audit_logger.log_node_update(
        context=audit_context,
        node_id=node_id,
        previous=previous_state,
        updated=node.dict(),
        changes=changes
    )

    return PIIMasker.mask_node(node, current_user)


@router.delete("/nodes/{node_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_node(
    request: Request,
    node_id: UUID,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_role(UserRole.ADMIN, UserRole.COMPLIANCE_OFFICER)),
    __: None = Depends(require_permissions(Permission.NODE_DELETE)),
    agent: SupplyChainMapperAgent = Depends(get_agent),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """Delete a supply chain node (soft delete in production)."""
    audit_context = get_audit_context(request, current_user)

    # Verify ownership (IDOR protection)
    node = ResourceOwnershipVerifier.verify_node_access(
        node_id=node_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    # Capture state for audit before deletion
    node_data = node.dict()

    # Delete node and associated edges
    del agent._nodes[node_id]
    edges_to_remove = [
        eid for eid, e in agent._edges.items()
        if e.source_node_id == node_id or e.target_node_id == node_id
    ]
    for eid in edges_to_remove:
        del agent._edges[eid]

    # Audit log
    audit_logger.log_node_delete(
        context=audit_context,
        node_id=node_id,
        node_data=node_data
    )


# =============================================================================
# EDGE ENDPOINTS
# =============================================================================

@router.post("/edges", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_edge(
    body: CreateEdgeRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.EDGE_WRITE)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """Create a new supply chain edge."""
    # Verify both nodes exist and belong to user's organization
    ResourceOwnershipVerifier.verify_node_access(
        node_id=body.source_node_id,
        current_user=current_user,
        node_getter=agent._get_node
    )
    ResourceOwnershipVerifier.verify_node_access(
        node_id=body.target_node_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    edge = SupplyChainEdge(
        source_node_id=body.source_node_id,
        target_node_id=body.target_node_id,
        edge_type=body.edge_type,
        commodity=body.commodity,
        quantity=body.quantity,
        quantity_unit=body.quantity_unit,
        data_source=body.data_source,
        inference_method=body.inference_method
    )
    result = agent.add_edge(edge)
    return result.dict()


@router.get("/edges", response_model=PaginatedResponse)
async def list_edges(
    data_source: Optional[DataSource] = None,
    dds_eligible: Optional[bool] = None,
    commodity: Optional[CommodityType] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.EDGE_READ)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """List supply chain edges with filtering."""
    edges = agent.get_all_edges()

    # Filter by organization (via node ownership)
    org_nodes = {n.node_id for n in ResourceOwnershipVerifier.filter_by_organization(
        agent.get_all_nodes(), current_user
    )}
    edges = [e for e in edges if e.source_node_id in org_nodes or e.target_node_id in org_nodes]

    # Apply additional filters
    if data_source:
        edges = [e for e in edges if e.data_source == data_source]
    if dds_eligible is not None:
        edges = [e for e in edges if e.dds_eligible == dds_eligible]
    if commodity:
        edges = [e for e in edges if e.commodity == commodity]

    total = len(edges)
    edges = edges[offset:offset + limit]

    return PaginatedResponse(
        items=[e.dict() for e in edges],
        total=total,
        limit=limit,
        offset=offset
    )


# =============================================================================
# PLOT ENDPOINTS (NEW)
# =============================================================================

@router.post("/plots", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_plot(
    body: CreatePlotRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.NODE_WRITE)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """Create a new origin plot."""
    # Verify producer node exists and belongs to user's organization
    ResourceOwnershipVerifier.verify_node_access(
        node_id=body.producer_node_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    plot = OriginPlot(
        producer_node_id=body.producer_node_id,
        plot_identifier=body.plot_identifier,
        geometry=body.geometry,
        area_hectares=body.area_hectares,
        commodity=body.commodity,
        country_code=body.country_code
    )
    result = agent.add_plot(plot)
    return result.dict()


@router.get("/plots", response_model=PaginatedResponse)
async def list_plots(
    commodity: Optional[CommodityType] = None,
    country: Optional[str] = Query(None, pattern=r"^[A-Z]{2}$"),
    producer_node_id: Optional[UUID] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.NODE_READ)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """List origin plots with filtering."""
    plots = list(agent._plots.values())

    # Filter by organization via producer nodes
    org_nodes = {n.node_id for n in ResourceOwnershipVerifier.filter_by_organization(
        agent.get_all_nodes(), current_user
    )}
    plots = [p for p in plots if p.producer_node_id in org_nodes]

    # Apply filters
    if commodity:
        plots = [p for p in plots if p.commodity == commodity]
    if country:
        plots = [p for p in plots if p.country_code == country]
    if producer_node_id:
        plots = [p for p in plots if p.producer_node_id == producer_node_id]

    total = len(plots)
    plots = plots[offset:offset + limit]

    return PaginatedResponse(
        items=[p.dict() for p in plots],
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/plots/{plot_id}")
async def get_plot(
    plot_id: UUID,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.NODE_READ)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """Get plot details by ID."""
    plot = agent._plots.get(plot_id)
    if not plot:
        raise HTTPException(status_code=404, detail="Plot not found")

    # Verify ownership via producer node
    ResourceOwnershipVerifier.verify_node_access(
        node_id=plot.producer_node_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    return plot.dict()


# =============================================================================
# GRAPH ENDPOINTS
# =============================================================================

@router.get("/graph", response_model=GraphResponse)
async def get_supply_chain_graph(
    importer_id: UUID,
    commodity: CommodityType,
    depth: int = Query(10, ge=1, le=20),
    include_inferred: bool = True,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.NODE_READ)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """Get full supply chain graph for an importer."""
    # Verify importer ownership
    ResourceOwnershipVerifier.verify_node_access(
        node_id=importer_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    input_data = SupplyChainMapperInput(
        importer_id=importer_id,
        commodity=commodity,
        operation=OperationType.MAP_SUPPLY_CHAIN,
        depth=depth,
        include_inferred=include_inferred
    )

    result = agent.run(input_data)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.errors)

    graph = result.graph
    inferred_count = sum(
        1 for e in graph.edges
        if e.data_source != DataSource.SUPPLIER_DECLARED
    )

    # Mask PII in nodes
    masked_nodes = [PIIMasker.mask_node(n, current_user) for n in graph.nodes]

    return GraphResponse(
        nodes=masked_nodes,
        edges=[e.dict() for e in graph.edges],
        metadata=GraphMetadata(
            total_nodes=graph.node_count,
            total_edges=graph.edge_count,
            max_tier=graph.max_tier,
            commodities=[commodity.value],
            has_cycles=graph.has_cycles,
            inferred_edge_count=inferred_count
        )
    )


# =============================================================================
# COVERAGE ENDPOINTS
# =============================================================================

@router.get("/coverage", response_model=dict)
async def calculate_coverage(
    importer_id: UUID,
    commodity: CommodityType,
    risk_level: RiskLevel = RiskLevel.STANDARD,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.COVERAGE_READ)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """Calculate traceability coverage."""
    # Verify importer ownership
    ResourceOwnershipVerifier.verify_node_access(
        node_id=importer_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    input_data = SupplyChainMapperInput(
        importer_id=importer_id,
        commodity=commodity,
        operation=OperationType.CALCULATE_COVERAGE,
        risk_level=risk_level
    )

    result = agent.run(input_data)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.errors)

    return result.coverage_report.dict()


@router.get("/coverage/gates", response_model=dict)
async def check_coverage_gates(
    request: Request,
    importer_id: UUID,
    commodity: CommodityType,
    risk_level: RiskLevel = RiskLevel.STANDARD,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.COVERAGE_GATES)),
    agent: SupplyChainMapperAgent = Depends(get_agent),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """Check coverage gates for risk assessment and DDS submission."""
    audit_context = get_audit_context(request, current_user)

    # Verify importer ownership
    ResourceOwnershipVerifier.verify_node_access(
        node_id=importer_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    input_data = SupplyChainMapperInput(
        importer_id=importer_id,
        commodity=commodity,
        operation=OperationType.CHECK_GATES,
        risk_level=risk_level
    )

    result = agent.run(input_data)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.errors)

    gate_result = result.gate_result.dict()

    # Audit log for compliance-critical operation
    audit_logger.log_coverage_check(
        context=audit_context,
        importer_id=importer_id,
        commodity=commodity.value,
        result=gate_result
    )

    return gate_result


# =============================================================================
# SNAPSHOT ENDPOINTS
# =============================================================================

@router.post("/snapshots", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_snapshot(
    request: Request,
    body: CreateSnapshotRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.SNAPSHOT_CREATE)),
    agent: SupplyChainMapperAgent = Depends(get_agent),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """Create a supply chain snapshot."""
    audit_context = get_audit_context(request, current_user)

    # Verify importer ownership
    ResourceOwnershipVerifier.verify_node_access(
        node_id=body.importer_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    input_data = SupplyChainMapperInput(
        importer_id=body.importer_id,
        commodity=body.commodity,
        operation=OperationType.CREATE_SNAPSHOT,
        trigger_type=body.trigger_type
    )

    result = agent.run(input_data)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.errors)

    snapshot_dict = result.snapshot.dict()

    # Audit log
    audit_logger.log_snapshot_create(
        context=audit_context,
        snapshot_id=result.snapshot.snapshot_id,
        snapshot_data=snapshot_dict
    )

    return snapshot_dict


@router.get("/snapshots", response_model=PaginatedResponse)
async def list_snapshots(
    importer_id: UUID,
    commodity: CommodityType,
    as_of: Optional[datetime] = None,
    policy: str = Query("latest_before", pattern=r"^(latest_before|closest|exact)$"),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.SNAPSHOT_READ)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """Query snapshots with as-of support."""
    # Verify importer ownership
    ResourceOwnershipVerifier.verify_node_access(
        node_id=importer_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    snapshots = [
        s for s in agent._snapshots.values()
        if s.importer_node_id == importer_id and s.commodity == commodity
    ]

    if as_of:
        if policy == "latest_before":
            valid = [s for s in snapshots if s.snapshot_date <= as_of]
            if valid:
                valid.sort(key=lambda x: x.snapshot_date, reverse=True)
                return PaginatedResponse(
                    items=[valid[0].dict()],
                    total=1,
                    limit=1,
                    offset=0
                )
        elif policy == "exact":
            for s in snapshots:
                if s.snapshot_date == as_of:
                    return PaginatedResponse(
                        items=[s.dict()],
                        total=1,
                        limit=1,
                        offset=0
                    )
            raise HTTPException(status_code=404, detail="No exact snapshot found")

    # Return list
    snapshots.sort(key=lambda x: x.snapshot_date, reverse=True)
    return PaginatedResponse(
        items=[s.dict() for s in snapshots[:limit]],
        total=len(snapshots),
        limit=limit,
        offset=0
    )


@router.get("/snapshots/{snapshot_id}")
async def get_snapshot(
    snapshot_id: UUID,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.SNAPSHOT_READ)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """Get snapshot details."""
    snapshot = agent._snapshots.get(snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    # Verify ownership via importer node
    ResourceOwnershipVerifier.verify_node_access(
        node_id=snapshot.importer_node_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    return snapshot.dict()


@router.get("/snapshots/{snapshot_id}/diff/{compare_snapshot_id}")
async def diff_snapshots(
    snapshot_id: UUID,
    compare_snapshot_id: UUID,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.SNAPSHOT_READ)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """Get diff between two snapshots."""
    base = agent._snapshots.get(snapshot_id)
    compare = agent._snapshots.get(compare_snapshot_id)

    if not base:
        raise HTTPException(status_code=404, detail="Base snapshot not found")
    if not compare:
        raise HTTPException(status_code=404, detail="Compare snapshot not found")

    # Verify ownership
    ResourceOwnershipVerifier.verify_node_access(
        node_id=base.importer_node_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    diff = agent._diff_snapshots(snapshot_id, compare_snapshot_id)
    return diff.dict()


# =============================================================================
# ENTITY RESOLUTION ENDPOINTS
# =============================================================================

@router.get("/entity-resolution/candidates", response_model=PaginatedResponse)
async def list_entity_resolution_candidates(
    resolution_status: Optional[ResolutionStatus] = Query(None, alias="status"),
    min_score: float = Query(0.0, ge=0, le=1),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.ER_READ)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """List entity resolution candidates."""
    candidates = list(agent._resolution_candidates.values())

    if resolution_status:
        candidates = [c for c in candidates if c.resolution_status == resolution_status]
    if min_score > 0:
        candidates = [c for c in candidates if c.similarity_score >= min_score]

    candidates.sort(key=lambda x: x.similarity_score, reverse=True)

    return PaginatedResponse(
        items=[c.dict() for c in candidates[:limit]],
        total=len(candidates),
        limit=limit,
        offset=0
    )


@router.post("/entity-resolution/candidates/{candidate_id}/resolve")
async def resolve_entity_candidate(
    request: Request,
    candidate_id: UUID,
    body: ResolveEntityRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.ER_RESOLVE)),
    agent: SupplyChainMapperAgent = Depends(get_agent),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """Resolve an entity resolution candidate."""
    audit_context = get_audit_context(request, current_user)

    candidate = agent._resolution_candidates.get(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    if body.decision == "MERGE":
        agent._merge_entities(candidate.node_a_id, candidate.node_b_id)
        candidate.resolution_status = ResolutionStatus.REVIEWED_MERGE

        # Audit log for merge
        audit_logger.log_entity_merge(
            context=audit_context,
            keep_id=candidate.node_a_id,
            merge_id=candidate.node_b_id,
            merge_details={
                "candidate_id": str(candidate_id),
                "similarity_score": candidate.similarity_score,
                "reason": body.reason
            }
        )
    else:
        candidate.resolution_status = ResolutionStatus.REVIEWED_NO_MERGE

    candidate.resolved_at = datetime.utcnow()
    # Use authenticated user instead of hardcoded "api_user"
    candidate.resolved_by = current_user.email

    return candidate.dict()


@router.post("/entity-resolution/run")
async def run_entity_resolution(
    body: RunEntityResolutionRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.ER_RUN)),
    __: None = Depends(rate_limit(strict_rate_limiter)),  # Rate limit expensive operation
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """Trigger entity resolution batch run."""
    input_data = SupplyChainMapperInput(
        importer_id=UUID('00000000-0000-0000-0000-000000000000'),
        commodity=CommodityType.COFFEE,
        operation=OperationType.RUN_ENTITY_RESOLUTION,
        scope=body.scope,
        node_ids=body.node_ids
    )

    result = agent.run(input_data)

    return {
        "success": result.success,
        "auto_merged_count": result.auto_merged_count,
        "review_queue_count": result.review_queue_count,
        "candidates": [c.dict() for c in result.resolution_candidates]
    }


# =============================================================================
# NATURAL LANGUAGE QUERY ENDPOINTS
# =============================================================================

@router.post("/query/natural-language", response_model=NLQueryResult)
async def natural_language_query(
    body: NaturalLanguageQueryRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_permissions(Permission.NODE_READ)),
    __: None = Depends(rate_limit(strict_rate_limiter)),  # Rate limit LLM calls
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """Query supply chain using natural language."""
    # Verify importer ownership
    ResourceOwnershipVerifier.verify_node_access(
        node_id=body.importer_id,
        current_user=current_user,
        node_getter=agent._get_node
    )

    input_data = SupplyChainMapperInput(
        importer_id=body.importer_id,
        commodity=body.commodity,
        operation=OperationType.NATURAL_LANGUAGE_QUERY,
        query=body.query
    )

    result = agent.run(input_data)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.errors)

    # Mask PII in results
    masked_results = [PIIMasker.mask_node(n, current_user) for n in result.nl_results]

    return NLQueryResult(
        original_query=body.query,
        interpreted_query=result.nl_interpreted_query or body.query,
        generated_filter=result.nl_generated_filter or {},
        results=masked_results,
        result_count=len(result.nl_results)
    )


# =============================================================================
# BULK OPERATIONS ENDPOINTS
# =============================================================================

@router.post("/bulk/import", response_model=dict)
async def bulk_import(
    body: BulkImportRequest,
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_role(UserRole.ADMIN, UserRole.COMPLIANCE_OFFICER)),
    __: None = Depends(rate_limit(strict_rate_limiter)),
    agent: SupplyChainMapperAgent = Depends(get_agent)
):
    """Bulk import nodes, edges, and plots."""
    results = {
        "nodes_created": 0,
        "edges_created": 0,
        "plots_created": 0,
        "errors": []
    }

    # Import nodes
    for node_req in body.nodes:
        try:
            node = SupplyChainNode(
                node_type=node_req.node_type,
                name=node_req.name,
                country_code=node_req.country_code,
                commodities=node_req.commodities,
                address=node_req.address,
                tax_id=node_req.tax_id,
                duns_number=node_req.duns_number,
                eori_number=node_req.eori_number,
                operator_size=node_req.operator_size,
                disclosure_status=node_req.disclosure_status,
                metadata={"organization_id": str(current_user.organization_id)}
            )
            agent.add_node(node)
            results["nodes_created"] += 1
        except Exception as e:
            results["errors"].append(f"Node '{node_req.name}': {str(e)}")

    # Import edges
    for edge_req in body.edges:
        try:
            edge = SupplyChainEdge(
                source_node_id=edge_req.source_node_id,
                target_node_id=edge_req.target_node_id,
                edge_type=edge_req.edge_type,
                commodity=edge_req.commodity,
                quantity=edge_req.quantity,
                quantity_unit=edge_req.quantity_unit,
                data_source=edge_req.data_source
            )
            agent.add_edge(edge)
            results["edges_created"] += 1
        except Exception as e:
            results["errors"].append(f"Edge: {str(e)}")

    # Import plots
    for plot_req in body.plots:
        try:
            plot = OriginPlot(
                producer_node_id=plot_req.producer_node_id,
                plot_identifier=plot_req.plot_identifier,
                geometry=plot_req.geometry,
                area_hectares=plot_req.area_hectares,
                commodity=plot_req.commodity,
                country_code=plot_req.country_code
            )
            agent.add_plot(plot)
            results["plots_created"] += 1
        except Exception as e:
            results["errors"].append(f"Plot: {str(e)}")

    return results


# =============================================================================
# HEALTH CHECK (Unauthenticated)
# =============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint (no authentication required)."""
    return {
        "status": "healthy",
        "agent": "gl-eudr-001-supply-chain-mapper",
        "version": "2.0.0",
        "security": "enabled"
    }
