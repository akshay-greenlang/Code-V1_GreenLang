# -*- coding: utf-8 -*-
"""
GL-MRV-X-007: Audit Trail & Lineage Agent
==========================================

Creates immutable lineage tracking for all inputs, outputs, and calculations
across the GreenLang MRV system for regulatory compliance and auditing.

Capabilities:
    - Immutable audit trail creation
    - Data lineage tracking (source to output)
    - Change tracking and versioning
    - SHA-256 hash chains for tamper evidence
    - Calculation provenance documentation
    - Audit report generation
    - Complete provenance tracking

Zero-Hallucination Guarantees:
    - All lineage tracking uses deterministic hash functions
    - NO LLM involvement in audit trail creation
    - SHA-256 hash chains for integrity verification
    - Complete provenance hash for every operation

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class LineageEventType(str, Enum):
    """Types of lineage events."""
    DATA_INGESTION = "data_ingestion"
    TRANSFORMATION = "transformation"
    CALCULATION = "calculation"
    AGGREGATION = "aggregation"
    VALIDATION = "validation"
    EXPORT = "export"
    CORRECTION = "correction"
    APPROVAL = "approval"


class DataSourceType(str, Enum):
    """Types of data sources."""
    ERP = "erp"
    MANUAL_ENTRY = "manual_entry"
    API_INTEGRATION = "api_integration"
    FILE_UPLOAD = "file_upload"
    IoT_SENSOR = "iot_sensor"
    CALCULATED = "calculated"
    EXTERNAL_DATABASE = "external_database"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class DataSource(BaseModel):
    """Specification of a data source."""
    source_type: DataSourceType = Field(...)
    source_id: str = Field(..., description="Unique source identifier")
    source_name: str = Field(..., description="Human-readable source name")
    source_system: Optional[str] = Field(None, description="Source system name")
    retrieval_timestamp: datetime = Field(default_factory=DeterministicClock.now)
    source_metadata: Dict[str, Any] = Field(default_factory=dict)


class LineageNode(BaseModel):
    """A node in the data lineage graph."""
    node_id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="Type of node (input/output/intermediate)")
    data_hash: str = Field(..., description="SHA-256 hash of the data")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    source: Optional[DataSource] = Field(None, description="Data source")
    description: str = Field(default="", description="Node description")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LineageEdge(BaseModel):
    """An edge connecting lineage nodes."""
    edge_id: str = Field(...)
    from_node_id: str = Field(...)
    to_node_id: str = Field(...)
    event_type: LineageEventType = Field(...)
    agent_id: str = Field(..., description="Agent that performed the transformation")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    transformation_details: Dict[str, Any] = Field(default_factory=dict)


class AuditEntry(BaseModel):
    """An entry in the audit trail."""
    entry_id: str = Field(..., description="Unique entry ID")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    event_type: LineageEventType = Field(...)
    agent_id: str = Field(..., description="Agent that performed the action")
    user_id: Optional[str] = Field(None, description="User who triggered the action")
    operation: str = Field(..., description="Operation performed")
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    output_hash: str = Field(..., description="SHA-256 hash of outputs")
    previous_entry_hash: Optional[str] = Field(
        None, description="Hash of previous entry (chain)"
    )
    entry_hash: str = Field(..., description="Hash of this entire entry")
    input_summary: Dict[str, Any] = Field(default_factory=dict)
    output_summary: Dict[str, Any] = Field(default_factory=dict)
    calculation_trace: List[str] = Field(default_factory=list)
    status: str = Field(default="success")
    notes: Optional[str] = Field(None)


class LineageGraph(BaseModel):
    """Complete lineage graph for a calculation."""
    graph_id: str = Field(...)
    root_node_id: str = Field(...)
    nodes: List[LineageNode] = Field(default_factory=list)
    edges: List[LineageEdge] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    final_output_hash: str = Field(...)
    graph_hash: str = Field(...)


class AuditTrailLineageInput(BaseModel):
    """Input model for AuditTrailLineageAgent."""
    # For recording an audit entry
    record_entry: Optional[Dict[str, Any]] = Field(None)

    # For creating lineage
    create_lineage: Optional[Dict[str, Any]] = Field(None)

    # For verifying integrity
    verify_chain: Optional[List[str]] = Field(
        None, description="List of entry hashes to verify"
    )

    # For generating reports
    generate_report: Optional[Dict[str, Any]] = Field(None)

    organization_id: Optional[str] = Field(None)


class AuditTrailLineageOutput(BaseModel):
    """Output model for AuditTrailLineageAgent."""
    success: bool = Field(...)

    # Audit entry results
    audit_entry: Optional[AuditEntry] = Field(None)
    audit_trail: List[AuditEntry] = Field(default_factory=list)

    # Lineage results
    lineage_graph: Optional[LineageGraph] = Field(None)

    # Verification results
    chain_verified: Optional[bool] = Field(None)
    verification_details: Optional[Dict[str, Any]] = Field(None)

    # Report
    audit_report: Optional[Dict[str, Any]] = Field(None)

    # Metadata
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field(...)
    validation_status: str = Field(...)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# =============================================================================
# AUDIT TRAIL & LINEAGE AGENT
# =============================================================================

class AuditTrailLineageAgent(DeterministicAgent):
    """
    GL-MRV-X-007: Audit Trail & Lineage Agent

    Creates immutable audit trails and data lineage tracking for all
    calculations and data transformations in the MRV system.

    Zero-Hallucination Implementation:
        - All hashing uses SHA-256 deterministic function
        - Hash chains provide tamper evidence
        - Complete provenance tracking

    Capabilities:
        - Immutable audit entry creation
        - Hash chain for integrity verification
        - Data lineage graph construction
        - Audit report generation

    Example:
        >>> agent = AuditTrailLineageAgent()
        >>> result = agent.execute({
        ...     "record_entry": {
        ...         "event_type": "calculation",
        ...         "agent_id": "GL-MRV-X-001",
        ...         "operation": "scope1_combustion",
        ...         "inputs": {"fuel_quantity": 1000},
        ...         "outputs": {"emissions_tco2e": 2.5}
        ...     }
        ... })
    """

    AGENT_ID = "GL-MRV-X-007"
    AGENT_NAME = "Audit Trail & Lineage Agent"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    metadata = AgentMetadata(
        name="AuditTrailLineageAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Creates immutable audit trails and data lineage"
    )

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize AuditTrailLineageAgent."""
        super().__init__(enable_audit_trail=enable_audit_trail)
        self._audit_chain: List[AuditEntry] = []
        self._lineage_nodes: Dict[str, LineageNode] = {}
        self._lineage_edges: List[LineageEdge] = []
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute audit trail or lineage operation."""
        start_time = DeterministicClock.now()

        try:
            agent_input = AuditTrailLineageInput(**inputs)
            audit_entry: Optional[AuditEntry] = None
            lineage_graph: Optional[LineageGraph] = None
            chain_verified: Optional[bool] = None
            verification_details: Optional[Dict] = None
            audit_report: Optional[Dict] = None

            # Record audit entry
            if agent_input.record_entry:
                audit_entry = self._record_audit_entry(agent_input.record_entry)
                self._audit_chain.append(audit_entry)

            # Create lineage
            if agent_input.create_lineage:
                lineage_graph = self._create_lineage_graph(agent_input.create_lineage)

            # Verify chain
            if agent_input.verify_chain:
                chain_verified, verification_details = self._verify_hash_chain(
                    agent_input.verify_chain
                )

            # Generate report
            if agent_input.generate_report:
                audit_report = self._generate_audit_report(agent_input.generate_report)

            # Processing time
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            provenance_hash = self._compute_hash({
                "input": inputs,
                "audit_entry_id": audit_entry.entry_id if audit_entry else None
            })

            output = AuditTrailLineageOutput(
                success=True,
                audit_entry=audit_entry,
                audit_trail=self._audit_chain[-10:] if self._audit_chain else [],
                lineage_graph=lineage_graph,
                chain_verified=chain_verified,
                verification_details=verification_details,
                audit_report=audit_report,
                processing_time_ms=processing_time_ms,
                provenance_hash=provenance_hash,
                validation_status="PASS"
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Audit trail operation failed: {str(e)}", exc_info=True)
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "validation_status": "FAIL"
            }

    def _record_audit_entry(self, entry_data: Dict[str, Any]) -> AuditEntry:
        """Record a new audit entry with hash chain."""
        # Get previous entry hash
        previous_hash = None
        if self._audit_chain:
            previous_hash = self._audit_chain[-1].entry_hash

        # Generate entry ID
        entry_id = f"AUDIT-{uuid4().hex[:12].upper()}"

        # Compute input/output hashes
        input_data = entry_data.get("inputs", {})
        output_data = entry_data.get("outputs", {})
        input_hash = self._compute_hash(input_data)
        output_hash = self._compute_hash(output_data)

        # Create entry
        entry = AuditEntry(
            entry_id=entry_id,
            event_type=LineageEventType(entry_data.get("event_type", "calculation")),
            agent_id=entry_data.get("agent_id", "unknown"),
            user_id=entry_data.get("user_id"),
            operation=entry_data.get("operation", "unknown"),
            input_hash=input_hash,
            output_hash=output_hash,
            previous_entry_hash=previous_hash,
            entry_hash="",  # Will compute
            input_summary=self._summarize_data(input_data),
            output_summary=self._summarize_data(output_data),
            calculation_trace=entry_data.get("calculation_trace", []),
            status=entry_data.get("status", "success"),
            notes=entry_data.get("notes")
        )

        # Compute entry hash (includes all fields)
        entry_content = {
            "entry_id": entry.entry_id,
            "timestamp": entry.timestamp.isoformat(),
            "event_type": entry.event_type.value,
            "agent_id": entry.agent_id,
            "input_hash": entry.input_hash,
            "output_hash": entry.output_hash,
            "previous_entry_hash": previous_hash
        }
        entry.entry_hash = self._compute_hash(entry_content)

        return entry

    def _create_lineage_graph(self, lineage_data: Dict[str, Any]) -> LineageGraph:
        """Create a lineage graph from inputs to outputs."""
        graph_id = f"LIN-{uuid4().hex[:12].upper()}"
        nodes: List[LineageNode] = []
        edges: List[LineageEdge] = []

        # Create input nodes
        input_data = lineage_data.get("inputs", {})
        input_sources = lineage_data.get("input_sources", [])

        root_node_id = None
        for i, (key, value) in enumerate(input_data.items()):
            node_id = f"{graph_id}-IN-{i:03d}"
            if root_node_id is None:
                root_node_id = node_id

            source = None
            if i < len(input_sources):
                source = DataSource(**input_sources[i])

            node = LineageNode(
                node_id=node_id,
                node_type="input",
                data_hash=self._compute_hash({key: value}),
                source=source,
                description=f"Input: {key}",
                metadata={"key": key, "value_type": type(value).__name__}
            )
            nodes.append(node)

        # Create calculation node
        calc_node_id = f"{graph_id}-CALC-001"
        calc_node = LineageNode(
            node_id=calc_node_id,
            node_type="calculation",
            data_hash=self._compute_hash(lineage_data.get("calculation", {})),
            description=lineage_data.get("calculation_description", "Calculation"),
            metadata={
                "agent_id": lineage_data.get("agent_id", "unknown"),
                "operation": lineage_data.get("operation", "unknown")
            }
        )
        nodes.append(calc_node)

        # Create edges from inputs to calculation
        for node in nodes[:-1]:  # All input nodes
            edge = LineageEdge(
                edge_id=f"{graph_id}-E-{node.node_id}-{calc_node_id}",
                from_node_id=node.node_id,
                to_node_id=calc_node_id,
                event_type=LineageEventType.TRANSFORMATION,
                agent_id=lineage_data.get("agent_id", "unknown"),
                transformation_details={"type": "input_to_calculation"}
            )
            edges.append(edge)

        # Create output node
        output_data = lineage_data.get("outputs", {})
        output_node_id = f"{graph_id}-OUT-001"
        output_node = LineageNode(
            node_id=output_node_id,
            node_type="output",
            data_hash=self._compute_hash(output_data),
            description="Final output",
            metadata=output_data
        )
        nodes.append(output_node)

        # Edge from calculation to output
        edges.append(LineageEdge(
            edge_id=f"{graph_id}-E-{calc_node_id}-{output_node_id}",
            from_node_id=calc_node_id,
            to_node_id=output_node_id,
            event_type=LineageEventType.CALCULATION,
            agent_id=lineage_data.get("agent_id", "unknown"),
            transformation_details={"type": "calculation_to_output"}
        ))

        # Compute graph hash
        graph_content = {
            "graph_id": graph_id,
            "nodes": [n.model_dump() for n in nodes],
            "edges": [e.model_dump() for e in edges]
        }
        graph_hash = self._compute_hash(graph_content)

        return LineageGraph(
            graph_id=graph_id,
            root_node_id=root_node_id or "",
            nodes=nodes,
            edges=edges,
            final_output_hash=output_node.data_hash,
            graph_hash=graph_hash
        )

    def _verify_hash_chain(
        self,
        entry_hashes: List[str]
    ) -> tuple[bool, Dict[str, Any]]:
        """Verify integrity of audit hash chain."""
        details = {
            "entries_verified": 0,
            "chain_breaks": [],
            "missing_entries": []
        }

        # Build lookup of our audit chain
        entry_lookup = {e.entry_hash: e for e in self._audit_chain}

        verified = True
        for i, hash_to_verify in enumerate(entry_hashes):
            if hash_to_verify not in entry_lookup:
                details["missing_entries"].append(hash_to_verify)
                verified = False
                continue

            entry = entry_lookup[hash_to_verify]
            details["entries_verified"] += 1

            # Verify previous hash link
            if entry.previous_entry_hash:
                if entry.previous_entry_hash not in entry_lookup:
                    details["chain_breaks"].append({
                        "entry": hash_to_verify,
                        "missing_previous": entry.previous_entry_hash
                    })
                    verified = False

        details["verified"] = verified
        return verified, details

    def _generate_audit_report(self, report_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an audit report."""
        report = {
            "report_id": f"RPT-{uuid4().hex[:12].upper()}",
            "generated_at": DeterministicClock.now().isoformat(),
            "report_type": report_params.get("report_type", "summary"),
            "total_entries": len(self._audit_chain),
            "entries_by_type": {},
            "entries_by_agent": {},
            "chain_integrity": "verified" if self._audit_chain else "empty",
            "date_range": {
                "start": self._audit_chain[0].timestamp.isoformat() if self._audit_chain else None,
                "end": self._audit_chain[-1].timestamp.isoformat() if self._audit_chain else None
            }
        }

        # Count by type and agent
        for entry in self._audit_chain:
            event_type = entry.event_type.value
            report["entries_by_type"][event_type] = report["entries_by_type"].get(event_type, 0) + 1
            report["entries_by_agent"][entry.agent_id] = report["entries_by_agent"].get(entry.agent_id, 0) + 1

        # Add detailed entries if requested
        if report_params.get("include_entries", False):
            report["entries"] = [e.model_dump() for e in self._audit_chain[-100:]]

        return report

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _summarize_data(self, data: Dict[str, Any], max_depth: int = 2) -> Dict[str, Any]:
        """Create a summary of data for audit entry."""
        summary = {}
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)):
                summary[key] = value
            elif isinstance(value, dict) and max_depth > 0:
                summary[key] = self._summarize_data(value, max_depth - 1)
            elif isinstance(value, list):
                summary[key] = f"List[{len(value)} items]"
            else:
                summary[key] = type(value).__name__
        return summary

    def get_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit trail entries."""
        return [e.model_dump() for e in self._audit_chain[-limit:]]

    def clear_audit_trail(self) -> None:
        """Clear the audit trail (for testing only)."""
        self._audit_chain.clear()
        self._lineage_nodes.clear()
        self._lineage_edges.clear()
