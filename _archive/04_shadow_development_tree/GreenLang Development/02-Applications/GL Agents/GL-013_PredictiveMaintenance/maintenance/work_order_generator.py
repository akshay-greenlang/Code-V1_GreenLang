"""GL-013 PredictiveMaintenance - Work Order Generator Module"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib, logging, uuid
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ActionType(str, Enum):
    INSPECT = "inspect"
    LUBRICATE = "lubricate"
    ALIGN = "align"
    BALANCE = "balance"
    REPLACE_BEARING = "replace_bearing"
    REPLACE_SEAL = "replace_seal"
    REPLACE_FILTER = "replace_filter"
    REPLACE_BELT = "replace_belt"
    ADJUST = "adjust"
    CLEAN = "clean"
    CALIBRATE = "calibrate"
    OVERHAUL = "overhaul"

class ApprovalStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

class EvidenceReference(BaseModel):
    evidence_type: str
    source: str
    timestamp: datetime
    value: Optional[float] = None
    description: str
    hash: str

class ApprovalWorkflow(BaseModel):
    workflow_id: str
    work_order_id: str
    status: ApprovalStatus = ApprovalStatus.DRAFT
    submitted_by: Optional[str] = None
    submitted_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    comments: List[str] = Field(default_factory=list)

class DraftWorkOrder(BaseModel):
    work_order_id: str
    asset_id: str
    asset_name: str
    title: str
    description: str
    action_type: ActionType
    priority: str = Field(default="medium")
    estimated_duration_hours: float = Field(default=2.0)
    required_skills: List[str] = Field(default_factory=list)
    required_parts: List[str] = Field(default_factory=list)
    safety_precautions: List[str] = Field(default_factory=list)
    evidence_references: List[EvidenceReference] = Field(default_factory=list)
    recommended_completion_date: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    approval_workflow: Optional[ApprovalWorkflow] = None
    provenance_hash: str

class WorkOrderGeneratorConfig(BaseModel):
    auto_submit_for_approval: bool = Field(default=False)
    include_safety_precautions: bool = Field(default=True)
    default_buffer_days: int = Field(default=3)
    require_evidence: bool = Field(default=True)

class WorkOrderGenerator:
    """Draft work orders with evidence references and human approval workflow."""

    def __init__(self, config: Optional[WorkOrderGeneratorConfig] = None):
        self.config = config or WorkOrderGeneratorConfig()
        self._wo_count = 0
        self._action_mapping = {
            "replace_immediate": ActionType.REPLACE_BEARING,
            "replace_scheduled": ActionType.REPLACE_BEARING,
            "condition_monitor": ActionType.INSPECT,
            "inspect_urgent": ActionType.INSPECT,
            "inspect_planned": ActionType.INSPECT,
            "lubricate": ActionType.LUBRICATE,
            "align": ActionType.ALIGN,
            "balance": ActionType.BALANCE,
        }
        logger.info("WorkOrderGenerator initialized")

    def generate_work_order(self, asset_id: str, asset_name: str, recommended_action: str, risk_score: float, days_until_action: int, evidence: List[Dict]) -> DraftWorkOrder:
        """Generate a draft work order with evidence references."""
        self._wo_count += 1
        now = datetime.now(timezone.utc)
        wo_id = f"WO-{asset_id}-{self._wo_count:06d}"
        action_type = self._action_mapping.get(recommended_action, ActionType.INSPECT)
        evidence_refs = [self._create_evidence_reference(e) for e in evidence]
        title = f"{action_type.value.replace('_', ' ').title()} - {asset_name}"
        desc = self._generate_description(asset_name, action_type, risk_score, evidence_refs)
        priority = "high" if risk_score >= 60 else ("medium" if risk_score >= 40 else "low")
        recommended_date = now + timedelta(days=max(1, days_until_action - self.config.default_buffer_days))
        precautions = self._get_safety_precautions(action_type) if self.config.include_safety_precautions else []
        prov = hashlib.sha256(f"{wo_id}|{asset_id}|{now.isoformat()}".encode()).hexdigest()
        wo = DraftWorkOrder(work_order_id=wo_id, asset_id=asset_id, asset_name=asset_name, title=title, description=desc, action_type=action_type, priority=priority, evidence_references=evidence_refs, recommended_completion_date=recommended_date, safety_precautions=precautions, provenance_hash=prov)
        if self.config.auto_submit_for_approval:
            wo.approval_workflow = ApprovalWorkflow(workflow_id=f"APPR-{wo_id}", work_order_id=wo_id, status=ApprovalStatus.PENDING_APPROVAL, submitted_at=now)
        return wo
    def _create_evidence_reference(self, evidence: Dict) -> EvidenceReference:
        """Create an evidence reference from raw evidence data."""
        evidence_type = evidence.get("type", "sensor_reading")
        source = evidence.get("source", "unknown")
        timestamp = evidence.get("timestamp", datetime.now(timezone.utc))
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        value = evidence.get("value")
        description = evidence.get("description", f"{evidence_type} from {source}")
        evidence_str = f"{evidence_type}|{source}|{timestamp.isoformat()}|{value}"
        evidence_hash = hashlib.sha256(evidence_str.encode()).hexdigest()[:16]
        return EvidenceReference(
            evidence_type=evidence_type, source=source, timestamp=timestamp,
            value=value, description=description, hash=evidence_hash
        )

    def _generate_description(self, asset_name: str, action_type: ActionType, risk_score: float, evidence_refs: List[EvidenceReference]) -> str:
        """Generate work order description with evidence summary."""
        risk_level = "HIGH" if risk_score >= 60 else ("MEDIUM" if risk_score >= 40 else "LOW")
        action_desc = action_type.value.replace('_', ' ')
        desc_lines = [
            f"Predictive maintenance work order for {asset_name}.",
            f"Recommended action: {action_desc}",
            f"Risk level: {risk_level} (score: {risk_score:.1f})",
            "", "Supporting Evidence:"
        ]
        for i, ref in enumerate(evidence_refs, 1):
            value_str = f" = {ref.value}" if ref.value is not None else ""
            desc_lines.append(f"  {i}. [{ref.evidence_type}] {ref.description}{value_str}")
        desc_lines.extend(["", "This work order was generated by the predictive maintenance system.",
            "Human review and approval required before execution."])
        return chr(10).join(desc_lines)

    def _get_safety_precautions(self, action_type: ActionType) -> List[str]:
        """Get safety precautions for a given action type."""
        base = [
            "Lock out / Tag out (LOTO) all energy sources before work",
            "Wear appropriate PPE (safety glasses, gloves, steel-toe boots)",
            "Verify zero energy state before beginning maintenance"
        ]
        action_precautions = {
            ActionType.REPLACE_BEARING: ["Use proper lifting techniques", "Ensure bearing handling cleanliness", "Verify shaft alignment after replacement"],
            ActionType.REPLACE_SEAL: ["Depressurize system completely", "Inspect mating surfaces", "Use manufacturer-approved lubricant"],
            ActionType.ALIGN: ["Allow equipment to reach operating temperature", "Document all alignment readings", "Verify coupling condition"],
            ActionType.BALANCE: ["Secure all loose components", "Use manufacturer-specified tolerances", "Verify vibration readings post-balance"],
            ActionType.LUBRICATE: ["Use only approved lubricant grades", "Do not over-grease", "Clean grease fittings before applying"],
            ActionType.INSPECT: ["Do not bypass safety interlocks", "Use calibrated instruments", "Document findings with photographs"],
            ActionType.OVERHAUL: ["Follow manufacturer procedures exactly", "Document all component conditions", "Replace all wear items", "Perform full functional test"],
            ActionType.CALIBRATE: ["Use NIST-traceable standards", "Document pre-calibration readings", "Verify against multiple reference points"],
            ActionType.CLEAN: ["Use only approved cleaning agents", "Ensure adequate ventilation", "Properly dispose of contaminated materials"]
        }
        return base + action_precautions.get(action_type, [])

    def submit_for_approval(self, work_order: DraftWorkOrder, submitted_by: str) -> DraftWorkOrder:
        """Submit a draft work order for human approval."""
        now = datetime.now(timezone.utc)
        if work_order.approval_workflow is None:
            work_order.approval_workflow = ApprovalWorkflow(
                workflow_id=f"APPR-{work_order.work_order_id}", work_order_id=work_order.work_order_id
            )
        work_order.approval_workflow.status = ApprovalStatus.PENDING_APPROVAL
        work_order.approval_workflow.submitted_by = submitted_by
        work_order.approval_workflow.submitted_at = now
        logger.info(f"Work order {work_order.work_order_id} submitted for approval by {submitted_by}")
        return work_order

    def approve_work_order(self, work_order: DraftWorkOrder, approved_by: str, comments: Optional[str] = None) -> DraftWorkOrder:
        """Approve a work order for execution."""
        if work_order.approval_workflow is None:
            raise ValueError("Work order has no approval workflow")
        if work_order.approval_workflow.status != ApprovalStatus.PENDING_APPROVAL:
            raise ValueError(f"Cannot approve work order in status: {work_order.approval_workflow.status}")
        now = datetime.now(timezone.utc)
        work_order.approval_workflow.status = ApprovalStatus.APPROVED
        work_order.approval_workflow.approved_by = approved_by
        work_order.approval_workflow.approved_at = now
        if comments:
            work_order.approval_workflow.comments.append(f"[{approved_by}] {comments}")
        logger.info(f"Work order {work_order.work_order_id} approved by {approved_by}")
        return work_order

    def reject_work_order(self, work_order: DraftWorkOrder, rejected_by: str, reason: str) -> DraftWorkOrder:
        """Reject a work order with reason."""
        if work_order.approval_workflow is None:
            raise ValueError("Work order has no approval workflow")
        if work_order.approval_workflow.status != ApprovalStatus.PENDING_APPROVAL:
            raise ValueError(f"Cannot reject work order in status: {work_order.approval_workflow.status}")
        work_order.approval_workflow.status = ApprovalStatus.REJECTED
        work_order.approval_workflow.rejection_reason = reason
        work_order.approval_workflow.comments.append(f"[{rejected_by}] Rejected: {reason}")
        logger.info(f"Work order {work_order.work_order_id} rejected by {rejected_by}: {reason}")
        return work_order

    def cancel_work_order(self, work_order: DraftWorkOrder, cancelled_by: str, reason: str) -> DraftWorkOrder:
        """Cancel a work order."""
        if work_order.approval_workflow is None:
            work_order.approval_workflow = ApprovalWorkflow(
                workflow_id=f"APPR-{work_order.work_order_id}", work_order_id=work_order.work_order_id
            )
        work_order.approval_workflow.status = ApprovalStatus.CANCELLED
        work_order.approval_workflow.comments.append(f"[{cancelled_by}] Cancelled: {reason}")
        logger.info(f"Work order {work_order.work_order_id} cancelled by {cancelled_by}: {reason}")
        return work_order

    def get_required_skills(self, action_type: ActionType) -> List[str]:
        """Get required technician skills for an action type."""
        skill_mapping = {
            ActionType.INSPECT: ["mechanical"], ActionType.LUBRICATE: ["lubrication"],
            ActionType.ALIGN: ["alignment", "mechanical"], ActionType.BALANCE: ["vibration", "mechanical"],
            ActionType.REPLACE_BEARING: ["mechanical"], ActionType.REPLACE_SEAL: ["mechanical"],
            ActionType.REPLACE_FILTER: ["mechanical"], ActionType.REPLACE_BELT: ["mechanical"],
            ActionType.ADJUST: ["mechanical"], ActionType.CLEAN: ["mechanical"],
            ActionType.CALIBRATE: ["instrumentation"], ActionType.OVERHAUL: ["mechanical", "electrical"]
        }
        return skill_mapping.get(action_type, ["mechanical"])

    def estimate_duration(self, action_type: ActionType, asset_class: Optional[str] = None) -> float:
        """Estimate work duration in hours for an action type."""
        base_duration = {
            ActionType.INSPECT: 1.0, ActionType.LUBRICATE: 0.5, ActionType.ALIGN: 3.0,
            ActionType.BALANCE: 4.0, ActionType.REPLACE_BEARING: 6.0, ActionType.REPLACE_SEAL: 4.0,
            ActionType.REPLACE_FILTER: 0.5, ActionType.REPLACE_BELT: 2.0, ActionType.ADJUST: 1.5,
            ActionType.CLEAN: 2.0, ActionType.CALIBRATE: 2.0, ActionType.OVERHAUL: 16.0
        }
        duration = base_duration.get(action_type, 2.0)
        if asset_class:
            class_multipliers = {"critical": 1.5, "large": 1.3, "complex": 1.4, "standard": 1.0, "simple": 0.8}
            duration *= class_multipliers.get(asset_class.lower(), 1.0)
        return round(duration, 1)
