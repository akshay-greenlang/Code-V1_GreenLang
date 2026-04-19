# -*- coding: utf-8 -*-
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

class UncertaintyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DecisionGate(Enum):
    AUTO_APPROVE = "auto_approve"
    HUMAN_REVIEW = "human_review"
    HUMAN_APPROVE = "human_approve"
    BLOCKED = "blocked"

class AuditAction(Enum):
    PREDICTION_MADE = "prediction_made"
    DECISION_GATED = "decision_gated"
    HUMAN_OVERRIDE = "human_override"
    WORK_ORDER_CREATED = "work_order_created"
    THRESHOLD_BREACHED = "threshold_breached"

@dataclass
class UncertaintyThresholds:
    low_max: float = 0.2
    medium_max: float = 0.4
    high_max: float = 0.7
    auto_approve_max: float = 0.15
    human_review_min: float = 0.3
    block_min: float = 0.8

@dataclass
class GatingDecision:
    decision_id: str
    prediction_id: str
    gate_type: DecisionGate
    uncertainty_level: UncertaintyLevel
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    confidence_interval_width: float
    reason: str
    requires_human_action: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

@dataclass
class HumanDecision:
    decision_id: str
    gating_decision_id: str
    reviewer_id: str
    action: str
    rationale: str
    approved: bool
    override_prediction: bool
    new_prediction_value: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AuditLogEntry:
    entry_id: str
    action: AuditAction
    asset_id: str
    prediction_id: Optional[str]
    decision_id: Optional[str]
    user_id: Optional[str]
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

class UncertaintyGate:
    def __init__(self, thresholds: Optional[UncertaintyThresholds] = None):
        self.thresholds = thresholds or UncertaintyThresholds()
        
    def classify_uncertainty(self, epistemic: float, aleatoric: float) -> UncertaintyLevel:
        total = epistemic + aleatoric
        if total <= self.thresholds.low_max:
            return UncertaintyLevel.LOW
        elif total <= self.thresholds.medium_max:
            return UncertaintyLevel.MEDIUM
        elif total <= self.thresholds.high_max:
            return UncertaintyLevel.HIGH
        return UncertaintyLevel.CRITICAL
    
    def determine_gate(self, epistemic: float, aleatoric: float, ci_width: float) -> DecisionGate:
        total = epistemic + aleatoric
        if total <= self.thresholds.auto_approve_max and ci_width < 0.2:
            return DecisionGate.AUTO_APPROVE
        elif total >= self.thresholds.block_min:
            return DecisionGate.BLOCKED
        elif total >= self.thresholds.human_review_min:
            return DecisionGate.HUMAN_APPROVE
        return DecisionGate.HUMAN_REVIEW
    
    def evaluate(self, prediction_id: str, epistemic: float, aleatoric: float, ci_width: float) -> GatingDecision:
        total = epistemic + aleatoric
        uncertainty_level = self.classify_uncertainty(epistemic, aleatoric)
        gate_type = self.determine_gate(epistemic, aleatoric, ci_width)
        
        reasons = {
            DecisionGate.AUTO_APPROVE: "Low uncertainty, auto-approved",
            DecisionGate.HUMAN_REVIEW: "Medium uncertainty, requires human review",
            DecisionGate.HUMAN_APPROVE: "High uncertainty, requires human approval",
            DecisionGate.BLOCKED: "Critical uncertainty, action blocked",
        }
        
        decision_id = hashlib.sha256(f"{prediction_id}{total}{datetime.utcnow()}".encode()).hexdigest()[:16]
        provenance = hashlib.sha256(f"{decision_id}{gate_type.value}".encode()).hexdigest()
        
        return GatingDecision(
            decision_id=decision_id,
            prediction_id=prediction_id,
            gate_type=gate_type,
            uncertainty_level=uncertainty_level,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            confidence_interval_width=ci_width,
            reason=reasons[gate_type],
            requires_human_action=gate_type != DecisionGate.AUTO_APPROVE,
            provenance_hash=provenance,
        )

class HumanInTheLoop:
    def __init__(self):
        self._pending_decisions: Dict[str, GatingDecision] = {}
        self._human_decisions: List[HumanDecision] = []
        
    def queue_for_review(self, decision: GatingDecision) -> None:
        if decision.requires_human_action:
            self._pending_decisions[decision.decision_id] = decision
            
    def get_pending_reviews(self) -> List[GatingDecision]:
        return list(self._pending_decisions.values())
    
    def record_decision(self, human_decision: HumanDecision) -> bool:
        if human_decision.gating_decision_id in self._pending_decisions:
            self._human_decisions.append(human_decision)
            del self._pending_decisions[human_decision.gating_decision_id]
            return True
        return False
    
    def get_decision_history(self, reviewer_id: Optional[str] = None) -> List[HumanDecision]:
        if reviewer_id:
            return [d for d in self._human_decisions if d.reviewer_id == reviewer_id]
        return self._human_decisions.copy()

class AuditLogger:
    def __init__(self):
        self._log: List[AuditLogEntry] = []
        
    def log(self, action: AuditAction, asset_id: str, details: Dict[str, Any], prediction_id: Optional[str] = None, decision_id: Optional[str] = None, user_id: Optional[str] = None) -> AuditLogEntry:
        entry_id = hashlib.sha256(f"{action.value}{asset_id}{datetime.utcnow()}".encode()).hexdigest()[:16]
        provenance = hashlib.sha256(f"{entry_id}{str(details)}".encode()).hexdigest()
        
        entry = AuditLogEntry(
            entry_id=entry_id,
            action=action,
            asset_id=asset_id,
            prediction_id=prediction_id,
            decision_id=decision_id,
            user_id=user_id,
            details=details,
            provenance_hash=provenance,
        )
        self._log.append(entry)
        return entry
    
    def get_audit_trail(self, asset_id: Optional[str] = None, action: Optional[AuditAction] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[AuditLogEntry]:
        result = self._log.copy()
        if asset_id:
            result = [e for e in result if e.asset_id == asset_id]
        if action:
            result = [e for e in result if e.action == action]
        if start_time:
            result = [e for e in result if e.timestamp >= start_time]
        if end_time:
            result = [e for e in result if e.timestamp <= end_time]
        return result
    
    def export_log(self) -> List[Dict]:
        return [{"entry_id": e.entry_id, "action": e.action.value, "asset_id": e.asset_id, "timestamp": e.timestamp.isoformat(), "provenance": e.provenance_hash} for e in self._log]
