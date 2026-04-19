"""
GL-002 FLAMEGUARD - Explanation Audit Logger

Audit logging for explainability decisions with provenance tracking.
Integrates with the main audit system for regulatory compliance.

This module provides:
- Audit logging for all explanation events
- Provenance tracking for explainability decisions
- Integration with DecisionExplainer for automatic logging
- Compliance export for regulatory reporting
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)


class ExplanationAuditEventType(Enum):
    """Types of explanation audit events."""
    EXPLANATION_GENERATED = "explanation_generated"
    EXPLANATION_ACCESSED = "explanation_accessed"
    EXPLANATION_EXPORTED = "explanation_exported"
    COUNTERFACTUAL_GENERATED = "counterfactual_generated"
    LIME_EXPLANATION_GENERATED = "lime_explanation_generated"
    PHYSICS_GROUNDING_GENERATED = "physics_grounding_generated"
    VISUALIZATION_DATA_GENERATED = "visualization_data_generated"
    EXPLANATION_CACHED = "explanation_cached"
    EXPLANATION_CACHE_HIT = "explanation_cache_hit"
    EXPLANATION_CACHE_CLEARED = "explanation_cache_cleared"


@dataclass
class ExplanationAuditEntry:
    """Single explanation audit log entry."""
    
    entry_id: str
    timestamp: datetime
    event_type: ExplanationAuditEventType
    explanation_id: str
    boiler_id: str
    
    explanation_type: str = ""
    target_variable: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    generation_time_ms: float = 0.0
    provenance_hash: str = ""
    input_hash: str = ""
    entry_hash: str = ""
    
    def __post_init__(self):
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute entry hash for tamper detection."""
        data = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "explanation_id": self.explanation_id,
            "boiler_id": self.boiler_id,
            "provenance_hash": self.provenance_hash,
        }
        json_data = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "explanation_id": self.explanation_id,
            "boiler_id": self.boiler_id,
            "explanation_type": self.explanation_type,
            "target_variable": self.target_variable,
            "details": self.details,
            "generation_time_ms": self.generation_time_ms,
            "provenance_hash": self.provenance_hash,
            "input_hash": self.input_hash,
            "entry_hash": self.entry_hash,
        }



class ExplanationAuditLogger:
    """
    Audit logger for explainability decisions.
    
    This class provides comprehensive audit logging for all explanation
    generation and access events, supporting regulatory compliance.
    
    Features:
    - Tamper-evident logging with hash chains
    - Provenance tracking for all explanations
    - Performance metrics capture
    - Export for compliance reporting
    """
    
    def __init__(
        self,
        agent_id: str = "GL-002",
        on_audit_event: Optional[Callable[[ExplanationAuditEntry], None]] = None,
        max_entries: int = 50000,
    ) -> None:
        """Initialize ExplanationAuditLogger."""
        self.agent_id = agent_id
        self._on_audit_event = on_audit_event
        self._max_entries = max_entries
        self._entries: List[ExplanationAuditEntry] = []
        self._chain_hash = "0" * 64
        self._stats = {
            "total_entries": 0,
            "explanations_generated": 0,
            "explanations_accessed": 0,
            "cache_hits": 0,
            "average_generation_time_ms": 0.0,
        }
        logger.info(f"ExplanationAuditLogger initialized: {agent_id}")
    
    def log(
        self,
        event_type: ExplanationAuditEventType,
        explanation_id: str,
        boiler_id: str,
        explanation_type: str = "",
        target_variable: str = "",
        details: Optional[Dict[str, Any]] = None,
        generation_time_ms: float = 0.0,
        provenance_hash: str = "",
        input_hash: str = "",
    ) -> ExplanationAuditEntry:
        """Create audit log entry."""
        entry = ExplanationAuditEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            explanation_id=explanation_id,
            boiler_id=boiler_id,
            explanation_type=explanation_type,
            target_variable=target_variable,
            details=details or {},
            generation_time_ms=generation_time_ms,
            provenance_hash=provenance_hash,
            input_hash=input_hash,
        )
        
        self._update_chain(entry)
        self._entries.append(entry)
        self._update_stats(event_type, generation_time_ms)
        
        if self._on_audit_event:
            self._on_audit_event(entry)
        
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]
        
        logger.debug(f"Explanation audit: {event_type.value} - {explanation_id}")
        return entry
    
    def _update_chain(self, entry: ExplanationAuditEntry) -> None:
        """Update chain hash for tamper detection."""
        combined = f"{self._chain_hash}:{entry.entry_hash}"
        self._chain_hash = hashlib.sha256(combined.encode()).hexdigest()
    
    def _update_stats(
        self, event_type: ExplanationAuditEventType, generation_time_ms: float
    ) -> None:
        """Update statistics."""
        self._stats["total_entries"] += 1
        
        if event_type == ExplanationAuditEventType.EXPLANATION_GENERATED:
            self._stats["explanations_generated"] += 1
            count = self._stats["explanations_generated"]
            old_avg = self._stats["average_generation_time_ms"]
            self._stats["average_generation_time_ms"] = (
                (old_avg * (count - 1) + generation_time_ms) / count
            )
        elif event_type == ExplanationAuditEventType.EXPLANATION_ACCESSED:
            self._stats["explanations_accessed"] += 1
        elif event_type == ExplanationAuditEventType.EXPLANATION_CACHE_HIT:
            self._stats["cache_hits"] += 1
    
    def log_explanation_generated(
        self,
        explanation_id: str,
        boiler_id: str,
        explanation_type: str,
        target_variable: str,
        provenance_hash: str,
        generation_time_ms: float,
        feature_count: int = 0,
        counterfactual_count: int = 0,
        input_data: Optional[Dict[str, float]] = None,
    ) -> ExplanationAuditEntry:
        """Log explanation generation event."""
        input_hash = ""
        if input_data:
            input_str = json.dumps(input_data, sort_keys=True)
            input_hash = hashlib.sha256(input_str.encode()).hexdigest()[:16]
        
        return self.log(
            event_type=ExplanationAuditEventType.EXPLANATION_GENERATED,
            explanation_id=explanation_id,
            boiler_id=boiler_id,
            explanation_type=explanation_type,
            target_variable=target_variable,
            details={
                "feature_count": feature_count,
                "counterfactual_count": counterfactual_count,
                "has_physics_grounding": True,
                "has_lime_explanation": explanation_type == "efficiency",
            },
            generation_time_ms=generation_time_ms,
            provenance_hash=provenance_hash,
            input_hash=input_hash,
        )
    
    def log_explanation_accessed(
        self,
        explanation_id: str,
        boiler_id: str,
        accessor: Optional[str] = None,
        access_method: str = "api",
    ) -> ExplanationAuditEntry:
        """Log explanation access event."""
        return self.log(
            event_type=ExplanationAuditEventType.EXPLANATION_ACCESSED,
            explanation_id=explanation_id,
            boiler_id=boiler_id,
            details={"accessor": accessor, "access_method": access_method},
        )
    
    def log_explanation_exported(
        self,
        explanation_id: str,
        boiler_id: str,
        export_format: str,
        exporter: Optional[str] = None,
    ) -> ExplanationAuditEntry:
        """Log explanation export event."""
        return self.log(
            event_type=ExplanationAuditEventType.EXPLANATION_EXPORTED,
            explanation_id=explanation_id,
            boiler_id=boiler_id,
            details={"export_format": export_format, "exporter": exporter},
        )
    
    def log_cache_event(
        self,
        event_type: ExplanationAuditEventType,
        explanation_id: str = "",
        boiler_id: str = "",
        cache_size: int = 0,
    ) -> ExplanationAuditEntry:
        """Log cache-related events."""
        return self.log(
            event_type=event_type,
            explanation_id=explanation_id,
            boiler_id=boiler_id,
            details={"cache_size": cache_size},
        )

    
    def get_entries(
        self,
        explanation_id: Optional[str] = None,
        boiler_id: Optional[str] = None,
        event_type: Optional[ExplanationAuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[ExplanationAuditEntry]:
        """Get audit entries with filters."""
        entries = self._entries
        
        if explanation_id:
            entries = [e for e in entries if e.explanation_id == explanation_id]
        if boiler_id:
            entries = [e for e in entries if e.boiler_id == boiler_id]
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
        
        entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]
    
    def get_explanation_audit_trail(
        self, explanation_id: str
    ) -> List[ExplanationAuditEntry]:
        """Get complete audit trail for an explanation."""
        return [e for e in self._entries if e.explanation_id == explanation_id]
    
    def export_for_compliance(
        self,
        boiler_id: str,
        start_time: datetime,
        end_time: datetime,
        include_chain_verification: bool = True,
    ) -> Dict[str, Any]:
        """Export audit data for regulatory compliance."""
        entries = self.get_entries(
            boiler_id=boiler_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )
        
        export = {
            "agent_id": self.agent_id,
            "boiler_id": boiler_id,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "period_start": start_time.isoformat(),
            "period_end": end_time.isoformat(),
            "entry_count": len(entries),
            "entries": [e.to_dict() for e in reversed(entries)],
            "statistics": self.get_statistics(),
        }
        
        if include_chain_verification:
            export["chain_hash"] = self._chain_hash
            export["chain_verified"] = self._verify_chain()
        
        return export
    
    def _verify_chain(self) -> bool:
        """Verify chain hash integrity."""
        computed_hash = "0" * 64
        for entry in self._entries:
            combined = f"{computed_hash}:{entry.entry_hash}"
            computed_hash = hashlib.sha256(combined.encode()).hexdigest()
        return computed_hash == self._chain_hash
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        cache_hit_rate = 0.0
        if self._stats["explanations_accessed"] > 0:
            cache_hit_rate = (
                self._stats["cache_hits"] / self._stats["explanations_accessed"]
            ) * 100
        
        return {
            **self._stats,
            "entries_count": len(self._entries),
            "chain_hash": self._chain_hash[:16] + "...",
            "chain_verified": self._verify_chain(),
            "cache_hit_rate": cache_hit_rate,
        }
    
    def clear(self) -> None:
        """Clear all audit entries."""
        self._entries.clear()
        self._chain_hash = "0" * 64
        self._stats = {
            "total_entries": 0,
            "explanations_generated": 0,
            "explanations_accessed": 0,
            "cache_hits": 0,
            "average_generation_time_ms": 0.0,
        }
        logger.info("Explanation audit log cleared")



class AuditedDecisionExplainer:
    """
    Wrapper for DecisionExplainer with automatic audit logging.
    
    This class wraps DecisionExplainer and automatically logs all
    explanation generation and access events to the audit system.
    """
    
    def __init__(
        self,
        explainer,  # DecisionExplainer instance
        audit_logger: ExplanationAuditLogger,
    ) -> None:
        """Initialize AuditedDecisionExplainer."""
        self._explainer = explainer
        self._audit_logger = audit_logger
        logger.info("AuditedDecisionExplainer initialized")
    
    def explain_efficiency(
        self,
        boiler_id: str,
        process_data: Dict[str, float],
        efficiency_result: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None,
    ):
        """Generate and log efficiency explanation."""
        start_time = datetime.now(timezone.utc)
        
        explanation = self._explainer.explain_efficiency(
            boiler_id=boiler_id,
            process_data=process_data,
            efficiency_result=efficiency_result,
            constraints=constraints,
        )
        
        generation_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000
        
        self._audit_logger.log_explanation_generated(
            explanation_id=explanation.explanation_id,
            boiler_id=boiler_id,
            explanation_type="efficiency",
            target_variable="efficiency_percent",
            provenance_hash=explanation.provenance_hash,
            generation_time_ms=generation_time,
            feature_count=len(explanation.feature_contributions),
            counterfactual_count=len(explanation.counterfactuals),
            input_data=process_data,
        )
        
        return explanation
    
    def explain_o2_trim_adjustment(
        self,
        boiler_id: str,
        current_o2: float,
        target_o2: float,
        current_co: float,
        load_percent: float,
        reason: str = "optimization",
    ):
        """Generate and log O2 trim explanation."""
        start_time = datetime.now(timezone.utc)
        
        explanation = self._explainer.explain_o2_trim_adjustment(
            boiler_id=boiler_id,
            current_o2=current_o2,
            target_o2=target_o2,
            current_co=current_co,
            load_percent=load_percent,
            reason=reason,
        )
        
        generation_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000
        
        self._audit_logger.log_explanation_generated(
            explanation_id=explanation.explanation_id,
            boiler_id=boiler_id,
            explanation_type="o2_trim",
            target_variable="o2_setpoint",
            provenance_hash=explanation.provenance_hash,
            generation_time_ms=generation_time,
            feature_count=len(explanation.feature_contributions),
            counterfactual_count=len(explanation.counterfactuals),
            input_data={
                "current_o2": current_o2,
                "target_o2": target_o2,
                "current_co": current_co,
                "load_percent": load_percent,
            },
        )
        
        return explanation
    
    def explain_safety_intervention(
        self,
        boiler_id: str,
        intervention_type: str,
        trigger_value: float,
        setpoint: float,
        tag: str,
        action_taken: str,
    ):
        """Generate and log safety intervention explanation."""
        start_time = datetime.now(timezone.utc)
        
        explanation = self._explainer.explain_safety_intervention(
            boiler_id=boiler_id,
            intervention_type=intervention_type,
            trigger_value=trigger_value,
            setpoint=setpoint,
            tag=tag,
            action_taken=action_taken,
        )
        
        generation_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000
        
        self._audit_logger.log_explanation_generated(
            explanation_id=explanation.explanation_id,
            boiler_id=boiler_id,
            explanation_type="safety",
            target_variable=tag,
            provenance_hash=explanation.provenance_hash,
            generation_time_ms=generation_time,
            feature_count=len(explanation.feature_contributions),
            counterfactual_count=0,
            input_data={
                "trigger_value": trigger_value,
                "setpoint": setpoint,
            },
        )
        
        return explanation
    
    def get_explanation(self, explanation_id: str, accessor: Optional[str] = None):
        """Get explanation and log access."""
        explanation = self._explainer.get_explanation(explanation_id)
        
        if explanation:
            self._audit_logger.log_explanation_accessed(
                explanation_id=explanation_id,
                boiler_id=explanation.boiler_id,
                accessor=accessor,
            )
        
        return explanation
    
    def get_recent_explanations(self, *args, **kwargs):
        """Passthrough to underlying explainer."""
        return self._explainer.get_recent_explanations(*args, **kwargs)
    
    def clear_cache(self) -> None:
        """Clear cache and log event."""
        cache_size = len(self._explainer._explanations)
        self._explainer.clear_cache()
        
        self._audit_logger.log_cache_event(
            event_type=ExplanationAuditEventType.EXPLANATION_CACHE_CLEARED,
            cache_size=cache_size,
        )
