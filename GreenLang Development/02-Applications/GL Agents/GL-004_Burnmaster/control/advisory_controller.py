"""
GL-004 BURNMASTER - Advisory Controller

This module implements the advisory control system for the burner management
control system. It provides operator recommendations with explanations,
requiring manual acceptance before any changes are applied.

Key Features:
    - Generate recommendations with explanations
    - Present recommendations to operators
    - Track operator responses and acceptance rates
    - Measure effectiveness of applied recommendations
    - Complete audit trail

Operating Mode: ADVISORY
    - Present recommendations with explanations
    - Require manual acceptance
    - No automatic writes to DCS

Reference Standards:
    - IEC 61511 Functional Safety
    - ISA-84 Safety Instrumented Systems

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import logging
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AdvisoryType(str, Enum):
    """Types of advisories."""
    OPTIMIZATION = "optimization"
    SAFETY = "safety"
    MAINTENANCE = "maintenance"
    EFFICIENCY = "efficiency"
    EMISSIONS = "emissions"
    COST_SAVINGS = "cost_savings"
    ALARM = "alarm"


class AdvisoryPriority(str, Enum):
    """Priority levels for advisories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AdvisoryStatus(str, Enum):
    """Status of an advisory."""
    PENDING = "pending"
    PRESENTED = "presented"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"
    APPLIED = "applied"
    SUPERSEDED = "superseded"


class ResponseType(str, Enum):
    """Types of operator responses to advisories."""
    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"
    MODIFY = "modify"
    ACKNOWLEDGE = "acknowledge"


class Advisory(BaseModel):
    """An advisory recommendation for the operator."""
    advisory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    advisory_type: AdvisoryType
    priority: AdvisoryPriority
    title: str
    description: str
    explanation: str = Field(default="")
    recommended_action: str = Field(default="")
    current_value: Optional[float] = None
    recommended_value: Optional[float] = None
    unit: str = Field(default="")
    estimated_benefit: str = Field(default="")
    risk_assessment: str = Field(default="low risk")
    source: str = Field(default="OPTIMIZER")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    status: AdvisoryStatus = Field(default=AdvisoryStatus.PENDING)
    presented_at: Optional[datetime] = None
    response: Optional["OperatorResponse"] = None
    applied_at: Optional[datetime] = None
    effectiveness: Optional["EffectivenessMetrics"] = None
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            hash_input = f"{self.advisory_id}|{self.advisory_type.value}|{self.title}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class OperatorResponse(BaseModel):
    """Operator's response to an advisory."""
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    advisory_id: str
    response_type: ResponseType
    operator_id: str
    comment: str = Field(default="")
    modified_value: Optional[float] = None
    response_time_seconds: float = Field(default=0.0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            hash_input = f"{self.response_id}|{self.advisory_id}|{self.response_type.value}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class PresentationResult(BaseModel):
    """Result of presenting an advisory to an operator."""
    advisory_id: str
    presented: bool
    presentation_channel: str = Field(default="HMI")
    presented_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    operator_id: Optional[str] = None
    error_message: Optional[str] = None


class EffectivenessMetrics(BaseModel):
    """Metrics measuring effectiveness of an applied advisory."""
    advisory_id: str
    measurement_period_seconds: float = Field(default=3600.0)
    actual_efficiency_change: float = Field(default=0.0)
    actual_emissions_change: float = Field(default=0.0)
    actual_fuel_savings_percent: float = Field(default=0.0)
    estimated_vs_actual_variance: float = Field(default=0.0)
    stable_operation: bool = Field(default=True)
    measured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = Field(default="")


class AdvisoryController:
    """
    Controls the advisory system for operator recommendations.

    This class manages the advisory workflow:
    - Generate advisories with explanations
    - Present to operators
    - Track responses and acceptance
    - Measure effectiveness

    Example:
        >>> controller = AdvisoryController()
        >>> advisory = controller.generate_advisory(
        ...     advisory_type=AdvisoryType.OPTIMIZATION,
        ...     title="Reduce Excess O2",
        ...     description="Current excess O2 is high",
        ...     recommended_value=2.5
        ... )
        >>> result = controller.present_advisory(advisory, "OPERATOR_001")
        >>> response = controller.record_response(advisory.advisory_id, ...)
    """

    # Default expiry times for different priority levels
    DEFAULT_EXPIRY: Dict[AdvisoryPriority, timedelta] = {
        AdvisoryPriority.LOW: timedelta(hours=24),
        AdvisoryPriority.MEDIUM: timedelta(hours=8),
        AdvisoryPriority.HIGH: timedelta(hours=1),
        AdvisoryPriority.CRITICAL: timedelta(minutes=15),
    }

    def __init__(self) -> None:
        """Initialize the advisory controller."""
        self._pending_advisories: Dict[str, Advisory] = {}
        self._presented_advisories: Dict[str, Advisory] = {}
        self._completed_advisories: List[Advisory] = []
        self._response_history: List[OperatorResponse] = []
        self._effectiveness_history: List[EffectivenessMetrics] = []
        self._audit_log: List[Dict[str, Any]] = []

        # Statistics
        self._total_generated = 0
        self._total_accepted = 0
        self._total_rejected = 0
        self._total_expired = 0

        logger.info("AdvisoryController initialized")

    def generate_advisory(
        self,
        advisory_type: AdvisoryType,
        title: str,
        description: str,
        priority: AdvisoryPriority = AdvisoryPriority.MEDIUM,
        explanation: str = "",
        recommended_action: str = "",
        current_value: Optional[float] = None,
        recommended_value: Optional[float] = None,
        unit: str = "",
        estimated_benefit: str = "",
        source: str = "OPTIMIZER",
        confidence: float = 0.8,
        expiry_override: Optional[timedelta] = None
    ) -> Advisory:
        """
        Generate a new advisory recommendation.

        Args:
            advisory_type: Type of advisory
            title: Short title for the advisory
            description: Detailed description
            priority: Priority level
            explanation: Technical explanation for operators
            recommended_action: Specific action to take
            current_value: Current value (if applicable)
            recommended_value: Recommended new value
            unit: Unit of measurement
            estimated_benefit: Expected benefit from applying
            source: Source of the recommendation
            confidence: Confidence level (0-1)
            expiry_override: Override default expiry time

        Returns:
            Generated Advisory
        """
        expiry = expiry_override or self.DEFAULT_EXPIRY.get(priority, timedelta(hours=8))
        expires_at = datetime.now(timezone.utc) + expiry

        # Auto-generate explanation if not provided
        if not explanation and current_value is not None and recommended_value is not None:
            change = recommended_value - current_value
            direction = "increase" if change > 0 else "decrease"
            explanation = (
                f"Analysis suggests {direction}ing the setpoint from {current_value}{unit} "
                f"to {recommended_value}{unit} ({abs(change):.2f}{unit} change). "
                f"This recommendation is based on current operating conditions with "
                f"{confidence*100:.0f}% confidence."
            )

        advisory = Advisory(
            advisory_type=advisory_type,
            priority=priority,
            title=title,
            description=description,
            explanation=explanation,
            recommended_action=recommended_action,
            current_value=current_value,
            recommended_value=recommended_value,
            unit=unit,
            estimated_benefit=estimated_benefit,
            source=source,
            confidence=confidence,
            expires_at=expires_at
        )

        self._pending_advisories[advisory.advisory_id] = advisory
        self._total_generated += 1

        self._log_event("ADVISORY_GENERATED", advisory)
        logger.info(f"Advisory generated: {title} [{priority.value}]")

        return advisory

    def present_advisory(
        self,
        advisory: Advisory,
        operator_id: Optional[str] = None,
        channel: str = "HMI"
    ) -> PresentationResult:
        """
        Present an advisory to an operator.

        Args:
            advisory: The advisory to present
            operator_id: ID of the operator (if known)
            channel: Presentation channel (HMI, MOBILE, EMAIL, etc.)

        Returns:
            PresentationResult with presentation details
        """
        # Check if advisory is still valid
        if advisory.status not in [AdvisoryStatus.PENDING]:
            return PresentationResult(
                advisory_id=advisory.advisory_id,
                presented=False,
                error_message=f"Advisory status is {advisory.status.value}, cannot present"
            )

        if advisory.expires_at and datetime.now(timezone.utc) > advisory.expires_at:
            advisory.status = AdvisoryStatus.EXPIRED
            self._total_expired += 1
            return PresentationResult(
                advisory_id=advisory.advisory_id,
                presented=False,
                error_message="Advisory has expired"
            )

        # Mark as presented
        advisory.status = AdvisoryStatus.PRESENTED
        advisory.presented_at = datetime.now(timezone.utc)

        # Move from pending to presented
        if advisory.advisory_id in self._pending_advisories:
            del self._pending_advisories[advisory.advisory_id]
        self._presented_advisories[advisory.advisory_id] = advisory

        result = PresentationResult(
            advisory_id=advisory.advisory_id,
            presented=True,
            presentation_channel=channel,
            operator_id=operator_id
        )

        self._log_event("ADVISORY_PRESENTED", result)
        logger.info(f"Advisory presented: {advisory.title} via {channel}")

        return result

    def record_response(
        self,
        advisory_id: str,
        response_type: ResponseType,
        operator_id: str,
        comment: str = "",
        modified_value: Optional[float] = None
    ) -> OperatorResponse:
        """
        Record an operator's response to an advisory.

        Args:
            advisory_id: ID of the advisory being responded to
            response_type: Type of response
            operator_id: ID of the responding operator
            comment: Optional comment
            modified_value: Modified value if operator modified the recommendation

        Returns:
            OperatorResponse record
        """
        advisory = self._presented_advisories.get(advisory_id)
        if not advisory:
            advisory = self._pending_advisories.get(advisory_id)

        if not advisory:
            raise ValueError(f"Advisory not found: {advisory_id}")

        # Calculate response time
        response_time = 0.0
        if advisory.presented_at:
            response_time = (datetime.now(timezone.utc) - advisory.presented_at).total_seconds()

        response = OperatorResponse(
            advisory_id=advisory_id,
            response_type=response_type,
            operator_id=operator_id,
            comment=comment,
            modified_value=modified_value,
            response_time_seconds=response_time
        )

        # Update advisory status
        if response_type == ResponseType.ACCEPT:
            advisory.status = AdvisoryStatus.ACCEPTED
            self._total_accepted += 1
        elif response_type == ResponseType.REJECT:
            advisory.status = AdvisoryStatus.REJECTED
            self._total_rejected += 1
        elif response_type == ResponseType.DEFER:
            # Keep as presented, operator will respond later
            pass
        elif response_type == ResponseType.MODIFY:
            advisory.status = AdvisoryStatus.ACCEPTED
            advisory.recommended_value = modified_value
            self._total_accepted += 1

        advisory.response = response
        self._response_history.append(response)

        # Move to completed if terminal state
        if advisory.status in [AdvisoryStatus.ACCEPTED, AdvisoryStatus.REJECTED]:
            if advisory.advisory_id in self._presented_advisories:
                del self._presented_advisories[advisory.advisory_id]
            self._completed_advisories.append(advisory)

        self._log_event("ADVISORY_RESPONSE", response)
        logger.info(f"Advisory response recorded: {response_type.value} by {operator_id}")

        return response

    def mark_applied(self, advisory_id: str) -> bool:
        """
        Mark an advisory as applied to the process.

        Args:
            advisory_id: ID of the advisory that was applied

        Returns:
            True if successfully marked
        """
        # Find the advisory
        advisory = None
        for a in self._completed_advisories:
            if a.advisory_id == advisory_id:
                advisory = a
                break

        if not advisory:
            return False

        if advisory.status != AdvisoryStatus.ACCEPTED:
            return False

        advisory.status = AdvisoryStatus.APPLIED
        advisory.applied_at = datetime.now(timezone.utc)

        self._log_event("ADVISORY_APPLIED", {"advisory_id": advisory_id})
        logger.info(f"Advisory marked as applied: {advisory_id}")

        return True

    def record_effectiveness(
        self,
        advisory_id: str,
        actual_efficiency_change: float = 0.0,
        actual_emissions_change: float = 0.0,
        actual_fuel_savings_percent: float = 0.0,
        stable_operation: bool = True,
        notes: str = ""
    ) -> EffectivenessMetrics:
        """
        Record the effectiveness of an applied advisory.

        Args:
            advisory_id: ID of the applied advisory
            actual_efficiency_change: Measured efficiency change
            actual_emissions_change: Measured emissions change
            actual_fuel_savings_percent: Measured fuel savings
            stable_operation: Whether operation remained stable
            notes: Additional notes

        Returns:
            EffectivenessMetrics record
        """
        # Find the advisory
        advisory = None
        for a in self._completed_advisories:
            if a.advisory_id == advisory_id:
                advisory = a
                break

        estimated_variance = 0.0
        if advisory:
            # Calculate variance from estimates if available
            # This is a simplified calculation
            pass

        metrics = EffectivenessMetrics(
            advisory_id=advisory_id,
            actual_efficiency_change=actual_efficiency_change,
            actual_emissions_change=actual_emissions_change,
            actual_fuel_savings_percent=actual_fuel_savings_percent,
            estimated_vs_actual_variance=estimated_variance,
            stable_operation=stable_operation,
            notes=notes
        )

        if advisory:
            advisory.effectiveness = metrics

        self._effectiveness_history.append(metrics)
        self._log_event("EFFECTIVENESS_RECORDED", metrics)

        logger.info(f"Effectiveness recorded for advisory: {advisory_id}")

        return metrics

    def expire_stale_advisories(self) -> List[str]:
        """
        Expire advisories that have passed their expiry time.

        Returns:
            List of expired advisory IDs
        """
        now = datetime.now(timezone.utc)
        expired = []

        # Check pending advisories
        for advisory_id, advisory in list(self._pending_advisories.items()):
            if advisory.expires_at and now > advisory.expires_at:
                advisory.status = AdvisoryStatus.EXPIRED
                del self._pending_advisories[advisory_id]
                self._completed_advisories.append(advisory)
                self._total_expired += 1
                expired.append(advisory_id)

        # Check presented advisories
        for advisory_id, advisory in list(self._presented_advisories.items()):
            if advisory.expires_at and now > advisory.expires_at:
                advisory.status = AdvisoryStatus.EXPIRED
                del self._presented_advisories[advisory_id]
                self._completed_advisories.append(advisory)
                self._total_expired += 1
                expired.append(advisory_id)

        if expired:
            self._log_event("ADVISORIES_EXPIRED", {"count": len(expired), "ids": expired})
            logger.info(f"Expired {len(expired)} stale advisories")

        return expired

    def supersede_advisory(self, old_advisory_id: str, new_advisory: Advisory) -> bool:
        """
        Supersede an existing advisory with a new one.

        Args:
            old_advisory_id: ID of the advisory to supersede
            new_advisory: The new advisory

        Returns:
            True if supersession successful
        """
        # Find old advisory
        old_advisory = self._pending_advisories.get(old_advisory_id)
        if not old_advisory:
            old_advisory = self._presented_advisories.get(old_advisory_id)

        if not old_advisory:
            return False

        old_advisory.status = AdvisoryStatus.SUPERSEDED

        # Remove from active lists
        if old_advisory_id in self._pending_advisories:
            del self._pending_advisories[old_advisory_id]
        if old_advisory_id in self._presented_advisories:
            del self._presented_advisories[old_advisory_id]

        self._completed_advisories.append(old_advisory)

        self._log_event("ADVISORY_SUPERSEDED", {
            "old_id": old_advisory_id,
            "new_id": new_advisory.advisory_id
        })

        return True

    def get_pending_advisories(self) -> List[Advisory]:
        """Get all pending advisories."""
        return list(self._pending_advisories.values())

    def get_presented_advisories(self) -> List[Advisory]:
        """Get all presented advisories awaiting response."""
        return list(self._presented_advisories.values())

    def get_advisory(self, advisory_id: str) -> Optional[Advisory]:
        """Get a specific advisory by ID."""
        if advisory_id in self._pending_advisories:
            return self._pending_advisories[advisory_id]
        if advisory_id in self._presented_advisories:
            return self._presented_advisories[advisory_id]
        for a in self._completed_advisories:
            if a.advisory_id == advisory_id:
                return a
        return None

    def get_acceptance_rate(self) -> float:
        """Get the overall advisory acceptance rate."""
        total_responses = self._total_accepted + self._total_rejected
        if total_responses == 0:
            return 0.0
        return self._total_accepted / total_responses

    def get_statistics(self) -> Dict[str, Any]:
        """Get advisory statistics."""
        return {
            "total_generated": self._total_generated,
            "total_accepted": self._total_accepted,
            "total_rejected": self._total_rejected,
            "total_expired": self._total_expired,
            "pending_count": len(self._pending_advisories),
            "presented_count": len(self._presented_advisories),
            "acceptance_rate": self.get_acceptance_rate(),
            "average_response_time_seconds": self._calculate_avg_response_time()
        }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the advisory controller."""
        return {
            "statistics": self.get_statistics(),
            "pending_advisories": len(self._pending_advisories),
            "presented_advisories": len(self._presented_advisories),
            "completed_advisories": len(self._completed_advisories)
        }

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from history."""
        if not self._response_history:
            return 0.0
        total = sum(r.response_time_seconds for r in self._response_history)
        return total / len(self._response_history)

    def _log_event(self, event_type: str, data: Any) -> None:
        """Log an event to the audit trail."""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "data": data.model_dump() if hasattr(data, 'model_dump') else data
        }
        self._audit_log.append(audit_entry)

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the audit log."""
        return list(reversed(self._audit_log[-limit:]))
