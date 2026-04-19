"""
Deployment Governance for GL-003 UNIFIEDSTEAM

Provides safe deployment practices including canary releases,
rollback policies, and gating criteria for ML model deployments.

Author: GL-003 MLOps Team
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import hashlib
import logging

logger = logging.getLogger(__name__)


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    DIRECT = "direct"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    SHADOW = "shadow"
    A_B_TEST = "a_b_test"


class GateStatus(Enum):
    """Gate check status."""
    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"
    SKIPPED = "skipped"


class RollbackReason(Enum):
    """Reasons for rollback."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_SPIKE = "error_rate_spike"
    LATENCY_INCREASE = "latency_increase"
    DRIFT_DETECTED = "drift_detected"
    MANUAL_REQUEST = "manual_request"
    GATE_FAILURE = "gate_failure"
    SAFETY_VIOLATION = "safety_violation"


@dataclass
class GatingCriteria:
    """
    Gating criteria for deployment decisions.

    Defines checks that must pass before deployment proceeds.
    """
    name: str
    description: str
    metric_name: str
    threshold: float
    comparison: str  # "above", "below", "equals"
    is_blocking: bool = True
    timeout_minutes: int = 60
    min_samples: int = 100

    def evaluate(self, value: float) -> GateStatus:
        """
        Evaluate if criteria is met.

        Args:
            value: Metric value to check

        Returns:
            GateStatus
        """
        if self.comparison == "above":
            passed = value >= self.threshold
        elif self.comparison == "below":
            passed = value <= self.threshold
        else:
            passed = abs(value - self.threshold) < 0.001

        return GateStatus.PASSED if passed else GateStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "is_blocking": self.is_blocking,
            "timeout_minutes": self.timeout_minutes,
            "min_samples": self.min_samples,
        }


@dataclass
class GateResult:
    """
    Result of a gate evaluation.
    """
    criteria: GatingCriteria
    status: GateStatus
    evaluated_at: datetime
    metric_value: Optional[float] = None
    sample_count: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "criteria_name": self.criteria.name,
            "status": self.status.value,
            "evaluated_at": self.evaluated_at.isoformat(),
            "metric_value": self.metric_value,
            "sample_count": self.sample_count,
            "message": self.message,
        }


@dataclass
class CanaryConfig:
    """
    Configuration for canary deployment.

    Defines traffic split and evaluation parameters.
    """
    initial_traffic_pct: float = 5.0
    increment_pct: float = 10.0
    evaluation_interval_minutes: int = 30
    min_evaluation_duration_minutes: int = 60
    max_evaluation_duration_minutes: int = 1440  # 24 hours

    # Automatic rollback thresholds
    auto_rollback_error_rate: float = 0.05
    auto_rollback_latency_increase_pct: float = 50.0
    auto_rollback_accuracy_drop_pct: float = 5.0

    # Success criteria
    min_traffic_for_success: float = 50.0
    success_duration_minutes: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initial_traffic_pct": self.initial_traffic_pct,
            "increment_pct": self.increment_pct,
            "evaluation_interval_minutes": self.evaluation_interval_minutes,
            "min_evaluation_duration_minutes": self.min_evaluation_duration_minutes,
            "max_evaluation_duration_minutes": self.max_evaluation_duration_minutes,
            "auto_rollback": {
                "error_rate": self.auto_rollback_error_rate,
                "latency_increase_pct": self.auto_rollback_latency_increase_pct,
                "accuracy_drop_pct": self.auto_rollback_accuracy_drop_pct,
            },
            "success_criteria": {
                "min_traffic_pct": self.min_traffic_for_success,
                "duration_minutes": self.success_duration_minutes,
            },
        }


@dataclass
class RollbackPolicy:
    """
    Policy for rollback decisions.

    Defines conditions and procedures for rolling back.
    """
    enabled: bool = True
    max_rollback_time_minutes: int = 5
    preserve_state: bool = True
    notify_on_rollback: List[str] = field(default_factory=list)

    # Automatic rollback triggers
    auto_rollback_enabled: bool = True
    error_rate_threshold: float = 0.05
    latency_p99_threshold_ms: float = 500.0
    accuracy_min_threshold: float = 0.75
    consecutive_failures: int = 3

    def should_auto_rollback(
        self,
        error_rate: float,
        latency_p99_ms: float,
        accuracy: float,
        consecutive_failures: int,
    ) -> Optional[RollbackReason]:
        """
        Check if automatic rollback should trigger.

        Args:
            error_rate: Current error rate
            latency_p99_ms: P99 latency in ms
            accuracy: Current accuracy
            consecutive_failures: Count of consecutive failures

        Returns:
            RollbackReason if rollback needed, None otherwise
        """
        if not self.auto_rollback_enabled:
            return None

        if error_rate > self.error_rate_threshold:
            return RollbackReason.ERROR_RATE_SPIKE

        if latency_p99_ms > self.latency_p99_threshold_ms:
            return RollbackReason.LATENCY_INCREASE

        if accuracy < self.accuracy_min_threshold:
            return RollbackReason.PERFORMANCE_DEGRADATION

        if consecutive_failures >= self.consecutive_failures:
            return RollbackReason.GATE_FAILURE

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "max_rollback_time_minutes": self.max_rollback_time_minutes,
            "preserve_state": self.preserve_state,
            "notify_on_rollback": self.notify_on_rollback,
            "auto_rollback": {
                "enabled": self.auto_rollback_enabled,
                "error_rate_threshold": self.error_rate_threshold,
                "latency_p99_threshold_ms": self.latency_p99_threshold_ms,
                "accuracy_min_threshold": self.accuracy_min_threshold,
                "consecutive_failures": self.consecutive_failures,
            },
        }


@dataclass
class DeploymentEvent:
    """
    Event in deployment lifecycle.
    """
    event_id: str
    event_type: str
    timestamp: datetime
    details: Dict[str, Any]
    triggered_by: str


@dataclass
class Deployment:
    """
    Deployment record.

    Tracks a model deployment through its lifecycle.
    """
    deployment_id: str
    model_id: str
    model_version: str
    target_environment: str
    strategy: DeploymentStrategy
    created_at: datetime
    created_by: str

    # Configuration
    canary_config: Optional[CanaryConfig] = None
    rollback_policy: Optional[RollbackPolicy] = None
    gating_criteria: List[GatingCriteria] = field(default_factory=list)

    # Status
    status: str = "pending"  # pending, deploying, canary, deployed, rolled_back, failed
    current_traffic_pct: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Previous version (for rollback)
    previous_version: Optional[str] = None

    # Results
    gate_results: List[GateResult] = field(default_factory=list)
    events: List[DeploymentEvent] = field(default_factory=list)
    rollback_reason: Optional[RollbackReason] = None

    def add_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        triggered_by: str = "system",
    ):
        """Add event to deployment."""
        import uuid
        event = DeploymentEvent(
            event_id=f"EVT-{uuid.uuid4().hex[:8].upper()}",
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            details=details,
            triggered_by=triggered_by,
        )
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "target_environment": self.target_environment,
            "strategy": self.strategy.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "canary_config": self.canary_config.to_dict() if self.canary_config else None,
            "rollback_policy": self.rollback_policy.to_dict() if self.rollback_policy else None,
            "gating_criteria": [g.to_dict() for g in self.gating_criteria],
            "status": self.status,
            "current_traffic_pct": self.current_traffic_pct,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "previous_version": self.previous_version,
            "gate_results": [g.to_dict() for g in self.gate_results],
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "timestamp": e.timestamp.isoformat(),
                    "details": e.details,
                    "triggered_by": e.triggered_by,
                }
                for e in self.events
            ],
            "rollback_reason": self.rollback_reason.value if self.rollback_reason else None,
        }


class DeploymentGovernor:
    """
    Deployment governance for GL-003 UNIFIEDSTEAM.

    Manages safe deployment practices with canary releases,
    gating criteria, and automatic rollback.

    Example:
        >>> governor = DeploymentGovernor()
        >>> deployment = governor.create_deployment(
        ...     model_id="trap-classifier",
        ...     model_version="1.0.0",
        ...     strategy=DeploymentStrategy.CANARY,
        ... )
        >>> governor.start_deployment(deployment.deployment_id)
    """

    def __init__(self):
        """Initialize deployment governor."""
        self._deployments: Dict[str, Deployment] = {}
        self._active_deployments: Dict[str, str] = {}  # environment -> deployment_id
        self._audit_log: List[Dict[str, Any]] = []

        # Default gating criteria
        self._default_gates = self._create_default_gates()

    def _create_default_gates(self) -> List[GatingCriteria]:
        """Create default gating criteria."""
        return [
            GatingCriteria(
                name="accuracy_gate",
                description="Model accuracy must meet minimum threshold",
                metric_name="accuracy",
                threshold=0.85,
                comparison="above",
                is_blocking=True,
            ),
            GatingCriteria(
                name="precision_gate",
                description="Model precision must meet minimum threshold",
                metric_name="precision",
                threshold=0.80,
                comparison="above",
                is_blocking=True,
            ),
            GatingCriteria(
                name="recall_gate",
                description="Model recall must meet minimum threshold",
                metric_name="recall",
                threshold=0.80,
                comparison="above",
                is_blocking=True,
            ),
            GatingCriteria(
                name="latency_gate",
                description="P95 latency must be below threshold",
                metric_name="latency_p95_ms",
                threshold=100.0,
                comparison="below",
                is_blocking=False,
            ),
            GatingCriteria(
                name="error_rate_gate",
                description="Error rate must be below threshold",
                metric_name="error_rate",
                threshold=0.01,
                comparison="below",
                is_blocking=True,
            ),
        ]

    def create_deployment(
        self,
        model_id: str,
        model_version: str,
        target_environment: str = "production",
        strategy: DeploymentStrategy = DeploymentStrategy.CANARY,
        canary_config: Optional[CanaryConfig] = None,
        rollback_policy: Optional[RollbackPolicy] = None,
        gating_criteria: Optional[List[GatingCriteria]] = None,
        previous_version: Optional[str] = None,
        created_by: str = "system",
    ) -> Deployment:
        """
        Create a new deployment.

        Args:
            model_id: Model being deployed
            model_version: Version being deployed
            target_environment: Target environment
            strategy: Deployment strategy
            canary_config: Canary configuration
            rollback_policy: Rollback policy
            gating_criteria: Gating criteria
            previous_version: Previous version (for rollback)
            created_by: User creating deployment

        Returns:
            Deployment record
        """
        import uuid

        deployment_id = f"DEPLOY-{uuid.uuid4().hex[:8].upper()}"

        deployment = Deployment(
            deployment_id=deployment_id,
            model_id=model_id,
            model_version=model_version,
            target_environment=target_environment,
            strategy=strategy,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            canary_config=canary_config or CanaryConfig(),
            rollback_policy=rollback_policy or RollbackPolicy(),
            gating_criteria=gating_criteria or self._default_gates,
            previous_version=previous_version,
        )

        deployment.add_event("created", {
            "model_id": model_id,
            "version": model_version,
            "strategy": strategy.value,
        }, created_by)

        self._deployments[deployment_id] = deployment

        self._log_action("create_deployment", {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "version": model_version,
        })

        logger.info(
            f"Created deployment {deployment_id} for "
            f"{model_id}:{model_version}"
        )
        return deployment

    def start_deployment(
        self,
        deployment_id: str,
        started_by: str = "system",
    ) -> Deployment:
        """
        Start a deployment.

        Args:
            deployment_id: Deployment to start
            started_by: User starting deployment

        Returns:
            Updated Deployment
        """
        if deployment_id not in self._deployments:
            raise KeyError(f"Deployment not found: {deployment_id}")

        deployment = self._deployments[deployment_id]

        if deployment.status != "pending":
            raise ValueError(
                f"Cannot start deployment in status: {deployment.status}"
            )

        deployment.status = "deploying"
        deployment.started_at = datetime.now(timezone.utc)

        if deployment.strategy == DeploymentStrategy.CANARY:
            deployment.current_traffic_pct = deployment.canary_config.initial_traffic_pct
            deployment.status = "canary"

        deployment.add_event("started", {
            "traffic_pct": deployment.current_traffic_pct,
        }, started_by)

        # Track active deployment
        self._active_deployments[deployment.target_environment] = deployment_id

        self._log_action("start_deployment", {
            "deployment_id": deployment_id,
            "status": deployment.status,
        })

        logger.info(f"Started deployment {deployment_id}")
        return deployment

    def evaluate_gates(
        self,
        deployment_id: str,
        metrics: Dict[str, float],
    ) -> List[GateResult]:
        """
        Evaluate gating criteria.

        Args:
            deployment_id: Deployment to evaluate
            metrics: Current metric values

        Returns:
            List of GateResults
        """
        if deployment_id not in self._deployments:
            raise KeyError(f"Deployment not found: {deployment_id}")

        deployment = self._deployments[deployment_id]
        results = []

        for criteria in deployment.gating_criteria:
            value = metrics.get(criteria.metric_name)
            sample_count = metrics.get("sample_count", 0)

            if value is None:
                result = GateResult(
                    criteria=criteria,
                    status=GateStatus.PENDING,
                    evaluated_at=datetime.now(timezone.utc),
                    message=f"Metric {criteria.metric_name} not available",
                )
            elif sample_count < criteria.min_samples:
                result = GateResult(
                    criteria=criteria,
                    status=GateStatus.PENDING,
                    evaluated_at=datetime.now(timezone.utc),
                    metric_value=value,
                    sample_count=sample_count,
                    message=f"Insufficient samples: {sample_count}/{criteria.min_samples}",
                )
            else:
                status = criteria.evaluate(value)
                result = GateResult(
                    criteria=criteria,
                    status=status,
                    evaluated_at=datetime.now(timezone.utc),
                    metric_value=value,
                    sample_count=sample_count,
                    message=(
                        f"{'Passed' if status == GateStatus.PASSED else 'Failed'}: "
                        f"{value:.3f} vs threshold {criteria.threshold}"
                    ),
                )

            results.append(result)

        deployment.gate_results = results

        # Log gate evaluation
        passed = sum(1 for r in results if r.status == GateStatus.PASSED)
        failed = sum(1 for r in results if r.status == GateStatus.FAILED)
        pending = sum(1 for r in results if r.status == GateStatus.PENDING)

        deployment.add_event("gates_evaluated", {
            "passed": passed,
            "failed": failed,
            "pending": pending,
        })

        return results

    def progress_canary(
        self,
        deployment_id: str,
        progressed_by: str = "system",
    ) -> Deployment:
        """
        Progress canary deployment to more traffic.

        Args:
            deployment_id: Deployment to progress
            progressed_by: User progressing

        Returns:
            Updated Deployment
        """
        if deployment_id not in self._deployments:
            raise KeyError(f"Deployment not found: {deployment_id}")

        deployment = self._deployments[deployment_id]

        if deployment.status != "canary":
            raise ValueError(
                f"Cannot progress deployment in status: {deployment.status}"
            )

        # Check gates
        failed_blocking = [
            r for r in deployment.gate_results
            if r.status == GateStatus.FAILED and r.criteria.is_blocking
        ]

        if failed_blocking:
            raise ValueError(
                f"Cannot progress: {len(failed_blocking)} blocking gates failed"
            )

        # Increase traffic
        old_traffic = deployment.current_traffic_pct
        new_traffic = min(
            100.0,
            old_traffic + deployment.canary_config.increment_pct
        )
        deployment.current_traffic_pct = new_traffic

        deployment.add_event("canary_progressed", {
            "old_traffic_pct": old_traffic,
            "new_traffic_pct": new_traffic,
        }, progressed_by)

        # Check if deployment is complete
        if new_traffic >= 100.0:
            deployment.status = "deployed"
            deployment.completed_at = datetime.now(timezone.utc)
            deployment.add_event("completed", {
                "final_traffic_pct": 100.0,
            }, progressed_by)
            logger.info(f"Deployment {deployment_id} completed successfully")

        self._log_action("progress_canary", {
            "deployment_id": deployment_id,
            "new_traffic_pct": new_traffic,
        })

        return deployment

    def rollback(
        self,
        deployment_id: str,
        reason: RollbackReason,
        rolled_back_by: str = "system",
        notes: str = "",
    ) -> Deployment:
        """
        Rollback a deployment.

        Args:
            deployment_id: Deployment to rollback
            reason: Reason for rollback
            rolled_back_by: User rolling back
            notes: Additional notes

        Returns:
            Updated Deployment
        """
        if deployment_id not in self._deployments:
            raise KeyError(f"Deployment not found: {deployment_id}")

        deployment = self._deployments[deployment_id]

        if deployment.status in ["pending", "rolled_back"]:
            raise ValueError(
                f"Cannot rollback deployment in status: {deployment.status}"
            )

        deployment.status = "rolled_back"
        deployment.rollback_reason = reason
        deployment.current_traffic_pct = 0.0
        deployment.completed_at = datetime.now(timezone.utc)

        deployment.add_event("rolled_back", {
            "reason": reason.value,
            "notes": notes,
            "previous_version": deployment.previous_version,
        }, rolled_back_by)

        # Clear active deployment
        if self._active_deployments.get(deployment.target_environment) == deployment_id:
            del self._active_deployments[deployment.target_environment]

        self._log_action("rollback", {
            "deployment_id": deployment_id,
            "reason": reason.value,
        })

        logger.warning(
            f"Rolled back deployment {deployment_id}: {reason.value}"
        )
        return deployment

    def check_auto_rollback(
        self,
        deployment_id: str,
        error_rate: float,
        latency_p99_ms: float,
        accuracy: float,
        consecutive_failures: int,
    ) -> Optional[Deployment]:
        """
        Check if automatic rollback should trigger.

        Args:
            deployment_id: Deployment to check
            error_rate: Current error rate
            latency_p99_ms: P99 latency
            accuracy: Current accuracy
            consecutive_failures: Consecutive failures

        Returns:
            Deployment if rolled back, None otherwise
        """
        if deployment_id not in self._deployments:
            return None

        deployment = self._deployments[deployment_id]

        if not deployment.rollback_policy or not deployment.rollback_policy.enabled:
            return None

        reason = deployment.rollback_policy.should_auto_rollback(
            error_rate=error_rate,
            latency_p99_ms=latency_p99_ms,
            accuracy=accuracy,
            consecutive_failures=consecutive_failures,
        )

        if reason:
            return self.rollback(
                deployment_id=deployment_id,
                reason=reason,
                rolled_back_by="auto_rollback",
            )

        return None

    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """Get deployment by ID."""
        return self._deployments.get(deployment_id)

    def get_active_deployment(
        self,
        environment: str,
    ) -> Optional[Deployment]:
        """Get active deployment for environment."""
        deployment_id = self._active_deployments.get(environment)
        if deployment_id:
            return self._deployments.get(deployment_id)
        return None

    def list_deployments(
        self,
        model_id: Optional[str] = None,
        status: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> List[Deployment]:
        """
        List deployments with optional filtering.

        Args:
            model_id: Filter by model
            status: Filter by status
            environment: Filter by environment

        Returns:
            List of deployments
        """
        deployments = list(self._deployments.values())

        if model_id:
            deployments = [d for d in deployments if d.model_id == model_id]
        if status:
            deployments = [d for d in deployments if d.status == status]
        if environment:
            deployments = [
                d for d in deployments if d.target_environment == environment
            ]

        return sorted(deployments, key=lambda d: d.created_at, reverse=True)

    def _log_action(self, action: str, details: Dict[str, Any]):
        """Log action to audit trail."""
        self._audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            **details,
        })

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return audit log."""
        return self._audit_log.copy()
