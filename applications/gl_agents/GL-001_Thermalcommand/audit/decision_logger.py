"""
GL-001 ThermalCommand: Decision Logger.

Logs all orchestration decisions with timestamps, rationale, and provenance
for regulatory compliance and audit trail requirements.

Implements:
- IEC 61511 decision traceability requirements
- EPA 40 CFR Part 75 data retention requirements
- SOX compliance for financial decisions
"""

import hashlib
import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# =============================================================================
# DECISION TYPES AND CATEGORIES
# =============================================================================


class DecisionType(str, Enum):
    """Types of orchestration decisions."""

    OPTIMIZATION = 'OPTIMIZATION'
    SAFETY = 'SAFETY'
    COMPLIANCE = 'COMPLIANCE'
    MAINTENANCE = 'MAINTENANCE'
    COORDINATION = 'COORDINATION'
    OVERRIDE = 'OVERRIDE'
    ALARM_RESPONSE = 'ALARM_RESPONSE'
    SETPOINT_CHANGE = 'SETPOINT_CHANGE'


class DecisionOutcome(str, Enum):
    """Outcome of a decision."""

    APPROVED = 'APPROVED'
    REJECTED = 'REJECTED'
    DEFERRED = 'DEFERRED'
    ESCALATED = 'ESCALATED'
    AUTO_EXECUTED = 'AUTO_EXECUTED'


class DecisionPriority(str, Enum):
    """Priority level of decisions."""

    CRITICAL = 'CRITICAL'
    HIGH = 'HIGH'
    MEDIUM = 'MEDIUM'
    LOW = 'LOW'
    INFORMATIONAL = 'INFORMATIONAL'


# =============================================================================
# DECISION RECORD DATA STRUCTURES
# =============================================================================


@dataclass
class DecisionContext:
    """Context in which a decision was made."""

    equipment_ids: List[str]
    operating_mode: str
    current_load_pct: Decimal
    ambient_conditions: Dict[str, Any]
    active_alarms: List[str]
    regulatory_constraints: List[str]


@dataclass
class DecisionRationale:
    """Rationale for the decision."""

    primary_reason: str
    supporting_factors: List[str]
    constraints_considered: List[str]
    alternatives_evaluated: List[Dict[str, Any]]
    risk_assessment: Optional[str] = None


@dataclass
class DecisionRecord:
    """Complete record of an orchestration decision."""

    # Identification
    decision_id: str
    timestamp: datetime
    decision_type: DecisionType
    priority: DecisionPriority

    # Decision details
    description: str
    context: DecisionContext
    rationale: DecisionRationale
    outcome: DecisionOutcome

    # Agent information
    source_agent: str
    target_agents: List[str]
    correlation_id: Optional[str] = None

    # Values
    input_values: Dict[str, Any] = field(default_factory=dict)
    output_values: Dict[str, Any] = field(default_factory=dict)
    setpoint_changes: List[Dict[str, Any]] = field(default_factory=list)

    # Provenance
    provenance_hash: str = ''
    parent_decision_id: Optional[str] = None
    child_decision_ids: List[str] = field(default_factory=list)

    # Metadata
    execution_time_ms: int = 0
    requires_human_review: bool = False
    review_status: Optional[str] = None

    def __post_init__(self):
        """Generate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._generate_provenance_hash()

    def _generate_provenance_hash(self) -> str:
        """Generate SHA-256 provenance hash for the decision."""
        # Serialize decision-critical fields
        hash_data = {
            'decision_id': self.decision_id,
            'timestamp': self.timestamp.isoformat(),
            'decision_type': self.decision_type.value,
            'description': self.description,
            'source_agent': self.source_agent,
            'outcome': self.outcome.value,
            'input_values': self._serialize_values(self.input_values),
            'output_values': self._serialize_values(self.output_values),
        }
        json_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    @staticmethod
    def _serialize_values(values: Dict[str, Any]) -> Dict[str, str]:
        """Convert Decimal values to strings for serialization."""
        result = {}
        for k, v in values.items():
            if isinstance(v, Decimal):
                result[k] = str(v)
            else:
                result[k] = str(v) if v is not None else None
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert enums to values
        data['decision_type'] = self.decision_type.value
        data['priority'] = self.priority.value
        data['outcome'] = self.outcome.value
        # Convert datetime
        data['timestamp'] = self.timestamp.isoformat()
        # Convert Decimals in context
        if self.context:
            data['context']['current_load_pct'] = str(self.context.current_load_pct)
        return data


# =============================================================================
# DECISION LOGGER
# =============================================================================


class DecisionLogger:
    """
    Thread-safe decision logger for orchestration decisions.

    Provides:
    - Immutable decision records
    - SHA-256 provenance hashing
    - Chained decision tracking
    - Regulatory-compliant retention
    """

    def __init__(
        self,
        log_directory: Optional[Path] = None,
        retention_days: int = 1095,  # 3 years per EPA Part 75
        enable_file_logging: bool = True,
    ):
        """
        Initialize the decision logger.

        Args:
            log_directory: Directory for log files (default: ./audit_logs)
            retention_days: Days to retain records (default: 3 years)
            enable_file_logging: Whether to write to files
        """
        self._log_directory = log_directory or Path('./audit_logs/decisions')
        self._retention_days = retention_days
        self._enable_file_logging = enable_file_logging
        self._lock = threading.RLock()
        self._decision_cache: Dict[str, DecisionRecord] = {}
        self._chain_index: Dict[str, List[str]] = {}  # parent -> children

        if self._enable_file_logging:
            self._log_directory.mkdir(parents=True, exist_ok=True)

    def log_decision(
        self,
        decision_type: DecisionType,
        description: str,
        context: DecisionContext,
        rationale: DecisionRationale,
        outcome: DecisionOutcome,
        source_agent: str,
        target_agents: List[str],
        priority: DecisionPriority = DecisionPriority.MEDIUM,
        input_values: Optional[Dict[str, Any]] = None,
        output_values: Optional[Dict[str, Any]] = None,
        setpoint_changes: Optional[List[Dict[str, Any]]] = None,
        correlation_id: Optional[str] = None,
        parent_decision_id: Optional[str] = None,
        requires_human_review: bool = False,
    ) -> DecisionRecord:
        """
        Log an orchestration decision.

        Args:
            decision_type: Type of decision
            description: Human-readable description
            context: Decision context
            rationale: Rationale for the decision
            outcome: Decision outcome
            source_agent: Agent making the decision
            target_agents: Agents affected by decision
            priority: Decision priority
            input_values: Input values considered
            output_values: Output values produced
            setpoint_changes: Any setpoint modifications
            correlation_id: External correlation ID
            parent_decision_id: Parent decision (for chaining)
            requires_human_review: Flag for human review

        Returns:
            The logged decision record
        """
        with self._lock:
            decision_id = self._generate_decision_id()

            record = DecisionRecord(
                decision_id=decision_id,
                timestamp=datetime.now(timezone.utc),
                decision_type=decision_type,
                priority=priority,
                description=description,
                context=context,
                rationale=rationale,
                outcome=outcome,
                source_agent=source_agent,
                target_agents=target_agents,
                correlation_id=correlation_id,
                input_values=input_values or {},
                output_values=output_values or {},
                setpoint_changes=setpoint_changes or [],
                parent_decision_id=parent_decision_id,
                requires_human_review=requires_human_review,
            )

            # Update chain index
            if parent_decision_id:
                if parent_decision_id not in self._chain_index:
                    self._chain_index[parent_decision_id] = []
                self._chain_index[parent_decision_id].append(decision_id)

                # Update parent record if in cache
                if parent_decision_id in self._decision_cache:
                    self._decision_cache[parent_decision_id].child_decision_ids.append(
                        decision_id
                    )

            # Cache the record
            self._decision_cache[decision_id] = record

            # Write to file if enabled
            if self._enable_file_logging:
                self._write_to_file(record)

            return record

    def log_safety_decision(
        self,
        description: str,
        sif_id: str,
        trigger_value: Any,
        action_taken: str,
        source_agent: str,
    ) -> DecisionRecord:
        """
        Log a safety-related decision (SIF activation).

        Args:
            description: Description of safety event
            sif_id: Safety Instrumented Function ID
            trigger_value: Value that triggered the action
            action_taken: Action performed
            source_agent: Agent that detected/responded

        Returns:
            The logged decision record
        """
        context = DecisionContext(
            equipment_ids=[sif_id],
            operating_mode='SAFETY_RESPONSE',
            current_load_pct=Decimal('0'),
            ambient_conditions={},
            active_alarms=[sif_id],
            regulatory_constraints=['IEC 61511', 'NFPA 85'],
        )

        rationale = DecisionRationale(
            primary_reason=f'SIF {sif_id} activation',
            supporting_factors=[f'Trigger value: {trigger_value}'],
            constraints_considered=['Safety interlock logic'],
            alternatives_evaluated=[],
            risk_assessment='Immediate safety response required',
        )

        return self.log_decision(
            decision_type=DecisionType.SAFETY,
            description=description,
            context=context,
            rationale=rationale,
            outcome=DecisionOutcome.AUTO_EXECUTED,
            source_agent=source_agent,
            target_agents=['BMS', 'DCS'],
            priority=DecisionPriority.CRITICAL,
            input_values={'trigger_value': trigger_value, 'sif_id': sif_id},
            output_values={'action': action_taken},
            requires_human_review=True,
        )

    def log_optimization_decision(
        self,
        description: str,
        optimization_target: str,
        before_value: Any,
        after_value: Any,
        improvement_pct: Decimal,
        source_agent: str,
        target_agents: List[str],
    ) -> DecisionRecord:
        """
        Log an optimization decision.

        Args:
            description: Description of optimization
            optimization_target: What was optimized
            before_value: Value before optimization
            after_value: Value after optimization
            improvement_pct: Improvement percentage
            source_agent: Agent performing optimization
            target_agents: Affected agents

        Returns:
            The logged decision record
        """
        context = DecisionContext(
            equipment_ids=[optimization_target],
            operating_mode='OPTIMIZATION',
            current_load_pct=Decimal('100'),
            ambient_conditions={},
            active_alarms=[],
            regulatory_constraints=[],
        )

        rationale = DecisionRationale(
            primary_reason=f'Optimize {optimization_target}',
            supporting_factors=[f'{improvement_pct}% improvement expected'],
            constraints_considered=['Operating limits', 'Equipment capacity'],
            alternatives_evaluated=[
                {'option': 'baseline', 'value': str(before_value)},
                {'option': 'optimized', 'value': str(after_value)},
            ],
        )

        return self.log_decision(
            decision_type=DecisionType.OPTIMIZATION,
            description=description,
            context=context,
            rationale=rationale,
            outcome=DecisionOutcome.APPROVED,
            source_agent=source_agent,
            target_agents=target_agents,
            priority=DecisionPriority.MEDIUM,
            input_values={'before': before_value},
            output_values={'after': after_value, 'improvement_pct': improvement_pct},
        )

    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """Retrieve a decision record by ID."""
        with self._lock:
            return self._decision_cache.get(decision_id)

    def get_decision_chain(self, root_decision_id: str) -> List[DecisionRecord]:
        """Get all decisions in a chain starting from root."""
        with self._lock:
            chain = []
            to_process = [root_decision_id]

            while to_process:
                current_id = to_process.pop(0)
                if current_id in self._decision_cache:
                    chain.append(self._decision_cache[current_id])
                if current_id in self._chain_index:
                    to_process.extend(self._chain_index[current_id])

            return chain

    def query_decisions(
        self,
        decision_type: Optional[DecisionType] = None,
        source_agent: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        priority: Optional[DecisionPriority] = None,
        limit: int = 100,
    ) -> List[DecisionRecord]:
        """Query decisions with filters."""
        with self._lock:
            results = []

            for record in self._decision_cache.values():
                if decision_type and record.decision_type != decision_type:
                    continue
                if source_agent and record.source_agent != source_agent:
                    continue
                if start_time and record.timestamp < start_time:
                    continue
                if end_time and record.timestamp > end_time:
                    continue
                if priority and record.priority != priority:
                    continue

                results.append(record)

                if len(results) >= limit:
                    break

            return sorted(results, key=lambda x: x.timestamp, reverse=True)

    def verify_provenance(self, decision_id: str) -> bool:
        """Verify the provenance hash of a decision record."""
        with self._lock:
            record = self._decision_cache.get(decision_id)
            if not record:
                return False

            # Recalculate hash and compare
            expected_hash = record._generate_provenance_hash()
            return record.provenance_hash == expected_hash

    def export_audit_report(
        self,
        start_time: datetime,
        end_time: datetime,
        include_safety: bool = True,
        include_compliance: bool = True,
    ) -> Dict[str, Any]:
        """Export decisions for audit report."""
        decisions = self.query_decisions(
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )

        report = {
            'report_generated': datetime.now(timezone.utc).isoformat(),
            'period_start': start_time.isoformat(),
            'period_end': end_time.isoformat(),
            'total_decisions': len(decisions),
            'by_type': {},
            'by_priority': {},
            'by_outcome': {},
            'decisions': [],
        }

        for record in decisions:
            # Count by type
            type_key = record.decision_type.value
            report['by_type'][type_key] = report['by_type'].get(type_key, 0) + 1

            # Count by priority
            priority_key = record.priority.value
            report['by_priority'][priority_key] = (
                report['by_priority'].get(priority_key, 0) + 1
            )

            # Count by outcome
            outcome_key = record.outcome.value
            report['by_outcome'][outcome_key] = (
                report['by_outcome'].get(outcome_key, 0) + 1
            )

            # Include record if matches filter
            include = True
            if not include_safety and record.decision_type == DecisionType.SAFETY:
                include = False
            if not include_compliance and record.decision_type == DecisionType.COMPLIANCE:
                include = False

            if include:
                report['decisions'].append(record.to_dict())

        return report

    def _generate_decision_id(self) -> str:
        """Generate unique decision ID."""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
        unique = uuid.uuid4().hex[:8]
        return f'DEC-{timestamp}-{unique}'

    def _write_to_file(self, record: DecisionRecord) -> None:
        """Write decision record to file."""
        date_str = record.timestamp.strftime('%Y-%m-%d')
        file_path = self._log_directory / f'decisions_{date_str}.jsonl'

        with open(file_path, 'a') as f:
            json.dump(record.to_dict(), f, default=str)
            f.write('\n')


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_decision_logger: Optional[DecisionLogger] = None
_logger_lock = threading.Lock()


def get_decision_logger() -> DecisionLogger:
    """Get the singleton decision logger instance."""
    global _decision_logger
    with _logger_lock:
        if _decision_logger is None:
            _decision_logger = DecisionLogger()
        return _decision_logger


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def log_decision(
    decision_type: DecisionType,
    description: str,
    source_agent: str,
    **kwargs,
) -> DecisionRecord:
    """Convenience function to log a decision."""
    logger = get_decision_logger()

    # Create minimal context if not provided
    if 'context' not in kwargs:
        kwargs['context'] = DecisionContext(
            equipment_ids=[],
            operating_mode='NORMAL',
            current_load_pct=Decimal('0'),
            ambient_conditions={},
            active_alarms=[],
            regulatory_constraints=[],
        )

    # Create minimal rationale if not provided
    if 'rationale' not in kwargs:
        kwargs['rationale'] = DecisionRationale(
            primary_reason=description,
            supporting_factors=[],
            constraints_considered=[],
            alternatives_evaluated=[],
        )

    # Default outcome if not provided
    if 'outcome' not in kwargs:
        kwargs['outcome'] = DecisionOutcome.APPROVED

    # Default target agents if not provided
    if 'target_agents' not in kwargs:
        kwargs['target_agents'] = []

    return logger.log_decision(
        decision_type=decision_type,
        description=description,
        source_agent=source_agent,
        **kwargs,
    )


if __name__ == '__main__':
    # Example usage
    logger = DecisionLogger(enable_file_logging=False)

    # Log an optimization decision
    record = logger.log_optimization_decision(
        description='Reduced excess air from 15% to 12%',
        optimization_target='boiler_1',
        before_value=Decimal('15.0'),
        after_value=Decimal('12.0'),
        improvement_pct=Decimal('0.5'),
        source_agent='GL-004',
        target_agents=['GL-002', 'GL-005'],
    )

    print(f'Logged decision: {record.decision_id}')
    print(f'Provenance hash: {record.provenance_hash}')
    print(f'Verified: {logger.verify_provenance(record.decision_id)}')
