# -*- coding: utf-8 -*-
"""
Decision Trace Module for GreenLang Agents
==========================================

Provides comprehensive decision audit trails for regulatory compliance,
capturing every decision point with inputs, outputs, and rationale.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from functools import wraps

from pydantic import BaseModel, Field


class DecisionOutcome(str, Enum):
    """Possible decision outcomes."""
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    DEFERRED = "deferred"
    MANUAL_REVIEW = "manual_review"
    ERROR = "error"


class DecisionConfidence(str, Enum):
    """Confidence levels for decisions."""
    HIGH = "high"  # >95% confidence
    MEDIUM = "medium"  # 80-95% confidence
    LOW = "low"  # <80% confidence
    UNCERTAIN = "uncertain"  # Requires human review


class RuleType(str, Enum):
    """Types of decision rules."""
    THRESHOLD = "threshold"
    BOOLEAN = "boolean"
    RANGE = "range"
    CLASSIFICATION = "classification"
    COMPOUND = "compound"
    REGULATORY = "regulatory"


@dataclass
class DecisionRule:
    """Single rule evaluated in a decision."""
    rule_id: str
    rule_name: str
    rule_type: RuleType
    condition: str  # Human-readable condition
    expected_value: Any
    actual_value: Any
    passed: bool
    weight: float = 1.0
    source: Optional[str] = None  # Standard/regulation reference

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.rule_name,
            "type": self.rule_type.value,
            "condition": self.condition,
            "expected": self.expected_value,
            "actual": self.actual_value,
            "passed": self.passed,
            "weight": self.weight,
            "source": self.source,
        }


@dataclass
class DecisionStep:
    """Single step in a decision process."""
    step_id: str
    step_number: int
    step_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Inputs and outputs
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    # Rules evaluated
    rules_evaluated: List[DecisionRule] = field(default_factory=list)

    # Decision at this step
    step_outcome: Optional[str] = None
    step_confidence: float = 1.0

    # Metadata
    duration_ms: float = 0.0
    notes: List[str] = field(default_factory=list)

    def add_rule(self, rule: DecisionRule) -> None:
        """Add an evaluated rule to this step."""
        self.rules_evaluated.append(rule)

    def rules_passed(self) -> int:
        """Count rules that passed."""
        return sum(1 for r in self.rules_evaluated if r.passed)

    def rules_failed(self) -> int:
        """Count rules that failed."""
        return sum(1 for r in self.rules_evaluated if not r.passed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "name": self.step_name,
            "timestamp": self.timestamp.isoformat(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "rules": [r.to_dict() for r in self.rules_evaluated],
            "outcome": self.step_outcome,
            "confidence": self.step_confidence,
            "duration_ms": self.duration_ms,
            "notes": self.notes,
        }


@dataclass
class DecisionNode:
    """Node in a decision tree structure."""
    node_id: str
    node_type: str  # "decision", "action", "condition", "terminal"
    label: str

    # For decision/condition nodes
    condition: Optional[str] = None
    threshold: Optional[float] = None

    # Tree structure
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)

    # Evaluation
    evaluated: bool = False
    evaluation_result: Optional[bool] = None
    actual_value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "type": self.node_type,
            "label": self.label,
            "condition": self.condition,
            "threshold": self.threshold,
            "parent_id": self.parent_id,
            "children": self.children,
            "evaluated": self.evaluated,
            "result": self.evaluation_result,
            "actual_value": self.actual_value,
        }


class DecisionTrace(BaseModel):
    """
    Complete audit trail for a decision process.

    Captures every step, rule, and input/output for regulatory compliance.
    """
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Decision identification
    decision_name: str
    decision_type: str = "general"
    version: str = "1.0"

    # Context
    context: Dict[str, Any] = Field(default_factory=dict)
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    # Steps
    steps: List[Dict[str, Any]] = Field(default_factory=list)

    # Final outcome
    outcome: DecisionOutcome = DecisionOutcome.DEFERRED
    outcome_value: Optional[Any] = None
    confidence: DecisionConfidence = DecisionConfidence.MEDIUM
    confidence_score: float = 0.0

    # Summary
    total_rules_evaluated: int = 0
    rules_passed: int = 0
    rules_failed: int = 0

    # Provenance
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    trace_hash: Optional[str] = None

    # Regulatory
    applicable_standards: List[str] = Field(default_factory=list)
    compliance_notes: List[str] = Field(default_factory=list)

    # Performance
    total_duration_ms: float = 0.0

    def add_step(self, step: DecisionStep) -> None:
        """Add a decision step to the trace."""
        self.steps.append(step.to_dict())
        self.total_rules_evaluated += len(step.rules_evaluated)
        self.rules_passed += step.rules_passed()
        self.rules_failed += step.rules_failed()
        self.total_duration_ms += step.duration_ms

    def finalize(
        self,
        outcome: DecisionOutcome,
        outcome_value: Any = None,
        confidence_score: float = 0.0,
    ) -> None:
        """Finalize the decision trace with outcome."""
        self.outcome = outcome
        self.outcome_value = outcome_value
        self.confidence_score = confidence_score

        # Determine confidence level
        if confidence_score >= 0.95:
            self.confidence = DecisionConfidence.HIGH
        elif confidence_score >= 0.80:
            self.confidence = DecisionConfidence.MEDIUM
        elif confidence_score >= 0.60:
            self.confidence = DecisionConfidence.LOW
        else:
            self.confidence = DecisionConfidence.UNCERTAIN

        # Calculate provenance hashes
        self._calculate_hashes()

    def _calculate_hashes(self) -> None:
        """Calculate provenance hashes for audit trail."""
        # Input hash from context
        context_str = json.dumps(self.context, sort_keys=True, default=str)
        self.input_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]

        # Output hash from outcome
        output_str = json.dumps({
            "outcome": self.outcome.value,
            "value": str(self.outcome_value),
            "confidence": self.confidence_score,
        }, sort_keys=True)
        self.output_hash = hashlib.sha256(output_str.encode()).hexdigest()[:16]

        # Complete trace hash
        trace_str = json.dumps({
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "steps": len(self.steps),
            "rules_passed": self.rules_passed,
            "rules_failed": self.rules_failed,
        }, sort_keys=True)
        self.trace_hash = hashlib.sha256(trace_str.encode()).hexdigest()[:16]

    def to_audit_report(self) -> str:
        """Generate human-readable audit report."""
        lines = [
            f"# Decision Audit Report",
            f"",
            f"**Trace ID**: {self.trace_id}",
            f"**Decision**: {self.decision_name}",
            f"**Timestamp**: {self.created_at.isoformat()}",
            f"",
            f"## Outcome",
            f"- **Result**: {self.outcome.value.upper()}",
            f"- **Confidence**: {self.confidence.value} ({self.confidence_score:.1%})",
            f"- **Value**: {self.outcome_value}",
            f"",
            f"## Summary",
            f"- Rules Evaluated: {self.total_rules_evaluated}",
            f"- Rules Passed: {self.rules_passed}",
            f"- Rules Failed: {self.rules_failed}",
            f"- Total Duration: {self.total_duration_ms:.2f} ms",
            f"",
            f"## Steps",
        ]

        for step in self.steps:
            lines.append(f"### Step {step['step_number']}: {step['name']}")
            lines.append(f"- Outcome: {step.get('outcome', 'N/A')}")
            if step.get('rules'):
                lines.append("- Rules:")
                for rule in step['rules']:
                    status = "✓" if rule['passed'] else "✗"
                    lines.append(f"  - {status} {rule['name']}: {rule['condition']}")

        if self.applicable_standards:
            lines.append("")
            lines.append("## Applicable Standards")
            for std in self.applicable_standards:
                lines.append(f"- {std}")

        lines.extend([
            "",
            "## Provenance",
            f"- Input Hash: {self.input_hash}",
            f"- Output Hash: {self.output_hash}",
            f"- Trace Hash: {self.trace_hash}",
        ])

        return "\n".join(lines)


class DecisionTracer:
    """
    Context manager and decorator for tracing decisions.

    Automatically captures inputs, outputs, and decision flow.
    """

    def __init__(
        self,
        decision_name: str,
        decision_type: str = "general",
        agent_id: Optional[str] = None,
    ):
        self.decision_name = decision_name
        self.decision_type = decision_type
        self.agent_id = agent_id
        self.trace: Optional[DecisionTrace] = None
        self._current_step: Optional[DecisionStep] = None
        self._step_counter = 0
        self._step_start_time: Optional[datetime] = None

    def __enter__(self) -> DecisionTracer:
        """Start tracing a decision."""
        self.trace = DecisionTrace(
            decision_name=self.decision_name,
            decision_type=self.decision_type,
            agent_id=self.agent_id,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Finish tracing."""
        if self._current_step is not None:
            self._finish_current_step()

        if exc_type is not None:
            # Error occurred
            self.trace.finalize(
                outcome=DecisionOutcome.ERROR,
                outcome_value=str(exc_val),
                confidence_score=0.0,
            )

    def set_context(self, context: Dict[str, Any]) -> None:
        """Set decision context."""
        if self.trace:
            self.trace.context = context

    def add_standard(self, standard: str) -> None:
        """Add applicable regulatory standard."""
        if self.trace:
            self.trace.applicable_standards.append(standard)

    def start_step(self, step_name: str, inputs: Dict[str, Any] = None) -> None:
        """Start a new decision step."""
        if self._current_step is not None:
            self._finish_current_step()

        self._step_counter += 1
        self._step_start_time = datetime.utcnow()

        self._current_step = DecisionStep(
            step_id=str(uuid.uuid4()),
            step_number=self._step_counter,
            step_name=step_name,
            inputs=inputs or {},
        )

    def evaluate_rule(
        self,
        rule_name: str,
        condition: str,
        expected: Any,
        actual: Any,
        passed: bool,
        rule_type: RuleType = RuleType.THRESHOLD,
        source: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        """Record evaluation of a rule."""
        if self._current_step is None:
            self.start_step("implicit_step")

        rule = DecisionRule(
            rule_id=str(uuid.uuid4()),
            rule_name=rule_name,
            rule_type=rule_type,
            condition=condition,
            expected_value=expected,
            actual_value=actual,
            passed=passed,
            weight=weight,
            source=source,
        )
        self._current_step.add_rule(rule)

    def finish_step(
        self,
        outputs: Dict[str, Any] = None,
        outcome: str = None,
        confidence: float = 1.0,
        notes: List[str] = None,
    ) -> None:
        """Finish current step with results."""
        if self._current_step is not None:
            self._current_step.outputs = outputs or {}
            self._current_step.step_outcome = outcome
            self._current_step.step_confidence = confidence
            self._current_step.notes = notes or []
            self._finish_current_step()

    def _finish_current_step(self) -> None:
        """Internal: finalize and add current step."""
        if self._current_step is not None and self._step_start_time is not None:
            duration = (datetime.utcnow() - self._step_start_time).total_seconds() * 1000
            self._current_step.duration_ms = duration
            self.trace.add_step(self._current_step)
            self._current_step = None
            self._step_start_time = None

    def finalize(
        self,
        outcome: DecisionOutcome,
        outcome_value: Any = None,
        confidence_score: float = None,
    ) -> DecisionTrace:
        """Finalize decision and return trace."""
        if self._current_step is not None:
            self._finish_current_step()

        # Calculate confidence if not provided
        if confidence_score is None:
            if self.trace.total_rules_evaluated > 0:
                confidence_score = self.trace.rules_passed / self.trace.total_rules_evaluated
            else:
                confidence_score = 1.0

        self.trace.finalize(outcome, outcome_value, confidence_score)
        return self.trace


# Type variable for decorated functions
T = TypeVar('T')


def trace_decision(
    decision_name: str,
    decision_type: str = "general",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically trace a decision function.

    Usage:
        @trace_decision("efficiency_calculation")
        def calculate_efficiency(inputs: dict) -> float:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with DecisionTracer(decision_name, decision_type) as tracer:
                tracer.set_context({"args": str(args), "kwargs": kwargs})
                tracer.start_step("execute", inputs=kwargs)

                try:
                    result = func(*args, **kwargs)
                    tracer.finish_step(outputs={"result": result})
                    tracer.finalize(
                        outcome=DecisionOutcome.APPROVED,
                        outcome_value=result,
                    )
                    return result
                except Exception as e:
                    tracer.finish_step(
                        outputs={"error": str(e)},
                        outcome="error",
                    )
                    raise

        return wrapper
    return decorator


class DecisionTree:
    """
    Traceable decision tree for rule-based decisions.

    Provides structured decision trees with full audit trail.
    """

    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, DecisionNode] = {}
        self.root_id: Optional[str] = None

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        condition: Optional[str] = None,
        threshold: Optional[float] = None,
        parent_id: Optional[str] = None,
    ) -> DecisionNode:
        """Add a node to the decision tree."""
        node = DecisionNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            condition=condition,
            threshold=threshold,
            parent_id=parent_id,
        )

        self.nodes[node_id] = node

        if parent_id is None:
            self.root_id = node_id
        elif parent_id in self.nodes:
            self.nodes[parent_id].children.append(node_id)

        return node

    def evaluate(
        self,
        inputs: Dict[str, Any],
        tracer: Optional[DecisionTracer] = None,
    ) -> Tuple[str, Any]:
        """
        Evaluate the decision tree with given inputs.

        Returns:
            Tuple of (terminal_node_id, result_value)
        """
        if self.root_id is None:
            raise ValueError("Decision tree has no root node")

        current_id = self.root_id
        path = []

        while current_id is not None:
            node = self.nodes[current_id]
            node.evaluated = True
            path.append(current_id)

            if node.node_type == "terminal":
                # Reached terminal node
                if tracer:
                    tracer.start_step(f"terminal_{node.node_id}")
                    tracer.finish_step(
                        outputs={"result": node.label},
                        outcome=node.label,
                    )
                return current_id, node.label

            elif node.node_type in ("decision", "condition"):
                # Evaluate condition
                if node.condition and node.threshold is not None:
                    # Simple threshold comparison
                    param = node.condition
                    actual = inputs.get(param, 0)
                    node.actual_value = actual
                    node.evaluation_result = actual >= node.threshold

                    if tracer:
                        tracer.start_step(f"evaluate_{node.node_id}")
                        tracer.evaluate_rule(
                            rule_name=node.label,
                            condition=f"{param} >= {node.threshold}",
                            expected=node.threshold,
                            actual=actual,
                            passed=node.evaluation_result,
                        )
                        tracer.finish_step(
                            outputs={"result": node.evaluation_result},
                            outcome="passed" if node.evaluation_result else "failed",
                        )

                    # Navigate to appropriate child
                    if len(node.children) >= 2:
                        current_id = node.children[0] if node.evaluation_result else node.children[1]
                    elif len(node.children) == 1:
                        current_id = node.children[0] if node.evaluation_result else None
                    else:
                        current_id = None
                else:
                    # No condition, go to first child
                    current_id = node.children[0] if node.children else None
            else:
                # Unknown node type, continue to first child
                current_id = node.children[0] if node.children else None

        return path[-1] if path else "", None

    def to_dict(self) -> Dict[str, Any]:
        """Export tree structure."""
        return {
            "name": self.name,
            "root_id": self.root_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }
