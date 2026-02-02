"""
GreenLang Framework - Enterprise AI Guardrails System
Runtime Safety, Security, and Compliance Controls

Based on:
- OWASP LLM Top 10 (2025)
- NIST AI RMF 1.0
- Anthropic Constitutional AI Principles
- Straiker AI Runtime Guardrails
- EU AI Act High-Risk System Requirements

This module provides multi-layered guardrails for AI agent safety
including prompt injection detection, action gating, and policy enforcement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, TypeVar, Union
import hashlib
import json
import logging
import re
import uuid
from collections import defaultdict


logger = logging.getLogger(__name__)

T = TypeVar('T')


class GuardrailType(Enum):
    """Types of guardrails."""
    INPUT = auto()       # Input validation
    OUTPUT = auto()      # Output filtering
    ACTION = auto()      # Action gating
    POLICY = auto()      # Policy enforcement
    SAFETY = auto()      # Safety constraints


class ViolationSeverity(Enum):
    """Severity of guardrail violations."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    BLOCKING = auto()


class ActionType(Enum):
    """Types of actions that can be gated."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    CONTROL = "control"
    OPTIMIZE = "optimize"
    RECOMMEND = "recommend"


@dataclass
class GuardrailViolation:
    """Record of a guardrail violation."""
    violation_id: str
    guardrail_name: str
    guardrail_type: GuardrailType
    severity: ViolationSeverity
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: str = ""
    remediation: str = ""

    def __post_init__(self):
        if not self.violation_id:
            self.violation_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "guardrail_name": self.guardrail_name,
            "guardrail_type": self.guardrail_type.name,
            "severity": self.severity.name,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "remediation": self.remediation
        }


@dataclass
class GuardrailResult:
    """Result of guardrail evaluation."""
    passed: bool
    violations: List[GuardrailViolation] = field(default_factory=list)
    transformed_input: Any = None
    execution_time_ms: float = 0.0

    @property
    def has_blocking_violation(self) -> bool:
        return any(v.severity == ViolationSeverity.BLOCKING for v in self.violations)

    @property
    def has_critical_violation(self) -> bool:
        return any(v.severity in (ViolationSeverity.CRITICAL, ViolationSeverity.BLOCKING)
                   for v in self.violations)


class Guardrail(ABC):
    """Abstract base class for guardrails."""

    def __init__(
        self,
        name: str,
        guardrail_type: GuardrailType,
        severity: ViolationSeverity = ViolationSeverity.ERROR,
        enabled: bool = True
    ):
        self.name = name
        self.guardrail_type = guardrail_type
        self.severity = severity
        self.enabled = enabled
        self._violation_count = 0

    @abstractmethod
    def evaluate(self, input_data: Any, context: Dict[str, Any]) -> GuardrailResult:
        """Evaluate the guardrail against input data."""
        pass

    def create_violation(
        self,
        message: str,
        context: Dict[str, Any] = None,
        remediation: str = ""
    ) -> GuardrailViolation:
        """Create a violation record."""
        self._violation_count += 1
        return GuardrailViolation(
            violation_id=str(uuid.uuid4()),
            guardrail_name=self.name,
            guardrail_type=self.guardrail_type,
            severity=self.severity,
            message=message,
            context=context or {},
            remediation=remediation
        )


# ============================================================================
# PROMPT INJECTION DETECTION (OWASP LLM01)
# ============================================================================

class PromptInjectionGuardrail(Guardrail):
    """
    Detects prompt injection attempts.

    OWASP LLM01: Prompt Injection is the #1 security risk
    for LLM applications in 2025.
    """

    # Common injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|above)\s+(instructions?|prompts?)",
        r"disregard\s+(your|all|previous)\s+(rules?|instructions?)",
        r"you\s+are\s+now\s+(a|an|the)\s+\w+",
        r"pretend\s+(to\s+be|you\s+are)",
        r"act\s+as\s+(if|though)",
        r"new\s+instructions?:?\s*",
        r"override\s+(previous|system)",
        r"forget\s+(everything|all|your)",
        r"from\s+now\s+on\s+you\s+(are|will)",
        r"system\s*:\s*",
        r"\[system\]",
        r"<\|system\|>",
        r"###\s*instruction",
        r"```\s*(system|admin)",
    ]

    def __init__(self, custom_patterns: Optional[List[str]] = None):
        super().__init__(
            name="PromptInjectionDetector",
            guardrail_type=GuardrailType.INPUT,
            severity=ViolationSeverity.BLOCKING
        )
        patterns = self.INJECTION_PATTERNS + (custom_patterns or [])
        self._patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def evaluate(self, input_data: Any, context: Dict[str, Any]) -> GuardrailResult:
        if not self.enabled:
            return GuardrailResult(passed=True)

        text = str(input_data).lower() if input_data else ""
        violations = []

        for pattern in self._patterns:
            if pattern.search(text):
                violations.append(self.create_violation(
                    message=f"Potential prompt injection detected: pattern '{pattern.pattern}'",
                    context={"input_preview": text[:200]},
                    remediation="Sanitize user input before processing"
                ))

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations
        )


# ============================================================================
# DATA LEAKAGE PREVENTION (OWASP LLM06)
# ============================================================================

class DataLeakageGuardrail(Guardrail):
    """
    Prevents sensitive data leakage in outputs.

    Detects PII, secrets, and confidential information.
    """

    # Common sensitive patterns
    SENSITIVE_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "api_key": r"(?:api[_-]?key|apikey)[\"']?\s*[:=]\s*[\"']?[\w-]{20,}",
        "password": r"(?:password|passwd|pwd)[\"']?\s*[:=]\s*[\"'][^\"']+[\"']",
        "aws_key": r"(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}",
        "private_key": r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
    }

    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        super().__init__(
            name="DataLeakagePrevention",
            guardrail_type=GuardrailType.OUTPUT,
            severity=ViolationSeverity.CRITICAL
        )
        patterns = {**self.SENSITIVE_PATTERNS, **(custom_patterns or {})}
        self._patterns = {name: re.compile(p, re.IGNORECASE) for name, p in patterns.items()}

    def evaluate(self, input_data: Any, context: Dict[str, Any]) -> GuardrailResult:
        if not self.enabled:
            return GuardrailResult(passed=True)

        text = str(input_data) if input_data else ""
        violations = []

        for name, pattern in self._patterns.items():
            matches = pattern.findall(text)
            if matches:
                violations.append(self.create_violation(
                    message=f"Sensitive data detected: {name} ({len(matches)} occurrences)",
                    context={"pattern_name": name, "match_count": len(matches)},
                    remediation=f"Redact or remove {name} from output"
                ))

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations
        )

    def redact(self, text: str) -> str:
        """Redact sensitive data from text."""
        for name, pattern in self._patterns.items():
            text = pattern.sub(f"[REDACTED_{name.upper()}]", text)
        return text


# ============================================================================
# ACTION GATE (Velocity and Authorization Control)
# ============================================================================

class ActionGate(Guardrail):
    """
    Gates actions based on authorization, velocity limits, and safety constraints.

    Implements the principle of least privilege for agent actions.
    """

    def __init__(
        self,
        allowed_actions: Optional[Set[ActionType]] = None,
        max_actions_per_minute: int = 60,
        require_confirmation_for: Optional[Set[ActionType]] = None
    ):
        super().__init__(
            name="ActionGate",
            guardrail_type=GuardrailType.ACTION,
            severity=ViolationSeverity.BLOCKING
        )
        self._allowed_actions = allowed_actions or {ActionType.READ, ActionType.RECOMMEND}
        self._max_actions_per_minute = max_actions_per_minute
        self._require_confirmation = require_confirmation_for or {ActionType.WRITE, ActionType.CONTROL}

        # Rate limiting state
        self._action_history: List[datetime] = []
        self._pending_confirmations: Dict[str, Dict[str, Any]] = {}

    def evaluate(self, input_data: Any, context: Dict[str, Any]) -> GuardrailResult:
        if not self.enabled:
            return GuardrailResult(passed=True)

        action_type = context.get("action_type")
        if not action_type:
            return GuardrailResult(passed=True)

        if isinstance(action_type, str):
            action_type = ActionType(action_type)

        violations = []

        # Check authorization
        if action_type not in self._allowed_actions:
            violations.append(self.create_violation(
                message=f"Action type not allowed: {action_type.value}",
                context={"action_type": action_type.value, "allowed": [a.value for a in self._allowed_actions]},
                remediation="Request authorization for this action type"
            ))

        # Check velocity limits
        now = datetime.now(timezone.utc)
        one_minute_ago = now - timedelta(minutes=1)
        self._action_history = [t for t in self._action_history if t > one_minute_ago]

        if len(self._action_history) >= self._max_actions_per_minute:
            violations.append(self.create_violation(
                message=f"Rate limit exceeded: {len(self._action_history)} actions in last minute",
                context={"current_rate": len(self._action_history), "limit": self._max_actions_per_minute},
                remediation="Wait before performing more actions"
            ))
        else:
            self._action_history.append(now)

        # Check confirmation requirement
        if action_type in self._require_confirmation:
            confirmation_id = context.get("confirmation_id")
            if not confirmation_id or confirmation_id not in self._pending_confirmations:
                violations.append(self.create_violation(
                    message=f"Confirmation required for action: {action_type.value}",
                    context={"action_type": action_type.value},
                    remediation="Request human confirmation before proceeding",
                    severity=ViolationSeverity.WARNING
                ))

        return GuardrailResult(
            passed=not any(v.severity == ViolationSeverity.BLOCKING for v in violations),
            violations=violations
        )

    def request_confirmation(self, action_type: ActionType, description: str) -> str:
        """Request confirmation for an action. Returns confirmation ID."""
        confirmation_id = str(uuid.uuid4())
        self._pending_confirmations[confirmation_id] = {
            "action_type": action_type,
            "description": description,
            "requested_at": datetime.now(timezone.utc)
        }
        return confirmation_id

    def confirm(self, confirmation_id: str) -> bool:
        """Confirm a pending action."""
        if confirmation_id in self._pending_confirmations:
            del self._pending_confirmations[confirmation_id]
            return True
        return False


# ============================================================================
# SAFETY ENVELOPE (Physical Constraints)
# ============================================================================

@dataclass
class PhysicalConstraint:
    """Physical constraint for industrial safety."""
    name: str
    parameter: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: str = ""
    severity: ViolationSeverity = ViolationSeverity.CRITICAL

    def check(self, value: float) -> Optional[str]:
        """Check if value violates constraint. Returns error message or None."""
        if self.min_value is not None and value < self.min_value:
            return f"{self.name}: {value} {self.unit} is below minimum {self.min_value} {self.unit}"
        if self.max_value is not None and value > self.max_value:
            return f"{self.name}: {value} {self.unit} exceeds maximum {self.max_value} {self.unit}"
        return None


class SafetyEnvelopeGuardrail(Guardrail):
    """
    Enforces physical safety constraints for industrial applications.

    Based on IEC 61511 and NFPA 85/86 safety requirements.
    """

    # Default industrial safety constraints
    DEFAULT_CONSTRAINTS = [
        PhysicalConstraint("Temperature", "temperature", min_value=-273.15, max_value=2000, unit="°C"),
        PhysicalConstraint("Pressure", "pressure", min_value=0, max_value=500, unit="bar"),
        PhysicalConstraint("Flow Rate", "flow_rate", min_value=0, unit="kg/s"),
        PhysicalConstraint("Efficiency", "efficiency", min_value=0, max_value=100, unit="%"),
        PhysicalConstraint("Excess Air", "excess_air", min_value=0, max_value=200, unit="%"),
        PhysicalConstraint("O2 Level", "o2_level", min_value=0, max_value=21, unit="%"),
        PhysicalConstraint("CO Level", "co_level", max_value=1000, unit="ppm"),
        PhysicalConstraint("NOx Level", "nox_level", max_value=500, unit="ppm"),
    ]

    def __init__(self, custom_constraints: Optional[List[PhysicalConstraint]] = None):
        super().__init__(
            name="SafetyEnvelope",
            guardrail_type=GuardrailType.SAFETY,
            severity=ViolationSeverity.CRITICAL
        )
        self._constraints = {c.parameter: c for c in self.DEFAULT_CONSTRAINTS}
        if custom_constraints:
            for c in custom_constraints:
                self._constraints[c.parameter] = c

    def evaluate(self, input_data: Any, context: Dict[str, Any]) -> GuardrailResult:
        if not self.enabled:
            return GuardrailResult(passed=True)

        violations = []

        # Check input data against constraints
        data = input_data if isinstance(input_data, dict) else {}

        for param, constraint in self._constraints.items():
            if param in data:
                value = data[param]
                if isinstance(value, (int, float)):
                    error = constraint.check(value)
                    if error:
                        violations.append(self.create_violation(
                            message=error,
                            context={"parameter": param, "value": value, "constraint": constraint.name},
                            remediation=f"Adjust {param} to within safe limits"
                        ))

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations
        )

    def add_constraint(self, constraint: PhysicalConstraint) -> None:
        """Add or update a constraint."""
        self._constraints[constraint.parameter] = constraint

    def get_safe_range(self, parameter: str) -> Optional[tuple[float, float]]:
        """Get the safe range for a parameter."""
        if parameter in self._constraints:
            c = self._constraints[parameter]
            return (c.min_value, c.max_value)
        return None


# ============================================================================
# POLICY ENFORCEMENT
# ============================================================================

@dataclass
class Policy:
    """Policy rule for agent behavior."""
    policy_id: str
    name: str
    description: str
    condition: Callable[[Any, Dict[str, Any]], bool]
    message: str
    severity: ViolationSeverity = ViolationSeverity.ERROR
    enabled: bool = True


class PolicyEnforcementGuardrail(Guardrail):
    """
    Enforces organizational policies on agent behavior.

    Supports custom policy rules for compliance and governance.
    """

    def __init__(self, policies: Optional[List[Policy]] = None):
        super().__init__(
            name="PolicyEnforcement",
            guardrail_type=GuardrailType.POLICY,
            severity=ViolationSeverity.ERROR
        )
        self._policies: Dict[str, Policy] = {}
        if policies:
            for p in policies:
                self._policies[p.policy_id] = p

    def add_policy(self, policy: Policy) -> None:
        """Add a policy."""
        self._policies[policy.policy_id] = policy

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            return True
        return False

    def evaluate(self, input_data: Any, context: Dict[str, Any]) -> GuardrailResult:
        if not self.enabled:
            return GuardrailResult(passed=True)

        violations = []

        for policy in self._policies.values():
            if not policy.enabled:
                continue
            try:
                if policy.condition(input_data, context):
                    violations.append(GuardrailViolation(
                        violation_id=str(uuid.uuid4()),
                        guardrail_name=self.name,
                        guardrail_type=self.guardrail_type,
                        severity=policy.severity,
                        message=policy.message,
                        context={"policy_id": policy.policy_id, "policy_name": policy.name}
                    ))
            except Exception as e:
                logger.warning(f"Policy evaluation error: {policy.policy_id}: {e}")

        return GuardrailResult(
            passed=not any(v.severity in (ViolationSeverity.CRITICAL, ViolationSeverity.BLOCKING)
                          for v in violations),
            violations=violations
        )


# ============================================================================
# GUARDRAIL ORCHESTRATOR
# ============================================================================

class GuardrailOrchestrator:
    """
    Orchestrates multiple guardrails for comprehensive protection.

    Provides a single entry point for evaluating all guardrails.
    """

    def __init__(self):
        self._input_guardrails: List[Guardrail] = []
        self._output_guardrails: List[Guardrail] = []
        self._action_guardrails: List[Guardrail] = []
        self._violation_history: List[GuardrailViolation] = []

    def add_input_guardrail(self, guardrail: Guardrail) -> 'GuardrailOrchestrator':
        """Add an input guardrail."""
        self._input_guardrails.append(guardrail)
        return self

    def add_output_guardrail(self, guardrail: Guardrail) -> 'GuardrailOrchestrator':
        """Add an output guardrail."""
        self._output_guardrails.append(guardrail)
        return self

    def add_action_guardrail(self, guardrail: Guardrail) -> 'GuardrailOrchestrator':
        """Add an action guardrail."""
        self._action_guardrails.append(guardrail)
        return self

    def check_input(self, input_data: Any, context: Dict[str, Any] = None) -> GuardrailResult:
        """Check input against all input guardrails."""
        import time
        start = time.time()

        all_violations = []
        for guardrail in self._input_guardrails:
            result = guardrail.evaluate(input_data, context or {})
            all_violations.extend(result.violations)

        self._violation_history.extend(all_violations)

        return GuardrailResult(
            passed=not any(v.severity == ViolationSeverity.BLOCKING for v in all_violations),
            violations=all_violations,
            execution_time_ms=(time.time() - start) * 1000
        )

    def check_output(self, output_data: Any, context: Dict[str, Any] = None) -> GuardrailResult:
        """Check output against all output guardrails."""
        import time
        start = time.time()

        all_violations = []
        for guardrail in self._output_guardrails:
            result = guardrail.evaluate(output_data, context or {})
            all_violations.extend(result.violations)

        self._violation_history.extend(all_violations)

        return GuardrailResult(
            passed=not any(v.severity == ViolationSeverity.BLOCKING for v in all_violations),
            violations=all_violations,
            execution_time_ms=(time.time() - start) * 1000
        )

    def check_action(self, action_data: Any, context: Dict[str, Any] = None) -> GuardrailResult:
        """Check action against all action guardrails."""
        import time
        start = time.time()

        all_violations = []
        for guardrail in self._action_guardrails:
            result = guardrail.evaluate(action_data, context or {})
            all_violations.extend(result.violations)

        self._violation_history.extend(all_violations)

        return GuardrailResult(
            passed=not any(v.severity == ViolationSeverity.BLOCKING for v in all_violations),
            violations=all_violations,
            execution_time_ms=(time.time() - start) * 1000
        )

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of all violations."""
        severity_counts = defaultdict(int)
        guardrail_counts = defaultdict(int)

        for v in self._violation_history:
            severity_counts[v.severity.name] += 1
            guardrail_counts[v.guardrail_name] += 1

        return {
            "total_violations": len(self._violation_history),
            "by_severity": dict(severity_counts),
            "by_guardrail": dict(guardrail_counts),
            "recent_violations": [v.to_dict() for v in self._violation_history[-10:]]
        }


# ============================================================================
# DEFAULT ORCHESTRATOR WITH COMMON GUARDRAILS
# ============================================================================

def create_default_orchestrator() -> GuardrailOrchestrator:
    """Create an orchestrator with default guardrails for industrial AI."""
    orchestrator = GuardrailOrchestrator()

    # Input guardrails
    orchestrator.add_input_guardrail(PromptInjectionGuardrail())

    # Output guardrails
    orchestrator.add_output_guardrail(DataLeakageGuardrail())

    # Action guardrails
    orchestrator.add_action_guardrail(ActionGate(
        allowed_actions={ActionType.READ, ActionType.RECOMMEND, ActionType.OPTIMIZE},
        max_actions_per_minute=100
    ))
    orchestrator.add_action_guardrail(SafetyEnvelopeGuardrail())

    return orchestrator


GREENLANG_GUARDRAILS = create_default_orchestrator()
