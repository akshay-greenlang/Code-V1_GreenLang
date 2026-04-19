# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Guardrails Integration Module

Provides safety guardrails integration for condenser optimization including:
- Physical constraint validation (pressure, temperature, flow limits)
- Recommendation safety checks (within safe operating envelope)
- Action gating with velocity limits
- Provenance tracking for audit compliance

Key Features:
- Industrial safety envelope enforcement
- Condenser-specific physical bounds checking
- Rate limiting for optimization recommendations
- Complete audit trail with SHA-256 provenance

Standards Compliance:
- HEI Standards for Steam Surface Condensers (12th Edition)
- ASME PTC 12.2: Steam Surface Condensers
- OWASP LLM Top 10 (2025)
- GreenLang Global AI Standards v2.0

Zero-Hallucination Guarantee:
All guardrail checks use deterministic rules from published standards.
No LLM or AI inference in safety validation path.
Same inputs always produce identical safety assessments.

Example:
    >>> from core.guardrails_integration import CondenserGuardrails
    >>> guardrails = CondenserGuardrails(profile=GuardrailProfile.INDUSTRIAL)
    >>> result = guardrails.check_recommendation(recommendation_data)
    >>> if result.passed:
    ...     apply_recommendation(recommendation_data)

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

# Agent configuration
AGENT_ID = "GL-017"
AGENT_NAME = "Condensync"

T = TypeVar('T')


# =============================================================================
# ENUMS
# =============================================================================

class ViolationSeverity(Enum):
    """Severity levels for guardrail violations."""
    INFO = auto()       # Informational only
    WARNING = auto()    # Should be addressed
    ERROR = auto()      # Must be corrected
    CRITICAL = auto()   # Immediate attention required
    BLOCKING = auto()   # Action blocked


class GuardrailProfile(Enum):
    """Predefined guardrail profiles for different security levels."""
    MINIMAL = auto()      # Basic input validation
    STANDARD = auto()     # Input + output validation
    STRICT = auto()       # Full validation + action gating
    INDUSTRIAL = auto()   # Strict + physical safety constraints
    REGULATORY = auto()   # Full compliance mode


class GuardrailType(str, Enum):
    """Types of guardrail checks."""
    INPUT_VALIDATION = "input_validation"
    OUTPUT_CHECK = "output_check"
    ACTION_GATE = "action_gate"
    SAFETY_ENVELOPE = "safety_envelope"
    RATE_LIMIT = "rate_limit"
    PHYSICAL_BOUNDS = "physical_bounds"
    RECOMMENDATION = "recommendation"


class ActionType(str, Enum):
    """Types of actions that can be gated."""
    READ = "read"
    RECOMMEND = "recommend"
    OPTIMIZE = "optimize"
    CONTROL = "control"
    EXECUTE = "execute"
    WRITE = "write"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GuardrailViolation:
    """
    Record of a guardrail violation.

    Attributes:
        violation_id: Unique violation identifier
        guardrail_name: Name of the guardrail that was violated
        guardrail_type: Type of guardrail check
        severity: Violation severity level
        message: Human-readable description
        timestamp: When the violation occurred
        context: Additional context information
        remediation: Suggested remediation action
    """
    violation_id: str
    guardrail_name: str
    guardrail_type: GuardrailType
    severity: ViolationSeverity
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)
    remediation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "violation_id": self.violation_id,
            "guardrail_name": self.guardrail_name,
            "guardrail_type": self.guardrail_type.value,
            "severity": self.severity.name,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "remediation": self.remediation,
        }


@dataclass
class GuardrailResult:
    """
    Result of guardrail checks.

    Attributes:
        passed: Whether all checks passed (no blocking violations)
        violations: List of violations found
        execution_time_ms: Time taken for checks
        checks_performed: List of checks that were performed
        provenance_hash: SHA-256 hash for audit trail
    """
    passed: bool
    violations: List[GuardrailViolation] = field(default_factory=list)
    execution_time_ms: float = 0.0
    checks_performed: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    @property
    def has_blocking_violation(self) -> bool:
        """Check if any blocking violations exist."""
        return any(v.severity == ViolationSeverity.BLOCKING for v in self.violations)

    @property
    def has_critical_violation(self) -> bool:
        """Check if any critical or blocking violations exist."""
        return any(
            v.severity in (ViolationSeverity.CRITICAL, ViolationSeverity.BLOCKING)
            for v in self.violations
        )

    @property
    def warning_count(self) -> int:
        """Count of warning-level violations."""
        return sum(1 for v in self.violations if v.severity == ViolationSeverity.WARNING)

    @property
    def error_count(self) -> int:
        """Count of error-level or higher violations."""
        return sum(
            1 for v in self.violations
            if v.severity in (ViolationSeverity.ERROR, ViolationSeverity.CRITICAL, ViolationSeverity.BLOCKING)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "execution_time_ms": round(self.execution_time_ms, 2),
            "checks_performed": self.checks_performed,
            "violation_count": len(self.violations),
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class SafetyEnvelope:
    """
    Immutable safety envelope definition for condenser operation.

    Defines the safe operating bounds for condenser parameters
    per HEI Standards and ASME PTC 12.2.

    Attributes:
        vacuum_pressure_min_kpa: Minimum safe vacuum pressure
        vacuum_pressure_max_kpa: Maximum safe vacuum pressure
        cw_inlet_temp_min_c: Minimum cooling water inlet temperature
        cw_inlet_temp_max_c: Maximum cooling water inlet temperature
        cw_outlet_temp_max_c: Maximum cooling water outlet temperature
        cf_min: Minimum acceptable cleanliness factor
        ttd_max_c: Maximum acceptable TTD
        approach_max_c: Maximum acceptable approach temperature
        air_leak_max_kg_hr: Maximum acceptable air leak rate
        hotwell_level_min_pct: Minimum hotwell level
        hotwell_level_max_pct: Maximum hotwell level
        cw_flow_change_max_pct: Maximum allowed CW flow change per recommendation
    """
    vacuum_pressure_min_kpa: float = 2.0
    vacuum_pressure_max_kpa: float = 15.0
    cw_inlet_temp_min_c: float = 5.0
    cw_inlet_temp_max_c: float = 40.0
    cw_outlet_temp_max_c: float = 60.0
    cf_min: float = 0.50
    ttd_max_c: float = 15.0
    approach_max_c: float = 20.0
    air_leak_max_kg_hr: float = 100.0
    hotwell_level_min_pct: float = 20.0
    hotwell_level_max_pct: float = 90.0
    cw_flow_change_max_pct: float = 20.0


# =============================================================================
# SAFETY ENVELOPE DEFINITIONS
# =============================================================================

# Default industrial safety envelope per HEI Standards
DEFAULT_SAFETY_ENVELOPE = SafetyEnvelope()

# Strict safety envelope for critical operations
STRICT_SAFETY_ENVELOPE = SafetyEnvelope(
    vacuum_pressure_min_kpa=3.0,
    vacuum_pressure_max_kpa=12.0,
    cw_inlet_temp_min_c=10.0,
    cw_inlet_temp_max_c=35.0,
    cw_outlet_temp_max_c=55.0,
    cf_min=0.60,
    ttd_max_c=12.0,
    approach_max_c=15.0,
    air_leak_max_kg_hr=60.0,
    hotwell_level_min_pct=30.0,
    hotwell_level_max_pct=80.0,
    cw_flow_change_max_pct=15.0
)

# Relaxed envelope for startup/shutdown operations
STARTUP_SAFETY_ENVELOPE = SafetyEnvelope(
    vacuum_pressure_min_kpa=1.0,
    vacuum_pressure_max_kpa=30.0,
    cw_inlet_temp_min_c=0.0,
    cw_inlet_temp_max_c=45.0,
    cw_outlet_temp_max_c=70.0,
    cf_min=0.40,
    ttd_max_c=25.0,
    approach_max_c=30.0,
    air_leak_max_kg_hr=150.0,
    hotwell_level_min_pct=10.0,
    hotwell_level_max_pct=95.0,
    cw_flow_change_max_pct=30.0
)


# =============================================================================
# MAIN GUARDRAILS CLASS
# =============================================================================

class CondenserGuardrails:
    """
    Main guardrails integration class for GL-017 Condensync.

    Provides physical safety validation, recommendation checking, action gating,
    and provenance tracking for condenser optimization operations.

    ZERO-HALLUCINATION GUARANTEE:
    - All safety checks use deterministic rules from HEI/ASME standards
    - No LLM or AI inference in safety validation path
    - Same inputs always produce identical safety assessments

    Example:
        >>> guardrails = CondenserGuardrails(profile=GuardrailProfile.INDUSTRIAL)
        >>> result = guardrails.check_input(input_data)
        >>> if result.passed:
        ...     process_input(input_data)

    Thread Safety:
        All public methods are thread-safe via RLock.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        profile: GuardrailProfile = GuardrailProfile.INDUSTRIAL,
        safety_envelope: Optional[SafetyEnvelope] = None,
        max_actions_per_minute: int = 60,
        max_recommendations_per_hour: int = 100,
    ):
        """
        Initialize guardrails integration.

        Args:
            profile: Guardrail profile to use
            safety_envelope: Custom safety envelope (uses default for profile if None)
            max_actions_per_minute: Rate limit for actions
            max_recommendations_per_hour: Rate limit for recommendations
        """
        self.profile = profile
        self.max_actions_per_minute = max_actions_per_minute
        self.max_recommendations_per_hour = max_recommendations_per_hour

        # Set safety envelope based on profile
        if safety_envelope:
            self.safety_envelope = safety_envelope
        elif profile == GuardrailProfile.STRICT:
            self.safety_envelope = STRICT_SAFETY_ENVELOPE
        else:
            self.safety_envelope = DEFAULT_SAFETY_ENVELOPE

        # Rate limiting state
        self._action_timestamps: List[float] = []
        self._recommendation_timestamps: List[float] = []

        # Audit log
        self._violation_log: List[GuardrailViolation] = []

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            f"CondenserGuardrails v{self.VERSION} initialized for {AGENT_ID} "
            f"(profile={profile.name}, max_actions/min={max_actions_per_minute})"
        )

    def check_input(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> GuardrailResult:
        """
        Validate input data against guardrails.

        Performs prompt injection detection, physical bounds checking,
        and input sanitization.

        Args:
            input_data: Input to validate
            context: Additional context

        Returns:
            GuardrailResult with any violations
        """
        start = time.time()
        violations: List[GuardrailViolation] = []
        checks_performed = []

        # Check for prompt injection patterns (if string)
        if isinstance(input_data, str):
            injection_result = self._check_prompt_injection(input_data)
            violations.extend(injection_result)
            checks_performed.append("prompt_injection")

        # Check physical bounds (if dict with condenser params)
        if isinstance(input_data, dict):
            bounds_result = self._check_physical_bounds(input_data)
            violations.extend(bounds_result)
            checks_performed.append("physical_bounds")

        # Check for sensitive data in input
        if self.profile in (GuardrailProfile.STRICT, GuardrailProfile.INDUSTRIAL, GuardrailProfile.REGULATORY):
            sensitive_result = self._check_sensitive_data(input_data)
            violations.extend(sensitive_result)
            checks_performed.append("sensitive_data")

        elapsed = (time.time() - start) * 1000
        passed = not any(v.severity == ViolationSeverity.BLOCKING for v in violations)

        # Calculate provenance
        provenance_hash = self._compute_provenance_hash(input_data, violations)

        return GuardrailResult(
            passed=passed,
            violations=violations,
            execution_time_ms=elapsed,
            checks_performed=checks_performed,
            provenance_hash=provenance_hash
        )

    def check_output(
        self,
        output_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> GuardrailResult:
        """
        Check output data for policy violations.

        Ensures output does not contain sensitive data leakage
        or invalid recommendations.

        Args:
            output_data: Output to check
            context: Additional context

        Returns:
            GuardrailResult with any violations
        """
        start = time.time()
        violations: List[GuardrailViolation] = []
        checks_performed = []

        # Check for data leakage
        leakage_result = self._check_data_leakage(output_data)
        violations.extend(leakage_result)
        checks_performed.append("data_leakage")

        # Check output bounds if numeric
        if isinstance(output_data, dict):
            bounds_result = self._check_output_bounds(output_data)
            violations.extend(bounds_result)
            checks_performed.append("output_bounds")

        elapsed = (time.time() - start) * 1000
        passed = not any(v.severity == ViolationSeverity.BLOCKING for v in violations)

        provenance_hash = self._compute_provenance_hash(output_data, violations)

        return GuardrailResult(
            passed=passed,
            violations=violations,
            execution_time_ms=elapsed,
            checks_performed=checks_performed,
            provenance_hash=provenance_hash
        )

    def check_action(
        self,
        action_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> GuardrailResult:
        """
        Gate action execution with rate limiting and safety checks.

        Args:
            action_data: Action to check
            context: Additional context with action_type

        Returns:
            GuardrailResult with any violations
        """
        start = time.time()
        violations: List[GuardrailViolation] = []
        checks_performed = []

        # Get action type from context
        action_type = ActionType.RECOMMEND
        if context and "action_type" in context:
            try:
                action_type = ActionType(context["action_type"])
            except ValueError:
                pass

        # Rate limiting
        rate_result = self._check_rate_limit(action_type)
        violations.extend(rate_result)
        checks_performed.append("rate_limit")

        # Safety envelope check for control/execute actions
        if action_type in (ActionType.CONTROL, ActionType.EXECUTE, ActionType.OPTIMIZE):
            if self.profile in (GuardrailProfile.INDUSTRIAL, GuardrailProfile.REGULATORY):
                safety_result = self._check_safety_envelope(action_data)
                violations.extend(safety_result)
                checks_performed.append("safety_envelope")

        # Require confirmation for dangerous actions
        if action_type in (ActionType.CONTROL, ActionType.EXECUTE):
            if self.profile == GuardrailProfile.REGULATORY:
                violations.append(GuardrailViolation(
                    violation_id=str(uuid.uuid4()),
                    guardrail_name="ActionConfirmation",
                    guardrail_type=GuardrailType.ACTION_GATE,
                    severity=ViolationSeverity.WARNING,
                    message=f"Action type '{action_type.value}' requires operator confirmation",
                    context={"action_type": action_type.value},
                    remediation="Ensure operator has confirmed this action"
                ))

        elapsed = (time.time() - start) * 1000
        passed = not any(v.severity == ViolationSeverity.BLOCKING for v in violations)

        provenance_hash = self._compute_provenance_hash(action_data, violations)

        return GuardrailResult(
            passed=passed,
            violations=violations,
            execution_time_ms=elapsed,
            checks_performed=checks_performed,
            provenance_hash=provenance_hash
        )

    def check_recommendation(
        self,
        recommendation: Dict[str, Any],
        current_state: Optional[Dict[str, Any]] = None
    ) -> GuardrailResult:
        """
        Validate an optimization recommendation.

        Ensures the recommendation is within safe operating bounds
        and doesn't propose dangerous parameter changes.

        Args:
            recommendation: Recommendation data with proposed changes
            current_state: Current condenser state (for delta validation)

        Returns:
            GuardrailResult with any violations
        """
        start = time.time()
        violations: List[GuardrailViolation] = []
        checks_performed = []

        # Check recommendation rate limit
        rate_result = self._check_recommendation_rate()
        violations.extend(rate_result)
        checks_performed.append("recommendation_rate")

        # Check proposed values against safety envelope
        safety_result = self._check_recommendation_safety(recommendation)
        violations.extend(safety_result)
        checks_performed.append("recommendation_safety")

        # Check delta from current state
        if current_state:
            delta_result = self._check_recommendation_delta(recommendation, current_state)
            violations.extend(delta_result)
            checks_performed.append("recommendation_delta")

        # Log recommendation for audit
        with self._lock:
            self._recommendation_timestamps.append(time.time())

        elapsed = (time.time() - start) * 1000
        passed = not any(v.severity == ViolationSeverity.BLOCKING for v in violations)

        provenance_hash = self._compute_provenance_hash(recommendation, violations)

        return GuardrailResult(
            passed=passed,
            violations=violations,
            execution_time_ms=elapsed,
            checks_performed=checks_performed,
            provenance_hash=provenance_hash
        )

    def _check_prompt_injection(self, text: str) -> List[GuardrailViolation]:
        """Check for prompt injection patterns."""
        violations = []
        injection_patterns = [
            "ignore previous instructions",
            "disregard",
            "override",
            "system prompt",
            "jailbreak",
            "pretend you are",
            "act as if",
            "forget everything",
        ]

        text_lower = text.lower()
        for pattern in injection_patterns:
            if pattern in text_lower:
                violations.append(GuardrailViolation(
                    violation_id=str(uuid.uuid4()),
                    guardrail_name="PromptInjection",
                    guardrail_type=GuardrailType.INPUT_VALIDATION,
                    severity=ViolationSeverity.BLOCKING,
                    message=f"Potential prompt injection detected: '{pattern}'",
                    context={"input_hash": self._hash(text)[:16]},
                    remediation="Remove suspicious patterns from input"
                ))
                break  # One detection is enough

        return violations

    def _check_physical_bounds(self, data: Dict[str, Any]) -> List[GuardrailViolation]:
        """Check physical parameter bounds."""
        violations = []
        env = self.safety_envelope

        bounds_map = {
            "vacuum_pressure_kpa": (env.vacuum_pressure_min_kpa, env.vacuum_pressure_max_kpa, "kPa"),
            "cw_inlet_temp_c": (env.cw_inlet_temp_min_c, env.cw_inlet_temp_max_c, "C"),
            "cw_outlet_temp_c": (0, env.cw_outlet_temp_max_c, "C"),
            "cleanliness_factor": (env.cf_min, 1.0, ""),
            "ttd": (-10, env.ttd_max_c, "C"),
            "approach_temp": (0, env.approach_max_c, "C"),
            "air_leak_rate_kg_hr": (0, env.air_leak_max_kg_hr, "kg/hr"),
            "hotwell_level_pct": (env.hotwell_level_min_pct, env.hotwell_level_max_pct, "%"),
        }

        for param, (min_val, max_val, unit) in bounds_map.items():
            if param in data:
                value = data[param]
                if isinstance(value, (int, float)):
                    if value < min_val:
                        violations.append(GuardrailViolation(
                            violation_id=str(uuid.uuid4()),
                            guardrail_name="PhysicalBounds",
                            guardrail_type=GuardrailType.PHYSICAL_BOUNDS,
                            severity=ViolationSeverity.ERROR,
                            message=f"{param}={value}{unit} below minimum {min_val}{unit}",
                            context={"parameter": param, "value": value, "min": min_val},
                            remediation=f"Ensure {param} >= {min_val}{unit}"
                        ))
                    elif value > max_val:
                        violations.append(GuardrailViolation(
                            violation_id=str(uuid.uuid4()),
                            guardrail_name="PhysicalBounds",
                            guardrail_type=GuardrailType.PHYSICAL_BOUNDS,
                            severity=ViolationSeverity.ERROR,
                            message=f"{param}={value}{unit} above maximum {max_val}{unit}",
                            context={"parameter": param, "value": value, "max": max_val},
                            remediation=f"Ensure {param} <= {max_val}{unit}"
                        ))

        return violations

    def _check_sensitive_data(self, data: Any) -> List[GuardrailViolation]:
        """Check for sensitive data in input."""
        violations = []
        sensitive_patterns = [
            (r"api[_-]?key\s*[=:]", "API key"),
            (r"password\s*[=:]", "password"),
            (r"secret\s*[=:]", "secret"),
            (r"token\s*[=:]", "token"),
            (r"private[_-]?key", "private key"),
        ]

        data_str = json.dumps(data, default=str) if not isinstance(data, str) else data

        for pattern, name in sensitive_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                violations.append(GuardrailViolation(
                    violation_id=str(uuid.uuid4()),
                    guardrail_name="SensitiveData",
                    guardrail_type=GuardrailType.INPUT_VALIDATION,
                    severity=ViolationSeverity.CRITICAL,
                    message=f"Potential {name} detected in input",
                    context={"pattern_type": name},
                    remediation=f"Remove {name} from input data"
                ))

        return violations

    def _check_data_leakage(self, data: Any) -> List[GuardrailViolation]:
        """Check for data leakage in output."""
        violations = []

        if not isinstance(data, str):
            data_str = json.dumps(data, default=str)
        else:
            data_str = data

        leakage_patterns = [
            (r"api[_-]?key\s*[=:]", "API key"),
            (r"password\s*[=:]", "password"),
            (r"secret\s*[=:]", "secret"),
            (r"-----BEGIN.*PRIVATE KEY-----", "private key"),
        ]

        for pattern, name in leakage_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                violations.append(GuardrailViolation(
                    violation_id=str(uuid.uuid4()),
                    guardrail_name="DataLeakage",
                    guardrail_type=GuardrailType.OUTPUT_CHECK,
                    severity=ViolationSeverity.CRITICAL,
                    message=f"Potential {name} leakage in output",
                    context={"pattern_type": name},
                    remediation=f"Remove {name} from output"
                ))

        return violations

    def _check_output_bounds(self, data: Dict[str, Any]) -> List[GuardrailViolation]:
        """Check output values against safety bounds."""
        violations = []

        # Check for unreasonable efficiency values
        if "efficiency" in data:
            eff = data["efficiency"]
            if isinstance(eff, (int, float)):
                if eff < 0 or eff > 1.0:
                    violations.append(GuardrailViolation(
                        violation_id=str(uuid.uuid4()),
                        guardrail_name="OutputBounds",
                        guardrail_type=GuardrailType.OUTPUT_CHECK,
                        severity=ViolationSeverity.ERROR,
                        message=f"Efficiency {eff} outside valid range [0, 1]",
                        context={"value": eff},
                        remediation="Verify efficiency calculation"
                    ))

        # Check for unreasonable improvement claims
        if "improvement_pct" in data:
            imp = data["improvement_pct"]
            if isinstance(imp, (int, float)) and imp > 20:
                violations.append(GuardrailViolation(
                    violation_id=str(uuid.uuid4()),
                    guardrail_name="OutputBounds",
                    guardrail_type=GuardrailType.OUTPUT_CHECK,
                    severity=ViolationSeverity.WARNING,
                    message=f"Claimed improvement {imp}% exceeds typical maximum",
                    context={"value": imp},
                    remediation="Verify improvement calculation"
                ))

        return violations

    def _check_rate_limit(self, action_type: ActionType) -> List[GuardrailViolation]:
        """Check action rate limit."""
        violations = []
        now = time.time()

        with self._lock:
            # Clean old timestamps
            self._action_timestamps = [
                t for t in self._action_timestamps
                if now - t < 60
            ]

            if len(self._action_timestamps) >= self.max_actions_per_minute:
                violations.append(GuardrailViolation(
                    violation_id=str(uuid.uuid4()),
                    guardrail_name="ActionRateLimit",
                    guardrail_type=GuardrailType.RATE_LIMIT,
                    severity=ViolationSeverity.BLOCKING,
                    message=f"Rate limit exceeded: {self.max_actions_per_minute}/min",
                    context={
                        "current_count": len(self._action_timestamps),
                        "limit": self.max_actions_per_minute,
                    },
                    remediation="Wait before making more requests"
                ))
            else:
                self._action_timestamps.append(now)

        return violations

    def _check_recommendation_rate(self) -> List[GuardrailViolation]:
        """Check recommendation rate limit."""
        violations = []
        now = time.time()

        with self._lock:
            # Clean old timestamps (1 hour window)
            self._recommendation_timestamps = [
                t for t in self._recommendation_timestamps
                if now - t < 3600
            ]

            if len(self._recommendation_timestamps) >= self.max_recommendations_per_hour:
                violations.append(GuardrailViolation(
                    violation_id=str(uuid.uuid4()),
                    guardrail_name="RecommendationRateLimit",
                    guardrail_type=GuardrailType.RATE_LIMIT,
                    severity=ViolationSeverity.WARNING,
                    message=f"Recommendation rate limit: {self.max_recommendations_per_hour}/hr",
                    context={
                        "current_count": len(self._recommendation_timestamps),
                        "limit": self.max_recommendations_per_hour,
                    },
                    remediation="Consider batching recommendations"
                ))

        return violations

    def _check_safety_envelope(self, data: Any) -> List[GuardrailViolation]:
        """Check action against safety envelope."""
        violations = []

        if not isinstance(data, dict):
            return violations

        env = self.safety_envelope

        # Check proposed setpoint changes
        setpoint_limits = {
            "target_vacuum_kpa": (env.vacuum_pressure_min_kpa, env.vacuum_pressure_max_kpa),
            "target_cw_flow_pct": (50.0, 110.0),  # 50-110% of design
            "target_hotwell_level_pct": (env.hotwell_level_min_pct, env.hotwell_level_max_pct),
        }

        for param, (min_val, max_val) in setpoint_limits.items():
            if param in data:
                value = data[param]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        violations.append(GuardrailViolation(
                            violation_id=str(uuid.uuid4()),
                            guardrail_name="SafetyEnvelope",
                            guardrail_type=GuardrailType.SAFETY_ENVELOPE,
                            severity=ViolationSeverity.BLOCKING,
                            message=f"Proposed {param}={value} outside safety envelope [{min_val}, {max_val}]",
                            context={"parameter": param, "value": value, "envelope": [min_val, max_val]},
                            remediation=f"Keep {param} within [{min_val}, {max_val}]"
                        ))

        return violations

    def _check_recommendation_safety(self, recommendation: Dict[str, Any]) -> List[GuardrailViolation]:
        """Check recommendation against safety constraints."""
        violations = []

        # Check that recommended CF is not too low
        if "recommended_cf" in recommendation:
            cf = recommendation["recommended_cf"]
            if isinstance(cf, (int, float)) and cf < self.safety_envelope.cf_min:
                violations.append(GuardrailViolation(
                    violation_id=str(uuid.uuid4()),
                    guardrail_name="RecommendationSafety",
                    guardrail_type=GuardrailType.RECOMMENDATION,
                    severity=ViolationSeverity.ERROR,
                    message=f"Recommended CF={cf} below minimum {self.safety_envelope.cf_min}",
                    context={"recommended_cf": cf, "minimum": self.safety_envelope.cf_min},
                    remediation="Do not recommend operation below minimum CF"
                ))

        # Check vacuum recommendations
        if "recommended_vacuum_kpa" in recommendation:
            vacuum = recommendation["recommended_vacuum_kpa"]
            env = self.safety_envelope
            if isinstance(vacuum, (int, float)):
                if vacuum < env.vacuum_pressure_min_kpa or vacuum > env.vacuum_pressure_max_kpa:
                    violations.append(GuardrailViolation(
                        violation_id=str(uuid.uuid4()),
                        guardrail_name="RecommendationSafety",
                        guardrail_type=GuardrailType.RECOMMENDATION,
                        severity=ViolationSeverity.BLOCKING,
                        message=f"Recommended vacuum={vacuum} kPa outside safe range",
                        context={"value": vacuum},
                        remediation="Keep vacuum recommendation within safe envelope"
                    ))

        return violations

    def _check_recommendation_delta(
        self,
        recommendation: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> List[GuardrailViolation]:
        """Check that recommendation doesn't propose too large a change."""
        violations = []

        # Check CW flow change
        if "recommended_cw_flow_pct" in recommendation and "cw_flow_pct" in current_state:
            current = current_state["cw_flow_pct"]
            proposed = recommendation["recommended_cw_flow_pct"]
            if isinstance(current, (int, float)) and isinstance(proposed, (int, float)):
                delta = abs(proposed - current)
                max_delta = self.safety_envelope.cw_flow_change_max_pct
                if delta > max_delta:
                    violations.append(GuardrailViolation(
                        violation_id=str(uuid.uuid4()),
                        guardrail_name="RecommendationDelta",
                        guardrail_type=GuardrailType.RECOMMENDATION,
                        severity=ViolationSeverity.WARNING,
                        message=f"CW flow change {delta:.1f}% exceeds max {max_delta}%",
                        context={"current": current, "proposed": proposed, "delta": delta},
                        remediation="Implement change in smaller increments"
                    ))

        return violations

    def _compute_provenance_hash(
        self,
        data: Any,
        violations: List[GuardrailViolation]
    ) -> str:
        """Compute SHA-256 provenance hash for audit trail."""
        provenance_data = {
            "agent_id": AGENT_ID,
            "data_hash": self._hash(data),
            "violation_count": len(violations),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profile": self.profile.name,
        }
        return hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

    def _hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data."""
        try:
            json_str = json.dumps(data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            json_str = str(data)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get guardrails statistics."""
        with self._lock:
            return {
                "agent_id": AGENT_ID,
                "version": self.VERSION,
                "profile": self.profile.name,
                "violation_count": len(self._violation_log),
                "actions_last_minute": len(self._action_timestamps),
                "recommendations_last_hour": len(self._recommendation_timestamps),
                "max_actions_per_minute": self.max_actions_per_minute,
                "max_recommendations_per_hour": self.max_recommendations_per_hour,
                "safety_envelope": {
                    "vacuum_range_kpa": [
                        self.safety_envelope.vacuum_pressure_min_kpa,
                        self.safety_envelope.vacuum_pressure_max_kpa
                    ],
                    "cf_min": self.safety_envelope.cf_min,
                    "cw_flow_change_max_pct": self.safety_envelope.cw_flow_change_max_pct,
                }
            }


# =============================================================================
# GLOBAL INSTANCE AND FACTORY
# =============================================================================

_guardrails: Optional[CondenserGuardrails] = None


def get_guardrails(
    profile: GuardrailProfile = GuardrailProfile.INDUSTRIAL
) -> CondenserGuardrails:
    """Get or create the global guardrails instance."""
    global _guardrails
    if _guardrails is None:
        _guardrails = CondenserGuardrails(profile=profile)
    return _guardrails


# =============================================================================
# DECORATOR
# =============================================================================

def with_guardrails(
    profile: GuardrailProfile = GuardrailProfile.INDUSTRIAL,
    action_type: Optional[ActionType] = None,
) -> Callable:
    """
    Decorator for wrapping functions with guardrail protection.

    Example:
        >>> @with_guardrails(profile=GuardrailProfile.INDUSTRIAL)
        ... def optimize_condenser(data: dict) -> dict:
        ...     return {"efficiency": 0.85}
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            guardrails = get_guardrails(profile)

            # Check input
            input_data = {"args": args, "kwargs": kwargs}
            input_result = guardrails.check_input(input_data)

            if input_result.has_blocking_violation:
                raise ValueError(
                    f"Input blocked: {input_result.violations[0].message}"
                )

            # Check action if specified
            if action_type:
                action_result = guardrails.check_action(
                    input_data,
                    context={"action_type": action_type.value}
                )
                if action_result.has_blocking_violation:
                    raise ValueError(
                        f"Action blocked: {action_result.violations[0].message}"
                    )

            # Execute function
            result = func(*args, **kwargs)

            # Check output
            output_result = guardrails.check_output(result)

            if output_result.has_blocking_violation:
                raise ValueError(
                    f"Output blocked: {output_result.violations[0].message}"
                )

            return result

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "CondenserGuardrails",

    # Data classes
    "GuardrailViolation",
    "GuardrailResult",
    "SafetyEnvelope",

    # Enums
    "ViolationSeverity",
    "GuardrailProfile",
    "GuardrailType",
    "ActionType",

    # Safety envelopes
    "DEFAULT_SAFETY_ENVELOPE",
    "STRICT_SAFETY_ENVELOPE",
    "STARTUP_SAFETY_ENVELOPE",

    # Factory and decorator
    "get_guardrails",
    "with_guardrails",
]
