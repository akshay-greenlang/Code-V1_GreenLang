"""
Policy Engine for GL-FOUND-X-003 Unit & Reference Normalizer.

This module implements the core PolicyEngine class that evaluates
requests against policies and produces deterministic, auditable
decisions. The engine supports STRICT and LENIENT modes and
integrates with compliance profiles.

Key Design Principles:
    - Deterministic evaluation for reproducibility
    - Complete audit trail of all decisions
    - Zero-hallucination: no LLM calls in decision path
    - Support for multiple compliance frameworks

Example:
    >>> from gl_normalizer_core.policy.engine import PolicyEngine
    >>> from gl_normalizer_core.policy.models import PolicyMode
    >>> engine = PolicyEngine()
    >>> decision = engine.evaluate(request, context)
    >>> if decision.allowed:
    ...     print(decision.effective_config.gwp_version)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import hashlib
import uuid

import structlog

from gl_normalizer_core.policy.models import (
    Policy,
    PolicyMode,
    PolicyDecision,
    PolicyDefaults,
    PolicyOverrides,
    PolicyWarning,
    AppliedDefault,
    EffectiveConfig,
    ReferenceConditions,
    ComplianceProfile,
    ConversionPolicy,
)
from gl_normalizer_core.policy.defaults import (
    get_system_defaults,
    get_profile_defaults,
    get_org_defaults,
    DEFAULT_GWP_VERSION,
    DEFAULT_BASIS,
    DEFAULT_TEMPERATURE_REF,
    DEFAULT_PRESSURE_REF,
)
from gl_normalizer_core.policy.compliance import (
    validate_against_profile,
    validate_against_profiles,
    get_required_fields_for_profile,
    create_warnings_from_validation,
    ProfileValidationResult,
    RuleSeverity,
)
from gl_normalizer_core.policy.loader import (
    load_org_policy,
    load_default_policy,
    merge_policies,
    merge_policy_chain,
)
from gl_normalizer_core.errors import PolicyViolationError
from gl_normalizer_core.errors.codes import GLNORMErrorCode

logger = structlog.get_logger(__name__)


class PolicyEngine:
    """
    Policy Engine for evaluating normalization requests.

    The PolicyEngine is the central component for policy evaluation.
    It takes a request and context, applies the effective policy,
    validates against compliance profiles, and produces a decision.

    Operating Modes:
        - STRICT: Fails on any missing required context; no defaults
          applied silently. Use for production compliance scenarios.
        - LENIENT: Applies defaults with warnings when context is
          missing. Use for exploratory or legacy data processing.

    Attributes:
        default_mode: Default operating mode when not specified.
        enable_caching: Whether to cache policy lookups.
        strict_compliance: Whether to fail on any compliance warning.

    Example:
        >>> engine = PolicyEngine(default_mode=PolicyMode.STRICT)
        >>> request = {"source_unit": "kg", "target_unit": "t"}
        >>> context = {"gwp_version": "AR5", "org_id": "org-acme"}
        >>> decision = engine.evaluate(request, context)
        >>> print(decision.allowed)
        True
    """

    def __init__(
        self,
        default_mode: PolicyMode = PolicyMode.STRICT,
        enable_caching: bool = True,
        strict_compliance: bool = False,
    ) -> None:
        """
        Initialize the PolicyEngine.

        Args:
            default_mode: Default operating mode.
            enable_caching: Whether to cache policy lookups.
            strict_compliance: Whether to fail on compliance warnings.
        """
        self.default_mode = default_mode
        self.enable_caching = enable_caching
        self.strict_compliance = strict_compliance

        logger.info(
            "PolicyEngine initialized",
            default_mode=default_mode.value,
            enable_caching=enable_caching,
            strict_compliance=strict_compliance,
        )

    def evaluate(
        self,
        request: Dict[str, Any],
        context: Dict[str, Any],
    ) -> PolicyDecision:
        """
        Evaluate a request against the effective policy.

        This is the main entry point for policy evaluation. It:
        1. Resolves the effective policy from context
        2. Validates request against policy rules
        3. Checks compliance profile requirements
        4. Applies defaults (in LENIENT mode) or fails (in STRICT mode)
        5. Returns a complete decision with audit trail

        Args:
            request: The normalization request data.
            context: Context including org_id, gwp_version, etc.

        Returns:
            PolicyDecision with allowed flag and effective config.

        Example:
            >>> request = {"value": 100, "source_unit": "kg", "target_unit": "t"}
            >>> context = {"org_id": "org-acme", "gwp_version": "AR5"}
            >>> decision = engine.evaluate(request, context)
        """
        start_time = datetime.utcnow()
        decision_id = f"dec-{uuid.uuid4().hex[:12]}"
        warnings: List[PolicyWarning] = []
        applied_defaults: List[AppliedDefault] = []

        try:
            # Step 1: Get effective policy
            effective_policy = self.get_effective_policy(
                org_id=context.get("org_id"),
                request_overrides=context.get("policy_overrides"),
            )

            mode = effective_policy.mode
            policy_hash = effective_policy.get_policy_hash()

            logger.debug(
                "Evaluating request",
                decision_id=decision_id,
                policy_id=effective_policy.policy_id,
                mode=mode.value,
            )

            # Step 2: Build evaluation context with request and context merged
            eval_context = self._build_evaluation_context(
                request=request,
                context=context,
                policy=effective_policy,
            )

            # Step 3: Check for missing required fields
            missing_fields = self._check_missing_fields(
                eval_context=eval_context,
                policy=effective_policy,
            )

            # Step 4: Handle missing fields based on mode
            if missing_fields:
                if mode == PolicyMode.STRICT:
                    # STRICT mode: fail on missing required context
                    evaluation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    return PolicyDecision(
                        allowed=False,
                        decision_id=decision_id,
                        policy_id=effective_policy.policy_id,
                        policy_version=effective_policy.version,
                        policy_hash=policy_hash,
                        mode=mode,
                        warnings=warnings,
                        applied_defaults=applied_defaults,
                        effective_config=self._create_minimal_config(effective_policy),
                        evaluation_time_ms=evaluation_time_ms,
                        denial_reason=f"Missing required context in STRICT mode: {', '.join(missing_fields)}",
                    )
                else:
                    # LENIENT mode: apply defaults with warnings
                    for field in missing_fields:
                        default_value, source = self._get_default_for_field(
                            field=field,
                            policy=effective_policy,
                        )
                        eval_context[field] = default_value
                        applied_defaults.append(AppliedDefault(
                            field=field,
                            default_value=default_value,
                            reason=f"Missing from request context",
                            source=source,
                        ))
                        warnings.append(PolicyWarning(
                            code="POL_WARN_DEFAULT_APPLIED",
                            message=f"Applied default value for '{field}': {default_value}",
                            field=field,
                            severity="medium",
                            hint=f"Specify '{field}' in context to avoid this warning",
                        ))

            # Step 5: Validate against compliance profiles
            compliance_warnings, compliance_errors = self._validate_compliance(
                eval_context=eval_context,
                policy=effective_policy,
            )

            warnings.extend(compliance_warnings)

            if compliance_errors:
                if mode == PolicyMode.STRICT or self.strict_compliance:
                    evaluation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    error_messages = [e.message for e in compliance_errors]
                    return PolicyDecision(
                        allowed=False,
                        decision_id=decision_id,
                        policy_id=effective_policy.policy_id,
                        policy_version=effective_policy.version,
                        policy_hash=policy_hash,
                        mode=mode,
                        warnings=warnings,
                        applied_defaults=applied_defaults,
                        effective_config=self._create_minimal_config(effective_policy),
                        evaluation_time_ms=evaluation_time_ms,
                        denial_reason=f"Compliance validation failed: {'; '.join(error_messages)}",
                    )
                else:
                    # LENIENT mode: convert errors to warnings
                    for error in compliance_errors:
                        warnings.append(PolicyWarning(
                            code=error.error_code.value if error.error_code else "POL_WARN_COMPLIANCE",
                            message=error.message,
                            severity="high",
                            hint=error.hint,
                        ))

            # Step 6: Validate request-specific rules
            request_validation = self._validate_request(
                request=request,
                eval_context=eval_context,
                policy=effective_policy,
            )

            if not request_validation["valid"]:
                if mode == PolicyMode.STRICT:
                    evaluation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    return PolicyDecision(
                        allowed=False,
                        decision_id=decision_id,
                        policy_id=effective_policy.policy_id,
                        policy_version=effective_policy.version,
                        policy_hash=policy_hash,
                        mode=mode,
                        warnings=warnings,
                        applied_defaults=applied_defaults,
                        effective_config=self._create_minimal_config(effective_policy),
                        evaluation_time_ms=evaluation_time_ms,
                        denial_reason=request_validation["reason"],
                    )
                else:
                    warnings.append(PolicyWarning(
                        code="POL_WARN_REQUEST",
                        message=request_validation["reason"],
                        severity="medium",
                    ))

            # Step 7: Build effective configuration
            effective_config = self._build_effective_config(
                eval_context=eval_context,
                policy=effective_policy,
            )

            # Step 8: Create successful decision
            evaluation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            decision = PolicyDecision(
                allowed=True,
                decision_id=decision_id,
                policy_id=effective_policy.policy_id,
                policy_version=effective_policy.version,
                policy_hash=policy_hash,
                mode=mode,
                warnings=warnings,
                applied_defaults=applied_defaults,
                effective_config=effective_config,
                evaluation_time_ms=evaluation_time_ms,
            )

            logger.info(
                "Policy evaluation complete",
                decision_id=decision_id,
                allowed=True,
                warning_count=len(warnings),
                defaults_applied=len(applied_defaults),
                evaluation_time_ms=evaluation_time_ms,
            )

            return decision

        except Exception as e:
            # Handle unexpected errors
            evaluation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(
                "Policy evaluation failed",
                decision_id=decision_id,
                error=str(e),
                exc_info=True,
            )

            # Return denial with error information
            return PolicyDecision(
                allowed=False,
                decision_id=decision_id,
                policy_id="unknown",
                policy_version="0.0.0",
                policy_hash="error",
                mode=self.default_mode,
                warnings=warnings,
                applied_defaults=applied_defaults,
                effective_config=EffectiveConfig(
                    gwp_version=DEFAULT_GWP_VERSION,
                    basis=DEFAULT_BASIS,
                    reference_conditions=ReferenceConditions(
                        temperature_c=DEFAULT_TEMPERATURE_REF,
                        pressure_kpa=DEFAULT_PRESSURE_REF,
                    ),
                    confidence_threshold=0.8,
                    precision_digits=6,
                ),
                evaluation_time_ms=evaluation_time_ms,
                denial_reason=f"Policy evaluation error: {str(e)}",
            )

    def get_effective_policy(
        self,
        org_id: Optional[str] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> Policy:
        """
        Get the effective policy for an organization with optional overrides.

        The effective policy is built by merging:
        1. System defaults
        2. Organization policy (if org_id provided)
        3. Request-level overrides (if provided)

        Args:
            org_id: Organization identifier.
            request_overrides: Request-level policy overrides.

        Returns:
            Effective Policy for the request.

        Example:
            >>> policy = engine.get_effective_policy(
            ...     org_id="org-acme",
            ...     request_overrides={"mode": "LENIENT"},
            ... )
        """
        policies: List[Policy] = []

        # Start with system default
        system_policy = load_default_policy(self.default_mode)
        policies.append(system_policy)

        # Add org policy if available
        if org_id:
            org_policy = load_org_policy(
                org_id=org_id,
                use_cache=self.enable_caching,
            )
            if org_policy:
                policies.append(org_policy)
                logger.debug(
                    "Loaded org policy",
                    org_id=org_id,
                    policy_id=org_policy.policy_id,
                )

        # Add request overrides as policy
        if request_overrides:
            override_policy = self._create_override_policy(request_overrides)
            policies.append(override_policy)
            logger.debug("Applied request overrides")

        # Merge all policies
        effective = merge_policy_chain(policies)

        logger.debug(
            "Built effective policy",
            policy_count=len(policies),
            effective_mode=effective.mode.value,
            profile_count=len(effective.compliance_profiles),
        )

        return effective

    def _build_evaluation_context(
        self,
        request: Dict[str, Any],
        context: Dict[str, Any],
        policy: Policy,
    ) -> Dict[str, Any]:
        """Build the complete evaluation context."""
        # Start with policy defaults
        eval_context: Dict[str, Any] = {
            "gwp_version": policy.defaults.gwp_version,
            "basis": policy.defaults.basis,
            "confidence_threshold": policy.defaults.confidence_threshold,
            "precision_digits": policy.defaults.precision_digits,
            "reference_conditions": policy.defaults.reference_conditions,
        }

        # Apply policy overrides
        if policy.overrides.gwp_version:
            eval_context["gwp_version"] = policy.overrides.gwp_version
        if policy.overrides.basis:
            eval_context["basis"] = policy.overrides.basis
        if policy.overrides.confidence_threshold is not None:
            eval_context["confidence_threshold"] = policy.overrides.confidence_threshold
        if policy.overrides.reference_conditions:
            eval_context["reference_conditions"] = policy.overrides.reference_conditions

        # Apply context values (override policy)
        for key, value in context.items():
            if value is not None and key != "policy_overrides":
                eval_context[key] = value

        # Apply request values
        for key, value in request.items():
            if value is not None:
                eval_context[key] = value

        return eval_context

    def _check_missing_fields(
        self,
        eval_context: Dict[str, Any],
        policy: Policy,
    ) -> List[str]:
        """Check for missing required fields based on compliance profiles."""
        missing: List[str] = []

        # Get required fields from all profiles
        required_fields: Set[str] = set()
        for profile in policy.compliance_profiles:
            required_fields |= get_required_fields_for_profile(profile)

        # Check each required field
        for field in required_fields:
            if field not in eval_context or eval_context[field] is None:
                missing.append(field)

        return missing

    def _get_default_for_field(
        self,
        field: str,
        policy: Policy,
    ) -> tuple:
        """Get the default value for a field."""
        # Check policy defaults first
        if field == "gwp_version":
            return policy.defaults.gwp_version, "policy_defaults"
        elif field == "basis":
            return policy.defaults.basis, "policy_defaults"
        elif field == "reference_conditions":
            return policy.defaults.reference_conditions, "policy_defaults"
        elif field == "confidence_threshold":
            return policy.defaults.confidence_threshold, "policy_defaults"
        elif field == "precision_digits":
            return policy.defaults.precision_digits, "policy_defaults"

        # Fall back to system defaults
        system_defaults = get_system_defaults()
        if hasattr(system_defaults, field):
            return getattr(system_defaults, field), "system_defaults"

        return None, "none"

    def _validate_compliance(
        self,
        eval_context: Dict[str, Any],
        policy: Policy,
    ) -> tuple:
        """Validate context against compliance profiles."""
        warnings: List[PolicyWarning] = []
        errors: List[Any] = []

        if not policy.compliance_profiles:
            return warnings, errors

        # Validate against each profile
        results = validate_against_profiles(eval_context, policy.compliance_profiles)

        for profile, result in results.items():
            # Collect errors
            for error in result.errors:
                errors.append(error)

            # Collect warnings
            profile_warnings = create_warnings_from_validation(result)
            warnings.extend(profile_warnings)

        return warnings, errors

    def _validate_request(
        self,
        request: Dict[str, Any],
        eval_context: Dict[str, Any],
        policy: Policy,
    ) -> Dict[str, Any]:
        """Validate request-specific rules."""
        # Check for blocked units
        if policy.overrides.blocked_units:
            source_unit = request.get("source_unit")
            target_unit = request.get("target_unit")

            if source_unit and source_unit in policy.overrides.blocked_units:
                return {
                    "valid": False,
                    "reason": f"Source unit '{source_unit}' is blocked by policy",
                }
            if target_unit and target_unit in policy.overrides.blocked_units:
                return {
                    "valid": False,
                    "reason": f"Target unit '{target_unit}' is blocked by policy",
                }

        # Check for allowed units restriction
        if policy.overrides.allowed_units:
            source_unit = request.get("source_unit")
            target_unit = request.get("target_unit")

            if source_unit and source_unit not in policy.overrides.allowed_units:
                return {
                    "valid": False,
                    "reason": f"Source unit '{source_unit}' is not in allowed units list",
                }
            if target_unit and target_unit not in policy.overrides.allowed_units:
                return {
                    "valid": False,
                    "reason": f"Target unit '{target_unit}' is not in allowed units list",
                }

        return {"valid": True, "reason": None}

    def _build_effective_config(
        self,
        eval_context: Dict[str, Any],
        policy: Policy,
    ) -> EffectiveConfig:
        """Build the effective configuration from evaluation context."""
        # Get reference conditions
        ref_conditions = eval_context.get("reference_conditions")
        if ref_conditions is None:
            ref_conditions = ReferenceConditions(
                temperature_c=DEFAULT_TEMPERATURE_REF,
                pressure_kpa=DEFAULT_PRESSURE_REF,
            )
        elif isinstance(ref_conditions, dict):
            ref_conditions = ReferenceConditions(
                temperature_c=ref_conditions.get("temperature_c", DEFAULT_TEMPERATURE_REF),
                pressure_kpa=ref_conditions.get("pressure_kpa", DEFAULT_PRESSURE_REF),
            )

        return EffectiveConfig(
            gwp_version=eval_context.get("gwp_version", DEFAULT_GWP_VERSION),
            basis=eval_context.get("basis", DEFAULT_BASIS),
            reference_conditions=ref_conditions,
            confidence_threshold=eval_context.get("confidence_threshold", 0.8),
            precision_digits=eval_context.get("precision_digits", 6),
            compliance_profiles=policy.compliance_profiles,
            allowed_units=policy.overrides.allowed_units,
            blocked_units=policy.overrides.blocked_units,
            custom_conversions=policy.overrides.custom_conversions,
            allow_deprecated_factors=policy.defaults.allow_deprecated_factors,
        )

    def _create_minimal_config(self, policy: Policy) -> EffectiveConfig:
        """Create a minimal effective config for error cases."""
        return EffectiveConfig(
            gwp_version=policy.defaults.gwp_version,
            basis=policy.defaults.basis,
            reference_conditions=policy.defaults.reference_conditions,
            confidence_threshold=policy.defaults.confidence_threshold,
            precision_digits=policy.defaults.precision_digits,
            compliance_profiles=policy.compliance_profiles,
        )

    def _create_override_policy(
        self,
        overrides: Dict[str, Any],
    ) -> Policy:
        """Create a policy from request overrides."""
        # Parse mode if present
        mode = self.default_mode
        if "mode" in overrides:
            try:
                mode = PolicyMode(overrides["mode"].upper())
            except ValueError:
                pass

        # Parse compliance profiles if present
        profiles = []
        if "compliance_profiles" in overrides:
            for profile_str in overrides["compliance_profiles"]:
                try:
                    profiles.append(ComplianceProfile(profile_str))
                except ValueError:
                    pass

        # Build overrides
        policy_overrides = PolicyOverrides(
            gwp_version=overrides.get("gwp_version"),
            basis=overrides.get("basis"),
            confidence_threshold=overrides.get("confidence_threshold"),
        )

        return Policy(
            policy_id="request-override",
            version="1.0.0",
            mode=mode,
            compliance_profiles=profiles,
            defaults=PolicyDefaults(),
            overrides=policy_overrides,
            description="Request-level policy overrides",
        )


# =============================================================================
# Module-level convenience functions
# =============================================================================

# Global engine instance
_default_engine: Optional[PolicyEngine] = None


def get_default_engine() -> PolicyEngine:
    """
    Get the default PolicyEngine instance.

    Creates a new instance if one does not exist.

    Returns:
        Default PolicyEngine instance.
    """
    global _default_engine
    if _default_engine is None:
        _default_engine = PolicyEngine()
    return _default_engine


def set_default_engine(engine: PolicyEngine) -> None:
    """
    Set the default PolicyEngine instance.

    Args:
        engine: PolicyEngine instance to use as default.
    """
    global _default_engine
    _default_engine = engine
    logger.info("Default PolicyEngine updated")


def evaluate(
    request: Dict[str, Any],
    context: Dict[str, Any],
) -> PolicyDecision:
    """
    Evaluate a request using the default engine.

    Convenience function that uses the default PolicyEngine.

    Args:
        request: The normalization request data.
        context: Context including org_id, gwp_version, etc.

    Returns:
        PolicyDecision with allowed flag and effective config.

    Example:
        >>> from gl_normalizer_core.policy.engine import evaluate
        >>> decision = evaluate(request, context)
    """
    return get_default_engine().evaluate(request, context)


__all__ = [
    "PolicyEngine",
    "get_default_engine",
    "set_default_engine",
    "evaluate",
]
