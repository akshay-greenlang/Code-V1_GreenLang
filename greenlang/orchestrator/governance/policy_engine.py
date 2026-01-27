# -*- coding: utf-8 -*-
"""
GL-FOUND-X-001: Hybrid OPA + YAML Policy Engine
================================================

This module implements the governance policy engine for the GreenLang orchestrator.
It provides hybrid policy evaluation using both Open Policy Agent (OPA) for complex
Rego policies and YAML-based declarative rules for simple conditions.

Policy Evaluation Points:
    - Pre-run: Pipeline + plan validation before execution starts
    - Pre-step: Permissions, cost, data residency before each step
    - Post-step: Artifact classification, export controls after step completion

Features:
    - OPA HTTP client for Rego policy evaluation
    - YAML rules parser for declarative conditions
    - Policy bundle management with versioning
    - Cost budget enforcement
    - Data residency rules
    - Namespace-level and organization baseline policies

Zero-Hallucination Guarantees:
    - All policy decisions are deterministic
    - Complete audit trail with SHA-256 hashes
    - No probabilistic decision making

Example:
    >>> engine = PolicyEngine(config)
    >>> decision = await engine.evaluate_pre_run(pipeline, run_config)
    >>> if not decision.allowed:
    ...     raise PolicyViolation(decision.reasons[0].message)

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import operator
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator

from greenlang.orchestrator.pipeline_schema import (
    DataClassification,
    ExecutionContext,
    PipelineDefinition,
    RunConfig,
    StepDefinition,
    StepResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class PolicyAction(str, Enum):
    """Actions that policies can require."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    WARN = "warn"
    AUDIT = "audit"


class PolicySeverity(str, Enum):
    """Severity levels for policy violations."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class EvaluationPoint(str, Enum):
    """Points where policy evaluation occurs."""

    PRE_RUN = "pre_run"
    PRE_STEP = "pre_step"
    POST_STEP = "post_step"


class ApprovalType(str, Enum):
    """Types of approvals that can be required."""

    MANAGER = "manager"
    SECURITY = "security"
    DATA_OWNER = "data_owner"
    COMPLIANCE = "compliance"
    COST_CENTER = "cost_center"


# Operators for YAML condition evaluation
YAML_OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
    "contains": lambda a, b: b in a,
    "starts_with": lambda a, b: str(a).startswith(str(b)),
    "ends_with": lambda a, b: str(a).endswith(str(b)),
    "matches": lambda a, b: bool(re.match(b, str(a))),
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class PolicyReason(BaseModel):
    """Reason for a policy decision."""

    rule_name: str = Field(..., description="Name of the rule that triggered")
    message: str = Field(..., description="Human-readable message")
    severity: PolicySeverity = Field(default=PolicySeverity.ERROR)
    action: PolicyAction = Field(..., description="Action taken")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class ApprovalRequirement(BaseModel):
    """Approval required by a policy."""

    approval_type: ApprovalType = Field(..., description="Type of approval")
    approver_id: Optional[str] = Field(None, description="Specific approver ID")
    approver_role: Optional[str] = Field(None, description="Required approver role")
    reason: str = Field(..., description="Why approval is required")
    deadline_hours: int = Field(default=24, ge=1, le=168, description="Approval deadline")
    auto_deny_on_timeout: bool = Field(
        default=True, description="Auto-deny if not approved"
    )


class PolicyDecision(BaseModel):
    """Result of policy evaluation."""

    allowed: bool = Field(..., description="Whether the action is allowed")
    reasons: List[PolicyReason] = Field(
        default_factory=list, description="Reasons for decision"
    )
    required_approvals: List[ApprovalRequirement] = Field(
        default_factory=list, description="Required approvals"
    )
    policy_version: str = Field(default="1.0.0", description="Policy bundle version")
    evaluation_time_ms: float = Field(default=0.0, description="Evaluation duration")
    evaluation_point: EvaluationPoint = Field(..., description="Evaluation point")
    provenance_hash: str = Field(default="", description="Decision provenance hash")
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Evaluation timestamp",
    )

    # Audit metadata
    evaluated_policies: List[str] = Field(
        default_factory=list, description="Policies that were evaluated"
    )
    warnings: List[str] = Field(default_factory=list, description="Non-blocking warnings")

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        content = {
            "allowed": self.allowed,
            "reasons": [r.model_dump() for r in self.reasons],
            "required_approvals": [a.model_dump() for a in self.required_approvals],
            "policy_version": self.policy_version,
            "evaluation_point": self.evaluation_point.value,
            "evaluated_at": self.evaluated_at.isoformat(),
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()


class YAMLRule(BaseModel):
    """A single declarative YAML rule."""

    name: str = Field(..., description="Rule name")
    description: Optional[str] = Field(None, description="Rule description")
    condition: str = Field(..., description="Condition expression")
    action: PolicyAction = Field(..., description="Action when condition matches")
    severity: PolicySeverity = Field(default=PolicySeverity.ERROR)
    message: Optional[str] = Field(None, description="Custom message template")
    enabled: bool = Field(default=True, description="Whether rule is active")
    evaluation_points: List[EvaluationPoint] = Field(
        default_factory=lambda: [EvaluationPoint.PRE_RUN],
        description="When to evaluate",
    )

    # Approval configuration
    approval_type: Optional[ApprovalType] = Field(None, description="Approval type")
    approval_role: Optional[str] = Field(None, description="Approver role")

    # Targeting
    namespaces: List[str] = Field(
        default_factory=list, description="Target namespaces (empty = all)"
    )
    pipelines: List[str] = Field(
        default_factory=list, description="Target pipelines (empty = all)"
    )

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, v: str) -> str:
        """Validate condition syntax."""
        # Basic validation - ensure it's not empty and has expected structure
        if not v or not v.strip():
            raise ValueError("Condition cannot be empty")
        return v.strip()


class YAMLRuleSet(BaseModel):
    """Collection of YAML rules."""

    rules: List[YAMLRule] = Field(default_factory=list, description="Rules")
    version: str = Field(default="1.0.0", description="Rule set version")
    name: str = Field(default="default", description="Rule set name")
    description: Optional[str] = Field(None, description="Rule set description")


class CostBudget(BaseModel):
    """Cost budget configuration."""

    max_cost_usd: float = Field(..., ge=0, description="Maximum cost in USD")
    warn_threshold_percent: float = Field(
        default=80.0, ge=0, le=100, description="Warning threshold"
    )
    enforce_on_estimate: bool = Field(
        default=True, description="Enforce on estimated cost"
    )
    allow_override: bool = Field(
        default=False, description="Allow override with approval"
    )
    override_approval_type: ApprovalType = Field(
        default=ApprovalType.COST_CENTER, description="Approval for override"
    )


class DataResidencyRule(BaseModel):
    """Data residency constraint."""

    name: str = Field(..., description="Rule name")
    allowed_regions: List[str] = Field(..., description="Allowed data regions")
    denied_regions: List[str] = Field(
        default_factory=list, description="Explicitly denied regions"
    )
    applies_to_classification: List[DataClassification] = Field(
        default_factory=lambda: [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED],
        description="Classification levels this applies to",
    )
    message: str = Field(
        default="Data residency violation", description="Violation message"
    )


class PolicyBundle(BaseModel):
    """A versioned collection of policies."""

    bundle_id: str = Field(..., description="Bundle identifier")
    version: str = Field(..., description="Bundle version")
    name: str = Field(..., description="Bundle name")
    description: Optional[str] = Field(None, description="Bundle description")

    # Rules and configurations
    yaml_rules: List[YAMLRule] = Field(default_factory=list, description="YAML rules")
    opa_policies: Dict[str, str] = Field(
        default_factory=dict, description="OPA policy names to paths"
    )

    # Budget and residency
    cost_budgets: Dict[str, CostBudget] = Field(
        default_factory=dict, description="Cost budgets by namespace"
    )
    residency_rules: List[DataResidencyRule] = Field(
        default_factory=list, description="Data residency rules"
    )

    # Scope
    namespace: Optional[str] = Field(None, description="Namespace scope (None = org)")
    inherits_from: Optional[str] = Field(None, description="Parent bundle to inherit")
    priority: int = Field(default=100, ge=0, le=1000, description="Evaluation priority")

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    created_by: Optional[str] = Field(None, description="Creator")

    def compute_hash(self) -> str:
        """Compute bundle hash for versioning."""
        content = {
            "bundle_id": self.bundle_id,
            "version": self.version,
            "yaml_rules": [r.model_dump() for r in self.yaml_rules],
            "opa_policies": self.opa_policies,
            "cost_budgets": {k: v.model_dump() for k, v in self.cost_budgets.items()},
            "residency_rules": [r.model_dump() for r in self.residency_rules],
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()


class PolicyEngineConfig(BaseModel):
    """Configuration for the policy engine."""

    # OPA configuration
    opa_enabled: bool = Field(default=True, description="Enable OPA evaluation")
    opa_url: str = Field(
        default="http://localhost:8181", description="OPA server URL"
    )
    opa_timeout_seconds: float = Field(default=5.0, ge=0.1, le=30.0)
    opa_retry_count: int = Field(default=2, ge=0, le=5)

    # YAML rules
    yaml_rules_enabled: bool = Field(default=True, description="Enable YAML rules")
    yaml_rules_path: Optional[str] = Field(None, description="Path to YAML rules file")

    # Caching
    cache_enabled: bool = Field(default=True, description="Enable decision caching")
    cache_ttl_seconds: int = Field(default=300, ge=0, le=3600)

    # Defaults
    default_action: PolicyAction = Field(
        default=PolicyAction.DENY, description="Default when no rules match"
    )
    strict_mode: bool = Field(
        default=True, description="Deny on evaluation errors"
    )

    # Audit
    audit_all_decisions: bool = Field(default=True, description="Audit all decisions")
    audit_log_path: Optional[str] = Field(None, description="Audit log path")


# =============================================================================
# OPA CLIENT
# =============================================================================


class OPAClient:
    """
    HTTP client for Open Policy Agent server.

    Provides methods to query OPA for policy decisions with proper
    error handling, retries, and timeout management.

    Example:
        >>> client = OPAClient("http://localhost:8181")
        >>> result = await client.query("data.policies.allow", input_doc)
    """

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 5.0,
        retry_count: int = 2,
    ) -> None:
        """
        Initialize OPA client.

        Args:
            base_url: OPA server base URL
            timeout_seconds: Request timeout
            retry_count: Number of retry attempts
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        self._session = None

        logger.info(f"OPA client initialized: {self.base_url}")

    async def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            try:
                import httpx
                self._session = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout_seconds)
                )
            except ImportError:
                # Fallback to aiohttp if httpx not available
                import aiohttp
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            await self._session.aclose()
            self._session = None

    async def query(
        self,
        policy_path: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Query OPA for a policy decision.

        Args:
            policy_path: OPA policy path (e.g., "data.policies.allow")
            input_data: Input document for policy evaluation

        Returns:
            OPA decision result

        Raises:
            OPAError: If OPA query fails
        """
        url = f"{self.base_url}/v1/{policy_path.replace('.', '/')}"
        payload = {"input": input_data}

        last_error = None
        for attempt in range(self.retry_count + 1):
            try:
                session = await self._get_session()

                # Handle both httpx and aiohttp
                if hasattr(session, "post"):
                    response = await session.post(url, json=payload)
                    if hasattr(response, "json"):
                        if callable(response.json):
                            # aiohttp
                            result = await response.json()
                        else:
                            # httpx
                            result = response.json()
                    else:
                        result = json.loads(await response.text())

                    status = response.status if hasattr(response, "status") else response.status_code

                    if status == 200:
                        return result
                    else:
                        last_error = f"OPA returned status {status}: {result}"
                else:
                    raise RuntimeError("No valid HTTP client available")

            except Exception as e:
                last_error = str(e)
                if attempt < self.retry_count:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    logger.warning(f"OPA query retry {attempt + 1}: {e}")

        raise OPAError(f"OPA query failed after {self.retry_count + 1} attempts: {last_error}")

    async def check_health(self) -> bool:
        """Check if OPA server is healthy."""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/health"

            if hasattr(session, "get"):
                response = await session.get(url)
                status = response.status if hasattr(response, "status") else response.status_code
                return status == 200
            return False
        except Exception as e:
            logger.warning(f"OPA health check failed: {e}")
            return False

    def format_input(
        self,
        pipeline: Optional[PipelineDefinition] = None,
        run_config: Optional[RunConfig] = None,
        step: Optional[StepDefinition] = None,
        context: Optional[ExecutionContext] = None,
        step_result: Optional[StepResult] = None,
    ) -> Dict[str, Any]:
        """
        Format input document for OPA query.

        Args:
            pipeline: Pipeline definition
            run_config: Run configuration
            step: Current step
            context: Execution context
            step_result: Step result (for post-step)

        Returns:
            Formatted input document for OPA
        """
        doc: Dict[str, Any] = {}

        if pipeline:
            doc["pipeline"] = pipeline.to_policy_doc()

        if run_config:
            doc["run"] = run_config.to_policy_doc()

        if step:
            doc["step"] = {
                "name": step.name,
                "agent_id": step.agent_id,
                "publishes_data": step.publishes_data,
                "accesses_pii": step.accesses_pii,
                "requires_approval": step.requires_approval,
                "data_regions": step.data_regions,
            }

        if context:
            doc["context"] = context.to_policy_doc()

        if step_result:
            doc["result"] = {
                "step_name": step_result.step_name,
                "success": step_result.success,
                "output_classification": (
                    step_result.output_classification.value
                    if step_result.output_classification
                    else None
                ),
                "export_destinations": step_result.export_destinations,
                "artifacts": step_result.artifacts,
            }

        return doc

    def parse_decision(
        self,
        opa_result: Dict[str, Any],
        evaluation_point: EvaluationPoint,
    ) -> PolicyDecision:
        """
        Parse OPA result into PolicyDecision.

        Args:
            opa_result: Raw OPA response
            evaluation_point: Evaluation point

        Returns:
            Parsed PolicyDecision
        """
        result_data = opa_result.get("result", {})

        # Handle different OPA response formats
        if isinstance(result_data, bool):
            allowed = result_data
            reasons = []
            approvals = []
        elif isinstance(result_data, dict):
            allowed = result_data.get("allow", result_data.get("allowed", False))
            reasons = [
                PolicyReason(
                    rule_name=r.get("rule", "opa_rule"),
                    message=r.get("message", "Policy violation"),
                    severity=PolicySeverity(r.get("severity", "error")),
                    action=PolicyAction.DENY,
                    details=r.get("details", {}),
                )
                for r in result_data.get("violations", [])
            ]
            approvals = [
                ApprovalRequirement(
                    approval_type=ApprovalType(a.get("type", "manager")),
                    reason=a.get("reason", "Approval required"),
                    approver_role=a.get("role"),
                )
                for a in result_data.get("required_approvals", [])
            ]
        else:
            allowed = False
            reasons = [
                PolicyReason(
                    rule_name="opa_error",
                    message=f"Unexpected OPA result format: {type(result_data)}",
                    severity=PolicySeverity.ERROR,
                    action=PolicyAction.DENY,
                )
            ]
            approvals = []

        decision = PolicyDecision(
            allowed=allowed and len(reasons) == 0,
            reasons=reasons,
            required_approvals=approvals,
            evaluation_point=evaluation_point,
            evaluated_policies=["opa"],
        )
        decision.provenance_hash = decision.compute_provenance_hash()

        return decision


class OPAError(Exception):
    """Exception for OPA-related errors."""

    pass


# =============================================================================
# YAML RULES PARSER
# =============================================================================


class YAMLRulesParser:
    """
    Parser for declarative YAML policy rules.

    Evaluates simple conditions like:
        - namespace == "production" and step.publishes_data
        - run.estimated_cost_usd > budget.max_cost_usd

    Example:
        >>> parser = YAMLRulesParser()
        >>> parser.load_rules(yaml_path)
        >>> decision = parser.evaluate(context, EvaluationPoint.PRE_RUN)
    """

    def __init__(self) -> None:
        """Initialize YAML rules parser."""
        self._rules: List[YAMLRule] = []
        self._version = "1.0.0"
        self._name = "default"

    def load_rules(self, yaml_path: str) -> None:
        """
        Load rules from YAML file.

        Args:
            yaml_path: Path to YAML rules file

        Raises:
            FileNotFoundError: If file not found
            ValueError: If YAML is invalid
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Rules file not found: {yaml_path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            logger.warning(f"Empty rules file: {yaml_path}")
            return

        if "rules" in data:
            rule_set = YAMLRuleSet(**data)
            self._rules = rule_set.rules
            self._version = rule_set.version
            self._name = rule_set.name
        else:
            # Single rule format
            self._rules = [YAMLRule(**data)]

        logger.info(f"Loaded {len(self._rules)} YAML rules from {yaml_path}")

    def load_rules_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load rules from dictionary.

        Args:
            data: Dictionary with rules configuration
        """
        if "rules" in data:
            rule_set = YAMLRuleSet(**data)
            self._rules = rule_set.rules
            self._version = rule_set.version
            self._name = rule_set.name
        else:
            self._rules = [YAMLRule(**data)]

    def add_rule(self, rule: YAMLRule) -> None:
        """Add a rule to the parser."""
        self._rules.append(rule)

    def get_rules(self) -> List[YAMLRule]:
        """Get all loaded rules."""
        return self._rules.copy()

    def evaluate(
        self,
        context: Dict[str, Any],
        evaluation_point: EvaluationPoint,
        namespace: Optional[str] = None,
        pipeline_name: Optional[str] = None,
    ) -> List[Tuple[YAMLRule, bool, Optional[str]]]:
        """
        Evaluate all rules against context.

        Args:
            context: Evaluation context dictionary
            evaluation_point: Current evaluation point
            namespace: Namespace filter
            pipeline_name: Pipeline name filter

        Returns:
            List of (rule, matched, message) tuples
        """
        results: List[Tuple[YAMLRule, bool, Optional[str]]] = []

        for rule in self._rules:
            # Skip disabled rules
            if not rule.enabled:
                continue

            # Check evaluation point
            if evaluation_point not in rule.evaluation_points:
                continue

            # Check namespace filter
            if rule.namespaces and namespace and namespace not in rule.namespaces:
                continue

            # Check pipeline filter
            if rule.pipelines and pipeline_name and pipeline_name not in rule.pipelines:
                continue

            try:
                matched = self._evaluate_condition(rule.condition, context)
                message = self._render_message(rule.message, context) if rule.message else None
                results.append((rule, matched, message))
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
                # On error, treat as not matched but log it
                results.append((rule, False, f"Evaluation error: {e}"))

        return results

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate a condition expression.

        Supports:
            - Simple comparisons: namespace == "production"
            - Nested access: step.publishes_data
            - Boolean operators: and, or, not
            - Operators: ==, !=, >, <, >=, <=, in, not_in, contains

        Args:
            condition: Condition string
            context: Context dictionary

        Returns:
            True if condition matches
        """
        # Handle boolean operators
        condition = condition.strip()

        # Split by 'and' (lowest precedence after 'or')
        if " and " in condition:
            parts = condition.split(" and ")
            return all(self._evaluate_condition(p.strip(), context) for p in parts)

        # Split by 'or'
        if " or " in condition:
            parts = condition.split(" or ")
            return any(self._evaluate_condition(p.strip(), context) for p in parts)

        # Handle 'not' prefix
        if condition.startswith("not "):
            return not self._evaluate_condition(condition[4:].strip(), context)

        # Handle parentheses (simple case)
        if condition.startswith("(") and condition.endswith(")"):
            return self._evaluate_condition(condition[1:-1], context)

        # Handle simple variable access (truthy check)
        for op_str, op_func in YAML_OPERATORS.items():
            if f" {op_str} " in condition:
                left_str, right_str = condition.split(f" {op_str} ", 1)
                left_val = self._resolve_value(left_str.strip(), context)
                right_val = self._resolve_value(right_str.strip(), context)
                return op_func(left_val, right_val)

        # Truthy check for simple variable
        value = self._resolve_value(condition, context)
        return bool(value)

    def _resolve_value(self, expr: str, context: Dict[str, Any]) -> Any:
        """
        Resolve a value expression from context.

        Args:
            expr: Expression string (e.g., "step.name", "'literal'", "123")
            context: Context dictionary

        Returns:
            Resolved value
        """
        expr = expr.strip()

        # String literal
        if (expr.startswith('"') and expr.endswith('"')) or \
           (expr.startswith("'") and expr.endswith("'")):
            return expr[1:-1]

        # Boolean literals
        if expr.lower() == "true":
            return True
        if expr.lower() == "false":
            return False
        if expr.lower() == "none" or expr.lower() == "null":
            return None

        # Numeric literals
        try:
            if "." in expr:
                return float(expr)
            return int(expr)
        except ValueError:
            pass

        # List literal
        if expr.startswith("[") and expr.endswith("]"):
            items = expr[1:-1].split(",")
            return [self._resolve_value(item.strip(), context) for item in items]

        # Variable access (dotted path)
        parts = expr.split(".")
        value = context

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None

            if value is None:
                return None

        return value

    def _render_message(self, template: str, context: Dict[str, Any]) -> str:
        """
        Render a message template with context values.

        Supports: {{ variable.path }} syntax

        Args:
            template: Message template
            context: Context dictionary

        Returns:
            Rendered message
        """
        if not template:
            return ""

        def replace_var(match):
            var_path = match.group(1).strip()
            value = self._resolve_value(var_path, context)
            return str(value) if value is not None else ""

        pattern = r"\{\{\s*([^}]+)\s*\}\}"
        return re.sub(pattern, replace_var, template)


# =============================================================================
# POLICY ENGINE
# =============================================================================


class PolicyEngine:
    """
    Hybrid OPA + YAML Policy Engine for GreenLang Orchestrator.

    Evaluates policies at multiple points during pipeline execution:
    - Pre-run: Validate pipeline and run configuration
    - Pre-step: Check permissions, cost, residency before each step
    - Post-step: Validate artifacts, export controls after step completion

    Features:
    - OPA Rego policies for complex rules
    - YAML declarative rules for simple conditions
    - Policy bundle management with versioning
    - Cost budget enforcement
    - Data residency rules
    - Approval workflow integration

    Example:
        >>> config = PolicyEngineConfig(opa_url="http://opa:8181")
        >>> engine = PolicyEngine(config)
        >>> await engine.load_bundles()
        >>> decision = await engine.evaluate_pre_run(pipeline, run_config)
    """

    def __init__(self, config: Optional[PolicyEngineConfig] = None) -> None:
        """
        Initialize policy engine.

        Args:
            config: Engine configuration
        """
        self.config = config or PolicyEngineConfig()
        self._bundles: Dict[str, PolicyBundle] = {}
        self._yaml_parser = YAMLRulesParser()
        self._opa_client: Optional[OPAClient] = None

        # Initialize OPA client if enabled
        if self.config.opa_enabled:
            self._opa_client = OPAClient(
                base_url=self.config.opa_url,
                timeout_seconds=self.config.opa_timeout_seconds,
                retry_count=self.config.opa_retry_count,
            )

        # Load YAML rules if path provided
        if self.config.yaml_rules_enabled and self.config.yaml_rules_path:
            self._yaml_parser.load_rules(self.config.yaml_rules_path)

        # Decision cache
        self._cache: Dict[str, Tuple[PolicyDecision, float]] = {}

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []

        logger.info(
            f"PolicyEngine initialized (OPA: {self.config.opa_enabled}, "
            f"YAML: {self.config.yaml_rules_enabled})"
        )

    async def close(self) -> None:
        """Close engine and release resources."""
        if self._opa_client:
            await self._opa_client.close()

    # =========================================================================
    # BUNDLE MANAGEMENT
    # =========================================================================

    def add_bundle(self, bundle: PolicyBundle) -> str:
        """
        Add a policy bundle.

        Args:
            bundle: Policy bundle to add

        Returns:
            Bundle hash
        """
        bundle_hash = bundle.compute_hash()
        self._bundles[bundle.bundle_id] = bundle

        # Load YAML rules from bundle
        for rule in bundle.yaml_rules:
            self._yaml_parser.add_rule(rule)

        logger.info(f"Added policy bundle: {bundle.bundle_id} v{bundle.version}")
        return bundle_hash

    def remove_bundle(self, bundle_id: str) -> bool:
        """Remove a policy bundle."""
        if bundle_id in self._bundles:
            del self._bundles[bundle_id]
            return True
        return False

    def get_bundle(self, bundle_id: str) -> Optional[PolicyBundle]:
        """Get a bundle by ID."""
        return self._bundles.get(bundle_id)

    def list_bundles(self) -> List[str]:
        """List all bundle IDs."""
        return list(self._bundles.keys())

    def get_effective_bundle(self, namespace: str) -> Optional[PolicyBundle]:
        """
        Get the effective bundle for a namespace.

        Resolves inheritance chain to find applicable bundle.

        Args:
            namespace: Target namespace

        Returns:
            Effective policy bundle or None
        """
        # Find namespace-specific bundle
        for bundle in self._bundles.values():
            if bundle.namespace == namespace:
                return bundle

        # Fall back to organization baseline (namespace=None)
        for bundle in self._bundles.values():
            if bundle.namespace is None:
                return bundle

        return None

    # =========================================================================
    # POLICY EVALUATION
    # =========================================================================

    async def evaluate_pre_run(
        self,
        pipeline: PipelineDefinition,
        run_config: RunConfig,
    ) -> PolicyDecision:
        """
        Evaluate policies before pipeline run starts.

        Checks:
        - Pipeline configuration validity
        - Run configuration compliance
        - Cost budget limits
        - Required approvals

        Args:
            pipeline: Pipeline definition
            run_config: Run configuration

        Returns:
            PolicyDecision indicating if run is allowed
        """
        start_time = time.perf_counter()
        evaluation_point = EvaluationPoint.PRE_RUN

        # Check cache
        cache_key = self._get_cache_key(
            evaluation_point, pipeline.metadata.name, run_config.run_id
        )
        cached = self._get_cached_decision(cache_key)
        if cached:
            return cached

        reasons: List[PolicyReason] = []
        approvals: List[ApprovalRequirement] = []
        warnings: List[str] = []
        evaluated_policies: List[str] = []

        # Build context for evaluation
        context = {
            "pipeline": pipeline.to_policy_doc(),
            "run": run_config.to_policy_doc(),
            "namespace": run_config.namespace,
            "environment": run_config.environment,
            "budget": self._get_budget(run_config.namespace),
        }

        # Evaluate YAML rules
        if self.config.yaml_rules_enabled:
            yaml_results = self._yaml_parser.evaluate(
                context,
                evaluation_point,
                namespace=run_config.namespace,
                pipeline_name=pipeline.metadata.name,
            )
            evaluated_policies.append("yaml_rules")

            for rule, matched, message in yaml_results:
                if matched:
                    if rule.action == PolicyAction.DENY:
                        reasons.append(PolicyReason(
                            rule_name=rule.name,
                            message=message or rule.description or f"Rule {rule.name} violated",
                            severity=rule.severity,
                            action=rule.action,
                        ))
                    elif rule.action == PolicyAction.REQUIRE_APPROVAL:
                        approvals.append(ApprovalRequirement(
                            approval_type=rule.approval_type or ApprovalType.MANAGER,
                            reason=message or rule.description or f"Approval required by {rule.name}",
                            approver_role=rule.approval_role,
                        ))
                    elif rule.action == PolicyAction.WARN:
                        warnings.append(message or rule.description or f"Warning from {rule.name}")

        # Evaluate OPA policies
        if self.config.opa_enabled and self._opa_client:
            try:
                opa_input = self._opa_client.format_input(
                    pipeline=pipeline, run_config=run_config
                )
                opa_result = await self._opa_client.query(
                    "data.greenlang.policies.pre_run", opa_input
                )
                opa_decision = self._opa_client.parse_decision(opa_result, evaluation_point)

                reasons.extend(opa_decision.reasons)
                approvals.extend(opa_decision.required_approvals)
                evaluated_policies.append("opa")
            except OPAError as e:
                logger.warning(f"OPA evaluation failed: {e}")
                if self.config.strict_mode:
                    reasons.append(PolicyReason(
                        rule_name="opa_unavailable",
                        message=f"OPA evaluation failed: {e}",
                        severity=PolicySeverity.ERROR,
                        action=PolicyAction.DENY,
                    ))

        # Check cost budget
        budget_result = self._check_cost_budget(run_config, reasons, approvals, warnings)

        # Check data residency
        self._check_data_residency(pipeline, run_config, reasons)

        # Build decision
        allowed = len(reasons) == 0
        evaluation_time_ms = (time.perf_counter() - start_time) * 1000

        decision = PolicyDecision(
            allowed=allowed,
            reasons=reasons,
            required_approvals=approvals,
            policy_version=self._get_policy_version(),
            evaluation_time_ms=evaluation_time_ms,
            evaluation_point=evaluation_point,
            evaluated_policies=evaluated_policies,
            warnings=warnings,
        )
        decision.provenance_hash = decision.compute_provenance_hash()

        # Cache and audit
        self._cache_decision(cache_key, decision)
        self._log_audit(decision, pipeline.metadata.name, run_config.run_id)

        return decision

    async def evaluate_pre_step(
        self,
        step: StepDefinition,
        context: ExecutionContext,
    ) -> PolicyDecision:
        """
        Evaluate policies before step execution.

        Checks:
        - Step permissions
        - Cost limits
        - Data residency for step
        - PII access controls

        Args:
            step: Step to execute
            context: Execution context

        Returns:
            PolicyDecision indicating if step can execute
        """
        start_time = time.perf_counter()
        evaluation_point = EvaluationPoint.PRE_STEP

        reasons: List[PolicyReason] = []
        approvals: List[ApprovalRequirement] = []
        warnings: List[str] = []
        evaluated_policies: List[str] = []

        # Build context for evaluation
        eval_context = {
            "step": {
                "name": step.name,
                "agent_id": step.agent_id,
                "publishes_data": step.publishes_data,
                "accesses_pii": step.accesses_pii,
                "requires_approval": step.requires_approval,
                "data_regions": step.data_regions,
            },
            "context": context.to_policy_doc(),
            "pipeline": context.pipeline.to_policy_doc(),
            "run": context.run_config.to_policy_doc(),
            "namespace": context.run_config.namespace,
            "budget": self._get_budget(context.run_config.namespace),
            "user_roles": context.user_roles,
            "permissions": list(context.permissions),
        }

        # Evaluate YAML rules
        if self.config.yaml_rules_enabled:
            yaml_results = self._yaml_parser.evaluate(
                eval_context,
                evaluation_point,
                namespace=context.run_config.namespace,
                pipeline_name=context.pipeline.metadata.name,
            )
            evaluated_policies.append("yaml_rules")

            for rule, matched, message in yaml_results:
                if matched:
                    self._apply_rule_result(rule, message, reasons, approvals, warnings)

        # Evaluate OPA policies
        if self.config.opa_enabled and self._opa_client:
            try:
                opa_input = self._opa_client.format_input(
                    pipeline=context.pipeline,
                    run_config=context.run_config,
                    step=step,
                    context=context,
                )
                opa_result = await self._opa_client.query(
                    "data.greenlang.policies.pre_step", opa_input
                )
                opa_decision = self._opa_client.parse_decision(opa_result, evaluation_point)

                reasons.extend(opa_decision.reasons)
                approvals.extend(opa_decision.required_approvals)
                evaluated_policies.append("opa")
            except OPAError as e:
                logger.warning(f"OPA evaluation failed for step {step.name}: {e}")
                if self.config.strict_mode:
                    reasons.append(PolicyReason(
                        rule_name="opa_unavailable",
                        message=f"OPA evaluation failed: {e}",
                        severity=PolicySeverity.ERROR,
                        action=PolicyAction.DENY,
                    ))

        # Check step-specific policies
        self._check_step_policies(step, context, reasons, approvals)

        # Check accumulated cost
        self._check_accumulated_cost(context, reasons, warnings)

        # Build decision
        allowed = len(reasons) == 0
        evaluation_time_ms = (time.perf_counter() - start_time) * 1000

        decision = PolicyDecision(
            allowed=allowed,
            reasons=reasons,
            required_approvals=approvals,
            policy_version=self._get_policy_version(),
            evaluation_time_ms=evaluation_time_ms,
            evaluation_point=evaluation_point,
            evaluated_policies=evaluated_policies,
            warnings=warnings,
        )
        decision.provenance_hash = decision.compute_provenance_hash()

        self._log_audit(decision, step.name, context.run_id)

        return decision

    async def evaluate_post_step(
        self,
        step: StepDefinition,
        result: StepResult,
        context: ExecutionContext,
    ) -> PolicyDecision:
        """
        Evaluate policies after step execution.

        Checks:
        - Artifact classification
        - Export controls
        - Data egress rules

        Args:
            step: Executed step
            result: Step execution result
            context: Execution context

        Returns:
            PolicyDecision for post-step validation
        """
        start_time = time.perf_counter()
        evaluation_point = EvaluationPoint.POST_STEP

        reasons: List[PolicyReason] = []
        approvals: List[ApprovalRequirement] = []
        warnings: List[str] = []
        evaluated_policies: List[str] = []

        # Build context for evaluation
        eval_context = {
            "step": {
                "name": step.name,
                "agent_id": step.agent_id,
                "publishes_data": step.publishes_data,
                "data_regions": step.data_regions,
            },
            "result": {
                "success": result.success,
                "output_classification": (
                    result.output_classification.value
                    if result.output_classification
                    else None
                ),
                "export_destinations": result.export_destinations,
                "artifacts": result.artifacts,
            },
            "context": context.to_policy_doc(),
            "pipeline": context.pipeline.to_policy_doc(),
            "run": context.run_config.to_policy_doc(),
            "namespace": context.run_config.namespace,
        }

        # Evaluate YAML rules
        if self.config.yaml_rules_enabled:
            yaml_results = self._yaml_parser.evaluate(
                eval_context,
                evaluation_point,
                namespace=context.run_config.namespace,
                pipeline_name=context.pipeline.metadata.name,
            )
            evaluated_policies.append("yaml_rules")

            for rule, matched, message in yaml_results:
                if matched:
                    self._apply_rule_result(rule, message, reasons, approvals, warnings)

        # Evaluate OPA policies
        if self.config.opa_enabled and self._opa_client:
            try:
                opa_input = self._opa_client.format_input(
                    pipeline=context.pipeline,
                    run_config=context.run_config,
                    step=step,
                    context=context,
                    step_result=result,
                )
                opa_result = await self._opa_client.query(
                    "data.greenlang.policies.post_step", opa_input
                )
                opa_decision = self._opa_client.parse_decision(opa_result, evaluation_point)

                reasons.extend(opa_decision.reasons)
                approvals.extend(opa_decision.required_approvals)
                evaluated_policies.append("opa")
            except OPAError as e:
                logger.warning(f"OPA evaluation failed for post-step {step.name}: {e}")
                if self.config.strict_mode:
                    reasons.append(PolicyReason(
                        rule_name="opa_unavailable",
                        message=f"OPA evaluation failed: {e}",
                        severity=PolicySeverity.ERROR,
                        action=PolicyAction.DENY,
                    ))

        # Check export controls
        self._check_export_controls(step, result, context, reasons)

        # Check artifact classification
        self._check_artifact_classification(result, context, reasons, warnings)

        # Build decision
        allowed = len(reasons) == 0
        evaluation_time_ms = (time.perf_counter() - start_time) * 1000

        decision = PolicyDecision(
            allowed=allowed,
            reasons=reasons,
            required_approvals=approvals,
            policy_version=self._get_policy_version(),
            evaluation_time_ms=evaluation_time_ms,
            evaluation_point=evaluation_point,
            evaluated_policies=evaluated_policies,
            warnings=warnings,
        )
        decision.provenance_hash = decision.compute_provenance_hash()

        self._log_audit(decision, step.name, context.run_id)

        return decision

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_budget(self, namespace: str) -> Dict[str, Any]:
        """Get cost budget for namespace."""
        bundle = self.get_effective_bundle(namespace)
        if bundle and namespace in bundle.cost_budgets:
            budget = bundle.cost_budgets[namespace]
            return budget.model_dump()
        return {"max_cost_usd": float("inf"), "enforce_on_estimate": False}

    def _check_cost_budget(
        self,
        run_config: RunConfig,
        reasons: List[PolicyReason],
        approvals: List[ApprovalRequirement],
        warnings: List[str],
    ) -> None:
        """Check cost budget compliance."""
        bundle = self.get_effective_bundle(run_config.namespace)
        if not bundle:
            return

        budget = bundle.cost_budgets.get(run_config.namespace)
        if not budget:
            return

        estimated = run_config.estimated_cost_usd or 0
        max_cost = run_config.max_cost_usd or budget.max_cost_usd

        if estimated > max_cost:
            if budget.allow_override:
                approvals.append(ApprovalRequirement(
                    approval_type=budget.override_approval_type,
                    reason=f"Run estimated cost ${estimated:.2f} exceeds budget ${max_cost:.2f}",
                ))
            else:
                reasons.append(PolicyReason(
                    rule_name="cost_budget_exceeded",
                    message=f"Estimated cost ${estimated:.2f} exceeds budget ${max_cost:.2f}",
                    severity=PolicySeverity.ERROR,
                    action=PolicyAction.DENY,
                    details={"estimated": estimated, "budget": max_cost},
                ))
        elif estimated > max_cost * (budget.warn_threshold_percent / 100):
            warnings.append(
                f"Estimated cost ${estimated:.2f} is at {estimated/max_cost*100:.0f}% of budget"
            )

    def _check_accumulated_cost(
        self,
        context: ExecutionContext,
        reasons: List[PolicyReason],
        warnings: List[str],
    ) -> None:
        """Check accumulated cost against budget."""
        max_cost = context.run_config.max_cost_usd
        if max_cost and context.total_cost_usd >= max_cost:
            reasons.append(PolicyReason(
                rule_name="cost_budget_exhausted",
                message=f"Accumulated cost ${context.total_cost_usd:.2f} reached budget ${max_cost:.2f}",
                severity=PolicySeverity.ERROR,
                action=PolicyAction.DENY,
            ))

    def _check_data_residency(
        self,
        pipeline: PipelineDefinition,
        run_config: RunConfig,
        reasons: List[PolicyReason],
    ) -> None:
        """Check data residency compliance."""
        for bundle in self._bundles.values():
            for rule in bundle.residency_rules:
                # Check if classification applies
                if run_config.classification_level not in rule.applies_to_classification:
                    continue

                # Check allowed regions
                for step in pipeline.spec.steps:
                    for region in step.data_regions:
                        if rule.denied_regions and region in rule.denied_regions:
                            reasons.append(PolicyReason(
                                rule_name=f"residency_{rule.name}",
                                message=f"Step {step.name} uses denied region: {region}",
                                severity=PolicySeverity.ERROR,
                                action=PolicyAction.DENY,
                            ))
                        elif rule.allowed_regions and region not in rule.allowed_regions:
                            reasons.append(PolicyReason(
                                rule_name=f"residency_{rule.name}",
                                message=rule.message or f"Region {region} not allowed",
                                severity=PolicySeverity.ERROR,
                                action=PolicyAction.DENY,
                            ))

    def _check_step_policies(
        self,
        step: StepDefinition,
        context: ExecutionContext,
        reasons: List[PolicyReason],
        approvals: List[ApprovalRequirement],
    ) -> None:
        """Check step-specific policy requirements."""
        # PII access control
        if step.accesses_pii:
            if "pii_access" not in context.permissions:
                reasons.append(PolicyReason(
                    rule_name="pii_access_denied",
                    message=f"Step {step.name} requires PII access permission",
                    severity=PolicySeverity.ERROR,
                    action=PolicyAction.DENY,
                ))

        # Step requires approval flag
        if step.requires_approval:
            approvals.append(ApprovalRequirement(
                approval_type=ApprovalType.MANAGER,
                reason=f"Step {step.name} requires approval before execution",
            ))

        # Production namespace checks
        if context.run_config.namespace == "production":
            if step.publishes_data and "publish_production" not in context.permissions:
                reasons.append(PolicyReason(
                    rule_name="production_publish_denied",
                    message=f"Step {step.name} cannot publish data in production without permission",
                    severity=PolicySeverity.ERROR,
                    action=PolicyAction.DENY,
                ))

    def _check_export_controls(
        self,
        step: StepDefinition,
        result: StepResult,
        context: ExecutionContext,
        reasons: List[PolicyReason],
    ) -> None:
        """Check export control compliance."""
        if not result.export_destinations:
            return

        classification = result.output_classification or DataClassification.INTERNAL

        for destination in result.export_destinations:
            # Restricted data cannot be exported externally
            if classification == DataClassification.RESTRICTED:
                if "external" in destination.lower() or not destination.startswith("internal"):
                    reasons.append(PolicyReason(
                        rule_name="export_control_restricted",
                        message=f"Restricted data cannot be exported to {destination}",
                        severity=PolicySeverity.ERROR,
                        action=PolicyAction.DENY,
                    ))

            # Check allowed export destinations from run config
            allowed_regions = context.run_config.data_regions
            if allowed_regions:
                region = self._extract_region(destination)
                if region and region not in allowed_regions:
                    reasons.append(PolicyReason(
                        rule_name="export_region_denied",
                        message=f"Export to region {region} not allowed",
                        severity=PolicySeverity.ERROR,
                        action=PolicyAction.DENY,
                    ))

    def _check_artifact_classification(
        self,
        result: StepResult,
        context: ExecutionContext,
        reasons: List[PolicyReason],
        warnings: List[str],
    ) -> None:
        """Check artifact classification compliance."""
        output_class = result.output_classification
        run_class = context.run_config.classification_level

        if output_class and run_class:
            # Output cannot be more classified than run allows
            class_order = [
                DataClassification.PUBLIC,
                DataClassification.INTERNAL,
                DataClassification.CONFIDENTIAL,
                DataClassification.RESTRICTED,
            ]

            output_level = class_order.index(output_class) if output_class in class_order else 0
            run_level = class_order.index(run_class) if run_class in class_order else 0

            if output_level > run_level:
                warnings.append(
                    f"Step output classification ({output_class.value}) "
                    f"exceeds run classification ({run_class.value})"
                )

    def _extract_region(self, destination: str) -> Optional[str]:
        """Extract region from destination string."""
        # Simple region extraction - can be enhanced
        patterns = [
            r"us-(?:east|west|central)-\d",
            r"eu-(?:west|central|north|south)-\d",
            r"ap-(?:northeast|southeast|south)-\d",
        ]
        for pattern in patterns:
            match = re.search(pattern, destination.lower())
            if match:
                return match.group(0)
        return None

    def _apply_rule_result(
        self,
        rule: YAMLRule,
        message: Optional[str],
        reasons: List[PolicyReason],
        approvals: List[ApprovalRequirement],
        warnings: List[str],
    ) -> None:
        """Apply rule evaluation result to decision."""
        msg = message or rule.description or f"Rule {rule.name} triggered"

        if rule.action == PolicyAction.DENY:
            reasons.append(PolicyReason(
                rule_name=rule.name,
                message=msg,
                severity=rule.severity,
                action=rule.action,
            ))
        elif rule.action == PolicyAction.REQUIRE_APPROVAL:
            approvals.append(ApprovalRequirement(
                approval_type=rule.approval_type or ApprovalType.MANAGER,
                reason=msg,
                approver_role=rule.approval_role,
            ))
        elif rule.action == PolicyAction.WARN:
            warnings.append(msg)

    def _get_policy_version(self) -> str:
        """Get current policy version."""
        if self._bundles:
            versions = [b.version for b in self._bundles.values()]
            return max(versions)
        return self._yaml_parser._version

    def _get_cache_key(self, point: EvaluationPoint, *args: str) -> str:
        """Generate cache key."""
        key_parts = [point.value] + list(args)
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()

    def _get_cached_decision(self, cache_key: str) -> Optional[PolicyDecision]:
        """Get cached decision if valid."""
        if not self.config.cache_enabled:
            return None

        if cache_key in self._cache:
            decision, cached_at = self._cache[cache_key]
            if time.time() - cached_at < self.config.cache_ttl_seconds:
                return decision
            del self._cache[cache_key]

        return None

    def _cache_decision(self, cache_key: str, decision: PolicyDecision) -> None:
        """Cache a decision."""
        if self.config.cache_enabled:
            self._cache[cache_key] = (decision, time.time())

    def _log_audit(
        self,
        decision: PolicyDecision,
        resource_name: str,
        run_id: str,
    ) -> None:
        """Log decision for audit."""
        if not self.config.audit_all_decisions and decision.allowed:
            return

        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "resource": resource_name,
            "evaluation_point": decision.evaluation_point.value,
            "allowed": decision.allowed,
            "reason_count": len(decision.reasons),
            "approval_count": len(decision.required_approvals),
            "provenance_hash": decision.provenance_hash,
            "policy_version": decision.policy_version,
            "evaluation_time_ms": decision.evaluation_time_ms,
        }

        self._audit_log.append(audit_entry)

        # Trim audit log if too large
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]

        logger.info(
            f"Policy decision: {decision.evaluation_point.value} "
            f"allowed={decision.allowed} resource={resource_name}"
        )

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit entries."""
        return self._audit_log[-limit:]

    def clear_cache(self) -> None:
        """Clear decision cache."""
        self._cache.clear()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "PolicyEngine",
    "PolicyEngineConfig",
    "OPAClient",
    "YAMLRulesParser",
    # Decision models
    "PolicyDecision",
    "PolicyReason",
    "ApprovalRequirement",
    # Enums
    "PolicyAction",
    "PolicySeverity",
    "EvaluationPoint",
    "ApprovalType",
    # Rule models
    "YAMLRule",
    "YAMLRuleSet",
    "CostBudget",
    "DataResidencyRule",
    "PolicyBundle",
    # Exceptions
    "OPAError",
]
