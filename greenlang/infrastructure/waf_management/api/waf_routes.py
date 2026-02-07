# -*- coding: utf-8 -*-
"""
WAF Management REST API Router - SEC-010

FastAPI APIRouter providing REST endpoints for WAF rule management,
attack detection, and DDoS protection.

Endpoints:
    GET    /api/v1/secops/waf/rules              - List WAF rules
    POST   /api/v1/secops/waf/rules              - Create WAF rule
    GET    /api/v1/secops/waf/rules/{id}         - Get rule details
    PUT    /api/v1/secops/waf/rules/{id}         - Update rule
    DELETE /api/v1/secops/waf/rules/{id}         - Delete rule
    POST   /api/v1/secops/waf/rules/{id}/test    - Test rule
    POST   /api/v1/secops/waf/rules/{id}/deploy  - Deploy rule
    POST   /api/v1/secops/waf/rules/{id}/disable - Disable rule
    GET    /api/v1/secops/waf/attacks            - List detected attacks
    GET    /api/v1/secops/waf/attacks/{id}       - Get attack details
    POST   /api/v1/secops/waf/attacks/{id}/mitigate - Manual mitigation
    GET    /api/v1/secops/waf/metrics            - WAF/DDoS metrics
    GET    /api/v1/secops/waf/shield/status      - Shield Advanced status

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.waf_management.api.waf_routes import waf_router
    >>> app = FastAPI()
    >>> app.include_router(waf_router)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, status, Body
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    Query = None  # type: ignore[assignment]
    status = None  # type: ignore[assignment]
    Body = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    Field = None  # type: ignore[assignment]

from greenlang.infrastructure.waf_management.config import WAFConfig, get_config
from greenlang.infrastructure.waf_management.models import (
    Attack,
    AttackSeverity,
    AttackType,
    MitigationStatus,
    RuleAction,
    RuleStatus,
    RuleType,
    WAFRule,
)


# ---------------------------------------------------------------------------
# Pydantic Request/Response Models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class RuleConditionRequest(BaseModel):
        """Request model for rule conditions."""

        field: str = Field(..., description="Request field to inspect")
        operator: str = Field(default="equals", description="Comparison operator")
        values: List[str] = Field(default_factory=list, description="Values to match")
        negated: bool = Field(default=False, description="Invert match result")
        transform: Optional[str] = Field(default=None, description="Text transformation")

    class CreateRuleRequest(BaseModel):
        """Request model for creating a WAF rule."""

        name: str = Field(..., min_length=2, max_length=128, description="Rule name")
        rule_type: str = Field(..., description="Rule type (rate_limit, geo_block, etc)")
        priority: int = Field(default=100, ge=0, le=10000, description="Evaluation priority")
        action: str = Field(default="block", description="Action (allow, block, count, captcha)")
        description: str = Field(default="", max_length=2048, description="Rule description")
        enabled: bool = Field(default=True, description="Whether rule is active")
        rate_limit_threshold: int = Field(default=2000, description="Rate limit threshold")
        rate_limit_window_seconds: int = Field(default=300, description="Rate limit window")
        blocked_countries: List[str] = Field(default_factory=list, description="Country codes to block")
        ip_set_arn: Optional[str] = Field(default=None, description="IP set ARN")
        regex_pattern: Optional[str] = Field(default=None, description="Custom regex pattern")
        managed_rule_group: Optional[str] = Field(default=None, description="AWS managed rule group")
        conditions: List[RuleConditionRequest] = Field(default_factory=list, description="Rule conditions")
        metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class UpdateRuleRequest(BaseModel):
        """Request model for updating a WAF rule."""

        name: Optional[str] = Field(default=None, max_length=128)
        priority: Optional[int] = Field(default=None, ge=0, le=10000)
        action: Optional[str] = Field(default=None)
        description: Optional[str] = Field(default=None, max_length=2048)
        enabled: Optional[bool] = Field(default=None)
        rate_limit_threshold: Optional[int] = Field(default=None)
        rate_limit_window_seconds: Optional[int] = Field(default=None)
        blocked_countries: Optional[List[str]] = Field(default=None)
        regex_pattern: Optional[str] = Field(default=None)
        conditions: Optional[List[RuleConditionRequest]] = Field(default=None)
        metadata: Optional[Dict[str, Any]] = Field(default=None)

    class RuleResponse(BaseModel):
        """Response model for a WAF rule."""

        id: str
        name: str
        rule_type: str
        priority: int
        action: str
        description: str
        enabled: bool
        status: str
        rate_limit_threshold: int
        rate_limit_window_seconds: int
        blocked_countries: List[str]
        ip_set_arn: Optional[str]
        regex_pattern: Optional[str]
        managed_rule_group: Optional[str]
        aws_rule_id: Optional[str]
        created_at: datetime
        updated_at: datetime
        deployed_at: Optional[datetime]
        created_by: str
        metrics: Dict[str, Any]
        metadata: Dict[str, Any]

    class RuleListResponse(BaseModel):
        """Response model for paginated rule list."""

        items: List[RuleResponse]
        total: int
        page: int
        page_size: int
        total_pages: int
        has_next: bool
        has_prev: bool

    class TestRuleRequest(BaseModel):
        """Request model for testing a rule."""

        test_requests: Optional[List[Dict[str, Any]]] = Field(
            default=None, description="Custom test requests"
        )
        include_legitimate: bool = Field(default=True, description="Include legitimate traffic tests")
        malicious_count: Optional[int] = Field(default=None, description="Max malicious requests")

    class TestRuleResponse(BaseModel):
        """Response model for rule test results."""

        rule_name: str
        rule_type: str
        total_requests: int
        total_matched: int
        true_positives: int
        false_positives: int
        false_negatives: int
        detection_rate: float
        false_positive_rate: float
        accuracy: float
        precision: float
        recall: float
        f1_score: float
        average_latency_us: float
        p99_latency_us: float
        recommendations: List[str]

    class DeployRuleRequest(BaseModel):
        """Request model for deploying a rule."""

        web_acl_id: Optional[str] = Field(default=None, description="Web ACL ID to deploy to")

    class AttackResponse(BaseModel):
        """Response model for an attack."""

        id: str
        attack_type: str
        severity: str
        source_ips: List[str]
        target_endpoints: List[str]
        requests_per_second: int
        total_requests: int
        bytes_per_second: int
        started_at: datetime
        detected_at: datetime
        mitigated_at: Optional[datetime]
        ended_at: Optional[datetime]
        status: str
        detection_source: str
        attack_signature: Optional[str]
        geographic_distribution: Dict[str, int]
        metadata: Dict[str, Any]

    class AttackListResponse(BaseModel):
        """Response model for paginated attack list."""

        items: List[AttackResponse]
        total: int
        page: int
        page_size: int
        active_count: int
        mitigated_count: int

    class MitigateAttackRequest(BaseModel):
        """Request model for manual attack mitigation."""

        actions: List[str] = Field(
            ..., description="Mitigation actions to take"
        )
        block_ips: List[str] = Field(default_factory=list, description="IPs to block")
        geo_block_countries: List[str] = Field(default_factory=list, description="Countries to block")
        engage_shield_drt: bool = Field(default=False, description="Engage AWS Shield DRT")

    class MitigationResponse(BaseModel):
        """Response model for mitigation result."""

        attack_id: str
        status: str
        actions_taken: List[Dict[str, Any]]
        effectiveness_score: float
        traffic_reduction_percent: float
        duration_seconds: float
        shield_engaged: bool
        recommendations: List[str]

    class MetricsResponse(BaseModel):
        """Response model for WAF/DDoS metrics."""

        timestamp: datetime
        waf: Dict[str, Any]
        ddos: Dict[str, Any]
        traffic: Dict[str, Any]
        shield: Dict[str, Any]

    class ShieldStatusResponse(BaseModel):
        """Response model for Shield status."""

        subscription_active: bool
        subscription_start: Optional[datetime]
        auto_renew: bool
        protections_count: int
        protections: List[Dict[str, Any]]
        proactive_engagement_enabled: bool
        attack_statistics: Dict[str, Any]


# ---------------------------------------------------------------------------
# In-Memory Storage (for demonstration)
# ---------------------------------------------------------------------------

# In production, these would be stored in PostgreSQL
_rules_store: Dict[str, WAFRule] = {}
_attacks_store: Dict[str, Attack] = {}


# ---------------------------------------------------------------------------
# Dependency Injection
# ---------------------------------------------------------------------------


def _get_config() -> WAFConfig:
    """FastAPI dependency that provides the WAFConfig."""
    return get_config()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _rule_to_response(rule: WAFRule) -> RuleResponse:
    """Convert WAFRule to API response."""
    return RuleResponse(
        id=rule.id,
        name=rule.name,
        rule_type=rule.rule_type.value,
        priority=rule.priority,
        action=rule.action.value,
        description=rule.description,
        enabled=rule.enabled,
        status=rule.status.value,
        rate_limit_threshold=rule.rate_limit_threshold,
        rate_limit_window_seconds=rule.rate_limit_window_seconds,
        blocked_countries=rule.blocked_countries,
        ip_set_arn=rule.ip_set_arn,
        regex_pattern=rule.regex_pattern,
        managed_rule_group=rule.managed_rule_group,
        aws_rule_id=rule.aws_rule_id,
        created_at=rule.created_at,
        updated_at=rule.updated_at,
        deployed_at=rule.deployed_at,
        created_by=rule.created_by,
        metrics=rule.metrics.model_dump(),
        metadata=rule.metadata,
    )


def _attack_to_response(attack: Attack) -> AttackResponse:
    """Convert Attack to API response."""
    return AttackResponse(
        id=attack.id,
        attack_type=attack.attack_type.value,
        severity=attack.severity.value,
        source_ips=attack.source_ips,
        target_endpoints=attack.target_endpoints,
        requests_per_second=attack.requests_per_second,
        total_requests=attack.total_requests,
        bytes_per_second=attack.bytes_per_second,
        started_at=attack.started_at,
        detected_at=attack.detected_at,
        mitigated_at=attack.mitigated_at,
        ended_at=attack.ended_at,
        status=attack.status.value,
        detection_source=attack.detection_source,
        attack_signature=attack.attack_signature,
        geographic_distribution=attack.geographic_distribution,
        metadata=attack.metadata,
    )


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    waf_router = APIRouter(
        prefix="/api/v1/secops/waf",
        tags=["WAF Management"],
        responses={
            400: {"description": "Bad Request"},
            404: {"description": "Not Found"},
            422: {"description": "Validation Error"},
            500: {"description": "Internal Server Error"},
        },
    )

    # -- List Rules ---------------------------------------------------------

    @waf_router.get(
        "/rules",
        response_model=RuleListResponse,
        summary="List WAF rules",
        description="Retrieve a paginated list of WAF rules.",
        operation_id="list_waf_rules",
    )
    async def list_rules(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        rule_type: Optional[str] = Query(None, description="Filter by rule type"),
        rule_status: Optional[str] = Query(None, alias="status", description="Filter by status"),
        enabled: Optional[bool] = Query(None, description="Filter by enabled state"),
        config: WAFConfig = Depends(_get_config),
    ) -> RuleListResponse:
        """List WAF rules with pagination and filters."""
        rules = list(_rules_store.values())

        # Apply filters
        if rule_type:
            rules = [r for r in rules if r.rule_type.value == rule_type]
        if rule_status:
            rules = [r for r in rules if r.status.value == rule_status]
        if enabled is not None:
            rules = [r for r in rules if r.enabled == enabled]

        # Sort by priority
        rules.sort(key=lambda r: r.priority)

        # Paginate
        total = len(rules)
        offset = (page - 1) * page_size
        rules = rules[offset:offset + page_size]
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0

        return RuleListResponse(
            items=[_rule_to_response(r) for r in rules],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )

    # -- Create Rule --------------------------------------------------------

    @waf_router.post(
        "/rules",
        response_model=RuleResponse,
        status_code=201,
        summary="Create WAF rule",
        description="Create a new WAF rule.",
        operation_id="create_waf_rule",
    )
    async def create_rule(
        request: CreateRuleRequest,
        config: WAFConfig = Depends(_get_config),
    ) -> RuleResponse:
        """Create a new WAF rule."""
        try:
            rule_type = RuleType(request.rule_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid rule_type '{request.rule_type}'. "
                       f"Valid types: {[t.value for t in RuleType]}",
            )

        try:
            action = RuleAction(request.action.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid action '{request.action}'. "
                       f"Valid actions: {[a.value for a in RuleAction]}",
            )

        # Check for duplicate name
        for existing in _rules_store.values():
            if existing.name == request.name:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Rule with name '{request.name}' already exists",
                )

        rule = WAFRule(
            name=request.name,
            rule_type=rule_type,
            priority=request.priority,
            action=action,
            description=request.description,
            enabled=request.enabled,
            rate_limit_threshold=request.rate_limit_threshold,
            rate_limit_window_seconds=request.rate_limit_window_seconds,
            blocked_countries=request.blocked_countries,
            ip_set_arn=request.ip_set_arn,
            regex_pattern=request.regex_pattern,
            managed_rule_group=request.managed_rule_group,
            metadata=request.metadata,
        )

        _rules_store[rule.id] = rule
        logger.info("Created WAF rule: name=%s, id=%s", rule.name, rule.id)

        return _rule_to_response(rule)

    # -- Get Rule -----------------------------------------------------------

    @waf_router.get(
        "/rules/{rule_id}",
        response_model=RuleResponse,
        summary="Get WAF rule",
        description="Retrieve a specific WAF rule by ID.",
        operation_id="get_waf_rule",
    )
    async def get_rule(
        rule_id: str,
        config: WAFConfig = Depends(_get_config),
    ) -> RuleResponse:
        """Get a WAF rule by ID."""
        rule = _rules_store.get(rule_id)
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rule '{rule_id}' not found",
            )
        return _rule_to_response(rule)

    # -- Update Rule --------------------------------------------------------

    @waf_router.put(
        "/rules/{rule_id}",
        response_model=RuleResponse,
        summary="Update WAF rule",
        description="Update an existing WAF rule.",
        operation_id="update_waf_rule",
    )
    async def update_rule(
        rule_id: str,
        request: UpdateRuleRequest,
        config: WAFConfig = Depends(_get_config),
    ) -> RuleResponse:
        """Update a WAF rule."""
        rule = _rules_store.get(rule_id)
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rule '{rule_id}' not found",
            )

        # Update fields
        updates = request.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No fields provided for update",
            )

        for field, value in updates.items():
            if field == "action":
                try:
                    value = RuleAction(value.lower())
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid action '{value}'",
                    )
            setattr(rule, field, value)

        rule.updated_at = datetime.now(timezone.utc)
        logger.info("Updated WAF rule: id=%s", rule_id)

        return _rule_to_response(rule)

    # -- Delete Rule --------------------------------------------------------

    @waf_router.delete(
        "/rules/{rule_id}",
        status_code=204,
        summary="Delete WAF rule",
        description="Delete a WAF rule.",
        operation_id="delete_waf_rule",
    )
    async def delete_rule(
        rule_id: str,
        config: WAFConfig = Depends(_get_config),
    ) -> None:
        """Delete a WAF rule."""
        if rule_id not in _rules_store:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rule '{rule_id}' not found",
            )

        del _rules_store[rule_id]
        logger.info("Deleted WAF rule: id=%s", rule_id)

    # -- Test Rule ----------------------------------------------------------

    @waf_router.post(
        "/rules/{rule_id}/test",
        response_model=TestRuleResponse,
        summary="Test WAF rule",
        description="Test a WAF rule against sample requests.",
        operation_id="test_waf_rule",
    )
    async def test_rule(
        rule_id: str,
        request: TestRuleRequest,
        config: WAFConfig = Depends(_get_config),
    ) -> TestRuleResponse:
        """Test a WAF rule."""
        from greenlang.infrastructure.waf_management.rule_tester import WAFRuleTester

        rule = _rules_store.get(rule_id)
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rule '{rule_id}' not found",
            )

        tester = WAFRuleTester(config)

        # Generate test requests
        test_requests = tester.generate_test_requests(
            rule.rule_type.value,
            include_legitimate=request.include_legitimate,
            malicious_count=request.malicious_count,
        )

        # Run tests
        import asyncio
        results = asyncio.get_event_loop().run_until_complete(
            tester.test_rule(rule, test_requests)
        )

        # Generate report
        report = tester.generate_test_report(results, rule)

        return TestRuleResponse(
            rule_name=report.rule_name,
            rule_type=report.rule_type,
            total_requests=report.total_requests,
            total_matched=report.total_matched,
            true_positives=report.true_positives,
            false_positives=report.false_positives,
            false_negatives=report.false_negatives,
            detection_rate=report.detection_rate,
            false_positive_rate=report.false_positive_rate,
            accuracy=report.accuracy,
            precision=report.precision,
            recall=report.recall,
            f1_score=report.f1_score,
            average_latency_us=report.average_latency_us,
            p99_latency_us=report.p99_latency_us,
            recommendations=report.recommendations,
        )

    # -- Deploy Rule --------------------------------------------------------

    @waf_router.post(
        "/rules/{rule_id}/deploy",
        response_model=RuleResponse,
        summary="Deploy WAF rule",
        description="Deploy a WAF rule to AWS WAF.",
        operation_id="deploy_waf_rule",
    )
    async def deploy_rule(
        rule_id: str,
        request: DeployRuleRequest,
        config: WAFConfig = Depends(_get_config),
    ) -> RuleResponse:
        """Deploy a WAF rule to AWS WAF."""
        from greenlang.infrastructure.waf_management.rule_builder import WAFRuleBuilder

        rule = _rules_store.get(rule_id)
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rule '{rule_id}' not found",
            )

        builder = WAFRuleBuilder(config)

        # Validate rule
        validation = builder.validate_rule(rule)
        if not validation.is_valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Rule validation failed: {', '.join(validation.errors)}",
            )

        try:
            updated_rule = await builder.deploy_rule(
                rule,
                web_acl_id=request.web_acl_id,
            )
            _rules_store[rule.id] = updated_rule
            return _rule_to_response(updated_rule)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

    # -- Disable Rule -------------------------------------------------------

    @waf_router.post(
        "/rules/{rule_id}/disable",
        response_model=RuleResponse,
        summary="Disable WAF rule",
        description="Disable a WAF rule without deleting it.",
        operation_id="disable_waf_rule",
    )
    async def disable_rule(
        rule_id: str,
        config: WAFConfig = Depends(_get_config),
    ) -> RuleResponse:
        """Disable a WAF rule."""
        rule = _rules_store.get(rule_id)
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rule '{rule_id}' not found",
            )

        rule.enabled = False
        rule.status = RuleStatus.DISABLED
        rule.updated_at = datetime.now(timezone.utc)
        logger.info("Disabled WAF rule: id=%s", rule_id)

        return _rule_to_response(rule)

    # -- List Attacks -------------------------------------------------------

    @waf_router.get(
        "/attacks",
        response_model=AttackListResponse,
        summary="List detected attacks",
        description="Retrieve a list of detected attacks.",
        operation_id="list_attacks",
    )
    async def list_attacks(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        attack_type: Optional[str] = Query(None, description="Filter by attack type"),
        attack_status: Optional[str] = Query(None, alias="status", description="Filter by status"),
        severity: Optional[str] = Query(None, description="Filter by severity"),
        config: WAFConfig = Depends(_get_config),
    ) -> AttackListResponse:
        """List detected attacks."""
        attacks = list(_attacks_store.values())

        # Apply filters
        if attack_type:
            attacks = [a for a in attacks if a.attack_type.value == attack_type]
        if attack_status:
            attacks = [a for a in attacks if a.status.value == attack_status]
        if severity:
            attacks = [a for a in attacks if a.severity.value == severity]

        # Sort by detected_at (most recent first)
        attacks.sort(key=lambda a: a.detected_at, reverse=True)

        # Calculate counts
        active_count = sum(1 for a in attacks if a.status in (
            MitigationStatus.PENDING, MitigationStatus.IN_PROGRESS
        ))
        mitigated_count = sum(1 for a in attacks if a.status == MitigationStatus.MITIGATED)

        # Paginate
        total = len(attacks)
        offset = (page - 1) * page_size
        attacks = attacks[offset:offset + page_size]

        return AttackListResponse(
            items=[_attack_to_response(a) for a in attacks],
            total=total,
            page=page,
            page_size=page_size,
            active_count=active_count,
            mitigated_count=mitigated_count,
        )

    # -- Get Attack ---------------------------------------------------------

    @waf_router.get(
        "/attacks/{attack_id}",
        response_model=AttackResponse,
        summary="Get attack details",
        description="Retrieve details of a specific attack.",
        operation_id="get_attack",
    )
    async def get_attack(
        attack_id: str,
        config: WAFConfig = Depends(_get_config),
    ) -> AttackResponse:
        """Get attack by ID."""
        attack = _attacks_store.get(attack_id)
        if not attack:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Attack '{attack_id}' not found",
            )
        return _attack_to_response(attack)

    # -- Mitigate Attack ----------------------------------------------------

    @waf_router.post(
        "/attacks/{attack_id}/mitigate",
        response_model=MitigationResponse,
        summary="Mitigate attack",
        description="Manually trigger attack mitigation.",
        operation_id="mitigate_attack",
    )
    async def mitigate_attack(
        attack_id: str,
        request: MitigateAttackRequest,
        config: WAFConfig = Depends(_get_config),
    ) -> MitigationResponse:
        """Manually mitigate an attack."""
        from greenlang.infrastructure.waf_management.anomaly_detector import AnomalyDetector

        attack = _attacks_store.get(attack_id)
        if not attack:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Attack '{attack_id}' not found",
            )

        if attack.status == MitigationStatus.MITIGATED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Attack has already been mitigated",
            )

        detector = AnomalyDetector(config)
        result = await detector.auto_mitigate(attack)

        _attacks_store[attack_id] = attack

        return MitigationResponse(
            attack_id=result.attack_id,
            status=result.status.value,
            actions_taken=[a.model_dump() if hasattr(a, 'model_dump') else a.__dict__ for a in result.actions_taken],
            effectiveness_score=result.effectiveness_score,
            traffic_reduction_percent=result.traffic_reduction_percent,
            duration_seconds=result.duration_seconds,
            shield_engaged=result.shield_engaged,
            recommendations=result.recommendations,
        )

    # -- Get Metrics --------------------------------------------------------

    @waf_router.get(
        "/metrics",
        response_model=MetricsResponse,
        summary="Get WAF/DDoS metrics",
        description="Retrieve current WAF and DDoS protection metrics.",
        operation_id="get_waf_metrics",
    )
    async def get_metrics(
        config: WAFConfig = Depends(_get_config),
    ) -> MetricsResponse:
        """Get WAF and DDoS metrics."""
        # In production, these would be collected from Prometheus or CloudWatch
        now = datetime.now(timezone.utc)

        return MetricsResponse(
            timestamp=now,
            waf={
                "rules_total": len(_rules_store),
                "rules_active": sum(1 for r in _rules_store.values() if r.enabled),
                "requests_evaluated_total": 0,
                "requests_blocked_total": 0,
                "false_positives_total": 0,
            },
            ddos={
                "attacks_total": len(_attacks_store),
                "attacks_active": sum(1 for a in _attacks_store.values()
                    if a.status in (MitigationStatus.PENDING, MitigationStatus.IN_PROGRESS)),
                "attacks_mitigated": sum(1 for a in _attacks_store.values()
                    if a.status == MitigationStatus.MITIGATED),
            },
            traffic={
                "requests_per_second": 0.0,
                "blocked_per_second": 0.0,
                "latency_p99_ms": 0.0,
            },
            shield={
                "subscription_active": config.shield_enabled,
                "protections_count": len(config.shield_resource_arns),
            },
        )

    # -- Get Shield Status --------------------------------------------------

    @waf_router.get(
        "/shield/status",
        response_model=ShieldStatusResponse,
        summary="Get Shield status",
        description="Retrieve AWS Shield Advanced status.",
        operation_id="get_shield_status",
    )
    async def get_shield_status(
        config: WAFConfig = Depends(_get_config),
    ) -> ShieldStatusResponse:
        """Get AWS Shield Advanced status."""
        from greenlang.infrastructure.waf_management.shield_manager import ShieldManager

        manager = ShieldManager(config)

        try:
            subscription = await manager.get_subscription_state()
            protections = await manager.list_protections()
            statistics = await manager.get_attack_statistics()

            return ShieldStatusResponse(
                subscription_active=subscription.is_active,
                subscription_start=subscription.subscription_start,
                auto_renew=subscription.auto_renew,
                protections_count=len(protections),
                protections=[
                    {
                        "id": p.id,
                        "resource_arn": p.resource_arn,
                        "name": p.protection_name,
                    }
                    for p in protections
                ],
                proactive_engagement_enabled=config.shield_proactive_engagement,
                attack_statistics=statistics.to_dict(),
            )
        except Exception as e:
            logger.warning("Failed to get Shield status: %s", str(e))
            return ShieldStatusResponse(
                subscription_active=False,
                subscription_start=None,
                auto_renew=False,
                protections_count=0,
                protections=[],
                proactive_engagement_enabled=False,
                attack_statistics={},
            )

    # Apply route protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import protect_router
        protect_router(waf_router)
    except ImportError:
        pass

else:
    waf_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - waf_router is None")


__all__ = ["waf_router"]
