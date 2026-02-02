# -*- coding: utf-8 -*-
"""
Unit tests for the PolicyEngine module.

Tests cover:
- YAML rules parsing and evaluation
- OPA client formatting
- Policy decision making
- Cost budget enforcement
- Data residency rules
- Pre-run, pre-step, and post-step evaluation

Author: GreenLang Team
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from greenlang.orchestrator.governance.policy_engine import (
    PolicyEngine,
    PolicyEngineConfig,
    PolicyDecision,
    PolicyReason,
    ApprovalRequirement,
    PolicyAction,
    PolicySeverity,
    EvaluationPoint,
    ApprovalType,
    YAMLRule,
    YAMLRuleSet,
    YAMLRulesParser,
    CostBudget,
    DataResidencyRule,
    PolicyBundle,
    OPAClient,
)
from greenlang.orchestrator.pipeline_schema import (
    PipelineDefinition,
    PipelineMetadata,
    PipelineSpec,
    StepDefinition,
    RunConfig,
    StepResult,
    ExecutionContext,
    DataClassification,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_pipeline() -> PipelineDefinition:
    """Create a simple pipeline for testing."""
    return PipelineDefinition(
        apiVersion="greenlang/v1",
        kind="Pipeline",
        metadata=PipelineMetadata(
            name="test-pipeline",
            namespace="default",
            version="1.0.0",
            owner="test-user",
            team="sustainability",
        ),
        spec=PipelineSpec(
            steps=[
                StepDefinition(
                    id="step1",
                    agent="GL-DATA-X-001",
                    publishes_data=False,
                    accesses_pii=False,
                ),
                StepDefinition(
                    id="step2",
                    agent="GL-CALC-A-001",
                    dependsOn=["step1"],
                    publishes_data=True,
                ),
            ]
        ),
    )


@pytest.fixture
def production_pipeline() -> PipelineDefinition:
    """Create a production pipeline for testing."""
    return PipelineDefinition(
        apiVersion="greenlang/v1",
        kind="Pipeline",
        metadata=PipelineMetadata(
            name="prod-pipeline",
            namespace="production",
            version="2.0.0",
        ),
        spec=PipelineSpec(
            steps=[
                StepDefinition(
                    id="ingest",
                    agent="GL-DATA-X-001",
                    publishes_data=True,
                    accesses_pii=True,
                ),
            ]
        ),
    )


@pytest.fixture
def basic_run_config() -> RunConfig:
    """Create basic run configuration."""
    return RunConfig(
        run_id="run-001",
        pipeline_name="test-pipeline",
        pipeline_version="1.0.0",
        namespace="default",
        user_id="user-123",
        environment="development",
        estimated_cost_usd=10.0,
        max_cost_usd=100.0,
    )


@pytest.fixture
def expensive_run_config() -> RunConfig:
    """Create expensive run configuration."""
    return RunConfig(
        run_id="run-002",
        pipeline_name="test-pipeline",
        pipeline_version="1.0.0",
        namespace="production",
        user_id="user-123",
        environment="production",
        estimated_cost_usd=500.0,
        max_cost_usd=100.0,
    )


@pytest.fixture
def basic_policy_config() -> PolicyEngineConfig:
    """Create basic policy engine config (OPA disabled)."""
    return PolicyEngineConfig(
        opa_enabled=False,
        yaml_rules_enabled=True,
        cache_enabled=False,
        strict_mode=False,
    )


@pytest.fixture
def yaml_rules_dict() -> Dict[str, Any]:
    """Sample YAML rules dictionary."""
    return {
        "name": "test-rules",
        "version": "1.0.0",
        "rules": [
            {
                "name": "deny_production_without_approval",
                "condition": "namespace == 'production'",
                "action": "require_approval",
                "severity": "error",
                "message": "Production deployments require approval",
                "approval_type": "manager",
                "evaluation_points": ["pre_run"],
            },
            {
                "name": "warn_high_cost",
                "condition": "run.estimated_cost_usd > 100",
                "action": "warn",
                "severity": "warning",
                "message": "High cost run: ${{ run.estimated_cost_usd }}",
                "evaluation_points": ["pre_run"],
            },
            {
                "name": "deny_pii_access_without_permission",
                "condition": "step.accesses_pii and 'pii_access' not_in permissions",
                "action": "deny",
                "severity": "error",
                "message": "PII access requires pii_access permission",
                "evaluation_points": ["pre_step"],
            },
        ],
    }


# =============================================================================
# YAML RULES PARSER TESTS
# =============================================================================


class TestYAMLRulesParser:
    """Tests for YAMLRulesParser class."""

    def test_load_rules_from_dict(self, yaml_rules_dict: Dict[str, Any]):
        """Test loading rules from dictionary."""
        parser = YAMLRulesParser()
        parser.load_rules_from_dict(yaml_rules_dict)

        rules = parser.get_rules()
        assert len(rules) == 3
        assert rules[0].name == "deny_production_without_approval"
        assert rules[1].name == "warn_high_cost"
        assert rules[2].name == "deny_pii_access_without_permission"

    def test_evaluate_simple_condition(self):
        """Test evaluating simple equality condition."""
        parser = YAMLRulesParser()
        parser.add_rule(YAMLRule(
            name="test_rule",
            condition="namespace == 'production'",
            action=PolicyAction.DENY,
        ))

        results = parser.evaluate(
            {"namespace": "production"},
            EvaluationPoint.PRE_RUN,
        )

        assert len(results) == 1
        rule, matched, message = results[0]
        assert rule.name == "test_rule"
        assert matched is True

    def test_evaluate_condition_not_matched(self):
        """Test condition that does not match."""
        parser = YAMLRulesParser()
        parser.add_rule(YAMLRule(
            name="test_rule",
            condition="namespace == 'production'",
            action=PolicyAction.DENY,
        ))

        results = parser.evaluate(
            {"namespace": "development"},
            EvaluationPoint.PRE_RUN,
        )

        assert len(results) == 1
        rule, matched, message = results[0]
        assert matched is False

    def test_evaluate_nested_access(self):
        """Test evaluating nested property access."""
        parser = YAMLRulesParser()
        parser.add_rule(YAMLRule(
            name="test_rule",
            condition="step.publishes_data == true",
            action=PolicyAction.DENY,
            evaluation_points=[EvaluationPoint.PRE_STEP],
        ))

        results = parser.evaluate(
            {"step": {"publishes_data": True}},
            EvaluationPoint.PRE_STEP,
        )

        assert len(results) == 1
        rule, matched, _ = results[0]
        assert matched is True

    def test_evaluate_and_condition(self):
        """Test evaluating AND condition."""
        parser = YAMLRulesParser()
        parser.add_rule(YAMLRule(
            name="test_rule",
            condition="namespace == 'production' and step.publishes_data == true",
            action=PolicyAction.DENY,
            evaluation_points=[EvaluationPoint.PRE_STEP],
        ))

        # Both conditions true
        results = parser.evaluate(
            {"namespace": "production", "step": {"publishes_data": True}},
            EvaluationPoint.PRE_STEP,
        )
        assert results[0][1] is True

        # One condition false
        results = parser.evaluate(
            {"namespace": "development", "step": {"publishes_data": True}},
            EvaluationPoint.PRE_STEP,
        )
        assert results[0][1] is False

    def test_evaluate_or_condition(self):
        """Test evaluating OR condition."""
        parser = YAMLRulesParser()
        parser.add_rule(YAMLRule(
            name="test_rule",
            condition="namespace == 'production' or namespace == 'staging'",
            action=PolicyAction.WARN,
        ))

        # First condition true
        results = parser.evaluate(
            {"namespace": "production"},
            EvaluationPoint.PRE_RUN,
        )
        assert results[0][1] is True

        # Second condition true
        results = parser.evaluate(
            {"namespace": "staging"},
            EvaluationPoint.PRE_RUN,
        )
        assert results[0][1] is True

        # Neither condition true
        results = parser.evaluate(
            {"namespace": "development"},
            EvaluationPoint.PRE_RUN,
        )
        assert results[0][1] is False

    def test_evaluate_numeric_comparison(self):
        """Test numeric comparisons."""
        parser = YAMLRulesParser()
        parser.add_rule(YAMLRule(
            name="test_rule",
            condition="run.estimated_cost_usd > 100",
            action=PolicyAction.DENY,
        ))

        # Greater than
        results = parser.evaluate(
            {"run": {"estimated_cost_usd": 150.0}},
            EvaluationPoint.PRE_RUN,
        )
        assert results[0][1] is True

        # Less than
        results = parser.evaluate(
            {"run": {"estimated_cost_usd": 50.0}},
            EvaluationPoint.PRE_RUN,
        )
        assert results[0][1] is False

    def test_evaluate_in_operator(self):
        """Test 'in' operator for list membership."""
        parser = YAMLRulesParser()
        parser.add_rule(YAMLRule(
            name="test_rule",
            condition="'admin' in user_roles",
            action=PolicyAction.ALLOW,
            evaluation_points=[EvaluationPoint.PRE_STEP],
        ))

        # Value in list
        results = parser.evaluate(
            {"user_roles": ["user", "admin", "viewer"]},
            EvaluationPoint.PRE_STEP,
        )
        assert results[0][1] is True

        # Value not in list
        results = parser.evaluate(
            {"user_roles": ["user", "viewer"]},
            EvaluationPoint.PRE_STEP,
        )
        assert results[0][1] is False

    def test_render_message_template(self):
        """Test message template rendering."""
        parser = YAMLRulesParser()
        parser.add_rule(YAMLRule(
            name="test_rule",
            condition="run.estimated_cost_usd > 100",
            action=PolicyAction.WARN,
            message="Cost is ${{ run.estimated_cost_usd }}",
        ))

        results = parser.evaluate(
            {"run": {"estimated_cost_usd": 250.0}},
            EvaluationPoint.PRE_RUN,
        )

        rule, matched, message = results[0]
        assert matched is True
        assert message == "Cost is $250.0"

    def test_filter_by_evaluation_point(self):
        """Test filtering rules by evaluation point."""
        parser = YAMLRulesParser()
        parser.add_rule(YAMLRule(
            name="pre_run_rule",
            condition="namespace == 'test'",
            action=PolicyAction.DENY,
            evaluation_points=[EvaluationPoint.PRE_RUN],
        ))
        parser.add_rule(YAMLRule(
            name="pre_step_rule",
            condition="namespace == 'test'",
            action=PolicyAction.DENY,
            evaluation_points=[EvaluationPoint.PRE_STEP],
        ))

        # Only pre_run rule should be evaluated
        results = parser.evaluate(
            {"namespace": "test"},
            EvaluationPoint.PRE_RUN,
        )
        assert len(results) == 1
        assert results[0][0].name == "pre_run_rule"

    def test_filter_by_namespace(self):
        """Test filtering rules by namespace."""
        parser = YAMLRulesParser()
        parser.add_rule(YAMLRule(
            name="prod_only_rule",
            condition="step.publishes_data == true",
            action=PolicyAction.DENY,
            namespaces=["production"],
            evaluation_points=[EvaluationPoint.PRE_STEP],
        ))

        # Should match for production
        results = parser.evaluate(
            {"step": {"publishes_data": True}},
            EvaluationPoint.PRE_STEP,
            namespace="production",
        )
        assert len(results) == 1

        # Should not match for development
        results = parser.evaluate(
            {"step": {"publishes_data": True}},
            EvaluationPoint.PRE_STEP,
            namespace="development",
        )
        assert len(results) == 0


# =============================================================================
# POLICY DECISION TESTS
# =============================================================================


class TestPolicyDecision:
    """Tests for PolicyDecision model."""

    def test_decision_provenance_hash(self):
        """Test provenance hash computation."""
        decision = PolicyDecision(
            allowed=True,
            evaluation_point=EvaluationPoint.PRE_RUN,
            policy_version="1.0.0",
        )

        hash1 = decision.compute_provenance_hash()
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters

        # Same decision should produce same hash
        hash2 = decision.compute_provenance_hash()
        assert hash1 == hash2

    def test_decision_with_reasons(self):
        """Test decision with denial reasons."""
        decision = PolicyDecision(
            allowed=False,
            evaluation_point=EvaluationPoint.PRE_RUN,
            reasons=[
                PolicyReason(
                    rule_name="cost_exceeded",
                    message="Cost exceeds budget",
                    severity=PolicySeverity.ERROR,
                    action=PolicyAction.DENY,
                ),
            ],
        )

        assert decision.allowed is False
        assert len(decision.reasons) == 1
        assert decision.reasons[0].rule_name == "cost_exceeded"

    def test_decision_with_approvals(self):
        """Test decision requiring approvals."""
        decision = PolicyDecision(
            allowed=True,
            evaluation_point=EvaluationPoint.PRE_RUN,
            required_approvals=[
                ApprovalRequirement(
                    approval_type=ApprovalType.MANAGER,
                    reason="Production deployment",
                ),
            ],
        )

        assert len(decision.required_approvals) == 1
        assert decision.required_approvals[0].approval_type == ApprovalType.MANAGER


# =============================================================================
# POLICY ENGINE TESTS
# =============================================================================


class TestPolicyEngine:
    """Tests for PolicyEngine class."""

    @pytest.mark.asyncio
    async def test_evaluate_pre_run_allowed(
        self,
        simple_pipeline: PipelineDefinition,
        basic_run_config: RunConfig,
        basic_policy_config: PolicyEngineConfig,
    ):
        """Test pre-run evaluation with allowed result."""
        engine = PolicyEngine(basic_policy_config)

        decision = await engine.evaluate_pre_run(simple_pipeline, basic_run_config)

        assert decision.allowed is True
        assert decision.evaluation_point == EvaluationPoint.PRE_RUN
        assert len(decision.reasons) == 0

        await engine.close()

    @pytest.mark.asyncio
    async def test_evaluate_pre_run_with_yaml_rules(
        self,
        production_pipeline: PipelineDefinition,
        basic_policy_config: PolicyEngineConfig,
        yaml_rules_dict: Dict[str, Any],
    ):
        """Test pre-run evaluation with YAML rules."""
        engine = PolicyEngine(basic_policy_config)
        engine._yaml_parser.load_rules_from_dict(yaml_rules_dict)

        run_config = RunConfig(
            run_id="run-003",
            pipeline_name="prod-pipeline",
            pipeline_version="2.0.0",
            namespace="production",
            user_id="user-123",
            environment="production",
        )

        decision = await engine.evaluate_pre_run(production_pipeline, run_config)

        # Should require approval for production
        assert len(decision.required_approvals) > 0
        assert decision.required_approvals[0].approval_type == ApprovalType.MANAGER

        await engine.close()

    @pytest.mark.asyncio
    async def test_cost_budget_enforcement(
        self,
        simple_pipeline: PipelineDefinition,
        expensive_run_config: RunConfig,
        basic_policy_config: PolicyEngineConfig,
    ):
        """Test cost budget enforcement."""
        engine = PolicyEngine(basic_policy_config)

        # Add cost budget
        bundle = PolicyBundle(
            bundle_id="test-bundle",
            version="1.0.0",
            name="Test Bundle",
            cost_budgets={
                "production": CostBudget(
                    max_cost_usd=100.0,
                    allow_override=False,
                ),
            },
        )
        engine.add_bundle(bundle)

        decision = await engine.evaluate_pre_run(simple_pipeline, expensive_run_config)

        # Should be denied due to cost
        assert decision.allowed is False
        assert any("cost" in r.rule_name.lower() for r in decision.reasons)

        await engine.close()

    @pytest.mark.asyncio
    async def test_cost_budget_with_override(
        self,
        simple_pipeline: PipelineDefinition,
        expensive_run_config: RunConfig,
        basic_policy_config: PolicyEngineConfig,
    ):
        """Test cost budget with override allowed."""
        engine = PolicyEngine(basic_policy_config)

        # Add cost budget with override
        bundle = PolicyBundle(
            bundle_id="test-bundle",
            version="1.0.0",
            name="Test Bundle",
            cost_budgets={
                "production": CostBudget(
                    max_cost_usd=100.0,
                    allow_override=True,
                    override_approval_type=ApprovalType.COST_CENTER,
                ),
            },
        )
        engine.add_bundle(bundle)

        decision = await engine.evaluate_pre_run(simple_pipeline, expensive_run_config)

        # Should require approval instead of deny
        assert len(decision.required_approvals) > 0
        assert decision.required_approvals[0].approval_type == ApprovalType.COST_CENTER

        await engine.close()

    @pytest.mark.asyncio
    async def test_evaluate_pre_step(
        self,
        simple_pipeline: PipelineDefinition,
        basic_run_config: RunConfig,
        basic_policy_config: PolicyEngineConfig,
    ):
        """Test pre-step evaluation."""
        engine = PolicyEngine(basic_policy_config)

        step = simple_pipeline.spec.steps[0]
        context = ExecutionContext(
            run_id=basic_run_config.run_id,
            pipeline=simple_pipeline,
            run_config=basic_run_config,
            current_step=step.get_effective_id(),
            permissions={"read", "write"},
        )

        decision = await engine.evaluate_pre_step(step, context)

        assert decision.allowed is True
        assert decision.evaluation_point == EvaluationPoint.PRE_STEP

        await engine.close()

    @pytest.mark.asyncio
    async def test_evaluate_pre_step_pii_denied(
        self,
        production_pipeline: PipelineDefinition,
        basic_policy_config: PolicyEngineConfig,
    ):
        """Test pre-step denies PII access without permission."""
        engine = PolicyEngine(basic_policy_config)

        run_config = RunConfig(
            run_id="run-004",
            pipeline_name="prod-pipeline",
            pipeline_version="2.0.0",
            namespace="production",
            user_id="user-123",
            environment="production",
        )

        step = production_pipeline.spec.steps[0]  # Has accesses_pii=True
        context = ExecutionContext(
            run_id=run_config.run_id,
            pipeline=production_pipeline,
            run_config=run_config,
            current_step=step.get_effective_id(),
            permissions={"read"},  # No pii_access permission
        )

        decision = await engine.evaluate_pre_step(step, context)

        # Should be denied due to missing pii_access permission
        assert decision.allowed is False
        assert any("pii" in r.rule_name.lower() for r in decision.reasons)

        await engine.close()

    @pytest.mark.asyncio
    async def test_evaluate_post_step(
        self,
        simple_pipeline: PipelineDefinition,
        basic_run_config: RunConfig,
        basic_policy_config: PolicyEngineConfig,
    ):
        """Test post-step evaluation."""
        engine = PolicyEngine(basic_policy_config)

        step = simple_pipeline.spec.steps[0]
        context = ExecutionContext(
            run_id=basic_run_config.run_id,
            pipeline=simple_pipeline,
            run_config=basic_run_config,
            current_step=step.get_effective_id(),
        )

        result = StepResult(
            step_name=step.get_effective_id(),
            success=True,
            outputs={"dataset": "s3://bucket/data"},
        )

        decision = await engine.evaluate_post_step(step, result, context)

        assert decision.allowed is True
        assert decision.evaluation_point == EvaluationPoint.POST_STEP

        await engine.close()

    @pytest.mark.asyncio
    async def test_export_controls_restricted(
        self,
        simple_pipeline: PipelineDefinition,
        basic_run_config: RunConfig,
        basic_policy_config: PolicyEngineConfig,
    ):
        """Test export controls for restricted data."""
        engine = PolicyEngine(basic_policy_config)

        step = simple_pipeline.spec.steps[0]
        context = ExecutionContext(
            run_id=basic_run_config.run_id,
            pipeline=simple_pipeline,
            run_config=basic_run_config,
            current_step=step.get_effective_id(),
        )

        result = StepResult(
            step_name=step.get_effective_id(),
            success=True,
            output_classification=DataClassification.RESTRICTED,
            export_destinations=["external-partner-api"],
        )

        decision = await engine.evaluate_post_step(step, result, context)

        # Should be denied due to export controls
        assert decision.allowed is False
        assert any("export" in r.rule_name.lower() for r in decision.reasons)

        await engine.close()


# =============================================================================
# POLICY BUNDLE TESTS
# =============================================================================


class TestPolicyBundle:
    """Tests for PolicyBundle management."""

    def test_add_bundle(self, basic_policy_config: PolicyEngineConfig):
        """Test adding a policy bundle."""
        engine = PolicyEngine(basic_policy_config)

        bundle = PolicyBundle(
            bundle_id="test-bundle",
            version="1.0.0",
            name="Test Bundle",
            yaml_rules=[
                YAMLRule(
                    name="test_rule",
                    condition="namespace == 'test'",
                    action=PolicyAction.DENY,
                ),
            ],
        )

        bundle_hash = engine.add_bundle(bundle)

        assert len(bundle_hash) == 64
        assert "test-bundle" in engine.list_bundles()

    def test_get_bundle(self, basic_policy_config: PolicyEngineConfig):
        """Test getting a bundle by ID."""
        engine = PolicyEngine(basic_policy_config)

        bundle = PolicyBundle(
            bundle_id="test-bundle",
            version="1.0.0",
            name="Test Bundle",
        )
        engine.add_bundle(bundle)

        retrieved = engine.get_bundle("test-bundle")

        assert retrieved is not None
        assert retrieved.bundle_id == "test-bundle"
        assert retrieved.version == "1.0.0"

    def test_remove_bundle(self, basic_policy_config: PolicyEngineConfig):
        """Test removing a bundle."""
        engine = PolicyEngine(basic_policy_config)

        bundle = PolicyBundle(
            bundle_id="test-bundle",
            version="1.0.0",
            name="Test Bundle",
        )
        engine.add_bundle(bundle)

        removed = engine.remove_bundle("test-bundle")

        assert removed is True
        assert "test-bundle" not in engine.list_bundles()

    def test_effective_bundle_namespace(self, basic_policy_config: PolicyEngineConfig):
        """Test getting effective bundle for namespace."""
        engine = PolicyEngine(basic_policy_config)

        # Add namespace-specific bundle
        ns_bundle = PolicyBundle(
            bundle_id="prod-bundle",
            version="1.0.0",
            name="Production Bundle",
            namespace="production",
        )
        engine.add_bundle(ns_bundle)

        # Add organization baseline
        org_bundle = PolicyBundle(
            bundle_id="org-bundle",
            version="1.0.0",
            name="Org Baseline",
            namespace=None,
        )
        engine.add_bundle(org_bundle)

        # Production should get namespace-specific bundle
        effective = engine.get_effective_bundle("production")
        assert effective is not None
        assert effective.bundle_id == "prod-bundle"

        # Other namespace should get org baseline
        effective = engine.get_effective_bundle("development")
        assert effective is not None
        assert effective.bundle_id == "org-bundle"


# =============================================================================
# DATA RESIDENCY TESTS
# =============================================================================


class TestDataResidency:
    """Tests for data residency rules."""

    @pytest.mark.asyncio
    async def test_residency_violation(
        self,
        basic_policy_config: PolicyEngineConfig,
    ):
        """Test data residency violation detection."""
        engine = PolicyEngine(basic_policy_config)

        # Add residency rule
        bundle = PolicyBundle(
            bundle_id="eu-bundle",
            version="1.0.0",
            name="EU Residency",
            residency_rules=[
                DataResidencyRule(
                    name="eu_only",
                    allowed_regions=["eu-west-1", "eu-central-1"],
                    applies_to_classification=[DataClassification.CONFIDENTIAL],
                    message="Confidential data must stay in EU",
                ),
            ],
        )
        engine.add_bundle(bundle)

        pipeline = PipelineDefinition(
            apiVersion="greenlang/v1",
            kind="Pipeline",
            metadata=PipelineMetadata(name="test-pipeline"),
            spec=PipelineSpec(
                steps=[
                    StepDefinition(
                        id="step1",
                        agent="GL-DATA-X-001",
                        data_regions=["us-east-1"],  # Violates EU-only rule
                    ),
                ]
            ),
        )

        run_config = RunConfig(
            run_id="run-005",
            pipeline_name="test-pipeline",
            pipeline_version="1.0.0",
            namespace="default",
            user_id="user-123",
            classification_level=DataClassification.CONFIDENTIAL,
        )

        decision = await engine.evaluate_pre_run(pipeline, run_config)

        # Should be denied due to residency violation
        assert decision.allowed is False
        assert any("residency" in r.rule_name.lower() for r in decision.reasons)

        await engine.close()


# =============================================================================
# OPA CLIENT TESTS
# =============================================================================


class TestOPAClient:
    """Tests for OPA client functionality."""

    def test_format_input_pipeline(self, simple_pipeline: PipelineDefinition):
        """Test OPA input formatting for pipeline."""
        client = OPAClient("http://localhost:8181")

        input_doc = client.format_input(pipeline=simple_pipeline)

        assert "pipeline" in input_doc
        assert input_doc["pipeline"]["name"] == "test-pipeline"
        assert input_doc["pipeline"]["step_count"] == 2

    def test_format_input_run_config(self, basic_run_config: RunConfig):
        """Test OPA input formatting for run config."""
        client = OPAClient("http://localhost:8181")

        input_doc = client.format_input(run_config=basic_run_config)

        assert "run" in input_doc
        assert input_doc["run"]["run_id"] == "run-001"
        assert input_doc["run"]["namespace"] == "default"

    def test_parse_decision_allow(self):
        """Test parsing OPA allow decision."""
        client = OPAClient("http://localhost:8181")

        opa_result = {"result": True}
        decision = client.parse_decision(opa_result, EvaluationPoint.PRE_RUN)

        assert decision.allowed is True
        assert len(decision.reasons) == 0

    def test_parse_decision_deny_with_violations(self):
        """Test parsing OPA deny decision with violations."""
        client = OPAClient("http://localhost:8181")

        opa_result = {
            "result": {
                "allow": False,
                "violations": [
                    {
                        "rule": "cost_limit",
                        "message": "Cost exceeds limit",
                        "severity": "error",
                    },
                ],
            }
        }
        decision = client.parse_decision(opa_result, EvaluationPoint.PRE_RUN)

        assert decision.allowed is False
        assert len(decision.reasons) == 1
        assert decision.reasons[0].rule_name == "cost_limit"


# =============================================================================
# AUDIT LOG TESTS
# =============================================================================


class TestAuditLog:
    """Tests for audit logging functionality."""

    @pytest.mark.asyncio
    async def test_audit_log_populated(
        self,
        simple_pipeline: PipelineDefinition,
        basic_run_config: RunConfig,
        basic_policy_config: PolicyEngineConfig,
    ):
        """Test audit log is populated after evaluation."""
        config = basic_policy_config.model_copy()
        config.audit_all_decisions = True

        engine = PolicyEngine(config)

        await engine.evaluate_pre_run(simple_pipeline, basic_run_config)

        audit_log = engine.get_audit_log()

        assert len(audit_log) >= 1
        assert audit_log[0]["run_id"] == "run-001"
        assert audit_log[0]["evaluation_point"] == "pre_run"

        await engine.close()

    @pytest.mark.asyncio
    async def test_cache_clear(
        self,
        simple_pipeline: PipelineDefinition,
        basic_run_config: RunConfig,
    ):
        """Test cache clearing."""
        config = PolicyEngineConfig(
            opa_enabled=False,
            yaml_rules_enabled=True,
            cache_enabled=True,
        )
        engine = PolicyEngine(config)

        await engine.evaluate_pre_run(simple_pipeline, basic_run_config)
        engine.clear_cache()

        # Cache should be empty
        assert len(engine._cache) == 0

        await engine.close()


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
