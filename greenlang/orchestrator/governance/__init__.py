# -*- coding: utf-8 -*-
"""
Orchestrator Governance Layer
==============================

Provides policy enforcement for the GreenLang orchestrator.

Components:
    - PolicyEngine: Hybrid OPA + YAML policy evaluation
    - PolicyDecision: Evaluation results with provenance
    - PolicyBundle: Versioned policy collections
    - OPAClient: HTTP client for OPA server
    - YAMLRulesParser: Declarative YAML rule evaluation
    - ApprovalWorkflow: Signed approval workflow (FR-043)

Evaluation Points:
    - Pre-run: Pipeline + plan validation
    - Pre-step: Permissions, cost, data residency
    - Post-step: Artifact classification, export controls

Example:
    >>> from greenlang.orchestrator.governance import PolicyEngine
    >>> engine = PolicyEngine()
    >>> decision = await engine.evaluate_pre_run(pipeline, run_config)
    >>> if not decision.allowed:
    ...     print(decision.reasons[0].message)

Author: GreenLang Team
Version: 1.0.0
"""

from greenlang.orchestrator.governance.policy_engine import (
    # Main classes
    PolicyEngine,
    PolicyEngineConfig,
    OPAClient,
    YAMLRulesParser,
    # Decision models
    PolicyDecision,
    PolicyReason,
    ApprovalRequirement,
    # Enums
    PolicyAction,
    PolicySeverity,
    EvaluationPoint,
    ApprovalType,
    # Rule models
    YAMLRule,
    YAMLRuleSet,
    CostBudget,
    DataResidencyRule,
    PolicyBundle,
    # Exceptions
    OPAError,
)

from greenlang.orchestrator.governance.approvals import (
    # Enums
    ApprovalStatus,
    ApprovalDecision,
    # Models
    ApprovalAttestation,
    ApprovalRequest,
    # Store
    ApprovalStore,
    InMemoryApprovalStore,
    # Utilities
    SignatureUtils,
    # Workflow
    ApprovalWorkflow,
    # Exceptions
    ApprovalError,
    ApprovalNotFoundError,
    ApprovalExpiredError,
    ApprovalAlreadyDecidedError,
    SignatureVerificationError,
    # Constants
    CRYPTO_AVAILABLE,
)

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
    # FR-043: Approval Workflow
    "ApprovalStatus",
    "ApprovalDecision",
    "ApprovalAttestation",
    "ApprovalRequest",
    "ApprovalStore",
    "InMemoryApprovalStore",
    "SignatureUtils",
    "ApprovalWorkflow",
    "ApprovalError",
    "ApprovalNotFoundError",
    "ApprovalExpiredError",
    "ApprovalAlreadyDecidedError",
    "SignatureVerificationError",
    "CRYPTO_AVAILABLE",
]
