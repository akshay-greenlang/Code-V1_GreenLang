# -*- coding: utf-8 -*-
"""
SOC 2 Type II Audit Preparation Platform - SEC-009

Enterprise-grade platform for preparing GreenLang for SOC 2 Type II compliance
audits. Provides self-assessment capabilities, evidence management, auditor
request tracking, finding remediation, and comprehensive audit project management.

Key Features:
    - Self-assessment against all 48 SOC 2 Trust Service Criteria
    - Automated evidence collection and management
    - Auditor request (PBC list) tracking with SLA monitoring
    - Gap analysis and prioritized remediation planning
    - Management attestation workflow with electronic signatures
    - Real-time readiness scoring and dashboards

Trust Service Categories Supported:
    - Security (Common Criteria CC1-CC9) - Required
    - Availability (A1) - Optional
    - Confidentiality (C1) - Optional
    - Processing Integrity (PI1) - Optional
    - Privacy (P1-P8) - Optional

Architecture:
    This module follows GreenLang's zero-hallucination principle for all
    calculations. Scores, gap analysis, and effort estimates use deterministic
    formulas only. LLM assistance is limited to evidence categorization and
    narrative generation where appropriate.

Example:
    >>> from greenlang.infrastructure.soc2_preparation import (
    ...     SOC2Config, get_config,
    ...     Assessment, AssessmentCriteria, Evidence,
    ...     Assessor, Scorer, GapAnalyzer, TSC_CRITERIA,
    ...     SOC2Metrics,
    ... )
    >>>
    >>> # Configure
    >>> config = get_config()
    >>>
    >>> # Run assessment
    >>> assessor = Assessor(config)
    >>> assessment = await assessor.run_assessment(user_id)
    >>>
    >>> # Calculate scores
    >>> scorer = Scorer()
    >>> overall = scorer.calculate_overall_score(assessment)
    >>> readiness = scorer.get_readiness_percentage(assessment)
    >>>
    >>> # Analyze gaps
    >>> analyzer = GapAnalyzer()
    >>> gaps = analyzer.analyze_gaps(assessment)
    >>> prioritized = analyzer.prioritize_gaps(gaps)
    >>> report = analyzer.generate_gap_report(prioritized)
    >>>
    >>> print(f"SOC 2 Readiness: {overall:.1f}%")
    >>> print(f"Criteria Compliant: {readiness:.1f}%")

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from greenlang.infrastructure.soc2_preparation.config import (
    SOC2Config,
    EnvironmentConfig,
    EnvironmentName,
    EnvironmentProfile,
    get_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

from greenlang.infrastructure.soc2_preparation.models import (
    # Enums
    ScoreLevel,
    FindingClassification,
    RequestPriority,
    TestType,
    EvidenceType,
    AssessmentStatus,
    ControlStatus,
    RemediationStatus,
    MilestoneStatus,
    TrustServiceCategory,
    # Base
    SOC2BaseModel,
    # Assessment Models
    Assessment,
    AssessmentCriteria,
    # Evidence Models
    Evidence,
    EvidencePackage,
    # Control Test Models
    ControlTest,
    TestResult,
    # Auditor Request Models
    AuditorRequest,
    # Finding Models
    Finding,
    Remediation,
    # Attestation Models
    Attestation,
    AttestationSignature,
    # Audit Project Models
    AuditProject,
    AuditMilestone,
)

# ---------------------------------------------------------------------------
# Self-Assessment Engine
# ---------------------------------------------------------------------------

from greenlang.infrastructure.soc2_preparation.self_assessment import (
    # Criteria Definitions
    TSC_CRITERIA,
    CATEGORY_WEIGHTS,
    CriterionDefinition,
    get_criterion,
    get_criteria_by_category,
    get_criteria_by_subcategory,
    get_criteria_by_risk_level,
    get_all_criterion_ids,
    get_category_criteria_count,
    get_security_criteria,
    # Assessor
    Assessor,
    AssessmentStorage,
    InMemoryStorage,
    create_assessor,
    # Scorer
    Scorer,
    MaturityLevel,
    ScoreThresholds,
    DEFAULT_THRESHOLDS,
    calculate_score,
    get_readiness,
    score_to_status,
    # Gap Analyzer
    GapAnalyzer,
    Gap,
    RiskLevel,
    EffortLevel,
    analyze_assessment_gaps,
    generate_report,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

from greenlang.infrastructure.soc2_preparation.metrics import (
    SOC2Metrics,
    record_assessment_completed,
    record_evidence_uploaded,
    record_evidence_verified,
    record_auditor_request,
    update_overdue_requests,
    update_gaps,
    update_remediation_effort,
    update_readiness_score,
    update_compliant_criteria,
    record_finding,
    record_attestation_signature,
    update_project_completion,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Configuration
    "SOC2Config",
    "EnvironmentConfig",
    "EnvironmentName",
    "EnvironmentProfile",
    "get_config",
    "reset_config",
    # Enums
    "ScoreLevel",
    "FindingClassification",
    "RequestPriority",
    "TestType",
    "EvidenceType",
    "AssessmentStatus",
    "ControlStatus",
    "RemediationStatus",
    "MilestoneStatus",
    "TrustServiceCategory",
    # Base Model
    "SOC2BaseModel",
    # Assessment Models
    "Assessment",
    "AssessmentCriteria",
    # Evidence Models
    "Evidence",
    "EvidencePackage",
    # Control Test Models
    "ControlTest",
    "TestResult",
    # Auditor Request Models
    "AuditorRequest",
    # Finding Models
    "Finding",
    "Remediation",
    # Attestation Models
    "Attestation",
    "AttestationSignature",
    # Audit Project Models
    "AuditProject",
    "AuditMilestone",
    # Criteria
    "TSC_CRITERIA",
    "CATEGORY_WEIGHTS",
    "CriterionDefinition",
    "get_criterion",
    "get_criteria_by_category",
    "get_criteria_by_subcategory",
    "get_criteria_by_risk_level",
    "get_all_criterion_ids",
    "get_category_criteria_count",
    "get_security_criteria",
    # Assessor
    "Assessor",
    "AssessmentStorage",
    "InMemoryStorage",
    "create_assessor",
    # Scorer
    "Scorer",
    "MaturityLevel",
    "ScoreThresholds",
    "DEFAULT_THRESHOLDS",
    "calculate_score",
    "get_readiness",
    "score_to_status",
    # Gap Analyzer
    "GapAnalyzer",
    "Gap",
    "RiskLevel",
    "EffortLevel",
    "analyze_assessment_gaps",
    "generate_report",
    # Metrics
    "SOC2Metrics",
    "record_assessment_completed",
    "record_evidence_uploaded",
    "record_evidence_verified",
    "record_auditor_request",
    "update_overdue_requests",
    "update_gaps",
    "update_remediation_effort",
    "update_readiness_score",
    "update_compliant_criteria",
    "record_finding",
    "record_attestation_signature",
    "update_project_completion",
]
