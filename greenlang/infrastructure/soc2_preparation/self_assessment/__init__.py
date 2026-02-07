# -*- coding: utf-8 -*-
"""
SOC 2 Self-Assessment Module - SEC-009

Provides self-assessment capabilities for SOC 2 Type II audit preparation.
Includes the complete Trust Service Criteria definitions, assessment engine,
scoring algorithms, and gap analysis.

Components:
    - criteria: All 48 SOC 2 Trust Service Criteria definitions
    - assessor: Assessment engine for evaluating controls
    - scorer: Scoring algorithms for maturity calculation
    - gap_analyzer: Gap identification and prioritization

Example:
    >>> from greenlang.infrastructure.soc2_preparation.self_assessment import (
    ...     Assessor, Scorer, GapAnalyzer, TSC_CRITERIA
    ... )
    >>> from greenlang.infrastructure.soc2_preparation.config import get_config
    >>>
    >>> config = get_config()
    >>> assessor = Assessor(config)
    >>> assessment = await assessor.run_assessment(user_id)
    >>>
    >>> scorer = Scorer()
    >>> overall_score = scorer.calculate_overall_score(assessment)
    >>> readiness = scorer.get_readiness_percentage(assessment)
    >>>
    >>> analyzer = GapAnalyzer()
    >>> gaps = analyzer.analyze_gaps(assessment)
    >>> prioritized = analyzer.prioritize_gaps(gaps)
    >>> report = analyzer.generate_gap_report(prioritized)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from greenlang.infrastructure.soc2_preparation.self_assessment.criteria import (
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
)

from greenlang.infrastructure.soc2_preparation.self_assessment.assessor import (
    Assessor,
    AssessmentStorage,
    InMemoryStorage,
    create_assessor,
)

from greenlang.infrastructure.soc2_preparation.self_assessment.scorer import (
    Scorer,
    MaturityLevel,
    ScoreThresholds,
    DEFAULT_THRESHOLDS,
    calculate_score,
    get_readiness,
    score_to_status,
)

from greenlang.infrastructure.soc2_preparation.self_assessment.gap_analyzer import (
    GapAnalyzer,
    Gap,
    RiskLevel,
    EffortLevel,
    analyze_assessment_gaps,
    generate_report,
)


__all__ = [
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
]
