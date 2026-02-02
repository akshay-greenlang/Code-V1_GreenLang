"""
GreenLang Framework - Standards Module

Defines quality standards, compliance levels, and requirements
for GreenLang AI agents based on global standards.

Includes the Global AI Standards v2.0 framework aligned with:
- Anthropic (Claude) AI Safety Standards
- OpenAI Safety & Alignment Framework
- Google DeepMind Responsible AI Principles
- ISO/IEC 42001:2023 (AI Management System)
- NIST AI RMF 1.0
- EU AI Act (High-Risk Systems)
"""

from .quality_standards import (
    QualityStandard,
    QualityDimension,
    ComplianceLevel,
    AgentClass,
    DomainCategory,
    RequirementLevel,
    GREENLANG_STANDARD,
)

# Global AI Standards v2.0 - The authoritative scoring framework
from .global_ai_standards import (
    GlobalAIStandard,
    ScoringCategory,
    EvaluationCriterion,
    ScoreTier,
    DomainStandard,
    AgentScore,
    GLOBAL_AI_STANDARD,
    CURRENT_AGENT_SCORES,
    get_improvement_roadmap,
)

__all__ = [
    # Original Quality Standards
    "QualityStandard",
    "QualityDimension",
    "ComplianceLevel",
    "AgentClass",
    "DomainCategory",
    "RequirementLevel",
    "GREENLANG_STANDARD",
    # Global AI Standards v2.0
    "GlobalAIStandard",
    "ScoringCategory",
    "EvaluationCriterion",
    "ScoreTier",
    "DomainStandard",
    "AgentScore",
    "GLOBAL_AI_STANDARD",
    "CURRENT_AGENT_SCORES",
    "get_improvement_roadmap",
]
