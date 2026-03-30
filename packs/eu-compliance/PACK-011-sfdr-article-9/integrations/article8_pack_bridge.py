# -*- coding: utf-8 -*-
"""
Article8PackBridge - PACK-010 Cross-Reference and Downgrade Detection
======================================================================

This module connects PACK-011 (SFDR Article 9) with PACK-010 (SFDR Article 8)
to support downgrade scenario analysis, shared PAI calculation reuse,
classification comparison, and migration workflows. When an Article 9 product
no longer meets the sustainable investment objective requirements, this bridge
provides structured downgrade assessment to Article 8 or Article 8+.

Architecture:
    PACK-011 Article 9 --> Article8PackBridge --> PACK-010 Article 8
                                |
                                v
    Downgrade Assessment, Shared PAI, Classification Comparison

Example:
    >>> config = Article8BridgeConfig()
    >>> bridge = Article8PackBridge(config)
    >>> assessment = bridge.assess_downgrade(pipeline_result)
    >>> print(f"Downgrade risk: {assessment.risk_level}")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# Utility Helpers
# =============================================================================

def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

# =============================================================================
# Agent Stub
# =============================================================================

class _AgentStub:
    """Deferred agent loader for lazy initialization."""

    def __init__(self, agent_id: str, module_path: str, class_name: str) -> None:
        self.agent_id = agent_id
        self.module_path = module_path
        self.class_name = class_name
        self._instance: Optional[Any] = None

    def load(self) -> Any:
        """Load and return the agent instance."""
        if self._instance is not None:
            return self._instance
        try:
            import importlib

            mod = importlib.import_module(self.module_path)
            cls = getattr(mod, self.class_name)
            self._instance = cls()
            return self._instance
        except Exception as exc:
            logger.warning(
                "AgentStub: failed to load %s: %s", self.agent_id, exc,
            )
            return None

    @property
    def is_loaded(self) -> bool:
        """Whether the agent has been loaded."""
        return self._instance is not None

# =============================================================================
# Enums
# =============================================================================

class DowngradeRiskLevel(str, Enum):
    """Risk level for downgrade assessment."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ClassificationType(str, Enum):
    """SFDR product classification."""
    ARTICLE_6 = "article_6"
    ARTICLE_8 = "article_8"
    ARTICLE_8_PLUS = "article_8_plus"
    ARTICLE_9 = "article_9"

class DowngradeReason(str, Enum):
    """Reasons for potential Article 9 downgrade."""
    SI_BELOW_THRESHOLD = "si_below_threshold"
    DNSH_FAILURES = "dnsh_failures"
    GOVERNANCE_FAILURES = "governance_failures"
    TAXONOMY_MISALIGNMENT = "taxonomy_misalignment"
    PAI_INCOMPLETE = "pai_incomplete"
    OBJECTIVE_NOT_MET = "objective_not_met"
    BENCHMARK_DEVIATION = "benchmark_deviation"
    REGULATORY_CHANGE = "regulatory_change"

# =============================================================================
# Data Models
# =============================================================================

class Article8BridgeConfig(BaseModel):
    """Configuration for the Article 8 Pack Bridge."""
    pack_010_path: str = Field(
        default="packs.eu_compliance.PACK_010_sfdr_article_8",
        description="Import path for PACK-010",
    )
    enable_downgrade_monitoring: bool = Field(
        default=True,
        description="Enable continuous downgrade risk monitoring",
    )
    si_downgrade_threshold_pct: float = Field(
        default=90.0, ge=0.0, le=100.0,
        description="SI % below which downgrade is triggered",
    )
    dnsh_downgrade_threshold_pct: float = Field(
        default=80.0, ge=0.0, le=100.0,
        description="DNSH pass % below which downgrade warning is raised",
    )
    governance_downgrade_threshold_pct: float = Field(
        default=90.0, ge=0.0, le=100.0,
        description="Governance pass % below which downgrade warning is raised",
    )
    enable_shared_pai: bool = Field(
        default=True,
        description="Enable shared PAI calculation with PACK-010",
    )
    auto_reclassify: bool = Field(
        default=False,
        description="Automatically reclassify on critical downgrade (requires approval)",
    )

class DowngradeAssessment(BaseModel):
    """Result of a downgrade risk assessment."""
    product_name: str = Field(default="", description="Product name")
    current_classification: str = Field(
        default="article_9", description="Current SFDR classification"
    )
    risk_level: DowngradeRiskLevel = Field(
        default=DowngradeRiskLevel.NONE, description="Overall downgrade risk"
    )
    recommended_classification: str = Field(
        default="article_9", description="Recommended classification"
    )
    downgrade_reasons: List[str] = Field(
        default_factory=list, description="Reasons for potential downgrade"
    )
    risk_factors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detailed risk factor analysis"
    )
    remediation_actions: List[str] = Field(
        default_factory=list, description="Suggested remediation actions"
    )
    regulatory_deadline: str = Field(
        default="", description="Deadline for reclassification if required"
    )
    assessed_at: str = Field(default="", description="Assessment timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class ClassificationComparison(BaseModel):
    """Comparison between Article 8 and Article 9 requirements."""
    product_name: str = Field(default="", description="Product name")
    article_8_requirements_met: Dict[str, bool] = Field(
        default_factory=dict, description="Article 8 requirements status"
    )
    article_9_requirements_met: Dict[str, bool] = Field(
        default_factory=dict, description="Article 9 requirements status"
    )
    classification_recommendation: str = Field(
        default="article_9", description="Recommended classification"
    )
    gap_analysis: List[Dict[str, Any]] = Field(
        default_factory=list, description="Gap analysis between classifications"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class SharedPAIResult(BaseModel):
    """Result of shared PAI calculation between packs."""
    indicator_id: int = Field(default=0, description="PAI indicator ID")
    indicator_name: str = Field(default="", description="Indicator name")
    article_8_value: float = Field(default=0.0, description="Value from PACK-010")
    article_9_value: float = Field(default=0.0, description="Value from PACK-011")
    delta: float = Field(default=0.0, description="Difference")
    calculation_shared: bool = Field(
        default=False, description="Whether calculation is shared"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# =============================================================================
# Article 8 / Article 9 Requirement Definitions
# =============================================================================

ARTICLE_8_REQUIREMENTS: Dict[str, str] = {
    "es_characteristics": "Promotes E/S characteristics",
    "binding_elements": "Binding elements defined",
    "pai_consideration": "PAI considered (mandatory from 2023)",
    "taxonomy_disclosure": "Taxonomy alignment disclosed (can be 0%)",
    "dnsh_basic": "Basic DNSH assessment",
    "good_governance_basic": "Good governance screening",
    "annex_ii": "Annex II pre-contractual disclosure",
    "annex_iv": "Annex IV periodic disclosure",
    "exclusion_policy": "Exclusion policy applied",
}

ARTICLE_9_REQUIREMENTS: Dict[str, str] = {
    "sustainable_objective": "Has sustainable investment as its objective",
    "100_pct_si": "All investments are sustainable (limited exceptions)",
    "enhanced_dnsh": "Enhanced DNSH across all 6 objectives",
    "good_governance_strict": "Strict good governance (Art 2(17))",
    "all_18_pai": "All 18 mandatory PAI indicators (no opt-out)",
    "taxonomy_alignment_env": "Taxonomy alignment for env SI",
    "impact_measurement": "Impact measurement and reporting",
    "annex_iii": "Annex III pre-contractual disclosure",
    "annex_v": "Annex V periodic disclosure",
    "benchmark_9_3": "CTB/PAB benchmark (for Art 9(3))",
}

# =============================================================================
# Article 8 Pack Bridge
# =============================================================================

class Article8PackBridge:
    """Bridge connecting PACK-011 (Art 9) with PACK-010 (Art 8).

    Provides downgrade risk assessment, shared PAI calculation reuse,
    classification comparison, and migration support between Article 9
    and Article 8 products.

    Attributes:
        config: Bridge configuration.
        _agents: Deferred agent stubs for PACK-010 engines.

    Example:
        >>> bridge = Article8PackBridge(Article8BridgeConfig())
        >>> assessment = bridge.assess_downgrade(pipeline_result)
        >>> print(f"Risk: {assessment.risk_level}")
    """

    def __init__(self, config: Optional[Article8BridgeConfig] = None) -> None:
        """Initialize the Article 8 Pack Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or Article8BridgeConfig()
        self.logger = logger

        self._agents: Dict[str, _AgentStub] = {
            "pack_010_orchestrator": _AgentStub(
                "PACK-010-ORCH",
                f"{self.config.pack_010_path}.integrations.pack_orchestrator",
                "SFDRPackOrchestrator",
            ),
            "pack_010_pai": _AgentStub(
                "PACK-010-PAI",
                f"{self.config.pack_010_path}.engines.pai_engine",
                "PAIEngine",
            ),
            "pack_010_compliance": _AgentStub(
                "PACK-010-COMPLIANCE",
                f"{self.config.pack_010_path}.engines.compliance_engine",
                "SFDRComplianceEngine",
            ),
        }

        self.logger.info(
            "Article8PackBridge initialized: pack_010=%s, downgrade_monitoring=%s",
            self.config.pack_010_path,
            self.config.enable_downgrade_monitoring,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def assess_downgrade(
        self,
        pipeline_data: Dict[str, Any],
    ) -> DowngradeAssessment:
        """Assess downgrade risk from Article 9 to Article 8.

        Evaluates the pipeline results against Article 9 thresholds and
        determines if a downgrade to Article 8 or 8+ is indicated.

        Args:
            pipeline_data: Pipeline result data from Article 9 orchestrator.

        Returns:
            DowngradeAssessment with risk level and recommendations.
        """
        product_name = pipeline_data.get("product_name", "")
        risk_factors: List[Dict[str, Any]] = []
        downgrade_reasons: List[str] = []
        remediation: List[str] = []

        # Factor 1: Sustainable investment coverage
        si_pct = float(pipeline_data.get("sustainable_investment_pct", 100.0))
        if si_pct < self.config.si_downgrade_threshold_pct:
            risk_factors.append({
                "factor": "sustainable_investment_coverage",
                "value": si_pct,
                "threshold": self.config.si_downgrade_threshold_pct,
                "severity": "critical",
            })
            downgrade_reasons.append(DowngradeReason.SI_BELOW_THRESHOLD.value)
            remediation.append(
                f"Increase sustainable investment coverage from {si_pct:.1f}% "
                f"to at least {self.config.si_downgrade_threshold_pct:.1f}%"
            )

        # Factor 2: Enhanced DNSH pass rate
        dnsh_pct = float(pipeline_data.get("enhanced_dnsh_pass_pct", 100.0))
        if dnsh_pct < self.config.dnsh_downgrade_threshold_pct:
            risk_factors.append({
                "factor": "enhanced_dnsh_pass_rate",
                "value": dnsh_pct,
                "threshold": self.config.dnsh_downgrade_threshold_pct,
                "severity": "high",
            })
            downgrade_reasons.append(DowngradeReason.DNSH_FAILURES.value)
            remediation.append(
                "Review DNSH failures and address non-compliant holdings"
            )

        # Factor 3: Good governance
        gov_pct = float(pipeline_data.get("good_governance_pass_pct", 100.0))
        if gov_pct < self.config.governance_downgrade_threshold_pct:
            risk_factors.append({
                "factor": "good_governance_pass_rate",
                "value": gov_pct,
                "threshold": self.config.governance_downgrade_threshold_pct,
                "severity": "high",
            })
            downgrade_reasons.append(DowngradeReason.GOVERNANCE_FAILURES.value)
            remediation.append(
                "Engage with investee companies on governance improvements"
            )

        # Factor 4: PAI completeness
        pai_count = int(pipeline_data.get("pai_indicators_calculated", 18))
        if pai_count < 18:
            risk_factors.append({
                "factor": "pai_completeness",
                "value": pai_count,
                "threshold": 18,
                "severity": "medium",
            })
            downgrade_reasons.append(DowngradeReason.PAI_INCOMPLETE.value)
            remediation.append(
                f"Obtain data for remaining {18 - pai_count} mandatory PAI indicators"
            )

        # Determine overall risk level
        risk_level = self._calculate_risk_level(risk_factors)

        # Recommend classification
        if risk_level in (DowngradeRiskLevel.CRITICAL, DowngradeRiskLevel.HIGH):
            recommended = "article_8_plus"
        elif risk_level == DowngradeRiskLevel.MEDIUM:
            recommended = "article_9"  # With remediation
        else:
            recommended = "article_9"

        assessment = DowngradeAssessment(
            product_name=product_name,
            current_classification="article_9",
            risk_level=risk_level,
            recommended_classification=recommended,
            downgrade_reasons=downgrade_reasons,
            risk_factors=risk_factors,
            remediation_actions=remediation,
            regulatory_deadline="",
            assessed_at=utcnow().isoformat(),
        )
        assessment.provenance_hash = _hash_data(assessment.model_dump())

        self.logger.info(
            "Downgrade assessment: product=%s, risk=%s, recommended=%s, "
            "factors=%d",
            product_name, risk_level.value, recommended, len(risk_factors),
        )
        return assessment

    def compare_classifications(
        self,
        pipeline_data: Dict[str, Any],
    ) -> ClassificationComparison:
        """Compare Article 8 vs Article 9 requirement satisfaction.

        Args:
            pipeline_data: Pipeline result data from Article 9 orchestrator.

        Returns:
            ClassificationComparison with gap analysis.
        """
        product_name = pipeline_data.get("product_name", "")
        si_pct = float(pipeline_data.get("sustainable_investment_pct", 0.0))
        dnsh_pct = float(pipeline_data.get("enhanced_dnsh_pass_pct", 0.0))
        gov_pct = float(pipeline_data.get("good_governance_pass_pct", 0.0))
        pai_count = int(pipeline_data.get("pai_indicators_calculated", 0))
        taxonomy_pct = float(pipeline_data.get("taxonomy_alignment_pct", 0.0))
        disclosures = int(pipeline_data.get("disclosures_generated", 0))

        # Article 8 requirements
        art8_met: Dict[str, bool] = {
            "es_characteristics": True,
            "binding_elements": True,
            "pai_consideration": pai_count > 0,
            "taxonomy_disclosure": True,
            "dnsh_basic": dnsh_pct > 50.0,
            "good_governance_basic": gov_pct > 50.0,
            "annex_ii": disclosures >= 1,
            "annex_iv": disclosures >= 2,
            "exclusion_policy": True,
        }

        # Article 9 requirements
        art9_met: Dict[str, bool] = {
            "sustainable_objective": si_pct >= 90.0,
            "100_pct_si": si_pct >= 95.0,
            "enhanced_dnsh": dnsh_pct >= 80.0,
            "good_governance_strict": gov_pct >= 90.0,
            "all_18_pai": pai_count >= 18,
            "taxonomy_alignment_env": taxonomy_pct >= 0.0,
            "impact_measurement": pipeline_data.get("impact_score", 0.0) > 0,
            "annex_iii": disclosures >= 1,
            "annex_v": disclosures >= 2,
            "benchmark_9_3": True,
        }

        # Gap analysis
        gaps: List[Dict[str, Any]] = []
        for req_id, req_desc in ARTICLE_9_REQUIREMENTS.items():
            is_met = art9_met.get(req_id, False)
            if not is_met:
                gaps.append({
                    "requirement": req_id,
                    "description": req_desc,
                    "status": "not_met",
                    "action_required": True,
                })

        art9_score = sum(1 for v in art9_met.values() if v)
        art8_score = sum(1 for v in art8_met.values() if v)

        if art9_score == len(art9_met):
            recommendation = "article_9"
        elif art8_score == len(art8_met):
            recommendation = "article_8_plus" if taxonomy_pct > 0 else "article_8"
        else:
            recommendation = "article_8"

        result = ClassificationComparison(
            product_name=product_name,
            article_8_requirements_met=art8_met,
            article_9_requirements_met=art9_met,
            classification_recommendation=recommendation,
            gap_analysis=gaps,
        )
        result.provenance_hash = _hash_data(result.model_dump())

        self.logger.info(
            "Classification comparison: art8=%d/%d, art9=%d/%d, recommend=%s",
            art8_score, len(art8_met), art9_score, len(art9_met), recommendation,
        )
        return result

    def get_shared_pai(
        self,
        pai_indicator_id: int,
        article_9_value: float,
    ) -> SharedPAIResult:
        """Get shared PAI calculation between Article 8 and Article 9.

        When both PACK-010 and PACK-011 are deployed for the same fund
        manager, PAI calculations can be shared to avoid duplicate work.

        Args:
            pai_indicator_id: PAI indicator number (1-18).
            article_9_value: Value calculated by PACK-011.

        Returns:
            SharedPAIResult with both pack values.
        """
        art8_value = 0.0
        calculation_shared = False

        if self.config.enable_shared_pai:
            engine = self._agents["pack_010_pai"].load()
            if engine is not None:
                try:
                    result = engine.calculate(pai_indicator_id)
                    art8_value = float(
                        result.get("value", 0.0) if isinstance(result, dict)
                        else getattr(result, "value", 0.0)
                    )
                    calculation_shared = True
                except Exception as exc:
                    self.logger.warning(
                        "Failed to get PACK-010 PAI %d: %s",
                        pai_indicator_id, exc,
                    )

        delta = round(article_9_value - art8_value, 4)

        result = SharedPAIResult(
            indicator_id=pai_indicator_id,
            indicator_name=f"PAI {pai_indicator_id}",
            article_8_value=art8_value,
            article_9_value=article_9_value,
            delta=delta,
            calculation_shared=calculation_shared,
        )
        result.provenance_hash = _hash_data(result.model_dump())
        return result

    def route_to_pack_010(
        self,
        request_type: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route a request to PACK-010 for cross-pack operations.

        Args:
            request_type: Type of request (orchestrate, pai, compliance).
            data: Request data.

        Returns:
            Response from PACK-010 or error dictionary.
        """
        if request_type == "downgrade_assessment":
            result = self.assess_downgrade(data)
            return result.model_dump()

        elif request_type == "classification_comparison":
            result = self.compare_classifications(data)
            return result.model_dump()

        elif request_type == "shared_pai":
            indicator_id = int(data.get("indicator_id", 1))
            art9_value = float(data.get("value", 0.0))
            result = self.get_shared_pai(indicator_id, art9_value)
            return result.model_dump()

        else:
            self.logger.warning("Unknown PACK-010 request type: %s", request_type)
            return {"error": f"Unknown request type: {request_type}"}

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _calculate_risk_level(
        self,
        risk_factors: List[Dict[str, Any]],
    ) -> DowngradeRiskLevel:
        """Calculate overall downgrade risk level from individual factors.

        Args:
            risk_factors: List of risk factor assessments.

        Returns:
            Overall DowngradeRiskLevel.
        """
        if not risk_factors:
            return DowngradeRiskLevel.NONE

        severities = [f.get("severity", "low") for f in risk_factors]

        if "critical" in severities:
            return DowngradeRiskLevel.CRITICAL
        if severities.count("high") >= 2:
            return DowngradeRiskLevel.HIGH
        if "high" in severities:
            return DowngradeRiskLevel.MEDIUM
        if severities:
            return DowngradeRiskLevel.LOW

        return DowngradeRiskLevel.NONE
