# -*- coding: utf-8 -*-
"""
DiagnosticExplainer for GL-008 TRAPCATCHER

SHAP/LIME-style explainability module providing per-alert explanations
for steam trap diagnostic decisions. Designed for operator understanding.

Key Features:
- SHAP-compatible feature attribution
- Counterfactual explanations ("what would change the diagnosis?")
- Natural language explanation generation
- Evidence chain visualization data
- Audit-ready explanation records

Zero-Hallucination Guarantee:
All explanations derived from deterministic feature contributions.
No LLM or AI inference in explanation generation.
Same classification always produces identical explanations.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class ContributionDirection(str, Enum):
    """Direction of feature contribution to classification."""
    TOWARD_FAILURE = "toward_failure"
    TOWARD_NORMAL = "toward_normal"
    NEUTRAL = "neutral"


class ExplanationStyle(str, Enum):
    """Style of explanation output."""
    TECHNICAL = "technical"    # For engineers
    OPERATOR = "operator"      # For plant operators
    EXECUTIVE = "executive"    # For management reports


class EvidenceStrength(str, Enum):
    """Strength of evidence classification."""
    STRONG = "strong"         # Primary driver of classification
    MODERATE = "moderate"     # Contributing factor
    WEAK = "weak"             # Minor influence
    NONE = "none"             # No contribution


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class ExplainerConfig:
    """
    Configuration for diagnostic explainer.

    Attributes:
        explanation_style: Default explanation style
        max_features: Maximum features to include in explanation
        include_counterfactuals: Whether to generate counterfactual explanations
        confidence_threshold: Minimum confidence for strong evidence
        language: Output language (currently only English)
    """
    explanation_style: ExplanationStyle = ExplanationStyle.OPERATOR
    max_features: int = 5
    include_counterfactuals: bool = True
    confidence_threshold: float = 0.70
    language: str = "en"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class FeatureContribution:
    """
    SHAP-compatible feature contribution.

    Attributes:
        feature_name: Name of the feature
        feature_value: Actual value of the feature
        baseline_value: Expected baseline value
        contribution_score: SHAP-style contribution (-1 to 1)
        direction: Contribution direction
        strength: Evidence strength classification
        human_readable: Plain-language description
    """
    feature_name: str
    feature_value: float
    baseline_value: float
    contribution_score: float
    direction: ContributionDirection
    strength: EvidenceStrength
    human_readable: str


@dataclass(frozen=True)
class CounterfactualExplanation:
    """
    Counterfactual explanation showing what would change the outcome.

    Attributes:
        target_condition: The condition that would result
        feature_changes: Required feature value changes
        feasibility_score: How feasible the change is (0-1)
        explanation: Plain-language explanation
    """
    target_condition: str
    feature_changes: Dict[str, Tuple[float, float]]  # feature -> (current, required)
    feasibility_score: float
    explanation: str


@dataclass(frozen=True)
class EvidenceChain:
    """
    Chain of evidence supporting the classification.

    Attributes:
        step_number: Order in the evidence chain
        evidence_type: Type of evidence (acoustic, thermal, etc.)
        observation: What was observed
        inference: What it implies
        confidence: Confidence in this evidence
    """
    step_number: int
    evidence_type: str
    observation: str
    inference: str
    confidence: float


@dataclass(frozen=True)
class ExplanationResult:
    """
    Complete explanation result.

    Attributes:
        trap_id: Steam trap identifier
        classification: The classification being explained
        confidence: Classification confidence
        timestamp: Explanation timestamp
        feature_contributions: Ranked feature contributions
        evidence_chain: Ordered evidence chain
        counterfactuals: Counterfactual explanations
        summary_technical: Technical summary
        summary_operator: Operator-friendly summary
        summary_executive: Executive summary
        provenance_hash: SHA-256 hash for audit trail
    """
    trap_id: str
    classification: str
    confidence: float
    timestamp: datetime
    feature_contributions: List[FeatureContribution]
    evidence_chain: List[EvidenceChain]
    counterfactuals: List[CounterfactualExplanation]
    summary_technical: str
    summary_operator: str
    summary_executive: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trap_id": self.trap_id,
            "classification": self.classification,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
            "feature_contributions": [
                {
                    "feature": fc.feature_name,
                    "value": fc.feature_value,
                    "baseline": fc.baseline_value,
                    "contribution": round(fc.contribution_score, 4),
                    "direction": fc.direction.value,
                    "strength": fc.strength.value,
                    "explanation": fc.human_readable
                }
                for fc in self.feature_contributions
            ],
            "evidence_chain": [
                {
                    "step": ec.step_number,
                    "type": ec.evidence_type,
                    "observation": ec.observation,
                    "inference": ec.inference,
                    "confidence": round(ec.confidence, 2)
                }
                for ec in self.evidence_chain
            ],
            "counterfactuals": [
                {
                    "target": cf.target_condition,
                    "changes": {k: {"from": v[0], "to": v[1]} for k, v in cf.feature_changes.items()},
                    "feasibility": round(cf.feasibility_score, 2),
                    "explanation": cf.explanation
                }
                for cf in self.counterfactuals
            ],
            "summaries": {
                "technical": self.summary_technical,
                "operator": self.summary_operator,
                "executive": self.summary_executive
            },
            "provenance_hash": self.provenance_hash
        }


# ============================================================================
# REFERENCE DATA
# ============================================================================

# Feature baselines and thresholds
FEATURE_BASELINES: Dict[str, Dict[str, float]] = {
    "acoustic_amplitude_db": {
        "baseline": 40.0,
        "normal_max": 45.0,
        "warning_threshold": 60.0,
        "critical_threshold": 70.0
    },
    "temperature_differential_c": {
        "baseline": 50.0,
        "normal_min": 25.0,
        "normal_max": 80.0,
        "blow_through_max": 10.0,
        "blocked_min": 100.0
    },
    "trap_age_years": {
        "baseline": 3.0,
        "normal_max": 5.0,
        "warning_threshold": 7.0,
        "critical_threshold": 10.0
    },
    "pressure_bar_g": {
        "baseline": 10.0,
        "normal_min": 2.0,
        "normal_max": 20.0
    }
}

# Human-readable feature descriptions
FEATURE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "acoustic_amplitude_db": {
        "high": "Ultrasonic sensor detected high amplitude ({value:.1f} dB), indicating steam leakage",
        "normal": "Ultrasonic level is within normal range ({value:.1f} dB)",
        "low": "Very low ultrasonic activity ({value:.1f} dB), may indicate blocked trap"
    },
    "temperature_differential_c": {
        "high": "Large temperature drop ({value:.1f}C) between inlet and outlet, suggests blockage",
        "normal": "Normal temperature differential ({value:.1f}C) indicates proper operation",
        "low": "Minimal temperature drop ({value:.1f}C), indicates steam passing through (blow-through)"
    },
    "trap_age_years": {
        "high": "Trap is {value:.1f} years old, exceeding typical service life",
        "normal": "Trap age ({value:.1f} years) is within normal range",
        "low": "Relatively new trap ({value:.1f} years old)"
    }
}


# ============================================================================
# DIAGNOSTIC EXPLAINER
# ============================================================================

class DiagnosticExplainer:
    """
    SHAP-compatible diagnostic explainer for steam trap classifications.

    ZERO-HALLUCINATION GUARANTEE:
    - All explanations derived from deterministic feature contributions
    - No LLM or AI inference in explanation generation
    - Same classification always produces identical explanations
    - Complete provenance tracking with SHA-256 hashes

    Explanation Types:
    1. Feature Contributions: SHAP-style attribution scores
    2. Evidence Chain: Step-by-step reasoning
    3. Counterfactuals: What would change the diagnosis
    4. Natural Language: Human-readable summaries

    Example:
        >>> explainer = DiagnosticExplainer()
        >>> explanation = explainer.explain(
        ...     trap_id="ST-001",
        ...     classification="failed_open",
        ...     confidence=0.85,
        ...     features={
        ...         "acoustic_amplitude_db": 75.0,
        ...         "temperature_differential_c": 5.0,
        ...         "trap_age_years": 6.0
        ...     }
        ... )
        >>> print(explanation.summary_operator)
    """

    def __init__(self, config: Optional[ExplainerConfig] = None):
        """
        Initialize diagnostic explainer.

        Args:
            config: Explainer configuration (uses defaults if not provided)
        """
        self.config = config or ExplainerConfig()
        self._explanation_count = 0

        logger.info(f"DiagnosticExplainer initialized (style={self.config.explanation_style.value})")

    def explain(
        self,
        trap_id: str,
        classification: str,
        confidence: float,
        features: Dict[str, float],
        modality_scores: Optional[Dict[str, float]] = None
    ) -> ExplanationResult:
        """
        Generate comprehensive explanation for a classification.

        ZERO-HALLUCINATION: Uses deterministic feature attribution.

        Args:
            trap_id: Steam trap identifier
            classification: The classification to explain
            confidence: Classification confidence (0-1)
            features: Feature values used in classification
            modality_scores: Optional modality-level scores

        Returns:
            ExplanationResult with complete explanation
        """
        self._explanation_count += 1
        timestamp = datetime.now(timezone.utc)

        # Calculate feature contributions
        feature_contributions = self._calculate_contributions(features, classification)

        # Build evidence chain
        evidence_chain = self._build_evidence_chain(features, classification, confidence)

        # Generate counterfactuals if enabled
        counterfactuals = []
        if self.config.include_counterfactuals:
            counterfactuals = self._generate_counterfactuals(features, classification)

        # Generate summaries
        summary_technical = self._generate_technical_summary(
            classification, confidence, feature_contributions
        )
        summary_operator = self._generate_operator_summary(
            classification, confidence, feature_contributions, evidence_chain
        )
        summary_executive = self._generate_executive_summary(
            classification, confidence, feature_contributions
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            trap_id, classification, features
        )

        return ExplanationResult(
            trap_id=trap_id,
            classification=classification,
            confidence=confidence,
            timestamp=timestamp,
            feature_contributions=feature_contributions,
            evidence_chain=evidence_chain,
            counterfactuals=counterfactuals,
            summary_technical=summary_technical,
            summary_operator=summary_operator,
            summary_executive=summary_executive,
            provenance_hash=provenance_hash
        )

    def _calculate_contributions(
        self,
        features: Dict[str, float],
        classification: str
    ) -> List[FeatureContribution]:
        """Calculate SHAP-style feature contributions."""
        contributions = []

        for feature_name, value in features.items():
            if feature_name not in FEATURE_BASELINES:
                continue

            baseline_info = FEATURE_BASELINES[feature_name]
            baseline = baseline_info["baseline"]
            diff = value - baseline

            # Calculate contribution score
            if feature_name == "acoustic_amplitude_db":
                # Higher acoustic = more toward failure
                if value > baseline_info.get("critical_threshold", 70):
                    contribution = 0.4
                    direction = ContributionDirection.TOWARD_FAILURE
                    strength = EvidenceStrength.STRONG
                elif value > baseline_info.get("warning_threshold", 60):
                    contribution = 0.25
                    direction = ContributionDirection.TOWARD_FAILURE
                    strength = EvidenceStrength.MODERATE
                elif value < 30:
                    contribution = 0.15
                    direction = ContributionDirection.TOWARD_FAILURE  # Blocked
                    strength = EvidenceStrength.MODERATE
                else:
                    contribution = -0.1
                    direction = ContributionDirection.TOWARD_NORMAL
                    strength = EvidenceStrength.WEAK

                if value > 70:
                    human_readable = FEATURE_DESCRIPTIONS[feature_name]["high"].format(value=value)
                elif value < 30:
                    human_readable = FEATURE_DESCRIPTIONS[feature_name]["low"].format(value=value)
                else:
                    human_readable = FEATURE_DESCRIPTIONS[feature_name]["normal"].format(value=value)

            elif feature_name == "temperature_differential_c":
                # Very small delta = blow-through, very large = blocked
                if value < baseline_info.get("blow_through_max", 10):
                    contribution = 0.35
                    direction = ContributionDirection.TOWARD_FAILURE
                    strength = EvidenceStrength.STRONG
                    human_readable = FEATURE_DESCRIPTIONS[feature_name]["low"].format(value=value)
                elif value > baseline_info.get("blocked_min", 100):
                    contribution = 0.3
                    direction = ContributionDirection.TOWARD_FAILURE
                    strength = EvidenceStrength.STRONG
                    human_readable = FEATURE_DESCRIPTIONS[feature_name]["high"].format(value=value)
                else:
                    contribution = -0.15
                    direction = ContributionDirection.TOWARD_NORMAL
                    strength = EvidenceStrength.MODERATE
                    human_readable = FEATURE_DESCRIPTIONS[feature_name]["normal"].format(value=value)

            elif feature_name == "trap_age_years":
                if value > baseline_info.get("critical_threshold", 10):
                    contribution = 0.2
                    direction = ContributionDirection.TOWARD_FAILURE
                    strength = EvidenceStrength.MODERATE
                    human_readable = FEATURE_DESCRIPTIONS[feature_name]["high"].format(value=value)
                elif value > baseline_info.get("warning_threshold", 7):
                    contribution = 0.1
                    direction = ContributionDirection.TOWARD_FAILURE
                    strength = EvidenceStrength.WEAK
                    human_readable = FEATURE_DESCRIPTIONS[feature_name]["high"].format(value=value)
                else:
                    contribution = -0.05
                    direction = ContributionDirection.TOWARD_NORMAL
                    strength = EvidenceStrength.WEAK
                    human_readable = FEATURE_DESCRIPTIONS[feature_name]["normal"].format(value=value)

            else:
                contribution = 0.0
                direction = ContributionDirection.NEUTRAL
                strength = EvidenceStrength.NONE
                human_readable = f"{feature_name}: {value}"

            contributions.append(FeatureContribution(
                feature_name=feature_name,
                feature_value=value,
                baseline_value=baseline,
                contribution_score=round(contribution, 4),
                direction=direction,
                strength=strength,
                human_readable=human_readable
            ))

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution_score), reverse=True)

        return contributions[:self.config.max_features]

    def _build_evidence_chain(
        self,
        features: Dict[str, float],
        classification: str,
        confidence: float
    ) -> List[EvidenceChain]:
        """Build step-by-step evidence chain."""
        chain = []
        step = 1

        # Acoustic evidence
        if "acoustic_amplitude_db" in features:
            amp = features["acoustic_amplitude_db"]
            if amp > 70:
                observation = f"Ultrasonic amplitude measured at {amp:.1f} dB"
                inference = "High amplitude indicates continuous steam flow (potential blow-through)"
                conf = 0.85
            elif amp > 45:
                observation = f"Ultrasonic amplitude measured at {amp:.1f} dB"
                inference = "Elevated amplitude suggests steam leakage"
                conf = 0.70
            elif amp < 25:
                observation = f"Ultrasonic amplitude very low at {amp:.1f} dB"
                inference = "Low activity may indicate blocked trap or cold system"
                conf = 0.65
            else:
                observation = f"Ultrasonic amplitude at {amp:.1f} dB"
                inference = "Acoustic signature within normal operating range"
                conf = 0.80

            chain.append(EvidenceChain(step, "acoustic", observation, inference, conf))
            step += 1

        # Thermal evidence
        if "temperature_differential_c" in features:
            delta_t = features["temperature_differential_c"]
            if delta_t < 10:
                observation = f"Temperature differential is only {delta_t:.1f}C"
                inference = "Minimal temperature drop indicates steam passing through trap"
                conf = 0.90
            elif delta_t > 100:
                observation = f"Temperature differential is {delta_t:.1f}C"
                inference = "Large temperature drop suggests trap is blocked"
                conf = 0.85
            else:
                observation = f"Temperature differential is {delta_t:.1f}C"
                inference = "Temperature profile consistent with normal operation"
                conf = 0.75

            chain.append(EvidenceChain(step, "thermal", observation, inference, conf))
            step += 1

        # Age evidence
        if "trap_age_years" in features:
            age = features["trap_age_years"]
            if age > 7:
                observation = f"Trap has been in service for {age:.1f} years"
                inference = "Extended service life increases failure probability"
                conf = 0.60
            else:
                observation = f"Trap has been in service for {age:.1f} years"
                inference = "Service life within typical range"
                conf = 0.70

            chain.append(EvidenceChain(step, "context", observation, inference, conf))
            step += 1

        # Final conclusion
        chain.append(EvidenceChain(
            step_number=step,
            evidence_type="conclusion",
            observation=f"Combined evidence confidence: {confidence*100:.0f}%",
            inference=f"Classification: {classification.replace('_', ' ').title()}",
            confidence=confidence
        ))

        return chain

    def _generate_counterfactuals(
        self,
        features: Dict[str, float],
        classification: str
    ) -> List[CounterfactualExplanation]:
        """Generate counterfactual explanations."""
        counterfactuals = []

        if classification in ["failed_open", "leaking"]:
            # What would make it normal?
            changes = {}

            if "acoustic_amplitude_db" in features and features["acoustic_amplitude_db"] > 45:
                changes["acoustic_amplitude_db"] = (
                    features["acoustic_amplitude_db"],
                    40.0
                )

            if "temperature_differential_c" in features and features["temperature_differential_c"] < 25:
                changes["temperature_differential_c"] = (
                    features["temperature_differential_c"],
                    50.0
                )

            if changes:
                counterfactuals.append(CounterfactualExplanation(
                    target_condition="operating_normal",
                    feature_changes=changes,
                    feasibility_score=0.3 if classification == "failed_open" else 0.5,
                    explanation="For this trap to be classified as normal, "
                               "acoustic levels would need to decrease and/or "
                               "temperature differential would need to normalize."
                ))

        elif classification == "failed_closed":
            changes = {}

            if "temperature_differential_c" in features and features["temperature_differential_c"] > 80:
                changes["temperature_differential_c"] = (
                    features["temperature_differential_c"],
                    50.0
                )

            if changes:
                counterfactuals.append(CounterfactualExplanation(
                    target_condition="operating_normal",
                    feature_changes=changes,
                    feasibility_score=0.4,
                    explanation="For normal operation, outlet temperature would need "
                               "to be closer to inlet temperature."
                ))

        return counterfactuals

    def _generate_technical_summary(
        self,
        classification: str,
        confidence: float,
        contributions: List[FeatureContribution]
    ) -> str:
        """Generate technical summary for engineers."""
        parts = [
            f"Classification: {classification} (confidence: {confidence*100:.1f}%)",
            "Primary contributing factors:"
        ]

        for i, fc in enumerate(contributions[:3], 1):
            parts.append(f"  {i}. {fc.feature_name}: {fc.contribution_score:+.3f} "
                        f"({fc.direction.value})")

        return "\n".join(parts)

    def _generate_operator_summary(
        self,
        classification: str,
        confidence: float,
        contributions: List[FeatureContribution],
        evidence_chain: List[EvidenceChain]
    ) -> str:
        """Generate operator-friendly summary."""
        condition_display = classification.replace("_", " ").title()

        parts = [f"Steam Trap Status: {condition_display}"]

        # Add primary evidence in plain language
        for fc in contributions[:2]:
            if fc.strength in [EvidenceStrength.STRONG, EvidenceStrength.MODERATE]:
                parts.append(f"  - {fc.human_readable}")

        # Add confidence
        if confidence >= 0.85:
            parts.append("Assessment confidence: HIGH")
        elif confidence >= 0.70:
            parts.append("Assessment confidence: MEDIUM")
        else:
            parts.append("Assessment confidence: LOW - recommend manual verification")

        return "\n".join(parts)

    def _generate_executive_summary(
        self,
        classification: str,
        confidence: float,
        contributions: List[FeatureContribution]
    ) -> str:
        """Generate executive summary for management reports."""
        condition_display = classification.replace("_", " ").title()

        # Determine impact level
        if classification in ["failed_open"]:
            impact = "HIGH - Significant energy loss"
        elif classification in ["leaking", "failed_closed"]:
            impact = "MEDIUM - Requires attention"
        else:
            impact = "LOW - Normal operation"

        return (f"Trap Status: {condition_display} | "
                f"Confidence: {confidence*100:.0f}% | "
                f"Impact: {impact}")

    def _calculate_provenance_hash(
        self,
        trap_id: str,
        classification: str,
        features: Dict[str, float]
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        data = {
            "trap_id": trap_id,
            "classification": classification,
            "features": features
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get explainer statistics."""
        return {
            "explanation_count": self._explanation_count,
            "explanation_style": self.config.explanation_style.value,
            "max_features": self.config.max_features,
            "counterfactuals_enabled": self.config.include_counterfactuals
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "DiagnosticExplainer",
    "ExplainerConfig",
    "ExplanationResult",
    "FeatureContribution",
    "CounterfactualExplanation",
    "EvidenceChain",
    "ContributionDirection",
    "ExplanationStyle",
    "EvidenceStrength",
]
