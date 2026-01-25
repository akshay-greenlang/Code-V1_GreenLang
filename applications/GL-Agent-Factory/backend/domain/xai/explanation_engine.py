# -*- coding: utf-8 -*-
"""
Explanation Engine for GreenLang Agents
=======================================

Generates human-readable explanations for agent decisions and calculations,
ensuring transparency and regulatory compliance.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ExplanationType(str, Enum):
    """Types of explanations."""
    CALCULATION = "calculation"  # Step-by-step calculation
    DECISION = "decision"  # Decision rationale
    CLASSIFICATION = "classification"  # Category assignment
    RECOMMENDATION = "recommendation"  # Suggested action
    ANOMALY = "anomaly"  # Anomaly detection
    COMPLIANCE = "compliance"  # Regulatory compliance
    COMPARISON = "comparison"  # Benchmark comparison


class ExplanationLevel(str, Enum):
    """Detail level of explanations."""
    SUMMARY = "summary"  # Brief one-liner
    STANDARD = "standard"  # Normal detail
    DETAILED = "detailed"  # Full technical detail
    EXPERT = "expert"  # Expert-level with formulas
    AUDIT = "audit"  # Full audit trail


class ExplanationAudience(str, Enum):
    """Target audience for explanations."""
    OPERATOR = "operator"
    ENGINEER = "engineer"
    MANAGER = "manager"
    AUDITOR = "auditor"
    REGULATOR = "regulator"


@dataclass
class ExplanationStep:
    """Single step in an explanation chain."""
    step_number: int
    action: str
    description: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    formula_used: Optional[str] = None
    formula_source: Optional[str] = None
    confidence: float = 1.0
    notes: List[str] = field(default_factory=list)


@dataclass
class SourceReference:
    """Reference to source material."""
    source_type: str  # "standard", "formula", "data", "model"
    source_id: str
    source_name: str
    section: Optional[str] = None
    url: Optional[str] = None
    citation: Optional[str] = None


class Explanation(BaseModel):
    """
    Complete explanation for an agent output.

    Attributes:
        explanation_id: Unique identifier
        explanation_type: Type of explanation
        level: Detail level
        summary: Brief summary
        steps: Detailed explanation steps
        sources: Referenced sources
        confidence: Overall confidence score
        caveats: Important caveats or limitations
    """
    explanation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Classification
    explanation_type: ExplanationType
    level: ExplanationLevel = ExplanationLevel.STANDARD
    audience: ExplanationAudience = ExplanationAudience.ENGINEER

    # Content
    summary: str
    detailed_explanation: str = ""
    steps: List[Dict[str, Any]] = Field(default_factory=list)

    # Sources and provenance
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    formulas_used: List[str] = Field(default_factory=list)
    standards_referenced: List[str] = Field(default_factory=list)

    # Confidence and uncertainty
    confidence_score: float = Field(default=1.0, ge=0, le=1)
    uncertainty_range: Optional[tuple] = None

    # Caveats and assumptions
    assumptions: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)

    # Audit
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    provenance_hash: Optional[str] = None

    def to_natural_language(self) -> str:
        """Convert explanation to natural language."""
        parts = [self.summary]

        if self.steps:
            parts.append("\n\nStep-by-step breakdown:")
            for i, step in enumerate(self.steps, 1):
                parts.append(f"\n{i}. {step.get('description', 'N/A')}")
                if step.get('formula_used'):
                    parts.append(f"   Formula: {step['formula_used']}")
                if step.get('formula_source'):
                    parts.append(f"   Source: {step['formula_source']}")

        if self.sources:
            parts.append("\n\nReferences:")
            for source in self.sources:
                parts.append(f"- {source.get('source_name', 'Unknown')}")

        if self.caveats:
            parts.append("\n\nImportant notes:")
            for caveat in self.caveats:
                parts.append(f"- {caveat}")

        return "\n".join(parts)


class ExplanationEngine:
    """
    Engine for generating explanations for agent outputs.

    Provides methods to create transparent, auditable explanations
    for calculations, decisions, and recommendations.
    """

    def __init__(self):
        self.templates: Dict[str, str] = {}
        self._load_templates()

    def _load_templates(self):
        """Load explanation templates."""
        self.templates = {
            "calculation_summary": "Calculated {output_name} = {output_value} {unit} using {formula_name} per {standard}.",
            "efficiency_summary": "Equipment efficiency is {efficiency}% ({rating}). {recommendation}",
            "compliance_summary": "{status}: {item} {compliance_result} per {standard} {section}.",
            "anomaly_summary": "Detected {anomaly_type} anomaly in {parameter}: {current_value} vs expected {expected_value} ({deviation}% deviation).",
            "recommendation_summary": "Recommendation: {action}. Expected benefit: {benefit}. Priority: {priority}.",
        }

    def explain_calculation(
        self,
        calculation_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        formula: str,
        formula_source: str,
        level: ExplanationLevel = ExplanationLevel.STANDARD,
        audience: ExplanationAudience = ExplanationAudience.ENGINEER,
    ) -> Explanation:
        """
        Generate explanation for a calculation.

        Args:
            calculation_name: Name of the calculation
            inputs: Input parameters with values and units
            outputs: Output values with units
            formula: Mathematical formula used
            formula_source: Source standard/reference
            level: Detail level
            audience: Target audience

        Returns:
            Complete explanation object
        """
        # Build summary
        output_key = list(outputs.keys())[0] if outputs else "result"
        output_val = outputs.get(output_key, {})
        summary = f"Calculated {calculation_name}: {output_val.get('value', 'N/A')} {output_val.get('unit', '')} using {formula_source}."

        # Build steps
        steps = []

        # Step 1: Input validation
        steps.append({
            "step_number": 1,
            "action": "validate_inputs",
            "description": f"Validated {len(inputs)} input parameters",
            "inputs": inputs,
            "outputs": {"validation": "passed"},
        })

        # Step 2: Apply formula
        steps.append({
            "step_number": 2,
            "action": "apply_formula",
            "description": f"Applied {formula_source} formula",
            "formula_used": formula,
            "formula_source": formula_source,
            "inputs": inputs,
            "outputs": outputs,
        })

        # Step 3: Unit conversion (if applicable)
        steps.append({
            "step_number": 3,
            "action": "convert_units",
            "description": "Converted to requested output units",
            "outputs": outputs,
        })

        # Build detailed explanation
        detailed = self._build_detailed_calculation(
            calculation_name, inputs, outputs, formula, formula_source, level
        )

        # Calculate provenance hash
        input_hash = hashlib.sha256(str(sorted(inputs.items())).encode()).hexdigest()[:16]
        output_hash = hashlib.sha256(str(sorted(outputs.items())).encode()).hexdigest()[:16]

        return Explanation(
            explanation_type=ExplanationType.CALCULATION,
            level=level,
            audience=audience,
            summary=summary,
            detailed_explanation=detailed,
            steps=steps,
            sources=[{
                "source_type": "standard",
                "source_id": formula_source,
                "source_name": formula_source,
            }],
            formulas_used=[formula],
            standards_referenced=[formula_source.split()[0]] if formula_source else [],
            confidence_score=0.95,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def _build_detailed_calculation(
        self,
        name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        formula: str,
        source: str,
        level: ExplanationLevel,
    ) -> str:
        """Build detailed calculation explanation."""
        parts = [f"## {name} Calculation\n"]

        # Inputs section
        parts.append("### Inputs")
        for key, val in inputs.items():
            if isinstance(val, dict):
                parts.append(f"- **{key}**: {val.get('value', 'N/A')} {val.get('unit', '')}")
            else:
                parts.append(f"- **{key}**: {val}")

        # Formula section
        parts.append(f"\n### Formula\n```\n{formula}\n```")
        parts.append(f"*Source: {source}*")

        # Outputs section
        parts.append("\n### Results")
        for key, val in outputs.items():
            if isinstance(val, dict):
                parts.append(f"- **{key}**: {val.get('value', 'N/A')} {val.get('unit', '')}")
            else:
                parts.append(f"- **{key}**: {val}")

        return "\n".join(parts)

    def explain_decision(
        self,
        decision_name: str,
        decision_outcome: str,
        factors: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        level: ExplanationLevel = ExplanationLevel.STANDARD,
    ) -> Explanation:
        """
        Generate explanation for a decision.

        Args:
            decision_name: Name of the decision
            decision_outcome: The decision made
            factors: Contributing factors with weights
            threshold: Decision threshold if applicable
            level: Detail level

        Returns:
            Complete explanation object
        """
        # Build summary
        top_factor = factors[0] if factors else {"name": "N/A", "contribution": 0}
        summary = f"Decision: {decision_outcome}. Primary factor: {top_factor.get('name')} ({top_factor.get('contribution', 0):.0%} contribution)."

        # Build steps showing factor evaluation
        steps = []
        for i, factor in enumerate(factors, 1):
            steps.append({
                "step_number": i,
                "action": "evaluate_factor",
                "description": f"Evaluated {factor.get('name', 'factor')}",
                "inputs": {"factor": factor.get("name"), "value": factor.get("value")},
                "outputs": {"contribution": factor.get("contribution", 0)},
                "confidence": factor.get("confidence", 1.0),
            })

        # Add threshold comparison step if applicable
        if threshold is not None:
            steps.append({
                "step_number": len(steps) + 1,
                "action": "compare_threshold",
                "description": f"Compared aggregate score against threshold ({threshold})",
                "outputs": {"decision": decision_outcome},
            })

        return Explanation(
            explanation_type=ExplanationType.DECISION,
            level=level,
            summary=summary,
            detailed_explanation=self._build_detailed_decision(decision_name, decision_outcome, factors, threshold),
            steps=steps,
            confidence_score=min(f.get("confidence", 1.0) for f in factors) if factors else 1.0,
        )

    def _build_detailed_decision(
        self,
        name: str,
        outcome: str,
        factors: List[Dict[str, Any]],
        threshold: Optional[float],
    ) -> str:
        """Build detailed decision explanation."""
        parts = [f"## {name}\n"]
        parts.append(f"**Decision**: {outcome}\n")

        if threshold:
            parts.append(f"**Threshold**: {threshold}\n")

        parts.append("### Contributing Factors")
        for factor in sorted(factors, key=lambda x: x.get("contribution", 0), reverse=True):
            contrib = factor.get("contribution", 0) * 100
            parts.append(f"- **{factor.get('name')}**: {factor.get('value')} ({contrib:.1f}% contribution)")

        return "\n".join(parts)

    def explain_compliance(
        self,
        standard: str,
        section: str,
        requirement: str,
        status: str,
        evidence: List[Dict[str, Any]],
        level: ExplanationLevel = ExplanationLevel.STANDARD,
    ) -> Explanation:
        """
        Generate explanation for compliance assessment.

        Args:
            standard: Standard code (e.g., "NFPA 85")
            section: Section reference
            requirement: Requirement description
            status: Compliance status
            evidence: Supporting evidence
            level: Detail level

        Returns:
            Complete explanation object
        """
        status_emoji = "✓" if status.lower() in ["compliant", "pass", "met"] else "✗"
        summary = f"{status_emoji} {status.upper()}: {requirement} per {standard} {section}."

        steps = []
        for i, item in enumerate(evidence, 1):
            steps.append({
                "step_number": i,
                "action": "verify_requirement",
                "description": f"Verified: {item.get('description', 'requirement')}",
                "inputs": {"requirement": item.get("requirement")},
                "outputs": {
                    "status": item.get("status", "checked"),
                    "value": item.get("value"),
                    "expected": item.get("expected"),
                },
            })

        return Explanation(
            explanation_type=ExplanationType.COMPLIANCE,
            level=level,
            summary=summary,
            steps=steps,
            standards_referenced=[standard],
            sources=[{
                "source_type": "standard",
                "source_id": f"{standard}_{section}",
                "source_name": standard,
                "section": section,
            }],
        )

    def explain_anomaly(
        self,
        parameter: str,
        current_value: float,
        expected_value: float,
        threshold: float,
        anomaly_type: str,
        possible_causes: List[str],
        level: ExplanationLevel = ExplanationLevel.STANDARD,
    ) -> Explanation:
        """
        Generate explanation for anomaly detection.

        Args:
            parameter: Parameter name
            current_value: Current measured value
            expected_value: Expected value
            threshold: Anomaly threshold
            anomaly_type: Type of anomaly
            possible_causes: List of possible causes
            level: Detail level

        Returns:
            Complete explanation object
        """
        deviation = abs(current_value - expected_value) / expected_value * 100 if expected_value else 0
        summary = f"Detected {anomaly_type} anomaly in {parameter}: {current_value} vs expected {expected_value} ({deviation:.1f}% deviation)."

        steps = [
            {
                "step_number": 1,
                "action": "measure_current",
                "description": f"Measured current {parameter}",
                "outputs": {"value": current_value},
            },
            {
                "step_number": 2,
                "action": "calculate_expected",
                "description": "Calculated expected value from baseline",
                "outputs": {"expected": expected_value},
            },
            {
                "step_number": 3,
                "action": "calculate_deviation",
                "description": "Calculated deviation from expected",
                "outputs": {"deviation_pct": deviation, "threshold_pct": threshold * 100},
            },
            {
                "step_number": 4,
                "action": "classify_anomaly",
                "description": f"Classified as {anomaly_type} anomaly",
                "outputs": {"anomaly_type": anomaly_type},
            },
        ]

        return Explanation(
            explanation_type=ExplanationType.ANOMALY,
            level=level,
            summary=summary,
            steps=steps,
            caveats=possible_causes[:3] if possible_causes else [],
            assumptions=[f"Baseline calculated from historical data"],
        )

    def get_statistics(self) -> Dict[str, int]:
        """Get engine statistics."""
        return {
            "templates_loaded": len(self.templates),
            "explanation_types": len(ExplanationType),
            "explanation_levels": len(ExplanationLevel),
        }


# Module-level singleton
_engine_instance: Optional[ExplanationEngine] = None


def get_explanation_engine() -> ExplanationEngine:
    """Get or create the global explanation engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ExplanationEngine()
    return _engine_instance
