# -*- coding: utf-8 -*-
"""Compliance Explainer for GL-010 EmissionsGuardian"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
import hashlib, logging, uuid

from .schemas import (
    AudienceLevel, ConfidenceLevel, DecisionTrace, Explanation,
    ExplanationType, ReasoningStep, TemplateVersion,
)

logger = logging.getLogger(__name__)

SEVERITY_THRESHOLDS = {"minor": (0, 10), "moderate": (10, 25), "major": (25, 50), "critical": (50, 1000)}

CORRECTIVE_ACTIONS = {
    "minor": ["Review parameters", "Check calibration"],
    "moderate": ["Notify coordinator", "Root cause analysis"],
    "major": ["Notify plant manager", "Reduce load", "Submit deviation report"],
    "critical": ["EMERGENCY: Reduce operations", "Notify agency", "Executive notification"],
}


class ComplianceExplainer:
    """Explains compliance decisions with regulatory context."""

    def __init__(self):
        self.template_version = TemplateVersion(
            template_id="COMPLIANCE_V1", version="1.0.0",
            effective_date=datetime(2024, 1, 1), approved_by="GL-010",
            checksum=hashlib.sha256(b"compliance_v1").hexdigest()
        )

    def explain_exceedance(
        self, measured: float, limit: float, unit: str, pollutant: str, rule_id: str,
        audience: AudienceLevel = AudienceLevel.TECHNICAL
    ) -> Explanation:
        """Explain why a measurement is an exceedance."""
        pct = ((measured - limit) / limit) * 100
        severity = self._determine_severity(pct)
        
        steps = [
            ReasoningStep(step_number=1, step_type="input",
                description=f"Measured: {measured} {unit}",
                input_values={"measured": measured}, output_values={"pct": pct}),
            ReasoningStep(step_number=2, step_type="decision",
                description=f"Severity: {severity}", output_values={"severity": severity})
        ]
        
        trace = DecisionTrace(
            trace_id=str(uuid.uuid4()), decision_type="exceedance",
            decision_result=severity, confidence=1.0, confidence_level=ConfidenceLevel.VERY_HIGH,
            steps=steps, rules_evaluated=[rule_id], rules_triggered=[rule_id],
            input_data_hash=hashlib.sha256(str(measured).encode()).hexdigest(),
            output_data_hash=hashlib.sha256(severity.encode()).hexdigest(),
            start_time=datetime.now(), processing_time_ms=0.0, agent_id="GL-010", agent_version="1.0.0"
        )
        
        return Explanation(
            explanation_id=str(uuid.uuid4()), explanation_type=ExplanationType.EXCEEDANCE,
            audience_level=audience, title=f"{pollutant} Exceedance",
            summary=f"{measured} {unit} exceeds {limit} {unit} by {pct:.1f}%",
            detailed_explanation=f"Pollutant: {pollutant}, Measured: {measured}, Limit: {limit}, Severity: {severity}",
            key_findings=[f"Severity: {severity}", f"Exceedance: {pct:.1f}%"],
            decision_trace=trace, regulatory_citations=[f"Rule {rule_id}"],
            confidence=1.0, confidence_level=ConfidenceLevel.VERY_HIGH,
            template_version=self.template_version,
            provenance_hash=trace.calculate_provenance_hash(),
            generated_by="GL-010-ComplianceExplainer"
        )

    def get_corrective_actions(self, severity: str) -> List[str]:
        return CORRECTIVE_ACTIONS.get(severity, CORRECTIVE_ACTIONS["minor"])

    def _determine_severity(self, pct: float) -> str:
        for sev, (low, high) in SEVERITY_THRESHOLDS.items():
            if low <= pct < high:
                return sev
        return "critical"
