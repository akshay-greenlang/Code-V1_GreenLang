"""GL-096: Cyber Shield Agent (CYBER-SHIELD).

Protects energy systems from cyber threats.

Standards: NIST CSF, IEC 62443
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SecurityControl(BaseModel):
    control_id: str
    name: str
    category: str
    implemented: bool = Field(default=False)
    effectiveness_pct: float = Field(ge=0, le=100)


class CyberShieldInput(BaseModel):
    system_id: str
    system_name: str = Field(default="Energy System")
    controls: List[SecurityControl] = Field(default_factory=list)
    ot_systems_count: int = Field(default=10, ge=0)
    it_ot_converged: bool = Field(default=True)
    remote_access_enabled: bool = Field(default=True)
    last_assessment_days: int = Field(default=365, ge=0)
    incidents_last_year: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SecurityGap(BaseModel):
    control_name: str
    category: str
    risk_level: str
    remediation_priority: str


class CyberShieldOutput(BaseModel):
    system_id: str
    security_score: float
    threat_level: ThreatLevel
    controls_implemented: int
    controls_total: int
    coverage_pct: float
    gaps: List[SecurityGap]
    risk_score: float
    compliance_status: str
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class CyberShieldAgent:
    AGENT_ID = "GL-096"
    AGENT_NAME = "CYBER-SHIELD"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"CyberShieldAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = CyberShieldInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: CyberShieldInput) -> CyberShieldOutput:
        recommendations = []
        gaps = []

        # Calculate security metrics
        implemented = [c for c in inp.controls if c.implemented]
        total = len(inp.controls)
        coverage = (len(implemented) / total * 100) if total > 0 else 0

        # Security score (weighted by effectiveness)
        if implemented:
            avg_effectiveness = sum(c.effectiveness_pct for c in implemented) / len(implemented)
            security_score = coverage * (avg_effectiveness / 100)
        else:
            security_score = 0

        # Risk score
        risk = 50  # Baseline
        if inp.it_ot_converged:
            risk += 10
        if inp.remote_access_enabled:
            risk += 10
        if inp.last_assessment_days > 180:
            risk += 10
        if inp.incidents_last_year > 0:
            risk += inp.incidents_last_year * 5
        risk -= security_score * 0.5
        risk = max(0, min(100, risk))

        # Threat level
        if risk >= 75:
            threat = ThreatLevel.CRITICAL
        elif risk >= 50:
            threat = ThreatLevel.HIGH
        elif risk >= 25:
            threat = ThreatLevel.MEDIUM
        else:
            threat = ThreatLevel.LOW

        # Identify gaps
        for control in inp.controls:
            if not control.implemented:
                if "access" in control.category.lower() or "authentication" in control.name.lower():
                    risk_level = "HIGH"
                    priority = "IMMEDIATE"
                elif "monitoring" in control.category.lower():
                    risk_level = "MEDIUM"
                    priority = "SHORT_TERM"
                else:
                    risk_level = "LOW"
                    priority = "PLANNED"

                gaps.append(SecurityGap(
                    control_name=control.name,
                    category=control.category,
                    risk_level=risk_level,
                    remediation_priority=priority
                ))

        # Compliance status
        if coverage >= 90:
            compliance = "COMPLIANT"
        elif coverage >= 70:
            compliance = "PARTIALLY_COMPLIANT"
        else:
            compliance = "NON_COMPLIANT"

        # Recommendations
        if inp.last_assessment_days > 365:
            recommendations.append(f"Security assessment overdue ({inp.last_assessment_days} days) - schedule immediately")
        if inp.remote_access_enabled:
            recommendations.append("Remote access enabled - ensure MFA and VPN are required")
        if inp.it_ot_converged:
            recommendations.append("IT/OT converged - implement network segmentation")

        high_gaps = [g for g in gaps if g.risk_level == "HIGH"]
        if high_gaps:
            recommendations.append(f"{len(high_gaps)} high-risk control gaps - prioritize remediation")

        if inp.incidents_last_year > 0:
            recommendations.append(f"{inp.incidents_last_year} incidents last year - conduct root cause analysis")

        if security_score < 50:
            recommendations.append("Low security score - implement foundational controls")

        calc_hash = hashlib.sha256(json.dumps({
            "system": inp.system_id,
            "score": round(security_score, 1),
            "threat": threat.value
        }).encode()).hexdigest()

        return CyberShieldOutput(
            system_id=inp.system_id,
            security_score=round(security_score, 1),
            threat_level=threat,
            controls_implemented=len(implemented),
            controls_total=total,
            coverage_pct=round(coverage, 1),
            gaps=gaps,
            risk_score=round(risk, 1),
            compliance_status=compliance,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-096", "name": "CYBER-SHIELD", "version": "1.0.0",
    "summary": "Cybersecurity protection for energy systems",
    "standards": [{"ref": "NIST CSF"}, {"ref": "IEC 62443"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
