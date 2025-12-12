"""GL-096: Cybersecurity Agent (CYBERSECURITY).

Manages cybersecurity for industrial energy systems.

Standards: NERC CIP, IEC 62443
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AssetCriticality(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CyberAsset(BaseModel):
    asset_id: str
    asset_name: str
    criticality: AssetCriticality = Field(default=AssetCriticality.MEDIUM)
    patched: bool = Field(default=True)
    monitored: bool = Field(default=True)
    backup_available: bool = Field(default=True)


class CybersecurityInput(BaseModel):
    facility_id: str
    assets: List[CyberAsset] = Field(default_factory=list)
    network_segmented: bool = Field(default=True)
    mfa_enabled: bool = Field(default=True)
    incident_response_plan: bool = Field(default=True)
    training_current: bool = Field(default=True)
    vulnerability_scan_days: int = Field(default=30, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VulnerabilityFinding(BaseModel):
    asset_id: str
    finding: str
    severity: str
    recommendation: str


class CybersecurityOutput(BaseModel):
    facility_id: str
    maturity_level: int
    maturity_score: float
    critical_assets: int
    assets_at_risk: int
    vulnerabilities: List[VulnerabilityFinding]
    compliance_gaps: List[str]
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class CybersecurityAgent:
    AGENT_ID = "GL-096B"
    AGENT_NAME = "CYBERSECURITY"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"CybersecurityAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = CybersecurityInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: CybersecurityInput) -> CybersecurityOutput:
        recommendations = []
        vulnerabilities = []
        compliance_gaps = []

        # Count critical assets
        critical = sum(1 for a in inp.assets if a.criticality in [AssetCriticality.HIGH, AssetCriticality.CRITICAL])

        # Identify at-risk assets
        at_risk = 0
        for asset in inp.assets:
            risks = []
            if not asset.patched:
                risks.append("Unpatched")
                vulnerabilities.append(VulnerabilityFinding(
                    asset_id=asset.asset_id,
                    finding="Missing security patches",
                    severity="HIGH" if asset.criticality in [AssetCriticality.HIGH, AssetCriticality.CRITICAL] else "MEDIUM",
                    recommendation="Apply latest security patches"
                ))
            if not asset.monitored:
                risks.append("Not monitored")
                vulnerabilities.append(VulnerabilityFinding(
                    asset_id=asset.asset_id,
                    finding="No security monitoring",
                    severity="MEDIUM",
                    recommendation="Enable security monitoring"
                ))
            if not asset.backup_available:
                risks.append("No backup")

            if risks:
                at_risk += 1

        # Maturity scoring
        score = 0
        if inp.network_segmented:
            score += 20
        else:
            compliance_gaps.append("Network segmentation not implemented")
        if inp.mfa_enabled:
            score += 20
        else:
            compliance_gaps.append("Multi-factor authentication not enabled")
        if inp.incident_response_plan:
            score += 20
        else:
            compliance_gaps.append("No incident response plan")
        if inp.training_current:
            score += 15
        else:
            compliance_gaps.append("Security training not current")
        if inp.vulnerability_scan_days <= 30:
            score += 15
        elif inp.vulnerability_scan_days <= 90:
            score += 10
        else:
            compliance_gaps.append("Vulnerability scanning overdue")

        # Asset health contribution
        if inp.assets:
            healthy_pct = (len(inp.assets) - at_risk) / len(inp.assets) * 100
            score += healthy_pct * 0.1

        # Maturity level (1-5)
        if score >= 90:
            level = 5
        elif score >= 75:
            level = 4
        elif score >= 60:
            level = 3
        elif score >= 40:
            level = 2
        else:
            level = 1

        # Recommendations
        if not inp.network_segmented:
            recommendations.append("Implement network segmentation between IT and OT")
        if not inp.mfa_enabled:
            recommendations.append("Enable MFA for all remote access")
        if at_risk > 0:
            recommendations.append(f"{at_risk} assets at risk - prioritize remediation")
        if inp.vulnerability_scan_days > 30:
            recommendations.append(f"Vulnerability scan {inp.vulnerability_scan_days} days old - scan immediately")
        if level < 3:
            recommendations.append("Maturity level below target - implement foundational controls")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "score": round(score, 1),
            "level": level
        }).encode()).hexdigest()

        return CybersecurityOutput(
            facility_id=inp.facility_id,
            maturity_level=level,
            maturity_score=round(score, 1),
            critical_assets=critical,
            assets_at_risk=at_risk,
            vulnerabilities=vulnerabilities,
            compliance_gaps=compliance_gaps,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-096B", "name": "CYBERSECURITY", "version": "1.0.0",
    "summary": "Industrial cybersecurity management",
    "standards": [{"ref": "NERC CIP"}, {"ref": "IEC 62443"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
