"""GL-097: Regulatory Tracker Agent (REGULATORY-TRACKER).

Tracks regulatory requirements for energy systems.

Standards: EPA, DOE, State Regulations
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RegulationType(str, Enum):
    FEDERAL = "FEDERAL"
    STATE = "STATE"
    LOCAL = "LOCAL"
    INDUSTRY = "INDUSTRY"


class ComplianceStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    PARTIAL = "PARTIAL"
    NON_COMPLIANT = "NON_COMPLIANT"
    PENDING = "PENDING"


class Regulation(BaseModel):
    regulation_id: str
    name: str
    regulation_type: RegulationType
    effective_date: datetime
    compliance_deadline: Optional[datetime] = None
    current_status: ComplianceStatus = Field(default=ComplianceStatus.PENDING)
    penalty_risk_usd: float = Field(default=0, ge=0)


class RegulatoryTrackerInput(BaseModel):
    facility_id: str
    jurisdiction: str = Field(default="US-CA")
    regulations: List[Regulation] = Field(default_factory=list)
    audit_date: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplianceItem(BaseModel):
    regulation_name: str
    regulation_type: str
    status: str
    days_until_deadline: Optional[int]
    risk_usd: float
    action_required: str


class RegulatoryTrackerOutput(BaseModel):
    facility_id: str
    jurisdiction: str
    total_regulations: int
    compliant_count: int
    non_compliant_count: int
    compliance_rate_pct: float
    total_penalty_exposure_usd: float
    compliance_items: List[ComplianceItem]
    upcoming_deadlines: List[str]
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class RegulatoryTrackerAgent:
    AGENT_ID = "GL-097B"
    AGENT_NAME = "REGULATORY-TRACKER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"RegulatoryTrackerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = RegulatoryTrackerInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: RegulatoryTrackerInput) -> RegulatoryTrackerOutput:
        recommendations = []
        items = []
        upcoming = []
        now = datetime.utcnow()

        compliant = 0
        non_compliant = 0
        total_exposure = 0

        for reg in inp.regulations:
            # Days until deadline
            if reg.compliance_deadline:
                days = (reg.compliance_deadline - now).days
            else:
                days = None

            # Determine action
            if reg.current_status == ComplianceStatus.COMPLIANT:
                action = "Maintain compliance"
                compliant += 1
            elif reg.current_status == ComplianceStatus.PARTIAL:
                action = "Complete remaining requirements"
            elif reg.current_status == ComplianceStatus.NON_COMPLIANT:
                action = "Immediate remediation required"
                non_compliant += 1
                total_exposure += reg.penalty_risk_usd
            else:
                action = "Assess compliance status"

            items.append(ComplianceItem(
                regulation_name=reg.name,
                regulation_type=reg.regulation_type.value,
                status=reg.current_status.value,
                days_until_deadline=days,
                risk_usd=reg.penalty_risk_usd,
                action_required=action
            ))

            # Track upcoming deadlines
            if days is not None and 0 < days <= 90:
                upcoming.append(f"{reg.name}: {days} days")

        # Compliance rate
        total = len(inp.regulations)
        rate = (compliant / total * 100) if total > 0 else 100

        # Recommendations
        if non_compliant > 0:
            recommendations.append(f"URGENT: {non_compliant} regulations non-compliant - ${total_exposure:,.0f} penalty exposure")

        urgent = [i for i in items if i.days_until_deadline and i.days_until_deadline < 30]
        if urgent:
            recommendations.append(f"{len(urgent)} deadlines within 30 days - prioritize compliance activities")

        if rate < 80:
            recommendations.append("Compliance rate below 80% - implement compliance management program")

        federal_non = [i for i in items if i.regulation_type == "FEDERAL" and i.status == "NON_COMPLIANT"]
        if federal_non:
            recommendations.append("Federal non-compliance carries higher penalties - address first")

        if not inp.audit_date or (now - inp.audit_date).days > 365:
            recommendations.append("Compliance audit overdue - schedule comprehensive review")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "rate": round(rate, 1),
            "exposure": round(total_exposure, 2)
        }).encode()).hexdigest()

        return RegulatoryTrackerOutput(
            facility_id=inp.facility_id,
            jurisdiction=inp.jurisdiction,
            total_regulations=total,
            compliant_count=compliant,
            non_compliant_count=non_compliant,
            compliance_rate_pct=round(rate, 1),
            total_penalty_exposure_usd=round(total_exposure, 2),
            compliance_items=items,
            upcoming_deadlines=upcoming,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-097B", "name": "REGULATORY-TRACKER", "version": "1.0.0",
    "summary": "Regulatory compliance tracking",
    "standards": [{"ref": "EPA"}, {"ref": "DOE"}, {"ref": "State Regulations"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
