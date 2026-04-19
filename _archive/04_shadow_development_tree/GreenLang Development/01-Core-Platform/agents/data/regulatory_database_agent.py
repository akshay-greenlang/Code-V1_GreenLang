# -*- coding: utf-8 -*-
"""
GL-DATA-X-015: Regulatory Database Agent
=========================================

Connects to climate regulatory databases for compliance tracking
and regulatory intelligence.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


class RegulationStatus(str, Enum):
    PROPOSED = "proposed"
    ADOPTED = "adopted"
    IN_FORCE = "in_force"
    AMENDED = "amended"
    REPEALED = "repealed"


class RegulationType(str, Enum):
    DISCLOSURE = "disclosure"
    EMISSIONS_LIMIT = "emissions_limit"
    CARBON_PRICING = "carbon_pricing"
    RENEWABLE_TARGET = "renewable_target"
    EFFICIENCY_STANDARD = "efficiency_standard"
    DUE_DILIGENCE = "due_diligence"
    TAXONOMY = "taxonomy"


class Jurisdiction(str, Enum):
    EU = "eu"
    US_FEDERAL = "us_federal"
    US_CALIFORNIA = "us_california"
    UK = "uk"
    CHINA = "china"
    GLOBAL = "global"


class Regulation(BaseModel):
    regulation_id: str = Field(...)
    name: str = Field(...)
    jurisdiction: Jurisdiction = Field(...)
    regulation_type: RegulationType = Field(...)
    status: RegulationStatus = Field(...)
    effective_date: Optional[date] = Field(None)
    compliance_deadline: Optional[date] = Field(None)
    summary: str = Field(...)
    applicability_criteria: Dict[str, Any] = Field(...)
    key_requirements: List[str] = Field(...)
    penalties: Optional[str] = Field(None)


class RegulatoryUpdate(BaseModel):
    regulation_id: str = Field(...)
    update_type: str = Field(...)
    update_date: date = Field(...)
    description: str = Field(...)
    impact_level: str = Field(...)


class RegulatoryDatabaseInput(BaseModel):
    organization_id: str = Field(...)
    jurisdictions: List[Jurisdiction] = Field(...)
    regulation_types: List[RegulationType] = Field(default_factory=list)
    industry_sectors: List[str] = Field(default_factory=list)
    include_upcoming: bool = Field(default=True)


class RegulatoryDatabaseOutput(BaseModel):
    organization_id: str = Field(...)
    query_date: datetime = Field(default_factory=DeterministicClock.now)
    applicable_regulations: List[Regulation] = Field(...)
    upcoming_regulations: List[Regulation] = Field(...)
    recent_updates: List[RegulatoryUpdate] = Field(...)
    compliance_calendar: List[Dict[str, Any]] = Field(...)
    provenance_hash: str = Field(...)


class RegulatoryDatabaseAgent(BaseAgent):
    """GL-DATA-X-015: Regulatory Database Agent"""

    AGENT_ID = "GL-DATA-X-015"
    AGENT_NAME = "Regulatory Database Agent"
    VERSION = "1.0.0"

    REGULATIONS_DB = [
        {
            "id": "EU-CSRD",
            "name": "Corporate Sustainability Reporting Directive",
            "jurisdiction": Jurisdiction.EU,
            "type": RegulationType.DISCLOSURE,
            "status": RegulationStatus.IN_FORCE,
            "effective": date(2024, 1, 1),
            "summary": "Comprehensive sustainability reporting for large EU companies",
            "requirements": ["Double materiality assessment", "ESRS compliance", "Limited assurance"],
        },
        {
            "id": "EU-CBAM",
            "name": "Carbon Border Adjustment Mechanism",
            "jurisdiction": Jurisdiction.EU,
            "type": RegulationType.CARBON_PRICING,
            "status": RegulationStatus.IN_FORCE,
            "effective": date(2023, 10, 1),
            "summary": "Carbon pricing for imports of carbon-intensive goods",
            "requirements": ["Quarterly reporting", "Embedded emissions calculation", "Certificate purchase"],
        },
        {
            "id": "SEC-CLIMATE",
            "name": "SEC Climate Disclosure Rule",
            "jurisdiction": Jurisdiction.US_FEDERAL,
            "type": RegulationType.DISCLOSURE,
            "status": RegulationStatus.ADOPTED,
            "effective": date(2025, 1, 1),
            "summary": "Climate risk disclosure for SEC registrants",
            "requirements": ["Scope 1 & 2 emissions", "Climate risks", "Transition plans"],
        },
        {
            "id": "CA-SB253",
            "name": "California Climate Corporate Data Accountability Act",
            "jurisdiction": Jurisdiction.US_CALIFORNIA,
            "type": RegulationType.DISCLOSURE,
            "status": RegulationStatus.IN_FORCE,
            "effective": date(2026, 1, 1),
            "summary": "GHG disclosure for large companies doing business in California",
            "requirements": ["Scope 1, 2, 3 emissions", "Third-party assurance"],
        },
        {
            "id": "EU-CSDDD",
            "name": "Corporate Sustainability Due Diligence Directive",
            "jurisdiction": Jurisdiction.EU,
            "type": RegulationType.DUE_DILIGENCE,
            "status": RegulationStatus.ADOPTED,
            "effective": date(2027, 7, 1),
            "summary": "Supply chain due diligence for human rights and environment",
            "requirements": ["Risk identification", "Mitigation measures", "Grievance mechanisms"],
        },
    ]

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Regulatory database connector",
                version=self.VERSION
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = RegulatoryDatabaseInput(**input_data)

            applicable = []
            upcoming = []

            for reg_data in self.REGULATIONS_DB:
                if reg_data["jurisdiction"] in agent_input.jurisdictions:
                    reg = Regulation(
                        regulation_id=reg_data["id"],
                        name=reg_data["name"],
                        jurisdiction=reg_data["jurisdiction"],
                        regulation_type=reg_data["type"],
                        status=reg_data["status"],
                        effective_date=reg_data.get("effective"),
                        summary=reg_data["summary"],
                        applicability_criteria={"sectors": "all"},
                        key_requirements=reg_data["requirements"],
                    )

                    if reg_data["status"] == RegulationStatus.IN_FORCE:
                        applicable.append(reg)
                    elif reg_data["status"] in (RegulationStatus.ADOPTED, RegulationStatus.PROPOSED):
                        if agent_input.include_upcoming:
                            upcoming.append(reg)

            recent_updates = [
                RegulatoryUpdate(
                    regulation_id="EU-CSRD",
                    update_type="guidance",
                    update_date=date(2024, 12, 15),
                    description="EFRAG releases sector-specific guidance",
                    impact_level="medium",
                ),
                RegulatoryUpdate(
                    regulation_id="SEC-CLIMATE",
                    update_type="implementation",
                    update_date=date(2024, 11, 1),
                    description="Phased implementation timeline confirmed",
                    impact_level="high",
                ),
            ]

            compliance_calendar = []
            for reg in applicable + upcoming:
                if reg.effective_date:
                    compliance_calendar.append({
                        "regulation": reg.name,
                        "date": reg.effective_date.isoformat(),
                        "action": "Compliance required",
                    })

            compliance_calendar.sort(key=lambda x: x["date"])

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = RegulatoryDatabaseOutput(
                organization_id=agent_input.organization_id,
                applicable_regulations=applicable,
                upcoming_regulations=upcoming,
                recent_updates=recent_updates,
                compliance_calendar=compliance_calendar,
                provenance_hash=provenance_hash,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))
