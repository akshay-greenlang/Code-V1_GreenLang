"""GL-098: Interoperability Agent (INTEROPERABILITY).

Ensures system interoperability for energy platforms.

Standards: IEEE 2030, OpenADR, BACnet
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProtocolType(str, Enum):
    MODBUS = "MODBUS"
    BACNET = "BACNET"
    OPENADR = "OPENADR"
    OCPP = "OCPP"
    IEEE_2030 = "IEEE_2030"
    REST_API = "REST_API"


class SystemInterface(BaseModel):
    interface_id: str
    system_name: str
    protocol: ProtocolType
    version: str = Field(default="1.0")
    certified: bool = Field(default=False)
    connection_status: str = Field(default="UNKNOWN")
    data_exchange_success_pct: float = Field(default=0, ge=0, le=100)


class InteroperabilityInput(BaseModel):
    platform_id: str
    platform_name: str = Field(default="Energy Platform")
    interfaces: List[SystemInterface] = Field(default_factory=list)
    required_protocols: List[ProtocolType] = Field(default_factory=list)
    uptime_target_pct: float = Field(default=99.9, ge=0, le=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InterfaceStatus(BaseModel):
    interface_id: str
    system_name: str
    protocol: str
    health_score: float
    issues: List[str]


class InteroperabilityOutput(BaseModel):
    platform_id: str
    total_interfaces: int
    healthy_interfaces: int
    interface_health_pct: float
    protocol_coverage_pct: float
    interface_statuses: List[InterfaceStatus]
    missing_protocols: List[str]
    integration_score: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class InteroperabilityAgent:
    AGENT_ID = "GL-098B"
    AGENT_NAME = "INTEROPERABILITY"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"InteroperabilityAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = InteroperabilityInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: InteroperabilityInput) -> InteroperabilityOutput:
        recommendations = []
        statuses = []
        healthy = 0

        for interface in inp.interfaces:
            issues = []

            # Health score calculation
            score = interface.data_exchange_success_pct

            if interface.connection_status != "CONNECTED":
                score *= 0.5
                issues.append("Connection not established")

            if not interface.certified:
                score *= 0.9
                issues.append("Not certified")

            if interface.data_exchange_success_pct < inp.uptime_target_pct:
                issues.append(f"Below {inp.uptime_target_pct}% uptime target")

            if score >= 90:
                healthy += 1

            statuses.append(InterfaceStatus(
                interface_id=interface.interface_id,
                system_name=interface.system_name,
                protocol=interface.protocol.value,
                health_score=round(score, 1),
                issues=issues
            ))

        # Protocol coverage
        implemented = set(i.protocol for i in inp.interfaces)
        required = set(inp.required_protocols)
        missing = [p.value for p in required - implemented]
        coverage = ((len(required) - len(missing)) / len(required) * 100) if required else 100

        # Interface health
        total = len(inp.interfaces)
        health_pct = (healthy / total * 100) if total > 0 else 0

        # Integration score
        integration = (health_pct + coverage) / 2

        # Recommendations
        if missing:
            recommendations.append(f"Missing protocols: {', '.join(missing)}")

        unhealthy = [s for s in statuses if s.health_score < 90]
        if unhealthy:
            recommendations.append(f"{len(unhealthy)} interfaces need attention")

        uncertified = [i for i in inp.interfaces if not i.certified]
        if uncertified:
            recommendations.append(f"{len(uncertified)} interfaces not certified - consider certification")

        if integration < 80:
            recommendations.append("Integration score below 80% - improve connectivity")

        legacy = [i for i in inp.interfaces if i.protocol == ProtocolType.MODBUS]
        if len(legacy) > len(inp.interfaces) * 0.5:
            recommendations.append("High reliance on legacy protocols - plan modernization")

        calc_hash = hashlib.sha256(json.dumps({
            "platform": inp.platform_id,
            "integration": round(integration, 1),
            "healthy": healthy
        }).encode()).hexdigest()

        return InteroperabilityOutput(
            platform_id=inp.platform_id,
            total_interfaces=total,
            healthy_interfaces=healthy,
            interface_health_pct=round(health_pct, 1),
            protocol_coverage_pct=round(coverage, 1),
            interface_statuses=statuses,
            missing_protocols=missing,
            integration_score=round(integration, 1),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-098B", "name": "INTEROPERABILITY", "version": "1.0.0",
    "summary": "System interoperability management",
    "standards": [{"ref": "IEEE 2030"}, {"ref": "OpenADR"}, {"ref": "BACnet"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
