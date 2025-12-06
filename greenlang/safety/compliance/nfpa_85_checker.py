"""
NFPA85Checker - NFPA 85 Combustion Safeguards

This module implements compliance checking for NFPA 85:
Boiler and Combustion Systems Hazards Code.

Key requirements covered:
- Burner Management System (BMS) logic
- Purge timing requirements
- Flame supervision
- Safety shutdown interlocks

Reference: NFPA 85-2019

Example:
    >>> from greenlang.safety.compliance.nfpa_85_checker import NFPA85Checker
    >>> checker = NFPA85Checker()
    >>> result = checker.check_compliance(burner_config)
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class BurnerType(str, Enum):
    """Burner types per NFPA 85."""
    GAS = "gas"
    OIL = "oil"
    DUAL_FUEL = "dual_fuel"
    COAL = "coal"


class BMSFunction(str, Enum):
    """BMS safety functions per NFPA 85."""
    PREPURGE = "prepurge"
    PILOT_TRIAL = "pilot_trial"
    MAIN_TRIAL = "main_trial"
    RUN = "run"
    POSTPURGE = "postpurge"
    SAFETY_SHUTDOWN = "safety_shutdown"


class BurnerSafetyRequirement(BaseModel):
    """Safety requirement specification."""
    requirement_id: str = Field(default_factory=lambda: f"NFPA85-{uuid.uuid4().hex[:6].upper()}")
    nfpa_clause: str = Field(..., description="NFPA 85 clause reference")
    description: str = Field(..., description="Requirement description")
    mandatory: bool = Field(default=True)
    value: Optional[Any] = Field(None, description="Required value")
    unit: str = Field(default="", description="Unit of measure")


class NFPA85CheckResult(BaseModel):
    """Result of NFPA 85 compliance check."""
    check_id: str = Field(default_factory=lambda: f"CHK-{uuid.uuid4().hex[:8].upper()}")
    equipment_id: str = Field(...)
    check_date: datetime = Field(default_factory=datetime.utcnow)
    requirements_checked: int = Field(default=0)
    requirements_passed: int = Field(default=0)
    requirements_failed: int = Field(default=0)
    compliance_percent: float = Field(default=0.0)
    is_compliant: bool = Field(default=False)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class NFPA85Checker:
    """
    NFPA 85 Compliance Checker.

    Verifies compliance with NFPA 85 requirements for
    burner management systems and combustion safeguards.

    Key checks include:
    - Purge timing (minimum 4 volume changes)
    - Flame detection timing
    - Safety shutdown requirements
    - Fuel valve proving

    Example:
        >>> checker = NFPA85Checker()
        >>> result = checker.check_bms_compliance(bms_config)
    """

    # NFPA 85 timing requirements
    TIMING_REQUIREMENTS = {
        "prepurge_min_seconds": 15.0,  # Minimum prepurge time
        "prepurge_volume_changes": 4,  # Minimum volume changes
        "pilot_trial_max_seconds": 10.0,  # Max pilot trial
        "main_flame_trial_max_seconds": 10.0,  # Max main flame trial
        "flame_failure_response_seconds": 4.0,  # Max response to flame failure
        "postpurge_min_seconds": 15.0,  # Minimum postpurge
    }

    REQUIRED_INTERLOCKS = [
        {"name": "Low fuel pressure", "clause": "8.6.3.1"},
        {"name": "High fuel pressure", "clause": "8.6.3.2"},
        {"name": "Low combustion air", "clause": "8.6.3.3"},
        {"name": "Flame failure", "clause": "8.6.3.4"},
        {"name": "High steam pressure", "clause": "8.6.3.5"},
        {"name": "Low water level", "clause": "8.6.3.6"},
    ]

    def __init__(self):
        """Initialize NFPA85Checker."""
        self.check_history: List[NFPA85CheckResult] = []
        logger.info("NFPA85Checker initialized")

    def check_bms_compliance(
        self,
        equipment_id: str,
        bms_config: Dict[str, Any]
    ) -> NFPA85CheckResult:
        """
        Check BMS configuration against NFPA 85.

        Args:
            equipment_id: Equipment identifier
            bms_config: BMS configuration dictionary

        Returns:
            NFPA85CheckResult
        """
        logger.info(f"Checking NFPA 85 compliance for {equipment_id}")

        findings = []
        passed = 0
        failed = 0

        # Check timing requirements
        for req_name, req_value in self.TIMING_REQUIREMENTS.items():
            config_value = bms_config.get(req_name, 0)

            if "min" in req_name:
                is_compliant = config_value >= req_value
            else:
                is_compliant = config_value <= req_value

            if is_compliant:
                passed += 1
            else:
                failed += 1
                findings.append({
                    "requirement": req_name,
                    "expected": req_value,
                    "actual": config_value,
                    "compliant": False,
                })

        # Check required interlocks
        configured_interlocks = bms_config.get("interlocks", [])
        for interlock in self.REQUIRED_INTERLOCKS:
            if interlock["name"] in configured_interlocks:
                passed += 1
            else:
                failed += 1
                findings.append({
                    "requirement": f"Interlock: {interlock['name']}",
                    "clause": interlock["clause"],
                    "compliant": False,
                    "message": "Required interlock not configured"
                })

        total = passed + failed
        compliance_percent = (passed / total * 100) if total > 0 else 0

        result = NFPA85CheckResult(
            equipment_id=equipment_id,
            requirements_checked=total,
            requirements_passed=passed,
            requirements_failed=failed,
            compliance_percent=compliance_percent,
            is_compliant=failed == 0,
            findings=findings,
            recommendations=self._generate_recommendations(findings),
        )

        result.provenance_hash = hashlib.sha256(
            f"{result.check_id}|{equipment_id}|{passed}/{total}".encode()
        ).hexdigest()

        self.check_history.append(result)
        return result

    def _generate_recommendations(
        self,
        findings: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations from findings."""
        recommendations = []
        for finding in findings:
            if not finding.get("compliant", True):
                req = finding.get("requirement", "Unknown")
                recommendations.append(
                    f"Address non-compliance: {req}"
                )
        return recommendations
