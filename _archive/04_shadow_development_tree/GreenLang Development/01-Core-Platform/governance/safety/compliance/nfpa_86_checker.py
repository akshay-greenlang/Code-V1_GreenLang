"""
NFPA86Checker - NFPA 86 Furnace Compliance

This module implements compliance checking for NFPA 86:
Standard for Ovens and Furnaces.

Key requirements covered:
- Furnace classification (Class A/B/C/D)
- Safety equipment requirements
- Ventilation requirements
- Atmosphere control

Reference: NFPA 86-2019

Example:
    >>> from greenlang.safety.compliance.nfpa_86_checker import NFPA86Checker
    >>> checker = NFPA86Checker()
    >>> result = checker.check_compliance(furnace_config)
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class FurnaceClassification(str, Enum):
    """Furnace classifications per NFPA 86."""
    CLASS_A = "class_a"  # Ovens with flammable volatiles
    CLASS_B = "class_b"  # Ovens with heated flammable materials
    CLASS_C = "class_c"  # Atmosphere furnaces with special atmospheres
    CLASS_D = "class_d"  # Vacuum furnaces


class AtmosphereType(str, Enum):
    """Furnace atmosphere types."""
    AIR = "air"
    NITROGEN = "nitrogen"
    HYDROGEN = "hydrogen"
    ENDOTHERMIC = "endothermic"
    EXOTHERMIC = "exothermic"
    VACUUM = "vacuum"


class NFPA86CheckResult(BaseModel):
    """Result of NFPA 86 compliance check."""
    check_id: str = Field(default_factory=lambda: f"F86-{uuid.uuid4().hex[:8].upper()}")
    equipment_id: str = Field(...)
    classification: FurnaceClassification = Field(...)
    check_date: datetime = Field(default_factory=datetime.utcnow)
    requirements_checked: int = Field(default=0)
    requirements_passed: int = Field(default=0)
    requirements_failed: int = Field(default=0)
    compliance_percent: float = Field(default=0.0)
    is_compliant: bool = Field(default=False)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class NFPA86Checker:
    """
    NFPA 86 Compliance Checker.

    Verifies compliance with NFPA 86 requirements for
    industrial ovens and furnaces.

    Example:
        >>> checker = NFPA86Checker()
        >>> result = checker.check_furnace_compliance(config)
    """

    # Requirements by classification
    CLASS_REQUIREMENTS = {
        FurnaceClassification.CLASS_A: {
            "ventilation_rate_min": 10,  # CFM per sq ft
            "lel_monitoring": True,
            "temperature_limit_monitoring": True,
            "safety_relief": True,
        },
        FurnaceClassification.CLASS_B: {
            "fire_suppression": True,
            "temperature_monitoring": True,
            "emergency_shutdown": True,
        },
        FurnaceClassification.CLASS_C: {
            "atmosphere_monitoring": True,
            "purge_capability": True,
            "burn_off_system": True,
            "pressure_relief": True,
        },
        FurnaceClassification.CLASS_D: {
            "vacuum_integrity": True,
            "leak_detection": True,
            "quench_system": True,
        },
    }

    def __init__(self):
        """Initialize NFPA86Checker."""
        self.check_history: List[NFPA86CheckResult] = []
        logger.info("NFPA86Checker initialized")

    def check_furnace_compliance(
        self,
        equipment_id: str,
        classification: FurnaceClassification,
        furnace_config: Dict[str, Any]
    ) -> NFPA86CheckResult:
        """
        Check furnace configuration against NFPA 86.

        Args:
            equipment_id: Equipment identifier
            classification: Furnace classification
            furnace_config: Furnace configuration

        Returns:
            NFPA86CheckResult
        """
        logger.info(f"Checking NFPA 86 compliance for {equipment_id}")

        requirements = self.CLASS_REQUIREMENTS.get(classification, {})
        findings = []
        passed = 0
        failed = 0

        for req_name, req_value in requirements.items():
            config_value = furnace_config.get(req_name)

            if isinstance(req_value, bool):
                is_compliant = config_value == req_value
            elif isinstance(req_value, (int, float)):
                is_compliant = config_value is not None and config_value >= req_value
            else:
                is_compliant = config_value == req_value

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

        total = passed + failed
        compliance_percent = (passed / total * 100) if total > 0 else 0

        result = NFPA86CheckResult(
            equipment_id=equipment_id,
            classification=classification,
            requirements_checked=total,
            requirements_passed=passed,
            requirements_failed=failed,
            compliance_percent=compliance_percent,
            is_compliant=failed == 0,
            findings=findings,
        )

        result.provenance_hash = hashlib.sha256(
            f"{result.check_id}|{equipment_id}|{classification.value}".encode()
        ).hexdigest()

        self.check_history.append(result)
        return result

    def classify_furnace(
        self,
        has_volatiles: bool,
        has_flammable_materials: bool,
        special_atmosphere: bool,
        is_vacuum: bool
    ) -> FurnaceClassification:
        """
        Determine furnace classification per NFPA 86.

        Args:
            has_volatiles: Process produces flammable volatiles
            has_flammable_materials: Contains heated flammable materials
            special_atmosphere: Uses special atmospheres
            is_vacuum: Is a vacuum furnace

        Returns:
            FurnaceClassification
        """
        if is_vacuum:
            return FurnaceClassification.CLASS_D
        elif special_atmosphere:
            return FurnaceClassification.CLASS_C
        elif has_flammable_materials:
            return FurnaceClassification.CLASS_B
        elif has_volatiles:
            return FurnaceClassification.CLASS_A
        else:
            return FurnaceClassification.CLASS_A  # Default
