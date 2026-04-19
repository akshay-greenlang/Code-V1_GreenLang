"""
EUIED - EU Industrial Emissions Directive Compliance

This module implements compliance support for the EU Industrial
Emissions Directive (2010/75/EU) including:
- Best Available Techniques (BAT) assessment
- Emission Limit Values (ELVs)
- Permit requirements
- Monitoring requirements

Reference: Directive 2010/75/EU

Example:
    >>> from greenlang.safety.compliance.eu_ied import EUIED
    >>> ied = EUIED(installation_id="INST-001")
    >>> assessment = ied.assess_bat_compliance()
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class IEDActivity(str, Enum):
    """IED Annex I activities."""
    ENERGY = "energy"
    METALS = "metals"
    MINERALS = "minerals"
    CHEMICALS = "chemicals"
    WASTE = "waste"
    OTHER = "other"


class IEDRequirement(BaseModel):
    """IED requirement specification."""
    requirement_id: str = Field(default_factory=lambda: f"IED-{uuid.uuid4().hex[:6].upper()}")
    article: str = Field(..., description="IED Article reference")
    description: str = Field(...)
    bat_reference: Optional[str] = Field(None, description="BAT Reference Document")
    emission_limit: Optional[float] = Field(None)
    emission_unit: str = Field(default="mg/Nm3")
    monitoring_frequency: str = Field(default="continuous")


class BATAssessment(BaseModel):
    """BAT compliance assessment result."""
    assessment_id: str = Field(default_factory=lambda: f"BAT-{uuid.uuid4().hex[:8].upper()}")
    installation_id: str = Field(...)
    activity: IEDActivity = Field(...)
    assessment_date: datetime = Field(default_factory=datetime.utcnow)
    bat_conclusions_ref: str = Field(...)
    techniques_assessed: int = Field(default=0)
    techniques_implemented: int = Field(default=0)
    compliance_percent: float = Field(default=0.0)
    aels_met: Dict[str, bool] = Field(default_factory=dict)  # Associated Emission Levels
    derogations_required: List[str] = Field(default_factory=list)
    improvement_actions: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class EUIED:
    """
    EU IED Compliance Manager.

    Supports compliance with EU Industrial Emissions Directive
    including BAT assessment and ELV monitoring.

    Example:
        >>> ied = EUIED(installation_id="INST-001")
        >>> assessment = ied.assess_bat_compliance(activity=IEDActivity.ENERGY)
    """

    # BAT-Associated Emission Levels (BAT-AELs) examples
    BAT_AELS = {
        IEDActivity.ENERGY: {
            "NOx": {"min": 50, "max": 85, "unit": "mg/Nm3"},
            "SO2": {"min": 10, "max": 35, "unit": "mg/Nm3"},
            "dust": {"min": 2, "max": 5, "unit": "mg/Nm3"},
            "CO": {"min": 10, "max": 30, "unit": "mg/Nm3"},
        },
        IEDActivity.CHEMICALS: {
            "VOC": {"min": 5, "max": 20, "unit": "mg/Nm3"},
            "NOx": {"min": 50, "max": 100, "unit": "mg/Nm3"},
        },
    }

    def __init__(self, installation_id: str, operator: str = ""):
        """Initialize EUIED manager."""
        self.installation_id = installation_id
        self.operator = operator
        self.assessments: List[BATAssessment] = []
        logger.info(f"EUIED manager initialized for {installation_id}")

    def assess_bat_compliance(
        self,
        activity: IEDActivity,
        bat_conclusions_ref: str,
        current_emissions: Dict[str, float],
        techniques_implemented: List[str]
    ) -> BATAssessment:
        """
        Assess BAT compliance for installation.

        Args:
            activity: IED activity category
            bat_conclusions_ref: BAT Conclusions reference
            current_emissions: Current emission values
            techniques_implemented: List of implemented BAT

        Returns:
            BATAssessment result
        """
        logger.info(f"Assessing BAT compliance for {self.installation_id}")

        aels = self.BAT_AELS.get(activity, {})
        aels_met = {}
        derogations = []
        improvements = []

        # Check each AEL
        for pollutant, limits in aels.items():
            current = current_emissions.get(pollutant)
            if current is not None:
                max_ael = limits["max"]
                aels_met[pollutant] = current <= max_ael

                if current > max_ael:
                    derogations.append(
                        f"{pollutant}: {current} > BAT-AEL max {max_ael} {limits['unit']}"
                    )
                    improvements.append(
                        f"Reduce {pollutant} emissions to meet BAT-AEL"
                    )

        # Calculate compliance
        total_aels = len(aels)
        met_count = sum(1 for met in aels_met.values() if met)
        compliance_percent = (met_count / total_aels * 100) if total_aels > 0 else 100

        assessment = BATAssessment(
            installation_id=self.installation_id,
            activity=activity,
            bat_conclusions_ref=bat_conclusions_ref,
            techniques_assessed=len(aels),
            techniques_implemented=len(techniques_implemented),
            compliance_percent=compliance_percent,
            aels_met=aels_met,
            derogations_required=derogations,
            improvement_actions=improvements,
        )

        assessment.provenance_hash = hashlib.sha256(
            f"{assessment.assessment_id}|{self.installation_id}|{compliance_percent}".encode()
        ).hexdigest()

        self.assessments.append(assessment)
        return assessment

    def get_monitoring_requirements(
        self,
        activity: IEDActivity
    ) -> List[IEDRequirement]:
        """Get IED monitoring requirements for activity."""
        requirements = []

        # Common monitoring requirements
        if activity == IEDActivity.ENERGY:
            requirements.extend([
                IEDRequirement(
                    article="Article 14",
                    description="Continuous emission monitoring for NOx, SO2, dust",
                    monitoring_frequency="continuous"
                ),
                IEDRequirement(
                    article="Article 15",
                    description="Annual compliance assessment",
                    monitoring_frequency="annual"
                ),
            ])

        return requirements
