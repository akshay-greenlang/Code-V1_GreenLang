# -*- coding: utf-8 -*-
"""
GreenLang Adaptation Agriculture Sector Agents
===============================================

Agriculture climate adaptation agents for resilient farming,
water management, and crop adaptation strategies.

Agents:
    GL-ADAPT-AGR-001 to GL-ADAPT-AGR-008
"""

from greenlang.agents.adaptation.agriculture.agents import (
    DroughtResilienceAgricultureAgent,
    FloodResilienceAgricultureAgent,
    HeatStressCropAgent,
    PestDiseaseClimateAgent,
    LivestockHeatStressAgent,
    WaterAvailabilityAgricultureAgent,
    CropCalendarShiftAgent,
    AgricultureClimateRiskAssessmentAgent,
)

__all__ = [
    "DroughtResilienceAgricultureAgent",
    "FloodResilienceAgricultureAgent",
    "HeatStressCropAgent",
    "PestDiseaseClimateAgent",
    "LivestockHeatStressAgent",
    "WaterAvailabilityAgricultureAgent",
    "CropCalendarShiftAgent",
    "AgricultureClimateRiskAssessmentAgent",
]

AGENT_REGISTRY = {
    "GL-ADAPT-AGR-001": DroughtResilienceAgricultureAgent,
    "GL-ADAPT-AGR-002": FloodResilienceAgricultureAgent,
    "GL-ADAPT-AGR-003": HeatStressCropAgent,
    "GL-ADAPT-AGR-004": PestDiseaseClimateAgent,
    "GL-ADAPT-AGR-005": LivestockHeatStressAgent,
    "GL-ADAPT-AGR-006": WaterAvailabilityAgricultureAgent,
    "GL-ADAPT-AGR-007": CropCalendarShiftAgent,
    "GL-ADAPT-AGR-008": AgricultureClimateRiskAssessmentAgent,
}
