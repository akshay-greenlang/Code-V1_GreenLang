# -*- coding: utf-8 -*-
"""
GreenLang Water MRV Agents
==========================

This package provides MRV (Monitoring, Reporting, Verification) agents
for water sector emissions measurement and reporting.

Agents:
    GL-MRV-WAT-001: WaterSupplyMRVAgent - Water treatment/distribution emissions
    GL-MRV-WAT-002: WastewaterMRVAgent - Wastewater treatment emissions
    GL-MRV-WAT-003: DesalinationMRVAgent - Desalination energy emissions
    GL-MRV-WAT-004: IrrigationMRVAgent - Agricultural water use emissions
    GL-MRV-WAT-005: IndustrialWaterMRVAgent - Industrial water process emissions

All agents follow the CRITICAL PATH pattern with:
    - Zero-hallucination guarantee (no LLM in calculation path)
    - Full audit trail with SHA-256 provenance hashing
    - GHG Protocol and regulatory compliance
    - Deterministic, reproducible calculations
"""

from greenlang.agents.mrv.water.water_supply import (
    WaterSupplyMRVAgent,
    WaterSupplyInput,
    WaterSupplyOutput,
    WaterTreatmentRecord,
    DistributionRecord,
)
from greenlang.agents.mrv.water.wastewater import (
    WastewaterMRVAgent,
    WastewaterInput,
    WastewaterOutput,
    WastewaterTreatmentRecord,
)
from greenlang.agents.mrv.water.desalination import (
    DesalinationMRVAgent,
    DesalinationInput,
    DesalinationOutput,
    DesalinationPlantRecord,
)
from greenlang.agents.mrv.water.irrigation import (
    IrrigationMRVAgent,
    IrrigationInput,
    IrrigationOutput,
    IrrigationSystemRecord,
)
from greenlang.agents.mrv.water.industrial_water import (
    IndustrialWaterMRVAgent,
    IndustrialWaterInput,
    IndustrialWaterOutput,
    IndustrialWaterRecord,
)

__all__ = [
    # Water Supply (GL-MRV-WAT-001)
    "WaterSupplyMRVAgent",
    "WaterSupplyInput",
    "WaterSupplyOutput",
    "WaterTreatmentRecord",
    "DistributionRecord",
    # Wastewater (GL-MRV-WAT-002)
    "WastewaterMRVAgent",
    "WastewaterInput",
    "WastewaterOutput",
    "WastewaterTreatmentRecord",
    # Desalination (GL-MRV-WAT-003)
    "DesalinationMRVAgent",
    "DesalinationInput",
    "DesalinationOutput",
    "DesalinationPlantRecord",
    # Irrigation (GL-MRV-WAT-004)
    "IrrigationMRVAgent",
    "IrrigationInput",
    "IrrigationOutput",
    "IrrigationSystemRecord",
    # Industrial Water (GL-MRV-WAT-005)
    "IndustrialWaterMRVAgent",
    "IndustrialWaterInput",
    "IndustrialWaterOutput",
    "IndustrialWaterRecord",
]
