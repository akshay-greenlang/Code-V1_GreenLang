# -*- coding: utf-8 -*-
"""
GreenLang Agriculture MRV Agents
================================

This package provides MRV (Monitoring, Reporting, Verification) agents
for agriculture sector emissions measurement and reporting.

Agents:
- GL-MRV-AGR-001: CropProductionMRVAgent - Crop production emissions
- GL-MRV-AGR-002: LivestockMRVAgent - Livestock emissions (enteric, manure)
- GL-MRV-AGR-003: FertilizerMRVAgent - Fertilizer application emissions
- GL-MRV-AGR-004: LandUseChangeMRVAgent - Land use change emissions
- GL-MRV-AGR-005: RiceCultivationMRVAgent - Rice paddy methane emissions
- GL-MRV-AGR-006: AgriculturalMachineryMRVAgent - Farm equipment emissions
- GL-MRV-AGR-007: IrrigationMRVAgent - Irrigation energy emissions
- GL-MRV-AGR-008: FoodProcessingMRVAgent - Food processing emissions

All agents follow the CRITICAL PATH pattern with zero-hallucination guarantee.
"""

from greenlang.agents.mrv.agriculture.crop_production import (
    CropProductionMRVAgent,
    CropProductionInput,
    CropProductionOutput,
    CropRecord,
)
from greenlang.agents.mrv.agriculture.livestock import (
    LivestockMRVAgent,
    LivestockInput,
    LivestockOutput,
    LivestockRecord,
)
from greenlang.agents.mrv.agriculture.fertilizer import (
    FertilizerMRVAgent,
    FertilizerInput,
    FertilizerOutput,
    FertilizerRecord,
)
from greenlang.agents.mrv.agriculture.land_use_change import (
    LandUseChangeMRVAgent,
    LandUseChangeInput,
    LandUseChangeOutput,
    LandUseRecord,
)
from greenlang.agents.mrv.agriculture.rice_cultivation import (
    RiceCultivationMRVAgent,
    RiceCultivationInput,
    RiceCultivationOutput,
    RiceFieldRecord,
)
from greenlang.agents.mrv.agriculture.agricultural_machinery import (
    AgriculturalMachineryMRVAgent,
    AgriculturalMachineryInput,
    AgriculturalMachineryOutput,
    MachineryRecord,
)
from greenlang.agents.mrv.agriculture.irrigation import (
    IrrigationMRVAgent,
    IrrigationInput,
    IrrigationOutput,
    IrrigationRecord,
)
from greenlang.agents.mrv.agriculture.food_processing import (
    FoodProcessingMRVAgent,
    FoodProcessingInput,
    FoodProcessingOutput,
    ProcessingRecord,
)

__all__ = [
    # Crop Production (GL-MRV-AGR-001)
    "CropProductionMRVAgent",
    "CropProductionInput",
    "CropProductionOutput",
    "CropRecord",
    # Livestock (GL-MRV-AGR-002)
    "LivestockMRVAgent",
    "LivestockInput",
    "LivestockOutput",
    "LivestockRecord",
    # Fertilizer (GL-MRV-AGR-003)
    "FertilizerMRVAgent",
    "FertilizerInput",
    "FertilizerOutput",
    "FertilizerRecord",
    # Land Use Change (GL-MRV-AGR-004)
    "LandUseChangeMRVAgent",
    "LandUseChangeInput",
    "LandUseChangeOutput",
    "LandUseRecord",
    # Rice Cultivation (GL-MRV-AGR-005)
    "RiceCultivationMRVAgent",
    "RiceCultivationInput",
    "RiceCultivationOutput",
    "RiceFieldRecord",
    # Agricultural Machinery (GL-MRV-AGR-006)
    "AgriculturalMachineryMRVAgent",
    "AgriculturalMachineryInput",
    "AgriculturalMachineryOutput",
    "MachineryRecord",
    # Irrigation (GL-MRV-AGR-007)
    "IrrigationMRVAgent",
    "IrrigationInput",
    "IrrigationOutput",
    "IrrigationRecord",
    # Food Processing (GL-MRV-AGR-008)
    "FoodProcessingMRVAgent",
    "FoodProcessingInput",
    "FoodProcessingOutput",
    "ProcessingRecord",
]
