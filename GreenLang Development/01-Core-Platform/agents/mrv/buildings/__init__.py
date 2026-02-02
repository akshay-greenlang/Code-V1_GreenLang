# -*- coding: utf-8 -*-
"""
GreenLang Buildings Sector MRV Agents
======================================

MRV (Measurement, Reporting, Verification) agents for the buildings sector.
These agents calculate and track emissions from commercial, residential,
and industrial buildings with full audit trail support.

Agents:
    GL-MRV-BLD-001: Commercial Buildings MRV - Office/retail emissions
    GL-MRV-BLD-002: Residential Buildings MRV - Housing emissions
    GL-MRV-BLD-003: Industrial Buildings MRV - Warehouse/factory buildings
    GL-MRV-BLD-004: HVAC Systems MRV - Heating/cooling/refrigerant emissions
    GL-MRV-BLD-005: Lighting Systems MRV - Lighting energy and emissions
    GL-MRV-BLD-006: Building Materials MRV - Embodied carbon
    GL-MRV-BLD-007: Building Operations MRV - Whole-building operations
    GL-MRV-BLD-008: Smart Building MRV - IoT-enabled buildings

Design Principles:
    - Zero-hallucination: All calculations are deterministic
    - Standards-compliant: GHG Protocol, ASHRAE, Energy Star
    - Auditable: SHA-256 provenance tracking
    - Extensible: Common base class for sector-specific implementations
"""

from greenlang.agents.mrv.buildings.base import (
    # Base classes
    BuildingMRVBaseAgent,
    BuildingMRVInput,
    BuildingMRVOutput,
    # Data models
    BuildingMetadata,
    EnergyConsumption,
    EmissionFactor,
    CalculationStep,
    EnergyUseIntensity,
    CarbonIntensity,
    # Enums
    BuildingType,
    ClimateZone,
    EnergySource,
    EndUseCategory,
    EmissionScope,
    VerificationStatus,
    DataQuality,
    CertificationStandard,
    # Constants
    NATURAL_GAS_EF_KGCO2E_PER_THERM,
    FUEL_OIL_EF_KGCO2E_PER_GALLON,
    PROPANE_EF_KGCO2E_PER_GALLON,
    GRID_EF_BY_REGION_KGCO2E_PER_KWH,
    SOURCE_TO_SITE_RATIO,
    BENCHMARK_EUI_BY_TYPE,
)

from greenlang.agents.mrv.buildings.commercial_buildings_mrv import (
    CommercialBuildingsMRVAgent,
    CommercialBuildingInput,
    CommercialBuildingOutput,
)

from greenlang.agents.mrv.buildings.residential_buildings_mrv import (
    ResidentialBuildingsMRVAgent,
    ResidentialBuildingInput,
    ResidentialBuildingOutput,
)

from greenlang.agents.mrv.buildings.industrial_buildings_mrv import (
    IndustrialBuildingsMRVAgent,
    IndustrialBuildingInput,
    IndustrialBuildingOutput,
)

from greenlang.agents.mrv.buildings.hvac_systems_mrv import (
    HVACSystemsMRVAgent,
    HVACSystemsInput,
    HVACSystemsOutput,
    HVACEquipment,
    HVACSystemType,
    RefrigerantType,
    REFRIGERANT_GWP,
)

from greenlang.agents.mrv.buildings.lighting_systems_mrv import (
    LightingSystemsMRVAgent,
    LightingSystemsInput,
    LightingSystemsOutput,
    LightingFixture,
    LightingType,
    LightingZone,
)

from greenlang.agents.mrv.buildings.building_materials_mrv import (
    BuildingMaterialsMRVAgent,
    BuildingMaterialsInput,
    BuildingMaterialsOutput,
    MaterialQuantity,
    MaterialCategory,
    LifecycleStage,
    MATERIAL_EF_KGCO2E_PER_KG,
)

from greenlang.agents.mrv.buildings.building_operations_mrv import (
    BuildingOperationsMRVAgent,
    BuildingOperationsInput,
    BuildingOperationsOutput,
    YearlyEmissions,
)

from greenlang.agents.mrv.buildings.smart_building_mrv import (
    SmartBuildingMRVAgent,
    SmartBuildingInput,
    SmartBuildingOutput,
    SensorReading,
    TimeSeriesData,
    AnomalyRecord,
    ZoneEmissions,
    SensorType,
    DataGranularity,
    AnomalyType,
)

__all__ = [
    # Base classes
    "BuildingMRVBaseAgent",
    "BuildingMRVInput",
    "BuildingMRVOutput",
    # Data models
    "BuildingMetadata",
    "EnergyConsumption",
    "EmissionFactor",
    "CalculationStep",
    "EnergyUseIntensity",
    "CarbonIntensity",
    # Enums
    "BuildingType",
    "ClimateZone",
    "EnergySource",
    "EndUseCategory",
    "EmissionScope",
    "VerificationStatus",
    "DataQuality",
    "CertificationStandard",
    # Constants
    "NATURAL_GAS_EF_KGCO2E_PER_THERM",
    "FUEL_OIL_EF_KGCO2E_PER_GALLON",
    "PROPANE_EF_KGCO2E_PER_GALLON",
    "GRID_EF_BY_REGION_KGCO2E_PER_KWH",
    "SOURCE_TO_SITE_RATIO",
    "BENCHMARK_EUI_BY_TYPE",
    # GL-MRV-BLD-001: Commercial Buildings
    "CommercialBuildingsMRVAgent",
    "CommercialBuildingInput",
    "CommercialBuildingOutput",
    # GL-MRV-BLD-002: Residential Buildings
    "ResidentialBuildingsMRVAgent",
    "ResidentialBuildingInput",
    "ResidentialBuildingOutput",
    # GL-MRV-BLD-003: Industrial Buildings
    "IndustrialBuildingsMRVAgent",
    "IndustrialBuildingInput",
    "IndustrialBuildingOutput",
    # GL-MRV-BLD-004: HVAC Systems
    "HVACSystemsMRVAgent",
    "HVACSystemsInput",
    "HVACSystemsOutput",
    "HVACEquipment",
    "HVACSystemType",
    "RefrigerantType",
    "REFRIGERANT_GWP",
    # GL-MRV-BLD-005: Lighting Systems
    "LightingSystemsMRVAgent",
    "LightingSystemsInput",
    "LightingSystemsOutput",
    "LightingFixture",
    "LightingType",
    "LightingZone",
    # GL-MRV-BLD-006: Building Materials
    "BuildingMaterialsMRVAgent",
    "BuildingMaterialsInput",
    "BuildingMaterialsOutput",
    "MaterialQuantity",
    "MaterialCategory",
    "LifecycleStage",
    "MATERIAL_EF_KGCO2E_PER_KG",
    # GL-MRV-BLD-007: Building Operations
    "BuildingOperationsMRVAgent",
    "BuildingOperationsInput",
    "BuildingOperationsOutput",
    "YearlyEmissions",
    # GL-MRV-BLD-008: Smart Building
    "SmartBuildingMRVAgent",
    "SmartBuildingInput",
    "SmartBuildingOutput",
    "SensorReading",
    "TimeSeriesData",
    "AnomalyRecord",
    "ZoneEmissions",
    "SensorType",
    "DataGranularity",
    "AnomalyType",
]
