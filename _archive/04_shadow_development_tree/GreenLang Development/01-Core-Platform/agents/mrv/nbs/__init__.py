# -*- coding: utf-8 -*-
"""
GreenLang MRV NBS (Nature-Based Solutions) Agents
=================================================

This package contains MRV agents specialized for Nature-Based Solutions
and Land Use sector carbon accounting. These agents follow the CRITICAL PATH
pattern with zero-hallucination guarantee for regulatory compliance.

Agents:
    GL-MRV-NBS-001: Forest Carbon MRV - Forest carbon sequestration measurement
    GL-MRV-NBS-002: Soil Carbon MRV - Soil organic carbon measurement
    GL-MRV-NBS-003: Wetland Carbon MRV - Wetland/peatland carbon measurement
    GL-MRV-NBS-004: Blue Carbon MRV - Coastal/marine carbon measurement
    GL-MRV-NBS-005: Agroforestry MRV - Agroforestry systems measurement
    GL-MRV-NBS-006: Land Use Change MRV - LUC emissions/removals measurement
    GL-MRV-NBS-007: Biodiversity Co-benefits MRV - Biodiversity metrics measurement

All agents provide:
    - IPCC-compliant calculation methodologies
    - Full audit trail with SHA-256 provenance hashing
    - Support for multiple carbon pools (aboveground, belowground, soil, litter, deadwood)
    - Uncertainty quantification following IPCC guidelines
    - GHG Protocol Land Sector and Removals Guidance compliance

Author: GreenLang Team
Version: 1.0.0
"""

from greenlang.agents.mrv.nbs.forest_carbon import (
    ForestCarbonMRVAgent,
    ForestCarbonInput,
    ForestCarbonOutput,
    ForestType,
    CarbonPool,
    ForestMeasurement,
    BiomassEstimate,
)

from greenlang.agents.mrv.nbs.soil_carbon import (
    SoilCarbonMRVAgent,
    SoilCarbonInput,
    SoilCarbonOutput,
    SoilType,
    LandManagementPractice,
    SoilSample,
    SOCEstimate,
)

from greenlang.agents.mrv.nbs.wetland_carbon import (
    WetlandCarbonMRVAgent,
    WetlandCarbonInput,
    WetlandCarbonOutput,
    WetlandType,
    PeatlandCondition,
    WetlandMeasurement,
    WetlandCarbonEstimate,
)

from greenlang.agents.mrv.nbs.blue_carbon import (
    BlueCarbonMRVAgent,
    BlueCarbonInput,
    BlueCarbonOutput,
    BlueEcosystemType,
    CoastalZone,
    BlueCarbonMeasurement,
    BlueCarbonEstimate,
)

from greenlang.agents.mrv.nbs.agroforestry import (
    AgroforestryMRVAgent,
    AgroforestryInput,
    AgroforestryOutput,
    AgroforestrySystem,
    TreeSpecies,
    AgroforestryMeasurement,
    AgroforestryEstimate,
)

from greenlang.agents.mrv.nbs.land_use_change import (
    LandUseChangeMRVAgent,
    LandUseChangeInput,
    LandUseChangeOutput,
    LandUseCategory,
    TransitionType,
    LandUseTransition,
    LUCEmissionsEstimate,
)

from greenlang.agents.mrv.nbs.biodiversity_cobenefits import (
    BiodiversityCobenefitsMRVAgent,
    BiodiversityInput,
    BiodiversityOutput,
    BiodiversityIndicator,
    EcosystemService,
    BiodiversityAssessment,
    CobenefitScore,
)

__all__ = [
    # Forest Carbon MRV (GL-MRV-NBS-001)
    "ForestCarbonMRVAgent",
    "ForestCarbonInput",
    "ForestCarbonOutput",
    "ForestType",
    "CarbonPool",
    "ForestMeasurement",
    "BiomassEstimate",
    # Soil Carbon MRV (GL-MRV-NBS-002)
    "SoilCarbonMRVAgent",
    "SoilCarbonInput",
    "SoilCarbonOutput",
    "SoilType",
    "LandManagementPractice",
    "SoilSample",
    "SOCEstimate",
    # Wetland Carbon MRV (GL-MRV-NBS-003)
    "WetlandCarbonMRVAgent",
    "WetlandCarbonInput",
    "WetlandCarbonOutput",
    "WetlandType",
    "PeatlandCondition",
    "WetlandMeasurement",
    "WetlandCarbonEstimate",
    # Blue Carbon MRV (GL-MRV-NBS-004)
    "BlueCarbonMRVAgent",
    "BlueCarbonInput",
    "BlueCarbonOutput",
    "BlueEcosystemType",
    "CoastalZone",
    "BlueCarbonMeasurement",
    "BlueCarbonEstimate",
    # Agroforestry MRV (GL-MRV-NBS-005)
    "AgroforestryMRVAgent",
    "AgroforestryInput",
    "AgroforestryOutput",
    "AgroforestrySystem",
    "TreeSpecies",
    "AgroforestryMeasurement",
    "AgroforestryEstimate",
    # Land Use Change MRV (GL-MRV-NBS-006)
    "LandUseChangeMRVAgent",
    "LandUseChangeInput",
    "LandUseChangeOutput",
    "LandUseCategory",
    "TransitionType",
    "LandUseTransition",
    "LUCEmissionsEstimate",
    # Biodiversity Co-benefits MRV (GL-MRV-NBS-007)
    "BiodiversityCobenefitsMRVAgent",
    "BiodiversityInput",
    "BiodiversityOutput",
    "BiodiversityIndicator",
    "EcosystemService",
    "BiodiversityAssessment",
    "CobenefitScore",
]

__version__ = "1.0.0"
