# -*- coding: utf-8 -*-
"""
GreenLang Adaptation NBS (Nature-Based Solutions) Agents
========================================================

This package contains agents for NBS-focused climate adaptation planning.

Agents:
    GL-ADAPT-NBS-001: Forest Fire Risk - Wildfire risk assessment
    GL-ADAPT-NBS-002: Ecosystem Vulnerability - Ecosystem vulnerability assessment
    GL-ADAPT-NBS-003: Species Migration - Climate-driven species migration modeling
    GL-ADAPT-NBS-004: Drought Impact on Forests - Forest drought risk assessment
    GL-ADAPT-NBS-005: Coastal Erosion Protection - Nature-based coastal defense
    GL-ADAPT-NBS-006: Urban Heat Island Mitigation - Urban cooling strategies
    GL-ADAPT-NBS-007: Watershed Protection - Watershed adaptation planning
    GL-ADAPT-NBS-008: Biodiversity Corridors - Wildlife corridor planning
    GL-ADAPT-NBS-009: Climate-Resilient Landscapes - Integrated landscape planning

All agents follow GreenLang's zero-hallucination principles with
deterministic calculations and complete provenance tracking.

Author: GreenLang Team
Version: 1.0.0
"""

from greenlang.agents.adaptation.nbs.forest_fire_risk import (
    ForestFireRiskAgent,
    ForestFireRiskInput,
    ForestFireRiskOutput,
    FireRiskAssessment,
)

from greenlang.agents.adaptation.nbs.ecosystem_vulnerability import (
    EcosystemVulnerabilityAgent,
    EcosystemVulnerabilityInput,
    EcosystemVulnerabilityOutput,
    EcosystemVulnerabilityAssessment,
)

from greenlang.agents.adaptation.nbs.species_migration import (
    SpeciesMigrationAgent,
    SpeciesMigrationInput,
    SpeciesMigrationOutput,
    MigrationProjection,
)

from greenlang.agents.adaptation.nbs.drought_forest_impact import (
    DroughtForestImpactAgent,
    DroughtImpactInput,
    DroughtImpactOutput,
    DroughtImpactAssessment,
)

from greenlang.agents.adaptation.nbs.coastal_erosion_protection import (
    CoastalErosionProtectionAgent,
    CoastalProtectionInput,
    CoastalProtectionOutput,
    CoastalProtectionPlan,
)

from greenlang.agents.adaptation.nbs.urban_heat_mitigation import (
    UrbanHeatMitigationAgent,
    UrbanHeatInput,
    UrbanHeatOutput,
    HeatMitigationPlan,
)

from greenlang.agents.adaptation.nbs.watershed_protection import (
    WatershedProtectionAgent,
    WatershedInput,
    WatershedOutput,
    WatershedProtectionPlan,
)

from greenlang.agents.adaptation.nbs.biodiversity_corridors import (
    BiodiversityCorridorsAgent,
    CorridorInput,
    CorridorOutput,
    CorridorPlan,
)

from greenlang.agents.adaptation.nbs.climate_resilient_landscapes import (
    ClimateResilientLandscapesAgent,
    LandscapeInput,
    LandscapeOutput,
    LandscapePlan,
)

__all__ = [
    # Forest Fire Risk (GL-ADAPT-NBS-001)
    "ForestFireRiskAgent",
    "ForestFireRiskInput",
    "ForestFireRiskOutput",
    "FireRiskAssessment",
    # Ecosystem Vulnerability (GL-ADAPT-NBS-002)
    "EcosystemVulnerabilityAgent",
    "EcosystemVulnerabilityInput",
    "EcosystemVulnerabilityOutput",
    "EcosystemVulnerabilityAssessment",
    # Species Migration (GL-ADAPT-NBS-003)
    "SpeciesMigrationAgent",
    "SpeciesMigrationInput",
    "SpeciesMigrationOutput",
    "MigrationProjection",
    # Drought Impact (GL-ADAPT-NBS-004)
    "DroughtForestImpactAgent",
    "DroughtImpactInput",
    "DroughtImpactOutput",
    "DroughtImpactAssessment",
    # Coastal Erosion (GL-ADAPT-NBS-005)
    "CoastalErosionProtectionAgent",
    "CoastalProtectionInput",
    "CoastalProtectionOutput",
    "CoastalProtectionPlan",
    # Urban Heat (GL-ADAPT-NBS-006)
    "UrbanHeatMitigationAgent",
    "UrbanHeatInput",
    "UrbanHeatOutput",
    "HeatMitigationPlan",
    # Watershed Protection (GL-ADAPT-NBS-007)
    "WatershedProtectionAgent",
    "WatershedInput",
    "WatershedOutput",
    "WatershedProtectionPlan",
    # Biodiversity Corridors (GL-ADAPT-NBS-008)
    "BiodiversityCorridorsAgent",
    "CorridorInput",
    "CorridorOutput",
    "CorridorPlan",
    # Climate-Resilient Landscapes (GL-ADAPT-NBS-009)
    "ClimateResilientLandscapesAgent",
    "LandscapeInput",
    "LandscapeOutput",
    "LandscapePlan",
]

__version__ = "1.0.0"
