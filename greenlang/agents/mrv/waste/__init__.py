# -*- coding: utf-8 -*-
"""
GreenLang Waste MRV Agents
==========================

This package provides MRV (Monitoring, Reporting, Verification) agents
for waste sector emissions measurement and reporting.

Agents:
    GL-MRV-WST-001: LandfillMRVAgent - Landfill methane emissions
    GL-MRV-WST-002: IncinerationMRVAgent - Waste-to-energy emissions
    GL-MRV-WST-003: RecyclingMRVAgent - Recycling emissions and benefits
    GL-MRV-WST-004: CompostingMRVAgent - Organic waste composting
    GL-MRV-WST-005: HazardousWasteMRVAgent - Hazardous waste treatment
    GL-MRV-WST-006: PlasticWasteMRVAgent - Plastic waste tracking

All agents follow the CRITICAL PATH pattern with zero-hallucination guarantee.

Reference Standards:
    - IPCC 2006/2019 Guidelines for National GHG Inventories (Waste Volume)
    - GHG Protocol Corporate Standard and Scope 3 Standard
    - EPA WARM Model (Waste Reduction Model)
    - EU ETS Monitoring and Reporting Regulation
    - Basel Convention on Hazardous Wastes

Example:
    >>> from greenlang.agents.mrv.waste import LandfillMRVAgent, LandfillInput
    >>> agent = LandfillMRVAgent()
    >>> input_data = LandfillInput(
    ...     organization_id="ORG001",
    ...     reporting_year=2024,
    ...     current_year_waste_tonnes=Decimal("10000"),
    ...     has_gas_collection=True,
    ... )
    >>> result = agent.calculate(input_data)
    >>> print(f"Emissions: {result.total_emissions_mt_co2e} MT CO2e")
"""

from greenlang.agents.mrv.waste.base import (
    # Base classes
    BaseWasteMRVAgent,
    WasteMRVInput,
    WasteMRVOutput,
    # Enums
    WasteType,
    TreatmentMethod,
    LandfillType,
    EmissionScope,
    VerificationStatus,
    DataQualityTier,
    CalculationMethod,
    # Models
    EmissionFactor,
    CalculationStep,
    # Constants
    GWP_AR6_100,
    IPCC_DOC_VALUES,
    IPCC_MCF_VALUES,
    IPCC_L0_VALUES,
)

from greenlang.agents.mrv.waste.landfill_mrv import (
    LandfillMRVAgent,
    LandfillInput,
    LandfillOutput,
    WasteDeposit,
)

from greenlang.agents.mrv.waste.incineration_mrv import (
    IncinerationMRVAgent,
    IncinerationInput,
    IncinerationOutput,
    WasteComponent,
)

from greenlang.agents.mrv.waste.recycling_mrv import (
    RecyclingMRVAgent,
    RecyclingInput,
    RecyclingOutput,
    RecycledMaterial,
    RecyclingType,
)

from greenlang.agents.mrv.waste.composting_mrv import (
    CompostingMRVAgent,
    CompostingInput,
    CompostingOutput,
    CompostingType,
    FeedstockType,
    OrganicFeedstock,
)

from greenlang.agents.mrv.waste.hazardous_waste_mrv import (
    HazardousWasteMRVAgent,
    HazardousWasteInput,
    HazardousWasteOutput,
    HazardousWasteCategory,
    HazardousTreatmentMethod,
    HazardousWasteRecord,
)

from greenlang.agents.mrv.waste.plastic_waste_mrv import (
    PlasticWasteMRVAgent,
    PlasticWasteInput,
    PlasticWasteOutput,
    PolymerType,
    PlasticDisposalPath,
    PlasticWasteRecord,
)

__all__ = [
    # Base classes
    "BaseWasteMRVAgent",
    "WasteMRVInput",
    "WasteMRVOutput",
    # Common enums
    "WasteType",
    "TreatmentMethod",
    "LandfillType",
    "EmissionScope",
    "VerificationStatus",
    "DataQualityTier",
    "CalculationMethod",
    # Common models
    "EmissionFactor",
    "CalculationStep",
    # Constants
    "GWP_AR6_100",
    "IPCC_DOC_VALUES",
    "IPCC_MCF_VALUES",
    "IPCC_L0_VALUES",
    # Landfill (GL-MRV-WST-001)
    "LandfillMRVAgent",
    "LandfillInput",
    "LandfillOutput",
    "WasteDeposit",
    # Incineration (GL-MRV-WST-002)
    "IncinerationMRVAgent",
    "IncinerationInput",
    "IncinerationOutput",
    "WasteComponent",
    # Recycling (GL-MRV-WST-003)
    "RecyclingMRVAgent",
    "RecyclingInput",
    "RecyclingOutput",
    "RecycledMaterial",
    "RecyclingType",
    # Composting (GL-MRV-WST-004)
    "CompostingMRVAgent",
    "CompostingInput",
    "CompostingOutput",
    "CompostingType",
    "FeedstockType",
    "OrganicFeedstock",
    # Hazardous Waste (GL-MRV-WST-005)
    "HazardousWasteMRVAgent",
    "HazardousWasteInput",
    "HazardousWasteOutput",
    "HazardousWasteCategory",
    "HazardousTreatmentMethod",
    "HazardousWasteRecord",
    # Plastic Waste (GL-MRV-WST-006)
    "PlasticWasteMRVAgent",
    "PlasticWasteInput",
    "PlasticWasteOutput",
    "PolymerType",
    "PlasticDisposalPath",
    "PlasticWasteRecord",
]
