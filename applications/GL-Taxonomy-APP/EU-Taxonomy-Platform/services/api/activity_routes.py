"""
GL-Taxonomy-APP Activity Catalog API

Provides browsing, searching, and lookup capabilities for the EU Taxonomy
activity catalog.  Supports all 13 sectors defined in the Climate Delegated
Act (CDA) and Environmental Delegated Act (EDA), with cross-referencing to
NACE codes, environmental objectives, and technical screening criteria.

EU Taxonomy Sectors (13):
    1. Forestry                           8. Information & Communication
    2. Environmental Protection           9. Professional/Scientific/Technical
    3. Manufacturing                     10. Financial & Insurance
    4. Energy                            11. Education
    5. Water Supply & Waste              12. Human Health & Social Work
    6. Transport                         13. Arts, Entertainment & Recreation
    7. Construction & Real Estate

Environmental Objectives (6):
    1. Climate Change Mitigation (CCM)
    2. Climate Change Adaptation (CCA)
    3. Sustainable Use of Water (WTR)
    4. Transition to Circular Economy (CE)
    5. Pollution Prevention & Control (PPC)
    6. Protection of Biodiversity (BIO)
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

router = APIRouter(prefix="/api/v1/taxonomy/activities", tags=["Activity Catalog"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaxonomySector(str, Enum):
    """EU Taxonomy sector classification."""
    FORESTRY = "forestry"
    ENVIRONMENTAL_PROTECTION = "environmental_protection"
    MANUFACTURING = "manufacturing"
    ENERGY = "energy"
    WATER_SUPPLY_WASTE = "water_supply_waste"
    TRANSPORT = "transport"
    CONSTRUCTION_REAL_ESTATE = "construction_real_estate"
    ICT = "information_communication"
    PROFESSIONAL_SCIENTIFIC = "professional_scientific"
    FINANCIAL_INSURANCE = "financial_insurance"
    EDUCATION = "education"
    HEALTH_SOCIAL = "health_social"
    ARTS_ENTERTAINMENT = "arts_entertainment"


class EnvironmentalObjective(str, Enum):
    """EU Taxonomy environmental objectives."""
    CCM = "climate_change_mitigation"
    CCA = "climate_change_adaptation"
    WTR = "water"
    CE = "circular_economy"
    PPC = "pollution_prevention"
    BIO = "biodiversity"


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ActivityResponse(BaseModel):
    """Taxonomy-eligible activity."""
    activity_code: str
    activity_name: str
    sector: str
    nace_codes: List[str]
    objectives: List[str]
    delegated_act: str
    description: str
    tsc_available: bool
    dnsh_criteria_count: int
    is_transitional: bool
    is_enabling: bool


class ActivityDetailResponse(BaseModel):
    """Detailed activity with TSC summaries."""
    activity_code: str
    activity_name: str
    sector: str
    nace_codes: List[str]
    objectives: List[str]
    delegated_act: str
    description: str
    tsc_summaries: Dict[str, str]
    dnsh_summaries: Dict[str, str]
    is_transitional: bool
    is_enabling: bool
    applicable_from: str
    last_amended: Optional[str]
    notes: Optional[str]


class ActivityStatisticsResponse(BaseModel):
    """Activity catalog statistics."""
    total_activities: int
    by_sector: Dict[str, int]
    by_objective: Dict[str, int]
    transitional_count: int
    enabling_count: int
    ccm_activities: int
    cca_activities: int
    environmental_da_activities: int
    generated_at: datetime


class SectorSummaryResponse(BaseModel):
    """Sector summary with activity count."""
    sector_id: str
    sector_name: str
    activity_count: int
    nace_code_range: str
    objectives_covered: List[str]


class SearchResultsResponse(BaseModel):
    """Activity search results."""
    query: str
    total_results: int
    activities: List[ActivityResponse]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Reference Data -- Activity Catalog
# ---------------------------------------------------------------------------

ACTIVITIES = [
    # Forestry
    {"activity_code": "1.1", "activity_name": "Afforestation", "sector": "forestry", "nace_codes": ["A1", "A2"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Establishment of forest through planting, deliberate seeding on land that was not previously forested.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "1.2", "activity_name": "Rehabilitation and restoration of forests", "sector": "forestry", "nace_codes": ["A1", "A2"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Rehabilitation and restoration of forests including reforestation after extreme events.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "1.3", "activity_name": "Forest management", "sector": "forestry", "nace_codes": ["A1", "A2"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Forest management activities to maintain and improve carbon sequestration.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "1.4", "activity_name": "Conservation forestry", "sector": "forestry", "nace_codes": ["A1", "A2"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "In-situ conservation of existing forests for carbon stock maintenance.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    # Energy
    {"activity_code": "4.1", "activity_name": "Electricity generation using solar photovoltaic technology", "sector": "energy", "nace_codes": ["D35.11"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Construction or operation of electricity generation facilities using solar PV technology.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "4.2", "activity_name": "Electricity generation using concentrated solar power", "sector": "energy", "nace_codes": ["D35.11"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Construction or operation of electricity generation using concentrated solar power.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "4.3", "activity_name": "Electricity generation from wind power", "sector": "energy", "nace_codes": ["D35.11"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Construction or operation of electricity generation using wind power.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "4.5", "activity_name": "Electricity generation from hydropower", "sector": "energy", "nace_codes": ["D35.11"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Construction or operation of electricity generation from hydropower.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "4.7", "activity_name": "Electricity generation from renewable non-fossil gaseous and liquid fuels", "sector": "energy", "nace_codes": ["D35.11"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Electricity generation from biogas, bio-liquid, or hydrogen.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "4.13", "activity_name": "Manufacture of biogas and biofuels", "sector": "energy", "nace_codes": ["D35.21"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Manufacture of biogas or biofuels for use in transport or energy generation.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "4.15", "activity_name": "District heating/cooling distribution", "sector": "energy", "nace_codes": ["D35.30"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Construction or operation of distribution systems for district heating/cooling.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": True, "is_enabling": False},
    {"activity_code": "4.25", "activity_name": "Production of heat/cool from geothermal energy", "sector": "energy", "nace_codes": ["D35.30"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Construction or operation of facilities producing heat from geothermal energy.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "4.29", "activity_name": "Electricity generation from fossil gaseous fuels", "sector": "energy", "nace_codes": ["D35.11"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA_COMP", "description": "Electricity generation from natural gas (Complementary CDA, transitional).", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": True, "is_enabling": False},
    # Manufacturing
    {"activity_code": "3.1", "activity_name": "Manufacture of renewable energy technologies", "sector": "manufacturing", "nace_codes": ["C25", "C27", "C28"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Manufacture of equipment for renewable energy generation and storage.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    {"activity_code": "3.2", "activity_name": "Manufacture of equipment for hydrogen production", "sector": "manufacturing", "nace_codes": ["C25", "C27", "C28"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Manufacture of hydrogen-related equipment (electrolysers, fuel cells).", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    {"activity_code": "3.3", "activity_name": "Manufacture of low carbon technologies for transport", "sector": "manufacturing", "nace_codes": ["C29.1"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Manufacture of EVs, rail vehicles, and other low-carbon transport.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    {"activity_code": "3.5", "activity_name": "Manufacture of energy efficiency equipment for buildings", "sector": "manufacturing", "nace_codes": ["C16", "C23", "C25", "C27", "C28"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Manufacture of insulation, windows, heat pumps for energy-efficient buildings.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    {"activity_code": "3.6", "activity_name": "Manufacture of other low carbon technologies", "sector": "manufacturing", "nace_codes": ["C22", "C25", "C26", "C27", "C28"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Manufacture of technologies for substantial GHG reduction in other sectors.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    {"activity_code": "3.9", "activity_name": "Manufacture of iron and steel", "sector": "manufacturing", "nace_codes": ["C24.10", "C24.20", "C24.31", "C24.32"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Manufacture of iron and steel with low GHG intensity.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": True, "is_enabling": False},
    {"activity_code": "3.10", "activity_name": "Manufacture of hydrogen", "sector": "manufacturing", "nace_codes": ["C20.11"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Manufacture of hydrogen via electrolysis or reforming with CCS.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "3.12", "activity_name": "Manufacture of cement", "sector": "manufacturing", "nace_codes": ["C23.51"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Manufacture of cement clinker, cement, or alternative binders.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": True, "is_enabling": False},
    {"activity_code": "3.14", "activity_name": "Manufacture of aluminium", "sector": "manufacturing", "nace_codes": ["C24.42", "C24.43"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Manufacture of primary and secondary aluminium.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": True, "is_enabling": False},
    {"activity_code": "3.17", "activity_name": "Manufacture of plastics in primary form", "sector": "manufacturing", "nace_codes": ["C20.16"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Manufacture of plastics from recycled or bio-based feedstock.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    # Transport
    {"activity_code": "6.1", "activity_name": "Passenger interurban rail transport", "sector": "transport", "nace_codes": ["H49.10"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Purchase, leasing, and operation of passenger rail vehicles.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "6.2", "activity_name": "Freight rail transport", "sector": "transport", "nace_codes": ["H49.20"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Purchase, leasing, and operation of freight rail vehicles.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "6.3", "activity_name": "Urban and suburban transport, road passenger transport", "sector": "transport", "nace_codes": ["H49.31", "H49.39"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Purchase and operation of zero direct emission urban transport.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "6.4", "activity_name": "Operation of personal mobility devices", "sector": "transport", "nace_codes": ["N77.11", "N77.21"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Sale, rental, leasing of personal zero-emission mobility devices.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "6.5", "activity_name": "Transport by motorbikes, passenger cars and light commercial vehicles", "sector": "transport", "nace_codes": ["H49.32", "H49.39", "N77.11"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Purchase and operation of zero direct emission vehicles (until 2025) or below 50 gCO2/km.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": True, "is_enabling": False},
    {"activity_code": "6.6", "activity_name": "Freight transport services by road", "sector": "transport", "nace_codes": ["H49.41", "H53.10", "H53.20"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Purchase and operation of zero direct emission heavy duty freight vehicles.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": True, "is_enabling": False},
    {"activity_code": "6.10", "activity_name": "Sea and coastal freight water transport", "sector": "transport", "nace_codes": ["H50.20"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Purchase and operation of vessels for sea and coastal freight transport.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": True, "is_enabling": False},
    {"activity_code": "6.14", "activity_name": "Infrastructure for rail transport", "sector": "transport", "nace_codes": ["F42.12", "F42.13", "M71.12", "M71.20"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Construction, modernization, operation of rail infrastructure.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    {"activity_code": "6.15", "activity_name": "Infrastructure enabling low-carbon road transport", "sector": "transport", "nace_codes": ["F42.11", "F42.13"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Construction and operation of EV charging, hydrogen refuelling stations.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    # Construction & Real Estate
    {"activity_code": "7.1", "activity_name": "Construction of new buildings", "sector": "construction_real_estate", "nace_codes": ["F41.1", "F41.2"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Construction of new buildings meeting NZEB - 10% energy performance.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "7.2", "activity_name": "Renovation of existing buildings", "sector": "construction_real_estate", "nace_codes": ["F41", "F43"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Major renovation achieving 30% primary energy demand reduction.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "7.3", "activity_name": "Installation, maintenance and repair of energy efficiency equipment", "sector": "construction_real_estate", "nace_codes": ["F42", "F43"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Individual energy efficiency measures (insulation, windows, HVAC, LED).", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    {"activity_code": "7.4", "activity_name": "Installation, maintenance and repair of charging stations", "sector": "construction_real_estate", "nace_codes": ["F42", "F43"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Installation and operation of electric vehicle charging stations in buildings.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    {"activity_code": "7.5", "activity_name": "Installation, maintenance and repair of instruments for measuring, regulation and controlling energy performance", "sector": "construction_real_estate", "nace_codes": ["F42", "F43"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Installation of smart building monitoring and automation systems.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    {"activity_code": "7.6", "activity_name": "Installation, maintenance and repair of renewable energy technologies", "sector": "construction_real_estate", "nace_codes": ["F42", "F43"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Installation of on-site renewable energy technologies.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    {"activity_code": "7.7", "activity_name": "Acquisition and ownership of buildings", "sector": "construction_real_estate", "nace_codes": ["L68"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Acquisition of buildings meeting top 15% energy performance or EPC A.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    # Water Supply & Waste
    {"activity_code": "5.1", "activity_name": "Construction, extension and operation of water collection, treatment and supply systems", "sector": "water_supply_waste", "nace_codes": ["E36.00", "F42.99"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Water collection, treatment, and supply infrastructure.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "5.3", "activity_name": "Construction, extension and operation of waste water collection and treatment", "sector": "water_supply_waste", "nace_codes": ["E37.00", "F42.99"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Waste water collection, treatment, and discharge systems.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    {"activity_code": "5.5", "activity_name": "Collection and transport of non-hazardous waste", "sector": "water_supply_waste", "nace_codes": ["E38.11"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Separately collected non-hazardous waste for recycling/reuse.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": True, "is_enabling": False},
    {"activity_code": "5.9", "activity_name": "Material recovery from non-hazardous waste", "sector": "water_supply_waste", "nace_codes": ["E38.32"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Material recovery from non-hazardous waste into secondary raw materials.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": False},
    # ICT
    {"activity_code": "8.1", "activity_name": "Data processing, hosting and related activities", "sector": "information_communication", "nace_codes": ["J63.11"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Storage, processing, hosting using energy-efficient best practices.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    {"activity_code": "8.2", "activity_name": "Data-driven solutions for GHG emissions reductions", "sector": "information_communication", "nace_codes": ["J61", "J62", "J63.11"], "objectives": ["climate_change_mitigation"], "delegated_act": "CDA", "description": "Software solutions enabling GHG emission reductions in other activities.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    # Financial & Insurance
    {"activity_code": "10.1", "activity_name": "Non-life insurance: underwriting of climate-related perils", "sector": "financial_insurance", "nace_codes": ["K65.12"], "objectives": ["climate_change_adaptation"], "delegated_act": "CDA", "description": "Non-life insurance products covering climate-related physical risks.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    {"activity_code": "10.2", "activity_name": "Reinsurance of climate-related perils", "sector": "financial_insurance", "nace_codes": ["K65.20"], "objectives": ["climate_change_adaptation"], "delegated_act": "CDA", "description": "Reinsurance covering climate-related physical risk perils.", "tsc_available": True, "dnsh_criteria_count": 5, "is_transitional": False, "is_enabling": True},
    # Environmental Protection (EDA)
    {"activity_code": "2.1", "activity_name": "Restoration of wetlands", "sector": "environmental_protection", "nace_codes": ["A1", "A2", "E39.00"], "objectives": ["biodiversity", "water", "climate_change_mitigation"], "delegated_act": "EDA", "description": "Rehabilitation and restoration of wetland ecosystems.", "tsc_available": True, "dnsh_criteria_count": 4, "is_transitional": False, "is_enabling": False},
]

SECTOR_NAMES = {
    "forestry": "Forestry",
    "environmental_protection": "Environmental Protection & Restoration",
    "manufacturing": "Manufacturing",
    "energy": "Energy",
    "water_supply_waste": "Water Supply, Sewerage, Waste Management",
    "transport": "Transport",
    "construction_real_estate": "Construction & Real Estate",
    "information_communication": "Information & Communication",
    "professional_scientific": "Professional, Scientific & Technical",
    "financial_insurance": "Financial & Insurance",
    "education": "Education",
    "health_social": "Human Health & Social Work",
    "arts_entertainment": "Arts, Entertainment & Recreation",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/",
    response_model=List[ActivityResponse],
    summary="List or search activities",
    description=(
        "Retrieve taxonomy-eligible activities. Supports filtering by "
        "text query, sector, environmental objective, NACE code, and "
        "delegated act version."
    ),
)
async def list_activities(
    query: Optional[str] = Query(None, description="Text search on name/description"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    objective: Optional[str] = Query(None, description="Filter by environmental objective"),
    delegated_act: Optional[str] = Query(None, description="CDA, CDA_COMP, or EDA"),
    is_transitional: Optional[bool] = Query(None, description="Filter transitional activities"),
    is_enabling: Optional[bool] = Query(None, description="Filter enabling activities"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> List[ActivityResponse]:
    """List or filter taxonomy activities."""
    results = list(ACTIVITIES)

    if query:
        q_lower = query.lower()
        results = [
            a for a in results
            if q_lower in a["activity_name"].lower() or q_lower in a["description"].lower()
        ]
    if sector:
        results = [a for a in results if a["sector"] == sector]
    if objective:
        results = [a for a in results if objective in a["objectives"]]
    if delegated_act:
        results = [a for a in results if a["delegated_act"] == delegated_act]
    if is_transitional is not None:
        results = [a for a in results if a["is_transitional"] == is_transitional]
    if is_enabling is not None:
        results = [a for a in results if a["is_enabling"] == is_enabling]

    paginated = results[offset:offset + limit]
    return [ActivityResponse(**a) for a in paginated]


@router.get(
    "/statistics",
    response_model=ActivityStatisticsResponse,
    summary="Activity catalog statistics",
    description="Get aggregate statistics for the activity catalog.",
)
async def get_statistics() -> ActivityStatisticsResponse:
    """Get activity catalog statistics."""
    by_sector: Dict[str, int] = {}
    by_objective: Dict[str, int] = {}
    transitional = 0
    enabling = 0
    ccm = 0
    cca = 0
    eda = 0

    for a in ACTIVITIES:
        sector_name = SECTOR_NAMES.get(a["sector"], a["sector"])
        by_sector[sector_name] = by_sector.get(sector_name, 0) + 1
        for obj in a["objectives"]:
            by_objective[obj] = by_objective.get(obj, 0) + 1
        if a["is_transitional"]:
            transitional += 1
        if a["is_enabling"]:
            enabling += 1
        if "climate_change_mitigation" in a["objectives"]:
            ccm += 1
        if "climate_change_adaptation" in a["objectives"]:
            cca += 1
        if a["delegated_act"] == "EDA":
            eda += 1

    return ActivityStatisticsResponse(
        total_activities=len(ACTIVITIES),
        by_sector=by_sector,
        by_objective=by_objective,
        transitional_count=transitional,
        enabling_count=enabling,
        ccm_activities=ccm,
        cca_activities=cca,
        environmental_da_activities=eda,
        generated_at=_now(),
    )


@router.get(
    "/sectors",
    response_model=List[SectorSummaryResponse],
    summary="List all 13 sectors with counts",
    description="List all EU Taxonomy sectors with activity counts and coverage.",
)
async def list_sectors() -> List[SectorSummaryResponse]:
    """List all 13 sectors with activity counts."""
    sector_data: Dict[str, Dict[str, Any]] = {}

    for sector_id, sector_name in SECTOR_NAMES.items():
        sector_data[sector_id] = {
            "sector_id": sector_id,
            "sector_name": sector_name,
            "activity_count": 0,
            "nace_codes": set(),
            "objectives": set(),
        }

    for a in ACTIVITIES:
        sid = a["sector"]
        if sid in sector_data:
            sector_data[sid]["activity_count"] += 1
            for nc in a["nace_codes"]:
                sector_data[sid]["nace_codes"].add(nc)
            for obj in a["objectives"]:
                sector_data[sid]["objectives"].add(obj)

    results = []
    for sd in sector_data.values():
        nace_sorted = sorted(sd["nace_codes"])
        nace_range = f"{nace_sorted[0]}-{nace_sorted[-1]}" if nace_sorted else "N/A"
        results.append(SectorSummaryResponse(
            sector_id=sd["sector_id"],
            sector_name=sd["sector_name"],
            activity_count=sd["activity_count"],
            nace_code_range=nace_range,
            objectives_covered=sorted(sd["objectives"]),
        ))

    results.sort(key=lambda s: s.sector_id)
    return results


@router.get(
    "/search",
    response_model=SearchResultsResponse,
    summary="Text search activities",
    description="Full-text search across activity names, descriptions, and NACE codes.",
)
async def search_activities(
    q: str = Query(..., min_length=2, max_length=200, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
) -> SearchResultsResponse:
    """Full-text search activities."""
    q_lower = q.lower()
    matches = [
        a for a in ACTIVITIES
        if (
            q_lower in a["activity_name"].lower()
            or q_lower in a["description"].lower()
            or any(q_lower in nc.lower() for nc in a["nace_codes"])
        )
    ]

    return SearchResultsResponse(
        query=q,
        total_results=len(matches),
        activities=[ActivityResponse(**a) for a in matches[:limit]],
        generated_at=_now(),
    )


@router.get(
    "/sector/{sector}",
    response_model=List[ActivityResponse],
    summary="Get activities by sector",
    description="Retrieve all taxonomy activities within a specific sector.",
)
async def get_by_sector(
    sector: str,
    limit: int = Query(50, ge=1, le=200),
) -> List[ActivityResponse]:
    """Get activities by sector."""
    results = [a for a in ACTIVITIES if a["sector"] == sector]
    if not results:
        valid = list(SECTOR_NAMES.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No activities found for sector '{sector}'. Valid sectors: {valid}",
        )
    return [ActivityResponse(**a) for a in results[:limit]]


@router.get(
    "/objective/{objective}",
    response_model=List[ActivityResponse],
    summary="Get activities by environmental objective",
    description="Retrieve all taxonomy activities contributing to a specific environmental objective.",
)
async def get_by_objective(
    objective: str,
    limit: int = Query(50, ge=1, le=200),
) -> List[ActivityResponse]:
    """Get activities by environmental objective."""
    results = [a for a in ACTIVITIES if objective in a["objectives"]]
    if not results:
        valid = [e.value for e in EnvironmentalObjective]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No activities found for objective '{objective}'. Valid objectives: {valid}",
        )
    return [ActivityResponse(**a) for a in results[:limit]]


@router.get(
    "/nace/{nace_code}",
    response_model=List[ActivityResponse],
    summary="Lookup by NACE code",
    description="Find taxonomy activities associated with a specific NACE code.",
)
async def get_by_nace(nace_code: str) -> List[ActivityResponse]:
    """Lookup activities by NACE code."""
    results = [
        a for a in ACTIVITIES
        if nace_code in a["nace_codes"] or any(nace_code in nc for nc in a["nace_codes"])
    ]
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No activities found for NACE code '{nace_code}'.",
        )
    return [ActivityResponse(**a) for a in results]


@router.get(
    "/{activity_code}",
    response_model=ActivityDetailResponse,
    summary="Get activity details",
    description="Retrieve detailed information for a specific activity including TSC and DNSH summaries.",
)
async def get_activity_detail(activity_code: str) -> ActivityDetailResponse:
    """Get detailed activity by code."""
    activity = next((a for a in ACTIVITIES if a["activity_code"] == activity_code), None)
    if not activity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Activity '{activity_code}' not found.",
        )

    # Build TSC summaries per objective
    tsc_summaries: Dict[str, str] = {}
    for obj in activity["objectives"]:
        if obj == "climate_change_mitigation":
            tsc_summaries[obj] = f"Activity {activity_code} must demonstrate substantial contribution to CCM per CDA Annex I criteria."
        elif obj == "climate_change_adaptation":
            tsc_summaries[obj] = f"Activity {activity_code} must implement adaptation solutions reducing material physical climate risks."
        else:
            tsc_summaries[obj] = f"Activity {activity_code} must meet TSC for {obj} per Environmental Delegated Act."

    # DNSH summaries for non-SC objectives
    all_objectives = ["climate_change_mitigation", "climate_change_adaptation", "water", "circular_economy", "pollution_prevention", "biodiversity"]
    dnsh_summaries: Dict[str, str] = {}
    for obj in all_objectives:
        if obj not in activity["objectives"]:
            dnsh_summaries[obj] = f"DNSH criteria apply for {obj}. See delegated act annex for specific requirements."

    return ActivityDetailResponse(
        activity_code=activity["activity_code"],
        activity_name=activity["activity_name"],
        sector=activity["sector"],
        nace_codes=activity["nace_codes"],
        objectives=activity["objectives"],
        delegated_act=activity["delegated_act"],
        description=activity["description"],
        tsc_summaries=tsc_summaries,
        dnsh_summaries=dnsh_summaries,
        is_transitional=activity["is_transitional"],
        is_enabling=activity["is_enabling"],
        applicable_from="2022-01-01",
        last_amended="2023-06-27",
        notes=None,
    )
