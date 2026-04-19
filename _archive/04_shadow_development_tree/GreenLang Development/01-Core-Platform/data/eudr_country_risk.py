"""
EUDR Country Risk Database
==========================
EU Deforestation Regulation (EU) 2023/1115 Country Risk Assessment

This module provides country and region-specific deforestation risk data based on:
- EC Benchmarking System (Article 29)
- FAO Global Forest Resources Assessment
- Global Forest Watch data
- World Resources Institute forest data

Risk Levels per EUDR:
- LOW: Standard due diligence (simplified)
- STANDARD: Standard due diligence (full)
- HIGH: Enhanced due diligence with satellite verification

Deforestation-free baseline: December 31, 2020
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import date


# =============================================================================
# Enums and Constants
# =============================================================================

class RiskLevel(Enum):
    """EUDR risk classification levels per EC benchmarking."""
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


class DueDiligenceLevel(Enum):
    """Required due diligence level based on risk."""
    STANDARD = "standard"           # Low risk countries
    ENHANCED = "enhanced"           # Standard risk countries
    FULL_VERIFICATION = "full_verification"  # High risk countries


class ForestType(Enum):
    """Primary forest types for risk assessment."""
    TROPICAL_RAINFOREST = "tropical_rainforest"
    TROPICAL_DRY = "tropical_dry"
    TEMPERATE = "temperate"
    BOREAL = "boreal"
    MANGROVE = "mangrove"
    PLANTATION = "plantation"
    SAVANNA_WOODLAND = "savanna_woodland"


# EUDR Deforestation-free cutoff date
EUDR_CUTOFF_DATE = date(2020, 12, 31)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ForestData:
    """Forest cover and deforestation data for a country/region."""
    forest_cover_percentage: float  # Current forest cover %
    forest_cover_2020: float       # Forest cover at EUDR cutoff
    annual_deforestation_rate: float  # % per year (negative = loss)
    primary_forest_percentage: float  # % of forest that is primary
    forest_area_km2: float         # Total forest area
    tree_cover_loss_2020_2023: float  # Tree cover loss since cutoff (km2)
    forest_type: ForestType        # Predominant forest type
    data_source: str               # Data source citation
    last_updated: date             # Date of last data update


@dataclass
class CountryRisk:
    """Country-level deforestation risk assessment."""
    country_code: str              # ISO 3166-1 alpha-2
    country_name: str              # Full country name
    risk_level: RiskLevel          # EC benchmarking risk level
    risk_score: float              # Numeric risk score 0-100
    forest_data: ForestData        # Forest statistics
    primary_deforestation_drivers: List[str]  # Main drivers
    commodity_risks: Dict[str, RiskLevel]  # Commodity-specific risks
    regions_of_concern: List[str]  # High-risk sub-national regions
    certifications_recognized: List[str]  # Recognized certification schemes
    governance_score: float        # Forest governance score (0-100)
    enforcement_score: float       # Law enforcement effectiveness (0-100)
    notes: str = ""                # Additional notes


@dataclass
class RegionRisk:
    """Sub-national region risk assessment."""
    region_name: str               # Region/state name
    country_code: str              # Parent country
    risk_level: RiskLevel          # Region-specific risk
    risk_score: float              # Numeric risk score
    forest_cover_percentage: float # Regional forest cover
    deforestation_hotspot: bool    # Is this a deforestation hotspot?
    protected_areas: List[str]     # Names of protected areas
    indigenous_territories: List[str]  # Indigenous territory names
    primary_commodities: List[str] # Main commodities produced
    coordinates_bounds: Tuple[Tuple[float, float], Tuple[float, float]]  # Bounding box


# =============================================================================
# Country Risk Database
# =============================================================================

# Comprehensive country risk database based on:
# - FAO Global Forest Resources Assessment 2020
# - Global Forest Watch 2023 data
# - EC country benchmarking (provisional)
# - World Resources Institute data

COUNTRY_RISK_DATABASE: Dict[str, CountryRisk] = {
    # =========================================================================
    # HIGH RISK COUNTRIES - Enhanced/Full Verification Required
    # =========================================================================

    "BR": CountryRisk(
        country_code="BR",
        country_name="Brazil",
        risk_level=RiskLevel.HIGH,
        risk_score=85,
        forest_data=ForestData(
            forest_cover_percentage=59.4,
            forest_cover_2020=59.9,
            annual_deforestation_rate=-0.5,
            primary_forest_percentage=57.3,
            forest_area_km2=4966196,
            tree_cover_loss_2020_2023=45000,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Cattle ranching expansion",
            "Soy cultivation",
            "Illegal logging",
            "Land speculation",
            "Infrastructure development"
        ],
        commodity_risks={
            "cattle": RiskLevel.HIGH,
            "soya": RiskLevel.HIGH,
            "wood": RiskLevel.HIGH,
            "coffee": RiskLevel.STANDARD,
            "cocoa": RiskLevel.STANDARD,
            "rubber": RiskLevel.STANDARD,
            "palm_oil": RiskLevel.STANDARD
        },
        regions_of_concern=[
            "Amazonia Legal", "Cerrado", "Mato Grosso",
            "Para", "Rondonia", "Maranhao", "Tocantins"
        ],
        certifications_recognized=["FSC", "PEFC", "Rainforest Alliance", "RTRS"],
        governance_score=55,
        enforcement_score=45,
        notes="Amazon and Cerrado biomes highest risk. Moratorium areas require extra scrutiny."
    ),

    "ID": CountryRisk(
        country_code="ID",
        country_name="Indonesia",
        risk_level=RiskLevel.HIGH,
        risk_score=80,
        forest_data=ForestData(
            forest_cover_percentage=49.1,
            forest_cover_2020=50.2,
            annual_deforestation_rate=-0.8,
            primary_forest_percentage=35.7,
            forest_area_km2=921332,
            tree_cover_loss_2020_2023=12500,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Palm oil plantation expansion",
            "Pulp and paper plantations",
            "Mining",
            "Peatland drainage",
            "Smallholder agriculture"
        ],
        commodity_risks={
            "palm_oil": RiskLevel.HIGH,
            "rubber": RiskLevel.HIGH,
            "wood": RiskLevel.HIGH,
            "cocoa": RiskLevel.STANDARD,
            "coffee": RiskLevel.STANDARD,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[
            "Sumatra", "Kalimantan", "Papua", "Sulawesi",
            "Riau", "West Kalimantan", "Central Kalimantan"
        ],
        certifications_recognized=["RSPO", "FSC", "PEFC", "ISPO"],
        governance_score=50,
        enforcement_score=40,
        notes="Peatland areas highest risk. RSPO certification recommended for palm oil."
    ),

    "MY": CountryRisk(
        country_code="MY",
        country_name="Malaysia",
        risk_level=RiskLevel.HIGH,
        risk_score=75,
        forest_data=ForestData(
            forest_cover_percentage=57.7,
            forest_cover_2020=58.2,
            annual_deforestation_rate=-0.3,
            primary_forest_percentage=18.4,
            forest_area_km2=190498,
            tree_cover_loss_2020_2023=4200,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Palm oil plantation expansion",
            "Logging (legal and illegal)",
            "Agricultural conversion",
            "Infrastructure development"
        ],
        commodity_risks={
            "palm_oil": RiskLevel.HIGH,
            "rubber": RiskLevel.STANDARD,
            "wood": RiskLevel.HIGH,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[
            "Sabah", "Sarawak", "Peninsular Malaysia"
        ],
        certifications_recognized=["RSPO", "MSPO", "FSC", "PEFC", "MTCS"],
        governance_score=60,
        enforcement_score=55,
        notes="Sabah and Sarawak require enhanced scrutiny. MSPO is national standard."
    ),

    "CG": CountryRisk(
        country_code="CG",
        country_name="Republic of the Congo",
        risk_level=RiskLevel.HIGH,
        risk_score=78,
        forest_data=ForestData(
            forest_cover_percentage=65.4,
            forest_cover_2020=66.0,
            annual_deforestation_rate=-0.3,
            primary_forest_percentage=55.1,
            forest_area_km2=223764,
            tree_cover_loss_2020_2023=2800,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Industrial logging",
            "Subsistence agriculture",
            "Charcoal production",
            "Mining"
        ],
        commodity_risks={
            "wood": RiskLevel.HIGH,
            "palm_oil": RiskLevel.STANDARD,
            "rubber": RiskLevel.STANDARD,
            "cocoa": RiskLevel.STANDARD,
            "coffee": RiskLevel.STANDARD,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=["Northern regions", "Pool region"],
        certifications_recognized=["FSC", "PEFC", "OLB"],
        governance_score=40,
        enforcement_score=35,
        notes="Congo Basin country. FLEGT VPA in place but implementation varies."
    ),

    "CD": CountryRisk(
        country_code="CD",
        country_name="Democratic Republic of the Congo",
        risk_level=RiskLevel.HIGH,
        risk_score=90,
        forest_data=ForestData(
            forest_cover_percentage=52.6,
            forest_cover_2020=53.5,
            annual_deforestation_rate=-0.8,
            primary_forest_percentage=60.0,
            forest_area_km2=1267000,
            tree_cover_loss_2020_2023=15000,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Subsistence agriculture",
            "Charcoal production",
            "Industrial logging",
            "Mining",
            "Armed conflict"
        ],
        commodity_risks={
            "wood": RiskLevel.HIGH,
            "cocoa": RiskLevel.HIGH,
            "coffee": RiskLevel.HIGH,
            "palm_oil": RiskLevel.HIGH,
            "rubber": RiskLevel.STANDARD,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[
            "Equateur", "Mai-Ndombe", "Tshuapa",
            "Orientale", "North Kivu", "South Kivu"
        ],
        certifications_recognized=["FSC"],
        governance_score=25,
        enforcement_score=20,
        notes="Highest risk country in Africa. Conflict areas require extreme scrutiny."
    ),

    "CI": CountryRisk(
        country_code="CI",
        country_name="Cote d'Ivoire",
        risk_level=RiskLevel.HIGH,
        risk_score=82,
        forest_data=ForestData(
            forest_cover_percentage=8.9,
            forest_cover_2020=10.2,
            annual_deforestation_rate=-3.0,
            primary_forest_percentage=2.0,
            forest_area_km2=28560,
            tree_cover_loss_2020_2023=3500,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Cocoa farming expansion",
            "Illegal farming in protected areas",
            "Logging",
            "Population pressure"
        ],
        commodity_risks={
            "cocoa": RiskLevel.HIGH,
            "rubber": RiskLevel.HIGH,
            "palm_oil": RiskLevel.STANDARD,
            "wood": RiskLevel.HIGH,
            "coffee": RiskLevel.STANDARD,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[
            "Tai National Park surroundings", "Mont Peko",
            "Southwest region", "Western region"
        ],
        certifications_recognized=["Rainforest Alliance", "UTZ", "Fairtrade", "FSC"],
        governance_score=45,
        enforcement_score=35,
        notes="World's largest cocoa producer. Protected area encroachment major issue."
    ),

    "GH": CountryRisk(
        country_code="GH",
        country_name="Ghana",
        risk_level=RiskLevel.STANDARD,
        risk_score=65,
        forest_data=ForestData(
            forest_cover_percentage=36.2,
            forest_cover_2020=37.5,
            annual_deforestation_rate=-2.0,
            primary_forest_percentage=5.8,
            forest_area_km2=86476,
            tree_cover_loss_2020_2023=2100,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Cocoa farming",
            "Mining (galamsey)",
            "Logging",
            "Agricultural expansion"
        ],
        commodity_risks={
            "cocoa": RiskLevel.STANDARD,
            "wood": RiskLevel.STANDARD,
            "palm_oil": RiskLevel.STANDARD,
            "rubber": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[
            "Western Region", "Ashanti Region", "Brong-Ahafo"
        ],
        certifications_recognized=["Rainforest Alliance", "UTZ", "FSC", "FLEGT"],
        governance_score=55,
        enforcement_score=50,
        notes="FLEGT VPA in force. Cocoa & Forests Initiative participant."
    ),

    "PY": CountryRisk(
        country_code="PY",
        country_name="Paraguay",
        risk_level=RiskLevel.HIGH,
        risk_score=75,
        forest_data=ForestData(
            forest_cover_percentage=38.6,
            forest_cover_2020=40.1,
            annual_deforestation_rate=-1.0,
            primary_forest_percentage=15.2,
            forest_area_km2=157044,
            tree_cover_loss_2020_2023=4800,
            forest_type=ForestType.TROPICAL_DRY,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Cattle ranching",
            "Soy cultivation",
            "Land clearing for agriculture"
        ],
        commodity_risks={
            "cattle": RiskLevel.HIGH,
            "soya": RiskLevel.HIGH,
            "wood": RiskLevel.STANDARD,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW
        },
        regions_of_concern=[
            "Chaco region", "San Pedro", "Canindeyu"
        ],
        certifications_recognized=["RTRS", "FSC"],
        governance_score=50,
        enforcement_score=40,
        notes="Gran Chaco deforestation hotspot. Zero Deforestation Law limited effectiveness."
    ),

    "BO": CountryRisk(
        country_code="BO",
        country_name="Bolivia",
        risk_level=RiskLevel.HIGH,
        risk_score=78,
        forest_data=ForestData(
            forest_cover_percentage=50.6,
            forest_cover_2020=51.8,
            annual_deforestation_rate=-0.8,
            primary_forest_percentage=42.1,
            forest_area_km2=554000,
            tree_cover_loss_2020_2023=6200,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Soy cultivation",
            "Cattle ranching",
            "Small-scale agriculture",
            "Fire-based land clearing"
        ],
        commodity_risks={
            "soya": RiskLevel.HIGH,
            "cattle": RiskLevel.HIGH,
            "wood": RiskLevel.STANDARD,
            "coffee": RiskLevel.STANDARD,
            "cocoa": RiskLevel.STANDARD,
            "palm_oil": RiskLevel.LOW,
            "rubber": RiskLevel.LOW
        },
        regions_of_concern=[
            "Santa Cruz", "Beni", "Pando", "Chiquitania"
        ],
        certifications_recognized=["FSC", "Rainforest Alliance"],
        governance_score=40,
        enforcement_score=35,
        notes="Chiquitania fires major concern. Government policies favor agricultural expansion."
    ),

    "PE": CountryRisk(
        country_code="PE",
        country_name="Peru",
        risk_level=RiskLevel.STANDARD,
        risk_score=62,
        forest_data=ForestData(
            forest_cover_percentage=57.8,
            forest_cover_2020=58.5,
            annual_deforestation_rate=-0.3,
            primary_forest_percentage=57.3,
            forest_area_km2=742000,
            tree_cover_loss_2020_2023=3800,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Small-scale agriculture",
            "Palm oil expansion",
            "Coca cultivation",
            "Mining",
            "Road construction"
        ],
        commodity_risks={
            "palm_oil": RiskLevel.STANDARD,
            "coffee": RiskLevel.STANDARD,
            "cocoa": RiskLevel.STANDARD,
            "wood": RiskLevel.STANDARD,
            "cattle": RiskLevel.STANDARD,
            "soya": RiskLevel.LOW,
            "rubber": RiskLevel.LOW
        },
        regions_of_concern=[
            "Ucayali", "San Martin", "Loreto", "Madre de Dios"
        ],
        certifications_recognized=["FSC", "Rainforest Alliance", "Organic"],
        governance_score=55,
        enforcement_score=45,
        notes="Amazon region requires scrutiny. FLEGT VPA not yet in place."
    ),

    "CM": CountryRisk(
        country_code="CM",
        country_name="Cameroon",
        risk_level=RiskLevel.HIGH,
        risk_score=72,
        forest_data=ForestData(
            forest_cover_percentage=41.7,
            forest_cover_2020=42.5,
            annual_deforestation_rate=-0.5,
            primary_forest_percentage=30.2,
            forest_area_km2=198000,
            tree_cover_loss_2020_2023=2400,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Industrial agriculture (palm oil, rubber)",
            "Logging",
            "Subsistence agriculture",
            "Infrastructure"
        ],
        commodity_risks={
            "wood": RiskLevel.HIGH,
            "rubber": RiskLevel.STANDARD,
            "palm_oil": RiskLevel.STANDARD,
            "cocoa": RiskLevel.STANDARD,
            "coffee": RiskLevel.STANDARD,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[
            "South Region", "East Region", "Center Region"
        ],
        certifications_recognized=["FSC", "PEFC", "FLEGT"],
        governance_score=45,
        enforcement_score=40,
        notes="FLEGT VPA in place. Herakles case highlighted agribusiness risks."
    ),

    "NG": CountryRisk(
        country_code="NG",
        country_name="Nigeria",
        risk_level=RiskLevel.HIGH,
        risk_score=76,
        forest_data=ForestData(
            forest_cover_percentage=7.2,
            forest_cover_2020=8.5,
            annual_deforestation_rate=-3.7,
            primary_forest_percentage=1.5,
            forest_area_km2=65510,
            tree_cover_loss_2020_2023=2800,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Agricultural expansion",
            "Logging",
            "Fuelwood collection",
            "Population pressure"
        ],
        commodity_risks={
            "cocoa": RiskLevel.HIGH,
            "palm_oil": RiskLevel.HIGH,
            "rubber": RiskLevel.STANDARD,
            "wood": RiskLevel.HIGH,
            "coffee": RiskLevel.STANDARD,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[
            "Cross River", "Edo", "Ondo", "Ogun"
        ],
        certifications_recognized=["Rainforest Alliance", "UTZ"],
        governance_score=35,
        enforcement_score=30,
        notes="One of highest deforestation rates globally. Cocoa expansion main driver."
    ),

    # =========================================================================
    # STANDARD RISK COUNTRIES
    # =========================================================================

    "CO": CountryRisk(
        country_code="CO",
        country_name="Colombia",
        risk_level=RiskLevel.STANDARD,
        risk_score=58,
        forest_data=ForestData(
            forest_cover_percentage=52.7,
            forest_cover_2020=53.5,
            annual_deforestation_rate=-0.5,
            primary_forest_percentage=46.2,
            forest_area_km2=585017,
            tree_cover_loss_2020_2023=4200,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Cattle ranching",
            "Coca cultivation",
            "Agricultural frontier expansion",
            "Illegal mining"
        ],
        commodity_risks={
            "coffee": RiskLevel.LOW,
            "palm_oil": RiskLevel.STANDARD,
            "cattle": RiskLevel.STANDARD,
            "cocoa": RiskLevel.STANDARD,
            "wood": RiskLevel.STANDARD,
            "rubber": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[
            "Caqueta", "Meta", "Guaviare", "Putumayo"
        ],
        certifications_recognized=["Rainforest Alliance", "UTZ", "FSC", "Fairtrade"],
        governance_score=55,
        enforcement_score=50,
        notes="Post-conflict deforestation increase. Coffee generally low risk."
    ),

    "VN": CountryRisk(
        country_code="VN",
        country_name="Vietnam",
        risk_level=RiskLevel.STANDARD,
        risk_score=52,
        forest_data=ForestData(
            forest_cover_percentage=42.1,
            forest_cover_2020=42.0,
            annual_deforestation_rate=0.5,  # Positive = reforestation
            primary_forest_percentage=0.6,
            forest_area_km2=137970,
            tree_cover_loss_2020_2023=1200,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Plantation establishment",
            "Infrastructure",
            "Agricultural conversion"
        ],
        commodity_risks={
            "coffee": RiskLevel.STANDARD,
            "rubber": RiskLevel.STANDARD,
            "wood": RiskLevel.STANDARD,
            "cocoa": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[
            "Central Highlands", "Tay Nguyen"
        ],
        certifications_recognized=["FSC", "PEFC", "Rainforest Alliance", "FLEGT"],
        governance_score=60,
        enforcement_score=55,
        notes="FLEGT VPA in force. Major wood processing hub - verify origin of imports."
    ),

    "TH": CountryRisk(
        country_code="TH",
        country_name="Thailand",
        risk_level=RiskLevel.STANDARD,
        risk_score=48,
        forest_data=ForestData(
            forest_cover_percentage=32.6,
            forest_cover_2020=32.0,
            annual_deforestation_rate=0.3,
            primary_forest_percentage=15.1,
            forest_area_km2=167590,
            tree_cover_loss_2020_2023=800,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Agricultural encroachment",
            "Rubber plantation expansion",
            "Infrastructure"
        ],
        commodity_risks={
            "rubber": RiskLevel.STANDARD,
            "palm_oil": RiskLevel.LOW,
            "wood": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=["Southern region", "Northeast"],
        certifications_recognized=["FSC", "PEFC", "Rainforest Alliance"],
        governance_score=65,
        enforcement_score=60,
        notes="World's largest rubber producer. Generally good governance."
    ),

    "EC": CountryRisk(
        country_code="EC",
        country_name="Ecuador",
        risk_level=RiskLevel.STANDARD,
        risk_score=55,
        forest_data=ForestData(
            forest_cover_percentage=51.5,
            forest_cover_2020=52.0,
            annual_deforestation_rate=-0.4,
            primary_forest_percentage=35.8,
            forest_area_km2=128600,
            tree_cover_loss_2020_2023=1500,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Palm oil expansion",
            "Cattle ranching",
            "Mining",
            "Road construction"
        ],
        commodity_risks={
            "cocoa": RiskLevel.STANDARD,
            "palm_oil": RiskLevel.STANDARD,
            "coffee": RiskLevel.LOW,
            "wood": RiskLevel.STANDARD,
            "cattle": RiskLevel.STANDARD,
            "rubber": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=["Esmeraldas", "Oriente region"],
        certifications_recognized=["Rainforest Alliance", "FSC", "Organic"],
        governance_score=55,
        enforcement_score=50,
        notes="Fine flavor cocoa producer. Amazon region higher risk."
    ),

    "HN": CountryRisk(
        country_code="HN",
        country_name="Honduras",
        risk_level=RiskLevel.STANDARD,
        risk_score=60,
        forest_data=ForestData(
            forest_cover_percentage=41.0,
            forest_cover_2020=42.5,
            annual_deforestation_rate=-1.0,
            primary_forest_percentage=12.5,
            forest_area_km2=46000,
            tree_cover_loss_2020_2023=1200,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Cattle ranching",
            "Palm oil expansion",
            "Coffee cultivation",
            "Illegal logging"
        ],
        commodity_risks={
            "palm_oil": RiskLevel.STANDARD,
            "coffee": RiskLevel.STANDARD,
            "wood": RiskLevel.STANDARD,
            "cattle": RiskLevel.STANDARD,
            "cocoa": RiskLevel.LOW,
            "rubber": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=["Mosquitia", "Olancho", "Colon"],
        certifications_recognized=["Rainforest Alliance", "FSC", "FLEGT"],
        governance_score=45,
        enforcement_score=40,
        notes="FLEGT VPA negotiations. Palm oil expansion concern."
    ),

    "GT": CountryRisk(
        country_code="GT",
        country_name="Guatemala",
        risk_level=RiskLevel.STANDARD,
        risk_score=58,
        forest_data=ForestData(
            forest_cover_percentage=33.0,
            forest_cover_2020=34.0,
            annual_deforestation_rate=-1.0,
            primary_forest_percentage=20.1,
            forest_area_km2=36000,
            tree_cover_loss_2020_2023=950,
            forest_type=ForestType.TROPICAL_RAINFOREST,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Palm oil expansion",
            "Cattle ranching",
            "Agricultural frontier",
            "Drug trafficking-related clearing"
        ],
        commodity_risks={
            "palm_oil": RiskLevel.STANDARD,
            "coffee": RiskLevel.LOW,
            "wood": RiskLevel.STANDARD,
            "rubber": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "cattle": RiskLevel.STANDARD,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=["Peten", "Izabal", "Alta Verapaz"],
        certifications_recognized=["Rainforest Alliance", "FSC", "RSPO"],
        governance_score=50,
        enforcement_score=45,
        notes="Maya Biosphere Reserve threatened. Palm oil expansion main driver."
    ),

    "ET": CountryRisk(
        country_code="ET",
        country_name="Ethiopia",
        risk_level=RiskLevel.STANDARD,
        risk_score=55,
        forest_data=ForestData(
            forest_cover_percentage=15.5,
            forest_cover_2020=16.0,
            annual_deforestation_rate=-0.8,
            primary_forest_percentage=3.2,
            forest_area_km2=170000,
            tree_cover_loss_2020_2023=1800,
            forest_type=ForestType.TROPICAL_DRY,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Agricultural expansion",
            "Coffee forest degradation",
            "Fuelwood collection",
            "Population pressure"
        ],
        commodity_risks={
            "coffee": RiskLevel.STANDARD,
            "wood": RiskLevel.STANDARD,
            "cattle": RiskLevel.STANDARD,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "rubber": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[
            "Oromia", "SNNPR", "Gambella"
        ],
        certifications_recognized=["Rainforest Alliance", "Organic", "Fairtrade"],
        governance_score=45,
        enforcement_score=40,
        notes="Birthplace of coffee. Forest coffee systems need protection."
    ),

    "MX": CountryRisk(
        country_code="MX",
        country_name="Mexico",
        risk_level=RiskLevel.STANDARD,
        risk_score=50,
        forest_data=ForestData(
            forest_cover_percentage=33.7,
            forest_cover_2020=34.0,
            annual_deforestation_rate=-0.2,
            primary_forest_percentage=21.5,
            forest_area_km2=659000,
            tree_cover_loss_2020_2023=2200,
            forest_type=ForestType.TROPICAL_DRY,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Cattle ranching",
            "Soy and palm oil expansion",
            "Avocado cultivation",
            "Illegal logging"
        ],
        commodity_risks={
            "cattle": RiskLevel.STANDARD,
            "wood": RiskLevel.STANDARD,
            "palm_oil": RiskLevel.STANDARD,
            "coffee": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "rubber": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[
            "Chiapas", "Yucatan Peninsula", "Tabasco"
        ],
        certifications_recognized=["FSC", "Rainforest Alliance"],
        governance_score=55,
        enforcement_score=50,
        notes="Ejido system creates complexity. Lacandon Jungle at risk."
    ),

    "IN": CountryRisk(
        country_code="IN",
        country_name="India",
        risk_level=RiskLevel.STANDARD,
        risk_score=45,
        forest_data=ForestData(
            forest_cover_percentage=24.1,
            forest_cover_2020=23.8,
            annual_deforestation_rate=0.4,  # Net gain
            primary_forest_percentage=2.2,
            forest_area_km2=802088,
            tree_cover_loss_2020_2023=1500,
            forest_type=ForestType.TROPICAL_DRY,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Mining",
            "Infrastructure",
            "Agricultural encroachment",
            "Plantation expansion"
        ],
        commodity_risks={
            "rubber": RiskLevel.STANDARD,
            "coffee": RiskLevel.LOW,
            "wood": RiskLevel.STANDARD,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[
            "Northeast India", "Western Ghats", "Central India"
        ],
        certifications_recognized=["FSC", "Rainforest Alliance", "Organic"],
        governance_score=60,
        enforcement_score=55,
        notes="Net forest gain but quality issues. Primary forest still declining."
    ),

    # =========================================================================
    # LOW RISK COUNTRIES - Standard Due Diligence Sufficient
    # =========================================================================

    "FR": CountryRisk(
        country_code="FR",
        country_name="France",
        risk_level=RiskLevel.LOW,
        risk_score=15,
        forest_data=ForestData(
            forest_cover_percentage=31.0,
            forest_cover_2020=30.8,
            annual_deforestation_rate=0.8,  # Net gain
            primary_forest_percentage=0.1,
            forest_area_km2=172530,
            tree_cover_loss_2020_2023=200,
            forest_type=ForestType.TEMPERATE,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Urbanization",
            "Infrastructure",
            "Agricultural conversion (minimal)"
        ],
        commodity_risks={
            "wood": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[],
        certifications_recognized=["PEFC", "FSC"],
        governance_score=90,
        enforcement_score=90,
        notes="EU member state. Strong forest governance. Net forest gain."
    ),

    "DE": CountryRisk(
        country_code="DE",
        country_name="Germany",
        risk_level=RiskLevel.LOW,
        risk_score=12,
        forest_data=ForestData(
            forest_cover_percentage=32.7,
            forest_cover_2020=32.5,
            annual_deforestation_rate=0.1,
            primary_forest_percentage=0.0,
            forest_area_km2=114190,
            tree_cover_loss_2020_2023=150,
            forest_type=ForestType.TEMPERATE,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Infrastructure",
            "Climate change impacts (bark beetle)",
            "Urbanization"
        ],
        commodity_risks={
            "wood": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[],
        certifications_recognized=["PEFC", "FSC"],
        governance_score=95,
        enforcement_score=95,
        notes="EU member state. Excellent forest governance."
    ),

    "SE": CountryRisk(
        country_code="SE",
        country_name="Sweden",
        risk_level=RiskLevel.LOW,
        risk_score=10,
        forest_data=ForestData(
            forest_cover_percentage=68.7,
            forest_cover_2020=68.5,
            annual_deforestation_rate=0.1,
            primary_forest_percentage=2.3,
            forest_area_km2=280730,
            tree_cover_loss_2020_2023=100,
            forest_type=ForestType.BOREAL,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Managed forestry rotation",
            "Infrastructure",
            "Mining"
        ],
        commodity_risks={
            "wood": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[],
        certifications_recognized=["FSC", "PEFC"],
        governance_score=95,
        enforcement_score=95,
        notes="EU member state. Major wood producer with excellent governance."
    ),

    "FI": CountryRisk(
        country_code="FI",
        country_name="Finland",
        risk_level=RiskLevel.LOW,
        risk_score=10,
        forest_data=ForestData(
            forest_cover_percentage=73.1,
            forest_cover_2020=73.0,
            annual_deforestation_rate=0.0,
            primary_forest_percentage=0.5,
            forest_area_km2=223900,
            tree_cover_loss_2020_2023=80,
            forest_type=ForestType.BOREAL,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Managed forestry",
            "Infrastructure",
            "Peatland drainage (historical)"
        ],
        commodity_risks={
            "wood": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[],
        certifications_recognized=["PEFC", "FSC"],
        governance_score=95,
        enforcement_score=95,
        notes="EU member state. World leader in sustainable forestry."
    ),

    "CA": CountryRisk(
        country_code="CA",
        country_name="Canada",
        risk_level=RiskLevel.LOW,
        risk_score=18,
        forest_data=ForestData(
            forest_cover_percentage=38.7,
            forest_cover_2020=38.8,
            annual_deforestation_rate=0.0,
            primary_forest_percentage=25.5,
            forest_area_km2=3470690,
            tree_cover_loss_2020_2023=2500,
            forest_type=ForestType.BOREAL,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Wildfires (natural)",
            "Oil sands development",
            "Mining",
            "Agricultural conversion"
        ],
        commodity_risks={
            "wood": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW
        },
        regions_of_concern=["Alberta oil sands region"],
        certifications_recognized=["FSC", "SFI", "CSA", "PEFC"],
        governance_score=90,
        enforcement_score=85,
        notes="Excellent governance. Boreal forest conversion for oil sands limited concern."
    ),

    "US": CountryRisk(
        country_code="US",
        country_name="United States",
        risk_level=RiskLevel.LOW,
        risk_score=20,
        forest_data=ForestData(
            forest_cover_percentage=33.9,
            forest_cover_2020=33.8,
            annual_deforestation_rate=0.0,
            primary_forest_percentage=7.5,
            forest_area_km2=3100950,
            tree_cover_loss_2020_2023=2000,
            forest_type=ForestType.TEMPERATE,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Urbanization",
            "Wildfires",
            "Agricultural conversion",
            "Energy development"
        ],
        commodity_risks={
            "wood": RiskLevel.LOW,
            "soya": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW
        },
        regions_of_concern=[],
        certifications_recognized=["FSC", "SFI", "ATFS", "PEFC"],
        governance_score=85,
        enforcement_score=85,
        notes="Strong forest governance. Soy generally low risk."
    ),

    "AU": CountryRisk(
        country_code="AU",
        country_name="Australia",
        risk_level=RiskLevel.LOW,
        risk_score=25,
        forest_data=ForestData(
            forest_cover_percentage=16.2,
            forest_cover_2020=16.5,
            annual_deforestation_rate=-0.1,
            primary_forest_percentage=14.7,
            forest_area_km2=1341000,
            tree_cover_loss_2020_2023=800,
            forest_type=ForestType.TEMPERATE,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Bushfires",
            "Agricultural clearing (Queensland)",
            "Mining",
            "Urbanization"
        ],
        commodity_risks={
            "cattle": RiskLevel.LOW,
            "wood": RiskLevel.LOW,
            "soya": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW
        },
        regions_of_concern=["Queensland (some areas)"],
        certifications_recognized=["FSC", "AFS", "PEFC"],
        governance_score=85,
        enforcement_score=80,
        notes="Generally low risk. Some Queensland land clearing concerns."
    ),

    "NZ": CountryRisk(
        country_code="NZ",
        country_name="New Zealand",
        risk_level=RiskLevel.LOW,
        risk_score=12,
        forest_data=ForestData(
            forest_cover_percentage=38.6,
            forest_cover_2020=38.4,
            annual_deforestation_rate=0.1,
            primary_forest_percentage=15.0,
            forest_area_km2=100680,
            tree_cover_loss_2020_2023=50,
            forest_type=ForestType.TEMPERATE,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Plantation harvest cycles",
            "Dairy conversion (historical)"
        ],
        commodity_risks={
            "wood": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[],
        certifications_recognized=["FSC", "PEFC"],
        governance_score=95,
        enforcement_score=95,
        notes="Excellent governance. Major plantation wood exporter."
    ),

    "CL": CountryRisk(
        country_code="CL",
        country_name="Chile",
        risk_level=RiskLevel.LOW,
        risk_score=22,
        forest_data=ForestData(
            forest_cover_percentage=24.4,
            forest_cover_2020=24.5,
            annual_deforestation_rate=0.0,
            primary_forest_percentage=17.3,
            forest_area_km2=182210,
            tree_cover_loss_2020_2023=300,
            forest_type=ForestType.TEMPERATE,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Wildfires",
            "Plantation expansion",
            "Native forest to plantation conversion"
        ],
        commodity_risks={
            "wood": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[],
        certifications_recognized=["FSC", "PEFC", "CERTFOR"],
        governance_score=80,
        enforcement_score=75,
        notes="Major plantation wood producer. Native forest conversion concern addressed."
    ),

    "UY": CountryRisk(
        country_code="UY",
        country_name="Uruguay",
        risk_level=RiskLevel.LOW,
        risk_score=20,
        forest_data=ForestData(
            forest_cover_percentage=11.8,
            forest_cover_2020=11.5,
            annual_deforestation_rate=1.5,  # Net gain - plantation expansion
            primary_forest_percentage=1.5,
            forest_area_km2=20700,
            tree_cover_loss_2020_2023=50,
            forest_type=ForestType.TEMPERATE,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Minimal - mostly plantation forest"
        ],
        commodity_risks={
            "cattle": RiskLevel.LOW,
            "wood": RiskLevel.LOW,
            "soya": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW
        },
        regions_of_concern=[],
        certifications_recognized=["FSC"],
        governance_score=80,
        enforcement_score=80,
        notes="Low native forest. Plantation-based wood production."
    ),

    "AR": CountryRisk(
        country_code="AR",
        country_name="Argentina",
        risk_level=RiskLevel.STANDARD,
        risk_score=55,
        forest_data=ForestData(
            forest_cover_percentage=10.2,
            forest_cover_2020=10.8,
            annual_deforestation_rate=-0.8,
            primary_forest_percentage=8.2,
            forest_area_km2=283760,
            tree_cover_loss_2020_2023=1500,
            forest_type=ForestType.TROPICAL_DRY,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Soy cultivation",
            "Cattle ranching",
            "Agricultural expansion"
        ],
        commodity_risks={
            "soya": RiskLevel.STANDARD,
            "cattle": RiskLevel.STANDARD,
            "wood": RiskLevel.STANDARD,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW
        },
        regions_of_concern=[
            "Gran Chaco", "Salta", "Santiago del Estero", "Chaco province"
        ],
        certifications_recognized=["FSC", "RTRS"],
        governance_score=55,
        enforcement_score=50,
        notes="Gran Chaco deforestation ongoing. Forest Law enforcement variable."
    ),

    "UA": CountryRisk(
        country_code="UA",
        country_name="Ukraine",
        risk_level=RiskLevel.STANDARD,
        risk_score=45,
        forest_data=ForestData(
            forest_cover_percentage=16.7,
            forest_cover_2020=16.5,
            annual_deforestation_rate=0.0,
            primary_forest_percentage=0.2,
            forest_area_km2=98770,
            tree_cover_loss_2020_2023=500,
            forest_type=ForestType.TEMPERATE,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Illegal logging",
            "Conflict-related damage",
            "Agricultural pressure"
        ],
        commodity_risks={
            "wood": RiskLevel.STANDARD,
            "soya": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW
        },
        regions_of_concern=["Carpathian region"],
        certifications_recognized=["FSC", "PEFC"],
        governance_score=50,
        enforcement_score=45,
        notes="FLEGT VPA negotiations. Illegal logging concerns in Carpathians."
    ),

    "RU": CountryRisk(
        country_code="RU",
        country_name="Russian Federation",
        risk_level=RiskLevel.STANDARD,
        risk_score=48,
        forest_data=ForestData(
            forest_cover_percentage=49.8,
            forest_cover_2020=49.8,
            annual_deforestation_rate=0.0,
            primary_forest_percentage=26.3,
            forest_area_km2=8151356,
            tree_cover_loss_2020_2023=5000,
            forest_type=ForestType.BOREAL,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Logging (legal and illegal)",
            "Wildfires",
            "Mining",
            "Infrastructure"
        ],
        commodity_risks={
            "wood": RiskLevel.STANDARD,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW
        },
        regions_of_concern=[
            "Far East", "Siberia", "Arkhangelsk"
        ],
        certifications_recognized=["FSC", "PEFC"],
        governance_score=45,
        enforcement_score=40,
        notes="Largest forest country. Illegal logging and wildfires major issues. EU sanctions apply."
    ),

    "CN": CountryRisk(
        country_code="CN",
        country_name="China",
        risk_level=RiskLevel.STANDARD,
        risk_score=40,
        forest_data=ForestData(
            forest_cover_percentage=23.3,
            forest_cover_2020=22.5,
            annual_deforestation_rate=1.0,  # Net gain - afforestation
            primary_forest_percentage=3.0,
            forest_area_km2=2199780,
            tree_cover_loss_2020_2023=1200,
            forest_type=ForestType.TEMPERATE,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Infrastructure",
            "Urbanization",
            "Agricultural conversion"
        ],
        commodity_risks={
            "wood": RiskLevel.STANDARD,
            "rubber": RiskLevel.STANDARD,
            "cattle": RiskLevel.LOW,
            "soya": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW
        },
        regions_of_concern=["Yunnan", "Hainan"],
        certifications_recognized=["FSC", "PEFC", "CFCC"],
        governance_score=60,
        enforcement_score=60,
        notes="Major processing hub. Verify origin of wood imports. Net forest gain."
    ),

    "JP": CountryRisk(
        country_code="JP",
        country_name="Japan",
        risk_level=RiskLevel.LOW,
        risk_score=15,
        forest_data=ForestData(
            forest_cover_percentage=68.4,
            forest_cover_2020=68.4,
            annual_deforestation_rate=0.0,
            primary_forest_percentage=5.8,
            forest_area_km2=249350,
            tree_cover_loss_2020_2023=100,
            forest_type=ForestType.TEMPERATE,
            data_source="FAO FRA 2020, Global Forest Watch 2023",
            last_updated=date(2024, 6, 1)
        ),
        primary_deforestation_drivers=[
            "Infrastructure",
            "Natural disasters"
        ],
        commodity_risks={
            "wood": RiskLevel.LOW,
            "cattle": RiskLevel.LOW,
            "palm_oil": RiskLevel.LOW,
            "cocoa": RiskLevel.LOW,
            "coffee": RiskLevel.LOW,
            "rubber": RiskLevel.LOW,
            "soya": RiskLevel.LOW
        },
        regions_of_concern=[],
        certifications_recognized=["FSC", "PEFC", "SGEC"],
        governance_score=90,
        enforcement_score=90,
        notes="Excellent governance. Under-utilized domestic forests."
    ),
}


# =============================================================================
# Region-Specific Risk Database
# =============================================================================

REGION_RISK_DATABASE: Dict[str, List[RegionRisk]] = {
    "BR": [
        RegionRisk(
            region_name="Amazonia Legal",
            country_code="BR",
            risk_level=RiskLevel.HIGH,
            risk_score=95,
            forest_cover_percentage=80.0,
            deforestation_hotspot=True,
            protected_areas=["Tumucumaque", "Jau", "Xingu"],
            indigenous_territories=["Kayapo", "Yanomami", "Xingu"],
            primary_commodities=["cattle", "soya", "wood"],
            coordinates_bounds=((-10.0, -74.0), (5.0, -44.0))
        ),
        RegionRisk(
            region_name="Cerrado",
            country_code="BR",
            risk_level=RiskLevel.HIGH,
            risk_score=90,
            forest_cover_percentage=25.0,
            deforestation_hotspot=True,
            protected_areas=["Chapada dos Veadeiros", "Emas"],
            indigenous_territories=["Xerente", "Kraho"],
            primary_commodities=["soya", "cattle", "coffee"],
            coordinates_bounds=((-24.0, -60.0), (-2.0, -41.0))
        ),
        RegionRisk(
            region_name="Mato Grosso",
            country_code="BR",
            risk_level=RiskLevel.HIGH,
            risk_score=88,
            forest_cover_percentage=45.0,
            deforestation_hotspot=True,
            protected_areas=["Xingu", "Cristalino"],
            indigenous_territories=["Xingu", "Enawene Nawe"],
            primary_commodities=["soya", "cattle"],
            coordinates_bounds=((-18.0, -62.0), (-7.0, -50.0))
        ),
    ],
    "ID": [
        RegionRisk(
            region_name="Sumatra",
            country_code="ID",
            risk_level=RiskLevel.HIGH,
            risk_score=85,
            forest_cover_percentage=35.0,
            deforestation_hotspot=True,
            protected_areas=["Leuser", "Kerinci Seblat"],
            indigenous_territories=["Orang Rimba"],
            primary_commodities=["palm_oil", "rubber", "wood"],
            coordinates_bounds=((-6.0, 95.0), (6.0, 106.0))
        ),
        RegionRisk(
            region_name="Kalimantan",
            country_code="ID",
            risk_level=RiskLevel.HIGH,
            risk_score=88,
            forest_cover_percentage=55.0,
            deforestation_hotspot=True,
            protected_areas=["Tanjung Puting", "Sebangau"],
            indigenous_territories=["Dayak"],
            primary_commodities=["palm_oil", "wood"],
            coordinates_bounds=((-4.5, 108.0), (4.5, 119.0))
        ),
        RegionRisk(
            region_name="Papua",
            country_code="ID",
            risk_level=RiskLevel.HIGH,
            risk_score=82,
            forest_cover_percentage=85.0,
            deforestation_hotspot=True,
            protected_areas=["Lorentz"],
            indigenous_territories=["Multiple Papuan tribes"],
            primary_commodities=["palm_oil", "wood"],
            coordinates_bounds=((-9.0, 129.0), (0.0, 141.0))
        ),
    ],
    "CI": [
        RegionRisk(
            region_name="Western Forest Zone",
            country_code="CI",
            risk_level=RiskLevel.HIGH,
            risk_score=90,
            forest_cover_percentage=12.0,
            deforestation_hotspot=True,
            protected_areas=["Tai National Park", "Mont Peko"],
            indigenous_territories=[],
            primary_commodities=["cocoa", "rubber"],
            coordinates_bounds=((5.0, -8.5), (7.5, -6.0))
        ),
    ],
}


# =============================================================================
# Lookup Functions
# =============================================================================

def get_country_risk(country_code: str) -> Optional[CountryRisk]:
    """
    Get risk assessment for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code

    Returns:
        CountryRisk object if found, None otherwise
    """
    return COUNTRY_RISK_DATABASE.get(country_code.upper())


def get_risk_level(country_code: str) -> RiskLevel:
    """
    Get risk level for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code

    Returns:
        RiskLevel enum value (defaults to STANDARD if not found)
    """
    country = get_country_risk(country_code)
    if country is None:
        return RiskLevel.STANDARD
    return country.risk_level


def get_risk_score(country_code: str) -> float:
    """
    Get numeric risk score for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code

    Returns:
        Risk score 0-100 (defaults to 50 if not found)
    """
    country = get_country_risk(country_code)
    if country is None:
        return 50.0
    return country.risk_score


def get_commodity_risk(country_code: str, commodity_type: str) -> RiskLevel:
    """
    Get commodity-specific risk for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code
        commodity_type: EUDR commodity type (cattle, cocoa, coffee, palm_oil, rubber, soya, wood)

    Returns:
        RiskLevel enum value
    """
    country = get_country_risk(country_code)
    if country is None:
        return RiskLevel.STANDARD

    commodity_lower = commodity_type.lower()
    return country.commodity_risks.get(commodity_lower, country.risk_level)


def get_due_diligence_level(country_code: str, commodity_type: str = None) -> DueDiligenceLevel:
    """
    Determine required due diligence level based on risk.

    Args:
        country_code: ISO 3166-1 alpha-2 country code
        commodity_type: Optional commodity type for commodity-specific assessment

    Returns:
        DueDiligenceLevel enum value
    """
    if commodity_type:
        risk_level = get_commodity_risk(country_code, commodity_type)
    else:
        risk_level = get_risk_level(country_code)

    if risk_level == RiskLevel.LOW:
        return DueDiligenceLevel.STANDARD
    elif risk_level == RiskLevel.STANDARD:
        return DueDiligenceLevel.ENHANCED
    else:
        return DueDiligenceLevel.FULL_VERIFICATION


def requires_satellite_verification(country_code: str, commodity_type: str) -> bool:
    """
    Determine if satellite verification is required.

    Args:
        country_code: ISO 3166-1 alpha-2 country code
        commodity_type: EUDR commodity type

    Returns:
        True if satellite verification required
    """
    risk_level = get_commodity_risk(country_code, commodity_type)
    return risk_level == RiskLevel.HIGH


def get_regions_of_concern(country_code: str) -> List[str]:
    """
    Get list of high-risk regions within a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code

    Returns:
        List of region names
    """
    country = get_country_risk(country_code)
    if country is None:
        return []
    return country.regions_of_concern


def get_region_risks(country_code: str) -> List[RegionRisk]:
    """
    Get detailed risk data for sub-national regions.

    Args:
        country_code: ISO 3166-1 alpha-2 country code

    Returns:
        List of RegionRisk objects
    """
    return REGION_RISK_DATABASE.get(country_code.upper(), [])


def is_deforestation_hotspot(country_code: str, region: str = None) -> bool:
    """
    Check if location is a deforestation hotspot.

    Args:
        country_code: ISO 3166-1 alpha-2 country code
        region: Optional region name

    Returns:
        True if hotspot
    """
    country = get_country_risk(country_code)
    if country is None:
        return False

    if country.risk_level == RiskLevel.HIGH:
        if region is None:
            return True

        regions = get_region_risks(country_code)
        for r in regions:
            if region.lower() in r.region_name.lower():
                return r.deforestation_hotspot

    return False


def get_forest_data(country_code: str) -> Optional[ForestData]:
    """
    Get forest statistics for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code

    Returns:
        ForestData object if found
    """
    country = get_country_risk(country_code)
    if country is None:
        return None
    return country.forest_data


def assess_country_risk(
    country_code: str,
    commodity_type: str,
    region: str = None,
    production_year: int = None
) -> Dict:
    """
    Perform comprehensive country risk assessment.

    Args:
        country_code: ISO 3166-1 alpha-2 country code
        commodity_type: EUDR commodity type
        region: Optional sub-national region
        production_year: Optional production year

    Returns:
        Complete risk assessment dictionary
    """
    country = get_country_risk(country_code)

    if country is None:
        return {
            "risk_level": RiskLevel.STANDARD.value,
            "risk_score": 50,
            "risk_uri": f"risk://eudr/unknown/{country_code}",
            "country_name": "Unknown",
            "forest_cover_percentage": None,
            "deforestation_rate": None,
            "primary_deforestation_drivers": [],
            "due_diligence_level": DueDiligenceLevel.ENHANCED.value,
            "satellite_verification_required": True,
            "data_sources": [],
            "last_updated": None,
            "warnings": [f"Country {country_code} not in risk database"]
        }

    commodity_risk = get_commodity_risk(country_code, commodity_type)
    dd_level = get_due_diligence_level(country_code, commodity_type)
    sat_required = requires_satellite_verification(country_code, commodity_type)

    # Check for region-specific risks
    region_warning = None
    if region:
        regions = get_region_risks(country_code)
        for r in regions:
            if region.lower() in r.region_name.lower():
                if r.risk_level == RiskLevel.HIGH:
                    region_warning = f"Region {region} is classified as high-risk deforestation area"
                break

    warnings = []
    if region_warning:
        warnings.append(region_warning)
    if commodity_risk == RiskLevel.HIGH:
        warnings.append(f"{commodity_type} is high-risk commodity in {country.country_name}")

    return {
        "risk_level": commodity_risk.value,
        "risk_score": country.risk_score,
        "risk_uri": f"risk://eudr/2024/{country_code}/{commodity_type}",
        "country_name": country.country_name,
        "country_code": country.country_code,
        "forest_cover_percentage": country.forest_data.forest_cover_percentage,
        "deforestation_rate": country.forest_data.annual_deforestation_rate,
        "primary_deforestation_drivers": country.primary_deforestation_drivers,
        "due_diligence_level": dd_level.value,
        "satellite_verification_required": sat_required,
        "regions_of_concern": country.regions_of_concern,
        "certifications_recognized": country.certifications_recognized,
        "governance_score": country.governance_score,
        "enforcement_score": country.enforcement_score,
        "data_sources": [
            country.forest_data.data_source,
            "EC Country Benchmarking System (provisional)",
            "Global Forest Watch 2023"
        ],
        "last_updated": country.forest_data.last_updated.isoformat(),
        "warnings": warnings
    }


def get_high_risk_countries() -> List[str]:
    """Get list of all high-risk country codes."""
    return [code for code, country in COUNTRY_RISK_DATABASE.items()
            if country.risk_level == RiskLevel.HIGH]


def get_low_risk_countries() -> List[str]:
    """Get list of all low-risk country codes."""
    return [code for code, country in COUNTRY_RISK_DATABASE.items()
            if country.risk_level == RiskLevel.LOW]


def get_database_stats() -> Dict:
    """Get statistics about the country risk database."""
    risk_counts = {level.value: 0 for level in RiskLevel}
    for country in COUNTRY_RISK_DATABASE.values():
        risk_counts[country.risk_level.value] += 1

    return {
        "total_countries": len(COUNTRY_RISK_DATABASE),
        "countries_by_risk_level": risk_counts,
        "regions_covered": sum(len(regions) for regions in REGION_RISK_DATABASE.values()),
        "cutoff_date": EUDR_CUTOFF_DATE.isoformat(),
        "data_version": "2024-06",
        "primary_sources": [
            "FAO Global Forest Resources Assessment 2020",
            "Global Forest Watch 2023",
            "EC Country Benchmarking System (provisional)",
            "World Resources Institute"
        ]
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "RiskLevel",
    "DueDiligenceLevel",
    "ForestType",
    # Constants
    "EUDR_CUTOFF_DATE",
    # Data classes
    "ForestData",
    "CountryRisk",
    "RegionRisk",
    # Databases
    "COUNTRY_RISK_DATABASE",
    "REGION_RISK_DATABASE",
    # Functions
    "get_country_risk",
    "get_risk_level",
    "get_risk_score",
    "get_commodity_risk",
    "get_due_diligence_level",
    "requires_satellite_verification",
    "get_regions_of_concern",
    "get_region_risks",
    "is_deforestation_hotspot",
    "get_forest_data",
    "assess_country_risk",
    "get_high_risk_countries",
    "get_low_risk_countries",
    "get_database_stats",
]
