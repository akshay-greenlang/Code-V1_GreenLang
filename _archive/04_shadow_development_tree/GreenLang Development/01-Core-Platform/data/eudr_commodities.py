"""
EUDR Commodities Database
=========================
EU Deforestation Regulation (EU) 2023/1115 Commodity Classifications

This module provides a comprehensive database of the 7 regulated commodities
and their derived products under EUDR, with CN (Combined Nomenclature) codes,
risk categories, and traceability requirements.

Regulatory Reference:
- Regulation (EU) 2023/1115 - Annex I (Relevant commodities and products)
- EU Combined Nomenclature 2024

CRITICAL: Deforestation-free cutoff date is December 31, 2020
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import date


# =============================================================================
# Enums and Constants
# =============================================================================

class CommodityType(Enum):
    """Seven regulated commodities under EUDR."""
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"
    DERIVED_PRODUCT = "derived_product"
    NOT_REGULATED = "not_regulated"


class RiskCategory(Enum):
    """Commodity-specific risk categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TraceabilityLevel(Enum):
    """Required traceability level."""
    PLOT = "plot"  # GPS coordinates of production plot
    FARM = "farm"  # Farm-level traceability
    REGION = "region"  # Regional-level (for low-risk areas)
    FACILITY = "facility"  # Processing facility


# EUDR Cutoff Date - Products must be deforestation-free after this date
EUDR_CUTOFF_DATE = date(2020, 12, 31)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CNCode:
    """EU Combined Nomenclature Code with EUDR classification."""
    code: str
    description: str
    commodity_type: CommodityType
    is_derived: bool = False
    derived_from: List[CommodityType] = field(default_factory=list)
    risk_category: RiskCategory = RiskCategory.MEDIUM
    traceability_requirements: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Commodity:
    """EUDR Regulated Commodity."""
    commodity_type: CommodityType
    name: str
    description: str
    cn_chapters: List[str]  # CN code chapter prefixes
    default_risk_category: RiskCategory
    traceability_level: TraceabilityLevel
    key_producing_countries: List[str]
    deforestation_drivers: List[str]
    derived_products: List[str]
    annex_reference: str  # EUDR Annex I reference


# =============================================================================
# Commodity Definitions
# =============================================================================

EUDR_COMMODITIES: Dict[CommodityType, Commodity] = {
    CommodityType.CATTLE: Commodity(
        commodity_type=CommodityType.CATTLE,
        name="Cattle",
        description="Live bovine animals and derived products including beef, leather",
        cn_chapters=["0102", "0201", "0202", "0206", "0210", "4101", "4104", "4107"],
        default_risk_category=RiskCategory.HIGH,
        traceability_level=TraceabilityLevel.FARM,
        key_producing_countries=["BR", "AR", "PY", "UY", "AU", "US"],
        deforestation_drivers=["Pasture expansion", "Feed crop production"],
        derived_products=[
            "Live cattle", "Bovine meat (fresh/chilled/frozen)",
            "Bovine offal", "Bovine leather", "Bovine hides"
        ],
        annex_reference="Annex I - Cattle"
    ),

    CommodityType.COCOA: Commodity(
        commodity_type=CommodityType.COCOA,
        name="Cocoa",
        description="Cocoa beans and cocoa-derived products including chocolate",
        cn_chapters=["1801", "1802", "1803", "1804", "1805", "1806"],
        default_risk_category=RiskCategory.HIGH,
        traceability_level=TraceabilityLevel.PLOT,
        key_producing_countries=["CI", "GH", "ID", "NG", "CM", "EC"],
        deforestation_drivers=["Farm expansion", "Shade-free cultivation"],
        derived_products=[
            "Cocoa beans", "Cocoa shells/husks", "Cocoa paste",
            "Cocoa butter/fat/oil", "Cocoa powder", "Chocolate"
        ],
        annex_reference="Annex I - Cocoa"
    ),

    CommodityType.COFFEE: Commodity(
        commodity_type=CommodityType.COFFEE,
        name="Coffee",
        description="Coffee beans and coffee-derived products",
        cn_chapters=["0901", "2101"],
        default_risk_category=RiskCategory.MEDIUM,
        traceability_level=TraceabilityLevel.PLOT,
        key_producing_countries=["BR", "VN", "CO", "ID", "ET", "HN", "PE"],
        deforestation_drivers=["Farm expansion", "Sun cultivation"],
        derived_products=[
            "Coffee (not roasted)", "Coffee (roasted)",
            "Coffee husks/skins", "Coffee substitutes", "Coffee extracts"
        ],
        annex_reference="Annex I - Coffee"
    ),

    CommodityType.PALM_OIL: Commodity(
        commodity_type=CommodityType.PALM_OIL,
        name="Palm Oil",
        description="Oil palm and palm oil derived products",
        cn_chapters=["1207", "1511", "1513", "2306", "2905", "3823", "3826"],
        default_risk_category=RiskCategory.HIGH,
        traceability_level=TraceabilityLevel.PLOT,
        key_producing_countries=["ID", "MY", "TH", "CO", "NG", "GT"],
        deforestation_drivers=["Plantation expansion", "Peatland drainage"],
        derived_products=[
            "Palm nuts/kernels", "Crude palm oil", "Refined palm oil",
            "Palm kernel oil", "Palm fatty acids", "Glycerol from palm",
            "Biodiesel from palm"
        ],
        annex_reference="Annex I - Oil palm"
    ),

    CommodityType.RUBBER: Commodity(
        commodity_type=CommodityType.RUBBER,
        name="Rubber",
        description="Natural rubber and rubber-derived products",
        cn_chapters=["4001", "4002", "4005", "4006", "4007", "4008", "4010",
                     "4011", "4012", "4013", "4015", "4016", "4017"],
        default_risk_category=RiskCategory.MEDIUM,
        traceability_level=TraceabilityLevel.PLOT,
        key_producing_countries=["TH", "ID", "VN", "MY", "CI", "CN"],
        deforestation_drivers=["Plantation expansion", "Smallholder expansion"],
        derived_products=[
            "Natural rubber latex", "Natural rubber (solid)",
            "Rubber plates/sheets", "Rubber tubes/pipes",
            "Rubber tires", "Rubber gloves"
        ],
        annex_reference="Annex I - Rubber"
    ),

    CommodityType.SOYA: Commodity(
        commodity_type=CommodityType.SOYA,
        name="Soya",
        description="Soya beans and soya-derived products",
        cn_chapters=["1201", "1208", "1507", "2304"],
        default_risk_category=RiskCategory.HIGH,
        traceability_level=TraceabilityLevel.PLOT,
        key_producing_countries=["BR", "US", "AR", "PY", "CA", "UA"],
        deforestation_drivers=["Agricultural expansion", "Cerrado conversion"],
        derived_products=[
            "Soya beans", "Soya bean flour/meal", "Soya bean oil",
            "Soya bean oilcake", "Soya protein", "Soya lecithin"
        ],
        annex_reference="Annex I - Soya"
    ),

    CommodityType.WOOD: Commodity(
        commodity_type=CommodityType.WOOD,
        name="Wood",
        description="Wood and wood-derived products including paper, furniture",
        cn_chapters=["44", "47", "48", "49", "94"],  # Full chapters
        default_risk_category=RiskCategory.MEDIUM,
        traceability_level=TraceabilityLevel.PLOT,
        key_producing_countries=["BR", "RU", "CA", "US", "ID", "MY", "CN"],
        deforestation_drivers=["Logging", "Land conversion", "Illegal harvesting"],
        derived_products=[
            "Fuel wood", "Wood charcoal", "Wood chips",
            "Sawn wood", "Veneer", "Plywood", "Particle board",
            "Wood pulp", "Paper/paperboard", "Printed materials",
            "Wooden furniture"
        ],
        annex_reference="Annex I - Wood"
    )
}


# =============================================================================
# CN Code Database
# =============================================================================

# Comprehensive CN codes for EUDR-regulated products
# Source: EU Combined Nomenclature 2024 + EUDR Annex I

CN_CODE_DATABASE: Dict[str, CNCode] = {
    # -------------------------------------------------------------------------
    # CATTLE (Chapter 01, 02, 41)
    # -------------------------------------------------------------------------
    "01022110": CNCode(
        code="01022110",
        description="Live purebred breeding bovine animals",
        commodity_type=CommodityType.CATTLE,
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Farm ID", "Animal passport", "Birth location GPS"]
    ),
    "01022190": CNCode(
        code="01022190",
        description="Live bovine animals (other purebred)",
        commodity_type=CommodityType.CATTLE,
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Farm ID", "Animal passport"]
    ),
    "01022910": CNCode(
        code="01022910",
        description="Live bovine animals for slaughter",
        commodity_type=CommodityType.CATTLE,
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Farm ID", "Slaughter destination"]
    ),
    "02011000": CNCode(
        code="02011000",
        description="Bovine carcasses and half-carcasses, fresh or chilled",
        commodity_type=CommodityType.CATTLE,
        is_derived=True,
        derived_from=[CommodityType.CATTLE],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Slaughterhouse ID", "Farm of origin"]
    ),
    "02012020": CNCode(
        code="02012020",
        description="Bovine meat cuts with bone in, fresh or chilled",
        commodity_type=CommodityType.CATTLE,
        is_derived=True,
        derived_from=[CommodityType.CATTLE],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Processing facility", "Batch number"]
    ),
    "02013000": CNCode(
        code="02013000",
        description="Bovine boneless meat, fresh or chilled",
        commodity_type=CommodityType.CATTLE,
        is_derived=True,
        derived_from=[CommodityType.CATTLE],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Processing facility", "Batch number"]
    ),
    "02021000": CNCode(
        code="02021000",
        description="Bovine carcasses and half-carcasses, frozen",
        commodity_type=CommodityType.CATTLE,
        is_derived=True,
        derived_from=[CommodityType.CATTLE],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Slaughterhouse ID", "Freezing date"]
    ),
    "02022030": CNCode(
        code="02022030",
        description="Bovine forequarters frozen",
        commodity_type=CommodityType.CATTLE,
        is_derived=True,
        derived_from=[CommodityType.CATTLE],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Processing facility"]
    ),
    "02023090": CNCode(
        code="02023090",
        description="Bovine boneless meat, frozen (other)",
        commodity_type=CommodityType.CATTLE,
        is_derived=True,
        derived_from=[CommodityType.CATTLE],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Processing facility", "Batch number"]
    ),
    "41012010": CNCode(
        code="41012010",
        description="Whole bovine hides and skins, fresh or preserved",
        commodity_type=CommodityType.CATTLE,
        is_derived=True,
        derived_from=[CommodityType.CATTLE],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Tannery ID", "Origin farm"]
    ),
    "41041111": CNCode(
        code="41041111",
        description="Bovine leather, full grains unsplit",
        commodity_type=CommodityType.CATTLE,
        is_derived=True,
        derived_from=[CommodityType.CATTLE],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Tannery ID", "Batch number"]
    ),

    # -------------------------------------------------------------------------
    # COCOA (Chapter 18)
    # -------------------------------------------------------------------------
    "18010000": CNCode(
        code="18010000",
        description="Cocoa beans, whole or broken, raw or roasted",
        commodity_type=CommodityType.COCOA,
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Farm GPS", "Harvest date", "Cooperative ID"]
    ),
    "18020000": CNCode(
        code="18020000",
        description="Cocoa shells, husks, skins and other cocoa waste",
        commodity_type=CommodityType.COCOA,
        is_derived=True,
        derived_from=[CommodityType.COCOA],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Processing facility"]
    ),
    "18031000": CNCode(
        code="18031000",
        description="Cocoa paste, not defatted",
        commodity_type=CommodityType.COCOA,
        is_derived=True,
        derived_from=[CommodityType.COCOA],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Processing facility", "Bean origin"]
    ),
    "18032000": CNCode(
        code="18032000",
        description="Cocoa paste, wholly or partly defatted",
        commodity_type=CommodityType.COCOA,
        is_derived=True,
        derived_from=[CommodityType.COCOA],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Processing facility", "Bean origin"]
    ),
    "18040000": CNCode(
        code="18040000",
        description="Cocoa butter, fat and oil",
        commodity_type=CommodityType.COCOA,
        is_derived=True,
        derived_from=[CommodityType.COCOA],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Processing facility", "Batch number"]
    ),
    "18050000": CNCode(
        code="18050000",
        description="Cocoa powder, not containing added sugar",
        commodity_type=CommodityType.COCOA,
        is_derived=True,
        derived_from=[CommodityType.COCOA],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Processing facility"]
    ),
    "18061015": CNCode(
        code="18061015",
        description="Cocoa powder with added sugar (10-65% sucrose)",
        commodity_type=CommodityType.COCOA,
        is_derived=True,
        derived_from=[CommodityType.COCOA],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Processing facility"]
    ),
    "18063100": CNCode(
        code="18063100",
        description="Chocolate in blocks, slabs or bars (filled)",
        commodity_type=CommodityType.COCOA,
        is_derived=True,
        derived_from=[CommodityType.COCOA],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Manufacturer", "Cocoa source declaration"]
    ),
    "18069019": CNCode(
        code="18069019",
        description="Chocolate confectionery (other)",
        commodity_type=CommodityType.COCOA,
        is_derived=True,
        derived_from=[CommodityType.COCOA],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Manufacturer", "Cocoa content %"]
    ),

    # -------------------------------------------------------------------------
    # COFFEE (Chapter 09, 21)
    # -------------------------------------------------------------------------
    "09011100": CNCode(
        code="09011100",
        description="Coffee, not roasted, not decaffeinated",
        commodity_type=CommodityType.COFFEE,
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Farm GPS", "Harvest date", "Variety"]
    ),
    "09011200": CNCode(
        code="09011200",
        description="Coffee, not roasted, decaffeinated",
        commodity_type=CommodityType.COFFEE,
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Farm GPS", "Processing facility"]
    ),
    "09012100": CNCode(
        code="09012100",
        description="Coffee, roasted, not decaffeinated",
        commodity_type=CommodityType.COFFEE,
        is_derived=True,
        derived_from=[CommodityType.COFFEE],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Roaster ID", "Green bean origin"]
    ),
    "09012200": CNCode(
        code="09012200",
        description="Coffee, roasted, decaffeinated",
        commodity_type=CommodityType.COFFEE,
        is_derived=True,
        derived_from=[CommodityType.COFFEE],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Roaster ID", "Processing details"]
    ),
    "09019010": CNCode(
        code="09019010",
        description="Coffee husks and skins",
        commodity_type=CommodityType.COFFEE,
        is_derived=True,
        derived_from=[CommodityType.COFFEE],
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Processing facility"]
    ),
    "21011100": CNCode(
        code="21011100",
        description="Extracts, essences of coffee",
        commodity_type=CommodityType.COFFEE,
        is_derived=True,
        derived_from=[CommodityType.COFFEE],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Manufacturer", "Bean source"]
    ),
    "21011200": CNCode(
        code="21011200",
        description="Preparations with basis of coffee extracts",
        commodity_type=CommodityType.COFFEE,
        is_derived=True,
        derived_from=[CommodityType.COFFEE],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Manufacturer", "Coffee content %"]
    ),

    # -------------------------------------------------------------------------
    # PALM OIL (Chapters 12, 15, 23, 29, 38)
    # -------------------------------------------------------------------------
    "12079110": CNCode(
        code="12079110",
        description="Palm nuts and kernels for sowing",
        commodity_type=CommodityType.PALM_OIL,
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Plantation GPS", "Harvest date"]
    ),
    "12079190": CNCode(
        code="12079190",
        description="Palm nuts and kernels (other)",
        commodity_type=CommodityType.PALM_OIL,
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Plantation GPS", "Mill ID"]
    ),
    "15111000": CNCode(
        code="15111000",
        description="Crude palm oil",
        commodity_type=CommodityType.PALM_OIL,
        is_derived=True,
        derived_from=[CommodityType.PALM_OIL],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Mill ID", "Plantation source", "RSPO status"]
    ),
    "15119011": CNCode(
        code="15119011",
        description="Palm oil fractions, solid, not chemically modified",
        commodity_type=CommodityType.PALM_OIL,
        is_derived=True,
        derived_from=[CommodityType.PALM_OIL],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Refinery ID", "Batch number"]
    ),
    "15119019": CNCode(
        code="15119019",
        description="Palm oil fractions (other solid)",
        commodity_type=CommodityType.PALM_OIL,
        is_derived=True,
        derived_from=[CommodityType.PALM_OIL],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Refinery ID"]
    ),
    "15119091": CNCode(
        code="15119091",
        description="Palm oil fractions, liquid, for industrial use",
        commodity_type=CommodityType.PALM_OIL,
        is_derived=True,
        derived_from=[CommodityType.PALM_OIL],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Refinery ID", "End use declaration"]
    ),
    "15119099": CNCode(
        code="15119099",
        description="Palm oil fractions, liquid (other)",
        commodity_type=CommodityType.PALM_OIL,
        is_derived=True,
        derived_from=[CommodityType.PALM_OIL],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Refinery ID"]
    ),
    "15132110": CNCode(
        code="15132110",
        description="Crude palm kernel oil",
        commodity_type=CommodityType.PALM_OIL,
        is_derived=True,
        derived_from=[CommodityType.PALM_OIL],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Mill ID", "Plantation source"]
    ),
    "15132190": CNCode(
        code="15132190",
        description="Crude palm kernel oil (other)",
        commodity_type=CommodityType.PALM_OIL,
        is_derived=True,
        derived_from=[CommodityType.PALM_OIL],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Mill ID"]
    ),
    "15132911": CNCode(
        code="15132911",
        description="Palm kernel oil fractions, solid",
        commodity_type=CommodityType.PALM_OIL,
        is_derived=True,
        derived_from=[CommodityType.PALM_OIL],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Refinery ID"]
    ),
    "23066000": CNCode(
        code="23066000",
        description="Palm nut or kernel oilcake",
        commodity_type=CommodityType.PALM_OIL,
        is_derived=True,
        derived_from=[CommodityType.PALM_OIL],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Mill ID"]
    ),
    "29054500": CNCode(
        code="29054500",
        description="Glycerol (from palm oil processing)",
        commodity_type=CommodityType.PALM_OIL,
        is_derived=True,
        derived_from=[CommodityType.PALM_OIL],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Processing facility"],
        notes="Only when derived from palm oil"
    ),
    "38260010": CNCode(
        code="38260010",
        description="Biodiesel containing palm oil",
        commodity_type=CommodityType.PALM_OIL,
        is_derived=True,
        derived_from=[CommodityType.PALM_OIL],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Biorefinery", "Feedstock declaration"]
    ),

    # -------------------------------------------------------------------------
    # RUBBER (Chapter 40)
    # -------------------------------------------------------------------------
    "40011000": CNCode(
        code="40011000",
        description="Natural rubber latex, whether or not prevulcanised",
        commodity_type=CommodityType.RUBBER,
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Plantation GPS", "Collection date"]
    ),
    "40012100": CNCode(
        code="40012100",
        description="Natural rubber in smoked sheets",
        commodity_type=CommodityType.RUBBER,
        is_derived=True,
        derived_from=[CommodityType.RUBBER],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Processing facility", "Grade"]
    ),
    "40012200": CNCode(
        code="40012200",
        description="Technically specified natural rubber (TSNR)",
        commodity_type=CommodityType.RUBBER,
        is_derived=True,
        derived_from=[CommodityType.RUBBER],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Processing facility", "TSR grade"]
    ),
    "40012900": CNCode(
        code="40012900",
        description="Natural rubber in other forms",
        commodity_type=CommodityType.RUBBER,
        is_derived=True,
        derived_from=[CommodityType.RUBBER],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Processing facility"]
    ),
    "40021100": CNCode(
        code="40021100",
        description="Styrene-butadiene rubber (SBR) latex",
        commodity_type=CommodityType.RUBBER,
        is_derived=True,
        derived_from=[CommodityType.RUBBER],
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Manufacturer"],
        notes="Only if containing natural rubber"
    ),
    "40111000": CNCode(
        code="40111000",
        description="New pneumatic tyres for motor cars",
        commodity_type=CommodityType.RUBBER,
        is_derived=True,
        derived_from=[CommodityType.RUBBER],
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Manufacturer", "Natural rubber content %"]
    ),
    "40112010": CNCode(
        code="40112010",
        description="New pneumatic tyres for buses and lorries",
        commodity_type=CommodityType.RUBBER,
        is_derived=True,
        derived_from=[CommodityType.RUBBER],
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Manufacturer", "Natural rubber content %"]
    ),
    "40151100": CNCode(
        code="40151100",
        description="Surgical rubber gloves",
        commodity_type=CommodityType.RUBBER,
        is_derived=True,
        derived_from=[CommodityType.RUBBER],
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Manufacturer"]
    ),
    "40169300": CNCode(
        code="40169300",
        description="Rubber gaskets, washers and other seals",
        commodity_type=CommodityType.RUBBER,
        is_derived=True,
        derived_from=[CommodityType.RUBBER],
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Manufacturer"],
        notes="Only if containing natural rubber"
    ),

    # -------------------------------------------------------------------------
    # SOYA (Chapters 12, 15, 23)
    # -------------------------------------------------------------------------
    "12010010": CNCode(
        code="12010010",
        description="Soya beans for sowing",
        commodity_type=CommodityType.SOYA,
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Farm GPS", "Harvest date", "Variety"]
    ),
    "12010090": CNCode(
        code="12010090",
        description="Soya beans (other)",
        commodity_type=CommodityType.SOYA,
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Farm GPS", "Elevator/silo ID"]
    ),
    "12081000": CNCode(
        code="12081000",
        description="Soya bean flour and meal",
        commodity_type=CommodityType.SOYA,
        is_derived=True,
        derived_from=[CommodityType.SOYA],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Processing facility", "Bean origin"]
    ),
    "15071000": CNCode(
        code="15071000",
        description="Crude soya bean oil",
        commodity_type=CommodityType.SOYA,
        is_derived=True,
        derived_from=[CommodityType.SOYA],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Crusher ID", "Bean source"]
    ),
    "15079010": CNCode(
        code="15079010",
        description="Soya bean oil, refined, for technical use",
        commodity_type=CommodityType.SOYA,
        is_derived=True,
        derived_from=[CommodityType.SOYA],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Refinery ID"]
    ),
    "15079090": CNCode(
        code="15079090",
        description="Soya bean oil, refined (other)",
        commodity_type=CommodityType.SOYA,
        is_derived=True,
        derived_from=[CommodityType.SOYA],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Refinery ID", "Batch number"]
    ),
    "23040000": CNCode(
        code="23040000",
        description="Soya bean oilcake and other solid residues",
        commodity_type=CommodityType.SOYA,
        is_derived=True,
        derived_from=[CommodityType.SOYA],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Crusher ID", "Bean origin"]
    ),

    # -------------------------------------------------------------------------
    # WOOD (Chapters 44, 47, 48, 94)
    # -------------------------------------------------------------------------
    "44011100": CNCode(
        code="44011100",
        description="Fuel wood, in logs, billets, twigs",
        commodity_type=CommodityType.WOOD,
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Forest concession GPS", "Harvest permit"]
    ),
    "44012100": CNCode(
        code="44012100",
        description="Wood in chips, coniferous",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Forest origin", "Species"]
    ),
    "44012200": CNCode(
        code="44012200",
        description="Wood in chips, non-coniferous",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Forest origin", "Species"]
    ),
    "44013100": CNCode(
        code="44013100",
        description="Wood pellets",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Producer", "Feedstock declaration"]
    ),
    "44021000": CNCode(
        code="44021000",
        description="Wood charcoal of bamboo",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Producer", "Feedstock source"]
    ),
    "44032100": CNCode(
        code="44032100",
        description="Wood in the rough, pine, treated",
        commodity_type=CommodityType.WOOD,
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Forest concession", "Harvest permit", "Species"]
    ),
    "44032400": CNCode(
        code="44032400",
        description="Wood in the rough, spruce/fir",
        commodity_type=CommodityType.WOOD,
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Forest concession", "Harvest permit"]
    ),
    "44034100": CNCode(
        code="44034100",
        description="Dark red meranti, light red meranti, meranti bakau",
        commodity_type=CommodityType.WOOD,
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Forest concession GPS", "FLEGT license"]
    ),
    "44039700": CNCode(
        code="44039700",
        description="Wood in the rough, poplar",
        commodity_type=CommodityType.WOOD,
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Forest/plantation origin"]
    ),
    "44071100": CNCode(
        code="44071100",
        description="Sawn wood, pine, thickness >6mm",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Sawmill ID", "Log origin"]
    ),
    "44072100": CNCode(
        code="44072100",
        description="Sawn wood, mahogany",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Sawmill ID", "Forest GPS", "CITES permit"]
    ),
    "44072910": CNCode(
        code="44072910",
        description="Sawn wood, iroko",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Sawmill ID", "FLEGT license"]
    ),
    "44081010": CNCode(
        code="44081010",
        description="Veneer sheets, coniferous, thickness <=6mm",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Manufacturer", "Log origin"]
    ),
    "44101100": CNCode(
        code="44101100",
        description="Particle board, unworked",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Manufacturer", "Feedstock declaration"]
    ),
    "44111200": CNCode(
        code="44111200",
        description="Medium density fibreboard (MDF), thickness <=5mm",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Manufacturer"]
    ),
    "44121000": CNCode(
        code="44121000",
        description="Plywood, bamboo only",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Manufacturer", "Bamboo source"]
    ),
    "44123100": CNCode(
        code="44123100",
        description="Plywood with tropical wood outer ply",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.HIGH,
        traceability_requirements=["Manufacturer", "Species declaration"]
    ),
    "47010000": CNCode(
        code="47010000",
        description="Mechanical wood pulp",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Mill ID", "Wood source"]
    ),
    "47020000": CNCode(
        code="47020000",
        description="Chemical wood pulp, dissolving grades",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Mill ID", "Certification status"]
    ),
    "47031100": CNCode(
        code="47031100",
        description="Chemical wood pulp, soda/sulphate, coniferous",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Mill ID"]
    ),
    "48010000": CNCode(
        code="48010000",
        description="Newsprint in rolls or sheets",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Mill ID", "Fiber source declaration"]
    ),
    "48025500": CNCode(
        code="48025500",
        description="Uncoated paper, weight 40-150g/m2",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Mill ID"]
    ),
    "48101300": CNCode(
        code="48101300",
        description="Paper coated with kaolin, for printing",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Mill ID"]
    ),
    "48191000": CNCode(
        code="48191000",
        description="Cartons, boxes of corrugated paper",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Manufacturer", "Recycled content %"]
    ),
    "49011000": CNCode(
        code="49011000",
        description="Printed books, brochures, leaflets",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.LOW,
        traceability_requirements=["Publisher", "Paper source declaration"]
    ),
    "94016100": CNCode(
        code="94016100",
        description="Upholstered wooden frame seats",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Manufacturer", "Wood species"]
    ),
    "94033000": CNCode(
        code="94033000",
        description="Wooden office furniture",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Manufacturer", "Wood species declaration"]
    ),
    "94034000": CNCode(
        code="94034000",
        description="Wooden kitchen furniture",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Manufacturer", "Wood origin"]
    ),
    "94035000": CNCode(
        code="94035000",
        description="Wooden bedroom furniture",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Manufacturer", "Wood species"]
    ),
    "94036010": CNCode(
        code="94036010",
        description="Wooden furniture (other), for domestic use",
        commodity_type=CommodityType.WOOD,
        is_derived=True,
        derived_from=[CommodityType.WOOD],
        risk_category=RiskCategory.MEDIUM,
        traceability_requirements=["Manufacturer"]
    ),
}


# =============================================================================
# Lookup Functions
# =============================================================================

def get_commodity_by_cn_code(cn_code: str) -> Optional[CNCode]:
    """
    Look up EUDR commodity information by CN code.

    Args:
        cn_code: EU Combined Nomenclature code (4-8 digits)

    Returns:
        CNCode object if found, None otherwise
    """
    # Exact match first
    if cn_code in CN_CODE_DATABASE:
        return CN_CODE_DATABASE[cn_code]

    # Try prefix match (broader categories)
    for code_length in [6, 4]:
        prefix = cn_code[:code_length] if len(cn_code) >= code_length else cn_code
        for db_code, cn_info in CN_CODE_DATABASE.items():
            if db_code.startswith(prefix):
                return cn_info

    return None


def is_eudr_regulated(cn_code: str) -> bool:
    """
    Check if a CN code falls under EUDR regulation.

    Args:
        cn_code: EU Combined Nomenclature code

    Returns:
        True if regulated, False otherwise
    """
    cn_info = get_commodity_by_cn_code(cn_code)
    if cn_info is None:
        return False
    return cn_info.commodity_type != CommodityType.NOT_REGULATED


def get_commodity_type(cn_code: str) -> CommodityType:
    """
    Get the primary commodity type for a CN code.

    Args:
        cn_code: EU Combined Nomenclature code

    Returns:
        CommodityType enum value
    """
    cn_info = get_commodity_by_cn_code(cn_code)
    if cn_info is None:
        return CommodityType.NOT_REGULATED
    return cn_info.commodity_type


def get_traceability_requirements(cn_code: str) -> List[str]:
    """
    Get traceability requirements for a CN code.

    Args:
        cn_code: EU Combined Nomenclature code

    Returns:
        List of traceability requirement strings
    """
    cn_info = get_commodity_by_cn_code(cn_code)
    if cn_info is None:
        return []
    return cn_info.traceability_requirements


def get_risk_category(cn_code: str) -> RiskCategory:
    """
    Get default risk category for a CN code.

    Args:
        cn_code: EU Combined Nomenclature code

    Returns:
        RiskCategory enum value
    """
    cn_info = get_commodity_by_cn_code(cn_code)
    if cn_info is None:
        return RiskCategory.LOW
    return cn_info.risk_category


def get_all_regulated_cn_codes() -> List[str]:
    """
    Get all CN codes that are EUDR-regulated.

    Returns:
        List of CN code strings
    """
    return [code for code, info in CN_CODE_DATABASE.items()
            if info.commodity_type != CommodityType.NOT_REGULATED]


def get_cn_codes_by_commodity(commodity_type: CommodityType) -> List[str]:
    """
    Get all CN codes for a specific commodity type.

    Args:
        commodity_type: CommodityType enum value

    Returns:
        List of CN code strings
    """
    return [code for code, info in CN_CODE_DATABASE.items()
            if info.commodity_type == commodity_type]


def get_commodity_info(commodity_type: CommodityType) -> Optional[Commodity]:
    """
    Get detailed information about a commodity.

    Args:
        commodity_type: CommodityType enum value

    Returns:
        Commodity object if found, None otherwise
    """
    return EUDR_COMMODITIES.get(commodity_type)


def classify_cn_code(cn_code: str, product_description: str = "") -> Dict:
    """
    Classify a CN code for EUDR compliance.

    Args:
        cn_code: EU Combined Nomenclature code
        product_description: Optional product description

    Returns:
        Classification result dictionary
    """
    cn_info = get_commodity_by_cn_code(cn_code)

    if cn_info is None:
        return {
            "eudr_regulated": False,
            "commodity_type": CommodityType.NOT_REGULATED.value,
            "cn_code": cn_code,
            "cn_description": "Unknown product",
            "classification_uri": f"cn://eudr/unclassified/{cn_code}",
            "risk_category": RiskCategory.LOW.value,
            "traceability_requirements": [],
            "derived_from": []
        }

    commodity_info = get_commodity_info(cn_info.commodity_type)

    return {
        "eudr_regulated": True,
        "commodity_type": cn_info.commodity_type.value,
        "cn_code": cn_info.code,
        "cn_description": cn_info.description,
        "classification_uri": f"cn://eudr/2023/{cn_info.code}",
        "risk_category": cn_info.risk_category.value,
        "traceability_requirements": cn_info.traceability_requirements,
        "derived_from": [d.value for d in cn_info.derived_from] if cn_info.is_derived else [],
        "is_derived_product": cn_info.is_derived,
        "key_producing_countries": commodity_info.key_producing_countries if commodity_info else [],
        "annex_reference": commodity_info.annex_reference if commodity_info else ""
    }


# =============================================================================
# Summary Statistics
# =============================================================================

def get_database_stats() -> Dict:
    """Get statistics about the commodity database."""
    commodity_counts = {}
    for commodity_type in CommodityType:
        if commodity_type not in [CommodityType.DERIVED_PRODUCT, CommodityType.NOT_REGULATED]:
            count = len(get_cn_codes_by_commodity(commodity_type))
            commodity_counts[commodity_type.value] = count

    return {
        "total_cn_codes": len(CN_CODE_DATABASE),
        "regulated_commodities": 7,
        "cn_codes_by_commodity": commodity_counts,
        "cutoff_date": EUDR_CUTOFF_DATE.isoformat(),
        "data_version": "2024-12",
        "source": "EU Combined Nomenclature 2024 + EUDR Annex I"
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "CommodityType",
    "RiskCategory",
    "TraceabilityLevel",
    # Constants
    "EUDR_CUTOFF_DATE",
    # Data classes
    "CNCode",
    "Commodity",
    # Databases
    "EUDR_COMMODITIES",
    "CN_CODE_DATABASE",
    # Functions
    "get_commodity_by_cn_code",
    "is_eudr_regulated",
    "get_commodity_type",
    "get_traceability_requirements",
    "get_risk_category",
    "get_all_regulated_cn_codes",
    "get_cn_codes_by_commodity",
    "get_commodity_info",
    "classify_cn_code",
    "get_database_stats",
]
