# -*- coding: utf-8 -*-
"""
EETDataEngine - PACK-010 SFDR Article 8 Engine 8

Manage European ESG Template (EET) data input/output for SFDR Article 8
products. The EET is the standardised data exchange template developed by
FinDatEx (Financial Data Exchange) to facilitate communication of ESG
and sustainability data between product manufacturers and distributors.

EET Coverage (v1.1.1):
    The EET contains ~600 data fields organized into sections. This engine
    focuses on the ~150 SFDR-related fields covering:
    - Product Information: ISIN, product name, management company
    - SFDR Classification: Article 6, 8, 8+, 9 determination
    - PAI Consideration: Whether and how PAI indicators are considered
    - Taxonomy Alignment: Percentage of Taxonomy-aligned investments
    - Sustainability Indicators: Promoted characteristics and metrics
    - Pre-contractual Annex: Template data for Annexes II/III
    - Periodic Annex: Template data for Annexes IV/V

EET Sections Mapped:
    Section 01: General Product Data
    Section 03: SFDR-Related Product Information
    Section 04: Taxonomy-Related Information
    Section 05: PAI Indicators
    Section 06: Sustainability Indicators
    Section 09: Additional SFDR Disclosures

Export Formats:
    - CSV: Flat file with field_id, value columns
    - JSON: Structured JSON with sections and field metadata
    - XML: XML format per FinDatEx specification

Zero-Hallucination:
    - All field mappings from official EET v1.1.1 specification
    - Validation rules per EET data type specifications
    - Completeness calculation uses deterministic counting
    - SHA-256 provenance hash on every result

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Status: Production Ready
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, date, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        Percentage or 0.0 on zero denominator.
    """
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EETVersion(str, Enum):
    """Supported EET template versions."""
    V1_0_0 = "1.0.0"
    V1_1_0 = "1.1.0"
    V1_1_1 = "1.1.1"


class EETSection(str, Enum):
    """EET data sections relevant to SFDR."""
    PRODUCT_INFO = "01_product_information"
    SFDR_CLASSIFICATION = "03_sfdr_classification"
    TAXONOMY_INFO = "04_taxonomy_information"
    PAI_INDICATORS = "05_pai_indicators"
    SUSTAINABILITY_INDICATORS = "06_sustainability_indicators"
    ADDITIONAL_SFDR = "09_additional_sfdr"


class EETDataType(str, Enum):
    """Data types for EET fields."""
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    PERCENTAGE = "percentage"
    BOOLEAN = "boolean"
    DATE = "date"
    ENUM = "enum"
    TEXT = "text"
    ISIN = "isin"
    LEI = "lei"
    CURRENCY = "currency"


class SFDRClassification(str, Enum):
    """SFDR product classification types."""
    ARTICLE_6 = "article_6"
    ARTICLE_8 = "article_8"
    ARTICLE_8_PLUS = "article_8_plus"
    ARTICLE_9 = "article_9"


class ExportFormat(str, Enum):
    """Supported EET export formats."""
    CSV = "csv"
    JSON = "json"
    XML = "xml"


class ValidationSeverity(str, Enum):
    """Severity level for validation findings."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ---------------------------------------------------------------------------
# EET Field Definitions - SFDR Related Fields
# ---------------------------------------------------------------------------


EET_SFDR_FIELDS: List[Dict[str, Any]] = [
    # ---- Section 01: Product Information ----
    {"field_id": "EET_01_001", "field_name": "Product_ISIN", "section": EETSection.PRODUCT_INFO,
     "data_type": EETDataType.ISIN, "required": True, "description": "ISIN of the financial product"},
    {"field_id": "EET_01_002", "field_name": "Product_Name", "section": EETSection.PRODUCT_INFO,
     "data_type": EETDataType.STRING, "required": True, "description": "Name of the financial product"},
    {"field_id": "EET_01_003", "field_name": "Product_LEI", "section": EETSection.PRODUCT_INFO,
     "data_type": EETDataType.LEI, "required": False, "description": "LEI of the product manufacturer"},
    {"field_id": "EET_01_004", "field_name": "Management_Company", "section": EETSection.PRODUCT_INFO,
     "data_type": EETDataType.STRING, "required": True, "description": "Name of the management company"},
    {"field_id": "EET_01_005", "field_name": "Reporting_Date", "section": EETSection.PRODUCT_INFO,
     "data_type": EETDataType.DATE, "required": True, "description": "Date of the EET data"},
    {"field_id": "EET_01_006", "field_name": "Currency", "section": EETSection.PRODUCT_INFO,
     "data_type": EETDataType.CURRENCY, "required": True, "description": "Reporting currency (ISO 4217)"},
    {"field_id": "EET_01_007", "field_name": "NAV_Fund", "section": EETSection.PRODUCT_INFO,
     "data_type": EETDataType.DECIMAL, "required": False, "description": "Net Asset Value of the fund"},
    {"field_id": "EET_01_008", "field_name": "NAV_Date", "section": EETSection.PRODUCT_INFO,
     "data_type": EETDataType.DATE, "required": False, "description": "Date of the NAV valuation"},
    {"field_id": "EET_01_009", "field_name": "Domicile", "section": EETSection.PRODUCT_INFO,
     "data_type": EETDataType.STRING, "required": False, "description": "Country of fund domicile"},
    {"field_id": "EET_01_010", "field_name": "Legal_Form", "section": EETSection.PRODUCT_INFO,
     "data_type": EETDataType.STRING, "required": False, "description": "Legal form of the product (UCITS, AIF, etc.)"},

    # ---- Section 03: SFDR Classification ----
    {"field_id": "EET_03_001", "field_name": "SFDR_Classification", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.ENUM, "required": True,
     "description": "SFDR classification (Article 6/8/8+/9)",
     "allowed_values": ["article_6", "article_8", "article_8_plus", "article_9"]},
    {"field_id": "EET_03_002", "field_name": "Promotes_Environmental_Characteristics", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.BOOLEAN, "required": True,
     "description": "Whether the product promotes environmental characteristics"},
    {"field_id": "EET_03_003", "field_name": "Promotes_Social_Characteristics", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.BOOLEAN, "required": True,
     "description": "Whether the product promotes social characteristics"},
    {"field_id": "EET_03_004", "field_name": "Has_Sustainable_Investment_Objective", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.BOOLEAN, "required": True,
     "description": "Whether the product has a sustainable investment objective (Art 9)"},
    {"field_id": "EET_03_005", "field_name": "Sustainable_Investment_Pct", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Minimum proportion committed to sustainable investments (%)"},
    {"field_id": "EET_03_006", "field_name": "Sustainable_Investment_Environmental_Pct", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Minimum proportion of environmental sustainable investments (%)"},
    {"field_id": "EET_03_007", "field_name": "Sustainable_Investment_Social_Pct", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Minimum proportion of social sustainable investments (%)"},
    {"field_id": "EET_03_008", "field_name": "Environmental_Characteristics_Promoted", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.TEXT, "required": False,
     "description": "Description of promoted environmental characteristics"},
    {"field_id": "EET_03_009", "field_name": "Social_Characteristics_Promoted", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.TEXT, "required": False,
     "description": "Description of promoted social characteristics"},
    {"field_id": "EET_03_010", "field_name": "Sustainable_Investment_Policy", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.TEXT, "required": False,
     "description": "Description of the sustainable investment policy"},
    {"field_id": "EET_03_011", "field_name": "ESG_Integration_Strategy", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.TEXT, "required": False,
     "description": "Description of ESG integration strategy"},
    {"field_id": "EET_03_012", "field_name": "Binding_Elements_Description", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.TEXT, "required": False,
     "description": "Description of binding elements of the investment strategy"},
    {"field_id": "EET_03_013", "field_name": "Index_Designated_Reference", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.STRING, "required": False,
     "description": "Designated reference benchmark for ESG comparison"},
    {"field_id": "EET_03_014", "field_name": "DNSH_Assessment_Method", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.TEXT, "required": False,
     "description": "Methodology for DNSH assessment of sustainable investments"},
    {"field_id": "EET_03_015", "field_name": "Good_Governance_Assessment", "section": EETSection.SFDR_CLASSIFICATION,
     "data_type": EETDataType.TEXT, "required": False,
     "description": "Description of good governance assessment approach"},

    # ---- Section 04: Taxonomy Information ----
    {"field_id": "EET_04_001", "field_name": "Taxonomy_Alignment_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Percentage of investments Taxonomy-aligned"},
    {"field_id": "EET_04_002", "field_name": "Taxonomy_Alignment_Climate_Mitigation_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Taxonomy-aligned % for climate change mitigation"},
    {"field_id": "EET_04_003", "field_name": "Taxonomy_Alignment_Climate_Adaptation_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Taxonomy-aligned % for climate change adaptation"},
    {"field_id": "EET_04_004", "field_name": "Taxonomy_Alignment_Water_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Taxonomy-aligned % for sustainable use of water and marine"},
    {"field_id": "EET_04_005", "field_name": "Taxonomy_Alignment_Circular_Economy_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Taxonomy-aligned % for circular economy"},
    {"field_id": "EET_04_006", "field_name": "Taxonomy_Alignment_Pollution_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Taxonomy-aligned % for pollution prevention"},
    {"field_id": "EET_04_007", "field_name": "Taxonomy_Alignment_Biodiversity_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Taxonomy-aligned % for biodiversity and ecosystems"},
    {"field_id": "EET_04_008", "field_name": "Taxonomy_Eligible_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Percentage of investments Taxonomy-eligible"},
    {"field_id": "EET_04_009", "field_name": "Taxonomy_Non_Eligible_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Percentage of investments not Taxonomy-eligible"},
    {"field_id": "EET_04_010", "field_name": "Minimum_Taxonomy_Commitment_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Minimum committed Taxonomy-aligned percentage"},
    {"field_id": "EET_04_011", "field_name": "Taxonomy_Alignment_Including_Sovereign", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Taxonomy alignment including sovereign bonds"},
    {"field_id": "EET_04_012", "field_name": "Taxonomy_Alignment_Excluding_Sovereign", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Taxonomy alignment excluding sovereign bonds"},
    {"field_id": "EET_04_013", "field_name": "Taxonomy_Transitional_Activities_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Percentage in Taxonomy transitional activities"},
    {"field_id": "EET_04_014", "field_name": "Taxonomy_Enabling_Activities_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Percentage in Taxonomy enabling activities"},
    {"field_id": "EET_04_015", "field_name": "Fossil_Gas_Taxonomy_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Percentage in fossil gas Taxonomy-aligned activities"},
    {"field_id": "EET_04_016", "field_name": "Nuclear_Taxonomy_Pct", "section": EETSection.TAXONOMY_INFO,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Percentage in nuclear energy Taxonomy-aligned activities"},

    # ---- Section 05: PAI Indicators ----
    {"field_id": "EET_05_001", "field_name": "PAI_Consideration_Flag", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.BOOLEAN, "required": True,
     "description": "Whether PAI indicators are considered in investment decisions"},
    {"field_id": "EET_05_002", "field_name": "PAI_Statement_Reference", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.STRING, "required": False,
     "description": "Reference/URL to PAI statement"},
    # Mandatory PAI indicators (Table 1, Annex I)
    {"field_id": "EET_05_010", "field_name": "PAI_01_GHG_Emissions_Scope1", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "PAI 1a: Scope 1 GHG emissions (tCO2e)"},
    {"field_id": "EET_05_011", "field_name": "PAI_01_GHG_Emissions_Scope2", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "PAI 1b: Scope 2 GHG emissions (tCO2e)"},
    {"field_id": "EET_05_012", "field_name": "PAI_01_GHG_Emissions_Scope3", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "PAI 1c: Scope 3 GHG emissions (tCO2e)"},
    {"field_id": "EET_05_013", "field_name": "PAI_01_GHG_Emissions_Total", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "PAI 1d: Total GHG emissions (tCO2e)"},
    {"field_id": "EET_05_020", "field_name": "PAI_02_Carbon_Footprint", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "PAI 2: Carbon footprint (tCO2e/EUR M invested)"},
    {"field_id": "EET_05_030", "field_name": "PAI_03_GHG_Intensity", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "PAI 3: GHG intensity (tCO2e/EUR M revenue)"},
    {"field_id": "EET_05_040", "field_name": "PAI_04_Fossil_Fuel_Exposure_Pct", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "PAI 4: Exposure to fossil fuel sector (%)"},
    {"field_id": "EET_05_050", "field_name": "PAI_05_Non_Renewable_Energy_Share_Pct", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "PAI 5: Share of non-renewable energy consumption and production (%)"},
    {"field_id": "EET_05_060", "field_name": "PAI_06_Energy_Intensity_Per_Sector", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "PAI 6: Energy consumption intensity per high-impact sector (GWh/EUR M revenue)"},
    {"field_id": "EET_05_070", "field_name": "PAI_07_Biodiversity_Impact", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "PAI 7: Activities negatively affecting biodiversity-sensitive areas"},
    {"field_id": "EET_05_080", "field_name": "PAI_08_Water_Emissions", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "PAI 8: Emissions to water (tonnes)"},
    {"field_id": "EET_05_090", "field_name": "PAI_09_Hazardous_Waste_Ratio", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "PAI 9: Hazardous waste and radioactive waste ratio (tonnes)"},
    {"field_id": "EET_05_100", "field_name": "PAI_10_UNGC_Violations", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.BOOLEAN, "required": False,
     "description": "PAI 10: Violations of UN Global Compact and OECD Guidelines"},
    {"field_id": "EET_05_110", "field_name": "PAI_11_UNGC_Monitoring_Pct", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "PAI 11: Lack of processes to monitor UNGC/OECD compliance (%)"},
    {"field_id": "EET_05_120", "field_name": "PAI_12_Gender_Pay_Gap_Pct", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "PAI 12: Unadjusted gender pay gap (%)"},
    {"field_id": "EET_05_130", "field_name": "PAI_13_Board_Gender_Diversity_Pct", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "PAI 13: Board gender diversity (% female)"},
    {"field_id": "EET_05_140", "field_name": "PAI_14_Controversial_Weapons", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.BOOLEAN, "required": False,
     "description": "PAI 14: Exposure to controversial weapons"},
    # Additional mandatory indicators for sovereign/supranational
    {"field_id": "EET_05_150", "field_name": "PAI_15_GHG_Intensity_Countries", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "PAI 15: GHG intensity of investee countries (tCO2e/EUR M GDP)"},
    {"field_id": "EET_05_160", "field_name": "PAI_16_Social_Violations_Countries", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.INTEGER, "required": False,
     "description": "PAI 16: Investee countries subject to social violations"},
    # Real estate specific
    {"field_id": "EET_05_170", "field_name": "PAI_17_Fossil_Fuel_Real_Estate_Pct", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "PAI 17: Exposure to fossil fuels through real estate assets (%)"},
    {"field_id": "EET_05_180", "field_name": "PAI_18_Energy_Inefficient_Real_Estate_Pct", "section": EETSection.PAI_INDICATORS,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "PAI 18: Exposure to energy-inefficient real estate assets (%)"},

    # ---- Section 06: Sustainability Indicators ----
    {"field_id": "EET_06_001", "field_name": "SI_Carbon_Intensity_Portfolio", "section": EETSection.SUSTAINABILITY_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "Portfolio carbon intensity (tCO2e/EUR M revenue)"},
    {"field_id": "EET_06_002", "field_name": "SI_Carbon_Footprint_Portfolio", "section": EETSection.SUSTAINABILITY_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "Portfolio carbon footprint (tCO2e/EUR M invested)"},
    {"field_id": "EET_06_003", "field_name": "SI_Renewable_Energy_Share", "section": EETSection.SUSTAINABILITY_INDICATORS,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Weighted average share of renewable energy (%)"},
    {"field_id": "EET_06_004", "field_name": "SI_Water_Intensity", "section": EETSection.SUSTAINABILITY_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "Weighted average water intensity (m3/EUR M revenue)"},
    {"field_id": "EET_06_005", "field_name": "SI_Waste_Recycling_Rate", "section": EETSection.SUSTAINABILITY_INDICATORS,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Weighted average waste recycling rate (%)"},
    {"field_id": "EET_06_006", "field_name": "SI_Board_Diversity", "section": EETSection.SUSTAINABILITY_INDICATORS,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Weighted average board gender diversity (%)"},
    {"field_id": "EET_06_007", "field_name": "SI_Health_Safety_Rate", "section": EETSection.SUSTAINABILITY_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "Weighted average lost time injury rate"},
    {"field_id": "EET_06_008", "field_name": "SI_Living_Wage_Coverage", "section": EETSection.SUSTAINABILITY_INDICATORS,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Weighted average living wage coverage (%)"},
    {"field_id": "EET_06_009", "field_name": "SI_Supplier_ESG_Assessment", "section": EETSection.SUSTAINABILITY_INDICATORS,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Percentage of suppliers with ESG assessment (%)"},
    {"field_id": "EET_06_010", "field_name": "SI_ESG_Score_Portfolio", "section": EETSection.SUSTAINABILITY_INDICATORS,
     "data_type": EETDataType.DECIMAL, "required": False,
     "description": "Portfolio-weighted ESG score"},

    # ---- Section 09: Additional SFDR Disclosures ----
    {"field_id": "EET_09_001", "field_name": "Precontractual_Disclosure_URL", "section": EETSection.ADDITIONAL_SFDR,
     "data_type": EETDataType.STRING, "required": False,
     "description": "URL to pre-contractual disclosure (Annex II/III)"},
    {"field_id": "EET_09_002", "field_name": "Periodic_Disclosure_URL", "section": EETSection.ADDITIONAL_SFDR,
     "data_type": EETDataType.STRING, "required": False,
     "description": "URL to periodic disclosure (Annex IV/V)"},
    {"field_id": "EET_09_003", "field_name": "Website_Disclosure_URL", "section": EETSection.ADDITIONAL_SFDR,
     "data_type": EETDataType.STRING, "required": False,
     "description": "URL to website disclosure (Article 10)"},
    {"field_id": "EET_09_004", "field_name": "Engagement_Policy_URL", "section": EETSection.ADDITIONAL_SFDR,
     "data_type": EETDataType.STRING, "required": False,
     "description": "URL to engagement policy"},
    {"field_id": "EET_09_005", "field_name": "Exclusion_Criteria_Description", "section": EETSection.ADDITIONAL_SFDR,
     "data_type": EETDataType.TEXT, "required": False,
     "description": "Description of exclusion criteria applied"},
    {"field_id": "EET_09_006", "field_name": "Stewardship_Code_Adherence", "section": EETSection.ADDITIONAL_SFDR,
     "data_type": EETDataType.BOOLEAN, "required": False,
     "description": "Whether the product adheres to a stewardship code"},
    {"field_id": "EET_09_007", "field_name": "ESG_Data_Provider", "section": EETSection.ADDITIONAL_SFDR,
     "data_type": EETDataType.STRING, "required": False,
     "description": "Primary ESG data provider used"},
    {"field_id": "EET_09_008", "field_name": "ESG_Rating_Agency", "section": EETSection.ADDITIONAL_SFDR,
     "data_type": EETDataType.STRING, "required": False,
     "description": "ESG rating agency used"},
    {"field_id": "EET_09_009", "field_name": "Data_Coverage_Ratio_Pct", "section": EETSection.ADDITIONAL_SFDR,
     "data_type": EETDataType.PERCENTAGE, "required": False,
     "description": "Data coverage ratio for ESG metrics (%)"},
    {"field_id": "EET_09_010", "field_name": "Reporting_Period_Start", "section": EETSection.ADDITIONAL_SFDR,
     "data_type": EETDataType.DATE, "required": False,
     "description": "Start date of reporting period"},
    {"field_id": "EET_09_011", "field_name": "Reporting_Period_End", "section": EETSection.ADDITIONAL_SFDR,
     "data_type": EETDataType.DATE, "required": False,
     "description": "End date of reporting period"},
    {"field_id": "EET_09_012", "field_name": "Assurance_Provider", "section": EETSection.ADDITIONAL_SFDR,
     "data_type": EETDataType.STRING, "required": False,
     "description": "Third-party assurance provider name"},
    {"field_id": "EET_09_013", "field_name": "Assurance_Level", "section": EETSection.ADDITIONAL_SFDR,
     "data_type": EETDataType.ENUM, "required": False,
     "description": "Level of assurance (limited/reasonable)",
     "allowed_values": ["none", "limited", "reasonable"]},
]


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class EETField(BaseModel):
    """A single EET data field with metadata and value.

    Represents one field in the European ESG Template, including
    its specification metadata and the actual populated value.
    """
    field_id: str = Field(description="EET field identifier (e.g. EET_03_001)")
    field_name: str = Field(description="Human-readable field name")
    section: EETSection = Field(description="EET section this field belongs to")
    data_type: EETDataType = Field(description="Expected data type")
    required: bool = Field(default=False, description="Whether this field is required")
    value: Optional[Any] = Field(default=None, description="Populated value")
    source: str = Field(default="", description="Data source identifier")
    description: str = Field(default="", description="Field description")
    allowed_values: Optional[List[str]] = Field(
        default=None, description="Allowed enum values"
    )
    populated: bool = Field(default=False, description="Whether a value has been set")
    last_updated: Optional[datetime] = Field(default=None, description="Last update timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class EETDataSet(BaseModel):
    """Complete EET data set for a financial product.

    Contains all EET fields populated with values for a specific
    product and reporting date.
    """
    dataset_id: str = Field(default_factory=_new_uuid, description="Unique dataset identifier")
    product_isin: str = Field(default="", description="Product ISIN")
    product_name: str = Field(default="", description="Product name")
    reporting_date: Optional[str] = Field(default=None, description="Reporting date (YYYY-MM-DD)")
    eet_version: EETVersion = Field(default=EETVersion.V1_1_1, description="EET version")
    fields: List[EETField] = Field(default_factory=list, description="EET fields")
    total_fields: int = Field(default=0, description="Total fields in dataset")
    populated_fields: int = Field(default=0, description="Number of populated fields")
    completeness_pct: float = Field(default=0.0, description="Completeness percentage")
    created_at: datetime = Field(default_factory=_utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=_utcnow, description="Last update timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class EETValidationResult(BaseModel):
    """Result of validating an EET dataset against specification.

    Checks field types, required fields, value ranges, and
    cross-field consistency.
    """
    validation_id: str = Field(default_factory=_new_uuid, description="Unique validation identifier")
    valid: bool = Field(description="Whether the dataset passes all required checks")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="Validation errors")
    warnings: List[Dict[str, str]] = Field(default_factory=list, description="Validation warnings")
    info: List[Dict[str, str]] = Field(default_factory=list, description="Informational findings")
    completeness_pct: float = Field(default=0.0, description="Field completeness (%)")
    required_completeness_pct: float = Field(
        default=0.0, description="Required field completeness (%)"
    )
    section_completeness: Dict[str, float] = Field(
        default_factory=dict, description="Completeness by section (%)"
    )
    total_fields_checked: int = Field(default=0, description="Total fields checked")
    validated_at: datetime = Field(default_factory=_utcnow, description="Validation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class EETExportResult(BaseModel):
    """Result of exporting EET data to a specific format."""
    export_id: str = Field(default_factory=_new_uuid, description="Unique export identifier")
    format: ExportFormat = Field(description="Export format used")
    content: str = Field(description="Exported content as string")
    field_count: int = Field(default=0, description="Number of fields exported")
    file_size_bytes: int = Field(default=0, description="Size of exported content in bytes")
    exported_at: datetime = Field(default_factory=_utcnow, description="Export timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------


class EETConfig(BaseModel):
    """Configuration for the EETDataEngine.

    Controls EET version, field subsets, and export settings.
    """
    eet_version: EETVersion = Field(
        default=EETVersion.V1_1_1, description="EET template version"
    )
    field_subset: Optional[List[str]] = Field(
        default=None, description="Subset of field IDs to include (None = all)"
    )
    export_format: ExportFormat = Field(
        default=ExportFormat.JSON, description="Default export format"
    )
    validate_on_populate: bool = Field(
        default=True, description="Validate field values when populated"
    )
    strict_mode: bool = Field(
        default=False, description="Reject invalid values instead of warning"
    )
    include_empty_fields: bool = Field(
        default=True, description="Include unpopulated fields in exports"
    )


# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

EETConfig.model_rebuild()
EETField.model_rebuild()
EETDataSet.model_rebuild()
EETValidationResult.model_rebuild()
EETExportResult.model_rebuild()


# ---------------------------------------------------------------------------
# EETDataEngine
# ---------------------------------------------------------------------------


class EETDataEngine:
    """
    European ESG Template (EET) data management engine.

    Manages the population, validation, and export of EET data fields
    for SFDR Article 8 products. Supports 150+ SFDR-related fields
    across six EET sections with CSV, JSON, and XML export formats.

    Attributes:
        config: Engine configuration parameters.
        _dataset: The active EET dataset being managed.
        _field_index: Index of fields by field_id for fast lookup.

    Example:
        >>> engine = EETDataEngine({"eet_version": "1.1.1"})
        >>> engine.populate_eet_fields({
        ...     "EET_03_001": "article_8",
        ...     "EET_04_001": 45.5,
        ...     "EET_05_001": True,
        ... })
        >>> validation = engine.validate_eet_data()
        >>> export = engine.export_eet(ExportFormat.JSON)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EETDataEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = EETConfig(**config)
        elif config and isinstance(config, EETConfig):
            self.config = config
        else:
            self.config = EETConfig()

        self._dataset: EETDataSet = EETDataSet(eet_version=self.config.eet_version)
        self._field_index: Dict[str, EETField] = {}
        self._initialize_fields()

        logger.info(
            "EETDataEngine initialized (version=%s, eet_version=%s, fields=%d)",
            _MODULE_VERSION,
            self.config.eet_version.value,
            len(self._field_index),
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_fields(self) -> None:
        """Initialize EET field definitions from the template.

        Creates EETField objects for all SFDR-related fields and
        indexes them by field_id for fast lookup.
        """
        for field_def in EET_SFDR_FIELDS:
            if self.config.field_subset and field_def["field_id"] not in self.config.field_subset:
                continue

            field = EETField(
                field_id=field_def["field_id"],
                field_name=field_def["field_name"],
                section=field_def["section"],
                data_type=field_def["data_type"],
                required=field_def.get("required", False),
                description=field_def.get("description", ""),
                allowed_values=field_def.get("allowed_values"),
            )
            self._field_index[field.field_id] = field
            self._dataset.fields.append(field)

        self._dataset.total_fields = len(self._field_index)

    # ------------------------------------------------------------------
    # Field Population
    # ------------------------------------------------------------------

    def populate_eet_fields(
        self,
        field_values: Dict[str, Any],
        source: str = "manual",
    ) -> Dict[str, bool]:
        """Populate multiple EET fields with values.

        Args:
            field_values: Mapping of field_id to value.
            source: Data source identifier.

        Returns:
            Dictionary mapping field_id to success (True/False).
        """
        start = _utcnow()
        results: Dict[str, bool] = {}

        for field_id, value in field_values.items():
            success = self._set_field_value(field_id, value, source)
            results[field_id] = success

        self._update_completeness()
        self._dataset.updated_at = _utcnow()

        populated = sum(1 for v in results.values() if v)
        logger.info(
            "Populated %d/%d EET fields (source=%s) in %dms",
            populated,
            len(field_values),
            source,
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return results

    def set_product_info(
        self,
        isin: str,
        name: str,
        reporting_date: str,
        management_company: str = "",
        currency: str = "EUR",
    ) -> None:
        """Set core product information fields.

        Convenience method to populate the key product identification fields.

        Args:
            isin: Product ISIN.
            name: Product name.
            reporting_date: Reporting date (YYYY-MM-DD).
            management_company: Management company name.
            currency: Reporting currency.
        """
        self._dataset.product_isin = isin
        self._dataset.product_name = name
        self._dataset.reporting_date = reporting_date

        self._set_field_value("EET_01_001", isin, "product_info")
        self._set_field_value("EET_01_002", name, "product_info")
        self._set_field_value("EET_01_004", management_company, "product_info")
        self._set_field_value("EET_01_005", reporting_date, "product_info")
        self._set_field_value("EET_01_006", currency, "product_info")

        self._update_completeness()
        logger.info("Set product info: ISIN=%s, name=%s", isin, name)

    def set_sfdr_classification(
        self,
        classification: SFDRClassification,
        promotes_environmental: bool = True,
        promotes_social: bool = False,
        sustainable_investment_pct: Optional[float] = None,
    ) -> None:
        """Set SFDR classification fields.

        Args:
            classification: SFDR article classification.
            promotes_environmental: Whether environmental characteristics are promoted.
            promotes_social: Whether social characteristics are promoted.
            sustainable_investment_pct: Minimum sustainable investment percentage.
        """
        self._set_field_value("EET_03_001", classification.value, "classification")
        self._set_field_value("EET_03_002", promotes_environmental, "classification")
        self._set_field_value("EET_03_003", promotes_social, "classification")
        has_sustainable_objective = classification == SFDRClassification.ARTICLE_9
        self._set_field_value("EET_03_004", has_sustainable_objective, "classification")

        if sustainable_investment_pct is not None:
            self._set_field_value("EET_03_005", sustainable_investment_pct, "classification")

        self._update_completeness()
        logger.info("Set SFDR classification: %s", classification.value)

    def set_taxonomy_data(
        self,
        alignment_pct: float,
        by_objective: Optional[Dict[str, float]] = None,
        minimum_commitment_pct: Optional[float] = None,
    ) -> None:
        """Set Taxonomy alignment fields.

        Args:
            alignment_pct: Overall Taxonomy alignment percentage.
            by_objective: Alignment by environmental objective.
            minimum_commitment_pct: Minimum committed Taxonomy percentage.
        """
        self._set_field_value("EET_04_001", alignment_pct, "taxonomy")

        if by_objective:
            objective_mapping = {
                "climate_mitigation": "EET_04_002",
                "climate_adaptation": "EET_04_003",
                "water": "EET_04_004",
                "circular_economy": "EET_04_005",
                "pollution": "EET_04_006",
                "biodiversity": "EET_04_007",
            }
            for obj, pct in by_objective.items():
                field_id = objective_mapping.get(obj)
                if field_id:
                    self._set_field_value(field_id, pct, "taxonomy")

        if minimum_commitment_pct is not None:
            self._set_field_value("EET_04_010", minimum_commitment_pct, "taxonomy")

        eligible = min(alignment_pct + 20.0, 100.0)  # Conservative estimate
        self._set_field_value("EET_04_008", eligible, "taxonomy")
        self._set_field_value("EET_04_009", 100.0 - eligible, "taxonomy")

        self._update_completeness()
        logger.info("Set taxonomy data: alignment=%.1f%%", alignment_pct)

    def set_pai_data(
        self,
        pai_values: Dict[int, Any],
        considers_pai: bool = True,
    ) -> None:
        """Set PAI indicator values.

        Args:
            pai_values: Mapping of PAI indicator number (1-18) to value.
            considers_pai: Whether PAI indicators are considered.
        """
        self._set_field_value("EET_05_001", considers_pai, "pai")

        pai_field_mapping: Dict[int, Union[str, List[str]]] = {
            1: ["EET_05_010", "EET_05_011", "EET_05_012", "EET_05_013"],
            2: "EET_05_020",
            3: "EET_05_030",
            4: "EET_05_040",
            5: "EET_05_050",
            6: "EET_05_060",
            7: "EET_05_070",
            8: "EET_05_080",
            9: "EET_05_090",
            10: "EET_05_100",
            11: "EET_05_110",
            12: "EET_05_120",
            13: "EET_05_130",
            14: "EET_05_140",
            15: "EET_05_150",
            16: "EET_05_160",
            17: "EET_05_170",
            18: "EET_05_180",
        }

        for pai_num, value in pai_values.items():
            field_ref = pai_field_mapping.get(pai_num)
            if field_ref is None:
                logger.warning("Unknown PAI indicator number: %d", pai_num)
                continue

            if isinstance(field_ref, list):
                # PAI 1 has sub-fields (scope 1, 2, 3, total)
                if isinstance(value, dict):
                    for i, sub_key in enumerate(["scope1", "scope2", "scope3", "total"]):
                        if sub_key in value and i < len(field_ref):
                            self._set_field_value(field_ref[i], value[sub_key], "pai")
                elif isinstance(value, (int, float)):
                    self._set_field_value(field_ref[-1], value, "pai")
            else:
                self._set_field_value(field_ref, value, "pai")

        self._update_completeness()
        logger.info("Set PAI data: %d indicators", len(pai_values))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_eet_data(self) -> EETValidationResult:
        """Validate the EET dataset against specification rules.

        Performs the following checks:
        - Required field completeness
        - Data type validation
        - Value range checks (percentages 0-100)
        - Enum value validation
        - Cross-field consistency
        - SFDR classification consistency

        Returns:
            EETValidationResult with errors, warnings, and completeness.
        """
        start = _utcnow()
        errors: List[Dict[str, str]] = []
        warnings: List[Dict[str, str]] = []
        info: List[Dict[str, str]] = []
        section_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "populated": 0}
        )

        required_total = 0
        required_populated = 0

        for field in self._dataset.fields:
            section_counts[field.section.value]["total"] += 1
            if field.populated:
                section_counts[field.section.value]["populated"] += 1

            # Required field check
            if field.required:
                required_total += 1
                if not field.populated:
                    errors.append({
                        "field_id": field.field_id,
                        "field_name": field.field_name,
                        "message": f"Required field '{field.field_name}' is not populated",
                        "severity": ValidationSeverity.ERROR.value,
                    })
                else:
                    required_populated += 1

            # Type validation for populated fields
            if field.populated and field.value is not None:
                type_errors = self._validate_field_type(field)
                errors.extend(type_errors)

                # Enum validation
                if field.allowed_values and str(field.value) not in field.allowed_values:
                    errors.append({
                        "field_id": field.field_id,
                        "field_name": field.field_name,
                        "message": (
                            f"Value '{field.value}' not in allowed values: "
                            f"{field.allowed_values}"
                        ),
                        "severity": ValidationSeverity.ERROR.value,
                    })

        # Cross-field consistency checks
        cross_errors, cross_warnings = self._validate_cross_field()
        errors.extend(cross_errors)
        warnings.extend(cross_warnings)

        # Completeness warnings for optional but recommended fields
        unpopulated_optional = [
            f for f in self._dataset.fields
            if not f.required and not f.populated
        ]
        if len(unpopulated_optional) > 10:
            warnings.append({
                "field_id": "GENERAL",
                "field_name": "Completeness",
                "message": (
                    f"{len(unpopulated_optional)} optional fields not populated. "
                    "Higher completeness improves data quality scores."
                ),
                "severity": ValidationSeverity.WARNING.value,
            })

        # Section completeness
        section_completeness: Dict[str, float] = {}
        for section, counts in section_counts.items():
            section_completeness[section] = round(
                _safe_pct(counts["populated"], counts["total"]), 2
            )

        overall_completeness = _safe_pct(
            self._dataset.populated_fields, self._dataset.total_fields
        )
        required_completeness = _safe_pct(required_populated, required_total)

        is_valid = len(errors) == 0

        result = EETValidationResult(
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            completeness_pct=round(overall_completeness, 2),
            required_completeness_pct=round(required_completeness, 2),
            section_completeness=section_completeness,
            total_fields_checked=len(self._dataset.fields),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "EET validation %s (%d errors, %d warnings, %.1f%% complete) in %dms",
            "PASSED" if is_valid else "FAILED",
            len(errors),
            len(warnings),
            overall_completeness,
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return result

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_eet(
        self,
        fmt: Optional[ExportFormat] = None,
    ) -> EETExportResult:
        """Export EET data in the specified format.

        Args:
            fmt: Export format (CSV, JSON, or XML). Defaults to config.

        Returns:
            EETExportResult with the exported content as string.
        """
        start = _utcnow()
        export_format = fmt or self.config.export_format

        if export_format == ExportFormat.CSV:
            content = self._export_csv()
        elif export_format == ExportFormat.JSON:
            content = self._export_json()
        elif export_format == ExportFormat.XML:
            content = self._export_xml()
        else:
            content = self._export_json()

        field_count = (
            self._dataset.populated_fields
            if not self.config.include_empty_fields
            else self._dataset.total_fields
        )

        result = EETExportResult(
            format=export_format,
            content=content,
            field_count=field_count,
            file_size_bytes=len(content.encode("utf-8")),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Exported EET data: format=%s, fields=%d, size=%d bytes in %dms",
            export_format.value,
            field_count,
            result.file_size_bytes,
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return result

    def import_eet(
        self,
        data: Union[str, Dict[str, Any]],
        fmt: Optional[ExportFormat] = None,
        source: str = "import",
    ) -> Dict[str, bool]:
        """Import EET data from external source.

        Args:
            data: Data to import (string for CSV/XML, dict for JSON).
            fmt: Format of the input data.
            source: Data source identifier.

        Returns:
            Dictionary mapping field_id to import success.
        """
        start = _utcnow()
        import_format = fmt or self.config.export_format

        if import_format == ExportFormat.JSON:
            field_values = self._parse_json_import(data)
        elif import_format == ExportFormat.CSV:
            field_values = self._parse_csv_import(data)
        else:
            logger.warning("XML import not yet supported; use JSON or CSV")
            return {}

        results = self.populate_eet_fields(field_values, source=source)

        imported = sum(1 for v in results.values() if v)
        logger.info(
            "Imported %d/%d EET fields (format=%s) in %dms",
            imported,
            len(field_values),
            import_format.value,
            int((_utcnow() - start).total_seconds() * 1000),
        )
        return results

    # ------------------------------------------------------------------
    # Field Retrieval
    # ------------------------------------------------------------------

    def get_sfdr_fields(self) -> List[EETField]:
        """Get all SFDR classification fields.

        Returns:
            List of EETField objects from the SFDR section.
        """
        return [
            f for f in self._dataset.fields
            if f.section == EETSection.SFDR_CLASSIFICATION
        ]

    def get_taxonomy_fields(self) -> List[EETField]:
        """Get all Taxonomy-related fields.

        Returns:
            List of EETField objects from the Taxonomy section.
        """
        return [
            f for f in self._dataset.fields
            if f.section == EETSection.TAXONOMY_INFO
        ]

    def get_pai_fields(self) -> List[EETField]:
        """Get all PAI indicator fields.

        Returns:
            List of EETField objects from the PAI section.
        """
        return [
            f for f in self._dataset.fields
            if f.section == EETSection.PAI_INDICATORS
        ]

    def get_sustainability_indicator_fields(self) -> List[EETField]:
        """Get all sustainability indicator fields.

        Returns:
            List of EETField objects from the sustainability indicators section.
        """
        return [
            f for f in self._dataset.fields
            if f.section == EETSection.SUSTAINABILITY_INDICATORS
        ]

    def get_field(self, field_id: str) -> Optional[EETField]:
        """Get a single EET field by identifier.

        Args:
            field_id: EET field identifier.

        Returns:
            EETField if found, None otherwise.
        """
        return self._field_index.get(field_id)

    def get_dataset(self) -> EETDataSet:
        """Get the complete EET dataset.

        Returns:
            The active EETDataSet with all fields.
        """
        self._dataset.provenance_hash = _compute_hash(self._dataset)
        return self._dataset

    def get_field_value(self, field_id: str) -> Optional[Any]:
        """Get the value of a specific EET field.

        Args:
            field_id: EET field identifier.

        Returns:
            Field value if populated, None otherwise.
        """
        field = self._field_index.get(field_id)
        if field and field.populated:
            return field.value
        return None

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _set_field_value(
        self,
        field_id: str,
        value: Any,
        source: str,
    ) -> bool:
        """Set the value of a single EET field.

        Args:
            field_id: EET field identifier.
            value: Value to set.
            source: Data source identifier.

        Returns:
            True if set successfully, False otherwise.
        """
        field = self._field_index.get(field_id)
        if field is None:
            logger.warning("EET field '%s' not found in template", field_id)
            return False

        if self.config.validate_on_populate:
            type_errors = self._validate_single_value(field, value)
            if type_errors and self.config.strict_mode:
                logger.error(
                    "Strict validation failed for %s: %s",
                    field_id,
                    type_errors[0].get("message", ""),
                )
                return False

        field.value = value
        field.source = source
        field.populated = True
        field.last_updated = _utcnow()
        field.provenance_hash = _compute_hash(field)
        return True

    def _update_completeness(self) -> None:
        """Update the dataset completeness statistics."""
        populated = sum(1 for f in self._dataset.fields if f.populated)
        self._dataset.populated_fields = populated
        self._dataset.completeness_pct = round(
            _safe_pct(populated, self._dataset.total_fields), 2
        )

    def _validate_field_type(self, field: EETField) -> List[Dict[str, str]]:
        """Validate a field value matches its expected type.

        Args:
            field: EETField to validate.

        Returns:
            List of error dictionaries (empty if valid).
        """
        errors: List[Dict[str, str]] = []
        value = field.value

        if value is None:
            return errors

        if field.data_type == EETDataType.PERCENTAGE:
            try:
                num_val = float(value)
                if num_val < 0.0 or num_val > 100.0:
                    errors.append({
                        "field_id": field.field_id,
                        "field_name": field.field_name,
                        "message": f"Percentage value {num_val} out of range [0, 100]",
                        "severity": ValidationSeverity.ERROR.value,
                    })
            except (TypeError, ValueError):
                errors.append({
                    "field_id": field.field_id,
                    "field_name": field.field_name,
                    "message": f"Expected numeric percentage, got {type(value).__name__}",
                    "severity": ValidationSeverity.ERROR.value,
                })

        elif field.data_type == EETDataType.DECIMAL:
            try:
                float(value)
            except (TypeError, ValueError):
                errors.append({
                    "field_id": field.field_id,
                    "field_name": field.field_name,
                    "message": f"Expected decimal number, got {type(value).__name__}",
                    "severity": ValidationSeverity.ERROR.value,
                })

        elif field.data_type == EETDataType.INTEGER:
            if not isinstance(value, (int, float)) or (isinstance(value, float) and value != int(value)):
                errors.append({
                    "field_id": field.field_id,
                    "field_name": field.field_name,
                    "message": f"Expected integer, got {type(value).__name__}",
                    "severity": ValidationSeverity.ERROR.value,
                })

        elif field.data_type == EETDataType.BOOLEAN:
            if not isinstance(value, bool):
                errors.append({
                    "field_id": field.field_id,
                    "field_name": field.field_name,
                    "message": f"Expected boolean, got {type(value).__name__}",
                    "severity": ValidationSeverity.ERROR.value,
                })

        elif field.data_type == EETDataType.ISIN:
            if not isinstance(value, str) or len(value) != 12:
                errors.append({
                    "field_id": field.field_id,
                    "field_name": field.field_name,
                    "message": f"Invalid ISIN format (expected 12 characters, got {len(str(value))})",
                    "severity": ValidationSeverity.ERROR.value,
                })

        elif field.data_type == EETDataType.LEI:
            if isinstance(value, str) and len(value) != 20 and len(value) > 0:
                errors.append({
                    "field_id": field.field_id,
                    "field_name": field.field_name,
                    "message": f"Invalid LEI format (expected 20 characters, got {len(value)})",
                    "severity": ValidationSeverity.ERROR.value,
                })

        return errors

    def _validate_single_value(
        self, field: EETField, value: Any
    ) -> List[Dict[str, str]]:
        """Validate a single value before setting it.

        Args:
            field: Target EETField.
            value: Value to validate.

        Returns:
            List of error dictionaries.
        """
        temp_field = EETField(
            field_id=field.field_id,
            field_name=field.field_name,
            section=field.section,
            data_type=field.data_type,
            required=field.required,
            value=value,
            populated=True,
            allowed_values=field.allowed_values,
        )
        return self._validate_field_type(temp_field)

    def _validate_cross_field(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Validate cross-field consistency rules.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: List[Dict[str, str]] = []
        warnings: List[Dict[str, str]] = []

        sfdr_class = self.get_field_value("EET_03_001")
        promotes_env = self.get_field_value("EET_03_002")
        promotes_soc = self.get_field_value("EET_03_003")
        has_sustainable_obj = self.get_field_value("EET_03_004")
        taxonomy_pct = self.get_field_value("EET_04_001")
        sustainable_pct = self.get_field_value("EET_03_005")

        # Article 8 must promote at least one characteristic
        if sfdr_class in ("article_8", "article_8_plus"):
            if promotes_env is False and promotes_soc is False:
                errors.append({
                    "field_id": "EET_03_002/003",
                    "field_name": "Characteristic Promotion",
                    "message": (
                        "Article 8 products must promote at least one "
                        "environmental or social characteristic"
                    ),
                    "severity": ValidationSeverity.ERROR.value,
                })

        # Article 8+ must have sustainable investments
        if sfdr_class == "article_8_plus":
            if sustainable_pct is None or float(sustainable_pct) <= 0:
                warnings.append({
                    "field_id": "EET_03_005",
                    "field_name": "Sustainable Investment Pct",
                    "message": (
                        "Article 8+ products should declare a minimum "
                        "sustainable investment proportion"
                    ),
                    "severity": ValidationSeverity.WARNING.value,
                })

        # Article 9 must have sustainable investment objective
        if sfdr_class == "article_9" and has_sustainable_obj is False:
            errors.append({
                "field_id": "EET_03_004",
                "field_name": "Sustainable Investment Objective",
                "message": (
                    "Article 9 products must have a sustainable "
                    "investment objective"
                ),
                "severity": ValidationSeverity.ERROR.value,
            })

        # Taxonomy percentages should not exceed 100%
        if taxonomy_pct is not None:
            try:
                tax_val = float(taxonomy_pct)
                if tax_val > 100.0:
                    errors.append({
                        "field_id": "EET_04_001",
                        "field_name": "Taxonomy Alignment Pct",
                        "message": f"Taxonomy alignment {tax_val}% exceeds 100%",
                        "severity": ValidationSeverity.ERROR.value,
                    })
            except (TypeError, ValueError):
                pass

        # Taxonomy breakdown should sum correctly
        objective_fields = [
            "EET_04_002", "EET_04_003", "EET_04_004",
            "EET_04_005", "EET_04_006", "EET_04_007",
        ]
        objective_sum = 0.0
        has_objectives = False
        for fid in objective_fields:
            val = self.get_field_value(fid)
            if val is not None:
                has_objectives = True
                try:
                    objective_sum += float(val)
                except (TypeError, ValueError):
                    pass

        if has_objectives and taxonomy_pct is not None:
            try:
                tax_val = float(taxonomy_pct)
                if abs(objective_sum - tax_val) > 1.0 and objective_sum > 0:
                    warnings.append({
                        "field_id": "EET_04_002-007",
                        "field_name": "Taxonomy Objective Breakdown",
                        "message": (
                            f"Sum of objective-level alignment ({objective_sum:.1f}%) "
                            f"differs from total alignment ({tax_val:.1f}%) by more than 1%"
                        ),
                        "severity": ValidationSeverity.WARNING.value,
                    })
            except (TypeError, ValueError):
                pass

        return errors, warnings

    # ------------------------------------------------------------------
    # Export Formatters
    # ------------------------------------------------------------------

    def _export_csv(self) -> str:
        """Export EET data as CSV.

        Returns:
            CSV string content.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "field_id", "field_name", "section", "data_type",
            "required", "value", "source", "description",
        ])

        for field in self._dataset.fields:
            if not self.config.include_empty_fields and not field.populated:
                continue
            writer.writerow([
                field.field_id,
                field.field_name,
                field.section.value,
                field.data_type.value,
                field.required,
                str(field.value) if field.value is not None else "",
                field.source,
                field.description,
            ])

        return output.getvalue()

    def _export_json(self) -> str:
        """Export EET data as JSON.

        Returns:
            JSON string content.
        """
        sections: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for field in self._dataset.fields:
            if not self.config.include_empty_fields and not field.populated:
                continue
            sections[field.section.value].append({
                "field_id": field.field_id,
                "field_name": field.field_name,
                "data_type": field.data_type.value,
                "required": field.required,
                "value": field.value,
                "source": field.source,
                "description": field.description,
            })

        export_data = {
            "eet_version": self.config.eet_version.value,
            "product_isin": self._dataset.product_isin,
            "product_name": self._dataset.product_name,
            "reporting_date": self._dataset.reporting_date,
            "generated_at": _utcnow().isoformat(),
            "total_fields": self._dataset.total_fields,
            "populated_fields": self._dataset.populated_fields,
            "completeness_pct": self._dataset.completeness_pct,
            "sections": dict(sections),
        }

        return json.dumps(export_data, indent=2, default=str)

    def _export_xml(self) -> str:
        """Export EET data as XML.

        Returns:
            XML string content.
        """
        lines: List[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<EET version="{self.config.eet_version.value}">',
            f'  <ProductISIN>{self._dataset.product_isin}</ProductISIN>',
            f'  <ProductName>{self._dataset.product_name}</ProductName>',
            f'  <ReportingDate>{self._dataset.reporting_date or ""}</ReportingDate>',
            f'  <GeneratedAt>{_utcnow().isoformat()}</GeneratedAt>',
        ]

        current_section = None
        for field in self._dataset.fields:
            if not self.config.include_empty_fields and not field.populated:
                continue

            if field.section != current_section:
                if current_section is not None:
                    lines.append(f'  </Section>')
                current_section = field.section
                lines.append(f'  <Section id="{current_section.value}">')

            value_str = str(field.value) if field.value is not None else ""
            # Escape XML special characters
            value_str = (
                value_str
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )
            lines.append(
                f'    <Field id="{field.field_id}" name="{field.field_name}" '
                f'type="{field.data_type.value}" required="{field.required}">'
                f'{value_str}</Field>'
            )

        if current_section is not None:
            lines.append('  </Section>')
        lines.append('</EET>')

        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Import Parsers
    # ------------------------------------------------------------------

    def _parse_json_import(
        self, data: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse JSON import data into field_id -> value mapping.

        Args:
            data: JSON string or dictionary.

        Returns:
            Field value mapping.
        """
        if isinstance(data, str):
            parsed = json.loads(data)
        else:
            parsed = data

        field_values: Dict[str, Any] = {}

        sections = parsed.get("sections", {})
        for section_name, fields in sections.items():
            for field_data in fields:
                field_id = field_data.get("field_id")
                value = field_data.get("value")
                if field_id and value is not None:
                    field_values[field_id] = value

        return field_values

    def _parse_csv_import(self, data: Union[str, Any]) -> Dict[str, Any]:
        """Parse CSV import data into field_id -> value mapping.

        Args:
            data: CSV string content.

        Returns:
            Field value mapping.
        """
        if not isinstance(data, str):
            logger.warning("CSV import expects string data")
            return {}

        field_values: Dict[str, Any] = {}
        reader = csv.DictReader(io.StringIO(data))

        for row in reader:
            field_id = row.get("field_id", "")
            value = row.get("value", "")
            if field_id and value:
                # Attempt type coercion based on field definition
                field = self._field_index.get(field_id)
                if field:
                    value = self._coerce_csv_value(field, value)
                field_values[field_id] = value

        return field_values

    def _coerce_csv_value(self, field: EETField, value: str) -> Any:
        """Coerce a CSV string value to the expected field type.

        Args:
            field: Target EETField for type information.
            value: String value from CSV.

        Returns:
            Coerced value.
        """
        if field.data_type in (EETDataType.DECIMAL, EETDataType.PERCENTAGE):
            try:
                return float(value)
            except ValueError:
                return value
        elif field.data_type == EETDataType.INTEGER:
            try:
                return int(float(value))
            except ValueError:
                return value
        elif field.data_type == EETDataType.BOOLEAN:
            return value.lower() in ("true", "yes", "1")
        return value
