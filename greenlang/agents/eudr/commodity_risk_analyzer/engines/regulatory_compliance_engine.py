# -*- coding: utf-8 -*-
"""
RegulatoryComplianceEngine - AGENT-EUDR-018 Engine 6: EUDR Article-Specific Compliance

Maps EUDR Article-specific requirements per commodity type, performs compliance
checks against documentation evidence, assesses penalty risk for non-compliance,
and generates gap analyses with prioritized remediation recommendations.

Zero-Hallucination Guarantees:
    - All compliance scoring uses deterministic weighted formula (Decimal).
    - Article-to-commodity mapping uses static regulatory lookup tables.
    - Penalty risk assessment uses Member State implementation data.
    - Documentation requirements are commodity-specific static lists.
    - SHA-256 provenance hashes on all output objects.

EUDR Articles Covered:
    - Article 3:  Prohibition (placing on market / making available / exporting)
    - Article 4:  Obligations of operators (due diligence system)
    - Article 5:  Obligations of traders (simplified due diligence for SMEs)
    - Article 8:  Collection of information
    - Article 9:  Due diligence statements
    - Article 10: Risk assessment requirements
    - Article 11: Risk mitigation measures
    - Article 12: Reporting obligations
    - Article 13: Record keeping (5 years)
    - Article 29: Benchmarking system (country risk classification)

Commodity-Specific Differentiation:
    - Cattle: Farm GPS, animal health records, grazing boundaries
    - Cocoa: Cooperative records, fermentation/drying facility IDs
    - Coffee: Washing station records, quality grade certificates
    - Oil Palm: Mill GPS, plantation boundaries, NDPE compliance
    - Rubber: Processing facility records, FSC chain of custody
    - Soya: Storage facility records, GMO status declaration
    - Wood: Species identification (genus/species), felling license

Performance Targets:
    - Full compliance check: <200ms per commodity
    - Article mapping: <10ms
    - Gap analysis generation: <100ms
    - Penalty risk assessment: <50ms

Regulatory References:
    - EU 2023/1115 (EUDR) - All articles
    - Member State implementing legislation (penalty ranges)
    - EUDR Enforcement: 30 Dec 2025 (large), 30 Jun 2026 (SMEs)

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-018, Engine 6 (Regulatory Compliance Engine)
Agent ID: GL-EUDR-CRA-018
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version
# ---------------------------------------------------------------------------

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, dataclass with to_dict, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "reg") -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _to_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid EUDR commodity types.
EUDR_COMMODITIES: frozenset = frozenset({
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
})

#: Valid market designations.
VALID_MARKETS: frozenset = frozenset({"EU", "UK", "CH"})

#: Minimum compliance score.
MIN_COMPLIANCE_SCORE: Decimal = Decimal("0")

#: Maximum compliance score.
MAX_COMPLIANCE_SCORE: Decimal = Decimal("100")

#: EUDR cutoff date.
EUDR_CUTOFF_DATE: str = "2020-12-31"

#: EUDR enforcement dates.
EUDR_ENFORCEMENT_LARGE: str = "2025-12-30"
EUDR_ENFORCEMENT_SME: str = "2026-06-30"

#: Record retention period in years per Article 13.
RECORD_RETENTION_YEARS: int = 5

# ---------------------------------------------------------------------------
# EUDR Article Definitions
# ---------------------------------------------------------------------------

EUDR_ARTICLES: Dict[str, Dict[str, Any]] = {
    "article_3": {
        "number": "3",
        "title": "Prohibition",
        "summary": "Relevant commodities and products shall not be placed on "
                   "the Union market, made available on the Union market, or "
                   "exported unless they are deforestation-free, have been "
                   "produced in accordance with relevant legislation of the "
                   "country of production, and are covered by a due diligence "
                   "statement.",
        "applies_to_all_commodities": True,
        "weight": Decimal("0.20"),
        "criticality": "MANDATORY",
    },
    "article_4": {
        "number": "4",
        "title": "Obligations of operators",
        "summary": "Operators shall exercise due diligence for each relevant "
                   "commodity or product placed on, made available on, or "
                   "exported from the Union market.",
        "applies_to_all_commodities": True,
        "weight": Decimal("0.15"),
        "criticality": "MANDATORY",
    },
    "article_5": {
        "number": "5",
        "title": "Obligations of traders",
        "summary": "Traders that are not SMEs shall exercise the same due "
                   "diligence as operators. SME traders have simplified "
                   "obligations.",
        "applies_to_all_commodities": True,
        "weight": Decimal("0.05"),
        "criticality": "CONDITIONAL",
    },
    "article_8": {
        "number": "8",
        "title": "Collection of information",
        "summary": "Operators shall collect information on the commodities "
                   "and products including geolocation, quantity, supplier "
                   "details, and deforestation-free status evidence.",
        "applies_to_all_commodities": True,
        "weight": Decimal("0.15"),
        "criticality": "MANDATORY",
    },
    "article_9": {
        "number": "9",
        "title": "Due diligence statements",
        "summary": "Operators shall submit a due diligence statement to the "
                   "competent authority before placing the product on the "
                   "market. The statement confirms due diligence has been "
                   "exercised and no or only negligible risk remains.",
        "applies_to_all_commodities": True,
        "weight": Decimal("0.15"),
        "criticality": "MANDATORY",
    },
    "article_10": {
        "number": "10",
        "title": "Risk assessment",
        "summary": "Operators shall assess the risk of non-compliance based "
                   "on the information collected under Article 8, considering "
                   "deforestation prevalence, country risk level, presence of "
                   "indigenous peoples, and complexity of supply chain.",
        "applies_to_all_commodities": True,
        "weight": Decimal("0.10"),
        "criticality": "MANDATORY",
    },
    "article_11": {
        "number": "11",
        "title": "Risk mitigation",
        "summary": "Where the risk assessment identifies a non-negligible risk, "
                   "operators shall take adequate and proportionate risk "
                   "mitigation measures.",
        "applies_to_all_commodities": True,
        "weight": Decimal("0.05"),
        "criticality": "CONDITIONAL",
    },
    "article_12": {
        "number": "12",
        "title": "Reporting obligations",
        "summary": "Operators shall report annually on their due diligence "
                   "system, including volumes, countries of origin, and risk "
                   "assessment outcomes.",
        "applies_to_all_commodities": True,
        "weight": Decimal("0.05"),
        "criticality": "MANDATORY",
    },
    "article_13": {
        "number": "13",
        "title": "Record keeping",
        "summary": "Operators shall keep records of their due diligence for "
                   "at least 5 years from the date on which the product was "
                   "placed on the market or exported.",
        "applies_to_all_commodities": True,
        "weight": Decimal("0.05"),
        "criticality": "MANDATORY",
    },
    "article_29": {
        "number": "29",
        "title": "Benchmarking system",
        "summary": "The Commission shall classify countries or parts thereof "
                   "as low, standard, or high risk based on criteria including "
                   "rate of deforestation, rate of forest degradation, "
                   "production trends, and governance indicators.",
        "applies_to_all_commodities": True,
        "weight": Decimal("0.05"),
        "criticality": "INFORMATIONAL",
    },
}

# ---------------------------------------------------------------------------
# Commodity-Specific Documentation Requirements
# ---------------------------------------------------------------------------

COMMODITY_DOCUMENTATION: Dict[str, List[Dict[str, Any]]] = {
    "cattle": [
        {"doc_type": "farm_gps_coordinates", "description": "GPS coordinates of the farm where cattle were raised", "article": "8", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "animal_health_records", "description": "Veterinary health records for the herd", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "grazing_area_boundaries", "description": "Polygon boundaries of grazing areas", "article": "8", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "slaughterhouse_records", "description": "Slaughterhouse identification and processing records", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "animal_movement_records", "description": "Records of animal movements between farms", "article": "10", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "ear_tag_identification", "description": "Individual animal identification via ear tags", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "deforestation_free_declaration", "description": "Declaration confirming no deforestation after Dec 31, 2020", "article": "9", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "country_legislation_compliance", "description": "Evidence of compliance with country of production legislation", "article": "3", "mandatory": True, "weight": Decimal("0.15")},
    ],
    "cocoa": [
        {"doc_type": "farm_gps_coordinates", "description": "GPS coordinates of cocoa farms", "article": "8", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "cooperative_records", "description": "Cocoa cooperative membership and purchase records", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "fermentation_facility_id", "description": "Identification of fermentation and drying facilities", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "export_certificate", "description": "Export certificate from country of origin", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "bean_quality_grade", "description": "Cocoa bean quality grade and classification", "article": "8", "mandatory": False, "weight": Decimal("0.05")},
        {"doc_type": "traceability_certificate", "description": "Farm-to-port traceability certification", "article": "10", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "deforestation_free_declaration", "description": "Declaration confirming no deforestation after Dec 31, 2020", "article": "9", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "country_legislation_compliance", "description": "Evidence of compliance with country of production legislation", "article": "3", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "sustainability_certification", "description": "Rainforest Alliance, UTZ, or Fairtrade certification", "article": "10", "mandatory": False, "weight": Decimal("0.05")},
    ],
    "coffee": [
        {"doc_type": "farm_gps_coordinates", "description": "GPS coordinates of coffee farms", "article": "8", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "washing_station_records", "description": "Wet mill / washing station processing records", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "cooperative_membership", "description": "Farmer cooperative membership documentation", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "quality_grade_certificate", "description": "Coffee quality grade (SCA score) certificate", "article": "8", "mandatory": False, "weight": Decimal("0.05")},
        {"doc_type": "export_certificate", "description": "ICO export certificate", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "traceability_certificate", "description": "Farm-to-export traceability documentation", "article": "10", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "deforestation_free_declaration", "description": "Declaration confirming no deforestation after Dec 31, 2020", "article": "9", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "country_legislation_compliance", "description": "Evidence of compliance with country of production legislation", "article": "3", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "altitude_verification", "description": "Farm altitude verification (for highland coffee claims)", "article": "10", "mandatory": False, "weight": Decimal("0.05")},
    ],
    "oil_palm": [
        {"doc_type": "mill_gps_coordinates", "description": "GPS coordinates of palm oil mills", "article": "8", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "plantation_boundaries", "description": "Polygon boundaries of oil palm plantations", "article": "8", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "ndpe_compliance", "description": "No Deforestation, No Peat, No Exploitation policy compliance", "article": "10", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "rspo_certificate", "description": "RSPO certification (Identity Preserved, Segregated, Mass Balance, or Credits)", "article": "10", "mandatory": False, "weight": Decimal("0.10")},
        {"doc_type": "iscc_certificate", "description": "ISCC certification for sustainability", "article": "10", "mandatory": False, "weight": Decimal("0.05")},
        {"doc_type": "supply_base_map", "description": "Map of the supply base including all sourcing areas", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "deforestation_free_declaration", "description": "Declaration confirming no deforestation after Dec 31, 2020", "article": "9", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "country_legislation_compliance", "description": "Evidence of compliance with country of production legislation", "article": "3", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "peat_assessment", "description": "Peat depth assessment for plantation areas", "article": "10", "mandatory": True, "weight": Decimal("0.05")},
    ],
    "rubber": [
        {"doc_type": "plantation_gps_coordinates", "description": "GPS coordinates of rubber plantations", "article": "8", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "processing_facility_records", "description": "Rubber processing factory records", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "chain_of_custody", "description": "FSC-equivalent chain of custody documentation", "article": "10", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "tapping_area_boundaries", "description": "Polygon boundaries of tapping concession areas", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "smallholder_registry", "description": "Registry of smallholder rubber farmers in supply chain", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "deforestation_free_declaration", "description": "Declaration confirming no deforestation after Dec 31, 2020", "article": "9", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "country_legislation_compliance", "description": "Evidence of compliance with country of production legislation", "article": "3", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "gps_track_verification", "description": "GPS tracking of rubber latex from plantation to factory", "article": "10", "mandatory": False, "weight": Decimal("0.10")},
    ],
    "soya": [
        {"doc_type": "farm_gps_coordinates", "description": "GPS coordinates of soya farms", "article": "8", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "storage_facility_records", "description": "Grain storage and silo facility records", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "gmo_status_declaration", "description": "GMO status declaration (GM or non-GM)", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "crushing_plant_id", "description": "Crushing plant identification and processing records", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "rtrs_certificate", "description": "Round Table on Responsible Soy certification", "article": "10", "mandatory": False, "weight": Decimal("0.05")},
        {"doc_type": "car_registration", "description": "CAR (Rural Environmental Registry) registration for Brazil", "article": "8", "mandatory": False, "weight": Decimal("0.10")},
        {"doc_type": "deforestation_free_declaration", "description": "Declaration confirming no deforestation after Dec 31, 2020", "article": "9", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "country_legislation_compliance", "description": "Evidence of compliance with country of production legislation", "article": "3", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "satellite_monitoring_report", "description": "Satellite monitoring evidence for soya plots", "article": "10", "mandatory": True, "weight": Decimal("0.10")},
    ],
    "wood": [
        {"doc_type": "species_identification", "description": "Tree species identification (genus and species)", "article": "8", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "forest_management_unit_gps", "description": "GPS coordinates and boundaries of forest management unit", "article": "8", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "felling_license", "description": "Felling license or harvest permit", "article": "8", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "sawmill_records", "description": "Sawmill processing records linking logs to products", "article": "8", "mandatory": True, "weight": Decimal("0.10")},
        {"doc_type": "fsc_certificate", "description": "FSC chain of custody certificate", "article": "10", "mandatory": False, "weight": Decimal("0.10")},
        {"doc_type": "pefc_certificate", "description": "PEFC chain of custody certificate", "article": "10", "mandatory": False, "weight": Decimal("0.05")},
        {"doc_type": "cites_permit", "description": "CITES permit if species is listed", "article": "8", "mandatory": False, "weight": Decimal("0.05")},
        {"doc_type": "deforestation_free_declaration", "description": "Declaration confirming no deforestation after Dec 31, 2020", "article": "9", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "country_legislation_compliance", "description": "Evidence of compliance with country of production legislation", "article": "3", "mandatory": True, "weight": Decimal("0.15")},
        {"doc_type": "timber_legality_verification", "description": "Independent timber legality verification", "article": "10", "mandatory": True, "weight": Decimal("0.10")},
    ],
}

# ---------------------------------------------------------------------------
# Penalty Risk Data per Member State
# ---------------------------------------------------------------------------

MEMBER_STATE_PENALTIES: Dict[str, Dict[str, Any]] = {
    "DE": {
        "country": "Germany",
        "min_fine_eur": Decimal("50000"),
        "max_fine_eur": Decimal("2000000"),
        "revenue_percentage": Decimal("4.0"),
        "criminal_liability": True,
        "import_ban_authority": True,
        "confiscation_authority": True,
    },
    "FR": {
        "country": "France",
        "min_fine_eur": Decimal("75000"),
        "max_fine_eur": Decimal("3000000"),
        "revenue_percentage": Decimal("4.0"),
        "criminal_liability": True,
        "import_ban_authority": True,
        "confiscation_authority": True,
    },
    "NL": {
        "country": "Netherlands",
        "min_fine_eur": Decimal("50000"),
        "max_fine_eur": Decimal("1000000"),
        "revenue_percentage": Decimal("3.0"),
        "criminal_liability": True,
        "import_ban_authority": True,
        "confiscation_authority": True,
    },
    "IT": {
        "country": "Italy",
        "min_fine_eur": Decimal("25000"),
        "max_fine_eur": Decimal("1500000"),
        "revenue_percentage": Decimal("4.0"),
        "criminal_liability": False,
        "import_ban_authority": True,
        "confiscation_authority": True,
    },
    "ES": {
        "country": "Spain",
        "min_fine_eur": Decimal("30000"),
        "max_fine_eur": Decimal("1200000"),
        "revenue_percentage": Decimal("3.5"),
        "criminal_liability": False,
        "import_ban_authority": True,
        "confiscation_authority": True,
    },
    "BE": {
        "country": "Belgium",
        "min_fine_eur": Decimal("40000"),
        "max_fine_eur": Decimal("800000"),
        "revenue_percentage": Decimal("3.0"),
        "criminal_liability": True,
        "import_ban_authority": True,
        "confiscation_authority": False,
    },
    "DEFAULT": {
        "country": "Default EU Member State",
        "min_fine_eur": Decimal("25000"),
        "max_fine_eur": Decimal("1000000"),
        "revenue_percentage": Decimal("4.0"),
        "criminal_liability": False,
        "import_ban_authority": True,
        "confiscation_authority": True,
    },
}

# ---------------------------------------------------------------------------
# Country Risk Benchmarking (Article 29)
# ---------------------------------------------------------------------------

COUNTRY_RISK_BENCHMARKS: Dict[str, str] = {
    # HIGH risk
    "BR": "HIGH", "ID": "HIGH", "MY": "HIGH", "CG": "HIGH",
    "CD": "HIGH", "CM": "HIGH", "PG": "HIGH", "MM": "HIGH",
    "BO": "HIGH", "PY": "HIGH", "LA": "HIGH", "KH": "HIGH",
    # STANDARD risk (default)
    "CI": "STANDARD", "GH": "STANDARD", "CO": "STANDARD",
    "PE": "STANDARD", "EC": "STANDARD", "GT": "STANDARD",
    "HN": "STANDARD", "NI": "STANDARD", "TH": "STANDARD",
    "VN": "STANDARD", "PH": "STANDARD", "IN": "STANDARD",
    "NG": "STANDARD", "ET": "STANDARD", "UG": "STANDARD",
    "TZ": "STANDARD", "MZ": "STANDARD", "AR": "STANDARD",
    # LOW risk
    "US": "LOW", "CA": "LOW", "AU": "LOW", "NZ": "LOW",
    "JP": "LOW", "KR": "LOW", "CL": "LOW", "UY": "LOW",
    "CR": "LOW",
}

# ---------------------------------------------------------------------------
# Regulatory Update Log (simulated static entries)
# ---------------------------------------------------------------------------

REGULATORY_UPDATES: List[Dict[str, Any]] = [
    {
        "update_id": "RU-2025-001",
        "date": "2025-06-15",
        "title": "EUDR Implementation Deadline Confirmed",
        "description": "European Commission confirms December 30, 2025 "
                       "enforcement date for large operators.",
        "affected_commodities": list(EUDR_COMMODITIES),
        "impact": "HIGH",
        "source": "European Commission Official Journal",
    },
    {
        "update_id": "RU-2025-002",
        "date": "2025-09-01",
        "title": "Country Benchmarking Initial Publication",
        "description": "First publication of country risk classifications "
                       "under Article 29.",
        "affected_commodities": list(EUDR_COMMODITIES),
        "impact": "HIGH",
        "source": "European Commission DG Environment",
    },
    {
        "update_id": "RU-2025-003",
        "date": "2025-10-15",
        "title": "Geolocation Requirements Technical Guidance",
        "description": "Technical guidance on geolocation data requirements "
                       "for plots > 4 hectares (polygon) and <= 4 hectares "
                       "(single GPS point).",
        "affected_commodities": list(EUDR_COMMODITIES),
        "impact": "MEDIUM",
        "source": "European Commission DG Environment",
    },
    {
        "update_id": "RU-2026-001",
        "date": "2026-01-15",
        "title": "SME Enforcement Grace Period Reminder",
        "description": "SME traders have until June 30, 2026 to comply with "
                       "simplified due diligence requirements.",
        "affected_commodities": list(EUDR_COMMODITIES),
        "impact": "MEDIUM",
        "source": "European Commission DG Environment",
    },
    {
        "update_id": "RU-2026-002",
        "date": "2026-02-01",
        "title": "Palm Oil Specific Guidance on Mill Traceability",
        "description": "Additional guidance on traceability to the mill level "
                       "for palm oil, requiring supply base documentation.",
        "affected_commodities": ["oil_palm"],
        "impact": "HIGH",
        "source": "European Commission DG Environment",
    },
    {
        "update_id": "RU-2026-003",
        "date": "2026-02-15",
        "title": "Wood Products Species Identification Standard",
        "description": "Mandatory genus-level species identification for "
                       "all wood products using DNA or isotope testing when "
                       "visual identification is insufficient.",
        "affected_commodities": ["wood"],
        "impact": "HIGH",
        "source": "European Commission DG Environment",
    },
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ComplianceRequirement:
    """A single compliance requirement for a commodity.

    Attributes:
        requirement_id: Unique requirement identifier.
        article: EUDR article number.
        article_title: Article title.
        description: Requirement description.
        commodity_type: Applicable commodity.
        documentation_required: Required documentation type.
        mandatory: Whether this requirement is mandatory.
        weight: Weight in compliance scoring (Decimal).
        status: Current compliance status (COMPLIANT, NON_COMPLIANT, PENDING).
        evidence_provided: Whether evidence has been provided.
        provenance_hash: SHA-256 hash.
    """

    requirement_id: str = ""
    article: str = ""
    article_title: str = ""
    description: str = ""
    commodity_type: str = ""
    documentation_required: str = ""
    mandatory: bool = True
    weight: Decimal = Decimal("0.10")
    status: str = "PENDING"
    evidence_provided: bool = False
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "requirement_id": self.requirement_id,
            "article": self.article,
            "article_title": self.article_title,
            "description": self.description,
            "commodity_type": self.commodity_type,
            "documentation_required": self.documentation_required,
            "mandatory": self.mandatory,
            "weight": str(self.weight),
            "status": self.status,
            "evidence_provided": self.evidence_provided,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class ComplianceGap:
    """A compliance gap identified during gap analysis.

    Attributes:
        gap_id: Unique gap identifier.
        article: EUDR article number.
        requirement: Missing or incomplete requirement.
        severity: Gap severity (CRITICAL, HIGH, MEDIUM, LOW).
        remediation: Recommended remediation action.
        estimated_effort_days: Estimated effort to remediate.
        priority_rank: Priority ranking (1 = highest).
        provenance_hash: SHA-256 hash.
    """

    gap_id: str = ""
    article: str = ""
    requirement: str = ""
    severity: str = "MEDIUM"
    remediation: str = ""
    estimated_effort_days: int = 0
    priority_rank: int = 0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "gap_id": self.gap_id,
            "article": self.article,
            "requirement": self.requirement,
            "severity": self.severity,
            "remediation": self.remediation,
            "estimated_effort_days": self.estimated_effort_days,
            "priority_rank": self.priority_rank,
            "provenance_hash": self.provenance_hash,
        }


# ---------------------------------------------------------------------------
# RegulatoryComplianceEngine
# ---------------------------------------------------------------------------


class RegulatoryComplianceEngine:
    """Production-grade EUDR regulatory compliance engine.

    Maps EUDR article-specific requirements per commodity, performs compliance
    checks against supplier data and documentation, assesses penalty risk,
    and generates gap analyses with prioritized remediation steps.

    Thread Safety:
        All mutable state is protected by a reentrant lock.

    Zero-Hallucination:
        All compliance scores use deterministic Decimal-weighted formulas.
        No ML/LLM models in any compliance assessment path.

    Attributes:
        _compliance_records: In-memory store of compliance check results.
        _lock: Reentrant lock for thread-safe state access.

    Example:
        >>> engine = RegulatoryComplianceEngine()
        >>> requirements = engine.get_requirements("wood")
        >>> assert len(requirements) > 0
        >>> result = engine.check_compliance(
        ...     "wood",
        ...     {"supplier_id": "S-001", "origin_country": "BR"},
        ...     {"species_identification": True, "felling_license": True},
        ... )
        >>> assert "compliance_score" in result
    """

    def __init__(self) -> None:
        """Initialize RegulatoryComplianceEngine with empty state."""
        self._compliance_records: Dict[str, Dict[str, Any]] = {}
        self._lock: threading.RLock = threading.RLock()
        logger.info(
            "RegulatoryComplianceEngine initialized (version=%s)",
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_requirements(
        self,
        commodity_type: str,
        market: str = "EU",
    ) -> List[Dict[str, Any]]:
        """Get all EUDR requirements for a specific commodity.

        Returns the combined list of article-level requirements and
        commodity-specific documentation requirements.

        Args:
            commodity_type: EUDR commodity type.
            market: Target market (EU, UK, CH). Default "EU".

        Returns:
            List of requirement dictionaries.

        Raises:
            ValueError: If commodity_type is not a valid EUDR commodity
                or market is not recognized.
        """
        self._validate_commodity(commodity_type)
        self._validate_market(market)

        requirements: List[Dict[str, Any]] = []

        # Article-level requirements
        for article_key, article in EUDR_ARTICLES.items():
            req = ComplianceRequirement(
                requirement_id=_generate_id("req"),
                article=article["number"],
                article_title=article["title"],
                description=article["summary"],
                commodity_type=commodity_type,
                documentation_required="",
                mandatory=article["criticality"] == "MANDATORY",
                weight=article["weight"],
            )
            req.provenance_hash = _compute_hash(req)
            requirements.append(req.to_dict())

        # Commodity-specific documentation requirements
        docs = COMMODITY_DOCUMENTATION.get(commodity_type, [])
        for doc in docs:
            req = ComplianceRequirement(
                requirement_id=_generate_id("doc"),
                article=doc["article"],
                article_title=f"Article {doc['article']} Documentation",
                description=doc["description"],
                commodity_type=commodity_type,
                documentation_required=doc["doc_type"],
                mandatory=doc["mandatory"],
                weight=doc["weight"],
            )
            req.provenance_hash = _compute_hash(req)
            requirements.append(req.to_dict())

        logger.debug(
            "Retrieved %d requirements for commodity=%s market=%s",
            len(requirements), commodity_type, market,
        )
        return requirements

    def check_compliance(
        self,
        commodity_type: str,
        supplier_data: Dict[str, Any],
        documentation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform a full compliance check against EUDR requirements.

        Evaluates each mandatory and optional documentation requirement
        against the provided evidence, computing a weighted compliance
        score.

        Args:
            commodity_type: EUDR commodity type.
            supplier_data: Supplier metadata with keys: ``supplier_id``,
                ``origin_country``, ``operator_size`` (optional).
            documentation: Dictionary of documentation provided, where
                keys are doc_type strings and values are booleans or
                evidence dicts.

        Returns:
            Dictionary with compliance_score, requirement_results,
            missing_mandatory, missing_optional, country_risk, and
            provenance_hash.

        Raises:
            ValueError: If commodity_type is invalid or supplier_data
                is missing required fields.
        """
        start_time = time.monotonic()

        self._validate_commodity(commodity_type)
        if not supplier_data.get("supplier_id"):
            raise ValueError("supplier_data must include 'supplier_id'")

        docs = COMMODITY_DOCUMENTATION.get(commodity_type, [])
        origin_country = supplier_data.get("origin_country", "")

        requirement_results: List[Dict[str, Any]] = []
        missing_mandatory: List[str] = []
        missing_optional: List[str] = []
        total_weight = Decimal("0")
        achieved_weight = Decimal("0")

        for doc in docs:
            doc_type = doc["doc_type"]
            mandatory = doc["mandatory"]
            weight = doc["weight"]
            total_weight += weight

            # Check if documentation is provided
            provided = bool(documentation.get(doc_type))

            status = "COMPLIANT" if provided else "NON_COMPLIANT"
            if provided:
                achieved_weight += weight

            if not provided and mandatory:
                missing_mandatory.append(doc_type)
            elif not provided and not mandatory:
                missing_optional.append(doc_type)

            requirement_results.append({
                "doc_type": doc_type,
                "description": doc["description"],
                "article": doc["article"],
                "mandatory": mandatory,
                "weight": str(weight),
                "status": status,
                "provided": provided,
            })

        # Compliance score
        if total_weight > 0:
            compliance_score = (
                (achieved_weight / total_weight) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            compliance_score = Decimal("0")

        compliance_score = min(MAX_COMPLIANCE_SCORE, max(
            MIN_COMPLIANCE_SCORE, compliance_score,
        ))

        # Country risk from benchmarking
        country_risk = COUNTRY_RISK_BENCHMARKS.get(origin_country, "STANDARD")

        # Overall compliance status
        if missing_mandatory:
            overall_status = "NON_COMPLIANT"
        elif compliance_score >= Decimal("80"):
            overall_status = "COMPLIANT"
        elif compliance_score >= Decimal("50"):
            overall_status = "PARTIALLY_COMPLIANT"
        else:
            overall_status = "NON_COMPLIANT"

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "commodity_type": commodity_type,
            "supplier_id": supplier_data.get("supplier_id", ""),
            "compliance_score": str(compliance_score),
            "overall_status": overall_status,
            "requirement_results": requirement_results,
            "missing_mandatory": missing_mandatory,
            "missing_mandatory_count": len(missing_mandatory),
            "missing_optional": missing_optional,
            "missing_optional_count": len(missing_optional),
            "total_requirements": len(docs),
            "country_risk": country_risk,
            "origin_country": origin_country,
            "processing_time_ms": round(processing_time_ms, 3),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        # Store compliance record
        record_key = f"{supplier_data.get('supplier_id', '')}:{commodity_type}"
        with self._lock:
            self._compliance_records[record_key] = result

        logger.info(
            "Compliance check: commodity=%s supplier=%s score=%s "
            "status=%s missing_mandatory=%d time_ms=%.1f",
            commodity_type, supplier_data.get("supplier_id", ""),
            compliance_score, overall_status,
            len(missing_mandatory), processing_time_ms,
        )
        return result

    def assess_penalty_risk(
        self,
        commodity_type: str,
        compliance_gaps: List[Dict[str, Any]],
        member_state: str = "DEFAULT",
    ) -> Dict[str, Any]:
        """Assess penalty risk for non-compliance with EUDR.

        Calculates potential penalty ranges based on Member State
        implementing legislation, number and severity of compliance gaps,
        and commodity risk factors.

        Args:
            commodity_type: EUDR commodity type.
            compliance_gaps: List of compliance gap dicts from gap analysis.
            member_state: EU Member State code (e.g., "DE", "FR"). Defaults
                to "DEFAULT" for generic assessment.

        Returns:
            Dictionary with penalty range, risk level, contributing factors,
            and remediation priority.

        Raises:
            ValueError: If commodity_type is invalid.
        """
        start_time = time.monotonic()

        self._validate_commodity(commodity_type)

        # Get Member State penalty data
        penalties = MEMBER_STATE_PENALTIES.get(
            member_state, MEMBER_STATE_PENALTIES["DEFAULT"],
        )

        # Count gap severities
        critical_count = sum(
            1 for g in compliance_gaps if g.get("severity") == "CRITICAL"
        )
        high_count = sum(
            1 for g in compliance_gaps if g.get("severity") == "HIGH"
        )
        medium_count = sum(
            1 for g in compliance_gaps if g.get("severity") == "MEDIUM"
        )
        low_count = sum(
            1 for g in compliance_gaps if g.get("severity") == "LOW"
        )

        total_gaps = len(compliance_gaps)

        # Risk multiplier based on gap composition
        risk_multiplier = (
            Decimal(str(critical_count)) * Decimal("0.4")
            + Decimal(str(high_count)) * Decimal("0.3")
            + Decimal(str(medium_count)) * Decimal("0.2")
            + Decimal(str(low_count)) * Decimal("0.1")
        )
        if total_gaps > 0:
            risk_multiplier = risk_multiplier / Decimal(str(total_gaps))
        else:
            risk_multiplier = Decimal("0")

        # Penalty range calculation
        min_fine = penalties["min_fine_eur"]
        max_fine = penalties["max_fine_eur"]
        estimated_min = (min_fine * risk_multiplier).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP,
        )
        estimated_max = (max_fine * risk_multiplier).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP,
        )

        # Overall penalty risk level
        if critical_count > 0:
            penalty_risk_level = "CRITICAL"
        elif high_count > 0:
            penalty_risk_level = "HIGH"
        elif medium_count > 0:
            penalty_risk_level = "MEDIUM"
        else:
            penalty_risk_level = "LOW"

        # Contributing factors
        factors: List[str] = []
        if critical_count > 0:
            factors.append(
                f"{critical_count} critical gaps (mandatory requirements missing)"
            )
        if high_count > 0:
            factors.append(f"{high_count} high-severity gaps")
        if penalties.get("criminal_liability"):
            factors.append("Criminal liability applies in this jurisdiction")
        if penalties.get("import_ban_authority"):
            factors.append("Import ban authority available to regulators")
        if penalties.get("confiscation_authority"):
            factors.append("Product confiscation authority available")

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "commodity_type": commodity_type,
            "member_state": member_state,
            "member_state_name": penalties["country"],
            "penalty_risk_level": penalty_risk_level,
            "estimated_fine_range": {
                "min_eur": str(estimated_min),
                "max_eur": str(estimated_max),
                "currency": "EUR",
            },
            "revenue_percentage_risk": str(penalties["revenue_percentage"]),
            "criminal_liability": penalties.get("criminal_liability", False),
            "import_ban_risk": penalties.get("import_ban_authority", False),
            "confiscation_risk": penalties.get("confiscation_authority", False),
            "gap_summary": {
                "total_gaps": total_gaps,
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
            },
            "contributing_factors": factors,
            "risk_multiplier": str(risk_multiplier.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP,
            )),
            "processing_time_ms": round(processing_time_ms, 3),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Penalty risk assessed: commodity=%s state=%s level=%s "
            "estimated_max=%s time_ms=%.1f",
            commodity_type, member_state, penalty_risk_level,
            estimated_max, processing_time_ms,
        )
        return result

    def get_documentation_requirements(
        self,
        commodity_type: str,
    ) -> List[Dict[str, Any]]:
        """Get required documentation per commodity type.

        Returns the commodity-specific documentation list with mandatory
        flags, weights, and article references.

        Args:
            commodity_type: EUDR commodity type.

        Returns:
            List of documentation requirement dictionaries.

        Raises:
            ValueError: If commodity_type is invalid.
        """
        self._validate_commodity(commodity_type)

        docs = COMMODITY_DOCUMENTATION.get(commodity_type, [])
        result: List[Dict[str, Any]] = []
        for doc in docs:
            entry = {
                "doc_type": doc["doc_type"],
                "description": doc["description"],
                "article": doc["article"],
                "mandatory": doc["mandatory"],
                "weight": str(doc["weight"]),
            }
            result.append(entry)

        logger.debug(
            "Documentation requirements for %s: %d items",
            commodity_type, len(result),
        )
        return result

    def map_articles_to_commodity(
        self,
        commodity_type: str,
    ) -> Dict[str, Any]:
        """Map EUDR articles (Art. 3-13, 29) to commodity-specific requirements.

        Provides article details, applicable documentation, and relevance
        assessment for the specified commodity.

        Args:
            commodity_type: EUDR commodity type.

        Returns:
            Dictionary mapping article numbers to their details and
            commodity-specific requirements.

        Raises:
            ValueError: If commodity_type is invalid.
        """
        self._validate_commodity(commodity_type)

        docs = COMMODITY_DOCUMENTATION.get(commodity_type, [])
        docs_by_article: Dict[str, List[Dict[str, Any]]] = {}
        for doc in docs:
            art = doc["article"]
            if art not in docs_by_article:
                docs_by_article[art] = []
            docs_by_article[art].append({
                "doc_type": doc["doc_type"],
                "description": doc["description"],
                "mandatory": doc["mandatory"],
            })

        mapping: Dict[str, Any] = {}
        for article_key, article in EUDR_ARTICLES.items():
            art_num = article["number"]
            article_docs = docs_by_article.get(art_num, [])
            mapping[f"article_{art_num}"] = {
                "number": art_num,
                "title": article["title"],
                "summary": article["summary"],
                "criticality": article["criticality"],
                "weight": str(article["weight"]),
                "commodity_specific_requirements": article_docs,
                "requirement_count": len(article_docs),
            }

        result = {
            "commodity_type": commodity_type,
            "article_mapping": mapping,
            "total_articles": len(EUDR_ARTICLES),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.debug(
            "Article mapping for %s: %d articles",
            commodity_type, len(mapping),
        )
        return result

    def check_import_ban_risk(
        self,
        commodity_type: str,
        origin_country: str,
    ) -> Dict[str, Any]:
        """Assess risk of import ban based on commodity and country of origin.

        Uses Article 29 benchmarking data and commodity-specific risk
        factors to determine import ban likelihood.

        Args:
            commodity_type: EUDR commodity type.
            origin_country: ISO 3166-1 alpha-2 country code.

        Returns:
            Dictionary with ban_risk_level, country_benchmark, factors,
            and provenance_hash.

        Raises:
            ValueError: If commodity_type is invalid or origin_country is empty.
        """
        self._validate_commodity(commodity_type)
        if not origin_country:
            raise ValueError("origin_country must be a non-empty string")

        country_benchmark = COUNTRY_RISK_BENCHMARKS.get(
            origin_country, "STANDARD",
        )

        # Risk factors
        factors: List[str] = []
        ban_risk_score = Decimal("0")

        if country_benchmark == "HIGH":
            ban_risk_score += Decimal("50")
            factors.append(
                f"Country {origin_country} classified as HIGH risk "
                f"under Article 29 benchmarking"
            )
        elif country_benchmark == "STANDARD":
            ban_risk_score += Decimal("20")
            factors.append(
                f"Country {origin_country} classified as STANDARD risk"
            )
        else:
            ban_risk_score += Decimal("5")
            factors.append(
                f"Country {origin_country} classified as LOW risk"
            )

        # Commodity-specific risk adjustment
        high_risk_commodities_in_country: Dict[str, List[str]] = {
            "BR": ["soya", "cattle", "wood"],
            "ID": ["oil_palm", "wood", "rubber"],
            "MY": ["oil_palm", "rubber"],
            "CI": ["cocoa"],
            "GH": ["cocoa"],
        }
        country_high_risk = high_risk_commodities_in_country.get(
            origin_country, [],
        )
        if commodity_type in country_high_risk:
            ban_risk_score += Decimal("25")
            factors.append(
                f"{commodity_type} is a high-risk commodity in "
                f"{origin_country}"
            )

        ban_risk_score = min(MAX_COMPLIANCE_SCORE, ban_risk_score)

        # Risk level classification
        if ban_risk_score >= Decimal("70"):
            ban_risk_level = "HIGH"
        elif ban_risk_score >= Decimal("40"):
            ban_risk_level = "MEDIUM"
        else:
            ban_risk_level = "LOW"

        result = {
            "commodity_type": commodity_type,
            "origin_country": origin_country,
            "country_benchmark": country_benchmark,
            "ban_risk_level": ban_risk_level,
            "ban_risk_score": str(ban_risk_score),
            "factors": factors,
            "enhanced_due_diligence_required": (
                country_benchmark == "HIGH"
                or ban_risk_level in ("HIGH", "MEDIUM")
            ),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Import ban risk: commodity=%s country=%s benchmark=%s "
            "level=%s score=%s",
            commodity_type, origin_country, country_benchmark,
            ban_risk_level, ban_risk_score,
        )
        return result

    def get_regulatory_updates(
        self,
        commodity_type: Optional[str] = None,
        since_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent regulatory changes affecting EUDR commodities.

        Args:
            commodity_type: Filter by commodity. If None, returns all updates.
            since_date: ISO date string. Only return updates on or after
                this date. If None, returns all updates.

        Returns:
            List of regulatory update dictionaries sorted by date descending.

        Raises:
            ValueError: If commodity_type is provided but invalid.
        """
        if commodity_type is not None:
            self._validate_commodity(commodity_type)

        updates = list(REGULATORY_UPDATES)

        # Filter by commodity
        if commodity_type is not None:
            updates = [
                u for u in updates
                if commodity_type in u.get("affected_commodities", [])
            ]

        # Filter by date
        if since_date is not None:
            updates = [
                u for u in updates
                if u.get("date", "") >= since_date
            ]

        # Sort by date descending
        updates.sort(key=lambda u: u.get("date", ""), reverse=True)

        logger.debug(
            "Regulatory updates: commodity=%s since=%s count=%d",
            commodity_type, since_date, len(updates),
        )
        return updates

    def calculate_compliance_score(
        self,
        commodity_type: str,
        evidence_package: Dict[str, Any],
    ) -> Decimal:
        """Calculate a compliance score (0-100) for an evidence package.

        Evaluates the evidence package against all mandatory and optional
        documentation requirements for the commodity, applying weights
        per requirement.

        Args:
            commodity_type: EUDR commodity type.
            evidence_package: Dictionary mapping doc_type strings to
                evidence data (truthy = provided, falsy = missing).

        Returns:
            Decimal compliance score between 0 and 100.

        Raises:
            ValueError: If commodity_type is invalid.
        """
        self._validate_commodity(commodity_type)

        docs = COMMODITY_DOCUMENTATION.get(commodity_type, [])
        if not docs:
            return Decimal("0")

        total_weight = Decimal("0")
        achieved_weight = Decimal("0")

        for doc in docs:
            weight = doc["weight"]
            total_weight += weight
            if evidence_package.get(doc["doc_type"]):
                achieved_weight += weight

        if total_weight == 0:
            return Decimal("0")

        score = (
            (achieved_weight / total_weight) * Decimal("100")
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        return min(MAX_COMPLIANCE_SCORE, max(MIN_COMPLIANCE_SCORE, score))

    def generate_compliance_gap_analysis(
        self,
        commodity_type: str,
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a gap analysis with prioritized remediation steps.

        Compares the current compliance state against all requirements
        for the commodity and produces a ranked list of gaps with
        remediation recommendations and effort estimates.

        Args:
            commodity_type: EUDR commodity type.
            current_state: Dictionary mapping doc_type strings to
                current status (True/False or evidence dict).

        Returns:
            Dictionary with gaps list, gap_count, compliance_score,
            estimated_total_effort_days, and provenance_hash.

        Raises:
            ValueError: If commodity_type is invalid.
        """
        start_time = time.monotonic()

        self._validate_commodity(commodity_type)

        docs = COMMODITY_DOCUMENTATION.get(commodity_type, [])
        gaps: List[ComplianceGap] = []

        for doc in docs:
            doc_type = doc["doc_type"]
            provided = bool(current_state.get(doc_type))

            if not provided:
                severity = self._classify_gap_severity(doc)
                effort = self._estimate_remediation_effort(doc_type, doc)
                remediation = self._get_remediation_recommendation(
                    doc_type, commodity_type,
                )

                gap = ComplianceGap(
                    gap_id=_generate_id("gap"),
                    article=doc["article"],
                    requirement=f"{doc_type}: {doc['description']}",
                    severity=severity,
                    remediation=remediation,
                    estimated_effort_days=effort,
                )
                gap.provenance_hash = _compute_hash(gap)
                gaps.append(gap)

        # Sort gaps by severity priority
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        gaps.sort(key=lambda g: severity_order.get(g.severity, 99))

        # Assign priority ranks
        for rank, gap in enumerate(gaps, start=1):
            gap.priority_rank = rank

        # Calculate compliance score
        compliance_score = self.calculate_compliance_score(
            commodity_type, current_state,
        )

        total_effort = sum(g.estimated_effort_days for g in gaps)
        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "commodity_type": commodity_type,
            "gaps": [g.to_dict() for g in gaps],
            "gap_count": len(gaps),
            "compliance_score": str(compliance_score),
            "estimated_total_effort_days": total_effort,
            "severity_breakdown": {
                "critical": sum(1 for g in gaps if g.severity == "CRITICAL"),
                "high": sum(1 for g in gaps if g.severity == "HIGH"),
                "medium": sum(1 for g in gaps if g.severity == "MEDIUM"),
                "low": sum(1 for g in gaps if g.severity == "LOW"),
            },
            "processing_time_ms": round(processing_time_ms, 3),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Gap analysis for %s: %d gaps, score=%s, effort=%d days, "
            "time_ms=%.1f",
            commodity_type, len(gaps), compliance_score,
            total_effort, processing_time_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_commodity(self, commodity_type: str) -> None:
        """Validate that commodity_type is a valid EUDR commodity.

        Args:
            commodity_type: Commodity to validate.

        Raises:
            ValueError: If not a valid EUDR commodity.
        """
        if commodity_type not in EUDR_COMMODITIES:
            raise ValueError(
                f"'{commodity_type}' is not a valid EUDR commodity. "
                f"Valid: {sorted(EUDR_COMMODITIES)}"
            )

    def _validate_market(self, market: str) -> None:
        """Validate market designation.

        Args:
            market: Market code.

        Raises:
            ValueError: If not a recognized market.
        """
        if market not in VALID_MARKETS:
            raise ValueError(
                f"'{market}' is not a valid market. Valid: {sorted(VALID_MARKETS)}"
            )

    def _classify_gap_severity(self, doc: Dict[str, Any]) -> str:
        """Classify gap severity based on document properties.

        Args:
            doc: Documentation requirement dictionary.

        Returns:
            Severity string: CRITICAL, HIGH, MEDIUM, or LOW.
        """
        mandatory = doc.get("mandatory", False)
        weight = doc.get("weight", Decimal("0"))
        article = doc.get("article", "")

        if mandatory and weight >= Decimal("0.15"):
            return "CRITICAL"
        if mandatory and article in ("3", "9"):
            return "CRITICAL"
        if mandatory:
            return "HIGH"
        if weight >= Decimal("0.10"):
            return "MEDIUM"
        return "LOW"

    def _estimate_remediation_effort(
        self,
        doc_type: str,
        doc: Dict[str, Any],
    ) -> int:
        """Estimate remediation effort in business days.

        Args:
            doc_type: Documentation type string.
            doc: Full documentation requirement dict.

        Returns:
            Estimated effort in business days.
        """
        effort_map: Dict[str, int] = {
            "farm_gps_coordinates": 5,
            "animal_health_records": 10,
            "grazing_area_boundaries": 7,
            "slaughterhouse_records": 5,
            "animal_movement_records": 10,
            "ear_tag_identification": 15,
            "cooperative_records": 7,
            "fermentation_facility_id": 3,
            "export_certificate": 5,
            "washing_station_records": 5,
            "cooperative_membership": 3,
            "quality_grade_certificate": 5,
            "mill_gps_coordinates": 5,
            "plantation_boundaries": 10,
            "ndpe_compliance": 30,
            "rspo_certificate": 60,
            "iscc_certificate": 45,
            "supply_base_map": 15,
            "processing_facility_records": 5,
            "chain_of_custody": 30,
            "tapping_area_boundaries": 10,
            "smallholder_registry": 20,
            "storage_facility_records": 5,
            "gmo_status_declaration": 5,
            "crushing_plant_id": 3,
            "rtrs_certificate": 45,
            "car_registration": 15,
            "species_identification": 10,
            "forest_management_unit_gps": 7,
            "felling_license": 10,
            "sawmill_records": 7,
            "fsc_certificate": 60,
            "pefc_certificate": 45,
            "cites_permit": 20,
            "timber_legality_verification": 30,
            "deforestation_free_declaration": 5,
            "country_legislation_compliance": 15,
            "traceability_certificate": 20,
            "sustainability_certification": 45,
            "satellite_monitoring_report": 10,
            "peat_assessment": 15,
            "altitude_verification": 3,
            "gps_track_verification": 7,
            "bean_quality_grade": 3,
        }
        return effort_map.get(doc_type, 10)

    def _get_remediation_recommendation(
        self,
        doc_type: str,
        commodity_type: str,
    ) -> str:
        """Get a remediation recommendation for a missing document.

        Args:
            doc_type: Missing documentation type.
            commodity_type: Commodity type.

        Returns:
            Remediation recommendation string.
        """
        recommendations: Dict[str, str] = {
            "farm_gps_coordinates": (
                "Collect GPS coordinates using a GNSS-enabled device with "
                "minimum 4-decimal-place precision. For plots > 4ha, collect "
                "polygon boundaries."
            ),
            "deforestation_free_declaration": (
                "Obtain a signed declaration from the supplier confirming no "
                "deforestation occurred after December 31, 2020 on the "
                "production plot(s)."
            ),
            "country_legislation_compliance": (
                "Verify and document compliance with all relevant legislation "
                "in the country of production including land use rights, "
                "environmental regulations, and labor laws."
            ),
            "species_identification": (
                "Obtain genus and species level identification of wood "
                "products. Use DNA testing or isotope analysis if visual "
                "identification is insufficient."
            ),
            "felling_license": (
                "Obtain a copy of the valid felling license or harvest permit "
                "from the competent authority in the country of production."
            ),
            "rspo_certificate": (
                "Obtain RSPO certification (IP, SG, MB, or Credits) from an "
                "accredited RSPO certification body."
            ),
            "chain_of_custody": (
                "Establish or obtain FSC-equivalent chain of custody "
                "documentation tracing material from source to final product."
            ),
            "ndpe_compliance": (
                "Implement and document a No Deforestation, No Peat, No "
                "Exploitation (NDPE) policy and verify supplier compliance."
            ),
        }
        default = (
            f"Obtain and submit the required {doc_type} documentation "
            f"for {commodity_type} to achieve EUDR compliance."
        )
        return recommendations.get(doc_type, default)
