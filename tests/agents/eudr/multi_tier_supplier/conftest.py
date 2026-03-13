# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-008 Multi-Tier Supplier Tracker test suite.

Provides reusable fixtures for sample supplier profiles, relationships,
certification records, supply chain hierarchies, engine instances, and
helper functions used across all test modules in this package.

Sample Suppliers (20+ predefined across all tiers, 7 commodities):
    COCOA_IMPORTER_EU, COCOA_TRADER_GH, COCOA_PROCESSOR_GH,
    COCOA_AGGREGATOR_GH, COCOA_COOPERATIVE_GH, COCOA_FARMER_1_GH,
    COCOA_FARMER_2_GH, COFFEE_IMPORTER_DE, COFFEE_EXPORTER_CO,
    COFFEE_MILL_CO, COFFEE_COOPERATIVE_CO, COFFEE_FARMER_CO,
    PALM_IMPORTER_NL, PALM_REFINERY_ID, PALM_MILL_ID,
    PALM_SMALLHOLDER_ID, SOYA_TRADER_BR, RUBBER_DEALER_TH,
    CATTLE_FEEDLOT_BR, TIMBER_SAWMILL_CD

Sample Relationships, Certifications, Supply Chain Hierarchies.
Fixture factories for all eight engine components.
Helper assertions and builder functions.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHA256_HEX_LENGTH: int = 64

EUDR_COMMODITIES: List[str] = [
    "cocoa", "coffee", "palm_oil", "soya", "rubber", "cattle", "wood",
]

CERTIFICATION_TYPES: List[str] = [
    "FSC", "RSPO", "UTZ", "RAINFOREST_ALLIANCE", "FAIRTRADE",
    "ORGANIC_EU", "ISO_14001", "4C", "PEFC",
]

RELATIONSHIP_STATES: List[str] = [
    "prospective", "onboarding", "active", "suspended", "terminated",
]

VALID_STATE_TRANSITIONS: List[Tuple[str, str]] = [
    ("prospective", "onboarding"),
    ("onboarding", "active"),
    ("active", "suspended"),
    ("active", "terminated"),
    ("suspended", "active"),
    ("suspended", "terminated"),
]

INVALID_STATE_TRANSITIONS: List[Tuple[str, str]] = [
    ("terminated", "active"),
    ("terminated", "onboarding"),
    ("prospective", "active"),
    ("prospective", "terminated"),
    ("onboarding", "suspended"),
    ("onboarding", "terminated"),
    ("active", "prospective"),
    ("suspended", "prospective"),
    ("suspended", "onboarding"),
]

COMPLIANCE_STATUSES: List[str] = [
    "compliant", "conditionally_compliant", "non_compliant",
    "unverified", "expired",
]

RISK_CATEGORIES: List[str] = [
    "deforestation_proximity", "country_risk", "certification_gap",
    "compliance_history", "data_quality", "concentration_risk",
]

RISK_CATEGORY_WEIGHTS: Dict[str, float] = {
    "deforestation_proximity": 0.30,
    "country_risk": 0.20,
    "certification_gap": 0.15,
    "compliance_history": 0.15,
    "data_quality": 0.10,
    "concentration_risk": 0.10,
}

GAP_SEVERITIES: List[str] = ["critical", "major", "minor"]

PROFILE_COMPLETENESS_WEIGHTS: Dict[str, float] = {
    "legal_identity": 0.25,
    "location": 0.20,
    "commodity": 0.15,
    "certification": 0.15,
    "compliance": 0.15,
    "contact": 0.10,
}

# Country risk scores (0-100, higher = riskier)
COUNTRY_RISK_SCORES: Dict[str, float] = {
    "GH": 45.0,   # Ghana
    "CI": 55.0,   # Cote d'Ivoire
    "CO": 40.0,   # Colombia
    "BR": 50.0,   # Brazil
    "ID": 60.0,   # Indonesia
    "MY": 35.0,   # Malaysia
    "TH": 30.0,   # Thailand
    "CD": 70.0,   # DRC
    "NG": 55.0,   # Nigeria
    "CM": 50.0,   # Cameroon
    "DE": 5.0,    # Germany
    "NL": 5.0,    # Netherlands
    "FR": 5.0,    # France
    "BE": 5.0,    # Belgium
    "US": 10.0,   # USA
    "XX": 75.0,   # Unknown
}

# Commodity-specific typical tier depths
COMMODITY_TIER_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "cocoa": {"min_tiers": 6, "max_tiers": 8, "typical": 7},
    "coffee": {"min_tiers": 5, "max_tiers": 7, "typical": 6},
    "palm_oil": {"min_tiers": 5, "max_tiers": 7, "typical": 6},
    "soya": {"min_tiers": 4, "max_tiers": 6, "typical": 5},
    "rubber": {"min_tiers": 5, "max_tiers": 7, "typical": 6},
    "cattle": {"min_tiers": 3, "max_tiers": 5, "typical": 4},
    "wood": {"min_tiers": 4, "max_tiers": 6, "typical": 5},
}


# ---------------------------------------------------------------------------
# Sample Supplier Profiles (20+ across all tiers and 7 commodities)
# ---------------------------------------------------------------------------

def _make_supplier_id() -> str:
    """Generate a deterministic-looking supplier ID."""
    return f"SUP-{uuid.uuid4().hex[:12].upper()}"


# Pre-generated IDs for deterministic cross-referencing
SUP_ID_COCOA_IMPORTER_EU = "SUP-COCOA-IMP-EU"
SUP_ID_COCOA_TRADER_GH = "SUP-COCOA-TRD-GH"
SUP_ID_COCOA_PROCESSOR_GH = "SUP-COCOA-PRC-GH"
SUP_ID_COCOA_AGGREGATOR_GH = "SUP-COCOA-AGG-GH"
SUP_ID_COCOA_COOPERATIVE_GH = "SUP-COCOA-COP-GH"
SUP_ID_COCOA_FARMER_1_GH = "SUP-COCOA-FM1-GH"
SUP_ID_COCOA_FARMER_2_GH = "SUP-COCOA-FM2-GH"

SUP_ID_COFFEE_IMPORTER_DE = "SUP-COFF-IMP-DE"
SUP_ID_COFFEE_EXPORTER_CO = "SUP-COFF-EXP-CO"
SUP_ID_COFFEE_MILL_CO = "SUP-COFF-MIL-CO"
SUP_ID_COFFEE_COOPERATIVE_CO = "SUP-COFF-COP-CO"
SUP_ID_COFFEE_FARMER_CO = "SUP-COFF-FM1-CO"

SUP_ID_PALM_IMPORTER_NL = "SUP-PALM-IMP-NL"
SUP_ID_PALM_REFINERY_ID = "SUP-PALM-REF-ID"
SUP_ID_PALM_MILL_ID = "SUP-PALM-MIL-ID"
SUP_ID_PALM_SMALLHOLDER_ID = "SUP-PALM-SH1-ID"

SUP_ID_SOYA_TRADER_BR = "SUP-SOYA-TRD-BR"
SUP_ID_RUBBER_DEALER_TH = "SUP-RUBB-DLR-TH"
SUP_ID_CATTLE_FEEDLOT_BR = "SUP-CATL-FDL-BR"
SUP_ID_TIMBER_SAWMILL_CD = "SUP-TMBR-SAW-CD"


COCOA_IMPORTER_EU: Dict[str, Any] = {
    "supplier_id": SUP_ID_COCOA_IMPORTER_EU,
    "legal_name": "EuroChoc GmbH",
    "registration_id": "DE-HRB-123456",
    "tax_id": "DE123456789",
    "duns": "123456789",
    "country_iso": "DE",
    "admin_region": "Hamburg",
    "gps_lat": 53.5511,
    "gps_lon": 9.9937,
    "address": "Speicherstrasse 10, 20095 Hamburg, Germany",
    "commodities": ["cocoa"],
    "tier": 0,
    "role": "importer",
    "annual_volume_mt": 50000.0,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 3,
    "primary_contact": "Hans Mueller",
    "compliance_contact": "Greta Fischer",
    "dds_references": ["DDS-EU-2026-001234"],
    "certifications": [],
    "status": "active",
}

COCOA_TRADER_GH: Dict[str, Any] = {
    "supplier_id": SUP_ID_COCOA_TRADER_GH,
    "legal_name": "Ghana Cocoa Trading Ltd",
    "registration_id": "GH-BRN-789012",
    "tax_id": "GH789012345",
    "duns": "234567890",
    "country_iso": "GH",
    "admin_region": "Greater Accra",
    "gps_lat": 5.6037,
    "gps_lon": -0.1870,
    "address": "Independence Avenue, Accra, Ghana",
    "commodities": ["cocoa"],
    "tier": 1,
    "role": "trader",
    "annual_volume_mt": 30000.0,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 5,
    "primary_contact": "Kwame Asante",
    "compliance_contact": "Ama Boateng",
    "dds_references": ["DDS-EU-2026-002345"],
    "certifications": ["UTZ-GH-2024-001"],
    "status": "active",
}

COCOA_PROCESSOR_GH: Dict[str, Any] = {
    "supplier_id": SUP_ID_COCOA_PROCESSOR_GH,
    "legal_name": "Accra Cocoa Processing Co",
    "registration_id": "GH-BRN-345678",
    "tax_id": "GH345678901",
    "duns": "345678901",
    "country_iso": "GH",
    "admin_region": "Greater Accra",
    "gps_lat": 5.5571,
    "gps_lon": -0.2013,
    "address": "Industrial Area, Tema, Ghana",
    "commodities": ["cocoa"],
    "tier": 2,
    "role": "processor",
    "annual_volume_mt": 20000.0,
    "processing_capacity_mt": 25000.0,
    "upstream_supplier_count": 8,
    "primary_contact": "Kofi Mensah",
    "compliance_contact": None,
    "dds_references": [],
    "certifications": ["UTZ-GH-2024-002"],
    "status": "active",
}

COCOA_AGGREGATOR_GH: Dict[str, Any] = {
    "supplier_id": SUP_ID_COCOA_AGGREGATOR_GH,
    "legal_name": "Ashanti Regional Aggregators",
    "registration_id": "GH-BRN-567890",
    "tax_id": None,
    "duns": None,
    "country_iso": "GH",
    "admin_region": "Ashanti",
    "gps_lat": 6.6885,
    "gps_lon": -1.6244,
    "address": "Kumasi, Ashanti Region, Ghana",
    "commodities": ["cocoa"],
    "tier": 3,
    "role": "aggregator",
    "annual_volume_mt": 8000.0,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 15,
    "primary_contact": "Yaw Osei",
    "compliance_contact": None,
    "dds_references": [],
    "certifications": [],
    "status": "active",
}

COCOA_COOPERATIVE_GH: Dict[str, Any] = {
    "supplier_id": SUP_ID_COCOA_COOPERATIVE_GH,
    "legal_name": "Ahafo Ano Cocoa Cooperative",
    "registration_id": "GH-COOP-112233",
    "tax_id": None,
    "duns": None,
    "country_iso": "GH",
    "admin_region": "Ashanti",
    "gps_lat": 6.9500,
    "gps_lon": -1.9800,
    "address": "Ahafo Ano, Ashanti Region, Ghana",
    "commodities": ["cocoa"],
    "tier": 4,
    "role": "cooperative",
    "annual_volume_mt": 2000.0,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 200,
    "primary_contact": "Abena Owusu",
    "compliance_contact": None,
    "dds_references": [],
    "certifications": [],
    "status": "active",
}

COCOA_FARMER_1_GH: Dict[str, Any] = {
    "supplier_id": SUP_ID_COCOA_FARMER_1_GH,
    "legal_name": "Kwaku Agyemang Farm",
    "registration_id": None,
    "tax_id": None,
    "duns": None,
    "country_iso": "GH",
    "admin_region": "Ashanti",
    "gps_lat": 6.9520,
    "gps_lon": -1.9820,
    "address": None,
    "commodities": ["cocoa"],
    "tier": 5,
    "role": "farmer",
    "annual_volume_mt": 5.0,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 0,
    "primary_contact": "Kwaku Agyemang",
    "compliance_contact": None,
    "dds_references": [],
    "certifications": [],
    "status": "active",
}

COCOA_FARMER_2_GH: Dict[str, Any] = {
    "supplier_id": SUP_ID_COCOA_FARMER_2_GH,
    "legal_name": "Akua Sarpong Farm",
    "registration_id": None,
    "tax_id": None,
    "duns": None,
    "country_iso": "GH",
    "admin_region": "Ashanti",
    "gps_lat": 6.9550,
    "gps_lon": -1.9850,
    "address": None,
    "commodities": ["cocoa"],
    "tier": 5,
    "role": "farmer",
    "annual_volume_mt": 3.5,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 0,
    "primary_contact": "Akua Sarpong",
    "compliance_contact": None,
    "dds_references": [],
    "certifications": [],
    "status": "active",
}

COFFEE_IMPORTER_DE: Dict[str, Any] = {
    "supplier_id": SUP_ID_COFFEE_IMPORTER_DE,
    "legal_name": "Kaffee Europa AG",
    "registration_id": "DE-HRB-654321",
    "tax_id": "DE654321987",
    "duns": "456789012",
    "country_iso": "DE",
    "admin_region": "Bremen",
    "gps_lat": 53.0793,
    "gps_lon": 8.8017,
    "address": "Ueberseestadt 5, 28195 Bremen, Germany",
    "commodities": ["coffee"],
    "tier": 0,
    "role": "importer",
    "annual_volume_mt": 25000.0,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 2,
    "primary_contact": "Karl Schulz",
    "compliance_contact": "Maria Weber",
    "dds_references": ["DDS-EU-2026-003456"],
    "certifications": [],
    "status": "active",
}

COFFEE_EXPORTER_CO: Dict[str, Any] = {
    "supplier_id": SUP_ID_COFFEE_EXPORTER_CO,
    "legal_name": "Exportadora Colombiana de Cafe SA",
    "registration_id": "CO-NIT-900123456",
    "tax_id": "CO900123456",
    "duns": "567890123",
    "country_iso": "CO",
    "admin_region": "Antioquia",
    "gps_lat": 6.2442,
    "gps_lon": -75.5812,
    "address": "Carrera 50, Medellin, Colombia",
    "commodities": ["coffee"],
    "tier": 1,
    "role": "exporter",
    "annual_volume_mt": 15000.0,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 4,
    "primary_contact": "Carlos Restrepo",
    "compliance_contact": "Ana Garcia",
    "dds_references": [],
    "certifications": ["FAIRTRADE-CO-2025-001"],
    "status": "active",
}

COFFEE_MILL_CO: Dict[str, Any] = {
    "supplier_id": SUP_ID_COFFEE_MILL_CO,
    "legal_name": "Beneficio Santa Rosa SAS",
    "registration_id": "CO-NIT-900234567",
    "tax_id": "CO900234567",
    "duns": None,
    "country_iso": "CO",
    "admin_region": "Caldas",
    "gps_lat": 5.0689,
    "gps_lon": -75.5174,
    "address": "Chinchina, Caldas, Colombia",
    "commodities": ["coffee"],
    "tier": 2,
    "role": "mill",
    "annual_volume_mt": 5000.0,
    "processing_capacity_mt": 6000.0,
    "upstream_supplier_count": 3,
    "primary_contact": "Diego Herrera",
    "compliance_contact": None,
    "dds_references": [],
    "certifications": ["4C-CO-2025-001"],
    "status": "active",
}

COFFEE_COOPERATIVE_CO: Dict[str, Any] = {
    "supplier_id": SUP_ID_COFFEE_COOPERATIVE_CO,
    "legal_name": "Cooperativa de Caficultores de Caldas",
    "registration_id": "CO-COOP-567890",
    "tax_id": None,
    "duns": None,
    "country_iso": "CO",
    "admin_region": "Caldas",
    "gps_lat": 5.0500,
    "gps_lon": -75.5000,
    "address": "Manizales, Caldas, Colombia",
    "commodities": ["coffee"],
    "tier": 3,
    "role": "cooperative",
    "annual_volume_mt": 2000.0,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 150,
    "primary_contact": "Miguel Torres",
    "compliance_contact": None,
    "dds_references": [],
    "certifications": ["FAIRTRADE-CO-2025-002"],
    "status": "active",
}

COFFEE_FARMER_CO: Dict[str, Any] = {
    "supplier_id": SUP_ID_COFFEE_FARMER_CO,
    "legal_name": "Finca La Esperanza",
    "registration_id": None,
    "tax_id": None,
    "duns": None,
    "country_iso": "CO",
    "admin_region": "Caldas",
    "gps_lat": 5.0550,
    "gps_lon": -75.5050,
    "address": None,
    "commodities": ["coffee"],
    "tier": 4,
    "role": "farmer",
    "annual_volume_mt": 8.0,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 0,
    "primary_contact": "Jose Ramirez",
    "compliance_contact": None,
    "dds_references": [],
    "certifications": [],
    "status": "active",
}

PALM_IMPORTER_NL: Dict[str, Any] = {
    "supplier_id": SUP_ID_PALM_IMPORTER_NL,
    "legal_name": "PalmOil Europe BV",
    "registration_id": "NL-KVK-12345678",
    "tax_id": "NL123456789B01",
    "duns": "678901234",
    "country_iso": "NL",
    "admin_region": "Zuid-Holland",
    "gps_lat": 51.9225,
    "gps_lon": 4.4792,
    "address": "Europoort, Rotterdam, Netherlands",
    "commodities": ["palm_oil"],
    "tier": 0,
    "role": "importer",
    "annual_volume_mt": 100000.0,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 5,
    "primary_contact": "Jan de Vries",
    "compliance_contact": "Sophie Bakker",
    "dds_references": ["DDS-EU-2026-004567"],
    "certifications": [],
    "status": "active",
}

PALM_REFINERY_ID: Dict[str, Any] = {
    "supplier_id": SUP_ID_PALM_REFINERY_ID,
    "legal_name": "PT Sawit Refinery Indonesia",
    "registration_id": "ID-NIB-1234567890",
    "tax_id": "ID1234567890",
    "duns": "789012345",
    "country_iso": "ID",
    "admin_region": "Kalimantan Barat",
    "gps_lat": -0.0263,
    "gps_lon": 109.3425,
    "address": "Pontianak, West Kalimantan, Indonesia",
    "commodities": ["palm_oil"],
    "tier": 1,
    "role": "refinery",
    "annual_volume_mt": 80000.0,
    "processing_capacity_mt": 100000.0,
    "upstream_supplier_count": 12,
    "primary_contact": "Budi Santoso",
    "compliance_contact": "Sri Wahyuni",
    "dds_references": [],
    "certifications": ["RSPO-ID-2025-001"],
    "status": "active",
}

PALM_MILL_ID: Dict[str, Any] = {
    "supplier_id": SUP_ID_PALM_MILL_ID,
    "legal_name": "PT Kelapa Sawit Mill",
    "registration_id": "ID-NIB-2345678901",
    "tax_id": "ID2345678901",
    "duns": None,
    "country_iso": "ID",
    "admin_region": "Kalimantan Barat",
    "gps_lat": -0.5000,
    "gps_lon": 109.5000,
    "address": "Ketapang, West Kalimantan, Indonesia",
    "commodities": ["palm_oil"],
    "tier": 2,
    "role": "mill",
    "annual_volume_mt": 20000.0,
    "processing_capacity_mt": 25000.0,
    "upstream_supplier_count": 50,
    "primary_contact": "Agus Wijaya",
    "compliance_contact": None,
    "dds_references": [],
    "certifications": ["RSPO-ID-2025-002"],
    "status": "active",
}

PALM_SMALLHOLDER_ID: Dict[str, Any] = {
    "supplier_id": SUP_ID_PALM_SMALLHOLDER_ID,
    "legal_name": "Pak Hasan Palm Plot",
    "registration_id": None,
    "tax_id": None,
    "duns": None,
    "country_iso": "ID",
    "admin_region": "Kalimantan Barat",
    "gps_lat": -0.5200,
    "gps_lon": 109.5200,
    "address": None,
    "commodities": ["palm_oil"],
    "tier": 3,
    "role": "smallholder",
    "annual_volume_mt": 10.0,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 0,
    "primary_contact": "Hasan",
    "compliance_contact": None,
    "dds_references": [],
    "certifications": [],
    "status": "active",
}

SOYA_TRADER_BR: Dict[str, Any] = {
    "supplier_id": SUP_ID_SOYA_TRADER_BR,
    "legal_name": "Agro Soja Comercio Ltda",
    "registration_id": "BR-CNPJ-12345678000100",
    "tax_id": "BR12345678000100",
    "duns": "890123456",
    "country_iso": "BR",
    "admin_region": "Mato Grosso",
    "gps_lat": -12.9700,
    "gps_lon": -55.3200,
    "address": "Sorriso, Mato Grosso, Brazil",
    "commodities": ["soya"],
    "tier": 1,
    "role": "trader",
    "annual_volume_mt": 200000.0,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 30,
    "primary_contact": "Ricardo Silva",
    "compliance_contact": "Fernanda Costa",
    "dds_references": [],
    "certifications": [],
    "status": "active",
}

RUBBER_DEALER_TH: Dict[str, Any] = {
    "supplier_id": SUP_ID_RUBBER_DEALER_TH,
    "legal_name": "Thai Rubber Dealers Co Ltd",
    "registration_id": "TH-DBD-0123456789",
    "tax_id": "TH0123456789",
    "duns": "901234567",
    "country_iso": "TH",
    "admin_region": "Surat Thani",
    "gps_lat": 9.1382,
    "gps_lon": 99.3217,
    "address": "Surat Thani, Thailand",
    "commodities": ["rubber"],
    "tier": 2,
    "role": "dealer",
    "annual_volume_mt": 15000.0,
    "processing_capacity_mt": None,
    "upstream_supplier_count": 100,
    "primary_contact": "Somchai Phanich",
    "compliance_contact": None,
    "dds_references": [],
    "certifications": [],
    "status": "active",
}

CATTLE_FEEDLOT_BR: Dict[str, Any] = {
    "supplier_id": SUP_ID_CATTLE_FEEDLOT_BR,
    "legal_name": "Pecuaria Centro-Oeste Ltda",
    "registration_id": "BR-CNPJ-98765432000199",
    "tax_id": "BR98765432000199",
    "duns": None,
    "country_iso": "BR",
    "admin_region": "Goias",
    "gps_lat": -15.7800,
    "gps_lon": -47.9300,
    "address": "Goiania, Goias, Brazil",
    "commodities": ["cattle"],
    "tier": 2,
    "role": "feedlot",
    "annual_volume_mt": 5000.0,
    "processing_capacity_mt": 8000.0,
    "upstream_supplier_count": 20,
    "primary_contact": "Pedro Oliveira",
    "compliance_contact": None,
    "dds_references": [],
    "certifications": [],
    "status": "active",
}

TIMBER_SAWMILL_CD: Dict[str, Any] = {
    "supplier_id": SUP_ID_TIMBER_SAWMILL_CD,
    "legal_name": "Scierie du Congo SARL",
    "registration_id": "CD-RCCM-KIN-2024-001",
    "tax_id": None,
    "duns": None,
    "country_iso": "CD",
    "admin_region": "Equateur",
    "gps_lat": -4.3220,
    "gps_lon": 15.3130,
    "address": "Mbandaka, Equateur, DRC",
    "commodities": ["wood"],
    "tier": 2,
    "role": "sawmill",
    "annual_volume_mt": 8000.0,
    "processing_capacity_mt": 10000.0,
    "upstream_supplier_count": 10,
    "primary_contact": "Jean-Pierre Mbuyi",
    "compliance_contact": None,
    "dds_references": [],
    "certifications": ["FSC-CD-2024-001"],
    "status": "active",
}

ALL_SAMPLE_SUPPLIERS: List[Dict[str, Any]] = [
    COCOA_IMPORTER_EU, COCOA_TRADER_GH, COCOA_PROCESSOR_GH,
    COCOA_AGGREGATOR_GH, COCOA_COOPERATIVE_GH,
    COCOA_FARMER_1_GH, COCOA_FARMER_2_GH,
    COFFEE_IMPORTER_DE, COFFEE_EXPORTER_CO, COFFEE_MILL_CO,
    COFFEE_COOPERATIVE_CO, COFFEE_FARMER_CO,
    PALM_IMPORTER_NL, PALM_REFINERY_ID, PALM_MILL_ID, PALM_SMALLHOLDER_ID,
    SOYA_TRADER_BR, RUBBER_DEALER_TH, CATTLE_FEEDLOT_BR, TIMBER_SAWMILL_CD,
]


# ---------------------------------------------------------------------------
# Sample Relationships (buyer -> supplier pairs)
# ---------------------------------------------------------------------------

REL_ID_COCOA_IMP_TO_TRADER = "REL-COCOA-001"
REL_ID_COCOA_TRADER_TO_PROC = "REL-COCOA-002"
REL_ID_COCOA_PROC_TO_AGG = "REL-COCOA-003"
REL_ID_COCOA_AGG_TO_COOP = "REL-COCOA-004"
REL_ID_COCOA_COOP_TO_FM1 = "REL-COCOA-005"
REL_ID_COCOA_COOP_TO_FM2 = "REL-COCOA-006"

REL_ID_COFFEE_IMP_TO_EXP = "REL-COFFEE-001"
REL_ID_COFFEE_EXP_TO_MILL = "REL-COFFEE-002"
REL_ID_COFFEE_MILL_TO_COOP = "REL-COFFEE-003"
REL_ID_COFFEE_COOP_TO_FM = "REL-COFFEE-004"

REL_ID_PALM_IMP_TO_REF = "REL-PALM-001"
REL_ID_PALM_REF_TO_MILL = "REL-PALM-002"
REL_ID_PALM_MILL_TO_SH = "REL-PALM-003"


def _make_relationship(
    rel_id: str,
    buyer_id: str,
    supplier_id: str,
    commodity: str,
    state: str = "active",
    volume_mt: float = 1000.0,
    start_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a relationship record dictionary."""
    return {
        "relationship_id": rel_id,
        "buyer_id": buyer_id,
        "supplier_id": supplier_id,
        "commodity": commodity,
        "state": state,
        "volume_mt": volume_mt,
        "frequency": "monthly",
        "exclusivity": False,
        "start_date": (start_date or datetime(2025, 1, 1, tzinfo=timezone.utc)).isoformat(),
        "end_date": None,
        "reason_code": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


SAMPLE_RELATIONSHIPS: List[Dict[str, Any]] = [
    _make_relationship(REL_ID_COCOA_IMP_TO_TRADER, SUP_ID_COCOA_IMPORTER_EU,
                       SUP_ID_COCOA_TRADER_GH, "cocoa", volume_mt=15000.0),
    _make_relationship(REL_ID_COCOA_TRADER_TO_PROC, SUP_ID_COCOA_TRADER_GH,
                       SUP_ID_COCOA_PROCESSOR_GH, "cocoa", volume_mt=10000.0),
    _make_relationship(REL_ID_COCOA_PROC_TO_AGG, SUP_ID_COCOA_PROCESSOR_GH,
                       SUP_ID_COCOA_AGGREGATOR_GH, "cocoa", volume_mt=5000.0),
    _make_relationship(REL_ID_COCOA_AGG_TO_COOP, SUP_ID_COCOA_AGGREGATOR_GH,
                       SUP_ID_COCOA_COOPERATIVE_GH, "cocoa", volume_mt=2000.0),
    _make_relationship(REL_ID_COCOA_COOP_TO_FM1, SUP_ID_COCOA_COOPERATIVE_GH,
                       SUP_ID_COCOA_FARMER_1_GH, "cocoa", volume_mt=5.0),
    _make_relationship(REL_ID_COCOA_COOP_TO_FM2, SUP_ID_COCOA_COOPERATIVE_GH,
                       SUP_ID_COCOA_FARMER_2_GH, "cocoa", volume_mt=3.5),
    _make_relationship(REL_ID_COFFEE_IMP_TO_EXP, SUP_ID_COFFEE_IMPORTER_DE,
                       SUP_ID_COFFEE_EXPORTER_CO, "coffee", volume_mt=8000.0),
    _make_relationship(REL_ID_COFFEE_EXP_TO_MILL, SUP_ID_COFFEE_EXPORTER_CO,
                       SUP_ID_COFFEE_MILL_CO, "coffee", volume_mt=4000.0),
    _make_relationship(REL_ID_COFFEE_MILL_TO_COOP, SUP_ID_COFFEE_MILL_CO,
                       SUP_ID_COFFEE_COOPERATIVE_CO, "coffee", volume_mt=2000.0),
    _make_relationship(REL_ID_COFFEE_COOP_TO_FM, SUP_ID_COFFEE_COOPERATIVE_CO,
                       SUP_ID_COFFEE_FARMER_CO, "coffee", volume_mt=8.0),
    _make_relationship(REL_ID_PALM_IMP_TO_REF, SUP_ID_PALM_IMPORTER_NL,
                       SUP_ID_PALM_REFINERY_ID, "palm_oil", volume_mt=40000.0),
    _make_relationship(REL_ID_PALM_REF_TO_MILL, SUP_ID_PALM_REFINERY_ID,
                       SUP_ID_PALM_MILL_ID, "palm_oil", volume_mt=15000.0),
    _make_relationship(REL_ID_PALM_MILL_TO_SH, SUP_ID_PALM_MILL_ID,
                       SUP_ID_PALM_SMALLHOLDER_ID, "palm_oil", volume_mt=10.0),
]


# ---------------------------------------------------------------------------
# Sample Certification Records
# ---------------------------------------------------------------------------

def _make_certification(
    cert_id: str,
    supplier_id: str,
    cert_type: str,
    status: str = "valid",
    issue_date: Optional[datetime] = None,
    expiry_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a certification record dictionary."""
    now = datetime.now(timezone.utc)
    return {
        "certification_id": cert_id,
        "supplier_id": supplier_id,
        "certification_type": cert_type,
        "certificate_number": f"{cert_type}-{cert_id}",
        "status": status,
        "issue_date": (issue_date or now - timedelta(days=365)).isoformat(),
        "expiry_date": (expiry_date or now + timedelta(days=365)).isoformat(),
        "issuing_body": f"{cert_type} International",
        "scope": "supply_chain",
        "verified": True,
    }


SAMPLE_CERTIFICATIONS: List[Dict[str, Any]] = [
    _make_certification("CERT-001", SUP_ID_COCOA_TRADER_GH, "UTZ"),
    _make_certification("CERT-002", SUP_ID_COCOA_PROCESSOR_GH, "UTZ"),
    _make_certification("CERT-003", SUP_ID_COFFEE_EXPORTER_CO, "FAIRTRADE"),
    _make_certification("CERT-004", SUP_ID_COFFEE_MILL_CO, "4C"),
    _make_certification("CERT-005", SUP_ID_COFFEE_COOPERATIVE_CO, "FAIRTRADE"),
    _make_certification("CERT-006", SUP_ID_PALM_REFINERY_ID, "RSPO"),
    _make_certification("CERT-007", SUP_ID_PALM_MILL_ID, "RSPO"),
    _make_certification("CERT-008", SUP_ID_TIMBER_SAWMILL_CD, "FSC"),
    _make_certification(
        "CERT-009", SUP_ID_SOYA_TRADER_BR, "ISO_14001",
        status="expired",
        issue_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        expiry_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    ),
    _make_certification(
        "CERT-010", SUP_ID_RUBBER_DEALER_TH, "PEFC",
        status="revoked",
        issue_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        expiry_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ),
]


# ---------------------------------------------------------------------------
# Sample Supply Chain Hierarchies
# ---------------------------------------------------------------------------

COCOA_CHAIN_7_TIER: List[Dict[str, Any]] = [
    {"tier": 0, "supplier_id": SUP_ID_COCOA_IMPORTER_EU, "role": "importer"},
    {"tier": 1, "supplier_id": SUP_ID_COCOA_TRADER_GH, "role": "trader"},
    {"tier": 2, "supplier_id": SUP_ID_COCOA_PROCESSOR_GH, "role": "processor"},
    {"tier": 3, "supplier_id": SUP_ID_COCOA_AGGREGATOR_GH, "role": "aggregator"},
    {"tier": 4, "supplier_id": SUP_ID_COCOA_COOPERATIVE_GH, "role": "cooperative"},
    {"tier": 5, "supplier_id": SUP_ID_COCOA_FARMER_1_GH, "role": "farmer"},
    {"tier": 6, "supplier_id": SUP_ID_COCOA_FARMER_2_GH, "role": "farmer"},
]

COFFEE_CHAIN_6_TIER: List[Dict[str, Any]] = [
    {"tier": 0, "supplier_id": SUP_ID_COFFEE_IMPORTER_DE, "role": "importer"},
    {"tier": 1, "supplier_id": SUP_ID_COFFEE_EXPORTER_CO, "role": "exporter"},
    {"tier": 2, "supplier_id": SUP_ID_COFFEE_MILL_CO, "role": "mill"},
    {"tier": 3, "supplier_id": SUP_ID_COFFEE_COOPERATIVE_CO, "role": "cooperative"},
    {"tier": 4, "supplier_id": SUP_ID_COFFEE_FARMER_CO, "role": "farmer"},
]

PALM_OIL_CHAIN_6_TIER: List[Dict[str, Any]] = [
    {"tier": 0, "supplier_id": SUP_ID_PALM_IMPORTER_NL, "role": "importer"},
    {"tier": 1, "supplier_id": SUP_ID_PALM_REFINERY_ID, "role": "refinery"},
    {"tier": 2, "supplier_id": SUP_ID_PALM_MILL_ID, "role": "mill"},
    {"tier": 3, "supplier_id": SUP_ID_PALM_SMALLHOLDER_ID, "role": "smallholder"},
]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def make_supplier(
    supplier_id: Optional[str] = None,
    legal_name: str = "Test Supplier",
    country_iso: str = "GH",
    tier: int = 1,
    role: str = "trader",
    commodity: str = "cocoa",
    gps_lat: Optional[float] = 5.6037,
    gps_lon: Optional[float] = -0.1870,
    registration_id: Optional[str] = None,
    tax_id: Optional[str] = None,
    certifications: Optional[List[str]] = None,
    dds_references: Optional[List[str]] = None,
    status: str = "active",
    primary_contact: Optional[str] = "Test Contact",
    compliance_contact: Optional[str] = None,
    annual_volume_mt: float = 100.0,
) -> Dict[str, Any]:
    """Build a supplier profile dictionary for testing.

    Args:
        supplier_id: Unique supplier ID (auto-generated if None).
        legal_name: Legal entity name.
        country_iso: ISO 3166-1 alpha-2 country code.
        tier: Tier level in supply chain (0 = operator/importer).
        role: Supplier role (farmer, cooperative, trader, etc.).
        commodity: EUDR commodity type.
        gps_lat: Latitude in decimal degrees.
        gps_lon: Longitude in decimal degrees.
        registration_id: Business registration ID.
        tax_id: Tax identification number.
        certifications: List of certification IDs.
        dds_references: List of DDS reference IDs.
        status: Supplier status (active, suspended, etc.).
        primary_contact: Primary contact name.
        compliance_contact: Compliance contact name.
        annual_volume_mt: Annual volume in metric tonnes.

    Returns:
        Dict with all supplier profile fields.
    """
    return {
        "supplier_id": supplier_id or f"SUP-{uuid.uuid4().hex[:12].upper()}",
        "legal_name": legal_name,
        "registration_id": registration_id,
        "tax_id": tax_id,
        "duns": None,
        "country_iso": country_iso,
        "admin_region": None,
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "address": None,
        "commodities": [commodity],
        "tier": tier,
        "role": role,
        "annual_volume_mt": annual_volume_mt,
        "processing_capacity_mt": None,
        "upstream_supplier_count": 0,
        "primary_contact": primary_contact,
        "compliance_contact": compliance_contact,
        "dds_references": dds_references or [],
        "certifications": certifications or [],
        "status": status,
    }


def make_relationship(
    buyer_id: str,
    supplier_id: str,
    commodity: str = "cocoa",
    state: str = "active",
    volume_mt: float = 1000.0,
    rel_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a relationship record dictionary for testing.

    Args:
        buyer_id: Buyer supplier ID.
        supplier_id: Seller supplier ID.
        commodity: EUDR commodity type.
        state: Relationship state.
        volume_mt: Transaction volume in metric tonnes.
        rel_id: Relationship ID (auto-generated if None).
        start_date: Relationship start date.
        end_date: Relationship end date.

    Returns:
        Dict with relationship fields.
    """
    return {
        "relationship_id": rel_id or f"REL-{uuid.uuid4().hex[:12].upper()}",
        "buyer_id": buyer_id,
        "supplier_id": supplier_id,
        "commodity": commodity,
        "state": state,
        "volume_mt": volume_mt,
        "frequency": "monthly",
        "exclusivity": False,
        "start_date": (start_date or datetime(2025, 1, 1, tzinfo=timezone.utc)).isoformat(),
        "end_date": end_date.isoformat() if end_date else None,
        "reason_code": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def make_cert(
    supplier_id: str,
    cert_type: str = "FSC",
    status: str = "valid",
    days_until_expiry: int = 365,
    cert_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a certification record dictionary for testing.

    Args:
        supplier_id: Associated supplier ID.
        cert_type: Certification type (FSC, RSPO, UTZ, etc.).
        status: Certification status (valid, expired, revoked).
        days_until_expiry: Days from now until expiry.
        cert_id: Certification ID (auto-generated if None).

    Returns:
        Dict with certification fields.
    """
    now = datetime.now(timezone.utc)
    return {
        "certification_id": cert_id or f"CERT-{uuid.uuid4().hex[:8].upper()}",
        "supplier_id": supplier_id,
        "certification_type": cert_type,
        "certificate_number": f"{cert_type}-{uuid.uuid4().hex[:8]}",
        "status": status,
        "issue_date": (now - timedelta(days=365)).isoformat(),
        "expiry_date": (now + timedelta(days=days_until_expiry)).isoformat(),
        "issuing_body": f"{cert_type} International",
        "scope": "supply_chain",
        "verified": status == "valid",
    }


def assert_valid_risk_score(score: float, min_val: float = 0.0, max_val: float = 100.0) -> None:
    """Assert that a risk score is within the valid range [0, 100].

    Args:
        score: The risk score to validate.
        min_val: Minimum allowed value (default 0.0).
        max_val: Maximum allowed value (default 100.0).

    Raises:
        AssertionError: If score is outside bounds.
    """
    assert isinstance(score, (int, float)), f"Risk score must be numeric, got {type(score)}"
    assert min_val <= score <= max_val, (
        f"Risk score {score} out of bounds [{min_val}, {max_val}]"
    )


def assert_valid_compliance_score(score: float) -> None:
    """Assert that a compliance score is within [0, 100]."""
    assert isinstance(score, (int, float)), f"Compliance score must be numeric, got {type(score)}"
    assert 0.0 <= score <= 100.0, f"Compliance score {score} out of bounds [0, 100]"


def assert_valid_completeness_score(score: float) -> None:
    """Assert that a completeness score is within [0, 100]."""
    assert isinstance(score, (int, float)), f"Completeness score must be numeric, got {type(score)}"
    assert 0.0 <= score <= 100.0, f"Completeness score {score} out of bounds [0, 100]"


def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash of data for provenance verification.

    Args:
        data: Data to hash (will be JSON-serialized).

    Returns:
        64-character hex digest string.
    """
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_linear_chain(
    commodity: str,
    tier_count: int,
    country_iso: str = "GH",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build a linear supply chain with N tiers for testing.

    Args:
        commodity: EUDR commodity type.
        tier_count: Number of tiers to create.
        country_iso: Country for all suppliers.

    Returns:
        Tuple of (suppliers_list, relationships_list).
    """
    roles = ["importer", "trader", "processor", "aggregator", "cooperative",
             "farmer", "farmer", "farmer", "farmer", "farmer",
             "farmer", "farmer", "farmer", "farmer", "farmer"]
    suppliers = []
    relationships = []
    for t in range(tier_count):
        sup = make_supplier(
            supplier_id=f"SUP-CHAIN-{commodity}-T{t}",
            legal_name=f"{commodity.title()} Tier-{t} Supplier",
            country_iso="DE" if t == 0 else country_iso,
            tier=t,
            role=roles[min(t, len(roles) - 1)],
            commodity=commodity,
        )
        suppliers.append(sup)
        if t > 0:
            rel = make_relationship(
                buyer_id=suppliers[t - 1]["supplier_id"],
                supplier_id=sup["supplier_id"],
                commodity=commodity,
                rel_id=f"REL-CHAIN-{commodity}-{t - 1}-{t}",
            )
            relationships.append(rel)
    return suppliers, relationships


def build_branching_chain(
    commodity: str = "cocoa",
    branch_factor: int = 2,
    depth: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build a branching supply chain tree for testing.

    Each supplier at tier < depth-1 has `branch_factor` sub-suppliers.

    Args:
        commodity: EUDR commodity type.
        branch_factor: Number of children per node.
        depth: Total depth of tree.

    Returns:
        Tuple of (suppliers_list, relationships_list).
    """
    suppliers: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    counter = 0

    def _add_node(tier: int, parent_id: Optional[str]) -> None:
        nonlocal counter
        node_id = f"SUP-BRANCH-{counter:04d}"
        counter += 1
        sup = make_supplier(
            supplier_id=node_id,
            legal_name=f"Branch Supplier {counter}",
            tier=tier,
            commodity=commodity,
        )
        suppliers.append(sup)
        if parent_id is not None:
            rel = make_relationship(
                buyer_id=parent_id,
                supplier_id=node_id,
                commodity=commodity,
                rel_id=f"REL-BRANCH-{counter:04d}",
            )
            relationships.append(rel)
        if tier < depth - 1:
            for _ in range(branch_factor):
                _add_node(tier + 1, node_id)

    _add_node(0, None)
    return suppliers, relationships


def build_diamond_chain(commodity: str = "cocoa") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build a diamond-shaped chain where two mid-tier suppliers share a sub-supplier.

    Structure:
        Importer -> Trader_A -> Processor -> Farmer
        Importer -> Trader_B -> Processor -> Farmer

    The Processor and Farmer are shared (diamond pattern).

    Args:
        commodity: EUDR commodity type.

    Returns:
        Tuple of (suppliers_list, relationships_list).
    """
    importer = make_supplier(supplier_id="SUP-DIAMOND-IMP", tier=0,
                             role="importer", commodity=commodity, legal_name="Diamond Importer")
    trader_a = make_supplier(supplier_id="SUP-DIAMOND-TRD-A", tier=1,
                             role="trader", commodity=commodity, legal_name="Diamond Trader A")
    trader_b = make_supplier(supplier_id="SUP-DIAMOND-TRD-B", tier=1,
                             role="trader", commodity=commodity, legal_name="Diamond Trader B")
    processor = make_supplier(supplier_id="SUP-DIAMOND-PRC", tier=2,
                              role="processor", commodity=commodity, legal_name="Diamond Processor")
    farmer = make_supplier(supplier_id="SUP-DIAMOND-FRM", tier=3,
                           role="farmer", commodity=commodity, legal_name="Diamond Farmer")

    suppliers = [importer, trader_a, trader_b, processor, farmer]
    relationships = [
        make_relationship("SUP-DIAMOND-IMP", "SUP-DIAMOND-TRD-A", commodity, rel_id="REL-DIAM-01"),
        make_relationship("SUP-DIAMOND-IMP", "SUP-DIAMOND-TRD-B", commodity, rel_id="REL-DIAM-02"),
        make_relationship("SUP-DIAMOND-TRD-A", "SUP-DIAMOND-PRC", commodity, rel_id="REL-DIAM-03"),
        make_relationship("SUP-DIAMOND-TRD-B", "SUP-DIAMOND-PRC", commodity, rel_id="REL-DIAM-04"),
        make_relationship("SUP-DIAMOND-PRC", "SUP-DIAMOND-FRM", commodity, rel_id="REL-DIAM-05"),
    ]
    return suppliers, relationships


# ---------------------------------------------------------------------------
# Configuration Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> Dict[str, Any]:
    """Create a MultiTierSupplierConfig-compatible dictionary with test defaults."""
    return {
        "database_url": "postgresql://localhost:5432/greenlang_test",
        "redis_url": "redis://localhost:6379/1",
        "log_level": "DEBUG",
        "max_tier_depth": 15,
        "max_batch_size": 10_000,
        "enable_provenance": True,
        "genesis_hash": "GL-EUDR-MST-008-TEST-GENESIS",
        "enable_metrics": False,
        "pool_size": 5,
        "discovery_confidence_threshold": 0.6,
        "risk_alert_threshold": 70.0,
        "compliance_alert_days": [30, 14, 7],
        "dedup_fuzzy_threshold": 0.85,
        "risk_category_weights": dict(RISK_CATEGORY_WEIGHTS),
        "profile_completeness_weights": dict(PROFILE_COMPLETENESS_WEIGHTS),
    }


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset any singleton state between tests to prevent cross-test contamination."""
    yield
    # Engines may use singletons; reset them after each test
    try:
        from greenlang.agents.eudr.multi_tier_supplier.config import reset_config
        reset_config()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Engine Fixtures (with graceful pytest.skip for unimplemented)
# ---------------------------------------------------------------------------


@pytest.fixture
def supplier_discovery_engine(config):
    """Create a SupplierDiscoveryEngine instance for testing.

    If the engine is not yet implemented, this fixture will skip the test
    with an informative message.
    """
    try:
        from greenlang.agents.eudr.multi_tier_supplier.supplier_discovery_engine import (
            SupplierDiscoveryEngine,
        )
        return SupplierDiscoveryEngine(config=config)
    except ImportError:
        pytest.skip("SupplierDiscoveryEngine not yet implemented")


@pytest.fixture
def supplier_profile_manager(config):
    """Create a SupplierProfileManager instance for testing."""
    try:
        from greenlang.agents.eudr.multi_tier_supplier.supplier_profile_manager import (
            SupplierProfileManager,
        )
        return SupplierProfileManager(config=config)
    except ImportError:
        pytest.skip("SupplierProfileManager not yet implemented")


@pytest.fixture
def tier_depth_tracker(config):
    """Create a TierDepthTracker instance for testing."""
    try:
        from greenlang.agents.eudr.multi_tier_supplier.tier_depth_tracker import (
            TierDepthTracker,
        )
        return TierDepthTracker(config=config)
    except ImportError:
        pytest.skip("TierDepthTracker not yet implemented")


@pytest.fixture
def relationship_manager(config):
    """Create a RelationshipManager instance for testing."""
    try:
        from greenlang.agents.eudr.multi_tier_supplier.relationship_manager import (
            RelationshipManager,
        )
        return RelationshipManager(config=config)
    except ImportError:
        pytest.skip("RelationshipManager not yet implemented")


@pytest.fixture
def risk_propagation_engine(config):
    """Create a RiskPropagationEngine instance for testing."""
    try:
        from greenlang.agents.eudr.multi_tier_supplier.risk_propagation_engine import (
            RiskPropagationEngine,
        )
        return RiskPropagationEngine(config=config)
    except ImportError:
        pytest.skip("RiskPropagationEngine not yet implemented")


@pytest.fixture
def compliance_monitor(config):
    """Create a ComplianceMonitor instance for testing."""
    try:
        from greenlang.agents.eudr.multi_tier_supplier.compliance_monitor import (
            ComplianceMonitor,
        )
        return ComplianceMonitor(config=config)
    except ImportError:
        pytest.skip("ComplianceMonitor not yet implemented")


@pytest.fixture
def gap_analyzer(config):
    """Create a GapAnalyzer instance for testing."""
    try:
        from greenlang.agents.eudr.multi_tier_supplier.gap_analyzer import (
            GapAnalyzer,
        )
        return GapAnalyzer(config=config)
    except ImportError:
        pytest.skip("GapAnalyzer not yet implemented")


@pytest.fixture
def audit_reporter(config):
    """Create an AuditReporter instance for testing."""
    try:
        from greenlang.agents.eudr.multi_tier_supplier.audit_reporter import (
            AuditReporter,
        )
        return AuditReporter(config=config)
    except ImportError:
        pytest.skip("AuditReporter not yet implemented")


@pytest.fixture
def service(config):
    """Create the top-level MultiTierSupplierService facade for testing."""
    try:
        from greenlang.agents.eudr.multi_tier_supplier.setup import (
            MultiTierSupplierService,
        )
        return MultiTierSupplierService(config=config)
    except ImportError:
        pytest.skip("MultiTierSupplierService not yet implemented")


# ---------------------------------------------------------------------------
# Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_suppliers() -> List[Dict[str, Any]]:
    """Return all 20+ sample supplier profiles."""
    return list(ALL_SAMPLE_SUPPLIERS)


@pytest.fixture
def cocoa_chain() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return the 7-tier cocoa supply chain (suppliers and relationships)."""
    cocoa_suppliers = [
        COCOA_IMPORTER_EU, COCOA_TRADER_GH, COCOA_PROCESSOR_GH,
        COCOA_AGGREGATOR_GH, COCOA_COOPERATIVE_GH,
        COCOA_FARMER_1_GH, COCOA_FARMER_2_GH,
    ]
    cocoa_rels = [r for r in SAMPLE_RELATIONSHIPS if r["commodity"] == "cocoa"]
    return cocoa_suppliers, cocoa_rels


@pytest.fixture
def coffee_chain() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return the 5-tier coffee supply chain (suppliers and relationships)."""
    coffee_suppliers = [
        COFFEE_IMPORTER_DE, COFFEE_EXPORTER_CO, COFFEE_MILL_CO,
        COFFEE_COOPERATIVE_CO, COFFEE_FARMER_CO,
    ]
    coffee_rels = [r for r in SAMPLE_RELATIONSHIPS if r["commodity"] == "coffee"]
    return coffee_suppliers, coffee_rels


@pytest.fixture
def palm_chain() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return the 4-tier palm oil supply chain (suppliers and relationships)."""
    palm_suppliers = [
        PALM_IMPORTER_NL, PALM_REFINERY_ID, PALM_MILL_ID, PALM_SMALLHOLDER_ID,
    ]
    palm_rels = [r for r in SAMPLE_RELATIONSHIPS if r["commodity"] == "palm_oil"]
    return palm_suppliers, palm_rels


@pytest.fixture
def sample_certifications() -> List[Dict[str, Any]]:
    """Return all sample certification records."""
    return list(SAMPLE_CERTIFICATIONS)


@pytest.fixture
def sample_relationships() -> List[Dict[str, Any]]:
    """Return all sample relationship records."""
    return list(SAMPLE_RELATIONSHIPS)


@pytest.fixture
def diamond_chain() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return a diamond-shaped chain for shared-supplier testing."""
    return build_diamond_chain("cocoa")


@pytest.fixture
def deep_linear_chain() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return a 15-tier linear chain for max-depth testing."""
    return build_linear_chain("cocoa", tier_count=15, country_iso="GH")


@pytest.fixture
def branching_chain() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return a branching chain with factor=2, depth=4 (15 nodes)."""
    return build_branching_chain("cocoa", branch_factor=2, depth=4)
