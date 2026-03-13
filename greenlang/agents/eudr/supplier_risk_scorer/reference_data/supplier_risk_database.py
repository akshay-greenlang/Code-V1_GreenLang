# -*- coding: utf-8 -*-
"""
Supplier Risk Database - AGENT-EUDR-017 Supplier Risk Scorer

Comprehensive supplier risk reference data for EUDR compliance supplier
assessment. Provides sample supplier profiles, risk factor benchmarks,
industry averages, peer group definitions, and non-conformance severity
matrix for deterministic supplier risk scoring.

Data includes:
    - SAMPLE_SUPPLIERS: 30+ sample supplier profiles with supplier_id, name,
      type, country, commodities, and baseline risk indicators
    - RISK_FACTOR_BENCHMARKS: benchmark thresholds for each of the 8 risk
      factors (low/medium/high/critical thresholds 0-100)
    - INDUSTRY_AVERAGES: average risk scores by commodity and region
    - PEER_GROUP_DEFINITIONS: peer group criteria for comparative analysis
    - NON_CONFORMANCE_SEVERITY_MATRIX: issue type to severity mapping

Data Sources:
    - GreenLang Platform Supplier Master Data 2025
    - EUDR Operator Due Diligence Statements 2024-2025
    - Industry benchmarking studies (FSC, RSPO, Rainforest Alliance)
    - European Commission EUDR guidance documents

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SupplierRecord = Dict[str, Any]
RiskFactorBenchmark = Dict[str, Any]
IndustryAverage = Dict[str, float]
PeerGroupDefinition = Dict[str, Any]
SeverityMapping = Dict[str, str]

# ---------------------------------------------------------------------------
# Data version and source metadata
# ---------------------------------------------------------------------------

DATA_VERSION: str = "2025-03"
DATA_SOURCES: List[str] = [
    "GreenLang Platform Supplier Master Data 2025",
    "EUDR Operator Due Diligence Statements 2024-2025",
    "FSC Forest Stewardship Council Benchmarking 2024",
    "RSPO Roundtable on Sustainable Palm Oil Member Data 2024",
    "Rainforest Alliance Certification Database 2024",
    "European Commission EUDR Guidance Documents 2024-2025",
]

# ---------------------------------------------------------------------------
# EUDR commodities
# ---------------------------------------------------------------------------

EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

# ===========================================================================
# Sample Suppliers (30+ profiles)
# ===========================================================================
#
# Each record keys:
#   supplier_id           - Unique identifier (SUP-{ISO3}-{SEQ})
#   name                  - Supplier name
#   type                  - producer | trader | processor | exporter | importer | cooperative
#   country               - ISO 3166-1 alpha-3 country code
#   region                - Geographic region
#   commodities           - List of EUDR commodities produced/traded
#   baseline_risk_score   - Baseline risk score 0-100
#   years_in_operation    - Years in business
#   employee_count        - Number of employees
#   annual_volume_tons    - Annual volume in metric tons
#   certifications        - List of certification schemes held

SAMPLE_SUPPLIERS: Dict[str, SupplierRecord] = {
    # ===================================================================
    # Brazil - Cattle & Soy
    # ===================================================================
    "SUP-BRA-001": {
        "supplier_id": "SUP-BRA-001",
        "name": "Fazenda Esperança Cattle Ranch",
        "type": "producer",
        "country": "BRA",
        "region": "south_america",
        "commodities": ["cattle"],
        "baseline_risk_score": 68,
        "years_in_operation": 15,
        "employee_count": 120,
        "annual_volume_tons": 2400,
        "certifications": [],
    },
    "SUP-BRA-002": {
        "supplier_id": "SUP-BRA-002",
        "name": "Cooperativa AgriSoja do Cerrado",
        "type": "cooperative",
        "country": "BRA",
        "region": "south_america",
        "commodities": ["soya"],
        "baseline_risk_score": 55,
        "years_in_operation": 22,
        "employee_count": 45,
        "annual_volume_tons": 18000,
        "certifications": ["RTRS", "ORGANIC"],
    },
    "SUP-BRA-003": {
        "supplier_id": "SUP-BRA-003",
        "name": "Madeiras Sustentáveis Ltda",
        "type": "producer",
        "country": "BRA",
        "region": "south_america",
        "commodities": ["wood"],
        "baseline_risk_score": 42,
        "years_in_operation": 28,
        "employee_count": 210,
        "annual_volume_tons": 8500,
        "certifications": ["FSC"],
    },
    "SUP-BRA-004": {
        "supplier_id": "SUP-BRA-004",
        "name": "Café Orgânico do Sul",
        "type": "producer",
        "country": "BRA",
        "region": "south_america",
        "commodities": ["coffee"],
        "baseline_risk_score": 38,
        "years_in_operation": 35,
        "employee_count": 80,
        "annual_volume_tons": 650,
        "certifications": ["ORGANIC", "RAINFOREST_ALLIANCE"],
    },

    # ===================================================================
    # Indonesia - Palm Oil & Rubber
    # ===================================================================
    "SUP-IDN-001": {
        "supplier_id": "SUP-IDN-001",
        "name": "PT Sawit Hijau Nusantara",
        "type": "producer",
        "country": "IDN",
        "region": "southeast_asia",
        "commodities": ["oil_palm"],
        "baseline_risk_score": 72,
        "years_in_operation": 18,
        "employee_count": 450,
        "annual_volume_tons": 28000,
        "certifications": [],
    },
    "SUP-IDN-002": {
        "supplier_id": "SUP-IDN-002",
        "name": "Sumatra Sustainable Palm Oil Co.",
        "type": "producer",
        "country": "IDN",
        "region": "southeast_asia",
        "commodities": ["oil_palm"],
        "baseline_risk_score": 48,
        "years_in_operation": 25,
        "employee_count": 680,
        "annual_volume_tons": 42000,
        "certifications": ["RSPO"],
    },
    "SUP-IDN-003": {
        "supplier_id": "SUP-IDN-003",
        "name": "Kalimantan Rubber Traders",
        "type": "trader",
        "country": "IDN",
        "region": "southeast_asia",
        "commodities": ["rubber"],
        "baseline_risk_score": 58,
        "years_in_operation": 12,
        "employee_count": 95,
        "annual_volume_tons": 6500,
        "certifications": [],
    },
    "SUP-IDN-004": {
        "supplier_id": "SUP-IDN-004",
        "name": "Java Coffee & Cocoa Exports",
        "type": "exporter",
        "country": "IDN",
        "region": "southeast_asia",
        "commodities": ["coffee", "cocoa"],
        "baseline_risk_score": 52,
        "years_in_operation": 30,
        "employee_count": 140,
        "annual_volume_tons": 3200,
        "certifications": ["UTZ", "RAINFOREST_ALLIANCE"],
    },

    # ===================================================================
    # Malaysia - Palm Oil
    # ===================================================================
    "SUP-MYS-001": {
        "supplier_id": "SUP-MYS-001",
        "name": "Sabah Palm Plantations Ltd",
        "type": "producer",
        "country": "MYS",
        "region": "southeast_asia",
        "commodities": ["oil_palm"],
        "baseline_risk_score": 62,
        "years_in_operation": 32,
        "employee_count": 850,
        "annual_volume_tons": 55000,
        "certifications": ["RSPO", "MSPO"],
    },
    "SUP-MYS-002": {
        "supplier_id": "SUP-MYS-002",
        "name": "Sarawak Timber Industries",
        "type": "processor",
        "country": "MYS",
        "region": "southeast_asia",
        "commodities": ["wood"],
        "baseline_risk_score": 56,
        "years_in_operation": 40,
        "employee_count": 320,
        "annual_volume_tons": 12000,
        "certifications": ["PEFC", "MTCC"],
    },

    # ===================================================================
    # Côte d'Ivoire - Cocoa
    # ===================================================================
    "SUP-CIV-001": {
        "supplier_id": "SUP-CIV-001",
        "name": "Coopérative Cacaoyers de la Lagune",
        "type": "cooperative",
        "country": "CIV",
        "region": "west_africa",
        "commodities": ["cocoa"],
        "baseline_risk_score": 64,
        "years_in_operation": 18,
        "employee_count": 35,
        "annual_volume_tons": 2800,
        "certifications": [],
    },
    "SUP-CIV-002": {
        "supplier_id": "SUP-CIV-002",
        "name": "Abidjan Cocoa Processing SA",
        "type": "processor",
        "country": "CIV",
        "region": "west_africa",
        "commodities": ["cocoa"],
        "baseline_risk_score": 58,
        "years_in_operation": 25,
        "employee_count": 280,
        "annual_volume_tons": 18000,
        "certifications": ["RAINFOREST_ALLIANCE", "FAIR_TRADE"],
    },

    # ===================================================================
    # Ghana - Cocoa
    # ===================================================================
    "SUP-GHA-001": {
        "supplier_id": "SUP-GHA-001",
        "name": "Ashanti Cocoa Farmers Union",
        "type": "cooperative",
        "country": "GHA",
        "region": "west_africa",
        "commodities": ["cocoa"],
        "baseline_risk_score": 62,
        "years_in_operation": 20,
        "employee_count": 28,
        "annual_volume_tons": 1900,
        "certifications": ["FAIR_TRADE"],
    },
    "SUP-GHA-002": {
        "supplier_id": "SUP-GHA-002",
        "name": "Ghana Sustainable Cocoa Ltd",
        "type": "exporter",
        "country": "GHA",
        "region": "west_africa",
        "commodities": ["cocoa"],
        "baseline_risk_score": 54,
        "years_in_operation": 35,
        "employee_count": 120,
        "annual_volume_tons": 8500,
        "certifications": ["RAINFOREST_ALLIANCE", "UTZ"],
    },

    # ===================================================================
    # Colombia - Coffee
    # ===================================================================
    "SUP-COL-001": {
        "supplier_id": "SUP-COL-001",
        "name": "Federación de Cafeteros de Antioquia",
        "type": "cooperative",
        "country": "COL",
        "region": "south_america",
        "commodities": ["coffee"],
        "baseline_risk_score": 45,
        "years_in_operation": 42,
        "employee_count": 85,
        "annual_volume_tons": 4200,
        "certifications": ["RAINFOREST_ALLIANCE", "ORGANIC"],
    },
    "SUP-COL-002": {
        "supplier_id": "SUP-COL-002",
        "name": "Bogotá Coffee Exporters SA",
        "type": "exporter",
        "country": "COL",
        "region": "south_america",
        "commodities": ["coffee"],
        "baseline_risk_score": 48,
        "years_in_operation": 38,
        "employee_count": 150,
        "annual_volume_tons": 9800,
        "certifications": ["FAIR_TRADE", "UTZ"],
    },

    # ===================================================================
    # Vietnam - Coffee & Rubber
    # ===================================================================
    "SUP-VNM-001": {
        "supplier_id": "SUP-VNM-001",
        "name": "Central Highlands Coffee Cooperative",
        "type": "cooperative",
        "country": "VNM",
        "region": "southeast_asia",
        "commodities": ["coffee"],
        "baseline_risk_score": 52,
        "years_in_operation": 15,
        "employee_count": 42,
        "annual_volume_tons": 3600,
        "certifications": ["UTZ"],
    },
    "SUP-VNM-002": {
        "supplier_id": "SUP-VNM-002",
        "name": "Vietnam Rubber Group",
        "type": "producer",
        "country": "VNM",
        "region": "southeast_asia",
        "commodities": ["rubber"],
        "baseline_risk_score": 56,
        "years_in_operation": 28,
        "employee_count": 380,
        "annual_volume_tons": 15000,
        "certifications": [],
    },

    # ===================================================================
    # Peru - Coffee & Cocoa
    # ===================================================================
    "SUP-PER-001": {
        "supplier_id": "SUP-PER-001",
        "name": "Amazonas Organic Coffee Coop",
        "type": "cooperative",
        "country": "PER",
        "region": "south_america",
        "commodities": ["coffee"],
        "baseline_risk_score": 42,
        "years_in_operation": 18,
        "employee_count": 38,
        "annual_volume_tons": 1850,
        "certifications": ["ORGANIC", "FAIR_TRADE"],
    },
    "SUP-PER-002": {
        "supplier_id": "SUP-PER-002",
        "name": "Cusco Cocoa Producers Ltd",
        "type": "producer",
        "country": "PER",
        "region": "south_america",
        "commodities": ["cocoa"],
        "baseline_risk_score": 48,
        "years_in_operation": 22,
        "employee_count": 95,
        "annual_volume_tons": 2200,
        "certifications": ["ORGANIC", "RAINFOREST_ALLIANCE"],
    },

    # ===================================================================
    # Argentina - Soy & Cattle
    # ===================================================================
    "SUP-ARG-001": {
        "supplier_id": "SUP-ARG-001",
        "name": "Pampas Soja Exportaciones",
        "type": "exporter",
        "country": "ARG",
        "region": "south_america",
        "commodities": ["soya"],
        "baseline_risk_score": 46,
        "years_in_operation": 32,
        "employee_count": 220,
        "annual_volume_tons": 45000,
        "certifications": ["RTRS"],
    },
    "SUP-ARG-002": {
        "supplier_id": "SUP-ARG-002",
        "name": "Ganadería Sustentable Argentina SA",
        "type": "producer",
        "country": "ARG",
        "region": "south_america",
        "commodities": ["cattle"],
        "baseline_risk_score": 52,
        "years_in_operation": 28,
        "employee_count": 180,
        "annual_volume_tons": 3800,
        "certifications": [],
    },

    # ===================================================================
    # Paraguay - Soy & Cattle
    # ===================================================================
    "SUP-PRY-001": {
        "supplier_id": "SUP-PRY-001",
        "name": "Chaco Soy Traders Ltd",
        "type": "trader",
        "country": "PRY",
        "region": "south_america",
        "commodities": ["soya"],
        "baseline_risk_score": 66,
        "years_in_operation": 12,
        "employee_count": 85,
        "annual_volume_tons": 22000,
        "certifications": [],
    },
    "SUP-PRY-002": {
        "supplier_id": "SUP-PRY-002",
        "name": "Asunción Beef Processors",
        "type": "processor",
        "country": "PRY",
        "region": "south_america",
        "commodities": ["cattle"],
        "baseline_risk_score": 62,
        "years_in_operation": 18,
        "employee_count": 340,
        "annual_volume_tons": 4500,
        "certifications": [],
    },

    # ===================================================================
    # Thailand - Rubber
    # ===================================================================
    "SUP-THA-001": {
        "supplier_id": "SUP-THA-001",
        "name": "Thai Rubber Smallholder Coop",
        "type": "cooperative",
        "country": "THA",
        "region": "southeast_asia",
        "commodities": ["rubber"],
        "baseline_risk_score": 50,
        "years_in_operation": 25,
        "employee_count": 52,
        "annual_volume_tons": 5800,
        "certifications": [],
    },
    "SUP-THA-002": {
        "supplier_id": "SUP-THA-002",
        "name": "Bangkok Rubber Exporters Ltd",
        "type": "exporter",
        "country": "THA",
        "region": "southeast_asia",
        "commodities": ["rubber"],
        "baseline_risk_score": 48,
        "years_in_operation": 38,
        "employee_count": 180,
        "annual_volume_tons": 18000,
        "certifications": [],
    },

    # ===================================================================
    # Ecuador - Cocoa & Coffee
    # ===================================================================
    "SUP-ECU-001": {
        "supplier_id": "SUP-ECU-001",
        "name": "Guayaquil Cacao Fino Ltd",
        "type": "producer",
        "country": "ECU",
        "region": "south_america",
        "commodities": ["cocoa"],
        "baseline_risk_score": 44,
        "years_in_operation": 32,
        "employee_count": 120,
        "annual_volume_tons": 2800,
        "certifications": ["ORGANIC", "RAINFOREST_ALLIANCE"],
    },
    "SUP-ECU-002": {
        "supplier_id": "SUP-ECU-002",
        "name": "Quito Coffee Growers Assoc",
        "type": "cooperative",
        "country": "ECU",
        "region": "south_america",
        "commodities": ["coffee"],
        "baseline_risk_score": 46,
        "years_in_operation": 28,
        "employee_count": 45,
        "annual_volume_tons": 1650,
        "certifications": ["FAIR_TRADE", "ORGANIC"],
    },

    # ===================================================================
    # Cameroon - Cocoa & Wood
    # ===================================================================
    "SUP-CMR-001": {
        "supplier_id": "SUP-CMR-001",
        "name": "Douala Cocoa Exporters SA",
        "type": "exporter",
        "country": "CMR",
        "region": "central_africa",
        "commodities": ["cocoa"],
        "baseline_risk_score": 68,
        "years_in_operation": 22,
        "employee_count": 95,
        "annual_volume_tons": 5800,
        "certifications": [],
    },
    "SUP-CMR-002": {
        "supplier_id": "SUP-CMR-002",
        "name": "Cameroon Timber Industries",
        "type": "producer",
        "country": "CMR",
        "region": "central_africa",
        "commodities": ["wood"],
        "baseline_risk_score": 72,
        "years_in_operation": 18,
        "employee_count": 280,
        "annual_volume_tons": 9500,
        "certifications": [],
    },

    # ===================================================================
    # Bolivia - Soy & Cattle
    # ===================================================================
    "SUP-BOL-001": {
        "supplier_id": "SUP-BOL-001",
        "name": "Santa Cruz Soja Exports",
        "type": "exporter",
        "country": "BOL",
        "region": "south_america",
        "commodities": ["soya"],
        "baseline_risk_score": 64,
        "years_in_operation": 15,
        "employee_count": 110,
        "annual_volume_tons": 28000,
        "certifications": [],
    },
    "SUP-BOL-002": {
        "supplier_id": "SUP-BOL-002",
        "name": "Beni Cattle Ranch Cooperative",
        "type": "cooperative",
        "country": "BOL",
        "region": "south_america",
        "commodities": ["cattle"],
        "baseline_risk_score": 70,
        "years_in_operation": 12,
        "employee_count": 65,
        "annual_volume_tons": 1850,
        "certifications": [],
    },
}

# ===========================================================================
# Risk Factor Benchmarks (8 factors)
# ===========================================================================
#
# Each factor has thresholds for low/medium/high/critical risk levels.
# Scores are 0-100 (higher = riskier).

RISK_FACTOR_BENCHMARKS: Dict[str, RiskFactorBenchmark] = {
    "geographic_sourcing": {
        "factor": "geographic_sourcing",
        "description": "Risk based on country risk, concentration, high-risk zones",
        "low_threshold": 25,
        "medium_threshold": 50,
        "high_threshold": 75,
        "critical_threshold": 90,
        "weight": 0.20,
    },
    "compliance_history": {
        "factor": "compliance_history",
        "description": "Non-conformances, violations, corrective actions",
        "low_threshold": 25,
        "medium_threshold": 50,
        "high_threshold": 75,
        "critical_threshold": 90,
        "weight": 0.15,
    },
    "documentation_quality": {
        "factor": "documentation_quality",
        "description": "EUDR documentation completeness, quality, expiry",
        "low_threshold": 25,
        "medium_threshold": 50,
        "high_threshold": 75,
        "critical_threshold": 90,
        "weight": 0.15,
    },
    "certification_status": {
        "factor": "certification_status",
        "description": "Certification scheme validity, chain-of-custody, scope",
        "low_threshold": 25,
        "medium_threshold": 50,
        "high_threshold": 75,
        "critical_threshold": 90,
        "weight": 0.15,
    },
    "traceability_completeness": {
        "factor": "traceability_completeness",
        "description": "Supply chain traceability, plot-level mapping, batch tracking",
        "low_threshold": 25,
        "medium_threshold": 50,
        "high_threshold": 75,
        "critical_threshold": 90,
        "weight": 0.10,
    },
    "financial_stability": {
        "factor": "financial_stability",
        "description": "Financial health, credit rating, payment history",
        "low_threshold": 25,
        "medium_threshold": 50,
        "high_threshold": 75,
        "critical_threshold": 90,
        "weight": 0.10,
    },
    "environmental_performance": {
        "factor": "environmental_performance",
        "description": "Environmental incidents, deforestation, protected area encroachment",
        "low_threshold": 25,
        "medium_threshold": 50,
        "high_threshold": 75,
        "critical_threshold": 90,
        "weight": 0.10,
    },
    "social_compliance": {
        "factor": "social_compliance",
        "description": "Labor rights, indigenous rights, community relations",
        "low_threshold": 25,
        "medium_threshold": 50,
        "high_threshold": 75,
        "critical_threshold": 90,
        "weight": 0.05,
    },
}

# ===========================================================================
# Industry Averages (by commodity and region)
# ===========================================================================
#
# Average risk scores for peer comparison.

INDUSTRY_AVERAGES: Dict[str, IndustryAverage] = {
    # Cattle
    "cattle_south_america": {"commodity": "cattle", "region": "south_america", "avg_risk_score": 62.0},
    "cattle_central_america": {"commodity": "cattle", "region": "central_america", "avg_risk_score": 58.0},
    "cattle_africa": {"commodity": "cattle", "region": "africa", "avg_risk_score": 64.0},
    # Cocoa
    "cocoa_west_africa": {"commodity": "cocoa", "region": "west_africa", "avg_risk_score": 60.0},
    "cocoa_south_america": {"commodity": "cocoa", "region": "south_america", "avg_risk_score": 48.0},
    "cocoa_southeast_asia": {"commodity": "cocoa", "region": "southeast_asia", "avg_risk_score": 54.0},
    # Coffee
    "coffee_south_america": {"commodity": "coffee", "region": "south_america", "avg_risk_score": 46.0},
    "coffee_central_america": {"commodity": "coffee", "region": "central_america", "avg_risk_score": 42.0},
    "coffee_africa": {"commodity": "coffee", "region": "africa", "avg_risk_score": 52.0},
    "coffee_southeast_asia": {"commodity": "coffee", "region": "southeast_asia", "avg_risk_score": 50.0},
    # Oil palm
    "oil_palm_southeast_asia": {"commodity": "oil_palm", "region": "southeast_asia", "avg_risk_score": 64.0},
    "oil_palm_africa": {"commodity": "oil_palm", "region": "africa", "avg_risk_score": 68.0},
    "oil_palm_south_america": {"commodity": "oil_palm", "region": "south_america", "avg_risk_score": 70.0},
    # Rubber
    "rubber_southeast_asia": {"commodity": "rubber", "region": "southeast_asia", "avg_risk_score": 54.0},
    "rubber_africa": {"commodity": "rubber", "region": "africa", "avg_risk_score": 62.0},
    "rubber_south_america": {"commodity": "rubber", "region": "south_america", "avg_risk_score": 58.0},
    # Soy
    "soya_south_america": {"commodity": "soya", "region": "south_america", "avg_risk_score": 56.0},
    "soya_north_america": {"commodity": "soya", "region": "north_america", "avg_risk_score": 32.0},
    "soya_asia": {"commodity": "soya", "region": "asia", "avg_risk_score": 48.0},
    # Wood
    "wood_south_america": {"commodity": "wood", "region": "south_america", "avg_risk_score": 58.0},
    "wood_africa": {"commodity": "wood", "region": "africa", "avg_risk_score": 72.0},
    "wood_southeast_asia": {"commodity": "wood", "region": "southeast_asia", "avg_risk_score": 60.0},
    "wood_europe": {"commodity": "wood", "region": "europe", "avg_risk_score": 28.0},
    "wood_north_america": {"commodity": "wood", "region": "north_america", "avg_risk_score": 32.0},
}

# ===========================================================================
# Peer Group Definitions
# ===========================================================================
#
# Criteria for grouping suppliers for peer comparison.

PEER_GROUP_DEFINITIONS: Dict[str, PeerGroupDefinition] = {
    "same_commodity_region": {
        "name": "Same Commodity & Region",
        "criteria": ["commodity", "region"],
        "description": "Suppliers producing the same commodity in the same region",
    },
    "same_commodity_country": {
        "name": "Same Commodity & Country",
        "criteria": ["commodity", "country"],
        "description": "Suppliers producing the same commodity in the same country",
    },
    "same_type_commodity": {
        "name": "Same Type & Commodity",
        "criteria": ["type", "commodity"],
        "description": "Suppliers of the same type (producer/trader) for the same commodity",
    },
    "same_size_commodity": {
        "name": "Same Size & Commodity",
        "criteria": ["size_tier", "commodity"],
        "description": "Suppliers of similar size (by volume) for the same commodity",
        "size_tiers": {
            "small": {"min_tons": 0, "max_tons": 5000},
            "medium": {"min_tons": 5001, "max_tons": 20000},
            "large": {"min_tons": 20001, "max_tons": 1000000},
        },
    },
}

# ===========================================================================
# Non-Conformance Severity Matrix
# ===========================================================================
#
# Maps issue types to severity levels (minor, major, critical).

NON_CONFORMANCE_SEVERITY_MATRIX: Dict[str, SeverityMapping] = {
    # Documentation issues
    "missing_geolocation": {"issue_type": "missing_geolocation", "severity": "major"},
    "missing_dds_reference": {"issue_type": "missing_dds_reference", "severity": "critical"},
    "expired_certificate": {"issue_type": "expired_certificate", "severity": "major"},
    "incomplete_product_description": {"issue_type": "incomplete_product_description", "severity": "minor"},
    "missing_harvest_date": {"issue_type": "missing_harvest_date", "severity": "major"},
    # Compliance issues
    "deforestation_detected": {"issue_type": "deforestation_detected", "severity": "critical"},
    "protected_area_encroachment": {"issue_type": "protected_area_encroachment", "severity": "critical"},
    "indigenous_territory_violation": {"issue_type": "indigenous_territory_violation", "severity": "critical"},
    "illegal_logging": {"issue_type": "illegal_logging", "severity": "critical"},
    "labor_rights_violation": {"issue_type": "labor_rights_violation", "severity": "major"},
    # Certification issues
    "certification_suspended": {"issue_type": "certification_suspended", "severity": "major"},
    "certification_revoked": {"issue_type": "certification_revoked", "severity": "critical"},
    "chain_of_custody_break": {"issue_type": "chain_of_custody_break", "severity": "major"},
    "scope_mismatch": {"issue_type": "scope_mismatch", "severity": "minor"},
    # Traceability issues
    "untraceable_origin": {"issue_type": "untraceable_origin", "severity": "critical"},
    "missing_plot_coordinates": {"issue_type": "missing_plot_coordinates", "severity": "major"},
    "batch_traceability_gap": {"issue_type": "batch_traceability_gap", "severity": "major"},
    "supplier_identity_mismatch": {"issue_type": "supplier_identity_mismatch", "severity": "major"},
    # Financial issues
    "payment_default": {"issue_type": "payment_default", "severity": "minor"},
    "bankruptcy_filed": {"issue_type": "bankruptcy_filed", "severity": "major"},
    "financial_sanctions": {"issue_type": "financial_sanctions", "severity": "critical"},
    # Environmental issues
    "pollution_incident": {"issue_type": "pollution_incident", "severity": "major"},
    "habitat_destruction": {"issue_type": "habitat_destruction", "severity": "critical"},
    "biodiversity_loss": {"issue_type": "biodiversity_loss", "severity": "major"},
    "pesticide_misuse": {"issue_type": "pesticide_misuse", "severity": "major"},
    # Social issues
    "forced_labor": {"issue_type": "forced_labor", "severity": "critical"},
    "child_labor": {"issue_type": "child_labor", "severity": "critical"},
    "unsafe_working_conditions": {"issue_type": "unsafe_working_conditions", "severity": "major"},
    "community_conflict": {"issue_type": "community_conflict", "severity": "major"},
}

# ===========================================================================
# Helper functions
# ===========================================================================


def get_supplier(supplier_id: str) -> Optional[SupplierRecord]:
    """
    Retrieve supplier record by supplier_id.

    Args:
        supplier_id: Unique supplier identifier (e.g., "SUP-BRA-001")

    Returns:
        SupplierRecord dict or None if not found
    """
    return SAMPLE_SUPPLIERS.get(supplier_id)


def get_benchmarks(factor: str) -> Optional[RiskFactorBenchmark]:
    """
    Retrieve risk factor benchmarks by factor name.

    Args:
        factor: Risk factor name (e.g., "geographic_sourcing")

    Returns:
        RiskFactorBenchmark dict or None if not found
    """
    return RISK_FACTOR_BENCHMARKS.get(factor)


def get_industry_average(commodity: str, region: str) -> Optional[float]:
    """
    Retrieve industry average risk score for commodity and region.

    Args:
        commodity: EUDR commodity (e.g., "cattle")
        region: Geographic region (e.g., "south_america")

    Returns:
        Average risk score (float) or None if not found
    """
    key = f"{commodity}_{region}"
    avg_data = INDUSTRY_AVERAGES.get(key)
    return avg_data["avg_risk_score"] if avg_data else None


def get_peer_group(peer_group_name: str) -> Optional[PeerGroupDefinition]:
    """
    Retrieve peer group definition by name.

    Args:
        peer_group_name: Peer group name (e.g., "same_commodity_region")

    Returns:
        PeerGroupDefinition dict or None if not found
    """
    return PEER_GROUP_DEFINITIONS.get(peer_group_name)


def get_nc_severity(issue_type: str) -> Optional[str]:
    """
    Retrieve non-conformance severity level by issue type.

    Args:
        issue_type: Issue type (e.g., "deforestation_detected")

    Returns:
        Severity level ("minor" | "major" | "critical") or None if not found
    """
    mapping = NON_CONFORMANCE_SEVERITY_MATRIX.get(issue_type)
    return mapping["severity"] if mapping else None


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    "DATA_VERSION",
    "DATA_SOURCES",
    "EUDR_COMMODITIES",
    "SAMPLE_SUPPLIERS",
    "RISK_FACTOR_BENCHMARKS",
    "INDUSTRY_AVERAGES",
    "PEER_GROUP_DEFINITIONS",
    "NON_CONFORMANCE_SEVERITY_MATRIX",
    "get_supplier",
    "get_benchmarks",
    "get_industry_average",
    "get_peer_group",
    "get_nc_severity",
]
