# -*- coding: utf-8 -*-
"""
Regulatory Requirements - AGENT-EUDR-018 Commodity Risk Analyzer

EUDR regulatory requirement database mapping EU 2023/1115 articles to
commodity-specific obligations, documentation requirements, and penalty
matrices. Each article is decomposed into per-commodity requirements
with specific documentation standards, evidence types, and compliance
criteria.

Articles covered:
    - Article 3:  Prohibition scope per commodity
    - Article 4:  Operator obligations per commodity
    - Article 9:  Due diligence statement (DDS) requirements per commodity
    - Article 10: Information requirements (geolocation per commodity)
    - Article 11: Risk assessment requirements per commodity
    - Article 12: Risk mitigation measures per commodity
    - Article 13: Competent authority reporting per commodity
    - Article 29: Benchmarking (country risk classification impact)

Documentation requirements are organized as:
    - Common requirements: Geolocation, supplier info, quantity, dates
    - Commodity-specific requirements: Species ID (wood), farm registration
      (cattle), mill registration (palm oil), cooperative records (cocoa/coffee),
      GMO status (soya), plantation boundaries (rubber)

Penalty matrix provides:
    - Violation categories (administrative, non-compliance, fraud)
    - Penalty ranges per category
    - Aggravating/mitigating factors per commodity

Data Sources:
    - EU Regulation 2023/1115 (EUDR) Official Text
    - European Commission FAQ on EUDR Implementation (2024)
    - European Commission Implementing Regulation (EU) 2024/3120
    - EUDR Delegated Acts on Benchmarking (Article 29)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data version and source metadata
# ---------------------------------------------------------------------------

DATA_VERSION: str = "2025-03"
DATA_SOURCES: List[str] = [
    "EU Regulation 2023/1115 (EUDR) Official Journal L 150, 9.6.2023",
    "European Commission FAQ on EUDR Implementation 2024",
    "European Commission Implementing Regulation (EU) 2024/3120",
    "EUDR Delegated Acts on Benchmarking (Article 29) 2025",
    "European Commission Guidance on Geolocation Requirements 2024",
]

# ===========================================================================
# EUDR Articles - Per-article requirements mapped to each commodity
# ===========================================================================

EUDR_ARTICLES: Dict[str, Dict[str, Any]] = {

    # -------------------------------------------------------------------
    # Article 3: Prohibition
    # -------------------------------------------------------------------
    "article_3": {
        "article_number": 3,
        "title": "Prohibition",
        "description": (
            "Relevant commodities and relevant products shall not be placed "
            "on the Union market or exported from it, unless they are "
            "deforestation-free, produced in accordance with relevant "
            "legislation, and covered by a due diligence statement."
        ),
        "commodity_requirements": {
            "cattle": {
                "prohibition_scope": "Live cattle, beef, leather, tallow, gelatin derived from cattle",
                "deforestation_free_definition": "Land not subject to deforestation after 31 December 2020",
                "degradation_check": True,
                "cutoff_date": "2020-12-31",
                "special_conditions": ["Must verify farm-level deforestation status", "Includes indirect land use change through pasture expansion"],
            },
            "cocoa": {
                "prohibition_scope": "Cocoa beans, paste, butter, powder, chocolate, preparations",
                "deforestation_free_definition": "Land not subject to deforestation after 31 December 2020",
                "degradation_check": True,
                "cutoff_date": "2020-12-31",
                "special_conditions": ["Smallholder aggregation may apply", "Cooperative-level verification accepted"],
            },
            "coffee": {
                "prohibition_scope": "Green and roasted coffee, instant coffee, extracts",
                "deforestation_free_definition": "Land not subject to deforestation after 31 December 2020",
                "degradation_check": True,
                "cutoff_date": "2020-12-31",
                "special_conditions": ["Shade-grown coffee may have different forest baseline", "Agroforestry systems must be assessed individually"],
            },
            "oil_palm": {
                "prohibition_scope": "Palm oil, palm kernel oil, oleochemicals, biodiesel",
                "deforestation_free_definition": "Land not subject to deforestation after 31 December 2020",
                "degradation_check": True,
                "cutoff_date": "2020-12-31",
                "special_conditions": ["Must trace to plantation/mill level", "NDPE compliance alone not sufficient", "Peatland drainage counts as degradation"],
            },
            "rubber": {
                "prohibition_scope": "Natural rubber, latex, RSS, TSR, tires, compounds",
                "deforestation_free_definition": "Land not subject to deforestation after 31 December 2020",
                "degradation_check": True,
                "cutoff_date": "2020-12-31",
                "special_conditions": ["Smallholder rubber often mixed at collection point", "Must verify plantation boundaries"],
            },
            "soya": {
                "prohibition_scope": "Soybeans, soy meal, soy oil, lecithin, biodiesel",
                "deforestation_free_definition": "Land not subject to deforestation after 31 December 2020",
                "degradation_check": True,
                "cutoff_date": "2020-12-31",
                "special_conditions": ["Cerrado biome included for Brazil", "GMO status must be declared", "Storage facility mixing is a traceability risk"],
            },
            "wood": {
                "prohibition_scope": "Wood in the rough, sawn wood, plywood, furniture, pulp, paper, charcoal",
                "deforestation_free_definition": "Land not subject to deforestation after 31 December 2020",
                "degradation_check": True,
                "cutoff_date": "2020-12-31",
                "special_conditions": ["Species identification (genus + species) required", "Legal harvest verification with felling license", "Existing EU Timber Regulation compliance may partially satisfy"],
            },
        },
    },

    # -------------------------------------------------------------------
    # Article 4: Operator obligations
    # -------------------------------------------------------------------
    "article_4": {
        "article_number": 4,
        "title": "Obligations of operators",
        "description": (
            "Operators shall exercise due diligence before placing relevant "
            "commodities and products on the Union market or exporting them."
        ),
        "commodity_requirements": {
            "cattle": {
                "dd_required": True,
                "dd_type": "full",
                "information_collection": True,
                "risk_assessment": True,
                "risk_mitigation": True,
                "specific_obligations": ["Verify animal health certificates", "Check farm registration with competent authority", "Confirm slaughterhouse licensing"],
            },
            "cocoa": {
                "dd_required": True,
                "dd_type": "full",
                "information_collection": True,
                "risk_assessment": True,
                "risk_mitigation": True,
                "specific_obligations": ["Verify cooperative membership", "Check fermentation/washing station ID", "Confirm export license"],
            },
            "coffee": {
                "dd_required": True,
                "dd_type": "full",
                "information_collection": True,
                "risk_assessment": True,
                "risk_mitigation": True,
                "specific_obligations": ["Verify washing station/cooperative registration", "Check phytosanitary certificate", "Confirm ICO certificate of origin"],
            },
            "oil_palm": {
                "dd_required": True,
                "dd_type": "full",
                "information_collection": True,
                "risk_assessment": True,
                "risk_mitigation": True,
                "specific_obligations": ["Verify mill registration (RSPO/ISPO/MSPO)", "Confirm plantation boundary maps", "Check NDPE compliance documentation"],
            },
            "rubber": {
                "dd_required": True,
                "dd_type": "full",
                "information_collection": True,
                "risk_assessment": True,
                "risk_mitigation": True,
                "specific_obligations": ["Verify processing facility registration", "Check plantation boundaries and concession maps", "Confirm GPSNR membership or equivalent"],
            },
            "soya": {
                "dd_required": True,
                "dd_type": "full",
                "information_collection": True,
                "risk_assessment": True,
                "risk_mitigation": True,
                "specific_obligations": ["Verify CAR (Rural Environmental Registry) for Brazil", "Check storage facility records", "Confirm GMO status declaration"],
            },
            "wood": {
                "dd_required": True,
                "dd_type": "full",
                "information_collection": True,
                "risk_assessment": True,
                "risk_mitigation": True,
                "specific_obligations": ["Verify species identification (genus + species)", "Check felling license/harvest permit", "Confirm forest management plan", "Verify CITES listing status"],
            },
        },
    },

    # -------------------------------------------------------------------
    # Article 9: Due Diligence Statement (DDS)
    # -------------------------------------------------------------------
    "article_9": {
        "article_number": 9,
        "title": "Due diligence statements",
        "description": (
            "Operators shall submit a due diligence statement to the competent "
            "authority before placing products on the market or exporting them, "
            "confirming deforestation-free status and legal compliance."
        ),
        "commodity_requirements": {
            "cattle": {
                "dds_required": True,
                "submission_timing": "before_market_placement",
                "reference_number_format": "EUDR-DDS-{country}-{year}-{sequence}",
                "mandatory_fields": ["operator_id", "commodity_type", "hs_code", "country_of_production", "geolocation", "quantity_kg", "production_date", "supplier_chain", "deforestation_free_declaration"],
                "specific_fields": ["farm_registration_id", "animal_health_certificate_ref", "slaughterhouse_approval_number"],
                "retention_years": 5,
            },
            "cocoa": {
                "dds_required": True,
                "submission_timing": "before_market_placement",
                "reference_number_format": "EUDR-DDS-{country}-{year}-{sequence}",
                "mandatory_fields": ["operator_id", "commodity_type", "hs_code", "country_of_production", "geolocation", "quantity_kg", "production_date", "supplier_chain", "deforestation_free_declaration"],
                "specific_fields": ["cooperative_id", "fermentation_facility_id", "export_license_ref"],
                "retention_years": 5,
            },
            "coffee": {
                "dds_required": True,
                "submission_timing": "before_market_placement",
                "reference_number_format": "EUDR-DDS-{country}-{year}-{sequence}",
                "mandatory_fields": ["operator_id", "commodity_type", "hs_code", "country_of_production", "geolocation", "quantity_kg", "production_date", "supplier_chain", "deforestation_free_declaration"],
                "specific_fields": ["washing_station_id", "ico_certificate_ref", "phytosanitary_certificate_ref"],
                "retention_years": 5,
            },
            "oil_palm": {
                "dds_required": True,
                "submission_timing": "before_market_placement",
                "reference_number_format": "EUDR-DDS-{country}-{year}-{sequence}",
                "mandatory_fields": ["operator_id", "commodity_type", "hs_code", "country_of_production", "geolocation", "quantity_kg", "production_date", "supplier_chain", "deforestation_free_declaration"],
                "specific_fields": ["mill_registration_id", "plantation_boundary_ref", "rspo_certificate_ref"],
                "retention_years": 5,
            },
            "rubber": {
                "dds_required": True,
                "submission_timing": "before_market_placement",
                "reference_number_format": "EUDR-DDS-{country}-{year}-{sequence}",
                "mandatory_fields": ["operator_id", "commodity_type", "hs_code", "country_of_production", "geolocation", "quantity_kg", "production_date", "supplier_chain", "deforestation_free_declaration"],
                "specific_fields": ["processing_facility_id", "plantation_boundary_ref"],
                "retention_years": 5,
            },
            "soya": {
                "dds_required": True,
                "submission_timing": "before_market_placement",
                "reference_number_format": "EUDR-DDS-{country}-{year}-{sequence}",
                "mandatory_fields": ["operator_id", "commodity_type", "hs_code", "country_of_production", "geolocation", "quantity_kg", "production_date", "supplier_chain", "deforestation_free_declaration"],
                "specific_fields": ["car_registration_number", "storage_facility_id", "gmo_status"],
                "retention_years": 5,
            },
            "wood": {
                "dds_required": True,
                "submission_timing": "before_market_placement",
                "reference_number_format": "EUDR-DDS-{country}-{year}-{sequence}",
                "mandatory_fields": ["operator_id", "commodity_type", "hs_code", "country_of_production", "geolocation", "quantity_kg", "production_date", "supplier_chain", "deforestation_free_declaration"],
                "specific_fields": ["species_genus", "species_name", "felling_license_ref", "forest_management_plan_ref", "cites_permit_ref"],
                "retention_years": 5,
            },
        },
    },

    # -------------------------------------------------------------------
    # Article 10: Information requirements
    # -------------------------------------------------------------------
    "article_10": {
        "article_number": 10,
        "title": "Information requirements",
        "description": (
            "Operators shall collect information including geolocation "
            "coordinates of all plots of land where the commodity was produced."
        ),
        "commodity_requirements": {
            "cattle": {
                "geolocation_type": "polygon_preferred",
                "geolocation_minimum": "single_point",
                "geolocation_precision": "plot_level",
                "area_threshold_hectares": 4.0,
                "polygon_required_above_threshold": True,
                "specific_info": ["Farm boundary coordinates", "Pasture area polygons", "Feedlot location"],
            },
            "cocoa": {
                "geolocation_type": "polygon_preferred",
                "geolocation_minimum": "single_point",
                "geolocation_precision": "plot_level",
                "area_threshold_hectares": 4.0,
                "polygon_required_above_threshold": True,
                "specific_info": ["Farm/plot boundary coordinates", "Cooperative aggregation polygons accepted for <4ha plots"],
            },
            "coffee": {
                "geolocation_type": "polygon_preferred",
                "geolocation_minimum": "single_point",
                "geolocation_precision": "plot_level",
                "area_threshold_hectares": 4.0,
                "polygon_required_above_threshold": True,
                "specific_info": ["Farm/plot boundary coordinates", "Washing station catchment area", "Altitude data for quality traceability"],
            },
            "oil_palm": {
                "geolocation_type": "polygon_required",
                "geolocation_minimum": "polygon",
                "geolocation_precision": "plantation_level",
                "area_threshold_hectares": 4.0,
                "polygon_required_above_threshold": True,
                "specific_info": ["Plantation boundary polygons", "Mill supply base mapping", "Concession area boundary"],
            },
            "rubber": {
                "geolocation_type": "polygon_preferred",
                "geolocation_minimum": "single_point",
                "geolocation_precision": "plot_level",
                "area_threshold_hectares": 4.0,
                "polygon_required_above_threshold": True,
                "specific_info": ["Plantation boundary coordinates", "Collection point locations", "Processing facility location"],
            },
            "soya": {
                "geolocation_type": "polygon_preferred",
                "geolocation_minimum": "single_point",
                "geolocation_precision": "plot_level",
                "area_threshold_hectares": 4.0,
                "polygon_required_above_threshold": True,
                "specific_info": ["Farm boundary coordinates (CAR boundary for Brazil)", "Storage facility location", "Field-level parcel polygons"],
            },
            "wood": {
                "geolocation_type": "polygon_required",
                "geolocation_minimum": "polygon",
                "geolocation_precision": "harvest_compartment",
                "area_threshold_hectares": 4.0,
                "polygon_required_above_threshold": True,
                "specific_info": ["Harvest compartment polygon", "Concession area boundary", "Forest management unit boundary", "Felling site GPS coordinates"],
            },
        },
    },

    # -------------------------------------------------------------------
    # Article 11: Risk assessment
    # -------------------------------------------------------------------
    "article_11": {
        "article_number": 11,
        "title": "Risk assessment",
        "description": (
            "Operators shall assess the risk that relevant commodities and "
            "products do not comply with EUDR requirements."
        ),
        "commodity_requirements": {
            "cattle": {
                "risk_factors": ["country_risk_level", "deforestation_prevalence", "indigenous_peoples_rights", "legal_compliance", "corruption_level", "supply_chain_complexity", "documentation_quality"],
                "minimum_assessment_frequency": "per_shipment",
                "enhanced_assessment_triggers": ["high_risk_country", "new_supplier", "volume_anomaly", "adverse_media"],
            },
            "cocoa": {
                "risk_factors": ["country_risk_level", "deforestation_prevalence", "child_labour_risk", "legal_compliance", "cooperative_governance", "documentation_quality"],
                "minimum_assessment_frequency": "per_shipment",
                "enhanced_assessment_triggers": ["high_risk_country", "new_supplier", "origin_change", "certification_lapse"],
            },
            "coffee": {
                "risk_factors": ["country_risk_level", "deforestation_prevalence", "legal_compliance", "agroforestry_status", "documentation_quality"],
                "minimum_assessment_frequency": "per_shipment",
                "enhanced_assessment_triggers": ["high_risk_country", "new_supplier", "origin_change"],
            },
            "oil_palm": {
                "risk_factors": ["country_risk_level", "deforestation_prevalence", "peatland_risk", "fire_alerts", "ndpe_compliance", "legal_compliance", "mill_governance"],
                "minimum_assessment_frequency": "per_shipment",
                "enhanced_assessment_triggers": ["high_risk_country", "fire_alert_near_source", "grievance_filed", "certification_suspension"],
            },
            "rubber": {
                "risk_factors": ["country_risk_level", "deforestation_prevalence", "legal_compliance", "concession_governance", "documentation_quality"],
                "minimum_assessment_frequency": "per_shipment",
                "enhanced_assessment_triggers": ["high_risk_country", "new_supplier", "concession_overlap_with_forest"],
            },
            "soya": {
                "risk_factors": ["country_risk_level", "deforestation_prevalence", "cerrado_conversion_risk", "legal_compliance", "documentation_quality"],
                "minimum_assessment_frequency": "per_shipment",
                "enhanced_assessment_triggers": ["high_risk_country", "cerrado_biome_origin", "new_supplier", "volume_anomaly"],
            },
            "wood": {
                "risk_factors": ["country_risk_level", "deforestation_prevalence", "illegal_logging_prevalence", "species_risk", "cites_listing", "legal_compliance", "governance_quality"],
                "minimum_assessment_frequency": "per_shipment",
                "enhanced_assessment_triggers": ["high_risk_country", "cites_listed_species", "conflict_timber_risk", "governance_below_threshold"],
            },
        },
    },

    # -------------------------------------------------------------------
    # Article 12: Risk mitigation
    # -------------------------------------------------------------------
    "article_12": {
        "article_number": 12,
        "title": "Risk mitigation",
        "description": (
            "Where the risk assessment reveals a non-negligible risk, operators "
            "shall take risk mitigation measures before placing products on the market."
        ),
        "commodity_requirements": {
            "cattle": {"mitigation_measures": ["additional_documentation", "third_party_verification", "satellite_monitoring", "site_visit", "supplier_audit"]},
            "cocoa": {"mitigation_measures": ["additional_documentation", "third_party_verification", "satellite_monitoring", "cooperative_audit", "origin_verification"]},
            "coffee": {"mitigation_measures": ["additional_documentation", "third_party_verification", "satellite_monitoring", "washing_station_audit"]},
            "oil_palm": {"mitigation_measures": ["additional_documentation", "third_party_verification", "satellite_monitoring", "mill_audit", "ndpe_verification", "grievance_resolution"]},
            "rubber": {"mitigation_measures": ["additional_documentation", "third_party_verification", "satellite_monitoring", "plantation_audit"]},
            "soya": {"mitigation_measures": ["additional_documentation", "third_party_verification", "satellite_monitoring", "car_verification", "field_visit"]},
            "wood": {"mitigation_measures": ["additional_documentation", "third_party_verification", "satellite_monitoring", "timber_tracking", "species_verification", "chain_of_custody_audit"]},
        },
    },

    # -------------------------------------------------------------------
    # Article 13: Competent authority reporting
    # -------------------------------------------------------------------
    "article_13": {
        "article_number": 13,
        "title": "Checks by competent authorities",
        "description": (
            "Competent authorities shall carry out checks to verify compliance, "
            "with check frequencies based on country risk classification."
        ),
        "commodity_requirements": {
            "cattle": {"check_rate_high_risk": 0.09, "check_rate_standard_risk": 0.03, "check_rate_low_risk": 0.01, "documentary_check_pct": 1.0, "physical_check_pct": 0.10},
            "cocoa": {"check_rate_high_risk": 0.09, "check_rate_standard_risk": 0.03, "check_rate_low_risk": 0.01, "documentary_check_pct": 1.0, "physical_check_pct": 0.10},
            "coffee": {"check_rate_high_risk": 0.09, "check_rate_standard_risk": 0.03, "check_rate_low_risk": 0.01, "documentary_check_pct": 1.0, "physical_check_pct": 0.10},
            "oil_palm": {"check_rate_high_risk": 0.09, "check_rate_standard_risk": 0.03, "check_rate_low_risk": 0.01, "documentary_check_pct": 1.0, "physical_check_pct": 0.10},
            "rubber": {"check_rate_high_risk": 0.09, "check_rate_standard_risk": 0.03, "check_rate_low_risk": 0.01, "documentary_check_pct": 1.0, "physical_check_pct": 0.10},
            "soya": {"check_rate_high_risk": 0.09, "check_rate_standard_risk": 0.03, "check_rate_low_risk": 0.01, "documentary_check_pct": 1.0, "physical_check_pct": 0.10},
            "wood": {"check_rate_high_risk": 0.09, "check_rate_standard_risk": 0.03, "check_rate_low_risk": 0.01, "documentary_check_pct": 1.0, "physical_check_pct": 0.15},
        },
    },

    # -------------------------------------------------------------------
    # Article 29: Benchmarking system
    # -------------------------------------------------------------------
    "article_29": {
        "article_number": 29,
        "title": "Benchmarking system",
        "description": (
            "The Commission shall classify countries or parts thereof as low, "
            "standard, or high risk based on deforestation rates, governance, "
            "and enforcement criteria."
        ),
        "commodity_requirements": {
            "cattle": {"benchmark_impact": "Check rates and DD intensity vary by country classification", "simplified_dd_for_low_risk": True, "enhanced_dd_for_high_risk": True},
            "cocoa": {"benchmark_impact": "Check rates and DD intensity vary by country classification", "simplified_dd_for_low_risk": True, "enhanced_dd_for_high_risk": True},
            "coffee": {"benchmark_impact": "Check rates and DD intensity vary by country classification", "simplified_dd_for_low_risk": True, "enhanced_dd_for_high_risk": True},
            "oil_palm": {"benchmark_impact": "Check rates and DD intensity vary by country classification", "simplified_dd_for_low_risk": True, "enhanced_dd_for_high_risk": True},
            "rubber": {"benchmark_impact": "Check rates and DD intensity vary by country classification", "simplified_dd_for_low_risk": True, "enhanced_dd_for_high_risk": True},
            "soya": {"benchmark_impact": "Check rates and DD intensity vary by country classification", "simplified_dd_for_low_risk": True, "enhanced_dd_for_high_risk": True},
            "wood": {"benchmark_impact": "Check rates and DD intensity vary by country classification", "simplified_dd_for_low_risk": True, "enhanced_dd_for_high_risk": True},
        },
    },
}

# ===========================================================================
# Documentation Requirements - Common and commodity-specific
# ===========================================================================

DOCUMENTATION_REQUIREMENTS: Dict[str, Any] = {

    "common": [
        {"document_type": "geolocation_data", "description": "GPS coordinates (point or polygon) of production plots", "format": "GeoJSON/KML/Shapefile", "mandatory": True, "evidence_type": "geospatial"},
        {"document_type": "supplier_information", "description": "Full identification of all suppliers in the chain", "format": "structured_data", "mandatory": True, "evidence_type": "documentary"},
        {"document_type": "quantity_declaration", "description": "Quantity and weight of commodity in the shipment", "format": "structured_data", "mandatory": True, "evidence_type": "documentary"},
        {"document_type": "production_date", "description": "Date or date range of production/harvest", "format": "ISO 8601", "mandatory": True, "evidence_type": "documentary"},
        {"document_type": "compliance_declaration", "description": "Operator declaration of EUDR compliance", "format": "signed_document", "mandatory": True, "evidence_type": "declaration"},
        {"document_type": "deforestation_free_statement", "description": "Statement that commodity is deforestation-free", "format": "signed_document", "mandatory": True, "evidence_type": "declaration"},
        {"document_type": "trade_documents", "description": "Commercial invoice, bill of lading, packing list", "format": "commercial_documents", "mandatory": True, "evidence_type": "documentary"},
    ],

    "cattle": [
        {"document_type": "farm_registration", "description": "Official farm registration with competent authority", "format": "certificate", "mandatory": True, "evidence_type": "registration"},
        {"document_type": "animal_health_certificate", "description": "Veterinary health certificate for livestock", "format": "certificate", "mandatory": True, "evidence_type": "certificate"},
        {"document_type": "slaughterhouse_records", "description": "Approved slaughterhouse processing records", "format": "structured_data", "mandatory": True, "evidence_type": "processing_record"},
        {"document_type": "ear_tag_records", "description": "Individual animal identification records", "format": "structured_data", "mandatory": False, "evidence_type": "traceability"},
        {"document_type": "pasture_management_plan", "description": "Pasture management and rotation plan", "format": "document", "mandatory": False, "evidence_type": "management_plan"},
    ],

    "cocoa": [
        {"document_type": "cooperative_registration", "description": "Cooperative or farmer group registration", "format": "certificate", "mandatory": True, "evidence_type": "registration"},
        {"document_type": "fermentation_facility_id", "description": "Fermentation/washing station identification", "format": "structured_data", "mandatory": True, "evidence_type": "facility_id"},
        {"document_type": "export_license", "description": "Government-issued export license for cocoa", "format": "license", "mandatory": True, "evidence_type": "license"},
        {"document_type": "quality_grading_certificate", "description": "Cocoa quality grading certificate", "format": "certificate", "mandatory": False, "evidence_type": "quality"},
    ],

    "coffee": [
        {"document_type": "washing_station_registration", "description": "Wet mill/washing station registration", "format": "certificate", "mandatory": True, "evidence_type": "registration"},
        {"document_type": "ico_certificate_of_origin", "description": "ICO Certificate of Origin (if applicable)", "format": "certificate", "mandatory": False, "evidence_type": "certificate"},
        {"document_type": "phytosanitary_certificate", "description": "Phytosanitary certificate for export", "format": "certificate", "mandatory": True, "evidence_type": "certificate"},
        {"document_type": "cupping_score_record", "description": "Quality cupping score documentation", "format": "structured_data", "mandatory": False, "evidence_type": "quality"},
    ],

    "oil_palm": [
        {"document_type": "mill_registration", "description": "Palm oil mill registration and license", "format": "certificate", "mandatory": True, "evidence_type": "registration"},
        {"document_type": "plantation_boundary_maps", "description": "GPS boundary polygons of supplying plantations", "format": "GeoJSON/KML/Shapefile", "mandatory": True, "evidence_type": "geospatial"},
        {"document_type": "ndpe_compliance_docs", "description": "No Deforestation, No Peat, No Exploitation policy documentation", "format": "document", "mandatory": False, "evidence_type": "policy"},
        {"document_type": "rspo_certificate", "description": "RSPO certification (if certified)", "format": "certificate", "mandatory": False, "evidence_type": "certification"},
        {"document_type": "supply_base_report", "description": "Mill supply base report with plantation list", "format": "structured_data", "mandatory": True, "evidence_type": "supply_chain"},
    ],

    "rubber": [
        {"document_type": "processing_facility_registration", "description": "Rubber processing facility registration", "format": "certificate", "mandatory": True, "evidence_type": "registration"},
        {"document_type": "plantation_boundaries", "description": "GPS boundary polygons of rubber plantations", "format": "GeoJSON/KML/Shapefile", "mandatory": True, "evidence_type": "geospatial"},
        {"document_type": "concession_license", "description": "Land concession license (if applicable)", "format": "license", "mandatory": False, "evidence_type": "license"},
        {"document_type": "gpsnr_membership", "description": "GPSNR membership documentation", "format": "certificate", "mandatory": False, "evidence_type": "membership"},
    ],

    "soya": [
        {"document_type": "car_registration", "description": "CAR (Cadastro Ambiental Rural) registration for Brazil", "format": "certificate", "mandatory": True, "evidence_type": "registration"},
        {"document_type": "storage_facility_records", "description": "Silo/storage facility receiving and shipping records", "format": "structured_data", "mandatory": True, "evidence_type": "processing_record"},
        {"document_type": "gmo_status_declaration", "description": "GMO or non-GMO status declaration", "format": "declaration", "mandatory": True, "evidence_type": "declaration"},
        {"document_type": "phytosanitary_certificate", "description": "Phytosanitary certificate for export", "format": "certificate", "mandatory": True, "evidence_type": "certificate"},
    ],

    "wood": [
        {"document_type": "species_identification", "description": "Scientific identification (genus + species) of timber", "format": "structured_data", "mandatory": True, "evidence_type": "species_id"},
        {"document_type": "felling_license", "description": "Government-issued felling/harvest license", "format": "license", "mandatory": True, "evidence_type": "license"},
        {"document_type": "forest_management_plan", "description": "Approved forest management plan", "format": "document", "mandatory": True, "evidence_type": "management_plan"},
        {"document_type": "cites_permit", "description": "CITES import/export permit (if CITES-listed species)", "format": "permit", "mandatory": False, "evidence_type": "permit"},
        {"document_type": "chain_of_custody_certificate", "description": "FSC/PEFC chain of custody certificate", "format": "certificate", "mandatory": False, "evidence_type": "certification"},
        {"document_type": "transport_documents", "description": "Timber transport permits and waybills", "format": "document", "mandatory": True, "evidence_type": "transport"},
    ],
}

# ===========================================================================
# Penalty Matrix - Per violation type per commodity
# ===========================================================================

PENALTY_MATRIX: Dict[str, Dict[str, Any]] = {
    "administrative_non_compliance": {
        "description": "Failure to submit DDS, incomplete documentation, late filing",
        "severity": "low",
        "penalty_range_eur": {"min": 500, "max": 50_000},
        "commodity_multipliers": {
            "cattle": 1.0, "cocoa": 1.0, "coffee": 1.0, "oil_palm": 1.0,
            "rubber": 1.0, "soya": 1.0, "wood": 1.0,
        },
        "aggravating_factors": ["repeat_offence", "volume_exceeds_threshold", "high_risk_country_origin"],
        "mitigating_factors": ["first_offence", "voluntary_disclosure", "sme_operator"],
    },
    "substantial_non_compliance": {
        "description": "Inadequate risk assessment, insufficient mitigation, false geolocation",
        "severity": "medium",
        "penalty_range_eur": {"min": 10_000, "max": 500_000},
        "commodity_multipliers": {
            "cattle": 1.2, "cocoa": 1.0, "coffee": 1.0, "oil_palm": 1.3,
            "rubber": 1.0, "soya": 1.1, "wood": 1.2,
        },
        "aggravating_factors": ["repeat_offence", "high_volume", "environmental_damage", "indigenous_rights_violation"],
        "mitigating_factors": ["corrective_action_taken", "cooperation_with_authority", "certification_in_place"],
    },
    "fraud_or_deception": {
        "description": "Deliberate falsification of documents, fraudulent DDS, knowingly placing non-compliant products",
        "severity": "critical",
        "penalty_range_eur": {"min": 100_000, "max": 4_000_000},
        "penalty_turnover_pct": 4.0,
        "commodity_multipliers": {
            "cattle": 1.3, "cocoa": 1.1, "coffee": 1.0, "oil_palm": 1.4,
            "rubber": 1.1, "soya": 1.2, "wood": 1.3,
        },
        "aggravating_factors": ["organised_scheme", "large_scale", "environmental_damage", "repeat_offence", "obstruction_of_investigation"],
        "mitigating_factors": [],
        "additional_sanctions": ["temporary_market_ban", "product_confiscation", "public_naming", "criminal_referral"],
    },
}


# ===========================================================================
# RegulatoryRequirementDatabase class
# ===========================================================================


class RegulatoryRequirementDatabase:
    """
    Stateless accessor for EUDR regulatory requirement data.

    Provides methods to query article requirements, documentation lists,
    and penalty information per commodity type.

    Example:
        >>> db = RegulatoryRequirementDatabase()
        >>> reqs = db.get_requirements("article_9", "wood")
        >>> assert reqs["dds_required"] is True
        >>> docs = db.get_documentation_list("oil_palm")
        >>> assert len(docs) > 0
    """

    def get_requirements(
        self, article: str, commodity_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get requirements for an article and commodity.

        Args:
            article: Article key (e.g., "article_3", "article_9").
            commodity_type: One of the 7 EUDR commodity types.

        Returns:
            Requirements dict or None if not found.
        """
        article_data = EUDR_ARTICLES.get(article)
        if article_data is None:
            return None
        return article_data.get("commodity_requirements", {}).get(commodity_type)

    def get_documentation_list(
        self, commodity_type: str
    ) -> List[Dict[str, Any]]:
        """Get all documentation requirements for a commodity.

        Returns common requirements plus commodity-specific requirements.

        Args:
            commodity_type: One of the 7 EUDR commodity types.

        Returns:
            List of document requirement dicts.
        """
        docs = list(DOCUMENTATION_REQUIREMENTS.get("common", []))
        docs.extend(DOCUMENTATION_REQUIREMENTS.get(commodity_type, []))
        return docs

    def get_penalty_info(
        self, violation_type: str, commodity_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get penalty information for a violation type.

        Args:
            violation_type: Violation category key.
            commodity_type: Optional commodity for multiplier lookup.

        Returns:
            Penalty info dict or None if violation type not found.
        """
        penalty = PENALTY_MATRIX.get(violation_type)
        if penalty is None:
            return None
        result = dict(penalty)
        if commodity_type:
            result["commodity_multiplier"] = (
                penalty.get("commodity_multipliers", {}).get(commodity_type, 1.0)
            )
        return result

    def get_all_articles(self) -> List[str]:
        """Get list of all article keys.

        Returns:
            List of article key strings.
        """
        return list(EUDR_ARTICLES.keys())


# ===========================================================================
# Module-level helper functions
# ===========================================================================


def get_requirements(
    article: str, commodity_type: str
) -> Optional[Dict[str, Any]]:
    """Get requirements for an article and commodity.

    Args:
        article: Article key (e.g., "article_3", "article_9").
        commodity_type: One of the 7 EUDR commodity types.

    Returns:
        Requirements dict or None if not found.
    """
    article_data = EUDR_ARTICLES.get(article)
    if article_data is None:
        return None
    return article_data.get("commodity_requirements", {}).get(commodity_type)


def get_documentation_list(commodity_type: str) -> List[Dict[str, Any]]:
    """Get all documentation requirements for a commodity.

    Returns common requirements plus commodity-specific requirements.

    Args:
        commodity_type: One of the 7 EUDR commodity types.

    Returns:
        List of document requirement dicts.
    """
    docs = list(DOCUMENTATION_REQUIREMENTS.get("common", []))
    docs.extend(DOCUMENTATION_REQUIREMENTS.get(commodity_type, []))
    return docs


def get_penalty_info(
    violation_type: str, commodity_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get penalty information for a violation type.

    Args:
        violation_type: Violation category key.
        commodity_type: Optional commodity for multiplier lookup.

    Returns:
        Penalty info dict with optional commodity_multiplier field.
    """
    penalty = PENALTY_MATRIX.get(violation_type)
    if penalty is None:
        return None
    result = dict(penalty)
    if commodity_type:
        result["commodity_multiplier"] = (
            penalty.get("commodity_multipliers", {}).get(commodity_type, 1.0)
        )
    return result


def get_article_requirements(article: str) -> Optional[Dict[str, Any]]:
    """Get full article data including all commodity requirements.

    Args:
        article: Article key (e.g., "article_3", "article_9").

    Returns:
        Full article data dict or None if not found.
    """
    return EUDR_ARTICLES.get(article)


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    "DATA_VERSION",
    "DATA_SOURCES",
    "EUDR_ARTICLES",
    "DOCUMENTATION_REQUIREMENTS",
    "PENALTY_MATRIX",
    "RegulatoryRequirementDatabase",
    "get_requirements",
    "get_documentation_list",
    "get_penalty_info",
    "get_article_requirements",
]
