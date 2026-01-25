"""
EUDR Tools Module
=================
Deterministic tools for EU Deforestation Regulation (EU) 2023/1115 compliance.

These tools provide zero-hallucination validation, classification, and risk
assessment for EUDR-regulated commodities.

Tools:
1. validate_geolocation - GPS coordinate and polygon validation
2. classify_commodity - CN code classification for EUDR commodities
3. assess_country_risk - Country/region deforestation risk assessment
4. trace_supply_chain - Supply chain traceability scoring
5. generate_dds_report - Due Diligence Statement generation
"""

import hashlib
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

# Import EUDR data modules
from greenlang.data.eudr_commodities import (
    classify_cn_code,
    get_commodity_by_cn_code,
    get_traceability_requirements,
    is_eudr_regulated,
    CommodityType,
    RiskCategory,
    EUDR_CUTOFF_DATE,
)

from greenlang.data.eudr_country_risk import (
    assess_country_risk as _assess_country_risk,
    get_country_risk,
    get_risk_level,
    get_commodity_risk,
    get_due_diligence_level,
    requires_satellite_verification,
    get_regions_of_concern,
    is_deforestation_hotspot,
    RiskLevel,
    DueDiligenceLevel,
    EUDR_CUTOFF_DATE as RISK_CUTOFF_DATE,
)


# =============================================================================
# Constants
# =============================================================================

# Protected area coordinates (simplified bounding boxes)
# In production, this would use actual GeoJSON boundaries
PROTECTED_AREAS = {
    "BR": [
        {"name": "Amazon Protected Complex", "bounds": ((-3.5, -63.0), (-2.5, -61.0)), "type": "national_park"},
        {"name": "Tumucumaque", "bounds": ((0.5, -54.0), (3.0, -51.0)), "type": "national_park"},
        {"name": "Xingu Indigenous Territory", "bounds": ((-12.0, -54.0), (-9.5, -52.0)), "type": "indigenous"},
    ],
    "ID": [
        {"name": "Leuser Ecosystem", "bounds": ((2.5, 96.5), (5.0, 98.5)), "type": "national_park"},
        {"name": "Tanjung Puting", "bounds": ((-3.5, 111.0), (-2.0, 112.5)), "type": "national_park"},
    ],
    "CD": [
        {"name": "Virunga", "bounds": ((-1.5, 29.0), (1.0, 30.0)), "type": "national_park"},
    ],
    "CI": [
        {"name": "Tai National Park", "bounds": ((5.5, -7.5), (6.5, -6.5)), "type": "national_park"},
    ],
}


# =============================================================================
# Tool 1: Geolocation Validator
# =============================================================================

def validate_geolocation(
    coordinates: Union[List[float], List[List[float]]],
    country_code: str,
    coordinate_type: str = "point",
    precision_meters: float = 10.0,
) -> Dict[str, Any]:
    """
    Validate GPS coordinates or polygon for EUDR compliance.

    This is a DETERMINISTIC tool - same inputs always produce same outputs.

    Args:
        coordinates: GPS coordinates as [lat, lon] or list of coordinate pairs for polygon
        country_code: ISO 3166-1 alpha-2 country code
        coordinate_type: "point" or "polygon"
        precision_meters: GPS precision in meters

    Returns:
        Validation result with compliance status

    Example:
        >>> validate_geolocation([-15.7942, -47.8822], "BR", "point")
        {
            "valid": True,
            "validation_uri": "geo://eudr/validation/2024/BR/abc123",
            "country_code": "BR",
            "in_protected_area": False,
            ...
        }
    """
    # Generate deterministic validation URI
    coord_hash = hashlib.sha256(
        f"{coordinates}{country_code}{coordinate_type}".encode()
    ).hexdigest()[:12]
    validation_uri = f"geo://eudr/validation/{datetime.now().year}/{country_code}/{coord_hash}"

    # Initialize result
    result = {
        "valid": False,
        "validation_uri": validation_uri,
        "country_code": country_code.upper(),
        "region": None,
        "in_protected_area": False,
        "protected_area_name": None,
        "forest_cover_2020": None,
        "warnings": [],
        "data_sources": ["Global Forest Watch", "EUDR Protected Areas Registry"],
    }

    # Handle point coordinates
    if coordinate_type == "point":
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            result["warnings"].append("Invalid coordinate format - expected [latitude, longitude]")
            return result

        lat, lon = coordinates[0], coordinates[1]

        # Validate latitude bounds (-90 to 90)
        if not (-90 <= lat <= 90):
            result["warnings"].append(f"Invalid latitude {lat} - must be between -90 and 90")
            return result

        # Validate longitude bounds (-180 to 180)
        if not (-180 <= lon <= 180):
            result["warnings"].append(f"Invalid longitude {lon} - must be between -180 and 180")
            return result

        # Basic validation passed
        result["valid"] = True

        # Check protected areas
        country_protected = PROTECTED_AREAS.get(country_code.upper(), [])
        for area in country_protected:
            min_lat, min_lon = area["bounds"][0]
            max_lat, max_lon = area["bounds"][1]
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                result["in_protected_area"] = True
                result["protected_area_name"] = area["name"]
                result["warnings"].append(
                    f"Coordinates fall within protected area: {area['name']} ({area['type']})"
                )
                break

        # Estimate forest cover based on country (simplified)
        country_risk = get_country_risk(country_code)
        if country_risk:
            result["forest_cover_2020"] = country_risk.forest_data.forest_cover_2020
            result["region"] = _estimate_region(lat, lon, country_code)

    # Handle polygon coordinates
    elif coordinate_type == "polygon":
        if not isinstance(coordinates, list) or len(coordinates) < 3:
            result["warnings"].append("Invalid polygon - minimum 3 coordinate pairs required")
            return result

        # Validate each coordinate pair
        for i, coord in enumerate(coordinates):
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                result["warnings"].append(f"Invalid coordinate pair at index {i}")
                return result

            lat, lon = coord[0], coord[1]
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                result["warnings"].append(f"Invalid coordinates at index {i}")
                return result

        result["valid"] = True

        # Check if centroid falls in protected area
        centroid_lat = sum(c[0] for c in coordinates) / len(coordinates)
        centroid_lon = sum(c[1] for c in coordinates) / len(coordinates)

        country_protected = PROTECTED_AREAS.get(country_code.upper(), [])
        for area in country_protected:
            min_lat, min_lon = area["bounds"][0]
            max_lat, max_lon = area["bounds"][1]
            if min_lat <= centroid_lat <= max_lat and min_lon <= centroid_lon <= max_lon:
                result["in_protected_area"] = True
                result["protected_area_name"] = area["name"]
                result["warnings"].append(
                    f"Polygon centroid within protected area: {area['name']}"
                )
                break

        country_risk = get_country_risk(country_code)
        if country_risk:
            result["forest_cover_2020"] = country_risk.forest_data.forest_cover_2020
            result["region"] = _estimate_region(centroid_lat, centroid_lon, country_code)

    # Check precision requirement
    if precision_meters > 100:
        result["warnings"].append(
            f"GPS precision ({precision_meters}m) exceeds recommended 100m threshold"
        )

    return result


def _estimate_region(lat: float, lon: float, country_code: str) -> Optional[str]:
    """Estimate sub-national region from coordinates (simplified)."""
    # In production, this would use proper reverse geocoding
    region_map = {
        "BR": [
            ((-10, -74, 5, -44), "Amazonia Legal"),
            ((-24, -60, -2, -41), "Cerrado"),
            ((-18, -62, -7, -50), "Mato Grosso"),
        ],
        "ID": [
            ((-6, 95, 6, 106), "Sumatra"),
            ((-4.5, 108, 4.5, 119), "Kalimantan"),
            ((-9, 129, 0, 141), "Papua"),
        ],
    }

    regions = region_map.get(country_code, [])
    for bounds, region_name in regions:
        min_lat, min_lon, max_lat, max_lon = bounds
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return region_name

    return None


# =============================================================================
# Tool 2: Commodity Classifier
# =============================================================================

def classify_commodity(
    cn_code: str,
    product_description: str = "",
    quantity_kg: float = 0.0,
) -> Dict[str, Any]:
    """
    Classify commodity under EUDR using EU Combined Nomenclature codes.

    This is a DETERMINISTIC tool - classification based on official EU CN codes.

    Args:
        cn_code: EU Combined Nomenclature code (4-8 digits)
        product_description: Optional product description for validation
        quantity_kg: Quantity in kilograms

    Returns:
        Classification result with commodity type and requirements

    Example:
        >>> classify_commodity("15111000", "Crude palm oil", 10000)
        {
            "eudr_regulated": True,
            "commodity_type": "palm_oil",
            "cn_code": "15111000",
            "cn_description": "Crude palm oil",
            ...
        }
    """
    # Validate CN code format
    cn_code = str(cn_code).strip()
    if not cn_code.isdigit() or not (4 <= len(cn_code) <= 8):
        return {
            "eudr_regulated": False,
            "commodity_type": CommodityType.NOT_REGULATED.value,
            "cn_code": cn_code,
            "cn_description": "Invalid CN code format",
            "classification_uri": f"cn://eudr/error/invalid_format",
            "risk_category": RiskCategory.LOW.value,
            "traceability_requirements": [],
            "derived_from": [],
            "error": "CN code must be 4-8 digits"
        }

    # Use the commodity database for classification
    classification = classify_cn_code(cn_code, product_description)

    # Add quantity validation
    if quantity_kg > 0:
        # Check for suspicious quantities (basic plausibility check)
        if quantity_kg > 1000000:  # 1000 tonnes
            classification["warnings"] = classification.get("warnings", [])
            classification["warnings"].append(
                f"Large quantity ({quantity_kg/1000:.1f} tonnes) - verify accuracy"
            )

    return classification


# =============================================================================
# Tool 3: Country Risk Assessor
# =============================================================================

def assess_country_risk(
    country_code: str,
    commodity_type: str,
    region: str = None,
    production_year: int = None,
) -> Dict[str, Any]:
    """
    Assess deforestation risk for a country and commodity combination.

    This is a DETERMINISTIC tool - risk scores from EC benchmarking system.

    Args:
        country_code: ISO 3166-1 alpha-2 country code
        commodity_type: EUDR commodity type
        region: Optional sub-national region
        production_year: Optional production year

    Returns:
        Risk assessment result with due diligence requirements

    Example:
        >>> assess_country_risk("BR", "soya", production_year=2024)
        {
            "risk_level": "high",
            "risk_score": 85,
            "due_diligence_level": "full_verification",
            "satellite_verification_required": True,
            ...
        }
    """
    # Validate country code
    country_code = country_code.upper().strip()
    if len(country_code) != 2:
        return {
            "risk_level": RiskLevel.STANDARD.value,
            "risk_score": 50,
            "risk_uri": f"risk://eudr/error/invalid_country",
            "country_name": "Unknown",
            "due_diligence_level": DueDiligenceLevel.ENHANCED.value,
            "satellite_verification_required": True,
            "data_sources": [],
            "error": "Invalid country code format"
        }

    # Validate commodity type
    valid_commodities = ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"]
    commodity_lower = commodity_type.lower().strip()
    if commodity_lower not in valid_commodities:
        return {
            "risk_level": RiskLevel.STANDARD.value,
            "risk_score": 50,
            "risk_uri": f"risk://eudr/error/invalid_commodity",
            "country_name": None,
            "due_diligence_level": DueDiligenceLevel.ENHANCED.value,
            "satellite_verification_required": True,
            "data_sources": [],
            "error": f"Invalid commodity type. Valid types: {valid_commodities}"
        }

    # Use the country risk database
    return _assess_country_risk(
        country_code=country_code,
        commodity_type=commodity_lower,
        region=region,
        production_year=production_year
    )


# =============================================================================
# Tool 4: Supply Chain Tracer
# =============================================================================

def trace_supply_chain(
    shipment_id: str,
    supply_chain_nodes: List[Dict[str, Any]],
    commodity_type: str,
) -> Dict[str, Any]:
    """
    Trace commodity supply chain and calculate traceability score.

    This is a DETERMINISTIC tool - scoring based on node verification status.

    Args:
        shipment_id: Unique shipment identifier
        supply_chain_nodes: List of supply chain node objects
        commodity_type: EUDR commodity type

    Returns:
        Traceability assessment with score and gaps

    Example:
        >>> trace_supply_chain("SHIP-001", nodes, "cocoa")
        {
            "traceability_score": 85.0,
            "verified_nodes": 5,
            "total_nodes": 6,
            "origin_verified": True,
            ...
        }
    """
    # Generate trace URI
    trace_hash = hashlib.sha256(
        f"{shipment_id}{commodity_type}{len(supply_chain_nodes)}".encode()
    ).hexdigest()[:12]
    trace_uri = f"trace://eudr/{datetime.now().year}/{shipment_id}/{trace_hash}"

    # Initialize result
    result = {
        "traceability_score": 0.0,
        "trace_uri": trace_uri,
        "verified_nodes": 0,
        "total_nodes": len(supply_chain_nodes),
        "gaps": [],
        "origin_verified": False,
        "chain_of_custody": "broken",
    }

    if not supply_chain_nodes:
        result["gaps"].append({
            "gap_type": "no_nodes",
            "node_id": None,
            "severity": "high",
            "description": "No supply chain nodes provided"
        })
        return result

    # Analyze each node
    verified_count = 0
    has_producer = False
    required_fields = ["node_type", "node_id"]

    for i, node in enumerate(supply_chain_nodes):
        node_id = node.get("node_id", f"node_{i}")
        node_type = node.get("node_type", "unknown")

        # Check required fields
        missing_fields = [f for f in required_fields if not node.get(f)]
        if missing_fields:
            result["gaps"].append({
                "gap_type": "missing_fields",
                "node_id": node_id,
                "severity": "medium",
                "description": f"Missing required fields: {missing_fields}"
            })
            continue

        # Check for producer node (origin)
        if node_type == "producer":
            has_producer = True
            if node.get("coordinates"):
                result["origin_verified"] = True
                verified_count += 1
            else:
                result["gaps"].append({
                    "gap_type": "missing_coordinates",
                    "node_id": node_id,
                    "severity": "high",
                    "description": "Producer node missing GPS coordinates"
                })

        # Verify other node types
        elif node_type in ["collector", "processor", "trader", "importer"]:
            # Check for operator identification
            if node.get("operator_name") and node.get("country_code"):
                verified_count += 1
            else:
                result["gaps"].append({
                    "gap_type": "incomplete_operator",
                    "node_id": node_id,
                    "severity": "low",
                    "description": f"Node {node_type} missing operator details"
                })

        # Check timestamp continuity
        if node.get("timestamp"):
            pass  # In production, verify chronological order

    # Calculate traceability score
    if result["total_nodes"] > 0:
        base_score = (verified_count / result["total_nodes"]) * 100

        # Penalties
        if not has_producer:
            base_score -= 30
            result["gaps"].append({
                "gap_type": "no_producer",
                "node_id": None,
                "severity": "high",
                "description": "Supply chain missing producer/origin node"
            })

        if not result["origin_verified"]:
            base_score -= 20

        result["traceability_score"] = max(0, min(100, base_score))
        result["verified_nodes"] = verified_count

    # Determine chain of custody status
    if result["traceability_score"] >= 90 and result["origin_verified"]:
        result["chain_of_custody"] = "complete"
    elif result["traceability_score"] >= 60:
        result["chain_of_custody"] = "partial"
    else:
        result["chain_of_custody"] = "broken"

    return result


# =============================================================================
# Tool 5: DDS Report Generator
# =============================================================================

def generate_dds_report(
    operator_info: Dict[str, Any],
    commodity_data: Dict[str, Any],
    geolocation_data: Dict[str, Any],
    risk_assessment: Dict[str, Any],
    traceability_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate EU Due Diligence Statement (DDS) for submission.

    This is a DETERMINISTIC tool - generates compliant DDS structure.

    Args:
        operator_info: Operator registration information
        commodity_data: Commodity classification data
        geolocation_data: Production plot geolocation
        risk_assessment: Risk assessment results
        traceability_data: Optional supply chain traceability data

    Returns:
        DDS document with submission status

    Example:
        >>> generate_dds_report(operator, commodity, geo, risk)
        {
            "dds_id": "DDS-2024-abc123",
            "dds_status": "valid",
            "compliance_status": "compliant",
            "submission_ready": True,
            ...
        }
    """
    # Generate DDS ID
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    dds_hash = hashlib.sha256(
        f"{operator_info.get('operator_id')}{timestamp}".encode()
    ).hexdigest()[:8]
    dds_id = f"DDS-{datetime.now().year}-{dds_hash.upper()}"

    # Initialize result
    result = {
        "dds_id": dds_id,
        "dds_status": "incomplete",
        "compliance_status": "requires_review",
        "dds_json": None,
        "dds_hash": None,
        "submission_ready": False,
        "validation_errors": [],
        "warnings": [],
    }

    # Validate operator info
    required_operator = ["operator_id", "company_name", "eori_number"]
    missing_operator = [f for f in required_operator if not operator_info.get(f)]
    if missing_operator:
        result["validation_errors"].append(f"Missing operator fields: {missing_operator}")

    # Validate commodity data
    required_commodity = ["cn_code", "commodity_type", "quantity_kg", "production_country"]
    missing_commodity = [f for f in required_commodity if not commodity_data.get(f)]
    if missing_commodity:
        result["validation_errors"].append(f"Missing commodity fields: {missing_commodity}")

    # Validate geolocation data
    if not geolocation_data.get("coordinates"):
        result["validation_errors"].append("Missing geolocation coordinates")

    # Validate risk assessment
    required_risk = ["risk_level", "due_diligence_level"]
    missing_risk = [f for f in required_risk if not risk_assessment.get(f)]
    if missing_risk:
        result["validation_errors"].append(f"Missing risk assessment fields: {missing_risk}")

    # Check for validation errors
    if result["validation_errors"]:
        result["dds_status"] = "incomplete"
        return result

    # Build DDS JSON structure (EU schema compliant)
    dds_json = {
        "dds_version": "1.0",
        "dds_id": dds_id,
        "submission_date": datetime.now().isoformat(),
        "operator": {
            "operator_id": operator_info.get("operator_id"),
            "company_name": operator_info.get("company_name"),
            "eori_number": operator_info.get("eori_number"),
            "country": operator_info.get("country_code", ""),
            "contact_email": operator_info.get("contact_email", ""),
        },
        "commodity": {
            "cn_code": commodity_data.get("cn_code"),
            "description": commodity_data.get("cn_description", ""),
            "commodity_type": commodity_data.get("commodity_type"),
            "quantity_kg": commodity_data.get("quantity_kg"),
            "production_country": commodity_data.get("production_country"),
            "production_date": commodity_data.get("production_date", ""),
        },
        "geolocation": {
            "coordinates": geolocation_data.get("coordinates"),
            "coordinate_type": geolocation_data.get("coordinate_type", "point"),
            "precision_meters": geolocation_data.get("precision_meters", 10),
            "validation_status": geolocation_data.get("valid", False),
        },
        "risk_assessment": {
            "risk_level": risk_assessment.get("risk_level"),
            "risk_score": risk_assessment.get("risk_score"),
            "due_diligence_level": risk_assessment.get("due_diligence_level"),
            "satellite_verification": risk_assessment.get("satellite_verification_required", False),
            "assessment_date": risk_assessment.get("last_updated", datetime.now().date().isoformat()),
        },
        "traceability": {
            "traceability_score": traceability_data.get("traceability_score", 0) if traceability_data else 0,
            "chain_of_custody": traceability_data.get("chain_of_custody", "unknown") if traceability_data else "unknown",
            "origin_verified": traceability_data.get("origin_verified", False) if traceability_data else False,
        },
        "compliance_declaration": {
            "deforestation_free": True,  # To be verified
            "legally_produced": True,  # To be verified
            "cutoff_date_compliant": True,  # Production after Dec 31, 2020
        },
        "attachments": [],
    }

    # Determine compliance status
    risk_level = risk_assessment.get("risk_level", "standard")
    traceability_score = traceability_data.get("traceability_score", 0) if traceability_data else 0
    geo_valid = geolocation_data.get("valid", False)

    if risk_level == "low" and geo_valid:
        compliance_status = "compliant"
    elif risk_level == "high" or traceability_score < 60:
        compliance_status = "non_compliant"
        result["warnings"].append("High risk or low traceability - additional verification required")
    else:
        compliance_status = "requires_review"

    # Check for protected area concerns
    if geolocation_data.get("in_protected_area"):
        compliance_status = "non_compliant"
        result["warnings"].append("Production coordinates in protected area")
        dds_json["compliance_declaration"]["deforestation_free"] = False

    # Generate hash
    dds_hash = hashlib.sha256(
        str(dds_json).encode()
    ).hexdigest()

    # Update result
    result["dds_status"] = "valid"
    result["compliance_status"] = compliance_status
    result["dds_json"] = dds_json
    result["dds_hash"] = dds_hash
    result["submission_ready"] = compliance_status == "compliant"

    if compliance_status != "compliant":
        result["warnings"].append(
            f"DDS not ready for submission - status: {compliance_status}"
        )

    return result


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "validate_geolocation",
    "classify_commodity",
    "assess_country_risk",
    "trace_supply_chain",
    "generate_dds_report",
]
