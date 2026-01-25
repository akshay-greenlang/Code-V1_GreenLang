# -*- coding: utf-8 -*-
"""
greenlang/regulations/eudr/validators.py

EUDR Validation Functions

This module provides validation functions for EUDR compliance.
All validations are DETERMINISTIC - same inputs always produce same outputs.

ZERO-HALLUCINATION GUARANTEE:
- All validations use fixed rules from EUDR regulation
- No LLM involvement in validation logic
- Complete audit trail for all validations
- Bit-perfect reproducibility

Reference: Regulation (EU) 2023/1115

Author: GreenLang Framework Team
Date: November 2025
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Optional, Tuple
import logging

from greenlang.regulations.eudr.models import (
    GeolocationData,
    ProductionPlot,
    EUDRProduct,
    EUDRCommodity,
    EUDRComplianceStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    validation_type: str
    message: str
    errors: List[str]
    warnings: List[str]
    details: Dict

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "validation_type": self.validation_type,
            "message": self.message,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


def validate_geolocation(geolocation: GeolocationData) -> ValidationResult:
    """
    Validate geolocation data per EUDR Article 9 requirements.

    Requirements:
    - Valid WGS84 coordinates
    - Polygon for plots > 4 hectares
    - Precision sufficient for verification

    Args:
        geolocation: GeolocationData to validate

    Returns:
        ValidationResult with validation outcome
    """
    errors = []
    warnings = []

    # Check coordinate validity
    if not (-90 <= geolocation.latitude <= 90):
        errors.append(f"Invalid latitude: {geolocation.latitude} (must be -90 to 90)")

    if not (-180 <= geolocation.longitude <= 180):
        errors.append(f"Invalid longitude: {geolocation.longitude} (must be -180 to 180)")

    # Check for null island (common error)
    if geolocation.latitude == 0 and geolocation.longitude == 0:
        errors.append("Coordinates are at null island (0, 0) - likely invalid")

    # Check polygon for large plots
    if geolocation.plot_area_hectares and geolocation.plot_area_hectares > 4.0:
        if not geolocation.polygon_coordinates:
            errors.append(
                f"Polygon required for plots > 4 hectares (plot is {geolocation.plot_area_hectares} ha)"
            )
        elif len(geolocation.polygon_coordinates) < 3:
            errors.append("Polygon must have at least 3 coordinates")

    # Check coordinate precision
    if geolocation.coordinate_precision < 4:
        warnings.append(
            f"Low coordinate precision ({geolocation.coordinate_precision} decimals). "
            "EUDR recommends at least 4 decimal places for verification."
        )

    # Check country code
    if not geolocation.country_code:
        warnings.append("Country code not specified - required for risk assessment")

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        validation_type="geolocation",
        message="Geolocation is valid" if is_valid else "Geolocation validation failed",
        errors=errors,
        warnings=warnings,
        details={
            "latitude": geolocation.latitude,
            "longitude": geolocation.longitude,
            "has_polygon": geolocation.polygon_coordinates is not None,
            "plot_area_hectares": geolocation.plot_area_hectares,
            "country_code": geolocation.country_code,
        }
    )


def validate_deforestation_free(
    production_plot: ProductionPlot,
    reference_date: Optional[date] = None
) -> ValidationResult:
    """
    Validate that a production plot is deforestation-free per EUDR Article 3.

    EUDR Cut-off Date: December 31, 2020
    - Products must not be produced on land that was deforested after this date
    - Products must not cause forest degradation

    Args:
        production_plot: ProductionPlot to validate
        reference_date: Optional reference date for validation

    Returns:
        ValidationResult with deforestation-free status
    """
    errors = []
    warnings = []
    cutoff_date = date(2020, 12, 31)

    # Check if production date is after cutoff
    if production_plot.production_date > cutoff_date:
        # Need to verify deforestation status
        if production_plot.was_forested_after_cutoff is None:
            errors.append(
                "Cannot determine deforestation status: was_forested_after_cutoff not specified"
            )
        elif production_plot.was_forested_after_cutoff:
            errors.append(
                f"Plot was forested after EUDR cutoff date ({cutoff_date}). "
                "Production on this land does not comply with EUDR."
            )

        if not production_plot.deforestation_verified:
            warnings.append(
                "Deforestation status has not been independently verified"
            )

    # Check forest degradation
    if not production_plot.forest_degradation_verified:
        warnings.append(
            "Forest degradation status has not been verified"
        )

    # Check for supporting evidence
    if production_plot.is_deforestation_free() and not production_plot.certification_scheme:
        warnings.append(
            "No certification scheme linked - consider adding third-party verification"
        )

    is_valid = len(errors) == 0 and production_plot.is_deforestation_free()

    return ValidationResult(
        is_valid=is_valid,
        validation_type="deforestation_free",
        message="Plot is deforestation-free" if is_valid else "Deforestation-free check failed",
        errors=errors,
        warnings=warnings,
        details={
            "plot_id": production_plot.plot_id,
            "production_date": production_plot.production_date.isoformat(),
            "cutoff_date": cutoff_date.isoformat(),
            "was_forested_after_cutoff": production_plot.was_forested_after_cutoff,
            "deforestation_verified": production_plot.deforestation_verified,
            "forest_degradation_verified": production_plot.forest_degradation_verified,
            "is_deforestation_free": production_plot.is_deforestation_free(),
        }
    )


def validate_production_legality(production_plot: ProductionPlot) -> ValidationResult:
    """
    Validate that production is in accordance with relevant legislation (EUDR Article 3).

    Checks:
    - Legal production confirmation
    - Supporting evidence provided
    - Country of production specified

    Args:
        production_plot: ProductionPlot to validate

    Returns:
        ValidationResult with legality status
    """
    errors = []
    warnings = []

    # Check legal production flag
    if not production_plot.is_legally_produced:
        errors.append(
            "Production is not marked as legally produced"
        )

    # Check country of production
    if not production_plot.production_country:
        errors.append(
            "Country of production not specified - required for EUDR compliance"
        )

    # Check for evidence
    if production_plot.is_legally_produced and not production_plot.legal_compliance_evidence:
        warnings.append(
            "Legal production confirmed but no supporting evidence provided"
        )

    # Check supplier information
    if not production_plot.supplier_id:
        warnings.append(
            "Supplier ID not specified - may impact traceability"
        )

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        validation_type="production_legality",
        message="Production is legal" if is_valid else "Legality check failed",
        errors=errors,
        warnings=warnings,
        details={
            "plot_id": production_plot.plot_id,
            "production_country": production_plot.production_country,
            "is_legally_produced": production_plot.is_legally_produced,
            "has_evidence": production_plot.legal_compliance_evidence is not None,
            "supplier_id": production_plot.supplier_id,
        }
    )


def validate_product_compliance(product: EUDRProduct) -> ValidationResult:
    """
    Validate complete EUDR compliance for a product.

    Performs all required checks:
    1. Geolocation validation for all plots
    2. Deforestation-free validation for all plots
    3. Legal production validation for all plots
    4. Traceability completeness

    Args:
        product: EUDRProduct to validate

    Returns:
        ValidationResult with overall compliance status
    """
    errors = []
    warnings = []
    plot_results = []

    # Check for production plots
    if not product.production_plots:
        errors.append(
            "No production plots linked to product - EUDR requires full traceability"
        )
    else:
        # Validate each production plot
        for plot in product.production_plots:
            # Geolocation
            geo_result = validate_geolocation(plot.geolocation)
            if not geo_result.is_valid:
                errors.extend([f"Plot {plot.plot_id}: {e}" for e in geo_result.errors])
            warnings.extend([f"Plot {plot.plot_id}: {w}" for w in geo_result.warnings])

            # Deforestation-free
            defo_result = validate_deforestation_free(plot)
            if not defo_result.is_valid:
                errors.extend([f"Plot {plot.plot_id}: {e}" for e in defo_result.errors])
            warnings.extend([f"Plot {plot.plot_id}: {w}" for w in defo_result.warnings])

            # Legal production
            legal_result = validate_production_legality(plot)
            if not legal_result.is_valid:
                errors.extend([f"Plot {plot.plot_id}: {e}" for e in legal_result.errors])
            warnings.extend([f"Plot {plot.plot_id}: {w}" for w in legal_result.warnings])

            plot_results.append({
                "plot_id": plot.plot_id,
                "geolocation_valid": geo_result.is_valid,
                "deforestation_free": defo_result.is_valid,
                "legally_produced": legal_result.is_valid,
            })

    # Check HS code
    if not product.hs_code:
        errors.append("HS code not specified - required for customs declaration")

    # Check traceability
    if not product.is_fully_traceable():
        errors.append("Product is not fully traceable - all plots must have valid geolocations")

    is_valid = len(errors) == 0

    # Determine compliance status
    if is_valid:
        status = EUDRComplianceStatus.COMPLIANT
    elif errors:
        status = EUDRComplianceStatus.NON_COMPLIANT
    else:
        status = EUDRComplianceStatus.PENDING_VERIFICATION

    return ValidationResult(
        is_valid=is_valid,
        validation_type="product_compliance",
        message=f"Product compliance: {status.value}",
        errors=errors,
        warnings=warnings,
        details={
            "product_id": product.product_id,
            "commodity": product.commodity.value,
            "hs_code": product.hs_code,
            "production_plots_count": len(product.production_plots),
            "is_fully_traceable": product.is_fully_traceable(),
            "countries_of_production": product.get_countries_of_production(),
            "compliance_status": status.value,
            "plot_validations": plot_results,
        }
    )
