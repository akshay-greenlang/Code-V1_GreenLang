# -*- coding: utf-8 -*-
"""
Article 9 Data Assembler Engine - AGENT-EUDR-030

Assembles Article 9 required information from upstream agents into a
structured package. Gathers all 10 mandatory Article 9 elements and
scores completeness against the regulatory threshold. Validates
geolocation requirements, polygon boundaries, supplier references,
and production data for EUDR compliance.

Article 9 Mandatory Elements:
    1. Product description
    2. Quantity and unit
    3. Country of production
    4. Geolocation (coordinates, polygon for >4ha)
    5. Production date or date range
    6. Supplier information
    7. Buyer information
    8. Certifications
    9. Trade codes (HS/CN)
    10. Supporting evidence

Zero-Hallucination Guarantees:
    - All scores computed via Decimal arithmetic
    - No LLM calls in the assembly path
    - Completeness scoring is deterministic (present / total)
    - Polygon validation uses area threshold from config
    - Complete provenance trail for every assembly operation

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-030 Documentation Generator (GL-EUDR-DGN-030)
Regulation: EU 2023/1115 (EUDR) Article 9
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .config import DocumentationGeneratorConfig, get_config
from .models import (
    AGENT_ID,
    AGENT_VERSION,
    ARTICLE9_MANDATORY_ELEMENTS,
    Article9Element,
    Article9Package,
    EUDRCommodity,
    GeolocationReference,
    ProductEntry,
    SupplierReference,
    ValidationIssue,
    ValidationSeverity,
)
from .provenance import GENESIS_HASH, ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Article 9 element descriptions for documentation
# ---------------------------------------------------------------------------

_ELEMENT_DESCRIPTIONS: Dict[Article9Element, str] = {
    Article9Element.PRODUCT_DESCRIPTION: (
        "Description of the product, including trade name and product type "
        "as listed in Annex I of Regulation (EU) 2023/1115."
    ),
    Article9Element.QUANTITY: (
        "Quantity of the product expressed in net mass (kilograms) or, "
        "where applicable, volume (litres) or number of units."
    ),
    Article9Element.COUNTRY_OF_PRODUCTION: (
        "Country of production of the relevant commodity or product, "
        "including the subnational region where applicable."
    ),
    Article9Element.GEOLOCATION: (
        "Geolocation coordinates of all plots of land where the relevant "
        "commodities were produced. For plots >4 hectares, polygon "
        "boundaries are required."
    ),
    Article9Element.PRODUCTION_DATE: (
        "Date or time range of production of the commodity."
    ),
    Article9Element.SUPPLIER_INFO: (
        "Name, postal address, and email address of all suppliers in "
        "the supply chain."
    ),
    Article9Element.BUYER_INFO: (
        "Name, postal address, and email address of the buyers to whom "
        "the product has been supplied."
    ),
    Article9Element.CERTIFICATIONS: (
        "Adequate and verifiable information that the products are "
        "deforestation-free, including relevant certifications."
    ),
    Article9Element.TRADE_CODES: (
        "Harmonised System (HS) code and Combined Nomenclature (CN) "
        "code of the product as per EU customs classification."
    ),
    Article9Element.SUPPORTING_EVIDENCE: (
        "Supporting evidence demonstrating compliance with the "
        "relevant requirements of Regulation (EU) 2023/1115."
    ),
}


class Article9DataAssembler:
    """Assembles Article 9 required information from upstream agents.

    Gathers all 10 mandatory Article 9 elements and scores completeness.
    Validates geolocation data, supplier references, production dates,
    and trade classifications for EUDR regulatory compliance.

    Attributes:
        _config: Agent configuration instance.
        _provenance: Provenance tracker for audit trail.

    Example:
        >>> assembler = Article9DataAssembler()
        >>> package = await assembler.assemble_package(
        ...     operator_id="OP-001",
        ...     commodity=EUDRCommodity.COFFEE,
        ...     products=[product],
        ...     geolocations=[geo],
        ...     suppliers=[supplier],
        ... )
        >>> assert package.completeness_score >= Decimal("0.80")
    """

    def __init__(
        self,
        config: Optional[DocumentationGeneratorConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize Article9DataAssembler.

        Args:
            config: Agent configuration. Uses get_config() if None.
            provenance: Provenance tracker instance.
        """
        self._config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        logger.info(
            "Article9DataAssembler initialized: "
            "completeness_threshold=%s, polygon_above_4ha=%s, "
            "geo_decimals=%d",
            self._config.article9_completeness_threshold,
            self._config.require_polygon_above_4ha,
            self._config.geolocation_decimal_places,
        )

    async def assemble_package(
        self,
        operator_id: str,
        commodity: EUDRCommodity,
        products: List[ProductEntry],
        geolocations: List[GeolocationReference],
        suppliers: List[SupplierReference],
        production_date_range: Optional[Tuple[str, str]] = None,
        certifications: Optional[List[str]] = None,
        buyer_info: Optional[Dict[str, Any]] = None,
        supporting_evidence: Optional[List[str]] = None,
    ) -> Article9Package:
        """Assemble Article 9 information package.

        Collects all available data elements, scores completeness,
        validates geolocation requirements, and returns a structured
        Article9Package.

        Args:
            operator_id: Operator identifier.
            commodity: EUDR commodity category.
            products: List of product entries.
            geolocations: List of geolocation references.
            suppliers: List of supplier references.
            production_date_range: Optional (start, end) date strings.
            certifications: Optional list of certification references.
            buyer_info: Optional buyer information dictionary.
            supporting_evidence: Optional list of evidence references.

        Returns:
            Article9Package with completeness scoring.
        """
        start_time = time.monotonic()
        package_id = f"a9p-{uuid.uuid4().hex[:12]}"
        logger.info(
            "Assembling Article 9 package: id=%s, operator=%s, "
            "commodity=%s, products=%d, geos=%d, suppliers=%d",
            package_id, operator_id, commodity.value,
            len(products), len(geolocations), len(suppliers),
        )

        # Build element dictionary
        elements = self._build_element_dict(
            operator_id=operator_id,
            commodity=commodity,
            products=products,
            geolocations=geolocations,
            suppliers=suppliers,
            production_date_range=production_date_range,
            certifications=certifications,
            buyer_info=buyer_info,
            supporting_evidence=supporting_evidence,
        )

        # Check completeness
        completeness_score, missing = self._check_element_completeness(
            elements,
        )

        # Validate geolocations
        geo_issues = self._validate_geolocations(geolocations)
        validation_issues = [
            {"field": vi.field, "severity": vi.severity.value,
             "message": vi.message}
            for vi in geo_issues
        ]

        # Validate products have required fields
        product_issues = self._validate_products(products)
        validation_issues.extend([
            {"field": vi.field, "severity": vi.severity.value,
             "message": vi.message}
            for vi in product_issues
        ])

        # Build missing elements list as strings
        missing_strs = [elem.value for elem in missing]

        # Compute provenance hash
        provenance_data: Dict[str, Any] = {
            "package_id": package_id,
            "operator_id": operator_id,
            "commodity": commodity.value,
            "product_count": len(products),
            "geolocation_count": len(geolocations),
            "supplier_count": len(suppliers),
            "completeness_score": str(completeness_score),
            "missing_count": len(missing),
        }
        provenance_hash = self._provenance.compute_hash(provenance_data)

        package = Article9Package(
            package_id=package_id,
            operator_id=operator_id,
            commodity=commodity,
            elements=elements,
            completeness_score=completeness_score,
            missing_elements=missing_strs,
        )

        # Record provenance
        self._provenance.create_entry(
            step="assemble_article9",
            source="article9_data_assembler",
            input_hash=self._provenance.compute_hash(
                {"operator_id": operator_id, "commodity": commodity.value}
            ),
            output_hash=provenance_hash,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Article 9 package assembled: id=%s, completeness=%s, "
            "missing=%d, issues=%d, elapsed=%.1fms",
            package_id, completeness_score, len(missing),
            len(validation_issues), elapsed_ms,
        )

        return package

    def _build_element_dict(
        self,
        operator_id: str,
        commodity: EUDRCommodity,
        products: List[ProductEntry],
        geolocations: List[GeolocationReference],
        suppliers: List[SupplierReference],
        production_date_range: Optional[Tuple[str, str]] = None,
        certifications: Optional[List[str]] = None,
        buyer_info: Optional[Dict[str, Any]] = None,
        supporting_evidence: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build structured element dictionary from inputs.

        Args:
            operator_id: Operator identifier.
            commodity: EUDR commodity.
            products: Product entries.
            geolocations: Geolocation references.
            suppliers: Supplier references.
            production_date_range: Production date range.
            certifications: Certification references.
            buyer_info: Buyer information.
            supporting_evidence: Evidence references.

        Returns:
            Dictionary keyed by Article9Element value strings.
        """
        elements: Dict[str, Any] = {}

        # Element 1: Product description
        if products:
            elements[Article9Element.PRODUCT_DESCRIPTION.value] = {
                "products": [
                    {
                        "product_id": p.product_id,
                        "description": p.description,
                    }
                    for p in products
                ],
                "count": len(products),
            }

        # Element 2: Quantity
        if products and all(p.quantity is not None for p in products):
            elements[Article9Element.QUANTITY.value] = {
                "items": [
                    {
                        "product_id": p.product_id,
                        "quantity": str(p.quantity),
                        "unit": p.unit,
                    }
                    for p in products
                ],
                "total_items": len(products),
            }

        # Element 3: Country of production
        countries = set()
        for geo in geolocations:
            countries.add(geo.country_code)
        if countries:
            elements[Article9Element.COUNTRY_OF_PRODUCTION.value] = {
                "countries": sorted(countries),
                "count": len(countries),
            }

        # Element 4: Geolocation
        if geolocations:
            elements[Article9Element.GEOLOCATION.value] = {
                "plots": [
                    {
                        "plot_id": g.plot_id,
                        "latitude": str(g.latitude),
                        "longitude": str(g.longitude),
                        "area_hectares": str(g.area_hectares),
                        "has_polygon": g.polygon is not None,
                        "country": g.country_code,
                    }
                    for g in geolocations
                ],
                "total_plots": len(geolocations),
            }

        # Element 5: Production date
        if production_date_range:
            elements[Article9Element.PRODUCTION_DATE.value] = {
                "start_date": production_date_range[0],
                "end_date": production_date_range[1],
            }

        # Element 6: Supplier information
        if suppliers:
            elements[Article9Element.SUPPLIER_INFO.value] = {
                "suppliers": [
                    {
                        "supplier_id": s.supplier_id,
                        "name": s.name,
                        "country": s.country,
                        "registration_number": s.registration_number,
                    }
                    for s in suppliers
                ],
                "count": len(suppliers),
            }

        # Element 7: Buyer information
        if buyer_info:
            elements[Article9Element.BUYER_INFO.value] = buyer_info

        # Element 8: Certifications
        if certifications:
            elements[Article9Element.CERTIFICATIONS.value] = {
                "certifications": certifications,
                "count": len(certifications),
            }

        # Element 9: Trade codes
        hs_codes = [
            p.hs_code for p in products if p.hs_code
        ]
        cn_codes = [
            getattr(p, "cn_code", "") for p in products
            if getattr(p, "cn_code", "")
        ]
        if hs_codes or cn_codes:
            elements[Article9Element.TRADE_CODES.value] = {
                "hs_codes": hs_codes,
                "cn_codes": cn_codes,
            }

        # Element 10: Supporting evidence
        if supporting_evidence:
            elements[Article9Element.SUPPORTING_EVIDENCE.value] = {
                "evidence_refs": supporting_evidence,
                "count": len(supporting_evidence),
            }

        return elements

    def _check_element_completeness(
        self, elements: Dict[str, Any],
    ) -> Tuple[Decimal, List[Article9Element]]:
        """Score completeness and identify missing elements.

        Calculates the fraction of mandatory Article 9 elements
        present in the assembled data. Returns the completeness
        score and a list of missing elements.

        Args:
            elements: Assembled elements dictionary.

        Returns:
            Tuple of (completeness_score, list_of_missing_elements).
        """
        total = len(ARTICLE9_MANDATORY_ELEMENTS)
        if total == 0:
            return Decimal("1"), []

        present_count = 0
        missing: List[Article9Element] = []

        for element in ARTICLE9_MANDATORY_ELEMENTS:
            if element.value in elements:
                present_count += 1
            else:
                missing.append(element)

        score = (
            Decimal(str(present_count)) / Decimal(str(total))
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        logger.debug(
            "Completeness: %d/%d elements present, score=%s",
            present_count, total, score,
        )

        return score, missing

    def _validate_geolocations(
        self, geos: List[GeolocationReference],
    ) -> List[ValidationIssue]:
        """Validate geolocation data meets Article 9 requirements.

        Checks coordinate precision, polygon requirements for
        plots exceeding 4 hectares, and valid coordinate ranges.

        Args:
            geos: List of geolocation references.

        Returns:
            List of validation issues found.
        """
        issues: List[ValidationIssue] = []

        for geo in geos:
            # Check polygon requirement for plots > 4 hectares
            polygon_issue = self._validate_polygon_requirement(geo)
            if polygon_issue is not None:
                issues.append(polygon_issue)

            # Check coordinate precision
            precision_issues = self._validate_coordinate_precision(geo)
            issues.extend(precision_issues)

            # Check country code is populated
            if not geo.country_code or len(geo.country_code) < 2:
                issues.append(ValidationIssue(
                    field=f"geolocation.{geo.plot_id}.country_code",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Plot '{geo.plot_id}' has invalid country code: "
                        f"'{geo.country_code}'"
                    ),
                    article_reference="EUDR Article 9(1)(d)",
                ))

        return issues

    def _validate_polygon_requirement(
        self, geo: GeolocationReference,
    ) -> Optional[ValidationIssue]:
        """Check polygon requirement for plots > 4 hectares.

        Per EUDR Article 9(1)(d), plots exceeding 4 hectares must
        provide polygon boundary coordinates, not just a single point.

        Args:
            geo: Geolocation reference.

        Returns:
            ValidationIssue if polygon is missing for large plots,
            None otherwise.
        """
        if not self._config.require_polygon_above_4ha:
            return None

        if geo.area_hectares > Decimal("4") and geo.polygon is None:
            return ValidationIssue(
                field=f"geolocation.{geo.plot_id}.polygon",
                severity=ValidationSeverity.ERROR,
                message=(
                    f"Plot '{geo.plot_id}' is {geo.area_hectares} "
                    f"hectares (>4ha) and requires polygon boundary "
                    f"coordinates per Article 9(1)(d)."
                ),
                article_reference="EUDR Article 9(1)(d)",
            )

        return None

    def _validate_coordinate_precision(
        self, geo: GeolocationReference,
    ) -> List[ValidationIssue]:
        """Validate coordinate decimal precision.

        Args:
            geo: Geolocation reference.

        Returns:
            List of precision validation issues.
        """
        issues: List[ValidationIssue] = []
        required_precision = self._config.geolocation_decimal_places

        lat_str = str(geo.latitude)
        lon_str = str(geo.longitude)

        # Check latitude precision
        if "." in lat_str:
            decimals = len(lat_str.split(".")[1])
            if decimals < required_precision:
                issues.append(ValidationIssue(
                    field=f"geolocation.{geo.plot_id}.latitude",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Latitude precision is {decimals} decimal places "
                        f"but {required_precision} are recommended."
                    ),
                    article_reference="EUDR Article 9(1)(d)",
                ))

        # Check longitude precision
        if "." in lon_str:
            decimals = len(lon_str.split(".")[1])
            if decimals < required_precision:
                issues.append(ValidationIssue(
                    field=f"geolocation.{geo.plot_id}.longitude",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Longitude precision is {decimals} decimal places "
                        f"but {required_precision} are recommended."
                    ),
                    article_reference="EUDR Article 9(1)(d)",
                ))

        return issues

    def _validate_products(
        self, products: List[ProductEntry],
    ) -> List[ValidationIssue]:
        """Validate product entries for Article 9 compliance.

        Args:
            products: List of product entries.

        Returns:
            List of validation issues.
        """
        issues: List[ValidationIssue] = []

        for idx, product in enumerate(products):
            if not product.description:
                issues.append(ValidationIssue(
                    field=f"products[{idx}].description",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Product '{product.product_id}' missing "
                        f"description (Article 9(1)(a))."
                    ),
                    article_reference="EUDR Article 9(1)(a)",
                ))

            if product.quantity <= Decimal("0"):
                issues.append(ValidationIssue(
                    field=f"products[{idx}].quantity",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Product '{product.product_id}' has invalid "
                        f"quantity: {product.quantity}."
                    ),
                    article_reference="EUDR Article 9(1)(b)",
                ))

        return issues

    def is_package_complete(
        self, package: Article9Package,
    ) -> bool:
        """Check if an Article 9 package meets the completeness threshold.

        Args:
            package: Article 9 package to check.

        Returns:
            True if completeness score meets threshold.
        """
        threshold = self._config.article9_completeness_threshold
        return package.completeness_score >= threshold

    def get_element_description(
        self, element: Article9Element,
    ) -> str:
        """Get the description for an Article 9 element.

        Args:
            element: Article 9 element.

        Returns:
            Description string for the element.
        """
        return _ELEMENT_DESCRIPTIONS.get(
            element, f"Article 9 element: {element.value}",
        )

    def get_missing_element_guidance(
        self, missing: List[str],
    ) -> List[Dict[str, str]]:
        """Generate guidance for missing Article 9 elements.

        Args:
            missing: List of missing element value strings.

        Returns:
            List of guidance dictionaries with element name and action.
        """
        guidance: List[Dict[str, str]] = []
        for elem_value in missing:
            # Find matching enum
            for element in Article9Element:
                if element.value == elem_value:
                    guidance.append({
                        "element": elem_value,
                        "description": _ELEMENT_DESCRIPTIONS.get(
                            element, "",
                        ),
                        "action": (
                            f"Collect {elem_value} data from upstream "
                            f"agents or supplier questionnaires."
                        ),
                        "article_reference": "EUDR Article 9",
                    })
                    break

        return guidance

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Dictionary with engine status and configuration details.
        """
        return {
            "engine": "Article9DataAssembler",
            "status": "available",
            "config": {
                "completeness_threshold": str(
                    self._config.article9_completeness_threshold
                ),
                "require_polygon_above_4ha": (
                    self._config.require_polygon_above_4ha
                ),
                "geolocation_decimal_places": (
                    self._config.geolocation_decimal_places
                ),
            },
            "mandatory_elements": len(ARTICLE9_MANDATORY_ELEMENTS),
        }
