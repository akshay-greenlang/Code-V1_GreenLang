# -*- coding: utf-8 -*-
"""
ReportingPackagerAgent_AI - Package CBAM Report for EU Transitional Registry

This agent is the final step in the CBAM pipeline:
1. Aggregates emissions across all shipments
2. Generates report metadata and importer declarations
3. Creates multi-dimensional summaries (by product, country, etc.)
4. Performs final CBAM compliance validations
5. Checks complex goods 20% threshold
6. Generates complete EU CBAM Transitional Registry report
7. Creates human-readable summary for management

Output: Submission-ready JSON conforming to registry_output.schema.json

Performance target: <1s for 10,000 shipments
Completeness target: 100% (all required sections)

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import hashlib
import json
import logging
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from xml.dom import minidom

import yaml
from pydantic import BaseModel, Field
from greenlang.determinism import DeterministicClock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ReportMetadata(BaseModel):
    """Report header and metadata."""
    report_id: str
    quarter: str
    reporting_period_start: str
    reporting_period_end: str
    generated_at: str
    generated_by: str = "GreenLang CBAM Importer Copilot v1.1.0"
    version: str = "1.1.0"


class ImporterDeclaration(BaseModel):
    """EU importer declaration information."""
    importer_name: str
    importer_country: str
    importer_eori: str
    declaration_date: str
    declarant_name: str
    declarant_position: str


class ValidationResult(BaseModel):
    """Validation check result."""
    rule_id: str
    rule_name: str
    status: str  # "pass", "fail", "warning"
    message: str
    affected_shipments: List[str] = []


class CertificateObligation(BaseModel):
    """v1.1: Certificate obligation summary for a reporting period."""
    total_embedded_emissions_tco2: float = 0.0
    carbon_price_deduction_tco2: float = 0.0
    free_allocation_deduction_tco2: float = 0.0
    net_emissions_tco2: float = 0.0
    ets_price_eur_per_tco2: float = 0.0
    gross_obligation_eur: float = 0.0
    deductions_eur: float = 0.0
    net_obligation_eur: float = 0.0
    certificates_required: int = 0
    quarterly_holding_pct: float = 0.50


class AmendmentMetadata(BaseModel):
    """v1.1: Tracks amendments to a previously submitted report."""
    is_amendment: bool = False
    original_report_id: Optional[str] = None
    amendment_number: int = 0
    amendment_reason: Optional[str] = None
    amendment_date: Optional[str] = None
    fields_amended: List[str] = Field(default_factory=list)


# ============================================================================
# REPORTING PACKAGER AGENT
# ============================================================================

class ReportingPackagerAgent:
    """
    Generate EU CBAM Transitional Registry report.

    This agent aggregates all calculated emissions and produces
    a submission-ready report with:
    - Complete metadata and declarations
    - Detailed goods listing
    - Multi-dimensional summaries
    - Final compliance validations
    - Provenance and audit trail

    All aggregations are deterministic (no LLM).
    """

    def __init__(
        self,
        cbam_rules_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the ReportingPackagerAgent.

        Args:
            cbam_rules_path: Path to CBAM rules YAML (for validation)
        """
        self.cbam_rules_path = Path(cbam_rules_path) if cbam_rules_path else None
        self.cbam_rules = self._load_cbam_rules() if self.cbam_rules_path else {}

        logger.info("ReportingPackagerAgent initialized")

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    def _load_cbam_rules(self) -> Dict[str, Any]:
        """Load CBAM rules for validation."""
        if not self.cbam_rules_path or not self.cbam_rules_path.exists():
            return {}

        try:
            with open(self.cbam_rules_path, 'r', encoding='utf-8') as f:
                rules = yaml.safe_load(f)
            logger.info("Loaded CBAM rules for validation")
            return rules
        except Exception as e:
            logger.warning(f"Failed to load CBAM rules: {e}")
            return {}

    # ========================================================================
    # QUARTER HANDLING
    # ========================================================================

    def _parse_quarter(self, quarter: str) -> Tuple[int, int]:
        """Parse quarter string (e.g., '2025Q4') into (year, quarter_num)."""
        year = int(quarter[:4])
        q = int(quarter[5])
        return year, q

    def _get_quarter_dates(self, quarter: str) -> Tuple[str, str]:
        """Get start and end dates for a quarter."""
        year, q = self._parse_quarter(quarter)

        quarter_dates = {
            1: ((year, 1, 1), (year, 3, 31)),
            2: ((year, 4, 1), (year, 6, 30)),
            3: ((year, 7, 1), (year, 9, 30)),
            4: ((year, 10, 1), (year, 12, 31))
        }

        start_tuple, end_tuple = quarter_dates[q]
        start_date = datetime(*start_tuple).strftime("%Y-%m-%d")
        end_date = datetime(*end_tuple).strftime("%Y-%m-%d")

        return start_date, end_date

    # ========================================================================
    # AGGREGATION (100% DETERMINISTIC)
    # ========================================================================

    def _aggregate_goods_summary(self, shipments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate goods summary statistics.

        All aggregations use Python built-ins (no LLM).
        """
        total_shipments = len(shipments)
        total_mass_kg = sum(s.get("net_mass_kg", 0) for s in shipments)
        total_mass_tonnes = round(total_mass_kg / 1000, 3)

        # By product group
        by_product_group = defaultdict(lambda: {
            "shipment_count": 0,
            "total_mass_kg": 0,
            "cn_codes": set()
        })

        for shipment in shipments:
            pg = shipment.get("product_group", "unknown")
            by_product_group[pg]["shipment_count"] += 1
            by_product_group[pg]["total_mass_kg"] += shipment.get("net_mass_kg", 0)
            by_product_group[pg]["cn_codes"].add(shipment.get("cn_code"))

        product_groups = []
        for pg, data in sorted(by_product_group.items()):
            product_groups.append({
                "product_group": pg,
                "shipment_count": data["shipment_count"],
                "total_mass_tonnes": round(data["total_mass_kg"] / 1000, 3),
                "cn_codes": sorted(list(data["cn_codes"]))
            })

        # By origin country
        by_origin = defaultdict(lambda: {
            "shipment_count": 0,
            "total_mass_kg": 0
        })

        for shipment in shipments:
            origin = shipment.get("origin_iso", "unknown")
            origin_name = shipment.get("origin_country", origin)
            by_origin[(origin, origin_name)]["shipment_count"] += 1
            by_origin[(origin, origin_name)]["total_mass_kg"] += shipment.get("net_mass_kg", 0)

        origin_countries = []
        for (origin_iso, origin_name), data in sorted(
            by_origin.items(),
            key=lambda x: -x[1]["total_mass_kg"]
        ):
            origin_countries.append({
                "country_iso": origin_iso,
                "country_name": origin_name,
                "shipment_count": data["shipment_count"],
                "total_mass_tonnes": round(data["total_mass_kg"] / 1000, 3)
            })

        return {
            "total_shipments": total_shipments,
            "total_mass_tonnes": total_mass_tonnes,
            "product_groups": product_groups,
            "origin_countries": origin_countries
        }

    def _aggregate_emissions_summary(self, shipments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate emissions summary statistics.

        All aggregations use Python arithmetic (no LLM).
        """
        total_direct = 0.0
        total_indirect = 0.0
        total_embedded = 0.0

        calc_methods = defaultdict(int)
        by_product_group = defaultdict(float)
        by_origin = defaultdict(float)

        for shipment in shipments:
            calc = shipment.get("emissions_calculation")
            if not calc:
                continue

            # Aggregate totals
            total_direct += calc.get("direct_emissions_tco2", 0)
            total_indirect += calc.get("indirect_emissions_tco2", 0)
            total_embedded += calc.get("total_emissions_tco2", 0)

            # Count calculation methods
            method = calc.get("calculation_method", "unknown")
            calc_methods[method] += 1

            # By product group
            pg = shipment.get("product_group", "unknown")
            by_product_group[pg] += calc.get("total_emissions_tco2", 0)

            # By origin
            origin = shipment.get("origin_iso", "unknown")
            by_origin[origin] += calc.get("total_emissions_tco2", 0)

        # Round totals
        total_direct = round(total_direct, 2)
        total_indirect = round(total_indirect, 2)
        total_embedded = round(total_embedded, 2)

        # Calculate average intensity
        total_mass = sum(s.get("net_mass_kg", 0) for s in shipments) / 1000
        emissions_intensity = round(total_embedded / total_mass, 3) if total_mass > 0 else 0.0

        # By product group with percentages
        emissions_by_product = []
        for pg, emissions in sorted(by_product_group.items(), key=lambda x: -x[1]):
            pct = (emissions / total_embedded * 100) if total_embedded > 0 else 0
            emissions_by_product.append({
                "product_group": pg,
                "total_emissions_tco2": round(emissions, 2),
                "percentage_of_total": round(pct, 1)
            })

        # By origin with percentages
        emissions_by_origin = []
        for origin, emissions in sorted(by_origin.items(), key=lambda x: -x[1]):
            # Get country name
            origin_name = next(
                (s.get("origin_country", origin)
                 for s in shipments if s.get("origin_iso") == origin),
                origin
            )
            pct = (emissions / total_embedded * 100) if total_embedded > 0 else 0
            emissions_by_origin.append({
                "country_iso": origin,
                "country_name": origin_name,
                "total_emissions_tco2": round(emissions, 2),
                "percentage_of_total": round(pct, 1)
            })

        return {
            "total_direct_emissions_tco2": total_direct,
            "total_indirect_emissions_tco2": total_indirect,
            "total_embedded_emissions_tco2": total_embedded,
            "emissions_by_product_group": emissions_by_product,
            "emissions_by_origin_country": emissions_by_origin[:10],  # Top 10
            "emissions_intensity_tco2_per_tonne": emissions_intensity,
            "calculation_methods_used": {
                "default_values_count": calc_methods.get("default_values", 0),
                "actual_data_count": calc_methods.get("actual_data", 0),
                "complex_goods_count": calc_methods.get("complex_goods", 0),
                "estimation_count": calc_methods.get("estimation", 0)
            }
        }

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _validate_report(
        self,
        goods_summary: Dict[str, Any],
        emissions_summary: Dict[str, Any],
        shipments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform final CBAM compliance validations.

        Returns validation results with pass/fail status.
        """
        validation_results = []
        errors = []
        warnings = []

        # VAL-041: Summary totals match details
        calculated_mass = sum(s.get("net_mass_kg", 0) for s in shipments) / 1000
        reported_mass = goods_summary["total_mass_tonnes"]

        if abs(calculated_mass - reported_mass) > 0.001:
            validation_results.append({
                "rule_id": "VAL-041",
                "rule_name": "Summary Totals Match Details",
                "status": "fail",
                "message": f"Mass mismatch: summary={reported_mass}, calculated={calculated_mass:.3f}"
            })
            errors.append({
                "error_code": "E005",
                "severity": "error",
                "message": "Summary mass does not match sum of shipments"
            })
        else:
            validation_results.append({
                "rule_id": "VAL-041",
                "rule_name": "Summary Totals Match Details",
                "status": "pass",
                "message": "Mass totals match"
            })

        # VAL-042: Emissions totals match
        calculated_emissions = sum(
            s.get("emissions_calculation", {}).get("total_emissions_tco2", 0)
            for s in shipments
        )
        reported_emissions = emissions_summary["total_embedded_emissions_tco2"]

        if abs(calculated_emissions - reported_emissions) > 0.01:
            validation_results.append({
                "rule_id": "VAL-042",
                "rule_name": "Emissions Summary Matches Details",
                "status": "fail",
                "message": f"Emissions mismatch: summary={reported_emissions}, calculated={calculated_emissions:.2f}"
            })
            errors.append({
                "error_code": "E005",
                "severity": "error",
                "message": "Summary emissions do not match sum of shipments"
            })
        else:
            validation_results.append({
                "rule_id": "VAL-042",
                "rule_name": "Emissions Summary Matches Details",
                "status": "pass",
                "message": "Emissions totals match"
            })

        # VAL-020: Complex goods 20% check
        complex_goods_count = emissions_summary["calculation_methods_used"].get("complex_goods_count", 0)
        total_shipments = len(shipments)
        complex_goods_pct = (complex_goods_count / total_shipments * 100) if total_shipments > 0 else 0

        complex_goods_check = {
            "complex_goods_count": complex_goods_count,
            "total_shipments_count": total_shipments,
            "complex_goods_percentage": round(complex_goods_pct, 1),
            "threshold_percentage": 20.0,
            "within_threshold": complex_goods_pct <= 20.0
        }

        if complex_goods_pct > 20.0:
            validation_results.append({
                "rule_id": "VAL-020",
                "rule_name": "Complex Goods 20% Cap",
                "status": "fail",
                "message": f"Complex goods exceed 20% threshold: {complex_goods_pct:.1f}%"
            })
            errors.append({
                "error_code": "E006",
                "severity": "error",
                "message": f"Complex goods {complex_goods_pct:.1f}% exceeds 20% cap"
            })
        else:
            validation_results.append({
                "rule_id": "VAL-020",
                "rule_name": "Complex Goods 20% Cap",
                "status": "pass",
                "message": f"Complex goods within threshold: {complex_goods_pct:.1f}%"
            })

        # Determine overall validity
        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "validation_timestamp": DeterministicClock.now().isoformat(),
            "rules_checked": validation_results,
            "errors": errors,
            "warnings": warnings,
            "complex_goods_check": complex_goods_check
        }

    # ========================================================================
    # v1.1: QUARTERLY REPORT GENERATION
    # ========================================================================

    def generate_quarterly_report(
        self,
        period: str,
        aggregations: Dict[str, Any],
        importer_info: Dict[str, str],
        certificate_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate EU CBAM Registry format quarterly report.

        This produces a structured quarterly report that includes goods
        summaries, emissions summaries, certificate obligations, and
        validation results conforming to the Implementing Regulation format.

        Args:
            period: Reporting quarter (e.g., '2026Q1')
            aggregations: Pre-calculated aggregation data
            importer_info: Importer declaration information
            certificate_config: Certificate engine configuration (optional)

        Returns:
            Quarterly report dictionary ready for submission
        """
        start_time = DeterministicClock.now()
        period_start, period_end = self._get_quarter_dates(period)
        year, q = self._parse_quarter(period)

        timestamp = start_time.strftime("%Y%m%d%H%M%S")
        report_id = (
            f"CBAM-QR-{period}-"
            f"{importer_info.get('importer_eori', 'UNKNOWN')[:15]}-"
            f"{timestamp}"
        )

        # Build quarterly report structure
        quarterly_report = {
            "report_metadata": {
                "report_id": report_id,
                "report_type": "quarterly",
                "quarter": period,
                "year": year,
                "quarter_number": q,
                "reporting_period_start": period_start,
                "reporting_period_end": period_end,
                "generated_at": start_time.isoformat(),
                "generated_by": "GreenLang CBAM Importer Copilot v1.1.0",
                "version": "1.1.0",
                "format": "EU CBAM Registry Quarterly Report"
            },
            "importer_declaration": {
                "importer_name": importer_info.get("importer_name"),
                "importer_country": importer_info.get("importer_country"),
                "importer_eori": importer_info.get("importer_eori"),
                "declaration_date": start_time.strftime("%Y-%m-%d"),
                "declarant_name": importer_info.get("declarant_name"),
                "declarant_position": importer_info.get("declarant_position")
            },
            "goods_summary": aggregations.get("goods_summary", {}),
            "emissions_summary": aggregations.get("emissions_summary", {}),
            "certificate_obligation": self._calculate_certificate_obligation(
                aggregations.get("emissions_summary", {}),
                certificate_config or {}
            ),
            "submission_metadata": {
                "submission_deadline": self._get_submission_deadline(period),
                "amendment_window_end": self._get_amendment_deadline(period),
                "submission_status": "draft"
            }
        }

        end_time = DeterministicClock.now()
        processing_time = (end_time - start_time).total_seconds()
        quarterly_report["report_metadata"]["processing_time_seconds"] = round(
            processing_time, 3
        )

        logger.info(
            f"Generated quarterly report {report_id} for {period} "
            f"in {processing_time:.3f}s"
        )
        return quarterly_report

    def _calculate_certificate_obligation(
        self,
        emissions_summary: Dict[str, Any],
        certificate_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate CBAM certificate obligation based on emissions.

        All calculations are DETERMINISTIC (Python arithmetic only).

        Args:
            emissions_summary: Aggregated emissions data
            certificate_config: Certificate engine configuration

        Returns:
            Certificate obligation summary dictionary
        """
        total_emissions = emissions_summary.get("total_embedded_emissions_tco2", 0.0)
        ets_price = certificate_config.get("ets_price_eur_per_tco2", 0.0)
        quarterly_holding = certificate_config.get("quarterly_holding_pct", 0.50)
        free_allocation_pct = certificate_config.get("free_allocation_pct", 0.0)
        carbon_price_deduction_tco2 = certificate_config.get(
            "carbon_price_deduction_tco2", 0.0
        )

        # Deterministic calculations
        free_allocation_deduction = round(total_emissions * free_allocation_pct, 3)
        net_emissions = round(
            total_emissions - carbon_price_deduction_tco2 - free_allocation_deduction,
            3
        )
        net_emissions = max(net_emissions, 0.0)

        gross_obligation = round(total_emissions * ets_price, 2)
        deductions = round(
            (carbon_price_deduction_tco2 + free_allocation_deduction) * ets_price, 2
        )
        net_obligation = round(max(gross_obligation - deductions, 0.0), 2)

        # Certificates required (1 certificate = 1 tCO2)
        import math
        certificates_required = math.ceil(net_emissions)

        return CertificateObligation(
            total_embedded_emissions_tco2=total_emissions,
            carbon_price_deduction_tco2=carbon_price_deduction_tco2,
            free_allocation_deduction_tco2=free_allocation_deduction,
            net_emissions_tco2=net_emissions,
            ets_price_eur_per_tco2=ets_price,
            gross_obligation_eur=gross_obligation,
            deductions_eur=deductions,
            net_obligation_eur=net_obligation,
            certificates_required=certificates_required,
            quarterly_holding_pct=quarterly_holding
        ).dict()

    def _get_submission_deadline(self, quarter: str) -> str:
        """Get the submission deadline for a given quarter (30 days after end)."""
        year, q = self._parse_quarter(quarter)
        deadlines = {
            1: f"{year}-04-30",
            2: f"{year}-07-31",
            3: f"{year}-10-31",
            4: f"{year + 1}-01-31"
        }
        return deadlines.get(q, "")

    def _get_amendment_deadline(self, quarter: str) -> str:
        """Get the amendment window deadline (60 days after submission deadline)."""
        year, q = self._parse_quarter(quarter)
        # Amendment window = 60 days after submission deadline
        amendment_deadlines = {
            1: f"{year}-06-29",
            2: f"{year}-09-29",
            3: f"{year}-12-30",
            4: f"{year + 1}-04-01"
        }
        return amendment_deadlines.get(q, "")

    # ========================================================================
    # v1.1: XML OUTPUT GENERATION
    # ========================================================================

    def generate_xml_output(self, report_data: Dict[str, Any]) -> str:
        """
        Generate CBAM Registry XML format per Implementing Regulation.

        Produces a structured XML document with:
        - Report header (period, importer EORI, submission date)
        - Goods section (CN codes, quantities, origins)
        - Emissions section (direct, indirect, calculation method)
        - Carbon price section (deductions claimed)

        Args:
            report_data: Complete report dictionary

        Returns:
            Formatted XML string
        """
        # Root element
        root = ET.Element("CBAMReport")
        root.set("xmlns", "urn:eu:cbam:registry:report:v1.1")
        root.set("version", "1.1.0")

        # Report Header
        header = ET.SubElement(root, "ReportHeader")
        meta = report_data.get("report_metadata", {})

        self._xml_add_text(header, "ReportID", meta.get("report_id", ""))
        self._xml_add_text(header, "Quarter", meta.get("quarter", ""))
        self._xml_add_text(
            header, "ReportingPeriodStart",
            meta.get("reporting_period_start", "")
        )
        self._xml_add_text(
            header, "ReportingPeriodEnd",
            meta.get("reporting_period_end", "")
        )
        self._xml_add_text(
            header, "SubmissionDate",
            meta.get("generated_at", "")[:10]
        )
        self._xml_add_text(header, "Version", meta.get("version", "1.1.0"))

        # Importer Declaration
        importer_elem = ET.SubElement(root, "ImporterDeclaration")
        decl = report_data.get("importer_declaration", {})

        self._xml_add_text(importer_elem, "ImporterName", decl.get("importer_name", ""))
        self._xml_add_text(importer_elem, "ImporterEORI", decl.get("importer_eori", ""))
        self._xml_add_text(
            importer_elem, "ImporterCountry",
            decl.get("importer_country", "")
        )
        self._xml_add_text(
            importer_elem, "DeclarantName",
            decl.get("declarant_name", "")
        )
        self._xml_add_text(
            importer_elem, "DeclarantPosition",
            decl.get("declarant_position", "")
        )
        self._xml_add_text(
            importer_elem, "DeclarationDate",
            decl.get("declaration_date", "")
        )

        # Goods Section
        goods_section = ET.SubElement(root, "GoodsImported")
        goods_summary = report_data.get("goods_summary", {})

        self._xml_add_text(
            goods_section, "TotalShipments",
            str(goods_summary.get("total_shipments", 0))
        )
        self._xml_add_text(
            goods_section, "TotalMassTonnes",
            f"{goods_summary.get('total_mass_tonnes', 0):.3f}"
        )

        # Individual goods entries from detailed_goods
        for shipment in report_data.get("detailed_goods", []):
            good_elem = ET.SubElement(goods_section, "Good")
            self._xml_add_text(
                good_elem, "ShipmentID",
                str(shipment.get("shipment_id", ""))
            )
            self._xml_add_text(
                good_elem, "CNCode",
                str(shipment.get("cn_code", ""))
            )
            self._xml_add_text(
                good_elem, "ProductGroup",
                str(shipment.get("product_group", ""))
            )
            self._xml_add_text(
                good_elem, "OriginCountry",
                str(shipment.get("origin_iso", ""))
            )
            self._xml_add_text(
                good_elem, "NetMassKg",
                f"{shipment.get('net_mass_kg', 0):.2f}"
            )
            self._xml_add_text(
                good_elem, "ImportDate",
                str(shipment.get("import_date", ""))
            )

        # Emissions Section
        emissions_section = ET.SubElement(root, "EmbeddedEmissions")
        em_summary = report_data.get("emissions_summary", {})

        self._xml_add_text(
            emissions_section, "TotalDirectEmissionsTCO2",
            f"{em_summary.get('total_direct_emissions_tco2', 0):.2f}"
        )
        self._xml_add_text(
            emissions_section, "TotalIndirectEmissionsTCO2",
            f"{em_summary.get('total_indirect_emissions_tco2', 0):.2f}"
        )
        self._xml_add_text(
            emissions_section, "TotalEmbeddedEmissionsTCO2",
            f"{em_summary.get('total_embedded_emissions_tco2', 0):.2f}"
        )
        self._xml_add_text(
            emissions_section, "EmissionsIntensityTCO2PerTonne",
            f"{em_summary.get('emissions_intensity_tco2_per_tonne', 0):.3f}"
        )

        methods_elem = ET.SubElement(emissions_section, "CalculationMethods")
        methods = em_summary.get("calculation_methods_used", {})
        self._xml_add_text(
            methods_elem, "DefaultValuesCount",
            str(methods.get("default_values_count", 0))
        )
        self._xml_add_text(
            methods_elem, "ActualDataCount",
            str(methods.get("actual_data_count", 0))
        )
        self._xml_add_text(
            methods_elem, "RegionalFactorCount",
            str(methods.get("regional_factor_count", 0))
        )

        # Per-shipment emissions detail
        for shipment in report_data.get("detailed_goods", []):
            calc = shipment.get("emissions_calculation")
            if not calc:
                continue

            emission_elem = ET.SubElement(emissions_section, "ShipmentEmissions")
            self._xml_add_text(
                emission_elem, "ShipmentID",
                str(shipment.get("shipment_id", ""))
            )
            self._xml_add_text(
                emission_elem, "CalculationMethod",
                str(calc.get("calculation_method", ""))
            )
            self._xml_add_text(
                emission_elem, "DataSource",
                str(calc.get("data_source", ""))
            )
            self._xml_add_text(
                emission_elem, "DirectEmissionsTCO2",
                f"{calc.get('direct_emissions_tco2', 0):.3f}"
            )
            self._xml_add_text(
                emission_elem, "IndirectEmissionsTCO2",
                f"{calc.get('indirect_emissions_tco2', 0):.3f}"
            )
            self._xml_add_text(
                emission_elem, "TotalEmissionsTCO2",
                f"{calc.get('total_emissions_tco2', 0):.3f}"
            )

        # Carbon Price Section (v1.1)
        cert_data = report_data.get("certificate_obligation", {})
        if cert_data:
            carbon_section = ET.SubElement(root, "CarbonPriceSection")
            self._xml_add_text(
                carbon_section, "ETSPriceEURPerTCO2",
                f"{cert_data.get('ets_price_eur_per_tco2', 0):.2f}"
            )
            self._xml_add_text(
                carbon_section, "CarbonPriceDeductionTCO2",
                f"{cert_data.get('carbon_price_deduction_tco2', 0):.3f}"
            )
            self._xml_add_text(
                carbon_section, "FreeAllocationDeductionTCO2",
                f"{cert_data.get('free_allocation_deduction_tco2', 0):.3f}"
            )
            self._xml_add_text(
                carbon_section, "NetEmissionsTCO2",
                f"{cert_data.get('net_emissions_tco2', 0):.3f}"
            )
            self._xml_add_text(
                carbon_section, "GrossObligationEUR",
                f"{cert_data.get('gross_obligation_eur', 0):.2f}"
            )
            self._xml_add_text(
                carbon_section, "DeductionsEUR",
                f"{cert_data.get('deductions_eur', 0):.2f}"
            )
            self._xml_add_text(
                carbon_section, "NetObligationEUR",
                f"{cert_data.get('net_obligation_eur', 0):.2f}"
            )
            self._xml_add_text(
                carbon_section, "CertificatesRequired",
                str(cert_data.get("certificates_required", 0))
            )

        # Amendment Metadata (v1.1)
        amendment_data = report_data.get("amendment_metadata", {})
        if amendment_data and amendment_data.get("is_amendment", False):
            amend_section = ET.SubElement(root, "AmendmentMetadata")
            self._xml_add_text(
                amend_section, "OriginalReportID",
                str(amendment_data.get("original_report_id", ""))
            )
            self._xml_add_text(
                amend_section, "AmendmentNumber",
                str(amendment_data.get("amendment_number", 0))
            )
            self._xml_add_text(
                amend_section, "AmendmentReason",
                str(amendment_data.get("amendment_reason", ""))
            )
            self._xml_add_text(
                amend_section, "AmendmentDate",
                str(amendment_data.get("amendment_date", ""))
            )
            for field in amendment_data.get("fields_amended", []):
                self._xml_add_text(amend_section, "FieldAmended", str(field))

        # Format XML with indentation
        xml_str = ET.tostring(root, encoding='unicode', xml_declaration=False)
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        formatted = minidom.parseString(xml_str).toprettyxml(indent="  ")
        # Remove minidom's own xml declaration (it adds one) and use ours
        lines = formatted.split('\n')
        if lines[0].startswith('<?xml'):
            lines[0] = xml_declaration.strip()
        result = '\n'.join(lines)

        logger.info(
            f"Generated XML output ({len(result)} bytes) for report "
            f"{meta.get('report_id', 'UNKNOWN')}"
        )
        return result

    def _xml_add_text(
        self,
        parent: ET.Element,
        tag: str,
        text: str
    ) -> ET.Element:
        """Add a text sub-element to an XML parent element."""
        elem = ET.SubElement(parent, tag)
        elem.text = text
        return elem

    def write_xml_output(
        self,
        report_data: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> None:
        """
        Write CBAM report as XML file.

        Args:
            report_data: Complete report dictionary
            output_path: Path for output XML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        xml_content = self.generate_xml_output(report_data)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)

        logger.info(f"Wrote XML report to {output_path}")

    # ========================================================================
    # v1.1: AMENDMENT TRACKING
    # ========================================================================

    def create_amendment(
        self,
        original_report: Dict[str, Any],
        amended_data: Dict[str, Any],
        reason: str
    ) -> Dict[str, Any]:
        """
        Create an amendment to a previously submitted report.

        Tracks all changes with full audit trail including the original
        report reference, amendment number, reason, and changed fields.

        Args:
            original_report: The original report to amend
            amended_data: Dictionary of fields to amend
            reason: Reason for the amendment

        Returns:
            New report with amendment metadata
        """
        original_id = original_report.get("report_metadata", {}).get("report_id", "")
        existing_amendment = original_report.get("amendment_metadata", {})
        amendment_number = existing_amendment.get("amendment_number", 0) + 1

        # Deep copy original report
        import copy
        amended_report = copy.deepcopy(original_report)

        # Track which fields are being amended
        fields_amended = list(amended_data.keys())

        # Apply amendments
        for key, value in amended_data.items():
            if key in amended_report:
                amended_report[key] = value

        # Generate new report ID
        timestamp = DeterministicClock.now().strftime("%Y%m%d%H%M%S")
        amended_report["report_metadata"]["report_id"] = (
            f"{original_id}-AMD{amendment_number:02d}-{timestamp}"
        )
        amended_report["report_metadata"]["generated_at"] = (
            DeterministicClock.now().isoformat()
        )

        # Add amendment metadata
        amended_report["amendment_metadata"] = AmendmentMetadata(
            is_amendment=True,
            original_report_id=original_id,
            amendment_number=amendment_number,
            amendment_reason=reason,
            amendment_date=DeterministicClock.now().strftime("%Y-%m-%d"),
            fields_amended=fields_amended
        ).dict()

        logger.info(
            f"Created amendment #{amendment_number} for report {original_id}: "
            f"{reason} (fields: {fields_amended})"
        )
        return amended_report

    # ========================================================================
    # PROVENANCE
    # ========================================================================

    def _generate_provenance(
        self,
        input_files: List[Dict[str, Any]],
        agents_performance: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate provenance and audit trail."""
        return {
            "input_files": input_files,
            "emission_factors_version": "1.0.0-demo (IEA 2018, WSA 2023, IAI 2023, IPCC 2019)",
            "agents_used": agents_performance,
            "processing_log": [
                {
                    "step": "Data ingestion and validation",
                    "timestamp": DeterministicClock.now().isoformat(),
                    "status": "success",
                    "agent": "ShipmentIntakeAgent_AI v1.1.0"
                },
                {
                    "step": "Emissions calculation",
                    "timestamp": DeterministicClock.now().isoformat(),
                    "status": "success",
                    "agent": "EmissionsCalculatorAgent_AI v1.1.0"
                },
                {
                    "step": "Report generation and validation",
                    "timestamp": DeterministicClock.now().isoformat(),
                    "status": "success",
                    "agent": "ReportingPackagerAgent_AI v1.1.0"
                }
            ]
        }

    # ========================================================================
    # MAIN REPORT GENERATION
    # ========================================================================

    def generate_report(
        self,
        shipments_with_emissions: List[Dict[str, Any]],
        importer_info: Dict[str, str],
        quarter: Optional[str] = None,
        input_files: Optional[List[Dict[str, Any]]] = None,
        certificate_config: Optional[Dict[str, Any]] = None,
        amendment_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete CBAM Transitional Registry report.

        v1.1 enhancements:
        - Certificate obligation summary section
        - Amendment tracking metadata
        - XML output support (via generate_xml_output)

        Args:
            shipments_with_emissions: Shipments with calculated emissions
            importer_info: Importer declaration information
            quarter: Reporting quarter (inferred from shipments if not provided)
            input_files: List of input files for provenance
            certificate_config: v1.1 certificate engine configuration (optional)
            amendment_info: v1.1 amendment metadata if this is an amendment (optional)

        Returns:
            Complete CBAM report dictionary
        """
        start_time = DeterministicClock.now()

        # Infer quarter from first shipment if not provided
        if not quarter and shipments_with_emissions:
            quarter = shipments_with_emissions[0].get("quarter", "2025Q4")

        # Generate report ID
        timestamp = start_time.strftime("%Y%m%d%H%M%S")
        report_id = f"CBAM-{quarter}-{importer_info.get('importer_name', 'UNKNOWN')[:10].upper()}-{timestamp}"

        # Get quarter dates
        period_start, period_end = self._get_quarter_dates(quarter)

        # Generate metadata
        report_metadata = {
            "report_id": report_id,
            "quarter": quarter,
            "reporting_period_start": period_start,
            "reporting_period_end": period_end,
            "generated_at": start_time.isoformat(),
            "generated_by": "GreenLang CBAM Importer Copilot v1.1.0",
            "version": "1.1.0"
        }

        # Importer declaration
        importer_declaration = {
            "importer_name": importer_info.get("importer_name"),
            "importer_country": importer_info.get("importer_country"),
            "importer_eori": importer_info.get("importer_eori"),
            "declaration_date": importer_info.get("declaration_date", start_time.strftime("%Y-%m-%d")),
            "declarant_name": importer_info.get("declarant_name"),
            "declarant_position": importer_info.get("declarant_position")
        }

        # Aggregate summaries
        goods_summary = self._aggregate_goods_summary(shipments_with_emissions)
        emissions_summary = self._aggregate_emissions_summary(shipments_with_emissions)

        # Validation
        validation_results = self._validate_report(
            goods_summary,
            emissions_summary,
            shipments_with_emissions
        )

        # Provenance
        end_time = DeterministicClock.now()
        processing_time = (end_time - start_time).total_seconds()

        provenance = self._generate_provenance(
            input_files or [],
            [
                {
                    "agent_name": "ShipmentIntakeAgent_AI",
                    "agent_version": "1.1.0",
                    "execution_time_ms": 0  # Would be from actual pipeline
                },
                {
                    "agent_name": "EmissionsCalculatorAgent_AI",
                    "agent_version": "1.1.0",
                    "execution_time_ms": 0
                },
                {
                    "agent_name": "ReportingPackagerAgent_AI",
                    "agent_version": "1.1.0",
                    "execution_time_ms": round(processing_time * 1000, 2)
                }
            ]
        )

        # v1.1: Certificate obligation summary
        cert_obligation = {}
        if certificate_config:
            cert_obligation = self._calculate_certificate_obligation(
                emissions_summary, certificate_config
            )

        # v1.1: Amendment metadata
        amend_metadata = {}
        if amendment_info:
            amend_metadata = AmendmentMetadata(
                is_amendment=True,
                original_report_id=amendment_info.get("original_report_id"),
                amendment_number=amendment_info.get("amendment_number", 1),
                amendment_reason=amendment_info.get("reason", ""),
                amendment_date=start_time.strftime("%Y-%m-%d"),
                fields_amended=amendment_info.get("fields_amended", [])
            ).dict()

        # Build complete report
        report = {
            "report_metadata": report_metadata,
            "importer_declaration": importer_declaration,
            "goods_summary": goods_summary,
            "detailed_goods": shipments_with_emissions,
            "emissions_summary": emissions_summary,
            "certificate_obligation": cert_obligation,
            "validation_results": validation_results,
            "amendment_metadata": amend_metadata,
            "provenance": provenance
        }

        logger.info(f"Generated CBAM report {report_id} in {processing_time:.3f}s")
        logger.info(f"Total emissions: {emissions_summary['total_embedded_emissions_tco2']:.2f} tCO2")
        logger.info(f"Validation: {'PASS' if validation_results['is_valid'] else 'FAIL'}")

        return report

    def write_report(self, report: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Write report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Wrote CBAM report to {output_path}")

    def generate_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable Markdown summary."""
        meta = report["report_metadata"]
        importer = report["importer_declaration"]
        goods = report["goods_summary"]
        emissions = report["emissions_summary"]
        validation = report["validation_results"]
        cert = report.get("certificate_obligation", {})
        amendment = report.get("amendment_metadata", {})

        summary = f"""# CBAM Transitional Registry Report

## Report Information
- **Report ID:** {meta['report_id']}
- **Quarter:** {meta['quarter']} ({meta['reporting_period_start']} to {meta['reporting_period_end']})
- **Generated:** {meta['generated_at']}
- **Generated By:** {meta['generated_by']}

## Importer Declaration
- **Company:** {importer['importer_name']}
- **Country:** {importer['importer_country']}
- **EORI:** {importer['importer_eori']}
- **Declarant:** {importer['declarant_name']} ({importer['declarant_position']})
- **Declaration Date:** {importer['declaration_date']}

## Summary Statistics

### Imported Goods
- **Total Shipments:** {goods['total_shipments']:,}
- **Total Mass:** {goods['total_mass_tonnes']:,.2f} tonnes

### Embedded Emissions
- **Total Emissions:** {emissions['total_embedded_emissions_tco2']:,.2f} tCO2e
- **Direct (Scope 1):** {emissions['total_direct_emissions_tco2']:,.2f} tCO2e
- **Indirect (Scope 2):** {emissions['total_indirect_emissions_tco2']:,.2f} tCO2e
- **Average Intensity:** {emissions['emissions_intensity_tco2_per_tonne']:.3f} tCO2e/tonne

### Calculation Methods
- **Default Values:** {emissions['calculation_methods_used']['default_values_count']}
- **Supplier Actual Data:** {emissions['calculation_methods_used']['actual_data_count']}
- **Complex Goods:** {emissions['calculation_methods_used']['complex_goods_count']}

## Breakdown by Product Group

| Product Group | Mass (tonnes) | Emissions (tCO2e) | % of Total |
|---------------|--------------|------------------|------------|
"""

        for pg in emissions['emissions_by_product_group']:
            summary += f"| {pg['product_group']} | {pg.get('total_mass_tonnes', 0):,.2f} | {pg['total_emissions_tco2']:,.2f} | {pg['percentage_of_total']:.1f}% |\n"

        summary += f"""
## Top Origin Countries

| Country | Mass (tonnes) | Emissions (tCO2e) | % of Total |
|---------|--------------|------------------|------------|
"""

        for country in emissions['emissions_by_origin_country'][:5]:
            summary += f"| {country['country_name']} ({country['country_iso']}) | {country.get('total_mass_tonnes', 0):,.2f} | {country['total_emissions_tco2']:,.2f} | {country['percentage_of_total']:.1f}% |\n"

        summary += f"""
## Validation Results

**Status:** {'✅ PASS - Ready for submission' if validation['is_valid'] else '❌ FAIL - Review errors below'}

### Validation Checks
"""

        for check in validation['rules_checked']:
            status_icon = "✅" if check['status'] == "pass" else "❌"
            summary += f"- {status_icon} {check['rule_name']}: {check['message']}\n"

        if validation['errors']:
            summary += "\n### Errors to Address\n"
            for error in validation['errors']:
                summary += f"- **{error['error_code']}:** {error['message']}\n"

        # v1.1: Certificate Obligation Summary
        if cert:
            summary += f"""
## Certificate Obligation Summary

- **Total Embedded Emissions:** {cert.get('total_embedded_emissions_tco2', 0):,.2f} tCO2e
- **Carbon Price Deduction:** {cert.get('carbon_price_deduction_tco2', 0):,.3f} tCO2e
- **Free Allocation Deduction:** {cert.get('free_allocation_deduction_tco2', 0):,.3f} tCO2e
- **Net Emissions:** {cert.get('net_emissions_tco2', 0):,.3f} tCO2e
- **ETS Price:** EUR {cert.get('ets_price_eur_per_tco2', 0):,.2f} / tCO2
- **Gross Obligation:** EUR {cert.get('gross_obligation_eur', 0):,.2f}
- **Deductions:** EUR {cert.get('deductions_eur', 0):,.2f}
- **Net Obligation:** EUR {cert.get('net_obligation_eur', 0):,.2f}
- **Certificates Required:** {cert.get('certificates_required', 0):,}
"""

        # v1.1: Amendment Information
        if amendment and amendment.get("is_amendment", False):
            summary += f"""
## Amendment Information

- **Original Report:** {amendment.get('original_report_id', 'N/A')}
- **Amendment Number:** {amendment.get('amendment_number', 0)}
- **Reason:** {amendment.get('amendment_reason', 'N/A')}
- **Date:** {amendment.get('amendment_date', 'N/A')}
- **Fields Amended:** {', '.join(amendment.get('fields_amended', []))}
"""

        summary += f"""
## Next Steps

1. **Review this report** for accuracy and completeness
2. **Submit to EU CBAM Transitional Registry** by the quarterly deadline
3. **Retain all supporting documents** (supplier EPDs, customs declarations) for audit

---

*Generated by GreenLang CBAM Importer Copilot v1.1.0*
"""

        return summary

    def write_summary(self, report: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Write human-readable summary to Markdown file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.generate_summary(report)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)

        logger.info(f"Wrote summary to {output_path}")


# ============================================================================
# CLI INTERFACE (for testing)
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CBAM Reporting Packager Agent")
    parser.add_argument("--input", required=True, help="Input shipments with emissions JSON")
    parser.add_argument("--output", required=True, help="Output report JSON file")
    parser.add_argument("--summary", help="Output summary Markdown file")
    parser.add_argument("--xml", help="Output XML file path (v1.1)")
    parser.add_argument("--rules", help="Path to CBAM rules YAML")
    parser.add_argument("--importer-name", required=True, help="EU importer name")
    parser.add_argument("--importer-country", required=True, help="EU country code")
    parser.add_argument("--importer-eori", required=True, help="EORI number")
    parser.add_argument("--declarant-name", required=True, help="Declarant name")
    parser.add_argument("--declarant-position", required=True, help="Declarant position")

    args = parser.parse_args()

    # Create agent
    agent = ReportingPackagerAgent(cbam_rules_path=args.rules)

    # Load input
    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    shipments = input_data.get("shipments", [])

    # Importer info
    importer_info = {
        "importer_name": args.importer_name,
        "importer_country": args.importer_country,
        "importer_eori": args.importer_eori,
        "declarant_name": args.declarant_name,
        "declarant_position": args.declarant_position
    }

    # Generate report
    report = agent.generate_report(shipments, importer_info)

    # Write outputs
    agent.write_report(report, args.output)

    if args.summary:
        agent.write_summary(report, args.summary)

    if args.xml:
        agent.write_xml_output(report, args.xml)

    # Print result
    print("\n" + "="*80)
    print("CBAM REPORT GENERATED")
    print("="*80)
    print(f"Report ID: {report['report_metadata']['report_id']}")
    print(f"Quarter: {report['report_metadata']['quarter']}")
    print(f"Total Shipments: {report['goods_summary']['total_shipments']}")
    print(f"Total Emissions: {report['emissions_summary']['total_embedded_emissions_tco2']:.2f} tCO2")
    print(f"Validation: {'PASS ✅' if report['validation_results']['is_valid'] else 'FAIL ❌'}")
    print(f"\nReport written to: {args.output}")
    if args.summary:
        print(f"Summary written to: {args.summary}")
    if args.xml:
        print(f"XML written to: {args.xml}")
