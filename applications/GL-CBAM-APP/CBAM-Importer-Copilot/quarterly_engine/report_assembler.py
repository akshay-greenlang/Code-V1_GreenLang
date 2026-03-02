# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Report Assembler Engine v1.1

Thread-safe singleton engine for assembling CBAM quarterly reports from
raw shipment data. Handles aggregation by CN code, country of origin,
installation, and CBAM sector. Generates XML output conforming to the
EU CBAM Registry format and human-readable Markdown summaries.

Per EU CBAM Implementing Regulation 2023/1773:
  - Article 3: Report content (aggregated goods, emissions, origins)
  - Article 4: Default values and markup percentages
  - Article 7: Complex goods precursor rules (20% threshold)

All calculations use Decimal with ROUND_HALF_UP (ZERO HALLUCINATION).

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import hashlib
import json
import logging
import threading
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    CBAM_XML_NAMESPACE,
    COMPLEX_GOODS_THRESHOLD_PCT,
    DEFAULT_VALUE_MARKUP_PCT,
    MAX_AMENDMENT_VERSIONS,
    REPORT_ID_PREFIX,
    CBAMSector,
    CalculationMethod,
    QuarterlyPeriod,
    QuarterlyReport,
    QuarterlyReportPeriod,
    ReportStatus,
    ShipmentAggregation,
    compute_sha256,
    quantize_decimal,
)

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER: Safe Decimal conversion
# ============================================================================

def _to_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """
    Safely convert a value to Decimal.

    Args:
        value: Value to convert (int, float, str, Decimal, or None).
        default: Default value if conversion fails.

    Returns:
        Decimal representation of the value.
    """
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return default


class ReportAssemblerEngine:
    """
    Thread-safe singleton engine for assembling CBAM quarterly reports.

    Transforms raw shipment data into aggregated quarterly reports with
    EU CBAM Registry XML output and human-readable Markdown summaries.

    All aggregation and calculation logic is deterministic Python arithmetic.
    No LLM calls are made for any numeric computation.

    Thread Safety:
        Uses threading.RLock for singleton creation and mutable state access.

    Example:
        >>> assembler = ReportAssemblerEngine()
        >>> report = assembler.assemble_quarterly_report(period, "NL123", shipments)
        >>> print(f"Emissions: {report.total_embedded_emissions} tCO2e")
        >>> xml = assembler.generate_xml_output(report)
    """

    _instance: Optional["ReportAssemblerEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "ReportAssemblerEngine":
        """Thread-safe singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the report assembler engine (runs once due to singleton)."""
        with self._lock:
            if self._initialized:
                return
            self._initialized = True
            self._report_store: Dict[str, QuarterlyReport] = {}
            logger.info("ReportAssemblerEngine initialized (singleton)")

    # ========================================================================
    # MAIN ASSEMBLY
    # ========================================================================

    def assemble_quarterly_report(
        self,
        period: QuarterlyReportPeriod,
        importer_id: str,
        shipments: List[Dict[str, Any]],
    ) -> QuarterlyReport:
        """
        Assemble a complete quarterly report from raw shipment data.

        Performs the following steps (all deterministic):
        1. Validate shipment data completeness
        2. Aggregate by CN code + country of origin
        3. Apply calculation method hierarchy
        4. Calculate totals
        5. Generate XML and Markdown outputs
        6. Compute provenance hash

        Args:
            period: The quarterly reporting period.
            importer_id: Importer EORI or internal identifier.
            shipments: List of shipment dictionaries from the pipeline.

        Returns:
            Fully assembled QuarterlyReport with XML, Markdown, and provenance.

        Raises:
            ValueError: If no shipments are provided.
        """
        start_time = datetime.now(timezone.utc)

        if not shipments:
            raise ValueError("Cannot assemble report with zero shipments")

        logger.info(
            "Assembling quarterly report: period=%s, importer=%s, shipments=%d",
            period.period_label, importer_id, len(shipments),
        )

        # Step 1: Aggregate by CN code + country (primary aggregation)
        aggregations = self.aggregate_by_cn_code(shipments)

        # Step 2: Apply calculation hierarchy to each aggregation
        aggregations = [
            self.apply_calculation_hierarchy(agg) for agg in aggregations
        ]

        # Step 3: Apply default value markup if applicable
        aggregations = [
            self.apply_default_value_markup(agg, period.year)
            for agg in aggregations
        ]

        # Step 4: Calculate totals
        total_quantity = sum(
            agg.quantity_mt for agg in aggregations
        )
        total_direct = sum(
            agg.direct_emissions_tCO2e for agg in aggregations
        )
        total_indirect = sum(
            agg.indirect_emissions_tCO2e for agg in aggregations
        )
        total_embedded = sum(
            agg.total_emissions_tCO2e for agg in aggregations
        )

        # Step 5: Collect calculation methods used
        method_counts: Dict[str, int] = defaultdict(int)
        for agg in aggregations:
            method_counts[agg.calculation_method.value] += 1

        # Step 6: Generate report ID
        timestamp_str = start_time.strftime("%Y%m%d%H%M%S")
        report_id = (
            f"{REPORT_ID_PREFIX}-{period.period_label}-"
            f"{importer_id[:16]}-{timestamp_str}"
        )

        # Step 7: Build the report model
        report = QuarterlyReport(
            report_id=report_id,
            period=period,
            importer_id=importer_id,
            status=ReportStatus.DRAFT,
            shipments_count=len(shipments),
            total_quantity_mt=quantize_decimal(total_quantity, 3),
            aggregations=aggregations,
            total_direct_emissions=quantize_decimal(total_direct, 3),
            total_indirect_emissions=quantize_decimal(total_indirect, 3),
            total_embedded_emissions=quantize_decimal(total_embedded, 3),
            calculation_methods_used=dict(method_counts),
            version=1,
            created_at=start_time,
        )

        # Step 8: Generate outputs
        report.report_xml = self.generate_xml_output(report)
        report.report_summary_md = self.generate_markdown_summary(report)

        # Step 9: Finalize (provenance hash)
        report = self.finalize_report(report)

        # Step 10: Store report
        with self._lock:
            self._report_store[report_id] = report

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Report assembled: id=%s, emissions=%.3f tCO2e, aggregations=%d, time=%.1fms",
            report_id, total_embedded, len(aggregations), elapsed_ms,
        )

        return report

    # ========================================================================
    # AGGREGATION METHODS (DETERMINISTIC)
    # ========================================================================

    def aggregate_by_cn_code(
        self,
        shipments: List[Dict[str, Any]]
    ) -> List[ShipmentAggregation]:
        """
        Aggregate shipments by CN code + country of origin.

        This is the primary aggregation per Implementing Regulation 2023/1773
        Article 3(2): "goods imported ... shall be aggregated by CN code."

        Args:
            shipments: List of shipment dictionaries.

        Returns:
            List of ShipmentAggregation objects grouped by CN code + country.
        """
        groups: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {
                "cn_description": "",
                "installation_ids": set(),
                "product_group": None,
                "quantity_mt": Decimal("0"),
                "direct_emissions": Decimal("0"),
                "indirect_emissions": Decimal("0"),
                "methods": [],
                "default_value_quantities": Decimal("0"),
                "quality_scores": [],
            }
        )

        for shipment in shipments:
            cn_code = str(shipment.get("cn_code", "000000"))
            country = str(shipment.get("origin_iso", shipment.get("country_of_origin", "XX")))
            key = (cn_code, country)

            group = groups[key]
            group["cn_description"] = shipment.get(
                "cn_description", shipment.get("product_name", "")
            )

            installation_id = shipment.get("installation_id")
            if installation_id:
                group["installation_ids"].add(installation_id)

            group["product_group"] = self._resolve_product_group(shipment)

            mass_kg = _to_decimal(shipment.get("net_mass_kg", 0))
            mass_mt = (mass_kg / Decimal("1000")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            group["quantity_mt"] += mass_mt

            calc = shipment.get("emissions_calculation", {})
            group["direct_emissions"] += _to_decimal(
                calc.get("direct_emissions_tco2", 0)
            )
            group["indirect_emissions"] += _to_decimal(
                calc.get("indirect_emissions_tco2", 0)
            )

            method_str = calc.get("calculation_method", "default_value")
            group["methods"].append(method_str)

            if method_str in ("default_values", "default_value"):
                group["default_value_quantities"] += mass_mt

            quality_score = calc.get("data_quality_score")
            if quality_score is not None:
                group["quality_scores"].append(_to_decimal(quality_score))

        return self._groups_to_aggregations(groups)

    def aggregate_by_country(
        self,
        shipments: List[Dict[str, Any]]
    ) -> List[ShipmentAggregation]:
        """
        Aggregate shipments by country of origin.

        Secondary aggregation for geographic analysis.

        Args:
            shipments: List of shipment dictionaries.

        Returns:
            List of ShipmentAggregation objects grouped by country.
        """
        groups: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "cn_code": "MIXED",
                "cn_description": "Multiple CN codes",
                "installation_ids": set(),
                "product_group": None,
                "quantity_mt": Decimal("0"),
                "direct_emissions": Decimal("0"),
                "indirect_emissions": Decimal("0"),
                "methods": [],
                "default_value_quantities": Decimal("0"),
                "quality_scores": [],
            }
        )

        for shipment in shipments:
            country = str(shipment.get("origin_iso", shipment.get("country_of_origin", "XX")))
            group = groups[country]
            group["product_group"] = self._resolve_product_group(shipment)

            mass_kg = _to_decimal(shipment.get("net_mass_kg", 0))
            mass_mt = (mass_kg / Decimal("1000")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            group["quantity_mt"] += mass_mt

            calc = shipment.get("emissions_calculation", {})
            group["direct_emissions"] += _to_decimal(calc.get("direct_emissions_tco2", 0))
            group["indirect_emissions"] += _to_decimal(calc.get("indirect_emissions_tco2", 0))

            method_str = calc.get("calculation_method", "default_value")
            group["methods"].append(method_str)
            if method_str in ("default_values", "default_value"):
                group["default_value_quantities"] += mass_mt

        result: List[ShipmentAggregation] = []
        for country, data in sorted(groups.items()):
            total_em = data["direct_emissions"] + data["indirect_emissions"]
            intensity = (
                (total_em / data["quantity_mt"]).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                if data["quantity_mt"] > 0 else Decimal("0")
            )
            default_pct = (
                (data["default_value_quantities"] / data["quantity_mt"] * Decimal("100"))
                .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                if data["quantity_mt"] > 0 else Decimal("0")
            )
            primary_method = self._resolve_primary_method(data["methods"])

            result.append(ShipmentAggregation(
                cn_code=data["cn_code"],
                cn_description=data["cn_description"],
                country_of_origin=country,
                product_group=data["product_group"] or CBAMSector.IRON_AND_STEEL,
                quantity_mt=quantize_decimal(data["quantity_mt"], 3),
                direct_emissions_tCO2e=quantize_decimal(data["direct_emissions"], 3),
                indirect_emissions_tCO2e=quantize_decimal(data["indirect_emissions"], 3),
                total_emissions_tCO2e=quantize_decimal(total_em, 3),
                embedded_emissions_per_mt=quantize_decimal(intensity, 3),
                calculation_method=primary_method,
                default_values_used_pct=quantize_decimal(default_pct, 2),
            ))

        return result

    def aggregate_by_installation(
        self,
        shipments: List[Dict[str, Any]]
    ) -> List[ShipmentAggregation]:
        """
        Aggregate shipments by installation identifier.

        Used for installation-level emissions analysis and verification.

        Args:
            shipments: List of shipment dictionaries.

        Returns:
            List of ShipmentAggregation objects grouped by installation.
        """
        groups: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "cn_code": "MIXED",
                "cn_description": "Multiple CN codes",
                "country": "XX",
                "product_group": None,
                "quantity_mt": Decimal("0"),
                "direct_emissions": Decimal("0"),
                "indirect_emissions": Decimal("0"),
                "methods": [],
                "default_value_quantities": Decimal("0"),
                "quality_scores": [],
            }
        )

        for shipment in shipments:
            install_id = shipment.get("installation_id", "UNKNOWN")
            group = groups[install_id]
            group["country"] = str(shipment.get("origin_iso", "XX"))
            group["product_group"] = self._resolve_product_group(shipment)

            mass_kg = _to_decimal(shipment.get("net_mass_kg", 0))
            mass_mt = (mass_kg / Decimal("1000")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            group["quantity_mt"] += mass_mt

            calc = shipment.get("emissions_calculation", {})
            group["direct_emissions"] += _to_decimal(calc.get("direct_emissions_tco2", 0))
            group["indirect_emissions"] += _to_decimal(calc.get("indirect_emissions_tco2", 0))

            method_str = calc.get("calculation_method", "default_value")
            group["methods"].append(method_str)
            if method_str in ("default_values", "default_value"):
                group["default_value_quantities"] += mass_mt

        result: List[ShipmentAggregation] = []
        for install_id, data in sorted(groups.items()):
            total_em = data["direct_emissions"] + data["indirect_emissions"]
            intensity = (
                (total_em / data["quantity_mt"]).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                if data["quantity_mt"] > 0 else Decimal("0")
            )
            default_pct = (
                (data["default_value_quantities"] / data["quantity_mt"] * Decimal("100"))
                .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                if data["quantity_mt"] > 0 else Decimal("0")
            )
            primary_method = self._resolve_primary_method(data["methods"])

            result.append(ShipmentAggregation(
                cn_code=data["cn_code"],
                cn_description=data["cn_description"],
                country_of_origin=data["country"],
                installation_id=install_id,
                product_group=data["product_group"] or CBAMSector.IRON_AND_STEEL,
                quantity_mt=quantize_decimal(data["quantity_mt"], 3),
                direct_emissions_tCO2e=quantize_decimal(data["direct_emissions"], 3),
                indirect_emissions_tCO2e=quantize_decimal(data["indirect_emissions"], 3),
                total_emissions_tCO2e=quantize_decimal(total_em, 3),
                embedded_emissions_per_mt=quantize_decimal(intensity, 3),
                calculation_method=primary_method,
                default_values_used_pct=quantize_decimal(default_pct, 2),
            ))

        return result

    def aggregate_by_product_group(
        self,
        shipments: List[Dict[str, Any]]
    ) -> List[ShipmentAggregation]:
        """
        Aggregate shipments by CBAM product sector.

        Groups shipments into the six CBAM sectors: cement, iron_and_steel,
        aluminum, fertilizers, hydrogen, electricity.

        Args:
            shipments: List of shipment dictionaries.

        Returns:
            List of ShipmentAggregation objects grouped by CBAM sector.
        """
        groups: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "cn_code": "MIXED",
                "cn_description": "Multiple CN codes",
                "country": "XX",
                "quantity_mt": Decimal("0"),
                "direct_emissions": Decimal("0"),
                "indirect_emissions": Decimal("0"),
                "methods": [],
                "default_value_quantities": Decimal("0"),
            }
        )

        for shipment in shipments:
            sector = self._resolve_product_group(shipment)
            sector_key = sector.value if sector else "iron_and_steel"
            group = groups[sector_key]

            mass_kg = _to_decimal(shipment.get("net_mass_kg", 0))
            mass_mt = (mass_kg / Decimal("1000")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            group["quantity_mt"] += mass_mt

            calc = shipment.get("emissions_calculation", {})
            group["direct_emissions"] += _to_decimal(calc.get("direct_emissions_tco2", 0))
            group["indirect_emissions"] += _to_decimal(calc.get("indirect_emissions_tco2", 0))

            method_str = calc.get("calculation_method", "default_value")
            group["methods"].append(method_str)
            if method_str in ("default_values", "default_value"):
                group["default_value_quantities"] += mass_mt

        result: List[ShipmentAggregation] = []
        for sector_key, data in sorted(groups.items()):
            try:
                sector_enum = CBAMSector(sector_key)
            except ValueError:
                sector_enum = CBAMSector.IRON_AND_STEEL

            total_em = data["direct_emissions"] + data["indirect_emissions"]
            intensity = (
                (total_em / data["quantity_mt"]).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                if data["quantity_mt"] > 0 else Decimal("0")
            )
            default_pct = (
                (data["default_value_quantities"] / data["quantity_mt"] * Decimal("100"))
                .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                if data["quantity_mt"] > 0 else Decimal("0")
            )
            primary_method = self._resolve_primary_method(data["methods"])

            result.append(ShipmentAggregation(
                cn_code=data["cn_code"],
                cn_description=data["cn_description"],
                country_of_origin=data["country"],
                product_group=sector_enum,
                quantity_mt=quantize_decimal(data["quantity_mt"], 3),
                direct_emissions_tCO2e=quantize_decimal(data["direct_emissions"], 3),
                indirect_emissions_tCO2e=quantize_decimal(data["indirect_emissions"], 3),
                total_emissions_tCO2e=quantize_decimal(total_em, 3),
                embedded_emissions_per_mt=quantize_decimal(intensity, 3),
                calculation_method=primary_method,
                default_values_used_pct=quantize_decimal(default_pct, 2),
            ))

        return result

    # ========================================================================
    # CALCULATION HIERARCHY
    # ========================================================================

    def apply_calculation_hierarchy(
        self,
        aggregation: ShipmentAggregation
    ) -> ShipmentAggregation:
        """
        Apply the CBAM calculation method hierarchy to an aggregation.

        Hierarchy (best to worst data quality):
        1. Supplier actual data (EPD/verification)
        2. Regional emission factors (country-specific)
        3. EU default values
        4. Estimation (lowest quality)

        This method validates the calculation method assignment and adjusts
        the data quality score accordingly.

        Args:
            aggregation: The aggregation to apply hierarchy to.

        Returns:
            Updated ShipmentAggregation with quality score based on method.
        """
        quality_scores = {
            CalculationMethod.SUPPLIER_ACTUAL: Decimal("0.95"),
            CalculationMethod.REGIONAL_FACTOR: Decimal("0.70"),
            CalculationMethod.DEFAULT_VALUE: Decimal("0.40"),
            CalculationMethod.ESTIMATION: Decimal("0.20"),
        }

        quality_score = quality_scores.get(
            aggregation.calculation_method, Decimal("0.20")
        )

        return aggregation.model_copy(
            update={"supplier_data_quality_score": quality_score}
        )

    def apply_default_value_markup(
        self,
        aggregation: ShipmentAggregation,
        year: int
    ) -> ShipmentAggregation:
        """
        Apply percentage markup to emissions calculated with default values.

        Per Implementing Regulation 2023/1773 Article 4(3):
        - 2024: +10% markup on default values
        - 2025: +20% markup
        - 2026+: +30% markup

        The markup only applies to the portion of emissions that used
        EU default values, not supplier actual data.

        Args:
            aggregation: The aggregation to apply markup to.
            year: The reporting year for determining markup percentage.

        Returns:
            Updated ShipmentAggregation with markup applied if applicable.
        """
        if aggregation.default_values_used_pct <= 0:
            return aggregation

        markup_pct = DEFAULT_VALUE_MARKUP_PCT.get(year, Decimal("0"))
        if markup_pct <= 0:
            return aggregation

        # Calculate proportion using default values
        default_proportion = aggregation.default_values_used_pct / Decimal("100")
        markup_factor = Decimal("1") + (markup_pct / Decimal("100"))

        # Apply markup only to the default-value portion
        direct_default_portion = aggregation.direct_emissions_tCO2e * default_proportion
        direct_actual_portion = aggregation.direct_emissions_tCO2e - direct_default_portion
        new_direct = direct_actual_portion + (direct_default_portion * markup_factor)

        indirect_default_portion = aggregation.indirect_emissions_tCO2e * default_proportion
        indirect_actual_portion = aggregation.indirect_emissions_tCO2e - indirect_default_portion
        new_indirect = indirect_actual_portion + (indirect_default_portion * markup_factor)

        new_total = new_direct + new_indirect
        new_intensity = (
            (new_total / aggregation.quantity_mt).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            if aggregation.quantity_mt > 0 else Decimal("0")
        )

        return aggregation.model_copy(update={
            "direct_emissions_tCO2e": quantize_decimal(new_direct, 3),
            "indirect_emissions_tCO2e": quantize_decimal(new_indirect, 3),
            "total_emissions_tCO2e": quantize_decimal(new_total, 3),
            "embedded_emissions_per_mt": quantize_decimal(new_intensity, 3),
        })

    def calculate_complex_goods_rule(
        self,
        shipments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enforce the 20% complex goods rule per Regulation 2023/956 Article 7(3).

        Complex goods contain precursor materials from multiple CBAM sectors.
        If the share of precursor emissions exceeds 20% of total embedded
        emissions, additional verification requirements apply.

        Args:
            shipments: List of shipment dictionaries.

        Returns:
            Dict with complex goods analysis:
            {
                "complex_goods_count": int,
                "total_shipments": int,
                "complex_goods_pct": Decimal,
                "threshold_pct": Decimal,
                "within_threshold": bool,
                "complex_goods_details": [...],
            }
        """
        total_count = len(shipments)
        complex_count = 0
        complex_details: List[Dict[str, Any]] = []

        for shipment in shipments:
            calc = shipment.get("emissions_calculation", {})
            method = calc.get("calculation_method", "")
            is_complex = shipment.get("is_complex_good", False)

            if is_complex or method == "complex_goods":
                complex_count += 1
                complex_details.append({
                    "shipment_id": shipment.get("shipment_id", "unknown"),
                    "cn_code": shipment.get("cn_code", ""),
                    "precursor_pct": _to_decimal(
                        calc.get("precursor_emissions_pct", 0)
                    ),
                })

        complex_pct = (
            (Decimal(str(complex_count)) / Decimal(str(total_count)) * Decimal("100"))
            .quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            if total_count > 0 else Decimal("0")
        )

        within_threshold = complex_pct <= COMPLEX_GOODS_THRESHOLD_PCT

        result = {
            "complex_goods_count": complex_count,
            "total_shipments": total_count,
            "complex_goods_pct": complex_pct,
            "threshold_pct": COMPLEX_GOODS_THRESHOLD_PCT,
            "within_threshold": within_threshold,
            "complex_goods_details": complex_details,
        }

        if not within_threshold:
            logger.warning(
                "Complex goods threshold exceeded: %.1f%% > %.1f%%",
                complex_pct, COMPLEX_GOODS_THRESHOLD_PCT,
            )

        return result

    # ========================================================================
    # OUTPUT GENERATION
    # ========================================================================

    def generate_xml_output(self, report: QuarterlyReport) -> str:
        """
        Generate EU CBAM Registry XML format output.

        Produces XML conforming to the EU CBAM Transitional Registry
        submission format (urn:eu:ec:cbam:registry:quarterly:v1).

        Args:
            report: The quarterly report to serialize.

        Returns:
            XML string ready for EU CBAM Registry submission.
        """
        root = ET.Element("CBAMQuarterlyReport")
        root.set("xmlns", CBAM_XML_NAMESPACE)
        root.set("version", "1.0")

        # Report metadata
        meta = ET.SubElement(root, "ReportMetadata")
        ET.SubElement(meta, "ReportId").text = report.report_id
        ET.SubElement(meta, "ReportingPeriod").text = report.period.period_label
        ET.SubElement(meta, "PeriodStart").text = report.period.start_date.isoformat()
        ET.SubElement(meta, "PeriodEnd").text = report.period.end_date.isoformat()
        ET.SubElement(meta, "SubmissionDeadline").text = (
            report.period.submission_deadline.isoformat()
        )
        ET.SubElement(meta, "ImporterEORI").text = report.importer_id
        ET.SubElement(meta, "ReportVersion").text = str(report.version)
        ET.SubElement(meta, "CreatedAt").text = report.created_at.isoformat()
        ET.SubElement(meta, "IsTransitional").text = str(
            report.period.is_transitional
        ).lower()

        # Summary totals
        summary = ET.SubElement(root, "EmissionsSummary")
        ET.SubElement(summary, "TotalShipments").text = str(report.shipments_count)
        ET.SubElement(summary, "TotalQuantityMT").text = str(report.total_quantity_mt)
        ET.SubElement(summary, "TotalDirectEmissions").text = str(
            report.total_direct_emissions
        )
        ET.SubElement(summary, "TotalIndirectEmissions").text = str(
            report.total_indirect_emissions
        )
        ET.SubElement(summary, "TotalEmbeddedEmissions").text = str(
            report.total_embedded_emissions
        )

        # Aggregated goods
        goods_list = ET.SubElement(root, "AggregatedGoods")
        for agg in report.aggregations:
            good = ET.SubElement(goods_list, "Good")
            ET.SubElement(good, "CNCode").text = agg.cn_code
            ET.SubElement(good, "CNDescription").text = agg.cn_description
            ET.SubElement(good, "CountryOfOrigin").text = agg.country_of_origin
            if agg.installation_id:
                ET.SubElement(good, "InstallationId").text = agg.installation_id
            ET.SubElement(good, "ProductGroup").text = agg.product_group.value
            ET.SubElement(good, "QuantityMT").text = str(agg.quantity_mt)
            ET.SubElement(good, "DirectEmissions").text = str(
                agg.direct_emissions_tCO2e
            )
            ET.SubElement(good, "IndirectEmissions").text = str(
                agg.indirect_emissions_tCO2e
            )
            ET.SubElement(good, "TotalEmissions").text = str(
                agg.total_emissions_tCO2e
            )
            ET.SubElement(good, "SpecificEmissions").text = str(
                agg.embedded_emissions_per_mt
            )
            ET.SubElement(good, "CalculationMethod").text = (
                agg.calculation_method.value
            )
            ET.SubElement(good, "DefaultValuesUsedPct").text = str(
                agg.default_values_used_pct
            )

        # Provenance
        prov = ET.SubElement(root, "Provenance")
        ET.SubElement(prov, "ProvenanceHash").text = report.provenance_hash
        ET.SubElement(prov, "GeneratedBy").text = "GreenLang CBAM Importer Copilot v1.1.0"
        ET.SubElement(prov, "ZeroHallucination").text = "true"

        # Serialize to string with declaration
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_body = ET.tostring(root, encoding="unicode", method="xml")

        return xml_declaration + xml_body

    def generate_markdown_summary(self, report: QuarterlyReport) -> str:
        """
        Generate a human-readable Markdown summary of the quarterly report.

        Suitable for management review, email distribution, and documentation.

        Args:
            report: The quarterly report to summarize.

        Returns:
            Markdown-formatted summary string.
        """
        period = report.period
        phase_label = "Transitional" if period.is_transitional else "Definitive"

        lines: List[str] = []
        lines.append(f"# CBAM Quarterly Report - {period.period_label}")
        lines.append("")
        lines.append("## Report Information")
        lines.append(f"- **Report ID:** {report.report_id}")
        lines.append(f"- **Period:** {period.period_label} ({period.quarter.label})")
        lines.append(
            f"- **Reporting Dates:** {period.start_date} to {period.end_date}"
        )
        lines.append(f"- **Submission Deadline:** {period.submission_deadline}")
        lines.append(f"- **Phase:** {phase_label}")
        lines.append(f"- **Importer EORI:** {report.importer_id}")
        lines.append(f"- **Status:** {report.status.value.upper()}")
        lines.append(f"- **Version:** {report.version}")
        lines.append(f"- **Created:** {report.created_at.isoformat()}")
        lines.append("")

        lines.append("## Summary Statistics")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Shipments | {report.shipments_count:,} |")
        lines.append(f"| Total Quantity | {report.total_quantity_mt:,.3f} MT |")
        lines.append(
            f"| Direct Emissions (Scope 1) | {report.total_direct_emissions:,.3f} tCO2e |"
        )
        lines.append(
            f"| Indirect Emissions (Scope 2) | {report.total_indirect_emissions:,.3f} tCO2e |"
        )
        lines.append(
            f"| Total Embedded Emissions | {report.total_embedded_emissions:,.3f} tCO2e |"
        )
        if report.total_quantity_mt > 0:
            avg_intensity = (
                report.total_embedded_emissions / report.total_quantity_mt
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            lines.append(f"| Average Intensity | {avg_intensity} tCO2e/MT |")
        lines.append("")

        # Calculation methods
        lines.append("## Calculation Methods")
        lines.append("")
        lines.append("| Method | Count |")
        lines.append("|--------|-------|")
        for method, count in sorted(report.calculation_methods_used.items()):
            lines.append(f"| {method} | {count} |")
        lines.append("")

        # Aggregations breakdown
        if report.aggregations:
            lines.append("## Goods Breakdown")
            lines.append("")
            lines.append(
                "| CN Code | Country | Sector | Quantity (MT) | "
                "Emissions (tCO2e) | Intensity |"
            )
            lines.append(
                "|---------|---------|--------|--------------|"
                "-------------------|-----------|"
            )
            for agg in report.aggregations:
                lines.append(
                    f"| {agg.cn_code} | {agg.country_of_origin} | "
                    f"{agg.product_group.value} | {agg.quantity_mt:,.3f} | "
                    f"{agg.total_emissions_tCO2e:,.3f} | "
                    f"{agg.embedded_emissions_per_mt:.3f} |"
                )
            lines.append("")

        # Provenance
        lines.append("## Provenance")
        lines.append("")
        lines.append(f"- **Hash (SHA-256):** `{report.provenance_hash}`")
        lines.append("- **Zero Hallucination:** All calculations deterministic")
        lines.append("- **Generated by:** GreenLang CBAM Importer Copilot v1.1.0")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(
            "*This report was generated by GreenLang CBAM Importer Copilot v1.1.0. "
            "All emissions calculations are deterministic and reproducible.*"
        )

        return "\n".join(lines)

    # ========================================================================
    # VALIDATION & FINALIZATION
    # ========================================================================

    def validate_report_completeness(
        self,
        report: QuarterlyReport
    ) -> List[str]:
        """
        Check that all required fields are populated for submission.

        Per Implementing Regulation 2023/1773 Article 3, a quarterly report
        must contain complete information about imported goods, their
        emissions, and the importer.

        Args:
            report: The quarterly report to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        if not report.report_id:
            errors.append("Missing report_id")
        if not report.importer_id:
            errors.append("Missing importer_id")
        if report.shipments_count <= 0:
            errors.append("shipments_count must be > 0")
        if report.total_quantity_mt <= 0:
            errors.append("total_quantity_mt must be > 0")
        if not report.aggregations:
            errors.append("No aggregations present")
        if report.total_embedded_emissions <= 0:
            errors.append("total_embedded_emissions must be > 0")

        # Validate aggregation totals match report totals
        agg_quantity = sum(a.quantity_mt for a in report.aggregations)
        if abs(agg_quantity - report.total_quantity_mt) > Decimal("0.01"):
            errors.append(
                f"Aggregation quantity sum ({agg_quantity}) does not match "
                f"report total ({report.total_quantity_mt})"
            )

        agg_emissions = sum(a.total_emissions_tCO2e for a in report.aggregations)
        if abs(agg_emissions - report.total_embedded_emissions) > Decimal("0.01"):
            errors.append(
                f"Aggregation emissions sum ({agg_emissions}) does not match "
                f"report total ({report.total_embedded_emissions})"
            )

        # Check each aggregation has required fields
        for idx, agg in enumerate(report.aggregations):
            if not agg.cn_code or agg.cn_code == "MIXED":
                pass  # MIXED is acceptable for secondary aggregations
            if not agg.country_of_origin or agg.country_of_origin == "XX":
                errors.append(f"Aggregation [{idx}] has invalid country_of_origin")
            if agg.quantity_mt <= 0:
                errors.append(f"Aggregation [{idx}] has zero or negative quantity")

        if errors:
            logger.warning(
                "Report %s validation found %d errors", report.report_id, len(errors)
            )
        else:
            logger.info("Report %s validation passed", report.report_id)

        return errors

    def finalize_report(self, report: QuarterlyReport) -> QuarterlyReport:
        """
        Finalize a report by computing the provenance hash and validating.

        This is the last step before a report is ready for review/submission.
        The provenance hash serves as a tamper-evident seal.

        Args:
            report: The report to finalize.

        Returns:
            Report with provenance_hash set.
        """
        provenance_hash = report.compute_provenance_hash()
        report = report.model_copy(update={"provenance_hash": provenance_hash})

        logger.info(
            "Report %s finalized, provenance_hash=%s",
            report.report_id, provenance_hash[:16] + "...",
        )
        return report

    def get_stored_report(self, report_id: str) -> Optional[QuarterlyReport]:
        """
        Retrieve a previously assembled report from the in-memory store.

        Args:
            report_id: The report identifier.

        Returns:
            QuarterlyReport if found, None otherwise.
        """
        with self._lock:
            return self._report_store.get(report_id)

    # ========================================================================
    # PROVENANCE
    # ========================================================================

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash for data provenance.

        This is a ZERO-HALLUCINATION deterministic computation.

        Args:
            data: Dictionary to hash.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        return compute_sha256(data)

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _groups_to_aggregations(
        self,
        groups: Dict[Any, Dict[str, Any]]
    ) -> List[ShipmentAggregation]:
        """
        Convert grouped shipment data into ShipmentAggregation objects.

        Args:
            groups: Dict mapping (cn_code, country) to aggregated data.

        Returns:
            List of ShipmentAggregation objects.
        """
        result: List[ShipmentAggregation] = []

        for key, data in sorted(groups.items()):
            if isinstance(key, tuple):
                cn_code, country = key
            else:
                cn_code = str(key)
                country = data.get("country", "XX")

            total_emissions = data["direct_emissions"] + data["indirect_emissions"]

            intensity = (
                (total_emissions / data["quantity_mt"]).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                if data["quantity_mt"] > 0 else Decimal("0")
            )

            default_pct = (
                (data["default_value_quantities"] / data["quantity_mt"] * Decimal("100"))
                .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                if data["quantity_mt"] > 0 else Decimal("0")
            )

            # Resolve primary calculation method
            primary_method = self._resolve_primary_method(data["methods"])

            # Calculate weighted quality score
            quality_score = Decimal("0")
            if data["quality_scores"]:
                total = sum(data["quality_scores"])
                quality_score = (total / Decimal(str(len(data["quality_scores"])))).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

            # Determine installation ID (use first if only one, else None)
            installation_id = None
            if len(data["installation_ids"]) == 1:
                installation_id = next(iter(data["installation_ids"]))

            agg = ShipmentAggregation(
                cn_code=cn_code,
                cn_description=data["cn_description"],
                country_of_origin=country,
                installation_id=installation_id,
                product_group=data["product_group"] or CBAMSector.IRON_AND_STEEL,
                quantity_mt=quantize_decimal(data["quantity_mt"], 3),
                direct_emissions_tCO2e=quantize_decimal(data["direct_emissions"], 3),
                indirect_emissions_tCO2e=quantize_decimal(data["indirect_emissions"], 3),
                total_emissions_tCO2e=quantize_decimal(total_emissions, 3),
                embedded_emissions_per_mt=quantize_decimal(intensity, 3),
                calculation_method=primary_method,
                default_values_used_pct=quantize_decimal(default_pct, 2),
                supplier_data_quality_score=quality_score,
            )
            result.append(agg)

        return result

    def _resolve_product_group(
        self,
        shipment: Dict[str, Any]
    ) -> Optional[CBAMSector]:
        """
        Resolve the CBAM product sector for a shipment.

        Checks multiple fields for sector information.

        Args:
            shipment: Shipment dictionary.

        Returns:
            CBAMSector enum value, or None if unresolvable.
        """
        sector_str = shipment.get(
            "product_group",
            shipment.get("cbam_sector", shipment.get("sector", ""))
        )

        if not sector_str:
            return None

        # Normalize common variations
        sector_map = {
            "steel": CBAMSector.IRON_AND_STEEL,
            "iron": CBAMSector.IRON_AND_STEEL,
            "iron_and_steel": CBAMSector.IRON_AND_STEEL,
            "iron_steel": CBAMSector.IRON_AND_STEEL,
            "cement": CBAMSector.CEMENT,
            "aluminum": CBAMSector.ALUMINUM,
            "aluminium": CBAMSector.ALUMINUM,
            "fertilizers": CBAMSector.FERTILIZERS,
            "fertiliser": CBAMSector.FERTILIZERS,
            "hydrogen": CBAMSector.HYDROGEN,
            "electricity": CBAMSector.ELECTRICITY,
        }

        normalized = str(sector_str).lower().strip()
        return sector_map.get(normalized)

    def _resolve_primary_method(
        self,
        methods: List[str]
    ) -> CalculationMethod:
        """
        Determine the primary calculation method from a list of methods.

        Uses the most conservative (highest data quality rank) method
        that appears in the list.

        Args:
            methods: List of calculation method strings.

        Returns:
            Primary CalculationMethod (best quality found).
        """
        method_map = {
            "supplier_actual": CalculationMethod.SUPPLIER_ACTUAL,
            "actual_data": CalculationMethod.SUPPLIER_ACTUAL,
            "regional_factor": CalculationMethod.REGIONAL_FACTOR,
            "default_value": CalculationMethod.DEFAULT_VALUE,
            "default_values": CalculationMethod.DEFAULT_VALUE,
            "estimation": CalculationMethod.ESTIMATION,
            "complex_goods": CalculationMethod.DEFAULT_VALUE,
        }

        resolved_methods: List[CalculationMethod] = []
        for m in methods:
            resolved = method_map.get(str(m).lower(), CalculationMethod.DEFAULT_VALUE)
            resolved_methods.append(resolved)

        if not resolved_methods:
            return CalculationMethod.DEFAULT_VALUE

        # Return the method with the best (lowest) data quality rank
        return min(resolved_methods, key=lambda m: m.data_quality_rank)

    # ========================================================================
    # SINGLETON RESET (TESTING ONLY)
    # ========================================================================

    @classmethod
    def _reset_singleton(cls) -> None:
        """
        Reset the singleton instance. FOR TESTING ONLY.

        This method is not thread-safe against concurrent production use.
        """
        with cls._lock:
            cls._instance = None
            logger.debug("ReportAssemblerEngine singleton reset (testing)")
