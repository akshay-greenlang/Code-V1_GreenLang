"""
Report Generator -- Multi-Format GHG Inventory Reports

Generates GHG inventory reports in JSON, CSV, Excel (structure), and
PDF (structure) formats.  Reports follow the GHG Protocol Corporate
Standard disclosure framework with sections covering:

  1. Executive Summary
  2. Organizational Boundary
  3. Operational Boundary
  4. Base Year
  5. Scope 1 Emissions
  6. Scope 2 Emissions
  7. Scope 3 Emissions
  8. Total Emissions
  9. Intensity Metrics
  10. Trend Analysis
  11. Uncertainty Assessment
  12. Data Quality
  13. Verification Statement
  14. Methodology Notes
  15. Appendices

Example:
    >>> gen = ReportGenerator(config)
    >>> report = gen.generate_report(inventory_id, ReportFormat.JSON)
    >>> print(report.sections)
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    GHGAppConfig,
    GHGGas,
    ReportFormat,
    Scope,
)
from .models import (
    GHGInventory,
    Report,
    ReportSection,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates GHG inventory reports in multiple formats.

    Each report is composed of configurable sections.  The generator
    builds structured data for each section, then serializes to the
    requested format.
    """

    REPORT_SECTIONS: List[str] = [
        "executive_summary",
        "organizational_boundary",
        "operational_boundary",
        "base_year",
        "scope1_emissions",
        "scope2_emissions",
        "scope3_emissions",
        "total_emissions",
        "intensity_metrics",
        "trend_analysis",
        "uncertainty_assessment",
        "data_quality",
        "verification_statement",
        "methodology_notes",
        "appendices",
    ]

    SECTION_TITLES: Dict[str, str] = {
        "executive_summary": "Executive Summary",
        "organizational_boundary": "Organizational Boundary",
        "operational_boundary": "Operational Boundary",
        "base_year": "Base Year and Recalculation Policy",
        "scope1_emissions": "Scope 1 Direct Emissions",
        "scope2_emissions": "Scope 2 Indirect Emissions (Energy)",
        "scope3_emissions": "Scope 3 Other Indirect Emissions",
        "total_emissions": "Total GHG Emissions",
        "intensity_metrics": "GHG Intensity Metrics",
        "trend_analysis": "Emissions Trend Analysis",
        "uncertainty_assessment": "Uncertainty Assessment",
        "data_quality": "Data Quality Assessment",
        "verification_statement": "Verification / Assurance Statement",
        "methodology_notes": "Methodology and Emission Factor Notes",
        "appendices": "Appendices",
    }

    def __init__(
        self,
        config: Optional[GHGAppConfig] = None,
        inventory_store: Optional[Dict[str, GHGInventory]] = None,
    ) -> None:
        """
        Initialize ReportGenerator.

        Args:
            config: Application configuration.
            inventory_store: Shared reference to inventory storage.
        """
        self.config = config or GHGAppConfig()
        self._inventory_store = inventory_store if inventory_store is not None else {}
        self._report_history: Dict[str, List[Report]] = {}
        logger.info("ReportGenerator initialized")

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def generate_report(
        self,
        inventory_id: str,
        report_format: ReportFormat = ReportFormat.JSON,
        sections: Optional[List[str]] = None,
    ) -> Report:
        """
        Generate a complete GHG inventory report.

        Args:
            inventory_id: Inventory ID.
            report_format: Output format.
            sections: Specific sections to include (None = all).

        Returns:
            Generated Report.
        """
        start = datetime.utcnow()
        inventory = self._get_inventory_or_raise(inventory_id)

        selected_sections = sections or self.REPORT_SECTIONS
        report_sections: List[ReportSection] = []

        for idx, section_key in enumerate(selected_sections):
            if section_key not in self.REPORT_SECTIONS:
                logger.warning("Unknown section '%s', skipping", section_key)
                continue

            content = self._generate_section(section_key, inventory)
            report_sections.append(
                ReportSection(
                    key=section_key,
                    title=self.SECTION_TITLES.get(section_key, section_key),
                    content=content,
                    order=idx,
                )
            )

        report = Report(
            inventory_id=inventory_id,
            format=report_format,
            sections=report_sections,
        )

        # Track in history
        if inventory_id not in self._report_history:
            self._report_history[inventory_id] = []
        self._report_history[inventory_id].append(report)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Generated %s report for inventory %s (%d sections) in %.1f ms",
            report_format.value,
            inventory_id,
            len(report_sections),
            elapsed_ms,
        )
        return report

    def generate_executive_summary(
        self,
        inventory: GHGInventory,
    ) -> Dict[str, Any]:
        """Generate executive summary section data."""
        return self._generate_executive_summary_content(inventory)

    def generate_scope1_section(
        self,
        inventory: GHGInventory,
    ) -> Dict[str, Any]:
        """Generate Scope 1 emissions section data."""
        return self._generate_scope1_content(inventory)

    def generate_scope2_section(
        self,
        inventory: GHGInventory,
    ) -> Dict[str, Any]:
        """Generate Scope 2 emissions section data."""
        return self._generate_scope2_content(inventory)

    def generate_scope3_section(
        self,
        inventory: GHGInventory,
    ) -> Dict[str, Any]:
        """Generate Scope 3 emissions section data."""
        return self._generate_scope3_content(inventory)

    def generate_trend_section(
        self,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Generate trend analysis section across all years.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with multi-year trend data.
        """
        inventories = sorted(
            [
                inv for inv in self._inventory_store.values()
                if inv.org_id == org_id
            ],
            key=lambda i: i.year,
        )

        if not inventories:
            return {"message": "No historical data available for trend analysis"}

        years: List[Dict[str, Any]] = []
        for inv in inventories:
            years.append({
                "year": inv.year,
                "total_tco2e": str(inv.grand_total_tco2e),
                "scope1": str(inv.scope1.total_tco2e) if inv.scope1 else "0",
                "scope2_location": str(inv.scope2_location.total_tco2e) if inv.scope2_location else "0",
                "scope2_market": str(inv.scope2_market.total_tco2e) if inv.scope2_market else "0",
                "scope3": str(inv.scope3.total_tco2e) if inv.scope3 else "0",
            })

        return {
            "org_id": org_id,
            "years_covered": len(years),
            "earliest_year": inventories[0].year,
            "latest_year": inventories[-1].year,
            "data": years,
        }

    # ------------------------------------------------------------------
    # Export Methods
    # ------------------------------------------------------------------

    def export_json(self, inventory_id: str) -> Dict[str, Any]:
        """
        Export inventory as a structured JSON document.

        Args:
            inventory_id: Inventory ID.

        Returns:
            JSON-serializable dict.
        """
        inventory = self._get_inventory_or_raise(inventory_id)
        report = self.generate_report(inventory_id, ReportFormat.JSON)

        export: Dict[str, Any] = {
            "metadata": {
                "report_id": report.id,
                "inventory_id": inventory_id,
                "org_id": inventory.org_id,
                "year": inventory.year,
                "generated_at": report.generated_at.isoformat(),
                "format": "json",
                "provenance_hash": report.provenance_hash,
            },
            "sections": {},
        }

        for section in report.sections:
            export["sections"][section.key] = {
                "title": section.title,
                "content": section.content,
            }

        return export

    def export_csv(self, inventory_id: str) -> str:
        """
        Export inventory as CSV string.

        Generates a flat table with scope, category, gas breakdown.

        Args:
            inventory_id: Inventory ID.

        Returns:
            CSV string.
        """
        inventory = self._get_inventory_or_raise(inventory_id)

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Scope",
            "Category",
            "Total_tCO2e",
            "CO2_tCO2e",
            "CH4_tCO2e",
            "N2O_tCO2e",
            "HFCs_tCO2e",
            "PFCs_tCO2e",
            "SF6_tCO2e",
            "NF3_tCO2e",
            "Biogenic_CO2",
            "Data_Quality_Tier",
        ])

        # Scope 1
        self._write_scope_csv(writer, inventory.scope1, "Scope 1")

        # Scope 2 Location
        self._write_scope_csv(writer, inventory.scope2_location, "Scope 2 (Location)")

        # Scope 2 Market
        self._write_scope_csv(writer, inventory.scope2_market, "Scope 2 (Market)")

        # Scope 3
        self._write_scope_csv(writer, inventory.scope3, "Scope 3")

        # Grand total row
        writer.writerow([
            "TOTAL",
            "All Scopes",
            str(inventory.grand_total_tco2e),
            "", "", "", "", "", "", "",
            "",
            "",
        ])

        return output.getvalue()

    def export_excel(self, inventory_id: str) -> Dict[str, Any]:
        """
        Export inventory as an Excel-compatible structure.

        Returns a dict describing multi-sheet workbook content
        (actual .xlsx generation would use openpyxl).

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with sheet definitions.
        """
        inventory = self._get_inventory_or_raise(inventory_id)

        return {
            "filename": f"ghg_inventory_{inventory.org_id}_{inventory.year}.xlsx",
            "sheets": {
                "Summary": self._excel_summary_sheet(inventory),
                "Scope_1": self._excel_scope_sheet(inventory.scope1, "Scope 1"),
                "Scope_2": self._excel_scope2_sheet(inventory),
                "Scope_3": self._excel_scope_sheet(inventory.scope3, "Scope 3"),
                "Gas_Breakdown": self._excel_gas_sheet(inventory),
                "Intensity_Metrics": self._excel_intensity_sheet(inventory),
                "Data_Quality": self._excel_quality_sheet(inventory),
            },
        }

    def export_pdf(self, inventory_id: str) -> Dict[str, Any]:
        """
        Export inventory as PDF-compatible structure.

        Returns a dict describing the document layout for a PDF
        renderer (actual PDF generation would use reportlab or weasyprint).

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with PDF document structure.
        """
        inventory = self._get_inventory_or_raise(inventory_id)
        report = self.generate_report(inventory_id, ReportFormat.PDF)

        return {
            "filename": f"ghg_report_{inventory.org_id}_{inventory.year}.pdf",
            "title": f"GHG Inventory Report - {inventory.year}",
            "subtitle": f"Organization: {inventory.org_id}",
            "generated_at": report.generated_at.isoformat(),
            "page_size": "A4",
            "sections": [
                {
                    "title": section.title,
                    "content_type": "structured",
                    "data": section.content,
                }
                for section in report.sections
            ],
            "footer": {
                "text": f"Generated by GL-GHG-APP v{self.config.app_version}",
                "provenance_hash": report.provenance_hash,
            },
        }

    def get_report_history(self, inventory_id: str) -> List[Report]:
        """
        Get all previously generated reports for an inventory.

        Args:
            inventory_id: Inventory ID.

        Returns:
            List of Report objects.
        """
        return self._report_history.get(inventory_id, [])

    # ------------------------------------------------------------------
    # Section Content Generators
    # ------------------------------------------------------------------

    def _generate_section(
        self,
        section_key: str,
        inventory: GHGInventory,
    ) -> Dict[str, Any]:
        """Route section generation to the appropriate method."""
        generators = {
            "executive_summary": self._generate_executive_summary_content,
            "organizational_boundary": self._generate_org_boundary_content,
            "operational_boundary": self._generate_op_boundary_content,
            "base_year": self._generate_base_year_content,
            "scope1_emissions": self._generate_scope1_content,
            "scope2_emissions": self._generate_scope2_content,
            "scope3_emissions": self._generate_scope3_content,
            "total_emissions": self._generate_total_content,
            "intensity_metrics": self._generate_intensity_content,
            "trend_analysis": self._generate_trend_content,
            "uncertainty_assessment": self._generate_uncertainty_content,
            "data_quality": self._generate_quality_content,
            "verification_statement": self._generate_verification_content,
            "methodology_notes": self._generate_methodology_content,
            "appendices": self._generate_appendix_content,
        }

        generator = generators.get(section_key)
        if generator is None:
            return {"error": f"Unknown section: {section_key}"}
        return generator(inventory)

    def _generate_executive_summary_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build executive summary content."""
        s1 = inv.scope1.total_tco2e if inv.scope1 else Decimal("0")
        s2l = inv.scope2_location.total_tco2e if inv.scope2_location else Decimal("0")
        s2m = inv.scope2_market.total_tco2e if inv.scope2_market else Decimal("0")
        s3 = inv.scope3.total_tco2e if inv.scope3 else Decimal("0")

        total = inv.grand_total_tco2e
        scope_shares = {}
        if total > 0:
            scope_shares = {
                "scope1_pct": str((s1 / total * 100).quantize(Decimal("0.1"))),
                "scope2_market_pct": str((s2m / total * 100).quantize(Decimal("0.1"))),
                "scope3_pct": str((s3 / total * 100).quantize(Decimal("0.1"))),
            }

        return {
            "reporting_year": inv.year,
            "org_id": inv.org_id,
            "total_emissions_tco2e": str(total),
            "scope1_tco2e": str(s1),
            "scope2_location_tco2e": str(s2l),
            "scope2_market_tco2e": str(s2m),
            "scope3_tco2e": str(s3),
            "scope_shares": scope_shares,
            "status": inv.status,
            "data_quality_score": str(inv.data_quality_score),
            "intensity_metrics_count": len(inv.intensity_metrics),
            "verification_status": (
                inv.verification.status.value if inv.verification else "none"
            ),
        }

    def _generate_org_boundary_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build organizational boundary content."""
        if inv.boundary is None:
            return {"message": "Organizational boundary not defined"}

        return {
            "consolidation_approach": inv.boundary.consolidation_approach.value,
            "entity_count": len(inv.boundary.entity_ids),
            "entity_ids": inv.boundary.entity_ids,
            "exclusion_count": len(inv.boundary.exclusions),
        }

    def _generate_op_boundary_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build operational boundary content."""
        if inv.boundary is None:
            return {"message": "Operational boundary not defined"}

        return {
            "scopes_included": [s.value for s in inv.boundary.scopes],
            "scope1_included": Scope.SCOPE_1 in inv.boundary.scopes,
            "scope2_location_included": Scope.SCOPE_2_LOCATION in inv.boundary.scopes,
            "scope2_market_included": Scope.SCOPE_2_MARKET in inv.boundary.scopes,
            "scope3_included": Scope.SCOPE_3 in inv.boundary.scopes,
        }

    def _generate_base_year_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build base year content."""
        if inv.boundary is None or inv.boundary.base_year is None:
            return {"message": "Base year not defined"}

        return {
            "base_year": inv.boundary.base_year,
            "reporting_year": inv.year,
            "recalculation_policy": (
                "Base year is recalculated when structural changes "
                "exceed the significance threshold (5% of base year emissions)."
            ),
        }

    def _generate_scope1_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build Scope 1 emissions content."""
        if inv.scope1 is None:
            return {"message": "Scope 1 not calculated"}

        return {
            "total_tco2e": str(inv.scope1.total_tco2e),
            "by_category": {k: str(v) for k, v in inv.scope1.by_category.items()},
            "by_gas": {k: str(v) for k, v in inv.scope1.by_gas.items()},
            "biogenic_co2": str(inv.scope1.biogenic_co2),
            "data_quality_tier": inv.scope1.data_quality_tier.value,
            "methodology": inv.scope1.methodology_notes or "",
        }

    def _generate_scope2_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build Scope 2 emissions content (dual reporting)."""
        loc = inv.scope2_location
        mkt = inv.scope2_market

        content: Dict[str, Any] = {
            "dual_reporting_note": (
                "Per GHG Protocol Scope 2 Guidance (2015), both location-based "
                "and market-based results are reported."
            ),
        }

        if loc:
            content["location_based"] = {
                "total_tco2e": str(loc.total_tco2e),
                "by_category": {k: str(v) for k, v in loc.by_category.items()},
                "methodology": loc.methodology_notes or "",
            }

        if mkt:
            content["market_based"] = {
                "total_tco2e": str(mkt.total_tco2e),
                "by_category": {k: str(v) for k, v in mkt.by_category.items()},
                "methodology": mkt.methodology_notes or "",
            }

        if loc and mkt:
            delta = mkt.total_tco2e - loc.total_tco2e
            content["comparison"] = {
                "delta_tco2e": str(delta),
                "market_lower": mkt.total_tco2e < loc.total_tco2e,
            }

        return content

    def _generate_scope3_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build Scope 3 emissions content."""
        if inv.scope3 is None:
            return {"message": "Scope 3 not calculated"}

        categories = {}
        for cat_name, cat_value in sorted(
            inv.scope3.by_category.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            pct = Decimal("0")
            if inv.scope3.total_tco2e > 0:
                pct = (cat_value / inv.scope3.total_tco2e * 100).quantize(Decimal("0.1"))
            categories[cat_name] = {
                "tco2e": str(cat_value),
                "pct_of_scope3": str(pct),
            }

        return {
            "total_tco2e": str(inv.scope3.total_tco2e),
            "categories_reported": len(inv.scope3.by_category),
            "by_category": categories,
            "methodology": inv.scope3.methodology_notes or "",
        }

    def _generate_total_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build total emissions content."""
        return {
            "grand_total_tco2e_market": str(inv.grand_total_tco2e),
            "grand_total_tco2e_location": str(inv.grand_total_location_tco2e),
            "scope1_tco2e": str(inv.scope1.total_tco2e if inv.scope1 else 0),
            "scope2_market_tco2e": str(inv.scope2_market.total_tco2e if inv.scope2_market else 0),
            "scope2_location_tco2e": str(inv.scope2_location.total_tco2e if inv.scope2_location else 0),
            "scope3_tco2e": str(inv.scope3.total_tco2e if inv.scope3 else 0),
            "note": (
                "Grand total uses Scope 2 market-based as primary. "
                "Location-based total provided for reference."
            ),
        }

    def _generate_intensity_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build intensity metrics content."""
        if not inv.intensity_metrics:
            return {"message": "No intensity metrics calculated"}

        metrics = []
        for m in inv.intensity_metrics:
            metrics.append({
                "type": m.denominator.value,
                "value": str(m.intensity_value),
                "unit": m.unit,
                "scope": m.scope.value if m.scope else "all",
            })

        return {
            "metric_count": len(metrics),
            "metrics": metrics,
        }

    def _generate_trend_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build trend analysis content (single year only)."""
        return {
            "year": inv.year,
            "total_tco2e": str(inv.grand_total_tco2e),
            "note": "Full trend analysis requires multiple reporting years.",
        }

    def _generate_uncertainty_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build uncertainty assessment content."""
        if inv.uncertainty is None:
            return {"message": "Uncertainty assessment not performed"}

        u = inv.uncertainty
        return {
            "method": "Monte Carlo simulation",
            "iterations": u.iterations,
            "mean_tco2e": str(u.mean),
            "p5_tco2e": str(u.p5),
            "p50_tco2e": str(u.p50),
            "p95_tco2e": str(u.p95),
            "std_dev": str(u.std_dev),
            "cv_pct": str(u.cv),
            "confidence_level": str(u.confidence_level),
        }

    def _generate_quality_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build data quality content."""
        return {
            "data_quality_score": str(inv.data_quality_score),
            "completeness": (
                {
                    "overall_pct": str(inv.completeness.overall_pct),
                    "gap_count": len(inv.completeness.gaps),
                }
                if inv.completeness
                else None
            ),
        }

    def _generate_verification_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build verification statement content."""
        if inv.verification is None:
            return {"message": "No verification performed"}

        v = inv.verification
        return {
            "level": v.level.value,
            "status": v.status.value,
            "verifier": v.verifier_name or "Internal",
            "organization": v.verifier_organization or "",
            "findings_count": len(v.findings),
            "open_findings": v.open_findings_count,
            "statement": v.statement or "",
        }

    def _generate_methodology_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build methodology notes content."""
        notes: Dict[str, str] = {}
        for scope_name, scope_data in [
            ("scope1", inv.scope1),
            ("scope2_location", inv.scope2_location),
            ("scope2_market", inv.scope2_market),
            ("scope3", inv.scope3),
        ]:
            if scope_data and scope_data.methodology_notes:
                notes[scope_name] = scope_data.methodology_notes

        return {
            "gwp_source": "IPCC AR5 (100-year GWP)",
            "notes_by_scope": notes,
            "general_approach": (
                "Emissions calculated per GHG Protocol Corporate "
                "Accounting and Reporting Standard using activity data "
                "and published emission factors."
            ),
        }

    def _generate_appendix_content(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build appendix content."""
        return {
            "inventory_id": inv.id,
            "org_id": inv.org_id,
            "provenance_hash": inv.provenance_hash,
            "created_at": inv.created_at.isoformat(),
            "updated_at": inv.updated_at.isoformat(),
        }

    # ------------------------------------------------------------------
    # CSV Helper
    # ------------------------------------------------------------------

    def _write_scope_csv(
        self,
        writer: Any,
        scope_data: Any,
        scope_label: str,
    ) -> None:
        """Write scope rows to CSV writer."""
        if scope_data is None:
            return

        if scope_data.by_category:
            for cat_name, cat_value in scope_data.by_category.items():
                row = [scope_label, cat_name, str(cat_value)]
                # Gas columns
                for gas in [
                    GHGGas.CO2, GHGGas.CH4, GHGGas.N2O,
                    GHGGas.HFCS, GHGGas.PFCS, GHGGas.SF6, GHGGas.NF3,
                ]:
                    row.append("")  # Per-category gas breakdown not available here
                row.append(str(scope_data.biogenic_co2))
                row.append(scope_data.data_quality_tier.value)
                writer.writerow(row)
        else:
            # Single total row
            row = [scope_label, "Total", str(scope_data.total_tco2e)]
            for gas in GHGGas:
                row.append(str(scope_data.by_gas.get(gas.value, Decimal("0"))))
            row.append(str(scope_data.biogenic_co2))
            row.append(scope_data.data_quality_tier.value)
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Excel Sheet Builders
    # ------------------------------------------------------------------

    def _excel_summary_sheet(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build summary sheet structure."""
        return {
            "headers": ["Metric", "Value", "Unit"],
            "rows": [
                ["Reporting Year", str(inv.year), ""],
                ["Total Emissions (Market)", str(inv.grand_total_tco2e), "tCO2e"],
                ["Total Emissions (Location)", str(inv.grand_total_location_tco2e), "tCO2e"],
                ["Scope 1", str(inv.scope1.total_tco2e if inv.scope1 else 0), "tCO2e"],
                ["Scope 2 (Location)", str(inv.scope2_location.total_tco2e if inv.scope2_location else 0), "tCO2e"],
                ["Scope 2 (Market)", str(inv.scope2_market.total_tco2e if inv.scope2_market else 0), "tCO2e"],
                ["Scope 3", str(inv.scope3.total_tco2e if inv.scope3 else 0), "tCO2e"],
                ["Data Quality Score", str(inv.data_quality_score), "/100"],
                ["Status", inv.status, ""],
            ],
        }

    def _excel_scope_sheet(self, scope_data: Any, label: str) -> Dict[str, Any]:
        """Build scope detail sheet structure."""
        if scope_data is None:
            return {"headers": ["Category", "tCO2e"], "rows": []}

        rows = []
        for cat_name, cat_value in scope_data.by_category.items():
            rows.append([cat_name, str(cat_value)])
        rows.append(["TOTAL", str(scope_data.total_tco2e)])

        return {
            "headers": ["Category", "tCO2e"],
            "rows": rows,
        }

    def _excel_scope2_sheet(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build Scope 2 comparison sheet."""
        rows = []
        if inv.scope2_location:
            for cat, val in inv.scope2_location.by_category.items():
                mkt_val = inv.scope2_market.by_category.get(cat, Decimal("0")) if inv.scope2_market else Decimal("0")
                rows.append([cat, str(val), str(mkt_val)])

        return {
            "headers": ["Category", "Location-Based (tCO2e)", "Market-Based (tCO2e)"],
            "rows": rows,
        }

    def _excel_gas_sheet(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build gas breakdown sheet."""
        gas_names = [g.value for g in GHGGas]
        headers = ["Scope"] + gas_names + ["Total"]
        rows = []

        for scope_label, scope_data in [
            ("Scope 1", inv.scope1),
            ("Scope 2 (Location)", inv.scope2_location),
            ("Scope 3", inv.scope3),
        ]:
            if scope_data is None:
                continue
            row = [scope_label]
            for gas in GHGGas:
                row.append(str(scope_data.by_gas.get(gas.value, Decimal("0"))))
            row.append(str(scope_data.total_tco2e))
            rows.append(row)

        return {"headers": headers, "rows": rows}

    def _excel_intensity_sheet(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build intensity metrics sheet."""
        headers = ["Type", "Value", "Unit", "Scope"]
        rows = []
        for m in inv.intensity_metrics:
            rows.append([
                m.denominator.value,
                str(m.intensity_value),
                m.unit,
                m.scope.value if m.scope else "all",
            ])
        return {"headers": headers, "rows": rows}

    def _excel_quality_sheet(self, inv: GHGInventory) -> Dict[str, Any]:
        """Build data quality sheet."""
        return {
            "headers": ["Metric", "Value"],
            "rows": [
                ["Data Quality Score", str(inv.data_quality_score)],
                ["Status", inv.status],
                ["Completeness", str(inv.completeness.overall_pct) if inv.completeness else "N/A"],
            ],
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_inventory_or_raise(self, inventory_id: str) -> GHGInventory:
        """Retrieve inventory from store or raise ValueError."""
        inventory = self._inventory_store.get(inventory_id)
        if inventory is None:
            raise ValueError(f"Inventory not found: {inventory_id}")
        return inventory
