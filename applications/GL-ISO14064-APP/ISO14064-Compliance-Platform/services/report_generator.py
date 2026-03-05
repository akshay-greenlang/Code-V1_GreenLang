"""
Report Generator -- ISO 14064-1:2018 Clause 8 Implementation

Generates ISO 14064-1 compliant GHG inventory reports with all 14 mandatory
elements per Clause 8.2.  Supports multiple export formats (JSON, CSV,
Excel structure, PDF structure).

The 14 mandatory reporting elements (Clause 8.2):
  MRE-01: Reporting organization description
  MRE-02: Responsible person
  MRE-03: Reporting period
  MRE-04: Organizational boundary and consolidation approach
  MRE-05: Direct GHG emissions (Category 1)
  MRE-06: Indirect GHG emissions from imported energy (Category 2)
  MRE-07: Quantification methodology description
  MRE-08: GHG emissions and removals by gas type
  MRE-09: Emission factors and GWP values used
  MRE-10: Biogenic CO2 emissions reported separately
  MRE-11: Base year and recalculation policy
  MRE-12: Significance assessment for indirect categories (3-6)
  MRE-13: Exclusions with justification
  MRE-14: Uncertainty assessment

Additional reporting (Clause 8.3):
  - GHG management plan
  - Per-gas breakdown
  - Biogenic CO2 details
  - Changes from previous period

Example:
    >>> gen = ReportGenerator(config)
    >>> report = gen.generate_report("inv-1", ReportFormat.JSON)
    >>> print(f"Compliance: {report.mandatory_compliance_pct}%")
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
    GHGGas,
    ISO14064AppConfig,
    ISOCategory,
    ISO_CATEGORY_NAMES,
    MANDATORY_REPORTING_ELEMENTS,
    ReportFormat,
    SignificanceLevel,
)
from .models import (
    CategoryResult,
    ISOInventory,
    MandatoryElement,
    Organization,
    Report,
    ReportSection,
    SignificanceAssessment,
    UncertaintyResult,
    VerificationRecord,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report Section Definitions (ordered per ISO 14064-1 Clause 8)
# ---------------------------------------------------------------------------

REPORT_SECTIONS: List[str] = [
    "reporting_organization",       # MRE-01
    "responsible_person",           # MRE-02
    "reporting_period",             # MRE-03
    "organizational_boundary",      # MRE-04
    "quantification_methodology",   # MRE-07
    "category_1_direct",            # MRE-05
    "category_2_energy_indirect",   # MRE-06
    "category_3_transport",
    "category_4_products_used",
    "category_5_products_from_org",
    "category_6_other",
    "total_emissions",
    "gas_breakdown",                # MRE-08
    "emission_factors_gwp",         # MRE-09
    "biogenic_co2",                 # MRE-10
    "base_year",                    # MRE-11
    "significance_assessment",      # MRE-12
    "exclusions",                   # MRE-13
    "uncertainty_assessment",       # MRE-14
    "changes_from_previous",
    "management_plan",
    "verification_statement",
    "appendices",
]

SECTION_TITLES: Dict[str, str] = {
    "reporting_organization": "1. Reporting Organization",
    "responsible_person": "2. Responsible Person",
    "reporting_period": "3. Reporting Period",
    "organizational_boundary": "4. Organizational Boundary",
    "quantification_methodology": "5. Quantification Methodology",
    "category_1_direct": "6. Category 1 -- Direct GHG Emissions and Removals",
    "category_2_energy_indirect": "7. Category 2 -- Energy Indirect Emissions",
    "category_3_transport": "8. Category 3 -- Transportation Indirect Emissions",
    "category_4_products_used": "9. Category 4 -- Product Input Indirect Emissions",
    "category_5_products_from_org": "10. Category 5 -- Product Output Indirect Emissions",
    "category_6_other": "11. Category 6 -- Other Indirect Emissions",
    "total_emissions": "12. Total GHG Emissions and Removals",
    "gas_breakdown": "13. Emissions by Greenhouse Gas",
    "emission_factors_gwp": "14. Emission Factors and GWP Values",
    "biogenic_co2": "15. Biogenic CO2 Emissions",
    "base_year": "16. Base Year and Recalculation Policy",
    "significance_assessment": "17. Significance Assessment",
    "exclusions": "18. Exclusions and Justifications",
    "uncertainty_assessment": "19. Uncertainty Assessment",
    "changes_from_previous": "20. Changes from Previous Reporting Period",
    "management_plan": "21. GHG Management Plan",
    "verification_statement": "22. Verification Statement",
    "appendices": "23. Appendices",
}

# Map section keys to the mandatory element IDs they satisfy
SECTION_TO_MRE: Dict[str, str] = {
    "reporting_organization": "MRE-01",
    "responsible_person": "MRE-02",
    "reporting_period": "MRE-03",
    "organizational_boundary": "MRE-04",
    "category_1_direct": "MRE-05",
    "category_2_energy_indirect": "MRE-06",
    "quantification_methodology": "MRE-07",
    "gas_breakdown": "MRE-08",
    "emission_factors_gwp": "MRE-09",
    "biogenic_co2": "MRE-10",
    "base_year": "MRE-11",
    "significance_assessment": "MRE-12",
    "exclusions": "MRE-13",
    "uncertainty_assessment": "MRE-14",
}

# Mandatory element definitions per Clause 9
_MRE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "MRE-01": {"name": "Reporting organization description", "clause": "9.3.1"},
    "MRE-02": {"name": "Responsible person", "clause": "9.3.2"},
    "MRE-03": {"name": "Reporting period", "clause": "9.3.3"},
    "MRE-04": {"name": "Organizational boundary", "clause": "9.3.4"},
    "MRE-05": {"name": "Direct GHG emissions (Category 1)", "clause": "9.3.5"},
    "MRE-06": {"name": "Indirect energy emissions (Category 2)", "clause": "9.3.6"},
    "MRE-07": {"name": "Quantification methodology", "clause": "9.3.7"},
    "MRE-08": {"name": "GHG emissions by gas type", "clause": "9.3.8"},
    "MRE-09": {"name": "Emission factors and GWP values", "clause": "9.3.9"},
    "MRE-10": {"name": "Biogenic CO2 reported separately", "clause": "9.3.10"},
    "MRE-11": {"name": "Base year and recalculation policy", "clause": "9.3.11"},
    "MRE-12": {"name": "Significance assessment", "clause": "9.3.12"},
    "MRE-13": {"name": "Exclusions with justification", "clause": "9.3.13"},
    "MRE-14": {"name": "Uncertainty assessment", "clause": "9.3.14"},
}


class ReportGenerator:
    """
    Generates ISO 14064-1 compliant GHG inventory reports.

    Assembles report sections from inventory data, checks for
    compliance with the 14 mandatory elements, and supports
    multi-format export.

    Attributes:
        config: Application configuration.
        _inventory_store: Shared reference to inventory storage.
        _org_store: Shared reference to organization storage.
        _report_history: In-memory report history.
    """

    def __init__(
        self,
        config: Optional[ISO14064AppConfig] = None,
        inventory_store: Optional[Dict[str, ISOInventory]] = None,
        org_store: Optional[Dict[str, Organization]] = None,
        category_store: Optional[Dict[str, Dict[str, CategoryResult]]] = None,
        significance_store: Optional[Dict[str, List[SignificanceAssessment]]] = None,
        uncertainty_store: Optional[Dict[str, UncertaintyResult]] = None,
        verification_store: Optional[Dict[str, VerificationRecord]] = None,
    ) -> None:
        """
        Initialize ReportGenerator.

        Args:
            config: Application configuration.
            inventory_store: Inventory storage keyed by ID.
            org_store: Organization storage keyed by ID.
            category_store: Category results keyed by inventory_id then cat value.
            significance_store: Significance assessments keyed by inventory_id.
            uncertainty_store: Uncertainty results keyed by inventory_id.
            verification_store: Verification records keyed by inventory_id.
        """
        self.config = config or ISO14064AppConfig()
        self._inventory_store = inventory_store or {}
        self._org_store = org_store or {}
        self._category_store = category_store or {}
        self._significance_store = significance_store or {}
        self._uncertainty_store = uncertainty_store or {}
        self._verification_store = verification_store or {}
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
        Generate a complete ISO 14064-1 inventory report.

        Args:
            inventory_id: Inventory ID.
            report_format: Output format.
            sections: Specific sections (None = all).

        Returns:
            Generated Report with mandatory element compliance check.
        """
        start = datetime.utcnow()
        inventory = self._get_inventory_or_raise(inventory_id)

        selected = sections or REPORT_SECTIONS
        report_sections: List[ReportSection] = []
        present_mres: List[str] = []

        for idx, section_key in enumerate(selected):
            if section_key not in REPORT_SECTIONS:
                logger.warning("Unknown section '%s', skipping", section_key)
                continue

            content = self._generate_section(section_key, inventory)
            is_mandatory = section_key in SECTION_TO_MRE
            has_content = bool(content) and "message" not in content

            if is_mandatory and has_content:
                mre_id = SECTION_TO_MRE[section_key]
                present_mres.append(mre_id)

            report_sections.append(
                ReportSection(
                    key=section_key,
                    title=SECTION_TITLES.get(section_key, section_key),
                    content=content,
                    order=idx,
                )
            )

        # Build MandatoryElement list
        mandatory_elements = self._build_mandatory_elements(present_mres)

        report = Report(
            inventory_id=inventory_id,
            org_id=inventory.org_id,
            reporting_year=inventory.year,
            format=report_format,
            sections=report_sections,
            mandatory_elements=mandatory_elements,
        )

        # Track in history
        if inventory_id not in self._report_history:
            self._report_history[inventory_id] = []
        self._report_history[inventory_id].append(report)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Generated %s report for inventory %s (%d sections, %.1f%% MRE compliance) in %.1f ms",
            report_format.value,
            inventory_id,
            len(report_sections),
            report.mandatory_compliance_pct,
            elapsed_ms,
        )
        return report

    def check_compliance(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """
        Check whether all 14 mandatory elements are present.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with compliance status and missing elements.
        """
        report = self.generate_report(inventory_id, ReportFormat.JSON)

        present_ids = {
            e.element_id for e in report.mandatory_elements if e.present
        }
        all_mres = set(MANDATORY_REPORTING_ELEMENTS)
        missing = all_mres - present_ids

        return {
            "inventory_id": inventory_id,
            "total_mandatory": len(all_mres),
            "present": len(present_ids),
            "missing_count": len(missing),
            "missing_elements": sorted(missing),
            "compliant": len(missing) == 0,
            "compliance_pct": str(report.mandatory_compliance_pct),
        }

    # ------------------------------------------------------------------
    # Export Methods
    # ------------------------------------------------------------------

    def export_json(self, inventory_id: str) -> Dict[str, Any]:
        """Export inventory as a structured JSON document."""
        inventory = self._get_inventory_or_raise(inventory_id)
        report = self.generate_report(inventory_id, ReportFormat.JSON)

        export: Dict[str, Any] = {
            "metadata": {
                "report_id": report.id,
                "inventory_id": inventory_id,
                "org_id": inventory.org_id,
                "year": inventory.year,
                "standard": "ISO 14064-1:2018",
                "generated_at": report.generated_at.isoformat(),
                "format": "json",
                "mandatory_compliance_pct": str(report.mandatory_compliance_pct),
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
        """Export inventory category totals as CSV string."""
        self._get_inventory_or_raise(inventory_id)
        cat_results = self._category_store.get(inventory_id, {})

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "ISO_Category",
            "Category_Name",
            "Total_tCO2e",
            "Biogenic_CO2",
            "Data_Quality_Tier",
            "Significance",
            "Source_Count",
        ])

        for cat in ISOCategory:
            cat_result = cat_results.get(cat.value)
            if cat_result:
                writer.writerow([
                    cat.value,
                    ISO_CATEGORY_NAMES.get(cat, cat.value),
                    str(cat_result.total_tco2e),
                    str(cat_result.biogenic_co2),
                    cat_result.data_quality_tier.value,
                    cat_result.significance.value,
                    len(cat_result.sources),
                ])
            else:
                writer.writerow([
                    cat.value,
                    ISO_CATEGORY_NAMES.get(cat, cat.value),
                    "0", "0", "", "", "0",
                ])

        return output.getvalue()

    def export_excel(self, inventory_id: str) -> Dict[str, Any]:
        """
        Export inventory as an Excel-compatible multi-sheet structure.

        Returns a dict describing workbook content for openpyxl rendering.
        """
        inventory = self._get_inventory_or_raise(inventory_id)
        cat_results = self._category_store.get(inventory_id, {})

        return {
            "filename": f"iso14064_inventory_{inventory.org_id}_{inventory.year}.xlsx",
            "sheets": {
                "Summary": self._excel_summary_sheet(inventory, cat_results),
                "Category_Detail": self._excel_category_sheet(cat_results),
                "Gas_Breakdown": self._excel_gas_sheet(cat_results),
                "Compliance_Check": self._excel_compliance_sheet(inventory_id),
            },
        }

    def export_pdf(self, inventory_id: str) -> Dict[str, Any]:
        """Export inventory as PDF-compatible document structure."""
        inventory = self._get_inventory_or_raise(inventory_id)
        report = self.generate_report(inventory_id, ReportFormat.PDF)

        return {
            "filename": f"iso14064_report_{inventory.org_id}_{inventory.year}.pdf",
            "title": f"ISO 14064-1:2018 GHG Inventory Report -- {inventory.year}",
            "subtitle": f"Organization: {inventory.org_id}",
            "standard": "ISO 14064-1:2018",
            "generated_at": report.generated_at.isoformat(),
            "page_size": "A4",
            "sections": [
                {
                    "title": section.title,
                    "data": section.content,
                }
                for section in report.sections
            ],
            "footer": {
                "text": f"Generated by GL-ISO14064-APP v{self.config.version}",
                "provenance_hash": report.provenance_hash,
            },
        }

    def get_report_history(self, inventory_id: str) -> List[Report]:
        """Get all previously generated reports for an inventory."""
        return self._report_history.get(inventory_id, [])

    # ------------------------------------------------------------------
    # Mandatory Element Builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_mandatory_elements(
        present_mre_ids: List[str],
    ) -> List[MandatoryElement]:
        """Build MandatoryElement list with presence tracking."""
        elements: List[MandatoryElement] = []
        present_set = set(present_mre_ids)

        for mre_id in MANDATORY_REPORTING_ELEMENTS:
            defn = _MRE_DEFINITIONS.get(mre_id, {})
            is_present = mre_id in present_set

            elements.append(MandatoryElement(
                element_id=mre_id,
                name=defn.get("name", mre_id),
                clause_reference=defn.get("clause", ""),
                required=True,
                present=is_present,
                evidence=f"Section populated" if is_present else None,
            ))

        return elements

    # ------------------------------------------------------------------
    # Section Content Generators
    # ------------------------------------------------------------------

    def _generate_section(
        self,
        section_key: str,
        inventory: ISOInventory,
    ) -> Dict[str, Any]:
        """Route to the appropriate section generator."""
        generators = {
            "reporting_organization": self._gen_reporting_org,
            "responsible_person": self._gen_responsible_person,
            "reporting_period": self._gen_reporting_period,
            "organizational_boundary": self._gen_org_boundary,
            "quantification_methodology": self._gen_methodology,
            "category_1_direct": lambda inv: self._gen_category(inv, ISOCategory.CATEGORY_1_DIRECT),
            "category_2_energy_indirect": lambda inv: self._gen_category(inv, ISOCategory.CATEGORY_2_ENERGY),
            "category_3_transport": lambda inv: self._gen_category(inv, ISOCategory.CATEGORY_3_TRANSPORT),
            "category_4_products_used": lambda inv: self._gen_category(inv, ISOCategory.CATEGORY_4_PRODUCTS_USED),
            "category_5_products_from_org": lambda inv: self._gen_category(inv, ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG),
            "category_6_other": lambda inv: self._gen_category(inv, ISOCategory.CATEGORY_6_OTHER),
            "total_emissions": self._gen_totals,
            "gas_breakdown": self._gen_gas_breakdown,
            "emission_factors_gwp": self._gen_ef_gwp,
            "biogenic_co2": self._gen_biogenic,
            "base_year": self._gen_base_year,
            "significance_assessment": self._gen_significance,
            "exclusions": self._gen_exclusions,
            "uncertainty_assessment": self._gen_uncertainty,
            "changes_from_previous": self._gen_changes,
            "management_plan": self._gen_management_plan,
            "verification_statement": self._gen_verification,
            "appendices": self._gen_appendices,
        }

        generator = generators.get(section_key)
        if generator is None:
            return {"error": f"Unknown section: {section_key}"}
        return generator(inventory)

    def _gen_reporting_org(self, inv: ISOInventory) -> Dict[str, Any]:
        """MRE-01: Reporting organization."""
        org = self._org_store.get(inv.org_id)
        if org is None:
            return {
                "org_id": inv.org_id,
                "note": "Organization details not available",
            }
        return {
            "org_id": org.id,
            "name": org.name,
            "industry": org.industry,
            "country": org.country,
            "description": org.description or "",
            "entity_count": len(org.entities),
        }

    def _gen_responsible_person(self, inv: ISOInventory) -> Dict[str, Any]:
        """MRE-02: Responsible person."""
        org = self._org_store.get(inv.org_id)
        contact = ""
        if org and org.contact_person:
            contact = org.contact_person
        return {
            "responsible_person": contact or "Designated GHG inventory manager",
            "org_id": inv.org_id,
        }

    def _gen_reporting_period(self, inv: ISOInventory) -> Dict[str, Any]:
        """MRE-03: Reporting period."""
        period_start = f"{inv.year}-01-01"
        period_end = f"{inv.year}-12-31"

        if inv.boundary:
            if inv.boundary.period_start:
                period_start = inv.boundary.period_start.isoformat()
            if inv.boundary.period_end:
                period_end = inv.boundary.period_end.isoformat()

        return {
            "reporting_year": inv.year,
            "period_start": period_start,
            "period_end": period_end,
        }

    def _gen_org_boundary(self, inv: ISOInventory) -> Dict[str, Any]:
        """MRE-04: Organizational boundary."""
        boundary_info: Dict[str, Any] = {
            "gwp_source": inv.gwp_source.value,
        }

        if inv.boundary:
            boundary_info["consolidation_approach"] = inv.boundary.consolidation_approach.value
            boundary_info["categories_included"] = [
                c.value for c in inv.boundary.categories_included
            ]
            boundary_info["entity_count"] = len(inv.boundary.entity_ids)
        else:
            boundary_info["note"] = "Boundary not yet configured"

        return boundary_info

    def _gen_methodology(self, inv: ISOInventory) -> Dict[str, Any]:
        """MRE-07: Quantification methodology."""
        return {
            "approach": "Calculation-based using activity data and emission factors",
            "standard": "ISO 14064-1:2018",
            "gwp_source": inv.gwp_source.value,
            "data_quality_management": "Per ISO 14064-1:2018 Clause 7",
        }

    def _gen_category(
        self,
        inv: ISOInventory,
        category: ISOCategory,
    ) -> Dict[str, Any]:
        """Generate content for a specific ISO category."""
        cat_results = self._category_store.get(inv.id, {})
        cat_result = cat_results.get(category.value)

        if cat_result is None:
            # Also try pulling from inventory object directly
            cat_attr = self._get_inventory_category(inv, category)
            if cat_attr is None:
                return {"message": f"{ISO_CATEGORY_NAMES.get(category, category.value)} not quantified"}
            cat_result = cat_attr

        gas_dict: Dict[str, str] = {}
        gb = cat_result.gas_breakdown
        if gb:
            gas_dict = {
                "CO2": str(gb.co2),
                "CH4": str(gb.ch4),
                "N2O": str(gb.n2o),
                "HFCs": str(gb.hfcs),
                "PFCs": str(gb.pfcs),
                "SF6": str(gb.sf6),
                "NF3": str(gb.nf3),
            }

        return {
            "category": category.value,
            "category_name": ISO_CATEGORY_NAMES.get(category, category.value),
            "total_tco2e": str(cat_result.total_tco2e),
            "biogenic_co2": str(cat_result.biogenic_co2),
            "data_quality_tier": cat_result.data_quality_tier.value,
            "source_count": len(cat_result.sources),
            "gas_breakdown": gas_dict,
            "significance": cat_result.significance.value,
            "provenance_hash": cat_result.provenance_hash,
        }

    def _gen_totals(self, inv: ISOInventory) -> Dict[str, Any]:
        """Total emissions across all categories."""
        cat_results = self._category_store.get(inv.id, {})
        gross = Decimal("0")
        by_category: Dict[str, str] = {}

        for cat in ISOCategory:
            cr = cat_results.get(cat.value)
            if cr is None:
                cr = self._get_inventory_category(inv, cat)
            if cr:
                gross += cr.total_tco2e
                by_category[cat.value] = str(cr.total_tco2e)

        removals = inv.total_removals_tco2e
        net = gross - removals

        return {
            "gross_emissions_tco2e": str(gross),
            "total_removals_tco2e": str(removals),
            "net_emissions_tco2e": str(net),
            "by_category": by_category,
        }

    def _gen_gas_breakdown(self, inv: ISOInventory) -> Dict[str, Any]:
        """MRE-08: By-gas breakdown."""
        cat_results = self._category_store.get(inv.id, {})
        gas_totals: Dict[str, Decimal] = {
            "CO2": Decimal("0"),
            "CH4": Decimal("0"),
            "N2O": Decimal("0"),
            "HFCs": Decimal("0"),
            "PFCs": Decimal("0"),
            "SF6": Decimal("0"),
            "NF3": Decimal("0"),
        }

        for cr in cat_results.values():
            gb = cr.gas_breakdown
            if gb:
                gas_totals["CO2"] += gb.co2
                gas_totals["CH4"] += gb.ch4
                gas_totals["N2O"] += gb.n2o
                gas_totals["HFCs"] += gb.hfcs
                gas_totals["PFCs"] += gb.pfcs
                gas_totals["SF6"] += gb.sf6
                gas_totals["NF3"] += gb.nf3

        return {
            "by_gas": {k: str(v) for k, v in gas_totals.items()},
            "gases_reported": [k for k, v in gas_totals.items() if v > 0],
        }

    def _gen_ef_gwp(self, inv: ISOInventory) -> Dict[str, Any]:
        """MRE-09: Emission factors and GWP."""
        return {
            "gwp_source": inv.gwp_source.value,
            "gwp_time_horizon": "100-year",
            "note": "Emission factors sourced from IPCC, EPA, and sector-specific databases.",
        }

    def _gen_biogenic(self, inv: ISOInventory) -> Dict[str, Any]:
        """MRE-10: Biogenic CO2."""
        cat_results = self._category_store.get(inv.id, {})
        total_biogenic = Decimal("0")
        by_category: Dict[str, str] = {}

        for cat in ISOCategory:
            cr = cat_results.get(cat.value)
            if cr is None:
                cr = self._get_inventory_category(inv, cat)
            if cr and cr.biogenic_co2 > 0:
                total_biogenic += cr.biogenic_co2
                by_category[cat.value] = str(cr.biogenic_co2)

        # Also include inventory-level biogenic data
        if inv.biogenic and inv.biogenic.total_biogenic_co2 > 0:
            total_biogenic = max(total_biogenic, inv.biogenic.total_biogenic_co2)

        return {
            "total_biogenic_co2_tco2e": str(total_biogenic),
            "by_category": by_category,
            "note": "Biogenic CO2 is reported separately per ISO 14064-1:2018.",
        }

    def _gen_base_year(self, inv: ISOInventory) -> Dict[str, Any]:
        """MRE-11: Base year."""
        return {
            "base_year": "Refer to Base Year Manager configuration",
            "recalculation_policy": (
                "Per ISO 14064-1:2018 Clause 7.3, the base year is recalculated "
                "when structural changes, methodology changes, or error corrections "
                "exceed the significance threshold."
            ),
        }

    def _gen_significance(self, inv: ISOInventory) -> Dict[str, Any]:
        """MRE-12: Significance assessment."""
        # Try external store first, then inventory-embedded assessments
        assessments = self._significance_store.get(inv.id, [])
        if not assessments and inv.significance_assessments:
            assessments = inv.significance_assessments

        if not assessments:
            return {"message": "Significance assessment not yet performed"}

        results: List[Dict[str, Any]] = []
        for a in assessments:
            results.append({
                "category": a.iso_category.value,
                "category_name": ISO_CATEGORY_NAMES.get(a.iso_category, a.iso_category.value),
                "composite_score": str(a.criteria.composite_score) if a.criteria else "0",
                "threshold_pct": str(a.threshold_pct),
                "determination": a.result.value,
                "justification": a.justification,
            })

        return {
            "assessment_count": len(results),
            "assessments": results,
        }

    def _gen_exclusions(self, inv: ISOInventory) -> Dict[str, Any]:
        """MRE-13: Exclusions."""
        assessments = self._significance_store.get(inv.id, [])
        if not assessments and inv.significance_assessments:
            assessments = inv.significance_assessments

        exclusions = [
            a for a in assessments
            if a.result == SignificanceLevel.NOT_SIGNIFICANT
        ]

        if not exclusions:
            return {"exclusion_count": 0, "exclusions": []}

        return {
            "exclusion_count": len(exclusions),
            "exclusions": [
                {
                    "category": a.iso_category.value,
                    "category_name": ISO_CATEGORY_NAMES.get(a.iso_category, a.iso_category.value),
                    "justification": a.justification,
                }
                for a in exclusions
            ],
        }

    def _gen_uncertainty(self, inv: ISOInventory) -> Dict[str, Any]:
        """MRE-14: Uncertainty assessment."""
        unc = self._uncertainty_store.get(inv.id)
        if unc is None and inv.uncertainty:
            unc = inv.uncertainty

        if unc is None:
            return {"message": "Uncertainty assessment not performed"}

        return {
            "methodology": unc.methodology,
            "iterations": unc.iterations,
            "central_estimate_tco2e": str(unc.central_estimate),
            "lower_bound_tco2e": str(unc.lower_bound),
            "upper_bound_tco2e": str(unc.upper_bound),
            "confidence_level": unc.confidence_level,
            "std_dev": str(unc.std_dev),
            "cv_pct": str(unc.cv_pct),
            "by_category": {
                k: {ki: str(vi) for ki, vi in v.items()}
                for k, v in unc.by_category.items()
            },
            "sensitivity_ranking": unc.sensitivity_ranking,
        }

    def _gen_changes(self, inv: ISOInventory) -> Dict[str, Any]:
        """Changes from previous reporting period."""
        return {
            "note": "Refer to year-over-year analysis in Base Year Manager.",
        }

    def _gen_management_plan(self, inv: ISOInventory) -> Dict[str, Any]:
        """Management plan summary."""
        return {
            "note": "Refer to Management Plan Engine for action details.",
        }

    def _gen_verification(self, inv: ISOInventory) -> Dict[str, Any]:
        """Verification statement."""
        record = self._verification_store.get(inv.id)
        if record is None and inv.verification:
            record = inv.verification

        if record is None:
            return {"message": "No verification performed"}

        return {
            "stage": record.stage.value,
            "assurance_level": record.level.value,
            "verifier": record.verifier_name or "Internal",
            "organization": record.verifier_organization or "",
            "accreditation": record.verifier_accreditation or "",
            "findings_count": len(record.findings),
            "open_findings": record.open_findings_count,
            "has_critical_findings": record.has_critical_findings,
            "statement": record.statement or "",
            "opinion": record.opinion or "",
        }

    def _gen_appendices(self, inv: ISOInventory) -> Dict[str, Any]:
        """Appendix content."""
        return {
            "inventory_id": inv.id,
            "org_id": inv.org_id,
            "year": inv.year,
            "provenance_hash": inv.provenance_hash,
            "created_at": inv.created_at.isoformat(),
            "updated_at": inv.updated_at.isoformat(),
        }

    # ------------------------------------------------------------------
    # Excel Sheet Builders
    # ------------------------------------------------------------------

    def _excel_summary_sheet(
        self,
        inv: ISOInventory,
        cat_results: Dict[str, CategoryResult],
    ) -> Dict[str, Any]:
        """Build summary sheet."""
        gross = sum(
            (cr.total_tco2e for cr in cat_results.values()),
            Decimal("0"),
        )
        removals = inv.total_removals_tco2e
        return {
            "headers": ["Metric", "Value", "Unit"],
            "rows": [
                ["Reporting Year", str(inv.year), ""],
                ["Standard", "ISO 14064-1:2018", ""],
                ["Gross Emissions", str(gross), "tCO2e"],
                ["Total Removals", str(removals), "tCO2e"],
                ["Net Emissions", str(gross - removals), "tCO2e"],
                ["GWP Source", inv.gwp_source.value, ""],
                ["Status", inv.status.value, ""],
            ],
        }

    def _excel_category_sheet(
        self,
        cat_results: Dict[str, CategoryResult],
    ) -> Dict[str, Any]:
        """Build category detail sheet."""
        headers = ["Category", "Name", "Total_tCO2e", "Biogenic_CO2", "Tier", "Sources"]
        rows: List[List[str]] = []
        for cat in ISOCategory:
            cr = cat_results.get(cat.value)
            if cr:
                rows.append([
                    cat.value,
                    ISO_CATEGORY_NAMES.get(cat, cat.value),
                    str(cr.total_tco2e),
                    str(cr.biogenic_co2),
                    cr.data_quality_tier.value,
                    str(len(cr.sources)),
                ])
        return {"headers": headers, "rows": rows}

    def _excel_gas_sheet(
        self,
        cat_results: Dict[str, CategoryResult],
    ) -> Dict[str, Any]:
        """Build gas breakdown sheet."""
        gas_names = ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"]
        headers = ["Category"] + gas_names + ["Total"]
        rows: List[List[str]] = []

        for cat in ISOCategory:
            cr = cat_results.get(cat.value)
            if cr is None:
                continue
            gb = cr.gas_breakdown
            row = [cat.value]
            if gb:
                row.extend([
                    str(gb.co2), str(gb.ch4), str(gb.n2o),
                    str(gb.hfcs), str(gb.pfcs), str(gb.sf6), str(gb.nf3),
                ])
            else:
                row.extend(["0"] * 7)
            row.append(str(cr.total_tco2e))
            rows.append(row)

        return {"headers": headers, "rows": rows}

    def _excel_compliance_sheet(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """Build compliance check sheet."""
        compliance = self.check_compliance(inventory_id)
        headers = ["Element", "Name", "Clause", "Status"]
        rows: List[List[str]] = []

        for mre_id in MANDATORY_REPORTING_ELEMENTS:
            defn = _MRE_DEFINITIONS.get(mre_id, {})
            status = "Present" if mre_id not in compliance["missing_elements"] else "MISSING"
            rows.append([
                mre_id,
                defn.get("name", mre_id),
                defn.get("clause", ""),
                status,
            ])

        return {"headers": headers, "rows": rows}

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_inventory_or_raise(self, inventory_id: str) -> ISOInventory:
        """Retrieve inventory or raise ValueError."""
        inventory = self._inventory_store.get(inventory_id)
        if inventory is None:
            raise ValueError(f"Inventory not found: {inventory_id}")
        return inventory

    @staticmethod
    def _get_inventory_category(
        inv: ISOInventory,
        category: ISOCategory,
    ) -> Optional[CategoryResult]:
        """Extract a category result from the inventory object."""
        attr_map = {
            ISOCategory.CATEGORY_1_DIRECT: inv.category_1,
            ISOCategory.CATEGORY_2_ENERGY: inv.category_2,
            ISOCategory.CATEGORY_3_TRANSPORT: inv.category_3,
            ISOCategory.CATEGORY_4_PRODUCTS_USED: inv.category_4,
            ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG: inv.category_5,
            ISOCategory.CATEGORY_6_OTHER: inv.category_6,
        }
        return attr_map.get(category)
