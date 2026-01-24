"""
CBAM XML Generator

Generates XML output conforming to EU CBAM Registry schema.
Implements the transitional period report format.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom

from cbam_pack.calculators.emissions_calculator import CalculationResult
from cbam_pack.models import CBAMConfig, EmissionsResult, AggregatedResult


class CBAMXMLGenerator:
    """
    Generates CBAM-compliant XML reports.

    Produces XML conforming to the EU CBAM Transitional Registry schema.
    """

    CBAM_NS = "urn:cbam:transitional:v1"
    XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

    def __init__(self):
        """Initialize the XML generator."""
        self.generated_at: Optional[datetime] = None

    def generate(
        self,
        calc_result: CalculationResult,
        config: CBAMConfig,
    ) -> str:
        """
        Generate CBAM XML report.

        Args:
            calc_result: Calculation results
            config: CBAM configuration

        Returns:
            XML string
        """
        self.generated_at = datetime.utcnow()

        # Create root element
        root = ET.Element("CBAMReport")
        root.set("xmlns", self.CBAM_NS)
        root.set("xmlns:xsi", self.XSI_NS)
        root.set("version", "1.0")

        # Add header
        self._add_header(root, config)

        # Add declarant info
        self._add_declarant(root, config)

        # Add reporting period
        self._add_reporting_period(root, config)

        # Add goods section
        self._add_goods(root, calc_result, config)

        # Add summary
        self._add_summary(root, calc_result)

        # Convert to pretty-printed string
        xml_string = ET.tostring(root, encoding="unicode")
        dom = minidom.parseString(xml_string)
        return dom.toprettyxml(indent="  ", encoding=None)

    def _add_header(self, root: ET.Element, config: CBAMConfig) -> None:
        """Add report header."""
        header = ET.SubElement(root, "Header")

        ET.SubElement(header, "GeneratedAt").text = self.generated_at.isoformat() + "Z"
        ET.SubElement(header, "ReportType").text = "TRANSITIONAL_QUARTERLY"
        ET.SubElement(header, "SchemaVersion").text = "1.0"

    def _add_declarant(self, root: ET.Element, config: CBAMConfig) -> None:
        """Add declarant information."""
        declarant = ET.SubElement(root, "Declarant")

        ET.SubElement(declarant, "Name").text = config.declarant.name
        ET.SubElement(declarant, "EORINumber").text = config.declarant.eori_number

        address = ET.SubElement(declarant, "Address")
        ET.SubElement(address, "Street").text = config.declarant.address.street
        ET.SubElement(address, "City").text = config.declarant.address.city
        ET.SubElement(address, "PostalCode").text = config.declarant.address.postal_code
        ET.SubElement(address, "Country").text = config.declarant.address.country

        contact = ET.SubElement(declarant, "Contact")
        ET.SubElement(contact, "Name").text = config.declarant.contact.name
        ET.SubElement(contact, "Email").text = config.declarant.contact.email
        if config.declarant.contact.phone:
            ET.SubElement(contact, "Phone").text = config.declarant.contact.phone

    def _add_reporting_period(self, root: ET.Element, config: CBAMConfig) -> None:
        """Add reporting period information."""
        period = ET.SubElement(root, "ReportingPeriod")

        ET.SubElement(period, "Quarter").text = config.reporting_period.quarter.value
        ET.SubElement(period, "Year").text = str(config.reporting_period.year)

        # Calculate period dates
        quarter_num = int(config.reporting_period.quarter.value[1])
        start_month = (quarter_num - 1) * 3 + 1
        end_month = quarter_num * 3

        year = config.reporting_period.year
        start_date = f"{year}-{start_month:02d}-01"

        # End date is last day of end_month
        if end_month == 12:
            end_date = f"{year}-12-31"
        else:
            next_month_start = datetime(year, end_month + 1, 1)
            from datetime import timedelta
            last_day = next_month_start - timedelta(days=1)
            end_date = last_day.strftime("%Y-%m-%d")

        ET.SubElement(period, "StartDate").text = start_date
        ET.SubElement(period, "EndDate").text = end_date

    def _add_goods(
        self,
        root: ET.Element,
        calc_result: CalculationResult,
        config: CBAMConfig,
    ) -> None:
        """Add imported goods section."""
        goods = ET.SubElement(root, "ImportedGoods")

        # Add aggregated results if available
        if calc_result.aggregated_results:
            for agg in calc_result.aggregated_results:
                self._add_aggregated_good(goods, agg)
        else:
            # Add line-level results
            for line_result in calc_result.line_results:
                self._add_line_result(goods, line_result)

    def _add_aggregated_good(
        self,
        parent: ET.Element,
        agg: AggregatedResult,
    ) -> None:
        """Add an aggregated good entry."""
        good = ET.SubElement(parent, "Good")

        ET.SubElement(good, "CNCode").text = agg.cn_code
        ET.SubElement(good, "CountryOfOrigin").text = agg.country_of_origin
        ET.SubElement(good, "TotalQuantityTonnes").text = str(agg.total_quantity_tonnes)

        emissions = ET.SubElement(good, "Emissions")
        ET.SubElement(emissions, "DirectEmissionsTCO2e").text = str(
            agg.total_direct_emissions_tco2e
        )
        ET.SubElement(emissions, "IndirectEmissionsTCO2e").text = str(
            agg.total_indirect_emissions_tco2e
        )
        ET.SubElement(emissions, "TotalEmissionsTCO2e").text = str(
            agg.total_emissions_tco2e
        )
        ET.SubElement(emissions, "WeightedIntensity").text = str(agg.weighted_intensity)

        ET.SubElement(good, "MethodUsed").text = agg.method_used
        ET.SubElement(good, "LineCount").text = str(agg.line_count)

    def _add_line_result(
        self,
        parent: ET.Element,
        result: EmissionsResult,
    ) -> None:
        """Add a line-level result entry."""
        line = ET.SubElement(parent, "Line")

        ET.SubElement(line, "LineId").text = result.line_id

        emissions = ET.SubElement(line, "Emissions")
        ET.SubElement(emissions, "DirectEmissionsTCO2e").text = str(
            result.direct_emissions_tco2e
        )
        ET.SubElement(emissions, "IndirectEmissionsTCO2e").text = str(
            result.indirect_emissions_tco2e
        )
        ET.SubElement(emissions, "TotalEmissionsTCO2e").text = str(
            result.total_emissions_tco2e
        )
        ET.SubElement(emissions, "Intensity").text = str(result.emissions_intensity)

        methods = ET.SubElement(line, "Methods")
        ET.SubElement(methods, "DirectMethod").text = result.method_direct.value
        ET.SubElement(methods, "IndirectMethod").text = result.method_indirect.value

        refs = ET.SubElement(line, "FactorReferences")
        ET.SubElement(refs, "DirectFactorRef").text = result.factor_direct_ref
        ET.SubElement(refs, "IndirectFactorRef").text = result.factor_indirect_ref

    def _add_summary(
        self,
        root: ET.Element,
        calc_result: CalculationResult,
    ) -> None:
        """Add summary section."""
        summary = ET.SubElement(root, "Summary")

        stats = calc_result.statistics

        ET.SubElement(summary, "TotalLines").text = str(stats.get("total_lines", 0))
        ET.SubElement(summary, "TotalDirectEmissionsTCO2e").text = str(
            stats.get("total_direct_emissions_tco2e", 0)
        )
        ET.SubElement(summary, "TotalIndirectEmissionsTCO2e").text = str(
            stats.get("total_indirect_emissions_tco2e", 0)
        )
        ET.SubElement(summary, "TotalEmissionsTCO2e").text = str(
            stats.get("total_emissions_tco2e", 0)
        )

        methods = ET.SubElement(summary, "MethodBreakdown")
        ET.SubElement(methods, "LinesWithSupplierDirectData").text = str(
            stats.get("lines_with_supplier_direct_data", 0)
        )
        ET.SubElement(methods, "LinesWithSupplierIndirectData").text = str(
            stats.get("lines_with_supplier_indirect_data", 0)
        )
        ET.SubElement(methods, "LinesUsingDefaults").text = str(
            stats.get("lines_using_defaults", 0)
        )
        ET.SubElement(methods, "DefaultUsagePercent").text = str(
            stats.get("default_usage_percent", 0)
        )

        # Add assumptions
        if calc_result.assumptions:
            assumptions = ET.SubElement(summary, "Assumptions")
            for assumption in calc_result.assumptions:
                assump_elem = ET.SubElement(assumptions, "Assumption")
                ET.SubElement(assump_elem, "Type").text = assumption.type.value
                ET.SubElement(assump_elem, "Description").text = assumption.description
                ET.SubElement(assump_elem, "Rationale").text = assumption.rationale
                ET.SubElement(assump_elem, "AppliesTo").text = ",".join(
                    assumption.applies_to[:10]  # Limit to first 10 for brevity
                )
                if assumption.factor_ref:
                    ET.SubElement(assump_elem, "FactorRef").text = assumption.factor_ref
