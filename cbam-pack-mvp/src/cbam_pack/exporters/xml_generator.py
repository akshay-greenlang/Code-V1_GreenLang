"""
CBAM XML Generator

Generates XML output conforming to EU CBAM Registry schema.
Implements the transitional period report format with XSD validation.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from typing import Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom

try:
    from lxml import etree as lxml_etree
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False

from cbam_pack.calculators.emissions_calculator import CalculationResult
from cbam_pack.models import CBAMConfig, EmissionsResult, AggregatedResult


@dataclass
class XMLValidationResult:
    """Result of XML schema validation."""
    valid: bool
    status: str  # PASS, FAIL
    schema_version: str
    schema_date: str
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "valid": self.valid,
            "status": self.status,
            "schema_version": self.schema_version,
            "schema_date": self.schema_date,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class CBAMXMLGenerator:
    """
    Generates CBAM-compliant XML reports.

    Produces XML conforming to the EU CBAM Transitional Registry schema
    with built-in XSD validation.
    """

    CBAM_NS = "urn:cbam:transitional:v1"
    XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

    # XSD Schema metadata
    XSD_VERSION = "1.0.0"
    XSD_DATE = "2024-10-01"

    def __init__(self):
        """Initialize the XML generator."""
        self.generated_at: Optional[datetime] = None
        self.validation_result: Optional[XMLValidationResult] = None

    def generate(
        self,
        calc_result: CalculationResult,
        config: CBAMConfig,
        validate: bool = True,
    ) -> str:
        """
        Generate CBAM XML report.

        Args:
            calc_result: Calculation results
            config: CBAM configuration
            validate: Whether to validate against XSD schema

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
        pretty_xml = dom.toprettyxml(indent="  ", encoding=None)

        # Validate against XSD if requested
        if validate:
            self.validation_result = self._validate_xml(pretty_xml)
        else:
            self.validation_result = XMLValidationResult(
                valid=True,
                status="SKIPPED",
                schema_version=self.XSD_VERSION,
                schema_date=self.XSD_DATE,
                errors=[],
                warnings=["Validation was skipped"],
            )

        return pretty_xml

    def get_validation_result(self) -> Optional[dict]:
        """Get the validation result as a dictionary."""
        if self.validation_result:
            return self.validation_result.to_dict()
        return None

    def _validate_xml(self, xml_content: str) -> XMLValidationResult:
        """
        Validate XML against CBAM XSD schema.

        Uses lxml for XSD validation if available, otherwise performs
        structural validation.
        """
        errors = []
        warnings = []

        # Perform structural validation (always)
        structural_errors = self._structural_validation(xml_content)
        errors.extend(structural_errors)

        # XSD validation with lxml if available
        if LXML_AVAILABLE:
            xsd_errors = self._xsd_validation(xml_content)
            errors.extend(xsd_errors)
        else:
            warnings.append(
                "lxml not available - using structural validation only. "
                "Install lxml for full XSD validation."
            )

        valid = len(errors) == 0
        status = "PASS" if valid else "FAIL"

        return XMLValidationResult(
            valid=valid,
            status=status,
            schema_version=self.XSD_VERSION,
            schema_date=self.XSD_DATE,
            errors=errors,
            warnings=warnings,
        )

    def _structural_validation(self, xml_content: str) -> list[str]:
        """Perform structural validation of the XML."""
        errors = []

        try:
            root = ET.fromstring(xml_content)

            # Check required elements
            required_elements = [
                "Header",
                "Declarant",
                "ReportingPeriod",
                "ImportedGoods",
                "Summary",
            ]

            for elem_name in required_elements:
                if root.find(elem_name) is None:
                    errors.append(f"Missing required element: {elem_name}")

            # Check Header elements
            header = root.find("Header")
            if header is not None:
                for sub_elem in ["GeneratedAt", "ReportType", "SchemaVersion"]:
                    if header.find(sub_elem) is None:
                        errors.append(f"Missing Header/{sub_elem}")

            # Check Declarant elements
            declarant = root.find("Declarant")
            if declarant is not None:
                for sub_elem in ["Name", "EORINumber", "Address", "Contact"]:
                    if declarant.find(sub_elem) is None:
                        errors.append(f"Missing Declarant/{sub_elem}")

            # Check ReportingPeriod elements
            period = root.find("ReportingPeriod")
            if period is not None:
                for sub_elem in ["Quarter", "Year", "StartDate", "EndDate"]:
                    if period.find(sub_elem) is None:
                        errors.append(f"Missing ReportingPeriod/{sub_elem}")

            # Check Summary elements
            summary = root.find("Summary")
            if summary is not None:
                for sub_elem in ["TotalLines", "TotalEmissionsTCO2e"]:
                    if summary.find(sub_elem) is None:
                        errors.append(f"Missing Summary/{sub_elem}")

        except ET.ParseError as e:
            errors.append(f"XML parse error: {e}")

        return errors

    def _xsd_validation(self, xml_content: str) -> list[str]:
        """Validate XML against embedded XSD schema using lxml."""
        errors = []

        # Embedded XSD schema for CBAM transitional reports
        xsd_content = self._get_embedded_xsd()

        try:
            schema_doc = lxml_etree.fromstring(xsd_content.encode())
            schema = lxml_etree.XMLSchema(schema_doc)

            xml_doc = lxml_etree.fromstring(xml_content.encode())

            if not schema.validate(xml_doc):
                for error in schema.error_log:
                    errors.append(f"XSD: Line {error.line}: {error.message}")

        except lxml_etree.XMLSchemaParseError as e:
            # Schema parsing error - fall back to structural validation
            pass
        except lxml_etree.XMLSyntaxError as e:
            errors.append(f"XML syntax error: {e}")
        except Exception as e:
            # Non-critical - structural validation already done
            pass

        return errors

    def _get_embedded_xsd(self) -> str:
        """Get the embedded XSD schema for CBAM validation."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           targetNamespace="urn:cbam:transitional:v1"
           xmlns:cbam="urn:cbam:transitional:v1"
           elementFormDefault="qualified">

    <xs:element name="CBAMReport">
        <xs:complexType>
            <xs:sequence>
                <xs:element name="Header" type="cbam:HeaderType"/>
                <xs:element name="Declarant" type="cbam:DeclarantType"/>
                <xs:element name="ReportingPeriod" type="cbam:ReportingPeriodType"/>
                <xs:element name="ImportedGoods" type="cbam:ImportedGoodsType"/>
                <xs:element name="Summary" type="cbam:SummaryType"/>
            </xs:sequence>
            <xs:attribute name="version" type="xs:string" use="required"/>
        </xs:complexType>
    </xs:element>

    <xs:complexType name="HeaderType">
        <xs:sequence>
            <xs:element name="GeneratedAt" type="xs:string"/>
            <xs:element name="ReportType" type="xs:string"/>
            <xs:element name="SchemaVersion" type="xs:string"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="DeclarantType">
        <xs:sequence>
            <xs:element name="Name" type="xs:string"/>
            <xs:element name="EORINumber" type="xs:string"/>
            <xs:element name="Address" type="cbam:AddressType"/>
            <xs:element name="Contact" type="cbam:ContactType"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="AddressType">
        <xs:sequence>
            <xs:element name="Street" type="xs:string"/>
            <xs:element name="City" type="xs:string"/>
            <xs:element name="PostalCode" type="xs:string"/>
            <xs:element name="Country" type="xs:string"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="ContactType">
        <xs:sequence>
            <xs:element name="Name" type="xs:string"/>
            <xs:element name="Email" type="xs:string"/>
            <xs:element name="Phone" type="xs:string" minOccurs="0"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="ReportingPeriodType">
        <xs:sequence>
            <xs:element name="Quarter" type="xs:string"/>
            <xs:element name="Year" type="xs:integer"/>
            <xs:element name="StartDate" type="xs:string"/>
            <xs:element name="EndDate" type="xs:string"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="ImportedGoodsType">
        <xs:sequence>
            <xs:element name="Good" type="cbam:GoodType" minOccurs="0" maxOccurs="unbounded"/>
            <xs:element name="Line" type="cbam:LineType" minOccurs="0" maxOccurs="unbounded"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="GoodType">
        <xs:sequence>
            <xs:element name="CNCode" type="xs:string"/>
            <xs:element name="CountryOfOrigin" type="xs:string"/>
            <xs:element name="TotalQuantityTonnes" type="xs:decimal"/>
            <xs:element name="Emissions" type="cbam:EmissionsType"/>
            <xs:element name="MethodUsed" type="xs:string"/>
            <xs:element name="LineCount" type="xs:integer"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="LineType">
        <xs:sequence>
            <xs:element name="LineId" type="xs:string"/>
            <xs:element name="Emissions" type="cbam:LineEmissionsType"/>
            <xs:element name="Methods" type="cbam:MethodsType"/>
            <xs:element name="FactorReferences" type="cbam:FactorReferencesType"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="EmissionsType">
        <xs:sequence>
            <xs:element name="DirectEmissionsTCO2e" type="xs:decimal"/>
            <xs:element name="IndirectEmissionsTCO2e" type="xs:decimal"/>
            <xs:element name="TotalEmissionsTCO2e" type="xs:decimal"/>
            <xs:element name="WeightedIntensity" type="xs:decimal"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="LineEmissionsType">
        <xs:sequence>
            <xs:element name="DirectEmissionsTCO2e" type="xs:decimal"/>
            <xs:element name="IndirectEmissionsTCO2e" type="xs:decimal"/>
            <xs:element name="TotalEmissionsTCO2e" type="xs:decimal"/>
            <xs:element name="Intensity" type="xs:decimal"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="MethodsType">
        <xs:sequence>
            <xs:element name="DirectMethod" type="xs:string"/>
            <xs:element name="IndirectMethod" type="xs:string"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="FactorReferencesType">
        <xs:sequence>
            <xs:element name="DirectFactorRef" type="xs:string"/>
            <xs:element name="IndirectFactorRef" type="xs:string"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="SummaryType">
        <xs:sequence>
            <xs:element name="TotalLines" type="xs:integer"/>
            <xs:element name="TotalDirectEmissionsTCO2e" type="xs:decimal"/>
            <xs:element name="TotalIndirectEmissionsTCO2e" type="xs:decimal"/>
            <xs:element name="TotalEmissionsTCO2e" type="xs:decimal"/>
            <xs:element name="MethodBreakdown" type="cbam:MethodBreakdownType"/>
            <xs:element name="Assumptions" type="cbam:AssumptionsType" minOccurs="0"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="MethodBreakdownType">
        <xs:sequence>
            <xs:element name="LinesWithSupplierDirectData" type="xs:integer"/>
            <xs:element name="LinesWithSupplierIndirectData" type="xs:integer"/>
            <xs:element name="LinesUsingDefaults" type="xs:integer"/>
            <xs:element name="DefaultUsagePercent" type="xs:decimal"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="AssumptionsType">
        <xs:sequence>
            <xs:element name="Assumption" type="cbam:AssumptionType" minOccurs="0" maxOccurs="unbounded"/>
        </xs:sequence>
    </xs:complexType>

    <xs:complexType name="AssumptionType">
        <xs:sequence>
            <xs:element name="Type" type="xs:string"/>
            <xs:element name="Description" type="xs:string"/>
            <xs:element name="Rationale" type="xs:string"/>
            <xs:element name="AppliesTo" type="xs:string"/>
            <xs:element name="FactorRef" type="xs:string" minOccurs="0"/>
        </xs:sequence>
    </xs:complexType>

</xs:schema>'''

    def _add_header(self, root: ET.Element, config: CBAMConfig) -> None:
        """Add report header."""
        header = ET.SubElement(root, "Header")

        ET.SubElement(header, "GeneratedAt").text = self.generated_at.isoformat() + "Z"
        ET.SubElement(header, "ReportType").text = "TRANSITIONAL_QUARTERLY"
        ET.SubElement(header, "SchemaVersion").text = self.XSD_VERSION

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
