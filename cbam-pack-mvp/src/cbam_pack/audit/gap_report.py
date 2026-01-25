"""
Gap Report Generator for CBAM Pack

Generates actionable gap reports showing:
- Which lines used defaults
- What supplier fields would remove defaults
- What the importer should ask suppliers for
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from cbam_pack.calculators.emissions_calculator import CalculationResult
from cbam_pack.models import ImportLineItem, MethodType, CBAMConfig


@dataclass
class SupplierDataGap:
    """A gap in supplier-specific data."""
    line_id: str
    cn_code: str
    product_description: str
    country_of_origin: str
    supplier_id: Optional[str]
    missing_fields: list[str]
    current_method: str
    recommended_action: str
    data_request_template: str


@dataclass
class GapSummary:
    """Summary of all data gaps."""
    total_lines: int
    lines_with_gaps: int
    gap_percentage: float
    gaps_by_country: dict
    gaps_by_cn_code: dict
    priority_actions: list[str]


class GapReportGenerator:
    """
    Generates comprehensive gap reports for CBAM compliance.

    Identifies missing supplier data and provides actionable
    recommendations for data collection.
    """

    # Required supplier fields for full compliance
    REQUIRED_SUPPLIER_FIELDS = {
        "supplier_id": "Supplier identification number",
        "installation_id": "Production installation ID (CBAM registry)",
        "supplier_direct_emissions": "Direct emission intensity (tCO2e/tonne product)",
        "supplier_indirect_emissions": "Indirect emission intensity (tCO2e/tonne product)",
        "supplier_certificate_ref": "Verification certificate reference",
    }

    def __init__(self):
        """Initialize the gap report generator."""
        pass

    def generate(
        self,
        calc_result: CalculationResult,
        lines: list[ImportLineItem],
        config: CBAMConfig,
        output_path: Path,
    ) -> dict:
        """
        Generate comprehensive gap report.

        Args:
            calc_result: Calculation results
            lines: Original import line items
            config: CBAM configuration
            output_path: Path to write gap_report.json

        Returns:
            Gap report dictionary
        """
        # Build line lookup
        line_lookup = {line.line_id: line for line in lines}
        result_lookup = {r.line_id: r for r in calc_result.line_results}

        # Identify gaps
        gaps: list[SupplierDataGap] = []
        gaps_by_country: dict[str, int] = {}
        gaps_by_cn_code: dict[str, int] = {}

        for line in lines:
            result = result_lookup.get(line.line_id)
            if not result:
                continue

            # Check if using defaults
            uses_default_direct = result.method_direct == MethodType.DEFAULT
            uses_default_indirect = result.method_indirect == MethodType.DEFAULT

            if uses_default_direct or uses_default_indirect:
                # Identify missing fields
                missing_fields = self._identify_missing_fields(line)

                # Build data request template
                template = self._build_data_request_template(line, missing_fields)

                # Determine recommended action
                action = self._determine_action(line, missing_fields)

                gap = SupplierDataGap(
                    line_id=line.line_id,
                    cn_code=line.cn_code,
                    product_description=line.product_description,
                    country_of_origin=line.country_of_origin,
                    supplier_id=line.supplier_id,
                    missing_fields=missing_fields,
                    current_method="default" if uses_default_direct else "partial_supplier",
                    recommended_action=action,
                    data_request_template=template,
                )
                gaps.append(gap)

                # Track by country and CN code
                gaps_by_country[line.country_of_origin] = gaps_by_country.get(
                    line.country_of_origin, 0
                ) + 1
                cn_prefix = line.cn_code[:4]
                gaps_by_cn_code[cn_prefix] = gaps_by_cn_code.get(cn_prefix, 0) + 1

        # Generate summary
        total_lines = len(lines)
        lines_with_gaps = len(gaps)
        gap_percentage = (lines_with_gaps / total_lines * 100) if total_lines > 0 else 0

        # Prioritize actions
        priority_actions = self._generate_priority_actions(
            gaps, gaps_by_country, gaps_by_cn_code, config
        )

        summary = GapSummary(
            total_lines=total_lines,
            lines_with_gaps=lines_with_gaps,
            gap_percentage=round(gap_percentage, 1),
            gaps_by_country=gaps_by_country,
            gaps_by_cn_code=gaps_by_cn_code,
            priority_actions=priority_actions,
        )

        # Build report
        report = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "reporting_period": f"{config.reporting_period.quarter.value} {config.reporting_period.year}",
            "summary": {
                "total_lines": summary.total_lines,
                "lines_with_gaps": summary.lines_with_gaps,
                "gap_percentage": summary.gap_percentage,
                "gaps_by_country": summary.gaps_by_country,
                "gaps_by_cn_code": summary.gaps_by_cn_code,
            },
            "priority_actions": summary.priority_actions,
            "gaps": [
                {
                    "line_id": g.line_id,
                    "cn_code": g.cn_code,
                    "product_description": g.product_description,
                    "country_of_origin": g.country_of_origin,
                    "supplier_id": g.supplier_id,
                    "missing_fields": g.missing_fields,
                    "current_method": g.current_method,
                    "recommended_action": g.recommended_action,
                    "data_request_template": g.data_request_template,
                }
                for g in gaps
            ],
            "supplier_data_requirements": {
                field: desc for field, desc in self.REQUIRED_SUPPLIER_FIELDS.items()
            },
        }

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def _identify_missing_fields(self, line: ImportLineItem) -> list[str]:
        """Identify which supplier fields are missing."""
        missing = []

        if line.supplier_direct_emissions is None:
            missing.append("supplier_direct_emissions")
        if line.supplier_indirect_emissions is None:
            missing.append("supplier_indirect_emissions")
        if not line.supplier_id:
            missing.append("supplier_id")
        if not line.installation_id:
            missing.append("installation_id")
        if not line.supplier_certificate_ref:
            missing.append("supplier_certificate_ref")

        return missing

    def _build_data_request_template(
        self,
        line: ImportLineItem,
        missing_fields: list[str],
    ) -> str:
        """Build a data request template for the supplier."""
        parts = [
            f"Data Request for CBAM Compliance",
            f"",
            f"Product: {line.product_description}",
            f"CN Code: {line.cn_code}",
            f"Country: {line.country_of_origin}",
            f"",
            f"Required Data:",
        ]

        for field in missing_fields:
            desc = self.REQUIRED_SUPPLIER_FIELDS.get(field, field)
            parts.append(f"  - {desc}")

        parts.extend([
            "",
            "Please provide:",
            "1. Direct emissions intensity (tCO2e per tonne of product)",
            "2. Indirect emissions intensity (tCO2e per tonne of product)",
            "3. Production installation ID (if registered in CBAM registry)",
            "4. Verification documentation reference",
        ])

        return "\n".join(parts)

    def _determine_action(
        self,
        line: ImportLineItem,
        missing_fields: list[str],
    ) -> str:
        """Determine recommended action based on missing data."""
        if not line.supplier_id:
            return "Identify supplier and request CBAM data package"

        if "supplier_direct_emissions" in missing_fields and "supplier_indirect_emissions" in missing_fields:
            return f"Request complete emission data from supplier {line.supplier_id}"

        if "supplier_direct_emissions" in missing_fields:
            return f"Request direct emission intensity from supplier {line.supplier_id}"

        if "supplier_indirect_emissions" in missing_fields:
            return f"Request indirect emission intensity from supplier {line.supplier_id}"

        if "installation_id" in missing_fields:
            return f"Confirm installation ID for supplier {line.supplier_id}"

        return "Verify and update supplier data"

    def _generate_priority_actions(
        self,
        gaps: list[SupplierDataGap],
        gaps_by_country: dict[str, int],
        gaps_by_cn_code: dict[str, int],
        config: CBAMConfig,
    ) -> list[str]:
        """Generate prioritized action list."""
        actions = []

        if not gaps:
            actions.append("No data gaps identified. All lines have supplier-specific data.")
            return actions

        # Priority 1: Countries with most gaps
        if gaps_by_country:
            top_country = max(gaps_by_country.items(), key=lambda x: x[1])
            actions.append(
                f"Priority 1: Focus on suppliers from {top_country[0]} "
                f"({top_country[1]} lines using defaults)"
            )

        # Priority 2: CN codes with most gaps
        if gaps_by_cn_code:
            top_cn = max(gaps_by_cn_code.items(), key=lambda x: x[1])
            actions.append(
                f"Priority 2: Address CN code group {top_cn[0]}xx "
                f"({top_cn[1]} lines using defaults)"
            )

        # Priority 3: Unidentified suppliers
        unknown_supplier_count = sum(1 for g in gaps if not g.supplier_id)
        if unknown_supplier_count > 0:
            actions.append(
                f"Priority 3: Identify {unknown_supplier_count} suppliers "
                "without supplier_id in your import records"
            )

        # Priority 4: General recommendation
        total_gaps = len(gaps)
        actions.append(
            f"Overall: Collect supplier emission data for {total_gaps} import lines "
            "to reduce reliance on default factors"
        )

        return actions
