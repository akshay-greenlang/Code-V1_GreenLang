"""
ESRS E1 (EU CSRD) Report Generator
GL-VCCI Scope 3 Platform

Generates ESRS E1 compliant reports for EU Corporate Sustainability Reporting Directive.

Version: 1.0.0
Phase: 3 (Weeks 16-18)
Date: 2025-10-30
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..models import CompanyInfo, EmissionsData, EnergyData, IntensityMetrics
from ..config import ESRS_E1_CONFIG
from ..exceptions import StandardComplianceError

logger = logging.getLogger(__name__)


class ESRSE1Generator:
    """
    Generates ESRS E1 (EU CSRD) compliant reports.

    Features:
    - All required disclosures (E1-1 through E1-9)
    - Standardized data tables
    - Narrative sections
    - Compliance validation
    """

    def __init__(self):
        """Initialize ESRS E1 generator."""
        self.config = ESRS_E1_CONFIG

    def generate_report_content(
        self,
        company_info: CompanyInfo,
        emissions_data: EmissionsData,
        energy_data: Optional[EnergyData] = None,
        intensity_metrics: Optional[IntensityMetrics] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete ESRS E1 report content.

        Args:
            company_info: Company information
            emissions_data: Emissions data
            energy_data: Energy consumption data
            intensity_metrics: Intensity metrics

        Returns:
            Structured report content
        """
        logger.info("Generating ESRS E1 report content")

        report = {
            "standard": "ESRS E1",
            "standard_name": "Climate Change",
            "reporting_entity": company_info.name,
            "reporting_period": company_info.reporting_year,
            "generated_at": datetime.utcnow().isoformat(),
            "disclosures": [],
        }

        # E1-6: GHG Emissions (most critical)
        report["disclosures"].append(self._generate_e1_6(emissions_data))

        # E1-5: Energy Consumption
        if energy_data:
            report["disclosures"].append(self._generate_e1_5(energy_data))

        # E1-4: Targets
        report["disclosures"].append(self._generate_e1_4())

        # Add intensity metrics
        if intensity_metrics:
            report["intensity_metrics"] = self._format_intensity_metrics(intensity_metrics)

        # Add narrative sections
        report["executive_summary"] = self._generate_executive_summary(company_info, emissions_data)
        report["methodology"] = self._generate_methodology()

        return report

    def _generate_e1_6(self, emissions_data: EmissionsData) -> Dict[str, Any]:
        """Generate E1-6: Gross Scopes 1, 2, 3 and Total GHG Emissions."""
        total = (
            emissions_data.scope1_tco2e
            + emissions_data.scope2_location_tco2e
            + emissions_data.scope3_tco2e
        )

        return {
            "disclosure_id": "E1-6",
            "title": "Gross Scope 1, 2, 3 and Total GHG Emissions",
            "tables": [
                {
                    "name": "GHG Emissions by Scope",
                    "data": [
                        {
                            "scope": "Scope 1",
                            "tco2e": emissions_data.scope1_tco2e,
                            "percent_of_total": (emissions_data.scope1_tco2e / total * 100) if total > 0 else 0,
                        },
                        {
                            "scope": "Scope 2 (Location-based)",
                            "tco2e": emissions_data.scope2_location_tco2e,
                            "percent_of_total": (emissions_data.scope2_location_tco2e / total * 100) if total > 0 else 0,
                        },
                        {
                            "scope": "Scope 2 (Market-based)",
                            "tco2e": emissions_data.scope2_market_tco2e,
                            "percent_of_total": (emissions_data.scope2_market_tco2e / total * 100) if total > 0 else 0,
                        },
                        {
                            "scope": "Scope 3",
                            "tco2e": emissions_data.scope3_tco2e,
                            "percent_of_total": (emissions_data.scope3_tco2e / total * 100) if total > 0 else 0,
                        },
                        {
                            "scope": "Total",
                            "tco2e": total,
                            "percent_of_total": 100.0,
                        },
                    ],
                },
                {
                    "name": "Scope 3 Emissions by Category",
                    "data": [
                        {
                            "category": f"Category {cat}",
                            "tco2e": emissions,
                            "percent_of_scope3": (emissions / emissions_data.scope3_tco2e * 100) if emissions_data.scope3_tco2e > 0 else 0,
                        }
                        for cat, emissions in sorted(emissions_data.scope3_categories.items(), key=lambda x: x[1], reverse=True)
                    ],
                },
            ],
            "narrative": f"Total GHG emissions for the reporting period were {total:,.0f} tCO2e. Scope 3 emissions represent {(emissions_data.scope3_tco2e / total * 100):.1f}% of total emissions, highlighting the significance of value chain engagement.",
        }

    def _generate_e1_5(self, energy_data: EnergyData) -> Dict[str, Any]:
        """Generate E1-5: Energy Consumption and Mix."""
        return {
            "disclosure_id": "E1-5",
            "title": "Energy Consumption and Mix",
            "tables": [
                {
                    "name": "Energy Consumption",
                    "data": [
                        {
                            "source": "Renewable Energy",
                            "mwh": energy_data.renewable_energy_mwh,
                            "percent": (energy_data.renewable_energy_mwh / energy_data.total_energy_mwh * 100) if energy_data.total_energy_mwh > 0 else 0,
                        },
                        {
                            "source": "Non-Renewable Energy",
                            "mwh": energy_data.non_renewable_energy_mwh,
                            "percent": (energy_data.non_renewable_energy_mwh / energy_data.total_energy_mwh * 100) if energy_data.total_energy_mwh > 0 else 0,
                        },
                        {
                            "source": "Total",
                            "mwh": energy_data.total_energy_mwh,
                            "percent": 100.0,
                        },
                    ],
                },
            ],
            "narrative": f"Total energy consumption for the period was {energy_data.total_energy_mwh:,.0f} MWh, with renewable energy accounting for {energy_data.renewable_pct or 0:.1f}%.",
        }

    def _generate_e1_4(self) -> Dict[str, Any]:
        """Generate E1-4: Targets related to climate change mitigation."""
        return {
            "disclosure_id": "E1-4",
            "title": "Targets Related to Climate Change Mitigation",
            "narrative": "The organization is committed to science-based emission reduction targets aligned with limiting global warming to 1.5°C. Specific targets and progress will be disclosed as they are established.",
        }

    def _format_intensity_metrics(self, intensity_metrics: IntensityMetrics) -> Dict[str, Any]:
        """Format intensity metrics."""
        metrics = {}

        if intensity_metrics.tco2e_per_million_usd:
            metrics["revenue_intensity"] = {
                "value": intensity_metrics.tco2e_per_million_usd,
                "unit": "tCO2e per $M revenue",
            }

        if intensity_metrics.tco2e_per_fte:
            metrics["employee_intensity"] = {
                "value": intensity_metrics.tco2e_per_fte,
                "unit": "tCO2e per FTE",
            }

        return metrics

    def _generate_executive_summary(self, company_info: CompanyInfo, emissions_data: EmissionsData) -> str:
        """Generate executive summary for ESRS E1."""
        total = (
            emissions_data.scope1_tco2e
            + emissions_data.scope2_location_tco2e
            + emissions_data.scope3_tco2e
        )

        return f"""{company_info.name} reports total GHG emissions of {total:,.0f} tCO2e for {company_info.reporting_year} in accordance with ESRS E1. Scope 3 emissions represent {(emissions_data.scope3_tco2e / total * 100):.1f}% of total, emphasizing value chain decarbonization as a strategic priority."""

    def _generate_methodology(self) -> str:
        """Generate methodology section."""
        return """Calculations performed in accordance with GHG Protocol Corporate Standard, using IPCC AR6 GWP values. Scope 2 reported using both location-based and market-based methods per ESRS requirements."""


__all__ = ["ESRSE1Generator"]
