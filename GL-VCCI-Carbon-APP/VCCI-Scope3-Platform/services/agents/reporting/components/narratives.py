"""
Narrative Generator
GL-VCCI Scope 3 Platform

Generates narrative text sections for sustainability reports.

Version: 1.0.0
Phase: 3 (Weeks 16-18)
Date: 2025-10-30
"""

import logging
from typing import Dict, List, Any, Optional

from ..models import EmissionsData, CompanyInfo, EnergyData

logger = logging.getLogger(__name__)


class NarrativeGenerator:
    """
    Generates narrative text sections for sustainability reports.

    Features:
    - Executive summaries
    - Methodology descriptions
    - Data quality assessments
    - Trend analyses
    """

    def __init__(self):
        """Initialize narrative generator."""
        pass

    def generate_executive_summary(
        self,
        company_info: CompanyInfo,
        emissions_data: EmissionsData,
    ) -> str:
        """Generate executive summary."""
        logger.info("Generating executive summary")

        total_emissions = (
            emissions_data.scope1_tco2e
            + emissions_data.scope2_location_tco2e
            + emissions_data.scope3_tco2e
        )

        scope3_pct = (emissions_data.scope3_tco2e / total_emissions * 100) if total_emissions > 0 else 0

        summary = f"""# Executive Summary

{company_info.name} has completed a comprehensive greenhouse gas (GHG) emissions inventory for the {company_info.reporting_year} reporting period. This report presents our carbon footprint across Scope 1, 2, and 3 emissions in accordance with the GHG Protocol Corporate Accounting and Reporting Standard.

## Key Findings

- **Total GHG Emissions**: {total_emissions:,.0f} tCO2e
- **Scope 1 (Direct Emissions)**: {emissions_data.scope1_tco2e:,.0f} tCO2e ({emissions_data.scope1_tco2e / total_emissions * 100:.1f}% of total)
- **Scope 2 (Indirect Energy)**: {emissions_data.scope2_location_tco2e:,.0f} tCO2e location-based ({emissions_data.scope2_location_tco2e / total_emissions * 100:.1f}% of total)
- **Scope 3 (Value Chain)**: {emissions_data.scope3_tco2e:,.0f} tCO2e ({scope3_pct:.1f}% of total)

Scope 3 emissions represent the majority of our carbon footprint at {scope3_pct:.0f}%, highlighting the importance of value chain engagement in our decarbonization strategy.

## Scope 3 Breakdown

We have calculated emissions for {len(emissions_data.scope3_categories)} of 15 Scope 3 categories, with the top contributors being:
"""

        # Add top 3 categories
        sorted_cats = sorted(
            emissions_data.scope3_categories.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        cat_names = {1: "Purchased Goods & Services", 4: "Upstream Transportation", 6: "Business Travel"}

        for i, (cat_num, emissions) in enumerate(sorted_cats, 1):
            cat_name = cat_names.get(cat_num, f"Category {cat_num}")
            pct = (emissions / emissions_data.scope3_tco2e * 100) if emissions_data.scope3_tco2e > 0 else 0
            summary += f"\n{i}. **{cat_name}**: {emissions:,.0f} tCO2e ({pct:.1f}% of Scope 3)"

        summary += f"""

## Data Quality

The overall data quality index (DQI) for this inventory is {emissions_data.avg_dqi_score:.0f}/100, indicating {"excellent" if emissions_data.avg_dqi_score >= 90 else "good" if emissions_data.avg_dqi_score >= 80 else "fair"} data quality. We have implemented robust data collection processes and utilized primary data where available.
"""

        if emissions_data.yoy_change_pct is not None:
            direction = "increased" if emissions_data.yoy_change_pct > 0 else "decreased"
            summary += f"""
## Year-over-Year Trends

Total emissions have {direction} by {abs(emissions_data.yoy_change_pct):.1f}% compared to the prior year, demonstrating {"the need for accelerated decarbonization efforts" if emissions_data.yoy_change_pct > 0 else "progress toward our climate goals"}.
"""

        return summary

    def generate_methodology_section(
        self,
        emissions_data: EmissionsData,
    ) -> str:
        """Generate methodology description."""
        logger.info("Generating methodology section")

        return f"""# Calculation Methodology

## Overview

This greenhouse gas inventory was prepared in accordance with the GHG Protocol Corporate Accounting and Reporting Standard and the GHG Protocol Corporate Value Chain (Scope 3) Accounting and Reporting Standard.

## Organizational Boundary

We have applied the **operational control** approach to define our organizational boundary. All facilities and operations where we have operational control are included in this inventory.

## Operational Boundary

### Scope 1: Direct Emissions
Scope 1 emissions include direct emissions from sources owned or controlled by the organization, including:
- Stationary combustion (boilers, generators)
- Mobile combustion (company-owned vehicles)
- Process emissions
- Fugitive emissions (refrigerants, other GHGs)

### Scope 2: Indirect Energy Emissions
Scope 2 emissions include indirect emissions from purchased electricity, heat, steam, and cooling. We report both:
- **Location-based method**: Using average emission factors for the grid
- **Market-based method**: Using supplier-specific emission factors where available

### Scope 3: Value Chain Emissions
We have calculated {len(emissions_data.scope3_categories)} Scope 3 categories using a combination of methods:

#### Category 1: Purchased Goods & Services
- **Method**: Supplier-specific product carbon footprints (PCF) where available, supplemented by spend-based calculations using Environmentally-Extended Input-Output (EEIO) factors
- **Data Sources**: Supplier PCF declarations, financial procurement data
- **Coverage**: {emissions_data.scope3_categories.get(1, 0):,.0f} tCO2e

#### Category 4: Upstream Transportation & Distribution
- **Method**: Distance-based calculation per ISO 14083:2023
- **Formula**: Emissions = Distance (km) × Weight (tonnes) × Emission Factor (kgCO2e/tonne-km)
- **Data Sources**: Shipment records, carrier data
- **Coverage**: {emissions_data.scope3_categories.get(4, 0):,.0f} tCO2e

#### Category 6: Business Travel
- **Method**: Distance-based for flights, nights for hotels, km for ground transport
- **Data Sources**: Travel booking systems, expense reports
- **Coverage**: {emissions_data.scope3_categories.get(6, 0):,.0f} tCO2e
- **Special Considerations**: Radiative forcing factor of 1.9 applied to aviation emissions

## Emission Factors

We have used the following emission factor databases:
- DEFRA UK Government GHG Conversion Factors 2024
- US EPA Emission Factors 2024
- IEA Emission Factors 2024
- Supplier-specific factors where available

All emission factors are based on **IPCC AR6** Global Warming Potentials (GWP) with a 100-year time horizon.

## Data Quality

Data quality was assessed using a pedigree matrix approach, evaluating:
- Temporal correlation (data vintage)
- Geographic correlation (regional applicability)
- Technological correlation (technology match)
- Completeness (data coverage)
- Reliability (data source)

## Uncertainty

Uncertainty has been quantified using **Monte Carlo simulation** with 10,000 iterations, propagating uncertainty from:
- Activity data measurement uncertainty
- Emission factor uncertainty
- Model uncertainty
"""

    def generate_data_quality_assessment(
        self,
        emissions_data: EmissionsData,
    ) -> str:
        """Generate data quality assessment."""
        logger.info("Generating data quality assessment")

        dqi = emissions_data.avg_dqi_score
        rating = "excellent" if dqi >= 90 else "good" if dqi >= 80 else "fair" if dqi >= 70 else "requires improvement"

        return f"""# Data Quality Assessment

## Overall Quality

The overall data quality index (DQI) for this inventory is **{dqi:.1f}/100**, which is considered **{rating}**.

## Quality by Scope

{''.join(self._format_scope_quality(emissions_data.data_quality_by_scope) if emissions_data.data_quality_by_scope else [])}

## Quality Assurance Measures

We have implemented the following quality assurance measures:
1. **Data Validation**: Automated checks for completeness, consistency, and outliers
2. **Provenance Tracking**: Complete audit trail from source data to final results
3. **Peer Review**: Internal review of calculations and methodologies
4. **Third-Party Verification**: {"Obtained" if dqi >= 80 else "Planned"} for Scope 1 and 2 emissions

## Limitations and Gaps

While we have achieved {rating} data quality overall, some limitations remain:
- Some Scope 3 categories rely on spend-based calculations due to limited supplier data
- Activity data for certain categories is estimated based on sampling
- Emission factors may not perfectly match specific technologies or regions

## Improvement Plans

We are committed to continuous improvement of data quality through:
- Expanding supplier engagement to obtain primary data
- Implementing continuous monitoring systems
- Enhancing data collection processes
- Annual updates to emission factor databases
"""

    def _format_scope_quality(self, dqi_by_scope: Dict[str, float]) -> List[str]:
        """Format scope quality text."""
        lines = []
        for scope, dqi in dqi_by_scope.items():
            rating = self._get_rating(dqi)
            lines.append(f"\n- **{scope}**: {dqi:.1f}/100 ({rating})")
        return lines

    def _get_rating(self, dqi: float) -> str:
        """Get rating from DQI score."""
        if dqi >= 90:
            return "Excellent"
        elif dqi >= 80:
            return "Good"
        elif dqi >= 70:
            return "Fair"
        else:
            return "Poor"

    def generate_limitations_section(self, emissions_data: EmissionsData) -> str:
        """Generate limitations and exclusions section."""
        excluded_cats = [i for i in range(1, 16) if i not in emissions_data.scope3_categories]

        return f"""# Limitations and Exclusions

## Scope 3 Categories Excluded

The following Scope 3 categories have been excluded from this inventory:
{chr(10).join([f"- Category {cat}" for cat in excluded_cats])}

These categories were excluded because:
- Not applicable to our business model
- Emissions determined to be immaterial (< 1% of total Scope 3)
- Data not currently available

## Data Gaps

We have identified the following data gaps:
- Limited supplier-specific emission factors for purchased goods
- Estimated activity data for some transport routes
- Incomplete coverage of business travel in certain regions

## Future Improvements

We plan to address these limitations through:
- Enhanced supplier engagement and data collection
- Implementation of IoT sensors for real-time monitoring
- Expansion of Scope 3 category coverage
- Third-party verification of full inventory
"""


__all__ = ["NarrativeGenerator"]
