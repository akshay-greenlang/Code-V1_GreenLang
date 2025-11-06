"""
Table Generator
GL-VCCI Scope 3 Platform

Generates data tables for sustainability reports.

Version: 1.0.0
Phase: 3 (Weeks 16-18)
Date: 2025-10-30
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd

from ..models import EmissionsData, EnergyData
from ..exceptions import ReportingError

logger = logging.getLogger(__name__)


class TableGenerator:
    """
    Generates formatted data tables for sustainability reports.

    Features:
    - GHG emissions tables
    - Scope 3 category breakdowns
    - Data quality tables
    - Intensity metrics tables
    """

    def __init__(self):
        """Initialize table generator."""
        pass

    def generate_ghg_emissions_table(self, emissions_data: EmissionsData) -> pd.DataFrame:
        """Generate GHG emissions by scope table."""
        logger.info("Generating GHG emissions table")

        total = (
            emissions_data.scope1_tco2e
            + emissions_data.scope2_location_tco2e
            + emissions_data.scope3_tco2e
        )

        data = [
            {
                "Scope": "Scope 1",
                "Emissions (tCO2e)": emissions_data.scope1_tco2e,
                "% of Total": (emissions_data.scope1_tco2e / total * 100) if total > 0 else 0,
                "Data Quality": self._get_dqi_rating(emissions_data.data_quality_by_scope, "Scope 1"),
            },
            {
                "Scope": "Scope 2 (Location-based)",
                "Emissions (tCO2e)": emissions_data.scope2_location_tco2e,
                "% of Total": (emissions_data.scope2_location_tco2e / total * 100) if total > 0 else 0,
                "Data Quality": self._get_dqi_rating(emissions_data.data_quality_by_scope, "Scope 2"),
            },
            {
                "Scope": "Scope 2 (Market-based)",
                "Emissions (tCO2e)": emissions_data.scope2_market_tco2e,
                "% of Total": (emissions_data.scope2_market_tco2e / total * 100) if total > 0 else 0,
                "Data Quality": self._get_dqi_rating(emissions_data.data_quality_by_scope, "Scope 2"),
            },
            {
                "Scope": "Scope 3",
                "Emissions (tCO2e)": emissions_data.scope3_tco2e,
                "% of Total": (emissions_data.scope3_tco2e / total * 100) if total > 0 else 0,
                "Data Quality": self._get_dqi_rating(emissions_data.data_quality_by_scope, "Scope 3"),
            },
            {
                "Scope": "**Total**",
                "Emissions (tCO2e)": total,
                "% of Total": 100.0,
                "Data Quality": self._get_dqi_rating({"Overall": emissions_data.avg_dqi_score}, "Overall"),
            },
        ]

        return pd.DataFrame(data)

    def generate_scope3_category_table(self, emissions_data: EmissionsData) -> pd.DataFrame:
        """Generate Scope 3 emissions by category table."""
        logger.info("Generating Scope 3 category table")

        scope3_total = emissions_data.scope3_tco2e
        category_names = {
            1: "Purchased Goods & Services",
            2: "Capital Goods",
            3: "Fuel & Energy Related Activities",
            4: "Upstream Transportation & Distribution",
            5: "Waste Generated in Operations",
            6: "Business Travel",
            7: "Employee Commuting",
            8: "Upstream Leased Assets",
            9: "Downstream Transportation & Distribution",
            10: "Processing of Sold Products",
            11: "Use of Sold Products",
            12: "End-of-Life Treatment of Sold Products",
            13: "Downstream Leased Assets",
            14: "Franchises",
            15: "Investments",
        }

        data = []
        for cat_num, emissions in emissions_data.scope3_categories.items():
            data.append({
                "Category": f"Cat {cat_num}",
                "Description": category_names.get(cat_num, f"Category {cat_num}"),
                "Emissions (tCO2e)": emissions,
                "% of Scope 3": (emissions / scope3_total * 100) if scope3_total > 0 else 0,
                "% of Total": (emissions / (emissions_data.scope1_tco2e + emissions_data.scope2_location_tco2e + scope3_total) * 100) if (emissions_data.scope1_tco2e + emissions_data.scope2_location_tco2e + scope3_total) > 0 else 0,
            })

        # Sort by emissions descending
        df = pd.DataFrame(data)
        df = df.sort_values("Emissions (tCO2e)", ascending=False)

        # Add total row
        total_row = pd.DataFrame([{
            "Category": "**Total**",
            "Description": "All Categories",
            "Emissions (tCO2e)": scope3_total,
            "% of Scope 3": 100.0,
            "% of Total": (scope3_total / (emissions_data.scope1_tco2e + emissions_data.scope2_location_tco2e + scope3_total) * 100) if (emissions_data.scope1_tco2e + emissions_data.scope2_location_tco2e + scope3_total) > 0 else 0,
        }])

        df = pd.concat([df, total_row], ignore_index=True)

        return df

    def generate_energy_consumption_table(self, energy_data: EnergyData) -> pd.DataFrame:
        """Generate energy consumption table."""
        logger.info("Generating energy consumption table")

        data = [
            {
                "Energy Source": "Renewable Energy",
                "Consumption (MWh)": energy_data.renewable_energy_mwh,
                "% of Total": (energy_data.renewable_energy_mwh / energy_data.total_energy_mwh * 100) if energy_data.total_energy_mwh > 0 else 0,
            },
            {
                "Energy Source": "Non-Renewable Energy",
                "Consumption (MWh)": energy_data.non_renewable_energy_mwh,
                "% of Total": (energy_data.non_renewable_energy_mwh / energy_data.total_energy_mwh * 100) if energy_data.total_energy_mwh > 0 else 0,
            },
            {
                "Energy Source": "**Total**",
                "Consumption (MWh)": energy_data.total_energy_mwh,
                "% of Total": 100.0,
            },
        ]

        return pd.DataFrame(data)

    def generate_intensity_metrics_table(
        self,
        emissions_data: EmissionsData,
        intensity_metrics: Dict[str, float],
    ) -> pd.DataFrame:
        """Generate intensity metrics table."""
        logger.info("Generating intensity metrics table")

        total_emissions = (
            emissions_data.scope1_tco2e
            + emissions_data.scope2_location_tco2e
            + emissions_data.scope3_tco2e
        )

        data = []

        if "tco2e_per_million_usd" in intensity_metrics:
            data.append({
                "Metric": "Carbon Intensity (Revenue)",
                "Value": intensity_metrics["tco2e_per_million_usd"],
                "Unit": "tCO2e per $M revenue",
            })

        if "tco2e_per_fte" in intensity_metrics:
            data.append({
                "Metric": "Carbon Intensity (Employees)",
                "Value": intensity_metrics["tco2e_per_fte"],
                "Unit": "tCO2e per FTE",
            })

        if "tco2e_per_unit" in intensity_metrics:
            unit_name = intensity_metrics.get("unit_name", "unit")
            data.append({
                "Metric": f"Carbon Intensity (Product)",
                "Value": intensity_metrics["tco2e_per_unit"],
                "Unit": f"tCO2e per {unit_name}",
            })

        return pd.DataFrame(data)

    def generate_yoy_comparison_table(
        self,
        emissions_data: EmissionsData,
    ) -> pd.DataFrame:
        """Generate year-over-year comparison table."""
        logger.info("Generating YoY comparison table")

        if not emissions_data.prior_year_emissions:
            raise ReportingError("Prior year emissions data required for YoY table")

        current_year = emissions_data.reporting_year
        prior_year = current_year - 1

        prior = emissions_data.prior_year_emissions

        current_total = (
            emissions_data.scope1_tco2e
            + emissions_data.scope2_location_tco2e
            + emissions_data.scope3_tco2e
        )
        prior_total = prior.get("total_tco2e", 0)

        data = [
            {
                "Scope": "Scope 1",
                f"{prior_year} (tCO2e)": prior.get("scope1_tco2e", 0),
                f"{current_year} (tCO2e)": emissions_data.scope1_tco2e,
                "Change (tCO2e)": emissions_data.scope1_tco2e - prior.get("scope1_tco2e", 0),
                "Change (%)": self._calculate_change_pct(prior.get("scope1_tco2e", 0), emissions_data.scope1_tco2e),
            },
            {
                "Scope": "Scope 2",
                f"{prior_year} (tCO2e)": prior.get("scope2_tco2e", 0),
                f"{current_year} (tCO2e)": emissions_data.scope2_location_tco2e,
                "Change (tCO2e)": emissions_data.scope2_location_tco2e - prior.get("scope2_tco2e", 0),
                "Change (%)": self._calculate_change_pct(prior.get("scope2_tco2e", 0), emissions_data.scope2_location_tco2e),
            },
            {
                "Scope": "Scope 3",
                f"{prior_year} (tCO2e)": prior.get("scope3_tco2e", 0),
                f"{current_year} (tCO2e)": emissions_data.scope3_tco2e,
                "Change (tCO2e)": emissions_data.scope3_tco2e - prior.get("scope3_tco2e", 0),
                "Change (%)": self._calculate_change_pct(prior.get("scope3_tco2e", 0), emissions_data.scope3_tco2e),
            },
            {
                "Scope": "**Total**",
                f"{prior_year} (tCO2e)": prior_total,
                f"{current_year} (tCO2e)": current_total,
                "Change (tCO2e)": current_total - prior_total,
                "Change (%)": self._calculate_change_pct(prior_total, current_total),
            },
        ]

        return pd.DataFrame(data)

    def generate_data_quality_table(self, emissions_data: EmissionsData) -> pd.DataFrame:
        """Generate data quality assessment table."""
        logger.info("Generating data quality table")

        dqi_by_scope = emissions_data.data_quality_by_scope or {}

        data = []
        for scope, dqi in dqi_by_scope.items():
            data.append({
                "Scope/Category": scope,
                "DQI Score": dqi,
                "Rating": self._get_dqi_rating_name(dqi),
            })

        # Add overall
        data.append({
            "Scope/Category": "**Overall**",
            "DQI Score": emissions_data.avg_dqi_score,
            "Rating": self._get_dqi_rating_name(emissions_data.avg_dqi_score),
        })

        return pd.DataFrame(data)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _get_dqi_rating(self, dqi_by_scope: Optional[Dict[str, float]], scope: str) -> str:
        """Get DQI rating for scope."""
        if not dqi_by_scope or scope not in dqi_by_scope:
            return "N/A"

        dqi = dqi_by_scope[scope]
        return self._get_dqi_rating_name(dqi)

    def _get_dqi_rating_name(self, dqi: float) -> str:
        """Get DQI rating name from score."""
        if dqi >= 90:
            return "Excellent"
        elif dqi >= 80:
            return "Good"
        elif dqi >= 70:
            return "Fair"
        else:
            return "Poor"

    def _calculate_change_pct(self, prior: float, current: float) -> float:
        """Calculate percentage change."""
        if prior == 0:
            return 0.0 if current == 0 else 100.0
        return ((current - prior) / prior) * 100


__all__ = ["TableGenerator"]
