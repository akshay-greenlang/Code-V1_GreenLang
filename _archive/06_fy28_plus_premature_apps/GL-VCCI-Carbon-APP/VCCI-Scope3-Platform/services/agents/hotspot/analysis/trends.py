# -*- coding: utf-8 -*-
"""
Trend Analysis Engine
GL-VCCI Scope 3 Platform

Time-series trend analysis for emissions data.
Analyzes temporal patterns and growth rates.

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
from statistics import mean, stdev

from ..exceptions import SegmentationError, DataValidationError

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Time-series trend analyzer for emissions data.

    Analyzes temporal patterns, calculates growth rates, and identifies trends.
    """

    def __init__(self):
        """Initialize trend analyzer."""
        logger.info("Initialized TrendAnalyzer")

    def analyze_monthly_trends(
        self,
        emissions_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze monthly emission trends.

        Args:
            emissions_data: Emission records with time_period field (YYYY-MM)

        Returns:
            Monthly trend analysis result
        """
        try:
            logger.info(f"Analyzing monthly trends for {len(emissions_data)} records")

            # Aggregate by month
            monthly_data = self._aggregate_by_period(emissions_data, "time_period")

            if not monthly_data:
                raise DataValidationError("No time period data available")

            # Sort by period
            sorted_periods = sorted(monthly_data.items())

            # Calculate metrics
            periods = [p[0] for p in sorted_periods]
            emissions = [p[1]["emissions_tco2e"] for p in sorted_periods]

            # Growth rate
            growth_rate = self._calculate_growth_rate(emissions)

            # Trend direction
            trend_direction = self._determine_trend(emissions)

            # Volatility
            volatility = stdev(emissions) if len(emissions) > 1 else 0

            result = {
                "analysis_type": "monthly_trends",
                "n_periods": len(periods),
                "periods": periods,
                "emissions_by_period": emissions,
                "total_emissions_tco2e": sum(emissions),
                "avg_monthly_emissions_tco2e": mean(emissions),
                "growth_rate_pct": round(growth_rate, 2),
                "trend_direction": trend_direction,
                "volatility": round(volatility, 2),
                "chart_data": self._generate_trend_chart_data(periods, emissions)
            }

            logger.info(
                f"Monthly trend analysis complete: {len(periods)} periods, "
                f"growth_rate={growth_rate:.1f}%, trend={trend_direction}"
            )

            return result

        except Exception as e:
            logger.error(f"Monthly trend analysis failed: {e}", exc_info=True)
            raise SegmentationError(f"Trend analysis failed: {e}") from e

    def analyze_quarterly_trends(
        self,
        emissions_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze quarterly emission trends.

        Args:
            emissions_data: Emission records with calculation_date

        Returns:
            Quarterly trend analysis result
        """
        try:
            logger.info(f"Analyzing quarterly trends for {len(emissions_data)} records")

            # Convert to quarterly periods
            quarterly_data = self._aggregate_by_quarter(emissions_data)

            if not quarterly_data:
                raise DataValidationError("No date data available for quarterly analysis")

            # Sort by period
            sorted_periods = sorted(quarterly_data.items())

            # Calculate metrics
            periods = [p[0] for p in sorted_periods]
            emissions = [p[1]["emissions_tco2e"] for p in sorted_periods]

            # Growth rate (quarter-over-quarter)
            growth_rate = self._calculate_growth_rate(emissions)

            # Year-over-year if applicable
            yoy_growth = self._calculate_yoy_growth(quarterly_data)

            result = {
                "analysis_type": "quarterly_trends",
                "n_periods": len(periods),
                "periods": periods,
                "emissions_by_period": emissions,
                "total_emissions_tco2e": sum(emissions),
                "avg_quarterly_emissions_tco2e": mean(emissions),
                "qoq_growth_rate_pct": round(growth_rate, 2),
                "yoy_growth_rate_pct": yoy_growth,
                "chart_data": self._generate_trend_chart_data(periods, emissions)
            }

            logger.info(
                f"Quarterly trend analysis complete: {len(periods)} periods, "
                f"QoQ growth={growth_rate:.1f}%"
            )

            return result

        except Exception as e:
            logger.error(f"Quarterly trend analysis failed: {e}", exc_info=True)
            raise SegmentationError(f"Trend analysis failed: {e}") from e

    def compare_periods(
        self,
        current_period: List[Dict[str, Any]],
        previous_period: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare emissions between two time periods.

        Args:
            current_period: Current period emission records
            previous_period: Previous period emission records

        Returns:
            Period comparison result
        """
        current_total = sum(r.get("emissions_tco2e", 0) for r in current_period)
        previous_total = sum(r.get("emissions_tco2e", 0) for r in previous_period)

        # Calculate change
        absolute_change = current_total - previous_total
        percent_change = (
            (absolute_change / previous_total * 100)
            if previous_total > 0 else 0
        )

        return {
            "current_period_tco2e": round(current_total, 2),
            "previous_period_tco2e": round(previous_total, 2),
            "absolute_change_tco2e": round(absolute_change, 2),
            "percent_change": round(percent_change, 2),
            "trend": "increasing" if absolute_change > 0 else "decreasing" if absolute_change < 0 else "stable"
        }

    def _aggregate_by_period(
        self,
        emissions_data: List[Dict[str, Any]],
        period_field: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate emissions by time period.

        Args:
            emissions_data: Emission records
            period_field: Field containing period (e.g., time_period)

        Returns:
            Aggregated data by period
        """
        aggregation = defaultdict(lambda: {"emissions_tco2e": 0, "record_count": 0})

        for record in emissions_data:
            period = record.get(period_field)
            if not period:
                continue

            aggregation[period]["emissions_tco2e"] += record.get("emissions_tco2e", 0)
            aggregation[period]["record_count"] += 1

        return dict(aggregation)

    def _aggregate_by_quarter(
        self,
        emissions_data: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate emissions by quarter from calculation_date.

        Args:
            emissions_data: Emission records with calculation_date

        Returns:
            Aggregated data by quarter
        """
        aggregation = defaultdict(lambda: {"emissions_tco2e": 0, "record_count": 0})

        for record in emissions_data:
            calc_date = record.get("calculation_date")
            if not calc_date:
                continue

            # Convert to datetime if string
            if isinstance(calc_date, str):
                try:
                    calc_date = datetime.fromisoformat(calc_date.replace('Z', '+00:00'))
                except Exception:
                    continue

            # Generate quarter key (e.g., "2025-Q1")
            quarter = (calc_date.month - 1) // 3 + 1
            quarter_key = f"{calc_date.year}-Q{quarter}"

            aggregation[quarter_key]["emissions_tco2e"] += record.get("emissions_tco2e", 0)
            aggregation[quarter_key]["record_count"] += 1

        return dict(aggregation)

    def _calculate_growth_rate(self, emissions: List[float]) -> float:
        """
        Calculate average growth rate across periods.

        Args:
            emissions: List of emission values

        Returns:
            Average growth rate as percentage
        """
        if len(emissions) < 2:
            return 0.0

        # Calculate period-over-period growth rates
        growth_rates = []
        for i in range(1, len(emissions)):
            if emissions[i - 1] > 0:
                rate = ((emissions[i] - emissions[i - 1]) / emissions[i - 1]) * 100
                growth_rates.append(rate)

        return mean(growth_rates) if growth_rates else 0.0

    def _calculate_yoy_growth(
        self,
        quarterly_data: Dict[str, Dict[str, float]]
    ) -> Optional[float]:
        """
        Calculate year-over-year growth rate.

        Args:
            quarterly_data: Data aggregated by quarter

        Returns:
            YoY growth rate or None if not enough data
        """
        if len(quarterly_data) < 4:
            return None

        # Get sorted quarters
        sorted_quarters = sorted(quarterly_data.items())

        # Compare Q1 to Q1, Q2 to Q2, etc. from different years
        yoy_rates = []
        for i in range(4, len(sorted_quarters)):
            current_emissions = sorted_quarters[i][1]["emissions_tco2e"]
            previous_year_emissions = sorted_quarters[i - 4][1]["emissions_tco2e"]

            if previous_year_emissions > 0:
                rate = ((current_emissions - previous_year_emissions) / previous_year_emissions) * 100
                yoy_rates.append(rate)

        return round(mean(yoy_rates), 2) if yoy_rates else None

    def _determine_trend(self, emissions: List[float]) -> str:
        """
        Determine overall trend direction.

        Args:
            emissions: List of emission values

        Returns:
            Trend direction (increasing, decreasing, stable)
        """
        if len(emissions) < 2:
            return "stable"

        # Simple linear trend
        first_half_avg = mean(emissions[:len(emissions)//2])
        second_half_avg = mean(emissions[len(emissions)//2:])

        diff_pct = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0

        if diff_pct > 5:
            return "increasing"
        elif diff_pct < -5:
            return "decreasing"
        else:
            return "stable"

    def _generate_trend_chart_data(
        self,
        periods: List[str],
        emissions: List[float]
    ) -> Dict[str, Any]:
        """
        Generate data for trend chart visualization.

        Args:
            periods: Time periods
            emissions: Emission values

        Returns:
            Chart data
        """
        return {
            "chart_type": "line",
            "title": "Emissions Trend Over Time",
            "data": [
                {
                    "period": period,
                    "value": round(emission, 2)
                }
                for period, emission in zip(periods, emissions)
            ],
            "x_axis": "Time Period",
            "y_axis": "Emissions (tCO2e)"
        }


__all__ = ["TrendAnalyzer"]
