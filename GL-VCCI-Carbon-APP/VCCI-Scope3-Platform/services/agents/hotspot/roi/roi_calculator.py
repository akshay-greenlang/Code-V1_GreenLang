"""
ROI Calculator
GL-VCCI Scope 3 Platform

Calculate return on investment for emission reduction initiatives.
Includes NPV, IRR, payback period, and carbon value analysis.

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

import logging
from typing import Optional
import numpy as np

from ..models import Initiative, ROIAnalysis
from ..config import ROIConfig
from ..exceptions import ROICalculationError

logger = logging.getLogger(__name__)


class ROICalculator:
    """
    ROI calculator for emission reduction initiatives.

    Calculates:
    - Cost per tCO2e reduced
    - Payback period
    - Net Present Value (NPV)
    - Internal Rate of Return (IRR)
    - Carbon value
    """

    def __init__(self, config: Optional[ROIConfig] = None):
        """
        Initialize ROI calculator.

        Args:
            config: ROI calculation configuration
        """
        self.config = config or ROIConfig()
        logger.info(
            f"Initialized ROICalculator with discount_rate={self.config.discount_rate}, "
            f"carbon_price=${self.config.carbon_price_usd_per_tco2e}/tCO2e"
        )

    def calculate(self, initiative: Initiative) -> ROIAnalysis:
        """
        Calculate comprehensive ROI for initiative.

        Args:
            initiative: Emission reduction initiative

        Returns:
            ROIAnalysis with all metrics

        Raises:
            ROICalculationError: If calculation fails
        """
        try:
            logger.info(
                f"Calculating ROI for initiative: {initiative.name}, "
                f"reduction={initiative.reduction_potential_tco2e:.1f} tCO2e"
            )

            # Validate initiative
            self._validate_initiative(initiative)

            # Calculate cost per tCO2e
            roi_usd_per_tco2e = self._calculate_cost_per_tco2e(initiative)

            # Calculate payback period
            payback_period = self._calculate_payback_period(initiative)

            # Calculate NPV
            npv = self._calculate_npv(initiative)

            # Calculate IRR
            irr = self._calculate_irr(initiative)

            # Calculate carbon value
            carbon_value = self._calculate_carbon_value(initiative)

            result = ROIAnalysis(
                initiative=initiative,
                roi_usd_per_tco2e=round(roi_usd_per_tco2e, 2),
                payback_period_years=round(payback_period, 2) if payback_period else None,
                annual_savings_usd=round(initiative.annual_savings_usd, 2),
                npv_10y_usd=round(npv, 2),
                irr=round(irr, 4) if irr else None,
                carbon_value_usd=round(carbon_value, 2),
                discount_rate=self.config.discount_rate,
                carbon_price_usd_per_tco2e=self.config.carbon_price_usd_per_tco2e,
                analysis_period_years=self.config.analysis_period_years
            )

            logger.info(
                f"ROI calculation complete: ${roi_usd_per_tco2e:.2f}/tCO2e, "
                f"NPV=${npv:,.0f}, payback={payback_period:.1f}y" if payback_period else "NPV negative"
            )

            return result

        except Exception as e:
            logger.error(f"ROI calculation failed: {e}", exc_info=True)
            raise ROICalculationError(f"ROI calculation failed: {e}") from e

    def _validate_initiative(self, initiative: Initiative) -> None:
        """
        Validate initiative data.

        Args:
            initiative: Initiative to validate

        Raises:
            ROICalculationError: If invalid
        """
        if initiative.reduction_potential_tco2e <= 0:
            raise ROICalculationError("Reduction potential must be positive")

        if initiative.implementation_cost_usd < 0:
            raise ROICalculationError("Implementation cost must be non-negative")

    def _calculate_cost_per_tco2e(self, initiative: Initiative) -> float:
        """
        Calculate cost per tCO2e reduced.

        Args:
            initiative: Initiative

        Returns:
            Cost per tCO2e (USD/tCO2e)
        """
        if initiative.reduction_potential_tco2e == 0:
            return float('inf')

        return initiative.implementation_cost_usd / initiative.reduction_potential_tco2e

    def _calculate_payback_period(self, initiative: Initiative) -> Optional[float]:
        """
        Calculate simple payback period.

        Args:
            initiative: Initiative

        Returns:
            Payback period in years or None if no payback
        """
        if initiative.implementation_cost_usd == 0:
            return 0.0

        # Calculate annual benefits
        annual_benefit = initiative.annual_savings_usd

        # Add carbon value as benefit
        annual_carbon_benefit = (
            initiative.reduction_potential_tco2e *
            self.config.carbon_price_usd_per_tco2e
        )
        annual_benefit += annual_carbon_benefit

        # Subtract annual operating costs
        annual_benefit -= initiative.annual_operating_cost_usd

        if annual_benefit <= 0:
            return None  # No payback

        payback = initiative.implementation_cost_usd / annual_benefit
        return payback

    def _calculate_npv(self, initiative: Initiative) -> float:
        """
        Calculate Net Present Value.

        Args:
            initiative: Initiative

        Returns:
            NPV over analysis period
        """
        discount_rate = self.config.discount_rate
        n_years = self.config.analysis_period_years

        # Initial investment (negative)
        cash_flows = [-initiative.implementation_cost_usd]

        # Annual cash flows
        for year in range(1, n_years + 1):
            annual_cf = 0.0

            # Operating savings
            annual_cf += initiative.annual_savings_usd

            # Carbon value
            annual_cf += (
                initiative.reduction_potential_tco2e *
                self.config.carbon_price_usd_per_tco2e
            )

            # Operating costs
            annual_cf -= initiative.annual_operating_cost_usd

            # Discount
            discounted_cf = annual_cf / ((1 + discount_rate) ** year)
            cash_flows.append(discounted_cf)

        npv = sum(cash_flows)
        return npv

    def _calculate_irr(self, initiative: Initiative) -> Optional[float]:
        """
        Calculate Internal Rate of Return.

        Args:
            initiative: Initiative

        Returns:
            IRR or None if cannot be calculated
        """
        n_years = self.config.analysis_period_years

        # Build cash flow array
        cash_flows = [-initiative.implementation_cost_usd]

        # Annual cash flows
        annual_cf = (
            initiative.annual_savings_usd +
            (initiative.reduction_potential_tco2e * self.config.carbon_price_usd_per_tco2e) -
            initiative.annual_operating_cost_usd
        )

        for _ in range(n_years):
            cash_flows.append(annual_cf)

        # Calculate IRR using numpy
        try:
            irr = np.irr(cash_flows)
            if np.isnan(irr) or np.isinf(irr):
                return None
            return irr
        except Exception as e:
            logger.warning(f"IRR calculation failed: {e}")
            return None

    def _calculate_carbon_value(self, initiative: Initiative) -> float:
        """
        Calculate total carbon value.

        Args:
            initiative: Initiative

        Returns:
            Total carbon value (USD)
        """
        return (
            initiative.reduction_potential_tco2e *
            self.config.carbon_price_usd_per_tco2e
        )


__all__ = ["ROICalculator"]
