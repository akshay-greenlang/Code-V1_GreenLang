# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Economics Module

This module provides steam loss calculations and economic analysis for
steam trap failures. Implements DOE methodology for quantifying energy
losses and calculating repair ROI.

Features:
    - Steam loss rate calculation (lb/hr)
    - Energy loss quantification (MMBTU/yr)
    - Cost impact analysis ($/yr)
    - CO2 emissions impact
    - ROI and payback calculations
    - Plant-wide economic summaries

Calculations are ZERO-HALLUCINATION: All formulas are deterministic
engineering calculations based on DOE and Spirax Sarco methodologies.

Standards:
    - DOE Steam Tip Sheet #1: Inspect and Repair Steam Traps
    - DOE Steam System Assessment Handbook
    - Spirax Sarco Steam Loss Calculations

Example:
    >>> from greenlang.agents.process_heat.gl_008_steam_trap_monitor.economics import (
    ...     SteamLossCalculator,
    ...     EconomicAnalyzer,
    ... )
    >>> calculator = SteamLossCalculator(config)
    >>> loss = calculator.calculate_steam_loss(
    ...     orifice_diameter_in=0.25,
    ...     steam_pressure_psig=100,
    ... )
    >>> print(f"Steam loss: {loss.steam_loss_lb_hr:.1f} lb/hr")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_008_steam_trap_monitor.config import (
    SteamTrapMonitorConfig,
    EconomicsConfig,
    TrapType,
    FailureMode,
)
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.schemas import (
    TrapDiagnosticOutput,
    SteamLossEstimate,
    EconomicAnalysisOutput,
    TrapStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class SteamConstants:
    """Steam property constants for loss calculations."""

    # Standard orifice sizes for common trap failures (inches)
    ORIFICE_SIZES = {
        "small_leak": 0.0625,      # 1/16"
        "medium_leak": 0.125,      # 1/8"
        "large_leak": 0.1875,      # 3/16"
        "failed_open": 0.25,       # 1/4"
        "blowthrough": 0.375,      # 3/8"
    }

    # Napier's equation coefficient for steam flow through orifice
    # W = C * A * P (lb/hr)
    # where C = 51.45 (for steam, accounting for choked flow)
    NAPIER_COEFFICIENT = 51.45

    # Typical failure orifice sizes by trap type (inches)
    FAILURE_ORIFICE_BY_TYPE = {
        TrapType.FLOAT_THERMOSTATIC: 0.1875,
        TrapType.INVERTED_BUCKET: 0.1875,
        TrapType.THERMOSTATIC: 0.125,
        TrapType.THERMODYNAMIC: 0.0625,
        TrapType.BIMETALLIC: 0.125,
    }


# =============================================================================
# STEAM LOSS CALCULATOR
# =============================================================================

class SteamLossCalculator:
    """
    Calculator for steam trap steam losses.

    Uses Napier's equation for steam flow through an orifice:
        W = C * A * P

    Where:
        W = Steam flow rate (lb/hr)
        C = Flow coefficient (51.45 for steam)
        A = Orifice area (in^2)
        P = Upstream pressure (psia)

    This is a ZERO-HALLUCINATION calculation using established
    thermodynamic relationships.

    Example:
        >>> calculator = SteamLossCalculator(config)
        >>> loss = calculator.calculate_steam_loss(
        ...     orifice_diameter_in=0.25,
        ...     steam_pressure_psig=100,
        ... )
    """

    def __init__(self, config: EconomicsConfig) -> None:
        """
        Initialize steam loss calculator.

        Args:
            config: Economics configuration
        """
        self.config = config
        self._calculation_count = 0

        logger.info("SteamLossCalculator initialized")

    def calculate_steam_loss(
        self,
        orifice_diameter_in: float,
        steam_pressure_psig: float,
        operating_hours_per_year: Optional[int] = None,
    ) -> SteamLossEstimate:
        """
        Calculate steam loss rate for a failed trap.

        Uses Napier's equation: W = 51.45 * A * P

        Args:
            orifice_diameter_in: Equivalent orifice diameter (inches)
            steam_pressure_psig: Steam pressure (psig)
            operating_hours_per_year: Annual operating hours

        Returns:
            SteamLossEstimate with complete loss analysis
        """
        self._calculation_count += 1

        # Operating hours
        if operating_hours_per_year is None:
            operating_hours = self.config.operating_hours_per_year
        else:
            operating_hours = operating_hours_per_year

        # Convert pressure to psia
        pressure_psia = steam_pressure_psig + 14.696

        # Calculate orifice area (in^2)
        area_in2 = math.pi * (orifice_diameter_in / 2) ** 2

        # Napier's equation: W = 51.45 * A * P
        # where W = lb/hr, A = in^2, P = psia
        steam_loss_lb_hr = SteamConstants.NAPIER_COEFFICIENT * area_in2 * pressure_psia

        # Annual steam loss
        steam_loss_lb_year = steam_loss_lb_hr * operating_hours

        # Get steam enthalpy
        if self.config.steam_enthalpy_btu_lb:
            enthalpy = self.config.steam_enthalpy_btu_lb
        else:
            enthalpy = self._estimate_steam_enthalpy(steam_pressure_psig)

        # Energy loss
        energy_loss_btu_hr = steam_loss_lb_hr * enthalpy
        energy_loss_mmbtu_hr = energy_loss_btu_hr / 1_000_000
        energy_loss_mmbtu_year = energy_loss_mmbtu_hr * operating_hours

        # Cost calculation
        if self.config.steam_cost_per_mmbtu:
            cost_per_mmbtu = self.config.steam_cost_per_mmbtu
        else:
            # Convert from $/Mlb to $/MMBTU
            # 1 Mlb steam = ~1 MMBTU (approximately, depends on pressure)
            cost_per_mmbtu = self.config.steam_cost_per_mlb * enthalpy / 1000

        cost_per_hour = energy_loss_mmbtu_hr * cost_per_mmbtu
        cost_per_year = energy_loss_mmbtu_year * cost_per_mmbtu

        # CO2 emissions
        co2_lb_hr = (
            energy_loss_mmbtu_hr *
            self.config.co2_factor_lb_per_mmbtu /
            (self.config.boiler_efficiency_pct / 100)
        )
        co2_tons_year = (
            co2_lb_hr * operating_hours / 2000  # Convert lb to tons
        )

        logger.debug(
            f"Steam loss calculated: {steam_loss_lb_hr:.1f} lb/hr, "
            f"${cost_per_year:.0f}/year"
        )

        return SteamLossEstimate(
            steam_loss_lb_hr=round(steam_loss_lb_hr, 2),
            steam_loss_lb_year=round(steam_loss_lb_year, 0),
            energy_loss_mmbtu_hr=round(energy_loss_mmbtu_hr, 4),
            energy_loss_mmbtu_year=round(energy_loss_mmbtu_year, 2),
            cost_per_hour_usd=round(cost_per_hour, 2),
            cost_per_year_usd=round(cost_per_year, 2),
            co2_emissions_lb_hr=round(co2_lb_hr, 2),
            co2_emissions_tons_year=round(co2_tons_year, 2),
            calculation_method="napier_orifice",
            orifice_diameter_in=orifice_diameter_in,
            operating_hours_per_year=operating_hours,
        )

    def calculate_loss_for_status(
        self,
        status: TrapStatus,
        trap_type: TrapType,
        steam_pressure_psig: float,
        operating_hours_per_year: Optional[int] = None,
    ) -> SteamLossEstimate:
        """
        Calculate steam loss based on trap status and type.

        Uses typical orifice sizes for different failure modes.

        Args:
            status: Trap status (failure mode)
            trap_type: Type of steam trap
            steam_pressure_psig: Steam pressure (psig)
            operating_hours_per_year: Annual operating hours

        Returns:
            SteamLossEstimate
        """
        # Determine orifice size based on status and trap type
        if status == TrapStatus.GOOD:
            # No steam loss for good traps
            return SteamLossEstimate()

        elif status == TrapStatus.FAILED_OPEN:
            # Full orifice size for failed open
            orifice = SteamConstants.FAILURE_ORIFICE_BY_TYPE.get(
                trap_type,
                SteamConstants.ORIFICE_SIZES["failed_open"],
            )

        elif status == TrapStatus.LEAKING:
            # Smaller orifice for leaking (partial failure)
            base_orifice = SteamConstants.FAILURE_ORIFICE_BY_TYPE.get(
                trap_type,
                SteamConstants.ORIFICE_SIZES["medium_leak"],
            )
            orifice = base_orifice * 0.5  # 50% of failed open

        elif status == TrapStatus.FAILED_CLOSED:
            # No steam loss (but condensate backup risk)
            return SteamLossEstimate()

        else:
            # Unknown - assume small leak
            orifice = SteamConstants.ORIFICE_SIZES["small_leak"]

        return self.calculate_steam_loss(
            orifice_diameter_in=orifice,
            steam_pressure_psig=steam_pressure_psig,
            operating_hours_per_year=operating_hours_per_year,
        )

    def _estimate_steam_enthalpy(self, pressure_psig: float) -> float:
        """
        Estimate steam enthalpy from pressure.

        Uses simplified steam table correlation.

        Args:
            pressure_psig: Steam pressure (psig)

        Returns:
            Steam enthalpy (BTU/lb)
        """
        # Correlation based on saturated steam tables
        # h_g (BTU/lb) = approximately 1150-1200 for typical industrial pressures
        if pressure_psig <= 0:
            return 1150.0
        elif pressure_psig <= 50:
            return 1160.0 + (pressure_psig / 50) * 10
        elif pressure_psig <= 150:
            return 1170.0 + ((pressure_psig - 50) / 100) * 20
        elif pressure_psig <= 300:
            return 1190.0 + ((pressure_psig - 150) / 150) * 10
        else:
            return 1200.0

    @property
    def calculation_count(self) -> int:
        """Get calculation count."""
        return self._calculation_count


# =============================================================================
# ROI CALCULATOR
# =============================================================================

class ROICalculator:
    """
    Calculator for repair return on investment.

    Calculates payback period, NPV, and ROI for trap repairs.
    """

    def __init__(self, config: EconomicsConfig) -> None:
        """
        Initialize ROI calculator.

        Args:
            config: Economics configuration
        """
        self.config = config

    def calculate_simple_payback(
        self,
        repair_cost: float,
        annual_savings: float,
    ) -> Optional[float]:
        """
        Calculate simple payback period.

        Args:
            repair_cost: Total repair cost ($)
            annual_savings: Annual savings from repair ($)

        Returns:
            Payback period in months, or None if no savings
        """
        if annual_savings <= 0:
            return None

        payback_years = repair_cost / annual_savings
        payback_months = payback_years * 12

        return round(payback_months, 1)

    def calculate_roi(
        self,
        repair_cost: float,
        annual_savings: float,
        years: int = 1,
    ) -> Optional[float]:
        """
        Calculate return on investment.

        ROI = (Savings - Cost) / Cost * 100%

        Args:
            repair_cost: Total repair cost ($)
            annual_savings: Annual savings ($)
            years: Analysis period

        Returns:
            ROI percentage, or None if no cost
        """
        if repair_cost <= 0:
            return None

        total_savings = annual_savings * years
        roi = ((total_savings - repair_cost) / repair_cost) * 100

        return round(roi, 1)

    def calculate_npv(
        self,
        repair_cost: float,
        annual_savings: float,
        years: Optional[int] = None,
        discount_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate Net Present Value.

        NPV = Sum of [Cash_Flow / (1 + r)^t] - Initial_Cost

        Args:
            repair_cost: Initial repair cost ($)
            annual_savings: Annual cash savings ($)
            years: Analysis period (default from config)
            discount_rate: Discount rate (default from config)

        Returns:
            NPV in dollars
        """
        if years is None:
            years = self.config.analysis_period_years
        if discount_rate is None:
            discount_rate = self.config.discount_rate_pct / 100

        # Calculate present value of annual savings
        pv_savings = 0.0
        for t in range(1, years + 1):
            pv_savings += annual_savings / ((1 + discount_rate) ** t)

        npv = pv_savings - repair_cost

        return round(npv, 2)

    def analyze_repair_economics(
        self,
        steam_loss: SteamLossEstimate,
        repair_cost: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Complete economic analysis for a trap repair.

        Args:
            steam_loss: Steam loss estimate for failed trap
            repair_cost: Repair cost (default from config)

        Returns:
            Dictionary with economic metrics
        """
        if repair_cost is None:
            repair_cost = self.config.average_repair_cost_usd

        annual_savings = steam_loss.cost_per_year_usd

        payback = self.calculate_simple_payback(repair_cost, annual_savings)
        roi_1yr = self.calculate_roi(repair_cost, annual_savings, years=1)
        roi_5yr = self.calculate_roi(repair_cost, annual_savings, years=5)
        npv_5yr = self.calculate_npv(repair_cost, annual_savings, years=5)

        return {
            "repair_cost_usd": repair_cost,
            "annual_savings_usd": annual_savings,
            "payback_months": payback,
            "roi_1year_pct": roi_1yr,
            "roi_5year_pct": roi_5yr,
            "npv_5year_usd": npv_5yr,
            "co2_reduction_tons_year": steam_loss.co2_emissions_tons_year,
        }


# =============================================================================
# COST BENEFIT ANALYZER
# =============================================================================

class CostBenefitAnalyzer:
    """
    Analyzer for plant-wide cost-benefit analysis.

    Aggregates individual trap analyses to provide plant-wide
    economics and prioritization.
    """

    def __init__(self, config: EconomicsConfig) -> None:
        """
        Initialize cost-benefit analyzer.

        Args:
            config: Economics configuration
        """
        self.config = config
        self._steam_calc = SteamLossCalculator(config)
        self._roi_calc = ROICalculator(config)

    def analyze_trap_portfolio(
        self,
        diagnostics: List[TrapDiagnosticOutput],
        steam_pressure_psig: float,
    ) -> EconomicAnalysisOutput:
        """
        Analyze economics for a portfolio of traps.

        Args:
            diagnostics: List of trap diagnostic outputs
            steam_pressure_psig: Operating steam pressure

        Returns:
            EconomicAnalysisOutput with portfolio analysis
        """
        total_traps = len(diagnostics)
        failed_traps = []
        total_steam_loss_lb_hr = 0.0
        total_annual_loss = 0.0
        total_repair_cost = 0.0

        for diag in diagnostics:
            status = TrapStatus(diag.condition.status)

            if status in [TrapStatus.FAILED_OPEN, TrapStatus.LEAKING]:
                # Calculate steam loss for this trap
                loss = self._steam_calc.calculate_loss_for_status(
                    status=status,
                    trap_type=TrapType.FLOAT_THERMOSTATIC,  # Default if unknown
                    steam_pressure_psig=steam_pressure_psig,
                )

                total_steam_loss_lb_hr += loss.steam_loss_lb_hr
                total_annual_loss += loss.cost_per_year_usd

                # Estimate repair cost
                if status == TrapStatus.FAILED_OPEN:
                    repair_cost = self.config.average_replacement_cost_usd
                else:
                    repair_cost = self.config.average_repair_cost_usd

                total_repair_cost += repair_cost

                failed_traps.append({
                    "trap_id": diag.trap_id,
                    "status": status.value,
                    "steam_loss_lb_hr": loss.steam_loss_lb_hr,
                    "annual_cost_usd": loss.cost_per_year_usd,
                    "repair_cost_usd": repair_cost,
                })

        # Sort failed traps by annual cost
        failed_traps.sort(key=lambda x: x["annual_cost_usd"], reverse=True)

        # Calculate overall metrics
        failure_rate = (len(failed_traps) / total_traps * 100) if total_traps > 0 else 0.0
        net_annual_savings = total_annual_loss - total_repair_cost

        # Calculate payback and ROI
        if total_annual_loss > 0:
            payback_months = total_repair_cost / total_annual_loss * 12
            roi_pct = ((total_annual_loss - total_repair_cost) / total_repair_cost * 100) if total_repair_cost > 0 else None
            npv_5yr = self._roi_calc.calculate_npv(
                total_repair_cost,
                total_annual_loss,
                years=5,
            )
        else:
            payback_months = None
            roi_pct = None
            npv_5yr = 0.0

        # CO2 impact
        total_co2_reduction = 0.0
        for trap in failed_traps:
            loss = self._steam_calc.calculate_loss_for_status(
                status=TrapStatus(trap["status"]),
                trap_type=TrapType.FLOAT_THERMOSTATIC,
                steam_pressure_psig=steam_pressure_psig,
            )
            total_co2_reduction += loss.co2_emissions_tons_year

        # Provenance
        provenance_data = {
            "total_traps": total_traps,
            "failed_count": len(failed_traps),
            "total_annual_loss": total_annual_loss,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return EconomicAnalysisOutput(
            request_id=hashlib.md5(
                f"portfolio_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12],
            total_traps_analyzed=total_traps,
            traps_failed=len(failed_traps),
            failure_rate_pct=round(failure_rate, 1),
            total_steam_loss_lb_hr=round(total_steam_loss_lb_hr, 2),
            total_steam_loss_lb_year=round(
                total_steam_loss_lb_hr * self.config.operating_hours_per_year, 0
            ),
            total_annual_loss_usd=round(total_annual_loss, 2),
            repair_cost_usd=round(total_repair_cost, 2),
            net_annual_savings_usd=round(net_annual_savings, 2),
            simple_payback_months=round(payback_months, 1) if payback_months else None,
            roi_pct=round(roi_pct, 1) if roi_pct else None,
            npv_5year_usd=npv_5yr,
            total_co2_reduction_tons_year=round(total_co2_reduction, 2),
            top_failing_traps=failed_traps[:10],  # Top 10
            provenance_hash=provenance_hash,
        )


# =============================================================================
# ECONOMIC ANALYZER (MAIN CLASS)
# =============================================================================

class EconomicAnalyzer:
    """
    Main economic analyzer for steam trap failures.

    Integrates steam loss calculations, ROI analysis, and cost-benefit
    analysis for individual traps and plant-wide portfolios.

    All calculations are ZERO-HALLUCINATION: deterministic engineering
    calculations based on DOE and Spirax Sarco methodologies.

    Example:
        >>> analyzer = EconomicAnalyzer(config.economics)
        >>> loss = analyzer.calculate_trap_loss(
        ...     status=TrapStatus.FAILED_OPEN,
        ...     trap_type=TrapType.INVERTED_BUCKET,
        ...     steam_pressure_psig=150,
        ... )
        >>> roi = analyzer.calculate_repair_roi(loss)
    """

    def __init__(self, config: EconomicsConfig) -> None:
        """
        Initialize economic analyzer.

        Args:
            config: Economics configuration
        """
        self.config = config
        self._steam_calc = SteamLossCalculator(config)
        self._roi_calc = ROICalculator(config)
        self._cb_analyzer = CostBenefitAnalyzer(config)
        self._analysis_count = 0

        logger.info("EconomicAnalyzer initialized")

    def calculate_trap_loss(
        self,
        status: TrapStatus,
        trap_type: TrapType,
        steam_pressure_psig: float,
        orifice_diameter_in: Optional[float] = None,
    ) -> SteamLossEstimate:
        """
        Calculate steam loss for a single trap.

        Args:
            status: Trap status (failure mode)
            trap_type: Type of steam trap
            steam_pressure_psig: Steam pressure (psig)
            orifice_diameter_in: Custom orifice size (optional)

        Returns:
            SteamLossEstimate
        """
        self._analysis_count += 1

        if orifice_diameter_in is not None:
            return self._steam_calc.calculate_steam_loss(
                orifice_diameter_in=orifice_diameter_in,
                steam_pressure_psig=steam_pressure_psig,
            )
        else:
            return self._steam_calc.calculate_loss_for_status(
                status=status,
                trap_type=trap_type,
                steam_pressure_psig=steam_pressure_psig,
            )

    def calculate_repair_roi(
        self,
        steam_loss: SteamLossEstimate,
        repair_cost: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate ROI for repairing a trap.

        Args:
            steam_loss: Steam loss estimate
            repair_cost: Custom repair cost (optional)

        Returns:
            Dictionary with ROI metrics
        """
        self._analysis_count += 1

        return self._roi_calc.analyze_repair_economics(
            steam_loss=steam_loss,
            repair_cost=repair_cost,
        )

    def analyze_portfolio(
        self,
        diagnostics: List[TrapDiagnosticOutput],
        steam_pressure_psig: float,
    ) -> EconomicAnalysisOutput:
        """
        Analyze economics for multiple traps.

        Args:
            diagnostics: List of trap diagnostics
            steam_pressure_psig: Steam pressure

        Returns:
            EconomicAnalysisOutput with portfolio analysis
        """
        self._analysis_count += 1

        return self._cb_analyzer.analyze_trap_portfolio(
            diagnostics=diagnostics,
            steam_pressure_psig=steam_pressure_psig,
        )

    def prioritize_repairs(
        self,
        diagnostics: List[TrapDiagnosticOutput],
        steam_pressure_psig: float,
        max_budget: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prioritize trap repairs by economic impact.

        Uses benefit-cost ratio to rank repairs.

        Args:
            diagnostics: List of trap diagnostics
            steam_pressure_psig: Steam pressure
            max_budget: Maximum budget constraint

        Returns:
            List of prioritized repairs
        """
        self._analysis_count += 1

        repairs = []

        for diag in diagnostics:
            status = TrapStatus(diag.condition.status)

            if status not in [TrapStatus.FAILED_OPEN, TrapStatus.LEAKING, TrapStatus.FAILED_CLOSED]:
                continue

            # Calculate economics
            loss = self._steam_calc.calculate_loss_for_status(
                status=status,
                trap_type=TrapType.FLOAT_THERMOSTATIC,
                steam_pressure_psig=steam_pressure_psig,
            )

            if status == TrapStatus.FAILED_OPEN:
                repair_cost = self.config.average_replacement_cost_usd
            else:
                repair_cost = self.config.average_repair_cost_usd

            # Benefit-cost ratio
            annual_benefit = loss.cost_per_year_usd
            bcr = annual_benefit / repair_cost if repair_cost > 0 else 0

            repairs.append({
                "trap_id": diag.trap_id,
                "status": status.value,
                "annual_savings_usd": annual_benefit,
                "repair_cost_usd": repair_cost,
                "benefit_cost_ratio": round(bcr, 2),
                "steam_loss_lb_hr": loss.steam_loss_lb_hr,
                "co2_reduction_tons_year": loss.co2_emissions_tons_year,
            })

        # Sort by benefit-cost ratio
        repairs.sort(key=lambda x: x["benefit_cost_ratio"], reverse=True)

        # Apply budget constraint if specified
        if max_budget is not None:
            cumulative_cost = 0.0
            filtered_repairs = []
            for repair in repairs:
                if cumulative_cost + repair["repair_cost_usd"] <= max_budget:
                    filtered_repairs.append(repair)
                    cumulative_cost += repair["repair_cost_usd"]
            repairs = filtered_repairs

        return repairs

    @property
    def analysis_count(self) -> int:
        """Get analysis count."""
        return self._analysis_count
