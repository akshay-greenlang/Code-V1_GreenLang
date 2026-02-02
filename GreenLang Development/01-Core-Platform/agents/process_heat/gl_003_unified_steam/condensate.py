"""
GL-003 UNIFIED STEAM SYSTEM OPTIMIZER - Condensate Return Optimization Module

This module provides condensate return system optimization including:
- Return rate analysis and improvement
- Temperature maximization for fuel savings
- Quality monitoring for contamination
- Steam trap survey integration
- Economic analysis of recovery improvements

Features:
    - Condensate return rate tracking
    - Heat recovery calculations
    - Contamination detection
    - Trap loss estimation
    - Fuel savings analysis

Example:
    >>> from greenlang.agents.process_heat.gl_003_unified_steam.condensate import (
    ...     CondensateReturnOptimizer,
    ... )
    >>>
    >>> optimizer = CondensateReturnOptimizer(config)
    >>> analysis = optimizer.analyze_return_system(readings)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field

from .config import CondensateConfig, SteamTrapSurveyConfig
from .schemas import (
    CondensateReading,
    CondensateReturnAnalysis,
    SteamTrapReading,
    TrapSurveyAnalysis,
    TrapStatus,
    OptimizationRecommendation,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class CondensateConstants:
    """Constants for condensate calculations."""

    # Specific heat of water (BTU/lb-F)
    CP_WATER = 1.0

    # Makeup water temperature (typical, F)
    MAKEUP_WATER_TEMP_F = 60.0

    # Boiler feedwater typical temperature (F)
    FEEDWATER_TEMP_F = 227.0  # For 5 psig DA

    # Economic factors
    BOILER_EFFICIENCY = 0.82  # 82% typical
    OPERATING_HOURS_YEAR = 8000

    # Steam trap loss estimates (lb/hr per trap at 100 psig)
    TRAP_LOSS_ESTIMATES = {
        "failed_open_small": 25.0,     # 1/2" trap
        "failed_open_medium": 50.0,    # 3/4" trap
        "failed_open_large": 100.0,    # 1" trap
        "leaking_small": 5.0,
        "leaking_medium": 10.0,
        "leaking_large": 20.0,
    }


class ContaminationLimits:
    """Contamination limits for condensate acceptance."""

    # TDS limits (ppm)
    MAX_TDS_CLEAN = 50.0
    MAX_TDS_ACCEPTABLE = 100.0
    MAX_TDS_MARGINAL = 200.0

    # Oil limits (ppm)
    MAX_OIL = 1.0
    WARNING_OIL = 0.5

    # Iron limits (ppb)
    MAX_IRON = 100.0
    WARNING_IRON = 50.0

    # pH limits
    MIN_PH = 8.5
    MAX_PH = 9.5


# =============================================================================
# CONDENSATE HEAT RECOVERY CALCULATOR
# =============================================================================

class CondensateHeatCalculator:
    """
    Calculator for condensate heat recovery analysis.

    Calculates heat content of returned condensate and
    potential savings from improved return rates.
    """

    def __init__(
        self,
        boiler_efficiency: float = 0.82,
        fuel_cost_per_mmbtu: float = 5.0,
    ) -> None:
        """
        Initialize heat calculator.

        Args:
            boiler_efficiency: Boiler combustion efficiency
            fuel_cost_per_mmbtu: Fuel cost ($/MMBTU)
        """
        self.boiler_efficiency = boiler_efficiency
        self.fuel_cost_per_mmbtu = fuel_cost_per_mmbtu

        logger.debug(
            f"CondensateHeatCalculator initialized: "
            f"eff={boiler_efficiency:.0%}, fuel=${fuel_cost_per_mmbtu}/MMBTU"
        )

    def calculate_heat_content(
        self,
        flow_rate_lb_hr: float,
        temperature_f: float,
        reference_temp_f: float = 60.0,
    ) -> float:
        """
        Calculate heat content of condensate stream.

        Args:
            flow_rate_lb_hr: Condensate flow (lb/hr)
            temperature_f: Condensate temperature (F)
            reference_temp_f: Reference temperature (F)

        Returns:
            Heat content (BTU/hr)
        """
        delta_t = temperature_f - reference_temp_f
        heat_content = flow_rate_lb_hr * CondensateConstants.CP_WATER * delta_t
        return max(0, heat_content)

    def calculate_heat_recovered(
        self,
        condensate_flow_lb_hr: float,
        condensate_temp_f: float,
        feedwater_temp_f: float = 227.0,
        makeup_temp_f: float = 60.0,
    ) -> Tuple[float, float]:
        """
        Calculate heat recovered from condensate return.

        Args:
            condensate_flow_lb_hr: Condensate return flow (lb/hr)
            condensate_temp_f: Condensate temperature (F)
            feedwater_temp_f: Required feedwater temperature (F)
            makeup_temp_f: Makeup water temperature (F)

        Returns:
            Tuple of (heat_recovered_btu_hr, equivalent_fuel_saved_mmbtu_hr)
        """
        # Heat in condensate above makeup water temp
        condensate_heat = self.calculate_heat_content(
            condensate_flow_lb_hr,
            condensate_temp_f,
            makeup_temp_f
        )

        # Fuel equivalent (accounting for boiler efficiency)
        if self.boiler_efficiency > 0:
            fuel_equivalent = condensate_heat / (1_000_000 * self.boiler_efficiency)
        else:
            fuel_equivalent = 0

        return condensate_heat, fuel_equivalent

    def calculate_return_rate_savings(
        self,
        steam_flow_lb_hr: float,
        current_return_rate_pct: float,
        target_return_rate_pct: float,
        condensate_temp_f: float = 180.0,
        makeup_temp_f: float = 60.0,
    ) -> Dict[str, float]:
        """
        Calculate savings from improving condensate return rate.

        Args:
            steam_flow_lb_hr: Total steam production (lb/hr)
            current_return_rate_pct: Current return rate (%)
            target_return_rate_pct: Target return rate (%)
            condensate_temp_f: Condensate temperature (F)
            makeup_temp_f: Makeup water temperature (F)

        Returns:
            Dictionary with savings analysis
        """
        # Current condensate return
        current_return = steam_flow_lb_hr * (current_return_rate_pct / 100)
        target_return = steam_flow_lb_hr * (target_return_rate_pct / 100)
        additional_return = target_return - current_return

        if additional_return <= 0:
            return {
                "additional_return_lb_hr": 0,
                "additional_heat_btu_hr": 0,
                "fuel_savings_mmbtu_hr": 0,
                "fuel_savings_usd_hr": 0,
                "annual_savings_usd": 0,
            }

        # Heat in additional condensate
        additional_heat = self.calculate_heat_content(
            additional_return,
            condensate_temp_f,
            makeup_temp_f
        )

        # Fuel savings
        fuel_savings_mmbtu = additional_heat / (1_000_000 * self.boiler_efficiency)
        cost_savings_hr = fuel_savings_mmbtu * self.fuel_cost_per_mmbtu
        annual_savings = cost_savings_hr * CondensateConstants.OPERATING_HOURS_YEAR

        return {
            "additional_return_lb_hr": additional_return,
            "additional_heat_btu_hr": additional_heat,
            "fuel_savings_mmbtu_hr": fuel_savings_mmbtu,
            "fuel_savings_usd_hr": cost_savings_hr,
            "annual_savings_usd": annual_savings,
        }

    def calculate_temperature_optimization_savings(
        self,
        condensate_flow_lb_hr: float,
        current_temp_f: float,
        target_temp_f: float,
    ) -> Dict[str, float]:
        """
        Calculate savings from increasing condensate temperature.

        Args:
            condensate_flow_lb_hr: Condensate flow (lb/hr)
            current_temp_f: Current return temperature (F)
            target_temp_f: Target return temperature (F)

        Returns:
            Dictionary with temperature optimization savings
        """
        if target_temp_f <= current_temp_f:
            return {
                "temperature_increase_f": 0,
                "additional_heat_btu_hr": 0,
                "fuel_savings_mmbtu_hr": 0,
                "fuel_savings_usd_hr": 0,
                "annual_savings_usd": 0,
            }

        temp_increase = target_temp_f - current_temp_f
        additional_heat = condensate_flow_lb_hr * CondensateConstants.CP_WATER * temp_increase

        fuel_savings_mmbtu = additional_heat / (1_000_000 * self.boiler_efficiency)
        cost_savings_hr = fuel_savings_mmbtu * self.fuel_cost_per_mmbtu
        annual_savings = cost_savings_hr * CondensateConstants.OPERATING_HOURS_YEAR

        return {
            "temperature_increase_f": temp_increase,
            "additional_heat_btu_hr": additional_heat,
            "fuel_savings_mmbtu_hr": fuel_savings_mmbtu,
            "fuel_savings_usd_hr": cost_savings_hr,
            "annual_savings_usd": annual_savings,
        }


# =============================================================================
# CONDENSATE QUALITY ANALYZER
# =============================================================================

class CondensateQualityAnalyzer:
    """
    Analyzer for condensate quality and contamination detection.

    Monitors condensate for contamination that could damage
    boiler or water treatment systems.
    """

    def __init__(
        self,
        config: CondensateConfig,
    ) -> None:
        """
        Initialize quality analyzer.

        Args:
            config: Condensate configuration
        """
        self.config = config

        logger.debug("CondensateQualityAnalyzer initialized")

    def analyze_contamination(
        self,
        reading: CondensateReading,
    ) -> Dict[str, Any]:
        """
        Analyze condensate for contamination.

        Args:
            reading: Condensate quality reading

        Returns:
            Dictionary with contamination analysis
        """
        issues = []
        warnings = []
        is_contaminated = False
        disposition = "return"  # return, divert, investigate

        # Check TDS
        if reading.tds_ppm is not None:
            tds_limit = self.config.max_contamination_tds_ppm

            if reading.tds_ppm > ContaminationLimits.MAX_TDS_MARGINAL:
                issues.append(
                    f"TDS {reading.tds_ppm:.0f} ppm critically high"
                )
                is_contaminated = True
                disposition = "divert"
            elif reading.tds_ppm > tds_limit:
                issues.append(
                    f"TDS {reading.tds_ppm:.0f} ppm exceeds limit {tds_limit:.0f} ppm"
                )
                is_contaminated = True
                disposition = "investigate"
            elif reading.tds_ppm > tds_limit * 0.8:
                warnings.append(
                    f"TDS {reading.tds_ppm:.0f} ppm approaching limit"
                )

        # Check oil
        if reading.oil_ppm is not None:
            oil_limit = self.config.max_oil_ppm

            if reading.oil_ppm > ContaminationLimits.MAX_OIL:
                issues.append(
                    f"Oil {reading.oil_ppm:.2f} ppm - likely process leak"
                )
                is_contaminated = True
                disposition = "divert"
            elif reading.oil_ppm > oil_limit:
                issues.append(
                    f"Oil {reading.oil_ppm:.2f} ppm exceeds limit"
                )
                is_contaminated = True
                disposition = "investigate"
            elif reading.oil_ppm > ContaminationLimits.WARNING_OIL:
                warnings.append(
                    f"Oil {reading.oil_ppm:.2f} ppm - monitor closely"
                )

        # Check iron
        if reading.iron_ppb is not None:
            iron_limit = self.config.max_iron_ppb

            if reading.iron_ppb > ContaminationLimits.MAX_IRON:
                issues.append(
                    f"Iron {reading.iron_ppb:.0f} ppb indicates corrosion"
                )
            elif reading.iron_ppb > iron_limit:
                warnings.append(
                    f"Iron {reading.iron_ppb:.0f} ppb elevated"
                )

        # Check pH
        if reading.ph is not None:
            if reading.ph < ContaminationLimits.MIN_PH:
                issues.append(
                    f"pH {reading.ph:.1f} too low - acidic condensate"
                )
                is_contaminated = True
            elif reading.ph > ContaminationLimits.MAX_PH:
                warnings.append(
                    f"pH {reading.ph:.1f} higher than typical"
                )

        # Determine contamination source
        source_diagnosis = []
        if is_contaminated:
            if reading.oil_ppm and reading.oil_ppm > 0.5:
                source_diagnosis.append("Process leak (oil contamination)")
            if reading.tds_ppm and reading.tds_ppm > 200:
                source_diagnosis.append("Cooling water leak or process contamination")
            if reading.ph and reading.ph < 8.5:
                source_diagnosis.append("CO2 corrosion or acid leak")
            if reading.iron_ppb and reading.iron_ppb > 100:
                source_diagnosis.append("System corrosion")

        return {
            "is_contaminated": is_contaminated,
            "disposition": disposition,
            "issues": issues,
            "warnings": warnings,
            "source_diagnosis": source_diagnosis,
            "quality_score": self._calculate_quality_score(reading),
        }

    def _calculate_quality_score(
        self,
        reading: CondensateReading,
    ) -> float:
        """Calculate condensate quality score (0-100)."""
        score = 100.0
        deductions = 0

        # TDS deduction
        if reading.tds_ppm is not None:
            if reading.tds_ppm > 50:
                deductions += min(30, (reading.tds_ppm - 50) * 0.3)

        # Oil deduction
        if reading.oil_ppm is not None:
            if reading.oil_ppm > 0:
                deductions += min(40, reading.oil_ppm * 40)

        # Iron deduction
        if reading.iron_ppb is not None:
            if reading.iron_ppb > 50:
                deductions += min(20, (reading.iron_ppb - 50) * 0.2)

        # pH deduction
        if reading.ph is not None:
            if reading.ph < 8.5:
                deductions += (8.5 - reading.ph) * 10
            elif reading.ph > 9.5:
                deductions += (reading.ph - 9.5) * 5

        return max(0, score - deductions)


# =============================================================================
# STEAM TRAP SURVEY ANALYZER
# =============================================================================

class SteamTrapSurveyAnalyzer:
    """
    Analyzer for steam trap surveys.

    Processes trap survey data to identify losses and
    prioritize repairs for improved condensate return.
    """

    def __init__(
        self,
        config: SteamTrapSurveyConfig,
    ) -> None:
        """
        Initialize trap survey analyzer.

        Args:
            config: Steam trap survey configuration
        """
        self.config = config
        self.steam_cost_per_mlb = config.steam_cost_per_mlb

        logger.debug("SteamTrapSurveyAnalyzer initialized")

    def analyze_survey(
        self,
        trap_readings: List[SteamTrapReading],
    ) -> TrapSurveyAnalysis:
        """
        Analyze steam trap survey results.

        Args:
            trap_readings: List of trap readings

        Returns:
            TrapSurveyAnalysis with results and recommendations
        """
        total_traps = len(trap_readings)

        if total_traps == 0:
            return TrapSurveyAnalysis(
                total_traps=0,
                failure_rate_pct=0,
                provenance_hash=self._calculate_provenance_hash([]),
            )

        # Count by status
        operating_count = sum(
            1 for t in trap_readings if t.status == TrapStatus.OPERATING
        )
        failed_open_count = sum(
            1 for t in trap_readings if t.status == TrapStatus.FAILED_OPEN
        )
        failed_closed_count = sum(
            1 for t in trap_readings if t.status == TrapStatus.FAILED_CLOSED
        )
        leaking_count = sum(
            1 for t in trap_readings if t.status == TrapStatus.LEAKING
        )
        unknown_count = sum(
            1 for t in trap_readings if t.status == TrapStatus.UNKNOWN
        )

        # Calculate failure rates
        failed_count = failed_open_count + failed_closed_count + leaking_count
        failure_rate = (failed_count / total_traps * 100) if total_traps > 0 else 0
        failed_open_rate = (failed_open_count / total_traps * 100) if total_traps > 0 else 0

        # Calculate steam losses
        total_steam_loss = 0.0
        for trap in trap_readings:
            if trap.status in [TrapStatus.FAILED_OPEN, TrapStatus.LEAKING]:
                if trap.steam_loss_lb_hr:
                    total_steam_loss += trap.steam_loss_lb_hr
                else:
                    # Estimate based on size
                    total_steam_loss += self._estimate_trap_loss(trap)

        # Annual loss
        annual_loss_lb = total_steam_loss * CondensateConstants.OPERATING_HOURS_YEAR
        annual_loss_mlb = annual_loss_lb / 1000
        annual_cost = annual_loss_mlb * self.steam_cost_per_mlb

        # Identify priority repairs
        priority_repairs = self._identify_priority_repairs(trap_readings)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            failure_rate,
            failed_open_rate,
            total_steam_loss,
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(trap_readings)

        return TrapSurveyAnalysis(
            total_traps=total_traps,
            operating_count=operating_count,
            failed_open_count=failed_open_count,
            failed_closed_count=failed_closed_count,
            unknown_count=unknown_count,
            failure_rate_pct=failure_rate,
            failed_open_rate_pct=failed_open_rate,
            total_steam_loss_lb_hr=total_steam_loss,
            annual_steam_loss_mlb=annual_loss_mlb,
            annual_cost_usd=annual_cost,
            priority_repairs=priority_repairs,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
        )

    def _estimate_trap_loss(
        self,
        trap: SteamTrapReading,
    ) -> float:
        """Estimate steam loss for a failed trap."""
        # Base loss estimate by size
        size = trap.size_inches

        if trap.status == TrapStatus.FAILED_OPEN:
            if size <= 0.5:
                base_loss = CondensateConstants.TRAP_LOSS_ESTIMATES["failed_open_small"]
            elif size <= 0.75:
                base_loss = CondensateConstants.TRAP_LOSS_ESTIMATES["failed_open_medium"]
            else:
                base_loss = CondensateConstants.TRAP_LOSS_ESTIMATES["failed_open_large"]
        elif trap.status == TrapStatus.LEAKING:
            if size <= 0.5:
                base_loss = CondensateConstants.TRAP_LOSS_ESTIMATES["leaking_small"]
            elif size <= 0.75:
                base_loss = CondensateConstants.TRAP_LOSS_ESTIMATES["leaking_medium"]
            else:
                base_loss = CondensateConstants.TRAP_LOSS_ESTIMATES["leaking_large"]
        else:
            return 0.0

        # Adjust for pressure (base is 100 psig)
        if trap.inlet_pressure_psig > 0:
            pressure_factor = math.sqrt(trap.inlet_pressure_psig / 100)
        else:
            pressure_factor = 1.0

        return base_loss * pressure_factor

    def _identify_priority_repairs(
        self,
        trap_readings: List[SteamTrapReading],
    ) -> List[str]:
        """Identify traps needing priority repair."""
        priority_list = []

        for trap in trap_readings:
            if trap.status == TrapStatus.FAILED_OPEN:
                loss = trap.steam_loss_lb_hr or self._estimate_trap_loss(trap)
                priority_list.append(
                    f"{trap.trap_id} at {trap.location} - "
                    f"Failed open, ~{loss:.0f} lb/hr loss"
                )
            elif trap.status == TrapStatus.FAILED_CLOSED:
                priority_list.append(
                    f"{trap.trap_id} at {trap.location} - "
                    f"Failed closed, backup risk"
                )

        # Sort by implied loss (failed open first, then failed closed)
        return priority_list[:10]  # Top 10 priorities

    def _generate_recommendations(
        self,
        failure_rate: float,
        failed_open_rate: float,
        total_loss: float,
    ) -> List[str]:
        """Generate recommendations based on survey results."""
        recommendations = []

        if failure_rate > self.config.failed_open_threshold_pct + self.config.failed_closed_threshold_pct:
            recommendations.append(
                f"Overall failure rate {failure_rate:.1f}% is high - "
                "consider trap replacement program"
            )

        if failed_open_rate > self.config.failed_open_threshold_pct:
            recommendations.append(
                f"Failed-open rate {failed_open_rate:.1f}% - "
                "priority attention needed for steam losses"
            )

        if total_loss > 1000:
            recommendations.append(
                f"Total steam loss {total_loss:.0f} lb/hr - "
                "immediate repairs recommended"
            )

        if failure_rate < 5:
            recommendations.append(
                "Trap population in good condition - maintain current program"
            )

        return recommendations

    def _calculate_provenance_hash(
        self,
        trap_readings: List[SteamTrapReading],
    ) -> str:
        """Calculate provenance hash for survey."""
        data = {
            "trap_count": len(trap_readings),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trap_ids": sorted([t.trap_id for t in trap_readings]),
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


# =============================================================================
# CONDENSATE RETURN OPTIMIZER
# =============================================================================

class CondensateReturnOptimizer:
    """
    Condensate return system optimizer.

    Comprehensive analysis and optimization of condensate return
    including rate, temperature, quality, and trap losses.

    Example:
        >>> config = CondensateConfig()
        >>> optimizer = CondensateReturnOptimizer(config)
        >>>
        >>> readings = [CondensateReading(...), ...]
        >>> analysis = optimizer.analyze_return_system(
        ...     steam_flow_lb_hr=50000,
        ...     readings=readings,
        ... )
    """

    def __init__(
        self,
        config: CondensateConfig,
        trap_survey_config: Optional[SteamTrapSurveyConfig] = None,
        fuel_cost_per_mmbtu: float = 5.0,
        boiler_efficiency: float = 0.82,
    ) -> None:
        """
        Initialize condensate return optimizer.

        Args:
            config: Condensate configuration
            trap_survey_config: Steam trap survey configuration
            fuel_cost_per_mmbtu: Fuel cost ($/MMBTU)
            boiler_efficiency: Boiler efficiency
        """
        self.config = config
        self.trap_config = trap_survey_config or SteamTrapSurveyConfig()

        self.heat_calc = CondensateHeatCalculator(
            boiler_efficiency=boiler_efficiency,
            fuel_cost_per_mmbtu=fuel_cost_per_mmbtu,
        )
        self.quality_analyzer = CondensateQualityAnalyzer(config)
        self.trap_analyzer = SteamTrapSurveyAnalyzer(self.trap_config)

        self.fuel_cost = fuel_cost_per_mmbtu
        self.boiler_efficiency = boiler_efficiency

        logger.info(
            f"CondensateReturnOptimizer initialized: "
            f"target return {config.target_return_rate_pct}%"
        )

    def analyze_return_system(
        self,
        steam_flow_lb_hr: float,
        condensate_readings: List[CondensateReading],
        trap_readings: Optional[List[SteamTrapReading]] = None,
    ) -> CondensateReturnAnalysis:
        """
        Analyze condensate return system performance.

        Args:
            steam_flow_lb_hr: Total steam production (lb/hr)
            condensate_readings: Condensate return readings
            trap_readings: Optional trap survey readings

        Returns:
            CondensateReturnAnalysis with results
        """
        start_time = datetime.now(timezone.utc)
        recommendations = []
        warnings = []

        # Calculate total condensate return
        total_return = sum(r.flow_rate_lb_hr for r in condensate_readings)
        return_rate = (total_return / steam_flow_lb_hr * 100) if steam_flow_lb_hr > 0 else 0

        # Calculate weighted average temperature
        if total_return > 0:
            weighted_temp = sum(
                r.flow_rate_lb_hr * r.temperature_f
                for r in condensate_readings
            ) / total_return
        else:
            weighted_temp = CondensateConstants.MAKEUP_WATER_TEMP_F

        # Temperature shortfall
        temp_shortfall = None
        if weighted_temp < self.config.target_return_temp_f:
            temp_shortfall = self.config.target_return_temp_f - weighted_temp

        # Heat recovered
        heat_recovered, fuel_equiv = self.heat_calc.calculate_heat_recovered(
            condensate_flow_lb_hr=total_return,
            condensate_temp_f=weighted_temp,
        )

        # Calculate makeup water requirement
        makeup_required = steam_flow_lb_hr - total_return

        # Calculate potential additional recovery
        additional_recovery = 0.0
        if return_rate < self.config.target_return_rate_pct:
            savings = self.heat_calc.calculate_return_rate_savings(
                steam_flow_lb_hr=steam_flow_lb_hr,
                current_return_rate_pct=return_rate,
                target_return_rate_pct=self.config.target_return_rate_pct,
                condensate_temp_f=weighted_temp,
            )
            additional_recovery = savings["additional_heat_btu_hr"]

            recommendations.append(
                f"Increase return rate from {return_rate:.1f}% to "
                f"{self.config.target_return_rate_pct:.0f}% for "
                f"${savings['annual_savings_usd']:,.0f}/year savings"
            )

        # Check for low return rate
        if return_rate < self.config.min_acceptable_return_pct:
            warnings.append(
                f"Return rate {return_rate:.1f}% below minimum "
                f"{self.config.min_acceptable_return_pct:.0f}%"
            )

        # Check for low temperature
        if weighted_temp < self.config.min_return_temp_f:
            warnings.append(
                f"Return temperature {weighted_temp:.0f}F below minimum "
                f"{self.config.min_return_temp_f:.0f}F"
            )

            temp_savings = self.heat_calc.calculate_temperature_optimization_savings(
                condensate_flow_lb_hr=total_return,
                current_temp_f=weighted_temp,
                target_temp_f=self.config.target_return_temp_f,
            )
            recommendations.append(
                f"Improve condensate temperature for "
                f"${temp_savings['annual_savings_usd']:,.0f}/year savings"
            )

        # Analyze quality
        contaminated_sources = []
        for reading in condensate_readings:
            quality_result = self.quality_analyzer.analyze_contamination(reading)
            if quality_result["is_contaminated"]:
                contaminated_sources.append({
                    "location": reading.location_id,
                    "issues": quality_result["issues"],
                    "disposition": quality_result["disposition"],
                })
                warnings.append(
                    f"Contamination at {reading.location_id}: "
                    f"{', '.join(quality_result['issues'])}"
                )

        # Analyze traps if data provided
        trap_loss_recovery = 0.0
        if trap_readings and self.config.trap_survey_enabled:
            trap_analysis = self.trap_analyzer.analyze_survey(trap_readings)

            if trap_analysis.total_steam_loss_lb_hr > 0:
                warnings.append(
                    f"Steam trap losses: {trap_analysis.total_steam_loss_lb_hr:.0f} lb/hr "
                    f"(${trap_analysis.annual_cost_usd:,.0f}/year)"
                )
                recommendations.extend(trap_analysis.recommendations)

        # Calculate fuel savings
        fuel_savings_hr = fuel_equiv * self.fuel_cost if fuel_equiv > 0 else 0
        potential_savings = (
            additional_recovery / (1_000_000 * self.boiler_efficiency) * self.fuel_cost
        ) if additional_recovery > 0 else 0

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            steam_flow_lb_hr,
            condensate_readings,
        )

        return CondensateReturnAnalysis(
            timestamp=datetime.now(timezone.utc),
            total_steam_flow_lb_hr=steam_flow_lb_hr,
            condensate_return_lb_hr=total_return,
            return_rate_pct=return_rate,
            target_return_rate_pct=self.config.target_return_rate_pct,
            avg_return_temperature_f=weighted_temp,
            target_return_temperature_f=self.config.target_return_temp_f,
            temperature_shortfall_f=temp_shortfall,
            heat_recovered_btu_hr=heat_recovered,
            potential_additional_recovery_btu_hr=additional_recovery,
            makeup_water_required_lb_hr=makeup_required,
            fuel_savings_usd_hr=fuel_savings_hr,
            potential_additional_savings_usd_hr=potential_savings,
            recommendations=recommendations,
            warnings=warnings,
            provenance_hash=provenance_hash,
        )

    def get_optimization_recommendations(
        self,
        analysis: CondensateReturnAnalysis,
    ) -> List[OptimizationRecommendation]:
        """
        Generate detailed optimization recommendations.

        Args:
            analysis: Condensate return analysis

        Returns:
            List of prioritized recommendations
        """
        recommendations = []

        # Return rate improvement
        if analysis.return_rate_pct < self.config.target_return_rate_pct:
            gap = self.config.target_return_rate_pct - analysis.return_rate_pct
            annual_savings = analysis.potential_additional_savings_usd_hr * 8000

            recommendations.append(OptimizationRecommendation(
                category="condensate_recovery",
                priority=1 if gap > 15 else 2,
                description=(
                    f"Return rate {analysis.return_rate_pct:.1f}% is below "
                    f"target {self.config.target_return_rate_pct:.0f}%"
                ),
                action="Survey condensate collection points and repair/add return lines",
                energy_savings_pct=gap * 0.1,  # Approximate
                cost_savings_usd_year=annual_savings,
                implementation_cost_usd=10000 * gap,  # Rough estimate
                payback_months=12 * (10000 * gap) / max(1, annual_savings),
                complexity="medium",
                requires_shutdown=False,
            ))

        # Temperature improvement
        if analysis.temperature_shortfall_f and analysis.temperature_shortfall_f > 10:
            recommendations.append(OptimizationRecommendation(
                category="heat_recovery",
                priority=2,
                description=(
                    f"Return temperature {analysis.avg_return_temperature_f:.0f}F "
                    f"is {analysis.temperature_shortfall_f:.0f}F below target"
                ),
                action="Insulate condensate lines and reduce line lengths",
                energy_savings_pct=0.5,
                complexity="low",
                requires_shutdown=False,
            ))

        return recommendations

    def _calculate_provenance_hash(
        self,
        steam_flow: float,
        readings: List[CondensateReading],
    ) -> str:
        """Calculate provenance hash."""
        data = {
            "steam_flow_lb_hr": steam_flow,
            "reading_count": len(readings),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
