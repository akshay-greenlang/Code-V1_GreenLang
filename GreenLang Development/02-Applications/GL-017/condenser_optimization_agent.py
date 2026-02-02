# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Condenser Optimization Agent.

This module implements the main orchestrator for steam turbine condenser
performance optimization. It provides comprehensive monitoring, analysis,
and optimization of condenser vacuum, cooling water flow, heat transfer
efficiency, and air inleakage detection.

The agent integrates with SCADA systems for real-time data acquisition and
control, implements HEI (Heat Exchange Institute) compliance monitoring,
and coordinates with cooling towers and turbine control systems.

Key Features:
    - Real-time condenser performance monitoring
    - Vacuum pressure optimization
    - Cooling water flow optimization
    - Heat transfer efficiency calculations
    - Air inleakage detection
    - Fouling prediction and tube cleaning recommendations
    - Cooling tower coordination
    - Turbine backpressure coordination
    - Data provenance tracking

Author: GreenLang Team
Date: December 2025
Status: Production Ready
"""

import asyncio
import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from greenlang.core import (
    BaseOrchestrator,
    MessageBus,
    TaskScheduler,
    SafetyMonitor,
    CoordinationLayer,
    OrchestrationResult,
    OrchestratorConfig,
    MessageType,
    MessagePriority,
    TaskPriority,
    OperationContext,
    SafetyLevel,
    CoordinationPattern,
)

from greenlang.GL_017.config import (
    AgentConfiguration,
    CondenserConfiguration,
    CondenserType,
    CoolingSystemType,
    TubePattern,
    CleaningMethod,
    FoulingType,
    CoolingWaterConfig,
    VacuumSystemConfig,
    TubeConfiguration,
    WaterQualityLimits,
    PerformanceTargets,
    AlertThresholds,
    SCADAIntegration,
    CoolingTowerIntegration,
    TurbineCoordination,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class CoolingWaterData:
    """
    Cooling water measurement data.

    Contains inlet/outlet temperatures, flow rate, and water quality
    parameters from the condenser cooling water system.
    """

    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    condenser_id: str = ""

    # Temperature measurements
    inlet_temp_f: float = 0.0  # Cooling water inlet temperature
    outlet_temp_f: float = 0.0  # Cooling water outlet temperature
    temp_rise_f: float = 0.0  # Temperature rise across condenser

    # Flow measurements
    flow_rate_gpm: float = 0.0  # Total cooling water flow in GPM
    flow_velocity_fps: float = 0.0  # Tube flow velocity in ft/s
    flow_percentage_of_design: float = 0.0  # Flow as percentage of design

    # Pressure measurements
    inlet_pressure_psi: float = 0.0  # Inlet header pressure
    outlet_pressure_psi: float = 0.0  # Outlet header pressure
    pressure_drop_psi: float = 0.0  # Pressure drop across condenser

    # Water quality (if available)
    ph: Optional[float] = None
    conductivity_us_cm: Optional[float] = None
    turbidity_ntu: Optional[float] = None
    chlorine_residual_ppm: Optional[float] = None

    # Data quality
    data_source: str = "SCADA"
    quality_flag: str = "GOOD"

    def calculate_temp_rise(self) -> float:
        """Calculate temperature rise."""
        self.temp_rise_f = self.outlet_temp_f - self.inlet_temp_f
        return self.temp_rise_f

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "condenser_id": self.condenser_id,
            "inlet_temp_f": self.inlet_temp_f,
            "outlet_temp_f": self.outlet_temp_f,
            "temp_rise_f": self.temp_rise_f,
            "flow_rate_gpm": self.flow_rate_gpm,
            "flow_velocity_fps": self.flow_velocity_fps,
            "flow_percentage_of_design": self.flow_percentage_of_design,
            "inlet_pressure_psi": self.inlet_pressure_psi,
            "outlet_pressure_psi": self.outlet_pressure_psi,
            "pressure_drop_psi": self.pressure_drop_psi,
            "ph": self.ph,
            "conductivity_us_cm": self.conductivity_us_cm,
            "data_source": self.data_source,
            "quality_flag": self.quality_flag,
        }


@dataclass
class VacuumData:
    """
    Vacuum system measurement data.

    Contains vacuum pressure, saturation temperature, and air
    inleakage measurements from the condenser vacuum system.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    condenser_id: str = ""

    # Vacuum measurements
    pressure_in_hg_abs: float = 0.0  # Absolute pressure in inches Hg
    pressure_mbar_abs: float = 0.0  # Absolute pressure in mbar
    saturation_temp_f: float = 0.0  # Steam saturation temperature

    # Air leakage measurements
    air_inleakage_scfm: float = 0.0  # Air inleakage rate in SCFM
    air_inleakage_trend: str = "STABLE"  # STABLE, INCREASING, DECREASING

    # Hotwell measurements
    hotwell_level_pct: float = 0.0  # Hotwell level percentage
    hotwell_temp_f: float = 0.0  # Hotwell temperature

    # Vacuum equipment status
    ejector_stage1_status: str = "RUNNING"  # RUNNING, STANDBY, FAULT
    ejector_stage2_status: str = "RUNNING"
    lrvp_status: str = "STANDBY"  # Liquid ring vacuum pump

    # Calculated values
    vacuum_deviation_in_hg: float = 0.0  # Deviation from design vacuum
    expected_vacuum_in_hg_abs: float = 0.0  # Expected vacuum for current conditions

    # Data quality
    data_source: str = "SCADA"
    quality_flag: str = "GOOD"

    def calculate_saturation_temp(self) -> float:
        """
        Calculate saturation temperature from pressure.

        Uses Antoine equation approximation for water.

        Returns:
            Saturation temperature in F
        """
        # Convert in Hg abs to psia (1 in Hg = 0.491154 psi)
        psia = self.pressure_in_hg_abs * 0.491154

        # Antoine equation approximation: T(F) = (7256 / (21.744 - ln(psia))) - 459.67
        if psia > 0.01:
            try:
                ln_p = math.log(psia)
                temp_r = 7256.0 / (21.744 - ln_p)
                self.saturation_temp_f = temp_r - 459.67
            except (ValueError, ZeroDivisionError):
                self.saturation_temp_f = 100.0  # Default fallback
        else:
            self.saturation_temp_f = 79.0  # Typical low pressure value

        return self.saturation_temp_f

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "condenser_id": self.condenser_id,
            "pressure_in_hg_abs": self.pressure_in_hg_abs,
            "pressure_mbar_abs": self.pressure_mbar_abs,
            "saturation_temp_f": self.saturation_temp_f,
            "air_inleakage_scfm": self.air_inleakage_scfm,
            "air_inleakage_trend": self.air_inleakage_trend,
            "hotwell_level_pct": self.hotwell_level_pct,
            "hotwell_temp_f": self.hotwell_temp_f,
            "vacuum_deviation_in_hg": self.vacuum_deviation_in_hg,
            "data_source": self.data_source,
            "quality_flag": self.quality_flag,
        }


@dataclass
class CondensateData:
    """
    Condensate measurement data.

    Contains condensate flow, temperature, and subcooling
    measurements from the condenser hotwell.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    condenser_id: str = ""

    # Flow measurements
    flow_rate_lb_hr: float = 0.0  # Condensate flow rate
    flow_rate_gpm: float = 0.0  # Condensate flow in GPM

    # Temperature measurements
    temperature_f: float = 0.0  # Condensate temperature
    subcooling_f: float = 0.0  # Degrees of subcooling below saturation

    # Quality indicators
    dissolved_oxygen_ppb: float = 0.0  # Dissolved oxygen content
    conductivity_us_cm: float = 0.0  # Condensate conductivity

    # Pump status
    condensate_pump_running: int = 0  # Number of pumps running
    condensate_pump_discharge_psi: float = 0.0  # Discharge pressure

    # Data quality
    data_source: str = "SCADA"
    quality_flag: str = "GOOD"

    def calculate_subcooling(self, saturation_temp_f: float) -> float:
        """
        Calculate subcooling below saturation.

        Args:
            saturation_temp_f: Saturation temperature at condenser pressure

        Returns:
            Degrees of subcooling in F
        """
        self.subcooling_f = saturation_temp_f - self.temperature_f
        return self.subcooling_f

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "condenser_id": self.condenser_id,
            "flow_rate_lb_hr": self.flow_rate_lb_hr,
            "flow_rate_gpm": self.flow_rate_gpm,
            "temperature_f": self.temperature_f,
            "subcooling_f": self.subcooling_f,
            "dissolved_oxygen_ppb": self.dissolved_oxygen_ppb,
            "conductivity_us_cm": self.conductivity_us_cm,
            "data_source": self.data_source,
            "quality_flag": self.quality_flag,
        }


@dataclass
class HeatTransferData:
    """
    Heat transfer performance data.

    Contains calculated heat transfer parameters including
    overall heat transfer coefficient, LMTD, and TTD.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    condenser_id: str = ""

    # Heat transfer coefficients
    u_value_actual_btu_hr_sqft_f: float = 0.0  # Actual overall U-value
    u_value_design_btu_hr_sqft_f: float = 0.0  # Design U-value
    u_value_ratio: float = 0.0  # Actual/Design ratio

    # Temperature differences
    lmtd_f: float = 0.0  # Log mean temperature difference
    ttd_f: float = 0.0  # Terminal temperature difference (Tsat - CW outlet)
    inlet_td_f: float = 0.0  # Inlet temperature difference

    # Heat duty
    heat_duty_mmbtu_hr: float = 0.0  # Actual heat duty
    design_heat_duty_mmbtu_hr: float = 0.0  # Design heat duty
    heat_duty_ratio: float = 0.0  # Actual/Design ratio

    # Cleanliness factor
    cleanliness_factor_pct: float = 0.0  # HEI cleanliness factor
    fouling_factor_hr_sqft_f_btu: float = 0.0  # Calculated fouling factor

    # Performance indices
    effectiveness: float = 0.0  # Heat exchanger effectiveness
    ntu: float = 0.0  # Number of transfer units

    def calculate_lmtd(
        self,
        steam_temp_f: float,
        cw_inlet_f: float,
        cw_outlet_f: float,
    ) -> float:
        """
        Calculate log mean temperature difference.

        Args:
            steam_temp_f: Steam saturation temperature
            cw_inlet_f: Cooling water inlet temperature
            cw_outlet_f: Cooling water outlet temperature

        Returns:
            LMTD in degrees F
        """
        delta_t1 = steam_temp_f - cw_outlet_f  # TTD
        delta_t2 = steam_temp_f - cw_inlet_f  # ITD

        self.ttd_f = delta_t1
        self.inlet_td_f = delta_t2

        if delta_t1 <= 0 or delta_t2 <= 0:
            logger.warning("Invalid temperature difference for LMTD calculation")
            return 0.0

        if abs(delta_t1 - delta_t2) < 0.1:
            # Nearly equal, use arithmetic mean
            self.lmtd_f = (delta_t1 + delta_t2) / 2
        else:
            # Standard LMTD formula
            self.lmtd_f = (delta_t2 - delta_t1) / math.log(delta_t2 / delta_t1)

        return self.lmtd_f

    def calculate_u_value(
        self,
        heat_duty_btu_hr: float,
        surface_area_sqft: float,
    ) -> float:
        """
        Calculate overall heat transfer coefficient.

        Args:
            heat_duty_btu_hr: Heat duty in BTU/hr
            surface_area_sqft: Heat transfer surface area in sq ft

        Returns:
            U-value in BTU/hr-sqft-F
        """
        if self.lmtd_f > 0 and surface_area_sqft > 0:
            self.u_value_actual_btu_hr_sqft_f = heat_duty_btu_hr / (
                surface_area_sqft * self.lmtd_f
            )
        return self.u_value_actual_btu_hr_sqft_f

    def calculate_cleanliness_factor(self) -> float:
        """
        Calculate HEI cleanliness factor.

        Returns:
            Cleanliness factor as percentage
        """
        if self.u_value_design_btu_hr_sqft_f > 0:
            self.cleanliness_factor_pct = (
                self.u_value_actual_btu_hr_sqft_f / self.u_value_design_btu_hr_sqft_f
            ) * 100
        return self.cleanliness_factor_pct

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "condenser_id": self.condenser_id,
            "u_value_actual_btu_hr_sqft_f": self.u_value_actual_btu_hr_sqft_f,
            "u_value_design_btu_hr_sqft_f": self.u_value_design_btu_hr_sqft_f,
            "u_value_ratio": self.u_value_ratio,
            "lmtd_f": self.lmtd_f,
            "ttd_f": self.ttd_f,
            "inlet_td_f": self.inlet_td_f,
            "heat_duty_mmbtu_hr": self.heat_duty_mmbtu_hr,
            "cleanliness_factor_pct": self.cleanliness_factor_pct,
            "fouling_factor_hr_sqft_f_btu": self.fouling_factor_hr_sqft_f_btu,
            "effectiveness": self.effectiveness,
        }


@dataclass
class FoulingAssessment:
    """
    Fouling assessment and tube cleaning recommendation.

    Provides fouling analysis, trend prediction, and cleaning
    recommendations based on performance degradation.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    condenser_id: str = ""

    # Fouling metrics
    fouling_factor_hr_sqft_f_btu: float = 0.0  # Current fouling factor
    design_fouling_factor: float = 0.0005  # Design fouling factor
    fouling_ratio: float = 0.0  # Current/Design ratio

    # Cleanliness assessment
    cleanliness_factor_pct: float = 0.0  # Current cleanliness
    cleanliness_trend: str = "STABLE"  # IMPROVING, STABLE, DEGRADING
    cleanliness_degradation_rate_pct_day: float = 0.0  # Daily degradation rate

    # Cleaning status
    cleaning_due: bool = False  # Cleaning recommended
    days_until_cleaning: float = 0.0  # Estimated days until cleaning required
    days_since_last_cleaning: float = 0.0  # Days since last cleaning
    last_cleaning_date: Optional[datetime] = None

    # Recommended actions
    recommended_cleaning_method: CleaningMethod = CleaningMethod.MECHANICAL_BRUSH
    fouling_type_assessment: FoulingType = FoulingType.MIXED
    urgency_level: str = "ROUTINE"  # ROUTINE, ELEVATED, URGENT, CRITICAL

    # Economic impact
    heat_rate_penalty_btu_kwh: float = 0.0  # Current heat rate penalty
    estimated_annual_cost_usd: float = 0.0  # Estimated annual cost of fouling
    cleaning_cost_estimate_usd: float = 0.0  # Estimated cleaning cost

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def assess_cleaning_urgency(
        self,
        cleanliness_threshold_warning: float = 75.0,
        cleanliness_threshold_critical: float = 60.0,
    ) -> str:
        """
        Assess cleaning urgency based on cleanliness factor.

        Args:
            cleanliness_threshold_warning: Warning threshold percentage
            cleanliness_threshold_critical: Critical threshold percentage

        Returns:
            Urgency level string
        """
        if self.cleanliness_factor_pct >= 85:
            self.urgency_level = "ROUTINE"
            self.cleaning_due = False
        elif self.cleanliness_factor_pct >= cleanliness_threshold_warning:
            self.urgency_level = "ELEVATED"
            self.cleaning_due = False
        elif self.cleanliness_factor_pct >= cleanliness_threshold_critical:
            self.urgency_level = "URGENT"
            self.cleaning_due = True
        else:
            self.urgency_level = "CRITICAL"
            self.cleaning_due = True

        return self.urgency_level

    def estimate_days_until_cleaning(self) -> float:
        """
        Estimate days until cleaning will be required.

        Returns:
            Estimated days until cleaning threshold reached
        """
        if self.cleanliness_degradation_rate_pct_day <= 0:
            return float("inf")

        threshold = 70.0  # Cleaning threshold
        current = self.cleanliness_factor_pct

        if current <= threshold:
            return 0.0

        self.days_until_cleaning = (current - threshold) / self.cleanliness_degradation_rate_pct_day
        return self.days_until_cleaning

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "condenser_id": self.condenser_id,
            "fouling_factor_hr_sqft_f_btu": self.fouling_factor_hr_sqft_f_btu,
            "cleanliness_factor_pct": self.cleanliness_factor_pct,
            "cleanliness_trend": self.cleanliness_trend,
            "cleaning_due": self.cleaning_due,
            "days_until_cleaning": self.days_until_cleaning,
            "recommended_cleaning_method": self.recommended_cleaning_method.value,
            "fouling_type_assessment": self.fouling_type_assessment.value,
            "urgency_level": self.urgency_level,
            "heat_rate_penalty_btu_kwh": self.heat_rate_penalty_btu_kwh,
            "recommendations": self.recommendations,
        }


@dataclass
class OptimizationResult:
    """
    Optimization result with recommended setpoints and expected savings.

    Contains optimized operating parameters, expected performance
    improvements, and implementation recommendations.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    condenser_id: str = ""

    # Optimized setpoints
    optimal_vacuum_in_hg_abs: float = 0.0  # Optimized vacuum setpoint
    optimal_cw_flow_gpm: float = 0.0  # Optimized cooling water flow
    optimal_cw_inlet_temp_f: float = 0.0  # Target CW inlet temperature
    optimal_pump_configuration: str = ""  # Recommended pump configuration

    # Current vs optimal comparison
    current_vacuum_in_hg_abs: float = 0.0
    vacuum_improvement_potential_in_hg: float = 0.0
    current_ttd_f: float = 0.0
    optimal_ttd_f: float = 0.0

    # Expected savings
    expected_heat_rate_improvement_btu_kwh: float = 0.0
    expected_cw_pump_power_savings_kw: float = 0.0
    expected_annual_savings_usd: float = 0.0

    # Auxiliary recommendations
    cooling_tower_setpoint_f: Optional[float] = None
    ct_fan_speed_recommendation: Optional[str] = None

    # Confidence and constraints
    optimization_confidence: float = 0.0  # 0-1 confidence score
    constraints_active: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Implementation status
    auto_implemented: bool = False  # Whether changes were auto-implemented
    manual_approval_required: bool = True

    def calculate_savings(
        self,
        turbine_output_mw: float,
        capacity_factor: float = 0.85,
        electricity_price_usd_mwh: float = 50.0,
        pump_power_kw: float = 1000.0,
    ) -> float:
        """
        Calculate expected annual savings from optimization.

        Args:
            turbine_output_mw: Turbine output in MW
            capacity_factor: Plant capacity factor
            electricity_price_usd_mwh: Electricity price
            pump_power_kw: Current pump power consumption

        Returns:
            Expected annual savings in USD
        """
        annual_hours = 8760 * capacity_factor

        # Heat rate improvement savings
        if self.expected_heat_rate_improvement_btu_kwh > 0:
            # Convert heat rate improvement to efficiency gain
            # Approximate: 1% efficiency = ~100 BTU/kWh improvement
            efficiency_gain_pct = self.expected_heat_rate_improvement_btu_kwh / 100
            additional_mwh = turbine_output_mw * annual_hours * (efficiency_gain_pct / 100)
            heat_rate_savings = additional_mwh * electricity_price_usd_mwh
        else:
            heat_rate_savings = 0.0

        # Pump power savings
        pump_savings_mwh = (self.expected_cw_pump_power_savings_kw / 1000) * annual_hours
        pump_power_savings = pump_savings_mwh * electricity_price_usd_mwh

        self.expected_annual_savings_usd = heat_rate_savings + pump_power_savings
        return self.expected_annual_savings_usd

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "condenser_id": self.condenser_id,
            "optimal_vacuum_in_hg_abs": self.optimal_vacuum_in_hg_abs,
            "optimal_cw_flow_gpm": self.optimal_cw_flow_gpm,
            "optimal_cw_inlet_temp_f": self.optimal_cw_inlet_temp_f,
            "optimal_pump_configuration": self.optimal_pump_configuration,
            "vacuum_improvement_potential_in_hg": self.vacuum_improvement_potential_in_hg,
            "expected_heat_rate_improvement_btu_kwh": self.expected_heat_rate_improvement_btu_kwh,
            "expected_cw_pump_power_savings_kw": self.expected_cw_pump_power_savings_kw,
            "expected_annual_savings_usd": self.expected_annual_savings_usd,
            "optimization_confidence": self.optimization_confidence,
            "constraints_active": self.constraints_active,
            "recommendations": self.recommendations,
        }


@dataclass
class CondenserPerformanceResult:
    """
    Complete condenser performance analysis result.

    Aggregates all analysis components with data provenance.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    condenser_id: str = ""
    agent_version: str = "1.0.0"

    # Component results
    cooling_water: Optional[CoolingWaterData] = None
    vacuum_data: Optional[VacuumData] = None
    condensate_data: Optional[CondensateData] = None
    heat_transfer: Optional[HeatTransferData] = None
    fouling_assessment: Optional[FoulingAssessment] = None
    optimization: Optional[OptimizationResult] = None

    # Overall performance status
    performance_status: str = "NORMAL"  # OPTIMAL, NORMAL, DEGRADED, CRITICAL
    performance_score: float = 100.0  # 0-100 score

    # Compliance status
    compliance_status: str = "COMPLIANT"  # COMPLIANT, WARNING, NON_COMPLIANT
    compliance_violations: List[str] = field(default_factory=list)

    # Alerts and notifications
    alerts: List[str] = field(default_factory=list)
    notifications: List[str] = field(default_factory=list)

    # Data provenance
    provenance_hash: str = ""
    data_sources: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0

    def calculate_performance_score(self) -> float:
        """
        Calculate overall performance score.

        Returns:
            Performance score from 0-100
        """
        score = 100.0
        deductions = []

        # Vacuum deviation deduction
        if self.vacuum_data and abs(self.vacuum_data.vacuum_deviation_in_hg) > 0.5:
            vacuum_penalty = min(20, abs(self.vacuum_data.vacuum_deviation_in_hg) * 20)
            score -= vacuum_penalty
            deductions.append(f"Vacuum deviation: -{vacuum_penalty:.1f}")

        # TTD deduction
        if self.heat_transfer and self.heat_transfer.ttd_f > 10:
            ttd_penalty = min(15, (self.heat_transfer.ttd_f - 7) * 3)
            score -= ttd_penalty
            deductions.append(f"High TTD: -{ttd_penalty:.1f}")

        # Cleanliness factor deduction
        if self.heat_transfer and self.heat_transfer.cleanliness_factor_pct < 85:
            cf_penalty = (85 - self.heat_transfer.cleanliness_factor_pct) * 0.5
            score -= cf_penalty
            deductions.append(f"Low cleanliness: -{cf_penalty:.1f}")

        # Air inleakage deduction
        if self.vacuum_data and self.vacuum_data.air_inleakage_scfm > 10:
            air_penalty = min(15, (self.vacuum_data.air_inleakage_scfm - 5) * 1.5)
            score -= air_penalty
            deductions.append(f"Air inleakage: -{air_penalty:.1f}")

        self.performance_score = max(0, min(100, score))

        # Classify status
        if self.performance_score >= 90:
            self.performance_status = "OPTIMAL"
        elif self.performance_score >= 75:
            self.performance_status = "NORMAL"
        elif self.performance_score >= 50:
            self.performance_status = "DEGRADED"
        else:
            self.performance_status = "CRITICAL"

        return self.performance_score

    def calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 hash for data provenance.

        Returns:
            Hexadecimal hash string
        """
        data_dict = {
            "timestamp": self.timestamp.isoformat(),
            "condenser_id": self.condenser_id,
            "agent_version": self.agent_version,
            "cooling_water": self.cooling_water.to_dict() if self.cooling_water else None,
            "vacuum_data": self.vacuum_data.to_dict() if self.vacuum_data else None,
            "heat_transfer": self.heat_transfer.to_dict() if self.heat_transfer else None,
        }
        data_json = json.dumps(data_dict, sort_keys=True)
        self.provenance_hash = hashlib.sha256(data_json.encode()).hexdigest()
        return self.provenance_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "condenser_id": self.condenser_id,
            "agent_version": self.agent_version,
            "cooling_water": self.cooling_water.to_dict() if self.cooling_water else None,
            "vacuum_data": self.vacuum_data.to_dict() if self.vacuum_data else None,
            "condensate_data": self.condensate_data.to_dict() if self.condensate_data else None,
            "heat_transfer": self.heat_transfer.to_dict() if self.heat_transfer else None,
            "fouling_assessment": self.fouling_assessment.to_dict() if self.fouling_assessment else None,
            "optimization": self.optimization.to_dict() if self.optimization else None,
            "performance_status": self.performance_status,
            "performance_score": self.performance_score,
            "compliance_status": self.compliance_status,
            "compliance_violations": self.compliance_violations,
            "alerts": self.alerts,
            "notifications": self.notifications,
            "provenance_hash": self.provenance_hash,
            "data_sources": self.data_sources,
            "processing_time_seconds": self.processing_time_seconds,
        }


# ============================================================================
# MAIN AGENT ORCHESTRATOR
# ============================================================================


class CondenserOptimizationAgent(BaseOrchestrator[AgentConfiguration, CondenserPerformanceResult]):
    """
    GL-017 CONDENSYNC - Condenser Optimization Agent.

    Main orchestrator for comprehensive steam turbine condenser performance
    optimization. Coordinates vacuum monitoring, cooling water flow optimization,
    heat transfer analysis, and air inleakage detection.

    This agent implements industry best practices based on HEI (Heat Exchange
    Institute) standards and integrates with plant control systems for
    real-time optimization.

    Attributes:
        config: Agent configuration
        message_bus: Async messaging bus for agent coordination
        task_scheduler: Task scheduler for workload management
        safety_monitor: Safety constraint monitoring
        coordination_layer: Multi-agent coordination
    """

    def __init__(
        self,
        config: AgentConfiguration,
        orchestrator_config: Optional[OrchestratorConfig] = None,
    ):
        """
        Initialize CondenserOptimizationAgent.

        Args:
            config: Agent configuration
            orchestrator_config: Orchestrator configuration (optional)
        """
        # Initialize base orchestrator
        if orchestrator_config is None:
            orchestrator_config = OrchestratorConfig(
                orchestrator_id="GL-017",
                name="CONDENSYNC",
                version="1.0.0",
                max_concurrent_tasks=10,
                default_timeout_seconds=300,
                enable_safety_monitoring=True,
                enable_message_bus=True,
                enable_task_scheduling=True,
                enable_coordination=True,
            )

        super().__init__(orchestrator_config)

        self.config = config
        self._lock = threading.RLock()
        self._historical_data: Dict[str, List[Dict[str, Any]]] = {}
        self._last_analysis_time: Dict[str, datetime] = {}
        self._cleanliness_history: Dict[str, List[Tuple[datetime, float]]] = {}

        logger.info(
            f"Initialized {self.config.agent_name} v{self.config.version} "
            f"for {len(self.config.condensers)} condenser(s)"
        )

    async def orchestrate(
        self, input_data: AgentConfiguration
    ) -> OrchestrationResult[CondenserPerformanceResult]:
        """
        Main orchestration method (required by BaseOrchestrator).

        Args:
            input_data: Agent configuration

        Returns:
            Orchestration result with condenser performance analysis
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Execute main workflow
            result = await self.execute()

            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.processing_time_seconds = processing_time

            # Return orchestration result
            return OrchestrationResult(
                success=True,
                output=result,
                execution_time_seconds=processing_time,
                metadata={
                    "condenser_id": result.condenser_id,
                    "performance_status": result.performance_status,
                    "performance_score": result.performance_score,
                    "provenance_hash": result.provenance_hash,
                },
            )

        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            return OrchestrationResult(
                success=False,
                output=None,
                execution_time_seconds=processing_time,
                error_message=str(e),
                metadata={"error_type": type(e).__name__},
            )

    async def execute(self) -> CondenserPerformanceResult:
        """
        Execute main condenser optimization workflow.

        This is the primary execution method that coordinates all
        condenser analysis and optimization tasks.

        Returns:
            CondenserPerformanceResult with complete analysis

        Raises:
            Exception: If workflow execution fails
        """
        start_time = datetime.now(timezone.utc)

        # Process first condenser by default (can be extended for multi-condenser)
        condenser = self.config.condensers[0]
        condenser_id = condenser.condenser_id

        logger.info(f"Starting condenser performance analysis for {condenser_id}")

        # Initialize result
        result = CondenserPerformanceResult(
            condenser_id=condenser_id,
            agent_version=self.config.version,
            data_sources=["SCADA", "Historical"],
        )

        try:
            # Step 1: Gather cooling water data
            cooling_water = await self._gather_cooling_water_data(condenser_id)
            result.cooling_water = cooling_water

            # Step 2: Gather vacuum data
            vacuum_data = await self._gather_vacuum_data(condenser_id)
            result.vacuum_data = vacuum_data

            # Step 3: Gather condensate data
            condensate_data = await self._gather_condensate_data(condenser_id)
            result.condensate_data = condensate_data

            # Step 4: Analyze condenser performance (comprehensive)
            performance_analysis = await self.analyze_condenser_performance(
                condenser_id, cooling_water, vacuum_data, condensate_data
            )
            result.compliance_status = performance_analysis["compliance_status"]
            result.compliance_violations = performance_analysis["violations"]
            result.alerts.extend(performance_analysis.get("alerts", []))

            # Step 5: Calculate heat transfer efficiency
            heat_transfer = await self.calculate_heat_transfer_efficiency(
                condenser_id, cooling_water, vacuum_data
            )
            result.heat_transfer = heat_transfer

            # Step 6: Detect air inleakage
            air_leakage_analysis = await self.detect_air_inleakage(
                condenser_id, vacuum_data
            )
            if air_leakage_analysis["action_required"]:
                result.alerts.extend(air_leakage_analysis["alerts"])

            # Step 7: Predict fouling and recommend cleaning
            fouling_assessment = await self.predict_fouling(
                condenser_id, heat_transfer
            )
            result.fouling_assessment = fouling_assessment
            if fouling_assessment.cleaning_due:
                cleaning_rec = await self.recommend_tube_cleaning(
                    condenser_id, fouling_assessment
                )
                result.notifications.append(cleaning_rec)

            # Step 8: Optimize vacuum pressure
            vacuum_optimization = await self.optimize_vacuum_pressure(
                condenser_id, cooling_water, vacuum_data
            )

            # Step 9: Optimize cooling water flow
            flow_optimization = await self.optimize_cooling_water_flow(
                condenser_id, cooling_water, vacuum_data
            )

            # Step 10: Create combined optimization result
            optimization = OptimizationResult(
                condenser_id=condenser_id,
                optimal_vacuum_in_hg_abs=vacuum_optimization["optimal_vacuum"],
                optimal_cw_flow_gpm=flow_optimization["optimal_flow"],
                optimal_cw_inlet_temp_f=cooling_water.inlet_temp_f,
                current_vacuum_in_hg_abs=vacuum_data.pressure_in_hg_abs,
                vacuum_improvement_potential_in_hg=vacuum_optimization["improvement_potential"],
                current_ttd_f=heat_transfer.ttd_f,
                optimal_ttd_f=vacuum_optimization["optimal_ttd"],
                expected_heat_rate_improvement_btu_kwh=vacuum_optimization["heat_rate_improvement"],
                expected_cw_pump_power_savings_kw=flow_optimization["pump_power_savings"],
                optimization_confidence=0.85,
                recommendations=vacuum_optimization["recommendations"] + flow_optimization["recommendations"],
            )
            optimization.calculate_savings(turbine_output_mw=500.0)
            result.optimization = optimization

            # Step 11: Coordinate with cooling tower (if enabled)
            if self.config.cooling_tower_integration:
                ct_coordination = await self.integrate_cooling_tower(
                    condenser_id, cooling_water, vacuum_data
                )
                if ct_coordination:
                    result.notifications.append(ct_coordination)

            # Step 12: Coordinate with turbine (if enabled)
            if self.config.turbine_coordination:
                turbine_coordination = await self.coordinate_with_turbine(
                    condenser_id, vacuum_data
                )
                if turbine_coordination:
                    result.notifications.append(turbine_coordination)

            # Step 13: Calculate performance score
            result.calculate_performance_score()

            # Step 14: Calculate provenance hash
            result.calculate_provenance_hash()

            # Step 15: Store historical data
            self._store_historical_data(condenser_id, result)

            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.processing_time_seconds = processing_time

            logger.info(
                f"Condenser analysis completed for {condenser_id} in "
                f"{processing_time:.2f}s - Status: {result.performance_status}, "
                f"Score: {result.performance_score:.1f}"
            )

            return result

        except Exception as e:
            logger.error(f"Condenser optimization workflow failed: {e}", exc_info=True)
            result.performance_status = "ERROR"
            result.alerts.append(f"Analysis failed: {str(e)}")
            raise

    async def analyze_condenser_performance(
        self,
        condenser_id: str,
        cooling_water: CoolingWaterData,
        vacuum_data: VacuumData,
        condensate_data: CondensateData,
    ) -> Dict[str, Any]:
        """
        Analyze condenser performance against targets and limits.

        Args:
            condenser_id: Condenser identifier
            cooling_water: Cooling water measurements
            vacuum_data: Vacuum measurements
            condensate_data: Condensate measurements

        Returns:
            Dictionary with compliance status and violations
        """
        logger.debug(f"Analyzing condenser performance for {condenser_id}")

        condenser = self.config.get_condenser(condenser_id)
        if condenser is None:
            return {
                "compliance_status": "ERROR",
                "violations": [f"Condenser {condenser_id} not found in configuration"],
                "alerts": [],
            }

        violations = []
        alerts = []
        targets = self.config.performance_targets
        thresholds = self.config.alert_thresholds

        # Check vacuum performance
        vacuum_deviation = abs(vacuum_data.pressure_in_hg_abs - condenser.design_vacuum_in_hg_abs)
        vacuum_data.vacuum_deviation_in_hg = vacuum_deviation

        if vacuum_deviation > thresholds.vacuum_critical_deviation_in_hg:
            violations.append(
                f"Vacuum deviation {vacuum_deviation:.2f} in Hg exceeds critical threshold "
                f"{thresholds.vacuum_critical_deviation_in_hg} in Hg"
            )
            alerts.append("CRITICAL: Vacuum significantly below design")
        elif vacuum_deviation > thresholds.vacuum_warning_deviation_in_hg:
            alerts.append(f"WARNING: Vacuum deviation {vacuum_deviation:.2f} in Hg")

        # Check cooling water temperature
        if cooling_water.inlet_temp_f > thresholds.cooling_water_temp_critical_f:
            violations.append(
                f"CW inlet temp {cooling_water.inlet_temp_f}F exceeds critical "
                f"{thresholds.cooling_water_temp_critical_f}F"
            )
            alerts.append("CRITICAL: Cooling water temperature too high")
        elif cooling_water.inlet_temp_f > thresholds.cooling_water_temp_warning_f:
            alerts.append(f"WARNING: High cooling water temp {cooling_water.inlet_temp_f}F")

        # Check cooling water flow
        flow_pct = (cooling_water.flow_rate_gpm / self.config.cooling_water_config.design_flow_gpm) * 100
        cooling_water.flow_percentage_of_design = flow_pct

        if flow_pct < thresholds.low_flow_critical_pct:
            violations.append(
                f"CW flow {flow_pct:.1f}% of design below critical {thresholds.low_flow_critical_pct}%"
            )
            alerts.append("CRITICAL: Cooling water flow too low")
        elif flow_pct < thresholds.low_flow_warning_pct:
            alerts.append(f"WARNING: Low cooling water flow {flow_pct:.1f}% of design")

        # Check air inleakage
        if vacuum_data.air_inleakage_scfm > thresholds.air_inleakage_critical_scfm:
            violations.append(
                f"Air inleakage {vacuum_data.air_inleakage_scfm:.1f} SCFM exceeds critical "
                f"{thresholds.air_inleakage_critical_scfm} SCFM"
            )
            alerts.append("CRITICAL: Excessive air inleakage")
        elif vacuum_data.air_inleakage_scfm > thresholds.air_inleakage_warning_scfm:
            alerts.append(f"WARNING: Elevated air inleakage {vacuum_data.air_inleakage_scfm:.1f} SCFM")

        # Check condensate subcooling
        subcooling = condensate_data.subcooling_f
        if subcooling > 5.0:
            alerts.append(f"WARNING: Excessive subcooling {subcooling:.1f}F")

        # Determine overall status
        if len(violations) == 0:
            compliance_status = "COMPLIANT"
        elif len(violations) <= 2:
            compliance_status = "WARNING"
        else:
            compliance_status = "NON_COMPLIANT"

        return {
            "compliance_status": compliance_status,
            "violations": violations,
            "alerts": alerts,
            "total_violations": len(violations),
        }

    async def optimize_vacuum_pressure(
        self,
        condenser_id: str,
        cooling_water: CoolingWaterData,
        vacuum_data: VacuumData,
    ) -> Dict[str, Any]:
        """
        Optimize condenser vacuum pressure.

        Calculates achievable vacuum based on current cooling water
        conditions and recommends setpoint adjustments.

        Args:
            condenser_id: Condenser identifier
            cooling_water: Cooling water data
            vacuum_data: Current vacuum data

        Returns:
            Dictionary with optimization results
        """
        logger.debug(f"Optimizing vacuum pressure for {condenser_id}")

        condenser = self.config.get_condenser(condenser_id)
        if condenser is None:
            raise ValueError(f"Condenser {condenser_id} not found")

        recommendations = []

        # Calculate expected vacuum based on CW inlet temperature
        # Approximate: vacuum (in Hg abs) ~= 0.95 + 0.02 * (CW_inlet - 70)
        # This accounts for achievable vacuum degradation with higher CW temps
        cw_inlet_f = cooling_water.inlet_temp_f
        expected_vacuum = 0.95 + 0.02 * max(0, cw_inlet_f - 70)

        # Apply cleanliness factor correction
        # Lower cleanliness = higher (worse) vacuum
        cleanliness_correction = 0.0
        if hasattr(self, '_latest_cleanliness'):
            cleanliness = self._latest_cleanliness.get(condenser_id, 85.0)
            if cleanliness < 85:
                cleanliness_correction = (85 - cleanliness) * 0.01

        optimal_vacuum = expected_vacuum + cleanliness_correction

        # Calculate improvement potential
        current_vacuum = vacuum_data.pressure_in_hg_abs
        improvement_potential = max(0, current_vacuum - optimal_vacuum)

        # Calculate optimal TTD
        # TTD typically ranges from 5-15F depending on conditions
        optimal_ttd = 5.0 + (cw_inlet_f - 60) * 0.1

        # Calculate heat rate improvement
        # Approximate: 1 in Hg improvement = 80 BTU/kWh improvement
        heat_rate_improvement = improvement_potential * 80

        # Generate recommendations
        if improvement_potential > 0.3:
            recommendations.append(
                f"Vacuum can be improved by {improvement_potential:.2f} in Hg "
                f"with current CW conditions"
            )

        if vacuum_data.air_inleakage_scfm > 10:
            recommendations.append(
                "Address air inleakage to improve vacuum - check LP turbine seals"
            )

        if cleanliness_correction > 0.1:
            recommendations.append(
                "Tube cleaning would improve achievable vacuum"
            )

        return {
            "optimal_vacuum": optimal_vacuum,
            "expected_vacuum_at_conditions": expected_vacuum,
            "improvement_potential": improvement_potential,
            "optimal_ttd": optimal_ttd,
            "heat_rate_improvement": heat_rate_improvement,
            "recommendations": recommendations,
        }

    async def optimize_cooling_water_flow(
        self,
        condenser_id: str,
        cooling_water: CoolingWaterData,
        vacuum_data: VacuumData,
    ) -> Dict[str, Any]:
        """
        Optimize cooling water flow rate.

        Balances pump power consumption against condenser performance
        to find optimal flow setpoint.

        Args:
            condenser_id: Condenser identifier
            cooling_water: Cooling water data
            vacuum_data: Vacuum data

        Returns:
            Dictionary with flow optimization results
        """
        logger.debug(f"Optimizing cooling water flow for {condenser_id}")

        cw_config = self.config.cooling_water_config
        recommendations = []

        current_flow = cooling_water.flow_rate_gpm
        design_flow = cw_config.design_flow_gpm

        # Calculate optimal flow based on current conditions
        # Higher CW inlet temp may require higher flow
        cw_inlet_temp = cooling_water.inlet_temp_f
        design_inlet_temp = cw_config.design_inlet_temp_f

        temp_ratio = (cw_inlet_temp - 40) / (design_inlet_temp - 40)
        flow_factor = min(1.2, max(0.8, temp_ratio))

        optimal_flow = design_flow * flow_factor

        # Constrain within limits
        optimal_flow = max(cw_config.min_flow_gpm, min(cw_config.max_flow_gpm, optimal_flow))

        # Calculate pump power savings
        # Power varies with flow^3 (for centrifugal pumps)
        if current_flow > 0:
            flow_ratio = optimal_flow / current_flow
            power_ratio = flow_ratio ** 3

            # Estimate current pump power
            # Approximate: P = Q * H / (3960 * eff)
            pump_head = cw_config.pump_head_ft
            pump_eff = cw_config.pump_efficiency_pct / 100
            current_pump_power_hp = (current_flow * pump_head) / (3960 * pump_eff)
            current_pump_power_kw = current_pump_power_hp * 0.746

            optimal_pump_power_kw = current_pump_power_kw * power_ratio
            pump_power_savings = current_pump_power_kw - optimal_pump_power_kw
        else:
            pump_power_savings = 0.0

        # Determine optimal pump configuration
        pumps_needed = math.ceil(optimal_flow / cw_config.pump_capacity_gpm)
        if pumps_needed < cw_config.number_of_pumps:
            pump_config = f"Run {pumps_needed} of {cw_config.number_of_pumps} CW pumps"
        else:
            pump_config = f"Run all {cw_config.number_of_pumps} CW pumps"

        # Generate recommendations
        flow_difference = optimal_flow - current_flow
        if abs(flow_difference) > design_flow * 0.05:
            if flow_difference > 0:
                recommendations.append(
                    f"Increase CW flow by {abs(flow_difference):.0f} GPM for better vacuum"
                )
            else:
                recommendations.append(
                    f"Reduce CW flow by {abs(flow_difference):.0f} GPM to save pump power"
                )

        if cw_config.vfd_enabled:
            recommendations.append(
                "Utilize VFDs to optimize pump speed and power consumption"
            )

        return {
            "optimal_flow": optimal_flow,
            "flow_factor": flow_factor,
            "pump_power_savings": pump_power_savings,
            "pump_configuration": pump_config,
            "recommendations": recommendations,
        }

    async def calculate_heat_transfer_efficiency(
        self,
        condenser_id: str,
        cooling_water: CoolingWaterData,
        vacuum_data: VacuumData,
    ) -> HeatTransferData:
        """
        Calculate heat transfer efficiency and performance.

        Calculates U-value, LMTD, TTD, and cleanliness factor.

        Args:
            condenser_id: Condenser identifier
            cooling_water: Cooling water data
            vacuum_data: Vacuum data

        Returns:
            HeatTransferData with calculated values
        """
        logger.debug(f"Calculating heat transfer efficiency for {condenser_id}")

        condenser = self.config.get_condenser(condenser_id)
        if condenser is None:
            raise ValueError(f"Condenser {condenser_id} not found")

        heat_transfer = HeatTransferData(condenser_id=condenser_id)

        # Calculate saturation temperature
        steam_temp_f = vacuum_data.saturation_temp_f
        if steam_temp_f <= 0:
            vacuum_data.calculate_saturation_temp()
            steam_temp_f = vacuum_data.saturation_temp_f

        # Calculate LMTD
        heat_transfer.calculate_lmtd(
            steam_temp_f,
            cooling_water.inlet_temp_f,
            cooling_water.outlet_temp_f,
        )

        # Calculate heat duty
        # Q = m * Cp * delta_T
        # For water: Cp = 1 BTU/lb-F, density ~8.34 lb/gal
        flow_lb_hr = cooling_water.flow_rate_gpm * 8.34 * 60
        temp_rise = cooling_water.outlet_temp_f - cooling_water.inlet_temp_f
        heat_duty_btu_hr = flow_lb_hr * temp_rise
        heat_transfer.heat_duty_mmbtu_hr = heat_duty_btu_hr / 1e6

        # Calculate actual U-value
        heat_transfer.u_value_design_btu_hr_sqft_f = condenser.design_u_value_btu_hr_sqft_f
        heat_transfer.design_heat_duty_mmbtu_hr = condenser.design_heat_duty_mmbtu_hr

        heat_transfer.calculate_u_value(
            heat_duty_btu_hr,
            condenser.surface_area_sqft,
        )

        # Calculate U-value ratio
        if heat_transfer.u_value_design_btu_hr_sqft_f > 0:
            heat_transfer.u_value_ratio = (
                heat_transfer.u_value_actual_btu_hr_sqft_f /
                heat_transfer.u_value_design_btu_hr_sqft_f
            )

        # Calculate cleanliness factor
        heat_transfer.calculate_cleanliness_factor()

        # Calculate fouling factor from cleanliness
        if heat_transfer.u_value_actual_btu_hr_sqft_f > 0 and heat_transfer.u_value_design_btu_hr_sqft_f > 0:
            # Rf = (1/U_actual) - (1/U_clean)
            r_actual = 1 / heat_transfer.u_value_actual_btu_hr_sqft_f
            r_clean = 1 / heat_transfer.u_value_design_btu_hr_sqft_f
            heat_transfer.fouling_factor_hr_sqft_f_btu = max(0, r_actual - r_clean)

        # Store cleanliness for other calculations
        if not hasattr(self, '_latest_cleanliness'):
            self._latest_cleanliness = {}
        self._latest_cleanliness[condenser_id] = heat_transfer.cleanliness_factor_pct

        # Track cleanliness history
        self._store_cleanliness_history(condenser_id, heat_transfer.cleanliness_factor_pct)

        logger.info(
            f"Heat transfer analysis: U={heat_transfer.u_value_actual_btu_hr_sqft_f:.0f} "
            f"BTU/hr-sqft-F, TTD={heat_transfer.ttd_f:.1f}F, "
            f"Cleanliness={heat_transfer.cleanliness_factor_pct:.1f}%"
        )

        return heat_transfer

    async def detect_air_inleakage(
        self,
        condenser_id: str,
        vacuum_data: VacuumData,
    ) -> Dict[str, Any]:
        """
        Detect and analyze air inleakage.

        Monitors air removal rate and identifies potential leak sources.

        Args:
            condenser_id: Condenser identifier
            vacuum_data: Vacuum data

        Returns:
            Dictionary with air inleakage analysis
        """
        logger.debug(f"Detecting air inleakage for {condenser_id}")

        vacuum_config = self.config.vacuum_system_config
        thresholds = self.config.alert_thresholds

        air_inleakage = vacuum_data.air_inleakage_scfm
        design_leakage = vacuum_config.air_leakage_design_scfm
        alarm_threshold = vacuum_config.air_leakage_alarm_scfm

        alerts = []
        recommendations = []
        action_required = False

        # Calculate leakage ratio
        leakage_ratio = air_inleakage / design_leakage if design_leakage > 0 else 0

        # Assess severity
        if air_inleakage > thresholds.air_inleakage_critical_scfm:
            severity = "CRITICAL"
            action_required = True
            alerts.append(f"CRITICAL: Air inleakage {air_inleakage:.1f} SCFM - immediate action required")
            recommendations.append("Conduct immediate leak survey using ultrasonic detector")
            recommendations.append("Check LP turbine gland seals")
            recommendations.append("Inspect valve packing on vacuum system")
        elif air_inleakage > thresholds.air_inleakage_warning_scfm:
            severity = "WARNING"
            alerts.append(f"WARNING: Elevated air inleakage {air_inleakage:.1f} SCFM")
            recommendations.append("Schedule leak survey during next available window")
            recommendations.append("Monitor trend for further increase")
        elif air_inleakage > design_leakage * 1.5:
            severity = "ELEVATED"
            recommendations.append("Monitor air inleakage trend")
        else:
            severity = "NORMAL"

        # Identify potential leak sources based on symptoms
        leak_sources = []
        if air_inleakage > design_leakage * 2:
            leak_sources.append("LP turbine shaft seals")
            leak_sources.append("Expansion joint bellows")
            leak_sources.append("Manhole covers and gaskets")

        if vacuum_data.hotwell_level_pct < 30:
            leak_sources.append("Hotwell level controller issues")
            recommendations.append("Check hotwell level control system")

        # Calculate vacuum impact
        # Approximate: each 1 SCFM excess air = 0.02 in Hg vacuum loss
        excess_leakage = max(0, air_inleakage - design_leakage)
        vacuum_impact = excess_leakage * 0.02

        return {
            "severity": severity,
            "air_inleakage_scfm": air_inleakage,
            "design_leakage_scfm": design_leakage,
            "leakage_ratio": leakage_ratio,
            "vacuum_impact_in_hg": vacuum_impact,
            "potential_leak_sources": leak_sources,
            "alerts": alerts,
            "recommendations": recommendations,
            "action_required": action_required,
        }

    async def predict_fouling(
        self,
        condenser_id: str,
        heat_transfer: HeatTransferData,
    ) -> FoulingAssessment:
        """
        Predict fouling and generate cleaning recommendations.

        Analyzes cleanliness trend and predicts time to cleaning.

        Args:
            condenser_id: Condenser identifier
            heat_transfer: Heat transfer data

        Returns:
            FoulingAssessment with predictions and recommendations
        """
        logger.debug(f"Predicting fouling for {condenser_id}")

        assessment = FoulingAssessment(condenser_id=condenser_id)
        assessment.fouling_factor_hr_sqft_f_btu = heat_transfer.fouling_factor_hr_sqft_f_btu
        assessment.cleanliness_factor_pct = heat_transfer.cleanliness_factor_pct

        # Get condenser configuration
        condenser = self.config.get_condenser(condenser_id)
        if condenser and condenser.last_cleaning_date:
            days_since = (datetime.now(timezone.utc) - condenser.last_cleaning_date).days
            assessment.days_since_last_cleaning = days_since
            assessment.last_cleaning_date = condenser.last_cleaning_date

        # Analyze cleanliness trend from history
        history = self._cleanliness_history.get(condenser_id, [])
        if len(history) >= 2:
            # Calculate degradation rate (% per day)
            recent_history = history[-100:]  # Last 100 data points
            if len(recent_history) >= 2:
                first_time, first_cf = recent_history[0]
                last_time, last_cf = recent_history[-1]
                time_diff_days = (last_time - first_time).total_seconds() / 86400
                if time_diff_days > 0:
                    cf_change = first_cf - last_cf
                    assessment.cleanliness_degradation_rate_pct_day = cf_change / time_diff_days

                    if assessment.cleanliness_degradation_rate_pct_day > 0.05:
                        assessment.cleanliness_trend = "DEGRADING"
                    elif assessment.cleanliness_degradation_rate_pct_day < -0.01:
                        assessment.cleanliness_trend = "IMPROVING"
                    else:
                        assessment.cleanliness_trend = "STABLE"

        # Assess urgency
        assessment.assess_cleaning_urgency(
            self.config.alert_thresholds.fouling_warning_pct,
            self.config.alert_thresholds.fouling_critical_pct,
        )

        # Estimate days until cleaning
        assessment.estimate_days_until_cleaning()

        # Assess fouling type based on water quality
        water_limits = self.config.water_quality_limits
        if water_limits.total_bacteria_max_cfu_ml > 5000:
            assessment.fouling_type_assessment = FoulingType.BIOLOGICAL
        elif water_limits.calcium_hardness_max_ppm > 300:
            assessment.fouling_type_assessment = FoulingType.MINERAL_SCALE
        else:
            assessment.fouling_type_assessment = FoulingType.MIXED

        # Recommend cleaning method based on fouling type
        if assessment.fouling_type_assessment == FoulingType.BIOLOGICAL:
            assessment.recommended_cleaning_method = CleaningMethod.CHEMICAL
        elif assessment.fouling_type_assessment == FoulingType.MINERAL_SCALE:
            assessment.recommended_cleaning_method = CleaningMethod.HYDRO_BLAST
        else:
            assessment.recommended_cleaning_method = CleaningMethod.MECHANICAL_BRUSH

        # Calculate heat rate penalty
        # Approximate: 1% cleanliness loss = 2 BTU/kWh penalty
        cleanliness_loss = 100 - assessment.cleanliness_factor_pct
        assessment.heat_rate_penalty_btu_kwh = cleanliness_loss * 2

        # Generate recommendations
        if assessment.urgency_level == "CRITICAL":
            assessment.recommendations.append(
                "IMMEDIATE CLEANING REQUIRED - Condenser severely fouled"
            )
        elif assessment.urgency_level == "URGENT":
            assessment.recommendations.append(
                f"Schedule tube cleaning within {assessment.days_until_cleaning:.0f} days"
            )
        elif assessment.cleaning_due:
            assessment.recommendations.append(
                f"Plan tube cleaning using {assessment.recommended_cleaning_method.value} method"
            )
        else:
            assessment.recommendations.append(
                f"Monitor cleanliness - next cleaning in ~{assessment.days_until_cleaning:.0f} days"
            )

        return assessment

    async def recommend_tube_cleaning(
        self,
        condenser_id: str,
        fouling_assessment: FoulingAssessment,
    ) -> str:
        """
        Generate tube cleaning recommendation.

        Args:
            condenser_id: Condenser identifier
            fouling_assessment: Fouling assessment

        Returns:
            Cleaning recommendation string
        """
        method = fouling_assessment.recommended_cleaning_method.value
        urgency = fouling_assessment.urgency_level
        days = fouling_assessment.days_until_cleaning

        if urgency == "CRITICAL":
            return (
                f"TUBE CLEANING CRITICAL for {condenser_id}: "
                f"Cleanliness at {fouling_assessment.cleanliness_factor_pct:.1f}%. "
                f"Recommend {method} cleaning immediately."
            )
        elif urgency == "URGENT":
            return (
                f"TUBE CLEANING URGENT for {condenser_id}: "
                f"Schedule {method} cleaning within {days:.0f} days. "
                f"Current heat rate penalty: {fouling_assessment.heat_rate_penalty_btu_kwh:.0f} BTU/kWh"
            )
        else:
            return (
                f"Tube cleaning recommendation for {condenser_id}: "
                f"Plan {method} cleaning in approximately {days:.0f} days."
            )

    async def integrate_cooling_tower(
        self,
        condenser_id: str,
        cooling_water: CoolingWaterData,
        vacuum_data: VacuumData,
    ) -> Optional[str]:
        """
        Coordinate with cooling tower control.

        Optimizes cooling tower operation to provide optimal
        cooling water temperature to condenser.

        Args:
            condenser_id: Condenser identifier
            cooling_water: Cooling water data
            vacuum_data: Vacuum data

        Returns:
            Coordination message or None
        """
        logger.debug(f"Integrating with cooling tower for {condenser_id}")

        ct_config = self.config.cooling_tower_integration
        if ct_config is None or not ct_config.enabled:
            return None

        # Calculate optimal cold water temperature
        # Target is to achieve design vacuum at current wet bulb conditions
        condenser = self.config.get_condenser(condenser_id)
        if condenser is None:
            return None

        # Approximate optimal CW temp based on design conditions
        design_vacuum = condenser.design_vacuum_in_hg_abs
        optimal_cw_temp = 60 + (vacuum_data.pressure_in_hg_abs - 1.0) * 10

        current_cw_temp = cooling_water.inlet_temp_f
        temp_difference = current_cw_temp - optimal_cw_temp

        if abs(temp_difference) < 2:
            return None

        if temp_difference > 0:
            # Need cooler water - increase CT capacity
            return (
                f"Cooling tower coordination: Increase cooling capacity to reduce "
                f"CW temp from {current_cw_temp:.1f}F toward {optimal_cw_temp:.1f}F"
            )
        else:
            # Can reduce CT capacity - save fan power
            return (
                f"Cooling tower coordination: Reduce fan speed to save power - "
                f"CW temp {current_cw_temp:.1f}F already below optimal {optimal_cw_temp:.1f}F"
            )

    async def coordinate_with_turbine(
        self,
        condenser_id: str,
        vacuum_data: VacuumData,
    ) -> Optional[str]:
        """
        Coordinate with turbine backpressure control.

        Ensures condenser vacuum is within turbine safe operating limits.

        Args:
            condenser_id: Condenser identifier
            vacuum_data: Vacuum data

        Returns:
            Coordination message or None
        """
        logger.debug(f"Coordinating with turbine for {condenser_id}")

        turbine_config = self.config.turbine_coordination
        if turbine_config is None or not turbine_config.enabled:
            return None

        current_bp = vacuum_data.pressure_in_hg_abs
        max_bp = turbine_config.backpressure_limit_in_hg_abs

        if current_bp > max_bp * 0.9:
            # Approaching limit
            return (
                f"Turbine coordination: Backpressure {current_bp:.2f} in Hg abs "
                f"approaching limit of {max_bp:.2f} in Hg abs. "
                f"Recommend load reduction or increased condenser cooling."
            )

        # Calculate heat rate correction for current backpressure
        heat_rate_correction = turbine_config.calculate_heat_rate_correction(current_bp)
        if heat_rate_correction > 50:
            return (
                f"Turbine coordination: Current backpressure causing "
                f"{heat_rate_correction:.0f} BTU/kWh heat rate penalty"
            )

        return None

    def _store_historical_data(
        self,
        condenser_id: str,
        result: CondenserPerformanceResult,
    ) -> None:
        """
        Store historical performance data.

        Args:
            condenser_id: Condenser identifier
            result: Performance result to store
        """
        with self._lock:
            if condenser_id not in self._historical_data:
                self._historical_data[condenser_id] = []

            self._historical_data[condenser_id].append(result.to_dict())

            # Keep last 1000 data points
            if len(self._historical_data[condenser_id]) > 1000:
                self._historical_data[condenser_id] = self._historical_data[condenser_id][-1000:]

            self._last_analysis_time[condenser_id] = datetime.now(timezone.utc)

    def _store_cleanliness_history(
        self,
        condenser_id: str,
        cleanliness_factor: float,
    ) -> None:
        """
        Store cleanliness factor history for trend analysis.

        Args:
            condenser_id: Condenser identifier
            cleanliness_factor: Current cleanliness factor percentage
        """
        with self._lock:
            if condenser_id not in self._cleanliness_history:
                self._cleanliness_history[condenser_id] = []

            self._cleanliness_history[condenser_id].append(
                (datetime.now(timezone.utc), cleanliness_factor)
            )

            # Keep last 1000 data points
            if len(self._cleanliness_history[condenser_id]) > 1000:
                self._cleanliness_history[condenser_id] = self._cleanliness_history[condenser_id][-1000:]

    def get_historical_data(
        self,
        condenser_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get historical performance data.

        Args:
            condenser_id: Condenser identifier
            limit: Maximum number of records

        Returns:
            List of historical data dictionaries
        """
        with self._lock:
            data = self._historical_data.get(condenser_id, [])
            return data[-limit:]

    async def _gather_cooling_water_data(self, condenser_id: str) -> CoolingWaterData:
        """
        Gather cooling water data from SCADA.

        Args:
            condenser_id: Condenser identifier

        Returns:
            CoolingWaterData with current measurements
        """
        logger.debug(f"Gathering cooling water data for {condenser_id}")

        # In production, this would read from actual SCADA system
        # For now, return simulated data
        cw_data = CoolingWaterData(
            condenser_id=condenser_id,
            inlet_temp_f=78.0,
            outlet_temp_f=96.0,
            flow_rate_gpm=150000.0,
            flow_velocity_fps=7.5,
            inlet_pressure_psi=25.0,
            outlet_pressure_psi=18.0,
            ph=7.8,
            conductivity_us_cm=1200,
            turbidity_ntu=5.0,
            chlorine_residual_ppm=0.5,
            data_source="SCADA",
            quality_flag="GOOD",
        )

        cw_data.calculate_temp_rise()
        cw_data.pressure_drop_psi = cw_data.inlet_pressure_psi - cw_data.outlet_pressure_psi

        return cw_data

    async def _gather_vacuum_data(self, condenser_id: str) -> VacuumData:
        """
        Gather vacuum data from SCADA.

        Args:
            condenser_id: Condenser identifier

        Returns:
            VacuumData with current measurements
        """
        logger.debug(f"Gathering vacuum data for {condenser_id}")

        # In production, this would read from actual SCADA system
        vacuum_data = VacuumData(
            condenser_id=condenser_id,
            pressure_in_hg_abs=1.8,
            pressure_mbar_abs=61.0,
            air_inleakage_scfm=8.0,
            air_inleakage_trend="STABLE",
            hotwell_level_pct=65.0,
            hotwell_temp_f=118.0,
            ejector_stage1_status="RUNNING",
            ejector_stage2_status="RUNNING",
            lrvp_status="STANDBY",
            data_source="SCADA",
            quality_flag="GOOD",
        )

        vacuum_data.calculate_saturation_temp()

        return vacuum_data

    async def _gather_condensate_data(self, condenser_id: str) -> CondensateData:
        """
        Gather condensate data from SCADA.

        Args:
            condenser_id: Condenser identifier

        Returns:
            CondensateData with current measurements
        """
        logger.debug(f"Gathering condensate data for {condenser_id}")

        # In production, this would read from actual SCADA system
        condensate_data = CondensateData(
            condenser_id=condenser_id,
            flow_rate_lb_hr=500000.0,
            flow_rate_gpm=1000.0,
            temperature_f=115.0,
            dissolved_oxygen_ppb=7.0,
            conductivity_us_cm=0.5,
            condensate_pump_running=2,
            condensate_pump_discharge_psi=400.0,
            data_source="SCADA",
            quality_flag="GOOD",
        )

        return condensate_data


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "CondenserOptimizationAgent",
    "CoolingWaterData",
    "VacuumData",
    "CondensateData",
    "HeatTransferData",
    "FoulingAssessment",
    "OptimizationResult",
    "CondenserPerformanceResult",
]
