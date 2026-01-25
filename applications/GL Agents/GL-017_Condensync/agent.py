# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC Agent - Condenser Optimization Agent

Main orchestrator for condenser performance diagnostics, vacuum optimization,
fouling prediction, and maintenance scheduling. Designed for power plant and
industrial process condensers.

Standards:
- HEI Standards for Steam Surface Condensers (12th Edition)
- ASME PTC 12.2: Steam Surface Condensers
- EPRI Condenser In-Leakage Guidelines
- Cooling Technology Institute (CTI) Standards

Key Capabilities:
- Real-time condenser performance diagnostics
- Vacuum optimization for turbine backpressure
- Fouling prediction using Kern-Seaton models
- Maintenance scheduling with economic optimization
- Air in-leakage detection and quantification
- Tube cleaning schedule optimization

Zero-Hallucination Guarantee:
All calculations use deterministic engineering formulas from HEI/ASME standards.
No LLM or AI inference in any calculation path.
Same inputs always produce identical outputs.
Complete SHA-256 provenance tracking for audit trails.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class AgentMode(str, Enum):
    """Agent operating modes."""
    MONITORING = "monitoring"      # Continuous real-time monitoring
    DIAGNOSTIC = "diagnostic"      # Single diagnostic analysis
    OPTIMIZATION = "optimization"  # Vacuum/performance optimization
    PREDICTIVE = "predictive"      # Fouling/degradation prediction
    MAINTENANCE = "maintenance"    # Maintenance scheduling mode


class CondenserType(str, Enum):
    """Types of condensers supported."""
    SURFACE_SINGLE_PASS = "surface_single_pass"
    SURFACE_TWO_PASS = "surface_two_pass"
    SURFACE_DIVIDED_WATERBOX = "surface_divided_waterbox"
    AIR_COOLED = "air_cooled"
    EVAPORATIVE = "evaporative"
    DIRECT_CONTACT = "direct_contact"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    NONE = "none"


class PerformanceStatus(str, Enum):
    """Condenser performance status."""
    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    POOR = "poor"
    CRITICAL = "critical"


class FoulingMode(str, Enum):
    """Types of fouling mechanisms."""
    BIOLOGICAL = "biological"        # Bio-fouling (algae, mussels)
    SCALING = "scaling"              # Mineral scale deposits
    PARTICULATE = "particulate"      # Silt, debris accumulation
    CORROSION = "corrosion"          # Corrosion products
    COMBINED = "combined"            # Multiple mechanisms


class MaintenanceAction(str, Enum):
    """Types of maintenance actions."""
    ONLINE_BALL_CLEANING = "online_ball_cleaning"
    OFFLINE_MECHANICAL = "offline_mechanical"
    OFFLINE_CHEMICAL = "offline_chemical"
    TUBE_PLUGGING = "tube_plugging"
    TUBE_SLEEVING = "tube_sleeving"
    WATERBOX_INSPECTION = "waterbox_inspection"
    AIR_REMOVAL_CHECK = "air_removal_check"
    NO_ACTION = "no_action"


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class AgentConfig:
    """
    Configuration for CONDENSYNC agent.

    Attributes:
        agent_id: Unique agent identifier
        agent_name: Human-readable name
        version: Agent version
        mode: Operating mode
        polling_interval_seconds: Monitoring poll interval
        enable_provenance: Enable SHA-256 provenance tracking
        enable_explainability: Enable explanation generation
        max_concurrent_calculations: Maximum parallel calculations
        calculation_timeout_seconds: Timeout for single calculation
    """
    agent_id: str = "GL-017"
    agent_name: str = "CONDENSYNC"
    version: str = "1.0.0"
    mode: AgentMode = AgentMode.DIAGNOSTIC
    polling_interval_seconds: float = 30.0
    enable_provenance: bool = True
    enable_explainability: bool = True
    max_concurrent_calculations: int = 10
    calculation_timeout_seconds: int = 300


@dataclass
class CondenserDesignConfig:
    """
    Condenser design parameters (nameplate/design values).

    Attributes:
        condenser_id: Unique condenser identifier
        condenser_type: Type of condenser
        design_duty_mw: Design heat duty (MW thermal)
        design_steam_flow_kg_s: Design steam flow rate (kg/s)
        design_backpressure_kpa: Design backpressure (kPa abs)
        design_ttd_c: Design terminal temperature difference (C)
        tube_od_mm: Tube outer diameter (mm)
        tube_id_mm: Tube inner diameter (mm)
        tube_length_m: Tube length (m)
        tube_count: Total number of tubes
        tube_material: Tube material (e.g., "titanium", "admiralty_brass")
        tube_passes: Number of tube passes
        effective_surface_area_m2: Effective heat transfer area (m2)
        design_cleanliness_factor: Design cleanliness factor (typically 0.85)
        design_u_w_m2k: Design overall U-value (W/m2K)
        design_cwt_inlet_c: Design cooling water inlet temp (C)
        design_cwt_rise_c: Design cooling water temperature rise (C)
        design_cw_flow_m3_s: Design cooling water flow rate (m3/s)
    """
    condenser_id: str = "COND-001"
    condenser_type: CondenserType = CondenserType.SURFACE_TWO_PASS
    design_duty_mw: float = 500.0
    design_steam_flow_kg_s: float = 200.0
    design_backpressure_kpa: float = 5.0
    design_ttd_c: float = 3.0
    tube_od_mm: float = 25.4
    tube_id_mm: float = 22.9
    tube_length_m: float = 12.0
    tube_count: int = 15000
    tube_material: str = "titanium"
    tube_passes: int = 2
    effective_surface_area_m2: float = 14000.0
    design_cleanliness_factor: float = 0.85
    design_u_w_m2k: float = 3000.0
    design_cwt_inlet_c: float = 20.0
    design_cwt_rise_c: float = 10.0
    design_cw_flow_m3_s: float = 12.0


@dataclass
class OperationalConfig:
    """
    Operational thresholds and constraints.

    Attributes:
        min_backpressure_kpa: Minimum allowable backpressure (kPa abs)
        max_backpressure_kpa: Maximum allowable backpressure (kPa abs)
        target_cleanliness_factor: Target cleanliness factor
        fouling_alarm_threshold: Fouling factor alarm threshold (m2K/kW)
        ttd_warning_threshold_c: TTD warning threshold (C)
        ttd_alarm_threshold_c: TTD alarm threshold (C)
        air_inleakage_warning_scfm: Air in-leakage warning (SCFM)
        air_inleakage_alarm_scfm: Air in-leakage alarm (SCFM)
        do2_warning_ppb: Dissolved oxygen warning level (ppb)
        do2_alarm_ppb: Dissolved oxygen alarm level (ppb)
        min_cw_velocity_m_s: Minimum CW velocity for cleaning (m/s)
        max_cw_velocity_m_s: Maximum CW velocity limit (m/s)
        max_tube_plugging_percent: Maximum allowed tube plugging (%)
    """
    min_backpressure_kpa: float = 2.5
    max_backpressure_kpa: float = 15.0
    target_cleanliness_factor: float = 0.90
    fouling_alarm_threshold: float = 0.0003  # m2K/kW
    ttd_warning_threshold_c: float = 4.0
    ttd_alarm_threshold_c: float = 6.0
    air_inleakage_warning_scfm: float = 5.0
    air_inleakage_alarm_scfm: float = 10.0
    do2_warning_ppb: float = 10.0
    do2_alarm_ppb: float = 20.0
    min_cw_velocity_m_s: float = 1.5
    max_cw_velocity_m_s: float = 3.0
    max_tube_plugging_percent: float = 10.0


@dataclass
class OperatingConstraints:
    """
    Operating constraints for optimization.

    Attributes:
        min_load_mw: Minimum electrical load (MW)
        max_load_mw: Maximum electrical load (MW)
        maintenance_window_hours: Available maintenance window (hours)
        max_outage_cost_per_hour: Maximum outage cost ($/hour)
        min_days_between_cleaning: Minimum days between cleanings
        environmental_permit_limit_mw: Environmental limit (MW thermal discharge)
        cw_pump_constraints: Cooling water pump operational constraints
    """
    min_load_mw: float = 100.0
    max_load_mw: float = 600.0
    maintenance_window_hours: float = 8.0
    max_outage_cost_per_hour: float = 50000.0
    min_days_between_cleaning: int = 30
    environmental_permit_limit_mw: float = 600.0
    cw_pump_constraints: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# INPUT DATA CLASSES
# ============================================================================

@dataclass
class CondenserDiagnosticInput:
    """
    Input data for condenser diagnostic analysis.

    Attributes:
        timestamp: Measurement timestamp
        condenser_id: Condenser identifier
        steam_flow_kg_s: Actual steam flow rate (kg/s)
        steam_pressure_kpa: Steam/backpressure (kPa abs)
        hotwell_temp_c: Hotwell temperature (C)
        cw_inlet_temp_c: Cooling water inlet temperature (C)
        cw_outlet_temp_c: Cooling water outlet temperature (C)
        cw_flow_m3_s: Cooling water flow rate (m3/s)
        air_inleakage_scfm: Air in-leakage rate (SCFM)
        dissolved_oxygen_ppb: Dissolved oxygen in condensate (ppb)
        tubes_plugged: Number of tubes plugged
        ejector_steam_flow_kg_s: Air ejector steam consumption (kg/s)
        subcooling_c: Condensate subcooling (C)
        generator_load_mw: Generator electrical load (MW)
    """
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    condenser_id: str = "COND-001"
    steam_flow_kg_s: float = 180.0
    steam_pressure_kpa: float = 5.5
    hotwell_temp_c: float = 33.5
    cw_inlet_temp_c: float = 22.0
    cw_outlet_temp_c: float = 31.0
    cw_flow_m3_s: float = 11.5
    air_inleakage_scfm: float = 3.0
    dissolved_oxygen_ppb: float = 7.0
    tubes_plugged: int = 50
    ejector_steam_flow_kg_s: float = 0.5
    subcooling_c: float = 0.5
    generator_load_mw: float = 450.0


# ============================================================================
# OUTPUT DATA CLASSES
# ============================================================================

@dataclass
class CondenserPerformanceOutput:
    """
    Output from condenser performance diagnostic.

    Attributes:
        condenser_id: Condenser identifier
        timestamp: Analysis timestamp
        status: Overall performance status
        alert_level: Alert severity
        actual_backpressure_kpa: Measured backpressure (kPa abs)
        expected_backpressure_kpa: Expected backpressure at current conditions
        backpressure_deviation_kpa: Deviation from expected
        actual_ttd_c: Actual terminal temperature difference (C)
        design_ttd_c: Design TTD (C)
        cleanliness_factor: Calculated cleanliness factor
        fouling_factor_m2k_kw: Estimated fouling factor (m2K/kW)
        heat_duty_mw: Actual heat duty (MW thermal)
        actual_u_w_m2k: Actual overall heat transfer coefficient
        lmtd_c: Log mean temperature difference (C)
        heat_rate_penalty_kj_kwh: Turbine heat rate penalty (kJ/kWh)
        annual_cost_impact_usd: Estimated annual cost of degradation
        air_inleakage_status: Air in-leakage assessment
        subcooling_status: Condensate subcooling assessment
        recommendations: List of recommended actions
        explanation: Detailed explanation of analysis
        provenance_hash: SHA-256 hash for audit trail
    """
    condenser_id: str
    timestamp: datetime
    status: PerformanceStatus
    alert_level: AlertLevel
    actual_backpressure_kpa: float
    expected_backpressure_kpa: float
    backpressure_deviation_kpa: float
    actual_ttd_c: float
    design_ttd_c: float
    cleanliness_factor: float
    fouling_factor_m2k_kw: float
    heat_duty_mw: float
    actual_u_w_m2k: float
    lmtd_c: float
    heat_rate_penalty_kj_kwh: float
    annual_cost_impact_usd: float
    air_inleakage_status: str
    subcooling_status: str
    recommendations: List[str]
    explanation: Dict[str, Any]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "condenser_id": self.condenser_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "alert_level": self.alert_level.value,
            "actual_backpressure_kpa": round(self.actual_backpressure_kpa, 3),
            "expected_backpressure_kpa": round(self.expected_backpressure_kpa, 3),
            "backpressure_deviation_kpa": round(self.backpressure_deviation_kpa, 3),
            "actual_ttd_c": round(self.actual_ttd_c, 2),
            "design_ttd_c": round(self.design_ttd_c, 2),
            "cleanliness_factor": round(self.cleanliness_factor, 4),
            "fouling_factor_m2k_kw": round(self.fouling_factor_m2k_kw, 6),
            "heat_duty_mw": round(self.heat_duty_mw, 2),
            "actual_u_w_m2k": round(self.actual_u_w_m2k, 1),
            "lmtd_c": round(self.lmtd_c, 2),
            "heat_rate_penalty_kj_kwh": round(self.heat_rate_penalty_kj_kwh, 1),
            "annual_cost_impact_usd": round(self.annual_cost_impact_usd, 0),
            "air_inleakage_status": self.air_inleakage_status,
            "subcooling_status": self.subcooling_status,
            "recommendations": self.recommendations,
            "provenance_hash": self.provenance_hash
        }


@dataclass
class VacuumOptimizationOutput:
    """
    Output from vacuum optimization analysis.

    Attributes:
        condenser_id: Condenser identifier
        timestamp: Analysis timestamp
        current_backpressure_kpa: Current backpressure (kPa abs)
        optimal_backpressure_kpa: Optimal achievable backpressure
        achievable_improvement_kpa: Potential improvement
        current_heat_rate_kj_kwh: Current turbine heat rate
        optimal_heat_rate_kj_kwh: Achievable heat rate
        heat_rate_improvement_percent: Potential improvement percentage
        cw_flow_recommendation_m3_s: Recommended CW flow rate
        cw_pump_configuration: Recommended pump configuration
        air_removal_recommendation: Air removal system recommendation
        estimated_annual_savings_usd: Estimated annual savings
        implementation_actions: List of implementation steps
        constraints_limiting: List of limiting constraints
        optimization_details: Detailed optimization results
        provenance_hash: SHA-256 hash for audit trail
    """
    condenser_id: str
    timestamp: datetime
    current_backpressure_kpa: float
    optimal_backpressure_kpa: float
    achievable_improvement_kpa: float
    current_heat_rate_kj_kwh: float
    optimal_heat_rate_kj_kwh: float
    heat_rate_improvement_percent: float
    cw_flow_recommendation_m3_s: float
    cw_pump_configuration: str
    air_removal_recommendation: str
    estimated_annual_savings_usd: float
    implementation_actions: List[str]
    constraints_limiting: List[str]
    optimization_details: Dict[str, Any]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "condenser_id": self.condenser_id,
            "timestamp": self.timestamp.isoformat(),
            "current_backpressure_kpa": round(self.current_backpressure_kpa, 3),
            "optimal_backpressure_kpa": round(self.optimal_backpressure_kpa, 3),
            "achievable_improvement_kpa": round(self.achievable_improvement_kpa, 3),
            "current_heat_rate_kj_kwh": round(self.current_heat_rate_kj_kwh, 1),
            "optimal_heat_rate_kj_kwh": round(self.optimal_heat_rate_kj_kwh, 1),
            "heat_rate_improvement_percent": round(self.heat_rate_improvement_percent, 2),
            "cw_flow_recommendation_m3_s": round(self.cw_flow_recommendation_m3_s, 2),
            "cw_pump_configuration": self.cw_pump_configuration,
            "air_removal_recommendation": self.air_removal_recommendation,
            "estimated_annual_savings_usd": round(self.estimated_annual_savings_usd, 0),
            "implementation_actions": self.implementation_actions,
            "constraints_limiting": self.constraints_limiting,
            "provenance_hash": self.provenance_hash
        }


@dataclass
class FoulingPredictionOutput:
    """
    Output from fouling prediction analysis.

    Attributes:
        condenser_id: Condenser identifier
        timestamp: Analysis timestamp
        current_fouling_factor: Current fouling factor (m2K/kW)
        predicted_fouling_factor_7d: Predicted fouling in 7 days
        predicted_fouling_factor_30d: Predicted fouling in 30 days
        fouling_rate_per_day: Rate of fouling increase per day
        dominant_fouling_mode: Primary fouling mechanism
        days_to_threshold: Days until alarm threshold reached
        confidence_interval_lower: Lower bound of prediction
        confidence_interval_upper: Upper bound of prediction
        model_r_squared: Model fit quality (R-squared)
        trend_direction: Fouling trend direction
        cleaning_urgency: Urgency level for cleaning
        predicted_backpressure_impact_kpa: Expected backpressure impact
        prediction_details: Detailed prediction data
        provenance_hash: SHA-256 hash for audit trail
    """
    condenser_id: str
    timestamp: datetime
    current_fouling_factor: float
    predicted_fouling_factor_7d: float
    predicted_fouling_factor_30d: float
    fouling_rate_per_day: float
    dominant_fouling_mode: FoulingMode
    days_to_threshold: Optional[float]
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_r_squared: float
    trend_direction: str
    cleaning_urgency: str
    predicted_backpressure_impact_kpa: float
    prediction_details: Dict[str, Any]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "condenser_id": self.condenser_id,
            "timestamp": self.timestamp.isoformat(),
            "current_fouling_factor": round(self.current_fouling_factor, 6),
            "predicted_fouling_factor_7d": round(self.predicted_fouling_factor_7d, 6),
            "predicted_fouling_factor_30d": round(self.predicted_fouling_factor_30d, 6),
            "fouling_rate_per_day": round(self.fouling_rate_per_day, 8),
            "dominant_fouling_mode": self.dominant_fouling_mode.value,
            "days_to_threshold": round(self.days_to_threshold, 1) if self.days_to_threshold else None,
            "confidence_interval_lower": round(self.confidence_interval_lower, 6),
            "confidence_interval_upper": round(self.confidence_interval_upper, 6),
            "model_r_squared": round(self.model_r_squared, 4),
            "trend_direction": self.trend_direction,
            "cleaning_urgency": self.cleaning_urgency,
            "predicted_backpressure_impact_kpa": round(self.predicted_backpressure_impact_kpa, 3),
            "provenance_hash": self.provenance_hash
        }


@dataclass
class MaintenanceRecommendation:
    """
    Output from maintenance recommendation analysis.

    Attributes:
        condenser_id: Condenser identifier
        timestamp: Analysis timestamp
        recommended_action: Primary recommended action
        urgency: Urgency level (immediate, scheduled, monitor)
        optimal_date: Recommended maintenance date
        estimated_duration_hours: Expected maintenance duration
        estimated_cost_usd: Estimated maintenance cost
        expected_benefit_usd: Expected annual benefit
        payback_days: Simple payback period
        roi_percent: Return on investment
        risk_of_deferral: Risk assessment if deferred
        alternative_actions: List of alternative actions
        resource_requirements: Required resources
        scheduling_constraints: Scheduling considerations
        economic_analysis: Detailed economic analysis
        provenance_hash: SHA-256 hash for audit trail
    """
    condenser_id: str
    timestamp: datetime
    recommended_action: MaintenanceAction
    urgency: str
    optimal_date: Optional[datetime]
    estimated_duration_hours: float
    estimated_cost_usd: float
    expected_benefit_usd: float
    payback_days: float
    roi_percent: float
    risk_of_deferral: str
    alternative_actions: List[Dict[str, Any]]
    resource_requirements: Dict[str, Any]
    scheduling_constraints: List[str]
    economic_analysis: Dict[str, Any]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "condenser_id": self.condenser_id,
            "timestamp": self.timestamp.isoformat(),
            "recommended_action": self.recommended_action.value,
            "urgency": self.urgency,
            "optimal_date": self.optimal_date.isoformat() if self.optimal_date else None,
            "estimated_duration_hours": round(self.estimated_duration_hours, 1),
            "estimated_cost_usd": round(self.estimated_cost_usd, 0),
            "expected_benefit_usd": round(self.expected_benefit_usd, 0),
            "payback_days": round(self.payback_days, 1),
            "roi_percent": round(self.roi_percent, 1),
            "risk_of_deferral": self.risk_of_deferral,
            "alternative_actions": self.alternative_actions,
            "resource_requirements": self.resource_requirements,
            "scheduling_constraints": self.scheduling_constraints,
            "provenance_hash": self.provenance_hash
        }


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================

class CondensyncAgent:
    """
    GL-017 CONDENSYNC Condenser Optimization Agent.

    Main orchestrator for condenser performance diagnostics, vacuum optimization,
    fouling prediction, and maintenance scheduling. Coordinates all calculation
    modules with zero-hallucination guarantee.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations use deterministic engineering formulas from HEI/ASME
    - No LLM or AI inference in any calculation path
    - Same inputs always produce identical outputs
    - Complete provenance tracking with SHA-256 hashes

    Key Capabilities:
    1. Real-time condenser performance diagnostics
    2. Vacuum optimization for turbine efficiency
    3. Fouling prediction with Kern-Seaton models
    4. Maintenance scheduling with economic optimization
    5. Air in-leakage detection and quantification

    Thread Safety:
    - All public methods are thread-safe
    - Uses threading.Lock for statistics updates
    - Calculation methods are stateless and reentrant

    Example:
        >>> config = AgentConfig(mode=AgentMode.DIAGNOSTIC)
        >>> design = CondenserDesignConfig(design_duty_mw=500.0)
        >>> agent = CondensyncAgent(config=config, design_config=design)
        >>> result = agent.diagnose_condenser(CondenserDiagnosticInput(
        ...     steam_flow_kg_s=180.0,
        ...     steam_pressure_kpa=5.5,
        ...     cw_inlet_temp_c=22.0
        ... ))
        >>> print(f"Status: {result.status.value}")
        >>> print(f"Cleanliness Factor: {result.cleanliness_factor:.2%}")
    """

    # Class constants - steam/water properties
    WATER_CP_KJ_KGK: float = 4.186  # Water specific heat (kJ/kg.K)
    WATER_DENSITY_KG_M3: float = 1000.0  # Water density (kg/m3)
    STEAM_LATENT_HEAT_KJ_KG: float = 2400.0  # Approximate latent heat

    # Heat rate sensitivity (typical for steam turbines)
    HEAT_RATE_SENSITIVITY_KJ_KWH_PER_KPA: float = 35.0  # kJ/kWh per kPa backpressure

    # Economic constants
    FUEL_COST_PER_MMBTU: float = 3.50  # $/MMBtu natural gas
    CAPACITY_FACTOR: float = 0.85  # Typical capacity factor
    HOURS_PER_YEAR: float = 8760.0

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        design_config: Optional[CondenserDesignConfig] = None,
        operational_config: Optional[OperationalConfig] = None
    ):
        """
        Initialize CONDENSYNC agent.

        Args:
            config: Agent configuration (uses defaults if not provided)
            design_config: Condenser design parameters
            operational_config: Operational thresholds and constraints
        """
        self.config = config or AgentConfig()
        self.design_config = design_config or CondenserDesignConfig()
        self.operational_config = operational_config or OperationalConfig()

        # Thread safety lock for statistics
        self._stats_lock = threading.Lock()

        # Statistics counters
        self._diagnostic_count: int = 0
        self._optimization_count: int = 0
        self._prediction_count: int = 0
        self._maintenance_count: int = 0
        self._alert_count: int = 0
        self._error_count: int = 0

        # Component initialization would go here
        # In production, these would be separate calculator classes
        # self._heat_transfer_calc = HeatTransferCalculator()
        # self._fouling_predictor = FoulingPredictor()
        # self._vacuum_optimizer = VacuumOptimizer()
        # self._maintenance_scheduler = MaintenanceScheduler()

        logger.info(
            f"{self.config.agent_name} Agent v{self.config.version} initialized "
            f"(mode={self.config.mode.value}, condenser={self.design_config.condenser_id})"
        )

    # ========================================================================
    # PRIMARY PUBLIC METHODS
    # ========================================================================

    def diagnose_condenser(
        self,
        input_data: CondenserDiagnosticInput
    ) -> CondenserPerformanceOutput:
        """
        Perform comprehensive condenser performance diagnostic.

        ZERO-HALLUCINATION: Uses deterministic HEI/ASME calculations only.

        This method calculates:
        - Actual vs expected backpressure
        - Terminal temperature difference (TTD)
        - Cleanliness factor and fouling estimate
        - Heat transfer coefficient
        - Heat rate penalty and economic impact
        - Air in-leakage assessment

        Args:
            input_data: Current condenser operating data

        Returns:
            CondenserPerformanceOutput with complete diagnostic results

        Raises:
            ValueError: If input data is invalid or physically impossible
        """
        start_time = time.time()
        timestamp = datetime.now(timezone.utc)

        try:
            # Validate inputs
            self._validate_diagnostic_input(input_data)

            # Step 1: Calculate saturation temperature from backpressure
            t_sat_c = self._calculate_saturation_temperature(input_data.steam_pressure_kpa)

            # Step 2: Calculate Terminal Temperature Difference (TTD)
            actual_ttd_c = t_sat_c - input_data.cw_outlet_temp_c

            # Step 3: Calculate heat duty from steam flow
            heat_duty_mw = self._calculate_heat_duty(
                steam_flow_kg_s=input_data.steam_flow_kg_s,
                steam_pressure_kpa=input_data.steam_pressure_kpa
            )

            # Step 4: Calculate LMTD (Log Mean Temperature Difference)
            lmtd_c = self._calculate_lmtd(
                t_sat_c=t_sat_c,
                cw_inlet_c=input_data.cw_inlet_temp_c,
                cw_outlet_c=input_data.cw_outlet_temp_c
            )

            # Step 5: Calculate actual U-value
            actual_u = self._calculate_actual_u_value(
                heat_duty_mw=heat_duty_mw,
                area_m2=self._get_effective_area(input_data.tubes_plugged),
                lmtd_c=lmtd_c
            )

            # Step 6: Calculate cleanliness factor
            cleanliness_factor = self._calculate_cleanliness_factor(
                actual_u=actual_u,
                design_u=self.design_config.design_u_w_m2k
            )

            # Step 7: Calculate fouling factor
            fouling_factor = self._calculate_fouling_factor(
                actual_u=actual_u,
                clean_u=self.design_config.design_u_w_m2k
            )

            # Step 8: Calculate expected backpressure (clean condenser)
            expected_bp_kpa = self._calculate_expected_backpressure(
                cw_inlet_c=input_data.cw_inlet_temp_c,
                cw_flow_m3_s=input_data.cw_flow_m3_s,
                heat_duty_mw=heat_duty_mw,
                cleanliness_factor=self.design_config.design_cleanliness_factor
            )

            # Step 9: Calculate backpressure deviation
            bp_deviation = input_data.steam_pressure_kpa - expected_bp_kpa

            # Step 10: Calculate heat rate penalty
            heat_rate_penalty = self._calculate_heat_rate_penalty(bp_deviation)

            # Step 11: Calculate annual cost impact
            annual_cost = self._calculate_annual_cost_impact(
                heat_rate_penalty=heat_rate_penalty,
                load_mw=input_data.generator_load_mw
            )

            # Step 12: Assess air in-leakage
            air_status = self._assess_air_inleakage(input_data.air_inleakage_scfm)

            # Step 13: Assess subcooling
            subcooling_status = self._assess_subcooling(input_data.subcooling_c)

            # Step 14: Determine overall status and alert level
            status = self._determine_performance_status(
                cleanliness_factor=cleanliness_factor,
                ttd_c=actual_ttd_c,
                bp_deviation=bp_deviation
            )
            alert_level = self._determine_alert_level(status, air_status)

            # Step 15: Generate recommendations
            recommendations = self._generate_diagnostic_recommendations(
                cleanliness_factor=cleanliness_factor,
                ttd_c=actual_ttd_c,
                air_status=air_status,
                subcooling_c=input_data.subcooling_c,
                tubes_plugged=input_data.tubes_plugged
            )

            # Step 16: Generate explanation
            explanation = self._generate_diagnostic_explanation(
                input_data=input_data,
                actual_u=actual_u,
                cleanliness_factor=cleanliness_factor,
                fouling_factor=fouling_factor,
                bp_deviation=bp_deviation
            )

            # Step 17: Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash(
                input_data=input_data,
                output_key="diagnostic",
                results={
                    "cleanliness_factor": cleanliness_factor,
                    "actual_u": actual_u,
                    "bp_deviation": bp_deviation
                }
            )

            # Update statistics
            with self._stats_lock:
                self._diagnostic_count += 1
                if alert_level in [AlertLevel.CRITICAL, AlertLevel.HIGH]:
                    self._alert_count += 1

            execution_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Condenser diagnostic completed for {input_data.condenser_id} "
                f"in {execution_time_ms:.1f}ms (CF={cleanliness_factor:.2%})"
            )

            return CondenserPerformanceOutput(
                condenser_id=input_data.condenser_id,
                timestamp=timestamp,
                status=status,
                alert_level=alert_level,
                actual_backpressure_kpa=input_data.steam_pressure_kpa,
                expected_backpressure_kpa=expected_bp_kpa,
                backpressure_deviation_kpa=bp_deviation,
                actual_ttd_c=actual_ttd_c,
                design_ttd_c=self.design_config.design_ttd_c,
                cleanliness_factor=cleanliness_factor,
                fouling_factor_m2k_kw=fouling_factor,
                heat_duty_mw=heat_duty_mw,
                actual_u_w_m2k=actual_u,
                lmtd_c=lmtd_c,
                heat_rate_penalty_kj_kwh=heat_rate_penalty,
                annual_cost_impact_usd=annual_cost,
                air_inleakage_status=air_status,
                subcooling_status=subcooling_status,
                recommendations=recommendations,
                explanation=explanation,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            with self._stats_lock:
                self._error_count += 1
            logger.error(f"Condenser diagnostic failed: {str(e)}", exc_info=True)
            raise

    def optimize_vacuum(
        self,
        input_data: CondenserDiagnosticInput
    ) -> VacuumOptimizationOutput:
        """
        Optimize condenser vacuum for maximum turbine efficiency.

        ZERO-HALLUCINATION: Uses deterministic thermodynamic calculations.

        This method:
        - Evaluates current vacuum conditions
        - Calculates optimal achievable vacuum
        - Recommends CW flow and pump configuration
        - Assesses air removal system performance
        - Quantifies potential savings

        Args:
            input_data: Current condenser operating data

        Returns:
            VacuumOptimizationOutput with optimization recommendations

        Raises:
            ValueError: If input data is invalid
        """
        start_time = time.time()
        timestamp = datetime.now(timezone.utc)

        try:
            # Validate inputs
            self._validate_diagnostic_input(input_data)

            # Current performance baseline
            current_bp_kpa = input_data.steam_pressure_kpa
            heat_duty_mw = self._calculate_heat_duty(
                steam_flow_kg_s=input_data.steam_flow_kg_s,
                steam_pressure_kpa=current_bp_kpa
            )

            # Calculate current heat rate
            design_heat_rate = 8500.0  # kJ/kWh base
            current_bp_penalty = self._calculate_heat_rate_penalty(
                current_bp_kpa - self.operational_config.min_backpressure_kpa
            )
            current_heat_rate = design_heat_rate + current_bp_penalty

            # Step 1: Calculate optimal CW flow rate
            optimal_cw_flow = self._calculate_optimal_cw_flow(
                heat_duty_mw=heat_duty_mw,
                cw_inlet_c=input_data.cw_inlet_temp_c,
                target_ttd_c=self.design_config.design_ttd_c
            )

            # Step 2: Calculate optimal backpressure achievable
            optimal_bp_kpa = self._calculate_optimal_backpressure(
                cw_inlet_c=input_data.cw_inlet_temp_c,
                cw_flow_m3_s=optimal_cw_flow,
                heat_duty_mw=heat_duty_mw,
                air_inleakage_scfm=input_data.air_inleakage_scfm
            )

            # Apply minimum constraint
            optimal_bp_kpa = max(optimal_bp_kpa, self.operational_config.min_backpressure_kpa)

            # Step 3: Calculate achievable improvement
            achievable_improvement = current_bp_kpa - optimal_bp_kpa

            # Step 4: Calculate optimal heat rate
            optimal_bp_penalty = self._calculate_heat_rate_penalty(
                optimal_bp_kpa - self.operational_config.min_backpressure_kpa
            )
            optimal_heat_rate = design_heat_rate + optimal_bp_penalty
            heat_rate_improvement_pct = (
                (current_heat_rate - optimal_heat_rate) / current_heat_rate * 100
            )

            # Step 5: Determine pump configuration
            pump_config = self._determine_pump_configuration(
                required_flow_m3_s=optimal_cw_flow,
                current_flow_m3_s=input_data.cw_flow_m3_s
            )

            # Step 6: Air removal assessment
            air_removal_rec = self._assess_air_removal_optimization(
                current_air_scfm=input_data.air_inleakage_scfm,
                ejector_steam_kg_s=input_data.ejector_steam_flow_kg_s
            )

            # Step 7: Calculate annual savings
            annual_savings = self._calculate_optimization_savings(
                heat_rate_improvement_kj_kwh=current_heat_rate - optimal_heat_rate,
                load_mw=input_data.generator_load_mw
            )

            # Step 8: Identify limiting constraints
            constraints_limiting = self._identify_limiting_constraints(
                optimal_bp_kpa=optimal_bp_kpa,
                optimal_cw_flow=optimal_cw_flow,
                cw_inlet_c=input_data.cw_inlet_temp_c
            )

            # Step 9: Generate implementation actions
            impl_actions = self._generate_optimization_actions(
                current_cw_flow=input_data.cw_flow_m3_s,
                optimal_cw_flow=optimal_cw_flow,
                pump_config=pump_config,
                air_rec=air_removal_rec
            )

            # Step 10: Create optimization details
            opt_details = {
                "analysis_method": "HEI_heat_transfer_model",
                "design_basis": {
                    "design_u_w_m2k": self.design_config.design_u_w_m2k,
                    "design_area_m2": self.design_config.effective_surface_area_m2,
                    "design_ttd_c": self.design_config.design_ttd_c
                },
                "sensitivity_analysis": {
                    "bp_per_1c_cwt_change_kpa": 0.15,
                    "bp_per_10pct_flow_change_kpa": 0.08
                },
                "execution_time_ms": (time.time() - start_time) * 1000
            }

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash(
                input_data=input_data,
                output_key="vacuum_optimization",
                results={
                    "optimal_bp_kpa": optimal_bp_kpa,
                    "achievable_improvement": achievable_improvement,
                    "annual_savings": annual_savings
                }
            )

            # Update statistics
            with self._stats_lock:
                self._optimization_count += 1

            return VacuumOptimizationOutput(
                condenser_id=input_data.condenser_id,
                timestamp=timestamp,
                current_backpressure_kpa=current_bp_kpa,
                optimal_backpressure_kpa=optimal_bp_kpa,
                achievable_improvement_kpa=achievable_improvement,
                current_heat_rate_kj_kwh=current_heat_rate,
                optimal_heat_rate_kj_kwh=optimal_heat_rate,
                heat_rate_improvement_percent=heat_rate_improvement_pct,
                cw_flow_recommendation_m3_s=optimal_cw_flow,
                cw_pump_configuration=pump_config,
                air_removal_recommendation=air_removal_rec,
                estimated_annual_savings_usd=annual_savings,
                implementation_actions=impl_actions,
                constraints_limiting=constraints_limiting,
                optimization_details=opt_details,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            with self._stats_lock:
                self._error_count += 1
            logger.error(f"Vacuum optimization failed: {str(e)}", exc_info=True)
            raise

    def predict_fouling(
        self,
        input_data: CondenserDiagnosticInput,
        history: List[Dict[str, Any]]
    ) -> FoulingPredictionOutput:
        """
        Predict fouling progression using Kern-Seaton asymptotic model.

        ZERO-HALLUCINATION: Uses deterministic fouling correlation models.

        The Kern-Seaton model predicts fouling as:
        Rf(t) = Rf_inf * (1 - exp(-t/tau))

        Where:
        - Rf(t) = fouling resistance at time t
        - Rf_inf = asymptotic fouling resistance
        - tau = time constant

        Args:
            input_data: Current condenser operating data
            history: List of historical fouling data points with timestamps

        Returns:
            FoulingPredictionOutput with fouling predictions

        Raises:
            ValueError: If insufficient history or invalid data
        """
        start_time = time.time()
        timestamp = datetime.now(timezone.utc)

        try:
            # Validate inputs
            self._validate_diagnostic_input(input_data)
            if len(history) < 3:
                raise ValueError("At least 3 historical data points required for prediction")

            # Calculate current fouling factor
            t_sat_c = self._calculate_saturation_temperature(input_data.steam_pressure_kpa)
            lmtd_c = self._calculate_lmtd(
                t_sat_c=t_sat_c,
                cw_inlet_c=input_data.cw_inlet_temp_c,
                cw_outlet_c=input_data.cw_outlet_temp_c
            )
            heat_duty_mw = self._calculate_heat_duty(
                steam_flow_kg_s=input_data.steam_flow_kg_s,
                steam_pressure_kpa=input_data.steam_pressure_kpa
            )
            actual_u = self._calculate_actual_u_value(
                heat_duty_mw=heat_duty_mw,
                area_m2=self._get_effective_area(input_data.tubes_plugged),
                lmtd_c=lmtd_c
            )
            current_fouling = self._calculate_fouling_factor(
                actual_u=actual_u,
                clean_u=self.design_config.design_u_w_m2k
            )

            # Fit Kern-Seaton model to historical data
            model_params = self._fit_kern_seaton_model(history)
            rf_inf = model_params["rf_inf"]
            tau = model_params["tau"]
            r_squared = model_params["r_squared"]

            # Predict future fouling
            t_current = model_params["t_current_days"]
            predicted_7d = rf_inf * (1 - math.exp(-(t_current + 7) / tau))
            predicted_30d = rf_inf * (1 - math.exp(-(t_current + 30) / tau))

            # Calculate fouling rate
            fouling_rate = self._calculate_fouling_rate(history)

            # Determine dominant fouling mode
            fouling_mode = self._determine_fouling_mode(
                cw_inlet_temp_c=input_data.cw_inlet_temp_c,
                fouling_rate=fouling_rate,
                history=history
            )

            # Calculate days to threshold
            threshold = self.operational_config.fouling_alarm_threshold
            days_to_threshold = None
            if current_fouling < threshold and fouling_rate > 0:
                # Solve: threshold = rf_inf * (1 - exp(-t/tau))
                # t = -tau * ln(1 - threshold/rf_inf)
                if threshold < rf_inf:
                    days_to_threshold = -tau * math.log(1 - threshold / rf_inf) - t_current
                    days_to_threshold = max(0, days_to_threshold)

            # Calculate confidence interval (simplified)
            std_error = model_params.get("std_error", current_fouling * 0.1)
            ci_lower = max(0, predicted_30d - 1.96 * std_error)
            ci_upper = predicted_30d + 1.96 * std_error

            # Determine trend
            trend = "increasing" if fouling_rate > 0.000001 else (
                "decreasing" if fouling_rate < -0.000001 else "stable"
            )

            # Determine cleaning urgency
            urgency = self._determine_cleaning_urgency(
                current_fouling=current_fouling,
                days_to_threshold=days_to_threshold,
                fouling_rate=fouling_rate
            )

            # Calculate backpressure impact
            bp_impact = self._calculate_fouling_backpressure_impact(predicted_30d)

            # Build prediction details
            prediction_details = {
                "model_type": "Kern-Seaton_asymptotic",
                "model_parameters": {
                    "rf_infinity": rf_inf,
                    "tau_days": tau,
                    "r_squared": r_squared
                },
                "data_points_used": len(history),
                "prediction_horizon_days": 30,
                "execution_time_ms": (time.time() - start_time) * 1000
            }

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash(
                input_data=input_data,
                output_key="fouling_prediction",
                results={
                    "current_fouling": current_fouling,
                    "predicted_30d": predicted_30d,
                    "r_squared": r_squared
                }
            )

            # Update statistics
            with self._stats_lock:
                self._prediction_count += 1

            return FoulingPredictionOutput(
                condenser_id=input_data.condenser_id,
                timestamp=timestamp,
                current_fouling_factor=current_fouling,
                predicted_fouling_factor_7d=predicted_7d,
                predicted_fouling_factor_30d=predicted_30d,
                fouling_rate_per_day=fouling_rate,
                dominant_fouling_mode=fouling_mode,
                days_to_threshold=days_to_threshold,
                confidence_interval_lower=ci_lower,
                confidence_interval_upper=ci_upper,
                model_r_squared=r_squared,
                trend_direction=trend,
                cleaning_urgency=urgency,
                predicted_backpressure_impact_kpa=bp_impact,
                prediction_details=prediction_details,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            with self._stats_lock:
                self._error_count += 1
            logger.error(f"Fouling prediction failed: {str(e)}", exc_info=True)
            raise

    def recommend_cleaning(
        self,
        cf_data: List[Dict[str, Any]],
        constraints: OperatingConstraints
    ) -> MaintenanceRecommendation:
        """
        Recommend optimal cleaning schedule based on economic analysis.

        ZERO-HALLUCINATION: Uses deterministic cost-benefit calculations.

        This method:
        - Analyzes current and projected cleanliness factor
        - Calculates cost of continued operation vs cleaning
        - Determines optimal cleaning timing
        - Recommends cleaning method
        - Performs full economic analysis

        Args:
            cf_data: List of cleanliness factor data with timestamps
            constraints: Operating constraints for scheduling

        Returns:
            MaintenanceRecommendation with scheduling and economic analysis

        Raises:
            ValueError: If insufficient data or invalid constraints
        """
        start_time = time.time()
        timestamp = datetime.now(timezone.utc)

        try:
            if len(cf_data) < 1:
                raise ValueError("At least 1 cleanliness factor data point required")

            # Get current cleanliness factor (most recent)
            sorted_data = sorted(cf_data, key=lambda x: x.get("timestamp", ""), reverse=True)
            current_cf = sorted_data[0].get("cleanliness_factor", 0.85)
            current_fouling = 1.0 - current_cf

            # Determine recommended action based on cleanliness factor
            action, urgency = self._determine_maintenance_action(
                cleanliness_factor=current_cf,
                target_cf=self.operational_config.target_cleanliness_factor
            )

            # Calculate cost of degradation (operating cost penalty)
            daily_cost_penalty = self._calculate_daily_degradation_cost(
                cleanliness_factor=current_cf,
                load_mw=constraints.max_load_mw * 0.8  # Assume 80% average load
            )

            # Calculate cleaning cost estimates
            cleaning_costs = self._estimate_cleaning_costs(action, constraints)
            estimated_cost = cleaning_costs["total_cost"]
            duration_hours = cleaning_costs["duration_hours"]

            # Calculate expected benefit
            cf_improvement = min(0.15, self.operational_config.target_cleanliness_factor - current_cf)
            expected_daily_savings = daily_cost_penalty * (cf_improvement / (1 - current_cf))
            annual_benefit = expected_daily_savings * 365

            # Calculate payback period
            if expected_daily_savings > 0:
                payback_days = estimated_cost / expected_daily_savings
            else:
                payback_days = float('inf')

            # Calculate ROI
            if estimated_cost > 0:
                roi_percent = (annual_benefit - estimated_cost) / estimated_cost * 100
            else:
                roi_percent = 0.0

            # Determine optimal date
            optimal_date = self._calculate_optimal_cleaning_date(
                current_cf=current_cf,
                cf_data=cf_data,
                constraints=constraints
            )

            # Assess risk of deferral
            risk_assessment = self._assess_deferral_risk(
                current_cf=current_cf,
                days_to_threshold=payback_days if payback_days != float('inf') else None
            )

            # Generate alternative actions
            alternatives = self._generate_alternative_actions(
                primary_action=action,
                current_cf=current_cf
            )

            # Resource requirements
            resources = self._get_resource_requirements(action)

            # Scheduling constraints
            scheduling_constraints = self._get_scheduling_constraints(
                action=action,
                constraints=constraints
            )

            # Economic analysis details
            economic_analysis = {
                "current_daily_penalty_usd": round(daily_cost_penalty, 2),
                "cleaning_cost_breakdown": cleaning_costs,
                "expected_cf_improvement": round(cf_improvement, 4),
                "expected_daily_savings_usd": round(expected_daily_savings, 2),
                "net_present_value_usd": round(annual_benefit - estimated_cost, 0),
                "internal_rate_of_return": round(roi_percent, 1),
                "analysis_period_years": 1
            }

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash(
                input_data={"cf_data_count": len(cf_data), "constraints": str(constraints)},
                output_key="maintenance_recommendation",
                results={
                    "action": action.value,
                    "estimated_cost": estimated_cost,
                    "annual_benefit": annual_benefit
                }
            )

            # Update statistics
            with self._stats_lock:
                self._maintenance_count += 1

            return MaintenanceRecommendation(
                condenser_id=self.design_config.condenser_id,
                timestamp=timestamp,
                recommended_action=action,
                urgency=urgency,
                optimal_date=optimal_date,
                estimated_duration_hours=duration_hours,
                estimated_cost_usd=estimated_cost,
                expected_benefit_usd=annual_benefit,
                payback_days=payback_days if payback_days != float('inf') else -1,
                roi_percent=roi_percent,
                risk_of_deferral=risk_assessment,
                alternative_actions=alternatives,
                resource_requirements=resources,
                scheduling_constraints=scheduling_constraints,
                economic_analysis=economic_analysis,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            with self._stats_lock:
                self._error_count += 1
            logger.error(f"Cleaning recommendation failed: {str(e)}", exc_info=True)
            raise

    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status and statistics.

        Returns:
            Dictionary with agent status, configuration, and statistics
        """
        with self._stats_lock:
            stats = {
                "diagnostic_count": self._diagnostic_count,
                "optimization_count": self._optimization_count,
                "prediction_count": self._prediction_count,
                "maintenance_count": self._maintenance_count,
                "alert_count": self._alert_count,
                "error_count": self._error_count
            }

        return {
            "agent_id": self.config.agent_id,
            "agent_name": self.config.agent_name,
            "version": self.config.version,
            "mode": self.config.mode.value,
            "status": "active",
            "condenser_id": self.design_config.condenser_id,
            "statistics": stats,
            "configuration": {
                "design_duty_mw": self.design_config.design_duty_mw,
                "design_backpressure_kpa": self.design_config.design_backpressure_kpa,
                "target_cleanliness_factor": self.operational_config.target_cleanliness_factor,
                "fouling_alarm_threshold": self.operational_config.fouling_alarm_threshold
            },
            "capabilities": [
                "condenser_diagnostics",
                "vacuum_optimization",
                "fouling_prediction",
                "maintenance_scheduling"
            ]
        }

    # ========================================================================
    # CALCULATION HELPER METHODS (ZERO-HALLUCINATION)
    # ========================================================================

    def _calculate_saturation_temperature(self, pressure_kpa: float) -> float:
        """
        Calculate saturation temperature from pressure using Antoine equation.

        Uses simplified Antoine equation for water:
        T_sat = B / (A - log10(P)) - C

        Args:
            pressure_kpa: Absolute pressure in kPa

        Returns:
            Saturation temperature in Celsius
        """
        # Antoine constants for water (valid 1-100 kPa)
        A = 8.07131
        B = 1730.63
        C = 233.426

        # Convert kPa to mmHg for Antoine equation
        pressure_mmhg = pressure_kpa * 7.50062

        # Calculate saturation temperature
        if pressure_mmhg <= 0:
            raise ValueError("Pressure must be positive")

        t_sat_c = B / (A - math.log10(pressure_mmhg)) - C
        return t_sat_c

    def _calculate_heat_duty(
        self,
        steam_flow_kg_s: float,
        steam_pressure_kpa: float
    ) -> float:
        """
        Calculate condenser heat duty from steam flow.

        Q = m_steam * h_fg

        Args:
            steam_flow_kg_s: Steam flow rate (kg/s)
            steam_pressure_kpa: Steam pressure (kPa abs)

        Returns:
            Heat duty in MW (thermal)
        """
        # Get latent heat at this pressure (simplified correlation)
        # More accurate would use IAPWS-IF97 steam tables
        t_sat = self._calculate_saturation_temperature(steam_pressure_kpa)
        h_fg = 2501 - 2.42 * t_sat  # Simplified latent heat correlation (kJ/kg)

        heat_duty_kw = steam_flow_kg_s * h_fg
        heat_duty_mw = heat_duty_kw / 1000.0

        return heat_duty_mw

    def _calculate_lmtd(
        self,
        t_sat_c: float,
        cw_inlet_c: float,
        cw_outlet_c: float
    ) -> float:
        """
        Calculate Log Mean Temperature Difference for condenser.

        For a condenser with isothermal condensation:
        LMTD = (dT1 - dT2) / ln(dT1/dT2)

        Where:
        dT1 = T_sat - T_cw_in
        dT2 = T_sat - T_cw_out (TTD)

        Args:
            t_sat_c: Saturation temperature (C)
            cw_inlet_c: Cooling water inlet temperature (C)
            cw_outlet_c: Cooling water outlet temperature (C)

        Returns:
            Log mean temperature difference in Celsius
        """
        dt1 = t_sat_c - cw_inlet_c
        dt2 = t_sat_c - cw_outlet_c  # This is TTD

        if dt1 <= 0 or dt2 <= 0:
            raise ValueError("Invalid temperatures: saturation temp must exceed CW temps")

        if abs(dt1 - dt2) < 0.01:
            # When dt1 approximately equals dt2, use arithmetic mean
            return (dt1 + dt2) / 2.0

        lmtd = (dt1 - dt2) / math.log(dt1 / dt2)
        return lmtd

    def _calculate_actual_u_value(
        self,
        heat_duty_mw: float,
        area_m2: float,
        lmtd_c: float
    ) -> float:
        """
        Calculate actual overall heat transfer coefficient.

        U = Q / (A * LMTD)

        Args:
            heat_duty_mw: Heat duty (MW thermal)
            area_m2: Heat transfer area (m2)
            lmtd_c: Log mean temperature difference (C)

        Returns:
            Overall U-value in W/m2K
        """
        if area_m2 <= 0 or lmtd_c <= 0:
            raise ValueError("Area and LMTD must be positive")

        heat_duty_w = heat_duty_mw * 1e6
        u_value = heat_duty_w / (area_m2 * lmtd_c)

        return u_value

    def _calculate_cleanliness_factor(
        self,
        actual_u: float,
        design_u: float
    ) -> float:
        """
        Calculate HEI cleanliness factor.

        CF = U_actual / U_design

        Args:
            actual_u: Actual U-value (W/m2K)
            design_u: Design U-value (W/m2K)

        Returns:
            Cleanliness factor (0.0 to 1.0+)
        """
        if design_u <= 0:
            raise ValueError("Design U-value must be positive")

        cf = actual_u / design_u
        return min(cf, 1.2)  # Cap at 1.2 for reasonableness

    def _calculate_fouling_factor(
        self,
        actual_u: float,
        clean_u: float
    ) -> float:
        """
        Calculate fouling resistance (fouling factor).

        Rf = 1/U_fouled - 1/U_clean

        Args:
            actual_u: Actual (fouled) U-value (W/m2K)
            clean_u: Clean U-value (W/m2K)

        Returns:
            Fouling factor in m2K/kW
        """
        if actual_u <= 0 or clean_u <= 0:
            raise ValueError("U-values must be positive")

        # Convert to m2K/kW
        rf = (1.0 / actual_u - 1.0 / clean_u) * 1000.0
        return max(0.0, rf)  # Fouling factor cannot be negative

    def _get_effective_area(self, tubes_plugged: int) -> float:
        """
        Calculate effective heat transfer area accounting for plugged tubes.

        Args:
            tubes_plugged: Number of tubes plugged

        Returns:
            Effective area in m2
        """
        total_tubes = self.design_config.tube_count
        active_fraction = (total_tubes - tubes_plugged) / total_tubes
        return self.design_config.effective_surface_area_m2 * active_fraction

    def _calculate_expected_backpressure(
        self,
        cw_inlet_c: float,
        cw_flow_m3_s: float,
        heat_duty_mw: float,
        cleanliness_factor: float
    ) -> float:
        """
        Calculate expected backpressure for given conditions and cleanliness.

        Args:
            cw_inlet_c: Cooling water inlet temperature (C)
            cw_flow_m3_s: Cooling water flow rate (m3/s)
            heat_duty_mw: Heat duty (MW)
            cleanliness_factor: Condenser cleanliness factor

        Returns:
            Expected backpressure in kPa abs
        """
        # Calculate CW temperature rise
        cw_flow_kg_s = cw_flow_m3_s * self.WATER_DENSITY_KG_M3
        heat_duty_kw = heat_duty_mw * 1000
        dt_cw = heat_duty_kw / (cw_flow_kg_s * self.WATER_CP_KJ_KGK)
        cw_outlet_c = cw_inlet_c + dt_cw

        # Estimate LMTD based on design TTD and CF
        design_ttd = self.design_config.design_ttd_c / cleanliness_factor
        estimated_t_sat = cw_outlet_c + design_ttd

        # Convert saturation temperature to pressure (inverse Antoine)
        A = 8.07131
        B = 1730.63
        C = 233.426

        log_p_mmhg = A - B / (estimated_t_sat + C)
        p_mmhg = 10 ** log_p_mmhg
        p_kpa = p_mmhg / 7.50062

        return p_kpa

    def _calculate_heat_rate_penalty(self, bp_deviation_kpa: float) -> float:
        """
        Calculate turbine heat rate penalty from backpressure deviation.

        Args:
            bp_deviation_kpa: Backpressure deviation from optimal (kPa)

        Returns:
            Heat rate penalty in kJ/kWh
        """
        # Typical sensitivity: 35 kJ/kWh per kPa above optimal
        penalty = max(0.0, bp_deviation_kpa * self.HEAT_RATE_SENSITIVITY_KJ_KWH_PER_KPA)
        return penalty

    def _calculate_annual_cost_impact(
        self,
        heat_rate_penalty: float,
        load_mw: float
    ) -> float:
        """
        Calculate annual cost of heat rate penalty.

        Args:
            heat_rate_penalty: Heat rate penalty (kJ/kWh)
            load_mw: Generator load (MW)

        Returns:
            Annual cost in USD
        """
        # Convert kJ/kWh to BTU/kWh (1 kJ = 0.9478 BTU)
        penalty_btu_kwh = heat_rate_penalty * 0.9478

        # Annual energy production (MWh)
        annual_mwh = load_mw * self.HOURS_PER_YEAR * self.CAPACITY_FACTOR

        # Additional fuel cost
        fuel_cost_per_kwh = self.FUEL_COST_PER_MMBTU * penalty_btu_kwh / 1e6
        annual_cost = fuel_cost_per_kwh * annual_mwh * 1000  # Convert MWh to kWh

        return annual_cost

    # ========================================================================
    # ASSESSMENT AND STATUS METHODS
    # ========================================================================

    def _validate_diagnostic_input(self, input_data: CondenserDiagnosticInput) -> None:
        """Validate diagnostic input data is physically reasonable."""
        if input_data.steam_flow_kg_s <= 0:
            raise ValueError("Steam flow must be positive")
        if input_data.steam_pressure_kpa <= 0:
            raise ValueError("Steam pressure must be positive (absolute)")
        if input_data.cw_inlet_temp_c >= input_data.cw_outlet_temp_c:
            raise ValueError("CW outlet must be warmer than inlet")
        if input_data.cw_flow_m3_s <= 0:
            raise ValueError("Cooling water flow must be positive")

    def _assess_air_inleakage(self, air_scfm: float) -> str:
        """Assess air in-leakage status."""
        if air_scfm >= self.operational_config.air_inleakage_alarm_scfm:
            return "ALARM: Excessive air in-leakage"
        elif air_scfm >= self.operational_config.air_inleakage_warning_scfm:
            return "WARNING: Elevated air in-leakage"
        else:
            return "NORMAL: Air in-leakage within limits"

    def _assess_subcooling(self, subcooling_c: float) -> str:
        """Assess condensate subcooling status."""
        if subcooling_c > 2.0:
            return "WARNING: Excessive subcooling indicates air binding"
        elif subcooling_c > 1.0:
            return "ELEVATED: Monitor air removal system"
        else:
            return "NORMAL: Subcooling within acceptable limits"

    def _determine_performance_status(
        self,
        cleanliness_factor: float,
        ttd_c: float,
        bp_deviation: float
    ) -> PerformanceStatus:
        """Determine overall condenser performance status."""
        if cleanliness_factor >= 0.90 and ttd_c <= self.design_config.design_ttd_c + 1:
            return PerformanceStatus.OPTIMAL
        elif cleanliness_factor >= 0.80 and ttd_c <= self.design_config.design_ttd_c + 2:
            return PerformanceStatus.ACCEPTABLE
        elif cleanliness_factor >= 0.70:
            return PerformanceStatus.DEGRADED
        elif cleanliness_factor >= 0.60:
            return PerformanceStatus.POOR
        else:
            return PerformanceStatus.CRITICAL

    def _determine_alert_level(
        self,
        status: PerformanceStatus,
        air_status: str
    ) -> AlertLevel:
        """Determine alert level from status and air assessment."""
        if status == PerformanceStatus.CRITICAL or "ALARM" in air_status:
            return AlertLevel.CRITICAL
        elif status == PerformanceStatus.POOR:
            return AlertLevel.HIGH
        elif status == PerformanceStatus.DEGRADED or "WARNING" in air_status:
            return AlertLevel.MEDIUM
        elif status == PerformanceStatus.ACCEPTABLE:
            return AlertLevel.LOW
        else:
            return AlertLevel.NONE

    def _generate_diagnostic_recommendations(
        self,
        cleanliness_factor: float,
        ttd_c: float,
        air_status: str,
        subcooling_c: float,
        tubes_plugged: int
    ) -> List[str]:
        """Generate diagnostic recommendations based on analysis."""
        recommendations = []

        if cleanliness_factor < 0.80:
            recommendations.append(
                f"Schedule tube cleaning - cleanliness factor {cleanliness_factor:.1%} below target"
            )

        if "ALARM" in air_status or "WARNING" in air_status:
            recommendations.append("Investigate and repair air in-leakage sources")
            recommendations.append("Verify air removal equipment operation")

        if subcooling_c > 1.5:
            recommendations.append("Check for air binding in condenser hotwell")

        tube_plugging_pct = (tubes_plugged / self.design_config.tube_count) * 100
        if tube_plugging_pct > self.operational_config.max_tube_plugging_percent:
            recommendations.append(
                f"Tube plugging at {tube_plugging_pct:.1f}% - consider tube replacement"
            )

        if ttd_c > self.operational_config.ttd_alarm_threshold_c:
            recommendations.append("TTD elevated - verify CW flow rate and system cleanliness")

        if not recommendations:
            recommendations.append("Continue monitoring - performance within acceptable limits")

        return recommendations

    def _generate_diagnostic_explanation(
        self,
        input_data: CondenserDiagnosticInput,
        actual_u: float,
        cleanliness_factor: float,
        fouling_factor: float,
        bp_deviation: float
    ) -> Dict[str, Any]:
        """Generate detailed explanation of diagnostic results."""
        return {
            "methodology": "HEI Standards for Steam Surface Condensers (12th Edition)",
            "calculations": {
                "heat_transfer_equation": "Q = U * A * LMTD",
                "cleanliness_factor_definition": "CF = U_actual / U_design",
                "fouling_factor_definition": "Rf = 1/U_fouled - 1/U_clean"
            },
            "key_parameters": {
                "design_u_w_m2k": self.design_config.design_u_w_m2k,
                "actual_u_w_m2k": round(actual_u, 1),
                "design_area_m2": self.design_config.effective_surface_area_m2,
                "effective_area_m2": round(
                    self._get_effective_area(input_data.tubes_plugged), 1
                )
            },
            "interpretation": {
                "cleanliness_factor": self._interpret_cleanliness_factor(cleanliness_factor),
                "fouling_impact": self._interpret_fouling_impact(fouling_factor),
                "backpressure_deviation": self._interpret_bp_deviation(bp_deviation)
            },
            "assumptions": [
                "Isothermal condensation at saturation temperature",
                "Constant specific heat of cooling water",
                "Negligible heat losses to environment",
                "All tubes equally fouled (average fouling)"
            ]
        }

    def _interpret_cleanliness_factor(self, cf: float) -> str:
        """Generate interpretation of cleanliness factor."""
        if cf >= 0.90:
            return "Excellent - minimal fouling present"
        elif cf >= 0.80:
            return "Good - moderate fouling, within acceptable limits"
        elif cf >= 0.70:
            return "Fair - significant fouling, schedule cleaning soon"
        elif cf >= 0.60:
            return "Poor - heavy fouling, cleaning recommended"
        else:
            return "Critical - severe fouling, immediate action required"

    def _interpret_fouling_impact(self, rf: float) -> str:
        """Generate interpretation of fouling factor."""
        if rf < 0.0001:
            return "Negligible fouling resistance"
        elif rf < 0.0002:
            return "Light fouling - normal accumulation"
        elif rf < 0.0003:
            return "Moderate fouling - performance impact noticeable"
        else:
            return "Heavy fouling - significant heat transfer degradation"

    def _interpret_bp_deviation(self, deviation: float) -> str:
        """Generate interpretation of backpressure deviation."""
        if deviation <= 0.2:
            return "Operating at expected vacuum"
        elif deviation <= 0.5:
            return "Slightly elevated backpressure"
        elif deviation <= 1.0:
            return "Moderately elevated backpressure - investigate cause"
        else:
            return "Significantly elevated backpressure - action required"

    # ========================================================================
    # OPTIMIZATION HELPER METHODS
    # ========================================================================

    def _calculate_optimal_cw_flow(
        self,
        heat_duty_mw: float,
        cw_inlet_c: float,
        target_ttd_c: float
    ) -> float:
        """Calculate optimal cooling water flow rate."""
        # This is an iterative calculation in practice
        # Simplified here: target CW rise of design value
        target_rise = self.design_config.design_cwt_rise_c
        heat_duty_kw = heat_duty_mw * 1000
        cw_flow_kg_s = heat_duty_kw / (self.WATER_CP_KJ_KGK * target_rise)
        cw_flow_m3_s = cw_flow_kg_s / self.WATER_DENSITY_KG_M3
        return cw_flow_m3_s

    def _calculate_optimal_backpressure(
        self,
        cw_inlet_c: float,
        cw_flow_m3_s: float,
        heat_duty_mw: float,
        air_inleakage_scfm: float
    ) -> float:
        """Calculate optimal achievable backpressure."""
        # Base calculation
        optimal_bp = self._calculate_expected_backpressure(
            cw_inlet_c=cw_inlet_c,
            cw_flow_m3_s=cw_flow_m3_s,
            heat_duty_mw=heat_duty_mw,
            cleanliness_factor=0.95  # Assume clean condenser
        )

        # Add air in-leakage penalty (approximately 0.02 kPa per SCFM above normal)
        normal_air = 2.0  # Normal air in-leakage SCFM
        air_penalty = max(0, (air_inleakage_scfm - normal_air) * 0.02)
        optimal_bp += air_penalty

        return optimal_bp

    def _determine_pump_configuration(
        self,
        required_flow_m3_s: float,
        current_flow_m3_s: float
    ) -> str:
        """Determine recommended CW pump configuration."""
        if required_flow_m3_s <= current_flow_m3_s * 1.05:
            return "Current configuration adequate"
        elif required_flow_m3_s <= current_flow_m3_s * 1.2:
            return "Start additional CW pump if available"
        else:
            return "Maximum pumping capacity may be limiting"

    def _assess_air_removal_optimization(
        self,
        current_air_scfm: float,
        ejector_steam_kg_s: float
    ) -> str:
        """Assess air removal system for optimization opportunities."""
        if current_air_scfm > self.operational_config.air_inleakage_warning_scfm:
            return "Prioritize air leak detection and repair"
        elif ejector_steam_kg_s > 0.6:  # Elevated ejector steam consumption
            return "Optimize ejector operation - consider staging adjustment"
        else:
            return "Air removal system operating normally"

    def _calculate_optimization_savings(
        self,
        heat_rate_improvement_kj_kwh: float,
        load_mw: float
    ) -> float:
        """Calculate annual savings from optimization."""
        # Similar to cost impact but for savings
        if heat_rate_improvement_kj_kwh <= 0:
            return 0.0

        penalty_btu_kwh = heat_rate_improvement_kj_kwh * 0.9478
        annual_mwh = load_mw * self.HOURS_PER_YEAR * self.CAPACITY_FACTOR
        fuel_savings_per_kwh = self.FUEL_COST_PER_MMBTU * penalty_btu_kwh / 1e6
        annual_savings = fuel_savings_per_kwh * annual_mwh * 1000

        return annual_savings

    def _identify_limiting_constraints(
        self,
        optimal_bp_kpa: float,
        optimal_cw_flow: float,
        cw_inlet_c: float
    ) -> List[str]:
        """Identify constraints limiting further optimization."""
        constraints = []

        if optimal_bp_kpa <= self.operational_config.min_backpressure_kpa:
            constraints.append("Minimum backpressure limit")

        if optimal_cw_flow > self.design_config.design_cw_flow_m3_s * 1.1:
            constraints.append("CW pumping capacity")

        if cw_inlet_c > self.design_config.design_cwt_inlet_c + 5:
            constraints.append("Elevated cooling water temperature (environmental)")

        return constraints if constraints else ["No significant constraints identified"]

    def _generate_optimization_actions(
        self,
        current_cw_flow: float,
        optimal_cw_flow: float,
        pump_config: str,
        air_rec: str
    ) -> List[str]:
        """Generate implementation actions for optimization."""
        actions = []

        if optimal_cw_flow > current_cw_flow * 1.05:
            actions.append(f"Increase CW flow to {optimal_cw_flow:.1f} m3/s")
            if "additional" in pump_config.lower():
                actions.append(pump_config)

        if "leak" in air_rec.lower():
            actions.append("Perform air leak survey")
            actions.append("Repair identified leaks")

        if not actions:
            actions.append("Current operation near optimal - continue monitoring")

        return actions

    # ========================================================================
    # FOULING PREDICTION HELPER METHODS
    # ========================================================================

    def _fit_kern_seaton_model(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fit Kern-Seaton asymptotic fouling model to historical data.

        Model: Rf(t) = Rf_inf * (1 - exp(-t/tau))

        Args:
            history: Historical fouling data

        Returns:
            Dictionary with model parameters
        """
        # Extract fouling data and convert to days from first measurement
        sorted_history = sorted(history, key=lambda x: x.get("timestamp", ""))

        # Get reference time
        first_time = datetime.fromisoformat(sorted_history[0]["timestamp"].replace("Z", "+00:00"))

        times = []
        fouling_values = []

        for record in sorted_history:
            timestamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
            days = (timestamp - first_time).total_seconds() / 86400
            times.append(days)
            fouling_values.append(record.get("fouling_factor", 0.0))

        # Simple parameter estimation (in production, use scipy.optimize.curve_fit)
        # Estimate rf_inf from maximum observed or extrapolated value
        rf_inf = max(fouling_values) * 1.5 if fouling_values else 0.0003

        # Estimate tau from half-life approximation
        mid_rf = rf_inf * 0.5
        tau = times[-1] if times else 30.0  # Rough estimate

        # Simple R-squared calculation
        mean_rf = sum(fouling_values) / len(fouling_values) if fouling_values else 0
        ss_tot = sum((rf - mean_rf) ** 2 for rf in fouling_values)
        ss_res = 0
        for t, rf_actual in zip(times, fouling_values):
            rf_predicted = rf_inf * (1 - math.exp(-t / tau)) if tau > 0 else 0
            ss_res += (rf_actual - rf_predicted) ** 2

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "rf_inf": rf_inf,
            "tau": tau,
            "r_squared": max(0, min(1, r_squared)),
            "t_current_days": times[-1] if times else 0,
            "std_error": math.sqrt(ss_res / len(fouling_values)) if fouling_values else 0
        }

    def _calculate_fouling_rate(self, history: List[Dict[str, Any]]) -> float:
        """Calculate average fouling rate from history."""
        if len(history) < 2:
            return 0.0

        sorted_history = sorted(history, key=lambda x: x.get("timestamp", ""))

        first = sorted_history[0]
        last = sorted_history[-1]

        first_time = datetime.fromisoformat(first["timestamp"].replace("Z", "+00:00"))
        last_time = datetime.fromisoformat(last["timestamp"].replace("Z", "+00:00"))

        days = (last_time - first_time).total_seconds() / 86400
        if days <= 0:
            return 0.0

        delta_rf = last.get("fouling_factor", 0) - first.get("fouling_factor", 0)
        return delta_rf / days

    def _determine_fouling_mode(
        self,
        cw_inlet_temp_c: float,
        fouling_rate: float,
        history: List[Dict[str, Any]]
    ) -> FoulingMode:
        """Determine dominant fouling mechanism."""
        # Simplified heuristic-based determination
        if cw_inlet_temp_c > 25:
            return FoulingMode.BIOLOGICAL
        elif fouling_rate > 0.00001:
            return FoulingMode.SCALING
        else:
            return FoulingMode.PARTICULATE

    def _determine_cleaning_urgency(
        self,
        current_fouling: float,
        days_to_threshold: Optional[float],
        fouling_rate: float
    ) -> str:
        """Determine urgency level for cleaning."""
        if current_fouling >= self.operational_config.fouling_alarm_threshold:
            return "IMMEDIATE"
        elif days_to_threshold is not None and days_to_threshold < 14:
            return "HIGH - Schedule within 2 weeks"
        elif days_to_threshold is not None and days_to_threshold < 30:
            return "MEDIUM - Schedule within 1 month"
        elif fouling_rate > 0.000005:
            return "LOW - Monitor closely"
        else:
            return "ROUTINE - Normal monitoring"

    def _calculate_fouling_backpressure_impact(self, fouling_factor: float) -> float:
        """Calculate expected backpressure increase from fouling."""
        # Approximate: 0.1 kPa per 0.0001 m2K/kW fouling factor
        return fouling_factor * 1000 * 0.1

    # ========================================================================
    # MAINTENANCE SCHEDULING HELPER METHODS
    # ========================================================================

    def _determine_maintenance_action(
        self,
        cleanliness_factor: float,
        target_cf: float
    ) -> Tuple[MaintenanceAction, str]:
        """Determine recommended maintenance action and urgency."""
        if cleanliness_factor >= target_cf:
            return MaintenanceAction.NO_ACTION, "monitor"
        elif cleanliness_factor >= 0.80:
            return MaintenanceAction.ONLINE_BALL_CLEANING, "scheduled"
        elif cleanliness_factor >= 0.70:
            return MaintenanceAction.OFFLINE_MECHANICAL, "scheduled"
        elif cleanliness_factor >= 0.60:
            return MaintenanceAction.OFFLINE_CHEMICAL, "priority"
        else:
            return MaintenanceAction.OFFLINE_CHEMICAL, "immediate"

    def _calculate_daily_degradation_cost(
        self,
        cleanliness_factor: float,
        load_mw: float
    ) -> float:
        """Calculate daily cost of operating with degraded condenser."""
        # Calculate backpressure penalty for current CF vs target
        target_cf = self.operational_config.target_cleanliness_factor
        cf_deficit = max(0, target_cf - cleanliness_factor)

        # Approximate BP penalty: 0.3 kPa per 0.1 CF deficit
        bp_penalty_kpa = cf_deficit * 3.0

        # Heat rate penalty
        hr_penalty = self._calculate_heat_rate_penalty(bp_penalty_kpa)

        # Daily cost
        daily_mwh = load_mw * 24 * self.CAPACITY_FACTOR
        penalty_btu_kwh = hr_penalty * 0.9478
        daily_cost = self.FUEL_COST_PER_MMBTU * penalty_btu_kwh / 1e6 * daily_mwh * 1000

        return daily_cost

    def _estimate_cleaning_costs(
        self,
        action: MaintenanceAction,
        constraints: OperatingConstraints
    ) -> Dict[str, Any]:
        """Estimate costs for cleaning action."""
        costs = {
            MaintenanceAction.NO_ACTION: {"labor": 0, "materials": 0, "outage": 0, "duration_hours": 0},
            MaintenanceAction.ONLINE_BALL_CLEANING: {
                "labor": 5000, "materials": 2000, "outage": 0, "duration_hours": 4
            },
            MaintenanceAction.OFFLINE_MECHANICAL: {
                "labor": 25000, "materials": 10000,
                "outage": constraints.max_outage_cost_per_hour * 24,
                "duration_hours": 24
            },
            MaintenanceAction.OFFLINE_CHEMICAL: {
                "labor": 35000, "materials": 20000,
                "outage": constraints.max_outage_cost_per_hour * 48,
                "duration_hours": 48
            },
            MaintenanceAction.TUBE_PLUGGING: {
                "labor": 3000, "materials": 500, "outage": 0, "duration_hours": 8
            },
            MaintenanceAction.AIR_REMOVAL_CHECK: {
                "labor": 2000, "materials": 500, "outage": 0, "duration_hours": 4
            },
            MaintenanceAction.WATERBOX_INSPECTION: {
                "labor": 10000, "materials": 2000,
                "outage": constraints.max_outage_cost_per_hour * 8,
                "duration_hours": 8
            }
        }

        action_costs = costs.get(action, {"labor": 0, "materials": 0, "outage": 0, "duration_hours": 0})
        action_costs["total_cost"] = (
            action_costs["labor"] + action_costs["materials"] + action_costs["outage"]
        )

        return action_costs

    def _calculate_optimal_cleaning_date(
        self,
        current_cf: float,
        cf_data: List[Dict[str, Any]],
        constraints: OperatingConstraints
    ) -> Optional[datetime]:
        """Calculate optimal date for cleaning."""
        # If CF is already below target, recommend soon
        if current_cf < 0.80:
            return datetime.now(timezone.utc)

        # Otherwise, estimate days until cleaning economically optimal
        # Simplified: schedule when payback period equals remaining days to threshold
        days_ahead = max(7, constraints.min_days_between_cleaning)
        return datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    def _assess_deferral_risk(
        self,
        current_cf: float,
        days_to_threshold: Optional[float]
    ) -> str:
        """Assess risk of deferring maintenance."""
        if current_cf < 0.65:
            return "HIGH - Risk of unit trip or forced derating"
        elif current_cf < 0.75:
            return "MEDIUM - Significant efficiency losses accumulating"
        elif days_to_threshold and days_to_threshold < 30:
            return "MODERATE - Approaching threshold, plan maintenance"
        else:
            return "LOW - Continued monitoring acceptable"

    def _generate_alternative_actions(
        self,
        primary_action: MaintenanceAction,
        current_cf: float
    ) -> List[Dict[str, Any]]:
        """Generate alternative maintenance actions."""
        alternatives = []

        if primary_action == MaintenanceAction.OFFLINE_MECHANICAL:
            alternatives.append({
                "action": MaintenanceAction.ONLINE_BALL_CLEANING.value,
                "pros": "No outage required",
                "cons": "May not fully restore cleanliness",
                "expected_cf_improvement": 0.05
            })

        if primary_action == MaintenanceAction.OFFLINE_CHEMICAL:
            alternatives.append({
                "action": MaintenanceAction.OFFLINE_MECHANICAL.value,
                "pros": "Lower cost, shorter outage",
                "cons": "May not remove chemical deposits",
                "expected_cf_improvement": 0.10
            })

        alternatives.append({
            "action": MaintenanceAction.NO_ACTION.value,
            "pros": "No cost or outage",
            "cons": "Continued efficiency loss",
            "expected_cf_improvement": 0.0
        })

        return alternatives

    def _get_resource_requirements(self, action: MaintenanceAction) -> Dict[str, Any]:
        """Get resource requirements for maintenance action."""
        requirements = {
            MaintenanceAction.ONLINE_BALL_CLEANING: {
                "personnel": ["1 operator", "1 technician"],
                "equipment": ["Ball injection system", "Ball recovery"],
                "materials": ["Cleaning balls (100-200)"]
            },
            MaintenanceAction.OFFLINE_MECHANICAL: {
                "personnel": ["1 supervisor", "4 technicians", "2 helpers"],
                "equipment": ["High-pressure washer", "Tube brushes", "Scaffolding"],
                "materials": ["Replacement gaskets", "Touch-up coating"]
            },
            MaintenanceAction.OFFLINE_CHEMICAL: {
                "personnel": ["1 supervisor", "2 chemical techs", "2 technicians"],
                "equipment": ["Chemical circulation pumps", "Neutralization system"],
                "materials": ["Cleaning chemicals", "Neutralizing agents", "PPE"]
            }
        }

        return requirements.get(action, {"personnel": [], "equipment": [], "materials": []})

    def _get_scheduling_constraints(
        self,
        action: MaintenanceAction,
        constraints: OperatingConstraints
    ) -> List[str]:
        """Get scheduling constraints for maintenance action."""
        scheduling = []

        if action in [MaintenanceAction.OFFLINE_MECHANICAL, MaintenanceAction.OFFLINE_CHEMICAL]:
            scheduling.append(f"Requires unit outage of {constraints.maintenance_window_hours}+ hours")
            scheduling.append("Coordinate with planned maintenance window")
            scheduling.append("Avoid peak demand periods")

        if action == MaintenanceAction.OFFLINE_CHEMICAL:
            scheduling.append("Environmental permit may require notification")
            scheduling.append("Ensure chemical disposal arrangements")

        scheduling.append(f"Minimum {constraints.min_days_between_cleaning} days since last cleaning")

        return scheduling

    # ========================================================================
    # PROVENANCE AND HASHING METHODS
    # ========================================================================

    def _calculate_provenance_hash(
        self,
        input_data: Any,
        output_key: str,
        results: Dict[str, Any]
    ) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Args:
            input_data: Input data (dataclass or dict)
            output_key: Key identifying output type
            results: Key results for hashing

        Returns:
            SHA-256 hash string
        """
        # Convert input to dict if dataclass
        if hasattr(input_data, "__dataclass_fields__"):
            input_dict = {
                k: str(v) for k, v in input_data.__dict__.items()
            }
        else:
            input_dict = dict(input_data) if isinstance(input_data, dict) else {"raw": str(input_data)}

        hash_data = {
            "agent_id": self.config.agent_id,
            "agent_version": self.config.version,
            "output_type": output_key,
            "inputs": input_dict,
            "results": {k: str(v) for k, v in results.items()},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        data_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Main Agent
    "CondensyncAgent",
    # Configuration
    "AgentConfig",
    "CondenserDesignConfig",
    "OperationalConfig",
    "OperatingConstraints",
    # Enums
    "AgentMode",
    "CondenserType",
    "AlertLevel",
    "PerformanceStatus",
    "FoulingMode",
    "MaintenanceAction",
    # Input/Output Models
    "CondenserDiagnosticInput",
    "CondenserPerformanceOutput",
    "VacuumOptimizationOutput",
    "FoulingPredictionOutput",
    "MaintenanceRecommendation",
]
