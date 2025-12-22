# -*- coding: utf-8 -*-
"""
Feedforward Controller for GL-005 CombustionControlAgent

Implements feedforward control to anticipate heat demand changes and adjust
fuel flow proactively. Zero-hallucination design using deterministic physics models.

Reference Standards:
- ISA-5.1: Instrumentation Symbols and Identification
- ANSI/ISA-51.1: Process Instrumentation Terminology
- Astrom & Murray: Feedback Systems - Feedforward Control Chapter
- Seborg et al.: Process Dynamics and Control

Mathematical Formulas:
- Feedforward Control: u_ff(t) = K_ff * d(t) + τ_ff * dd(t)/dt
- Heat Demand Prediction: Q̇_demand = ṁ_steam * h_fg + Q̇_losses
- Fuel Flow Prediction: ṁ_fuel = Q̇_demand / (LHV * η_combustion)
- Lead-Lag Compensation: G_ff(s) = K_ff * (τ_lead*s + 1) / (τ_lag*s + 1)
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import math
import logging
import hashlib
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


class DisturbanceType(str, Enum):
    """Types of measurable disturbances"""
    HEAT_DEMAND_CHANGE = "heat_demand_change"
    LOAD_STEP = "load_step"
    FUEL_QUALITY_CHANGE = "fuel_quality_change"
    AMBIENT_TEMPERATURE = "ambient_temperature"
    FEED_WATER_TEMPERATURE = "feed_water_temperature"


class CompensationType(str, Enum):
    """Feedforward compensation methods"""
    STATIC = "static"  # Static gain only
    DYNAMIC = "dynamic"  # Lead-lag compensation
    RATE_BASED = "rate_based"  # Include derivative term


@dataclass
class FeedforwardGains:
    """Feedforward controller gains"""
    static_gain: float  # K_ff (steady-state gain)
    lead_time_constant: float  # τ_lead (seconds)
    lag_time_constant: float  # τ_lag (seconds)
    rate_gain: float  # Gain on rate of change

    def validate(self) -> bool:
        """Validate gains are reasonable"""
        return (
            self.static_gain >= 0 and
            self.lead_time_constant >= 0 and
            self.lag_time_constant >= 0 and
            self.rate_gain >= 0
        )


@dataclass
class ProcessModel:
    """Simple process model for feedforward compensation"""
    process_gain: float  # Steady-state gain (output/input)
    time_constant: float  # Process time constant (seconds)
    dead_time: float  # Process dead time (seconds)

    def transfer_function_gain(self, frequency: float) -> float:
        """Calculate frequency response gain"""
        omega = 2 * math.pi * frequency
        magnitude = self.process_gain / math.sqrt(1 + (omega * self.time_constant) ** 2)
        return magnitude


class FeedforwardInput(BaseModel):
    """Input parameters for feedforward controller"""

    # Disturbance measurements
    heat_demand_kw: float = Field(
        ...,
        ge=0,
        description="Current heat demand in kW"
    )
    heat_demand_setpoint_kw: float = Field(
        ...,
        ge=0,
        description="Target heat demand setpoint in kW"
    )
    heat_demand_rate_kw_per_sec: Optional[float] = Field(
        default=0,
        description="Rate of change of heat demand"
    )

    # Fuel properties
    fuel_heating_value_mj_per_kg: float = Field(
        ...,
        gt=0,
        le=100,
        description="Lower heating value of fuel"
    )
    expected_combustion_efficiency: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Expected combustion efficiency (0-1)"
    )

    # Process conditions
    current_fuel_flow_kg_per_hr: float = Field(
        ...,
        ge=0,
        description="Current fuel flow rate"
    )
    ambient_temperature_c: float = Field(
        default=25.0,
        ge=-50,
        le=60
    )
    feed_water_temperature_c: Optional[float] = Field(
        None,
        ge=0,
        le=200,
        description="Feed water temperature (for boilers)"
    )

    # Controller parameters
    compensation_type: CompensationType = Field(
        default=CompensationType.DYNAMIC
    )
    enable_rate_compensation: bool = Field(
        default=True,
        description="Enable derivative (rate) compensation"
    )

    # Gain scheduling parameters
    enable_gain_scheduling: bool = Field(
        default=False,
        description="Enable gain scheduling based on operating point"
    )
    operating_load_percent: float = Field(
        default=100.0,
        ge=0,
        le=150,
        description="Current operating load as percentage of rated"
    )

    # Timing
    timestamp: float = Field(
        ...,
        ge=0,
        description="Current timestamp in seconds"
    )
    sample_time_sec: float = Field(
        default=1.0,
        gt=0,
        le=60,
        description="Controller sample time"
    )

    # Limits
    fuel_flow_min_kg_per_hr: float = Field(
        default=0,
        ge=0
    )
    fuel_flow_max_kg_per_hr: float = Field(
        default=10000,
        gt=0
    )

    @field_validator('fuel_flow_max_kg_per_hr')
    @classmethod
    def validate_fuel_flow_limits(cls, v, info):
        """Ensure max > min"""
        if 'fuel_flow_min_kg_per_hr' in info.data and v <= info.data['fuel_flow_min_kg_per_hr']:
            raise ValueError("fuel_flow_max must be greater than fuel_flow_min")
        return v


class FeedforwardOutput(BaseModel):
    """Feedforward controller output"""

    # Feedforward control signal
    feedforward_fuel_flow_kg_per_hr: float = Field(
        ...,
        description="Feedforward fuel flow command"
    )
    feedforward_correction: float = Field(
        ...,
        description="Feedforward correction relative to current flow"
    )

    # Component breakdown
    static_compensation: float = Field(
        ...,
        description="Static feedforward compensation"
    )
    dynamic_compensation: float = Field(
        default=0,
        description="Dynamic (lead-lag) compensation"
    )
    rate_compensation: float = Field(
        default=0,
        description="Rate-based compensation"
    )

    # Heat balance
    predicted_heat_output_kw: float = Field(
        ...,
        description="Predicted heat output with feedforward action"
    )
    heat_demand_error_kw: float = Field(
        ...,
        description="Difference between demand and predicted output"
    )

    # Timing and prediction
    anticipation_time_sec: float = Field(
        ...,
        description="How far ahead the controller is anticipating"
    )
    fuel_transport_delay_sec: float = Field(
        default=0,
        description="Estimated fuel transport delay"
    )

    # Status
    compensation_type: CompensationType
    gain_scheduled: bool = Field(
        default=False,
        description="Whether gain scheduling was applied"
    )
    limited: bool = Field(
        ...,
        description="Whether output was limited"
    )

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for calculation provenance"
    )


class FeedforwardController:
    """
    Feedforward controller for combustion systems.

    Feedforward control anticipates disturbances (heat demand changes) and
    adjusts fuel flow proactively, improving response time and reducing
    deviation from setpoint.

    Key Advantages:
        - Faster response than feedback control alone
        - Reduces peak error during load changes
        - Improves fuel efficiency during transients
        - Reduces thermal stress on equipment

    Mathematical Basis:
        Static Feedforward:
            u_ff = K_ff * (d - d_0)

        Dynamic Feedforward (Lead-Lag):
            u_ff = K_ff * (τ_lead*dd/dt + d) / (τ_lag + 1)

        Combined (with rate term):
            u_ff = K_ff * d + K_rate * dd/dt

    Where:
        - u_ff = feedforward control output
        - d = measured disturbance (heat demand)
        - K_ff = static feedforward gain
        - τ_lead, τ_lag = lead/lag time constants
    """

    def __init__(
        self,
        static_gain: float = 1.0,
        lead_time_constant: float = 10.0,
        lag_time_constant: float = 5.0,
        rate_gain: float = 0.1
    ):
        """
        Initialize feedforward controller.

        Args:
            static_gain: Static feedforward gain K_ff
            lead_time_constant: Lead time constant τ_lead (seconds)
            lag_time_constant: Lag time constant τ_lag (seconds)
            rate_gain: Gain on rate of change
        """
        self.logger = logging.getLogger(__name__)

        # Controller gains
        self.gains = FeedforwardGains(
            static_gain=static_gain,
            lead_time_constant=lead_time_constant,
            lag_time_constant=lag_time_constant,
            rate_gain=rate_gain
        )

        # Internal state
        self.previous_disturbance = 0.0
        self.filtered_disturbance = 0.0
        self.time_previous = None

        # History for analysis
        self.disturbance_history = deque(maxlen=100)
        self.output_history = deque(maxlen=100)

        self.logger.info(
            f"Feedforward controller initialized: K_ff={static_gain}, "
            f"τ_lead={lead_time_constant}s, τ_lag={lag_time_constant}s"
        )

    def calculate_feedforward_action(
        self,
        ff_input: FeedforwardInput
    ) -> FeedforwardOutput:
        """
        Calculate feedforward control action.

        Algorithm:
            1. Calculate static compensation based on heat demand change
            2. Apply dynamic compensation (lead-lag) if enabled
            3. Add rate compensation if enabled
            4. Apply gain scheduling if enabled
            5. Calculate predicted fuel flow
            6. Apply limits and return output

        Args:
            ff_input: Feedforward input parameters

        Returns:
            FeedforwardOutput with control action and diagnostics
        """
        # Step 1: Calculate heat demand change (disturbance)
        heat_demand_change = ff_input.heat_demand_setpoint_kw - ff_input.heat_demand_kw

        # Step 2: Calculate required fuel flow change using energy balance
        # Q̇ = ṁ_fuel * LHV * η
        # ṁ_fuel = Q̇ / (LHV * η)
        required_fuel_change = self._calculate_fuel_requirement(
            heat_demand_change,
            ff_input.fuel_heating_value_mj_per_kg,
            ff_input.expected_combustion_efficiency
        )

        # Step 3: Apply static compensation
        static_compensation = self.gains.static_gain * required_fuel_change

        # Step 4: Calculate time step
        if self.time_previous is None:
            dt = ff_input.sample_time_sec
            self.time_previous = ff_input.timestamp
        else:
            dt = ff_input.timestamp - self.time_previous
            if dt <= 0:
                dt = ff_input.sample_time_sec

        # Step 5: Apply dynamic compensation (lead-lag)
        dynamic_compensation = 0.0
        if ff_input.compensation_type == CompensationType.DYNAMIC:
            dynamic_compensation = self._apply_lead_lag_compensation(
                heat_demand_change,
                self.previous_disturbance,
                dt
            )

        # Step 6: Apply rate compensation
        rate_compensation = 0.0
        if ff_input.enable_rate_compensation:
            if ff_input.heat_demand_rate_kw_per_sec is not None:
                rate_kw_per_sec = ff_input.heat_demand_rate_kw_per_sec
            else:
                # Estimate rate from history
                rate_kw_per_sec = (heat_demand_change - self.previous_disturbance) / dt if dt > 0 else 0

            rate_compensation = self.gains.rate_gain * rate_kw_per_sec

        # Step 7: Apply gain scheduling
        gain_scheduled = False
        gain_schedule_factor = 1.0
        if ff_input.enable_gain_scheduling:
            gain_schedule_factor = self._calculate_gain_schedule(
                ff_input.operating_load_percent
            )
            gain_scheduled = True

        # Step 8: Calculate total feedforward action
        total_compensation = (
            (static_compensation + dynamic_compensation + rate_compensation) *
            gain_schedule_factor
        )

        # Step 9: Calculate feedforward fuel flow command
        feedforward_fuel_flow = ff_input.current_fuel_flow_kg_per_hr + total_compensation

        # Step 10: Apply limits
        feedforward_fuel_flow_limited = self._clamp(
            feedforward_fuel_flow,
            ff_input.fuel_flow_min_kg_per_hr,
            ff_input.fuel_flow_max_kg_per_hr
        )

        limited = (feedforward_fuel_flow != feedforward_fuel_flow_limited)

        # Step 11: Calculate predicted heat output
        predicted_heat_output = self._calculate_heat_output(
            feedforward_fuel_flow_limited,
            ff_input.fuel_heating_value_mj_per_kg,
            ff_input.expected_combustion_efficiency
        )

        heat_demand_error = ff_input.heat_demand_setpoint_kw - predicted_heat_output

        # Step 12: Calculate anticipation time (based on lead time constant)
        anticipation_time = self.gains.lead_time_constant

        # Step 13: Estimate fuel transport delay
        fuel_transport_delay = self._estimate_fuel_transport_delay(
            ff_input.current_fuel_flow_kg_per_hr
        )

        # Update state
        self.previous_disturbance = heat_demand_change
        self.time_previous = ff_input.timestamp

        # Update history
        self.disturbance_history.append(heat_demand_change)
        self.output_history.append(feedforward_fuel_flow_limited)

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance(
            ff_input,
            feedforward_fuel_flow_limited,
            static_compensation,
            dynamic_compensation,
            rate_compensation
        )

        return FeedforwardOutput(
            feedforward_fuel_flow_kg_per_hr=self._round_decimal(feedforward_fuel_flow_limited, 4),
            feedforward_correction=self._round_decimal(total_compensation, 4),
            static_compensation=self._round_decimal(static_compensation, 4),
            dynamic_compensation=self._round_decimal(dynamic_compensation, 4),
            rate_compensation=self._round_decimal(rate_compensation, 4),
            predicted_heat_output_kw=self._round_decimal(predicted_heat_output, 2),
            heat_demand_error_kw=self._round_decimal(heat_demand_error, 2),
            anticipation_time_sec=self._round_decimal(anticipation_time, 2),
            fuel_transport_delay_sec=self._round_decimal(fuel_transport_delay, 2),
            compensation_type=ff_input.compensation_type,
            gain_scheduled=gain_scheduled,
            limited=limited,
            provenance_hash=provenance_hash
        )

    def _calculate_fuel_requirement(
        self,
        heat_demand_kw: float,
        fuel_lhv_mj_per_kg: float,
        efficiency: float
    ) -> float:
        """
        Calculate required fuel flow for heat demand.

        Formula:
            ṁ_fuel (kg/hr) = Q̇ (kW) * 3600 (s/hr) / (LHV (MJ/kg) * 1000 * η)

        Args:
            heat_demand_kw: Heat demand in kW
            fuel_lhv_mj_per_kg: Fuel lower heating value
            efficiency: Combustion efficiency (0-1)

        Returns:
            Required fuel flow in kg/hr
        """
        if fuel_lhv_mj_per_kg <= 0 or efficiency <= 0:
            return 0.0

        # Convert kW to MJ/hr: kW * 3.6 = MJ/hr
        heat_demand_mj_per_hr = heat_demand_kw * 3.6

        # Calculate fuel requirement
        fuel_flow_kg_per_hr = heat_demand_mj_per_hr / (fuel_lhv_mj_per_kg * efficiency)

        return fuel_flow_kg_per_hr

    def _calculate_heat_output(
        self,
        fuel_flow_kg_per_hr: float,
        fuel_lhv_mj_per_kg: float,
        efficiency: float
    ) -> float:
        """
        Calculate heat output from fuel flow.

        Formula:
            Q̇ (kW) = ṁ_fuel (kg/hr) * LHV (MJ/kg) * η / 3.6

        Args:
            fuel_flow_kg_per_hr: Fuel flow rate
            fuel_lhv_mj_per_kg: Fuel lower heating value
            efficiency: Combustion efficiency (0-1)

        Returns:
            Heat output in kW
        """
        # Convert MJ/hr to kW: MJ/hr / 3.6 = kW
        heat_input_mj_per_hr = fuel_flow_kg_per_hr * fuel_lhv_mj_per_kg
        heat_output_kw = (heat_input_mj_per_hr * efficiency) / 3.6

        return heat_output_kw

    def _apply_lead_lag_compensation(
        self,
        disturbance: float,
        previous_disturbance: float,
        dt: float
    ) -> float:
        """
        Apply lead-lag compensation.

        Discrete approximation of lead-lag filter:
            G(s) = K * (τ_lead*s + 1) / (τ_lag*s + 1)

        Args:
            disturbance: Current disturbance value
            previous_disturbance: Previous disturbance value
            dt: Time step

        Returns:
            Lead-lag compensated value
        """
        tau_lead = self.gains.lead_time_constant
        tau_lag = self.gains.lag_time_constant

        if tau_lag <= 0 or dt <= 0:
            return 0.0

        # Calculate derivative term (discrete)
        derivative = (disturbance - previous_disturbance) / dt

        # Lead term
        lead_term = tau_lead * derivative + disturbance

        # Lag filter (first-order low-pass)
        alpha = dt / (tau_lag + dt)
        self.filtered_disturbance = alpha * lead_term + (1 - alpha) * self.filtered_disturbance

        return self.filtered_disturbance - disturbance

    def _calculate_gain_schedule(
        self,
        operating_load_percent: float
    ) -> float:
        """
        Calculate gain scheduling factor based on operating load.

        Gain scheduling compensates for process nonlinearities at different
        operating points.

        Args:
            operating_load_percent: Current load (% of rated)

        Returns:
            Gain schedule factor (multiplier)
        """
        # Typical gain scheduling curve for combustion systems
        # Higher gains at low loads (more sensitive)
        # Lower gains at high loads (more stable)

        if operating_load_percent < 30:
            # Low load - increase gain by 20%
            return 1.2
        elif operating_load_percent < 70:
            # Mid load - nominal gain
            return 1.0
        else:
            # High load - reduce gain by 10%
            return 0.9

    def _estimate_fuel_transport_delay(
        self,
        fuel_flow_kg_per_hr: float
    ) -> float:
        """
        Estimate fuel transport delay from valve to burner.

        Args:
            fuel_flow_kg_per_hr: Current fuel flow rate

        Returns:
            Estimated transport delay in seconds
        """
        # Simple model: delay inversely proportional to flow
        # Typical range: 1-5 seconds
        if fuel_flow_kg_per_hr <= 0:
            return 5.0

        base_delay = 2.0  # seconds at nominal flow
        nominal_flow = 1000.0  # kg/hr

        delay = base_delay * (nominal_flow / fuel_flow_kg_per_hr) ** 0.5

        return self._clamp(delay, 1.0, 5.0)

    def _calculate_provenance(
        self,
        ff_input: FeedforwardInput,
        fuel_flow: float,
        static_comp: float,
        dynamic_comp: float,
        rate_comp: float
    ) -> str:
        """Calculate SHA-256 hash for complete audit trail."""
        provenance_data = {
            'heat_demand_kw': ff_input.heat_demand_kw,
            'heat_demand_setpoint_kw': ff_input.heat_demand_setpoint_kw,
            'fuel_heating_value': ff_input.fuel_heating_value_mj_per_kg,
            'current_fuel_flow': ff_input.current_fuel_flow_kg_per_hr,
            'timestamp': ff_input.timestamp,
            'static_compensation': static_comp,
            'dynamic_compensation': dynamic_comp,
            'rate_compensation': rate_comp,
            'fuel_flow_output': fuel_flow,
            'gains': {
                'static': self.gains.static_gain,
                'lead': self.gains.lead_time_constant,
                'lag': self.gains.lag_time_constant,
                'rate': self.gains.rate_gain
            }
        }

        provenance_str = str(provenance_data)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))

    def _round_decimal(self, value: float, places: int) -> float:
        """Round to specified decimal places using ROUND_HALF_UP"""
        decimal_value = Decimal(str(value))
        quantize_string = '0.' + '0' * places if places > 0 else '1'
        rounded = decimal_value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)
        return float(rounded)
