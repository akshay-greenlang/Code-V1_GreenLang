# -*- coding: utf-8 -*-
"""
Thermal Constraints - Safety Constraints for Blowdown Control

This module implements thermal safety constraints for blowdown operations,
including ramp rate limits, minimum intervals, and thermal shock protection.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: ASME Boiler Code, NFPA 85
Agent: GL-016_Waterguard

Zero Hallucination Guarantee:
- All calculations are deterministic
- Complete provenance tracking with SHA-256 hashes
- No LLM inference in calculation path
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import math

from .provenance import ProvenanceTracker, ProvenanceRecord, create_calculation_hash


class ConstraintViolation(Enum):
    """Types of constraint violations."""
    NONE = "none"
    RAMP_RATE_EXCEEDED = "ramp_rate_exceeded"
    MINIMUM_INTERVAL_VIOLATED = "minimum_interval_violated"
    THERMAL_SHOCK_RISK = "thermal_shock_risk"
    SCALING_RISK = "scaling_risk"
    MINIMUM_BLOWDOWN_VIOLATED = "minimum_blowdown_violated"
    MAXIMUM_BLOWDOWN_EXCEEDED = "maximum_blowdown_exceeded"


class RiskLevel(Enum):
    """Risk level classification."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConstraintValidationResult:
    """Result of constraint validation."""
    is_valid: bool
    violations: List[ConstraintViolation]
    risk_level: RiskLevel
    messages: List[str]
    recommendations: List[str]
    provenance: Optional[ProvenanceRecord] = None
    calculation_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'violations': [v.value for v in self.violations],
            'risk_level': self.risk_level.value,
            'messages': self.messages,
            'recommendations': self.recommendations,
            'calculation_hash': self.calculation_hash,
            'provenance': self.provenance.to_dict() if self.provenance else None
        }


@dataclass
class ThermalShockAssessment:
    """Assessment of thermal shock risk."""
    temperature_delta_c: Decimal
    allowable_delta_c: Decimal
    risk_level: RiskLevel
    time_to_safe_change_s: Optional[Decimal] = None
    recommended_ramp_rate_c_min: Optional[Decimal] = None
    provenance: Optional[ProvenanceRecord] = None
    calculation_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'temperature_delta_c': float(self.temperature_delta_c),
            'allowable_delta_c': float(self.allowable_delta_c),
            'risk_level': self.risk_level.value,
            'time_to_safe_change_s': float(self.time_to_safe_change_s) if self.time_to_safe_change_s else None,
            'recommended_ramp_rate_c_min': float(self.recommended_ramp_rate_c_min) if self.recommended_ramp_rate_c_min else None,
            'calculation_hash': self.calculation_hash,
            'provenance': self.provenance.to_dict() if self.provenance else None
        }


@dataclass
class ScalingRiskAssessment:
    """Assessment of scaling/fouling risk from blowdown changes."""
    current_tds_ppm: Decimal
    projected_tds_ppm: Decimal
    max_allowable_tds_ppm: Decimal
    cycles_of_concentration: Decimal
    scaling_tendency: RiskLevel
    time_to_limit_hours: Optional[Decimal] = None
    heat_transfer_degradation_percent: Optional[Decimal] = None
    recommendations: List[str] = field(default_factory=list)
    provenance: Optional[ProvenanceRecord] = None
    calculation_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'current_tds_ppm': float(self.current_tds_ppm),
            'projected_tds_ppm': float(self.projected_tds_ppm),
            'max_allowable_tds_ppm': float(self.max_allowable_tds_ppm),
            'cycles_of_concentration': float(self.cycles_of_concentration),
            'scaling_tendency': self.scaling_tendency.value,
            'time_to_limit_hours': float(self.time_to_limit_hours) if self.time_to_limit_hours else None,
            'heat_transfer_degradation_percent': float(self.heat_transfer_degradation_percent) if self.heat_transfer_degradation_percent else None,
            'recommendations': self.recommendations,
            'calculation_hash': self.calculation_hash,
            'provenance': self.provenance.to_dict() if self.provenance else None
        }


class BlowdownRampRateLimit:
    """
    Blowdown Ramp Rate Limiter.

    Prevents rapid changes in blowdown rate that could cause:
    - Thermal shock to boiler components
    - Water level instability
    - Pressure fluctuations
    - Control system oscillation

    Zero Hallucination Guarantee:
    - All limits are configurable and deterministic
    - Complete provenance tracking
    """

    # Default ramp rate limits based on boiler pressure class
    DEFAULT_LIMITS = {
        'low_pressure': {  # < 15 bar
            'max_increase_percent_per_min': Decimal('5.0'),
            'max_decrease_percent_per_min': Decimal('10.0'),
            'description': 'Low pressure boiler (< 15 bar)'
        },
        'medium_pressure': {  # 15-40 bar
            'max_increase_percent_per_min': Decimal('3.0'),
            'max_decrease_percent_per_min': Decimal('5.0'),
            'description': 'Medium pressure boiler (15-40 bar)'
        },
        'high_pressure': {  # 40-100 bar
            'max_increase_percent_per_min': Decimal('2.0'),
            'max_decrease_percent_per_min': Decimal('3.0'),
            'description': 'High pressure boiler (40-100 bar)'
        },
        'very_high_pressure': {  # > 100 bar
            'max_increase_percent_per_min': Decimal('1.0'),
            'max_decrease_percent_per_min': Decimal('2.0'),
            'description': 'Very high pressure boiler (> 100 bar)'
        }
    }

    def __init__(
        self,
        max_increase_percent_per_min: float = 5.0,
        max_decrease_percent_per_min: float = 10.0,
        version: str = "1.0.0"
    ):
        """
        Initialize ramp rate limiter.

        Args:
            max_increase_percent_per_min: Maximum increase rate (%/min)
            max_decrease_percent_per_min: Maximum decrease rate (%/min)
            version: Calculator version
        """
        self.max_increase = Decimal(str(max_increase_percent_per_min))
        self.max_decrease = Decimal(str(max_decrease_percent_per_min))
        self.version = version

    @classmethod
    def from_pressure_class(cls, pressure_bar: float, version: str = "1.0.0"):
        """
        Create limiter from boiler pressure class.

        Args:
            pressure_bar: Boiler operating pressure (bar)
            version: Calculator version

        Returns:
            BlowdownRampRateLimit configured for pressure class
        """
        pressure = Decimal(str(pressure_bar))

        if pressure < Decimal('15'):
            limits = cls.DEFAULT_LIMITS['low_pressure']
        elif pressure < Decimal('40'):
            limits = cls.DEFAULT_LIMITS['medium_pressure']
        elif pressure < Decimal('100'):
            limits = cls.DEFAULT_LIMITS['high_pressure']
        else:
            limits = cls.DEFAULT_LIMITS['very_high_pressure']

        return cls(
            max_increase_percent_per_min=float(limits['max_increase_percent_per_min']),
            max_decrease_percent_per_min=float(limits['max_decrease_percent_per_min']),
            version=version
        )

    def validate_change(
        self,
        current_blowdown_percent: float,
        proposed_blowdown_percent: float,
        time_interval_seconds: float
    ) -> ConstraintValidationResult:
        """
        Validate a proposed blowdown change against ramp rate limits.

        Args:
            current_blowdown_percent: Current blowdown rate (%)
            proposed_blowdown_percent: Proposed blowdown rate (%)
            time_interval_seconds: Time for the change (seconds)

        Returns:
            ConstraintValidationResult with validation outcome
        """
        tracker = ProvenanceTracker(
            calculation_id="ramp_rate_validation",
            calculation_type="ramp_rate_limit",
            version=self.version
        )

        tracker.record_inputs({
            'current_blowdown_percent': current_blowdown_percent,
            'proposed_blowdown_percent': proposed_blowdown_percent,
            'time_interval_seconds': time_interval_seconds,
            'max_increase_percent_per_min': float(self.max_increase),
            'max_decrease_percent_per_min': float(self.max_decrease)
        })

        current = Decimal(str(current_blowdown_percent))
        proposed = Decimal(str(proposed_blowdown_percent))
        time_s = Decimal(str(time_interval_seconds))
        time_min = time_s / Decimal('60')

        # Calculate change rate
        change = proposed - current
        change_rate = abs(change) / time_min if time_min > 0 else Decimal('999999')

        tracker.record_step(
            operation="calculate_change_rate",
            description="Calculate blowdown change rate",
            inputs={
                'change_percent': change,
                'time_minutes': time_min
            },
            output_value=change_rate,
            output_name="change_rate_percent_per_min",
            formula="Rate = |Change| / Time",
            units="%/min"
        )

        # Check limits
        violations = []
        messages = []
        recommendations = []

        if change > 0:
            # Increase
            if change_rate > self.max_increase:
                violations.append(ConstraintViolation.RAMP_RATE_EXCEEDED)
                messages.append(
                    f"Increase rate {float(change_rate):.2f}%/min exceeds limit {float(self.max_increase):.2f}%/min"
                )

                # Calculate minimum time for safe change
                min_time = abs(change) / self.max_increase
                recommendations.append(
                    f"Extend change time to at least {float(min_time):.1f} minutes"
                )
        else:
            # Decrease
            if change_rate > self.max_decrease:
                violations.append(ConstraintViolation.RAMP_RATE_EXCEEDED)
                messages.append(
                    f"Decrease rate {float(change_rate):.2f}%/min exceeds limit {float(self.max_decrease):.2f}%/min"
                )

                min_time = abs(change) / self.max_decrease
                recommendations.append(
                    f"Extend change time to at least {float(min_time):.1f} minutes"
                )

        # Determine risk level
        if not violations:
            risk_level = RiskLevel.NONE
        else:
            if change_rate > self.max_increase * 2 or change_rate > self.max_decrease * 2:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.MEDIUM

        tracker.record_step(
            operation="risk_assessment",
            description="Assess ramp rate violation risk",
            inputs={
                'change_rate': change_rate,
                'violations_count': len(violations)
            },
            output_value=risk_level.value,
            output_name="risk_level",
            formula="Based on violation severity",
            units="category"
        )

        provenance = tracker.get_provenance_record(change_rate)

        return ConstraintValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            risk_level=risk_level,
            messages=messages,
            recommendations=recommendations,
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )

    def calculate_safe_setpoint(
        self,
        current_blowdown_percent: float,
        target_blowdown_percent: float,
        elapsed_time_seconds: float
    ) -> Decimal:
        """
        Calculate the safe setpoint considering ramp rate limits.

        Args:
            current_blowdown_percent: Current blowdown rate (%)
            target_blowdown_percent: Target blowdown rate (%)
            elapsed_time_seconds: Time elapsed since start (seconds)

        Returns:
            Safe setpoint that respects ramp rate limits
        """
        current = Decimal(str(current_blowdown_percent))
        target = Decimal(str(target_blowdown_percent))
        elapsed_min = Decimal(str(elapsed_time_seconds)) / Decimal('60')

        change_direction = Decimal('1') if target > current else Decimal('-1')

        if target > current:
            max_rate = self.max_increase
        else:
            max_rate = self.max_decrease

        # Calculate maximum allowed change
        max_change = max_rate * elapsed_min

        # Calculate actual change needed
        actual_change = abs(target - current)

        # Apply limit
        if actual_change <= max_change:
            safe_setpoint = target
        else:
            safe_setpoint = current + (change_direction * max_change)

        return safe_setpoint.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


class MinimumBlowdownInterval:
    """
    Minimum Interval Between Intermittent Blowdowns.

    Enforces minimum time between intermittent blowdown operations
    to prevent:
    - Excessive thermal cycling
    - Control valve wear
    - Water level hunting
    - Steam drum stress

    Zero Hallucination Guarantee:
    - All calculations deterministic
    - Complete provenance tracking
    """

    # Default intervals based on boiler size
    DEFAULT_INTERVALS = {
        'small': {  # < 5 t/h
            'min_interval_seconds': 300,  # 5 minutes
            'description': 'Small boiler (< 5 t/h)'
        },
        'medium': {  # 5-25 t/h
            'min_interval_seconds': 600,  # 10 minutes
            'description': 'Medium boiler (5-25 t/h)'
        },
        'large': {  # 25-100 t/h
            'min_interval_seconds': 900,  # 15 minutes
            'description': 'Large boiler (25-100 t/h)'
        },
        'very_large': {  # > 100 t/h
            'min_interval_seconds': 1800,  # 30 minutes
            'description': 'Very large boiler (> 100 t/h)'
        }
    }

    def __init__(
        self,
        min_interval_seconds: int = 600,
        max_blowdowns_per_hour: int = 6,
        version: str = "1.0.0"
    ):
        """
        Initialize minimum interval constraint.

        Args:
            min_interval_seconds: Minimum interval between blowdowns (seconds)
            max_blowdowns_per_hour: Maximum blowdown events per hour
            version: Calculator version
        """
        self.min_interval = Decimal(str(min_interval_seconds))
        self.max_per_hour = max_blowdowns_per_hour
        self.version = version

    @classmethod
    def from_boiler_capacity(cls, capacity_t_h: float, version: str = "1.0.0"):
        """
        Create constraint from boiler capacity.

        Args:
            capacity_t_h: Boiler steam capacity (t/h)
            version: Calculator version

        Returns:
            MinimumBlowdownInterval configured for boiler size
        """
        capacity = Decimal(str(capacity_t_h))

        if capacity < Decimal('5'):
            config = cls.DEFAULT_INTERVALS['small']
        elif capacity < Decimal('25'):
            config = cls.DEFAULT_INTERVALS['medium']
        elif capacity < Decimal('100'):
            config = cls.DEFAULT_INTERVALS['large']
        else:
            config = cls.DEFAULT_INTERVALS['very_large']

        return cls(
            min_interval_seconds=config['min_interval_seconds'],
            max_blowdowns_per_hour=int(3600 / config['min_interval_seconds']),
            version=version
        )

    def validate_blowdown_request(
        self,
        last_blowdown_timestamp: datetime,
        current_timestamp: datetime,
        blowdowns_last_hour: int = 0
    ) -> ConstraintValidationResult:
        """
        Validate if a blowdown can be performed now.

        Args:
            last_blowdown_timestamp: Timestamp of last blowdown
            current_timestamp: Current timestamp
            blowdowns_last_hour: Number of blowdowns in last hour

        Returns:
            ConstraintValidationResult with validation outcome
        """
        tracker = ProvenanceTracker(
            calculation_id="interval_validation",
            calculation_type="minimum_interval",
            version=self.version
        )

        # Calculate elapsed time
        elapsed = (current_timestamp - last_blowdown_timestamp).total_seconds()
        elapsed_decimal = Decimal(str(elapsed))

        tracker.record_inputs({
            'last_blowdown': last_blowdown_timestamp.isoformat(),
            'current_time': current_timestamp.isoformat(),
            'elapsed_seconds': float(elapsed_decimal),
            'min_interval_seconds': float(self.min_interval),
            'blowdowns_last_hour': blowdowns_last_hour,
            'max_per_hour': self.max_per_hour
        })

        violations = []
        messages = []
        recommendations = []

        # Check minimum interval
        if elapsed_decimal < self.min_interval:
            violations.append(ConstraintViolation.MINIMUM_INTERVAL_VIOLATED)
            wait_time = self.min_interval - elapsed_decimal
            messages.append(
                f"Minimum interval not met. Wait {float(wait_time):.0f} more seconds."
            )
            recommendations.append(
                f"Next blowdown allowed at: "
                f"{(last_blowdown_timestamp + timedelta(seconds=float(self.min_interval))).isoformat()}"
            )

        # Check hourly limit
        if blowdowns_last_hour >= self.max_per_hour:
            violations.append(ConstraintViolation.MINIMUM_INTERVAL_VIOLATED)
            messages.append(
                f"Hourly limit reached ({self.max_per_hour} blowdowns/hour)"
            )
            recommendations.append("Wait for oldest blowdown to fall out of 1-hour window")

        tracker.record_step(
            operation="interval_check",
            description="Check minimum interval constraint",
            inputs={
                'elapsed_seconds': elapsed_decimal,
                'min_interval': self.min_interval
            },
            output_value=elapsed_decimal >= self.min_interval,
            output_name="interval_met",
            formula="elapsed >= min_interval",
            units="boolean"
        )

        # Determine risk level
        if not violations:
            risk_level = RiskLevel.NONE
        else:
            risk_level = RiskLevel.MEDIUM

        provenance = tracker.get_provenance_record(elapsed_decimal)

        return ConstraintValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            risk_level=risk_level,
            messages=messages,
            recommendations=recommendations,
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )

    def get_next_allowed_time(
        self,
        last_blowdown_timestamp: datetime
    ) -> datetime:
        """
        Get the next allowed blowdown time.

        Args:
            last_blowdown_timestamp: Timestamp of last blowdown

        Returns:
            Earliest allowed time for next blowdown
        """
        return last_blowdown_timestamp + timedelta(seconds=float(self.min_interval))


class ThermalShockProtection:
    """
    Thermal Shock Protection for Boiler Components.

    Assesses and prevents thermal shock from rapid temperature changes
    that could damage:
    - Boiler tubes
    - Steam drum
    - Headers
    - Refractory

    Based on ASME thermal stress guidelines.

    Zero Hallucination Guarantee:
    - All calculations deterministic
    - Based on material properties and ASME guidelines
    - Complete provenance tracking
    """

    # Material-specific thermal limits (delta T in Celsius)
    MATERIAL_LIMITS = {
        'carbon_steel': {
            'max_delta_t_c': Decimal('30'),  # Max temperature change
            'max_rate_c_min': Decimal('3'),   # Max rate of change
            'thermal_conductivity': Decimal('50'),  # W/(m*K)
        },
        'low_alloy_steel': {
            'max_delta_t_c': Decimal('25'),
            'max_rate_c_min': Decimal('2.5'),
            'thermal_conductivity': Decimal('42'),
        },
        'stainless_steel': {
            'max_delta_t_c': Decimal('40'),
            'max_rate_c_min': Decimal('4'),
            'thermal_conductivity': Decimal('16'),
        },
        'cast_iron': {
            'max_delta_t_c': Decimal('15'),
            'max_rate_c_min': Decimal('1.5'),
            'thermal_conductivity': Decimal('52'),
        }
    }

    def __init__(
        self,
        material: str = 'carbon_steel',
        wall_thickness_mm: float = 10.0,
        version: str = "1.0.0"
    ):
        """
        Initialize thermal shock protection.

        Args:
            material: Boiler material type
            wall_thickness_mm: Wall thickness (mm)
            version: Calculator version
        """
        if material not in self.MATERIAL_LIMITS:
            raise ValueError(f"Unknown material: {material}")

        self.material = material
        self.limits = self.MATERIAL_LIMITS[material]
        self.wall_thickness = Decimal(str(wall_thickness_mm))
        self.version = version

    def assess_thermal_shock_risk(
        self,
        current_temperature_c: float,
        target_temperature_c: float,
        proposed_time_seconds: float
    ) -> ThermalShockAssessment:
        """
        Assess thermal shock risk from a temperature change.

        Args:
            current_temperature_c: Current temperature (C)
            target_temperature_c: Target temperature (C)
            proposed_time_seconds: Proposed time for change (seconds)

        Returns:
            ThermalShockAssessment with risk evaluation
        """
        tracker = ProvenanceTracker(
            calculation_id="thermal_shock_assessment",
            calculation_type="thermal_shock",
            version=self.version
        )

        tracker.record_inputs({
            'current_temperature_c': current_temperature_c,
            'target_temperature_c': target_temperature_c,
            'proposed_time_seconds': proposed_time_seconds,
            'material': self.material,
            'wall_thickness_mm': float(self.wall_thickness)
        })

        current = Decimal(str(current_temperature_c))
        target = Decimal(str(target_temperature_c))
        time_s = Decimal(str(proposed_time_seconds))
        time_min = time_s / Decimal('60')

        # Calculate temperature change
        delta_t = abs(target - current)

        tracker.record_step(
            operation="delta_t_calculation",
            description="Calculate temperature change magnitude",
            inputs={'current': current, 'target': target},
            output_value=delta_t,
            output_name="delta_t_c",
            formula="delta_T = |T_target - T_current|",
            units="C"
        )

        # Calculate rate of change
        rate = delta_t / time_min if time_min > 0 else Decimal('999999')

        tracker.record_step(
            operation="rate_calculation",
            description="Calculate temperature change rate",
            inputs={'delta_t': delta_t, 'time_min': time_min},
            output_value=rate,
            output_name="rate_c_min",
            formula="Rate = delta_T / time",
            units="C/min"
        )

        # Get material limits
        max_delta = self.limits['max_delta_t_c']
        max_rate = self.limits['max_rate_c_min']

        # Adjust limits for wall thickness (thicker = more conservative)
        thickness_factor = self.wall_thickness / Decimal('10')  # Normalize to 10mm
        adjusted_max_delta = max_delta / thickness_factor
        adjusted_max_rate = max_rate / thickness_factor

        tracker.record_step(
            operation="limit_adjustment",
            description="Adjust limits for wall thickness",
            inputs={
                'base_max_delta': max_delta,
                'base_max_rate': max_rate,
                'thickness_factor': thickness_factor
            },
            output_value=adjusted_max_delta,
            output_name="adjusted_max_delta_c",
            formula="Adjusted = Base / (thickness / 10mm)",
            units="C"
        )

        # Assess risk level
        if delta_t <= adjusted_max_delta and rate <= adjusted_max_rate:
            risk_level = RiskLevel.NONE
        elif delta_t <= adjusted_max_delta * Decimal('1.5') and rate <= adjusted_max_rate * Decimal('1.5'):
            risk_level = RiskLevel.LOW
        elif delta_t <= adjusted_max_delta * Decimal('2') and rate <= adjusted_max_rate * Decimal('2'):
            risk_level = RiskLevel.MEDIUM
        elif delta_t <= adjusted_max_delta * Decimal('3') and rate <= adjusted_max_rate * Decimal('3'):
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL

        tracker.record_step(
            operation="risk_assessment",
            description="Assess thermal shock risk level",
            inputs={
                'delta_t': delta_t,
                'rate': rate,
                'max_delta': adjusted_max_delta,
                'max_rate': adjusted_max_rate
            },
            output_value=risk_level.value,
            output_name="risk_level",
            formula="Based on comparison with adjusted limits",
            units="category"
        )

        # Calculate safe change time if needed
        time_to_safe = None
        recommended_rate = None
        if rate > adjusted_max_rate:
            time_to_safe = delta_t / adjusted_max_rate * Decimal('60')  # seconds
            recommended_rate = adjusted_max_rate

        provenance = tracker.get_provenance_record(delta_t)

        return ThermalShockAssessment(
            temperature_delta_c=delta_t.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            allowable_delta_c=adjusted_max_delta.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            risk_level=risk_level,
            time_to_safe_change_s=time_to_safe.quantize(Decimal('1'), rounding=ROUND_HALF_UP) if time_to_safe else None,
            recommended_ramp_rate_c_min=recommended_rate.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP) if recommended_rate else None,
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )


class ScalingRiskAssessment:
    """
    Scaling Risk Assessment for Blowdown Changes.

    Evaluates the risk of scale formation when reducing blowdown,
    linking to heat transfer degradation.

    Based on:
    - Langelier Saturation Index
    - TDS accumulation dynamics
    - ABMA water quality guidelines

    Zero Hallucination Guarantee:
    - All calculations deterministic
    - Based on industry standards
    - Complete provenance tracking
    """

    # TDS limits by pressure class (ppm) - ABMA guidelines
    TDS_LIMITS = {
        'low_pressure': Decimal('3500'),      # < 15 bar
        'medium_pressure': Decimal('3000'),   # 15-40 bar
        'high_pressure': Decimal('2000'),     # 40-60 bar
        'very_high_pressure': Decimal('1000') # > 60 bar
    }

    # Heat transfer degradation per 100 ppm TDS above limit
    DEGRADATION_FACTOR = Decimal('0.5')  # 0.5% per 100 ppm

    def __init__(
        self,
        boiler_pressure_bar: float,
        water_volume_m3: float,
        steam_flow_kg_h: float,
        version: str = "1.0.0"
    ):
        """
        Initialize scaling risk assessment.

        Args:
            boiler_pressure_bar: Boiler operating pressure (bar)
            water_volume_m3: Boiler water volume (m3)
            steam_flow_kg_h: Steam production rate (kg/h)
            version: Calculator version
        """
        self.pressure = Decimal(str(boiler_pressure_bar))
        self.water_volume = Decimal(str(water_volume_m3))
        self.steam_flow = Decimal(str(steam_flow_kg_h))
        self.version = version

        # Set TDS limit based on pressure class
        if self.pressure < Decimal('15'):
            self.max_tds = self.TDS_LIMITS['low_pressure']
        elif self.pressure < Decimal('40'):
            self.max_tds = self.TDS_LIMITS['medium_pressure']
        elif self.pressure < Decimal('60'):
            self.max_tds = self.TDS_LIMITS['high_pressure']
        else:
            self.max_tds = self.TDS_LIMITS['very_high_pressure']

    def assess_scaling_risk(
        self,
        current_tds_ppm: float,
        feedwater_tds_ppm: float,
        current_blowdown_percent: float,
        proposed_blowdown_percent: float,
        time_horizon_hours: float = 24.0
    ) -> 'ScalingRiskAssessmentResult':
        """
        Assess scaling risk from a blowdown reduction.

        Args:
            current_tds_ppm: Current boiler water TDS (ppm)
            feedwater_tds_ppm: Feedwater TDS (ppm)
            current_blowdown_percent: Current blowdown rate (%)
            proposed_blowdown_percent: Proposed blowdown rate (%)
            time_horizon_hours: Time horizon for projection (hours)

        Returns:
            ScalingRiskAssessment with risk evaluation
        """
        tracker = ProvenanceTracker(
            calculation_id="scaling_risk_assessment",
            calculation_type="scaling_risk",
            version=self.version
        )

        tracker.record_inputs({
            'current_tds_ppm': current_tds_ppm,
            'feedwater_tds_ppm': feedwater_tds_ppm,
            'current_blowdown_percent': current_blowdown_percent,
            'proposed_blowdown_percent': proposed_blowdown_percent,
            'time_horizon_hours': time_horizon_hours,
            'max_tds_ppm': float(self.max_tds)
        })

        tds_current = Decimal(str(current_tds_ppm))
        tds_feedwater = Decimal(str(feedwater_tds_ppm))
        bd_current = Decimal(str(current_blowdown_percent))
        bd_proposed = Decimal(str(proposed_blowdown_percent))
        time_h = Decimal(str(time_horizon_hours))

        # Calculate cycles of concentration
        # Cycles = 1 / (BD% / 100) approximately
        if bd_proposed > 0:
            cycles_proposed = Decimal('100') / bd_proposed
        else:
            cycles_proposed = Decimal('100')  # Maximum cycles if no blowdown

        tracker.record_step(
            operation="cycles_calculation",
            description="Calculate cycles of concentration at proposed blowdown",
            inputs={'proposed_blowdown_percent': bd_proposed},
            output_value=cycles_proposed,
            output_name="cycles_of_concentration",
            formula="Cycles = 100 / BD%",
            units="dimensionless"
        )

        # Calculate steady-state TDS at proposed blowdown
        # TDS_ss = TDS_feedwater * Cycles
        tds_steady_state = tds_feedwater * cycles_proposed

        tracker.record_step(
            operation="steady_state_tds",
            description="Calculate steady-state TDS at proposed blowdown",
            inputs={
                'feedwater_tds': tds_feedwater,
                'cycles': cycles_proposed
            },
            output_value=tds_steady_state,
            output_name="tds_steady_state_ppm",
            formula="TDS_ss = TDS_fw * Cycles",
            units="ppm"
        )

        # Calculate time constant for TDS change
        # tau = Water_Volume / (Blowdown_rate + Evaporation_rate)
        bd_flow = self.steam_flow * bd_proposed / Decimal('100')
        tau = self.water_volume * Decimal('1000') / bd_flow if bd_flow > 0 else Decimal('999999')

        tracker.record_step(
            operation="time_constant",
            description="Calculate TDS response time constant",
            inputs={
                'water_volume_m3': self.water_volume,
                'blowdown_flow_kg_h': bd_flow
            },
            output_value=tau,
            output_name="time_constant_hours",
            formula="tau = V / Q_bd",
            units="hours"
        )

        # Project TDS after time horizon using exponential approach
        # TDS(t) = TDS_ss + (TDS_current - TDS_ss) * exp(-t/tau)
        if tau > 0 and tau < Decimal('999999'):
            exp_factor = Decimal(str(math.exp(-float(time_h / tau))))
            tds_projected = tds_steady_state + (tds_current - tds_steady_state) * exp_factor
        else:
            tds_projected = tds_current

        tracker.record_step(
            operation="tds_projection",
            description="Project TDS at end of time horizon",
            inputs={
                'tds_current': tds_current,
                'tds_steady_state': tds_steady_state,
                'time_hours': time_h,
                'time_constant': tau
            },
            output_value=tds_projected,
            output_name="tds_projected_ppm",
            formula="TDS(t) = TDS_ss + (TDS_0 - TDS_ss) * exp(-t/tau)",
            units="ppm"
        )

        # Calculate time to reach TDS limit
        time_to_limit = None
        if tds_steady_state > self.max_tds and tds_current < self.max_tds:
            # Solve for t when TDS(t) = TDS_max
            # TDS_max = TDS_ss + (TDS_0 - TDS_ss) * exp(-t/tau)
            # exp(-t/tau) = (TDS_max - TDS_ss) / (TDS_0 - TDS_ss)
            numerator = self.max_tds - tds_steady_state
            denominator = tds_current - tds_steady_state
            if denominator != 0:
                ratio = numerator / denominator
                if ratio > 0:
                    time_to_limit = -tau * Decimal(str(math.log(float(ratio))))

        tracker.record_step(
            operation="time_to_limit",
            description="Calculate time to reach TDS limit",
            inputs={
                'tds_current': tds_current,
                'tds_steady_state': tds_steady_state,
                'max_tds': self.max_tds
            },
            output_value=time_to_limit if time_to_limit else Decimal('-1'),
            output_name="time_to_limit_hours",
            formula="Solve exponential equation for t",
            units="hours"
        )

        # Calculate heat transfer degradation
        degradation = Decimal('0')
        if tds_projected > self.max_tds:
            excess_tds = tds_projected - self.max_tds
            degradation = (excess_tds / Decimal('100')) * self.DEGRADATION_FACTOR

        tracker.record_step(
            operation="degradation_calculation",
            description="Calculate heat transfer degradation from scaling",
            inputs={
                'tds_projected': tds_projected,
                'max_tds': self.max_tds,
                'degradation_factor': self.DEGRADATION_FACTOR
            },
            output_value=degradation,
            output_name="degradation_percent",
            formula="Degradation = (TDS - TDS_max) / 100 * Factor",
            units="%"
        )

        # Assess risk level
        if tds_projected <= self.max_tds * Decimal('0.8'):
            risk_level = RiskLevel.NONE
        elif tds_projected <= self.max_tds:
            risk_level = RiskLevel.LOW
        elif tds_projected <= self.max_tds * Decimal('1.2'):
            risk_level = RiskLevel.MEDIUM
        elif tds_projected <= self.max_tds * Decimal('1.5'):
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL

        # Generate recommendations
        recommendations = []
        if risk_level != RiskLevel.NONE:
            if time_to_limit and time_to_limit < time_h:
                recommendations.append(
                    f"TDS limit will be reached in {float(time_to_limit):.1f} hours"
                )
            if degradation > 0:
                recommendations.append(
                    f"Expected {float(degradation):.1f}% heat transfer degradation"
                )
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                recommendations.append(
                    "Consider increasing blowdown rate or improving water treatment"
                )

        provenance = tracker.get_provenance_record(tds_projected)

        return ScalingRiskAssessmentResult(
            current_tds_ppm=tds_current.quantize(Decimal('1'), rounding=ROUND_HALF_UP),
            projected_tds_ppm=tds_projected.quantize(Decimal('1'), rounding=ROUND_HALF_UP),
            max_allowable_tds_ppm=self.max_tds,
            cycles_of_concentration=cycles_proposed.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            scaling_tendency=risk_level,
            time_to_limit_hours=time_to_limit.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP) if time_to_limit else None,
            heat_transfer_degradation_percent=degradation.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            recommendations=recommendations,
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )


# Use the dataclass from above but rename for clarity
ScalingRiskAssessmentResult = ScalingRiskAssessment


def validate_blowdown_change(
    current_blowdown_percent: float,
    proposed_blowdown_percent: float,
    max_ramp_rate_percent_per_min: float = 5.0,
    time_interval_seconds: float = 60.0
) -> ConstraintValidationResult:
    """
    Validate a blowdown change - convenience function.

    Args:
        current_blowdown_percent: Current blowdown rate (%)
        proposed_blowdown_percent: Proposed blowdown rate (%)
        max_ramp_rate_percent_per_min: Maximum allowed ramp rate (%/min)
        time_interval_seconds: Time for the change (seconds)

    Returns:
        ConstraintValidationResult with validation outcome
    """
    limiter = BlowdownRampRateLimit(
        max_increase_percent_per_min=max_ramp_rate_percent_per_min,
        max_decrease_percent_per_min=max_ramp_rate_percent_per_min * 2
    )
    return limiter.validate_change(
        current_blowdown_percent,
        proposed_blowdown_percent,
        time_interval_seconds
    )
