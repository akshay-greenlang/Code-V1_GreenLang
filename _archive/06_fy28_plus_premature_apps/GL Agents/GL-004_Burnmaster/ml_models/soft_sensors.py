# -*- coding: utf-8 -*-
"""
SoftSensorModels - Virtual Sensor Models for Combustion Systems

This module implements soft sensors that infer unmeasured process variables
from available measurements using physics-informed models.

Key Features:
    - Fuel quality inference from combustion products
    - Excess air estimation from stack O2
    - Heat duty inference from temperature and flow measurements
    - Sensor bias detection for validation

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConfidenceInterval(BaseModel):
    """Confidence interval for estimates."""
    lower: float
    upper: float
    confidence_level: float = Field(default=0.90)


class ProvenanceRecord(BaseModel):
    """Provenance tracking for audit trails."""
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    calculation_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_id: str = Field(default="")
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")
    computation_time_ms: float = Field(default=0.0)

    @classmethod
    def create(cls, calculation_type: str, inputs: Dict, outputs: Dict,
               model_id: str = "", computation_time_ms: float = 0.0) -> "ProvenanceRecord":
        return cls(
            calculation_type=calculation_type, model_id=model_id,
            input_hash=hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            output_hash=hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(),
            computation_time_ms=computation_time_ms
        )


class FuelQualityEstimate(BaseModel):
    """Inferred fuel quality from combustion measurements."""
    estimate_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    heating_value_mj_kg: float = Field(..., ge=0.0, description="Lower heating value in MJ/kg")
    heating_value_confidence: ConfidenceInterval
    methane_fraction: float = Field(default=0.95, ge=0.0, le=1.0, description="Estimated CH4 fraction")
    hydrogen_index: float = Field(default=4.0, ge=0.0, le=6.0, description="H/C atomic ratio")
    quality_grade: str = Field(..., description="Quality grade (A, B, C, D)")
    anomaly_detected: bool = Field(default=False, description="Unusual fuel quality detected")
    provenance: ProvenanceRecord


class ExcessAirEstimate(BaseModel):
    """Inferred excess air from stack O2 measurement."""
    estimate_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    excess_air_percent: float = Field(..., ge=0.0, description="Estimated excess air percentage")
    excess_air_confidence: ConfidenceInterval
    stoichiometric_air_ratio: float = Field(..., ge=0.0, description="Actual/stoichiometric air ratio")
    lambda_value: float = Field(..., ge=0.0, description="Lambda (air-fuel equivalence ratio)")
    within_optimal_range: bool = Field(default=True, description="Within optimal range for fuel type")
    provenance: ProvenanceRecord


class HeatDutyEstimate(BaseModel):
    """Inferred heat duty from temperature and flow measurements."""
    estimate_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    heat_duty_mw: float = Field(..., ge=0.0, description="Heat duty in MW")
    heat_duty_confidence: ConfidenceInterval
    heat_input_mw: float = Field(..., ge=0.0, description="Heat input from fuel")
    thermal_efficiency: float = Field(..., ge=0.0, le=1.0, description="Thermal efficiency")
    heat_loss_mw: float = Field(..., ge=0.0, description="Heat losses")
    provenance: ProvenanceRecord


class BiasDetection(BaseModel):
    """Sensor bias detection result."""
    detection_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sensor_name: str = Field(..., description="Name of sensor being validated")
    measured_value: float = Field(..., description="Measured sensor value")
    inferred_value: float = Field(..., description="Soft sensor inferred value")
    bias: float = Field(..., description="Detected bias (measured - inferred)")
    bias_percent: float = Field(..., description="Bias as percentage of inferred")
    is_significant: bool = Field(default=False, description="Bias exceeds threshold")
    severity: str = Field(default="LOW", description="Severity (LOW, MEDIUM, HIGH, CRITICAL)")
    recommended_action: str = Field(default="", description="Recommended corrective action")
    provenance: ProvenanceRecord


class SoftSensorModels:
    """
    Physics-informed soft sensors for combustion systems.

    Soft sensors infer unmeasured or hard-to-measure process variables
    from available measurements using physics-based models combined
    with empirical corrections.

    Key capabilities:
    1. Fuel quality inference from O2, CO, and heat balance
    2. Excess air estimation from stack O2
    3. Heat duty inference from temperatures and flows
    4. Sensor bias detection for validation

    Example:
        >>> soft_sensors = SoftSensorModels()
        >>> fuel_quality = soft_sensors.infer_fuel_quality(
        ...     o2=3.0, co=50.0, heat_balance=0.95
        ... )
        >>> print(f"LHV: {fuel_quality.heating_value_mj_kg:.1f} MJ/kg")
    """

    # Standard fuel properties for natural gas
    NATURAL_GAS_LHV = 50.0  # MJ/kg
    NATURAL_GAS_STOICH_AIR = 17.2  # kg air / kg fuel
    NATURAL_GAS_CO2_FACTOR = 2.75  # kg CO2 / kg fuel

    def __init__(self, fuel_type: str = "natural_gas"):
        """Initialize SoftSensorModels."""
        self.fuel_type = fuel_type
        self._model_id = f"soft_sensors_{uuid4().hex[:8]}"
        self._calibration_offsets: Dict[str, float] = {}

        # Fuel-specific parameters
        self._fuel_params = self._get_fuel_parameters(fuel_type)

        logger.info(f"SoftSensorModels initialized for fuel_type={fuel_type}")

    def infer_fuel_quality(
        self,
        o2: float,
        co: float,
        heat_balance: float
    ) -> FuelQualityEstimate:
        """
        Infer fuel quality from combustion measurements.

        Uses O2, CO levels, and heat balance to estimate fuel heating value
        and composition. This is useful when fuel quality varies (e.g., biogas).

        Args:
            o2: Stack O2 percentage (0-21%)
            co: CO concentration in ppm
            heat_balance: Heat balance ratio (actual/expected heat output)

        Returns:
            FuelQualityEstimate with heating value and quality grade
        """
        start_time = time.time()

        # Infer heating value from heat balance
        reference_lhv = self._fuel_params['lhv']
        heating_value = reference_lhv * heat_balance

        # Adjust for combustion efficiency (indicated by CO)
        if co > 200:
            # High CO suggests incomplete combustion, not low fuel quality
            combustion_factor = min(1.0, 500.0 / co)
            heating_value = heating_value / combustion_factor

        # Estimate methane fraction (for natural gas)
        if self.fuel_type == "natural_gas":
            # Lower LHV suggests more inerts or heavier hydrocarbons
            methane_fraction = min(1.0, heating_value / 55.5)  # Pure CH4 = 55.5 MJ/kg
        else:
            methane_fraction = 0.0

        # Hydrogen index (H/C ratio)
        hydrogen_index = 4.0 * methane_fraction + 2.0 * (1 - methane_fraction)

        # Quality grade
        if heating_value >= 0.95 * reference_lhv:
            quality_grade = "A"
        elif heating_value >= 0.85 * reference_lhv:
            quality_grade = "B"
        elif heating_value >= 0.75 * reference_lhv:
            quality_grade = "C"
        else:
            quality_grade = "D"

        # Anomaly detection
        anomaly_detected = (
            heating_value < 0.7 * reference_lhv or
            heating_value > 1.1 * reference_lhv or
            co > 500
        )

        # Confidence interval
        uncertainty = 0.05 * heating_value  # 5% uncertainty
        confidence = ConfidenceInterval(
            lower=heating_value - 1.645 * uncertainty,
            upper=heating_value + 1.645 * uncertainty
        )

        computation_time_ms = (time.time() - start_time) * 1000

        return FuelQualityEstimate(
            heating_value_mj_kg=heating_value,
            heating_value_confidence=confidence,
            methane_fraction=methane_fraction,
            hydrogen_index=hydrogen_index,
            quality_grade=quality_grade,
            anomaly_detected=anomaly_detected,
            provenance=ProvenanceRecord.create(
                "fuel_quality_inference",
                {"o2": o2, "co": co, "heat_balance": heat_balance},
                {"heating_value": heating_value, "quality_grade": quality_grade},
                self._model_id, computation_time_ms
            )
        )

    def infer_excess_air(
        self,
        stack_o2: float,
        fuel_type: str = "natural_gas"
    ) -> ExcessAirEstimate:
        """
        Infer excess air from stack O2 measurement.

        Uses the oxygen balance equation to calculate excess air percentage
        from measured stack O2 concentration.

        Args:
            stack_o2: Measured stack O2 percentage (0-21%)
            fuel_type: Type of fuel being burned

        Returns:
            ExcessAirEstimate with excess air percentage and lambda
        """
        start_time = time.time()

        # Get stoichiometric parameters
        params = self._get_fuel_parameters(fuel_type)
        stoich_o2 = params.get('stoich_o2', 0.0)  # O2 at stoichiometric

        # Calculate excess air from O2
        # O2_dry = 21 * EA / (1 + EA) for dry basis
        # Rearranging: EA = O2 / (21 - O2)
        if stack_o2 >= 21.0:
            excess_air_percent = 1000.0  # Very high excess air
        elif stack_o2 <= stoich_o2:
            excess_air_percent = 0.0  # At or below stoichiometric
        else:
            excess_air_percent = 100.0 * stack_o2 / (21.0 - stack_o2)

        # Lambda (air-fuel equivalence ratio)
        lambda_value = 1.0 + excess_air_percent / 100.0

        # Stoichiometric air ratio
        stoich_ratio = lambda_value

        # Check if within optimal range
        optimal_range = params.get('optimal_excess_air', (10.0, 20.0))
        within_optimal = optimal_range[0] <= excess_air_percent <= optimal_range[1]

        # Confidence interval (O2 measurement typically +/- 0.2%)
        o2_uncertainty = 0.2
        ea_sensitivity = 100.0 / (21.0 - stack_o2) ** 2  # d(EA)/d(O2)
        ea_uncertainty = ea_sensitivity * o2_uncertainty

        confidence = ConfidenceInterval(
            lower=max(0.0, excess_air_percent - 1.645 * ea_uncertainty),
            upper=excess_air_percent + 1.645 * ea_uncertainty
        )

        computation_time_ms = (time.time() - start_time) * 1000

        return ExcessAirEstimate(
            excess_air_percent=excess_air_percent,
            excess_air_confidence=confidence,
            stoichiometric_air_ratio=stoich_ratio,
            lambda_value=lambda_value,
            within_optimal_range=within_optimal,
            provenance=ProvenanceRecord.create(
                "excess_air_inference",
                {"stack_o2": stack_o2, "fuel_type": fuel_type},
                {"excess_air_percent": excess_air_percent, "lambda": lambda_value},
                self._model_id, computation_time_ms
            )
        )

    def infer_heat_duty(
        self,
        temps: Dict[str, float],
        flows: Dict[str, float]
    ) -> HeatDutyEstimate:
        """
        Infer heat duty from temperature and flow measurements.

        Calculates heat duty using energy balance equations for
        process streams.

        Args:
            temps: Dictionary with temperature measurements in C
                   Keys: 'inlet', 'outlet', 'stack', 'ambient'
            flows: Dictionary with flow measurements
                   Keys: 'fuel_kg_s', 'air_kg_s', 'process_kg_s'

        Returns:
            HeatDutyEstimate with heat duty and efficiency
        """
        start_time = time.time()

        # Extract values with defaults
        t_inlet = temps.get('inlet', 25.0)
        t_outlet = temps.get('outlet', 200.0)
        t_stack = temps.get('stack', 250.0)
        t_ambient = temps.get('ambient', 25.0)

        fuel_flow = flows.get('fuel_kg_s', 0.1)
        process_flow = flows.get('process_kg_s', 1.0)

        # Calculate heat input from fuel
        lhv = self._fuel_params['lhv'] * 1e6  # Convert to J/kg
        heat_input_w = fuel_flow * lhv
        heat_input_mw = heat_input_w / 1e6

        # Calculate process heat duty
        # Q = m * Cp * (T_out - T_in)
        cp_process = 4.186e3  # J/kg.K for water, approximate
        heat_duty_w = process_flow * cp_process * (t_outlet - t_inlet)
        heat_duty_mw = heat_duty_w / 1e6

        # Calculate stack losses
        # Approximate stack loss from temperature
        cp_flue = 1100  # J/kg.K for flue gas
        flue_gas_factor = 1.1  # Flue gas / fuel mass ratio (approximate)
        flue_gas_flow = fuel_flow * self._fuel_params['stoich_air'] * flue_gas_factor
        stack_loss_w = flue_gas_flow * cp_flue * (t_stack - t_ambient)
        stack_loss_mw = stack_loss_w / 1e6

        # Other losses (radiation, unaccounted)
        other_losses_mw = max(0.0, heat_input_mw - heat_duty_mw - stack_loss_mw)
        total_losses_mw = stack_loss_mw + other_losses_mw

        # Thermal efficiency
        if heat_input_mw > 0:
            thermal_efficiency = heat_duty_mw / heat_input_mw
        else:
            thermal_efficiency = 0.0

        thermal_efficiency = max(0.0, min(1.0, thermal_efficiency))

        # Confidence interval (5% uncertainty)
        uncertainty = 0.05 * heat_duty_mw
        confidence = ConfidenceInterval(
            lower=max(0.0, heat_duty_mw - 1.645 * uncertainty),
            upper=heat_duty_mw + 1.645 * uncertainty
        )

        computation_time_ms = (time.time() - start_time) * 1000

        return HeatDutyEstimate(
            heat_duty_mw=heat_duty_mw,
            heat_duty_confidence=confidence,
            heat_input_mw=heat_input_mw,
            thermal_efficiency=thermal_efficiency,
            heat_loss_mw=total_losses_mw,
            provenance=ProvenanceRecord.create(
                "heat_duty_inference",
                {"temps": temps, "flows": flows},
                {"heat_duty_mw": heat_duty_mw, "efficiency": thermal_efficiency},
                self._model_id, computation_time_ms
            )
        )

    def detect_sensor_bias(
        self,
        measured: float,
        inferred: float,
        sensor_name: str = "O2_analyzer",
        threshold_percent: float = 5.0
    ) -> BiasDetection:
        """
        Detect sensor bias by comparing measured to inferred value.

        Uses soft sensor inference to validate physical sensor readings
        and detect calibration drift or sensor faults.

        Args:
            measured: Measured sensor value
            inferred: Soft sensor inferred value
            sensor_name: Name of sensor being validated
            threshold_percent: Threshold for significant bias (%)

        Returns:
            BiasDetection with bias analysis and recommendations
        """
        start_time = time.time()

        # Calculate bias
        bias = measured - inferred
        if abs(inferred) > 1e-6:
            bias_percent = 100.0 * bias / inferred
        else:
            bias_percent = 100.0 if abs(bias) > 1e-6 else 0.0

        # Determine severity
        abs_bias_pct = abs(bias_percent)
        if abs_bias_pct < threshold_percent:
            severity = "LOW"
            is_significant = False
            action = "No action required"
        elif abs_bias_pct < 2 * threshold_percent:
            severity = "MEDIUM"
            is_significant = True
            action = f"Schedule calibration check for {sensor_name}"
        elif abs_bias_pct < 3 * threshold_percent:
            severity = "HIGH"
            is_significant = True
            action = f"Perform calibration of {sensor_name} soon"
        else:
            severity = "CRITICAL"
            is_significant = True
            action = f"IMMEDIATE: Verify {sensor_name} - possible sensor fault"

        computation_time_ms = (time.time() - start_time) * 1000

        return BiasDetection(
            sensor_name=sensor_name,
            measured_value=measured,
            inferred_value=inferred,
            bias=bias,
            bias_percent=bias_percent,
            is_significant=is_significant,
            severity=severity,
            recommended_action=action,
            provenance=ProvenanceRecord.create(
                "sensor_bias_detection",
                {"measured": measured, "inferred": inferred, "sensor": sensor_name},
                {"bias": bias, "severity": severity},
                self._model_id, computation_time_ms
            )
        )

    def _get_fuel_parameters(self, fuel_type: str) -> Dict[str, Any]:
        """Get fuel-specific parameters."""
        fuel_params = {
            "natural_gas": {
                "lhv": 50.0,  # MJ/kg
                "stoich_air": 17.2,  # kg air / kg fuel
                "stoich_o2": 0.0,  # O2 at stoichiometric
                "optimal_excess_air": (10.0, 20.0),
                "co2_factor": 2.75  # kg CO2 / kg fuel
            },
            "fuel_oil": {
                "lhv": 42.5,
                "stoich_air": 14.7,
                "stoich_o2": 0.0,
                "optimal_excess_air": (15.0, 25.0),
                "co2_factor": 3.15
            },
            "propane": {
                "lhv": 46.4,
                "stoich_air": 15.7,
                "stoich_o2": 0.0,
                "optimal_excess_air": (10.0, 20.0),
                "co2_factor": 3.0
            },
            "biogas": {
                "lhv": 25.0,  # Variable, 50-60% CH4
                "stoich_air": 10.0,
                "stoich_o2": 0.0,
                "optimal_excess_air": (15.0, 30.0),
                "co2_factor": 1.5
            }
        }
        return fuel_params.get(fuel_type, fuel_params["natural_gas"])

    @property
    def model_id(self) -> str:
        """Get model identifier."""
        return self._model_id
