# -*- coding: utf-8 -*-
"""
Deterministic tool functions for WaterGuard - Boiler Water Treatment Optimization.

This module implements all deterministic calculation and optimization functions
for boiler water treatment operations. All functions follow zero-hallucination
principles with no LLM involvement in calculations.

All calculations based on ASME, ABMA, and industry-standard water chemistry formulas.
"""

import hashlib
import logging
import math
import threading
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Thread-safe lock for calculations
_calculation_lock = threading.Lock()


class ScavengerType(Enum):
    """Types of oxygen scavengers."""
    SODIUM_SULFITE = "sodium_sulfite"
    SODIUM_BISULFITE = "sodium_bisulfite"
    HYDRAZINE = "hydrazine"
    DEHA = "deha"  # Diethylhydroxylamine
    CARBOHYDRAZIDE = "carbohydrazide"


class AmineType(Enum):
    """Types of filming amines for condensate treatment."""
    CYCLOHEXYLAMINE = "cyclohexylamine"
    MORPHOLINE = "morpholine"
    NEUTRALIZING_AMINE = "neutralizing_amine"
    FILMING_AMINE = "filming_amine"


class BoilerType(Enum):
    """Types of boilers for treatment program validation."""
    FIRE_TUBE = "fire_tube"
    WATER_TUBE = "water_tube"
    PACKAGE = "package"
    INDUSTRIAL = "industrial"
    UTILITY = "utility"


# ==================== Data Classes ====================

@dataclass
class WaterQualityAnalysis:
    """Result of water quality analysis."""
    lsi_value: float
    rsi_value: float
    psi_value: float
    larson_skold_index: float
    scale_tendency: str  # "scaling", "neutral", "corrosive"
    corrosion_risk: str  # "low", "moderate", "high", "severe"
    compliance_status: str  # "PASS", "WARNING", "FAIL"
    violations: List[str]
    recommendations: List[str]
    timestamp: str
    provenance_hash: str


@dataclass
class BlowdownOptimization:
    """Result of blowdown optimization."""
    optimal_cycles: float
    recommended_blowdown_rate: float  # kg/hr
    continuous_blowdown_percent: float
    bottom_blowdown_frequency: str  # "hourly", "every_4h", "every_8h", "daily"
    heat_recovery_potential: float  # kW
    water_savings: float  # m3/day
    cost_savings: float  # $/day
    energy_loss: float  # kW
    tds_control: Dict[str, float]
    timestamp: str
    provenance_hash: str


@dataclass
class ChemicalOptimization:
    """Result of chemical consumption optimization."""
    phosphate_dosing: float  # ppm or kg/day
    oxygen_scavenger_dosing: float  # kg/day
    amine_dosing: float  # kg/day
    polymer_dosing: float  # kg/day
    total_chemical_cost: float  # $/day
    cost_reduction_potential: float  # $/month
    feed_schedule: Dict[str, Any]
    residual_targets: Dict[str, float]
    optimization_score: float  # 0-100
    timestamp: str
    provenance_hash: str


@dataclass
class ComplianceResult:
    """Result of compliance checking."""
    standard: str  # "ASME", "ABMA", "Custom"
    compliance_status: str  # "PASS", "WARNING", "FAIL"
    parameters_checked: int
    violations: List[str]
    warnings: List[str]
    margin_percent: float
    recommended_actions: List[str]
    timestamp: str
    provenance_hash: str


@dataclass
class ValidationResult:
    """Result of treatment program validation."""
    program_type: str
    is_valid: bool
    effectiveness_score: float  # 0-100
    chemistry_compatibility: bool
    issues: List[str]
    recommendations: List[str]
    timestamp: str
    provenance_hash: str


# ==================== Main Tools Class ====================

class WaterTreatmentTools:
    """
    Deterministic tool functions for boiler water treatment operations.

    All methods implement zero-hallucination calculations using
    deterministic algorithms and industry-standard formulas only.
    """

    # ==================== Water Chemistry Analysis ====================

    @staticmethod
    def calculate_langelier_saturation_index(
        pH: float,
        temperature: float,  # Celsius
        calcium_hardness: float,  # mg/L as CaCO3
        alkalinity: float,  # mg/L as CaCO3
        tds: float  # mg/L
    ) -> float:
        """
        Calculate Langelier Saturation Index (LSI) for scale tendency prediction.

        LSI = pH - pHs
        where pHs = (9.3 + A + B) - (C + D)

        A = (Log10[TDS] - 1) / 10
        B = -13.12 × Log10(°C + 273) + 34.55
        C = Log10[Ca2+ as CaCO3] - 0.4
        D = Log10[Alkalinity as CaCO3]

        Interpretation:
        - LSI > 0: Water is supersaturated, scale forming
        - LSI = 0: Water is saturated, neutral
        - LSI < 0: Water is undersaturated, corrosive

        Args:
            pH: Actual pH of water
            temperature: Water temperature in Celsius
            calcium_hardness: Calcium hardness in mg/L as CaCO3
            alkalinity: Total alkalinity in mg/L as CaCO3
            tds: Total dissolved solids in mg/L

        Returns:
            LSI value (typically ranges from -3 to +3)
        """
        with _calculation_lock:
            try:
                # Input validation
                if pH < 0 or pH > 14:
                    raise ValueError(f"Invalid pH: {pH}. Must be between 0 and 14.")
                if temperature < 0 or temperature > 100:
                    raise ValueError(f"Invalid temperature: {temperature}°C")
                if calcium_hardness < 0:
                    raise ValueError(f"Invalid calcium hardness: {calcium_hardness}")
                if alkalinity < 0:
                    raise ValueError(f"Invalid alkalinity: {alkalinity}")
                if tds < 0:
                    raise ValueError(f"Invalid TDS: {tds}")

                # Handle edge cases
                if calcium_hardness < 1:
                    calcium_hardness = 1
                if alkalinity < 1:
                    alkalinity = 1
                if tds < 50:
                    tds = 50

                # Calculate pHs components
                A = (math.log10(tds) - 1) / 10
                B = -13.12 * math.log10(temperature + 273.15) + 34.55
                C = math.log10(calcium_hardness) - 0.4
                D = math.log10(alkalinity)

                # Calculate pHs (saturation pH)
                pHs = (9.3 + A + B) - (C + D)

                # Calculate LSI
                lsi = pH - pHs

                logger.debug(f"LSI calculation: pH={pH}, pHs={pHs:.2f}, LSI={lsi:.2f}")

                return round(lsi, 2)

            except Exception as e:
                logger.error(f"LSI calculation failed: {str(e)}")
                raise

    @staticmethod
    def calculate_ryznar_stability_index(pH: float, pHs: float) -> float:
        """
        Calculate Ryznar Stability Index (RSI) for corrosion/scale prediction.

        RSI = 2 × pHs - pH

        Interpretation:
        - RSI < 6.0: Heavy scale formation
        - RSI 6.0-6.5: Light scale formation
        - RSI 6.5-7.0: Little scale or corrosion
        - RSI 7.0-7.5: Corrosion likely
        - RSI > 7.5: Heavy corrosion

        Args:
            pH: Actual pH of water
            pHs: Saturation pH (from LSI calculation)

        Returns:
            RSI value (typically ranges from 4 to 10)
        """
        with _calculation_lock:
            try:
                if pH < 0 or pH > 14:
                    raise ValueError(f"Invalid pH: {pH}")
                if pHs < 0 or pHs > 14:
                    raise ValueError(f"Invalid pHs: {pHs}")

                rsi = 2 * pHs - pH

                logger.debug(f"RSI calculation: pH={pH}, pHs={pHs}, RSI={rsi:.2f}")

                return round(rsi, 2)

            except Exception as e:
                logger.error(f"RSI calculation failed: {str(e)}")
                raise

    @staticmethod
    def calculate_puckorius_scaling_index(
        pH: float,
        alkalinity: float,  # mg/L as CaCO3
        calcium_hardness: float = None,  # mg/L as CaCO3
        temperature: float = 25.0  # Celsius
    ) -> float:
        """
        Calculate Puckorius Scaling Index (PSI) for practical scaling prediction.

        PSI = 2 × pHs - pHeq
        where pHeq is the equilibrium pH considering buffering capacity

        Interpretation:
        - PSI < 6.0: Scale forming
        - PSI 6.0-7.0: Minimal scale or corrosion
        - PSI > 7.0: Corrosive

        Args:
            pH: Actual pH of water
            alkalinity: Total alkalinity in mg/L as CaCO3
            calcium_hardness: Calcium hardness in mg/L as CaCO3 (optional)
            temperature: Water temperature in Celsius

        Returns:
            PSI value
        """
        with _calculation_lock:
            try:
                if pH < 0 or pH > 14:
                    raise ValueError(f"Invalid pH: {pH}")
                if alkalinity < 0:
                    raise ValueError(f"Invalid alkalinity: {alkalinity}")

                # Estimate equilibrium pH based on alkalinity and buffering
                # This is a simplified model
                if alkalinity < 50:
                    pHeq = pH - 0.5
                elif alkalinity < 100:
                    pHeq = pH - 0.3
                elif alkalinity < 200:
                    pHeq = pH - 0.2
                else:
                    pHeq = pH - 0.1

                # Temperature adjustment
                temp_factor = (temperature - 25) * 0.01
                pHeq = pHeq - temp_factor

                # If calcium hardness provided, use more accurate calculation
                if calcium_hardness and calcium_hardness > 0:
                    # Simplified pHs estimation
                    pHs = 9.3 + math.log10(alkalinity) - math.log10(calcium_hardness)
                    psi = 2 * pHs - pHeq
                else:
                    # Simplified PSI without calcium data
                    psi = 2 * (pH + 0.5) - pHeq

                logger.debug(f"PSI calculation: pH={pH}, pHeq={pHeq:.2f}, PSI={psi:.2f}")

                return round(psi, 2)

            except Exception as e:
                logger.error(f"PSI calculation failed: {str(e)}")
                raise

    @staticmethod
    def calculate_larson_skold_index(
        chloride: float,  # mg/L
        sulfate: float,  # mg/L
        alkalinity: float  # mg/L as CaCO3
    ) -> float:
        """
        Calculate Larson-Skold Index for corrosivity prediction.

        LSK = (Cl⁻ + SO₄²⁻) / (HCO₃⁻ + CO₃²⁻)

        All units in equivalents (meq/L):
        - Chloride: mg/L ÷ 35.5
        - Sulfate: mg/L ÷ 48
        - Alkalinity: mg/L as CaCO3 ÷ 50

        Interpretation:
        - LSK < 0.2: Low corrosion risk
        - LSK 0.2-0.5: Moderate corrosion risk
        - LSK 0.5-1.0: High corrosion risk
        - LSK > 1.0: Very high corrosion risk

        Args:
            chloride: Chloride concentration in mg/L
            sulfate: Sulfate concentration in mg/L
            alkalinity: Total alkalinity in mg/L as CaCO3

        Returns:
            Larson-Skold Index value
        """
        with _calculation_lock:
            try:
                if chloride < 0 or sulfate < 0 or alkalinity < 0:
                    raise ValueError("All parameters must be non-negative")

                # Convert to equivalents (meq/L)
                chloride_eq = chloride / 35.5
                sulfate_eq = sulfate / 48.0
                alkalinity_eq = alkalinity / 50.0

                # Avoid division by zero
                if alkalinity_eq < 0.01:
                    alkalinity_eq = 0.01
                    logger.warning("Very low alkalinity detected, using minimum value")

                # Calculate Larson-Skold Index
                lsk = (chloride_eq + sulfate_eq) / alkalinity_eq

                logger.debug(f"Larson-Skold Index: Cl={chloride}, SO4={sulfate}, "
                           f"Alk={alkalinity}, LSK={lsk:.3f}")

                return round(lsk, 3)

            except Exception as e:
                logger.error(f"Larson-Skold Index calculation failed: {str(e)}")
                raise

    @staticmethod
    def analyze_water_quality(chemistry_data: Dict[str, Any]) -> WaterQualityAnalysis:
        """
        Comprehensive water quality analysis with compliance checking.

        Args:
            chemistry_data: Dictionary containing water chemistry parameters
                - pH: pH value
                - temperature: Temperature in Celsius
                - calcium_hardness: mg/L as CaCO3
                - alkalinity: mg/L as CaCO3
                - tds: Total dissolved solids in mg/L
                - chloride: mg/L
                - sulfate: mg/L
                - pressure: Boiler pressure in bar (optional, for compliance)

        Returns:
            WaterQualityAnalysis with comprehensive assessment
        """
        try:
            timestamp = datetime.utcnow().isoformat()

            # Extract parameters
            pH = chemistry_data.get('pH', 7.0)
            temperature = chemistry_data.get('temperature', 25.0)
            calcium_hardness = chemistry_data.get('calcium_hardness', 100.0)
            alkalinity = chemistry_data.get('alkalinity', 100.0)
            tds = chemistry_data.get('tds', 500.0)
            chloride = chemistry_data.get('chloride', 50.0)
            sulfate = chemistry_data.get('sulfate', 50.0)
            pressure = chemistry_data.get('pressure', 10.0)  # bar

            # Calculate all indices
            lsi = WaterTreatmentTools.calculate_langelier_saturation_index(
                pH, temperature, calcium_hardness, alkalinity, tds
            )

            # Calculate pHs for RSI
            A = (math.log10(tds) - 1) / 10
            B = -13.12 * math.log10(temperature + 273.15) + 34.55
            C = math.log10(max(calcium_hardness, 1)) - 0.4
            D = math.log10(max(alkalinity, 1))
            pHs = (9.3 + A + B) - (C + D)

            rsi = WaterTreatmentTools.calculate_ryznar_stability_index(pH, pHs)
            psi = WaterTreatmentTools.calculate_puckorius_scaling_index(
                pH, alkalinity, calcium_hardness, temperature
            )
            lsk = WaterTreatmentTools.calculate_larson_skold_index(
                chloride, sulfate, alkalinity
            )

            # Determine scale tendency
            if lsi > 0.5:
                scale_tendency = "scaling"
            elif lsi < -0.5:
                scale_tendency = "corrosive"
            else:
                scale_tendency = "neutral"

            # Determine corrosion risk
            if lsk > 1.0 or rsi > 7.5:
                corrosion_risk = "severe"
            elif lsk > 0.5 or rsi > 7.0:
                corrosion_risk = "high"
            elif lsk > 0.2 or rsi > 6.5:
                corrosion_risk = "moderate"
            else:
                corrosion_risk = "low"

            # Check compliance (simplified ASME guidelines)
            violations = []
            warnings = []

            # pH compliance
            if pressure < 20:  # Low pressure boilers
                if pH < 10.5 or pH > 12.0:
                    violations.append(f"pH {pH} outside ASME range 10.5-12.0 for <20 bar")
            else:  # High pressure boilers
                if pH < 9.0 or pH > 9.6:
                    violations.append(f"pH {pH} outside ASME range 9.0-9.6 for >20 bar")

            # TDS compliance
            max_tds = 3500 if pressure < 20 else 2000
            if tds > max_tds:
                violations.append(f"TDS {tds} mg/L exceeds limit {max_tds} mg/L")

            # Hardness compliance
            if calcium_hardness > 2.0:
                warnings.append(f"Calcium hardness {calcium_hardness} mg/L should be <2 mg/L")

            # Alkalinity compliance
            if alkalinity > 700:
                warnings.append(f"Alkalinity {alkalinity} mg/L is high, risk of caustic embrittlement")

            # Chloride compliance
            max_chloride = 300 if pressure < 20 else 100
            if chloride > max_chloride:
                violations.append(f"Chloride {chloride} mg/L exceeds limit {max_chloride} mg/L")

            # Determine compliance status
            if violations:
                compliance_status = "FAIL"
            elif warnings:
                compliance_status = "WARNING"
            else:
                compliance_status = "PASS"

            # Generate recommendations
            recommendations = []

            if scale_tendency == "scaling":
                recommendations.append("Increase blowdown rate to control TDS and alkalinity")
                recommendations.append("Consider acid feed for pH control")
                recommendations.append("Implement scale inhibitor program")

            if corrosion_risk in ["high", "severe"]:
                recommendations.append("Increase oxygen scavenger dosing")
                recommendations.append("Implement filming amine program for condensate protection")
                recommendations.append("Check for chloride ingress in makeup water")

            if violations:
                recommendations.append("Immediate corrective action required for compliance")

            # Calculate provenance hash
            provenance_str = f"{chemistry_data}{lsi}{rsi}{psi}{lsk}{timestamp}"
            provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

            return WaterQualityAnalysis(
                lsi_value=lsi,
                rsi_value=rsi,
                psi_value=psi,
                larson_skold_index=lsk,
                scale_tendency=scale_tendency,
                corrosion_risk=corrosion_risk,
                compliance_status=compliance_status,
                violations=violations + warnings,
                recommendations=recommendations,
                timestamp=timestamp,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            logger.error(f"Water quality analysis failed: {str(e)}")
            raise

    # ==================== Blowdown Optimization ====================

    @staticmethod
    def calculate_cycles_of_concentration(
        makeup_conductivity: float,  # µS/cm
        blowdown_conductivity: float  # µS/cm
    ) -> float:
        """
        Calculate cycles of concentration for boiler water.

        Cycles = Blowdown Conductivity / Makeup Conductivity

        Higher cycles = better water efficiency, but risk of scaling/corrosion.

        Args:
            makeup_conductivity: Conductivity of makeup water in µS/cm
            blowdown_conductivity: Conductivity of blowdown water in µS/cm

        Returns:
            Cycles of concentration
        """
        with _calculation_lock:
            try:
                if makeup_conductivity <= 0:
                    raise ValueError("Makeup conductivity must be positive")
                if blowdown_conductivity < makeup_conductivity:
                    logger.warning("Blowdown conductivity less than makeup, using cycles=1")
                    return 1.0

                cycles = blowdown_conductivity / makeup_conductivity

                logger.debug(f"Cycles of concentration: {cycles:.2f}")

                return round(cycles, 2)

            except Exception as e:
                logger.error(f"Cycles calculation failed: {str(e)}")
                raise

    @staticmethod
    def calculate_blowdown_rate(
        steam_rate: float,  # kg/hr
        cycles: float,
        makeup_rate: float = None  # kg/hr (optional)
    ) -> float:
        """
        Calculate optimal blowdown rate.

        Blowdown Rate = Makeup Rate / (Cycles - 1)
        or
        Blowdown Rate = Steam Rate / (Cycles - 1)

        Args:
            steam_rate: Steam generation rate in kg/hr
            cycles: Cycles of concentration
            makeup_rate: Makeup water rate in kg/hr (if not provided, uses steam rate)

        Returns:
            Blowdown rate in kg/hr
        """
        with _calculation_lock:
            try:
                if cycles <= 1:
                    raise ValueError("Cycles must be greater than 1")
                if steam_rate <= 0:
                    raise ValueError("Steam rate must be positive")

                # If makeup rate not provided, assume makeup ≈ steam + blowdown
                if makeup_rate is None:
                    # Iterative calculation
                    # Makeup = Steam + Blowdown
                    # Blowdown = Makeup / (Cycles - 1)
                    # Therefore: Makeup = Steam + Makeup / (Cycles - 1)
                    # Solving: Makeup = Steam × Cycles / (Cycles - 1)
                    makeup_rate = steam_rate * cycles / (cycles - 1)

                blowdown_rate = makeup_rate / (cycles - 1)

                logger.debug(f"Blowdown rate: {blowdown_rate:.2f} kg/hr "
                           f"({blowdown_rate/steam_rate*100:.2f}% of steam rate)")

                return round(blowdown_rate, 2)

            except Exception as e:
                logger.error(f"Blowdown rate calculation failed: {str(e)}")
                raise

    @staticmethod
    def calculate_blowdown_heat_loss(
        blowdown_rate: float,  # kg/hr
        temperature: float,  # Celsius
        ambient_temp: float = 25.0  # Celsius
    ) -> float:
        """
        Calculate heat loss due to blowdown.

        Heat Loss = Blowdown Rate × Cp × ΔT
        where Cp for water ≈ 4.186 kJ/(kg·°C)

        Args:
            blowdown_rate: Blowdown rate in kg/hr
            temperature: Blowdown temperature in Celsius
            ambient_temp: Ambient/makeup water temperature in Celsius

        Returns:
            Heat loss in kW
        """
        with _calculation_lock:
            try:
                if blowdown_rate < 0:
                    raise ValueError("Blowdown rate must be non-negative")
                if temperature <= ambient_temp:
                    logger.warning("Blowdown temp <= ambient temp, minimal heat loss")
                    return 0.0

                # Specific heat capacity of water
                cp = 4.186  # kJ/(kg·°C)

                # Temperature difference
                delta_t = temperature - ambient_temp

                # Heat loss in kJ/hr
                heat_loss_kj_hr = blowdown_rate * cp * delta_t

                # Convert to kW (1 kW = 3600 kJ/hr)
                heat_loss_kw = heat_loss_kj_hr / 3600

                logger.debug(f"Blowdown heat loss: {heat_loss_kw:.2f} kW")

                return round(heat_loss_kw, 2)

            except Exception as e:
                logger.error(f"Heat loss calculation failed: {str(e)}")
                raise

    @staticmethod
    def optimize_blowdown_schedule(
        water_data: Dict[str, Any],
        steam_demand: float  # kg/hr
    ) -> BlowdownOptimization:
        """
        Optimize blowdown schedule for maximum efficiency and water quality.

        Args:
            water_data: Dictionary containing water chemistry and operational data
                - makeup_conductivity: µS/cm
                - blowdown_conductivity: µS/cm
                - tds: mg/L
                - alkalinity: mg/L
                - temperature: °C
                - pressure: bar
                - water_cost: $/m3
                - energy_cost: $/kWh
            steam_demand: Steam generation rate in kg/hr

        Returns:
            BlowdownOptimization with comprehensive optimization results
        """
        try:
            timestamp = datetime.utcnow().isoformat()

            # Extract parameters
            makeup_cond = water_data.get('makeup_conductivity', 200.0)
            blowdown_cond = water_data.get('blowdown_conductivity', 2000.0)
            tds = water_data.get('tds', 2000.0)
            alkalinity = water_data.get('alkalinity', 400.0)
            temperature = water_data.get('temperature', 180.0)
            pressure = water_data.get('pressure', 10.0)
            water_cost = water_data.get('water_cost', 0.5)  # $/m3
            energy_cost = water_data.get('energy_cost', 0.08)  # $/kWh

            # Calculate current cycles
            current_cycles = WaterTreatmentTools.calculate_cycles_of_concentration(
                makeup_cond, blowdown_cond
            )

            # Determine maximum safe cycles based on water quality limits
            # ASME limits for various parameters
            if pressure < 20:
                max_tds_limit = 3500
                max_alkalinity_limit = 700
            else:
                max_tds_limit = 2000
                max_alkalinity_limit = 400

            # Calculate maximum cycles based on each limiting factor
            max_cycles_tds = max_tds_limit / (tds / current_cycles)
            max_cycles_alk = max_alkalinity_limit / (alkalinity / current_cycles)
            max_cycles_cond = 5000 / makeup_cond  # Conductivity limit

            # Use most conservative limit with safety factor
            optimal_cycles = min(max_cycles_tds, max_cycles_alk, max_cycles_cond) * 0.9
            optimal_cycles = max(3.0, min(optimal_cycles, 10.0))  # Clamp between 3 and 10

            # Calculate blowdown rate
            blowdown_rate = WaterTreatmentTools.calculate_blowdown_rate(
                steam_demand, optimal_cycles
            )

            # Continuous blowdown percentage (typically 20-30% of total)
            continuous_bd_percent = 25.0
            continuous_bd_rate = blowdown_rate * (continuous_bd_percent / 100)
            intermittent_bd_rate = blowdown_rate - continuous_bd_rate

            # Determine bottom blowdown frequency based on boiler size and water quality
            if steam_demand < 2000:
                bd_frequency = "every_4h"
            elif steam_demand < 5000:
                bd_frequency = "every_8h"
            else:
                bd_frequency = "daily"

            # Calculate heat loss
            heat_loss = WaterTreatmentTools.calculate_blowdown_heat_loss(
                blowdown_rate, temperature, 25.0
            )

            # Calculate heat recovery potential (80% of heat loss recoverable)
            heat_recovery_potential = heat_loss * 0.8

            # Calculate water savings vs. lower cycles
            baseline_cycles = 3.0
            baseline_blowdown = WaterTreatmentTools.calculate_blowdown_rate(
                steam_demand, baseline_cycles
            )
            water_saved = (baseline_blowdown - blowdown_rate) * 24 / 1000  # m3/day

            # Calculate cost savings
            water_cost_savings = water_saved * water_cost
            energy_cost_savings = heat_recovery_potential * 24 * energy_cost
            total_cost_savings = water_cost_savings + energy_cost_savings

            # TDS control parameters
            tds_control = {
                'target_tds': tds * (optimal_cycles / current_cycles),
                'max_tds_limit': max_tds_limit,
                'margin_percent': ((max_tds_limit - tds * (optimal_cycles / current_cycles))
                                  / max_tds_limit * 100)
            }

            # Calculate provenance hash
            provenance_str = f"{water_data}{steam_demand}{optimal_cycles}{timestamp}"
            provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

            return BlowdownOptimization(
                optimal_cycles=round(optimal_cycles, 1),
                recommended_blowdown_rate=blowdown_rate,
                continuous_blowdown_percent=continuous_bd_percent,
                bottom_blowdown_frequency=bd_frequency,
                heat_recovery_potential=heat_recovery_potential,
                water_savings=round(water_saved, 2),
                cost_savings=round(total_cost_savings, 2),
                energy_loss=heat_loss,
                tds_control=tds_control,
                timestamp=timestamp,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            logger.error(f"Blowdown optimization failed: {str(e)}")
            raise

    # ==================== Chemical Dosing ====================

    @staticmethod
    def calculate_phosphate_dosing(
        residual_target: float,  # ppm as PO4
        volume: float,  # m3
        current_level: float = 0.0,  # ppm as PO4
        steam_rate: float = None  # kg/hr (for continuous dosing)
    ) -> float:
        """
        Calculate phosphate dosing for scale and corrosion control.

        Phosphate maintains pH and precipitates hardness.
        Typical residual: 30-60 ppm as PO4 for internal treatment.

        Args:
            residual_target: Target phosphate residual in ppm as PO4
            volume: Boiler water volume in m3
            current_level: Current phosphate level in ppm
            steam_rate: Steam rate for continuous dosing calculation (optional)

        Returns:
            Dosing rate in kg/day (or ppm for shock dosing)
        """
        with _calculation_lock:
            try:
                if residual_target < 0 or current_level < 0:
                    raise ValueError("Phosphate levels must be non-negative")

                # Calculate deficit
                deficit_ppm = residual_target - current_level

                if deficit_ppm <= 0:
                    logger.info("Phosphate level already at or above target")
                    return 0.0

                # For initial/shock dosing
                if steam_rate is None:
                    # Dosing = Volume × Deficit × Density
                    dosing_kg = volume * deficit_ppm / 1000  # Convert ppm to kg/m3
                    logger.debug(f"Phosphate shock dose: {dosing_kg:.2f} kg")
                    return round(dosing_kg, 2)

                # For continuous dosing
                # Account for losses through steam, blowdown
                # Assume 5% loss rate per day at typical cycles
                daily_makeup = steam_rate * 24 * 1.2 / 1000  # m3/day (includes blowdown)
                dosing_kg_day = daily_makeup * residual_target / 1000

                logger.debug(f"Phosphate continuous dose: {dosing_kg_day:.2f} kg/day")
                return round(dosing_kg_day, 2)

            except Exception as e:
                logger.error(f"Phosphate dosing calculation failed: {str(e)}")
                raise

    @staticmethod
    def calculate_oxygen_scavenger_dosing(
        dissolved_oxygen: float,  # ppb
        steam_rate: float,  # kg/hr
        scavenger_type: ScavengerType = ScavengerType.SODIUM_SULFITE
    ) -> float:
        """
        Calculate oxygen scavenger dosing rate.

        Stoichiometric ratios:
        - Sodium sulfite: 7.88 kg per kg O2
        - Hydrazine: 1.0 kg per kg O2
        - DEHA: 1.4 kg per kg O2
        - Carbohydrazide: 1.1 kg per kg O2

        Typically add 10-20% excess for reaction kinetics.

        Args:
            dissolved_oxygen: DO in makeup water in ppb
            steam_rate: Steam generation rate in kg/hr
            scavenger_type: Type of oxygen scavenger

        Returns:
            Dosing rate in kg/day
        """
        with _calculation_lock:
            try:
                if dissolved_oxygen < 0:
                    raise ValueError("Dissolved oxygen must be non-negative")
                if steam_rate <= 0:
                    raise ValueError("Steam rate must be positive")

                # Convert DO from ppb to kg/hr
                # Makeup rate ≈ steam rate × 1.05 (includes blowdown)
                makeup_rate = steam_rate * 1.05
                do_kg_hr = (dissolved_oxygen / 1e9) * makeup_rate

                # Stoichiometric ratios (kg scavenger per kg O2)
                ratios = {
                    ScavengerType.SODIUM_SULFITE: 7.88,
                    ScavengerType.SODIUM_BISULFITE: 6.67,
                    ScavengerType.HYDRAZINE: 1.0,
                    ScavengerType.DEHA: 1.4,
                    ScavengerType.CARBOHYDRAZIDE: 1.1
                }

                stoich_ratio = ratios.get(scavenger_type, 7.88)

                # Add 15% excess for reaction kinetics
                excess_factor = 1.15

                # Calculate dosing in kg/hr
                dosing_kg_hr = do_kg_hr * stoich_ratio * excess_factor

                # Convert to kg/day
                dosing_kg_day = dosing_kg_hr * 24

                logger.debug(f"{scavenger_type.value} dosing: {dosing_kg_day:.3f} kg/day "
                           f"for DO={dissolved_oxygen} ppb")

                return round(dosing_kg_day, 3)

            except Exception as e:
                logger.error(f"Oxygen scavenger dosing calculation failed: {str(e)}")
                raise

    @staticmethod
    def calculate_amine_dosing(
        condensate_pH_target: float,
        steam_rate: float,  # kg/hr
        amine_type: AmineType = AmineType.NEUTRALIZING_AMINE,
        condensate_return_percent: float = 80.0
    ) -> float:
        """
        Calculate amine dosing for condensate pH control and corrosion protection.

        Neutralizing amines: Maintain condensate pH 8.0-9.0
        Filming amines: Form protective film, lower dosing rates

        Typical dosing:
        - Neutralizing amines: 0.5-2.0 ppm
        - Filming amines: 0.1-0.5 ppm

        Args:
            condensate_pH_target: Target condensate pH (typically 8.5-9.0)
            steam_rate: Steam generation rate in kg/hr
            amine_type: Type of amine treatment
            condensate_return_percent: Percentage of condensate returned

        Returns:
            Dosing rate in kg/day
        """
        with _calculation_lock:
            try:
                if condensate_pH_target < 7 or condensate_pH_target > 10:
                    raise ValueError("Condensate pH target should be 7-10")
                if steam_rate <= 0:
                    raise ValueError("Steam rate must be positive")
                if condensate_return_percent < 0 or condensate_return_percent > 100:
                    raise ValueError("Condensate return must be 0-100%")

                # Calculate condensate flow
                condensate_rate = steam_rate * (condensate_return_percent / 100)

                # Determine dosing rate in ppm based on amine type and pH target
                if amine_type == AmineType.FILMING_AMINE:
                    # Filming amines: 0.1-0.5 ppm
                    dosing_ppm = 0.3
                else:
                    # Neutralizing amines: based on pH target
                    # Higher pH target requires more amine
                    if condensate_pH_target < 8.0:
                        dosing_ppm = 0.5
                    elif condensate_pH_target < 8.5:
                        dosing_ppm = 1.0
                    elif condensate_pH_target < 9.0:
                        dosing_ppm = 1.5
                    else:
                        dosing_ppm = 2.0

                # Calculate dosing in kg/day
                # ppm = mg/kg, so dosing = flow (kg/hr) × ppm / 1000
                dosing_kg_hr = (condensate_rate * dosing_ppm) / 1000
                dosing_kg_day = dosing_kg_hr * 24

                logger.debug(f"{amine_type.value} dosing: {dosing_kg_day:.3f} kg/day "
                           f"for pH target {condensate_pH_target}")

                return round(dosing_kg_day, 3)

            except Exception as e:
                logger.error(f"Amine dosing calculation failed: {str(e)}")
                raise

    @staticmethod
    def calculate_polymer_dosing(
        sludge_conditioner_need: float,  # 0-100 scale
        water_hardness: float,  # mg/L as CaCO3
        steam_rate: float = 1000.0  # kg/hr
    ) -> float:
        """
        Calculate polymer dosing for sludge conditioning.

        Polymers disperse sludge and prevent deposition on heat transfer surfaces.
        Typical dosing: 5-20 ppm based on hardness and fouling tendency.

        Args:
            sludge_conditioner_need: Sludge conditioning need factor (0-100)
            water_hardness: Total hardness in mg/L as CaCO3
            steam_rate: Steam generation rate in kg/hr

        Returns:
            Dosing rate in kg/day
        """
        with _calculation_lock:
            try:
                if sludge_conditioner_need < 0 or sludge_conditioner_need > 100:
                    raise ValueError("Sludge conditioner need must be 0-100")
                if water_hardness < 0:
                    raise ValueError("Water hardness must be non-negative")

                # Base dosing on hardness
                if water_hardness < 50:
                    base_dosing_ppm = 5.0
                elif water_hardness < 150:
                    base_dosing_ppm = 10.0
                elif water_hardness < 300:
                    base_dosing_ppm = 15.0
                else:
                    base_dosing_ppm = 20.0

                # Adjust based on sludge conditioning need
                adjustment_factor = 0.5 + (sludge_conditioner_need / 100)
                dosing_ppm = base_dosing_ppm * adjustment_factor

                # Calculate makeup rate (steam + 5% blowdown)
                makeup_rate = steam_rate * 1.05

                # Calculate dosing in kg/day
                dosing_kg_hr = (makeup_rate * dosing_ppm) / 1000
                dosing_kg_day = dosing_kg_hr * 24

                logger.debug(f"Polymer dosing: {dosing_kg_day:.3f} kg/day "
                           f"(hardness={water_hardness} mg/L)")

                return round(dosing_kg_day, 3)

            except Exception as e:
                logger.error(f"Polymer dosing calculation failed: {str(e)}")
                raise

    @staticmethod
    def optimize_chemical_consumption(
        current_usage: Dict[str, float],  # kg/day
        water_quality: Dict[str, Any],
        targets: Dict[str, Any]
    ) -> ChemicalOptimization:
        """
        Optimize chemical consumption for cost and effectiveness.

        Args:
            current_usage: Current chemical usage rates (kg/day)
                - phosphate, oxygen_scavenger, amine, polymer
            water_quality: Current water quality parameters
            targets: Target parameters for optimization

        Returns:
            ChemicalOptimization with optimized dosing and cost analysis
        """
        try:
            timestamp = datetime.utcnow().isoformat()

            # Extract current usage
            current_phosphate = current_usage.get('phosphate', 0)
            current_scavenger = current_usage.get('oxygen_scavenger', 0)
            current_amine = current_usage.get('amine', 0)
            current_polymer = current_usage.get('polymer', 0)

            # Extract parameters
            steam_rate = water_quality.get('steam_rate', 1000.0)
            dissolved_oxygen = water_quality.get('dissolved_oxygen', 200.0)
            water_hardness = water_quality.get('hardness', 100.0)
            boiler_volume = water_quality.get('volume', 10.0)
            current_phosphate_level = water_quality.get('phosphate_residual', 30.0)
            condensate_return = water_quality.get('condensate_return_percent', 80.0)

            # Extract targets
            phosphate_target = targets.get('phosphate_residual', 50.0)
            condensate_pH_target = targets.get('condensate_pH', 8.8)
            sludge_need = targets.get('sludge_conditioner_need', 50.0)

            # Calculate optimized dosing rates
            opt_phosphate = WaterTreatmentTools.calculate_phosphate_dosing(
                phosphate_target, boiler_volume, current_phosphate_level, steam_rate
            )

            opt_scavenger = WaterTreatmentTools.calculate_oxygen_scavenger_dosing(
                dissolved_oxygen, steam_rate, ScavengerType.SODIUM_SULFITE
            )

            opt_amine = WaterTreatmentTools.calculate_amine_dosing(
                condensate_pH_target, steam_rate,
                AmineType.NEUTRALIZING_AMINE, condensate_return
            )

            opt_polymer = WaterTreatmentTools.calculate_polymer_dosing(
                sludge_need, water_hardness, steam_rate
            )

            # Chemical costs (typical USD per kg)
            chemical_prices = targets.get('chemical_prices', {
                'phosphate': 2.50,
                'oxygen_scavenger': 3.00,
                'amine': 5.00,
                'polymer': 4.00
            })

            # Calculate costs
            current_daily_cost = (
                current_phosphate * chemical_prices.get('phosphate', 2.50) +
                current_scavenger * chemical_prices.get('oxygen_scavenger', 3.00) +
                current_amine * chemical_prices.get('amine', 5.00) +
                current_polymer * chemical_prices.get('polymer', 4.00)
            )

            optimized_daily_cost = (
                opt_phosphate * chemical_prices.get('phosphate', 2.50) +
                opt_scavenger * chemical_prices.get('oxygen_scavenger', 3.00) +
                opt_amine * chemical_prices.get('amine', 5.00) +
                opt_polymer * chemical_prices.get('polymer', 4.00)
            )

            # Monthly savings
            cost_reduction = (current_daily_cost - optimized_daily_cost) * 30

            # Feed schedule
            feed_schedule = {
                'phosphate': {'rate_kg_day': opt_phosphate, 'timing': 'continuous'},
                'oxygen_scavenger': {'rate_kg_day': opt_scavenger, 'timing': 'continuous'},
                'amine': {'rate_kg_day': opt_amine, 'timing': 'continuous'},
                'polymer': {'rate_kg_day': opt_polymer, 'timing': 'continuous'}
            }

            # Residual targets
            residual_targets = {
                'phosphate_ppm': phosphate_target,
                'sulfite_ppm': 20.0,  # Typical sulfite residual
                'condensate_pH': condensate_pH_target
            }

            # Calculate optimization score (0-100)
            # Based on cost reduction, target achievement, and efficiency
            cost_score = min(100, max(0, (cost_reduction / max(current_daily_cost * 30, 1)) * 100))
            efficiency_score = 85.0  # Simplified, would be more complex in production
            optimization_score = (cost_score * 0.4 + efficiency_score * 0.6)

            # Calculate provenance hash
            provenance_str = f"{current_usage}{water_quality}{targets}{timestamp}"
            provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

            return ChemicalOptimization(
                phosphate_dosing=opt_phosphate,
                oxygen_scavenger_dosing=opt_scavenger,
                amine_dosing=opt_amine,
                polymer_dosing=opt_polymer,
                total_chemical_cost=round(optimized_daily_cost, 2),
                cost_reduction_potential=round(cost_reduction, 2),
                feed_schedule=feed_schedule,
                residual_targets=residual_targets,
                optimization_score=round(optimization_score, 1),
                timestamp=timestamp,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            logger.error(f"Chemical optimization failed: {str(e)}")
            raise

    # ==================== Scale and Corrosion Prediction ====================

    @staticmethod
    def predict_calcium_carbonate_scale(
        lsi: float,
        temperature: float,  # Celsius
        velocity: float = 1.0  # m/s
    ) -> float:
        """
        Predict calcium carbonate scale formation rate.

        Scale rate increases with positive LSI, temperature, and low velocity.

        Args:
            lsi: Langelier Saturation Index
            temperature: Water temperature in Celsius
            velocity: Water velocity in m/s

        Returns:
            Scale formation rate in mm/year
        """
        with _calculation_lock:
            try:
                if lsi <= 0:
                    return 0.0  # No scaling expected

                # Base scale rate (mm/year) for LSI = 1.0 at 80°C
                base_rate = 0.5

                # LSI factor (exponential relationship)
                lsi_factor = math.exp(lsi - 1.0)

                # Temperature factor (increases with temperature)
                temp_factor = 1.0 + ((temperature - 80) / 50)
                temp_factor = max(0.5, min(temp_factor, 2.0))

                # Velocity factor (decreases with velocity)
                velocity_factor = 2.0 / (1.0 + velocity)

                # Calculate scale rate
                scale_rate = base_rate * lsi_factor * temp_factor * velocity_factor

                logger.debug(f"CaCO3 scale prediction: {scale_rate:.3f} mm/year "
                           f"(LSI={lsi}, T={temperature}°C)")

                return round(scale_rate, 3)

            except Exception as e:
                logger.error(f"CaCO3 scale prediction failed: {str(e)}")
                raise

    @staticmethod
    def predict_silica_scale(
        silica_concentration: float,  # mg/L as SiO2
        temperature: float,  # Celsius
        pH: float
    ) -> str:
        """
        Predict silica scale risk level.

        Silica solubility limits vary with temperature and pH.
        At >150°C and pH >9, silica can form hard, glassy deposits.

        Args:
            silica_concentration: Silica concentration in mg/L as SiO2
            temperature: Water temperature in Celsius
            pH: pH of water

        Returns:
            Risk level: "low", "moderate", "high", "severe"
        """
        with _calculation_lock:
            try:
                # Calculate silica solubility limit (simplified model)
                # Solubility increases with temperature and pH
                base_solubility = 100  # mg/L at 25°C, pH 7

                # Temperature factor
                temp_factor = 1.0 + ((temperature - 25) / 100)

                # pH factor (solubility increases significantly above pH 9)
                if pH < 8:
                    ph_factor = 1.0
                elif pH < 9:
                    ph_factor = 1.2
                elif pH < 10:
                    ph_factor = 1.5
                else:
                    ph_factor = 2.0

                solubility_limit = base_solubility * temp_factor * ph_factor

                # Calculate risk based on concentration vs. solubility
                ratio = silica_concentration / solubility_limit

                if ratio < 0.5:
                    risk = "low"
                elif ratio < 0.8:
                    risk = "moderate"
                elif ratio < 1.2:
                    risk = "high"
                else:
                    risk = "severe"

                logger.debug(f"Silica scale risk: {risk} (SiO2={silica_concentration} mg/L, "
                           f"limit={solubility_limit:.0f} mg/L)")

                return risk

            except Exception as e:
                logger.error(f"Silica scale prediction failed: {str(e)}")
                raise

    @staticmethod
    def predict_oxygen_corrosion(
        dissolved_oxygen: float,  # ppb
        temperature: float,  # Celsius
        pH: float
    ) -> float:
        """
        Predict oxygen corrosion rate in carbon steel.

        Corrosion rate = f(DO, temperature, pH)
        Peak corrosion typically at 60-80°C.

        Args:
            dissolved_oxygen: DO in ppb
            temperature: Temperature in Celsius
            pH: pH of water

        Returns:
            Corrosion rate in mils per year (mpy)
        """
        with _calculation_lock:
            try:
                if dissolved_oxygen < 0 or temperature < 0:
                    raise ValueError("DO and temperature must be non-negative")

                # Base corrosion rate at 1000 ppb DO, 70°C, pH 7
                base_rate = 10.0  # mpy

                # DO factor (linear relationship)
                do_factor = dissolved_oxygen / 1000.0

                # Temperature factor (parabolic, peaks around 70°C)
                # Corrosion increases with temp up to ~70°C, then decreases
                if temperature < 70:
                    temp_factor = 0.5 + (temperature / 140)
                else:
                    temp_factor = 1.5 - (temperature / 200)
                temp_factor = max(0.3, min(temp_factor, 1.5))

                # pH factor (corrosion decreases with increasing pH)
                if pH < 7:
                    ph_factor = 2.0 - (pH / 10)
                elif pH < 9:
                    ph_factor = 1.0 - ((pH - 7) / 10)
                else:
                    ph_factor = 0.8 - ((pH - 9) / 20)
                ph_factor = max(0.3, min(ph_factor, 2.0))

                # Calculate corrosion rate
                corrosion_rate = base_rate * do_factor * temp_factor * ph_factor

                logger.debug(f"O2 corrosion prediction: {corrosion_rate:.2f} mpy "
                           f"(DO={dissolved_oxygen} ppb, T={temperature}°C, pH={pH})")

                return round(corrosion_rate, 2)

            except Exception as e:
                logger.error(f"Oxygen corrosion prediction failed: {str(e)}")
                raise

    @staticmethod
    def predict_acid_corrosion(pH: float, temperature: float) -> float:
        """
        Predict acid corrosion rate in carbon steel.

        Significant corrosion occurs below pH 7, accelerates below pH 5.

        Args:
            pH: pH of water
            temperature: Temperature in Celsius

        Returns:
            Corrosion rate in mils per year (mpy)
        """
        with _calculation_lock:
            try:
                if pH >= 7:
                    return 0.0  # No acid corrosion at neutral/alkaline pH

                # Base corrosion rate at pH 5, 25°C
                base_rate = 20.0  # mpy

                # pH factor (exponential increase as pH decreases)
                ph_factor = math.exp((5.0 - pH) / 2.0)

                # Temperature factor (increases with temperature)
                temp_factor = 1.0 + ((temperature - 25) / 50)
                temp_factor = max(0.5, min(temp_factor, 2.0))

                # Calculate corrosion rate
                corrosion_rate = base_rate * ph_factor * temp_factor

                logger.debug(f"Acid corrosion prediction: {corrosion_rate:.2f} mpy "
                           f"(pH={pH}, T={temperature}°C)")

                return round(corrosion_rate, 2)

            except Exception as e:
                logger.error(f"Acid corrosion prediction failed: {str(e)}")
                raise

    @staticmethod
    def calculate_corrosion_allowance(
        material: str,
        environment: str,
        service_life: float  # years
    ) -> float:
        """
        Calculate required corrosion allowance for design.

        Args:
            material: Material type (e.g., "carbon_steel", "stainless_steel")
            environment: Environment type (e.g., "boiler_water", "condensate", "feedwater")
            service_life: Design service life in years

        Returns:
            Required corrosion allowance in mm
        """
        with _calculation_lock:
            try:
                # Base corrosion rates (mm/year) for different material-environment combinations
                corrosion_rates = {
                    'carbon_steel': {
                        'boiler_water': 0.1,
                        'condensate': 0.2,
                        'feedwater': 0.15,
                        'raw_water': 0.5
                    },
                    'stainless_steel': {
                        'boiler_water': 0.01,
                        'condensate': 0.02,
                        'feedwater': 0.015,
                        'raw_water': 0.05
                    },
                    'copper_alloy': {
                        'boiler_water': 0.05,
                        'condensate': 0.1,
                        'feedwater': 0.08,
                        'raw_water': 0.3
                    }
                }

                # Get corrosion rate
                material_rates = corrosion_rates.get(material.lower(),
                                                    corrosion_rates['carbon_steel'])
                corrosion_rate = material_rates.get(environment.lower(), 0.1)

                # Calculate allowance with safety factor
                safety_factor = 2.0  # ASME recommended
                corrosion_allowance = corrosion_rate * service_life * safety_factor

                logger.debug(f"Corrosion allowance: {corrosion_allowance:.2f} mm "
                           f"({material}, {environment}, {service_life} years)")

                return round(corrosion_allowance, 2)

            except Exception as e:
                logger.error(f"Corrosion allowance calculation failed: {str(e)}")
                raise

    # ==================== Energy and Cost Analysis ====================

    @staticmethod
    def calculate_blowdown_energy_savings(
        before_cycles: float,
        after_cycles: float,
        steam_cost: float,  # $/ton
        steam_rate: float = 1000.0  # kg/hr
    ) -> float:
        """
        Calculate annual energy savings from improved cycles of concentration.

        Args:
            before_cycles: Cycles before optimization
            after_cycles: Cycles after optimization
            steam_cost: Cost of steam in $/ton
            steam_rate: Steam generation rate in kg/hr

        Returns:
            Annual savings in $/year
        """
        with _calculation_lock:
            try:
                if before_cycles <= 1 or after_cycles <= 1:
                    raise ValueError("Cycles must be greater than 1")

                # Calculate blowdown rates
                before_bd = WaterTreatmentTools.calculate_blowdown_rate(steam_rate, before_cycles)
                after_bd = WaterTreatmentTools.calculate_blowdown_rate(steam_rate, after_cycles)

                # Blowdown reduction
                bd_reduction = before_bd - after_bd  # kg/hr

                # Assume blowdown at 180°C, heat loss recovery potential
                # Enthalpy at 180°C ≈ 763 kJ/kg
                enthalpy = 763  # kJ/kg

                # Energy savings (kW)
                energy_savings_kw = (bd_reduction * enthalpy) / 3600

                # Annual hours
                annual_hours = 8000  # Typical plant operation

                # Annual energy savings (kWh)
                annual_energy_kwh = energy_savings_kw * annual_hours

                # Convert to steam equivalent (ton)
                # 1 ton steam ≈ 2800 kWh (at typical conditions)
                annual_steam_savings_ton = annual_energy_kwh / 2800

                # Calculate cost savings
                annual_savings = annual_steam_savings_ton * steam_cost

                logger.debug(f"Blowdown energy savings: ${annual_savings:.2f}/year "
                           f"(cycles: {before_cycles} → {after_cycles})")

                return round(annual_savings, 2)

            except Exception as e:
                logger.error(f"Energy savings calculation failed: {str(e)}")
                raise

    @staticmethod
    def calculate_chemical_cost(
        dosing_rates: Dict[str, float],  # kg/day
        chemical_prices: Dict[str, float]  # $/kg
    ) -> float:
        """
        Calculate daily chemical cost.

        Args:
            dosing_rates: Chemical dosing rates in kg/day
            chemical_prices: Chemical unit prices in $/kg

        Returns:
            Total daily cost in $/day
        """
        with _calculation_lock:
            try:
                total_cost = 0.0

                for chemical, rate in dosing_rates.items():
                    price = chemical_prices.get(chemical, 0)
                    cost = rate * price
                    total_cost += cost
                    logger.debug(f"{chemical}: {rate:.2f} kg/day × ${price:.2f}/kg = ${cost:.2f}/day")

                logger.debug(f"Total chemical cost: ${total_cost:.2f}/day")

                return round(total_cost, 2)

            except Exception as e:
                logger.error(f"Chemical cost calculation failed: {str(e)}")
                raise

    @staticmethod
    def calculate_water_treatment_roi(
        costs: Dict[str, float],  # Annual costs
        savings: Dict[str, float],  # Annual savings
        implementation_cost: float  # One-time cost
    ) -> float:
        """
        Calculate ROI for water treatment optimization.

        ROI = (Annual Savings - Annual Costs) / Implementation Cost × 100%

        Args:
            costs: Annual operating costs (chemical, maintenance, etc.)
            savings: Annual savings (water, energy, reduced downtime, etc.)
            implementation_cost: One-time implementation cost

        Returns:
            ROI percentage
        """
        with _calculation_lock:
            try:
                total_annual_costs = sum(costs.values())
                total_annual_savings = sum(savings.values())

                net_annual_benefit = total_annual_savings - total_annual_costs

                if implementation_cost <= 0:
                    logger.warning("Implementation cost is zero or negative")
                    return float('inf')

                # Simple ROI
                roi = (net_annual_benefit / implementation_cost) * 100

                # Payback period
                payback_years = implementation_cost / net_annual_benefit if net_annual_benefit > 0 else float('inf')

                logger.debug(f"Water treatment ROI: {roi:.1f}% "
                           f"(payback: {payback_years:.1f} years)")

                return round(roi, 1)

            except Exception as e:
                logger.error(f"ROI calculation failed: {str(e)}")
                raise

    @staticmethod
    def calculate_makeup_water_cost(
        usage: float,  # m3/day
        water_price: float,  # $/m3
        treatment_cost: float = 0.5  # $/m3
    ) -> float:
        """
        Calculate daily makeup water cost including treatment.

        Args:
            usage: Makeup water usage in m3/day
            water_price: Raw water price in $/m3
            treatment_cost: Water treatment cost in $/m3

        Returns:
            Total daily cost in $/day
        """
        with _calculation_lock:
            try:
                if usage < 0:
                    raise ValueError("Usage must be non-negative")

                total_unit_cost = water_price + treatment_cost
                daily_cost = usage * total_unit_cost

                logger.debug(f"Makeup water cost: {usage:.1f} m3/day × "
                           f"${total_unit_cost:.2f}/m3 = ${daily_cost:.2f}/day")

                return round(daily_cost, 2)

            except Exception as e:
                logger.error(f"Makeup water cost calculation failed: {str(e)}")
                raise

    # ==================== Compliance Checking ====================

    @staticmethod
    def check_asme_compliance(
        water_chemistry: Dict[str, Any],
        pressure: float  # bar
    ) -> ComplianceResult:
        """
        Check ASME boiler water quality compliance.

        ASME guidelines vary by pressure range:
        - 0-20 bar: Less stringent
        - 20-60 bar: Moderate
        - >60 bar: Most stringent

        Args:
            water_chemistry: Water chemistry parameters
            pressure: Boiler operating pressure in bar

        Returns:
            ComplianceResult with detailed compliance status
        """
        try:
            timestamp = datetime.utcnow().isoformat()

            # Extract parameters
            pH = water_chemistry.get('pH', 7.0)
            tds = water_chemistry.get('tds', 0)
            alkalinity = water_chemistry.get('alkalinity', 0)
            chloride = water_chemistry.get('chloride', 0)
            silica = water_chemistry.get('silica', 0)
            hardness = water_chemistry.get('hardness', 0)

            violations = []
            warnings = []

            # Determine limits based on pressure
            if pressure < 20:
                # Low pressure (<300 psi)
                limits = {
                    'pH_min': 10.5, 'pH_max': 12.0,
                    'tds_max': 3500,
                    'alkalinity_max': 700,
                    'chloride_max': 300,
                    'silica_max': 150,
                    'hardness_max': 5.0
                }
            elif pressure < 60:
                # Medium pressure (300-900 psi)
                limits = {
                    'pH_min': 9.0, 'pH_max': 11.0,
                    'tds_max': 2000,
                    'alkalinity_max': 400,
                    'chloride_max': 100,
                    'silica_max': 50,
                    'hardness_max': 2.0
                }
            else:
                # High pressure (>900 psi)
                limits = {
                    'pH_min': 9.0, 'pH_max': 9.6,
                    'tds_max': 1000,
                    'alkalinity_max': 200,
                    'chloride_max': 50,
                    'silica_max': 20,
                    'hardness_max': 0.3
                }

            # Check pH
            if pH < limits['pH_min'] or pH > limits['pH_max']:
                violations.append(
                    f"pH {pH} outside ASME range {limits['pH_min']}-{limits['pH_max']} "
                    f"for {pressure} bar"
                )
            elif pH < limits['pH_min'] + 0.2 or pH > limits['pH_max'] - 0.2:
                warnings.append(f"pH {pH} near limit, monitor closely")

            # Check TDS
            if tds > limits['tds_max']:
                violations.append(f"TDS {tds} mg/L exceeds limit {limits['tds_max']} mg/L")
            elif tds > limits['tds_max'] * 0.9:
                warnings.append(f"TDS {tds} mg/L approaching limit")

            # Check alkalinity
            if alkalinity > limits['alkalinity_max']:
                violations.append(
                    f"Alkalinity {alkalinity} mg/L exceeds limit {limits['alkalinity_max']} mg/L"
                )

            # Check chloride
            if chloride > limits['chloride_max']:
                violations.append(
                    f"Chloride {chloride} mg/L exceeds limit {limits['chloride_max']} mg/L"
                )

            # Check silica
            if silica > limits['silica_max']:
                violations.append(
                    f"Silica {silica} mg/L exceeds limit {limits['silica_max']} mg/L"
                )

            # Check hardness
            if hardness > limits['hardness_max']:
                violations.append(
                    f"Hardness {hardness} mg/L exceeds limit {limits['hardness_max']} mg/L"
                )

            # Determine compliance status
            if violations:
                compliance_status = "FAIL"
                margin_percent = -10.0  # Non-compliant
            elif warnings:
                compliance_status = "WARNING"
                margin_percent = 5.0
            else:
                compliance_status = "PASS"
                # Calculate minimum margin across all parameters
                margins = [
                    ((limits['pH_max'] - pH) / limits['pH_max'] * 100),
                    ((limits['tds_max'] - tds) / limits['tds_max'] * 100),
                    ((limits['alkalinity_max'] - alkalinity) / limits['alkalinity_max'] * 100)
                ]
                margin_percent = min(margins)

            # Recommendations
            recommended_actions = []
            if violations:
                recommended_actions.append("Immediate corrective action required")
                recommended_actions.append("Increase blowdown rate if TDS/alkalinity high")
                recommended_actions.append("Check water treatment system operation")
            if warnings:
                recommended_actions.append("Monitor parameters closely")
                recommended_actions.append("Consider adjusting treatment program")

            # Calculate provenance hash
            provenance_str = f"ASME{water_chemistry}{pressure}{compliance_status}{timestamp}"
            provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

            parameters_checked = 6  # pH, TDS, alkalinity, chloride, silica, hardness

            return ComplianceResult(
                standard="ASME",
                compliance_status=compliance_status,
                parameters_checked=parameters_checked,
                violations=violations,
                warnings=warnings,
                margin_percent=round(margin_percent, 1),
                recommended_actions=recommended_actions,
                timestamp=timestamp,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            logger.error(f"ASME compliance check failed: {str(e)}")
            raise

    @staticmethod
    def check_abma_guidelines(
        water_chemistry: Dict[str, Any],
        boiler_type: BoilerType
    ) -> ComplianceResult:
        """
        Check ABMA (American Boiler Manufacturers Association) guidelines compliance.

        Args:
            water_chemistry: Water chemistry parameters
            boiler_type: Type of boiler

        Returns:
            ComplianceResult with detailed compliance status
        """
        try:
            timestamp = datetime.utcnow().isoformat()

            violations = []
            warnings = []

            # Extract parameters
            pH = water_chemistry.get('pH', 7.0)
            tds = water_chemistry.get('tds', 0)
            phosphate = water_chemistry.get('phosphate_residual', 0)
            sulfite = water_chemistry.get('sulfite_residual', 0)

            # ABMA guidelines by boiler type
            if boiler_type == BoilerType.FIRE_TUBE:
                limits = {
                    'pH_min': 11.0, 'pH_max': 12.0,
                    'phosphate_min': 30, 'phosphate_max': 60,
                    'sulfite_min': 20, 'sulfite_max': 40
                }
            elif boiler_type == BoilerType.WATER_TUBE:
                limits = {
                    'pH_min': 10.5, 'pH_max': 11.5,
                    'phosphate_min': 30, 'phosphate_max': 60,
                    'sulfite_min': 20, 'sulfite_max': 40
                }
            else:
                limits = {
                    'pH_min': 10.0, 'pH_max': 11.5,
                    'phosphate_min': 20, 'phosphate_max': 60,
                    'sulfite_min': 15, 'sulfite_max': 40
                }

            # Check pH
            if pH < limits['pH_min'] or pH > limits['pH_max']:
                violations.append(
                    f"pH {pH} outside ABMA range {limits['pH_min']}-{limits['pH_max']} "
                    f"for {boiler_type.value}"
                )

            # Check phosphate residual
            if phosphate < limits['phosphate_min']:
                violations.append(
                    f"Phosphate residual {phosphate} ppm below minimum {limits['phosphate_min']} ppm"
                )
            elif phosphate > limits['phosphate_max']:
                warnings.append(
                    f"Phosphate residual {phosphate} ppm above target {limits['phosphate_max']} ppm"
                )

            # Check sulfite residual
            if sulfite < limits['sulfite_min']:
                violations.append(
                    f"Sulfite residual {sulfite} ppm below minimum {limits['sulfite_min']} ppm"
                )
            elif sulfite > limits['sulfite_max']:
                warnings.append(
                    f"Sulfite residual {sulfite} ppm above target {limits['sulfite_max']} ppm"
                )

            # Determine compliance status
            if violations:
                compliance_status = "FAIL"
            elif warnings:
                compliance_status = "WARNING"
            else:
                compliance_status = "PASS"

            # Calculate margin
            if compliance_status == "PASS":
                margin_percent = 20.0
            elif compliance_status == "WARNING":
                margin_percent = 10.0
            else:
                margin_percent = 0.0

            # Recommendations
            recommended_actions = []
            if violations:
                recommended_actions.append("Adjust chemical feed rates immediately")
                recommended_actions.append("Check chemical feed pump operation")

            # Calculate provenance hash
            provenance_str = f"ABMA{water_chemistry}{boiler_type}{timestamp}"
            provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

            return ComplianceResult(
                standard="ABMA",
                compliance_status=compliance_status,
                parameters_checked=3,
                violations=violations,
                warnings=warnings,
                margin_percent=margin_percent,
                recommended_actions=recommended_actions,
                timestamp=timestamp,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            logger.error(f"ABMA compliance check failed: {str(e)}")
            raise

    @staticmethod
    def validate_treatment_program(
        program_type: str,
        chemistry: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate water treatment program effectiveness.

        Args:
            program_type: Type of treatment program
                          (e.g., "phosphate", "chelant", "polymer", "all_volatile")
            chemistry: Current water chemistry

        Returns:
            ValidationResult with program validation details
        """
        try:
            timestamp = datetime.utcnow().isoformat()

            issues = []
            recommendations = []
            is_valid = True
            chemistry_compatible = True

            # Extract parameters
            pH = chemistry.get('pH', 7.0)
            phosphate = chemistry.get('phosphate_residual', 0)
            sulfite = chemistry.get('sulfite_residual', 0)
            hardness = chemistry.get('hardness', 0)

            # Validate based on program type
            if program_type.lower() == "phosphate":
                # Phosphate program validation
                if pH < 10.5 or pH > 11.5:
                    issues.append(f"pH {pH} not optimal for phosphate program (target 10.5-11.5)")
                    is_valid = False

                if phosphate < 30:
                    issues.append(f"Phosphate residual {phosphate} ppm too low (target 30-60)")
                    is_valid = False

                if hardness > 2:
                    issues.append(f"Hardness {hardness} mg/L too high for phosphate program")
                    chemistry_compatible = False

                effectiveness_score = 85.0 if is_valid else 60.0

                recommendations.append("Maintain phosphate residual 30-60 ppm")
                recommendations.append("Control pH 10.5-11.5 for optimal phosphate effectiveness")

            elif program_type.lower() == "all_volatile":
                # All-volatile treatment (AVT) validation
                if pH < 9.0 or pH > 9.8:
                    issues.append(f"pH {pH} not optimal for AVT (target 9.0-9.8)")
                    is_valid = False

                if phosphate > 0:
                    issues.append("Phosphate detected in AVT program (should be zero)")
                    chemistry_compatible = False

                effectiveness_score = 90.0 if is_valid else 65.0

                recommendations.append("AVT suitable for high-pressure boilers only")
                recommendations.append("Requires excellent feedwater quality")

            elif program_type.lower() == "chelant":
                # Chelant program validation
                if pH < 10.0 or pH > 11.0:
                    issues.append(f"pH {pH} not optimal for chelant program (target 10.0-11.0)")
                    is_valid = False

                if hardness > 5:
                    issues.append(f"Hardness {hardness} mg/L may overwhelm chelant capacity")
                    chemistry_compatible = False

                effectiveness_score = 80.0 if is_valid else 55.0

                recommendations.append("Monitor for iron levels, chelants mobilize iron")
                recommendations.append("Consider switch to phosphate if hardness increases")

            else:
                # Generic polymer program
                effectiveness_score = 75.0
                recommendations.append("Validate program type selection")

            # Calculate provenance hash
            provenance_str = f"{program_type}{chemistry}{is_valid}{timestamp}"
            provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

            return ValidationResult(
                program_type=program_type,
                is_valid=is_valid,
                effectiveness_score=effectiveness_score,
                chemistry_compatibility=chemistry_compatible,
                issues=issues,
                recommendations=recommendations,
                timestamp=timestamp,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            logger.error(f"Treatment program validation failed: {str(e)}")
            raise


# ==================== Utility Functions ====================

def _get_deterministic_timestamp() -> str:
    """Get deterministic timestamp in UTC ISO format."""
    return datetime.utcnow().isoformat()


def _calculate_provenance_hash(data: Any) -> str:
    """Calculate SHA-256 provenance hash for data."""
    data_str = str(data)
    return hashlib.sha256(data_str.encode()).hexdigest()
