"""
Heat Recovery Tools - Tool schemas and registry for GL-006 HeatRecoveryMaximizer

This module defines all tool schemas for heat recovery optimization operations
including pinch analysis, heat exchanger design, exergy calculations, and
economic optimization. All tools follow zero-hallucination principles.

Example:
    >>> from tools import TOOL_REGISTRY
    >>> pinch_tool = TOOL_REGISTRY["perform_pinch_analysis"]
    >>> result = await pinch_tool.execute(stream_data)
"""

from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from enum import Enum
import json
import hashlib
from pydantic import BaseModel, Field, validator, root_validator
import numpy as np
import pandas as pd

# GreenLang imports
from greenlang_tools import ToolSchema, ToolRegistry, ToolResult
from greenlang_validation import ValidationError


class ToolCategory(str, Enum):
    """Categories of heat recovery tools."""
    ANALYSIS = "analysis"
    DESIGN = "design"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    REPORTING = "reporting"


class ToolPriority(str, Enum):
    """Execution priority for tools."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


# Base tool schema
class HeatRecoveryTool(BaseModel):
    """Base schema for heat recovery tools."""

    name: str = Field(..., description="Tool name")
    category: ToolCategory = Field(..., description="Tool category")
    priority: ToolPriority = Field(ToolPriority.NORMAL, description="Execution priority")
    version: str = Field("1.0.0", description="Tool version")
    zero_hallucination: bool = Field(True, description="Uses only deterministic calculations")
    requires_validation: bool = Field(True, description="Output requires validation")
    timeout_seconds: int = Field(300, ge=1, le=3600, description="Maximum execution time")
    retry_count: int = Field(3, ge=0, le=10, description="Number of retries on failure")

    class Config:
        use_enum_values = True


# Tool 1: Stream Analysis Tool
class StreamAnalysisTool(HeatRecoveryTool):
    """Tool for analyzing process streams and identifying waste heat."""

    name: str = Field(default="stream_analysis", const=True)
    category: ToolCategory = Field(default=ToolCategory.ANALYSIS, const=True)

    class InputSchema(BaseModel):
        streams: List[Dict[str, Any]] = Field(..., description="Process stream data")
        min_recoverable_temp_c: float = Field(60.0, ge=0, le=1000, description="Minimum temperature for recovery")
        min_recoverable_duty_kw: float = Field(10.0, ge=0, description="Minimum heat duty for recovery")

        @validator('streams')
        def validate_streams(cls, v):
            if not v:
                raise ValueError("At least one stream required")
            return v

    class OutputSchema(BaseModel):
        waste_heat_streams: List[Dict[str, Any]] = Field(..., description="Identified waste heat streams")
        total_waste_heat_kw: float = Field(..., ge=0, description="Total waste heat available")
        temperature_distribution: Dict[str, int] = Field(..., description="Temperature range distribution")
        recovery_potential_by_grade: Dict[str, float] = Field(..., description="Recovery potential by temperature grade")
        provenance_hash: str = Field(..., description="SHA-256 hash for traceability")

    async def execute(self, input_data: InputSchema) -> OutputSchema:
        """Execute stream analysis to identify waste heat opportunities."""
        waste_heat_streams = []
        total_waste_heat = 0
        temp_distribution = {"<100C": 0, "100-200C": 0, "200-400C": 0, ">400C": 0}
        grade_potential = {"low": 0, "medium": 0, "high": 0}

        for stream in input_data.streams:
            temp = stream.get("inlet_temp_c", 0)
            duty = stream.get("heat_duty_kw", 0)

            # Check if stream qualifies as waste heat
            if temp >= input_data.min_recoverable_temp_c and duty >= input_data.min_recoverable_duty_kw:
                stream["is_recoverable"] = True
                stream["grade"] = self._classify_heat_grade(temp)
                waste_heat_streams.append(stream)
                total_waste_heat += duty

                # Update distributions
                if temp < 100:
                    temp_distribution["<100C"] += 1
                elif temp < 200:
                    temp_distribution["100-200C"] += 1
                elif temp < 400:
                    temp_distribution["200-400C"] += 1
                else:
                    temp_distribution[">400C"] += 1

                grade_potential[stream["grade"]] += duty

        # Generate provenance hash
        provenance_data = json.dumps({
            "input": input_data.dict(),
            "waste_heat_count": len(waste_heat_streams),
            "total_heat": total_waste_heat
        }, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        return self.OutputSchema(
            waste_heat_streams=waste_heat_streams,
            total_waste_heat_kw=total_waste_heat,
            temperature_distribution=temp_distribution,
            recovery_potential_by_grade=grade_potential,
            provenance_hash=provenance_hash
        )

    def _classify_heat_grade(self, temp_c: float) -> str:
        """Classify heat grade based on temperature."""
        if temp_c < 150:
            return "low"
        elif temp_c < 400:
            return "medium"
        else:
            return "high"


# Tool 2: Pinch Analysis Tool
class PinchAnalysisTool(HeatRecoveryTool):
    """Tool for performing pinch analysis on process streams."""

    name: str = Field(default="pinch_analysis", const=True)
    category: ToolCategory = Field(default=ToolCategory.ANALYSIS, const=True)
    priority: ToolPriority = Field(default=ToolPriority.CRITICAL, const=True)

    class InputSchema(BaseModel):
        hot_streams: List[Dict[str, float]] = Field(..., description="Hot process streams")
        cold_streams: List[Dict[str, float]] = Field(..., description="Cold process streams")
        min_temp_approach_c: float = Field(10.0, ge=1, le=50, description="Minimum temperature approach")
        hot_utility_temp_c: float = Field(500.0, ge=100, description="Hot utility temperature")
        cold_utility_temp_c: float = Field(20.0, le=100, description="Cold utility temperature")

    class OutputSchema(BaseModel):
        pinch_temp_hot_c: float = Field(..., description="Pinch temperature (hot side)")
        pinch_temp_cold_c: float = Field(..., description="Pinch temperature (cold side)")
        min_hot_utility_kw: float = Field(..., ge=0, description="Minimum hot utility")
        min_cold_utility_kw: float = Field(..., ge=0, description="Minimum cold utility")
        heat_recovery_kw: float = Field(..., ge=0, description="Maximum heat recovery")
        problem_table: List[Dict[str, float]] = Field(..., description="Problem table cascade")
        composite_curves: Dict[str, List[tuple]] = Field(..., description="Composite curve data")
        provenance_hash: str = Field(..., description="SHA-256 hash")

    async def execute(self, input_data: InputSchema) -> OutputSchema:
        """Execute pinch analysis using problem table algorithm."""
        # Build temperature intervals
        all_temps = []
        for stream in input_data.hot_streams:
            all_temps.extend([stream["inlet_temp_c"], stream["outlet_temp_c"]])
        for stream in input_data.cold_streams:
            all_temps.extend([
                stream["inlet_temp_c"] + input_data.min_temp_approach_c,
                stream["outlet_temp_c"] + input_data.min_temp_approach_c
            ])

        all_temps = sorted(set(all_temps), reverse=True)

        # Build problem table
        problem_table = []
        cumulative_heat = 0
        min_cumulative = float('inf')
        pinch_interval = 0

        for i in range(len(all_temps) - 1):
            t_high = all_temps[i]
            t_low = all_temps[i + 1]
            delta_t = t_high - t_low

            # Calculate net heat capacity flow rate
            cp_hot = sum(
                s["flow_rate_kg_s"] * s["specific_heat_kj_kg_k"]
                for s in input_data.hot_streams
                if s["outlet_temp_c"] <= t_high <= s["inlet_temp_c"]
            )

            cp_cold = sum(
                s["flow_rate_kg_s"] * s["specific_heat_kj_kg_k"]
                for s in input_data.cold_streams
                if s["inlet_temp_c"] <= t_high - input_data.min_temp_approach_c <= s["outlet_temp_c"]
            )

            net_cp = cp_cold - cp_hot
            interval_heat = net_cp * delta_t
            cumulative_heat += interval_heat

            if cumulative_heat < min_cumulative:
                min_cumulative = cumulative_heat
                pinch_interval = i

            problem_table.append({
                "t_high_c": t_high,
                "t_low_c": t_low,
                "delta_t_c": delta_t,
                "cp_hot_kw_k": cp_hot,
                "cp_cold_kw_k": cp_cold,
                "net_cp_kw_k": net_cp,
                "interval_heat_kw": interval_heat,
                "cumulative_heat_kw": cumulative_heat
            })

        # Calculate utilities
        min_hot_utility = abs(min_cumulative) if min_cumulative < 0 else 0
        min_cold_utility = cumulative_heat + min_hot_utility

        # Determine pinch temperature
        pinch_temp_hot = all_temps[pinch_interval]
        pinch_temp_cold = pinch_temp_hot - input_data.min_temp_approach_c

        # Calculate maximum heat recovery
        total_hot_duty = sum(
            s["flow_rate_kg_s"] * s["specific_heat_kj_kg_k"] *
            abs(s["inlet_temp_c"] - s["outlet_temp_c"])
            for s in input_data.hot_streams
        )
        heat_recovery = total_hot_duty - min_hot_utility

        # Generate composite curves data
        composite_curves = self._generate_composite_curves(
            input_data.hot_streams,
            input_data.cold_streams,
            input_data.min_temp_approach_c
        )

        # Provenance hash
        provenance_data = json.dumps({
            "streams": len(input_data.hot_streams) + len(input_data.cold_streams),
            "pinch": pinch_temp_hot,
            "recovery": heat_recovery
        }, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        return self.OutputSchema(
            pinch_temp_hot_c=pinch_temp_hot,
            pinch_temp_cold_c=pinch_temp_cold,
            min_hot_utility_kw=min_hot_utility,
            min_cold_utility_kw=min_cold_utility,
            heat_recovery_kw=heat_recovery,
            problem_table=problem_table,
            composite_curves=composite_curves,
            provenance_hash=provenance_hash
        )

    def _generate_composite_curves(self, hot_streams, cold_streams, min_approach):
        """Generate composite curve data points."""
        # Simplified composite curve generation
        hot_curve = []
        cold_curve = []

        # Hot composite curve
        cumulative_duty = 0
        for temp in sorted(set([s["inlet_temp_c"] for s in hot_streams] +
                              [s["outlet_temp_c"] for s in hot_streams]), reverse=True):
            duty_at_temp = sum(
                s["flow_rate_kg_s"] * s["specific_heat_kj_kg_k"] *
                max(0, min(s["inlet_temp_c"], temp) - s["outlet_temp_c"])
                for s in hot_streams
            )
            hot_curve.append((duty_at_temp, temp))

        # Cold composite curve (shifted by min approach)
        for temp in sorted(set([s["inlet_temp_c"] for s in cold_streams] +
                              [s["outlet_temp_c"] for s in cold_streams])):
            duty_at_temp = sum(
                s["flow_rate_kg_s"] * s["specific_heat_kj_kg_k"] *
                max(0, s["outlet_temp_c"] - max(s["inlet_temp_c"], temp))
                for s in cold_streams
            )
            cold_curve.append((duty_at_temp, temp))

        return {"hot_composite": hot_curve, "cold_composite": cold_curve}


# Tool 3: Heat Exchanger Design Tool
class HeatExchangerDesignTool(HeatRecoveryTool):
    """Tool for designing heat exchangers based on thermal requirements."""

    name: str = Field(default="heat_exchanger_design", const=True)
    category: ToolCategory = Field(default=ToolCategory.DESIGN, const=True)

    class InputSchema(BaseModel):
        hot_stream: Dict[str, float] = Field(..., description="Hot stream properties")
        cold_stream: Dict[str, float] = Field(..., description="Cold stream properties")
        exchanger_type: str = Field("shell_and_tube", description="Type of heat exchanger")
        fouling_resistance: float = Field(0.0002, ge=0, description="Fouling resistance m2.K/W")
        max_pressure_drop_bar: float = Field(0.5, ge=0.1, le=2, description="Max allowable pressure drop")
        material: str = Field("carbon_steel", description="Construction material")

    class OutputSchema(BaseModel):
        exchanger_id: str = Field(..., description="Unique exchanger ID")
        duty_kw: float = Field(..., ge=0, description="Heat duty")
        area_m2: float = Field(..., ge=0, description="Required heat transfer area")
        overall_htc_w_m2_k: float = Field(..., ge=0, description="Overall heat transfer coefficient")
        lmtd_c: float = Field(..., ge=0, description="Log mean temperature difference")
        effectiveness: float = Field(..., ge=0, le=1, description="Heat exchanger effectiveness")
        ntu: float = Field(..., ge=0, description="Number of transfer units")
        pressure_drop_hot_bar: float = Field(..., ge=0, description="Hot side pressure drop")
        pressure_drop_cold_bar: float = Field(..., ge=0, description="Cold side pressure drop")
        capital_cost_usd: float = Field(..., ge=0, description="Estimated capital cost")
        design_parameters: Dict[str, Any] = Field(..., description="Detailed design parameters")
        provenance_hash: str = Field(..., description="SHA-256 hash")

    async def execute(self, input_data: InputSchema) -> OutputSchema:
        """Design heat exchanger based on stream requirements."""
        import uuid

        exchanger_id = f"HX-{uuid.uuid4().hex[:8]}"

        # Calculate heat duty (limited by stream with lower capacity)
        q_hot = (input_data.hot_stream["flow_rate_kg_s"] *
                 input_data.hot_stream["specific_heat_kj_kg_k"] *
                 (input_data.hot_stream["inlet_temp_c"] - input_data.hot_stream["outlet_temp_c"]))

        q_cold = (input_data.cold_stream["flow_rate_kg_s"] *
                  input_data.cold_stream["specific_heat_kj_kg_k"] *
                  (input_data.cold_stream["outlet_temp_c"] - input_data.cold_stream["inlet_temp_c"]))

        duty_kw = min(abs(q_hot), abs(q_cold))

        # Calculate LMTD
        dt1 = input_data.hot_stream["inlet_temp_c"] - input_data.cold_stream["outlet_temp_c"]
        dt2 = input_data.hot_stream["outlet_temp_c"] - input_data.cold_stream["inlet_temp_c"]

        if dt1 <= 0 or dt2 <= 0:
            raise ValueError("Temperature cross detected - heat exchange not feasible")

        if abs(dt1 - dt2) < 0.1:
            lmtd = dt1
        else:
            lmtd = (dt1 - dt2) / np.log(dt1 / dt2)

        # Calculate heat transfer coefficients (simplified correlations)
        h_hot = self._calculate_htc(input_data.hot_stream, "hot")
        h_cold = self._calculate_htc(input_data.cold_stream, "cold")

        # Overall heat transfer coefficient
        u_clean = 1 / (1/h_hot + 1/h_cold)
        u_fouled = 1 / (1/u_clean + input_data.fouling_resistance)

        # Required area
        area = duty_kw * 1000 / (u_fouled * lmtd)

        # Calculate effectiveness
        c_hot = (input_data.hot_stream["flow_rate_kg_s"] *
                 input_data.hot_stream["specific_heat_kj_kg_k"])
        c_cold = (input_data.cold_stream["flow_rate_kg_s"] *
                  input_data.cold_stream["specific_heat_kj_kg_k"])
        c_min = min(c_hot, c_cold)
        c_max = max(c_hot, c_cold)
        c_ratio = c_min / c_max

        ntu = u_fouled * area / (c_min * 1000)
        effectiveness = (1 - np.exp(-ntu * (1 - c_ratio))) / (1 - c_ratio * np.exp(-ntu * (1 - c_ratio)))

        # Pressure drop calculation (simplified)
        pressure_drop_hot = self._calculate_pressure_drop(
            input_data.hot_stream,
            area,
            input_data.exchanger_type
        )
        pressure_drop_cold = self._calculate_pressure_drop(
            input_data.cold_stream,
            area,
            input_data.exchanger_type
        )

        # Capital cost estimation (simplified correlation)
        capital_cost = self._estimate_capital_cost(
            area,
            input_data.exchanger_type,
            input_data.material
        )

        # Design parameters
        design_params = {
            "exchanger_type": input_data.exchanger_type,
            "material": input_data.material,
            "tube_passes": 2 if input_data.exchanger_type == "shell_and_tube" else 1,
            "shell_passes": 1,
            "baffle_spacing_m": 0.3,
            "tube_diameter_m": 0.025,
            "tube_pitch_m": 0.032,
            "design_pressure_bar": max(
                input_data.hot_stream.get("pressure_bar", 1),
                input_data.cold_stream.get("pressure_bar", 1)
            ) * 1.5
        }

        # Provenance
        provenance_data = json.dumps({
            "exchanger_id": exchanger_id,
            "duty": duty_kw,
            "area": area
        }, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        return self.OutputSchema(
            exchanger_id=exchanger_id,
            duty_kw=duty_kw,
            area_m2=area,
            overall_htc_w_m2_k=u_fouled,
            lmtd_c=lmtd,
            effectiveness=effectiveness,
            ntu=ntu,
            pressure_drop_hot_bar=pressure_drop_hot,
            pressure_drop_cold_bar=pressure_drop_cold,
            capital_cost_usd=capital_cost,
            design_parameters=design_params,
            provenance_hash=provenance_hash
        )

    def _calculate_htc(self, stream: Dict, side: str) -> float:
        """Calculate heat transfer coefficient (W/m2.K)."""
        # Simplified correlation - would use detailed correlations in production
        base_htc = 1000 if stream.get("phase", "liquid") == "liquid" else 500

        # Adjust for flow rate
        flow_factor = (stream["flow_rate_kg_s"] / 10) ** 0.8

        return base_htc * flow_factor

    def _calculate_pressure_drop(self, stream: Dict, area: float, exchanger_type: str) -> float:
        """Calculate pressure drop (bar)."""
        # Simplified pressure drop calculation
        velocity = stream["flow_rate_kg_s"] / (1000 * 0.01)  # Assumed cross-section
        friction_factor = 0.02  # Simplified

        # Length based on area
        length = area / 3  # Assumed width of 3m

        pressure_drop_pa = friction_factor * length * 1000 * velocity**2 / 2
        return pressure_drop_pa / 100000  # Convert to bar

    def _estimate_capital_cost(self, area: float, exchanger_type: str, material: str) -> float:
        """Estimate capital cost based on area and type."""
        # Base cost per m2
        base_costs = {
            "shell_and_tube": 500,
            "plate": 300,
            "finned_tube": 600,
            "regenerative": 800,
            "heat_pipe": 1000
        }

        # Material factors
        material_factors = {
            "carbon_steel": 1.0,
            "stainless_steel": 2.5,
            "titanium": 5.0,
            "hastelloy": 8.0
        }

        base_cost = base_costs.get(exchanger_type, 500)
        material_factor = material_factors.get(material, 1.0)

        # Cost correlation: Cost = base * area^0.65 * material_factor
        return base_cost * (area ** 0.65) * material_factor


# Tool 4: Exergy Analysis Tool
class ExergyAnalysisTool(HeatRecoveryTool):
    """Tool for performing exergy analysis on heat recovery systems."""

    name: str = Field(default="exergy_analysis", const=True)
    category: ToolCategory = Field(default=ToolCategory.ANALYSIS, const=True)

    class InputSchema(BaseModel):
        streams: List[Dict[str, Any]] = Field(..., description="Process streams")
        ambient_temp_c: float = Field(25.0, ge=-50, le=50, description="Ambient temperature")
        ambient_pressure_bar: float = Field(1.013, ge=0.5, le=2, description="Ambient pressure")
        system_boundary: str = Field("process", description="System boundary definition")

    class OutputSchema(BaseModel):
        total_exergy_input_kw: float = Field(..., ge=0, description="Total exergy input")
        total_exergy_output_kw: float = Field(..., ge=0, description="Total exergy output")
        exergy_destroyed_kw: float = Field(..., ge=0, description="Exergy destruction")
        exergy_efficiency_percent: float = Field(..., ge=0, le=100, description="Exergy efficiency")
        exergy_by_stream: List[Dict[str, float]] = Field(..., description="Exergy for each stream")
        destruction_by_component: Dict[str, float] = Field(..., description="Exergy destruction breakdown")
        improvement_potential_kw: float = Field(..., ge=0, description="Theoretical improvement potential")
        provenance_hash: str = Field(..., description="SHA-256 hash")

    async def execute(self, input_data: InputSchema) -> OutputSchema:
        """Perform exergy analysis on the system."""
        t_ambient_k = input_data.ambient_temp_c + 273.15
        p_ambient = input_data.ambient_pressure_bar

        total_exergy_input = 0
        total_exergy_output = 0
        exergy_by_stream = []

        for stream in input_data.streams:
            # Physical exergy
            t_stream_k = stream.get("temperature_c", 25) + 273.15
            p_stream = stream.get("pressure_bar", 1)
            flow_rate = stream.get("flow_rate_kg_s", 0)
            cp = stream.get("specific_heat_kj_kg_k", 4.186)

            # Physical exergy calculation
            exergy_thermal = cp * flow_rate * (
                (t_stream_k - t_ambient_k) - t_ambient_k * np.log(t_stream_k / t_ambient_k)
            )

            # Pressure contribution (simplified for ideal gas)
            r_specific = 0.287  # kJ/kg.K
            exergy_pressure = r_specific * t_ambient_k * flow_rate * np.log(p_stream / p_ambient)

            total_exergy = exergy_thermal + exergy_pressure

            stream_exergy = {
                "stream_id": stream.get("stream_id", "unknown"),
                "physical_exergy_kw": exergy_thermal,
                "pressure_exergy_kw": exergy_pressure,
                "total_exergy_kw": total_exergy,
                "exergy_factor": total_exergy / (flow_rate * cp * abs(t_stream_k - t_ambient_k)) if flow_rate > 0 else 0
            }
            exergy_by_stream.append(stream_exergy)

            # Categorize as input or output
            if stream.get("is_input", True):
                total_exergy_input += total_exergy
            else:
                total_exergy_output += abs(total_exergy)

        # Calculate exergy destruction
        exergy_destroyed = total_exergy_input - total_exergy_output

        # Exergy efficiency
        exergy_efficiency = (total_exergy_output / total_exergy_input * 100) if total_exergy_input > 0 else 0

        # Component-wise destruction (simplified)
        destruction_breakdown = {
            "heat_transfer": exergy_destroyed * 0.4,
            "pressure_drop": exergy_destroyed * 0.2,
            "mixing": exergy_destroyed * 0.2,
            "chemical_reaction": exergy_destroyed * 0.1,
            "other": exergy_destroyed * 0.1
        }

        # Theoretical improvement potential (Carnot limit)
        carnot_efficiency = 1 - t_ambient_k / max(s.get("temperature_c", 25) + 273.15 for s in input_data.streams)
        theoretical_output = total_exergy_input * carnot_efficiency
        improvement_potential = theoretical_output - total_exergy_output

        # Provenance
        provenance_data = json.dumps({
            "streams": len(input_data.streams),
            "efficiency": exergy_efficiency,
            "destroyed": exergy_destroyed
        }, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        return self.OutputSchema(
            total_exergy_input_kw=total_exergy_input,
            total_exergy_output_kw=total_exergy_output,
            exergy_destroyed_kw=exergy_destroyed,
            exergy_efficiency_percent=exergy_efficiency,
            exergy_by_stream=exergy_by_stream,
            destruction_by_component=destruction_breakdown,
            improvement_potential_kw=improvement_potential,
            provenance_hash=provenance_hash
        )


# Tool 5: Network Optimization Tool
class NetworkOptimizationTool(HeatRecoveryTool):
    """Tool for optimizing heat exchanger network configuration."""

    name: str = Field(default="network_optimization", const=True)
    category: ToolCategory = Field(default=ToolCategory.OPTIMIZATION, const=True)
    priority: ToolPriority = Field(default=ToolPriority.HIGH, const=True)

    class InputSchema(BaseModel):
        hot_streams: List[Dict[str, Any]] = Field(..., description="Hot streams")
        cold_streams: List[Dict[str, Any]] = Field(..., description="Cold streams")
        pinch_temp_hot_c: float = Field(..., description="Pinch temperature hot side")
        pinch_temp_cold_c: float = Field(..., description="Pinch temperature cold side")
        min_approach_temp_c: float = Field(10.0, ge=1, description="Minimum approach temperature")
        capital_recovery_factor: float = Field(0.15, ge=0.05, le=0.3, description="Capital recovery factor")
        energy_cost_usd_kwh: float = Field(0.10, ge=0.01, description="Energy cost")

    class OutputSchema(BaseModel):
        network_configuration: List[Dict[str, Any]] = Field(..., description="Optimized network configuration")
        total_exchangers: int = Field(..., ge=0, description="Number of heat exchangers")
        total_area_m2: float = Field(..., ge=0, description="Total heat transfer area")
        total_capital_cost_usd: float = Field(..., ge=0, description="Total capital cost")
        annual_operating_cost_usd: float = Field(..., ge=0, description="Annual operating cost")
        total_heat_recovery_kw: float = Field(..., ge=0, description="Total heat recovered")
        network_complexity_index: float = Field(..., ge=0, description="Network complexity metric")
        provenance_hash: str = Field(..., description="SHA-256 hash")

    async def execute(self, input_data: InputSchema) -> OutputSchema:
        """Optimize heat exchanger network using pinch design method."""
        network_config = []
        total_area = 0
        total_capital = 0
        total_recovery = 0

        # Separate streams above and below pinch
        hot_above_pinch = [s for s in input_data.hot_streams
                           if s["inlet_temp_c"] > input_data.pinch_temp_hot_c]
        cold_above_pinch = [s for s in input_data.cold_streams
                            if s["outlet_temp_c"] > input_data.pinch_temp_cold_c]

        hot_below_pinch = [s for s in input_data.hot_streams
                           if s["outlet_temp_c"] <= input_data.pinch_temp_hot_c]
        cold_below_pinch = [s for s in input_data.cold_streams
                            if s["inlet_temp_c"] <= input_data.pinch_temp_cold_c]

        # Design network above pinch
        matches_above = self._match_streams(
            hot_above_pinch,
            cold_above_pinch,
            input_data.min_approach_temp_c
        )

        # Design network below pinch
        matches_below = self._match_streams(
            hot_below_pinch,
            cold_below_pinch,
            input_data.min_approach_temp_c
        )

        # Combine matches and size exchangers
        all_matches = matches_above + matches_below
        exchange_id = 0

        for match in all_matches:
            exchange_id += 1
            exchanger = {
                "id": f"HX-{exchange_id:03d}",
                "hot_stream": match["hot_stream"],
                "cold_stream": match["cold_stream"],
                "duty_kw": match["duty_kw"],
                "lmtd_c": match["lmtd"],
                "area_m2": match["duty_kw"] * 1000 / (500 * match["lmtd"]),  # U = 500 W/m2.K assumed
                "capital_cost_usd": 0,
                "location": match["location"]  # above or below pinch
            }

            # Estimate capital cost
            exchanger["capital_cost_usd"] = 500 * (exchanger["area_m2"] ** 0.65)

            network_config.append(exchanger)
            total_area += exchanger["area_m2"]
            total_capital += exchanger["capital_cost_usd"]
            total_recovery += exchanger["duty_kw"]

        # Calculate network complexity
        complexity = len(network_config) + sum(1 for ex in network_config if ex["area_m2"] > 100)

        # Annual operating cost (pumping, maintenance)
        annual_operating = total_capital * 0.05 + total_area * 10  # Simplified

        # Provenance
        provenance_data = json.dumps({
            "exchangers": len(network_config),
            "area": total_area,
            "recovery": total_recovery
        }, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        return self.OutputSchema(
            network_configuration=network_config,
            total_exchangers=len(network_config),
            total_area_m2=total_area,
            total_capital_cost_usd=total_capital,
            annual_operating_cost_usd=annual_operating,
            total_heat_recovery_kw=total_recovery,
            network_complexity_index=complexity,
            provenance_hash=provenance_hash
        )

    def _match_streams(self, hot_streams, cold_streams, min_approach):
        """Match hot and cold streams for heat exchange."""
        matches = []
        location = "above_pinch" if any(s.get("inlet_temp_c", 0) > 200 for s in hot_streams) else "below_pinch"

        for hot in hot_streams:
            for cold in cold_streams:
                # Check feasibility
                if (hot["inlet_temp_c"] - cold["outlet_temp_c"] >= min_approach and
                    hot["outlet_temp_c"] - cold["inlet_temp_c"] >= min_approach):

                    # Calculate duty
                    hot_duty = (hot["flow_rate_kg_s"] * hot["specific_heat_kj_kg_k"] *
                               abs(hot["inlet_temp_c"] - hot["outlet_temp_c"]))
                    cold_duty = (cold["flow_rate_kg_s"] * cold["specific_heat_kj_kg_k"] *
                                abs(cold["outlet_temp_c"] - cold["inlet_temp_c"]))
                    duty = min(hot_duty, cold_duty)

                    # Calculate LMTD
                    dt1 = hot["inlet_temp_c"] - cold["outlet_temp_c"]
                    dt2 = hot["outlet_temp_c"] - cold["inlet_temp_c"]
                    lmtd = (dt1 - dt2) / np.log(dt1 / dt2) if abs(dt1 - dt2) > 0.1 else dt1

                    matches.append({
                        "hot_stream": hot.get("stream_id", "H"),
                        "cold_stream": cold.get("stream_id", "C"),
                        "duty_kw": duty,
                        "lmtd": lmtd,
                        "location": location
                    })

        return matches


# Tool 6: Economic Analysis Tool
class EconomicAnalysisTool(HeatRecoveryTool):
    """Tool for economic analysis of heat recovery projects."""

    name: str = Field(default="economic_analysis", const=True)
    category: ToolCategory = Field(default=ToolCategory.ANALYSIS, const=True)

    class InputSchema(BaseModel):
        capital_cost_usd: float = Field(..., ge=0, description="Total capital investment")
        annual_savings_usd: float = Field(..., ge=0, description="Annual cost savings")
        project_lifetime_years: int = Field(15, ge=1, le=30, description="Project lifetime")
        discount_rate: float = Field(0.10, ge=0, le=0.3, description="Discount rate")
        inflation_rate: float = Field(0.03, ge=0, le=0.1, description="Inflation rate")
        energy_price_escalation: float = Field(0.05, ge=0, le=0.15, description="Energy price escalation rate")
        carbon_price_usd_ton: float = Field(50, ge=0, description="Carbon price")
        maintenance_cost_percent: float = Field(0.03, ge=0, le=0.1, description="Annual maintenance as % of capital")

    class OutputSchema(BaseModel):
        npv_usd: float = Field(..., description="Net present value")
        irr_percent: float = Field(..., description="Internal rate of return")
        payback_years: float = Field(..., ge=0, description="Simple payback period")
        discounted_payback_years: float = Field(..., ge=0, description="Discounted payback period")
        profitability_index: float = Field(..., ge=0, description="Profitability index")
        lcoe_usd_kwh: float = Field(..., ge=0, description="Levelized cost of energy")
        cash_flows: List[float] = Field(..., description="Annual cash flows")
        cumulative_npv: List[float] = Field(..., description="Cumulative NPV by year")
        sensitivity_analysis: Dict[str, Dict[str, float]] = Field(..., description="Sensitivity analysis results")
        provenance_hash: str = Field(..., description="SHA-256 hash")

    async def execute(self, input_data: InputSchema) -> OutputSchema:
        """Perform comprehensive economic analysis."""
        cash_flows = []
        cumulative_npv = []
        cumulative = 0

        # Year 0 - Initial investment
        cash_flows.append(-input_data.capital_cost_usd)
        cumulative -= input_data.capital_cost_usd
        cumulative_npv.append(cumulative)

        # Annual cash flows
        for year in range(1, input_data.project_lifetime_years + 1):
            # Escalate savings
            annual_savings = input_data.annual_savings_usd * \
                           (1 + input_data.energy_price_escalation) ** (year - 1)

            # Maintenance costs
            maintenance = input_data.capital_cost_usd * input_data.maintenance_cost_percent * \
                         (1 + input_data.inflation_rate) ** (year - 1)

            # Net cash flow
            net_cash_flow = annual_savings - maintenance

            # Discount to present value
            pv_cash_flow = net_cash_flow / (1 + input_data.discount_rate) ** year

            cash_flows.append(net_cash_flow)
            cumulative += pv_cash_flow
            cumulative_npv.append(cumulative)

        # Calculate NPV
        npv = sum(cf / (1 + input_data.discount_rate) ** i for i, cf in enumerate(cash_flows))

        # Calculate IRR (simplified Newton-Raphson)
        irr = self._calculate_irr(cash_flows)

        # Simple payback
        cumulative_cash = 0
        payback = input_data.project_lifetime_years
        for year, cf in enumerate(cash_flows[1:], 1):
            cumulative_cash += cf
            if cumulative_cash >= input_data.capital_cost_usd:
                payback = year
                break

        # Discounted payback
        cumulative_discounted = 0
        discounted_payback = input_data.project_lifetime_years
        for year, cf in enumerate(cash_flows[1:], 1):
            cumulative_discounted += cf / (1 + input_data.discount_rate) ** year
            if cumulative_discounted >= input_data.capital_cost_usd:
                discounted_payback = year
                break

        # Profitability index
        pv_benefits = sum(cf / (1 + input_data.discount_rate) ** i
                         for i, cf in enumerate(cash_flows[1:], 1) if cf > 0)
        profitability_index = pv_benefits / input_data.capital_cost_usd if input_data.capital_cost_usd > 0 else 0

        # LCOE (simplified)
        total_costs = input_data.capital_cost_usd + sum(
            input_data.capital_cost_usd * input_data.maintenance_cost_percent *
            (1 + input_data.inflation_rate) ** (year - 1) / (1 + input_data.discount_rate) ** year
            for year in range(1, input_data.project_lifetime_years + 1)
        )
        total_energy_kwh = input_data.annual_savings_usd / 0.10 * input_data.project_lifetime_years  # Assuming $0.10/kWh
        lcoe = total_costs / total_energy_kwh if total_energy_kwh > 0 else 0

        # Sensitivity analysis
        sensitivity = await self._perform_sensitivity_analysis(input_data)

        # Provenance
        provenance_data = json.dumps({
            "npv": npv,
            "irr": irr * 100,
            "payback": payback
        }, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        return self.OutputSchema(
            npv_usd=npv,
            irr_percent=irr * 100,
            payback_years=payback,
            discounted_payback_years=discounted_payback,
            profitability_index=profitability_index,
            lcoe_usd_kwh=lcoe,
            cash_flows=cash_flows,
            cumulative_npv=cumulative_npv,
            sensitivity_analysis=sensitivity,
            provenance_hash=provenance_hash
        )

    def _calculate_irr(self, cash_flows):
        """Calculate IRR using Newton-Raphson method."""
        irr = 0.1
        tolerance = 1e-6
        max_iterations = 100

        for _ in range(max_iterations):
            npv = sum(cf / (1 + irr) ** i for i, cf in enumerate(cash_flows))
            if abs(npv) < tolerance:
                return irr

            npv_derivative = sum(-i * cf / (1 + irr) ** (i + 1) for i, cf in enumerate(cash_flows))
            if npv_derivative == 0:
                break

            irr = irr - npv / npv_derivative
            irr = max(-0.99, min(irr, 10.0))  # Bound IRR

        return irr

    async def _perform_sensitivity_analysis(self, base_case):
        """Perform sensitivity analysis on key parameters."""
        sensitivity_params = {
            "capital_cost": [-20, -10, 0, 10, 20],
            "energy_savings": [-20, -10, 0, 10, 20],
            "discount_rate": [-2, -1, 0, 1, 2],
            "energy_escalation": [-2, -1, 0, 1, 2]
        }

        results = {}
        for param, variations in sensitivity_params.items():
            param_results = {}
            for variation in variations:
                # Modify parameter
                modified_input = base_case.dict()
                if param == "capital_cost":
                    modified_input["capital_cost_usd"] *= (1 + variation / 100)
                elif param == "energy_savings":
                    modified_input["annual_savings_usd"] *= (1 + variation / 100)
                elif param == "discount_rate":
                    modified_input["discount_rate"] += variation / 100
                elif param == "energy_escalation":
                    modified_input["energy_price_escalation"] += variation / 100

                # Recalculate NPV
                cash_flows = [-modified_input["capital_cost_usd"]]
                for year in range(1, modified_input["project_lifetime_years"] + 1):
                    annual_savings = modified_input["annual_savings_usd"] * \
                                   (1 + modified_input["energy_price_escalation"]) ** (year - 1)
                    cash_flows.append(annual_savings)

                npv = sum(cf / (1 + modified_input["discount_rate"]) ** i
                         for i, cf in enumerate(cash_flows))

                param_results[f"{variation:+d}%"] = npv

            results[param] = param_results

        return results


# Tool Registry
TOOL_REGISTRY = ToolRegistry()

# Register all tools
tools_to_register = [
    StreamAnalysisTool(),
    PinchAnalysisTool(),
    HeatExchangerDesignTool(),
    ExergyAnalysisTool(),
    NetworkOptimizationTool(),
    EconomicAnalysisTool()
]

for tool in tools_to_register:
    TOOL_REGISTRY.register(tool.name, tool)


# Tool schema generation for API documentation
def generate_tool_schemas() -> Dict[str, Any]:
    """Generate OpenAPI schemas for all registered tools."""
    schemas = {}
    for tool_name, tool in TOOL_REGISTRY.tools.items():
        schemas[tool_name] = {
            "name": tool.name,
            "category": tool.category,
            "priority": tool.priority,
            "version": tool.version,
            "zero_hallucination": tool.zero_hallucination,
            "input_schema": tool.InputSchema.schema() if hasattr(tool, 'InputSchema') else {},
            "output_schema": tool.OutputSchema.schema() if hasattr(tool, 'OutputSchema') else {},
            "description": tool.__doc__ or "No description available"
        }
    return schemas


# Export schemas for API
TOOL_SCHEMAS = generate_tool_schemas()

__all__ = [
    "TOOL_REGISTRY",
    "TOOL_SCHEMAS",
    "StreamAnalysisTool",
    "PinchAnalysisTool",
    "HeatExchangerDesignTool",
    "ExergyAnalysisTool",
    "NetworkOptimizationTool",
    "EconomicAnalysisTool",
    "HeatRecoveryTool",
    "ToolCategory",
    "ToolPriority"
]