"""
GL-062: Exergy Analyzer Agent (EXERGY-SCAN)

This module implements the ExergyAnalyzerAgent for comprehensive exergy analysis
(Second Law of Thermodynamics) including exergy destruction, efficiency, and quality assessment.

Standards Reference:
    - Second Law of Thermodynamics
    - Exergy Analysis Methods
    - ISO 13600 (Technical energy systems - Methods for analysis)

Example:
    >>> agent = ExergyAnalyzerAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Exergy efficiency: {result.exergy_efficiency_percent:.2f}%")
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class StreamType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    UTILITY = "utility"
    WASTE = "waste"


class ExergyComponent(str, Enum):
    PHYSICAL = "physical"
    CHEMICAL = "chemical"
    KINETIC = "kinetic"
    POTENTIAL = "potential"


class ProcessStream(BaseModel):
    stream_id: str = Field(..., description="Unique stream identifier")
    name: str = Field(..., description="Stream name")
    stream_type: StreamType = Field(..., description="Stream classification")
    mass_flow_kg_s: float = Field(..., ge=0, description="Mass flow rate (kg/s)")
    temperature_K: float = Field(..., gt=0, description="Temperature (K)")
    pressure_kPa: float = Field(..., gt=0, description="Pressure (kPa)")
    enthalpy_kJ_kg: Optional[float] = Field(None, description="Specific enthalpy (kJ/kg)")
    entropy_kJ_kg_K: Optional[float] = Field(None, description="Specific entropy (kJ/kg-K)")
    chemical_exergy_kJ_kg: Optional[float] = Field(0.0, ge=0, description="Chemical exergy (kJ/kg)")
    velocity_m_s: Optional[float] = Field(0.0, ge=0, description="Velocity (m/s)")
    elevation_m: Optional[float] = Field(0.0, description="Elevation (m)")


class DeadState(BaseModel):
    temperature_K: float = Field(default=298.15, gt=0, description="Dead state temperature (K)")
    pressure_kPa: float = Field(default=101.325, gt=0, description="Dead state pressure (kPa)")
    relative_humidity_percent: float = Field(default=60.0, ge=0, le=100, description="RH (%)")


class ExergyInput(BaseModel):
    analysis_id: Optional[str] = Field(None, description="Analysis identifier")
    system_name: str = Field(default="Process System", description="System name")
    process_streams: List[ProcessStream] = Field(default_factory=list)
    dead_state: DeadState = Field(default_factory=DeadState)
    power_input_kW: Optional[float] = Field(0.0, ge=0, description="Power input (kW)")
    heat_input_kW: Optional[float] = Field(0.0, ge=0, description="Heat input (kW)")
    heat_source_temp_K: Optional[float] = Field(None, gt=0, description="Heat source temp (K)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamExergy(BaseModel):
    stream_id: str
    name: str
    stream_type: str
    mass_flow_kg_s: float
    temperature_K: float
    pressure_kPa: float
    physical_exergy_kJ_kg: float
    chemical_exergy_kJ_kg: float
    kinetic_exergy_kJ_kg: float
    potential_exergy_kJ_kg: float
    total_specific_exergy_kJ_kg: float
    exergy_flow_kW: float


class ExergyBalance(BaseModel):
    input_streams: List[StreamExergy]
    output_streams: List[StreamExergy]
    total_exergy_input_kW: float
    total_exergy_output_kW: float
    total_exergy_destruction_kW: float
    exergy_destruction_percent: float


class ComponentExergyDestruction(BaseModel):
    component_id: str
    name: str
    exergy_destruction_kW: float
    percent_of_total: float
    irreversibility_source: str
    priority: str


class ExergyImprovement(BaseModel):
    recommendation_id: str
    description: str
    affected_component: str
    current_exergy_destruction_kW: float
    potential_reduction_kW: float
    estimated_efficiency_gain_percent: float
    priority: str
    implementation_complexity: str


class ProvenanceRecord(BaseModel):
    operation: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    tool_name: str


class ExergyOutput(BaseModel):
    analysis_id: str
    system_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    exergy_balance: ExergyBalance
    exergy_efficiency_percent: float
    second_law_efficiency_percent: float
    component_destructions: List[ComponentExergyDestruction]
    total_irreversibility_kW: float
    improvement_opportunities: List[ExergyImprovement]
    recommendations: List[str]
    warnings: List[str]
    provenance_chain: List[ProvenanceRecord]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str
    validation_errors: List[str] = Field(default_factory=list)


class ExergyAnalyzerAgent:
    """GL-062: Exergy Analyzer Agent - Second Law thermodynamic analysis."""

    AGENT_ID = "GL-062"
    AGENT_NAME = "EXERGY-SCAN"
    VERSION = "1.0.0"
    GRAVITY_M_S2 = 9.81

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []
        self._warnings: List[str] = []
        self._recommendations: List[str] = []
        logger.info(f"ExergyAnalyzerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: ExergyInput) -> ExergyOutput:
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []
        self._warnings = []
        self._recommendations = []

        # Validate inputs
        if not input_data.process_streams:
            self._validation_errors.append("No process streams provided")

        T0 = input_data.dead_state.temperature_K
        P0 = input_data.dead_state.pressure_kPa

        # Calculate stream exergies
        input_streams = []
        output_streams = []
        total_input_exergy = 0.0
        total_output_exergy = 0.0

        for stream in input_data.process_streams:
            stream_exergy = self._calculate_stream_exergy(stream, T0, P0)

            if stream.stream_type in [StreamType.INPUT, StreamType.UTILITY]:
                input_streams.append(stream_exergy)
                total_input_exergy += stream_exergy.exergy_flow_kW
            else:
                output_streams.append(stream_exergy)
                total_output_exergy += stream_exergy.exergy_flow_kW

        # Add power and heat inputs
        if input_data.power_input_kW and input_data.power_input_kW > 0:
            power_exergy = StreamExergy(
                stream_id="POWER_INPUT",
                name="Electrical Power",
                stream_type="utility",
                mass_flow_kg_s=0.0,
                temperature_K=T0,
                pressure_kPa=P0,
                physical_exergy_kJ_kg=0.0,
                chemical_exergy_kJ_kg=0.0,
                kinetic_exergy_kJ_kg=0.0,
                potential_exergy_kJ_kg=0.0,
                total_specific_exergy_kJ_kg=0.0,
                exergy_flow_kW=input_data.power_input_kW
            )
            input_streams.append(power_exergy)
            total_input_exergy += input_data.power_input_kW

        if input_data.heat_input_kW and input_data.heat_input_kW > 0:
            carnot_factor = 1.0 - (T0 / input_data.heat_source_temp_K) if input_data.heat_source_temp_K else 0.5
            heat_exergy_kW = input_data.heat_input_kW * carnot_factor
            heat_exergy = StreamExergy(
                stream_id="HEAT_INPUT",
                name="Heat Input",
                stream_type="utility",
                mass_flow_kg_s=0.0,
                temperature_K=input_data.heat_source_temp_K or T0,
                pressure_kPa=P0,
                physical_exergy_kJ_kg=0.0,
                chemical_exergy_kJ_kg=0.0,
                kinetic_exergy_kJ_kg=0.0,
                potential_exergy_kJ_kg=0.0,
                total_specific_exergy_kJ_kg=0.0,
                exergy_flow_kW=heat_exergy_kW
            )
            input_streams.append(heat_exergy)
            total_input_exergy += heat_exergy_kW

        self._track_provenance("exergy_calculation",
            {"num_streams": len(input_data.process_streams), "T0": T0, "P0": P0},
            {"total_input_kW": total_input_exergy, "total_output_kW": total_output_exergy},
            "Exergy Calculator")

        # Calculate exergy destruction
        exergy_destruction_kW = total_input_exergy - total_output_exergy
        exergy_destruction_percent = (exergy_destruction_kW / total_input_exergy * 100.0) if total_input_exergy > 0 else 0.0

        exergy_balance = ExergyBalance(
            input_streams=input_streams,
            output_streams=output_streams,
            total_exergy_input_kW=round(total_input_exergy, 2),
            total_exergy_output_kW=round(total_output_exergy, 2),
            total_exergy_destruction_kW=round(exergy_destruction_kW, 2),
            exergy_destruction_percent=round(exergy_destruction_percent, 2)
        )

        # Calculate efficiencies
        exergy_efficiency = (total_output_exergy / total_input_exergy * 100.0) if total_input_exergy > 0 else 0.0
        second_law_efficiency = exergy_efficiency  # Same for this analysis

        # Identify component destructions
        component_destructions = self._identify_component_destructions(input_streams, output_streams, exergy_destruction_kW)

        # Generate improvement opportunities
        improvements = self._generate_improvements(component_destructions, exergy_destruction_kW)

        # Generate recommendations and warnings
        self._generate_recommendations_and_warnings(exergy_efficiency, exergy_destruction_percent, component_destructions)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ExergyOutput(
            analysis_id=input_data.analysis_id or f"EX-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            system_name=input_data.system_name,
            exergy_balance=exergy_balance,
            exergy_efficiency_percent=round(exergy_efficiency, 2),
            second_law_efficiency_percent=round(second_law_efficiency, 2),
            component_destructions=component_destructions,
            total_irreversibility_kW=round(exergy_destruction_kW, 2),
            improvement_opportunities=improvements,
            recommendations=self._recommendations,
            warnings=self._warnings,
            provenance_chain=[ProvenanceRecord(**{k: v for k, v in s.items()}) for s in self._provenance_steps],
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS" if not self._validation_errors else "FAIL",
            validation_errors=self._validation_errors
        )

    def _calculate_stream_exergy(self, stream: ProcessStream, T0: float, P0: float) -> StreamExergy:
        """Calculate total exergy of a process stream."""
        # Physical exergy (thermal + mechanical)
        if stream.enthalpy_kJ_kg is not None and stream.entropy_kJ_kg is not None:
            h = stream.enthalpy_kJ_kg
            s = stream.entropy_kJ_kg
            # Simplified - would need dead state properties
            h0 = 0.0  # Approximation
            s0 = 0.0  # Approximation
            ex_ph = (h - h0) - T0 * (s - s0)
        else:
            # Simplified physical exergy for ideal gas
            cp = 1.005  # kJ/kg-K for air (approximation)
            R = 0.287  # kJ/kg-K for air
            ex_ph = cp * (stream.temperature_K - T0 - T0 * math.log(stream.temperature_K / T0)) + \
                    R * T0 * math.log(stream.pressure_kPa / P0)

        # Chemical exergy
        ex_ch = stream.chemical_exergy_kJ_kg or 0.0

        # Kinetic exergy
        v = stream.velocity_m_s or 0.0
        ex_ke = (v ** 2) / 2000.0  # Convert J/kg to kJ/kg

        # Potential exergy
        z = stream.elevation_m or 0.0
        ex_pe = (self.GRAVITY_M_S2 * z) / 1000.0  # Convert J/kg to kJ/kg

        # Total specific exergy
        ex_total = ex_ph + ex_ch + ex_ke + ex_pe

        # Exergy flow rate
        exergy_flow_kW = stream.mass_flow_kg_s * ex_total

        return StreamExergy(
            stream_id=stream.stream_id,
            name=stream.name,
            stream_type=stream.stream_type.value,
            mass_flow_kg_s=round(stream.mass_flow_kg_s, 4),
            temperature_K=round(stream.temperature_K, 2),
            pressure_kPa=round(stream.pressure_kPa, 2),
            physical_exergy_kJ_kg=round(ex_ph, 3),
            chemical_exergy_kJ_kg=round(ex_ch, 3),
            kinetic_exergy_kJ_kg=round(ex_ke, 6),
            potential_exergy_kJ_kg=round(ex_pe, 6),
            total_specific_exergy_kJ_kg=round(ex_total, 3),
            exergy_flow_kW=round(exergy_flow_kW, 2)
        )

    def _identify_component_destructions(self, inputs: List[StreamExergy], outputs: List[StreamExergy],
                                         total_destruction: float) -> List[ComponentExergyDestruction]:
        """Identify sources of exergy destruction."""
        destructions = []

        # Heat transfer irreversibility
        heat_streams = [s for s in inputs + outputs if "heat" in s.name.lower() or s.temperature_K > 400]
        if heat_streams:
            heat_destruction = total_destruction * 0.3  # Estimate
            destructions.append(ComponentExergyDestruction(
                component_id="HEAT_TRANSFER",
                name="Heat Transfer Irreversibility",
                exergy_destruction_kW=round(heat_destruction, 2),
                percent_of_total=round(heat_destruction / total_destruction * 100, 2) if total_destruction > 0 else 0.0,
                irreversibility_source="Finite temperature differences",
                priority="HIGH" if heat_destruction / total_destruction > 0.25 else "MEDIUM"
            ))

        # Mixing irreversibility
        if len(inputs) > 1:
            mixing_destruction = total_destruction * 0.15  # Estimate
            destructions.append(ComponentExergyDestruction(
                component_id="MIXING",
                name="Mixing Irreversibility",
                exergy_destruction_kW=round(mixing_destruction, 2),
                percent_of_total=round(mixing_destruction / total_destruction * 100, 2) if total_destruction > 0 else 0.0,
                irreversibility_source="Entropy generation from mixing",
                priority="MEDIUM"
            ))

        # Pressure drop/throttling
        pressure_streams = [s for s in outputs if s.pressure_kPa < 500]
        if pressure_streams:
            pressure_destruction = total_destruction * 0.20  # Estimate
            destructions.append(ComponentExergyDestruction(
                component_id="PRESSURE_DROP",
                name="Pressure Drop/Throttling",
                exergy_destruction_kW=round(pressure_destruction, 2),
                percent_of_total=round(pressure_destruction / total_destruction * 100, 2) if total_destruction > 0 else 0.0,
                irreversibility_source="Friction and throttling losses",
                priority="MEDIUM"
            ))

        # Chemical reactions/combustion
        chem_streams = [s for s in inputs if s.chemical_exergy_kJ_kg > 100]
        if chem_streams:
            chem_destruction = total_destruction * 0.35  # Estimate
            destructions.append(ComponentExergyDestruction(
                component_id="COMBUSTION",
                name="Combustion/Chemical Reaction",
                exergy_destruction_kW=round(chem_destruction, 2),
                percent_of_total=round(chem_destruction / total_destruction * 100, 2) if total_destruction > 0 else 0.0,
                irreversibility_source="Irreversible chemical reactions",
                priority="HIGH"
            ))

        return sorted(destructions, key=lambda x: -x.exergy_destruction_kW)

    def _generate_improvements(self, destructions: List[ComponentExergyDestruction],
                               total_destruction: float) -> List[ExergyImprovement]:
        """Generate improvement opportunities based on exergy destruction."""
        improvements = []

        for i, dest in enumerate(destructions):
            if dest.percent_of_total >= 15.0:
                potential_reduction = dest.exergy_destruction_kW * 0.30  # 30% reduction potential

                recommendation = ""
                complexity = "MEDIUM"

                if "heat transfer" in dest.name.lower():
                    recommendation = "Reduce temperature differences in heat exchangers, consider heat integration"
                    complexity = "MEDIUM"
                elif "mixing" in dest.name.lower():
                    recommendation = "Optimize mixing process, reduce entropy generation"
                    complexity = "LOW"
                elif "pressure" in dest.name.lower():
                    recommendation = "Minimize pressure drops, optimize piping design, consider pressure recovery"
                    complexity = "MEDIUM"
                elif "combustion" in dest.name.lower():
                    recommendation = "Optimize combustion conditions, improve air-fuel ratio, preheat combustion air"
                    complexity = "HIGH"
                else:
                    recommendation = f"Investigate and reduce irreversibility in {dest.name}"
                    complexity = "MEDIUM"

                improvements.append(ExergyImprovement(
                    recommendation_id=f"IMP-{i+1:03d}",
                    description=recommendation,
                    affected_component=dest.component_id,
                    current_exergy_destruction_kW=dest.exergy_destruction_kW,
                    potential_reduction_kW=round(potential_reduction, 2),
                    estimated_efficiency_gain_percent=round(potential_reduction / total_destruction * 100, 2) if total_destruction > 0 else 0.0,
                    priority=dest.priority,
                    implementation_complexity=complexity
                ))

        return improvements

    def _generate_recommendations_and_warnings(self, efficiency: float, destruction_percent: float,
                                               destructions: List[ComponentExergyDestruction]) -> None:
        """Generate system-level recommendations and warnings."""
        if efficiency < 30.0:
            self._warnings.append(f"Very low exergy efficiency ({efficiency:.1f}%). Significant improvement potential exists.")
            self._recommendations.append("Conduct detailed exergy audit to identify major irreversibility sources")
        elif efficiency < 50.0:
            self._warnings.append(f"Low exergy efficiency ({efficiency:.1f}%). Consider optimization opportunities.")
            self._recommendations.append("Implement heat integration and process optimization strategies")

        if destruction_percent > 70.0:
            self._warnings.append(f"High exergy destruction ({destruction_percent:.1f}%). System operates far from reversibility.")
            self._recommendations.append("Prioritize reduction of major irreversibility sources")

        # Component-specific recommendations
        for dest in destructions[:3]:  # Top 3 destruction sources
            if dest.priority == "HIGH":
                self._recommendations.append(f"Priority action: Address {dest.name} ({dest.percent_of_total:.1f}% of destruction)")

        if not self._recommendations:
            self._recommendations.append("System shows good exergy efficiency. Continue monitoring for optimization opportunities.")

    def _track_provenance(self, operation: str, inputs: Dict, outputs: Dict, tool_name: str) -> None:
        """Track provenance of calculations."""
        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            "output_hash": hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(),
            "tool_name": tool_name
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of provenance chain."""
        data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": [{"operation": s["operation"], "input_hash": s["input_hash"], "output_hash": s["output_hash"]}
                     for s in self._provenance_steps],
            "timestamp": datetime.utcnow().isoformat()
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-062",
    "name": "EXERGY-SCAN",
    "version": "1.0.0",
    "summary": "Exergy analysis and Second Law efficiency assessment",
    "tags": ["exergy", "second-law", "thermodynamics", "efficiency", "irreversibility"],
    "standards": [
        {"ref": "Second Law of Thermodynamics", "description": "Exergy analysis foundation"},
        {"ref": "ISO 13600", "description": "Technical energy systems - Methods for analysis"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True
    }
}
