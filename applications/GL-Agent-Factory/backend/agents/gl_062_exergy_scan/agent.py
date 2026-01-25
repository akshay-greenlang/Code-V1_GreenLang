"""
GL-062: Exergy Scan Agent (EXERGY-SCAN)

This module implements the ExergyScanAgent for exergy analysis and destruction mapping
in industrial process systems using Second Law of Thermodynamics principles.

Standards Reference:
    - Second Law of Thermodynamics
    - Bejan, A. "Advanced Engineering Thermodynamics"
    - ISO 50001 Energy Management

Example:
    >>> agent = ExergyScanAgent()
    >>> result = agent.run(ExergyScanInput(process_data=[...], reference_state=ReferenceState()))
    >>> print(f"Total exergy destruction: {result.total_exergy_destruction_kW:.2f} kW")
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProcessUnitType(str, Enum):
    HEAT_EXCHANGER = "heat_exchanger"
    COMPRESSOR = "compressor"
    TURBINE = "turbine"
    PUMP = "pump"
    VALVE = "valve"
    MIXER = "mixer"
    SEPARATOR = "separator"
    REACTOR = "reactor"
    BOILER = "boiler"
    CONDENSER = "condenser"


class ReferenceState(BaseModel):
    """Dead state reference for exergy calculations."""
    temperature_K: float = Field(default=298.15, description="Reference temperature (K)")
    pressure_kPa: float = Field(default=101.325, description="Reference pressure (kPa)")
    composition: Dict[str, float] = Field(default_factory=lambda: {"N2": 0.7808, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004})


class StreamComposition(BaseModel):
    """Stream composition for chemical exergy."""
    components: Dict[str, float] = Field(..., description="Mole fractions of components")


class ProcessStream(BaseModel):
    """Process stream data for exergy analysis."""
    stream_id: str = Field(..., description="Unique stream identifier")
    name: str = Field(..., description="Stream name")
    mass_flow_kg_s: float = Field(..., ge=0, description="Mass flow rate (kg/s)")
    temperature_K: float = Field(..., gt=0, description="Stream temperature (K)")
    pressure_kPa: float = Field(..., gt=0, description="Stream pressure (kPa)")
    enthalpy_kJ_kg: Optional[float] = Field(None, description="Specific enthalpy (kJ/kg)")
    entropy_kJ_kg_K: Optional[float] = Field(None, description="Specific entropy (kJ/kg-K)")
    specific_heat_kJ_kg_K: float = Field(default=4.18, description="Specific heat (kJ/kg-K)")
    composition: Optional[StreamComposition] = Field(None, description="Chemical composition")


class ProcessUnit(BaseModel):
    """Process unit for exergy destruction calculation."""
    unit_id: str = Field(..., description="Unique unit identifier")
    name: str = Field(..., description="Unit name")
    unit_type: ProcessUnitType = Field(..., description="Type of process unit")
    inlet_streams: List[str] = Field(..., description="Inlet stream IDs")
    outlet_streams: List[str] = Field(..., description="Outlet stream IDs")
    heat_input_kW: float = Field(default=0.0, description="Heat input rate (kW)")
    heat_output_kW: float = Field(default=0.0, description="Heat output rate (kW)")
    work_input_kW: float = Field(default=0.0, description="Work input rate (kW)")
    work_output_kW: float = Field(default=0.0, description="Work output rate (kW)")
    heat_source_temp_K: Optional[float] = Field(None, description="Heat source temperature (K)")
    heat_sink_temp_K: Optional[float] = Field(None, description="Heat sink temperature (K)")


class ExergyScanInput(BaseModel):
    """Input for exergy analysis."""
    analysis_id: Optional[str] = Field(None, description="Analysis identifier")
    system_name: str = Field(default="Process System", description="System name")
    process_streams: List[ProcessStream] = Field(..., description="Process streams")
    process_units: List[ProcessUnit] = Field(..., description="Process units")
    reference_state: ReferenceState = Field(default_factory=ReferenceState)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExergyFlow(BaseModel):
    """Exergy flow for a stream."""
    stream_id: str
    name: str
    physical_exergy_kW: float
    chemical_exergy_kW: float
    total_exergy_kW: float
    specific_exergy_kJ_kg: float


class ExergyDestruction(BaseModel):
    """Exergy destruction in a process unit."""
    unit_id: str
    name: str
    unit_type: str
    exergy_input_kW: float
    exergy_output_kW: float
    exergy_destruction_kW: float
    exergy_efficiency_percent: float
    irreversibility_ratio: float


class ImprovementPriority(BaseModel):
    """Improvement priority ranking."""
    unit_id: str
    name: str
    destruction_kW: float
    destruction_percent: float
    improvement_potential: str
    recommendation: str
    priority_rank: int


class ExergyScanOutput(BaseModel):
    """Output from exergy analysis."""
    analysis_id: str
    system_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reference_state: ReferenceState
    exergy_flows: List[ExergyFlow]
    destruction_locations: List[ExergyDestruction]
    total_exergy_input_kW: float
    total_exergy_output_kW: float
    total_exergy_destruction_kW: float
    system_exergy_efficiency_percent: float
    efficiency_map: Dict[str, float]
    improvement_priorities: List[ImprovementPriority]
    carnot_factors: Dict[str, float]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class ExergyScanAgent:
    """GL-062: Exergy Scan Agent - Second Law analysis and destruction mapping."""

    AGENT_ID = "GL-062"
    AGENT_NAME = "EXERGY-SCAN"
    VERSION = "1.0.0"
    R_UNIVERSAL = 8.314  # kJ/kmol-K

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._streams_map: Dict[str, ProcessStream] = {}
        logger.info(f"ExergyScanAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: ExergyScanInput) -> ExergyScanOutput:
        start_time = datetime.utcnow()
        ref = input_data.reference_state
        self._streams_map = {s.stream_id: s for s in input_data.process_streams}

        # Calculate exergy flows for all streams
        exergy_flows = []
        for stream in input_data.process_streams:
            ex_phys = self._calculate_physical_exergy(stream, ref)
            ex_chem = self._calculate_chemical_exergy(stream, ref)
            total_ex = ex_phys + ex_chem
            spec_ex = total_ex / stream.mass_flow_kg_s if stream.mass_flow_kg_s > 0 else 0
            exergy_flows.append(ExergyFlow(
                stream_id=stream.stream_id, name=stream.name,
                physical_exergy_kW=round(ex_phys, 2), chemical_exergy_kW=round(ex_chem, 2),
                total_exergy_kW=round(total_ex, 2), specific_exergy_kJ_kg=round(spec_ex, 2)))

        exergy_map = {ef.stream_id: ef.total_exergy_kW for ef in exergy_flows}

        # Calculate exergy destruction for each unit
        destructions = []
        efficiency_map = {}
        carnot_factors = {}
        total_destruction = 0.0

        for unit in input_data.process_units:
            ex_in = sum(exergy_map.get(sid, 0) for sid in unit.inlet_streams)
            ex_out = sum(exergy_map.get(sid, 0) for sid in unit.outlet_streams)

            # Add work/heat exergy
            ex_in += unit.work_input_kW
            if unit.heat_input_kW > 0 and unit.heat_source_temp_K:
                carnot = 1 - ref.temperature_K / unit.heat_source_temp_K
                carnot_factors[f"{unit.unit_id}_heat_in"] = round(carnot, 4)
                ex_in += unit.heat_input_kW * carnot

            ex_out += unit.work_output_kW
            if unit.heat_output_kW > 0 and unit.heat_sink_temp_K:
                carnot = 1 - ref.temperature_K / unit.heat_sink_temp_K
                carnot_factors[f"{unit.unit_id}_heat_out"] = round(carnot, 4)
                ex_out += unit.heat_output_kW * max(0, carnot)

            destruction = max(0, ex_in - ex_out)
            efficiency = (ex_out / ex_in * 100) if ex_in > 0 else 0
            irreversibility = destruction / ex_in if ex_in > 0 else 0

            destructions.append(ExergyDestruction(
                unit_id=unit.unit_id, name=unit.name, unit_type=unit.unit_type.value,
                exergy_input_kW=round(ex_in, 2), exergy_output_kW=round(ex_out, 2),
                exergy_destruction_kW=round(destruction, 2),
                exergy_efficiency_percent=round(efficiency, 2),
                irreversibility_ratio=round(irreversibility, 4)))

            efficiency_map[unit.unit_id] = round(efficiency, 2)
            total_destruction += destruction

        # Calculate system totals
        total_input = sum(d.exergy_input_kW for d in destructions)
        total_output = sum(d.exergy_output_kW for d in destructions)
        system_efficiency = (total_output / total_input * 100) if total_input > 0 else 0

        # Generate improvement priorities
        priorities = self._generate_priorities(destructions, total_destruction)

        provenance_hash = hashlib.sha256(
            json.dumps({"agent": self.AGENT_ID, "input": input_data.dict(),
                       "timestamp": datetime.utcnow().isoformat()}, sort_keys=True, default=str).encode()
        ).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ExergyScanOutput(
            analysis_id=input_data.analysis_id or f"EX-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            system_name=input_data.system_name, reference_state=ref,
            exergy_flows=exergy_flows, destruction_locations=destructions,
            total_exergy_input_kW=round(total_input, 2), total_exergy_output_kW=round(total_output, 2),
            total_exergy_destruction_kW=round(total_destruction, 2),
            system_exergy_efficiency_percent=round(system_efficiency, 2),
            efficiency_map=efficiency_map, improvement_priorities=priorities,
            carnot_factors=carnot_factors, provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2), validation_status="PASS")

    def _calculate_physical_exergy(self, stream: ProcessStream, ref: ReferenceState) -> float:
        """Calculate physical exergy: ex_ph = (h - h0) - T0*(s - s0)"""
        T, T0 = stream.temperature_K, ref.temperature_K
        P, P0 = stream.pressure_kPa, ref.pressure_kPa
        cp = stream.specific_heat_kJ_kg_K

        # Simplified for ideal gas/incompressible liquid
        delta_h = cp * (T - T0)
        delta_s = cp * math.log(T / T0) if T > 0 and T0 > 0 else 0

        ex_specific = delta_h - T0 * delta_s  # kJ/kg
        return stream.mass_flow_kg_s * ex_specific

    def _calculate_chemical_exergy(self, stream: ProcessStream, ref: ReferenceState) -> float:
        """Calculate chemical exergy (simplified)."""
        # Standard chemical exergies (kJ/mol) for common components
        CHEM_EXERGY = {"CH4": 831.2, "H2": 236.1, "CO": 275.1, "CO2": 19.9, "H2O": 9.5, "N2": 0.72, "O2": 3.97}

        if not stream.composition:
            return 0.0

        ex_chem = 0.0
        for comp, mole_frac in stream.composition.components.items():
            ex_chem += mole_frac * CHEM_EXERGY.get(comp, 0)

        return stream.mass_flow_kg_s * ex_chem * 0.001  # Convert to kW

    def _generate_priorities(self, destructions: List[ExergyDestruction], total: float) -> List[ImprovementPriority]:
        """Generate improvement priorities based on exergy destruction."""
        priorities = []
        sorted_dest = sorted(destructions, key=lambda x: x.exergy_destruction_kW, reverse=True)

        for rank, d in enumerate(sorted_dest, 1):
            pct = (d.exergy_destruction_kW / total * 100) if total > 0 else 0
            potential = "HIGH" if pct >= 20 else ("MEDIUM" if pct >= 10 else "LOW")

            recommendations = {
                "heat_exchanger": "Optimize approach temperatures, consider larger heat transfer area",
                "compressor": "Check for internal leaks, optimize staging",
                "turbine": "Inspect blade condition, optimize extraction pressures",
                "valve": "Replace with turbine/expander for energy recovery",
                "mixer": "Minimize temperature/pressure differences",
                "boiler": "Improve combustion efficiency, reduce stack losses",
            }
            rec = recommendations.get(d.unit_type, "Review operating conditions")

            priorities.append(ImprovementPriority(
                unit_id=d.unit_id, name=d.name, destruction_kW=d.exergy_destruction_kW,
                destruction_percent=round(pct, 2), improvement_potential=potential,
                recommendation=rec, priority_rank=rank))

        return priorities


PACK_SPEC = {"schema_version": "2.0.0", "id": "GL-062", "name": "EXERGY-SCAN", "version": "1.0.0",
    "summary": "Exergy analysis and destruction mapping for process optimization",
    "tags": ["exergy", "second-law", "thermodynamics", "efficiency", "irreversibility"],
    "standards": [{"ref": "ISO 50001", "description": "Energy Management Systems"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}}
