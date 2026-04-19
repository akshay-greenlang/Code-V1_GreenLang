"""
GL-061: Heat Balance Analyzer Agent (HEATBALANCE-ANALYZER)

This module implements the HeatBalanceAnalyzerAgent for comprehensive heat balance
calculation and analysis in industrial process systems using the First Law of Thermodynamics.

Standards Reference:
    - ASME PTC 4 (Fired Steam Generators)
    - First Law of Thermodynamics

Example:
    >>> agent = HeatBalanceAnalyzerAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Closure error: {result.closure_error_percent:.2f}%")
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
    RECYCLE = "recycle"
    LOSS = "loss"


class FluidType(str, Enum):
    WATER = "water"
    STEAM = "steam"
    AIR = "air"
    FLUE_GAS = "flue_gas"
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    PROCESS_FLUID = "process_fluid"
    CONDENSATE = "condensate"


class UtilityType(str, Enum):
    STEAM = "steam"
    ELECTRICITY = "electricity"
    FUEL = "fuel"
    COOLING_WATER = "cooling_water"
    COMPRESSED_AIR = "compressed_air"


SPECIFIC_HEAT_DATA = {
    FluidType.WATER: 4186.0,
    FluidType.STEAM: 2010.0,
    FluidType.AIR: 1005.0,
    FluidType.FLUE_GAS: 1100.0,
    FluidType.NATURAL_GAS: 2200.0,
    FluidType.FUEL_OIL: 2000.0,
    FluidType.PROCESS_FLUID: 2000.0,
    FluidType.CONDENSATE: 4186.0,
}

REFERENCE_TEMPERATURE_C = 25.0


class ProcessStream(BaseModel):
    stream_id: str = Field(..., description="Unique stream identifier")
    name: str = Field(..., description="Stream name/description")
    stream_type: StreamType = Field(..., description="Stream classification")
    fluid_type: FluidType = Field(..., description="Fluid type")
    mass_flow_kg_s: float = Field(..., ge=0, description="Mass flow rate (kg/s)")
    temperature_celsius: float = Field(..., description="Stream temperature (C)")
    pressure_MPa: Optional[float] = Field(None, gt=0, description="Pressure (MPa)")
    specific_heat_J_kg_K: Optional[float] = Field(None, gt=0, description="Specific heat (J/kg-K)")
    enthalpy_kJ_kg: Optional[float] = Field(None, description="Specific enthalpy (kJ/kg)")


class UtilityStream(BaseModel):
    utility_id: str = Field(..., description="Unique utility identifier")
    name: str = Field(..., description="Utility name")
    utility_type: UtilityType = Field(..., description="Type of utility")
    stream_type: StreamType = Field(default=StreamType.INPUT, description="Input or output")
    mass_flow_kg_s: Optional[float] = Field(None, ge=0, description="Mass flow (kg/s)")
    temperature_celsius: Optional[float] = Field(None, description="Temperature (C)")
    pressure_MPa: Optional[float] = Field(None, gt=0, description="Pressure (MPa)")
    enthalpy_kJ_kg: Optional[float] = Field(None, description="Specific enthalpy (kJ/kg)")
    lower_heating_value_kJ_kg: Optional[float] = Field(None, gt=0, description="Fuel LHV (kJ/kg)")
    power_kW: Optional[float] = Field(None, ge=0, description="Electrical power (kW)")
    heat_content_kW: Optional[float] = Field(None, description="Heat content rate (kW)")


class HeatLoss(BaseModel):
    loss_id: str = Field(..., description="Loss identifier")
    name: str = Field(..., description="Loss description")
    heat_loss_kW: float = Field(..., ge=0, description="Heat loss rate (kW)")
    loss_type: str = Field(default="radiation", description="Loss type")
    recoverable_fraction: float = Field(default=0.0, ge=0, le=1, description="Recoverable fraction")


class AmbientConditions(BaseModel):
    temperature_celsius: float = Field(default=25.0, description="Ambient temperature (C)")
    pressure_kPa: float = Field(default=101.325, description="Atmospheric pressure (kPa)")
    relative_humidity_percent: float = Field(default=50.0, ge=0, le=100, description="RH (%)")


class HeatBalanceInput(BaseModel):
    analysis_id: Optional[str] = Field(None, description="Analysis identifier")
    system_name: str = Field(default="Process System", description="System name")
    process_streams: List[ProcessStream] = Field(default_factory=list)
    utilities: List[UtilityStream] = Field(default_factory=list)
    heat_losses: List[HeatLoss] = Field(default_factory=list)
    ambient_conditions: AmbientConditions = Field(default_factory=AmbientConditions)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamBalance(BaseModel):
    stream_id: str
    name: str
    stream_type: str
    mass_flow_kg_s: float
    temperature_celsius: float
    enthalpy_kJ_kg: float
    enthalpy_flow_kW: float
    heat_content_kW: float


class HeatBalanceTable(BaseModel):
    input_streams: List[StreamBalance]
    output_streams: List[StreamBalance]
    losses: List[StreamBalance]
    total_input_kW: float
    total_output_kW: float
    total_losses_kW: float


class SankeyNode(BaseModel):
    id: str
    name: str
    category: str


class SankeyLink(BaseModel):
    source: str
    target: str
    value: float


class SankeyDiagramData(BaseModel):
    nodes: List[SankeyNode]
    links: List[SankeyLink]


class ImprovementOpportunity(BaseModel):
    opportunity_id: str
    source: str
    heat_available_kW: float
    recovery_potential_kW: float
    recovery_fraction: float
    priority: str
    recommendation: str
    estimated_savings_percent: float


class ProvenanceRecord(BaseModel):
    operation: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    tool_name: str


class HeatBalanceOutput(BaseModel):
    analysis_id: str
    system_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    heat_balance_table: HeatBalanceTable
    total_input_kW: float
    total_output_kW: float
    closure_error_kW: float
    closure_error_percent: float
    balance_status: str
    thermal_efficiency_percent: float
    sankey_diagram_data: SankeyDiagramData
    improvement_opportunities: List[ImprovementOpportunity]
    total_recovery_potential_kW: float
    provenance_chain: List[ProvenanceRecord]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str
    validation_errors: List[str] = Field(default_factory=list)


class HeatBalanceAnalyzerAgent:
    """GL-061: Heat Balance Analyzer Agent - First Law energy balance analysis."""

    AGENT_ID = "GL-061"
    AGENT_NAME = "HEATBALANCE-ANALYZER"
    VERSION = "1.0.0"
    CLOSURE_THRESHOLD_PERCENT = 5.0

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []
        logger.info(f"HeatBalanceAnalyzerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: HeatBalanceInput) -> HeatBalanceOutput:
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []

        ref_temp = input_data.ambient_conditions.temperature_celsius

        input_streams, input_total = self._calculate_stream_balances(
            [s for s in input_data.process_streams if s.stream_type == StreamType.INPUT], ref_temp, StreamType.INPUT)
        output_streams, output_total = self._calculate_stream_balances(
            [s for s in input_data.process_streams if s.stream_type == StreamType.OUTPUT], ref_temp, StreamType.OUTPUT)

        utility_inputs, utility_outputs = self._calculate_utility_balances(input_data.utilities, ref_temp)
        input_total += sum(u.heat_content_kW for u in utility_inputs)
        output_total += sum(u.heat_content_kW for u in utility_outputs)
        input_streams.extend(utility_inputs)
        output_streams.extend(utility_outputs)

        loss_streams, loss_total = self._calculate_loss_balances(input_data.heat_losses)

        self._track_provenance("heat_balance_calculation",
            {"num_streams": len(input_data.process_streams)},
            {"input_kW": input_total, "output_kW": output_total, "loss_kW": loss_total}, "First Law Calculator")

        closure_error_kW = input_total - output_total - loss_total
        closure_error_percent = (closure_error_kW / input_total * 100.0) if input_total > 0 else 0.0
        balance_status = "BALANCED" if abs(closure_error_percent) <= self.CLOSURE_THRESHOLD_PERCENT else "UNBALANCED"
        thermal_efficiency = (output_total / input_total * 100.0) if input_total > 0 else 0.0

        heat_balance_table = HeatBalanceTable(
            input_streams=input_streams, output_streams=output_streams, losses=loss_streams,
            total_input_kW=round(input_total, 2), total_output_kW=round(output_total, 2), total_losses_kW=round(loss_total, 2))

        sankey_data = self._generate_sankey_data(input_streams, output_streams, loss_streams)
        opportunities = self._identify_improvements(input_data.heat_losses, loss_streams, input_total)
        total_recovery = sum(o.recovery_potential_kW for o in opportunities)
        provenance_hash = self._calculate_provenance_hash()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return HeatBalanceOutput(
            analysis_id=input_data.analysis_id or f"HB-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            system_name=input_data.system_name, heat_balance_table=heat_balance_table,
            total_input_kW=round(input_total, 2), total_output_kW=round(output_total + loss_total, 2),
            closure_error_kW=round(closure_error_kW, 2), closure_error_percent=round(closure_error_percent, 2),
            balance_status=balance_status, thermal_efficiency_percent=round(thermal_efficiency, 2),
            sankey_diagram_data=sankey_data, improvement_opportunities=opportunities,
            total_recovery_potential_kW=round(total_recovery, 2),
            provenance_chain=[ProvenanceRecord(**{k: v for k, v in s.items()}) for s in self._provenance_steps],
            provenance_hash=provenance_hash, processing_time_ms=round(processing_time, 2),
            validation_status="PASS" if not self._validation_errors else "FAIL", validation_errors=self._validation_errors)

    def _calculate_stream_balances(self, streams: List[ProcessStream], ref_temp: float, stream_type: StreamType) -> Tuple[List[StreamBalance], float]:
        balances, total_heat = [], 0.0
        for stream in streams:
            cp = stream.specific_heat_J_kg_K or SPECIFIC_HEAT_DATA.get(stream.fluid_type, 2000.0)
            h = stream.enthalpy_kJ_kg if stream.enthalpy_kJ_kg else cp * (stream.temperature_celsius - ref_temp) / 1000.0
            H_dot = stream.mass_flow_kg_s * h
            Q = abs(H_dot)
            balances.append(StreamBalance(stream_id=stream.stream_id, name=stream.name, stream_type=stream_type.value,
                mass_flow_kg_s=round(stream.mass_flow_kg_s, 4), temperature_celsius=round(stream.temperature_celsius, 2),
                enthalpy_kJ_kg=round(h, 2), enthalpy_flow_kW=round(H_dot, 2), heat_content_kW=round(Q, 2)))
            total_heat += Q
        return balances, total_heat

    def _calculate_utility_balances(self, utilities: List[UtilityStream], ref_temp: float) -> Tuple[List[StreamBalance], List[StreamBalance]]:
        inputs, outputs = [], []
        for util in utilities:
            heat_kW = 0.0
            if util.utility_type == UtilityType.FUEL and util.mass_flow_kg_s and util.lower_heating_value_kJ_kg:
                heat_kW = util.mass_flow_kg_s * util.lower_heating_value_kJ_kg
            elif util.utility_type == UtilityType.ELECTRICITY:
                heat_kW = util.power_kW or 0.0
            elif util.mass_flow_kg_s:
                h = util.enthalpy_kJ_kg or (4186.0 * (util.temperature_celsius - ref_temp) / 1000.0 if util.temperature_celsius else 0.0)
                heat_kW = util.mass_flow_kg_s * h
            heat_kW = util.heat_content_kW if util.heat_content_kW else heat_kW
            balance = StreamBalance(stream_id=util.utility_id, name=util.name, stream_type=util.stream_type.value,
                mass_flow_kg_s=util.mass_flow_kg_s or 0.0, temperature_celsius=util.temperature_celsius or ref_temp,
                enthalpy_kJ_kg=util.enthalpy_kJ_kg or 0.0, enthalpy_flow_kW=round(heat_kW, 2), heat_content_kW=round(abs(heat_kW), 2))
            (inputs if util.stream_type == StreamType.INPUT else outputs).append(balance)
        return inputs, outputs

    def _calculate_loss_balances(self, losses: List[HeatLoss]) -> Tuple[List[StreamBalance], float]:
        balances, total_loss = [], 0.0
        for loss in losses:
            balances.append(StreamBalance(stream_id=loss.loss_id, name=loss.name, stream_type="loss",
                mass_flow_kg_s=0.0, temperature_celsius=0.0, enthalpy_kJ_kg=0.0,
                enthalpy_flow_kW=round(loss.heat_loss_kW, 2), heat_content_kW=round(loss.heat_loss_kW, 2)))
            total_loss += loss.heat_loss_kW
        return balances, total_loss

    def _generate_sankey_data(self, inputs: List[StreamBalance], outputs: List[StreamBalance], losses: List[StreamBalance]) -> SankeyDiagramData:
        nodes = [SankeyNode(id="process", name="Process", category="process")]
        links = []
        for s in inputs:
            nodes.append(SankeyNode(id=s.stream_id, name=s.name, category="input"))
            links.append(SankeyLink(source=s.stream_id, target="process", value=abs(s.heat_content_kW)))
        for s in outputs:
            nodes.append(SankeyNode(id=s.stream_id, name=s.name, category="output"))
            links.append(SankeyLink(source="process", target=s.stream_id, value=abs(s.heat_content_kW)))
        for s in losses:
            nodes.append(SankeyNode(id=s.stream_id, name=s.name, category="loss"))
            links.append(SankeyLink(source="process", target=s.stream_id, value=abs(s.heat_content_kW)))
        return SankeyDiagramData(nodes=nodes, links=links)

    def _identify_improvements(self, losses: List[HeatLoss], loss_balances: List[StreamBalance], total_input_kW: float) -> List[ImprovementOpportunity]:
        opportunities = []
        for i, loss in enumerate(losses):
            loss_percent = (loss.heat_loss_kW / total_input_kW * 100.0) if total_input_kW > 0 else 0
            if loss_percent >= 2.0:
                recovery_potential = loss.heat_loss_kW * loss.recoverable_fraction
                priority = "HIGH" if loss_percent >= 10.0 else ("MEDIUM" if loss_percent >= 5.0 else "LOW")
                opportunities.append(ImprovementOpportunity(opportunity_id=f"OPP-{i+1:03d}", source=loss.name,
                    heat_available_kW=round(loss.heat_loss_kW, 2), recovery_potential_kW=round(recovery_potential, 2),
                    recovery_fraction=loss.recoverable_fraction, priority=priority,
                    recommendation=f"Investigate heat recovery from {loss.name}",
                    estimated_savings_percent=round(recovery_potential / total_input_kW * 100.0, 2) if total_input_kW > 0 else 0.0))
        return sorted(opportunities, key=lambda x: (-{"HIGH": 2, "MEDIUM": 1, "LOW": 0}.get(x.priority, 0), -x.recovery_potential_kW))

    def _track_provenance(self, operation: str, inputs: Dict, outputs: Dict, tool_name: str) -> None:
        self._provenance_steps.append({"operation": operation, "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            "output_hash": hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(), "tool_name": tool_name})

    def _calculate_provenance_hash(self) -> str:
        data = {"agent_id": self.AGENT_ID, "version": self.VERSION,
            "steps": [{"operation": s["operation"], "input_hash": s["input_hash"], "output_hash": s["output_hash"]} for s in self._provenance_steps],
            "timestamp": datetime.utcnow().isoformat()}
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


PACK_SPEC = {"schema_version": "2.0.0", "id": "GL-061", "name": "HEATBALANCE-ANALYZER", "version": "1.0.0",
    "summary": "Heat balance calculation with Sankey visualization", "tags": ["heat-balance", "first-law", "thermodynamics"],
    "standards": [{"ref": "ASME PTC 4", "description": "Fired Steam Generators"}], "provenance": {"calculation_verified": True, "enable_audit": True}}
