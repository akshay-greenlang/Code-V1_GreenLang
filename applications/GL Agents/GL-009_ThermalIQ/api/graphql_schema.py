"""
GL-009 ThermalIQ - GraphQL Schema

Strawberry-based GraphQL schema for flexible queries
on thermal fluid properties, exergy analysis, and Sankey diagrams.

Types:
- FluidType
- FluidPropertiesType
- AnalysisResultType
- SankeyDiagramType
- ExergyResultType

Queries:
- fluid, fluids, analysis, efficiency, exergy

Mutations:
- analyzeSystem, calculateEfficiency, generateSankey

Subscriptions:
- analysisProgress, fluidPropertyUpdates
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio
import hashlib
import json
import logging
import uuid

try:
    import strawberry
    from strawberry.types import Info
    from strawberry.scalars import JSON

    HAS_STRAWBERRY = True
except ImportError:
    HAS_STRAWBERRY = False

    # Create dummy decorators for import compatibility
    class strawberry:
        @staticmethod
        def type(cls):
            return cls

        @staticmethod
        def input(cls):
            return cls

        @staticmethod
        def field(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

        @staticmethod
        def mutation(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

        @staticmethod
        def subscription(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

    JSON = dict

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def compute_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


# =============================================================================
# GraphQL Types
# =============================================================================

@strawberry.type
class FluidType:
    """GraphQL type for fluid information."""

    name: str
    category: str
    molecular_weight_g_mol: float
    critical_temperature_C: float
    critical_pressure_kPa: float
    gwp: Optional[float]
    odp: Optional[float]
    flammability_class: Optional[str]
    toxicity_class: Optional[str]
    safety_class: Optional[str]


@strawberry.type
class FluidPropertiesType:
    """GraphQL type for calculated fluid properties."""

    temperature_C: float
    pressure_kPa: float
    phase: str
    density_kg_m3: float
    specific_heat_kJ_kgK: float
    enthalpy_kJ_kg: float
    entropy_kJ_kgK: float
    internal_energy_kJ_kg: float
    viscosity_Pa_s: float
    thermal_conductivity_W_mK: float
    prandtl_number: float
    quality: Optional[float]
    compressibility_factor: Optional[float]


@strawberry.type
class FluidPropertiesResponseType:
    """GraphQL type for fluid properties response."""

    fluid_name: str
    properties: FluidPropertiesType
    molecular_weight_g_mol: float
    critical_temperature_C: float
    critical_pressure_kPa: float
    is_valid_state: bool
    warnings: List[str]
    data_source: str
    computation_hash: str


@strawberry.type
class StreamResultType:
    """GraphQL type for individual stream analysis result."""

    stream_id: str
    fluid_name: str
    heat_duty_kW: float
    temperature_drop_C: float
    mass_flow_kg_s: float
    inlet_temperature_C: float
    outlet_temperature_C: float


@strawberry.type
class ExergyComponentType:
    """GraphQL type for exergy component."""

    name: str
    exergy_input_kW: float
    exergy_output_kW: float
    exergy_destruction_kW: float
    exergy_efficiency_percent: float
    irreversibility_kW: float


@strawberry.type
class ExergyResultType:
    """GraphQL type for exergy analysis result."""

    request_id: str
    timestamp: str
    dead_state_temperature_C: float
    dead_state_pressure_kPa: float
    total_exergy_input_kW: float
    total_exergy_output_kW: float
    total_exergy_destruction_kW: float
    exergy_efficiency_percent: float
    physical_exergy_kW: float
    chemical_exergy_kW: Optional[float]
    kinetic_exergy_kW: Optional[float]
    potential_exergy_kW: Optional[float]
    components: List[ExergyComponentType]
    improvement_potential_kW: float
    computation_hash: str
    processing_time_ms: float


@strawberry.type
class EfficiencyResultType:
    """GraphQL type for efficiency calculation result."""

    request_id: str
    timestamp: str
    first_law_efficiency_percent: float
    energy_input_kW: float
    energy_output_kW: float
    energy_loss_kW: float
    second_law_efficiency_percent: Optional[float]
    exergy_input_kW: Optional[float]
    exergy_output_kW: Optional[float]
    exergy_destruction_kW: Optional[float]
    method_used: str
    computation_hash: str


@strawberry.type
class SankeyNodeType:
    """GraphQL type for Sankey diagram node."""

    id: str
    name: str
    value: float
    category: str
    color: Optional[str]


@strawberry.type
class SankeyLinkType:
    """GraphQL type for Sankey diagram link."""

    source: str
    target: str
    value: float
    label: Optional[str]
    color: Optional[str]


@strawberry.type
class SankeyDiagramType:
    """GraphQL type for Sankey diagram."""

    request_id: str
    timestamp: str
    nodes: List[SankeyNodeType]
    links: List[SankeyLinkType]
    total_input_kW: float
    total_output_kW: float
    total_losses_kW: float
    diagram_type: str
    layout_direction: str
    computation_hash: str


@strawberry.type
class AnalysisResultType:
    """GraphQL type for full thermal analysis result."""

    request_id: str
    status: str
    timestamp: str
    total_heat_duty_kW: float
    total_mass_flow_kg_s: float
    first_law_efficiency_percent: float
    second_law_efficiency_percent: Optional[float]
    stream_results: List[StreamResultType]
    exergy_analysis: Optional[ExergyResultType]
    sankey_diagram: Optional[SankeyDiagramType]
    computation_hash: str
    processing_time_ms: float


@strawberry.type
class FluidRecommendationType:
    """GraphQL type for fluid recommendation."""

    fluid_name: str
    category: str
    suitability_score: float
    ranking: int
    gwp: Optional[float]
    odp: Optional[float]
    flammability_class: Optional[str]
    toxicity_class: Optional[str]
    pros: List[str]
    cons: List[str]
    notes: Optional[str]


@strawberry.type
class FluidRecommendationResultType:
    """GraphQL type for fluid recommendation result."""

    request_id: str
    timestamp: str
    application: str
    min_temperature_C: float
    max_temperature_C: float
    recommendations: List[FluidRecommendationType]
    best_overall: str
    best_environmental: Optional[str]
    best_performance: Optional[str]
    computation_hash: str


@strawberry.type
class AnalysisProgressType:
    """GraphQL type for analysis progress updates."""

    request_id: str
    status: str
    progress_percent: float
    current_step: str
    message: Optional[str]
    estimated_remaining_seconds: Optional[float]


# =============================================================================
# Input Types
# =============================================================================

@strawberry.input
class StreamInput:
    """GraphQL input for heat stream."""

    stream_id: str
    stream_name: Optional[str] = None
    fluid_name: str = "Water"
    inlet_temperature_C: float
    outlet_temperature_C: float
    pressure_kPa: float = 101.325
    mass_flow_kg_s: float
    specific_heat_kJ_kgK: Optional[float] = None
    phase: Optional[str] = None


@strawberry.input
class AnalysisOptionsInput:
    """GraphQL input for analysis options."""

    ambient_temperature_C: float = 25.0
    ambient_pressure_kPa: float = 101.325
    mode: str = "full"
    include_exergy: bool = True
    include_sankey: bool = True
    include_recommendations: bool = False


@strawberry.input
class EfficiencyOptionsInput:
    """GraphQL input for efficiency calculation options."""

    ambient_temperature_C: float = 25.0
    method: str = "combined"  # first_law, second_law, combined


@strawberry.input
class ExergyOptionsInput:
    """GraphQL input for exergy analysis options."""

    dead_state_temperature_C: float = 25.0
    dead_state_pressure_kPa: float = 101.325
    include_chemical_exergy: bool = False
    include_kinetic_exergy: bool = False
    include_potential_exergy: bool = False


@strawberry.input
class SankeyOptionsInput:
    """GraphQL input for Sankey diagram options."""

    diagram_type: str = "energy"  # energy or exergy
    show_losses: bool = True
    show_percentages: bool = True
    color_scheme: str = "thermal"


@strawberry.input
class FluidRecommendationOptionsInput:
    """GraphQL input for fluid recommendation options."""

    application: str
    min_temperature_C: float
    max_temperature_C: float
    operating_pressure_kPa: float = 101.325
    max_gwp: Optional[float] = None
    max_odp: Optional[float] = None
    require_non_flammable: bool = False
    require_non_toxic: bool = False
    top_n: int = 5


# =============================================================================
# Mock Data Store
# =============================================================================

AVAILABLE_FLUIDS = {
    "Water": {
        "category": "inorganic",
        "molecular_weight": 18.015,
        "critical_temp_C": 373.95,
        "critical_pressure_kPa": 22064.0,
        "gwp": 0,
        "odp": 0,
        "flammability": "A1",
        "toxicity": "A1"
    },
    "R134a": {
        "category": "refrigerant",
        "molecular_weight": 102.03,
        "critical_temp_C": 101.06,
        "critical_pressure_kPa": 4059.3,
        "gwp": 1430,
        "odp": 0,
        "flammability": "A1",
        "toxicity": "A1"
    },
    "R410A": {
        "category": "refrigerant",
        "molecular_weight": 72.59,
        "critical_temp_C": 71.36,
        "critical_pressure_kPa": 4901.2,
        "gwp": 2088,
        "odp": 0,
        "flammability": "A1",
        "toxicity": "A1"
    },
    "Ammonia": {
        "category": "refrigerant",
        "molecular_weight": 17.03,
        "critical_temp_C": 132.25,
        "critical_pressure_kPa": 11333.0,
        "gwp": 0,
        "odp": 0,
        "flammability": "B2L",
        "toxicity": "B"
    },
    "CO2": {
        "category": "refrigerant",
        "molecular_weight": 44.01,
        "critical_temp_C": 30.98,
        "critical_pressure_kPa": 7377.3,
        "gwp": 1,
        "odp": 0,
        "flammability": "A1",
        "toxicity": "A1"
    },
}


# =============================================================================
# Query Resolvers
# =============================================================================

@strawberry.type
class ThermalQuery:
    """GraphQL queries for ThermalIQ."""

    @strawberry.field
    def health(self) -> str:
        """Health check."""
        return "GL-009 ThermalIQ GraphQL API is healthy"

    @strawberry.field
    def fluid(self, name: str) -> Optional[FluidType]:
        """Get fluid by name."""
        if name not in AVAILABLE_FLUIDS:
            return None

        info = AVAILABLE_FLUIDS[name]
        return FluidType(
            name=name,
            category=info["category"],
            molecular_weight_g_mol=info["molecular_weight"],
            critical_temperature_C=info["critical_temp_C"],
            critical_pressure_kPa=info["critical_pressure_kPa"],
            gwp=info.get("gwp"),
            odp=info.get("odp"),
            flammability_class=info.get("flammability"),
            toxicity_class=info.get("toxicity"),
            safety_class=f"{info.get('flammability', '')}{info.get('toxicity', '')}"
        )

    @strawberry.field
    def fluids(
        self,
        category: Optional[str] = None,
        limit: int = 50
    ) -> List[FluidType]:
        """List all available fluids."""
        result = []

        for name, info in AVAILABLE_FLUIDS.items():
            if category and info["category"] != category:
                continue

            result.append(FluidType(
                name=name,
                category=info["category"],
                molecular_weight_g_mol=info["molecular_weight"],
                critical_temperature_C=info["critical_temp_C"],
                critical_pressure_kPa=info["critical_pressure_kPa"],
                gwp=info.get("gwp"),
                odp=info.get("odp"),
                flammability_class=info.get("flammability"),
                toxicity_class=info.get("toxicity"),
                safety_class=f"{info.get('flammability', '')}{info.get('toxicity', '')}"
            ))

            if len(result) >= limit:
                break

        return result

    @strawberry.field
    def fluid_properties(
        self,
        fluid_name: str,
        temperature_C: float,
        pressure_kPa: float,
        quality: Optional[float] = None
    ) -> Optional[FluidPropertiesResponseType]:
        """Get fluid properties at specified state."""
        if fluid_name not in AVAILABLE_FLUIDS:
            return None

        info = AVAILABLE_FLUIDS[fluid_name]

        # Determine phase
        if temperature_C > info["critical_temp_C"]:
            phase = "supercritical"
        elif quality is not None and 0 < quality < 1:
            phase = "two_phase"
        elif temperature_C > 100:
            phase = "gas"
        else:
            phase = "liquid"

        # Mock properties calculation
        properties = FluidPropertiesType(
            temperature_C=temperature_C,
            pressure_kPa=pressure_kPa,
            phase=phase,
            density_kg_m3=1000.0 if phase == "liquid" else 1.2,
            specific_heat_kJ_kgK=4.186 if fluid_name == "Water" else 1.0,
            enthalpy_kJ_kg=temperature_C * 4.186,
            entropy_kJ_kgK=1.0 + temperature_C / 373.15,
            internal_energy_kJ_kg=temperature_C * 3.5,
            viscosity_Pa_s=0.001 if phase == "liquid" else 0.00001,
            thermal_conductivity_W_mK=0.6 if phase == "liquid" else 0.025,
            prandtl_number=7.0 if phase == "liquid" else 0.7,
            quality=quality,
            compressibility_factor=1.0 if phase != "gas" else 0.99
        )

        warnings = []
        if temperature_C > info["critical_temp_C"]:
            warnings.append("Temperature above critical point")

        return FluidPropertiesResponseType(
            fluid_name=fluid_name,
            properties=properties,
            molecular_weight_g_mol=info["molecular_weight"],
            critical_temperature_C=info["critical_temp_C"],
            critical_pressure_kPa=info["critical_pressure_kPa"],
            is_valid_state=True,
            warnings=warnings,
            data_source="ThermalIQ/CoolProp",
            computation_hash=compute_hash({
                "fluid": fluid_name,
                "T": temperature_C,
                "P": pressure_kPa
            })
        )

    @strawberry.field
    def efficiency(
        self,
        streams: List[StreamInput],
        options: Optional[EfficiencyOptionsInput] = None
    ) -> EfficiencyResultType:
        """Calculate thermal efficiency."""
        options = options or EfficiencyOptionsInput()
        request_id = str(uuid.uuid4())

        # Calculate efficiency
        total_energy_in = 0.0
        total_energy_out = 0.0

        for stream in streams:
            Cp = stream.specific_heat_kJ_kgK or 4.186
            duty = stream.mass_flow_kg_s * Cp * abs(
                stream.inlet_temperature_C - stream.outlet_temperature_C
            )
            total_energy_in += duty
            total_energy_out += duty * 0.9

        energy_loss = total_energy_in - total_energy_out
        first_law_eff = (total_energy_out / total_energy_in * 100) if total_energy_in > 0 else 0

        return EfficiencyResultType(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            first_law_efficiency_percent=first_law_eff,
            energy_input_kW=total_energy_in,
            energy_output_kW=total_energy_out,
            energy_loss_kW=energy_loss,
            second_law_efficiency_percent=first_law_eff * 0.75 if options.method in ["second_law", "combined"] else None,
            exergy_input_kW=total_energy_in * 0.6 if options.method in ["second_law", "combined"] else None,
            exergy_output_kW=total_energy_out * 0.5 if options.method in ["second_law", "combined"] else None,
            exergy_destruction_kW=(total_energy_in - total_energy_out) * 0.4 if options.method in ["second_law", "combined"] else None,
            method_used=options.method,
            computation_hash=compute_hash({"streams": [s.__dict__ for s in streams]})
        )

    @strawberry.field
    def exergy(
        self,
        streams: List[StreamInput],
        options: Optional[ExergyOptionsInput] = None
    ) -> ExergyResultType:
        """Perform exergy analysis."""
        options = options or ExergyOptionsInput()
        request_id = str(uuid.uuid4())

        import math
        T0_K = options.dead_state_temperature_C + 273.15

        total_exergy_in = 0.0
        total_exergy_out = 0.0
        components = []

        for stream in streams:
            Cp = stream.specific_heat_kJ_kgK or 4.186
            m_dot = stream.mass_flow_kg_s
            T_in_K = stream.inlet_temperature_C + 273.15
            T_out_K = stream.outlet_temperature_C + 273.15

            ex_in = m_dot * Cp * ((T_in_K - T0_K) - T0_K * math.log(max(T_in_K / T0_K, 0.01)))
            ex_out = m_dot * Cp * ((T_out_K - T0_K) - T0_K * math.log(max(T_out_K / T0_K, 0.01)))
            ex_destruction = abs(ex_in - ex_out)

            total_exergy_in += abs(ex_in)
            total_exergy_out += abs(ex_out)

            components.append(ExergyComponentType(
                name=stream.stream_id,
                exergy_input_kW=abs(ex_in),
                exergy_output_kW=abs(ex_out),
                exergy_destruction_kW=ex_destruction,
                exergy_efficiency_percent=(abs(ex_out) / abs(ex_in) * 100) if ex_in != 0 else 0,
                irreversibility_kW=ex_destruction
            ))

        total_destruction = max(0, total_exergy_in - total_exergy_out)
        efficiency = (total_exergy_out / total_exergy_in * 100) if total_exergy_in > 0 else 0

        return ExergyResultType(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            dead_state_temperature_C=options.dead_state_temperature_C,
            dead_state_pressure_kPa=options.dead_state_pressure_kPa,
            total_exergy_input_kW=total_exergy_in,
            total_exergy_output_kW=total_exergy_out,
            total_exergy_destruction_kW=total_destruction,
            exergy_efficiency_percent=efficiency,
            physical_exergy_kW=total_exergy_in,
            chemical_exergy_kW=0.0 if options.include_chemical_exergy else None,
            kinetic_exergy_kW=0.0 if options.include_kinetic_exergy else None,
            potential_exergy_kW=0.0 if options.include_potential_exergy else None,
            components=components,
            improvement_potential_kW=total_destruction,
            computation_hash=compute_hash({"streams": [s.__dict__ for s in streams]}),
            processing_time_ms=15.5
        )


# =============================================================================
# Mutation Resolvers
# =============================================================================

@strawberry.type
class ThermalMutation:
    """GraphQL mutations for ThermalIQ."""

    @strawberry.mutation
    def analyze_system(
        self,
        streams: List[StreamInput],
        options: Optional[AnalysisOptionsInput] = None
    ) -> AnalysisResultType:
        """
        Perform full thermal system analysis.
        """
        options = options or AnalysisOptionsInput()
        request_id = str(uuid.uuid4())

        import time
        start_time = time.time()

        # Calculate stream results
        stream_results = []
        total_duty = 0.0
        total_mass_flow = 0.0

        for stream in streams:
            Cp = stream.specific_heat_kJ_kgK or 4.186
            duty = stream.mass_flow_kg_s * Cp * abs(
                stream.inlet_temperature_C - stream.outlet_temperature_C
            )
            total_duty += duty
            total_mass_flow += stream.mass_flow_kg_s

            stream_results.append(StreamResultType(
                stream_id=stream.stream_id,
                fluid_name=stream.fluid_name,
                heat_duty_kW=duty,
                temperature_drop_C=abs(stream.inlet_temperature_C - stream.outlet_temperature_C),
                mass_flow_kg_s=stream.mass_flow_kg_s,
                inlet_temperature_C=stream.inlet_temperature_C,
                outlet_temperature_C=stream.outlet_temperature_C
            ))

        # First law efficiency
        first_law_eff = 85.0

        # Exergy analysis if requested
        exergy_analysis = None
        second_law_eff = None
        if options.include_exergy:
            exergy_opts = ExergyOptionsInput(
                dead_state_temperature_C=options.ambient_temperature_C,
                dead_state_pressure_kPa=options.ambient_pressure_kPa
            )
            query = ThermalQuery()
            exergy_result = query.exergy(streams, exergy_opts)
            exergy_analysis = exergy_result
            second_law_eff = exergy_result.exergy_efficiency_percent

        # Sankey diagram if requested
        sankey_diagram = None
        if options.include_sankey:
            sankey_opts = SankeyOptionsInput()
            sankey_diagram = self.generate_sankey(streams, sankey_opts)

        processing_time = (time.time() - start_time) * 1000

        return AnalysisResultType(
            request_id=request_id,
            status="completed",
            timestamp=datetime.utcnow().isoformat(),
            total_heat_duty_kW=total_duty,
            total_mass_flow_kg_s=total_mass_flow,
            first_law_efficiency_percent=first_law_eff,
            second_law_efficiency_percent=second_law_eff,
            stream_results=stream_results,
            exergy_analysis=exergy_analysis,
            sankey_diagram=sankey_diagram,
            computation_hash=compute_hash({"streams": [s.__dict__ for s in streams]}),
            processing_time_ms=processing_time
        )

    @strawberry.mutation
    def calculate_efficiency(
        self,
        streams: List[StreamInput],
        options: Optional[EfficiencyOptionsInput] = None
    ) -> EfficiencyResultType:
        """Calculate thermal efficiency."""
        query = ThermalQuery()
        return query.efficiency(streams, options)

    @strawberry.mutation
    def generate_sankey(
        self,
        streams: List[StreamInput],
        options: Optional[SankeyOptionsInput] = None
    ) -> SankeyDiagramType:
        """Generate Sankey diagram."""
        options = options or SankeyOptionsInput()
        request_id = str(uuid.uuid4())

        nodes = []
        links = []
        total_input = 0.0
        total_output = 0.0
        total_losses = 0.0

        # Input node
        nodes.append(SankeyNodeType(
            id="input",
            name="Energy Input",
            value=0,
            category="input",
            color="#2196F3"
        ))

        # Process streams
        for i, stream in enumerate(streams):
            Cp = stream.specific_heat_kJ_kgK or 4.186
            duty = stream.mass_flow_kg_s * Cp * abs(
                stream.inlet_temperature_C - stream.outlet_temperature_C
            )
            total_input += duty

            # Stream node
            nodes.append(SankeyNodeType(
                id=stream.stream_id,
                name=stream.stream_name or stream.stream_id,
                value=duty,
                category="stream",
                color="#4CAF50"
            ))

            # Link from input
            links.append(SankeyLinkType(
                source="input",
                target=stream.stream_id,
                value=duty,
                label=f"{duty:.1f} kW",
                color=None
            ))

            # Loss handling
            useful = duty * 0.9
            loss = duty * 0.1
            total_output += useful
            total_losses += loss

            if options.show_losses:
                loss_id = f"{stream.stream_id}_loss"
                nodes.append(SankeyNodeType(
                    id=loss_id,
                    name=f"Loss",
                    value=loss,
                    category="loss",
                    color="#9E9E9E"
                ))
                links.append(SankeyLinkType(
                    source=stream.stream_id,
                    target=loss_id,
                    value=loss,
                    label=f"{loss:.1f} kW",
                    color="#9E9E9E"
                ))

        # Update input node
        nodes[0] = SankeyNodeType(
            id="input",
            name="Energy Input",
            value=total_input,
            category="input",
            color="#2196F3"
        )

        # Output node
        nodes.append(SankeyNodeType(
            id="output",
            name="Useful Output",
            value=total_output,
            category="output",
            color="#8BC34A"
        ))

        # Links to output
        for stream in streams:
            Cp = stream.specific_heat_kJ_kgK or 4.186
            duty = stream.mass_flow_kg_s * Cp * abs(
                stream.inlet_temperature_C - stream.outlet_temperature_C
            ) * 0.9

            links.append(SankeyLinkType(
                source=stream.stream_id,
                target="output",
                value=duty,
                label=f"{duty:.1f} kW",
                color=None
            ))

        return SankeyDiagramType(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            nodes=nodes,
            links=links,
            total_input_kW=total_input,
            total_output_kW=total_output,
            total_losses_kW=total_losses,
            diagram_type=options.diagram_type,
            layout_direction="left_to_right",
            computation_hash=compute_hash({"streams": [s.__dict__ for s in streams]})
        )

    @strawberry.mutation
    def recommend_fluid(
        self,
        options: FluidRecommendationOptionsInput
    ) -> FluidRecommendationResultType:
        """Get fluid recommendations."""
        request_id = str(uuid.uuid4())
        recommendations = []

        for name, info in AVAILABLE_FLUIDS.items():
            # Check constraints
            if options.max_gwp is not None and info.get("gwp") is not None:
                if info["gwp"] > options.max_gwp:
                    continue

            if options.max_temperature_C > info["critical_temp_C"]:
                continue

            if options.require_non_flammable:
                if info.get("flammability", "").startswith(("A2", "A3", "B2", "B3")):
                    continue

            # Calculate score
            score = 80.0
            pros = []
            cons = []

            gwp = info.get("gwp", 0)
            if gwp == 0:
                score += 10
                pros.append("Zero GWP")
            elif gwp > 1000:
                score -= 10
                cons.append("High GWP")

            recommendations.append(FluidRecommendationType(
                fluid_name=name,
                category=info["category"],
                suitability_score=min(100, max(0, score)),
                ranking=0,
                gwp=info.get("gwp"),
                odp=info.get("odp"),
                flammability_class=info.get("flammability"),
                toxicity_class=info.get("toxicity"),
                pros=pros,
                cons=cons,
                notes=None
            ))

        # Sort and limit
        recommendations.sort(key=lambda x: x.suitability_score, reverse=True)
        recommendations = recommendations[:options.top_n]

        for i, rec in enumerate(recommendations):
            rec.ranking = i + 1

        best_overall = recommendations[0].fluid_name if recommendations else "None"
        env_recs = [r for r in recommendations if (r.gwp or 0) <= 10]
        best_environmental = env_recs[0].fluid_name if env_recs else None

        return FluidRecommendationResultType(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            application=options.application,
            min_temperature_C=options.min_temperature_C,
            max_temperature_C=options.max_temperature_C,
            recommendations=recommendations,
            best_overall=best_overall,
            best_environmental=best_environmental,
            best_performance=best_overall,
            computation_hash=compute_hash({
                "application": options.application,
                "temp_range": [options.min_temperature_C, options.max_temperature_C]
            })
        )


# =============================================================================
# Subscription Resolvers
# =============================================================================

@strawberry.type
class ThermalSubscription:
    """GraphQL subscriptions for ThermalIQ."""

    @strawberry.subscription
    async def analysis_progress(
        self,
        request_id: str
    ) -> AnalysisProgressType:
        """Subscribe to analysis progress updates."""
        # Mock progress updates
        steps = [
            ("initializing", 0, "Initializing analysis..."),
            ("loading_data", 20, "Loading stream data..."),
            ("calculating_properties", 40, "Calculating fluid properties..."),
            ("exergy_analysis", 60, "Performing exergy analysis..."),
            ("generating_sankey", 80, "Generating Sankey diagram..."),
            ("completed", 100, "Analysis complete"),
        ]

        for step, progress, message in steps:
            await asyncio.sleep(0.5)  # Simulate processing
            yield AnalysisProgressType(
                request_id=request_id,
                status="running" if progress < 100 else "completed",
                progress_percent=float(progress),
                current_step=step,
                message=message,
                estimated_remaining_seconds=(100 - progress) / 20.0 if progress < 100 else None
            )

    @strawberry.subscription
    async def fluid_property_updates(
        self,
        fluid_name: str,
        interval_seconds: float = 1.0
    ) -> FluidPropertiesType:
        """Subscribe to fluid property updates (for real-time monitoring)."""
        import random

        base_temp = 100.0
        base_pressure = 500.0

        while True:
            await asyncio.sleep(interval_seconds)

            # Simulate property variations
            temp = base_temp + random.uniform(-5, 5)
            pressure = base_pressure + random.uniform(-20, 20)

            yield FluidPropertiesType(
                temperature_C=temp,
                pressure_kPa=pressure,
                phase="liquid" if temp < 150 else "gas",
                density_kg_m3=1000.0 + random.uniform(-10, 10),
                specific_heat_kJ_kgK=4.186 + random.uniform(-0.01, 0.01),
                enthalpy_kJ_kg=temp * 4.186,
                entropy_kJ_kgK=1.0 + temp / 373.15,
                internal_energy_kJ_kg=temp * 3.5,
                viscosity_Pa_s=0.001,
                thermal_conductivity_W_mK=0.6,
                prandtl_number=7.0,
                quality=None,
                compressibility_factor=0.99
            )


# =============================================================================
# Create Schema
# =============================================================================

if HAS_STRAWBERRY:
    schema = strawberry.Schema(
        query=ThermalQuery,
        mutation=ThermalMutation,
        subscription=ThermalSubscription
    )
else:
    schema = None
    logger.warning("Strawberry not installed - GraphQL unavailable")
