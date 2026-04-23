"""
GL-003 UNIFIEDSTEAM - Physics-Based Explainer

Provides physics-based explanations for steam system calculations:
- Thermodynamic path tracing (P/T -> h -> spray-water requirement)
- Property calculation explanations with formula references
- Active constraint identification
- Mass/energy balance explanations

IMPORTANT: This module explains calculations but does NOT perform them.
All numeric calculations must be done by deterministic thermodynamic modules.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging
import uuid

logger = logging.getLogger(__name__)


class PropertyType(Enum):
    """Steam property types."""
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    ENTHALPY = "enthalpy"
    ENTROPY = "entropy"
    SPECIFIC_VOLUME = "specific_volume"
    QUALITY = "quality"
    SUPERHEAT = "superheat"
    FLOW_RATE = "flow_rate"


class ConstraintType(Enum):
    """Types of constraints in steam system optimization."""
    MASS_BALANCE = "mass_balance"
    ENERGY_BALANCE = "energy_balance"
    PRESSURE_LIMIT = "pressure_limit"
    TEMPERATURE_LIMIT = "temperature_limit"
    FLOW_LIMIT = "flow_limit"
    SUPERHEAT_MINIMUM = "superheat_minimum"
    QUALITY_MINIMUM = "quality_minimum"
    EQUIPMENT_CAPACITY = "equipment_capacity"
    SAFETY_ENVELOPE = "safety_envelope"


@dataclass
class ThermodynamicState:
    """Represents a thermodynamic state point."""
    state_id: str
    location: str  # e.g., "boiler_outlet", "header_hp", "desuperheater_inlet"
    timestamp: datetime

    # Primary properties
    pressure_psig: float
    temperature_f: float
    enthalpy_btu_lb: Optional[float] = None
    entropy_btu_lb_r: Optional[float] = None
    specific_volume_ft3_lb: Optional[float] = None

    # Quality/superheat
    quality: Optional[float] = None  # For wet steam (0-1)
    superheat_f: Optional[float] = None  # Degrees above saturation

    # Flow
    flow_rate_klb_hr: Optional[float] = None

    # Source signals
    source_signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "state_id": self.state_id,
            "location": self.location,
            "timestamp": self.timestamp.isoformat(),
            "pressure_psig": self.pressure_psig,
            "temperature_f": self.temperature_f,
            "enthalpy_btu_lb": self.enthalpy_btu_lb,
            "entropy_btu_lb_r": self.entropy_btu_lb_r,
            "specific_volume_ft3_lb": self.specific_volume_ft3_lb,
            "quality": self.quality,
            "superheat_f": self.superheat_f,
            "flow_rate_klb_hr": self.flow_rate_klb_hr,
            "source_signals": self.source_signals,
        }


@dataclass
class StateTransition:
    """Represents a transition between two thermodynamic states."""
    from_state: str
    to_state: str
    process_type: str  # "isenthalpic", "isentropic", "isobaric", "desuperheating"
    equipment: str  # Equipment causing the transition

    # Energy/mass changes
    enthalpy_change_btu_lb: Optional[float] = None
    mass_added_klb_hr: Optional[float] = None
    heat_transfer_btu_hr: Optional[float] = None

    # Explanation
    physical_description: str = ""
    formula_reference: str = ""

    def to_dict(self) -> Dict:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "process_type": self.process_type,
            "equipment": self.equipment,
            "enthalpy_change_btu_lb": self.enthalpy_change_btu_lb,
            "mass_added_klb_hr": self.mass_added_klb_hr,
            "heat_transfer_btu_hr": self.heat_transfer_btu_hr,
            "physical_description": self.physical_description,
            "formula_reference": self.formula_reference,
        }


@dataclass
class PhysicsTrace:
    """Complete physics trace for a recommendation."""
    trace_id: str
    timestamp: datetime
    recommendation_id: str

    # State sequence
    states: List[ThermodynamicState]
    transitions: List[StateTransition]

    # Driving signals
    primary_drivers: List[Dict[str, Any]]  # Signals that drove the calculation

    # Path description
    path_summary: str = ""
    technical_narrative: str = ""

    # Validation
    mass_balance_check: bool = True
    energy_balance_check: bool = True
    balance_residuals: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "recommendation_id": self.recommendation_id,
            "states": [s.to_dict() for s in self.states],
            "transitions": [t.to_dict() for t in self.transitions],
            "primary_drivers": self.primary_drivers,
            "path_summary": self.path_summary,
            "technical_narrative": self.technical_narrative,
            "mass_balance_check": self.mass_balance_check,
            "energy_balance_check": self.energy_balance_check,
            "balance_residuals": self.balance_residuals,
        }


@dataclass
class PropertyExplanation:
    """Explanation for a property calculation."""
    explanation_id: str
    property_name: str
    property_type: PropertyType
    calculated_value: float
    unit: str

    # Inputs
    input_properties: Dict[str, float]
    input_sources: Dict[str, str]  # Maps input to source signal/tag

    # Formula
    formula_name: str
    formula_description: str
    formula_latex: Optional[str] = None

    # Calculation path
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)

    # Uncertainty
    uncertainty_percent: Optional[float] = None
    uncertainty_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "explanation_id": self.explanation_id,
            "property_name": self.property_name,
            "property_type": self.property_type.value,
            "calculated_value": self.calculated_value,
            "unit": self.unit,
            "input_properties": self.input_properties,
            "input_sources": self.input_sources,
            "formula_name": self.formula_name,
            "formula_description": self.formula_description,
            "formula_latex": self.formula_latex,
            "intermediate_steps": self.intermediate_steps,
            "uncertainty_percent": self.uncertainty_percent,
            "uncertainty_sources": self.uncertainty_sources,
        }


@dataclass
class ActiveConstraint:
    """Represents an active constraint in optimization."""
    constraint_id: str
    constraint_type: ConstraintType
    constraint_name: str

    # Values
    limit_value: float
    actual_value: float
    margin: float  # How close to the limit
    unit: str

    # Binding status
    is_binding: bool  # True if at limit
    shadow_price: Optional[float] = None  # Economic impact of relaxing

    # Location
    affected_equipment: List[str] = field(default_factory=list)

    # Explanation
    physical_meaning: str = ""
    impact_description: str = ""
    relaxation_options: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type.value,
            "constraint_name": self.constraint_name,
            "limit_value": self.limit_value,
            "actual_value": self.actual_value,
            "margin": self.margin,
            "unit": self.unit,
            "is_binding": self.is_binding,
            "shadow_price": self.shadow_price,
            "affected_equipment": self.affected_equipment,
            "physical_meaning": self.physical_meaning,
            "impact_description": self.impact_description,
            "relaxation_options": self.relaxation_options,
        }


@dataclass
class BalanceExplanation:
    """Explanation for mass/energy balance."""
    explanation_id: str
    balance_type: str  # "mass", "energy"
    control_volume: str  # Equipment or system boundary
    timestamp: datetime

    # Inputs
    inflows: Dict[str, float]  # Stream name -> value
    outflows: Dict[str, float]
    accumulation: float = 0.0

    # Balance check
    imbalance: float = 0.0
    imbalance_percent: float = 0.0
    is_balanced: bool = True
    tolerance_used: float = 0.01

    # Units
    unit: str = ""

    # Explanation
    narrative: str = ""

    def to_dict(self) -> Dict:
        return {
            "explanation_id": self.explanation_id,
            "balance_type": self.balance_type,
            "control_volume": self.control_volume,
            "timestamp": self.timestamp.isoformat(),
            "inflows": self.inflows,
            "outflows": self.outflows,
            "accumulation": self.accumulation,
            "imbalance": self.imbalance,
            "imbalance_percent": self.imbalance_percent,
            "is_balanced": self.is_balanced,
            "tolerance_used": self.tolerance_used,
            "unit": self.unit,
            "narrative": self.narrative,
        }


class PhysicsExplainer:
    """
    Explains physics-based calculations for steam system optimization.

    Features:
    - Thermodynamic path tracing (P/T -> h -> spray-water)
    - Property calculation explanations with formulas
    - Active constraint identification
    - Mass/energy balance explanations

    IMPORTANT: This class explains calculations, it does NOT perform them.
    All numeric calculations must use deterministic thermodynamic modules.
    """

    def __init__(
        self,
        agent_id: str = "GL-003",
        steam_system_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent_id = agent_id
        self.steam_system_config = steam_system_config or {}

        # Location descriptions
        self._location_descriptions = {
            "boiler_outlet": "Boiler steam outlet (superheated)",
            "header_hp": "High-pressure steam header",
            "header_mp": "Medium-pressure steam header",
            "header_lp": "Low-pressure steam header",
            "prv_inlet": "Pressure reducing valve inlet",
            "prv_outlet": "Pressure reducing valve outlet",
            "desuperheater_inlet": "Desuperheater inlet",
            "desuperheater_outlet": "Desuperheater outlet (controlled temp)",
            "turbine_inlet": "Steam turbine inlet",
            "turbine_outlet": "Steam turbine exhaust",
            "trap_inlet": "Steam trap inlet",
            "condensate_return": "Condensate return line",
        }

        # Formula references
        self._formula_references = {
            "iapws97_enthalpy": "IAPWS-IF97: h = f(P, T) from region equations",
            "iapws97_saturation": "IAPWS-IF97: Saturation properties from Eq. (31)",
            "desuperheater_balance": "Energy balance: m_steam * h_in + m_water * h_water = m_out * h_out",
            "prv_isenthalpic": "Isenthalpic throttling: h_out = h_in",
            "turbine_isentropic": "Isentropic expansion: h_out = h_in - eta * (h_in - h_s)",
            "mass_balance": "Conservation of mass: sum(m_in) = sum(m_out) + dm/dt",
            "energy_balance": "First Law: sum(m_in * h_in) = sum(m_out * h_out) + Q + W",
        }

        # Process descriptions
        self._process_descriptions = {
            "isenthalpic": "Constant enthalpy process (throttling through valve)",
            "isentropic": "Constant entropy process (ideal expansion/compression)",
            "isobaric": "Constant pressure process (heating/cooling at fixed pressure)",
            "desuperheating": "Temperature reduction by spray water injection",
            "condensation": "Phase change from vapor to liquid",
            "flashing": "Pressure reduction causing partial vaporization",
        }

        # Trace storage
        self._traces: Dict[str, PhysicsTrace] = {}
        self._property_explanations: Dict[str, PropertyExplanation] = {}

        logger.info(f"PhysicsExplainer initialized: {agent_id}")

    def trace_thermodynamic_path(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        intermediate_states: List[Dict[str, Any]],
    ) -> PhysicsTrace:
        """
        Trace the thermodynamic path from inputs to outputs.

        Maps the chain: measured signals -> thermodynamic states -> calculated outputs
        Example: P/T measurement -> enthalpy lookup -> spray water requirement

        Args:
            inputs: Input measurements and their sources
            outputs: Calculated outputs
            intermediate_states: List of intermediate thermodynamic states

        Returns:
            PhysicsTrace with complete path explanation
        """
        trace_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Build state objects
        states = []
        for i, state_data in enumerate(intermediate_states):
            state = ThermodynamicState(
                state_id=f"state_{i}",
                location=state_data.get("location", f"point_{i}"),
                timestamp=timestamp,
                pressure_psig=state_data.get("pressure_psig", 0.0),
                temperature_f=state_data.get("temperature_f", 0.0),
                enthalpy_btu_lb=state_data.get("enthalpy_btu_lb"),
                entropy_btu_lb_r=state_data.get("entropy_btu_lb_r"),
                superheat_f=state_data.get("superheat_f"),
                flow_rate_klb_hr=state_data.get("flow_rate_klb_hr"),
                source_signals=state_data.get("source_signals", {}),
            )
            states.append(state)

        # Build transitions
        transitions = self._identify_transitions(states)

        # Identify primary drivers
        primary_drivers = self._identify_primary_drivers(inputs, outputs)

        # Generate path summary
        path_summary = self._generate_path_summary(states, transitions)
        technical_narrative = self._generate_technical_narrative(
            states, transitions, inputs, outputs
        )

        trace = PhysicsTrace(
            trace_id=trace_id,
            timestamp=timestamp,
            recommendation_id=outputs.get("recommendation_id", ""),
            states=states,
            transitions=transitions,
            primary_drivers=primary_drivers,
            path_summary=path_summary,
            technical_narrative=technical_narrative,
        )

        self._traces[trace_id] = trace
        logger.info(f"Created physics trace: {trace_id}")

        return trace

    def _identify_transitions(
        self,
        states: List[ThermodynamicState],
    ) -> List[StateTransition]:
        """Identify transitions between states."""
        transitions = []

        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]

            # Determine process type
            process_type = self._determine_process_type(from_state, to_state)

            # Calculate changes
            enthalpy_change = None
            if from_state.enthalpy_btu_lb and to_state.enthalpy_btu_lb:
                enthalpy_change = to_state.enthalpy_btu_lb - from_state.enthalpy_btu_lb

            # Determine equipment
            equipment = self._infer_equipment(
                from_state.location, to_state.location, process_type
            )

            transition = StateTransition(
                from_state=from_state.state_id,
                to_state=to_state.state_id,
                process_type=process_type,
                equipment=equipment,
                enthalpy_change_btu_lb=enthalpy_change,
                physical_description=self._process_descriptions.get(
                    process_type, f"Process: {process_type}"
                ),
                formula_reference=self._get_process_formula(process_type),
            )
            transitions.append(transition)

        return transitions

    def _determine_process_type(
        self,
        from_state: ThermodynamicState,
        to_state: ThermodynamicState,
    ) -> str:
        """Determine the thermodynamic process type between two states."""
        # Check for isenthalpic (throttling)
        if from_state.enthalpy_btu_lb and to_state.enthalpy_btu_lb:
            h_diff = abs(to_state.enthalpy_btu_lb - from_state.enthalpy_btu_lb)
            if h_diff < 1.0:  # Small enthalpy change
                if to_state.pressure_psig < from_state.pressure_psig:
                    return "isenthalpic"

        # Check for isobaric
        p_diff = abs(to_state.pressure_psig - from_state.pressure_psig)
        if p_diff < 1.0:  # Small pressure change
            return "isobaric"

        # Check for desuperheating (temperature drop with spray water)
        if "desuperheater" in to_state.location.lower():
            return "desuperheating"

        # Check for isentropic (turbine)
        if "turbine" in to_state.location.lower():
            return "isentropic"

        return "general"

    def _infer_equipment(
        self,
        from_location: str,
        to_location: str,
        process_type: str,
    ) -> str:
        """Infer equipment based on locations and process type."""
        if process_type == "isenthalpic":
            if "prv" in to_location.lower() or "prv" in from_location.lower():
                return "Pressure Reducing Valve (PRV)"
            return "Control Valve"

        if process_type == "desuperheating":
            return "Desuperheater (Spray Water)"

        if process_type == "isentropic":
            return "Steam Turbine"

        if "header" in to_location.lower():
            return "Steam Header"

        return "Steam System"

    def _get_process_formula(self, process_type: str) -> str:
        """Get formula reference for a process type."""
        formula_map = {
            "isenthalpic": self._formula_references["prv_isenthalpic"],
            "isentropic": self._formula_references["turbine_isentropic"],
            "desuperheating": self._formula_references["desuperheater_balance"],
            "isobaric": self._formula_references["iapws97_enthalpy"],
        }
        return formula_map.get(process_type, self._formula_references["iapws97_enthalpy"])

    def _identify_primary_drivers(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Identify primary driving signals for the calculation."""
        drivers = []

        # Priority signals for steam systems
        priority_signals = [
            ("header_pressure", "Header pressure drives saturation temperature"),
            ("steam_temperature", "Steam temperature determines superheat"),
            ("steam_flow", "Steam flow rate determines energy flow"),
            ("spray_water_temp", "Spray water temperature affects cooling capacity"),
            ("prv_position", "PRV position affects downstream pressure"),
        ]

        for signal_key, description in priority_signals:
            for input_key, input_value in inputs.items():
                if signal_key in input_key.lower():
                    drivers.append({
                        "signal": input_key,
                        "value": input_value,
                        "description": description,
                        "impact": "primary",
                    })

        return drivers

    def _generate_path_summary(
        self,
        states: List[ThermodynamicState],
        transitions: List[StateTransition],
    ) -> str:
        """Generate a summary of the thermodynamic path."""
        if not states:
            return "No thermodynamic path to trace."

        parts = []
        parts.append(f"Path from {states[0].location} to {states[-1].location}:")

        for transition in transitions:
            parts.append(
                f"  -> {transition.process_type} through {transition.equipment}"
            )

        return " ".join(parts)

    def _generate_technical_narrative(
        self,
        states: List[ThermodynamicState],
        transitions: List[StateTransition],
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> str:
        """Generate a detailed technical narrative."""
        lines = []

        lines.append("Thermodynamic Analysis:")
        lines.append("")

        # Input conditions
        lines.append("Input Conditions:")
        for key, value in list(inputs.items())[:5]:
            lines.append(f"  - {key}: {value}")

        # State progression
        lines.append("")
        lines.append("State Progression:")
        for i, state in enumerate(states):
            lines.append(
                f"  State {i + 1} ({state.location}): "
                f"P={state.pressure_psig:.1f} psig, T={state.temperature_f:.1f} F"
            )
            if state.enthalpy_btu_lb:
                lines.append(f"    h={state.enthalpy_btu_lb:.1f} BTU/lb")

        # Transitions
        if transitions:
            lines.append("")
            lines.append("Process Transitions:")
            for trans in transitions:
                lines.append(f"  - {trans.physical_description}")
                lines.append(f"    Formula: {trans.formula_reference}")

        return "\n".join(lines)

    def explain_property_calculation(
        self,
        property_name: str,
        inputs: Dict[str, float],
        formula_used: str,
        calculated_value: float,
        unit: str,
        input_sources: Optional[Dict[str, str]] = None,
    ) -> PropertyExplanation:
        """
        Explain how a property was calculated.

        Maps each calculated property back to its inputs and formula.

        Args:
            property_name: Name of the calculated property
            inputs: Input values used in calculation
            formula_used: Reference to formula used
            calculated_value: The calculated result
            unit: Unit of the result
            input_sources: Maps inputs to source signals/tags

        Returns:
            PropertyExplanation with full calculation trace
        """
        explanation_id = str(uuid.uuid4())

        # Determine property type
        property_type = self._infer_property_type(property_name)

        # Get formula details
        formula_description = self._formula_references.get(
            formula_used,
            f"Formula: {formula_used}"
        )

        # Build intermediate steps explanation
        intermediate_steps = self._build_calculation_steps(
            property_name, inputs, formula_used
        )

        explanation = PropertyExplanation(
            explanation_id=explanation_id,
            property_name=property_name,
            property_type=property_type,
            calculated_value=calculated_value,
            unit=unit,
            input_properties=inputs,
            input_sources=input_sources or {},
            formula_name=formula_used,
            formula_description=formula_description,
            intermediate_steps=intermediate_steps,
        )

        self._property_explanations[explanation_id] = explanation
        logger.debug(f"Created property explanation: {property_name}")

        return explanation

    def _infer_property_type(self, property_name: str) -> PropertyType:
        """Infer property type from name."""
        name_lower = property_name.lower()

        if "pressure" in name_lower:
            return PropertyType.PRESSURE
        if "temperature" in name_lower or "temp" in name_lower:
            return PropertyType.TEMPERATURE
        if "enthalpy" in name_lower:
            return PropertyType.ENTHALPY
        if "entropy" in name_lower:
            return PropertyType.ENTROPY
        if "volume" in name_lower:
            return PropertyType.SPECIFIC_VOLUME
        if "quality" in name_lower:
            return PropertyType.QUALITY
        if "superheat" in name_lower:
            return PropertyType.SUPERHEAT
        if "flow" in name_lower:
            return PropertyType.FLOW_RATE

        return PropertyType.ENTHALPY  # Default

    def _build_calculation_steps(
        self,
        property_name: str,
        inputs: Dict[str, float],
        formula_used: str,
    ) -> List[Dict[str, Any]]:
        """Build intermediate calculation steps for explanation."""
        steps = []

        # Step 1: Identify inputs
        steps.append({
            "step": 1,
            "description": "Gather input measurements",
            "inputs": list(inputs.keys()),
            "values": inputs,
        })

        # Step 2: Select formula
        steps.append({
            "step": 2,
            "description": f"Apply formula: {formula_used}",
            "formula_reference": self._formula_references.get(formula_used, formula_used),
        })

        # Step 3: Result
        steps.append({
            "step": 3,
            "description": f"Calculate {property_name}",
            "note": "Calculation performed by deterministic thermodynamic module",
        })

        return steps

    def identify_active_constraints(
        self,
        optimization_result: Dict[str, Any],
        constraint_set: Dict[str, Any],
    ) -> List[ActiveConstraint]:
        """
        Identify which constraints are active in optimization.

        Args:
            optimization_result: Result from optimization
            constraint_set: Set of constraints that were applied

        Returns:
            List of ActiveConstraint with binding status and impact
        """
        active_constraints = []

        # Process each constraint type
        for constraint_name, constraint_config in constraint_set.items():
            constraint_type = self._infer_constraint_type(constraint_name)

            limit_value = constraint_config.get("limit", constraint_config.get("max", 0))
            actual_value = optimization_result.get(constraint_name, 0)

            # Calculate margin
            if limit_value != 0:
                margin = abs(limit_value - actual_value) / abs(limit_value)
            else:
                margin = 1.0

            # Determine if binding
            is_binding = margin < 0.01  # Within 1% of limit

            constraint = ActiveConstraint(
                constraint_id=str(uuid.uuid4()),
                constraint_type=constraint_type,
                constraint_name=constraint_name,
                limit_value=limit_value,
                actual_value=actual_value,
                margin=margin,
                unit=constraint_config.get("unit", ""),
                is_binding=is_binding,
                shadow_price=optimization_result.get(f"{constraint_name}_shadow_price"),
                physical_meaning=self._get_constraint_meaning(constraint_type),
                impact_description=self._describe_constraint_impact(
                    constraint_name, is_binding, margin
                ),
                relaxation_options=self._get_relaxation_options(constraint_type),
            )

            active_constraints.append(constraint)

        # Sort by binding status and margin
        active_constraints.sort(key=lambda c: (not c.is_binding, c.margin))

        return active_constraints

    def _infer_constraint_type(self, constraint_name: str) -> ConstraintType:
        """Infer constraint type from name."""
        name_lower = constraint_name.lower()

        if "mass" in name_lower or "flow" in name_lower:
            return ConstraintType.MASS_BALANCE
        if "energy" in name_lower or "heat" in name_lower:
            return ConstraintType.ENERGY_BALANCE
        if "pressure" in name_lower:
            return ConstraintType.PRESSURE_LIMIT
        if "temperature" in name_lower or "temp" in name_lower:
            return ConstraintType.TEMPERATURE_LIMIT
        if "superheat" in name_lower:
            return ConstraintType.SUPERHEAT_MINIMUM
        if "quality" in name_lower:
            return ConstraintType.QUALITY_MINIMUM
        if "capacity" in name_lower or "limit" in name_lower:
            return ConstraintType.EQUIPMENT_CAPACITY
        if "safety" in name_lower:
            return ConstraintType.SAFETY_ENVELOPE

        return ConstraintType.EQUIPMENT_CAPACITY

    def _get_constraint_meaning(self, constraint_type: ConstraintType) -> str:
        """Get physical meaning of constraint type."""
        meanings = {
            ConstraintType.MASS_BALANCE: "Mass must be conserved at all nodes",
            ConstraintType.ENERGY_BALANCE: "Energy must be conserved (First Law)",
            ConstraintType.PRESSURE_LIMIT: "Equipment pressure ratings must not be exceeded",
            ConstraintType.TEMPERATURE_LIMIT: "Material temperature limits must be respected",
            ConstraintType.FLOW_LIMIT: "Pipe and equipment flow capacities",
            ConstraintType.SUPERHEAT_MINIMUM: "Minimum superheat to prevent condensation",
            ConstraintType.QUALITY_MINIMUM: "Minimum steam quality for equipment protection",
            ConstraintType.EQUIPMENT_CAPACITY: "Equipment operating range limits",
            ConstraintType.SAFETY_ENVELOPE: "Safety system limits must not be violated",
        }
        return meanings.get(constraint_type, "Operating constraint")

    def _describe_constraint_impact(
        self,
        constraint_name: str,
        is_binding: bool,
        margin: float,
    ) -> str:
        """Describe the impact of a constraint."""
        if is_binding:
            return (
                f"Constraint '{constraint_name}' is binding. "
                f"This is limiting the optimization solution."
            )
        elif margin < 0.1:
            return (
                f"Constraint '{constraint_name}' is near binding ({margin:.1%} margin). "
                f"Small changes could activate this constraint."
            )
        else:
            return (
                f"Constraint '{constraint_name}' has adequate margin ({margin:.1%}). "
                f"Not limiting current operation."
            )

    def _get_relaxation_options(
        self,
        constraint_type: ConstraintType,
    ) -> List[str]:
        """Get options for relaxing a constraint."""
        options = {
            ConstraintType.PRESSURE_LIMIT: [
                "Verify pressure rating of downstream equipment",
                "Consider installing additional PRV",
                "Review piping pressure class",
            ],
            ConstraintType.TEMPERATURE_LIMIT: [
                "Increase desuperheater spray water flow",
                "Check spray water temperature",
                "Verify temperature measurement accuracy",
            ],
            ConstraintType.SUPERHEAT_MINIMUM: [
                "Increase boiler outlet temperature",
                "Reduce PRV pressure drop",
                "Reduce header heat losses",
            ],
            ConstraintType.FLOW_LIMIT: [
                "Open parallel flow path",
                "Reduce demand from non-critical users",
                "Check for valve restrictions",
            ],
            ConstraintType.EQUIPMENT_CAPACITY: [
                "Bring additional equipment online",
                "Reduce demand",
                "Check equipment maintenance status",
            ],
        }
        return options.get(constraint_type, ["Review constraint configuration"])

    def generate_balance_explanation(
        self,
        balance_result: Dict[str, Any],
    ) -> BalanceExplanation:
        """
        Generate explanation for mass/energy balance.

        Args:
            balance_result: Result from balance calculation

        Returns:
            BalanceExplanation with detailed breakdown
        """
        explanation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        balance_type = balance_result.get("type", "mass")
        control_volume = balance_result.get("control_volume", "system")

        inflows = balance_result.get("inflows", {})
        outflows = balance_result.get("outflows", {})
        accumulation = balance_result.get("accumulation", 0.0)

        # Calculate imbalance
        total_in = sum(inflows.values())
        total_out = sum(outflows.values())
        imbalance = total_in - total_out - accumulation

        if total_in > 0:
            imbalance_percent = abs(imbalance) / total_in * 100
        else:
            imbalance_percent = 0.0

        tolerance = balance_result.get("tolerance", 0.01)
        is_balanced = imbalance_percent < (tolerance * 100)

        # Generate narrative
        narrative = self._generate_balance_narrative(
            balance_type, control_volume, inflows, outflows,
            imbalance, imbalance_percent, is_balanced
        )

        explanation = BalanceExplanation(
            explanation_id=explanation_id,
            balance_type=balance_type,
            control_volume=control_volume,
            timestamp=timestamp,
            inflows=inflows,
            outflows=outflows,
            accumulation=accumulation,
            imbalance=imbalance,
            imbalance_percent=imbalance_percent,
            is_balanced=is_balanced,
            tolerance_used=tolerance,
            unit=balance_result.get("unit", "klb/hr" if balance_type == "mass" else "MMBTU/hr"),
            narrative=narrative,
        )

        return explanation

    def _generate_balance_narrative(
        self,
        balance_type: str,
        control_volume: str,
        inflows: Dict[str, float],
        outflows: Dict[str, float],
        imbalance: float,
        imbalance_percent: float,
        is_balanced: bool,
    ) -> str:
        """Generate narrative for balance explanation."""
        lines = []

        lines.append(f"{balance_type.title()} Balance for {control_volume}:")
        lines.append("")

        # Inflows
        lines.append("Inflows:")
        for stream, value in inflows.items():
            lines.append(f"  + {stream}: {value:.2f}")
        lines.append(f"  Total In: {sum(inflows.values()):.2f}")

        # Outflows
        lines.append("")
        lines.append("Outflows:")
        for stream, value in outflows.items():
            lines.append(f"  - {stream}: {value:.2f}")
        lines.append(f"  Total Out: {sum(outflows.values()):.2f}")

        # Balance status
        lines.append("")
        if is_balanced:
            lines.append(f"Balance Status: CLOSED (imbalance: {imbalance_percent:.2f}%)")
        else:
            lines.append(f"Balance Status: OPEN (imbalance: {imbalance_percent:.2f}%)")
            lines.append(f"Missing {balance_type}: {abs(imbalance):.2f}")

        return "\n".join(lines)

    def get_trace(self, trace_id: str) -> Optional[PhysicsTrace]:
        """Get a physics trace by ID."""
        return self._traces.get(trace_id)

    def get_property_explanation(
        self,
        explanation_id: str,
    ) -> Optional[PropertyExplanation]:
        """Get a property explanation by ID."""
        return self._property_explanations.get(explanation_id)
