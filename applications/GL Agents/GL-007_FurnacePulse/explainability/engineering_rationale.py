"""
GL-007 FurnacePulse - Engineering Rationale Generator

Translates model predictions to engineering language by linking
outputs to physical phenomena. Provides deterministic, rule-based
explanations grounded in furnace engineering principles.

This module provides:
- Translation of model outputs to engineering language
- Linking predictions to physical phenomena (heat transfer, combustion)
- Root cause analysis suggestions
- Recommended corrective actions with engineering justification

Zero-Hallucination: All rationales are derived from deterministic
rules based on furnace engineering principles, not LLM generation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class PhenomenonCategory(Enum):
    """Category of physical phenomenon."""
    HEAT_TRANSFER = "heat_transfer"
    COMBUSTION = "combustion"
    FLUID_DYNAMICS = "fluid_dynamics"
    MATERIAL_DEGRADATION = "material_degradation"
    FOULING = "fouling"
    MECHANICAL = "mechanical"
    PROCESS = "process"


class SeverityLevel(Enum):
    """Severity level for conditions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ActionPriority(Enum):
    """Priority for corrective actions."""
    IMMEDIATE = "immediate"      # Within 1 hour
    URGENT = "urgent"            # Within 4 hours
    SCHEDULED = "scheduled"      # Next maintenance window
    MONITORING = "monitoring"    # Continue observation


@dataclass
class PhysicalPhenomenon:
    """Description of a physical phenomenon."""

    phenomenon_id: str
    category: PhenomenonCategory
    name: str
    description: str
    indicators: List[str]
    typical_causes: List[str]
    affected_components: List[str]
    reference_literature: List[str]


@dataclass
class RootCauseAnalysis:
    """Root cause analysis result."""

    primary_cause: str
    contributing_factors: List[str]
    evidence: Dict[str, Any]
    confidence: float
    phenomenon: PhysicalPhenomenon
    causal_chain: List[str]
    alternative_hypotheses: List[Dict[str, Any]]


@dataclass
class CorrectiveAction:
    """Recommended corrective action."""

    action_id: str
    action_description: str
    priority: ActionPriority
    engineering_justification: str
    expected_outcome: str
    implementation_steps: List[str]
    safety_considerations: List[str]
    estimated_time_hours: float
    required_resources: List[str]
    success_criteria: List[str]


@dataclass
class RationaleItem:
    """Single rationale item."""

    category: str
    rule_name: str
    description: str
    applied: bool
    impact: str
    confidence: float
    evidence: Dict[str, Any]
    references: List[str]


@dataclass
class EngineeringRationale:
    """Complete engineering rationale for a prediction."""

    prediction_type: str
    prediction_value: float
    timestamp: datetime
    primary_phenomenon: PhysicalPhenomenon
    root_cause_analysis: RootCauseAnalysis
    corrective_actions: List[CorrectiveAction]
    rationale_items: List[RationaleItem]
    engineering_summary: str
    operator_guidance: str
    safety_implications: List[str]
    computation_hash: str


class EngineeringRationale:
    """
    Generator for engineering-based explanations.

    Translates model predictions to engineering language using
    deterministic rules based on furnace physics, heat transfer
    principles, and combustion engineering.

    Physical Phenomena Covered:
    - Heat flux distribution and tube wall temperatures
    - Flame patterns and combustion efficiency
    - Fouling and coking mechanisms
    - Creep and material degradation
    - Flow maldistribution

    Example:
        >>> generator = EngineeringRationaleGenerator()
        >>> rationale = generator.generate_rationale(
        ...     prediction_type="hotspot",
        ...     prediction_value=75.0,
        ...     sensor_readings=readings,
        ...     feature_importance=shap_importance
        ... )
        >>> print(rationale.engineering_summary)
        >>> for action in rationale.corrective_actions:
        ...     print(f"{action.priority.value}: {action.action_description}")
    """

    VERSION = "1.0.0"

    # Physical phenomena knowledge base
    PHENOMENA_DB: Dict[str, PhysicalPhenomenon] = {
        "flame_impingement": PhysicalPhenomenon(
            phenomenon_id="PHE001",
            category=PhenomenonCategory.COMBUSTION,
            name="Flame Impingement",
            description=(
                "Direct contact of flame with tube surface causing localized "
                "overheating. Occurs when flame length exceeds design or flame "
                "deflection occurs due to draft imbalance."
            ),
            indicators=[
                "Localized high TMT readings",
                "Asymmetric temperature distribution",
                "Visible flame deflection",
            ],
            typical_causes=[
                "Burner misalignment",
                "Excess fuel pressure",
                "Draft imbalance",
                "Burner tip erosion",
            ],
            affected_components=["Radiant tubes", "Tube hangers", "Refractory"],
            reference_literature=["API 560", "Kern's Process Heat Transfer"],
        ),
        "flame_instability": PhysicalPhenomenon(
            phenomenon_id="PHE002",
            category=PhenomenonCategory.COMBUSTION,
            name="Flame Instability",
            description=(
                "Fluctuating combustion causing uneven heat release. Results "
                "in cyclic thermal stresses and uneven heating patterns."
            ),
            indicators=[
                "Fluctuating TMT readings",
                "Varying flame intensity",
                "Pressure oscillations",
                "Audible combustion noise",
            ],
            typical_causes=[
                "Air-fuel ratio instability",
                "Low fuel pressure",
                "Burner fouling",
                "Draft fluctuations",
            ],
            affected_components=["Burners", "Radiant section", "Control systems"],
            reference_literature=["NFPA 86", "John Zink Combustion Handbook"],
        ),
        "tube_fouling": PhysicalPhenomenon(
            phenomenon_id="PHE003",
            category=PhenomenonCategory.FOULING,
            name="Internal Tube Fouling",
            description=(
                "Accumulation of deposits (coke, scale) on tube inner surface "
                "reducing heat transfer and increasing tube wall temperature."
            ),
            indicators=[
                "Gradual TMT increase over time",
                "Reduced process outlet temperature",
                "Increased pressure drop",
                "Decreased efficiency",
            ],
            typical_causes=[
                "Coking from hydrocarbon cracking",
                "Scale from water-side deposits",
                "Polymerization",
                "Process upset",
            ],
            affected_components=["Process tubes", "Return bends", "Tube sheets"],
            reference_literature=["API 530", "HTRI Design Manual"],
        ),
        "external_fouling": PhysicalPhenomenon(
            phenomenon_id="PHE004",
            category=PhenomenonCategory.FOULING,
            name="External Tube Fouling",
            description=(
                "Accumulation of soot, ash, or deposits on tube external surface "
                "reducing radiant heat absorption."
            ),
            indicators=[
                "High flue gas temperatures",
                "Low tube wall temperatures",
                "Visible deposits on tubes",
                "Reduced heat duty",
            ],
            typical_causes=[
                "Incomplete combustion",
                "High sulfur fuel",
                "Poor atomization",
                "Insufficient excess air",
            ],
            affected_components=["Radiant tubes", "Convection section", "Stack"],
            reference_literature=["API 560", "Combustion Engineering"],
        ),
        "flow_maldistribution": PhysicalPhenomenon(
            phenomenon_id="PHE005",
            category=PhenomenonCategory.FLUID_DYNAMICS,
            name="Process Flow Maldistribution",
            description=(
                "Uneven distribution of process fluid among parallel tubes "
                "causing some tubes to receive less cooling and overheat."
            ),
            indicators=[
                "Large TMT spread across passes",
                "Some tubes significantly hotter",
                "Varying outlet temperatures",
                "Pressure imbalances",
            ],
            typical_causes=[
                "Partial tube blockage",
                "Header design issues",
                "Two-phase flow instability",
                "Valve malfunction",
            ],
            affected_components=["Process tubes", "Headers", "Manifolds"],
            reference_literature=["API 530", "Perry's Chemical Engineers Handbook"],
        ),
        "creep_damage": PhysicalPhenomenon(
            phenomenon_id="PHE006",
            category=PhenomenonCategory.MATERIAL_DEGRADATION,
            name="High Temperature Creep",
            description=(
                "Time-dependent deformation of tube material under stress at "
                "elevated temperatures, leading to eventual rupture if unchecked."
            ),
            indicators=[
                "Tube diameter increase",
                "Wall thickness reduction",
                "Metallurgical changes",
                "Sustained high TMT operation",
            ],
            typical_causes=[
                "Operation above design temperature",
                "Original design margin consumed",
                "Localized overheating",
                "Material aging",
            ],
            affected_components=["Radiant tubes", "Return bends", "Tube supports"],
            reference_literature=["API 530", "ASME BPVC Section VIII"],
        ),
        "draft_imbalance": PhysicalPhenomenon(
            phenomenon_id="PHE007",
            category=PhenomenonCategory.FLUID_DYNAMICS,
            name="Furnace Draft Imbalance",
            description=(
                "Non-uniform air flow distribution causing uneven combustion "
                "and heat release across the firebox."
            ),
            indicators=[
                "Asymmetric temperature patterns",
                "Varying O2 readings across burners",
                "Visible flame lean/deflection",
                "Damper position anomalies",
            ],
            typical_causes=[
                "Damper malfunction",
                "Wind effects",
                "Stack blockage",
                "Air register issues",
            ],
            affected_components=["Burners", "Air registers", "Stack", "Dampers"],
            reference_literature=["NFPA 86", "API 560"],
        ),
        "combustion_inefficiency": PhysicalPhenomenon(
            phenomenon_id="PHE008",
            category=PhenomenonCategory.COMBUSTION,
            name="Combustion Inefficiency",
            description=(
                "Suboptimal combustion resulting in wasted fuel energy, "
                "either from excess air (stack loss) or insufficient air (unburned fuel)."
            ),
            indicators=[
                "High excess O2 (>4%)",
                "Elevated CO emissions",
                "High flue gas temperature",
                "Low efficiency readings",
            ],
            typical_causes=[
                "Incorrect air-fuel ratio",
                "Poor fuel atomization",
                "Air leaks",
                "Burner maintenance issues",
            ],
            affected_components=["Burners", "Air supply", "Fuel system", "Controls"],
            reference_literature=["ASME PTC 4", "EPA Combustion Guidelines"],
        ),
    }

    # Root cause decision rules
    ROOT_CAUSE_RULES = [
        {
            "condition": lambda r, i: (
                any("tmt" in k.lower() and v > 550 for k, v in r.items()) and
                any("flame" in k.lower() and abs(i.get(k, 0)) > 0.1 for k in i)
            ),
            "phenomenon": "flame_impingement",
            "confidence": 0.85,
        },
        {
            "condition": lambda r, i: (
                any("flame_stability" in k.lower() and v < 0.7 for k, v in r.items())
            ),
            "phenomenon": "flame_instability",
            "confidence": 0.80,
        },
        {
            "condition": lambda r, i: (
                any("pressure_drop" in k.lower() and v > 1.2 for k, v in r.items())
            ),
            "phenomenon": "tube_fouling",
            "confidence": 0.75,
        },
        {
            "condition": lambda r, i: (
                any("flue_temp" in k.lower() and v > 400 for k, v in r.items()) and
                any("efficiency" in k.lower() and v < 85 for k, v in r.items())
            ),
            "phenomenon": "external_fouling",
            "confidence": 0.70,
        },
        {
            "condition": lambda r, i: (
                len([k for k in r if "tmt" in k.lower()]) > 1 and
                max(r.get(k, 0) for k in r if "tmt" in k.lower()) -
                min(r.get(k, 0) for k in r if "tmt" in k.lower()) > 50
            ),
            "phenomenon": "flow_maldistribution",
            "confidence": 0.75,
        },
        {
            "condition": lambda r, i: (
                any("o2" in k.lower() and (v > 5 or v < 1.5) for k, v in r.items())
            ),
            "phenomenon": "combustion_inefficiency",
            "confidence": 0.80,
        },
    ]

    def __init__(self) -> None:
        """Initialize Engineering Rationale Generator."""
        self._action_counter = 0

    def generate_rationale(
        self,
        prediction_type: str,
        prediction_value: float,
        sensor_readings: Dict[str, float],
        feature_importance: Dict[str, float],
        threshold_config: Optional[Dict[str, float]] = None,
    ) -> EngineeringRationale:
        """
        Generate complete engineering rationale for a prediction.

        Args:
            prediction_type: Type of prediction (hotspot, efficiency, rul)
            prediction_value: The predicted value
            sensor_readings: Current sensor values
            feature_importance: Feature importance from SHAP/LIME
            threshold_config: Optional thresholds for severity

        Returns:
            EngineeringRationale with full explanation
        """
        timestamp = datetime.now(timezone.utc)

        # Identify primary phenomenon
        primary_phenomenon = self._identify_phenomenon(
            sensor_readings, feature_importance, prediction_type
        )

        # Perform root cause analysis
        root_cause = self._analyze_root_cause(
            sensor_readings, feature_importance, primary_phenomenon
        )

        # Generate corrective actions
        corrective_actions = self._generate_corrective_actions(
            prediction_type, prediction_value, primary_phenomenon, root_cause
        )

        # Build rationale items
        rationale_items = self._build_rationale_items(
            sensor_readings, feature_importance, primary_phenomenon
        )

        # Generate summaries
        engineering_summary = self._generate_engineering_summary(
            prediction_type, prediction_value, primary_phenomenon, root_cause
        )

        operator_guidance = self._generate_operator_guidance(
            prediction_type, prediction_value, corrective_actions
        )

        safety_implications = self._assess_safety_implications(
            prediction_type, prediction_value, primary_phenomenon
        )

        # Compute hash
        computation_hash = self._compute_hash(
            prediction_type, prediction_value, primary_phenomenon.phenomenon_id
        )

        return EngineeringRationale(
            prediction_type=prediction_type,
            prediction_value=prediction_value,
            timestamp=timestamp,
            primary_phenomenon=primary_phenomenon,
            root_cause_analysis=root_cause,
            corrective_actions=corrective_actions,
            rationale_items=rationale_items,
            engineering_summary=engineering_summary,
            operator_guidance=operator_guidance,
            safety_implications=safety_implications,
            computation_hash=computation_hash,
        )

    def explain_hotspot(
        self,
        tube_id: str,
        tmt_value: float,
        sensor_readings: Dict[str, float],
        feature_importance: Dict[str, float],
        burner_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate hotspot-specific engineering explanation.

        Args:
            tube_id: Identifier of the affected tube
            tmt_value: Tube metal temperature reading
            sensor_readings: All sensor readings
            feature_importance: Feature importance from explainer
            burner_data: Optional burner-specific data

        Returns:
            Dictionary with hotspot explanation
        """
        # Determine severity
        severity = self._classify_tmt_severity(tmt_value)

        # Find correlated factors
        correlated_factors = self._find_correlated_factors(
            feature_importance, prefix_filter="flame"
        )

        # Identify likely burner
        likely_burner = self._identify_likely_burner(
            tube_id, sensor_readings, burner_data
        )

        # Generate physical explanation
        if likely_burner and correlated_factors:
            physical_explanation = (
                f"High TMT ({tmt_value:.1f}C) on {tube_id} correlates with "
                f"{correlated_factors[0]} from burner {likely_burner}. "
                f"This suggests {self._get_heat_transfer_explanation(tmt_value, sensor_readings)}."
            )
        else:
            physical_explanation = (
                f"Elevated TMT ({tmt_value:.1f}C) on {tube_id} indicates "
                f"localized heat flux exceeding design. "
                f"{self._get_generic_hotspot_explanation(severity)}."
            )

        # Recommended immediate actions
        immediate_actions = self._get_immediate_hotspot_actions(severity, likely_burner)

        return {
            "tube_id": tube_id,
            "tmt_value": tmt_value,
            "severity": severity.value,
            "physical_explanation": physical_explanation,
            "likely_cause": correlated_factors[0] if correlated_factors else "Unknown",
            "associated_burner": likely_burner,
            "immediate_actions": immediate_actions,
            "monitoring_recommendations": self._get_monitoring_recommendations(severity),
        }

    def explain_efficiency_loss(
        self,
        efficiency_value: float,
        sensor_readings: Dict[str, float],
        feature_importance: Dict[str, float],
        baseline_efficiency: float = 92.0,
    ) -> Dict[str, Any]:
        """
        Generate efficiency loss explanation.

        Args:
            efficiency_value: Current efficiency percentage
            sensor_readings: All sensor readings
            feature_importance: Feature importance from explainer
            baseline_efficiency: Expected baseline efficiency

        Returns:
            Dictionary with efficiency explanation
        """
        efficiency_loss = baseline_efficiency - efficiency_value
        loss_category = self._categorize_efficiency_loss(efficiency_loss)

        # Identify contributing factors
        contributors = self._identify_efficiency_contributors(
            sensor_readings, feature_importance
        )

        # Calculate loss breakdown
        loss_breakdown = self._calculate_loss_breakdown(
            sensor_readings, baseline_efficiency
        )

        # Generate explanation
        explanation_parts = []

        if loss_breakdown["stack_loss"] > 2.0:
            explanation_parts.append(
                f"Stack loss of {loss_breakdown['stack_loss']:.1f}% "
                f"(flue gas at {sensor_readings.get('flue_temp', 0):.0f}C)"
            )

        if loss_breakdown["radiation_loss"] > 1.0:
            explanation_parts.append(
                f"Radiation loss of {loss_breakdown['radiation_loss']:.1f}% "
                "from casing/surfaces"
            )

        if loss_breakdown["unburned_loss"] > 0.5:
            explanation_parts.append(
                f"Unburned fuel loss of {loss_breakdown['unburned_loss']:.1f}% "
                f"(CO at {sensor_readings.get('co_ppm', 0):.0f} ppm)"
            )

        return {
            "current_efficiency": efficiency_value,
            "baseline_efficiency": baseline_efficiency,
            "efficiency_loss": round(efficiency_loss, 2),
            "loss_category": loss_category,
            "loss_breakdown": loss_breakdown,
            "top_contributors": contributors[:3],
            "explanation": " | ".join(explanation_parts) if explanation_parts else "Within normal range",
            "improvement_potential": self._calculate_improvement_potential(loss_breakdown),
            "recommended_actions": self._get_efficiency_improvement_actions(loss_breakdown),
        }

    def explain_rul_prediction(
        self,
        rul_hours: float,
        sensor_readings: Dict[str, float],
        feature_importance: Dict[str, float],
        component_id: str = "furnace_tube",
    ) -> Dict[str, Any]:
        """
        Generate RUL prediction explanation.

        Args:
            rul_hours: Predicted remaining useful life in hours
            sensor_readings: Current sensor readings
            feature_importance: Feature importance from explainer
            component_id: Component being assessed

        Returns:
            Dictionary with RUL explanation
        """
        # Categorize RUL
        rul_category = self._categorize_rul(rul_hours)

        # Identify degradation drivers
        degradation_drivers = self._identify_degradation_drivers(
            sensor_readings, feature_importance
        )

        # Estimate degradation rate
        degradation_rate = self._estimate_degradation_rate(sensor_readings)

        # Generate explanation
        if rul_hours < 1000:
            urgency = "Critical"
            explanation = (
                f"Component {component_id} shows significant degradation. "
                f"Estimated {rul_hours:.0f} hours remaining based on "
                f"current operating conditions. Primary driver: {degradation_drivers[0] if degradation_drivers else 'accumulated stress'}."
            )
        elif rul_hours < 5000:
            urgency = "Warning"
            explanation = (
                f"Component {component_id} approaching maintenance threshold. "
                f"At current degradation rate ({degradation_rate:.2f}%/1000h), "
                f"intervention recommended within {rul_hours/24:.0f} days."
            )
        else:
            urgency = "Normal"
            explanation = (
                f"Component {component_id} within normal operating envelope. "
                f"Projected life of {rul_hours:.0f} hours under current conditions."
            )

        return {
            "component_id": component_id,
            "rul_hours": rul_hours,
            "rul_days": round(rul_hours / 24, 1),
            "category": rul_category,
            "urgency": urgency,
            "degradation_rate_per_1000h": round(degradation_rate, 3),
            "primary_drivers": degradation_drivers[:3],
            "explanation": explanation,
            "maintenance_window": self._calculate_maintenance_window(rul_hours),
            "life_extension_options": self._get_life_extension_options(degradation_drivers),
        }

    def _identify_phenomenon(
        self,
        sensor_readings: Dict[str, float],
        feature_importance: Dict[str, float],
        prediction_type: str,
    ) -> PhysicalPhenomenon:
        """Identify the primary physical phenomenon."""
        # Check rules in priority order
        for rule in self.ROOT_CAUSE_RULES:
            try:
                if rule["condition"](sensor_readings, feature_importance):
                    phenomenon_id = rule["phenomenon"]
                    if phenomenon_id in self.PHENOMENA_DB:
                        return self.PHENOMENA_DB[phenomenon_id]
            except Exception as e:
                logger.debug(f"Rule evaluation error: {e}")
                continue

        # Default phenomenon based on prediction type
        default_map = {
            "hotspot": "flame_impingement",
            "efficiency": "combustion_inefficiency",
            "rul": "creep_damage",
        }

        phenomenon_id = default_map.get(prediction_type, "combustion_inefficiency")
        return self.PHENOMENA_DB.get(
            phenomenon_id,
            self.PHENOMENA_DB["combustion_inefficiency"]
        )

    def _analyze_root_cause(
        self,
        sensor_readings: Dict[str, float],
        feature_importance: Dict[str, float],
        phenomenon: PhysicalPhenomenon,
    ) -> RootCauseAnalysis:
        """Perform root cause analysis."""
        # Select primary cause from typical causes
        primary_cause = phenomenon.typical_causes[0] if phenomenon.typical_causes else "Unknown"

        # Find contributing factors from feature importance
        contributing_factors = []
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for feat_name, importance in sorted_features[:5]:
            if abs(importance) > 0.05:
                contributing_factors.append(
                    f"{feat_name} (importance: {importance:.3f})"
                )

        # Build evidence
        evidence = {}
        for indicator in phenomenon.indicators:
            indicator_lower = indicator.lower()
            for key, value in sensor_readings.items():
                if any(word in key.lower() for word in indicator_lower.split()):
                    evidence[key] = value
                    break

        # Build causal chain
        causal_chain = [
            primary_cause,
            phenomenon.name,
            f"Affects: {', '.join(phenomenon.affected_components[:2])}",
        ]

        # Alternative hypotheses
        alternative_hypotheses = []
        for alt_cause in phenomenon.typical_causes[1:3]:
            alternative_hypotheses.append({
                "cause": alt_cause,
                "likelihood": "moderate",
                "evidence_required": f"Verify {alt_cause.lower()} through inspection",
            })

        return RootCauseAnalysis(
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            evidence=evidence,
            confidence=0.75,
            phenomenon=phenomenon,
            causal_chain=causal_chain,
            alternative_hypotheses=alternative_hypotheses,
        )

    def _generate_corrective_actions(
        self,
        prediction_type: str,
        prediction_value: float,
        phenomenon: PhysicalPhenomenon,
        root_cause: RootCauseAnalysis,
    ) -> List[CorrectiveAction]:
        """Generate corrective actions based on analysis."""
        actions = []
        self._action_counter += 1

        # Determine priority based on severity
        if prediction_type == "hotspot" and prediction_value > 80:
            priority = ActionPriority.IMMEDIATE
        elif prediction_type == "hotspot" and prediction_value > 60:
            priority = ActionPriority.URGENT
        elif prediction_type == "rul" and prediction_value < 1000:
            priority = ActionPriority.URGENT
        else:
            priority = ActionPriority.SCHEDULED

        # Generate actions based on phenomenon
        action_templates = self._get_action_templates(phenomenon.phenomenon_id)

        for template in action_templates:
            action = CorrectiveAction(
                action_id=f"ACT-{self._action_counter:04d}-{len(actions)+1}",
                action_description=template["description"],
                priority=priority if len(actions) == 0 else ActionPriority.SCHEDULED,
                engineering_justification=template["justification"],
                expected_outcome=template["outcome"],
                implementation_steps=template["steps"],
                safety_considerations=template.get("safety", []),
                estimated_time_hours=template.get("time", 2.0),
                required_resources=template.get("resources", []),
                success_criteria=template.get("criteria", []),
            )
            actions.append(action)

        return actions

    def _get_action_templates(
        self,
        phenomenon_id: str,
    ) -> List[Dict[str, Any]]:
        """Get corrective action templates for phenomenon."""
        templates = {
            "flame_impingement": [
                {
                    "description": "Adjust burner firing rate and air registers",
                    "justification": "Reducing flame length prevents direct tube contact",
                    "outcome": "Reduced localized heat flux, lower TMT readings",
                    "steps": [
                        "Reduce fuel flow by 5-10%",
                        "Adjust air registers to optimize flame pattern",
                        "Monitor TMT response for 15 minutes",
                        "Fine-tune for optimal operation",
                    ],
                    "safety": ["Ensure combustibles don't drop below LEL", "Monitor O2 levels"],
                    "time": 1.0,
                    "resources": ["Burner technician", "Control room operator"],
                    "criteria": ["TMT reduction >20C", "Stable flame pattern"],
                },
                {
                    "description": "Inspect burner alignment and tip condition",
                    "justification": "Misaligned burners or eroded tips cause flame deflection",
                    "outcome": "Corrected flame direction, even heat distribution",
                    "steps": [
                        "Schedule burner inspection during safe window",
                        "Check burner tip for erosion/damage",
                        "Verify alignment with sight glass",
                        "Replace or realign as needed",
                    ],
                    "time": 4.0,
                    "resources": ["Burner technician", "Replacement tips"],
                },
            ],
            "flame_instability": [
                {
                    "description": "Stabilize fuel supply pressure",
                    "justification": "Pressure fluctuations cause unstable combustion",
                    "outcome": "Stable flame, consistent heat release",
                    "steps": [
                        "Check fuel supply pressure",
                        "Inspect pressure regulators",
                        "Verify valve positions",
                        "Adjust setpoints if needed",
                    ],
                    "time": 2.0,
                },
                {
                    "description": "Clean burner components",
                    "justification": "Fouled burners cause uneven fuel distribution",
                    "outcome": "Improved atomization and flame stability",
                    "steps": [
                        "Schedule burner cleaning",
                        "Clean fuel nozzles/tips",
                        "Clear air passages",
                        "Test flame stability",
                    ],
                    "time": 4.0,
                },
            ],
            "tube_fouling": [
                {
                    "description": "Initiate decoking or cleaning procedure",
                    "justification": "Removing deposits restores heat transfer",
                    "outcome": "Reduced TMT, improved heat transfer",
                    "steps": [
                        "Prepare for cleaning procedure",
                        "Follow plant-specific decoking SOP",
                        "Monitor temperatures during procedure",
                        "Verify effectiveness post-cleaning",
                    ],
                    "safety": ["Follow hot work permit requirements", "Monitor emissions"],
                    "time": 24.0,
                },
            ],
            "combustion_inefficiency": [
                {
                    "description": "Optimize excess air levels",
                    "justification": "Correct O2 levels maximize efficiency",
                    "outcome": "Improved efficiency, lower stack temperature",
                    "steps": [
                        "Review current O2 readings",
                        "Adjust combustion air dampers",
                        "Target 2-3% O2 in flue gas",
                        "Monitor efficiency improvement",
                    ],
                    "time": 1.0,
                    "criteria": ["O2 between 2-3%", "Efficiency improvement >1%"],
                },
            ],
            "creep_damage": [
                {
                    "description": "Reduce operating temperature to design limits",
                    "justification": "Lower temperature slows creep degradation rate",
                    "outcome": "Extended component life",
                    "steps": [
                        "Review current vs design temperatures",
                        "Reduce firing rate if above design",
                        "Adjust process throughput if needed",
                        "Document new operating envelope",
                    ],
                    "time": 2.0,
                },
                {
                    "description": "Schedule metallurgical inspection",
                    "justification": "Assess actual creep damage for life estimation",
                    "outcome": "Accurate remaining life assessment",
                    "steps": [
                        "Plan inspection during turnaround",
                        "Perform wall thickness measurements",
                        "Conduct metallurgical sampling",
                        "Update life assessment",
                    ],
                    "time": 8.0,
                },
            ],
        }

        return templates.get(phenomenon_id, [
            {
                "description": "Investigate root cause and monitor",
                "justification": "Further analysis needed",
                "outcome": "Better understanding of condition",
                "steps": ["Gather additional data", "Consult with specialist"],
                "time": 4.0,
            }
        ])

    def _build_rationale_items(
        self,
        sensor_readings: Dict[str, float],
        feature_importance: Dict[str, float],
        phenomenon: PhysicalPhenomenon,
    ) -> List[RationaleItem]:
        """Build list of rationale items."""
        items = []

        # Phenomenon-based rationale
        items.append(RationaleItem(
            category=phenomenon.category.value,
            rule_name=phenomenon.name,
            description=phenomenon.description,
            applied=True,
            impact="Primary driver of prediction",
            confidence=0.85,
            evidence={"indicators": phenomenon.indicators},
            references=phenomenon.reference_literature,
        ))

        # Feature-based rationales
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        for feat_name, importance in top_features:
            value = sensor_readings.get(feat_name, 0.0)
            direction = "increases" if importance > 0 else "decreases"

            items.append(RationaleItem(
                category="feature_contribution",
                rule_name=f"{feat_name} Effect",
                description=f"{feat_name} at {value:.2f} {direction} prediction",
                applied=abs(importance) > 0.05,
                impact=f"Importance: {importance:.4f}",
                confidence=0.75,
                evidence={"value": value, "importance": importance},
                references=[],
            ))

        return items

    def _generate_engineering_summary(
        self,
        prediction_type: str,
        prediction_value: float,
        phenomenon: PhysicalPhenomenon,
        root_cause: RootCauseAnalysis,
    ) -> str:
        """Generate engineering summary text."""
        type_descriptions = {
            "hotspot": f"hotspot severity of {prediction_value:.1f}%",
            "efficiency": f"efficiency of {prediction_value:.1f}%",
            "rul": f"remaining useful life of {prediction_value:.0f} hours",
        }

        type_desc = type_descriptions.get(prediction_type, f"value of {prediction_value}")

        summary = (
            f"Analysis indicates {type_desc} driven primarily by {phenomenon.name}. "
            f"Root cause assessment points to {root_cause.primary_cause.lower()} "
            f"with {root_cause.confidence:.0%} confidence. "
            f"Key indicators: {', '.join(phenomenon.indicators[:2])}. "
            f"Recommended focus: {phenomenon.affected_components[0] if phenomenon.affected_components else 'system review'}."
        )

        return summary

    def _generate_operator_guidance(
        self,
        prediction_type: str,
        prediction_value: float,
        corrective_actions: List[CorrectiveAction],
    ) -> str:
        """Generate operator-friendly guidance."""
        if not corrective_actions:
            return "Continue normal monitoring. No immediate action required."

        primary_action = corrective_actions[0]

        if primary_action.priority == ActionPriority.IMMEDIATE:
            urgency = "IMMEDIATE ACTION REQUIRED"
        elif primary_action.priority == ActionPriority.URGENT:
            urgency = "Action needed within 4 hours"
        else:
            urgency = "Schedule during next maintenance window"

        guidance = (
            f"{urgency}: {primary_action.action_description}. "
            f"Expected outcome: {primary_action.expected_outcome}. "
            f"Safety note: {primary_action.safety_considerations[0] if primary_action.safety_considerations else 'Follow standard procedures'}."
        )

        return guidance

    def _assess_safety_implications(
        self,
        prediction_type: str,
        prediction_value: float,
        phenomenon: PhysicalPhenomenon,
    ) -> List[str]:
        """Assess safety implications of the condition."""
        implications = []

        if prediction_type == "hotspot" and prediction_value > 80:
            implications.append("High risk of tube failure - monitor continuously")
            implications.append("Ensure emergency procedures are ready")

        if prediction_type == "rul" and prediction_value < 500:
            implications.append("Component failure risk elevated")
            implications.append("Consider load reduction to extend life")

        if phenomenon.category == PhenomenonCategory.COMBUSTION:
            implications.append("Monitor for combustion-related hazards")
            implications.append("Verify flame detection systems operational")

        if not implications:
            implications.append("No immediate safety concerns identified")
            implications.append("Continue routine monitoring")

        return implications

    def _classify_tmt_severity(self, tmt_value: float) -> SeverityLevel:
        """Classify TMT reading severity."""
        if tmt_value >= 600:
            return SeverityLevel.CRITICAL
        elif tmt_value >= 550:
            return SeverityLevel.HIGH
        elif tmt_value >= 500:
            return SeverityLevel.MEDIUM
        elif tmt_value >= 450:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.INFO

    def _find_correlated_factors(
        self,
        feature_importance: Dict[str, float],
        prefix_filter: str = "",
    ) -> List[str]:
        """Find factors correlated with prediction."""
        correlated = []

        for name, importance in feature_importance.items():
            if abs(importance) > 0.05:
                if not prefix_filter or prefix_filter in name.lower():
                    correlated.append(f"{name} (impact: {importance:.3f})")

        return sorted(correlated, key=lambda x: abs(float(x.split(":")[-1].strip(")"))), reverse=True)

    def _identify_likely_burner(
        self,
        tube_id: str,
        sensor_readings: Dict[str, float],
        burner_data: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Identify likely burner affecting the tube."""
        # Simple heuristic: look for burner number in tube_id or nearby readings
        if burner_data:
            return burner_data.get("nearest_burner")

        # Extract number from tube_id
        tube_num = ''.join(c for c in tube_id if c.isdigit())
        if tube_num:
            # Assume tube N is nearest to burner N/4 (simplified)
            burner_num = max(1, int(tube_num) // 4)
            return f"burner_{burner_num}"

        return None

    def _get_heat_transfer_explanation(
        self,
        tmt_value: float,
        sensor_readings: Dict[str, float],
    ) -> str:
        """Generate heat transfer explanation."""
        if tmt_value > 550:
            return "localized radiant heat flux exceeds design limits"
        elif tmt_value > 500:
            return "elevated heat flux in radiant zone"
        else:
            return "moderate heat flux conditions"

    def _get_generic_hotspot_explanation(self, severity: SeverityLevel) -> str:
        """Generate generic hotspot explanation."""
        explanations = {
            SeverityLevel.CRITICAL: "Immediate investigation required to prevent tube damage",
            SeverityLevel.HIGH: "Elevated temperature requires attention and monitoring",
            SeverityLevel.MEDIUM: "Temperature trending above normal operating range",
            SeverityLevel.LOW: "Minor elevation, continue monitoring",
            SeverityLevel.INFO: "Within normal operating parameters",
        }
        return explanations.get(severity, "Monitor and evaluate")

    def _get_immediate_hotspot_actions(
        self,
        severity: SeverityLevel,
        likely_burner: Optional[str],
    ) -> List[str]:
        """Get immediate actions for hotspot."""
        actions = []

        if severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            actions.append("Reduce firing rate by 10-15%")
            if likely_burner:
                actions.append(f"Check {likely_burner} flame pattern")
            actions.append("Verify process flow rates")
            actions.append("Alert shift supervisor")

        elif severity == SeverityLevel.MEDIUM:
            actions.append("Increase monitoring frequency")
            actions.append("Check for recent process changes")

        else:
            actions.append("Continue normal monitoring")

        return actions

    def _get_monitoring_recommendations(
        self,
        severity: SeverityLevel,
    ) -> List[str]:
        """Get monitoring recommendations based on severity."""
        recommendations = {
            SeverityLevel.CRITICAL: [
                "Continuous TMT monitoring (1-minute intervals)",
                "Alert on any further temperature increase",
                "Prepare for emergency shutdown if needed",
            ],
            SeverityLevel.HIGH: [
                "Increase monitoring to 5-minute intervals",
                "Set alert threshold 10C above current reading",
                "Review trend over past 24 hours",
            ],
            SeverityLevel.MEDIUM: [
                "Monitor every 15 minutes",
                "Compare with adjacent tubes",
                "Document in shift log",
            ],
            SeverityLevel.LOW: [
                "Standard monitoring frequency",
                "Note in daily report",
            ],
            SeverityLevel.INFO: [
                "No additional monitoring required",
            ],
        }
        return recommendations.get(severity, ["Continue standard monitoring"])

    def _categorize_efficiency_loss(self, loss: float) -> str:
        """Categorize efficiency loss magnitude."""
        if loss > 5:
            return "significant"
        elif loss > 2:
            return "moderate"
        elif loss > 0.5:
            return "minor"
        else:
            return "negligible"

    def _identify_efficiency_contributors(
        self,
        sensor_readings: Dict[str, float],
        feature_importance: Dict[str, float],
    ) -> List[str]:
        """Identify top contributors to efficiency loss."""
        contributors = []

        for name, importance in sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]:
            if importance < 0:  # Negative impact on efficiency
                contributors.append(f"{name} (negative impact)")
            else:
                contributors.append(f"{name} (positive impact)")

        return contributors

    def _calculate_loss_breakdown(
        self,
        sensor_readings: Dict[str, float],
        baseline_efficiency: float,
    ) -> Dict[str, float]:
        """Calculate breakdown of efficiency losses."""
        flue_temp = sensor_readings.get("flue_temp", 300)
        o2_percent = sensor_readings.get("o2", 3)
        co_ppm = sensor_readings.get("co_ppm", 50)

        # Simplified loss calculations
        stack_loss = max(0, (flue_temp - 200) * 0.02)  # ~2% per 100C above 200C
        radiation_loss = 1.5  # Typical fixed loss
        unburned_loss = co_ppm * 0.001  # Proportional to CO

        return {
            "stack_loss": round(stack_loss, 2),
            "radiation_loss": round(radiation_loss, 2),
            "unburned_loss": round(unburned_loss, 2),
            "other_losses": round(max(0, baseline_efficiency - 100 + stack_loss + radiation_loss + unburned_loss), 2),
        }

    def _calculate_improvement_potential(
        self,
        loss_breakdown: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate potential efficiency improvement."""
        return {
            "stack_loss_reduction": round(loss_breakdown["stack_loss"] * 0.5, 2),
            "unburned_loss_reduction": round(loss_breakdown["unburned_loss"] * 0.8, 2),
            "total_potential": round(
                loss_breakdown["stack_loss"] * 0.5 +
                loss_breakdown["unburned_loss"] * 0.8,
                2
            ),
        }

    def _get_efficiency_improvement_actions(
        self,
        loss_breakdown: Dict[str, float],
    ) -> List[str]:
        """Get actions to improve efficiency."""
        actions = []

        if loss_breakdown["stack_loss"] > 3:
            actions.append("Optimize excess air to reduce stack temperature")
            actions.append("Consider economizer maintenance")

        if loss_breakdown["unburned_loss"] > 0.5:
            actions.append("Improve fuel atomization")
            actions.append("Check combustion air distribution")

        if not actions:
            actions.append("Efficiency within acceptable range")

        return actions

    def _categorize_rul(self, rul_hours: float) -> str:
        """Categorize RUL value."""
        if rul_hours < 500:
            return "critical"
        elif rul_hours < 2000:
            return "warning"
        elif rul_hours < 5000:
            return "attention"
        else:
            return "normal"

    def _identify_degradation_drivers(
        self,
        sensor_readings: Dict[str, float],
        feature_importance: Dict[str, float],
    ) -> List[str]:
        """Identify drivers of component degradation."""
        drivers = []

        # Look for temperature-related drivers
        for name, importance in feature_importance.items():
            if "temp" in name.lower() or "tmt" in name.lower():
                if importance < -0.05:
                    drivers.append(f"High {name} accelerating degradation")

        # Look for stress-related drivers
        for name, value in sensor_readings.items():
            if "pressure" in name.lower() and value > 10:
                drivers.append(f"Elevated {name} contributing to stress")

        if not drivers:
            drivers.append("Normal wear and aging")

        return drivers

    def _estimate_degradation_rate(
        self,
        sensor_readings: Dict[str, float],
    ) -> float:
        """Estimate degradation rate per 1000 hours."""
        # Simplified model: degradation increases with temperature
        max_temp = max(
            (v for k, v in sensor_readings.items() if "tmt" in k.lower() or "temp" in k.lower()),
            default=450
        )

        # Arrhenius-like relationship (simplified)
        if max_temp > 550:
            rate = 2.0
        elif max_temp > 500:
            rate = 1.0
        elif max_temp > 450:
            rate = 0.5
        else:
            rate = 0.2

        return rate

    def _calculate_maintenance_window(
        self,
        rul_hours: float,
    ) -> Dict[str, Any]:
        """Calculate recommended maintenance window."""
        safety_factor = 0.8  # 20% safety margin

        return {
            "recommended_hours": round(rul_hours * safety_factor),
            "recommended_days": round(rul_hours * safety_factor / 24),
            "latest_hours": round(rul_hours * 0.95),
            "safety_factor_applied": safety_factor,
        }

    def _get_life_extension_options(
        self,
        degradation_drivers: List[str],
    ) -> List[str]:
        """Get options to extend component life."""
        options = []

        if any("temp" in d.lower() for d in degradation_drivers):
            options.append("Reduce operating temperature by 10-20C")

        if any("pressure" in d.lower() for d in degradation_drivers):
            options.append("Reduce pressure cycling")

        options.append("Implement condition-based monitoring")
        options.append("Consider metallurgical assessment")

        return options

    def _compute_hash(
        self,
        prediction_type: str,
        prediction_value: float,
        phenomenon_id: str,
    ) -> str:
        """Compute SHA-256 hash for provenance."""
        data = {
            "prediction_type": prediction_type,
            "prediction_value": prediction_value,
            "phenomenon_id": phenomenon_id,
            "version": self.VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
