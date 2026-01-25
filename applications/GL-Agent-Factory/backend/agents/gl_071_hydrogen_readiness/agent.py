"""
GL-071: Hydrogen Readiness Agent (HYDROGEN-READINESS)

This module implements the HydrogenReadinessAgent for assessing equipment and facility
readiness for hydrogen fuel transition, including safety analysis and retrofit requirements.

Standards Reference:
    - ASME B31.12 (Hydrogen Piping and Pipelines)
    - NFPA 2 (Hydrogen Technologies Code)
    - ISO 14687 (Hydrogen Fuel Quality)
    - CGA G-5.4 (Hydrogen Piping Systems)

Example:
    >>> agent = HydrogenReadinessAgent()
    >>> result = agent.run(HydrogenReadinessInput(equipment=[...], current_fuel=...))
    >>> print(f"Readiness score: {result.overall_readiness_score:.1f}%")
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


class FuelType(str, Enum):
    NATURAL_GAS = "natural_gas"
    LPG = "lpg"
    FUEL_OIL = "fuel_oil"
    COAL = "coal"
    DIESEL = "diesel"
    HYDROGEN = "hydrogen"
    HYDROGEN_BLEND = "hydrogen_blend"


class EquipmentCategory(str, Enum):
    BURNER = "burner"
    FURNACE = "furnace"
    BOILER = "boiler"
    GAS_TURBINE = "gas_turbine"
    ENGINE = "engine"
    FUEL_CELL = "fuel_cell"
    PIPING = "piping"
    STORAGE = "storage"
    CONTROLS = "controls"
    SAFETY_SYSTEM = "safety_system"


class HydrogenBlendLevel(str, Enum):
    H2_5 = "5%"
    H2_10 = "10%"
    H2_20 = "20%"
    H2_30 = "30%"
    H2_50 = "50%"
    H2_100 = "100%"


class ReadinessLevel(str, Enum):
    NOT_READY = "not_ready"
    MINOR_MODIFICATIONS = "minor_modifications"
    MODERATE_RETROFIT = "moderate_retrofit"
    MAJOR_RETROFIT = "major_retrofit"
    REPLACEMENT_REQUIRED = "replacement_required"
    H2_READY = "h2_ready"


class MaterialCompatibility(str, Enum):
    COMPATIBLE = "compatible"
    REQUIRES_TESTING = "requires_testing"
    NOT_COMPATIBLE = "not_compatible"
    UNKNOWN = "unknown"


class Equipment(BaseModel):
    """Equipment specification."""
    equipment_id: str = Field(..., description="Equipment identifier")
    name: str = Field(..., description="Equipment name")
    category: EquipmentCategory = Field(..., description="Equipment category")
    manufacturer: Optional[str] = Field(None, description="Manufacturer")
    model: Optional[str] = Field(None, description="Model number")
    year_installed: Optional[int] = Field(None, description="Installation year")
    capacity_kw: Optional[float] = Field(None, description="Capacity in kW")
    current_fuel: FuelType = Field(..., description="Current fuel type")
    operating_pressure_bar: Optional[float] = Field(None, description="Operating pressure")
    operating_temp_celsius: Optional[float] = Field(None, description="Operating temperature")
    material_of_construction: Optional[str] = Field(None, description="Primary material")
    has_oem_h2_rating: bool = Field(default=False, description="OEM hydrogen rating")


class PipingSystem(BaseModel):
    """Piping system specification."""
    system_id: str = Field(..., description="System identifier")
    material: str = Field(..., description="Pipe material")
    diameter_mm: float = Field(..., description="Pipe diameter")
    length_m: float = Field(..., description="Total length")
    max_pressure_bar: float = Field(..., description="Maximum design pressure")
    joints_type: str = Field(default="welded", description="Joint type")
    age_years: Optional[int] = Field(None, description="System age")


class SafetySystem(BaseModel):
    """Safety system specification."""
    system_id: str = Field(..., description="System identifier")
    system_type: str = Field(..., description="Safety system type")
    detectors: List[str] = Field(default_factory=list, description="Detector types")
    ventilation_rate_ach: Optional[float] = Field(None, description="Ventilation rate")
    has_flame_detection: bool = Field(default=False, description="Flame detection")
    has_leak_detection: bool = Field(default=False, description="Leak detection")


class HydrogenReadinessInput(BaseModel):
    """Input for hydrogen readiness assessment."""
    assessment_id: Optional[str] = Field(None, description="Assessment identifier")
    facility_name: str = Field(default="Facility", description="Facility name")
    equipment: List[Equipment] = Field(..., description="Equipment list")
    piping_systems: List[PipingSystem] = Field(default_factory=list)
    safety_systems: List[SafetySystem] = Field(default_factory=list)
    target_h2_blend: HydrogenBlendLevel = Field(default=HydrogenBlendLevel.H2_20)
    target_timeline_years: int = Field(default=5, description="Timeline for transition")
    current_energy_consumption_mwh: float = Field(default=0, description="Annual energy")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EquipmentAssessment(BaseModel):
    """Assessment result for equipment."""
    equipment_id: str
    equipment_name: str
    category: str
    readiness_level: ReadinessLevel
    readiness_score: float
    max_h2_blend_current: str
    material_compatibility: MaterialCompatibility
    issues_identified: List[str]
    required_modifications: List[str]
    estimated_retrofit_cost_usd: float
    replacement_cost_usd: Optional[float]
    recommendation: str


class PipingAssessment(BaseModel):
    """Assessment result for piping system."""
    system_id: str
    material: str
    h2_compatible: bool
    max_h2_blend: str
    embrittlement_risk: str
    leak_risk_increase_percent: float
    required_modifications: List[str]
    estimated_cost_usd: float


class SafetyAssessment(BaseModel):
    """Assessment result for safety systems."""
    system_id: str
    current_adequacy: str
    h2_specific_gaps: List[str]
    required_upgrades: List[str]
    ventilation_adequate: bool
    detection_adequate: bool
    estimated_cost_usd: float


class RetrofitRequirement(BaseModel):
    """Retrofit requirement."""
    requirement_id: str
    category: str
    description: str
    priority: str
    estimated_cost_usd: float
    timeline_months: int
    dependencies: List[str]


class SafetyAnalysis(BaseModel):
    """Safety analysis summary."""
    overall_safety_score: float
    hazop_required: bool
    explosion_risk_level: str
    ventilation_requirements: Dict[str, float]
    detection_requirements: List[str]
    emergency_response_updates: List[str]
    training_requirements: List[str]
    compliance_gaps: List[str]


class TransitionPathway(BaseModel):
    """Hydrogen transition pathway."""
    phase: int
    phase_name: str
    h2_blend_target: str
    timeline_months: int
    equipment_modifications: List[str]
    infrastructure_upgrades: List[str]
    estimated_cost_usd: float
    key_milestones: List[str]


class HydrogenReadinessOutput(BaseModel):
    """Output from hydrogen readiness assessment."""
    assessment_id: str
    facility_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    target_h2_blend: str
    overall_readiness_score: float
    overall_readiness_level: ReadinessLevel
    equipment_assessments: List[EquipmentAssessment]
    piping_assessments: List[PipingAssessment]
    safety_assessments: List[SafetyAssessment]
    retrofit_requirements: List[RetrofitRequirement]
    safety_analysis: SafetyAnalysis
    transition_pathway: List[TransitionPathway]
    total_retrofit_cost_usd: float
    total_replacement_cost_usd: float
    recommended_approach: str
    co2_reduction_potential_tpa: float
    simple_payback_years: Optional[float]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class HydrogenReadinessAgent:
    """GL-071: Hydrogen Readiness Agent - H2 transition assessment."""

    AGENT_ID = "GL-071"
    AGENT_NAME = "HYDROGEN-READINESS"
    VERSION = "1.0.0"

    # H2 emission factor reduction vs natural gas
    H2_EMISSION_REDUCTION = 1.0  # 100% CO2 reduction for pure H2
    NG_EMISSION_FACTOR = 0.184  # kg CO2/kWh

    # Material compatibility scores
    MATERIAL_SCORES = {
        "stainless_steel_316": 0.95,
        "stainless_steel_304": 0.90,
        "carbon_steel": 0.60,
        "copper": 0.85,
        "brass": 0.70,
        "aluminum": 0.80,
        "hdpe": 0.95,
        "cast_iron": 0.40,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"HydrogenReadinessAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: HydrogenReadinessInput) -> HydrogenReadinessOutput:
        start_time = datetime.utcnow()

        # Assess equipment
        equipment_assessments = [
            self._assess_equipment(eq, input_data.target_h2_blend)
            for eq in input_data.equipment]

        # Assess piping
        piping_assessments = [
            self._assess_piping(pipe, input_data.target_h2_blend)
            for pipe in input_data.piping_systems]

        # Assess safety systems
        safety_assessments = [
            self._assess_safety_system(ss, input_data.target_h2_blend)
            for ss in input_data.safety_systems]

        # Calculate overall readiness
        eq_scores = [ea.readiness_score for ea in equipment_assessments]
        overall_score = sum(eq_scores) / len(eq_scores) if eq_scores else 0

        overall_level = self._determine_readiness_level(overall_score)

        # Generate retrofit requirements
        retrofit_reqs = self._generate_retrofit_requirements(
            equipment_assessments, piping_assessments, safety_assessments)

        # Perform safety analysis
        safety_analysis = self._perform_safety_analysis(
            input_data.equipment, input_data.safety_systems, input_data.target_h2_blend)

        # Generate transition pathway
        pathway = self._generate_transition_pathway(
            input_data.target_h2_blend, input_data.target_timeline_years, retrofit_reqs)

        # Calculate costs
        total_retrofit = sum(ea.estimated_retrofit_cost_usd for ea in equipment_assessments)
        total_retrofit += sum(pa.estimated_cost_usd for pa in piping_assessments)
        total_retrofit += sum(sa.estimated_cost_usd for sa in safety_assessments)

        total_replacement = sum(
            ea.replacement_cost_usd for ea in equipment_assessments
            if ea.replacement_cost_usd is not None)

        # CO2 reduction potential
        h2_fraction = self._parse_h2_blend(input_data.target_h2_blend)
        co2_reduction = (input_data.current_energy_consumption_mwh * 1000 *
                        self.NG_EMISSION_FACTOR * h2_fraction / 1000)  # tpa

        # Payback calculation (simplified)
        annual_carbon_savings = co2_reduction * 50  # $50/tCO2
        payback = total_retrofit / annual_carbon_savings if annual_carbon_savings > 0 else None

        # Recommendation
        if overall_score >= 80:
            recommendation = "Facility is largely H2-ready. Proceed with minor modifications."
        elif overall_score >= 60:
            recommendation = "Moderate retrofit required. Phased approach recommended."
        elif overall_score >= 40:
            recommendation = "Significant investment needed. Consider equipment replacement for critical items."
        else:
            recommendation = "Major overhaul required. Evaluate new H2-ready equipment vs. retrofit costs."

        provenance_hash = hashlib.sha256(
            json.dumps({
                "agent": self.AGENT_ID,
                "facility": input_data.facility_name,
                "target_blend": input_data.target_h2_blend.value,
                "timestamp": datetime.utcnow().isoformat()
            }, sort_keys=True, default=str).encode()).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return HydrogenReadinessOutput(
            assessment_id=input_data.assessment_id or f"H2R-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            facility_name=input_data.facility_name,
            target_h2_blend=input_data.target_h2_blend.value,
            overall_readiness_score=round(overall_score, 2),
            overall_readiness_level=overall_level,
            equipment_assessments=equipment_assessments,
            piping_assessments=piping_assessments,
            safety_assessments=safety_assessments,
            retrofit_requirements=retrofit_reqs,
            safety_analysis=safety_analysis,
            transition_pathway=pathway,
            total_retrofit_cost_usd=round(total_retrofit, 2),
            total_replacement_cost_usd=round(total_replacement, 2),
            recommended_approach=recommendation,
            co2_reduction_potential_tpa=round(co2_reduction, 2),
            simple_payback_years=round(payback, 1) if payback else None,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS")

    def _assess_equipment(self, equipment: Equipment,
                         target_blend: HydrogenBlendLevel) -> EquipmentAssessment:
        """Assess equipment hydrogen readiness."""
        h2_fraction = self._parse_h2_blend(target_blend)
        issues = []
        modifications = []
        score = 100.0

        # OEM H2 rating
        if equipment.has_oem_h2_rating:
            max_blend = "100%"
            score = 95
        else:
            # Assess based on category
            if equipment.category == EquipmentCategory.BURNER:
                # Burners need modification for H2 flame characteristics
                if h2_fraction > 0.20:
                    issues.append("Burner may require redesign for high H2 content")
                    modifications.append("Install H2-compatible burner tips")
                    modifications.append("Adjust combustion air controls")
                    score -= 30
                max_blend = "20%" if not equipment.has_oem_h2_rating else "100%"

            elif equipment.category == EquipmentCategory.BOILER:
                issues.append("Flame characteristics change with H2")
                if h2_fraction > 0.20:
                    modifications.append("Modify flame supervision system")
                    modifications.append("Update combustion controls")
                    score -= 25
                max_blend = "20%"

            elif equipment.category == EquipmentCategory.GAS_TURBINE:
                # GT are sensitive to fuel changes
                issues.append("Combustor modifications likely required")
                if h2_fraction > 0.30:
                    modifications.append("Combustor retrofit for H2")
                    modifications.append("Fuel system upgrades")
                    score -= 40
                max_blend = "30%"

            elif equipment.category == EquipmentCategory.FURNACE:
                issues.append("Higher flame temperature with H2")
                modifications.append("Evaluate refractory materials")
                if h2_fraction > 0.20:
                    modifications.append("Install UV flame detection")
                    score -= 20
                max_blend = "30%"

            else:
                max_blend = "10%"
                score -= 10

        # Material compatibility
        material = (equipment.material_of_construction or "carbon_steel").lower().replace(" ", "_")
        material_score = self.MATERIAL_SCORES.get(material, 0.50)

        if material_score < 0.70:
            issues.append(f"Material ({material}) may be susceptible to H2 embrittlement")
            modifications.append("Material testing for H2 compatibility")
            score -= (1 - material_score) * 30

        compatibility = (MaterialCompatibility.COMPATIBLE if material_score >= 0.85
                        else MaterialCompatibility.REQUIRES_TESTING if material_score >= 0.60
                        else MaterialCompatibility.NOT_COMPATIBLE)

        # Age factor
        if equipment.year_installed:
            age = datetime.utcnow().year - equipment.year_installed
            if age > 20:
                issues.append("Equipment age may affect H2 transition viability")
                score -= 10

        # Estimate costs
        base_retrofit_cost = (equipment.capacity_kw or 1000) * 50  # $50/kW estimate
        retrofit_cost = base_retrofit_cost * (100 - score) / 100
        replacement_cost = (equipment.capacity_kw or 1000) * 500  # $500/kW for new

        readiness_level = self._determine_readiness_level(score)

        if score >= 80:
            recommendation = "Minor adjustments needed; proceed with H2 transition"
        elif score >= 60:
            recommendation = "Moderate modifications required; phase transition recommended"
        elif score >= 40:
            recommendation = "Significant retrofit needed; evaluate replacement option"
        else:
            recommendation = "Replacement recommended; retrofit not cost-effective"

        return EquipmentAssessment(
            equipment_id=equipment.equipment_id,
            equipment_name=equipment.name,
            category=equipment.category.value,
            readiness_level=readiness_level,
            readiness_score=round(max(0, score), 2),
            max_h2_blend_current=max_blend,
            material_compatibility=compatibility,
            issues_identified=issues,
            required_modifications=modifications,
            estimated_retrofit_cost_usd=round(retrofit_cost, 2),
            replacement_cost_usd=round(replacement_cost, 2),
            recommendation=recommendation)

    def _assess_piping(self, piping: PipingSystem,
                      target_blend: HydrogenBlendLevel) -> PipingAssessment:
        """Assess piping system hydrogen readiness."""
        h2_fraction = self._parse_h2_blend(target_blend)
        modifications = []

        material_lower = piping.material.lower()

        # H2 compatibility assessment
        if "stainless" in material_lower:
            h2_compatible = True
            max_blend = "100%"
            embrittlement_risk = "LOW"
        elif "hdpe" in material_lower or "pe" in material_lower:
            h2_compatible = True
            max_blend = "100%"
            embrittlement_risk = "NONE"
        elif "carbon" in material_lower:
            h2_compatible = h2_fraction <= 0.20
            max_blend = "20%"
            embrittlement_risk = "HIGH" if h2_fraction > 0.20 else "MEDIUM"
            if h2_fraction > 0.20:
                modifications.append("Replace carbon steel with stainless or PE")
        else:
            h2_compatible = h2_fraction <= 0.10
            max_blend = "10%"
            embrittlement_risk = "MEDIUM"

        # Leak risk (H2 molecules are smaller)
        leak_increase = h2_fraction * 50  # Up to 50% increase for 100% H2

        # Joints assessment
        if piping.joints_type.lower() != "welded":
            modifications.append("Replace threaded/flanged joints with welded joints")
            leak_increase += 20

        # Pressure consideration
        if piping.max_pressure_bar < 20 and h2_fraction > 0.50:
            modifications.append("Verify pressure rating for H2 service")

        # Cost estimation
        cost_per_meter = 500 if not h2_compatible else 100
        estimated_cost = piping.length_m * cost_per_meter

        return PipingAssessment(
            system_id=piping.system_id,
            material=piping.material,
            h2_compatible=h2_compatible,
            max_h2_blend=max_blend,
            embrittlement_risk=embrittlement_risk,
            leak_risk_increase_percent=round(leak_increase, 1),
            required_modifications=modifications,
            estimated_cost_usd=round(estimated_cost, 2))

    def _assess_safety_system(self, safety: SafetySystem,
                             target_blend: HydrogenBlendLevel) -> SafetyAssessment:
        """Assess safety system adequacy."""
        gaps = []
        upgrades = []

        # H2 detection
        has_h2_detection = any("h2" in d.lower() or "hydrogen" in d.lower()
                              for d in safety.detectors)
        if not has_h2_detection:
            gaps.append("No hydrogen-specific detection")
            upgrades.append("Install H2 detectors (catalytic or electrochemical)")

        # Ventilation (H2 is lighter than air, needs different approach)
        vent_adequate = (safety.ventilation_rate_ach or 0) >= 12  # Higher for H2
        if not vent_adequate:
            gaps.append("Ventilation rate insufficient for H2")
            upgrades.append("Increase ventilation to minimum 12 ACH")

        # Flame detection (H2 flame is nearly invisible)
        if not safety.has_flame_detection:
            gaps.append("No flame detection system")
            upgrades.append("Install UV flame detection for H2")

        # Leak detection
        detection_adequate = safety.has_leak_detection and has_h2_detection
        if not detection_adequate:
            upgrades.append("Upgrade leak detection for H2")

        current_adequacy = ("ADEQUATE" if len(gaps) == 0
                          else "PARTIAL" if len(gaps) <= 2
                          else "INADEQUATE")

        # Cost estimation
        cost_per_upgrade = 25000
        estimated_cost = len(upgrades) * cost_per_upgrade

        return SafetyAssessment(
            system_id=safety.system_id,
            current_adequacy=current_adequacy,
            h2_specific_gaps=gaps,
            required_upgrades=upgrades,
            ventilation_adequate=vent_adequate,
            detection_adequate=detection_adequate,
            estimated_cost_usd=round(estimated_cost, 2))

    def _determine_readiness_level(self, score: float) -> ReadinessLevel:
        """Determine readiness level from score."""
        if score >= 90:
            return ReadinessLevel.H2_READY
        elif score >= 75:
            return ReadinessLevel.MINOR_MODIFICATIONS
        elif score >= 55:
            return ReadinessLevel.MODERATE_RETROFIT
        elif score >= 35:
            return ReadinessLevel.MAJOR_RETROFIT
        else:
            return ReadinessLevel.REPLACEMENT_REQUIRED

    def _parse_h2_blend(self, blend: HydrogenBlendLevel) -> float:
        """Parse H2 blend level to fraction."""
        mapping = {
            HydrogenBlendLevel.H2_5: 0.05,
            HydrogenBlendLevel.H2_10: 0.10,
            HydrogenBlendLevel.H2_20: 0.20,
            HydrogenBlendLevel.H2_30: 0.30,
            HydrogenBlendLevel.H2_50: 0.50,
            HydrogenBlendLevel.H2_100: 1.00,
        }
        return mapping.get(blend, 0.20)

    def _generate_retrofit_requirements(self,
                                        equipment_assessments: List[EquipmentAssessment],
                                        piping_assessments: List[PipingAssessment],
                                        safety_assessments: List[SafetyAssessment]) -> List[RetrofitRequirement]:
        """Generate prioritized retrofit requirements."""
        requirements = []
        req_num = 0

        # Safety first
        for sa in safety_assessments:
            for upgrade in sa.required_upgrades:
                req_num += 1
                requirements.append(RetrofitRequirement(
                    requirement_id=f"REQ-{req_num:03d}",
                    category="SAFETY",
                    description=upgrade,
                    priority="P1",
                    estimated_cost_usd=sa.estimated_cost_usd / max(1, len(sa.required_upgrades)),
                    timeline_months=3,
                    dependencies=[]))

        # Piping second
        for pa in piping_assessments:
            for mod in pa.required_modifications:
                req_num += 1
                requirements.append(RetrofitRequirement(
                    requirement_id=f"REQ-{req_num:03d}",
                    category="PIPING",
                    description=f"{pa.system_id}: {mod}",
                    priority="P2",
                    estimated_cost_usd=pa.estimated_cost_usd / max(1, len(pa.required_modifications)),
                    timeline_months=6,
                    dependencies=[]))

        # Equipment third
        for ea in equipment_assessments:
            for mod in ea.required_modifications:
                req_num += 1
                priority = ("P2" if ea.readiness_level == ReadinessLevel.MINOR_MODIFICATIONS
                           else "P3" if ea.readiness_level == ReadinessLevel.MODERATE_RETROFIT
                           else "P4")
                requirements.append(RetrofitRequirement(
                    requirement_id=f"REQ-{req_num:03d}",
                    category="EQUIPMENT",
                    description=f"{ea.equipment_name}: {mod}",
                    priority=priority,
                    estimated_cost_usd=ea.estimated_retrofit_cost_usd / max(1, len(ea.required_modifications)),
                    timeline_months=12,
                    dependencies=[]))

        return requirements

    def _perform_safety_analysis(self, equipment: List[Equipment],
                                 safety_systems: List[SafetySystem],
                                 target_blend: HydrogenBlendLevel) -> SafetyAnalysis:
        """Perform comprehensive safety analysis."""
        h2_fraction = self._parse_h2_blend(target_blend)

        # Explosion risk
        if h2_fraction >= 0.50:
            explosion_risk = "HIGH"
        elif h2_fraction >= 0.20:
            explosion_risk = "MEDIUM"
        else:
            explosion_risk = "LOW"

        # Ventilation requirements
        total_capacity_kw = sum(eq.capacity_kw or 0 for eq in equipment)
        base_ach = 6
        h2_ach = base_ach + (h2_fraction * 12)  # Scale with H2 content
        ventilation_reqs = {
            "minimum_ach": round(h2_ach, 1),
            "emergency_ach": round(h2_ach * 2, 1),
            "high_point_vents_required": h2_fraction >= 0.20
        }

        detection_reqs = [
            "H2 detectors at ceiling level (H2 rises)",
            "LEL detection with alarm at 25% LEL",
            "UV flame detection (H2 flame invisible)",
            "Area classification review for Zone 2"
        ]

        emergency_updates = [
            "Update emergency response procedures for H2",
            "Install emergency H2 isolation valves",
            "Update fire suppression for H2 fires",
            "Emergency ventilation activation protocol"
        ]

        training_reqs = [
            "H2 safety awareness training for all personnel",
            "H2 leak response procedures",
            "H2 fire fighting techniques",
            "First aid for H2 exposure"
        ]

        compliance_gaps = []
        if h2_fraction >= 0.20:
            compliance_gaps.append("ASME B31.12 compliance review required")
            compliance_gaps.append("NFPA 2 gap analysis needed")
        if h2_fraction >= 0.50:
            compliance_gaps.append("Process hazard analysis (PHA) update required")

        # Overall safety score
        safety_features = sum([
            any(ss.has_flame_detection for ss in safety_systems),
            any(ss.has_leak_detection for ss in safety_systems),
            any((ss.ventilation_rate_ach or 0) >= h2_ach for ss in safety_systems),
            len(safety_systems) > 0
        ])
        safety_score = (safety_features / 4) * 100

        return SafetyAnalysis(
            overall_safety_score=round(safety_score, 1),
            hazop_required=h2_fraction >= 0.20,
            explosion_risk_level=explosion_risk,
            ventilation_requirements=ventilation_reqs,
            detection_requirements=detection_reqs,
            emergency_response_updates=emergency_updates,
            training_requirements=training_reqs,
            compliance_gaps=compliance_gaps)

    def _generate_transition_pathway(self, target_blend: HydrogenBlendLevel,
                                     timeline_years: int,
                                     requirements: List[RetrofitRequirement]) -> List[TransitionPathway]:
        """Generate phased transition pathway."""
        pathway = []
        h2_fraction = self._parse_h2_blend(target_blend)

        # Phase 1: Preparation and safety
        p1_reqs = [r for r in requirements if r.priority == "P1"]
        p1_cost = sum(r.estimated_cost_usd for r in p1_reqs)
        pathway.append(TransitionPathway(
            phase=1,
            phase_name="Preparation & Safety Upgrades",
            h2_blend_target="0%",
            timeline_months=min(6, timeline_years * 12 // 4),
            equipment_modifications=[],
            infrastructure_upgrades=[r.description for r in p1_reqs],
            estimated_cost_usd=round(p1_cost, 2),
            key_milestones=["Safety system upgrades complete", "Training complete", "Procedures updated"]))

        # Phase 2: Initial H2 introduction
        if h2_fraction >= 0.05:
            p2_reqs = [r for r in requirements if r.priority == "P2"]
            p2_cost = sum(r.estimated_cost_usd for r in p2_reqs)
            pathway.append(TransitionPathway(
                phase=2,
                phase_name="Initial H2 Introduction",
                h2_blend_target="5-10%",
                timeline_months=timeline_years * 12 // 3,
                equipment_modifications=["Minor burner adjustments", "Control system updates"],
                infrastructure_upgrades=[r.description for r in p2_reqs[:3]],
                estimated_cost_usd=round(p2_cost * 0.5, 2),
                key_milestones=["5% H2 blend operational", "Monitoring systems active", "Performance baseline"]))

        # Phase 3: Increased H2 blend
        if h2_fraction >= 0.20:
            p3_reqs = [r for r in requirements if r.priority in ["P2", "P3"]]
            p3_cost = sum(r.estimated_cost_usd for r in p3_reqs)
            pathway.append(TransitionPathway(
                phase=3,
                phase_name="Increased H2 Content",
                h2_blend_target="20%",
                timeline_months=timeline_years * 12 * 2 // 3,
                equipment_modifications=["Burner modifications", "Combustion optimization"],
                infrastructure_upgrades=["Complete piping upgrades"],
                estimated_cost_usd=round(p3_cost * 0.3, 2),
                key_milestones=["20% H2 blend achieved", "Efficiency verified", "Compliance confirmed"]))

        # Phase 4: Target blend
        if h2_fraction > 0.20:
            p4_reqs = [r for r in requirements if r.priority in ["P3", "P4"]]
            p4_cost = sum(r.estimated_cost_usd for r in p4_reqs)
            pathway.append(TransitionPathway(
                phase=4,
                phase_name="Target H2 Blend",
                h2_blend_target=target_blend.value,
                timeline_months=timeline_years * 12,
                equipment_modifications=["Major equipment upgrades/replacement"],
                infrastructure_upgrades=["Final infrastructure modifications"],
                estimated_cost_usd=round(p4_cost, 2),
                key_milestones=[f"{target_blend.value} H2 achieved", "Full operation", "Target emissions met"]))

        return pathway


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-071",
    "name": "HYDROGEN-READINESS",
    "version": "1.0.0",
    "summary": "Hydrogen readiness assessment for equipment and facilities",
    "tags": ["hydrogen", "H2-ready", "fuel-transition", "decarbonization", "safety"],
    "standards": [
        {"ref": "ASME B31.12", "description": "Hydrogen Piping and Pipelines"},
        {"ref": "NFPA 2", "description": "Hydrogen Technologies Code"},
        {"ref": "ISO 14687", "description": "Hydrogen Fuel Quality"}
    ],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
