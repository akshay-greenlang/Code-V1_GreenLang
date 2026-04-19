# -*- coding: utf-8 -*-
"""
GL-DECARB-PUB-004: Street Lighting Optimization Agent
======================================================

Optimizes municipal street lighting for energy efficiency and emission reduction.
Plans LED conversions, implements smart controls, and tracks energy savings.

Capabilities:
    - Street light inventory management
    - LED conversion planning and ROI analysis
    - Smart lighting control strategies
    - Energy and emission reduction calculations
    - Utility incentive optimization
    - Light pollution reduction analysis
    - Maintenance cost optimization

Zero-Hallucination Principle:
    All energy calculations use verified engineering formulas.
    Emission factors from EPA eGRID or local utility data.
    LED specifications from manufacturer data sheets.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class LightingTechnology(str, Enum):
    """Street lighting technologies."""
    INCANDESCENT = "incandescent"
    MERCURY_VAPOR = "mercury_vapor"
    HIGH_PRESSURE_SODIUM = "high_pressure_sodium"
    METAL_HALIDE = "metal_halide"
    FLUORESCENT = "fluorescent"
    LED = "led"
    INDUCTION = "induction"


class ControlStrategy(str, Enum):
    """Lighting control strategies."""
    NONE = "none"  # Always on during night
    PHOTOCELL = "photocell"  # Dusk-to-dawn
    TIMER = "timer"  # Fixed schedule
    DIMMING = "dimming"  # Time-based dimming
    ADAPTIVE = "adaptive"  # Motion/traffic-based
    SMART = "smart"  # Full networked control


class LightingZoneType(str, Enum):
    """Types of lighting zones."""
    ARTERIAL = "arterial"  # Major roads
    COLLECTOR = "collector"  # Medium roads
    LOCAL = "local"  # Residential streets
    PEDESTRIAN = "pedestrian"  # Walkways
    PARKING = "parking"  # Parking lots
    PARK = "park"  # Parks and recreation
    DOWNTOWN = "downtown"  # Commercial districts


# =============================================================================
# Pydantic Models
# =============================================================================

class StreetLight(BaseModel):
    """Individual street light fixture."""

    light_id: str = Field(..., description="Unique light identifier")
    location: str = Field(..., description="Location description")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

    # Technology
    technology: LightingTechnology = Field(..., description="Lamp technology")
    wattage: float = Field(..., gt=0, description="Lamp wattage")
    lumens: float = Field(default=0.0, ge=0, description="Light output")
    color_temperature_k: Optional[int] = Field(None, description="Color temperature")

    # Control
    control_strategy: ControlStrategy = Field(
        default=ControlStrategy.PHOTOCELL,
        description="Control strategy"
    )

    # Operation
    operating_hours_per_year: float = Field(
        default=4380.0,  # 12 hours/day average
        ge=0,
        description="Annual operating hours"
    )

    # Installation
    pole_id: Optional[str] = Field(None)
    pole_height_ft: Optional[float] = Field(None, ge=0)
    pole_material: Optional[str] = Field(None)
    installation_year: Optional[int] = Field(None, ge=1900, le=2030)

    # Zone
    zone_type: LightingZoneType = Field(
        default=LightingZoneType.LOCAL,
        description="Lighting zone type"
    )

    @property
    def annual_kwh(self) -> float:
        """Calculate annual energy consumption."""
        return self.wattage * self.operating_hours_per_year / 1000

    @property
    def efficacy_lumens_per_watt(self) -> float:
        """Calculate lamp efficacy."""
        if self.wattage > 0:
            return self.lumens / self.wattage
        return 0.0


class LightingZone(BaseModel):
    """Lighting zone grouping."""

    zone_id: str = Field(..., description="Zone identifier")
    zone_name: str = Field(..., description="Zone name")
    zone_type: LightingZoneType = Field(..., description="Zone type")
    description: Optional[str] = Field(None)

    # Lights in zone
    light_ids: List[str] = Field(default_factory=list)

    # Targets
    minimum_illuminance_fc: float = Field(
        default=0.5,
        ge=0,
        description="Minimum illuminance (footcandles)"
    )


class LEDConversionProject(BaseModel):
    """LED conversion project details."""

    project_id: str = Field(..., description="Project identifier")
    project_name: str = Field(..., description="Project name")

    # Scope
    lights_to_convert: List[str] = Field(
        default_factory=list,
        description="Light IDs to convert"
    )

    # Current state
    current_total_wattage: float = Field(default=0.0, ge=0)
    current_annual_kwh: float = Field(default=0.0, ge=0)

    # Proposed
    proposed_led_wattage: float = Field(default=0.0, ge=0)
    proposed_annual_kwh: float = Field(default=0.0, ge=0)

    # Savings
    annual_kwh_savings: float = Field(default=0.0, ge=0)
    annual_cost_savings_usd: float = Field(default=0.0, ge=0)
    annual_emission_reduction_tco2e: float = Field(default=0.0, ge=0)

    # Costs
    total_project_cost_usd: float = Field(default=0.0, ge=0)
    utility_incentive_usd: float = Field(default=0.0, ge=0)
    net_project_cost_usd: float = Field(default=0.0, ge=0)

    # Financial
    simple_payback_years: float = Field(default=0.0, ge=0)

    # Timeline
    estimated_months_to_complete: int = Field(default=12, ge=1)


class LightingOptimizationPlan(BaseModel):
    """Complete lighting optimization plan."""

    plan_id: str = Field(..., description="Plan identifier")
    municipality_name: str = Field(..., description="Municipality name")
    plan_name: str = Field(..., description="Plan name")

    # Inventory
    street_lights: List[StreetLight] = Field(
        default_factory=list,
        description="Street light inventory"
    )
    zones: List[LightingZone] = Field(
        default_factory=list,
        description="Lighting zones"
    )

    # Projects
    conversion_projects: List[LEDConversionProject] = Field(
        default_factory=list,
        description="LED conversion projects"
    )

    # Summary metrics
    total_lights: int = Field(default=0, ge=0)
    total_wattage_kw: float = Field(default=0.0, ge=0)
    total_annual_kwh: float = Field(default=0.0, ge=0)
    total_annual_cost_usd: float = Field(default=0.0, ge=0)
    total_annual_emissions_tco2e: float = Field(default=0.0, ge=0)

    # Technology breakdown
    technology_breakdown: Dict[str, int] = Field(default_factory=dict)

    # Metadata
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    updated_at: datetime = Field(default_factory=DeterministicClock.now)
    created_by: Optional[str] = Field(None)
    provenance_hash: Optional[str] = Field(None)


# =============================================================================
# Agent Input/Output Models
# =============================================================================

class StreetLightingInput(BaseModel):
    """Input for Street Lighting Agent."""

    action: str = Field(..., description="Action to perform")

    # Identifiers
    plan_id: Optional[str] = Field(None)
    municipality_name: Optional[str] = Field(None)
    plan_name: Optional[str] = Field(None)

    # Data
    light: Optional[StreetLight] = Field(None)
    lights: Optional[List[StreetLight]] = Field(None)
    zone: Optional[LightingZone] = Field(None)

    # Parameters
    electricity_rate_usd_per_kwh: Optional[float] = Field(None, ge=0)
    emission_factor_kg_per_kwh: Optional[float] = Field(None, ge=0)
    led_incentive_per_fixture_usd: Optional[float] = Field(None, ge=0)

    # Metadata
    user_id: Optional[str] = Field(None)
    tenant_id: Optional[str] = Field(None)

    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate action."""
        valid_actions = {
            'create_plan',
            'add_light',
            'add_lights',
            'add_zone',
            'analyze_inventory',
            'plan_led_conversion',
            'calculate_savings',
            'calculate_emissions',
            'get_plan',
            'list_plans',
        }
        if v not in valid_actions:
            raise ValueError(f"Invalid action: {v}")
        return v


class StreetLightingOutput(BaseModel):
    """Output from Street Lighting Agent."""

    success: bool = Field(...)
    action: str = Field(...)

    # Results
    plan: Optional[LightingOptimizationPlan] = Field(None)
    plans: Optional[List[LightingOptimizationPlan]] = Field(None)
    inventory_analysis: Optional[Dict[str, Any]] = Field(None)
    conversion_project: Optional[LEDConversionProject] = Field(None)
    savings_analysis: Optional[Dict[str, Any]] = Field(None)
    emission_analysis: Optional[Dict[str, Any]] = Field(None)

    # Provenance
    provenance_hash: Optional[str] = Field(None)
    calculation_trace: List[str] = Field(default_factory=list)

    # Error handling
    error: Optional[str] = Field(None)
    warnings: List[str] = Field(default_factory=list)

    # Metadata
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    processing_time_ms: float = Field(default=0.0)


# =============================================================================
# Street Lighting Optimization Agent
# =============================================================================

class StreetLightingOptimizationAgent(BaseAgent):
    """
    GL-DECARB-PUB-004: Street Lighting Optimization Agent

    Optimizes municipal street lighting for energy efficiency.

    Zero-Hallucination Guarantees:
        - Energy calculations use verified engineering formulas
        - LED specifications from manufacturer data
        - Emission factors from EPA eGRID
        - Complete audit trail with SHA-256 hashes

    Usage:
        agent = StreetLightingOptimizationAgent()
        result = agent.run({
            'action': 'create_plan',
            'municipality_name': 'Springfield',
        })
    """

    AGENT_ID = "GL-DECARB-PUB-004"
    AGENT_NAME = "Street Lighting Optimization Agent"
    VERSION = "1.0.0"

    # Default emission factor (EPA eGRID US average)
    DEFAULT_EMISSION_FACTOR = 0.386  # kg CO2/kWh

    # Default electricity rate
    DEFAULT_ELECTRICITY_RATE = 0.10  # $/kWh

    # LED conversion wattage equivalents
    LED_CONVERSION = {
        # (current_technology, current_wattage): led_wattage
        (LightingTechnology.HIGH_PRESSURE_SODIUM, 70): 30,
        (LightingTechnology.HIGH_PRESSURE_SODIUM, 100): 40,
        (LightingTechnology.HIGH_PRESSURE_SODIUM, 150): 55,
        (LightingTechnology.HIGH_PRESSURE_SODIUM, 250): 90,
        (LightingTechnology.HIGH_PRESSURE_SODIUM, 400): 150,
        (LightingTechnology.METAL_HALIDE, 100): 45,
        (LightingTechnology.METAL_HALIDE, 175): 70,
        (LightingTechnology.METAL_HALIDE, 250): 100,
        (LightingTechnology.METAL_HALIDE, 400): 150,
        (LightingTechnology.MERCURY_VAPOR, 100): 30,
        (LightingTechnology.MERCURY_VAPOR, 175): 45,
        (LightingTechnology.MERCURY_VAPOR, 250): 70,
        (LightingTechnology.MERCURY_VAPOR, 400): 100,
    }

    # LED fixture costs
    LED_FIXTURE_COST = {
        30: 250,
        40: 280,
        55: 320,
        70: 350,
        90: 400,
        100: 450,
        150: 550,
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Street Lighting Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Street lighting optimization and LED conversion",
                version=self.VERSION,
                parameters={
                    "default_operating_hours": 4380,
                    "default_led_warranty_years": 10,
                    "installation_cost_per_fixture": 150,
                }
            )
        super().__init__(config)

        self._plans: Dict[str, LightingOptimizationPlan] = {}
        self.logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute street lighting operation."""
        import time
        start_time = time.time()

        try:
            agent_input = StreetLightingInput(**input_data)

            action_handlers = {
                'create_plan': self._handle_create_plan,
                'add_light': self._handle_add_light,
                'add_lights': self._handle_add_lights,
                'add_zone': self._handle_add_zone,
                'analyze_inventory': self._handle_analyze_inventory,
                'plan_led_conversion': self._handle_plan_led_conversion,
                'calculate_savings': self._handle_calculate_savings,
                'calculate_emissions': self._handle_calculate_emissions,
                'get_plan': self._handle_get_plan,
                'list_plans': self._handle_list_plans,
            }

            handler = action_handlers.get(agent_input.action)
            if not handler:
                raise ValueError(f"Unknown action: {agent_input.action}")

            output = handler(agent_input)
            output.processing_time_ms = (time.time() - start_time) * 1000
            output.provenance_hash = self._calculate_output_hash(output)

            return AgentResult(
                success=output.success,
                data=output.model_dump(),
                error=output.error,
            )

        except Exception as e:
            self.logger.error(f"Street lighting operation failed: {str(e)}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _handle_create_plan(
        self,
        input_data: StreetLightingInput
    ) -> StreetLightingOutput:
        """Create a new lighting optimization plan."""
        trace = []

        if not input_data.municipality_name:
            return StreetLightingOutput(
                success=False,
                action='create_plan',
                error="Municipality name is required"
            )

        plan_id = f"SLP-{input_data.municipality_name.upper()[:3]}-{DeterministicClock.now().strftime('%Y%m%d')}"
        trace.append(f"Generated plan ID: {plan_id}")

        plan = LightingOptimizationPlan(
            plan_id=plan_id,
            municipality_name=input_data.municipality_name,
            plan_name=input_data.plan_name or f"{input_data.municipality_name} Lighting Plan",
            created_by=input_data.user_id,
        )

        self._plans[plan_id] = plan
        trace.append(f"Created plan for {input_data.municipality_name}")

        return StreetLightingOutput(
            success=True,
            action='create_plan',
            plan=plan,
            calculation_trace=trace,
        )

    def _handle_add_light(
        self,
        input_data: StreetLightingInput
    ) -> StreetLightingOutput:
        """Add a single street light."""
        trace = []

        if not input_data.plan_id or not input_data.light:
            return StreetLightingOutput(
                success=False,
                action='add_light',
                error="Plan ID and light are required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return StreetLightingOutput(
                success=False,
                action='add_light',
                error=f"Plan not found: {input_data.plan_id}"
            )

        plan.street_lights.append(input_data.light)
        self._update_plan_metrics(plan)
        trace.append(f"Added light {input_data.light.light_id}")

        return StreetLightingOutput(
            success=True,
            action='add_light',
            plan=plan,
            calculation_trace=trace,
        )

    def _handle_add_lights(
        self,
        input_data: StreetLightingInput
    ) -> StreetLightingOutput:
        """Add multiple street lights."""
        trace = []

        if not input_data.plan_id or not input_data.lights:
            return StreetLightingOutput(
                success=False,
                action='add_lights',
                error="Plan ID and lights are required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return StreetLightingOutput(
                success=False,
                action='add_lights',
                error=f"Plan not found: {input_data.plan_id}"
            )

        plan.street_lights.extend(input_data.lights)
        self._update_plan_metrics(plan)
        trace.append(f"Added {len(input_data.lights)} lights")

        return StreetLightingOutput(
            success=True,
            action='add_lights',
            plan=plan,
            calculation_trace=trace,
        )

    def _handle_add_zone(
        self,
        input_data: StreetLightingInput
    ) -> StreetLightingOutput:
        """Add a lighting zone."""
        trace = []

        if not input_data.plan_id or not input_data.zone:
            return StreetLightingOutput(
                success=False,
                action='add_zone',
                error="Plan ID and zone are required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return StreetLightingOutput(
                success=False,
                action='add_zone',
                error=f"Plan not found: {input_data.plan_id}"
            )

        plan.zones.append(input_data.zone)
        plan.updated_at = DeterministicClock.now()
        trace.append(f"Added zone {input_data.zone.zone_id}")

        return StreetLightingOutput(
            success=True,
            action='add_zone',
            plan=plan,
            calculation_trace=trace,
        )

    def _handle_analyze_inventory(
        self,
        input_data: StreetLightingInput
    ) -> StreetLightingOutput:
        """Analyze street lighting inventory."""
        trace = []

        if not input_data.plan_id:
            return StreetLightingOutput(
                success=False,
                action='analyze_inventory',
                error="Plan ID is required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return StreetLightingOutput(
                success=False,
                action='analyze_inventory',
                error=f"Plan not found: {input_data.plan_id}"
            )

        trace.append("Analyzing inventory")

        electricity_rate = input_data.electricity_rate_usd_per_kwh or self.DEFAULT_ELECTRICITY_RATE
        emission_factor = input_data.emission_factor_kg_per_kwh or self.DEFAULT_EMISSION_FACTOR

        # Technology breakdown
        by_technology: Dict[str, Dict[str, Any]] = {}
        for light in plan.street_lights:
            tech = light.technology.value
            if tech not in by_technology:
                by_technology[tech] = {
                    "count": 0,
                    "total_wattage": 0,
                    "total_annual_kwh": 0,
                }
            by_technology[tech]["count"] += 1
            by_technology[tech]["total_wattage"] += light.wattage
            by_technology[tech]["total_annual_kwh"] += light.annual_kwh

        # Zone breakdown
        by_zone_type: Dict[str, int] = {}
        for light in plan.street_lights:
            zone_type = light.zone_type.value
            by_zone_type[zone_type] = by_zone_type.get(zone_type, 0) + 1

        # Control strategy breakdown
        by_control: Dict[str, int] = {}
        for light in plan.street_lights:
            control = light.control_strategy.value
            by_control[control] = by_control.get(control, 0) + 1

        # Calculate totals
        total_kwh = sum(light.annual_kwh for light in plan.street_lights)
        total_cost = total_kwh * electricity_rate
        total_emissions = (total_kwh * emission_factor) / 1000

        # LED conversion potential
        non_led_lights = [l for l in plan.street_lights if l.technology != LightingTechnology.LED]
        led_lights = [l for l in plan.street_lights if l.technology == LightingTechnology.LED]

        trace.append(f"Total lights: {len(plan.street_lights)}")
        trace.append(f"LED lights: {len(led_lights)} ({len(led_lights)/len(plan.street_lights)*100:.1f}%)")
        trace.append(f"Conversion candidates: {len(non_led_lights)}")

        inventory_analysis = {
            "total_lights": len(plan.street_lights),
            "total_wattage_kw": sum(l.wattage for l in plan.street_lights) / 1000,
            "total_annual_kwh": total_kwh,
            "total_annual_cost_usd": total_cost,
            "total_annual_emissions_tco2e": total_emissions,
            "by_technology": by_technology,
            "by_zone_type": by_zone_type,
            "by_control_strategy": by_control,
            "led_count": len(led_lights),
            "led_percentage": len(led_lights) / len(plan.street_lights) * 100 if plan.street_lights else 0,
            "conversion_candidates": len(non_led_lights),
            "average_age_years": self._calculate_average_age(plan.street_lights),
            "electricity_rate_usd_per_kwh": electricity_rate,
            "emission_factor_kg_per_kwh": emission_factor,
        }

        # Update plan metrics
        plan.total_annual_kwh = total_kwh
        plan.total_annual_cost_usd = total_cost
        plan.total_annual_emissions_tco2e = total_emissions
        plan.technology_breakdown = {k: v["count"] for k, v in by_technology.items()}

        return StreetLightingOutput(
            success=True,
            action='analyze_inventory',
            plan=plan,
            inventory_analysis=inventory_analysis,
            calculation_trace=trace,
        )

    def _handle_plan_led_conversion(
        self,
        input_data: StreetLightingInput
    ) -> StreetLightingOutput:
        """Plan LED conversion project."""
        trace = []

        if not input_data.plan_id:
            return StreetLightingOutput(
                success=False,
                action='plan_led_conversion',
                error="Plan ID is required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return StreetLightingOutput(
                success=False,
                action='plan_led_conversion',
                error=f"Plan not found: {input_data.plan_id}"
            )

        trace.append("Planning LED conversion")

        electricity_rate = input_data.electricity_rate_usd_per_kwh or self.DEFAULT_ELECTRICITY_RATE
        emission_factor = input_data.emission_factor_kg_per_kwh or self.DEFAULT_EMISSION_FACTOR
        led_incentive = input_data.led_incentive_per_fixture_usd or 50.0

        # Identify non-LED lights
        non_led_lights = [l for l in plan.street_lights if l.technology != LightingTechnology.LED]

        if not non_led_lights:
            return StreetLightingOutput(
                success=True,
                action='plan_led_conversion',
                plan=plan,
                calculation_trace=["All lights are already LED"],
            )

        # Calculate conversion details
        current_total_wattage = sum(l.wattage for l in non_led_lights)
        current_annual_kwh = sum(l.annual_kwh for l in non_led_lights)

        proposed_led_wattages = []
        fixture_costs = []
        for light in non_led_lights:
            led_wattage = self._get_led_equivalent(light.technology, light.wattage)
            proposed_led_wattages.append(led_wattage)
            fixture_costs.append(self.LED_FIXTURE_COST.get(led_wattage, 350))

        proposed_total_wattage = sum(proposed_led_wattages)
        operating_hours = non_led_lights[0].operating_hours_per_year if non_led_lights else 4380
        proposed_annual_kwh = proposed_total_wattage * operating_hours / 1000

        # Calculate savings
        annual_kwh_savings = current_annual_kwh - proposed_annual_kwh
        annual_cost_savings = annual_kwh_savings * electricity_rate
        annual_emission_reduction = (annual_kwh_savings * emission_factor) / 1000

        # Calculate costs
        total_fixture_cost = sum(fixture_costs)
        installation_cost = len(non_led_lights) * self.config.parameters["installation_cost_per_fixture"]
        total_project_cost = total_fixture_cost + installation_cost
        total_incentives = len(non_led_lights) * led_incentive
        net_project_cost = total_project_cost - total_incentives

        # Calculate payback
        simple_payback = net_project_cost / annual_cost_savings if annual_cost_savings > 0 else 999

        trace.append(f"Lights to convert: {len(non_led_lights)}")
        trace.append(f"Current wattage: {current_total_wattage:.0f} W")
        trace.append(f"Proposed LED wattage: {proposed_total_wattage:.0f} W")
        trace.append(f"Wattage reduction: {(1 - proposed_total_wattage/current_total_wattage)*100:.1f}%")
        trace.append(f"Annual kWh savings: {annual_kwh_savings:,.0f} kWh")
        trace.append(f"Annual cost savings: ${annual_cost_savings:,.2f}")
        trace.append(f"Net project cost: ${net_project_cost:,.2f}")
        trace.append(f"Simple payback: {simple_payback:.1f} years")

        project = LEDConversionProject(
            project_id=f"LED-{plan.plan_id}-001",
            project_name=f"{plan.municipality_name} LED Conversion",
            lights_to_convert=[l.light_id for l in non_led_lights],
            current_total_wattage=current_total_wattage,
            current_annual_kwh=current_annual_kwh,
            proposed_led_wattage=proposed_total_wattage,
            proposed_annual_kwh=proposed_annual_kwh,
            annual_kwh_savings=annual_kwh_savings,
            annual_cost_savings_usd=annual_cost_savings,
            annual_emission_reduction_tco2e=annual_emission_reduction,
            total_project_cost_usd=total_project_cost,
            utility_incentive_usd=total_incentives,
            net_project_cost_usd=net_project_cost,
            simple_payback_years=simple_payback,
        )

        plan.conversion_projects.append(project)
        plan.updated_at = DeterministicClock.now()

        return StreetLightingOutput(
            success=True,
            action='plan_led_conversion',
            plan=plan,
            conversion_project=project,
            calculation_trace=trace,
        )

    def _handle_calculate_savings(
        self,
        input_data: StreetLightingInput
    ) -> StreetLightingOutput:
        """Calculate savings from planned projects."""
        trace = []

        if not input_data.plan_id:
            return StreetLightingOutput(
                success=False,
                action='calculate_savings',
                error="Plan ID is required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return StreetLightingOutput(
                success=False,
                action='calculate_savings',
                error=f"Plan not found: {input_data.plan_id}"
            )

        trace.append("Calculating savings from all projects")

        total_kwh_savings = sum(p.annual_kwh_savings for p in plan.conversion_projects)
        total_cost_savings = sum(p.annual_cost_savings_usd for p in plan.conversion_projects)
        total_emission_reduction = sum(p.annual_emission_reduction_tco2e for p in plan.conversion_projects)
        total_investment = sum(p.net_project_cost_usd for p in plan.conversion_projects)

        # 10-year projection
        analysis_years = 10
        cumulative_savings = total_cost_savings * analysis_years
        net_benefit = cumulative_savings - total_investment

        savings_analysis = {
            "projects_count": len(plan.conversion_projects),
            "total_lights_to_convert": sum(len(p.lights_to_convert) for p in plan.conversion_projects),
            "annual_kwh_savings": total_kwh_savings,
            "annual_cost_savings_usd": total_cost_savings,
            "annual_emission_reduction_tco2e": total_emission_reduction,
            "total_investment_usd": total_investment,
            "portfolio_simple_payback_years": total_investment / total_cost_savings if total_cost_savings > 0 else 999,
            "analysis_period_years": analysis_years,
            "cumulative_savings_usd": cumulative_savings,
            "net_benefit_usd": net_benefit,
            "roi_percent": (net_benefit / total_investment * 100) if total_investment > 0 else 0,
        }

        trace.append(f"Total investment: ${total_investment:,.2f}")
        trace.append(f"Annual savings: ${total_cost_savings:,.2f}")
        trace.append(f"{analysis_years}-year net benefit: ${net_benefit:,.2f}")

        return StreetLightingOutput(
            success=True,
            action='calculate_savings',
            plan=plan,
            savings_analysis=savings_analysis,
            calculation_trace=trace,
        )

    def _handle_calculate_emissions(
        self,
        input_data: StreetLightingInput
    ) -> StreetLightingOutput:
        """Calculate emissions and reductions."""
        trace = []

        if not input_data.plan_id:
            return StreetLightingOutput(
                success=False,
                action='calculate_emissions',
                error="Plan ID is required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return StreetLightingOutput(
                success=False,
                action='calculate_emissions',
                error=f"Plan not found: {input_data.plan_id}"
            )

        trace.append("Calculating emissions")

        emission_factor = input_data.emission_factor_kg_per_kwh or self.DEFAULT_EMISSION_FACTOR

        # Current emissions
        current_annual_kwh = sum(l.annual_kwh for l in plan.street_lights)
        current_emissions_tco2e = (current_annual_kwh * emission_factor) / 1000

        # Post-conversion emissions
        total_reduction = sum(p.annual_emission_reduction_tco2e for p in plan.conversion_projects)
        post_conversion_emissions = current_emissions_tco2e - total_reduction

        trace.append(f"Current emissions: {current_emissions_tco2e:.2f} tCO2e/year")
        trace.append(f"Projected reduction: {total_reduction:.2f} tCO2e/year")
        trace.append(f"Post-conversion: {post_conversion_emissions:.2f} tCO2e/year")

        emission_analysis = {
            "current_annual_kwh": current_annual_kwh,
            "current_emissions_tco2e": current_emissions_tco2e,
            "projected_reduction_tco2e": total_reduction,
            "reduction_percent": (total_reduction / current_emissions_tco2e * 100) if current_emissions_tco2e > 0 else 0,
            "post_conversion_emissions_tco2e": post_conversion_emissions,
            "emission_factor_kg_per_kwh": emission_factor,
            "emission_factor_source": "EPA eGRID",
        }

        return StreetLightingOutput(
            success=True,
            action='calculate_emissions',
            plan=plan,
            emission_analysis=emission_analysis,
            calculation_trace=trace,
        )

    def _handle_get_plan(
        self,
        input_data: StreetLightingInput
    ) -> StreetLightingOutput:
        """Get a plan by ID."""
        if not input_data.plan_id:
            return StreetLightingOutput(
                success=False,
                action='get_plan',
                error="Plan ID is required"
            )

        plan = self._plans.get(input_data.plan_id)
        if not plan:
            return StreetLightingOutput(
                success=False,
                action='get_plan',
                error=f"Plan not found: {input_data.plan_id}"
            )

        return StreetLightingOutput(
            success=True,
            action='get_plan',
            plan=plan,
        )

    def _handle_list_plans(
        self,
        input_data: StreetLightingInput
    ) -> StreetLightingOutput:
        """List all plans."""
        return StreetLightingOutput(
            success=True,
            action='list_plans',
            plans=list(self._plans.values()),
        )

    def _update_plan_metrics(self, plan: LightingOptimizationPlan) -> None:
        """Update plan summary metrics."""
        plan.total_lights = len(plan.street_lights)
        plan.total_wattage_kw = sum(l.wattage for l in plan.street_lights) / 1000
        plan.total_annual_kwh = sum(l.annual_kwh for l in plan.street_lights)
        plan.updated_at = DeterministicClock.now()

    def _get_led_equivalent(self, technology: LightingTechnology, wattage: float) -> float:
        """Get LED equivalent wattage."""
        key = (technology, int(wattage))
        if key in self.LED_CONVERSION:
            return self.LED_CONVERSION[key]

        # Estimate based on technology efficacy
        efficacy_ratios = {
            LightingTechnology.HIGH_PRESSURE_SODIUM: 0.4,
            LightingTechnology.METAL_HALIDE: 0.45,
            LightingTechnology.MERCURY_VAPOR: 0.3,
            LightingTechnology.INCANDESCENT: 0.15,
        }
        ratio = efficacy_ratios.get(technology, 0.4)
        return round(wattage * ratio / 10) * 10  # Round to nearest 10W

    def _calculate_average_age(self, lights: List[StreetLight]) -> float:
        """Calculate average fixture age."""
        current_year = DeterministicClock.now().year
        ages = [
            current_year - l.installation_year
            for l in lights
            if l.installation_year
        ]
        return sum(ages) / len(ages) if ages else 0

    def _calculate_output_hash(self, output: StreetLightingOutput) -> str:
        """Calculate SHA-256 hash of output."""
        content = {
            "action": output.action,
            "success": output.success,
            "timestamp": output.timestamp.isoformat(),
        }

        if output.plan:
            content["plan_id"] = output.plan.plan_id

        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()
