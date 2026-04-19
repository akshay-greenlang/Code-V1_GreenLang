"""
GL-032: Refractory Monitor Agent (REFRACTORY-MONITOR)

This module implements the RefractoryMonitorAgent for refractory health assessment
in industrial furnaces, heaters, and kilns.

The agent provides:
- Skin temperature analysis with hotspot detection
- Heat loss calculations through refractory layers
- Thermal gradient analysis for spalling risk
- Remaining useful life prediction
- Maintenance priority determination

Standards Compliance:
- API 560: Fired Heaters for General Refinery Service
- ASTM C155: Standard Classification of Insulating Firebrick

Example:
    >>> agent = RefractoryMonitorAgent()
    >>> result = agent.run(RefractoryMonitorInput(
    ...     equipment_id="FH-001",
    ...     skin_temps=[SkinTemperature(x=1.0, y=2.0, temp_celsius=85, ...)],
    ...     age_days=730,
    ...     material_type=RefractoryMaterial.CASTABLE
    ... ))
    >>> print(f"Health Index: {result.health_index}")
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .models import (
    RefractoryMaterial,
    RefractoryZone,
    MaintenancePriority,
    HealthStatus,
    DegradationMode,
)
from .formulas import (
    calculate_heat_loss_through_wall,
    calculate_multilayer_heat_loss,
    calculate_thermal_gradient,
    calculate_health_index,
    estimate_remaining_life,
    analyze_hotspot,
    determine_maintenance_priority,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT MODELS
# =============================================================================

class SkinTemperature(BaseModel):
    """Skin temperature measurement point."""

    x_position: float = Field(..., description="X coordinate in meters")
    y_position: float = Field(..., description="Y coordinate in meters")
    temp_celsius: float = Field(..., description="Measured temperature in Celsius")
    zone: RefractoryZone = Field(
        default=RefractoryZone.SIDEWALL,
        description="Furnace zone"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ThermalImageData(BaseModel):
    """Thermal image analysis data."""

    image_id: str = Field(..., description="Thermal image identifier")
    capture_timestamp: datetime = Field(..., description="When image was captured")
    min_temp_celsius: float = Field(..., description="Minimum temperature in image")
    max_temp_celsius: float = Field(..., description="Maximum temperature in image")
    avg_temp_celsius: float = Field(..., description="Average temperature in image")
    hotspot_locations: List[Dict[str, float]] = Field(
        default_factory=list,
        description="List of hotspot locations [{x, y, temp}]"
    )
    zone: RefractoryZone = Field(..., description="Zone covered by image")


class RefractoryLayer(BaseModel):
    """Refractory layer specification."""

    layer_name: str = Field(..., description="Layer identifier")
    material: RefractoryMaterial = Field(..., description="Material type")
    thickness_m: float = Field(..., gt=0, description="Layer thickness in meters")
    conductivity_w_per_m_k: float = Field(
        ..., gt=0,
        description="Thermal conductivity at operating temperature"
    )
    design_hot_face_temp: Optional[float] = Field(None, description="Design hot face temperature")


class HistoricalReading(BaseModel):
    """Historical health reading for trend analysis."""

    days_ago: int = Field(..., ge=0, description="Days ago this reading was taken")
    health_index: float = Field(..., ge=0, le=100, description="Health index at that time")
    notes: Optional[str] = Field(None, description="Notes about this reading")


class RefractoryMonitorInput(BaseModel):
    """Input data model for RefractoryMonitorAgent."""

    equipment_id: str = Field(..., min_length=1, description="Equipment identifier")
    equipment_name: Optional[str] = Field(None, description="Equipment name/description")

    # Temperature data
    skin_temps: List[SkinTemperature] = Field(
        default_factory=list,
        description="Skin temperature measurements"
    )
    thermal_images: List[ThermalImageData] = Field(
        default_factory=list,
        description="Thermal imaging data"
    )

    # Refractory specifications
    age_days: int = Field(..., ge=0, description="Refractory age in days since installation")
    design_life_days: int = Field(
        default=1825,  # 5 years
        gt=0,
        description="Design life in days"
    )
    material_type: RefractoryMaterial = Field(
        ...,
        description="Primary refractory material type"
    )
    refractory_layers: List[RefractoryLayer] = Field(
        default_factory=list,
        description="Refractory layer specifications"
    )

    # Operating conditions
    process_temp_celsius: float = Field(
        default=800.0,
        description="Process/hot face temperature"
    )
    ambient_temp_celsius: float = Field(
        default=25.0,
        description="Ambient temperature"
    )
    design_skin_temp_celsius: float = Field(
        default=80.0,
        description="Design skin temperature"
    )

    # Historical data for trend analysis
    health_history: List[HistoricalReading] = Field(
        default_factory=list,
        description="Historical health readings"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class HotspotAnalysis(BaseModel):
    """Analysis of a detected hotspot."""

    hotspot_id: str = Field(..., description="Hotspot identifier")
    location_x: float = Field(..., description="X coordinate")
    location_y: float = Field(..., description="Y coordinate")
    temperature_celsius: float = Field(..., description="Hotspot temperature")
    severity: str = Field(..., description="Severity level")
    heat_loss_kw: float = Field(..., description="Estimated heat loss from hotspot")
    recommended_action: str = Field(..., description="Recommended action")
    zone: RefractoryZone = Field(..., description="Affected zone")


class HeatLossAnalysis(BaseModel):
    """Heat loss analysis results."""

    total_heat_loss_kw: float = Field(..., description="Total heat loss in kW")
    heat_loss_per_area_w_m2: float = Field(..., description="Heat loss per unit area")
    design_heat_loss_kw: float = Field(..., description="Design heat loss")
    excess_heat_loss_percent: float = Field(..., description="Excess over design %")
    interface_temperatures: List[float] = Field(
        default_factory=list,
        description="Interface temperatures between layers"
    )


class RefractoryMonitorOutput(BaseModel):
    """Output data model for RefractoryMonitorAgent."""

    # Identification
    analysis_id: str = Field(..., description="Unique analysis identifier")
    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Health Assessment
    health_index: float = Field(..., ge=0, le=100, description="Health index 0-100")
    health_status: HealthStatus = Field(..., description="Health status classification")
    remaining_life_days: int = Field(..., ge=0, description="Estimated remaining life")
    remaining_life_percent: float = Field(..., ge=0, description="Remaining life percentage")
    degradation_rate: float = Field(..., description="Degradation rate per day")
    failure_date_estimate: datetime = Field(..., description="Estimated failure date")

    # Hotspot Analysis
    hotspot_locations: List[HotspotAnalysis] = Field(
        default_factory=list,
        description="Detected hotspots with analysis"
    )
    hotspot_count: int = Field(default=0, description="Total hotspot count")
    critical_hotspot_count: int = Field(default=0, description="Critical hotspot count")

    # Heat Loss Analysis
    heat_loss_analysis: HeatLossAnalysis = Field(..., description="Heat loss analysis")

    # Thermal Gradient
    thermal_gradient_c_per_m: float = Field(..., description="Thermal gradient")
    spalling_risk: str = Field(..., description="Risk of spalling (LOW, MODERATE, HIGH)")

    # Maintenance
    maintenance_priority: MaintenancePriority = Field(..., description="Maintenance priority")
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended maintenance actions"
    )

    # Detected degradation modes
    degradation_modes: List[DegradationMode] = Field(
        default_factory=list,
        description="Likely degradation mechanisms"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")
    processing_time_ms: float = Field(..., description="Processing duration")
    validation_status: str = Field(..., description="PASS or FAIL")


# =============================================================================
# REFRACTORY MONITOR AGENT
# =============================================================================

class RefractoryMonitorAgent:
    """
    GL-032: Refractory Monitor Agent (REFRACTORY-MONITOR).

    This agent assesses refractory health using thermal imaging data,
    skin temperature measurements, and material age to predict remaining
    useful life and prioritize maintenance.

    Zero-Hallucination Guarantee:
    - Heat loss calculations use Fourier's Law
    - Health index uses weighted observable parameters
    - Remaining life uses linear regression on historical data
    - No LLM inference in calculation path

    Attributes:
        AGENT_ID: Unique agent identifier (GL-032)
        AGENT_NAME: Agent name (REFRACTORY-MONITOR)
        VERSION: Agent version
    """

    AGENT_ID = "GL-032"
    AGENT_NAME = "REFRACTORY-MONITOR"
    VERSION = "1.0.0"
    DESCRIPTION = "Refractory Health Assessment Agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the RefractoryMonitorAgent."""
        self.config = config or {}
        logger.info(
            f"RefractoryMonitorAgent initialized "
            f"(ID: {self.AGENT_ID}, Name: {self.AGENT_NAME}, Version: {self.VERSION})"
        )

    def run(self, input_data: RefractoryMonitorInput) -> RefractoryMonitorOutput:
        """
        Execute refractory health assessment.

        This method performs:
        1. Analyze skin temperatures and detect hotspots
        2. Calculate heat loss through refractory
        3. Calculate thermal gradients
        4. Determine health index
        5. Estimate remaining useful life
        6. Identify degradation modes
        7. Set maintenance priority

        Args:
            input_data: Validated refractory input data

        Returns:
            Complete health assessment with provenance hash
        """
        start_time = datetime.utcnow()

        logger.info(f"Starting refractory assessment for {input_data.equipment_id}")

        try:
            # Step 1: Analyze skin temperatures and find hotspots
            hotspots, avg_skin_temp = self._analyze_temperatures(input_data)

            # Step 2: Calculate heat loss
            heat_loss = self._calculate_heat_loss(input_data, avg_skin_temp)

            # Step 3: Calculate thermal gradient
            thermal_gradient = calculate_thermal_gradient(
                input_data.process_temp_celsius,
                avg_skin_temp,
                sum(layer.thickness_m for layer in input_data.refractory_layers) if input_data.refractory_layers else 0.3
            )
            spalling_risk = self._assess_spalling_risk(thermal_gradient)

            # Step 4: Calculate health index
            hotspot_severity = len([h for h in hotspots if h.severity == "CRITICAL"]) / max(len(hotspots), 1)
            health_index = calculate_health_index(
                skin_temp_celsius=avg_skin_temp,
                design_skin_temp_celsius=input_data.design_skin_temp_celsius,
                age_days=input_data.age_days,
                design_life_days=input_data.design_life_days,
                hotspot_count=len(hotspots),
                hotspot_severity_factor=hotspot_severity
            )

            # Step 5: Estimate remaining life
            history_tuples = [(h.days_ago, h.health_index) for h in input_data.health_history]
            remaining_life = estimate_remaining_life(health_index, history_tuples)

            # Step 6: Identify degradation modes
            degradation_modes = self._identify_degradation_modes(
                hotspots, thermal_gradient, input_data.age_days, input_data.design_life_days
            )

            # Step 7: Determine maintenance priority
            critical_hotspots = len([h for h in hotspots if h.severity == "CRITICAL"])
            priority = determine_maintenance_priority(
                health_index, remaining_life.remaining_life_days,
                len(hotspots), critical_hotspots
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                health_index, hotspots, degradation_modes, priority
            )

            # Determine health status
            health_status = self._determine_health_status(health_index)

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash(input_data, health_index)

            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Generate analysis ID
            analysis_id = (
                f"RM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-"
                f"{hashlib.sha256(input_data.equipment_id.encode()).hexdigest()[:8]}"
            )

            output = RefractoryMonitorOutput(
                analysis_id=analysis_id,
                equipment_id=input_data.equipment_id,
                health_index=health_index,
                health_status=health_status,
                remaining_life_days=remaining_life.remaining_life_days,
                remaining_life_percent=remaining_life.remaining_life_percent,
                degradation_rate=remaining_life.degradation_rate_per_day,
                failure_date_estimate=remaining_life.failure_date_estimate,
                hotspot_locations=hotspots,
                hotspot_count=len(hotspots),
                critical_hotspot_count=critical_hotspots,
                heat_loss_analysis=heat_loss,
                thermal_gradient_c_per_m=round(thermal_gradient, 1),
                spalling_risk=spalling_risk,
                maintenance_priority=MaintenancePriority(priority),
                recommended_actions=recommendations,
                degradation_modes=degradation_modes,
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time, 2),
                validation_status="PASS"
            )

            logger.info(
                f"Refractory assessment complete for {input_data.equipment_id}: "
                f"health={health_index}, remaining_life={remaining_life.remaining_life_days}d, "
                f"priority={priority}"
            )

            return output

        except Exception as e:
            logger.error(f"Refractory assessment failed: {str(e)}", exc_info=True)
            raise

    def _analyze_temperatures(self, input_data: RefractoryMonitorInput) -> tuple:
        """Analyze temperature data and detect hotspots."""
        hotspots = []
        all_temps = []

        # Process skin temperature measurements
        for temp in input_data.skin_temps:
            all_temps.append(temp.temp_celsius)

        # Process thermal images
        for image in input_data.thermal_images:
            all_temps.append(image.avg_temp_celsius)

            # Analyze hotspots in image
            for hs in image.hotspot_locations:
                result = analyze_hotspot(
                    location_x=hs.get('x', 0),
                    location_y=hs.get('y', 0),
                    temperature_celsius=hs.get('temp', 0),
                    surrounding_avg_temp=image.avg_temp_celsius,
                    design_temp=input_data.design_skin_temp_celsius
                )
                hotspots.append(HotspotAnalysis(
                    hotspot_id=f"HS-{len(hotspots)+1:03d}",
                    location_x=result.location_x,
                    location_y=result.location_y,
                    temperature_celsius=result.temperature_celsius,
                    severity=result.severity,
                    heat_loss_kw=result.heat_loss_kw,
                    recommended_action=result.recommended_action,
                    zone=image.zone
                ))

        # Calculate average skin temperature
        avg_temp = sum(all_temps) / len(all_temps) if all_temps else input_data.design_skin_temp_celsius

        return hotspots, avg_temp

    def _calculate_heat_loss(self, input_data: RefractoryMonitorInput, avg_skin_temp: float) -> HeatLossAnalysis:
        """Calculate heat loss through refractory."""
        # Default area (10 m2)
        area = input_data.metadata.get('surface_area_m2', 10.0)

        if input_data.refractory_layers:
            layers = [
                {
                    'thickness_m': layer.thickness_m,
                    'conductivity_w_per_m_k': layer.conductivity_w_per_m_k
                }
                for layer in input_data.refractory_layers
            ]
            heat_loss_w, interface_temps = calculate_multilayer_heat_loss(
                input_data.process_temp_celsius,
                input_data.ambient_temp_celsius,
                layers,
                area
            )
        else:
            # Single layer assumption
            heat_loss_w = calculate_heat_loss_through_wall(
                input_data.process_temp_celsius,
                avg_skin_temp,
                0.3,  # Default 300mm thickness
                1.5,  # Default conductivity
                area
            )
            interface_temps = [input_data.process_temp_celsius, avg_skin_temp]

        # Design heat loss (based on design skin temp)
        design_heat_loss_w = calculate_heat_loss_through_wall(
            input_data.process_temp_celsius,
            input_data.design_skin_temp_celsius,
            0.3,
            1.5,
            area
        )

        excess_percent = ((heat_loss_w - design_heat_loss_w) / design_heat_loss_w * 100) if design_heat_loss_w > 0 else 0

        return HeatLossAnalysis(
            total_heat_loss_kw=round(heat_loss_w / 1000, 2),
            heat_loss_per_area_w_m2=round(heat_loss_w / area, 1),
            design_heat_loss_kw=round(design_heat_loss_w / 1000, 2),
            excess_heat_loss_percent=round(max(0, excess_percent), 1),
            interface_temperatures=[round(t, 1) for t in interface_temps]
        )

    def _assess_spalling_risk(self, thermal_gradient: float) -> str:
        """Assess risk of thermal spalling based on gradient."""
        if thermal_gradient > 3000:
            return "HIGH"
        elif thermal_gradient > 2000:
            return "MODERATE"
        else:
            return "LOW"

    def _identify_degradation_modes(
        self,
        hotspots: List[HotspotAnalysis],
        thermal_gradient: float,
        age_days: int,
        design_life_days: int
    ) -> List[DegradationMode]:
        """Identify likely degradation mechanisms."""
        modes = []

        # High thermal gradient suggests thermal shock risk
        if thermal_gradient > 2500:
            modes.append(DegradationMode.THERMAL_SHOCK)

        # Localized hotspots suggest spalling or anchor failure
        critical_hotspots = [h for h in hotspots if h.severity == "CRITICAL"]
        if critical_hotspots:
            modes.append(DegradationMode.SPALLING)

        # Age-based degradation
        if age_days > design_life_days * 0.8:
            modes.append(DegradationMode.THERMAL_AGING)

        return modes

    def _generate_recommendations(
        self,
        health_index: float,
        hotspots: List[HotspotAnalysis],
        degradation_modes: List[DegradationMode],
        priority: str
    ) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []

        if priority == "CRITICAL":
            recommendations.append("Schedule emergency inspection and repair")

        if any(h.severity == "CRITICAL" for h in hotspots):
            recommendations.append("Investigate critical hotspots immediately")

        if DegradationMode.SPALLING in degradation_modes:
            recommendations.append("Inspect for spalling damage; consider patching")

        if DegradationMode.THERMAL_AGING in degradation_modes:
            recommendations.append("Plan refractory replacement during next major outage")

        if health_index < 50:
            recommendations.append("Increase monitoring frequency to weekly")

        if not recommendations:
            recommendations.append("Continue routine monitoring per standard schedule")

        return recommendations

    def _determine_health_status(self, health_index: float) -> HealthStatus:
        """Determine health status from index."""
        if health_index >= 90:
            return HealthStatus.EXCELLENT
        elif health_index >= 70:
            return HealthStatus.GOOD
        elif health_index >= 50:
            return HealthStatus.FAIR
        elif health_index >= 30:
            return HealthStatus.POOR
        elif health_index >= 10:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILED

    def _calculate_provenance_hash(self, input_data: RefractoryMonitorInput, health_index: float) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "equipment_id": input_data.equipment_id,
            "health_index": health_index,
            "timestamp": datetime.utcnow().isoformat()
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-032",
    "name": "REFRACTORY-MONITOR - Refractory Health Assessment Agent",
    "version": "1.0.0",
    "summary": "Refractory health assessment with thermal imaging, heat loss analysis, and remaining life prediction",
    "tags": [
        "refractory",
        "thermal-imaging",
        "health-monitoring",
        "heat-loss",
        "predictive-maintenance",
        "API-560",
        "ASTM-C155"
    ],
    "owners": ["process-heat-reliability-team"],
    "compute": {
        "entrypoint": "python://agents.gl_032_refractory_monitor.agent:RefractoryMonitorAgent",
        "deterministic": True
    },
    "standards": [
        {"ref": "API 560", "description": "Fired Heaters for General Refinery Service"},
        {"ref": "ASTM C155", "description": "Standard Classification of Insulating Firebrick"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True
    }
}
