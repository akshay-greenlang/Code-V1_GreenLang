"""
GL-015 INSULSCAN - IR Thermography Survey Integration

Integrates IR thermography survey data for insulation condition
assessment, hot spot identification, and heat loss quantification.

All calculations are DETERMINISTIC - zero hallucination.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math
import uuid

from greenlang.agents.process_heat.gl_015_insulation_analysis.config import (
    IRSurveyConfig,
    InsulationAnalysisConfig,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.schemas import (
    InsulationInput,
    IRSurveyResult,
    IRHotSpot,
    GeometryType,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.materials import (
    InsulationMaterialDatabase,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.heat_loss import (
    HeatLossCalculator,
)

logger = logging.getLogger(__name__)


@dataclass
class ThermalImageData:
    """Thermal image data from IR camera."""
    image_id: str
    location: str
    timestamp: datetime
    min_temp_f: float
    max_temp_f: float
    avg_temp_f: float
    spot_temps: Dict[str, float]  # Named spots
    ambient_temp_f: float
    emissivity: float
    distance_ft: float
    camera_model: Optional[str] = None
    image_path: Optional[str] = None


@dataclass
class AnomalyDetection:
    """Detected thermal anomaly."""
    anomaly_id: str
    location: str
    measured_temp_f: float
    expected_temp_f: float
    delta_t_f: float
    anomaly_type: str  # hot_spot, damaged, missing, saturated
    severity: str  # low, medium, high, critical
    estimated_heat_loss_btu_hr: float
    probable_cause: str
    recommended_action: str


class IRThermographySurvey:
    """
    IR thermography survey analyzer for insulation assessment.

    Processes IR survey data to identify insulation deficiencies,
    quantify excess heat loss, and prioritize repairs.

    Features:
        - Hot spot detection and classification
        - Damaged/missing insulation identification
        - Heat loss quantification from anomalies
        - Repair prioritization
        - Survey reporting

    Methodology:
        - Compare measured surface temps to expected temps
        - Delta-T analysis for anomaly detection
        - Heat loss calculation for identified issues
        - Cost-benefit analysis for repairs

    Attributes:
        heat_loss_calc: Heat loss calculator
        ir_config: IR survey configuration

    Example:
        >>> survey = IRThermographySurvey(config)
        >>> result = survey.analyze_survey(images, baseline_inputs)
        >>> print(f"Found {result.total_anomalies} anomalies")
    """

    def __init__(
        self,
        config: InsulationAnalysisConfig,
        material_database: Optional[InsulationMaterialDatabase] = None,
        heat_loss_calculator: Optional[HeatLossCalculator] = None,
    ) -> None:
        """
        Initialize the IR thermography survey analyzer.

        Args:
            config: Analysis configuration
            material_database: Material database
            heat_loss_calculator: Heat loss calculator
        """
        self.config = config
        self.ir_config = config.ir_survey
        self.economic = config.economic
        self.material_db = material_database or InsulationMaterialDatabase()
        self.heat_loss_calc = heat_loss_calculator or HeatLossCalculator(
            material_database=self.material_db,
        )
        self._survey_count = 0

        logger.info("IRThermographySurvey initialized")

    def analyze_survey(
        self,
        thermal_images: List[ThermalImageData],
        baseline_inputs: Dict[str, InsulationInput],
    ) -> IRSurveyResult:
        """
        Analyze IR survey data and identify insulation issues.

        Args:
            thermal_images: List of thermal image data
            baseline_inputs: Expected inputs by location/tag

        Returns:
            IRSurveyResult with findings and recommendations
        """
        self._survey_count += 1
        survey_id = str(uuid.uuid4())

        logger.info(f"Analyzing IR survey {survey_id} with {len(thermal_images)} images")

        hot_spots: List[IRHotSpot] = []
        total_excess_heat_loss = 0.0
        ambient_temps = []
        wind_speeds = []

        for image in thermal_images:
            # Get baseline for this location
            baseline = baseline_inputs.get(image.location)

            if baseline is None:
                logger.warning(f"No baseline for location: {image.location}")
                continue

            ambient_temps.append(image.ambient_temp_f)

            # Calculate expected surface temperature
            expected_result = self.heat_loss_calc.calculate_heat_loss(baseline)
            expected_temp = expected_result.outer_surface_temperature_f

            # Analyze temperature spots
            for spot_name, measured_temp in image.spot_temps.items():
                delta_t = measured_temp - expected_temp

                # Check against thresholds
                anomaly = self._classify_anomaly(
                    location=f"{image.location}/{spot_name}",
                    measured_temp_f=measured_temp,
                    expected_temp_f=expected_temp,
                    delta_t_f=delta_t,
                    baseline_input=baseline,
                )

                if anomaly:
                    hot_spot = IRHotSpot(
                        spot_id=anomaly.anomaly_id,
                        location_description=anomaly.location,
                        measured_temperature_f=anomaly.measured_temp_f,
                        expected_temperature_f=anomaly.expected_temp_f,
                        delta_t_f=anomaly.delta_t_f,
                        severity=anomaly.severity,
                        estimated_heat_loss_btu_hr=anomaly.estimated_heat_loss_btu_hr,
                        probable_cause=anomaly.probable_cause,
                        recommended_action=anomaly.recommended_action,
                        image_reference=image.image_id,
                    )
                    hot_spots.append(hot_spot)
                    total_excess_heat_loss += anomaly.estimated_heat_loss_btu_hr

            # Also check overall image stats
            if image.max_temp_f > expected_temp + self.ir_config.hot_spot_threshold_delta_f:
                # Significant hot spot in image
                anomaly = self._classify_anomaly(
                    location=f"{image.location}/MAX",
                    measured_temp_f=image.max_temp_f,
                    expected_temp_f=expected_temp,
                    delta_t_f=image.max_temp_f - expected_temp,
                    baseline_input=baseline,
                )
                if anomaly:
                    hot_spot = IRHotSpot(
                        spot_id=anomaly.anomaly_id,
                        location_description=anomaly.location,
                        measured_temperature_f=anomaly.measured_temp_f,
                        expected_temperature_f=anomaly.expected_temp_f,
                        delta_t_f=anomaly.delta_t_f,
                        severity=anomaly.severity,
                        estimated_heat_loss_btu_hr=anomaly.estimated_heat_loss_btu_hr,
                        probable_cause=anomaly.probable_cause,
                        recommended_action=anomaly.recommended_action,
                        image_reference=image.image_id,
                    )
                    hot_spots.append(hot_spot)
                    total_excess_heat_loss += anomaly.estimated_heat_loss_btu_hr

        # Calculate annual cost
        annual_excess_cost = self._calculate_annual_energy_cost(total_excess_heat_loss)

        # Count critical repairs
        critical_repairs = sum(
            1 for hs in hot_spots if hs.severity == "critical"
        )

        # Estimate repair costs
        estimated_repair_cost = self._estimate_repair_cost(hot_spots)

        # Estimate annual savings
        estimated_savings = annual_excess_cost  # If all fixed

        # Average survey conditions
        avg_ambient = sum(ambient_temps) / len(ambient_temps) if ambient_temps else 77.0

        return IRSurveyResult(
            survey_id=survey_id,
            survey_date=datetime.now(timezone.utc),
            ambient_temperature_f=round(avg_ambient, 1),
            wind_speed_mph=0.0,  # Would come from survey conditions
            humidity_pct=50.0,   # Would come from survey conditions
            camera_settings={
                "emissivity": self.ir_config.emissivity_default,
                "distance_ft": self.ir_config.distance_ft,
            },
            items_surveyed=len(baseline_inputs),
            hot_spots_identified=hot_spots,
            total_anomalies=len(hot_spots),
            total_excess_heat_loss_btu_hr=round(total_excess_heat_loss, 0),
            annual_excess_energy_cost_usd=round(annual_excess_cost, 2),
            critical_repairs_needed=critical_repairs,
            estimated_repair_cost_usd=round(estimated_repair_cost, 2),
            estimated_annual_savings_usd=round(estimated_savings, 2),
        )

    def _classify_anomaly(
        self,
        location: str,
        measured_temp_f: float,
        expected_temp_f: float,
        delta_t_f: float,
        baseline_input: InsulationInput,
    ) -> Optional[AnomalyDetection]:
        """
        Classify a thermal anomaly.

        Args:
            location: Location description
            measured_temp_f: Measured temperature
            expected_temp_f: Expected temperature
            delta_t_f: Temperature difference
            baseline_input: Baseline input data

        Returns:
            AnomalyDetection or None if not an anomaly
        """
        # Check if delta-T exceeds threshold
        if abs(delta_t_f) < self.ir_config.hot_spot_threshold_delta_f:
            return None

        # Determine anomaly type and severity
        operating_temp = baseline_input.operating_temperature_f
        ambient_temp = baseline_input.ambient_temperature_f

        # Calculate heat loss increase percentage
        if expected_temp_f != ambient_temp:
            temp_rise_expected = expected_temp_f - ambient_temp
            temp_rise_actual = measured_temp_f - ambient_temp
            heat_loss_increase_pct = (
                (temp_rise_actual - temp_rise_expected) /
                abs(temp_rise_expected) * 100
            ) if temp_rise_expected != 0 else 0
        else:
            heat_loss_increase_pct = delta_t_f * 10  # Rough estimate

        # Classify anomaly type
        if heat_loss_increase_pct > self.ir_config.missing_insulation_threshold_pct:
            anomaly_type = "missing"
            probable_cause = "Missing or removed insulation"
            recommended_action = "Install new insulation to specification"
        elif heat_loss_increase_pct > self.ir_config.damaged_insulation_threshold_pct:
            anomaly_type = "damaged"
            probable_cause = "Damaged, compressed, or water-saturated insulation"
            recommended_action = "Remove and replace damaged insulation"
        else:
            anomaly_type = "hot_spot"
            probable_cause = "Thermal bridging, thin insulation, or joint gap"
            recommended_action = "Repair or add insulation at hot spot"

        # Classify severity
        if delta_t_f > 100 or heat_loss_increase_pct > 200:
            severity = "critical"
        elif delta_t_f > 50 or heat_loss_increase_pct > 100:
            severity = "high"
        elif delta_t_f > 25 or heat_loss_increase_pct > 50:
            severity = "medium"
        else:
            severity = "low"

        # Estimate excess heat loss
        excess_heat_loss = self._estimate_excess_heat_loss(
            baseline_input=baseline_input,
            measured_temp_f=measured_temp_f,
            expected_temp_f=expected_temp_f,
        )

        return AnomalyDetection(
            anomaly_id=str(uuid.uuid4())[:8],
            location=location,
            measured_temp_f=measured_temp_f,
            expected_temp_f=expected_temp_f,
            delta_t_f=delta_t_f,
            anomaly_type=anomaly_type,
            severity=severity,
            estimated_heat_loss_btu_hr=excess_heat_loss,
            probable_cause=probable_cause,
            recommended_action=recommended_action,
        )

    def _estimate_excess_heat_loss(
        self,
        baseline_input: InsulationInput,
        measured_temp_f: float,
        expected_temp_f: float,
    ) -> float:
        """
        Estimate excess heat loss from anomaly.

        Uses simplified calculation based on surface temperature
        difference and estimated affected area.

        Args:
            baseline_input: Baseline input
            measured_temp_f: Measured surface temperature
            expected_temp_f: Expected surface temperature

        Returns:
            Estimated excess heat loss (BTU/hr)
        """
        ambient_temp = baseline_input.ambient_temperature_f

        # Estimate affected area (assume 1 sqft for spot anomaly)
        affected_area_sqft = 1.0

        # Combined heat transfer coefficient (estimated)
        h_total = 2.0  # BTU/hr-sqft-F (typical for still air)

        # Heat loss at expected condition
        q_expected = h_total * affected_area_sqft * abs(expected_temp_f - ambient_temp)

        # Heat loss at measured condition
        q_measured = h_total * affected_area_sqft * abs(measured_temp_f - ambient_temp)

        # Excess heat loss
        excess = max(0, q_measured - q_expected)

        return excess

    def _calculate_annual_energy_cost(self, heat_loss_btu_hr: float) -> float:
        """Calculate annual energy cost from heat loss."""
        annual_mmbtu = (
            heat_loss_btu_hr *
            self.economic.operating_hours_per_year /
            1_000_000
        )
        return annual_mmbtu * self.economic.energy_cost_per_mmbtu

    def _estimate_repair_cost(self, hot_spots: List[IRHotSpot]) -> float:
        """Estimate total repair cost for identified issues."""
        total_cost = 0.0

        for hs in hot_spots:
            # Base cost per repair by severity
            if hs.severity == "critical":
                base_cost = 500.0
            elif hs.severity == "high":
                base_cost = 300.0
            elif hs.severity == "medium":
                base_cost = 150.0
            else:
                base_cost = 75.0

            total_cost += base_cost

        return total_cost

    def analyze_single_image(
        self,
        image: ThermalImageData,
        baseline_input: InsulationInput,
    ) -> List[IRHotSpot]:
        """
        Analyze a single thermal image.

        Args:
            image: Thermal image data
            baseline_input: Expected baseline

        Returns:
            List of identified hot spots
        """
        hot_spots = []

        # Calculate expected temperature
        expected_result = self.heat_loss_calc.calculate_heat_loss(baseline_input)
        expected_temp = expected_result.outer_surface_temperature_f

        # Analyze spots
        for spot_name, measured_temp in image.spot_temps.items():
            delta_t = measured_temp - expected_temp

            anomaly = self._classify_anomaly(
                location=f"{image.location}/{spot_name}",
                measured_temp_f=measured_temp,
                expected_temp_f=expected_temp,
                delta_t_f=delta_t,
                baseline_input=baseline_input,
            )

            if anomaly:
                hot_spot = IRHotSpot(
                    spot_id=anomaly.anomaly_id,
                    location_description=anomaly.location,
                    measured_temperature_f=anomaly.measured_temp_f,
                    expected_temperature_f=anomaly.expected_temp_f,
                    delta_t_f=anomaly.delta_t_f,
                    severity=anomaly.severity,
                    estimated_heat_loss_btu_hr=anomaly.estimated_heat_loss_btu_hr,
                    probable_cause=anomaly.probable_cause,
                    recommended_action=anomaly.recommended_action,
                    image_reference=image.image_id,
                )
                hot_spots.append(hot_spot)

        return hot_spots

    def generate_survey_report(
        self,
        survey_result: IRSurveyResult,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive survey report.

        Args:
            survey_result: Survey analysis result

        Returns:
            Report dictionary
        """
        report = {
            "survey_id": survey_result.survey_id,
            "survey_date": survey_result.survey_date.isoformat(),
            "conditions": {
                "ambient_temperature_f": survey_result.ambient_temperature_f,
                "wind_speed_mph": survey_result.wind_speed_mph,
                "humidity_pct": survey_result.humidity_pct,
            },
            "summary": {
                "items_surveyed": survey_result.items_surveyed,
                "total_anomalies": survey_result.total_anomalies,
                "critical_repairs": survey_result.critical_repairs_needed,
            },
            "heat_loss_impact": {
                "total_excess_btu_hr": survey_result.total_excess_heat_loss_btu_hr,
                "annual_cost_usd": survey_result.annual_excess_energy_cost_usd,
            },
            "repair_economics": {
                "estimated_repair_cost_usd": survey_result.estimated_repair_cost_usd,
                "estimated_annual_savings_usd": survey_result.estimated_annual_savings_usd,
                "simple_payback_years": (
                    survey_result.estimated_repair_cost_usd /
                    survey_result.estimated_annual_savings_usd
                ) if survey_result.estimated_annual_savings_usd > 0 else 999,
            },
            "hot_spots": [],
            "recommendations": [],
        }

        # Add hot spot details
        for hs in survey_result.hot_spots_identified:
            report["hot_spots"].append({
                "id": hs.spot_id,
                "location": hs.location_description,
                "measured_temp_f": hs.measured_temperature_f,
                "expected_temp_f": hs.expected_temperature_f,
                "delta_t_f": hs.delta_t_f,
                "severity": hs.severity,
                "probable_cause": hs.probable_cause,
                "recommended_action": hs.recommended_action,
            })

        # Sort hot spots by severity for prioritization
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        report["hot_spots"].sort(key=lambda x: severity_order.get(x["severity"], 4))

        # Generate recommendations
        if survey_result.critical_repairs_needed > 0:
            report["recommendations"].append(
                f"URGENT: Address {survey_result.critical_repairs_needed} critical repairs immediately"
            )

        if survey_result.estimated_annual_savings_usd > 1000:
            report["recommendations"].append(
                f"High ROI opportunity: ${survey_result.estimated_annual_savings_usd:.0f}/year savings available"
            )

        if survey_result.total_anomalies > 10:
            report["recommendations"].append(
                "Consider comprehensive insulation replacement program"
            )

        return report

    def calculate_roi_for_repairs(
        self,
        hot_spots: List[IRHotSpot],
    ) -> Dict[str, Any]:
        """
        Calculate ROI for repairing identified issues.

        Args:
            hot_spots: List of identified hot spots

        Returns:
            ROI analysis dictionary
        """
        total_repair_cost = self._estimate_repair_cost(hot_spots)
        total_excess_heat_loss = sum(hs.estimated_heat_loss_btu_hr for hs in hot_spots)
        annual_savings = self._calculate_annual_energy_cost(total_excess_heat_loss)

        if annual_savings > 0:
            simple_payback = total_repair_cost / annual_savings
        else:
            simple_payback = float('inf')

        # NPV calculation (10 year horizon)
        npv = -total_repair_cost
        discount_rate = self.economic.discount_rate_pct / 100
        for year in range(1, 11):
            npv += annual_savings / ((1 + discount_rate) ** year)

        # ROI
        if total_repair_cost > 0:
            roi = (annual_savings * 10 - total_repair_cost) / total_repair_cost * 100
        else:
            roi = 0

        return {
            "total_repair_cost_usd": round(total_repair_cost, 2),
            "annual_energy_savings_usd": round(annual_savings, 2),
            "simple_payback_years": round(simple_payback, 2) if simple_payback != float('inf') else 999,
            "npv_10_years_usd": round(npv, 2),
            "roi_10_years_pct": round(roi, 1),
        }

    @property
    def survey_count(self) -> int:
        """Get total surveys performed."""
        return self._survey_count
