"""
GL-016 WATERGUARD Agent - Feedwater Quality Module

Implements feedwater quality monitoring per ASME guidelines including:
- Dissolved oxygen control
- Hardness monitoring
- Corrosion product transport (iron, copper)
- pH control
- Conductivity monitoring

All calculations are deterministic with zero hallucination.

References:
    - ASME Consensus on Operating Practices for Control of Feedwater/Boiler Water
    - EPRI Guidelines for Feedwater Quality
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    FeedwaterInput,
    FeedwaterOutput,
    FeedwaterLimits,
    WaterQualityResult,
    WaterQualityStatus,
    BoilerPressureClass,
    ChemicalType,
)
from greenlang.agents.process_heat.gl_016_water_treatment.config import (
    get_feedwater_limits,
    get_scavenger_config,
    determine_pressure_class,
    OxygenScavengerConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class FeedwaterConstants:
    """Constants for feedwater quality calculations."""

    # Dissolved oxygen limits by pressure class (ppb)
    DO_LIMITS = {
        "low_pressure": 7,
        "medium_pressure": 7,
        "high_pressure": 5,
        "supercritical": 3,
    }

    # Iron limits by pressure class (ppb)
    IRON_LIMITS = {
        "low_pressure": 100,
        "medium_pressure": 20,
        "high_pressure": 10,
        "supercritical": 5,
    }

    # Copper limits by pressure class (ppb)
    COPPER_LIMITS = {
        "low_pressure": 50,
        "medium_pressure": 15,
        "high_pressure": 5,
        "supercritical": 2,
    }

    # Hardness limits (ppm as CaCO3)
    HARDNESS_MAX = 0.3

    # Oxygen scavenger stoichiometry (ppm scavenger per ppb O2)
    SULFITE_RATIO = 0.008  # ~8 ppm sulfite per ppm O2
    HYDRAZINE_RATIO = 0.001


# =============================================================================
# FEEDWATER ANALYZER CLASS
# =============================================================================

class FeedwaterAnalyzer:
    """
    Analyzes feedwater quality per ASME guidelines.

    Feedwater quality is critical for boiler protection and efficiency.
    This analyzer monitors dissolved oxygen, hardness, corrosion products,
    and oxygen scavenger performance.

    Attributes:
        pressure_class: Boiler pressure classification
        limits: ASME feedwater limits
        scavenger_config: Oxygen scavenger configuration

    Example:
        >>> analyzer = FeedwaterAnalyzer(
        ...     pressure_class=BoilerPressureClass.MEDIUM_PRESSURE,
        ...     scavenger_type=ChemicalType.SULFITE
        ... )
        >>> result = analyzer.analyze(feedwater_input)
    """

    def __init__(
        self,
        pressure_class: BoilerPressureClass = BoilerPressureClass.MEDIUM_PRESSURE,
        scavenger_type: ChemicalType = ChemicalType.SULFITE,
    ) -> None:
        """
        Initialize FeedwaterAnalyzer.

        Args:
            pressure_class: Boiler pressure classification
            scavenger_type: Oxygen scavenger chemical type
        """
        self.pressure_class = pressure_class
        self.scavenger_type = scavenger_type

        # Load ASME limits
        self.limits = get_feedwater_limits(pressure_class)

        # Load scavenger configuration
        self.scavenger_config = get_scavenger_config(scavenger_type)

        logger.info(
            f"FeedwaterAnalyzer initialized: {pressure_class.value}, "
            f"scavenger: {scavenger_type.value}"
        )

    def analyze(self, input_data: FeedwaterInput) -> FeedwaterOutput:
        """
        Analyze feedwater quality.

        Args:
            input_data: Feedwater sample data

        Returns:
            FeedwaterOutput with comprehensive analysis
        """
        start_time = datetime.now(timezone.utc)
        logger.debug(f"Analyzing feedwater sample: {input_data.sample_id}")

        parameter_results: List[WaterQualityResult] = []
        recommendations: List[str] = []

        # Analyze pH
        ph_result = self._analyze_ph(input_data)
        parameter_results.append(ph_result)

        # Analyze dissolved oxygen
        do_result = self._analyze_dissolved_oxygen(input_data)
        parameter_results.append(do_result)

        # Analyze hardness
        if input_data.total_hardness_ppm is not None:
            hardness_result = self._analyze_hardness(input_data)
            parameter_results.append(hardness_result)

        # Analyze iron
        if input_data.iron_ppb is not None:
            iron_result = self._analyze_iron(input_data)
            parameter_results.append(iron_result)

        # Analyze copper
        if input_data.copper_ppb is not None:
            copper_result = self._analyze_copper(input_data)
            parameter_results.append(copper_result)

        # Analyze silica
        if input_data.silica_ppm is not None:
            silica_result = self._analyze_silica(input_data)
            parameter_results.append(silica_result)

        # Analyze conductivity
        conductivity_result = self._analyze_conductivity(input_data)
        parameter_results.append(conductivity_result)

        # Analyze oxygen scavenger
        oxygen_control_adequate = True
        scavenger_adjustment = None
        if input_data.oxygen_scavenger_residual_ppm is not None:
            scavenger_result = self._analyze_scavenger_residual(input_data)
            parameter_results.append(scavenger_result)
            oxygen_control_adequate, scavenger_adjustment = self._evaluate_oxygen_control(
                input_data
            )

        # Evaluate corrosion product transport
        iron_concern = self._evaluate_iron_transport(input_data)
        copper_concern = self._evaluate_copper_transport(input_data)

        # Determine overall status
        overall_status = self._determine_overall_status(parameter_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            input_data, parameter_results, oxygen_control_adequate,
            iron_concern, copper_concern
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(input_data)
        processing_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        return FeedwaterOutput(
            sample_id=input_data.sample_id,
            timestamp=datetime.now(timezone.utc),
            overall_status=overall_status,
            parameter_results=parameter_results,
            oxygen_control_adequate=oxygen_control_adequate,
            oxygen_scavenger_dose_adjustment=scavenger_adjustment,
            iron_transport_concern=iron_concern,
            copper_transport_concern=copper_concern,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
        )

    def _analyze_ph(self, input_data: FeedwaterInput) -> WaterQualityResult:
        """Analyze feedwater pH."""
        value = input_data.ph
        min_limit = self.limits.ph_min
        max_limit = self.limits.ph_max
        target = (min_limit + max_limit) / 2

        if value < min_limit - 0.5 or value > max_limit + 0.5:
            status = WaterQualityStatus.CRITICAL
        elif value < min_limit or value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif abs(value - target) <= 0.2:
            status = WaterQualityStatus.EXCELLENT
        elif abs(value - target) <= 0.4:
            status = WaterQualityStatus.GOOD
        else:
            status = WaterQualityStatus.ACCEPTABLE

        deviation_pct = ((value - target) / target) * 100 if target != 0 else 0

        return WaterQualityResult(
            parameter="pH",
            value=value,
            unit="pH units",
            min_limit=min_limit,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=round(deviation_pct, 2),
        )

    def _analyze_dissolved_oxygen(
        self, input_data: FeedwaterInput
    ) -> WaterQualityResult:
        """
        Analyze dissolved oxygen.

        Dissolved oxygen is the primary cause of corrosion in feedwater
        and boiler systems. ASME limits are typically 7 ppb or less.
        """
        value = input_data.dissolved_oxygen_ppb
        max_limit = self.limits.dissolved_oxygen_max_ppb
        target = max_limit * 0.5  # Target 50% of limit

        if value > max_limit * 2:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value <= target:
            status = WaterQualityStatus.EXCELLENT
        elif value <= max_limit * 0.75:
            status = WaterQualityStatus.GOOD
        else:
            status = WaterQualityStatus.WARNING

        deviation_pct = ((value - target) / target) * 100 if target != 0 else 0

        return WaterQualityResult(
            parameter="Dissolved Oxygen",
            value=value,
            unit="ppb",
            min_limit=None,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=round(deviation_pct, 2),
        )

    def _analyze_hardness(self, input_data: FeedwaterInput) -> WaterQualityResult:
        """
        Analyze total hardness.

        Hardness causes scale formation and must be minimized.
        """
        value = input_data.total_hardness_ppm or 0
        max_limit = self.limits.total_hardness_max_ppm
        target = 0.0  # Zero hardness is ideal

        if value > max_limit * 3:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value <= max_limit * 0.3:
            status = WaterQualityStatus.EXCELLENT
        elif value <= max_limit * 0.6:
            status = WaterQualityStatus.GOOD
        else:
            status = WaterQualityStatus.ACCEPTABLE

        return WaterQualityResult(
            parameter="Total Hardness",
            value=value,
            unit="ppm CaCO3",
            min_limit=None,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=None,
        )

    def _analyze_iron(self, input_data: FeedwaterInput) -> WaterQualityResult:
        """
        Analyze total iron (corrosion product transport).

        Elevated iron indicates upstream corrosion and contributes to
        boiler deposition.
        """
        value = input_data.iron_ppb or 0
        max_limit = self.limits.iron_max_ppb
        target = max_limit * 0.5

        if value > max_limit * 3:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value <= target:
            status = WaterQualityStatus.EXCELLENT
        elif value <= max_limit * 0.75:
            status = WaterQualityStatus.GOOD
        else:
            status = WaterQualityStatus.WARNING

        deviation_pct = ((value - target) / target) * 100 if target != 0 else 0

        return WaterQualityResult(
            parameter="Total Iron",
            value=value,
            unit="ppb",
            min_limit=None,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=round(deviation_pct, 2),
        )

    def _analyze_copper(self, input_data: FeedwaterInput) -> WaterQualityResult:
        """
        Analyze total copper (corrosion product transport).

        Copper transport from feedwater heaters leads to boiler deposition.
        """
        value = input_data.copper_ppb or 0
        max_limit = self.limits.copper_max_ppb
        target = max_limit * 0.5

        if value > max_limit * 3:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value <= target:
            status = WaterQualityStatus.EXCELLENT
        elif value <= max_limit * 0.75:
            status = WaterQualityStatus.GOOD
        else:
            status = WaterQualityStatus.WARNING

        deviation_pct = ((value - target) / target) * 100 if target != 0 else 0

        return WaterQualityResult(
            parameter="Total Copper",
            value=value,
            unit="ppb",
            min_limit=None,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=round(deviation_pct, 2),
        )

    def _analyze_silica(self, input_data: FeedwaterInput) -> WaterQualityResult:
        """Analyze silica concentration."""
        value = input_data.silica_ppm or 0
        max_limit = self.limits.silica_max_ppm
        target = max_limit * 0.5

        if value > max_limit * 2:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value <= target:
            status = WaterQualityStatus.EXCELLENT
        elif value <= max_limit * 0.75:
            status = WaterQualityStatus.GOOD
        else:
            status = WaterQualityStatus.ACCEPTABLE

        return WaterQualityResult(
            parameter="Silica (SiO2)",
            value=value,
            unit="ppm",
            min_limit=None,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=None,
        )

    def _analyze_conductivity(self, input_data: FeedwaterInput) -> WaterQualityResult:
        """Analyze cation conductivity."""
        value = input_data.cation_conductivity_umho or input_data.specific_conductivity_umho
        max_limit = self.limits.cation_conductivity_max_umho
        target = max_limit * 0.5

        if value > max_limit * 2:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value <= target:
            status = WaterQualityStatus.EXCELLENT
        elif value <= max_limit * 0.75:
            status = WaterQualityStatus.GOOD
        else:
            status = WaterQualityStatus.ACCEPTABLE

        param_name = "Cation Conductivity" if input_data.cation_conductivity_umho else "Specific Conductivity"

        return WaterQualityResult(
            parameter=param_name,
            value=value,
            unit="umho/cm",
            min_limit=None,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=None,
        )

    def _analyze_scavenger_residual(
        self, input_data: FeedwaterInput
    ) -> WaterQualityResult:
        """Analyze oxygen scavenger residual."""
        value = input_data.oxygen_scavenger_residual_ppm or 0

        if self.scavenger_config:
            min_limit = self.scavenger_config.residual_min_ppm
            max_limit = self.scavenger_config.residual_max_ppm
            target = self.scavenger_config.residual_target_ppm
        else:
            # Default sulfite residual targets
            min_limit = 20.0
            max_limit = 40.0
            target = 30.0

        if value < min_limit * 0.5:
            status = WaterQualityStatus.CRITICAL
        elif value < min_limit or value > max_limit * 1.5:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif abs(value - target) / target <= 0.1:
            status = WaterQualityStatus.EXCELLENT
        elif abs(value - target) / target <= 0.25:
            status = WaterQualityStatus.GOOD
        else:
            status = WaterQualityStatus.ACCEPTABLE

        deviation_pct = ((value - target) / target) * 100 if target != 0 else 0

        return WaterQualityResult(
            parameter="Oxygen Scavenger Residual",
            value=value,
            unit="ppm",
            min_limit=min_limit,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=round(deviation_pct, 2),
        )

    def _evaluate_oxygen_control(
        self, input_data: FeedwaterInput
    ) -> Tuple[bool, Optional[float]]:
        """
        Evaluate oxygen control adequacy and calculate dose adjustment.

        Returns:
            Tuple of (is_adequate, adjustment_percentage)
        """
        do_ppb = input_data.dissolved_oxygen_ppb
        residual_ppm = input_data.oxygen_scavenger_residual_ppm or 0

        if self.scavenger_config:
            min_residual = self.scavenger_config.residual_min_ppm
            target_residual = self.scavenger_config.residual_target_ppm
        else:
            min_residual = 20.0
            target_residual = 30.0

        # Check if oxygen is controlled
        do_limit = self.limits.dissolved_oxygen_max_ppb
        oxygen_controlled = do_ppb <= do_limit

        # Check residual adequacy
        residual_adequate = residual_ppm >= min_residual

        is_adequate = oxygen_controlled and residual_adequate

        # Calculate dose adjustment
        adjustment = None
        if not is_adequate:
            if residual_ppm < min_residual:
                # Need more scavenger
                adjustment = ((target_residual - residual_ppm) / target_residual) * 100
            elif residual_ppm > self.scavenger_config.residual_max_ppm if self.scavenger_config else 40:
                # Too much scavenger
                adjustment = ((target_residual - residual_ppm) / target_residual) * 100

        return is_adequate, round(adjustment, 1) if adjustment else None

    def _evaluate_iron_transport(self, input_data: FeedwaterInput) -> bool:
        """Evaluate if iron transport is a concern."""
        if input_data.iron_ppb is None:
            return False
        return input_data.iron_ppb > self.limits.iron_max_ppb

    def _evaluate_copper_transport(self, input_data: FeedwaterInput) -> bool:
        """Evaluate if copper transport is a concern."""
        if input_data.copper_ppb is None:
            return False
        return input_data.copper_ppb > self.limits.copper_max_ppb

    def _determine_overall_status(
        self, results: List[WaterQualityResult]
    ) -> WaterQualityStatus:
        """Determine overall feedwater quality status."""
        if not results:
            return WaterQualityStatus.WARNING

        status_priority = {
            WaterQualityStatus.CRITICAL: 0,
            WaterQualityStatus.OUT_OF_SPEC: 1,
            WaterQualityStatus.WARNING: 2,
            WaterQualityStatus.ACCEPTABLE: 3,
            WaterQualityStatus.GOOD: 4,
            WaterQualityStatus.EXCELLENT: 5,
        }

        worst_status = WaterQualityStatus.EXCELLENT
        for result in results:
            if status_priority[result.status] < status_priority[worst_status]:
                worst_status = result.status

        return worst_status

    def _generate_recommendations(
        self,
        input_data: FeedwaterInput,
        results: List[WaterQualityResult],
        oxygen_control_adequate: bool,
        iron_concern: bool,
        copper_concern: bool,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Oxygen control recommendations
        if not oxygen_control_adequate:
            if input_data.dissolved_oxygen_ppb > self.limits.dissolved_oxygen_max_ppb:
                recommendations.append(
                    "Verify deaerator operation - dissolved oxygen exceeds limit"
                )
            if (input_data.oxygen_scavenger_residual_ppm or 0) < (
                self.scavenger_config.residual_min_ppm if self.scavenger_config else 20
            ):
                recommendations.append(
                    "Increase oxygen scavenger dosing rate"
                )

        # Corrosion product recommendations
        if iron_concern:
            recommendations.append(
                "Investigate iron source - check condensate system, "
                "heaters, and piping for corrosion"
            )

        if copper_concern:
            recommendations.append(
                "Elevated copper indicates heater tube corrosion - "
                "verify ammonia levels and pH"
            )

        # Parameter-specific recommendations
        for result in results:
            if result.status in [WaterQualityStatus.CRITICAL, WaterQualityStatus.OUT_OF_SPEC]:
                if "hardness" in result.parameter.lower():
                    recommendations.append(
                        "Hardness breakthrough - check softener/demin regeneration"
                    )
                elif "silica" in result.parameter.lower():
                    recommendations.append(
                        "Silica exceeds limit - verify makeup treatment system"
                    )
                elif "ph" in result.parameter.lower():
                    if result.value < (result.min_limit or 8.5):
                        recommendations.append(
                            "Increase amine dosing to raise feedwater pH"
                        )
                    else:
                        recommendations.append(
                            "Reduce amine dosing - pH too high"
                        )

        return recommendations

    def _calculate_provenance_hash(self, input_data: FeedwaterInput) -> str:
        """Calculate SHA-256 hash for audit trail."""
        import json
        data_str = json.dumps(input_data.dict(), sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()


def calculate_scavenger_requirement(
    dissolved_oxygen_ppb: float,
    scavenger_type: ChemicalType,
    excess_factor: float = 1.5,
) -> float:
    """
    Calculate oxygen scavenger dosing requirement.

    Args:
        dissolved_oxygen_ppb: Dissolved oxygen concentration (ppb)
        scavenger_type: Type of oxygen scavenger
        excess_factor: Excess dosing factor (default 1.5 = 50% excess)

    Returns:
        Required scavenger dose in ppm
    """
    config = get_scavenger_config(scavenger_type)
    if not config:
        # Default to sulfite
        stoich_ratio = 7.9
    else:
        stoich_ratio = config.stoichiometric_ratio

    # Convert O2 from ppb to ppm
    o2_ppm = dissolved_oxygen_ppb / 1000

    # Calculate required dose
    # Dose (ppm) = O2 (ppm) * stoich_ratio * excess_factor
    required_dose = o2_ppm * stoich_ratio * excess_factor

    return round(required_dose, 2)
