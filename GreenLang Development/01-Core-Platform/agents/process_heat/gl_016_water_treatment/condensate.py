"""
GL-016 WATERGUARD Agent - Condensate Return Quality Module

Implements condensate quality monitoring including:
- Corrosion product tracking (iron, copper)
- Contamination detection
- Amine treatment effectiveness
- Return quality assessment

All calculations are deterministic with zero hallucination.

References:
    - ASME Guidelines for Condensate Quality
    - EPRI Recommendations for Condensate Polishing
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    CondensateInput,
    CondensateOutput,
    CondensateLimits,
    WaterQualityResult,
    WaterQualityStatus,
    ChemicalType,
    CorrosionMechanism,
)
from greenlang.agents.process_heat.gl_016_water_treatment.config import (
    get_amine_config,
    AmineConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class CondensateConstants:
    """Constants for condensate quality calculations."""

    # Iron limits (ppb)
    IRON_EXCELLENT = 20
    IRON_GOOD = 50
    IRON_WARNING = 100
    IRON_ACTION = 200

    # Copper limits (ppb)
    COPPER_EXCELLENT = 5
    COPPER_GOOD = 10
    COPPER_WARNING = 20
    COPPER_ACTION = 50

    # pH targets
    PH_TARGET_MIN = 8.5
    PH_TARGET_MAX = 9.0

    # Contamination thresholds
    HARDNESS_MAX_PPM = 0.5
    OIL_MAX_PPM = 1.0
    CATION_CONDUCTIVITY_MAX = 1.0

    # Corrosion rate estimation factors (mpy from ppb iron)
    # Based on empirical correlations
    IRON_TO_CORROSION_RATE = 0.02  # mpy per ppb Fe


# =============================================================================
# CONDENSATE ANALYZER CLASS
# =============================================================================

class CondensateAnalyzer:
    """
    Analyzes condensate return quality.

    Monitors corrosion product transport, contamination, and amine
    treatment effectiveness for condensate return systems.

    Attributes:
        amine_type: Amine treatment chemical type
        amine_config: Amine treatment configuration
        limits: Condensate quality limits

    Example:
        >>> analyzer = CondensateAnalyzer(amine_type=ChemicalType.MORPHOLINE)
        >>> result = analyzer.analyze(condensate_input)
    """

    def __init__(
        self,
        amine_type: Optional[ChemicalType] = ChemicalType.MORPHOLINE,
        custom_limits: Optional[CondensateLimits] = None,
    ) -> None:
        """
        Initialize CondensateAnalyzer.

        Args:
            amine_type: Amine treatment chemical type
            custom_limits: Optional custom condensate limits
        """
        self.amine_type = amine_type
        self.amine_config = get_amine_config(amine_type) if amine_type else None

        # Use custom limits or defaults
        self.limits = custom_limits or CondensateLimits(
            ph_min=CondensateConstants.PH_TARGET_MIN,
            ph_max=CondensateConstants.PH_TARGET_MAX,
            iron_max_ppb=CondensateConstants.IRON_GOOD,
            iron_action_level_ppb=CondensateConstants.IRON_ACTION,
            copper_max_ppb=CondensateConstants.COPPER_GOOD,
            copper_action_level_ppb=CondensateConstants.COPPER_ACTION,
            hardness_max_ppm=CondensateConstants.HARDNESS_MAX_PPM,
            oil_max_ppm=CondensateConstants.OIL_MAX_PPM,
            cation_conductivity_max_umho=CondensateConstants.CATION_CONDUCTIVITY_MAX,
        )

        logger.info(f"CondensateAnalyzer initialized with amine: {amine_type}")

    def analyze(self, input_data: CondensateInput) -> CondensateOutput:
        """
        Analyze condensate quality.

        Args:
            input_data: Condensate sample data

        Returns:
            CondensateOutput with comprehensive analysis
        """
        start_time = datetime.now(timezone.utc)
        logger.debug(f"Analyzing condensate sample: {input_data.sample_id}")

        parameter_results: List[WaterQualityResult] = []
        recommendations: List[str] = []

        # Analyze pH
        ph_result = self._analyze_ph(input_data)
        parameter_results.append(ph_result)

        # Analyze iron (primary corrosion indicator)
        iron_result = self._analyze_iron(input_data)
        parameter_results.append(iron_result)

        # Analyze copper
        if input_data.copper_ppb is not None:
            copper_result = self._analyze_copper(input_data)
            parameter_results.append(copper_result)

        # Analyze conductivity
        conductivity_result = self._analyze_conductivity(input_data)
        parameter_results.append(conductivity_result)

        # Analyze dissolved oxygen
        if input_data.dissolved_oxygen_ppb is not None:
            do_result = self._analyze_dissolved_oxygen(input_data)
            parameter_results.append(do_result)

        # Analyze amine residual
        amine_adequate = None
        amine_adjustment = None
        if input_data.amine_residual_ppm is not None:
            amine_result = self._analyze_amine_residual(input_data)
            parameter_results.append(amine_result)
            amine_adequate, amine_adjustment = self._evaluate_amine_program(input_data)

        # Contamination detection
        contamination_detected, contamination_source = self._detect_contamination(
            input_data
        )

        # Analyze contamination parameters
        if input_data.hardness_ppm is not None:
            hardness_result = self._analyze_hardness(input_data)
            parameter_results.append(hardness_result)

        if input_data.oil_ppm is not None:
            oil_result = self._analyze_oil(input_data)
            parameter_results.append(oil_result)

        # Calculate corrosion metrics
        corrosion_rate, corrosion_mechanism = self._estimate_corrosion(input_data)

        # Determine return quality acceptability
        return_quality_acceptable = self._evaluate_return_quality(
            parameter_results, contamination_detected
        )

        # Determine overall status
        overall_status = self._determine_overall_status(parameter_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            input_data, parameter_results, corrosion_mechanism,
            contamination_detected, amine_adequate
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(input_data)
        processing_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        return CondensateOutput(
            sample_id=input_data.sample_id,
            timestamp=datetime.now(timezone.utc),
            overall_status=overall_status,
            return_quality_acceptable=return_quality_acceptable,
            parameter_results=parameter_results,
            corrosion_rate_mpy=corrosion_rate,
            corrosion_mechanism=corrosion_mechanism,
            contamination_detected=contamination_detected,
            contamination_source=contamination_source,
            amine_dose_adequate=amine_adequate,
            amine_adjustment_ppm=amine_adjustment,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
        )

    def _analyze_ph(self, input_data: CondensateInput) -> WaterQualityResult:
        """Analyze condensate pH."""
        value = input_data.ph
        min_limit = self.limits.ph_min
        max_limit = self.limits.ph_max
        target = (min_limit + max_limit) / 2

        # For condensate, low pH is particularly concerning (carbonic acid)
        if value < 7.0:
            status = WaterQualityStatus.CRITICAL
        elif value < min_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value > max_limit + 0.5:
            status = WaterQualityStatus.WARNING
        elif value > max_limit:
            status = WaterQualityStatus.ACCEPTABLE
        elif abs(value - target) <= 0.2:
            status = WaterQualityStatus.EXCELLENT
        else:
            status = WaterQualityStatus.GOOD

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

    def _analyze_iron(self, input_data: CondensateInput) -> WaterQualityResult:
        """
        Analyze iron concentration (primary corrosion indicator).

        Iron in condensate indicates active corrosion in the return system.
        """
        value = input_data.iron_ppb
        max_limit = self.limits.iron_max_ppb
        action_level = self.limits.iron_action_level_ppb
        target = CondensateConstants.IRON_EXCELLENT

        if value > action_level:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value <= target:
            status = WaterQualityStatus.EXCELLENT
        elif value <= CondensateConstants.IRON_GOOD:
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

    def _analyze_copper(self, input_data: CondensateInput) -> WaterQualityResult:
        """Analyze copper concentration."""
        value = input_data.copper_ppb or 0
        max_limit = self.limits.copper_max_ppb
        action_level = self.limits.copper_action_level_ppb
        target = CondensateConstants.COPPER_EXCELLENT

        if value > action_level:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value <= target:
            status = WaterQualityStatus.EXCELLENT
        elif value <= CondensateConstants.COPPER_GOOD:
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

    def _analyze_conductivity(self, input_data: CondensateInput) -> WaterQualityResult:
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

    def _analyze_dissolved_oxygen(
        self, input_data: CondensateInput
    ) -> WaterQualityResult:
        """Analyze dissolved oxygen in condensate."""
        value = input_data.dissolved_oxygen_ppb or 0
        max_limit = 20.0  # ppb
        target = 10.0

        if value > 50:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.WARNING
        elif value <= target:
            status = WaterQualityStatus.EXCELLENT
        else:
            status = WaterQualityStatus.GOOD

        return WaterQualityResult(
            parameter="Dissolved Oxygen",
            value=value,
            unit="ppb",
            min_limit=None,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=None,
        )

    def _analyze_amine_residual(
        self, input_data: CondensateInput
    ) -> WaterQualityResult:
        """Analyze amine treatment residual."""
        value = input_data.amine_residual_ppm or 0

        if self.amine_config:
            target = self.amine_config.typical_dose_ppm
        else:
            target = 5.0  # Default for morpholine

        min_limit = target * 0.5
        max_limit = target * 2.0

        if value < min_limit * 0.5:
            status = WaterQualityStatus.CRITICAL
        elif value < min_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value > max_limit:
            status = WaterQualityStatus.WARNING
        elif abs(value - target) / target <= 0.2:
            status = WaterQualityStatus.EXCELLENT
        else:
            status = WaterQualityStatus.GOOD

        deviation_pct = ((value - target) / target) * 100 if target != 0 else 0

        return WaterQualityResult(
            parameter="Amine Residual",
            value=value,
            unit="ppm",
            min_limit=min_limit,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=round(deviation_pct, 2),
        )

    def _analyze_hardness(self, input_data: CondensateInput) -> WaterQualityResult:
        """Analyze hardness (contamination indicator)."""
        value = input_data.hardness_ppm or 0
        max_limit = self.limits.hardness_max_ppm
        target = 0.0

        if value > max_limit * 3:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value <= max_limit * 0.3:
            status = WaterQualityStatus.EXCELLENT
        elif value <= max_limit * 0.6:
            status = WaterQualityStatus.GOOD
        else:
            status = WaterQualityStatus.WARNING

        return WaterQualityResult(
            parameter="Hardness",
            value=value,
            unit="ppm CaCO3",
            min_limit=None,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=None,
        )

    def _analyze_oil(self, input_data: CondensateInput) -> WaterQualityResult:
        """Analyze oil contamination."""
        value = input_data.oil_ppm or 0
        max_limit = self.limits.oil_max_ppm
        target = 0.0

        if value > max_limit * 2:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value <= max_limit * 0.3:
            status = WaterQualityStatus.EXCELLENT
        elif value <= max_limit * 0.6:
            status = WaterQualityStatus.GOOD
        else:
            status = WaterQualityStatus.WARNING

        return WaterQualityResult(
            parameter="Oil Contamination",
            value=value,
            unit="ppm",
            min_limit=None,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=None,
        )

    def _evaluate_amine_program(
        self, input_data: CondensateInput
    ) -> Tuple[bool, Optional[float]]:
        """
        Evaluate amine treatment effectiveness.

        Returns:
            Tuple of (is_adequate, adjustment_ppm)
        """
        if not self.amine_config:
            return True, None

        residual = input_data.amine_residual_ppm or 0
        target_residual = self.amine_config.typical_dose_ppm
        target_ph_min, target_ph_max = self.amine_config.target_ph_range

        # Check pH is in target range
        ph_adequate = target_ph_min <= input_data.ph <= target_ph_max

        # Check residual is sufficient
        residual_adequate = residual >= target_residual * 0.5

        is_adequate = ph_adequate and residual_adequate

        # Calculate adjustment
        adjustment = None
        if not is_adequate:
            if input_data.ph < target_ph_min:
                # Need more amine
                adjustment = target_residual - residual
            elif input_data.ph > target_ph_max and residual > target_residual * 1.5:
                # Too much amine
                adjustment = target_residual - residual  # Will be negative

        return is_adequate, round(adjustment, 2) if adjustment else None

    def _detect_contamination(
        self, input_data: CondensateInput
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect contamination in condensate.

        Returns:
            Tuple of (contamination_detected, suspected_source)
        """
        contamination_detected = False
        source = None

        # Check hardness (cooling water or process leak)
        if input_data.hardness_ppm and input_data.hardness_ppm > self.limits.hardness_max_ppm:
            contamination_detected = True
            source = "Cooling water or process leak (hardness detected)"

        # Check oil
        if input_data.oil_ppm and input_data.oil_ppm > self.limits.oil_max_ppm:
            contamination_detected = True
            if source:
                source += "; Oil contamination (process or equipment leak)"
            else:
                source = "Oil contamination from process or equipment"

        # Check TDS spike
        if input_data.tds_ppm and input_data.tds_ppm > 50:
            contamination_detected = True
            if source:
                source += "; Elevated TDS"
            else:
                source = "Contamination source unknown (elevated TDS)"

        # Check high conductivity
        cond = input_data.cation_conductivity_umho or input_data.specific_conductivity_umho
        if cond > self.limits.cation_conductivity_max_umho * 3:
            contamination_detected = True
            if not source:
                source = "Contamination suspected (high conductivity)"

        return contamination_detected, source

    def _estimate_corrosion(
        self, input_data: CondensateInput
    ) -> Tuple[Optional[float], Optional[CorrosionMechanism]]:
        """
        Estimate corrosion rate and mechanism from condensate analysis.

        Returns:
            Tuple of (estimated_corrosion_rate_mpy, corrosion_mechanism)
        """
        iron_ppb = input_data.iron_ppb
        ph = input_data.ph
        do = input_data.dissolved_oxygen_ppb or 0

        # Estimate corrosion rate from iron transport
        # Empirical correlation: mpy ~ iron_ppb * factor / return_rate
        return_factor = input_data.condensate_return_pct / 100
        if return_factor > 0:
            corrosion_rate = (
                iron_ppb * CondensateConstants.IRON_TO_CORROSION_RATE / return_factor
            )
        else:
            corrosion_rate = iron_ppb * CondensateConstants.IRON_TO_CORROSION_RATE

        # Determine likely corrosion mechanism
        mechanism = None

        if ph < 7.5 and iron_ppb > 100:
            mechanism = CorrosionMechanism.CARBONIC_ACID
        elif do > 20 and iron_ppb > 100:
            mechanism = CorrosionMechanism.OXYGEN_PITTING
        elif iron_ppb > 200:
            mechanism = CorrosionMechanism.FLOW_ACCELERATED

        return round(corrosion_rate, 2), mechanism

    def _evaluate_return_quality(
        self,
        results: List[WaterQualityResult],
        contamination_detected: bool,
    ) -> bool:
        """Determine if condensate is acceptable for return."""
        # Reject if contaminated
        if contamination_detected:
            return False

        # Check critical parameters
        for result in results:
            if result.status == WaterQualityStatus.CRITICAL:
                return False
            if "iron" in result.parameter.lower() and result.status == WaterQualityStatus.OUT_OF_SPEC:
                if result.value > self.limits.iron_action_level_ppb:
                    return False

        return True

    def _determine_overall_status(
        self, results: List[WaterQualityResult]
    ) -> WaterQualityStatus:
        """Determine overall condensate quality status."""
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
        input_data: CondensateInput,
        results: List[WaterQualityResult],
        corrosion_mechanism: Optional[CorrosionMechanism],
        contamination_detected: bool,
        amine_adequate: Optional[bool],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Contamination recommendations
        if contamination_detected:
            recommendations.append(
                "URGENT: Contamination detected - investigate source and "
                "consider diverting condensate until resolved"
            )

        # Corrosion mechanism recommendations
        if corrosion_mechanism:
            if corrosion_mechanism == CorrosionMechanism.CARBONIC_ACID:
                recommendations.append(
                    "Low pH indicates carbonic acid corrosion - increase amine dosing"
                )
            elif corrosion_mechanism == CorrosionMechanism.OXYGEN_PITTING:
                recommendations.append(
                    "Oxygen ingress detected - check vacuum breakers and "
                    "condensate receiver venting"
                )
            elif corrosion_mechanism == CorrosionMechanism.FLOW_ACCELERATED:
                recommendations.append(
                    "High iron may indicate FAC - inspect elbows and "
                    "velocity-limited areas"
                )

        # Amine program recommendations
        if amine_adequate is False:
            if input_data.ph < self.limits.ph_min:
                recommendations.append(
                    "Increase amine dosing to achieve target pH range"
                )
            elif (input_data.amine_residual_ppm or 0) < (
                self.amine_config.typical_dose_ppm * 0.5 if self.amine_config else 2.5
            ):
                recommendations.append(
                    "Amine residual low - verify feed pump operation and dosing rate"
                )

        # Parameter-specific recommendations
        for result in results:
            if result.status in [WaterQualityStatus.CRITICAL, WaterQualityStatus.OUT_OF_SPEC]:
                if "iron" in result.parameter.lower():
                    if result.value > self.limits.iron_action_level_ppb:
                        recommendations.append(
                            "ACTION LEVEL: Iron exceeds limit - inspect return lines, "
                            "consider condensate polishing"
                        )
                elif "copper" in result.parameter.lower():
                    recommendations.append(
                        "Elevated copper indicates heater tube corrosion - "
                        "check ammonia levels"
                    )

        return recommendations

    def _calculate_provenance_hash(self, input_data: CondensateInput) -> str:
        """Calculate SHA-256 hash for audit trail."""
        import json
        data_str = json.dumps(input_data.dict(), sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()


def calculate_amine_requirement(
    steam_flow_lb_hr: float,
    co2_in_steam_ppm: float,
    amine_type: ChemicalType,
    target_ph: float = 8.8,
) -> float:
    """
    Calculate amine dosing requirement for condensate pH control.

    Args:
        steam_flow_lb_hr: Steam flow rate (lb/hr)
        co2_in_steam_ppm: CO2 concentration in steam (ppm)
        amine_type: Type of neutralizing amine
        target_ph: Target condensate pH

    Returns:
        Required amine dose (ppm in steam)
    """
    config = get_amine_config(amine_type)
    if not config:
        # Default neutralizing capacity
        neutralizing_capacity = 0.5
    else:
        neutralizing_capacity = config.neutralizing_capacity

    # Calculate amine required to neutralize CO2
    # Simplified: amine_ppm = co2_ppm / neutralizing_capacity
    amine_required = co2_in_steam_ppm / neutralizing_capacity

    # Add safety factor for distribution ratio effects
    safety_factor = 1.2
    amine_required *= safety_factor

    return round(amine_required, 2)
