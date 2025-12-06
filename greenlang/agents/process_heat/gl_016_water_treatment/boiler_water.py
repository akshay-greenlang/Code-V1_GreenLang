"""
GL-016 WATERGUARD Agent - Boiler Water Chemistry Module

Implements boiler water chemistry analysis per ASME/ABMA guidelines including:
- pH and alkalinity control
- Phosphate program management (coordinated, congruent, precipitate)
- Conductivity and TDS monitoring
- Silica control for carryover prevention
- Corrosion and scaling risk assessment

All calculations are deterministic with zero hallucination.

References:
    - ASME Consensus on Operating Practices for Control of Feedwater/Boiler Water
    - ABMA Guidelines for Water Quality in Industrial Boilers
    - EPRI Boiler Water Chemistry Guidelines for Drum Boilers
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    BoilerWaterInput,
    BoilerWaterOutput,
    BoilerWaterLimits,
    WaterQualityResult,
    WaterQualityStatus,
    BoilerPressureClass,
    TreatmentProgram,
    CorrosionMechanism,
)
from greenlang.agents.process_heat.gl_016_water_treatment.config import (
    get_boiler_water_limits,
    get_phosphate_config,
    determine_pressure_class,
    ASME_BOILER_WATER_LIMITS,
    PhosphateTreatmentConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Per ASME/ABMA/EPRI Guidelines
# =============================================================================

class BoilerWaterConstants:
    """Constants for boiler water chemistry calculations."""

    # pH temperature correction factor (per 10C)
    PH_TEMP_CORRECTION_PER_10C = 0.03

    # Conductivity to TDS conversion factor (typical)
    CONDUCTIVITY_TO_TDS_FACTOR = 0.65

    # Silica solubility limit factors by pressure (ppm)
    SILICA_SOLUBILITY = {
        0: 200,     # 0 psig
        100: 150,   # 100 psig
        200: 100,   # 200 psig
        300: 60,    # 300 psig
        400: 40,    # 400 psig
        600: 25,    # 600 psig
        900: 10,    # 900 psig
    }

    # Phosphate molecular weight
    MW_PO4 = 95.0
    MW_NA = 23.0
    MW_NA3PO4 = 164.0
    MW_NA2HPO4 = 142.0

    # Corrosion risk thresholds
    IRON_WARNING_PPB = 100
    IRON_CRITICAL_PPB = 200
    COPPER_WARNING_PPB = 20
    COPPER_CRITICAL_PPB = 50

    # Steam purity limits (cation conductivity umho/cm)
    STEAM_PURITY_EXCELLENT = 0.2
    STEAM_PURITY_GOOD = 0.5
    STEAM_PURITY_ACCEPTABLE = 1.0


# =============================================================================
# BOILER WATER ANALYZER CLASS
# =============================================================================

class BoilerWaterAnalyzer:
    """
    Analyzes boiler water chemistry per ASME/ABMA guidelines.

    This class provides deterministic analysis of boiler water chemistry
    including pH control, phosphate management, and risk assessment.
    All calculations follow engineering standards with full provenance.

    Attributes:
        pressure_class: Boiler pressure classification
        treatment_program: Water treatment program type
        limits: ASME recommended limits for pressure class

    Example:
        >>> analyzer = BoilerWaterAnalyzer(
        ...     pressure_class=BoilerPressureClass.MEDIUM_PRESSURE,
        ...     treatment_program=TreatmentProgram.COORDINATED_PHOSPHATE
        ... )
        >>> result = analyzer.analyze(boiler_water_input)
        >>> print(result.overall_status)
    """

    def __init__(
        self,
        pressure_class: BoilerPressureClass = BoilerPressureClass.MEDIUM_PRESSURE,
        treatment_program: TreatmentProgram = TreatmentProgram.PHOSPHATE_POLYMER,
    ) -> None:
        """
        Initialize BoilerWaterAnalyzer.

        Args:
            pressure_class: Boiler pressure classification
            treatment_program: Water treatment program type
        """
        self.pressure_class = pressure_class
        self.treatment_program = treatment_program

        # Load ASME limits for pressure class
        self._load_limits()

        # Load phosphate program configuration
        self.phosphate_config = get_phosphate_config(treatment_program)

        logger.info(
            f"BoilerWaterAnalyzer initialized: {pressure_class.value}, "
            f"{treatment_program.value}"
        )

    def _load_limits(self) -> None:
        """Load ASME recommended limits for pressure class."""
        self.limits = get_boiler_water_limits(self.pressure_class)

    def analyze(self, input_data: BoilerWaterInput) -> BoilerWaterOutput:
        """
        Analyze boiler water chemistry.

        Args:
            input_data: Boiler water sample data

        Returns:
            BoilerWaterOutput with comprehensive analysis results

        Raises:
            ValueError: If input data is invalid
        """
        start_time = datetime.now(timezone.utc)
        logger.debug(f"Analyzing boiler water sample: {input_data.sample_id}")

        # Determine pressure class from operating pressure
        actual_pressure_class = determine_pressure_class(input_data.operating_pressure_psig)
        if actual_pressure_class != self.pressure_class:
            logger.warning(
                f"Operating pressure suggests {actual_pressure_class.value} "
                f"but configured for {self.pressure_class.value}"
            )

        # Analyze individual parameters
        parameter_results: List[WaterQualityResult] = []

        # pH analysis
        ph_result = self._analyze_ph(input_data)
        parameter_results.append(ph_result)

        # Phosphate analysis
        if input_data.phosphate_ppm is not None:
            phosphate_result = self._analyze_phosphate(input_data)
            parameter_results.append(phosphate_result)

        # Conductivity analysis
        conductivity_result = self._analyze_conductivity(input_data)
        parameter_results.append(conductivity_result)

        # Silica analysis
        if input_data.silica_ppm is not None:
            silica_result = self._analyze_silica(input_data)
            parameter_results.append(silica_result)

        # TDS analysis
        if input_data.tds_ppm is not None:
            tds_result = self._analyze_tds(input_data)
            parameter_results.append(tds_result)

        # Iron analysis
        if input_data.iron_ppb is not None:
            iron_result = self._analyze_iron(input_data)
            parameter_results.append(iron_result)

        # Copper analysis
        if input_data.copper_ppb is not None:
            copper_result = self._analyze_copper(input_data)
            parameter_results.append(copper_result)

        # Dissolved oxygen analysis
        if input_data.dissolved_oxygen_ppb is not None:
            do_result = self._analyze_dissolved_oxygen(input_data)
            parameter_results.append(do_result)

        # Calculate phosphate control (Na:PO4 ratio)
        phosphate_ratio = None
        phosphate_control_status = None
        if (input_data.phosphate_ppm is not None and
                input_data.p_alkalinity_ppm is not None):
            phosphate_ratio, phosphate_control_status = self._analyze_phosphate_control(
                input_data
            )

        # Calculate risk scores
        corrosion_risk, corrosion_mechanisms = self._calculate_corrosion_risk(
            input_data, parameter_results
        )
        scaling_risk = self._calculate_scaling_risk(input_data, parameter_results)
        deposition_risk = self._calculate_deposition_risk(input_data, parameter_results)

        # Determine overall status
        overall_status = self._determine_overall_status(parameter_results)
        status_message = self._generate_status_message(
            overall_status, parameter_results, corrosion_mechanisms
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            input_data, parameter_results, corrosion_mechanisms
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(input_data)

        # Calculate processing time
        processing_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        return BoilerWaterOutput(
            sample_id=input_data.sample_id,
            timestamp=datetime.now(timezone.utc),
            overall_status=overall_status,
            status_message=status_message,
            parameter_results=parameter_results,
            phosphate_sodium_ratio=phosphate_ratio,
            phosphate_control_status=phosphate_control_status,
            corrosion_risk_score=corrosion_risk,
            corrosion_mechanisms=corrosion_mechanisms,
            scaling_risk_score=scaling_risk,
            deposition_risk_score=deposition_risk,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
        )

    def _analyze_ph(self, input_data: BoilerWaterInput) -> WaterQualityResult:
        """
        Analyze pH per ASME guidelines.

        pH control is critical for corrosion prevention and phosphate
        treatment effectiveness.
        """
        value = input_data.ph

        # Get limits from phosphate config if available, else ASME defaults
        if self.phosphate_config:
            min_limit = self.phosphate_config.ph_min
            max_limit = self.phosphate_config.ph_max
        else:
            min_limit = self.limits.ph_min
            max_limit = self.limits.ph_max

        target = (min_limit + max_limit) / 2

        # Determine status
        if value < min_limit - 0.5 or value > max_limit + 0.5:
            status = WaterQualityStatus.CRITICAL
        elif value < min_limit or value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif abs(value - target) <= 0.3:
            status = WaterQualityStatus.EXCELLENT
        elif abs(value - target) <= 0.5:
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

    def _analyze_phosphate(self, input_data: BoilerWaterInput) -> WaterQualityResult:
        """Analyze phosphate concentration per treatment program."""
        value = input_data.phosphate_ppm or 0

        # Get limits from phosphate config
        if self.phosphate_config:
            min_limit = self.phosphate_config.phosphate_min_ppm
            max_limit = self.phosphate_config.phosphate_max_ppm
            target = self.phosphate_config.phosphate_target_ppm
        else:
            # Default limits for precipitate program
            min_limit = 10.0
            max_limit = 50.0
            target = 30.0

        # Determine status
        if value < min_limit * 0.5 or value > max_limit * 1.5:
            status = WaterQualityStatus.CRITICAL
        elif value < min_limit or value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif abs(value - target) / target <= 0.1:
            status = WaterQualityStatus.EXCELLENT
        elif abs(value - target) / target <= 0.2:
            status = WaterQualityStatus.GOOD
        else:
            status = WaterQualityStatus.ACCEPTABLE

        deviation_pct = ((value - target) / target) * 100 if target != 0 else 0

        return WaterQualityResult(
            parameter="Phosphate (PO4)",
            value=value,
            unit="ppm",
            min_limit=min_limit,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=round(deviation_pct, 2),
        )

    def _analyze_conductivity(self, input_data: BoilerWaterInput) -> WaterQualityResult:
        """Analyze specific conductivity."""
        value = input_data.specific_conductivity_umho
        max_limit = self.limits.conductivity_max_umho
        target = max_limit * 0.6  # Target 60% of limit

        # Determine status
        if value > max_limit * 1.2:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value <= target:
            status = WaterQualityStatus.EXCELLENT
        elif value <= max_limit * 0.8:
            status = WaterQualityStatus.GOOD
        else:
            status = WaterQualityStatus.WARNING

        deviation_pct = ((value - target) / target) * 100 if target != 0 else 0

        return WaterQualityResult(
            parameter="Specific Conductivity",
            value=value,
            unit="umho/cm",
            min_limit=None,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=round(deviation_pct, 2),
        )

    def _analyze_silica(self, input_data: BoilerWaterInput) -> WaterQualityResult:
        """
        Analyze silica concentration for carryover prevention.

        Silica limits are pressure-dependent due to increased volatility
        at higher pressures.
        """
        value = input_data.silica_ppm or 0
        max_limit = self.limits.silica_max_ppm

        # Get pressure-specific silica limit
        pressure = input_data.operating_pressure_psig
        pressure_limit = self._get_silica_limit_for_pressure(pressure)
        max_limit = min(max_limit, pressure_limit)

        target = max_limit * 0.5

        # Determine status
        if value > max_limit * 1.5:
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
            parameter="Silica (SiO2)",
            value=value,
            unit="ppm",
            min_limit=None,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=round(deviation_pct, 2),
        )

    def _analyze_tds(self, input_data: BoilerWaterInput) -> WaterQualityResult:
        """Analyze total dissolved solids."""
        value = input_data.tds_ppm or 0
        max_limit = self.limits.tds_max_ppm
        target = max_limit * 0.6

        if value > max_limit * 1.2:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.OUT_OF_SPEC
        elif value <= target:
            status = WaterQualityStatus.EXCELLENT
        elif value <= max_limit * 0.8:
            status = WaterQualityStatus.GOOD
        else:
            status = WaterQualityStatus.WARNING

        deviation_pct = ((value - target) / target) * 100 if target != 0 else 0

        return WaterQualityResult(
            parameter="Total Dissolved Solids",
            value=value,
            unit="ppm",
            min_limit=None,
            max_limit=max_limit,
            target_value=target,
            status=status,
            deviation_pct=round(deviation_pct, 2),
        )

    def _analyze_iron(self, input_data: BoilerWaterInput) -> WaterQualityResult:
        """Analyze iron concentration (corrosion indicator)."""
        value = input_data.iron_ppb or 0
        max_limit = 100.0  # ppb
        target = 50.0

        if value > BoilerWaterConstants.IRON_CRITICAL_PPB:
            status = WaterQualityStatus.CRITICAL
        elif value > BoilerWaterConstants.IRON_WARNING_PPB:
            status = WaterQualityStatus.WARNING
        elif value <= target:
            status = WaterQualityStatus.EXCELLENT
        else:
            status = WaterQualityStatus.GOOD

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

    def _analyze_copper(self, input_data: BoilerWaterInput) -> WaterQualityResult:
        """Analyze copper concentration (corrosion indicator)."""
        value = input_data.copper_ppb or 0
        max_limit = 15.0  # ppb
        target = 5.0

        if value > BoilerWaterConstants.COPPER_CRITICAL_PPB:
            status = WaterQualityStatus.CRITICAL
        elif value > BoilerWaterConstants.COPPER_WARNING_PPB:
            status = WaterQualityStatus.WARNING
        elif value <= target:
            status = WaterQualityStatus.EXCELLENT
        else:
            status = WaterQualityStatus.GOOD

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

    def _analyze_dissolved_oxygen(
        self, input_data: BoilerWaterInput
    ) -> WaterQualityResult:
        """Analyze dissolved oxygen (should be essentially zero in boiler)."""
        value = input_data.dissolved_oxygen_ppb or 0
        max_limit = 10.0  # ppb - should be very low in boiler water
        target = 0.0

        if value > 20:
            status = WaterQualityStatus.CRITICAL
        elif value > max_limit:
            status = WaterQualityStatus.WARNING
        elif value <= 5:
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

    def _analyze_phosphate_control(
        self, input_data: BoilerWaterInput
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Analyze phosphate control using Na:PO4 molar ratio.

        For coordinated/congruent phosphate programs, the Na:PO4 ratio
        determines whether free hydroxide or acid phosphate conditions exist.

        Na:PO4 = 3.0: Trisodium phosphate (neutral)
        Na:PO4 > 3.0: Free hydroxide (caustic)
        Na:PO4 < 2.6: Acid phosphate (corrosive)
        """
        if not self.phosphate_config:
            return None, None

        # Skip if not a coordinated/congruent program
        if self.treatment_program not in [
            TreatmentProgram.COORDINATED_PHOSPHATE,
            TreatmentProgram.CONGRUENT_PHOSPHATE,
        ]:
            return None, "Not applicable - precipitate program"

        phosphate_ppm = input_data.phosphate_ppm or 0
        p_alkalinity = input_data.p_alkalinity_ppm or 0

        if phosphate_ppm <= 0:
            return None, "Insufficient phosphate"

        # Calculate Na:PO4 ratio from phosphate and P-alkalinity
        # P-alkalinity = OH + 1/2 CO3 (assume CO3 negligible)
        # Na:PO4 = 3 + (OH as Na) / (PO4 as Na)
        # Simplified: Na:PO4 ~ 3 + (P-alk * 0.8) / (PO4 * 2.4)

        # Convert to molar basis
        po4_molar = phosphate_ppm / BoilerWaterConstants.MW_PO4
        na_from_po4 = po4_molar * 3 * BoilerWaterConstants.MW_NA

        # Estimate sodium from alkalinity (simplified)
        na_from_alkalinity = p_alkalinity * (BoilerWaterConstants.MW_NA / 50)

        # Total Na estimate
        total_na = na_from_po4 + na_from_alkalinity

        # Calculate ratio
        if po4_molar > 0:
            na_po4_ratio = (total_na / BoilerWaterConstants.MW_NA) / po4_molar
        else:
            na_po4_ratio = 3.0

        # Determine control status
        if self.phosphate_config.na_po4_ratio_min and self.phosphate_config.na_po4_ratio_max:
            if na_po4_ratio < self.phosphate_config.na_po4_ratio_min:
                status = "ACID PHOSPHATE - Risk of acid phosphate corrosion"
            elif na_po4_ratio > self.phosphate_config.na_po4_ratio_max:
                status = "EXCESS CAUSTIC - Risk of caustic gouging"
            elif abs(na_po4_ratio - (self.phosphate_config.na_po4_ratio_target or 2.8)) < 0.1:
                status = "OPTIMAL - Within control band"
            else:
                status = "ACCEPTABLE - Within limits"
        else:
            status = "Unable to assess - check configuration"

        return round(na_po4_ratio, 2), status

    def _calculate_corrosion_risk(
        self,
        input_data: BoilerWaterInput,
        results: List[WaterQualityResult],
    ) -> Tuple[float, List[CorrosionMechanism]]:
        """
        Calculate corrosion risk score and identify mechanisms.

        Returns:
            Tuple of (risk_score 0-100, list of corrosion mechanisms)
        """
        risk_score = 0.0
        mechanisms: List[CorrosionMechanism] = []

        # Check pH-related corrosion risks
        ph = input_data.ph
        if ph < 9.0:
            risk_score += 30
            mechanisms.append(CorrosionMechanism.HYDROGEN_DAMAGE)
        elif ph > 12.0:
            risk_score += 25
            mechanisms.append(CorrosionMechanism.CAUSTIC_EMBRITTLEMENT)
            mechanisms.append(CorrosionMechanism.CAUSTIC_GOUGING)

        # Check dissolved oxygen
        if input_data.dissolved_oxygen_ppb and input_data.dissolved_oxygen_ppb > 10:
            risk_score += 25
            mechanisms.append(CorrosionMechanism.OXYGEN_PITTING)

        # Check iron levels (indicator of active corrosion)
        if input_data.iron_ppb:
            if input_data.iron_ppb > BoilerWaterConstants.IRON_CRITICAL_PPB:
                risk_score += 20
            elif input_data.iron_ppb > BoilerWaterConstants.IRON_WARNING_PPB:
                risk_score += 10

        # Check phosphate control for acid phosphate corrosion
        if input_data.phosphate_ppm and input_data.phosphate_ppm > 0:
            if ph < 9.5 and input_data.phosphate_ppm > 5:
                risk_score += 15
                mechanisms.append(CorrosionMechanism.ACID_PHOSPHATE_CORROSION)

        # Check for under-deposit corrosion conditions
        if input_data.tds_ppm and input_data.tds_ppm > self.limits.tds_max_ppm:
            risk_score += 10
            mechanisms.append(CorrosionMechanism.UNDER_DEPOSIT)

        # Cap at 100
        risk_score = min(risk_score, 100)

        return round(risk_score, 1), mechanisms

    def _calculate_scaling_risk(
        self,
        input_data: BoilerWaterInput,
        results: List[WaterQualityResult],
    ) -> float:
        """
        Calculate scaling risk score (0-100).

        Scaling risk increases with high hardness, silica, and TDS.
        """
        risk_score = 0.0

        # Silica risk
        if input_data.silica_ppm:
            silica_limit = self.limits.silica_max_ppm
            silica_ratio = input_data.silica_ppm / silica_limit
            if silica_ratio > 1.5:
                risk_score += 40
            elif silica_ratio > 1.0:
                risk_score += 25
            elif silica_ratio > 0.7:
                risk_score += 10

        # TDS risk
        if input_data.tds_ppm:
            tds_limit = self.limits.tds_max_ppm
            tds_ratio = input_data.tds_ppm / tds_limit
            if tds_ratio > 1.2:
                risk_score += 30
            elif tds_ratio > 1.0:
                risk_score += 15
            elif tds_ratio > 0.8:
                risk_score += 5

        # Conductivity risk
        cond_limit = self.limits.conductivity_max_umho
        cond_ratio = input_data.specific_conductivity_umho / cond_limit
        if cond_ratio > 1.0:
            risk_score += 20

        return min(risk_score, 100)

    def _calculate_deposition_risk(
        self,
        input_data: BoilerWaterInput,
        results: List[WaterQualityResult],
    ) -> float:
        """
        Calculate deposition risk score (0-100).

        Deposition risk from iron, copper, and suspended solids.
        """
        risk_score = 0.0

        # Iron deposition
        if input_data.iron_ppb:
            if input_data.iron_ppb > 200:
                risk_score += 40
            elif input_data.iron_ppb > 100:
                risk_score += 25
            elif input_data.iron_ppb > 50:
                risk_score += 10

        # Copper deposition
        if input_data.copper_ppb:
            if input_data.copper_ppb > 50:
                risk_score += 30
            elif input_data.copper_ppb > 20:
                risk_score += 15
            elif input_data.copper_ppb > 10:
                risk_score += 5

        # High TDS can lead to precipitation
        if input_data.tds_ppm and input_data.tds_ppm > self.limits.tds_max_ppm:
            risk_score += 20

        return min(risk_score, 100)

    def _determine_overall_status(
        self, results: List[WaterQualityResult]
    ) -> WaterQualityStatus:
        """Determine overall water quality status from individual results."""
        if not results:
            return WaterQualityStatus.WARNING

        # Define status priority (lower = worse)
        status_priority = {
            WaterQualityStatus.CRITICAL: 0,
            WaterQualityStatus.OUT_OF_SPEC: 1,
            WaterQualityStatus.WARNING: 2,
            WaterQualityStatus.ACCEPTABLE: 3,
            WaterQualityStatus.GOOD: 4,
            WaterQualityStatus.EXCELLENT: 5,
            # Also handle string values (from use_enum_values=True)
            "critical": 0,
            "out_of_spec": 1,
            "warning": 2,
            "acceptable": 3,
            "good": 4,
            "excellent": 5,
        }

        # Find worst status
        worst_priority = 5
        for result in results:
            priority = status_priority.get(result.status, 3)
            if priority < worst_priority:
                worst_priority = priority

        # Map priority back to status
        priority_to_status = {
            0: WaterQualityStatus.CRITICAL,
            1: WaterQualityStatus.OUT_OF_SPEC,
            2: WaterQualityStatus.WARNING,
            3: WaterQualityStatus.ACCEPTABLE,
            4: WaterQualityStatus.GOOD,
            5: WaterQualityStatus.EXCELLENT,
        }

        return priority_to_status.get(worst_priority, WaterQualityStatus.ACCEPTABLE)

    def _generate_status_message(
        self,
        overall_status: WaterQualityStatus,
        results: List[WaterQualityResult],
        mechanisms: List[CorrosionMechanism],
    ) -> str:
        """Generate human-readable status message."""
        # Handle both enum and string values for status
        critical_statuses = [WaterQualityStatus.CRITICAL, WaterQualityStatus.OUT_OF_SPEC,
                            "critical", "out_of_spec"]
        out_of_spec = [r for r in results if r.status in critical_statuses]

        if overall_status == WaterQualityStatus.CRITICAL:
            params = ", ".join([r.parameter for r in out_of_spec])
            return f"CRITICAL: Immediate attention required. Parameters: {params}"
        elif overall_status == WaterQualityStatus.OUT_OF_SPEC:
            params = ", ".join([r.parameter for r in out_of_spec])
            return f"OUT OF SPEC: {len(out_of_spec)} parameter(s) exceed limits: {params}"
        elif overall_status == WaterQualityStatus.WARNING:
            return "WARNING: Some parameters approaching limits"
        elif overall_status == WaterQualityStatus.ACCEPTABLE:
            return "ACCEPTABLE: All parameters within limits but not optimal"
        elif overall_status == WaterQualityStatus.GOOD:
            return "GOOD: Water chemistry well controlled"
        else:
            return "EXCELLENT: Optimal water chemistry"

    def _generate_recommendations(
        self,
        input_data: BoilerWaterInput,
        results: List[WaterQualityResult],
        mechanisms: List[CorrosionMechanism],
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        for result in results:
            if result.status in [WaterQualityStatus.CRITICAL, WaterQualityStatus.OUT_OF_SPEC]:
                rec = self._get_parameter_recommendation(result, input_data)
                if rec:
                    recommendations.append(rec)

        # Add corrosion mechanism-specific recommendations
        for mechanism in mechanisms:
            rec = self._get_corrosion_recommendation(mechanism)
            if rec and rec not in recommendations:
                recommendations.append(rec)

        return recommendations

    def _get_parameter_recommendation(
        self,
        result: WaterQualityResult,
        input_data: BoilerWaterInput,
    ) -> Optional[str]:
        """Get recommendation for specific parameter deviation."""
        param = result.parameter.lower()

        if "ph" in param:
            if result.value < (result.min_limit or 9.0):
                return "Increase caustic/phosphate dosing to raise pH"
            else:
                return "Reduce caustic dosing - pH too high"

        elif "phosphate" in param:
            if result.value < (result.min_limit or 0):
                return "Increase phosphate dosing to maintain control range"
            else:
                return "Increase blowdown to reduce phosphate concentration"

        elif "conductivity" in param or "tds" in param:
            return "Increase blowdown rate to reduce dissolved solids"

        elif "silica" in param:
            return "Increase blowdown rate - silica approaching carryover limit"

        elif "iron" in param:
            return "Investigate corrosion source - elevated iron indicates active corrosion"

        elif "copper" in param:
            return "Check condensate pH - elevated copper indicates condensate corrosion"

        elif "oxygen" in param:
            return "Verify deaerator operation and oxygen scavenger dosing"

        return None

    def _get_corrosion_recommendation(
        self, mechanism: CorrosionMechanism
    ) -> Optional[str]:
        """Get recommendation for specific corrosion mechanism."""
        recommendations = {
            CorrosionMechanism.OXYGEN_PITTING:
                "Verify deaerator performance and increase oxygen scavenger dose",
            CorrosionMechanism.CAUSTIC_EMBRITTLEMENT:
                "Reduce pH - risk of caustic stress corrosion cracking",
            CorrosionMechanism.CAUSTIC_GOUGING:
                "Reduce caustic levels and verify waterwall circulation",
            CorrosionMechanism.HYDROGEN_DAMAGE:
                "Increase pH immediately - hydrogen damage risk at low pH",
            CorrosionMechanism.ACID_PHOSPHATE_CORROSION:
                "Increase Na:PO4 ratio - acid phosphate conditions detected",
            CorrosionMechanism.UNDER_DEPOSIT:
                "Chemical clean required - deposit-related corrosion risk",
        }
        return recommendations.get(mechanism)

    def _get_silica_limit_for_pressure(self, pressure_psig: float) -> float:
        """
        Get silica concentration limit for given pressure.

        Silica volatility increases with pressure, requiring lower limits
        at higher pressures to prevent turbine deposits.
        """
        pressures = sorted(BoilerWaterConstants.SILICA_SOLUBILITY.keys())

        # Find bracketing pressures
        for i, p in enumerate(pressures):
            if pressure_psig <= p:
                return BoilerWaterConstants.SILICA_SOLUBILITY[p]
            if i < len(pressures) - 1 and pressure_psig < pressures[i + 1]:
                # Linear interpolation
                p1, p2 = p, pressures[i + 1]
                s1 = BoilerWaterConstants.SILICA_SOLUBILITY[p1]
                s2 = BoilerWaterConstants.SILICA_SOLUBILITY[p2]
                return s1 + (s2 - s1) * (pressure_psig - p1) / (p2 - p1)

        # Return lowest limit for very high pressures
        return BoilerWaterConstants.SILICA_SOLUBILITY[pressures[-1]]

    def _calculate_provenance_hash(self, input_data: BoilerWaterInput) -> str:
        """Calculate SHA-256 hash for audit trail."""
        import json
        data_str = json.dumps(input_data.dict(), sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_cycles_of_concentration(
    boiler_tds_ppm: float,
    feedwater_tds_ppm: float,
) -> float:
    """
    Calculate cycles of concentration from TDS values.

    Cycles = Boiler TDS / Feedwater TDS

    Args:
        boiler_tds_ppm: Boiler water TDS (ppm)
        feedwater_tds_ppm: Feedwater TDS (ppm)

    Returns:
        Cycles of concentration
    """
    if feedwater_tds_ppm <= 0:
        return 1.0
    return boiler_tds_ppm / feedwater_tds_ppm


def calculate_blowdown_rate_from_cycles(cycles: float) -> float:
    """
    Calculate blowdown rate percentage from cycles of concentration.

    Blowdown % = 100 / Cycles

    Args:
        cycles: Cycles of concentration

    Returns:
        Blowdown rate percentage
    """
    if cycles <= 1:
        return 100.0
    return 100.0 / cycles


def estimate_tds_from_conductivity(
    conductivity_umho: float,
    conversion_factor: float = 0.65,
) -> float:
    """
    Estimate TDS from specific conductivity.

    TDS (ppm) = Conductivity (umho/cm) * Conversion Factor

    Args:
        conductivity_umho: Specific conductivity in umho/cm
        conversion_factor: Conversion factor (typically 0.55-0.70)

    Returns:
        Estimated TDS in ppm
    """
    return conductivity_umho * conversion_factor
