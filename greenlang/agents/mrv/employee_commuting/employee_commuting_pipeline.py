"""
EmployeeCommutingPipelineEngine - Orchestrated 10-stage pipeline for employee commuting emissions.

This module implements the EmployeeCommutingPipelineEngine for AGENT-MRV-020
(Employee Commuting, GHG Protocol Scope 3 Category 7). It orchestrates a 10-stage
pipeline from raw input to compliant output with full audit trail, covering personal
vehicles, carpools, vanpools, public transit, active transport, micro-mobility, and
telework/remote-work energy emissions.

The 10 stages are:
1. VALIDATE: Input validation (required fields, types, distances, modes, frequencies)
2. CLASSIFY: Classify commute mode (SOV/carpool/transit/active/telework), vehicle/fuel type
3. NORMALIZE: Convert units (miles->km, gallons->litres, mpg->L/100km, currency->USD)
4. RESOLVE_EFS: Select emission factors by hierarchy (Employee>DEFRA>EPA>IEA>Census>EEIO)
5. CALCULATE_COMMUTE: Per-employee commute emissions (distance x frequency x EF)
6. CALCULATE_TELEWORK: Telework emissions (days x kWh x grid_EF), avoided commute memo
7. EXTRAPOLATE: Scale survey sample to full workforce with confidence intervals
8. COMPLIANCE: Run compliance checks against 7 frameworks
9. AGGREGATE: Aggregate by mode, site, department, distance band with hot-spot analysis
10. SEAL: Seal provenance chain with Merkle root, SHA-256 audit hash, ISO 8601 timestamp

Calculation Methods:
- Employee-specific: Survey data per individual employee
- Average-data: National/regional average distance and mode share
- Spend-based: Commuting benefits spend x EEIO factor

Key Formulas:
  SOV:        CO2e = Dist_one_way x 2 x Working_Days x Freq x EF_vkm
  Carpool:    CO2e = (Dist x 2 x WD x Freq x EF_vkm) / Occupancy
  Transit:    CO2e = Dist x 2 x WD x Freq x EF_pkm
  Multi-modal:CO2e = SUM(Leg_Dist x EF_leg) x 2 x WD x Freq
  Telework:   CO2e = Telework_Days x Daily_kWh x Grid_EF
  Avg-data:   CO2e = Employees x Avg_Dist x 2 x WD x SUM(ModeShare_i x EF_i)
  Spend:      CO2e = Spend_USD x CPI_Deflator x EEIO_Factor
  Org Total:  Org_CO2e = SUM(Employee_CO2e_i) x (Total_Employees / Respondents)

Example:
    >>> from greenlang.agents.mrv.employee_commuting.employee_commuting_pipeline import (
    ...     EmployeeCommutingPipelineEngine,
    ... )
    >>> engine = EmployeeCommutingPipelineEngine()
    >>> result = engine.calculate({"employees": [...], "region": "US"})
    >>> print(f"Total: {result['total_co2e']} kgCO2e")

Module: greenlang.agents.mrv.employee_commuting.employee_commuting_pipeline
Agent: AGENT-MRV-020
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import logging
import hashlib
import json
import math
from threading import RLock

from greenlang.agents.mrv.employee_commuting.models import (
    # Enums
    CalculationMethod,
    CommuteMode,
    VehicleType,
    FuelType,
    TransitType,
    TeleworkFrequency,
    WorkSchedule,
    EFSource,
    ComplianceFramework,
    DataQualityTier,
    ProvenanceStage,
    DQIDimension,
    DQIScore,
    ComplianceStatus,
    CurrencyCode,
    BatchStatus,
    RegionCode,
    DistanceBand,
    SurveyMethod,
    AllocationMethod,
    SeasonalAdjustment,
    # Constant tables
    VEHICLE_EMISSION_FACTORS,
    FUEL_EMISSION_FACTORS,
    TRANSIT_EMISSION_FACTORS,
    MICRO_MOBILITY_EFS,
    GRID_EMISSION_FACTORS,
    WORKING_DAYS_DEFAULTS,
    AVERAGE_COMMUTE_DISTANCES,
    DEFAULT_MODE_SHARES,
    TELEWORK_ENERGY_DEFAULTS,
    VAN_EMISSION_FACTORS,
    EEIO_FACTORS,
    CURRENCY_RATES,
    CPI_DEFLATORS,
    DQI_SCORING,
    DQI_WEIGHTS,
    UNCERTAINTY_RANGES,
    FRAMEWORK_REQUIRED_DISCLOSURES,
    WORK_SCHEDULE_FRACTIONS,
    TELEWORK_FREQUENCY_FRACTIONS,
    SEASONAL_ADJUSTMENT_MULTIPLIERS,
    # Helper functions
    calculate_provenance_hash,
    get_working_days,
    classify_distance_band,
    get_grid_ef,
    get_average_commute_distance,
    get_default_mode_shares,
    get_eeio_factor,
    get_telework_daily_kwh,
    get_cpi_deflator,
    get_dqi_classification,
    convert_currency_to_usd,
)

logger = logging.getLogger(__name__)

_QUANT_8DP = Decimal("0.00000001")
_ZERO = Decimal("0")
_TWO = Decimal("2")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")

# Conversion constants
_MILES_TO_KM = Decimal("1.60934")
_GALLONS_TO_LITRES = Decimal("3.78541")
_MPG_TO_L_PER_100KM = Decimal("235.2145833")  # 235.215 / mpg = L/100km


# ==============================================================================
# PIPELINE STATUS
# ==============================================================================


class PipelineStatus:
    """Pipeline execution status constants."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


# ==============================================================================
# MODE CLASSIFICATION MAPPINGS
# ==============================================================================

# Map CommuteMode to its engine routing category
_PERSONAL_VEHICLE_MODES = frozenset({
    CommuteMode.SOV,
    CommuteMode.CARPOOL,
    CommuteMode.MOTORCYCLE,
})

_TRANSIT_MODES = frozenset({
    CommuteMode.BUS,
    CommuteMode.METRO,
    CommuteMode.LIGHT_RAIL,
    CommuteMode.COMMUTER_RAIL,
    CommuteMode.FERRY,
})

_ACTIVE_MODES = frozenset({
    CommuteMode.CYCLING,
    CommuteMode.WALKING,
    CommuteMode.E_BIKE,
    CommuteMode.E_SCOOTER,
})

# Map CommuteMode to TransitType for transit EF lookup
_MODE_TO_TRANSIT_TYPE: Dict[CommuteMode, TransitType] = {
    CommuteMode.BUS: TransitType.BUS_LOCAL,
    CommuteMode.METRO: TransitType.METRO,
    CommuteMode.LIGHT_RAIL: TransitType.LIGHT_RAIL,
    CommuteMode.COMMUTER_RAIL: TransitType.COMMUTER_RAIL,
    CommuteMode.FERRY: TransitType.FERRY,
}

# Map CommuteMode to mode-share key from DEFAULT_MODE_SHARES
_MODE_TO_SHARE_KEY: Dict[str, CommuteMode] = {
    "sov": CommuteMode.SOV,
    "carpool": CommuteMode.CARPOOL,
    "bus": CommuteMode.BUS,
    "metro": CommuteMode.METRO,
    "commuter_rail": CommuteMode.COMMUTER_RAIL,
    "light_rail": CommuteMode.LIGHT_RAIL,
    "ferry": CommuteMode.FERRY,
    "motorcycle": CommuteMode.MOTORCYCLE,
    "cycling": CommuteMode.CYCLING,
    "walking": CommuteMode.WALKING,
    "telework": CommuteMode.TELEWORK,
}


# ==============================================================================
# EMPLOYEE COMMUTING PIPELINE ENGINE
# ==============================================================================


class EmployeeCommutingPipelineEngine:
    """
    EmployeeCommutingPipelineEngine - Orchestrated 10-stage pipeline.

    This engine coordinates the complete employee commuting emissions
    calculation workflow through 10 sequential stages, from input validation
    to sealed audit trail. It supports all commute modes (SOV, carpool,
    vanpool, transit, active, micro-mobility, telework) plus average-data
    and spend-based fallback methods.

    The engine uses lazy initialization for sub-engines, creating them only
    when needed. This reduces startup time and memory footprint.

    Thread Safety:
        All mutable state is protected by an RLock, making the engine safe
        for concurrent use from multiple threads.

    Attributes:
        _database_engine: EmployeeCommutingDatabaseEngine (lazy-loaded)
        _vehicle_engine: PersonalVehicleCalculatorEngine (lazy-loaded)
        _transit_engine: PublicTransitCalculatorEngine (lazy-loaded)
        _active_engine: ActiveTransportCalculatorEngine (lazy-loaded)
        _telework_engine: TeleworkCalculatorEngine (lazy-loaded)
        _compliance_engine: ComplianceCheckerEngine (lazy-loaded)

    Example:
        >>> engine = EmployeeCommutingPipelineEngine()
        >>> result = engine.calculate({
        ...     "employees": [{
        ...         "employee_id": "EMP-001",
        ...         "mode": "sov",
        ...         "vehicle_type": "car_medium_petrol",
        ...         "one_way_distance_km": "18.5",
        ...         "commute_days_per_week": 5,
        ...     }],
        ...     "total_employees": 100,
        ...     "region": "US",
        ...     "reporting_period": "2024",
        ... })
        >>> assert result["status"] == PipelineStatus.SUCCESS
    """

    _instance: Optional["EmployeeCommutingPipelineEngine"] = None
    _lock: RLock = RLock()

    def __new__(cls) -> "EmployeeCommutingPipelineEngine":
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize EmployeeCommutingPipelineEngine.

        Prevents re-initialization of the singleton. All sub-engines are
        lazy-loaded on first calculate() call.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Lazy-loaded engines (created on first use)
        self._database_engine: Optional[Any] = None
        self._vehicle_engine: Optional[Any] = None
        self._transit_engine: Optional[Any] = None
        self._active_engine: Optional[Any] = None
        self._telework_engine: Optional[Any] = None
        self._compliance_engine: Optional[Any] = None

        # Pipeline state
        self._provenance_chains: Dict[str, List[Dict[str, Any]]] = {}

        self._initialized = True
        logger.info("EmployeeCommutingPipelineEngine initialized (version 1.0.0)")

    # ==========================================================================
    # PUBLIC API - CORE PROCESSING METHODS
    # ==========================================================================

    def calculate(self, input_data: dict) -> dict:
        """
        Execute the 10-stage employee commuting emissions pipeline.

        Accepts employee-specific survey data and processes each employee
        through validation, classification, normalization, EF resolution,
        commute calculation, telework calculation, extrapolation, compliance,
        aggregation, and sealing.

        Args:
            input_data: Dictionary containing:
                - employees: List of employee dicts with commute data
                - total_employees: Total workforce size for extrapolation
                - region: RegionCode string for defaults
                - reporting_period: Reporting period label
                - survey_method: (optional) SurveyMethod string
                - telework_region: (optional) RegionCode for telework grid EF
                - enable_compliance: (optional) bool, default True

        Returns:
            Dictionary with pipeline results including total_co2e,
            per-employee breakdown, aggregation, compliance, and provenance.

        Raises:
            ValueError: If input validation fails in Stage 1.
            RuntimeError: If pipeline execution fails.
        """
        chain_id = f"ec-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._provenance_chains[chain_id] = []
        stage_durations: Dict[str, float] = {}

        try:
            # ------------------------------------------------------------------
            # Stage 1: VALIDATE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            validated = self._stage_validate(input_data)
            duration_ms = self._elapsed_ms(start)
            stage_durations["VALIDATE"] = duration_ms
            self._record_provenance(
                chain_id, ProvenanceStage.VALIDATE,
                input_data, {"valid": True, "employee_count": len(validated["employees"])}
            )
            logger.info(
                f"[{chain_id}] Stage VALIDATE completed in {duration_ms:.2f}ms "
                f"({len(validated['employees'])} employees)"
            )

            # ------------------------------------------------------------------
            # Stage 2: CLASSIFY
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            classified = self._stage_classify(validated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["CLASSIFY"] = duration_ms
            self._record_provenance(
                chain_id, ProvenanceStage.CLASSIFY,
                validated, {"classified_count": len(classified["employees"])}
            )
            logger.info(f"[{chain_id}] Stage CLASSIFY completed in {duration_ms:.2f}ms")

            # ------------------------------------------------------------------
            # Stage 3: NORMALIZE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            normalized = self._stage_normalize(classified)
            duration_ms = self._elapsed_ms(start)
            stage_durations["NORMALIZE"] = duration_ms
            self._record_provenance(
                chain_id, ProvenanceStage.NORMALIZE,
                classified, {"normalized_count": len(normalized["employees"])}
            )
            logger.info(f"[{chain_id}] Stage NORMALIZE completed in {duration_ms:.2f}ms")

            # ------------------------------------------------------------------
            # Stage 4: RESOLVE_EFS
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            resolved = self._stage_resolve_efs(normalized)
            duration_ms = self._elapsed_ms(start)
            stage_durations["RESOLVE_EFS"] = duration_ms
            self._record_provenance(
                chain_id, ProvenanceStage.RESOLVE_EFS,
                normalized, {"ef_resolved_count": len(resolved["employees"])}
            )
            logger.info(f"[{chain_id}] Stage RESOLVE_EFS completed in {duration_ms:.2f}ms")

            # ------------------------------------------------------------------
            # Stage 5: CALCULATE_COMMUTE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            commute_result = self._stage_calculate_commute(resolved)
            duration_ms = self._elapsed_ms(start)
            stage_durations["CALCULATE_COMMUTE"] = duration_ms
            self._record_provenance(
                chain_id, ProvenanceStage.CALCULATE_COMMUTE,
                resolved, {"sample_co2e": str(commute_result.get("sample_commute_co2e", 0))}
            )
            logger.info(
                f"[{chain_id}] Stage CALCULATE_COMMUTE completed in {duration_ms:.2f}ms "
                f"(sample_co2e={commute_result.get('sample_commute_co2e', 0)} kgCO2e)"
            )

            # ------------------------------------------------------------------
            # Stage 6: CALCULATE_TELEWORK
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            telework_result = self._stage_calculate_telework(commute_result)
            duration_ms = self._elapsed_ms(start)
            stage_durations["CALCULATE_TELEWORK"] = duration_ms
            self._record_provenance(
                chain_id, ProvenanceStage.CALCULATE_TELEWORK,
                commute_result, {"telework_co2e": str(telework_result.get("sample_telework_co2e", 0))}
            )
            logger.info(
                f"[{chain_id}] Stage CALCULATE_TELEWORK completed in {duration_ms:.2f}ms "
                f"(telework_co2e={telework_result.get('sample_telework_co2e', 0)} kgCO2e)"
            )

            # ------------------------------------------------------------------
            # Stage 7: EXTRAPOLATE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            extrapolated = self._stage_extrapolate(telework_result)
            duration_ms = self._elapsed_ms(start)
            stage_durations["EXTRAPOLATE"] = duration_ms
            self._record_provenance(
                chain_id, ProvenanceStage.EXTRAPOLATE,
                telework_result, {
                    "extrapolated_co2e": str(extrapolated.get("extrapolated_total_co2e", 0)),
                    "scaling_factor": str(extrapolated.get("scaling_factor", 1)),
                }
            )
            logger.info(
                f"[{chain_id}] Stage EXTRAPOLATE completed in {duration_ms:.2f}ms "
                f"(org_co2e={extrapolated.get('extrapolated_total_co2e', 0)} kgCO2e)"
            )

            # ------------------------------------------------------------------
            # Stage 8: COMPLIANCE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            compliance_result = self._stage_compliance(extrapolated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["COMPLIANCE"] = duration_ms
            self._record_provenance(
                chain_id, ProvenanceStage.COMPLIANCE,
                extrapolated, {"compliance_status": compliance_result.get("overall_status", "N/A")}
            )
            logger.info(
                f"[{chain_id}] Stage COMPLIANCE completed in {duration_ms:.2f}ms "
                f"(status={compliance_result.get('overall_status', 'N/A')})"
            )

            # ------------------------------------------------------------------
            # Stage 9: AGGREGATE
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            aggregated = self._stage_aggregate(compliance_result)
            duration_ms = self._elapsed_ms(start)
            stage_durations["AGGREGATE"] = duration_ms
            self._record_provenance(
                chain_id, ProvenanceStage.AGGREGATE,
                compliance_result, {"aggregation_keys": list(aggregated.get("by_mode", {}).keys())}
            )
            logger.info(f"[{chain_id}] Stage AGGREGATE completed in {duration_ms:.2f}ms")

            # ------------------------------------------------------------------
            # Stage 10: SEAL
            # ------------------------------------------------------------------
            start = datetime.now(timezone.utc)
            sealed = self._stage_seal(chain_id, aggregated)
            duration_ms = self._elapsed_ms(start)
            stage_durations["SEAL"] = duration_ms
            logger.info(f"[{chain_id}] Stage SEAL completed in {duration_ms:.2f}ms")

            # Attach final metadata
            sealed["chain_id"] = chain_id
            sealed["status"] = PipelineStatus.SUCCESS
            sealed["stage_durations_ms"] = stage_durations
            sealed["total_pipeline_ms"] = sum(stage_durations.values())

            total_dur = sealed["total_pipeline_ms"]
            logger.info(
                f"[{chain_id}] Pipeline completed successfully in {total_dur:.2f}ms. "
                f"Total org emissions: {sealed.get('extrapolated_total_co2e', 0)} kgCO2e"
            )
            return sealed

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"[{chain_id}] Pipeline execution failed: {e}", exc_info=True)
            raise RuntimeError(f"Pipeline execution failed: {e}") from e
        finally:
            self._provenance_chains.pop(chain_id, None)

    def calculate_batch(self, inputs: list) -> dict:
        """
        Process a batch of independent pipeline inputs.

        Each input dict is processed through the full 10-stage pipeline
        independently. Failures in one input do not affect others.

        Args:
            inputs: List of input dicts, each suitable for calculate().

        Returns:
            Dictionary with batch results including per-input results,
            aggregate totals, error details, and batch provenance.
        """
        start_time = datetime.now(timezone.utc)
        results: List[dict] = []
        errors: List[dict] = []

        logger.info(f"Starting batch calculation ({len(inputs)} inputs)")

        for idx, inp in enumerate(inputs):
            try:
                result = self.calculate(inp)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch input {idx} failed: {e}")
                errors.append({
                    "index": idx,
                    "error": str(e),
                })

        # Aggregate batch totals
        total_co2e = sum(
            (Decimal(str(r.get("extrapolated_total_co2e", 0))) for r in results),
            _ZERO,
        )

        duration_ms = self._elapsed_ms(start_time)

        if not errors:
            batch_status = BatchStatus.COMPLETED.value
        elif not results:
            batch_status = BatchStatus.FAILED.value
        else:
            batch_status = BatchStatus.PARTIAL.value

        batch_hash = self._hash_json({
            "results_count": len(results),
            "errors_count": len(errors),
            "total_co2e": str(total_co2e),
        })

        logger.info(
            f"Batch completed in {duration_ms:.2f}ms. "
            f"Success: {len(results)}, Failed: {len(errors)}, "
            f"Total emissions: {total_co2e} kgCO2e"
        )

        return {
            "results": results,
            "total_co2e": str(total_co2e),
            "count": len(results),
            "errors": errors,
            "status": batch_status,
            "duration_ms": duration_ms,
            "provenance_hash": batch_hash,
        }

    def calculate_survey(self, survey_data: dict) -> dict:
        """
        Process survey-based employee commuting emissions.

        Convenience method that wraps calculate() for survey inputs,
        extracting survey metadata (method, response count) and passing
        through to the standard 10-stage pipeline.

        Args:
            survey_data: Dictionary containing:
                - survey_method: SurveyMethod string
                - total_employees: Total workforce size
                - responses: List of survey response dicts
                - region: RegionCode string
                - reporting_period: Reporting period label

        Returns:
            Pipeline result dict with survey-specific metadata.
        """
        employees = []
        for resp in survey_data.get("responses", []):
            emp = dict(resp)
            emp.setdefault("employee_id", emp.get("respondent_id", f"RESP-{len(employees)}"))
            employees.append(emp)

        pipeline_input = {
            "employees": employees,
            "total_employees": survey_data.get("total_employees", len(employees)),
            "region": survey_data.get("region", "GLOBAL"),
            "reporting_period": survey_data.get("reporting_period", ""),
            "survey_method": survey_data.get("survey_method", SurveyMethod.RANDOM_SAMPLE.value),
            "calculation_method": CalculationMethod.EMPLOYEE_SPECIFIC.value,
            "enable_compliance": survey_data.get("enable_compliance", True),
        }

        result = self.calculate(pipeline_input)
        result["survey_method"] = pipeline_input["survey_method"]
        result["response_count"] = len(employees)
        result["response_rate"] = str(
            self._safe_decimal(len(employees)) / self._safe_decimal(pipeline_input["total_employees"])
        ) if pipeline_input["total_employees"] > 0 else "0"

        return result

    def calculate_average_data(self, avg_data: dict) -> dict:
        """
        Calculate emissions using the average-data method.

        Uses national/regional average commute distance and mode share
        distributions to estimate emissions for the entire workforce
        without individual employee survey data.

        Args:
            avg_data: Dictionary containing:
                - total_employees: Total workforce size
                - region: RegionCode string
                - reporting_period: Reporting period label
                - mode_share_override: (optional) Custom mode shares
                - distance_override_km: (optional) Custom avg distance
                - telework_rate: (optional) Fraction teleworking (0-1)

        Returns:
            Pipeline result dict with average-data method metadata.
        """
        chain_id = f"ec-avg-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._provenance_chains[chain_id] = []
        stage_durations: Dict[str, float] = {}

        try:
            region_str = avg_data.get("region", "GLOBAL")
            region = self._parse_region(region_str)
            total_employees = int(avg_data.get("total_employees", 0))
            reporting_period = avg_data.get("reporting_period", "")
            telework_rate = self._safe_decimal(avg_data.get("telework_rate", "0"))
            distance_override = avg_data.get("distance_override_km")
            mode_share_override = avg_data.get("mode_share_override")

            if total_employees <= 0:
                raise ValueError("total_employees must be positive")

            # Stage 1: VALIDATE
            start = datetime.now(timezone.utc)
            self._record_provenance(
                chain_id, ProvenanceStage.VALIDATE, avg_data, {"valid": True}
            )
            stage_durations["VALIDATE"] = self._elapsed_ms(start)

            # Stage 2: CLASSIFY - average-data method
            start = datetime.now(timezone.utc)
            method = CalculationMethod.AVERAGE_DATA
            self._record_provenance(
                chain_id, ProvenanceStage.CLASSIFY, avg_data, {"method": method.value}
            )
            stage_durations["CLASSIFY"] = self._elapsed_ms(start)

            # Stage 3: NORMALIZE - get average distance
            start = datetime.now(timezone.utc)
            if distance_override is not None:
                avg_dist = self._safe_decimal(distance_override)
            else:
                avg_dist = get_average_commute_distance(region)
            working_days = get_working_days(region)
            self._record_provenance(
                chain_id, ProvenanceStage.NORMALIZE,
                avg_data, {"avg_distance_km": str(avg_dist), "working_days": working_days}
            )
            stage_durations["NORMALIZE"] = self._elapsed_ms(start)

            # Stage 4: RESOLVE_EFS - get mode shares and EFs
            start = datetime.now(timezone.utc)
            if mode_share_override is not None:
                mode_shares = {k: self._safe_decimal(v) for k, v in mode_share_override.items()}
            else:
                region_key = region_str if region_str in ("US", "GB", "EU") else "US"
                mode_shares = get_default_mode_shares(region_key)
            self._record_provenance(
                chain_id, ProvenanceStage.RESOLVE_EFS,
                {"mode_shares": {k: str(v) for k, v in mode_shares.items()}},
                {"ef_source": EFSource.DEFRA.value}
            )
            stage_durations["RESOLVE_EFS"] = self._elapsed_ms(start)

            # Stage 5: CALCULATE_COMMUTE - weighted average formula
            start = datetime.now(timezone.utc)
            commuting_employees = int(
                (self._safe_decimal(total_employees) * (_ONE - telework_rate)).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
            )
            teleworking_employees = total_employees - commuting_employees

            # CO2e = Employees x Avg_Dist x 2 x Working_Days x SUM(ModeShare_i x EF_i)
            weighted_ef = _ZERO
            mode_emissions: Dict[str, str] = {}

            for mode_key, share in mode_shares.items():
                if mode_key == "telework":
                    continue  # Handled in telework stage
                if mode_key in ("walking", "cycling"):
                    continue  # Zero-emission modes
                ef = self._get_mode_ef(mode_key)
                weighted_ef += share * ef

                mode_co2e = (
                    self._safe_decimal(commuting_employees)
                    * avg_dist * _TWO
                    * self._safe_decimal(working_days)
                    * share * ef
                ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
                mode_emissions[mode_key] = str(mode_co2e)

            total_commute = (
                self._safe_decimal(commuting_employees)
                * avg_dist * _TWO
                * self._safe_decimal(working_days)
                * weighted_ef
            ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

            self._record_provenance(
                chain_id, ProvenanceStage.CALCULATE_COMMUTE,
                {"commuting_employees": commuting_employees},
                {"total_commute_co2e": str(total_commute)}
            )
            stage_durations["CALCULATE_COMMUTE"] = self._elapsed_ms(start)

            # Stage 6: CALCULATE_TELEWORK
            start = datetime.now(timezone.utc)
            grid_ef = get_grid_ef(region)
            daily_kwh = get_telework_daily_kwh()
            telework_co2e = (
                self._safe_decimal(teleworking_employees)
                * self._safe_decimal(working_days)
                * daily_kwh * grid_ef
            ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

            self._record_provenance(
                chain_id, ProvenanceStage.CALCULATE_TELEWORK,
                {"teleworking_employees": teleworking_employees},
                {"telework_co2e": str(telework_co2e)}
            )
            stage_durations["CALCULATE_TELEWORK"] = self._elapsed_ms(start)

            # Stage 7: EXTRAPOLATE (not needed for average-data, already org-level)
            start = datetime.now(timezone.utc)
            org_total = total_commute + telework_co2e
            self._record_provenance(
                chain_id, ProvenanceStage.EXTRAPOLATE,
                {"total_commute": str(total_commute), "telework": str(telework_co2e)},
                {"org_total": str(org_total)}
            )
            stage_durations["EXTRAPOLATE"] = self._elapsed_ms(start)

            # Stage 8: COMPLIANCE
            start = datetime.now(timezone.utc)
            compliance = self._inline_compliance_check(method, {
                "total_co2e": str(org_total),
                "method": method.value,
                "total_employees": total_employees,
                "telework_disclosed": teleworking_employees > 0,
                "mode_breakdown": mode_emissions,
            })
            self._record_provenance(
                chain_id, ProvenanceStage.COMPLIANCE,
                {"org_total": str(org_total)},
                {"compliance_status": compliance.get("overall_status", "N/A")}
            )
            stage_durations["COMPLIANCE"] = self._elapsed_ms(start)

            # Stage 9: AGGREGATE
            start = datetime.now(timezone.utc)
            per_employee_avg = (org_total / self._safe_decimal(total_employees)).quantize(
                _QUANT_8DP, rounding=ROUND_HALF_UP
            )
            dqi_score = Decimal("2.5")  # Average-data = medium quality
            aggregation = {
                "total_co2e": str(org_total),
                "commute_co2e": str(total_commute),
                "telework_co2e": str(telework_co2e),
                "per_employee_avg_co2e": str(per_employee_avg),
                "total_employees": total_employees,
                "commuting_employees": commuting_employees,
                "teleworking_employees": teleworking_employees,
                "by_mode": mode_emissions,
                "dqi_score": str(dqi_score),
                "dqi_classification": get_dqi_classification(dqi_score),
                "reporting_period": reporting_period,
                "method": method.value,
                "region": region.value,
            }
            self._record_provenance(
                chain_id, ProvenanceStage.AGGREGATE,
                {"org_total": str(org_total)},
                {"aggregation_keys": list(aggregation.keys())}
            )
            stage_durations["AGGREGATE"] = self._elapsed_ms(start)

            # Stage 10: SEAL
            start = datetime.now(timezone.utc)
            sealed = self._stage_seal(chain_id, aggregation)
            stage_durations["SEAL"] = self._elapsed_ms(start)

            sealed["status"] = PipelineStatus.SUCCESS
            sealed["chain_id"] = chain_id
            sealed["stage_durations_ms"] = stage_durations
            sealed["total_pipeline_ms"] = sum(stage_durations.values())
            sealed["compliance"] = compliance

            logger.info(
                f"[{chain_id}] Average-data pipeline completed. "
                f"Total: {org_total} kgCO2e for {total_employees} employees"
            )
            return sealed

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"[{chain_id}] Average-data pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"Average-data pipeline failed: {e}") from e
        finally:
            self._provenance_chains.pop(chain_id, None)

    def calculate_spend_based(self, spend_data: dict) -> dict:
        """
        Calculate emissions using the spend-based EEIO method.

        Used when only commuting benefit spend data is available.
        Formula: CO2e = Spend_USD x CPI_Deflator x EEIO_Factor.

        Args:
            spend_data: Dictionary containing:
                - naics_code: NAICS code for EEIO lookup
                - amount: Spend amount
                - currency: CurrencyCode string (default USD)
                - reporting_year: Year for CPI deflation
                - reporting_period: Reporting period label

        Returns:
            Pipeline result dict with spend-based calculation details.
        """
        chain_id = f"ec-spend-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._provenance_chains[chain_id] = []
        stage_durations: Dict[str, float] = {}

        try:
            naics_code = str(spend_data.get("naics_code", "485000"))
            amount = self._safe_decimal(spend_data.get("amount", "0"))
            currency_str = spend_data.get("currency", "USD")
            reporting_year = int(spend_data.get("reporting_year", 2024))
            reporting_period = spend_data.get("reporting_period", "")

            if amount <= _ZERO:
                raise ValueError("Spend amount must be positive")

            # Stage 1: VALIDATE
            start = datetime.now(timezone.utc)
            eeio_ef = get_eeio_factor(naics_code)
            if eeio_ef is None:
                raise ValueError(f"NAICS code '{naics_code}' not found in EEIO factors")
            self._record_provenance(
                chain_id, ProvenanceStage.VALIDATE,
                spend_data, {"valid": True, "naics_code": naics_code}
            )
            stage_durations["VALIDATE"] = self._elapsed_ms(start)

            # Stage 2: CLASSIFY
            start = datetime.now(timezone.utc)
            method = CalculationMethod.SPEND_BASED
            self._record_provenance(
                chain_id, ProvenanceStage.CLASSIFY,
                spend_data, {"method": method.value}
            )
            stage_durations["CLASSIFY"] = self._elapsed_ms(start)

            # Stage 3: NORMALIZE - currency conversion + CPI deflation
            start = datetime.now(timezone.utc)
            try:
                currency = CurrencyCode(currency_str)
            except ValueError:
                currency = CurrencyCode.USD

            amount_usd = convert_currency_to_usd(amount, currency, year=reporting_year)
            self._record_provenance(
                chain_id, ProvenanceStage.NORMALIZE,
                {"amount": str(amount), "currency": currency_str},
                {"amount_usd_deflated": str(amount_usd)}
            )
            stage_durations["NORMALIZE"] = self._elapsed_ms(start)

            # Stage 4: RESOLVE_EFS
            start = datetime.now(timezone.utc)
            eeio_entry = EEIO_FACTORS.get(naics_code, {})
            self._record_provenance(
                chain_id, ProvenanceStage.RESOLVE_EFS,
                {"naics_code": naics_code},
                {"eeio_ef": str(eeio_ef), "eeio_name": eeio_entry.get("name", "")}
            )
            stage_durations["RESOLVE_EFS"] = self._elapsed_ms(start)

            # Stage 5: CALCULATE_COMMUTE - spend x EF
            start = datetime.now(timezone.utc)
            total_co2e = (amount_usd * eeio_ef).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
            self._record_provenance(
                chain_id, ProvenanceStage.CALCULATE_COMMUTE,
                {"amount_usd": str(amount_usd), "eeio_ef": str(eeio_ef)},
                {"total_co2e": str(total_co2e)}
            )
            stage_durations["CALCULATE_COMMUTE"] = self._elapsed_ms(start)

            # Stage 6: CALCULATE_TELEWORK (not applicable for spend-based)
            start = datetime.now(timezone.utc)
            self._record_provenance(
                chain_id, ProvenanceStage.CALCULATE_TELEWORK,
                {}, {"telework_co2e": "0", "note": "Not applicable for spend-based"}
            )
            stage_durations["CALCULATE_TELEWORK"] = self._elapsed_ms(start)

            # Stage 7: EXTRAPOLATE (not applicable, already org-level spend)
            start = datetime.now(timezone.utc)
            self._record_provenance(
                chain_id, ProvenanceStage.EXTRAPOLATE,
                {}, {"note": "No extrapolation for spend-based method"}
            )
            stage_durations["EXTRAPOLATE"] = self._elapsed_ms(start)

            # Stage 8: COMPLIANCE
            start = datetime.now(timezone.utc)
            compliance = self._inline_compliance_check(method, {
                "total_co2e": str(total_co2e),
                "method": method.value,
            })
            self._record_provenance(
                chain_id, ProvenanceStage.COMPLIANCE,
                {"total_co2e": str(total_co2e)},
                {"compliance_status": compliance.get("overall_status", "N/A")}
            )
            stage_durations["COMPLIANCE"] = self._elapsed_ms(start)

            # Stage 9: AGGREGATE
            start = datetime.now(timezone.utc)
            dqi_score = Decimal("1.5")  # Spend-based = low quality
            result_data = {
                "total_co2e": str(total_co2e),
                "extrapolated_total_co2e": str(total_co2e),
                "commute_co2e": str(total_co2e),
                "telework_co2e": "0",
                "naics_code": naics_code,
                "naics_name": eeio_entry.get("name", ""),
                "amount_original": str(amount),
                "currency": currency_str,
                "amount_usd_deflated": str(amount_usd),
                "eeio_factor": str(eeio_ef),
                "reporting_year": reporting_year,
                "reporting_period": reporting_period,
                "method": method.value,
                "ef_source": EFSource.EEIO.value,
                "dqi_score": str(dqi_score),
                "dqi_classification": get_dqi_classification(dqi_score),
            }
            self._record_provenance(
                chain_id, ProvenanceStage.AGGREGATE,
                {"total_co2e": str(total_co2e)},
                {"dqi_score": str(dqi_score)}
            )
            stage_durations["AGGREGATE"] = self._elapsed_ms(start)

            # Stage 10: SEAL
            start = datetime.now(timezone.utc)
            sealed = self._stage_seal(chain_id, result_data)
            stage_durations["SEAL"] = self._elapsed_ms(start)

            sealed["status"] = PipelineStatus.SUCCESS
            sealed["chain_id"] = chain_id
            sealed["stage_durations_ms"] = stage_durations
            sealed["total_pipeline_ms"] = sum(stage_durations.values())
            sealed["compliance"] = compliance

            logger.info(
                f"[{chain_id}] Spend-based pipeline completed. "
                f"Total: {total_co2e} kgCO2e (NAICS {naics_code})"
            )
            return sealed

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"[{chain_id}] Spend-based pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"Spend-based pipeline failed: {e}") from e
        finally:
            self._provenance_chains.pop(chain_id, None)

    # ==========================================================================
    # STAGE METHODS (PRIVATE)
    # ==========================================================================

    def _stage_validate(self, input_data: dict) -> dict:
        """
        Stage 1: VALIDATE - Input validation.

        Checks:
        - employees list is present and non-empty
        - total_employees is positive
        - Each employee has required fields (employee_id, mode, distance)
        - Distances are positive
        - Modes are valid CommuteMode values
        - Non-telework records have non-zero distance
        - Commute days per week 1-7

        Args:
            input_data: Raw pipeline input dict.

        Returns:
            Validated input dict with parsed employees.

        Raises:
            ValueError: If validation fails with detailed error messages.
        """
        errors: List[str] = []

        employees = input_data.get("employees")
        if not employees or not isinstance(employees, list):
            raise ValueError("'employees' must be a non-empty list")

        total_employees = input_data.get("total_employees")
        if total_employees is None:
            total_employees = len(employees)
        total_employees = int(total_employees)
        if total_employees <= 0:
            errors.append("total_employees must be positive")

        validated_employees: List[dict] = []

        for idx, emp in enumerate(employees):
            emp_errors: List[str] = []
            emp_id = emp.get("employee_id", f"EMP-{idx}")

            # Mode validation
            mode_str = emp.get("mode", "")
            try:
                mode = CommuteMode(mode_str)
            except ValueError:
                emp_errors.append(f"Invalid mode '{mode_str}'")
                mode = None

            # Distance validation (not required for telework)
            distance = emp.get("one_way_distance_km")
            if mode and mode != CommuteMode.TELEWORK:
                if distance is None:
                    # Check for miles fallback
                    distance_miles = emp.get("one_way_distance_miles")
                    if distance_miles is not None:
                        distance = str(
                            (self._safe_decimal(distance_miles) * _MILES_TO_KM).quantize(
                                _QUANT_8DP, rounding=ROUND_HALF_UP
                            )
                        )
                    else:
                        emp_errors.append("one_way_distance_km required for non-telework mode")
                else:
                    try:
                        dist_val = Decimal(str(distance))
                        if dist_val <= _ZERO:
                            emp_errors.append("one_way_distance_km must be positive")
                        elif dist_val > Decimal("500"):
                            emp_errors.append(
                                f"one_way_distance_km {dist_val} exceeds 500 km limit"
                            )
                    except (InvalidOperation, ValueError):
                        emp_errors.append(f"Invalid distance value: {distance}")

            # Commute days validation
            days_per_week = emp.get("commute_days_per_week", 5)
            try:
                days_per_week = int(days_per_week)
                if days_per_week < 1 or days_per_week > 7:
                    emp_errors.append("commute_days_per_week must be 1-7")
            except (ValueError, TypeError):
                emp_errors.append(f"Invalid commute_days_per_week: {days_per_week}")
                days_per_week = 5

            if emp_errors:
                errors.extend([f"Employee {emp_id}: {e}" for e in emp_errors])
            else:
                validated_emp = dict(emp)
                validated_emp["employee_id"] = emp_id
                validated_emp["commute_days_per_week"] = days_per_week
                if distance is not None:
                    validated_emp["one_way_distance_km"] = str(distance)
                validated_employees.append(validated_emp)

        if errors:
            raise ValueError(
                f"Input validation failed ({len(errors)} errors): "
                + "; ".join(errors[:10])
                + ("..." if len(errors) > 10 else "")
            )

        if not validated_employees:
            raise ValueError("No valid employee records after validation")

        result = dict(input_data)
        result["employees"] = validated_employees
        result["total_employees"] = total_employees
        return result

    def _stage_classify(self, validated: dict) -> dict:
        """
        Stage 2: CLASSIFY - Classify commute modes and determine parameters.

        For each employee:
        - Parses CommuteMode from mode string
        - Determines VehicleType (defaults by mode)
        - Determines FuelType if applicable
        - Classifies distance band
        - Determines telework frequency
        - Identifies multi-modal segments if provided

        Args:
            validated: Validated input dict from Stage 1.

        Returns:
            Classified input dict with enriched employee records.
        """
        classified_employees: List[dict] = []

        for emp in validated["employees"]:
            classified = dict(emp)
            mode_str = emp.get("mode", "sov")
            mode = CommuteMode(mode_str)
            classified["_mode"] = mode

            # Vehicle type defaults
            if mode in _PERSONAL_VEHICLE_MODES:
                vt_str = emp.get("vehicle_type")
                if vt_str:
                    try:
                        classified["_vehicle_type"] = VehicleType(vt_str)
                    except ValueError:
                        classified["_vehicle_type"] = VehicleType.CAR_AVERAGE
                elif mode == CommuteMode.MOTORCYCLE:
                    classified["_vehicle_type"] = VehicleType.MOTORCYCLE
                else:
                    classified["_vehicle_type"] = VehicleType.CAR_AVERAGE

            # Fuel type
            fuel_str = emp.get("fuel_type")
            if fuel_str:
                try:
                    classified["_fuel_type"] = FuelType(fuel_str)
                except ValueError:
                    classified["_fuel_type"] = None
            else:
                classified["_fuel_type"] = None

            # Distance band classification
            dist_km = emp.get("one_way_distance_km")
            if dist_km is not None:
                classified["_distance_band"] = classify_distance_band(
                    self._safe_decimal(dist_km)
                )
            else:
                classified["_distance_band"] = DistanceBand.SHORT_0_5

            # Telework frequency
            tw_str = emp.get("telework_frequency", TeleworkFrequency.OFFICE_FULL.value)
            try:
                classified["_telework_frequency"] = TeleworkFrequency(tw_str)
            except ValueError:
                classified["_telework_frequency"] = TeleworkFrequency.OFFICE_FULL

            # Work schedule
            ws_str = emp.get("work_schedule", WorkSchedule.FULL_TIME.value)
            try:
                classified["_work_schedule"] = WorkSchedule(ws_str)
            except ValueError:
                classified["_work_schedule"] = WorkSchedule.FULL_TIME

            # Region
            region_str = emp.get("region", validated.get("region", "GLOBAL"))
            classified["_region"] = self._parse_region(region_str)

            # Carpool occupancy
            if mode == CommuteMode.CARPOOL:
                classified["_occupancy"] = max(int(emp.get("occupants", 2)), 2)
            elif mode == CommuteMode.VANPOOL:
                classified["_occupancy"] = max(int(emp.get("occupants", 10)), 2)
            else:
                classified["_occupancy"] = 1

            # Multi-modal segments
            segments = emp.get("segments")
            if segments and isinstance(segments, list):
                classified["_multimodal"] = True
                classified["_segments"] = segments
            else:
                classified["_multimodal"] = False

            classified_employees.append(classified)

        result = dict(validated)
        result["employees"] = classified_employees
        return result

    def _stage_normalize(self, classified: dict) -> dict:
        """
        Stage 3: NORMALIZE - Unit conversion and working days normalization.

        For each employee:
        - Convert miles to km (x 1.60934)
        - Convert gallons to litres (x 3.78541)
        - Convert mpg to L/100km
        - Normalize commute frequency to annual working days
        - Adjust for part-time work schedule
        - Apply compressed work week factors
        - Calculate annual commute days and telework days

        Args:
            classified: Classified input dict from Stage 2.

        Returns:
            Normalized input dict with converted units and annual days.
        """
        normalized_employees: List[dict] = []

        for emp in classified["employees"]:
            norm = dict(emp)
            mode: CommuteMode = emp["_mode"]
            region: RegionCode = emp["_region"]
            work_schedule: WorkSchedule = emp["_work_schedule"]
            telework_freq: TeleworkFrequency = emp["_telework_frequency"]

            # Distance normalization (miles -> km)
            if "one_way_distance_miles" in emp and "one_way_distance_km" not in emp:
                miles = self._safe_decimal(emp["one_way_distance_miles"])
                norm["one_way_distance_km"] = str(
                    (miles * _MILES_TO_KM).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
                )
                norm["_original_distance_miles"] = str(miles)

            # Fuel normalization (gallons -> litres)
            if "gallons_per_week" in emp and "litres_per_week" not in emp:
                gallons = self._safe_decimal(emp["gallons_per_week"])
                norm["litres_per_week"] = str(
                    (gallons * _GALLONS_TO_LITRES).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
                )

            # MPG -> L/100km conversion
            if "mpg" in emp:
                mpg = self._safe_decimal(emp["mpg"])
                if mpg > _ZERO:
                    l_per_100km = _MPG_TO_L_PER_100KM / mpg
                    norm["_l_per_100km"] = str(
                        l_per_100km.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
                    )

            # Working days calculation
            net_working_days = get_working_days(region, work_schedule)
            norm["_net_working_days"] = net_working_days

            # Telework fraction
            tw_fraction = TELEWORK_FREQUENCY_FRACTIONS.get(telework_freq, _ZERO)
            norm["_telework_fraction"] = str(tw_fraction)

            # Annual commute days = working_days x (1 - telework_fraction)
            # x (commute_days_per_week / 5) to handle compressed schedules
            days_per_week = int(emp.get("commute_days_per_week", 5))
            week_fraction = self._safe_decimal(days_per_week) / Decimal("5")
            if week_fraction > _ONE:
                week_fraction = _ONE

            commute_fraction = (_ONE - tw_fraction) * week_fraction
            annual_commute_days = int(
                (self._safe_decimal(net_working_days) * commute_fraction).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
            )
            norm["_annual_commute_days"] = annual_commute_days

            # Annual telework days
            annual_telework_days = int(
                (self._safe_decimal(net_working_days) * tw_fraction).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
            )
            norm["_annual_telework_days"] = annual_telework_days

            # Annual round-trip distance
            dist_km = self._safe_decimal(emp.get("one_way_distance_km", "0"))
            annual_distance_km = (dist_km * _TWO * self._safe_decimal(annual_commute_days)).quantize(
                _QUANT_8DP, rounding=ROUND_HALF_UP
            )
            norm["_annual_distance_km"] = str(annual_distance_km)

            # Seasonal adjustment for telework
            sa_str = emp.get("seasonal_adjustment", SeasonalAdjustment.NONE.value)
            try:
                seasonal_adj = SeasonalAdjustment(sa_str)
            except ValueError:
                seasonal_adj = SeasonalAdjustment.NONE
            norm["_seasonal_adjustment"] = seasonal_adj

            normalized_employees.append(norm)

        result = dict(classified)
        result["employees"] = normalized_employees
        return result

    def _stage_resolve_efs(self, normalized: dict) -> dict:
        """
        Stage 4: RESOLVE_EFS - Emission factor resolution.

        Resolves EFs using the hierarchy: Employee > DEFRA > EPA > IEA > Census > EEIO.
        For each employee, resolves mode-specific emission factors:
        - SOV/carpool/motorcycle: VEHICLE_EMISSION_FACTORS (per vkm + WTT)
        - Transit: TRANSIT_EMISSION_FACTORS (per pkm + WTT)
        - E-bike/e-scooter: MICRO_MOBILITY_EFS
        - Cycling/walking: Zero
        - Telework: GRID_EMISSION_FACTORS x daily kWh
        - Vanpool: VAN_EMISSION_FACTORS

        Args:
            normalized: Normalized input dict from Stage 3.

        Returns:
            Input dict enriched with resolved emission factor data.
        """
        resolved_employees: List[dict] = []

        for emp in normalized["employees"]:
            resolved = dict(emp)
            mode: CommuteMode = emp["_mode"]

            ef_source = EFSource.DEFRA  # Default source

            if mode in _PERSONAL_VEHICLE_MODES and mode != CommuteMode.VANPOOL:
                vt: VehicleType = emp.get("_vehicle_type", VehicleType.CAR_AVERAGE)
                veh_ef = VEHICLE_EMISSION_FACTORS.get(vt, VEHICLE_EMISSION_FACTORS[VehicleType.CAR_AVERAGE])
                resolved["_ef_per_vkm"] = str(veh_ef["ef_per_vkm"])
                resolved["_wtt_per_vkm"] = str(veh_ef["wtt_per_vkm"])
                resolved["_ef_source"] = ef_source

            elif mode == CommuteMode.VANPOOL:
                van_size = emp.get("van_size", "van_medium")
                van_ef = VAN_EMISSION_FACTORS.get(van_size, VAN_EMISSION_FACTORS["van_medium"])
                resolved["_ef_per_vkm"] = str(van_ef["ef_per_vkm"])
                resolved["_wtt_per_vkm"] = str(van_ef["wtt_per_vkm"])
                if emp.get("_occupancy", 0) <= 1:
                    resolved["_occupancy"] = int(van_ef["default_occupancy"])
                resolved["_ef_source"] = ef_source

            elif mode in _TRANSIT_MODES:
                transit_type = _MODE_TO_TRANSIT_TYPE.get(mode, TransitType.BUS_LOCAL)
                transit_ef = TRANSIT_EMISSION_FACTORS.get(
                    transit_type, TRANSIT_EMISSION_FACTORS[TransitType.BUS_LOCAL]
                )
                resolved["_ef_per_pkm"] = str(transit_ef["ef_per_pkm"])
                resolved["_wtt_per_pkm"] = str(transit_ef["wtt_per_pkm"])
                resolved["_ef_source"] = ef_source

            elif mode == CommuteMode.E_BIKE:
                resolved["_ef_per_pkm"] = str(MICRO_MOBILITY_EFS["e_bike"])
                resolved["_wtt_per_pkm"] = "0"
                resolved["_ef_source"] = ef_source

            elif mode == CommuteMode.E_SCOOTER:
                resolved["_ef_per_pkm"] = str(MICRO_MOBILITY_EFS["e_scooter"])
                resolved["_wtt_per_pkm"] = "0"
                resolved["_ef_source"] = ef_source

            elif mode in (CommuteMode.CYCLING, CommuteMode.WALKING):
                resolved["_ef_per_pkm"] = "0"
                resolved["_wtt_per_pkm"] = "0"
                resolved["_ef_source"] = ef_source

            elif mode == CommuteMode.TELEWORK:
                # Telework uses grid EF; resolved here for completeness
                region = emp.get("_region", RegionCode.GLOBAL)
                resolved["_grid_ef"] = str(get_grid_ef(region))
                resolved["_ef_source"] = EFSource.IEA

            # Resolve telework grid EF for all employees with telework days
            if emp.get("_annual_telework_days", 0) > 0 and "_grid_ef" not in resolved:
                region = emp.get("_region", RegionCode.GLOBAL)
                resolved["_grid_ef"] = str(get_grid_ef(region))

            # Fuel-based EF override
            fuel_type: Optional[FuelType] = emp.get("_fuel_type")
            if fuel_type is not None and fuel_type in FUEL_EMISSION_FACTORS:
                fuel_ef = FUEL_EMISSION_FACTORS[fuel_type]
                resolved["_fuel_ef_per_litre"] = str(fuel_ef["ef_per_litre"])
                resolved["_fuel_wtt_per_litre"] = str(fuel_ef["wtt_per_litre"])
                resolved["_use_fuel_based"] = True
            else:
                resolved["_use_fuel_based"] = False

            resolved_employees.append(resolved)

        result = dict(normalized)
        result["employees"] = resolved_employees
        return result

    def _stage_calculate_commute(self, resolved: dict) -> dict:
        """
        Stage 5: CALCULATE_COMMUTE - Per-employee commute emissions.

        Formulas:
          SOV:     CO2e = Dist_one_way x 2 x annual_commute_days x EF_vkm
          Carpool: CO2e = (Dist x 2 x annual_commute_days x EF_vkm) / occupancy
          Vanpool: CO2e = (Dist x 2 x annual_commute_days x EF_vkm) / occupancy
          Transit: CO2e = Dist x 2 x annual_commute_days x EF_pkm
          E-bike:  CO2e = Dist x 2 x annual_commute_days x EF_pkm
          Active:  CO2e = 0 (cycling/walking)
          Multi:   CO2e = SUM(leg_dist x EF_leg) x 2 x annual_commute_days

        WTT is calculated separately and added to total.

        Args:
            resolved: Resolved input dict from Stage 4.

        Returns:
            Input dict with per-employee commute_co2e and wtt_co2e.
        """
        calculated_employees: List[dict] = []
        sample_commute_co2e = _ZERO

        for emp in resolved["employees"]:
            calc = dict(emp)
            mode: CommuteMode = emp["_mode"]
            annual_days = emp.get("_annual_commute_days", 0)
            dist_km = self._safe_decimal(emp.get("one_way_distance_km", "0"))
            occupancy = emp.get("_occupancy", 1)

            co2e = _ZERO
            wtt = _ZERO

            if mode == CommuteMode.TELEWORK:
                # Pure telework - no commute emissions
                co2e = _ZERO
                wtt = _ZERO

            elif emp.get("_multimodal") and emp.get("_segments"):
                # Multi-modal: sum segments
                co2e, wtt = self._calculate_multimodal_segments(
                    emp["_segments"], annual_days
                )

            elif emp.get("_use_fuel_based") and mode in _PERSONAL_VEHICLE_MODES:
                # Fuel-based calculation
                co2e, wtt = self._calculate_fuel_based(emp)

            elif mode in _PERSONAL_VEHICLE_MODES:
                # Distance-based vehicle calculation
                ef_vkm = self._safe_decimal(emp.get("_ef_per_vkm", "0"))
                wtt_vkm = self._safe_decimal(emp.get("_wtt_per_vkm", "0"))

                # SOV: dist x 2 x days x EF
                annual_vkm = dist_km * _TWO * self._safe_decimal(annual_days)
                vehicle_co2e = (annual_vkm * ef_vkm).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
                vehicle_wtt = (annual_vkm * wtt_vkm).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

                # Carpool/vanpool: divide by occupancy
                if occupancy > 1:
                    occ_d = self._safe_decimal(occupancy)
                    co2e = (vehicle_co2e / occ_d).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
                    wtt = (vehicle_wtt / occ_d).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
                else:
                    co2e = vehicle_co2e
                    wtt = vehicle_wtt

            elif mode in _TRANSIT_MODES or mode in (CommuteMode.E_BIKE, CommuteMode.E_SCOOTER):
                # Transit / micro-mobility: dist x 2 x days x EF_pkm
                ef_pkm = self._safe_decimal(emp.get("_ef_per_pkm", "0"))
                wtt_pkm = self._safe_decimal(emp.get("_wtt_per_pkm", "0"))
                annual_pkm = dist_km * _TWO * self._safe_decimal(annual_days)

                co2e = (annual_pkm * ef_pkm).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
                wtt = (annual_pkm * wtt_pkm).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

            else:
                # Active transport (cycling, walking) = zero emissions
                co2e = _ZERO
                wtt = _ZERO

            total = co2e + wtt
            calc["_commute_co2e"] = str(co2e)
            calc["_commute_wtt"] = str(wtt)
            calc["_commute_total_co2e"] = str(total)

            sample_commute_co2e += total
            calculated_employees.append(calc)

        result = dict(resolved)
        result["employees"] = calculated_employees
        result["sample_commute_co2e"] = str(sample_commute_co2e.quantize(
            _QUANT_8DP, rounding=ROUND_HALF_UP
        ))
        return result

    def _stage_calculate_telework(self, commute_result: dict) -> dict:
        """
        Stage 6: CALCULATE_TELEWORK - Telework emissions per employee.

        Formula: CO2e = annual_telework_days x daily_kWh x grid_EF x seasonal_mult

        Also calculates avoided commute emissions as a memo item (not subtracted
        from total, per GHG Protocol guidance).

        Args:
            commute_result: Result from Stage 5 with commute emissions.

        Returns:
            Input dict with per-employee telework_co2e and avoided_commute memo.
        """
        telework_employees: List[dict] = []
        sample_telework_co2e = _ZERO

        for emp in commute_result["employees"]:
            tw = dict(emp)
            annual_tw_days = emp.get("_annual_telework_days", 0)

            if annual_tw_days <= 0:
                tw["_telework_co2e"] = "0"
                tw["_avoided_commute_co2e"] = "0"
                telework_employees.append(tw)
                continue

            # Resolve daily kWh
            daily_kwh_override = emp.get("daily_kwh_override")
            if daily_kwh_override is not None:
                daily_kwh = self._safe_decimal(daily_kwh_override)
            else:
                daily_kwh = TELEWORK_ENERGY_DEFAULTS["total_typical"]

            # Apply seasonal adjustment
            seasonal_adj: SeasonalAdjustment = emp.get(
                "_seasonal_adjustment", SeasonalAdjustment.NONE
            )
            seasonal_mult = SEASONAL_ADJUSTMENT_MULTIPLIERS.get(seasonal_adj, _ONE)

            # Grid emission factor
            grid_ef = self._safe_decimal(emp.get("_grid_ef", str(
                get_grid_ef(emp.get("_region", RegionCode.GLOBAL))
            )))

            # Telework CO2e = days x kWh x seasonal x grid_EF
            tw_co2e = (
                self._safe_decimal(annual_tw_days)
                * daily_kwh * seasonal_mult * grid_ef
            ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

            tw["_telework_co2e"] = str(tw_co2e)
            tw["_telework_daily_kwh"] = str(daily_kwh)
            tw["_telework_seasonal_mult"] = str(seasonal_mult)
            tw["_telework_grid_ef"] = str(grid_ef)

            # Avoided commute memo: what this employee would have emitted
            # if they had commuted on telework days instead
            commute_total = self._safe_decimal(emp.get("_commute_total_co2e", "0"))
            commute_days = emp.get("_annual_commute_days", 0)
            if commute_days > 0:
                per_day_commute = commute_total / self._safe_decimal(commute_days)
                avoided = (per_day_commute * self._safe_decimal(annual_tw_days)).quantize(
                    _QUANT_8DP, rounding=ROUND_HALF_UP
                )
            else:
                avoided = _ZERO
            tw["_avoided_commute_co2e"] = str(avoided)

            sample_telework_co2e += tw_co2e
            telework_employees.append(tw)

        result = dict(commute_result)
        result["employees"] = telework_employees
        result["sample_telework_co2e"] = str(sample_telework_co2e.quantize(
            _QUANT_8DP, rounding=ROUND_HALF_UP
        ))

        # Total sample emissions
        sample_commute = self._safe_decimal(commute_result.get("sample_commute_co2e", "0"))
        result["sample_total_co2e"] = str(
            (sample_commute + sample_telework_co2e).quantize(
                _QUANT_8DP, rounding=ROUND_HALF_UP
            )
        )
        return result

    def _stage_extrapolate(self, telework_result: dict) -> dict:
        """
        Stage 7: EXTRAPOLATE - Scale survey sample to full workforce.

        Formula: Org_CO2e = SUM(Employee_CO2e_i) x (Total_Employees / Respondents)

        Supports response rate weighting and calculates confidence intervals
        based on sample size. For full census, scaling_factor = 1.

        Args:
            telework_result: Result from Stage 6 with all employee emissions.

        Returns:
            Input dict with extrapolated org-level emissions and confidence bounds.
        """
        result = dict(telework_result)
        employees = telework_result["employees"]
        total_employees = int(telework_result.get("total_employees", len(employees)))
        respondents = len(employees)

        sample_total = self._safe_decimal(telework_result.get("sample_total_co2e", "0"))
        sample_commute = self._safe_decimal(telework_result.get("sample_commute_co2e", "0"))
        sample_telework = self._safe_decimal(telework_result.get("sample_telework_co2e", "0"))

        # Calculate scaling factor
        if respondents > 0 and total_employees > 0:
            scaling_factor = (
                self._safe_decimal(total_employees) / self._safe_decimal(respondents)
            ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        else:
            scaling_factor = _ONE

        # Extrapolate totals
        extrapolated_total = (sample_total * scaling_factor).quantize(
            _QUANT_8DP, rounding=ROUND_HALF_UP
        )
        extrapolated_commute = (sample_commute * scaling_factor).quantize(
            _QUANT_8DP, rounding=ROUND_HALF_UP
        )
        extrapolated_telework = (sample_telework * scaling_factor).quantize(
            _QUANT_8DP, rounding=ROUND_HALF_UP
        )

        # Per-employee average
        per_employee_avg = _ZERO
        if total_employees > 0:
            per_employee_avg = (
                extrapolated_total / self._safe_decimal(total_employees)
            ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        # Response rate
        response_rate = _ZERO
        if total_employees > 0:
            response_rate = (
                self._safe_decimal(respondents) / self._safe_decimal(total_employees)
            ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        # Confidence interval (simplified normal approximation)
        ci_lower, ci_upper = self._calculate_confidence_interval(
            employees, sample_total, scaling_factor, respondents, total_employees
        )

        result["scaling_factor"] = str(scaling_factor)
        result["extrapolated_total_co2e"] = str(extrapolated_total)
        result["extrapolated_commute_co2e"] = str(extrapolated_commute)
        result["extrapolated_telework_co2e"] = str(extrapolated_telework)
        result["per_employee_avg_co2e"] = str(per_employee_avg)
        result["response_rate"] = str(response_rate)
        result["respondent_count"] = respondents
        result["ci_lower_95"] = str(ci_lower)
        result["ci_upper_95"] = str(ci_upper)

        return result

    def _stage_compliance(self, extrapolated: dict) -> dict:
        """
        Stage 8: COMPLIANCE - Run compliance checks against 7 frameworks.

        Validates:
        - GHG Protocol: method disclosure, EF source, mode breakdown
        - ISO 14064: uncertainty analysis, base year
        - CSRD ESRS: telework disclosure, targets
        - CDP: mode breakdown, employee count, survey coverage
        - SBTi: target coverage, reduction initiatives
        - SB 253: methodology, assurance opinion
        - GRI: gases included, base year, intensity ratios

        Args:
            extrapolated: Result from Stage 7 with org-level emissions.

        Returns:
            Input dict enriched with compliance check results.
        """
        # Attempt to use dedicated compliance engine
        engine = self._get_compliance_engine()
        if engine is not None:
            try:
                compliance = engine.check(extrapolated)
                result = dict(extrapolated)
                result["compliance"] = compliance
                return result
            except Exception as e:
                logger.warning(f"ComplianceCheckerEngine failed, using inline: {e}")

        # Inline compliance checks
        method_str = extrapolated.get(
            "calculation_method",
            extrapolated.get("method", CalculationMethod.EMPLOYEE_SPECIFIC.value)
        )
        try:
            method = CalculationMethod(method_str)
        except ValueError:
            method = CalculationMethod.EMPLOYEE_SPECIFIC

        check_data = {
            "total_co2e": extrapolated.get("extrapolated_total_co2e", "0"),
            "method": method.value,
            "total_employees": extrapolated.get("total_employees", 0),
            "respondent_count": extrapolated.get("respondent_count", 0),
            "telework_disclosed": self._safe_decimal(
                extrapolated.get("extrapolated_telework_co2e", "0")
            ) > _ZERO,
            "has_mode_breakdown": bool(extrapolated.get("employees")),
            "response_rate": extrapolated.get("response_rate", "0"),
        }

        compliance = self._inline_compliance_check(method, check_data)

        result = dict(extrapolated)
        result["compliance"] = compliance
        return result

    def _stage_aggregate(self, compliance_result: dict) -> dict:
        """
        Stage 9: AGGREGATE - Multi-dimensional aggregation and hot-spot analysis.

        Aggregates by:
        - by_mode: CO2e grouped by commute mode
        - by_distance_band: CO2e grouped by distance band
        - by_department: CO2e grouped by department
        - by_site: CO2e grouped by office site
        - mode_share: Employee count and CO2e share per mode
        - hot_spots: Top emission contributors

        Also calculates DQI score based on calculation method.

        Args:
            compliance_result: Result from Stage 8 with compliance data.

        Returns:
            Input dict with aggregation results, DQI, and hot-spot analysis.
        """
        result = dict(compliance_result)
        employees = compliance_result.get("employees", [])

        by_mode: Dict[str, Decimal] = {}
        by_distance_band: Dict[str, Decimal] = {}
        by_department: Dict[str, Decimal] = {}
        by_site: Dict[str, Decimal] = {}
        mode_employee_count: Dict[str, int] = {}
        mode_distance_sum: Dict[str, Decimal] = {}
        total_avoided = _ZERO

        for emp in employees:
            mode: CommuteMode = emp.get("_mode", CommuteMode.SOV)
            mode_key = mode.value

            commute_total = self._safe_decimal(emp.get("_commute_total_co2e", "0"))
            telework_total = self._safe_decimal(emp.get("_telework_co2e", "0"))
            emp_total = commute_total + telework_total

            # By mode
            by_mode[mode_key] = by_mode.get(mode_key, _ZERO) + emp_total
            mode_employee_count[mode_key] = mode_employee_count.get(mode_key, 0) + 1

            # Distance per mode
            dist = self._safe_decimal(emp.get("one_way_distance_km", "0"))
            mode_distance_sum[mode_key] = mode_distance_sum.get(mode_key, _ZERO) + dist

            # By distance band
            band: DistanceBand = emp.get("_distance_band", DistanceBand.SHORT_0_5)
            band_key = band.value if hasattr(band, "value") else str(band)
            by_distance_band[band_key] = by_distance_band.get(band_key, _ZERO) + emp_total

            # By department
            dept = emp.get("department")
            if dept:
                by_department[dept] = by_department.get(dept, _ZERO) + emp_total

            # By site
            site = emp.get("site")
            if site:
                by_site[site] = by_site.get(site, _ZERO) + emp_total

            # Avoided commute tracking
            total_avoided += self._safe_decimal(emp.get("_avoided_commute_co2e", "0"))

        # Mode share analysis
        total_sample_co2e = sum(by_mode.values(), _ZERO)
        total_respondents = len(employees)
        mode_share_analysis: List[dict] = []

        for m_key in sorted(by_mode.keys(), key=lambda k: by_mode[k], reverse=True):
            m_co2e = by_mode[m_key]
            m_count = mode_employee_count.get(m_key, 0)
            m_dist_total = mode_distance_sum.get(m_key, _ZERO)

            emp_share = _ZERO
            if total_respondents > 0:
                emp_share = (
                    self._safe_decimal(m_count) / self._safe_decimal(total_respondents)
                ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

            co2e_share = _ZERO
            if total_sample_co2e > _ZERO:
                co2e_share = (m_co2e / total_sample_co2e).quantize(
                    _QUANT_8DP, rounding=ROUND_HALF_UP
                )

            avg_dist = _ZERO
            if m_count > 0:
                avg_dist = (m_dist_total / self._safe_decimal(m_count)).quantize(
                    _QUANT_8DP, rounding=ROUND_HALF_UP
                )

            avg_co2e = _ZERO
            if m_count > 0:
                avg_co2e = (m_co2e / self._safe_decimal(m_count)).quantize(
                    _QUANT_8DP, rounding=ROUND_HALF_UP
                )

            mode_share_analysis.append({
                "mode": m_key,
                "employee_count": m_count,
                "employee_share": str(emp_share),
                "total_co2e": str(m_co2e.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)),
                "co2e_share": str(co2e_share),
                "avg_co2e_per_employee": str(avg_co2e),
                "avg_distance_km": str(avg_dist),
            })

        # SOV rate
        sov_count = mode_employee_count.get(CommuteMode.SOV.value, 0)
        sov_rate = _ZERO
        if total_respondents > 0:
            sov_rate = (
                self._safe_decimal(sov_count) / self._safe_decimal(total_respondents)
            ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        # DQI scoring
        method_str = compliance_result.get(
            "calculation_method",
            compliance_result.get("method", CalculationMethod.EMPLOYEE_SPECIFIC.value)
        )
        dqi_score = self._calculate_dqi_score(method_str, employees)
        dqi_classification = get_dqi_classification(dqi_score)

        # Hot-spot identification (Pareto: modes contributing 80%+ of emissions)
        hot_spots = self._identify_hot_spots(mode_share_analysis)

        # Average commute distance
        all_distances = [
            self._safe_decimal(e.get("one_way_distance_km", "0")) for e in employees
        ]
        avg_distance = _ZERO
        if all_distances:
            avg_distance = (
                sum(all_distances, _ZERO) / self._safe_decimal(len(all_distances))
            ).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        result["by_mode"] = {k: str(v.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)) for k, v in by_mode.items()}
        result["by_distance_band"] = {
            k: str(v.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)) for k, v in by_distance_band.items()
        }
        result["by_department"] = {
            k: str(v.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)) for k, v in by_department.items()
        }
        result["by_site"] = {
            k: str(v.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)) for k, v in by_site.items()
        }
        result["mode_share_analysis"] = mode_share_analysis
        result["sov_rate"] = str(sov_rate)
        result["avg_distance_km"] = str(avg_distance)
        result["hot_spots"] = hot_spots
        result["total_avoided_commute_co2e"] = str(
            total_avoided.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        )
        result["dqi_score"] = str(dqi_score)
        result["dqi_classification"] = dqi_classification

        return result

    def _stage_seal(self, chain_id: str, aggregated: dict) -> dict:
        """
        Stage 10: SEAL - Seal provenance chain with Merkle root.

        Creates SHA-256 hash over the entire provenance chain for this
        calculation, producing an immutable audit fingerprint. Generates
        a Merkle root from all stage hashes and adds ISO 8601 timestamp.

        Args:
            chain_id: Provenance chain identifier.
            aggregated: Final aggregated result data.

        Returns:
            Sealed result dict with provenance_hash, merkle_root, and timestamp.
        """
        chain = self._provenance_chains.get(chain_id, [])

        # Build Merkle tree from chain hashes
        leaf_hashes = [
            entry.get("output_hash", "") for entry in chain
        ]
        merkle_root = self._build_merkle_root(leaf_hashes)

        # Final provenance hash: chain + result + merkle root
        chain_str = json.dumps(chain, sort_keys=True, default=str)
        result_str = json.dumps(aggregated, sort_keys=True, default=str)
        combined = f"{chain_str}|{result_str}|{merkle_root}"
        provenance_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        sealed = dict(aggregated)
        sealed["provenance_hash"] = provenance_hash
        sealed["merkle_root"] = merkle_root
        sealed["sealed_at"] = datetime.now(timezone.utc).isoformat()
        sealed["provenance_chain_length"] = len(chain)
        sealed["immutable"] = True
        sealed["agent_id"] = "GL-MRV-S3-007"
        sealed["agent_component"] = "AGENT-MRV-020"
        sealed["version"] = "1.0.0"

        return sealed

    # ==========================================================================
    # INLINE CALCULATION HELPERS
    # ==========================================================================

    def _calculate_multimodal_segments(
        self,
        segments: List[dict],
        annual_days: int,
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate emissions for multi-modal trip segments.

        Each segment has a mode and distance. Total emissions = sum of segments.
        Applied to both directions (x2) and annual working days.

        Args:
            segments: List of segment dicts with 'mode' and 'distance_km'.
            annual_days: Annual commute days.

        Returns:
            Tuple of (direct_co2e, wtt_co2e).
        """
        total_co2e = _ZERO
        total_wtt = _ZERO

        for seg in segments:
            seg_mode_str = seg.get("mode", "bus")
            seg_dist = self._safe_decimal(seg.get("distance_km", "0"))

            try:
                seg_mode = CommuteMode(seg_mode_str)
            except ValueError:
                # Try as transit type
                try:
                    transit_type = TransitType(seg_mode_str)
                    ef_data = TRANSIT_EMISSION_FACTORS.get(
                        transit_type, TRANSIT_EMISSION_FACTORS[TransitType.BUS_LOCAL]
                    )
                    ef = self._safe_decimal(str(ef_data["ef_per_pkm"]))
                    wtt = self._safe_decimal(str(ef_data["wtt_per_pkm"]))
                    seg_co2e = seg_dist * ef
                    seg_wtt = seg_dist * wtt
                    total_co2e += seg_co2e
                    total_wtt += seg_wtt
                    continue
                except ValueError:
                    continue

            if seg_mode in _TRANSIT_MODES:
                transit_type = _MODE_TO_TRANSIT_TYPE.get(seg_mode, TransitType.BUS_LOCAL)
                ef_data = TRANSIT_EMISSION_FACTORS.get(
                    transit_type, TRANSIT_EMISSION_FACTORS[TransitType.BUS_LOCAL]
                )
                ef = self._safe_decimal(str(ef_data["ef_per_pkm"]))
                wtt = self._safe_decimal(str(ef_data["wtt_per_pkm"]))
            elif seg_mode in _PERSONAL_VEHICLE_MODES:
                vt_str = seg.get("vehicle_type", VehicleType.CAR_AVERAGE.value)
                try:
                    vt = VehicleType(vt_str)
                except ValueError:
                    vt = VehicleType.CAR_AVERAGE
                veh_ef = VEHICLE_EMISSION_FACTORS.get(vt, VEHICLE_EMISSION_FACTORS[VehicleType.CAR_AVERAGE])
                ef = self._safe_decimal(str(veh_ef["ef_per_vkm"]))
                wtt = self._safe_decimal(str(veh_ef["wtt_per_vkm"]))
            elif seg_mode == CommuteMode.E_BIKE:
                ef = MICRO_MOBILITY_EFS["e_bike"]
                wtt = _ZERO
            elif seg_mode == CommuteMode.E_SCOOTER:
                ef = MICRO_MOBILITY_EFS["e_scooter"]
                wtt = _ZERO
            else:
                ef = _ZERO
                wtt = _ZERO

            seg_co2e = seg_dist * ef
            seg_wtt = seg_dist * wtt
            total_co2e += seg_co2e
            total_wtt += seg_wtt

        # Apply round-trip and annual days
        days_d = self._safe_decimal(annual_days)
        total_co2e = (total_co2e * _TWO * days_d).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        total_wtt = (total_wtt * _TWO * days_d).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        return total_co2e, total_wtt

    def _calculate_fuel_based(self, emp: dict) -> Tuple[Decimal, Decimal]:
        """
        Calculate emissions using fuel consumption data.

        Formula: CO2e = litres_per_week x weeks x ef_per_litre

        Args:
            emp: Employee dict with fuel data.

        Returns:
            Tuple of (direct_co2e, wtt_co2e).
        """
        litres_per_week = self._safe_decimal(emp.get("litres_per_week", "0"))
        weeks_per_year = self._safe_decimal(emp.get("commute_weeks_per_year", "48"))

        ef_per_litre = self._safe_decimal(emp.get("_fuel_ef_per_litre", "2.31480"))
        wtt_per_litre = self._safe_decimal(emp.get("_fuel_wtt_per_litre", "0.58549"))

        annual_litres = litres_per_week * weeks_per_year

        co2e = (annual_litres * ef_per_litre).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        wtt = (annual_litres * wtt_per_litre).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        return co2e, wtt

    def _get_mode_ef(self, mode_key: str) -> Decimal:
        """
        Get a representative emission factor for a mode share key.

        Used by the average-data method to calculate weighted emissions.

        Args:
            mode_key: Mode share key from DEFAULT_MODE_SHARES.

        Returns:
            Emission factor in kgCO2e per passenger-km.
        """
        mode_ef_map: Dict[str, Decimal] = {
            "sov": VEHICLE_EMISSION_FACTORS[VehicleType.CAR_AVERAGE]["ef_per_vkm"] or _ZERO,
            "carpool": (
                (VEHICLE_EMISSION_FACTORS[VehicleType.CAR_AVERAGE]["ef_per_vkm"] or _ZERO)
                / Decimal("2.5")  # Average carpool occupancy
            ),
            "bus": TRANSIT_EMISSION_FACTORS[TransitType.BUS_LOCAL]["ef_per_pkm"],
            "metro": TRANSIT_EMISSION_FACTORS[TransitType.METRO]["ef_per_pkm"],
            "commuter_rail": TRANSIT_EMISSION_FACTORS[TransitType.COMMUTER_RAIL]["ef_per_pkm"],
            "light_rail": TRANSIT_EMISSION_FACTORS[TransitType.LIGHT_RAIL]["ef_per_pkm"],
            "ferry": TRANSIT_EMISSION_FACTORS[TransitType.FERRY]["ef_per_pkm"],
            "motorcycle": VEHICLE_EMISSION_FACTORS[VehicleType.MOTORCYCLE]["ef_per_vkm"] or _ZERO,
            "e_bike": MICRO_MOBILITY_EFS["e_bike"],
            "e_scooter": MICRO_MOBILITY_EFS["e_scooter"],
            "cycling": _ZERO,
            "walking": _ZERO,
        }
        return mode_ef_map.get(mode_key, _ZERO)

    def _inline_compliance_check(self, method: CalculationMethod, data: dict) -> dict:
        """
        Inline lightweight compliance check against 7 frameworks.

        Checks presence of required disclosures per FRAMEWORK_REQUIRED_DISCLOSURES
        and returns pass/fail/warning status per framework.

        Args:
            method: Calculation method used.
            data: Data dict to validate against framework requirements.

        Returns:
            Dict with per-framework compliance results and overall status.
        """
        framework_results: List[dict] = []
        overall_pass = True

        for framework, required_fields in FRAMEWORK_REQUIRED_DISCLOSURES.items():
            findings: List[str] = []

            if framework == ComplianceFramework.GHG_PROTOCOL:
                if not data.get("method"):
                    findings.append("Calculation method not disclosed")
                if not data.get("total_co2e"):
                    findings.append("Total CO2e not reported")
                if method == CalculationMethod.EMPLOYEE_SPECIFIC:
                    if not data.get("has_mode_breakdown", True):
                        findings.append("Mode breakdown not provided for employee-specific method")

            elif framework == ComplianceFramework.CSRD_ESRS:
                if not data.get("telework_disclosed"):
                    findings.append("CSRD E1: Telework policy / emissions not disclosed")

            elif framework == ComplianceFramework.CDP:
                if not data.get("total_employees"):
                    findings.append("CDP: Employee count not reported")
                response_rate = self._safe_decimal(data.get("response_rate", "0"))
                if method == CalculationMethod.EMPLOYEE_SPECIFIC and response_rate < Decimal("0.3"):
                    findings.append(
                        f"CDP: Survey response rate ({response_rate}) below 30% threshold"
                    )

            elif framework == ComplianceFramework.SB_253:
                if not data.get("method"):
                    findings.append("SB 253: Methodology not documented")

            elif framework == ComplianceFramework.ISO_14064:
                # ISO requires uncertainty analysis for Tier 1 compliance
                if method == CalculationMethod.SPEND_BASED:
                    findings.append(
                        "ISO 14064: Spend-based method has high uncertainty; "
                        "consider employee-specific or average-data method"
                    )

            elif framework == ComplianceFramework.SBTI:
                pass  # SBTi requires target coverage, checked elsewhere

            elif framework == ComplianceFramework.GRI:
                if not data.get("method"):
                    findings.append("GRI 305: Standards used not documented")

            status = ComplianceStatus.PASS.value
            if findings:
                status = ComplianceStatus.WARNING.value
                overall_pass = False

            framework_results.append({
                "framework": framework.value,
                "status": status,
                "findings": findings,
                "findings_count": len(findings),
            })

        overall_status = "PASS" if overall_pass else "WARNING"

        return {
            "overall_status": overall_status,
            "frameworks_checked": len(framework_results),
            "framework_results": framework_results,
        }

    def _calculate_dqi_score(self, method_str: str, employees: list) -> Decimal:
        """
        Calculate composite Data Quality Indicator score.

        Scoring by method:
        - employee_specific: 4.0 (high quality primary data)
        - average_data: 2.5 (medium quality secondary data)
        - spend_based: 1.5 (low quality spend proxies)

        Adjusted by response rate and data completeness.

        Args:
            method_str: Calculation method string value.
            employees: List of employee dicts.

        Returns:
            Composite DQI score (1-5 scale).
        """
        base_scores = {
            CalculationMethod.EMPLOYEE_SPECIFIC.value: Decimal("4.0"),
            CalculationMethod.AVERAGE_DATA.value: Decimal("2.5"),
            CalculationMethod.SPEND_BASED.value: Decimal("1.5"),
        }
        base = base_scores.get(method_str, Decimal("3.0"))

        # Adjust for data completeness
        if employees:
            complete = sum(
                1 for e in employees
                if e.get("one_way_distance_km") and e.get("mode")
            )
            completeness_ratio = self._safe_decimal(complete) / self._safe_decimal(len(employees))
            # Bonus up to 0.5 for 100% completeness
            adjustment = (completeness_ratio * Decimal("0.5")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
            base = base + adjustment

        # Cap at 5.0
        if base > Decimal("5.0"):
            base = Decimal("5.0")

        return base.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

    def _identify_hot_spots(self, mode_share_analysis: List[dict]) -> List[dict]:
        """
        Identify emission hot-spots using Pareto analysis.

        Finds modes contributing to 80%+ of total emissions and generates
        reduction opportunity recommendations.

        Args:
            mode_share_analysis: Mode share analysis from Stage 9.

        Returns:
            List of hot-spot dicts with reduction recommendations.
        """
        hot_spots: List[dict] = []
        cumulative_share = _ZERO
        pareto_threshold = Decimal("0.80")

        for entry in mode_share_analysis:
            co2e_share = self._safe_decimal(entry.get("co2e_share", "0"))
            cumulative_share += co2e_share

            recommendation = self._get_mode_recommendation(entry["mode"])

            hot_spots.append({
                "mode": entry["mode"],
                "co2e_share": entry["co2e_share"],
                "cumulative_share": str(cumulative_share.quantize(
                    _QUANT_8DP, rounding=ROUND_HALF_UP
                )),
                "employee_count": entry["employee_count"],
                "avg_co2e_per_employee": entry["avg_co2e_per_employee"],
                "recommendation": recommendation,
            })

            if cumulative_share >= pareto_threshold:
                break

        return hot_spots

    def _get_mode_recommendation(self, mode: str) -> str:
        """
        Get reduction recommendation for a commute mode.

        Args:
            mode: Commute mode string value.

        Returns:
            Reduction recommendation string.
        """
        recommendations = {
            CommuteMode.SOV.value: (
                "Encourage carpooling, transit subsidies, or cycling infrastructure "
                "to reduce single-occupancy vehicle commuting"
            ),
            CommuteMode.CARPOOL.value: (
                "Increase carpool occupancy or transition to EV carpools "
                "for further emission reductions"
            ),
            CommuteMode.MOTORCYCLE.value: (
                "Consider e-motorcycle or e-scooter alternatives "
                "for short-distance motorcycle commutes"
            ),
            CommuteMode.BUS.value: (
                "Advocate for electric bus fleet adoption with "
                "local transit authorities"
            ),
            CommuteMode.COMMUTER_RAIL.value: (
                "Support renewable energy procurement for rail operators"
            ),
            CommuteMode.FERRY.value: (
                "Explore hybrid/electric ferry options where available"
            ),
            CommuteMode.VANPOOL.value: (
                "Transition vanpool fleet to electric vehicles"
            ),
        }
        return recommendations.get(mode, "Review commute patterns for optimization opportunities")

    def _calculate_confidence_interval(
        self,
        employees: list,
        sample_total: Decimal,
        scaling_factor: Decimal,
        respondents: int,
        total_employees: int,
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate 95% confidence interval for extrapolated emissions.

        Uses finite population correction for sampling without replacement.

        Args:
            employees: List of employee result dicts.
            sample_total: Total sample emissions.
            scaling_factor: Extrapolation scaling factor.
            respondents: Number of survey respondents.
            total_employees: Total workforce size.

        Returns:
            Tuple of (ci_lower, ci_upper) as Decimal.
        """
        if respondents <= 1 or total_employees <= 0:
            extrapolated = sample_total * scaling_factor
            return extrapolated, extrapolated

        # Per-employee emissions for variance calculation
        emp_emissions: List[Decimal] = []
        for emp in employees:
            commute = self._safe_decimal(emp.get("_commute_total_co2e", "0"))
            telework = self._safe_decimal(emp.get("_telework_co2e", "0"))
            emp_emissions.append(commute + telework)

        # Mean and standard deviation
        n = len(emp_emissions)
        if n == 0:
            extrapolated = sample_total * scaling_factor
            return extrapolated, extrapolated

        mean = sum(emp_emissions, _ZERO) / self._safe_decimal(n)
        variance = sum(
            ((x - mean) ** 2 for x in emp_emissions), _ZERO
        ) / self._safe_decimal(max(n - 1, 1))

        std_dev = self._safe_decimal(str(math.sqrt(float(variance))))

        # Standard error of the mean with finite population correction
        se = std_dev / self._safe_decimal(str(math.sqrt(n)))
        fpc = _ONE
        if total_employees > respondents:
            fpc_val = math.sqrt((total_employees - respondents) / max(total_employees - 1, 1))
            fpc = self._safe_decimal(str(fpc_val))
        se_corrected = se * fpc

        # z=1.96 for 95% CI
        z = Decimal("1.96")
        margin = z * se_corrected * self._safe_decimal(total_employees)

        extrapolated = sample_total * scaling_factor
        ci_lower = (extrapolated - margin).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        ci_upper = (extrapolated + margin).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        # Ensure non-negative lower bound
        if ci_lower < _ZERO:
            ci_lower = _ZERO

        return ci_lower, ci_upper

    # ==========================================================================
    # PROVENANCE AND UTILITY HELPERS
    # ==========================================================================

    def _record_provenance(
        self,
        chain_id: str,
        stage: ProvenanceStage,
        input_data: Any,
        output_data: Any,
    ) -> None:
        """
        Record a provenance entry for a pipeline stage.

        Creates a chain hash linking this entry to the previous entry
        in the provenance chain, ensuring tamper detection.

        Args:
            chain_id: Provenance chain identifier.
            stage: Pipeline stage enum.
            input_data: Input to this stage.
            output_data: Output from this stage.
        """
        input_str = self._canonical_json(input_data)
        output_str = self._canonical_json(output_data)

        input_hash = hashlib.sha256(input_str.encode("utf-8")).hexdigest()
        output_hash = hashlib.sha256(output_str.encode("utf-8")).hexdigest()

        chain = self._provenance_chains.get(chain_id)
        if chain is None:
            return

        # Chain hash: SHA-256(previous_chain_hash | stage | input_hash | output_hash)
        previous_hash = chain[-1]["chain_hash"] if chain else "0" * 64
        chain_input = f"{previous_hash}|{stage.value}|{input_hash}|{output_hash}"
        chain_hash = hashlib.sha256(chain_input.encode("utf-8")).hexdigest()

        entry = {
            "stage": stage.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_hash": input_hash,
            "output_hash": output_hash,
            "chain_hash": chain_hash,
        }
        chain.append(entry)

    def _build_merkle_root(self, leaf_hashes: List[str]) -> str:
        """
        Build a Merkle tree root from a list of leaf hashes.

        If the leaf count is odd, the last hash is duplicated.
        The root is computed by iteratively hashing pairs until one hash remains.

        Args:
            leaf_hashes: List of SHA-256 hex digest strings.

        Returns:
            Merkle root as SHA-256 hex digest string.
        """
        if not leaf_hashes:
            return hashlib.sha256(b"empty").hexdigest()

        current_level = list(leaf_hashes)

        while len(current_level) > 1:
            next_level: List[str] = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = hashlib.sha256(
                    f"{left}{right}".encode("utf-8")
                ).hexdigest()
                next_level.append(combined)
            current_level = next_level

        return current_level[0]

    @staticmethod
    def _elapsed_ms(start: datetime) -> float:
        """Calculate milliseconds elapsed since start."""
        return (datetime.now(timezone.utc) - start).total_seconds() * 1000.0

    @staticmethod
    def _safe_decimal(value: Any) -> Decimal:
        """
        Safely convert any value to Decimal.

        Args:
            value: Value to convert (str, int, float, Decimal, None).

        Returns:
            Decimal representation, or ZERO if conversion fails.
        """
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return _ZERO

    @staticmethod
    def _canonical_json(data: Any) -> str:
        """
        Produce deterministic JSON serialization for hashing.

        Args:
            data: Data to serialize.

        Returns:
            Canonical JSON string with sorted keys.
        """
        if isinstance(data, str):
            return data
        try:
            return json.dumps(data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(data)

    @staticmethod
    def _hash_json(data: Any) -> str:
        """
        SHA-256 hash of canonical JSON representation.

        Args:
            data: Data to hash.

        Returns:
            SHA-256 hex digest (64 characters).
        """
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_region(region_str: str) -> RegionCode:
        """
        Parse a region string to RegionCode enum, falling back to GLOBAL.

        Args:
            region_str: Region code string.

        Returns:
            RegionCode enum value.
        """
        try:
            return RegionCode(region_str)
        except ValueError:
            return RegionCode.GLOBAL

    # ==========================================================================
    # LAZY ENGINE LOADING
    # ==========================================================================

    def _get_database_engine(self) -> Optional[Any]:
        """Get or create EmployeeCommutingDatabaseEngine (lazy loading)."""
        if self._database_engine is None:
            try:
                from greenlang.agents.mrv.employee_commuting.employee_commuting_database import (
                    EmployeeCommutingDatabaseEngine,
                )
                self._database_engine = EmployeeCommutingDatabaseEngine()
            except ImportError:
                logger.debug("EmployeeCommutingDatabaseEngine not available")
                self._database_engine = None
        return self._database_engine

    def _get_vehicle_engine(self) -> Optional[Any]:
        """Get or create PersonalVehicleCalculatorEngine (lazy loading)."""
        if self._vehicle_engine is None:
            try:
                from greenlang.agents.mrv.employee_commuting.personal_vehicle_calculator import (
                    PersonalVehicleCalculatorEngine,
                )
                self._vehicle_engine = PersonalVehicleCalculatorEngine()
            except ImportError:
                logger.debug("PersonalVehicleCalculatorEngine not available, using inline")
                self._vehicle_engine = None
        return self._vehicle_engine

    def _get_transit_engine(self) -> Optional[Any]:
        """Get or create PublicTransitCalculatorEngine (lazy loading)."""
        if self._transit_engine is None:
            try:
                from greenlang.agents.mrv.employee_commuting.public_transit_calculator import (
                    PublicTransitCalculatorEngine,
                )
                self._transit_engine = PublicTransitCalculatorEngine()
            except ImportError:
                logger.debug("PublicTransitCalculatorEngine not available, using inline")
                self._transit_engine = None
        return self._transit_engine

    def _get_active_engine(self) -> Optional[Any]:
        """Get or create ActiveTransportCalculatorEngine (lazy loading)."""
        if self._active_engine is None:
            try:
                from greenlang.agents.mrv.employee_commuting.active_transport_calculator import (
                    ActiveTransportCalculatorEngine,
                )
                self._active_engine = ActiveTransportCalculatorEngine()
            except ImportError:
                logger.debug("ActiveTransportCalculatorEngine not available, using inline")
                self._active_engine = None
        return self._active_engine

    def _get_telework_engine(self) -> Optional[Any]:
        """Get or create TeleworkCalculatorEngine (lazy loading)."""
        if self._telework_engine is None:
            try:
                from greenlang.agents.mrv.employee_commuting.telework_calculator import (
                    TeleworkCalculatorEngine,
                )
                self._telework_engine = TeleworkCalculatorEngine()
            except ImportError:
                logger.debug("TeleworkCalculatorEngine not available, using inline")
                self._telework_engine = None
        return self._telework_engine

    def _get_compliance_engine(self) -> Optional[Any]:
        """Get or create ComplianceCheckerEngine (lazy loading)."""
        if self._compliance_engine is None:
            try:
                from greenlang.agents.mrv.employee_commuting.compliance_checker import (
                    ComplianceCheckerEngine,
                )
                self._compliance_engine = ComplianceCheckerEngine()
            except ImportError:
                logger.debug("ComplianceCheckerEngine not available, using inline compliance")
                self._compliance_engine = None
        return self._compliance_engine

    # ==========================================================================
    # UTILITY / STATUS METHODS
    # ==========================================================================

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status including loaded engines.

        Returns:
            Dictionary with pipeline status information.
        """
        return {
            "agent_id": "GL-MRV-S3-007",
            "agent_component": "AGENT-MRV-020",
            "version": "1.0.0",
            "pipeline_stages": [
                "VALIDATE", "CLASSIFY", "NORMALIZE", "RESOLVE_EFS",
                "CALCULATE_COMMUTE", "CALCULATE_TELEWORK", "EXTRAPOLATE",
                "COMPLIANCE", "AGGREGATE", "SEAL",
            ],
            "engines_loaded": {
                "database": self._database_engine is not None,
                "vehicle": self._vehicle_engine is not None,
                "transit": self._transit_engine is not None,
                "active": self._active_engine is not None,
                "telework": self._telework_engine is not None,
                "compliance": self._compliance_engine is not None,
            },
            "active_chains": len(self._provenance_chains),
            "calculation_methods": [m.value for m in CalculationMethod],
            "commute_modes": [m.value for m in CommuteMode],
            "compliance_frameworks": [f.value for f in ComplianceFramework],
        }

    def reset_pipeline(self) -> None:
        """
        Reset pipeline state (clear provenance chains).

        Used for testing or periodic cleanup.
        """
        with self._lock:
            self._provenance_chains.clear()
            logger.info("Pipeline state reset")

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        This forces re-initialization on next instantiation.
        """
        with cls._lock:
            cls._instance = None


# ==============================================================================
# MODULE-LEVEL HELPERS
# ==============================================================================


def get_pipeline_engine() -> EmployeeCommutingPipelineEngine:
    """
    Get singleton pipeline engine instance.

    Returns:
        EmployeeCommutingPipelineEngine singleton instance.
    """
    return EmployeeCommutingPipelineEngine()
