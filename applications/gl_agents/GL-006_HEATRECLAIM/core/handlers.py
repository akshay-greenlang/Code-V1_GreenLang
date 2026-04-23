"""
GL-006 HEATRECLAIM - Request Handlers

Handlers for processing optimization requests, stream data,
and integration with external systems.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
import logging

from .config import HeatReclaimConfig, StreamType, Phase
from .schemas import (
    HeatStream,
    UtilityCost,
    OptimizationRequest,
    OptimizationResult,
    HENDesign,
    PinchAnalysisResult,
    APIResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]


class StreamDataHandler:
    """
    Handler for processing and validating heat stream data.

    Ensures all stream data meets requirements for deterministic
    calculations including unit normalization and bound checking.
    """

    def __init__(self, config: Optional[HeatReclaimConfig] = None) -> None:
        self.config = config or HeatReclaimConfig()

    def validate_streams(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
    ) -> ValidationResult:
        """
        Validate stream data for optimization.

        Args:
            hot_streams: Hot process streams
            cold_streams: Cold process streams

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

        # Check minimum requirements
        if not hot_streams:
            errors.append("At least one hot stream is required")
        if not cold_streams:
            errors.append("At least one cold stream is required")

        # Validate each hot stream
        for stream in hot_streams:
            stream_errors = self._validate_stream(stream, "hot")
            errors.extend(stream_errors)

            # Check temperature direction
            if stream.T_supply_C <= stream.T_target_C:
                errors.append(
                    f"Hot stream {stream.stream_id}: supply temp "
                    f"({stream.T_supply_C}°C) must be > target temp "
                    f"({stream.T_target_C}°C)"
                )

        # Validate each cold stream
        for stream in cold_streams:
            stream_errors = self._validate_stream(stream, "cold")
            errors.extend(stream_errors)

            # Check temperature direction
            if stream.T_supply_C >= stream.T_target_C:
                errors.append(
                    f"Cold stream {stream.stream_id}: supply temp "
                    f"({stream.T_supply_C}°C) must be < target temp "
                    f"({stream.T_target_C}°C)"
                )

        # Check for duplicate stream IDs
        all_ids = [s.stream_id for s in hot_streams + cold_streams]
        if len(all_ids) != len(set(all_ids)):
            errors.append("Duplicate stream IDs detected")

        # Check for potential heat recovery
        if hot_streams and cold_streams:
            max_hot_T = max(s.T_supply_C for s in hot_streams)
            min_cold_T = min(s.T_supply_C for s in cold_streams)
            dt_min = self.config.delta_t_min_C

            if max_hot_T - dt_min < min_cold_T:
                warnings.append(
                    "Limited heat recovery potential: hot stream temperatures "
                    "are close to cold stream temperatures"
                )

        # Check energy balance feasibility
        total_hot_duty = sum(s.duty_kW for s in hot_streams)
        total_cold_duty = sum(s.duty_kW for s in cold_streams)

        if total_hot_duty > 0 and total_cold_duty > 0:
            ratio = total_hot_duty / total_cold_duty
            if ratio < 0.1 or ratio > 10:
                warnings.append(
                    f"Large imbalance between hot ({total_hot_duty:.1f} kW) "
                    f"and cold ({total_cold_duty:.1f} kW) duties"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_stream(
        self,
        stream: HeatStream,
        stream_type: str,
    ) -> List[str]:
        """Validate individual stream."""
        errors = []

        # Check required fields
        if not stream.stream_id:
            errors.append(f"{stream_type.capitalize()} stream missing ID")
            return errors

        prefix = f"{stream_type.capitalize()} stream {stream.stream_id}"

        # Check flow rate
        if stream.m_dot_kg_s <= 0:
            errors.append(f"{prefix}: flow rate must be positive")

        # Check Cp
        if stream.Cp_kJ_kgK <= 0:
            errors.append(f"{prefix}: Cp must be positive")

        # Check temperature range
        if stream.T_supply_C < -200 or stream.T_supply_C > 1500:
            errors.append(
                f"{prefix}: supply temperature {stream.T_supply_C}°C "
                "outside valid range (-200 to 1500°C)"
            )

        if stream.T_target_C < -200 or stream.T_target_C > 1500:
            errors.append(
                f"{prefix}: target temperature {stream.T_target_C}°C "
                "outside valid range (-200 to 1500°C)"
            )

        # Check fouling factor
        if stream.fouling_factor_m2K_W < 0:
            errors.append(f"{prefix}: fouling factor cannot be negative")

        return errors

    def normalize_units(
        self,
        streams: List[HeatStream],
    ) -> List[HeatStream]:
        """
        Normalize stream data to standard units.

        Standard units:
        - Temperature: °C
        - Flow rate: kg/s
        - Cp: kJ/(kg·K)
        - Pressure: kPa

        Args:
            streams: Input streams (may have various units)

        Returns:
            Streams with normalized units
        """
        normalized = []
        for stream in streams:
            # Streams should already be in standard units
            # This is a placeholder for unit conversion logic
            normalized.append(stream)

        return normalized

    def parse_from_dict(
        self,
        data: Dict[str, Any],
    ) -> HeatStream:
        """
        Parse stream from dictionary format.

        Args:
            data: Dictionary with stream properties

        Returns:
            HeatStream object
        """
        stream_type_str = data.get("stream_type", "hot")
        if stream_type_str.lower() in ["hot", "utility_hot"]:
            stream_type = StreamType.HOT
        else:
            stream_type = StreamType.COLD

        phase_str = data.get("phase", "liquid")
        phase = Phase.LIQUID
        if phase_str.lower() == "gas":
            phase = Phase.GAS
        elif phase_str.lower() == "two_phase":
            phase = Phase.TWO_PHASE

        return HeatStream(
            stream_id=data.get("stream_id", ""),
            stream_name=data.get("stream_name", ""),
            stream_type=stream_type,
            fluid_name=data.get("fluid_name", "Unknown"),
            phase=phase,
            T_supply_C=float(data.get("T_supply_C", 0)),
            T_target_C=float(data.get("T_target_C", 0)),
            m_dot_kg_s=float(data.get("m_dot_kg_s", 0)),
            Cp_kJ_kgK=float(data.get("Cp_kJ_kgK", 4.186)),
            pressure_kPa=float(data.get("pressure_kPa", 101.325)),
            fouling_factor_m2K_W=float(data.get("fouling_factor_m2K_W", 0.0001)),
            availability=float(data.get("availability", 1.0)),
            min_approach_C=float(data.get("min_approach_C", 10.0)),
            source_system=data.get("source_system", "api"),
            data_quality=data.get("data_quality", "good"),
        )


class RequestHandler:
    """
    Handler for processing optimization requests.

    Validates requests, prepares data, and coordinates
    with the orchestrator.
    """

    def __init__(self, config: Optional[HeatReclaimConfig] = None) -> None:
        self.config = config or HeatReclaimConfig()
        self.stream_handler = StreamDataHandler(config)

    def build_request(
        self,
        hot_streams_data: List[Dict[str, Any]],
        cold_streams_data: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None,
    ) -> OptimizationRequest:
        """
        Build optimization request from raw data.

        Args:
            hot_streams_data: List of hot stream dictionaries
            cold_streams_data: List of cold stream dictionaries
            options: Optimization options

        Returns:
            OptimizationRequest ready for optimization
        """
        options = options or {}

        # Parse streams
        hot_streams = [
            self.stream_handler.parse_from_dict(s)
            for s in hot_streams_data
        ]
        cold_streams = [
            self.stream_handler.parse_from_dict(s)
            for s in cold_streams_data
        ]

        # Validate
        validation = self.stream_handler.validate_streams(
            hot_streams, cold_streams
        )

        if not validation.is_valid:
            raise ValueError(f"Invalid stream data: {validation.errors}")

        # Build request
        return OptimizationRequest(
            hot_streams=hot_streams,
            cold_streams=cold_streams,
            delta_t_min_C=options.get("delta_t_min_C", self.config.delta_t_min_C),
            include_exergy_analysis=options.get("include_exergy", True),
            include_uncertainty=options.get("include_uncertainty", False),
            generate_pareto=options.get("generate_pareto", False),
            n_pareto_points=options.get("n_pareto_points", 20),
            max_time_seconds=options.get("max_time", 300.0),
            requested_by=options.get("user", "api"),
        )

    def format_response(
        self,
        result: OptimizationResult,
        include_details: bool = True,
    ) -> APIResponse:
        """
        Format optimization result as API response.

        Args:
            result: Optimization result
            include_details: Include detailed breakdown

        Returns:
            APIResponse with formatted data
        """
        data = {
            "request_id": result.request_id,
            "status": result.status.value,
            "summary": {
                "pinch_temperature_C": result.pinch_analysis.pinch_temperature_C,
                "minimum_hot_utility_kW": result.pinch_analysis.minimum_hot_utility_kW,
                "minimum_cold_utility_kW": result.pinch_analysis.minimum_cold_utility_kW,
                "maximum_heat_recovery_kW": result.pinch_analysis.maximum_heat_recovery_kW,
                "exchanger_count": result.recommended_design.exchanger_count,
                "total_heat_recovered_kW": result.recommended_design.total_heat_recovered_kW,
            },
            "optimization_time_seconds": result.optimization_time_seconds,
        }

        if include_details:
            # Add design details
            data["design"] = {
                "exchangers": [
                    {
                        "id": hx.exchanger_id,
                        "hot_stream": hx.hot_stream_id,
                        "cold_stream": hx.cold_stream_id,
                        "duty_kW": hx.duty_kW,
                        "area_m2": hx.area_m2,
                    }
                    for hx in result.recommended_design.exchangers
                ],
            }

            # Add economic analysis
            if result.recommended_design.economic_analysis:
                econ = result.recommended_design.economic_analysis
                data["economics"] = {
                    "total_capital_cost_usd": econ.total_capital_cost_usd,
                    "annual_savings_usd": econ.annual_utility_savings_usd,
                    "payback_years": econ.payback_period_years,
                    "npv_usd": econ.npv_usd,
                }

            # Add explanation
            data["explanation"] = result.explanation_summary
            data["key_drivers"] = result.key_drivers

        warnings = []
        if result.robustness_score < 0.8:
            warnings.append(
                f"Design robustness score is low ({result.robustness_score:.2f})"
            )

        return APIResponse(
            success=result.status == OptimizationStatus.COMPLETED,
            message=f"Optimization {result.status.value}",
            data=data,
            warnings=warnings,
        )


class CallbackHandler:
    """
    Handler for async callbacks and event notifications.
    """

    def __init__(self) -> None:
        self._callbacks: Dict[str, List[Callable]] = {
            "optimization_started": [],
            "optimization_completed": [],
            "optimization_failed": [],
            "pinch_calculated": [],
        }

    def register(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """Register callback for event."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def emit(
        self,
        event: str,
        data: Any,
    ) -> None:
        """Emit event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
