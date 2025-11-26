# -*- coding: utf-8 -*-
"""
GreenLang Provenance Framework for GL-006 HeatRecoveryMaximizer.

This module provides specialized provenance tracking for heat recovery calculations,
ensuring full traceability of data sources, transformations, and results.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

# Re-export core provenance
from greenlang_core.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    ProvenanceAction,
    DataLineage,
    DataSourceType,
)


class HeatRecoveryDataSource(str, Enum):
    """Heat recovery specific data sources."""
    PROCESS_HISTORIAN = "process_historian"
    SCADA_SYSTEM = "scada_system"
    THERMAL_IMAGING = "thermal_imaging"
    FLOW_METER = "flow_meter"
    TEMPERATURE_SENSOR = "temperature_sensor"
    PRESSURE_SENSOR = "pressure_sensor"
    ENERGY_METER = "energy_meter"
    MANUAL_INPUT = "manual_input"
    CALCULATION = "calculation"
    OPTIMIZATION = "optimization"


class HeatRecoveryAction(str, Enum):
    """Heat recovery specific actions."""
    STREAM_ANALYSIS = "stream_analysis"
    PINCH_ANALYSIS = "pinch_analysis"
    EXERGY_CALCULATION = "exergy_calculation"
    NETWORK_SYNTHESIS = "network_synthesis"
    EXCHANGER_DESIGN = "exchanger_design"
    ROI_CALCULATION = "roi_calculation"
    OPPORTUNITY_RANKING = "opportunity_ranking"
    COMPLIANCE_CHECK = "compliance_check"


@dataclass
class HeatStreamLineage(DataLineage):
    """
    Extended lineage for heat stream data.

    Attributes:
        stream_id: Identifier of the heat stream
        stream_type: Type of stream (hot/cold)
        temperature_source: Source of temperature data
        flow_source: Source of flow data
        measurement_uncertainty: Data uncertainty percentage
    """
    stream_id: str = ""
    stream_type: str = "hot"
    temperature_source: str = ""
    flow_source: str = ""
    measurement_uncertainty: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "stream_id": self.stream_id,
            "stream_type": self.stream_type,
            "temperature_source": self.temperature_source,
            "flow_source": self.flow_source,
            "measurement_uncertainty": self.measurement_uncertainty,
        })
        return base


@dataclass
class CalculationLineage(DataLineage):
    """
    Extended lineage for calculation results.

    Attributes:
        calculation_type: Type of calculation performed
        algorithm_version: Version of the algorithm used
        input_parameters: Parameters used in calculation
        convergence_achieved: Whether calculation converged
        iterations: Number of iterations performed
    """
    calculation_type: str = ""
    algorithm_version: str = "1.0.0"
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    convergence_achieved: bool = True
    iterations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "calculation_type": self.calculation_type,
            "algorithm_version": self.algorithm_version,
            "input_parameters": self.input_parameters,
            "convergence_achieved": self.convergence_achieved,
            "iterations": self.iterations,
        })
        return base


class HeatRecoveryProvenanceTracker(ProvenanceTracker):
    """
    Specialized provenance tracker for heat recovery operations.

    Provides enhanced tracking for heat recovery calculations with
    domain-specific lineage types and actions.
    """

    def __init__(self, agent_id: str = "GL-006", max_records: int = 10000):
        """Initialize the heat recovery provenance tracker."""
        super().__init__(agent_id, max_records)
        self._stream_lineages: Dict[str, HeatStreamLineage] = {}
        self._calculation_lineages: Dict[str, CalculationLineage] = {}

    def track_stream_data(
        self,
        stream_id: str,
        stream_type: str,
        temperature_source: str,
        flow_source: str,
        data: Any = None,
        measurement_uncertainty: float = 0.02,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HeatStreamLineage:
        """
        Track heat stream data provenance.

        Args:
            stream_id: Identifier for the stream
            stream_type: Type of stream (hot/cold)
            temperature_source: Source of temperature data
            flow_source: Source of flow data
            data: The actual data
            measurement_uncertainty: Uncertainty percentage
            metadata: Additional metadata

        Returns:
            HeatStreamLineage record
        """
        lineage = HeatStreamLineage(
            source_type=DataSourceType.SENSOR,
            source_id=stream_id,
            source_name=f"{stream_type}_stream_{stream_id}",
            stream_id=stream_id,
            stream_type=stream_type,
            temperature_source=temperature_source,
            flow_source=flow_source,
            measurement_uncertainty=measurement_uncertainty,
            metadata=metadata or {},
        )

        if data is not None:
            lineage.compute_checksum(data)

        self._stream_lineages[lineage.lineage_id] = lineage
        self._lineages[lineage.lineage_id] = lineage

        # Record the action
        self.record_action(
            action=ProvenanceAction.IMPORT,
            inputs={"stream_id": stream_id, "stream_type": stream_type},
            lineage=lineage,
            metadata={"source": "stream_data_import"},
        )

        return lineage

    def track_calculation(
        self,
        calculation_type: str,
        input_lineages: List[str],
        result: Any = None,
        parameters: Optional[Dict[str, Any]] = None,
        algorithm_version: str = "1.0.0",
        convergence_achieved: bool = True,
        iterations: int = 0,
        duration_ms: Optional[float] = None,
    ) -> CalculationLineage:
        """
        Track calculation provenance.

        Args:
            calculation_type: Type of calculation
            input_lineages: List of input lineage IDs
            result: Calculation result
            parameters: Calculation parameters
            algorithm_version: Version of algorithm
            convergence_achieved: Whether converged
            iterations: Number of iterations
            duration_ms: Duration in milliseconds

        Returns:
            CalculationLineage record
        """
        lineage = CalculationLineage(
            source_type=DataSourceType.CALCULATION,
            source_id=f"{calculation_type}_{uuid.uuid4().hex[:8]}",
            source_name=calculation_type,
            calculation_type=calculation_type,
            algorithm_version=algorithm_version,
            input_parameters=parameters or {},
            convergence_achieved=convergence_achieved,
            iterations=iterations,
            parent_lineages=input_lineages,
        )

        if result is not None:
            lineage.compute_checksum(result)

        lineage.add_transformation(f"Calculation: {calculation_type}")

        self._calculation_lineages[lineage.lineage_id] = lineage
        self._lineages[lineage.lineage_id] = lineage

        # Record the action
        self.record_action(
            action=ProvenanceAction.TRANSFORM,
            inputs={"input_lineages": input_lineages},
            outputs={"result_lineage": lineage.lineage_id},
            parameters=parameters,
            lineage=lineage,
            duration_ms=duration_ms,
            metadata={"calculation_type": calculation_type},
        )

        return lineage

    def track_pinch_analysis(
        self,
        input_streams: List[str],
        pinch_temperature: float,
        min_utility: Dict[str, float],
        parameters: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> CalculationLineage:
        """
        Track pinch analysis provenance.

        Args:
            input_streams: List of input stream lineage IDs
            pinch_temperature: Calculated pinch temperature
            min_utility: Minimum utility requirements
            parameters: Analysis parameters
            duration_ms: Duration in milliseconds

        Returns:
            CalculationLineage record
        """
        return self.track_calculation(
            calculation_type="pinch_analysis",
            input_lineages=input_streams,
            result={
                "pinch_temperature": pinch_temperature,
                "min_utility": min_utility,
            },
            parameters=parameters,
            duration_ms=duration_ms,
        )

    def track_roi_calculation(
        self,
        input_lineages: List[str],
        roi: float,
        npv: float,
        payback_years: float,
        parameters: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> CalculationLineage:
        """
        Track ROI calculation provenance.

        Args:
            input_lineages: Input data lineage IDs
            roi: Calculated ROI
            npv: Net present value
            payback_years: Payback period
            parameters: Calculation parameters
            duration_ms: Duration in milliseconds

        Returns:
            CalculationLineage record
        """
        return self.track_calculation(
            calculation_type="roi_calculation",
            input_lineages=input_lineages,
            result={
                "roi": roi,
                "npv": npv,
                "payback_years": payback_years,
            },
            parameters=parameters,
            duration_ms=duration_ms,
        )

    def get_stream_lineage(self, lineage_id: str) -> Optional[HeatStreamLineage]:
        """Get a stream lineage by ID."""
        return self._stream_lineages.get(lineage_id)

    def get_calculation_lineage(self, lineage_id: str) -> Optional[CalculationLineage]:
        """Get a calculation lineage by ID."""
        return self._calculation_lineages.get(lineage_id)

    def get_lineage_report(self, lineage_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive lineage report.

        Args:
            lineage_id: ID of the lineage to report on

        Returns:
            Comprehensive lineage report
        """
        chain = self.get_full_lineage_chain(lineage_id)

        return {
            "lineage_id": lineage_id,
            "chain_length": len(chain),
            "chain": [l.to_dict() for l in chain],
            "data_sources": list(set(
                l.source_type.value for l in chain
                if isinstance(l.source_type, DataSourceType)
            )),
            "transformations": [
                t for l in chain for t in l.transformations
            ],
            "total_records": len(self._records),
            "generated_at": datetime.utcnow().isoformat(),
        }


__all__ = [
    # Re-exports from core
    'ProvenanceTracker',
    'ProvenanceRecord',
    'ProvenanceAction',
    'DataLineage',
    'DataSourceType',
    # Module-specific
    'HeatRecoveryDataSource',
    'HeatRecoveryAction',
    'HeatStreamLineage',
    'CalculationLineage',
    'HeatRecoveryProvenanceTracker',
]
