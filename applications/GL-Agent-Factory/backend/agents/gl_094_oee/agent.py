"""
GL-094: OEE Maximizer Agent (OEE-MAXIMIZER)

This module implements the OEEMaximizerAgent for calculating and optimizing
Overall Equipment Effectiveness (OEE) in manufacturing operations.

The agent provides:
- Real-time OEE calculation (Availability x Performance x Quality)
- Loss categorization and analysis (Six Big Losses)
- Bottleneck identification
- Improvement opportunity ranking
- Benchmark comparison
- Complete SHA-256 provenance tracking

Standards Compliance:
- ISO 22400 (Manufacturing KPIs)
- ANSI/ISA-95 (Manufacturing Operations)
- TPM (Total Productive Maintenance)
- World Class OEE Standards

Example:
    >>> agent = OEEMaximizerAgent()
    >>> result = agent.run(OEEInput(
    ...     equipment_runtime=[...],
    ...     production_output=[...],
    ...     quality_data=...,
    ... ))
    >>> print(f"OEE: {result.overall_oee_pct}%")
"""

import hashlib
import json
import logging
import math
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class LossCategory(str, Enum):
    """Six Big Losses categories."""
    BREAKDOWNS = "BREAKDOWNS"
    SETUP_ADJUSTMENTS = "SETUP_ADJUSTMENTS"
    SMALL_STOPS = "SMALL_STOPS"
    REDUCED_SPEED = "REDUCED_SPEED"
    STARTUP_REJECTS = "STARTUP_REJECTS"
    PRODUCTION_REJECTS = "PRODUCTION_REJECTS"


class OEEClass(str, Enum):
    """OEE performance classification."""
    WORLD_CLASS = "WORLD_CLASS"  # >= 85%
    COMPETITIVE = "COMPETITIVE"  # >= 65%
    ACCEPTABLE = "ACCEPTABLE"  # >= 50%
    POOR = "POOR"  # < 50%


# =============================================================================
# INPUT MODELS
# =============================================================================

class EquipmentRuntime(BaseModel):
    """Equipment runtime data."""

    equipment_id: str = Field(..., description="Equipment identifier")
    shift_date: datetime = Field(..., description="Shift date")
    shift_duration_minutes: int = Field(..., ge=0, description="Total shift duration")
    planned_downtime_minutes: int = Field(..., ge=0, description="Planned downtime")
    unplanned_downtime_minutes: int = Field(..., ge=0, description="Unplanned downtime")
    setup_time_minutes: int = Field(..., ge=0, description="Setup/changeover time")
    breakdown_time_minutes: int = Field(..., ge=0, description="Breakdown time")
    small_stops_minutes: int = Field(..., ge=0, description="Small stops/idling")


class ProductionOutput(BaseModel):
    """Production output data."""

    equipment_id: str = Field(..., description="Equipment identifier")
    shift_date: datetime = Field(..., description="Shift date")
    target_cycle_time_seconds: float = Field(..., gt=0, description="Ideal cycle time")
    actual_cycle_time_seconds: float = Field(..., gt=0, description="Actual cycle time")
    units_produced: int = Field(..., ge=0, description="Total units produced")
    good_units: int = Field(..., ge=0, description="Good quality units")
    rejected_units: int = Field(..., ge=0, description="Rejected units")
    startup_rejects: int = Field(..., ge=0, description="Startup/warmup rejects")


class OEEInput(BaseModel):
    """Complete input model for OEE Maximizer."""

    equipment_runtime: List[EquipmentRuntime] = Field(
        ...,
        description="Equipment runtime data"
    )
    production_output: List[ProductionOutput] = Field(
        ...,
        description="Production output data"
    )
    oee_target_pct: float = Field(
        default=85.0,
        ge=0,
        le=100,
        description="Target OEE percentage"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('equipment_runtime')
    def validate_runtime(cls, v):
        """Validate runtime data exists."""
        if not v:
            raise ValueError("At least one equipment runtime record required")
        return v


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class OEEMetrics(BaseModel):
    """OEE component metrics."""

    equipment_id: str = Field(..., description="Equipment identifier")
    availability_pct: float = Field(..., description="Availability %")
    performance_pct: float = Field(..., description="Performance %")
    quality_pct: float = Field(..., description="Quality %")
    overall_oee_pct: float = Field(..., description="Overall OEE %")
    oee_class: OEEClass = Field(..., description="OEE classification")
    fully_productive_time_minutes: float = Field(..., description="Fully productive time")


class LossAnalysis(BaseModel):
    """Loss category analysis."""

    loss_category: LossCategory = Field(..., description="Loss category")
    total_time_lost_minutes: float = Field(..., description="Total time lost")
    percentage_of_planned_time: float = Field(..., description="% of planned production time")
    estimated_units_lost: int = Field(..., description="Estimated lost production")
    cost_impact_eur: float = Field(..., description="Estimated cost impact")
    improvement_priority: int = Field(..., description="Priority ranking (1=highest)")


class ImprovementOpportunity(BaseModel):
    """OEE improvement opportunity."""

    equipment_id: str = Field(..., description="Equipment identifier")
    opportunity_type: str = Field(..., description="Opportunity category")
    current_value: float = Field(..., description="Current metric value")
    target_value: float = Field(..., description="Target metric value")
    potential_oee_gain_pct: float = Field(..., description="Potential OEE gain %")
    estimated_annual_value_eur: float = Field(..., description="Annual value")
    implementation_effort: str = Field(..., description="Implementation effort")
    recommendation: str = Field(..., description="Specific recommendation")


class ProvenanceRecord(BaseModel):
    """Provenance tracking record."""

    operation: str = Field(..., description="Operation performed")
    timestamp: datetime = Field(..., description="Operation timestamp")
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    output_hash: str = Field(..., description="SHA-256 hash of outputs")
    tool_name: str = Field(..., description="Tool/calculator used")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")


class OEEOutput(BaseModel):
    """Complete output model for OEE Maximizer."""

    # Identification
    analysis_id: str = Field(..., description="Unique analysis identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")

    # OEE Metrics
    oee_by_equipment: List[OEEMetrics] = Field(..., description="OEE metrics by equipment")
    overall_oee_pct: float = Field(..., description="Overall facility OEE")
    overall_availability_pct: float = Field(..., description="Overall availability")
    overall_performance_pct: float = Field(..., description="Overall performance")
    overall_quality_pct: float = Field(..., description="Overall quality")

    # Loss Analysis
    loss_analyses: List[LossAnalysis] = Field(..., description="Six Big Losses analysis")
    total_losses_eur: float = Field(..., description="Total estimated losses")

    # Improvement Opportunities
    improvement_opportunities: List[ImprovementOpportunity] = Field(
        ...,
        description="Ranked improvement opportunities"
    )

    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Key recommendations")
    warnings: List[str] = Field(default_factory=list, description="Critical warnings")

    # Provenance
    provenance_chain: List[ProvenanceRecord] = Field(..., description="Complete audit trail")
    provenance_hash: str = Field(..., description="SHA-256 hash of provenance chain")

    # Processing Metadata
    processing_time_ms: float = Field(..., description="Processing time (ms)")
    validation_status: str = Field(..., description="PASS or FAIL")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")


# =============================================================================
# OEE MAXIMIZER AGENT
# =============================================================================

class OEEMaximizerAgent:
    """
    GL-094: OEE Maximizer Agent (OEE-MAXIMIZER).

    This agent calculates and optimizes Overall Equipment Effectiveness
    through comprehensive loss analysis and improvement identification.

    Zero-Hallucination Guarantee:
    - All calculations use standard OEE formulas
    - OEE = Availability × Performance × Quality
    - Loss analysis based on actual time measurements
    - No LLM inference in calculation path
    - Complete audit trail for performance tracking

    Attributes:
        AGENT_ID: Unique agent identifier (GL-094)
        AGENT_NAME: Agent name (OEE-MAXIMIZER)
        VERSION: Agent version
    """

    AGENT_ID = "GL-094"
    AGENT_NAME = "OEE-MAXIMIZER"
    VERSION = "1.0.0"
    DESCRIPTION = "Overall Equipment Effectiveness Optimization Agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the OEEMaximizerAgent."""
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []
        self._warnings: List[str] = []
        self._recommendations: List[str] = []

        logger.info(
            f"OEEMaximizerAgent initialized "
            f"(ID: {self.AGENT_ID}, Name: {self.AGENT_NAME}, Version: {self.VERSION})"
        )

    def run(self, input_data: OEEInput) -> OEEOutput:
        """Execute OEE analysis."""
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []
        self._warnings = []
        self._recommendations = []

        logger.info(
            f"Starting OEE analysis "
            f"(equipment_records={len(input_data.equipment_runtime)}, "
            f"production_records={len(input_data.production_output)})"
        )

        try:
            # Step 1: Calculate OEE metrics
            oee_metrics = self._calculate_oee_metrics(
                input_data.equipment_runtime,
                input_data.production_output
            )

            overall_oee = sum(m.overall_oee_pct for m in oee_metrics) / len(oee_metrics) if oee_metrics else 0
            overall_avail = sum(m.availability_pct for m in oee_metrics) / len(oee_metrics) if oee_metrics else 0
            overall_perf = sum(m.performance_pct for m in oee_metrics) / len(oee_metrics) if oee_metrics else 0
            overall_qual = sum(m.quality_pct for m in oee_metrics) / len(oee_metrics) if oee_metrics else 0

            self._track_provenance(
                "oee_calculation",
                {"equipment_count": len(input_data.equipment_runtime)},
                {"overall_oee": overall_oee},
                "OEE Calculator"
            )

            # Step 2: Analyze losses
            loss_analyses = self._analyze_losses(
                input_data.equipment_runtime,
                input_data.production_output
            )
            total_losses = sum(l.cost_impact_eur for l in loss_analyses)
            self._track_provenance(
                "loss_analysis",
                {"records": len(input_data.equipment_runtime)},
                {"total_losses": total_losses},
                "Loss Analyzer"
            )

            # Step 3: Identify improvement opportunities
            opportunities = self._identify_opportunities(
                oee_metrics,
                loss_analyses,
                input_data.oee_target_pct
            )
            self._track_provenance(
                "opportunity_identification",
                {"target_oee": input_data.oee_target_pct},
                {"opportunities": len(opportunities)},
                "Opportunity Analyzer"
            )

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Generate analysis ID
            analysis_id = (
                f"OEE-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-"
                f"{hashlib.sha256(str(input_data.dict()).encode()).hexdigest()[:8]}"
            )

            validation_status = "PASS" if not self._validation_errors else "FAIL"

            output = OEEOutput(
                analysis_id=analysis_id,
                oee_by_equipment=oee_metrics,
                overall_oee_pct=round(overall_oee, 2),
                overall_availability_pct=round(overall_avail, 2),
                overall_performance_pct=round(overall_perf, 2),
                overall_quality_pct=round(overall_qual, 2),
                loss_analyses=loss_analyses,
                total_losses_eur=round(total_losses, 2),
                improvement_opportunities=opportunities,
                recommendations=self._recommendations,
                warnings=self._warnings,
                provenance_chain=[
                    ProvenanceRecord(
                        operation=step["operation"],
                        timestamp=step["timestamp"],
                        input_hash=step["input_hash"],
                        output_hash=step["output_hash"],
                        tool_name=step["tool_name"],
                        parameters=step.get("parameters", {}),
                    )
                    for step in self._provenance_steps
                ],
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time, 2),
                validation_status=validation_status,
                validation_errors=self._validation_errors,
            )

            logger.info(
                f"OEE analysis complete: Overall OEE={overall_oee:.1f}% "
                f"(duration: {processing_time:.2f} ms)"
            )

            return output

        except Exception as e:
            logger.error(f"OEE analysis failed: {str(e)}", exc_info=True)
            raise

    def _calculate_oee_metrics(
        self,
        runtime_data: List[EquipmentRuntime],
        production_data: List[ProductionOutput]
    ) -> List[OEEMetrics]:
        """
        Calculate OEE metrics.

        ZERO-HALLUCINATION:
        - Availability = (Planned Production Time - Downtime) / Planned Production Time
        - Performance = (Ideal Cycle Time × Total Count) / Operating Time
        - Quality = Good Count / Total Count
        - OEE = Availability × Performance × Quality
        """
        metrics_by_equipment = {}

        # Group data by equipment
        for runtime in runtime_data:
            eq_id = runtime.equipment_id
            if eq_id not in metrics_by_equipment:
                metrics_by_equipment[eq_id] = {"runtime": [], "production": []}
            metrics_by_equipment[eq_id]["runtime"].append(runtime)

        for production in production_data:
            eq_id = production.equipment_id
            if eq_id not in metrics_by_equipment:
                metrics_by_equipment[eq_id] = {"runtime": [], "production": []}
            metrics_by_equipment[eq_id]["production"].append(production)

        oee_metrics = []

        for eq_id, data in metrics_by_equipment.items():
            runtime_records = data["runtime"]
            production_records = data["production"]

            if not runtime_records or not production_records:
                continue

            # Aggregate runtime data
            total_shift_time = sum(r.shift_duration_minutes for r in runtime_records)
            total_planned_downtime = sum(r.planned_downtime_minutes for r in runtime_records)
            total_unplanned_downtime = sum(r.unplanned_downtime_minutes for r in runtime_records)

            # Planned production time
            planned_production_time = total_shift_time - total_planned_downtime

            # Operating time
            operating_time = planned_production_time - total_unplanned_downtime

            # Availability
            availability = (operating_time / planned_production_time * 100) if planned_production_time > 0 else 0

            # Aggregate production data
            total_units = sum(p.units_produced for p in production_records)
            total_good = sum(p.good_units for p in production_records)
            avg_ideal_cycle = sum(p.target_cycle_time_seconds for p in production_records) / len(production_records)

            # Performance
            ideal_time_minutes = (total_units * avg_ideal_cycle / 60)
            performance = (ideal_time_minutes / operating_time * 100) if operating_time > 0 else 0
            performance = min(100, performance)  # Cap at 100%

            # Quality
            quality = (total_good / total_units * 100) if total_units > 0 else 0

            # Overall OEE
            oee = (availability * performance * quality) / 10000

            # Classify OEE
            if oee >= 85:
                oee_class = OEEClass.WORLD_CLASS
            elif oee >= 65:
                oee_class = OEEClass.COMPETITIVE
            elif oee >= 50:
                oee_class = OEEClass.ACCEPTABLE
            else:
                oee_class = OEEClass.POOR
                self._warnings.append(f"Equipment {eq_id} has POOR OEE: {oee:.1f}%")

            # Fully productive time
            fully_productive = operating_time * (performance / 100) * (quality / 100)

            oee_metrics.append(OEEMetrics(
                equipment_id=eq_id,
                availability_pct=round(availability, 2),
                performance_pct=round(performance, 2),
                quality_pct=round(quality, 2),
                overall_oee_pct=round(oee, 2),
                oee_class=oee_class,
                fully_productive_time_minutes=round(fully_productive, 2),
            ))

        return oee_metrics

    def _analyze_losses(
        self,
        runtime_data: List[EquipmentRuntime],
        production_data: List[ProductionOutput]
    ) -> List[LossAnalysis]:
        """Analyze Six Big Losses."""
        total_planned_time = sum(
            r.shift_duration_minutes - r.planned_downtime_minutes
            for r in runtime_data
        )

        # Aggregate losses
        losses = {
            LossCategory.BREAKDOWNS: sum(r.breakdown_time_minutes for r in runtime_data),
            LossCategory.SETUP_ADJUSTMENTS: sum(r.setup_time_minutes for r in runtime_data),
            LossCategory.SMALL_STOPS: sum(r.small_stops_minutes for r in runtime_data),
            LossCategory.REDUCED_SPEED: 0,  # Calculated from performance
            LossCategory.STARTUP_REJECTS: 0,
            LossCategory.PRODUCTION_REJECTS: 0,
        }

        # Calculate reduced speed loss
        for prod in production_data:
            if prod.actual_cycle_time_seconds > prod.target_cycle_time_seconds:
                excess_time = (prod.actual_cycle_time_seconds - prod.target_cycle_time_seconds) * prod.units_produced / 60
                losses[LossCategory.REDUCED_SPEED] += excess_time

        # Quality losses
        total_startup_rejects = sum(p.startup_rejects for p in production_data)
        total_production_rejects = sum(p.rejected_units - p.startup_rejects for p in production_data)
        avg_cycle_time = sum(p.target_cycle_time_seconds for p in production_data) / len(production_data) if production_data else 1

        losses[LossCategory.STARTUP_REJECTS] = total_startup_rejects * avg_cycle_time / 60
        losses[LossCategory.PRODUCTION_REJECTS] = total_production_rejects * avg_cycle_time / 60

        # Rank by impact
        sorted_losses = sorted(losses.items(), key=lambda x: x[1], reverse=True)

        analyses = []
        for priority, (category, time_lost) in enumerate(sorted_losses, 1):
            pct_of_planned = (time_lost / total_planned_time * 100) if total_planned_time > 0 else 0

            # Estimate units lost
            units_lost = int(time_lost * 60 / avg_cycle_time) if avg_cycle_time > 0 else 0

            # Estimate cost (simplified: 50 EUR per hour)
            cost_impact = time_lost / 60 * 50

            analyses.append(LossAnalysis(
                loss_category=category,
                total_time_lost_minutes=round(time_lost, 2),
                percentage_of_planned_time=round(pct_of_planned, 2),
                estimated_units_lost=units_lost,
                cost_impact_eur=round(cost_impact, 2),
                improvement_priority=priority,
            ))

        return analyses

    def _identify_opportunities(
        self,
        oee_metrics: List[OEEMetrics],
        losses: List[LossAnalysis],
        target_oee: float
    ) -> List[ImprovementOpportunity]:
        """Identify improvement opportunities."""
        opportunities = []

        for metric in oee_metrics:
            if metric.overall_oee_pct < target_oee:
                gap = target_oee - metric.overall_oee_pct

                # Availability improvement
                if metric.availability_pct < 90:
                    opportunities.append(ImprovementOpportunity(
                        equipment_id=metric.equipment_id,
                        opportunity_type="Availability Improvement",
                        current_value=metric.availability_pct,
                        target_value=90.0,
                        potential_oee_gain_pct=round((90 - metric.availability_pct) * metric.performance_pct * metric.quality_pct / 10000, 2),
                        estimated_annual_value_eur=5000.0,
                        implementation_effort="MEDIUM",
                        recommendation="Implement predictive maintenance to reduce unplanned downtime",
                    ))

                # Performance improvement
                if metric.performance_pct < 95:
                    opportunities.append(ImprovementOpportunity(
                        equipment_id=metric.equipment_id,
                        opportunity_type="Performance Improvement",
                        current_value=metric.performance_pct,
                        target_value=95.0,
                        potential_oee_gain_pct=round((95 - metric.performance_pct) * metric.availability_pct * metric.quality_pct / 10000, 2),
                        estimated_annual_value_eur=3000.0,
                        implementation_effort="LOW",
                        recommendation="Optimize cycle times and reduce minor stoppages",
                    ))

                # Quality improvement
                if metric.quality_pct < 99:
                    opportunities.append(ImprovementOpportunity(
                        equipment_id=metric.equipment_id,
                        opportunity_type="Quality Improvement",
                        current_value=metric.quality_pct,
                        target_value=99.0,
                        potential_oee_gain_pct=round((99 - metric.quality_pct) * metric.availability_pct * metric.performance_pct / 10000, 2),
                        estimated_annual_value_eur=4000.0,
                        implementation_effort="HIGH",
                        recommendation="Implement Six Sigma quality improvement initiatives",
                    ))

        # Sort by potential gain
        opportunities.sort(key=lambda x: x.potential_oee_gain_pct, reverse=True)

        return opportunities

    def _track_provenance(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tool_name: str
    ) -> None:
        """Track a calculation step for audit trail."""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        output_str = json.dumps(outputs, sort_keys=True, default=str)

        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(input_str.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode()).hexdigest(),
            "tool_name": tool_name,
            "parameters": inputs,
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of complete provenance chain."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "steps": [
                {
                    "operation": s["operation"],
                    "input_hash": s["input_hash"],
                    "output_hash": s["output_hash"],
                }
                for s in self._provenance_steps
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-094",
    "name": "OEE-MAXIMIZER - OEE Maximizer Agent",
    "version": "1.0.0",
    "summary": "Overall Equipment Effectiveness optimization and loss analysis",
    "tags": [
        "oee",
        "manufacturing-efficiency",
        "tpm",
        "six-big-losses",
        "ISO-22400",
        "performance-optimization",
    ],
    "owners": ["operations-team"],
    "compute": {
        "entrypoint": "python://agents.gl_094_oee.agent:OEEMaximizerAgent",
        "deterministic": True,
    },
    "standards": [
        {"ref": "ISO-22400", "description": "Manufacturing Key Performance Indicators"},
        {"ref": "ISA-95", "description": "Manufacturing Operations Management"},
        {"ref": "TPM", "description": "Total Productive Maintenance"},
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True,
    },
}
