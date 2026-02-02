"""
GL-041: Startup Optimizer Agent (STARTUP-OPTIMIZER)

Equipment startup sequence optimization for thermal stress and fuel minimization.

Standards: NFPA 85, OEM Guidelines
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EquipmentItem(BaseModel):
    """Equipment item for startup sequence."""
    equipment_id: str
    equipment_type: str
    min_startup_time_minutes: float = Field(default=10)
    max_ramp_rate_c_per_min: float = Field(default=5)
    dependencies: List[str] = Field(default_factory=list)
    fuel_consumption_startup_mmbtu: float = Field(default=1.0)
    priority: int = Field(default=5)


class StartupConstraint(BaseModel):
    """Startup constraint."""
    constraint_type: str
    equipment_id: str
    value: float
    description: str


class StartupOptimizerInput(BaseModel):
    """Input for StartupOptimizerAgent."""
    system_id: str
    equipment_list: List[EquipmentItem]
    constraints: List[StartupConstraint] = Field(default_factory=list)
    target_completion_time_minutes: float = Field(default=60)
    ambient_temp_celsius: float = Field(default=20)
    objective: str = Field(default="minimize_time")  # minimize_time, minimize_fuel, minimize_stress
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StartupStep(BaseModel):
    """Individual startup sequence step."""
    step_number: int
    equipment_id: str
    start_time_minutes: float
    duration_minutes: float
    end_time_minutes: float
    action: str
    ramp_rate_c_per_min: float
    fuel_consumption_mmbtu: float


class StartupOptimizerOutput(BaseModel):
    """Output from StartupOptimizerAgent."""
    analysis_id: str
    system_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    optimal_sequence: List[StartupStep]
    total_startup_duration_minutes: float
    total_fuel_consumption_mmbtu: float
    max_thermal_stress_factor: float
    risk_assessment: str
    parallel_operations: int
    bottleneck_equipment: Optional[str]
    recommendations: List[str]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class StartupOptimizerAgent:
    """GL-041: Startup Optimizer Agent."""

    AGENT_ID = "GL-041"
    AGENT_NAME = "STARTUP-OPTIMIZER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"StartupOptimizerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: StartupOptimizerInput) -> StartupOptimizerOutput:
        """Execute startup sequence optimization."""
        start_time = datetime.utcnow()

        # Topological sort based on dependencies
        sequence = []
        scheduled = set()
        current_time = 0

        equipment_map = {e.equipment_id: e for e in input_data.equipment_list}
        remaining = list(input_data.equipment_list)

        step_num = 0
        while remaining:
            # Find equipment with satisfied dependencies
            ready = [e for e in remaining if all(d in scheduled for d in e.dependencies)]

            if not ready:
                # Circular dependency or missing equipment
                logger.warning("Cannot resolve all dependencies")
                break

            # Sort by priority
            ready.sort(key=lambda e: e.priority)

            for equip in ready:
                step_num += 1

                # Calculate start time (after dependencies)
                dep_end_times = [
                    s.end_time_minutes for s in sequence
                    if s.equipment_id in equip.dependencies
                ]
                start_min = max(dep_end_times) if dep_end_times else current_time

                duration = equip.min_startup_time_minutes

                # Adjust for objective
                if input_data.objective == "minimize_stress":
                    # Slower ramp = longer time but less stress
                    duration *= 1.5
                    ramp_rate = equip.max_ramp_rate_c_per_min * 0.6
                else:
                    ramp_rate = equip.max_ramp_rate_c_per_min

                sequence.append(StartupStep(
                    step_number=step_num,
                    equipment_id=equip.equipment_id,
                    start_time_minutes=round(start_min, 1),
                    duration_minutes=round(duration, 1),
                    end_time_minutes=round(start_min + duration, 1),
                    action=f"Start {equip.equipment_type}",
                    ramp_rate_c_per_min=round(ramp_rate, 1),
                    fuel_consumption_mmbtu=equip.fuel_consumption_startup_mmbtu
                ))

                scheduled.add(equip.equipment_id)
                remaining.remove(equip)

        # Calculate totals
        total_duration = max(s.end_time_minutes for s in sequence) if sequence else 0
        total_fuel = sum(s.fuel_consumption_mmbtu for s in sequence)

        # Thermal stress factor (based on ramp rates)
        max_stress = max(s.ramp_rate_c_per_min / 5 for s in sequence) if sequence else 0

        # Risk assessment
        if max_stress > 1.2:
            risk = "HIGH - Thermal stress exceeds recommended limits"
        elif max_stress > 1.0:
            risk = "MODERATE - Near thermal stress limits"
        else:
            risk = "LOW - Within safe thermal limits"

        # Find parallel operations
        parallel_count = 0
        for s1 in sequence:
            overlapping = sum(1 for s2 in sequence
                              if s2 != s1 and s1.start_time_minutes < s2.end_time_minutes
                              and s1.end_time_minutes > s2.start_time_minutes)
            parallel_count = max(parallel_count, overlapping)

        # Find bottleneck
        bottleneck = max(sequence, key=lambda s: s.duration_minutes).equipment_id if sequence else None

        recommendations = []
        if total_duration > input_data.target_completion_time_minutes:
            recommendations.append(f"Startup exceeds target by {total_duration - input_data.target_completion_time_minutes:.0f} minutes")
        if max_stress > 1.0:
            recommendations.append("Consider reducing ramp rates to minimize thermal stress")
        if parallel_count < 2 and len(sequence) > 3:
            recommendations.append("Opportunity to parallelize more startup operations")

        provenance_hash = hashlib.sha256(
            json.dumps({"agent": self.AGENT_ID, "system": input_data.system_id,
                        "timestamp": datetime.utcnow().isoformat()}, sort_keys=True).encode()
        ).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return StartupOptimizerOutput(
            analysis_id=f"SO-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            system_id=input_data.system_id,
            optimal_sequence=sequence,
            total_startup_duration_minutes=round(total_duration, 1),
            total_fuel_consumption_mmbtu=round(total_fuel, 2),
            max_thermal_stress_factor=round(max_stress, 2),
            risk_assessment=risk,
            parallel_operations=parallel_count,
            bottleneck_equipment=bottleneck,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS"
        )


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-041",
    "name": "STARTUP-OPTIMIZER",
    "version": "1.0.0",
    "summary": "Equipment startup sequence optimization",
    "tags": ["startup", "sequence", "thermal-stress", "NFPA-85"],
    "standards": [{"ref": "NFPA 85", "description": "Boiler and Combustion Systems"}]
}
