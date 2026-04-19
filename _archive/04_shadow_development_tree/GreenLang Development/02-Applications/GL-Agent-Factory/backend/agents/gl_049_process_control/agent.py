"""GL-049: Process Control Agent (PROCESS-CONTROL).

Optimizes process control for thermal efficiency.

Standards: ISA-95, ISA-88
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ControlMode(str, Enum):
    MANUAL = "MANUAL"
    AUTO = "AUTO"
    CASCADE = "CASCADE"
    RATIO = "RATIO"


class LoopStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    ACCEPTABLE = "ACCEPTABLE"
    NEEDS_TUNING = "NEEDS_TUNING"
    UNSTABLE = "UNSTABLE"


class ControlLoop(BaseModel):
    loop_id: str
    loop_name: str
    control_mode: ControlMode = Field(default=ControlMode.AUTO)
    setpoint: float
    process_value: float
    output_pct: float = Field(ge=0, le=100)
    kp: float = Field(default=1.0, description="Proportional gain")
    ki: float = Field(default=0.1, description="Integral gain")
    kd: float = Field(default=0.0, description="Derivative gain")


class ProcessControlInput(BaseModel):
    equipment_id: str
    equipment_name: str = Field(default="Process")
    control_loops: List[ControlLoop] = Field(default_factory=list)
    sample_rate_seconds: float = Field(default=1.0, gt=0)
    target_efficiency_pct: float = Field(default=90, ge=0, le=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LoopAnalysis(BaseModel):
    loop_id: str
    loop_name: str
    error: float
    error_pct: float
    status: LoopStatus
    iae: float  # Integral Absolute Error
    response_quality: str
    tuning_recommendation: Optional[str]


class ProcessControlOutput(BaseModel):
    equipment_id: str
    equipment_name: str
    overall_control_score: float
    loops_analyzed: int
    loops_optimal: int
    loops_needing_attention: int
    loop_analyses: List[LoopAnalysis]
    overall_efficiency_pct: float
    energy_waste_pct: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class ProcessControlAgent:
    AGENT_ID = "GL-049"
    AGENT_NAME = "PROCESS-CONTROL"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"ProcessControlAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = ProcessControlInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _analyze_loop(self, loop: ControlLoop) -> LoopAnalysis:
        """Analyze a single control loop."""
        error = loop.process_value - loop.setpoint
        error_pct = abs(error / loop.setpoint * 100) if loop.setpoint != 0 else abs(error)

        # Simulate IAE (in practice would come from historical data)
        iae = abs(error) * 10  # Simplified

        # Determine status
        if error_pct <= 1:
            status = LoopStatus.OPTIMAL
            quality = "Excellent"
        elif error_pct <= 3:
            status = LoopStatus.ACCEPTABLE
            quality = "Good"
        elif error_pct <= 10:
            status = LoopStatus.NEEDS_TUNING
            quality = "Fair"
        else:
            status = LoopStatus.UNSTABLE
            quality = "Poor"

        # Tuning recommendation
        tuning = None
        if status in [LoopStatus.NEEDS_TUNING, LoopStatus.UNSTABLE]:
            if error > 0:  # PV > SP, need more aggressive control
                tuning = f"Increase Kp to {loop.kp * 1.2:.2f} or reduce Ki"
            else:
                tuning = f"Decrease Kp to {loop.kp * 0.8:.2f} for stability"

        return LoopAnalysis(
            loop_id=loop.loop_id,
            loop_name=loop.loop_name,
            error=round(error, 3),
            error_pct=round(error_pct, 2),
            status=status,
            iae=round(iae, 3),
            response_quality=quality,
            tuning_recommendation=tuning
        )

    def _process(self, inp: ProcessControlInput) -> ProcessControlOutput:
        recommendations = []
        analyses = []
        optimal_count = 0
        attention_count = 0

        for loop in inp.control_loops:
            analysis = self._analyze_loop(loop)
            analyses.append(analysis)

            if analysis.status == LoopStatus.OPTIMAL:
                optimal_count += 1
            if analysis.status in [LoopStatus.NEEDS_TUNING, LoopStatus.UNSTABLE]:
                attention_count += 1

        # Overall control score
        total = len(analyses)
        if total > 0:
            score = (optimal_count * 100 + (total - attention_count - optimal_count) * 70) / total
        else:
            score = 100

        # Efficiency based on control quality
        efficiency = inp.target_efficiency_pct * (score / 100)
        energy_waste = inp.target_efficiency_pct - efficiency

        # Manual loops recommendation
        manual_loops = [l for l in inp.control_loops if l.control_mode == ControlMode.MANUAL]
        if manual_loops:
            recommendations.append(f"{len(manual_loops)} loops in manual mode - consider auto")

        # Unstable loops
        unstable = [a for a in analyses if a.status == LoopStatus.UNSTABLE]
        if unstable:
            recommendations.append(f"URGENT: {len(unstable)} loops unstable - requires immediate tuning")

        # Energy waste
        if energy_waste > 5:
            recommendations.append(f"Control inefficiency causing {energy_waste:.1f}% energy waste")

        # General tuning
        if attention_count > 0:
            recommendations.append(f"{attention_count} loops need tuning - schedule maintenance")

        calc_hash = hashlib.sha256(json.dumps({
            "equipment": inp.equipment_id,
            "score": round(score, 1),
            "loops": total
        }).encode()).hexdigest()

        return ProcessControlOutput(
            equipment_id=inp.equipment_id,
            equipment_name=inp.equipment_name,
            overall_control_score=round(score, 1),
            loops_analyzed=total,
            loops_optimal=optimal_count,
            loops_needing_attention=attention_count,
            loop_analyses=analyses,
            overall_efficiency_pct=round(efficiency, 1),
            energy_waste_pct=round(energy_waste, 1),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-049", "name": "PROCESS-CONTROL", "version": "1.0.0",
    "summary": "Process control optimization for thermal efficiency",
    "standards": [{"ref": "ISA-95"}, {"ref": "ISA-88"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
