"""
FMEAAnalyzer - Failure Mode and Effects Analysis

This module implements FMEA methodology per IEC 60812
for systematic analysis of failure modes and their effects.

Key concepts:
- Failure Modes: Ways components can fail
- Effects: Consequences of failures
- Severity/Occurrence/Detection: Risk factors
- RPN: Risk Priority Number

Reference: IEC 60812:2018

Example:
    >>> from greenlang.safety.risk.fmea_analyzer import FMEAAnalyzer
    >>> analyzer = FMEAAnalyzer()
    >>> study = analyzer.create_study("Valve Assembly")
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class FailureModeCategory(str, Enum):
    """Failure mode categories."""
    STUCK = "stuck"
    LEAK = "leak"
    RUPTURE = "rupture"
    DEGRADED = "degraded"
    SPURIOUS = "spurious"
    DRIFT = "drift"
    SHORT_CIRCUIT = "short_circuit"
    OPEN_CIRCUIT = "open_circuit"


class FailureMode(BaseModel):
    """FMEA failure mode record."""
    fm_id: str = Field(default_factory=lambda: f"FM-{uuid.uuid4().hex[:6].upper()}")
    component_id: str = Field(...)
    component_name: str = Field(...)
    function: str = Field(...)
    failure_mode: str = Field(...)
    failure_category: FailureModeCategory = Field(...)
    local_effect: str = Field(...)
    system_effect: str = Field(...)
    end_effect: str = Field(...)
    cause: str = Field(...)
    detection_method: str = Field(default="")
    severity: int = Field(..., ge=1, le=10, description="Severity (1-10)")
    occurrence: int = Field(..., ge=1, le=10, description="Occurrence (1-10)")
    detection: int = Field(..., ge=1, le=10, description="Detection difficulty (1-10)")
    rpn: int = Field(default=0, description="Risk Priority Number")
    recommended_action: str = Field(default="")
    responsibility: str = Field(default="")
    target_date: Optional[datetime] = Field(None)
    action_taken: str = Field(default="")
    new_severity: Optional[int] = Field(None)
    new_occurrence: Optional[int] = Field(None)
    new_detection: Optional[int] = Field(None)
    new_rpn: Optional[int] = Field(None)


class FMEAStudy(BaseModel):
    """Complete FMEA study record."""
    study_id: str = Field(default_factory=lambda: f"FMEA-{uuid.uuid4().hex[:8].upper()}")
    title: str = Field(...)
    system_name: str = Field(...)
    prepared_by: str = Field(...)
    study_date: datetime = Field(default_factory=datetime.utcnow)
    revision: str = Field(default="1.0")
    failure_modes: List[FailureMode] = Field(default_factory=list)
    total_failure_modes: int = Field(default=0)
    high_rpn_count: int = Field(default=0)
    rpn_threshold: int = Field(default=100)
    status: str = Field(default="in_progress")
    provenance_hash: str = Field(default="")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class FMEAAnalyzer:
    """
    FMEA Study Analyzer.

    Implements FMEA methodology per IEC 60812 for systematic
    failure mode analysis.

    Example:
        >>> analyzer = FMEAAnalyzer()
        >>> study = analyzer.create_study("Pressure Relief Valve", "PRV-001")
    """

    # RPN action thresholds
    RPN_THRESHOLDS = {
        "critical": 200,  # Immediate action required
        "high": 100,  # Action required
        "medium": 50,  # Action recommended
        "low": 0,  # Monitor
    }

    def __init__(self):
        """Initialize FMEAAnalyzer."""
        self.studies: Dict[str, FMEAStudy] = {}
        logger.info("FMEAAnalyzer initialized")

    def create_study(
        self,
        title: str,
        system_name: str,
        prepared_by: str,
        rpn_threshold: int = 100
    ) -> FMEAStudy:
        """Create a new FMEA study."""
        study = FMEAStudy(
            title=title,
            system_name=system_name,
            prepared_by=prepared_by,
            rpn_threshold=rpn_threshold,
        )

        self.studies[study.study_id] = study
        logger.info(f"FMEA study created: {study.study_id}")
        return study

    def add_failure_mode(
        self,
        study_id: str,
        failure_mode: FailureMode
    ) -> FailureMode:
        """Add a failure mode to study."""
        if study_id not in self.studies:
            raise ValueError(f"Study not found: {study_id}")

        study = self.studies[study_id]

        # Calculate RPN
        failure_mode.rpn = (
            failure_mode.severity *
            failure_mode.occurrence *
            failure_mode.detection
        )

        study.failure_modes.append(failure_mode)
        study.total_failure_modes += 1

        if failure_mode.rpn >= study.rpn_threshold:
            study.high_rpn_count += 1

        logger.info(f"Failure mode added: {failure_mode.fm_id}, RPN={failure_mode.rpn}")
        return failure_mode

    def update_after_action(
        self,
        study_id: str,
        fm_id: str,
        action_taken: str,
        new_severity: Optional[int] = None,
        new_occurrence: Optional[int] = None,
        new_detection: Optional[int] = None
    ) -> FailureMode:
        """Update failure mode after corrective action."""
        if study_id not in self.studies:
            raise ValueError(f"Study not found: {study_id}")

        study = self.studies[study_id]
        for fm in study.failure_modes:
            if fm.fm_id == fm_id:
                fm.action_taken = action_taken
                fm.new_severity = new_severity or fm.severity
                fm.new_occurrence = new_occurrence or fm.occurrence
                fm.new_detection = new_detection or fm.detection
                fm.new_rpn = fm.new_severity * fm.new_occurrence * fm.new_detection

                logger.info(f"Failure mode updated: {fm_id}, new RPN={fm.new_rpn}")
                return fm

        raise ValueError(f"Failure mode not found: {fm_id}")

    def get_pareto_analysis(
        self,
        study_id: str,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get Pareto analysis of failure modes by RPN."""
        if study_id not in self.studies:
            raise ValueError(f"Study not found: {study_id}")

        study = self.studies[study_id]
        sorted_fms = sorted(
            study.failure_modes,
            key=lambda x: x.rpn,
            reverse=True
        )[:top_n]

        return [
            {
                "fm_id": fm.fm_id,
                "component": fm.component_name,
                "failure_mode": fm.failure_mode,
                "rpn": fm.rpn,
                "severity": fm.severity,
                "occurrence": fm.occurrence,
                "detection": fm.detection,
            }
            for fm in sorted_fms
        ]

    def complete_study(self, study_id: str) -> FMEAStudy:
        """Mark study as complete."""
        if study_id not in self.studies:
            raise ValueError(f"Study not found: {study_id}")

        study = self.studies[study_id]
        study.status = "completed"
        study.provenance_hash = hashlib.sha256(
            f"{study.study_id}|{study.total_failure_modes}|{study.high_rpn_count}".encode()
        ).hexdigest()

        return study
