"""GL-097: Data Quality Agent (DATA-QUALITY).

Ensures data quality for energy management systems.

Standards: ISO 8000, DAMA DMBOK
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QualityDimension(str, Enum):
    COMPLETENESS = "COMPLETENESS"
    ACCURACY = "ACCURACY"
    TIMELINESS = "TIMELINESS"
    CONSISTENCY = "CONSISTENCY"
    VALIDITY = "VALIDITY"


class DataSource(BaseModel):
    source_id: str
    source_name: str
    records_total: int = Field(ge=0)
    records_complete: int = Field(ge=0)
    records_valid: int = Field(ge=0)
    last_update_hours: float = Field(ge=0)
    error_count: int = Field(ge=0)


class DataQualityInput(BaseModel):
    system_id: str
    system_name: str = Field(default="Energy Data System")
    sources: List[DataSource] = Field(default_factory=list)
    freshness_threshold_hours: float = Field(default=1, ge=0)
    accuracy_target_pct: float = Field(default=99, ge=0, le=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DimensionScore(BaseModel):
    dimension: str
    score: float
    status: str
    issues_count: int


class DataQualityOutput(BaseModel):
    system_id: str
    overall_quality_score: float
    dimension_scores: List[DimensionScore]
    total_records: int
    quality_issues_count: int
    sources_below_threshold: int
    data_freshness_status: str
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class DataQualityAgent:
    AGENT_ID = "GL-097"
    AGENT_NAME = "DATA-QUALITY"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"DataQualityAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = DataQualityInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: DataQualityInput) -> DataQualityOutput:
        recommendations = []
        dimension_scores = []

        total_records = sum(s.records_total for s in inp.sources)
        total_complete = sum(s.records_complete for s in inp.sources)
        total_valid = sum(s.records_valid for s in inp.sources)
        total_errors = sum(s.error_count for s in inp.sources)

        # Completeness
        completeness = (total_complete / total_records * 100) if total_records > 0 else 0
        completeness_status = "PASS" if completeness >= 95 else "WARN" if completeness >= 90 else "FAIL"
        dimension_scores.append(DimensionScore(
            dimension="COMPLETENESS",
            score=round(completeness, 1),
            status=completeness_status,
            issues_count=total_records - total_complete
        ))

        # Validity
        validity = (total_valid / total_records * 100) if total_records > 0 else 0
        validity_status = "PASS" if validity >= inp.accuracy_target_pct else "WARN" if validity >= 95 else "FAIL"
        dimension_scores.append(DimensionScore(
            dimension="VALIDITY",
            score=round(validity, 1),
            status=validity_status,
            issues_count=total_records - total_valid
        ))

        # Timeliness
        stale_sources = sum(1 for s in inp.sources if s.last_update_hours > inp.freshness_threshold_hours)
        timeliness = ((len(inp.sources) - stale_sources) / len(inp.sources) * 100) if inp.sources else 0
        timeliness_status = "PASS" if timeliness >= 95 else "WARN" if timeliness >= 80 else "FAIL"
        dimension_scores.append(DimensionScore(
            dimension="TIMELINESS",
            score=round(timeliness, 1),
            status=timeliness_status,
            issues_count=stale_sources
        ))

        # Consistency (based on error rate)
        error_rate = (total_errors / total_records * 100) if total_records > 0 else 0
        consistency = 100 - error_rate
        consistency_status = "PASS" if consistency >= 99 else "WARN" if consistency >= 95 else "FAIL"
        dimension_scores.append(DimensionScore(
            dimension="CONSISTENCY",
            score=round(consistency, 1),
            status=consistency_status,
            issues_count=total_errors
        ))

        # Overall score
        overall = (completeness + validity + timeliness + consistency) / 4

        # Quality issues
        quality_issues = (total_records - total_complete) + (total_records - total_valid) + total_errors

        # Freshness status
        if timeliness >= 95:
            freshness = "CURRENT"
        elif timeliness >= 80:
            freshness = "SLIGHTLY_STALE"
        else:
            freshness = "STALE"

        # Recommendations
        if completeness < 95:
            recommendations.append(f"Completeness {completeness:.1f}% - investigate missing data sources")
        if validity < inp.accuracy_target_pct:
            recommendations.append(f"Validity {validity:.1f}% below {inp.accuracy_target_pct}% target - review validation rules")
        if stale_sources > 0:
            recommendations.append(f"{stale_sources} sources exceed freshness threshold - check data pipelines")
        if total_errors > 0:
            recommendations.append(f"{total_errors} data errors detected - implement error handling")
        if overall < 90:
            recommendations.append("Overall quality below 90% - implement data quality program")

        calc_hash = hashlib.sha256(json.dumps({
            "system": inp.system_id,
            "overall": round(overall, 1),
            "issues": quality_issues
        }).encode()).hexdigest()

        return DataQualityOutput(
            system_id=inp.system_id,
            overall_quality_score=round(overall, 1),
            dimension_scores=dimension_scores,
            total_records=total_records,
            quality_issues_count=quality_issues,
            sources_below_threshold=stale_sources,
            data_freshness_status=freshness,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-097", "name": "DATA-QUALITY", "version": "1.0.0",
    "summary": "Energy data quality management",
    "standards": [{"ref": "ISO 8000"}, {"ref": "DAMA DMBOK"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
