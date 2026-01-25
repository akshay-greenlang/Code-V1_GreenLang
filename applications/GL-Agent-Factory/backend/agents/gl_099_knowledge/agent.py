"""GL-099: Knowledge Management Agent (KNOWLEDGE).

Manages organizational knowledge for energy programs.

Standards: ISO 30401, Knowledge Management Best Practices
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class KnowledgeType(str, Enum):
    TECHNICAL = "TECHNICAL"
    OPERATIONAL = "OPERATIONAL"
    REGULATORY = "REGULATORY"
    LESSONS_LEARNED = "LESSONS_LEARNED"
    BEST_PRACTICES = "BEST_PRACTICES"


class KnowledgeAsset(BaseModel):
    asset_id: str
    title: str
    knowledge_type: KnowledgeType
    created_date: datetime
    last_reviewed: datetime
    access_count: int = Field(default=0, ge=0)
    quality_score: float = Field(default=0, ge=0, le=100)
    owner: str = Field(default="Unknown")


class KnowledgeInput(BaseModel):
    organization_id: str
    knowledge_assets: List[KnowledgeAsset] = Field(default_factory=list)
    active_users: int = Field(default=100, ge=0)
    content_requests_month: int = Field(default=500, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeMetric(BaseModel):
    knowledge_type: str
    asset_count: int
    avg_quality: float
    avg_age_days: float
    utilization_rate: float


class KnowledgeOutput(BaseModel):
    organization_id: str
    total_assets: int
    knowledge_metrics: List[KnowledgeMetric]
    overall_health_score: float
    coverage_gaps: List[str]
    stale_content_count: int
    high_value_assets: List[str]
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class KnowledgeAgent:
    AGENT_ID = "GL-099"
    AGENT_NAME = "KNOWLEDGE"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"KnowledgeAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = KnowledgeInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: KnowledgeInput) -> KnowledgeOutput:
        recommendations = []
        metrics = []
        now = datetime.utcnow()

        # Group by type
        by_type = {}
        for asset in inp.knowledge_assets:
            kt = asset.knowledge_type.value
            if kt not in by_type:
                by_type[kt] = []
            by_type[kt].append(asset)

        total_access = sum(a.access_count for a in inp.knowledge_assets)
        stale_count = 0
        high_value = []

        for kt, assets in by_type.items():
            count = len(assets)
            avg_quality = sum(a.quality_score for a in assets) / count if count > 0 else 0
            avg_age = sum((now - a.last_reviewed).days for a in assets) / count if count > 0 else 0
            type_access = sum(a.access_count for a in assets)
            utilization = (type_access / total_access * 100) if total_access > 0 else 0

            metrics.append(KnowledgeMetric(
                knowledge_type=kt,
                asset_count=count,
                avg_quality=round(avg_quality, 1),
                avg_age_days=round(avg_age, 0),
                utilization_rate=round(utilization, 1)
            ))

            # Stale content (>1 year since review)
            stale = [a for a in assets if (now - a.last_reviewed).days > 365]
            stale_count += len(stale)

            # High value (high quality + high access)
            for a in assets:
                if a.quality_score >= 80 and a.access_count >= 50:
                    high_value.append(a.title)

        # Coverage gaps
        all_types = set(kt.value for kt in KnowledgeType)
        covered = set(by_type.keys())
        gaps = list(all_types - covered)

        # Overall health score
        if inp.knowledge_assets:
            avg_quality = sum(a.quality_score for a in inp.knowledge_assets) / len(inp.knowledge_assets)
            stale_penalty = (stale_count / len(inp.knowledge_assets)) * 20
            gap_penalty = len(gaps) * 5
            health = max(0, avg_quality - stale_penalty - gap_penalty)
        else:
            health = 0

        # Recommendations
        if gaps:
            recommendations.append(f"Knowledge gaps in: {', '.join(gaps)}")
        if stale_count > 0:
            recommendations.append(f"{stale_count} assets need review (>1 year old)")
        if health < 70:
            recommendations.append("Knowledge health below 70% - implement content improvement program")

        low_quality = [a for a in inp.knowledge_assets if a.quality_score < 50]
        if low_quality:
            recommendations.append(f"{len(low_quality)} low-quality assets need improvement")

        if inp.active_users > 0 and inp.content_requests_month / inp.active_users < 2:
            recommendations.append("Low engagement - improve knowledge discoverability")

        calc_hash = hashlib.sha256(json.dumps({
            "organization": inp.organization_id,
            "assets": len(inp.knowledge_assets),
            "health": round(health, 1)
        }).encode()).hexdigest()

        return KnowledgeOutput(
            organization_id=inp.organization_id,
            total_assets=len(inp.knowledge_assets),
            knowledge_metrics=metrics,
            overall_health_score=round(health, 1),
            coverage_gaps=gaps,
            stale_content_count=stale_count,
            high_value_assets=high_value[:5],
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-099", "name": "KNOWLEDGE", "version": "1.0.0",
    "summary": "Knowledge management for energy organizations",
    "standards": [{"ref": "ISO 30401"}, {"ref": "Knowledge Management Best Practices"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
