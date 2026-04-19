# -*- coding: utf-8 -*-
"""
GL-MRV-X-010: Inventory Boundary Agent
=======================================

Manages organizational and operational boundaries for GHG inventory
following GHG Protocol standards.

Capabilities:
    - Organizational boundary definition
    - Operational boundary scoping
    - Exclusion management and justification
    - Boundary change tracking
    - Materiality assessment
    - Complete provenance tracking

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


class BoundaryApproach(str, Enum):
    """Organizational boundary approaches."""
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class FacilityType(str, Enum):
    """Types of facilities."""
    MANUFACTURING = "manufacturing"
    OFFICE = "office"
    WAREHOUSE = "warehouse"
    DATA_CENTER = "data_center"
    RETAIL = "retail"
    TRANSPORT_HUB = "transport_hub"


class ExclusionReason(str, Enum):
    """Reasons for boundary exclusions."""
    DE_MINIMIS = "de_minimis"
    NOT_RELEVANT = "not_relevant"
    DATA_UNAVAILABLE = "data_unavailable"
    REGULATORY_EXEMPTION = "regulatory_exemption"
    TEMPORARY = "temporary"


class Facility(BaseModel):
    """A facility within the inventory boundary."""
    facility_id: str = Field(...)
    facility_name: str = Field(...)
    facility_type: FacilityType = Field(...)
    country: str = Field(...)
    region: Optional[str] = Field(None)
    address: Optional[str] = Field(None)
    operational_control: bool = Field(default=True)
    financial_control: bool = Field(default=True)
    equity_share: float = Field(default=1.0, ge=0, le=1)
    floor_area_sqm: Optional[float] = Field(None)
    employees: Optional[int] = Field(None)
    included: bool = Field(default=True)
    exclusion_reason: Optional[ExclusionReason] = Field(None)
    exclusion_justification: Optional[str] = Field(None)


class BoundaryScope(BaseModel):
    """Scope-level boundary definition."""
    scope: str = Field(..., description="scope1, scope2, or scope3")
    included: bool = Field(default=True)
    categories_included: List[str] = Field(default_factory=list)
    categories_excluded: List[str] = Field(default_factory=list)
    exclusion_justifications: Dict[str, str] = Field(default_factory=dict)


class InventoryBoundaryInput(BaseModel):
    """Input model for InventoryBoundaryAgent."""
    boundary_approach: BoundaryApproach = Field(
        default=BoundaryApproach.OPERATIONAL_CONTROL
    )
    facilities: List[Facility] = Field(default_factory=list)
    scope_boundaries: List[BoundaryScope] = Field(default_factory=list)
    materiality_threshold_pct: float = Field(default=5.0)
    organization_id: Optional[str] = Field(None)
    reporting_year: Optional[int] = Field(None)


class BoundaryAssessment(BaseModel):
    """Assessment of inventory boundary."""
    total_facilities: int = Field(...)
    included_facilities: int = Field(...)
    excluded_facilities: int = Field(...)
    coverage_percentage: float = Field(...)
    facilities_by_type: Dict[str, int] = Field(default_factory=dict)
    facilities_by_country: Dict[str, int] = Field(default_factory=dict)
    exclusions_summary: Dict[str, int] = Field(default_factory=dict)
    scope_coverage: Dict[str, bool] = Field(default_factory=dict)
    materiality_assessment: Dict[str, Any] = Field(default_factory=dict)
    boundary_completeness: str = Field(...)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)


class InventoryBoundaryOutput(BaseModel):
    """Output model for InventoryBoundaryAgent."""
    success: bool = Field(...)
    boundary_assessment: Optional[BoundaryAssessment] = Field(None)
    included_facilities: List[str] = Field(default_factory=list)
    excluded_facilities: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field(...)
    validation_status: str = Field(...)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


class InventoryBoundaryAgent(DeterministicAgent):
    """
    GL-MRV-X-010: Inventory Boundary Agent

    Manages organizational and operational boundaries for GHG inventory.

    Example:
        >>> agent = InventoryBoundaryAgent()
        >>> result = agent.execute({
        ...     "boundary_approach": "operational_control",
        ...     "facilities": [
        ...         {"facility_id": "F001", "facility_name": "HQ",
        ...          "facility_type": "office", "country": "US",
        ...          "operational_control": True}
        ...     ]
        ... })
    """

    AGENT_ID = "GL-MRV-X-010"
    AGENT_NAME = "Inventory Boundary Agent"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    metadata = AgentMetadata(
        name="InventoryBoundaryAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Manages organizational boundaries for GHG inventory"
    )

    def __init__(self, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute boundary assessment."""
        start_time = DeterministicClock.now()

        try:
            boundary_input = InventoryBoundaryInput(**inputs)

            # Assess facilities
            included = []
            excluded = []
            facilities_by_type: Dict[str, int] = {}
            facilities_by_country: Dict[str, int] = {}
            exclusions_summary: Dict[str, int] = {}

            for facility in boundary_input.facilities:
                # Check inclusion based on approach
                is_included = self._check_inclusion(
                    facility, boundary_input.boundary_approach
                )

                if is_included and facility.included:
                    included.append(facility.facility_id)
                    facilities_by_type[facility.facility_type.value] = (
                        facilities_by_type.get(facility.facility_type.value, 0) + 1
                    )
                    facilities_by_country[facility.country] = (
                        facilities_by_country.get(facility.country, 0) + 1
                    )
                else:
                    excluded.append({
                        "facility_id": facility.facility_id,
                        "reason": facility.exclusion_reason.value if facility.exclusion_reason else "no_control",
                        "justification": facility.exclusion_justification
                    })
                    if facility.exclusion_reason:
                        exclusions_summary[facility.exclusion_reason.value] = (
                            exclusions_summary.get(facility.exclusion_reason.value, 0) + 1
                        )

            # Calculate coverage
            total = len(boundary_input.facilities)
            coverage = (len(included) / total * 100) if total > 0 else 0

            # Scope coverage
            scope_coverage = {
                "scope1": True,
                "scope2": True,
                "scope3": False  # Typically optional
            }
            for sb in boundary_input.scope_boundaries:
                scope_coverage[sb.scope] = sb.included

            # Assess completeness
            if coverage >= 95:
                completeness = "complete"
            elif coverage >= 80:
                completeness = "substantial"
            elif coverage >= 50:
                completeness = "partial"
            else:
                completeness = "incomplete"

            # Generate recommendations
            recommendations = []
            if coverage < 95:
                recommendations.append(f"Consider including additional facilities to improve coverage from {coverage:.1f}%")
            if not scope_coverage.get("scope3", False):
                recommendations.append("Consider including material Scope 3 categories")

            provenance_hash = self._compute_hash({
                "approach": boundary_input.boundary_approach.value,
                "coverage": coverage,
                "included_count": len(included)
            })

            assessment = BoundaryAssessment(
                total_facilities=total,
                included_facilities=len(included),
                excluded_facilities=len(excluded),
                coverage_percentage=round(coverage, 2),
                facilities_by_type=facilities_by_type,
                facilities_by_country=facilities_by_country,
                exclusions_summary=exclusions_summary,
                scope_coverage=scope_coverage,
                materiality_assessment={"threshold_pct": boundary_input.materiality_threshold_pct},
                boundary_completeness=completeness,
                recommendations=recommendations,
                provenance_hash=provenance_hash
            )

            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            output = InventoryBoundaryOutput(
                success=True,
                boundary_assessment=assessment,
                included_facilities=included,
                excluded_facilities=excluded,
                processing_time_ms=processing_time_ms,
                provenance_hash=provenance_hash,
                validation_status="PASS"
            )

            self._capture_audit_entry(
                operation="assess_boundary",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=[f"Assessed {total} facilities, {len(included)} included"]
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Boundary assessment failed: {str(e)}", exc_info=True)
            end_time = DeterministicClock.now()
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "validation_status": "FAIL"
            }

    def _check_inclusion(
        self,
        facility: Facility,
        approach: BoundaryApproach
    ) -> bool:
        """Check if facility should be included based on approach."""
        if approach == BoundaryApproach.OPERATIONAL_CONTROL:
            return facility.operational_control
        elif approach == BoundaryApproach.FINANCIAL_CONTROL:
            return facility.financial_control
        else:  # EQUITY_SHARE
            return facility.equity_share > 0

    def _compute_hash(self, data: Any) -> str:
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
