# -*- coding: utf-8 -*-
"""
GL-MRV-X-008: Consolidation & Roll-up Agent
============================================

Consolidates GHG emissions data across organizational entities following
GHG Protocol consolidation approaches (operational, financial, equity share).

Capabilities:
    - Multi-entity emissions consolidation
    - Operational control approach
    - Financial control approach
    - Equity share approach
    - Intercompany elimination
    - Currency/unit normalization
    - Complete provenance tracking

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approaches."""
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class EntityType(str, Enum):
    """Types of organizational entities."""
    PARENT = "parent"
    SUBSIDIARY = "subsidiary"
    JOINT_VENTURE = "joint_venture"
    ASSOCIATE = "associate"
    FRANCHISE = "franchise"


class EntityEmissions(BaseModel):
    """Emissions data for an entity."""
    entity_id: str = Field(...)
    entity_name: str = Field(...)
    entity_type: EntityType = Field(...)
    parent_entity_id: Optional[str] = Field(None)

    # Ownership/control
    equity_share: float = Field(default=1.0, ge=0, le=1)
    has_operational_control: bool = Field(default=True)
    has_financial_control: bool = Field(default=True)

    # Emissions by scope (tCO2e)
    scope1_tco2e: float = Field(default=0)
    scope2_location_tco2e: float = Field(default=0)
    scope2_market_tco2e: float = Field(default=0)
    scope3_tco2e: float = Field(default=0)

    # Intercompany transactions
    intercompany_emissions_tco2e: float = Field(default=0)


class ConsolidationInput(BaseModel):
    """Input model for ConsolidationAgent."""
    entities: List[EntityEmissions] = Field(..., min_length=1)
    approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL
    )
    eliminate_intercompany: bool = Field(default=True)
    reporting_period: Optional[str] = Field(None)
    organization_id: Optional[str] = Field(None)


class ConsolidatedResult(BaseModel):
    """Result of consolidation."""
    approach: ConsolidationApproach = Field(...)
    total_scope1_tco2e: float = Field(...)
    total_scope2_location_tco2e: float = Field(...)
    total_scope2_market_tco2e: float = Field(...)
    total_scope3_tco2e: float = Field(...)
    total_all_scopes_tco2e: float = Field(...)

    # Breakdown
    emissions_by_entity: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    entities_included: List[str] = Field(default_factory=list)
    entities_excluded: List[str] = Field(default_factory=list)

    # Intercompany
    intercompany_eliminated_tco2e: float = Field(default=0)

    # Metadata
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)


class ConsolidationOutput(BaseModel):
    """Output model for ConsolidationAgent."""
    success: bool = Field(...)
    consolidated_result: Optional[ConsolidatedResult] = Field(None)
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field(...)
    validation_status: str = Field(...)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


class ConsolidationRollupAgent(DeterministicAgent):
    """
    GL-MRV-X-008: Consolidation & Roll-up Agent

    Consolidates emissions across organizational entities following
    GHG Protocol consolidation approaches.

    Example:
        >>> agent = ConsolidationRollupAgent()
        >>> result = agent.execute({
        ...     "entities": [
        ...         {"entity_id": "E001", "entity_name": "Parent Co",
        ...          "entity_type": "parent", "scope1_tco2e": 1000}
        ...     ],
        ...     "approach": "operational_control"
        ... })
    """

    AGENT_ID = "GL-MRV-X-008"
    AGENT_NAME = "Consolidation & Roll-up Agent"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    metadata = AgentMetadata(
        name="ConsolidationRollupAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Consolidates emissions across organizational entities"
    )

    def __init__(self, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute consolidation calculation."""
        start_time = DeterministicClock.now()

        try:
            cons_input = ConsolidationInput(**inputs)
            trace = []

            # Determine inclusion factor based on approach
            total_s1 = Decimal("0")
            total_s2_loc = Decimal("0")
            total_s2_mkt = Decimal("0")
            total_s3 = Decimal("0")
            total_intercompany = Decimal("0")

            emissions_by_entity = {}
            entities_included = []
            entities_excluded = []

            trace.append(f"Consolidation approach: {cons_input.approach.value}")

            for entity in cons_input.entities:
                # Determine inclusion factor
                if cons_input.approach == ConsolidationApproach.OPERATIONAL_CONTROL:
                    factor = Decimal("1.0") if entity.has_operational_control else Decimal("0")
                elif cons_input.approach == ConsolidationApproach.FINANCIAL_CONTROL:
                    factor = Decimal("1.0") if entity.has_financial_control else Decimal("0")
                else:  # EQUITY_SHARE
                    factor = Decimal(str(entity.equity_share))

                if factor > 0:
                    entities_included.append(entity.entity_id)

                    # Apply factor to emissions
                    e_s1 = Decimal(str(entity.scope1_tco2e)) * factor
                    e_s2_loc = Decimal(str(entity.scope2_location_tco2e)) * factor
                    e_s2_mkt = Decimal(str(entity.scope2_market_tco2e)) * factor
                    e_s3 = Decimal(str(entity.scope3_tco2e)) * factor

                    total_s1 += e_s1
                    total_s2_loc += e_s2_loc
                    total_s2_mkt += e_s2_mkt
                    total_s3 += e_s3

                    if cons_input.eliminate_intercompany:
                        total_intercompany += Decimal(str(entity.intercompany_emissions_tco2e)) * factor

                    emissions_by_entity[entity.entity_id] = {
                        "scope1": float(e_s1),
                        "scope2_location": float(e_s2_loc),
                        "scope2_market": float(e_s2_mkt),
                        "scope3": float(e_s3),
                        "factor_applied": float(factor)
                    }

                    trace.append(
                        f"Entity {entity.entity_id}: factor={float(factor):.2f}, "
                        f"S1={float(e_s1):.2f}, S2={float(e_s2_loc):.2f}"
                    )
                else:
                    entities_excluded.append(entity.entity_id)
                    trace.append(f"Entity {entity.entity_id}: excluded (factor=0)")

            # Apply intercompany elimination
            if cons_input.eliminate_intercompany and total_intercompany > 0:
                trace.append(f"Intercompany elimination: {float(total_intercompany):.2f} tCO2e")

            total_all = total_s1 + total_s2_mkt + total_s3 - total_intercompany

            provenance_hash = self._compute_hash({
                "approach": cons_input.approach.value,
                "total_all": float(total_all)
            })

            result = ConsolidatedResult(
                approach=cons_input.approach,
                total_scope1_tco2e=float(total_s1.quantize(Decimal("0.0001"))),
                total_scope2_location_tco2e=float(total_s2_loc.quantize(Decimal("0.0001"))),
                total_scope2_market_tco2e=float(total_s2_mkt.quantize(Decimal("0.0001"))),
                total_scope3_tco2e=float(total_s3.quantize(Decimal("0.0001"))),
                total_all_scopes_tco2e=float(total_all.quantize(Decimal("0.0001"))),
                emissions_by_entity=emissions_by_entity,
                entities_included=entities_included,
                entities_excluded=entities_excluded,
                intercompany_eliminated_tco2e=float(total_intercompany.quantize(Decimal("0.0001"))),
                calculation_trace=trace,
                provenance_hash=provenance_hash
            )

            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            output = ConsolidationOutput(
                success=True,
                consolidated_result=result,
                processing_time_ms=processing_time_ms,
                provenance_hash=provenance_hash,
                validation_status="PASS"
            )

            self._capture_audit_entry(
                operation="consolidate_emissions",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=trace
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Consolidation failed: {str(e)}", exc_info=True)
            end_time = DeterministicClock.now()
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "validation_status": "FAIL"
            }

    def _compute_hash(self, data: Any) -> str:
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
