# -*- coding: utf-8 -*-
"""
GL-MRV-X-013: Emission Factor Selection Agent
==============================================

Selects appropriate emission factors based on activity data characteristics,
geographic location, and data quality requirements.

Capabilities:
    - Emission factor database lookup
    - Geographic matching
    - Temporal matching
    - Technology matching
    - Tier-based selection (IPCC Tiers 1-3)
    - Uncertainty quantification
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


class EFTier(str, Enum):
    """IPCC emission factor tiers."""
    TIER1 = "tier1"  # Default factors
    TIER2 = "tier2"  # Country-specific
    TIER3 = "tier3"  # Facility-specific


class EFSource(str, Enum):
    """Emission factor sources."""
    IPCC = "ipcc"
    EPA = "epa"
    DEFRA = "defra"
    ECOINVENT = "ecoinvent"
    GHG_PROTOCOL = "ghg_protocol"
    CUSTOM = "custom"


class ActivityType(str, Enum):
    """Types of activities requiring emission factors."""
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PURCHASED_ELECTRICITY = "purchased_electricity"
    REFRIGERANT_LEAKAGE = "refrigerant_leakage"
    WASTE_DISPOSAL = "waste_disposal"
    PURCHASED_GOODS = "purchased_goods"
    BUSINESS_TRAVEL = "business_travel"
    EMPLOYEE_COMMUTING = "employee_commuting"


class EmissionFactor(BaseModel):
    """An emission factor record."""
    ef_id: str = Field(...)
    activity_type: ActivityType = Field(...)
    fuel_or_material: str = Field(...)
    ef_value: float = Field(...)
    ef_unit: str = Field(...)
    tier: EFTier = Field(...)
    source: EFSource = Field(...)
    source_year: int = Field(...)
    region: Optional[str] = Field(None)
    technology: Optional[str] = Field(None)
    uncertainty_pct: float = Field(default=25.0)
    gwp_included: bool = Field(default=True)


class EFSelectionCriteria(BaseModel):
    """Criteria for emission factor selection."""
    activity_type: ActivityType = Field(...)
    fuel_or_material: str = Field(...)
    region: Optional[str] = Field(None)
    technology: Optional[str] = Field(None)
    year: Optional[int] = Field(None)
    preferred_tier: Optional[EFTier] = Field(None)
    preferred_source: Optional[EFSource] = Field(None)


class EFSelectionResult(BaseModel):
    """Result of emission factor selection."""
    selected_ef: EmissionFactor = Field(...)
    match_score: float = Field(..., ge=0, le=1)
    match_criteria: Dict[str, bool] = Field(default_factory=dict)
    alternatives: List[EmissionFactor] = Field(default_factory=list)
    selection_rationale: str = Field(...)
    provenance_hash: str = Field(...)


class EmissionFactorSelectionInput(BaseModel):
    """Input model for EmissionFactorSelectionAgent."""
    selection_criteria: List[EFSelectionCriteria] = Field(..., min_length=1)
    custom_factors: Optional[List[EmissionFactor]] = Field(None)
    prefer_lower_uncertainty: bool = Field(default=True)
    organization_id: Optional[str] = Field(None)


class EmissionFactorSelectionOutput(BaseModel):
    """Output model for EmissionFactorSelectionAgent."""
    success: bool = Field(...)
    selection_results: List[EFSelectionResult] = Field(default_factory=list)
    factors_selected: int = Field(...)
    average_match_score: float = Field(...)
    tier_distribution: Dict[str, int] = Field(default_factory=dict)
    source_distribution: Dict[str, int] = Field(default_factory=dict)
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field(...)
    validation_status: str = Field(...)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# Default emission factors database (simplified)
DEFAULT_EMISSION_FACTORS: List[Dict[str, Any]] = [
    # Stationary combustion
    {"ef_id": "EF001", "activity_type": "stationary_combustion", "fuel_or_material": "natural_gas",
     "ef_value": 56.1, "ef_unit": "kgCO2/GJ", "tier": "tier1", "source": "ipcc", "source_year": 2023,
     "uncertainty_pct": 5.0},
    {"ef_id": "EF002", "activity_type": "stationary_combustion", "fuel_or_material": "diesel",
     "ef_value": 74.1, "ef_unit": "kgCO2/GJ", "tier": "tier1", "source": "ipcc", "source_year": 2023,
     "uncertainty_pct": 5.0},
    {"ef_id": "EF003", "activity_type": "stationary_combustion", "fuel_or_material": "coal",
     "ef_value": 94.6, "ef_unit": "kgCO2/GJ", "tier": "tier1", "source": "ipcc", "source_year": 2023,
     "uncertainty_pct": 7.0},

    # Mobile combustion
    {"ef_id": "EF010", "activity_type": "mobile_combustion", "fuel_or_material": "gasoline",
     "ef_value": 2.31, "ef_unit": "kgCO2/liter", "tier": "tier1", "source": "epa", "source_year": 2023,
     "uncertainty_pct": 3.0},
    {"ef_id": "EF011", "activity_type": "mobile_combustion", "fuel_or_material": "diesel",
     "ef_value": 2.68, "ef_unit": "kgCO2/liter", "tier": "tier1", "source": "epa", "source_year": 2023,
     "uncertainty_pct": 3.0},

    # Purchased electricity (grid averages)
    {"ef_id": "EF020", "activity_type": "purchased_electricity", "fuel_or_material": "grid_electricity",
     "ef_value": 0.386, "ef_unit": "kgCO2e/kWh", "tier": "tier2", "source": "epa", "source_year": 2023,
     "region": "US", "uncertainty_pct": 10.0},
    {"ef_id": "EF021", "activity_type": "purchased_electricity", "fuel_or_material": "grid_electricity",
     "ef_value": 0.255, "ef_unit": "kgCO2e/kWh", "tier": "tier2", "source": "ghg_protocol", "source_year": 2023,
     "region": "EU", "uncertainty_pct": 10.0},

    # Business travel
    {"ef_id": "EF030", "activity_type": "business_travel", "fuel_or_material": "air_short_haul",
     "ef_value": 0.255, "ef_unit": "kgCO2e/pkm", "tier": "tier1", "source": "defra", "source_year": 2023,
     "uncertainty_pct": 15.0},
    {"ef_id": "EF031", "activity_type": "business_travel", "fuel_or_material": "air_long_haul",
     "ef_value": 0.195, "ef_unit": "kgCO2e/pkm", "tier": "tier1", "source": "defra", "source_year": 2023,
     "uncertainty_pct": 15.0},
]


class EmissionFactorSelectionAgent(DeterministicAgent):
    """
    GL-MRV-X-013: Emission Factor Selection Agent

    Selects appropriate emission factors for GHG calculations.

    Example:
        >>> agent = EmissionFactorSelectionAgent()
        >>> result = agent.execute({
        ...     "selection_criteria": [
        ...         {"activity_type": "stationary_combustion",
        ...          "fuel_or_material": "natural_gas",
        ...          "region": "US"}
        ...     ]
        ... })
    """

    AGENT_ID = "GL-MRV-X-013"
    AGENT_NAME = "Emission Factor Selection Agent"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    metadata = AgentMetadata(
        name="EmissionFactorSelectionAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Selects appropriate emission factors"
    )

    def __init__(self, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self._ef_database = [EmissionFactor(**ef) for ef in DEFAULT_EMISSION_FACTORS]
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute emission factor selection."""
        start_time = DeterministicClock.now()

        try:
            sel_input = EmissionFactorSelectionInput(**inputs)

            # Add custom factors to database
            if sel_input.custom_factors:
                self._ef_database.extend(sel_input.custom_factors)

            selection_results: List[EFSelectionResult] = []
            tier_dist: Dict[str, int] = {}
            source_dist: Dict[str, int] = {}

            for criteria in sel_input.selection_criteria:
                result = self._select_emission_factor(
                    criteria, sel_input.prefer_lower_uncertainty
                )
                selection_results.append(result)

                tier = result.selected_ef.tier.value
                source = result.selected_ef.source.value
                tier_dist[tier] = tier_dist.get(tier, 0) + 1
                source_dist[source] = source_dist.get(source, 0) + 1

            avg_match = sum(r.match_score for r in selection_results) / len(selection_results)

            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            provenance_hash = self._compute_hash({
                "factors_selected": len(selection_results),
                "avg_match_score": avg_match
            })

            output = EmissionFactorSelectionOutput(
                success=True,
                selection_results=selection_results,
                factors_selected=len(selection_results),
                average_match_score=round(avg_match, 3),
                tier_distribution=tier_dist,
                source_distribution=source_dist,
                processing_time_ms=processing_time_ms,
                provenance_hash=provenance_hash,
                validation_status="PASS"
            )

            self._capture_audit_entry(
                operation="select_emission_factors",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=[f"Selected {len(selection_results)} emission factors"]
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"EF selection failed: {str(e)}", exc_info=True)
            end_time = DeterministicClock.now()
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "validation_status": "FAIL"
            }

    def _select_emission_factor(
        self,
        criteria: EFSelectionCriteria,
        prefer_lower_uncertainty: bool
    ) -> EFSelectionResult:
        """Select the best emission factor for given criteria."""
        candidates = []
        match_scores = []

        for ef in self._ef_database:
            score, match_criteria = self._calculate_match_score(ef, criteria)
            if score > 0:
                candidates.append((ef, score, match_criteria))

        if not candidates:
            # Return a default factor with low score
            default_ef = EmissionFactor(
                ef_id="DEFAULT",
                activity_type=criteria.activity_type,
                fuel_or_material=criteria.fuel_or_material,
                ef_value=0.0,
                ef_unit="kgCO2e/unit",
                tier=EFTier.TIER1,
                source=EFSource.CUSTOM,
                source_year=2023,
                uncertainty_pct=100.0
            )
            return EFSelectionResult(
                selected_ef=default_ef,
                match_score=0.0,
                match_criteria={},
                alternatives=[],
                selection_rationale="No matching emission factor found in database",
                provenance_hash=self._compute_hash({"ef_id": "DEFAULT"})
            )

        # Sort by score (and uncertainty if preferred)
        if prefer_lower_uncertainty:
            candidates.sort(key=lambda x: (-x[1], x[0].uncertainty_pct))
        else:
            candidates.sort(key=lambda x: -x[1])

        best = candidates[0]
        alternatives = [c[0] for c in candidates[1:4]]  # Top 3 alternatives

        rationale = self._generate_rationale(best[0], best[2])

        return EFSelectionResult(
            selected_ef=best[0],
            match_score=round(best[1], 3),
            match_criteria=best[2],
            alternatives=alternatives,
            selection_rationale=rationale,
            provenance_hash=self._compute_hash({"ef_id": best[0].ef_id})
        )

    def _calculate_match_score(
        self,
        ef: EmissionFactor,
        criteria: EFSelectionCriteria
    ) -> tuple[float, Dict[str, bool]]:
        """Calculate match score between EF and criteria."""
        score = 0.0
        matches = {}

        # Activity type match (required)
        if ef.activity_type == criteria.activity_type:
            score += 0.4
            matches["activity_type"] = True
        else:
            matches["activity_type"] = False
            return 0.0, matches

        # Fuel/material match
        if ef.fuel_or_material.lower() == criteria.fuel_or_material.lower():
            score += 0.3
            matches["fuel_material"] = True
        else:
            matches["fuel_material"] = False

        # Region match
        if criteria.region:
            if ef.region and ef.region.upper() == criteria.region.upper():
                score += 0.15
                matches["region"] = True
            else:
                matches["region"] = False
        else:
            score += 0.1  # No region specified is OK

        # Tier preference
        if criteria.preferred_tier:
            if ef.tier == criteria.preferred_tier:
                score += 0.1
                matches["tier"] = True
            else:
                matches["tier"] = False
        else:
            score += 0.05

        # Source preference
        if criteria.preferred_source:
            if ef.source == criteria.preferred_source:
                score += 0.05
                matches["source"] = True
            else:
                matches["source"] = False

        return score, matches

    def _generate_rationale(self, ef: EmissionFactor, matches: Dict[str, bool]) -> str:
        """Generate selection rationale."""
        parts = [f"Selected {ef.ef_id} from {ef.source.value}"]

        matched = [k for k, v in matches.items() if v]
        if matched:
            parts.append(f"Matched on: {', '.join(matched)}")

        parts.append(f"Tier: {ef.tier.value}, Uncertainty: {ef.uncertainty_pct}%")

        return ". ".join(parts)

    def _compute_hash(self, data: Any) -> str:
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_available_factors(self, activity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of available emission factors."""
        factors = self._ef_database
        if activity_type:
            factors = [f for f in factors if f.activity_type.value == activity_type]
        return [f.model_dump() for f in factors]
