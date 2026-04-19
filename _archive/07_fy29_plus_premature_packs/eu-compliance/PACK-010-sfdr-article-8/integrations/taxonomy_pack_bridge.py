# -*- coding: utf-8 -*-
"""
TaxonomyPackBridge - PACK-008 EU Taxonomy Alignment Integration
================================================================

This module connects PACK-010 (SFDR Article 8) with PACK-008 (EU Taxonomy
Alignment) to provide taxonomy ratio calculations for SFDR disclosures.
It maps SFDR taxonomy disclosure requirements to PACK-008 engine outputs
and handles alignment ratio extraction for both pre-contractual and
periodic disclosure documents.

Architecture:
    PACK-010 SFDR --> TaxonomyPackBridge --> PACK-008 Engines
                          |
                          v
    Annex II/IV <-- Alignment Ratios, Eligibility, Objective Breakdown

Example:
    >>> config = TaxonomyBridgeConfig()
    >>> bridge = TaxonomyPackBridge(config)
    >>> ratio = bridge.get_alignment_ratio(portfolio_holdings)
    >>> print(f"Taxonomy alignment: {ratio['aligned_pct']:.1f}%")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# =============================================================================
# Utility Helpers
# =============================================================================

def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

# =============================================================================
# Agent Stub
# =============================================================================

class _AgentStub:
    """Deferred agent loader for lazy initialization."""

    def __init__(self, agent_id: str, module_path: str, class_name: str) -> None:
        self.agent_id = agent_id
        self.module_path = module_path
        self.class_name = class_name
        self._instance: Optional[Any] = None

    def load(self) -> Any:
        """Load and return the agent instance."""
        if self._instance is not None:
            return self._instance
        try:
            import importlib
            mod = importlib.import_module(self.module_path)
            cls = getattr(mod, self.class_name)
            self._instance = cls()
            return self._instance
        except Exception as exc:
            logger.warning(
                "AgentStub: failed to load %s from %s: %s",
                self.agent_id, self.module_path, exc,
            )
            return None

    @property
    def is_loaded(self) -> bool:
        """Whether the agent has been loaded."""
        return self._instance is not None

# =============================================================================
# Enums
# =============================================================================

class TaxonomyObjective(str, Enum):
    """EU Taxonomy environmental objectives."""
    CLIMATE_CHANGE_MITIGATION = "climate_change_mitigation"
    CLIMATE_CHANGE_ADAPTATION = "climate_change_adaptation"
    WATER_MARINE_RESOURCES = "water_marine_resources"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION_PREVENTION = "pollution_prevention"
    BIODIVERSITY_ECOSYSTEMS = "biodiversity_ecosystems"

class AlignmentMethodology(str, Enum):
    """Methodology for calculating taxonomy alignment."""
    TURNOVER = "turnover"
    CAPEX = "capex"
    OPEX = "opex"

class AlignmentStatus(str, Enum):
    """Alignment status for a holding."""
    ALIGNED = "aligned"
    ELIGIBLE_NOT_ALIGNED = "eligible_not_aligned"
    NOT_ELIGIBLE = "not_eligible"
    DATA_UNAVAILABLE = "data_unavailable"

# =============================================================================
# Data Models
# =============================================================================

class TaxonomyBridgeConfig(BaseModel):
    """Configuration for the Taxonomy Pack Bridge."""
    pack_008_path: str = Field(
        default="packs.eu_compliance.PACK_008_eu_taxonomy_alignment",
        description="Import path for PACK-008",
    )
    enabled_objectives: List[str] = Field(
        default_factory=lambda: [
            TaxonomyObjective.CLIMATE_CHANGE_MITIGATION.value,
            TaxonomyObjective.CLIMATE_CHANGE_ADAPTATION.value,
        ],
        description="Active taxonomy objectives",
    )
    alignment_methodology: AlignmentMethodology = Field(
        default=AlignmentMethodology.TURNOVER,
        description="Primary methodology for alignment calculation",
    )
    use_pack_008: bool = Field(
        default=True,
        description="Whether to use PACK-008 engines or built-in ratios",
    )
    fallback_to_estimates: bool = Field(
        default=True,
        description="Use estimated ratios when PACK-008 is unavailable",
    )
    min_data_quality_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum data quality score to include in calculations",
    )

class AlignmentResult(BaseModel):
    """Result of a taxonomy alignment calculation."""
    aligned_pct: float = Field(default=0.0, description="Taxonomy-aligned %")
    eligible_pct: float = Field(default=0.0, description="Taxonomy-eligible %")
    not_eligible_pct: float = Field(default=0.0, description="Not eligible %")
    data_unavailable_pct: float = Field(
        default=0.0, description="Data unavailable %"
    )
    methodology: str = Field(default="turnover", description="Methodology used")
    holdings_assessed: int = Field(default=0, description="Holdings assessed")
    objective_breakdown: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Per-objective breakdown"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    calculated_at: str = Field(default="", description="Calculation timestamp")
    source: str = Field(
        default="pack_008", description="Data source (pack_008 or estimated)"
    )

class EligibilityResult(BaseModel):
    """Result of taxonomy eligibility vs alignment comparison."""
    total_eligible_pct: float = Field(default=0.0, description="Total eligible %")
    total_aligned_pct: float = Field(default=0.0, description="Total aligned %")
    gap_pct: float = Field(
        default=0.0, description="Gap between eligible and aligned"
    )
    by_objective: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Per-objective eligible/aligned",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# =============================================================================
# Field Mappings
# =============================================================================

FIELD_MAPPINGS: Dict[str, str] = {
    # SFDR field -> PACK-008 field
    "taxonomy_aligned_turnover_pct": "alignment_ratio_turnover",
    "taxonomy_aligned_capex_pct": "alignment_ratio_capex",
    "taxonomy_aligned_opex_pct": "alignment_ratio_opex",
    "taxonomy_eligible_turnover_pct": "eligibility_ratio_turnover",
    "taxonomy_eligible_capex_pct": "eligibility_ratio_capex",
    "taxonomy_eligible_opex_pct": "eligibility_ratio_opex",
    "climate_mitigation_aligned_pct": "objective_ccm_alignment",
    "climate_adaptation_aligned_pct": "objective_cca_alignment",
    "water_aligned_pct": "objective_wtr_alignment",
    "circular_economy_aligned_pct": "objective_ce_alignment",
    "pollution_aligned_pct": "objective_ppc_alignment",
    "biodiversity_aligned_pct": "objective_bio_alignment",
    "climate_mitigation_eligible_pct": "objective_ccm_eligibility",
    "climate_adaptation_eligible_pct": "objective_cca_eligibility",
    "water_eligible_pct": "objective_wtr_eligibility",
    "circular_economy_eligible_pct": "objective_ce_eligibility",
    "pollution_eligible_pct": "objective_ppc_eligibility",
    "biodiversity_eligible_pct": "objective_bio_eligibility",
    "dnsh_compliant_pct": "dnsh_pass_rate",
    "minimum_safeguards_pct": "safeguards_pass_rate",
    "substantial_contribution_pct": "sc_pass_rate",
    "transitional_activities_pct": "transitional_ratio",
    "enabling_activities_pct": "enabling_ratio",
    "sovereign_bonds_pct": "sovereign_exposure",
    "derivatives_pct": "derivatives_exposure",
    "non_eu_exposure_pct": "non_eu_ratio",
    "gas_nuclear_aligned_pct": "complementary_delegated_act_ratio",
    "fossil_gas_turnover_pct": "fossil_gas_alignment",
    "nuclear_turnover_pct": "nuclear_alignment",
    "data_quality_score": "data_quality_index",
    "coverage_pct": "coverage_ratio",
}

# =============================================================================
# Taxonomy Pack Bridge
# =============================================================================

class TaxonomyPackBridge:
    """Bridge connecting SFDR Article 8 disclosures to PACK-008 taxonomy engines.

    Maps SFDR taxonomy disclosure requirements to PACK-008 engine outputs
    and handles alignment ratio extraction for pre-contractual and periodic
    disclosures. Falls back to built-in estimations when PACK-008 is unavailable.

    Attributes:
        config: Bridge configuration.
        _agents: Deferred agent stubs for taxonomy engines.

    Example:
        >>> bridge = TaxonomyPackBridge(TaxonomyBridgeConfig())
        >>> ratio = bridge.get_alignment_ratio(holdings)
        >>> print(f"Aligned: {ratio.aligned_pct:.1f}%")
    """

    def __init__(self, config: Optional[TaxonomyBridgeConfig] = None) -> None:
        """Initialize the Taxonomy Pack Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or TaxonomyBridgeConfig()
        self.logger = logger

        self._agents: Dict[str, _AgentStub] = {
            "alignment_engine": _AgentStub(
                "GL-TAXONOMY-ALIGN",
                f"{self.config.pack_008_path}.engines.alignment_engine",
                "AlignmentEngine",
            ),
            "eligibility_engine": _AgentStub(
                "GL-TAXONOMY-ELIG",
                f"{self.config.pack_008_path}.engines.eligibility_engine",
                "EligibilityEngine",
            ),
            "objective_engine": _AgentStub(
                "GL-TAXONOMY-OBJ",
                f"{self.config.pack_008_path}.engines.objective_engine",
                "ObjectiveEngine",
            ),
            "dnsh_engine": _AgentStub(
                "GL-TAXONOMY-DNSH",
                f"{self.config.pack_008_path}.engines.dnsh_engine",
                "TaxonomyDNSHEngine",
            ),
        }

        self.logger.info(
            "TaxonomyPackBridge initialized: pack_008=%s, objectives=%s, "
            "methodology=%s",
            self.config.pack_008_path,
            self.config.enabled_objectives,
            self.config.alignment_methodology.value,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def get_alignment_ratio(
        self,
        holdings: List[Dict[str, Any]],
        methodology: Optional[AlignmentMethodology] = None,
    ) -> AlignmentResult:
        """Calculate taxonomy alignment ratio for a portfolio.

        Routes the calculation to PACK-008 alignment engine if available,
        otherwise uses built-in deterministic calculation.

        Args:
            holdings: List of portfolio holding records.
            methodology: Override alignment methodology (turnover/capex/opex).

        Returns:
            AlignmentResult with alignment and eligibility percentages.
        """
        method = methodology or self.config.alignment_methodology
        start_time = utcnow()

        # Attempt PACK-008 route
        if self.config.use_pack_008:
            pack_result = self._route_to_pack_008_alignment(holdings, method)
            if pack_result is not None:
                return pack_result

        # Fallback to built-in calculation
        if not self.config.fallback_to_estimates:
            self.logger.error("PACK-008 unavailable and fallback disabled")
            return AlignmentResult(
                calculated_at=start_time.isoformat(),
                source="error",
            )

        self.logger.info(
            "Using built-in taxonomy alignment calculation (%d holdings)",
            len(holdings),
        )
        return self._calculate_alignment_builtin(holdings, method)

    def get_objective_breakdown(
        self,
        holdings: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """Get per-objective alignment and eligibility breakdown.

        Args:
            holdings: List of portfolio holding records.

        Returns:
            Dictionary keyed by objective with aligned/eligible percentages.
        """
        total_weight = sum(float(h.get("weight", 0.0)) for h in holdings)
        if total_weight <= 0:
            return {}

        breakdown: Dict[str, Dict[str, float]] = {}

        for obj_value in self.config.enabled_objectives:
            obj_aligned = 0.0
            obj_eligible = 0.0

            for h in holdings:
                weight = float(h.get("weight", 0.0))
                obj_data = h.get("taxonomy_objectives", {}).get(obj_value, {})

                if obj_data.get("aligned", False):
                    obj_aligned += weight
                elif obj_data.get("eligible", False):
                    obj_eligible += weight

            breakdown[obj_value] = {
                "aligned_pct": round((obj_aligned / total_weight) * 100, 2),
                "eligible_pct": round(
                    ((obj_aligned + obj_eligible) / total_weight) * 100, 2
                ),
                "eligible_not_aligned_pct": round(
                    (obj_eligible / total_weight) * 100, 2
                ),
            }

        self.logger.info(
            "Objective breakdown calculated for %d objectives, %d holdings",
            len(breakdown), len(holdings),
        )
        return breakdown

    def get_eligible_vs_aligned(
        self,
        holdings: List[Dict[str, Any]],
    ) -> EligibilityResult:
        """Compare taxonomy eligibility against alignment.

        Highlights the gap between eligible and aligned percentages to
        identify improvement opportunities.

        Args:
            holdings: List of portfolio holding records.

        Returns:
            EligibilityResult with gap analysis.
        """
        alignment = self.get_alignment_ratio(holdings)
        by_objective = self.get_objective_breakdown(holdings)

        gap = round(alignment.eligible_pct - alignment.aligned_pct, 2)

        result = EligibilityResult(
            total_eligible_pct=alignment.eligible_pct,
            total_aligned_pct=alignment.aligned_pct,
            gap_pct=gap,
            by_objective=by_objective,
        )
        result.provenance_hash = _hash_data(result.model_dump())

        self.logger.info(
            "Eligibility vs aligned: eligible=%.1f%%, aligned=%.1f%%, gap=%.1f%%",
            result.total_eligible_pct, result.total_aligned_pct, result.gap_pct,
        )
        return result

    def route_taxonomy_request(
        self,
        request_type: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route a taxonomy request to the appropriate engine.

        Args:
            request_type: Type of request (alignment, eligibility, dnsh, objective).
            data: Request data including holdings and parameters.

        Returns:
            Response dictionary from the routed engine.
        """
        holdings = data.get("holdings", [])
        methodology = data.get("methodology")

        if request_type == "alignment":
            method = (
                AlignmentMethodology(methodology) if methodology
                else self.config.alignment_methodology
            )
            result = self.get_alignment_ratio(holdings, method)
            return result.model_dump()

        elif request_type == "eligibility":
            result = self.get_eligible_vs_aligned(holdings)
            return result.model_dump()

        elif request_type == "objective_breakdown":
            return self.get_objective_breakdown(holdings)

        elif request_type == "field_mapping":
            sfdr_field = data.get("sfdr_field", "")
            pack_008_field = FIELD_MAPPINGS.get(sfdr_field)
            return {
                "sfdr_field": sfdr_field,
                "pack_008_field": pack_008_field,
                "mapped": pack_008_field is not None,
            }

        else:
            self.logger.warning("Unknown taxonomy request type: %s", request_type)
            return {"error": f"Unknown request type: {request_type}"}

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _route_to_pack_008_alignment(
        self,
        holdings: List[Dict[str, Any]],
        methodology: AlignmentMethodology,
    ) -> Optional[AlignmentResult]:
        """Attempt to route alignment calculation to PACK-008.

        Args:
            holdings: Portfolio holdings.
            methodology: Alignment methodology.

        Returns:
            AlignmentResult from PACK-008, or None if unavailable.
        """
        engine = self._agents["alignment_engine"].load()
        if engine is None:
            self.logger.info("PACK-008 alignment engine not available")
            return None

        try:
            raw_result = engine.calculate_alignment(
                holdings=holdings,
                methodology=methodology.value,
            )
            return self._map_pack_008_result(raw_result, methodology)
        except Exception as exc:
            self.logger.warning(
                "PACK-008 alignment calculation failed: %s", exc
            )
            return None

    def _map_pack_008_result(
        self,
        raw_result: Any,
        methodology: AlignmentMethodology,
    ) -> AlignmentResult:
        """Map PACK-008 raw result to SFDR AlignmentResult.

        Args:
            raw_result: Raw result from PACK-008 engine.
            methodology: Methodology used.

        Returns:
            Mapped AlignmentResult.
        """
        if isinstance(raw_result, dict):
            data = raw_result
        elif hasattr(raw_result, "model_dump"):
            data = raw_result.model_dump()
        else:
            data = {"aligned_pct": 0.0, "eligible_pct": 0.0}

        # Map PACK-008 fields to SFDR fields
        method_key = methodology.value
        aligned = float(
            data.get(f"alignment_ratio_{method_key}",
                     data.get("aligned_pct", 0.0))
        )
        eligible = float(
            data.get(f"eligibility_ratio_{method_key}",
                     data.get("eligible_pct", 0.0))
        )

        result = AlignmentResult(
            aligned_pct=round(aligned, 2),
            eligible_pct=round(eligible, 2),
            not_eligible_pct=round(100.0 - eligible, 2),
            methodology=method_key,
            holdings_assessed=int(data.get("holdings_assessed", 0)),
            calculated_at=utcnow().isoformat(),
            source="pack_008",
        )
        result.provenance_hash = _hash_data(result.model_dump())
        return result

    def _calculate_alignment_builtin(
        self,
        holdings: List[Dict[str, Any]],
        methodology: AlignmentMethodology,
    ) -> AlignmentResult:
        """Built-in taxonomy alignment calculation (deterministic).

        Used when PACK-008 is not available. Calculates alignment ratios
        from holding-level taxonomy flags.

from greenlang.schemas import utcnow

        Args:
            holdings: Portfolio holdings with taxonomy flags.
            methodology: Alignment methodology.

        Returns:
            AlignmentResult with built-in calculation.
        """
        total_weight = sum(float(h.get("weight", 0.0)) for h in holdings)
        if total_weight <= 0:
            return AlignmentResult(
                calculated_at=utcnow().isoformat(),
                source="estimated",
            )

        aligned_weight = 0.0
        eligible_weight = 0.0
        unavailable_weight = 0.0

        for h in holdings:
            weight = float(h.get("weight", 0.0))
            status = self._classify_holding_alignment(h)

            if status == AlignmentStatus.ALIGNED:
                aligned_weight += weight
                eligible_weight += weight
            elif status == AlignmentStatus.ELIGIBLE_NOT_ALIGNED:
                eligible_weight += weight
            elif status == AlignmentStatus.DATA_UNAVAILABLE:
                unavailable_weight += weight

        aligned_pct = round((aligned_weight / total_weight) * 100, 2)
        eligible_pct = round((eligible_weight / total_weight) * 100, 2)
        not_eligible_pct = round(
            100.0 - eligible_pct - (unavailable_weight / total_weight) * 100, 2
        )
        unavailable_pct = round((unavailable_weight / total_weight) * 100, 2)

        objective_breakdown = self.get_objective_breakdown(holdings)

        result = AlignmentResult(
            aligned_pct=aligned_pct,
            eligible_pct=eligible_pct,
            not_eligible_pct=max(not_eligible_pct, 0.0),
            data_unavailable_pct=unavailable_pct,
            methodology=methodology.value,
            holdings_assessed=len(holdings),
            objective_breakdown=objective_breakdown,
            calculated_at=utcnow().isoformat(),
            source="estimated",
        )
        result.provenance_hash = _hash_data(result.model_dump())

        self.logger.info(
            "Built-in alignment: aligned=%.1f%%, eligible=%.1f%%, "
            "not_eligible=%.1f%%, unavailable=%.1f%%",
            aligned_pct, eligible_pct, not_eligible_pct, unavailable_pct,
        )
        return result

    def _classify_holding_alignment(
        self, holding: Dict[str, Any]
    ) -> AlignmentStatus:
        """Classify a single holding's taxonomy alignment status.

        Args:
            holding: Holding record with taxonomy flags.

        Returns:
            AlignmentStatus for the holding.
        """
        taxonomy_aligned = holding.get("taxonomy_aligned")
        taxonomy_eligible = holding.get("taxonomy_eligible")

        if taxonomy_aligned is None and taxonomy_eligible is None:
            return AlignmentStatus.DATA_UNAVAILABLE

        if taxonomy_aligned:
            return AlignmentStatus.ALIGNED

        if taxonomy_eligible:
            return AlignmentStatus.ELIGIBLE_NOT_ALIGNED

        return AlignmentStatus.NOT_ELIGIBLE
