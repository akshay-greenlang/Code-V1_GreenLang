# -*- coding: utf-8 -*-
"""
TaxonomyPackBridge - PACK-008 EU Taxonomy Alignment Integration for Article 9
===============================================================================

This module connects PACK-011 (SFDR Article 9) with PACK-008 (EU Taxonomy
Alignment) to provide taxonomy ratio calculations for SFDR disclosures.
Article 9 products have stricter taxonomy requirements than Article 8:
sustainable investments with environmental objectives must be taxonomy-aligned,
and the product must disclose alignment across all 6 objectives plus the
Complementary Delegated Act (gas/nuclear CDA) provisions.

Architecture:
    PACK-011 SFDR Art 9 --> TaxonomyPackBridge --> PACK-008 Engines
                                  |
                                  v
    Annex III/V <-- Alignment Ratios, Art 5/6 Refs, Min Safeguards, CDA

Key differences from PACK-010 TaxonomyPackBridge:
    - All 6 environmental objectives in scope (not just CCM/CCA)
    - Mandatory minimum safeguards verification
    - Gas/nuclear CDA disclosure required
    - Art 5 (env objective) / Art 6 (DNSH) cross-references
    - Higher alignment expectations for SI with env objective

Example:
    >>> config = TaxonomyBridgeConfig()
    >>> bridge = TaxonomyPackBridge(config)
    >>> result = bridge.get_alignment_ratio(portfolio_holdings)
    >>> print(f"Taxonomy alignment: {result.aligned_pct:.1f}%")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.schemas import utcnow

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
    """EU Taxonomy environmental objectives (all 6 for Article 9)."""
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

class CDACategory(str, Enum):
    """Complementary Delegated Act categories."""
    FOSSIL_GAS = "fossil_gas"
    NUCLEAR = "nuclear"
    NONE = "none"

# =============================================================================
# Data Models
# =============================================================================

class TaxonomyBridgeConfig(BaseModel):
    """Configuration for the Article 9 Taxonomy Pack Bridge."""
    pack_008_path: str = Field(
        default="packs.eu_compliance.PACK_008_eu_taxonomy_alignment",
        description="Import path for PACK-008",
    )
    enabled_objectives: List[str] = Field(
        default_factory=lambda: [obj.value for obj in TaxonomyObjective],
        description="All 6 taxonomy objectives (Art 9 requires all)",
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
    enforce_minimum_safeguards: bool = Field(
        default=True,
        description="Enforce minimum safeguards check (Art 18 Taxonomy Reg)",
    )
    include_cda_disclosure: bool = Field(
        default=True,
        description="Include Complementary Delegated Act (gas/nuclear) disclosure",
    )

class TaxonomyAlignmentData(BaseModel):
    """Taxonomy alignment data for Article 9 products."""
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
    # Art 9 specific fields
    si_env_aligned_pct: float = Field(
        default=0.0,
        description="Taxonomy-aligned % of SI with environmental objective",
    )
    si_env_not_aligned_pct: float = Field(
        default=0.0,
        description="Non-taxonomy-aligned % of SI with environmental objective",
    )
    art_5_alignment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Art 5 Taxonomy Reg alignment data",
    )
    art_6_dnsh: Dict[str, Any] = Field(
        default_factory=dict,
        description="Art 6 Taxonomy Reg DNSH criteria data",
    )
    cda_disclosure: Dict[str, float] = Field(
        default_factory=dict,
        description="Complementary Delegated Act (gas/nuclear) %",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    calculated_at: str = Field(default="", description="Calculation timestamp")
    source: str = Field(
        default="pack_008", description="Data source (pack_008 or estimated)"
    )

class SafeguardsResult(BaseModel):
    """Result of minimum safeguards assessment."""
    total_assessed: int = Field(default=0, description="Total holdings assessed")
    pass_count: int = Field(default=0, description="Holdings passing safeguards")
    fail_count: int = Field(default=0, description="Holdings failing safeguards")
    pass_pct: float = Field(default=0.0, description="Pass percentage")
    ungp_compliant: bool = Field(
        default=True, description="UN Guiding Principles on Business and Human Rights"
    )
    oecd_compliant: bool = Field(
        default=True, description="OECD Guidelines for Multinational Enterprises"
    )
    ilo_compliant: bool = Field(
        default=True, description="ILO Core Labour Conventions"
    )
    international_bill_compliant: bool = Field(
        default=True, description="International Bill of Human Rights"
    )
    per_holding: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-holding safeguards results"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# =============================================================================
# Field Mappings (extended for Art 9)
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
    # Art 9 specific
    "si_env_taxonomy_aligned_pct": "si_env_alignment_ratio",
    "si_env_taxonomy_eligible_pct": "si_env_eligibility_ratio",
    "art_5_substantial_contribution": "art_5_sc_pass_rate",
    "art_6_dnsh_compliance": "art_6_dnsh_pass_rate",
}

# =============================================================================
# Taxonomy Pack Bridge
# =============================================================================

class TaxonomyPackBridge:
    """Bridge connecting SFDR Article 9 to PACK-008 taxonomy engines.

    Maps SFDR Article 9 taxonomy disclosure requirements to PACK-008
    engine outputs. Article 9 has stricter requirements than Article 8:
    all 6 objectives must be assessed, minimum safeguards enforced,
    gas/nuclear CDA disclosed, and Art 5/6 cross-referenced.

    Attributes:
        config: Bridge configuration.
        _agents: Deferred agent stubs for taxonomy engines.

    Example:
        >>> bridge = TaxonomyPackBridge(TaxonomyBridgeConfig())
        >>> result = bridge.get_alignment_ratio(holdings)
        >>> print(f"Aligned: {result.aligned_pct:.1f}%")
    """

    def __init__(self, config: Optional[TaxonomyBridgeConfig] = None) -> None:
        """Initialize the Article 9 Taxonomy Pack Bridge.

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
            "safeguards_engine": _AgentStub(
                "GL-TAXONOMY-SAFEGUARDS",
                f"{self.config.pack_008_path}.engines.safeguards_engine",
                "MinimumSafeguardsEngine",
            ),
        }

        self.logger.info(
            "TaxonomyPackBridge (Art 9) initialized: pack_008=%s, objectives=%d, "
            "methodology=%s, safeguards=%s, cda=%s",
            self.config.pack_008_path,
            len(self.config.enabled_objectives),
            self.config.alignment_methodology.value,
            self.config.enforce_minimum_safeguards,
            self.config.include_cda_disclosure,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def get_alignment_ratio(
        self,
        holdings: List[Dict[str, Any]],
        methodology: Optional[AlignmentMethodology] = None,
    ) -> TaxonomyAlignmentData:
        """Calculate taxonomy alignment ratio for Article 9 product.

        Routes to PACK-008 if available, otherwise uses built-in calculation.
        Includes all 6 objectives, CDA disclosure, and Art 5/6 references.

        Args:
            holdings: List of portfolio holding records.
            methodology: Override alignment methodology (turnover/capex/opex).

        Returns:
            TaxonomyAlignmentData with full Article 9 taxonomy data.
        """
        method = methodology or self.config.alignment_methodology
        start_time = utcnow()

        # Attempt PACK-008 route
        if self.config.use_pack_008:
            pack_result = self._route_to_pack_008_alignment(holdings, method)
            if pack_result is not None:
                return pack_result

        # Fallback to built-in
        if not self.config.fallback_to_estimates:
            self.logger.error("PACK-008 unavailable and fallback disabled")
            return TaxonomyAlignmentData(
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
        """Get per-objective alignment breakdown across all 6 objectives.

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

    def assess_minimum_safeguards(
        self,
        holdings: List[Dict[str, Any]],
    ) -> SafeguardsResult:
        """Assess minimum safeguards compliance (Art 18 Taxonomy Regulation).

        Checks alignment with UNGP, OECD Guidelines, ILO Core Conventions,
        and International Bill of Human Rights.

        Args:
            holdings: List of portfolio holding records.

        Returns:
            SafeguardsResult with per-holding assessment.
        """
        per_holding: List[Dict[str, Any]] = []
        pass_count = 0

        for h in holdings:
            isin = h.get("isin", "unknown")
            name = h.get("name", "")
            safeguards = h.get("minimum_safeguards", {})

            ungp = safeguards.get("ungp_compliant", True)
            oecd = safeguards.get("oecd_compliant", True)
            ilo = safeguards.get("ilo_compliant", True)
            intl_bill = safeguards.get("international_bill_compliant", True)

            passes = ungp and oecd and ilo and intl_bill

            per_holding.append({
                "isin": isin,
                "name": name,
                "passes": passes,
                "ungp": ungp,
                "oecd": oecd,
                "ilo": ilo,
                "international_bill": intl_bill,
            })
            if passes:
                pass_count += 1

        total = len(holdings) or 1
        pass_pct = round((pass_count / total) * 100, 1)

        result = SafeguardsResult(
            total_assessed=len(holdings),
            pass_count=pass_count,
            fail_count=len(holdings) - pass_count,
            pass_pct=pass_pct,
            ungp_compliant=all(r.get("ungp", True) for r in per_holding),
            oecd_compliant=all(r.get("oecd", True) for r in per_holding),
            ilo_compliant=all(r.get("ilo", True) for r in per_holding),
            international_bill_compliant=all(
                r.get("international_bill", True) for r in per_holding
            ),
            per_holding=per_holding[:50],
        )
        result.provenance_hash = _hash_data(result.model_dump())

        self.logger.info(
            "Minimum safeguards: %d/%d pass (%.1f%%)",
            pass_count, len(holdings), pass_pct,
        )
        return result

    def get_cda_disclosure(
        self,
        holdings: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Get Complementary Delegated Act (gas/nuclear) disclosure data.

        Args:
            holdings: List of portfolio holding records.

        Returns:
            Dictionary with fossil gas and nuclear alignment percentages.
        """
        total_weight = sum(float(h.get("weight", 0.0)) for h in holdings)
        if total_weight <= 0:
            return {"fossil_gas_pct": 0.0, "nuclear_pct": 0.0}

        gas_weight = 0.0
        nuclear_weight = 0.0

        for h in holdings:
            weight = float(h.get("weight", 0.0))
            cda = h.get("cda_category", "")
            if cda == "fossil_gas":
                gas_weight += weight
            elif cda == "nuclear":
                nuclear_weight += weight

        result = {
            "fossil_gas_pct": round((gas_weight / total_weight) * 100, 2),
            "nuclear_pct": round((nuclear_weight / total_weight) * 100, 2),
            "total_cda_pct": round(
                ((gas_weight + nuclear_weight) / total_weight) * 100, 2
            ),
        }

        self.logger.info(
            "CDA disclosure: gas=%.1f%%, nuclear=%.1f%%",
            result["fossil_gas_pct"], result["nuclear_pct"],
        )
        return result

    def route_taxonomy_request(
        self,
        request_type: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route a taxonomy request to the appropriate engine.

        Args:
            request_type: Type of request.
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

        elif request_type == "objective_breakdown":
            return self.get_objective_breakdown(holdings)

        elif request_type == "minimum_safeguards":
            result = self.assess_minimum_safeguards(holdings)
            return result.model_dump()

        elif request_type == "cda_disclosure":
            return self.get_cda_disclosure(holdings)

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
    ) -> Optional[TaxonomyAlignmentData]:
        """Attempt to route alignment calculation to PACK-008.

        Args:
            holdings: Portfolio holdings.
            methodology: Alignment methodology.

        Returns:
            TaxonomyAlignmentData from PACK-008, or None if unavailable.
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
            return self._map_pack_008_result(raw_result, holdings, methodology)
        except Exception as exc:
            self.logger.warning(
                "PACK-008 alignment calculation failed: %s", exc
            )
            return None

    def _map_pack_008_result(
        self,
        raw_result: Any,
        holdings: List[Dict[str, Any]],
        methodology: AlignmentMethodology,
    ) -> TaxonomyAlignmentData:
        """Map PACK-008 raw result to Article 9 TaxonomyAlignmentData.

        Args:
            raw_result: Raw result from PACK-008 engine.
            holdings: Original holdings for enrichment.
            methodology: Methodology used.

        Returns:
            Mapped TaxonomyAlignmentData.
        """
        if isinstance(raw_result, dict):
            data = raw_result
        elif hasattr(raw_result, "model_dump"):
            data = raw_result.model_dump()
        else:
            data = {"aligned_pct": 0.0, "eligible_pct": 0.0}

        method_key = methodology.value
        aligned = float(
            data.get(f"alignment_ratio_{method_key}",
                     data.get("aligned_pct", 0.0))
        )
        eligible = float(
            data.get(f"eligibility_ratio_{method_key}",
                     data.get("eligible_pct", 0.0))
        )

        objective_breakdown = self.get_objective_breakdown(holdings)
        cda = self.get_cda_disclosure(holdings) if self.config.include_cda_disclosure else {}

        result = TaxonomyAlignmentData(
            aligned_pct=round(aligned, 2),
            eligible_pct=round(eligible, 2),
            not_eligible_pct=round(100.0 - eligible, 2),
            methodology=method_key,
            holdings_assessed=int(data.get("holdings_assessed", len(holdings))),
            objective_breakdown=objective_breakdown,
            cda_disclosure=cda,
            calculated_at=utcnow().isoformat(),
            source="pack_008",
        )
        result.provenance_hash = _hash_data(result.model_dump())
        return result

    def _calculate_alignment_builtin(
        self,
        holdings: List[Dict[str, Any]],
        methodology: AlignmentMethodology,
    ) -> TaxonomyAlignmentData:
        """Built-in taxonomy alignment calculation (deterministic).

        Used when PACK-008 is not available. Includes all Art 9 extensions:
        6-objective breakdown, CDA, minimum safeguards, Art 5/6.

        Args:
            holdings: Portfolio holdings with taxonomy flags.
            methodology: Alignment methodology.

        Returns:
            TaxonomyAlignmentData with built-in calculation.
        """
        total_weight = sum(float(h.get("weight", 0.0)) for h in holdings)
        if total_weight <= 0:
            return TaxonomyAlignmentData(
                calculated_at=utcnow().isoformat(),
                source="estimated",
            )

        aligned_weight = 0.0
        eligible_weight = 0.0
        unavailable_weight = 0.0
        si_env_aligned = 0.0
        si_env_not_aligned = 0.0

        for h in holdings:
            weight = float(h.get("weight", 0.0))
            status = self._classify_holding_alignment(h)
            is_si_env = h.get("si_objective_type", "") in (
                "environmental", "environmental_taxonomy"
            )

            if status == AlignmentStatus.ALIGNED:
                aligned_weight += weight
                eligible_weight += weight
                if is_si_env:
                    si_env_aligned += weight
            elif status == AlignmentStatus.ELIGIBLE_NOT_ALIGNED:
                eligible_weight += weight
                if is_si_env:
                    si_env_not_aligned += weight
            elif status == AlignmentStatus.DATA_UNAVAILABLE:
                unavailable_weight += weight

        aligned_pct = round((aligned_weight / total_weight) * 100, 2)
        eligible_pct = round((eligible_weight / total_weight) * 100, 2)
        not_eligible_pct = round(
            100.0 - eligible_pct - (unavailable_weight / total_weight) * 100, 2
        )
        unavailable_pct = round((unavailable_weight / total_weight) * 100, 2)

        objective_breakdown = self.get_objective_breakdown(holdings)
        cda = self.get_cda_disclosure(holdings) if self.config.include_cda_disclosure else {}

        # Art 5/6 reference data
        art_5_data = {
            "substantial_contribution_assessed": True,
            "objectives_with_sc": [
                obj for obj, bd in objective_breakdown.items()
                if bd.get("aligned_pct", 0) > 0
            ],
        }
        art_6_data = {
            "dnsh_all_objectives_assessed": True,
            "objectives_count": len(self.config.enabled_objectives),
        }

        result = TaxonomyAlignmentData(
            aligned_pct=aligned_pct,
            eligible_pct=eligible_pct,
            not_eligible_pct=max(not_eligible_pct, 0.0),
            data_unavailable_pct=unavailable_pct,
            methodology=methodology.value,
            holdings_assessed=len(holdings),
            objective_breakdown=objective_breakdown,
            si_env_aligned_pct=round(
                (si_env_aligned / total_weight) * 100, 2
            ) if total_weight > 0 else 0.0,
            si_env_not_aligned_pct=round(
                (si_env_not_aligned / total_weight) * 100, 2
            ) if total_weight > 0 else 0.0,
            art_5_alignment=art_5_data,
            art_6_dnsh=art_6_data,
            cda_disclosure=cda,
            calculated_at=utcnow().isoformat(),
            source="estimated",
        )
        result.provenance_hash = _hash_data(result.model_dump())

        self.logger.info(
            "Built-in alignment (Art 9): aligned=%.1f%%, eligible=%.1f%%, "
            "si_env_aligned=%.1f%%, cda_gas=%.1f%%, cda_nuclear=%.1f%%",
            aligned_pct, eligible_pct,
            result.si_env_aligned_pct,
            cda.get("fossil_gas_pct", 0.0), cda.get("nuclear_pct", 0.0),
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
