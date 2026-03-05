"""
Category Aggregator -- ISO 14064-1:2018 Categories 1-6 Aggregation Engine

Maps 28 GreenLang MRV agents to the six ISO 14064-1:2018 categories and
provides comprehensive aggregation, breakdown, and reporting capabilities.

ISO 14064-1 Category Mapping:
  Category 1 (Direct): MRV-001 to MRV-008 (Scope 1 agents) + removals
  Category 2 (Energy): MRV-009 to MRV-013 (Scope 2 agents)
  Category 3 (Transport): MRV-017, MRV-019, MRV-020, MRV-022
  Category 4 (Products Used): MRV-014, MRV-015, MRV-016, MRV-018, MRV-021
  Category 5 (Products From Org): MRV-023, MRV-024, MRV-025, MRV-026, MRV-027
  Category 6 (Other): MRV-028

All aggregation is deterministic (zero-hallucination).  Emissions are broken
down by gas, facility, and source with full provenance tracking.

Uses in-memory storage for v1.0.

Example:
    >>> agg = CategoryAggregator(config)
    >>> result = agg.aggregate_all("inventory-123")
    >>> print(result["grand_total"].net_emissions_tco2e)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    DataQualityTier,
    GHGGas,
    GWP_AR5,
    ISO14064AppConfig,
    ISO_CATEGORY_NAMES,
    ISOCategory,
    SignificanceLevel,
)
from .models import (
    CategoryResult,
    EmissionSource,
    InventoryTotals,
    RemovalSource,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)

# Data quality tier numeric mapping for weighted averages
_DQ_TIER_SCORES: Dict[DataQualityTier, Decimal] = {
    DataQualityTier.TIER_1: Decimal("1"),
    DataQualityTier.TIER_2: Decimal("2"),
    DataQualityTier.TIER_3: Decimal("3"),
    DataQualityTier.TIER_4: Decimal("4"),
}


class CategoryAggregator:
    """
    Aggregates emissions from 28 MRV agents into ISO 14064-1 Categories 1-6.

    Supports two modes of operation:
      1. MRV agent-based aggregation (aggregate_category / aggregate_all)
         - Calls or simulates MRV agent results for each category.
         - Uses cached / injected agent results.
      2. Source-list-based aggregation (aggregate_inventory)
         - Accepts pre-built EmissionSource and RemovalSource lists.
         - Used when sources are already quantified.

    Provides:
      - Per-category gas, facility, and source breakdowns
      - Significance assessment for Categories 3-6
      - Grand total: net Category 1 + significant Categories 2-6
      - Year-over-year change calculations
      - Biogenic CO2 separate reporting

    Attributes:
        config: Application configuration.
    """

    # ------------------------------------------------------------------
    # MRV Agent Mappings to ISO 14064-1 Categories
    # ------------------------------------------------------------------

    CATEGORY_1_AGENTS: Dict[str, str] = {
        "stationary_combustion": "MRV-001",
        "refrigerants_fgas": "MRV-002",
        "mobile_combustion": "MRV-003",
        "process_emissions": "MRV-004",
        "fugitive_emissions": "MRV-005",
        "land_use": "MRV-006",
        "waste_treatment": "MRV-007",
        "agricultural": "MRV-008",
    }

    CATEGORY_2_AGENTS: Dict[str, str] = {
        "scope2_location": "MRV-009",
        "scope2_market": "MRV-010",
        "steam_heat_purchase": "MRV-011",
        "cooling_purchase": "MRV-012",
        "dual_reporting": "MRV-013",
    }

    CATEGORY_3_AGENTS: Dict[str, str] = {
        "upstream_transportation": "MRV-017",
        "business_travel": "MRV-019",
        "employee_commuting": "MRV-020",
        "downstream_transportation": "MRV-022",
    }

    CATEGORY_4_AGENTS: Dict[str, str] = {
        "purchased_goods_services": "MRV-014",
        "capital_goods": "MRV-015",
        "fuel_energy_activities": "MRV-016",
        "waste_generated": "MRV-018",
        "upstream_leased_assets": "MRV-021",
    }

    CATEGORY_5_AGENTS: Dict[str, str] = {
        "processing_sold_products": "MRV-023",
        "use_of_sold_products": "MRV-024",
        "end_of_life_treatment": "MRV-025",
        "downstream_leased_assets": "MRV-026",
        "franchises": "MRV-027",
    }

    CATEGORY_6_AGENTS: Dict[str, str] = {
        "investments": "MRV-028",
    }

    # All categories mapped to their agent dicts
    CATEGORY_AGENT_MAP: Dict[ISOCategory, Dict[str, str]] = {
        ISOCategory.CATEGORY_1_DIRECT: CATEGORY_1_AGENTS,
        ISOCategory.CATEGORY_2_ENERGY: CATEGORY_2_AGENTS,
        ISOCategory.CATEGORY_3_TRANSPORT: CATEGORY_3_AGENTS,
        ISOCategory.CATEGORY_4_PRODUCTS_USED: CATEGORY_4_AGENTS,
        ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG: CATEGORY_5_AGENTS,
        ISOCategory.CATEGORY_6_OTHER: CATEGORY_6_AGENTS,
    }

    # Default gas profiles per ISO category (percentage of total)
    _CATEGORY_GAS_PROFILES: Dict[ISOCategory, Dict[GHGGas, Decimal]] = {
        ISOCategory.CATEGORY_1_DIRECT: {
            GHGGas.CO2: Decimal("75.0"),
            GHGGas.CH4: Decimal("12.0"),
            GHGGas.N2O: Decimal("5.0"),
            GHGGas.HFCS: Decimal("4.0"),
            GHGGas.PFCS: Decimal("2.0"),
            GHGGas.SF6: Decimal("1.5"),
            GHGGas.NF3: Decimal("0.5"),
        },
        ISOCategory.CATEGORY_2_ENERGY: {
            GHGGas.CO2: Decimal("97.0"),
            GHGGas.CH4: Decimal("1.5"),
            GHGGas.N2O: Decimal("1.5"),
        },
        ISOCategory.CATEGORY_3_TRANSPORT: {
            GHGGas.CO2: Decimal("92.0"),
            GHGGas.CH4: Decimal("4.0"),
            GHGGas.N2O: Decimal("4.0"),
        },
        ISOCategory.CATEGORY_4_PRODUCTS_USED: {
            GHGGas.CO2: Decimal("88.0"),
            GHGGas.CH4: Decimal("6.0"),
            GHGGas.N2O: Decimal("4.0"),
            GHGGas.HFCS: Decimal("2.0"),
        },
        ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG: {
            GHGGas.CO2: Decimal("85.0"),
            GHGGas.CH4: Decimal("8.0"),
            GHGGas.N2O: Decimal("5.0"),
            GHGGas.HFCS: Decimal("2.0"),
        },
        ISOCategory.CATEGORY_6_OTHER: {
            GHGGas.CO2: Decimal("90.0"),
            GHGGas.CH4: Decimal("5.0"),
            GHGGas.N2O: Decimal("3.0"),
            GHGGas.HFCS: Decimal("2.0"),
        },
    }

    def __init__(
        self,
        config: Optional[ISO14064AppConfig] = None,
        inventory_store: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize CategoryAggregator.

        Args:
            config: Application configuration.  Uses defaults if None.
            inventory_store: Shared reference to inventory data store.
        """
        self.config = config or ISO14064AppConfig()
        self._inventory_store = inventory_store if inventory_store is not None else {}
        self._agent_results_cache: Dict[str, Dict[str, Any]] = {}
        self._category_results: Dict[str, Dict[str, CategoryResult]] = {}
        self._inventory_totals: Dict[str, InventoryTotals] = {}
        self._previous_year_totals: Dict[str, Dict[str, Decimal]] = {}
        logger.info(
            "CategoryAggregator initialized with %d MRV agent mappings across 6 categories",
            sum(len(agents) for agents in self.CATEGORY_AGENT_MAP.values()),
        )

    # ------------------------------------------------------------------
    # Mode 1: MRV Agent-Based Aggregation
    # ------------------------------------------------------------------

    def aggregate_category(
        self,
        inventory_id: str,
        category: str,
        entity_data: Optional[Dict[str, Any]] = None,
        removals_tco2e: Decimal = Decimal("0"),
    ) -> CategoryResult:
        """
        Aggregate emissions for a single ISO 14064-1 category via MRV agents.

        Calls the mapped MRV agents, aggregates results by gas, facility,
        and source, then stores the result in the internal cache.

        Args:
            inventory_id: Inventory ID.
            category: ISO category value (e.g. 'category_1_direct').
            entity_data: Optional pre-loaded entity/agent data.
            removals_tco2e: Removals to subtract (Category 1 only).

        Returns:
            CategoryResult with aggregated emissions.

        Raises:
            ValueError: If category is invalid.
        """
        start = datetime.now(timezone.utc)
        iso_cat = ISOCategory(category)
        agents = self.CATEGORY_AGENT_MAP.get(iso_cat, {})

        logger.info(
            "Aggregating %s for inventory %s (%d agents)",
            iso_cat.value, inventory_id, len(agents),
        )

        total_tco2e = Decimal("0")
        by_gas: Dict[str, Decimal] = {gas.value: Decimal("0") for gas in GHGGas}
        by_facility: Dict[str, Decimal] = {}
        by_source: Dict[str, Decimal] = {}
        biogenic_co2 = Decimal("0")
        source_count = 0

        for source_name, agent_id in agents.items():
            agent_result = self._call_mrv_agent(
                agent_id, source_name, inventory_id, entity_data,
            )

            src_total = Decimal(str(agent_result.get("total_tco2e", 0)))
            by_source[source_name] = src_total
            total_tco2e += src_total
            source_count += 1

            # Distribute across gases using category profile
            gas_breakdown = self._distribute_by_gas(iso_cat, src_total)
            for gas_name, gas_value in gas_breakdown.items():
                by_gas[gas_name] = by_gas.get(gas_name, Decimal("0")) + gas_value

            # Per-facility breakdown from agent results
            facility_data = agent_result.get("by_entity", {})
            for fid, fvalue in facility_data.items():
                by_facility[fid] = (
                    by_facility.get(fid, Decimal("0"))
                    + Decimal(str(fvalue))
                )

            biogenic_co2 += Decimal(str(agent_result.get("biogenic_co2", 0)))

        # Apply removals only to Category 1
        actual_removals = Decimal("0")
        if iso_cat == ISOCategory.CATEGORY_1_DIRECT:
            actual_removals = removals_tco2e

        net_tco2e = total_tco2e - actual_removals

        # Determine data quality tier from agent results
        dq_tier = self._assess_aggregate_tier(entity_data)

        result = CategoryResult(
            category=iso_cat,
            category_name=ISO_CATEGORY_NAMES.get(iso_cat, iso_cat.value),
            total_tco2e=total_tco2e,
            removals_tco2e=actual_removals,
            net_tco2e=net_tco2e,
            by_gas=by_gas,
            by_facility=by_facility,
            by_source=by_source,
            biogenic_co2=biogenic_co2,
            significance=SignificanceLevel.SIGNIFICANT,
            source_count=source_count,
            data_quality_tier=dq_tier,
        )

        # Cache the result
        if inventory_id not in self._category_results:
            self._category_results[inventory_id] = {}
        self._category_results[inventory_id][iso_cat.value] = result

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "%s aggregation: total=%.4f, removals=%.4f, net=%.4f tCO2e "
            "(%d agents, %d sources) in %.1f ms",
            iso_cat.value, total_tco2e, actual_removals, net_tco2e,
            len(agents), source_count, elapsed_ms,
        )
        return result

    def aggregate_all(
        self,
        inventory_id: str,
        entity_data: Optional[Dict[str, Any]] = None,
        removals_tco2e: Decimal = Decimal("0"),
        previous_year_net: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate all 6 ISO 14064-1 categories and compute grand totals.

        Computes:
          - Per-category results with gas, facility, and source breakdowns
          - Significance assessment for Categories 3-6
          - Grand total: net Category 1 + significant Categories 2-6
          - Year-over-year change if previous year data is available
          - Biogenic CO2 reported separately

        Args:
            inventory_id: Inventory ID.
            entity_data: Optional pre-loaded entity/agent data.
            removals_tco2e: Total removals (applied to Category 1 only).
            previous_year_net: Previous year net emissions for YoY comparison.

        Returns:
            Dict with categories, grand_total, significance_assessment,
            biogenic_co2, by_gas, and by_facility keys.
        """
        start = datetime.now(timezone.utc)
        logger.info(
            "Starting full ISO 14064-1 aggregation for inventory %s",
            inventory_id,
        )

        categories: Dict[str, CategoryResult] = {}

        # Aggregate each category
        for iso_cat in ISOCategory:
            cat_removals = (
                removals_tco2e
                if iso_cat == ISOCategory.CATEGORY_1_DIRECT
                else Decimal("0")
            )
            result = self.aggregate_category(
                inventory_id=inventory_id,
                category=iso_cat.value,
                entity_data=entity_data,
                removals_tco2e=cat_removals,
            )
            categories[iso_cat.value] = result

        # Compute gross total for significance assessment
        gross_total = sum(c.total_tco2e for c in categories.values())

        # Perform significance assessment for Categories 3-6
        significance_assessment = self._assess_significance(
            categories, gross_total,
        )

        # Apply significance to cached category results
        for cat_value, sig_result in significance_assessment.items():
            if cat_value in categories:
                categories[cat_value].significance = SignificanceLevel(
                    sig_result["significance"]
                )
                # Update cache
                if inventory_id in self._category_results:
                    self._category_results[inventory_id][cat_value] = categories[cat_value]

        # Compute grand total
        grand_total = self._build_grand_total(
            inventory_id, categories, removals_tco2e, previous_year_net,
        )
        self._inventory_totals[inventory_id] = grand_total

        # Biogenic CO2 total
        biogenic_total = sum(c.biogenic_co2 for c in categories.values())

        # Cross-category breakdowns
        combined_by_gas = self._combine_gas_breakdowns(categories)
        combined_by_facility = self._combine_facility_breakdowns(categories)

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "Full aggregation complete for inventory %s: "
            "gross=%.4f, removals=%.4f, net=%.4f tCO2e, "
            "Cat1_net=%.4f, significant_indirect=%.4f, "
            "biogenic=%.4f in %.1f ms",
            inventory_id, grand_total.gross_emissions_tco2e,
            grand_total.total_removals_tco2e, grand_total.net_emissions_tco2e,
            grand_total.category_1_net, grand_total.significant_indirect_tco2e,
            biogenic_total, elapsed_ms,
        )

        return {
            "inventory_id": inventory_id,
            "categories": categories,
            "grand_total": grand_total,
            "significance_assessment": significance_assessment,
            "biogenic_co2": {
                "total": biogenic_total,
                "by_category": {
                    cv: cr.biogenic_co2
                    for cv, cr in categories.items()
                },
            },
            "by_gas": combined_by_gas,
            "by_facility": combined_by_facility,
        }

    # ------------------------------------------------------------------
    # Mode 2: Source-List-Based Aggregation
    # ------------------------------------------------------------------

    def aggregate_inventory(
        self,
        inventory_id: str,
        sources: List[EmissionSource],
        removals: List[RemovalSource],
        significance_map: Optional[Dict[ISOCategory, SignificanceLevel]] = None,
    ) -> InventoryTotals:
        """
        Aggregate pre-built source and removal lists into category and inventory totals.

        This is the direct aggregation mode where EmissionSource objects have
        already been quantified and assigned to categories.

        Args:
            inventory_id: Inventory ID.
            sources: All emission sources for the inventory.
            removals: All removal sources for the inventory.
            significance_map: Significance per category (defaults to all significant).

        Returns:
            InventoryTotals with all aggregated results.
        """
        start = datetime.now(timezone.utc)
        sig_map = significance_map or {}

        # Group sources by category
        by_category: Dict[ISOCategory, List[EmissionSource]] = {}
        for cat in ISOCategory:
            by_category[cat] = []
        for src in sources:
            by_category.setdefault(src.category, []).append(src)

        # Build category results
        category_results: Dict[str, CategoryResult] = {}
        for cat in ISOCategory:
            cat_sources = by_category.get(cat, [])
            cat_removals = (
                removals if cat == ISOCategory.CATEGORY_1_DIRECT else []
            )
            result = self._aggregate_from_sources(
                category=cat,
                sources=cat_sources,
                removals=cat_removals,
                significance=sig_map.get(cat, SignificanceLevel.SIGNIFICANT),
            )
            category_results[cat.value] = result

        self._category_results[inventory_id] = category_results

        # Compute inventory totals
        totals = self._compute_totals(inventory_id, category_results)
        self._inventory_totals[inventory_id] = totals

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "Aggregated inventory %s from %d sources + %d removals: "
            "gross=%.2f, removals=%.2f, net=%.2f tCO2e in %.1f ms",
            inventory_id, len(sources), len(removals),
            totals.gross_emissions_tco2e, totals.total_removals_tco2e,
            totals.net_emissions_tco2e, elapsed_ms,
        )
        return totals

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    def get_category_results(
        self,
        inventory_id: str,
    ) -> Dict[str, CategoryResult]:
        """Get cached category results for an inventory."""
        return self._category_results.get(inventory_id, {})

    def get_category_result(
        self,
        inventory_id: str,
        category: ISOCategory,
    ) -> Optional[CategoryResult]:
        """Get result for a specific category."""
        results = self._category_results.get(inventory_id, {})
        return results.get(category.value)

    def get_inventory_totals(
        self,
        inventory_id: str,
    ) -> Optional[InventoryTotals]:
        """Get cached inventory totals."""
        return self._inventory_totals.get(inventory_id)

    # ------------------------------------------------------------------
    # Per-Gas Breakdown
    # ------------------------------------------------------------------

    def get_gas_breakdown(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """
        Get emissions breakdown by GHG gas across all categories.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with per-gas totals, percentages, and GWP values.
        """
        cached = self._category_results.get(inventory_id, {})
        if not cached:
            return {"error": "No aggregation results found. Run aggregate_all first."}

        combined: Dict[str, Decimal] = {gas.value: Decimal("0") for gas in GHGGas}
        for cat_result in cached.values():
            for gas_name, gas_value in cat_result.by_gas.items():
                combined[gas_name] = (
                    combined.get(gas_name, Decimal("0")) + gas_value
                )

        grand_total = sum(combined.values())

        gases: Dict[str, Any] = {}
        for gas_name, gas_value in combined.items():
            pct = Decimal("0")
            if grand_total > 0:
                pct = (gas_value / grand_total * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )
            gwp_value = GWP_AR5.get(
                GHGGas(gas_name), 1,
            ) if gas_name in [g.value for g in GHGGas] else 1
            gases[gas_name] = {
                "tco2e": str(gas_value),
                "percentage": str(pct),
                "gwp_ar5": gwp_value,
            }

        return {
            "total_tco2e": str(grand_total),
            "gases": gases,
        }

    # ------------------------------------------------------------------
    # Per-Facility Breakdown
    # ------------------------------------------------------------------

    def get_facility_breakdown(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """
        Get emissions breakdown by facility across all categories.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with per-facility totals and per-category sub-totals.
        """
        cached = self._category_results.get(inventory_id, {})
        if not cached:
            return {"error": "No aggregation results found. Run aggregate_all first."}

        facilities: Dict[str, Dict[str, Decimal]] = {}
        for cat_value, cat_result in cached.items():
            for fid, fvalue in cat_result.by_facility.items():
                if fid not in facilities:
                    facilities[fid] = {}
                facilities[fid][cat_value] = fvalue

        result: Dict[str, Any] = {}
        for fid, cat_values in facilities.items():
            total = sum(cat_values.values())
            result[fid] = {
                "total_tco2e": str(total),
                "by_category": {k: str(v) for k, v in cat_values.items()},
            }

        return result

    # ------------------------------------------------------------------
    # Significance Assessment
    # ------------------------------------------------------------------

    def assess_significance(
        self,
        inventory_id: str,
        category_emissions: Dict[str, Decimal],
        total_emissions: Decimal,
    ) -> Dict[str, Any]:
        """
        Perform significance assessment for indirect categories (3-6).

        Per ISO 14064-1 Clause 5.2.2, organizations shall assess the
        significance of indirect emission categories to determine which
        to include in the inventory.  Categories 1 and 2 are always mandatory.

        Args:
            inventory_id: Inventory ID.
            category_emissions: Dict of category value to emissions.
            total_emissions: Total gross emissions across all categories.

        Returns:
            Dict with per-category significance assessment results.
        """
        threshold = self.config.significance_threshold_percent
        results: Dict[str, Any] = {}

        for cat_value, emissions in category_emissions.items():
            iso_cat = ISOCategory(cat_value)

            # Categories 1 and 2 are always significant (mandatory)
            if iso_cat in (
                ISOCategory.CATEGORY_1_DIRECT,
                ISOCategory.CATEGORY_2_ENERGY,
            ):
                results[cat_value] = {
                    "category": cat_value,
                    "category_name": ISO_CATEGORY_NAMES.get(iso_cat, cat_value),
                    "emissions_tco2e": str(emissions),
                    "percentage_of_total": "N/A",
                    "significance": SignificanceLevel.SIGNIFICANT.value,
                    "is_significant": True,
                    "reason": "Mandatory category per ISO 14064-1",
                }
                continue

            # Assess significance for Categories 3-6
            if total_emissions > 0:
                pct = (emissions / total_emissions * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )
            else:
                pct = Decimal("0")

            is_significant = pct >= threshold

            results[cat_value] = {
                "category": cat_value,
                "category_name": ISO_CATEGORY_NAMES.get(iso_cat, cat_value),
                "emissions_tco2e": str(emissions),
                "percentage_of_total": str(pct),
                "threshold_pct": str(threshold),
                "significance": (
                    SignificanceLevel.SIGNIFICANT.value
                    if is_significant
                    else SignificanceLevel.NOT_SIGNIFICANT.value
                ),
                "is_significant": is_significant,
            }

        sig_count = sum(
            1 for r in results.values() if r.get("is_significant", True)
        )
        logger.info(
            "Significance assessment for inventory %s: %d/%d categories significant",
            inventory_id, sig_count, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Year-over-Year Change
    # ------------------------------------------------------------------

    def calculate_yoy_change(
        self,
        current_totals: InventoryTotals,
        previous_totals: Optional[InventoryTotals] = None,
        current_net: Optional[Decimal] = None,
        previous_net: Optional[Decimal] = None,
    ) -> InventoryTotals:
        """
        Calculate year-over-year change and update totals.

        Accepts either InventoryTotals objects or raw Decimal values.

        Args:
            current_totals: Current year totals (updated in place).
            previous_totals: Previous year totals object.
            current_net: Current year net emissions override.
            previous_net: Previous year net emissions override.

        Returns:
            Updated totals with YoY fields populated.
        """
        curr = current_net or current_totals.net_emissions_tco2e
        prev = Decimal("0")

        if previous_totals is not None:
            prev = previous_totals.net_emissions_tco2e
        elif previous_net is not None:
            prev = previous_net

        if prev != Decimal("0"):
            change = curr - prev
            change_pct = (
                (change / prev) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            current_totals.yoy_change_tco2e = change
            current_totals.yoy_change_pct = change_pct
            logger.info(
                "YoY change: %.4f -> %.4f tCO2e (%.2f%%)",
                prev, curr, change_pct,
            )

        return current_totals

    def calculate_category_yoy(
        self,
        inventory_id: str,
        current_categories: Dict[str, Decimal],
        previous_categories: Dict[str, Decimal],
    ) -> Dict[str, Any]:
        """
        Calculate year-over-year change for each category individually.

        Args:
            inventory_id: Inventory ID.
            current_categories: Current year per-category emissions.
            previous_categories: Previous year per-category emissions.

        Returns:
            Dict with per-category YoY change details.
        """
        results: Dict[str, Any] = {}
        all_categories = set(
            list(current_categories.keys())
            + list(previous_categories.keys())
        )

        for cat_value in all_categories:
            current = current_categories.get(cat_value, Decimal("0"))
            previous = previous_categories.get(cat_value, Decimal("0"))
            absolute = current - previous

            if previous != 0:
                pct = (absolute / previous * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )
            else:
                pct = Decimal("0") if current == 0 else Decimal("100")

            direction = (
                "increase" if absolute > 0
                else "decrease" if absolute < 0
                else "no_change"
            )

            results[cat_value] = {
                "current_tco2e": str(current),
                "previous_tco2e": str(previous),
                "absolute_change": str(absolute),
                "percentage_change": str(pct),
                "direction": direction,
            }

        return {
            "inventory_id": inventory_id,
            "categories": results,
        }

    # ------------------------------------------------------------------
    # Biogenic CO2 Reporting
    # ------------------------------------------------------------------

    def get_biogenic_co2_report(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """
        Generate biogenic CO2 report per ISO 14064-1 Clause 9.

        Per the standard, biogenic CO2 shall be reported separately
        from the main GHG inventory totals.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with biogenic CO2 totals by category.
        """
        cached = self._category_results.get(inventory_id, {})
        if not cached:
            return {"error": "No aggregation results found. Run aggregate_all first."}

        total_biogenic = Decimal("0")
        by_category: Dict[str, str] = {}

        for cat_value, cat_result in cached.items():
            by_category[cat_value] = str(cat_result.biogenic_co2)
            total_biogenic += cat_result.biogenic_co2

        return {
            "inventory_id": inventory_id,
            "total_biogenic_co2_tco2": str(total_biogenic),
            "by_category": by_category,
            "reporting_note": (
                "Biogenic CO2 emissions are reported separately from the "
                "main inventory per ISO 14064-1:2018 Clause 9 and shall not "
                "be included in the organization's GHG inventory totals."
            ),
        }

    # ------------------------------------------------------------------
    # Category Detail Queries
    # ------------------------------------------------------------------

    def get_category_detail(
        self,
        inventory_id: str,
        category: str,
    ) -> Dict[str, Any]:
        """
        Get detailed breakdown for a single category.

        Args:
            inventory_id: Inventory ID.
            category: ISO category value.

        Returns:
            Dict with detailed emissions breakdown for the category.
        """
        cached = self._category_results.get(inventory_id, {})
        cat_result = cached.get(category)
        if cat_result is None:
            return {"error": f"No results for {category}. Run aggregation first."}

        iso_cat = ISOCategory(category)
        agents = self.CATEGORY_AGENT_MAP.get(iso_cat, {})

        # Build per-source detail with percentage
        source_details: Dict[str, Any] = {}
        for source_name, source_total in cat_result.by_source.items():
            pct = Decimal("0")
            if cat_result.total_tco2e > 0:
                pct = (
                    source_total / cat_result.total_tco2e * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            source_details[source_name] = {
                "tco2e": str(source_total),
                "percentage": str(pct),
                "agent_id": agents.get(source_name, "unknown"),
            }

        # Sort sources by emissions descending
        sorted_sources = dict(sorted(
            source_details.items(),
            key=lambda x: Decimal(x[1]["tco2e"]),
            reverse=True,
        ))

        # Gas breakdown with percentages
        gas_details: Dict[str, Any] = {}
        for gas_name, gas_value in cat_result.by_gas.items():
            pct = Decimal("0")
            if cat_result.total_tco2e > 0:
                pct = (
                    gas_value / cat_result.total_tco2e * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            gas_details[gas_name] = {
                "tco2e": str(gas_value),
                "percentage": str(pct),
            }

        return {
            "category": category,
            "category_name": cat_result.category_name,
            "total_tco2e": str(cat_result.total_tco2e),
            "removals_tco2e": str(cat_result.removals_tco2e),
            "net_tco2e": str(cat_result.net_tco2e),
            "biogenic_co2": str(cat_result.biogenic_co2),
            "significance": cat_result.significance.value,
            "data_quality_tier": cat_result.data_quality_tier.value,
            "source_count": cat_result.source_count,
            "sources": sorted_sources,
            "by_gas": gas_details,
            "by_facility": {k: str(v) for k, v in cat_result.by_facility.items()},
            "provenance_hash": cat_result.provenance_hash,
        }

    def list_categories_summary(
        self,
        inventory_id: str,
    ) -> List[Dict[str, Any]]:
        """
        List a summary of all 6 categories for an inventory.

        Args:
            inventory_id: Inventory ID.

        Returns:
            List of category summary dicts in category order.
        """
        cached = self._category_results.get(inventory_id, {})
        if not cached:
            return []

        summaries: List[Dict[str, Any]] = []
        for iso_cat in ISOCategory:
            cat_result = cached.get(iso_cat.value)
            if cat_result is None:
                continue
            summaries.append({
                "category": iso_cat.value,
                "category_name": cat_result.category_name,
                "total_tco2e": str(cat_result.total_tco2e),
                "net_tco2e": str(cat_result.net_tco2e),
                "significance": cat_result.significance.value,
                "source_count": cat_result.source_count,
                "data_quality_tier": cat_result.data_quality_tier.value,
            })

        return summaries

    # ------------------------------------------------------------------
    # Agent Result Injection (Testing & Integration)
    # ------------------------------------------------------------------

    def inject_agent_results(
        self,
        inventory_id: str,
        agent_id: str,
        results: Dict[str, Any],
    ) -> None:
        """
        Inject pre-computed MRV agent results into the cache.

        Args:
            inventory_id: Inventory ID.
            agent_id: MRV agent identifier (e.g. MRV-001).
            results: Agent calculation results dict.
        """
        cache_key = f"{inventory_id}:{agent_id}"
        self._agent_results_cache[cache_key] = results
        logger.info(
            "Injected results for %s in inventory %s", agent_id, inventory_id,
        )

    def set_previous_year_totals(
        self,
        inventory_id: str,
        category_totals: Dict[str, Decimal],
    ) -> None:
        """
        Set previous year category totals for YoY comparison.

        Args:
            inventory_id: Inventory ID.
            category_totals: Dict of category value to net emissions.
        """
        self._previous_year_totals[inventory_id] = category_totals
        logger.info(
            "Set previous year totals for inventory %s: %d categories",
            inventory_id, len(category_totals),
        )

    def clear_cache(self, inventory_id: Optional[str] = None) -> int:
        """
        Clear agent results cache and category results.

        Args:
            inventory_id: If provided, only clear for this inventory.

        Returns:
            Number of cache entries cleared.
        """
        cleared = 0

        if inventory_id:
            agent_keys = [
                k for k in self._agent_results_cache
                if k.startswith(f"{inventory_id}:")
            ]
            for key in agent_keys:
                del self._agent_results_cache[key]
            cleared += len(agent_keys)

            if inventory_id in self._category_results:
                cleared += len(self._category_results[inventory_id])
                del self._category_results[inventory_id]
            if inventory_id in self._inventory_totals:
                cleared += 1
                del self._inventory_totals[inventory_id]
        else:
            cleared = (
                len(self._agent_results_cache)
                + sum(len(v) for v in self._category_results.values())
                + len(self._inventory_totals)
            )
            self._agent_results_cache.clear()
            self._category_results.clear()
            self._inventory_totals.clear()

        logger.info("Cleared %d cache entries", cleared)
        return cleared

    # ------------------------------------------------------------------
    # Private: MRV Agent Integration (Simulation for v1.0)
    # ------------------------------------------------------------------

    def _call_mrv_agent(
        self,
        agent_id: str,
        source_name: str,
        inventory_id: str,
        entity_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Call (or simulate) an MRV agent for emission calculations.

        In v1.0, returns simulated results (zero emissions) or cached
        injected results.  In production, this will make actual calls
        to the MRV agent microservices.

        Args:
            agent_id: MRV agent identifier.
            source_name: Source name for logging.
            inventory_id: Inventory ID.
            entity_data: Entity-level activity data.

        Returns:
            Dict with agent calculation results.
        """
        cache_key = f"{inventory_id}:{agent_id}"

        if cache_key in self._agent_results_cache:
            logger.debug("Cache hit for %s", cache_key)
            return self._agent_results_cache[cache_key]

        if entity_data and agent_id in entity_data:
            result = entity_data[agent_id]
            self._agent_results_cache[cache_key] = result
            logger.debug(
                "Using provided entity data for %s (%s)", agent_id, source_name,
            )
            return result

        # Simulation: return zero emissions for v1.0
        result: Dict[str, Any] = {
            "agent_id": agent_id,
            "source": source_name,
            "total_tco2e": 0,
            "by_entity": {},
            "by_gas": {},
            "biogenic_co2": 0,
            "data_quality": DataQualityTier.TIER_1.value,
            "status": "simulated",
            "provenance_hash": _sha256(
                f"{agent_id}:{inventory_id}:simulated"
            ),
        }
        self._agent_results_cache[cache_key] = result
        logger.debug(
            "Simulated call to %s (%s) for inventory %s",
            agent_id, source_name, inventory_id,
        )
        return result

    # ------------------------------------------------------------------
    # Private: Source-List Aggregation
    # ------------------------------------------------------------------

    def _aggregate_from_sources(
        self,
        category: ISOCategory,
        sources: List[EmissionSource],
        removals: List[RemovalSource],
        significance: SignificanceLevel,
    ) -> CategoryResult:
        """
        Aggregate a single ISO category from pre-built source lists.

        Args:
            category: ISO 14064-1 category.
            sources: Emission sources for this category.
            removals: Removal sources (non-empty only for Category 1).
            significance: Pre-assessed significance level.

        Returns:
            CategoryResult with aggregated emissions.
        """
        total_tco2e = Decimal("0")
        biogenic_co2 = Decimal("0")
        by_gas: Dict[str, Decimal] = {}
        by_facility: Dict[str, Decimal] = {}
        by_source: Dict[str, Decimal] = {}
        dq_scores: List[Decimal] = []
        dq_weights: List[Decimal] = []

        for src in sources:
            total_tco2e += src.tco2e
            biogenic_co2 += src.biogenic_co2

            gas_key = (
                src.gas.value if hasattr(src.gas, "value") else str(src.gas)
            )
            by_gas[gas_key] = by_gas.get(gas_key, Decimal("0")) + src.tco2e

            if src.facility_id:
                by_facility[src.facility_id] = (
                    by_facility.get(src.facility_id, Decimal("0")) + src.tco2e
                )

            by_source[src.source_name] = (
                by_source.get(src.source_name, Decimal("0")) + src.tco2e
            )

            tier_score = _DQ_TIER_SCORES.get(
                src.data_quality_tier, Decimal("1"),
            )
            dq_scores.append(tier_score * src.tco2e)
            dq_weights.append(src.tco2e)

        # Removals (Category 1 only)
        removals_tco2e = Decimal("0")
        for rem in removals:
            removals_tco2e += rem.credited_removals_tco2e

        net_tco2e = total_tco2e - removals_tco2e

        # Weighted average data quality tier
        weighted_dq = DataQualityTier.TIER_1
        if dq_weights and sum(dq_weights) > 0:
            weighted_score = sum(dq_scores) / sum(dq_weights)
            if weighted_score >= Decimal("3.5"):
                weighted_dq = DataQualityTier.TIER_4
            elif weighted_score >= Decimal("2.5"):
                weighted_dq = DataQualityTier.TIER_3
            elif weighted_score >= Decimal("1.5"):
                weighted_dq = DataQualityTier.TIER_2

        return CategoryResult(
            category=category,
            category_name=ISO_CATEGORY_NAMES.get(category, category.value),
            total_tco2e=total_tco2e,
            removals_tco2e=removals_tco2e,
            net_tco2e=net_tco2e,
            by_gas=by_gas,
            by_facility=by_facility,
            by_source=by_source,
            biogenic_co2=biogenic_co2,
            significance=significance,
            source_count=len(sources),
            data_quality_tier=weighted_dq,
        )

    # ------------------------------------------------------------------
    # Private: Grand Total Computation
    # ------------------------------------------------------------------

    def _build_grand_total(
        self,
        inventory_id: str,
        categories: Dict[str, CategoryResult],
        removals_tco2e: Decimal,
        previous_year_net: Optional[Decimal],
    ) -> InventoryTotals:
        """Build InventoryTotals from category results."""
        gross_total = sum(c.total_tco2e for c in categories.values())
        cat1_net = categories[ISOCategory.CATEGORY_1_DIRECT.value].net_tco2e
        biogenic_total = sum(c.biogenic_co2 for c in categories.values())

        # Significant indirect categories (2-6)
        significant_indirect = Decimal("0")
        for iso_cat in (
            ISOCategory.CATEGORY_2_ENERGY,
            ISOCategory.CATEGORY_3_TRANSPORT,
            ISOCategory.CATEGORY_4_PRODUCTS_USED,
            ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
            ISOCategory.CATEGORY_6_OTHER,
        ):
            cat_result = categories[iso_cat.value]
            if cat_result.significance == SignificanceLevel.SIGNIFICANT:
                significant_indirect += cat_result.total_tco2e

        net_total = cat1_net + significant_indirect

        # Per-category net totals
        by_category: Dict[str, Decimal] = {
            cv: cr.net_tco2e for cv, cr in categories.items()
        }

        # Cross-category breakdowns
        combined_by_gas = self._combine_gas_breakdowns(categories)
        combined_by_facility = self._combine_facility_breakdowns(categories)

        # YoY
        yoy_pct: Optional[Decimal] = None
        yoy_abs: Optional[Decimal] = None
        if previous_year_net is not None and previous_year_net != 0:
            yoy_abs = net_total - previous_year_net
            yoy_pct = (
                (yoy_abs / previous_year_net) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        return InventoryTotals(
            inventory_id=inventory_id,
            gross_emissions_tco2e=gross_total,
            total_removals_tco2e=removals_tco2e,
            net_emissions_tco2e=net_total,
            category_1_net=cat1_net,
            significant_indirect_tco2e=significant_indirect,
            biogenic_co2_total=biogenic_total,
            by_gas=combined_by_gas,
            by_category=by_category,
            by_facility=combined_by_facility,
            yoy_change_pct=yoy_pct,
            yoy_change_tco2e=yoy_abs,
        )

    def _compute_totals(
        self,
        inventory_id: str,
        category_results: Dict[str, CategoryResult],
    ) -> InventoryTotals:
        """Compute inventory-wide totals from source-based category results."""
        gross = Decimal("0")
        removals = Decimal("0")
        biogenic = Decimal("0")
        by_gas: Dict[str, Decimal] = {}
        by_category: Dict[str, Decimal] = {}
        by_facility: Dict[str, Decimal] = {}
        significant_indirect = Decimal("0")

        for cat_key, result in category_results.items():
            gross += result.total_tco2e
            removals += result.removals_tco2e
            biogenic += result.biogenic_co2
            by_category[cat_key] = result.net_tco2e

            for gas, val in result.by_gas.items():
                by_gas[gas] = by_gas.get(gas, Decimal("0")) + val

            for fac, val in result.by_facility.items():
                by_facility[fac] = by_facility.get(fac, Decimal("0")) + val

            # Track significant indirect categories (3-6)
            if (
                result.category not in (
                    ISOCategory.CATEGORY_1_DIRECT,
                    ISOCategory.CATEGORY_2_ENERGY,
                )
                and result.significance == SignificanceLevel.SIGNIFICANT
            ):
                significant_indirect += result.total_tco2e

        cat1 = category_results.get(ISOCategory.CATEGORY_1_DIRECT.value)
        cat1_net = cat1.net_tco2e if cat1 else Decimal("0")

        return InventoryTotals(
            inventory_id=inventory_id,
            gross_emissions_tco2e=gross,
            total_removals_tco2e=removals,
            net_emissions_tco2e=gross - removals,
            category_1_net=cat1_net,
            significant_indirect_tco2e=significant_indirect,
            biogenic_co2_total=biogenic,
            by_gas=by_gas,
            by_category=by_category,
            by_facility=by_facility,
        )

    # ------------------------------------------------------------------
    # Private: Gas & Facility Helpers
    # ------------------------------------------------------------------

    def _distribute_by_gas(
        self,
        category: ISOCategory,
        total: Decimal,
    ) -> Dict[str, Decimal]:
        """Distribute total emissions across gases using category profile."""
        profile = self._CATEGORY_GAS_PROFILES.get(
            category, {GHGGas.CO2: Decimal("100.0")},
        )
        result: Dict[str, Decimal] = {}
        for gas, pct in profile.items():
            result[gas.value] = (total * pct / Decimal("100")).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP,
            )
        return result

    def _assess_aggregate_tier(
        self,
        entity_data: Optional[Dict[str, Any]],
    ) -> DataQualityTier:
        """Assess overall data quality tier from entity data (most conservative)."""
        if entity_data is None:
            return DataQualityTier.TIER_1

        tiers: List[DataQualityTier] = []
        for key, data in entity_data.items():
            if isinstance(data, dict) and "data_quality" in data:
                try:
                    tiers.append(DataQualityTier(data["data_quality"]))
                except ValueError:
                    tiers.append(DataQualityTier.TIER_1)

        if not tiers:
            return DataQualityTier.TIER_1

        tier_order = {
            DataQualityTier.TIER_1: 1,
            DataQualityTier.TIER_2: 2,
            DataQualityTier.TIER_3: 3,
            DataQualityTier.TIER_4: 4,
        }
        return min(tiers, key=lambda t: tier_order[t])

    def _assess_significance(
        self,
        categories: Dict[str, CategoryResult],
        gross_total: Decimal,
    ) -> Dict[str, Any]:
        """Internal significance assessment for all categories."""
        threshold = self.config.significance_threshold_percent
        results: Dict[str, Any] = {}

        for cat_value, cat_result in categories.items():
            iso_cat = ISOCategory(cat_value)

            if iso_cat in (
                ISOCategory.CATEGORY_1_DIRECT,
                ISOCategory.CATEGORY_2_ENERGY,
            ):
                results[cat_value] = {
                    "category": cat_value,
                    "significance": SignificanceLevel.SIGNIFICANT.value,
                    "is_significant": True,
                    "reason": "Mandatory category",
                }
                continue

            if gross_total > 0:
                pct = (
                    cat_result.total_tco2e / gross_total * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            else:
                pct = Decimal("0")

            is_significant = pct >= threshold

            results[cat_value] = {
                "category": cat_value,
                "significance": (
                    SignificanceLevel.SIGNIFICANT.value
                    if is_significant
                    else SignificanceLevel.NOT_SIGNIFICANT.value
                ),
                "is_significant": is_significant,
                "percentage_of_total": str(pct),
                "threshold_pct": str(threshold),
            }

        return results

    def _combine_gas_breakdowns(
        self,
        categories: Dict[str, CategoryResult],
    ) -> Dict[str, Decimal]:
        """Combine gas breakdowns across all categories."""
        combined: Dict[str, Decimal] = {
            gas.value: Decimal("0") for gas in GHGGas
        }
        for cat_result in categories.values():
            for gas_name, gas_value in cat_result.by_gas.items():
                combined[gas_name] = (
                    combined.get(gas_name, Decimal("0")) + gas_value
                )
        return combined

    def _combine_facility_breakdowns(
        self,
        categories: Dict[str, CategoryResult],
    ) -> Dict[str, Decimal]:
        """Combine facility breakdowns across all categories."""
        combined: Dict[str, Decimal] = {}
        for cat_result in categories.values():
            for fid, fvalue in cat_result.by_facility.items():
                combined[fid] = combined.get(fid, Decimal("0")) + fvalue
        return combined
