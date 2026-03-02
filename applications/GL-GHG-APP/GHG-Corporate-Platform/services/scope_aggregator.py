"""
Scope Aggregator -- Core Engine Integrating 28 MRV Agents

This is the central aggregation engine for the GL-GHG-APP platform.
It calls (or simulates calling) the 28 existing MRV agents to build
Scope 1, Scope 2, and Scope 3 emissions totals per the GHG Protocol
Corporate Standard.

Agent mapping:
  Scope 1 (8 agents):  MRV-001 through MRV-008
  Scope 2 (5 agents):  MRV-009 through MRV-013
  Scope 3 (15 agents): MRV-014 through MRV-028

All aggregation is deterministic (zero-hallucination).  Emissions are
broken down by gas using IPCC AR5 GWP values, by category, and by entity.

Example:
    >>> agg = ScopeAggregator(config)
    >>> inventory = agg.aggregate_all(inventory_id)
    >>> print(inventory.grand_total_tco2e)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ConsolidationApproach,
    DataQualityTier,
    GHGAppConfig,
    GHGGas,
    GWP_AR5,
    Scope,
    Scope1Category,
    Scope3Category,
)
from .models import (
    GHGInventory,
    ScopeEmissions,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


class ScopeAggregator:
    """
    Aggregates emissions from 28 MRV agents into Scope 1/2/3 totals.

    Each aggregation method simulates calling the corresponding MRV agent,
    applies consolidation rules (equity share / control approach), and
    builds per-gas, per-category, and per-entity breakdowns.

    Attributes:
        config: Application configuration.
    """

    # ------------------------------------------------------------------
    # Agent Mappings
    # ------------------------------------------------------------------

    SCOPE1_CATEGORIES: Dict[str, str] = {
        "stationary_combustion": "MRV-001",
        "refrigerants_fgas": "MRV-002",
        "mobile_combustion": "MRV-003",
        "process_emissions": "MRV-004",
        "fugitive_emissions": "MRV-005",
        "land_use": "MRV-006",
        "waste_treatment": "MRV-007",
        "agricultural": "MRV-008",
    }

    SCOPE2_AGENTS: Dict[str, str] = {
        "location_based": "MRV-009",
        "market_based": "MRV-010",
        "steam_heat": "MRV-011",
        "cooling": "MRV-012",
        "dual_reporting": "MRV-013",
    }

    SCOPE3_CATEGORIES: Dict[str, str] = {
        "cat1_purchased_goods": "MRV-014",
        "cat2_capital_goods": "MRV-015",
        "cat3_fuel_energy": "MRV-016",
        "cat4_upstream_transport": "MRV-017",
        "cat5_waste_generated": "MRV-018",
        "cat6_business_travel": "MRV-019",
        "cat7_employee_commuting": "MRV-020",
        "cat8_upstream_leased": "MRV-021",
        "cat9_downstream_transport": "MRV-022",
        "cat10_processing_sold": "MRV-023",
        "cat11_use_of_sold": "MRV-024",
        "cat12_end_of_life": "MRV-025",
        "cat13_downstream_leased": "MRV-026",
        "cat14_franchises": "MRV-027",
        "cat15_investments": "MRV-028",
    }

    # Default gas profiles per Scope 1 category (percentage of total)
    _SCOPE1_GAS_PROFILES: Dict[str, Dict[GHGGas, Decimal]] = {
        "stationary_combustion": {
            GHGGas.CO2: Decimal("95.0"),
            GHGGas.CH4: Decimal("3.0"),
            GHGGas.N2O: Decimal("2.0"),
        },
        "refrigerants_fgas": {
            GHGGas.HFCS: Decimal("85.0"),
            GHGGas.PFCS: Decimal("10.0"),
            GHGGas.SF6: Decimal("5.0"),
        },
        "mobile_combustion": {
            GHGGas.CO2: Decimal("92.0"),
            GHGGas.CH4: Decimal("4.0"),
            GHGGas.N2O: Decimal("4.0"),
        },
        "process_emissions": {
            GHGGas.CO2: Decimal("80.0"),
            GHGGas.N2O: Decimal("10.0"),
            GHGGas.CH4: Decimal("5.0"),
            GHGGas.PFCS: Decimal("5.0"),
        },
        "fugitive_emissions": {
            GHGGas.CH4: Decimal("70.0"),
            GHGGas.CO2: Decimal("20.0"),
            GHGGas.SF6: Decimal("10.0"),
        },
        "land_use": {
            GHGGas.CO2: Decimal("60.0"),
            GHGGas.CH4: Decimal("25.0"),
            GHGGas.N2O: Decimal("15.0"),
        },
        "waste_treatment": {
            GHGGas.CH4: Decimal("55.0"),
            GHGGas.CO2: Decimal("30.0"),
            GHGGas.N2O: Decimal("15.0"),
        },
        "agricultural": {
            GHGGas.CH4: Decimal("50.0"),
            GHGGas.N2O: Decimal("35.0"),
            GHGGas.CO2: Decimal("15.0"),
        },
    }

    def __init__(
        self,
        config: Optional[GHGAppConfig] = None,
        inventory_store: Optional[Dict[str, GHGInventory]] = None,
    ) -> None:
        """
        Initialize ScopeAggregator.

        Args:
            config: Application configuration.
            inventory_store: Shared reference to inventory storage.
        """
        self.config = config or GHGAppConfig()
        self._inventory_store = inventory_store if inventory_store is not None else {}
        self._agent_results_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("ScopeAggregator initialized with %d MRV agent mappings", 28)

    # ------------------------------------------------------------------
    # Public Aggregation Methods
    # ------------------------------------------------------------------

    def aggregate_scope1(
        self,
        inventory_id: str,
        entity_data: Optional[Dict[str, Any]] = None,
    ) -> ScopeEmissions:
        """
        Aggregate Scope 1 (direct) emissions from 8 MRV agents.

        Calls each Scope 1 agent, aggregates results by gas, category,
        and entity.

        Args:
            inventory_id: Inventory ID.
            entity_data: Optional entity-level activity data.

        Returns:
            ScopeEmissions for Scope 1.
        """
        start = datetime.utcnow()
        logger.info("Aggregating Scope 1 for inventory %s", inventory_id)

        total_tco2e = Decimal("0")
        by_gas: Dict[str, Decimal] = {gas.value: Decimal("0") for gas in GHGGas}
        by_category: Dict[str, Decimal] = {}
        by_entity: Dict[str, Decimal] = {}
        biogenic_co2 = Decimal("0")

        for category_name, agent_id in self.SCOPE1_CATEGORIES.items():
            agent_result = self._call_mrv_agent(
                agent_id, category_name, inventory_id, entity_data
            )

            cat_total = Decimal(str(agent_result.get("total_tco2e", 0)))
            by_category[category_name] = cat_total
            total_tco2e += cat_total

            # Distribute across gases using profiles
            gas_breakdown = self._distribute_by_gas(
                category_name, cat_total, self._SCOPE1_GAS_PROFILES
            )
            for gas_name, gas_value in gas_breakdown.items():
                by_gas[gas_name] = by_gas.get(gas_name, Decimal("0")) + gas_value

            # Per-entity breakdown
            entity_breakdown = agent_result.get("by_entity", {})
            for eid, evalue in entity_breakdown.items():
                by_entity[eid] = by_entity.get(eid, Decimal("0")) + Decimal(str(evalue))

            # Biogenic CO2 (reported separately per GHG Protocol)
            biogenic_co2 += Decimal(str(agent_result.get("biogenic_co2", 0)))

        scope1 = ScopeEmissions(
            scope=Scope.SCOPE_1,
            total_tco2e=total_tco2e,
            by_gas=by_gas,
            by_category=by_category,
            by_entity=by_entity,
            biogenic_co2=biogenic_co2,
            data_quality_tier=self._assess_aggregate_tier(entity_data),
            methodology_notes="Scope 1 aggregated from 8 MRV agents (MRV-001 to MRV-008)",
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Scope 1 aggregation complete: %.2f tCO2e (%d categories) in %.1f ms",
            total_tco2e,
            len(by_category),
            elapsed_ms,
        )
        return scope1

    def aggregate_scope2(
        self,
        inventory_id: str,
        entity_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ScopeEmissions, ScopeEmissions]:
        """
        Aggregate Scope 2 (indirect energy) emissions.

        Returns both location-based and market-based totals as required
        by the GHG Protocol Scope 2 Guidance (2015).

        Args:
            inventory_id: Inventory ID.
            entity_data: Optional entity-level data.

        Returns:
            Tuple of (location-based ScopeEmissions, market-based ScopeEmissions).
        """
        start = datetime.utcnow()
        logger.info("Aggregating Scope 2 for inventory %s", inventory_id)

        # Location-based
        location_total = Decimal("0")
        location_by_category: Dict[str, Decimal] = {}
        location_by_entity: Dict[str, Decimal] = {}

        # Market-based
        market_total = Decimal("0")
        market_by_category: Dict[str, Decimal] = {}
        market_by_entity: Dict[str, Decimal] = {}

        biogenic_co2 = Decimal("0")

        for agent_name, agent_id in self.SCOPE2_AGENTS.items():
            agent_result = self._call_mrv_agent(
                agent_id, agent_name, inventory_id, entity_data
            )

            # Location-based component
            loc_value = Decimal(str(agent_result.get("location_based_tco2e", 0)))
            location_by_category[agent_name] = loc_value
            location_total += loc_value

            # Market-based component
            mkt_value = Decimal(str(agent_result.get("market_based_tco2e", 0)))
            market_by_category[agent_name] = mkt_value
            market_total += mkt_value

            # Per-entity
            for eid, edata in agent_result.get("by_entity", {}).items():
                if isinstance(edata, dict):
                    loc_e = Decimal(str(edata.get("location", 0)))
                    mkt_e = Decimal(str(edata.get("market", 0)))
                else:
                    loc_e = Decimal(str(edata))
                    mkt_e = Decimal(str(edata))
                location_by_entity[eid] = location_by_entity.get(eid, Decimal("0")) + loc_e
                market_by_entity[eid] = market_by_entity.get(eid, Decimal("0")) + mkt_e

            biogenic_co2 += Decimal(str(agent_result.get("biogenic_co2", 0)))

        # Scope 2 is predominantly CO2 from electricity
        location_by_gas = self._scope2_gas_breakdown(location_total)
        market_by_gas = self._scope2_gas_breakdown(market_total)

        scope2_location = ScopeEmissions(
            scope=Scope.SCOPE_2_LOCATION,
            total_tco2e=location_total,
            by_gas=location_by_gas,
            by_category=location_by_category,
            by_entity=location_by_entity,
            biogenic_co2=biogenic_co2,
            data_quality_tier=self._assess_aggregate_tier(entity_data),
            methodology_notes=(
                "Scope 2 location-based: grid-average emission factors "
                "(eGRID/IEA) per GHG Protocol Scope 2 Guidance"
            ),
        )

        scope2_market = ScopeEmissions(
            scope=Scope.SCOPE_2_MARKET,
            total_tco2e=market_total,
            by_gas=market_by_gas,
            by_category=market_by_category,
            by_entity=market_by_entity,
            biogenic_co2=biogenic_co2,
            data_quality_tier=self._assess_aggregate_tier(entity_data),
            methodology_notes=(
                "Scope 2 market-based: contractual instruments, residual mix "
                "per GHG Protocol Scope 2 Guidance"
            ),
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Scope 2 aggregation complete: location=%.2f, market=%.2f tCO2e in %.1f ms",
            location_total,
            market_total,
            elapsed_ms,
        )
        return scope2_location, scope2_market

    def aggregate_scope3(
        self,
        inventory_id: str,
        entity_data: Optional[Dict[str, Any]] = None,
    ) -> ScopeEmissions:
        """
        Aggregate Scope 3 (value chain) emissions from 15 category agents.

        Args:
            inventory_id: Inventory ID.
            entity_data: Optional entity-level data.

        Returns:
            ScopeEmissions for Scope 3.
        """
        start = datetime.utcnow()
        logger.info("Aggregating Scope 3 for inventory %s", inventory_id)

        total_tco2e = Decimal("0")
        by_gas: Dict[str, Decimal] = {gas.value: Decimal("0") for gas in GHGGas}
        by_category: Dict[str, Decimal] = {}
        by_entity: Dict[str, Decimal] = {}
        biogenic_co2 = Decimal("0")

        for category_name, agent_id in self.SCOPE3_CATEGORIES.items():
            agent_result = self._call_mrv_agent(
                agent_id, category_name, inventory_id, entity_data
            )

            cat_total = Decimal(str(agent_result.get("total_tco2e", 0)))
            by_category[category_name] = cat_total
            total_tco2e += cat_total

            # Scope 3 gas breakdown (predominantly CO2)
            gas_breakdown = self._scope3_gas_breakdown(cat_total)
            for gas_name, gas_value in gas_breakdown.items():
                by_gas[gas_name] = by_gas.get(gas_name, Decimal("0")) + gas_value

            # Per-entity
            for eid, evalue in agent_result.get("by_entity", {}).items():
                by_entity[eid] = by_entity.get(eid, Decimal("0")) + Decimal(str(evalue))

            biogenic_co2 += Decimal(str(agent_result.get("biogenic_co2", 0)))

        scope3 = ScopeEmissions(
            scope=Scope.SCOPE_3,
            total_tco2e=total_tco2e,
            by_gas=by_gas,
            by_category=by_category,
            by_entity=by_entity,
            biogenic_co2=biogenic_co2,
            data_quality_tier=DataQualityTier.TIER_1,  # Scope 3 typically tier 1
            methodology_notes=(
                "Scope 3 aggregated from 15 category agents (MRV-014 to MRV-028). "
                "Methods vary by category (spend-based, distance-based, supplier-specific)."
            ),
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Scope 3 aggregation complete: %.2f tCO2e (%d categories) in %.1f ms",
            total_tco2e,
            len(by_category),
            elapsed_ms,
        )
        return scope3

    def aggregate_all(
        self,
        inventory_id: str,
        entity_data: Optional[Dict[str, Any]] = None,
    ) -> GHGInventory:
        """
        Aggregate all scopes and update the inventory.

        This is the primary entry point for complete inventory calculation.

        Args:
            inventory_id: Inventory ID.
            entity_data: Optional entity-level data.

        Returns:
            Updated GHGInventory with all scope emissions.

        Raises:
            ValueError: If inventory not found.
        """
        start = datetime.utcnow()
        logger.info("Starting full aggregation for inventory %s", inventory_id)

        inventory = self._get_inventory_or_raise(inventory_id)

        # Aggregate all scopes
        scope1 = self.aggregate_scope1(inventory_id, entity_data)
        scope2_location, scope2_market = self.aggregate_scope2(inventory_id, entity_data)
        scope3 = self.aggregate_scope3(inventory_id, entity_data)

        # Update inventory
        inventory.scope1 = scope1
        inventory.scope2_location = scope2_location
        inventory.scope2_market = scope2_market
        inventory.scope3 = scope3
        inventory.recalculate_totals()
        inventory.updated_at = _now()
        inventory.provenance_hash = _sha256(
            f"aggregate:{inventory.org_id}:{inventory.year}:{inventory.grand_total_tco2e}"
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Full aggregation complete for inventory %s: "
            "S1=%.2f, S2L=%.2f, S2M=%.2f, S3=%.2f, Total=%.2f tCO2e in %.1f ms",
            inventory_id,
            scope1.total_tco2e,
            scope2_location.total_tco2e,
            scope2_market.total_tco2e,
            scope3.total_tco2e,
            inventory.grand_total_tco2e,
            elapsed_ms,
        )
        return inventory

    # ------------------------------------------------------------------
    # Breakdown Queries
    # ------------------------------------------------------------------

    def get_scope1_breakdown(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """
        Get detailed Scope 1 breakdown by category.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with per-category emissions and percentage breakdown.
        """
        inventory = self._get_inventory_or_raise(inventory_id)
        if inventory.scope1 is None:
            return {"error": "Scope 1 not yet aggregated"}

        total = inventory.scope1.total_tco2e
        breakdown: Dict[str, Any] = {
            "total_tco2e": str(total),
            "categories": {},
        }

        for cat_name, cat_value in inventory.scope1.by_category.items():
            pct = Decimal("0")
            if total > 0:
                pct = (cat_value / total * 100).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            breakdown["categories"][cat_name] = {
                "tco2e": str(cat_value),
                "percentage": str(pct),
                "agent": self.SCOPE1_CATEGORIES.get(cat_name, "unknown"),
            }

        return breakdown

    def get_scope2_comparison(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """
        Compare location-based vs market-based Scope 2 emissions.

        The GHG Protocol requires dual reporting for Scope 2.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict comparing the two methods with delta analysis.
        """
        inventory = self._get_inventory_or_raise(inventory_id)

        loc_total = (
            inventory.scope2_location.total_tco2e
            if inventory.scope2_location
            else Decimal("0")
        )
        mkt_total = (
            inventory.scope2_market.total_tco2e
            if inventory.scope2_market
            else Decimal("0")
        )
        delta = mkt_total - loc_total
        delta_pct = Decimal("0")
        if loc_total > 0:
            delta_pct = (delta / loc_total * 100).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        return {
            "location_based_tco2e": str(loc_total),
            "market_based_tco2e": str(mkt_total),
            "delta_tco2e": str(delta),
            "delta_pct": str(delta_pct),
            "market_is_lower": mkt_total < loc_total,
            "explanation": (
                "Market-based is lower when contractual instruments "
                "(RECs, PPAs, green tariffs) reduce market-based emissions "
                "below the grid-average location-based emissions."
                if mkt_total < loc_total
                else "Market-based is equal to or higher than location-based, "
                "indicating limited use of contractual instruments."
            ),
            "location_by_category": (
                {k: str(v) for k, v in inventory.scope2_location.by_category.items()}
                if inventory.scope2_location
                else {}
            ),
            "market_by_category": (
                {k: str(v) for k, v in inventory.scope2_market.by_category.items()}
                if inventory.scope2_market
                else {}
            ),
        }

    def get_scope3_category_breakdown(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """
        Get Scope 3 breakdown by all 15 categories.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with per-category emissions, percentages, and materiality flags.
        """
        inventory = self._get_inventory_or_raise(inventory_id)
        if inventory.scope3 is None:
            return {"error": "Scope 3 not yet aggregated"}

        total = inventory.scope3.total_tco2e
        categories: Dict[str, Any] = {}
        materiality_threshold = Decimal("1.0")  # 1% of Scope 3 total

        for cat_name, cat_value in inventory.scope3.by_category.items():
            pct = Decimal("0")
            if total > 0:
                pct = (cat_value / total * 100).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            categories[cat_name] = {
                "tco2e": str(cat_value),
                "percentage": str(pct),
                "agent": self.SCOPE3_CATEGORIES.get(cat_name, "unknown"),
                "material": pct >= materiality_threshold,
            }

        # Sort by emissions descending for hot-spot analysis
        sorted_cats = sorted(
            categories.items(),
            key=lambda x: Decimal(x[1]["tco2e"]),
            reverse=True,
        )

        return {
            "total_tco2e": str(total),
            "category_count": len(categories),
            "material_categories": sum(1 for c in categories.values() if c["material"]),
            "categories": dict(sorted_cats),
        }

    def get_gas_breakdown(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """
        Get emissions breakdown by GHG gas across all scopes.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with per-gas totals and percentages.
        """
        inventory = self._get_inventory_or_raise(inventory_id)

        combined_by_gas: Dict[str, Decimal] = {gas.value: Decimal("0") for gas in GHGGas}

        for scope_data in [
            inventory.scope1,
            inventory.scope2_location,
            inventory.scope3,
        ]:
            if scope_data:
                for gas_name, gas_value in scope_data.by_gas.items():
                    combined_by_gas[gas_name] = (
                        combined_by_gas.get(gas_name, Decimal("0")) + gas_value
                    )

        grand_total = sum(combined_by_gas.values())
        result: Dict[str, Any] = {
            "total_tco2e": str(grand_total),
            "gases": {},
        }

        for gas_name, gas_value in combined_by_gas.items():
            pct = Decimal("0")
            if grand_total > 0:
                pct = (gas_value / grand_total * 100).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            result["gases"][gas_name] = {
                "tco2e": str(gas_value),
                "percentage": str(pct),
                "gwp_ar5": GWP_AR5.get(GHGGas(gas_name), 0) if gas_name in [g.value for g in GHGGas] else 0,
            }

        return result

    def get_entity_breakdown(
        self,
        inventory_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get emissions breakdown by entity across all scopes.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict keyed by entity ID with per-scope emissions.
        """
        inventory = self._get_inventory_or_raise(inventory_id)
        entities: Dict[str, Dict[str, Decimal]] = {}

        scope_map = {
            "scope1": inventory.scope1,
            "scope2_location": inventory.scope2_location,
            "scope2_market": inventory.scope2_market,
            "scope3": inventory.scope3,
        }

        for scope_name, scope_data in scope_map.items():
            if scope_data is None:
                continue
            for eid, evalue in scope_data.by_entity.items():
                if eid not in entities:
                    entities[eid] = {}
                entities[eid][scope_name] = evalue

        result: Dict[str, Dict[str, Any]] = {}
        for eid, scope_values in entities.items():
            total = sum(scope_values.values())
            result[eid] = {
                "total_tco2e": str(total),
                **{k: str(v) for k, v in scope_values.items()},
            }

        return result

    def get_geographic_breakdown(
        self,
        inventory_id: str,
        entity_country_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Decimal]:
        """
        Get emissions breakdown by country/region.

        Args:
            inventory_id: Inventory ID.
            entity_country_map: Mapping of entity ID to country code.

        Returns:
            Dict of country code to total emissions.
        """
        entity_breakdown = self.get_entity_breakdown(inventory_id)
        country_map = entity_country_map or {}

        geo_totals: Dict[str, Decimal] = {}
        for eid, edata in entity_breakdown.items():
            country = country_map.get(eid, "UNKNOWN")
            total = Decimal(str(edata.get("total_tco2e", 0)))
            geo_totals[country] = geo_totals.get(country, Decimal("0")) + total

        return geo_totals

    def calculate_biogenic_co2(
        self,
        inventory_id: str,
    ) -> Decimal:
        """
        Calculate total biogenic CO2 across all scopes.

        Per GHG Protocol, biogenic CO2 is reported separately from
        the main inventory totals.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Total biogenic CO2 in tCO2.
        """
        inventory = self._get_inventory_or_raise(inventory_id)
        total_biogenic = Decimal("0")

        for scope_data in [
            inventory.scope1,
            inventory.scope2_location,
            inventory.scope3,
        ]:
            if scope_data:
                total_biogenic += scope_data.biogenic_co2

        logger.info(
            "Biogenic CO2 for inventory %s: %.2f tCO2",
            inventory_id,
            total_biogenic,
        )
        return total_biogenic

    # ------------------------------------------------------------------
    # MRV Agent Integration (Simulation for v1.0)
    # ------------------------------------------------------------------

    def _call_mrv_agent(
        self,
        agent_id: str,
        category_name: str,
        inventory_id: str,
        entity_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Call (or simulate) an MRV agent for emission calculations.

        In v1.0, this returns simulated results.  In production, this
        will make actual calls to the MRV agent microservices.

        Args:
            agent_id: MRV agent identifier (e.g. MRV-001).
            category_name: Category name for logging.
            inventory_id: Inventory ID for context.
            entity_data: Entity-level activity data.

        Returns:
            Dict with agent calculation results.
        """
        cache_key = f"{inventory_id}:{agent_id}"
        if cache_key in self._agent_results_cache:
            logger.debug("Cache hit for %s", cache_key)
            return self._agent_results_cache[cache_key]

        # Check if entity_data contains pre-computed results for this agent
        if entity_data and agent_id in entity_data:
            result = entity_data[agent_id]
            self._agent_results_cache[cache_key] = result
            logger.debug("Using provided entity data for %s (%s)", agent_id, category_name)
            return result

        # Simulation: return zero emissions for v1.0
        # Production: call actual MRV agent endpoint
        result: Dict[str, Any] = {
            "agent_id": agent_id,
            "category": category_name,
            "total_tco2e": 0,
            "location_based_tco2e": 0,
            "market_based_tco2e": 0,
            "by_entity": {},
            "by_gas": {},
            "biogenic_co2": 0,
            "data_quality": DataQualityTier.TIER_1.value,
            "status": "simulated",
            "provenance_hash": _sha256(f"{agent_id}:{inventory_id}:simulated"),
        }

        self._agent_results_cache[cache_key] = result
        logger.debug("Simulated call to %s (%s) for inventory %s", agent_id, category_name, inventory_id)
        return result

    def inject_agent_results(
        self,
        inventory_id: str,
        agent_id: str,
        results: Dict[str, Any],
    ) -> None:
        """
        Inject pre-computed MRV agent results into the cache.

        This allows testing and integration with actual MRV agents
        without going through the full agent call flow.

        Args:
            inventory_id: Inventory ID.
            agent_id: MRV agent identifier.
            results: Agent calculation results.
        """
        cache_key = f"{inventory_id}:{agent_id}"
        self._agent_results_cache[cache_key] = results
        logger.info("Injected results for %s in inventory %s", agent_id, inventory_id)

    def clear_cache(self, inventory_id: Optional[str] = None) -> int:
        """
        Clear agent results cache.

        Args:
            inventory_id: If provided, only clear for this inventory.

        Returns:
            Number of cache entries cleared.
        """
        if inventory_id:
            keys_to_remove = [
                k for k in self._agent_results_cache
                if k.startswith(f"{inventory_id}:")
            ]
            for key in keys_to_remove:
                del self._agent_results_cache[key]
            cleared = len(keys_to_remove)
        else:
            cleared = len(self._agent_results_cache)
            self._agent_results_cache.clear()

        logger.info("Cleared %d agent result cache entries", cleared)
        return cleared

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_inventory_or_raise(self, inventory_id: str) -> GHGInventory:
        """Retrieve inventory from store or raise ValueError."""
        inventory = self._inventory_store.get(inventory_id)
        if inventory is None:
            raise ValueError(f"Inventory not found: {inventory_id}")
        return inventory

    def _distribute_by_gas(
        self,
        category_name: str,
        total: Decimal,
        profiles: Dict[str, Dict[GHGGas, Decimal]],
    ) -> Dict[str, Decimal]:
        """
        Distribute total emissions across gases using category profile.

        Args:
            category_name: Category name for profile lookup.
            total: Total emissions to distribute.
            profiles: Gas distribution profiles.

        Returns:
            Dict of gas name to emission value.
        """
        profile = profiles.get(category_name, {GHGGas.CO2: Decimal("100.0")})
        result: Dict[str, Decimal] = {}

        for gas, pct in profile.items():
            result[gas.value] = (total * pct / Decimal("100")).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )

        return result

    def _scope2_gas_breakdown(self, total: Decimal) -> Dict[str, Decimal]:
        """Scope 2 gas breakdown (predominantly CO2 from electricity)."""
        return {
            GHGGas.CO2.value: (total * Decimal("0.97")).quantize(Decimal("0.0001")),
            GHGGas.CH4.value: (total * Decimal("0.015")).quantize(Decimal("0.0001")),
            GHGGas.N2O.value: (total * Decimal("0.015")).quantize(Decimal("0.0001")),
        }

    def _scope3_gas_breakdown(self, total: Decimal) -> Dict[str, Decimal]:
        """Scope 3 gas breakdown (varies, defaulting to CO2-dominant)."""
        return {
            GHGGas.CO2.value: (total * Decimal("0.90")).quantize(Decimal("0.0001")),
            GHGGas.CH4.value: (total * Decimal("0.05")).quantize(Decimal("0.0001")),
            GHGGas.N2O.value: (total * Decimal("0.03")).quantize(Decimal("0.0001")),
            GHGGas.HFCS.value: (total * Decimal("0.02")).quantize(Decimal("0.0001")),
        }

    def _assess_aggregate_tier(
        self,
        entity_data: Optional[Dict[str, Any]],
    ) -> DataQualityTier:
        """Assess overall data quality tier from entity data."""
        if entity_data is None:
            return DataQualityTier.TIER_1

        tiers = []
        for key, data in entity_data.items():
            if isinstance(data, dict) and "data_quality" in data:
                tier_str = data["data_quality"]
                try:
                    tiers.append(DataQualityTier(tier_str))
                except ValueError:
                    tiers.append(DataQualityTier.TIER_1)

        if not tiers:
            return DataQualityTier.TIER_1

        # Return the lowest (most conservative) tier
        tier_order = {
            DataQualityTier.TIER_1: 1,
            DataQualityTier.TIER_2: 2,
            DataQualityTier.TIER_3: 3,
        }
        return min(tiers, key=lambda t: tier_order[t])
