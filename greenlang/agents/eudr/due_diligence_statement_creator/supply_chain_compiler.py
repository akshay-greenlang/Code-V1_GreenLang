# -*- coding: utf-8 -*-
"""
Supply Chain Compiler Engine - AGENT-EUDR-037

Engine 4 of 7: Aggregates traceability data from EUDR-001 (Supply Chain
Mapper) through EUDR-015 (Mobile Data Collector) into a structured
supply chain overview for the DDS. Tracks multi-tier suppliers,
countries of production, chain of custody models, plot counts, and
traceability completeness scores.

Algorithm:
    1. Collect supply chain data from EUDR-001 Supply Chain Mapper
    2. Integrate chain of custody from EUDR-009
    3. Map supplier tiers from EUDR-008 Multi-Tier Tracker
    4. Aggregate countries of production from geolocation data
    5. Compute traceability completeness score
    6. Validate supply chain integrity
    7. Compute provenance hash for audit trail

Zero-Hallucination Guarantees:
    - All aggregation via deterministic counting and summing
    - No LLM involvement in supply chain compilation
    - Traceability score computed with Decimal precision
    - Complete provenance trail for every compilation

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-037 (GL-EUDR-DDSC-037)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import DDSCreatorConfig, get_config
from .models import CommodityType, SupplyChainData
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class SupplyChainCompiler:
    """Supply chain traceability compilation engine.

    Aggregates data from 15 upstream EUDR supply chain agents into
    a structured format for DDS inclusion.

    Attributes:
        config: Agent configuration.
        _provenance: SHA-256 provenance tracker.

    Example:
        >>> compiler = SupplyChainCompiler()
        >>> sc = await compiler.compile_supply_chain(
        ...     supply_chain_id="SC-001",
        ...     operator_id="OP-001",
        ...     commodity="cocoa",
        ...     suppliers=[{"name": "Farm A", "tier": 1, "country_code": "CI"}],
        ... )
        >>> assert sc.supplier_count == 1
    """

    def __init__(
        self,
        config: Optional[DDSCreatorConfig] = None,
    ) -> None:
        """Initialize the supply chain compiler engine.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._compilation_count = 0
        logger.info("SupplyChainCompiler engine initialized")

    async def compile_supply_chain(
        self,
        supply_chain_id: str,
        operator_id: str,
        commodity: str,
        suppliers: Optional[List[Dict[str, Any]]] = None,
        countries_of_production: Optional[List[str]] = None,
        chain_of_custody_model: str = "segregation",
        **kwargs: Any,
    ) -> SupplyChainData:
        """Compile supply chain data into structured format.

        Args:
            supply_chain_id: Supply chain identifier.
            operator_id: EUDR operator identifier.
            commodity: Commodity type string.
            suppliers: List of supplier dictionaries.
            countries_of_production: ISO country codes.
            chain_of_custody_model: CoC model type.
            **kwargs: Additional fields.

        Returns:
            Compiled SupplyChainData record.
        """
        start = time.monotonic()

        try:
            commodity_type = CommodityType(commodity)
        except ValueError:
            commodity_type = CommodityType.WOOD

        supplier_list = suppliers or []
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Compute tier count from unique tier values
        tier_count = len(
            set(s.get("tier", 0) for s in supplier_list)
        ) if supplier_list else 0

        # Sum plot counts across suppliers
        plot_count = sum(s.get("plot_count", 0) for s in supplier_list)

        # Extract unique countries from suppliers if not provided
        countries = countries_of_production or []
        if not countries and supplier_list:
            countries = list(set(
                s.get("country_code", "")
                for s in supplier_list
                if s.get("country_code")
            ))

        # Compute traceability score
        traceability_score = Decimal(
            str(kwargs.get("traceability_score", 0.0))
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # If no explicit score, compute from supplier completeness
        if traceability_score == Decimal("0") and supplier_list:
            traceability_score = self._compute_traceability_score(
                supplier_list
            )

        sc = SupplyChainData(
            supply_chain_id=supply_chain_id,
            operator_id=operator_id,
            commodity=commodity_type,
            tier_count=tier_count,
            supplier_count=len(supplier_list),
            suppliers=supplier_list,
            chain_of_custody_model=chain_of_custody_model,
            plot_count=plot_count,
            countries_of_production=countries,
            traceability_score=traceability_score,
            last_updated=now,
            provenance_hash=self._provenance.compute_hash({
                "supply_chain_id": supply_chain_id,
                "operator_id": operator_id,
                "commodity": commodity_type.value,
                "supplier_count": len(supplier_list),
                "compiled_at": now.isoformat(),
            }),
        )

        self._compilation_count += 1
        elapsed = time.monotonic() - start
        logger.info(
            "Supply chain %s compiled: %d suppliers, %d tiers, "
            "%d plots, score=%.1f in %.1fms",
            supply_chain_id, len(supplier_list), tier_count,
            plot_count, float(traceability_score), elapsed * 1000,
        )

        return sc

    def _compute_traceability_score(
        self,
        suppliers: List[Dict[str, Any]],
    ) -> Decimal:
        """Compute traceability completeness score from supplier data.

        Evaluates the completeness of supplier records based on
        required fields: name, country, tier, plot data.

        Args:
            suppliers: List of supplier dictionaries.

        Returns:
            Traceability score (0-100).
        """
        if not suppliers:
            return Decimal("0")

        required_fields = [
            "name", "country_code", "tier", "plot_count",
        ]
        total_fields = len(suppliers) * len(required_fields)
        filled_fields = 0

        for supplier in suppliers:
            for field in required_fields:
                value = supplier.get(field)
                if value is not None and value != "" and value != 0:
                    filled_fields += 1

        if total_fields == 0:
            return Decimal("0")

        score = (Decimal(str(filled_fields)) / Decimal(str(total_fields))) * Decimal("100")
        return score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    async def validate_completeness(
        self,
        sc: SupplyChainData,
    ) -> Dict[str, Any]:
        """Validate supply chain data completeness.

        Args:
            sc: Compiled supply chain data.

        Returns:
            Validation result dictionary.
        """
        issues: List[str] = []

        if sc.supplier_count == 0:
            issues.append("No suppliers in supply chain")
        if not sc.countries_of_production:
            issues.append("No countries of production specified")
        if sc.traceability_score < Decimal("50"):
            issues.append(
                f"Traceability score {sc.traceability_score}% below "
                "minimum threshold of 50%"
            )
        if sc.tier_count == 0 and sc.supplier_count > 0:
            issues.append("Supplier tier information missing")

        return {
            "complete": len(issues) == 0,
            "issues": issues,
            "supplier_count": sc.supplier_count,
            "tier_count": sc.tier_count,
            "traceability_score": float(sc.traceability_score),
        }

    async def get_countries_summary(
        self,
        sc: SupplyChainData,
    ) -> Dict[str, int]:
        """Get a summary of supplier counts by country.

        Args:
            sc: Compiled supply chain data.

        Returns:
            Dictionary mapping country codes to supplier counts.
        """
        country_counts: Dict[str, int] = {}
        for supplier in sc.suppliers:
            country = supplier.get("country_code", "unknown")
            country_counts[country] = country_counts.get(country, 0) + 1
        return country_counts

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Health check dictionary.
        """
        return {
            "engine": "SupplyChainCompiler",
            "status": "healthy",
            "compilations_completed": self._compilation_count,
        }
