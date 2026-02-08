# -*- coding: utf-8 -*-
"""
Emissions Calculator - AGENT-DATA-003: ERP/Finance Connector
==============================================================

Calculates spend-based EEIO (Environmentally Extended Input-Output)
emissions estimates for classified spend records. Supports vendor-specific,
material-specific, and default emission factors with a strict priority
hierarchy.

Supports:
    - Batch emissions calculation for spend records
    - Single-record emissions calculation
    - Multi-tier emission factor lookup (vendor > material > category > default)
    - Custom emission factor registration
    - Emissions summary by Scope 3 category
    - Total emissions aggregation
    - Emissions-by-vendor breakdown
    - Emission factors table export
    - Deterministic EEIO formula (amount_usd * factor_kgCO2e_per_usd)
    - Thread-safe statistics counters

Zero-Hallucination Guarantees:
    - All calculations use deterministic EEIO formula
    - Emission factors from published EPA/EXIOBASE sources
    - No LLM or ML model in calculation path
    - Complete factor source tracking for provenance
    - SHA-256 provenance hashes for audit trails

Example:
    >>> from greenlang.erp_connector.emissions_calculator import EmissionsCalculator
    >>> calc = EmissionsCalculator()
    >>> results = calc.calculate_emissions(records)
    >>> total = calc.get_total_emissions(results)
    >>> summary = calc.get_emissions_summary(results)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-003 ERP/Finance Connector
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Layer 1 imports
from greenlang.agents.data.erp_connector_agent import (
    Scope3Category,
    SpendCategory,
    SpendRecord,
    DEFAULT_EMISSION_FACTORS,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EmissionResult",
    "EmissionsCalculator",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class EmissionResult(BaseModel):
    """Emission calculation result for a single spend record.

    Captures the input record reference, calculated emissions,
    emission factor used, and the source of the factor for
    complete audit traceability.
    """

    record_id: str = Field(..., description="Source spend record ID")
    vendor_id: str = Field(..., description="Vendor identifier")
    vendor_name: str = Field(..., description="Vendor name")
    amount_usd: float = Field(..., description="Spend amount in USD")
    spend_category: Optional[str] = Field(
        None, description="Spend category value",
    )
    scope3_category: Optional[str] = Field(
        None, description="Scope 3 category value",
    )
    emission_factor_kgco2e_per_usd: float = Field(
        ..., description="Emission factor applied",
    )
    emission_factor_source: str = Field(
        ..., description="Source of the emission factor",
    )
    emissions_kgco2e: float = Field(
        ..., description="Calculated emissions in kgCO2e",
    )
    methodology: str = Field(
        default="eeio", description="Calculation methodology",
    )
    provenance_hash: Optional[str] = Field(
        None, description="SHA-256 provenance hash",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# EmissionsCalculator
# ---------------------------------------------------------------------------


class EmissionsCalculator:
    """Spend-based EEIO emissions calculator.

    Calculates emissions using the formula:
        emissions_kgCO2e = amount_USD * emission_factor_kgCO2e_per_USD

    Emission factors are looked up in priority order:
    1. Vendor-specific custom factor
    2. Material-specific custom factor
    3. Spend-category default factor (from Layer 1 DEFAULT_EMISSION_FACTORS)
    4. Global default factor (0.25 kgCO2e/USD)

    Attributes:
        _config: Configuration dictionary.
        _custom_factors: Custom emission factors keyed by entity_id.
        _factor_sources: Source metadata for custom factors.
        _lock: Threading lock for statistics.
        _stats: Calculation statistics counters.

    Example:
        >>> calc = EmissionsCalculator()
        >>> results = calc.calculate_emissions(records)
        >>> total = calc.get_total_emissions(results)
        >>> assert total > 0
    """

    # Global default factor when no other match is found
    _GLOBAL_DEFAULT_FACTOR: float = 0.25
    _GLOBAL_DEFAULT_SOURCE: str = "global_default"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EmissionsCalculator.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``global_default_factor``: float (default 0.25)
                - ``methodology``: str (default "eeio")
        """
        self._config = config or {}
        self._global_default: float = self._config.get(
            "global_default_factor", self._GLOBAL_DEFAULT_FACTOR,
        )
        self._methodology: str = self._config.get("methodology", "eeio")
        self._custom_factors: Dict[str, float] = {}
        self._factor_sources: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "calculations_performed": 0,
            "total_emissions_kgco2e": 0.0,
            "by_methodology": {},
            "by_source": {},
            "errors": 0,
        }
        logger.info(
            "EmissionsCalculator initialised: methodology=%s, "
            "default_factor=%.4f, default_factors=%d",
            self._methodology,
            self._global_default,
            len(DEFAULT_EMISSION_FACTORS),
        )

    # ------------------------------------------------------------------
    # Public API - Calculation
    # ------------------------------------------------------------------

    def calculate_emissions(
        self,
        records: List[SpendRecord],
        methodology: str = "eeio",
    ) -> List[EmissionResult]:
        """Calculate emissions for all spend records.

        Applies emission factors to each record and returns detailed
        calculation results.

        Args:
            records: List of SpendRecord objects.
            methodology: Calculation methodology (default "eeio").

        Returns:
            List of EmissionResult objects.
        """
        start = time.monotonic()
        results: List[EmissionResult] = []

        for record in records:
            result = self.calculate_single(record, methodology)
            results.append(result)

        elapsed_ms = (time.monotonic() - start) * 1000
        total = sum(r.emissions_kgco2e for r in results)

        logger.info(
            "Calculated emissions for %d records: "
            "total=%.3f kgCO2e, methodology=%s (%.1f ms)",
            len(results), total, methodology, elapsed_ms,
        )
        return results

    def calculate_single(
        self,
        record: SpendRecord,
        methodology: str = "eeio",
    ) -> EmissionResult:
        """Calculate emissions for a single spend record.

        Looks up the appropriate emission factor and applies the
        EEIO formula.

        Args:
            record: SpendRecord to calculate for.
            methodology: Calculation methodology (default "eeio").

        Returns:
            EmissionResult with calculated emissions.
        """
        amount_usd = (
            record.amount_usd if record.amount_usd is not None
            else record.amount
        )

        # Lookup emission factor
        factor, source = self.get_emission_factor(
            vendor_id=record.vendor_id,
            spend_category=(
                record.spend_category.value
                if record.spend_category is not None
                else None
            ),
            material_id=record.material_group,
        )

        # Apply EEIO calculation
        emissions = self._eeio_calculation(amount_usd, factor)

        # Provenance hash
        provenance_hash = self._compute_provenance(
            record.record_id, str(amount_usd), str(factor),
            str(emissions), methodology,
        )

        result = EmissionResult(
            record_id=record.record_id,
            vendor_id=record.vendor_id,
            vendor_name=record.vendor_name,
            amount_usd=amount_usd,
            spend_category=(
                record.spend_category.value
                if record.spend_category is not None
                else None
            ),
            scope3_category=(
                record.scope3_category.value
                if record.scope3_category is not None
                else None
            ),
            emission_factor_kgco2e_per_usd=factor,
            emission_factor_source=source,
            emissions_kgco2e=emissions,
            methodology=methodology,
            provenance_hash=provenance_hash,
        )

        # Update statistics
        with self._lock:
            self._stats["calculations_performed"] += 1
            self._stats["total_emissions_kgco2e"] += emissions

            meth_counts = self._stats["by_methodology"]
            meth_counts[methodology] = meth_counts.get(methodology, 0) + 1

            src_counts = self._stats["by_source"]
            src_counts[source] = src_counts.get(source, 0) + 1

        return result

    # ------------------------------------------------------------------
    # Public API - Factor Management
    # ------------------------------------------------------------------

    def get_emission_factor(
        self,
        vendor_id: Optional[str] = None,
        spend_category: Optional[str] = None,
        material_id: Optional[str] = None,
    ) -> Tuple[float, str]:
        """Look up the emission factor with priority resolution.

        Priority order:
        1. Vendor-specific custom factor
        2. Material-specific custom factor
        3. Spend-category default factor
        4. Global default factor

        Args:
            vendor_id: Vendor identifier (optional).
            spend_category: Spend category value string (optional).
            material_id: Material or material group ID (optional).

        Returns:
            Tuple of (factor_kgCO2e_per_USD, source_description).
        """
        # Priority 1: Vendor-specific custom factor
        if vendor_id and vendor_id in self._custom_factors:
            source = self._factor_sources.get(
                vendor_id, "custom_vendor",
            )
            return self._custom_factors[vendor_id], source

        # Priority 2: Material-specific custom factor
        if material_id and material_id in self._custom_factors:
            source = self._factor_sources.get(
                material_id, "custom_material",
            )
            return self._custom_factors[material_id], source

        # Priority 3: Spend-category default factor
        if spend_category:
            try:
                cat_enum = SpendCategory(spend_category)
                if cat_enum in DEFAULT_EMISSION_FACTORS:
                    return (
                        DEFAULT_EMISSION_FACTORS[cat_enum],
                        "epa_eeio_default",
                    )
            except ValueError:
                pass

        # Priority 4: Global default
        return self._global_default, self._GLOBAL_DEFAULT_SOURCE

    def set_custom_factor(
        self,
        entity_id: str,
        factor_kgco2e_per_usd: float,
        source: str = "custom",
    ) -> None:
        """Set a custom emission factor for a vendor or material.

        Args:
            entity_id: Vendor ID or material ID.
            factor_kgco2e_per_usd: Emission factor in kgCO2e per USD.
            source: Source description for the factor.

        Raises:
            ValueError: If factor is negative.
        """
        if factor_kgco2e_per_usd < 0:
            raise ValueError(
                f"Emission factor must be non-negative, "
                f"got {factor_kgco2e_per_usd}"
            )

        self._custom_factors[entity_id] = factor_kgco2e_per_usd
        self._factor_sources[entity_id] = source

        logger.info(
            "Set custom emission factor: %s = %.4f kgCO2e/USD (source=%s)",
            entity_id, factor_kgco2e_per_usd, source,
        )

    # ------------------------------------------------------------------
    # Public API - Analysis
    # ------------------------------------------------------------------

    def get_emissions_summary(
        self,
        results: List[EmissionResult],
    ) -> Dict[str, Any]:
        """Get emissions summary grouped by Scope 3 category.

        Args:
            results: List of EmissionResult objects.

        Returns:
            Dictionary with category-level emissions totals,
            record counts, and grand total.
        """
        by_category: Dict[str, Dict[str, Any]] = {}

        for r in results:
            cat_key = r.scope3_category or "unclassified"
            if cat_key not in by_category:
                by_category[cat_key] = {
                    "category": cat_key,
                    "emissions_kgco2e": 0.0,
                    "spend_usd": 0.0,
                    "record_count": 0,
                }
            by_category[cat_key]["emissions_kgco2e"] += r.emissions_kgco2e
            by_category[cat_key]["spend_usd"] += r.amount_usd
            by_category[cat_key]["record_count"] += 1

        # Round values
        for entry in by_category.values():
            entry["emissions_kgco2e"] = round(
                entry["emissions_kgco2e"], 3,
            )
            entry["spend_usd"] = round(entry["spend_usd"], 2)

        total_emissions = round(
            sum(e["emissions_kgco2e"] for e in by_category.values()), 3,
        )
        total_spend = round(
            sum(e["spend_usd"] for e in by_category.values()), 2,
        )

        return {
            "total_emissions_kgco2e": total_emissions,
            "total_spend_usd": total_spend,
            "record_count": len(results),
            "categories": len(by_category),
            "by_category": by_category,
            "methodology": self._methodology,
            "provenance_hash": self._compute_provenance(
                "summary", str(total_emissions), str(len(results)),
            ),
        }

    def get_total_emissions(
        self,
        results: List[EmissionResult],
    ) -> float:
        """Get total emissions across all results.

        Args:
            results: List of EmissionResult objects.

        Returns:
            Total emissions in kgCO2e.
        """
        return round(
            sum(r.emissions_kgco2e for r in results), 3,
        )

    def get_emissions_by_vendor(
        self,
        results: List[EmissionResult],
    ) -> Dict[str, float]:
        """Get emissions breakdown by vendor.

        Args:
            results: List of EmissionResult objects.

        Returns:
            Dictionary of vendor_id -> total emissions (kgCO2e).
        """
        by_vendor: Dict[str, float] = defaultdict(float)
        for r in results:
            by_vendor[r.vendor_id] += r.emissions_kgco2e
        return {k: round(v, 3) for k, v in by_vendor.items()}

    def get_emission_factors_table(self) -> Dict[str, Dict[str, Any]]:
        """Get all emission factors with their sources.

        Returns a comprehensive table of default and custom factors.

        Returns:
            Dictionary of identifier -> factor details.
        """
        table: Dict[str, Dict[str, Any]] = {}

        # Default factors from Layer 1
        for cat, factor in DEFAULT_EMISSION_FACTORS.items():
            table[f"default:{cat.value}"] = {
                "entity_type": "spend_category",
                "entity_id": cat.value,
                "factor_kgco2e_per_usd": factor,
                "source": "epa_eeio_default",
                "is_custom": False,
            }

        # Custom factors
        for entity_id, factor in self._custom_factors.items():
            source = self._factor_sources.get(entity_id, "custom")
            table[f"custom:{entity_id}"] = {
                "entity_type": "custom",
                "entity_id": entity_id,
                "factor_kgco2e_per_usd": factor,
                "source": source,
                "is_custom": True,
            }

        # Global default
        table["global_default"] = {
            "entity_type": "global",
            "entity_id": "global",
            "factor_kgco2e_per_usd": self._global_default,
            "source": self._GLOBAL_DEFAULT_SOURCE,
            "is_custom": False,
        }

        return table

    def get_statistics(self) -> Dict[str, Any]:
        """Return calculator statistics.

        Returns:
            Dictionary of counter values and breakdown.
        """
        with self._lock:
            return {
                "calculations_performed": self._stats[
                    "calculations_performed"
                ],
                "total_emissions_kgco2e": round(
                    self._stats["total_emissions_kgco2e"], 3,
                ),
                "by_methodology": dict(self._stats["by_methodology"]),
                "by_source": dict(self._stats["by_source"]),
                "custom_factors_count": len(self._custom_factors),
                "default_factors_count": len(DEFAULT_EMISSION_FACTORS),
                "errors": self._stats["errors"],
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Internal calculation
    # ------------------------------------------------------------------

    def _eeio_calculation(
        self,
        amount_usd: float,
        factor: float,
    ) -> float:
        """Apply the EEIO emissions calculation formula.

        Formula: emissions_kgCO2e = amount_USD * factor_kgCO2e_per_USD

        This is the ONLY calculation method and is purely deterministic.
        No LLM or ML model is involved.

        Args:
            amount_usd: Spend amount in USD.
            factor: Emission factor in kgCO2e per USD.

        Returns:
            Emissions in kgCO2e, rounded to 3 decimal places.
        """
        return round(amount_usd * factor, 3)

    def _compute_provenance(self, *parts: str) -> str:
        """Compute SHA-256 provenance hash from parts.

        Args:
            *parts: Strings to include in the hash.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        combined = json.dumps(
            {"parts": list(parts), "timestamp": _utcnow().isoformat()},
            sort_keys=True,
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
