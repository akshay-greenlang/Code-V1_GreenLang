# -*- coding: utf-8 -*-
"""
HybridAggregatorEngine - AGENT-MRV-025 Engine 5

GHG Protocol Scope 3 Category 12 hybrid multi-method aggregator with
avoided emissions and circularity metrics.

This engine orchestrates the method waterfall, combining producer-specific,
waste-type-specific, average-data, and spend-based results into a single
coherent calculation. It also computes avoided emissions (reported
SEPARATELY per GHG Protocol) and circularity metrics per the Ellen
MacArthur Foundation framework.

Method Waterfall (highest to lowest priority):
    1. Producer-specific (if EPD/PCF data available) -- Tier 1
    2. Waste-type-specific (if material composition known) -- Tier 2
    3. Average-data (if product category known) -- Tier 2/3
    4. Spend-based EEIO fallback (placeholder EF) -- Tier 3

Avoided Emissions:
    Recycling credits and energy recovery credits are calculated but NEVER
    netted from gross emissions. This is a strict GHG Protocol requirement
    (Scope 3 Guidance Chapter 12, page 73): "Companies should not subtract
    avoided emissions from their scope 3 inventory."

Circularity Metrics:
    - recycling_rate = recycled_weight / total_weight
    - diversion_rate = (recycled + composted + AD) / total_weight
    - circularity_index per Ellen MacArthur Foundation MCI
    - waste_hierarchy_score = EU Waste Framework Directive compliance
    - material_recovery_rate = (recycled + composted) / (total - energy_recovery)

Thread Safety:
    Thread-safe singleton with threading.RLock() and double-checked locking.

References:
    - GHG Protocol Technical Guidance for Scope 3 Emissions, Category 12
    - Ellen MacArthur Foundation Material Circularity Indicator (MCI) v3.0
    - EU Waste Framework Directive 2008/98/EC (waste hierarchy)
    - ISO 14040/14044 LCA methodology
    - EPA WARM v16 avoided emissions methodology

Example:
    >>> engine = HybridAggregatorEngine.get_instance()
    >>> result = engine.calculate(
    ...     products=[
    ...         {"product_id": "P-001", "category": "electronics",
    ...          "units_sold": 1000, "weight_per_unit_kg": Decimal("0.5"),
    ...          "epd_data": {"c1_deconstruction_kgco2e": Decimal("0.05"),
    ...                       "c2_transport_kgco2e": Decimal("0.10"),
    ...                       "c3_waste_processing_kgco2e": Decimal("0.30"),
    ...                       "c4_disposal_kgco2e": Decimal("0.20"),
    ...                       "verification_status": "third_party_verified"}},
    ...         {"product_id": "P-002", "category": "packaging",
    ...          "units_sold": 50000, "weight_per_unit_kg": Decimal("0.05")},
    ...     ],
    ...     org_id="ORG-001", year=2025)
    >>> result["total_co2e_kg"] > Decimal("0")
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-012
"""

import hashlib
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# GRACEFUL IMPORTS
# ==============================================================================

try:
    from greenlang.end_of_life_treatment.config import get_config
except ImportError:
    def get_config() -> Any:  # type: ignore[misc]
        """Fallback configuration stub."""
        return None

try:
    from greenlang.end_of_life_treatment.metrics import get_metrics
except ImportError:
    def get_metrics() -> Any:  # type: ignore[misc]
        """Fallback metrics stub."""
        return None

try:
    from greenlang.end_of_life_treatment.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment,misc]

try:
    from greenlang.end_of_life_treatment.average_data_calculator import (
        AverageDataCalculatorEngine,
        COMPOSITE_EOL_EF,
        REGIONAL_ADJUSTMENT_FACTORS,
        DEFAULT_PRODUCT_WEIGHTS,
    )
except ImportError:
    AverageDataCalculatorEngine = None  # type: ignore[assignment,misc]
    COMPOSITE_EOL_EF = {}
    REGIONAL_ADJUSTMENT_FACTORS = {}
    DEFAULT_PRODUCT_WEIGHTS = {}

try:
    from greenlang.end_of_life_treatment.producer_specific_calculator import (
        ProducerSpecificCalculatorEngine,
        VERIFICATION_LEVELS,
        RECYCLED_CONTENT_CREDIT_FACTORS,
    )
except ImportError:
    ProducerSpecificCalculatorEngine = None  # type: ignore[assignment,misc]
    VERIFICATION_LEVELS = {}
    RECYCLED_CONTENT_CREDIT_FACTORS = {}

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-012"
AGENT_COMPONENT: str = "AGENT-MRV-025"
ENGINE_ID: str = "hybrid_aggregator_engine"
ENGINE_VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_eol_"

# ==============================================================================
# DECIMAL CONSTANTS
# ==============================================================================

PRECISION: int = 6
ROUNDING: str = ROUND_HALF_UP
_QUANT_6DP: Decimal = Decimal("0.000001")
_QUANT_2DP: Decimal = Decimal("0.01")
_ZERO: Decimal = Decimal("0")
_ONE: Decimal = Decimal("1")
_TWO: Decimal = Decimal("2")
_HUNDRED: Decimal = Decimal("100")
KG_PER_TONNE: Decimal = Decimal("1000")
TONNES_PER_KG: Decimal = Decimal("0.001")

# ==============================================================================
# DATA TABLE: CALCULATION METHOD HIERARCHY
# ==============================================================================

METHOD_HIERARCHY: Dict[str, Dict[str, Any]] = {
    "producer_specific": {
        "priority": 1,
        "tier": "tier_1",
        "description": "Producer-specific EPD/PCF data (highest quality)",
        "requires": ["epd_data", "pcf_data", "declared_eol_kgco2e_per_unit"],
        "uncertainty_pct": Decimal("0.10"),
    },
    "waste_type_specific": {
        "priority": 2,
        "tier": "tier_2",
        "description": "Waste-type-specific material composition data",
        "requires": ["material_composition", "treatment_split"],
        "uncertainty_pct": Decimal("0.25"),
    },
    "average_data": {
        "priority": 3,
        "tier": "tier_2",
        "description": "Product category average composite EFs",
        "requires": ["category"],
        "uncertainty_pct": Decimal("0.40"),
    },
    "spend_based": {
        "priority": 4,
        "tier": "tier_3",
        "description": "EEIO spend-based fallback (lowest quality)",
        "requires": ["eol_spend_usd"],
        "uncertainty_pct": Decimal("0.75"),
    },
}

# ==============================================================================
# DATA TABLE: SPEND-BASED EEIO FALLBACK FACTORS
# ==============================================================================
# Placeholder EEIO factors for waste management services (kgCO2e per USD).
# Used only when no other data is available.

SPEND_BASED_EOL_EF: Dict[str, Dict[str, Any]] = {
    "waste_management_general": {
        "ef_kgco2e_per_usd": Decimal("0.480"),
        "naics": "562000",
        "description": "General waste management services",
    },
    "recycling_services": {
        "ef_kgco2e_per_usd": Decimal("0.280"),
        "naics": "562920",
        "description": "Materials recovery facilities",
    },
    "hazardous_waste": {
        "ef_kgco2e_per_usd": Decimal("0.650"),
        "naics": "562211",
        "description": "Hazardous waste treatment and disposal",
    },
    "landfill_services": {
        "ef_kgco2e_per_usd": Decimal("0.580"),
        "naics": "562212",
        "description": "Solid waste landfill",
    },
    "incineration_services": {
        "ef_kgco2e_per_usd": Decimal("0.720"),
        "naics": "562213",
        "description": "Solid waste combustors and incinerators",
    },
}

# ==============================================================================
# DATA TABLE: AVOIDED EMISSIONS FACTORS
# ==============================================================================
# Material substitution benefits from recycling (kgCO2e avoided per kg).
# Source: EPA WARM v16, ecoinvent 3.9.

RECYCLING_AVOIDED_EF: Dict[str, Decimal] = {
    "plastics_pet": Decimal("1.710"),
    "plastics_hdpe": Decimal("0.860"),
    "plastics_ldpe": Decimal("0.980"),
    "plastics_pp": Decimal("0.870"),
    "plastics_mixed": Decimal("1.080"),
    "aluminum": Decimal("11.060"),
    "steel": Decimal("2.020"),
    "glass": Decimal("0.308"),
    "paper_cardboard": Decimal("1.718"),
    "textiles": Decimal("2.520"),
    "wood": Decimal("0.273"),
    "rubber": Decimal("1.600"),
    "electronics": Decimal("5.300"),
    "mixed_materials": Decimal("0.750"),
}

# Energy recovery avoided EF (kgCO2e per MJ recovered)
# Displaced grid electricity emission factor.
ENERGY_RECOVERY_AVOIDED_EF: Dict[str, Decimal] = {
    "US": Decimal("0.116"),   # US average grid EF (kgCO2e/MJ)
    "EU": Decimal("0.072"),   # EU average
    "DE": Decimal("0.098"),   # Germany
    "GB": Decimal("0.058"),   # UK
    "JP": Decimal("0.125"),   # Japan
    "CN": Decimal("0.168"),   # China
    "IN": Decimal("0.195"),   # India
    "GLOBAL": Decimal("0.130"),
}

# Net calorific values for WtE energy recovery (MJ/kg)
NET_CALORIFIC_VALUES: Dict[str, Decimal] = {
    "mixed_waste": Decimal("10.0"),
    "plastics": Decimal("35.0"),
    "paper_cardboard": Decimal("14.0"),
    "wood": Decimal("15.0"),
    "textiles": Decimal("16.5"),
    "rubber": Decimal("22.5"),
    "food_waste": Decimal("4.5"),
    "default": Decimal("10.0"),
}

# WtE plant thermal efficiency
WTE_EFFICIENCY: Dict[str, Decimal] = {
    "electrical": Decimal("0.22"),
    "thermal": Decimal("0.50"),
    "combined": Decimal("0.72"),
    "default": Decimal("0.22"),
}

# ==============================================================================
# DATA TABLE: EU WASTE HIERARCHY WEIGHTS
# ==============================================================================
# EU Waste Framework Directive 2008/98/EC Article 4 waste hierarchy.
# Higher weight = more preferred pathway.

WASTE_HIERARCHY_WEIGHTS: Dict[str, Decimal] = {
    "prevention": Decimal("5.0"),
    "reuse": Decimal("4.0"),
    "recycling": Decimal("3.0"),
    "recovery": Decimal("2.0"),      # energy recovery / other recovery
    "composting": Decimal("3.0"),     # equivalent to recycling
    "anaerobic_digestion": Decimal("2.5"),
    "incineration_wte": Decimal("2.0"),
    "incineration": Decimal("1.0"),   # without energy recovery
    "landfill": Decimal("0.5"),
    "open_burning": Decimal("0.0"),
    "disposal": Decimal("0.5"),       # generic disposal
}

# Maximum possible hierarchy score (all waste at prevention level)
MAX_HIERARCHY_SCORE: Decimal = Decimal("5.0")

# ==============================================================================
# GWP VALUES
# ==============================================================================

GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "ar4": {"co2": Decimal("1"), "ch4": Decimal("25"), "n2o": Decimal("298")},
    "ar5": {"co2": Decimal("1"), "ch4": Decimal("28"), "n2o": Decimal("265")},
    "ar6": {"co2": Decimal("1"), "ch4": Decimal("27.9"), "n2o": Decimal("273")},
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _round_decimal(value: Decimal, precision: int = PRECISION) -> Decimal:
    """
    Round a Decimal to the specified number of decimal places.

    Args:
        value: Decimal value to round.
        precision: Number of decimal places.

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * precision
    return value.quantize(Decimal(quantize_str), rounding=ROUNDING)


def _compute_hash(data: str) -> str:
    """
    Compute SHA-256 hash for provenance tracking.

    Args:
        data: String data to hash.

    Returns:
        SHA-256 hex digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "eol_hyb") -> str:
    """
    Generate a unique identifier with prefix.

    Args:
        prefix: Identifier prefix.

    Returns:
        Unique ID string.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _safe_decimal(value: Any) -> Decimal:
    """
    Safely convert a value to Decimal.

    Args:
        value: Value to convert.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If conversion fails.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(
            f"Cannot convert {value!r} (type={type(value).__name__}) to Decimal"
        ) from exc


# ==============================================================================
# ENGINE CLASS
# ==============================================================================


class HybridAggregatorEngine:
    """
    Engine 5: Hybrid multi-method aggregator with avoided emissions and circularity.

    Orchestrates the method waterfall for GHG Protocol Scope 3 Category 12,
    selecting the best calculation method per product based on data
    availability. Also calculates avoided emissions (reported SEPARATELY)
    and circularity metrics.

    Method Waterfall:
        1. Producer-specific (EPD/PCF available) -- Tier 1
        2. Waste-type-specific (material composition known) -- Tier 2
        3. Average-data (product category known) -- Tier 2/3
        4. Spend-based EEIO (fallback) -- Tier 3

    Thread Safety:
        Singleton with threading.RLock() and double-checked locking.

    Zero-Hallucination:
        All calculations are deterministic Decimal arithmetic.

    Attributes:
        _avg_engine: AverageDataCalculatorEngine singleton.
        _ps_engine: ProducerSpecificCalculatorEngine singleton.
        _calculation_count: Running count of calculations.

    Example:
        >>> engine = HybridAggregatorEngine.get_instance()
        >>> result = engine.calculate(products=[...], org_id="ORG-001", year=2025)
    """

    _instance: Optional["HybridAggregatorEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "HybridAggregatorEngine":
        """Thread-safe singleton instantiation with double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self, gwp_version: str = "ar5") -> None:
        """
        Initialize the HybridAggregatorEngine.

        Lazily resolves child engine singletons for average-data and
        producer-specific calculations.

        Args:
            gwp_version: IPCC assessment report version for GWP values.
        """
        if self._initialized:
            return

        self._gwp_version: str = gwp_version
        gwp_table = GWP_VALUES.get(gwp_version, GWP_VALUES["ar5"])
        self._gwp_ch4: Decimal = gwp_table["ch4"]
        self._gwp_n2o: Decimal = gwp_table["n2o"]

        self._config = get_config()
        self._metrics = get_metrics()
        self._calculation_count: int = 0
        self._batch_count: int = 0
        self._count_lock: threading.RLock = threading.RLock()

        # Lazy-loaded child engines
        self._avg_engine: Optional[Any] = None
        self._ps_engine: Optional[Any] = None

        self._initialized: bool = True

        logger.info(
            "HybridAggregatorEngine initialized: engine=%s, version=%s, "
            "gwp=%s, methods=%d",
            ENGINE_ID,
            ENGINE_VERSION,
            gwp_version,
            len(METHOD_HIERARCHY),
        )

    # ==========================================================================
    # SINGLETON MANAGEMENT
    # ==========================================================================

    @classmethod
    def get_instance(
        cls, gwp_version: str = "ar5"
    ) -> "HybridAggregatorEngine":
        """
        Get singleton instance with thread-safe double-checked locking.

        Args:
            gwp_version: IPCC GWP version (only used on first instantiation).

        Returns:
            HybridAggregatorEngine singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls(gwp_version=gwp_version)
        return cls._instance  # type: ignore[return-value]

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance. Used in testing only."""
        with cls._lock:
            cls._instance = None
            logger.info("HybridAggregatorEngine singleton reset")

    # ==========================================================================
    # INTERNAL HELPERS
    # ==========================================================================

    def _get_avg_engine(self) -> Any:
        """Lazy-load the AverageDataCalculatorEngine singleton."""
        if self._avg_engine is None and AverageDataCalculatorEngine is not None:
            self._avg_engine = AverageDataCalculatorEngine.get_instance(
                gwp_version=self._gwp_version
            )
        return self._avg_engine

    def _get_ps_engine(self) -> Any:
        """Lazy-load the ProducerSpecificCalculatorEngine singleton."""
        if self._ps_engine is None and ProducerSpecificCalculatorEngine is not None:
            self._ps_engine = ProducerSpecificCalculatorEngine.get_instance(
                gwp_version=self._gwp_version
            )
        return self._ps_engine

    def _increment_calculation_count(self) -> int:
        """Increment and return the calculation counter thread-safely."""
        with self._count_lock:
            self._calculation_count += 1
            return self._calculation_count

    def _record_metrics(
        self,
        method: str,
        co2e_kg: Decimal,
        duration: float,
        status: str,
    ) -> None:
        """Record Prometheus metrics for calculation."""
        if self._metrics is None:
            return
        try:
            self._metrics.record_calculation(
                engine=ENGINE_ID,
                method=method,
                category="hybrid",
                co2e_kg=float(co2e_kg),
                duration=duration,
                status=status,
            )
        except Exception as exc:
            logger.debug("Metrics recording failed (non-critical): %s", exc)

    # ==========================================================================
    # METHOD SELECTION
    # ==========================================================================

    def select_best_method(self, product: Dict[str, Any]) -> str:
        """
        Select the best calculation method for a product based on data availability.

        Waterfall logic:
            1. If epd_data, pcf_data, or declared_eol_kgco2e_per_unit --> producer_specific
            2. If material_composition and treatment_split --> waste_type_specific
            3. If category is known --> average_data
            4. If eol_spend_usd --> spend_based
            5. Fallback to average_data with mixed_products

        Args:
            product: Product dictionary.

        Returns:
            Method name string.
        """
        # Check producer-specific data
        has_epd = product.get("epd_data") is not None
        has_pcf = product.get("pcf_data") is not None
        has_declared = product.get("declared_eol_kgco2e_per_unit") is not None
        if has_epd or has_pcf or has_declared:
            return "producer_specific"

        # Check waste-type-specific data
        has_composition = product.get("material_composition") is not None
        has_treatment = product.get("treatment_split") is not None
        if has_composition and has_treatment:
            return "waste_type_specific"

        # Check average-data (category available)
        has_category = product.get("category") is not None
        if has_category:
            return "average_data"

        # Check spend-based
        has_spend = product.get("eol_spend_usd") is not None
        if has_spend:
            return "spend_based"

        # Fallback to average_data with mixed_products
        return "average_data"

    # ==========================================================================
    # CORE CALCULATION
    # ==========================================================================

    def calculate(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        year: int,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate hybrid EOL emissions using method waterfall.

        Selects the best method per product based on data availability,
        then aggregates results.

        Args:
            products: List of product dictionaries.
            org_id: Organization identifier.
            year: Reporting year.
            region: Optional region for average-data adjustment.

        Returns:
            Comprehensive result dictionary with per-product results,
            method summary, avoided emissions, and circularity metrics.
        """
        start_time = time.monotonic()
        calc_id = _generate_id("eol_hyb")

        logger.info(
            "HybridAggregatorEngine.calculate: calc_id=%s, org=%s, "
            "year=%d, products=%d, region=%s",
            calc_id, org_id, year, len(products), region,
        )

        self._validate_inputs_internal(products, org_id, year)

        product_results: List[Dict[str, Any]] = []
        total_co2e_kg = _ZERO
        total_weight_kg = _ZERO
        errors: List[Dict[str, str]] = []
        method_counts: Dict[str, int] = {}
        method_emissions: Dict[str, Decimal] = {}

        for idx, product in enumerate(products):
            try:
                method = self.select_best_method(product)
                result = self._calculate_by_method(
                    product, method, idx, region
                )
                product_results.append(result)
                total_co2e_kg += result["co2e_kg"]
                total_weight_kg += result.get("weight_kg", _ZERO)

                method_counts[method] = method_counts.get(method, 0) + 1
                method_emissions[method] = (
                    method_emissions.get(method, _ZERO) + result["co2e_kg"]
                )
            except Exception as exc:
                product_id = product.get("product_id", f"product_{idx}")
                logger.error(
                    "Hybrid calculation failed: product=%s, error=%s",
                    product_id, str(exc),
                )
                errors.append({
                    "product_id": product_id,
                    "error": str(exc),
                })

        total_co2e_kg = _round_decimal(total_co2e_kg)
        total_co2e_tonnes = _round_decimal(total_co2e_kg * TONNES_PER_KG)
        total_weight_kg = _round_decimal(total_weight_kg, 2)

        # Compute avoided emissions
        avoided = self.calculate_avoided_emissions(product_results, region)

        # Compute circularity metrics
        circularity = self.calculate_circularity_score(product_results)

        # Compute waste hierarchy compliance
        hierarchy = self.calculate_waste_hierarchy_compliance(product_results)

        # Compute DQI and uncertainty
        dqi = self.compute_dqi_score(method_counts, method_emissions)
        uncertainty = self.compute_uncertainty(
            total_co2e_kg, method_counts, method_emissions
        )

        # Method summary
        method_summary = []
        for method_name, count in sorted(
            method_counts.items(),
            key=lambda x: METHOD_HIERARCHY.get(x[0], {}).get("priority", 99),
        ):
            emissions = method_emissions.get(method_name, _ZERO)
            pct = (
                _round_decimal(emissions / total_co2e_kg * _HUNDRED, 2)
                if total_co2e_kg > _ZERO
                else _ZERO
            )
            method_summary.append({
                "method": method_name,
                "product_count": count,
                "co2e_kg": _round_decimal(emissions),
                "percentage_of_total": pct,
                "tier": METHOD_HIERARCHY.get(method_name, {}).get(
                    "tier", "tier_3"
                ),
            })

        provenance_data = (
            f"{calc_id}|{org_id}|{year}|hybrid|"
            f"{total_co2e_kg}|{total_weight_kg}|"
            f"{len(product_results)}|{len(errors)}|"
            f"{','.join(sorted(method_counts.keys()))}"
        )
        provenance_hash = _compute_hash(provenance_data)

        duration = time.monotonic() - start_time
        self._record_metrics(
            method="hybrid",
            co2e_kg=total_co2e_kg,
            duration=duration,
            status="success" if not errors else "partial",
        )
        count = self._increment_calculation_count()

        return {
            "calculation_id": calc_id,
            "org_id": org_id,
            "year": year,
            "region": region,
            "method": "hybrid",
            "method_description": (
                "GHG Protocol Scope 3 Category 12 hybrid method. "
                "Selects best method per product based on data availability."
            ),
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "total_weight_kg": total_weight_kg,
            "product_count": len(products),
            "success_count": len(product_results),
            "error_count": len(errors),
            "product_results": product_results,
            "errors": errors,
            "method_summary": method_summary,
            "method_counts": method_counts,
            "avoided_emissions": avoided,
            "circularity_metrics": circularity,
            "waste_hierarchy_compliance": hierarchy,
            "dqi_score": dqi,
            "uncertainty": uncertainty,
            "gwp_version": self._gwp_version,
            "provenance_hash": provenance_hash,
            "agent_id": AGENT_ID,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "calculation_number": count,
            "processing_time_ms": round(duration * 1000, 2),
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ==========================================================================
    # PER-METHOD CALCULATION DISPATCH
    # ==========================================================================

    def _calculate_by_method(
        self,
        product: Dict[str, Any],
        method: str,
        index: int,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a single product using the specified method.

        Args:
            product: Product dictionary.
            method: Calculation method name.
            index: Product index.
            region: Optional region for adjustment.

        Returns:
            Per-product result dictionary.
        """
        product_id = product.get("product_id", f"product_{index}")
        units_sold = int(product.get("units_sold", 0))
        if units_sold <= 0:
            raise ValueError(f"Product '{product_id}': units_sold must be > 0")

        if method == "producer_specific":
            return self._calc_producer_specific(product, product_id, units_sold)

        elif method == "waste_type_specific":
            return self._calc_waste_type_specific(
                product, product_id, units_sold
            )

        elif method == "average_data":
            return self._calc_average_data(
                product, product_id, units_sold, region
            )

        elif method == "spend_based":
            return self._calc_spend_based(product, product_id, units_sold)

        else:
            # Fallback to average data
            logger.warning(
                "Unknown method '%s' for product '%s'; falling back to "
                "average_data",
                method, product_id,
            )
            return self._calc_average_data(
                product, product_id, units_sold, region
            )

    def _calc_producer_specific(
        self,
        product: Dict[str, Any],
        product_id: str,
        units_sold: int,
    ) -> Dict[str, Any]:
        """Calculate using producer-specific data (EPD/PCF/declared)."""
        ps_engine = self._get_ps_engine()
        epd_data = product.get("epd_data")
        pcf_data = product.get("pcf_data")
        declared_eol = product.get("declared_eol_kgco2e_per_unit")

        eol_per_unit = _ZERO
        data_source = "none"
        verification = "default"
        module_d_memo = _ZERO
        eol_breakdown: Dict[str, Any] = {}

        if epd_data is not None:
            if ps_engine is not None:
                scenario = ps_engine.parse_eol_scenario(epd_data)
                eol_per_unit = scenario["total_eol_kgco2e"]
                module_d_memo = scenario.get("module_d_benefits", _ZERO)
                eol_breakdown = scenario.get("module_breakdown", {})
            else:
                # Manual EPD parsing without engine
                c1 = _safe_decimal(
                    epd_data.get("c1_deconstruction_kgco2e", _ZERO)
                )
                c2 = _safe_decimal(
                    epd_data.get("c2_transport_kgco2e", _ZERO)
                )
                c3 = _safe_decimal(
                    epd_data.get("c3_waste_processing_kgco2e", _ZERO)
                )
                c4 = _safe_decimal(
                    epd_data.get("c4_disposal_kgco2e", _ZERO)
                )
                eol_per_unit = c1 + c2 + c3 + c4
                module_d_memo = _safe_decimal(
                    epd_data.get("d_benefits_kgco2e", _ZERO)
                )
                eol_breakdown = {
                    "C1_deconstruction": c1,
                    "C2_transport": c2,
                    "C3_waste_processing": c3,
                    "C4_disposal": c4,
                }
            data_source = "epd"
            verification = epd_data.get(
                "verification_status", "self_declared"
            )

        elif pcf_data is not None:
            eol_per_unit = _safe_decimal(
                pcf_data.get("eol_emissions_kgco2e", _ZERO)
            )
            data_source = "pcf"
            verification = pcf_data.get("verification_status", "self_declared")
            module_d_memo = _safe_decimal(
                pcf_data.get("avoided_emissions_kgco2e_memo", _ZERO)
            )
            eol_breakdown = {"eol_total": eol_per_unit}

        elif declared_eol is not None:
            eol_per_unit = _safe_decimal(declared_eol)
            data_source = "declared"
            verification = product.get(
                "verification_status", "self_declared"
            )
            eol_breakdown = {"declared_eol_per_unit": eol_per_unit}

        co2e_kg = _round_decimal(eol_per_unit * Decimal(str(units_sold)))
        weight_per_unit = product.get("weight_per_unit_kg")
        weight_kg = _ZERO
        if weight_per_unit is not None:
            weight_kg = _round_decimal(
                _safe_decimal(weight_per_unit) * Decimal(str(units_sold)), 2
            )

        module_d_total = _round_decimal(
            module_d_memo * Decimal(str(units_sold))
        )

        provenance_data = (
            f"{product_id}|producer_specific|{data_source}|"
            f"{units_sold}|{eol_per_unit}|{co2e_kg}"
        )

        return {
            "product_id": product_id,
            "method": "producer_specific",
            "data_source": data_source,
            "verification_status": verification,
            "tier": VERIFICATION_LEVELS.get(verification, {}).get(
                "tier", "tier_3"
            ) if VERIFICATION_LEVELS else "tier_1",
            "units_sold": units_sold,
            "weight_per_unit_kg": weight_per_unit,
            "weight_kg": weight_kg,
            "eol_per_unit_kgco2e": _round_decimal(eol_per_unit),
            "co2e_kg": co2e_kg,
            "co2e_tonnes": _round_decimal(co2e_kg * TONNES_PER_KG),
            "eol_breakdown": eol_breakdown,
            "module_d_benefits_memo_kgco2e": module_d_total,
            "category": product.get("category", "unknown"),
            "treatment_type": product.get("treatment_type", "mixed"),
            "material_type": product.get("material_type", "unknown"),
            "provenance_hash": _compute_hash(provenance_data),
        }

    def _calc_waste_type_specific(
        self,
        product: Dict[str, Any],
        product_id: str,
        units_sold: int,
    ) -> Dict[str, Any]:
        """Calculate using waste-type-specific material composition data."""
        weight_per_unit = _safe_decimal(
            product.get("weight_per_unit_kg", _ZERO)
        )
        if weight_per_unit <= _ZERO:
            raise ValueError(
                f"Product '{product_id}': weight_per_unit_kg required "
                f"for waste_type_specific method"
            )

        total_weight_kg = _round_decimal(
            weight_per_unit * Decimal(str(units_sold)), 2
        )

        composition = product.get("material_composition", [])
        treatment_split = product.get("treatment_split", {})

        total_co2e = _ZERO
        material_results: List[Dict[str, Any]] = []

        for mat in composition:
            mat_type = mat.get("material_type", "mixed_materials")
            mat_fraction = _safe_decimal(mat.get("fraction", _ZERO))
            mat_weight = _round_decimal(total_weight_kg * mat_fraction, 2)
            mat_ef = _safe_decimal(
                mat.get("ef_kgco2e_per_kg", Decimal("0.587"))
            )

            mat_co2e = _round_decimal(mat_weight * mat_ef)
            total_co2e += mat_co2e

            material_results.append({
                "material_type": mat_type,
                "fraction": mat_fraction,
                "weight_kg": mat_weight,
                "ef_kgco2e_per_kg": mat_ef,
                "co2e_kg": mat_co2e,
            })

        total_co2e = _round_decimal(total_co2e)

        provenance_data = (
            f"{product_id}|waste_type_specific|{units_sold}|"
            f"{total_weight_kg}|{total_co2e}"
        )

        return {
            "product_id": product_id,
            "method": "waste_type_specific",
            "data_source": "material_composition",
            "verification_status": "estimated",
            "tier": "tier_2",
            "units_sold": units_sold,
            "weight_per_unit_kg": weight_per_unit,
            "weight_kg": total_weight_kg,
            "co2e_kg": total_co2e,
            "co2e_tonnes": _round_decimal(total_co2e * TONNES_PER_KG),
            "material_breakdown": material_results,
            "treatment_split": treatment_split,
            "category": product.get("category", "unknown"),
            "treatment_type": product.get("treatment_type", "mixed"),
            "material_type": product.get("primary_material", "mixed"),
            "provenance_hash": _compute_hash(provenance_data),
        }

    def _calc_average_data(
        self,
        product: Dict[str, Any],
        product_id: str,
        units_sold: int,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate using average-data composite EFs."""
        category = product.get("category", "mixed_products")
        weight_per_unit = product.get("weight_per_unit_kg")
        weight_estimated = False

        # Resolve weight
        if weight_per_unit is not None:
            weight_per_unit = _safe_decimal(weight_per_unit)
        else:
            weight_estimated = True
            weight_per_unit = DEFAULT_PRODUCT_WEIGHTS.get(
                category,
                DEFAULT_PRODUCT_WEIGHTS.get("mixed_products", _ONE),
            ) if DEFAULT_PRODUCT_WEIGHTS else _ONE

        total_weight_kg = _round_decimal(
            weight_per_unit * Decimal(str(units_sold)), 2
        )

        # Resolve EF
        ef = Decimal("0.587")  # Default fallback
        if COMPOSITE_EOL_EF:
            entry = COMPOSITE_EOL_EF.get(category)
            if entry is not None:
                ef = entry["ef_kgco2e_per_kg"]
            else:
                fallback = COMPOSITE_EOL_EF.get("mixed_products")
                if fallback is not None:
                    ef = fallback["ef_kgco2e_per_kg"]

        # Apply regional adjustment
        regional_factor = _ONE
        if region and REGIONAL_ADJUSTMENT_FACTORS:
            region_entry = REGIONAL_ADJUSTMENT_FACTORS.get(region)
            if region_entry is not None:
                regional_factor = region_entry["factor"]
            else:
                global_entry = REGIONAL_ADJUSTMENT_FACTORS.get("GLOBAL")
                if global_entry is not None:
                    regional_factor = global_entry["factor"]

        adjusted_ef = _round_decimal(ef * regional_factor)
        co2e_kg = _round_decimal(total_weight_kg * adjusted_ef)

        provenance_data = (
            f"{product_id}|average_data|{category}|{units_sold}|"
            f"{total_weight_kg}|{adjusted_ef}|{co2e_kg}"
        )

        return {
            "product_id": product_id,
            "method": "average_data",
            "data_source": "composite_eol_ef",
            "verification_status": "default",
            "tier": "tier_2",
            "units_sold": units_sold,
            "weight_per_unit_kg": weight_per_unit,
            "weight_estimated": weight_estimated,
            "weight_kg": total_weight_kg,
            "category": category,
            "ef_kgco2e_per_kg": ef,
            "regional_factor": regional_factor,
            "adjusted_ef_kgco2e_per_kg": adjusted_ef,
            "co2e_kg": co2e_kg,
            "co2e_tonnes": _round_decimal(co2e_kg * TONNES_PER_KG),
            "treatment_type": product.get("treatment_type", "mixed"),
            "material_type": product.get("material_type", "unknown"),
            "provenance_hash": _compute_hash(provenance_data),
        }

    def _calc_spend_based(
        self,
        product: Dict[str, Any],
        product_id: str,
        units_sold: int,
    ) -> Dict[str, Any]:
        """Calculate using spend-based EEIO fallback."""
        eol_spend = _safe_decimal(product.get("eol_spend_usd", _ZERO))
        if eol_spend <= _ZERO:
            raise ValueError(
                f"Product '{product_id}': eol_spend_usd must be > 0 "
                f"for spend-based method"
            )

        service_type = product.get(
            "waste_service_type", "waste_management_general"
        )
        ef_entry = SPEND_BASED_EOL_EF.get(
            service_type,
            SPEND_BASED_EOL_EF["waste_management_general"],
        )
        eeio_ef = ef_entry["ef_kgco2e_per_usd"]

        co2e_kg = _round_decimal(eol_spend * eeio_ef)

        weight_per_unit = product.get("weight_per_unit_kg")
        weight_kg = _ZERO
        if weight_per_unit is not None:
            weight_kg = _round_decimal(
                _safe_decimal(weight_per_unit) * Decimal(str(units_sold)), 2
            )

        provenance_data = (
            f"{product_id}|spend_based|{service_type}|"
            f"{eol_spend}|{eeio_ef}|{co2e_kg}"
        )

        return {
            "product_id": product_id,
            "method": "spend_based",
            "data_source": "eeio",
            "verification_status": "default",
            "tier": "tier_3",
            "units_sold": units_sold,
            "weight_per_unit_kg": weight_per_unit,
            "weight_kg": weight_kg,
            "eol_spend_usd": eol_spend,
            "eeio_ef_kgco2e_per_usd": eeio_ef,
            "service_type": service_type,
            "co2e_kg": co2e_kg,
            "co2e_tonnes": _round_decimal(co2e_kg * TONNES_PER_KG),
            "category": product.get("category", "unknown"),
            "treatment_type": product.get("treatment_type", "mixed"),
            "material_type": product.get("material_type", "unknown"),
            "provenance_hash": _compute_hash(provenance_data),
        }

    # ==========================================================================
    # AGGREGATION METHODS
    # ==========================================================================

    def aggregate_by_treatment(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate product results by treatment type.

        Args:
            results: List of per-product result dicts.

        Returns:
            Aggregation dictionary by treatment type.
        """
        treatment_map: Dict[str, Dict[str, Decimal]] = {}
        for r in results:
            treatment = r.get("treatment_type", "mixed")
            if treatment not in treatment_map:
                treatment_map[treatment] = {
                    "co2e_kg": _ZERO,
                    "weight_kg": _ZERO,
                    "count": _ZERO,
                }
            treatment_map[treatment]["co2e_kg"] += r.get("co2e_kg", _ZERO)
            treatment_map[treatment]["weight_kg"] += r.get("weight_kg", _ZERO)
            treatment_map[treatment]["count"] += _ONE

        total_co2e = sum(v["co2e_kg"] for v in treatment_map.values())

        treatments = []
        for key, vals in sorted(
            treatment_map.items(),
            key=lambda x: x[1]["co2e_kg"],
            reverse=True,
        ):
            pct = (
                _round_decimal(vals["co2e_kg"] / total_co2e * _HUNDRED, 2)
                if total_co2e > _ZERO
                else _ZERO
            )
            treatments.append({
                "treatment_type": key,
                "co2e_kg": _round_decimal(vals["co2e_kg"]),
                "weight_kg": _round_decimal(vals["weight_kg"], 2),
                "product_count": int(vals["count"]),
                "percentage_of_total": pct,
            })

        return {
            "dimension": "treatment_type",
            "groups": treatments,
            "total_co2e_kg": _round_decimal(total_co2e),
        }

    def aggregate_by_material(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate product results by material type.

        Args:
            results: List of per-product result dicts.

        Returns:
            Aggregation dictionary by material type.
        """
        material_map: Dict[str, Dict[str, Decimal]] = {}
        for r in results:
            material = r.get("material_type", "unknown")
            if material not in material_map:
                material_map[material] = {
                    "co2e_kg": _ZERO,
                    "weight_kg": _ZERO,
                    "count": _ZERO,
                }
            material_map[material]["co2e_kg"] += r.get("co2e_kg", _ZERO)
            material_map[material]["weight_kg"] += r.get("weight_kg", _ZERO)
            material_map[material]["count"] += _ONE

        total_co2e = sum(v["co2e_kg"] for v in material_map.values())

        materials = []
        for key, vals in sorted(
            material_map.items(),
            key=lambda x: x[1]["co2e_kg"],
            reverse=True,
        ):
            pct = (
                _round_decimal(vals["co2e_kg"] / total_co2e * _HUNDRED, 2)
                if total_co2e > _ZERO
                else _ZERO
            )
            materials.append({
                "material_type": key,
                "co2e_kg": _round_decimal(vals["co2e_kg"]),
                "weight_kg": _round_decimal(vals["weight_kg"], 2),
                "product_count": int(vals["count"]),
                "percentage_of_total": pct,
            })

        return {
            "dimension": "material_type",
            "groups": materials,
            "total_co2e_kg": _round_decimal(total_co2e),
        }

    def aggregate_by_category(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate product results by product category.

        Args:
            results: List of per-product result dicts.

        Returns:
            Aggregation dictionary by product category.
        """
        category_map: Dict[str, Dict[str, Decimal]] = {}
        for r in results:
            cat = r.get("category", "unknown")
            if cat not in category_map:
                category_map[cat] = {
                    "co2e_kg": _ZERO,
                    "weight_kg": _ZERO,
                    "count": _ZERO,
                }
            category_map[cat]["co2e_kg"] += r.get("co2e_kg", _ZERO)
            category_map[cat]["weight_kg"] += r.get("weight_kg", _ZERO)
            category_map[cat]["count"] += _ONE

        total_co2e = sum(v["co2e_kg"] for v in category_map.values())

        categories = []
        for key, vals in sorted(
            category_map.items(),
            key=lambda x: x[1]["co2e_kg"],
            reverse=True,
        ):
            pct = (
                _round_decimal(vals["co2e_kg"] / total_co2e * _HUNDRED, 2)
                if total_co2e > _ZERO
                else _ZERO
            )
            categories.append({
                "category": key,
                "co2e_kg": _round_decimal(vals["co2e_kg"]),
                "weight_kg": _round_decimal(vals["weight_kg"], 2),
                "product_count": int(vals["count"]),
                "percentage_of_total": pct,
            })

        return {
            "dimension": "product_category",
            "groups": categories,
            "total_co2e_kg": _round_decimal(total_co2e),
        }

    # ==========================================================================
    # AVOIDED EMISSIONS
    # ==========================================================================

    def calculate_avoided_emissions(
        self,
        results: List[Dict[str, Any]],
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate avoided emissions from recycling and energy recovery.

        IMPORTANT: Avoided emissions are reported SEPARATELY per GHG Protocol.
        They must NEVER be netted from gross emissions.

        Args:
            results: List of per-product result dicts.
            region: Region for energy recovery grid displacement factor.

        Returns:
            Avoided emissions breakdown dictionary.
        """
        total_recycling_avoided = _ZERO
        total_energy_avoided = _ZERO
        recycling_details: List[Dict[str, Any]] = []
        energy_details: List[Dict[str, Any]] = []

        for r in results:
            product_id = r.get("product_id", "unknown")
            weight_kg = r.get("weight_kg", _ZERO)
            treatment = r.get("treatment_type", "mixed")
            material = r.get("material_type", "unknown")

            # Module D from EPD (already calculated)
            module_d = r.get("module_d_benefits_memo_kgco2e", _ZERO)
            if module_d != _ZERO:
                total_recycling_avoided += abs(module_d)
                recycling_details.append({
                    "product_id": product_id,
                    "source": "epd_module_d",
                    "avoided_kgco2e": abs(module_d),
                })
                continue

            # Estimate recycling avoided emissions from material type
            if treatment in ("recycling", "mixed") and weight_kg > _ZERO:
                recycling_fraction = Decimal("0.35")  # Default
                if treatment == "recycling":
                    recycling_fraction = _ONE

                recycled_weight = _round_decimal(
                    weight_kg * recycling_fraction, 2
                )
                credit_per_kg = RECYCLING_AVOIDED_EF.get(
                    material,
                    RECYCLING_AVOIDED_EF.get("mixed_materials", _ZERO),
                )
                avoided = _round_decimal(recycled_weight * credit_per_kg)
                if avoided > _ZERO:
                    total_recycling_avoided += avoided
                    recycling_details.append({
                        "product_id": product_id,
                        "source": "material_substitution",
                        "material": material,
                        "recycled_weight_kg": recycled_weight,
                        "credit_per_kg": credit_per_kg,
                        "avoided_kgco2e": avoided,
                    })

            # Estimate energy recovery avoided emissions
            if treatment in (
                "incineration_wte", "incineration", "mixed"
            ) and weight_kg > _ZERO:
                wte_fraction = Decimal("0.15")  # Default for mixed
                if treatment == "incineration_wte":
                    wte_fraction = _ONE
                elif treatment == "incineration":
                    wte_fraction = Decimal("0.50")

                wte_weight = _round_decimal(weight_kg * wte_fraction, 2)
                ncv = NET_CALORIFIC_VALUES.get(
                    material,
                    NET_CALORIFIC_VALUES.get("default", Decimal("10.0")),
                )
                efficiency = WTE_EFFICIENCY.get(
                    "default", Decimal("0.22")
                )
                energy_mj = _round_decimal(wte_weight * ncv * efficiency, 2)

                grid_ef_region = region or "GLOBAL"
                grid_ef = ENERGY_RECOVERY_AVOIDED_EF.get(
                    grid_ef_region,
                    ENERGY_RECOVERY_AVOIDED_EF["GLOBAL"],
                )
                energy_avoided = _round_decimal(energy_mj * grid_ef)

                if energy_avoided > _ZERO:
                    total_energy_avoided += energy_avoided
                    energy_details.append({
                        "product_id": product_id,
                        "source": "wte_grid_displacement",
                        "wte_weight_kg": wte_weight,
                        "energy_recovered_mj": energy_mj,
                        "grid_ef_kgco2e_per_mj": grid_ef,
                        "avoided_kgco2e": energy_avoided,
                    })

        total_avoided = _round_decimal(
            total_recycling_avoided + total_energy_avoided
        )

        return {
            "total_avoided_kgco2e_memo": total_avoided,
            "total_avoided_tonnes_memo": _round_decimal(
                total_avoided * TONNES_PER_KG
            ),
            "recycling_credits_kgco2e": _round_decimal(
                total_recycling_avoided
            ),
            "energy_recovery_credits_kgco2e": _round_decimal(
                total_energy_avoided
            ),
            "recycling_details": recycling_details,
            "energy_recovery_details": energy_details,
            "note": (
                "MEMO ITEM ONLY. Avoided emissions must NOT be deducted from "
                "gross Scope 3 Category 12 emissions per GHG Protocol "
                "Technical Guidance (Chapter 12, page 73)."
            ),
        }

    # ==========================================================================
    # CIRCULARITY METRICS
    # ==========================================================================

    def calculate_circularity_score(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate circularity metrics for all products.

        Metrics:
            - recycling_rate = recycled_weight / total_weight
            - diversion_rate = (recycled + composted + AD) / total_weight
            - circularity_index = (recycled_input + recycling_output) /
              (2 x total_throughput) -- Ellen MacArthur Foundation MCI
            - material_recovery_rate = (recycled + composted) /
              (total - energy_recovery)

        Args:
            results: List of per-product result dicts.

        Returns:
            Circularity metrics dictionary.
        """
        total_weight = _ZERO
        recycled_weight = _ZERO
        composted_weight = _ZERO
        ad_weight = _ZERO
        energy_recovery_weight = _ZERO
        landfill_weight = _ZERO
        other_weight = _ZERO

        # Default treatment fractions when treatment_type is not specified
        default_fractions = {
            "recycling": Decimal("0.35"),
            "composting": Decimal("0.05"),
            "anaerobic_digestion": Decimal("0.03"),
            "energy_recovery": Decimal("0.15"),
            "landfill": Decimal("0.35"),
            "other": Decimal("0.07"),
        }

        for r in results:
            weight = r.get("weight_kg", _ZERO)
            if weight <= _ZERO:
                continue
            total_weight += weight

            treatment = r.get("treatment_type", "mixed")

            if treatment == "recycling":
                recycled_weight += weight
            elif treatment == "composting":
                composted_weight += weight
            elif treatment == "anaerobic_digestion":
                ad_weight += weight
            elif treatment in ("incineration_wte", "energy_recovery"):
                energy_recovery_weight += weight
            elif treatment == "landfill":
                landfill_weight += weight
            elif treatment == "mixed":
                recycled_weight += _round_decimal(
                    weight * default_fractions["recycling"], 2
                )
                composted_weight += _round_decimal(
                    weight * default_fractions["composting"], 2
                )
                ad_weight += _round_decimal(
                    weight * default_fractions["anaerobic_digestion"], 2
                )
                energy_recovery_weight += _round_decimal(
                    weight * default_fractions["energy_recovery"], 2
                )
                landfill_weight += _round_decimal(
                    weight * default_fractions["landfill"], 2
                )
                other_weight += _round_decimal(
                    weight * default_fractions["other"], 2
                )
            else:
                other_weight += weight

        # Recycling rate
        recycling_rate = (
            _round_decimal(recycled_weight / total_weight, 4)
            if total_weight > _ZERO
            else _ZERO
        )

        # Diversion rate = (recycled + composted + AD) / total
        diverted = recycled_weight + composted_weight + ad_weight
        diversion_rate = (
            _round_decimal(diverted / total_weight, 4)
            if total_weight > _ZERO
            else _ZERO
        )

        # Circularity index (Ellen MacArthur Foundation simplified MCI)
        # MCI = (recycled_input + recycling_output) / (2 * total_throughput)
        # We approximate recycled_input as post-consumer recycled content
        # and recycling_output as recycled_weight
        recycled_input_estimate = _round_decimal(
            total_weight * Decimal("0.10"), 2
        )  # Conservative 10% recycled input estimate
        recycling_output = recycled_weight
        circularity_index = (
            _round_decimal(
                (recycled_input_estimate + recycling_output)
                / (_TWO * total_weight),
                4,
            )
            if total_weight > _ZERO
            else _ZERO
        )

        # Material recovery rate = (recycled + composted) / (total - energy_recovery)
        denominator = total_weight - energy_recovery_weight
        material_recovery_rate = (
            _round_decimal(
                (recycled_weight + composted_weight) / denominator, 4
            )
            if denominator > _ZERO
            else _ZERO
        )

        return {
            "total_weight_kg": _round_decimal(total_weight, 2),
            "recycled_weight_kg": _round_decimal(recycled_weight, 2),
            "composted_weight_kg": _round_decimal(composted_weight, 2),
            "ad_weight_kg": _round_decimal(ad_weight, 2),
            "energy_recovery_weight_kg": _round_decimal(
                energy_recovery_weight, 2
            ),
            "landfill_weight_kg": _round_decimal(landfill_weight, 2),
            "other_weight_kg": _round_decimal(other_weight, 2),
            "recycling_rate": recycling_rate,
            "recycling_rate_pct": _round_decimal(
                recycling_rate * _HUNDRED, 1
            ),
            "diversion_rate": diversion_rate,
            "diversion_rate_pct": _round_decimal(
                diversion_rate * _HUNDRED, 1
            ),
            "circularity_index": circularity_index,
            "circularity_index_pct": _round_decimal(
                circularity_index * _HUNDRED, 1
            ),
            "material_recovery_rate": material_recovery_rate,
            "material_recovery_rate_pct": _round_decimal(
                material_recovery_rate * _HUNDRED, 1
            ),
            "methodology": "Ellen MacArthur Foundation MCI v3.0 (simplified)",
        }

    # ==========================================================================
    # WASTE HIERARCHY COMPLIANCE
    # ==========================================================================

    def calculate_waste_hierarchy_compliance(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate EU Waste Framework Directive waste hierarchy compliance.

        Scores each treatment pathway according to the EU waste hierarchy
        (prevention > reuse > recycling > recovery > disposal) and computes
        a weighted compliance score.

        Args:
            results: List of per-product result dicts.

        Returns:
            Waste hierarchy compliance dictionary.
        """
        total_weight = _ZERO
        weighted_score_sum = _ZERO
        treatment_weights: Dict[str, Decimal] = {}

        default_fractions = {
            "recycling": Decimal("0.35"),
            "composting": Decimal("0.05"),
            "anaerobic_digestion": Decimal("0.03"),
            "incineration_wte": Decimal("0.15"),
            "incineration": Decimal("0.05"),
            "landfill": Decimal("0.30"),
            "open_burning": Decimal("0.07"),
        }

        for r in results:
            weight = r.get("weight_kg", _ZERO)
            if weight <= _ZERO:
                continue
            total_weight += weight

            treatment = r.get("treatment_type", "mixed")

            if treatment == "mixed":
                for t, frac in default_fractions.items():
                    t_weight = _round_decimal(weight * frac, 2)
                    h_score = WASTE_HIERARCHY_WEIGHTS.get(t, _ZERO)
                    weighted_score_sum += t_weight * h_score
                    treatment_weights[t] = (
                        treatment_weights.get(t, _ZERO) + t_weight
                    )
            else:
                h_score = WASTE_HIERARCHY_WEIGHTS.get(treatment, _ZERO)
                weighted_score_sum += weight * h_score
                treatment_weights[treatment] = (
                    treatment_weights.get(treatment, _ZERO) + weight
                )

        # Compute normalized score (0-100)
        hierarchy_score = _ZERO
        if total_weight > _ZERO and MAX_HIERARCHY_SCORE > _ZERO:
            raw_score = weighted_score_sum / total_weight
            hierarchy_score = _round_decimal(
                raw_score / MAX_HIERARCHY_SCORE * _HUNDRED, 1
            )

        # Grade assignment
        if hierarchy_score >= Decimal("80"):
            grade = "A"
            grade_description = "Excellent waste hierarchy compliance"
        elif hierarchy_score >= Decimal("60"):
            grade = "B"
            grade_description = "Good waste hierarchy compliance"
        elif hierarchy_score >= Decimal("40"):
            grade = "C"
            grade_description = "Moderate waste hierarchy compliance"
        elif hierarchy_score >= Decimal("20"):
            grade = "D"
            grade_description = "Poor waste hierarchy compliance"
        else:
            grade = "F"
            grade_description = "Very poor waste hierarchy compliance"

        # Treatment breakdown with hierarchy scores
        treatment_breakdown = []
        for t, tw in sorted(
            treatment_weights.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            pct = (
                _round_decimal(tw / total_weight * _HUNDRED, 1)
                if total_weight > _ZERO
                else _ZERO
            )
            treatment_breakdown.append({
                "treatment": t,
                "weight_kg": _round_decimal(tw, 2),
                "percentage": pct,
                "hierarchy_score": WASTE_HIERARCHY_WEIGHTS.get(t, _ZERO),
                "hierarchy_level": self._get_hierarchy_level(t),
            })

        return {
            "hierarchy_score": hierarchy_score,
            "grade": grade,
            "grade_description": grade_description,
            "total_weight_kg": _round_decimal(total_weight, 2),
            "treatment_breakdown": treatment_breakdown,
            "framework": "EU Waste Framework Directive 2008/98/EC Article 4",
            "recommendations": self._generate_hierarchy_recommendations(
                treatment_weights, total_weight
            ),
        }

    def _get_hierarchy_level(self, treatment: str) -> str:
        """Map treatment type to waste hierarchy level name."""
        level_map = {
            "prevention": "Prevention (Level 1 - Most preferred)",
            "reuse": "Reuse (Level 2)",
            "recycling": "Recycling (Level 3)",
            "composting": "Recycling (Level 3)",
            "anaerobic_digestion": "Recovery (Level 4)",
            "recovery": "Recovery (Level 4)",
            "incineration_wte": "Recovery (Level 4)",
            "incineration": "Disposal (Level 5)",
            "landfill": "Disposal (Level 5)",
            "open_burning": "Disposal (Level 5 - Least preferred)",
            "disposal": "Disposal (Level 5)",
        }
        return level_map.get(treatment, "Unknown")

    def _generate_hierarchy_recommendations(
        self,
        treatment_weights: Dict[str, Decimal],
        total_weight: Decimal,
    ) -> List[str]:
        """Generate improvement recommendations based on treatment mix."""
        recommendations: List[str] = []
        if total_weight <= _ZERO:
            return recommendations

        landfill_pct = _ZERO
        landfill_w = treatment_weights.get("landfill", _ZERO)
        if landfill_w > _ZERO:
            landfill_pct = landfill_w / total_weight * _HUNDRED

        incineration_w = treatment_weights.get("incineration", _ZERO)
        incin_pct = _ZERO
        if incineration_w > _ZERO:
            incin_pct = incineration_w / total_weight * _HUNDRED

        open_burn_w = treatment_weights.get("open_burning", _ZERO)

        recycling_w = treatment_weights.get("recycling", _ZERO)
        recycling_pct = _ZERO
        if recycling_w > _ZERO:
            recycling_pct = recycling_w / total_weight * _HUNDRED

        if landfill_pct > Decimal("40"):
            recommendations.append(
                f"High landfill rate ({_round_decimal(landfill_pct, 0)}%). "
                f"Target < 10% through increased recycling and composting."
            )
        if incin_pct > Decimal("30"):
            recommendations.append(
                f"Incineration without energy recovery is "
                f"{_round_decimal(incin_pct, 0)}%. Consider upgrading to WtE."
            )
        if open_burn_w > _ZERO:
            recommendations.append(
                "Open burning detected. This should be eliminated entirely."
            )
        if recycling_pct < Decimal("30"):
            recommendations.append(
                f"Recycling rate is {_round_decimal(recycling_pct, 0)}%. "
                f"Design products for recyclability (DfR) to exceed 50%."
            )
        if not recommendations:
            recommendations.append(
                "Good waste hierarchy performance. Continue monitoring and "
                "targeting prevention and reuse opportunities."
            )

        return recommendations

    # ==========================================================================
    # HOTSPOT ANALYSIS
    # ==========================================================================

    def hotspot_analysis(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Perform Pareto 80/20 hotspot analysis across products, materials,
        and treatment types.

        Identifies the products, materials, and treatments that contribute
        to 80% of total emissions.

        Args:
            results: List of per-product result dicts.

        Returns:
            Hotspot analysis dictionary with top contributors.
        """
        total_co2e = sum(r.get("co2e_kg", _ZERO) for r in results)
        if total_co2e <= _ZERO:
            return {
                "total_co2e_kg": _ZERO,
                "product_hotspots": [],
                "category_hotspots": [],
                "material_hotspots": [],
            }

        # Product hotspots
        sorted_products = sorted(
            results,
            key=lambda x: x.get("co2e_kg", _ZERO),
            reverse=True,
        )
        product_hotspots: List[Dict[str, Any]] = []
        cumulative = _ZERO
        pareto_threshold = total_co2e * Decimal("0.80")

        for r in sorted_products:
            co2e = r.get("co2e_kg", _ZERO)
            cumulative += co2e
            pct = _round_decimal(co2e / total_co2e * _HUNDRED, 2)
            cum_pct = _round_decimal(cumulative / total_co2e * _HUNDRED, 2)
            product_hotspots.append({
                "product_id": r.get("product_id", "unknown"),
                "category": r.get("category", "unknown"),
                "co2e_kg": _round_decimal(co2e),
                "percentage": pct,
                "cumulative_percentage": cum_pct,
                "is_hotspot": cumulative <= pareto_threshold,
            })

        # Category hotspots
        cat_agg = self.aggregate_by_category(results)
        category_hotspots: List[Dict[str, Any]] = []
        cumulative = _ZERO
        for group in cat_agg.get("groups", []):
            co2e = group.get("co2e_kg", _ZERO)
            cumulative += co2e
            cum_pct = _round_decimal(cumulative / total_co2e * _HUNDRED, 2)
            category_hotspots.append({
                "category": group.get("category", "unknown"),
                "co2e_kg": co2e,
                "percentage": group.get("percentage_of_total", _ZERO),
                "cumulative_percentage": cum_pct,
                "is_hotspot": cumulative <= pareto_threshold,
            })

        # Material hotspots
        mat_agg = self.aggregate_by_material(results)
        material_hotspots: List[Dict[str, Any]] = []
        cumulative = _ZERO
        for group in mat_agg.get("groups", []):
            co2e = group.get("co2e_kg", _ZERO)
            cumulative += co2e
            cum_pct = _round_decimal(cumulative / total_co2e * _HUNDRED, 2)
            material_hotspots.append({
                "material": group.get("material_type", "unknown"),
                "co2e_kg": co2e,
                "percentage": group.get("percentage_of_total", _ZERO),
                "cumulative_percentage": cum_pct,
                "is_hotspot": cumulative <= pareto_threshold,
            })

        hotspot_products = [
            p for p in product_hotspots if p.get("is_hotspot")
        ]

        return {
            "total_co2e_kg": _round_decimal(total_co2e),
            "pareto_threshold_pct": Decimal("80"),
            "product_hotspots": product_hotspots,
            "category_hotspots": category_hotspots,
            "material_hotspots": material_hotspots,
            "hotspot_product_count": len(hotspot_products),
            "total_product_count": len(results),
            "hotspot_coverage_pct": (
                _round_decimal(
                    sum(p["co2e_kg"] for p in hotspot_products)
                    / total_co2e
                    * _HUNDRED,
                    1,
                )
                if hotspot_products
                else _ZERO
            ),
        }

    # ==========================================================================
    # DQI AND UNCERTAINTY
    # ==========================================================================

    def compute_dqi_score(
        self,
        method_counts: Dict[str, int],
        method_emissions: Dict[str, Decimal],
    ) -> Dict[str, Any]:
        """
        Compute blended data quality score across methods.

        Weights DQI by emissions contribution of each method.

        Args:
            method_counts: Count of products by method.
            method_emissions: Total emissions by method.

        Returns:
            Blended DQI score dictionary.
        """
        total_emissions = sum(method_emissions.values())
        if total_emissions <= _ZERO:
            return {
                "composite_score": Decimal("5.0"),
                "classification": "very_poor",
                "tier": "tier_3",
            }

        # DQI composite scores by method
        method_dqi: Dict[str, Decimal] = {
            "producer_specific": Decimal("1.0"),
            "waste_type_specific": Decimal("2.5"),
            "average_data": Decimal("3.4"),
            "spend_based": Decimal("4.5"),
        }

        weighted_sum = _ZERO
        for method, emissions in method_emissions.items():
            dqi = method_dqi.get(method, Decimal("4.5"))
            weight = emissions / total_emissions
            weighted_sum += dqi * weight

        composite = _round_decimal(weighted_sum, 2)

        if composite <= Decimal("1.5"):
            classification = "very_good"
        elif composite <= Decimal("2.5"):
            classification = "good"
        elif composite <= Decimal("3.5"):
            classification = "fair"
        elif composite <= Decimal("4.5"):
            classification = "poor"
        else:
            classification = "very_poor"

        # Determine dominant tier
        if composite <= Decimal("2.0"):
            tier = "tier_1"
        elif composite <= Decimal("3.5"):
            tier = "tier_2"
        else:
            tier = "tier_3"

        return {
            "composite_score": composite,
            "classification": classification,
            "tier": tier,
            "method_weights": {
                method: _round_decimal(
                    emissions / total_emissions * _HUNDRED, 1
                )
                for method, emissions in method_emissions.items()
            },
            "improvement_recommendations": self._generate_dqi_recommendations(
                method_counts, method_emissions, total_emissions
            ),
        }

    def _generate_dqi_recommendations(
        self,
        method_counts: Dict[str, int],
        method_emissions: Dict[str, Decimal],
        total_emissions: Decimal,
    ) -> List[str]:
        """Generate DQI improvement recommendations."""
        recommendations: List[str] = []

        spend_emissions = method_emissions.get("spend_based", _ZERO)
        avg_emissions = method_emissions.get("average_data", _ZERO)

        if total_emissions > _ZERO:
            spend_pct = spend_emissions / total_emissions * _HUNDRED
            avg_pct = avg_emissions / total_emissions * _HUNDRED

            if spend_pct > Decimal("30"):
                recommendations.append(
                    f"Spend-based method covers {_round_decimal(spend_pct, 0)}% "
                    f"of emissions. Obtain product-level data to improve quality."
                )
            if avg_pct > Decimal("50"):
                recommendations.append(
                    f"Average-data method covers {_round_decimal(avg_pct, 0)}% "
                    f"of emissions. Request EPDs from top suppliers."
                )
            if "producer_specific" not in method_counts:
                recommendations.append(
                    "No producer-specific data used. Request EPDs/PCFs from "
                    "suppliers of high-emission products."
                )

        if not recommendations:
            recommendations.append(
                "Good data quality mix. Continue collecting producer-specific "
                "data for remaining products."
            )

        return recommendations

    def compute_uncertainty(
        self,
        total_co2e_kg: Decimal,
        method_counts: Dict[str, int],
        method_emissions: Dict[str, Decimal],
    ) -> Dict[str, Any]:
        """
        Compute blended uncertainty range across methods.

        Uses emissions-weighted uncertainty percentages.

        Args:
            total_co2e_kg: Total calculated emissions.
            method_counts: Count of products by method.
            method_emissions: Total emissions by method.

        Returns:
            Uncertainty dictionary.
        """
        total_emissions = sum(method_emissions.values())
        if total_emissions <= _ZERO:
            return {
                "total_co2e_kg": total_co2e_kg,
                "lower_bound_co2e_kg": _ZERO,
                "upper_bound_co2e_kg": _ZERO,
                "blended_uncertainty_pct": Decimal("0.50"),
            }

        # Emissions-weighted uncertainty
        weighted_uncertainty = _ZERO
        for method, emissions in method_emissions.items():
            method_info = METHOD_HIERARCHY.get(method, {})
            unc_pct = method_info.get("uncertainty_pct", Decimal("0.50"))
            weight = emissions / total_emissions
            weighted_uncertainty += unc_pct * weight

        blended_pct = _round_decimal(weighted_uncertainty, 4)
        lower_bound = _round_decimal(total_co2e_kg * (_ONE - blended_pct))
        upper_bound = _round_decimal(total_co2e_kg * (_ONE + blended_pct))

        return {
            "method": "hybrid_blended",
            "total_co2e_kg": total_co2e_kg,
            "lower_bound_co2e_kg": lower_bound,
            "upper_bound_co2e_kg": upper_bound,
            "blended_uncertainty_pct": blended_pct,
            "blended_uncertainty_pct_display": _round_decimal(
                blended_pct * _HUNDRED, 1
            ),
            "confidence_level": Decimal("0.95"),
            "method_uncertainties": {
                method: METHOD_HIERARCHY.get(method, {}).get(
                    "uncertainty_pct", Decimal("0.50")
                )
                for method in method_emissions
            },
        }

    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    def _validate_inputs_internal(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        year: int,
    ) -> None:
        """Validate inputs for the hybrid calculation."""
        if not products:
            raise ValueError("products list must not be empty")
        if not isinstance(products, list):
            raise ValueError(
                f"products must be a list, got {type(products).__name__}"
            )
        if not org_id or not isinstance(org_id, str):
            raise ValueError(
                f"org_id must be a non-empty string, got {org_id!r}"
            )
        if not isinstance(year, int) or year < 2000 or year > 2100:
            raise ValueError(
                f"year must be integer between 2000 and 2100, got {year}"
            )

    def validate_inputs(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Validate inputs and return detailed validation result.

        Args:
            products: List of product dictionaries.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Validation result dictionary.
        """
        errors: List[str] = []
        warnings: List[str] = []
        method_assignments: Dict[str, int] = {}

        if not org_id or not isinstance(org_id, str):
            errors.append("org_id must be a non-empty string")
        if not isinstance(year, int) or year < 2000 or year > 2100:
            errors.append("year must be integer between 2000 and 2100")
        if not products:
            errors.append("products list must not be empty")
        elif not isinstance(products, list):
            errors.append("products must be a list")
        else:
            for idx, product in enumerate(products):
                product_id = product.get("product_id", f"product_{idx}")
                if not isinstance(product, dict):
                    errors.append(
                        f"Product at index {idx} must be a dictionary"
                    )
                    continue

                units = product.get("units_sold", 0)
                try:
                    if int(units) <= 0:
                        errors.append(
                            f"Product '{product_id}': units_sold must be > 0"
                        )
                        continue
                except (TypeError, ValueError):
                    errors.append(
                        f"Product '{product_id}': units_sold must be numeric"
                    )
                    continue

                method = self.select_best_method(product)
                method_assignments[method] = (
                    method_assignments.get(method, 0) + 1
                )

                if method == "spend_based":
                    warnings.append(
                        f"Product '{product_id}' will use spend-based fallback "
                        f"(Tier 3). Consider obtaining EPD or product data."
                    )
                elif method == "average_data":
                    cat = product.get("category")
                    if cat is None:
                        warnings.append(
                            f"Product '{product_id}' has no category; "
                            f"will use mixed_products fallback EF"
                        )

        return {
            "is_valid": len(errors) == 0,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
            "method_assignments": method_assignments,
        }

    # ==========================================================================
    # HEALTH CHECK
    # ==========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Perform engine health check.

        Returns:
            Health check result dictionary.
        """
        checks: List[Dict[str, Any]] = []
        overall_healthy = True

        # Check 1: Method hierarchy defined
        mh_count = len(METHOD_HIERARCHY)
        mh_ok = mh_count >= 4
        checks.append({
            "check": "method_hierarchy",
            "status": "pass" if mh_ok else "fail",
            "detail": f"{mh_count} methods in hierarchy (expected >= 4)",
        })
        if not mh_ok:
            overall_healthy = False

        # Check 2: Spend-based EF table loaded
        sb_count = len(SPEND_BASED_EOL_EF)
        sb_ok = sb_count >= 5
        checks.append({
            "check": "spend_based_efs",
            "status": "pass" if sb_ok else "fail",
            "detail": f"{sb_count} spend-based EFs loaded (expected >= 5)",
        })
        if not sb_ok:
            overall_healthy = False

        # Check 3: Recycling avoided EF table loaded
        ra_count = len(RECYCLING_AVOIDED_EF)
        ra_ok = ra_count >= 10
        checks.append({
            "check": "recycling_avoided_efs",
            "status": "pass" if ra_ok else "fail",
            "detail": f"{ra_count} recycling avoided EFs loaded (expected >= 10)",
        })
        if not ra_ok:
            overall_healthy = False

        # Check 4: Energy recovery EF table loaded
        er_count = len(ENERGY_RECOVERY_AVOIDED_EF)
        er_ok = er_count >= 7
        checks.append({
            "check": "energy_recovery_efs",
            "status": "pass" if er_ok else "fail",
            "detail": f"{er_count} energy recovery EFs loaded (expected >= 7)",
        })
        if not er_ok:
            overall_healthy = False

        # Check 5: Waste hierarchy weights loaded
        wh_count = len(WASTE_HIERARCHY_WEIGHTS)
        wh_ok = wh_count >= 10
        checks.append({
            "check": "waste_hierarchy_weights",
            "status": "pass" if wh_ok else "fail",
            "detail": f"{wh_count} hierarchy weights loaded (expected >= 10)",
        })
        if not wh_ok:
            overall_healthy = False

        # Check 6: Average data engine available
        avg_available = AverageDataCalculatorEngine is not None
        checks.append({
            "check": "average_data_engine",
            "status": "pass" if avg_available else "warn",
            "detail": (
                "AverageDataCalculatorEngine available"
                if avg_available
                else "AverageDataCalculatorEngine not importable (fallback active)"
            ),
        })

        # Check 7: Producer specific engine available
        ps_available = ProducerSpecificCalculatorEngine is not None
        checks.append({
            "check": "producer_specific_engine",
            "status": "pass" if ps_available else "warn",
            "detail": (
                "ProducerSpecificCalculatorEngine available"
                if ps_available
                else "ProducerSpecificCalculatorEngine not importable (fallback active)"
            ),
        })

        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "status": "healthy" if overall_healthy else "unhealthy",
            "checks": checks,
            "calculation_count": self._calculation_count,
            "child_engines": {
                "average_data": avg_available,
                "producer_specific": ps_available,
            },
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }


# ==============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ==============================================================================


def get_hybrid_aggregator(
    gwp_version: str = "ar5",
) -> HybridAggregatorEngine:
    """
    Get the singleton HybridAggregatorEngine instance.

    Args:
        gwp_version: IPCC GWP version ('ar4', 'ar5', 'ar6').

    Returns:
        HybridAggregatorEngine singleton instance.
    """
    return HybridAggregatorEngine.get_instance(gwp_version=gwp_version)


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Engine class
    "HybridAggregatorEngine",
    # Convenience function
    "get_hybrid_aggregator",
    # Data tables
    "METHOD_HIERARCHY",
    "SPEND_BASED_EOL_EF",
    "RECYCLING_AVOIDED_EF",
    "ENERGY_RECOVERY_AVOIDED_EF",
    "NET_CALORIFIC_VALUES",
    "WTE_EFFICIENCY",
    "WASTE_HIERARCHY_WEIGHTS",
    "MAX_HIERARCHY_SCORE",
    "GWP_VALUES",
    # Constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "ENGINE_ID",
    "ENGINE_VERSION",
    "TABLE_PREFIX",
    "PRECISION",
]
