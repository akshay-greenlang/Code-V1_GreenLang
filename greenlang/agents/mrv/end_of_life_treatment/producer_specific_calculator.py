# -*- coding: utf-8 -*-
"""
ProducerSpecificCalculatorEngine - AGENT-MRV-025 Engine 4

GHG Protocol Scope 3 Category 12 producer/supplier-specific EOL calculator
(Method D).

This engine calculates end-of-life treatment emissions using producer-declared
data sources: Environmental Product Declarations (EPDs), Product Carbon
Footprints (PCFs), Extended Producer Responsibility (EPR) scheme data, and
take-back program records.

Data Quality:
    Producer-specific data is the highest quality method (Tier 1) when
    third-party verified. Self-declared data is Tier 2. The GHG Protocol
    hierarchy recommends using this method whenever EPD/PCF data is available.

Key Features:
    - EPD (Environmental Product Declaration) EOL scenario parsing per ISO 14025
    - PCF (Product Carbon Footprint) end-of-life module extraction
    - EN 15804+A2 construction product EPD conformance validation
    - Producer-declared treatment scenarios with verification status tracking
    - Extended Producer Responsibility (EPR) scheme data integration
    - Take-back program emissions (collection + transport + treatment)
    - Recycled content tracking (pre-consumer vs post-consumer)
    - Verification status levels: third_party_verified, self_declared,
      estimated, default

Thread Safety:
    Thread-safe singleton with threading.RLock() and double-checked locking.

References:
    - GHG Protocol Technical Guidance for Calculating Scope 3 Emissions, Cat 12
    - ISO 14025:2006 Environmental labels and declarations
    - EN 15804+A2:2019 Sustainability of construction works
    - ISO 14067:2018 Carbon footprint of products
    - GHG Protocol Product Life Cycle Accounting and Reporting Standard (2011)
    - EU Directive 2019/904 Single-Use Plastics / EPR obligations

Example:
    >>> engine = ProducerSpecificCalculatorEngine.get_instance()
    >>> result = engine.calculate(
    ...     products=[{
    ...         "product_id": "P-001",
    ...         "epd_data": {"eol_co2e_per_unit": Decimal("2.5"),
    ...                      "verification_status": "third_party_verified"},
    ...         "units_sold": 1000,
    ...     }],
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
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# GRACEFUL IMPORTS
# ==============================================================================

try:
    from greenlang.agents.mrv.end_of_life_treatment.config import get_config
except ImportError:
    def get_config() -> Any:  # type: ignore[misc]
        """Fallback configuration stub."""
        return None

try:
    from greenlang.agents.mrv.end_of_life_treatment.metrics import get_metrics
except ImportError:
    def get_metrics() -> Any:  # type: ignore[misc]
        """Fallback metrics stub."""
        return None

try:
    from greenlang.agents.mrv.end_of_life_treatment.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment,misc]

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-012"
AGENT_COMPONENT: str = "AGENT-MRV-025"
ENGINE_ID: str = "producer_specific_calculator_engine"
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
_HUNDRED: Decimal = Decimal("100")
KG_PER_TONNE: Decimal = Decimal("1000")
TONNES_PER_KG: Decimal = Decimal("0.001")

# ==============================================================================
# DATA TABLE: VERIFICATION STATUS DEFINITIONS
# ==============================================================================

VERIFICATION_LEVELS: Dict[str, Dict[str, Any]] = {
    "third_party_verified": {
        "level": 1,
        "description": (
            "EPD/PCF independently verified by an accredited third-party "
            "verifier per ISO 14025 or ISO 14064-3"
        ),
        "dqi_reliability": 1,
        "uncertainty_pct": Decimal("0.10"),
        "tier": "tier_1",
        "accepted_by": [
            "ghg_protocol", "iso_14064", "csrd_esrs", "cdp", "sbti",
        ],
    },
    "self_declared": {
        "level": 2,
        "description": (
            "Producer self-declared EOL data without independent verification. "
            "Complies with ISO 14021 self-declared environmental claims."
        ),
        "dqi_reliability": 2,
        "uncertainty_pct": Decimal("0.20"),
        "tier": "tier_2",
        "accepted_by": [
            "ghg_protocol", "cdp",
        ],
    },
    "estimated": {
        "level": 3,
        "description": (
            "Producer-estimated data based on product design and material "
            "composition. Not independently verified."
        ),
        "dqi_reliability": 3,
        "uncertainty_pct": Decimal("0.35"),
        "tier": "tier_2",
        "accepted_by": [
            "ghg_protocol",
        ],
    },
    "default": {
        "level": 4,
        "description": (
            "Default EOL scenario from product category or industry average. "
            "No producer-specific data available."
        ),
        "dqi_reliability": 4,
        "uncertainty_pct": Decimal("0.50"),
        "tier": "tier_3",
        "accepted_by": [
            "ghg_protocol",
        ],
    },
}

# ==============================================================================
# DATA TABLE: EPD LIFECYCLE MODULES (EN 15804+A2)
# ==============================================================================
# EN 15804 defines product lifecycle modules A1-A5, B1-B7, C1-C4, D.
# This engine focuses on modules C1-C4 (end-of-life) and D (benefits).

EPD_LIFECYCLE_MODULES: Dict[str, Dict[str, str]] = {
    "A1": {"name": "Raw material supply", "stage": "product"},
    "A2": {"name": "Transport to manufacturer", "stage": "product"},
    "A3": {"name": "Manufacturing", "stage": "product"},
    "A4": {"name": "Transport to site", "stage": "construction"},
    "A5": {"name": "Installation", "stage": "construction"},
    "B1": {"name": "Use", "stage": "use"},
    "B2": {"name": "Maintenance", "stage": "use"},
    "B3": {"name": "Repair", "stage": "use"},
    "B4": {"name": "Replacement", "stage": "use"},
    "B5": {"name": "Refurbishment", "stage": "use"},
    "B6": {"name": "Operational energy use", "stage": "use"},
    "B7": {"name": "Operational water use", "stage": "use"},
    "C1": {"name": "Deconstruction/demolition", "stage": "end_of_life"},
    "C2": {"name": "Transport to waste processing", "stage": "end_of_life"},
    "C3": {"name": "Waste processing", "stage": "end_of_life"},
    "C4": {"name": "Disposal", "stage": "end_of_life"},
    "D": {"name": "Benefits and loads beyond system boundary", "stage": "beyond"},
}

# EOL modules relevant to Category 12 calculations
EOL_MODULES: List[str] = ["C1", "C2", "C3", "C4"]
BEYOND_SYSTEM_MODULE: str = "D"

# ==============================================================================
# DATA TABLE: EPR SCHEME EMISSION FACTORS
# ==============================================================================
# Extended Producer Responsibility scheme emission factors by region.
# Covers collection logistics, sorting, and administrative overhead.
# Source: OECD EPR data, EU packaging directive reporting 2024.

EPR_SCHEME_FACTORS: Dict[str, Dict[str, Any]] = {
    "EU_packaging": {
        "description": "EU Packaging and Packaging Waste Directive 2018/852",
        "collection_ef_kgco2e_per_kg": Decimal("0.025"),
        "sorting_ef_kgco2e_per_kg": Decimal("0.018"),
        "admin_ef_kgco2e_per_kg": Decimal("0.003"),
        "total_ef_kgco2e_per_kg": Decimal("0.046"),
        "regions": ["EU", "DE", "FR", "NL", "BE", "AT", "IT", "ES"],
        "coverage": "packaging",
    },
    "EU_WEEE": {
        "description": "EU WEEE Directive 2012/19/EU (electronics)",
        "collection_ef_kgco2e_per_kg": Decimal("0.045"),
        "sorting_ef_kgco2e_per_kg": Decimal("0.032"),
        "admin_ef_kgco2e_per_kg": Decimal("0.005"),
        "total_ef_kgco2e_per_kg": Decimal("0.082"),
        "regions": ["EU", "DE", "FR", "NL", "BE", "AT", "IT", "ES"],
        "coverage": "electronics",
    },
    "EU_batteries": {
        "description": "EU Battery Regulation 2023/1542",
        "collection_ef_kgco2e_per_kg": Decimal("0.065"),
        "sorting_ef_kgco2e_per_kg": Decimal("0.040"),
        "admin_ef_kgco2e_per_kg": Decimal("0.008"),
        "total_ef_kgco2e_per_kg": Decimal("0.113"),
        "regions": ["EU", "DE", "FR", "NL", "BE", "AT", "IT", "ES"],
        "coverage": "batteries",
    },
    "EU_ELV": {
        "description": "EU End-of-Life Vehicles Directive 2000/53/EC",
        "collection_ef_kgco2e_per_kg": Decimal("0.035"),
        "sorting_ef_kgco2e_per_kg": Decimal("0.028"),
        "admin_ef_kgco2e_per_kg": Decimal("0.004"),
        "total_ef_kgco2e_per_kg": Decimal("0.067"),
        "regions": ["EU", "DE", "FR", "NL", "BE", "AT", "IT", "ES"],
        "coverage": "automotive",
    },
    "EU_textiles": {
        "description": "EU Textile Strategy (proposed EPR 2025+)",
        "collection_ef_kgco2e_per_kg": Decimal("0.030"),
        "sorting_ef_kgco2e_per_kg": Decimal("0.022"),
        "admin_ef_kgco2e_per_kg": Decimal("0.004"),
        "total_ef_kgco2e_per_kg": Decimal("0.056"),
        "regions": ["EU", "FR"],
        "coverage": "textiles",
    },
    "US_e_waste": {
        "description": "US state-level e-waste EPR programs (avg)",
        "collection_ef_kgco2e_per_kg": Decimal("0.055"),
        "sorting_ef_kgco2e_per_kg": Decimal("0.035"),
        "admin_ef_kgco2e_per_kg": Decimal("0.006"),
        "total_ef_kgco2e_per_kg": Decimal("0.096"),
        "regions": ["US"],
        "coverage": "electronics",
    },
    "JP_containers_packaging": {
        "description": "Japan Containers and Packaging Recycling Act",
        "collection_ef_kgco2e_per_kg": Decimal("0.020"),
        "sorting_ef_kgco2e_per_kg": Decimal("0.015"),
        "admin_ef_kgco2e_per_kg": Decimal("0.003"),
        "total_ef_kgco2e_per_kg": Decimal("0.038"),
        "regions": ["JP"],
        "coverage": "packaging",
    },
    "JP_home_appliances": {
        "description": "Japan Home Appliance Recycling Act",
        "collection_ef_kgco2e_per_kg": Decimal("0.040"),
        "sorting_ef_kgco2e_per_kg": Decimal("0.030"),
        "admin_ef_kgco2e_per_kg": Decimal("0.005"),
        "total_ef_kgco2e_per_kg": Decimal("0.075"),
        "regions": ["JP"],
        "coverage": "appliances",
    },
    "KR_EPR": {
        "description": "South Korea EPR system (packaging, electronics, batteries)",
        "collection_ef_kgco2e_per_kg": Decimal("0.022"),
        "sorting_ef_kgco2e_per_kg": Decimal("0.016"),
        "admin_ef_kgco2e_per_kg": Decimal("0.003"),
        "total_ef_kgco2e_per_kg": Decimal("0.041"),
        "regions": ["KR"],
        "coverage": "packaging,electronics,batteries",
    },
    "CA_provincial": {
        "description": "Canadian provincial EPR programs (BC, ON, QC average)",
        "collection_ef_kgco2e_per_kg": Decimal("0.028"),
        "sorting_ef_kgco2e_per_kg": Decimal("0.020"),
        "admin_ef_kgco2e_per_kg": Decimal("0.004"),
        "total_ef_kgco2e_per_kg": Decimal("0.052"),
        "regions": ["CA"],
        "coverage": "packaging,electronics",
    },
    "GLOBAL_default": {
        "description": "Global default EPR overhead (conservative estimate)",
        "collection_ef_kgco2e_per_kg": Decimal("0.035"),
        "sorting_ef_kgco2e_per_kg": Decimal("0.025"),
        "admin_ef_kgco2e_per_kg": Decimal("0.005"),
        "total_ef_kgco2e_per_kg": Decimal("0.065"),
        "regions": ["GLOBAL"],
        "coverage": "all",
    },
}

# ==============================================================================
# DATA TABLE: TAKE-BACK PROGRAM EMISSION FACTORS
# ==============================================================================
# Emission factors for producer take-back program logistics.

TAKE_BACK_EF: Dict[str, Dict[str, Decimal]] = {
    "collection_point_road": {
        "ef_kgco2e_per_tonne_km": Decimal("0.10694"),
        "description": "Road freight to collection point (rigid HGV)",
    },
    "collection_point_van": {
        "ef_kgco2e_per_tonne_km": Decimal("0.60459"),
        "description": "Van collection (light commercial vehicle)",
    },
    "reverse_logistics_road": {
        "ef_kgco2e_per_tonne_km": Decimal("0.08976"),
        "description": "Reverse logistics road freight (articulated HGV)",
    },
    "reverse_logistics_rail": {
        "ef_kgco2e_per_tonne_km": Decimal("0.02726"),
        "description": "Reverse logistics rail freight",
    },
    "consolidation_hub": {
        "ef_kgco2e_per_kg": Decimal("0.012"),
        "description": "Consolidation hub handling emissions",
    },
    "refurbishment_facility": {
        "ef_kgco2e_per_kg": Decimal("0.085"),
        "description": "Refurbishment/remanufacturing facility processing",
    },
    "disassembly": {
        "ef_kgco2e_per_kg": Decimal("0.035"),
        "description": "Manual and automated disassembly operations",
    },
}

# ==============================================================================
# DATA TABLE: RECYCLED CONTENT CREDIT FACTORS
# ==============================================================================
# Virgin displacement factors for recycled content (kgCO2e avoided per kg).

RECYCLED_CONTENT_CREDIT_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "plastics_pet": {
        "virgin_ef": Decimal("2.730"),
        "recycled_ef": Decimal("1.020"),
        "credit_per_kg": Decimal("1.710"),
    },
    "plastics_hdpe": {
        "virgin_ef": Decimal("1.520"),
        "recycled_ef": Decimal("0.660"),
        "credit_per_kg": Decimal("0.860"),
    },
    "plastics_mixed": {
        "virgin_ef": Decimal("1.860"),
        "recycled_ef": Decimal("0.780"),
        "credit_per_kg": Decimal("1.080"),
    },
    "aluminum": {
        "virgin_ef": Decimal("11.890"),
        "recycled_ef": Decimal("0.830"),
        "credit_per_kg": Decimal("11.060"),
    },
    "steel": {
        "virgin_ef": Decimal("2.890"),
        "recycled_ef": Decimal("0.870"),
        "credit_per_kg": Decimal("2.020"),
    },
    "glass": {
        "virgin_ef": Decimal("0.843"),
        "recycled_ef": Decimal("0.535"),
        "credit_per_kg": Decimal("0.308"),
    },
    "paper_cardboard": {
        "virgin_ef": Decimal("3.428"),
        "recycled_ef": Decimal("1.710"),
        "credit_per_kg": Decimal("1.718"),
    },
    "textiles_cotton": {
        "virgin_ef": Decimal("5.300"),
        "recycled_ef": Decimal("2.780"),
        "credit_per_kg": Decimal("2.520"),
    },
    "wood": {
        "virgin_ef": Decimal("0.423"),
        "recycled_ef": Decimal("0.150"),
        "credit_per_kg": Decimal("0.273"),
    },
    "rubber": {
        "virgin_ef": Decimal("3.100"),
        "recycled_ef": Decimal("1.500"),
        "credit_per_kg": Decimal("1.600"),
    },
}

# ==============================================================================
# DATA TABLE: ISO 14025 EPD VALIDATION REQUIREMENTS
# ==============================================================================

ISO_14025_REQUIRED_FIELDS: List[str] = [
    "epd_registration_number",
    "product_name",
    "program_operator",
    "pcr_reference",
    "declared_unit",
    "validity_period_years",
    "publication_date",
    "verification_status",
]

ISO_14025_EOL_REQUIRED_FIELDS: List[str] = [
    "c1_deconstruction_kgco2e",
    "c2_transport_kgco2e",
    "c3_waste_processing_kgco2e",
    "c4_disposal_kgco2e",
]

EN_15804_REQUIRED_FIELDS: List[str] = [
    "epd_registration_number",
    "product_name",
    "program_operator",
    "pcr_reference",
    "declared_unit",
    "reference_service_life",
    "scenario_description",
]

# ==============================================================================
# DATA TABLE: PCF VALIDATION REQUIREMENTS
# ==============================================================================

PCF_REQUIRED_FIELDS: List[str] = [
    "pcf_id",
    "product_name",
    "functional_unit",
    "system_boundary",
    "eol_emissions_kgco2e",
    "data_quality_rating",
    "reference_year",
]

GHG_PROTOCOL_PRODUCT_STANDARD_FIELDS: List[str] = [
    "product_name",
    "functional_unit",
    "system_boundary",
    "eol_emissions_kgco2e",
    "allocation_method",
    "data_quality_rating",
]

# ==============================================================================
# GWP VALUES
# ==============================================================================

GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "ar4": {"co2": Decimal("1"), "ch4": Decimal("25"), "n2o": Decimal("298")},
    "ar5": {"co2": Decimal("1"), "ch4": Decimal("28"), "n2o": Decimal("265")},
    "ar6": {"co2": Decimal("1"), "ch4": Decimal("27.9"), "n2o": Decimal("273")},
}

# ==============================================================================
# DQI SCORES FOR PRODUCER-SPECIFIC METHOD
# ==============================================================================

DQI_PRODUCER_SPECIFIC: Dict[str, Dict[str, int]] = {
    "third_party_verified": {
        "temporal": 1,
        "geographical": 1,
        "technological": 1,
        "completeness": 1,
        "reliability": 1,
    },
    "self_declared": {
        "temporal": 2,
        "geographical": 1,
        "technological": 1,
        "completeness": 2,
        "reliability": 2,
    },
    "estimated": {
        "temporal": 2,
        "geographical": 2,
        "technological": 2,
        "completeness": 3,
        "reliability": 3,
    },
    "default": {
        "temporal": 3,
        "geographical": 3,
        "technological": 3,
        "completeness": 4,
        "reliability": 4,
    },
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


def _generate_id(prefix: str = "eol_ps") -> str:
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
        value: Value to convert (int, float, str, or Decimal).

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


class ProducerSpecificCalculatorEngine:
    """
    Engine 4: Producer/supplier-specific EOL emissions calculator (Method D).

    Implements the producer-specific calculation method for GHG Protocol
    Scope 3 Category 12 using EPDs, PCFs, EPR scheme data, and take-back
    program records. This is the highest quality method (Tier 1) when data
    is third-party verified.

    Data Sources:
        - EPD (Environmental Product Declaration) per ISO 14025
        - PCF (Product Carbon Footprint) per ISO 14067 / GHG Protocol
        - EPR (Extended Producer Responsibility) scheme reporting
        - Take-back program logistics and treatment data
        - Producer-declared treatment scenarios

    Thread Safety:
        Singleton pattern with threading.RLock() and double-checked locking.

    Zero-Hallucination:
        All calculations use deterministic Decimal arithmetic. No LLM
        calls for any numeric computation.

    Attributes:
        _config: Configuration from get_config().
        _metrics: Prometheus metrics from get_metrics().
        _calculation_count: Running count of calculations performed.

    Example:
        >>> engine = ProducerSpecificCalculatorEngine.get_instance()
        >>> result = engine.calculate(
        ...     products=[{
        ...         "product_id": "SKU-001",
        ...         "epd_data": {
        ...             "c1_deconstruction_kgco2e": Decimal("0.1"),
        ...             "c2_transport_kgco2e": Decimal("0.3"),
        ...             "c3_waste_processing_kgco2e": Decimal("1.2"),
        ...             "c4_disposal_kgco2e": Decimal("0.9"),
        ...             "verification_status": "third_party_verified",
        ...         },
        ...         "units_sold": 500,
        ...     }],
        ...     org_id="ORG-001", year=2025)
    """

    _instance: Optional["ProducerSpecificCalculatorEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(
        cls, *args: Any, **kwargs: Any
    ) -> "ProducerSpecificCalculatorEngine":
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
        Initialize the ProducerSpecificCalculatorEngine.

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

        self._initialized: bool = True

        logger.info(
            "ProducerSpecificCalculatorEngine initialized: engine=%s, "
            "version=%s, gwp=%s, epr_schemes=%d, verification_levels=%d",
            ENGINE_ID,
            ENGINE_VERSION,
            gwp_version,
            len(EPR_SCHEME_FACTORS),
            len(VERIFICATION_LEVELS),
        )

    # ==========================================================================
    # SINGLETON MANAGEMENT
    # ==========================================================================

    @classmethod
    def get_instance(
        cls, gwp_version: str = "ar5"
    ) -> "ProducerSpecificCalculatorEngine":
        """
        Get singleton instance with thread-safe double-checked locking.

        Args:
            gwp_version: IPCC GWP version (only used on first instantiation).

        Returns:
            ProducerSpecificCalculatorEngine singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls(gwp_version=gwp_version)
        return cls._instance  # type: ignore[return-value]

    @classmethod
    def reset(cls) -> None:
        """
        Reset singleton instance. Used in testing only.

        Thread Safety:
            Protected by the class-level RLock.
        """
        with cls._lock:
            cls._instance = None
            logger.info("ProducerSpecificCalculatorEngine singleton reset")

    # ==========================================================================
    # INTERNAL HELPERS
    # ==========================================================================

    def _increment_calculation_count(self) -> int:
        """Increment and return the calculation counter thread-safely."""
        with self._count_lock:
            self._calculation_count += 1
            return self._calculation_count

    def _record_metrics(
        self,
        data_source: str,
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
                method="producer_specific",
                category=data_source,
                co2e_kg=float(co2e_kg),
                duration=duration,
                status=status,
            )
        except Exception as exc:
            logger.debug("Metrics recording failed (non-critical): %s", exc)

    # ==========================================================================
    # EPD VALIDATION
    # ==========================================================================

    def validate_epd(self, epd_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an Environmental Product Declaration for ISO 14025 conformance.

        Checks for required fields, valid date ranges, valid verification
        status, and presence of EOL modules (C1-C4).

        Args:
            epd_data: EPD data dictionary.

        Returns:
            Validation result with is_valid, errors, warnings, completeness.

        Example:
            >>> engine = ProducerSpecificCalculatorEngine.get_instance()
            >>> result = engine.validate_epd({
            ...     "epd_registration_number": "EPD-2024-001",
            ...     "product_name": "Widget A",
            ...     "program_operator": "IBU",
            ...     "pcr_reference": "PCR-2024-001",
            ...     "declared_unit": "1 piece",
            ...     "validity_period_years": 5,
            ...     "publication_date": "2024-01-01",
            ...     "verification_status": "third_party_verified",
            ...     "c1_deconstruction_kgco2e": Decimal("0.1"),
            ...     "c2_transport_kgco2e": Decimal("0.3"),
            ...     "c3_waste_processing_kgco2e": Decimal("1.2"),
            ...     "c4_disposal_kgco2e": Decimal("0.9"),
            ... })
            >>> result["is_valid"]
            True
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(epd_data, dict):
            return {
                "is_valid": False,
                "errors": ["epd_data must be a dictionary"],
                "warnings": [],
                "completeness_pct": Decimal("0"),
            }

        # Check ISO 14025 required fields
        present_count = 0
        for field in ISO_14025_REQUIRED_FIELDS:
            if field in epd_data and epd_data[field] is not None:
                present_count += 1
            else:
                errors.append(f"Missing required ISO 14025 field: '{field}'")

        # Check EOL module fields (C1-C4)
        eol_present = 0
        for field in ISO_14025_EOL_REQUIRED_FIELDS:
            if field in epd_data and epd_data[field] is not None:
                eol_present += 1
                try:
                    val = _safe_decimal(epd_data[field])
                    if val < _ZERO:
                        warnings.append(
                            f"EOL module '{field}' has negative value "
                            f"({val}); verify this is intentional"
                        )
                except ValueError:
                    errors.append(
                        f"EOL module '{field}' must be numeric, "
                        f"got {epd_data[field]!r}"
                    )
            else:
                warnings.append(
                    f"Missing EOL module field: '{field}' "
                    f"(will use zero if calculating)"
                )

        # Check verification status
        verification = epd_data.get("verification_status", "default")
        if verification not in VERIFICATION_LEVELS:
            errors.append(
                f"Invalid verification_status: '{verification}'. "
                f"Valid: {list(VERIFICATION_LEVELS.keys())}"
            )

        # Check validity period
        pub_date = epd_data.get("publication_date")
        validity_years = epd_data.get("validity_period_years", 5)
        if pub_date is not None:
            try:
                if isinstance(pub_date, str):
                    pub_dt = datetime.fromisoformat(pub_date).date()
                elif isinstance(pub_date, (datetime, date)):
                    pub_dt = pub_date if isinstance(pub_date, date) else pub_date.date()
                else:
                    pub_dt = None

                if pub_dt is not None:
                    expiry = date(
                        pub_dt.year + int(validity_years),
                        pub_dt.month,
                        pub_dt.day,
                    )
                    today = date.today()
                    if today > expiry:
                        warnings.append(
                            f"EPD expired on {expiry.isoformat()}. "
                            f"Consider requesting an updated EPD."
                        )
            except (ValueError, TypeError, OverflowError) as exc:
                warnings.append(f"Could not parse publication_date: {exc}")

        # Check Module D (benefits beyond system boundary)
        module_d = epd_data.get("d_benefits_kgco2e")
        if module_d is not None:
            try:
                d_val = _safe_decimal(module_d)
                if d_val > _ZERO:
                    warnings.append(
                        "Module D (benefits) is positive; typically Module D "
                        "should be negative (representing avoided emissions)"
                    )
            except ValueError:
                warnings.append(
                    f"Module D value is not numeric: {module_d!r}"
                )

        # EN 15804 check
        en_15804_present = 0
        for field in EN_15804_REQUIRED_FIELDS:
            if field in epd_data and epd_data[field] is not None:
                en_15804_present += 1
        en_15804_compliant = en_15804_present == len(EN_15804_REQUIRED_FIELDS)
        if not en_15804_compliant:
            warnings.append(
                f"EPD does not fully conform to EN 15804+A2 "
                f"({en_15804_present}/{len(EN_15804_REQUIRED_FIELDS)} fields)"
            )

        total_fields = len(ISO_14025_REQUIRED_FIELDS) + len(
            ISO_14025_EOL_REQUIRED_FIELDS
        )
        completeness = _round_decimal(
            Decimal(str(present_count + eol_present))
            / Decimal(str(total_fields))
            * _HUNDRED,
            1,
        )

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "iso_14025_compliant": present_count == len(ISO_14025_REQUIRED_FIELDS),
            "en_15804_compliant": en_15804_compliant,
            "eol_modules_present": eol_present,
            "eol_modules_total": len(ISO_14025_EOL_REQUIRED_FIELDS),
            "completeness_pct": completeness,
            "verification_status": epd_data.get("verification_status", "default"),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }

    # ==========================================================================
    # PCF VALIDATION
    # ==========================================================================

    def validate_pcf(self, pcf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a Product Carbon Footprint for GHG Protocol Product Standard
        conformance.

        Args:
            pcf_data: PCF data dictionary.

        Returns:
            Validation result with is_valid, errors, warnings, completeness.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(pcf_data, dict):
            return {
                "is_valid": False,
                "errors": ["pcf_data must be a dictionary"],
                "warnings": [],
                "completeness_pct": Decimal("0"),
            }

        # Check required fields
        present_count = 0
        for field in PCF_REQUIRED_FIELDS:
            if field in pcf_data and pcf_data[field] is not None:
                present_count += 1
            else:
                errors.append(f"Missing required PCF field: '{field}'")

        # Check EOL emissions value
        eol_val = pcf_data.get("eol_emissions_kgco2e")
        if eol_val is not None:
            try:
                val = _safe_decimal(eol_val)
                if val < _ZERO:
                    warnings.append(
                        f"eol_emissions_kgco2e is negative ({val}); "
                        f"verify this includes only gross emissions"
                    )
            except ValueError:
                errors.append(
                    f"eol_emissions_kgco2e must be numeric, got {eol_val!r}"
                )

        # Check system boundary
        boundary = pcf_data.get("system_boundary", "")
        if boundary and "cradle" not in str(boundary).lower():
            warnings.append(
                "system_boundary does not appear to be cradle-to-grave; "
                "EOL module may not be included"
            )

        # Check GHG Protocol Product Standard conformance
        ghg_present = 0
        for field in GHG_PROTOCOL_PRODUCT_STANDARD_FIELDS:
            if field in pcf_data and pcf_data[field] is not None:
                ghg_present += 1

        ghg_compliant = ghg_present == len(GHG_PROTOCOL_PRODUCT_STANDARD_FIELDS)

        completeness = _round_decimal(
            Decimal(str(present_count))
            / Decimal(str(len(PCF_REQUIRED_FIELDS)))
            * _HUNDRED,
            1,
        )

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "ghg_protocol_product_standard_compliant": ghg_compliant,
            "completeness_pct": completeness,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }

    # ==========================================================================
    # EOL SCENARIO PARSING
    # ==========================================================================

    def parse_eol_scenario(self, epd_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and aggregate end-of-life scenario from EPD data.

        Parses EN 15804 modules C1-C4 and optional Module D (benefits).
        Module D is reported separately per GHG Protocol (never netted).

        Args:
            epd_data: EPD data dictionary with C1-C4 and optional D modules.

        Returns:
            Parsed EOL scenario with module breakdown and totals.
        """
        modules: Dict[str, Decimal] = {}
        total_eol = _ZERO

        # Parse C1-C4 modules
        for module_key in EOL_MODULES:
            field_map = {
                "C1": "c1_deconstruction_kgco2e",
                "C2": "c2_transport_kgco2e",
                "C3": "c3_waste_processing_kgco2e",
                "C4": "c4_disposal_kgco2e",
            }
            field = field_map.get(module_key, "")
            raw_val = epd_data.get(field, _ZERO)
            try:
                val = _safe_decimal(raw_val)
            except ValueError:
                val = _ZERO
                logger.warning(
                    "Cannot parse EPD module %s value %r; defaulting to 0",
                    module_key, raw_val,
                )

            modules[module_key] = _round_decimal(val)
            total_eol += val

        total_eol = _round_decimal(total_eol)

        # Parse Module D (benefits beyond system boundary)
        module_d_raw = epd_data.get("d_benefits_kgco2e", _ZERO)
        try:
            module_d = _round_decimal(_safe_decimal(module_d_raw))
        except ValueError:
            module_d = _ZERO

        # Scenario description
        scenario_desc = epd_data.get(
            "scenario_description",
            "Producer-declared end-of-life scenario from EPD"
        )

        return {
            "modules": modules,
            "module_d_benefits": module_d,
            "total_eol_kgco2e": total_eol,
            "total_with_d_kgco2e": _round_decimal(total_eol + module_d),
            "scenario_description": scenario_desc,
            "declared_unit": epd_data.get("declared_unit", "1 unit"),
            "module_breakdown": {
                "C1_deconstruction": modules.get("C1", _ZERO),
                "C2_transport": modules.get("C2", _ZERO),
                "C3_waste_processing": modules.get("C3", _ZERO),
                "C4_disposal": modules.get("C4", _ZERO),
                "D_benefits_memo": module_d,
            },
        }

    # ==========================================================================
    # TAKE-BACK PROGRAM CALCULATION
    # ==========================================================================

    def calculate_take_back(
        self,
        product: Dict[str, Any],
        take_back_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate emissions from a producer take-back program.

        Includes collection logistics, reverse transport, consolidation
        hub handling, and treatment facility processing.

        Args:
            product: Product dictionary with product_id, weight_kg, units.
            take_back_data: Take-back program parameters:
                - collection_distance_km (Decimal): Distance to collection point.
                - transport_mode (str): 'road', 'van', 'rail'.
                - reverse_distance_km (Decimal): Reverse logistics distance.
                - reverse_mode (str): 'road', 'rail'.
                - include_consolidation (bool): Include hub handling.
                - include_refurbishment (bool): Include refurbishment.
                - include_disassembly (bool): Include disassembly.
                - treatment_ef_kgco2e_per_kg (Decimal, optional): Custom treatment EF.

        Returns:
            Take-back emissions breakdown dictionary.
        """
        product_id = product.get("product_id", "unknown")
        weight_kg = _safe_decimal(product.get("weight_kg", 0))
        units = int(product.get("units", product.get("units_sold", 1)))
        total_weight_kg = weight_kg * Decimal(str(units))
        total_weight_tonnes = total_weight_kg * TONNES_PER_KG

        # Collection emissions
        collection_dist = _safe_decimal(
            take_back_data.get("collection_distance_km", _ZERO)
        )
        collection_mode = take_back_data.get("transport_mode", "road")
        collection_ef_key = f"collection_point_{collection_mode}"
        collection_ef = TAKE_BACK_EF.get(
            collection_ef_key,
            TAKE_BACK_EF["collection_point_road"],
        )
        collection_emissions = _round_decimal(
            total_weight_tonnes * collection_dist
            * collection_ef["ef_kgco2e_per_tonne_km"]
        )

        # Reverse logistics emissions
        reverse_dist = _safe_decimal(
            take_back_data.get("reverse_distance_km", _ZERO)
        )
        reverse_mode = take_back_data.get("reverse_mode", "road")
        reverse_ef_key = f"reverse_logistics_{reverse_mode}"
        reverse_ef = TAKE_BACK_EF.get(
            reverse_ef_key,
            TAKE_BACK_EF["reverse_logistics_road"],
        )
        reverse_emissions = _round_decimal(
            total_weight_tonnes * reverse_dist
            * reverse_ef["ef_kgco2e_per_tonne_km"]
        )

        # Hub handling
        hub_emissions = _ZERO
        if take_back_data.get("include_consolidation", False):
            hub_ef = TAKE_BACK_EF["consolidation_hub"]["ef_kgco2e_per_kg"]
            hub_emissions = _round_decimal(total_weight_kg * hub_ef)

        # Refurbishment
        refurb_emissions = _ZERO
        if take_back_data.get("include_refurbishment", False):
            refurb_ef = TAKE_BACK_EF["refurbishment_facility"]["ef_kgco2e_per_kg"]
            refurb_emissions = _round_decimal(total_weight_kg * refurb_ef)

        # Disassembly
        disassembly_emissions = _ZERO
        if take_back_data.get("include_disassembly", False):
            disassembly_ef = TAKE_BACK_EF["disassembly"]["ef_kgco2e_per_kg"]
            disassembly_emissions = _round_decimal(total_weight_kg * disassembly_ef)

        # Treatment
        treatment_ef = take_back_data.get("treatment_ef_kgco2e_per_kg")
        treatment_emissions = _ZERO
        if treatment_ef is not None:
            treatment_emissions = _round_decimal(
                total_weight_kg * _safe_decimal(treatment_ef)
            )

        total_emissions = _round_decimal(
            collection_emissions + reverse_emissions + hub_emissions
            + refurb_emissions + disassembly_emissions + treatment_emissions
        )

        provenance_data = (
            f"take_back|{product_id}|{total_weight_kg}|"
            f"{collection_dist}|{reverse_dist}|{total_emissions}"
        )

        return {
            "product_id": product_id,
            "total_weight_kg": total_weight_kg,
            "units": units,
            "collection_emissions_kgco2e": collection_emissions,
            "reverse_logistics_emissions_kgco2e": reverse_emissions,
            "hub_handling_emissions_kgco2e": hub_emissions,
            "refurbishment_emissions_kgco2e": refurb_emissions,
            "disassembly_emissions_kgco2e": disassembly_emissions,
            "treatment_emissions_kgco2e": treatment_emissions,
            "total_take_back_emissions_kgco2e": total_emissions,
            "provenance_hash": _compute_hash(provenance_data),
        }

    # ==========================================================================
    # EPR OBLIGATIONS CALCULATION
    # ==========================================================================

    def calculate_epr_obligations(
        self,
        products: List[Dict[str, Any]],
        region: str,
    ) -> Dict[str, Any]:
        """
        Calculate EPR scheme emissions for products in a given region.

        Maps product categories to applicable EPR schemes and calculates
        the collection, sorting, and administrative overhead emissions.

        Args:
            products: List of product dicts with weight_kg and units_sold.
            region: Region code (e.g., 'EU', 'US', 'JP').

        Returns:
            EPR emissions breakdown by scheme.
        """
        # Find applicable schemes for region
        applicable_schemes: List[str] = []
        for scheme_id, scheme in EPR_SCHEME_FACTORS.items():
            if region in scheme["regions"] or "GLOBAL" in scheme["regions"]:
                applicable_schemes.append(scheme_id)

        if not applicable_schemes:
            applicable_schemes = ["GLOBAL_default"]

        total_epr_emissions = _ZERO
        scheme_results: List[Dict[str, Any]] = []

        for product in products:
            product_id = product.get("product_id", "unknown")
            weight_kg = _safe_decimal(product.get("weight_kg", 0))
            units = int(product.get("units_sold", product.get("units", 1)))
            total_weight_kg = weight_kg * Decimal(str(units))
            category = product.get("category", "mixed")

            # Find best matching scheme
            best_scheme = self._match_epr_scheme(
                category, applicable_schemes
            )
            scheme = EPR_SCHEME_FACTORS.get(
                best_scheme, EPR_SCHEME_FACTORS["GLOBAL_default"]
            )

            collection = _round_decimal(
                total_weight_kg * scheme["collection_ef_kgco2e_per_kg"]
            )
            sorting = _round_decimal(
                total_weight_kg * scheme["sorting_ef_kgco2e_per_kg"]
            )
            admin = _round_decimal(
                total_weight_kg * scheme["admin_ef_kgco2e_per_kg"]
            )
            total = _round_decimal(collection + sorting + admin)

            total_epr_emissions += total

            scheme_results.append({
                "product_id": product_id,
                "scheme_id": best_scheme,
                "scheme_description": scheme["description"],
                "total_weight_kg": total_weight_kg,
                "collection_kgco2e": collection,
                "sorting_kgco2e": sorting,
                "admin_kgco2e": admin,
                "total_kgco2e": total,
            })

        return {
            "region": region,
            "applicable_schemes": applicable_schemes,
            "product_count": len(products),
            "total_epr_emissions_kgco2e": _round_decimal(total_epr_emissions),
            "total_epr_emissions_tonnes": _round_decimal(
                total_epr_emissions * TONNES_PER_KG
            ),
            "scheme_results": scheme_results,
        }

    def _match_epr_scheme(
        self,
        category: str,
        applicable_schemes: List[str],
    ) -> str:
        """
        Match a product category to the best EPR scheme.

        Args:
            category: Product category string.
            applicable_schemes: List of applicable scheme IDs.

        Returns:
            Best matching scheme ID.
        """
        category_lower = category.lower()

        # Category to coverage keyword mapping
        category_coverage_map = {
            "electronics": "electronics",
            "appliances": "electronics",
            "batteries": "batteries",
            "automotive": "automotive",
            "tires": "automotive",
            "packaging": "packaging",
            "clothing": "textiles",
            "textiles": "textiles",
        }

        target_coverage = category_coverage_map.get(
            category_lower, "packaging"
        )

        for scheme_id in applicable_schemes:
            scheme = EPR_SCHEME_FACTORS.get(scheme_id, {})
            coverage = scheme.get("coverage", "")
            if target_coverage in coverage:
                return scheme_id

        # Fall back to GLOBAL or first applicable
        if "GLOBAL_default" in applicable_schemes:
            return "GLOBAL_default"
        return applicable_schemes[0] if applicable_schemes else "GLOBAL_default"

    # ==========================================================================
    # RECYCLED CONTENT TRACKING
    # ==========================================================================

    def track_recycled_content(
        self, product: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Track recycled content in a product (pre-consumer and post-consumer).

        Pre-consumer: manufacturing scrap reprocessed in-house.
        Post-consumer: material from consumer waste streams.

        Args:
            product: Product dictionary with:
                - product_id (str)
                - total_weight_kg (Decimal)
                - materials (list): List of material dicts with:
                    - material_type (str): e.g., 'plastics_pet', 'aluminum'
                    - weight_kg (Decimal)
                    - recycled_content_pct (Decimal): 0-100
                    - pre_consumer_pct (Decimal): Fraction that is pre-consumer
                    - post_consumer_pct (Decimal): Fraction that is post-consumer

        Returns:
            Recycled content tracking result with credits.
        """
        product_id = product.get("product_id", "unknown")
        total_weight = _safe_decimal(product.get("total_weight_kg", _ZERO))
        materials = product.get("materials", [])

        if not materials:
            return {
                "product_id": product_id,
                "total_weight_kg": total_weight,
                "total_recycled_content_pct": _ZERO,
                "pre_consumer_pct": _ZERO,
                "post_consumer_pct": _ZERO,
                "total_avoided_kgco2e_memo": _ZERO,
                "materials": [],
            }

        total_recycled_kg = _ZERO
        total_pre_consumer_kg = _ZERO
        total_post_consumer_kg = _ZERO
        total_avoided = _ZERO
        material_results: List[Dict[str, Any]] = []

        for mat in materials:
            mat_type = mat.get("material_type", "unknown")
            mat_weight = _safe_decimal(mat.get("weight_kg", _ZERO))
            recycled_pct = _safe_decimal(
                mat.get("recycled_content_pct", _ZERO)
            ) / _HUNDRED
            pre_pct = _safe_decimal(
                mat.get("pre_consumer_pct", _ZERO)
            ) / _HUNDRED
            post_pct = _safe_decimal(
                mat.get("post_consumer_pct", _ZERO)
            ) / _HUNDRED

            recycled_kg = _round_decimal(mat_weight * recycled_pct, 2)
            pre_kg = _round_decimal(recycled_kg * pre_pct, 2)
            post_kg = _round_decimal(recycled_kg * post_pct, 2)

            # Look up credit factor
            credit_info = RECYCLED_CONTENT_CREDIT_FACTORS.get(mat_type, {})
            credit_per_kg = credit_info.get("credit_per_kg", _ZERO)
            avoided = _round_decimal(recycled_kg * credit_per_kg)

            total_recycled_kg += recycled_kg
            total_pre_consumer_kg += pre_kg
            total_post_consumer_kg += post_kg
            total_avoided += avoided

            material_results.append({
                "material_type": mat_type,
                "weight_kg": mat_weight,
                "recycled_content_pct": _round_decimal(
                    recycled_pct * _HUNDRED, 1
                ),
                "recycled_kg": recycled_kg,
                "pre_consumer_kg": pre_kg,
                "post_consumer_kg": post_kg,
                "virgin_displacement_credit_kgco2e": credit_per_kg,
                "avoided_emissions_kgco2e_memo": avoided,
            })

        total_recycled_pct = (
            _round_decimal(total_recycled_kg / total_weight * _HUNDRED, 1)
            if total_weight > _ZERO
            else _ZERO
        )

        return {
            "product_id": product_id,
            "total_weight_kg": total_weight,
            "total_recycled_kg": total_recycled_kg,
            "total_recycled_content_pct": total_recycled_pct,
            "pre_consumer_kg": total_pre_consumer_kg,
            "post_consumer_kg": total_post_consumer_kg,
            "pre_consumer_pct": (
                _round_decimal(
                    total_pre_consumer_kg / total_weight * _HUNDRED, 1
                )
                if total_weight > _ZERO
                else _ZERO
            ),
            "post_consumer_pct": (
                _round_decimal(
                    total_post_consumer_kg / total_weight * _HUNDRED, 1
                )
                if total_weight > _ZERO
                else _ZERO
            ),
            "total_avoided_kgco2e_memo": _round_decimal(total_avoided),
            "materials": material_results,
            "note": (
                "Avoided emissions are a MEMO item only. They must NOT be "
                "deducted from gross emissions per GHG Protocol."
            ),
        }

    # ==========================================================================
    # CORE CALCULATION
    # ==========================================================================

    def calculate(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Calculate producer-specific EOL emissions for sold products.

        Uses producer-declared data from EPDs, PCFs, or direct declarations.
        Each product must have at least one of: epd_data, pcf_data, or
        declared_eol_kgco2e_per_unit.

        Args:
            products: List of product dictionaries, each containing:
                - product_id (str)
                - units_sold (int)
                - weight_per_unit_kg (Decimal, optional)
                - epd_data (dict, optional): EPD with C1-C4 modules.
                - pcf_data (dict, optional): PCF with eol_emissions_kgco2e.
                - declared_eol_kgco2e_per_unit (Decimal, optional): Direct.
                - verification_status (str, optional): Override level.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Calculation result dictionary.
        """
        start_time = time.monotonic()
        calc_id = _generate_id("eol_ps")

        logger.info(
            "ProducerSpecificCalculatorEngine.calculate: calc_id=%s, "
            "org=%s, year=%d, products=%d",
            calc_id, org_id, year, len(products),
        )

        self._validate_product_list(products, org_id, year)

        product_results: List[Dict[str, Any]] = []
        total_co2e_kg = _ZERO
        total_weight_kg = _ZERO
        errors: List[Dict[str, str]] = []
        verification_summary: Dict[str, int] = {}

        for idx, product in enumerate(products):
            try:
                result = self._calculate_single_product(product, idx)
                product_results.append(result)
                total_co2e_kg += result["co2e_kg"]
                total_weight_kg += result.get("weight_kg", _ZERO)

                v_status = result.get("verification_status", "default")
                verification_summary[v_status] = (
                    verification_summary.get(v_status, 0) + 1
                )
            except Exception as exc:
                product_id = product.get("product_id", f"product_{idx}")
                logger.error(
                    "Product calculation failed: product=%s, error=%s",
                    product_id, str(exc),
                )
                errors.append({
                    "product_id": product_id,
                    "error": str(exc),
                })

        total_co2e_kg = _round_decimal(total_co2e_kg)
        total_co2e_tonnes = _round_decimal(total_co2e_kg * TONNES_PER_KG)
        total_weight_kg = _round_decimal(total_weight_kg, 2)

        # Determine dominant verification level
        dominant_verification = max(
            verification_summary,
            key=lambda k: verification_summary[k],
            default="default",
        ) if verification_summary else "default"

        dqi_score = self.compute_dqi_score(dominant_verification)
        uncertainty = self.compute_uncertainty(
            total_co2e_kg, dominant_verification
        )

        provenance_data = (
            f"{calc_id}|{org_id}|{year}|producer_specific|"
            f"{total_co2e_kg}|{total_weight_kg}|"
            f"{len(product_results)}|{dominant_verification}"
        )
        provenance_hash = _compute_hash(provenance_data)

        duration = time.monotonic() - start_time
        self._record_metrics(
            data_source="producer_specific",
            co2e_kg=total_co2e_kg,
            duration=duration,
            status="success" if not errors else "partial",
        )
        count = self._increment_calculation_count()

        return {
            "calculation_id": calc_id,
            "org_id": org_id,
            "year": year,
            "method": "producer_specific",
            "method_description": (
                "GHG Protocol Scope 3 Category 12 producer-specific method "
                "(Method D). Uses EPD, PCF, or producer-declared data."
            ),
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "total_weight_kg": total_weight_kg,
            "product_count": len(products),
            "success_count": len(product_results),
            "error_count": len(errors),
            "product_results": product_results,
            "errors": errors,
            "verification_summary": verification_summary,
            "dominant_verification": dominant_verification,
            "dqi_score": dqi_score,
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
    # SINGLE PRODUCT CALCULATION
    # ==========================================================================

    def _calculate_single_product(
        self,
        product: Dict[str, Any],
        index: int,
    ) -> Dict[str, Any]:
        """
        Calculate EOL emissions for a single product using producer data.

        Priority:
            1. EPD data (C1-C4 modules)
            2. PCF data (eol_emissions_kgco2e)
            3. declared_eol_kgco2e_per_unit (direct declaration)

        Args:
            product: Product dictionary.
            index: Index in products list.

        Returns:
            Per-product result dictionary.
        """
        product_id = product.get("product_id", f"product_{index}")
        units_sold = int(product.get("units_sold", 0))
        if units_sold <= 0:
            raise ValueError(
                f"Product '{product_id}': units_sold must be > 0"
            )

        weight_per_unit = product.get("weight_per_unit_kg")
        if weight_per_unit is not None:
            weight_per_unit = _safe_decimal(weight_per_unit)

        # Determine data source and calculate per-unit EOL
        epd_data = product.get("epd_data")
        pcf_data = product.get("pcf_data")
        declared_eol = product.get("declared_eol_kgco2e_per_unit")

        eol_per_unit = _ZERO
        data_source = "none"
        eol_breakdown: Dict[str, Any] = {}
        verification = product.get("verification_status", "default")
        module_d_memo = _ZERO

        if epd_data is not None:
            # Priority 1: EPD data
            eol_scenario = self.parse_eol_scenario(epd_data)
            eol_per_unit = eol_scenario["total_eol_kgco2e"]
            module_d_memo = eol_scenario.get("module_d_benefits", _ZERO)
            data_source = "epd"
            eol_breakdown = eol_scenario["module_breakdown"]
            verification = epd_data.get("verification_status", verification)

        elif pcf_data is not None:
            # Priority 2: PCF data
            raw_eol = pcf_data.get("eol_emissions_kgco2e", _ZERO)
            eol_per_unit = _safe_decimal(raw_eol)
            data_source = "pcf"
            eol_breakdown = {"eol_total": eol_per_unit}
            verification = pcf_data.get("verification_status", verification)
            module_d_raw = pcf_data.get("avoided_emissions_kgco2e_memo", _ZERO)
            module_d_memo = _safe_decimal(module_d_raw)

        elif declared_eol is not None:
            # Priority 3: Direct declaration
            eol_per_unit = _safe_decimal(declared_eol)
            data_source = "declared"
            eol_breakdown = {"declared_eol_per_unit": eol_per_unit}
            if verification == "default":
                verification = "self_declared"

        else:
            raise ValueError(
                f"Product '{product_id}': no producer-specific data found. "
                f"Provide epd_data, pcf_data, or declared_eol_kgco2e_per_unit."
            )

        if verification not in VERIFICATION_LEVELS:
            verification = "default"

        # Calculate total emissions
        co2e_kg = _round_decimal(eol_per_unit * Decimal(str(units_sold)))
        total_weight_kg = _ZERO
        if weight_per_unit is not None:
            total_weight_kg = _round_decimal(
                weight_per_unit * Decimal(str(units_sold)), 2
            )

        # Module D (avoided/benefits) as memo
        module_d_total = _round_decimal(
            module_d_memo * Decimal(str(units_sold))
        )

        provenance_data = (
            f"{product_id}|{data_source}|{units_sold}|"
            f"{eol_per_unit}|{verification}|{co2e_kg}"
        )

        return {
            "product_id": product_id,
            "units_sold": units_sold,
            "weight_per_unit_kg": weight_per_unit,
            "weight_kg": total_weight_kg,
            "data_source": data_source,
            "verification_status": verification,
            "verification_tier": VERIFICATION_LEVELS.get(
                verification, {}
            ).get("tier", "tier_3"),
            "eol_per_unit_kgco2e": _round_decimal(eol_per_unit),
            "co2e_kg": co2e_kg,
            "co2e_tonnes": _round_decimal(co2e_kg * TONNES_PER_KG),
            "eol_breakdown": eol_breakdown,
            "module_d_benefits_memo_kgco2e": module_d_total,
            "note_module_d": (
                "Module D benefits (avoided emissions) are a MEMO item. "
                "NOT deducted from gross emissions per GHG Protocol."
            ),
            "provenance_hash": _compute_hash(provenance_data),
        }

    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    def _validate_product_list(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        year: int,
    ) -> None:
        """Validate the input product list."""
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
        valid_count = 0

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
                    errors.append(f"Product at index {idx} must be a dict")
                    continue

                has_data = any([
                    product.get("epd_data"),
                    product.get("pcf_data"),
                    product.get("declared_eol_kgco2e_per_unit"),
                ])
                if not has_data:
                    errors.append(
                        f"Product '{product_id}': needs epd_data, pcf_data, "
                        f"or declared_eol_kgco2e_per_unit"
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

                # Validate EPD if present
                if product.get("epd_data"):
                    epd_val = self.validate_epd(product["epd_data"])
                    if not epd_val["is_valid"]:
                        for err in epd_val["errors"]:
                            warnings.append(
                                f"Product '{product_id}' EPD: {err}"
                            )

                valid_count += 1

        return {
            "is_valid": len(errors) == 0,
            "valid_product_count": valid_count,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }

    # ==========================================================================
    # DATA QUALITY INDICATOR
    # ==========================================================================

    def compute_dqi_score(
        self, verification_status: str = "third_party_verified"
    ) -> Dict[str, Any]:
        """
        Compute data quality indicator scores for producer-specific method.

        Producer-specific with third-party verification is Tier 1 (highest).

        Args:
            verification_status: Verification level of the dominant data source.

        Returns:
            DQI score dictionary.
        """
        scores = DQI_PRODUCER_SPECIFIC.get(
            verification_status,
            DQI_PRODUCER_SPECIFIC["default"],
        )
        composite = Decimal(str(sum(scores.values()))) / Decimal(
            str(len(scores))
        )
        composite = _round_decimal(composite, 2)

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

        verification_info = VERIFICATION_LEVELS.get(
            verification_status,
            VERIFICATION_LEVELS["default"],
        )

        return {
            "method": "producer_specific",
            "tier": verification_info.get("tier", "tier_3"),
            "verification_status": verification_status,
            "dimension_scores": dict(scores),
            "composite_score": composite,
            "classification": classification,
            "description": verification_info.get("description", ""),
        }

    # ==========================================================================
    # UNCERTAINTY
    # ==========================================================================

    def compute_uncertainty(
        self,
        total_co2e_kg: Decimal,
        verification_status: str = "third_party_verified",
    ) -> Dict[str, Any]:
        """
        Compute uncertainty range for producer-specific calculation.

        Uncertainty depends on verification level:
            - third_party_verified: +/-10%
            - self_declared: +/-20%
            - estimated: +/-35%
            - default: +/-50%

        Args:
            total_co2e_kg: Total calculated emissions.
            verification_status: Verification level.

        Returns:
            Uncertainty dictionary.
        """
        verification_info = VERIFICATION_LEVELS.get(
            verification_status,
            VERIFICATION_LEVELS["default"],
        )
        pct = verification_info["uncertainty_pct"]

        lower_bound = _round_decimal(total_co2e_kg * (_ONE - pct))
        upper_bound = _round_decimal(total_co2e_kg * (_ONE + pct))

        return {
            "method": "producer_specific_uncertainty",
            "verification_status": verification_status,
            "total_co2e_kg": total_co2e_kg,
            "lower_bound_co2e_kg": lower_bound,
            "upper_bound_co2e_kg": upper_bound,
            "uncertainty_pct": pct,
            "confidence_level": Decimal("0.95"),
            "description": (
                f"Producer-specific uncertainty for {verification_status}: "
                f"+/-{pct * _HUNDRED}% at 95% confidence."
            ),
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

        # Check 1: Verification levels loaded
        vl_count = len(VERIFICATION_LEVELS)
        vl_ok = vl_count >= 4
        checks.append({
            "check": "verification_levels",
            "status": "pass" if vl_ok else "fail",
            "detail": f"{vl_count} verification levels loaded (expected >= 4)",
        })
        if not vl_ok:
            overall_healthy = False

        # Check 2: EPR schemes loaded
        epr_count = len(EPR_SCHEME_FACTORS)
        epr_ok = epr_count >= 10
        checks.append({
            "check": "epr_schemes",
            "status": "pass" if epr_ok else "fail",
            "detail": f"{epr_count} EPR schemes loaded (expected >= 10)",
        })
        if not epr_ok:
            overall_healthy = False

        # Check 3: Take-back EFs loaded
        tb_count = len(TAKE_BACK_EF)
        tb_ok = tb_count >= 7
        checks.append({
            "check": "take_back_efs",
            "status": "pass" if tb_ok else "fail",
            "detail": f"{tb_count} take-back EFs loaded (expected >= 7)",
        })
        if not tb_ok:
            overall_healthy = False

        # Check 4: Recycled content factors loaded
        rc_count = len(RECYCLED_CONTENT_CREDIT_FACTORS)
        rc_ok = rc_count >= 10
        checks.append({
            "check": "recycled_content_factors",
            "status": "pass" if rc_ok else "fail",
            "detail": f"{rc_count} recycled content factors loaded (expected >= 10)",
        })
        if not rc_ok:
            overall_healthy = False

        # Check 5: EPD validation fields defined
        epd_ok = len(ISO_14025_REQUIRED_FIELDS) >= 8
        checks.append({
            "check": "epd_validation_fields",
            "status": "pass" if epd_ok else "fail",
            "detail": (
                f"{len(ISO_14025_REQUIRED_FIELDS)} ISO 14025 fields "
                f"+ {len(ISO_14025_EOL_REQUIRED_FIELDS)} EOL fields defined"
            ),
        })
        if not epd_ok:
            overall_healthy = False

        # Check 6: GLOBAL EPR default exists
        global_ok = "GLOBAL_default" in EPR_SCHEME_FACTORS
        checks.append({
            "check": "global_epr_default",
            "status": "pass" if global_ok else "fail",
            "detail": (
                "GLOBAL_default EPR scheme exists"
                if global_ok
                else "MISSING GLOBAL_default EPR scheme"
            ),
        })
        if not global_ok:
            overall_healthy = False

        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "status": "healthy" if overall_healthy else "unhealthy",
            "checks": checks,
            "calculation_count": self._calculation_count,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }


# ==============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ==============================================================================


def get_producer_specific_calculator(
    gwp_version: str = "ar5",
) -> ProducerSpecificCalculatorEngine:
    """
    Get the singleton ProducerSpecificCalculatorEngine instance.

    Args:
        gwp_version: IPCC GWP version ('ar4', 'ar5', 'ar6').

    Returns:
        ProducerSpecificCalculatorEngine singleton instance.
    """
    return ProducerSpecificCalculatorEngine.get_instance(
        gwp_version=gwp_version
    )


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Engine class
    "ProducerSpecificCalculatorEngine",
    # Convenience function
    "get_producer_specific_calculator",
    # Data tables
    "VERIFICATION_LEVELS",
    "EPD_LIFECYCLE_MODULES",
    "EOL_MODULES",
    "EPR_SCHEME_FACTORS",
    "TAKE_BACK_EF",
    "RECYCLED_CONTENT_CREDIT_FACTORS",
    "ISO_14025_REQUIRED_FIELDS",
    "ISO_14025_EOL_REQUIRED_FIELDS",
    "EN_15804_REQUIRED_FIELDS",
    "PCF_REQUIRED_FIELDS",
    "GHG_PROTOCOL_PRODUCT_STANDARD_FIELDS",
    "DQI_PRODUCER_SPECIFIC",
    "GWP_VALUES",
    # Constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "ENGINE_ID",
    "ENGINE_VERSION",
    "TABLE_PREFIX",
    "PRECISION",
]
