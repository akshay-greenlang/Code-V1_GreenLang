# -*- coding: utf-8 -*-
"""
Waste Generated in Operations Prometheus Metrics - AGENT-MRV-018

14 Prometheus metrics with gl_wg_ prefix for monitoring the
GL-MRV-S3-005 Waste Generated in Operations Agent.

This module provides Prometheus metrics tracking for waste generated in
operations emissions calculations (Scope 3, Category 5) including
waste-type-specific, treatment-specific, supplier-specific, and
average-data calculation methods across all waste treatment pathways
(landfill, incineration, recycling, composting, anaerobic digestion,
wastewater treatment, and open burning).

Thread-safe singleton pattern with graceful fallback if Prometheus unavailable.

Metrics prefix: gl_wg_

14 Prometheus Metrics:
    1.  gl_wg_calculations_total              - Counter: total calculations performed
    2.  gl_wg_calculation_errors_total         - Counter: calculation errors by type
    3.  gl_wg_calculation_duration_seconds     - Histogram: calculation durations
    4.  gl_wg_emissions_co2e_tonnes            - Counter: total emissions in tCO2e
    5.  gl_wg_waste_mass_tonnes                - Counter: total waste mass processed
    6.  gl_wg_landfill_ch4_generated_kg        - Counter: landfill methane generated
    7.  gl_wg_incineration_energy_recovered_mwh - Counter: incineration energy recovered
    8.  gl_wg_recycling_avoided_emissions_co2e - Counter: avoided emissions from recycling
    9.  gl_wg_wastewater_organic_load_kg       - Counter: wastewater organic load processed
    10. gl_wg_diversion_rate                   - Gauge: waste diversion rate by facility
    11. gl_wg_compliance_checks_total          - Counter: compliance checks performed
    12. gl_wg_data_quality_score               - Gauge: data quality score by source
    13. gl_wg_batch_size                       - Histogram: batch calculation sizes
    14. gl_wg_ef_lookups_total                 - Counter: emission factor lookups

GHG Protocol Scope 3 Category 5 covers waste generated in operations:
    A. Disposal and treatment of waste generated in the reporting
       company's owned or controlled operations (in facilities not
       owned or controlled by the reporting company).
    B. Waste treatment methods include landfill, incineration (with
       and without energy recovery), recycling, composting, anaerobic
       digestion, wastewater treatment, and open burning.
    C. Emissions from waste-in-transit are excluded (covered by
       Category 4 or 9).

Calculation methods defined by GHG Protocol:
    - Waste-type-specific: waste mass x waste-type-specific EF per treatment
    - Treatment-specific: waste mass x treatment-specific EF
    - Supplier-specific: primary data from waste management providers
    - Average-data: waste mass x average EF across treatments
    - IPCC First Order Decay (FOD): for landfill methane modelling

Waste treatment pathways tracked:
    - Landfill (managed aerobic, managed anaerobic, unmanaged, semi-aerobic)
    - Incineration (mass burn, fluidized bed, RDF, with/without energy recovery)
    - Recycling (mechanical, chemical, feedstock, downcycling)
    - Composting (windrow, in-vessel, aerated static pile, vermicomposting)
    - Anaerobic digestion (wet, dry, thermophilic, mesophilic)
    - Wastewater treatment (aerobic, anaerobic, lagoon, septic, constructed wetland)
    - Open burning (controlled, uncontrolled)
    - Other (waste-to-energy, pyrolysis, gasification)

Example:
    >>> metrics = WasteGeneratedMetrics()
    >>> metrics.record_calculation(
    ...     method="waste_type_specific",
    ...     treatment="landfill",
    ...     waste_category="municipal_solid",
    ...     tenant_id="tenant-001",
    ...     status="success",
    ...     emissions_tco2e=8.45,
    ...     duration_s=0.35
    ... )
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional, Generator
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful Prometheus import -- fall back to no-op stubs when the client
# library is not installed, ensuring the agent still operates correctly.
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not available, metrics will be no-ops")
    PROMETHEUS_AVAILABLE = False

    class Counter:  # type: ignore[no-redef]
        """No-op Counter stub for environments without prometheus_client."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> "Counter":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

    class Histogram:  # type: ignore[no-redef]
        """No-op Histogram stub for environments without prometheus_client."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> "Histogram":
            return self

        def observe(self, amount: float) -> None:
            pass

    class Gauge:  # type: ignore[no-redef]
        """No-op Gauge stub for environments without prometheus_client."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> "Gauge":
            return self

        def set(self, value: float) -> None:
            pass

        def inc(self, amount: float = 1) -> None:
            pass

        def dec(self, amount: float = 1) -> None:
            pass

    class Info:  # type: ignore[no-redef]
        """No-op Info stub for environments without prometheus_client."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def info(self, data: Dict[str, str]) -> None:
            pass


# ===========================================================================
# Enumerations -- Waste Generated domain-specific label value sets
# ===========================================================================

class CalculationMethod(str, Enum):
    """
    Calculation methods for waste generated in operations emissions.

    GHG Protocol Scope 3 Category 5 supports several approaches depending
    on data availability and the level of accuracy required:
        - Waste-type-specific: Uses waste mass by type x type-specific EFs
        - Treatment-specific: Uses waste mass x treatment-pathway-specific EFs
        - Supplier-specific: Uses primary data from waste management providers
        - Average-data: Uses total waste mass x average blended EFs
        - IPCC FOD: First Order Decay model for landfill methane modelling
    """
    WASTE_TYPE_SPECIFIC = "waste_type_specific"
    TREATMENT_SPECIFIC = "treatment_specific"
    SUPPLIER_SPECIFIC = "supplier_specific"
    AVERAGE_DATA = "average_data"
    IPCC_FOD = "ipcc_fod"


class TreatmentMethod(str, Enum):
    """
    Waste treatment methods for emissions tracking.

    Covers the primary waste treatment pathways defined in the GHG Protocol
    Technical Guidance for Calculating Scope 3 Emissions, Category 5,
    including landfill, incineration, recycling, composting, anaerobic
    digestion, wastewater treatment, open burning, and other methods.
    """
    LANDFILL = "landfill"
    LANDFILL_MANAGED_AEROBIC = "landfill_managed_aerobic"
    LANDFILL_MANAGED_ANAEROBIC = "landfill_managed_anaerobic"
    LANDFILL_UNMANAGED = "landfill_unmanaged"
    LANDFILL_SEMI_AEROBIC = "landfill_semi_aerobic"
    INCINERATION = "incineration"
    INCINERATION_ENERGY_RECOVERY = "incineration_energy_recovery"
    INCINERATION_NO_RECOVERY = "incineration_no_recovery"
    RECYCLING = "recycling"
    COMPOSTING = "composting"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"
    WASTEWATER_TREATMENT = "wastewater_treatment"
    OPEN_BURNING = "open_burning"
    WASTE_TO_ENERGY = "waste_to_energy"
    PYROLYSIS = "pyrolysis"
    GASIFICATION = "gasification"
    REUSE = "reuse"
    OTHER = "other"


class WasteCategory(str, Enum):
    """
    Waste categories tracked for waste-type-specific emission calculations.

    Aligned with IPCC 2006 Guidelines for National Greenhouse Gas
    Inventories (Volume 5, Waste), EU Waste Framework Directive
    classifications, and EPA waste characterization studies.
    """
    MUNICIPAL_SOLID = "municipal_solid"
    FOOD_WASTE = "food_waste"
    PAPER_CARDBOARD = "paper_cardboard"
    WOOD = "wood"
    TEXTILES = "textiles"
    GARDEN_WASTE = "garden_waste"
    PLASTIC = "plastic"
    GLASS = "glass"
    METAL = "metal"
    RUBBER_LEATHER = "rubber_leather"
    CONSTRUCTION_DEMOLITION = "construction_demolition"
    INDUSTRIAL = "industrial"
    COMMERCIAL = "commercial"
    HAZARDOUS = "hazardous"
    ELECTRONIC_WASTE = "electronic_waste"
    MEDICAL = "medical"
    SLUDGE = "sludge"
    ASH = "ash"
    MINERAL = "mineral"
    CHEMICAL = "chemical"
    MIXED = "mixed"
    OTHER = "other"


class CalculationStatus(str, Enum):
    """Calculation operation status for waste generated calculations."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    INSUFFICIENT_DATA = "insufficient_data"
    SKIPPED = "skipped"


class ClimateZone(str, Enum):
    """
    IPCC climate zones for landfill methane modelling.

    Landfill methane generation rates (k values) and methane generation
    potential (L0) vary significantly with climate conditions. The IPCC
    defines climate zones for selecting default parameters in the First
    Order Decay (FOD) model.
    """
    BOREAL_ARCTIC_DRY = "boreal_arctic_dry"
    BOREAL_ARCTIC_WET = "boreal_arctic_wet"
    TEMPERATE_DRY = "temperate_dry"
    TEMPERATE_WET = "temperate_wet"
    TROPICAL_DRY = "tropical_dry"
    TROPICAL_WET = "tropical_wet"
    TROPICAL_MONTANE = "tropical_montane"
    DEFAULT = "default"


class IncineratorType(str, Enum):
    """
    Incinerator types for incineration energy recovery tracking.

    Different incinerator technologies have varying combustion
    efficiencies, energy recovery rates, and emission profiles.
    """
    MASS_BURN = "mass_burn"
    MODULAR = "modular"
    FLUIDIZED_BED = "fluidized_bed"
    ROTARY_KILN = "rotary_kiln"
    RDF = "rdf"
    CEMENT_KILN = "cement_kiln"
    INDUSTRIAL = "industrial"
    MEDICAL_WASTE = "medical_waste"
    HAZARDOUS_WASTE = "hazardous_waste"
    OTHER = "other"


class RecyclingType(str, Enum):
    """
    Recycling process types for avoided emissions tracking.

    Different recycling approaches displace varying amounts of
    virgin material production, leading to different avoided
    emission credit calculations.
    """
    MECHANICAL = "mechanical"
    CHEMICAL = "chemical"
    FEEDSTOCK = "feedstock"
    DOWNCYCLING = "downcycling"
    CLOSED_LOOP = "closed_loop"
    OPEN_LOOP = "open_loop"
    UPCYCLING = "upcycling"
    OTHER = "other"


class MeasurementBasis(str, Enum):
    """
    Measurement basis for wastewater organic load tracking.

    Wastewater emissions depend on the organic load measured as
    Biochemical Oxygen Demand (BOD) or Chemical Oxygen Demand (COD),
    with different emission factors for each.
    """
    BOD = "bod"
    COD = "cod"
    TOC = "toc"
    TSS = "tss"
    VOLUMETRIC = "volumetric"
    OTHER = "other"


class WastewaterTreatmentSystem(str, Enum):
    """
    Wastewater treatment system types for emissions tracking.

    Different treatment systems produce varying amounts of CH4 and N2O
    depending on the treatment technology and operating conditions.
    """
    AEROBIC_CENTRALIZED = "aerobic_centralized"
    AEROBIC_DECENTRALIZED = "aerobic_decentralized"
    ANAEROBIC_REACTOR = "anaerobic_reactor"
    ANAEROBIC_LAGOON = "anaerobic_lagoon"
    FACULTATIVE_LAGOON = "facultative_lagoon"
    SEPTIC_SYSTEM = "septic_system"
    CONSTRUCTED_WETLAND = "constructed_wetland"
    TRICKLING_FILTER = "trickling_filter"
    ACTIVATED_SLUDGE = "activated_sludge"
    MEMBRANE_BIOREACTOR = "membrane_bioreactor"
    SEQUENCING_BATCH = "sequencing_batch"
    OXIDATION_DITCH = "oxidation_ditch"
    UNTREATED_DISCHARGE = "untreated_discharge"
    OTHER = "other"


class ErrorType(str, Enum):
    """Error types for detailed tracking of waste calculation failures."""
    VALIDATION_ERROR = "validation_error"
    CALCULATION_ERROR = "calculation_error"
    DATABASE_ERROR = "database_error"
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_NOT_FOUND = "data_not_found"
    EMISSION_FACTOR_UNAVAILABLE = "emission_factor_unavailable"
    WASTE_TYPE_UNKNOWN = "waste_type_unknown"
    TREATMENT_METHOD_UNKNOWN = "treatment_method_unknown"
    MASS_CONVERSION_ERROR = "mass_conversion_error"
    COMPOSITION_INVALID = "composition_invalid"
    DOC_CALCULATION_ERROR = "doc_calculation_error"
    FOD_MODEL_ERROR = "fod_model_error"
    ENERGY_RECOVERY_ERROR = "energy_recovery_error"
    ORGANIC_LOAD_ERROR = "organic_load_error"
    DIVERSION_RATE_ERROR = "diversion_rate_error"
    SUPPLIER_DATA_UNAVAILABLE = "supplier_data_unavailable"


class Framework(str, Enum):
    """
    Compliance frameworks for waste generated reporting.

    Tracks validation against regulatory and voluntary reporting
    standards applicable to Scope 3 Category 5 emissions.
    """
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    ISO_14001 = "iso_14001"
    CSRD = "csrd"
    ESRS = "esrs"
    TCFD = "tcfd"
    CDP = "cdp"
    SBTI = "sbti"
    EU_WASTE_FRAMEWORK = "eu_waste_framework"
    EPA_WARM = "epa_warm"
    IPCC_2006 = "ipcc_2006"
    IPCC_2019 = "ipcc_2019"
    ZERO_WASTE = "zero_waste"
    SEC_CLIMATE = "sec_climate"
    EU_TAXONOMY = "eu_taxonomy"
    TNFD = "tnfd"


class ComplianceStatus(str, Enum):
    """Compliance check result status for waste generated calculations."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_REVIEW = "needs_review"


class EFSource(str, Enum):
    """
    Emission factor sources for waste-related emission calculations.

    Tracks the origin of emission factors used in waste treatment
    calculations, enabling monitoring of factor database coverage
    and source distribution.
    """
    IPCC_2006 = "ipcc_2006"
    IPCC_2019 = "ipcc_2019"
    EPA_WARM = "epa_warm"
    DEFRA = "defra"
    ECOINVENT = "ecoinvent"
    EXIOBASE = "exiobase"
    GHG_PROTOCOL = "ghg_protocol"
    EU_ETS = "eu_ets"
    NATIONAL_INVENTORY = "national_inventory"
    SUPPLIER_SPECIFIC = "supplier_specific"
    CUSTOM = "custom"
    OTHER = "other"


class DataSource(str, Enum):
    """
    Data sources for data quality score tracking.

    Tracks the origin and quality tier of input data used in
    waste generated calculations.
    """
    DIRECT_MEASUREMENT = "direct_measurement"
    WASTE_MANIFEST = "waste_manifest"
    WASTE_AUDIT = "waste_audit"
    SUPPLIER_REPORT = "supplier_report"
    PURCHASE_RECORDS = "purchase_records"
    ERP_SYSTEM = "erp_system"
    ESTIMATION = "estimation"
    INDUSTRY_AVERAGE = "industry_average"
    PROXY_DATA = "proxy_data"
    OTHER = "other"


class BatchStatus(str, Enum):
    """Batch processing job status for waste generated calculations."""
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


# ===========================================================================
# WasteGeneratedMetrics -- Thread-safe Singleton
# ===========================================================================

class WasteGeneratedMetrics:
    """
    Thread-safe singleton metrics collector for Waste Generated Agent (MRV-018).

    Provides 14 Prometheus metrics for tracking Scope 3 Category 5
    waste generated in operations emissions calculations, including
    waste-type-specific, treatment-specific, supplier-specific, and
    average-data methods across all waste treatment pathways.

    All metrics use the ``gl_wg_`` prefix to ensure namespace isolation
    within the GreenLang Prometheus ecosystem.

    Scope 3 Category 5 Sub-Categories Tracked:
        A. Disposal and treatment of solid waste from owned/controlled operations
        B. Disposal and treatment of wastewater from owned/controlled operations
        C. Waste treatment methods: landfill, incineration, recycling, composting,
           anaerobic digestion, wastewater treatment, open burning, and others

    Calculation Methods Supported:
        - Waste-type-specific: waste mass x waste-type-specific EF per treatment
        - Treatment-specific: waste mass x treatment-pathway-specific EF
        - Supplier-specific: primary data from waste management providers
        - Average-data: total waste mass x average blended EF
        - IPCC FOD: First Order Decay model for landfill methane

    Waste Treatment Pathways:
        - Landfill (managed aerobic, managed anaerobic, unmanaged, semi-aerobic)
        - Incineration (mass burn, fluidized bed, RDF, with/without energy recovery)
        - Recycling (mechanical, chemical, feedstock, closed-loop, open-loop)
        - Composting (windrow, in-vessel, aerated static pile, vermicomposting)
        - Anaerobic digestion (wet, dry, thermophilic, mesophilic)
        - Wastewater treatment (aerobic, anaerobic, lagoon, septic, wetland)
        - Open burning (controlled, uncontrolled)
        - Other (waste-to-energy, pyrolysis, gasification)

    Attributes:
        calculations_total: Counter for total calculation operations
        calculation_errors_total: Counter for calculation errors by type
        calculation_duration_seconds: Histogram for operation durations
        emissions_co2e_tonnes: Counter for total emissions in tCO2e
        waste_mass_tonnes: Counter for total waste mass processed
        landfill_ch4_generated_kg: Counter for landfill methane generated
        incineration_energy_recovered_mwh: Counter for energy recovered
        recycling_avoided_emissions_co2e: Counter for recycling avoided emissions
        wastewater_organic_load_kg: Counter for wastewater organic load
        diversion_rate: Gauge for waste diversion rate by facility
        compliance_checks_total: Counter for compliance checks
        data_quality_score: Gauge for data quality score by source
        batch_size: Histogram for batch calculation sizes
        ef_lookups_total: Counter for emission factor lookups

    Example:
        >>> metrics = WasteGeneratedMetrics()
        >>> metrics.record_calculation(
        ...     method="waste_type_specific",
        ...     treatment="landfill",
        ...     waste_category="municipal_solid",
        ...     tenant_id="tenant-001",
        ...     status="success",
        ...     emissions_tco2e=8.45,
        ...     duration_s=0.35
        ... )
        >>> summary = metrics.get_metrics_summary()
        >>> assert summary['calculations'] == 1
    """

    _instance: Optional["WasteGeneratedMetrics"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "WasteGeneratedMetrics":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize metrics (only once due to singleton)."""
        if hasattr(self, '_initialized'):
            return

        self._initialized: bool = True
        self._start_time: datetime = datetime.utcnow()
        self._stats_lock: threading.Lock = threading.Lock()
        self._in_memory_stats: Dict[str, Any] = {
            'calculations': 0,
            'calculation_errors': 0,
            'emissions_tco2e': 0.0,
            'waste_mass_tonnes': 0.0,
            'landfill_ch4_kg': 0.0,
            'incineration_energy_mwh': 0.0,
            'recycling_avoided_co2e': 0.0,
            'wastewater_organic_load_kg': 0.0,
            'compliance_checks': 0,
            'ef_lookups': 0,
            'batch_jobs': 0,
            'errors': 0,
        }

        # Initialize Prometheus metrics
        self._init_metrics()

        logger.info(
            "WasteGeneratedMetrics initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable"
        )

    def _init_metrics(self) -> None:
        """
        Initialize all 14 Prometheus metrics with gl_wg_ prefix.

        Each metric is documented with its purpose, label set, and the
        Scope 3 Category 5 aspect it supports.

        When Prometheus is available, metric registration may fail if the
        metrics were previously registered (e.g., after a reset() call in
        tests). In that case we unregister from the default registry and
        re-register to obtain fresh collector objects.
        """
        # Helper to safely create a Prometheus metric, handling the case
        # where a metric with the same name is already registered.
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import REGISTRY

            def _safe_create(metric_cls: type, name: str, *args: Any, **kwargs: Any) -> Any:
                """Create a metric, unregistering any prior collector on conflict."""
                try:
                    return metric_cls(name, *args, **kwargs)
                except ValueError:
                    # Already registered -- unregister the old collector and retry
                    try:
                        REGISTRY.unregister(REGISTRY._names_to_collectors.get(name))
                    except Exception:
                        # Fallback: walk collectors to find the one owning this name
                        for collector in list(REGISTRY._names_to_collectors.values()):
                            try:
                                REGISTRY.unregister(collector)
                            except Exception:
                                pass
                    return metric_cls(name, *args, **kwargs)
        else:
            def _safe_create(metric_cls: type, name: str, *args: Any, **kwargs: Any) -> Any:
                """No-op stub creation (Prometheus not available)."""
                return metric_cls(name, *args, **kwargs)

        # ------------------------------------------------------------------
        # 1. gl_wg_calculations_total (Counter)
        #    Total waste generated emission calculations performed.
        #    Labels:
        #      - method: waste_type_specific, treatment_specific,
        #                supplier_specific, average_data, ipcc_fod
        #      - treatment: landfill, incineration, recycling, composting,
        #                   anaerobic_digestion, wastewater_treatment,
        #                   open_burning, waste_to_energy, other
        #      - waste_category: municipal_solid, food_waste, paper_cardboard,
        #                        plastic, glass, metal, hazardous, etc.
        #      - tenant_id: Tenant identifier for multi-tenant isolation
        #    Primary throughput counter for all calculation operations
        #    across methods, treatments, waste categories, and tenants.
        # ------------------------------------------------------------------
        self.calculations_total = _safe_create(Counter,
            'gl_wg_calculations_total',
            'Total waste generated emission calculations performed',
            ['method', 'treatment', 'waste_category', 'tenant_id']
        )

        # ------------------------------------------------------------------
        # 2. gl_wg_calculation_errors_total (Counter)
        #    Total calculation errors encountered during waste processing.
        #    Labels:
        #      - error_type: validation_error, calculation_error,
        #                    emission_factor_unavailable, fod_model_error,
        #                    waste_type_unknown, treatment_method_unknown, etc.
        #      - treatment: Treatment method where the error occurred
        #      - tenant_id: Tenant identifier
        #    Tracks error frequency by type and treatment for diagnostics,
        #    alerting, and root cause analysis.
        # ------------------------------------------------------------------
        self.calculation_errors_total = _safe_create(Counter,
            'gl_wg_calculation_errors_total',
            'Total waste generated calculation errors encountered',
            ['error_type', 'treatment', 'tenant_id']
        )

        # ------------------------------------------------------------------
        # 3. gl_wg_calculation_duration_seconds (Histogram)
        #    Duration of calculation operations in seconds.
        #    Labels:
        #      - method: Calculation method used
        #      - treatment: Waste treatment pathway
        #    Buckets tuned for typical waste calculation latencies:
        #      - 10ms for cached factor lookups and simple mass x EF
        #      - 50-100ms for waste-type-specific with composition analysis
        #      - 250ms-1s for FOD model runs with multi-year projections
        #      - 1-10s for complex batch or full-facility calculations
        # ------------------------------------------------------------------
        self.calculation_duration_seconds = _safe_create(Histogram,
            'gl_wg_calculation_duration_seconds',
            'Duration of waste generated calculation operations',
            ['method', 'treatment'],
            buckets=[
                0.01, 0.05, 0.1, 0.25, 0.5,
                1.0, 2.5, 5.0, 10.0
            ]
        )

        # ------------------------------------------------------------------
        # 4. gl_wg_emissions_co2e_tonnes (Counter)
        #    Total emissions calculated in tonnes CO2-equivalent (tCO2e).
        #    Labels:
        #      - treatment: Treatment pathway generating emissions
        #      - waste_category: Category of waste generating emissions
        #      - tenant_id: Tenant identifier
        #    Tracks cumulative emissions output enabling rate calculation,
        #    treatment pathway analysis, and waste category breakdown
        #    reporting. Uses tCO2e as the standard unit for Scope 3
        #    Category 5 reporting aligned with GHG Protocol guidance.
        # ------------------------------------------------------------------
        self.emissions_co2e_tonnes = _safe_create(Counter,
            'gl_wg_emissions_co2e_tonnes',
            'Total waste generated emissions calculated in tCO2e',
            ['treatment', 'waste_category', 'tenant_id']
        )

        # ------------------------------------------------------------------
        # 5. gl_wg_waste_mass_tonnes (Counter)
        #    Total waste mass processed through calculations in tonnes.
        #    Labels:
        #      - waste_category: Category of waste processed
        #      - treatment: Treatment pathway for the waste
        #      - tenant_id: Tenant identifier
        #    Tracks cumulative waste throughput by category and treatment
        #    enabling waste composition analysis, treatment split monitoring,
        #    and diversion rate calculation support.
        # ------------------------------------------------------------------
        self.waste_mass_tonnes = _safe_create(Counter,
            'gl_wg_waste_mass_tonnes',
            'Total waste mass processed in tonnes',
            ['waste_category', 'treatment', 'tenant_id']
        )

        # ------------------------------------------------------------------
        # 6. gl_wg_landfill_ch4_generated_kg (Counter)
        #    Total landfill methane (CH4) generated in kilograms.
        #    Labels:
        #      - waste_category: Category of waste landfilled
        #      - climate_zone: IPCC climate zone affecting decay rate
        #      - tenant_id: Tenant identifier
        #    Tracks methane generation from landfill decomposition.
        #    Landfill is typically the largest source of waste-related
        #    GHG emissions. CH4 generation depends on waste composition,
        #    degradable organic carbon (DOC) content, climate zone
        #    (temperature and moisture affecting decay rate k), and
        #    landfill management practice (gas collection efficiency).
        #    Uses kg as the unit since individual waste batches may
        #    generate relatively small CH4 quantities.
        # ------------------------------------------------------------------
        self.landfill_ch4_generated_kg = _safe_create(Counter,
            'gl_wg_landfill_ch4_generated_kg',
            'Total landfill methane generated in kg CH4',
            ['waste_category', 'climate_zone', 'tenant_id']
        )

        # ------------------------------------------------------------------
        # 7. gl_wg_incineration_energy_recovered_mwh (Counter)
        #    Total energy recovered from waste incineration in MWh.
        #    Labels:
        #      - incinerator_type: Type of incineration technology
        #      - tenant_id: Tenant identifier
        #    Tracks energy recovery from waste-to-energy incineration.
        #    Energy recovery offsets grid electricity or heat production,
        #    reducing net emissions from incineration. The net emission
        #    from incineration = gross emissions - avoided grid emissions.
        #    MWh is the standard unit for energy recovery reporting.
        # ------------------------------------------------------------------
        self.incineration_energy_recovered_mwh = _safe_create(Counter,
            'gl_wg_incineration_energy_recovered_mwh',
            'Total energy recovered from waste incineration in MWh',
            ['incinerator_type', 'tenant_id']
        )

        # ------------------------------------------------------------------
        # 8. gl_wg_recycling_avoided_emissions_co2e (Counter)
        #    Total avoided emissions from recycling in tCO2e.
        #    Labels:
        #      - recycling_type: Type of recycling process
        #      - waste_category: Category of waste recycled
        #      - tenant_id: Tenant identifier
        #    Tracks emission credits from recycling that displaces virgin
        #    material production. Avoided emissions = mass recycled x
        #    (virgin production EF - recycling process EF). These credits
        #    may be reported separately or netted against gross emissions
        #    depending on the reporting framework.
        # ------------------------------------------------------------------
        self.recycling_avoided_emissions_co2e = _safe_create(Counter,
            'gl_wg_recycling_avoided_emissions_co2e',
            'Total avoided emissions from recycling in tCO2e',
            ['recycling_type', 'waste_category', 'tenant_id']
        )

        # ------------------------------------------------------------------
        # 9. gl_wg_wastewater_organic_load_kg (Counter)
        #    Total wastewater organic load processed in kilograms.
        #    Labels:
        #      - measurement_basis: BOD, COD, TOC, TSS, volumetric
        #      - treatment_system: aerobic_centralized, anaerobic_reactor,
        #                          anaerobic_lagoon, septic_system, etc.
        #      - tenant_id: Tenant identifier
        #    Tracks the organic load of wastewater processed through
        #    treatment calculations. Wastewater CH4 emissions are
        #    proportional to the organic load (BOD or COD) and the
        #    methane correction factor (MCF) for the treatment system.
        #    N2O emissions depend on nitrogen content and the treatment
        #    pathway (nitrification/denitrification efficiency).
        # ------------------------------------------------------------------
        self.wastewater_organic_load_kg = _safe_create(Counter,
            'gl_wg_wastewater_organic_load_kg',
            'Total wastewater organic load processed in kg',
            ['measurement_basis', 'treatment_system', 'tenant_id']
        )

        # ------------------------------------------------------------------
        # 10. gl_wg_diversion_rate (Gauge)
        #     Waste diversion rate (0.0-1.0) by facility.
        #     Labels:
        #       - facility_id: Facility identifier
        #       - tenant_id: Tenant identifier
        #     Tracks the proportion of waste diverted from landfill
        #     through recycling, composting, anaerobic digestion, and
        #     reuse. Diversion rate = (total waste - landfill waste) /
        #     total waste. Key performance indicator for zero-waste
        #     targets and circular economy reporting.
        # ------------------------------------------------------------------
        self.diversion_rate = _safe_create(Gauge,
            'gl_wg_diversion_rate',
            'Waste diversion rate (0.0-1.0) by facility',
            ['facility_id', 'tenant_id']
        )

        # ------------------------------------------------------------------
        # 11. gl_wg_compliance_checks_total (Counter)
        #     Total compliance checks performed for waste generated.
        #     Labels:
        #       - framework: ghg_protocol, iso_14064, csrd, cdp, sbti,
        #                    eu_waste_framework, epa_warm, ipcc_2006, etc.
        #       - result: compliant, partially_compliant, non_compliant,
        #                 warning, not_applicable, needs_review
        #       - tenant_id: Tenant identifier
        #     Tracks regulatory compliance validation for Scope 3 Cat 5
        #     reporting across GHG Protocol, ISO, CSRD, CDP, and other
        #     waste-specific standards.
        # ------------------------------------------------------------------
        self.compliance_checks_total = _safe_create(Counter,
            'gl_wg_compliance_checks_total',
            'Total waste generated compliance checks performed',
            ['framework', 'result', 'tenant_id']
        )

        # ------------------------------------------------------------------
        # 12. gl_wg_data_quality_score (Gauge)
        #     Data quality score (0.0-5.0) by data source.
        #     Labels:
        #       - data_source: direct_measurement, waste_manifest,
        #                      waste_audit, supplier_report, estimation, etc.
        #       - tenant_id: Tenant identifier
        #     Tracks the quality tier of input data used in waste
        #     calculations. GHG Protocol requires data quality assessment
        #     for Scope 3 reporting. Score follows the 1-5 scale:
        #       5 = Direct measurement (highest accuracy)
        #       4 = Waste manifest / audit data
        #       3 = Supplier-reported data
        #       2 = Estimation / purchase records
        #       1 = Industry average / proxy data (lowest accuracy)
        # ------------------------------------------------------------------
        self.data_quality_score = _safe_create(Gauge,
            'gl_wg_data_quality_score',
            'Data quality score (0.0-5.0) by data source',
            ['data_source', 'tenant_id']
        )

        # ------------------------------------------------------------------
        # 13. gl_wg_batch_size (Histogram)
        #     Size of batch calculation operations (number of waste records).
        #     Labels:
        #       - method: Calculation method used for the batch
        #     Buckets cover typical batch sizes from single waste stream
        #     to large-scale facility portfolio calculations.
        #     Enables monitoring of batch size distribution for capacity
        #     planning and performance tuning.
        # ------------------------------------------------------------------
        self.batch_size = _safe_create(Histogram,
            'gl_wg_batch_size',
            'Batch calculation size for waste generated operations',
            ['method'],
            buckets=[
                1, 5, 10, 25, 50, 100, 250, 500
            ]
        )

        # ------------------------------------------------------------------
        # 14. gl_wg_ef_lookups_total (Counter)
        #     Total emission factor lookups performed.
        #     Labels:
        #       - source: ipcc_2006, ipcc_2019, epa_warm, defra,
        #                 ecoinvent, exiobase, ghg_protocol, supplier, etc.
        #       - waste_category: Category for which the EF was looked up
        #     Tracks the frequency and source distribution of emission
        #     factor retrievals, enabling cache hit ratio optimization
        #     and database coverage monitoring.
        # ------------------------------------------------------------------
        self.ef_lookups_total = _safe_create(Counter,
            'gl_wg_ef_lookups_total',
            'Total emission factor lookups for waste generated',
            ['source', 'waste_category']
        )

    # ======================================================================
    # Primary recording methods
    # ======================================================================

    def record_calculation(
        self,
        method: str,
        treatment: str,
        waste_category: str,
        tenant_id: str,
        status: str = "success",
        emissions_tco2e: Optional[float] = None,
        waste_mass_tonnes: Optional[float] = None,
        duration_s: Optional[float] = None
    ) -> None:
        """
        Record a waste generated emission calculation operation.

        This is the primary recording method that increments the calculations
        counter, optionally observes the duration histogram, and optionally
        tracks emissions and waste mass output. It covers all calculation
        methods and treatment pathways for Scope 3 Category 5.

        Args:
            method: Calculation method (waste_type_specific/treatment_specific/
                     supplier_specific/average_data/ipcc_fod)
            treatment: Treatment pathway (landfill/incineration/recycling/
                        composting/anaerobic_digestion/wastewater_treatment/
                        open_burning/other)
            waste_category: Waste category (municipal_solid/food_waste/
                             paper_cardboard/plastic/hazardous/etc)
            tenant_id: Tenant identifier for multi-tenant isolation
            status: Calculation status (success/error/partial/
                     insufficient_data/skipped)
            emissions_tco2e: Emissions calculated in tCO2e (optional)
            waste_mass_tonnes: Waste mass processed in tonnes (optional)
            duration_s: Operation duration in seconds (optional)

        Example:
            >>> metrics.record_calculation(
            ...     method="waste_type_specific",
            ...     treatment="landfill",
            ...     waste_category="municipal_solid",
            ...     tenant_id="tenant-001",
            ...     status="success",
            ...     emissions_tco2e=8.45,
            ...     waste_mass_tonnes=25.0,
            ...     duration_s=0.35
            ... )
        """
        try:
            # Validate and normalize inputs
            method = self._validate_enum_value(
                method, CalculationMethod, CalculationMethod.WASTE_TYPE_SPECIFIC.value
            )
            treatment = self._validate_enum_value(
                treatment, TreatmentMethod, TreatmentMethod.OTHER.value
            )
            waste_category = self._validate_enum_value(
                waste_category, WasteCategory, WasteCategory.OTHER.value
            )
            status = self._validate_enum_value(
                status, CalculationStatus, CalculationStatus.ERROR.value
            )

            # Sanitize tenant_id: truncate long values to prevent label explosion
            if tenant_id is None:
                tenant_id = "unknown"
            elif len(tenant_id) > 64:
                tenant_id = tenant_id[:64]

            # 1. Increment calculation counter
            self.calculations_total.labels(
                method=method,
                treatment=treatment,
                waste_category=waste_category,
                tenant_id=tenant_id
            ).inc()

            # 2. Observe duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    method=method,
                    treatment=treatment
                ).observe(duration_s)

            # 3. Record emissions if provided
            if emissions_tco2e is not None and emissions_tco2e > 0:
                self.emissions_co2e_tonnes.labels(
                    treatment=treatment,
                    waste_category=waste_category,
                    tenant_id=tenant_id
                ).inc(emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += emissions_tco2e

            # 4. Record waste mass if provided
            if waste_mass_tonnes is not None and waste_mass_tonnes > 0:
                self.waste_mass_tonnes.labels(
                    waste_category=waste_category,
                    treatment=treatment,
                    tenant_id=tenant_id
                ).inc(waste_mass_tonnes)

                with self._stats_lock:
                    self._in_memory_stats['waste_mass_tonnes'] += waste_mass_tonnes

            # 5. Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['calculations'] += 1

            logger.debug(
                "Recorded calculation: method=%s, treatment=%s, "
                "waste_category=%s, tenant=%s, status=%s, "
                "duration=%.3fs, emissions=%.4f tCO2e, mass=%.2f t",
                method, treatment, waste_category, tenant_id, status,
                duration_s if duration_s else 0.0,
                emissions_tco2e if emissions_tco2e else 0.0,
                waste_mass_tonnes if waste_mass_tonnes else 0.0
            )

        except Exception as e:
            logger.error("Failed to record calculation metrics: %s", e, exc_info=True)

    def record_calculation_error(
        self,
        error_type: str,
        treatment: str,
        tenant_id: str,
        operation: Optional[str] = None,
        details: Optional[str] = None
    ) -> None:
        """
        Record a calculation error occurrence.

        Tracks error frequency by type and treatment for diagnostics,
        alerting, and root cause analysis. Error types specific to Scope 3
        Category 5 include emission factor unavailable, waste type unknown,
        treatment method unknown, FOD model error, composition invalid,
        DOC calculation error, and energy recovery error.

        Args:
            error_type: Type of error (validation_error/calculation_error/
                         emission_factor_unavailable/fod_model_error/
                         waste_type_unknown/treatment_method_unknown/etc)
            treatment: Treatment method where the error occurred
            tenant_id: Tenant identifier
            operation: Operation where the error occurred (optional, for logging)
            details: Additional error details for logging (optional)

        Example:
            >>> metrics.record_calculation_error(
            ...     error_type="emission_factor_unavailable",
            ...     treatment="landfill",
            ...     tenant_id="tenant-001",
            ...     operation="calculate_landfill_ch4",
            ...     details="No EF for waste_type=rubber_leather in IPCC 2006"
            ... )
        """
        try:
            # Validate and normalize inputs
            error_type = self._validate_enum_value(
                error_type, ErrorType, ErrorType.VALIDATION_ERROR.value
            )
            treatment = self._validate_enum_value(
                treatment, TreatmentMethod, TreatmentMethod.OTHER.value
            )

            # Sanitize tenant_id
            if tenant_id is None:
                tenant_id = "unknown"
            elif len(tenant_id) > 64:
                tenant_id = tenant_id[:64]

            # Increment error counter
            self.calculation_errors_total.labels(
                error_type=error_type,
                treatment=treatment,
                tenant_id=tenant_id
            ).inc()

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['calculation_errors'] += 1
                self._in_memory_stats['errors'] += 1

            logger.debug(
                "Recorded calculation error: type=%s, treatment=%s, "
                "tenant=%s, operation=%s, details=%s",
                error_type, treatment, tenant_id,
                operation if operation else "N/A",
                details if details else "N/A"
            )

        except Exception as e:
            logger.error(
                "Failed to record calculation error metrics: %s",
                e, exc_info=True
            )

    def record_emissions(
        self,
        treatment: str,
        waste_category: str,
        tenant_id: str,
        emissions_tco2e: float,
        waste_mass_tonnes: Optional[float] = None,
        method: Optional[str] = None
    ) -> None:
        """
        Record emissions output from a waste treatment calculation.

        Standalone emissions recording method for cases where the caller
        wants to record emissions separately from the primary calculation
        counter (e.g., when updating emissions after a recalculation or
        correction).

        Args:
            treatment: Treatment pathway (landfill/incineration/recycling/etc)
            waste_category: Waste category (municipal_solid/food_waste/etc)
            tenant_id: Tenant identifier
            emissions_tco2e: Emissions in tCO2e (must be positive)
            waste_mass_tonnes: Waste mass in tonnes (optional)
            method: Calculation method used (optional, for logging)

        Example:
            >>> metrics.record_emissions(
            ...     treatment="incineration",
            ...     waste_category="plastic",
            ...     tenant_id="tenant-001",
            ...     emissions_tco2e=3.21,
            ...     waste_mass_tonnes=1.5
            ... )
        """
        try:
            # Validate and normalize inputs
            treatment = self._validate_enum_value(
                treatment, TreatmentMethod, TreatmentMethod.OTHER.value
            )
            waste_category = self._validate_enum_value(
                waste_category, WasteCategory, WasteCategory.OTHER.value
            )

            # Sanitize tenant_id
            if tenant_id is None:
                tenant_id = "unknown"
            elif len(tenant_id) > 64:
                tenant_id = tenant_id[:64]

            # Record emissions
            if emissions_tco2e is not None and emissions_tco2e > 0:
                self.emissions_co2e_tonnes.labels(
                    treatment=treatment,
                    waste_category=waste_category,
                    tenant_id=tenant_id
                ).inc(emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += emissions_tco2e

            # Record waste mass if provided
            if waste_mass_tonnes is not None and waste_mass_tonnes > 0:
                self.waste_mass_tonnes.labels(
                    waste_category=waste_category,
                    treatment=treatment,
                    tenant_id=tenant_id
                ).inc(waste_mass_tonnes)

                with self._stats_lock:
                    self._in_memory_stats['waste_mass_tonnes'] += waste_mass_tonnes

            logger.debug(
                "Recorded emissions: treatment=%s, category=%s, "
                "tenant=%s, emissions=%.4f tCO2e, mass=%.2f t, method=%s",
                treatment, waste_category, tenant_id,
                emissions_tco2e if emissions_tco2e else 0.0,
                waste_mass_tonnes if waste_mass_tonnes else 0.0,
                method if method else "N/A"
            )

        except Exception as e:
            logger.error("Failed to record emissions metrics: %s", e, exc_info=True)

    def record_waste_mass(
        self,
        waste_category: str,
        treatment: str,
        tenant_id: str,
        mass_tonnes: float,
        facility_id: Optional[str] = None
    ) -> None:
        """
        Record waste mass processed through a treatment pathway.

        Standalone waste mass recording for tracking throughput by
        category and treatment. Useful for diversion rate calculations
        and waste composition analysis.

        Args:
            waste_category: Waste category being processed
            treatment: Treatment pathway for the waste
            tenant_id: Tenant identifier
            mass_tonnes: Waste mass in metric tonnes (must be positive)
            facility_id: Facility identifier (optional, for logging)

        Example:
            >>> metrics.record_waste_mass(
            ...     waste_category="food_waste",
            ...     treatment="composting",
            ...     tenant_id="tenant-001",
            ...     mass_tonnes=15.0,
            ...     facility_id="facility-west-01"
            ... )
        """
        try:
            # Validate and normalize inputs
            waste_category = self._validate_enum_value(
                waste_category, WasteCategory, WasteCategory.OTHER.value
            )
            treatment = self._validate_enum_value(
                treatment, TreatmentMethod, TreatmentMethod.OTHER.value
            )

            # Sanitize tenant_id
            if tenant_id is None:
                tenant_id = "unknown"
            elif len(tenant_id) > 64:
                tenant_id = tenant_id[:64]

            # Record waste mass
            if mass_tonnes is not None and mass_tonnes > 0:
                self.waste_mass_tonnes.labels(
                    waste_category=waste_category,
                    treatment=treatment,
                    tenant_id=tenant_id
                ).inc(mass_tonnes)

                with self._stats_lock:
                    self._in_memory_stats['waste_mass_tonnes'] += mass_tonnes

            logger.debug(
                "Recorded waste mass: category=%s, treatment=%s, "
                "tenant=%s, mass=%.2f t, facility=%s",
                waste_category, treatment, tenant_id,
                mass_tonnes if mass_tonnes else 0.0,
                facility_id if facility_id else "N/A"
            )

        except Exception as e:
            logger.error("Failed to record waste mass metrics: %s", e, exc_info=True)

    def record_landfill_ch4(
        self,
        waste_category: str,
        climate_zone: str,
        tenant_id: str,
        ch4_kg: float,
        doc_fraction: Optional[float] = None,
        mcf: Optional[float] = None,
        oxidation_factor: Optional[float] = None,
        decay_rate_k: Optional[float] = None,
        duration_s: Optional[float] = None
    ) -> None:
        """
        Record landfill methane generation from waste decomposition.

        Tracks CH4 generated from landfill using the IPCC First Order
        Decay (FOD) model or simplified methods. Landfill emissions are
        the dominant source of Scope 3 Category 5 GHG emissions for
        most reporting companies.

        FOD Model Parameters (for logging/diagnostics):
            CH4 generated = DDOCm x F x (16/12)
            where DDOCm = W x DOC x DOCf x MCF
            - W: waste mass deposited
            - DOC: degradable organic carbon fraction
            - DOCf: fraction of DOC that decomposes (0.5 default)
            - MCF: methane correction factor (depends on management)
            - F: fraction of CH4 in landfill gas (0.5 default)

        Args:
            waste_category: Category of waste landfilled
            climate_zone: IPCC climate zone (affects decay rate k)
            tenant_id: Tenant identifier
            ch4_kg: Methane generated in kilograms
            doc_fraction: Degradable organic carbon fraction (optional)
            mcf: Methane correction factor (optional)
            oxidation_factor: Oxidation factor for CH4 cover (optional)
            decay_rate_k: First-order decay rate constant (optional)
            duration_s: Calculation duration in seconds (optional)

        Example:
            >>> metrics.record_landfill_ch4(
            ...     waste_category="food_waste",
            ...     climate_zone="temperate_wet",
            ...     tenant_id="tenant-001",
            ...     ch4_kg=245.8,
            ...     doc_fraction=0.15,
            ...     mcf=1.0,
            ...     oxidation_factor=0.1,
            ...     decay_rate_k=0.185
            ... )
        """
        try:
            # Validate and normalize inputs
            waste_category = self._validate_enum_value(
                waste_category, WasteCategory, WasteCategory.OTHER.value
            )
            climate_zone = self._validate_enum_value(
                climate_zone, ClimateZone, ClimateZone.DEFAULT.value
            )

            # Sanitize tenant_id
            if tenant_id is None:
                tenant_id = "unknown"
            elif len(tenant_id) > 64:
                tenant_id = tenant_id[:64]

            # Record CH4 generation
            if ch4_kg is not None and ch4_kg > 0:
                self.landfill_ch4_generated_kg.labels(
                    waste_category=waste_category,
                    climate_zone=climate_zone,
                    tenant_id=tenant_id
                ).inc(ch4_kg)

                with self._stats_lock:
                    self._in_memory_stats['landfill_ch4_kg'] += ch4_kg

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    method=CalculationMethod.IPCC_FOD.value,
                    treatment=TreatmentMethod.LANDFILL.value
                ).observe(duration_s)

            logger.debug(
                "Recorded landfill CH4: category=%s, zone=%s, "
                "tenant=%s, ch4=%.2f kg, DOC=%.3f, MCF=%.2f, "
                "OX=%.2f, k=%.4f",
                waste_category, climate_zone, tenant_id,
                ch4_kg if ch4_kg else 0.0,
                doc_fraction if doc_fraction else 0.0,
                mcf if mcf else 0.0,
                oxidation_factor if oxidation_factor else 0.0,
                decay_rate_k if decay_rate_k else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record landfill CH4 metrics: %s",
                e, exc_info=True
            )

    def record_incineration_energy(
        self,
        incinerator_type: str,
        tenant_id: str,
        energy_mwh: float,
        waste_mass_tonnes: Optional[float] = None,
        efficiency: Optional[float] = None,
        waste_category: Optional[str] = None,
        duration_s: Optional[float] = None
    ) -> None:
        """
        Record energy recovered from waste incineration.

        Tracks energy recovery from waste-to-energy incineration
        facilities. Energy recovery reduces net emissions by displacing
        grid electricity or heat production.

        Net incineration emissions = Gross emissions - Avoided grid emissions
        Avoided emissions = Energy recovered (MWh) x Grid emission factor

        Args:
            incinerator_type: Type of incinerator (mass_burn/fluidized_bed/
                               rdf/rotary_kiln/cement_kiln/etc)
            tenant_id: Tenant identifier
            energy_mwh: Energy recovered in megawatt-hours
            waste_mass_tonnes: Waste mass incinerated in tonnes (optional)
            efficiency: Energy recovery efficiency 0.0-1.0 (optional)
            waste_category: Category of waste incinerated (optional)
            duration_s: Calculation duration in seconds (optional)

        Example:
            >>> metrics.record_incineration_energy(
            ...     incinerator_type="mass_burn",
            ...     tenant_id="tenant-001",
            ...     energy_mwh=12.5,
            ...     waste_mass_tonnes=50.0,
            ...     efficiency=0.25
            ... )
        """
        try:
            # Validate and normalize inputs
            incinerator_type = self._validate_enum_value(
                incinerator_type, IncineratorType, IncineratorType.OTHER.value
            )

            # Sanitize tenant_id
            if tenant_id is None:
                tenant_id = "unknown"
            elif len(tenant_id) > 64:
                tenant_id = tenant_id[:64]

            # Record energy recovery
            if energy_mwh is not None and energy_mwh > 0:
                self.incineration_energy_recovered_mwh.labels(
                    incinerator_type=incinerator_type,
                    tenant_id=tenant_id
                ).inc(energy_mwh)

                with self._stats_lock:
                    self._in_memory_stats['incineration_energy_mwh'] += energy_mwh

            # Record waste mass if provided
            if waste_mass_tonnes is not None and waste_mass_tonnes > 0:
                wc = self._validate_enum_value(
                    waste_category, WasteCategory, WasteCategory.OTHER.value
                ) if waste_category else WasteCategory.OTHER.value

                self.waste_mass_tonnes.labels(
                    waste_category=wc,
                    treatment=TreatmentMethod.INCINERATION.value,
                    tenant_id=tenant_id
                ).inc(waste_mass_tonnes)

                with self._stats_lock:
                    self._in_memory_stats['waste_mass_tonnes'] += waste_mass_tonnes

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    method=CalculationMethod.WASTE_TYPE_SPECIFIC.value,
                    treatment=TreatmentMethod.INCINERATION.value
                ).observe(duration_s)

            logger.debug(
                "Recorded incineration energy: type=%s, tenant=%s, "
                "energy=%.2f MWh, mass=%.2f t, efficiency=%.2f",
                incinerator_type, tenant_id,
                energy_mwh if energy_mwh else 0.0,
                waste_mass_tonnes if waste_mass_tonnes else 0.0,
                efficiency if efficiency else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record incineration energy metrics: %s",
                e, exc_info=True
            )

    def record_recycling_avoided_emissions(
        self,
        recycling_type: str,
        waste_category: str,
        tenant_id: str,
        avoided_tco2e: float,
        waste_mass_tonnes: Optional[float] = None,
        virgin_ef: Optional[float] = None,
        recycled_ef: Optional[float] = None,
        duration_s: Optional[float] = None
    ) -> None:
        """
        Record avoided emissions from waste recycling.

        Tracks emission credits from recycling that displaces virgin
        material production. Avoided emissions represent the difference
        between producing material from virgin feedstock vs. from
        recycled content.

        Avoided emissions = mass recycled x (virgin production EF - recycling EF)

        Note: Avoided emissions may be reported separately as an
        informational metric or netted against gross emissions depending
        on the reporting framework (GHG Protocol recommends separate
        reporting).

        Args:
            recycling_type: Type of recycling (mechanical/chemical/feedstock/
                             closed_loop/open_loop/downcycling/upcycling)
            waste_category: Category of waste recycled
            tenant_id: Tenant identifier
            avoided_tco2e: Avoided emissions in tCO2e (must be positive)
            waste_mass_tonnes: Mass of waste recycled in tonnes (optional)
            virgin_ef: Virgin production emission factor (optional, for logging)
            recycled_ef: Recycled production emission factor (optional, for logging)
            duration_s: Calculation duration in seconds (optional)

        Example:
            >>> metrics.record_recycling_avoided_emissions(
            ...     recycling_type="mechanical",
            ...     waste_category="plastic",
            ...     tenant_id="tenant-001",
            ...     avoided_tco2e=5.12,
            ...     waste_mass_tonnes=3.0,
            ...     virgin_ef=2.5,
            ...     recycled_ef=0.79
            ... )
        """
        try:
            # Validate and normalize inputs
            recycling_type = self._validate_enum_value(
                recycling_type, RecyclingType, RecyclingType.OTHER.value
            )
            waste_category = self._validate_enum_value(
                waste_category, WasteCategory, WasteCategory.OTHER.value
            )

            # Sanitize tenant_id
            if tenant_id is None:
                tenant_id = "unknown"
            elif len(tenant_id) > 64:
                tenant_id = tenant_id[:64]

            # Record avoided emissions
            if avoided_tco2e is not None and avoided_tco2e > 0:
                self.recycling_avoided_emissions_co2e.labels(
                    recycling_type=recycling_type,
                    waste_category=waste_category,
                    tenant_id=tenant_id
                ).inc(avoided_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['recycling_avoided_co2e'] += avoided_tco2e

            # Record waste mass if provided
            if waste_mass_tonnes is not None and waste_mass_tonnes > 0:
                self.waste_mass_tonnes.labels(
                    waste_category=waste_category,
                    treatment=TreatmentMethod.RECYCLING.value,
                    tenant_id=tenant_id
                ).inc(waste_mass_tonnes)

                with self._stats_lock:
                    self._in_memory_stats['waste_mass_tonnes'] += waste_mass_tonnes

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    method=CalculationMethod.WASTE_TYPE_SPECIFIC.value,
                    treatment=TreatmentMethod.RECYCLING.value
                ).observe(duration_s)

            logger.debug(
                "Recorded recycling avoided emissions: type=%s, "
                "category=%s, tenant=%s, avoided=%.4f tCO2e, "
                "mass=%.2f t, virgin_ef=%.4f, recycled_ef=%.4f",
                recycling_type, waste_category, tenant_id,
                avoided_tco2e if avoided_tco2e else 0.0,
                waste_mass_tonnes if waste_mass_tonnes else 0.0,
                virgin_ef if virgin_ef else 0.0,
                recycled_ef if recycled_ef else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record recycling avoided emissions metrics: %s",
                e, exc_info=True
            )

    def record_wastewater_organic_load(
        self,
        measurement_basis: str,
        treatment_system: str,
        tenant_id: str,
        organic_load_kg: float,
        emissions_tco2e: Optional[float] = None,
        ch4_kg: Optional[float] = None,
        n2o_kg: Optional[float] = None,
        duration_s: Optional[float] = None
    ) -> None:
        """
        Record wastewater organic load processed through treatment.

        Tracks the organic load of wastewater processed, which is the
        primary driver for wastewater CH4 and N2O emissions. Wastewater
        CH4 emissions are proportional to the organic load (BOD or COD)
        and the methane correction factor (MCF) for the treatment system.

        IPCC Wastewater CH4 Formula:
            CH4 = TOW x EF - R
            where TOW = total organics in wastewater (kg BOD or COD)
                  EF = emission factor (kg CH4/kg BOD or COD) = Bo x MCF
                  R = CH4 recovered (flaring, energy recovery)

        Args:
            measurement_basis: Organic load measurement (bod/cod/toc/tss/volumetric)
            treatment_system: Treatment system type (aerobic_centralized/
                               anaerobic_reactor/anaerobic_lagoon/septic_system/etc)
            tenant_id: Tenant identifier
            organic_load_kg: Organic load in kilograms (BOD/COD)
            emissions_tco2e: Total emissions in tCO2e (optional)
            ch4_kg: CH4 generated in kg (optional)
            n2o_kg: N2O generated in kg (optional)
            duration_s: Calculation duration in seconds (optional)

        Example:
            >>> metrics.record_wastewater_organic_load(
            ...     measurement_basis="bod",
            ...     treatment_system="anaerobic_lagoon",
            ...     tenant_id="tenant-001",
            ...     organic_load_kg=5000.0,
            ...     emissions_tco2e=2.15,
            ...     ch4_kg=85.0,
            ...     n2o_kg=0.12
            ... )
        """
        try:
            # Validate and normalize inputs
            measurement_basis = self._validate_enum_value(
                measurement_basis, MeasurementBasis, MeasurementBasis.OTHER.value
            )
            treatment_system = self._validate_enum_value(
                treatment_system, WastewaterTreatmentSystem,
                WastewaterTreatmentSystem.OTHER.value
            )

            # Sanitize tenant_id
            if tenant_id is None:
                tenant_id = "unknown"
            elif len(tenant_id) > 64:
                tenant_id = tenant_id[:64]

            # Record organic load
            if organic_load_kg is not None and organic_load_kg > 0:
                self.wastewater_organic_load_kg.labels(
                    measurement_basis=measurement_basis,
                    treatment_system=treatment_system,
                    tenant_id=tenant_id
                ).inc(organic_load_kg)

                with self._stats_lock:
                    self._in_memory_stats['wastewater_organic_load_kg'] += organic_load_kg

            # Record associated emissions if provided
            if emissions_tco2e is not None and emissions_tco2e > 0:
                self.emissions_co2e_tonnes.labels(
                    treatment=TreatmentMethod.WASTEWATER_TREATMENT.value,
                    waste_category=WasteCategory.SLUDGE.value,
                    tenant_id=tenant_id
                ).inc(emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += emissions_tco2e

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    method=CalculationMethod.TREATMENT_SPECIFIC.value,
                    treatment=TreatmentMethod.WASTEWATER_TREATMENT.value
                ).observe(duration_s)

            logger.debug(
                "Recorded wastewater organic load: basis=%s, system=%s, "
                "tenant=%s, load=%.2f kg, emissions=%.4f tCO2e, "
                "ch4=%.2f kg, n2o=%.4f kg",
                measurement_basis, treatment_system, tenant_id,
                organic_load_kg if organic_load_kg else 0.0,
                emissions_tco2e if emissions_tco2e else 0.0,
                ch4_kg if ch4_kg else 0.0,
                n2o_kg if n2o_kg else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record wastewater organic load metrics: %s",
                e, exc_info=True
            )

    def record_compliance_check(
        self,
        framework: str,
        result: str,
        tenant_id: str,
        duration_s: Optional[float] = None,
        details: Optional[str] = None
    ) -> None:
        """
        Record a compliance check operation for waste generated reporting.

        Tracks validation of Scope 3 Category 5 calculations against
        regulatory and voluntary reporting frameworks including GHG Protocol,
        ISO 14064, CSRD, CDP, SBTi, EU Waste Framework Directive, EPA WARM,
        and IPCC Guidelines.

        Args:
            framework: Compliance framework (ghg_protocol/iso_14064/csrd/
                        cdp/sbti/eu_waste_framework/epa_warm/ipcc_2006/etc)
            result: Compliance check result (compliant/partially_compliant/
                     non_compliant/warning/not_applicable/needs_review)
            tenant_id: Tenant identifier
            duration_s: Check duration in seconds (optional)
            details: Additional compliance details for logging (optional)

        Example:
            >>> metrics.record_compliance_check(
            ...     framework="ghg_protocol",
            ...     result="compliant",
            ...     tenant_id="tenant-001",
            ...     duration_s=0.08,
            ...     details="All Category 5 treatment methods validated"
            ... )
        """
        try:
            # Validate and normalize inputs
            framework = self._validate_enum_value(
                framework, Framework, Framework.GHG_PROTOCOL.value
            )
            result = self._validate_enum_value(
                result, ComplianceStatus, ComplianceStatus.NOT_APPLICABLE.value
            )

            # Sanitize tenant_id
            if tenant_id is None:
                tenant_id = "unknown"
            elif len(tenant_id) > 64:
                tenant_id = tenant_id[:64]

            # Increment compliance check counter
            self.compliance_checks_total.labels(
                framework=framework,
                result=result,
                tenant_id=tenant_id
            ).inc()

            # Record check duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    method=CalculationMethod.WASTE_TYPE_SPECIFIC.value,
                    treatment=TreatmentMethod.OTHER.value
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['compliance_checks'] += 1

            logger.debug(
                "Recorded compliance check: framework=%s, result=%s, "
                "tenant=%s, details=%s",
                framework, result, tenant_id,
                details if details else "N/A"
            )

        except Exception as e:
            logger.error(
                "Failed to record compliance check metrics: %s",
                e, exc_info=True
            )

    def record_ef_lookup(
        self,
        source: str,
        waste_category: str,
        count: int = 1,
        duration_s: Optional[float] = None
    ) -> None:
        """
        Record an emission factor database lookup operation.

        Tracks lookups to waste-specific emission factor databases
        including IPCC, EPA WARM, DEFRA, ecoinvent, EXIOBASE, and
        supplier-specific factor databases.

        Args:
            source: Emission factor source (ipcc_2006/ipcc_2019/epa_warm/
                     defra/ecoinvent/exiobase/ghg_protocol/supplier_specific/etc)
            waste_category: Waste category for which the EF was looked up
            count: Number of lookups in this operation (default: 1)
            duration_s: Lookup duration in seconds (optional)

        Example:
            >>> metrics.record_ef_lookup(
            ...     source="ipcc_2006",
            ...     waste_category="food_waste",
            ...     count=1,
            ...     duration_s=0.003
            ... )
        """
        try:
            # Validate and normalize inputs
            source = self._validate_enum_value(
                source, EFSource, EFSource.OTHER.value
            )
            waste_category = self._validate_enum_value(
                waste_category, WasteCategory, WasteCategory.OTHER.value
            )

            # Clamp count to non-negative
            count = max(0, count)

            # Increment EF lookup counter
            self.ef_lookups_total.labels(
                source=source,
                waste_category=waste_category
            ).inc(count)

            # Record lookup duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    method=CalculationMethod.WASTE_TYPE_SPECIFIC.value,
                    treatment=TreatmentMethod.OTHER.value
                ).observe(duration_s)

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['ef_lookups'] += count

            logger.debug(
                "Recorded EF lookup: source=%s, category=%s, count=%d",
                source, waste_category, count
            )

        except Exception as e:
            logger.error(
                "Failed to record EF lookup metrics: %s",
                e, exc_info=True
            )

    def record_batch(
        self,
        method: str,
        size: int,
        successful: Optional[int] = None,
        failed: Optional[int] = None,
        duration_s: Optional[float] = None,
        total_emissions_tco2e: Optional[float] = None,
        total_waste_mass_tonnes: Optional[float] = None,
        tenant_id: Optional[str] = None
    ) -> None:
        """
        Record a batch calculation operation.

        Batch operations process multiple waste stream records in a
        single operation, typically for facility-level calculations
        covering a full reporting period or multi-site portfolio.

        Args:
            method: Calculation method used for the batch
            size: Number of waste records in the batch
            successful: Number of successful calculations (optional)
            failed: Number of failed calculations (optional)
            duration_s: Total batch duration in seconds (optional)
            total_emissions_tco2e: Total emissions for the batch in tCO2e (optional)
            total_waste_mass_tonnes: Total waste mass in tonnes (optional)
            tenant_id: Tenant identifier (optional)

        Example:
            >>> metrics.record_batch(
            ...     method="waste_type_specific",
            ...     size=150,
            ...     successful=148,
            ...     failed=2,
            ...     duration_s=8.5,
            ...     total_emissions_tco2e=425.0,
            ...     total_waste_mass_tonnes=1200.0
            ... )
        """
        try:
            # Validate and normalize method
            method = self._validate_enum_value(
                method, CalculationMethod, CalculationMethod.WASTE_TYPE_SPECIFIC.value
            )

            # Clamp size to non-negative
            size = max(0, size)

            # Observe batch size in histogram
            if size > 0:
                self.batch_size.labels(
                    method=method
                ).observe(size)

            # Record duration if provided
            if duration_s is not None and duration_s > 0:
                self.calculation_duration_seconds.labels(
                    method=method,
                    treatment=TreatmentMethod.OTHER.value
                ).observe(duration_s)

            # Record successful calculations in primary counter
            if successful is not None and successful > 0:
                tenant_id_val = tenant_id or "unknown"
                if len(tenant_id_val) > 64:
                    tenant_id_val = tenant_id_val[:64]
                self.calculations_total.labels(
                    method=method,
                    treatment=TreatmentMethod.OTHER.value,
                    waste_category=WasteCategory.MIXED.value,
                    tenant_id=tenant_id_val
                ).inc(successful)

            # Record failed calculations in error counter
            if failed is not None and failed > 0:
                tenant_id_val = tenant_id or "unknown"
                if len(tenant_id_val) > 64:
                    tenant_id_val = tenant_id_val[:64]
                self.calculation_errors_total.labels(
                    error_type=ErrorType.CALCULATION_ERROR.value,
                    treatment=TreatmentMethod.OTHER.value,
                    tenant_id=tenant_id_val
                ).inc(failed)

            # Record aggregate emissions if provided
            if total_emissions_tco2e is not None and total_emissions_tco2e > 0:
                tenant_id_val = tenant_id or "unknown"
                if len(tenant_id_val) > 64:
                    tenant_id_val = tenant_id_val[:64]
                self.emissions_co2e_tonnes.labels(
                    treatment=TreatmentMethod.OTHER.value,
                    waste_category=WasteCategory.MIXED.value,
                    tenant_id=tenant_id_val
                ).inc(total_emissions_tco2e)

                with self._stats_lock:
                    self._in_memory_stats['emissions_tco2e'] += total_emissions_tco2e

            # Record aggregate waste mass if provided
            if total_waste_mass_tonnes is not None and total_waste_mass_tonnes > 0:
                tenant_id_val = tenant_id or "unknown"
                if len(tenant_id_val) > 64:
                    tenant_id_val = tenant_id_val[:64]
                self.waste_mass_tonnes.labels(
                    waste_category=WasteCategory.MIXED.value,
                    treatment=TreatmentMethod.OTHER.value,
                    tenant_id=tenant_id_val
                ).inc(total_waste_mass_tonnes)

                with self._stats_lock:
                    self._in_memory_stats['waste_mass_tonnes'] += total_waste_mass_tonnes

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['batch_jobs'] += 1
                if size > 0:
                    self._in_memory_stats['calculations'] += size

            logger.info(
                "Recorded batch: method=%s, size=%d, successful=%s, "
                "failed=%s, duration=%.2fs, emissions=%.4f tCO2e, "
                "mass=%.2f t",
                method, size,
                successful if successful is not None else "N/A",
                failed if failed is not None else "N/A",
                duration_s if duration_s else 0.0,
                total_emissions_tco2e if total_emissions_tco2e else 0.0,
                total_waste_mass_tonnes if total_waste_mass_tonnes else 0.0
            )

        except Exception as e:
            logger.error(
                "Failed to record batch metrics: %s",
                e, exc_info=True
            )

    def record_error(
        self,
        error_type: str,
        operation: Optional[str] = None,
        treatment: Optional[str] = None,
        method: Optional[str] = None
    ) -> None:
        """
        Record a calculation error occurrence (lightweight).

        Lightweight error recording that increments the in-memory error
        count and logs the event without requiring all calculation
        parameters. For full error tracking with Prometheus labels,
        use record_calculation_error() instead.

        Args:
            error_type: Type of error (validation_error/emission_factor_unavailable/
                         fod_model_error/waste_type_unknown/etc)
            operation: Operation where the error occurred (optional, for logging)
            treatment: Treatment method related to error (optional)
            method: Calculation method related to error (optional)

        Example:
            >>> metrics.record_error(
            ...     error_type="emission_factor_unavailable",
            ...     operation="calculate_landfill_emissions",
            ...     treatment="landfill"
            ... )
        """
        try:
            # Validate error type
            error_type = self._validate_enum_value(
                error_type, ErrorType, ErrorType.VALIDATION_ERROR.value
            )

            # Update in-memory stats
            with self._stats_lock:
                self._in_memory_stats['errors'] += 1

            logger.debug(
                "Recorded error: type=%s, operation=%s, treatment=%s, method=%s",
                error_type,
                operation if operation else "N/A",
                treatment if treatment else "N/A",
                method if method else "N/A"
            )

        except Exception as e:
            logger.error("Failed to record error metrics: %s", e, exc_info=True)

    # ======================================================================
    # Gauge methods (diversion rate, data quality)
    # ======================================================================

    def set_diversion_rate(
        self,
        facility_id: str,
        tenant_id: str,
        rate: float
    ) -> None:
        """
        Set the waste diversion rate gauge for a specific facility.

        Diversion rate = (total waste - landfill waste) / total waste
        Range: 0.0 (all to landfill) to 1.0 (zero waste to landfill)

        Key performance indicator for zero-waste targets and circular
        economy reporting. Many companies set diversion rate targets
        (e.g., 90% diversion by 2030).

        Args:
            facility_id: Facility identifier
            tenant_id: Tenant identifier
            rate: Diversion rate as a decimal (0.0 to 1.0)

        Example:
            >>> metrics.set_diversion_rate(
            ...     facility_id="facility-west-01",
            ...     tenant_id="tenant-001",
            ...     rate=0.82
            ... )
        """
        try:
            # Sanitize facility_id
            if facility_id is None:
                facility_id = "unknown"
            elif len(facility_id) > 64:
                facility_id = facility_id[:64]

            # Sanitize tenant_id
            if tenant_id is None:
                tenant_id = "unknown"
            elif len(tenant_id) > 64:
                tenant_id = tenant_id[:64]

            # Clamp rate to [0.0, 1.0]
            rate = max(0.0, min(1.0, rate))

            self.diversion_rate.labels(
                facility_id=facility_id,
                tenant_id=tenant_id
            ).set(rate)

            logger.debug(
                "Set diversion rate: facility=%s, tenant=%s, rate=%.4f",
                facility_id, tenant_id, rate
            )

        except Exception as e:
            logger.error(
                "Failed to set diversion rate gauge: %s",
                e, exc_info=True
            )

    def set_data_quality_score(
        self,
        data_source: str,
        tenant_id: str,
        score: float
    ) -> None:
        """
        Set the data quality score gauge for a specific data source.

        GHG Protocol data quality scoring (1-5 scale):
            5 = Direct measurement (highest accuracy)
            4 = Waste manifest / audit data
            3 = Supplier-reported data
            2 = Estimation / purchase records
            1 = Industry average / proxy data (lowest accuracy)

        Args:
            data_source: Data source type (direct_measurement/waste_manifest/
                          waste_audit/supplier_report/estimation/etc)
            tenant_id: Tenant identifier
            score: Quality score (0.0 to 5.0)

        Example:
            >>> metrics.set_data_quality_score(
            ...     data_source="waste_manifest",
            ...     tenant_id="tenant-001",
            ...     score=4.0
            ... )
        """
        try:
            # Validate data source
            data_source = self._validate_enum_value(
                data_source, DataSource, DataSource.OTHER.value
            )

            # Sanitize tenant_id
            if tenant_id is None:
                tenant_id = "unknown"
            elif len(tenant_id) > 64:
                tenant_id = tenant_id[:64]

            # Clamp score to [0.0, 5.0]
            score = max(0.0, min(5.0, score))

            self.data_quality_score.labels(
                data_source=data_source,
                tenant_id=tenant_id
            ).set(score)

            logger.debug(
                "Set data quality score: source=%s, tenant=%s, score=%.2f",
                data_source, tenant_id, score
            )

        except Exception as e:
            logger.error(
                "Failed to set data quality score gauge: %s",
                e, exc_info=True
            )

    def inc_active(self, method: str = "waste_type_specific", amount: float = 1) -> None:
        """
        Increment a notional active calculations count for a specific method.

        This is a lightweight way to track concurrent calculations using
        the calculation_duration_seconds histogram by observing zero-duration
        entries. For real active calculation tracking, use the track_calculation
        context manager.

        Note: Unlike MRV-017 which has a dedicated active_calculations gauge,
        MRV-018 tracks active calculations via context managers that adjust
        the in-memory stats and record durations.

        Args:
            method: Calculation method (default: waste_type_specific)
            amount: Amount to increment by (default: 1)

        Example:
            >>> metrics.inc_active("ipcc_fod")
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethod, CalculationMethod.WASTE_TYPE_SPECIFIC.value
            )
            logger.debug("Active calculations incremented: method=%s, amount=%.0f", method, amount)

        except Exception as e:
            logger.error(
                "Failed to increment active calculations: %s",
                e, exc_info=True
            )

    def dec_active(self, method: str = "waste_type_specific", amount: float = 1) -> None:
        """
        Decrement a notional active calculations count for a specific method.

        Args:
            method: Calculation method (default: waste_type_specific)
            amount: Amount to decrement by (default: 1)

        Example:
            >>> metrics.dec_active("ipcc_fod")
        """
        try:
            method = self._validate_enum_value(
                method, CalculationMethod, CalculationMethod.WASTE_TYPE_SPECIFIC.value
            )
            logger.debug("Active calculations decremented: method=%s, amount=%.0f", method, amount)

        except Exception as e:
            logger.error(
                "Failed to decrement active calculations: %s",
                e, exc_info=True
            )

    # ======================================================================
    # Summary and reset
    # ======================================================================

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metrics collected since initialization.

        Returns a dictionary with all in-memory counters, uptime information,
        calculated rates (per-hour throughput), method breakdown, and
        treatment pathway distribution.

        Returns:
            Dictionary with metrics summary including counts, uptime, rates,
            method breakdown, and treatment pathway analysis.

        Example:
            >>> summary = metrics.get_metrics_summary()
            >>> print(summary['calculations'])
            5432
            >>> print(summary['rates']['calculations_per_hour'])
            271.6
            >>> print(summary['emissions_tco2e'])
            12500.0
        """
        try:
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600 if uptime_seconds > 0 else 1.0

            with self._stats_lock:
                stats_snapshot = dict(self._in_memory_stats)

            summary: Dict[str, Any] = {
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'agent': 'waste_generated',
                'agent_id': 'GL-MRV-S3-005',
                'prefix': 'gl_wg_',
                'scope': 'Scope 3 Category 5',
                'description': 'Waste Generated in Operations',
                'metrics_count': 14,
                'uptime_seconds': uptime_seconds,
                'uptime_hours': uptime_seconds / 3600,
                'start_time': self._start_time.isoformat(),
                'current_time': datetime.utcnow().isoformat(),
                **stats_snapshot,
                'rates': {
                    'calculations_per_hour': (
                        stats_snapshot['calculations'] / uptime_hours
                    ),
                    'emissions_tco2e_per_hour': (
                        stats_snapshot['emissions_tco2e'] / uptime_hours
                    ),
                    'waste_mass_tonnes_per_hour': (
                        stats_snapshot['waste_mass_tonnes'] / uptime_hours
                    ),
                    'landfill_ch4_kg_per_hour': (
                        stats_snapshot['landfill_ch4_kg'] / uptime_hours
                    ),
                    'incineration_energy_mwh_per_hour': (
                        stats_snapshot['incineration_energy_mwh'] / uptime_hours
                    ),
                    'recycling_avoided_co2e_per_hour': (
                        stats_snapshot['recycling_avoided_co2e'] / uptime_hours
                    ),
                    'wastewater_organic_load_kg_per_hour': (
                        stats_snapshot['wastewater_organic_load_kg'] / uptime_hours
                    ),
                    'compliance_checks_per_hour': (
                        stats_snapshot['compliance_checks'] / uptime_hours
                    ),
                    'ef_lookups_per_hour': (
                        stats_snapshot['ef_lookups'] / uptime_hours
                    ),
                    'batch_jobs_per_hour': (
                        stats_snapshot['batch_jobs'] / uptime_hours
                    ),
                    'errors_per_hour': (
                        stats_snapshot['errors'] / uptime_hours
                    ),
                },
                'treatment_breakdown': {
                    'landfill': {
                        'ch4_kg': stats_snapshot['landfill_ch4_kg'],
                    },
                    'incineration': {
                        'energy_mwh': stats_snapshot['incineration_energy_mwh'],
                    },
                    'recycling': {
                        'avoided_co2e': stats_snapshot['recycling_avoided_co2e'],
                    },
                    'wastewater': {
                        'organic_load_kg': stats_snapshot['wastewater_organic_load_kg'],
                    },
                },
                'operational': {
                    'compliance_checks': stats_snapshot['compliance_checks'],
                    'ef_lookups': stats_snapshot['ef_lookups'],
                    'batch_jobs': stats_snapshot['batch_jobs'],
                    'calculation_errors': stats_snapshot['calculation_errors'],
                    'errors': stats_snapshot['errors'],
                },
            }

            logger.debug(
                "Generated metrics summary: %d calculations tracked",
                stats_snapshot['calculations']
            )
            return summary

        except Exception as e:
            logger.error("Failed to generate metrics summary: %s", e, exc_info=True)
            return {
                'error': str(e),
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'agent': 'waste_generated',
            }

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance for testing purposes.

        This destroys the existing singleton so that a fresh instance
        will be created on next access. Primarily used in unit tests.

        WARNING: This is NOT safe for concurrent use. It should only
        be called in test teardown when no other threads are accessing
        the metrics instance.

        Example:
            >>> WasteGeneratedMetrics.reset()
            >>> metrics = WasteGeneratedMetrics()  # Fresh instance
        """
        with cls._lock:
            if cls._instance is not None:
                # Clear the initialized flag so __init__ runs again
                if hasattr(cls._instance, '_initialized'):
                    del cls._instance._initialized
                cls._instance = None

                # Also reset the module-level singleton
                global _metrics_instance
                _metrics_instance = None

                logger.info("WasteGeneratedMetrics singleton reset")

    def reset_stats(self) -> None:
        """
        Reset in-memory statistics (not Prometheus metrics).

        Note: This only resets the in-memory counters used for get_metrics_summary().
        Prometheus metrics are cumulative and cannot be reset without restarting
        the process. This method also resets the start_time for rate calculations.

        Example:
            >>> metrics.reset_stats()
            >>> summary = metrics.get_metrics_summary()
            >>> assert summary['calculations'] == 0
        """
        try:
            with self._stats_lock:
                self._in_memory_stats = {
                    'calculations': 0,
                    'calculation_errors': 0,
                    'emissions_tco2e': 0.0,
                    'waste_mass_tonnes': 0.0,
                    'landfill_ch4_kg': 0.0,
                    'incineration_energy_mwh': 0.0,
                    'recycling_avoided_co2e': 0.0,
                    'wastewater_organic_load_kg': 0.0,
                    'compliance_checks': 0,
                    'ef_lookups': 0,
                    'batch_jobs': 0,
                    'errors': 0,
                }
            self._start_time = datetime.utcnow()

            logger.info("Reset in-memory statistics for WasteGeneratedMetrics")

        except Exception as e:
            logger.error("Failed to reset statistics: %s", e, exc_info=True)

    # ======================================================================
    # Internal helpers
    # ======================================================================

    @staticmethod
    def _validate_enum_value(
        value: Optional[str],
        enum_class: type,
        default: str
    ) -> str:
        """
        Validate a string value against an Enum class.

        If the value is None or not a valid member of the enum, logs a
        warning and returns the default. This ensures that label values
        in Prometheus metrics always have bounded cardinality.

        Args:
            value: The string value to validate
            enum_class: The Enum class to validate against
            default: The default value if validation fails

        Returns:
            Validated value or default

        Example:
            >>> WasteGeneratedMetrics._validate_enum_value(
            ...     "landfill", TreatmentMethod, "other"
            ... )
            'landfill'
            >>> WasteGeneratedMetrics._validate_enum_value(
            ...     "invalid_treatment", TreatmentMethod, "other"
            ... )
            'other'
        """
        if value is None:
            return default

        valid_values = [m.value for m in enum_class]
        if value not in valid_values:
            logger.warning(
                "Invalid %s value '%s', using default '%s'",
                enum_class.__name__, value, default
            )
            return default

        return value


# ===========================================================================
# Module-level singleton accessor
# ===========================================================================

_metrics_instance: Optional[WasteGeneratedMetrics] = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics() -> WasteGeneratedMetrics:
    """
    Get the singleton WasteGeneratedMetrics instance.

    Thread-safe accessor for the global metrics instance. Prefer this
    function over direct instantiation for consistency across the
    waste generated agent codebase.

    Returns:
        WasteGeneratedMetrics singleton instance

    Example:
        >>> from greenlang.agents.mrv.waste_generated.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_calculation(
        ...     method="waste_type_specific",
        ...     treatment="landfill",
        ...     waste_category="municipal_solid",
        ...     tenant_id="tenant-001",
        ...     status="success",
        ...     emissions_tco2e=8.45,
        ...     duration_s=0.35
        ... )
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = WasteGeneratedMetrics()

    return _metrics_instance


def reset_metrics() -> None:
    """
    Reset the singleton metrics instance for testing purposes.

    Convenience function that delegates to WasteGeneratedMetrics.reset().
    Should only be called in test teardown.

    Example:
        >>> from greenlang.agents.mrv.waste_generated.metrics import reset_metrics
        >>> reset_metrics()
    """
    WasteGeneratedMetrics.reset()


# ===========================================================================
# Context manager helpers
# ===========================================================================

@contextmanager
def track_calculation(
    method: str = "waste_type_specific",
    treatment: str = "landfill",
    waste_category: str = "municipal_solid",
    tenant_id: str = "unknown"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a calculation's lifecycle.

    Automatically increments/decrements the active calculation tracking
    and records duration when the context exits. The caller can set
    ``context['emissions_tco2e']``, ``context['waste_mass_tonnes']``,
    ``context['status']`` before exiting to record those values.

    Args:
        method: Calculation method being used
        treatment: Waste treatment pathway being calculated
        waste_category: Waste category being calculated
        tenant_id: Tenant identifier

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_calculation("waste_type_specific", "landfill",
        ...                        "food_waste", "tenant-001") as ctx:
        ...     result = perform_landfill_calculation()
        ...     ctx['emissions_tco2e'] = result.total_tco2e
        ...     ctx['waste_mass_tonnes'] = result.mass_tonnes
        ...     ctx['status'] = "success"
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'method': method,
        'treatment': treatment,
        'waste_category': waste_category,
        'tenant_id': tenant_id,
        'status': 'success',
        'emissions_tco2e': None,
        'waste_mass_tonnes': None,
        'start_time': time.monotonic(),
    }

    # Increment active gauge
    metrics.inc_active(method)

    try:
        yield context

    except Exception as exc:
        context['status'] = 'error'
        context['error'] = str(exc)
        logger.error(
            "Calculation failed in track_calculation context: %s",
            exc, exc_info=True
        )
        raise

    finally:
        # Calculate duration
        duration_s = time.monotonic() - context['start_time']

        # Decrement active gauge
        metrics.dec_active(context['method'])

        # Record the calculation
        metrics.record_calculation(
            method=context['method'],
            treatment=context['treatment'],
            waste_category=context['waste_category'],
            tenant_id=context['tenant_id'],
            status=context['status'],
            emissions_tco2e=context.get('emissions_tco2e'),
            waste_mass_tonnes=context.get('waste_mass_tonnes'),
            duration_s=duration_s
        )


@contextmanager
def track_landfill_calculation(
    waste_category: str = "municipal_solid",
    climate_zone: str = "default",
    tenant_id: str = "unknown"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a landfill methane calculation lifecycle.

    Automatically records the landfill CH4 metrics when the context exits.
    The caller should populate ``context['ch4_kg']``,
    ``context['emissions_tco2e']``, and optionally FOD model parameters
    before exiting.

    Corresponds to IPCC First Order Decay (FOD) model for landfill:
        CH4 generated = DDOCm x F x (16/12) - R x (1 - OX)

    Args:
        waste_category: Category of waste being landfilled
        climate_zone: IPCC climate zone for decay rate selection
        tenant_id: Tenant identifier

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_landfill_calculation("food_waste", "temperate_wet",
        ...                                  "tenant-001") as ctx:
        ...     result = fod_engine.calculate(waste_data)
        ...     ctx['ch4_kg'] = result.ch4_kg
        ...     ctx['emissions_tco2e'] = result.total_tco2e
        ...     ctx['doc_fraction'] = result.doc
        ...     ctx['mcf'] = result.mcf
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'waste_category': waste_category,
        'climate_zone': climate_zone,
        'tenant_id': tenant_id,
        'status': 'success',
        'ch4_kg': None,
        'emissions_tco2e': None,
        'doc_fraction': None,
        'mcf': None,
        'oxidation_factor': None,
        'decay_rate_k': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active(CalculationMethod.IPCC_FOD.value)

    try:
        yield context

    except Exception as exc:
        context['status'] = 'error'
        context['error'] = str(exc)
        logger.error(
            "Landfill calculation failed: category=%s, zone=%s, error=%s",
            waste_category, climate_zone, exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active(CalculationMethod.IPCC_FOD.value)

        # Record landfill CH4
        if context.get('ch4_kg') is not None:
            metrics.record_landfill_ch4(
                waste_category=context['waste_category'],
                climate_zone=context['climate_zone'],
                tenant_id=context['tenant_id'],
                ch4_kg=context['ch4_kg'],
                doc_fraction=context.get('doc_fraction'),
                mcf=context.get('mcf'),
                oxidation_factor=context.get('oxidation_factor'),
                decay_rate_k=context.get('decay_rate_k'),
                duration_s=duration_s
            )

        # Also record as a primary calculation
        metrics.record_calculation(
            method=CalculationMethod.IPCC_FOD.value,
            treatment=TreatmentMethod.LANDFILL.value,
            waste_category=context['waste_category'],
            tenant_id=context['tenant_id'],
            status=context['status'],
            emissions_tco2e=context.get('emissions_tco2e'),
            duration_s=duration_s
        )


@contextmanager
def track_incineration_calculation(
    incinerator_type: str = "mass_burn",
    waste_category: str = "municipal_solid",
    tenant_id: str = "unknown"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks an incineration calculation lifecycle.

    Automatically records the incineration energy recovery and emissions
    metrics when the context exits. The caller should populate
    ``context['emissions_tco2e']``, ``context['energy_mwh']``, and
    ``context['waste_mass_tonnes']`` before exiting.

    Incineration emissions include:
        - CO2 from fossil carbon (plastics, synthetic materials)
        - N2O from combustion
        - Avoided emissions from energy recovery (grid displacement)

    Args:
        incinerator_type: Type of incinerator technology
        waste_category: Category of waste being incinerated
        tenant_id: Tenant identifier

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_incineration_calculation("mass_burn", "mixed",
        ...                                      "tenant-001") as ctx:
        ...     result = incineration_engine.calculate(waste_data)
        ...     ctx['emissions_tco2e'] = result.total_tco2e
        ...     ctx['energy_mwh'] = result.energy_recovered_mwh
        ...     ctx['waste_mass_tonnes'] = result.mass_tonnes
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'incinerator_type': incinerator_type,
        'waste_category': waste_category,
        'tenant_id': tenant_id,
        'status': 'success',
        'emissions_tco2e': None,
        'energy_mwh': None,
        'waste_mass_tonnes': None,
        'efficiency': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active(CalculationMethod.WASTE_TYPE_SPECIFIC.value)

    try:
        yield context

    except Exception as exc:
        context['status'] = 'error'
        context['error'] = str(exc)
        logger.error(
            "Incineration calculation failed: type=%s, category=%s, error=%s",
            incinerator_type, waste_category, exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active(CalculationMethod.WASTE_TYPE_SPECIFIC.value)

        # Record energy recovery
        if context.get('energy_mwh') is not None:
            metrics.record_incineration_energy(
                incinerator_type=context['incinerator_type'],
                tenant_id=context['tenant_id'],
                energy_mwh=context['energy_mwh'],
                waste_mass_tonnes=context.get('waste_mass_tonnes'),
                efficiency=context.get('efficiency'),
                waste_category=context.get('waste_category'),
                duration_s=duration_s
            )

        # Record as a primary calculation
        metrics.record_calculation(
            method=CalculationMethod.WASTE_TYPE_SPECIFIC.value,
            treatment=TreatmentMethod.INCINERATION.value,
            waste_category=context['waste_category'],
            tenant_id=context['tenant_id'],
            status=context['status'],
            emissions_tco2e=context.get('emissions_tco2e'),
            waste_mass_tonnes=context.get('waste_mass_tonnes'),
            duration_s=duration_s
        )


@contextmanager
def track_recycling_calculation(
    recycling_type: str = "mechanical",
    waste_category: str = "plastic",
    tenant_id: str = "unknown"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a recycling calculation lifecycle.

    Automatically records the recycling avoided emissions and waste mass
    metrics when the context exits. The caller should populate
    ``context['avoided_tco2e']``, ``context['emissions_tco2e']``, and
    ``context['waste_mass_tonnes']`` before exiting.

    Recycling emissions include:
        - Process emissions from recycling operations
        - Avoided emissions from displacing virgin material production
        - Net emissions = process emissions - avoided emissions

    Args:
        recycling_type: Type of recycling process
        waste_category: Category of waste being recycled
        tenant_id: Tenant identifier

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_recycling_calculation("mechanical", "plastic",
        ...                                   "tenant-001") as ctx:
        ...     result = recycling_engine.calculate(waste_data)
        ...     ctx['emissions_tco2e'] = result.process_tco2e
        ...     ctx['avoided_tco2e'] = result.avoided_tco2e
        ...     ctx['waste_mass_tonnes'] = result.mass_tonnes
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'recycling_type': recycling_type,
        'waste_category': waste_category,
        'tenant_id': tenant_id,
        'status': 'success',
        'emissions_tco2e': None,
        'avoided_tco2e': None,
        'waste_mass_tonnes': None,
        'virgin_ef': None,
        'recycled_ef': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active(CalculationMethod.WASTE_TYPE_SPECIFIC.value)

    try:
        yield context

    except Exception as exc:
        context['status'] = 'error'
        context['error'] = str(exc)
        logger.error(
            "Recycling calculation failed: type=%s, category=%s, error=%s",
            recycling_type, waste_category, exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active(CalculationMethod.WASTE_TYPE_SPECIFIC.value)

        # Record avoided emissions
        if context.get('avoided_tco2e') is not None:
            metrics.record_recycling_avoided_emissions(
                recycling_type=context['recycling_type'],
                waste_category=context['waste_category'],
                tenant_id=context['tenant_id'],
                avoided_tco2e=context['avoided_tco2e'],
                waste_mass_tonnes=context.get('waste_mass_tonnes'),
                virgin_ef=context.get('virgin_ef'),
                recycled_ef=context.get('recycled_ef'),
                duration_s=duration_s
            )

        # Record as a primary calculation
        metrics.record_calculation(
            method=CalculationMethod.WASTE_TYPE_SPECIFIC.value,
            treatment=TreatmentMethod.RECYCLING.value,
            waste_category=context['waste_category'],
            tenant_id=context['tenant_id'],
            status=context['status'],
            emissions_tco2e=context.get('emissions_tco2e'),
            waste_mass_tonnes=context.get('waste_mass_tonnes'),
            duration_s=duration_s
        )


@contextmanager
def track_wastewater_calculation(
    measurement_basis: str = "bod",
    treatment_system: str = "aerobic_centralized",
    tenant_id: str = "unknown"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a wastewater treatment calculation lifecycle.

    Automatically records the wastewater organic load and emissions
    metrics when the context exits. The caller should populate
    ``context['organic_load_kg']``, ``context['emissions_tco2e']``,
    ``context['ch4_kg']``, and ``context['n2o_kg']`` before exiting.

    IPCC Wastewater CH4:
        CH4 = (TOW x EF) - R
        where EF = Bo x MCF (max CH4 producing capacity x correction factor)

    Args:
        measurement_basis: Organic load measurement basis (bod/cod/toc)
        treatment_system: Treatment system type
        tenant_id: Tenant identifier

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_wastewater_calculation("bod", "anaerobic_lagoon",
        ...                                    "tenant-001") as ctx:
        ...     result = wastewater_engine.calculate(ww_data)
        ...     ctx['organic_load_kg'] = result.organic_load_kg
        ...     ctx['emissions_tco2e'] = result.total_tco2e
        ...     ctx['ch4_kg'] = result.ch4_kg
        ...     ctx['n2o_kg'] = result.n2o_kg
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'measurement_basis': measurement_basis,
        'treatment_system': treatment_system,
        'tenant_id': tenant_id,
        'status': 'success',
        'organic_load_kg': None,
        'emissions_tco2e': None,
        'ch4_kg': None,
        'n2o_kg': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active(CalculationMethod.TREATMENT_SPECIFIC.value)

    try:
        yield context

    except Exception as exc:
        context['status'] = 'error'
        context['error'] = str(exc)
        logger.error(
            "Wastewater calculation failed: basis=%s, system=%s, error=%s",
            measurement_basis, treatment_system, exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active(CalculationMethod.TREATMENT_SPECIFIC.value)

        # Record wastewater organic load
        if context.get('organic_load_kg') is not None:
            metrics.record_wastewater_organic_load(
                measurement_basis=context['measurement_basis'],
                treatment_system=context['treatment_system'],
                tenant_id=context['tenant_id'],
                organic_load_kg=context['organic_load_kg'],
                emissions_tco2e=context.get('emissions_tco2e'),
                ch4_kg=context.get('ch4_kg'),
                n2o_kg=context.get('n2o_kg'),
                duration_s=duration_s
            )

        # Record as a primary calculation
        metrics.record_calculation(
            method=CalculationMethod.TREATMENT_SPECIFIC.value,
            treatment=TreatmentMethod.WASTEWATER_TREATMENT.value,
            waste_category=WasteCategory.SLUDGE.value,
            tenant_id=context['tenant_id'],
            status=context['status'],
            emissions_tco2e=context.get('emissions_tco2e'),
            duration_s=duration_s
        )


@contextmanager
def track_compliance_check(
    framework: str = "ghg_protocol",
    tenant_id: str = "unknown"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a compliance check lifecycle.

    Args:
        framework: Compliance framework being checked
        tenant_id: Tenant identifier

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_compliance_check("ghg_protocol", "tenant-001") as ctx:
        ...     result = compliance_engine.check(report_data)
        ...     ctx['result'] = result.status
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'framework': framework,
        'tenant_id': tenant_id,
        'result': 'compliant',
        'start_time': time.monotonic(),
    }

    try:
        yield context

    except Exception as exc:
        context['result'] = 'non_compliant'
        context['error'] = str(exc)
        logger.error(
            "Compliance check failed: framework=%s, error=%s",
            framework, exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']

        metrics.record_compliance_check(
            framework=context['framework'],
            result=context['result'],
            tenant_id=context['tenant_id'],
            duration_s=duration_s
        )


@contextmanager
def track_ef_lookup(
    source: str = "ipcc_2006",
    waste_category: str = "municipal_solid"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks an emission factor lookup lifecycle.

    Args:
        source: Emission factor source database
        waste_category: Waste category for the lookup

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_ef_lookup("ipcc_2006", "food_waste") as ctx:
        ...     ef = ef_database.lookup(waste_type="food_waste", treatment="landfill")
        ...     ctx['count'] = 1
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'source': source,
        'waste_category': waste_category,
        'count': 1,
        'start_time': time.monotonic(),
    }

    try:
        yield context

    except Exception as exc:
        context['error'] = str(exc)
        metrics.record_error(
            error_type=ErrorType.EMISSION_FACTOR_UNAVAILABLE.value,
            operation="ef_lookup",
            treatment=TreatmentMethod.OTHER.value
        )
        logger.error(
            "EF lookup failed: source=%s, category=%s, error=%s",
            source, waste_category, exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']

        if 'error' not in context:
            metrics.record_ef_lookup(
                source=context['source'],
                waste_category=context['waste_category'],
                count=context.get('count', 1),
                duration_s=duration_s
            )


@contextmanager
def track_batch(
    method: str = "waste_type_specific"
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a batch calculation lifecycle.

    Args:
        method: Primary calculation method for the batch

    Yields:
        Mutable dictionary for the caller to populate with results

    Example:
        >>> with track_batch("waste_type_specific") as ctx:
        ...     results = batch_engine.process(waste_records)
        ...     ctx['size'] = len(waste_records)
        ...     ctx['successful'] = sum(1 for r in results if r.ok)
        ...     ctx['failed'] = sum(1 for r in results if not r.ok)
        ...     ctx['total_emissions_tco2e'] = sum(r.tco2e for r in results if r.ok)
        ...     ctx['total_waste_mass_tonnes'] = sum(r.mass for r in results if r.ok)
    """
    metrics = get_metrics()
    context: Dict[str, Any] = {
        'method': method,
        'status': 'completed',
        'size': 0,
        'successful': None,
        'failed': None,
        'total_emissions_tco2e': None,
        'total_waste_mass_tonnes': None,
        'tenant_id': None,
        'start_time': time.monotonic(),
    }

    metrics.inc_active(method)

    try:
        yield context

    except Exception as exc:
        context['status'] = 'failed'
        context['error'] = str(exc)
        logger.error(
            "Batch calculation failed: method=%s, error=%s",
            method, exc, exc_info=True
        )
        raise

    finally:
        duration_s = time.monotonic() - context['start_time']
        metrics.dec_active(context['method'])

        size = context.get('size', 0)
        if size > 0:
            metrics.record_batch(
                method=context['method'],
                size=size,
                successful=context.get('successful'),
                failed=context.get('failed'),
                duration_s=duration_s,
                total_emissions_tco2e=context.get('total_emissions_tco2e'),
                total_waste_mass_tonnes=context.get('total_waste_mass_tonnes'),
                tenant_id=context.get('tenant_id')
            )


@contextmanager
def track_duration(
    method: str = "waste_type_specific",
    treatment: str = "landfill"
) -> Generator[None, None, None]:
    """
    Context manager that tracks the duration of an arbitrary operation.

    Records the elapsed time in the calculation_duration_seconds histogram
    when the context exits. Lightweight alternative to track_calculation
    when only duration tracking is needed.

    Args:
        method: Calculation method label for the duration histogram
        treatment: Treatment method label for the duration histogram

    Yields:
        None

    Example:
        >>> with track_duration("ipcc_fod", "landfill"):
        ...     factors = load_landfill_decay_parameters()
    """
    metrics = get_metrics()
    start = time.monotonic()

    try:
        yield

    finally:
        duration_s = time.monotonic() - start

        try:
            metrics.calculation_duration_seconds.labels(
                method=method,
                treatment=treatment
            ).observe(duration_s)

            logger.debug(
                "Tracked duration: method=%s, treatment=%s, duration=%.4fs",
                method, treatment, duration_s
            )

        except Exception as e:
            logger.error(
                "Failed to record duration for method=%s, treatment=%s: %s",
                method, treatment, e, exc_info=True
            )


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    # Main class and accessors
    'WasteGeneratedMetrics',
    'get_metrics',
    'reset_metrics',

    # Context managers
    'track_calculation',
    'track_landfill_calculation',
    'track_incineration_calculation',
    'track_recycling_calculation',
    'track_wastewater_calculation',
    'track_compliance_check',
    'track_ef_lookup',
    'track_batch',
    'track_duration',

    # Enumerations
    'CalculationMethod',
    'TreatmentMethod',
    'WasteCategory',
    'CalculationStatus',
    'ClimateZone',
    'IncineratorType',
    'RecyclingType',
    'MeasurementBasis',
    'WastewaterTreatmentSystem',
    'ErrorType',
    'Framework',
    'ComplianceStatus',
    'EFSource',
    'DataSource',
    'BatchStatus',

    # Availability flag
    'PROMETHEUS_AVAILABLE',
]
