# -*- coding: utf-8 -*-
"""
MaterialBalanceEngine - Carbon Mass Balance Tracking (Engine 3 of 6)

AGENT-MRV-004: Process Emissions Agent

Implements the mass balance approach to calculating process emissions per
IPCC 2006 Guidelines Volume 3, Chapter 2 (Tier 2/3).  Tracks raw material
inputs, calculates carbon balance across all inputs and outputs, and handles
by-product emission credits for industrial process calculations.

Fundamental Mass Balance Equation:
    CO2_emissions = (C_input - C_output - C_stock_change) x (44/12)

Where:
    C_input   = sum(material_qty x carbon_content x fraction_oxidized)
                for all non-product materials (raw materials, fuels, etc.)
    C_output  = sum(product_qty x carbon_content)
                for all products and by-products
    44/12     = molecular weight ratio of CO2 to C (3.6667)

Carbon Content Defaults (by material type):
    Calcium Carbonate (CaCO3): 0.120  (12.0% carbon)
    Magnesium Carbonate (MgCO3): 0.143  (14.3% carbon)
    Iron Carbonate (FeCO3): 0.103  (10.3% carbon)
    Limestone: 0.120  (predominantly CaCO3)
    Dolomite: 0.130  (mixed CaCO3/MgCO3)
    Clinker: 0.000  (product, carbon already released)
    Coke: 0.850  (85.0% carbon)
    Coal: 0.750  (75.0% carbon)
    Iron Ore: 0.000  (negligible carbon)
    Scrap Metal: 0.005  (0.5% carbon, residual)
    Natural Gas Feedstock: 0.750  (75.0% carbon)
    Naphtha: 0.836  (83.6% carbon)
    Ethane: 0.799  (79.9% carbon)
    Bauxite: 0.000  (aluminum ore, no carbon)
    Alumina: 0.000  (Al2O3, no carbon)

Carbonate Decomposition CO2 Factors (tCO2/t carbonate):
    Calcite (CaCO3):           0.440  (MW: 100 -> 56 + 44)
    Dolomite (CaMg(CO3)2):    0.477  (MW: 184 -> 96 + 88; 88/184)
    Magnesite (MgCO3):        0.522  (MW: 84.3 -> 40.3 + 44)
    Siderite (FeCO3):         0.380  (MW: 115.9 -> 71.9 + 44)
    Ankerite (Ca(Fe,Mg)(CO3)2): 0.407  (approximate mixed carbonate)

Process-Specific Parameters:
    Clinker-to-Cement Ratio: default 0.95, valid range [0.50, 1.00]
    CKD Correction Factor: default 1.02 (2% for cement kiln dust losses)
    BF Slag Ratio: 0.30 (tonnes slag per tonne steel in BF-BOF)
    EAF Scrap Fraction: 0.85 (typical scrap input for EAF steelmaking)

Process-Specific Balance Implementations:
    - Cement: clinker-based mass balance with CKD correction, organic
      carbon in raw meal, carbonate decomposition
    - Iron/Steel: BF-BOF (coke + iron ore), EAF (scrap + electrode),
      DRI (natural gas reduction) routes
    - Aluminum: carbon anode consumption mass balance, prebake vs Soderberg
    - Ammonia: natural gas feedstock carbon, steam methane reforming
    - Petrochemical: naphtha/ethane cracker carbon balance, product carbon

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal with 8+ decimal places.
    - No LLM calls in any calculation path.
    - Every step is recorded in the calculation trace.
    - SHA-256 provenance hash for every result.
    - Same inputs always produce identical outputs (deterministic).

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.process_emissions.material_balance import (
    ...     MaterialBalanceEngine, MaterialInput,
    ... )
    >>> from decimal import Decimal
    >>> engine = MaterialBalanceEngine()
    >>> materials = [
    ...     MaterialInput(
    ...         material_id="limestone-001",
    ...         material_type="LIMESTONE",
    ...         quantity=Decimal("100000"),
    ...         carbon_content=Decimal("0.120"),
    ...     ),
    ...     MaterialInput(
    ...         material_id="clinker-001",
    ...         material_type="CLINKER",
    ...         quantity=Decimal("62000"),
    ...         carbon_content=Decimal("0.000"),
    ...         is_product=True,
    ...     ),
    ... ]
    >>> balance = engine.calculate_carbon_balance(materials)
    >>> print(balance.co2_emissions_tonnes)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-004 Process Emissions (GL-MRV-SCOPE1-004)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["MaterialBalanceEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.process_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.process_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.process_emissions.metrics import (
        record_material_operation as _record_material_operation,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_material_operation = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Decimal shorthand
# ---------------------------------------------------------------------------

_D = Decimal


def _d(value: Any) -> Decimal:
    """Convert a numeric value to Decimal via string to avoid float artefacts."""
    return Decimal(str(value))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Molecular weight ratio CO2/C  (44/12 = 3.666...)
CARBON_TO_CO2: Decimal = _D("3.66666667")

#: Exact rational 44/12 for high-precision calculations.
CARBON_TO_CO2_EXACT: Decimal = _D("44") / _D("12")

#: Default carbon content fractions by material type.
#: Sources: IPCC 2006 Guidelines Vol 3, Table 2.1-2.3; EPA Subpart data.
DEFAULT_CARBON_CONTENT: Dict[str, Decimal] = {
    "CALCIUM_CARBONATE": _D("0.120"),
    "MAGNESIUM_CARBONATE": _D("0.143"),
    "IRON_CARBONATE": _D("0.103"),
    "LIMESTONE": _D("0.120"),
    "DOLOMITE": _D("0.130"),
    "CLINKER": _D("0.000"),
    "COKE": _D("0.850"),
    "COAL": _D("0.750"),
    "IRON_ORE": _D("0.000"),
    "SCRAP_METAL": _D("0.005"),
    "NATURAL_GAS_FEEDSTOCK": _D("0.750"),
    "NAPHTHA": _D("0.836"),
    "ETHANE": _D("0.799"),
    "BAUXITE": _D("0.000"),
    "ALUMINA": _D("0.000"),
    "OTHER": _D("0.000"),
    # Additional oxide/hydroxide entries
    "CALCIUM_OXIDE": _D("0.000"),
    "CALCIUM_HYDROXIDE": _D("0.000"),
    "CHALK": _D("0.120"),
    "CITE": _D("0.000"),
}

#: Carbonate decomposition CO2 factors (tCO2 per tonne carbonate decomposed).
#: Source: IPCC 2006, Volume 3, Table 2.1.
CARBONATE_CO2_FACTORS: Dict[str, Decimal] = {
    "CALCITE": _D("0.440"),
    "DOLOMITE": _D("0.477"),
    "MAGNESITE": _D("0.522"),
    "SIDERITE": _D("0.380"),
    "ANKERITE": _D("0.407"),
}

# ---------------------------------------------------------------------------
# Process-specific parameters
# ---------------------------------------------------------------------------

#: Default clinker-to-cement ratio.
CLINKER_TO_CEMENT_RATIO_DEFAULT: Decimal = _D("0.95")
CLINKER_TO_CEMENT_RATIO_MIN: Decimal = _D("0.50")
CLINKER_TO_CEMENT_RATIO_MAX: Decimal = _D("1.00")

#: CKD (cement kiln dust) correction factor.  Accounts for partially
#: calcined raw material lost as kiln dust. Typical range 1.00-1.05.
CKD_CORRECTION_FACTOR_DEFAULT: Decimal = _D("1.02")

#: Blast furnace slag ratio (tonnes slag / tonne hot metal) for BF-BOF.
BF_SLAG_RATIO: Decimal = _D("0.30")

#: Typical scrap metal fraction in EAF charge (dimensionless, 0-1).
EAF_SCRAP_FRACTION: Decimal = _D("0.85")

#: Anode consumption factor for prebake aluminum smelting (t anode / t Al).
PREBAKE_ANODE_CONSUMPTION: Decimal = _D("0.45")

#: Anode consumption factor for Soderberg smelting (t paste / t Al).
SODERBERG_PASTE_CONSUMPTION: Decimal = _D("0.55")

#: Carbon content of prebake anodes.
ANODE_CARBON_CONTENT: Decimal = _D("0.850")

#: Carbon content of Soderberg paste.
SODERBERG_PASTE_CARBON_CONTENT: Decimal = _D("0.800")

#: Steam methane reforming carbon yield (fraction of feedstock carbon
#: released as CO2). Remainder is in H2 product (no carbon) and
#: unreacted CH4.
SMR_CARBON_OXIDATION_FRACTION: Decimal = _D("0.995")

#: Ethylene cracker carbon conversion (fraction of feedstock carbon
#: ending up in primary products vs CO2).
CRACKER_PRODUCT_CARBON_FRACTION: Decimal = _D("0.60")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ProcessRoute(str, Enum):
    """Production routes for process-specific balance calculations."""

    # Cement
    CEMENT = "CEMENT"

    # Iron and steel
    BF_BOF = "BF_BOF"
    EAF = "EAF"
    DRI = "DRI"

    # Aluminum
    ALUMINUM_PREBAKE = "ALUMINUM_PREBAKE"
    ALUMINUM_SODERBERG = "ALUMINUM_SODERBERG"

    # Chemical
    AMMONIA = "AMMONIA"
    PETROCHEMICAL = "PETROCHEMICAL"

    # Generic
    GENERIC = "GENERIC"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MaterialInput:
    """A single material input or output in a mass balance calculation.

    Attributes:
        material_id: Unique identifier for this material batch.
        material_type: Classification of the material (e.g. LIMESTONE,
            COKE, CLINKER).  Used for default carbon content lookup.
        quantity: Mass in metric tonnes.
        carbon_content: Carbon mass fraction (0-1).  Overrides the
            default for the material type when provided.
        carbonate_content: Carbonate mineral mass fraction (0-1).
            Relevant only for carbonate-bearing materials.
        carbonate_type: Type of carbonate mineral (CALCITE, DOLOMITE,
            MAGNESITE, SIDERITE, ANKERITE).
        moisture_content: Moisture fraction (0-1).  Reduces the
            effective dry mass.
        is_product: True if this material is a primary product (output).
        is_by_product: True if this material is a by-product/co-product.
        fraction_oxidized: Fraction of carbon that is oxidized (0-1).
            Defaults to 1.0 (100% oxidation).
    """

    material_id: str = ""
    material_type: str = "OTHER"
    quantity: Decimal = _D("0")
    carbon_content: Decimal = _D("-1")  # sentinel: -1 means "use default"
    carbonate_content: Decimal = _D("0")
    carbonate_type: str = ""
    moisture_content: Decimal = _D("0")
    is_product: bool = False
    is_by_product: bool = False
    fraction_oxidized: Decimal = _D("1.0")


@dataclass
class CarbonBalance:
    """Result of a carbon mass balance calculation.

    Attributes:
        total_carbon_input_tonnes: Total carbon entering the process
            from all non-product materials.
        total_carbon_output_tonnes: Total carbon leaving the process in
            products and by-products.
        carbon_stock_change_tonnes: Change in on-site carbon stock
            (positive = accumulation, reduces emissions).
        net_carbon_emissions_tonnes: Net carbon emitted to atmosphere.
        co2_emissions_tonnes: Net CO2 emitted (net_carbon x 44/12).
        material_details: Per-material breakdown of carbon flows.
        calculation_trace: Human-readable ordered list of calc steps.
        provenance_hash: SHA-256 hash for audit trail integrity.
        timestamp: UTC timestamp of the calculation.
    """

    total_carbon_input_tonnes: Decimal = _D("0")
    total_carbon_output_tonnes: Decimal = _D("0")
    carbon_stock_change_tonnes: Decimal = _D("0")
    net_carbon_emissions_tonnes: Decimal = _D("0")
    co2_emissions_tonnes: Decimal = _D("0")
    material_details: List[Dict[str, Any]] = field(default_factory=list)
    calculation_trace: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    timestamp: str = ""


@dataclass
class ByProductCredit:
    """Emission credit for carbon retained in a by-product.

    Attributes:
        by_product_type: Material type of the by-product.
        quantity_tonnes: Mass of by-product in tonnes.
        carbon_content: Carbon fraction of the by-product.
        credit_co2_tonnes: CO2 credit = quantity x carbon_content x 44/12.
    """

    by_product_type: str = ""
    quantity_tonnes: Decimal = _D("0")
    carbon_content: Decimal = _D("0")
    credit_co2_tonnes: Decimal = _D("0")


@dataclass
class MaterialSummary:
    """Aggregated summary statistics for a set of material inputs/outputs.

    Attributes:
        total_input_mass_tonnes: Total mass of all raw material inputs.
        total_output_mass_tonnes: Total mass of all products/by-products.
        total_input_carbon_tonnes: Total carbon mass in inputs.
        total_output_carbon_tonnes: Total carbon mass in outputs.
        input_count: Number of input material records.
        output_count: Number of output material records.
        by_product_count: Number of by-product records.
        dominant_input_type: Material type with the largest input mass.
        dominant_output_type: Material type with the largest output mass.
        mass_balance_residual_tonnes: Mass in minus mass out.
        mass_balance_residual_pct: Residual as percentage of input mass.
        provenance_hash: SHA-256 hash of the summary.
    """

    total_input_mass_tonnes: Decimal = _D("0")
    total_output_mass_tonnes: Decimal = _D("0")
    total_input_carbon_tonnes: Decimal = _D("0")
    total_output_carbon_tonnes: Decimal = _D("0")
    input_count: int = 0
    output_count: int = 0
    by_product_count: int = 0
    dominant_input_type: str = ""
    dominant_output_type: str = ""
    mass_balance_residual_tonnes: Decimal = _D("0")
    mass_balance_residual_pct: Decimal = _D("0")
    provenance_hash: str = ""


# ---------------------------------------------------------------------------
# MaterialBalanceEngine
# ---------------------------------------------------------------------------


class MaterialBalanceEngine:
    """Engine 3: Material Balance - Carbon mass balance tracking.

    Implements IPCC 2006 mass balance methodology for industrial
    process emission calculations.  Tracks carbon through material
    inputs and product outputs to derive net process CO2 emissions.

    This engine supports:
        - Generic carbon balance for any process
        - Carbonate decomposition emission calculation
        - By-product carbon credit accounting
        - Mass balance verification (closure check)
        - Process-specific balance routines for cement, iron/steel,
          aluminum, ammonia, and petrochemical processes

    Thread Safety:
        All mutable state is protected by ``self._lock``
        (``threading.RLock``).

    Attributes:
        _precision: Number of decimal places for intermediate rounding.
        _tracked_materials: Registry of materials by material_id.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> engine = MaterialBalanceEngine()
        >>> materials = [
        ...     MaterialInput(
        ...         material_id="ls-1",
        ...         material_type="LIMESTONE",
        ...         quantity=Decimal("50000"),
        ...         carbon_content=Decimal("0.120"),
        ...     ),
        ... ]
        >>> balance = engine.calculate_carbon_balance(materials)
        >>> assert balance.co2_emissions_tonnes > 0
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, precision: int = 8) -> None:
        """Initialise the MaterialBalanceEngine.

        Args:
            precision: Number of decimal places for intermediate
                rounding.  Defaults to 8.  Can be overridden from
                the global ProcessEmissionsConfig.
        """
        if _CONFIG_AVAILABLE:
            try:
                cfg = _get_config()  # type: ignore[misc]
                precision = cfg.decimal_precision
            except Exception:
                pass

        self._precision: int = precision
        self._tracked_materials: Dict[str, MaterialInput] = {}
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "MaterialBalanceEngine initialised: precision=%d",
            self._precision,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _round(self, value: Decimal) -> Decimal:
        """Round a Decimal value to the configured precision.

        Args:
            value: The value to round.

        Returns:
            Value rounded to ``self._precision`` decimal places using
            ROUND_HALF_UP.
        """
        quantize_str = "1." + "0" * self._precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _now_iso(self) -> str:
        """Return current UTC time as ISO-8601 string."""
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _hash(self, data: str) -> str:
        """Compute SHA-256 hex digest for provenance."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _record_provenance(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        details: Dict[str, Any],
    ) -> Optional[str]:
        """Record a provenance entry if the tracker is available.

        Args:
            entity_type: Type of entity (e.g. "material_balance").
            entity_id: Unique identifier for the entity.
            action: Action performed (e.g. "calculate").
            details: Additional details to record.

        Returns:
            Provenance hash if recorded, None otherwise.
        """
        if not _PROVENANCE_AVAILABLE:
            return None
        try:
            tracker = _get_provenance_tracker()  # type: ignore[misc]
            entry = tracker.record(
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                data=details,
            )
            return entry.hash if hasattr(entry, "hash") else None
        except Exception as exc:
            logger.warning("Provenance recording failed: %s", exc)
            return None

    def _record_metric(self, operation: str, status: str = "success") -> None:
        """Record a Prometheus metric for a material operation.

        Args:
            operation: Operation name (e.g. "carbon_balance").
            status: Result status ("success" or "error").
        """
        if _METRICS_AVAILABLE and _record_material_operation is not None:
            try:
                _record_material_operation(operation, status)
            except Exception:
                pass

    def _resolve_carbon_content(self, material: MaterialInput) -> Decimal:
        """Resolve the carbon content for a material input.

        If the material has an explicit carbon_content >= 0, use it.
        Otherwise look up the default by material_type.  If no default
        is found, return 0.

        Args:
            material: The material input to resolve.

        Returns:
            Carbon content as a Decimal fraction (0-1).
        """
        if material.carbon_content >= _D("0"):
            return material.carbon_content

        mat_key = material.material_type.upper().replace(" ", "_")
        default = DEFAULT_CARBON_CONTENT.get(mat_key, _D("0"))
        return default

    def _dry_mass(self, material: MaterialInput) -> Decimal:
        """Calculate the dry mass of a material input.

        Removes moisture content from the total quantity.

        Args:
            material: Material input with quantity and moisture_content.

        Returns:
            Dry mass in tonnes.
        """
        moisture = max(_D("0"), min(material.moisture_content, _D("1")))
        return self._round(material.quantity * (_D("1") - moisture))

    # ------------------------------------------------------------------
    # Public API: track_materials
    # ------------------------------------------------------------------

    def track_materials(
        self,
        materials: List[MaterialInput],
    ) -> Dict[str, Any]:
        """Register material inputs/outputs for tracking.

        Validates and stores materials in the internal registry keyed
        by material_id.  Materials with blank IDs are assigned a UUID.

        Args:
            materials: List of MaterialInput records to register.

        Returns:
            Dict with keys:
                - registered_count (int): Number of materials registered.
                - material_ids (List[str]): List of registered IDs.
                - provenance_hash (str): SHA-256 hash.

        Raises:
            ValueError: If a material has a negative quantity.
        """
        start_time = time.monotonic()
        registered_ids: List[str] = []

        with self._lock:
            for mat in materials:
                # Assign ID if blank
                if not mat.material_id:
                    mat.material_id = f"mat_{uuid4().hex[:12]}"

                # Validate quantity
                if mat.quantity < _D("0"):
                    raise ValueError(
                        f"Material '{mat.material_id}' has negative "
                        f"quantity: {mat.quantity}"
                    )

                # Validate moisture
                if mat.moisture_content < _D("0") or mat.moisture_content > _D("1"):
                    raise ValueError(
                        f"Material '{mat.material_id}' has invalid "
                        f"moisture_content: {mat.moisture_content}"
                    )

                # Validate fraction_oxidized
                if mat.fraction_oxidized < _D("0") or mat.fraction_oxidized > _D("1"):
                    raise ValueError(
                        f"Material '{mat.material_id}' has invalid "
                        f"fraction_oxidized: {mat.fraction_oxidized}"
                    )

                self._tracked_materials[mat.material_id] = mat
                registered_ids.append(mat.material_id)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        provenance_hash = self._hash(
            json.dumps(registered_ids, sort_keys=True)
        )

        self._record_metric("track_materials")
        self._record_provenance(
            entity_type="material_balance",
            entity_id=provenance_hash[:16],
            action="track_materials",
            details={
                "registered_count": len(registered_ids),
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )

        logger.info(
            "Tracked %d materials in %.2f ms",
            len(registered_ids),
            elapsed_ms,
        )

        return {
            "registered_count": len(registered_ids),
            "material_ids": registered_ids,
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # Public API: calculate_carbon_balance
    # ------------------------------------------------------------------

    def calculate_carbon_balance(
        self,
        materials: List[MaterialInput],
        carbon_stock_change: Decimal = _D("0"),
    ) -> CarbonBalance:
        """Calculate the carbon mass balance for a set of materials.

        Implements the IPCC 2006 mass balance equation:
            CO2 = (C_in - C_out - delta_stock) x 44/12

        Where C_in is the sum of (dry_mass x carbon_content x
        fraction_oxidized) for all non-product inputs, and C_out is
        the sum of (dry_mass x carbon_content) for all products and
        by-products.

        Args:
            materials: List of MaterialInput records.  Materials with
                is_product=True or is_by_product=True are treated as
                outputs; all others as inputs.
            carbon_stock_change: Change in on-site carbon stock in
                tonnes.  Positive values reduce net emissions.

        Returns:
            CarbonBalance with all calculated fields populated.

        Raises:
            ValueError: If materials list is empty or all quantities
                are zero.
        """
        start_time = time.monotonic()

        if not materials:
            raise ValueError("Materials list must not be empty")

        trace: List[str] = []
        details: List[Dict[str, Any]] = []

        total_carbon_in = _D("0")
        total_carbon_out = _D("0")

        trace.append(
            f"Starting carbon balance calculation with "
            f"{len(materials)} material(s)"
        )

        # ----- Process each material -----
        for mat in materials:
            dry_mass = self._dry_mass(mat)
            cc = self._resolve_carbon_content(mat)

            if mat.is_product or mat.is_by_product:
                # Output stream
                carbon_out = self._round(dry_mass * cc)
                total_carbon_out += carbon_out
                flow = "output"
                trace.append(
                    f"  OUTPUT {mat.material_type} "
                    f"[{mat.material_id}]: "
                    f"dry_mass={dry_mass} t, "
                    f"carbon_content={cc}, "
                    f"carbon_out={carbon_out} tC"
                )
            else:
                # Input stream
                fox = max(_D("0"), min(mat.fraction_oxidized, _D("1")))
                carbon_in = self._round(dry_mass * cc * fox)
                total_carbon_in += carbon_in
                flow = "input"
                trace.append(
                    f"  INPUT  {mat.material_type} "
                    f"[{mat.material_id}]: "
                    f"dry_mass={dry_mass} t, "
                    f"carbon_content={cc}, "
                    f"fraction_oxidized={fox}, "
                    f"carbon_in={carbon_in} tC"
                )

            details.append({
                "material_id": mat.material_id,
                "material_type": mat.material_type,
                "quantity_tonnes": str(mat.quantity),
                "dry_mass_tonnes": str(dry_mass),
                "carbon_content": str(cc),
                "flow_direction": flow,
                "carbon_tonnes": str(
                    carbon_in if flow == "input" else carbon_out  # type: ignore[possibly-undefined]
                ),
            })

        # ----- Net carbon emissions -----
        total_carbon_in = self._round(total_carbon_in)
        total_carbon_out = self._round(total_carbon_out)
        stock_change = self._round(carbon_stock_change)

        net_carbon = self._round(
            total_carbon_in - total_carbon_out - stock_change
        )
        # Clamp to zero if negative (indicates carbon accumulation)
        if net_carbon < _D("0"):
            trace.append(
                f"  Net carbon is negative ({net_carbon} tC), indicating "
                f"carbon accumulation. Clamping CO2 emissions to zero."
            )
            co2_emissions = _D("0")
        else:
            co2_emissions = self._round(net_carbon * CARBON_TO_CO2_EXACT)

        trace.append(
            f"  Total carbon input:  {total_carbon_in} tC"
        )
        trace.append(
            f"  Total carbon output: {total_carbon_out} tC"
        )
        trace.append(
            f"  Carbon stock change: {stock_change} tC"
        )
        trace.append(
            f"  Net carbon emitted:  {net_carbon} tC"
        )
        trace.append(
            f"  CO2 emissions:       {co2_emissions} tCO2 "
            f"(net_carbon x 44/12)"
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        trace.append(f"  Elapsed: {elapsed_ms:.2f} ms")

        # Provenance
        prov_data = json.dumps(
            {
                "carbon_in": str(total_carbon_in),
                "carbon_out": str(total_carbon_out),
                "stock_change": str(stock_change),
                "net_carbon": str(net_carbon),
                "co2": str(co2_emissions),
                "material_count": len(materials),
            },
            sort_keys=True,
        )
        prov_hash = self._hash(prov_data)

        self._record_provenance(
            entity_type="material_balance",
            entity_id=prov_hash[:16],
            action="calculate",
            details={"co2_tonnes": str(co2_emissions)},
        )
        self._record_metric("carbon_balance")

        result = CarbonBalance(
            total_carbon_input_tonnes=total_carbon_in,
            total_carbon_output_tonnes=total_carbon_out,
            carbon_stock_change_tonnes=stock_change,
            net_carbon_emissions_tonnes=net_carbon,
            co2_emissions_tonnes=co2_emissions,
            material_details=details,
            calculation_trace=trace,
            provenance_hash=prov_hash,
            timestamp=self._now_iso(),
        )

        logger.info(
            "Carbon balance: C_in=%s tC, C_out=%s tC, "
            "net=%s tC, CO2=%s t in %.2f ms",
            total_carbon_in,
            total_carbon_out,
            net_carbon,
            co2_emissions,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: get_carbonate_emissions
    # ------------------------------------------------------------------

    def get_carbonate_emissions(
        self,
        materials: List[MaterialInput],
        fraction_calcined: Decimal = _D("1.0"),
    ) -> Decimal:
        """Calculate CO2 emissions from carbonate decomposition.

        For each carbonate-bearing material:
            CO2 = quantity x carbonate_content x carbonate_co2_factor
                  x fraction_calcined

        This is the stoichiometric approach complementing the mass
        balance.  It directly computes CO2 from the chemical
        decomposition of carbonates (e.g. CaCO3 -> CaO + CO2).

        Args:
            materials: List of MaterialInput records that may contain
                carbonate minerals.
            fraction_calcined: Fraction of carbonate actually
                decomposed (0-1).  Defaults to 1.0 (fully calcined).

        Returns:
            Total CO2 emissions from carbonate decomposition in tonnes.

        Raises:
            ValueError: If fraction_calcined is outside [0, 1].
        """
        if fraction_calcined < _D("0") or fraction_calcined > _D("1"):
            raise ValueError(
                f"fraction_calcined must be in [0, 1], "
                f"got {fraction_calcined}"
            )

        total_co2 = _D("0")

        for mat in materials:
            if mat.carbonate_content <= _D("0"):
                continue

            carb_key = mat.carbonate_type.upper().replace(" ", "_")
            carb_factor = CARBONATE_CO2_FACTORS.get(carb_key)

            if carb_factor is None:
                logger.warning(
                    "Unknown carbonate type '%s' for material '%s'; "
                    "skipping carbonate emission calculation",
                    mat.carbonate_type,
                    mat.material_id,
                )
                continue

            dry_mass = self._dry_mass(mat)
            co2 = self._round(
                dry_mass
                * mat.carbonate_content
                * carb_factor
                * fraction_calcined
            )
            total_co2 += co2

            logger.debug(
                "Carbonate CO2: %s [%s] = %s t x %s x %s x %s = %s tCO2",
                mat.material_type,
                mat.material_id,
                dry_mass,
                mat.carbonate_content,
                carb_factor,
                fraction_calcined,
                co2,
            )

        total_co2 = self._round(total_co2)
        self._record_metric("carbonate_emissions")

        logger.info(
            "Carbonate emissions: %s tCO2 from %d material(s)",
            total_co2,
            len(materials),
        )

        return total_co2

    # ------------------------------------------------------------------
    # Public API: calculate_by_product_credits
    # ------------------------------------------------------------------

    def calculate_by_product_credits(
        self,
        by_products: List[MaterialInput],
    ) -> List[ByProductCredit]:
        """Calculate CO2 emission credits for carbon retained in by-products.

        By-product credits represent carbon that exits the process
        boundary in a solid/liquid by-product rather than being
        released as CO2 to the atmosphere.  This carbon is subtracted
        from gross emissions in the final balance.

        Credit_CO2 = quantity x carbon_content x (44/12)

        Args:
            by_products: List of MaterialInput records with
                is_by_product=True.

        Returns:
            List of ByProductCredit records, one per by-product.
        """
        credits: List[ByProductCredit] = []

        for bp in by_products:
            if not bp.is_by_product:
                logger.debug(
                    "Skipping material '%s': not marked as by-product",
                    bp.material_id,
                )
                continue

            dry_mass = self._dry_mass(bp)
            cc = self._resolve_carbon_content(bp)
            credit_co2 = self._round(dry_mass * cc * CARBON_TO_CO2_EXACT)

            credit = ByProductCredit(
                by_product_type=bp.material_type,
                quantity_tonnes=dry_mass,
                carbon_content=cc,
                credit_co2_tonnes=credit_co2,
            )
            credits.append(credit)

            logger.debug(
                "By-product credit: %s = %s t x %s x 44/12 = %s tCO2",
                bp.material_type,
                dry_mass,
                cc,
                credit_co2,
            )

        total_credit = sum(
            (c.credit_co2_tonnes for c in credits), _D("0")
        )
        self._record_metric("by_product_credits")

        logger.info(
            "By-product credits: %d by-products, total credit = %s tCO2",
            len(credits),
            total_credit,
        )

        return credits

    # ------------------------------------------------------------------
    # Public API: get_material_summary
    # ------------------------------------------------------------------

    def get_material_summary(
        self,
        materials: List[MaterialInput],
    ) -> MaterialSummary:
        """Generate summary statistics for a set of material inputs/outputs.

        Provides aggregate totals for input mass, output mass, carbon
        content, and mass balance residual.

        Args:
            materials: List of MaterialInput records.

        Returns:
            MaterialSummary with all fields populated.
        """
        total_in_mass = _D("0")
        total_out_mass = _D("0")
        total_in_carbon = _D("0")
        total_out_carbon = _D("0")
        input_count = 0
        output_count = 0
        bp_count = 0

        max_in_mass = _D("-1")
        max_out_mass = _D("-1")
        dominant_in = ""
        dominant_out = ""

        for mat in materials:
            dry_mass = self._dry_mass(mat)
            cc = self._resolve_carbon_content(mat)
            carbon_mass = self._round(dry_mass * cc)

            if mat.is_product or mat.is_by_product:
                total_out_mass += dry_mass
                total_out_carbon += carbon_mass
                output_count += 1
                if mat.is_by_product:
                    bp_count += 1
                if dry_mass > max_out_mass:
                    max_out_mass = dry_mass
                    dominant_out = mat.material_type
            else:
                total_in_mass += dry_mass
                total_in_carbon += carbon_mass
                input_count += 1
                if dry_mass > max_in_mass:
                    max_in_mass = dry_mass
                    dominant_in = mat.material_type

        total_in_mass = self._round(total_in_mass)
        total_out_mass = self._round(total_out_mass)
        total_in_carbon = self._round(total_in_carbon)
        total_out_carbon = self._round(total_out_carbon)

        residual = self._round(total_in_mass - total_out_mass)
        residual_pct = _D("0")
        if total_in_mass > _D("0"):
            residual_pct = self._round(
                (residual / total_in_mass) * _D("100")
            )

        prov_data = json.dumps(
            {
                "in_mass": str(total_in_mass),
                "out_mass": str(total_out_mass),
                "in_carbon": str(total_in_carbon),
                "out_carbon": str(total_out_carbon),
            },
            sort_keys=True,
        )
        prov_hash = self._hash(prov_data)

        summary = MaterialSummary(
            total_input_mass_tonnes=total_in_mass,
            total_output_mass_tonnes=total_out_mass,
            total_input_carbon_tonnes=total_in_carbon,
            total_output_carbon_tonnes=total_out_carbon,
            input_count=input_count,
            output_count=output_count,
            by_product_count=bp_count,
            dominant_input_type=dominant_in,
            dominant_output_type=dominant_out,
            mass_balance_residual_tonnes=residual,
            mass_balance_residual_pct=residual_pct,
            provenance_hash=prov_hash,
        )

        self._record_metric("material_summary")

        logger.info(
            "Material summary: %d inputs (%s t), %d outputs (%s t), "
            "residual=%s t (%.2f%%)",
            input_count,
            total_in_mass,
            output_count,
            total_out_mass,
            residual,
            float(residual_pct),
        )

        return summary

    # ------------------------------------------------------------------
    # Public API: verify_mass_balance
    # ------------------------------------------------------------------

    def verify_mass_balance(
        self,
        materials: List[MaterialInput],
        tolerance: Decimal = _D("0.05"),
    ) -> Tuple[bool, str]:
        """Verify that mass balance closes within a tolerance.

        Checks that:
            |mass_in - mass_out| / mass_in <= tolerance

        A closed mass balance indicates that all significant material
        flows have been accounted for.

        Args:
            materials: List of MaterialInput records.
            tolerance: Maximum acceptable fractional residual (0-1).
                Defaults to 0.05 (5%).

        Returns:
            Tuple of (passed: bool, message: str).
                passed is True if the mass balance closes within
                tolerance.
        """
        if tolerance < _D("0") or tolerance > _D("1"):
            raise ValueError(
                f"tolerance must be in [0, 1], got {tolerance}"
            )

        summary = self.get_material_summary(materials)
        total_in = summary.total_input_mass_tonnes
        total_out = summary.total_output_mass_tonnes

        if total_in <= _D("0"):
            return False, "No input mass recorded; cannot verify mass balance."

        residual_abs = abs(total_in - total_out)
        residual_frac = self._round(residual_abs / total_in)

        passed = residual_frac <= tolerance

        if passed:
            msg = (
                f"Mass balance PASSED: residual = {residual_frac} "
                f"({self._round(residual_frac * _D('100'))}%), "
                f"tolerance = {tolerance} "
                f"({self._round(tolerance * _D('100'))}%). "
                f"Input = {total_in} t, Output = {total_out} t."
            )
        else:
            msg = (
                f"Mass balance FAILED: residual = {residual_frac} "
                f"({self._round(residual_frac * _D('100'))}%), "
                f"exceeds tolerance = {tolerance} "
                f"({self._round(tolerance * _D('100'))}%). "
                f"Input = {total_in} t, Output = {total_out} t. "
                f"Review material flows for completeness."
            )

        self._record_metric(
            "verify_mass_balance",
            "success" if passed else "warning",
        )

        logger.info(
            "Mass balance verification: %s (residual=%s, tolerance=%s)",
            "PASSED" if passed else "FAILED",
            residual_frac,
            tolerance,
        )

        return passed, msg

    # ------------------------------------------------------------------
    # Public API: get_clinker_ratio
    # ------------------------------------------------------------------

    def get_clinker_ratio(
        self,
        cement_tonnes: Decimal,
        clinker_tonnes: Decimal,
    ) -> Decimal:
        """Calculate the clinker-to-cement mass ratio.

        Args:
            cement_tonnes: Total cement production in tonnes.
            clinker_tonnes: Total clinker consumed in tonnes.

        Returns:
            Clinker ratio as a Decimal (typically 0.50-1.00).

        Raises:
            ValueError: If cement_tonnes <= 0 or clinker_tonnes < 0.
            ValueError: If the computed ratio is outside [0.50, 1.00].
        """
        if cement_tonnes <= _D("0"):
            raise ValueError(
                f"cement_tonnes must be > 0, got {cement_tonnes}"
            )
        if clinker_tonnes < _D("0"):
            raise ValueError(
                f"clinker_tonnes must be >= 0, got {clinker_tonnes}"
            )

        ratio = self._round(clinker_tonnes / cement_tonnes)

        if ratio < CLINKER_TO_CEMENT_RATIO_MIN:
            logger.warning(
                "Clinker ratio %.4f is below minimum %.4f; "
                "verify input data",
                float(ratio),
                float(CLINKER_TO_CEMENT_RATIO_MIN),
            )
        if ratio > CLINKER_TO_CEMENT_RATIO_MAX:
            logger.warning(
                "Clinker ratio %.4f exceeds maximum %.4f; "
                "verify input data",
                float(ratio),
                float(CLINKER_TO_CEMENT_RATIO_MAX),
            )

        logger.debug(
            "Clinker ratio: %s t clinker / %s t cement = %s",
            clinker_tonnes,
            cement_tonnes,
            ratio,
        )

        return ratio

    # ------------------------------------------------------------------
    # Public API: calculate_ckd_correction
    # ------------------------------------------------------------------

    def calculate_ckd_correction(
        self,
        clinker_co2: Decimal,
        ckd_factor: Optional[Decimal] = None,
    ) -> Decimal:
        """Apply cement kiln dust (CKD) correction to clinker CO2.

        CKD is partially calcined raw material that exits the kiln
        before full calcination.  The CKD correction adds back the
        CO2 that would have been fully released if the CKD had
        remained in the kiln.

        Corrected_CO2 = clinker_co2 x ckd_factor

        Args:
            clinker_co2: CO2 emissions from clinker production in
                tonnes.
            ckd_factor: CKD correction factor (dimensionless,
                typically 1.00-1.05).  Defaults to 1.02 (2%).

        Returns:
            CKD-corrected CO2 emissions in tonnes.

        Raises:
            ValueError: If clinker_co2 < 0 or ckd_factor < 1.0.
        """
        if clinker_co2 < _D("0"):
            raise ValueError(
                f"clinker_co2 must be >= 0, got {clinker_co2}"
            )

        if ckd_factor is None:
            ckd_factor = CKD_CORRECTION_FACTOR_DEFAULT

        if ckd_factor < _D("1.0"):
            raise ValueError(
                f"ckd_factor must be >= 1.0, got {ckd_factor}"
            )
        if ckd_factor > _D("1.10"):
            logger.warning(
                "CKD factor %.4f exceeds typical maximum 1.10; "
                "verify value",
                float(ckd_factor),
            )

        corrected = self._round(clinker_co2 * ckd_factor)

        logger.debug(
            "CKD correction: %s tCO2 x %s = %s tCO2",
            clinker_co2,
            ckd_factor,
            corrected,
        )

        return corrected

    # ------------------------------------------------------------------
    # Public API: get_process_specific_balance
    # ------------------------------------------------------------------

    def get_process_specific_balance(
        self,
        process_type: str,
        materials: List[MaterialInput],
        production_route: str = "",
    ) -> CarbonBalance:
        """Dispatch to a process-specific carbon balance calculation.

        Routes the calculation to the appropriate process-specific
        method based on ``process_type``.

        Supported process types:
            - CEMENT / CEMENT_PRODUCTION
            - IRON_STEEL (with routes BF_BOF, EAF, DRI)
            - ALUMINUM / ALUMINUM_SMELTING (with routes PREBAKE,
              SODERBERG)
            - AMMONIA / AMMONIA_PRODUCTION
            - PETROCHEMICAL

        For unrecognised process types, falls back to the generic
        carbon balance.

        Args:
            process_type: Industrial process type string.
            materials: List of MaterialInput records.
            production_route: Production route (e.g. BF_BOF, EAF, DRI,
                PREBAKE, SODERBERG).

        Returns:
            CarbonBalance with process-specific adjustments.
        """
        pt_upper = process_type.upper().replace(" ", "_")

        dispatch: Dict[str, Any] = {
            "CEMENT": self._cement_balance,
            "CEMENT_PRODUCTION": self._cement_balance,
            "IRON_STEEL": self._iron_steel_balance,
            "ALUMINUM": self._aluminum_balance,
            "ALUMINUM_SMELTING": self._aluminum_balance,
            "AMMONIA": self._ammonia_balance,
            "AMMONIA_PRODUCTION": self._ammonia_balance,
            "PETROCHEMICAL": self._petrochemical_balance,
        }

        handler = dispatch.get(pt_upper)
        if handler is None:
            logger.info(
                "No process-specific balance for '%s'; "
                "using generic carbon balance",
                process_type,
            )
            return self.calculate_carbon_balance(materials)

        if pt_upper in ("IRON_STEEL",):
            return handler(materials, production_route)
        elif pt_upper in ("ALUMINUM", "ALUMINUM_SMELTING"):
            return handler(materials, production_route)
        else:
            return handler(materials)

    # ------------------------------------------------------------------
    # Process-specific: Cement
    # ------------------------------------------------------------------

    def _cement_balance(
        self,
        materials: List[MaterialInput],
    ) -> CarbonBalance:
        """Cement-specific carbon mass balance.

        Cement process emissions arise from:
        1. Carbonate decomposition in the raw meal (limestone, chalk,
           marl) during clinker production.
        2. Organic carbon in the raw meal.
        3. CKD (cement kiln dust) correction for partially calcined
           material lost as dust.

        The mass balance approach for cement:
            CO2 = (C_raw_materials - C_clinker - C_ckd_recirculated)
                  x 44/12 x CKD_correction

        Args:
            materials: List of MaterialInput records for the cement
                process.

        Returns:
            CarbonBalance with cement-specific adjustments.
        """
        trace: List[str] = [
            "Cement-specific carbon balance calculation"
        ]

        # Step 1: Generic carbon balance
        base_balance = self.calculate_carbon_balance(materials)

        # Step 2: Carbonate emissions
        carbonate_co2 = self.get_carbonate_emissions(materials)

        # Step 3: CKD correction
        total_co2 = base_balance.co2_emissions_tonnes
        if carbonate_co2 > _D("0"):
            total_co2 = max(total_co2, carbonate_co2)
            trace.append(
                f"  Using carbonate stoichiometric CO2: {carbonate_co2} tCO2"
            )

        corrected_co2 = self.calculate_ckd_correction(total_co2)
        trace.append(
            f"  CKD-corrected CO2: {corrected_co2} tCO2"
        )

        # Step 4: By-product credits
        by_products = [m for m in materials if m.is_by_product]
        credits = self.calculate_by_product_credits(by_products)
        total_credit = sum(
            (c.credit_co2_tonnes for c in credits), _D("0")
        )
        final_co2 = self._round(corrected_co2 - total_credit)
        if final_co2 < _D("0"):
            final_co2 = _D("0")

        trace.append(
            f"  By-product credit: {total_credit} tCO2"
        )
        trace.append(
            f"  Final cement CO2: {final_co2} tCO2"
        )

        # Combine traces
        combined_trace = base_balance.calculation_trace + trace

        prov_data = json.dumps(
            {
                "process": "cement",
                "base_co2": str(base_balance.co2_emissions_tonnes),
                "carbonate_co2": str(carbonate_co2),
                "corrected_co2": str(corrected_co2),
                "credit_co2": str(total_credit),
                "final_co2": str(final_co2),
            },
            sort_keys=True,
        )

        result = CarbonBalance(
            total_carbon_input_tonnes=base_balance.total_carbon_input_tonnes,
            total_carbon_output_tonnes=base_balance.total_carbon_output_tonnes,
            carbon_stock_change_tonnes=base_balance.carbon_stock_change_tonnes,
            net_carbon_emissions_tonnes=base_balance.net_carbon_emissions_tonnes,
            co2_emissions_tonnes=final_co2,
            material_details=base_balance.material_details,
            calculation_trace=combined_trace,
            provenance_hash=self._hash(prov_data),
            timestamp=self._now_iso(),
        )

        self._record_metric("cement_balance")
        logger.info("Cement balance: final CO2 = %s t", final_co2)

        return result

    # ------------------------------------------------------------------
    # Process-specific: Iron and Steel
    # ------------------------------------------------------------------

    def _iron_steel_balance(
        self,
        materials: List[MaterialInput],
        production_route: str = "",
    ) -> CarbonBalance:
        """Iron and steel process carbon mass balance.

        Iron and steel emissions depend on the production route:

        BF-BOF (Blast Furnace - Basic Oxygen Furnace):
            Carbon inputs: coke, coal, iron ore (minor), limestone flux
            Carbon outputs: pig iron/steel, slag, BF gas
            CO2 = (C_coke + C_coal + C_flux - C_steel - C_slag) x 44/12

        EAF (Electric Arc Furnace):
            Carbon inputs: scrap metal (minor), electrodes, DRI charge
            Carbon outputs: steel product
            CO2 = (C_scrap + C_electrode + C_DRI - C_steel) x 44/12

        DRI (Direct Reduced Iron):
            Carbon inputs: natural gas feedstock, iron ore (minor)
            Carbon outputs: DRI product
            CO2 = (C_gas_feedstock - C_DRI) x 44/12

        Args:
            materials: List of MaterialInput records.
            production_route: One of BF_BOF, EAF, DRI.  Defaults to
                BF_BOF if not specified.

        Returns:
            CarbonBalance with route-specific adjustments.
        """
        route = production_route.upper().replace(" ", "_")
        if route not in ("BF_BOF", "EAF", "DRI"):
            route = "BF_BOF"

        trace: List[str] = [
            f"Iron/Steel carbon balance: route={route}"
        ]

        # Generic balance
        base = self.calculate_carbon_balance(materials)

        # Route-specific adjustments
        if route == "BF_BOF":
            # BF-BOF: Apply slag carbon credit
            slag_credit = _D("0")
            for mat in materials:
                if mat.is_by_product and "SLAG" in mat.material_type.upper():
                    dry_mass = self._dry_mass(mat)
                    cc = self._resolve_carbon_content(mat)
                    slag_credit += self._round(
                        dry_mass * cc * CARBON_TO_CO2_EXACT
                    )

            adjusted_co2 = self._round(
                base.co2_emissions_tonnes - slag_credit
            )
            if adjusted_co2 < _D("0"):
                adjusted_co2 = _D("0")

            trace.append(
                f"  BF-BOF slag carbon credit: {slag_credit} tCO2"
            )
            trace.append(
                f"  Adjusted CO2: {adjusted_co2} tCO2"
            )
            final_co2 = adjusted_co2

        elif route == "EAF":
            # EAF: Lower carbon input from scrap; electrode consumption
            # dominates process emissions
            electrode_carbon = _D("0")
            for mat in materials:
                mat_type = mat.material_type.upper()
                if "ELECTRODE" in mat_type or "GRAPHITE" in mat_type:
                    dry_mass = self._dry_mass(mat)
                    cc = self._resolve_carbon_content(mat)
                    if cc <= _D("0"):
                        cc = _D("0.850")  # graphite electrode default
                    electrode_carbon += self._round(dry_mass * cc)

            if electrode_carbon > _D("0"):
                electrode_co2 = self._round(
                    electrode_carbon * CARBON_TO_CO2_EXACT
                )
                trace.append(
                    f"  EAF electrode carbon: {electrode_carbon} tC "
                    f"-> {electrode_co2} tCO2"
                )
                # Use the larger of generic balance and electrode calc
                final_co2 = max(base.co2_emissions_tonnes, electrode_co2)
            else:
                final_co2 = base.co2_emissions_tonnes

        else:
            # DRI: Natural gas feedstock carbon
            gas_carbon = _D("0")
            for mat in materials:
                mat_type = mat.material_type.upper()
                if "NATURAL_GAS" in mat_type or "GAS" in mat_type:
                    if not mat.is_product and not mat.is_by_product:
                        dry_mass = self._dry_mass(mat)
                        cc = self._resolve_carbon_content(mat)
                        fox = mat.fraction_oxidized
                        gas_carbon += self._round(dry_mass * cc * fox)

            if gas_carbon > _D("0"):
                gas_co2 = self._round(gas_carbon * CARBON_TO_CO2_EXACT)
                trace.append(
                    f"  DRI gas carbon: {gas_carbon} tC "
                    f"-> {gas_co2} tCO2"
                )
                final_co2 = max(base.co2_emissions_tonnes, gas_co2)
            else:
                final_co2 = base.co2_emissions_tonnes

        final_co2 = self._round(final_co2)
        combined_trace = base.calculation_trace + trace

        prov_data = json.dumps(
            {
                "process": "iron_steel",
                "route": route,
                "base_co2": str(base.co2_emissions_tonnes),
                "final_co2": str(final_co2),
            },
            sort_keys=True,
        )

        result = CarbonBalance(
            total_carbon_input_tonnes=base.total_carbon_input_tonnes,
            total_carbon_output_tonnes=base.total_carbon_output_tonnes,
            carbon_stock_change_tonnes=base.carbon_stock_change_tonnes,
            net_carbon_emissions_tonnes=base.net_carbon_emissions_tonnes,
            co2_emissions_tonnes=final_co2,
            material_details=base.material_details,
            calculation_trace=combined_trace,
            provenance_hash=self._hash(prov_data),
            timestamp=self._now_iso(),
        )

        self._record_metric("iron_steel_balance")
        logger.info(
            "Iron/Steel balance (%s): CO2 = %s t", route, final_co2,
        )

        return result

    # ------------------------------------------------------------------
    # Process-specific: Aluminum
    # ------------------------------------------------------------------

    def _aluminum_balance(
        self,
        materials: List[MaterialInput],
        production_route: str = "",
    ) -> CarbonBalance:
        """Aluminum smelting carbon mass balance.

        Aluminum process CO2 emissions arise from carbon anode
        consumption during electrolytic reduction of alumina (Al2O3).

        Prebake:
            CO2 = anode_consumption x anode_carbon_content x 44/12
            Typical: 0.45 t anode/t Al, 85% carbon -> ~1.4 tCO2/t Al

        Soderberg:
            CO2 = paste_consumption x paste_carbon_content x 44/12
            Typical: 0.55 t paste/t Al, 80% carbon -> ~1.61 tCO2/t Al

        Note: PFC emissions (CF4, C2F6) from anode effects are
        calculated separately by the EmissionCalculatorEngine.

        Args:
            materials: List of MaterialInput records.
            production_route: One of PREBAKE, SODERBERG.

        Returns:
            CarbonBalance with aluminum-specific adjustments.
        """
        route = production_route.upper().replace(" ", "_")
        if route not in ("PREBAKE", "SODERBERG", "SODERBERG_VSS",
                         "SODERBERG_HSS", "CWPB", "SWPB"):
            route = "PREBAKE"

        # Normalise Soderberg variants
        is_soderberg = route.startswith("SODERBERG") or route in ("CWPB", "SWPB")

        trace: List[str] = [
            f"Aluminum carbon balance: route={route}"
        ]

        # Generic balance first
        base = self.calculate_carbon_balance(materials)

        # Calculate anode/paste carbon
        anode_carbon = _D("0")
        aluminum_output = _D("0")

        for mat in materials:
            mat_upper = mat.material_type.upper()
            if mat.is_product and ("ALUMINUM" in mat_upper or "ALUMINIUM" in mat_upper):
                aluminum_output += self._dry_mass(mat)
            elif not mat.is_product and not mat.is_by_product:
                if "ANODE" in mat_upper or "PASTE" in mat_upper:
                    dry_mass = self._dry_mass(mat)
                    cc = self._resolve_carbon_content(mat)
                    if cc <= _D("0"):
                        if is_soderberg:
                            cc = SODERBERG_PASTE_CARBON_CONTENT
                        else:
                            cc = ANODE_CARBON_CONTENT
                    anode_carbon += self._round(dry_mass * cc)

        # If no explicit anode materials but we have aluminum output,
        # estimate anode consumption from default factors
        if anode_carbon <= _D("0") and aluminum_output > _D("0"):
            if is_soderberg:
                anode_mass = self._round(
                    aluminum_output * SODERBERG_PASTE_CONSUMPTION
                )
                anode_carbon = self._round(
                    anode_mass * SODERBERG_PASTE_CARBON_CONTENT
                )
                trace.append(
                    f"  Estimated Soderberg paste: {anode_mass} t, "
                    f"carbon: {anode_carbon} tC"
                )
            else:
                anode_mass = self._round(
                    aluminum_output * PREBAKE_ANODE_CONSUMPTION
                )
                anode_carbon = self._round(
                    anode_mass * ANODE_CARBON_CONTENT
                )
                trace.append(
                    f"  Estimated prebake anode: {anode_mass} t, "
                    f"carbon: {anode_carbon} tC"
                )

        if anode_carbon > _D("0"):
            anode_co2 = self._round(anode_carbon * CARBON_TO_CO2_EXACT)
            trace.append(
                f"  Anode carbon: {anode_carbon} tC -> {anode_co2} tCO2"
            )
            final_co2 = max(base.co2_emissions_tonnes, anode_co2)
        else:
            final_co2 = base.co2_emissions_tonnes

        final_co2 = self._round(final_co2)
        combined_trace = base.calculation_trace + trace

        prov_data = json.dumps(
            {
                "process": "aluminum",
                "route": route,
                "anode_carbon": str(anode_carbon),
                "base_co2": str(base.co2_emissions_tonnes),
                "final_co2": str(final_co2),
            },
            sort_keys=True,
        )

        result = CarbonBalance(
            total_carbon_input_tonnes=base.total_carbon_input_tonnes,
            total_carbon_output_tonnes=base.total_carbon_output_tonnes,
            carbon_stock_change_tonnes=base.carbon_stock_change_tonnes,
            net_carbon_emissions_tonnes=base.net_carbon_emissions_tonnes,
            co2_emissions_tonnes=final_co2,
            material_details=base.material_details,
            calculation_trace=combined_trace,
            provenance_hash=self._hash(prov_data),
            timestamp=self._now_iso(),
        )

        self._record_metric("aluminum_balance")
        logger.info(
            "Aluminum balance (%s): CO2 = %s t", route, final_co2,
        )

        return result

    # ------------------------------------------------------------------
    # Process-specific: Ammonia
    # ------------------------------------------------------------------

    def _ammonia_balance(
        self,
        materials: List[MaterialInput],
    ) -> CarbonBalance:
        """Ammonia production carbon mass balance.

        Ammonia is produced primarily via steam methane reforming (SMR)
        of natural gas:
            CH4 + H2O -> CO + 3H2  (reforming)
            CO + H2O  -> CO2 + H2  (water-gas shift)

        The carbon in the natural gas feedstock is converted almost
        entirely to CO2 (99.5% oxidation fraction).

        CO2 = feed_gas_quantity x carbon_content x fraction_oxidized x 44/12

        If CO2 is captured for urea production, that credit is applied
        via by-product accounting.

        Args:
            materials: List of MaterialInput records.

        Returns:
            CarbonBalance with ammonia-specific adjustments.
        """
        trace: List[str] = [
            "Ammonia carbon balance (steam methane reforming)"
        ]

        # Identify natural gas feedstock
        gas_carbon = _D("0")
        for mat in materials:
            if mat.is_product or mat.is_by_product:
                continue
            mat_upper = mat.material_type.upper()
            if "NATURAL_GAS" in mat_upper or "METHANE" in mat_upper:
                dry_mass = self._dry_mass(mat)
                cc = self._resolve_carbon_content(mat)
                if cc <= _D("0"):
                    cc = DEFAULT_CARBON_CONTENT["NATURAL_GAS_FEEDSTOCK"]
                fox = mat.fraction_oxidized
                if fox >= _D("1"):
                    fox = SMR_CARBON_OXIDATION_FRACTION
                gas_carbon += self._round(dry_mass * cc * fox)
                trace.append(
                    f"  NG feedstock: {dry_mass} t x {cc} x {fox} "
                    f"= {self._round(dry_mass * cc * fox)} tC"
                )

        # Generic balance
        base = self.calculate_carbon_balance(materials)

        if gas_carbon > _D("0"):
            gas_co2 = self._round(gas_carbon * CARBON_TO_CO2_EXACT)
            trace.append(
                f"  Feedstock CO2: {gas_co2} tCO2"
            )
            # By-product credits (e.g. CO2 captured for urea)
            by_products = [m for m in materials if m.is_by_product]
            credits = self.calculate_by_product_credits(by_products)
            total_credit = sum(
                (c.credit_co2_tonnes for c in credits), _D("0")
            )

            final_co2 = self._round(gas_co2 - total_credit)
            if final_co2 < _D("0"):
                final_co2 = _D("0")

            trace.append(f"  CO2 credit (urea etc.): {total_credit} tCO2")
            trace.append(f"  Final ammonia CO2: {final_co2} tCO2")
        else:
            final_co2 = base.co2_emissions_tonnes

        final_co2 = self._round(final_co2)
        combined_trace = base.calculation_trace + trace

        prov_data = json.dumps(
            {
                "process": "ammonia",
                "gas_carbon": str(gas_carbon),
                "base_co2": str(base.co2_emissions_tonnes),
                "final_co2": str(final_co2),
            },
            sort_keys=True,
        )

        result = CarbonBalance(
            total_carbon_input_tonnes=base.total_carbon_input_tonnes,
            total_carbon_output_tonnes=base.total_carbon_output_tonnes,
            carbon_stock_change_tonnes=base.carbon_stock_change_tonnes,
            net_carbon_emissions_tonnes=base.net_carbon_emissions_tonnes,
            co2_emissions_tonnes=final_co2,
            material_details=base.material_details,
            calculation_trace=combined_trace,
            provenance_hash=self._hash(prov_data),
            timestamp=self._now_iso(),
        )

        self._record_metric("ammonia_balance")
        logger.info("Ammonia balance: CO2 = %s t", final_co2)

        return result

    # ------------------------------------------------------------------
    # Process-specific: Petrochemical
    # ------------------------------------------------------------------

    def _petrochemical_balance(
        self,
        materials: List[MaterialInput],
    ) -> CarbonBalance:
        """Petrochemical cracker carbon mass balance.

        Petrochemical processes (ethylene/propylene production) crack
        hydrocarbon feedstocks (naphtha, ethane, propane, LPG):

        Feedstock carbon splits between:
            1. Primary products (ethylene, propylene, etc.) - retained
            2. Co-products (hydrogen, C4+, aromatics) - retained
            3. Fuel gas (burned) - CO2 emissions (combustion scope)
            4. Process CO2 - direct emissions (this calculation)

        Mass balance:
            CO2_process = (C_feedstock - C_products - C_co_products
                          - C_fuel_gas) x 44/12

        Typical feedstock-to-product carbon splits:
            Naphtha cracker: ~60% in products, ~35% in fuel gas, ~5% process
            Ethane cracker:  ~80% in products, ~15% in fuel gas, ~5% process

        Args:
            materials: List of MaterialInput records.

        Returns:
            CarbonBalance with petrochemical-specific adjustments.
        """
        trace: List[str] = [
            "Petrochemical cracker carbon balance"
        ]

        # Identify feedstock carbon
        feedstock_carbon = _D("0")
        for mat in materials:
            if mat.is_product or mat.is_by_product:
                continue
            mat_upper = mat.material_type.upper()
            if mat_upper in ("NAPHTHA", "ETHANE", "NATURAL_GAS_FEEDSTOCK"):
                dry_mass = self._dry_mass(mat)
                cc = self._resolve_carbon_content(mat)
                fox = mat.fraction_oxidized
                feedstock_carbon += self._round(dry_mass * cc * fox)
                trace.append(
                    f"  Feedstock {mat.material_type}: {dry_mass} t x "
                    f"{cc} x {fox} = "
                    f"{self._round(dry_mass * cc * fox)} tC"
                )

        # Identify product carbon
        product_carbon = _D("0")
        for mat in materials:
            if not (mat.is_product or mat.is_by_product):
                continue
            dry_mass = self._dry_mass(mat)
            cc = self._resolve_carbon_content(mat)
            product_carbon += self._round(dry_mass * cc)
            trace.append(
                f"  Product {mat.material_type}: {dry_mass} t x {cc} "
                f"= {self._round(dry_mass * cc)} tC"
            )

        # Generic balance
        base = self.calculate_carbon_balance(materials)

        if feedstock_carbon > _D("0"):
            net_process_carbon = self._round(
                feedstock_carbon - product_carbon
            )
            if net_process_carbon < _D("0"):
                net_process_carbon = _D("0")

            process_co2 = self._round(
                net_process_carbon * CARBON_TO_CO2_EXACT
            )
            trace.append(
                f"  Net process carbon: {net_process_carbon} tC"
            )
            trace.append(
                f"  Process CO2: {process_co2} tCO2"
            )
            final_co2 = process_co2
        else:
            final_co2 = base.co2_emissions_tonnes

        final_co2 = self._round(final_co2)
        combined_trace = base.calculation_trace + trace

        prov_data = json.dumps(
            {
                "process": "petrochemical",
                "feedstock_carbon": str(feedstock_carbon),
                "product_carbon": str(product_carbon),
                "base_co2": str(base.co2_emissions_tonnes),
                "final_co2": str(final_co2),
            },
            sort_keys=True,
        )

        result = CarbonBalance(
            total_carbon_input_tonnes=base.total_carbon_input_tonnes,
            total_carbon_output_tonnes=base.total_carbon_output_tonnes,
            carbon_stock_change_tonnes=base.carbon_stock_change_tonnes,
            net_carbon_emissions_tonnes=base.net_carbon_emissions_tonnes,
            co2_emissions_tonnes=final_co2,
            material_details=base.material_details,
            calculation_trace=combined_trace,
            provenance_hash=self._hash(prov_data),
            timestamp=self._now_iso(),
        )

        self._record_metric("petrochemical_balance")
        logger.info("Petrochemical balance: CO2 = %s t", final_co2)

        return result

    # ------------------------------------------------------------------
    # Utility: get tracked materials
    # ------------------------------------------------------------------

    def get_tracked_materials(self) -> Dict[str, MaterialInput]:
        """Return a copy of the tracked materials registry.

        Returns:
            Dict mapping material_id to MaterialInput.
        """
        with self._lock:
            return dict(self._tracked_materials)

    def clear_tracked_materials(self) -> int:
        """Clear all tracked materials and return the count cleared.

        Returns:
            Number of materials removed from the registry.
        """
        with self._lock:
            count = len(self._tracked_materials)
            self._tracked_materials.clear()

        logger.info("Cleared %d tracked materials", count)
        return count

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise engine state to a dictionary.

        Returns:
            Dict with engine configuration and tracked material count.
        """
        with self._lock:
            return {
                "engine": "MaterialBalanceEngine",
                "precision": self._precision,
                "tracked_material_count": len(self._tracked_materials),
                "constants": {
                    "CARBON_TO_CO2": str(CARBON_TO_CO2),
                    "CKD_CORRECTION_FACTOR_DEFAULT": str(
                        CKD_CORRECTION_FACTOR_DEFAULT
                    ),
                    "CLINKER_TO_CEMENT_RATIO_DEFAULT": str(
                        CLINKER_TO_CEMENT_RATIO_DEFAULT
                    ),
                    "BF_SLAG_RATIO": str(BF_SLAG_RATIO),
                    "EAF_SCRAP_FRACTION": str(EAF_SCRAP_FRACTION),
                },
            }
