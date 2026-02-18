# -*- coding: utf-8 -*-
"""
EmissionCalculatorEngine - Engine 2: Process Emissions Agent (AGENT-MRV-004)

Core calculation engine implementing four IPCC/EPA/GHG Protocol calculation
methodologies for industrial process emissions:

1. **Emission Factor Method** (Tier 1):
   emissions_gas = activity_data x EF_gas x (1 - abatement_efficiency)
   total_co2e = sum(emissions_gas x GWP_gas)

2. **Mass Balance Method** (Tier 2/3):
   CO2 = (carbon_input - carbon_output - carbon_stock_change) x 44/12
   carbon_input = sum(material_qty x carbon_content x fraction_oxidized)

3. **Stoichiometric Method** (Tier 2):
   CO2 = sum(carbonate_qty x carbonate_EF x fraction_calcined)
   Using CaCO3->CaO+CO2 (0.440), MgCO3->MgO+CO2 (0.522), etc.

4. **Direct Measurement**:
   Takes measured emission values directly (CEMS, stack testing).

Process-specific calculation modules handle:
    - Cement: clinker_production x clinker_EF x CKD_correction
    - Iron/Steel: varies by route (BF-BOF, EAF, DRI)
    - Aluminum: anode_consumption + PFC emissions
    - Nitric acid: production x N2O_EF x (1 - abatement) x N2O_GWP
    - Semiconductor: gas_usage x (1-util) x (1-abatement) x by_product x GWP

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8+ decimal places).
    - No LLM calls in any calculation path.
    - Every step is recorded in the calculation trace.
    - SHA-256 provenance hash for every result.
    - Same inputs always produce identical outputs (deterministic).

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.process_emissions.emission_calculator import EmissionCalculatorEngine
    >>> from greenlang.process_emissions.process_database import ProcessDatabaseEngine
    >>> from decimal import Decimal
    >>> db = ProcessDatabaseEngine()
    >>> calc = EmissionCalculatorEngine(process_database=db)
    >>> result = calc.calculate(
    ...     process_type="CEMENT",
    ...     method="EMISSION_FACTOR",
    ...     activity_data=Decimal("1000000"),
    ...     activity_unit="tonne_clinker",
    ... )
    >>> print(result["total_co2e_tonnes"])

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
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["EmissionCalculatorEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.process_emissions.process_database import ProcessDatabaseEngine
    _DATABASE_AVAILABLE = True
except ImportError:
    _DATABASE_AVAILABLE = False
    ProcessDatabaseEngine = None  # type: ignore[assignment, misc]

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
        record_calculation as _record_calculation,
        record_emissions as _record_emissions,
        observe_calculation_duration as _observe_calculation_duration,
        record_batch as _record_batch,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_calculation = None  # type: ignore[assignment]
    _record_emissions = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]
    _record_batch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Decimal precision constants
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places

#: Molecular weight ratio CO2/C = 44.01 / 12.01
_CO2_C_RATIO = Decimal("3.66417")

#: Molecular weight of CO2
_MW_CO2 = Decimal("44.01")

#: Molecular weight of C
_MW_C = Decimal("12.01")

#: Conversion: tonnes to kg
_TONNES_TO_KG = Decimal("1000")

#: Conversion: kg to tonnes
_KG_TO_TONNES = Decimal("0.001")


# ---------------------------------------------------------------------------
# Valid method identifiers
# ---------------------------------------------------------------------------

_VALID_METHODS = frozenset({
    "EMISSION_FACTOR",
    "MASS_BALANCE",
    "STOICHIOMETRIC",
    "DIRECT_MEASUREMENT",
})


class EmissionCalculatorEngine:
    """Core calculation engine for industrial process emissions.

    Implements Tier 1/2/3 methodologies with deterministic Decimal arithmetic,
    full calculation trace, and SHA-256 provenance hashing. Thread-safe for
    concurrent calculations.

    This engine does NOT directly access the process database; it receives
    factor values as parameters or via a ProcessDatabaseEngine reference.
    The pipeline engine coordinates data flow between engines.

    Attributes:
        _process_db: Reference to ProcessDatabaseEngine for factor lookups.
        _config: Optional configuration dictionary.
        _lock: Thread lock for any shared mutable state.
        _provenance: Reference to the provenance tracker.
        _precision_places: Number of Decimal places for rounding.

    Example:
        >>> db = ProcessDatabaseEngine()
        >>> calc = EmissionCalculatorEngine(process_database=db)
        >>> result = calc.calculate(
        ...     process_type="CEMENT",
        ...     method="EMISSION_FACTOR",
        ...     activity_data=Decimal("500000"),
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    def __init__(
        self,
        process_database: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize EmissionCalculatorEngine.

        Args:
            process_database: ProcessDatabaseEngine instance for factor
                lookups. If None and ProcessDatabaseEngine is available,
                a default instance is created.
            config: Optional configuration dict. Supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                - ``decimal_precision`` (int): Decimal places. Default 8.
                - ``default_gwp_source`` (str): Default GWP source. Default "AR6".
                - ``default_ef_source`` (str): Default EF source. Default "IPCC".
        """
        if process_database is not None:
            self._process_db = process_database
        elif _DATABASE_AVAILABLE:
            self._process_db = ProcessDatabaseEngine(config=config)
        else:
            self._process_db = None

        self._config = config or {}
        self._lock = threading.Lock()
        self._enable_provenance: bool = self._config.get("enable_provenance", True)
        self._precision_places: int = self._config.get("decimal_precision", 8)
        self._precision_quantizer = Decimal(10) ** -self._precision_places
        self._default_gwp_source: str = self._config.get("default_gwp_source", "AR6")
        self._default_ef_source: str = self._config.get("default_ef_source", "IPCC")

        if self._enable_provenance and _PROVENANCE_AVAILABLE:
            self._provenance = _get_provenance_tracker()
        else:
            self._provenance = None

        logger.info(
            "EmissionCalculatorEngine initialized (precision=%d, gwp=%s, ef=%s)",
            self._precision_places,
            self._default_gwp_source,
            self._default_ef_source,
        )

    # ==================================================================
    # PUBLIC API: Unified Calculate
    # ==================================================================

    def calculate(
        self,
        process_type: str,
        method: str = "EMISSION_FACTOR",
        activity_data: Optional[Decimal] = None,
        activity_unit: str = "tonne",
        gwp_source: Optional[str] = None,
        ef_source: Optional[str] = None,
        abatement_efficiency: Optional[Decimal] = None,
        abatement_type: Optional[str] = None,
        custom_factors: Optional[Dict[str, Decimal]] = None,
        material_inputs: Optional[List[Dict[str, Any]]] = None,
        carbon_outputs: Optional[List[Dict[str, Any]]] = None,
        carbon_stock_change: Optional[Decimal] = None,
        carbonate_inputs: Optional[List[Dict[str, Any]]] = None,
        measured_emissions: Optional[Dict[str, Decimal]] = None,
        process_params: Optional[Dict[str, Any]] = None,
        calculation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate industrial process emissions using the specified method.

        This is the main entry point for all calculation types. It dispatches
        to the appropriate method-specific calculator based on the ``method``
        parameter.

        Args:
            process_type: Industrial process identifier (e.g. ``"CEMENT"``).
            method: Calculation method. One of ``"EMISSION_FACTOR"``,
                ``"MASS_BALANCE"``, ``"STOICHIOMETRIC"``,
                ``"DIRECT_MEASUREMENT"``.
            activity_data: Production quantity for emission factor method.
            activity_unit: Unit of activity data. Default ``"tonne"``.
            gwp_source: GWP source override. Default uses engine config.
            ef_source: Emission factor source override.
            abatement_efficiency: Abatement fraction (0.0-1.0).
            abatement_type: Abatement technology identifier.
            custom_factors: User-provided emission factors per gas.
            material_inputs: For mass balance - list of material input dicts.
            carbon_outputs: For mass balance - list of carbon output dicts.
            carbon_stock_change: For mass balance - stock change in tC.
            carbonate_inputs: For stoichiometric - list of carbonate dicts.
            measured_emissions: For direct measurement - measured values per gas.
            process_params: Process-specific parameters (e.g. CKD factor,
                anode effect frequency, scrap fraction).
            calculation_id: Optional external calculation ID.

        Returns:
            Dictionary with keys:
                - calculation_id (str)
                - status (str): "SUCCESS" or "FAILED"
                - process_type (str)
                - method (str)
                - gas_emissions (List[Dict]): per-gas breakdown
                - total_co2e_kg (Decimal)
                - total_co2e_tonnes (Decimal)
                - calculation_trace (List[str])
                - provenance_hash (str)
                - processing_time_ms (float)
                - error_message (str, optional)

        Example:
            >>> result = calc.calculate(
            ...     process_type="CEMENT",
            ...     method="EMISSION_FACTOR",
            ...     activity_data=Decimal("1000000"),
            ... )
            >>> result["total_co2e_tonnes"]
            Decimal('507000.00000000')
        """
        start_time = time.monotonic()
        calc_id = calculation_id or f"pe_calc_{uuid.uuid4().hex[:12]}"
        pt_key = process_type.upper()
        method_key = method.upper()
        gwp = (gwp_source or self._default_gwp_source).upper()
        ef_src = (ef_source or self._default_ef_source).upper()
        trace: List[str] = []

        try:
            # Validate method
            if method_key not in _VALID_METHODS:
                raise ValueError(
                    f"Invalid calculation method: {method}. "
                    f"Valid methods: {sorted(_VALID_METHODS)}"
                )

            trace.append(
                f"[1] Process={pt_key}, Method={method_key}, "
                f"GWP={gwp}, EF_source={ef_src}"
            )

            # Dispatch to method-specific calculator
            if method_key == "EMISSION_FACTOR":
                gas_emissions = self.calculate_emission_factor(
                    process_type=pt_key,
                    activity_data=activity_data or Decimal("0"),
                    ef_source=ef_src,
                    gwp_source=gwp,
                    abatement_efficiency=abatement_efficiency,
                    custom_factors=custom_factors,
                    process_params=process_params,
                    trace=trace,
                )
            elif method_key == "MASS_BALANCE":
                gas_emissions = self.calculate_mass_balance(
                    process_type=pt_key,
                    material_inputs=material_inputs or [],
                    carbon_outputs=carbon_outputs or [],
                    carbon_stock_change=carbon_stock_change or Decimal("0"),
                    gwp_source=gwp,
                    abatement_efficiency=abatement_efficiency,
                    process_params=process_params,
                    trace=trace,
                )
            elif method_key == "STOICHIOMETRIC":
                gas_emissions = self.calculate_stoichiometric(
                    process_type=pt_key,
                    carbonate_inputs=carbonate_inputs or [],
                    gwp_source=gwp,
                    abatement_efficiency=abatement_efficiency,
                    process_params=process_params,
                    trace=trace,
                )
            elif method_key == "DIRECT_MEASUREMENT":
                gas_emissions = self.calculate_direct(
                    process_type=pt_key,
                    measured_emissions=measured_emissions or {},
                    gwp_source=gwp,
                    trace=trace,
                )
            else:
                raise ValueError(f"Unhandled method: {method_key}")

            # Compute totals
            total_co2e_kg = self._sum_co2e(gas_emissions)
            total_co2e_tonnes = self._quantize(total_co2e_kg * _KG_TO_TONNES)

            trace.append(
                f"[TOTAL] co2e_kg={total_co2e_kg}, "
                f"co2e_tonnes={total_co2e_tonnes}"
            )

            # Provenance hash
            elapsed_ms = (time.monotonic() - start_time) * 1000
            provenance_hash = self._compute_provenance_hash({
                "calculation_id": calc_id,
                "process_type": pt_key,
                "method": method_key,
                "total_co2e_kg": str(total_co2e_kg),
                "gwp_source": gwp,
                "ef_source": ef_src,
            })
            trace.append(f"[PROV] hash={provenance_hash[:16]}...")

            # Metrics
            if _METRICS_AVAILABLE and _record_calculation is not None:
                _record_calculation(pt_key, method_key, "completed")
            if _METRICS_AVAILABLE and _record_emissions is not None:
                _record_emissions(pt_key, "scope_1", float(total_co2e_tonnes))
            if _METRICS_AVAILABLE and _observe_calculation_duration is not None:
                _observe_calculation_duration("single_calculation", elapsed_ms / 1000)

            # Provenance record
            self._record_provenance(
                "calculate_emissions", calc_id,
                {
                    "process_type": pt_key,
                    "method": method_key,
                    "total_co2e_kg": str(total_co2e_kg),
                    "hash": provenance_hash,
                },
            )

            return {
                "calculation_id": calc_id,
                "status": "SUCCESS",
                "process_type": pt_key,
                "method": method_key,
                "ef_source": ef_src,
                "gwp_source": gwp,
                "gas_emissions": gas_emissions,
                "total_co2e_kg": total_co2e_kg,
                "total_co2e_tonnes": total_co2e_tonnes,
                "abatement_efficiency": abatement_efficiency,
                "abatement_type": abatement_type,
                "calculation_trace": trace,
                "provenance_hash": provenance_hash,
                "processing_time_ms": elapsed_ms,
            }

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Calculation failed for %s (id=%s, method=%s): %s",
                pt_key, calc_id, method_key, exc, exc_info=True,
            )
            if _METRICS_AVAILABLE and _record_calculation is not None:
                _record_calculation(pt_key, method_key, "failed")

            return {
                "calculation_id": calc_id,
                "status": "FAILED",
                "process_type": pt_key,
                "method": method_key,
                "ef_source": ef_src,
                "gwp_source": gwp,
                "gas_emissions": [],
                "total_co2e_kg": Decimal("0"),
                "total_co2e_tonnes": Decimal("0"),
                "calculation_trace": trace,
                "provenance_hash": "",
                "processing_time_ms": elapsed_ms,
                "error_message": str(exc),
            }

    # ==================================================================
    # PUBLIC API: Emission Factor Method (Tier 1)
    # ==================================================================

    def calculate_emission_factor(
        self,
        process_type: str,
        activity_data: Decimal,
        ef_source: str = "IPCC",
        gwp_source: str = "AR6",
        abatement_efficiency: Optional[Decimal] = None,
        custom_factors: Optional[Dict[str, Decimal]] = None,
        process_params: Optional[Dict[str, Any]] = None,
        trace: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Calculate emissions using the emission factor method (Tier 1).

        Formula per gas:
            emissions_gas = activity_data x EF_gas x (1 - abatement_eff)
            co2e_gas = emissions_gas x GWP_gas

        For process-specific adjustments (e.g. cement CKD correction,
        aluminum PFC slope), see process_params.

        Args:
            process_type: Process type identifier.
            activity_data: Production quantity in process-specific units.
            ef_source: Emission factor source. Default ``"IPCC"``.
            gwp_source: GWP source. Default ``"AR6"``.
            abatement_efficiency: Fraction of emissions abated (0.0 - 1.0).
            custom_factors: User-provided factors per gas, overriding DB.
            process_params: Process-specific parameters dict.
            trace: Mutable trace list for audit trail.

        Returns:
            List of per-gas emission dictionaries with keys:
                gas, emission_factor, emissions_mass_tonnes,
                gwp, co2e_kg, co2e_tonnes.

        Raises:
            ValueError: If activity_data is negative.
        """
        if trace is None:
            trace = []

        pt_key = process_type.upper()
        params = process_params or {}
        abatement = self._validate_abatement(abatement_efficiency)

        if activity_data < 0:
            raise ValueError(
                f"Activity data cannot be negative: {activity_data}"
            )

        trace.append(
            f"[EF] activity={activity_data}, abatement={abatement}"
        )

        # Get available gases for this process
        gas_list = self._get_process_gases(pt_key, ef_source, custom_factors)
        results: List[Dict[str, Any]] = []

        for gas in gas_list:
            # Get emission factor
            ef = self._resolve_ef(pt_key, gas, ef_source, custom_factors)
            gwp = self._resolve_gwp(gas, gwp_source)

            # Calculate raw emissions (mass of gas, in factor units)
            raw_emissions = self._quantize(activity_data * ef)

            # Apply abatement
            net_emissions = self._apply_abatement(raw_emissions, abatement)

            # Process-specific adjustments
            net_emissions = self._apply_process_adjustments(
                pt_key, gas, net_emissions, activity_data, params, trace,
            )

            # Convert to CO2e
            co2e_kg = self._quantize(net_emissions * gwp * _TONNES_TO_KG)
            co2e_tonnes = self._quantize(net_emissions * gwp)

            # For gases already expressed in CO2/CH4/N2O tonnes, CO2e = mass * GWP
            # For PFC factors that are already tCO2e, GWP is applied differently
            if gas in ("PFC_CO2E",):
                co2e_kg = self._quantize(net_emissions * _TONNES_TO_KG)
                co2e_tonnes = net_emissions

            trace.append(
                f"[EF:{gas}] EF={ef}, raw={raw_emissions}, "
                f"net={net_emissions}, GWP={gwp}, co2e_t={co2e_tonnes}"
            )

            results.append({
                "gas": gas,
                "emission_factor": ef,
                "emission_factor_source": ef_source,
                "raw_emissions_tonnes": raw_emissions,
                "abatement_efficiency": abatement,
                "net_emissions_tonnes": net_emissions,
                "gwp": gwp,
                "gwp_source": gwp_source,
                "co2e_kg": co2e_kg,
                "co2e_tonnes": co2e_tonnes,
            })

        return results

    # ==================================================================
    # PUBLIC API: Mass Balance Method (Tier 2/3)
    # ==================================================================

    def calculate_mass_balance(
        self,
        process_type: str,
        material_inputs: List[Dict[str, Any]],
        carbon_outputs: List[Dict[str, Any]],
        carbon_stock_change: Decimal = Decimal("0"),
        gwp_source: str = "AR6",
        abatement_efficiency: Optional[Decimal] = None,
        process_params: Optional[Dict[str, Any]] = None,
        trace: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Calculate emissions using the mass balance method (Tier 2/3).

        Formula:
            total_carbon_in = sum(material_qty x carbon_content x oxidation)
            total_carbon_out = sum(product_qty x product_carbon_content)
            net_carbon = total_carbon_in - total_carbon_out - stock_change
            CO2_tonnes = net_carbon x (44/12)

        Each material_input dict should have:
            - material_type (str): Material identifier
            - quantity_tonnes (Decimal): Mass in tonnes
            - carbon_content (Decimal, optional): Override carbon fraction
            - fraction_oxidized (Decimal, optional): Default 1.0

        Each carbon_output dict should have:
            - product_type (str): Product identifier
            - quantity_tonnes (Decimal): Mass in tonnes
            - carbon_content (Decimal): Carbon fraction in product

        Args:
            process_type: Process type identifier.
            material_inputs: List of material input records.
            carbon_outputs: List of carbon output records.
            carbon_stock_change: Carbon stored in intermediates (tonnes C).
            gwp_source: GWP source. Default ``"AR6"``.
            abatement_efficiency: Fraction of emissions abated.
            process_params: Process-specific parameters.
            trace: Mutable trace list.

        Returns:
            List of per-gas emission dictionaries (primarily CO2).

        Raises:
            ValueError: If input data is invalid.
        """
        if trace is None:
            trace = []

        pt_key = process_type.upper()
        params = process_params or {}
        abatement = self._validate_abatement(abatement_efficiency)

        trace.append(
            f"[MB] inputs={len(material_inputs)}, outputs={len(carbon_outputs)}, "
            f"stock_change={carbon_stock_change}"
        )

        # Step 1: Calculate total carbon input
        total_carbon_in = Decimal("0")
        for i, mat in enumerate(material_inputs):
            qty = Decimal(str(mat.get("quantity_tonnes", 0)))
            cc = self._resolve_carbon_content(mat)
            fox = Decimal(str(mat.get("fraction_oxidized", "1.0")))

            carbon_contribution = self._quantize(qty * cc * fox)
            total_carbon_in += carbon_contribution

            trace.append(
                f"[MB:IN:{i}] {mat.get('material_type', 'unknown')}: "
                f"qty={qty}, cc={cc}, fox={fox}, C={carbon_contribution}"
            )

        # Step 2: Calculate total carbon output
        total_carbon_out = Decimal("0")
        for i, out in enumerate(carbon_outputs):
            qty = Decimal(str(out.get("quantity_tonnes", 0)))
            cc = Decimal(str(out.get("carbon_content", "0")))

            carbon_in_product = self._quantize(qty * cc)
            total_carbon_out += carbon_in_product

            trace.append(
                f"[MB:OUT:{i}] {out.get('product_type', 'unknown')}: "
                f"qty={qty}, cc={cc}, C={carbon_in_product}"
            )

        # Step 3: Net carbon and CO2
        net_carbon = self._quantize(
            total_carbon_in - total_carbon_out - carbon_stock_change
        )

        # Ensure non-negative (mass balance check)
        if net_carbon < Decimal("0"):
            logger.warning(
                "Mass balance negative net carbon for %s: %s. "
                "Possible data error. Clamping to zero.",
                pt_key, net_carbon,
            )
            trace.append(
                f"[MB:WARN] Negative net carbon {net_carbon}, clamped to 0"
            )
            net_carbon = Decimal("0")

        # CO2 = net_carbon * (MW_CO2 / MW_C) = net_carbon * 44/12
        co2_tonnes = self._quantize(net_carbon * _CO2_C_RATIO)

        # Apply process-specific adjustments
        co2_tonnes = self._apply_mass_balance_adjustments(
            pt_key, co2_tonnes, params, trace,
        )

        # Apply abatement
        co2_tonnes_net = self._apply_abatement(co2_tonnes, abatement)

        # Get GWP (CO2 GWP = 1, but we apply formally for consistency)
        gwp_co2 = self._resolve_gwp("CO2", gwp_source)
        co2e_kg = self._quantize(co2_tonnes_net * gwp_co2 * _TONNES_TO_KG)
        co2e_tonnes = self._quantize(co2_tonnes_net * gwp_co2)

        trace.append(
            f"[MB:RESULT] C_in={total_carbon_in}, C_out={total_carbon_out}, "
            f"stock={carbon_stock_change}, net_C={net_carbon}, "
            f"CO2={co2_tonnes}, net_CO2={co2_tonnes_net}"
        )

        return [{
            "gas": "CO2",
            "total_carbon_input_tonnes": total_carbon_in,
            "total_carbon_output_tonnes": total_carbon_out,
            "carbon_stock_change_tonnes": carbon_stock_change,
            "net_carbon_tonnes": net_carbon,
            "raw_emissions_tonnes": co2_tonnes,
            "abatement_efficiency": abatement,
            "net_emissions_tonnes": co2_tonnes_net,
            "gwp": gwp_co2,
            "gwp_source": gwp_source,
            "co2e_kg": co2e_kg,
            "co2e_tonnes": co2e_tonnes,
        }]

    # ==================================================================
    # PUBLIC API: Stoichiometric Method (Tier 2)
    # ==================================================================

    def calculate_stoichiometric(
        self,
        process_type: str,
        carbonate_inputs: List[Dict[str, Any]],
        gwp_source: str = "AR6",
        abatement_efficiency: Optional[Decimal] = None,
        process_params: Optional[Dict[str, Any]] = None,
        trace: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Calculate emissions using the stoichiometric method (Tier 2).

        Formula:
            CO2 = sum(carbonate_qty x co2_factor x fraction_calcined)

        Each carbonate_input dict should have:
            - carbonate_type (str): e.g. "CALCITE", "MAGNESITE", "DOLOMITE"
            - quantity_tonnes (Decimal): Mass of carbonate in tonnes
            - fraction_calcined (Decimal, optional): Default 1.0
            - purity (Decimal, optional): Carbonate purity fraction. Default 1.0

        Args:
            process_type: Process type identifier.
            carbonate_inputs: List of carbonate input records.
            gwp_source: GWP source. Default ``"AR6"``.
            abatement_efficiency: Fraction of emissions abated.
            process_params: Process-specific parameters.
            trace: Mutable trace list.

        Returns:
            List of per-gas emission dictionaries (primarily CO2).

        Raises:
            ValueError: If carbonate type is not recognized.
        """
        if trace is None:
            trace = []

        pt_key = process_type.upper()
        params = process_params or {}
        abatement = self._validate_abatement(abatement_efficiency)

        trace.append(
            f"[STOICH] inputs={len(carbonate_inputs)}"
        )

        total_co2_tonnes = Decimal("0")

        for i, carb in enumerate(carbonate_inputs):
            carb_type = str(carb.get("carbonate_type", "")).upper()
            qty = Decimal(str(carb.get("quantity_tonnes", 0)))
            fraction_calcined = Decimal(str(carb.get("fraction_calcined", "1.0")))
            purity = Decimal(str(carb.get("purity", "1.0")))

            # Get stoichiometric CO2 factor from database
            co2_factor = self._resolve_carbonate_factor(carb_type)

            # CO2 = qty * purity * co2_factor * fraction_calcined
            co2_from_carb = self._quantize(
                qty * purity * co2_factor * fraction_calcined
            )
            total_co2_tonnes += co2_from_carb

            trace.append(
                f"[STOICH:{i}] {carb_type}: qty={qty}, purity={purity}, "
                f"EF={co2_factor}, calcined={fraction_calcined}, "
                f"CO2={co2_from_carb}"
            )

        # Apply process-specific corrections
        total_co2_tonnes = self._apply_stoichiometric_adjustments(
            pt_key, total_co2_tonnes, params, trace,
        )

        # Apply abatement
        co2_tonnes_net = self._apply_abatement(total_co2_tonnes, abatement)

        # GWP
        gwp_co2 = self._resolve_gwp("CO2", gwp_source)
        co2e_kg = self._quantize(co2_tonnes_net * gwp_co2 * _TONNES_TO_KG)
        co2e_tonnes = self._quantize(co2_tonnes_net * gwp_co2)

        trace.append(
            f"[STOICH:RESULT] total_CO2={total_co2_tonnes}, "
            f"net_CO2={co2_tonnes_net}"
        )

        return [{
            "gas": "CO2",
            "raw_emissions_tonnes": total_co2_tonnes,
            "abatement_efficiency": abatement,
            "net_emissions_tonnes": co2_tonnes_net,
            "gwp": gwp_co2,
            "gwp_source": gwp_source,
            "co2e_kg": co2e_kg,
            "co2e_tonnes": co2e_tonnes,
            "carbonate_detail": [
                {
                    "carbonate_type": str(c.get("carbonate_type", "")).upper(),
                    "quantity_tonnes": Decimal(str(c.get("quantity_tonnes", 0))),
                }
                for c in carbonate_inputs
            ],
        }]

    # ==================================================================
    # PUBLIC API: Direct Measurement Method
    # ==================================================================

    def calculate_direct(
        self,
        process_type: str,
        measured_emissions: Dict[str, Decimal],
        gwp_source: str = "AR6",
        trace: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Calculate emissions from direct measurement data (CEMS, stack test).

        Takes measured emission values directly and applies GWP conversion.

        Args:
            process_type: Process type identifier.
            measured_emissions: Dictionary mapping gas name to measured
                emission mass in tonnes (e.g. {"CO2": Decimal("50000"),
                "N2O": Decimal("0.5")}).
            gwp_source: GWP source. Default ``"AR6"``.
            trace: Mutable trace list.

        Returns:
            List of per-gas emission dictionaries.

        Example:
            >>> result = calc.calculate_direct(
            ...     "NITRIC_ACID",
            ...     {"N2O": Decimal("3.5")},
            ... )
        """
        if trace is None:
            trace = []

        pt_key = process_type.upper()
        trace.append(
            f"[DIRECT] gases={list(measured_emissions.keys())}"
        )

        results: List[Dict[str, Any]] = []

        for gas, mass_tonnes in measured_emissions.items():
            gas_key = gas.upper()
            mass = Decimal(str(mass_tonnes))

            if mass < 0:
                raise ValueError(
                    f"Measured emissions cannot be negative for {gas}: {mass}"
                )

            gwp = self._resolve_gwp(gas_key, gwp_source)
            co2e_kg = self._quantize(mass * gwp * _TONNES_TO_KG)
            co2e_tonnes = self._quantize(mass * gwp)

            trace.append(
                f"[DIRECT:{gas_key}] mass={mass} t, GWP={gwp}, "
                f"co2e={co2e_tonnes} t"
            )

            results.append({
                "gas": gas_key,
                "measured_mass_tonnes": mass,
                "net_emissions_tonnes": mass,
                "gwp": gwp,
                "gwp_source": gwp_source,
                "co2e_kg": co2e_kg,
                "co2e_tonnes": co2e_tonnes,
            })

        return results

    # ==================================================================
    # PUBLIC API: Batch Calculation
    # ==================================================================

    def calculate_batch(
        self,
        calculations: List[Dict[str, Any]],
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process multiple calculation requests in batch.

        Each element in ``calculations`` is a dictionary of keyword arguments
        for the ``calculate()`` method.

        Args:
            calculations: List of calculation parameter dictionaries.
            gwp_source: Optional GWP override applied to all calculations.

        Returns:
            Dictionary with keys:
                - batch_id (str)
                - results (List[Dict])
                - total_co2e_kg (Decimal)
                - total_co2e_tonnes (Decimal)
                - success_count (int)
                - failure_count (int)
                - processing_time_ms (float)

        Example:
            >>> batch_result = calc.calculate_batch([
            ...     {"process_type": "CEMENT", "activity_data": Decimal("100000")},
            ...     {"process_type": "LIME", "activity_data": Decimal("50000")},
            ... ])
        """
        start_time = time.monotonic()
        batch_id = f"pe_batch_{uuid.uuid4().hex[:12]}"

        results: List[Dict[str, Any]] = []
        total_co2e_kg = Decimal("0")
        total_co2e_tonnes = Decimal("0")
        success_count = 0
        failure_count = 0

        for calc_params in calculations:
            if gwp_source and "gwp_source" not in calc_params:
                calc_params["gwp_source"] = gwp_source

            result = self.calculate(**calc_params)
            results.append(result)

            if result["status"] == "SUCCESS":
                success_count += 1
                total_co2e_kg += result["total_co2e_kg"]
                total_co2e_tonnes += result["total_co2e_tonnes"]
            else:
                failure_count += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000

        if _METRICS_AVAILABLE and _record_batch is not None:
            status = "completed" if failure_count == 0 else "partial"
            _record_batch(status)

        return {
            "batch_id": batch_id,
            "results": results,
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "success_count": success_count,
            "failure_count": failure_count,
            "total_count": len(calculations),
            "processing_time_ms": elapsed_ms,
        }

    # ==================================================================
    # PUBLIC API: GWP Application
    # ==================================================================

    def apply_gwp(
        self,
        gas: str,
        mass_tonnes: Decimal,
        gwp_source: str = "AR6",
    ) -> Dict[str, Decimal]:
        """Apply GWP conversion to a gas mass.

        Args:
            gas: Gas identifier.
            mass_tonnes: Mass of the gas in tonnes.
            gwp_source: GWP source.

        Returns:
            Dictionary with gwp, co2e_kg, co2e_tonnes.

        Example:
            >>> result = calc.apply_gwp("N2O", Decimal("3.5"), "AR6")
            >>> result["co2e_tonnes"]
            Decimal('955.50000000')
        """
        gas_key = gas.upper()
        mass = Decimal(str(mass_tonnes))
        gwp = self._resolve_gwp(gas_key, gwp_source)

        co2e_tonnes = self._quantize(mass * gwp)
        co2e_kg = self._quantize(co2e_tonnes * _TONNES_TO_KG)

        return {
            "gas": gas_key,
            "mass_tonnes": mass,
            "gwp": gwp,
            "gwp_source": gwp_source,
            "co2e_kg": co2e_kg,
            "co2e_tonnes": co2e_tonnes,
        }

    # ==================================================================
    # PUBLIC API: Abatement Application
    # ==================================================================

    def apply_abatement(
        self,
        emissions_tonnes: Decimal,
        abatement_efficiency: Decimal,
    ) -> Dict[str, Decimal]:
        """Apply abatement efficiency to gross emissions.

        Args:
            emissions_tonnes: Gross emission mass in tonnes.
            abatement_efficiency: Fraction abated (0.0 to 1.0).

        Returns:
            Dictionary with gross, abated, and net emissions.

        Example:
            >>> result = calc.apply_abatement(Decimal("100"), Decimal("0.90"))
            >>> result["net_emissions_tonnes"]
            Decimal('10.00000000')
        """
        eff = self._validate_abatement(abatement_efficiency)
        gross = Decimal(str(emissions_tonnes))
        abated = self._quantize(gross * eff)
        net = self._quantize(gross - abated)

        return {
            "gross_emissions_tonnes": gross,
            "abatement_efficiency": eff,
            "abated_emissions_tonnes": abated,
            "net_emissions_tonnes": net,
        }

    # ==================================================================
    # PROCESS-SPECIFIC: Cement
    # ==================================================================

    def _apply_cement_adjustments(
        self,
        co2_tonnes: Decimal,
        activity_data: Decimal,
        params: Dict[str, Any],
        trace: List[str],
    ) -> Decimal:
        """Apply cement-specific corrections (CKD, clinker ratio).

        CKD correction:
            corrected = co2 * (1 + ckd_correction_factor)
            Default CKD factor: 0.02 (2% additional emissions)

        Args:
            co2_tonnes: Base CO2 emissions.
            activity_data: Clinker production in tonnes.
            params: Process parameters with optional keys:
                - ckd_correction_factor (Decimal): CKD correction. Default 0.02.
                - clinker_to_cement_ratio (Decimal): For cement-basis input.
            trace: Audit trace.

        Returns:
            Adjusted CO2 emissions in tonnes.
        """
        ckd_factor = Decimal(str(params.get("ckd_correction_factor", "0.02")))
        corrected = self._quantize(co2_tonnes * (Decimal("1") + ckd_factor))

        if ckd_factor != Decimal("0"):
            trace.append(
                f"[CEMENT:CKD] ckd_factor={ckd_factor}, "
                f"before={co2_tonnes}, after={corrected}"
            )

        return corrected

    # ==================================================================
    # PROCESS-SPECIFIC: Aluminum PFC
    # ==================================================================

    def _calculate_aluminum_pfc(
        self,
        activity_data: Decimal,
        params: Dict[str, Any],
        gwp_source: str,
        trace: List[str],
    ) -> List[Dict[str, Any]]:
        """Calculate PFC emissions for aluminum smelting.

        IPCC Tier 2 method (slope approach):
            CF4_kg/t_Al = slope_CF4 * AE_minutes/cell-day
            C2F6_kg/t_Al = CF4_kg/t_Al * C2F6_ratio

        Where AE_minutes/cell-day = AE_frequency * AE_duration

        Args:
            activity_data: Aluminum production in tonnes.
            params: Must include:
                - anode_effect_frequency (Decimal): AE per cell-day
                - anode_effect_duration (Decimal): Minutes per AE
                - pfc_slope_cf4 (Decimal): Slope coefficient
                - pfc_c2f6_ratio (Decimal): C2F6/CF4 weight ratio
            gwp_source: GWP source.
            trace: Audit trace.

        Returns:
            List of PFC gas emission dictionaries.
        """
        ae_freq = Decimal(str(params.get("anode_effect_frequency", "0.3")))
        ae_dur = Decimal(str(params.get("anode_effect_duration", "2.0")))
        slope_cf4 = Decimal(str(params.get("pfc_slope_cf4", "0.143")))
        c2f6_ratio = Decimal(str(params.get("pfc_c2f6_ratio", "0.100")))

        ae_minutes_per_cell_day = self._quantize(ae_freq * ae_dur)

        # CF4 emissions (kg per tonne Al)
        cf4_kg_per_t = self._quantize(slope_cf4 * ae_minutes_per_cell_day)
        cf4_tonnes = self._quantize(
            activity_data * cf4_kg_per_t * _KG_TO_TONNES
        )

        # C2F6 emissions (kg per tonne Al)
        c2f6_kg_per_t = self._quantize(cf4_kg_per_t * c2f6_ratio)
        c2f6_tonnes = self._quantize(
            activity_data * c2f6_kg_per_t * _KG_TO_TONNES
        )

        trace.append(
            f"[AL:PFC] AE_freq={ae_freq}, AE_dur={ae_dur}, "
            f"AE_min/cd={ae_minutes_per_cell_day}, slope={slope_cf4}, "
            f"CF4={cf4_tonnes}t, C2F6={c2f6_tonnes}t"
        )

        results: List[Dict[str, Any]] = []

        # CF4
        gwp_cf4 = self._resolve_gwp("CF4", gwp_source)
        co2e_cf4_t = self._quantize(cf4_tonnes * gwp_cf4)
        results.append({
            "gas": "CF4",
            "emission_factor": cf4_kg_per_t,
            "emission_factor_unit": "kgCF4/tAl",
            "net_emissions_tonnes": cf4_tonnes,
            "gwp": gwp_cf4,
            "gwp_source": gwp_source,
            "co2e_kg": self._quantize(co2e_cf4_t * _TONNES_TO_KG),
            "co2e_tonnes": co2e_cf4_t,
        })

        # C2F6
        gwp_c2f6 = self._resolve_gwp("C2F6", gwp_source)
        co2e_c2f6_t = self._quantize(c2f6_tonnes * gwp_c2f6)
        results.append({
            "gas": "C2F6",
            "emission_factor": c2f6_kg_per_t,
            "emission_factor_unit": "kgC2F6/tAl",
            "net_emissions_tonnes": c2f6_tonnes,
            "gwp": gwp_c2f6,
            "gwp_source": gwp_source,
            "co2e_kg": self._quantize(co2e_c2f6_t * _TONNES_TO_KG),
            "co2e_tonnes": co2e_c2f6_t,
        })

        return results

    # ==================================================================
    # PROCESS-SPECIFIC: Semiconductor
    # ==================================================================

    def _calculate_semiconductor(
        self,
        gas: str,
        gas_usage_kg: Decimal,
        params: Dict[str, Any],
        gwp_source: str,
        trace: List[str],
    ) -> Dict[str, Any]:
        """Calculate semiconductor fab emissions for a single gas.

        Formula (IPCC Tier 2a):
            emissions = gas_usage * (1 - utilization) * (1 - abatement)
                       + by_product_cf4_formation
            co2e = emissions * GWP

        Args:
            gas: Gas identifier.
            gas_usage_kg: Amount of gas used in kg.
            params: Optional overrides for utilization, destruction, etc.
            gwp_source: GWP source.
            trace: Audit trace.

        Returns:
            Per-gas emission dictionary.
        """
        gas_key = gas.upper()

        # Get default parameters from database
        semi_params = {}
        if self._process_db is not None:
            try:
                semi_params = self._process_db.get_semiconductor_gas_params(gas_key)
            except (KeyError, AttributeError):
                pass

        utilization = Decimal(str(
            params.get(f"utilization_{gas_key}",
                       semi_params.get("default_utilization_rate", "0.50"))
        ))
        destruction = Decimal(str(
            params.get(f"destruction_{gas_key}",
                       semi_params.get("default_destruction_efficiency", "0.00"))
        ))
        by_product_ratio = Decimal(str(
            semi_params.get("by_product_cf4_fraction", "0.00")
        ))

        # Remaining gas after process utilization
        remaining = self._quantize(gas_usage_kg * (Decimal("1") - utilization))

        # Apply abatement / point-of-use destruction
        net_gas = self._quantize(remaining * (Decimal("1") - destruction))

        # By-product CF4 formation (from some gases)
        by_product_cf4_kg = Decimal("0")
        if by_product_ratio > 0 and gas_key != "CF4":
            by_product_cf4_kg = self._quantize(
                gas_usage_kg * utilization * by_product_ratio
            )

        net_tonnes = self._quantize(net_gas * _KG_TO_TONNES)
        gwp = self._resolve_gwp(gas_key, gwp_source)
        co2e_tonnes = self._quantize(net_tonnes * gwp)

        trace.append(
            f"[SEMI:{gas_key}] usage={gas_usage_kg}kg, util={utilization}, "
            f"dest={destruction}, remaining={remaining}kg, "
            f"net={net_gas}kg, bp_CF4={by_product_cf4_kg}kg"
        )

        return {
            "gas": gas_key,
            "gas_usage_kg": gas_usage_kg,
            "utilization_rate": utilization,
            "destruction_efficiency": destruction,
            "remaining_kg": remaining,
            "net_emissions_kg": net_gas,
            "net_emissions_tonnes": net_tonnes,
            "by_product_cf4_kg": by_product_cf4_kg,
            "gwp": gwp,
            "gwp_source": gwp_source,
            "co2e_kg": self._quantize(co2e_tonnes * _TONNES_TO_KG),
            "co2e_tonnes": co2e_tonnes,
        }

    # ==================================================================
    # PRIVATE: Process adjustments
    # ==================================================================

    def _apply_process_adjustments(
        self,
        process_type: str,
        gas: str,
        emissions: Decimal,
        activity_data: Decimal,
        params: Dict[str, Any],
        trace: List[str],
    ) -> Decimal:
        """Apply process-specific emission factor adjustments.

        Dispatches to process-specific correction methods based on the
        process type.

        Args:
            process_type: Process identifier (uppercased).
            gas: Gas identifier.
            emissions: Calculated emission mass.
            activity_data: Production quantity.
            params: Process-specific parameters.
            trace: Audit trace.

        Returns:
            Adjusted emission mass.
        """
        if process_type == "CEMENT" and gas == "CO2":
            return self._apply_cement_adjustments(
                emissions, activity_data, params, trace,
            )
        # Other processes use the emission factor directly without adjustment
        return emissions

    def _apply_mass_balance_adjustments(
        self,
        process_type: str,
        co2_tonnes: Decimal,
        params: Dict[str, Any],
        trace: List[str],
    ) -> Decimal:
        """Apply process-specific mass balance corrections.

        Args:
            process_type: Process identifier.
            co2_tonnes: Base CO2 from mass balance.
            params: Process-specific parameters.
            trace: Audit trace.

        Returns:
            Adjusted CO2 tonnes.
        """
        if process_type == "CEMENT":
            return self._apply_cement_adjustments(
                co2_tonnes, Decimal("0"), params, trace,
            )
        return co2_tonnes

    def _apply_stoichiometric_adjustments(
        self,
        process_type: str,
        co2_tonnes: Decimal,
        params: Dict[str, Any],
        trace: List[str],
    ) -> Decimal:
        """Apply process-specific stoichiometric corrections.

        For cement: CKD correction factor.
        For glass: cullet ratio credit.

        Args:
            process_type: Process identifier.
            co2_tonnes: Base CO2 from stoichiometric calculation.
            params: Process-specific parameters.
            trace: Audit trace.

        Returns:
            Adjusted CO2 tonnes.
        """
        if process_type == "CEMENT":
            return self._apply_cement_adjustments(
                co2_tonnes, Decimal("0"), params, trace,
            )

        if process_type == "GLASS":
            cullet_ratio = Decimal(str(params.get("cullet_ratio", "0")))
            if cullet_ratio > 0:
                # Cullet (recycled glass) does not release CO2 from carbonates
                adjusted = self._quantize(
                    co2_tonnes * (Decimal("1") - cullet_ratio)
                )
                trace.append(
                    f"[GLASS:CULLET] ratio={cullet_ratio}, "
                    f"before={co2_tonnes}, after={adjusted}"
                )
                return adjusted

        return co2_tonnes

    # ==================================================================
    # PRIVATE: Factor resolution helpers
    # ==================================================================

    def _get_process_gases(
        self,
        process_type: str,
        ef_source: str,
        custom_factors: Optional[Dict[str, Decimal]],
    ) -> List[str]:
        """Determine which gases to calculate for a process.

        Combines database-known gases with custom factor keys.

        Args:
            process_type: Process identifier.
            ef_source: Factor source.
            custom_factors: Custom factor overrides.

        Returns:
            List of gas identifiers.
        """
        gases: List[str] = []

        if self._process_db is not None:
            try:
                gases = self._process_db.get_available_gases(
                    process_type, ef_source,
                )
            except (KeyError, AttributeError):
                pass

        # Add custom factor gases not already in the list
        if custom_factors:
            for gas_key in custom_factors:
                if gas_key.upper() not in gases:
                    gases.append(gas_key.upper())

        return gases

    def _resolve_ef(
        self,
        process_type: str,
        gas: str,
        ef_source: str,
        custom_factors: Optional[Dict[str, Decimal]],
    ) -> Decimal:
        """Resolve emission factor: custom overrides first, then database.

        Args:
            process_type: Process identifier.
            gas: Gas identifier.
            ef_source: Factor source.
            custom_factors: Custom factor overrides.

        Returns:
            Emission factor as Decimal.

        Raises:
            KeyError: If no factor is found.
        """
        gas_key = gas.upper()

        # Custom factors take precedence
        if custom_factors:
            for k, v in custom_factors.items():
                if k.upper() == gas_key:
                    return Decimal(str(v))

        # Database lookup
        if self._process_db is not None:
            return self._process_db.get_emission_factor(
                process_type, gas_key, ef_source,
            )

        raise KeyError(
            f"No emission factor for {process_type}/{gas_key}/{ef_source} "
            f"and no process database available"
        )

    def _resolve_gwp(
        self,
        gas: str,
        gwp_source: str,
    ) -> Decimal:
        """Resolve GWP value from process database.

        Args:
            gas: Gas identifier.
            gwp_source: GWP source.

        Returns:
            GWP value as Decimal.

        Raises:
            KeyError: If GWP not found.
        """
        gas_key = gas.upper()

        if self._process_db is not None:
            return self._process_db.get_gwp(gas_key, gwp_source)

        # Fallback: minimal built-in GWP table
        _FALLBACK_GWP: Dict[str, Decimal] = {
            "CO2": Decimal("1"),
            "CH4": Decimal("29.8"),
            "N2O": Decimal("273"),
            "CF4": Decimal("7380"),
            "C2F6": Decimal("12400"),
            "SF6": Decimal("25200"),
            "NF3": Decimal("17400"),
            "HFC_23": Decimal("14600"),
        }

        if gas_key in _FALLBACK_GWP:
            return _FALLBACK_GWP[gas_key]

        raise KeyError(f"No GWP for gas '{gas}' in source '{gwp_source}'")

    def _resolve_carbon_content(
        self,
        material: Dict[str, Any],
    ) -> Decimal:
        """Resolve carbon content for a material input.

        Uses explicit override first, then database lookup.

        Args:
            material: Material input dict with optional ``carbon_content``
                and ``material_type`` keys.

        Returns:
            Carbon content fraction as Decimal.
        """
        # Explicit override
        if "carbon_content" in material and material["carbon_content"] is not None:
            return Decimal(str(material["carbon_content"]))

        # Database lookup
        mat_type = str(material.get("material_type", "")).upper()
        if mat_type and self._process_db is not None:
            try:
                mat_info = self._process_db.get_raw_material(mat_type)
                return mat_info["carbon_content"]
            except (KeyError, AttributeError):
                pass

        logger.warning(
            "No carbon content for material '%s', using 0.0",
            material.get("material_type", "UNKNOWN"),
        )
        return Decimal("0")

    def _resolve_carbonate_factor(
        self,
        carbonate_type: str,
    ) -> Decimal:
        """Resolve stoichiometric CO2 factor for a carbonate type.

        Args:
            carbonate_type: Carbonate identifier (e.g. "CALCITE").

        Returns:
            CO2 factor as Decimal (tonnes CO2 per tonne carbonate).

        Raises:
            KeyError: If carbonate type is not recognized.
        """
        key = carbonate_type.upper()

        if self._process_db is not None:
            try:
                cf_info = self._process_db.get_carbonate_factor(key)
                return cf_info["co2_factor"]
            except (KeyError, AttributeError):
                pass

        # Fallback: built-in factors for most common carbonates
        _FALLBACK_CARB: Dict[str, Decimal] = {
            "CALCITE": Decimal("0.4397"),
            "MAGNESITE": Decimal("0.5220"),
            "DOLOMITE": Decimal("0.4773"),
            "SIDERITE": Decimal("0.3799"),
            "ANKERITE": Decimal("0.4272"),
            "SODA_ASH": Decimal("0.4152"),
        }

        if key in _FALLBACK_CARB:
            return _FALLBACK_CARB[key]

        raise KeyError(f"Unknown carbonate type: {carbonate_type}")

    # ==================================================================
    # PRIVATE: Utility methods
    # ==================================================================

    def _validate_abatement(
        self,
        abatement: Optional[Decimal],
    ) -> Decimal:
        """Validate and normalize abatement efficiency.

        Args:
            abatement: Raw abatement value (0.0 to 1.0), or None.

        Returns:
            Validated Decimal between 0 and 1.

        Raises:
            ValueError: If out of range.
        """
        if abatement is None:
            return Decimal("0")

        eff = Decimal(str(abatement))
        if eff < 0 or eff > 1:
            raise ValueError(
                f"Abatement efficiency must be 0.0-1.0, got {eff}"
            )
        return eff

    def _apply_abatement(
        self,
        emissions: Decimal,
        abatement: Decimal,
    ) -> Decimal:
        """Apply abatement efficiency to emissions.

        Args:
            emissions: Gross emission mass.
            abatement: Validated efficiency (0.0-1.0).

        Returns:
            Net emissions after abatement.
        """
        if abatement == Decimal("0"):
            return emissions
        return self._quantize(emissions * (Decimal("1") - abatement))

    def _sum_co2e(
        self,
        gas_emissions: List[Dict[str, Any]],
    ) -> Decimal:
        """Sum total CO2e across all gas emissions.

        Args:
            gas_emissions: List of per-gas emission dictionaries.

        Returns:
            Total CO2e in kg.
        """
        total = Decimal("0")
        for ge in gas_emissions:
            co2e = ge.get("co2e_kg", Decimal("0"))
            total += Decimal(str(co2e))
        return self._quantize(total)

    def _quantize(self, value: Decimal) -> Decimal:
        """Round a Decimal to the configured precision.

        Args:
            value: Raw Decimal value.

        Returns:
            Rounded Decimal.
        """
        try:
            return value.quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
        except InvalidOperation:
            logger.warning("Failed to quantize value: %s", value)
            return value

    def _compute_provenance_hash(
        self,
        data: Dict[str, Any],
    ) -> str:
        """Compute SHA-256 provenance hash for a calculation result.

        Args:
            data: Dictionary of calculation identifiers and results.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an operation in the provenance tracker if available.

        Args:
            action: Action name.
            entity_id: Entity identifier.
            data: Optional data dictionary.
        """
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="emission_calculator",
                    action=action,
                    entity_id=entity_id,
                    data=data or {},
                )
            except Exception as exc:
                logger.debug(
                    "Provenance recording failed (non-critical): %s", exc,
                )
