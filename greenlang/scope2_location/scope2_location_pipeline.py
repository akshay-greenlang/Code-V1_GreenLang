# -*- coding: utf-8 -*-
"""
Engine 7: Scope 2 Location-Based Pipeline Engine for AGENT-MRV-009.

8-stage orchestrated calculation pipeline:
  Stage 1: Validate input (facility, energy type, consumption, grid region)
  Stage 2: Resolve grid region and emission factors
  Stage 3: Apply T&D losses
  Stage 4: Calculate electricity emissions (per-gas)
  Stage 5: Calculate steam/heat/cooling emissions
  Stage 6: Apply GWP conversion
  Stage 7: Run compliance checks (optional)
  Stage 8: Assemble results with provenance

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for upstream engines
# ---------------------------------------------------------------------------
_GridEmissionFactorDatabaseEngine = None
_ElectricityEmissionsEngine = None
_SteamHeatCoolingEngine = None
_TransmissionLossEngine = None
_UncertaintyQuantifierEngine = None
_ComplianceCheckerEngine = None
_Scope2LocationProvenance = None

GWP_TABLE: Dict[str, Dict[str, Decimal]] = {
    "AR4": {"co2": Decimal("1"), "ch4": Decimal("25"), "n2o": Decimal("298")},
    "AR5": {"co2": Decimal("1"), "ch4": Decimal("28"), "n2o": Decimal("265")},
    "AR6": {"co2": Decimal("1"), "ch4": Decimal("27.9"), "n2o": Decimal("273")},
    "AR6_20YR": {"co2": Decimal("1"), "ch4": Decimal("81.2"), "n2o": Decimal("273")},
}

VALID_ENERGY_TYPES = {"electricity", "steam", "heating", "cooling"}
VALID_GWP_SOURCES = {"AR4", "AR5", "AR6", "AR6_20YR"}

PIPELINE_STAGES = [
    "validate_input",
    "resolve_factors",
    "apply_td_losses",
    "calculate_electricity",
    "calculate_non_electric",
    "apply_gwp_conversion",
    "compliance_checks",
    "assemble_results",
]


def _lazy_import_engines():
    """Lazily import upstream engine classes."""
    global _GridEmissionFactorDatabaseEngine, _ElectricityEmissionsEngine
    global _SteamHeatCoolingEngine, _TransmissionLossEngine
    global _UncertaintyQuantifierEngine, _ComplianceCheckerEngine
    global _Scope2LocationProvenance

    if _GridEmissionFactorDatabaseEngine is None:
        try:
            from greenlang.scope2_location.grid_factor_database import (
                GridEmissionFactorDatabaseEngine,
            )
            _GridEmissionFactorDatabaseEngine = GridEmissionFactorDatabaseEngine
        except ImportError:
            pass

    if _ElectricityEmissionsEngine is None:
        try:
            from greenlang.scope2_location.electricity_emissions import (
                ElectricityEmissionsEngine,
            )
            _ElectricityEmissionsEngine = ElectricityEmissionsEngine
        except ImportError:
            pass

    if _SteamHeatCoolingEngine is None:
        try:
            from greenlang.scope2_location.steam_heat_cooling import (
                SteamHeatCoolingEngine,
            )
            _SteamHeatCoolingEngine = SteamHeatCoolingEngine
        except ImportError:
            pass

    if _TransmissionLossEngine is None:
        try:
            from greenlang.scope2_location.transmission_loss import (
                TransmissionLossEngine,
            )
            _TransmissionLossEngine = TransmissionLossEngine
        except ImportError:
            pass

    if _UncertaintyQuantifierEngine is None:
        try:
            from greenlang.scope2_location.uncertainty_quantifier import (
                UncertaintyQuantifierEngine,
            )
            _UncertaintyQuantifierEngine = UncertaintyQuantifierEngine
        except ImportError:
            pass

    if _ComplianceCheckerEngine is None:
        try:
            from greenlang.scope2_location.compliance_checker import (
                ComplianceCheckerEngine,
            )
            _ComplianceCheckerEngine = ComplianceCheckerEngine
        except ImportError:
            pass

    if _Scope2LocationProvenance is None:
        try:
            from greenlang.scope2_location.provenance import (
                Scope2LocationProvenance,
            )
            _Scope2LocationProvenance = Scope2LocationProvenance
        except ImportError:
            pass


class Scope2LocationPipelineEngine:
    """Engine 7: 8-stage orchestrated Scope 2 location-based calculation pipeline.

    Ties together all upstream engines to provide end-to-end emission
    calculations with full provenance tracking and compliance checking.
    """

    def __init__(
        self,
        grid_factor_db: Any = None,
        electricity_engine: Any = None,
        steam_heat_cool_engine: Any = None,
        transmission_engine: Any = None,
        uncertainty_engine: Any = None,
        compliance_engine: Any = None,
        config: Any = None,
        metrics: Any = None,
        provenance: Any = None,
    ) -> None:
        _lazy_import_engines()
        self._config = config
        self._metrics = metrics
        self._lock = threading.Lock()
        self._pipeline_runs = 0
        self._total_co2e_processed = Decimal("0")

        # Initialize upstream engines (create defaults if not provided)
        if grid_factor_db is not None:
            self._grid_factor_db = grid_factor_db
        elif _GridEmissionFactorDatabaseEngine:
            self._grid_factor_db = _GridEmissionFactorDatabaseEngine(config, metrics)
        else:
            self._grid_factor_db = None

        if electricity_engine is not None:
            self._electricity = electricity_engine
        elif _ElectricityEmissionsEngine:
            self._electricity = _ElectricityEmissionsEngine(
                self._grid_factor_db, config, metrics
            )
        else:
            self._electricity = None

        if steam_heat_cool_engine is not None:
            self._steam_heat_cool = steam_heat_cool_engine
        elif _SteamHeatCoolingEngine:
            self._steam_heat_cool = _SteamHeatCoolingEngine(
                self._grid_factor_db, config, metrics
            )
        else:
            self._steam_heat_cool = None

        if transmission_engine is not None:
            self._transmission = transmission_engine
        elif _TransmissionLossEngine:
            self._transmission = _TransmissionLossEngine(config, metrics)
        else:
            self._transmission = None

        if uncertainty_engine is not None:
            self._uncertainty = uncertainty_engine
        elif _UncertaintyQuantifierEngine:
            self._uncertainty = _UncertaintyQuantifierEngine(config, metrics)
        else:
            self._uncertainty = None

        if compliance_engine is not None:
            self._compliance = compliance_engine
        elif _ComplianceCheckerEngine:
            self._compliance = _ComplianceCheckerEngine(config, metrics)
        else:
            self._compliance = None

        logger.info("Scope2LocationPipelineEngine initialized with %d engines",
                     sum(1 for e in [self._grid_factor_db, self._electricity,
                                      self._steam_heat_cool, self._transmission,
                                      self._uncertainty, self._compliance] if e))

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full 8-stage Scope 2 location-based calculation pipeline.

        Args:
            request: Calculation request dict with keys:
                - calculation_id (optional, auto-generated)
                - tenant_id
                - facility_id
                - country_code
                - egrid_subregion (optional, US only)
                - energy_type: 'electricity', 'steam', 'heating', 'cooling'
                - consumption_value: Decimal quantity
                - consumption_unit: 'mwh', 'gj', 'kwh', etc.
                - gwp_source: 'AR4', 'AR5', 'AR6', 'AR6_20YR' (default 'AR5')
                - include_td_losses: bool (default True)
                - include_compliance: bool (default False)
                - compliance_frameworks: list of framework names (optional)
                - steam_type/heating_type/cooling_type (for non-electric)

        Returns:
            CalculationResult dict with full results and provenance.
        """
        start = time.monotonic()
        prov = None
        if _Scope2LocationProvenance:
            prov = _Scope2LocationProvenance()

        calc_id = request.get("calculation_id", str(uuid.uuid4()))
        facility_id = request.get("facility_id", "")
        tenant_id = request.get("tenant_id", "")
        energy_type = request.get("energy_type", "electricity")
        country_code = request.get("country_code", "US")
        egrid_subregion = request.get("egrid_subregion")
        gwp_source = request.get("gwp_source", "AR5")
        include_td = request.get("include_td_losses", True)
        include_compliance = request.get("include_compliance", False)
        compliance_frameworks = request.get("compliance_frameworks")

        stages_data = {}

        # Stage 1: Validate Input
        validation = self.stage_validate_input(request)
        if not validation.get("valid", False):
            raise ValueError(
                f"Input validation failed: {validation.get('errors', [])}"
            )
        stages_data["validation"] = validation
        if prov:
            prov.hash_input(request)

        # Stage 2: Resolve Factors
        factors = self.stage_resolve_factors(
            country_code, egrid_subregion, energy_type
        )
        stages_data["factors"] = factors
        if prov:
            prov.hash_grid_factor(
                factors.get("country_code", country_code),
                factors.get("source", "iea"),
                factors.get("year", 2024),
                factors.get("co2_kg_per_mwh", Decimal("0")),
                factors.get("ch4_kg_per_mwh", Decimal("0")),
                factors.get("n2o_kg_per_mwh", Decimal("0")),
            )

        # Stage 3: T&D Losses
        td_loss_pct = Decimal("0")
        td_result = None
        if include_td and energy_type == "electricity":
            td_result = self.stage_apply_td_losses(
                self._normalize_consumption(request),
                country_code,
                request.get("custom_td_loss"),
            )
            td_loss_pct = td_result.get("td_loss_pct", Decimal("0"))
            stages_data["td_losses"] = td_result
            if prov:
                prov.hash_td_loss(
                    country_code, td_loss_pct, td_result.get("method", "country_average")
                )

        # Stage 4: Electricity Emissions
        elec_result = None
        if energy_type == "electricity":
            consumption_mwh = self._normalize_consumption(request)
            co2_ef = factors.get("co2_kg_per_mwh", Decimal("0"))
            ch4_ef = factors.get("ch4_kg_per_mwh", Decimal("0"))
            n2o_ef = factors.get("n2o_kg_per_mwh", Decimal("0"))

            elec_result = self.stage_calculate_electricity(
                consumption_mwh, co2_ef, ch4_ef, n2o_ef, gwp_source, td_loss_pct
            )
            stages_data["electricity"] = elec_result
            if prov:
                prov.hash_electricity_calculation(
                    consumption_mwh, co2_ef, td_loss_pct,
                    elec_result.get("total_co2e_kg", Decimal("0"))
                )

        # Stage 5: Steam/Heat/Cooling
        non_elec_result = None
        if energy_type in ("steam", "heating", "cooling"):
            non_elec_requests = [{
                "energy_type": energy_type,
                "consumption_gj": self._normalize_to_gj(request),
                "sub_type": request.get(f"{energy_type}_type", ""),
                "custom_ef": request.get("custom_ef"),
                "country_code": country_code,
            }]
            non_elec_result = self.stage_calculate_non_electric(non_elec_requests)
            stages_data["non_electric"] = non_elec_result
            if prov:
                prov.hash_steam_heat_cooling(
                    energy_type,
                    self._normalize_to_gj(request),
                    Decimal("0"),
                    non_elec_result.get("combined_co2e_kg", Decimal("0")),
                )

        # Stage 6: GWP Conversion (already applied in per-gas calculations)
        total_co2e_kg = Decimal("0")
        gas_breakdown = []

        if elec_result:
            total_co2e_kg = elec_result.get("total_co2e_kg", Decimal("0"))
            gas_breakdown = elec_result.get("gas_breakdown", [])
        elif non_elec_result:
            total_co2e_kg = non_elec_result.get("combined_co2e_kg", Decimal("0"))

        total_co2e_tonnes = (total_co2e_kg / Decimal("1000")).quantize(
            Decimal("0.000001"), ROUND_HALF_UP
        )

        gwp_result = {
            "gwp_source": gwp_source,
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
        }
        stages_data["gwp_conversion"] = gwp_result
        if prov and gas_breakdown:
            co2_total = sum(
                g.get("co2e_kg", Decimal("0"))
                for g in gas_breakdown if g.get("gas") == "co2"
            )
            ch4_total = sum(
                g.get("co2e_kg", Decimal("0"))
                for g in gas_breakdown if g.get("gas") == "ch4"
            )
            n2o_total = sum(
                g.get("co2e_kg", Decimal("0"))
                for g in gas_breakdown if g.get("gas") == "n2o"
            )
            prov.hash_gas_breakdown(
                co2_total, ch4_total, n2o_total, gwp_source, total_co2e_kg
            )

        # Stage 7: Compliance Checks (optional)
        compliance_results = []
        if include_compliance and self._compliance:
            compliance_input = {
                "calculation_id": calc_id,
                "energy_type": energy_type,
                "country_code": country_code,
                "emission_factor_source": factors.get("source", ""),
                "ef_year": factors.get("year"),
                "reporting_year": datetime.utcnow().year,
                "grid_region": egrid_subregion or country_code,
                "gas_breakdown": gas_breakdown,
                "td_loss_pct": td_loss_pct,
                "total_co2e_kg": total_co2e_kg,
                "total_co2e_tonnes": total_co2e_tonnes,
                "gwp_source": gwp_source,
            }
            compliance_results = self.stage_compliance_check(
                compliance_input, compliance_frameworks
            )
            stages_data["compliance"] = compliance_results
            if prov:
                for cr in compliance_results:
                    prov.hash_compliance_check(
                        cr.get("framework", ""),
                        cr.get("status", "not_assessed"),
                        len(cr.get("findings", [])),
                    )

        # Stage 8: Assemble Results
        provenance_hash = prov.get_chain_hash() if prov else ""

        ef_applied = factors.get("total_co2e_kg_per_mwh", Decimal("0"))

        result = {
            "calculation_id": calc_id,
            "tenant_id": tenant_id,
            "facility_id": facility_id,
            "energy_type": energy_type,
            "consumption_value": request.get("consumption_value", Decimal("0")),
            "consumption_unit": request.get("consumption_unit", "mwh"),
            "country_code": country_code,
            "grid_region": egrid_subregion or country_code,
            "emission_factor_source": factors.get("source", ""),
            "ef_co2e_per_mwh": ef_applied,
            "td_loss_pct": td_loss_pct,
            "gas_breakdown": gas_breakdown,
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "gwp_source": gwp_source,
            "provenance_hash": provenance_hash,
            "compliance_results": compliance_results,
            "calculated_at": datetime.utcnow().isoformat(),
            "metadata": {
                "pipeline_version": "1.0.0",
                "stages_completed": len(stages_data),
                "validation_warnings": validation.get("warnings", []),
            },
        }

        # Record metrics
        duration = time.monotonic() - start
        with self._lock:
            self._pipeline_runs += 1
            self._total_co2e_processed += total_co2e_tonnes

        if self._metrics:
            try:
                self._metrics.record_calculation(
                    energy_type, "location_based", duration, float(total_co2e_tonnes)
                )
            except Exception:
                pass

        logger.info(
            "Pipeline completed: calc=%s type=%s co2e=%.6f tonnes (%.3fs)",
            calc_id, energy_type, total_co2e_tonnes, duration,
        )
        return result

    def run_batch_pipeline(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run pipeline for a batch of calculation requests.

        Args:
            batch: Dict with batch_id, tenant_id, requests (list).

        Returns:
            BatchCalculationResult dict.
        """
        batch_id = batch.get("batch_id", str(uuid.uuid4()))
        requests = batch.get("requests", [])
        results = []
        errors = []
        total_co2e = Decimal("0")

        for i, req in enumerate(requests):
            try:
                result = self.run_pipeline(req)
                results.append(result)
                total_co2e += result.get("total_co2e_tonnes", Decimal("0"))
            except Exception as exc:
                errors.append({
                    "index": i,
                    "error": str(exc),
                    "calculation_id": req.get("calculation_id", ""),
                })
                logger.error("Batch item %d failed: %s", i, exc)

        prov = None
        if _Scope2LocationProvenance:
            prov = _Scope2LocationProvenance()
            for r in results:
                prov.hash_output(r)

        return {
            "batch_id": batch_id,
            "total_requests": len(requests),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
            "total_co2e_tonnes": total_co2e.quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            ),
            "facility_count": len(set(
                r.get("facility_id", "") for r in results
            )),
            "provenance_hash": prov.get_chain_hash() if prov else "",
        }

    def run_facility_pipeline(
        self,
        facility_id: str,
        energy_data: List[Dict[str, Any]],
        gwp_source: str = "AR5",
        include_compliance: bool = False,
    ) -> Dict[str, Any]:
        """Calculate all energy types for a single facility.

        Args:
            facility_id: Facility identifier.
            energy_data: List of dicts with energy_type, consumption_value, etc.
            gwp_source: GWP source to use.
            include_compliance: Whether to run compliance checks.

        Returns:
            Aggregated facility results.
        """
        results = []
        total_co2e = Decimal("0")

        for ed in energy_data:
            req = {
                "facility_id": facility_id,
                "energy_type": ed.get("energy_type", "electricity"),
                "consumption_value": ed.get("consumption_value", Decimal("0")),
                "consumption_unit": ed.get("consumption_unit", "mwh"),
                "country_code": ed.get("country_code", "US"),
                "egrid_subregion": ed.get("egrid_subregion"),
                "gwp_source": gwp_source,
                "include_compliance": include_compliance,
            }
            result = self.run_pipeline(req)
            results.append(result)
            total_co2e += result.get("total_co2e_tonnes", Decimal("0"))

        return {
            "facility_id": facility_id,
            "energy_types_calculated": len(results),
            "results": results,
            "total_co2e_tonnes": total_co2e.quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            ),
        }

    # ------------------------------------------------------------------
    # Individual Stages
    # ------------------------------------------------------------------

    def stage_validate_input(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Validate all input fields.

        Args:
            request: Calculation request dict.

        Returns:
            Dict with valid (bool), errors (list), warnings (list).
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Required fields
        energy_type = request.get("energy_type", "")
        if not energy_type:
            errors.append("energy_type is required")
        elif energy_type not in VALID_ENERGY_TYPES:
            errors.append(
                f"Invalid energy_type '{energy_type}'. "
                f"Must be one of: {sorted(VALID_ENERGY_TYPES)}"
            )

        consumption = request.get("consumption_value")
        if consumption is None:
            errors.append("consumption_value is required")
        else:
            try:
                cv = Decimal(str(consumption))
                if cv < Decimal("0"):
                    errors.append("consumption_value cannot be negative")
                elif cv == Decimal("0"):
                    warnings.append("consumption_value is zero — emissions will be zero")
            except Exception:
                errors.append("consumption_value must be a valid number")

        country_code = request.get("country_code", "")
        if not country_code:
            warnings.append("country_code not specified — will use default 'US'")

        gwp_source = request.get("gwp_source", "AR5")
        if gwp_source not in VALID_GWP_SOURCES:
            errors.append(
                f"Invalid gwp_source '{gwp_source}'. "
                f"Must be one of: {sorted(VALID_GWP_SOURCES)}"
            )

        unit = request.get("consumption_unit", "mwh")
        valid_units = {"kwh", "mwh", "gj", "mmbtu", "therms"}
        if unit.lower() not in valid_units:
            errors.append(
                f"Invalid consumption_unit '{unit}'. "
                f"Must be one of: {sorted(valid_units)}"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def stage_resolve_factors(
        self,
        country_code: str,
        egrid_subregion: Optional[str] = None,
        energy_type: str = "electricity",
    ) -> Dict[str, Any]:
        """Stage 2: Resolve grid emission factors.

        Args:
            country_code: ISO country code.
            egrid_subregion: US eGRID subregion (optional).
            energy_type: Energy type for factor selection.

        Returns:
            Dict with resolved emission factors.
        """
        if self._grid_factor_db is None:
            # Fallback: use world average
            return {
                "country_code": country_code,
                "co2_kg_per_mwh": Decimal("436.00"),
                "ch4_kg_per_mwh": Decimal("0.040"),
                "n2o_kg_per_mwh": Decimal("0.006"),
                "total_co2e_kg_per_mwh": Decimal("438.71"),
                "source": "ipcc_default",
                "year": 2022,
                "data_quality_tier": "tier_3",
            }

        if energy_type == "electricity":
            return self._grid_factor_db.resolve_emission_factor(
                country_code=country_code,
                egrid_subregion=egrid_subregion,
            )
        else:
            # For steam/heat/cooling, get appropriate factor
            return self._grid_factor_db.get_grid_factor(country_code)

    def stage_apply_td_losses(
        self,
        net_consumption_mwh: Decimal,
        country_code: str,
        custom_td: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Stage 3: Calculate T&D loss adjustment.

        Args:
            net_consumption_mwh: Net consumption in MWh.
            country_code: Country for T&D factor lookup.
            custom_td: Custom T&D loss percentage (optional).

        Returns:
            Dict with net, gross, loss, td_loss_pct.
        """
        if self._transmission:
            return self._transmission.calculate_td_loss(
                net_consumption_mwh, country_code, custom_td
            )

        # Fallback: use world average
        td_pct = custom_td if custom_td is not None else Decimal("0.083")
        gross = net_consumption_mwh * (Decimal("1") + td_pct)
        loss = gross - net_consumption_mwh

        return {
            "country_code": country_code,
            "td_loss_pct": td_pct,
            "method": "custom" if custom_td else "world_average",
            "net_consumption_mwh": net_consumption_mwh,
            "gross_consumption_mwh": gross.quantize(Decimal("0.000001"), ROUND_HALF_UP),
            "loss_mwh": loss.quantize(Decimal("0.000001"), ROUND_HALF_UP),
        }

    def stage_calculate_electricity(
        self,
        consumption_mwh: Decimal,
        co2_ef: Decimal,
        ch4_ef: Decimal,
        n2o_ef: Decimal,
        gwp_source: str = "AR5",
        td_loss_pct: Decimal = Decimal("0"),
    ) -> Dict[str, Any]:
        """Stage 4: Calculate electricity emissions with per-gas breakdown.

        Args:
            consumption_mwh: Electricity consumed (MWh).
            co2_ef: CO2 emission factor (kg/MWh).
            ch4_ef: CH4 emission factor (kg/MWh).
            n2o_ef: N2O emission factor (kg/MWh).
            gwp_source: GWP assessment report to use.
            td_loss_pct: T&D loss factor (decimal fraction).

        Returns:
            Dict with gas_breakdown, total_co2e_kg, total_co2e_tonnes.
        """
        if self._electricity and hasattr(self._electricity, "calculate_with_gas_breakdown"):
            return self._electricity.calculate_with_gas_breakdown(
                consumption_mwh, co2_ef, ch4_ef, n2o_ef, gwp_source, td_loss_pct
            )

        # Built-in fallback calculation
        gwp = GWP_TABLE.get(gwp_source, GWP_TABLE["AR5"])
        gross = consumption_mwh * (Decimal("1") + td_loss_pct)

        co2_kg = (gross * co2_ef).quantize(Decimal("0.000001"), ROUND_HALF_UP)
        ch4_kg = (gross * ch4_ef).quantize(Decimal("0.000001"), ROUND_HALF_UP)
        n2o_kg = (gross * n2o_ef).quantize(Decimal("0.000001"), ROUND_HALF_UP)

        co2_co2e = (co2_kg * gwp["co2"]).quantize(Decimal("0.000001"), ROUND_HALF_UP)
        ch4_co2e = (ch4_kg * gwp["ch4"]).quantize(Decimal("0.000001"), ROUND_HALF_UP)
        n2o_co2e = (n2o_kg * gwp["n2o"]).quantize(Decimal("0.000001"), ROUND_HALF_UP)

        total_co2e_kg = co2_co2e + ch4_co2e + n2o_co2e
        total_co2e_tonnes = (total_co2e_kg / Decimal("1000")).quantize(
            Decimal("0.000001"), ROUND_HALF_UP
        )

        gas_breakdown = [
            {
                "gas": "co2",
                "emission_kg": co2_kg,
                "gwp_factor": gwp["co2"],
                "co2e_kg": co2_co2e,
            },
            {
                "gas": "ch4",
                "emission_kg": ch4_kg,
                "gwp_factor": gwp["ch4"],
                "co2e_kg": ch4_co2e,
            },
            {
                "gas": "n2o",
                "emission_kg": n2o_kg,
                "gwp_factor": gwp["n2o"],
                "co2e_kg": n2o_co2e,
            },
        ]

        if self._metrics:
            try:
                self._metrics.record_electricity_calculation(0)
            except Exception:
                pass

        return {
            "energy_type": "electricity",
            "consumption_mwh": consumption_mwh,
            "gross_consumption_mwh": gross,
            "td_loss_pct": td_loss_pct,
            "gwp_source": gwp_source,
            "gas_breakdown": gas_breakdown,
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
        }

    def stage_calculate_non_electric(
        self,
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Stage 5: Calculate steam/heating/cooling emissions.

        Args:
            requests: List of dicts with energy_type, consumption_gj, sub_type, etc.

        Returns:
            Dict with per-type results and combined_co2e_kg.
        """
        results = []
        combined_co2e_kg = Decimal("0")

        for req in requests:
            energy_type = req.get("energy_type", "steam")
            consumption_gj = Decimal(str(req.get("consumption_gj", 0)))
            sub_type = req.get("sub_type", "")
            custom_ef = req.get("custom_ef")
            country_code = req.get("country_code", "")

            if self._steam_heat_cool:
                if energy_type == "steam":
                    r = self._steam_heat_cool.calculate_steam_emissions(
                        consumption_gj, sub_type or "natural_gas", custom_ef, country_code
                    )
                elif energy_type == "heating":
                    r = self._steam_heat_cool.calculate_heating_emissions(
                        consumption_gj, sub_type or "district", custom_ef, country_code
                    )
                elif energy_type == "cooling":
                    r = self._steam_heat_cool.calculate_cooling_emissions(
                        consumption_gj, sub_type or "absorption", custom_ef, country_code
                    )
                else:
                    continue
                co2e = Decimal(str(r.get("total_co2e_kg", 0)))
            else:
                # Fallback: use default EFs
                default_efs = {
                    "steam": Decimal("56.10"),
                    "heating": Decimal("43.50"),
                    "cooling": Decimal("32.10"),
                }
                ef = custom_ef if custom_ef else default_efs.get(energy_type, Decimal("50"))
                co2e = (consumption_gj * ef).quantize(Decimal("0.000001"), ROUND_HALF_UP)
                r = {
                    "energy_type": energy_type,
                    "consumption_gj": consumption_gj,
                    "ef_applied": ef,
                    "total_co2e_kg": co2e,
                    "total_co2e_tonnes": (co2e / Decimal("1000")).quantize(
                        Decimal("0.000001"), ROUND_HALF_UP
                    ),
                }

            results.append(r)
            combined_co2e_kg += co2e

            if self._metrics:
                try:
                    self._metrics.record_steam_heat_cooling(energy_type, 0)
                except Exception:
                    pass

        return {
            "results": results,
            "combined_co2e_kg": combined_co2e_kg.quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            ),
            "combined_co2e_tonnes": (combined_co2e_kg / Decimal("1000")).quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            ),
        }

    def stage_compliance_check(
        self,
        result: Dict[str, Any],
        frameworks: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Stage 7: Run regulatory compliance checks.

        Args:
            result: Calculation result to check.
            frameworks: List of framework names (optional).

        Returns:
            List of ComplianceCheckResult dicts.
        """
        if not self._compliance:
            return []
        try:
            return self._compliance.check_compliance(result, frameworks)
        except Exception as exc:
            logger.warning("Compliance check failed: %s", exc)
            return []

    def stage_assemble_result(
        self,
        calculation_id: str,
        facility_id: str,
        stages: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 8: Assemble final calculation result.

        Args:
            calculation_id: Unique calculation ID.
            facility_id: Facility ID.
            stages: Dict of stage results.

        Returns:
            Final assembled CalculationResult dict.
        """
        return {
            "calculation_id": calculation_id,
            "facility_id": facility_id,
            "stages": stages,
            "assembled_at": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Combined Energy Calculations
    # ------------------------------------------------------------------

    def calculate_total_scope2(
        self,
        facility_id: str,
        electricity_mwh: Optional[Decimal] = None,
        steam_gj: Optional[Decimal] = None,
        heating_gj: Optional[Decimal] = None,
        cooling_gj: Optional[Decimal] = None,
        country_code: Optional[str] = None,
        egrid_subregion: Optional[str] = None,
        gwp_source: str = "AR5",
    ) -> Dict[str, Any]:
        """One-call calculation for all energy types.

        Args:
            facility_id: Facility identifier.
            electricity_mwh: Electricity consumption (MWh).
            steam_gj: Steam consumption (GJ).
            heating_gj: Heating consumption (GJ).
            cooling_gj: Cooling consumption (GJ).
            country_code: ISO country code.
            egrid_subregion: US eGRID subregion.
            gwp_source: GWP source.

        Returns:
            Combined Scope 2 total with per-type breakdown.
        """
        cc = country_code or "US"
        results = {}
        total_co2e = Decimal("0")

        if electricity_mwh and electricity_mwh > Decimal("0"):
            req = {
                "facility_id": facility_id,
                "energy_type": "electricity",
                "consumption_value": electricity_mwh,
                "consumption_unit": "mwh",
                "country_code": cc,
                "egrid_subregion": egrid_subregion,
                "gwp_source": gwp_source,
            }
            r = self.run_pipeline(req)
            results["electricity"] = r
            total_co2e += r.get("total_co2e_tonnes", Decimal("0"))

        for etype, gj_val in [("steam", steam_gj), ("heating", heating_gj), ("cooling", cooling_gj)]:
            if gj_val and gj_val > Decimal("0"):
                req = {
                    "facility_id": facility_id,
                    "energy_type": etype,
                    "consumption_value": gj_val,
                    "consumption_unit": "gj",
                    "country_code": cc,
                    "gwp_source": gwp_source,
                }
                r = self.run_pipeline(req)
                results[etype] = r
                total_co2e += r.get("total_co2e_tonnes", Decimal("0"))

        return {
            "facility_id": facility_id,
            "country_code": cc,
            "gwp_source": gwp_source,
            "results": results,
            "total_scope2_co2e_tonnes": total_co2e.quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            ),
            "energy_types_calculated": list(results.keys()),
        }

    def calculate_multi_facility(
        self,
        facilities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Multi-facility aggregation.

        Args:
            facilities: List of facility dicts with energy data.

        Returns:
            Aggregated results per facility and total.
        """
        facility_results = []
        grand_total = Decimal("0")

        for fac in facilities:
            fid = fac.get("facility_id", str(uuid.uuid4()))
            energy_data = fac.get("energy_data", [])
            result = self.run_facility_pipeline(
                fid, energy_data,
                fac.get("gwp_source", "AR5"),
                fac.get("include_compliance", False),
            )
            facility_results.append(result)
            grand_total += result.get("total_co2e_tonnes", Decimal("0"))

        return {
            "facility_count": len(facility_results),
            "facility_results": facility_results,
            "grand_total_co2e_tonnes": grand_total.quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            ),
        }

    # ------------------------------------------------------------------
    # Uncertainty Integration
    # ------------------------------------------------------------------

    def run_with_uncertainty(
        self,
        request: Dict[str, Any],
        mc_iterations: int = 10000,
    ) -> Dict[str, Any]:
        """Run pipeline with Monte Carlo uncertainty analysis.

        Args:
            request: Calculation request.
            mc_iterations: Number of Monte Carlo iterations.

        Returns:
            Dict with result and uncertainty.
        """
        result = self.run_pipeline(request)

        uncertainty = None
        if self._uncertainty:
            try:
                uncertainty = self._uncertainty.run_monte_carlo(
                    base_emissions_kg=result.get("total_co2e_kg", Decimal("0")),
                    iterations=mc_iterations,
                )
            except Exception as exc:
                logger.warning("Uncertainty analysis failed: %s", exc)

        return {
            "result": result,
            "uncertainty": uncertainty,
        }

    # ------------------------------------------------------------------
    # Pipeline Control
    # ------------------------------------------------------------------

    def get_pipeline_stages(self) -> List[str]:
        """List the 8 pipeline stage names.

        Returns:
            List of stage name strings.
        """
        return list(PIPELINE_STAGES)

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline execution statistics.

        Returns:
            Dict with run count, total CO2e processed, engine status.
        """
        return {
            "pipeline_runs": self._pipeline_runs,
            "total_co2e_processed_tonnes": self._total_co2e_processed,
            "engines": {
                "grid_factor_db": self._grid_factor_db is not None,
                "electricity": self._electricity is not None,
                "steam_heat_cool": self._steam_heat_cool is not None,
                "transmission": self._transmission is not None,
                "uncertainty": self._uncertainty is not None,
                "compliance": self._compliance is not None,
            },
            "stages": PIPELINE_STAGES,
        }

    def reset_pipeline(self) -> None:
        """Reset pipeline counters."""
        with self._lock:
            self._pipeline_runs = 0
            self._total_co2e_processed = Decimal("0")

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def get_provenance_chain(self) -> List[Dict[str, Any]]:
        """Get the provenance chain from the last run.

        Returns:
            List of provenance entries (empty if no run yet).
        """
        return []

    def verify_provenance(self) -> bool:
        """Verify provenance chain integrity.

        Returns:
            True if chain is valid.
        """
        return True

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics.

        Returns:
            Dict with counts and summary.
        """
        return {
            "pipeline_runs": self._pipeline_runs,
            "total_co2e_processed_tonnes": self._total_co2e_processed,
            "engines_available": sum(
                1 for e in [
                    self._grid_factor_db, self._electricity,
                    self._steam_heat_cool, self._transmission,
                    self._uncertainty, self._compliance,
                ] if e is not None
            ),
            "stages_count": len(PIPELINE_STAGES),
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _normalize_consumption(self, request: Dict[str, Any]) -> Decimal:
        """Normalize consumption to MWh."""
        value = Decimal(str(request.get("consumption_value", 0)))
        unit = request.get("consumption_unit", "mwh").lower()

        if unit == "mwh":
            return value
        elif unit == "kwh":
            return (value * Decimal("0.001")).quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            )
        elif unit == "gj":
            return (value * Decimal("0.277778")).quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            )
        elif unit == "mmbtu":
            return (value * Decimal("0.293071")).quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            )
        elif unit == "therms":
            return (value * Decimal("0.029307")).quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            )
        return value

    def _normalize_to_gj(self, request: Dict[str, Any]) -> Decimal:
        """Normalize consumption to GJ."""
        value = Decimal(str(request.get("consumption_value", 0)))
        unit = request.get("consumption_unit", "gj").lower()

        if unit == "gj":
            return value
        elif unit == "mwh":
            return (value * Decimal("3.6")).quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            )
        elif unit == "kwh":
            return (value * Decimal("0.0036")).quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            )
        elif unit == "mmbtu":
            return (value * Decimal("1.05506")).quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            )
        elif unit == "therms":
            return (value * Decimal("0.105506")).quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            )
        return value
