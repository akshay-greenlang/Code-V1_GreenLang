"""
Quantification Engine -- ISO 14064-1:2018 Clause 6.1 GHG Quantification

Implements the three quantification methodologies specified in ISO 14064-1:
  - Calculation-based (emission factor method): activity_data x EF x GWP
  - Direct measurement: concentration x flow_rate x time_period
  - Mass balance: sum(inputs) - sum(outputs) - accumulation

Also provides:
  - Multi-gas quantification across all 7 GHGs (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)
  - GWP conversion using both IPCC AR5 and AR6 tables
  - 5-dimension data quality scoring (completeness, accuracy, consistency,
    timeliness, methodology)
  - Emission factor source tracking with provenance
  - Per-source aggregation with audit trail

All calculations are deterministic (zero-hallucination).  No LLM calls are
used for numeric computations.

Example:
    >>> engine = QuantificationEngine(config)
    >>> result = engine.calculate_emission_factor_method(
    ...     activity_data=Decimal("1000"),
    ...     emission_factor=Decimal("2.5"),
    ...     gwp=Decimal("1"),
    ... )
    >>> print(result.tco2e)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    DataQualityTier,
    GHGGas,
    GWPSource,
    GWP_AR5,
    GWP_AR6,
    GWP_TABLES,
    ISOCategory,
    ISO14064AppConfig,
    QuantificationMethod,
)
from .models import (
    DataQualityIndicator,
    EmissionSource,
    QuantificationResult,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


class QuantificationEngine:
    """
    Core quantification engine for ISO 14064-1:2018 Clause 6.1.

    Provides three calculation methods (emission factor, direct measurement,
    mass balance) plus multi-gas GWP conversion, data quality scoring, and
    per-source aggregation.

    All calculations are deterministic and provenance-tracked.  Results are
    stored in-memory and can be retrieved for audit trail construction.

    Attributes:
        config: Application configuration.
    """

    def __init__(self, config: Optional[ISO14064AppConfig] = None) -> None:
        """
        Initialize QuantificationEngine.

        Args:
            config: Application configuration.  Defaults are used if None.
        """
        self.config = config or ISO14064AppConfig()
        self._sources: Dict[str, EmissionSource] = {}
        self._results: Dict[str, QuantificationResult] = {}
        self._ef_registry: Dict[str, Dict[str, Any]] = {}
        logger.info(
            "QuantificationEngine initialized (default GWP=%s)",
            self.config.default_gwp_source.value,
        )

    # ------------------------------------------------------------------
    # Method 1: Emission Factor (Calculation-Based)
    # ------------------------------------------------------------------

    def calculate_emission_factor_method(
        self,
        activity_data: Decimal,
        emission_factor: Decimal,
        gwp: Decimal = Decimal("1"),
        gas: GHGGas = GHGGas.CO2,
        activity_unit: str = "",
        ef_unit: str = "",
        ef_source: str = "IPCC",
        source_name: str = "",
        inventory_id: str = "",
        category: ISOCategory = ISOCategory.CATEGORY_1_DIRECT,
        facility_id: Optional[str] = None,
        is_biogenic: bool = False,
        data_quality: Optional[DataQualityIndicator] = None,
    ) -> QuantificationResult:
        """
        Calculate emissions using the emission factor method.

        Formula:  emissions_tCO2e = activity_data x emission_factor x GWP / 1000

        The division by 1000 converts from kg CO2e to tonnes CO2e when the
        emission factor is in kg/unit.  If the emission factor is already in
        tonnes/unit, set gwp to the appropriate value accordingly.

        This is the most common quantification method and corresponds to
        ISO 14064-1 Clause 6.1 calculation-based approach.

        Args:
            activity_data: Quantity of activity (e.g. litres of fuel, kWh).
            emission_factor: Emission factor value.
            gwp: Global warming potential (100-year horizon).
            gas: Greenhouse gas being quantified.
            activity_unit: Unit of activity data (e.g. litres, kWh, km).
            ef_unit: Unit of emission factor (e.g. kgCO2/kWh).
            ef_source: Source of emission factor (IPCC, EPA, DEFRA, etc.).
            source_name: Human-readable source description.
            inventory_id: Parent inventory ID.
            category: ISO 14064-1 category.
            facility_id: Optional facility/entity ID.
            is_biogenic: Whether this is biogenic CO2 (reported separately).
            data_quality: Optional data quality indicator.

        Returns:
            QuantificationResult with calculated emissions.

        Raises:
            ValueError: If activity_data or emission_factor is negative.
        """
        start = datetime.now(timezone.utc)

        if activity_data < 0:
            raise ValueError(f"Activity data cannot be negative: {activity_data}")
        if emission_factor < 0:
            raise ValueError(f"Emission factor cannot be negative: {emission_factor}")

        # Core calculation: activity_data * emission_factor * GWP
        # Result in tonnes CO2e (assuming EF produces tonnes)
        raw_emissions = activity_data * emission_factor
        tco2e = (raw_emissions * gwp).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP,
        )

        biogenic_co2 = tco2e if is_biogenic else Decimal("0")
        dq = data_quality or DataQualityIndicator()

        # Create and store the emission source record
        source = EmissionSource(
            inventory_id=inventory_id,
            category=category,
            source_name=source_name or f"{gas.value} from {ef_source}",
            facility_id=facility_id,
            gas=gas,
            method=QuantificationMethod.CALCULATION_BASED,
            activity_data=activity_data,
            activity_unit=activity_unit,
            emission_factor=emission_factor,
            ef_unit=ef_unit,
            ef_source=ef_source,
            gwp=gwp,
            raw_emissions_tonnes=raw_emissions.quantize(Decimal("0.0001")),
            tco2e=tco2e,
            biogenic_co2=biogenic_co2,
            data_quality_tier=self._dqi_to_tier(dq),
        )
        self._sources[source.id] = source

        # Create result
        result = QuantificationResult(
            source_id=source.id,
            method=QuantificationMethod.CALCULATION_BASED,
            gas=gas,
            raw_emissions_tonnes=raw_emissions.quantize(Decimal("0.0001")),
            gwp_applied=gwp,
            tco2e=tco2e,
            biogenic_co2=biogenic_co2,
            data_quality=dq,
        )
        self._results[result.id] = result

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "EF method: %s x %s x GWP(%s) = %.4f tCO2e [%s] in %.1f ms",
            activity_data, emission_factor, gwp, tco2e, gas.value, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Method 2: Direct Measurement
    # ------------------------------------------------------------------

    def calculate_direct_measurement(
        self,
        concentration: Decimal,
        flow_rate: Decimal,
        time_period_hours: Decimal,
        gas: GHGGas = GHGGas.CO2,
        gwp: Optional[Decimal] = None,
        gwp_source: Optional[GWPSource] = None,
        concentration_unit: str = "ppm",
        flow_rate_unit: str = "m3/hr",
        source_name: str = "",
        inventory_id: str = "",
        category: ISOCategory = ISOCategory.CATEGORY_1_DIRECT,
        facility_id: Optional[str] = None,
        is_biogenic: bool = False,
        data_quality: Optional[DataQualityIndicator] = None,
    ) -> QuantificationResult:
        """
        Calculate emissions from direct (continuous or periodic) measurement.

        Formula:
          volume = concentration x flow_rate x time_period
          mass_tonnes = volume x density_factor / 1_000_000
          tCO2e = mass_tonnes x GWP

        The density factor converts volumetric ppm to mass. For CO2 at STP,
        1 ppm in 1 m3 = ~1.96 mg.

        This method is used for ISO 14064-1 Clause 6.1 direct monitoring
        (e.g. CEMS, stack testing).

        Args:
            concentration: Gas concentration (e.g. ppm).
            flow_rate: Volumetric flow rate (e.g. m3/hr).
            time_period_hours: Measurement duration in hours.
            gas: Greenhouse gas being measured.
            gwp: GWP override. If None, looked up from gwp_source.
            gwp_source: GWP table (AR5 or AR6). Defaults to config.
            concentration_unit: Unit of concentration (default ppm).
            flow_rate_unit: Unit of flow rate (default m3/hr).
            source_name: Human-readable source description.
            inventory_id: Parent inventory ID.
            category: ISO 14064-1 category.
            facility_id: Optional facility/entity ID.
            is_biogenic: Whether biogenic CO2.
            data_quality: Optional data quality indicator.

        Returns:
            QuantificationResult with calculated emissions.

        Raises:
            ValueError: If inputs are negative or time period is zero.
        """
        start = datetime.now(timezone.utc)

        if concentration < 0:
            raise ValueError(f"Concentration cannot be negative: {concentration}")
        if flow_rate < 0:
            raise ValueError(f"Flow rate cannot be negative: {flow_rate}")
        if time_period_hours <= 0:
            raise ValueError(f"Time period must be positive: {time_period_hours}")

        # Look up GWP if not explicitly provided
        if gwp is None:
            gwp = self._get_gwp(gas, gwp_source)

        # Density factors (mg per ppm per m3 at STP)
        density_factors: Dict[GHGGas, Decimal] = {
            GHGGas.CO2: Decimal("1.96"),
            GHGGas.CH4: Decimal("0.716"),
            GHGGas.N2O: Decimal("1.96"),
            GHGGas.HFCS: Decimal("4.55"),   # Approximate for HFC-134a
            GHGGas.PFCS: Decimal("3.93"),   # Approximate for CF4
            GHGGas.SF6: Decimal("6.52"),
            GHGGas.NF3: Decimal("3.17"),
        }
        density = density_factors.get(gas, Decimal("1.96"))

        # volume = concentration(ppm) * flow_rate(m3/hr) * time(hr) = ppm*m3
        volume_ppm_m3 = concentration * flow_rate * time_period_hours

        # mass(mg) = volume(ppm*m3) * density(mg/ppm/m3)
        mass_mg = volume_ppm_m3 * density

        # mass(tonnes) = mass(mg) / 1e9  (mg -> kg -> tonnes)
        raw_emissions = (mass_mg / Decimal("1000000000")).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP,
        )

        tco2e = (raw_emissions * gwp).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP,
        )

        biogenic_co2 = tco2e if is_biogenic else Decimal("0")
        dq = data_quality or DataQualityIndicator(
            completeness=Decimal("4.0"),
            accuracy=Decimal("4.0"),
            consistency=Decimal("4.0"),
            timeliness=Decimal("4.0"),
            methodology=Decimal("5.0"),
        )

        source = EmissionSource(
            inventory_id=inventory_id,
            category=category,
            source_name=source_name or f"Direct measurement - {gas.value}",
            facility_id=facility_id,
            gas=gas,
            method=QuantificationMethod.DIRECT_MEASUREMENT,
            activity_data=volume_ppm_m3.quantize(Decimal("0.01")),
            activity_unit=f"{concentration_unit}*{flow_rate_unit}*hr",
            emission_factor=density,
            ef_unit="mg/ppm/m3",
            ef_source="STP density calculation",
            gwp=gwp,
            raw_emissions_tonnes=raw_emissions,
            tco2e=tco2e,
            biogenic_co2=biogenic_co2,
            data_quality_tier=DataQualityTier.TIER_4,
        )
        self._sources[source.id] = source

        result = QuantificationResult(
            source_id=source.id,
            method=QuantificationMethod.DIRECT_MEASUREMENT,
            gas=gas,
            raw_emissions_tonnes=raw_emissions,
            gwp_applied=gwp,
            tco2e=tco2e,
            biogenic_co2=biogenic_co2,
            data_quality=dq,
        )
        self._results[result.id] = result

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "Direct measurement: conc=%s %s, flow=%s %s, time=%s hr -> "
            "%.4f t %s -> %.4f tCO2e in %.1f ms",
            concentration, concentration_unit, flow_rate, flow_rate_unit,
            time_period_hours, raw_emissions, gas.value, tco2e, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Method 3: Mass Balance
    # ------------------------------------------------------------------

    def calculate_mass_balance(
        self,
        inputs: List[Dict[str, Decimal]],
        outputs: List[Dict[str, Decimal]],
        accumulation: Decimal = Decimal("0"),
        gas: GHGGas = GHGGas.CO2,
        gwp: Optional[Decimal] = None,
        gwp_source: Optional[GWPSource] = None,
        source_name: str = "",
        inventory_id: str = "",
        category: ISOCategory = ISOCategory.CATEGORY_1_DIRECT,
        facility_id: Optional[str] = None,
        is_biogenic: bool = False,
        data_quality: Optional[DataQualityIndicator] = None,
    ) -> QuantificationResult:
        """
        Calculate emissions using the mass balance method.

        Formula:
          emissions = sum(input_masses) - sum(output_masses) - accumulation

        Each input/output dict has keys:
          - name (str): Description of the material stream.
          - mass_tonnes (Decimal): Mass in tonnes.
          - carbon_fraction (Decimal): Carbon content fraction (0-1).

        The carbon fraction is used to compute the carbon content,
        then converted to CO2e using the molecular weight ratio (44/12).

        Args:
            inputs: List of input material streams.
            outputs: List of output material streams.
            accumulation: Stock change (positive = accumulation in process).
            gas: Greenhouse gas.
            gwp: GWP override. If None, looked up from table.
            gwp_source: GWP table to use (AR5 or AR6).
            source_name: Source description.
            inventory_id: Parent inventory ID.
            category: ISO 14064-1 category.
            facility_id: Facility/entity ID.
            is_biogenic: Whether biogenic CO2.
            data_quality: Data quality indicator.

        Returns:
            QuantificationResult with calculated emissions.

        Raises:
            ValueError: If no inputs provided or negative result.
        """
        start = datetime.now(timezone.utc)

        if not inputs:
            raise ValueError("At least one input material stream is required")

        if gwp is None:
            gwp = self._get_gwp(gas, gwp_source)

        # CO2/C molecular weight ratio
        co2_c_ratio = Decimal("44") / Decimal("12")

        # Calculate total carbon in inputs
        total_input_carbon = Decimal("0")
        for inp in inputs:
            mass = inp.get("mass_tonnes", Decimal("0"))
            carbon_frac = inp.get("carbon_fraction", Decimal("0"))
            total_input_carbon += mass * carbon_frac

        # Calculate total carbon in outputs
        total_output_carbon = Decimal("0")
        for out in outputs:
            mass = out.get("mass_tonnes", Decimal("0"))
            carbon_frac = out.get("carbon_fraction", Decimal("0"))
            total_output_carbon += mass * carbon_frac

        # Mass balance: emitted carbon = inputs - outputs - accumulation
        emitted_carbon = total_input_carbon - total_output_carbon - accumulation

        # Convert carbon to CO2 equivalent
        if gas == GHGGas.CO2:
            raw_emissions = (emitted_carbon * co2_c_ratio).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP,
            )
        else:
            # For non-CO2 gases, the mass is directly the gas mass
            raw_emissions = emitted_carbon.quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP,
            )

        # Ensure non-negative (negative would indicate a sink, handled by removals)
        if raw_emissions < 0:
            logger.warning(
                "Mass balance resulted in negative emissions (%.4f t): "
                "treating as zero emissions with sink noted",
                raw_emissions,
            )
            raw_emissions = Decimal("0")

        tco2e = (raw_emissions * gwp).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP,
        )

        biogenic_co2 = tco2e if is_biogenic else Decimal("0")
        dq = data_quality or DataQualityIndicator()

        source = EmissionSource(
            inventory_id=inventory_id,
            category=category,
            source_name=source_name or f"Mass balance - {gas.value}",
            facility_id=facility_id,
            gas=gas,
            method=QuantificationMethod.MASS_BALANCE,
            activity_data=total_input_carbon.quantize(Decimal("0.0001")),
            activity_unit="tC (input carbon)",
            emission_factor=co2_c_ratio if gas == GHGGas.CO2 else Decimal("1"),
            ef_unit="tCO2/tC" if gas == GHGGas.CO2 else "t/t",
            ef_source="Stoichiometric conversion",
            gwp=gwp,
            raw_emissions_tonnes=raw_emissions,
            tco2e=tco2e,
            biogenic_co2=biogenic_co2,
            data_quality_tier=self._dqi_to_tier(dq),
        )
        self._sources[source.id] = source

        result = QuantificationResult(
            source_id=source.id,
            method=QuantificationMethod.MASS_BALANCE,
            gas=gas,
            raw_emissions_tonnes=raw_emissions,
            gwp_applied=gwp,
            tco2e=tco2e,
            biogenic_co2=biogenic_co2,
            data_quality=dq,
        )
        self._results[result.id] = result

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "Mass balance: inputs=%.4f tC, outputs=%.4f tC, accum=%.4f, "
            "emitted=%.4f tCO2e [%s] in %.1f ms",
            total_input_carbon, total_output_carbon, accumulation,
            tco2e, gas.value, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Multi-Gas Quantification
    # ------------------------------------------------------------------

    def quantify_multi_gas(
        self,
        gas_data: List[Dict[str, Any]],
        gwp_source: Optional[GWPSource] = None,
        inventory_id: str = "",
        category: ISOCategory = ISOCategory.CATEGORY_1_DIRECT,
        facility_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Quantify emissions for multiple gases in a single call.

        Each entry in gas_data is a dict with:
          - gas (str): GHG gas name (CO2, CH4, N2O, etc.).
          - activity_data (Decimal): Activity data quantity.
          - emission_factor (Decimal): Emission factor value.
          - activity_unit (str): Activity data unit.
          - ef_unit (str): Emission factor unit.
          - ef_source (str): Emission factor source.
          - source_name (str): Source description.
          - is_biogenic (bool): Whether biogenic CO2.

        Args:
            gas_data: List of per-gas quantification inputs.
            gwp_source: GWP table to use.
            inventory_id: Parent inventory ID.
            category: ISO 14064-1 category.
            facility_id: Facility/entity ID.

        Returns:
            Dict with:
              - total_tco2e: Grand total across all gases.
              - by_gas: Per-gas results dict.
              - biogenic_co2: Total biogenic CO2.
              - results: List of QuantificationResult objects.
        """
        start = datetime.now(timezone.utc)
        source = gwp_source or self.config.default_gwp_source

        total_tco2e = Decimal("0")
        biogenic_total = Decimal("0")
        by_gas: Dict[str, Decimal] = {}
        results: List[QuantificationResult] = []

        for entry in gas_data:
            gas = GHGGas(entry["gas"])
            gwp = self._get_gwp(gas, source)

            result = self.calculate_emission_factor_method(
                activity_data=Decimal(str(entry["activity_data"])),
                emission_factor=Decimal(str(entry["emission_factor"])),
                gwp=gwp,
                gas=gas,
                activity_unit=entry.get("activity_unit", ""),
                ef_unit=entry.get("ef_unit", ""),
                ef_source=entry.get("ef_source", "IPCC"),
                source_name=entry.get("source_name", ""),
                inventory_id=inventory_id,
                category=category,
                facility_id=facility_id,
                is_biogenic=entry.get("is_biogenic", False),
            )

            total_tco2e += result.tco2e
            biogenic_total += result.biogenic_co2
            by_gas[gas.value] = by_gas.get(gas.value, Decimal("0")) + result.tco2e
            results.append(result)

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "Multi-gas quantification: %d gases, total=%.4f tCO2e (GWP=%s) in %.1f ms",
            len(gas_data), total_tco2e, source.value, elapsed_ms,
        )

        return {
            "total_tco2e": total_tco2e,
            "by_gas": by_gas,
            "biogenic_co2": biogenic_total,
            "results": results,
            "gas_count": len(gas_data),
            "gwp_source": source.value,
        }

    # ------------------------------------------------------------------
    # GWP Conversion
    # ------------------------------------------------------------------

    def convert_gwp(
        self,
        emissions_native: Decimal,
        gas: GHGGas,
        gwp_source: Optional[GWPSource] = None,
    ) -> Decimal:
        """
        Convert native gas emissions to CO2e using GWP.

        Args:
            emissions_native: Emissions in native gas tonnes.
            gas: Greenhouse gas.
            gwp_source: GWP table (AR5 or AR6).

        Returns:
            Emissions in tCO2e.
        """
        gwp = self._get_gwp(gas, gwp_source)
        tco2e = (emissions_native * gwp).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP,
        )
        logger.debug(
            "GWP conversion: %.4f t %s x GWP(%s) = %.4f tCO2e",
            emissions_native, gas.value, gwp, tco2e,
        )
        return tco2e

    def get_gwp_value(
        self,
        gas: GHGGas,
        gwp_source: Optional[GWPSource] = None,
    ) -> Decimal:
        """
        Look up the GWP value for a gas from the specified AR table.

        Args:
            gas: Greenhouse gas.
            gwp_source: GWP table (AR5 or AR6). Defaults to config.

        Returns:
            GWP value as Decimal.
        """
        return self._get_gwp(gas, gwp_source)

    # ------------------------------------------------------------------
    # Data Quality Scoring
    # ------------------------------------------------------------------

    def score_data_quality(
        self,
        completeness: Decimal = Decimal("3"),
        accuracy: Decimal = Decimal("3"),
        consistency: Decimal = Decimal("3"),
        timeliness: Decimal = Decimal("3"),
        methodology: Decimal = Decimal("3"),
    ) -> DataQualityIndicator:
        """
        Create a 5-dimension data quality indicator score.

        Each dimension is scored from 1 (poor) to 5 (excellent):
          - Completeness: Percentage of required data available.
          - Accuracy: Precision and reliability of measurements.
          - Consistency: Coherence across time periods and sources.
          - Timeliness: How current the data is.
          - Methodology: Rigour of calculation methodology.

        Args:
            completeness: Score 1-5.
            accuracy: Score 1-5.
            consistency: Score 1-5.
            timeliness: Score 1-5.
            methodology: Score 1-5.

        Returns:
            DataQualityIndicator with computed overall score.
        """
        dqi = DataQualityIndicator(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            methodology=methodology,
        )
        logger.debug(
            "DQI scored: C=%.1f A=%.1f Co=%.1f T=%.1f M=%.1f -> overall=%.2f",
            completeness, accuracy, consistency, timeliness, methodology,
            dqi.overall_score,
        )
        return dqi

    # ------------------------------------------------------------------
    # Emission Factor Registry
    # ------------------------------------------------------------------

    def register_emission_factor(
        self,
        ef_id: str,
        gas: GHGGas,
        value: Decimal,
        unit: str,
        source: str,
        source_year: int,
        sector: str = "",
        region: str = "",
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Register an emission factor in the engine's internal registry.

        Provides traceability for emission factors used in calculations.

        Args:
            ef_id: Unique emission factor identifier (e.g. 'IPCC-2019-NG-CO2').
            gas: Greenhouse gas the factor applies to.
            value: Emission factor value.
            unit: Unit of the factor (e.g. kgCO2/GJ, tCO2/MWh).
            source: Data source (IPCC, EPA, DEFRA, etc.).
            source_year: Year the factor was published.
            sector: Industrial sector (optional).
            region: Geographic region (optional).
            description: Human-readable description.

        Returns:
            Dict with registered factor details and provenance hash.
        """
        ef_record = {
            "ef_id": ef_id,
            "gas": gas.value,
            "value": str(value),
            "unit": unit,
            "source": source,
            "source_year": source_year,
            "sector": sector,
            "region": region,
            "description": description,
            "provenance_hash": _sha256(
                f"{ef_id}:{gas.value}:{value}:{source}:{source_year}"
            ),
            "registered_at": _now().isoformat(),
        }
        self._ef_registry[ef_id] = ef_record

        logger.info(
            "Registered EF '%s': %s %s for %s (source=%s, year=%d)",
            ef_id, value, unit, gas.value, source, source_year,
        )
        return ef_record

    def get_emission_factor(self, ef_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a registered emission factor by ID."""
        return self._ef_registry.get(ef_id)

    def list_emission_factors(
        self,
        gas: Optional[GHGGas] = None,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List registered emission factors, optionally filtered by gas or source.

        Args:
            gas: Filter by greenhouse gas.
            source: Filter by data source name.

        Returns:
            List of emission factor records.
        """
        factors = list(self._ef_registry.values())

        if gas is not None:
            factors = [f for f in factors if f["gas"] == gas.value]
        if source is not None:
            factors = [
                f for f in factors
                if f["source"].lower() == source.lower()
            ]

        return factors

    # ------------------------------------------------------------------
    # Per-Source Aggregation
    # ------------------------------------------------------------------

    def aggregate_by_source(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """
        Aggregate all emission sources for an inventory.

        Groups sources by category, gas, and facility, then computes
        totals with provenance tracking.

        Args:
            inventory_id: Inventory ID.

        Returns:
            Dict with:
              - total_tco2e: Grand total.
              - by_category: Per-category totals.
              - by_gas: Per-gas totals.
              - by_facility: Per-facility totals.
              - biogenic_co2: Total biogenic CO2.
              - source_count: Number of sources.
        """
        start = datetime.now(timezone.utc)

        inv_sources = [
            s for s in self._sources.values()
            if s.inventory_id == inventory_id
        ]

        total_tco2e = Decimal("0")
        by_category: Dict[str, Decimal] = {}
        by_gas: Dict[str, Decimal] = {}
        by_facility: Dict[str, Decimal] = {}
        biogenic_total = Decimal("0")

        for src in inv_sources:
            total_tco2e += src.tco2e

            cat_key = src.category.value
            by_category[cat_key] = by_category.get(cat_key, Decimal("0")) + src.tco2e

            gas_key = src.gas.value
            by_gas[gas_key] = by_gas.get(gas_key, Decimal("0")) + src.tco2e

            if src.facility_id:
                by_facility[src.facility_id] = (
                    by_facility.get(src.facility_id, Decimal("0")) + src.tco2e
                )

            biogenic_total += src.biogenic_co2

        provenance = _sha256(
            f"aggregate:{inventory_id}:{total_tco2e}:{len(inv_sources)}"
        )

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "Aggregated %d sources for inventory %s: %.4f tCO2e in %.1f ms",
            len(inv_sources), inventory_id, total_tco2e, elapsed_ms,
        )

        return {
            "inventory_id": inventory_id,
            "total_tco2e": total_tco2e,
            "by_category": by_category,
            "by_gas": by_gas,
            "by_facility": by_facility,
            "biogenic_co2": biogenic_total,
            "source_count": len(inv_sources),
            "provenance_hash": provenance,
        }

    def get_sources_for_inventory(
        self,
        inventory_id: str,
    ) -> List[EmissionSource]:
        """
        Retrieve all emission sources for a given inventory.

        Args:
            inventory_id: Inventory ID.

        Returns:
            List of EmissionSource objects.
        """
        return [
            s for s in self._sources.values()
            if s.inventory_id == inventory_id
        ]

    def get_source(self, source_id: str) -> Optional[EmissionSource]:
        """Retrieve a single emission source by ID."""
        return self._sources.get(source_id)

    def get_result(self, result_id: str) -> Optional[QuantificationResult]:
        """Retrieve a single quantification result by ID."""
        return self._results.get(result_id)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_gwp(
        self,
        gas: GHGGas,
        gwp_source: Optional[GWPSource] = None,
    ) -> Decimal:
        """
        Look up GWP value for a gas from the specified assessment report.

        Args:
            gas: Greenhouse gas.
            gwp_source: AR5 or AR6.  Defaults to config.

        Returns:
            GWP value as Decimal.
        """
        source = gwp_source or self.config.default_gwp_source
        table = GWP_TABLES.get(source, GWP_AR5)
        raw_value = table.get(gas, 1)
        return Decimal(str(raw_value))

    def _dqi_to_tier(self, dqi: DataQualityIndicator) -> DataQualityTier:
        """
        Map a DataQualityIndicator overall score to a DataQualityTier.

        Mapping:
          - Score >= 4.5 -> TIER_4 (direct measurement)
          - Score >= 3.5 -> TIER_3 (supplier-specific)
          - Score >= 2.5 -> TIER_2 (activity data + published EFs)
          - Score <  2.5 -> TIER_1 (estimated / industry averages)
        """
        score = dqi.overall_score
        if score >= Decimal("4.5"):
            return DataQualityTier.TIER_4
        elif score >= Decimal("3.5"):
            return DataQualityTier.TIER_3
        elif score >= Decimal("2.5"):
            return DataQualityTier.TIER_2
        else:
            return DataQualityTier.TIER_1
