"""
GL-002 FLAMEGUARD - Emissions Calculator

Emissions calculations using EPA factors and CEMS data.
Supports NOx, CO, CO2, SO2, PM, and GHG reporting.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


# EPA emission factors (40 CFR Part 98)
EPA_EMISSION_FACTORS = {
    "natural_gas": {
        "co2_kg_mmbtu": 53.06,
        "ch4_kg_mmbtu": 0.001,
        "n2o_kg_mmbtu": 0.0001,
        "nox_lb_mmbtu": 0.098,  # Uncontrolled
        "co_lb_mmbtu": 0.082,
        "so2_lb_mmbtu": 0.0006,
        "pm_lb_mmbtu": 0.0075,
    },
    "fuel_oil_no2": {
        "co2_kg_mmbtu": 73.96,
        "ch4_kg_mmbtu": 0.003,
        "n2o_kg_mmbtu": 0.0006,
        "nox_lb_mmbtu": 0.15,
        "co_lb_mmbtu": 0.036,
        "so2_lb_mmbtu": 0.5,  # Depends on sulfur content
        "pm_lb_mmbtu": 0.02,
    },
    "coal_bituminous": {
        "co2_kg_mmbtu": 93.28,
        "ch4_kg_mmbtu": 0.011,
        "n2o_kg_mmbtu": 0.0016,
        "nox_lb_mmbtu": 0.5,
        "co_lb_mmbtu": 0.5,
        "so2_lb_mmbtu": 1.2,
        "pm_lb_mmbtu": 0.3,
    },
}

# Global Warming Potentials (AR5)
GWP = {
    "co2": 1,
    "ch4": 28,
    "n2o": 265,
}


@dataclass
class EmissionsInput:
    """Input for emissions calculation."""

    fuel_type: str
    heat_input_mmbtu_hr: float
    flue_gas_nox_ppm: Optional[float] = None
    flue_gas_co_ppm: Optional[float] = None
    flue_gas_o2_percent: float = 3.0
    sulfur_content_percent: Optional[float] = None
    stack_flow_acfm: Optional[float] = None
    stack_temperature_f: float = 400.0
    use_cems_data: bool = False


@dataclass
class EmissionsResult:
    """Emissions calculation result."""

    calculation_id: str
    timestamp: datetime

    # Mass emission rates
    co2_lb_hr: float
    co2_metric_tons_hr: float
    nox_lb_hr: float
    co_lb_hr: float
    so2_lb_hr: float
    pm_lb_hr: float

    # Emission factors (actual)
    co2_lb_mmbtu: float
    nox_lb_mmbtu: float
    co_lb_mmbtu: float
    so2_lb_mmbtu: float
    pm_lb_mmbtu: float

    # Corrected values
    nox_ppm_3pct_o2: float

    # GHG totals
    co2e_metric_tons_hr: float

    # Provenance (required - moved before defaults)
    input_hash: str
    method: str

    # Optional fields with defaults
    ghg_scope: int = 1

    # Compliance
    nox_compliant: bool = True
    co_compliant: bool = True
    so2_compliant: bool = True
    overall_compliant: bool = True


class EmissionsCalculator:
    """
    Emissions calculator for combustion sources.

    Supports:
    - EPA emission factor method
    - CEMS data integration
    - O2 correction for NOx
    - GHG Protocol reporting
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        emission_factors: Optional[Dict] = None,
        nox_limit_lb_mmbtu: float = 0.10,
        co_limit_lb_mmbtu: float = 0.08,
        so2_limit_lb_mmbtu: float = 0.50,
    ) -> None:
        """Initialize emissions calculator."""
        self.emission_factors = emission_factors or EPA_EMISSION_FACTORS
        self.nox_limit = nox_limit_lb_mmbtu
        self.co_limit = co_limit_lb_mmbtu
        self.so2_limit = so2_limit_lb_mmbtu

    def calculate(self, inp: EmissionsInput) -> EmissionsResult:
        """
        Calculate emissions from combustion.

        Args:
            inp: Emissions input data

        Returns:
            EmissionsResult with all pollutants
        """
        factors = self.emission_factors.get(
            inp.fuel_type,
            self.emission_factors["natural_gas"]
        )

        heat_input = inp.heat_input_mmbtu_hr

        # CO2 emissions (always use factor method)
        co2_kg_hr = heat_input * factors["co2_kg_mmbtu"]
        co2_lb_hr = co2_kg_hr * 2.205
        co2_mt_hr = co2_kg_hr / 1000

        # CH4 and N2O for CO2e
        ch4_kg_hr = heat_input * factors.get("ch4_kg_mmbtu", 0)
        n2o_kg_hr = heat_input * factors.get("n2o_kg_mmbtu", 0)
        co2e_mt_hr = (
            co2_kg_hr * GWP["co2"] +
            ch4_kg_hr * GWP["ch4"] +
            n2o_kg_hr * GWP["n2o"]
        ) / 1000

        # NOx emissions
        if inp.use_cems_data and inp.flue_gas_nox_ppm is not None:
            # Calculate from CEMS
            nox_lb_mmbtu = self._ppm_to_lb_mmbtu(
                inp.flue_gas_nox_ppm,
                inp.flue_gas_o2_percent,
                46.0,  # MW of NO2
            )
        else:
            nox_lb_mmbtu = factors["nox_lb_mmbtu"]

        nox_lb_hr = heat_input * nox_lb_mmbtu

        # Correct NOx to 3% O2
        nox_ppm_3pct = self._correct_to_3pct_o2(
            inp.flue_gas_nox_ppm or 50,
            inp.flue_gas_o2_percent,
        )

        # CO emissions
        if inp.use_cems_data and inp.flue_gas_co_ppm is not None:
            co_lb_mmbtu = self._ppm_to_lb_mmbtu(
                inp.flue_gas_co_ppm,
                inp.flue_gas_o2_percent,
                28.0,  # MW of CO
            )
        else:
            co_lb_mmbtu = factors["co_lb_mmbtu"]

        co_lb_hr = heat_input * co_lb_mmbtu

        # SO2 emissions
        if inp.sulfur_content_percent is not None:
            # Calculate from sulfur content
            # Assume all sulfur converts to SO2
            # SO2/S mass ratio = 64/32 = 2
            fuel_lb_hr = heat_input * 1e6 / 23875  # Approximate for gas
            so2_lb_hr = fuel_lb_hr * inp.sulfur_content_percent / 100 * 2
            so2_lb_mmbtu = so2_lb_hr / heat_input if heat_input > 0 else 0
        else:
            so2_lb_mmbtu = factors["so2_lb_mmbtu"]
            so2_lb_hr = heat_input * so2_lb_mmbtu

        # PM emissions
        pm_lb_mmbtu = factors["pm_lb_mmbtu"]
        pm_lb_hr = heat_input * pm_lb_mmbtu

        # Compliance check
        nox_compliant = nox_lb_mmbtu <= self.nox_limit
        co_compliant = co_lb_mmbtu <= self.co_limit
        so2_compliant = so2_lb_mmbtu <= self.so2_limit
        overall_compliant = nox_compliant and co_compliant and so2_compliant

        # Input hash
        input_hash = self._compute_hash(inp.__dict__)

        return EmissionsResult(
            calculation_id=f"EMIS-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(timezone.utc),
            co2_lb_hr=round(co2_lb_hr, 1),
            co2_metric_tons_hr=round(co2_mt_hr, 4),
            nox_lb_hr=round(nox_lb_hr, 3),
            co_lb_hr=round(co_lb_hr, 3),
            so2_lb_hr=round(so2_lb_hr, 3),
            pm_lb_hr=round(pm_lb_hr, 3),
            co2_lb_mmbtu=round(co2_lb_hr / heat_input if heat_input > 0 else 0, 2),
            nox_lb_mmbtu=round(nox_lb_mmbtu, 4),
            co_lb_mmbtu=round(co_lb_mmbtu, 4),
            so2_lb_mmbtu=round(so2_lb_mmbtu, 4),
            pm_lb_mmbtu=round(pm_lb_mmbtu, 4),
            nox_ppm_3pct_o2=round(nox_ppm_3pct, 1),
            co2e_metric_tons_hr=round(co2e_mt_hr, 4),
            nox_compliant=nox_compliant,
            co_compliant=co_compliant,
            so2_compliant=so2_compliant,
            overall_compliant=overall_compliant,
            input_hash=input_hash,
            method="cems" if inp.use_cems_data else "factor",
        )

    def _ppm_to_lb_mmbtu(
        self,
        ppm: float,
        o2_percent: float,
        molecular_weight: float,
    ) -> float:
        """Convert ppm to lb/MMBTU."""
        # F-factor for natural gas: 8710 dscf/MMBTU
        f_factor = 8710

        # Correct to 0% O2
        ppm_0pct = ppm * (21 - o2_percent) / 21

        # lb = ppm * MW / 385.1 * F_factor / 1e6
        return ppm_0pct * molecular_weight / 385.1 * f_factor / 1e6

    def _correct_to_3pct_o2(
        self,
        ppm: float,
        measured_o2: float,
    ) -> float:
        """Correct concentration to 3% O2 reference."""
        if measured_o2 >= 21:
            return ppm
        return ppm * (21 - 3) / (21 - measured_o2)

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
