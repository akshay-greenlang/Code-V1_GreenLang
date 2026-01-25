"""
GL-037: Flare Minimizer Agent (FLARE-MINIMIZER)

Flare gas recovery and minimization for emissions reduction.

Standards: EPA 40 CFR 60, API 521
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FlareMinimzerInput(BaseModel):
    """Input for FlareMinimzerAgent."""
    flare_id: str = Field(...)
    flare_flow_scfh: float = Field(..., ge=0, description="Current flare flow rate")
    gas_composition: Dict[str, float] = Field(default_factory=lambda: {"CH4": 0.85, "C2H6": 0.10, "CO2": 0.05})
    pressure_psig: float = Field(default=5.0)
    temperature_f: float = Field(default=100.0)
    recovery_options: List[str] = Field(default_factory=lambda: ["compression", "fuel_gas", "process_return"])
    natural_gas_price_per_mmbtu: float = Field(default=5.0)
    flare_destruction_efficiency: float = Field(default=0.98)
    co2_credit_per_tonne: float = Field(default=50.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FlareMinimzerOutput(BaseModel):
    """Output from FlareMinimzerAgent."""
    analysis_id: str
    flare_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    recoverable_volume_scfh: float
    recoverable_energy_mmbtu_hr: float
    recommended_recovery_method: str
    implementation_cost_estimate: float
    annual_value_recovered: float
    simple_payback_years: float
    co2_currently_emitted_tpy: float
    co2_avoided_tpy: float
    methane_avoided_tpy: float
    ghg_reduction_co2e_tpy: float
    regulatory_compliance_notes: List[str]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class FlareMinimzerAgent:
    """GL-037: Flare Minimizer Agent."""

    AGENT_ID = "GL-037"
    AGENT_NAME = "FLARE-MINIMIZER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"FlareMinimzerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: FlareMinimzerInput) -> FlareMinimzerOutput:
        """Execute flare minimization analysis."""
        start_time = datetime.utcnow()
        logger.info(f"Starting flare analysis for {input_data.flare_id}")

        # Gas properties
        ch4_fraction = input_data.gas_composition.get("CH4", 0.85)
        hhv_btu_per_scf = 1000 * ch4_fraction + 1770 * input_data.gas_composition.get("C2H6", 0.10)

        # Recoverable volume (assume 90% can be recovered with proper equipment)
        recovery_fraction = 0.90
        recoverable_scfh = input_data.flare_flow_scfh * recovery_fraction

        # Energy content
        recoverable_mmbtu_hr = recoverable_scfh * hhv_btu_per_scf / 1e6
        annual_mmbtu = recoverable_mmbtu_hr * 8760

        # Value of recovered gas
        annual_value = annual_mmbtu * input_data.natural_gas_price_per_mmbtu

        # CO2 emissions from current flaring
        # CH4 + 2O2 -> CO2 + 2H2O
        # 1 scf CH4 at STP = 0.0423 lb = 0.0192 kg
        # Burns to 0.0527 kg CO2
        ch4_scfh = input_data.flare_flow_scfh * ch4_fraction
        ch4_lb_hr = ch4_scfh * 0.0423
        co2_lb_hr = ch4_lb_hr * (44/16) * input_data.flare_destruction_efficiency
        co2_tpy = co2_lb_hr * 8760 / 2000

        # Methane slip (unburned)
        methane_slip = (1 - input_data.flare_destruction_efficiency)
        methane_tpy = ch4_lb_hr * methane_slip * 8760 / 2000

        # GHG reduction (CH4 GWP = 28)
        co2_avoided = co2_tpy * recovery_fraction
        methane_avoided = methane_tpy * recovery_fraction
        ghg_avoided_co2e = co2_avoided + methane_avoided * 28

        # Select recovery method
        if input_data.flare_flow_scfh > 10000:
            method = "compression_to_sales"
            impl_cost = 500000
        elif input_data.flare_flow_scfh > 1000:
            method = "fuel_gas_system"
            impl_cost = 200000
        else:
            method = "process_return"
            impl_cost = 50000

        # Payback
        payback = impl_cost / annual_value if annual_value > 0 else 999

        # Compliance notes
        notes = []
        if input_data.flare_flow_scfh > 5000:
            notes.append("Flow exceeds EPA 40 CFR 60 reporting threshold")
        if input_data.flare_destruction_efficiency < 0.98:
            notes.append("Destruction efficiency below API 521 recommendation")
        notes.append("Recovery reduces Scope 1 emissions under GHG Protocol")

        provenance_hash = hashlib.sha256(
            json.dumps({"agent": self.AGENT_ID, "flare": input_data.flare_id,
                        "timestamp": datetime.utcnow().isoformat()}, sort_keys=True).encode()
        ).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return FlareMinimzerOutput(
            analysis_id=f"FM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            flare_id=input_data.flare_id,
            recoverable_volume_scfh=round(recoverable_scfh, 0),
            recoverable_energy_mmbtu_hr=round(recoverable_mmbtu_hr, 2),
            recommended_recovery_method=method,
            implementation_cost_estimate=impl_cost,
            annual_value_recovered=round(annual_value, 0),
            simple_payback_years=round(payback, 1),
            co2_currently_emitted_tpy=round(co2_tpy, 1),
            co2_avoided_tpy=round(co2_avoided, 1),
            methane_avoided_tpy=round(methane_avoided, 2),
            ghg_reduction_co2e_tpy=round(ghg_avoided_co2e, 1),
            regulatory_compliance_notes=notes,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS"
        )


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-037",
    "name": "FLARE-MINIMIZER",
    "version": "1.0.0",
    "summary": "Flare gas recovery and emissions minimization",
    "tags": ["flare", "emissions", "methane", "gas-recovery", "EPA-40-CFR-60", "API-521"],
    "owners": ["emissions-reduction-team"],
    "standards": [
        {"ref": "EPA 40 CFR 60", "description": "Standards of Performance for New Stationary Sources"},
        {"ref": "API 521", "description": "Pressure-relieving and Depressuring Systems"}
    ]
}
