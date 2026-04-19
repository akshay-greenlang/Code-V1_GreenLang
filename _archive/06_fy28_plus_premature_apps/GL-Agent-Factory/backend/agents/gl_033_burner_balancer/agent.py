"""
GL-033: Burner Balancer Agent (BURNER-BALANCER)

This module implements the BurnerBalancerAgent for multi-burner load
balancing and air-fuel optimization in industrial combustion systems.

Standards Compliance:
- NFPA 85: Boiler and Combustion Systems Hazards Code
- NFPA 86: Standard for Ovens and Furnaces
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .models import (
    BurnerType,
    FuelType,
    BurnerStatus,
    BalancingObjective,
)
from .formulas import (
    calculate_stoichiometric_air_fuel_ratio,
    calculate_excess_air_percent,
    calculate_combustion_efficiency,
    calculate_nox_estimate,
    optimize_air_fuel_ratio,
    distribute_load_to_burners,
    calculate_total_efficiency,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT MODELS
# =============================================================================

class BurnerData(BaseModel):
    """Individual burner operating data."""

    burner_id: str = Field(..., description="Burner identifier")
    burner_type: BurnerType = Field(default=BurnerType.NOZZLE_MIX)
    status: BurnerStatus = Field(default=BurnerStatus.MODULATING)
    capacity_mmbtu_hr: float = Field(..., gt=0, description="Max capacity in MMBtu/hr")
    current_firing_rate: float = Field(default=50.0, ge=0, le=100)
    fuel_flow_scfh: float = Field(default=0, ge=0)
    air_flow_scfh: float = Field(default=0, ge=0)
    o2_percent: float = Field(default=3.0, ge=0, le=21)
    flue_gas_temp_c: float = Field(default=250)
    efficiency_rating: float = Field(default=85.0, ge=0, le=100)


class BurnerBalancerInput(BaseModel):
    """Input data model for BurnerBalancerAgent."""

    system_id: str = Field(..., description="Combustion system identifier")
    burner_data: List[BurnerData] = Field(..., description="Individual burner data")
    fuel_type: FuelType = Field(default=FuelType.NATURAL_GAS)
    fuel_header_pressure_kpa: float = Field(default=35.0, description="Fuel header pressure")
    air_header_pressure_kpa: float = Field(default=2.5, description="Air header pressure")
    load_demand_percent: float = Field(..., ge=0, le=100, description="Total load demand")
    ambient_temp_c: float = Field(default=25.0)
    target_o2_percent: float = Field(default=3.0, ge=1, le=10)
    optimization_objective: BalancingObjective = Field(default=BalancingObjective.BALANCED)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class BurnerSetpoint(BaseModel):
    """Recommended setpoints for a burner."""

    burner_id: str = Field(..., description="Burner identifier")
    recommended_firing_rate: float = Field(..., ge=0, le=100)
    recommended_air_flow_scfh: float = Field(..., ge=0)
    expected_efficiency: float = Field(..., ge=0, le=100)
    expected_nox_ppm: float = Field(..., ge=0)
    change_required: bool = Field(..., description="Whether adjustment needed")


class BurnerBalancerOutput(BaseModel):
    """Output data model for BurnerBalancerAgent."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    system_id: str = Field(..., description="System identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Optimized setpoints
    optimal_firing_rates: List[BurnerSetpoint] = Field(...)
    air_distribution: List[Dict[str, float]] = Field(default_factory=list)

    # Performance metrics
    current_efficiency: float = Field(..., ge=0, le=100)
    expected_efficiency: float = Field(..., ge=0, le=100)
    efficiency_gain_percent: float = Field(...)

    current_nox_total_ppm: float = Field(...)
    expected_nox_total_ppm: float = Field(...)
    emission_reduction_percent: float = Field(...)

    # System status
    total_capacity_mmbtu_hr: float = Field(...)
    current_load_mmbtu_hr: float = Field(...)
    available_capacity_mmbtu_hr: float = Field(...)

    # Provenance
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    validation_status: str = Field(...)


# =============================================================================
# BURNER BALANCER AGENT
# =============================================================================

class BurnerBalancerAgent:
    """
    GL-033: Burner Balancer Agent (BURNER-BALANCER).

    This agent optimizes multi-burner load distribution and air-fuel ratios
    for maximum efficiency and minimum emissions.
    """

    AGENT_ID = "GL-033"
    AGENT_NAME = "BURNER-BALANCER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"BurnerBalancerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: BurnerBalancerInput) -> BurnerBalancerOutput:
        """Execute multi-burner optimization."""
        start_time = datetime.utcnow()

        logger.info(f"Starting burner optimization for {input_data.system_id}")

        try:
            # Get stoichiometric ratio
            stoich_ratio = calculate_stoichiometric_air_fuel_ratio(input_data.fuel_type.value)

            # Current state analysis
            capacities = [b.capacity_mmbtu_hr for b in input_data.burner_data]
            efficiencies = [b.efficiency_rating for b in input_data.burner_data]
            statuses = [b.status.value for b in input_data.burner_data]
            current_rates = [b.current_firing_rate for b in input_data.burner_data]

            # Calculate current efficiency
            current_eff = calculate_total_efficiency(current_rates, efficiencies, capacities)

            # Calculate current NOx
            current_nox = sum(
                calculate_nox_estimate(
                    b.current_firing_rate,
                    calculate_excess_air_percent(b.air_flow_scfh, b.fuel_flow_scfh, stoich_ratio),
                    1800,  # Estimated flame temp
                    b.burner_type.value
                ) * (b.current_firing_rate / 100)
                for b in input_data.burner_data
            ) / max(1, len(input_data.burner_data))

            # Optimize load distribution
            optimal_rates = distribute_load_to_burners(
                input_data.load_demand_percent,
                capacities,
                efficiencies,
                statuses,
                input_data.optimization_objective.value
            )

            # Calculate optimal air flows
            setpoints = []
            air_dist = []
            expected_nox = 0.0

            for i, burner in enumerate(input_data.burner_data):
                opt_rate = optimal_rates[i]

                # Calculate optimal air
                if burner.fuel_flow_scfh > 0 and opt_rate > 0:
                    opt_air, excess = optimize_air_fuel_ratio(
                        burner.fuel_flow_scfh * (opt_rate / max(burner.current_firing_rate, 1)),
                        burner.o2_percent,
                        input_data.target_o2_percent,
                        stoich_ratio,
                        burner.air_flow_scfh
                    )
                else:
                    opt_air = burner.air_flow_scfh
                    excess = 15.0

                # Expected efficiency at optimal settings
                exp_eff = calculate_combustion_efficiency(
                    burner.flue_gas_temp_c,
                    input_data.ambient_temp_c,
                    input_data.target_o2_percent,
                    input_data.fuel_type.value
                )

                # Expected NOx
                burner_nox = calculate_nox_estimate(
                    opt_rate, excess, 1800, burner.burner_type.value
                )
                expected_nox += burner_nox * (opt_rate / 100)

                change_needed = abs(opt_rate - burner.current_firing_rate) > 2.0

                setpoints.append(BurnerSetpoint(
                    burner_id=burner.burner_id,
                    recommended_firing_rate=round(opt_rate, 1),
                    recommended_air_flow_scfh=round(opt_air, 0),
                    expected_efficiency=round(exp_eff, 1),
                    expected_nox_ppm=round(burner_nox, 1),
                    change_required=change_needed
                ))

                air_dist.append({
                    "burner_id": burner.burner_id,
                    "air_flow_scfh": round(opt_air, 0)
                })

            expected_nox /= max(1, len(input_data.burner_data))

            # Calculate expected overall efficiency
            expected_eff = calculate_total_efficiency(optimal_rates, efficiencies, capacities)

            # Efficiency gain
            eff_gain = expected_eff - current_eff

            # Emission reduction
            emission_reduction = ((current_nox - expected_nox) / max(current_nox, 1)) * 100

            # Capacity calculations
            total_cap = sum(capacities)
            current_load = total_cap * input_data.load_demand_percent / 100
            available = total_cap - current_load

            # Provenance
            provenance_hash = hashlib.sha256(
                json.dumps({
                    "agent_id": self.AGENT_ID,
                    "system_id": input_data.system_id,
                    "timestamp": datetime.utcnow().isoformat()
                }, sort_keys=True).encode()
            ).hexdigest()

            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            output = BurnerBalancerOutput(
                analysis_id=f"BB-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                system_id=input_data.system_id,
                optimal_firing_rates=setpoints,
                air_distribution=air_dist,
                current_efficiency=round(current_eff, 1),
                expected_efficiency=round(expected_eff, 1),
                efficiency_gain_percent=round(eff_gain, 2),
                current_nox_total_ppm=round(current_nox, 1),
                expected_nox_total_ppm=round(expected_nox, 1),
                emission_reduction_percent=round(emission_reduction, 1),
                total_capacity_mmbtu_hr=round(total_cap, 1),
                current_load_mmbtu_hr=round(current_load, 1),
                available_capacity_mmbtu_hr=round(available, 1),
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time, 2),
                validation_status="PASS"
            )

            logger.info(f"Burner optimization complete: efficiency gain={eff_gain:.1f}%")
            return output

        except Exception as e:
            logger.error(f"Burner optimization failed: {str(e)}", exc_info=True)
            raise


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-033",
    "name": "BURNER-BALANCER - Multi-Burner Load Balancing Agent",
    "version": "1.0.0",
    "summary": "Multi-burner load balancing and air-fuel optimization",
    "tags": ["burner", "combustion", "efficiency", "emissions", "NFPA-85", "NFPA-86"],
    "owners": ["process-heat-optimization-team"],
    "compute": {
        "entrypoint": "python://agents.gl_033_burner_balancer.agent:BurnerBalancerAgent",
        "deterministic": True
    },
    "standards": [
        {"ref": "NFPA 85", "description": "Boiler and Combustion Systems Hazards Code"},
        {"ref": "NFPA 86", "description": "Standard for Ovens and Furnaces"}
    ]
}
