# -*- coding: utf-8 -*-
"""
GL-MRV-X-015: Process Emissions Agent
======================================

Calculates non-combustion process emissions from industrial processes
including cement, chemicals, metals, and other manufacturing.

Capabilities:
    - Cement production emissions
    - Chemical process emissions
    - Metal production emissions
    - Electronics/semiconductor emissions
    - Adipic acid, nitric acid emissions
    - Complete provenance tracking

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


class ProcessType(str, Enum):
    """Types of industrial processes."""
    CEMENT_PRODUCTION = "cement_production"
    LIME_PRODUCTION = "lime_production"
    GLASS_PRODUCTION = "glass_production"
    AMMONIA_PRODUCTION = "ammonia_production"
    NITRIC_ACID = "nitric_acid"
    ADIPIC_ACID = "adipic_acid"
    ALUMINUM_SMELTING = "aluminum_smelting"
    IRON_STEEL = "iron_steel"
    SEMICONDUCTOR = "semiconductor"
    MAGNESIUM_PRODUCTION = "magnesium_production"
    HCFC_22_PRODUCTION = "hcfc22_production"


# Process emission factors (various units)
PROCESS_EMISSION_FACTORS: Dict[ProcessType, Dict[str, Any]] = {
    ProcessType.CEMENT_PRODUCTION: {
        "co2_factor": Decimal("0.507"),  # tCO2/tonne clinker
        "unit": "tonne_clinker",
        "description": "Calcination of limestone"
    },
    ProcessType.LIME_PRODUCTION: {
        "co2_factor": Decimal("0.785"),  # tCO2/tonne lime
        "unit": "tonne_lime",
        "description": "Calcination of limestone to quicklime"
    },
    ProcessType.GLASS_PRODUCTION: {
        "co2_factor": Decimal("0.208"),  # tCO2/tonne glass
        "unit": "tonne_glass",
        "description": "Decomposition of carbonates in batch"
    },
    ProcessType.AMMONIA_PRODUCTION: {
        "co2_factor": Decimal("1.5"),  # tCO2/tonne ammonia (natural gas based)
        "unit": "tonne_ammonia",
        "description": "Steam methane reforming"
    },
    ProcessType.NITRIC_ACID: {
        "n2o_factor": Decimal("0.007"),  # tN2O/tonne nitric acid
        "unit": "tonne_nitric_acid",
        "description": "Oxidation of ammonia"
    },
    ProcessType.ADIPIC_ACID: {
        "n2o_factor": Decimal("0.3"),  # tN2O/tonne adipic acid (unabated)
        "unit": "tonne_adipic_acid",
        "description": "Oxidation of cyclohexanol"
    },
    ProcessType.ALUMINUM_SMELTING: {
        "co2_factor": Decimal("1.5"),  # tCO2/tonne aluminum
        "pfc_factor": Decimal("0.5"),  # tCO2e/tonne aluminum (anode effects)
        "unit": "tonne_aluminum",
        "description": "Electrolytic reduction"
    },
    ProcessType.IRON_STEEL: {
        "co2_factor": Decimal("1.9"),  # tCO2/tonne steel (BF-BOF route)
        "unit": "tonne_steel",
        "description": "Blast furnace reduction"
    },
    ProcessType.SEMICONDUCTOR: {
        "pfc_factor": Decimal("0.01"),  # tCO2e per wafer (varies widely)
        "unit": "wafer",
        "description": "Plasma etching and CVD"
    },
}

# GWP for N2O
N2O_GWP_AR6 = Decimal("273")


class ProcessActivity(BaseModel):
    """A process activity record."""
    activity_id: str = Field(...)
    process_type: ProcessType = Field(...)
    quantity: float = Field(..., gt=0)
    unit: str = Field(...)
    facility_id: Optional[str] = Field(None)
    period: Optional[str] = Field(None)
    custom_emission_factor: Optional[float] = Field(None)
    abatement_efficiency: float = Field(default=0.0, ge=0, le=1)


class ProcessEmissionResult(BaseModel):
    """Result of process emissions calculation."""
    activity_id: str = Field(...)
    process_type: ProcessType = Field(...)
    quantity: float = Field(...)
    co2_emissions_tonnes: float = Field(default=0)
    n2o_emissions_tonnes: float = Field(default=0)
    pfc_emissions_tco2e: float = Field(default=0)
    total_tco2e: float = Field(...)
    emission_factor_used: Dict[str, float] = Field(default_factory=dict)
    abatement_applied: float = Field(default=0)
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)


class ProcessEmissionsInput(BaseModel):
    """Input model for ProcessEmissionsAgent."""
    activities: List[ProcessActivity] = Field(..., min_length=1)
    gwp_source: str = Field(default="AR6")
    organization_id: Optional[str] = Field(None)
    reporting_period: Optional[str] = Field(None)


class ProcessEmissionsOutput(BaseModel):
    """Output model for ProcessEmissionsAgent."""
    success: bool = Field(...)
    emission_results: List[ProcessEmissionResult] = Field(default_factory=list)
    total_co2_tonnes: float = Field(...)
    total_n2o_tonnes: float = Field(...)
    total_pfc_tco2e: float = Field(...)
    total_tco2e: float = Field(...)
    emissions_by_process: Dict[str, float] = Field(default_factory=dict)
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field(...)
    validation_status: str = Field(...)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


class ProcessEmissionsAgent(DeterministicAgent):
    """
    GL-MRV-X-015: Process Emissions Agent

    Calculates non-combustion process emissions from industrial processes.

    Example:
        >>> agent = ProcessEmissionsAgent()
        >>> result = agent.execute({
        ...     "activities": [
        ...         {"activity_id": "P001", "process_type": "cement_production",
        ...          "quantity": 1000000, "unit": "tonne_clinker"}
        ...     ]
        ... })
    """

    AGENT_ID = "GL-MRV-X-015"
    AGENT_NAME = "Process Emissions Agent"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    metadata = AgentMetadata(
        name="ProcessEmissionsAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Calculates non-combustion process emissions"
    )

    def __init__(self, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute process emissions calculation."""
        start_time = DeterministicClock.now()

        try:
            proc_input = ProcessEmissionsInput(**inputs)
            emission_results: List[ProcessEmissionResult] = []
            emissions_by_process: Dict[str, float] = {}

            total_co2 = Decimal("0")
            total_n2o = Decimal("0")
            total_pfc = Decimal("0")

            for activity in proc_input.activities:
                result = self._calculate_process_emissions(activity)
                emission_results.append(result)

                total_co2 += Decimal(str(result.co2_emissions_tonnes))
                total_n2o += Decimal(str(result.n2o_emissions_tonnes))
                total_pfc += Decimal(str(result.pfc_emissions_tco2e))

                process_key = activity.process_type.value
                emissions_by_process[process_key] = (
                    emissions_by_process.get(process_key, 0) + result.total_tco2e
                )

            # Calculate total CO2e
            n2o_co2e = total_n2o * N2O_GWP_AR6
            total_tco2e = total_co2 + n2o_co2e + total_pfc

            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            provenance_hash = self._compute_hash({
                "total_tco2e": float(total_tco2e),
                "activities_processed": len(emission_results)
            })

            output = ProcessEmissionsOutput(
                success=True,
                emission_results=emission_results,
                total_co2_tonnes=float(total_co2.quantize(Decimal("0.001"))),
                total_n2o_tonnes=float(total_n2o.quantize(Decimal("0.000001"))),
                total_pfc_tco2e=float(total_pfc.quantize(Decimal("0.001"))),
                total_tco2e=float(total_tco2e.quantize(Decimal("0.001"))),
                emissions_by_process=emissions_by_process,
                processing_time_ms=processing_time_ms,
                provenance_hash=provenance_hash,
                validation_status="PASS"
            )

            self._capture_audit_entry(
                operation="calculate_process_emissions",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=[f"Calculated {len(emission_results)} process activities"]
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Process emissions calculation failed: {str(e)}", exc_info=True)
            end_time = DeterministicClock.now()
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "validation_status": "FAIL"
            }

    def _calculate_process_emissions(self, activity: ProcessActivity) -> ProcessEmissionResult:
        """Calculate emissions for a single process activity."""
        trace = []
        co2_tonnes = Decimal("0")
        n2o_tonnes = Decimal("0")
        pfc_tco2e = Decimal("0")
        factors_used = {}

        trace.append(f"Process: {activity.process_type.value}")
        trace.append(f"Quantity: {activity.quantity} {activity.unit}")

        # Get emission factors
        factors = PROCESS_EMISSION_FACTORS.get(activity.process_type, {})
        quantity = Decimal(str(activity.quantity))

        # Apply abatement
        abatement = Decimal(str(activity.abatement_efficiency))
        effective_factor = Decimal("1") - abatement

        # Calculate CO2 emissions
        if "co2_factor" in factors:
            co2_ef = factors["co2_factor"]
            if activity.custom_emission_factor:
                co2_ef = Decimal(str(activity.custom_emission_factor))
            co2_tonnes = quantity * co2_ef * effective_factor
            factors_used["co2"] = float(co2_ef)
            trace.append(f"CO2: {float(quantity)} * {float(co2_ef)} * (1-{float(abatement)}) = {float(co2_tonnes):.2f} t")

        # Calculate N2O emissions
        if "n2o_factor" in factors:
            n2o_ef = factors["n2o_factor"]
            n2o_tonnes = quantity * n2o_ef * effective_factor
            factors_used["n2o"] = float(n2o_ef)
            trace.append(f"N2O: {float(quantity)} * {float(n2o_ef)} * (1-{float(abatement)}) = {float(n2o_tonnes):.6f} t")

        # Calculate PFC emissions
        if "pfc_factor" in factors:
            pfc_ef = factors["pfc_factor"]
            pfc_tco2e = quantity * pfc_ef * effective_factor
            factors_used["pfc"] = float(pfc_ef)
            trace.append(f"PFC: {float(pfc_tco2e):.4f} tCO2e")

        # Total CO2e
        n2o_co2e = n2o_tonnes * N2O_GWP_AR6
        total_tco2e = co2_tonnes + n2o_co2e + pfc_tco2e
        trace.append(f"Total: {float(total_tco2e):.4f} tCO2e")

        provenance_hash = self._compute_hash({
            "activity_id": activity.activity_id,
            "total_tco2e": float(total_tco2e)
        })

        return ProcessEmissionResult(
            activity_id=activity.activity_id,
            process_type=activity.process_type,
            quantity=activity.quantity,
            co2_emissions_tonnes=float(co2_tonnes.quantize(Decimal("0.001"))),
            n2o_emissions_tonnes=float(n2o_tonnes.quantize(Decimal("0.000001"))),
            pfc_emissions_tco2e=float(pfc_tco2e.quantize(Decimal("0.001"))),
            total_tco2e=float(total_tco2e.quantize(Decimal("0.001"))),
            emission_factor_used=factors_used,
            abatement_applied=activity.abatement_efficiency,
            calculation_trace=trace,
            provenance_hash=provenance_hash
        )

    def _compute_hash(self, data: Any) -> str:
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_supported_processes(self) -> List[str]:
        """Get list of supported process types."""
        return [pt.value for pt in ProcessType]
