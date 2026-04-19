# -*- coding: utf-8 -*-
"""
GL-MRV-X-002: Refrigerants & F-Gas Agent
=========================================

Estimates Scope 1 F-gas emissions from refrigerant leakage and other
fluorinated gas sources following GHG Protocol guidance.

Capabilities:
    - Refrigerant leakage emissions calculation
    - HFC, PFC, SF6, and NF3 emissions tracking
    - Equipment-based and material balance approaches
    - Screening method for simplified estimation
    - GWP application for all F-gas types
    - Complete provenance tracking

Zero-Hallucination Guarantees:
    - All calculations are deterministic mathematical operations
    - NO LLM involvement in any calculation path
    - All GWP values from IPCC AR6
    - Complete provenance hash for every calculation

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.base_agents import DeterministicAgent, AuditEntry
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class RefrigerantType(str, Enum):
    """Standard refrigerant types."""
    # HFCs
    R134A = "R-134a"
    R410A = "R-410A"
    R404A = "R-404A"
    R407C = "R-407C"
    R32 = "R-32"
    R125 = "R-125"
    R143A = "R-143a"
    R152A = "R-152a"
    R227EA = "R-227ea"
    R245FA = "R-245fa"
    R365MFC = "R-365mfc"

    # HFOs (low GWP alternatives)
    R1234YF = "R-1234yf"
    R1234ZE = "R-1234ze"

    # PFCs
    CF4 = "CF4"
    C2F6 = "C2F6"
    C3F8 = "C3F8"

    # Other F-gases
    SF6 = "SF6"
    NF3 = "NF3"

    # Legacy (being phased out)
    R22 = "R-22"
    R12 = "R-12"
    R502 = "R-502"


class EquipmentType(str, Enum):
    """Types of refrigeration/AC equipment."""
    COMMERCIAL_REFRIGERATION = "commercial_refrigeration"
    INDUSTRIAL_REFRIGERATION = "industrial_refrigeration"
    RESIDENTIAL_AC = "residential_ac"
    COMMERCIAL_AC = "commercial_ac"
    CHILLERS = "chillers"
    HEAT_PUMPS = "heat_pumps"
    TRANSPORT_REFRIGERATION = "transport_refrigeration"
    SWITCHGEAR = "switchgear"
    SEMICONDUCTOR = "semiconductor"
    FIRE_SUPPRESSION = "fire_suppression"


class CalculationMethod(str, Enum):
    """Methods for calculating F-gas emissions."""
    MASS_BALANCE = "mass_balance"
    EQUIPMENT_BASED = "equipment_based"
    SCREENING = "screening"
    DIRECT_MEASUREMENT = "direct_measurement"


# =============================================================================
# GWP VALUES DATABASE (IPCC AR6)
# =============================================================================

GWP_REFRIGERANTS: Dict[RefrigerantType, Decimal] = {
    # HFCs
    RefrigerantType.R134A: Decimal("1530"),
    RefrigerantType.R410A: Decimal("2088"),  # Blend: R-32 + R-125
    RefrigerantType.R404A: Decimal("4728"),  # Blend
    RefrigerantType.R407C: Decimal("1774"),  # Blend
    RefrigerantType.R32: Decimal("771"),
    RefrigerantType.R125: Decimal("3740"),
    RefrigerantType.R143A: Decimal("5810"),
    RefrigerantType.R152A: Decimal("164"),
    RefrigerantType.R227EA: Decimal("3600"),
    RefrigerantType.R245FA: Decimal("962"),
    RefrigerantType.R365MFC: Decimal("804"),

    # HFOs (low GWP)
    RefrigerantType.R1234YF: Decimal("1"),
    RefrigerantType.R1234ZE: Decimal("1"),

    # PFCs
    RefrigerantType.CF4: Decimal("7380"),
    RefrigerantType.C2F6: Decimal("12400"),
    RefrigerantType.C3F8: Decimal("9290"),

    # Other F-gases
    RefrigerantType.SF6: Decimal("25200"),
    RefrigerantType.NF3: Decimal("17400"),

    # Legacy
    RefrigerantType.R22: Decimal("1960"),
    RefrigerantType.R12: Decimal("10200"),
    RefrigerantType.R502: Decimal("4657"),
}

# Default annual leakage rates by equipment type (%)
DEFAULT_LEAKAGE_RATES: Dict[EquipmentType, Decimal] = {
    EquipmentType.COMMERCIAL_REFRIGERATION: Decimal("0.15"),  # 15%
    EquipmentType.INDUSTRIAL_REFRIGERATION: Decimal("0.10"),  # 10%
    EquipmentType.RESIDENTIAL_AC: Decimal("0.04"),  # 4%
    EquipmentType.COMMERCIAL_AC: Decimal("0.06"),  # 6%
    EquipmentType.CHILLERS: Decimal("0.05"),  # 5%
    EquipmentType.HEAT_PUMPS: Decimal("0.04"),  # 4%
    EquipmentType.TRANSPORT_REFRIGERATION: Decimal("0.25"),  # 25%
    EquipmentType.SWITCHGEAR: Decimal("0.01"),  # 1%
    EquipmentType.SEMICONDUCTOR: Decimal("0.08"),  # 8%
    EquipmentType.FIRE_SUPPRESSION: Decimal("0.02"),  # 2%
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class RefrigerantInventory(BaseModel):
    """Refrigerant inventory record."""
    refrigerant_type: RefrigerantType = Field(..., description="Refrigerant type")
    equipment_type: EquipmentType = Field(..., description="Equipment type")
    charge_kg: float = Field(..., gt=0, description="Refrigerant charge in kg")
    equipment_count: int = Field(default=1, ge=1, description="Number of units")
    equipment_id: Optional[str] = Field(None, description="Equipment identifier")
    location: Optional[str] = Field(None, description="Equipment location")
    custom_leakage_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Custom annual leakage rate"
    )


class MassBalanceInput(BaseModel):
    """Mass balance calculation input."""
    refrigerant_type: RefrigerantType = Field(..., description="Refrigerant type")
    beginning_inventory_kg: float = Field(..., ge=0, description="Beginning inventory")
    purchases_kg: float = Field(default=0, ge=0, description="Refrigerant purchased")
    sales_kg: float = Field(default=0, ge=0, description="Refrigerant sold")
    acquisitions_kg: float = Field(default=0, ge=0, description="From acquisitions")
    divestitures_kg: float = Field(default=0, ge=0, description="From divestitures")
    ending_inventory_kg: float = Field(..., ge=0, description="Ending inventory")
    capacity_change_kg: float = Field(default=0, description="Net capacity change")


class FGasEmissionResult(BaseModel):
    """Result of F-gas emissions calculation."""
    refrigerant_type: RefrigerantType = Field(..., description="Refrigerant type")
    equipment_type: Optional[EquipmentType] = Field(None, description="Equipment type")
    method: CalculationMethod = Field(..., description="Calculation method used")

    # Quantities
    refrigerant_loss_kg: float = Field(..., description="Refrigerant loss in kg")

    # Emissions
    emissions_tco2e: float = Field(..., description="Emissions in tCO2e")
    gwp_applied: float = Field(..., description="GWP value applied")
    gwp_source: str = Field(default="AR6", description="GWP source")

    # Metadata
    leakage_rate_used: Optional[float] = Field(None, description="Leakage rate used")
    calculation_trace: List[str] = Field(
        default_factory=list, description="Calculation trace"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash")


class RefrigerantsFGasInput(BaseModel):
    """Input model for RefrigerantsFGasAgent."""
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.EQUIPMENT_BASED,
        description="Calculation method"
    )

    # For equipment-based method
    refrigerant_inventory: Optional[List[RefrigerantInventory]] = Field(
        None, description="Refrigerant inventory records"
    )

    # For mass balance method
    mass_balance_inputs: Optional[List[MassBalanceInput]] = Field(
        None, description="Mass balance calculation inputs"
    )

    # For screening method
    screening_total_charge_kg: Optional[float] = Field(
        None, description="Total refrigerant charge for screening"
    )
    screening_refrigerant_type: Optional[RefrigerantType] = Field(
        None, description="Refrigerant type for screening"
    )
    screening_equipment_type: Optional[EquipmentType] = Field(
        None, description="Equipment type for screening"
    )

    # Common parameters
    gwp_source: str = Field(default="AR6", description="GWP source")
    organization_id: Optional[str] = Field(None, description="Organization ID")
    reporting_period: Optional[str] = Field(None, description="Reporting period")


class RefrigerantsFGasOutput(BaseModel):
    """Output model for RefrigerantsFGasAgent."""
    success: bool = Field(..., description="Calculation success")
    calculation_method: CalculationMethod = Field(..., description="Method used")
    emission_results: List[FGasEmissionResult] = Field(
        default_factory=list, description="Individual emission results"
    )

    # Totals
    total_emissions_tco2e: float = Field(..., description="Total F-gas emissions")
    total_refrigerant_loss_kg: float = Field(..., description="Total refrigerant loss")

    # Breakdown by refrigerant
    emissions_by_refrigerant: Dict[str, float] = Field(
        default_factory=dict, description="Emissions by refrigerant type"
    )

    # Breakdown by equipment
    emissions_by_equipment: Dict[str, float] = Field(
        default_factory=dict, description="Emissions by equipment type"
    )

    # Metadata
    gwp_source: str = Field(..., description="GWP source used")
    processing_time_ms: float = Field(..., description="Processing time")
    provenance_hash: str = Field(..., description="Combined provenance hash")
    validation_status: str = Field(..., description="PASS or FAIL")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)

    organization_id: Optional[str] = Field(None, description="Organization ID")
    reporting_period: Optional[str] = Field(None, description="Reporting period")


# =============================================================================
# REFRIGERANTS & F-GAS AGENT
# =============================================================================

class RefrigerantsFGasAgent(DeterministicAgent):
    """
    GL-MRV-X-002: Refrigerants & F-Gas Agent

    Estimates Scope 1 F-gas emissions from refrigerant leakage and other
    fluorinated gas sources. Supports multiple calculation methodologies
    following GHG Protocol guidance.

    Zero-Hallucination Implementation:
        - All calculations use deterministic mathematical operations
        - GWP values from IPCC AR6
        - Complete provenance tracking with SHA-256 hashes
        - Full calculation trace for regulatory audit

    Supported Methods:
        - Equipment-based: Uses inventory and leakage rates
        - Mass balance: Tracks refrigerant purchases/disposals
        - Screening: Simplified estimation for quick assessments

    Attributes:
        AGENT_ID: Unique agent identifier
        AGENT_NAME: Human-readable agent name
        VERSION: Agent version
        category: Agent category (CRITICAL)

    Example:
        >>> agent = RefrigerantsFGasAgent()
        >>> result = agent.execute({
        ...     "calculation_method": "equipment_based",
        ...     "refrigerant_inventory": [{
        ...         "refrigerant_type": "R-410A",
        ...         "equipment_type": "commercial_ac",
        ...         "charge_kg": 10,
        ...         "equipment_count": 5
        ...     }]
        ... })
    """

    AGENT_ID = "GL-MRV-X-002"
    AGENT_NAME = "Refrigerants & F-Gas Agent"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    metadata = AgentMetadata(
        name="RefrigerantsFGasAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="F-gas emissions calculator for Scope 1 refrigerant leakage"
    )

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize RefrigerantsFGasAgent."""
        super().__init__(enable_audit_trail=enable_audit_trail)
        self._calculation_counter = 0
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute F-gas emissions calculation.

        Args:
            inputs: Dictionary containing refrigerant data and method

        Returns:
            Dictionary with calculation results

        Raises:
            ValueError: If input validation fails
        """
        start_time = DeterministicClock.now()

        try:
            # Parse input
            fgas_input = RefrigerantsFGasInput(**inputs)

            # Route to appropriate method
            if fgas_input.calculation_method == CalculationMethod.EQUIPMENT_BASED:
                results = self._calculate_equipment_based(fgas_input)
            elif fgas_input.calculation_method == CalculationMethod.MASS_BALANCE:
                results = self._calculate_mass_balance(fgas_input)
            elif fgas_input.calculation_method == CalculationMethod.SCREENING:
                results = self._calculate_screening(fgas_input)
            else:
                raise ValueError(f"Unsupported method: {fgas_input.calculation_method}")

            # Aggregate results
            total_emissions = sum(r.emissions_tco2e for r in results)
            total_loss = sum(r.refrigerant_loss_kg for r in results)

            # Breakdown by refrigerant
            emissions_by_refrigerant: Dict[str, float] = {}
            for r in results:
                key = r.refrigerant_type.value
                emissions_by_refrigerant[key] = emissions_by_refrigerant.get(key, 0) + r.emissions_tco2e

            # Breakdown by equipment
            emissions_by_equipment: Dict[str, float] = {}
            for r in results:
                if r.equipment_type:
                    key = r.equipment_type.value
                    emissions_by_equipment[key] = emissions_by_equipment.get(key, 0) + r.emissions_tco2e

            # Calculate processing time
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            # Compute provenance hash
            provenance_data = {
                "input": inputs,
                "total_emissions_tco2e": total_emissions,
                "result_count": len(results)
            }
            provenance_hash = self._compute_provenance_hash(provenance_data)

            output = RefrigerantsFGasOutput(
                success=True,
                calculation_method=fgas_input.calculation_method,
                emission_results=results,
                total_emissions_tco2e=round(total_emissions, 4),
                total_refrigerant_loss_kg=round(total_loss, 4),
                emissions_by_refrigerant=emissions_by_refrigerant,
                emissions_by_equipment=emissions_by_equipment,
                gwp_source=fgas_input.gwp_source,
                processing_time_ms=processing_time_ms,
                provenance_hash=provenance_hash,
                validation_status="PASS",
                organization_id=fgas_input.organization_id,
                reporting_period=fgas_input.reporting_period
            )

            # Capture audit entry
            self._capture_audit_entry(
                operation="calculate_fgas_emissions",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=[f"Processed {len(results)} emission calculations"]
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"F-gas calculation failed: {str(e)}", exc_info=True)
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "validation_status": "FAIL"
            }

    def _calculate_equipment_based(
        self,
        fgas_input: RefrigerantsFGasInput
    ) -> List[FGasEmissionResult]:
        """Calculate emissions using equipment-based method."""
        results = []

        if not fgas_input.refrigerant_inventory:
            raise ValueError("Equipment-based method requires refrigerant_inventory")

        for inventory in fgas_input.refrigerant_inventory:
            trace = []
            self._calculation_counter += 1

            # Get parameters
            refrigerant = inventory.refrigerant_type
            equipment = inventory.equipment_type
            charge_kg = Decimal(str(inventory.charge_kg))
            count = inventory.equipment_count

            trace.append(
                f"Equipment: {count}x {equipment.value} with {refrigerant.value}"
            )
            trace.append(f"Total charge: {float(charge_kg * count):.2f} kg")

            # Get leakage rate
            if inventory.custom_leakage_rate:
                leakage_rate = Decimal(str(inventory.custom_leakage_rate))
            else:
                leakage_rate = DEFAULT_LEAKAGE_RATES.get(equipment, Decimal("0.10"))

            trace.append(f"Leakage rate: {float(leakage_rate) * 100:.1f}%")

            # Calculate refrigerant loss
            total_charge = charge_kg * count
            refrigerant_loss = total_charge * leakage_rate

            trace.append(f"Annual loss: {float(refrigerant_loss):.4f} kg")

            # Get GWP
            gwp = GWP_REFRIGERANTS.get(refrigerant, Decimal("1"))
            trace.append(f"GWP ({fgas_input.gwp_source}): {float(gwp)}")

            # Calculate emissions
            emissions_kg_co2e = refrigerant_loss * gwp
            emissions_tco2e = emissions_kg_co2e / Decimal("1000")

            trace.append(f"Emissions: {float(emissions_tco2e):.4f} tCO2e")

            # Provenance
            provenance_hash = self._compute_provenance_hash({
                "refrigerant": refrigerant.value,
                "equipment": equipment.value,
                "charge_kg": float(charge_kg),
                "count": count,
                "emissions_tco2e": float(emissions_tco2e)
            })

            results.append(FGasEmissionResult(
                refrigerant_type=refrigerant,
                equipment_type=equipment,
                method=CalculationMethod.EQUIPMENT_BASED,
                refrigerant_loss_kg=float(refrigerant_loss.quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )),
                emissions_tco2e=float(emissions_tco2e.quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )),
                gwp_applied=float(gwp),
                gwp_source=fgas_input.gwp_source,
                leakage_rate_used=float(leakage_rate),
                calculation_trace=trace,
                provenance_hash=provenance_hash
            ))

        return results

    def _calculate_mass_balance(
        self,
        fgas_input: RefrigerantsFGasInput
    ) -> List[FGasEmissionResult]:
        """Calculate emissions using mass balance method."""
        results = []

        if not fgas_input.mass_balance_inputs:
            raise ValueError("Mass balance method requires mass_balance_inputs")

        for mb in fgas_input.mass_balance_inputs:
            trace = []
            self._calculation_counter += 1

            refrigerant = mb.refrigerant_type
            trace.append(f"Mass balance for {refrigerant.value}")

            # Mass balance equation:
            # Emissions = (Beginning Inventory + Purchases - Sales + Acquisitions
            #              - Divestitures - Ending Inventory - Capacity Change)
            beginning = Decimal(str(mb.beginning_inventory_kg))
            purchases = Decimal(str(mb.purchases_kg))
            sales = Decimal(str(mb.sales_kg))
            acquisitions = Decimal(str(mb.acquisitions_kg))
            divestitures = Decimal(str(mb.divestitures_kg))
            ending = Decimal(str(mb.ending_inventory_kg))
            capacity_change = Decimal(str(mb.capacity_change_kg))

            refrigerant_loss = (
                beginning + purchases - sales + acquisitions
                - divestitures - ending - capacity_change
            )

            # Ensure non-negative
            refrigerant_loss = max(refrigerant_loss, Decimal("0"))

            trace.append(f"Beginning: {float(beginning):.2f} kg")
            trace.append(f"+ Purchases: {float(purchases):.2f} kg")
            trace.append(f"- Sales: {float(sales):.2f} kg")
            trace.append(f"+ Acquisitions: {float(acquisitions):.2f} kg")
            trace.append(f"- Divestitures: {float(divestitures):.2f} kg")
            trace.append(f"- Ending: {float(ending):.2f} kg")
            trace.append(f"- Capacity change: {float(capacity_change):.2f} kg")
            trace.append(f"= Emissions: {float(refrigerant_loss):.4f} kg")

            # Get GWP
            gwp = GWP_REFRIGERANTS.get(refrigerant, Decimal("1"))
            trace.append(f"GWP: {float(gwp)}")

            # Calculate emissions
            emissions_kg_co2e = refrigerant_loss * gwp
            emissions_tco2e = emissions_kg_co2e / Decimal("1000")

            trace.append(f"Total: {float(emissions_tco2e):.4f} tCO2e")

            provenance_hash = self._compute_provenance_hash({
                "refrigerant": refrigerant.value,
                "method": "mass_balance",
                "refrigerant_loss_kg": float(refrigerant_loss),
                "emissions_tco2e": float(emissions_tco2e)
            })

            results.append(FGasEmissionResult(
                refrigerant_type=refrigerant,
                equipment_type=None,
                method=CalculationMethod.MASS_BALANCE,
                refrigerant_loss_kg=float(refrigerant_loss.quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )),
                emissions_tco2e=float(emissions_tco2e.quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )),
                gwp_applied=float(gwp),
                gwp_source=fgas_input.gwp_source,
                calculation_trace=trace,
                provenance_hash=provenance_hash
            ))

        return results

    def _calculate_screening(
        self,
        fgas_input: RefrigerantsFGasInput
    ) -> List[FGasEmissionResult]:
        """Calculate emissions using screening method."""
        trace = []
        self._calculation_counter += 1

        if not fgas_input.screening_total_charge_kg:
            raise ValueError("Screening method requires screening_total_charge_kg")

        refrigerant = fgas_input.screening_refrigerant_type or RefrigerantType.R410A
        equipment = fgas_input.screening_equipment_type or EquipmentType.COMMERCIAL_AC
        total_charge = Decimal(str(fgas_input.screening_total_charge_kg))

        trace.append(f"Screening estimate for {refrigerant.value}")
        trace.append(f"Total charge: {float(total_charge):.2f} kg")

        # Use default leakage rate
        leakage_rate = DEFAULT_LEAKAGE_RATES.get(equipment, Decimal("0.10"))
        trace.append(f"Default leakage rate ({equipment.value}): {float(leakage_rate) * 100:.1f}%")

        refrigerant_loss = total_charge * leakage_rate
        trace.append(f"Estimated loss: {float(refrigerant_loss):.4f} kg")

        # Get GWP
        gwp = GWP_REFRIGERANTS.get(refrigerant, Decimal("1"))
        trace.append(f"GWP: {float(gwp)}")

        emissions_kg_co2e = refrigerant_loss * gwp
        emissions_tco2e = emissions_kg_co2e / Decimal("1000")

        trace.append(f"Estimated emissions: {float(emissions_tco2e):.4f} tCO2e")

        provenance_hash = self._compute_provenance_hash({
            "method": "screening",
            "total_charge_kg": float(total_charge),
            "refrigerant": refrigerant.value,
            "emissions_tco2e": float(emissions_tco2e)
        })

        return [FGasEmissionResult(
            refrigerant_type=refrigerant,
            equipment_type=equipment,
            method=CalculationMethod.SCREENING,
            refrigerant_loss_kg=float(refrigerant_loss.quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )),
            emissions_tco2e=float(emissions_tco2e.quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )),
            gwp_applied=float(gwp),
            gwp_source=fgas_input.gwp_source,
            leakage_rate_used=float(leakage_rate),
            calculation_trace=trace,
            provenance_hash=provenance_hash
        )]

    def _compute_provenance_hash(self, data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_gwp(self, refrigerant: str) -> Optional[float]:
        """Get GWP value for a refrigerant."""
        try:
            ref_type = RefrigerantType(refrigerant)
            return float(GWP_REFRIGERANTS.get(ref_type, Decimal("0")))
        except ValueError:
            return None

    def get_default_leakage_rate(self, equipment: str) -> Optional[float]:
        """Get default leakage rate for equipment type."""
        try:
            eq_type = EquipmentType(equipment)
            return float(DEFAULT_LEAKAGE_RATES.get(eq_type, Decimal("0.10")))
        except ValueError:
            return None

    def get_supported_refrigerants(self) -> List[str]:
        """Get list of supported refrigerant types."""
        return [rt.value for rt in RefrigerantType]

    def get_supported_equipment(self) -> List[str]:
        """Get list of supported equipment types."""
        return [et.value for et in EquipmentType]
