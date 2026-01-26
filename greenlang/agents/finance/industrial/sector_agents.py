# -*- coding: utf-8 -*-
"""
GreenLang Industrial Finance Sector Agents
===========================================

Finance agents for industrial sector climate investments:
    - GL-FIN-IND-001 to IND-012

Features:
    - Carbon cost projections
    - Green financing eligibility
    - Investment case modeling
    - ROI/NPV analysis

Author: GreenLang Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FinancingType(str, Enum):
    """Green financing types."""
    GREEN_BOND = "GREEN_BOND"
    SUSTAINABILITY_LINKED_LOAN = "SUSTAINABILITY_LINKED_LOAN"
    TRANSITION_FINANCE = "TRANSITION_FINANCE"
    CARBON_CREDIT_PREPURCHASE = "CARBON_CREDIT_PREPURCHASE"
    EQUITY = "EQUITY"


class FinanceInput(BaseModel):
    """Input for industrial finance agents."""
    facility_id: str
    sector: str
    project_name: str = Field(default="")

    # Investment details
    capex_usd: Decimal = Field(gt=0)
    project_lifetime_years: int = Field(default=20, ge=1, le=50)

    # Emissions impact
    annual_emission_reduction_tco2e: Decimal = Field(ge=0)

    # Carbon pricing assumptions
    carbon_price_usd_per_tco2: Decimal = Field(default=Decimal("50"))
    carbon_price_escalation_pct: Decimal = Field(default=Decimal("3"))

    # Financial assumptions
    discount_rate_pct: Decimal = Field(default=Decimal("8"))
    opex_savings_usd_annual: Decimal = Field(default=Decimal("0"))

    class Config:
        json_encoders = {Decimal: str}


class FinanceOutput(BaseModel):
    """Output from industrial finance agents."""
    calculation_id: str
    agent_id: str
    timestamp: str

    # Input summary
    facility_id: str
    sector: str
    capex_usd: Decimal

    # Financial metrics
    npv_usd: Decimal = Field(default=Decimal("0"))
    irr_pct: Optional[Decimal] = None
    payback_years: Optional[Decimal] = None
    levelized_abatement_cost_usd_per_tco2: Decimal = Field(default=Decimal("0"))

    # Carbon value
    total_carbon_savings_tco2e: Decimal = Field(default=Decimal("0"))
    total_carbon_value_usd: Decimal = Field(default=Decimal("0"))

    # Financing recommendations
    eligible_financing_types: List[FinancingType] = Field(default_factory=list)
    green_bond_eligible: bool = Field(default=False)
    eu_taxonomy_aligned: bool = Field(default=False)

    # Provenance
    provenance_hash: str = Field(default="")
    is_valid: bool = Field(default=True)

    class Config:
        json_encoders = {Decimal: str}


class IndustrialFinanceBaseAgent(ABC):
    """Base class for industrial finance agents."""

    AGENT_ID: str = "GL-FIN-IND-BASE"
    AGENT_VERSION: str = "1.0.0"
    SECTOR: str = "Industrial"

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def analyze(self, input_data: FinanceInput) -> FinanceOutput:
        """Analyze financial metrics for industrial investment."""
        pass

    def process(self, input_data: FinanceInput) -> FinanceOutput:
        """Main processing method."""
        try:
            self.logger.info(f"{self.AGENT_ID} analyzing: {input_data.facility_id}")
            return self.analyze(input_data)
        except Exception as e:
            self.logger.error(f"{self.AGENT_ID} failed: {str(e)}", exc_info=True)
            raise

    def _calculate_npv(
        self,
        capex: Decimal,
        annual_savings: Decimal,
        carbon_savings_tco2e: Decimal,
        carbon_price: Decimal,
        escalation_pct: Decimal,
        discount_rate_pct: Decimal,
        years: int
    ) -> Decimal:
        """Calculate Net Present Value."""
        npv = -capex
        for year in range(1, years + 1):
            carbon_price_year = carbon_price * (
                (Decimal("1") + escalation_pct / Decimal("100")) ** year
            )
            annual_carbon_value = carbon_savings_tco2e * carbon_price_year
            annual_cashflow = annual_savings + annual_carbon_value
            discount_factor = (Decimal("1") + discount_rate_pct / Decimal("100")) ** year
            npv += annual_cashflow / discount_factor
        return npv.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_payback(
        self,
        capex: Decimal,
        annual_savings: Decimal,
        carbon_savings_tco2e: Decimal,
        carbon_price: Decimal
    ) -> Optional[Decimal]:
        """Calculate simple payback period."""
        annual_value = annual_savings + (carbon_savings_tco2e * carbon_price)
        if annual_value <= 0:
            return None
        return (capex / annual_value).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

    def _get_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        import json
        def convert(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, Enum):
                return obj.value
            return obj
        converted = convert(data)
        json_str = json.dumps(converted, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# SECTOR-SPECIFIC FINANCE AGENTS
# =============================================================================

class SteelFinanceAgent(IndustrialFinanceBaseAgent):
    """GL-FIN-IND-001: Steel Finance Agent"""
    AGENT_ID = "GL-FIN-IND-001"
    SECTOR = "Steel"

    def analyze(self, input_data: FinanceInput) -> FinanceOutput:
        calc_id = hashlib.sha256(f"{self.AGENT_ID}:{input_data.facility_id}:{self._get_timestamp()}".encode()).hexdigest()[:16]

        total_carbon = input_data.annual_emission_reduction_tco2e * input_data.project_lifetime_years
        total_carbon_value = total_carbon * input_data.carbon_price_usd_per_tco2

        npv = self._calculate_npv(
            input_data.capex_usd, input_data.opex_savings_usd_annual,
            input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2,
            input_data.carbon_price_escalation_pct, input_data.discount_rate_pct,
            input_data.project_lifetime_years
        )
        payback = self._calculate_payback(
            input_data.capex_usd, input_data.opex_savings_usd_annual,
            input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2
        )
        lac = (input_data.capex_usd / total_carbon).quantize(Decimal("0.01")) if total_carbon > 0 else Decimal("0")

        # Steel decarbonization is typically green bond eligible
        eligible_types = [FinancingType.GREEN_BOND, FinancingType.TRANSITION_FINANCE]
        if input_data.annual_emission_reduction_tco2e > 10000:
            eligible_types.append(FinancingType.SUSTAINABILITY_LINKED_LOAN)

        return FinanceOutput(
            calculation_id=calc_id, agent_id=self.AGENT_ID, timestamp=self._get_timestamp(),
            facility_id=input_data.facility_id, sector=self.SECTOR, capex_usd=input_data.capex_usd,
            npv_usd=npv, payback_years=payback, levelized_abatement_cost_usd_per_tco2=lac,
            total_carbon_savings_tco2e=total_carbon, total_carbon_value_usd=total_carbon_value,
            eligible_financing_types=eligible_types, green_bond_eligible=True, eu_taxonomy_aligned=True,
            provenance_hash=self._calculate_hash({"input": input_data.model_dump(), "npv": str(npv)}),
            is_valid=True
        )


class CementFinanceAgent(IndustrialFinanceBaseAgent):
    """GL-FIN-IND-002: Cement Finance Agent"""
    AGENT_ID = "GL-FIN-IND-002"
    SECTOR = "Cement"

    def analyze(self, input_data: FinanceInput) -> FinanceOutput:
        calc_id = hashlib.sha256(f"{self.AGENT_ID}:{input_data.facility_id}".encode()).hexdigest()[:16]
        total_carbon = input_data.annual_emission_reduction_tco2e * input_data.project_lifetime_years
        npv = self._calculate_npv(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2, input_data.carbon_price_escalation_pct, input_data.discount_rate_pct, input_data.project_lifetime_years)
        payback = self._calculate_payback(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2)
        lac = (input_data.capex_usd / total_carbon).quantize(Decimal("0.01")) if total_carbon > 0 else Decimal("0")
        return FinanceOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, capex_usd=input_data.capex_usd, npv_usd=npv, payback_years=payback, levelized_abatement_cost_usd_per_tco2=lac, total_carbon_savings_tco2e=total_carbon, eligible_financing_types=[FinancingType.TRANSITION_FINANCE, FinancingType.GREEN_BOND], green_bond_eligible=True, is_valid=True)


class ChemicalsFinanceAgent(IndustrialFinanceBaseAgent):
    """GL-FIN-IND-003: Chemicals Finance Agent"""
    AGENT_ID = "GL-FIN-IND-003"
    SECTOR = "Chemicals"

    def analyze(self, input_data: FinanceInput) -> FinanceOutput:
        calc_id = hashlib.sha256(f"{self.AGENT_ID}:{input_data.facility_id}".encode()).hexdigest()[:16]
        total_carbon = input_data.annual_emission_reduction_tco2e * input_data.project_lifetime_years
        npv = self._calculate_npv(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2, input_data.carbon_price_escalation_pct, input_data.discount_rate_pct, input_data.project_lifetime_years)
        payback = self._calculate_payback(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2)
        lac = (input_data.capex_usd / total_carbon).quantize(Decimal("0.01")) if total_carbon > 0 else Decimal("0")
        return FinanceOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, capex_usd=input_data.capex_usd, npv_usd=npv, payback_years=payback, levelized_abatement_cost_usd_per_tco2=lac, total_carbon_savings_tco2e=total_carbon, eligible_financing_types=[FinancingType.TRANSITION_FINANCE], green_bond_eligible=True, is_valid=True)


class AluminumFinanceAgent(IndustrialFinanceBaseAgent):
    """GL-FIN-IND-004: Aluminum Finance Agent"""
    AGENT_ID = "GL-FIN-IND-004"
    SECTOR = "Aluminum"

    def analyze(self, input_data: FinanceInput) -> FinanceOutput:
        calc_id = hashlib.sha256(f"{self.AGENT_ID}:{input_data.facility_id}".encode()).hexdigest()[:16]
        total_carbon = input_data.annual_emission_reduction_tco2e * input_data.project_lifetime_years
        npv = self._calculate_npv(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2, input_data.carbon_price_escalation_pct, input_data.discount_rate_pct, input_data.project_lifetime_years)
        payback = self._calculate_payback(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2)
        lac = (input_data.capex_usd / total_carbon).quantize(Decimal("0.01")) if total_carbon > 0 else Decimal("0")
        return FinanceOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, capex_usd=input_data.capex_usd, npv_usd=npv, payback_years=payback, levelized_abatement_cost_usd_per_tco2=lac, total_carbon_savings_tco2e=total_carbon, eligible_financing_types=[FinancingType.GREEN_BOND, FinancingType.SUSTAINABILITY_LINKED_LOAN], green_bond_eligible=True, is_valid=True)


class PulpPaperFinanceAgent(IndustrialFinanceBaseAgent):
    """GL-FIN-IND-005: Pulp & Paper Finance Agent"""
    AGENT_ID = "GL-FIN-IND-005"
    SECTOR = "Pulp & Paper"

    def analyze(self, input_data: FinanceInput) -> FinanceOutput:
        calc_id = hashlib.sha256(f"{self.AGENT_ID}:{input_data.facility_id}".encode()).hexdigest()[:16]
        total_carbon = input_data.annual_emission_reduction_tco2e * input_data.project_lifetime_years
        npv = self._calculate_npv(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2, input_data.carbon_price_escalation_pct, input_data.discount_rate_pct, input_data.project_lifetime_years)
        lac = (input_data.capex_usd / total_carbon).quantize(Decimal("0.01")) if total_carbon > 0 else Decimal("0")
        return FinanceOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, capex_usd=input_data.capex_usd, npv_usd=npv, levelized_abatement_cost_usd_per_tco2=lac, total_carbon_savings_tco2e=total_carbon, eligible_financing_types=[FinancingType.GREEN_BOND], green_bond_eligible=True, is_valid=True)


class GlassFinanceAgent(IndustrialFinanceBaseAgent):
    """GL-FIN-IND-006: Glass Finance Agent"""
    AGENT_ID = "GL-FIN-IND-006"
    SECTOR = "Glass"

    def analyze(self, input_data: FinanceInput) -> FinanceOutput:
        calc_id = hashlib.sha256(f"{self.AGENT_ID}:{input_data.facility_id}".encode()).hexdigest()[:16]
        total_carbon = input_data.annual_emission_reduction_tco2e * input_data.project_lifetime_years
        npv = self._calculate_npv(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2, input_data.carbon_price_escalation_pct, input_data.discount_rate_pct, input_data.project_lifetime_years)
        lac = (input_data.capex_usd / total_carbon).quantize(Decimal("0.01")) if total_carbon > 0 else Decimal("0")
        return FinanceOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, capex_usd=input_data.capex_usd, npv_usd=npv, levelized_abatement_cost_usd_per_tco2=lac, total_carbon_savings_tco2e=total_carbon, eligible_financing_types=[FinancingType.TRANSITION_FINANCE], green_bond_eligible=True, is_valid=True)


class FoodProcessingFinanceAgent(IndustrialFinanceBaseAgent):
    """GL-FIN-IND-007: Food Processing Finance Agent"""
    AGENT_ID = "GL-FIN-IND-007"
    SECTOR = "Food Processing"

    def analyze(self, input_data: FinanceInput) -> FinanceOutput:
        calc_id = hashlib.sha256(f"{self.AGENT_ID}:{input_data.facility_id}".encode()).hexdigest()[:16]
        total_carbon = input_data.annual_emission_reduction_tco2e * input_data.project_lifetime_years
        npv = self._calculate_npv(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2, input_data.carbon_price_escalation_pct, input_data.discount_rate_pct, input_data.project_lifetime_years)
        lac = (input_data.capex_usd / total_carbon).quantize(Decimal("0.01")) if total_carbon > 0 else Decimal("0")
        return FinanceOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, capex_usd=input_data.capex_usd, npv_usd=npv, levelized_abatement_cost_usd_per_tco2=lac, total_carbon_savings_tco2e=total_carbon, eligible_financing_types=[FinancingType.GREEN_BOND, FinancingType.SUSTAINABILITY_LINKED_LOAN], green_bond_eligible=True, is_valid=True)


class PharmaceuticalFinanceAgent(IndustrialFinanceBaseAgent):
    """GL-FIN-IND-008: Pharmaceutical Finance Agent"""
    AGENT_ID = "GL-FIN-IND-008"
    SECTOR = "Pharmaceutical"

    def analyze(self, input_data: FinanceInput) -> FinanceOutput:
        calc_id = hashlib.sha256(f"{self.AGENT_ID}:{input_data.facility_id}".encode()).hexdigest()[:16]
        total_carbon = input_data.annual_emission_reduction_tco2e * input_data.project_lifetime_years
        npv = self._calculate_npv(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2, input_data.carbon_price_escalation_pct, input_data.discount_rate_pct, input_data.project_lifetime_years)
        lac = (input_data.capex_usd / total_carbon).quantize(Decimal("0.01")) if total_carbon > 0 else Decimal("0")
        return FinanceOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, capex_usd=input_data.capex_usd, npv_usd=npv, levelized_abatement_cost_usd_per_tco2=lac, total_carbon_savings_tco2e=total_carbon, eligible_financing_types=[FinancingType.GREEN_BOND], green_bond_eligible=True, is_valid=True)


class ElectronicsFinanceAgent(IndustrialFinanceBaseAgent):
    """GL-FIN-IND-009: Electronics Finance Agent"""
    AGENT_ID = "GL-FIN-IND-009"
    SECTOR = "Electronics"

    def analyze(self, input_data: FinanceInput) -> FinanceOutput:
        calc_id = hashlib.sha256(f"{self.AGENT_ID}:{input_data.facility_id}".encode()).hexdigest()[:16]
        total_carbon = input_data.annual_emission_reduction_tco2e * input_data.project_lifetime_years
        npv = self._calculate_npv(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2, input_data.carbon_price_escalation_pct, input_data.discount_rate_pct, input_data.project_lifetime_years)
        lac = (input_data.capex_usd / total_carbon).quantize(Decimal("0.01")) if total_carbon > 0 else Decimal("0")
        return FinanceOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, capex_usd=input_data.capex_usd, npv_usd=npv, levelized_abatement_cost_usd_per_tco2=lac, total_carbon_savings_tco2e=total_carbon, eligible_financing_types=[FinancingType.GREEN_BOND, FinancingType.SUSTAINABILITY_LINKED_LOAN], green_bond_eligible=True, is_valid=True)


class AutomotiveFinanceAgent(IndustrialFinanceBaseAgent):
    """GL-FIN-IND-010: Automotive Finance Agent"""
    AGENT_ID = "GL-FIN-IND-010"
    SECTOR = "Automotive"

    def analyze(self, input_data: FinanceInput) -> FinanceOutput:
        calc_id = hashlib.sha256(f"{self.AGENT_ID}:{input_data.facility_id}".encode()).hexdigest()[:16]
        total_carbon = input_data.annual_emission_reduction_tco2e * input_data.project_lifetime_years
        npv = self._calculate_npv(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2, input_data.carbon_price_escalation_pct, input_data.discount_rate_pct, input_data.project_lifetime_years)
        lac = (input_data.capex_usd / total_carbon).quantize(Decimal("0.01")) if total_carbon > 0 else Decimal("0")
        return FinanceOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, capex_usd=input_data.capex_usd, npv_usd=npv, levelized_abatement_cost_usd_per_tco2=lac, total_carbon_savings_tco2e=total_carbon, eligible_financing_types=[FinancingType.GREEN_BOND, FinancingType.TRANSITION_FINANCE], green_bond_eligible=True, eu_taxonomy_aligned=True, is_valid=True)


class TextilesFinanceAgent(IndustrialFinanceBaseAgent):
    """GL-FIN-IND-011: Textiles Finance Agent"""
    AGENT_ID = "GL-FIN-IND-011"
    SECTOR = "Textiles"

    def analyze(self, input_data: FinanceInput) -> FinanceOutput:
        calc_id = hashlib.sha256(f"{self.AGENT_ID}:{input_data.facility_id}".encode()).hexdigest()[:16]
        total_carbon = input_data.annual_emission_reduction_tco2e * input_data.project_lifetime_years
        npv = self._calculate_npv(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2, input_data.carbon_price_escalation_pct, input_data.discount_rate_pct, input_data.project_lifetime_years)
        lac = (input_data.capex_usd / total_carbon).quantize(Decimal("0.01")) if total_carbon > 0 else Decimal("0")
        return FinanceOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, capex_usd=input_data.capex_usd, npv_usd=npv, levelized_abatement_cost_usd_per_tco2=lac, total_carbon_savings_tco2e=total_carbon, eligible_financing_types=[FinancingType.SUSTAINABILITY_LINKED_LOAN], green_bond_eligible=True, is_valid=True)


class MiningFinanceAgent(IndustrialFinanceBaseAgent):
    """GL-FIN-IND-012: Mining Finance Agent"""
    AGENT_ID = "GL-FIN-IND-012"
    SECTOR = "Mining"

    def analyze(self, input_data: FinanceInput) -> FinanceOutput:
        calc_id = hashlib.sha256(f"{self.AGENT_ID}:{input_data.facility_id}".encode()).hexdigest()[:16]
        total_carbon = input_data.annual_emission_reduction_tco2e * input_data.project_lifetime_years
        npv = self._calculate_npv(input_data.capex_usd, input_data.opex_savings_usd_annual, input_data.annual_emission_reduction_tco2e, input_data.carbon_price_usd_per_tco2, input_data.carbon_price_escalation_pct, input_data.discount_rate_pct, input_data.project_lifetime_years)
        lac = (input_data.capex_usd / total_carbon).quantize(Decimal("0.01")) if total_carbon > 0 else Decimal("0")
        return FinanceOutput(calculation_id=calc_id, agent_id=self.AGENT_ID, timestamp=self._get_timestamp(), facility_id=input_data.facility_id, sector=self.SECTOR, capex_usd=input_data.capex_usd, npv_usd=npv, levelized_abatement_cost_usd_per_tco2=lac, total_carbon_savings_tco2e=total_carbon, eligible_financing_types=[FinancingType.TRANSITION_FINANCE], green_bond_eligible=False, is_valid=True)
