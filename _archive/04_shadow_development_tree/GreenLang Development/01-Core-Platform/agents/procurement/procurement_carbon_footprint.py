# -*- coding: utf-8 -*-
"""
GL-PROC-X-004: Procurement Carbon Footprint Agent
==================================================

Calculates carbon footprint of procurement activities using
spend-based and supplier-specific emission factors.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


class CalculationMethod(str, Enum):
    SPEND_BASED = "spend_based"
    SUPPLIER_SPECIFIC = "supplier_specific"
    AVERAGE_DATA = "average_data"
    HYBRID = "hybrid"


class SpendCategory(str, Enum):
    RAW_MATERIALS = "raw_materials"
    PACKAGING = "packaging"
    CAPITAL_GOODS = "capital_goods"
    SERVICES = "services"
    LOGISTICS = "logistics"
    IT_EQUIPMENT = "it_equipment"
    ENERGY = "energy"
    OFFICE_SUPPLIES = "office_supplies"


class ProcurementItem(BaseModel):
    item_id: str = Field(...)
    description: str = Field(...)
    category: SpendCategory = Field(...)
    spend_amount_usd: float = Field(..., ge=0)
    supplier_id: Optional[str] = Field(None)
    supplier_emission_factor: Optional[float] = Field(None)
    quantity: Optional[float] = Field(None)
    unit: Optional[str] = Field(None)


class EmissionCalculation(BaseModel):
    item_id: str = Field(...)
    category: SpendCategory = Field(...)
    method_used: CalculationMethod = Field(...)
    emissions_tco2e: float = Field(...)
    emission_factor_used: float = Field(...)
    emission_factor_source: str = Field(...)
    data_quality_score: float = Field(..., ge=0, le=1)


class ProcurementSummary(BaseModel):
    total_spend_usd: float = Field(...)
    total_emissions_tco2e: float = Field(...)
    emissions_intensity_kgco2e_per_usd: float = Field(...)
    category_breakdown: Dict[str, float] = Field(...)
    method_breakdown: Dict[str, float] = Field(...)
    data_quality_avg: float = Field(...)
    recommendations: List[str] = Field(...)


class ProcurementFootprintInput(BaseModel):
    organization_id: str = Field(...)
    reporting_period: str = Field(...)
    procurement_items: List[ProcurementItem] = Field(...)
    preferred_method: CalculationMethod = Field(default=CalculationMethod.HYBRID)


class ProcurementFootprintOutput(BaseModel):
    organization_id: str = Field(...)
    reporting_period: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    item_calculations: List[EmissionCalculation] = Field(...)
    summary: ProcurementSummary = Field(...)
    provenance_hash: str = Field(...)


class ProcurementCarbonFootprintAgent(BaseAgent):
    """GL-PROC-X-004: Procurement Carbon Footprint Agent"""

    AGENT_ID = "GL-PROC-X-004"
    AGENT_NAME = "Procurement Carbon Footprint Agent"
    VERSION = "1.0.0"

    SPEND_BASED_FACTORS = {
        SpendCategory.RAW_MATERIALS: {"factor": 0.5, "source": "EEIO_2024"},
        SpendCategory.PACKAGING: {"factor": 0.35, "source": "EEIO_2024"},
        SpendCategory.CAPITAL_GOODS: {"factor": 0.25, "source": "EEIO_2024"},
        SpendCategory.SERVICES: {"factor": 0.1, "source": "EEIO_2024"},
        SpendCategory.LOGISTICS: {"factor": 0.45, "source": "EEIO_2024"},
        SpendCategory.IT_EQUIPMENT: {"factor": 0.3, "source": "EEIO_2024"},
        SpendCategory.ENERGY: {"factor": 0.6, "source": "IEA_2024"},
        SpendCategory.OFFICE_SUPPLIES: {"factor": 0.15, "source": "EEIO_2024"},
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Procurement carbon footprint calculation",
                version=self.VERSION
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = ProcurementFootprintInput(**input_data)
            calculations = []
            category_emissions = {}
            method_emissions = {}

            for item in agent_input.procurement_items:
                calc = self._calculate_item(item, agent_input.preferred_method)
                calculations.append(calc)

                cat_key = calc.category.value
                category_emissions[cat_key] = category_emissions.get(cat_key, 0) + calc.emissions_tco2e

                method_key = calc.method_used.value
                method_emissions[method_key] = method_emissions.get(method_key, 0) + calc.emissions_tco2e

            total_spend = sum(item.spend_amount_usd for item in agent_input.procurement_items)
            total_emissions = sum(c.emissions_tco2e for c in calculations)
            avg_quality = sum(c.data_quality_score for c in calculations) / len(calculations) if calculations else 0

            intensity = (total_emissions * 1000) / total_spend if total_spend > 0 else 0

            recommendations = self._generate_recommendations(
                category_emissions, total_emissions, avg_quality
            )

            summary = ProcurementSummary(
                total_spend_usd=round(total_spend, 2),
                total_emissions_tco2e=round(total_emissions, 2),
                emissions_intensity_kgco2e_per_usd=round(intensity, 4),
                category_breakdown=category_emissions,
                method_breakdown=method_emissions,
                data_quality_avg=round(avg_quality, 2),
                recommendations=recommendations,
            )

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = ProcurementFootprintOutput(
                organization_id=agent_input.organization_id,
                reporting_period=agent_input.reporting_period,
                item_calculations=calculations,
                summary=summary,
                provenance_hash=provenance_hash,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _calculate_item(
        self, item: ProcurementItem, preferred_method: CalculationMethod
    ) -> EmissionCalculation:
        if item.supplier_emission_factor and preferred_method in (
            CalculationMethod.SUPPLIER_SPECIFIC, CalculationMethod.HYBRID
        ):
            method = CalculationMethod.SUPPLIER_SPECIFIC
            factor = item.supplier_emission_factor
            source = f"Supplier {item.supplier_id}"
            quality = 0.9
        else:
            method = CalculationMethod.SPEND_BASED
            factor_data = self.SPEND_BASED_FACTORS.get(
                item.category,
                {"factor": 0.2, "source": "EEIO_default"}
            )
            factor = factor_data["factor"]
            source = factor_data["source"]
            quality = 0.5

        emissions = item.spend_amount_usd * factor / 1000

        return EmissionCalculation(
            item_id=item.item_id,
            category=item.category,
            method_used=method,
            emissions_tco2e=round(emissions, 4),
            emission_factor_used=factor,
            emission_factor_source=source,
            data_quality_score=quality,
        )

    def _generate_recommendations(
        self,
        category_emissions: Dict[str, float],
        total_emissions: float,
        avg_quality: float
    ) -> List[str]:
        recommendations = []

        if avg_quality < 0.6:
            recommendations.append(
                "Improve data quality by collecting supplier-specific emission factors"
            )

        if total_emissions > 0:
            sorted_cats = sorted(category_emissions.items(), key=lambda x: x[1], reverse=True)
            top_cat = sorted_cats[0][0] if sorted_cats else None
            if top_cat:
                recommendations.append(
                    f"Focus supplier engagement on {top_cat} category (highest emissions)"
                )

        recommendations.append("Request PCF data from top 20 suppliers by spend")
        recommendations.append("Set science-based supplier targets for emissions reduction")

        return recommendations
