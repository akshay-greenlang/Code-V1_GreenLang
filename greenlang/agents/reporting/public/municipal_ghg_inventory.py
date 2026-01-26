# -*- coding: utf-8 -*-
"""
GL-REP-PUB-001: Municipal GHG Inventory Agent
==============================================

Creates city-level GHG inventories following the Global Protocol for
Community-Scale GHG Inventories (GPC).

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class GPCSector(str, Enum):
    """GPC sectors."""
    STATIONARY_ENERGY = "stationary_energy"
    TRANSPORTATION = "transportation"
    WASTE = "waste"
    IPPU = "ippu"  # Industrial processes and product use
    AFOLU = "afolu"  # Agriculture, forestry, land use


class SectorEmission(BaseModel):
    """Emissions for a GPC sector."""
    sector: GPCSector = Field(...)
    subsector: str = Field(...)
    scope_1_tco2e: float = Field(default=0.0, ge=0)
    scope_2_tco2e: float = Field(default=0.0, ge=0)
    scope_3_tco2e: float = Field(default=0.0, ge=0)
    data_quality: str = Field(default="medium")
    methodology: str = Field(default="activity_data")
    source_reference: Optional[str] = Field(None)


class GPCInventory(BaseModel):
    """GPC-compliant GHG inventory."""
    inventory_id: str = Field(...)
    municipality_name: str = Field(...)
    reporting_year: int = Field(...)
    population: int = Field(default=0, ge=0)
    emissions: List[SectorEmission] = Field(default_factory=list)
    total_scope_1_tco2e: float = Field(default=0.0, ge=0)
    total_scope_2_tco2e: float = Field(default=0.0, ge=0)
    total_scope_3_tco2e: float = Field(default=0.0, ge=0)
    total_basic_tco2e: float = Field(default=0.0, ge=0)
    total_basic_plus_tco2e: float = Field(default=0.0, ge=0)
    per_capita_tco2e: float = Field(default=0.0, ge=0)
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    provenance_hash: Optional[str] = Field(None)


class MunicipalGHGInventoryInput(BaseModel):
    """Input for Municipal GHG Inventory Agent."""
    action: str = Field(...)
    inventory_id: Optional[str] = Field(None)
    municipality_name: Optional[str] = Field(None)
    reporting_year: Optional[int] = Field(None)
    population: Optional[int] = Field(None)
    sector_emission: Optional[SectorEmission] = Field(None)
    user_id: Optional[str] = Field(None)

    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid = {'create_inventory', 'add_sector_emission', 'calculate_totals',
                 'generate_report', 'get_inventory', 'list_inventories'}
        if v not in valid:
            raise ValueError(f"Invalid action: {v}")
        return v


class MunicipalGHGInventoryOutput(BaseModel):
    """Output from Municipal GHG Inventory Agent."""
    success: bool = Field(...)
    action: str = Field(...)
    inventory: Optional[GPCInventory] = Field(None)
    inventories: Optional[List[GPCInventory]] = Field(None)
    report: Optional[Dict[str, Any]] = Field(None)
    provenance_hash: Optional[str] = Field(None)
    calculation_trace: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(None)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    processing_time_ms: float = Field(default=0.0)


class MunicipalGHGInventoryAgent(BaseAgent):
    """GL-REP-PUB-001: Municipal GHG Inventory Agent"""

    AGENT_ID = "GL-REP-PUB-001"
    AGENT_NAME = "Municipal GHG Inventory Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(name=self.AGENT_NAME, description="GPC-compliant municipal GHG inventories", version=self.VERSION)
        super().__init__(config)
        self._inventories: Dict[str, GPCInventory] = {}

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        import time
        start_time = time.time()
        try:
            inp = MunicipalGHGInventoryInput(**input_data)
            handlers = {
                'create_inventory': self._create_inventory,
                'add_sector_emission': self._add_emission,
                'calculate_totals': self._calculate_totals,
                'generate_report': self._generate_report,
                'get_inventory': self._get_inventory,
                'list_inventories': self._list_inventories,
            }
            out = handlers[inp.action](inp)
            out.processing_time_ms = (time.time() - start_time) * 1000
            out.provenance_hash = hashlib.sha256(json.dumps({"action": out.action, "success": out.success}, sort_keys=True).encode()).hexdigest()
            return AgentResult(success=out.success, data=out.model_dump(), error=out.error)
        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _create_inventory(self, inp: MunicipalGHGInventoryInput) -> MunicipalGHGInventoryOutput:
        if not inp.municipality_name or not inp.reporting_year:
            return MunicipalGHGInventoryOutput(success=False, action='create_inventory', error="Municipality and year required")
        inv_id = f"GPC-{inp.municipality_name.upper()[:3]}-{inp.reporting_year}"
        inv = GPCInventory(inventory_id=inv_id, municipality_name=inp.municipality_name, reporting_year=inp.reporting_year, population=inp.population or 0)
        self._inventories[inv_id] = inv
        return MunicipalGHGInventoryOutput(success=True, action='create_inventory', inventory=inv, calculation_trace=[f"Created {inv_id}"])

    def _add_emission(self, inp: MunicipalGHGInventoryInput) -> MunicipalGHGInventoryOutput:
        if not inp.inventory_id or not inp.sector_emission:
            return MunicipalGHGInventoryOutput(success=False, action='add_sector_emission', error="Inventory ID and emission required")
        inv = self._inventories.get(inp.inventory_id)
        if not inv:
            return MunicipalGHGInventoryOutput(success=False, action='add_sector_emission', error="Inventory not found")
        inv.emissions.append(inp.sector_emission)
        return MunicipalGHGInventoryOutput(success=True, action='add_sector_emission', inventory=inv, calculation_trace=[f"Added {inp.sector_emission.subsector}"])

    def _calculate_totals(self, inp: MunicipalGHGInventoryInput) -> MunicipalGHGInventoryOutput:
        if not inp.inventory_id:
            return MunicipalGHGInventoryOutput(success=False, action='calculate_totals', error="Inventory ID required")
        inv = self._inventories.get(inp.inventory_id)
        if not inv:
            return MunicipalGHGInventoryOutput(success=False, action='calculate_totals', error="Inventory not found")
        inv.total_scope_1_tco2e = sum(e.scope_1_tco2e for e in inv.emissions)
        inv.total_scope_2_tco2e = sum(e.scope_2_tco2e for e in inv.emissions)
        inv.total_scope_3_tco2e = sum(e.scope_3_tco2e for e in inv.emissions)
        inv.total_basic_tco2e = inv.total_scope_1_tco2e + inv.total_scope_2_tco2e
        inv.total_basic_plus_tco2e = inv.total_basic_tco2e + inv.total_scope_3_tco2e
        inv.per_capita_tco2e = inv.total_basic_plus_tco2e / inv.population if inv.population > 0 else 0
        return MunicipalGHGInventoryOutput(success=True, action='calculate_totals', inventory=inv, calculation_trace=[f"Total: {inv.total_basic_plus_tco2e:.2f} tCO2e"])

    def _generate_report(self, inp: MunicipalGHGInventoryInput) -> MunicipalGHGInventoryOutput:
        if not inp.inventory_id:
            return MunicipalGHGInventoryOutput(success=False, action='generate_report', error="Inventory ID required")
        inv = self._inventories.get(inp.inventory_id)
        if not inv:
            return MunicipalGHGInventoryOutput(success=False, action='generate_report', error="Inventory not found")
        by_sector = {}
        for e in inv.emissions:
            s = e.sector.value
            if s not in by_sector:
                by_sector[s] = 0
            by_sector[s] += e.scope_1_tco2e + e.scope_2_tco2e + e.scope_3_tco2e
        report = {"municipality": inv.municipality_name, "year": inv.reporting_year, "total_tco2e": inv.total_basic_plus_tco2e, "per_capita": inv.per_capita_tco2e, "by_sector": by_sector}
        return MunicipalGHGInventoryOutput(success=True, action='generate_report', inventory=inv, report=report, calculation_trace=["Report generated"])

    def _get_inventory(self, inp: MunicipalGHGInventoryInput) -> MunicipalGHGInventoryOutput:
        inv = self._inventories.get(inp.inventory_id) if inp.inventory_id else None
        if not inv:
            return MunicipalGHGInventoryOutput(success=False, action='get_inventory', error="Inventory not found")
        return MunicipalGHGInventoryOutput(success=True, action='get_inventory', inventory=inv)

    def _list_inventories(self, inp: MunicipalGHGInventoryInput) -> MunicipalGHGInventoryOutput:
        return MunicipalGHGInventoryOutput(success=True, action='list_inventories', inventories=list(self._inventories.values()))
