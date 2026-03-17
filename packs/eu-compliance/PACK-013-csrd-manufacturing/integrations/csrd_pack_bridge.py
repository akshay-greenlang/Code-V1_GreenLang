"""
PACK-013 CSRD Manufacturing Pack - CSRD Pack Bridge.

Bridge to PACK-001 (Starter), PACK-002 (Professional), and PACK-003
(Enterprise) for base CSRD reporting.  Manufacturing-specific data is
mapped onto the standard ESRS disclosure templates so the base packs
can produce the final CSRD report.
"""

import hashlib
import importlib
import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CSRDPackTier(str, Enum):
    """Supported CSRD pack tiers."""
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class ESRSStandard(str, Enum):
    """ESRS topical standards relevant to manufacturing."""
    E1 = "E1"   # Climate change
    E2 = "E2"   # Pollution
    E3 = "E3"   # Water and marine resources
    E4 = "E4"   # Biodiversity
    E5 = "E5"   # Circular economy
    S1 = "S1"   # Own workforce
    S2 = "S2"   # Workers in the value chain
    G1 = "G1"   # Business conduct


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class CSRDBridgeConfig(BaseModel):
    """Configuration for the CSRD pack bridge."""
    pack_tier: CSRDPackTier = Field(default=CSRDPackTier.PROFESSIONAL)
    enabled_esrs: List[ESRSStandard] = Field(
        default_factory=lambda: [
            ESRSStandard.E1,
            ESRSStandard.E2,
            ESRSStandard.E3,
            ESRSStandard.E5,
        ]
    )
    pack_module_prefix: str = Field(
        default="packs.eu_compliance",
        description="Python module prefix for CSRD packs",
    )
    materiality_threshold: float = Field(
        default=0.05,
        description="Minimum materiality threshold (0-1)",
    )
    include_governance: bool = Field(default=True)
    include_strategy: bool = Field(default=True)
    double_materiality: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class MaterialityItem(BaseModel):
    """A single materiality assessment item."""
    topic: str
    esrs: str
    impact_score: float = Field(ge=0.0, le=1.0)
    financial_score: float = Field(ge=0.0, le=1.0)
    material: bool = Field(default=False)
    rationale: str = Field(default="")


class ESRSDisclosureData(BaseModel):
    """Data prepared for a single ESRS standard."""
    standard: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    narrative: str = Field(default="")
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    complete: bool = Field(default=False)


class CSRDBridgeResult(BaseModel):
    """Result of submitting manufacturing data to the CSRD base pack."""
    esrs_e1_data: Optional[ESRSDisclosureData] = Field(default=None)
    esrs_e2_data: Optional[ESRSDisclosureData] = Field(default=None)
    esrs_e3_data: Optional[ESRSDisclosureData] = Field(default=None)
    esrs_e5_data: Optional[ESRSDisclosureData] = Field(default=None)
    materiality_result: List[MaterialityItem] = Field(
        default_factory=list
    )
    governance_data: Dict[str, Any] = Field(default_factory=dict)
    strategy_data: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    pack_tier: CSRDPackTier = Field(default=CSRDPackTier.PROFESSIONAL)
    submission_status: str = Field(default="pending")


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class CSRDPackBridge:
    """
    Bridge between PACK-013 manufacturing data and PACK-001/002/003
    base CSRD reporting packs.

    Transforms manufacturing-specific metrics (process emissions,
    energy intensity, water withdrawal, waste streams) into the
    standard ESRS disclosure format expected by the base packs.
    """

    def __init__(
        self, config: Optional[CSRDBridgeConfig] = None
    ) -> None:
        self.config = config or CSRDBridgeConfig()
        self._pack_module: Any = None
        self._load_pack()

    # -- pack loading --------------------------------------------------------

    def _load_pack(self) -> None:
        """Import the appropriate base CSRD pack module."""
        tier_map = {
            CSRDPackTier.STARTER: "PACK_001_csrd_starter",
            CSRDPackTier.PROFESSIONAL: "PACK_002_csrd_professional",
            CSRDPackTier.ENTERPRISE: "PACK_003_csrd_enterprise",
        }
        module_name = (
            f"{self.config.pack_module_prefix}."
            f"{tier_map[self.config.pack_tier]}"
        )
        try:
            self._pack_module = importlib.import_module(module_name)
            logger.info("Loaded CSRD pack: %s", module_name)
        except ImportError:
            logger.warning(
                "CSRD pack %s not available; bridge will use "
                "built-in templates",
                module_name,
            )
            self._pack_module = None

    @staticmethod
    def _compute_hash(data: Any) -> str:
        raw = str(data).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    # -- public API ----------------------------------------------------------

    def get_base_csrd_data(self) -> Dict[str, Any]:
        """
        Retrieve base CSRD configuration and template structure from
        the loaded pack.
        """
        if self._pack_module and hasattr(
            self._pack_module, "get_config"
        ):
            return self._pack_module.get_config()

        # Built-in fallback
        return {
            "pack_tier": self.config.pack_tier.value,
            "enabled_esrs": [e.value for e in self.config.enabled_esrs],
            "double_materiality": self.config.double_materiality,
            "disclosure_templates": self._default_templates(),
        }

    def submit_manufacturing_data(
        self,
        emissions: Dict[str, Any],
        energy: Dict[str, Any],
        water: Dict[str, Any],
        waste: Dict[str, Any],
    ) -> CSRDBridgeResult:
        """
        Transform and submit manufacturing data to the base CSRD pack.

        Args:
            emissions: Scope 1/2/3 emission totals and breakdowns.
            energy: Energy consumption and intensity data.
            water: Water withdrawal, discharge, pollutant loads.
            waste: Waste generation and recycling data.

        Returns:
            CSRDBridgeResult with ESRS-formatted data and status.
        """
        esrs_e1 = self._build_e1(emissions, energy)
        esrs_e2 = self._build_e2(emissions, water)
        esrs_e3 = self._build_e3(water)
        esrs_e5 = self._build_e5(waste)

        materiality = self._assess_materiality(
            emissions, energy, water, waste
        )

        governance = (
            self._build_governance(emissions)
            if self.config.include_governance else {}
        )
        strategy = (
            self._build_strategy(emissions, energy)
            if self.config.include_strategy else {}
        )

        combined = {
            "e1": esrs_e1.model_dump() if esrs_e1 else {},
            "e2": esrs_e2.model_dump() if esrs_e2 else {},
            "e3": esrs_e3.model_dump() if esrs_e3 else {},
            "e5": esrs_e5.model_dump() if esrs_e5 else {},
            "materiality": [m.model_dump() for m in materiality],
        }

        return CSRDBridgeResult(
            esrs_e1_data=esrs_e1,
            esrs_e2_data=esrs_e2,
            esrs_e3_data=esrs_e3,
            esrs_e5_data=esrs_e5,
            materiality_result=materiality,
            governance_data=governance,
            strategy_data=strategy,
            provenance_hash=self._compute_hash(combined),
            pack_tier=self.config.pack_tier,
            submission_status="submitted",
        )

    def get_materiality_assessment(self) -> Dict[str, Any]:
        """Return a default manufacturing materiality template."""
        topics = [
            ("Climate change mitigation", "E1", 0.9, 0.85),
            ("Pollution to air", "E2", 0.8, 0.6),
            ("Water use", "E3", 0.7, 0.5),
            ("Circular economy", "E5", 0.75, 0.65),
            ("Worker health and safety", "S1", 0.85, 0.7),
            ("Supply chain labour", "S2", 0.6, 0.4),
            ("Anti-corruption", "G1", 0.5, 0.45),
        ]
        items = []
        for topic, esrs, impact, financial in topics:
            combined = (impact + financial) / 2
            items.append({
                "topic": topic,
                "esrs": esrs,
                "impact_score": impact,
                "financial_score": financial,
                "material": combined >= self.config.materiality_threshold,
            })
        return {"materiality_items": items}

    def get_esrs_disclosure_template(self) -> Dict[str, Any]:
        """Return ESRS disclosure templates for enabled standards."""
        return self._default_templates()

    # -- ESRS builders -------------------------------------------------------

    def _build_e1(
        self,
        emissions: Dict[str, Any],
        energy: Dict[str, Any],
    ) -> ESRSDisclosureData:
        """Build E1 Climate Change disclosure data."""
        scope1 = emissions.get("scope1_total", 0.0)
        scope2 = emissions.get("scope2_total", 0.0)
        scope3 = emissions.get("scope3_total", 0.0)

        metrics = {
            "E1-6_gross_scope1_tco2e": scope1,
            "E1-6_gross_scope2_tco2e": scope2,
            "E1-6_gross_scope3_tco2e": scope3,
            "E1-6_total_ghg_tco2e": scope1 + scope2 + scope3,
            "E1-5_energy_consumption_mwh": energy.get(
                "total_consumption_mwh", 0.0
            ),
            "E1-5_energy_intensity": energy.get(
                "energy_intensity", 0.0
            ),
            "E1-5_renewable_share_pct": energy.get(
                "renewable_share_pct", 0.0
            ),
            "E1-4_transition_plan": emissions.get(
                "transition_plan", "not_provided"
            ),
        }
        complete = scope1 > 0 or scope2 > 0
        return ESRSDisclosureData(
            standard="E1",
            metrics=metrics,
            narrative=(
                f"Total GHG emissions: {scope1 + scope2 + scope3:.2f} "
                f"tCO2e (Scope 1: {scope1:.2f}, Scope 2: {scope2:.2f}, "
                f"Scope 3: {scope3:.2f})"
            ),
            data_quality_score=85.0 if complete else 0.0,
            complete=complete,
        )

    def _build_e2(
        self,
        emissions: Dict[str, Any],
        water: Dict[str, Any],
    ) -> ESRSDisclosureData:
        """Build E2 Pollution disclosure data."""
        pollutant_loads = water.get("pollutant_loads", {})
        air_emissions = emissions.get("air_pollutants", {})

        metrics = {
            "E2-4_pollutants_to_water": pollutant_loads,
            "E2-4_pollutants_to_air": air_emissions,
            "E2-4_substances_of_concern": emissions.get(
                "substances_of_concern", []
            ),
        }
        return ESRSDisclosureData(
            standard="E2",
            metrics=metrics,
            narrative="Pollution disclosure for manufacturing operations",
            data_quality_score=70.0,
            complete=bool(pollutant_loads or air_emissions),
        )

    def _build_e3(
        self, water: Dict[str, Any]
    ) -> ESRSDisclosureData:
        """Build E3 Water and Marine Resources disclosure data."""
        metrics = {
            "E3-4_water_withdrawal_m3": water.get(
                "water_withdrawal_m3", 0.0
            ),
            "E3-4_water_discharge_m3": water.get(
                "water_discharge_m3", 0.0
            ),
            "E3-4_water_consumption_m3": water.get(
                "water_consumption_m3", 0.0
            ),
            "E3-4_water_intensity": water.get("water_intensity", 0.0),
            "E3-4_water_stressed_areas": water.get(
                "water_stressed_areas", False
            ),
        }
        return ESRSDisclosureData(
            standard="E3",
            metrics=metrics,
            narrative="Water and marine resources disclosure",
            data_quality_score=75.0,
            complete=water.get("water_withdrawal_m3", 0.0) > 0,
        )

    def _build_e5(
        self, waste: Dict[str, Any]
    ) -> ESRSDisclosureData:
        """Build E5 Circular Economy disclosure data."""
        metrics = {
            "E5-5_total_waste_kg": waste.get("total_waste_kg", 0.0),
            "E5-5_recycled_kg": waste.get("recycled_kg", 0.0),
            "E5-5_landfill_kg": waste.get("landfill_kg", 0.0),
            "E5-5_recycling_rate_pct": waste.get(
                "recycling_rate_pct", 0.0
            ),
            "E5-5_circular_material_use_rate": waste.get(
                "circular_material_use_rate", 0.0
            ),
            "E5-5_hazardous_waste_kg": waste.get(
                "hazardous_waste_kg", 0.0
            ),
        }
        return ESRSDisclosureData(
            standard="E5",
            metrics=metrics,
            narrative="Circular economy and waste management disclosure",
            data_quality_score=80.0,
            complete=waste.get("total_waste_kg", 0.0) > 0,
        )

    # -- materiality ---------------------------------------------------------

    def _assess_materiality(
        self,
        emissions: Dict[str, Any],
        energy: Dict[str, Any],
        water: Dict[str, Any],
        waste: Dict[str, Any],
    ) -> List[MaterialityItem]:
        """Run double materiality assessment for manufacturing."""
        threshold = self.config.materiality_threshold
        items: List[MaterialityItem] = []

        scope1 = emissions.get("scope1_total", 0.0)
        climate_impact = min(scope1 / 10000.0, 1.0) if scope1 > 0 else 0.3
        items.append(MaterialityItem(
            topic="Climate change - GHG emissions",
            esrs="E1",
            impact_score=climate_impact,
            financial_score=0.85,
            material=(climate_impact + 0.85) / 2 >= threshold,
            rationale=f"Scope 1 emissions: {scope1:.2f} tCO2e",
        ))

        water_vol = water.get("water_withdrawal_m3", 0.0)
        water_impact = min(water_vol / 100000.0, 1.0) if water_vol > 0 else 0.2
        items.append(MaterialityItem(
            topic="Water use and pollution",
            esrs="E3",
            impact_score=water_impact,
            financial_score=0.5,
            material=(water_impact + 0.5) / 2 >= threshold,
            rationale=f"Water withdrawal: {water_vol:.0f} m3",
        ))

        waste_total = waste.get("total_waste_kg", 0.0)
        waste_impact = min(waste_total / 500000.0, 1.0) if waste_total > 0 else 0.2
        items.append(MaterialityItem(
            topic="Circular economy - waste",
            esrs="E5",
            impact_score=waste_impact,
            financial_score=0.65,
            material=(waste_impact + 0.65) / 2 >= threshold,
            rationale=f"Total waste: {waste_total:.0f} kg",
        ))

        items.append(MaterialityItem(
            topic="Pollution to air and water",
            esrs="E2",
            impact_score=0.7,
            financial_score=0.55,
            material=True,
            rationale="Manufacturing inherently produces pollutants",
        ))

        return items

    # -- governance and strategy ---------------------------------------------

    @staticmethod
    def _build_governance(
        emissions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build governance section data."""
        return {
            "board_oversight": True,
            "management_responsibility": (
                "Chief Sustainability Officer oversees climate targets"
            ),
            "climate_in_remuneration": emissions.get(
                "climate_in_remuneration", False
            ),
            "risk_management_integration": True,
        }

    @staticmethod
    def _build_strategy(
        emissions: Dict[str, Any],
        energy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build strategy section data."""
        return {
            "transition_plan": emissions.get(
                "transition_plan", "under_development"
            ),
            "scenario_analysis": emissions.get(
                "scenario_analysis", "not_performed"
            ),
            "energy_efficiency_target_pct": energy.get(
                "efficiency_target_pct", 0.0
            ),
            "renewable_target_pct": energy.get(
                "renewable_target_pct", 0.0
            ),
        }

    # -- templates -----------------------------------------------------------

    @staticmethod
    def _default_templates() -> Dict[str, Any]:
        """Return built-in ESRS disclosure templates."""
        return {
            "E1": {
                "required_metrics": [
                    "E1-6_gross_scope1_tco2e",
                    "E1-6_gross_scope2_tco2e",
                    "E1-6_gross_scope3_tco2e",
                    "E1-5_energy_consumption_mwh",
                ],
                "optional_metrics": [
                    "E1-5_energy_intensity",
                    "E1-4_transition_plan",
                ],
            },
            "E2": {
                "required_metrics": [
                    "E2-4_pollutants_to_water",
                    "E2-4_pollutants_to_air",
                ],
                "optional_metrics": [
                    "E2-4_substances_of_concern",
                ],
            },
            "E3": {
                "required_metrics": [
                    "E3-4_water_withdrawal_m3",
                    "E3-4_water_consumption_m3",
                ],
                "optional_metrics": [
                    "E3-4_water_intensity",
                    "E3-4_water_stressed_areas",
                ],
            },
            "E5": {
                "required_metrics": [
                    "E5-5_total_waste_kg",
                    "E5-5_recycling_rate_pct",
                ],
                "optional_metrics": [
                    "E5-5_circular_material_use_rate",
                    "E5-5_hazardous_waste_kg",
                ],
            },
        }
