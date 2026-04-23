"""
BACT (Best Available Control Technology) Analyzer for GL-004 BURNMASTER

Comprehensive analyzer for evaluating air pollution control technologies under
EPA's New Source Review (NSR) Prevention of Significant Deterioration (PSD) program.

This module implements:
- Control technology database with efficiency and cost data
- Top-Down BACT analysis methodology (5-step process)
- Cost-effectiveness analysis per EPA guidance
- RACT/BACT/LAER comparison for attainment status
- Multi-pollutant co-benefit analysis
- Technology combination optimization
- RBLC-format documentation generation

Regulatory References:
- 40 CFR 52.21: Prevention of Significant Deterioration
- EPA's Draft NSR Workshop Manual (1990)
- EPA's RACT/BACT/LAER Clearinghouse (RBLC)
- 40 CFR 51.166: PSD requirements for SIPs
- EPA's Economic Guidance for Air Quality Management (EPA 452/R-11-013)

Author: GL-RegulatoryIntelligence
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json


# =============================================================================
# Enums and Type Definitions
# =============================================================================

class PollutantType(str, Enum):
    """Regulated criteria pollutants for BACT analysis."""
    NOX = "NOx"
    SO2 = "SO2"
    PM = "PM"
    PM10 = "PM10"
    PM25 = "PM2.5"
    CO = "CO"
    VOC = "VOC"
    NH3 = "NH3"  # Ammonia slip from SCR/SNCR


class AttainmentStatus(str, Enum):
    """NAAQS attainment status for determining BACT vs LAER applicability."""
    ATTAINMENT = "attainment"           # PSD applies -> BACT required
    NONATTAINMENT = "nonattainment"     # NSR applies -> LAER required
    UNCLASSIFIABLE = "unclassifiable"   # Treated as attainment for PSD


class FeasibilityStatus(str, Enum):
    """Technical feasibility determination for control technologies."""
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    CONDITIONALLY_FEASIBLE = "conditionally_feasible"


class TechnologyRanking(str, Enum):
    """Rankings from top-down analysis."""
    BACT = "bact"           # Selected as BACT
    ELIMINATED_COST = "eliminated_cost"        # Eliminated due to cost
    ELIMINATED_TECHNICAL = "eliminated_technical"  # Technically infeasible
    ELIMINATED_ENERGY = "eliminated_energy"    # Energy impacts
    ELIMINATED_ENVIRONMENTAL = "eliminated_environmental"  # Environmental impacts
    CONSIDERED = "considered"  # Under consideration


class FuelType(str, Enum):
    """Fuel types for applicability determination."""
    NATURAL_GAS = "gas"
    FUEL_OIL = "oil"
    COAL = "coal"
    BIOMASS = "biomass"
    REFINERY_GAS = "refinery_gas"
    HYDROGEN = "hydrogen"


# =============================================================================
# Control Technology Database
# =============================================================================

# NOx Control Technologies
# Reference: EPA RBLC, AP-42 Chapter 1, vendor data
NOX_CONTROL_TECHNOLOGIES: Dict[str, Dict[str, Any]] = {
    "SCR": {
        "name": "Selective Catalytic Reduction",
        "description": "Post-combustion NOx reduction using ammonia/urea catalyst",
        "efficiency_range": (0.85, 0.95),
        "typical_efficiency": 0.90,
        "cost_per_ton_range": (5000, 15000),
        "typical_cost_per_ton": 8500,
        "capital_cost_per_mmbtu_hr": 15000,  # $/MMBtu/hr capacity
        "operating_cost_pct_of_capital": 0.08,  # Annual O&M as % of capital
        "applicable_fuels": [FuelType.NATURAL_GAS, FuelType.FUEL_OIL, FuelType.COAL],
        "temperature_range_f": (600, 750),  # Optimal operating range
        "ammonia_slip_ppm": 5,  # Typical ammonia slip
        "catalyst_life_years": 5,
        "requires_catalyst_regeneration": True,
        "energy_penalty_pct": 0.5,  # % of plant output
        "achievable_emission_rates": {
            "natural_gas_ppm": 5,
            "fuel_oil_ppm": 20,
            "coal_lb_mmbtu": 0.04
        },
        "rblc_process_codes": ["11.110", "11.310"],  # Boilers/heaters
        "co_benefits": {"mercury": 0.10, "dioxins": 0.05},  # Co-removal rates
        "environmental_impacts": ["ammonia_storage", "catalyst_disposal"]
    },
    "SNCR": {
        "name": "Selective Non-Catalytic Reduction",
        "description": "Post-combustion NOx reduction using ammonia/urea without catalyst",
        "efficiency_range": (0.30, 0.60),
        "typical_efficiency": 0.50,
        "cost_per_ton_range": (1500, 3500),
        "typical_cost_per_ton": 2500,
        "capital_cost_per_mmbtu_hr": 3500,
        "operating_cost_pct_of_capital": 0.12,
        "applicable_fuels": [FuelType.NATURAL_GAS, FuelType.FUEL_OIL, FuelType.COAL],
        "temperature_range_f": (1600, 2100),  # Critical temperature window
        "ammonia_slip_ppm": 10,
        "catalyst_life_years": None,  # No catalyst
        "requires_catalyst_regeneration": False,
        "energy_penalty_pct": 0.2,
        "achievable_emission_rates": {
            "natural_gas_ppm": 20,
            "fuel_oil_ppm": 50,
            "coal_lb_mmbtu": 0.15
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {},
        "environmental_impacts": ["ammonia_storage", "N2O_formation"]
    },
    "Low_NOx_Burner": {
        "name": "Low NOx Burner (LNB)",
        "description": "Staged combustion burner reducing thermal NOx formation",
        "efficiency_range": (0.40, 0.60),
        "typical_efficiency": 0.50,
        "cost_per_ton_range": (500, 2500),
        "typical_cost_per_ton": 1200,
        "capital_cost_per_mmbtu_hr": 2500,
        "operating_cost_pct_of_capital": 0.03,
        "applicable_fuels": [FuelType.NATURAL_GAS, FuelType.FUEL_OIL],
        "temperature_range_f": None,  # Integrated into burner
        "ammonia_slip_ppm": 0,
        "catalyst_life_years": None,
        "requires_catalyst_regeneration": False,
        "energy_penalty_pct": 0.0,  # May improve efficiency
        "achievable_emission_rates": {
            "natural_gas_ppm": 30,
            "fuel_oil_ppm": 100
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {},
        "environmental_impacts": []
    },
    "Ultra_Low_NOx_Burner": {
        "name": "Ultra Low NOx Burner (ULNB)",
        "description": "Advanced staged combustion with internal FGR",
        "efficiency_range": (0.70, 0.85),
        "typical_efficiency": 0.75,
        "cost_per_ton_range": (1000, 3500),
        "typical_cost_per_ton": 2000,
        "capital_cost_per_mmbtu_hr": 4500,
        "operating_cost_pct_of_capital": 0.04,
        "applicable_fuels": [FuelType.NATURAL_GAS],  # Primarily gas
        "temperature_range_f": None,
        "ammonia_slip_ppm": 0,
        "catalyst_life_years": None,
        "requires_catalyst_regeneration": False,
        "energy_penalty_pct": 0.0,
        "achievable_emission_rates": {
            "natural_gas_ppm": 9
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {},
        "environmental_impacts": []
    },
    "FGR": {
        "name": "Flue Gas Recirculation",
        "description": "Recirculates cooled flue gas to reduce flame temperature",
        "efficiency_range": (0.50, 0.70),
        "typical_efficiency": 0.60,
        "cost_per_ton_range": (800, 3000),
        "typical_cost_per_ton": 1500,
        "capital_cost_per_mmbtu_hr": 3000,
        "operating_cost_pct_of_capital": 0.05,
        "applicable_fuels": [FuelType.NATURAL_GAS, FuelType.FUEL_OIL],
        "temperature_range_f": None,
        "ammonia_slip_ppm": 0,
        "catalyst_life_years": None,
        "requires_catalyst_regeneration": False,
        "energy_penalty_pct": 1.0,  # Fan power
        "achievable_emission_rates": {
            "natural_gas_ppm": 15,
            "fuel_oil_ppm": 60
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {},
        "environmental_impacts": []
    },
    "LNB_FGR": {
        "name": "Low NOx Burner + Flue Gas Recirculation",
        "description": "Combined LNB and FGR for enhanced control",
        "efficiency_range": (0.70, 0.85),
        "typical_efficiency": 0.80,
        "cost_per_ton_range": (1500, 4000),
        "typical_cost_per_ton": 2500,
        "capital_cost_per_mmbtu_hr": 5500,
        "operating_cost_pct_of_capital": 0.05,
        "applicable_fuels": [FuelType.NATURAL_GAS, FuelType.FUEL_OIL],
        "temperature_range_f": None,
        "ammonia_slip_ppm": 0,
        "catalyst_life_years": None,
        "requires_catalyst_regeneration": False,
        "energy_penalty_pct": 1.0,
        "achievable_emission_rates": {
            "natural_gas_ppm": 9,
            "fuel_oil_ppm": 40
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {},
        "environmental_impacts": []
    },
    "ULNB_SCR": {
        "name": "Ultra Low NOx Burner + SCR",
        "description": "Maximum NOx control combining ULNB with SCR",
        "efficiency_range": (0.95, 0.99),
        "typical_efficiency": 0.97,
        "cost_per_ton_range": (10000, 25000),
        "typical_cost_per_ton": 15000,
        "capital_cost_per_mmbtu_hr": 20000,
        "operating_cost_pct_of_capital": 0.08,
        "applicable_fuels": [FuelType.NATURAL_GAS],
        "temperature_range_f": (600, 750),
        "ammonia_slip_ppm": 5,
        "catalyst_life_years": 5,
        "requires_catalyst_regeneration": True,
        "energy_penalty_pct": 0.5,
        "achievable_emission_rates": {
            "natural_gas_ppm": 2
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {"mercury": 0.10},
        "environmental_impacts": ["ammonia_storage", "catalyst_disposal"]
    }
}

# SO2 Control Technologies
SO2_CONTROL_TECHNOLOGIES: Dict[str, Dict[str, Any]] = {
    "Wet_FGD": {
        "name": "Wet Flue Gas Desulfurization",
        "description": "Wet limestone scrubber for SO2 removal",
        "efficiency_range": (0.92, 0.98),
        "typical_efficiency": 0.95,
        "cost_per_ton_range": (400, 1200),
        "typical_cost_per_ton": 700,
        "capital_cost_per_mmbtu_hr": 25000,
        "operating_cost_pct_of_capital": 0.06,
        "applicable_fuels": [FuelType.COAL, FuelType.FUEL_OIL],
        "temperature_range_f": (120, 180),  # Outlet temperature
        "water_consumption_gal_mwh": 300,
        "reagent": "limestone",
        "byproduct": "gypsum",
        "achievable_emission_rates": {
            "coal_lb_mmbtu": 0.04,
            "fuel_oil_lb_mmbtu": 0.02
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {"HCl": 0.95, "HF": 0.95, "PM": 0.50, "mercury": 0.60},
        "environmental_impacts": ["wastewater", "solids_disposal"]
    },
    "Dry_FGD": {
        "name": "Dry Flue Gas Desulfurization (SDA)",
        "description": "Spray dryer absorber with lime slurry",
        "efficiency_range": (0.85, 0.93),
        "typical_efficiency": 0.90,
        "cost_per_ton_range": (500, 1400),
        "typical_cost_per_ton": 900,
        "capital_cost_per_mmbtu_hr": 20000,
        "operating_cost_pct_of_capital": 0.07,
        "applicable_fuels": [FuelType.COAL, FuelType.FUEL_OIL],
        "temperature_range_f": (275, 350),
        "water_consumption_gal_mwh": 100,
        "reagent": "lime",
        "byproduct": "calcium_sulfite_mixture",
        "achievable_emission_rates": {
            "coal_lb_mmbtu": 0.06,
            "fuel_oil_lb_mmbtu": 0.03
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {"HCl": 0.90, "HF": 0.90, "PM": 0.30},
        "environmental_impacts": ["solids_disposal"]
    },
    "DSI": {
        "name": "Dry Sorbent Injection",
        "description": "Injection of trona or sodium bicarbonate",
        "efficiency_range": (0.40, 0.60),
        "typical_efficiency": 0.50,
        "cost_per_ton_range": (200, 800),
        "typical_cost_per_ton": 500,
        "capital_cost_per_mmbtu_hr": 5000,
        "operating_cost_pct_of_capital": 0.15,
        "applicable_fuels": [FuelType.COAL, FuelType.FUEL_OIL, FuelType.NATURAL_GAS],
        "temperature_range_f": (300, 600),
        "reagent": "trona_or_sodium_bicarbonate",
        "byproduct": "sodium_sulfate_mixture",
        "achievable_emission_rates": {
            "coal_lb_mmbtu": 0.15,
            "fuel_oil_lb_mmbtu": 0.08
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {"HCl": 0.70, "SO3": 0.90},
        "environmental_impacts": ["increased_pm_loading"]
    },
    "CDS": {
        "name": "Circulating Dry Scrubber",
        "description": "Circulating fluidized bed with lime reagent",
        "efficiency_range": (0.90, 0.97),
        "typical_efficiency": 0.93,
        "cost_per_ton_range": (450, 1100),
        "typical_cost_per_ton": 750,
        "capital_cost_per_mmbtu_hr": 22000,
        "operating_cost_pct_of_capital": 0.06,
        "applicable_fuels": [FuelType.COAL, FuelType.FUEL_OIL],
        "temperature_range_f": (160, 200),
        "reagent": "hydrated_lime",
        "achievable_emission_rates": {
            "coal_lb_mmbtu": 0.05
        },
        "rblc_process_codes": ["11.110"],
        "co_benefits": {"HCl": 0.95, "SO3": 0.95},
        "environmental_impacts": ["solids_disposal"]
    },
    "Low_Sulfur_Fuel": {
        "name": "Low Sulfur Fuel Switching",
        "description": "Use of inherently low sulfur fuel",
        "efficiency_range": (0.80, 0.99),
        "typical_efficiency": 0.90,
        "cost_per_ton_range": (100, 500),
        "typical_cost_per_ton": 300,
        "capital_cost_per_mmbtu_hr": 0,
        "operating_cost_pct_of_capital": 0.0,
        "applicable_fuels": [FuelType.COAL, FuelType.FUEL_OIL],
        "achievable_emission_rates": {
            "coal_lb_mmbtu": 0.10,
            "fuel_oil_lb_mmbtu": 0.05
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {},
        "environmental_impacts": ["fuel_supply_constraints"]
    }
}

# PM Control Technologies
PM_CONTROL_TECHNOLOGIES: Dict[str, Dict[str, Any]] = {
    "ESP": {
        "name": "Electrostatic Precipitator",
        "description": "High voltage electrostatic particle collection",
        "efficiency_range": (0.97, 0.9995),
        "typical_efficiency": 0.99,
        "cost_per_ton_range": (1000, 4000),
        "typical_cost_per_ton": 2000,
        "capital_cost_per_mmbtu_hr": 18000,
        "operating_cost_pct_of_capital": 0.04,
        "applicable_fuels": [FuelType.COAL, FuelType.FUEL_OIL, FuelType.NATURAL_GAS],
        "sca_range": (300, 600),  # Specific collection area (ft2/1000 acfm)
        "pressure_drop_in_wc": 0.5,
        "power_consumption_kw_per_1000acfm": 0.5,
        "achievable_emission_rates": {
            "coal_lb_mmbtu": 0.012,
            "fuel_oil_lb_mmbtu": 0.01
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {"mercury": 0.30},  # Higher with SO3 conditioning
        "environmental_impacts": ["ash_disposal"]
    },
    "Fabric_Filter": {
        "name": "Fabric Filter (Baghouse)",
        "description": "Bag filtration for particulate removal",
        "efficiency_range": (0.99, 0.9999),
        "typical_efficiency": 0.995,
        "cost_per_ton_range": (1500, 5000),
        "typical_cost_per_ton": 3000,
        "capital_cost_per_mmbtu_hr": 22000,
        "operating_cost_pct_of_capital": 0.06,
        "applicable_fuels": [FuelType.COAL, FuelType.FUEL_OIL],
        "air_to_cloth_ratio": 4.0,  # acfm/ft2
        "pressure_drop_in_wc": 6.0,
        "bag_life_years": 4,
        "achievable_emission_rates": {
            "coal_lb_mmbtu": 0.010,
            "fuel_oil_lb_mmbtu": 0.008
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {"mercury": 0.40},  # With activated carbon injection
        "environmental_impacts": ["ash_disposal", "bag_replacement"]
    },
    "Wet_ESP": {
        "name": "Wet Electrostatic Precipitator",
        "description": "ESP with water spray for fine PM and SO3",
        "efficiency_range": (0.995, 0.9999),
        "typical_efficiency": 0.998,
        "cost_per_ton_range": (2500, 7000),
        "typical_cost_per_ton": 4500,
        "capital_cost_per_mmbtu_hr": 30000,
        "operating_cost_pct_of_capital": 0.08,
        "applicable_fuels": [FuelType.COAL, FuelType.FUEL_OIL],
        "pressure_drop_in_wc": 2.0,
        "water_consumption_gal_mwh": 50,
        "achievable_emission_rates": {
            "coal_lb_mmbtu": 0.008,
            "fuel_oil_lb_mmbtu": 0.005
        },
        "rblc_process_codes": ["11.110"],
        "co_benefits": {"SO3": 0.99, "mercury": 0.70},
        "environmental_impacts": ["wastewater", "ash_disposal"]
    },
    "Cyclone": {
        "name": "Mechanical Cyclone",
        "description": "Centrifugal separation for coarse particles",
        "efficiency_range": (0.70, 0.90),
        "typical_efficiency": 0.80,
        "cost_per_ton_range": (200, 800),
        "typical_cost_per_ton": 500,
        "capital_cost_per_mmbtu_hr": 3000,
        "operating_cost_pct_of_capital": 0.02,
        "applicable_fuels": [FuelType.COAL, FuelType.FUEL_OIL, FuelType.BIOMASS],
        "pressure_drop_in_wc": 4.0,
        "minimum_particle_size_micron": 10,
        "achievable_emission_rates": {
            "coal_lb_mmbtu": 0.10
        },
        "rblc_process_codes": ["11.110"],
        "co_benefits": {},
        "environmental_impacts": []
    },
    "Multi_Cyclone_FF": {
        "name": "Multi-Cyclone + Fabric Filter",
        "description": "Combined mechanical and filtration control",
        "efficiency_range": (0.995, 0.9999),
        "typical_efficiency": 0.997,
        "cost_per_ton_range": (2000, 5500),
        "typical_cost_per_ton": 3500,
        "capital_cost_per_mmbtu_hr": 25000,
        "operating_cost_pct_of_capital": 0.06,
        "applicable_fuels": [FuelType.COAL, FuelType.BIOMASS],
        "achievable_emission_rates": {
            "coal_lb_mmbtu": 0.008
        },
        "rblc_process_codes": ["11.110"],
        "co_benefits": {"mercury": 0.40},
        "environmental_impacts": ["ash_disposal", "bag_replacement"]
    }
}

# CO Control Technologies
CO_CONTROL_TECHNOLOGIES: Dict[str, Dict[str, Any]] = {
    "Oxidation_Catalyst": {
        "name": "Catalytic Oxidation",
        "description": "Precious metal catalyst for CO to CO2 conversion",
        "efficiency_range": (0.90, 0.98),
        "typical_efficiency": 0.95,
        "cost_per_ton_range": (2000, 6000),
        "typical_cost_per_ton": 4000,
        "capital_cost_per_mmbtu_hr": 8000,
        "operating_cost_pct_of_capital": 0.05,
        "applicable_fuels": [FuelType.NATURAL_GAS, FuelType.FUEL_OIL],
        "temperature_range_f": (400, 800),
        "catalyst_life_years": 7,
        "achievable_emission_rates": {
            "natural_gas_ppm": 10,
            "fuel_oil_ppm": 25
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {"VOC": 0.90, "formaldehyde": 0.95},
        "environmental_impacts": ["catalyst_disposal"]
    },
    "Good_Combustion_Practice": {
        "name": "Good Combustion Practice (GCP)",
        "description": "Optimized air-fuel ratio and burner maintenance",
        "efficiency_range": (0.50, 0.80),
        "typical_efficiency": 0.70,
        "cost_per_ton_range": (100, 500),
        "typical_cost_per_ton": 300,
        "capital_cost_per_mmbtu_hr": 500,
        "operating_cost_pct_of_capital": 0.10,
        "applicable_fuels": [FuelType.NATURAL_GAS, FuelType.FUEL_OIL, FuelType.COAL],
        "achievable_emission_rates": {
            "natural_gas_ppm": 50,
            "fuel_oil_ppm": 100
        },
        "rblc_process_codes": ["11.110", "11.310"],
        "co_benefits": {},
        "environmental_impacts": []
    }
}

# Consolidated control technology database
CONTROL_TECHNOLOGIES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "NOx": NOX_CONTROL_TECHNOLOGIES,
    "SO2": SO2_CONTROL_TECHNOLOGIES,
    "PM": PM_CONTROL_TECHNOLOGIES,
    "CO": CO_CONTROL_TECHNOLOGIES
}


# =============================================================================
# BACT Cost-Effectiveness Thresholds by Pollutant
# Reference: EPA guidance and precedent analyses
# =============================================================================

# Cost-effectiveness thresholds ($/ton removed) by pollutant
# These are typical upper bounds - actual thresholds vary by region/permit
COST_EFFECTIVENESS_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "NOx": {
        "lower": 5000,     # Generally cost-effective below this
        "upper": 15000,    # Generally not cost-effective above this
        "typical_bact": 10000  # Typical threshold for BACT
    },
    "SO2": {
        "lower": 500,
        "upper": 3000,
        "typical_bact": 1500
    },
    "PM": {
        "lower": 3000,
        "upper": 10000,
        "typical_bact": 6000
    },
    "PM10": {
        "lower": 4000,
        "upper": 12000,
        "typical_bact": 8000
    },
    "PM2.5": {
        "lower": 5000,
        "upper": 15000,
        "typical_bact": 10000
    },
    "CO": {
        "lower": 1000,
        "upper": 5000,
        "typical_bact": 3000
    },
    "VOC": {
        "lower": 2000,
        "upper": 8000,
        "typical_bact": 5000
    }
}


# =============================================================================
# Data Classes for Analysis Results
# =============================================================================

@dataclass
class TechnologyCosts:
    """Detailed cost breakdown for a control technology."""
    capital_cost: Decimal
    annual_operating_cost: Decimal
    annualized_capital_cost: Decimal
    total_annual_cost: Decimal
    cost_per_ton_removed: Decimal
    emissions_removed_tpy: Decimal
    interest_rate: float
    equipment_life_years: int
    calculation_method: str
    provenance_hash: str = field(default="", init=False)

    def __post_init__(self):
        self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = {
            "capital": str(self.capital_cost),
            "annual_cost": str(self.total_annual_cost),
            "cost_per_ton": str(self.cost_per_ton_removed)
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class FeasibilityResult:
    """Technical feasibility assessment result."""
    technology: str
    status: FeasibilityStatus
    applicable_to_fuel: bool
    temperature_compatible: bool
    space_available: bool
    utility_available: bool
    retrofit_feasible: bool
    reasons: List[str]
    constraints: List[str]
    provenance_hash: str = field(default="", init=False)

    def __post_init__(self):
        self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = {
            "technology": self.technology,
            "status": self.status.value,
            "reasons": self.reasons
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class CostEffectivenessResult:
    """Cost-effectiveness analysis result."""
    technology: str
    pollutant: str
    baseline_emissions_tpy: Decimal
    controlled_emissions_tpy: Decimal
    emissions_reduction_tpy: Decimal
    control_efficiency: float
    total_annual_cost: Decimal
    cost_per_ton_removed: Decimal
    cost_effectiveness_threshold: Decimal
    is_cost_effective: bool
    incremental_cost_per_ton: Optional[Decimal]
    ranking: TechnologyRanking
    calculation_steps: List[Dict[str, Any]]
    provenance_hash: str = field(default="", init=False)

    def __post_init__(self):
        self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = {
            "technology": self.technology,
            "pollutant": self.pollutant,
            "cost_per_ton": str(self.cost_per_ton_removed),
            "is_cost_effective": self.is_cost_effective
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class BACTDetermination:
    """Final BACT determination result."""
    pollutant: str
    selected_technology: str
    emission_limit: Decimal
    emission_limit_units: str
    control_efficiency: float
    cost_per_ton: Decimal
    annual_cost: Decimal
    ranked_technologies: List[Dict[str, Any]]
    eliminated_technologies: List[Dict[str, Any]]
    justification: str
    rblc_comparable_permits: List[str]
    monitoring_requirements: List[str]
    compliance_demonstration: str
    analysis_date: datetime
    provenance_hash: str = field(default="", init=False)

    def __post_init__(self):
        self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = {
            "pollutant": self.pollutant,
            "technology": self.selected_technology,
            "limit": str(self.emission_limit),
            "date": self.analysis_date.isoformat()
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class MultiPollutantResult:
    """Multi-pollutant co-benefit analysis result."""
    primary_pollutant: str
    primary_technology: str
    co_benefits: Dict[str, float]
    co_benefit_emissions_reduced_tpy: Dict[str, Decimal]
    co_benefit_value_usd: Decimal
    total_cost_with_co_benefits: Decimal
    adjusted_cost_per_ton: Decimal
    provenance_hash: str = field(default="", init=False)

    def __post_init__(self):
        self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = {
            "primary": self.primary_pollutant,
            "technology": self.primary_technology,
            "co_benefits": self.co_benefits
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


# =============================================================================
# BACT Analyzer Class
# =============================================================================

class BACTAnalyzer:
    """
    Comprehensive BACT (Best Available Control Technology) Analyzer.

    Implements EPA's Top-Down BACT methodology:
    1. Identify all available control technologies
    2. Eliminate technically infeasible options
    3. Rank by control effectiveness
    4. Evaluate economic, energy, environmental impacts
    5. Select BACT

    Supports RACT/BACT/LAER comparison based on attainment status.

    Example:
        >>> analyzer = BACTAnalyzer(
        ...     unit_capacity_mmbtu_hr=100.0,
        ...     fuel_type=FuelType.NATURAL_GAS,
        ...     operating_hours_yr=8760
        ... )
        >>> result = analyzer.perform_bact_analysis(
        ...     pollutant=PollutantType.NOX,
        ...     baseline_emissions_tpy=50.0
        ... )
        >>> print(f"Selected BACT: {result.selected_technology}")
    """

    def __init__(
        self,
        unit_capacity_mmbtu_hr: float,
        fuel_type: FuelType,
        operating_hours_yr: float = 8760,
        attainment_status: AttainmentStatus = AttainmentStatus.ATTAINMENT,
        interest_rate: float = 0.07,
        equipment_life_years: int = 20,
        cost_year: int = 2024,
        precision: int = 2
    ):
        """
        Initialize BACT analyzer.

        Args:
            unit_capacity_mmbtu_hr: Unit heat input capacity (MMBtu/hr)
            fuel_type: Primary fuel type
            operating_hours_yr: Annual operating hours
            attainment_status: NAAQS attainment status for the area
            interest_rate: Interest rate for cost annualization
            equipment_life_years: Expected equipment life
            cost_year: Year for cost basis (for escalation)
            precision: Decimal precision for calculations
        """
        self.unit_capacity = unit_capacity_mmbtu_hr
        self.fuel_type = fuel_type
        self.operating_hours = operating_hours_yr
        self.attainment_status = attainment_status
        self.interest_rate = interest_rate
        self.equipment_life = equipment_life_years
        self.cost_year = cost_year
        self.precision = precision
        self._quantize_str = '0.' + '0' * precision

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _compute_crf(self) -> float:
        """
        Compute Capital Recovery Factor for cost annualization.

        CRF = i(1+i)^n / ((1+i)^n - 1)

        Where:
            i = interest rate
            n = equipment life (years)
        """
        i = self.interest_rate
        n = self.equipment_life
        crf = (i * (1 + i) ** n) / ((1 + i) ** n - 1)
        return round(crf, 6)

    # =========================================================================
    # Step 1: Identify Available Control Technologies
    # =========================================================================

    def identify_control_technologies(
        self,
        pollutant: PollutantType
    ) -> List[Dict[str, Any]]:
        """
        Step 1 of Top-Down BACT: Identify all available control technologies.

        Reviews RBLC database, vendor information, and EPA guidance to compile
        comprehensive list of potentially applicable technologies.

        Args:
            pollutant: Pollutant for control technology identification

        Returns:
            List of technology dictionaries with full specifications
        """
        pollutant_key = pollutant.value
        if pollutant_key in ["PM10", "PM2.5"]:
            pollutant_key = "PM"

        if pollutant_key not in CONTROL_TECHNOLOGIES:
            return []

        technologies = []
        tech_db = CONTROL_TECHNOLOGIES[pollutant_key]

        for tech_id, tech_data in tech_db.items():
            tech_info = {
                "id": tech_id,
                "pollutant": pollutant.value,
                **tech_data,
                "step_1_identified": True,
                "identification_source": "RBLC/Vendor/EPA Guidance"
            }
            technologies.append(tech_info)

        return technologies

    # =========================================================================
    # Step 2: Eliminate Technically Infeasible Options
    # =========================================================================

    def evaluate_technical_feasibility(
        self,
        technology: Dict[str, Any],
        site_constraints: Optional[Dict[str, Any]] = None
    ) -> FeasibilityResult:
        """
        Step 2 of Top-Down BACT: Evaluate technical feasibility.

        Assesses whether a technology is technically feasible for the specific
        application considering fuel type, temperature, space, utilities, etc.

        Args:
            technology: Technology specification dictionary
            site_constraints: Site-specific constraints (optional)

        Returns:
            FeasibilityResult with detailed assessment
        """
        reasons = []
        constraints = []
        site_constraints = site_constraints or {}

        # Check fuel applicability
        applicable_fuels = technology.get("applicable_fuels", [])
        fuel_compatible = self.fuel_type in applicable_fuels

        if not fuel_compatible:
            reasons.append(
                f"Not applicable to {self.fuel_type.value} fuel. "
                f"Applicable fuels: {[f.value for f in applicable_fuels]}"
            )

        # Check temperature compatibility (if applicable)
        temp_range = technology.get("temperature_range_f")
        temp_compatible = True
        if temp_range:
            flue_temp = site_constraints.get("flue_gas_temperature_f", 500)
            if not (temp_range[0] <= flue_temp <= temp_range[1]):
                temp_compatible = False
                reasons.append(
                    f"Temperature incompatible. Required: {temp_range[0]}-{temp_range[1]}F, "
                    f"Available: {flue_temp}F"
                )

        # Check space availability
        space_available = site_constraints.get("space_available", True)
        if not space_available:
            reasons.append("Insufficient space for equipment installation")
            constraints.append("Space constraint - may require major modifications")

        # Check utility availability
        utility_available = True
        if technology.get("requires_catalyst_regeneration"):
            ammonia_available = site_constraints.get("ammonia_supply_available", True)
            if not ammonia_available:
                utility_available = False
                reasons.append("Ammonia/urea supply not available for reagent")

        if technology.get("water_consumption_gal_mwh", 0) > 0:
            water_available = site_constraints.get("process_water_available", True)
            if not water_available:
                utility_available = False
                reasons.append("Process water not available")

        # Check retrofit feasibility
        retrofit_feasible = site_constraints.get("retrofit_feasible", True)
        if not retrofit_feasible:
            constraints.append("Retrofit may require extended outage")

        # Determine overall feasibility
        if not fuel_compatible:
            status = FeasibilityStatus.INFEASIBLE
        elif not temp_compatible:
            status = FeasibilityStatus.INFEASIBLE
        elif not utility_available:
            status = FeasibilityStatus.INFEASIBLE
        elif constraints:
            status = FeasibilityStatus.CONDITIONALLY_FEASIBLE
        else:
            status = FeasibilityStatus.FEASIBLE
            reasons.append("Technically feasible for this application")

        return FeasibilityResult(
            technology=technology.get("id", "Unknown"),
            status=status,
            applicable_to_fuel=fuel_compatible,
            temperature_compatible=temp_compatible,
            space_available=space_available,
            utility_available=utility_available,
            retrofit_feasible=retrofit_feasible,
            reasons=reasons,
            constraints=constraints
        )

    def eliminate_infeasible_technologies(
        self,
        technologies: List[Dict[str, Any]],
        site_constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter technologies to remove technically infeasible options.

        Args:
            technologies: List of identified technologies
            site_constraints: Site-specific constraints

        Returns:
            Tuple of (feasible_technologies, eliminated_technologies)
        """
        feasible = []
        eliminated = []

        for tech in technologies:
            feasibility = self.evaluate_technical_feasibility(tech, site_constraints)

            if feasibility.status == FeasibilityStatus.FEASIBLE:
                tech["feasibility_assessment"] = feasibility
                tech["step_2_feasible"] = True
                feasible.append(tech)
            elif feasibility.status == FeasibilityStatus.CONDITIONALLY_FEASIBLE:
                tech["feasibility_assessment"] = feasibility
                tech["step_2_feasible"] = True
                tech["conditional_constraints"] = feasibility.constraints
                feasible.append(tech)
            else:
                tech["feasibility_assessment"] = feasibility
                tech["step_2_feasible"] = False
                tech["elimination_reason"] = feasibility.reasons
                eliminated.append(tech)

        return feasible, eliminated

    # =========================================================================
    # Step 3: Rank by Control Effectiveness
    # =========================================================================

    def rank_by_effectiveness(
        self,
        technologies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Step 3 of Top-Down BACT: Rank by control effectiveness.

        Ranks technically feasible technologies from most to least effective
        based on typical control efficiency.

        Args:
            technologies: List of feasible technologies

        Returns:
            List sorted by effectiveness (highest first)
        """
        def get_effectiveness(tech: Dict[str, Any]) -> float:
            return tech.get("typical_efficiency", 0)

        ranked = sorted(technologies, key=get_effectiveness, reverse=True)

        for i, tech in enumerate(ranked):
            tech["effectiveness_rank"] = i + 1
            tech["step_3_ranked"] = True

        return ranked

    # =========================================================================
    # Step 4: Economic, Energy, and Environmental Impact Analysis
    # =========================================================================

    def calculate_cost_effectiveness(
        self,
        technology: Dict[str, Any],
        baseline_emissions_tpy: float,
        previous_technology_cost: Optional[Decimal] = None,
        previous_technology_control: Optional[float] = None
    ) -> CostEffectivenessResult:
        """
        Step 4 of Top-Down BACT: Economic impact analysis.

        Calculates total annualized cost and cost-effectiveness ($/ton removed).
        Also calculates incremental cost-effectiveness for comparison with
        less stringent technologies.

        Args:
            technology: Technology specification
            baseline_emissions_tpy: Baseline (uncontrolled) emissions (tons/year)
            previous_technology_cost: Annual cost of previous technology in ranking
            previous_technology_control: Control efficiency of previous technology

        Returns:
            CostEffectivenessResult with detailed cost analysis
        """
        calculation_steps = []
        pollutant = technology.get("pollutant", "NOx")

        # Step 1: Calculate capital cost
        capital_per_mmbtu = technology.get("capital_cost_per_mmbtu_hr", 0)
        capital_cost = Decimal(str(capital_per_mmbtu * self.unit_capacity))
        calculation_steps.append({
            "step": 1,
            "description": "Calculate capital cost",
            "capital_per_mmbtu_hr": capital_per_mmbtu,
            "unit_capacity_mmbtu_hr": self.unit_capacity,
            "capital_cost": str(capital_cost)
        })

        # Step 2: Calculate annualized capital cost using CRF
        crf = self._compute_crf()
        annualized_capital = capital_cost * Decimal(str(crf))
        calculation_steps.append({
            "step": 2,
            "description": "Annualize capital cost",
            "capital_recovery_factor": crf,
            "interest_rate": self.interest_rate,
            "equipment_life_years": self.equipment_life,
            "annualized_capital": str(annualized_capital)
        })

        # Step 3: Calculate annual operating cost
        o_and_m_pct = technology.get("operating_cost_pct_of_capital", 0.05)
        annual_operating = capital_cost * Decimal(str(o_and_m_pct))
        calculation_steps.append({
            "step": 3,
            "description": "Calculate operating cost",
            "o_and_m_percent": o_and_m_pct,
            "annual_operating_cost": str(annual_operating)
        })

        # Step 4: Total annual cost
        total_annual_cost = self._quantize(annualized_capital + annual_operating)
        calculation_steps.append({
            "step": 4,
            "description": "Total annual cost",
            "total_annual_cost": str(total_annual_cost)
        })

        # Step 5: Calculate emissions reduction
        efficiency = technology.get("typical_efficiency", 0)
        baseline_decimal = Decimal(str(baseline_emissions_tpy))
        emissions_reduction = baseline_decimal * Decimal(str(efficiency))
        controlled_emissions = baseline_decimal - emissions_reduction
        calculation_steps.append({
            "step": 5,
            "description": "Calculate emissions reduction",
            "baseline_emissions_tpy": str(baseline_decimal),
            "control_efficiency": efficiency,
            "emissions_reduction_tpy": str(emissions_reduction),
            "controlled_emissions_tpy": str(controlled_emissions)
        })

        # Step 6: Calculate cost-effectiveness
        if emissions_reduction > 0:
            cost_per_ton = self._quantize(total_annual_cost / emissions_reduction)
        else:
            cost_per_ton = Decimal("999999")
        calculation_steps.append({
            "step": 6,
            "description": "Calculate cost-effectiveness",
            "cost_per_ton_removed": str(cost_per_ton)
        })

        # Step 7: Calculate incremental cost-effectiveness (if applicable)
        incremental_cost_per_ton = None
        if previous_technology_cost is not None and previous_technology_control is not None:
            incremental_cost = total_annual_cost - previous_technology_cost
            incremental_reduction = emissions_reduction - (
                baseline_decimal * Decimal(str(previous_technology_control))
            )
            if incremental_reduction > 0:
                incremental_cost_per_ton = self._quantize(
                    incremental_cost / incremental_reduction
                )
            calculation_steps.append({
                "step": 7,
                "description": "Calculate incremental cost-effectiveness",
                "incremental_cost": str(incremental_cost),
                "incremental_reduction": str(incremental_reduction),
                "incremental_cost_per_ton": str(incremental_cost_per_ton) if incremental_cost_per_ton else "N/A"
            })

        # Determine cost-effectiveness status
        thresholds = COST_EFFECTIVENESS_THRESHOLDS.get(
            pollutant, {"typical_bact": 10000}
        )
        threshold = Decimal(str(thresholds.get("typical_bact", 10000)))
        is_cost_effective = cost_per_ton <= threshold

        # Determine ranking
        if is_cost_effective:
            ranking = TechnologyRanking.CONSIDERED
        else:
            ranking = TechnologyRanking.ELIMINATED_COST

        return CostEffectivenessResult(
            technology=technology.get("id", "Unknown"),
            pollutant=pollutant,
            baseline_emissions_tpy=baseline_decimal,
            controlled_emissions_tpy=self._quantize(controlled_emissions),
            emissions_reduction_tpy=self._quantize(emissions_reduction),
            control_efficiency=efficiency,
            total_annual_cost=total_annual_cost,
            cost_per_ton_removed=cost_per_ton,
            cost_effectiveness_threshold=threshold,
            is_cost_effective=is_cost_effective,
            incremental_cost_per_ton=incremental_cost_per_ton,
            ranking=ranking,
            calculation_steps=calculation_steps
        )

    def evaluate_energy_impacts(
        self,
        technology: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate energy impacts of control technology.

        Args:
            technology: Technology specification

        Returns:
            Dictionary with energy impact assessment
        """
        energy_penalty_pct = technology.get("energy_penalty_pct", 0)
        pressure_drop = technology.get("pressure_drop_in_wc", 0)
        power_consumption = technology.get("power_consumption_kw_per_1000acfm", 0)

        # Calculate fan power for pressure drop
        # Approximate: 1 in WC = 0.1 kW per 1000 acfm
        fan_power_estimate = pressure_drop * 0.1

        total_energy_impact = energy_penalty_pct + (fan_power_estimate * 0.1)

        is_acceptable = total_energy_impact < 5.0  # 5% threshold

        return {
            "technology": technology.get("id"),
            "energy_penalty_pct": energy_penalty_pct,
            "pressure_drop_in_wc": pressure_drop,
            "fan_power_estimate_kw_per_1000acfm": fan_power_estimate,
            "total_energy_impact_pct": total_energy_impact,
            "is_acceptable": is_acceptable,
            "assessment": "Acceptable" if is_acceptable else "May need justification"
        }

    def evaluate_environmental_impacts(
        self,
        technology: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate environmental impacts (collateral emissions, waste).

        Args:
            technology: Technology specification

        Returns:
            Dictionary with environmental impact assessment
        """
        impacts = technology.get("environmental_impacts", [])
        ammonia_slip = technology.get("ammonia_slip_ppm", 0)
        co_benefits = technology.get("co_benefits", {})

        # Assess impacts
        impact_assessment = []
        for impact in impacts:
            if impact == "ammonia_storage":
                impact_assessment.append({
                    "impact": "Ammonia storage",
                    "concern": "Safety hazard, requires RMP compliance",
                    "mitigation": "Proper storage, safety systems"
                })
            elif impact == "catalyst_disposal":
                impact_assessment.append({
                    "impact": "Spent catalyst disposal",
                    "concern": "Hazardous waste (may contain heavy metals)",
                    "mitigation": "Proper characterization and disposal"
                })
            elif impact == "wastewater":
                impact_assessment.append({
                    "impact": "Wastewater generation",
                    "concern": "Requires treatment and discharge permit",
                    "mitigation": "Wastewater treatment system"
                })
            elif impact == "solids_disposal":
                impact_assessment.append({
                    "impact": "Solid waste generation",
                    "concern": "Landfill requirements",
                    "mitigation": "Beneficial reuse or proper disposal"
                })
            elif impact == "N2O_formation":
                impact_assessment.append({
                    "impact": "N2O formation",
                    "concern": "Potent greenhouse gas",
                    "mitigation": "Temperature control, proper reagent injection"
                })

        # Assess co-benefits
        co_benefit_assessment = []
        for pollutant, removal in co_benefits.items():
            co_benefit_assessment.append({
                "pollutant": pollutant,
                "co_removal_efficiency": removal,
                "benefit": f"Reduces {pollutant} emissions by {removal*100:.0f}%"
            })

        net_environmental_impact = len(impact_assessment) - len(co_benefit_assessment) * 2
        is_acceptable = net_environmental_impact <= 0

        return {
            "technology": technology.get("id"),
            "environmental_impacts": impact_assessment,
            "co_benefits": co_benefit_assessment,
            "ammonia_slip_ppm": ammonia_slip,
            "net_impact_score": net_environmental_impact,
            "is_acceptable": is_acceptable,
            "assessment": "Acceptable" if is_acceptable else "May need additional justification"
        }

    # =========================================================================
    # Step 5: Select BACT
    # =========================================================================

    def select_bact(
        self,
        ranked_technologies: List[Dict[str, Any]],
        cost_analyses: List[CostEffectivenessResult],
        pollutant: PollutantType
    ) -> BACTDetermination:
        """
        Step 5 of Top-Down BACT: Select BACT.

        Reviews ranked technologies starting from most effective and selects
        the most stringent option that is cost-effective.

        Args:
            ranked_technologies: Technologies ranked by effectiveness
            cost_analyses: Cost-effectiveness results for each technology
            pollutant: Pollutant for BACT determination

        Returns:
            BACTDetermination with selected technology and justification
        """
        # Create mapping of technology to cost analysis
        cost_map = {ca.technology: ca for ca in cost_analyses}

        selected_tech = None
        selected_cost = None
        eliminated = []

        # Review from most to least effective
        for tech in ranked_technologies:
            tech_id = tech.get("id")
            cost_analysis = cost_map.get(tech_id)

            if cost_analysis is None:
                continue

            # Evaluate energy impacts
            energy_eval = self.evaluate_energy_impacts(tech)
            # Evaluate environmental impacts
            env_eval = self.evaluate_environmental_impacts(tech)

            if cost_analysis.is_cost_effective:
                if energy_eval["is_acceptable"] and env_eval["is_acceptable"]:
                    selected_tech = tech
                    selected_cost = cost_analysis
                    break
                else:
                    # Eliminated due to energy or environmental
                    eliminated.append({
                        "technology": tech_id,
                        "reason": "energy_or_environmental_impacts",
                        "energy_evaluation": energy_eval,
                        "environmental_evaluation": env_eval
                    })
            else:
                # Eliminated due to cost
                eliminated.append({
                    "technology": tech_id,
                    "reason": "cost_ineffective",
                    "cost_per_ton": str(cost_analysis.cost_per_ton_removed),
                    "threshold": str(cost_analysis.cost_effectiveness_threshold)
                })

        if selected_tech is None:
            # If no technology meets criteria, select least costly option
            # or Good Combustion Practice as baseline
            for tech in reversed(ranked_technologies):
                tech_id = tech.get("id")
                if cost_map.get(tech_id):
                    selected_tech = tech
                    selected_cost = cost_map[tech_id]
                    break

        # Determine emission limit based on selected technology
        if selected_tech:
            achievable_rates = selected_tech.get("achievable_emission_rates", {})
            fuel_key = f"{self.fuel_type.value}_ppm"
            alt_fuel_key = f"{self.fuel_type.value}_lb_mmbtu"

            if fuel_key in achievable_rates:
                emission_limit = Decimal(str(achievable_rates[fuel_key]))
                emission_units = "ppmvd @ 3% O2"
            elif alt_fuel_key in achievable_rates:
                emission_limit = Decimal(str(achievable_rates[alt_fuel_key]))
                emission_units = "lb/MMBtu"
            else:
                # Use first available
                for key, value in achievable_rates.items():
                    emission_limit = Decimal(str(value))
                    if "ppm" in key:
                        emission_units = "ppmvd @ 3% O2"
                    else:
                        emission_units = "lb/MMBtu"
                    break
                else:
                    emission_limit = Decimal("0")
                    emission_units = "TBD"
        else:
            emission_limit = Decimal("0")
            emission_units = "TBD"

        # Generate justification
        justification = self._generate_bact_justification(
            selected_tech, selected_cost, eliminated, pollutant
        )

        # Monitoring requirements
        monitoring = self._determine_monitoring_requirements(
            selected_tech, pollutant
        )

        # Compliance demonstration
        compliance_demo = self._generate_compliance_demonstration(
            selected_tech, emission_limit, emission_units
        )

        # Prepare ranked technologies summary
        ranked_summary = []
        for i, tech in enumerate(ranked_technologies):
            tech_id = tech.get("id")
            cost_analysis = cost_map.get(tech_id)
            ranked_summary.append({
                "rank": i + 1,
                "technology": tech_id,
                "efficiency": tech.get("typical_efficiency"),
                "cost_per_ton": str(cost_analysis.cost_per_ton_removed) if cost_analysis else "N/A",
                "is_selected": tech_id == selected_tech.get("id") if selected_tech else False
            })

        return BACTDetermination(
            pollutant=pollutant.value,
            selected_technology=selected_tech.get("id") if selected_tech else "None",
            emission_limit=self._quantize(emission_limit),
            emission_limit_units=emission_units,
            control_efficiency=selected_tech.get("typical_efficiency", 0) if selected_tech else 0,
            cost_per_ton=selected_cost.cost_per_ton_removed if selected_cost else Decimal("0"),
            annual_cost=selected_cost.total_annual_cost if selected_cost else Decimal("0"),
            ranked_technologies=ranked_summary,
            eliminated_technologies=eliminated,
            justification=justification,
            rblc_comparable_permits=self._get_rblc_comparables(pollutant, selected_tech),
            monitoring_requirements=monitoring,
            compliance_demonstration=compliance_demo,
            analysis_date=datetime.utcnow()
        )

    def _generate_bact_justification(
        self,
        selected_tech: Optional[Dict[str, Any]],
        selected_cost: Optional[CostEffectivenessResult],
        eliminated: List[Dict[str, Any]],
        pollutant: PollutantType
    ) -> str:
        """Generate BACT justification narrative."""
        if selected_tech is None:
            return "No feasible control technology identified."

        tech_name = selected_tech.get("name", selected_tech.get("id"))

        justification = f"""
BACT DETERMINATION JUSTIFICATION FOR {pollutant.value}

SELECTED TECHNOLOGY: {tech_name}

1. IDENTIFICATION OF CONTROL OPTIONS (Step 1):
   All available control technologies were identified through review of:
   - EPA RACT/BACT/LAER Clearinghouse (RBLC)
   - Vendor information and performance data
   - EPA guidance documents and AP-42

2. TECHNICAL FEASIBILITY (Step 2):
   {tech_name} is technically feasible for this {self.fuel_type.value}-fired unit.
   Fuel type: {self.fuel_type.value}
   Unit capacity: {self.unit_capacity} MMBtu/hr
   Operating hours: {self.operating_hours} hours/year

3. RANKING BY EFFECTIVENESS (Step 3):
   {tech_name} achieves {selected_tech.get('typical_efficiency', 0) * 100:.0f}% control efficiency,
   ranking among the most effective options for this pollutant.

4. ECONOMIC/ENERGY/ENVIRONMENTAL ANALYSIS (Step 4):
   - Annual cost: ${selected_cost.total_annual_cost if selected_cost else 0:,.0f}
   - Cost-effectiveness: ${selected_cost.cost_per_ton_removed if selected_cost else 0:,.0f}/ton removed
   - Energy penalty: {selected_tech.get('energy_penalty_pct', 0)}%
   - Environmental impacts: Manageable with proper design

5. BACT SELECTION (Step 5):
   {tech_name} is selected as BACT based on:
   - Superior control efficiency
   - Cost-effective per EPA guidance
   - Acceptable energy and environmental impacts
   - Demonstrated in practice on similar sources

ELIMINATED TECHNOLOGIES:
"""
        for elim in eliminated:
            justification += f"   - {elim['technology']}: {elim['reason']}\n"

        return justification

    def _determine_monitoring_requirements(
        self,
        tech: Optional[Dict[str, Any]],
        pollutant: PollutantType
    ) -> List[str]:
        """Determine monitoring requirements for BACT."""
        requirements = []

        if pollutant == PollutantType.NOX:
            requirements.extend([
                "Continuous Emission Monitoring System (CEMS) for NOx",
                "CEMS for O2 or CO2 (diluent)",
                "Quarterly RATA testing per 40 CFR Part 60",
                "Daily calibration drift checks"
            ])
            if tech and "SCR" in tech.get("id", ""):
                requirements.extend([
                    "Ammonia slip monitoring (quarterly)",
                    "Catalyst activity monitoring (annual)",
                    "Reagent flow monitoring (continuous)"
                ])
        elif pollutant == PollutantType.SO2:
            requirements.extend([
                "Continuous Emission Monitoring System (CEMS) for SO2",
                "Fuel sulfur content monitoring",
                "Quarterly stack testing"
            ])
        elif pollutant in [PollutantType.PM, PollutantType.PM10, PollutantType.PM25]:
            requirements.extend([
                "Quarterly stack testing (EPA Method 5/201A)",
                "Opacity monitoring (continuous or periodic)",
                "Baghouse pressure drop monitoring (if applicable)"
            ])
        elif pollutant == PollutantType.CO:
            requirements.extend([
                "Continuous Emission Monitoring System (CEMS) for CO",
                "Combustion optimization monitoring",
                "Quarterly calibration checks"
            ])

        requirements.append("Recordkeeping of all monitoring data for 5 years minimum")

        return requirements

    def _generate_compliance_demonstration(
        self,
        tech: Optional[Dict[str, Any]],
        limit: Decimal,
        units: str
    ) -> str:
        """Generate compliance demonstration requirements."""
        if tech is None:
            return "Compliance demonstration method to be determined."

        demo = f"""
COMPLIANCE DEMONSTRATION

Emission Limit: {limit} {units}

Initial Compliance:
- Performance test within 180 days of startup
- Three 1-hour test runs minimum
- Use EPA reference methods

Continuous Compliance:
- 30-day rolling average for CEMS-monitored parameters
- Annual performance testing
- Operating parameter monitoring

Reporting:
- Quarterly excess emissions reports
- Annual compliance certification
- Prompt deviation reporting (within 2 days)
"""
        return demo

    def _get_rblc_comparables(
        self,
        pollutant: PollutantType,
        tech: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Get comparable RBLC entries for reference."""
        # These would typically come from RBLC database query
        # Returning example entries for illustration
        return [
            f"RBLC ID: XX-0001 - Similar {self.fuel_type.value} unit, {tech.get('id') if tech else 'N/A'}",
            f"RBLC ID: XX-0002 - Comparable capacity, same technology",
            f"RBLC ID: XX-0003 - Recent permit, {pollutant.value} BACT determination"
        ]

    # =========================================================================
    # Complete BACT Analysis
    # =========================================================================

    def perform_bact_analysis(
        self,
        pollutant: PollutantType,
        baseline_emissions_tpy: float,
        site_constraints: Optional[Dict[str, Any]] = None
    ) -> BACTDetermination:
        """
        Perform complete Top-Down BACT analysis.

        Executes all 5 steps of the BACT analysis methodology:
        1. Identify available technologies
        2. Eliminate infeasible options
        3. Rank by effectiveness
        4. Economic/energy/environmental analysis
        5. Select BACT

        Args:
            pollutant: Pollutant for BACT determination
            baseline_emissions_tpy: Uncontrolled emissions (tons/year)
            site_constraints: Site-specific constraints

        Returns:
            BACTDetermination with complete analysis
        """
        # Step 1: Identify technologies
        all_technologies = self.identify_control_technologies(pollutant)

        # Step 2: Eliminate infeasible
        feasible, eliminated = self.eliminate_infeasible_technologies(
            all_technologies, site_constraints
        )

        # Step 3: Rank by effectiveness
        ranked = self.rank_by_effectiveness(feasible)

        # Step 4: Cost-effectiveness analysis
        cost_analyses = []
        prev_cost = None
        prev_efficiency = None

        for tech in ranked:
            cost_result = self.calculate_cost_effectiveness(
                technology=tech,
                baseline_emissions_tpy=baseline_emissions_tpy,
                previous_technology_cost=prev_cost,
                previous_technology_control=prev_efficiency
            )
            cost_analyses.append(cost_result)
            prev_cost = cost_result.total_annual_cost
            prev_efficiency = cost_result.control_efficiency

        # Step 5: Select BACT
        determination = self.select_bact(ranked, cost_analyses, pollutant)

        return determination

    # =========================================================================
    # RACT/LAER Comparison
    # =========================================================================

    def compare_ract_bact_laer(
        self,
        pollutant: PollutantType,
        baseline_emissions_tpy: float
    ) -> Dict[str, Any]:
        """
        Compare RACT, BACT, and LAER requirements.

        - RACT: Reasonably Available Control Technology (existing sources, SIP)
        - BACT: Best Available Control Technology (PSD, attainment areas)
        - LAER: Lowest Achievable Emission Rate (NSR, nonattainment areas)

        Args:
            pollutant: Pollutant for comparison
            baseline_emissions_tpy: Baseline emissions

        Returns:
            Dictionary with comparative analysis
        """
        # Perform BACT analysis
        bact_result = self.perform_bact_analysis(pollutant, baseline_emissions_tpy)

        # LAER: Most stringent technology regardless of cost
        all_tech = self.identify_control_technologies(pollutant)
        feasible, _ = self.eliminate_infeasible_technologies(all_tech)
        ranked = self.rank_by_effectiveness(feasible)

        if ranked:
            laer_tech = ranked[0]  # Most effective
            laer_limit = None
            achievable = laer_tech.get("achievable_emission_rates", {})
            for key, value in achievable.items():
                laer_limit = value
                break
        else:
            laer_tech = None
            laer_limit = None

        # RACT: Consider cost more heavily, less stringent
        ract_tech = None
        for tech in reversed(ranked):
            cost_result = self.calculate_cost_effectiveness(
                tech, baseline_emissions_tpy
            )
            # RACT typically lower cost threshold
            if cost_result.cost_per_ton_removed < Decimal("5000"):
                ract_tech = tech
                break

        if ract_tech is None and ranked:
            ract_tech = ranked[-1]  # Least stringent if all expensive

        return {
            "pollutant": pollutant.value,
            "baseline_emissions_tpy": baseline_emissions_tpy,
            "ract": {
                "technology": ract_tech.get("id") if ract_tech else "None",
                "typical_efficiency": ract_tech.get("typical_efficiency") if ract_tech else 0,
                "description": "RACT - Reasonably Available Control Technology",
                "applicability": "Existing sources in SIP areas"
            },
            "bact": {
                "technology": bact_result.selected_technology,
                "emission_limit": str(bact_result.emission_limit),
                "units": bact_result.emission_limit_units,
                "efficiency": bact_result.control_efficiency,
                "cost_per_ton": str(bact_result.cost_per_ton),
                "description": "BACT - Best Available Control Technology",
                "applicability": "New/modified major sources in attainment areas (PSD)"
            },
            "laer": {
                "technology": laer_tech.get("id") if laer_tech else "None",
                "emission_limit": laer_limit,
                "efficiency": laer_tech.get("typical_efficiency") if laer_tech else 0,
                "description": "LAER - Lowest Achievable Emission Rate",
                "applicability": "New/modified major sources in nonattainment areas"
            },
            "recommendation": self._get_control_recommendation()
        }

    def _get_control_recommendation(self) -> str:
        """Get recommendation based on attainment status."""
        if self.attainment_status == AttainmentStatus.ATTAINMENT:
            return (
                "Area is in attainment - BACT analysis required under PSD. "
                "Cost-effectiveness is a valid consideration."
            )
        elif self.attainment_status == AttainmentStatus.NONATTAINMENT:
            return (
                "Area is in nonattainment - LAER required. "
                "Cost is NOT a valid consideration; must achieve lowest rate."
            )
        else:
            return (
                "Area is unclassifiable - treated as attainment for PSD purposes. "
                "BACT analysis required."
            )

    # =========================================================================
    # Multi-Pollutant Co-Benefits Analysis
    # =========================================================================

    def analyze_multi_pollutant_cobenefits(
        self,
        primary_pollutant: PollutantType,
        primary_technology: str,
        baseline_emissions: Dict[str, float]
    ) -> MultiPollutantResult:
        """
        Analyze multi-pollutant co-benefits of a control technology.

        Some technologies provide co-benefits by removing multiple pollutants,
        which can improve overall cost-effectiveness.

        Args:
            primary_pollutant: Primary pollutant being controlled
            primary_technology: Selected control technology
            baseline_emissions: Dict of pollutant -> baseline TPY

        Returns:
            MultiPollutantResult with co-benefit analysis
        """
        # Get technology data
        pollutant_key = primary_pollutant.value
        if pollutant_key in ["PM10", "PM2.5"]:
            pollutant_key = "PM"

        tech_db = CONTROL_TECHNOLOGIES.get(pollutant_key, {})
        tech_data = tech_db.get(primary_technology, {})

        co_benefits = tech_data.get("co_benefits", {})

        # Calculate co-benefit reductions
        co_benefit_reductions = {}
        co_benefit_value = Decimal("0")

        # Value per ton for different pollutants (example values)
        pollutant_values = {
            "mercury": 50000,  # $/lb (high value)
            "HCl": 5000,
            "HF": 5000,
            "dioxins": 100000,  # Very high value
            "SO3": 3000,
            "VOC": 2000,
            "formaldehyde": 5000
        }

        for pollutant, removal_eff in co_benefits.items():
            if pollutant in baseline_emissions:
                reduction = Decimal(str(
                    baseline_emissions[pollutant] * removal_eff
                ))
                co_benefit_reductions[pollutant] = reduction

                # Calculate value
                value_per_ton = pollutant_values.get(pollutant, 1000)
                co_benefit_value += reduction * Decimal(str(value_per_ton))

        # Calculate primary technology cost
        all_tech = self.identify_control_technologies(primary_pollutant)
        tech_dict = next(
            (t for t in all_tech if t.get("id") == primary_technology),
            None
        )

        if tech_dict:
            primary_baseline = baseline_emissions.get(primary_pollutant.value, 0)
            cost_result = self.calculate_cost_effectiveness(
                tech_dict, primary_baseline
            )
            total_cost = cost_result.total_annual_cost
            primary_reduction = cost_result.emissions_reduction_tpy
        else:
            total_cost = Decimal("0")
            primary_reduction = Decimal("0")

        # Adjust cost-effectiveness considering co-benefits
        adjusted_cost = total_cost - co_benefit_value
        if primary_reduction > 0:
            adjusted_cost_per_ton = self._quantize(adjusted_cost / primary_reduction)
        else:
            adjusted_cost_per_ton = Decimal("0")

        return MultiPollutantResult(
            primary_pollutant=primary_pollutant.value,
            primary_technology=primary_technology,
            co_benefits=co_benefits,
            co_benefit_emissions_reduced_tpy={
                k: self._quantize(v) for k, v in co_benefit_reductions.items()
            },
            co_benefit_value_usd=self._quantize(co_benefit_value),
            total_cost_with_co_benefits=self._quantize(adjusted_cost),
            adjusted_cost_per_ton=adjusted_cost_per_ton
        )

    # =========================================================================
    # Technology Combination Analysis
    # =========================================================================

    def analyze_technology_combinations(
        self,
        pollutant: PollutantType,
        baseline_emissions_tpy: float
    ) -> List[Dict[str, Any]]:
        """
        Analyze combinations of control technologies.

        Some applications benefit from combining technologies
        (e.g., LNB + FGR, or ULNB + SCR).

        Args:
            pollutant: Pollutant for analysis
            baseline_emissions_tpy: Baseline emissions

        Returns:
            List of technology combination analyses
        """
        combinations = []

        if pollutant == PollutantType.NOX:
            # Define potential combinations
            combo_specs = [
                ("Low_NOx_Burner", "FGR", "LNB_FGR"),
                ("Ultra_Low_NOx_Burner", "SCR", "ULNB_SCR"),
                ("Low_NOx_Burner", "SNCR", None),
                ("FGR", "SNCR", None)
            ]

            for tech1_id, tech2_id, combined_id in combo_specs:
                # Get individual technologies
                all_tech = self.identify_control_technologies(pollutant)
                tech1 = next((t for t in all_tech if t.get("id") == tech1_id), None)
                tech2 = next((t for t in all_tech if t.get("id") == tech2_id), None)

                if tech1 is None or tech2 is None:
                    continue

                # Check if pre-defined combination exists
                if combined_id:
                    combined_tech = next(
                        (t for t in all_tech if t.get("id") == combined_id),
                        None
                    )
                    if combined_tech:
                        cost_result = self.calculate_cost_effectiveness(
                            combined_tech, baseline_emissions_tpy
                        )
                        combinations.append({
                            "combination": f"{tech1_id} + {tech2_id}",
                            "combined_technology": combined_id,
                            "combined_efficiency": combined_tech.get("typical_efficiency"),
                            "total_annual_cost": str(cost_result.total_annual_cost),
                            "cost_per_ton": str(cost_result.cost_per_ton_removed),
                            "is_cost_effective": cost_result.is_cost_effective,
                            "synergy": "Yes - dedicated combined technology available"
                        })
                else:
                    # Calculate combined efficiency (not simply additive)
                    eff1 = tech1.get("typical_efficiency", 0)
                    eff2 = tech2.get("typical_efficiency", 0)
                    # Combined: 1 - (1-eff1)*(1-eff2)
                    combined_eff = 1 - (1 - eff1) * (1 - eff2)

                    # Estimate combined cost (sum of individual)
                    cost1 = self.calculate_cost_effectiveness(tech1, baseline_emissions_tpy)
                    # For second technology, baseline is reduced by first
                    remaining = baseline_emissions_tpy * (1 - eff1)
                    cost2 = self.calculate_cost_effectiveness(tech2, remaining)

                    total_cost = cost1.total_annual_cost + cost2.total_annual_cost
                    combined_reduction = Decimal(str(baseline_emissions_tpy * combined_eff))
                    cost_per_ton = self._quantize(total_cost / combined_reduction) if combined_reduction > 0 else Decimal("999999")

                    threshold = COST_EFFECTIVENESS_THRESHOLDS.get(
                        pollutant.value, {}
                    ).get("typical_bact", 10000)

                    combinations.append({
                        "combination": f"{tech1_id} + {tech2_id}",
                        "combined_efficiency": round(combined_eff, 3),
                        "tech1_efficiency": eff1,
                        "tech2_efficiency": eff2,
                        "total_annual_cost": str(total_cost),
                        "cost_per_ton": str(cost_per_ton),
                        "is_cost_effective": cost_per_ton <= Decimal(str(threshold)),
                        "synergy": "Combined efficiency calculated"
                    })

        return combinations

    # =========================================================================
    # Documentation Generation
    # =========================================================================

    def generate_bact_report(
        self,
        determination: BACTDetermination,
        include_cost_details: bool = True,
        include_rblc_format: bool = True
    ) -> str:
        """
        Generate comprehensive BACT analysis report.

        Args:
            determination: BACT determination result
            include_cost_details: Include detailed cost breakdown
            include_rblc_format: Include RBLC-format summary

        Returns:
            Formatted report string
        """
        report = f"""
================================================================================
                    BACT ANALYSIS REPORT
================================================================================

FACILITY INFORMATION
--------------------
Unit Capacity: {self.unit_capacity} MMBtu/hr
Fuel Type: {self.fuel_type.value}
Operating Hours: {self.operating_hours} hours/year
Attainment Status: {self.attainment_status.value}
Analysis Date: {determination.analysis_date.strftime('%Y-%m-%d %H:%M UTC')}
Provenance Hash: {determination.provenance_hash}

================================================================================
                    POLLUTANT: {determination.pollutant}
================================================================================

BACT DETERMINATION
------------------
Selected Technology: {determination.selected_technology}
Emission Limit: {determination.emission_limit} {determination.emission_limit_units}
Control Efficiency: {determination.control_efficiency * 100:.1f}%
Cost-Effectiveness: ${determination.cost_per_ton:,.0f}/ton
Annual Cost: ${determination.annual_cost:,.0f}

TECHNOLOGY RANKING
------------------
"""
        for tech in determination.ranked_technologies:
            selected_marker = " [SELECTED]" if tech.get("is_selected") else ""
            report += f"  {tech['rank']}. {tech['technology']} - {tech.get('efficiency', 0) * 100:.0f}% efficiency, ${tech.get('cost_per_ton', 'N/A')}/ton{selected_marker}\n"

        report += """
ELIMINATED TECHNOLOGIES
-----------------------
"""
        for elim in determination.eliminated_technologies:
            report += f"  - {elim['technology']}: {elim['reason']}\n"

        report += f"""
{determination.justification}

MONITORING REQUIREMENTS
-----------------------
"""
        for req in determination.monitoring_requirements:
            report += f"  - {req}\n"

        report += f"""
{determination.compliance_demonstration}

COMPARABLE RBLC ENTRIES
-----------------------
"""
        for entry in determination.rblc_comparable_permits:
            report += f"  - {entry}\n"

        if include_rblc_format:
            report += self._generate_rblc_format(determination)

        return report

    def _generate_rblc_format(
        self,
        determination: BACTDetermination
    ) -> str:
        """Generate RBLC-format entry."""
        return f"""
================================================================================
                    RBLC FORMAT ENTRY
================================================================================

RBLC Entry Number: [To be assigned]
State: [State code]
Facility Name: [Facility name]
Permit Number: [Permit number]
SIC Code: 4911 (Electric Services) / [Applicable SIC]
NAICS Code: 221112 (Fossil Fuel Electric Power Generation) / [Applicable NAICS]

Process Description: {self.fuel_type.value.upper()}-fired combustion unit,
                     {self.unit_capacity} MMBtu/hr capacity

Pollutant: {determination.pollutant}
Control Technology: {determination.selected_technology}
Control Efficiency: {determination.control_efficiency * 100:.1f}%
Emission Limit: {determination.emission_limit} {determination.emission_limit_units}
Averaging Period: [1-hour / 30-day rolling / Annual]

Throughput/Capacity: {self.unit_capacity} MMBtu/hr
Fuel Type: {self.fuel_type.value}

Basis of Limit: BACT
Case-by-Case Basis: Yes
Emission Standard Reference: [40 CFR XX.XX or State rule]

Cost-Effectiveness: ${determination.cost_per_ton:,.0f}/ton
Total Annual Cost: ${determination.annual_cost:,.0f}

Date of Permit/Latest Update: {determination.analysis_date.strftime('%m/%d/%Y')}

Notes: [Additional notes on technology selection, operational limits, etc.]

================================================================================
"""

    def generate_technology_comparison_table(
        self,
        pollutant: PollutantType,
        baseline_emissions_tpy: float
    ) -> str:
        """
        Generate technology comparison table.

        Args:
            pollutant: Pollutant for comparison
            baseline_emissions_tpy: Baseline emissions

        Returns:
            Formatted comparison table
        """
        all_tech = self.identify_control_technologies(pollutant)
        feasible, eliminated = self.eliminate_infeasible_technologies(all_tech)
        ranked = self.rank_by_effectiveness(feasible)

        table = f"""
================================================================================
            CONTROL TECHNOLOGY COMPARISON - {pollutant.value}
================================================================================

Baseline Emissions: {baseline_emissions_tpy:.1f} tons/year
Unit Capacity: {self.unit_capacity} MMBtu/hr
Fuel Type: {self.fuel_type.value}

--------------------------------------------------------------------------------
| Rank | Technology           | Efficiency | Cost/Ton   | Annual Cost  | Feasible |
--------------------------------------------------------------------------------
"""
        for tech in ranked:
            cost_result = self.calculate_cost_effectiveness(tech, baseline_emissions_tpy)
            table += f"| {tech.get('effectiveness_rank', '-'):4} | {tech.get('id', '')[:20]:20} | {tech.get('typical_efficiency', 0) * 100:9.0f}% | ${cost_result.cost_per_ton_removed:9,.0f} | ${cost_result.total_annual_cost:11,.0f} | {'Yes':8} |\n"

        table += "--------------------------------------------------------------------------------\n"

        if eliminated:
            table += "\nELIMINATED TECHNOLOGIES (Technically Infeasible):\n"
            for tech in eliminated:
                table += f"  - {tech.get('id')}: {tech.get('elimination_reason', ['Unknown'])[0]}\n"

        return table

    def generate_cost_summary(
        self,
        pollutant: PollutantType,
        baseline_emissions_tpy: float
    ) -> str:
        """
        Generate cost-effectiveness summary.

        Args:
            pollutant: Pollutant for analysis
            baseline_emissions_tpy: Baseline emissions

        Returns:
            Formatted cost summary
        """
        all_tech = self.identify_control_technologies(pollutant)
        feasible, _ = self.eliminate_infeasible_technologies(all_tech)

        summary = f"""
================================================================================
            COST-EFFECTIVENESS SUMMARY - {pollutant.value}
================================================================================

Economic Parameters:
  Interest Rate: {self.interest_rate * 100:.1f}%
  Equipment Life: {self.equipment_life} years
  Capital Recovery Factor: {self._compute_crf():.4f}
  Cost Year: {self.cost_year}

Cost-Effectiveness Thresholds for {pollutant.value}:
"""
        thresholds = COST_EFFECTIVENESS_THRESHOLDS.get(pollutant.value, {})
        summary += f"  Lower Bound: ${thresholds.get('lower', 'N/A'):,}/ton\n"
        summary += f"  Upper Bound: ${thresholds.get('upper', 'N/A'):,}/ton\n"
        summary += f"  Typical BACT Threshold: ${thresholds.get('typical_bact', 'N/A'):,}/ton\n\n"

        summary += "Technology Cost Analysis:\n"
        summary += "-" * 80 + "\n"

        for tech in feasible:
            cost_result = self.calculate_cost_effectiveness(tech, baseline_emissions_tpy)
            status = "COST-EFFECTIVE" if cost_result.is_cost_effective else "NOT COST-EFFECTIVE"
            summary += f"""
{tech.get('id')}:
  Capital Cost: ${Decimal(str(tech.get('capital_cost_per_mmbtu_hr', 0) * self.unit_capacity)):,.0f}
  Annual O&M: ${cost_result.total_annual_cost - Decimal(str(tech.get('capital_cost_per_mmbtu_hr', 0) * self.unit_capacity * self._compute_crf())):,.0f}
  Total Annual: ${cost_result.total_annual_cost:,.0f}
  Emissions Reduced: {cost_result.emissions_reduction_tpy:.1f} tons/year
  Cost-Effectiveness: ${cost_result.cost_per_ton_removed:,.0f}/ton
  Status: {status}
"""

        return summary


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_bact_analysis(
    pollutant: str,
    unit_capacity_mmbtu_hr: float,
    fuel_type: str,
    baseline_emissions_tpy: float,
    operating_hours_yr: float = 8760
) -> BACTDetermination:
    """
    Perform quick BACT analysis with minimal parameters.

    Args:
        pollutant: Pollutant name (NOx, SO2, PM, CO)
        unit_capacity_mmbtu_hr: Unit capacity
        fuel_type: Fuel type (gas, oil, coal)
        baseline_emissions_tpy: Uncontrolled emissions
        operating_hours_yr: Operating hours per year

    Returns:
        BACTDetermination result
    """
    # Map string inputs to enums
    pollutant_map = {
        "NOx": PollutantType.NOX,
        "NOX": PollutantType.NOX,
        "SO2": PollutantType.SO2,
        "PM": PollutantType.PM,
        "PM10": PollutantType.PM10,
        "PM2.5": PollutantType.PM25,
        "CO": PollutantType.CO,
        "VOC": PollutantType.VOC
    }

    fuel_map = {
        "gas": FuelType.NATURAL_GAS,
        "natural_gas": FuelType.NATURAL_GAS,
        "oil": FuelType.FUEL_OIL,
        "fuel_oil": FuelType.FUEL_OIL,
        "coal": FuelType.COAL,
        "biomass": FuelType.BIOMASS
    }

    pollutant_enum = pollutant_map.get(pollutant, PollutantType.NOX)
    fuel_enum = fuel_map.get(fuel_type.lower(), FuelType.NATURAL_GAS)

    analyzer = BACTAnalyzer(
        unit_capacity_mmbtu_hr=unit_capacity_mmbtu_hr,
        fuel_type=fuel_enum,
        operating_hours_yr=operating_hours_yr
    )

    return analyzer.perform_bact_analysis(
        pollutant=pollutant_enum,
        baseline_emissions_tpy=baseline_emissions_tpy
    )


def get_control_technology_info(
    pollutant: str,
    technology_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific control technology.

    Args:
        pollutant: Pollutant type
        technology_id: Technology identifier

    Returns:
        Technology specification dictionary or None
    """
    pollutant_key = pollutant.upper()
    if pollutant_key in ["PM10", "PM2.5"]:
        pollutant_key = "PM"

    tech_db = CONTROL_TECHNOLOGIES.get(pollutant_key, {})
    return tech_db.get(technology_id)


def list_available_technologies(pollutant: str) -> List[str]:
    """
    List available control technologies for a pollutant.

    Args:
        pollutant: Pollutant type

    Returns:
        List of technology identifiers
    """
    pollutant_key = pollutant.upper()
    if pollutant_key in ["PM10", "PM2.5"]:
        pollutant_key = "PM"

    tech_db = CONTROL_TECHNOLOGIES.get(pollutant_key, {})
    return list(tech_db.keys())


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "PollutantType",
    "AttainmentStatus",
    "FeasibilityStatus",
    "TechnologyRanking",
    "FuelType",
    # Data classes
    "TechnologyCosts",
    "FeasibilityResult",
    "CostEffectivenessResult",
    "BACTDetermination",
    "MultiPollutantResult",
    # Main class
    "BACTAnalyzer",
    # Technology databases
    "CONTROL_TECHNOLOGIES",
    "NOX_CONTROL_TECHNOLOGIES",
    "SO2_CONTROL_TECHNOLOGIES",
    "PM_CONTROL_TECHNOLOGIES",
    "CO_CONTROL_TECHNOLOGIES",
    "COST_EFFECTIVENESS_THRESHOLDS",
    # Convenience functions
    "quick_bact_analysis",
    "get_control_technology_info",
    "list_available_technologies",
]
