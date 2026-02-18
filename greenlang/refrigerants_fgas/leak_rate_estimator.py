# -*- coding: utf-8 -*-
"""
LeakRateEstimatorEngine - Leak Rate Estimation (Engine 4 of 7)

AGENT-MRV-SCOPE1-002: Refrigerants & F-Gas Agent

Sophisticated leak rate estimation engine that provides default annual
leak rates per IPCC 2006 Vol 3 Ch 7 and industry data, with multiple
adjustment factors for equipment age, climate zone, and Leak Detection
and Repair (LDAR) program effectiveness.

The engine supports 15 equipment types across three lifecycle stages
(INSTALLATION, OPERATING, END_OF_LIFE), each with distinct base rates
sourced from IPCC 2006 Guidelines, EPA 40 CFR Part 98, and ASHRAE/
industry consensus data.

Leak Rate Sources:
    - IPCC 2006 Vol 3 Ch 7: Default emission factors by sub-application
    - EPA 40 CFR Part 98 Subpart OO: Industrial gas suppliers
    - ASHRAE Refrigerant Leak Rate Database
    - RTOC (Refrigeration, Air-Conditioning and Heat Pumps Technical
      Options Committee) Assessment Reports

Default Annual Leak Rates by Equipment Type:
    - Commercial refrigeration (centralized): 15-25%
    - Commercial refrigeration (standalone):  2-5%
    - Industrial refrigeration:               8-15%
    - Residential AC (split):                 2-5%
    - Commercial AC (packaged):               4-8%
    - Chillers (centrifugal):                 2-5%
    - Chillers (screw/scroll):                3-6%
    - Heat pumps:                             3-5%
    - Transport refrigeration:                15-30%
    - Switchgear (SF6):                       0.5-2%
    - Semiconductor:                          5-10%
    - Fire suppression:                       1-3%
    - Foam blowing:                           1-5% (of banked charge)
    - Aerosols/MDI:                           100% (emissive use)
    - Solvents:                               50-90%

Lifecycle Stage Rates (per IPCC):
    - INSTALLATION: charge x install_rate (0.5-2% typically)
    - OPERATING:    charge x annual_rate (as above)
    - END_OF_LIFE:  charge x (1 - recovery_efficiency)

Age Adjustment Factors:
    -  0-5  years: 1.00x (baseline)
    -  5-10 years: 1.15x
    - 10-15 years: 1.35x
    - 15+   years: 1.60x

Climate Zone Adjustments:
    - Tropical:     1.15x (higher temps -> more leaks)
    - Subtropical:  1.10x
    - Temperate:    1.00x (baseline)
    - Continental:  0.95x
    - Polar:        0.90x

LDAR Program Effectiveness:
    - No LDAR:              1.00x
    - Annual inspections:   0.85x
    - Quarterly inspections: 0.70x
    - Continuous monitoring: 0.50x

Zero-Hallucination Guarantees:
    - All rates are deterministic lookups from coded tables.
    - No LLM involvement in any numeric path.
    - Decimal arithmetic for bit-perfect reproducibility.
    - Every result carries a SHA-256 provenance hash.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.refrigerants_fgas.leak_rate_estimator import (
    ...     LeakRateEstimatorEngine,
    ... )
    >>> engine = LeakRateEstimatorEngine()
    >>> profile = engine.estimate_leak_rate(
    ...     equipment_type="COMMERCIAL_REFRIGERATION_CENTRALIZED",
    ...     lifecycle_stage="OPERATING",
    ...     age_years=12,
    ...     climate_zone="SUBTROPICAL",
    ...     ldar_level="QUARTERLY",
    ... )
    >>> print(profile.effective_rate)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-SCOPE1-002 Refrigerants & F-Gas Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["LeakRateEstimatorEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.refrigerants_fgas.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.refrigerants_fgas.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.refrigerants_fgas.metrics import (
        PROMETHEUS_AVAILABLE as _METRICS_AVAILABLE,
        record_leak_rate_selection as _record_leak_rate_selection,
        observe_calculation_duration as _observe_calculation_duration,
    )
except ImportError:
    _METRICS_AVAILABLE = False
    _record_leak_rate_selection = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ===========================================================================
# Enumerations
# ===========================================================================


class LeakEquipmentType(str, Enum):
    """Equipment type classification for leak rate estimation.

    Covers all 15 equipment / sub-application categories tracked by the
    IPCC 2006 Guidelines Vol 3 Ch 7 and industry reference databases.

    COMMERCIAL_REFRIGERATION_CENTRALIZED: Multi-compressor racks with
        extensive piping (supermarkets, cold stores). 15-25% annual rate.
    COMMERCIAL_REFRIGERATION_STANDALONE: Self-contained units (vending
        machines, display cases). 2-5% annual rate.
    INDUSTRIAL_REFRIGERATION: Large ammonia or HFC systems for process
        cooling. 8-15% annual rate.
    RESIDENTIAL_AC_SPLIT: Split-system residential air conditioners.
        2-5% annual rate.
    COMMERCIAL_AC_PACKAGED: Packaged rooftop or unitary commercial AC
        systems. 4-8% annual rate.
    CHILLER_CENTRIFUGAL: Centrifugal chillers with hermetic or
        semi-hermetic compressors. 2-5% annual rate.
    CHILLER_SCREW_SCROLL: Screw or scroll compressor chillers. 3-6%.
    HEAT_PUMP: Air-source and ground-source heat pumps. 3-5%.
    TRANSPORT_REFRIGERATION: Truck/trailer/container refrigeration
        units with high vibration stress. 15-30%.
    SWITCHGEAR_SF6: High-voltage switchgear and circuit breakers
        containing SF6. 0.5-2%.
    SEMICONDUCTOR: Semiconductor manufacturing using PFCs, SF6, NF3.
        5-10% per production cycle.
    FIRE_SUPPRESSION: HFC-based fire suppression systems (HFC-227ea,
        HFC-23). 1-3%.
    FOAM_BLOWING: Closed-cell foam insulation containing blowing agents
        (banked HFC emissions). 1-5% of banked charge.
    AEROSOL_MDI: Metered-dose inhalers and aerosol products. 100%
        emissive use during the year of production/sale.
    SOLVENT: Fluorinated solvents used in cleaning and degreasing.
        50-90% per year.
    """

    COMMERCIAL_REFRIGERATION_CENTRALIZED = "COMMERCIAL_REFRIGERATION_CENTRALIZED"
    COMMERCIAL_REFRIGERATION_STANDALONE = "COMMERCIAL_REFRIGERATION_STANDALONE"
    INDUSTRIAL_REFRIGERATION = "INDUSTRIAL_REFRIGERATION"
    RESIDENTIAL_AC_SPLIT = "RESIDENTIAL_AC_SPLIT"
    COMMERCIAL_AC_PACKAGED = "COMMERCIAL_AC_PACKAGED"
    CHILLER_CENTRIFUGAL = "CHILLER_CENTRIFUGAL"
    CHILLER_SCREW_SCROLL = "CHILLER_SCREW_SCROLL"
    HEAT_PUMP = "HEAT_PUMP"
    TRANSPORT_REFRIGERATION = "TRANSPORT_REFRIGERATION"
    SWITCHGEAR_SF6 = "SWITCHGEAR_SF6"
    SEMICONDUCTOR = "SEMICONDUCTOR"
    FIRE_SUPPRESSION = "FIRE_SUPPRESSION"
    FOAM_BLOWING = "FOAM_BLOWING"
    AEROSOL_MDI = "AEROSOL_MDI"
    SOLVENT = "SOLVENT"


class LifecycleStage(str, Enum):
    """Lifecycle stage for emission accounting per IPCC.

    INSTALLATION: Losses during initial equipment charging and setup.
        Typically 0.5-2% of initial charge.
    OPERATING: Annual operating leak rate during the equipment's
        service life. This is the dominant emission source.
    END_OF_LIFE: Losses during equipment decommissioning, representing
        the fraction not recovered. Depends on recovery efficiency.
    """

    INSTALLATION = "INSTALLATION"
    OPERATING = "OPERATING"
    END_OF_LIFE = "END_OF_LIFE"


class ClimateZone(str, Enum):
    """Climate zone classification affecting leak rates.

    Higher ambient temperatures increase refrigerant system pressures,
    accelerating leak rates through seals and joints.

    TROPICAL:    Avg annual temp > 25C. Factor 1.15x.
    SUBTROPICAL: Avg annual temp 18-25C. Factor 1.10x.
    TEMPERATE:   Avg annual temp 10-18C. Factor 1.00x (baseline).
    CONTINENTAL: Avg annual temp 0-10C. Factor 0.95x.
    POLAR:       Avg annual temp < 0C. Factor 0.90x.
    """

    TROPICAL = "TROPICAL"
    SUBTROPICAL = "SUBTROPICAL"
    TEMPERATE = "TEMPERATE"
    CONTINENTAL = "CONTINENTAL"
    POLAR = "POLAR"


class LDARLevel(str, Enum):
    """Leak Detection and Repair (LDAR) program intensity levels.

    NONE:         No systematic leak detection program. Factor 1.00x.
    ANNUAL:       Annual inspection and repair program. Factor 0.85x.
    QUARTERLY:    Quarterly inspections with prompt repair. Factor 0.70x.
    CONTINUOUS:   Continuous online monitoring with automated alerts
                  and immediate repair dispatch. Factor 0.50x.
    """

    NONE = "NONE"
    ANNUAL = "ANNUAL"
    QUARTERLY = "QUARTERLY"
    CONTINUOUS = "CONTINUOUS"


# ===========================================================================
# Dataclasses for results
# ===========================================================================


@dataclass
class LeakRateProfile:
    """Complete leak rate profile with all adjustment factors and provenance.

    Attributes:
        profile_id: Unique identifier for this leak rate estimation.
        equipment_type: Equipment type classification.
        lifecycle_stage: Lifecycle stage used for rate selection.
        base_rate: Base annual leak rate from the default database,
            as a Decimal fraction (e.g. 0.15 for 15%).
        base_rate_range: Tuple of (low, high) range for the base rate.
        age_years: Equipment age in years.
        age_factor: Multiplier applied for equipment age degradation.
        climate_zone: Climate zone used for adjustment.
        climate_factor: Multiplier applied for climate zone conditions.
        ldar_level: LDAR program level used for adjustment.
        ldar_factor: Multiplier applied for LDAR program effectiveness.
        custom_rate: Optional custom override rate (if provided).
        effective_rate: Final computed effective leak rate after all
            adjustments, as a Decimal fraction.
        effective_rate_pct: Effective rate expressed as a percentage.
        rate_source: Description of the rate source authority.
        provenance_hash: SHA-256 hash of the estimation computation.
        timestamp: UTC ISO-formatted timestamp.
        metadata: Additional metadata dictionary.
    """

    profile_id: str
    equipment_type: str
    lifecycle_stage: str
    base_rate: Decimal
    base_rate_range: Tuple[Decimal, Decimal]
    age_years: int
    age_factor: Decimal
    climate_zone: str
    climate_factor: Decimal
    ldar_level: str
    ldar_factor: Decimal
    custom_rate: Optional[Decimal]
    effective_rate: Decimal
    effective_rate_pct: Decimal
    rate_source: str
    provenance_hash: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the profile to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "profile_id": self.profile_id,
            "equipment_type": self.equipment_type,
            "lifecycle_stage": self.lifecycle_stage,
            "base_rate": str(self.base_rate),
            "base_rate_range": [str(self.base_rate_range[0]), str(self.base_rate_range[1])],
            "age_years": self.age_years,
            "age_factor": str(self.age_factor),
            "climate_zone": self.climate_zone,
            "climate_factor": str(self.climate_factor),
            "ldar_level": self.ldar_level,
            "ldar_factor": str(self.ldar_factor),
            "custom_rate": str(self.custom_rate) if self.custom_rate is not None else None,
            "effective_rate": str(self.effective_rate),
            "effective_rate_pct": str(self.effective_rate_pct),
            "rate_source": self.rate_source,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class LifetimeEmissionsProfile:
    """Lifetime emission estimate across all lifecycle stages.

    Attributes:
        profile_id: Unique identifier for this lifetime estimate.
        equipment_type: Equipment type classification.
        charge_kg: Initial refrigerant charge in kilograms.
        gwp: Global Warming Potential applied.
        lifetime_years: Assumed equipment service lifetime in years.
        installation_loss_kg: Refrigerant lost during installation.
        installation_emissions_tco2e: Installation loss as tCO2e.
        annual_operating_loss_kg: Annual operating leak rate loss.
        total_operating_loss_kg: Cumulative operating loss over lifetime.
        total_operating_emissions_tco2e: Operating emissions as tCO2e.
        end_of_life_loss_kg: Loss at decommissioning.
        end_of_life_emissions_tco2e: End-of-life loss as tCO2e.
        total_lifetime_loss_kg: Total refrigerant loss across all stages.
        total_lifetime_emissions_tco2e: Total emissions across all stages.
        recovery_efficiency: Recovery efficiency used for end-of-life.
        provenance_hash: SHA-256 hash.
        timestamp: UTC ISO-formatted timestamp.
    """

    profile_id: str
    equipment_type: str
    charge_kg: Decimal
    gwp: Decimal
    lifetime_years: int
    installation_loss_kg: Decimal
    installation_emissions_tco2e: Decimal
    annual_operating_loss_kg: Decimal
    total_operating_loss_kg: Decimal
    total_operating_emissions_tco2e: Decimal
    end_of_life_loss_kg: Decimal
    end_of_life_emissions_tco2e: Decimal
    total_lifetime_loss_kg: Decimal
    total_lifetime_emissions_tco2e: Decimal
    recovery_efficiency: Decimal
    provenance_hash: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "profile_id": self.profile_id,
            "equipment_type": self.equipment_type,
            "charge_kg": str(self.charge_kg),
            "gwp": str(self.gwp),
            "lifetime_years": self.lifetime_years,
            "installation_loss_kg": str(self.installation_loss_kg),
            "installation_emissions_tco2e": str(self.installation_emissions_tco2e),
            "annual_operating_loss_kg": str(self.annual_operating_loss_kg),
            "total_operating_loss_kg": str(self.total_operating_loss_kg),
            "total_operating_emissions_tco2e": str(self.total_operating_emissions_tco2e),
            "end_of_life_loss_kg": str(self.end_of_life_loss_kg),
            "end_of_life_emissions_tco2e": str(self.end_of_life_emissions_tco2e),
            "total_lifetime_loss_kg": str(self.total_lifetime_loss_kg),
            "total_lifetime_emissions_tco2e": str(self.total_lifetime_emissions_tco2e),
            "recovery_efficiency": str(self.recovery_efficiency),
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
        }


# ===========================================================================
# Default Leak Rate Database
# IPCC 2006 Vol 3 Ch 7, EPA 40 CFR Part 98, ASHRAE, RTOC
# ===========================================================================

# Structure: equipment_type -> {
#   "default": Decimal (midpoint annual operating rate as fraction),
#   "range": (Decimal, Decimal) (low, high as fractions),
#   "source": str,
#   "lifecycle": {
#       "INSTALLATION": Decimal (fraction of charge lost at install),
#       "OPERATING": Decimal (annual operating rate as fraction),
#       "END_OF_LIFE": Decimal (1 - recovery_efficiency),
#   },
#   "recovery_efficiency": Decimal,
#   "typical_charge_kg": Decimal (reference typical charge),
#   "typical_lifetime_years": int,
# }

_DEFAULT_LEAK_RATES: Dict[str, Dict[str, Any]] = {
    LeakEquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED.value: {
        "default": Decimal("0.20"),
        "range": (Decimal("0.15"), Decimal("0.25")),
        "source": "IPCC 2006 Vol 3 Ch 7 Table 7.9; RTOC 2018",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.01"),
            LifecycleStage.OPERATING.value: Decimal("0.20"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.15"),
        },
        "recovery_efficiency": Decimal("0.85"),
        "typical_charge_kg": Decimal("200"),
        "typical_lifetime_years": 15,
    },
    LeakEquipmentType.COMMERCIAL_REFRIGERATION_STANDALONE.value: {
        "default": Decimal("0.035"),
        "range": (Decimal("0.02"), Decimal("0.05")),
        "source": "IPCC 2006 Vol 3 Ch 7 Table 7.9; ASHRAE",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.005"),
            LifecycleStage.OPERATING.value: Decimal("0.035"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.10"),
        },
        "recovery_efficiency": Decimal("0.90"),
        "typical_charge_kg": Decimal("0.5"),
        "typical_lifetime_years": 12,
    },
    LeakEquipmentType.INDUSTRIAL_REFRIGERATION.value: {
        "default": Decimal("0.12"),
        "range": (Decimal("0.08"), Decimal("0.15")),
        "source": "IPCC 2006 Vol 3 Ch 7 Table 7.9; EPA OO",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.01"),
            LifecycleStage.OPERATING.value: Decimal("0.12"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.10"),
        },
        "recovery_efficiency": Decimal("0.90"),
        "typical_charge_kg": Decimal("500"),
        "typical_lifetime_years": 20,
    },
    LeakEquipmentType.RESIDENTIAL_AC_SPLIT.value: {
        "default": Decimal("0.035"),
        "range": (Decimal("0.02"), Decimal("0.05")),
        "source": "IPCC 2006 Vol 3 Ch 7 Table 7.9; JRAIA",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.005"),
            LifecycleStage.OPERATING.value: Decimal("0.035"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.20"),
        },
        "recovery_efficiency": Decimal("0.80"),
        "typical_charge_kg": Decimal("1.5"),
        "typical_lifetime_years": 15,
    },
    LeakEquipmentType.COMMERCIAL_AC_PACKAGED.value: {
        "default": Decimal("0.06"),
        "range": (Decimal("0.04"), Decimal("0.08")),
        "source": "IPCC 2006 Vol 3 Ch 7 Table 7.9; ASHRAE",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.01"),
            LifecycleStage.OPERATING.value: Decimal("0.06"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.15"),
        },
        "recovery_efficiency": Decimal("0.85"),
        "typical_charge_kg": Decimal("10"),
        "typical_lifetime_years": 15,
    },
    LeakEquipmentType.CHILLER_CENTRIFUGAL.value: {
        "default": Decimal("0.035"),
        "range": (Decimal("0.02"), Decimal("0.05")),
        "source": "IPCC 2006 Vol 3 Ch 7 Table 7.9; AHRI",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.005"),
            LifecycleStage.OPERATING.value: Decimal("0.035"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.05"),
        },
        "recovery_efficiency": Decimal("0.95"),
        "typical_charge_kg": Decimal("300"),
        "typical_lifetime_years": 25,
    },
    LeakEquipmentType.CHILLER_SCREW_SCROLL.value: {
        "default": Decimal("0.045"),
        "range": (Decimal("0.03"), Decimal("0.06")),
        "source": "IPCC 2006 Vol 3 Ch 7 Table 7.9; AHRI",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.005"),
            LifecycleStage.OPERATING.value: Decimal("0.045"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.05"),
        },
        "recovery_efficiency": Decimal("0.95"),
        "typical_charge_kg": Decimal("100"),
        "typical_lifetime_years": 20,
    },
    LeakEquipmentType.HEAT_PUMP.value: {
        "default": Decimal("0.04"),
        "range": (Decimal("0.03"), Decimal("0.05")),
        "source": "IPCC 2006 Vol 3 Ch 7 Table 7.9; EHPA",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.005"),
            LifecycleStage.OPERATING.value: Decimal("0.04"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.10"),
        },
        "recovery_efficiency": Decimal("0.90"),
        "typical_charge_kg": Decimal("3"),
        "typical_lifetime_years": 15,
    },
    LeakEquipmentType.TRANSPORT_REFRIGERATION.value: {
        "default": Decimal("0.225"),
        "range": (Decimal("0.15"), Decimal("0.30")),
        "source": "IPCC 2006 Vol 3 Ch 7 Table 7.9; RTOC 2018",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.02"),
            LifecycleStage.OPERATING.value: Decimal("0.225"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.20"),
        },
        "recovery_efficiency": Decimal("0.80"),
        "typical_charge_kg": Decimal("6"),
        "typical_lifetime_years": 12,
    },
    LeakEquipmentType.SWITCHGEAR_SF6.value: {
        "default": Decimal("0.01"),
        "range": (Decimal("0.005"), Decimal("0.02")),
        "source": "IPCC 2006 Vol 3 Ch 7; IEC 62271; CIGRE",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.005"),
            LifecycleStage.OPERATING.value: Decimal("0.01"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.02"),
        },
        "recovery_efficiency": Decimal("0.98"),
        "typical_charge_kg": Decimal("50"),
        "typical_lifetime_years": 30,
    },
    LeakEquipmentType.SEMICONDUCTOR.value: {
        "default": Decimal("0.075"),
        "range": (Decimal("0.05"), Decimal("0.10")),
        "source": "IPCC 2006 Vol 3 Ch 6; SEMI/WSC",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.01"),
            LifecycleStage.OPERATING.value: Decimal("0.075"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.05"),
        },
        "recovery_efficiency": Decimal("0.95"),
        "typical_charge_kg": Decimal("20"),
        "typical_lifetime_years": 10,
    },
    LeakEquipmentType.FIRE_SUPPRESSION.value: {
        "default": Decimal("0.02"),
        "range": (Decimal("0.01"), Decimal("0.03")),
        "source": "IPCC 2006 Vol 3 Ch 7 Table 7.9; NFPA",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.005"),
            LifecycleStage.OPERATING.value: Decimal("0.02"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.05"),
        },
        "recovery_efficiency": Decimal("0.95"),
        "typical_charge_kg": Decimal("100"),
        "typical_lifetime_years": 20,
    },
    LeakEquipmentType.FOAM_BLOWING.value: {
        "default": Decimal("0.03"),
        "range": (Decimal("0.01"), Decimal("0.05")),
        "source": "IPCC 2006 Vol 3 Ch 7; FTOC Assessment",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.10"),
            LifecycleStage.OPERATING.value: Decimal("0.03"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.50"),
        },
        "recovery_efficiency": Decimal("0.50"),
        "typical_charge_kg": Decimal("5"),
        "typical_lifetime_years": 50,
    },
    LeakEquipmentType.AEROSOL_MDI.value: {
        "default": Decimal("1.00"),
        "range": (Decimal("1.00"), Decimal("1.00")),
        "source": "IPCC 2006 Vol 3 Ch 7; MTOC Assessment",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.00"),
            LifecycleStage.OPERATING.value: Decimal("1.00"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.00"),
        },
        "recovery_efficiency": Decimal("0.00"),
        "typical_charge_kg": Decimal("0.015"),
        "typical_lifetime_years": 1,
    },
    LeakEquipmentType.SOLVENT.value: {
        "default": Decimal("0.70"),
        "range": (Decimal("0.50"), Decimal("0.90")),
        "source": "IPCC 2006 Vol 3 Ch 7; CTOC Assessment",
        "lifecycle": {
            LifecycleStage.INSTALLATION.value: Decimal("0.00"),
            LifecycleStage.OPERATING.value: Decimal("0.70"),
            LifecycleStage.END_OF_LIFE.value: Decimal("0.20"),
        },
        "recovery_efficiency": Decimal("0.80"),
        "typical_charge_kg": Decimal("10"),
        "typical_lifetime_years": 1,
    },
}


# ===========================================================================
# Age Adjustment Factor Table
# ===========================================================================

# List of (max_age_exclusive, factor) pairs, searched in order.
# Equipment older than all listed thresholds uses the last factor.
_AGE_ADJUSTMENT_TABLE: List[Tuple[int, Decimal]] = [
    (5, Decimal("1.00")),    # 0-4 years: baseline
    (10, Decimal("1.15")),   # 5-9 years: +15%
    (15, Decimal("1.35")),   # 10-14 years: +35%
]
_AGE_ADJUSTMENT_MAX: Decimal = Decimal("1.60")  # 15+ years: +60%


# ===========================================================================
# Climate Zone Adjustment Factor Table
# ===========================================================================

_CLIMATE_ZONE_FACTORS: Dict[str, Decimal] = {
    ClimateZone.TROPICAL.value: Decimal("1.15"),
    ClimateZone.SUBTROPICAL.value: Decimal("1.10"),
    ClimateZone.TEMPERATE.value: Decimal("1.00"),
    ClimateZone.CONTINENTAL.value: Decimal("0.95"),
    ClimateZone.POLAR.value: Decimal("0.90"),
}


# ===========================================================================
# LDAR Program Effectiveness Factor Table
# ===========================================================================

_LDAR_FACTORS: Dict[str, Decimal] = {
    LDARLevel.NONE.value: Decimal("1.00"),
    LDARLevel.ANNUAL.value: Decimal("0.85"),
    LDARLevel.QUARTERLY.value: Decimal("0.70"),
    LDARLevel.CONTINUOUS.value: Decimal("0.50"),
}


# ===========================================================================
# Default Recovery Efficiencies by Equipment Type
# ===========================================================================

_DEFAULT_RECOVERY_EFFICIENCIES: Dict[str, Decimal] = {
    et: _DEFAULT_LEAK_RATES[et]["recovery_efficiency"]
    for et in _DEFAULT_LEAK_RATES
}


# ===========================================================================
# LeakRateEstimatorEngine
# ===========================================================================


class LeakRateEstimatorEngine:
    """Sophisticated leak rate estimation engine for refrigerant and F-gas
    equipment with multiple adjustment factors.

    Provides deterministic, zero-hallucination leak rate estimation using
    coded lookup tables sourced from IPCC 2006 Vol 3 Ch 7, EPA 40 CFR
    Part 98, ASHRAE, and RTOC assessment reports. All arithmetic uses
    Python Decimal for bit-perfect reproducibility.

    The engine supports:
        - 15 equipment types with distinct base leak rates
        - 3 lifecycle stages (installation, operating, end-of-life)
        - 4-tier age degradation adjustment (0-5/5-10/10-15/15+ years)
        - 5 climate zone adjustments (tropical through polar)
        - 4 LDAR program effectiveness levels
        - Custom rate overrides for facility-specific data
        - Lifetime emission projections across all stages
        - Registration of custom leak rates for non-standard equipment
        - SHA-256 provenance hash for every estimation

    Thread Safety:
        All mutable state (_custom_rates, _estimation_history) is
        protected by a reentrant lock. Concurrent callers are safe.

    Attributes:
        _custom_rates: Dictionary of registered custom leak rates
            keyed by ``"equipment_type:lifecycle_stage"``.
        _estimation_history: List of all LeakRateProfile results
            produced by this engine instance.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> engine = LeakRateEstimatorEngine()
        >>> profile = engine.estimate_leak_rate(
        ...     equipment_type="COMMERCIAL_REFRIGERATION_CENTRALIZED",
        ...     lifecycle_stage="OPERATING",
        ...     age_years=12,
        ...     climate_zone="SUBTROPICAL",
        ...     ldar_level="QUARTERLY",
        ... )
        >>> print(profile.effective_rate_pct)
    """

    def __init__(self) -> None:
        """Initialize the LeakRateEstimatorEngine.

        Loads the default leak rate database and sets up internal state
        for custom rate registration and estimation history tracking.
        """
        self._custom_rates: Dict[str, Decimal] = {}
        self._estimation_history: List[LeakRateProfile] = []
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "LeakRateEstimatorEngine initialized: "
            "%d equipment types loaded, "
            "%d climate zones, "
            "%d LDAR levels",
            len(_DEFAULT_LEAK_RATES),
            len(_CLIMATE_ZONE_FACTORS),
            len(_LDAR_FACTORS),
        )

    # ------------------------------------------------------------------
    # Public API: Core Estimation
    # ------------------------------------------------------------------

    def estimate_leak_rate(
        self,
        equipment_type: str,
        lifecycle_stage: str = "OPERATING",
        age_years: int = 0,
        climate_zone: str = "TEMPERATE",
        ldar_level: str = "NONE",
        custom_rate: Optional[Decimal] = None,
    ) -> LeakRateProfile:
        """Estimate the effective leak rate with all adjustment factors.

        Computes the effective leak rate by starting from the base rate
        for the given equipment type and lifecycle stage, then applying
        multiplicative adjustment factors for equipment age, climate
        zone, and LDAR program effectiveness. If a custom_rate is
        provided, it replaces the base rate before adjustments are
        applied.

        Formula:
            effective_rate = base_rate * age_factor * climate_factor * ldar_factor

        Where base_rate is either:
            - The lifecycle-stage-specific rate from the default database
            - Or the custom_rate if provided

        Args:
            equipment_type: Equipment type string matching a
                LeakEquipmentType enum value (e.g.
                "COMMERCIAL_REFRIGERATION_CENTRALIZED").
            lifecycle_stage: Lifecycle stage string matching a
                LifecycleStage enum value. Defaults to "OPERATING".
            age_years: Equipment age in years for age-based adjustment.
                Must be >= 0. Defaults to 0 (new equipment).
            climate_zone: Climate zone string matching a ClimateZone
                enum value. Defaults to "TEMPERATE".
            ldar_level: LDAR program level string matching an LDARLevel
                enum value. Defaults to "NONE".
            custom_rate: Optional custom leak rate as a Decimal fraction
                (e.g. Decimal("0.10") for 10%). If provided, overrides
                the default base rate.

        Returns:
            LeakRateProfile with the effective rate and all factors.

        Raises:
            ValueError: If equipment_type, lifecycle_stage, climate_zone,
                or ldar_level are not recognized, or if age_years < 0,
                or if custom_rate is outside [0, 1].
        """
        t_start = time.monotonic()

        # Validate inputs
        self._validate_equipment_type(equipment_type)
        self._validate_lifecycle_stage(lifecycle_stage)
        self._validate_climate_zone(climate_zone)
        self._validate_ldar_level(ldar_level)

        if age_years < 0:
            raise ValueError(
                f"age_years must be >= 0, got {age_years}"
            )
        if custom_rate is not None:
            if not isinstance(custom_rate, Decimal):
                custom_rate = Decimal(str(custom_rate))
            if custom_rate < Decimal("0") or custom_rate > Decimal("1"):
                raise ValueError(
                    f"custom_rate must be in [0, 1], got {custom_rate}"
                )

        # Check for registered custom rate
        custom_key = f"{equipment_type}:{lifecycle_stage}"
        registered_custom = None
        with self._lock:
            if custom_key in self._custom_rates:
                registered_custom = self._custom_rates[custom_key]

        # Determine base rate
        rate_entry = _DEFAULT_LEAK_RATES[equipment_type]
        if custom_rate is not None:
            base_rate = custom_rate
            rate_source = "custom_override"
        elif registered_custom is not None:
            base_rate = registered_custom
            rate_source = "registered_custom"
        else:
            base_rate = rate_entry["lifecycle"][lifecycle_stage]
            rate_source = rate_entry["source"]

        base_range = rate_entry["range"]

        # Compute adjustment factors
        age_factor = self.calculate_age_factor(age_years)
        climate_factor = self.calculate_climate_factor(climate_zone)
        ldar_factor = self.calculate_ldar_factor(ldar_level)

        # Compute effective rate
        # For INSTALLATION and END_OF_LIFE stages, age and climate
        # factors are less relevant but still applied for consistency.
        # LDAR only applies to OPERATING stage.
        if lifecycle_stage == LifecycleStage.OPERATING.value:
            effective_rate = (
                base_rate * age_factor * climate_factor * ldar_factor
            )
        elif lifecycle_stage == LifecycleStage.INSTALLATION.value:
            # Installation losses are one-time; only climate affects
            effective_rate = base_rate * climate_factor
        else:
            # END_OF_LIFE: recovery efficiency determines the loss
            effective_rate = base_rate
            # age_factor not applied (decommissioning is independent of age)

        # Cap at 1.0 (100%)
        if effective_rate > Decimal("1"):
            effective_rate = Decimal("1")

        # Round to 6 decimal places
        effective_rate = effective_rate.quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )
        effective_rate_pct = (effective_rate * Decimal("100")).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        # Build provenance hash
        provenance_data = {
            "equipment_type": equipment_type,
            "lifecycle_stage": lifecycle_stage,
            "base_rate": str(base_rate),
            "age_years": age_years,
            "age_factor": str(age_factor),
            "climate_zone": climate_zone,
            "climate_factor": str(climate_factor),
            "ldar_level": ldar_level,
            "ldar_factor": str(ldar_factor),
            "custom_rate": str(custom_rate) if custom_rate is not None else None,
            "effective_rate": str(effective_rate),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        timestamp = _utcnow().isoformat()

        profile = LeakRateProfile(
            profile_id=f"lr_{uuid4().hex[:12]}",
            equipment_type=equipment_type,
            lifecycle_stage=lifecycle_stage,
            base_rate=base_rate,
            base_rate_range=base_range,
            age_years=age_years,
            age_factor=age_factor,
            climate_zone=climate_zone,
            climate_factor=climate_factor,
            ldar_level=ldar_level,
            ldar_factor=ldar_factor,
            custom_rate=custom_rate,
            effective_rate=effective_rate,
            effective_rate_pct=effective_rate_pct,
            rate_source=rate_source,
            provenance_hash=provenance_hash,
            timestamp=timestamp,
            metadata={
                "base_rate_range_low": str(base_range[0]),
                "base_rate_range_high": str(base_range[1]),
                "typical_charge_kg": str(rate_entry["typical_charge_kg"]),
                "typical_lifetime_years": rate_entry["typical_lifetime_years"],
            },
        )

        # Record in history
        with self._lock:
            self._estimation_history.append(profile)

        # Record provenance
        if _PROVENANCE_AVAILABLE and _get_provenance_tracker is not None:
            try:
                tracker = _get_provenance_tracker()
                tracker.record(
                    entity_type="leak_rate",
                    action="estimate_leak_rate",
                    entity_id=profile.profile_id,
                    data=provenance_data,
                    metadata={"equipment_type": equipment_type},
                )
            except Exception:
                logger.debug("Provenance recording skipped", exc_info=True)

        # Record metrics
        elapsed = time.monotonic() - t_start
        if _METRICS_AVAILABLE and _record_leak_rate_selection is not None:
            try:
                _record_leak_rate_selection(equipment_type, lifecycle_stage)
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)
        if _METRICS_AVAILABLE and _observe_calculation_duration is not None:
            try:
                _observe_calculation_duration(elapsed)
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)

        logger.debug(
            "Leak rate estimated: equipment=%s stage=%s "
            "base=%.4f effective=%.4f (%.2f%%) age_factor=%.2f "
            "climate_factor=%.2f ldar_factor=%.2f in %.1fms",
            equipment_type,
            lifecycle_stage,
            base_rate,
            effective_rate,
            effective_rate_pct,
            age_factor,
            climate_factor,
            ldar_factor,
            elapsed * 1000,
        )

        return profile

    def get_default_rate(self, equipment_type: str) -> Decimal:
        """Get the default midpoint annual operating leak rate for an
        equipment type.

        Returns the midpoint of the operating range as stored in the
        default leak rate database, without any adjustment factors.

        Args:
            equipment_type: Equipment type string matching a
                LeakEquipmentType enum value.

        Returns:
            Decimal fraction representing the default annual operating
            leak rate (e.g. Decimal("0.20") for 20%).

        Raises:
            ValueError: If equipment_type is not recognized.
        """
        self._validate_equipment_type(equipment_type)
        return _DEFAULT_LEAK_RATES[equipment_type]["default"]

    def get_lifecycle_rates(
        self, equipment_type: str,
    ) -> Dict[str, Decimal]:
        """Get the lifecycle-stage-specific leak rates for an equipment type.

        Returns a dictionary mapping each lifecycle stage (INSTALLATION,
        OPERATING, END_OF_LIFE) to its base rate from the default
        database, without any adjustment factors.

        Args:
            equipment_type: Equipment type string matching a
                LeakEquipmentType enum value.

        Returns:
            Dictionary mapping lifecycle stage strings to Decimal rates.

        Raises:
            ValueError: If equipment_type is not recognized.
        """
        self._validate_equipment_type(equipment_type)
        return dict(_DEFAULT_LEAK_RATES[equipment_type]["lifecycle"])

    def calculate_age_factor(self, age_years: int) -> Decimal:
        """Calculate the age degradation adjustment factor.

        Equipment age increases leak rates due to seal degradation,
        joint fatigue, and corrosion. The factor is determined by
        the age bracket:
            -  0-4 years:  1.00x (baseline)
            -  5-9 years:  1.15x
            - 10-14 years: 1.35x
            - 15+  years:  1.60x

        Args:
            age_years: Equipment age in years. Must be >= 0.

        Returns:
            Decimal multiplier for age adjustment.

        Raises:
            ValueError: If age_years is negative.
        """
        if age_years < 0:
            raise ValueError(f"age_years must be >= 0, got {age_years}")

        for max_age, factor in _AGE_ADJUSTMENT_TABLE:
            if age_years < max_age:
                return factor
        return _AGE_ADJUSTMENT_MAX

    def calculate_climate_factor(self, zone: str) -> Decimal:
        """Calculate the climate zone adjustment factor.

        Higher ambient temperatures increase system pressures and
        thermal cycling, which accelerate refrigerant leaks.

        Args:
            zone: Climate zone string matching a ClimateZone enum value.

        Returns:
            Decimal multiplier for climate zone adjustment.

        Raises:
            ValueError: If zone is not recognized.
        """
        self._validate_climate_zone(zone)
        return _CLIMATE_ZONE_FACTORS[zone]

    def calculate_ldar_factor(self, level: str) -> Decimal:
        """Calculate the LDAR program effectiveness factor.

        Active Leak Detection and Repair programs reduce effective leak
        rates by enabling early detection and prompt repair.

        Args:
            level: LDAR level string matching an LDARLevel enum value.

        Returns:
            Decimal multiplier for LDAR effectiveness adjustment.

        Raises:
            ValueError: If level is not recognized.
        """
        self._validate_ldar_level(level)
        return _LDAR_FACTORS[level]

    def estimate_annual_loss(
        self,
        charge_kg: Decimal,
        equipment_type: str,
        lifecycle_stage: str = "OPERATING",
        age_years: int = 0,
        climate_zone: str = "TEMPERATE",
        ldar_level: str = "NONE",
        custom_rate: Optional[Decimal] = None,
    ) -> Decimal:
        """Estimate the annual refrigerant loss in kilograms.

        Computes the effective leak rate via estimate_leak_rate and
        multiplies by the equipment charge to produce the mass loss.

        Formula:
            annual_loss_kg = charge_kg * effective_rate

        Args:
            charge_kg: Refrigerant charge in kilograms. Must be > 0.
            equipment_type: Equipment type string.
            lifecycle_stage: Lifecycle stage string. Defaults to "OPERATING".
            age_years: Equipment age in years. Defaults to 0.
            climate_zone: Climate zone string. Defaults to "TEMPERATE".
            ldar_level: LDAR level string. Defaults to "NONE".
            custom_rate: Optional custom override rate.

        Returns:
            Decimal mass loss in kilograms, rounded to 3 decimal places.

        Raises:
            ValueError: If charge_kg <= 0 or other inputs are invalid.
        """
        if not isinstance(charge_kg, Decimal):
            charge_kg = Decimal(str(charge_kg))
        if charge_kg <= Decimal("0"):
            raise ValueError(f"charge_kg must be > 0, got {charge_kg}")

        profile = self.estimate_leak_rate(
            equipment_type=equipment_type,
            lifecycle_stage=lifecycle_stage,
            age_years=age_years,
            climate_zone=climate_zone,
            ldar_level=ldar_level,
            custom_rate=custom_rate,
        )

        annual_loss = charge_kg * profile.effective_rate
        return annual_loss.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def estimate_lifetime_emissions(
        self,
        charge_kg: Decimal,
        equipment_type: str,
        gwp: Decimal,
        lifetime_years: Optional[int] = None,
        climate_zone: str = "TEMPERATE",
        ldar_level: str = "NONE",
        recovery_efficiency: Optional[Decimal] = None,
    ) -> LifetimeEmissionsProfile:
        """Estimate total lifetime emissions across all lifecycle stages.

        Calculates refrigerant loss and resulting tCO2e emissions for:
        1. INSTALLATION: One-time loss during initial charging.
        2. OPERATING: Annual losses over the equipment service life,
           with age degradation applied year-by-year.
        3. END_OF_LIFE: Loss at decommissioning based on recovery
           efficiency.

        The operating stage applies increasing age factors as the
        equipment progresses through age brackets, providing a more
        accurate cumulative estimate than using a single average rate.

        Args:
            charge_kg: Initial refrigerant charge in kilograms.
            equipment_type: Equipment type string.
            gwp: Global Warming Potential (dimensionless) for the
                refrigerant. E.g. Decimal("2088") for R-410A AR6.
            lifetime_years: Equipment service lifetime in years. If
                None, uses the typical lifetime from the database.
            climate_zone: Climate zone string. Defaults to "TEMPERATE".
            ldar_level: LDAR level string. Defaults to "NONE".
            recovery_efficiency: End-of-life recovery efficiency as
                Decimal fraction (0 to 1). If None, uses the default
                from the database.

        Returns:
            LifetimeEmissionsProfile with detailed breakdown.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(charge_kg, Decimal):
            charge_kg = Decimal(str(charge_kg))
        if charge_kg <= Decimal("0"):
            raise ValueError(f"charge_kg must be > 0, got {charge_kg}")
        if not isinstance(gwp, Decimal):
            gwp = Decimal(str(gwp))
        if gwp < Decimal("0"):
            raise ValueError(f"gwp must be >= 0, got {gwp}")

        self._validate_equipment_type(equipment_type)
        rate_entry = _DEFAULT_LEAK_RATES[equipment_type]

        if lifetime_years is None:
            lifetime_years = rate_entry["typical_lifetime_years"]
        if lifetime_years < 0:
            raise ValueError(
                f"lifetime_years must be >= 0, got {lifetime_years}"
            )

        if recovery_efficiency is None:
            recovery_efficiency = rate_entry["recovery_efficiency"]
        if not isinstance(recovery_efficiency, Decimal):
            recovery_efficiency = Decimal(str(recovery_efficiency))
        if recovery_efficiency < Decimal("0") or recovery_efficiency > Decimal("1"):
            raise ValueError(
                f"recovery_efficiency must be in [0, 1], got {recovery_efficiency}"
            )

        # -- 1. Installation losses --
        install_profile = self.estimate_leak_rate(
            equipment_type=equipment_type,
            lifecycle_stage=LifecycleStage.INSTALLATION.value,
            age_years=0,
            climate_zone=climate_zone,
            ldar_level=LDARLevel.NONE.value,
        )
        installation_loss_kg = (charge_kg * install_profile.effective_rate).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # -- 2. Operating losses (year-by-year with age degradation) --
        climate_factor = self.calculate_climate_factor(climate_zone)
        ldar_factor_val = self.calculate_ldar_factor(ldar_level)
        operating_base = rate_entry["lifecycle"][LifecycleStage.OPERATING.value]

        remaining_charge = charge_kg - installation_loss_kg
        total_operating_loss = Decimal("0")

        for year in range(lifetime_years):
            if remaining_charge <= Decimal("0"):
                break
            age_factor = self.calculate_age_factor(year)
            year_rate = operating_base * age_factor * climate_factor * ldar_factor_val
            if year_rate > Decimal("1"):
                year_rate = Decimal("1")
            year_loss = remaining_charge * year_rate
            total_operating_loss += year_loss
            remaining_charge -= year_loss

        total_operating_loss = total_operating_loss.quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Average annual loss for reporting
        if lifetime_years > 0:
            annual_operating_loss = (total_operating_loss / Decimal(str(lifetime_years))).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
        else:
            annual_operating_loss = Decimal("0")

        # -- 3. End-of-life losses --
        end_of_life_loss_kg = (
            remaining_charge * (Decimal("1") - recovery_efficiency)
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        # -- Compute emissions in tCO2e --
        kg_to_tonne = Decimal("0.001")

        installation_emissions = (
            installation_loss_kg * gwp * kg_to_tonne
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        operating_emissions = (
            total_operating_loss * gwp * kg_to_tonne
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        eol_emissions = (
            end_of_life_loss_kg * gwp * kg_to_tonne
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        total_loss = (
            installation_loss_kg + total_operating_loss + end_of_life_loss_kg
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        total_emissions = (
            installation_emissions + operating_emissions + eol_emissions
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        # Provenance hash
        provenance_data = {
            "equipment_type": equipment_type,
            "charge_kg": str(charge_kg),
            "gwp": str(gwp),
            "lifetime_years": lifetime_years,
            "climate_zone": climate_zone,
            "ldar_level": ldar_level,
            "recovery_efficiency": str(recovery_efficiency),
            "installation_loss_kg": str(installation_loss_kg),
            "total_operating_loss_kg": str(total_operating_loss),
            "end_of_life_loss_kg": str(end_of_life_loss_kg),
            "total_emissions_tco2e": str(total_emissions),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        timestamp = _utcnow().isoformat()

        profile = LifetimeEmissionsProfile(
            profile_id=f"le_{uuid4().hex[:12]}",
            equipment_type=equipment_type,
            charge_kg=charge_kg,
            gwp=gwp,
            lifetime_years=lifetime_years,
            installation_loss_kg=installation_loss_kg,
            installation_emissions_tco2e=installation_emissions,
            annual_operating_loss_kg=annual_operating_loss,
            total_operating_loss_kg=total_operating_loss,
            total_operating_emissions_tco2e=operating_emissions,
            end_of_life_loss_kg=end_of_life_loss_kg,
            end_of_life_emissions_tco2e=eol_emissions,
            total_lifetime_loss_kg=total_loss,
            total_lifetime_emissions_tco2e=total_emissions,
            recovery_efficiency=recovery_efficiency,
            provenance_hash=provenance_hash,
            timestamp=timestamp,
        )

        # Record provenance
        if _PROVENANCE_AVAILABLE and _get_provenance_tracker is not None:
            try:
                tracker = _get_provenance_tracker()
                tracker.record(
                    entity_type="leak_rate",
                    action="estimate_lifetime_emissions",
                    entity_id=profile.profile_id,
                    data=provenance_data,
                    metadata={"equipment_type": equipment_type},
                )
            except Exception:
                logger.debug("Provenance recording skipped", exc_info=True)

        logger.info(
            "Lifetime emissions estimated: equipment=%s charge=%.1fkg "
            "gwp=%s lifetime=%dyr total=%.3f tCO2e "
            "(install=%.3f + operating=%.3f + eol=%.3f)",
            equipment_type,
            charge_kg,
            gwp,
            lifetime_years,
            total_emissions,
            installation_emissions,
            operating_emissions,
            eol_emissions,
        )

        return profile

    # ------------------------------------------------------------------
    # Public API: Custom Rate Management
    # ------------------------------------------------------------------

    def register_custom_rate(
        self,
        equipment_type: str,
        lifecycle_stage: str,
        rate: Decimal,
    ) -> str:
        """Register a custom leak rate for a specific equipment type
        and lifecycle stage.

        Custom rates are used as the base rate when no explicit
        custom_rate argument is passed to estimate_leak_rate.
        They persist for the lifetime of this engine instance.

        Args:
            equipment_type: Equipment type string matching a
                LeakEquipmentType enum value.
            lifecycle_stage: Lifecycle stage string.
            rate: Custom leak rate as Decimal fraction (0 to 1).

        Returns:
            Registration key string (equipment_type:lifecycle_stage).

        Raises:
            ValueError: If inputs are invalid or rate is out of [0, 1].
        """
        self._validate_equipment_type(equipment_type)
        self._validate_lifecycle_stage(lifecycle_stage)

        if not isinstance(rate, Decimal):
            rate = Decimal(str(rate))
        if rate < Decimal("0") or rate > Decimal("1"):
            raise ValueError(f"rate must be in [0, 1], got {rate}")

        custom_key = f"{equipment_type}:{lifecycle_stage}"
        with self._lock:
            self._custom_rates[custom_key] = rate

        # Record provenance
        if _PROVENANCE_AVAILABLE and _get_provenance_tracker is not None:
            try:
                tracker = _get_provenance_tracker()
                tracker.record(
                    entity_type="leak_rate",
                    action="register_custom_rate",
                    entity_id=custom_key,
                    data={
                        "equipment_type": equipment_type,
                        "lifecycle_stage": lifecycle_stage,
                        "rate": str(rate),
                    },
                )
            except Exception:
                logger.debug("Provenance recording skipped", exc_info=True)

        logger.info(
            "Custom leak rate registered: %s = %.4f (%.2f%%)",
            custom_key,
            rate,
            rate * Decimal("100"),
        )

        return custom_key

    def unregister_custom_rate(
        self,
        equipment_type: str,
        lifecycle_stage: str,
    ) -> bool:
        """Remove a previously registered custom leak rate.

        Args:
            equipment_type: Equipment type string.
            lifecycle_stage: Lifecycle stage string.

        Returns:
            True if the custom rate was found and removed, False if no
            custom rate was registered for the given key.
        """
        custom_key = f"{equipment_type}:{lifecycle_stage}"
        with self._lock:
            if custom_key in self._custom_rates:
                del self._custom_rates[custom_key]
                logger.info("Custom leak rate unregistered: %s", custom_key)
                return True
        return False

    # ------------------------------------------------------------------
    # Public API: Listing and Querying
    # ------------------------------------------------------------------

    def list_rates(
        self, equipment_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List default leak rates, optionally filtered by equipment type.

        Returns a list of dictionaries, each containing the equipment
        type, default operating rate, rate range, lifecycle rates,
        typical charge, typical lifetime, source authority, and any
        registered custom overrides.

        Args:
            equipment_type: Optional equipment type to filter by. If
                None, returns rates for all equipment types.

        Returns:
            List of dictionaries with rate information.

        Raises:
            ValueError: If equipment_type is provided but not recognized.
        """
        if equipment_type is not None:
            self._validate_equipment_type(equipment_type)
            types_to_list = [equipment_type]
        else:
            types_to_list = list(_DEFAULT_LEAK_RATES.keys())

        results: List[Dict[str, Any]] = []
        for et in types_to_list:
            entry = _DEFAULT_LEAK_RATES[et]
            rate_info: Dict[str, Any] = {
                "equipment_type": et,
                "default_operating_rate": str(entry["default"]),
                "default_operating_rate_pct": str(
                    entry["default"] * Decimal("100")
                ),
                "range_low": str(entry["range"][0]),
                "range_high": str(entry["range"][1]),
                "range_low_pct": str(entry["range"][0] * Decimal("100")),
                "range_high_pct": str(entry["range"][1] * Decimal("100")),
                "lifecycle_rates": {
                    stage: str(rate)
                    for stage, rate in entry["lifecycle"].items()
                },
                "recovery_efficiency": str(entry["recovery_efficiency"]),
                "typical_charge_kg": str(entry["typical_charge_kg"]),
                "typical_lifetime_years": entry["typical_lifetime_years"],
                "source": entry["source"],
                "custom_overrides": {},
            }

            # Include any registered custom overrides
            with self._lock:
                for stage in [
                    LifecycleStage.INSTALLATION.value,
                    LifecycleStage.OPERATING.value,
                    LifecycleStage.END_OF_LIFE.value,
                ]:
                    ck = f"{et}:{stage}"
                    if ck in self._custom_rates:
                        rate_info["custom_overrides"][stage] = str(
                            self._custom_rates[ck]
                        )

            results.append(rate_info)

        return results

    def list_equipment_types(self) -> List[str]:
        """Return a sorted list of all supported equipment type strings.

        Returns:
            List of equipment type identifier strings.
        """
        return sorted(_DEFAULT_LEAK_RATES.keys())

    def list_climate_zones(self) -> List[Dict[str, str]]:
        """Return all climate zones with their adjustment factors.

        Returns:
            List of dictionaries with zone name and factor.
        """
        return [
            {"zone": zone, "factor": str(factor)}
            for zone, factor in sorted(_CLIMATE_ZONE_FACTORS.items())
        ]

    def list_ldar_levels(self) -> List[Dict[str, str]]:
        """Return all LDAR levels with their effectiveness factors.

        Returns:
            List of dictionaries with level name and factor.
        """
        return [
            {"level": level, "factor": str(factor)}
            for level, factor in sorted(_LDAR_FACTORS.items())
        ]

    def list_age_factors(self) -> List[Dict[str, Any]]:
        """Return all age bracket adjustment factors.

        Returns:
            List of dictionaries with age bracket and factor.
        """
        results: List[Dict[str, Any]] = []
        prev = 0
        for max_age, factor in _AGE_ADJUSTMENT_TABLE:
            results.append({
                "age_range": f"{prev}-{max_age - 1}",
                "min_age": prev,
                "max_age": max_age - 1,
                "factor": str(factor),
            })
            prev = max_age
        results.append({
            "age_range": f"{prev}+",
            "min_age": prev,
            "max_age": None,
            "factor": str(_AGE_ADJUSTMENT_MAX),
        })
        return results

    def get_recovery_efficiency(self, equipment_type: str) -> Decimal:
        """Get the default recovery efficiency for an equipment type.

        Args:
            equipment_type: Equipment type string.

        Returns:
            Decimal recovery efficiency (0 to 1).

        Raises:
            ValueError: If equipment_type is not recognized.
        """
        self._validate_equipment_type(equipment_type)
        return _DEFAULT_RECOVERY_EFFICIENCIES[equipment_type]

    def get_typical_charge(self, equipment_type: str) -> Decimal:
        """Get the typical refrigerant charge for an equipment type.

        Args:
            equipment_type: Equipment type string.

        Returns:
            Decimal charge in kilograms.

        Raises:
            ValueError: If equipment_type is not recognized.
        """
        self._validate_equipment_type(equipment_type)
        return _DEFAULT_LEAK_RATES[equipment_type]["typical_charge_kg"]

    def get_typical_lifetime(self, equipment_type: str) -> int:
        """Get the typical service lifetime for an equipment type.

        Args:
            equipment_type: Equipment type string.

        Returns:
            Integer lifetime in years.

        Raises:
            ValueError: If equipment_type is not recognized.
        """
        self._validate_equipment_type(equipment_type)
        return _DEFAULT_LEAK_RATES[equipment_type]["typical_lifetime_years"]

    # ------------------------------------------------------------------
    # Public API: History and Stats
    # ------------------------------------------------------------------

    def get_history(
        self,
        equipment_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[LeakRateProfile]:
        """Return leak rate estimation history.

        Args:
            equipment_type: Optional filter by equipment type.
            limit: Optional maximum number of recent entries to return.

        Returns:
            List of LeakRateProfile objects, oldest first.
        """
        with self._lock:
            entries = list(self._estimation_history)

        if equipment_type:
            entries = [e for e in entries if e.equipment_type == equipment_type]

        if limit is not None and limit > 0 and len(entries) > limit:
            entries = entries[-limit:]

        return entries

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics.

        Returns:
            Dictionary with counts and operational statistics.
        """
        with self._lock:
            history_count = len(self._estimation_history)
            custom_count = len(self._custom_rates)

            # Count by equipment type
            by_equipment: Dict[str, int] = {}
            for entry in self._estimation_history:
                by_equipment[entry.equipment_type] = (
                    by_equipment.get(entry.equipment_type, 0) + 1
                )

        return {
            "total_estimations": history_count,
            "custom_rates_registered": custom_count,
            "equipment_types_available": len(_DEFAULT_LEAK_RATES),
            "climate_zones_available": len(_CLIMATE_ZONE_FACTORS),
            "ldar_levels_available": len(_LDAR_FACTORS),
            "estimations_by_equipment": by_equipment,
        }

    def clear(self) -> None:
        """Clear all custom rates and estimation history.

        Intended for testing and engine reset scenarios.
        """
        with self._lock:
            self._custom_rates.clear()
            self._estimation_history.clear()
        logger.info("LeakRateEstimatorEngine cleared")

    # ------------------------------------------------------------------
    # Validation Helpers (Private)
    # ------------------------------------------------------------------

    def _validate_equipment_type(self, equipment_type: str) -> None:
        """Validate that equipment_type is a known type."""
        if equipment_type not in _DEFAULT_LEAK_RATES:
            valid = sorted(_DEFAULT_LEAK_RATES.keys())
            raise ValueError(
                f"Unknown equipment_type '{equipment_type}'. "
                f"Valid types: {valid}"
            )

    @staticmethod
    def _validate_lifecycle_stage(lifecycle_stage: str) -> None:
        """Validate that lifecycle_stage is a known stage."""
        valid = {e.value for e in LifecycleStage}
        if lifecycle_stage not in valid:
            raise ValueError(
                f"Unknown lifecycle_stage '{lifecycle_stage}'. "
                f"Valid stages: {sorted(valid)}"
            )

    @staticmethod
    def _validate_climate_zone(climate_zone: str) -> None:
        """Validate that climate_zone is a known zone."""
        if climate_zone not in _CLIMATE_ZONE_FACTORS:
            raise ValueError(
                f"Unknown climate_zone '{climate_zone}'. "
                f"Valid zones: {sorted(_CLIMATE_ZONE_FACTORS.keys())}"
            )

    @staticmethod
    def _validate_ldar_level(ldar_level: str) -> None:
        """Validate that ldar_level is a known level."""
        if ldar_level not in _LDAR_FACTORS:
            raise ValueError(
                f"Unknown ldar_level '{ldar_level}'. "
                f"Valid levels: {sorted(_LDAR_FACTORS.keys())}"
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        with self._lock:
            hist_count = len(self._estimation_history)
            custom_count = len(self._custom_rates)
        return (
            f"LeakRateEstimatorEngine("
            f"equipment_types={len(_DEFAULT_LEAK_RATES)}, "
            f"estimations={hist_count}, "
            f"custom_rates={custom_count})"
        )

    def __len__(self) -> int:
        """Return the number of estimations performed."""
        with self._lock:
            return len(self._estimation_history)
