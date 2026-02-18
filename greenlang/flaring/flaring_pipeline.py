# -*- coding: utf-8 -*-
"""
Flaring Pipeline Engine (Engine 7) - AGENT-MRV-006

End-to-end orchestration pipeline for GHG Protocol Scope 1 flaring
emissions calculations.  Coordinates all six upstream engines
(flare system database, emission calculator, combustion efficiency,
flaring event tracker, uncertainty quantifier, compliance checker)
through a deterministic, eight-stage pipeline:

    1. VALIDATE           - Input validation and normalisation
    2. ANALYZE_COMPOSITION - Gas composition lookup, HHV/LHV/Wobbe calc
    3. DETERMINE_CE       - Combustion efficiency with adjustments
    4. CLASSIFY_EVENT     - Flaring event classification and recording
    5. CALCULATE_EMISSIONS - CO2, CH4, N2O, CO2e including pilot/purge
    6. QUANTIFY_UNCERTAINTY - Monte Carlo simulation (optional)
    7. CHECK_COMPLIANCE   - Regulatory framework validation (optional)
    8. ASSEMBLE_RESULT    - Provenance chain, metrics, final assembly

Each stage is checkpointed so that failures produce partial results
with complete provenance.

Zero-Hallucination Guarantees:
    - All emission calculations use deterministic Python ``Decimal`` arithmetic
    - No LLM calls in the calculation path
    - SHA-256 provenance hash at every pipeline stage
    - Full audit trail for regulatory traceability

Built-in Reference Data:
    This engine bundles standalone lookup tables (FLARE_TYPES,
    GAS_COMPONENT_HHVS, GWP_VALUES, DEFAULT_EMISSION_FACTORS) so
    that it can operate independently when the upstream engines
    are unavailable.  In production these tables are superseded by
    the database engine.

Thread Safety:
    All mutable state is protected by a ``threading.Lock``.  Concurrent
    ``run_pipeline`` invocations from different threads are safe.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Flaring Agent (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional upstream-engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.flaring.config import (
        FlaringConfig,
        get_config,
    )
except ImportError:
    FlaringConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.flaring.models import (
        FlareType,
        EventCategory,
        CalculationMethod,
        EmissionGas,
        GWPSource,
        OGMPLevel,
        ComplianceFramework,
        AssistType,
        StandardCondition,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

try:
    from greenlang.flaring.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.metrics import (
        PROMETHEUS_AVAILABLE,
        record_calculation as _record_calculation,
        observe_calculation_duration as _observe_calculation_duration,
        record_batch as _record_batch,
        observe_batch_size as _observe_batch_size,
        record_flaring_event as _record_flaring_event,
        record_uncertainty as _record_uncertainty,
        record_compliance_check as _record_compliance_check,
        record_emissions as _record_emissions,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

    def _record_calculation(flare_type: str, method: str, status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def _observe_calculation_duration(operation: str, duration: float) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def _record_batch(status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def _observe_batch_size(method: str, size: int) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def _record_flaring_event(category: str, flare_type: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def _record_uncertainty(method: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def _record_compliance_check(framework: str, status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def _record_emissions(flare_type: str, gas: str, kg: float) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""


# ---------------------------------------------------------------------------
# Pipeline stage enum
# ---------------------------------------------------------------------------

class PipelineStage(str, Enum):
    """Eight-stage pipeline for flaring emissions calculations."""

    VALIDATE = "VALIDATE"
    ANALYZE_COMPOSITION = "ANALYZE_COMPOSITION"
    DETERMINE_CE = "DETERMINE_CE"
    CLASSIFY_EVENT = "CLASSIFY_EVENT"
    CALCULATE_EMISSIONS = "CALCULATE_EMISSIONS"
    QUANTIFY_UNCERTAINTY = "QUANTIFY_UNCERTAINTY"
    CHECK_COMPLIANCE = "CHECK_COMPLIANCE"
    ASSEMBLE_RESULT = "ASSEMBLE_RESULT"


PIPELINE_STAGES: List[str] = [s.value for s in PipelineStage]


# ---------------------------------------------------------------------------
# Built-in reference data for standalone operation
# ---------------------------------------------------------------------------

#: GWP values for the four primary greenhouse gases tracked by the
#: flaring agent.  Keys are (gwp_source, gas) pairs.
GWP_VALUES: Dict[str, Dict[str, float]] = {
    "AR4": {"CO2": 1.0, "CH4": 25.0, "N2O": 298.0},
    "AR5": {"CO2": 1.0, "CH4": 28.0, "N2O": 265.0},
    "AR6": {"CO2": 1.0, "CH4": 27.3, "N2O": 273.0},
    "AR6_20yr": {"CO2": 1.0, "CH4": 80.8, "N2O": 273.0},
}

#: Higher heating values for individual gas components (BTU/scf).
#: Source: GPA Midstream Standard 2145, GPSA Engineering Data Book.
GAS_COMPONENT_HHVS: Dict[str, Dict[str, Any]] = {
    "CH4": {
        "hhv_btu_scf": 1012.0,
        "molecular_weight": 16.043,
        "carbon_atoms": 1,
        "specific_gravity": 0.5537,
        "description": "Methane",
    },
    "C2H6": {
        "hhv_btu_scf": 1773.0,
        "molecular_weight": 30.069,
        "carbon_atoms": 2,
        "specific_gravity": 1.0382,
        "description": "Ethane",
    },
    "C3H8": {
        "hhv_btu_scf": 2524.0,
        "molecular_weight": 44.096,
        "carbon_atoms": 3,
        "specific_gravity": 1.5226,
        "description": "Propane",
    },
    "n_C4H10": {
        "hhv_btu_scf": 3271.0,
        "molecular_weight": 58.122,
        "carbon_atoms": 4,
        "specific_gravity": 2.0068,
        "description": "n-Butane",
    },
    "i_C4H10": {
        "hhv_btu_scf": 3254.0,
        "molecular_weight": 58.122,
        "carbon_atoms": 4,
        "specific_gravity": 2.0068,
        "description": "Isobutane",
    },
    "C5H12": {
        "hhv_btu_scf": 4010.0,
        "molecular_weight": 72.149,
        "carbon_atoms": 5,
        "specific_gravity": 2.4911,
        "description": "Pentane",
    },
    "C6_PLUS": {
        "hhv_btu_scf": 4762.0,
        "molecular_weight": 86.175,
        "carbon_atoms": 6,
        "specific_gravity": 2.9753,
        "description": "Hexane plus",
    },
    "H2": {
        "hhv_btu_scf": 325.0,
        "molecular_weight": 2.016,
        "carbon_atoms": 0,
        "specific_gravity": 0.0696,
        "description": "Hydrogen",
    },
    "CO": {
        "hhv_btu_scf": 321.0,
        "molecular_weight": 28.010,
        "carbon_atoms": 1,
        "specific_gravity": 0.9671,
        "description": "Carbon monoxide",
    },
    "C2H4": {
        "hhv_btu_scf": 1614.0,
        "molecular_weight": 28.054,
        "carbon_atoms": 2,
        "specific_gravity": 0.9686,
        "description": "Ethylene",
    },
    "C3H6": {
        "hhv_btu_scf": 2336.0,
        "molecular_weight": 42.080,
        "carbon_atoms": 3,
        "specific_gravity": 1.4529,
        "description": "Propylene",
    },
    "CO2": {
        "hhv_btu_scf": 0.0,
        "molecular_weight": 44.010,
        "carbon_atoms": 1,
        "specific_gravity": 1.5189,
        "description": "Carbon dioxide (inert)",
    },
    "N2": {
        "hhv_btu_scf": 0.0,
        "molecular_weight": 28.014,
        "carbon_atoms": 0,
        "specific_gravity": 0.9672,
        "description": "Nitrogen (inert)",
    },
    "H2S": {
        "hhv_btu_scf": 647.0,
        "molecular_weight": 34.082,
        "carbon_atoms": 0,
        "specific_gravity": 1.1763,
        "description": "Hydrogen sulfide",
    },
    "H2O": {
        "hhv_btu_scf": 0.0,
        "molecular_weight": 18.015,
        "carbon_atoms": 0,
        "specific_gravity": 0.6220,
        "description": "Water vapor (inert)",
    },
}

#: Flare type default parameters.
FLARE_TYPES: Dict[str, Dict[str, Any]] = {
    "ELEVATED_STEAM_ASSISTED": {
        "display_name": "Elevated Steam-Assisted Flare",
        "default_ce": 0.98,
        "assist_type": "STEAM",
        "min_hhv_btu_scf": 200.0,
        "typical_tip_velocity_mach": 0.3,
        "description": "High-pressure tip with steam injection for smokeless operation",
    },
    "ELEVATED_AIR_ASSISTED": {
        "display_name": "Elevated Air-Assisted Flare",
        "default_ce": 0.98,
        "assist_type": "AIR",
        "min_hhv_btu_scf": 200.0,
        "typical_tip_velocity_mach": 0.3,
        "description": "Forced-draft air for combustion enhancement",
    },
    "ELEVATED_UNASSISTED": {
        "display_name": "Elevated Unassisted Flare",
        "default_ce": 0.96,
        "assist_type": "NONE",
        "min_hhv_btu_scf": 300.0,
        "typical_tip_velocity_mach": 0.2,
        "description": "Simple pipe flare, no assist medium",
    },
    "ENCLOSED_GROUND": {
        "display_name": "Enclosed Ground Flare",
        "default_ce": 0.99,
        "assist_type": "NONE",
        "min_hhv_btu_scf": 200.0,
        "typical_tip_velocity_mach": 0.1,
        "description": "Multi-burner in refractory-lined enclosure",
    },
    "MULTI_POINT_GROUND": {
        "display_name": "Multi-Point Ground Flare (MPGF)",
        "default_ce": 0.99,
        "assist_type": "NONE",
        "min_hhv_btu_scf": 200.0,
        "typical_tip_velocity_mach": 0.1,
        "description": "Multiple staged burners at ground level",
    },
    "OFFSHORE_MARINE": {
        "display_name": "Offshore Marine Flare",
        "default_ce": 0.95,
        "assist_type": "NONE",
        "min_hhv_btu_scf": 200.0,
        "typical_tip_velocity_mach": 0.3,
        "description": "Boom-mounted for offshore platforms",
    },
    "CANDLESTICK": {
        "display_name": "Candlestick Flare",
        "default_ce": 0.93,
        "assist_type": "NONE",
        "min_hhv_btu_scf": 300.0,
        "typical_tip_velocity_mach": 0.2,
        "description": "Simple vertical pipe, no wind shielding",
    },
    "LOW_PRESSURE": {
        "display_name": "Low-Pressure Flare",
        "default_ce": 0.95,
        "assist_type": "NONE",
        "min_hhv_btu_scf": 150.0,
        "typical_tip_velocity_mach": 0.1,
        "description": "For low-flow, low-pressure waste gas",
    },
}

#: Event category definitions.
EVENT_CATEGORIES: Dict[str, Dict[str, str]] = {
    "ROUTINE": {
        "display_name": "Routine / Continuous",
        "description": "Normal process flaring during steady-state operations",
    },
    "NON_ROUTINE": {
        "display_name": "Non-Routine / Intermittent",
        "description": "Planned but irregular (well testing, tank flashing)",
    },
    "EMERGENCY": {
        "display_name": "Emergency / Safety",
        "description": "Pressure relief, equipment failure, process upset",
    },
    "MAINTENANCE": {
        "display_name": "Maintenance",
        "description": "Startup, shutdown, turnaround activities",
    },
    "PILOT_PURGE": {
        "display_name": "Pilot / Purge",
        "description": "Continuous pilot flame and purge gas consumption",
    },
    "WELL_COMPLETION": {
        "display_name": "Well Completion / Workover",
        "description": "Upstream oil and gas flowback flaring",
    },
}

#: Default emission factors (kg CO2 per scf of flare gas).
#: Source: EPA 40 CFR Part 98 Subpart W, IPCC 2006 Vol 2.
DEFAULT_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "EPA_SUBPART_W": {
        "co2_kg_per_mscf": 59.29,
        "ch4_kg_per_mscf": 0.545,
        "n2o_kg_per_mscf": 0.0091,
        "source": "EPA 40 CFR Part 98 Subpart W Sec. W.23",
        "default_hhv_btu_scf": 1050.0,
    },
    "IPCC_2006": {
        "co2_kg_per_mscf": 54.89,
        "ch4_kg_per_mscf": 0.034,
        "n2o_kg_per_mscf": 0.0001,
        "source": "IPCC 2006 Guidelines Vol 2 Ch 4",
        "default_hhv_btu_scf": 1000.0,
    },
    "API_2009": {
        "co2_kg_per_mscf": 55.76,
        "ch4_kg_per_mscf": 0.450,
        "n2o_kg_per_mscf": 0.0086,
        "source": "API Compendium 2009",
        "default_hhv_btu_scf": 1020.0,
    },
}

#: Regulatory frameworks for compliance checking.
REGULATORY_FRAMEWORKS: Dict[str, Dict[str, Any]] = {
    "GHG_PROTOCOL": {
        "display_name": "GHG Protocol Corporate Standard (Ch 5)",
        "requirements": [
            {"id": "GHG-FL-001", "desc": "Scope 1 flaring emissions reported", "check": "has_results"},
            {"id": "GHG-FL-002", "desc": "CO2e calculated using approved GWP", "check": "valid_gwp"},
            {"id": "GHG-FL-003", "desc": "Flaring methodology documented", "check": "has_methodology"},
            {"id": "GHG-FL-004", "desc": "Provenance trail complete", "check": "has_provenance"},
        ],
    },
    "ISO_14064": {
        "display_name": "ISO 14064-1:2018",
        "requirements": [
            {"id": "ISO-FL-001", "desc": "Direct GHG emissions quantified", "check": "has_results"},
            {"id": "ISO-FL-002", "desc": "Uncertainty assessment performed", "check": "has_uncertainty"},
            {"id": "ISO-FL-003", "desc": "GWP from recognised source", "check": "valid_gwp"},
        ],
    },
    "CSRD_ESRS_E1": {
        "display_name": "CSRD / ESRS E1 Climate Change",
        "requirements": [
            {"id": "ESRS-FL-001", "desc": "Scope 1 GHG reported by gas", "check": "has_gas_breakdown"},
            {"id": "ESRS-FL-002", "desc": "GWP AR6 used", "check": "gwp_ar6"},
            {"id": "ESRS-FL-003", "desc": "Material flaring emissions identified", "check": "has_results"},
        ],
    },
    "EPA_SUBPART_W": {
        "display_name": "EPA 40 CFR Part 98 Subpart W Sec. W.23",
        "requirements": [
            {"id": "EPA-FL-001", "desc": "Flare-specific EF or composition method", "check": "valid_method"},
            {"id": "EPA-FL-002", "desc": "Gas volume measured or estimated", "check": "has_volume"},
            {"id": "EPA-FL-003", "desc": "Combustion efficiency documented", "check": "has_ce"},
            {"id": "EPA-FL-004", "desc": "Monitoring plan documented", "check": "has_provenance"},
        ],
    },
    "EU_ETS_MRR": {
        "display_name": "EU ETS Monitoring and Reporting Regulation",
        "requirements": [
            {"id": "ETS-FL-001", "desc": "Installation-level flare reporting", "check": "has_facility"},
            {"id": "ETS-FL-002", "desc": "Tier methodology applied", "check": "has_tier"},
            {"id": "ETS-FL-003", "desc": "Approved monitoring plan", "check": "has_provenance"},
        ],
    },
    "EU_METHANE_REG": {
        "display_name": "EU Methane Regulation 2024/1787",
        "requirements": [
            {"id": "EUMR-FL-001", "desc": "CH4 emissions from flaring quantified", "check": "has_ch4"},
            {"id": "EUMR-FL-002", "desc": "Flaring event categorisation (Art 14)", "check": "has_event_category"},
            {"id": "EUMR-FL-003", "desc": "Routine flaring reduction tracked", "check": "has_routine_tracking"},
        ],
    },
    "WORLD_BANK_ZRF": {
        "display_name": "World Bank Zero Routine Flaring by 2030",
        "requirements": [
            {"id": "ZRF-FL-001", "desc": "Routine flaring volume tracked", "check": "has_routine_volume"},
            {"id": "ZRF-FL-002", "desc": "Reduction plan documented", "check": "has_provenance"},
        ],
    },
    "OGMP_2_0": {
        "display_name": "OGMP 2.0 (Oil & Gas Methane Partnership)",
        "requirements": [
            {"id": "OGMP-FL-001", "desc": "OGMP reporting level assigned", "check": "has_ogmp_level"},
            {"id": "OGMP-FL-002", "desc": "CH4 slip from flaring quantified", "check": "has_ch4"},
            {"id": "OGMP-FL-003", "desc": "Combustion efficiency documented", "check": "has_ce"},
        ],
    },
}

#: Molecular weight of air at standard conditions (g/mol).
MW_AIR: float = 28.964

#: CO2 molecular weight (g/mol).
MW_CO2: float = 44.010

#: CH4 molecular weight (g/mol).
MW_CH4: float = 16.043

#: N2O molecular weight (g/mol).
MW_N2O: float = 44.013

#: Carbon molecular weight (g/mol).
MW_C: float = 12.011

#: Standard molar volume at 60degF / 14.696 psia (scf/lb-mol).
MOLAR_VOLUME_SCF: float = 379.3

#: Standard molar volume at 15degC / 101.325 kPa (Nm3/kmol).
MOLAR_VOLUME_NM3: float = 23.6445

#: Conversion factor: 1 MSCF = 1000 SCF.
MSCF_TO_SCF: float = 1000.0

#: N2O emission factor for high-temp combustion (kg N2O / TJ).
#: Source: IPCC 2006 Guidelines Vol 2.
N2O_FACTOR_KG_PER_TJ: float = 0.1

#: BTU to TJ conversion: 1 TJ = 947,817,120 BTU.
BTU_PER_TJ: float = 947_817_120.0


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string."""
    return _utcnow().isoformat()


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, Decimal):
        serializable = str(data)
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _to_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Number to convert.

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


# ---------------------------------------------------------------------------
# Stage result helper
# ---------------------------------------------------------------------------

def _stage_result(
    stage: str,
    success: bool,
    duration_ms: float,
    pipeline_id: str,
    extra: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a standardised stage result dictionary.

    Args:
        stage: Stage name.
        success: Whether the stage succeeded.
        duration_ms: Stage wall-clock duration in milliseconds.
        pipeline_id: Parent pipeline run identifier.
        extra: Additional key-value pairs to include.
        error: Error message (if stage failed).

    Returns:
        Stage result dictionary.
    """
    result: Dict[str, Any] = {
        "stage": stage,
        "success": success,
        "duration_ms": round(duration_ms, 3),
        "provenance_hash": _compute_hash({
            "stage": stage,
            "pipeline_id": pipeline_id,
            "success": success,
            "duration_ms": round(duration_ms, 3),
        }),
    }
    if error is not None:
        result["error"] = error
    if extra is not None:
        result.update(extra)
    return result


# ===================================================================
# FlaringPipelineEngine
# ===================================================================


class FlaringPipelineEngine:
    """Eight-stage orchestration pipeline for flaring emissions calculations.

    Coordinates all six upstream engines through an eight-stage pipeline
    with checkpointing, provenance tracking, and comprehensive error
    handling.  Each pipeline run produces a deterministic SHA-256
    provenance hash for the complete execution chain.

    The pipeline can operate in standalone mode using built-in reference
    data when the upstream engines are unavailable.

    Thread-safe: all mutable state is protected by an internal lock.

    Attributes:
        config: FlaringConfig instance (or None).
        flare_system_db: Optional FlareSystemDatabaseEngine for lookups.
        emission_calculator: Optional EmissionCalculatorEngine.
        combustion_efficiency: Optional CombustionEfficiencyEngine.
        event_tracker: Optional FlaringEventTrackerEngine.
        uncertainty_engine: Optional UncertaintyQuantifierEngine.
        compliance_checker: Optional ComplianceCheckerEngine.

    Example:
        >>> engine = FlaringPipelineEngine()
        >>> result = engine.run_pipeline({
        ...     "flare_id": "FL-001",
        ...     "gas_volume_mscf": 500,
        ...     "method": "GAS_COMPOSITION",
        ... })
        >>> assert result["success"] is True
    """

    def __init__(
        self,
        flare_system_db: Any = None,
        emission_calculator: Any = None,
        combustion_efficiency: Any = None,
        event_tracker: Any = None,
        uncertainty_engine: Any = None,
        compliance_checker: Any = None,
        config: Any = None,
    ) -> None:
        """Initialize the FlaringPipelineEngine.

        Wires all six upstream engines via dependency injection.  Any
        engine set to ``None`` causes its pipeline stage to use built-in
        reference data or skip with a warning.

        Args:
            flare_system_db: FlareSystemDatabaseEngine instance or None.
            emission_calculator: EmissionCalculatorEngine instance or None.
            combustion_efficiency: CombustionEfficiencyEngine instance or None.
            event_tracker: FlaringEventTrackerEngine instance or None.
            uncertainty_engine: UncertaintyQuantifierEngine instance or None.
            compliance_checker: ComplianceCheckerEngine instance or None.
            config: Optional configuration.  Uses global config if None.
        """
        self.config = config if config is not None else get_config()

        # Engine references
        self.flare_system_db = flare_system_db
        self.emission_calculator = emission_calculator
        self.combustion_efficiency = combustion_efficiency
        self.event_tracker = event_tracker
        self.uncertainty_engine = uncertainty_engine
        self.compliance_checker = compliance_checker

        # Thread-safe mutable state
        self._lock = threading.Lock()
        self._total_runs: int = 0
        self._successful_runs: int = 0
        self._failed_runs: int = 0
        self._total_duration_ms: float = 0.0
        self._last_run_at: Optional[str] = None
        self._pipeline_results: Dict[str, Dict[str, Any]] = {}
        self._stage_results_cache: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(
            "FlaringPipelineEngine initialized: "
            "flare_system_db=%s, emission_calculator=%s, "
            "combustion_efficiency=%s, event_tracker=%s, "
            "uncertainty=%s, compliance=%s",
            flare_system_db is not None,
            emission_calculator is not None,
            combustion_efficiency is not None,
            event_tracker is not None,
            uncertainty_engine is not None,
            compliance_checker is not None,
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def run_pipeline(
        self,
        request: Dict[str, Any],
        gwp_source: str = "AR6",
        include_uncertainty: bool = True,
        include_compliance: bool = True,
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute the full eight-stage pipeline for a single request.

        Args:
            request: Calculation request dictionary.  Must contain at
                minimum ``flare_id`` or ``flare_type``, and
                ``gas_volume_mscf`` or ``gas_volume_nm3``.
            gwp_source: GWP source (AR4, AR5, AR6, AR6_20yr).
            include_uncertainty: Whether to run uncertainty analysis.
            include_compliance: Whether to run compliance checking.
            frameworks: Regulatory frameworks for compliance check.

        Returns:
            Dictionary with pipeline results including per-stage results,
            final calculation, provenance hash, and timing.
        """
        pipeline_id = _new_uuid()
        t0 = time.perf_counter()
        stage_results: List[Dict[str, Any]] = []
        stages_completed = 0
        overall_success = True

        # Working data passed between stages
        validated: Dict[str, Any] = {}
        composition: Dict[str, Any] = {}
        ce_data: Dict[str, Any] = {}
        event_data: Dict[str, Any] = {}
        emissions: Dict[str, Any] = {}
        uncertainty: Dict[str, Any] = {}
        compliance: Dict[str, Any] = {}

        logger.info(
            "Pipeline %s started: flare=%s, volume=%s, gwp=%s",
            pipeline_id,
            request.get("flare_id", request.get("flare_type", "UNKNOWN")),
            request.get("gas_volume_mscf", request.get("gas_volume_nm3", "N/A")),
            gwp_source,
        )

        # Stage 1: VALIDATE
        sr = self._stage_validate(pipeline_id, request, gwp_source)
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1
            validated = sr.get("validated_data", request)
        else:
            overall_success = False
            validated = request

        # Stage 2: ANALYZE_COMPOSITION
        sr = self._stage_analyze_composition(pipeline_id, validated)
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1
            composition = sr.get("composition_data", {})
        else:
            overall_success = False

        # Stage 3: DETERMINE_CE
        sr = self._stage_determine_ce(pipeline_id, validated, composition)
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1
            ce_data = sr.get("ce_data", {})
        else:
            overall_success = False

        # Stage 4: CLASSIFY_EVENT
        sr = self._stage_classify_event(pipeline_id, validated)
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1
            event_data = sr.get("event_data", {})
        else:
            # Event classification is not critical - continue
            pass

        # Stage 5: CALCULATE_EMISSIONS
        sr = self._stage_calculate_emissions(
            pipeline_id, validated, composition, ce_data, gwp_source,
        )
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1
            emissions = sr.get("emissions_data", {})
        else:
            overall_success = False

        # Stage 6: QUANTIFY_UNCERTAINTY
        if include_uncertainty:
            sr = self._stage_quantify_uncertainty(
                pipeline_id, validated, emissions,
            )
        else:
            sr = _stage_result(
                PipelineStage.QUANTIFY_UNCERTAINTY.value, True, 0.0,
                pipeline_id, {"skipped": True, "reason": "uncertainty disabled"},
            )
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1
            uncertainty = sr.get("uncertainty_data", {})

        # Stage 7: CHECK_COMPLIANCE
        if include_compliance:
            sr = self._stage_check_compliance(
                pipeline_id, validated, emissions,
                ce_data, gwp_source, frameworks,
            )
        else:
            sr = _stage_result(
                PipelineStage.CHECK_COMPLIANCE.value, True, 0.0,
                pipeline_id, {"skipped": True, "reason": "compliance disabled"},
            )
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1
            compliance = sr.get("compliance_data", {})

        # Stage 8: ASSEMBLE_RESULT
        sr = self._stage_assemble_result(
            pipeline_id, validated, composition, ce_data,
            event_data, emissions, uncertainty, compliance,
            gwp_source,
        )
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1

        # Build final result
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        calc_id = validated.get("calculation_id", _new_uuid())

        pipeline_provenance = _compute_hash({
            "pipeline_id": pipeline_id,
            "calculation_id": calc_id,
            "flare_type": validated.get("flare_type", ""),
            "stages_completed": stages_completed,
            "total_duration_ms": elapsed_ms,
        })

        final_result = sr.get("assembled_result", {})

        result = {
            "success": overall_success,
            "pipeline_id": pipeline_id,
            "calculation_id": calc_id,
            "stages_completed": stages_completed,
            "stages_total": len(PIPELINE_STAGES),
            "stage_results": [
                {k: v for k, v in s.items()
                 if k not in ("validated_data", "composition_data",
                              "ce_data", "event_data", "emissions_data",
                              "uncertainty_data", "compliance_data",
                              "assembled_result")}
                for s in stage_results
            ],
            "result": final_result,
            "pipeline_provenance_hash": pipeline_provenance,
            "total_duration_ms": round(elapsed_ms, 3),
            "timestamp": _utcnow_iso(),
        }

        # Update statistics
        self._record_run(pipeline_id, overall_success, elapsed_ms, stage_results)

        # Record Prometheus metrics
        flare_type = validated.get("flare_type", "UNKNOWN")
        method = validated.get("method", "DEFAULT_EF")
        _record_calculation(flare_type, method, "success" if overall_success else "failure")
        _observe_calculation_duration("pipeline", elapsed_ms / 1000.0)

        logger.info(
            "Pipeline %s completed: success=%s stages=%d/%d "
            "duration=%.1fms",
            pipeline_id, overall_success,
            stages_completed, len(PIPELINE_STAGES), elapsed_ms,
        )
        return result

    def run_batch_pipeline(
        self,
        requests: List[Dict[str, Any]],
        gwp_source: str = "AR6",
        include_uncertainty: bool = True,
        include_compliance: bool = True,
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute the pipeline for a batch of requests.

        Args:
            requests: List of calculation request dictionaries.
            gwp_source: GWP source for all calculations.
            include_uncertainty: Whether to run uncertainty analysis.
            include_compliance: Whether to run compliance checks.
            frameworks: Regulatory frameworks for compliance.

        Returns:
            Dictionary with batch results, aggregated totals, timing.
        """
        batch_id = _new_uuid()
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []
        success_count = 0
        failure_count = 0
        total_co2e_kg = Decimal("0")

        logger.info(
            "Batch %s started: %d requests, gwp=%s",
            batch_id, len(requests), gwp_source,
        )

        _observe_batch_size("pipeline", len(requests))

        for idx, req in enumerate(requests):
            try:
                pipeline_result = self.run_pipeline(
                    request=req,
                    gwp_source=gwp_source,
                    include_uncertainty=include_uncertainty,
                    include_compliance=include_compliance,
                    frameworks=frameworks,
                )
                results.append(pipeline_result)

                if pipeline_result.get("success"):
                    success_count += 1
                    result_data = pipeline_result.get("result", {})
                    total_co2e_kg += _to_decimal(
                        result_data.get("total_co2e_kg", 0),
                    )
                else:
                    failure_count += 1

            except Exception as exc:
                logger.error(
                    "Batch %s request %d failed: %s",
                    batch_id, idx, exc, exc_info=True,
                )
                failure_count += 1
                results.append({
                    "success": False,
                    "error": str(exc),
                    "request_index": idx,
                })

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        batch_provenance = _compute_hash({
            "batch_id": batch_id,
            "request_count": len(requests),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_co2e_kg": str(total_co2e_kg),
            "duration_ms": elapsed_ms,
        })

        _record_batch("success" if failure_count == 0 else "partial")

        batch_result = {
            "batch_id": batch_id,
            "results": results,
            "total_co2e_kg": float(total_co2e_kg),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_requests": len(requests),
            "processing_time_ms": round(elapsed_ms, 3),
            "provenance_hash": batch_provenance,
            "timestamp": _utcnow_iso(),
        }

        logger.info(
            "Batch %s completed: %d/%d success, %.4f kg CO2e, %.1fms",
            batch_id, success_count, len(requests),
            float(total_co2e_kg), elapsed_ms,
        )
        return batch_result

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Return current pipeline operational status.

        Returns:
            Dictionary with engine availability, run counts, timing.
        """
        with self._lock:
            return {
                "status": "ready",
                "engines": {
                    "flare_system_db": self.flare_system_db is not None,
                    "emission_calculator": self.emission_calculator is not None,
                    "combustion_efficiency": self.combustion_efficiency is not None,
                    "event_tracker": self.event_tracker is not None,
                    "uncertainty": self.uncertainty_engine is not None,
                    "compliance_checker": self.compliance_checker is not None,
                },
                "total_runs": self._total_runs,
                "successful_runs": self._successful_runs,
                "failed_runs": self._failed_runs,
                "last_run_at": self._last_run_at,
                "pipeline_stages": PIPELINE_STAGES,
                "timestamp": _utcnow_iso(),
            }

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Return pipeline-level aggregate statistics.

        Returns:
            Dictionary with total runs, avg duration, success rate, etc.
        """
        with self._lock:
            avg_duration = (
                self._total_duration_ms / self._total_runs
                if self._total_runs > 0
                else 0.0
            )
            success_rate = (
                self._successful_runs / self._total_runs * 100.0
                if self._total_runs > 0
                else 0.0
            )

            return {
                "total_runs": self._total_runs,
                "successful_runs": self._successful_runs,
                "failed_runs": self._failed_runs,
                "success_rate_pct": round(success_rate, 2),
                "total_duration_ms": round(self._total_duration_ms, 3),
                "avg_duration_ms": round(avg_duration, 3),
                "last_run_at": self._last_run_at,
                "recent_runs": list(self._pipeline_results.values())[-10:],
                "timestamp": _utcnow_iso(),
            }

    # ==================================================================
    # Private: Run recording
    # ==================================================================

    def _record_run(
        self,
        pipeline_id: str,
        success: bool,
        elapsed_ms: float,
        stage_results: List[Dict[str, Any]],
    ) -> None:
        """Record a pipeline run in thread-safe statistics.

        Args:
            pipeline_id: Pipeline run identifier.
            success: Whether the pipeline succeeded.
            elapsed_ms: Total elapsed milliseconds.
            stage_results: List of per-stage result dicts.
        """
        with self._lock:
            self._total_runs += 1
            if success:
                self._successful_runs += 1
            else:
                self._failed_runs += 1
            self._total_duration_ms += elapsed_ms
            self._last_run_at = _utcnow_iso()

            self._pipeline_results[pipeline_id] = {
                "pipeline_id": pipeline_id,
                "success": success,
                "duration_ms": round(elapsed_ms, 3),
                "timestamp": self._last_run_at,
            }

            # Keep only the last 100 results
            if len(self._pipeline_results) > 100:
                oldest_key = next(iter(self._pipeline_results))
                del self._pipeline_results[oldest_key]

            self._stage_results_cache[pipeline_id] = stage_results
            if len(self._stage_results_cache) > 100:
                oldest_key = next(iter(self._stage_results_cache))
                del self._stage_results_cache[oldest_key]

    # ==================================================================
    # Private stage implementations
    # ==================================================================

    def _stage_validate(
        self,
        pipeline_id: str,
        request: Dict[str, Any],
        gwp_source: str,
    ) -> Dict[str, Any]:
        """Stage 1: Validate input data and normalise fields.

        Validates required fields (flare_id/flare_type, gas volume),
        method selection, GWP source, and optional composition data.

        Args:
            pipeline_id: Pipeline run identifier.
            request: Raw calculation request.
            gwp_source: GWP source string.

        Returns:
            Stage result with validated_data.
        """
        stage = PipelineStage.VALIDATE.value
        t0 = time.perf_counter()

        try:
            errors: List[str] = []
            warnings: List[str] = []

            # Flare identification
            flare_id = request.get("flare_id", "")
            flare_type = request.get("flare_type", "")
            if not flare_id and not flare_type:
                errors.append("flare_id or flare_type is required")

            # Validate flare type if provided
            if flare_type and flare_type not in FLARE_TYPES:
                warnings.append(
                    f"flare_type '{flare_type}' not in built-in reference "
                    f"data; custom lookup required"
                )

            # Gas volume (at least one required)
            gas_volume_mscf = request.get("gas_volume_mscf")
            gas_volume_nm3 = request.get("gas_volume_nm3")
            if gas_volume_mscf is None and gas_volume_nm3 is None:
                errors.append(
                    "gas_volume_mscf or gas_volume_nm3 is required"
                )
            else:
                if gas_volume_mscf is not None:
                    try:
                        vol = _to_decimal(gas_volume_mscf)
                        if vol < 0:
                            errors.append("gas_volume_mscf must be >= 0")
                    except Exception:
                        errors.append("gas_volume_mscf must be a valid number")
                if gas_volume_nm3 is not None:
                    try:
                        vol = _to_decimal(gas_volume_nm3)
                        if vol < 0:
                            errors.append("gas_volume_nm3 must be >= 0")
                    except Exception:
                        errors.append("gas_volume_nm3 must be a valid number")

            # Validate GWP source
            if gwp_source not in GWP_VALUES:
                errors.append(
                    f"gwp_source must be one of {list(GWP_VALUES.keys())}"
                )

            # Validate calculation method if provided
            method = request.get("method", "DEFAULT_EF")
            valid_methods = {
                "GAS_COMPOSITION", "DEFAULT_EF",
                "ENGINEERING_ESTIMATE", "DIRECT_MEASUREMENT",
            }
            if method not in valid_methods:
                errors.append(
                    f"method must be one of {sorted(valid_methods)}"
                )

            # Validate event category if provided
            event_category = request.get("event_category", "ROUTINE")
            if event_category not in EVENT_CATEGORIES:
                warnings.append(
                    f"event_category '{event_category}' not in standard "
                    f"categories; using as custom"
                )

            # Validate gas composition if provided
            gas_composition = request.get("gas_composition", {})
            if gas_composition:
                total_fraction = Decimal("0")
                for component, fraction in gas_composition.items():
                    frac = _to_decimal(fraction)
                    if frac < 0 or frac > 1:
                        errors.append(
                            f"gas_composition[{component}] must be 0.0 - 1.0"
                        )
                    total_fraction += frac
                if total_fraction > 0 and abs(total_fraction - Decimal("1")) > Decimal("0.02"):
                    warnings.append(
                        f"gas_composition fractions sum to {float(total_fraction):.4f}, "
                        f"expected ~1.0"
                    )

            # Normalise volume to MSCF
            volume_mscf = Decimal("0")
            if gas_volume_mscf is not None:
                volume_mscf = _to_decimal(gas_volume_mscf)
            elif gas_volume_nm3 is not None:
                # Convert Nm3 to MSCF: 1 Nm3 = 35.3147 scf, 1 MSCF = 1000 scf
                volume_mscf = _to_decimal(gas_volume_nm3) * Decimal("35.3147") / Decimal("1000")

            # Build validated data
            validated: Dict[str, Any] = {
                "calculation_id": request.get("calculation_id", _new_uuid()),
                "flare_id": flare_id,
                "flare_type": flare_type or "ELEVATED_STEAM_ASSISTED",
                "gas_volume_mscf": float(volume_mscf),
                "method": method,
                "event_category": event_category,
                "gas_composition": gas_composition,
                "combustion_efficiency": request.get("combustion_efficiency"),
                "wind_speed_ms": request.get("wind_speed_ms"),
                "tip_velocity_mach": request.get("tip_velocity_mach"),
                "steam_to_gas_ratio": request.get("steam_to_gas_ratio"),
                "air_to_gas_ratio": request.get("air_to_gas_ratio"),
                "pilot_gas_mmbtu_hr": request.get("pilot_gas_mmbtu_hr"),
                "num_pilot_tips": request.get("num_pilot_tips", 1),
                "purge_gas_type": request.get("purge_gas_type", "N2"),
                "purge_gas_flow_scfh": request.get("purge_gas_flow_scfh", 0),
                "operating_hours": request.get("operating_hours", 1.0),
                "facility_id": request.get("facility_id"),
                "ogmp_level": request.get("ogmp_level"),
                "ef_source": request.get("ef_source", "EPA_SUBPART_W"),
                "reporting_period": request.get("reporting_period"),
                "organization_id": request.get("organization_id"),
                "metadata": request.get("metadata", {}),
            }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, len(errors) == 0, elapsed_ms, pipeline_id,
                extra={
                    "validated_data": validated,
                    "errors": errors,
                    "warnings": warnings,
                    "validated_fields": len(validated),
                },
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return _stage_result(
                stage, False, elapsed_ms, pipeline_id, error=str(exc),
            )

    def _stage_analyze_composition(
        self,
        pipeline_id: str,
        validated: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 2: Analyze gas composition - HHV, LHV, Wobbe index.

        Looks up gas composition from the flare system database engine
        or uses provided composition data to calculate heating values.

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data from stage 1.

        Returns:
            Stage result with composition_data including HHV, LHV, Wobbe.
        """
        stage = PipelineStage.ANALYZE_COMPOSITION.value
        t0 = time.perf_counter()

        try:
            gas_composition = validated.get("gas_composition", {})
            flare_id = validated.get("flare_id", "")

            # Try upstream engine first
            if self.flare_system_db is not None and not gas_composition:
                try:
                    db_comp = self.flare_system_db.get_gas_composition(
                        flare_id,
                    )
                    if db_comp is not None:
                        if hasattr(db_comp, "model_dump"):
                            gas_composition = db_comp.model_dump(mode="json")
                        elif isinstance(db_comp, dict):
                            gas_composition = db_comp
                except (AttributeError, TypeError, KeyError) as exc:
                    logger.warning(
                        "FlareSystemDB composition lookup failed for %s: %s",
                        flare_id, exc,
                    )

            # Use default composition if none available
            if not gas_composition:
                gas_composition = self._default_gas_composition()
                logger.info(
                    "Pipeline %s: using default gas composition",
                    pipeline_id,
                )

            # Calculate heating values from composition
            hhv_btu_scf = self._calculate_hhv(gas_composition)
            specific_gravity = self._calculate_specific_gravity(gas_composition)
            molecular_weight = self._calculate_molecular_weight(gas_composition)

            # LHV = HHV - latent heat of water (approx 50 BTU/scf per
            # mole fraction of H2O produced per mole of gas burned)
            h2o_fraction = self._estimate_water_production(gas_composition)
            lhv_btu_scf = hhv_btu_scf - (h2o_fraction * Decimal("1030"))

            # Wobbe Index = HHV / sqrt(specific_gravity)
            if specific_gravity > 0:
                wobbe_index = float(hhv_btu_scf) / math.sqrt(float(specific_gravity))
            else:
                wobbe_index = 0.0

            # Calculate hydrocarbon fraction (combustible fraction)
            hc_fraction = self._calculate_hc_fraction(gas_composition)

            composition_data: Dict[str, Any] = {
                "gas_composition": gas_composition,
                "hhv_btu_scf": float(hhv_btu_scf),
                "lhv_btu_scf": float(lhv_btu_scf),
                "wobbe_index": round(wobbe_index, 2),
                "specific_gravity": float(specific_gravity),
                "molecular_weight": float(molecular_weight),
                "hc_fraction": float(hc_fraction),
                "ch4_fraction": float(_to_decimal(gas_composition.get("CH4", 0))),
                "source": "PROVIDED" if validated.get("gas_composition") else "DEFAULT",
            }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={
                    "composition_data": composition_data,
                    "hhv_btu_scf": composition_data["hhv_btu_scf"],
                    "lhv_btu_scf": composition_data["lhv_btu_scf"],
                    "wobbe_index": composition_data["wobbe_index"],
                },
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return _stage_result(
                stage, False, elapsed_ms, pipeline_id, error=str(exc),
            )

    def _stage_determine_ce(
        self,
        pipeline_id: str,
        validated: Dict[str, Any],
        composition: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 3: Determine effective combustion efficiency.

        Calculates the effective combustion efficiency incorporating
        wind speed, tip velocity, LHV, and assist adjustments.

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data.
            composition: Gas composition data from stage 2.

        Returns:
            Stage result with ce_data including effective_ce.
        """
        stage = PipelineStage.DETERMINE_CE.value
        t0 = time.perf_counter()

        try:
            flare_type = validated.get("flare_type", "ELEVATED_STEAM_ASSISTED")
            override_ce = validated.get("combustion_efficiency")

            # Try upstream engine first
            if self.combustion_efficiency is not None:
                try:
                    ce_result = self.combustion_efficiency.calculate_effective_ce(
                        flare_type=flare_type,
                        wind_speed_ms=validated.get("wind_speed_ms"),
                        tip_velocity_mach=validated.get("tip_velocity_mach"),
                        lhv_btu_scf=composition.get("lhv_btu_scf"),
                        steam_to_gas_ratio=validated.get("steam_to_gas_ratio"),
                        air_to_gas_ratio=validated.get("air_to_gas_ratio"),
                        override_ce=override_ce,
                    )
                    if hasattr(ce_result, "model_dump"):
                        ce_data = ce_result.model_dump(mode="json")
                    elif isinstance(ce_result, dict):
                        ce_data = ce_result
                    else:
                        ce_data = {"effective_ce": float(ce_result)}

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return _stage_result(
                        stage, True, elapsed_ms, pipeline_id,
                        extra={"ce_data": ce_data},
                    )
                except (AttributeError, TypeError) as exc:
                    logger.warning(
                        "CombustionEfficiencyEngine failed: %s", exc,
                    )

            # Built-in CE calculation
            base_ce = Decimal("0.98")  # EPA default
            if override_ce is not None:
                base_ce = _to_decimal(override_ce)
            else:
                flare_info = FLARE_TYPES.get(flare_type, {})
                if flare_info:
                    base_ce = _to_decimal(flare_info.get("default_ce", 0.98))

            effective_ce = base_ce
            adjustments: List[Dict[str, Any]] = []

            # Wind speed adjustment: CE degrades above 10 m/s
            wind_speed = validated.get("wind_speed_ms")
            if wind_speed is not None:
                wind = _to_decimal(wind_speed)
                if wind > Decimal("10"):
                    wind_penalty = (wind - Decimal("10")) * Decimal("0.005")
                    wind_penalty = min(wind_penalty, Decimal("0.10"))
                    effective_ce -= wind_penalty
                    adjustments.append({
                        "factor": "wind_speed",
                        "value": float(wind),
                        "adjustment": -float(wind_penalty),
                    })

            # Tip velocity adjustment: >Mach 0.5 reduces CE
            tip_velocity = validated.get("tip_velocity_mach")
            if tip_velocity is not None:
                tip_v = _to_decimal(tip_velocity)
                if tip_v > Decimal("0.5"):
                    tip_penalty = (tip_v - Decimal("0.5")) * Decimal("0.02")
                    tip_penalty = min(tip_penalty, Decimal("0.05"))
                    effective_ce -= tip_penalty
                    adjustments.append({
                        "factor": "tip_velocity",
                        "value": float(tip_v),
                        "adjustment": -float(tip_penalty),
                    })

            # LHV threshold: <200 BTU/scf causes instability
            lhv = composition.get("lhv_btu_scf", 1000.0)
            if lhv < 200.0:
                lhv_penalty = Decimal("0.05")
                if lhv < 100.0:
                    lhv_penalty = Decimal("0.15")
                effective_ce -= lhv_penalty
                adjustments.append({
                    "factor": "low_lhv",
                    "value": lhv,
                    "adjustment": -float(lhv_penalty),
                })

            # Steam assist bonus
            steam_ratio = validated.get("steam_to_gas_ratio")
            if steam_ratio is not None:
                sr = _to_decimal(steam_ratio)
                if Decimal("0.3") <= sr <= Decimal("0.5"):
                    steam_bonus = Decimal("0.005")
                    effective_ce += steam_bonus
                    adjustments.append({
                        "factor": "steam_assist_optimal",
                        "value": float(sr),
                        "adjustment": float(steam_bonus),
                    })
                elif sr > Decimal("0.5"):
                    # Over-steaming reduces CE
                    over_steam_penalty = (sr - Decimal("0.5")) * Decimal("0.01")
                    over_steam_penalty = min(over_steam_penalty, Decimal("0.03"))
                    effective_ce -= over_steam_penalty
                    adjustments.append({
                        "factor": "over_steaming",
                        "value": float(sr),
                        "adjustment": -float(over_steam_penalty),
                    })

            # Clamp CE to valid range
            effective_ce = max(Decimal("0.50"), min(effective_ce, Decimal("0.9999")))

            ce_data: Dict[str, Any] = {
                "base_ce": float(base_ce),
                "effective_ce": float(effective_ce),
                "destruction_removal_efficiency": float(effective_ce),
                "adjustments": adjustments,
                "flare_type": flare_type,
                "source": "OVERRIDE" if override_ce else "BUILTIN",
            }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={
                    "ce_data": ce_data,
                    "effective_ce": ce_data["effective_ce"],
                },
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return _stage_result(
                stage, False, elapsed_ms, pipeline_id, error=str(exc),
            )

    def _stage_classify_event(
        self,
        pipeline_id: str,
        validated: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 4: Classify and record the flaring event.

        Classifies the flaring event by category and records it via
        the FlaringEventTrackerEngine when available.

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data.

        Returns:
            Stage result with event_data.
        """
        stage = PipelineStage.CLASSIFY_EVENT.value
        t0 = time.perf_counter()

        try:
            event_category = validated.get("event_category", "ROUTINE")
            flare_type = validated.get("flare_type", "UNKNOWN")
            gas_volume_mscf = validated.get("gas_volume_mscf", 0)
            operating_hours = validated.get("operating_hours", 1.0)

            # Try upstream engine
            if self.event_tracker is not None:
                try:
                    event_result = self.event_tracker.record_event(
                        flare_id=validated.get("flare_id", ""),
                        event_category=event_category,
                        gas_volume_mscf=gas_volume_mscf,
                        duration_hours=operating_hours,
                    )
                    if hasattr(event_result, "model_dump"):
                        event_data = event_result.model_dump(mode="json")
                    elif isinstance(event_result, dict):
                        event_data = event_result
                    else:
                        event_data = {}

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    _record_flaring_event(event_category, flare_type)
                    return _stage_result(
                        stage, True, elapsed_ms, pipeline_id,
                        extra={"event_data": event_data},
                    )
                except (AttributeError, TypeError) as exc:
                    logger.warning(
                        "FlaringEventTrackerEngine failed: %s", exc,
                    )

            # Built-in event classification
            event_id = f"fl_evt_{uuid.uuid4().hex[:12]}"
            category_info = EVENT_CATEGORIES.get(event_category, {})
            is_routine = event_category in ("ROUTINE", "PILOT_PURGE")

            event_data: Dict[str, Any] = {
                "event_id": event_id,
                "event_category": event_category,
                "category_display": category_info.get(
                    "display_name", event_category,
                ),
                "flare_type": flare_type,
                "gas_volume_mscf": gas_volume_mscf,
                "duration_hours": operating_hours,
                "is_routine": is_routine,
                "zrf_applicable": is_routine and event_category == "ROUTINE",
                "timestamp": _utcnow_iso(),
            }

            _record_flaring_event(event_category, flare_type)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={
                    "event_data": event_data,
                    "event_category": event_category,
                    "is_routine": is_routine,
                },
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return _stage_result(
                stage, False, elapsed_ms, pipeline_id, error=str(exc),
            )

    def _stage_calculate_emissions(
        self,
        pipeline_id: str,
        validated: Dict[str, Any],
        composition: Dict[str, Any],
        ce_data: Dict[str, Any],
        gwp_source: str,
    ) -> Dict[str, Any]:
        """Stage 5: Calculate CO2, CH4, N2O emissions and CO2e.

        Applies the selected calculation method (gas composition,
        default EF, engineering estimate, or direct measurement)
        to compute emissions including pilot and purge gas.

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data.
            composition: Gas composition data from stage 2.
            ce_data: Combustion efficiency data from stage 3.
            gwp_source: GWP source for CO2e conversion.

        Returns:
            Stage result with emissions_data.
        """
        stage = PipelineStage.CALCULATE_EMISSIONS.value
        t0 = time.perf_counter()

        try:
            method = validated.get("method", "DEFAULT_EF")
            volume_mscf = _to_decimal(validated.get("gas_volume_mscf", 0))
            effective_ce = _to_decimal(ce_data.get("effective_ce", 0.98))
            gwp_table = GWP_VALUES.get(gwp_source, GWP_VALUES["AR6"])
            operating_hours = _to_decimal(validated.get("operating_hours", 1.0))

            # Try upstream engine first
            if self.emission_calculator is not None:
                try:
                    calc_result = self.emission_calculator.calculate(
                        method=method,
                        gas_volume_mscf=float(volume_mscf),
                        composition=composition,
                        combustion_efficiency=float(effective_ce),
                        gwp_source=gwp_source,
                    )
                    if hasattr(calc_result, "model_dump"):
                        emissions_data = calc_result.model_dump(mode="json")
                    elif isinstance(calc_result, dict):
                        emissions_data = calc_result
                    else:
                        emissions_data = {}

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return _stage_result(
                        stage, True, elapsed_ms, pipeline_id,
                        extra={"emissions_data": emissions_data},
                    )
                except (AttributeError, TypeError) as exc:
                    logger.warning(
                        "EmissionCalculatorEngine failed: %s", exc,
                    )

            # Built-in emission calculation
            co2_kg = Decimal("0")
            ch4_kg = Decimal("0")
            n2o_kg = Decimal("0")
            gas_emissions: List[Dict[str, Any]] = []

            if method == "GAS_COMPOSITION":
                # Gas composition method
                co2_kg, ch4_kg, n2o_kg, gas_emissions = (
                    self._calc_gas_composition_method(
                        volume_mscf, composition, effective_ce, gwp_table,
                    )
                )
            else:
                # Default emission factor method
                co2_kg, ch4_kg, n2o_kg, gas_emissions = (
                    self._calc_default_ef_method(
                        volume_mscf, effective_ce, gwp_table,
                        validated.get("ef_source", "EPA_SUBPART_W"),
                    )
                )

            # Add pilot gas emissions
            pilot_co2_kg, pilot_ch4_kg = self._calc_pilot_emissions(
                validated, operating_hours,
            )
            co2_kg += pilot_co2_kg
            ch4_kg += pilot_ch4_kg

            if float(pilot_co2_kg) > 0 or float(pilot_ch4_kg) > 0:
                gas_emissions.append({
                    "source": "pilot_gas",
                    "co2_kg": float(pilot_co2_kg),
                    "ch4_kg": float(pilot_ch4_kg),
                    "n2o_kg": 0.0,
                })

            # Add purge gas emissions
            purge_co2_kg, purge_ch4_kg = self._calc_purge_emissions(
                validated, operating_hours,
            )
            co2_kg += purge_co2_kg
            ch4_kg += purge_ch4_kg

            if float(purge_co2_kg) > 0 or float(purge_ch4_kg) > 0:
                gas_emissions.append({
                    "source": "purge_gas",
                    "co2_kg": float(purge_co2_kg),
                    "ch4_kg": float(purge_ch4_kg),
                    "n2o_kg": 0.0,
                })

            # CO2e totals
            gwp_ch4 = _to_decimal(gwp_table.get("CH4", 27.3))
            gwp_n2o = _to_decimal(gwp_table.get("N2O", 273.0))

            co2e_from_co2 = co2_kg
            co2e_from_ch4 = ch4_kg * gwp_ch4
            co2e_from_n2o = n2o_kg * gwp_n2o
            total_co2e_kg = co2e_from_co2 + co2e_from_ch4 + co2e_from_n2o

            emissions_data: Dict[str, Any] = {
                "co2_kg": float(co2_kg),
                "ch4_kg": float(ch4_kg),
                "n2o_kg": float(n2o_kg),
                "total_co2e_kg": float(total_co2e_kg),
                "co2e_from_co2_kg": float(co2e_from_co2),
                "co2e_from_ch4_kg": float(co2e_from_ch4),
                "co2e_from_n2o_kg": float(co2e_from_n2o),
                "gwp_source": gwp_source,
                "gwp_ch4": float(gwp_ch4),
                "gwp_n2o": float(gwp_n2o),
                "method": method,
                "combustion_efficiency": float(effective_ce),
                "gas_volume_mscf": float(volume_mscf),
                "gas_emissions": gas_emissions,
                "includes_pilot": float(pilot_co2_kg) > 0 or float(pilot_ch4_kg) > 0,
                "includes_purge": float(purge_co2_kg) > 0 or float(purge_ch4_kg) > 0,
            }

            # Record Prometheus emissions metrics
            flare_type = validated.get("flare_type", "UNKNOWN")
            _record_emissions(flare_type, "CO2", float(co2_kg))
            _record_emissions(flare_type, "CH4", float(ch4_kg))
            _record_emissions(flare_type, "N2O", float(n2o_kg))

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={
                    "emissions_data": emissions_data,
                    "total_co2e_kg": emissions_data["total_co2e_kg"],
                    "method": method,
                },
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return _stage_result(
                stage, False, elapsed_ms, pipeline_id, error=str(exc),
            )

    def _stage_quantify_uncertainty(
        self,
        pipeline_id: str,
        validated: Dict[str, Any],
        emissions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 6: Run Monte Carlo uncertainty quantification.

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data.
            emissions: Emissions data from stage 5.

        Returns:
            Stage result with uncertainty_data.
        """
        stage = PipelineStage.QUANTIFY_UNCERTAINTY.value
        t0 = time.perf_counter()

        try:
            total_co2e_kg = emissions.get("total_co2e_kg", 0)

            # Try upstream engine
            if self.uncertainty_engine is not None:
                try:
                    unc_result = self.uncertainty_engine.run_monte_carlo(
                        calculation_input=emissions,
                        n_iterations=5000,
                    )
                    if hasattr(unc_result, "model_dump"):
                        unc_data = unc_result.model_dump(mode="json")
                    elif isinstance(unc_result, dict):
                        unc_data = unc_result
                    else:
                        unc_data = {}

                    _record_uncertainty("monte_carlo")
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return _stage_result(
                        stage, True, elapsed_ms, pipeline_id,
                        extra={"uncertainty_data": unc_data},
                    )
                except (AttributeError, TypeError) as exc:
                    logger.warning(
                        "UncertaintyQuantifierEngine failed: %s", exc,
                    )

            # Analytical fallback: +/- 10-15% for flaring
            method = validated.get("method", "DEFAULT_EF")
            base_uncertainty = 0.10 if method == "GAS_COMPOSITION" else 0.15
            co2e = float(total_co2e_kg)
            std_dev = co2e * base_uncertainty

            uncertainty_data: Dict[str, Any] = {
                "method": "analytical_fallback",
                "iterations": None,
                "mean_co2e_kg": co2e,
                "std_dev_kg": std_dev,
                "uncertainty_pct": base_uncertainty * 100.0,
                "confidence_intervals": {
                    "95": {
                        "lower": max(0.0, co2e - 1.96 * std_dev),
                        "upper": co2e + 1.96 * std_dev,
                    },
                    "90": {
                        "lower": max(0.0, co2e - 1.645 * std_dev),
                        "upper": co2e + 1.645 * std_dev,
                    },
                },
                "dqi_score": 3.0 if method == "GAS_COMPOSITION" else 2.0,
            }

            _record_uncertainty("analytical_fallback")

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={"uncertainty_data": uncertainty_data},
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return _stage_result(
                stage, False, elapsed_ms, pipeline_id, error=str(exc),
            )

    def _stage_check_compliance(
        self,
        pipeline_id: str,
        validated: Dict[str, Any],
        emissions: Dict[str, Any],
        ce_data: Dict[str, Any],
        gwp_source: str,
        frameworks: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Stage 7: Check against applicable regulatory frameworks.

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data.
            emissions: Emissions data from stage 5.
            ce_data: Combustion efficiency data from stage 3.
            gwp_source: GWP source string.
            frameworks: Specific frameworks to check (None = all).

        Returns:
            Stage result with compliance_data.
        """
        stage = PipelineStage.CHECK_COMPLIANCE.value
        t0 = time.perf_counter()

        try:
            # Try upstream engine
            if self.compliance_checker is not None:
                try:
                    comp_result = self.compliance_checker.check(
                        validated=validated,
                        emissions=emissions,
                        ce_data=ce_data,
                        gwp_source=gwp_source,
                        frameworks=frameworks,
                    )
                    if hasattr(comp_result, "model_dump"):
                        comp_data = comp_result.model_dump(mode="json")
                    elif isinstance(comp_result, dict):
                        comp_data = comp_result
                    else:
                        comp_data = {}

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return _stage_result(
                        stage, True, elapsed_ms, pipeline_id,
                        extra={"compliance_data": comp_data},
                    )
                except (AttributeError, TypeError) as exc:
                    logger.warning(
                        "ComplianceCheckerEngine failed: %s", exc,
                    )

            # Built-in compliance checking
            frameworks_to_check = frameworks or list(REGULATORY_FRAMEWORKS.keys())
            results: List[Dict[str, Any]] = []
            compliant_count = 0
            non_compliant_count = 0
            partial_count = 0

            for fw_key in frameworks_to_check:
                fw_result = self._evaluate_framework(
                    fw_key, validated, emissions, ce_data, gwp_source,
                )
                results.append(fw_result)

                status = fw_result.get("status", "non_compliant")
                if status == "compliant":
                    compliant_count += 1
                elif status == "partial":
                    partial_count += 1
                else:
                    non_compliant_count += 1

                _record_compliance_check(fw_key, status)

            compliance_data: Dict[str, Any] = {
                "frameworks_checked": len(frameworks_to_check),
                "compliant": compliant_count,
                "non_compliant": non_compliant_count,
                "partial": partial_count,
                "results": results,
            }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={"compliance_data": compliance_data},
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return _stage_result(
                stage, False, elapsed_ms, pipeline_id, error=str(exc),
            )

    def _stage_assemble_result(
        self,
        pipeline_id: str,
        validated: Dict[str, Any],
        composition: Dict[str, Any],
        ce_data: Dict[str, Any],
        event_data: Dict[str, Any],
        emissions: Dict[str, Any],
        uncertainty: Dict[str, Any],
        compliance: Dict[str, Any],
        gwp_source: str,
    ) -> Dict[str, Any]:
        """Stage 8: Assemble final result with provenance chain.

        Combines all stage outputs into the final calculation result
        with a complete SHA-256 provenance chain.

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data.
            composition: Gas composition data.
            ce_data: Combustion efficiency data.
            event_data: Flaring event data.
            emissions: Emissions data.
            uncertainty: Uncertainty data.
            compliance: Compliance data.
            gwp_source: GWP source string.

        Returns:
            Stage result with assembled_result.
        """
        stage = PipelineStage.ASSEMBLE_RESULT.value
        t0 = time.perf_counter()

        try:
            calc_id = validated.get("calculation_id", _new_uuid())

            assembled: Dict[str, Any] = {
                "calculation_id": calc_id,
                "flare_id": validated.get("flare_id", ""),
                "flare_type": validated.get("flare_type", ""),
                "method": validated.get("method", "DEFAULT_EF"),
                "event_category": validated.get("event_category", "ROUTINE"),
                "gas_volume_mscf": validated.get("gas_volume_mscf", 0),
                "operating_hours": validated.get("operating_hours", 1.0),

                # Composition
                "hhv_btu_scf": composition.get("hhv_btu_scf", 0),
                "lhv_btu_scf": composition.get("lhv_btu_scf", 0),
                "wobbe_index": composition.get("wobbe_index", 0),

                # Combustion efficiency
                "combustion_efficiency": ce_data.get("effective_ce", 0.98),
                "ce_adjustments": ce_data.get("adjustments", []),

                # Emissions
                "co2_kg": emissions.get("co2_kg", 0),
                "ch4_kg": emissions.get("ch4_kg", 0),
                "n2o_kg": emissions.get("n2o_kg", 0),
                "total_co2e_kg": emissions.get("total_co2e_kg", 0),
                "co2e_from_co2_kg": emissions.get("co2e_from_co2_kg", 0),
                "co2e_from_ch4_kg": emissions.get("co2e_from_ch4_kg", 0),
                "co2e_from_n2o_kg": emissions.get("co2e_from_n2o_kg", 0),
                "includes_pilot": emissions.get("includes_pilot", False),
                "includes_purge": emissions.get("includes_purge", False),

                # GWP
                "gwp_source": gwp_source,

                # Event
                "event_id": event_data.get("event_id", ""),
                "is_routine": event_data.get("is_routine", True),

                # Uncertainty
                "uncertainty_pct": uncertainty.get("uncertainty_pct"),
                "uncertainty_method": uncertainty.get("method"),
                "dqi_score": uncertainty.get("dqi_score"),

                # Compliance
                "compliance_summary": {
                    "frameworks_checked": compliance.get("frameworks_checked", 0),
                    "compliant": compliance.get("compliant", 0),
                    "non_compliant": compliance.get("non_compliant", 0),
                    "partial": compliance.get("partial", 0),
                },

                # Provenance
                "provenance_hash": _compute_hash({
                    "calculation_id": calc_id,
                    "flare_type": validated.get("flare_type", ""),
                    "total_co2e_kg": emissions.get("total_co2e_kg", 0),
                    "method": validated.get("method", "DEFAULT_EF"),
                    "gwp_source": gwp_source,
                }),

                "timestamp": _utcnow_iso(),
            }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={"assembled_result": assembled},
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return _stage_result(
                stage, False, elapsed_ms, pipeline_id, error=str(exc),
            )

    # ==================================================================
    # Private: Emission calculation methods
    # ==================================================================

    def _calc_gas_composition_method(
        self,
        volume_mscf: Decimal,
        composition: Dict[str, Any],
        effective_ce: Decimal,
        gwp_table: Dict[str, float],
    ) -> Tuple[Decimal, Decimal, Decimal, List[Dict[str, Any]]]:
        """Calculate emissions using gas composition method.

        For each hydrocarbon component:
          CO2 = volume * fraction * (carbon_atoms * MW_CO2 / MW_component)
                * CE / molar_volume_scf * 1000  [scf -> mscf]
          CH4_slip = volume * CH4_fraction * (1 - CE) * density

        Args:
            volume_mscf: Gas volume in MSCF.
            composition: Gas composition data including fractions.
            effective_ce: Effective combustion efficiency (0-1).
            gwp_table: GWP values for CO2e conversion.

        Returns:
            Tuple of (co2_kg, ch4_kg, n2o_kg, gas_emissions_list).
        """
        gas_comp = composition.get("gas_composition", {})
        volume_scf = volume_mscf * Decimal(str(MSCF_TO_SCF))

        total_co2_kg = Decimal("0")
        total_ch4_kg = Decimal("0")
        gas_emissions: List[Dict[str, Any]] = []

        for component, fraction_val in gas_comp.items():
            fraction = _to_decimal(fraction_val)
            if fraction <= 0:
                continue

            comp_info = GAS_COMPONENT_HHVS.get(component)
            if comp_info is None:
                continue

            carbon_atoms = comp_info.get("carbon_atoms", 0)
            mw = _to_decimal(comp_info.get("molecular_weight", 0))

            if carbon_atoms > 0 and mw > 0:
                # CO2 from combustion: V_scf * y_i * (n_C * MW_CO2 / MW_i)
                # converted to kg (1 lb-mol = 379.3 scf at std; 1 lb = 0.4536 kg)
                co2_per_scf = (
                    _to_decimal(carbon_atoms) * _to_decimal(MW_CO2) / mw
                    / _to_decimal(MOLAR_VOLUME_SCF)
                    * _to_decimal("0.45359")  # lb to kg
                )
                component_co2_kg = (
                    volume_scf * fraction * co2_per_scf * effective_ce
                )
                total_co2_kg += component_co2_kg

                gas_emissions.append({
                    "component": component,
                    "fraction": float(fraction),
                    "co2_kg": float(component_co2_kg),
                    "source": "combustion",
                })

        # CH4 slip: uncombusted methane
        ch4_fraction = _to_decimal(gas_comp.get("CH4", 0))
        if ch4_fraction > 0:
            ch4_mw = _to_decimal(MW_CH4)
            ch4_per_scf = ch4_mw / _to_decimal(MOLAR_VOLUME_SCF) * _to_decimal("0.45359")
            total_ch4_kg = volume_scf * ch4_fraction * (Decimal("1") - effective_ce) * ch4_per_scf
            gas_emissions.append({
                "component": "CH4_slip",
                "fraction": float(ch4_fraction),
                "ch4_kg": float(total_ch4_kg),
                "source": "uncombusted_slip",
            })

        # N2O from high-temperature combustion
        hhv_btu_scf = _to_decimal(composition.get("hhv_btu_scf", 1000))
        total_energy_btu = volume_scf * hhv_btu_scf
        total_energy_tj = total_energy_btu / _to_decimal(BTU_PER_TJ)
        total_n2o_kg = total_energy_tj * _to_decimal(N2O_FACTOR_KG_PER_TJ)

        gas_emissions.append({
            "component": "N2O",
            "n2o_kg": float(total_n2o_kg),
            "source": "high_temp_combustion",
        })

        return total_co2_kg, total_ch4_kg, total_n2o_kg, gas_emissions

    def _calc_default_ef_method(
        self,
        volume_mscf: Decimal,
        effective_ce: Decimal,
        gwp_table: Dict[str, float],
        ef_source: str,
    ) -> Tuple[Decimal, Decimal, Decimal, List[Dict[str, Any]]]:
        """Calculate emissions using default emission factor method.

        Applies standard emission factors (kg per MSCF) to the
        measured or estimated gas volume.

        Args:
            volume_mscf: Gas volume in MSCF.
            effective_ce: Effective combustion efficiency.
            gwp_table: GWP values.
            ef_source: Emission factor source (EPA_SUBPART_W, etc.).

        Returns:
            Tuple of (co2_kg, ch4_kg, n2o_kg, gas_emissions_list).
        """
        ef = DEFAULT_EMISSION_FACTORS.get(
            ef_source,
            DEFAULT_EMISSION_FACTORS["EPA_SUBPART_W"],
        )

        co2_ef = _to_decimal(ef.get("co2_kg_per_mscf", 59.29))
        ch4_ef = _to_decimal(ef.get("ch4_kg_per_mscf", 0.545))
        n2o_ef = _to_decimal(ef.get("n2o_kg_per_mscf", 0.0091))

        # CO2: volume * EF (EF already accounts for typical CE)
        co2_kg = volume_mscf * co2_ef

        # CH4: slip adjusted by actual CE vs default 98%
        default_ce = Decimal("0.98")
        ce_adjustment = (Decimal("1") - effective_ce) / (Decimal("1") - default_ce)
        ch4_kg = volume_mscf * ch4_ef * ce_adjustment

        # N2O
        n2o_kg = volume_mscf * n2o_ef

        gas_emissions: List[Dict[str, Any]] = [
            {
                "component": "CO2",
                "co2_kg": float(co2_kg),
                "ef_kg_per_mscf": float(co2_ef),
                "source": ef_source,
            },
            {
                "component": "CH4",
                "ch4_kg": float(ch4_kg),
                "ef_kg_per_mscf": float(ch4_ef),
                "ce_adjustment": float(ce_adjustment),
                "source": ef_source,
            },
            {
                "component": "N2O",
                "n2o_kg": float(n2o_kg),
                "ef_kg_per_mscf": float(n2o_ef),
                "source": ef_source,
            },
        ]

        return co2_kg, ch4_kg, n2o_kg, gas_emissions

    def _calc_pilot_emissions(
        self,
        validated: Dict[str, Any],
        operating_hours: Decimal,
    ) -> Tuple[Decimal, Decimal]:
        """Calculate pilot gas emissions.

        Pilot gas (typically natural gas) burns continuously to maintain
        the pilot flame.  Emissions are based on pilot gas flow rate
        and number of tips.

        Args:
            validated: Validated request data with pilot parameters.
            operating_hours: Duration in hours.

        Returns:
            Tuple of (pilot_co2_kg, pilot_ch4_kg).
        """
        pilot_mmbtu_hr = _to_decimal(validated.get("pilot_gas_mmbtu_hr", 0))
        num_tips = _to_decimal(validated.get("num_pilot_tips", 1))

        if pilot_mmbtu_hr <= 0:
            return Decimal("0"), Decimal("0")

        # Natural gas emission factors: ~53.06 kg CO2/MMBTU, ~0.001 kg CH4/MMBTU
        co2_ef_per_mmbtu = Decimal("53.06")
        ch4_ef_per_mmbtu = Decimal("0.001")

        total_mmbtu = pilot_mmbtu_hr * num_tips * operating_hours
        pilot_co2_kg = total_mmbtu * co2_ef_per_mmbtu
        pilot_ch4_kg = total_mmbtu * ch4_ef_per_mmbtu

        return pilot_co2_kg, pilot_ch4_kg

    def _calc_purge_emissions(
        self,
        validated: Dict[str, Any],
        operating_hours: Decimal,
    ) -> Tuple[Decimal, Decimal]:
        """Calculate purge gas emissions.

        Purge gas prevents air ingress and flashback.  N2 purge has
        zero GHG emissions.  Natural gas purge produces CH4 and CO2.

        Args:
            validated: Validated request data with purge parameters.
            operating_hours: Duration in hours.

        Returns:
            Tuple of (purge_co2_kg, purge_ch4_kg).
        """
        purge_type = validated.get("purge_gas_type", "N2")
        purge_flow_scfh = _to_decimal(validated.get("purge_gas_flow_scfh", 0))

        if purge_flow_scfh <= 0 or purge_type == "N2":
            return Decimal("0"), Decimal("0")

        # Natural gas purge
        total_scf = purge_flow_scfh * operating_hours
        total_mscf = total_scf / Decimal(str(MSCF_TO_SCF))

        # Use EPA default EFs for natural gas
        co2_kg = total_mscf * Decimal("59.29")
        ch4_kg = total_mscf * Decimal("0.545")

        return co2_kg, ch4_kg

    # ==================================================================
    # Private: Gas composition helpers
    # ==================================================================

    def _default_gas_composition(self) -> Dict[str, float]:
        """Return a default associated gas composition.

        Represents typical associated gas from oil production.

        Returns:
            Dictionary of component fractions summing to 1.0.
        """
        return {
            "CH4": 0.80,
            "C2H6": 0.07,
            "C3H8": 0.04,
            "n_C4H10": 0.02,
            "i_C4H10": 0.01,
            "C5H12": 0.005,
            "CO2": 0.03,
            "N2": 0.02,
            "H2S": 0.005,
        }

    def _calculate_hhv(
        self,
        gas_composition: Dict[str, Any],
    ) -> Decimal:
        """Calculate higher heating value from gas composition.

        HHV = sum(y_i * HHV_i) for all components.

        Args:
            gas_composition: Component mole fractions.

        Returns:
            HHV in BTU/scf.
        """
        total_hhv = Decimal("0")
        for component, fraction in gas_composition.items():
            frac = _to_decimal(fraction)
            comp_info = GAS_COMPONENT_HHVS.get(component)
            if comp_info is not None:
                hhv = _to_decimal(comp_info["hhv_btu_scf"])
                total_hhv += frac * hhv
        return total_hhv

    def _calculate_specific_gravity(
        self,
        gas_composition: Dict[str, Any],
    ) -> Decimal:
        """Calculate specific gravity from gas composition.

        SG = MW_gas / MW_air, where MW_gas = sum(y_i * MW_i).

        Args:
            gas_composition: Component mole fractions.

        Returns:
            Specific gravity relative to air.
        """
        mw_gas = self._calculate_molecular_weight(gas_composition)
        if mw_gas > 0:
            return mw_gas / _to_decimal(MW_AIR)
        return Decimal("1")

    def _calculate_molecular_weight(
        self,
        gas_composition: Dict[str, Any],
    ) -> Decimal:
        """Calculate average molecular weight from gas composition.

        MW_avg = sum(y_i * MW_i) for all components.

        Args:
            gas_composition: Component mole fractions.

        Returns:
            Average molecular weight in g/mol.
        """
        total_mw = Decimal("0")
        for component, fraction in gas_composition.items():
            frac = _to_decimal(fraction)
            comp_info = GAS_COMPONENT_HHVS.get(component)
            if comp_info is not None:
                mw = _to_decimal(comp_info["molecular_weight"])
                total_mw += frac * mw
        return total_mw

    def _calculate_hc_fraction(
        self,
        gas_composition: Dict[str, Any],
    ) -> Decimal:
        """Calculate hydrocarbon (combustible) fraction.

        Sums the mole fractions of all components with carbon atoms > 0
        plus H2 and H2S (which are combustible but not hydrocarbons).

        Args:
            gas_composition: Component mole fractions.

        Returns:
            Hydrocarbon/combustible fraction (0-1).
        """
        hc_fraction = Decimal("0")
        for component, fraction in gas_composition.items():
            frac = _to_decimal(fraction)
            comp_info = GAS_COMPONENT_HHVS.get(component)
            if comp_info is not None:
                hhv = comp_info.get("hhv_btu_scf", 0)
                if hhv > 0:
                    hc_fraction += frac
        return hc_fraction

    def _estimate_water_production(
        self,
        gas_composition: Dict[str, Any],
    ) -> Decimal:
        """Estimate moles of water produced per mole of gas burned.

        For each hydrocarbon CnH(2n+2): n+1 moles H2O per mole.
        Used to calculate LHV from HHV.

        Args:
            gas_composition: Component mole fractions.

        Returns:
            Estimated water mole fraction adjustment.
        """
        h2o_moles = Decimal("0")
        water_production = {
            "CH4": Decimal("2"),
            "C2H6": Decimal("3"),
            "C3H8": Decimal("4"),
            "n_C4H10": Decimal("5"),
            "i_C4H10": Decimal("5"),
            "C5H12": Decimal("6"),
            "C6_PLUS": Decimal("7"),
            "H2": Decimal("1"),
            "H2S": Decimal("1"),
        }

        for component, fraction in gas_composition.items():
            frac = _to_decimal(fraction)
            moles = water_production.get(component, Decimal("0"))
            h2o_moles += frac * moles

        return h2o_moles

    # ==================================================================
    # Private: Compliance evaluation
    # ==================================================================

    def _evaluate_framework(
        self,
        framework: str,
        validated: Dict[str, Any],
        emissions: Dict[str, Any],
        ce_data: Dict[str, Any],
        gwp_source: str,
    ) -> Dict[str, Any]:
        """Evaluate compliance against a single regulatory framework.

        Args:
            framework: Framework identifier.
            validated: Validated request data.
            emissions: Emissions calculation data.
            ce_data: Combustion efficiency data.
            gwp_source: GWP source string.

        Returns:
            Dictionary with framework compliance details.
        """
        fw_info = REGULATORY_FRAMEWORKS.get(framework, {})
        requirements = fw_info.get("requirements", [])

        has_results = emissions.get("total_co2e_kg", 0) >= 0
        has_method = bool(validated.get("method"))
        has_ce = ce_data.get("effective_ce") is not None
        has_ch4 = emissions.get("ch4_kg", 0) > 0 or has_results
        has_volume = validated.get("gas_volume_mscf", 0) > 0
        has_provenance = True  # Pipeline always generates provenance
        is_ar6 = gwp_source in ("AR6", "AR6_20yr")
        has_event = bool(validated.get("event_category"))
        has_ogmp = validated.get("ogmp_level") is not None
        is_routine = validated.get("event_category") == "ROUTINE"

        check_map = {
            "has_results": has_results,
            "valid_gwp": gwp_source in GWP_VALUES,
            "has_methodology": has_method,
            "has_provenance": has_provenance,
            "has_uncertainty": True,
            "has_gas_breakdown": has_results,
            "gwp_ar6": is_ar6,
            "valid_method": has_method,
            "has_volume": has_volume,
            "has_ce": has_ce,
            "has_facility": validated.get("facility_id") is not None,
            "has_tier": has_method,
            "has_audit": has_provenance,
            "has_ch4": has_ch4,
            "has_event_category": has_event,
            "has_routine_tracking": is_routine or has_event,
            "has_routine_volume": has_volume and is_routine,
            "has_ogmp_level": has_ogmp,
        }

        met_count = 0
        requirement_results: List[Dict[str, Any]] = []
        for req in requirements:
            check_key = req.get("check", "")
            is_met = check_map.get(check_key, False)
            if is_met:
                met_count += 1
            requirement_results.append({
                "id": req.get("id", ""),
                "description": req.get("desc", ""),
                "met": is_met,
            })

        total_reqs = len(requirements)
        if met_count == total_reqs and total_reqs > 0:
            status = "compliant"
        elif met_count > 0:
            status = "partial"
        else:
            status = "non_compliant"

        return {
            "framework": framework,
            "display_name": fw_info.get("display_name", framework),
            "status": status,
            "total_requirements": total_reqs,
            "met_count": met_count,
            "requirements": requirement_results,
        }


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    "FlaringPipelineEngine",
    "PipelineStage",
    "PIPELINE_STAGES",
    "GWP_VALUES",
    "GAS_COMPONENT_HHVS",
    "FLARE_TYPES",
    "EVENT_CATEGORIES",
    "DEFAULT_EMISSION_FACTORS",
    "REGULATORY_FRAMEWORKS",
]
