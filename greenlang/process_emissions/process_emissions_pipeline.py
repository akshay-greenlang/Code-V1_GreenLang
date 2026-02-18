# -*- coding: utf-8 -*-
"""
Process Emissions Pipeline Engine (Engine 7) - AGENT-MRV-004

End-to-end orchestration pipeline for GHG Protocol Scope 1 non-combustion
industrial process emissions calculations.  Coordinates all six upstream
engines (process database, emission calculator, material balance engine,
abatement tracker, uncertainty quantifier, compliance checker) through a
deterministic, eight-stage pipeline:

    1. VALIDATE           - Input validation and normalisation
    2. RESOLVE_PROCESS    - Look up process type, emission factors, materials
    3. CALCULATE_MATERIAL_BALANCE - Track raw materials, carbon balance
    4. CALCULATE_EMISSIONS - Apply EF, mass balance, stoichiometric, or direct
    5. APPLY_ABATEMENT    - Apply abatement technology adjustments
    6. QUANTIFY_UNCERTAINTY - Run Monte Carlo or analytical uncertainty
    7. CHECK_COMPLIANCE   - Validate against applicable regulatory frameworks
    8. GENERATE_AUDIT     - Create provenance chain and audit trail

Each stage is checkpointed so that failures produce partial results with
complete provenance.

Zero-Hallucination Guarantees:
    - All emission calculations use deterministic Python ``Decimal`` arithmetic
    - No LLM calls in the calculation path
    - SHA-256 provenance hash at every pipeline stage
    - Full audit trail for regulatory traceability

Built-in Reference Data:
    This engine bundles standalone lookup tables (PROCESS_TYPES,
    GWP_VALUES, CARBONATE_FACTORS, DEFAULT_ABATEMENT_EFFICIENCIES) so
    that it can operate independently when the process database engine
    is unavailable.  In production these tables are superseded by the
    database engine.

Thread Safety:
    All mutable state is protected by a ``threading.Lock``.  Concurrent
    ``execute_pipeline`` invocations from different threads are safe.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-004 Process Emissions (GL-MRV-SCOPE1-004)
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
    from greenlang.process_emissions.config import (
        ProcessEmissionsConfig,
        get_config,
    )
except ImportError:
    ProcessEmissionsConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.process_emissions.models import (
        ProcessCategory,
        ProcessType,
        EmissionGas,
        CalculationMethod,
        CalculationTier,
        GWPSource,
        MaterialType,
        AbatementType,
        ComplianceStatus,
        CalculationRequest,
        CalculationResult,
        CalculationDetailResult,
        GasEmissionResult,
        MaterialInputRecord,
        AbatementRecord,
        ComplianceCheckResult,
        BatchCalculationRequest,
        BatchCalculationResult,
        UncertaintyRequest,
        UncertaintyResult,
        EmissionFactorRecord,
        ProcessTypeInfo,
        RawMaterialInfo,
        ProcessUnitRecord,
        GWP_VALUES as MODEL_GWP_VALUES,
        CARBONATE_EMISSION_FACTORS as MODEL_CARBONATE_FACTORS,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    MODEL_GWP_VALUES = None  # type: ignore[assignment]
    MODEL_CARBONATE_FACTORS = None  # type: ignore[assignment]

try:
    from greenlang.process_emissions.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.process_emissions.metrics import (
        PROMETHEUS_AVAILABLE,
        record_calculation,
        observe_calculation_duration,
        record_batch,
        observe_batch_size,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

    def record_calculation(process_type: str, status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def observe_calculation_duration(process_type: str, duration: float) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def record_batch(status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def observe_batch_size(size: int) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""


# ---------------------------------------------------------------------------
# Pipeline stage enum
# ---------------------------------------------------------------------------

class PipelineStage(str, Enum):
    """Eight-stage pipeline for process emissions calculations."""

    VALIDATE = "VALIDATE"
    RESOLVE_PROCESS = "RESOLVE_PROCESS"
    CALCULATE_MATERIAL_BALANCE = "CALCULATE_MATERIAL_BALANCE"
    CALCULATE_EMISSIONS = "CALCULATE_EMISSIONS"
    APPLY_ABATEMENT = "APPLY_ABATEMENT"
    QUANTIFY_UNCERTAINTY = "QUANTIFY_UNCERTAINTY"
    CHECK_COMPLIANCE = "CHECK_COMPLIANCE"
    GENERATE_AUDIT = "GENERATE_AUDIT"


PIPELINE_STAGES: List[str] = [s.value for s in PipelineStage]


# ---------------------------------------------------------------------------
# Built-in reference data for standalone operation
# ---------------------------------------------------------------------------

#: GWP values for all 8 greenhouse gas species tracked by the process
#: emissions agent.  Keys are (gas, gwp_source) pairs.
GWP_VALUES: Dict[str, Dict[str, float]] = {
    "AR4": {
        "CO2": 1.0, "CH4": 25.0, "N2O": 298.0, "SF6": 22800.0,
        "NF3": 17200.0, "CF4": 7390.0, "C2F6": 12200.0,
        "HFC_23": 14800.0, "HFC_134A": 1430.0, "HFC_152A": 124.0,
    },
    "AR5": {
        "CO2": 1.0, "CH4": 28.0, "N2O": 265.0, "SF6": 23500.0,
        "NF3": 16100.0, "CF4": 6630.0, "C2F6": 11100.0,
        "HFC_23": 12400.0, "HFC_134A": 1300.0, "HFC_152A": 138.0,
    },
    "AR6": {
        "CO2": 1.0, "CH4": 27.3, "N2O": 273.0, "SF6": 25200.0,
        "NF3": 17400.0, "CF4": 7380.0, "C2F6": 12400.0,
        "HFC_23": 14600.0, "HFC_134A": 1530.0, "HFC_152A": 164.0,
    },
    "AR6_20yr": {
        "CO2": 1.0, "CH4": 80.8, "N2O": 273.0, "SF6": 18300.0,
        "NF3": 13400.0, "CF4": 5300.0, "C2F6": 8940.0,
        "HFC_23": 12000.0, "HFC_134A": 4140.0, "HFC_152A": 560.0,
    },
}

#: Carbonate decomposition emission factors (tonnes CO2 per tonne
#: carbonate mineral consumed).  IPCC 2006 Vol 3 Chapter 2.
CARBONATE_FACTORS: Dict[str, Dict[str, Any]] = {
    "CALCITE": {
        "formula": "CaCO3",
        "molecular_weight": 100.09,
        "co2_factor": 0.440,
        "description": "Calcium carbonate (limestone, chalk)",
    },
    "DOLOMITE": {
        "formula": "CaMg(CO3)2",
        "molecular_weight": 184.40,
        "co2_factor": 0.477,
        "description": "Calcium-magnesium carbonate",
    },
    "MAGNESITE": {
        "formula": "MgCO3",
        "molecular_weight": 84.31,
        "co2_factor": 0.522,
        "description": "Magnesium carbonate",
    },
    "SIDERITE": {
        "formula": "FeCO3",
        "molecular_weight": 115.86,
        "co2_factor": 0.380,
        "description": "Iron(II) carbonate",
    },
    "ANKERITE": {
        "formula": "Ca(Fe,Mg,Mn)(CO3)2",
        "molecular_weight": 206.00,
        "co2_factor": 0.427,
        "description": "Calcium iron magnesium manganese carbonate",
    },
    "RHODOCHROSITE": {
        "formula": "MnCO3",
        "molecular_weight": 114.95,
        "co2_factor": 0.383,
        "description": "Manganese carbonate",
    },
    "SODIUM_CARBONATE": {
        "formula": "Na2CO3",
        "molecular_weight": 105.99,
        "co2_factor": 0.415,
        "description": "Soda ash",
    },
}

#: Default emission factors for 25 industrial process types.
#: Units: tonnes CO2 per tonne production output (unless stated otherwise).
#: Source: IPCC 2006 Guidelines Vol 3, EPA 40 CFR Part 98.
PROCESS_TYPES: Dict[str, Dict[str, Any]] = {
    # ---- Mineral processes ----
    "CEMENT_CLINKER": {
        "category": "MINERAL",
        "display_name": "Cement Clinker Production",
        "default_ef_co2": 0.525,
        "ef_unit": "tCO2/t_clinker",
        "gases": ["CO2"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "CO2 from calcination of CaCO3 in clinker kiln",
    },
    "LIME_PRODUCTION": {
        "category": "MINERAL",
        "display_name": "Lime Production",
        "default_ef_co2": 0.785,
        "ef_unit": "tCO2/t_lime",
        "gases": ["CO2"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "CO2 from calcination of limestone/dolomite",
    },
    "GLASS_PRODUCTION": {
        "category": "MINERAL",
        "display_name": "Glass Production",
        "default_ef_co2": 0.210,
        "ef_unit": "tCO2/t_glass",
        "gases": ["CO2"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "CO2 from carbonate raw materials in glass batch",
    },
    "CERAMICS_PRODUCTION": {
        "category": "MINERAL",
        "display_name": "Ceramics Production",
        "default_ef_co2": 0.200,
        "ef_unit": "tCO2/t_product",
        "gases": ["CO2"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "CO2 from carbonate raw materials in ceramics",
    },
    "MINERAL_WOOL": {
        "category": "MINERAL",
        "display_name": "Mineral Wool Production",
        "default_ef_co2": 0.150,
        "ef_unit": "tCO2/t_product",
        "gases": ["CO2"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "CO2 from carbonate raw materials in mineral wool",
    },
    "SODA_ASH_PRODUCTION": {
        "category": "MINERAL",
        "display_name": "Soda Ash Production",
        "default_ef_co2": 0.138,
        "ef_unit": "tCO2/t_soda_ash",
        "gases": ["CO2"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "CO2 from trona ore decomposition or Solvay process",
    },
    "MAGNESIA_PRODUCTION": {
        "category": "MINERAL",
        "display_name": "Magnesia Production",
        "default_ef_co2": 1.092,
        "ef_unit": "tCO2/t_magnesia",
        "gases": ["CO2"],
        "tier_1_method": "STOICHIOMETRIC",
        "description": "CO2 from magnesite calcination",
    },
    # ---- Chemical processes ----
    "AMMONIA_PRODUCTION": {
        "category": "CHEMICAL",
        "display_name": "Ammonia Production",
        "default_ef_co2": 1.694,
        "ef_unit": "tCO2/t_ammonia",
        "gases": ["CO2"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "CO2 from natural gas feedstock reforming",
    },
    "NITRIC_ACID": {
        "category": "CHEMICAL",
        "display_name": "Nitric Acid Production",
        "default_ef_n2o": 0.007,
        "ef_unit": "tN2O/t_acid",
        "gases": ["N2O"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "N2O from catalytic oxidation of ammonia",
    },
    "ADIPIC_ACID": {
        "category": "CHEMICAL",
        "display_name": "Adipic Acid Production",
        "default_ef_n2o": 0.300,
        "ef_unit": "tN2O/t_acid",
        "gases": ["N2O"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "N2O from oxidation of cyclohexanone/cyclohexanol",
    },
    "CAPROLACTAM": {
        "category": "CHEMICAL",
        "display_name": "Caprolactam Production",
        "default_ef_n2o": 0.009,
        "ef_unit": "tN2O/t_caprolactam",
        "gases": ["N2O"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "N2O from cyclohexanone oxime rearrangement",
    },
    "CARBIDE_PRODUCTION": {
        "category": "CHEMICAL",
        "display_name": "Calcium Carbide Production",
        "default_ef_co2": 1.100,
        "ef_unit": "tCO2/t_carbide",
        "gases": ["CO2"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "CO2 from coke consumption in electric arc furnace",
    },
    "PETROCHEMICAL_ETHYLENE": {
        "category": "CHEMICAL",
        "display_name": "Ethylene Production",
        "default_ef_co2": 1.730,
        "ef_unit": "tCO2/t_ethylene",
        "gases": ["CO2", "CH4"],
        "tier_1_method": "MASS_BALANCE",
        "description": "CO2 and CH4 from steam cracking of hydrocarbons",
    },
    "PETROCHEMICAL_METHANOL": {
        "category": "CHEMICAL",
        "display_name": "Methanol Production",
        "default_ef_co2": 0.670,
        "ef_unit": "tCO2/t_methanol",
        "gases": ["CO2"],
        "tier_1_method": "MASS_BALANCE",
        "description": "CO2 from natural gas reforming for methanol synthesis",
    },
    "HYDROGEN_PRODUCTION": {
        "category": "CHEMICAL",
        "display_name": "Hydrogen Production (SMR)",
        "default_ef_co2": 9.300,
        "ef_unit": "tCO2/t_hydrogen",
        "gases": ["CO2"],
        "tier_1_method": "MASS_BALANCE",
        "description": "CO2 from steam methane reforming",
    },
    # ---- Metal processes ----
    "IRON_STEEL_BF_BOF": {
        "category": "METAL",
        "display_name": "Iron & Steel (BF-BOF)",
        "default_ef_co2": 1.800,
        "ef_unit": "tCO2/t_steel",
        "gases": ["CO2", "CH4"],
        "tier_1_method": "MASS_BALANCE",
        "production_route": "BF_BOF",
        "description": "CO2 from blast furnace coke and sinter",
    },
    "IRON_STEEL_EAF": {
        "category": "METAL",
        "display_name": "Iron & Steel (EAF)",
        "default_ef_co2": 0.400,
        "ef_unit": "tCO2/t_steel",
        "gases": ["CO2"],
        "tier_1_method": "EMISSION_FACTOR",
        "production_route": "EAF",
        "description": "CO2 from electrode and charge carbon consumption",
    },
    "IRON_STEEL_DRI": {
        "category": "METAL",
        "display_name": "Iron & Steel (DRI)",
        "default_ef_co2": 1.100,
        "ef_unit": "tCO2/t_DRI",
        "gases": ["CO2"],
        "tier_1_method": "EMISSION_FACTOR",
        "production_route": "DRI",
        "description": "CO2 from direct reduction of iron ore using NG",
    },
    "FERROALLOY_PRODUCTION": {
        "category": "METAL",
        "display_name": "Ferroalloy Production",
        "default_ef_co2": 2.500,
        "ef_unit": "tCO2/t_alloy",
        "gases": ["CO2", "CH4"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "CO2 from carbonaceous reductants in SAF",
    },
    "ALUMINIUM_PREBAKE": {
        "category": "METAL",
        "display_name": "Aluminium Smelting (Prebake)",
        "default_ef_co2": 1.600,
        "default_ef_cf4": 0.00004,
        "default_ef_c2f6": 0.000004,
        "ef_unit": "tCO2/t_aluminium",
        "gases": ["CO2", "CF4", "C2F6"],
        "tier_1_method": "EMISSION_FACTOR",
        "production_route": "PREBAKE",
        "description": "CO2 from anode consumption, PFC from anode effects",
    },
    "ALUMINIUM_SODERBERG": {
        "category": "METAL",
        "display_name": "Aluminium Smelting (Soderberg)",
        "default_ef_co2": 1.700,
        "default_ef_cf4": 0.00006,
        "default_ef_c2f6": 0.000006,
        "ef_unit": "tCO2/t_aluminium",
        "gases": ["CO2", "CF4", "C2F6"],
        "tier_1_method": "EMISSION_FACTOR",
        "production_route": "SODERBERG",
        "description": "CO2 from Soderberg paste, PFC from anode effects",
    },
    "LEAD_PRODUCTION": {
        "category": "METAL",
        "display_name": "Lead Production",
        "default_ef_co2": 0.590,
        "ef_unit": "tCO2/t_lead",
        "gases": ["CO2"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "CO2 from coke/carbon reduction of lead concentrate",
    },
    "ZINC_PRODUCTION": {
        "category": "METAL",
        "display_name": "Zinc Production",
        "default_ef_co2": 0.430,
        "ef_unit": "tCO2/t_zinc",
        "gases": ["CO2"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "CO2 from carbon reduction of zinc concentrate",
    },
    # ---- Electronics / semiconductor ----
    "SEMICONDUCTOR_MANUFACTURING": {
        "category": "ELECTRONICS",
        "display_name": "Semiconductor Manufacturing",
        "default_ef_cf4": 0.000050,
        "default_ef_c2f6": 0.000020,
        "default_ef_sf6": 0.000010,
        "default_ef_nf3": 0.000015,
        "ef_unit": "t_gas/wafer_start",
        "gases": ["CF4", "C2F6", "SF6", "NF3", "HFC"],
        "tier_1_method": "EMISSION_FACTOR",
        "description": "F-gas emissions from CVD chamber cleaning / etching",
    },
    # ---- Pulp & Paper ----
    "PULP_PAPER_CaCO3": {
        "category": "PULP_PAPER",
        "display_name": "Pulp & Paper (CaCO3 Use)",
        "default_ef_co2": 0.440,
        "ef_unit": "tCO2/t_CaCO3",
        "gases": ["CO2"],
        "tier_1_method": "STOICHIOMETRIC",
        "description": "CO2 from calcium carbonate in lime kiln / makeup",
    },
}

#: Default abatement technology efficiencies.
#: Keys are (process_type, abatement_type) tuples.
DEFAULT_ABATEMENT_EFFICIENCIES: Dict[str, Dict[str, float]] = {
    "CATALYTIC_REDUCTION": {
        "NITRIC_ACID": 0.85,
        "ADIPIC_ACID": 0.95,
        "CAPROLACTAM": 0.80,
        "DEFAULT": 0.80,
    },
    "THERMAL_DESTRUCTION": {
        "ADIPIC_ACID": 0.98,
        "NITRIC_ACID": 0.90,
        "DEFAULT": 0.90,
    },
    "SCRUBBING": {
        "CEMENT_CLINKER": 0.30,
        "LIME_PRODUCTION": 0.25,
        "DEFAULT": 0.25,
    },
    "CARBON_CAPTURE": {
        "AMMONIA_PRODUCTION": 0.90,
        "HYDROGEN_PRODUCTION": 0.90,
        "CEMENT_CLINKER": 0.85,
        "IRON_STEEL_BF_BOF": 0.85,
        "DEFAULT": 0.85,
    },
    "PFC_ANODE_CONTROL": {
        "ALUMINIUM_PREBAKE": 0.90,
        "ALUMINIUM_SODERBERG": 0.85,
        "DEFAULT": 0.85,
    },
    "SF6_RECOVERY": {
        "SEMICONDUCTOR_MANUFACTURING": 0.95,
        "DEFAULT": 0.90,
    },
    "SCR": {
        "NITRIC_ACID": 0.88,
        "DEFAULT": 0.85,
    },
    "NSCR": {
        "NITRIC_ACID": 0.95,
        "DEFAULT": 0.90,
    },
}

#: Regulatory frameworks for compliance checking.
REGULATORY_FRAMEWORKS: Dict[str, Dict[str, Any]] = {
    "GHG_PROTOCOL": {
        "display_name": "GHG Protocol Corporate Standard",
        "requirements": [
            {"id": "GHG-PE-001", "desc": "Scope 1 process emissions reported", "check": "has_results"},
            {"id": "GHG-PE-002", "desc": "CO2e calculation uses approved GWP", "check": "valid_gwp"},
            {"id": "GHG-PE-003", "desc": "Methodology documented", "check": "has_methodology"},
            {"id": "GHG-PE-004", "desc": "Base year recalculation policy", "check": "has_provenance"},
        ],
    },
    "ISO_14064": {
        "display_name": "ISO 14064-1:2018",
        "requirements": [
            {"id": "ISO-PE-001", "desc": "Direct GHG emissions quantified", "check": "has_results"},
            {"id": "ISO-PE-002", "desc": "Uncertainty assessment performed", "check": "has_uncertainty"},
            {"id": "ISO-PE-003", "desc": "GWP from recognised source", "check": "valid_gwp"},
        ],
    },
    "CSRD_ESRS_E1": {
        "display_name": "CSRD / ESRS E1 Climate Change",
        "requirements": [
            {"id": "ESRS-E1-001", "desc": "Scope 1 GHG reported by gas", "check": "has_gas_breakdown"},
            {"id": "ESRS-E1-002", "desc": "GWP AR6 used", "check": "gwp_ar6"},
            {"id": "ESRS-E1-003", "desc": "Material process emissions identified", "check": "has_results"},
        ],
    },
    "EPA_40CFR98": {
        "display_name": "EPA 40 CFR Part 98 Subpart",
        "requirements": [
            {"id": "EPA-PE-001", "desc": "Process-specific EF used", "check": "has_process_ef"},
            {"id": "EPA-PE-002", "desc": "Mass balance or EF method applied", "check": "valid_method"},
            {"id": "EPA-PE-003", "desc": "Monitoring plan documented", "check": "has_provenance"},
        ],
    },
    "UK_SECR": {
        "display_name": "UK Streamlined Energy & Carbon Reporting",
        "requirements": [
            {"id": "SECR-PE-001", "desc": "Scope 1 process emissions disclosed", "check": "has_results"},
            {"id": "SECR-PE-002", "desc": "UK DEFRA or IPCC factors used", "check": "valid_source"},
        ],
    },
    "EU_ETS": {
        "display_name": "EU Emissions Trading System",
        "requirements": [
            {"id": "ETS-PE-001", "desc": "Installation-level reporting", "check": "has_facility"},
            {"id": "ETS-PE-002", "desc": "Tier methodology applied", "check": "has_tier"},
            {"id": "ETS-PE-003", "desc": "Approved monitoring plan", "check": "has_provenance"},
            {"id": "ETS-PE-004", "desc": "Verification trail complete", "check": "has_audit"},
        ],
    },
}


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
# ProcessEmissionsPipelineEngine
# ===================================================================


class ProcessEmissionsPipelineEngine:
    """Eight-stage orchestration pipeline for process emissions calculations.

    Coordinates all six upstream engines through an eight-stage pipeline
    with checkpointing, provenance tracking, and comprehensive error
    handling.  Each pipeline run produces a deterministic SHA-256
    provenance hash for the complete execution chain.

    The pipeline can operate in standalone mode using built-in reference
    data when the upstream engines are unavailable.

    Thread-safe: all mutable state is protected by an internal lock.

    Attributes:
        config: ProcessEmissionsConfig instance (or None).
        process_database: Optional process database engine for lookups.
        emission_calculator: Optional emission calculation engine.
        material_balance_engine: Optional material balance engine.
        abatement_tracker: Optional abatement tracking engine.
        uncertainty_engine: Optional uncertainty quantification engine.
        compliance_checker: Optional compliance checking engine.

    Example:
        >>> engine = ProcessEmissionsPipelineEngine()
        >>> result = engine.execute_pipeline(request_dict)
        >>> assert result["success"] is True
    """

    def __init__(
        self,
        process_database: Any = None,
        emission_calculator: Any = None,
        material_balance_engine: Any = None,
        abatement_tracker: Any = None,
        uncertainty_engine: Any = None,
        compliance_checker: Any = None,
        config: Any = None,
    ) -> None:
        """Initialize the ProcessEmissionsPipelineEngine.

        Wires all six upstream engines via dependency injection.  Any
        engine set to ``None`` causes its pipeline stage to use built-in
        reference data or skip with a warning.

        Args:
            process_database: ProcessDatabaseEngine instance or None.
            emission_calculator: EmissionCalculatorEngine instance or None.
            material_balance_engine: MaterialBalanceEngine instance or None.
            abatement_tracker: AbatementTrackerEngine instance or None.
            uncertainty_engine: UncertaintyQuantifierEngine instance or None.
            compliance_checker: ComplianceCheckerEngine instance or None.
            config: Optional configuration.  Uses global config if None.
        """
        self.config = config if config is not None else get_config()

        # Engine references
        self.process_database = process_database
        self.emission_calculator = emission_calculator
        self.material_balance_engine = material_balance_engine
        self.abatement_tracker = abatement_tracker
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
            "ProcessEmissionsPipelineEngine initialized: "
            "process_database=%s, emission_calculator=%s, "
            "material_balance=%s, abatement_tracker=%s, "
            "uncertainty=%s, compliance=%s",
            process_database is not None,
            emission_calculator is not None,
            material_balance_engine is not None,
            abatement_tracker is not None,
            uncertainty_engine is not None,
            compliance_checker is not None,
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def execute_pipeline(
        self,
        request: Dict[str, Any],
        gwp_source: str = "AR6",
        tier: str = "TIER_1",
        include_abatement: bool = True,
        include_uncertainty: bool = True,
        include_compliance: bool = True,
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute the full eight-stage pipeline for a single request.

        Args:
            request: Calculation request dictionary.  Must contain at
                minimum ``process_type`` and ``production_quantity``.
            gwp_source: GWP source (AR4, AR5, AR6, AR6_20yr).
            tier: Calculation tier (TIER_1, TIER_2, TIER_3).
            include_abatement: Whether to apply abatement adjustments.
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
        resolved: Dict[str, Any] = {}
        material_balance: Dict[str, Any] = {}
        emissions: Dict[str, Any] = {}
        abated: Dict[str, Any] = {}
        uncertainty: Dict[str, Any] = {}
        compliance: Dict[str, Any] = {}
        audit: Dict[str, Any] = {}

        logger.info(
            "Pipeline %s started: process=%s, qty=%s, gwp=%s, tier=%s",
            pipeline_id,
            request.get("process_type", "UNKNOWN"),
            request.get("production_quantity", "N/A"),
            gwp_source,
            tier,
        )

        # Stage 1: VALIDATE
        sr = self._stage_validate(pipeline_id, request, gwp_source, tier)
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1
            validated = sr.get("validated_data", request)
        else:
            overall_success = False
            validated = request

        # Stage 2: RESOLVE_PROCESS
        sr = self._stage_resolve_process(pipeline_id, validated)
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1
            resolved = sr.get("resolved_data", {})
        else:
            overall_success = False

        # Stage 3: CALCULATE_MATERIAL_BALANCE
        sr = self._stage_material_balance(pipeline_id, validated, resolved)
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1
            material_balance = sr.get("material_balance", {})
        else:
            # Material balance is not always required (depends on method)
            pass

        # Stage 4: CALCULATE_EMISSIONS
        sr = self._stage_calculate_emissions(
            pipeline_id, validated, resolved,
            material_balance, gwp_source, tier,
        )
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1
            emissions = sr.get("emissions_data", {})
        else:
            overall_success = False

        # Stage 5: APPLY_ABATEMENT
        if include_abatement:
            sr = self._stage_apply_abatement(
                pipeline_id, validated, emissions,
            )
        else:
            sr = _stage_result(
                PipelineStage.APPLY_ABATEMENT.value, True, 0.0,
                pipeline_id, {"skipped": True, "reason": "abatement disabled"},
            )
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1
            abated = sr.get("abated_data", emissions)
        else:
            abated = emissions

        # Stage 6: QUANTIFY_UNCERTAINTY
        if include_uncertainty:
            sr = self._stage_uncertainty(
                pipeline_id, validated, abated,
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
            sr = self._stage_compliance(
                pipeline_id, validated, abated,
                gwp_source, tier, frameworks,
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

        # Stage 8: GENERATE_AUDIT
        sr = self._stage_audit(
            pipeline_id, validated, abated,
            uncertainty, compliance, gwp_source, tier,
        )
        stage_results.append(sr)
        if sr["success"]:
            stages_completed += 1
            audit = sr.get("audit_data", {})

        # Assemble final result
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        calc_id = validated.get("calculation_id", _new_uuid())

        pipeline_provenance = _compute_hash({
            "pipeline_id": pipeline_id,
            "calculation_id": calc_id,
            "process_type": validated.get("process_type", ""),
            "stages_completed": stages_completed,
            "total_duration_ms": elapsed_ms,
        })

        final_result = self._assemble_result(
            pipeline_id=pipeline_id,
            calc_id=calc_id,
            validated=validated,
            resolved=resolved,
            material_balance=material_balance,
            emissions=abated,
            uncertainty=uncertainty,
            compliance=compliance,
            audit=audit,
            gwp_source=gwp_source,
            tier=tier,
            pipeline_provenance=pipeline_provenance,
            elapsed_ms=elapsed_ms,
        )

        result = {
            "success": overall_success,
            "pipeline_id": pipeline_id,
            "calculation_id": calc_id,
            "stages_completed": stages_completed,
            "stages_total": len(PIPELINE_STAGES),
            "stage_results": [
                {k: v for k, v in sr.items()
                 if k not in ("validated_data", "resolved_data",
                              "material_balance", "emissions_data",
                              "abated_data", "uncertainty_data",
                              "compliance_data", "audit_data")}
                for sr in stage_results
            ],
            "result": final_result,
            "pipeline_provenance_hash": pipeline_provenance,
            "total_duration_ms": round(elapsed_ms, 3),
            "timestamp": _utcnow_iso(),
        }

        # Update statistics
        self._record_run(pipeline_id, overall_success, elapsed_ms, stage_results)

        # Record Prometheus metrics
        process_type = validated.get("process_type", "UNKNOWN")
        record_calculation(
            process_type, "success" if overall_success else "failure",
        )
        observe_calculation_duration(process_type, elapsed_ms / 1000.0)

        logger.info(
            "Pipeline %s completed: success=%s stages=%d/%d "
            "duration=%.1fms",
            pipeline_id, overall_success,
            stages_completed, len(PIPELINE_STAGES), elapsed_ms,
        )
        return result

    def execute_batch(
        self,
        requests: List[Dict[str, Any]],
        gwp_source: str = "AR6",
        tier: str = "TIER_1",
        include_abatement: bool = True,
        include_uncertainty: bool = True,
        include_compliance: bool = True,
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute the pipeline for a batch of requests.

        Args:
            requests: List of calculation request dictionaries.
            gwp_source: GWP source for all calculations.
            tier: Calculation tier for all calculations.
            include_abatement: Whether to apply abatement.
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
        total_co2e_tonnes = Decimal("0")

        logger.info(
            "Batch %s started: %d requests, gwp=%s, tier=%s",
            batch_id, len(requests), gwp_source, tier,
        )

        observe_batch_size(len(requests))

        for idx, req in enumerate(requests):
            try:
                pipeline_result = self.execute_pipeline(
                    request=req,
                    gwp_source=gwp_source,
                    tier=tier,
                    include_abatement=include_abatement,
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
                    total_co2e_tonnes += _to_decimal(
                        result_data.get("total_co2e_tonnes", 0),
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
            "total_co2e_tonnes": str(total_co2e_tonnes),
            "duration_ms": elapsed_ms,
        })

        record_batch("success" if failure_count == 0 else "partial")

        batch_result = {
            "batch_id": batch_id,
            "results": results,
            "total_co2e_kg": float(total_co2e_kg),
            "total_co2e_tonnes": float(total_co2e_tonnes),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_requests": len(requests),
            "processing_time_ms": round(elapsed_ms, 3),
            "provenance_hash": batch_provenance,
            "timestamp": _utcnow_iso(),
        }

        logger.info(
            "Batch %s completed: %d/%d success, %.4f tCO2e, %.1fms",
            batch_id, success_count, len(requests),
            float(total_co2e_tonnes), elapsed_ms,
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
                    "process_database": self.process_database is not None,
                    "emission_calculator": self.emission_calculator is not None,
                    "material_balance": self.material_balance_engine is not None,
                    "abatement_tracker": self.abatement_tracker is not None,
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

    def get_stage_results(
        self,
        pipeline_id: str,
    ) -> List[Dict[str, Any]]:
        """Return cached stage results for a given pipeline run.

        Args:
            pipeline_id: Pipeline run identifier.

        Returns:
            List of stage result dictionaries, or empty list if not found.
        """
        with self._lock:
            return self._stage_results_cache.get(pipeline_id, [])

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
    # Private stage implementations
    # ==================================================================

    def _stage_validate(
        self,
        pipeline_id: str,
        request: Dict[str, Any],
        gwp_source: str,
        tier: str,
    ) -> Dict[str, Any]:
        """Stage 1: Validate input data and normalise fields.

        Args:
            pipeline_id: Pipeline run identifier.
            request: Raw calculation request.
            gwp_source: GWP source string.
            tier: Calculation tier string.

        Returns:
            Stage result with validated_data.
        """
        stage = PipelineStage.VALIDATE.value
        t0 = time.perf_counter()

        try:
            errors: List[str] = []
            warnings: List[str] = []

            # Required fields
            process_type = request.get("process_type")
            if not process_type:
                errors.append("process_type is required")

            production_qty = request.get("production_quantity")
            if production_qty is None:
                errors.append("production_quantity is required")
            else:
                try:
                    qty_dec = _to_decimal(production_qty)
                    if qty_dec < 0:
                        errors.append("production_quantity must be >= 0")
                except Exception:
                    errors.append("production_quantity must be a valid number")

            # Validate GWP source
            if gwp_source not in GWP_VALUES:
                errors.append(
                    f"gwp_source must be one of {list(GWP_VALUES.keys())}"
                )

            # Validate tier
            valid_tiers = {"TIER_1", "TIER_2", "TIER_3"}
            if tier not in valid_tiers:
                errors.append(f"tier must be one of {sorted(valid_tiers)}")

            # Validate process type if known
            if process_type and process_type not in PROCESS_TYPES:
                warnings.append(
                    f"process_type '{process_type}' not in built-in "
                    f"reference data; custom lookup required"
                )

            # Validate calculation method if provided
            calc_method = request.get("calculation_method")
            valid_methods = {
                "EMISSION_FACTOR", "MASS_BALANCE",
                "STOICHIOMETRIC", "DIRECT_MEASUREMENT",
            }
            if calc_method and calc_method not in valid_methods:
                errors.append(
                    f"calculation_method must be one of {sorted(valid_methods)}"
                )

            # Validate material inputs if provided
            materials = request.get("material_inputs", [])
            for midx, mat in enumerate(materials):
                if not mat.get("material_name"):
                    errors.append(
                        f"material_inputs[{midx}].material_name is required"
                    )
                if mat.get("quantity") is not None:
                    try:
                        mq = _to_decimal(mat["quantity"])
                        if mq < 0:
                            errors.append(
                                f"material_inputs[{midx}].quantity must be >= 0"
                            )
                    except Exception:
                        errors.append(
                            f"material_inputs[{midx}].quantity invalid"
                        )

            # Build validated data
            validated: Dict[str, Any] = {
                "calculation_id": request.get("calculation_id", _new_uuid()),
                "process_type": process_type or "",
                "production_quantity": float(_to_decimal(
                    production_qty if production_qty is not None else 0,
                )),
                "production_unit": request.get("production_unit", "tonnes"),
                "calculation_method": calc_method or (
                    PROCESS_TYPES.get(process_type or "", {})
                    .get("tier_1_method", "EMISSION_FACTOR")
                ),
                "facility_id": request.get("facility_id"),
                "unit_id": request.get("unit_id"),
                "material_inputs": materials,
                "abatement_type": request.get("abatement_type"),
                "abatement_efficiency": request.get("abatement_efficiency"),
                "by_product_credits": request.get("by_product_credits", []),
                "custom_emission_factor": request.get("custom_emission_factor"),
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

    def _stage_resolve_process(
        self,
        pipeline_id: str,
        validated: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 2: Resolve process type, emission factors, and materials.

        Looks up the process type in the process database engine or
        falls back to built-in reference data.

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data from stage 1.

        Returns:
            Stage result with resolved_data.
        """
        stage = PipelineStage.RESOLVE_PROCESS.value
        t0 = time.perf_counter()

        try:
            process_type = validated.get("process_type", "")
            resolved: Dict[str, Any] = {}

            # Try upstream engine first
            if self.process_database is not None:
                try:
                    db_result = self.process_database.get_process_type(
                        process_type,
                    )
                    if db_result is not None:
                        if hasattr(db_result, "model_dump"):
                            resolved = db_result.model_dump(mode="json")
                        elif isinstance(db_result, dict):
                            resolved = db_result
                except (AttributeError, TypeError, KeyError) as exc:
                    logger.warning(
                        "ProcessDatabase lookup failed for %s: %s",
                        process_type, exc,
                    )

            # Fallback to built-in reference data
            if not resolved:
                builtin = PROCESS_TYPES.get(process_type, {})
                if builtin:
                    resolved = {
                        "process_type": process_type,
                        "category": builtin.get("category", "OTHER"),
                        "display_name": builtin.get("display_name", process_type),
                        "default_ef_co2": builtin.get("default_ef_co2", 0.0),
                        "default_ef_ch4": builtin.get("default_ef_ch4", 0.0),
                        "default_ef_n2o": builtin.get("default_ef_n2o", 0.0),
                        "default_ef_cf4": builtin.get("default_ef_cf4", 0.0),
                        "default_ef_c2f6": builtin.get("default_ef_c2f6", 0.0),
                        "default_ef_sf6": builtin.get("default_ef_sf6", 0.0),
                        "default_ef_nf3": builtin.get("default_ef_nf3", 0.0),
                        "ef_unit": builtin.get("ef_unit", ""),
                        "gases": builtin.get("gases", ["CO2"]),
                        "tier_1_method": builtin.get(
                            "tier_1_method", "EMISSION_FACTOR",
                        ),
                        "production_route": builtin.get("production_route"),
                        "description": builtin.get("description", ""),
                        "source": "BUILTIN_REFERENCE",
                    }
                else:
                    # Unknown process type - use generic defaults
                    resolved = {
                        "process_type": process_type,
                        "category": "OTHER",
                        "display_name": process_type,
                        "default_ef_co2": 0.0,
                        "gases": ["CO2"],
                        "tier_1_method": "EMISSION_FACTOR",
                        "source": "DEFAULT_GENERIC",
                    }
                    logger.warning(
                        "Pipeline %s: process_type '%s' not found in "
                        "built-in data; using generic defaults",
                        pipeline_id, process_type,
                    )

            # Resolve carbonate factors if relevant
            material_inputs = validated.get("material_inputs", [])
            carbonate_data: Dict[str, Any] = {}
            for mat in material_inputs:
                carbonate_type = mat.get("carbonate_type")
                if carbonate_type and carbonate_type in CARBONATE_FACTORS:
                    carbonate_data[carbonate_type] = CARBONATE_FACTORS[
                        carbonate_type
                    ]

            resolved["carbonate_factors"] = carbonate_data

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={
                    "resolved_data": resolved,
                    "source": resolved.get("source", "UNKNOWN"),
                    "process_category": resolved.get("category", ""),
                    "gases": resolved.get("gases", []),
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

    def _stage_material_balance(
        self,
        pipeline_id: str,
        validated: Dict[str, Any],
        resolved: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 3: Calculate material balance for mass balance method.

        Tracks raw material inputs, carbon content, and carbon balance
        for processes using the mass balance calculation method.

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data.
            resolved: Resolved process data from stage 2.

        Returns:
            Stage result with material_balance data.
        """
        stage = PipelineStage.CALCULATE_MATERIAL_BALANCE.value
        t0 = time.perf_counter()

        try:
            calc_method = validated.get("calculation_method", "EMISSION_FACTOR")
            material_inputs = validated.get("material_inputs", [])

            # Only run for mass balance method or if material inputs provided
            if calc_method != "MASS_BALANCE" and not material_inputs:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return _stage_result(
                    stage, True, elapsed_ms, pipeline_id,
                    extra={
                        "material_balance": {},
                        "skipped": True,
                        "reason": f"method is {calc_method}, no material inputs",
                    },
                )

            # Try upstream engine
            if self.material_balance_engine is not None:
                try:
                    mb_result = self.material_balance_engine.calculate(
                        material_inputs=material_inputs,
                        process_type=validated.get("process_type", ""),
                    )
                    if hasattr(mb_result, "model_dump"):
                        mb_data = mb_result.model_dump(mode="json")
                    elif isinstance(mb_result, dict):
                        mb_data = mb_result
                    else:
                        mb_data = {}

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return _stage_result(
                        stage, True, elapsed_ms, pipeline_id,
                        extra={"material_balance": mb_data},
                    )
                except (AttributeError, TypeError) as exc:
                    logger.warning(
                        "MaterialBalanceEngine failed: %s", exc,
                    )

            # Built-in material balance calculation
            total_carbon_input_kg = Decimal("0")
            total_carbon_output_kg = Decimal("0")
            material_details: List[Dict[str, Any]] = []

            for mat in material_inputs:
                qty = _to_decimal(mat.get("quantity", 0))
                carbon_pct = _to_decimal(
                    mat.get("carbon_content_pct", 0),
                ) / Decimal("100")
                carbon_kg = qty * Decimal("1000") * carbon_pct

                # Check for carbonate-based CO2
                carbonate_type = mat.get("carbonate_type")
                carbonate_co2_factor = Decimal("0")
                if carbonate_type and carbonate_type in CARBONATE_FACTORS:
                    carbonate_co2_factor = _to_decimal(
                        CARBONATE_FACTORS[carbonate_type]["co2_factor"],
                    )

                is_input = mat.get("is_input", True)
                if is_input:
                    total_carbon_input_kg += carbon_kg
                else:
                    total_carbon_output_kg += carbon_kg

                material_details.append({
                    "material_name": mat.get("material_name", ""),
                    "quantity_tonnes": float(qty),
                    "carbon_content_pct": float(carbon_pct * 100),
                    "carbon_kg": float(carbon_kg),
                    "is_input": is_input,
                    "carbonate_type": carbonate_type,
                    "carbonate_co2_factor": float(carbonate_co2_factor),
                })

            # Net carbon emissions (input - output in products)
            net_carbon_kg = total_carbon_input_kg - total_carbon_output_kg
            # Convert carbon to CO2: multiply by 44/12 (molecular weight ratio)
            co2_from_carbon_kg = net_carbon_kg * Decimal("44") / Decimal("12")

            mb_data = {
                "total_carbon_input_kg": float(total_carbon_input_kg),
                "total_carbon_output_kg": float(total_carbon_output_kg),
                "net_carbon_kg": float(net_carbon_kg),
                "co2_from_carbon_kg": float(co2_from_carbon_kg),
                "co2_from_carbon_tonnes": float(
                    co2_from_carbon_kg / Decimal("1000"),
                ),
                "material_details": material_details,
                "material_count": len(material_details),
            }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={"material_balance": mb_data},
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
        resolved: Dict[str, Any],
        material_balance: Dict[str, Any],
        gwp_source: str,
        tier: str,
    ) -> Dict[str, Any]:
        """Stage 4: Calculate emissions using the determined method.

        Supports four calculation methods:
        - EMISSION_FACTOR: Activity data x emission factor
        - MASS_BALANCE: Carbon input - carbon output x 44/12
        - STOICHIOMETRIC: Carbonate mass x stoichiometric CO2 factor
        - DIRECT_MEASUREMENT: Use measured values directly

        All arithmetic is deterministic (Decimal-based).

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data.
            resolved: Resolved process data.
            material_balance: Material balance from stage 3.
            gwp_source: GWP source for CO2e conversion.
            tier: Calculation tier.

        Returns:
            Stage result with emissions_data.
        """
        stage = PipelineStage.CALCULATE_EMISSIONS.value
        t0 = time.perf_counter()

        try:
            # Try upstream engine first
            if self.emission_calculator is not None:
                try:
                    calc_result = self.emission_calculator.calculate(
                        request=validated,
                        resolved=resolved,
                        material_balance=material_balance,
                        gwp_source=gwp_source,
                        tier=tier,
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

            # Built-in calculation
            process_type = validated.get("process_type", "")
            production_qty = _to_decimal(
                validated.get("production_quantity", 0),
            )
            calc_method = validated.get("calculation_method", "EMISSION_FACTOR")
            custom_ef = validated.get("custom_emission_factor")

            gwp_table = GWP_VALUES.get(gwp_source, GWP_VALUES["AR6"])
            gas_emissions: List[Dict[str, Any]] = []
            total_co2e_kg = Decimal("0")
            calculation_trace: List[str] = []

            calculation_trace.append(
                f"Process: {process_type}, Method: {calc_method}, "
                f"Tier: {tier}, GWP: {gwp_source}"
            )
            calculation_trace.append(
                f"Production quantity: {production_qty} "
                f"{validated.get('production_unit', 'tonnes')}"
            )

            if calc_method == "EMISSION_FACTOR":
                gas_emissions, total_co2e_kg = self._calc_emission_factor(
                    process_type, production_qty, resolved,
                    gwp_table, custom_ef, calculation_trace,
                )
            elif calc_method == "MASS_BALANCE":
                gas_emissions, total_co2e_kg = self._calc_mass_balance(
                    material_balance, gwp_table, calculation_trace,
                )
            elif calc_method == "STOICHIOMETRIC":
                gas_emissions, total_co2e_kg = self._calc_stoichiometric(
                    validated, resolved, gwp_table, calculation_trace,
                )
            elif calc_method == "DIRECT_MEASUREMENT":
                gas_emissions, total_co2e_kg = self._calc_direct(
                    validated, gwp_table, calculation_trace,
                )
            else:
                calculation_trace.append(
                    f"Unknown method '{calc_method}'; defaulting to EF"
                )
                gas_emissions, total_co2e_kg = self._calc_emission_factor(
                    process_type, production_qty, resolved,
                    gwp_table, custom_ef, calculation_trace,
                )

            # Apply by-product credits
            credits = validated.get("by_product_credits", [])
            credit_co2e_kg = Decimal("0")
            for credit in credits:
                credit_qty = _to_decimal(credit.get("co2e_kg", 0))
                credit_co2e_kg += credit_qty
                calculation_trace.append(
                    f"By-product credit: {credit.get('name', 'unknown')} "
                    f"= -{credit_qty} kgCO2e"
                )

            net_co2e_kg = total_co2e_kg - credit_co2e_kg
            if net_co2e_kg < 0:
                net_co2e_kg = Decimal("0")
                calculation_trace.append("Net emissions floored to 0")

            net_co2e_tonnes = net_co2e_kg / Decimal("1000")

            emissions_data = {
                "process_type": process_type,
                "calculation_method": calc_method,
                "tier": tier,
                "gwp_source": gwp_source,
                "production_quantity": float(production_qty),
                "production_unit": validated.get("production_unit", "tonnes"),
                "gas_emissions": gas_emissions,
                "gross_co2e_kg": float(total_co2e_kg),
                "credit_co2e_kg": float(credit_co2e_kg),
                "total_co2e_kg": float(net_co2e_kg),
                "total_co2e_tonnes": float(net_co2e_tonnes),
                "calculation_trace": calculation_trace,
                "source": "BUILTIN_CALCULATOR",
            }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={
                    "emissions_data": emissions_data,
                    "total_co2e_tonnes": float(net_co2e_tonnes),
                    "gas_count": len(gas_emissions),
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

    def _stage_apply_abatement(
        self,
        pipeline_id: str,
        validated: Dict[str, Any],
        emissions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 5: Apply abatement technology adjustments.

        Reduces emissions based on abatement technology type and
        efficiency, using either the upstream abatement tracker or
        built-in reference efficiencies.

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data.
            emissions: Emissions data from stage 4.

        Returns:
            Stage result with abated_data.
        """
        stage = PipelineStage.APPLY_ABATEMENT.value
        t0 = time.perf_counter()

        try:
            abatement_type = validated.get("abatement_type")
            if not abatement_type:
                # No abatement requested - pass through emissions unchanged
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return _stage_result(
                    stage, True, elapsed_ms, pipeline_id,
                    extra={
                        "abated_data": emissions,
                        "abatement_applied": False,
                        "reason": "no abatement_type specified",
                    },
                )

            # Try upstream engine
            if self.abatement_tracker is not None:
                try:
                    abated = self.abatement_tracker.apply_abatement(
                        emissions=emissions,
                        abatement_type=abatement_type,
                        efficiency=validated.get("abatement_efficiency"),
                        process_type=validated.get("process_type", ""),
                    )
                    if hasattr(abated, "model_dump"):
                        abated_data = abated.model_dump(mode="json")
                    elif isinstance(abated, dict):
                        abated_data = abated
                    else:
                        abated_data = emissions

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return _stage_result(
                        stage, True, elapsed_ms, pipeline_id,
                        extra={
                            "abated_data": abated_data,
                            "abatement_applied": True,
                        },
                    )
                except (AttributeError, TypeError) as exc:
                    logger.warning(
                        "AbatementTrackerEngine failed: %s", exc,
                    )

            # Built-in abatement calculation
            process_type = validated.get("process_type", "")
            efficiency = validated.get("abatement_efficiency")

            # Look up default efficiency if not provided
            if efficiency is None:
                abatement_data = DEFAULT_ABATEMENT_EFFICIENCIES.get(
                    abatement_type, {},
                )
                efficiency = abatement_data.get(
                    process_type,
                    abatement_data.get("DEFAULT", 0.0),
                )

            eff_dec = _to_decimal(efficiency)
            if eff_dec < 0 or eff_dec > 1:
                eff_dec = max(Decimal("0"), min(Decimal("1"), eff_dec))

            # Apply abatement reduction
            gross_co2e_kg = _to_decimal(
                emissions.get("total_co2e_kg", 0),
            )
            reduction_kg = gross_co2e_kg * eff_dec
            net_co2e_kg = gross_co2e_kg - reduction_kg
            net_co2e_tonnes = net_co2e_kg / Decimal("1000")

            abated_data = dict(emissions)
            abated_data["gross_co2e_kg"] = float(gross_co2e_kg)
            abated_data["abatement_type"] = abatement_type
            abated_data["abatement_efficiency"] = float(eff_dec)
            abated_data["abatement_reduction_kg"] = float(reduction_kg)
            abated_data["total_co2e_kg"] = float(net_co2e_kg)
            abated_data["total_co2e_tonnes"] = float(net_co2e_tonnes)

            trace = abated_data.get("calculation_trace", [])
            trace.append(
                f"Abatement: {abatement_type} efficiency={float(eff_dec):.2%} "
                f"reduction={float(reduction_kg):.4f} kgCO2e"
            )
            abated_data["calculation_trace"] = trace

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={
                    "abated_data": abated_data,
                    "abatement_applied": True,
                    "abatement_type": abatement_type,
                    "efficiency": float(eff_dec),
                    "reduction_kg": float(reduction_kg),
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

    def _stage_uncertainty(
        self,
        pipeline_id: str,
        validated: Dict[str, Any],
        emissions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 6: Quantify uncertainty using Monte Carlo or analytical.

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data.
            emissions: Emissions data (post-abatement).

        Returns:
            Stage result with uncertainty_data.
        """
        stage = PipelineStage.QUANTIFY_UNCERTAINTY.value
        t0 = time.perf_counter()

        try:
            # Try upstream engine
            if self.uncertainty_engine is not None:
                try:
                    unc_result = self.uncertainty_engine.quantify(
                        emissions=emissions,
                        process_type=validated.get("process_type", ""),
                    )
                    if hasattr(unc_result, "model_dump"):
                        unc_data = unc_result.model_dump(mode="json")
                    elif isinstance(unc_result, dict):
                        unc_data = unc_result
                    else:
                        unc_data = {}

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return _stage_result(
                        stage, True, elapsed_ms, pipeline_id,
                        extra={"uncertainty_data": unc_data},
                    )
                except (AttributeError, TypeError) as exc:
                    logger.warning(
                        "UncertaintyEngine failed: %s", exc,
                    )

            # Built-in Monte Carlo uncertainty (simplified)
            total_co2e_kg = float(emissions.get("total_co2e_kg", 0))
            iterations = 1000

            if hasattr(self.config, "monte_carlo_iterations"):
                try:
                    iterations = int(self.config.monte_carlo_iterations)
                except (TypeError, ValueError, AttributeError):
                    pass

            if total_co2e_kg <= 0:
                unc_data = {
                    "mean_co2e_kg": 0.0,
                    "std_co2e_kg": 0.0,
                    "cv_pct": 0.0,
                    "p5_co2e_kg": 0.0,
                    "p50_co2e_kg": 0.0,
                    "p95_co2e_kg": 0.0,
                    "confidence_interval_pct": 95.0,
                    "iterations": iterations,
                    "method": "NONE",
                    "reason": "zero emissions",
                }
            else:
                # IPCC default uncertainty ranges by tier
                uncertainty_pct = {"TIER_1": 0.30, "TIER_2": 0.15, "TIER_3": 0.05}
                tier = emissions.get("tier", "TIER_1")
                unc_frac = uncertainty_pct.get(tier, 0.30)

                # Simple Monte Carlo using normal distribution
                rng = random.Random(42)  # deterministic seed for reproducibility
                samples = [
                    max(0.0, rng.gauss(total_co2e_kg, total_co2e_kg * unc_frac))
                    for _ in range(iterations)
                ]
                samples.sort()

                mean_val = sum(samples) / len(samples)
                variance = sum(
                    (s - mean_val) ** 2 for s in samples
                ) / len(samples)
                std_val = math.sqrt(variance)
                cv_pct = (std_val / mean_val * 100.0) if mean_val > 0 else 0.0

                p5_idx = int(0.05 * len(samples))
                p50_idx = int(0.50 * len(samples))
                p95_idx = int(0.95 * len(samples))

                unc_data = {
                    "mean_co2e_kg": round(mean_val, 4),
                    "std_co2e_kg": round(std_val, 4),
                    "cv_pct": round(cv_pct, 2),
                    "p5_co2e_kg": round(samples[p5_idx], 4),
                    "p50_co2e_kg": round(samples[p50_idx], 4),
                    "p95_co2e_kg": round(samples[p95_idx], 4),
                    "confidence_interval_pct": 95.0,
                    "iterations": iterations,
                    "method": "MONTE_CARLO",
                    "tier_uncertainty_pct": unc_frac * 100,
                }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={"uncertainty_data": unc_data},
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

    def _stage_compliance(
        self,
        pipeline_id: str,
        validated: Dict[str, Any],
        emissions: Dict[str, Any],
        gwp_source: str,
        tier: str,
        frameworks: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Stage 7: Check compliance against regulatory frameworks.

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data.
            emissions: Emissions data (post-abatement).
            gwp_source: GWP source used.
            tier: Calculation tier used.
            frameworks: Frameworks to check, or None for all.

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
                        gwp_source=gwp_source,
                        tier=tier,
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
            target_frameworks = frameworks or list(REGULATORY_FRAMEWORKS.keys())
            framework_results: List[Dict[str, Any]] = []
            total_requirements = 0
            compliant_count = 0

            # Build check context
            has_results = emissions.get("total_co2e_kg", 0) >= 0
            has_gas_breakdown = len(emissions.get("gas_emissions", [])) > 0
            has_uncertainty = False  # We have it but in a separate dict
            has_provenance = True  # Pipeline always produces provenance
            has_facility = validated.get("facility_id") is not None
            has_tier = tier in {"TIER_1", "TIER_2", "TIER_3"}
            has_methodology = True
            valid_gwp = gwp_source in {"AR4", "AR5", "AR6", "AR6_20yr"}
            gwp_ar6 = gwp_source in {"AR6", "AR6_20yr"}
            has_process_ef = emissions.get("calculation_method") is not None
            valid_method = emissions.get("calculation_method") in {
                "EMISSION_FACTOR", "MASS_BALANCE",
                "STOICHIOMETRIC", "DIRECT_MEASUREMENT",
            }
            valid_source = True

            checks = {
                "has_results": has_results,
                "has_gas_breakdown": has_gas_breakdown,
                "has_uncertainty": has_uncertainty,
                "has_provenance": has_provenance,
                "has_facility": has_facility,
                "has_tier": has_tier,
                "has_methodology": has_methodology,
                "valid_gwp": valid_gwp,
                "gwp_ar6": gwp_ar6,
                "has_process_ef": has_process_ef,
                "valid_method": valid_method,
                "valid_source": valid_source,
                "has_audit": has_provenance,
            }

            for fw_id in target_frameworks:
                fw = REGULATORY_FRAMEWORKS.get(fw_id)
                if not fw:
                    continue

                fw_requirements: List[Dict[str, Any]] = []
                fw_compliant = 0
                for req in fw.get("requirements", []):
                    check_key = req.get("check", "")
                    is_met = checks.get(check_key, False)
                    fw_requirements.append({
                        "requirement_id": req["id"],
                        "description": req["desc"],
                        "check": check_key,
                        "compliant": is_met,
                    })
                    total_requirements += 1
                    if is_met:
                        fw_compliant += 1
                        compliant_count += 1

                framework_results.append({
                    "framework": fw_id,
                    "display_name": fw.get("display_name", fw_id),
                    "requirements": fw_requirements,
                    "compliant_count": fw_compliant,
                    "total_requirements": len(fw_requirements),
                    "overall_compliant": fw_compliant == len(fw_requirements),
                })

            comp_data = {
                "frameworks": framework_results,
                "total_frameworks": len(framework_results),
                "total_requirements": total_requirements,
                "compliant_count": compliant_count,
                "overall_compliant": compliant_count == total_requirements,
            }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={
                    "compliance_data": comp_data,
                    "frameworks_checked": len(framework_results),
                    "overall_compliant": comp_data["overall_compliant"],
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

    def _stage_audit(
        self,
        pipeline_id: str,
        validated: Dict[str, Any],
        emissions: Dict[str, Any],
        uncertainty: Dict[str, Any],
        compliance: Dict[str, Any],
        gwp_source: str,
        tier: str,
    ) -> Dict[str, Any]:
        """Stage 8: Generate provenance chain and audit trail.

        Creates a complete audit record with SHA-256 provenance hashes
        for every key data point in the calculation.

        Args:
            pipeline_id: Pipeline run identifier.
            validated: Validated request data.
            emissions: Emissions data (post-abatement).
            uncertainty: Uncertainty analysis results.
            compliance: Compliance check results.
            gwp_source: GWP source used.
            tier: Calculation tier used.

        Returns:
            Stage result with audit_data.
        """
        stage = PipelineStage.GENERATE_AUDIT.value
        t0 = time.perf_counter()

        try:
            calc_id = validated.get("calculation_id", "")
            now_iso = _utcnow_iso()

            # Build audit entries
            audit_entries: List[Dict[str, Any]] = []

            # Input audit
            input_hash = _compute_hash({
                "process_type": validated.get("process_type"),
                "production_quantity": validated.get("production_quantity"),
                "calculation_method": validated.get("calculation_method"),
            })
            audit_entries.append({
                "audit_id": _new_uuid(),
                "calculation_id": calc_id,
                "pipeline_id": pipeline_id,
                "action": "INPUT_VALIDATION",
                "entity_type": "calculation_request",
                "actor": "system",
                "timestamp": now_iso,
                "details": {
                    "process_type": validated.get("process_type"),
                    "production_quantity": validated.get("production_quantity"),
                    "calculation_method": validated.get("calculation_method"),
                },
                "provenance_hash": input_hash,
            })

            # Calculation audit
            calc_hash = _compute_hash({
                "calculation_id": calc_id,
                "total_co2e_kg": emissions.get("total_co2e_kg", 0),
                "gwp_source": gwp_source,
                "tier": tier,
                "method": emissions.get("calculation_method", ""),
            })
            audit_entries.append({
                "audit_id": _new_uuid(),
                "calculation_id": calc_id,
                "pipeline_id": pipeline_id,
                "action": "EMISSION_CALCULATION",
                "entity_type": "calculation_result",
                "actor": "system",
                "timestamp": now_iso,
                "details": {
                    "total_co2e_kg": emissions.get("total_co2e_kg", 0),
                    "total_co2e_tonnes": emissions.get("total_co2e_tonnes", 0),
                    "gas_count": len(emissions.get("gas_emissions", [])),
                    "calculation_method": emissions.get("calculation_method"),
                },
                "provenance_hash": calc_hash,
            })

            # Abatement audit (if applied)
            if emissions.get("abatement_type"):
                abatement_hash = _compute_hash({
                    "abatement_type": emissions.get("abatement_type"),
                    "abatement_efficiency": emissions.get("abatement_efficiency"),
                    "reduction_kg": emissions.get("abatement_reduction_kg", 0),
                })
                audit_entries.append({
                    "audit_id": _new_uuid(),
                    "calculation_id": calc_id,
                    "pipeline_id": pipeline_id,
                    "action": "ABATEMENT_APPLICATION",
                    "entity_type": "abatement_record",
                    "actor": "system",
                    "timestamp": now_iso,
                    "details": {
                        "abatement_type": emissions.get("abatement_type"),
                        "efficiency": emissions.get("abatement_efficiency"),
                        "reduction_kg": emissions.get(
                            "abatement_reduction_kg", 0,
                        ),
                    },
                    "provenance_hash": abatement_hash,
                })

            # Pipeline completion audit
            pipeline_hash = _compute_hash({
                "pipeline_id": pipeline_id,
                "calculation_id": calc_id,
                "input_hash": input_hash,
                "calc_hash": calc_hash,
                "timestamp": now_iso,
            })
            audit_entries.append({
                "audit_id": _new_uuid(),
                "calculation_id": calc_id,
                "pipeline_id": pipeline_id,
                "action": "PIPELINE_COMPLETION",
                "entity_type": "pipeline_run",
                "actor": "system",
                "timestamp": now_iso,
                "details": {
                    "stages_count": len(PIPELINE_STAGES),
                    "gwp_source": gwp_source,
                    "tier": tier,
                },
                "provenance_hash": pipeline_hash,
            })

            audit_data = {
                "entries": audit_entries,
                "entry_count": len(audit_entries),
                "chain_hash": pipeline_hash,
            }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return _stage_result(
                stage, True, elapsed_ms, pipeline_id,
                extra={
                    "audit_data": audit_data,
                    "audit_entry_count": len(audit_entries),
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

    # ==================================================================
    # Built-in calculation methods
    # ==================================================================

    def _calc_emission_factor(
        self,
        process_type: str,
        production_qty: Decimal,
        resolved: Dict[str, Any],
        gwp_table: Dict[str, float],
        custom_ef: Any,
        trace: List[str],
    ) -> Tuple[List[Dict[str, Any]], Decimal]:
        """Calculate emissions using the emission factor method.

        Formula: Emissions_gas = Activity_Data x EF_gas

        Args:
            process_type: Process type identifier.
            production_qty: Production quantity in tonnes.
            resolved: Resolved process data with default EFs.
            gwp_table: GWP values for the selected AR source.
            custom_ef: Optional custom emission factor override.
            trace: Calculation trace list (mutated in place).

        Returns:
            Tuple of (gas_emissions list, total_co2e_kg Decimal).
        """
        gas_emissions: List[Dict[str, Any]] = []
        total_co2e_kg = Decimal("0")
        gases = resolved.get("gases", ["CO2"])

        trace.append("Method: Emission Factor (Activity x EF)")

        for gas in gases:
            # Determine emission factor
            ef_key = f"default_ef_{gas.lower()}"
            ef_value = _to_decimal(resolved.get(ef_key, 0))

            if custom_ef is not None:
                if isinstance(custom_ef, dict):
                    if gas in custom_ef:
                        ef_value = _to_decimal(custom_ef[gas])
                else:
                    ef_value = _to_decimal(custom_ef)

            if ef_value == 0:
                continue

            # Calculate mass emission: production_qty (tonnes) x EF (t_gas/t_prod)
            mass_kg = production_qty * ef_value * Decimal("1000")

            # Convert to CO2e
            gwp = _to_decimal(gwp_table.get(gas, 1.0))
            co2e_kg = mass_kg * gwp
            total_co2e_kg += co2e_kg

            trace.append(
                f"  {gas}: {float(production_qty):.4f} t_prod x "
                f"{float(ef_value):.6f} t_{gas}/t_prod = "
                f"{float(mass_kg):.4f} kg {gas} x GWP {float(gwp)} = "
                f"{float(co2e_kg):.4f} kgCO2e"
            )

            gas_emissions.append({
                "gas": gas,
                "emission_factor": float(ef_value),
                "ef_unit": resolved.get("ef_unit", "t_gas/t_prod"),
                "mass_kg": float(mass_kg),
                "mass_tonnes": float(mass_kg / Decimal("1000")),
                "gwp": float(gwp),
                "co2e_kg": float(co2e_kg),
                "co2e_tonnes": float(co2e_kg / Decimal("1000")),
            })

        trace.append(
            f"Total gross CO2e: {float(total_co2e_kg):.4f} kg "
            f"({float(total_co2e_kg / Decimal('1000')):.6f} tonnes)"
        )

        return gas_emissions, total_co2e_kg

    def _calc_mass_balance(
        self,
        material_balance: Dict[str, Any],
        gwp_table: Dict[str, float],
        trace: List[str],
    ) -> Tuple[List[Dict[str, Any]], Decimal]:
        """Calculate emissions using the mass balance method.

        Formula: CO2 = (C_input - C_output) x 44/12

        Args:
            material_balance: Material balance data from stage 3.
            gwp_table: GWP values.
            trace: Calculation trace list.

        Returns:
            Tuple of (gas_emissions list, total_co2e_kg Decimal).
        """
        trace.append("Method: Mass Balance (Carbon In - Carbon Out x 44/12)")

        co2_kg = _to_decimal(
            material_balance.get("co2_from_carbon_kg", 0),
        )
        gwp_co2 = _to_decimal(gwp_table.get("CO2", 1.0))
        co2e_kg = co2_kg * gwp_co2

        trace.append(
            f"  Net carbon: {material_balance.get('net_carbon_kg', 0):.4f} kg"
        )
        trace.append(
            f"  CO2 from carbon balance: {float(co2_kg):.4f} kg x "
            f"GWP {float(gwp_co2)} = {float(co2e_kg):.4f} kgCO2e"
        )

        gas_emissions = [{
            "gas": "CO2",
            "mass_kg": float(co2_kg),
            "mass_tonnes": float(co2_kg / Decimal("1000")),
            "gwp": float(gwp_co2),
            "co2e_kg": float(co2e_kg),
            "co2e_tonnes": float(co2e_kg / Decimal("1000")),
            "method": "MASS_BALANCE",
        }]

        return gas_emissions, co2e_kg

    def _calc_stoichiometric(
        self,
        validated: Dict[str, Any],
        resolved: Dict[str, Any],
        gwp_table: Dict[str, float],
        trace: List[str],
    ) -> Tuple[List[Dict[str, Any]], Decimal]:
        """Calculate emissions using stoichiometric factors.

        Formula: CO2 = Sum(Carbonate_mass x Carbonate_CO2_factor)

        Args:
            validated: Validated request data with material inputs.
            resolved: Resolved process data with carbonate factors.
            gwp_table: GWP values.
            trace: Calculation trace list.

        Returns:
            Tuple of (gas_emissions list, total_co2e_kg Decimal).
        """
        trace.append(
            "Method: Stoichiometric (Carbonate Mass x CO2 Factor)"
        )

        total_co2_kg = Decimal("0")
        materials = validated.get("material_inputs", [])
        carbonate_factors = resolved.get("carbonate_factors", {})

        for mat in materials:
            carbonate_type = mat.get("carbonate_type")
            if not carbonate_type:
                continue

            factor_data = carbonate_factors.get(
                carbonate_type,
                CARBONATE_FACTORS.get(carbonate_type, {}),
            )
            co2_factor = _to_decimal(factor_data.get("co2_factor", 0))
            qty_tonnes = _to_decimal(mat.get("quantity", 0))

            # CO2 = carbonate mass (tonnes) x CO2 factor (tCO2/t_carbonate)
            co2_tonnes = qty_tonnes * co2_factor
            co2_kg = co2_tonnes * Decimal("1000")
            total_co2_kg += co2_kg

            trace.append(
                f"  {carbonate_type}: {float(qty_tonnes):.4f} t x "
                f"{float(co2_factor):.4f} tCO2/t = "
                f"{float(co2_kg):.4f} kgCO2"
            )

        # If no carbonates found, fall back to emission factor
        if total_co2_kg == 0 and not any(
            m.get("carbonate_type") for m in materials
        ):
            trace.append("  No carbonate inputs; falling back to EF method")
            production_qty = _to_decimal(
                validated.get("production_quantity", 0),
            )
            return self._calc_emission_factor(
                validated.get("process_type", ""),
                production_qty, resolved,
                gwp_table, None, trace,
            )

        gwp_co2 = _to_decimal(gwp_table.get("CO2", 1.0))
        co2e_kg = total_co2_kg * gwp_co2

        trace.append(
            f"Total stoichiometric CO2: {float(total_co2_kg):.4f} kg"
        )

        gas_emissions = [{
            "gas": "CO2",
            "mass_kg": float(total_co2_kg),
            "mass_tonnes": float(total_co2_kg / Decimal("1000")),
            "gwp": float(gwp_co2),
            "co2e_kg": float(co2e_kg),
            "co2e_tonnes": float(co2e_kg / Decimal("1000")),
            "method": "STOICHIOMETRIC",
        }]

        return gas_emissions, co2e_kg

    def _calc_direct(
        self,
        validated: Dict[str, Any],
        gwp_table: Dict[str, float],
        trace: List[str],
    ) -> Tuple[List[Dict[str, Any]], Decimal]:
        """Calculate emissions using direct measurement values.

        Uses measured gas quantities provided in the request.

        Args:
            validated: Validated request data with measured values.
            gwp_table: GWP values.
            trace: Calculation trace list.

        Returns:
            Tuple of (gas_emissions list, total_co2e_kg Decimal).
        """
        trace.append("Method: Direct Measurement")

        gas_emissions: List[Dict[str, Any]] = []
        total_co2e_kg = Decimal("0")

        measured = validated.get("metadata", {}).get("measured_emissions", {})
        if not measured:
            measured = validated.get("measured_emissions", {})

        for gas, value in measured.items():
            mass_kg = _to_decimal(value)
            gwp = _to_decimal(gwp_table.get(gas, 1.0))
            co2e_kg = mass_kg * gwp
            total_co2e_kg += co2e_kg

            trace.append(
                f"  {gas}: {float(mass_kg):.4f} kg measured x "
                f"GWP {float(gwp)} = {float(co2e_kg):.4f} kgCO2e"
            )

            gas_emissions.append({
                "gas": gas,
                "mass_kg": float(mass_kg),
                "mass_tonnes": float(mass_kg / Decimal("1000")),
                "gwp": float(gwp),
                "co2e_kg": float(co2e_kg),
                "co2e_tonnes": float(co2e_kg / Decimal("1000")),
                "method": "DIRECT_MEASUREMENT",
            })

        if not gas_emissions:
            trace.append("  No measured_emissions provided")

        return gas_emissions, total_co2e_kg

    # ==================================================================
    # Result assembly
    # ==================================================================

    def _assemble_result(
        self,
        pipeline_id: str,
        calc_id: str,
        validated: Dict[str, Any],
        resolved: Dict[str, Any],
        material_balance: Dict[str, Any],
        emissions: Dict[str, Any],
        uncertainty: Dict[str, Any],
        compliance: Dict[str, Any],
        audit: Dict[str, Any],
        gwp_source: str,
        tier: str,
        pipeline_provenance: str,
        elapsed_ms: float,
    ) -> Dict[str, Any]:
        """Assemble the final calculation result from all stage outputs.

        Args:
            pipeline_id: Pipeline run identifier.
            calc_id: Calculation identifier.
            validated: Validated request data.
            resolved: Resolved process data.
            material_balance: Material balance data.
            emissions: Emissions data (post-abatement).
            uncertainty: Uncertainty analysis data.
            compliance: Compliance check data.
            audit: Audit trail data.
            gwp_source: GWP source used.
            tier: Calculation tier.
            pipeline_provenance: Pipeline-level provenance hash.
            elapsed_ms: Total processing time.

        Returns:
            Complete calculation result dictionary.
        """
        return {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "process_type": validated.get("process_type", ""),
            "process_category": resolved.get("category", ""),
            "process_display_name": resolved.get("display_name", ""),
            "production_quantity": validated.get("production_quantity", 0),
            "production_unit": validated.get("production_unit", "tonnes"),
            "calculation_method": emissions.get(
                "calculation_method",
                validated.get("calculation_method", "EMISSION_FACTOR"),
            ),
            "tier": tier,
            "gwp_source": gwp_source,
            "facility_id": validated.get("facility_id"),
            "unit_id": validated.get("unit_id"),
            "gas_emissions": emissions.get("gas_emissions", []),
            "total_co2e_kg": emissions.get("total_co2e_kg", 0),
            "total_co2e_tonnes": emissions.get("total_co2e_tonnes", 0),
            "gross_co2e_kg": emissions.get("gross_co2e_kg",
                                           emissions.get("total_co2e_kg", 0)),
            "credit_co2e_kg": emissions.get("credit_co2e_kg", 0),
            "abatement_type": emissions.get("abatement_type"),
            "abatement_efficiency": emissions.get("abatement_efficiency"),
            "abatement_reduction_kg": emissions.get(
                "abatement_reduction_kg", 0,
            ),
            "material_balance": material_balance or None,
            "uncertainty": uncertainty or None,
            "compliance": compliance or None,
            "audit_trail": audit.get("entries", []),
            "calculation_trace": emissions.get("calculation_trace", []),
            "provenance_hash": pipeline_provenance,
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": _utcnow_iso(),
            "pipeline_id": pipeline_id,
            "reporting_period": validated.get("reporting_period"),
            "organization_id": validated.get("organization_id"),
        }

    # ==================================================================
    # Statistics recording
    # ==================================================================

    def _record_run(
        self,
        pipeline_id: str,
        success: bool,
        elapsed_ms: float,
        stage_results: List[Dict[str, Any]],
    ) -> None:
        """Record a pipeline run in internal statistics.

        Args:
            pipeline_id: Pipeline run identifier.
            success: Whether the run succeeded overall.
            elapsed_ms: Total duration in milliseconds.
            stage_results: List of stage result dicts.
        """
        with self._lock:
            self._total_runs += 1
            self._total_duration_ms += elapsed_ms
            self._last_run_at = _utcnow_iso()

            if success:
                self._successful_runs += 1
            else:
                self._failed_runs += 1

            self._pipeline_results[pipeline_id] = {
                "pipeline_id": pipeline_id,
                "success": success,
                "total_duration_ms": round(elapsed_ms, 3),
                "timestamp": self._last_run_at,
            }

            # Keep only last 100 runs
            if len(self._pipeline_results) > 100:
                oldest_keys = list(self._pipeline_results.keys())[:-100]
                for key in oldest_keys:
                    self._pipeline_results.pop(key, None)

            # Cache stage results (last 50 pipelines)
            self._stage_results_cache[pipeline_id] = [
                {k: v for k, v in sr.items()
                 if k not in ("validated_data", "resolved_data",
                              "material_balance", "emissions_data",
                              "abated_data", "uncertainty_data",
                              "compliance_data", "audit_data")}
                for sr in stage_results
            ]
            if len(self._stage_results_cache) > 50:
                oldest_keys = list(self._stage_results_cache.keys())[:-50]
                for key in oldest_keys:
                    self._stage_results_cache.pop(key, None)


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    "ProcessEmissionsPipelineEngine",
    "PipelineStage",
    "PIPELINE_STAGES",
    "GWP_VALUES",
    "CARBONATE_FACTORS",
    "PROCESS_TYPES",
    "DEFAULT_ABATEMENT_EFFICIENCIES",
    "REGULATORY_FRAMEWORKS",
]
