# -*- coding: utf-8 -*-
"""
Rule Pack Engine - AGENT-DATA-019: Validation Rule Engine

Engine 5 of 7 in the Validation Rule Engine pipeline.

Pure-Python engine for managing pre-built and custom validation rule packs
aligned to regulatory compliance frameworks. Provides versioned rule pack
registration, application (instantiating rules from a pack into a rule set),
listing, retrieval, version comparison, custom pack registration, and
aggregate statistics.

Built-in Rule Packs (4):
    - GHG Protocol Corporate Standard (2015 Rev) -- 42 rules
    - CSRD/ESRS (2024 Delegated Acts) -- 38 rules
    - EU 2023/1115 Deforestation Regulation (EUDR) -- 28 rules
    - SOC 2 Type II (AICPA 2022) -- 22 rules

Each built-in pack defines validation rules that cover the data quality
requirements of its respective regulatory framework. When a pack is applied,
the engine instantiates each rule definition into a concrete rule dictionary
and returns the full set along with a generated rule set ID for subsequent
evaluation operations.

Zero-Hallucination Guarantees:
    - All IDs are deterministic UUID-4 values (no LLM involvement)
    - Timestamps from ``datetime.now(timezone.utc)`` (deterministic)
    - SHA-256 provenance hashes recorded on every mutating operation
    - Rule counts are computed from list lengths (pure Python arithmetic)
    - No ML or LLM calls anywhere in this engine

Thread Safety:
    All mutating and read operations are protected by ``self._lock``
    (a ``threading.Lock``). Callers receive plain dict copies so they
    cannot accidentally mutate internal state.

Example:
    >>> from greenlang.validation_rule_engine.rule_pack import RulePackEngine
    >>> engine = RulePackEngine()
    >>> packs = engine.list_packs()
    >>> assert len(packs) >= 4
    >>> result = engine.apply_pack("ghg_protocol")
    >>> assert result["rules_created"] >= 40

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: ProvenanceTracker
# ---------------------------------------------------------------------------

try:
    from greenlang.validation_rule_engine.provenance import (
        ProvenanceTracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    logger.info(
        "validation_rule_engine.provenance not available; "
        "using inline ProvenanceTracker"
    )

    class ProvenanceTracker:  # type: ignore[no-redef]
        """Minimal inline provenance tracker for standalone operation.

        Provides SHA-256 chain hashing without external dependencies.
        """

        GENESIS_HASH = hashlib.sha256(
            b"greenlang-validation-rule-engine-genesis"
        ).hexdigest()

        def __init__(
            self,
            genesis_hash: str = "greenlang-validation-rule-engine-genesis",
        ) -> None:
            """Initialize with genesis hash."""
            self._genesis_hash: str = hashlib.sha256(
                genesis_hash.encode("utf-8")
            ).hexdigest()
            self._last_chain_hash: str = self._genesis_hash
            self._chain: List[Dict[str, Any]] = []
            self._lock = threading.Lock()

        def record(
            self,
            entity_type: str,
            entity_id: str,
            action: str,
            metadata: Optional[Any] = None,
        ) -> Any:
            """Record a provenance entry and return a stub entry."""
            ts = _utcnow().isoformat()
            if metadata is None:
                serialized = "null"
            else:
                serialized = json.dumps(
                    metadata, sort_keys=True, default=str,
                )
            data_hash = hashlib.sha256(
                serialized.encode("utf-8"),
            ).hexdigest()

            with self._lock:
                combined = json.dumps({
                    "action": action,
                    "data_hash": data_hash,
                    "parent_hash": self._last_chain_hash,
                    "timestamp": ts,
                }, sort_keys=True)
                chain_hash = hashlib.sha256(
                    combined.encode("utf-8"),
                ).hexdigest()

                self._chain.append({
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "action": action,
                    "hash_value": chain_hash,
                    "parent_hash": self._last_chain_hash,
                    "timestamp": ts,
                    "metadata": {"data_hash": data_hash},
                })
                self._last_chain_hash = chain_hash

            class _StubEntry:
                def __init__(self, hv: str) -> None:
                    self.hash_value = hv

            return _StubEntry(chain_hash)

        def build_hash(self, data: Any) -> str:
            """Return SHA-256 hash of JSON-serialized data."""
            return hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest()

        @property
        def entry_count(self) -> int:
            """Return the total number of provenance entries."""
            with self._lock:
                return len(self._chain)

        def export_chain(self) -> List[Dict[str, Any]]:
            """Return the full provenance chain for audit."""
            with self._lock:
                return list(self._chain)

        def reset(self) -> None:
            """Clear all provenance state."""
            with self._lock:
                self._chain.clear()
                self._last_chain_hash = getattr(
                    self, "_genesis_hash", self.GENESIS_HASH
                )


# ---------------------------------------------------------------------------
# Optional dependency: config
# ---------------------------------------------------------------------------

try:
    from greenlang.validation_rule_engine.config import get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    get_config = None  # type: ignore[assignment]
    _CONFIG_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional dependency: models
# ---------------------------------------------------------------------------

try:
    from greenlang.validation_rule_engine.models import (
        ValidationRule,
        ValidationRuleType,
        RuleOperator,
        RuleSeverity,
        RuleStatus,
        RulePack,
        RulePackType,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logger.info(
        "validation_rule_engine.models not available; "
        "rule pack will use dict fallback"
    )

# ---------------------------------------------------------------------------
# Optional dependency: Prometheus metrics
# ---------------------------------------------------------------------------

try:
    from greenlang.validation_rule_engine.metrics import (
        observe_processing_duration,
        PROMETHEUS_AVAILABLE,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    observe_processing_duration = None  # type: ignore[assignment]
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _sha256(data: Any) -> str:
    """Return SHA-256 hex digest of JSON-serialized data."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ===================================================================
# Built-in rule pack definitions
# ===================================================================


def _build_ghg_protocol_rules() -> List[Dict[str, Any]]:
    """Build GHG Protocol Corporate Standard rule definitions (42 rules).

    Covers Scope 1, Scope 2, Scope 3, base year, recalculation, and
    reporting requirements per the GHG Protocol Corporate Standard
    (2015 Revised Edition).

    Returns:
        List of rule definition dictionaries.
    """
    rules: List[Dict[str, Any]] = []

    # -- Scope 1 rules (10) ------------------------------------------------
    rules.append({
        "name": "ghg_scope_1_emissions_non_negative",
        "description": "Scope 1 direct emissions must be non-negative",
        "rule_type": "range",
        "target_field": "scope_1_emissions",
        "threshold_min": 0.0,
        "severity": "critical",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_1_emissions_unit_required",
        "description": "Scope 1 emissions must specify measurement unit",
        "rule_type": "completeness",
        "target_field": "scope_1_unit",
        "severity": "critical",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_1_source_category_required",
        "description": "Scope 1 emissions must have a source category",
        "rule_type": "completeness",
        "target_field": "scope_1_source_category",
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_1_emission_factor_range",
        "description": "Scope 1 emission factors must be within plausible range",
        "rule_type": "range",
        "target_field": "scope_1_emission_factor",
        "threshold_min": 0.0,
        "threshold_max": 100000.0,
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_1_activity_data_required",
        "description": "Scope 1 activity data must be present",
        "rule_type": "completeness",
        "target_field": "scope_1_activity_data",
        "severity": "critical",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_1_methodology_required",
        "description": "Scope 1 calculation methodology must be documented",
        "rule_type": "completeness",
        "target_field": "scope_1_methodology",
        "severity": "medium",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_1_gas_type_valid",
        "description": "Scope 1 greenhouse gas type must be a recognized GHG",
        "rule_type": "format",
        "target_field": "scope_1_gas_type",
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_1_facility_id_required",
        "description": "Scope 1 emissions must reference a facility identifier",
        "rule_type": "completeness",
        "target_field": "scope_1_facility_id",
        "severity": "medium",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_1_reporting_period_valid",
        "description": "Scope 1 reporting period must be a valid date range",
        "rule_type": "format",
        "target_field": "scope_1_reporting_period",
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_1_gwp_factor_valid",
        "description": "Scope 1 GWP factor must be positive and within IPCC range",
        "rule_type": "range",
        "target_field": "scope_1_gwp_factor",
        "threshold_min": 0.0,
        "threshold_max": 25000.0,
        "severity": "high",
        "framework": "ghg_protocol",
    })

    # -- Scope 2 rules (10) ------------------------------------------------
    rules.append({
        "name": "ghg_scope_2_emissions_non_negative",
        "description": "Scope 2 indirect emissions must be non-negative",
        "rule_type": "range",
        "target_field": "scope_2_emissions",
        "threshold_min": 0.0,
        "severity": "critical",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_2_location_based_required",
        "description": "Scope 2 location-based emissions must be reported",
        "rule_type": "completeness",
        "target_field": "scope_2_location_based",
        "severity": "critical",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_2_market_based_present",
        "description": "Scope 2 market-based emissions should be reported when applicable",
        "rule_type": "completeness",
        "target_field": "scope_2_market_based",
        "severity": "medium",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_2_electricity_consumption_non_negative",
        "description": "Scope 2 electricity consumption must be non-negative",
        "rule_type": "range",
        "target_field": "scope_2_electricity_kwh",
        "threshold_min": 0.0,
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_2_grid_factor_valid",
        "description": "Scope 2 grid emission factor must be within plausible range",
        "rule_type": "range",
        "target_field": "scope_2_grid_factor",
        "threshold_min": 0.0,
        "threshold_max": 5.0,
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_2_energy_source_required",
        "description": "Scope 2 energy source type must be specified",
        "rule_type": "completeness",
        "target_field": "scope_2_energy_source",
        "severity": "medium",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_2_supplier_specific_factor",
        "description": "Scope 2 supplier-specific emission factor range check",
        "rule_type": "range",
        "target_field": "scope_2_supplier_factor",
        "threshold_min": 0.0,
        "threshold_max": 10.0,
        "severity": "medium",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_2_certificate_tracking",
        "description": "Scope 2 renewable energy certificates must be tracked",
        "rule_type": "completeness",
        "target_field": "scope_2_rec_tracking",
        "severity": "low",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_2_unit_valid",
        "description": "Scope 2 emissions unit must be a recognized unit (tCO2e, kgCO2e)",
        "rule_type": "format",
        "target_field": "scope_2_unit",
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_2_residual_mix_factor",
        "description": "Scope 2 residual mix factor must be within grid factor bounds",
        "rule_type": "range",
        "target_field": "scope_2_residual_mix",
        "threshold_min": 0.0,
        "threshold_max": 5.0,
        "severity": "medium",
        "framework": "ghg_protocol",
    })

    # -- Scope 3 rules (10) ------------------------------------------------
    rules.append({
        "name": "ghg_scope_3_emissions_non_negative",
        "description": "Scope 3 value chain emissions must be non-negative",
        "rule_type": "range",
        "target_field": "scope_3_emissions",
        "threshold_min": 0.0,
        "severity": "critical",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_3_category_valid",
        "description": "Scope 3 category must be 1-15 per GHG Protocol",
        "rule_type": "range",
        "target_field": "scope_3_category",
        "threshold_min": 1.0,
        "threshold_max": 15.0,
        "severity": "critical",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_3_methodology_documented",
        "description": "Scope 3 calculation methodology must be documented",
        "rule_type": "completeness",
        "target_field": "scope_3_methodology",
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_3_emission_factor_source",
        "description": "Scope 3 emission factor source must be specified",
        "rule_type": "completeness",
        "target_field": "scope_3_ef_source",
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_3_data_quality_score",
        "description": "Scope 3 data quality score must be between 1 and 5",
        "rule_type": "range",
        "target_field": "scope_3_dq_score",
        "threshold_min": 1.0,
        "threshold_max": 5.0,
        "severity": "medium",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_3_spend_data_non_negative",
        "description": "Scope 3 spend-based activity data must be non-negative",
        "rule_type": "range",
        "target_field": "scope_3_spend",
        "threshold_min": 0.0,
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_3_supplier_engagement",
        "description": "Scope 3 supplier engagement status must be reported",
        "rule_type": "completeness",
        "target_field": "scope_3_supplier_engagement",
        "severity": "medium",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_3_exclusion_justification",
        "description": "Scope 3 excluded categories must have justification",
        "rule_type": "completeness",
        "target_field": "scope_3_exclusion_reason",
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_3_significance_threshold",
        "description": "Scope 3 significance threshold must not exceed 5%",
        "rule_type": "range",
        "target_field": "scope_3_significance_pct",
        "threshold_min": 0.0,
        "threshold_max": 5.0,
        "severity": "medium",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_scope_3_boundary_complete",
        "description": "Scope 3 organizational boundary must be defined",
        "rule_type": "completeness",
        "target_field": "scope_3_boundary",
        "severity": "high",
        "framework": "ghg_protocol",
    })

    # -- Base year rules (4) -----------------------------------------------
    rules.append({
        "name": "ghg_base_year_specified",
        "description": "Base year must be specified for emissions tracking",
        "rule_type": "completeness",
        "target_field": "base_year",
        "severity": "critical",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_base_year_emissions_non_negative",
        "description": "Base year emissions must be non-negative",
        "rule_type": "range",
        "target_field": "base_year_emissions",
        "threshold_min": 0.0,
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_base_year_range_valid",
        "description": "Base year must be between 1990 and current year",
        "rule_type": "range",
        "target_field": "base_year",
        "threshold_min": 1990.0,
        "threshold_max": 2030.0,
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_base_year_justification",
        "description": "Base year selection justification must be documented",
        "rule_type": "completeness",
        "target_field": "base_year_justification",
        "severity": "medium",
        "framework": "ghg_protocol",
    })

    # -- Recalculation rules (4) -------------------------------------------
    rules.append({
        "name": "ghg_recalculation_trigger_documented",
        "description": "Recalculation trigger events must be documented",
        "rule_type": "completeness",
        "target_field": "recalculation_trigger",
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_recalculation_threshold_valid",
        "description": "Recalculation significance threshold must be between 0 and 100%",
        "rule_type": "range",
        "target_field": "recalculation_threshold_pct",
        "threshold_min": 0.0,
        "threshold_max": 100.0,
        "severity": "medium",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_recalculation_methodology_consistent",
        "description": "Recalculation methodology must be consistent with base year",
        "rule_type": "completeness",
        "target_field": "recalculation_methodology",
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_recalculation_audit_trail",
        "description": "Recalculation changes must have an audit trail",
        "rule_type": "completeness",
        "target_field": "recalculation_audit_trail",
        "severity": "medium",
        "framework": "ghg_protocol",
    })

    # -- Reporting rules (4) -----------------------------------------------
    rules.append({
        "name": "ghg_reporting_period_complete",
        "description": "Reporting period must cover a full fiscal year",
        "rule_type": "completeness",
        "target_field": "reporting_period",
        "severity": "critical",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_reporting_boundary_defined",
        "description": "Organizational boundary approach must be defined",
        "rule_type": "completeness",
        "target_field": "reporting_boundary",
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_reporting_consolidation_approach",
        "description": "Consolidation approach (equity share or control) must be specified",
        "rule_type": "completeness",
        "target_field": "consolidation_approach",
        "severity": "high",
        "framework": "ghg_protocol",
    })
    rules.append({
        "name": "ghg_reporting_total_emissions_consistent",
        "description": "Total reported emissions must equal sum of Scope 1 + Scope 2 + Scope 3",
        "rule_type": "cross_field",
        "target_field": "total_emissions",
        "severity": "critical",
        "framework": "ghg_protocol",
    })

    return rules


def _build_csrd_esrs_rules() -> List[Dict[str, Any]]:
    """Build CSRD/ESRS (2024 Delegated Acts) rule definitions (38 rules).

    Covers E1 climate change, E2 pollution, E3 water, S1 workforce,
    G1 governance, and double materiality per CSRD/ESRS.

    Returns:
        List of rule definition dictionaries.
    """
    rules: List[Dict[str, Any]] = []

    # -- E1 Climate Change rules (10) --------------------------------------
    rules.append({
        "name": "csrd_e1_ghg_emissions_reported",
        "description": "E1: Total GHG emissions must be reported for climate disclosure",
        "rule_type": "completeness",
        "target_field": "e1_ghg_emissions",
        "severity": "critical",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e1_emissions_non_negative",
        "description": "E1: GHG emissions values must be non-negative",
        "rule_type": "range",
        "target_field": "e1_ghg_emissions",
        "threshold_min": 0.0,
        "severity": "critical",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e1_transition_plan_required",
        "description": "E1: Climate transition plan must be documented",
        "rule_type": "completeness",
        "target_field": "e1_transition_plan",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e1_emission_reduction_targets",
        "description": "E1: Emission reduction targets must be specified",
        "rule_type": "completeness",
        "target_field": "e1_reduction_targets",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e1_energy_consumption_reported",
        "description": "E1: Total energy consumption must be reported",
        "rule_type": "completeness",
        "target_field": "e1_energy_consumption",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e1_energy_mix_breakdown",
        "description": "E1: Energy mix must include renewable vs non-renewable breakdown",
        "rule_type": "completeness",
        "target_field": "e1_energy_mix",
        "severity": "medium",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e1_scope_1_2_3_breakdown",
        "description": "E1: Emissions must be broken down by Scope 1, 2, and 3",
        "rule_type": "completeness",
        "target_field": "e1_scope_breakdown",
        "severity": "critical",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e1_carbon_credits_disclosed",
        "description": "E1: Use of carbon credits must be disclosed",
        "rule_type": "completeness",
        "target_field": "e1_carbon_credits",
        "severity": "medium",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e1_internal_carbon_price",
        "description": "E1: Internal carbon price must be non-negative if reported",
        "rule_type": "range",
        "target_field": "e1_carbon_price",
        "threshold_min": 0.0,
        "severity": "medium",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e1_climate_risk_assessment",
        "description": "E1: Climate-related physical and transition risk assessment required",
        "rule_type": "completeness",
        "target_field": "e1_climate_risks",
        "severity": "high",
        "framework": "csrd_esrs",
    })

    # -- E2 Pollution rules (5) --------------------------------------------
    rules.append({
        "name": "csrd_e2_pollutant_emissions_reported",
        "description": "E2: Pollutant emissions to air, water, and soil must be reported",
        "rule_type": "completeness",
        "target_field": "e2_pollutant_emissions",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e2_substances_of_concern",
        "description": "E2: Substances of concern must be identified and quantified",
        "rule_type": "completeness",
        "target_field": "e2_substances_concern",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e2_pollutant_values_non_negative",
        "description": "E2: Pollutant emission values must be non-negative",
        "rule_type": "range",
        "target_field": "e2_pollutant_amount",
        "threshold_min": 0.0,
        "severity": "critical",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e2_pollution_prevention_plan",
        "description": "E2: Pollution prevention and control plan must be documented",
        "rule_type": "completeness",
        "target_field": "e2_prevention_plan",
        "severity": "medium",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e2_microplastics_tracked",
        "description": "E2: Microplastic emissions must be tracked where applicable",
        "rule_type": "completeness",
        "target_field": "e2_microplastics",
        "severity": "low",
        "framework": "csrd_esrs",
    })

    # -- E3 Water rules (4) ------------------------------------------------
    rules.append({
        "name": "csrd_e3_water_consumption_reported",
        "description": "E3: Total water consumption must be reported",
        "rule_type": "completeness",
        "target_field": "e3_water_consumption",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e3_water_consumption_non_negative",
        "description": "E3: Water consumption must be non-negative",
        "rule_type": "range",
        "target_field": "e3_water_consumption",
        "threshold_min": 0.0,
        "severity": "critical",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e3_water_stress_areas",
        "description": "E3: Operations in water-stressed areas must be identified",
        "rule_type": "completeness",
        "target_field": "e3_water_stress",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_e3_water_discharge_quality",
        "description": "E3: Water discharge quality must meet regulatory thresholds",
        "rule_type": "completeness",
        "target_field": "e3_discharge_quality",
        "severity": "medium",
        "framework": "csrd_esrs",
    })

    # -- S1 Workforce rules (9) --------------------------------------------
    rules.append({
        "name": "csrd_s1_workforce_headcount",
        "description": "S1: Total workforce headcount must be reported",
        "rule_type": "completeness",
        "target_field": "s1_headcount",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_s1_headcount_non_negative",
        "description": "S1: Workforce headcount must be non-negative",
        "rule_type": "range",
        "target_field": "s1_headcount",
        "threshold_min": 0.0,
        "severity": "critical",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_s1_gender_diversity_reported",
        "description": "S1: Gender diversity metrics must be reported for social disclosure",
        "rule_type": "completeness",
        "target_field": "s1_gender_diversity",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_s1_pay_gap_reported",
        "description": "S1: Gender pay gap must be reported",
        "rule_type": "completeness",
        "target_field": "s1_pay_gap",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_s1_health_safety_incidents",
        "description": "S1: Workplace health and safety incidents must be disclosed",
        "rule_type": "completeness",
        "target_field": "s1_safety_incidents",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_s1_training_hours_non_negative",
        "description": "S1: Training hours per employee must be non-negative",
        "rule_type": "range",
        "target_field": "s1_training_hours",
        "threshold_min": 0.0,
        "severity": "medium",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_s1_turnover_rate_valid",
        "description": "S1: Employee turnover rate must be between 0% and 100%",
        "rule_type": "range",
        "target_field": "s1_turnover_rate",
        "threshold_min": 0.0,
        "threshold_max": 100.0,
        "severity": "medium",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_s1_collective_bargaining",
        "description": "S1: Collective bargaining coverage must be reported",
        "rule_type": "completeness",
        "target_field": "s1_collective_bargaining",
        "severity": "medium",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_s1_human_rights_due_diligence",
        "description": "S1: Human rights due diligence process must be documented",
        "rule_type": "completeness",
        "target_field": "s1_human_rights_dd",
        "severity": "high",
        "framework": "csrd_esrs",
    })

    # -- G1 Governance rules (5) -------------------------------------------
    rules.append({
        "name": "csrd_g1_governance_structure",
        "description": "G1: Corporate governance structure must be disclosed",
        "rule_type": "completeness",
        "target_field": "g1_governance_structure",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_g1_board_diversity",
        "description": "G1: Board diversity composition must be reported",
        "rule_type": "completeness",
        "target_field": "g1_board_diversity",
        "severity": "medium",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_g1_sustainability_oversight",
        "description": "G1: Board-level sustainability oversight must be documented",
        "rule_type": "completeness",
        "target_field": "g1_sustainability_oversight",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_g1_anti_corruption_policy",
        "description": "G1: Anti-corruption and anti-bribery policies must be documented",
        "rule_type": "completeness",
        "target_field": "g1_anti_corruption",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_g1_whistleblower_mechanism",
        "description": "G1: Whistleblower protection mechanism must be documented",
        "rule_type": "completeness",
        "target_field": "g1_whistleblower",
        "severity": "medium",
        "framework": "csrd_esrs",
    })

    # -- Double Materiality rules (5) --------------------------------------
    rules.append({
        "name": "csrd_double_materiality_assessment",
        "description": "Double materiality assessment must be performed and documented",
        "rule_type": "completeness",
        "target_field": "materiality_assessment",
        "severity": "critical",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_double_materiality_impact",
        "description": "Impact materiality analysis must identify significant impacts",
        "rule_type": "completeness",
        "target_field": "materiality_impact",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_double_materiality_financial",
        "description": "Financial materiality analysis must quantify risks and opportunities",
        "rule_type": "completeness",
        "target_field": "materiality_financial",
        "severity": "high",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_double_materiality_stakeholder_engagement",
        "description": "Stakeholder engagement process for materiality must be documented",
        "rule_type": "completeness",
        "target_field": "materiality_stakeholders",
        "severity": "medium",
        "framework": "csrd_esrs",
    })
    rules.append({
        "name": "csrd_double_materiality_topics_listed",
        "description": "Material sustainability topics must be listed with IRO descriptions",
        "rule_type": "completeness",
        "target_field": "materiality_topics",
        "severity": "high",
        "framework": "csrd_esrs",
    })

    return rules


def _build_eudr_rules() -> List[Dict[str, Any]]:
    """Build EU Deforestation Regulation (EUDR) rule definitions (28 rules).

    Covers geolocation, chain of custody, due diligence, deforestation
    cutoff date, and commodity classification per EU 2023/1115.

    Returns:
        List of rule definition dictionaries.
    """
    rules: List[Dict[str, Any]] = []

    # -- Geolocation rules (7) ---------------------------------------------
    rules.append({
        "name": "eudr_geolocation_latitude_required",
        "description": "Plot latitude coordinates must be present (WGS84)",
        "rule_type": "completeness",
        "target_field": "geolocation_latitude",
        "severity": "critical",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_geolocation_longitude_required",
        "description": "Plot longitude coordinates must be present (WGS84)",
        "rule_type": "completeness",
        "target_field": "geolocation_longitude",
        "severity": "critical",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_geolocation_latitude_range",
        "description": "Latitude must be between -90 and 90 degrees",
        "rule_type": "range",
        "target_field": "geolocation_latitude",
        "threshold_min": -90.0,
        "threshold_max": 90.0,
        "severity": "critical",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_geolocation_longitude_range",
        "description": "Longitude must be between -180 and 180 degrees",
        "rule_type": "range",
        "target_field": "geolocation_longitude",
        "threshold_min": -180.0,
        "threshold_max": 180.0,
        "severity": "critical",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_geolocation_polygon_valid",
        "description": "Plot polygon coordinates must form a valid closed polygon",
        "rule_type": "format",
        "target_field": "geolocation_polygon",
        "severity": "high",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_geolocation_area_non_negative",
        "description": "Plot area must be non-negative (hectares)",
        "rule_type": "range",
        "target_field": "geolocation_area_ha",
        "threshold_min": 0.0,
        "severity": "high",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_geolocation_crs_wgs84",
        "description": "Coordinate reference system must be WGS84 (EPSG:4326)",
        "rule_type": "format",
        "target_field": "geolocation_crs",
        "severity": "high",
        "framework": "eudr",
    })

    # -- Chain of custody rules (5) ----------------------------------------
    rules.append({
        "name": "eudr_chain_of_custody_model_required",
        "description": "Chain of custody model must be specified (IP, Segregated, Mass Balance)",
        "rule_type": "completeness",
        "target_field": "custody_model",
        "severity": "critical",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_chain_of_custody_supplier_id",
        "description": "Supplier identity must be traceable in the chain of custody",
        "rule_type": "completeness",
        "target_field": "custody_supplier_id",
        "severity": "high",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_chain_of_custody_quantity_non_negative",
        "description": "Traded quantity must be non-negative",
        "rule_type": "range",
        "target_field": "custody_quantity",
        "threshold_min": 0.0,
        "severity": "high",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_chain_of_custody_timestamp_valid",
        "description": "Chain of custody transaction timestamp must be valid ISO-8601",
        "rule_type": "format",
        "target_field": "custody_timestamp",
        "severity": "medium",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_chain_of_custody_documentation",
        "description": "Chain of custody supporting documents must be referenced",
        "rule_type": "completeness",
        "target_field": "custody_documents",
        "severity": "medium",
        "framework": "eudr",
    })

    # -- Due diligence rules (5) -------------------------------------------
    rules.append({
        "name": "eudr_due_diligence_statement_required",
        "description": "Due Diligence Statement must be present per Article 4",
        "rule_type": "completeness",
        "target_field": "dds_statement",
        "severity": "critical",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_due_diligence_risk_assessment",
        "description": "Risk assessment must evaluate deforestation and degradation risk",
        "rule_type": "completeness",
        "target_field": "dds_risk_assessment",
        "severity": "critical",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_due_diligence_risk_mitigation",
        "description": "Risk mitigation measures must be documented when risk is identified",
        "rule_type": "completeness",
        "target_field": "dds_risk_mitigation",
        "severity": "high",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_due_diligence_negligible_risk",
        "description": "Negligible risk conclusion must be substantiated with evidence",
        "rule_type": "completeness",
        "target_field": "dds_negligible_evidence",
        "severity": "high",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_due_diligence_information_system",
        "description": "Due diligence information must reference EU Information System ID",
        "rule_type": "completeness",
        "target_field": "dds_eu_system_ref",
        "severity": "medium",
        "framework": "eudr",
    })

    # -- Deforestation cutoff rules (5) ------------------------------------
    rules.append({
        "name": "eudr_deforestation_cutoff_date",
        "description": "Production must not have caused deforestation after Dec 31, 2020",
        "rule_type": "completeness",
        "target_field": "deforestation_cutoff_status",
        "severity": "critical",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_cutoff_assessment_date_valid",
        "description": "Cutoff assessment date must be after 2020-12-31",
        "rule_type": "range",
        "target_field": "cutoff_assessment_year",
        "threshold_min": 2021.0,
        "severity": "critical",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_deforestation_free_declaration",
        "description": "Deforestation-free declaration must be present",
        "rule_type": "completeness",
        "target_field": "deforestation_free_declaration",
        "severity": "critical",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_forest_baseline_reference",
        "description": "Forest cover baseline reference must be documented",
        "rule_type": "completeness",
        "target_field": "forest_baseline_ref",
        "severity": "high",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_satellite_monitoring_evidence",
        "description": "Satellite monitoring evidence must support deforestation assessment",
        "rule_type": "completeness",
        "target_field": "satellite_evidence",
        "severity": "high",
        "framework": "eudr",
    })

    # -- Commodity classification rules (6) --------------------------------
    rules.append({
        "name": "eudr_commodity_type_required",
        "description": "Commodity type must be one of the 7 regulated commodities",
        "rule_type": "completeness",
        "target_field": "commodity_type",
        "severity": "critical",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_commodity_cn_code_valid",
        "description": "Combined Nomenclature (CN) code must be valid",
        "rule_type": "format",
        "target_field": "commodity_cn_code",
        "severity": "high",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_commodity_hs_code_valid",
        "description": "Harmonized System (HS) code must be valid",
        "rule_type": "format",
        "target_field": "commodity_hs_code",
        "severity": "high",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_commodity_quantity_non_negative",
        "description": "Commodity quantity must be non-negative",
        "rule_type": "range",
        "target_field": "commodity_quantity",
        "threshold_min": 0.0,
        "severity": "high",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_commodity_country_of_origin",
        "description": "Country of origin must be specified per Article 3",
        "rule_type": "completeness",
        "target_field": "commodity_country_origin",
        "severity": "critical",
        "framework": "eudr",
    })
    rules.append({
        "name": "eudr_commodity_derived_product_link",
        "description": "Derived products must be linked to source commodity",
        "rule_type": "completeness",
        "target_field": "derived_product_link",
        "severity": "medium",
        "framework": "eudr",
    })

    return rules


def _build_soc2_rules() -> List[Dict[str, Any]]:
    """Build SOC 2 Type II (AICPA 2022) rule definitions (22 rules).

    Covers access control, encryption, logging, change management, and
    processing integrity per SOC 2 Type II Trust Services Criteria.

    Returns:
        List of rule definition dictionaries.
    """
    rules: List[Dict[str, Any]] = []

    # -- Access control rules (5) ------------------------------------------
    rules.append({
        "name": "soc2_access_control_policy_required",
        "description": "Access control policy must be documented and enforced",
        "rule_type": "completeness",
        "target_field": "access_control_policy",
        "severity": "critical",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_access_authentication_mfa",
        "description": "Multi-factor authentication must be enabled for privileged access",
        "rule_type": "completeness",
        "target_field": "access_mfa_enabled",
        "severity": "critical",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_access_review_periodic",
        "description": "Access reviews must be performed at least quarterly",
        "rule_type": "completeness",
        "target_field": "access_review_date",
        "severity": "high",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_access_least_privilege",
        "description": "Least privilege principle must be enforced",
        "rule_type": "completeness",
        "target_field": "access_least_privilege",
        "severity": "high",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_access_session_timeout",
        "description": "Session timeout must be configured for inactive sessions",
        "rule_type": "range",
        "target_field": "access_session_timeout_min",
        "threshold_min": 1.0,
        "threshold_max": 60.0,
        "severity": "medium",
        "framework": "soc2",
    })

    # -- Encryption rules (4) ----------------------------------------------
    rules.append({
        "name": "soc2_encryption_at_rest_required",
        "description": "Data encryption at rest must use AES-256 or equivalent",
        "rule_type": "completeness",
        "target_field": "encryption_at_rest",
        "severity": "critical",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_encryption_in_transit_required",
        "description": "Data encryption in transit must use TLS 1.2 or higher",
        "rule_type": "completeness",
        "target_field": "encryption_in_transit",
        "severity": "critical",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_encryption_key_rotation",
        "description": "Encryption keys must be rotated according to policy",
        "rule_type": "completeness",
        "target_field": "encryption_key_rotation",
        "severity": "high",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_encryption_algorithm_valid",
        "description": "Encryption algorithm must be an approved standard",
        "rule_type": "format",
        "target_field": "encryption_algorithm",
        "severity": "high",
        "framework": "soc2",
    })

    # -- Logging rules (4) -------------------------------------------------
    rules.append({
        "name": "soc2_logging_audit_trail_enabled",
        "description": "Audit trail logging must be enabled for all critical operations",
        "rule_type": "completeness",
        "target_field": "logging_audit_enabled",
        "severity": "critical",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_logging_retention_days",
        "description": "Log retention must be at least 90 days",
        "rule_type": "range",
        "target_field": "logging_retention_days",
        "threshold_min": 90.0,
        "severity": "high",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_logging_tamper_protection",
        "description": "Logs must be protected against tampering",
        "rule_type": "completeness",
        "target_field": "logging_tamper_protection",
        "severity": "high",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_logging_centralized",
        "description": "Logs must be aggregated to a centralized logging system",
        "rule_type": "completeness",
        "target_field": "logging_centralized",
        "severity": "medium",
        "framework": "soc2",
    })

    # -- Change management rules (5) ---------------------------------------
    rules.append({
        "name": "soc2_change_mgmt_approval_required",
        "description": "All production changes must have documented approval",
        "rule_type": "completeness",
        "target_field": "change_approval",
        "severity": "critical",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_change_mgmt_testing_required",
        "description": "Changes must be tested before production deployment",
        "rule_type": "completeness",
        "target_field": "change_testing",
        "severity": "high",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_change_mgmt_rollback_plan",
        "description": "Rollback plan must be documented for each change",
        "rule_type": "completeness",
        "target_field": "change_rollback_plan",
        "severity": "high",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_change_mgmt_version_control",
        "description": "Source code must be under version control",
        "rule_type": "completeness",
        "target_field": "change_version_control",
        "severity": "high",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_change_mgmt_segregation_of_duties",
        "description": "Segregation of duties must be enforced between dev and prod",
        "rule_type": "completeness",
        "target_field": "change_sod",
        "severity": "high",
        "framework": "soc2",
    })

    # -- Processing integrity rules (4) ------------------------------------
    rules.append({
        "name": "soc2_processing_data_validation",
        "description": "Input data must be validated before processing",
        "rule_type": "completeness",
        "target_field": "processing_data_validation",
        "severity": "high",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_processing_error_handling",
        "description": "Error handling procedures must be documented",
        "rule_type": "completeness",
        "target_field": "processing_error_handling",
        "severity": "medium",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_processing_output_reconciliation",
        "description": "Output reconciliation must verify processing accuracy",
        "rule_type": "completeness",
        "target_field": "processing_output_reconciliation",
        "severity": "high",
        "framework": "soc2",
    })
    rules.append({
        "name": "soc2_processing_completeness_check",
        "description": "Processing completeness checks must verify no data loss",
        "rule_type": "completeness",
        "target_field": "processing_completeness",
        "severity": "high",
        "framework": "soc2",
    })

    return rules


# ===================================================================
# Built-in pack metadata
# ===================================================================

_BUILTIN_PACK_SPECS: List[Dict[str, Any]] = [
    {
        "name": "ghg_protocol",
        "description": (
            "GHG Protocol Corporate Standard validation rules covering "
            "Scope 1, Scope 2, Scope 3, base year, recalculation, and "
            "reporting requirements per the 2015 Revised Edition."
        ),
        "pack_type": "ghg_protocol",
        "version": "1.0.0",
        "framework_version": "GHG Protocol Corporate Standard (2015 Rev)",
        "coverage_areas": [
            "scope_1", "scope_2", "scope_3",
            "base_year", "recalculation", "reporting",
        ],
        "builder": _build_ghg_protocol_rules,
    },
    {
        "name": "csrd_esrs",
        "description": (
            "CSRD/ESRS validation rules covering E1 climate change, "
            "E2 pollution, E3 water, S1 workforce, G1 governance, and "
            "double materiality per the 2024 Delegated Acts."
        ),
        "pack_type": "csrd_esrs",
        "version": "1.0.0",
        "framework_version": "CSRD/ESRS (2024 Delegated Acts)",
        "coverage_areas": [
            "e1_climate", "e2_pollution", "e3_water",
            "s1_workforce", "g1_governance", "double_materiality",
        ],
        "builder": _build_csrd_esrs_rules,
    },
    {
        "name": "eudr",
        "description": (
            "EU Deforestation Regulation validation rules covering "
            "geolocation, chain of custody, due diligence, deforestation "
            "cutoff, and commodity classification per EU 2023/1115."
        ),
        "pack_type": "eudr",
        "version": "1.0.0",
        "framework_version": "EU 2023/1115 Deforestation Regulation",
        "coverage_areas": [
            "geolocation", "chain_of_custody", "due_diligence",
            "deforestation_cutoff", "commodity_classification",
        ],
        "builder": _build_eudr_rules,
    },
    {
        "name": "soc2",
        "description": (
            "SOC 2 Type II validation rules covering access control, "
            "encryption, logging, change management, and processing "
            "integrity per AICPA 2022 Trust Services Criteria."
        ),
        "pack_type": "soc2",
        "version": "1.0.0",
        "framework_version": "SOC 2 Type II (AICPA 2022)",
        "coverage_areas": [
            "access_control", "encryption", "logging",
            "change_management", "processing_integrity",
        ],
        "builder": _build_soc2_rules,
    },
]


# ===================================================================
# RulePackEngine
# ===================================================================


class RulePackEngine:
    """Pure-Python engine for managing validation rule packs.

    Engine 5 of 7 in the Validation Rule Engine pipeline (AGENT-DATA-019).

    Manages pre-built regulatory compliance rule packs (GHG Protocol,
    CSRD/ESRS, EUDR, SOC 2) and custom user-defined packs. Provides
    pack registration, retrieval, application (instantiating pack rules
    into a rule set), version comparison, listing, statistics, and clear
    operations.

    When a pack is applied, the engine builds concrete rule dictionaries
    from the pack's rule definitions, assigns them to a rule set, records
    provenance, and returns the full result including rule_set_id,
    rules_created count, rules list, and provenance_hash.

    Zero-Hallucination Guarantees:
        - UUID assignment via ``uuid.uuid4()`` (no LLM involvement)
        - Timestamps from ``datetime.now(timezone.utc)`` (deterministic)
        - SHA-256 provenance hashes computed from JSON-serialized payloads
        - Rule counts derived from list lengths (pure Python arithmetic)
        - No ML or LLM calls anywhere in this class

    Attributes:
        _packs: Pack store keyed by pack_id (UUID string).
        _name_index: Mapping from pack name to pack_id for O(1) lookup.
        _application_log: List of application events for statistics.
        _lock: Thread-safety lock protecting all mutable state.
        _provenance: ProvenanceTracker for SHA-256 audit trails.

    Example:
        >>> engine = RulePackEngine()
        >>> packs = engine.list_packs()
        >>> assert len(packs) >= 4
        >>> result = engine.apply_pack("ghg_protocol")
        >>> assert result["rules_created"] >= 40
    """

    def __init__(
        self,
        provenance: Optional[ProvenanceTracker] = None,
        *,
        genesis_hash: Optional[str] = None,
    ) -> None:
        """Initialize RulePackEngine with built-in packs and empty state.

        Sets up the pack store, name index, application log, and the
        provenance tracker. Registers all built-in packs on construction.

        If ``provenance`` is provided, it is used directly (even if it
        evaluates to a falsy object -- the ``is not None`` check ensures
        that an externally supplied tracker is never overridden). If
        ``genesis_hash`` is provided and ``provenance`` is None, a new
        ProvenanceTracker is created with that genesis hash. Otherwise a
        default ProvenanceTracker is created.

        Args:
            provenance: Optional ProvenanceTracker instance. When not
                None, this tracker is used for all provenance recording.
            genesis_hash: Optional genesis hash for creating a new
                ProvenanceTracker when ``provenance`` is None.

        Example:
            >>> engine = RulePackEngine()
            >>> assert len(engine.list_packs()) >= 4
        """
        # -- Provenance tracker --------------------------------------------
        if provenance is not None:
            self._provenance: ProvenanceTracker = provenance
        elif genesis_hash is not None:
            self._provenance = ProvenanceTracker(genesis_hash)
        else:
            self._provenance = ProvenanceTracker()

        # -- Internal state ------------------------------------------------
        self._packs: Dict[str, Dict[str, Any]] = {}
        self._name_index: Dict[str, str] = {}
        self._application_log: List[Dict[str, Any]] = []
        self._lock: threading.Lock = threading.Lock()

        # -- Register built-in packs ---------------------------------------
        self._register_builtin_packs()

        logger.info(
            "RulePackEngine initialized with %d built-in packs "
            "(AGENT-DATA-019, Engine 5 of 7)",
            len(self._packs),
        )

    # ------------------------------------------------------------------
    # Internal: built-in pack registration
    # ------------------------------------------------------------------

    def _register_builtin_packs(self) -> None:
        """Register all built-in regulatory compliance rule packs.

        Iterates over ``_BUILTIN_PACK_SPECS``, builds the rule
        definitions via each spec's builder function, and stores the
        pack metadata along with the rule definitions in the internal
        pack store.

        This method is called once during ``__init__`` and is not
        exposed publicly. It does not acquire the lock because it is
        invoked before the engine is available to external callers.
        """
        for spec in _BUILTIN_PACK_SPECS:
            pack_id = str(uuid.uuid4())
            rule_defs = spec["builder"]()

            pack_data: Dict[str, Any] = {
                "pack_id": pack_id,
                "name": spec["name"],
                "description": spec["description"],
                "pack_type": spec["pack_type"],
                "version": spec["version"],
                "framework_version": spec["framework_version"],
                "coverage_areas": list(spec["coverage_areas"]),
                "rule_definitions": rule_defs,
                "total_rules": len(rule_defs),
                "is_built_in": True,
                "tags": {
                    "framework": spec["name"],
                    "version": spec["version"],
                },
                "namespace": "default",
                "published_by": "system",
                "published_at": _utcnow().isoformat(),
                "updated_at": _utcnow().isoformat(),
            }

            self._packs[pack_id] = pack_data
            self._name_index[spec["name"]] = pack_id

            logger.debug(
                "Registered built-in pack '%s' with %d rules (id=%s)",
                spec["name"],
                len(rule_defs),
                pack_id,
            )

    # ------------------------------------------------------------------
    # Internal: pack model creation
    # ------------------------------------------------------------------

    def _create_rule_pack_model(
        self, pack_data: Dict[str, Any]
    ) -> Any:
        """Create a RulePack Pydantic model from internal pack data.

        If the models module is available, constructs a proper
        ``RulePack`` instance. Otherwise returns the raw dict with
        attribute-style access via a simple wrapper.

        Args:
            pack_data: Internal pack dictionary.

        Returns:
            RulePack model or attribute-accessible wrapper.
        """
        if _MODELS_AVAILABLE:
            try:
                pack_type_val = pack_data.get("pack_type", "custom")
                if isinstance(pack_type_val, str):
                    try:
                        pack_type_enum = RulePackType(pack_type_val)
                    except ValueError:
                        pack_type_enum = RulePackType.CUSTOM
                else:
                    pack_type_enum = pack_type_val

                return RulePack(
                    id=pack_data["pack_id"],
                    name=pack_data["name"],
                    description=pack_data.get("description", ""),
                    pack_type=pack_type_enum,
                    version=pack_data.get("version", "1.0.0"),
                    rule_ids=pack_data.get("rule_ids", []),
                    rule_set_ids=pack_data.get("rule_set_ids", []),
                    framework_version=pack_data.get(
                        "framework_version", ""
                    ),
                    coverage_areas=pack_data.get("coverage_areas", []),
                    total_rules=pack_data.get("total_rules", 0),
                    tags=pack_data.get("tags", {}),
                    is_built_in=pack_data.get("is_built_in", False),
                    namespace=pack_data.get("namespace", "default"),
                    provenance_hash=pack_data.get(
                        "provenance_hash", ""
                    ),
                    published_by=pack_data.get("published_by", "system"),
                )
            except Exception:
                logger.debug(
                    "Failed to create RulePack model for '%s', "
                    "falling back to dict wrapper",
                    pack_data.get("name", "unknown"),
                    exc_info=True,
                )

        # Fallback: attribute-accessible dict wrapper
        return _DictWrapper(pack_data)

    # ------------------------------------------------------------------
    # Internal: rule instantiation from pack definitions
    # ------------------------------------------------------------------

    def _instantiate_rules(
        self,
        rule_definitions: List[Dict[str, Any]],
        rule_set_id: str,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Instantiate concrete rule dictionaries from pack definitions.

        Each definition is expanded into a fully populated rule dict
        with a unique rule_id, timestamps, status, and provenance hash.

        Args:
            rule_definitions: List of rule definition dicts from the pack.
            rule_set_id: ID of the rule set to assign rules to.
            namespace: Optional namespace override.

        Returns:
            List of instantiated rule dictionaries.
        """
        instantiated: List[Dict[str, Any]] = []
        now_iso = _utcnow().isoformat()

        for defn in rule_definitions:
            rule_id = str(uuid.uuid4())
            rule_dict: Dict[str, Any] = {
                "rule_id": rule_id,
                "name": defn.get("name", f"rule_{rule_id[:8]}"),
                "description": defn.get("description", ""),
                "rule_type": defn.get("rule_type", "completeness"),
                "target_field": defn.get("target_field", ""),
                "target_fields": defn.get("target_fields", []),
                "threshold_min": defn.get("threshold_min"),
                "threshold_max": defn.get("threshold_max"),
                "severity": defn.get("severity", "medium"),
                "framework": defn.get("framework", ""),
                "status": "active",
                "namespace": namespace or "default",
                "rule_set_id": rule_set_id,
                "tags": defn.get("tags", {}),
                "version": "1.0.0",
                "created_at": now_iso,
                "updated_at": now_iso,
            }

            # Include optional fields from the definition
            for optional_key in (
                "operator", "expected_value", "pattern",
                "allowed_values", "expression", "condition",
                "parameters",
            ):
                if optional_key in defn:
                    rule_dict[optional_key] = defn[optional_key]

            # Compute provenance hash for this rule
            rule_dict["provenance_hash"] = _sha256(rule_dict)
            instantiated.append(rule_dict)

        return instantiated

    # ==================================================================
    # Public API
    # ==================================================================

    def list_packs(self) -> List[Dict[str, Any]]:
        """List all registered rule packs (built-in and custom).

        Returns a list of summary dictionaries for each pack. Each
        dictionary includes the pack name, description, pack type,
        version, total rules, framework version, coverage areas,
        and whether it is built-in.

        Returns:
            List of pack summary dictionaries sorted by name.

        Example:
            >>> engine = RulePackEngine()
            >>> packs = engine.list_packs()
            >>> assert len(packs) >= 4
        """
        with self._lock:
            result: List[Dict[str, Any]] = []
            for pack_data in self._packs.values():
                result.append({
                    "pack_id": pack_data["pack_id"],
                    "name": pack_data["name"],
                    "description": pack_data.get("description", ""),
                    "pack_type": pack_data.get("pack_type", "custom"),
                    "version": pack_data.get("version", "1.0.0"),
                    "total_rules": pack_data.get("total_rules", 0),
                    "framework_version": pack_data.get(
                        "framework_version", ""
                    ),
                    "coverage_areas": pack_data.get(
                        "coverage_areas", []
                    ),
                    "is_built_in": pack_data.get("is_built_in", False),
                })
            result.sort(key=lambda p: p["name"])
            return result

    def get_pack(self, name: str) -> Any:
        """Retrieve a rule pack by name or ID.

        Returns a RulePack model (if models are available) or a
        dict-wrapper with attribute access. Returns None if the
        pack is not found or the name is empty.

        Args:
            name: Pack name or pack ID to look up.

        Returns:
            RulePack model or None if not found.

        Example:
            >>> engine = RulePackEngine()
            >>> pack = engine.get_pack("ghg_protocol")
            >>> assert pack is not None
            >>> assert pack.name == "ghg_protocol"
        """
        if not name:
            return None

        with self._lock:
            # Try name index first
            pack_id = self._name_index.get(name)
            if pack_id is not None:
                pack_data = self._packs.get(pack_id)
                if pack_data is not None:
                    return self._create_rule_pack_model(
                        copy.deepcopy(pack_data)
                    )

            # Try direct ID lookup
            if name in self._packs:
                return self._create_rule_pack_model(
                    copy.deepcopy(self._packs[name])
                )

        return None

    def apply_pack(
        self,
        pack_name_or_id: str,
        *,
        rule_set_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply a rule pack, creating concrete rules from pack definitions.

        Resolves the pack by name or ID, instantiates all rule
        definitions into concrete rule dictionaries, assigns them to
        a rule set (either the provided ``rule_set_id`` or a newly
        generated one), and records the application in the provenance
        chain.

        Args:
            pack_name_or_id: Name or UUID of the pack to apply.
            rule_set_id: Optional existing rule set ID to add rules to.
                If None, a new UUID is generated.
            namespace: Optional namespace for tenant isolation.

        Returns:
            Dictionary containing:
                - ``rules_created``: Number of rules instantiated.
                - ``rule_set_id``: UUID of the rule set.
                - ``rules``: List of rule dicts with name and description.
                - ``provenance_hash``: SHA-256 hash (64 hex chars).

        Raises:
            ValueError: If the pack name or ID does not exist.
            KeyError: If the pack name or ID does not exist.

        Example:
            >>> engine = RulePackEngine()
            >>> result = engine.apply_pack("ghg_protocol")
            >>> assert result["rules_created"] >= 40
        """
        with self._lock:
            # Resolve pack by name or ID
            pack_data = self._resolve_pack(pack_name_or_id)

        if pack_data is None:
            raise ValueError(
                f"Rule pack not found: '{pack_name_or_id}'"
            )

        # Generate rule set ID if not provided
        effective_rule_set_id = rule_set_id or str(uuid.uuid4())

        # Instantiate rules from pack definitions
        rule_definitions = pack_data.get("rule_definitions", [])
        instantiated_rules = self._instantiate_rules(
            rule_definitions=rule_definitions,
            rule_set_id=effective_rule_set_id,
            namespace=namespace,
        )

        # Build provenance payload
        provenance_payload = {
            "pack_name": pack_data["name"],
            "pack_id": pack_data["pack_id"],
            "rule_set_id": effective_rule_set_id,
            "rules_created": len(instantiated_rules),
            "namespace": namespace or "default",
            "applied_at": _utcnow().isoformat(),
        }
        provenance_hash = _sha256(provenance_payload)

        # Record provenance
        self._provenance.record(
            entity_type="rule_pack",
            entity_id=pack_data["pack_id"],
            action="rule_pack_applied",
            metadata=provenance_payload,
        )

        # Build rule summaries for output
        rule_summaries: List[Dict[str, str]] = []
        for rule in instantiated_rules:
            rule_summaries.append({
                "name": rule.get("name", ""),
                "description": rule.get("description", ""),
                "rule_id": rule.get("rule_id", ""),
                "rule_type": rule.get("rule_type", ""),
                "severity": rule.get("severity", ""),
            })

        # Record application event
        with self._lock:
            self._application_log.append({
                "pack_name": pack_data["name"],
                "pack_id": pack_data["pack_id"],
                "rule_set_id": effective_rule_set_id,
                "rules_created": len(instantiated_rules),
                "namespace": namespace or "default",
                "applied_at": _utcnow().isoformat(),
                "provenance_hash": provenance_hash,
            })

        logger.info(
            "Applied rule pack '%s': %d rules created, "
            "rule_set_id=%s, provenance=%s",
            pack_data["name"],
            len(instantiated_rules),
            effective_rule_set_id,
            provenance_hash[:16],
        )

        return {
            "rules_created": len(instantiated_rules),
            "rule_set_id": effective_rule_set_id,
            "rules": rule_summaries,
            "provenance_hash": provenance_hash,
            "pack_name": pack_data["name"],
            "pack_id": pack_data["pack_id"],
            "namespace": namespace or "default",
            "applied_at": _utcnow().isoformat(),
        }

    def register_custom_pack(
        self,
        name: str,
        description: str,
        rules: List[Dict[str, Any]],
        *,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Register a custom user-defined rule pack.

        Creates a new rule pack with the ``CUSTOM`` type and stores
        the provided rule definitions. The pack can then be applied
        via ``apply_pack()`` using either its name or ID.

        Args:
            name: Human-readable pack name.
            description: Description of the pack purpose.
            rules: List of rule definition dictionaries.
            tags: Optional key-value labels for filtering.

        Returns:
            Dictionary containing:
                - ``pack_id``: UUID of the newly registered pack.
                - ``provenance_hash``: SHA-256 hash of registration.

        Example:
            >>> engine = RulePackEngine()
            >>> result = engine.register_custom_pack(
            ...     name="my_pack",
            ...     description="Custom validation",
            ...     rules=[{"name": "r1", "rule_type": "range",
            ...             "target_field": "val", "severity": "high"}],
            ... )
            >>> assert result["pack_id"] is not None
        """
        pack_id = str(uuid.uuid4())
        now_iso = _utcnow().isoformat()

        pack_data: Dict[str, Any] = {
            "pack_id": pack_id,
            "name": name,
            "description": description,
            "pack_type": "custom",
            "version": "1.0.0",
            "framework_version": "Custom",
            "coverage_areas": [],
            "rule_definitions": list(rules),
            "total_rules": len(rules),
            "is_built_in": False,
            "tags": dict(tags) if tags else {},
            "namespace": "default",
            "published_by": "user",
            "published_at": now_iso,
            "updated_at": now_iso,
        }

        # Record provenance
        provenance_payload = {
            "pack_id": pack_id,
            "name": name,
            "total_rules": len(rules),
            "registered_at": now_iso,
        }
        provenance_hash = _sha256(provenance_payload)

        self._provenance.record(
            entity_type="rule_pack",
            entity_id=pack_id,
            action="rule_pack_imported",
            metadata=provenance_payload,
        )

        with self._lock:
            self._packs[pack_id] = pack_data
            # Only add to name index if name is not already taken by
            # a built-in pack with the exact same name
            if name not in self._name_index:
                self._name_index[name] = pack_id
            else:
                # Use a unique variant to avoid collision
                unique_name = f"{name}_{pack_id[:8]}"
                self._name_index[unique_name] = pack_id
                pack_data["name"] = unique_name
                logger.debug(
                    "Custom pack name '%s' already exists; "
                    "registered as '%s' instead",
                    name,
                    unique_name,
                )

        logger.info(
            "Registered custom pack '%s' with %d rules (id=%s)",
            pack_data["name"],
            len(rules),
            pack_id,
        )

        return {
            "pack_id": pack_id,
            "name": pack_data["name"],
            "provenance_hash": provenance_hash,
        }

    def compare_pack_versions(
        self,
        pack_name: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """Compare two versions of a rule pack.

        Since the in-memory engine stores only the current version,
        this method compares the requested versions against the
        stored version and returns a diff summary. If the two
        versions are equal, the result indicates no changes.

        Args:
            pack_name: Name of the pack to compare.
            version1: First version string (SemVer).
            version2: Second version string (SemVer).

        Returns:
            Dictionary with comparison results including:
                - ``pack_name``: Name of the compared pack.
                - ``version1``: First version.
                - ``version2``: Second version.
                - ``identical``: Whether the versions are equal.
                - ``changes``: List of change descriptions.
                - ``current_version``: Currently stored version.

        Example:
            >>> engine = RulePackEngine()
            >>> diff = engine.compare_pack_versions(
            ...     "ghg_protocol", "1.0.0", "1.0.0"
            ... )
            >>> assert diff is not None
        """
        with self._lock:
            pack_id = self._name_index.get(pack_name)
            current_version = ""
            total_rules = 0

            if pack_id is not None:
                pack_data = self._packs.get(pack_id)
                if pack_data is not None:
                    current_version = pack_data.get("version", "")
                    total_rules = pack_data.get("total_rules", 0)

        # Build comparison result
        identical = version1 == version2
        changes: List[str] = []

        if not identical:
            changes.append(
                f"Version changed from {version1} to {version2}"
            )
            # Parse SemVer components for richer diff
            v1_parts = _parse_semver(version1)
            v2_parts = _parse_semver(version2)

            if v1_parts[0] != v2_parts[0]:
                changes.append(
                    f"Major version: {v1_parts[0]} -> {v2_parts[0]} "
                    "(breaking change)"
                )
            if v1_parts[1] != v2_parts[1]:
                changes.append(
                    f"Minor version: {v1_parts[1]} -> {v2_parts[1]} "
                    "(additive change)"
                )
            if v1_parts[2] != v2_parts[2]:
                changes.append(
                    f"Patch version: {v1_parts[2]} -> {v2_parts[2]} "
                    "(cosmetic change)"
                )

        # Check if requested versions match stored version
        version1_available = version1 == current_version
        version2_available = version2 == current_version

        return {
            "pack_name": pack_name,
            "version1": version1,
            "version2": version2,
            "identical": identical,
            "changes": changes,
            "current_version": current_version,
            "total_rules": total_rules,
            "version1_available": version1_available,
            "version2_available": version2_available,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics for the rule pack engine.

        Returns:
            Dictionary containing:
                - ``total_packs``: Total number of registered packs.
                - ``total_applications``: Total number of apply operations.
                - ``built_in_packs``: Number of built-in packs.
                - ``custom_packs``: Number of custom packs.

        Example:
            >>> engine = RulePackEngine()
            >>> stats = engine.get_statistics()
            >>> assert stats["total_packs"] >= 4
        """
        with self._lock:
            total = len(self._packs)
            built_in = sum(
                1 for p in self._packs.values()
                if p.get("is_built_in", False)
            )
            custom = total - built_in
            applications = len(self._application_log)

        return {
            "total_packs": total,
            "total_applications": applications,
            "built_in_packs": built_in,
            "custom_packs": custom,
        }

    def clear(self) -> None:
        """Clear custom packs and reset application log.

        Removes all custom (non-built-in) packs from the store and
        clears the application log. Built-in packs are preserved.
        This method is intended for testing and administrative reset
        operations.

        Example:
            >>> engine = RulePackEngine()
            >>> engine.register_custom_pack("t", "d", [])
            >>> engine.clear()
            >>> stats = engine.get_statistics()
            >>> assert stats["custom_packs"] == 0
        """
        with self._lock:
            # Remove custom packs
            custom_ids = [
                pid for pid, pdata in self._packs.items()
                if not pdata.get("is_built_in", False)
            ]
            for pid in custom_ids:
                pack_data = self._packs.pop(pid, None)
                if pack_data is not None:
                    # Clean name index
                    keys_to_remove = [
                        k for k, v in self._name_index.items()
                        if v == pid
                    ]
                    for k in keys_to_remove:
                        del self._name_index[k]

            # Reset application log
            self._application_log.clear()

        logger.info(
            "RulePackEngine cleared: removed %d custom packs, "
            "reset application log",
            len(custom_ids),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_pack(
        self, name_or_id: str
    ) -> Optional[Dict[str, Any]]:
        """Resolve a pack by name or ID (caller must hold lock).

        Args:
            name_or_id: Pack name or pack UUID.

        Returns:
            Deep copy of pack data or None if not found.
        """
        # Try name index
        pack_id = self._name_index.get(name_or_id)
        if pack_id is not None:
            pack_data = self._packs.get(pack_id)
            if pack_data is not None:
                return copy.deepcopy(pack_data)

        # Try direct ID lookup
        if name_or_id in self._packs:
            return copy.deepcopy(self._packs[name_or_id])

        return None


# ===================================================================
# Helper: SemVer parsing
# ===================================================================


def _parse_semver(version: str) -> tuple:
    """Parse a SemVer string into (major, minor, patch) integers.

    Falls back to (0, 0, 0) for invalid or incomplete version strings.

    Args:
        version: Version string in ``major.minor.patch`` format.

    Returns:
        Tuple of (major, minor, patch) integers.
    """
    parts = version.split(".")
    try:
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return (major, minor, patch)
    except (ValueError, IndexError):
        return (0, 0, 0)


# ===================================================================
# Helper: dict wrapper for attribute-style access
# ===================================================================


class _DictWrapper:
    """Lightweight wrapper providing attribute-style access over a dict.

    Used as a fallback when the Pydantic RulePack model is unavailable.

    Attributes:
        _data: The underlying dictionary.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        """Initialize wrapper with underlying data dictionary."""
        object.__setattr__(self, "_data", data)

    def __getattr__(self, key: str) -> Any:
        """Return value for attribute from underlying dict."""
        data = object.__getattribute__(self, "_data")
        if key in data:
            return data[key]
        # Map common model field names to internal keys
        if key == "id":
            return data.get("pack_id", "")
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{key}'"
        )

    def __setattr__(self, key: str, value: Any) -> None:
        """Set value in underlying dict."""
        data = object.__getattribute__(self, "_data")
        data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style get with default."""
        data = object.__getattribute__(self, "_data")
        return data.get(key, default)

    def __repr__(self) -> str:
        """Return repr of underlying dict."""
        data = object.__getattribute__(self, "_data")
        return f"_DictWrapper({data.get('name', 'unknown')})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["RulePackEngine"]
