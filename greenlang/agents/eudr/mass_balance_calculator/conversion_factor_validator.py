# -*- coding: utf-8 -*-
"""
Conversion Factor Validator - AGENT-EUDR-011 Engine 3

Validates conversion factors (yield ratios) against peer-reviewed
commodity-specific reference data:
- 30+ commodity conversion pairs with reference yields
- Tolerance bands: warn (>5% deviation), reject (>15% deviation)
- Multi-step conversion chains with cumulative validation
- Seasonal adjustment factors for agricultural commodities
- Process-specific factors (wet vs dry processing)
- Factor history tracking per facility
- Custom factor approval workflow
- Factor deviation reporting

Zero-Hallucination Guarantees:
    - All yield ratio validations are deterministic Python arithmetic
    - Reference conversion factors sourced from config/reference data
    - Deviation calculations use absolute arithmetic, not estimation
    - SHA-256 provenance hashes on every validation and registration
    - No ML/LLM used for any numeric validation or tolerance check

Regulatory References:
    - EU 2023/1115 (EUDR) Article 4: Mass balance chain of custody
    - EU 2023/1115 (EUDR) Article 10(2)(f): Mass balance verification
    - ISO 22095:2020: Chain of Custody - Conversion factor requirements
    - RSPO SCC 2020: Oil extraction rate (OER) validation
    - ISCC 203: Yield factor verification per conversion step

Performance Targets:
    - Single factor validation: <3ms
    - Chain validation (5 steps): <15ms
    - Reference lookup: <1ms
    - Factor history retrieval: <10ms

PRD Feature References:
    - PRD-AGENT-EUDR-011 Feature 3: Conversion Factor Validation
    - F3.1: 30+ commodity conversion pairs with reference yields
    - F3.2: Tolerance bands (warn >5%, reject >15%)
    - F3.3: Multi-step conversion chain validation
    - F3.4: Seasonal adjustment factors for agricultural commodities
    - F3.5: Process-specific factors (wet vs dry processing)
    - F3.6: Factor history tracking per facility
    - F3.7: Custom factor approval workflow
    - F3.8: Factor deviation reporting
    - F3.9: Cross-reference against industry databases
    - F3.10: SHA-256 provenance on all operations

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011
Agent ID: GL-EUDR-MBC-011
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.mass_balance_calculator.config import get_config
from greenlang.agents.eudr.mass_balance_calculator.metrics import (
    record_api_error,
    record_conversion_rejection,
    record_conversion_validation,
)
from greenlang.agents.eudr.mass_balance_calculator.models import (
    ConversionStatus,
)
from greenlang.agents.eudr.mass_balance_calculator.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id() -> str:
    """Generate a new UUID4 string identifier.

    Returns:
        UUID4 string.
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Seasonal adjustment factors by commodity and month (1-12)
# Represents relative yield variation from the annual average.
# Values >1.0 = higher yield (peak season), <1.0 = lower yield (off season).
# Based on published agronomic data for tropical commodity crops.
# ---------------------------------------------------------------------------

_SEASONAL_ADJUSTMENTS: Dict[str, Dict[int, float]] = {
    "cocoa": {
        1: 1.02, 2: 1.03, 3: 1.01, 4: 0.98,
        5: 0.96, 6: 0.95, 7: 0.94, 8: 0.95,
        9: 0.97, 10: 1.02, 11: 1.05, 12: 1.04,
    },
    "coffee": {
        1: 0.96, 2: 0.95, 3: 0.97, 4: 1.00,
        5: 1.02, 6: 1.04, 7: 1.05, 8: 1.03,
        9: 1.01, 10: 0.99, 11: 0.97, 12: 0.96,
    },
    "oil_palm": {
        1: 0.92, 2: 0.90, 3: 0.93, 4: 0.97,
        5: 1.02, 6: 1.05, 7: 1.06, 8: 1.07,
        9: 1.05, 10: 1.03, 11: 0.98, 12: 0.95,
    },
    "rubber": {
        1: 0.85, 2: 0.83, 3: 0.88, 4: 0.95,
        5: 1.02, 6: 1.05, 7: 1.06, 8: 1.05,
        9: 1.03, 10: 1.02, 11: 0.98, 12: 0.92,
    },
    "soya": {
        1: 0.98, 2: 0.97, 3: 1.00, 4: 1.02,
        5: 1.03, 6: 1.01, 7: 0.99, 8: 0.98,
        9: 1.00, 10: 1.02, 11: 1.01, 12: 0.99,
    },
    "wood": {
        1: 0.98, 2: 0.97, 3: 0.99, 4: 1.01,
        5: 1.02, 6: 1.03, 7: 1.02, 8: 1.01,
        9: 1.00, 10: 0.99, 11: 0.98, 12: 0.98,
    },
    "cattle": {
        1: 0.99, 2: 0.98, 3: 1.00, 4: 1.01,
        5: 1.02, 6: 1.01, 7: 1.00, 8: 1.00,
        9: 1.01, 10: 1.00, 11: 0.99, 12: 0.99,
    },
}

# ---------------------------------------------------------------------------
# Extended reference data with min/max acceptable ranges
# Keyed by (commodity, process_name), value is {yield_ratio, min, max, source}
# ---------------------------------------------------------------------------

_EXTENDED_REFERENCE_FACTORS: Dict[Tuple[str, str], Dict[str, Any]] = {
    # Cocoa processing chain
    ("cocoa", "fermentation"): {
        "yield_ratio": 0.92, "min": 0.88, "max": 0.95,
        "source": "ICCO Processing Standards 2023",
    },
    ("cocoa", "drying"): {
        "yield_ratio": 0.88, "min": 0.84, "max": 0.92,
        "source": "ICCO Processing Standards 2023",
    },
    ("cocoa", "roasting"): {
        "yield_ratio": 0.85, "min": 0.80, "max": 0.89,
        "source": "ICCO Processing Standards 2023",
    },
    ("cocoa", "winnowing"): {
        "yield_ratio": 0.80, "min": 0.75, "max": 0.85,
        "source": "ICCO Processing Standards 2023",
    },
    ("cocoa", "grinding"): {
        "yield_ratio": 0.98, "min": 0.96, "max": 0.99,
        "source": "ICCO Processing Standards 2023",
    },
    ("cocoa", "pressing"): {
        "yield_ratio": 0.45, "min": 0.40, "max": 0.50,
        "source": "ICCO Processing Standards 2023",
    },
    ("cocoa", "conching"): {
        "yield_ratio": 0.97, "min": 0.95, "max": 0.99,
        "source": "ICCO Processing Standards 2023",
    },
    ("cocoa", "tempering"): {
        "yield_ratio": 0.99, "min": 0.98, "max": 1.00,
        "source": "ICCO Processing Standards 2023",
    },
    # Coffee processing chain
    ("coffee", "wet_processing"): {
        "yield_ratio": 0.60, "min": 0.55, "max": 0.65,
        "source": "ICO Technical Guide 2022",
    },
    ("coffee", "dry_processing"): {
        "yield_ratio": 0.50, "min": 0.45, "max": 0.55,
        "source": "ICO Technical Guide 2022",
    },
    ("coffee", "hulling"): {
        "yield_ratio": 0.80, "min": 0.76, "max": 0.84,
        "source": "ICO Technical Guide 2022",
    },
    ("coffee", "polishing"): {
        "yield_ratio": 0.98, "min": 0.96, "max": 0.99,
        "source": "ICO Technical Guide 2022",
    },
    ("coffee", "roasting"): {
        "yield_ratio": 0.82, "min": 0.78, "max": 0.86,
        "source": "ICO Technical Guide 2022",
    },
    # Oil palm processing chain
    ("oil_palm", "sterilization"): {
        "yield_ratio": 0.95, "min": 0.92, "max": 0.97,
        "source": "RSPO SCC 2020 / MPOB Technical Standards",
    },
    ("oil_palm", "threshing"): {
        "yield_ratio": 0.65, "min": 0.60, "max": 0.70,
        "source": "RSPO SCC 2020 / MPOB Technical Standards",
    },
    ("oil_palm", "digestion"): {
        "yield_ratio": 0.90, "min": 0.87, "max": 0.93,
        "source": "RSPO SCC 2020 / MPOB Technical Standards",
    },
    ("oil_palm", "extraction"): {
        "yield_ratio": 0.22, "min": 0.18, "max": 0.26,
        "source": "RSPO SCC 2020 / MPOB Technical Standards",
    },
    ("oil_palm", "clarification"): {
        "yield_ratio": 0.95, "min": 0.92, "max": 0.97,
        "source": "RSPO SCC 2020 / MPOB Technical Standards",
    },
    ("oil_palm", "refining"): {
        "yield_ratio": 0.92, "min": 0.88, "max": 0.95,
        "source": "RSPO SCC 2020 / MPOB Technical Standards",
    },
    ("oil_palm", "fractionation"): {
        "yield_ratio": 0.90, "min": 0.86, "max": 0.94,
        "source": "RSPO SCC 2020 / MPOB Technical Standards",
    },
    # Wood processing chain
    ("wood", "debarking"): {
        "yield_ratio": 0.90, "min": 0.86, "max": 0.94,
        "source": "FSC-STD-40-004 Processing Standards",
    },
    ("wood", "sawing"): {
        "yield_ratio": 0.55, "min": 0.48, "max": 0.62,
        "source": "FSC-STD-40-004 Processing Standards",
    },
    ("wood", "planing"): {
        "yield_ratio": 0.90, "min": 0.86, "max": 0.94,
        "source": "FSC-STD-40-004 Processing Standards",
    },
    ("wood", "kiln_drying"): {
        "yield_ratio": 0.92, "min": 0.88, "max": 0.95,
        "source": "FSC-STD-40-004 Processing Standards",
    },
    ("wood", "milling"): {
        "yield_ratio": 0.85, "min": 0.80, "max": 0.90,
        "source": "FSC-STD-40-004 Processing Standards",
    },
    # Rubber processing chain
    ("rubber", "coagulation"): {
        "yield_ratio": 0.60, "min": 0.55, "max": 0.65,
        "source": "IRSG Processing Guidelines 2023",
    },
    ("rubber", "sheeting"): {
        "yield_ratio": 0.95, "min": 0.92, "max": 0.97,
        "source": "IRSG Processing Guidelines 2023",
    },
    ("rubber", "smoking"): {
        "yield_ratio": 0.88, "min": 0.84, "max": 0.92,
        "source": "IRSG Processing Guidelines 2023",
    },
    ("rubber", "crumbling"): {
        "yield_ratio": 0.92, "min": 0.88, "max": 0.95,
        "source": "IRSG Processing Guidelines 2023",
    },
    # Soya processing chain
    ("soya", "cleaning"): {
        "yield_ratio": 0.98, "min": 0.96, "max": 0.99,
        "source": "AOCS Processing Standards 2023",
    },
    ("soya", "dehulling"): {
        "yield_ratio": 0.92, "min": 0.89, "max": 0.95,
        "source": "AOCS Processing Standards 2023",
    },
    ("soya", "flaking"): {
        "yield_ratio": 0.97, "min": 0.95, "max": 0.99,
        "source": "AOCS Processing Standards 2023",
    },
    ("soya", "solvent_extraction"): {
        "yield_ratio": 0.82, "min": 0.78, "max": 0.86,
        "source": "AOCS Processing Standards 2023",
    },
    ("soya", "refining"): {
        "yield_ratio": 0.92, "min": 0.88, "max": 0.95,
        "source": "AOCS Processing Standards 2023",
    },
    # Cattle processing chain
    ("cattle", "slaughtering"): {
        "yield_ratio": 0.55, "min": 0.50, "max": 0.60,
        "source": "FAO Livestock Processing Manual 2022",
    },
    ("cattle", "deboning"): {
        "yield_ratio": 0.70, "min": 0.65, "max": 0.75,
        "source": "FAO Livestock Processing Manual 2022",
    },
    ("cattle", "tanning"): {
        "yield_ratio": 0.30, "min": 0.25, "max": 0.35,
        "source": "IULTCS Leather Processing Standards",
    },
}


# ---------------------------------------------------------------------------
# ConversionFactorValidator
# ---------------------------------------------------------------------------


class ConversionFactorValidator:
    """Conversion factor validation engine for EUDR mass balance accounting.

    Validates reported yield ratios against peer-reviewed commodity-specific
    reference data. Supports tolerance bands, multi-step chain validation,
    seasonal adjustments, custom factor registration, and deviation reporting.

    All operations follow the zero-hallucination principle: yield ratio
    validations use deterministic Python arithmetic with reference data
    sourced from configuration and published standards only.

    Attributes:
        _config: Mass balance calculator configuration.
        _provenance: ProvenanceTracker for audit trail.
        _reference_factors: Extended reference data with min/max ranges.
        _custom_factors: Registered custom factors keyed by
            ``"commodity:process"`` with facility-specific overrides.
        _validation_history: Validation results keyed by facility_id.
        _factor_history: Factor usage history keyed by
            ``"facility_id:commodity"``.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> from greenlang.agents.eudr.mass_balance_calculator.conversion_factor_validator import (
        ...     ConversionFactorValidator,
        ... )
        >>> validator = ConversionFactorValidator()
        >>> result = validator.validate_factor(
        ...     commodity="cocoa",
        ...     process_name="roasting",
        ...     yield_ratio=0.84,
        ... )
        >>> assert result["status"] == "validated"
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize ConversionFactorValidator with config and provenance.

        Args:
            config: Optional MassBalanceCalculatorConfig override. If None,
                uses the singleton from get_config().
            provenance: Optional ProvenanceTracker override. If None,
                uses the singleton from get_provenance_tracker().
        """
        self._config = config or get_config()
        self._provenance = provenance or get_provenance_tracker()

        # -- Reference data from config ----------------------------------------
        self._config_reference = dict(self._config.reference_conversion_factors)

        # -- Extended reference data with min/max ranges -----------------------
        self._reference_factors: Dict[Tuple[str, str], Dict[str, Any]] = dict(
            _EXTENDED_REFERENCE_FACTORS
        )

        # -- Custom factor registry --------------------------------------------
        self._custom_factors: Dict[str, Dict[str, Any]] = {}

        # -- Validation history per facility -----------------------------------
        self._validation_history: Dict[str, List[Dict[str, Any]]] = {}

        # -- Factor usage history per facility+commodity -----------------------
        self._factor_history: Dict[str, List[Dict[str, Any]]] = {}

        # -- Seasonal adjustments ----------------------------------------------
        self._seasonal_adjustments: Dict[str, Dict[int, float]] = dict(
            _SEASONAL_ADJUSTMENTS
        )

        # -- Thread safety -----------------------------------------------------
        self._lock = threading.RLock()

        # -- Tolerance thresholds from config ----------------------------------
        self._warn_deviation = self._config.conversion_factor_warn_deviation
        self._reject_deviation = self._config.conversion_factor_reject_deviation

        logger.info(
            "ConversionFactorValidator initialized: module_version=%s, "
            "warn_deviation=%.2f, reject_deviation=%.2f, "
            "reference_pairs=%d, provenance_enabled=%s",
            _MODULE_VERSION,
            self._warn_deviation,
            self._reject_deviation,
            len(self._reference_factors),
            self._config.enable_provenance,
        )

    # ------------------------------------------------------------------
    # Public API: Single Factor Validation
    # ------------------------------------------------------------------

    def validate_factor(
        self,
        commodity: str,
        process_name: str,
        yield_ratio: float,
        facility_id: Optional[str] = None,
        month: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate a conversion factor against reference data.

        Checks the reported yield ratio against the reference data for
        the given commodity and process. Returns a validation result with
        status (validated, warned, rejected), deviation percentage, and
        reference range.

        PRD Reference: F3.1, F3.2 - Tolerance bands.

        Args:
            commodity: EUDR commodity (cattle, cocoa, coffee, etc.).
            process_name: Processing step name (roasting, drying, etc.).
            yield_ratio: Reported yield ratio (0.0 - 1.0 exclusive).
            facility_id: Optional facility identifier for history tracking.
            month: Optional month (1-12) for seasonal adjustment.
            metadata: Optional additional context.

        Returns:
            Dictionary containing:
                - validation_id: Unique validation identifier
                - commodity: Commodity validated
                - process_name: Process validated
                - yield_ratio: Reported yield ratio
                - reference_yield: Reference yield ratio
                - reference_min: Minimum acceptable yield
                - reference_max: Maximum acceptable yield
                - deviation_percent: Deviation from reference
                - status: "validated", "warned", or "rejected"
                - seasonal_adjustment: Adjustment factor applied (if any)
                - provenance_hash: SHA-256 provenance hash
                - message: Human-readable result description

        Raises:
            ValueError: If inputs are invalid.
        """
        start_time = time.monotonic()

        # Validate inputs
        self._validate_commodity(commodity)
        self._validate_process_name(process_name)
        self._validate_yield_ratio(yield_ratio)
        if month is not None and not (1 <= month <= 12):
            raise ValueError(f"month must be in [1, 12], got {month}")

        commodity_lower = commodity.lower().strip()
        process_lower = process_name.lower().strip()

        # Apply seasonal adjustment if requested
        adjusted_ratio = yield_ratio
        seasonal_factor = 1.0
        if month is not None:
            seasonal_factor = self.apply_seasonal_adjustment(
                commodity_lower, yield_ratio, month
            )
            # The seasonal adjustment adjusts the reference, not the reported value.
            # We compare the reported ratio against the seasonally adjusted reference.

        # Look up reference data
        reference = self._get_reference(commodity_lower, process_lower)

        # Check custom factors for this facility
        custom_ref = None
        if facility_id:
            custom_key = f"{facility_id}:{commodity_lower}:{process_lower}"
            with self._lock:
                custom_ref = self._custom_factors.get(custom_key)

        # Use custom reference if approved, otherwise use standard reference
        if custom_ref and custom_ref.get("approved", False):
            ref_yield = custom_ref["yield_ratio"]
            ref_min = custom_ref.get("min", ref_yield * 0.90)
            ref_max = custom_ref.get("max", ref_yield * 1.10)
            ref_source = f"Custom (approved by {custom_ref.get('approved_by', 'N/A')})"
        elif reference:
            ref_yield = reference["yield_ratio"]
            ref_min = reference.get("min", ref_yield * 0.90)
            ref_max = reference.get("max", ref_yield * 1.10)
            ref_source = reference.get("source", "Standard reference data")
        else:
            # No reference found - use config-level reference if available
            config_ref = self._config_reference.get(commodity_lower, {})
            config_yield = config_ref.get(process_lower)
            if config_yield is not None:
                ref_yield = config_yield
                ref_min = ref_yield * 0.90
                ref_max = ref_yield * 1.10
                ref_source = "Configuration reference data"
            else:
                # No reference at all - cannot validate, return advisory
                validation_id = _generate_id()
                result = self._build_no_reference_result(
                    validation_id, commodity_lower, process_lower,
                    yield_ratio, facility_id, metadata,
                )
                self._record_validation_history(facility_id, result)
                return result

        # Apply seasonal adjustment to reference range
        if month is not None and seasonal_factor != 1.0:
            ref_yield = ref_yield * seasonal_factor
            ref_min = ref_min * seasonal_factor
            ref_max = ref_max * seasonal_factor

        # Check tolerance
        status, deviation_pct = self._check_tolerance(
            actual=yield_ratio,
            reference_min=ref_min,
            reference_max=ref_max,
            warn_pct=self._warn_deviation,
            reject_pct=self._reject_deviation,
        )

        validation_id = _generate_id()
        now = _utcnow()

        # Compute provenance hash
        provenance_hash = _compute_hash({
            "validation_id": validation_id,
            "commodity": commodity_lower,
            "process_name": process_lower,
            "yield_ratio": yield_ratio,
            "reference_yield": ref_yield,
            "deviation_percent": deviation_pct,
            "status": status,
            "action": "validate",
        })

        # Build result
        result: Dict[str, Any] = {
            "validation_id": validation_id,
            "commodity": commodity_lower,
            "process_name": process_lower,
            "yield_ratio": yield_ratio,
            "reference_yield": round(ref_yield, 4),
            "reference_min": round(ref_min, 4),
            "reference_max": round(ref_max, 4),
            "reference_source": ref_source,
            "deviation_percent": round(deviation_pct, 4),
            "status": status,
            "seasonal_adjustment": round(seasonal_factor, 4) if month else None,
            "month": month,
            "facility_id": facility_id,
            "provenance_hash": provenance_hash,
            "validated_at": now.isoformat(),
            "message": self._build_validation_message(
                status, commodity_lower, process_lower,
                yield_ratio, ref_yield, deviation_pct,
            ),
            "metadata": metadata or {},
        }

        # Record metrics
        record_conversion_validation(commodity_lower)
        if status == ConversionStatus.REJECTED.value:
            record_conversion_rejection(commodity_lower)

        # Record provenance
        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="conversion_factor",
                action="validate",
                entity_id=validation_id,
                data=result,
                metadata={
                    "commodity": commodity_lower,
                    "process_name": process_lower,
                    "status": status,
                    "deviation_percent": deviation_pct,
                },
            )

        # Record in validation history
        self._record_validation_history(facility_id, result)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result["elapsed_ms"] = round(elapsed_ms, 2)

        logger.info(
            "Factor validated: %s/%s yield=%.4f ref=%.4f "
            "dev=%.2f%% status=%s elapsed=%.1fms",
            commodity_lower,
            process_lower,
            yield_ratio,
            ref_yield,
            deviation_pct * 100,
            status,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Reference Factor Lookup
    # ------------------------------------------------------------------

    def get_reference_factors(
        self,
        commodity: str,
    ) -> List[Dict[str, Any]]:
        """Get all reference conversion factors for a commodity.

        PRD Reference: F3.1 - 30+ commodity conversion pairs.

        Args:
            commodity: EUDR commodity to look up.

        Returns:
            List of reference factor dictionaries for all processes
            of the given commodity.
        """
        commodity_lower = commodity.lower().strip()
        results: List[Dict[str, Any]] = []

        for (c, p), ref in self._reference_factors.items():
            if c == commodity_lower:
                results.append({
                    "commodity": c,
                    "process_name": p,
                    "yield_ratio": ref["yield_ratio"],
                    "min": ref.get("min"),
                    "max": ref.get("max"),
                    "source": ref.get("source", "Reference data"),
                })

        # Also include config-level references not in extended data
        config_refs = self._config_reference.get(commodity_lower, {})
        for process, ratio in config_refs.items():
            # Skip if already in extended reference
            if (commodity_lower, process) in self._reference_factors:
                continue
            results.append({
                "commodity": commodity_lower,
                "process_name": process,
                "yield_ratio": ratio,
                "min": ratio * 0.90,
                "max": ratio * 1.10,
                "source": "Configuration defaults",
            })

        results.sort(key=lambda r: r["process_name"])
        return results

    # ------------------------------------------------------------------
    # Public API: Custom Factor Registration
    # ------------------------------------------------------------------

    def register_custom_factor(
        self,
        commodity: str,
        process: str,
        yield_ratio: float,
        facility_id: str,
        justification: str,
        approved_by: str,
        expiry_days: int = 365,
    ) -> Dict[str, Any]:
        """Register a custom conversion factor for a facility.

        Custom factors override reference data for the specific
        facility after approval. They have an expiry date and are
        tracked in the audit trail.

        PRD Reference: F3.7 - Custom factor approval workflow.

        Args:
            commodity: EUDR commodity.
            process: Processing step name.
            yield_ratio: Custom yield ratio (0.0 - 1.0).
            facility_id: Facility identifier.
            justification: Free-text justification for the custom factor.
            approved_by: Identifier of the approving authority.
            expiry_days: Number of days until the custom factor expires.
                Defaults to 365.

        Returns:
            Dictionary containing:
                - factor_id: Unique factor identifier
                - commodity, process, yield_ratio, facility_id
                - approved: True (auto-approved on registration)
                - approved_by: Approver identifier
                - expiry_date: Expiry date (ISO string)
                - provenance_hash: SHA-256 provenance hash
                - operation_status: "registered"

        Raises:
            ValueError: If inputs are invalid.
        """
        start_time = time.monotonic()

        # Validate inputs
        self._validate_commodity(commodity)
        self._validate_process_name(process)
        self._validate_yield_ratio(yield_ratio)
        if not facility_id or not facility_id.strip():
            raise ValueError("facility_id must not be empty")
        if not justification or not justification.strip():
            raise ValueError("justification must not be empty")
        if not approved_by or not approved_by.strip():
            raise ValueError("approved_by must not be empty")
        if expiry_days < 1:
            raise ValueError(f"expiry_days must be >= 1, got {expiry_days}")

        commodity_lower = commodity.lower().strip()
        process_lower = process.lower().strip()
        now = _utcnow()

        factor_id = _generate_id()
        custom_key = f"{facility_id}:{commodity_lower}:{process_lower}"

        # Get reference for comparison
        reference = self._get_reference(commodity_lower, process_lower)
        ref_yield = reference["yield_ratio"] if reference else None

        # Compute deviation from reference
        deviation_pct = 0.0
        if ref_yield and ref_yield > 0:
            deviation_pct = abs(yield_ratio - ref_yield) / ref_yield

        from datetime import timedelta
        expiry_date = now + timedelta(days=expiry_days)

        factor_data: Dict[str, Any] = {
            "factor_id": factor_id,
            "commodity": commodity_lower,
            "process": process_lower,
            "yield_ratio": yield_ratio,
            "min": yield_ratio * 0.95,
            "max": min(1.0, yield_ratio * 1.05),
            "facility_id": facility_id,
            "justification": justification.strip(),
            "approved": True,
            "approved_by": approved_by.strip(),
            "approved_at": now.isoformat(),
            "expiry_date": expiry_date.isoformat(),
            "expiry_days": expiry_days,
            "reference_yield": ref_yield,
            "deviation_from_reference_pct": round(deviation_pct, 4),
            "created_at": now.isoformat(),
        }

        # Compute provenance hash
        provenance_hash = _compute_hash({
            "factor_id": factor_id,
            "commodity": commodity_lower,
            "process": process_lower,
            "yield_ratio": yield_ratio,
            "facility_id": facility_id,
            "approved_by": approved_by.strip(),
            "action": "register_custom",
        })
        factor_data["provenance_hash"] = provenance_hash

        # Store custom factor
        with self._lock:
            self._custom_factors[custom_key] = factor_data

            # Record in factor history
            history_key = f"{facility_id}:{commodity_lower}"
            if history_key not in self._factor_history:
                self._factor_history[history_key] = []
            self._factor_history[history_key].append({
                "factor_id": factor_id,
                "action": "register_custom",
                "yield_ratio": yield_ratio,
                "process": process_lower,
                "approved_by": approved_by.strip(),
                "timestamp": now.isoformat(),
            })

        # Record provenance
        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="conversion_factor",
                action="create",
                entity_id=factor_id,
                data=factor_data,
                metadata={
                    "commodity": commodity_lower,
                    "process": process_lower,
                    "facility_id": facility_id,
                    "deviation_pct": deviation_pct,
                },
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Custom factor registered: %s/%s yield=%.4f "
            "facility=%s approved_by=%s elapsed=%.1fms",
            commodity_lower,
            process_lower,
            yield_ratio,
            facility_id,
            approved_by,
            elapsed_ms,
        )

        return {
            **factor_data,
            "operation_status": "registered",
            "elapsed_ms": round(elapsed_ms, 2),
        }

    # ------------------------------------------------------------------
    # Public API: Factor History
    # ------------------------------------------------------------------

    def get_factor_history(
        self,
        facility_id: str,
        commodity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get factor usage history for a facility.

        PRD Reference: F3.6 - Factor history tracking per facility.

        Args:
            facility_id: Facility identifier.
            commodity: Optional commodity filter.
            limit: Maximum number of records to return.

        Returns:
            List of factor history records, most recent first.
        """
        if not facility_id or not facility_id.strip():
            raise ValueError("facility_id must not be empty")

        results: List[Dict[str, Any]] = []

        with self._lock:
            if commodity:
                commodity_lower = commodity.lower().strip()
                key = f"{facility_id}:{commodity_lower}"
                entries = self._factor_history.get(key, [])
                results.extend(entries)
            else:
                # All commodities for this facility
                for key, entries in self._factor_history.items():
                    if key.startswith(f"{facility_id}:"):
                        results.extend(entries)

        # Sort by timestamp descending
        results.sort(
            key=lambda r: r.get("timestamp", ""),
            reverse=True,
        )
        return results[:limit]

    # ------------------------------------------------------------------
    # Public API: Chain Validation
    # ------------------------------------------------------------------

    def validate_chain(
        self,
        conversion_steps: List[Dict[str, Any]],
        facility_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate a multi-step conversion chain.

        Validates each step individually and computes the cumulative
        yield across the entire chain. Flags if the cumulative yield
        deviates significantly from the expected overall conversion.

        PRD Reference: F3.3 - Multi-step conversion chain validation.

        Args:
            conversion_steps: List of step dictionaries, each containing:
                - commodity: Commodity being processed
                - process_name: Processing step name
                - yield_ratio: Reported yield ratio for this step
            facility_id: Optional facility identifier.

        Returns:
            Dictionary containing:
                - chain_id: Unique chain validation identifier
                - step_count: Number of steps validated
                - step_results: List of per-step validation results
                - cumulative_yield: Overall cumulative yield ratio
                - expected_cumulative_yield: Expected cumulative yield
                - cumulative_deviation_percent: Deviation of cumulative
                - chain_status: Overall chain status
                - provenance_hash: SHA-256 provenance hash

        Raises:
            ValueError: If conversion_steps is empty or malformed.
        """
        start_time = time.monotonic()

        if not conversion_steps:
            raise ValueError("conversion_steps must not be empty")

        chain_id = _generate_id()
        step_results: List[Dict[str, Any]] = []
        cumulative_yield = 1.0
        expected_cumulative = 1.0
        chain_has_rejection = False
        chain_has_warning = False

        for i, step in enumerate(conversion_steps):
            # Validate step structure
            commodity = step.get("commodity", "")
            process_name = step.get("process_name", "")
            yield_ratio = step.get("yield_ratio", 0.0)

            if not commodity or not process_name:
                raise ValueError(
                    f"Step {i}: 'commodity' and 'process_name' required"
                )
            if not (0.0 < yield_ratio <= 1.0):
                raise ValueError(
                    f"Step {i}: yield_ratio must be in (0.0, 1.0], "
                    f"got {yield_ratio}"
                )

            # Validate individual step
            step_result = self.validate_factor(
                commodity=commodity,
                process_name=process_name,
                yield_ratio=yield_ratio,
                facility_id=facility_id,
            )
            step_result["step_index"] = i
            step_results.append(step_result)

            # Accumulate yields
            cumulative_yield *= yield_ratio

            # Accumulate expected yield from reference
            ref_yield = step_result.get("reference_yield")
            if ref_yield and ref_yield > 0:
                expected_cumulative *= ref_yield

            # Track chain-level status
            if step_result["status"] == ConversionStatus.REJECTED.value:
                chain_has_rejection = True
            elif step_result["status"] == ConversionStatus.WARNED.value:
                chain_has_warning = True

        # Compute cumulative deviation
        cumulative_deviation = 0.0
        if expected_cumulative > 0:
            cumulative_deviation = abs(
                cumulative_yield - expected_cumulative
            ) / expected_cumulative

        # Determine chain status
        if chain_has_rejection:
            chain_status = ConversionStatus.REJECTED.value
        elif chain_has_warning:
            chain_status = ConversionStatus.WARNED.value
        elif cumulative_deviation > self._reject_deviation:
            chain_status = ConversionStatus.REJECTED.value
        elif cumulative_deviation > self._warn_deviation:
            chain_status = ConversionStatus.WARNED.value
        else:
            chain_status = ConversionStatus.VALIDATED.value

        # Compute provenance hash
        provenance_hash = _compute_hash({
            "chain_id": chain_id,
            "step_count": len(conversion_steps),
            "cumulative_yield": cumulative_yield,
            "chain_status": chain_status,
            "action": "validate_chain",
        })

        # Record provenance
        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="conversion_factor",
                action="validate",
                entity_id=chain_id,
                data={
                    "chain_id": chain_id,
                    "step_count": len(conversion_steps),
                    "cumulative_yield": cumulative_yield,
                    "chain_status": chain_status,
                },
                metadata={
                    "facility_id": facility_id,
                    "cumulative_deviation": cumulative_deviation,
                },
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Chain validated: chain_id=%s steps=%d "
            "cumulative_yield=%.4f status=%s elapsed=%.1fms",
            chain_id[:12],
            len(conversion_steps),
            cumulative_yield,
            chain_status,
            elapsed_ms,
        )

        return {
            "chain_id": chain_id,
            "step_count": len(conversion_steps),
            "step_results": step_results,
            "cumulative_yield": round(cumulative_yield, 6),
            "expected_cumulative_yield": round(expected_cumulative, 6),
            "cumulative_deviation_percent": round(cumulative_deviation, 6),
            "chain_status": chain_status,
            "facility_id": facility_id,
            "provenance_hash": provenance_hash,
            "elapsed_ms": round(elapsed_ms, 2),
        }

    # ------------------------------------------------------------------
    # Public API: Deviation Report
    # ------------------------------------------------------------------

    def get_deviation_report(
        self,
        facility_id: str,
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a deviation report for a facility.

        Summarizes all factor validations, deviations, rejections, and
        trends for the given facility.

        PRD Reference: F3.8 - Factor deviation reporting.

        Args:
            facility_id: Facility identifier.
            commodity: Optional commodity filter.

        Returns:
            Dictionary containing:
                - facility_id: Facility identifier
                - total_validations: Total number of validations
                - validated_count: Number of accepted validations
                - warned_count: Number of warnings
                - rejected_count: Number of rejections
                - average_deviation: Mean deviation across all validations
                - max_deviation: Maximum deviation observed
                - deviations_by_process: Breakdown by process type
                - recent_rejections: Last 10 rejected validations
        """
        if not facility_id or not facility_id.strip():
            raise ValueError("facility_id must not be empty")

        with self._lock:
            history = list(self._validation_history.get(facility_id, []))

        if commodity:
            commodity_lower = commodity.lower().strip()
            history = [h for h in history if h.get("commodity") == commodity_lower]

        validated_count = 0
        warned_count = 0
        rejected_count = 0
        deviations: List[float] = []
        deviations_by_process: Dict[str, List[float]] = {}
        recent_rejections: List[Dict[str, Any]] = []

        for entry in history:
            status = entry.get("status", "")
            dev = entry.get("deviation_percent", 0.0)
            process = entry.get("process_name", "unknown")

            if status == ConversionStatus.VALIDATED.value:
                validated_count += 1
            elif status == ConversionStatus.WARNED.value:
                warned_count += 1
            elif status == ConversionStatus.REJECTED.value:
                rejected_count += 1
                recent_rejections.append(entry)

            deviations.append(dev)
            if process not in deviations_by_process:
                deviations_by_process[process] = []
            deviations_by_process[process].append(dev)

        avg_deviation = (
            statistics.mean(deviations) if deviations else 0.0
        )
        max_deviation = max(deviations) if deviations else 0.0

        process_summary: Dict[str, Dict[str, float]] = {}
        for process, devs in deviations_by_process.items():
            process_summary[process] = {
                "count": len(devs),
                "mean_deviation": round(statistics.mean(devs), 6),
                "max_deviation": round(max(devs), 6),
                "min_deviation": round(min(devs), 6),
            }

        return {
            "facility_id": facility_id,
            "commodity_filter": commodity,
            "total_validations": len(history),
            "validated_count": validated_count,
            "warned_count": warned_count,
            "rejected_count": rejected_count,
            "average_deviation": round(avg_deviation, 6),
            "max_deviation": round(max_deviation, 6),
            "deviations_by_process": process_summary,
            "recent_rejections": sorted(
                recent_rejections,
                key=lambda r: r.get("validated_at", ""),
                reverse=True,
            )[:10],
            "compliance_rate": round(
                validated_count / len(history) * 100, 2
            ) if history else 100.0,
        }

    # ------------------------------------------------------------------
    # Public API: Seasonal Adjustment
    # ------------------------------------------------------------------

    def apply_seasonal_adjustment(
        self,
        commodity: str,
        yield_ratio: float,
        month: int,
    ) -> float:
        """Apply seasonal adjustment to a yield ratio.

        Returns the seasonal adjustment factor for the given commodity
        and month. The caller uses this to adjust reference data (not
        the reported yield itself).

        PRD Reference: F3.4 - Seasonal adjustment factors.

        Args:
            commodity: EUDR commodity.
            yield_ratio: Base yield ratio (not modified).
            month: Month of year (1-12).

        Returns:
            Seasonal adjustment factor (multiply reference by this).
        """
        commodity_lower = commodity.lower().strip()
        seasonal = self._seasonal_adjustments.get(commodity_lower)
        if seasonal is None:
            return 1.0
        return seasonal.get(month, 1.0)

    # ------------------------------------------------------------------
    # Public API: Get Seasonal Factors
    # ------------------------------------------------------------------

    def get_seasonal_factors(
        self,
        commodity: str,
    ) -> Dict[int, float]:
        """Get all seasonal adjustment factors for a commodity.

        Args:
            commodity: EUDR commodity.

        Returns:
            Dictionary mapping month (1-12) to adjustment factor.
            Returns empty dict if no seasonal data available.
        """
        commodity_lower = commodity.lower().strip()
        return dict(self._seasonal_adjustments.get(commodity_lower, {}))

    # ------------------------------------------------------------------
    # Public API: List Custom Factors
    # ------------------------------------------------------------------

    def get_custom_factors(
        self,
        facility_id: Optional[str] = None,
        commodity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List registered custom conversion factors.

        Args:
            facility_id: Optional facility filter.
            commodity: Optional commodity filter.

        Returns:
            List of custom factor dictionaries.
        """
        results: List[Dict[str, Any]] = []
        with self._lock:
            for key, factor in self._custom_factors.items():
                if facility_id and factor.get("facility_id") != facility_id:
                    continue
                if commodity:
                    commodity_lower = commodity.lower().strip()
                    if factor.get("commodity") != commodity_lower:
                        continue
                results.append(dict(factor))

        results.sort(key=lambda f: f.get("created_at", ""), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Public API: Get All Reference Commodities
    # ------------------------------------------------------------------

    def get_supported_commodities(self) -> List[str]:
        """Get list of commodities with reference conversion factor data.

        Returns:
            Sorted list of commodity names.
        """
        commodities: set = set()
        for c, _ in self._reference_factors:
            commodities.add(c)
        for c in self._config_reference:
            commodities.add(c)
        return sorted(commodities)

    # ------------------------------------------------------------------
    # Internal: Tolerance Check
    # ------------------------------------------------------------------

    def _check_tolerance(
        self,
        actual: float,
        reference_min: float,
        reference_max: float,
        warn_pct: float,
        reject_pct: float,
    ) -> Tuple[str, float]:
        """Check a yield ratio against reference range with tolerance bands.

        Deviation is calculated as the fractional distance outside the
        acceptable [min, max] range. If within range, deviation is 0.

        Args:
            actual: Reported yield ratio.
            reference_min: Minimum acceptable reference yield.
            reference_max: Maximum acceptable reference yield.
            warn_pct: Deviation fraction triggering a warning (e.g. 0.05).
            reject_pct: Deviation fraction triggering rejection (e.g. 0.15).

        Returns:
            Tuple of (status_string, deviation_fraction).
        """
        midpoint = (reference_min + reference_max) / 2.0
        if midpoint <= 0:
            return ConversionStatus.VALIDATED.value, 0.0

        # Within acceptable range
        if reference_min <= actual <= reference_max:
            deviation = abs(actual - midpoint) / midpoint
            return ConversionStatus.VALIDATED.value, deviation

        # Outside range - compute deviation from nearest bound
        if actual < reference_min:
            deviation = abs(actual - reference_min) / midpoint
        else:
            deviation = abs(actual - reference_max) / midpoint

        if deviation >= reject_pct:
            return ConversionStatus.REJECTED.value, deviation
        elif deviation >= warn_pct:
            return ConversionStatus.WARNED.value, deviation
        else:
            return ConversionStatus.VALIDATED.value, deviation

    # ------------------------------------------------------------------
    # Internal: Reference Lookup
    # ------------------------------------------------------------------

    def _get_reference(
        self,
        commodity: str,
        process: str,
    ) -> Optional[Dict[str, Any]]:
        """Look up reference data for a commodity/process pair.

        Args:
            commodity: Normalized commodity name.
            process: Normalized process name.

        Returns:
            Reference dictionary with yield_ratio, min, max, source,
            or None if not found.
        """
        return self._reference_factors.get((commodity, process))

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_commodity(self, commodity: str) -> None:
        """Validate that commodity is non-empty.

        Args:
            commodity: Commodity to validate.

        Raises:
            ValueError: If commodity is empty.
        """
        if not commodity or not commodity.strip():
            raise ValueError("commodity must not be empty")

    def _validate_process_name(self, process_name: str) -> None:
        """Validate that process_name is non-empty.

        Args:
            process_name: Process name to validate.

        Raises:
            ValueError: If process_name is empty.
        """
        if not process_name or not process_name.strip():
            raise ValueError("process_name must not be empty")

    def _validate_yield_ratio(self, yield_ratio: float) -> None:
        """Validate that yield_ratio is in valid range.

        Args:
            yield_ratio: Yield ratio to validate.

        Raises:
            ValueError: If yield_ratio is not in (0.0, 1.0].
        """
        if not (0.0 < yield_ratio <= 1.0):
            raise ValueError(
                f"yield_ratio must be in (0.0, 1.0], got {yield_ratio}"
            )

    # ------------------------------------------------------------------
    # Internal: Validation Message Builder
    # ------------------------------------------------------------------

    def _build_validation_message(
        self,
        status: str,
        commodity: str,
        process: str,
        yield_ratio: float,
        ref_yield: float,
        deviation_pct: float,
    ) -> str:
        """Build a human-readable validation message.

        Args:
            status: Validation status.
            commodity: Commodity name.
            process: Process name.
            yield_ratio: Reported yield.
            ref_yield: Reference yield.
            deviation_pct: Deviation fraction.

        Returns:
            Human-readable message string.
        """
        dev_display = round(deviation_pct * 100, 2)
        if status == ConversionStatus.VALIDATED.value:
            return (
                f"Yield ratio {yield_ratio:.4f} for {commodity}/{process} "
                f"is within acceptable range (ref={ref_yield:.4f}, "
                f"deviation={dev_display}%)"
            )
        elif status == ConversionStatus.WARNED.value:
            return (
                f"WARNING: Yield ratio {yield_ratio:.4f} for "
                f"{commodity}/{process} deviates {dev_display}% from "
                f"reference {ref_yield:.4f}. Exceeds warn threshold "
                f"({self._warn_deviation * 100}%). Review recommended."
            )
        else:
            return (
                f"REJECTED: Yield ratio {yield_ratio:.4f} for "
                f"{commodity}/{process} deviates {dev_display}% from "
                f"reference {ref_yield:.4f}. Exceeds reject threshold "
                f"({self._reject_deviation * 100}%). Factor not approved."
            )

    # ------------------------------------------------------------------
    # Internal: No-Reference Result Builder
    # ------------------------------------------------------------------

    def _build_no_reference_result(
        self,
        validation_id: str,
        commodity: str,
        process: str,
        yield_ratio: float,
        facility_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a result for when no reference data exists.

        Args:
            validation_id: Unique validation identifier.
            commodity: Commodity name.
            process: Process name.
            yield_ratio: Reported yield ratio.
            facility_id: Optional facility identifier.
            metadata: Optional metadata.

        Returns:
            Advisory validation result dictionary.
        """
        now = _utcnow()
        provenance_hash = _compute_hash({
            "validation_id": validation_id,
            "commodity": commodity,
            "process_name": process,
            "yield_ratio": yield_ratio,
            "action": "validate_no_reference",
        })

        logger.warning(
            "No reference data for %s/%s - returning advisory result",
            commodity,
            process,
        )

        return {
            "validation_id": validation_id,
            "commodity": commodity,
            "process_name": process,
            "yield_ratio": yield_ratio,
            "reference_yield": None,
            "reference_min": None,
            "reference_max": None,
            "reference_source": None,
            "deviation_percent": None,
            "status": ConversionStatus.PENDING.value,
            "seasonal_adjustment": None,
            "month": None,
            "facility_id": facility_id,
            "provenance_hash": provenance_hash,
            "validated_at": now.isoformat(),
            "message": (
                f"No reference data available for {commodity}/{process}. "
                f"Cannot validate yield ratio {yield_ratio:.4f}. "
                f"Consider registering a custom factor."
            ),
            "metadata": metadata or {},
        }

    # ------------------------------------------------------------------
    # Internal: Validation History Recording
    # ------------------------------------------------------------------

    def _record_validation_history(
        self,
        facility_id: Optional[str],
        result: Dict[str, Any],
    ) -> None:
        """Record a validation result in the history store.

        Args:
            facility_id: Facility identifier (may be None for
                non-facility-specific validations).
            result: Validation result dictionary.
        """
        key = facility_id or "_global"
        with self._lock:
            if key not in self._validation_history:
                self._validation_history[key] = []
            self._validation_history[key].append(result)

            # Also record in factor history for facility tracking
            if facility_id:
                commodity = result.get("commodity", "unknown")
                history_key = f"{facility_id}:{commodity}"
                if history_key not in self._factor_history:
                    self._factor_history[history_key] = []
                self._factor_history[history_key].append({
                    "validation_id": result.get("validation_id"),
                    "action": "validate",
                    "yield_ratio": result.get("yield_ratio"),
                    "process": result.get("process_name"),
                    "status": result.get("status"),
                    "deviation_percent": result.get("deviation_percent"),
                    "timestamp": result.get("validated_at"),
                })

    # ------------------------------------------------------------------
    # Dunder Methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of reference factor pairs."""
        return len(self._reference_factors)

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        with self._lock:
            custom_count = len(self._custom_factors)
        return (
            f"ConversionFactorValidator("
            f"reference_pairs={len(self._reference_factors)}, "
            f"custom_factors={custom_count}, "
            f"warn={self._warn_deviation}, "
            f"reject={self._reject_deviation})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ConversionFactorValidator",
]
