# -*- coding: utf-8 -*-
"""
MultiRegulationConsistencyEngine - PACK-009 EU Climate Compliance Bundle Engine 6

Validates that shared data points reported across multiple EU regulations
(CSRD, CBAM, EU Taxonomy, EUDR) are consistent. Detects conflicts, applies
configurable tolerance-based comparison, and optionally auto-resolves low-
severity discrepancies using deterministic resolution strategies.

Capabilities:
    1. Cross-regulation data consistency checking
    2. Tolerance-based numeric comparison (STRICT / TOLERANT / FUZZY)
    3. Exact match for categorical fields
    4. Date-range comparison for temporal fields
    5. Automated conflict resolution (highest confidence, most recent)
    6. Correction propagation across regulations
    7. Reconciliation report generation

Shared-Data Scope:
    ~60 data fields that appear in 2 or more of the 4 regulations, grouped
    into categories: Emissions, Financial, Organisational, Supply-Chain,
    Environmental, and Governance.

Zero-Hallucination:
    - Comparison logic is purely arithmetic (delta, tolerance %)
    - Resolution strategies are deterministic rule-based
    - No LLM involvement in any decision path
    - SHA-256 provenance hash on all results

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-009 EU Climate Compliance Bundle
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    elif isinstance(data, list):
        serializable = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in data
        ]
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning *default* when denominator is zero."""
    if denominator == 0.0:
        return default
    return numerator / denominator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ComparisonMode(str, Enum):
    """How strictly values are compared."""
    STRICT = "STRICT"
    TOLERANT = "TOLERANT"
    FUZZY = "FUZZY"

class ConsistencyLevel(str, Enum):
    """Status of a consistency check."""
    CONSISTENT = "CONSISTENT"
    MINOR_DEVIATION = "MINOR_DEVIATION"
    MAJOR_DEVIATION = "MAJOR_DEVIATION"
    CONFLICT = "CONFLICT"

class FieldType(str, Enum):
    """Data type of a shared field."""
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    TEMPORAL = "TEMPORAL"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"

class ResolutionStrategy(str, Enum):
    """Strategy for auto-resolving conflicts."""
    HIGHEST_CONFIDENCE = "HIGHEST_CONFIDENCE"
    MOST_RECENT = "MOST_RECENT"
    AVERAGE = "AVERAGE"
    MANUAL = "MANUAL"

class FieldCategory(str, Enum):
    """Category grouping for shared fields."""
    EMISSIONS = "EMISSIONS"
    FINANCIAL = "FINANCIAL"
    ORGANISATIONAL = "ORGANISATIONAL"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    ENVIRONMENTAL = "ENVIRONMENTAL"
    GOVERNANCE = "GOVERNANCE"

# ---------------------------------------------------------------------------
# Reference Data - Shared Fields Across Regulations
# ---------------------------------------------------------------------------

SHARED_DATA_FIELDS: Dict[str, Dict[str, Any]] = {
    # --- EMISSIONS (13 fields) ---
    "scope1_total_emissions": {"type": "NUMERIC", "unit": "tCO2e", "category": "EMISSIONS", "regulations": ["CSRD", "CBAM", "EU_TAXONOMY"]},
    "scope2_total_emissions": {"type": "NUMERIC", "unit": "tCO2e", "category": "EMISSIONS", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "scope3_total_emissions": {"type": "NUMERIC", "unit": "tCO2e", "category": "EMISSIONS", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "scope3_cat1_emissions": {"type": "NUMERIC", "unit": "tCO2e", "category": "EMISSIONS", "regulations": ["CSRD", "CBAM"]},
    "embedded_emissions_total": {"type": "NUMERIC", "unit": "tCO2e", "category": "EMISSIONS", "regulations": ["CBAM", "EU_TAXONOMY"]},
    "ghg_intensity_revenue": {"type": "NUMERIC", "unit": "tCO2e/EUR_m", "category": "EMISSIONS", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "ghg_intensity_product": {"type": "NUMERIC", "unit": "tCO2e/unit", "category": "EMISSIONS", "regulations": ["CBAM", "EU_TAXONOMY"]},
    "energy_consumption_total": {"type": "NUMERIC", "unit": "MWh", "category": "EMISSIONS", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "renewable_energy_pct": {"type": "NUMERIC", "unit": "%", "category": "EMISSIONS", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "emission_factor_source": {"type": "CATEGORICAL", "unit": "text", "category": "EMISSIONS", "regulations": ["CSRD", "CBAM"]},
    "ghg_accounting_standard": {"type": "CATEGORICAL", "unit": "text", "category": "EMISSIONS", "regulations": ["CSRD", "CBAM", "EU_TAXONOMY"]},
    "base_year_emissions": {"type": "NUMERIC", "unit": "tCO2e", "category": "EMISSIONS", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "emissions_reduction_target_pct": {"type": "NUMERIC", "unit": "%", "category": "EMISSIONS", "regulations": ["CSRD", "EU_TAXONOMY"]},

    # --- FINANCIAL (12 fields) ---
    "total_revenue": {"type": "NUMERIC", "unit": "EUR", "category": "FINANCIAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "taxonomy_eligible_turnover": {"type": "NUMERIC", "unit": "EUR", "category": "FINANCIAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "taxonomy_aligned_turnover": {"type": "NUMERIC", "unit": "EUR", "category": "FINANCIAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "taxonomy_eligible_capex": {"type": "NUMERIC", "unit": "EUR", "category": "FINANCIAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "taxonomy_aligned_capex": {"type": "NUMERIC", "unit": "EUR", "category": "FINANCIAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "taxonomy_eligible_opex": {"type": "NUMERIC", "unit": "EUR", "category": "FINANCIAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "taxonomy_aligned_opex": {"type": "NUMERIC", "unit": "EUR", "category": "FINANCIAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "carbon_price_exposure": {"type": "NUMERIC", "unit": "EUR", "category": "FINANCIAL", "regulations": ["CSRD", "CBAM"]},
    "cbam_financial_liability": {"type": "NUMERIC", "unit": "EUR", "category": "FINANCIAL", "regulations": ["CBAM", "CSRD"]},
    "transition_plan_capex": {"type": "NUMERIC", "unit": "EUR", "category": "FINANCIAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "import_value_total": {"type": "NUMERIC", "unit": "EUR", "category": "FINANCIAL", "regulations": ["CBAM", "EUDR"]},
    "deforestation_risk_exposure": {"type": "NUMERIC", "unit": "EUR", "category": "FINANCIAL", "regulations": ["EUDR", "CSRD"]},

    # --- ORGANISATIONAL (8 fields) ---
    "company_name": {"type": "TEXT", "unit": "text", "category": "ORGANISATIONAL", "regulations": ["CSRD", "CBAM", "EU_TAXONOMY", "EUDR"]},
    "company_lei": {"type": "TEXT", "unit": "text", "category": "ORGANISATIONAL", "regulations": ["CSRD", "CBAM", "EU_TAXONOMY", "EUDR"]},
    "reporting_period_start": {"type": "TEMPORAL", "unit": "date", "category": "ORGANISATIONAL", "regulations": ["CSRD", "CBAM", "EU_TAXONOMY", "EUDR"]},
    "reporting_period_end": {"type": "TEMPORAL", "unit": "date", "category": "ORGANISATIONAL", "regulations": ["CSRD", "CBAM", "EU_TAXONOMY", "EUDR"]},
    "fiscal_year": {"type": "CATEGORICAL", "unit": "year", "category": "ORGANISATIONAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "employee_count": {"type": "NUMERIC", "unit": "count", "category": "ORGANISATIONAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "consolidation_method": {"type": "CATEGORICAL", "unit": "text", "category": "ORGANISATIONAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "nace_codes": {"type": "TEXT", "unit": "text", "category": "ORGANISATIONAL", "regulations": ["CSRD", "EU_TAXONOMY", "CBAM"]},

    # --- SUPPLY CHAIN (12 fields) ---
    "supplier_count_total": {"type": "NUMERIC", "unit": "count", "category": "SUPPLY_CHAIN", "regulations": ["CSRD", "CBAM", "EUDR"]},
    "supplier_countries": {"type": "TEXT", "unit": "list", "category": "SUPPLY_CHAIN", "regulations": ["CBAM", "EUDR"]},
    "commodity_types": {"type": "TEXT", "unit": "list", "category": "SUPPLY_CHAIN", "regulations": ["CBAM", "EUDR"]},
    "origin_countries": {"type": "TEXT", "unit": "list", "category": "SUPPLY_CHAIN", "regulations": ["CBAM", "EUDR"]},
    "supply_chain_mapped_pct": {"type": "NUMERIC", "unit": "%", "category": "SUPPLY_CHAIN", "regulations": ["CSRD", "EUDR"]},
    "supplier_engagement_pct": {"type": "NUMERIC", "unit": "%", "category": "SUPPLY_CHAIN", "regulations": ["CSRD", "CBAM"]},
    "verified_supplier_data_pct": {"type": "NUMERIC", "unit": "%", "category": "SUPPLY_CHAIN", "regulations": ["CBAM", "EUDR"]},
    "traceability_depth": {"type": "NUMERIC", "unit": "levels", "category": "SUPPLY_CHAIN", "regulations": ["CBAM", "EUDR"]},
    "due_diligence_completed": {"type": "NUMERIC", "unit": "count", "category": "SUPPLY_CHAIN", "regulations": ["CSRD", "EUDR"]},
    "import_volume_tonnes": {"type": "NUMERIC", "unit": "tonnes", "category": "SUPPLY_CHAIN", "regulations": ["CBAM", "EUDR"]},
    "high_risk_suppliers": {"type": "NUMERIC", "unit": "count", "category": "SUPPLY_CHAIN", "regulations": ["CBAM", "EUDR", "CSRD"]},
    "supplier_audit_count": {"type": "NUMERIC", "unit": "count", "category": "SUPPLY_CHAIN", "regulations": ["CSRD", "EUDR"]},

    # --- ENVIRONMENTAL (9 fields) ---
    "deforestation_free_pct": {"type": "NUMERIC", "unit": "%", "category": "ENVIRONMENTAL", "regulations": ["EUDR", "CSRD"]},
    "biodiversity_impact_assessed": {"type": "BOOLEAN", "unit": "bool", "category": "ENVIRONMENTAL", "regulations": ["CSRD", "EU_TAXONOMY", "EUDR"]},
    "water_consumption_total": {"type": "NUMERIC", "unit": "m3", "category": "ENVIRONMENTAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "waste_total_generated": {"type": "NUMERIC", "unit": "tonnes", "category": "ENVIRONMENTAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "circular_material_use_pct": {"type": "NUMERIC", "unit": "%", "category": "ENVIRONMENTAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "land_use_change_area": {"type": "NUMERIC", "unit": "ha", "category": "ENVIRONMENTAL", "regulations": ["EUDR", "CSRD"]},
    "pollution_to_water": {"type": "NUMERIC", "unit": "tonnes", "category": "ENVIRONMENTAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "pollution_to_air": {"type": "NUMERIC", "unit": "tonnes", "category": "ENVIRONMENTAL", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "environmental_incidents": {"type": "NUMERIC", "unit": "count", "category": "ENVIRONMENTAL", "regulations": ["CSRD", "EU_TAXONOMY", "EUDR"]},

    # --- GOVERNANCE (6 fields) ---
    "board_sustainability_oversight": {"type": "BOOLEAN", "unit": "bool", "category": "GOVERNANCE", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "sustainability_due_diligence_policy": {"type": "BOOLEAN", "unit": "bool", "category": "GOVERNANCE", "regulations": ["CSRD", "EUDR"]},
    "human_rights_policy": {"type": "BOOLEAN", "unit": "bool", "category": "GOVERNANCE", "regulations": ["CSRD", "EU_TAXONOMY", "EUDR"]},
    "minimum_safeguards_met": {"type": "BOOLEAN", "unit": "bool", "category": "GOVERNANCE", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "anti_corruption_policy": {"type": "BOOLEAN", "unit": "bool", "category": "GOVERNANCE", "regulations": ["CSRD", "EU_TAXONOMY"]},
    "stakeholder_engagement_process": {"type": "BOOLEAN", "unit": "bool", "category": "GOVERNANCE", "regulations": ["CSRD", "EU_TAXONOMY", "EUDR"]},
}

# Tolerance defaults per comparison mode
_TOLERANCE_DEFAULTS: Dict[str, float] = {
    "STRICT": 0.01,
    "TOLERANT": 5.0,
    "FUZZY": 15.0,
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConsistencyConfig(BaseModel):
    """Configuration for the MultiRegulationConsistencyEngine."""

    tolerance_pct: float = Field(
        default=5.0, ge=0.0, le=100.0,
        description="Default tolerance percentage for numeric comparisons",
    )
    auto_resolve_threshold: float = Field(
        default=2.0, ge=0.0, le=100.0,
        description="Maximum deviation pct for auto-resolution",
    )
    comparison_mode: ComparisonMode = Field(
        default=ComparisonMode.TOLERANT,
        description="How strictly values are compared (STRICT/TOLERANT/FUZZY)",
    )
    default_resolution_strategy: ResolutionStrategy = Field(
        default=ResolutionStrategy.MOST_RECENT,
        description="Default strategy for resolving conflicts",
    )
    field_overrides: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-field tolerance overrides (field_name -> tolerance_pct)",
    )
    date_tolerance_days: int = Field(
        default=7, ge=0, le=90,
        description="Tolerance window in days for temporal comparisons",
    )

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class DataPoint(BaseModel):
    """A single data value reported for a given field under a specific regulation."""

    regulation: str = Field(..., description="Regulation that reported this value")
    field_name: str = Field(..., description="Shared field name")
    value: Any = Field(..., description="Reported value")
    unit: str = Field(default="", description="Unit of the value")
    timestamp: str = Field(default="", description="ISO-8601 timestamp of the report")
    source: str = Field(default="", description="Data source identifier")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score 0-1")

class ConflictResolution(BaseModel):
    """Details of how a conflict was or should be resolved."""

    resolution_id: str = Field(default_factory=_new_uuid, description="Resolution identifier")
    strategy_used: str = Field(default="", description="Strategy used for resolution")
    resolved_value: Any = Field(default=None, description="The value chosen after resolution")
    auto_resolved: bool = Field(default=False, description="Whether resolution was automatic")
    rationale: str = Field(default="", description="Explanation of the resolution")

class ConsistencyCheck(BaseModel):
    """Result of a consistency check for a single shared field."""

    check_id: str = Field(default_factory=_new_uuid, description="Check identifier")
    field_name: str = Field(..., description="Shared field name")
    field_category: str = Field(default="", description="Field category (EMISSIONS, FINANCIAL, etc.)")
    field_type: str = Field(default="", description="Field data type")
    data_points: List[DataPoint] = Field(default_factory=list, description="All reported values")
    status: str = Field(default="CONSISTENT", description="Consistency status")
    max_deviation: float = Field(default=0.0, description="Maximum deviation between values")
    deviation_pct: float = Field(default=0.0, description="Deviation as percentage")
    resolution: Optional[ConflictResolution] = Field(default=None, description="Conflict resolution details")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class ReconciliationItem(BaseModel):
    """A single row in a reconciliation report."""

    field_name: str = Field(..., description="Field name")
    category: str = Field(default="", description="Field category")
    regulations_involved: List[str] = Field(default_factory=list, description="Regulations reporting this field")
    values_reported: Dict[str, Any] = Field(default_factory=dict, description="Regulation -> value mapping")
    status: str = Field(default="CONSISTENT", description="Consistency status")
    action_required: str = Field(default="NONE", description="Required action (NONE, REVIEW, CORRECT)")

class ConsistencyResult(BaseModel):
    """Complete result from the MultiRegulationConsistencyEngine."""

    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    checks: List[ConsistencyCheck] = Field(default_factory=list, description="All consistency checks")
    conflicts: List[ConsistencyCheck] = Field(default_factory=list, description="Checks that found conflicts")
    consistency_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall consistency score 0-100")
    auto_resolved_count: int = Field(default=0, ge=0, description="Number of auto-resolved conflicts")
    total_fields_checked: int = Field(default=0, ge=0, description="Total fields checked")
    consistent_count: int = Field(default=0, ge=0, description="Number of consistent fields")
    conflict_count: int = Field(default=0, ge=0, description="Number of conflicting fields")
    reconciliation_items: List[ReconciliationItem] = Field(
        default_factory=list, description="Reconciliation report items"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MultiRegulationConsistencyEngine:
    """
    Validates that shared data points are consistent across 4 EU regulations.

    Comparison logic is entirely deterministic:
      - Numeric fields: tolerance-based percentage comparison
      - Categorical fields: exact string match
      - Temporal fields: date-range window comparison
      - Boolean fields: exact match
      - Text fields: case-insensitive equality

    Attributes:
        config: Engine configuration.

    Example:
        >>> config = ConsistencyConfig(tolerance_pct=5.0)
        >>> engine = MultiRegulationConsistencyEngine(config)
        >>> data_points = [
        ...     DataPoint(regulation="CSRD", field_name="scope1_total_emissions", value=1000),
        ...     DataPoint(regulation="CBAM", field_name="scope1_total_emissions", value=1020),
        ... ]
        >>> result = engine.check_consistency(data_points)
        >>> assert result.consistency_score > 90
    """

    def __init__(self, config: Optional[ConsistencyConfig] = None) -> None:
        """Initialize the MultiRegulationConsistencyEngine.

        Args:
            config: Engine configuration. Uses defaults when *None*.
        """
        self.config = config or ConsistencyConfig()
        logger.info("MultiRegulationConsistencyEngine v%s initialised", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_consistency(self, data_points: List[DataPoint]) -> ConsistencyResult:
        """Run consistency checks across all shared data points.

        Data points are grouped by field_name. For each group with 2+ data
        points, the engine runs the appropriate comparison and flags conflicts.

        Args:
            data_points: List of reported data points across regulations.

        Returns:
            ConsistencyResult with checks, conflicts, and consistency score.
        """
        start = utcnow()

        # Group by field_name
        groups: Dict[str, List[DataPoint]] = {}
        for dp in data_points:
            groups.setdefault(dp.field_name, []).append(dp)

        checks: List[ConsistencyCheck] = []
        conflicts: List[ConsistencyCheck] = []
        auto_resolved = 0

        for field_name, dps in sorted(groups.items()):
            if len(dps) < 2:
                # Need at least 2 data points to compare
                continue

            field_def = SHARED_DATA_FIELDS.get(field_name, {})
            field_type = field_def.get("type", "TEXT")
            field_category = field_def.get("category", "UNKNOWN")

            check = self._compare_values(field_name, field_type, field_category, dps)

            # Auto-resolve if within threshold
            if check.status in (ConsistencyLevel.MINOR_DEVIATION.value, ConsistencyLevel.MAJOR_DEVIATION.value):
                resolution = self.auto_resolve(check)
                if resolution is not None and resolution.auto_resolved:
                    check.resolution = resolution
                    auto_resolved += 1

            check.provenance_hash = _compute_hash(check)
            checks.append(check)

            if check.status != ConsistencyLevel.CONSISTENT.value:
                conflicts.append(check)

        consistent_count = len(checks) - len(conflicts)
        total = len(checks)
        score = _safe_div(consistent_count, total) * 100.0 if total > 0 else 100.0

        reconciliation = self.generate_reconciliation_report(checks)

        elapsed_ms = (utcnow() - start).total_seconds() * 1000

        result = ConsistencyResult(
            checks=checks,
            conflicts=conflicts,
            consistency_score=round(score, 2),
            auto_resolved_count=auto_resolved,
            total_fields_checked=total,
            consistent_count=consistent_count,
            conflict_count=len(conflicts),
            reconciliation_items=reconciliation,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        logger.info(
            "Consistency check complete: score=%.2f, fields=%d, conflicts=%d, auto_resolved=%d",
            score, total, len(conflicts), auto_resolved,
        )
        return result

    def detect_conflicts(self, data_points: List[DataPoint]) -> List[ConsistencyCheck]:
        """Detect conflicts without running full consistency analysis.

        A convenience method that returns only fields with deviations.

        Args:
            data_points: Reported data points.

        Returns:
            List of ConsistencyCheck objects where status is not CONSISTENT.
        """
        result = self.check_consistency(data_points)
        return result.conflicts

    def auto_resolve(self, check: ConsistencyCheck) -> Optional[ConflictResolution]:
        """Attempt automatic resolution of a consistency conflict.

        Auto-resolution succeeds only when the deviation is within the
        auto_resolve_threshold. The strategy is determined by the config.

        Args:
            check: A consistency check with a detected conflict.

        Returns:
            ConflictResolution if auto-resolved, *None* otherwise.
        """
        if check.deviation_pct > self.config.auto_resolve_threshold:
            return ConflictResolution(
                strategy_used=ResolutionStrategy.MANUAL.value,
                auto_resolved=False,
                rationale=f"Deviation {check.deviation_pct:.2f}% exceeds auto-resolve threshold "
                          f"({self.config.auto_resolve_threshold}%); manual review required.",
            )

        strategy = self.config.default_resolution_strategy
        data_points = check.data_points

        if not data_points:
            return None

        if strategy == ResolutionStrategy.HIGHEST_CONFIDENCE:
            resolved = self._resolve_highest_confidence(data_points)
        elif strategy == ResolutionStrategy.MOST_RECENT:
            resolved = self._resolve_most_recent(data_points)
        elif strategy == ResolutionStrategy.AVERAGE:
            resolved = self._resolve_average(data_points)
        else:
            return ConflictResolution(
                strategy_used=ResolutionStrategy.MANUAL.value,
                auto_resolved=False,
                rationale="Manual resolution strategy configured.",
            )

        return ConflictResolution(
            strategy_used=strategy.value,
            resolved_value=resolved,
            auto_resolved=True,
            rationale=f"Auto-resolved using {strategy.value} strategy "
                      f"(deviation {check.deviation_pct:.2f}% within threshold "
                      f"{self.config.auto_resolve_threshold}%).",
        )

    def propagate_correction(
        self,
        field_name: str,
        corrected_value: Any,
        data_points: List[DataPoint],
    ) -> List[DataPoint]:
        """Propagate a corrected value to all regulations referencing a field.

        Creates new DataPoint objects with the corrected value for each
        regulation that originally reported this field.

        Args:
            field_name: The shared field name.
            corrected_value: The corrected value to propagate.
            data_points: Original data points.

        Returns:
            List of updated DataPoint objects.
        """
        updated: List[DataPoint] = []
        now_str = utcnow().isoformat()
        for dp in data_points:
            if dp.field_name == field_name:
                updated.append(DataPoint(
                    regulation=dp.regulation,
                    field_name=dp.field_name,
                    value=corrected_value,
                    unit=dp.unit,
                    timestamp=now_str,
                    source=f"correction_propagation:{dp.source}",
                    confidence=dp.confidence,
                ))
            else:
                updated.append(dp)
        logger.info(
            "Propagated correction for '%s' to %d data points",
            field_name, sum(1 for dp in updated if dp.field_name == field_name),
        )
        return updated

    def get_consistency_score(self, data_points: List[DataPoint]) -> float:
        """Compute overall consistency score without full result.

        Args:
            data_points: Reported data points.

        Returns:
            Consistency score 0-100.
        """
        result = self.check_consistency(data_points)
        return result.consistency_score

    def generate_reconciliation_report(
        self, checks: List[ConsistencyCheck]
    ) -> List[ReconciliationItem]:
        """Generate a reconciliation report from consistency checks.

        Args:
            checks: List of completed consistency checks.

        Returns:
            List of ReconciliationItem objects for review.
        """
        items: List[ReconciliationItem] = []
        for check in checks:
            values_map: Dict[str, Any] = {}
            regs: List[str] = []
            for dp in check.data_points:
                values_map[dp.regulation] = dp.value
                if dp.regulation not in regs:
                    regs.append(dp.regulation)

            action = "NONE"
            if check.status == ConsistencyLevel.CONFLICT.value:
                action = "CORRECT"
            elif check.status == ConsistencyLevel.MAJOR_DEVIATION.value:
                action = "REVIEW"
            elif check.status == ConsistencyLevel.MINOR_DEVIATION.value:
                if check.resolution is None or not check.resolution.auto_resolved:
                    action = "REVIEW"

            items.append(ReconciliationItem(
                field_name=check.field_name,
                category=check.field_category,
                regulations_involved=regs,
                values_reported=values_map,
                status=check.status,
                action_required=action,
            ))
        return items

    # ------------------------------------------------------------------
    # Comparison Logic
    # ------------------------------------------------------------------

    def _compare_values(
        self,
        field_name: str,
        field_type: str,
        field_category: str,
        data_points: List[DataPoint],
    ) -> ConsistencyCheck:
        """Route comparison to the appropriate type-specific method."""
        if field_type == FieldType.NUMERIC.value:
            return self._compare_numeric(field_name, field_category, data_points)
        elif field_type == FieldType.CATEGORICAL.value:
            return self._compare_categorical(field_name, field_category, data_points)
        elif field_type == FieldType.TEMPORAL.value:
            return self._compare_temporal(field_name, field_category, data_points)
        elif field_type == FieldType.BOOLEAN.value:
            return self._compare_boolean(field_name, field_category, data_points)
        else:
            return self._compare_text(field_name, field_category, data_points)

    def _compare_numeric(
        self, field_name: str, field_category: str, data_points: List[DataPoint]
    ) -> ConsistencyCheck:
        """Compare numeric values using tolerance-based approach."""
        values = []
        for dp in data_points:
            try:
                values.append(float(dp.value))
            except (ValueError, TypeError):
                values.append(0.0)

        if not values:
            return self._make_check(field_name, field_category, "NUMERIC", data_points, ConsistencyLevel.CONSISTENT, 0.0, 0.0)

        min_val = min(values)
        max_val = max(values)
        deviation = max_val - min_val
        mean_val = sum(values) / len(values)
        deviation_pct = _safe_div(deviation, abs(mean_val)) * 100.0 if mean_val != 0.0 else (0.0 if deviation == 0.0 else 100.0)

        tolerance = self.config.field_overrides.get(field_name, self._get_tolerance())
        status = self._classify_numeric_deviation(deviation_pct, tolerance)

        return self._make_check(
            field_name, field_category, "NUMERIC", data_points,
            status, round(deviation, 6), round(deviation_pct, 4),
        )

    def _compare_categorical(
        self, field_name: str, field_category: str, data_points: List[DataPoint]
    ) -> ConsistencyCheck:
        """Compare categorical values using exact string match."""
        values = [str(dp.value).strip().lower() for dp in data_points]
        unique = set(values)

        if len(unique) <= 1:
            status = ConsistencyLevel.CONSISTENT
        else:
            status = ConsistencyLevel.CONFLICT

        return self._make_check(
            field_name, field_category, "CATEGORICAL", data_points,
            status, 0.0, 0.0 if len(unique) <= 1 else 100.0,
        )

    def _compare_temporal(
        self, field_name: str, field_category: str, data_points: List[DataPoint]
    ) -> ConsistencyCheck:
        """Compare temporal values using date-range tolerance."""
        dates: List[datetime] = []
        for dp in data_points:
            parsed = self._parse_date(dp.value)
            if parsed is not None:
                dates.append(parsed)

        if len(dates) < 2:
            return self._make_check(
                field_name, field_category, "TEMPORAL", data_points,
                ConsistencyLevel.CONSISTENT, 0.0, 0.0,
            )

        min_date = min(dates)
        max_date = max(dates)
        delta_days = (max_date - min_date).days

        if delta_days <= self.config.date_tolerance_days:
            status = ConsistencyLevel.CONSISTENT
        elif delta_days <= self.config.date_tolerance_days * 3:
            status = ConsistencyLevel.MINOR_DEVIATION
        elif delta_days <= self.config.date_tolerance_days * 10:
            status = ConsistencyLevel.MAJOR_DEVIATION
        else:
            status = ConsistencyLevel.CONFLICT

        return self._make_check(
            field_name, field_category, "TEMPORAL", data_points,
            status, float(delta_days), 0.0,
        )

    def _compare_boolean(
        self, field_name: str, field_category: str, data_points: List[DataPoint]
    ) -> ConsistencyCheck:
        """Compare boolean values using exact match."""
        values = set()
        for dp in data_points:
            if isinstance(dp.value, bool):
                values.add(dp.value)
            elif isinstance(dp.value, str):
                values.add(dp.value.strip().lower() in ("true", "1", "yes"))
            else:
                values.add(bool(dp.value))

        if len(values) <= 1:
            status = ConsistencyLevel.CONSISTENT
        else:
            status = ConsistencyLevel.CONFLICT

        return self._make_check(
            field_name, field_category, "BOOLEAN", data_points,
            status, 0.0, 0.0 if len(values) <= 1 else 100.0,
        )

    def _compare_text(
        self, field_name: str, field_category: str, data_points: List[DataPoint]
    ) -> ConsistencyCheck:
        """Compare text values using case-insensitive equality."""
        values = [str(dp.value).strip().lower() for dp in data_points]
        unique = set(values)

        if len(unique) <= 1:
            status = ConsistencyLevel.CONSISTENT
        elif self.config.comparison_mode == ComparisonMode.FUZZY:
            # In fuzzy mode, check if values are substrings of each other
            sorted_vals = sorted(unique, key=len, reverse=True)
            all_contained = all(
                any(v in longer for longer in sorted_vals if longer != v)
                for v in sorted_vals[1:]
            )
            status = ConsistencyLevel.MINOR_DEVIATION if all_contained else ConsistencyLevel.CONFLICT
        else:
            status = ConsistencyLevel.CONFLICT

        return self._make_check(
            field_name, field_category, "TEXT", data_points,
            status, 0.0, 0.0 if len(unique) <= 1 else 100.0,
        )

    # ------------------------------------------------------------------
    # Resolution Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_highest_confidence(data_points: List[DataPoint]) -> Any:
        """Pick the value with the highest confidence score."""
        best = max(data_points, key=lambda dp: dp.confidence)
        return best.value

    @staticmethod
    def _resolve_most_recent(data_points: List[DataPoint]) -> Any:
        """Pick the most recently reported value."""
        with_ts = [dp for dp in data_points if dp.timestamp]
        if not with_ts:
            return data_points[0].value
        most_recent = max(with_ts, key=lambda dp: dp.timestamp)
        return most_recent.value

    @staticmethod
    def _resolve_average(data_points: List[DataPoint]) -> Any:
        """Average numeric values; for non-numeric, pick the first."""
        numeric_vals: List[float] = []
        for dp in data_points:
            try:
                numeric_vals.append(float(dp.value))
            except (ValueError, TypeError):
                return data_points[0].value
        if not numeric_vals:
            return data_points[0].value
        return round(sum(numeric_vals) / len(numeric_vals), 6)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_tolerance(self) -> float:
        """Return the effective tolerance based on comparison mode."""
        if self.config.tolerance_pct > 0.0:
            return self.config.tolerance_pct
        return _TOLERANCE_DEFAULTS.get(self.config.comparison_mode.value, 5.0)

    @staticmethod
    def _classify_numeric_deviation(
        deviation_pct: float, tolerance: float
    ) -> ConsistencyLevel:
        """Classify a numeric deviation against the tolerance."""
        if deviation_pct <= tolerance * 0.1:
            return ConsistencyLevel.CONSISTENT
        elif deviation_pct <= tolerance:
            return ConsistencyLevel.MINOR_DEVIATION
        elif deviation_pct <= tolerance * 3:
            return ConsistencyLevel.MAJOR_DEVIATION
        return ConsistencyLevel.CONFLICT

    @staticmethod
    def _make_check(
        field_name: str,
        field_category: str,
        field_type: str,
        data_points: List[DataPoint],
        status: ConsistencyLevel,
        max_deviation: float,
        deviation_pct: float,
    ) -> ConsistencyCheck:
        """Create a ConsistencyCheck object."""
        return ConsistencyCheck(
            field_name=field_name,
            field_category=field_category,
            field_type=field_type,
            data_points=data_points,
            status=status.value,
            max_deviation=max_deviation,
            deviation_pct=deviation_pct,
        )

    @staticmethod
    def _parse_date(value: Any) -> Optional[datetime]:
        """Parse a date from various formats."""
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None
