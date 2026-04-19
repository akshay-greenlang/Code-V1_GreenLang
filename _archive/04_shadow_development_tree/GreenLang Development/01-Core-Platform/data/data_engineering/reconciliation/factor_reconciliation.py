"""
Factor Reconciliation Engine
============================

Cross-source validation, conflict detection, and intelligent factor selection
for emission factor data from multiple sources.

Handles:
- Detecting conflicting factors across sources
- Applying priority rules for source selection
- Maintaining audit trails for factor selection decisions
- Merging and deduplicating factors

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from enum import Enum, IntEnum
from datetime import datetime, date
from decimal import Decimal
from dataclasses import dataclass, field
import logging
import hashlib
import statistics
from collections import defaultdict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving conflicting emission factors."""
    HIGHEST_PRIORITY = "highest_priority"  # Use source with highest priority
    MOST_RECENT = "most_recent"  # Use most recent data
    HIGHEST_QUALITY = "highest_quality"  # Use highest DQI score
    MOST_SPECIFIC = "most_specific"  # Use most geographically/technologically specific
    AVERAGE = "average"  # Calculate average of conflicting values
    WEIGHTED_AVERAGE = "weighted_average"  # Quality-weighted average
    CONSERVATIVE = "conservative"  # Use highest emission factor (conservative)
    MANUAL = "manual"  # Flag for manual resolution


class SourcePriority(IntEnum):
    """Priority levels for data sources (higher = more trusted)."""
    TIER_1_VERIFIED = 100  # Facility-specific verified data
    TIER_1_MEASURED = 90   # Direct measurements
    GOVERNMENT_PRIMARY = 80  # DEFRA, EPA official data
    ACADEMIC_PRIMARY = 70   # Ecoinvent, peer-reviewed
    GOVERNMENT_SECONDARY = 60  # IEA, World Bank
    INDUSTRY_AVERAGE = 50   # Industry association data
    CALCULATED = 40         # Calculated/derived values
    ESTIMATED = 30          # Qualified estimates
    DEFAULT = 20            # Default values
    UNKNOWN = 10            # Unknown source


class ConflictSeverity(str, Enum):
    """Severity of detected conflicts."""
    CRITICAL = "critical"  # >50% difference
    HIGH = "high"          # 25-50% difference
    MEDIUM = "medium"      # 10-25% difference
    LOW = "low"            # <10% difference
    NONE = "none"          # No conflict


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class FactorConflict:
    """Represents a conflict between emission factors."""
    conflict_id: str
    matching_key: str  # Key used to identify matching factors
    factors: List[Dict[str, Any]]
    severity: ConflictSeverity
    percent_difference: float
    min_value: float
    max_value: float
    mean_value: float
    sources_involved: List[str]
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    resolved_factor: Optional[Dict[str, Any]] = None
    resolution_reason: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.utcnow)


class ReconciliationResult(BaseModel):
    """Result of factor reconciliation process."""
    run_id: str
    total_factors_input: int
    unique_factors: int
    duplicate_factors: int
    conflicts_detected: int
    conflicts_resolved: int
    conflicts_pending: int
    critical_conflicts: int
    reconciled_factors: List[Dict[str, Any]] = Field(default_factory=list)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


# =============================================================================
# SOURCE PRIORITY CONFIGURATION
# =============================================================================

# Default source priority rankings
DEFAULT_SOURCE_PRIORITIES: Dict[str, SourcePriority] = {
    # Government primary sources
    'defra': SourcePriority.GOVERNMENT_PRIMARY,
    'epa_egrid': SourcePriority.GOVERNMENT_PRIMARY,
    'epa_ghg': SourcePriority.GOVERNMENT_PRIMARY,
    'ipcc_ar6': SourcePriority.GOVERNMENT_PRIMARY,
    'ipcc_ar7': SourcePriority.GOVERNMENT_PRIMARY,

    # Academic/database sources
    'ecoinvent': SourcePriority.ACADEMIC_PRIMARY,

    # International organizations
    'iea': SourcePriority.GOVERNMENT_SECONDARY,
    'world_bank': SourcePriority.GOVERNMENT_SECONDARY,
    'fao': SourcePriority.GOVERNMENT_SECONDARY,

    # Derived/calculated
    'calculated': SourcePriority.CALCULATED,
    'customer_provided': SourcePriority.ESTIMATED,

    # Unknown
    'unknown': SourcePriority.UNKNOWN,
}


# =============================================================================
# FACTOR RECONCILIATION ENGINE
# =============================================================================

class FactorReconciler:
    """
    Emission Factor Reconciliation Engine.

    Handles cross-source validation, conflict detection, and intelligent
    factor selection for emission factors from multiple data sources.

    Features:
    - Configurable matching keys (product, region, scope, etc.)
    - Multiple conflict resolution strategies
    - Source priority system
    - Quality-based selection
    - Full audit trail
    """

    # Matching key templates
    MATCHING_KEY_TEMPLATES = {
        'strict': ['product_code', 'country_code', 'scope_type', 'reference_year'],
        'standard': ['product_name', 'region', 'scope_type', 'reference_year'],
        'loose': ['industry', 'region', 'scope_type'],
        'cbam': ['product_code', 'country_code', 'production_route'],
    }

    def __init__(
        self,
        matching_key_template: str = 'standard',
        custom_matching_keys: Optional[List[str]] = None,
        default_resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_QUALITY,
        source_priorities: Optional[Dict[str, SourcePriority]] = None,
        conflict_threshold_percent: float = 10.0,
        enable_audit: bool = True,
    ):
        """
        Initialize factor reconciler.

        Args:
            matching_key_template: Template name for matching keys
            custom_matching_keys: Custom list of fields for matching
            default_resolution_strategy: Default strategy for resolving conflicts
            source_priorities: Custom source priority mappings
            conflict_threshold_percent: % difference threshold for conflict detection
            enable_audit: Whether to maintain audit trail
        """
        self.matching_keys = (
            custom_matching_keys or
            self.MATCHING_KEY_TEMPLATES.get(matching_key_template, self.MATCHING_KEY_TEMPLATES['standard'])
        )
        self.default_resolution_strategy = default_resolution_strategy
        self.source_priorities = source_priorities or DEFAULT_SOURCE_PRIORITIES
        self.conflict_threshold_percent = conflict_threshold_percent
        self.enable_audit = enable_audit

        self._audit_trail: List[Dict[str, Any]] = []

    def reconcile(
        self,
        factors: List[Dict[str, Any]],
        resolution_overrides: Optional[Dict[str, ConflictResolutionStrategy]] = None
    ) -> ReconciliationResult:
        """
        Reconcile emission factors from multiple sources.

        Args:
            factors: List of emission factors to reconcile
            resolution_overrides: Override resolution strategy by matching key

        Returns:
            ReconciliationResult with reconciled factors and conflict details
        """
        run_id = self._generate_run_id()
        started_at = datetime.utcnow()

        logger.info(f"Starting reconciliation run {run_id} with {len(factors)} factors")

        # Step 1: Group factors by matching key
        factor_groups = self._group_factors(factors)

        # Step 2: Detect conflicts
        conflicts = self._detect_conflicts(factor_groups)

        # Step 3: Resolve conflicts
        resolved_factors = []
        resolved_count = 0
        pending_count = 0

        for key, group in factor_groups.items():
            if len(group) == 1:
                # No conflict - use single factor
                resolved_factors.append(group[0])
                self._audit("single_factor", key, group[0])
            else:
                # Multiple factors - need resolution
                conflict = next((c for c in conflicts if c.matching_key == key), None)

                if conflict:
                    strategy = (
                        resolution_overrides.get(key) if resolution_overrides
                        else self.default_resolution_strategy
                    )

                    resolved_factor = self._resolve_conflict(conflict, strategy)

                    if resolved_factor:
                        resolved_factors.append(resolved_factor)
                        conflict.resolved_factor = resolved_factor
                        conflict.resolution_strategy = strategy
                        resolved_count += 1
                        self._audit("conflict_resolved", key, resolved_factor, strategy.value)
                    else:
                        pending_count += 1
                        self._audit("conflict_pending", key, None, "manual_review_required")

        # Calculate statistics
        critical_conflicts = sum(1 for c in conflicts if c.severity == ConflictSeverity.CRITICAL)

        completed_at = datetime.utcnow()

        result = ReconciliationResult(
            run_id=run_id,
            total_factors_input=len(factors),
            unique_factors=len(factor_groups),
            duplicate_factors=len(factors) - len(factor_groups),
            conflicts_detected=len(conflicts),
            conflicts_resolved=resolved_count,
            conflicts_pending=pending_count,
            critical_conflicts=critical_conflicts,
            reconciled_factors=resolved_factors,
            conflicts=[self._conflict_to_dict(c) for c in conflicts],
            audit_trail=self._audit_trail if self.enable_audit else [],
            statistics=self._calculate_statistics(factors, conflicts, resolved_factors),
            started_at=started_at,
            completed_at=completed_at,
        )

        logger.info(
            f"Reconciliation complete: {len(resolved_factors)} factors, "
            f"{len(conflicts)} conflicts ({resolved_count} resolved, {pending_count} pending)"
        )

        return result

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        content = f"reconciliation:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _group_factors(self, factors: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group factors by matching key."""
        groups = defaultdict(list)

        for factor in factors:
            key = self._generate_matching_key(factor)
            groups[key].append(factor)

        return dict(groups)

    def _generate_matching_key(self, factor: Dict[str, Any]) -> str:
        """Generate matching key from factor attributes."""
        key_parts = []

        for field in self.matching_keys:
            value = self._get_field_value(factor, field)
            key_parts.append(str(value).lower() if value else 'null')

        return ':'.join(key_parts)

    def _get_field_value(self, factor: Dict[str, Any], field: str) -> Any:
        """Get field value, handling nested fields."""
        if '.' in field:
            parts = field.split('.')
            value = factor
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return value
        return factor.get(field)

    def _detect_conflicts(
        self,
        factor_groups: Dict[str, List[Dict[str, Any]]]
    ) -> List[FactorConflict]:
        """Detect conflicts in factor groups."""
        conflicts = []

        for key, group in factor_groups.items():
            if len(group) < 2:
                continue

            # Extract factor values
            values = []
            for factor in group:
                value = factor.get('factor_value')
                if value is not None:
                    try:
                        values.append(float(value))
                    except (ValueError, TypeError):
                        pass

            if len(values) < 2:
                continue

            # Calculate statistics
            min_val = min(values)
            max_val = max(values)
            mean_val = statistics.mean(values)

            # Calculate percent difference
            if mean_val > 0:
                percent_diff = ((max_val - min_val) / mean_val) * 100
            else:
                percent_diff = 0

            # Determine severity
            severity = self._determine_conflict_severity(percent_diff)

            # Only create conflict if above threshold
            if percent_diff >= self.conflict_threshold_percent:
                sources = list(set(
                    self._get_source_name(f) for f in group
                ))

                conflict = FactorConflict(
                    conflict_id=hashlib.sha256(key.encode()).hexdigest()[:8],
                    matching_key=key,
                    factors=group,
                    severity=severity,
                    percent_difference=round(percent_diff, 2),
                    min_value=round(min_val, 6),
                    max_value=round(max_val, 6),
                    mean_value=round(mean_val, 6),
                    sources_involved=sources,
                )

                conflicts.append(conflict)
                logger.debug(f"Conflict detected: {key}, {percent_diff:.1f}% difference")

        return conflicts

    def _determine_conflict_severity(self, percent_diff: float) -> ConflictSeverity:
        """Determine conflict severity based on percent difference."""
        if percent_diff >= 50:
            return ConflictSeverity.CRITICAL
        elif percent_diff >= 25:
            return ConflictSeverity.HIGH
        elif percent_diff >= 10:
            return ConflictSeverity.MEDIUM
        elif percent_diff > 0:
            return ConflictSeverity.LOW
        else:
            return ConflictSeverity.NONE

    def _get_source_name(self, factor: Dict[str, Any]) -> str:
        """Get source name from factor."""
        source = factor.get('source', {})
        if isinstance(source, dict):
            return source.get('source_type', source.get('source_name', 'unknown'))
        return 'unknown'

    def _resolve_conflict(
        self,
        conflict: FactorConflict,
        strategy: ConflictResolutionStrategy
    ) -> Optional[Dict[str, Any]]:
        """Resolve a conflict using the specified strategy."""
        factors = conflict.factors

        if strategy == ConflictResolutionStrategy.HIGHEST_PRIORITY:
            return self._resolve_by_priority(factors)

        elif strategy == ConflictResolutionStrategy.MOST_RECENT:
            return self._resolve_by_recency(factors)

        elif strategy == ConflictResolutionStrategy.HIGHEST_QUALITY:
            return self._resolve_by_quality(factors)

        elif strategy == ConflictResolutionStrategy.MOST_SPECIFIC:
            return self._resolve_by_specificity(factors)

        elif strategy == ConflictResolutionStrategy.AVERAGE:
            return self._resolve_by_average(factors, weighted=False)

        elif strategy == ConflictResolutionStrategy.WEIGHTED_AVERAGE:
            return self._resolve_by_average(factors, weighted=True)

        elif strategy == ConflictResolutionStrategy.CONSERVATIVE:
            return self._resolve_conservative(factors)

        elif strategy == ConflictResolutionStrategy.MANUAL:
            return None  # Flag for manual review

        return self._resolve_by_priority(factors)  # Default fallback

    def _resolve_by_priority(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select factor from highest priority source."""
        def get_priority(factor: Dict) -> int:
            source_type = self._get_source_name(factor)
            return self.source_priorities.get(source_type, SourcePriority.UNKNOWN)

        sorted_factors = sorted(factors, key=get_priority, reverse=True)
        selected = sorted_factors[0]
        selected['_reconciliation'] = {
            'strategy': 'highest_priority',
            'source_priority': get_priority(selected),
        }
        return selected

    def _resolve_by_recency(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select most recent factor."""
        def get_year(factor: Dict) -> int:
            return factor.get('reference_year', 0)

        sorted_factors = sorted(factors, key=get_year, reverse=True)
        selected = sorted_factors[0]
        selected['_reconciliation'] = {
            'strategy': 'most_recent',
            'reference_year': get_year(selected),
        }
        return selected

    def _resolve_by_quality(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select highest quality factor."""
        def get_dqi(factor: Dict) -> float:
            quality = factor.get('quality', {})
            if isinstance(quality, dict):
                return quality.get('aggregate_dqi', 0) or 0
            return 0

        sorted_factors = sorted(factors, key=get_dqi, reverse=True)
        selected = sorted_factors[0]
        selected['_reconciliation'] = {
            'strategy': 'highest_quality',
            'dqi_score': get_dqi(selected),
        }
        return selected

    def _resolve_by_specificity(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select most specific factor (geographic/technological)."""
        def get_specificity_score(factor: Dict) -> int:
            score = 0

            # Geographic specificity
            if factor.get('facility_id'):
                score += 5
            elif factor.get('state_province'):
                score += 4
            elif factor.get('country_code'):
                score += 3
            elif factor.get('region') and factor.get('region') != 'global':
                score += 2

            # Technological specificity
            if factor.get('production_route'):
                score += 2

            return score

        sorted_factors = sorted(factors, key=get_specificity_score, reverse=True)
        selected = sorted_factors[0]
        selected['_reconciliation'] = {
            'strategy': 'most_specific',
            'specificity_score': get_specificity_score(selected),
        }
        return selected

    def _resolve_by_average(
        self,
        factors: List[Dict[str, Any]],
        weighted: bool = False
    ) -> Dict[str, Any]:
        """Calculate average (optionally weighted by quality)."""
        values = []
        weights = []

        for factor in factors:
            value = factor.get('factor_value')
            if value is not None:
                try:
                    values.append(float(value))

                    if weighted:
                        quality = factor.get('quality', {})
                        dqi = quality.get('aggregate_dqi', 50) if isinstance(quality, dict) else 50
                        weights.append(dqi)
                    else:
                        weights.append(1)
                except (ValueError, TypeError):
                    pass

        if not values:
            return factors[0]

        if weighted and sum(weights) > 0:
            avg_value = sum(v * w for v, w in zip(values, weights)) / sum(weights)
        else:
            avg_value = statistics.mean(values)

        # Use first factor as template, update value
        result = factors[0].copy()
        result['factor_value'] = Decimal(str(round(avg_value, 6)))
        result['_reconciliation'] = {
            'strategy': 'weighted_average' if weighted else 'average',
            'input_values': values,
            'weights': weights if weighted else None,
            'calculated_value': round(avg_value, 6),
        }

        return result

    def _resolve_conservative(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select highest emission factor (conservative approach)."""
        values_with_factors = []

        for factor in factors:
            value = factor.get('factor_value')
            if value is not None:
                try:
                    values_with_factors.append((float(value), factor))
                except (ValueError, TypeError):
                    pass

        if not values_with_factors:
            return factors[0]

        sorted_pairs = sorted(values_with_factors, key=lambda x: x[0], reverse=True)
        selected = sorted_pairs[0][1]
        selected['_reconciliation'] = {
            'strategy': 'conservative',
            'selected_value': sorted_pairs[0][0],
        }
        return selected

    def _conflict_to_dict(self, conflict: FactorConflict) -> Dict[str, Any]:
        """Convert conflict to dictionary."""
        return {
            'conflict_id': conflict.conflict_id,
            'matching_key': conflict.matching_key,
            'severity': conflict.severity.value,
            'percent_difference': conflict.percent_difference,
            'min_value': conflict.min_value,
            'max_value': conflict.max_value,
            'mean_value': conflict.mean_value,
            'sources_involved': conflict.sources_involved,
            'factor_count': len(conflict.factors),
            'resolution_strategy': conflict.resolution_strategy.value if conflict.resolution_strategy else None,
            'resolution_reason': conflict.resolution_reason,
            'detected_at': conflict.detected_at.isoformat(),
        }

    def _calculate_statistics(
        self,
        input_factors: List[Dict[str, Any]],
        conflicts: List[FactorConflict],
        resolved_factors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate reconciliation statistics."""
        # Source distribution
        source_counts = defaultdict(int)
        for factor in input_factors:
            source = self._get_source_name(factor)
            source_counts[source] += 1

        # Conflict severity distribution
        severity_counts = {s.value: 0 for s in ConflictSeverity}
        for conflict in conflicts:
            severity_counts[conflict.severity.value] += 1

        # Value statistics for resolved factors
        resolved_values = []
        for factor in resolved_factors:
            value = factor.get('factor_value')
            if value is not None:
                try:
                    resolved_values.append(float(value))
                except (ValueError, TypeError):
                    pass

        return {
            'source_distribution': dict(source_counts),
            'conflict_severity_distribution': severity_counts,
            'resolved_factors_count': len(resolved_factors),
            'value_statistics': {
                'min': round(min(resolved_values), 6) if resolved_values else None,
                'max': round(max(resolved_values), 6) if resolved_values else None,
                'mean': round(statistics.mean(resolved_values), 6) if resolved_values else None,
                'median': round(statistics.median(resolved_values), 6) if resolved_values else None,
            } if resolved_values else {},
        }

    def _audit(
        self,
        action: str,
        matching_key: str,
        factor: Optional[Dict[str, Any]],
        details: Optional[str] = None
    ) -> None:
        """Record audit trail entry."""
        if not self.enable_audit:
            return

        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'matching_key': matching_key,
            'factor_id': factor.get('factor_id') if factor else None,
            'source': self._get_source_name(factor) if factor else None,
            'details': details,
        }

        self._audit_trail.append(entry)

    def validate_cross_source(
        self,
        factors: List[Dict[str, Any]],
        tolerance_percent: float = 20.0
    ) -> Dict[str, Any]:
        """
        Validate factors across sources for consistency.

        Returns validation report with discrepancies.
        """
        groups = self._group_factors(factors)

        validations = []
        discrepancies = []

        for key, group in groups.items():
            if len(group) < 2:
                continue

            values = []
            for factor in group:
                value = factor.get('factor_value')
                if value is not None:
                    try:
                        values.append((float(value), self._get_source_name(factor)))
                    except (ValueError, TypeError):
                        pass

            if len(values) < 2:
                continue

            # Check consistency
            numeric_values = [v[0] for v in values]
            mean_val = statistics.mean(numeric_values)

            for value, source in values:
                if mean_val > 0:
                    diff_percent = abs((value - mean_val) / mean_val) * 100
                    if diff_percent > tolerance_percent:
                        discrepancies.append({
                            'matching_key': key,
                            'source': source,
                            'value': value,
                            'mean_value': round(mean_val, 6),
                            'difference_percent': round(diff_percent, 2),
                        })

            validations.append({
                'matching_key': key,
                'sources': [v[1] for v in values],
                'values': [v[0] for v in values],
                'mean': round(mean_val, 6),
                'is_consistent': len([
                    v for v in numeric_values
                    if abs((v - mean_val) / mean_val) * 100 <= tolerance_percent
                ]) == len(numeric_values) if mean_val > 0 else True,
            })

        return {
            'total_groups': len(groups),
            'multi_source_groups': len(validations),
            'discrepancies_found': len(discrepancies),
            'tolerance_percent': tolerance_percent,
            'discrepancies': discrepancies[:20],  # First 20
            'validations': validations[:20],
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_cbam_reconciler() -> FactorReconciler:
    """Create reconciler configured for CBAM requirements."""
    return FactorReconciler(
        matching_key_template='cbam',
        default_resolution_strategy=ConflictResolutionStrategy.HIGHEST_QUALITY,
        conflict_threshold_percent=10.0,
        enable_audit=True,
    )


def create_strict_reconciler() -> FactorReconciler:
    """Create reconciler with strict matching."""
    return FactorReconciler(
        matching_key_template='strict',
        default_resolution_strategy=ConflictResolutionStrategy.HIGHEST_PRIORITY,
        conflict_threshold_percent=5.0,
        enable_audit=True,
    )


def create_loose_reconciler() -> FactorReconciler:
    """Create reconciler with loose matching for broader deduplication."""
    return FactorReconciler(
        matching_key_template='loose',
        default_resolution_strategy=ConflictResolutionStrategy.WEIGHTED_AVERAGE,
        conflict_threshold_percent=15.0,
        enable_audit=True,
    )
