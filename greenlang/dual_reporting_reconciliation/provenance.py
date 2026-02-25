"""
Dual Reporting Reconciliation Agent - Provenance Tracking Module

This module provides SHA-256 provenance chain tracking for GL-MRV-X-024 (AGENT-MRV-013).
Thread-safe singleton pattern ensures audit trail integrity across all reconciliation operations.

Agent: GL-MRV-X-024 (AGENT-MRV-013)
Purpose: Track location-based vs market-based dual reporting reconciliation lineage
Regulatory: GHG Protocol Scope 2 Guidance, CDP Climate Change, RE100, SBTi
"""

import hashlib
import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal

class ProvenanceStage(str, Enum):
    """17 provenance stages for dual reporting reconciliation."""

    COLLECT_LOCATION_RESULTS = "collect_location_results"
    COLLECT_MARKET_RESULTS = "collect_market_results"
    ALIGN_BOUNDARIES = "align_boundaries"
    MAP_ENERGY_TYPES = "map_energy_types"
    CALCULATE_TOTALS = "calculate_totals"
    ANALYZE_DISCREPANCIES = "analyze_discrepancies"
    CLASSIFY_MATERIALITY = "classify_materiality"
    WATERFALL_DECOMPOSITION = "waterfall_decomposition"
    SCORE_COMPLETENESS = "score_completeness"
    SCORE_CONSISTENCY = "score_consistency"
    SCORE_ACCURACY = "score_accuracy"
    SCORE_TRANSPARENCY = "score_transparency"
    GENERATE_TABLES = "generate_tables"
    ANALYZE_TRENDS = "analyze_trends"
    CHECK_COMPLIANCE = "check_compliance"
    ASSEMBLE_REPORT = "assemble_report"
    SEAL_PROVENANCE = "seal_provenance"


@dataclass
class ProvenanceEntry:
    """Single provenance chain entry."""

    stage: ProvenanceStage
    timestamp: datetime
    input_hash: str
    output_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_hash: Optional[str] = None
    chain_hash: Optional[str] = None


@dataclass
class ProvenanceChain:
    """Complete provenance chain for reconciliation."""

    reconciliation_id: str
    organization_id: str
    reporting_period: str
    created_at: datetime
    entries: List[ProvenanceEntry] = field(default_factory=list)
    sealed: bool = False
    seal_hash: Optional[str] = None
    seal_timestamp: Optional[datetime] = None


class DualReportingProvenanceTracker:
    """
    Thread-safe singleton provenance tracker for dual reporting reconciliation.

    Tracks 17-stage reconciliation lineage with SHA-256 chain integrity.
    Supports multi-period batch reconciliation with per-period chains.

    Thread Safety:
        - Singleton with double-checked locking
        - Per-chain locks for concurrent operations
        - Atomic chain sealing

    Example:
        >>> tracker = DualReportingProvenanceTracker.get_instance()
        >>> chain_id = tracker.create_chain("REC-001", "ORG-123", "2024-Q1")
        >>> tracker.add_stage(
        ...     chain_id,
        ...     ProvenanceStage.COLLECT_LOCATION_RESULTS,
        ...     {"upstream_count": 5},
        ...     location_results
        ... )
        >>> tracker.seal_chain(chain_id)
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize provenance tracker (use get_instance instead)."""
        if DualReportingProvenanceTracker._instance is not None:
            raise RuntimeError("Use get_instance() to access singleton")

        self._chains: Dict[str, ProvenanceChain] = {}
        self._chain_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "DualReportingProvenanceTracker":
        """Get singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def create_chain(
        self,
        reconciliation_id: str,
        organization_id: str,
        reporting_period: str
    ) -> str:
        """
        Create new provenance chain.

        Args:
            reconciliation_id: Unique reconciliation identifier
            organization_id: Organization identifier
            reporting_period: Reporting period (YYYY-QN or YYYY)

        Returns:
            Chain ID (same as reconciliation_id)

        Raises:
            ValueError: If chain already exists
        """
        with self._global_lock:
            if reconciliation_id in self._chains:
                raise ValueError(f"Chain {reconciliation_id} already exists")

            chain = ProvenanceChain(
                reconciliation_id=reconciliation_id,
                organization_id=organization_id,
                reporting_period=reporting_period,
                created_at=datetime.utcnow()
            )

            self._chains[reconciliation_id] = chain
            self._chain_locks[reconciliation_id] = threading.Lock()

        return reconciliation_id

    def add_stage(
        self,
        chain_id: str,
        stage: ProvenanceStage,
        metadata: Dict[str, Any],
        output_data: Any
    ) -> str:
        """
        Add stage to provenance chain.

        Args:
            chain_id: Chain identifier
            stage: Provenance stage
            metadata: Stage metadata
            output_data: Output data to hash

        Returns:
            Chain hash after adding stage

        Raises:
            ValueError: If chain not found or sealed
        """
        if chain_id not in self._chains:
            raise ValueError(f"Chain {chain_id} not found")

        with self._chain_locks[chain_id]:
            chain = self._chains[chain_id]

            if chain.sealed:
                raise ValueError(f"Chain {chain_id} is sealed")

            # Get previous hash
            previous_hash = None
            if chain.entries:
                previous_hash = chain.entries[-1].chain_hash

            # Calculate input hash (previous chain hash + stage + metadata)
            input_data = {
                "previous_hash": previous_hash,
                "stage": stage.value,
                "metadata": metadata
            }
            input_hash = self._hash_dict(input_data)

            # Calculate output hash
            output_hash = self._hash_data(output_data)

            # Calculate chain hash (input + output)
            chain_data = {
                "input_hash": input_hash,
                "output_hash": output_hash
            }
            chain_hash = self._hash_dict(chain_data)

            # Create entry
            entry = ProvenanceEntry(
                stage=stage,
                timestamp=datetime.utcnow(),
                input_hash=input_hash,
                output_hash=output_hash,
                metadata=metadata,
                previous_hash=previous_hash,
                chain_hash=chain_hash
            )

            chain.entries.append(entry)

        return chain_hash

    def seal_chain(self, chain_id: str) -> str:
        """
        Seal provenance chain (immutable after sealing).

        Args:
            chain_id: Chain identifier

        Returns:
            Final seal hash

        Raises:
            ValueError: If chain not found or already sealed
        """
        if chain_id not in self._chains:
            raise ValueError(f"Chain {chain_id} not found")

        with self._chain_locks[chain_id]:
            chain = self._chains[chain_id]

            if chain.sealed:
                raise ValueError(f"Chain {chain_id} already sealed")

            if not chain.entries:
                raise ValueError(f"Cannot seal empty chain {chain_id}")

            # Calculate seal hash (all entry hashes + metadata)
            seal_data = {
                "reconciliation_id": chain.reconciliation_id,
                "organization_id": chain.organization_id,
                "reporting_period": chain.reporting_period,
                "created_at": chain.created_at.isoformat(),
                "entry_count": len(chain.entries),
                "entry_hashes": [e.chain_hash for e in chain.entries]
            }

            seal_hash = self._hash_dict(seal_data)

            chain.sealed = True
            chain.seal_hash = seal_hash
            chain.seal_timestamp = datetime.utcnow()

        return seal_hash

    def verify_chain(self, chain_id: str) -> Tuple[bool, Optional[str]]:
        """
        Verify provenance chain integrity.

        Args:
            chain_id: Chain identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        if chain_id not in self._chains:
            return False, f"Chain {chain_id} not found"

        with self._chain_locks[chain_id]:
            chain = self._chains[chain_id]

            if not chain.entries:
                return False, "Chain has no entries"

            # Verify each entry's chain hash
            previous_hash = None
            for i, entry in enumerate(chain.entries):
                # Verify previous hash linkage
                if entry.previous_hash != previous_hash:
                    return False, f"Entry {i} previous hash mismatch"

                # Recalculate input hash
                input_data = {
                    "previous_hash": previous_hash,
                    "stage": entry.stage.value,
                    "metadata": entry.metadata
                }
                expected_input_hash = self._hash_dict(input_data)

                if entry.input_hash != expected_input_hash:
                    return False, f"Entry {i} input hash mismatch"

                # Recalculate chain hash
                chain_data = {
                    "input_hash": entry.input_hash,
                    "output_hash": entry.output_hash
                }
                expected_chain_hash = self._hash_dict(chain_data)

                if entry.chain_hash != expected_chain_hash:
                    return False, f"Entry {i} chain hash mismatch"

                previous_hash = entry.chain_hash

            # Verify seal hash if sealed
            if chain.sealed:
                seal_data = {
                    "reconciliation_id": chain.reconciliation_id,
                    "organization_id": chain.organization_id,
                    "reporting_period": chain.reporting_period,
                    "created_at": chain.created_at.isoformat(),
                    "entry_count": len(chain.entries),
                    "entry_hashes": [e.chain_hash for e in chain.entries]
                }
                expected_seal_hash = self._hash_dict(seal_data)

                if chain.seal_hash != expected_seal_hash:
                    return False, "Seal hash mismatch"

        return True, None

    def get_chain(self, chain_id: str) -> Optional[ProvenanceChain]:
        """Get provenance chain (returns copy for thread safety)."""
        if chain_id not in self._chains:
            return None

        with self._chain_locks[chain_id]:
            chain = self._chains[chain_id]
            # Return deep copy to prevent external modification
            return ProvenanceChain(
                reconciliation_id=chain.reconciliation_id,
                organization_id=chain.organization_id,
                reporting_period=chain.reporting_period,
                created_at=chain.created_at,
                entries=list(chain.entries),
                sealed=chain.sealed,
                seal_hash=chain.seal_hash,
                seal_timestamp=chain.seal_timestamp
            )

    def export_chain(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Export provenance chain to JSON-serializable dict."""
        chain = self.get_chain(chain_id)
        if not chain:
            return None

        return {
            "reconciliation_id": chain.reconciliation_id,
            "organization_id": chain.organization_id,
            "reporting_period": chain.reporting_period,
            "created_at": chain.created_at.isoformat(),
            "sealed": chain.sealed,
            "seal_hash": chain.seal_hash,
            "seal_timestamp": chain.seal_timestamp.isoformat() if chain.seal_timestamp else None,
            "entries": [
                {
                    "stage": entry.stage.value,
                    "timestamp": entry.timestamp.isoformat(),
                    "input_hash": entry.input_hash,
                    "output_hash": entry.output_hash,
                    "metadata": entry.metadata,
                    "previous_hash": entry.previous_hash,
                    "chain_hash": entry.chain_hash
                }
                for entry in chain.entries
            ]
        }

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Hash dictionary using SHA-256."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _hash_data(self, data: Any) -> str:
        """Hash arbitrary data using SHA-256."""
        if isinstance(data, dict):
            return self._hash_dict(data)
        elif isinstance(data, (list, tuple)):
            return self._hash_dict({"items": data})
        elif isinstance(data, Decimal):
            return hashlib.sha256(str(data).encode()).hexdigest()
        elif isinstance(data, datetime):
            return hashlib.sha256(data.isoformat().encode()).hexdigest()
        else:
            return hashlib.sha256(str(data).encode()).hexdigest()


# ============================================================================
# Hash Helper Functions
# ============================================================================


def hash_upstream_result(
    agent: str,
    method: str,
    energy_type: str,
    emissions: Decimal,
    ef_used: Optional[Decimal],
    provenance_hash: str
) -> str:
    """
    Hash upstream calculation result.

    Args:
        agent: Agent identifier (e.g., "GL-MRV-007")
        method: Calculation method (e.g., "location_based")
        energy_type: Energy type (e.g., "electricity_grid")
        emissions: Calculated emissions (tCO2e)
        ef_used: Emission factor used (if applicable)
        provenance_hash: Upstream provenance hash

    Returns:
        SHA-256 hash of upstream result
    """
    data = {
        "agent": agent,
        "method": method,
        "energy_type": energy_type,
        "emissions": str(emissions),
        "ef_used": str(ef_used) if ef_used else None,
        "provenance_hash": provenance_hash
    }
    return _hash_helper(data)


def hash_discrepancy(
    type: str,
    direction: str,
    materiality: str,
    absolute: Decimal,
    percentage: Decimal
) -> str:
    """
    Hash discrepancy analysis result.

    Args:
        type: Discrepancy type (e.g., "total_difference")
        direction: Direction (e.g., "location_higher")
        materiality: Materiality classification (e.g., "MATERIAL")
        absolute: Absolute difference (tCO2e)
        percentage: Percentage difference (%)

    Returns:
        SHA-256 hash of discrepancy
    """
    data = {
        "type": type,
        "direction": direction,
        "materiality": materiality,
        "absolute": str(absolute),
        "percentage": str(percentage)
    }
    return _hash_helper(data)


def hash_quality_assessment(
    composite_score: Decimal,
    grade: str,
    dimension_scores: Dict[str, Decimal]
) -> str:
    """
    Hash quality assessment result.

    Args:
        composite_score: Composite quality score (0-100)
        grade: Quality grade (A+/A/B/C/D/F)
        dimension_scores: Per-dimension scores

    Returns:
        SHA-256 hash of quality assessment
    """
    data = {
        "composite_score": str(composite_score),
        "grade": grade,
        "dimension_scores": {k: str(v) for k, v in dimension_scores.items()}
    }
    return _hash_helper(data)


def hash_framework_table(
    framework: str,
    row_count: int,
    footnote_count: int
) -> str:
    """
    Hash reporting framework table.

    Args:
        framework: Framework identifier (e.g., "ghg_protocol")
        row_count: Number of table rows
        footnote_count: Number of footnotes

    Returns:
        SHA-256 hash of framework table
    """
    data = {
        "framework": framework,
        "row_count": row_count,
        "footnote_count": footnote_count
    }
    return _hash_helper(data)


def hash_trend_point(
    period: str,
    location: Decimal,
    market: Decimal,
    pif: Decimal
) -> str:
    """
    Hash trend analysis point.

    Args:
        period: Period identifier (e.g., "2024-Q1")
        location: Location-based emissions (tCO2e)
        market: Market-based emissions (tCO2e)
        pif: Purchased instrument factor

    Returns:
        SHA-256 hash of trend point
    """
    data = {
        "period": period,
        "location": str(location),
        "market": str(market),
        "pif": str(pif)
    }
    return _hash_helper(data)


def hash_compliance_result(
    framework: str,
    status: str,
    met: int,
    total: int
) -> str:
    """
    Hash compliance check result.

    Args:
        framework: Framework identifier (e.g., "ghg_protocol")
        status: Compliance status (PASS/FAIL/PARTIAL)
        met: Number of requirements met
        total: Total number of requirements

    Returns:
        SHA-256 hash of compliance result
    """
    data = {
        "framework": framework,
        "status": status,
        "met": met,
        "total": total
    }
    return _hash_helper(data)


def hash_reconciliation_report(
    reconciliation_id: str,
    location_total: Decimal,
    market_total: Decimal,
    pif: Decimal,
    quality_grade: str
) -> str:
    """
    Hash reconciliation report.

    Args:
        reconciliation_id: Reconciliation identifier
        location_total: Total location-based emissions (tCO2e)
        market_total: Total market-based emissions (tCO2e)
        pif: Purchased instrument factor
        quality_grade: Quality grade (A+/A/B/C/D/F)

    Returns:
        SHA-256 hash of reconciliation report
    """
    data = {
        "reconciliation_id": reconciliation_id,
        "location_total": str(location_total),
        "market_total": str(market_total),
        "pif": str(pif),
        "quality_grade": quality_grade
    }
    return _hash_helper(data)


def hash_waterfall(
    total_discrepancy: Decimal,
    item_count: int
) -> str:
    """
    Hash waterfall decomposition.

    Args:
        total_discrepancy: Total discrepancy (tCO2e)
        item_count: Number of waterfall items

    Returns:
        SHA-256 hash of waterfall
    """
    data = {
        "total_discrepancy": str(total_discrepancy),
        "item_count": item_count
    }
    return _hash_helper(data)


def hash_batch_result(
    batch_id: str,
    total_periods: int,
    completed: int,
    failed: int
) -> str:
    """
    Hash batch reconciliation result.

    Args:
        batch_id: Batch identifier
        total_periods: Total number of periods
        completed: Number of completed periods
        failed: Number of failed periods

    Returns:
        SHA-256 hash of batch result
    """
    data = {
        "batch_id": batch_id,
        "total_periods": total_periods,
        "completed": completed,
        "failed": failed
    }
    return _hash_helper(data)


def hash_boundary_alignment(
    location_count: int,
    market_count: int,
    matched: int,
    unmatched_location: int,
    unmatched_market: int
) -> str:
    """
    Hash boundary alignment result.

    Args:
        location_count: Count of location-based results
        market_count: Count of market-based results
        matched: Count of matched results
        unmatched_location: Count of unmatched location results
        unmatched_market: Count of unmatched market results

    Returns:
        SHA-256 hash of boundary alignment
    """
    data = {
        "location_count": location_count,
        "market_count": market_count,
        "matched": matched,
        "unmatched_location": unmatched_location,
        "unmatched_market": unmatched_market
    }
    return _hash_helper(data)


def hash_energy_mapping(
    source_type: str,
    target_type: str,
    mapping_confidence: Decimal
) -> str:
    """
    Hash energy type mapping.

    Args:
        source_type: Source energy type
        target_type: Target energy type
        mapping_confidence: Mapping confidence (0-1)

    Returns:
        SHA-256 hash of energy mapping
    """
    data = {
        "source_type": source_type,
        "target_type": target_type,
        "mapping_confidence": str(mapping_confidence)
    }
    return _hash_helper(data)


def hash_completeness_score(
    location_coverage: Decimal,
    market_coverage: Decimal,
    documentation_coverage: Decimal,
    composite_score: Decimal
) -> str:
    """
    Hash completeness dimension score.

    Args:
        location_coverage: Location-based coverage (%)
        market_coverage: Market-based coverage (%)
        documentation_coverage: Documentation coverage (%)
        composite_score: Composite completeness score (0-100)

    Returns:
        SHA-256 hash of completeness score
    """
    data = {
        "location_coverage": str(location_coverage),
        "market_coverage": str(market_coverage),
        "documentation_coverage": str(documentation_coverage),
        "composite_score": str(composite_score)
    }
    return _hash_helper(data)


def hash_consistency_score(
    boundary_alignment: Decimal,
    temporal_consistency: Decimal,
    methodology_consistency: Decimal,
    composite_score: Decimal
) -> str:
    """
    Hash consistency dimension score.

    Args:
        boundary_alignment: Boundary alignment score (0-100)
        temporal_consistency: Temporal consistency score (0-100)
        methodology_consistency: Methodology consistency score (0-100)
        composite_score: Composite consistency score (0-100)

    Returns:
        SHA-256 hash of consistency score
    """
    data = {
        "boundary_alignment": str(boundary_alignment),
        "temporal_consistency": str(temporal_consistency),
        "methodology_consistency": str(methodology_consistency),
        "composite_score": str(composite_score)
    }
    return _hash_helper(data)


def hash_accuracy_score(
    ef_quality: Decimal,
    calculation_precision: Decimal,
    uncertainty_assessment: Decimal,
    composite_score: Decimal
) -> str:
    """
    Hash accuracy dimension score.

    Args:
        ef_quality: Emission factor quality score (0-100)
        calculation_precision: Calculation precision score (0-100)
        uncertainty_assessment: Uncertainty assessment score (0-100)
        composite_score: Composite accuracy score (0-100)

    Returns:
        SHA-256 hash of accuracy score
    """
    data = {
        "ef_quality": str(ef_quality),
        "calculation_precision": str(calculation_precision),
        "uncertainty_assessment": str(uncertainty_assessment),
        "composite_score": str(composite_score)
    }
    return _hash_helper(data)


def hash_transparency_score(
    methodology_disclosure: Decimal,
    assumption_disclosure: Decimal,
    limitation_disclosure: Decimal,
    composite_score: Decimal
) -> str:
    """
    Hash transparency dimension score.

    Args:
        methodology_disclosure: Methodology disclosure score (0-100)
        assumption_disclosure: Assumption disclosure score (0-100)
        limitation_disclosure: Limitation disclosure score (0-100)
        composite_score: Composite transparency score (0-100)

    Returns:
        SHA-256 hash of transparency score
    """
    data = {
        "methodology_disclosure": str(methodology_disclosure),
        "assumption_disclosure": str(assumption_disclosure),
        "limitation_disclosure": str(limitation_disclosure),
        "composite_score": str(composite_score)
    }
    return _hash_helper(data)


def hash_materiality_classification(
    discrepancy_percentage: Decimal,
    absolute_discrepancy: Decimal,
    threshold: Decimal,
    classification: str
) -> str:
    """
    Hash materiality classification.

    Args:
        discrepancy_percentage: Discrepancy percentage (%)
        absolute_discrepancy: Absolute discrepancy (tCO2e)
        threshold: Materiality threshold (%)
        classification: Classification (MATERIAL/IMMATERIAL)

    Returns:
        SHA-256 hash of materiality classification
    """
    data = {
        "discrepancy_percentage": str(discrepancy_percentage),
        "absolute_discrepancy": str(absolute_discrepancy),
        "threshold": str(threshold),
        "classification": classification
    }
    return _hash_helper(data)


def hash_waterfall_item(
    item_name: str,
    contribution: Decimal,
    percentage_of_total: Decimal
) -> str:
    """
    Hash waterfall decomposition item.

    Args:
        item_name: Item name (e.g., "EAC_adjustments")
        contribution: Contribution to discrepancy (tCO2e)
        percentage_of_total: Percentage of total discrepancy (%)

    Returns:
        SHA-256 hash of waterfall item
    """
    data = {
        "item_name": item_name,
        "contribution": str(contribution),
        "percentage_of_total": str(percentage_of_total)
    }
    return _hash_helper(data)


def hash_trend_analysis(
    period_count: int,
    location_trend: str,
    market_trend: str,
    pif_trend: str
) -> str:
    """
    Hash trend analysis result.

    Args:
        period_count: Number of periods analyzed
        location_trend: Location-based trend (INCREASING/DECREASING/STABLE)
        market_trend: Market-based trend (INCREASING/DECREASING/STABLE)
        pif_trend: PIF trend (INCREASING/DECREASING/STABLE)

    Returns:
        SHA-256 hash of trend analysis
    """
    data = {
        "period_count": period_count,
        "location_trend": location_trend,
        "market_trend": market_trend,
        "pif_trend": pif_trend
    }
    return _hash_helper(data)


def hash_compliance_framework(
    framework_id: str,
    framework_version: str,
    requirement_count: int
) -> str:
    """
    Hash compliance framework metadata.

    Args:
        framework_id: Framework identifier (e.g., "ghg_protocol")
        framework_version: Framework version (e.g., "2015")
        requirement_count: Number of requirements

    Returns:
        SHA-256 hash of compliance framework
    """
    data = {
        "framework_id": framework_id,
        "framework_version": framework_version,
        "requirement_count": requirement_count
    }
    return _hash_helper(data)


def hash_report_metadata(
    reconciliation_id: str,
    organization_id: str,
    reporting_period: str,
    generated_at: datetime
) -> str:
    """
    Hash report metadata.

    Args:
        reconciliation_id: Reconciliation identifier
        organization_id: Organization identifier
        reporting_period: Reporting period
        generated_at: Report generation timestamp

    Returns:
        SHA-256 hash of report metadata
    """
    data = {
        "reconciliation_id": reconciliation_id,
        "organization_id": organization_id,
        "reporting_period": reporting_period,
        "generated_at": generated_at.isoformat()
    }
    return _hash_helper(data)


def _hash_helper(data: Dict[str, Any]) -> str:
    """Internal helper for SHA-256 hashing."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


# ============================================================================
# Batch Provenance Tracking
# ============================================================================


class BatchProvenanceTracker:
    """
    Track provenance for multi-period batch reconciliation.

    Manages per-period chains with batch-level aggregation.
    """

    def __init__(self):
        """Initialize batch provenance tracker."""
        self.tracker = DualReportingProvenanceTracker.get_instance()
        self._batch_chains: Dict[str, List[str]] = {}
        self._batch_lock = threading.Lock()

    def create_batch(self, batch_id: str) -> str:
        """
        Create batch provenance container.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch ID
        """
        with self._batch_lock:
            if batch_id in self._batch_chains:
                raise ValueError(f"Batch {batch_id} already exists")

            self._batch_chains[batch_id] = []

        return batch_id

    def add_period_chain(self, batch_id: str, chain_id: str) -> None:
        """
        Add period chain to batch.

        Args:
            batch_id: Batch identifier
            chain_id: Period chain identifier
        """
        with self._batch_lock:
            if batch_id not in self._batch_chains:
                raise ValueError(f"Batch {batch_id} not found")

            self._batch_chains[batch_id].append(chain_id)

    def seal_batch(self, batch_id: str) -> str:
        """
        Seal batch provenance (seals all period chains).

        Args:
            batch_id: Batch identifier

        Returns:
            Batch seal hash
        """
        with self._batch_lock:
            if batch_id not in self._batch_chains:
                raise ValueError(f"Batch {batch_id} not found")

            chain_ids = self._batch_chains[batch_id]

            # Seal all period chains
            seal_hashes = []
            for chain_id in chain_ids:
                try:
                    seal_hash = self.tracker.seal_chain(chain_id)
                    seal_hashes.append(seal_hash)
                except ValueError:
                    # Chain already sealed
                    chain = self.tracker.get_chain(chain_id)
                    if chain and chain.seal_hash:
                        seal_hashes.append(chain.seal_hash)

            # Calculate batch seal hash
            batch_data = {
                "batch_id": batch_id,
                "period_count": len(chain_ids),
                "period_seal_hashes": seal_hashes
            }

            batch_seal_hash = _hash_helper(batch_data)

        return batch_seal_hash

    def verify_batch(self, batch_id: str) -> Tuple[bool, Optional[str]]:
        """
        Verify batch provenance integrity.

        Args:
            batch_id: Batch identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        with self._batch_lock:
            if batch_id not in self._batch_chains:
                return False, f"Batch {batch_id} not found"

            chain_ids = self._batch_chains[batch_id]

            # Verify each period chain
            for chain_id in chain_ids:
                is_valid, error = self.tracker.verify_chain(chain_id)
                if not is_valid:
                    return False, f"Period {chain_id}: {error}"

        return True, None

    def get_batch_summary(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get batch provenance summary.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch summary dict or None if not found
        """
        with self._batch_lock:
            if batch_id not in self._batch_chains:
                return None

            chain_ids = self._batch_chains[batch_id]

            # Collect period summaries
            period_summaries = []
            for chain_id in chain_ids:
                chain = self.tracker.get_chain(chain_id)
                if chain:
                    period_summaries.append({
                        "reconciliation_id": chain.reconciliation_id,
                        "reporting_period": chain.reporting_period,
                        "stage_count": len(chain.entries),
                        "sealed": chain.sealed,
                        "seal_hash": chain.seal_hash
                    })

            return {
                "batch_id": batch_id,
                "period_count": len(chain_ids),
                "periods": period_summaries
            }


# ============================================================================
# Provenance Query Utilities
# ============================================================================


def get_stage_hash(chain_id: str, stage: ProvenanceStage) -> Optional[str]:
    """Get hash for specific stage in chain."""
    tracker = DualReportingProvenanceTracker.get_instance()
    chain = tracker.get_chain(chain_id)

    if not chain:
        return None

    for entry in chain.entries:
        if entry.stage == stage:
            return entry.chain_hash

    return None


def get_stage_metadata(chain_id: str, stage: ProvenanceStage) -> Optional[Dict[str, Any]]:
    """Get metadata for specific stage in chain."""
    tracker = DualReportingProvenanceTracker.get_instance()
    chain = tracker.get_chain(chain_id)

    if not chain:
        return None

    for entry in chain.entries:
        if entry.stage == stage:
            return entry.metadata

    return None


def get_chain_summary(chain_id: str) -> Optional[Dict[str, Any]]:
    """Get chain summary with stage progression."""
    tracker = DualReportingProvenanceTracker.get_instance()
    chain = tracker.get_chain(chain_id)

    if not chain:
        return None

    return {
        "reconciliation_id": chain.reconciliation_id,
        "organization_id": chain.organization_id,
        "reporting_period": chain.reporting_period,
        "created_at": chain.created_at.isoformat(),
        "stage_count": len(chain.entries),
        "stages_completed": [entry.stage.value for entry in chain.entries],
        "sealed": chain.sealed,
        "seal_hash": chain.seal_hash,
        "seal_timestamp": chain.seal_timestamp.isoformat() if chain.seal_timestamp else None
    }


def verify_upstream_hash(
    chain_id: str,
    stage: ProvenanceStage,
    expected_hash: str
) -> bool:
    """Verify upstream hash matches chain entry."""
    actual_hash = get_stage_hash(chain_id, stage)
    return actual_hash == expected_hash if actual_hash else False


# ============================================================================
# Alias for backward compatibility
# ============================================================================

#: Alias used by the pipeline engine, setup.py, and __init__.py.
DualReportingReconciliationProvenance = DualReportingProvenanceTracker
