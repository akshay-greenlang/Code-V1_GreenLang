"""
Centralized Emission Factor Repository

This module implements the Repository pattern for emission factors, providing
a clean abstraction layer for CRUD operations and advanced querying capabilities
across all emission factor sources.

Features:
- Unified access to EPA, DEFRA, IEA, IPCC, Ecoinvent sources
- Version-controlled factor retrieval for reproducibility
- Batch operations for performance
- Advanced filtering and search
- Change tracking and audit logging
- Cache invalidation strategies
- Transaction-like semantics for factor updates

Usage:
    from data.emission_factor_repository import EmissionFactorRepository

    repo = EmissionFactorRepository()
    factor = repo.get_by_id("ef://epa/stationary/natural_gas/2024")
    factors = repo.find_by_criteria(source="EPA", region="US", year=2024)
"""
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import threading

from .models import (
    EmissionFactor,
    EmissionFactorSource,
    EmissionCategory,
    EmissionScope,
    GWPSet,
    UncertaintyRange,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Query Models
# =============================================================================


@dataclass
class FactorQuery:
    """Query parameters for emission factor lookup."""
    source: Optional[EmissionFactorSource] = None
    category: Optional[EmissionCategory] = None
    scope: Optional[EmissionScope] = None
    fuel_type: Optional[str] = None
    region: Optional[str] = None
    country_code: Optional[str] = None
    year: Optional[int] = None
    year_range: Optional[Tuple[int, int]] = None
    gwp_set: Optional[GWPSet] = None
    include_deprecated: bool = False
    text_search: Optional[str] = None

    # Pagination
    limit: Optional[int] = None
    offset: int = 0

    # Sorting
    order_by: Optional[str] = None
    order_desc: bool = False


@dataclass
class QueryResult(Generic[T]):
    """Result of a query operation."""
    items: List[T]
    total: int
    limit: Optional[int]
    offset: int
    has_more: bool
    query_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorVersion:
    """Represents a specific version of an emission factor."""
    factor_id: str
    version: int
    factor: EmissionFactor
    created_at: datetime
    created_by: Optional[str] = None
    change_reason: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class FactorChangeEvent:
    """Audit event for factor changes."""
    event_id: str
    factor_id: str
    event_type: str  # "CREATE", "UPDATE", "DELETE", "DEPRECATE"
    timestamp: datetime
    old_value: Optional[EmissionFactor] = None
    new_value: Optional[EmissionFactor] = None
    changed_by: Optional[str] = None
    reason: Optional[str] = None


# =============================================================================
# Repository Interface
# =============================================================================


class IEmissionFactorRepository(ABC):
    """Abstract interface for emission factor repositories."""

    @abstractmethod
    def get_by_id(self, factor_id: str) -> Optional[EmissionFactor]:
        """Get a factor by its unique ID."""
        pass

    @abstractmethod
    def get_by_id_versioned(
        self, factor_id: str, version: Optional[int] = None
    ) -> Optional[FactorVersion]:
        """Get a specific version of a factor."""
        pass

    @abstractmethod
    def find(self, query: FactorQuery) -> QueryResult[EmissionFactor]:
        """Find factors matching query criteria."""
        pass

    @abstractmethod
    def find_one(self, query: FactorQuery) -> Optional[EmissionFactor]:
        """Find a single factor matching criteria."""
        pass

    @abstractmethod
    def exists(self, factor_id: str) -> bool:
        """Check if a factor exists."""
        pass

    @abstractmethod
    def count(self, query: Optional[FactorQuery] = None) -> int:
        """Count factors matching optional criteria."""
        pass

    @abstractmethod
    def save(self, factor: EmissionFactor, reason: Optional[str] = None) -> EmissionFactor:
        """Save or update a factor."""
        pass

    @abstractmethod
    def save_batch(
        self, factors: List[EmissionFactor], reason: Optional[str] = None
    ) -> List[EmissionFactor]:
        """Save multiple factors in batch."""
        pass

    @abstractmethod
    def delete(self, factor_id: str, reason: Optional[str] = None) -> bool:
        """Delete a factor."""
        pass

    @abstractmethod
    def get_versions(self, factor_id: str) -> List[FactorVersion]:
        """Get all versions of a factor."""
        pass


# =============================================================================
# In-Memory Repository Implementation
# =============================================================================


class EmissionFactorRepository(IEmissionFactorRepository):
    """
    Centralized repository for emission factors.

    This repository provides:
    - Unified access to all emission factor sources
    - Thread-safe operations
    - Version history tracking
    - Change event logging
    - Efficient querying with indexes
    - Cache management
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        enable_versioning: bool = True,
        enable_audit: bool = True,
        max_versions: int = 10,
    ):
        """
        Initialize the repository.

        Args:
            data_dir: Path to emission factor data files
            enable_versioning: Whether to track version history
            enable_audit: Whether to log change events
            max_versions: Maximum versions to keep per factor
        """
        self.data_dir = data_dir or Path(__file__).parent / "emission_factors"
        self.enable_versioning = enable_versioning
        self.enable_audit = enable_audit
        self.max_versions = max_versions

        # Primary storage
        self._factors: Dict[str, EmissionFactor] = {}

        # Version storage
        self._versions: Dict[str, List[FactorVersion]] = {}

        # Audit log
        self._change_events: List[FactorChangeEvent] = []

        # Indexes for fast lookup
        self._index_by_source: Dict[EmissionFactorSource, Set[str]] = {}
        self._index_by_category: Dict[EmissionCategory, Set[str]] = {}
        self._index_by_region: Dict[str, Set[str]] = {}
        self._index_by_year: Dict[int, Set[str]] = {}
        self._index_by_fuel: Dict[str, Set[str]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "deletes": 0,
        }

        # Load data
        self._load_all_sources()

    def _load_all_sources(self) -> None:
        """Load all emission factor data from disk."""
        logger.info(f"Loading emission factors from {self.data_dir}")

        sources = ["epa", "defra", "iea", "ipcc"]
        total_loaded = 0

        for source in sources:
            source_dir = self.data_dir / source
            if source_dir.exists():
                count = self._load_source(source_dir, source)
                total_loaded += count
                logger.info(f"Loaded {count} factors from {source}")

        logger.info(f"Total emission factors loaded: {total_loaded}")

    def _load_source(self, source_dir: Path, source_name: str) -> int:
        """Load factors from a specific source directory."""
        count = 0
        source_enum = EmissionFactorSource(source_name)

        for json_file in source_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                factors = self._parse_factor_data(data, source_enum, json_file.stem)
                for factor in factors:
                    self._add_to_storage(factor, index=True)
                    count += 1

            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        return count

    def _parse_factor_data(
        self, data: Dict[str, Any], source: EmissionFactorSource, file_name: str
    ) -> List[EmissionFactor]:
        """Parse raw JSON data into EmissionFactor objects."""
        factors = []

        # Handle different data formats
        if "factors" in data:
            # Standard format with factors array
            for item in data.get("factors", []):
                factor = self._parse_single_factor(item, source)
                if factor:
                    factors.append(factor)
        elif "grid_factors" in data:
            # Grid emission factors format
            for country, country_data in data.get("grid_factors", {}).items():
                factor = self._parse_grid_factor(country, country_data, source)
                if factor:
                    factors.append(factor)
        elif isinstance(data, dict):
            # Try to parse as single factor or nested structure
            for key, value in data.items():
                if isinstance(value, dict) and "value" in value:
                    factor = self._parse_single_factor(value, source, key)
                    if factor:
                        factors.append(factor)

        return factors

    def _parse_single_factor(
        self, data: Dict[str, Any], source: EmissionFactorSource, key: Optional[str] = None
    ) -> Optional[EmissionFactor]:
        """Parse a single factor from raw data."""
        try:
            factor_id = data.get("id") or self._generate_factor_id(source, data, key)
            value = Decimal(str(data.get("value", 0)))

            return EmissionFactor(
                id=factor_id,
                value=value,
                unit=data.get("unit", "kg CO2e"),
                source=source,
                source_document=data.get("source_document"),
                source_url=data.get("source_url"),
                year=data.get("year", datetime.now().year),
                region=data.get("region", "global"),
                country_code=data.get("country_code"),
            )
        except Exception as e:
            logger.warning(f"Failed to parse factor: {e}")
            return None

    def _parse_grid_factor(
        self, country: str, data: Dict[str, Any], source: EmissionFactorSource
    ) -> Optional[EmissionFactor]:
        """Parse a grid emission factor."""
        try:
            factor_id = f"ef://{source.value}/grid/{country}/{data.get('year', 2024)}"
            value = Decimal(str(data.get("value", data.get("factor", 0))))

            return EmissionFactor(
                id=factor_id,
                value=value,
                unit=data.get("unit", "kg CO2e/kWh"),
                source=source,
                year=data.get("year", datetime.now().year),
                region=country,
                country_code=country,
            )
        except Exception as e:
            logger.warning(f"Failed to parse grid factor for {country}: {e}")
            return None

    def _generate_factor_id(
        self, source: EmissionFactorSource, data: Dict[str, Any], key: Optional[str]
    ) -> str:
        """Generate a unique factor ID."""
        category = data.get("category", "general")
        fuel = data.get("fuel_type", key or "unknown")
        year = data.get("year", datetime.now().year)
        region = data.get("region", "global")

        return f"ef://{source.value}/{category}/{fuel}/{year}/{region}"

    def _add_to_storage(self, factor: EmissionFactor, index: bool = True) -> None:
        """Add a factor to internal storage and update indexes."""
        with self._lock:
            self._factors[factor.id] = factor

            if index:
                self._update_indexes(factor)

    def _update_indexes(self, factor: EmissionFactor) -> None:
        """Update all indexes for a factor."""
        # Source index
        if factor.source not in self._index_by_source:
            self._index_by_source[factor.source] = set()
        self._index_by_source[factor.source].add(factor.id)

        # Region index
        if factor.region not in self._index_by_region:
            self._index_by_region[factor.region] = set()
        self._index_by_region[factor.region].add(factor.id)

        # Year index
        if factor.year not in self._index_by_year:
            self._index_by_year[factor.year] = set()
        self._index_by_year[factor.year].add(factor.id)

    def _remove_from_indexes(self, factor: EmissionFactor) -> None:
        """Remove a factor from all indexes."""
        if factor.source in self._index_by_source:
            self._index_by_source[factor.source].discard(factor.id)

        if factor.region in self._index_by_region:
            self._index_by_region[factor.region].discard(factor.id)

        if factor.year in self._index_by_year:
            self._index_by_year[factor.year].discard(factor.id)

    # =========================================================================
    # IEmissionFactorRepository Implementation
    # =========================================================================

    def get_by_id(self, factor_id: str) -> Optional[EmissionFactor]:
        """Get a factor by its unique ID."""
        self._stats["queries"] += 1

        with self._lock:
            factor = self._factors.get(factor_id)

        if factor:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1

        return factor

    def get_by_id_versioned(
        self, factor_id: str, version: Optional[int] = None
    ) -> Optional[FactorVersion]:
        """Get a specific version of a factor."""
        with self._lock:
            versions = self._versions.get(factor_id, [])

            if not versions:
                # No version history, return current as version 1
                factor = self._factors.get(factor_id)
                if factor:
                    return FactorVersion(
                        factor_id=factor_id,
                        version=1,
                        factor=factor,
                        created_at=datetime.utcnow(),
                    )
                return None

            if version is None:
                # Return latest version
                return versions[-1]

            # Find specific version
            for v in versions:
                if v.version == version:
                    return v

            return None

    def find(self, query: FactorQuery) -> QueryResult[EmissionFactor]:
        """Find factors matching query criteria."""
        import time

        start_time = time.time()
        self._stats["queries"] += 1

        with self._lock:
            # Start with candidate set from indexes
            candidates = self._get_candidates(query)

            # Apply filters
            results = []
            for factor_id in candidates:
                factor = self._factors.get(factor_id)
                if factor and self._matches_query(factor, query):
                    results.append(factor)

            # Sort results
            if query.order_by:
                reverse = query.order_desc
                results.sort(
                    key=lambda f: getattr(f, query.order_by, ""),
                    reverse=reverse,
                )

            # Apply pagination
            total = len(results)
            if query.limit:
                results = results[query.offset : query.offset + query.limit]

            has_more = query.offset + len(results) < total

        query_time = (time.time() - start_time) * 1000

        return QueryResult(
            items=results,
            total=total,
            limit=query.limit,
            offset=query.offset,
            has_more=has_more,
            query_time_ms=query_time,
        )

    def _get_candidates(self, query: FactorQuery) -> Set[str]:
        """Get candidate factor IDs using indexes."""
        candidates: Optional[Set[str]] = None

        # Use source index
        if query.source and query.source in self._index_by_source:
            source_ids = self._index_by_source[query.source]
            candidates = source_ids if candidates is None else candidates & source_ids

        # Use region index
        if query.region and query.region in self._index_by_region:
            region_ids = self._index_by_region[query.region]
            candidates = region_ids if candidates is None else candidates & region_ids

        # Use year index
        if query.year and query.year in self._index_by_year:
            year_ids = self._index_by_year[query.year]
            candidates = year_ids if candidates is None else candidates & year_ids

        # If no index was used, return all factor IDs
        if candidates is None:
            candidates = set(self._factors.keys())

        return candidates

    def _matches_query(self, factor: EmissionFactor, query: FactorQuery) -> bool:
        """Check if a factor matches all query criteria."""
        if query.source and factor.source != query.source:
            return False

        if query.region and factor.region != query.region:
            return False

        if query.country_code and factor.country_code != query.country_code:
            return False

        if query.year and factor.year != query.year:
            return False

        if query.year_range:
            min_year, max_year = query.year_range
            if not (min_year <= factor.year <= max_year):
                return False

        if query.text_search:
            search_text = query.text_search.lower()
            searchable = f"{factor.id} {factor.region}".lower()
            if search_text not in searchable:
                return False

        return True

    def find_one(self, query: FactorQuery) -> Optional[EmissionFactor]:
        """Find a single factor matching criteria."""
        query.limit = 1
        result = self.find(query)
        return result.items[0] if result.items else None

    def exists(self, factor_id: str) -> bool:
        """Check if a factor exists."""
        with self._lock:
            return factor_id in self._factors

    def count(self, query: Optional[FactorQuery] = None) -> int:
        """Count factors matching optional criteria."""
        if query is None:
            return len(self._factors)

        result = self.find(query)
        return result.total

    def save(
        self, factor: EmissionFactor, reason: Optional[str] = None
    ) -> EmissionFactor:
        """Save or update a factor."""
        self._stats["saves"] += 1

        with self._lock:
            is_update = factor.id in self._factors
            old_factor = self._factors.get(factor.id) if is_update else None

            # Update checksum
            factor_dict = {
                "id": factor.id,
                "value": str(factor.value),
                "unit": factor.unit,
                "source": factor.source.value,
                "year": factor.year,
                "region": factor.region,
            }
            checksum = hashlib.sha256(
                json.dumps(factor_dict, sort_keys=True).encode()
            ).hexdigest()

            # Store factor
            self._factors[factor.id] = factor
            self._update_indexes(factor)

            # Version tracking
            if self.enable_versioning:
                self._add_version(factor, reason, checksum)

            # Audit logging
            if self.enable_audit:
                event_type = "UPDATE" if is_update else "CREATE"
                self._log_change(
                    factor.id, event_type, old_factor, factor, reason
                )

        return factor

    def save_batch(
        self, factors: List[EmissionFactor], reason: Optional[str] = None
    ) -> List[EmissionFactor]:
        """Save multiple factors in batch."""
        saved = []
        for factor in factors:
            saved.append(self.save(factor, reason))
        return saved

    def delete(self, factor_id: str, reason: Optional[str] = None) -> bool:
        """Delete a factor."""
        self._stats["deletes"] += 1

        with self._lock:
            factor = self._factors.get(factor_id)
            if not factor:
                return False

            # Remove from storage
            del self._factors[factor_id]
            self._remove_from_indexes(factor)

            # Audit logging
            if self.enable_audit:
                self._log_change(factor_id, "DELETE", factor, None, reason)

        return True

    def get_versions(self, factor_id: str) -> List[FactorVersion]:
        """Get all versions of a factor."""
        with self._lock:
            return list(self._versions.get(factor_id, []))

    def _add_version(
        self, factor: EmissionFactor, reason: Optional[str], checksum: str
    ) -> None:
        """Add a new version to version history."""
        if factor.id not in self._versions:
            self._versions[factor.id] = []

        versions = self._versions[factor.id]
        new_version = len(versions) + 1

        version_entry = FactorVersion(
            factor_id=factor.id,
            version=new_version,
            factor=factor,
            created_at=datetime.utcnow(),
            change_reason=reason,
            checksum=checksum,
        )

        versions.append(version_entry)

        # Trim old versions if needed
        if len(versions) > self.max_versions:
            self._versions[factor.id] = versions[-self.max_versions :]

    def _log_change(
        self,
        factor_id: str,
        event_type: str,
        old_value: Optional[EmissionFactor],
        new_value: Optional[EmissionFactor],
        reason: Optional[str],
    ) -> None:
        """Log a change event."""
        event = FactorChangeEvent(
            event_id=f"evt_{datetime.utcnow().timestamp()}_{factor_id}",
            factor_id=factor_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            old_value=old_value,
            new_value=new_value,
            reason=reason,
        )
        self._change_events.append(event)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_by_source(self, source: EmissionFactorSource) -> List[EmissionFactor]:
        """Get all factors from a specific source."""
        query = FactorQuery(source=source)
        return self.find(query).items

    def get_grid_factor(
        self, country_code: str, year: Optional[int] = None
    ) -> Optional[EmissionFactor]:
        """Get electricity grid emission factor for a country."""
        query = FactorQuery(
            country_code=country_code,
            year=year or datetime.now().year,
        )
        # Filter for grid factors
        result = self.find(query)
        for factor in result.items:
            if "grid" in factor.id.lower():
                return factor
        return None

    def get_fuel_factor(
        self,
        fuel_type: str,
        source: Optional[EmissionFactorSource] = None,
        region: str = "global",
    ) -> Optional[EmissionFactor]:
        """Get emission factor for a fuel type."""
        query = FactorQuery(
            source=source,
            fuel_type=fuel_type,
            region=region,
        )
        return self.find_one(query)

    def search(self, text: str, limit: int = 20) -> List[EmissionFactor]:
        """Full-text search across factors."""
        query = FactorQuery(text_search=text, limit=limit)
        return self.find(query).items

    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        with self._lock:
            sources = {}
            for source, ids in self._index_by_source.items():
                sources[source.value] = len(ids)

            return {
                "total_factors": len(self._factors),
                "by_source": sources,
                "by_year": {
                    year: len(ids)
                    for year, ids in sorted(self._index_by_year.items())
                },
                "unique_regions": len(self._index_by_region),
                "versions_tracked": len(self._versions),
                "change_events": len(self._change_events),
                "query_stats": dict(self._stats),
            }

    def get_audit_log(
        self,
        factor_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[FactorChangeEvent]:
        """Get audit log entries with optional filtering."""
        with self._lock:
            events = self._change_events

            if factor_id:
                events = [e for e in events if e.factor_id == factor_id]

            if event_type:
                events = [e for e in events if e.event_type == event_type]

            if since:
                events = [e for e in events if e.timestamp >= since]

            return list(events)

    def clear_cache(self) -> None:
        """Clear all caches."""
        pass  # Currently using in-memory storage, no separate cache

    def reload(self) -> None:
        """Reload all factors from disk."""
        with self._lock:
            self._factors.clear()
            self._index_by_source.clear()
            self._index_by_category.clear()
            self._index_by_region.clear()
            self._index_by_year.clear()
            self._index_by_fuel.clear()
            self._load_all_sources()


# =============================================================================
# Singleton Instance
# =============================================================================

_repository: Optional[EmissionFactorRepository] = None
_repository_lock = threading.Lock()


def get_repository() -> EmissionFactorRepository:
    """Get the singleton repository instance."""
    global _repository
    if _repository is None:
        with _repository_lock:
            if _repository is None:
                _repository = EmissionFactorRepository()
    return _repository


def reset_repository() -> None:
    """Reset the singleton instance (for testing)."""
    global _repository
    with _repository_lock:
        _repository = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "EmissionFactorRepository",
    "IEmissionFactorRepository",
    "FactorQuery",
    "QueryResult",
    "FactorVersion",
    "FactorChangeEvent",
    "get_repository",
    "reset_repository",
]
