"""
Data Catalog
============

Comprehensive data catalog for emission factor management.
Provides metadata management, source lineage tracking, version history,
and search/discovery capabilities.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
import logging
import hashlib
import json
from pathlib import Path
from collections import defaultdict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

class LineageType(str, Enum):
    """Types of lineage relationships."""
    DERIVED_FROM = "derived_from"  # Factor derived from another
    AGGREGATED_FROM = "aggregated_from"  # Factor aggregated from multiple
    UPDATED_FROM = "updated_from"  # New version of existing factor
    IMPORTED_FROM = "imported_from"  # Imported from external source
    CALCULATED_FROM = "calculated_from"  # Calculated using other factors
    RECONCILED_FROM = "reconciled_from"  # Result of reconciliation


class FactorStatus(str, Enum):
    """Factor lifecycle status."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"


@dataclass
class SourceLineage:
    """Source lineage information for an emission factor."""
    lineage_id: str
    factor_id: str
    lineage_type: LineageType
    source_factor_ids: List[str]  # IDs of source factors
    source_system: str  # Original data source system
    source_table: Optional[str] = None
    source_query: Optional[str] = None  # Query/filter used
    transformation: Optional[str] = None  # Description of transformation
    transformation_code: Optional[str] = None  # Actual code/SQL
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'lineage_id': self.lineage_id,
            'factor_id': self.factor_id,
            'lineage_type': self.lineage_type.value,
            'source_factor_ids': self.source_factor_ids,
            'source_system': self.source_system,
            'source_table': self.source_table,
            'transformation': self.transformation,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
        }


@dataclass
class VersionHistory:
    """Version history entry for an emission factor."""
    version_id: str
    factor_id: str
    version_number: str  # Semantic version
    previous_version_id: Optional[str] = None
    status: FactorStatus = FactorStatus.ACTIVE
    change_type: str = "update"  # create, update, deprecate, etc.
    change_summary: str = ""
    change_details: Dict[str, Any] = field(default_factory=dict)
    changed_fields: List[str] = field(default_factory=list)
    old_values: Dict[str, Any] = field(default_factory=dict)
    new_values: Dict[str, Any] = field(default_factory=dict)
    effective_from: date = field(default_factory=date.today)
    effective_to: Optional[date] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'version_id': self.version_id,
            'factor_id': self.factor_id,
            'version_number': self.version_number,
            'previous_version_id': self.previous_version_id,
            'status': self.status.value,
            'change_type': self.change_type,
            'change_summary': self.change_summary,
            'changed_fields': self.changed_fields,
            'effective_from': self.effective_from.isoformat(),
            'effective_to': self.effective_to.isoformat() if self.effective_to else None,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
        }


class CatalogEntry(BaseModel):
    """Catalog entry for an emission factor."""
    factor_id: str = Field(..., description="Unique factor identifier")
    factor_hash: str = Field(..., description="Content hash for deduplication")

    # Classification
    industry: str
    product_code: Optional[str] = None
    product_name: str
    product_category: Optional[str] = None
    product_subcategory: Optional[str] = None

    # Geographic
    region: str
    country_code: Optional[str] = None
    geographic_scope: str = "national"  # facility, local, regional, national, global

    # Factor details
    ghg_type: str = "CO2e"
    scope_type: str
    factor_value: float
    factor_unit: str
    reference_year: int

    # Source information
    source_type: str
    source_name: str
    source_url: Optional[str] = None
    publication_date: Optional[date] = None

    # Quality
    quality_tier: str = "tier_1"
    dqi_score: Optional[float] = None

    # Regulatory
    cbam_eligible: bool = False
    csrd_compliant: bool = False
    ghg_protocol_compliant: bool = True

    # Lifecycle
    status: FactorStatus = FactorStatus.ACTIVE
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    valid_from: date = Field(default_factory=date.today)
    valid_to: Optional[date] = None

    # Search metadata
    tags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    description: Optional[str] = None

    # Relationships
    supersedes_id: Optional[str] = None  # ID of factor this supersedes
    superseded_by_id: Optional[str] = None  # ID of factor that supersedes this

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
        }


class CatalogSearchResult(BaseModel):
    """Search result from catalog."""
    total_results: int
    page: int = 1
    page_size: int = 20
    total_pages: int
    results: List[CatalogEntry]
    facets: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    search_time_ms: float = 0.0


class FactorLookupResult(BaseModel):
    """Result of factor lookup."""
    found: bool
    factor: Optional[CatalogEntry] = None
    alternatives: List[CatalogEntry] = Field(default_factory=list)
    match_type: str = "exact"  # exact, partial, fallback
    match_confidence: float = 1.0
    lookup_path: List[str] = Field(default_factory=list)  # Search path taken


# =============================================================================
# DATA CATALOG
# =============================================================================

class DataCatalog:
    """
    Emission Factor Data Catalog.

    Provides:
    - Factor metadata management
    - Source lineage tracking
    - Version history
    - Search and discovery
    - Hierarchical factor lookup
    - API for factor retrieval
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 3600,
    ):
        """
        Initialize data catalog.

        Args:
            storage_path: Path for persisting catalog data
            enable_caching: Enable in-memory caching
            cache_ttl_seconds: Cache time-to-live
        """
        self.storage_path = Path(storage_path) if storage_path else None

        # In-memory storage
        self.entries: Dict[str, CatalogEntry] = {}
        self.lineage: Dict[str, List[SourceLineage]] = defaultdict(list)
        self.versions: Dict[str, List[VersionHistory]] = defaultdict(list)

        # Indexes for fast lookup
        self._index_by_hash: Dict[str, str] = {}
        self._index_by_product: Dict[str, Set[str]] = defaultdict(set)
        self._index_by_region: Dict[str, Set[str]] = defaultdict(set)
        self._index_by_industry: Dict[str, Set[str]] = defaultdict(set)
        self._index_by_source: Dict[str, Set[str]] = defaultdict(set)
        self._index_by_tag: Dict[str, Set[str]] = defaultdict(set)

        # Caching
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Tuple[Any, datetime]] = {}

        if self.storage_path and self.storage_path.exists():
            self._load_catalog()

    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================

    def add_factor(
        self,
        factor: Union[CatalogEntry, Dict[str, Any]],
        lineage: Optional[SourceLineage] = None,
        created_by: Optional[str] = None,
    ) -> CatalogEntry:
        """
        Add a factor to the catalog.

        Args:
            factor: Factor data
            lineage: Optional lineage information
            created_by: User/system creating the entry

        Returns:
            Created CatalogEntry
        """
        if isinstance(factor, dict):
            factor = CatalogEntry(**factor)

        # Check for duplicates by hash
        if factor.factor_hash in self._index_by_hash:
            existing_id = self._index_by_hash[factor.factor_hash]
            logger.warning(f"Duplicate factor detected: {existing_id}")
            return self.entries[existing_id]

        # Store entry
        self.entries[factor.factor_id] = factor
        self._update_indexes(factor)

        # Add initial version history
        version = VersionHistory(
            version_id=self._generate_id(f"{factor.factor_id}:v1"),
            factor_id=factor.factor_id,
            version_number=factor.version,
            status=factor.status,
            change_type="create",
            change_summary="Initial factor creation",
            created_by=created_by,
        )
        self.versions[factor.factor_id].append(version)

        # Add lineage if provided
        if lineage:
            lineage.factor_id = factor.factor_id
            self.lineage[factor.factor_id].append(lineage)

        logger.debug(f"Added factor to catalog: {factor.factor_id}")
        self._save_catalog()

        return factor

    def update_factor(
        self,
        factor_id: str,
        updates: Dict[str, Any],
        change_summary: str = "",
        updated_by: Optional[str] = None,
    ) -> CatalogEntry:
        """
        Update a factor in the catalog.

        Args:
            factor_id: Factor to update
            updates: Fields to update
            change_summary: Summary of changes
            updated_by: User/system updating

        Returns:
            Updated CatalogEntry
        """
        if factor_id not in self.entries:
            raise ValueError(f"Factor not found: {factor_id}")

        old_entry = self.entries[factor_id]
        old_values = {}
        new_values = {}
        changed_fields = []

        # Track changes
        for field, new_value in updates.items():
            if hasattr(old_entry, field):
                old_value = getattr(old_entry, field)
                if old_value != new_value:
                    old_values[field] = old_value
                    new_values[field] = new_value
                    changed_fields.append(field)

        if not changed_fields:
            return old_entry

        # Create new entry with updates
        entry_dict = old_entry.dict()
        entry_dict.update(updates)
        entry_dict['updated_at'] = datetime.utcnow()
        entry_dict['version'] = self._increment_version(old_entry.version)

        new_entry = CatalogEntry(**entry_dict)

        # Update storage and indexes
        self._remove_from_indexes(old_entry)
        self.entries[factor_id] = new_entry
        self._update_indexes(new_entry)

        # Add version history
        version = VersionHistory(
            version_id=self._generate_id(f"{factor_id}:{new_entry.version}"),
            factor_id=factor_id,
            version_number=new_entry.version,
            previous_version_id=self.versions[factor_id][-1].version_id if self.versions[factor_id] else None,
            status=new_entry.status,
            change_type="update",
            change_summary=change_summary,
            changed_fields=changed_fields,
            old_values=old_values,
            new_values=new_values,
            created_by=updated_by,
        )
        self.versions[factor_id].append(version)

        logger.debug(f"Updated factor: {factor_id}, changed: {changed_fields}")
        self._save_catalog()

        return new_entry

    def deprecate_factor(
        self,
        factor_id: str,
        reason: str,
        superseded_by: Optional[str] = None,
        deprecated_by: Optional[str] = None,
    ) -> CatalogEntry:
        """
        Deprecate a factor.

        Args:
            factor_id: Factor to deprecate
            reason: Reason for deprecation
            superseded_by: ID of replacing factor
            deprecated_by: User/system deprecating

        Returns:
            Updated CatalogEntry
        """
        updates = {
            'status': FactorStatus.DEPRECATED,
            'valid_to': date.today(),
        }

        if superseded_by:
            updates['superseded_by_id'] = superseded_by
            # Update the new factor to reference this one
            if superseded_by in self.entries:
                self.update_factor(
                    superseded_by,
                    {'supersedes_id': factor_id},
                    f"Supersedes deprecated factor {factor_id}"
                )

        return self.update_factor(
            factor_id,
            updates,
            change_summary=f"Deprecated: {reason}",
            updated_by=deprecated_by,
        )

    def delete_factor(self, factor_id: str) -> bool:
        """
        Delete a factor from the catalog (soft delete - archives).

        Args:
            factor_id: Factor to delete

        Returns:
            Success status
        """
        if factor_id not in self.entries:
            return False

        return self.update_factor(
            factor_id,
            {'status': FactorStatus.ARCHIVED},
            change_summary="Archived (deleted)"
        ) is not None

    def get_factor(self, factor_id: str) -> Optional[CatalogEntry]:
        """Get a factor by ID."""
        return self.entries.get(factor_id)

    def get_factors(self, factor_ids: List[str]) -> List[CatalogEntry]:
        """Get multiple factors by IDs."""
        return [
            self.entries[fid]
            for fid in factor_ids
            if fid in self.entries
        ]

    # =========================================================================
    # SEARCH AND DISCOVERY
    # =========================================================================

    def search(
        self,
        query: Optional[str] = None,
        industry: Optional[str] = None,
        region: Optional[str] = None,
        country_code: Optional[str] = None,
        product_code: Optional[str] = None,
        source_type: Optional[str] = None,
        scope_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: FactorStatus = FactorStatus.ACTIVE,
        cbam_only: bool = False,
        csrd_only: bool = False,
        min_dqi: Optional[float] = None,
        reference_year: Optional[int] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> CatalogSearchResult:
        """
        Search the catalog with filters.

        Args:
            query: Text search query
            industry: Filter by industry
            region: Filter by region
            country_code: Filter by country
            product_code: Filter by product code
            source_type: Filter by source type
            scope_type: Filter by GHG scope
            tags: Filter by tags
            status: Filter by status
            cbam_only: Only CBAM-eligible factors
            csrd_only: Only CSRD-compliant factors
            min_dqi: Minimum DQI score
            reference_year: Filter by reference year
            page: Page number
            page_size: Results per page

        Returns:
            CatalogSearchResult with matching factors
        """
        start_time = datetime.utcnow()

        # Start with all entries or use index
        candidate_ids: Set[str] = set(self.entries.keys())

        # Apply index-based filters
        if industry:
            candidate_ids &= self._index_by_industry.get(industry.lower(), set())
        if region:
            candidate_ids &= self._index_by_region.get(region.lower(), set())
        if source_type:
            candidate_ids &= self._index_by_source.get(source_type.lower(), set())
        if product_code:
            candidate_ids &= self._index_by_product.get(product_code.lower(), set())
        if tags:
            for tag in tags:
                candidate_ids &= self._index_by_tag.get(tag.lower(), set())

        # Get candidates
        candidates = [self.entries[fid] for fid in candidate_ids if fid in self.entries]

        # Apply remaining filters
        results = []
        for entry in candidates:
            # Status filter
            if entry.status != status:
                continue

            # Country filter
            if country_code and entry.country_code != country_code:
                continue

            # Scope filter
            if scope_type and entry.scope_type != scope_type:
                continue

            # Regulatory filters
            if cbam_only and not entry.cbam_eligible:
                continue
            if csrd_only and not entry.csrd_compliant:
                continue

            # Quality filter
            if min_dqi and (entry.dqi_score is None or entry.dqi_score < min_dqi):
                continue

            # Year filter
            if reference_year and entry.reference_year != reference_year:
                continue

            # Text search
            if query:
                query_lower = query.lower()
                searchable = ' '.join([
                    entry.product_name or '',
                    entry.product_code or '',
                    entry.description or '',
                    ' '.join(entry.tags),
                    ' '.join(entry.keywords),
                ]).lower()

                if query_lower not in searchable:
                    continue

            results.append(entry)

        # Calculate facets
        facets = self._calculate_facets(results)

        # Pagination
        total_results = len(results)
        total_pages = (total_results + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_results = results[start_idx:end_idx]

        search_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return CatalogSearchResult(
            total_results=total_results,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            results=paginated_results,
            facets=facets,
            search_time_ms=round(search_time, 2),
        )

    def _calculate_facets(self, results: List[CatalogEntry]) -> Dict[str, Dict[str, int]]:
        """Calculate facets from search results."""
        facets = {
            'industry': defaultdict(int),
            'region': defaultdict(int),
            'source_type': defaultdict(int),
            'scope_type': defaultdict(int),
            'quality_tier': defaultdict(int),
            'reference_year': defaultdict(int),
        }

        for entry in results:
            facets['industry'][entry.industry] += 1
            facets['region'][entry.region] += 1
            facets['source_type'][entry.source_type] += 1
            facets['scope_type'][entry.scope_type] += 1
            facets['quality_tier'][entry.quality_tier] += 1
            facets['reference_year'][str(entry.reference_year)] += 1

        return {k: dict(v) for k, v in facets.items()}

    # =========================================================================
    # FACTOR LOOKUP API
    # =========================================================================

    def lookup_factor(
        self,
        product_code: Optional[str] = None,
        product_name: Optional[str] = None,
        region: Optional[str] = None,
        country_code: Optional[str] = None,
        scope_type: str = "scope_1",
        reference_year: Optional[int] = None,
        fallback_to_global: bool = True,
        fallback_to_regional: bool = True,
    ) -> FactorLookupResult:
        """
        Look up the best matching emission factor.

        Implements hierarchical lookup:
        1. Exact match (product + country + scope + year)
        2. Regional fallback (product + region + scope)
        3. Global fallback (product + global + scope)
        4. Industry average fallback

        Args:
            product_code: Product code (CN, HS, etc.)
            product_name: Product name (if no code)
            region: Target region
            country_code: Target country
            scope_type: GHG Protocol scope
            reference_year: Preferred reference year
            fallback_to_global: Allow global fallback
            fallback_to_regional: Allow regional fallback

        Returns:
            FactorLookupResult with best match
        """
        lookup_path = []
        year = reference_year or datetime.now().year

        # Try exact country match
        if country_code:
            lookup_path.append(f"country:{country_code}")
            result = self._find_factor(
                product_code=product_code,
                product_name=product_name,
                country_code=country_code,
                scope_type=scope_type,
                reference_year=year,
            )
            if result:
                return FactorLookupResult(
                    found=True,
                    factor=result,
                    match_type="exact_country",
                    match_confidence=1.0,
                    lookup_path=lookup_path,
                )

        # Try regional fallback
        if fallback_to_regional and region:
            lookup_path.append(f"region:{region}")
            result = self._find_factor(
                product_code=product_code,
                product_name=product_name,
                region=region,
                scope_type=scope_type,
                reference_year=year,
            )
            if result:
                return FactorLookupResult(
                    found=True,
                    factor=result,
                    match_type="regional",
                    match_confidence=0.8,
                    lookup_path=lookup_path,
                )

        # Try global fallback
        if fallback_to_global:
            lookup_path.append("region:global")
            result = self._find_factor(
                product_code=product_code,
                product_name=product_name,
                region="global",
                scope_type=scope_type,
                reference_year=year,
            )
            if result:
                return FactorLookupResult(
                    found=True,
                    factor=result,
                    match_type="global",
                    match_confidence=0.6,
                    lookup_path=lookup_path,
                )

        # Find alternatives
        alternatives = self._find_alternatives(
            product_code=product_code,
            product_name=product_name,
            scope_type=scope_type,
        )

        return FactorLookupResult(
            found=False,
            factor=None,
            alternatives=alternatives[:5],
            match_type="not_found",
            match_confidence=0.0,
            lookup_path=lookup_path,
        )

    def _find_factor(
        self,
        product_code: Optional[str] = None,
        product_name: Optional[str] = None,
        country_code: Optional[str] = None,
        region: Optional[str] = None,
        scope_type: Optional[str] = None,
        reference_year: Optional[int] = None,
    ) -> Optional[CatalogEntry]:
        """Find a specific factor."""
        candidates = []

        for entry in self.entries.values():
            if entry.status != FactorStatus.ACTIVE:
                continue

            # Match product
            if product_code:
                if entry.product_code != product_code:
                    continue
            elif product_name:
                if product_name.lower() not in entry.product_name.lower():
                    continue
            else:
                continue

            # Match geography
            if country_code and entry.country_code != country_code:
                continue
            if region and entry.region.lower() != region.lower():
                continue

            # Match scope
            if scope_type and entry.scope_type != scope_type:
                continue

            candidates.append(entry)

        if not candidates:
            return None

        # Sort by quality and recency
        candidates.sort(
            key=lambda e: (
                -(e.dqi_score or 0),  # Higher DQI first
                -e.reference_year,     # More recent first
            )
        )

        # Filter by year if specified
        if reference_year:
            year_matches = [c for c in candidates if c.reference_year == reference_year]
            if year_matches:
                return year_matches[0]

        return candidates[0]

    def _find_alternatives(
        self,
        product_code: Optional[str] = None,
        product_name: Optional[str] = None,
        scope_type: Optional[str] = None,
    ) -> List[CatalogEntry]:
        """Find alternative factors."""
        alternatives = []

        for entry in self.entries.values():
            if entry.status != FactorStatus.ACTIVE:
                continue

            # Partial product match
            if product_name:
                words = product_name.lower().split()
                if any(word in entry.product_name.lower() for word in words):
                    alternatives.append(entry)
                    continue

            # Same scope
            if scope_type and entry.scope_type == scope_type:
                alternatives.append(entry)

        # Sort by DQI
        alternatives.sort(key=lambda e: -(e.dqi_score or 0))

        return alternatives

    # =========================================================================
    # LINEAGE AND VERSION TRACKING
    # =========================================================================

    def add_lineage(
        self,
        factor_id: str,
        lineage_type: LineageType,
        source_factor_ids: List[str],
        source_system: str,
        transformation: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> SourceLineage:
        """Add lineage information for a factor."""
        if factor_id not in self.entries:
            raise ValueError(f"Factor not found: {factor_id}")

        lineage = SourceLineage(
            lineage_id=self._generate_id(f"lineage:{factor_id}:{len(self.lineage[factor_id])}"),
            factor_id=factor_id,
            lineage_type=lineage_type,
            source_factor_ids=source_factor_ids,
            source_system=source_system,
            transformation=transformation,
            created_by=created_by,
        )

        self.lineage[factor_id].append(lineage)
        self._save_catalog()

        return lineage

    def get_lineage(self, factor_id: str) -> List[SourceLineage]:
        """Get lineage history for a factor."""
        return self.lineage.get(factor_id, [])

    def get_version_history(self, factor_id: str) -> List[VersionHistory]:
        """Get version history for a factor."""
        return self.versions.get(factor_id, [])

    def get_factor_at_version(
        self,
        factor_id: str,
        version: str
    ) -> Optional[Dict[str, Any]]:
        """Reconstruct factor at a specific version."""
        versions = self.versions.get(factor_id, [])

        # Find target version
        target_idx = None
        for idx, v in enumerate(versions):
            if v.version_number == version:
                target_idx = idx
                break

        if target_idx is None:
            return None

        # Start with current and apply inverse changes
        current = self.entries.get(factor_id)
        if not current:
            return None

        factor_dict = current.dict()

        # Apply inverse changes from newest to target
        for v in reversed(versions[target_idx + 1:]):
            for field, old_value in v.old_values.items():
                factor_dict[field] = old_value

        return factor_dict

    # =========================================================================
    # INDEX MANAGEMENT
    # =========================================================================

    def _update_indexes(self, entry: CatalogEntry) -> None:
        """Update search indexes for an entry."""
        fid = entry.factor_id

        self._index_by_hash[entry.factor_hash] = fid

        if entry.product_code:
            self._index_by_product[entry.product_code.lower()].add(fid)

        self._index_by_region[entry.region.lower()].add(fid)
        self._index_by_industry[entry.industry.lower()].add(fid)
        self._index_by_source[entry.source_type.lower()].add(fid)

        for tag in entry.tags:
            self._index_by_tag[tag.lower()].add(fid)

    def _remove_from_indexes(self, entry: CatalogEntry) -> None:
        """Remove entry from search indexes."""
        fid = entry.factor_id

        if entry.factor_hash in self._index_by_hash:
            del self._index_by_hash[entry.factor_hash]

        if entry.product_code:
            self._index_by_product[entry.product_code.lower()].discard(fid)

        self._index_by_region[entry.region.lower()].discard(fid)
        self._index_by_industry[entry.industry.lower()].discard(fid)
        self._index_by_source[entry.source_type.lower()].discard(fid)

        for tag in entry.tags:
            self._index_by_tag[tag.lower()].discard(fid)

    def rebuild_indexes(self) -> None:
        """Rebuild all search indexes."""
        self._index_by_hash.clear()
        self._index_by_product.clear()
        self._index_by_region.clear()
        self._index_by_industry.clear()
        self._index_by_source.clear()
        self._index_by_tag.clear()

        for entry in self.entries.values():
            self._update_indexes(entry)

        logger.info(f"Rebuilt indexes for {len(self.entries)} entries")

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _save_catalog(self) -> None:
        """Persist catalog to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Save entries
        entries_file = self.storage_path / "entries.json"
        with open(entries_file, 'w') as f:
            json.dump(
                {fid: e.dict() for fid, e in self.entries.items()},
                f,
                indent=2,
                default=str
            )

        # Save lineage
        lineage_file = self.storage_path / "lineage.json"
        with open(lineage_file, 'w') as f:
            json.dump(
                {fid: [l.to_dict() for l in lin] for fid, lin in self.lineage.items()},
                f,
                indent=2,
                default=str
            )

        # Save versions
        versions_file = self.storage_path / "versions.json"
        with open(versions_file, 'w') as f:
            json.dump(
                {fid: [v.to_dict() for v in vers] for fid, vers in self.versions.items()},
                f,
                indent=2,
                default=str
            )

    def _load_catalog(self) -> None:
        """Load catalog from storage."""
        if not self.storage_path:
            return

        # Load entries
        entries_file = self.storage_path / "entries.json"
        if entries_file.exists():
            with open(entries_file, 'r') as f:
                data = json.load(f)
                for fid, entry_dict in data.items():
                    self.entries[fid] = CatalogEntry(**entry_dict)

        self.rebuild_indexes()
        logger.info(f"Loaded {len(self.entries)} catalog entries")

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _generate_id(self, content: str) -> str:
        """Generate ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _increment_version(self, version: str) -> str:
        """Increment semantic version."""
        parts = version.split('.')
        if len(parts) == 3:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{major}.{minor}.{patch + 1}"
        return f"{version}.1"

    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        active_count = sum(1 for e in self.entries.values() if e.status == FactorStatus.ACTIVE)

        industries = defaultdict(int)
        regions = defaultdict(int)
        sources = defaultdict(int)

        for entry in self.entries.values():
            if entry.status == FactorStatus.ACTIVE:
                industries[entry.industry] += 1
                regions[entry.region] += 1
                sources[entry.source_type] += 1

        return {
            'total_entries': len(self.entries),
            'active_entries': active_count,
            'deprecated_entries': sum(1 for e in self.entries.values() if e.status == FactorStatus.DEPRECATED),
            'archived_entries': sum(1 for e in self.entries.values() if e.status == FactorStatus.ARCHIVED),
            'by_industry': dict(industries),
            'by_region': dict(regions),
            'by_source': dict(sources),
            'total_lineage_records': sum(len(l) for l in self.lineage.values()),
            'total_version_records': sum(len(v) for v in self.versions.values()),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_catalog(storage_path: str = None) -> DataCatalog:
    """Create a new data catalog."""
    return DataCatalog(storage_path=storage_path)


def load_catalog_from_factors(
    factors: List[Dict[str, Any]],
    storage_path: str = None
) -> DataCatalog:
    """Create and populate catalog from factor list."""
    catalog = DataCatalog(storage_path=storage_path)

    for factor in factors:
        entry = CatalogEntry(
            factor_id=factor.get('factor_id'),
            factor_hash=factor.get('factor_hash', ''),
            industry=factor.get('industry', 'general'),
            product_code=factor.get('product_code'),
            product_name=factor.get('product_name', ''),
            region=factor.get('region', 'global'),
            country_code=factor.get('country_code'),
            ghg_type=factor.get('ghg_type', 'CO2e'),
            scope_type=factor.get('scope_type', 'scope_1'),
            factor_value=float(factor.get('factor_value', 0)),
            factor_unit=factor.get('factor_unit', ''),
            reference_year=factor.get('reference_year', datetime.now().year),
            source_type=factor.get('source', {}).get('source_type', 'unknown'),
            source_name=factor.get('source', {}).get('source_name', ''),
            quality_tier=factor.get('quality', {}).get('quality_tier', 'tier_1'),
            dqi_score=factor.get('quality', {}).get('aggregate_dqi'),
            cbam_eligible=factor.get('cbam_eligible', False),
            csrd_compliant=factor.get('csrd_compliant', False),
            tags=factor.get('tags', []),
        )

        lineage = SourceLineage(
            lineage_id=catalog._generate_id(f"import:{entry.factor_id}"),
            factor_id=entry.factor_id,
            lineage_type=LineageType.IMPORTED_FROM,
            source_factor_ids=[],
            source_system=entry.source_type,
        )

        catalog.add_factor(entry, lineage=lineage)

    return catalog
