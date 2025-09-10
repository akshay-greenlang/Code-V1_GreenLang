"""
Pack Discovery and Index System for GreenLang Hub
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import httpx
from functools import lru_cache

from .client import HubClient
from .manifest import PackManifest

logger = logging.getLogger(__name__)


class PackCategory(Enum):
    """Standard pack categories"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    SECURITY = "security"
    DATA_PROCESSING = "data-processing"
    MACHINE_LEARNING = "ml-ai"
    WEB = "web"
    CLI = "cli"
    UTILITY = "utility"
    FRAMEWORK = "framework"
    LIBRARY = "library"
    TEMPLATE = "template"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class SortOrder(Enum):
    """Sort order for search results"""
    RELEVANCE = "relevance"
    DOWNLOADS = "downloads"
    STARS = "stars"
    UPDATED = "updated"
    CREATED = "created"
    NAME = "name"
    AUTHOR = "author"


@dataclass
class PackInfo:
    """Pack information for discovery"""
    
    # Basic info
    id: str
    name: str
    version: str
    description: str
    
    # Author info
    author: Dict[str, str]
    maintainers: List[Dict[str, str]] = field(default_factory=list)
    
    # Metadata
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Stats
    downloads: int = 0
    weekly_downloads: int = 0
    stars: int = 0
    watchers: int = 0
    forks: int = 0
    
    # Dates
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    
    # Links
    homepage: Optional[str] = None
    repository: Optional[str] = None
    documentation: Optional[str] = None
    issues: Optional[str] = None
    
    # Dependencies
    dependencies: List[Dict[str, str]] = field(default_factory=list)
    dev_dependencies: List[Dict[str, str]] = field(default_factory=list)
    
    # Compatibility
    greenlang_version: Optional[str] = None
    python_version: Optional[str] = None
    platforms: List[str] = field(default_factory=list)
    
    # Quality indicators
    verified: bool = False
    official: bool = False
    deprecated: bool = False
    featured: bool = False
    trending: bool = False
    
    # Security
    has_vulnerabilities: bool = False
    security_score: Optional[float] = None
    last_audit: Optional[datetime] = None
    
    # License
    license: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'maintainers': self.maintainers,
            'categories': self.categories,
            'tags': self.tags,
            'keywords': self.keywords,
            'downloads': self.downloads,
            'weekly_downloads': self.weekly_downloads,
            'stars': self.stars,
            'watchers': self.watchers,
            'forks': self.forks,
            'homepage': self.homepage,
            'repository': self.repository,
            'documentation': self.documentation,
            'issues': self.issues,
            'dependencies': self.dependencies,
            'dev_dependencies': self.dev_dependencies,
            'greenlang_version': self.greenlang_version,
            'python_version': self.python_version,
            'platforms': self.platforms,
            'verified': self.verified,
            'official': self.official,
            'deprecated': self.deprecated,
            'featured': self.featured,
            'trending': self.trending,
            'has_vulnerabilities': self.has_vulnerabilities,
            'security_score': self.security_score,
            'license': self.license
        }
        
        # Convert datetime objects
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        if self.published_at:
            data['published_at'] = self.published_at.isoformat()
        if self.last_audit:
            data['last_audit'] = self.last_audit.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PackInfo':
        """Create from dictionary"""
        # Parse datetime strings
        for date_field in ['created_at', 'updated_at', 'published_at', 'last_audit']:
            if date_field in data and data[date_field]:
                if isinstance(data[date_field], str):
                    data[date_field] = datetime.fromisoformat(data[date_field])
        
        return cls(**data)


@dataclass
class SearchFilters:
    """Search filters for pack discovery"""
    
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    author: Optional[str] = None
    license: Optional[str] = None
    platform: Optional[str] = None
    min_stars: Optional[int] = None
    min_downloads: Optional[int] = None
    verified_only: bool = False
    official_only: bool = False
    exclude_deprecated: bool = True
    has_vulnerabilities: Optional[bool] = None
    greenlang_version: Optional[str] = None
    python_version: Optional[str] = None
    
    def to_params(self) -> Dict[str, Any]:
        """Convert to query parameters"""
        params = {}
        
        if self.categories:
            params['categories'] = ','.join(self.categories)
        if self.tags:
            params['tags'] = ','.join(self.tags)
        if self.author:
            params['author'] = self.author
        if self.license:
            params['license'] = self.license
        if self.platform:
            params['platform'] = self.platform
        if self.min_stars is not None:
            params['min_stars'] = self.min_stars
        if self.min_downloads is not None:
            params['min_downloads'] = self.min_downloads
        if self.verified_only:
            params['verified'] = 'true'
        if self.official_only:
            params['official'] = 'true'
        if self.exclude_deprecated:
            params['exclude_deprecated'] = 'true'
        if self.has_vulnerabilities is not None:
            params['has_vulnerabilities'] = str(self.has_vulnerabilities).lower()
        if self.greenlang_version:
            params['greenlang_version'] = self.greenlang_version
        if self.python_version:
            params['python_version'] = self.python_version
        
        return params


class PackIndex:
    """Pack index and discovery system"""
    
    def __init__(self, client: Optional[HubClient] = None,
                 cache_dir: Optional[Path] = None,
                 cache_ttl: int = 3600):
        """
        Initialize PackIndex
        
        Args:
            client: HubClient instance
            cache_dir: Directory for caching index data
            cache_ttl: Cache time-to-live in seconds
        """
        self.client = client or HubClient()
        self.cache_ttl = cache_ttl
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".greenlang" / "index_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Local index cache
        self._index_cache = {}
        self._cache_timestamps = {}
        
        logger.info(f"PackIndex initialized with cache at {self.cache_dir}")
    
    def search(self, query: str = None,
              filters: Optional[SearchFilters] = None,
              sort: SortOrder = SortOrder.RELEVANCE,
              limit: int = 20,
              offset: int = 0) -> List[PackInfo]:
        """
        Search for packs in registry
        
        Args:
            query: Search query string
            filters: Search filters
            sort: Sort order
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of matching PackInfo objects
        """
        params = {
            "limit": limit,
            "offset": offset,
            "sort": sort.value
        }
        
        if query:
            params["q"] = query
        
        if filters:
            params.update(filters.to_params())
        
        try:
            logger.info(f"Searching packs with query: {query}")
            response = self.client.session.get("/api/v1/search", params=params)
            response.raise_for_status()
            
            results = response.json()
            packs = [PackInfo.from_dict(p) for p in results.get("results", [])]
            
            # Cache results
            self._cache_results(packs)
            
            logger.info(f"Found {len(packs)} packs")
            return packs
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Try to return cached results if available
            return self._get_cached_search(query, filters)
    
    def get_featured(self, limit: int = 10) -> List[PackInfo]:
        """
        Get featured/recommended packs
        
        Args:
            limit: Maximum number of featured packs
            
        Returns:
            List of featured PackInfo objects
        """
        cache_key = f"featured_{limit}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self._index_cache[cache_key]
        
        try:
            logger.info("Fetching featured packs")
            response = self.client.session.get(
                "/api/v1/featured",
                params={"limit": limit}
            )
            response.raise_for_status()
            
            results = response.json()
            packs = [PackInfo.from_dict(p) for p in results]
            
            # Cache results
            self._index_cache[cache_key] = packs
            self._cache_timestamps[cache_key] = datetime.now()
            
            logger.info(f"Retrieved {len(packs)} featured packs")
            return packs
            
        except Exception as e:
            logger.error(f"Failed to get featured packs: {e}")
            return []
    
    def get_trending(self, period: str = "week", limit: int = 10) -> List[PackInfo]:
        """
        Get trending packs
        
        Args:
            period: Time period (day, week, month)
            limit: Maximum number of trending packs
            
        Returns:
            List of trending PackInfo objects
        """
        cache_key = f"trending_{period}_{limit}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self._index_cache[cache_key]
        
        try:
            logger.info(f"Fetching trending packs for {period}")
            response = self.client.session.get(
                "/api/v1/trending",
                params={"period": period, "limit": limit}
            )
            response.raise_for_status()
            
            results = response.json()
            packs = [PackInfo.from_dict(p) for p in results]
            
            # Cache results
            self._index_cache[cache_key] = packs
            self._cache_timestamps[cache_key] = datetime.now()
            
            logger.info(f"Retrieved {len(packs)} trending packs")
            return packs
            
        except Exception as e:
            logger.error(f"Failed to get trending packs: {e}")
            return []
    
    def get_by_category(self, category: PackCategory, 
                        limit: int = 20,
                        offset: int = 0) -> List[PackInfo]:
        """
        Get packs by category
        
        Args:
            category: Pack category
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of PackInfo objects in category
        """
        filters = SearchFilters(categories=[category.value])
        return self.search(filters=filters, limit=limit, offset=offset)
    
    def get_by_author(self, author: str,
                     limit: int = 50,
                     offset: int = 0) -> List[PackInfo]:
        """
        Get all packs by a specific author
        
        Args:
            author: Author username
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of author's PackInfo objects
        """
        filters = SearchFilters(author=author)
        return self.search(filters=filters, limit=limit, offset=offset)
    
    def get_similar(self, pack_id: str, limit: int = 10) -> List[PackInfo]:
        """
        Get packs similar to a given pack
        
        Args:
            pack_id: Pack ID to find similar packs for
            limit: Maximum results
            
        Returns:
            List of similar PackInfo objects
        """
        try:
            logger.info(f"Finding packs similar to {pack_id}")
            response = self.client.session.get(
                f"/api/v1/packs/{pack_id}/similar",
                params={"limit": limit}
            )
            response.raise_for_status()
            
            results = response.json()
            packs = [PackInfo.from_dict(p) for p in results]
            
            logger.info(f"Found {len(packs)} similar packs")
            return packs
            
        except Exception as e:
            logger.error(f"Failed to get similar packs: {e}")
            return []
    
    def get_dependencies(self, pack_id: str, 
                        recursive: bool = False) -> List[PackInfo]:
        """
        Get pack dependencies
        
        Args:
            pack_id: Pack ID
            recursive: Include transitive dependencies
            
        Returns:
            List of dependency PackInfo objects
        """
        try:
            logger.info(f"Getting dependencies for {pack_id}")
            response = self.client.session.get(
                f"/api/v1/packs/{pack_id}/dependencies",
                params={"recursive": recursive}
            )
            response.raise_for_status()
            
            results = response.json()
            packs = [PackInfo.from_dict(p) for p in results]
            
            logger.info(f"Found {len(packs)} dependencies")
            return packs
            
        except Exception as e:
            logger.error(f"Failed to get dependencies: {e}")
            return []
    
    def get_recommendations(self, user_packs: List[str] = None,
                           limit: int = 10) -> List[PackInfo]:
        """
        Get personalized pack recommendations
        
        Args:
            user_packs: List of pack IDs user has installed
            limit: Maximum recommendations
            
        Returns:
            List of recommended PackInfo objects
        """
        try:
            logger.info("Getting personalized recommendations")
            
            data = {
                "installed_packs": user_packs or [],
                "limit": limit
            }
            
            response = self.client.session.post(
                "/api/v1/recommendations",
                json=data
            )
            response.raise_for_status()
            
            results = response.json()
            packs = [PackInfo.from_dict(p) for p in results]
            
            logger.info(f"Generated {len(packs)} recommendations")
            return packs
            
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            # Fallback to featured packs
            return self.get_featured(limit)
    
    def get_categories(self) -> List[Dict[str, Any]]:
        """
        Get all available categories with counts
        
        Returns:
            List of category information
        """
        cache_key = "categories"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self._index_cache[cache_key]
        
        try:
            logger.info("Fetching categories")
            response = self.client.session.get("/api/v1/categories")
            response.raise_for_status()
            
            categories = response.json()
            
            # Cache results
            self._index_cache[cache_key] = categories
            self._cache_timestamps[cache_key] = datetime.now()
            
            return categories
            
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            # Return default categories
            return [
                {"name": cat.value, "display_name": cat.name.replace("_", " ").title()}
                for cat in PackCategory
            ]
    
    def get_tags(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get popular tags
        
        Args:
            limit: Maximum number of tags
            
        Returns:
            List of tag information with counts
        """
        cache_key = f"tags_{limit}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self._index_cache[cache_key]
        
        try:
            logger.info("Fetching popular tags")
            response = self.client.session.get(
                "/api/v1/tags",
                params={"limit": limit}
            )
            response.raise_for_status()
            
            tags = response.json()
            
            # Cache results
            self._index_cache[cache_key] = tags
            self._cache_timestamps[cache_key] = datetime.now()
            
            return tags
            
        except Exception as e:
            logger.error(f"Failed to get tags: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics
        
        Returns:
            Dictionary of registry statistics
        """
        cache_key = "statistics"
        
        # Check cache (shorter TTL for stats)
        if self._is_cache_valid(cache_key, ttl=300):
            return self._index_cache[cache_key]
        
        try:
            logger.info("Fetching registry statistics")
            response = self.client.session.get("/api/v1/stats")
            response.raise_for_status()
            
            stats = response.json()
            
            # Cache results
            self._index_cache[cache_key] = stats
            self._cache_timestamps[cache_key] = datetime.now()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def update_local_index(self):
        """Update local index cache from registry"""
        try:
            logger.info("Updating local index")
            
            # Fetch index data
            response = self.client.session.get("/api/v1/index")
            response.raise_for_status()
            
            index_data = response.json()
            
            # Save to local cache file
            index_file = self.cache_dir / "index.json"
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            # Update memory cache
            self._load_local_index()
            
            logger.info("Local index updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update local index: {e}")
    
    def search_local(self, query: str = None,
                    filters: Optional[SearchFilters] = None) -> List[PackInfo]:
        """
        Search in local index (offline mode)
        
        Args:
            query: Search query
            filters: Search filters
            
        Returns:
            List of matching PackInfo objects from local cache
        """
        # Load local index if not loaded
        if not hasattr(self, '_local_index'):
            self._load_local_index()
        
        if not self._local_index:
            logger.warning("No local index available")
            return []
        
        results = []
        query_lower = query.lower() if query else ""
        
        for pack_data in self._local_index:
            pack = PackInfo.from_dict(pack_data)
            
            # Apply query filter
            if query and not self._matches_query(pack, query_lower):
                continue
            
            # Apply other filters
            if filters and not self._matches_filters(pack, filters):
                continue
            
            results.append(pack)
        
        return results
    
    def _load_local_index(self):
        """Load local index from cache file"""
        index_file = self.cache_dir / "index.json"
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                self._local_index = index_data.get("packs", [])
                logger.info(f"Loaded {len(self._local_index)} packs from local index")
            except Exception as e:
                logger.error(f"Failed to load local index: {e}")
                self._local_index = []
        else:
            self._local_index = []
    
    def _matches_query(self, pack: PackInfo, query: str) -> bool:
        """Check if pack matches search query"""
        # Search in name, description, tags, keywords
        searchable = [
            pack.name.lower(),
            pack.description.lower(),
            ' '.join(pack.tags).lower(),
            ' '.join(pack.keywords).lower(),
            pack.author.get('name', '').lower()
        ]
        
        return any(query in text for text in searchable)
    
    def _matches_filters(self, pack: PackInfo, filters: SearchFilters) -> bool:
        """Check if pack matches filters"""
        if filters.categories and not any(cat in pack.categories for cat in filters.categories):
            return False
        
        if filters.tags and not any(tag in pack.tags for tag in filters.tags):
            return False
        
        if filters.author and pack.author.get('name') != filters.author:
            return False
        
        if filters.license and pack.license != filters.license:
            return False
        
        if filters.min_stars is not None and pack.stars < filters.min_stars:
            return False
        
        if filters.min_downloads is not None and pack.downloads < filters.min_downloads:
            return False
        
        if filters.verified_only and not pack.verified:
            return False
        
        if filters.official_only and not pack.official:
            return False
        
        if filters.exclude_deprecated and pack.deprecated:
            return False
        
        return True
    
    def _cache_results(self, packs: List[PackInfo]):
        """Cache search results"""
        for pack in packs:
            cache_key = f"pack_{pack.id}"
            self._index_cache[cache_key] = pack
            self._cache_timestamps[cache_key] = datetime.now()
    
    def _is_cache_valid(self, key: str, ttl: Optional[int] = None) -> bool:
        """Check if cache entry is still valid"""
        if key not in self._index_cache:
            return False
        
        if key not in self._cache_timestamps:
            return False
        
        ttl = ttl or self.cache_ttl
        age = (datetime.now() - self._cache_timestamps[key]).total_seconds()
        
        return age < ttl
    
    def _get_cached_search(self, query: str = None,
                          filters: Optional[SearchFilters] = None) -> List[PackInfo]:
        """Get cached search results as fallback"""
        logger.info("Using cached search results")
        
        # Search through cached packs
        results = []
        query_lower = query.lower() if query else ""
        
        for key, value in self._index_cache.items():
            if key.startswith("pack_") and isinstance(value, PackInfo):
                if query and not self._matches_query(value, query_lower):
                    continue
                if filters and not self._matches_filters(value, filters):
                    continue
                results.append(value)
        
        return results