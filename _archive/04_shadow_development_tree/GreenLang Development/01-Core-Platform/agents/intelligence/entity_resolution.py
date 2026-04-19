"""
Entity Resolution Services - Multi-source entity identification and validation

This module implements entity resolution using multiple authoritative sources:
- GLEIF (Global Legal Entity Identifier Foundation) for LEI lookups
- Dun & Bradstreet for DUNS number lookups
- OpenCorporates for public company registry data

The module provides a unified interface with fallback strategies, caching,
and confidence scoring for entity matching.

Example:
    >>> resolver = EntityResolver()
    >>> result = await resolver.resolve_entity("Apple Inc", country="US")
    >>> print(result.confidence_score)
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import hashlib
import logging
import asyncio
import aiohttp
from functools import lru_cache
import json
import re
from enum import Enum
from difflib import SequenceMatcher
import time
from collections import defaultdict
from urllib.parse import quote

logger = logging.getLogger(__name__)


class EntitySource(Enum):
    """Supported entity resolution sources."""
    GLEIF = "gleif"
    DUNS = "duns"
    OPENCORPORATES = "opencorporates"
    INTERNAL = "internal"


class EntityType(Enum):
    """Types of entities we can resolve."""
    CORPORATION = "corporation"
    LLC = "llc"
    PARTNERSHIP = "partnership"
    NONPROFIT = "nonprofit"
    GOVERNMENT = "government"
    SUBSIDIARY = "subsidiary"
    BRANCH = "branch"


class EntityMatch(BaseModel):
    """Represents a matched entity from any source."""

    # Core identifiers
    entity_name: str = Field(..., description="Official entity name")
    legal_name: Optional[str] = Field(None, description="Full legal name if different")

    # Identifiers
    lei: Optional[str] = Field(None, description="Legal Entity Identifier (20 chars)")
    duns: Optional[str] = Field(None, description="D-U-N-S Number (9 digits)")
    opencorporates_id: Optional[str] = Field(None, description="OpenCorporates jurisdiction/number")

    # Location
    country: str = Field(..., description="ISO country code")
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction")
    address: Optional[Dict[str, str]] = Field(None, description="Registered address")

    # Company details
    entity_type: Optional[EntityType] = Field(None, description="Type of entity")
    status: Optional[str] = Field(None, description="Active, Inactive, etc.")
    incorporation_date: Optional[datetime] = Field(None, description="Date of incorporation")

    # Parent/ownership
    parent_lei: Optional[str] = Field(None, description="Parent company LEI")
    parent_name: Optional[str] = Field(None, description="Parent company name")
    ultimate_parent_lei: Optional[str] = Field(None, description="Ultimate parent LEI")

    # Metadata
    source: EntitySource = Field(..., description="Data source")
    confidence_score: float = Field(..., ge=0, le=1, description="Match confidence 0-1")
    match_method: str = Field(..., description="How the match was made")
    retrieved_at: datetime = Field(default_factory=datetime.now)
    data_freshness: Optional[datetime] = Field(None, description="When source data was last updated")

    # Additional data from source
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Original source data")

    @validator('lei')
    def validate_lei(cls, v):
        """Validate LEI format (20 alphanumeric characters)."""
        if v and not re.match(r'^[A-Z0-9]{20}$', v):
            raise ValueError(f"Invalid LEI format: {v}")
        return v

    @validator('duns')
    def validate_duns(cls, v):
        """Validate DUNS format (9 digits)."""
        if v and not re.match(r'^\d{9}$', v):
            raise ValueError(f"Invalid DUNS format: {v}")
        return v


class ResolutionRequest(BaseModel):
    """Request for entity resolution."""

    query: str = Field(..., description="Company name or identifier to search")
    country: Optional[str] = Field(None, description="ISO country code to filter by")
    jurisdiction: Optional[str] = Field(None, description="Specific jurisdiction")

    # Optional identifiers if known
    lei: Optional[str] = Field(None, description="Known LEI to validate")
    duns: Optional[str] = Field(None, description="Known DUNS to validate")

    # Search parameters
    fuzzy_match: bool = Field(True, description="Enable fuzzy name matching")
    min_confidence: float = Field(0.7, ge=0, le=1, description="Minimum confidence threshold")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results per source")

    # Source preferences
    preferred_sources: List[EntitySource] = Field(
        default_factory=lambda: [EntitySource.GLEIF, EntitySource.DUNS, EntitySource.OPENCORPORATES],
        description="Sources to query in order"
    )
    timeout_seconds: int = Field(30, description="Timeout for external API calls")


class ResolutionResult(BaseModel):
    """Complete result of entity resolution across all sources."""

    request: ResolutionRequest = Field(..., description="Original request")
    best_match: Optional[EntityMatch] = Field(None, description="Highest confidence match")
    all_matches: List[EntityMatch] = Field(default_factory=list, description="All found matches")

    # Resolution metadata
    sources_queried: List[EntitySource] = Field(default_factory=list)
    sources_succeeded: List[EntitySource] = Field(default_factory=list)
    sources_failed: Dict[str, str] = Field(default_factory=dict, description="Source -> error message")

    # Performance metrics
    resolution_time_ms: float = Field(..., description="Total resolution time")
    api_calls_made: int = Field(0, description="Number of external API calls")
    cache_hits: int = Field(0, description="Number of cache hits")

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    resolved_at: datetime = Field(default_factory=datetime.now)


class GLEIFClient:
    """Client for GLEIF (Global Legal Entity Identifier Foundation) API."""

    BASE_URL = "https://api.gleif.org/api/v1"
    RATE_LIMIT = 60  # requests per minute

    def __init__(self, api_key: Optional[str] = None):
        """Initialize GLEIF client."""
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = RateLimiter(self.RATE_LIMIT, 60)
        self._cache = {}  # Simple in-memory cache

    async def __aenter__(self):
        """Create session on context entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session on context exit."""
        if self.session:
            await self.session.close()

    async def search_by_name(self, name: str, country: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for entities by name.

        Args:
            name: Company name to search
            country: Optional ISO country code filter

        Returns:
            List of matching LEI records
        """
        cache_key = f"gleif_name_{name}_{country}"
        if cache_key in self._cache:
            logger.debug(f"GLEIF cache hit for {name}")
            return self._cache[cache_key]

        await self._rate_limiter.acquire()

        params = {
            "filter[entity.legalName]": name,
            "page[size]": 10
        }

        if country:
            params["filter[entity.legalAddress.country]"] = country

        url = f"{self.BASE_URL}/lei-records"

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("data", [])
                    self._cache[cache_key] = results
                    return results
                elif response.status == 429:
                    logger.warning("GLEIF rate limit exceeded")
                    return []
                else:
                    logger.error(f"GLEIF API error: {response.status}")
                    return []

        except asyncio.TimeoutError:
            logger.error("GLEIF API timeout")
            return []
        except Exception as e:
            logger.error(f"GLEIF API exception: {e}")
            return []

    async def get_by_lei(self, lei: str) -> Optional[Dict[str, Any]]:
        """
        Get entity details by LEI.

        Args:
            lei: Legal Entity Identifier (20 characters)

        Returns:
            LEI record if found
        """
        cache_key = f"gleif_lei_{lei}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        await self._rate_limiter.acquire()

        url = f"{self.BASE_URL}/lei-records/{lei}"

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("data")
                    self._cache[cache_key] = result
                    return result
                else:
                    return None

        except Exception as e:
            logger.error(f"GLEIF LEI lookup error: {e}")
            return None

    def parse_lei_record(self, record: Dict[str, Any]) -> EntityMatch:
        """Parse GLEIF API response into EntityMatch."""
        attrs = record.get("attributes", {})
        entity = attrs.get("entity", {})
        address = entity.get("legalAddress", {})

        # Get parent information if available
        relationships = record.get("relationships", {})
        parent_lei = None
        ultimate_parent_lei = None

        if relationships:
            parent_ref = relationships.get("direct-parent", {})
            if parent_ref:
                parent_lei = parent_ref.get("data", {}).get("id")

            ultimate_ref = relationships.get("ultimate-parent", {})
            if ultimate_ref:
                ultimate_parent_lei = ultimate_ref.get("data", {}).get("id")

        return EntityMatch(
            entity_name=entity.get("legalName", {}).get("name", ""),
            lei=attrs.get("lei"),
            country=address.get("country", ""),
            jurisdiction=address.get("region"),
            address={
                "line1": address.get("addressLine1"),
                "city": address.get("city"),
                "region": address.get("region"),
                "postal_code": address.get("postalCode"),
                "country": address.get("country")
            },
            entity_type=self._map_entity_type(entity.get("legalForm")),
            status="ACTIVE" if attrs.get("entity", {}).get("status") == "ACTIVE" else "INACTIVE",
            parent_lei=parent_lei,
            ultimate_parent_lei=ultimate_parent_lei,
            source=EntitySource.GLEIF,
            confidence_score=1.0,  # Direct LEI lookup is 100% confident
            match_method="lei_lookup",
            data_freshness=datetime.fromisoformat(attrs.get("lastUpdateDate", datetime.now().isoformat())),
            raw_data=record
        )

    def _map_entity_type(self, legal_form: Optional[Dict]) -> Optional[EntityType]:
        """Map GLEIF legal form to our entity type."""
        if not legal_form:
            return None

        # This is simplified - would need comprehensive mapping
        form_code = legal_form.get("id", "").upper()
        if "CORP" in form_code or "INC" in form_code:
            return EntityType.CORPORATION
        elif "LLC" in form_code:
            return EntityType.LLC
        elif "PARTNER" in form_code:
            return EntityType.PARTNERSHIP
        else:
            return EntityType.CORPORATION  # Default


class DUNSClient:
    """Client for Dun & Bradstreet API (simulated for this implementation)."""

    BASE_URL = "https://api.dnb.com/v1"  # Actual D&B API endpoint
    RATE_LIMIT = 100  # requests per minute

    def __init__(self, api_key: str, api_secret: str):
        """Initialize DUNS client with credentials."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = RateLimiter(self.RATE_LIMIT, 60)
        self._cache = {}
        self._token = None
        self._token_expires = datetime.now()

    async def __aenter__(self):
        """Create session on context entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session on context exit."""
        if self.session:
            await self.session.close()

    async def _get_auth_token(self) -> Optional[str]:
        """Get or refresh authentication token."""
        if self._token and datetime.now() < self._token_expires:
            return self._token

        # In production, implement OAuth2 flow
        # This is a placeholder
        logger.info("D&B authentication would happen here")
        return "simulated_token"

    async def search_by_name(self, name: str, country: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for companies by name in D&B database.

        Args:
            name: Company name
            country: Optional country filter

        Returns:
            List of matching companies
        """
        cache_key = f"duns_name_{name}_{country}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # In production, implement actual D&B API call
        # This is a simulated response
        logger.info(f"Would search D&B for: {name} in {country}")

        # Simulate some matches for demonstration
        if "microsoft" in name.lower():
            result = [{
                "duns": "081466849",
                "name": "Microsoft Corporation",
                "tradeName": "Microsoft",
                "country": "US",
                "address": {
                    "line1": "One Microsoft Way",
                    "city": "Redmond",
                    "state": "WA",
                    "postalCode": "98052",
                    "country": "US"
                },
                "numberOfEmployees": 221000,
                "salesRevenue": 198270000000,
                "incorporationYear": 1975
            }]
            self._cache[cache_key] = result
            return result

        return []

    async def get_by_duns(self, duns: str) -> Optional[Dict[str, Any]]:
        """
        Get company details by DUNS number.

        Args:
            duns: 9-digit DUNS number

        Returns:
            Company details if found
        """
        cache_key = f"duns_{duns}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # In production, implement actual API call
        logger.info(f"Would lookup DUNS: {duns}")
        return None

    def parse_duns_record(self, record: Dict[str, Any]) -> EntityMatch:
        """Parse D&B API response into EntityMatch."""
        return EntityMatch(
            entity_name=record.get("name", ""),
            legal_name=record.get("legalName"),
            duns=record.get("duns"),
            country=record.get("country", ""),
            address=record.get("address"),
            entity_type=EntityType.CORPORATION,
            status="ACTIVE",
            incorporation_date=datetime(record.get("incorporationYear", 2000), 1, 1) if "incorporationYear" in record else None,
            source=EntitySource.DUNS,
            confidence_score=0.95,
            match_method="duns_search",
            raw_data=record
        )


class OpenCorporatesClient:
    """Client for OpenCorporates API."""

    BASE_URL = "https://api.opencorporates.com/v0.4"
    RATE_LIMIT = 60  # for free tier

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenCorporates client."""
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = RateLimiter(self.RATE_LIMIT, 60)
        self._cache = {}

    async def __aenter__(self):
        """Create session on context entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session on context exit."""
        if self.session:
            await self.session.close()

    async def search_companies(
        self,
        name: str,
        jurisdiction: Optional[str] = None,
        country: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for companies in OpenCorporates.

        Args:
            name: Company name to search
            jurisdiction: Specific jurisdiction (e.g., "us_de" for Delaware)
            country: Country code

        Returns:
            List of matching companies
        """
        cache_key = f"oc_search_{name}_{jurisdiction}_{country}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        await self._rate_limiter.acquire()

        params = {
            "q": name,
            "order": "score",
            "per_page": 10
        }

        if self.api_key:
            params["api_token"] = self.api_key

        if jurisdiction:
            params["jurisdiction_code"] = jurisdiction
        elif country:
            params["country_code"] = country

        url = f"{self.BASE_URL}/companies/search"

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(url, params=params, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    companies = data.get("results", {}).get("companies", [])
                    results = [c.get("company", {}) for c in companies]
                    self._cache[cache_key] = results
                    return results
                else:
                    logger.error(f"OpenCorporates API error: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"OpenCorporates exception: {e}")
            return []

    async def get_company(self, jurisdiction: str, company_number: str) -> Optional[Dict[str, Any]]:
        """
        Get company details by jurisdiction and number.

        Args:
            jurisdiction: Jurisdiction code (e.g., "us_de")
            company_number: Company registration number

        Returns:
            Company details if found
        """
        cache_key = f"oc_company_{jurisdiction}_{company_number}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        await self._rate_limiter.acquire()

        url = f"{self.BASE_URL}/companies/{jurisdiction}/{company_number}"
        params = {}
        if self.api_key:
            params["api_token"] = self.api_key

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("results", {}).get("company")
                    self._cache[cache_key] = result
                    return result
                else:
                    return None

        except Exception as e:
            logger.error(f"OpenCorporates lookup error: {e}")
            return None

    def parse_company_record(self, record: Dict[str, Any], confidence: float = 0.85) -> EntityMatch:
        """Parse OpenCorporates response into EntityMatch."""
        address = record.get("registered_address", {})

        return EntityMatch(
            entity_name=record.get("name", ""),
            opencorporates_id=f"{record.get('jurisdiction_code')}/{record.get('company_number')}",
            country=address.get("country") or record.get("jurisdiction_code", "").split("_")[0].upper(),
            jurisdiction=record.get("jurisdiction_code"),
            address={
                "line1": address.get("street_address"),
                "city": address.get("locality"),
                "region": address.get("region"),
                "postal_code": address.get("postal_code"),
                "country": address.get("country")
            } if address else None,
            entity_type=self._map_company_type(record.get("company_type")),
            status=record.get("current_status", "").upper(),
            incorporation_date=datetime.fromisoformat(record.get("incorporation_date")) if record.get("incorporation_date") else None,
            source=EntitySource.OPENCORPORATES,
            confidence_score=confidence,
            match_method="opencorporates_search",
            data_freshness=datetime.fromisoformat(record.get("updated_at", datetime.now().isoformat())),
            raw_data=record
        )

    def _map_company_type(self, company_type: Optional[str]) -> EntityType:
        """Map OpenCorporates company type to our enum."""
        if not company_type:
            return EntityType.CORPORATION

        type_lower = company_type.lower()
        if "llc" in type_lower or "limited liability" in type_lower:
            return EntityType.LLC
        elif "partnership" in type_lower:
            return EntityType.PARTNERSHIP
        elif "nonprofit" in type_lower or "charity" in type_lower:
            return EntityType.NONPROFIT
        else:
            return EntityType.CORPORATION


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_calls: int, period_seconds: int):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed
            period_seconds: Time period for the limit
        """
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.calls = defaultdict(list)

    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        key = "default"

        # Remove old calls outside the window
        self.calls[key] = [t for t in self.calls[key] if now - t < self.period_seconds]

        # If at limit, wait
        if len(self.calls[key]) >= self.max_calls:
            sleep_time = self.period_seconds - (now - self.calls[key][0]) + 0.1
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)

        # Record this call
        self.calls[key].append(now)


class EntityResolver:
    """
    Unified entity resolver that combines multiple data sources.

    This class orchestrates entity resolution across GLEIF, D&B, and OpenCorporates,
    providing intelligent fallback, caching, and confidence scoring.
    """

    def __init__(
        self,
        gleif_api_key: Optional[str] = None,
        duns_api_key: Optional[str] = None,
        duns_api_secret: Optional[str] = None,
        opencorporates_api_key: Optional[str] = None,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize EntityResolver with API credentials.

        Args:
            gleif_api_key: Optional GLEIF API key
            duns_api_key: D&B API key
            duns_api_secret: D&B API secret
            opencorporates_api_key: Optional OpenCorporates API key
            cache_ttl_hours: Cache TTL in hours
        """
        self.gleif_client = GLEIFClient(gleif_api_key)
        self.duns_client = DUNSClient(duns_api_key or "demo", duns_api_secret or "demo")
        self.oc_client = OpenCorporatesClient(opencorporates_api_key)

        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self._resolution_cache = {}

        # Statistics
        self.stats = {
            "total_resolutions": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "source_successes": defaultdict(int),
            "source_failures": defaultdict(int)
        }

    async def resolve_entity(self, request: Union[ResolutionRequest, str]) -> ResolutionResult:
        """
        Main entry point for entity resolution.

        Args:
            request: ResolutionRequest or company name string

        Returns:
            ResolutionResult with all matches and metadata
        """
        start_time = time.time()

        # Convert string to request if needed
        if isinstance(request, str):
            request = ResolutionRequest(query=request)

        # Check cache
        cache_key = self._get_cache_key(request)
        if cache_key in self._resolution_cache:
            cached_result, cached_time = self._resolution_cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                self.stats["cache_hits"] += 1
                cached_result.cache_hits = 1
                return cached_result

        # Initialize result
        result = ResolutionResult(
            request=request,
            resolution_time_ms=0,
            provenance_hash=""
        )

        all_matches = []

        # Try each source
        for source in request.preferred_sources:
            try:
                matches = await self._resolve_from_source(source, request)
                if matches:
                    all_matches.extend(matches)
                    result.sources_succeeded.append(source)
                result.sources_queried.append(source)

            except Exception as e:
                logger.error(f"Error with {source.value}: {e}")
                result.sources_failed[source.value] = str(e)
                self.stats["source_failures"][source.value] += 1

        # Sort by confidence and apply threshold
        all_matches = [m for m in all_matches if m.confidence_score >= request.min_confidence]
        all_matches.sort(key=lambda x: x.confidence_score, reverse=True)

        # Take top N results
        result.all_matches = all_matches[:request.max_results]
        result.best_match = all_matches[0] if all_matches else None

        # Calculate metrics
        result.resolution_time_ms = (time.time() - start_time) * 1000
        result.api_calls_made = self.stats["api_calls"]

        # Generate provenance hash
        result.provenance_hash = self._calculate_provenance(request, result)

        # Cache the result
        self._resolution_cache[cache_key] = (result, datetime.now())

        # Update stats
        self.stats["total_resolutions"] += 1

        return result

    async def _resolve_from_source(
        self,
        source: EntitySource,
        request: ResolutionRequest
    ) -> List[EntityMatch]:
        """
        Resolve entity from a specific source.

        Args:
            source: Which source to query
            request: Resolution request

        Returns:
            List of EntityMatch objects
        """
        matches = []

        if source == EntitySource.GLEIF:
            matches = await self._resolve_from_gleif(request)
        elif source == EntitySource.DUNS:
            matches = await self._resolve_from_duns(request)
        elif source == EntitySource.OPENCORPORATES:
            matches = await self._resolve_from_opencorporates(request)

        return matches

    async def _resolve_from_gleif(self, request: ResolutionRequest) -> List[EntityMatch]:
        """Resolve using GLEIF API."""
        matches = []

        try:
            async with self.gleif_client as client:
                # If LEI provided, validate it
                if request.lei:
                    self.stats["api_calls"] += 1
                    record = await client.get_by_lei(request.lei)
                    if record:
                        match = client.parse_lei_record(record)
                        matches.append(match)

                # Otherwise search by name
                else:
                    self.stats["api_calls"] += 1
                    results = await client.search_by_name(request.query, request.country)

                    for record in results[:request.max_results]:
                        match = client.parse_lei_record(record)

                        # Calculate name similarity
                        similarity = self._calculate_name_similarity(
                            request.query,
                            match.entity_name
                        )
                        match.confidence_score = similarity
                        match.match_method = "name_search"

                        if similarity >= request.min_confidence:
                            matches.append(match)

            self.stats["source_successes"]["gleif"] += 1

        except Exception as e:
            logger.error(f"GLEIF resolution error: {e}")
            self.stats["source_failures"]["gleif"] += 1

        return matches

    async def _resolve_from_duns(self, request: ResolutionRequest) -> List[EntityMatch]:
        """Resolve using D&B API."""
        matches = []

        try:
            async with self.duns_client as client:
                # If DUNS provided, validate it
                if request.duns:
                    self.stats["api_calls"] += 1
                    record = await client.get_by_duns(request.duns)
                    if record:
                        match = client.parse_duns_record(record)
                        matches.append(match)

                # Otherwise search by name
                else:
                    self.stats["api_calls"] += 1
                    results = await client.search_by_name(request.query, request.country)

                    for record in results[:request.max_results]:
                        match = client.parse_duns_record(record)

                        # Calculate name similarity
                        similarity = self._calculate_name_similarity(
                            request.query,
                            match.entity_name
                        )
                        match.confidence_score = similarity

                        if similarity >= request.min_confidence:
                            matches.append(match)

            self.stats["source_successes"]["duns"] += 1

        except Exception as e:
            logger.error(f"D&B resolution error: {e}")
            self.stats["source_failures"]["duns"] += 1

        return matches

    async def _resolve_from_opencorporates(self, request: ResolutionRequest) -> List[EntityMatch]:
        """Resolve using OpenCorporates API."""
        matches = []

        try:
            async with self.oc_client as client:
                self.stats["api_calls"] += 1
                results = await client.search_companies(
                    request.query,
                    request.jurisdiction,
                    request.country
                )

                for record in results[:request.max_results]:
                    # Calculate name similarity
                    similarity = self._calculate_name_similarity(
                        request.query,
                        record.get("name", "")
                    )

                    if similarity >= request.min_confidence:
                        match = client.parse_company_record(record, similarity)
                        matches.append(match)

            self.stats["source_successes"]["opencorporates"] += 1

        except Exception as e:
            logger.error(f"OpenCorporates resolution error: {e}")
            self.stats["source_failures"]["opencorporates"] += 1

        return matches

    def _calculate_name_similarity(self, query: str, candidate: str) -> float:
        """
        Calculate similarity between query and candidate name.

        Uses multiple techniques:
        - Exact match: 1.0
        - Case-insensitive match: 0.95
        - Sequence matching for fuzzy comparison

        Args:
            query: Search query
            candidate: Candidate company name

        Returns:
            Similarity score 0-1
        """
        if not query or not candidate:
            return 0.0

        # Normalize for comparison
        query_norm = self._normalize_company_name(query)
        candidate_norm = self._normalize_company_name(candidate)

        # Exact match
        if query_norm == candidate_norm:
            return 1.0

        # Use SequenceMatcher for fuzzy matching
        similarity = SequenceMatcher(None, query_norm, candidate_norm).ratio()

        # Boost score if one contains the other
        if query_norm in candidate_norm or candidate_norm in query_norm:
            similarity = max(similarity, 0.85)

        return similarity

    def _normalize_company_name(self, name: str) -> str:
        """
        Normalize company name for matching.

        Args:
            name: Company name

        Returns:
            Normalized name
        """
        # Convert to lowercase
        normalized = name.lower()

        # Remove common suffixes
        suffixes = [
            " inc", " incorporated", " corp", " corporation",
            " llc", " ltd", " limited", " plc", " gmbh", " ag",
            " co", " company", " & co", " and co"
        ]

        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break

        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        return normalized

    def _get_cache_key(self, request: ResolutionRequest) -> str:
        """Generate cache key for request."""
        key_parts = [
            request.query,
            request.country or "",
            request.jurisdiction or "",
            request.lei or "",
            request.duns or "",
            str(request.min_confidence)
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _calculate_provenance(self, request: ResolutionRequest, result: ResolutionResult) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "request": request.dict(),
            "best_match": result.best_match.dict() if result.best_match else None,
            "match_count": len(result.all_matches),
            "sources_queried": [s.value for s in result.sources_queried],
            "resolved_at": result.resolved_at.isoformat()
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    async def batch_resolve(
        self,
        queries: List[Union[str, ResolutionRequest]],
        max_concurrent: int = 5
    ) -> List[ResolutionResult]:
        """
        Resolve multiple entities in batch.

        Args:
            queries: List of queries or requests
            max_concurrent: Maximum concurrent resolutions

        Returns:
            List of resolution results
        """
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def resolve_with_semaphore(query):
            async with semaphore:
                return await self.resolve_entity(query)

        # Run all resolutions concurrently
        tasks = [resolve_with_semaphore(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch resolve error for query {i}: {result}")
                # Create error result
                req = queries[i] if isinstance(queries[i], ResolutionRequest) else ResolutionRequest(query=queries[i])
                error_result = ResolutionResult(
                    request=req,
                    resolution_time_ms=0,
                    provenance_hash="error",
                    sources_failed={"batch": str(result)}
                )
                final_results.append(error_result)
            else:
                final_results.append(result)

        return final_results

    def get_statistics(self) -> Dict[str, Any]:
        """Get resolver statistics."""
        return {
            "total_resolutions": self.stats["total_resolutions"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["total_resolutions"]),
            "total_api_calls": self.stats["api_calls"],
            "source_successes": dict(self.stats["source_successes"]),
            "source_failures": dict(self.stats["source_failures"]),
            "cache_size": len(self._resolution_cache)
        }


# Example usage and testing
async def example_usage():
    """Demonstrate entity resolution capabilities."""

    # Initialize resolver
    resolver = EntityResolver(
        gleif_api_key=None,  # Optional for GLEIF
        duns_api_key="your_duns_key",  # Required for D&B
        duns_api_secret="your_duns_secret",
        opencorporates_api_key=None  # Optional for basic OC access
    )

    print("=" * 80)
    print("Entity Resolution Examples")
    print("=" * 80)

    # Example 1: Simple company name search
    print("\n1. Simple name search:")
    result = await resolver.resolve_entity("Apple Inc")
    if result.best_match:
        print(f"   Best match: {result.best_match.entity_name}")
        print(f"   LEI: {result.best_match.lei}")
        print(f"   Confidence: {result.best_match.confidence_score:.2f}")
        print(f"   Source: {result.best_match.source.value}")

    # Example 2: Search with country filter
    print("\n2. Search with country filter:")
    request = ResolutionRequest(
        query="Toyota",
        country="JP",
        min_confidence=0.8
    )
    result = await resolver.resolve_entity(request)
    print(f"   Found {len(result.all_matches)} matches")
    for match in result.all_matches[:3]:
        print(f"   - {match.entity_name} ({match.source.value}, {match.confidence_score:.2f})")

    # Example 3: LEI validation
    print("\n3. LEI validation:")
    request = ResolutionRequest(
        query="Microsoft",
        lei="INR2EJN1ERAN0W5ZP974"  # Microsoft's actual LEI
    )
    result = await resolver.resolve_entity(request)
    if result.best_match:
        print(f"   Validated: {result.best_match.entity_name}")
        print(f"   Status: {result.best_match.status}")

    # Example 4: Batch resolution
    print("\n4. Batch resolution:")
    companies = [
        "Google LLC",
        "Amazon.com Inc",
        "Tesla Inc",
        "Samsung Electronics"
    ]
    results = await resolver.batch_resolve(companies)
    for company, result in zip(companies, results):
        match = result.best_match
        if match:
            print(f"   {company} -> {match.entity_name} ({match.source.value})")

    # Example 5: Multi-source resolution with fallback
    print("\n5. Multi-source resolution:")
    request = ResolutionRequest(
        query="Volkswagen",
        preferred_sources=[EntitySource.GLEIF, EntitySource.OPENCORPORATES, EntitySource.DUNS],
        min_confidence=0.7
    )
    result = await resolver.resolve_entity(request)
    print(f"   Sources queried: {[s.value for s in result.sources_queried]}")
    print(f"   Sources succeeded: {[s.value for s in result.sources_succeeded]}")
    if result.sources_failed:
        print(f"   Sources failed: {result.sources_failed}")
    print(f"   Total matches found: {len(result.all_matches)}")

    # Show statistics
    print("\n" + "=" * 80)
    print("Resolution Statistics:")
    stats = resolver.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_usage())