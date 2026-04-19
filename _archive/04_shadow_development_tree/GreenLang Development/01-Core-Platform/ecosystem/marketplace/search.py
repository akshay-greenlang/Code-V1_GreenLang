# -*- coding: utf-8 -*-
"""
Agent Search Engine

Implements full-text search, filtering, faceting, and autocomplete
for marketplace agents using PostgreSQL full-text search.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from sqlalchemy import func, and_, or_, desc, text
from sqlalchemy.orm import Session

from greenlang.marketplace.models import (
    MarketplaceAgent,
    AgentCategory,
    AgentTagModel,
    agent_tags,
    AgentSearchHistory,
)

logger = logging.getLogger(__name__)


class SortBy(str, Enum):
    """Search sort options"""
    RELEVANCE = "relevance"
    DOWNLOADS = "downloads"
    RATING = "rating"
    NEWEST = "newest"
    UPDATED = "updated"
    ALPHABETICAL = "alphabetical"


@dataclass
class SearchFilter:
    """Search filter parameters"""
    categories: List[int] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    pricing_types: List[str] = field(default_factory=list)
    licenses: List[str] = field(default_factory=list)
    min_rating: Optional[float] = None
    verified_only: bool = False
    featured_only: bool = False
    min_greenlang_version: Optional[str] = None
    max_greenlang_version: Optional[str] = None


@dataclass
class SearchFacets:
    """Search facets (filter counts)"""
    categories: Dict[int, int] = field(default_factory=dict)  # {category_id: count}
    tags: Dict[str, int] = field(default_factory=dict)  # {tag: count}
    pricing_types: Dict[str, int] = field(default_factory=dict)
    price_ranges: Dict[str, int] = field(default_factory=dict)
    ratings: Dict[str, int] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Single search result"""
    agent: MarketplaceAgent
    score: float
    highlight: Optional[str] = None


@dataclass
class SearchResponse:
    """Complete search response"""
    results: List[SearchResult]
    total_count: int
    facets: SearchFacets
    page: int
    page_size: int
    total_pages: int


class AgentSearchEngine:
    """
    Main search engine for agents.

    Provides full-text search with filtering, sorting, and faceting.
    """

    def __init__(self, session: Session):
        self.session = session

    def search(
        self,
        query: Optional[str] = None,
        filters: Optional[SearchFilter] = None,
        sort_by: SortBy = SortBy.RELEVANCE,
        page: int = 1,
        page_size: int = 20,
        user_id: Optional[str] = None
    ) -> SearchResponse:
        """
        Search for agents.

        Args:
            query: Search query string
            filters: Filter parameters
            sort_by: Sort order
            page: Page number (1-indexed)
            page_size: Results per page
            user_id: Optional user ID for tracking

        Returns:
            Search response with results and facets
        """
        if filters is None:
            filters = SearchFilter()

        # Base query
        q = self.session.query(MarketplaceAgent).filter(
            MarketplaceAgent.status == "published"
        )

        # Full-text search
        if query:
            # PostgreSQL full-text search
            search_query = func.to_tsquery('english', query)
            q = q.filter(
                MarketplaceAgent.search_vector.op('@@')(search_query)
            )

        # Apply filters
        q = self._apply_filters(q, filters)

        # Count total before pagination
        total_count = q.count()

        # Apply sorting
        q = self._apply_sorting(q, sort_by, query)

        # Pagination
        offset = (page - 1) * page_size
        results = q.limit(page_size).offset(offset).all()

        # Calculate relevance scores (simplified)
        search_results = []
        for agent in results:
            score = self._calculate_score(agent, query, sort_by)
            search_results.append(SearchResult(
                agent=agent,
                score=score,
                highlight=self._get_highlight(agent, query) if query else None
            ))

        # Calculate facets
        facets = self._calculate_facets(query, filters)

        # Track search
        if user_id and query:
            self._track_search(user_id, query, filters, total_count)

        total_pages = (total_count + page_size - 1) // page_size

        return SearchResponse(
            results=search_results,
            total_count=total_count,
            facets=facets,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    def _apply_filters(
        self,
        query,
        filters: SearchFilter
    ):
        """Apply search filters to query"""
        # Category filter
        if filters.categories:
            query = query.filter(
                MarketplaceAgent.category_id.in_(filters.categories)
            )

        # Tag filter
        if filters.tags:
            query = query.join(agent_tags).join(AgentTagModel).filter(
                AgentTagModel.name.in_(filters.tags)
            )

        # Price filter
        if filters.min_price is not None:
            query = query.filter(MarketplaceAgent.price >= filters.min_price)

        if filters.max_price is not None:
            query = query.filter(MarketplaceAgent.price <= filters.max_price)

        # Pricing type filter
        if filters.pricing_types:
            query = query.filter(
                MarketplaceAgent.pricing_type.in_(filters.pricing_types)
            )

        # Rating filter
        if filters.min_rating is not None:
            query = query.filter(MarketplaceAgent.rating_avg >= filters.min_rating)

        # Verified filter
        if filters.verified_only:
            query = query.filter(MarketplaceAgent.verified == True)

        # Featured filter
        if filters.featured_only:
            query = query.filter(MarketplaceAgent.featured == True)

        return query

    def _apply_sorting(self, query, sort_by: SortBy, search_query: Optional[str]):
        """Apply sorting to query"""
        if sort_by == SortBy.RELEVANCE and search_query:
            # Sort by relevance (ts_rank)
            search_q = func.to_tsquery('english', search_query)
            query = query.order_by(
                desc(func.ts_rank(MarketplaceAgent.search_vector, search_q))
            )

        elif sort_by == SortBy.DOWNLOADS:
            query = query.order_by(desc(MarketplaceAgent.downloads))

        elif sort_by == SortBy.RATING:
            query = query.order_by(
                desc(MarketplaceAgent.wilson_score),
                desc(MarketplaceAgent.rating_count)
            )

        elif sort_by == SortBy.NEWEST:
            query = query.order_by(desc(MarketplaceAgent.published_at))

        elif sort_by == SortBy.UPDATED:
            query = query.order_by(desc(MarketplaceAgent.updated_at))

        elif sort_by == SortBy.ALPHABETICAL:
            query = query.order_by(MarketplaceAgent.name)

        else:
            # Default: wilson score
            query = query.order_by(desc(MarketplaceAgent.wilson_score))

        return query

    def _calculate_score(
        self,
        agent: MarketplaceAgent,
        query: Optional[str],
        sort_by: SortBy
    ) -> float:
        """Calculate relevance score for result"""
        score = 0.0

        if sort_by == SortBy.RELEVANCE and query:
            # Text relevance (simplified)
            query_lower = query.lower()
            if query_lower in agent.name.lower():
                score += 10.0
            if query_lower in agent.description.lower():
                score += 5.0

        elif sort_by == SortBy.DOWNLOADS:
            score = float(agent.downloads)

        elif sort_by == SortBy.RATING:
            score = agent.wilson_score

        return score

    def _get_highlight(
        self,
        agent: MarketplaceAgent,
        query: str
    ) -> Optional[str]:
        """Get highlighted snippet"""
        # Simple highlighting (in production, use ts_headline)
        query_lower = query.lower()

        if query_lower in agent.description.lower():
            # Find position and extract context
            pos = agent.description.lower().find(query_lower)
            start = max(0, pos - 50)
            end = min(len(agent.description), pos + len(query) + 50)
            snippet = agent.description[start:end]

            # Highlight query
            snippet = snippet.replace(
                query,
                f"<mark>{query}</mark>"
            )

            return f"...{snippet}..."

        return None

    def _calculate_facets(
        self,
        query: Optional[str],
        filters: SearchFilter
    ) -> SearchFacets:
        """Calculate facet counts"""
        facets = SearchFacets()

        # Base query (without current filters)
        base_q = self.session.query(MarketplaceAgent).filter(
            MarketplaceAgent.status == "published"
        )

        if query:
            search_query = func.to_tsquery('english', query)
            base_q = base_q.filter(
                MarketplaceAgent.search_vector.op('@@')(search_query)
            )

        # Category facets
        cat_counts = self.session.query(
            MarketplaceAgent.category_id,
            func.count(MarketplaceAgent.id)
        ).filter(
            MarketplaceAgent.status == "published"
        ).group_by(MarketplaceAgent.category_id).all()

        facets.categories = {cat_id: count for cat_id, count in cat_counts if cat_id}

        # Pricing type facets
        pricing_counts = self.session.query(
            MarketplaceAgent.pricing_type,
            func.count(MarketplaceAgent.id)
        ).filter(
            MarketplaceAgent.status == "published"
        ).group_by(MarketplaceAgent.pricing_type).all()

        facets.pricing_types = {pt: count for pt, count in pricing_counts}

        # Rating facets
        facets.ratings = {
            "5_stars": base_q.filter(MarketplaceAgent.rating_avg >= 4.5).count(),
            "4_stars": base_q.filter(
                and_(
                    MarketplaceAgent.rating_avg >= 4.0,
                    MarketplaceAgent.rating_avg < 4.5
                )
            ).count(),
            "3_stars": base_q.filter(
                and_(
                    MarketplaceAgent.rating_avg >= 3.0,
                    MarketplaceAgent.rating_avg < 4.0
                )
            ).count(),
        }

        return facets

    def _track_search(
        self,
        user_id: str,
        query: str,
        filters: SearchFilter,
        results_count: int
    ):
        """Track search for analytics"""
        search_history = AgentSearchHistory(
            user_id=user_id,
            query=query,
            filters={
                "categories": filters.categories,
                "tags": filters.tags,
                "pricing_types": filters.pricing_types,
            },
            results_count=results_count
        )

        self.session.add(search_history)
        self.session.commit()


class SearchSuggestions:
    """
    Search autocomplete and suggestions.

    Provides query suggestions based on popular searches and agent names.
    """

    def __init__(self, session: Session):
        self.session = session

    def get_suggestions(
        self,
        query: str,
        limit: int = 10
    ) -> List[str]:
        """
        Get search suggestions.

        Args:
            query: Partial query
            limit: Maximum suggestions

        Returns:
            List of suggestions
        """
        suggestions = []

        # Agent name suggestions
        agents = self.session.query(MarketplaceAgent.name).filter(
            and_(
                MarketplaceAgent.status == "published",
                MarketplaceAgent.name.ilike(f"%{query}%")
            )
        ).limit(limit).all()

        suggestions.extend([a.name for a in agents])

        # Popular search suggestions
        if len(suggestions) < limit:
            popular = self.session.query(AgentSearchHistory.query).filter(
                AgentSearchHistory.query.ilike(f"%{query}%")
            ).group_by(AgentSearchHistory.query).order_by(
                desc(func.count(AgentSearchHistory.id))
            ).limit(limit - len(suggestions)).all()

            suggestions.extend([p.query for p in popular])

        return suggestions[:limit]

    def get_popular_searches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular recent searches"""
        popular = self.session.query(
            AgentSearchHistory.query,
            func.count(AgentSearchHistory.id).label('count')
        ).group_by(AgentSearchHistory.query).order_by(
            desc('count')
        ).limit(limit).all()

        return [
            {"query": query, "count": count}
            for query, count in popular
        ]
