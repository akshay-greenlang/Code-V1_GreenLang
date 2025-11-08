"""
Marketplace API Routes

FastAPI routes for the GreenLang Agent Marketplace.
Provides comprehensive REST API for agents, search, reviews, purchases, and analytics.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import logging

from greenlang.marketplace.models import (
    MarketplaceAgent,
    AgentVersion,
    AgentReview,
    AgentCategory,
)
from greenlang.marketplace.search import AgentSearchEngine, SearchFilter, SortBy
from greenlang.marketplace.rating_system import RatingSystem, ReviewModerator, ReviewSortBy
from greenlang.marketplace.recommendation import RecommendationEngine
from greenlang.marketplace.publisher import AgentPublisher
from greenlang.marketplace.validator import AgentValidator
from greenlang.marketplace.versioning import VersionManager
from greenlang.marketplace.dependency_resolver import DependencyResolver
from greenlang.marketplace.categories import CategoryManager
from greenlang.marketplace.monetization import MonetizationManager
from greenlang.marketplace.license_manager import LicenseManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/marketplace", tags=["marketplace"])


# Pydantic models for requests/responses
class AgentListResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: str
    author_name: str
    category: Optional[str]
    tags: List[str]
    price: float
    currency: str
    pricing_type: str
    downloads: int
    rating_avg: float
    rating_count: int
    verified: bool
    featured: bool


class AgentDetailResponse(AgentListResponse):
    long_description: Optional[str]
    readme: Optional[str]
    homepage_url: Optional[str]
    repository_url: Optional[str]
    documentation_url: Optional[str]
    current_version: Optional[str]
    created_at: str
    updated_at: Optional[str]


class SearchRequest(BaseModel):
    query: Optional[str] = None
    categories: List[int] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    verified_only: bool = False
    featured_only: bool = False
    sort_by: str = "relevance"
    page: int = 1
    page_size: int = 20


class ReviewRequest(BaseModel):
    rating: int = Field(ge=1, le=5)
    title: Optional[str] = None
    review_text: Optional[str] = None


class PublishRequest(BaseModel):
    name: str
    description: str
    category_id: int
    tags: List[str]
    pricing_type: str
    price: float = 0.0
    license_id: int
    readme: str


# Dependency for database session
def get_db() -> Session:
    """Get database session (implement based on your DB setup)"""
    # In production, yield actual session
    pass


def get_current_user() -> str:
    """Get current user ID from auth token"""
    # In production, extract from JWT token
    return "user_123"


# Agent endpoints
@router.get("/agents", response_model=List[AgentListResponse])
async def list_agents(
    category_id: Optional[int] = None,
    tag: Optional[str] = None,
    featured: bool = False,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List published agents with optional filters"""
    query = db.query(MarketplaceAgent).filter(
        MarketplaceAgent.status == "published"
    )

    if category_id:
        query = query.filter(MarketplaceAgent.category_id == category_id)

    if featured:
        query = query.filter(MarketplaceAgent.featured == True)

    query = query.order_by(MarketplaceAgent.wilson_score.desc())
    agents = query.limit(limit).offset(offset).all()

    return [AgentListResponse(**agent.to_dict()) for agent in agents]


@router.get("/agents/{agent_id}", response_model=AgentDetailResponse)
async def get_agent(agent_id: str, db: Session = Depends(get_db)):
    """Get agent details by ID"""
    agent = db.query(MarketplaceAgent).filter(
        MarketplaceAgent.id == agent_id
    ).first()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    return AgentDetailResponse(**agent.to_dict())


@router.post("/agents")
async def create_agent(
    request: PublishRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Create a new agent draft"""
    publisher = AgentPublisher(db)

    draft = publisher.create_draft(user_id, "Author Name")

    return {
        "draft_id": draft["draft_id"],
        "checklist": draft["checklist"]
    }


@router.post("/agents/{draft_id}/upload")
async def upload_agent_code(
    draft_id: str,
    code: UploadFile = File(...),
    readme: str = Body(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Upload and validate agent code"""
    publisher = AgentPublisher(db)

    code_content = await code.read()
    result = publisher.validate_and_upload(
        draft_id,
        code_content,
        code.filename,
        readme
    )

    if not result["success"]:
        return JSONResponse(
            status_code=400,
            content=result
        )

    return result


@router.put("/agents/{agent_id}")
async def update_agent(
    agent_id: str,
    request: PublishRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Update agent metadata"""
    agent = db.query(MarketplaceAgent).filter(
        MarketplaceAgent.id == agent_id,
        MarketplaceAgent.author_id == user_id
    ).first()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found or unauthorized")

    agent.name = request.name
    agent.description = request.description
    agent.category_id = request.category_id
    agent.pricing_type = request.pricing_type
    agent.price = request.price
    agent.license_id = request.license_id
    agent.readme = request.readme

    db.commit()

    return {"success": True, "agent_id": agent_id}


@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Delete agent (soft delete - set to deleted status)"""
    agent = db.query(MarketplaceAgent).filter(
        MarketplaceAgent.id == agent_id,
        MarketplaceAgent.author_id == user_id
    ).first()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found or unauthorized")

    agent.status = "deleted"
    db.commit()

    return {"success": True}


# Search endpoints
@router.post("/search")
async def search_agents(
    request: SearchRequest,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user)
):
    """Search agents with filters"""
    search_engine = AgentSearchEngine(db)

    filters = SearchFilter(
        categories=request.categories,
        tags=request.tags,
        min_price=request.min_price,
        max_price=request.max_price,
        min_rating=request.min_rating,
        verified_only=request.verified_only,
        featured_only=request.featured_only
    )

    sort_by = SortBy(request.sort_by) if request.sort_by else SortBy.RELEVANCE

    response = search_engine.search(
        query=request.query,
        filters=filters,
        sort_by=sort_by,
        page=request.page,
        page_size=request.page_size,
        user_id=user_id
    )

    return {
        "results": [
            {**result.agent.to_dict(), "score": result.score}
            for result in response.results
        ],
        "total_count": response.total_count,
        "page": response.page,
        "total_pages": response.total_pages,
        "facets": {
            "categories": response.facets.categories,
            "pricing_types": response.facets.pricing_types,
            "ratings": response.facets.ratings
        }
    }


@router.get("/search/suggestions")
async def get_suggestions(
    query: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get search suggestions"""
    from greenlang.marketplace.search import SearchSuggestions

    suggestions = SearchSuggestions(db)
    results = suggestions.get_suggestions(query, limit)

    return {"suggestions": results}


# Review endpoints
@router.get("/agents/{agent_id}/reviews")
async def get_reviews(
    agent_id: str,
    sort_by: str = "most_helpful",
    verified_only: bool = False,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get reviews for an agent"""
    moderator = ReviewModerator(db)

    sort = ReviewSortBy(sort_by) if sort_by else ReviewSortBy.MOST_HELPFUL

    reviews = moderator.get_reviews(
        agent_id,
        sort_by=sort,
        limit=limit,
        offset=offset,
        verified_only=verified_only
    )

    return {
        "reviews": [review.to_dict() for review in reviews],
        "count": len(reviews)
    }


@router.post("/agents/{agent_id}/reviews")
async def submit_review(
    agent_id: str,
    request: ReviewRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Submit a review for an agent"""
    moderator = ReviewModerator(db)

    success, review, errors = moderator.submit_review(
        user_id,
        agent_id,
        request.rating,
        request.title,
        request.review_text
    )

    if not success:
        return JSONResponse(
            status_code=400,
            content={"errors": errors}
        )

    return {"success": True, "review": review.to_dict()}


@router.post("/reviews/{review_id}/helpful")
async def vote_helpful(
    review_id: str,
    helpful: bool = True,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Vote if review is helpful"""
    moderator = ReviewModerator(db)
    success = moderator.vote_helpful(review_id, user_id, helpful)

    return {"success": success}


# Recommendation endpoints
@router.get("/recommendations/for-you")
async def get_personalized_recommendations(
    limit: int = 10,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Get personalized recommendations"""
    engine = RecommendationEngine(db)
    recommendations = engine.get_personalized_recommendations(user_id, limit)

    return {
        "recommendations": [
            {
                **rec.agent.to_dict(),
                "score": rec.score,
                "reason": rec.reason
            }
            for rec in recommendations
        ]
    }


@router.get("/recommendations/trending")
async def get_trending(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get trending agents"""
    engine = RecommendationEngine(db)
    trending = engine.popularity.get_trending(limit=limit)

    return {
        "trending": [
            {**rec.agent.to_dict(), "reason": rec.reason}
            for rec in trending
        ]
    }


@router.get("/agents/{agent_id}/similar")
async def get_similar_agents(
    agent_id: str,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """Get agents similar to given agent"""
    engine = RecommendationEngine(db)
    similar = engine.get_recommendations_for_agent(agent_id, limit)

    return {
        "similar": [
            {**rec.agent.to_dict(), "reason": rec.reason}
            for rec in similar
        ]
    }


# Category endpoints
@router.get("/categories")
async def get_categories(db: Session = Depends(get_db)):
    """Get category tree"""
    manager = CategoryManager(db)
    tree = manager.get_category_tree()

    def node_to_dict(node):
        return {
            "id": node.id,
            "name": node.name,
            "slug": node.slug,
            "icon": node.icon,
            "description": node.description,
            "agent_count": node.agent_count,
            "children": [node_to_dict(child) for child in node.children]
        }

    return {"categories": [node_to_dict(node) for node in tree]}


# Purchase endpoints
@router.post("/agents/{agent_id}/purchase")
async def purchase_agent(
    agent_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Purchase or subscribe to an agent"""
    manager = MonetizationManager(db)

    success, purchase_data, errors = manager.purchase_agent(agent_id, user_id)

    if not success:
        return JSONResponse(
            status_code=400,
            content={"errors": errors}
        )

    return purchase_data


@router.post("/agents/{agent_id}/install")
async def install_agent(
    agent_id: str,
    version: Optional[str] = None,
    machine_id: str = Body(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Install an agent (track installation)"""
    # Check if user has purchased agent (if paid)
    agent = db.query(MarketplaceAgent).filter(
        MarketplaceAgent.id == agent_id
    ).first()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Track installation
    from greenlang.marketplace.models import AgentInstall

    install = AgentInstall(
        user_id=user_id,
        agent_id=agent_id,
        version=version or "latest",
        installation_id=machine_id,
        active=True
    )

    db.add(install)

    # Increment download count
    agent.downloads += 1

    db.commit()

    return {"success": True, "installation_id": str(install.id)}


# License endpoints
@router.post("/licenses/activate")
async def activate_license(
    license_key: str = Body(...),
    machine_id: str = Body(...),
    agent_id: str = Body(...),
    db: Session = Depends(get_db)
):
    """Activate license on a machine"""
    manager = LicenseManager(db)

    success, activation_id, errors = manager.activate_license(
        license_key,
        machine_id,
        agent_id
    )

    if not success:
        return JSONResponse(
            status_code=400,
            content={"errors": errors}
        )

    return {"success": True, "activation_id": activation_id}


@router.post("/licenses/deactivate")
async def deactivate_license(
    license_key: str = Body(...),
    machine_id: str = Body(...),
    db: Session = Depends(get_db)
):
    """Deactivate license on a machine"""
    manager = LicenseManager(db)

    success, errors = manager.deactivate_license(license_key, machine_id)

    if not success:
        return JSONResponse(
            status_code=400,
            content={"errors": errors}
        )

    return {"success": True}


# Analytics endpoints
@router.get("/analytics/revenue")
async def get_revenue_stats(
    agent_id: Optional[str] = None,
    period_days: int = 30,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Get revenue statistics (for agent authors)"""
    manager = MonetizationManager(db)

    stats = manager.get_revenue_stats(
        agent_id=agent_id,
        author_id=user_id,
        period_days=period_days
    )

    return {
        "total_revenue": float(stats.total_revenue),
        "period_revenue": float(stats.period_revenue),
        "total_purchases": stats.total_purchases,
        "period_purchases": stats.period_purchases,
        "average_transaction": float(stats.average_transaction),
        "top_agents": stats.top_agents
    }


# Version endpoints
@router.get("/agents/{agent_id}/versions")
async def get_versions(
    agent_id: str,
    include_deprecated: bool = False,
    db: Session = Depends(get_db)
):
    """Get all versions of an agent"""
    manager = VersionManager(db)
    versions = manager.get_versions(agent_id, include_deprecated)

    return {
        "versions": [version.to_dict() for version in versions]
    }


@router.post("/agents/{agent_id}/versions")
async def publish_version(
    agent_id: str,
    version: str = Body(...),
    changelog: str = Body(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Publish a new version of an agent"""
    # Implementation would use PublishingWorkflow
    return {"success": True, "version": version}


# Dependency endpoints
@router.get("/agents/{agent_id}/dependencies")
async def resolve_dependencies(
    agent_id: str,
    version: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Resolve agent dependencies"""
    resolver = DependencyResolver(db)

    result = resolver.resolve(agent_id, version)

    if not result.success:
        return JSONResponse(
            status_code=400,
            content={
                "errors": result.errors,
                "conflicts": [
                    {
                        "agent_id": c.agent_id,
                        "agent_name": c.agent_name,
                        "required_versions": c.required_versions
                    }
                    for c in result.conflicts
                ]
            }
        )

    return {
        "resolved_versions": result.resolved_versions,
        "install_order": result.install_order
    }
