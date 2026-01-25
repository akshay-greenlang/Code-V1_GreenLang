# -*- coding: utf-8 -*-
"""
REST API endpoints for dashboard management.

This module provides CRUD operations for dashboards, widgets,
sharing, and templates.
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, validator
from sqlalchemy import and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from greenlang.api.dependencies import get_current_user, get_db
from greenlang.utilities.determinism import DeterministicClock
from greenlang.db.models_analytics import (
    Dashboard,
    DashboardWidget,
    DashboardShare,
    DashboardTemplate,
    DashboardFolder
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboards", tags=["dashboards"])


# Pydantic models for request/response

class WidgetDataSource(BaseModel):
    """Widget data source configuration."""
    channel: str
    metricName: Optional[str] = None
    tags: Optional[dict] = None
    aggregation: Optional[str] = None


class WidgetConfigModel(BaseModel):
    """Widget configuration model."""
    id: str
    type: str
    title: str
    config: dict = Field(default_factory=dict)
    dataSource: WidgetDataSource
    position: dict = Field(default_factory=dict)


class DashboardCreateModel(BaseModel):
    """Dashboard creation request model."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    layout: dict = Field(default_factory=dict)
    widgets: List[WidgetConfigModel] = Field(default_factory=list)
    folderId: Optional[UUID] = None
    tags: List[str] = Field(default_factory=list)
    accessLevel: str = Field(default='private')

    @validator('accessLevel')
    def validate_access_level(cls, v):
        allowed = ['private', 'team', 'public']
        if v not in allowed:
            raise ValueError(f"Access level must be one of: {allowed}")
        return v


class DashboardUpdateModel(BaseModel):
    """Dashboard update request model."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    layout: Optional[dict] = None
    widgets: Optional[List[WidgetConfigModel]] = None
    folderId: Optional[UUID] = None
    tags: Optional[List[str]] = None
    accessLevel: Optional[str] = None

    @validator('accessLevel')
    def validate_access_level(cls, v):
        if v is not None:
            allowed = ['private', 'team', 'public']
            if v not in allowed:
                raise ValueError(f"Access level must be one of: {allowed}")
        return v


class DashboardShareModel(BaseModel):
    """Dashboard share request model."""
    expiresIn: Optional[int] = Field(None, description="Expiration time in hours")
    canView: bool = True
    canEdit: bool = False


class DashboardResponse(BaseModel):
    """Dashboard response model."""
    id: UUID
    name: str
    description: Optional[str]
    layout: dict
    widgets: List[dict]
    tags: List[str]
    accessLevel: str
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True


class DashboardShareResponse(BaseModel):
    """Dashboard share response model."""
    id: UUID
    token: str
    expiresAt: Optional[datetime]
    canView: bool
    canEdit: bool
    createdAt: datetime

    class Config:
        from_attributes = True


class DashboardTemplateResponse(BaseModel):
    """Dashboard template response model."""
    id: UUID
    name: str
    description: Optional[str]
    category: str
    layout: dict
    widgets: List[dict]
    tags: List[str]
    usageCount: int

    class Config:
        from_attributes = True


# Helper functions

async def get_dashboard_or_404(
    dashboard_id: UUID,
    user_id: UUID,
    db: AsyncSession
) -> Dashboard:
    """Get dashboard by ID or raise 404."""
    stmt = select(Dashboard).where(
        and_(
            Dashboard.id == dashboard_id,
            or_(
                Dashboard.created_by == user_id,
                Dashboard.access_level.in_(['team', 'public'])
            )
        )
    )
    result = await db.execute(stmt)
    dashboard = result.scalar_one_or_none()

    if not dashboard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard not found"
        )

    return dashboard


def serialize_dashboard(dashboard: Dashboard) -> dict:
    """Serialize dashboard to response format."""
    widgets = [
        {
            'id': str(widget.id),
            'type': widget.widget_type,
            'title': widget.title,
            'config': widget.config,
            'dataSource': widget.data_source,
            'position': {
                'x': widget.position_x,
                'y': widget.position_y,
                'w': widget.width,
                'h': widget.height
            }
        }
        for widget in dashboard.widgets
    ]

    return {
        'id': dashboard.id,
        'name': dashboard.name,
        'description': dashboard.description,
        'layout': dashboard.layout,
        'widgets': widgets,
        'tags': dashboard.tags or [],
        'accessLevel': dashboard.access_level,
        'createdAt': dashboard.created_at,
        'updatedAt': dashboard.updated_at
    }


# API endpoints

@router.get("/", response_model=List[DashboardResponse])
async def list_dashboards(
    folder_id: Optional[UUID] = Query(None),
    tag: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List user dashboards with filtering and pagination."""
    user_id = UUID(current_user['sub'])

    # Build query
    filters = [
        or_(
            Dashboard.created_by == user_id,
            Dashboard.access_level.in_(['team', 'public'])
        ),
        Dashboard.is_archived == False
    ]

    if folder_id:
        filters.append(Dashboard.folder_id == folder_id)

    if search:
        filters.append(
            or_(
                Dashboard.name.ilike(f"%{search}%"),
                Dashboard.description.ilike(f"%{search}%")
            )
        )

    stmt = select(Dashboard).where(and_(*filters)).offset(skip).limit(limit)
    result = await db.execute(stmt)
    dashboards = result.scalars().all()

    # Filter by tag if specified
    if tag:
        dashboards = [d for d in dashboards if tag in (d.tags or [])]

    return [serialize_dashboard(d) for d in dashboards]


@router.post("/", response_model=DashboardResponse, status_code=status.HTTP_201_CREATED)
async def create_dashboard(
    dashboard: DashboardCreateModel,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new dashboard."""
    user_id = UUID(current_user['sub'])

    # Create dashboard
    new_dashboard = Dashboard(
        name=dashboard.name,
        description=dashboard.description,
        layout=dashboard.layout,
        created_by=user_id,
        folder_id=dashboard.folderId,
        tags=dashboard.tags,
        access_level=dashboard.accessLevel
    )

    db.add(new_dashboard)
    await db.flush()

    # Create widgets
    for widget_data in dashboard.widgets:
        widget = DashboardWidget(
            dashboard_id=new_dashboard.id,
            widget_type=widget_data.type,
            title=widget_data.title,
            config=widget_data.config,
            data_source={
                'channel': widget_data.dataSource.channel,
                'metricName': widget_data.dataSource.metricName,
                'tags': widget_data.dataSource.tags,
                'aggregation': widget_data.dataSource.aggregation
            },
            position_x=widget_data.position.get('x', 0),
            position_y=widget_data.position.get('y', 0),
            width=widget_data.position.get('w', 6),
            height=widget_data.position.get('h', 4)
        )
        db.add(widget)

    await db.commit()
    await db.refresh(new_dashboard)

    logger.info(f"Created dashboard {new_dashboard.id} by user {user_id}")

    return serialize_dashboard(new_dashboard)


@router.get("/{dashboard_id}", response_model=DashboardResponse)
async def get_dashboard(
    dashboard_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get dashboard by ID."""
    user_id = UUID(current_user['sub'])
    dashboard = await get_dashboard_or_404(dashboard_id, user_id, db)
    return serialize_dashboard(dashboard)


@router.put("/{dashboard_id}", response_model=DashboardResponse)
async def update_dashboard(
    dashboard_id: UUID,
    update_data: DashboardUpdateModel,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update dashboard."""
    user_id = UUID(current_user['sub'])
    dashboard = await get_dashboard_or_404(dashboard_id, user_id, db)

    # Check permissions
    if dashboard.created_by != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this dashboard"
        )

    # Update fields
    if update_data.name is not None:
        dashboard.name = update_data.name
    if update_data.description is not None:
        dashboard.description = update_data.description
    if update_data.layout is not None:
        dashboard.layout = update_data.layout
    if update_data.folderId is not None:
        dashboard.folder_id = update_data.folderId
    if update_data.tags is not None:
        dashboard.tags = update_data.tags
    if update_data.accessLevel is not None:
        dashboard.access_level = update_data.accessLevel

    # Update widgets if provided
    if update_data.widgets is not None:
        # Remove existing widgets
        for widget in dashboard.widgets:
            await db.delete(widget)

        # Add new widgets
        for widget_data in update_data.widgets:
            widget = DashboardWidget(
                dashboard_id=dashboard.id,
                widget_type=widget_data.type,
                title=widget_data.title,
                config=widget_data.config,
                data_source={
                    'channel': widget_data.dataSource.channel,
                    'metricName': widget_data.dataSource.metricName,
                    'tags': widget_data.dataSource.tags,
                    'aggregation': widget_data.dataSource.aggregation
                },
                position_x=widget_data.position.get('x', 0),
                position_y=widget_data.position.get('y', 0),
                width=widget_data.position.get('w', 6),
                height=widget_data.position.get('h', 4)
            )
            db.add(widget)

    await db.commit()
    await db.refresh(dashboard)

    logger.info(f"Updated dashboard {dashboard_id}")

    return serialize_dashboard(dashboard)


@router.delete("/{dashboard_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dashboard(
    dashboard_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete dashboard."""
    user_id = UUID(current_user['sub'])
    dashboard = await get_dashboard_or_404(dashboard_id, user_id, db)

    # Check permissions
    if dashboard.created_by != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this dashboard"
        )

    await db.delete(dashboard)
    await db.commit()

    logger.info(f"Deleted dashboard {dashboard_id}")


@router.post("/{dashboard_id}/share", response_model=DashboardShareResponse)
async def share_dashboard(
    dashboard_id: UUID,
    share_data: DashboardShareModel,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Share dashboard via token."""
    user_id = UUID(current_user['sub'])
    dashboard = await get_dashboard_or_404(dashboard_id, user_id, db)

    # Check permissions
    if dashboard.created_by != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to share this dashboard"
        )

    # Generate share token
    token = secrets.token_urlsafe(32)

    # Calculate expiration
    expires_at = None
    if share_data.expiresIn:
        expires_at = DeterministicClock.utcnow() + timedelta(hours=share_data.expiresIn)

    # Create share
    share = DashboardShare(
        dashboard_id=dashboard.id,
        token=token,
        expires_at=expires_at,
        can_view=share_data.canView,
        can_edit=share_data.canEdit,
        created_by=user_id
    )

    db.add(share)
    await db.commit()
    await db.refresh(share)

    logger.info(f"Created share link for dashboard {dashboard_id}")

    return {
        'id': share.id,
        'token': share.token,
        'expiresAt': share.expires_at,
        'canView': share.can_view,
        'canEdit': share.can_edit,
        'createdAt': share.created_at
    }


@router.get("/shared/{token}", response_model=DashboardResponse)
async def get_shared_dashboard(
    token: str,
    db: AsyncSession = Depends(get_db)
):
    """Access dashboard via share token."""
    stmt = select(DashboardShare).where(DashboardShare.token == token)
    result = await db.execute(stmt)
    share = result.scalar_one_or_none()

    if not share:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found"
        )

    # Check expiration
    if share.is_expired():
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Share link has expired"
        )

    # Update access statistics
    share.accessed_count += 1
    share.last_accessed_at = DeterministicClock.utcnow()
    await db.commit()

    # Get dashboard
    stmt = select(Dashboard).where(Dashboard.id == share.dashboard_id)
    result = await db.execute(stmt)
    dashboard = result.scalar_one_or_none()

    if not dashboard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard not found"
        )

    return serialize_dashboard(dashboard)


@router.get("/templates/", response_model=List[DashboardTemplateResponse])
async def list_templates(
    category: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """List dashboard templates."""
    filters = [DashboardTemplate.is_public == True]

    if category:
        filters.append(DashboardTemplate.category == category)

    stmt = select(DashboardTemplate).where(and_(*filters)).offset(skip).limit(limit)
    result = await db.execute(stmt)
    templates = result.scalars().all()

    return [
        {
            'id': t.id,
            'name': t.name,
            'description': t.description,
            'category': t.category,
            'layout': t.layout,
            'widgets': t.widgets,
            'tags': t.tags or [],
            'usageCount': t.usage_count
        }
        for t in templates
    ]


@router.post("/from-template/{template_id}", response_model=DashboardResponse)
async def create_from_template(
    template_id: UUID,
    name: str = Query(..., min_length=1),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create dashboard from template."""
    user_id = UUID(current_user['sub'])

    # Get template
    stmt = select(DashboardTemplate).where(DashboardTemplate.id == template_id)
    result = await db.execute(stmt)
    template = result.scalar_one_or_none()

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found"
        )

    # Create dashboard from template
    dashboard = Dashboard(
        name=name,
        description=template.description,
        layout=template.layout,
        created_by=user_id,
        tags=template.tags or []
    )

    db.add(dashboard)
    await db.flush()

    # Create widgets from template
    for widget_data in template.widgets:
        widget = DashboardWidget(
            dashboard_id=dashboard.id,
            widget_type=widget_data['type'],
            title=widget_data['title'],
            config=widget_data['config'],
            data_source=widget_data['dataSource'],
            position_x=widget_data['position']['x'],
            position_y=widget_data['position']['y'],
            width=widget_data['position']['w'],
            height=widget_data['position']['h']
        )
        db.add(widget)

    # Update template usage count
    template.usage_count += 1

    await db.commit()
    await db.refresh(dashboard)

    logger.info(f"Created dashboard {dashboard.id} from template {template_id}")

    return serialize_dashboard(dashboard)
