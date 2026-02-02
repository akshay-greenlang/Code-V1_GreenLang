# -*- coding: utf-8 -*-
"""
Category Management System

Manages agent categories with hierarchical structure, icons, and statistics.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from sqlalchemy.orm import Session
from sqlalchemy import desc

from greenlang.marketplace.models import AgentCategory, MarketplaceAgent

logger = logging.getLogger(__name__)


# Predefined category hierarchy with icons
CATEGORY_HIERARCHY = {
    "Data Processing": {
        "icon": "database",
        "description": "Agents for data processing and transformation",
        "children": {
            "CSV/Excel Processing": {"icon": "table", "description": "Process CSV and Excel files"},
            "JSON/XML Processing": {"icon": "code", "description": "Handle JSON and XML data"},
            "Data Validation": {"icon": "check-circle", "description": "Validate data integrity"},
            "Data Transformation": {"icon": "shuffle", "description": "Transform and enrich data"}
        }
    },
    "AI/ML": {
        "icon": "brain",
        "description": "Artificial Intelligence and Machine Learning agents",
        "children": {
            "Natural Language Processing": {"icon": "message-square", "description": "Text and language processing"},
            "Computer Vision": {"icon": "eye", "description": "Image and video analysis"},
            "Time Series Analysis": {"icon": "trending-up", "description": "Forecast and analyze time series"},
            "Reinforcement Learning": {"icon": "zap", "description": "RL-based optimization"}
        }
    },
    "Integration": {
        "icon": "link",
        "description": "Integration with external services and APIs",
        "children": {
            "APIs": {"icon": "cloud", "description": "REST, GraphQL, SOAP integrations"},
            "Databases": {"icon": "server", "description": "SQL and NoSQL databases"},
            "Cloud Services": {"icon": "cloud-upload", "description": "AWS, Azure, GCP services"},
            "Messaging": {"icon": "mail", "description": "Kafka, RabbitMQ, Redis"}
        }
    },
    "DevOps": {
        "icon": "settings",
        "description": "DevOps and infrastructure automation",
        "children": {
            "Monitoring": {"icon": "activity", "description": "System monitoring and alerts"},
            "Logging": {"icon": "file-text", "description": "Log aggregation and analysis"},
            "Deployment": {"icon": "upload-cloud", "description": "Deployment automation"},
            "CI/CD": {"icon": "git-branch", "description": "Continuous integration and deployment"}
        }
    },
    "Business": {
        "icon": "briefcase",
        "description": "Business process automation",
        "children": {
            "Accounting": {"icon": "dollar-sign", "description": "Financial and accounting tasks"},
            "CRM Integration": {"icon": "users", "description": "Customer relationship management"},
            "E-commerce": {"icon": "shopping-cart", "description": "E-commerce platforms"},
            "Analytics": {"icon": "bar-chart", "description": "Business analytics and reporting"}
        }
    },
    "Utilities": {
        "icon": "tool",
        "description": "Utility and helper agents",
        "children": {
            "Date/Time": {"icon": "calendar", "description": "Date and time manipulation"},
            "Math/Statistics": {"icon": "hash", "description": "Mathematical operations"},
            "File System": {"icon": "folder", "description": "File operations"},
            "Networking": {"icon": "globe", "description": "Network operations"}
        }
    }
}


@dataclass
class CategoryNode:
    """Category tree node"""
    id: int
    name: str
    slug: str
    icon: str
    description: str
    agent_count: int
    children: List['CategoryNode']
    parent_id: Optional[int] = None


class CategoryManager:
    """
    Manage agent categories.

    Provides category operations, tree building, and statistics.
    """

    def __init__(self, session: Session):
        self.session = session

    def get_category_tree(self) -> List[CategoryNode]:
        """
        Get full category tree.

        Returns:
            List of root category nodes with children
        """
        # Get all categories
        categories = self.session.query(AgentCategory).order_by(
            AgentCategory.display_order
        ).all()

        # Build tree
        category_map = {cat.id: self._category_to_node(cat) for cat in categories}

        # Link children to parents
        roots = []
        for cat in categories:
            node = category_map[cat.id]

            if cat.parent_id:
                parent = category_map.get(cat.parent_id)
                if parent:
                    parent.children.append(node)
                    node.parent_id = cat.parent_id
            else:
                roots.append(node)

        return roots

    def _category_to_node(self, category: AgentCategory) -> CategoryNode:
        """Convert category model to node"""
        return CategoryNode(
            id=category.id,
            name=category.name,
            slug=category.slug,
            icon=category.icon or "folder",
            description=category.description or "",
            agent_count=category.agent_count,
            children=[]
        )

    def get_category(self, category_id: int) -> Optional[AgentCategory]:
        """Get category by ID"""
        return self.session.query(AgentCategory).filter(
            AgentCategory.id == category_id
        ).first()

    def get_popular_categories(self, limit: int = 10) -> List[AgentCategory]:
        """Get categories with most agents"""
        return self.session.query(AgentCategory).order_by(
            desc(AgentCategory.agent_count)
        ).limit(limit).all()

    def update_category_counts(self):
        """Update agent counts for all categories"""
        categories = self.session.query(AgentCategory).all()

        for category in categories:
            count = self.session.query(MarketplaceAgent).filter(
                MarketplaceAgent.category_id == category.id,
                MarketplaceAgent.status == "published"
            ).count()

            category.agent_count = count

        self.session.commit()
        logger.info("Updated category counts")


def get_category_tree() -> Dict[str, Any]:
    """Get predefined category hierarchy as dict"""
    return CATEGORY_HIERARCHY
