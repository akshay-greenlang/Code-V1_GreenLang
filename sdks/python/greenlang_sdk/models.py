"""
GreenLang SDK Models

Pydantic models for type-safe API interactions.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentCategory(str, Enum):
    """Agent categories"""
    CARBON = "carbon"
    SUSTAINABILITY = "sustainability"
    CLIMATE = "climate"
    ESG = "esg"
    ENERGY = "energy"
    RESEARCH = "research"
    ANALYSIS = "analysis"


class CitationType(str, Enum):
    """Citation types"""
    WEB = "web"
    ACADEMIC = "academic"
    DATABASE = "database"
    API = "api"
    INTERNAL = "internal"


class Workflow(BaseModel):
    """
    Workflow model

    Represents a workflow definition in GreenLang.
    """
    id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    category: Optional[str] = Field(None, description="Workflow category")
    agents: List[Dict[str, Any]] = Field(default_factory=list, description="Agent configurations")
    config: Dict[str, Any] = Field(default_factory=dict, description="Workflow configuration")
    is_public: bool = Field(False, description="Whether workflow is public")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    version: str = Field("1.0", description="Workflow version")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "wf_abc123",
                "name": "Carbon Footprint Analysis",
                "description": "Analyze carbon footprint from various data sources",
                "category": "carbon",
                "agents": [
                    {"agent_id": "carbon_analyzer", "config": {}}
                ],
                "is_public": False,
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z"
            }
        }


class WorkflowDefinition(BaseModel):
    """
    Workflow definition for creation

    Used when creating a new workflow.
    """
    name: str = Field(..., description="Workflow name", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="Workflow description", max_length=500)
    category: Optional[str] = Field(None, description="Workflow category")
    agents: List[Dict[str, Any]] = Field(..., description="Agent configurations", min_length=1)
    config: Dict[str, Any] = Field(default_factory=dict, description="Workflow configuration")
    is_public: bool = Field(False, description="Whether workflow is public")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Carbon Footprint Analysis",
                "description": "Analyze carbon footprint",
                "category": "carbon",
                "agents": [
                    {
                        "agent_id": "carbon_analyzer",
                        "config": {"threshold": 100}
                    }
                ],
                "is_public": False
            }
        }


class Agent(BaseModel):
    """
    Agent model

    Represents an AI agent in GreenLang.
    """
    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    category: AgentCategory = Field(..., description="Agent category")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    config_schema: Dict[str, Any] = Field(default_factory=dict, description="Configuration schema")
    is_public: bool = Field(True, description="Whether agent is public")
    version: str = Field("1.0", description="Agent version")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "carbon_analyzer",
                "name": "Carbon Analyzer",
                "description": "Analyzes carbon emissions from various sources",
                "category": "carbon",
                "capabilities": ["emission_calculation", "data_analysis"],
                "is_public": True,
                "created_at": "2025-01-01T00:00:00Z"
            }
        }


class ExecutionResult(BaseModel):
    """
    Execution result model

    Represents the result of a workflow or agent execution.
    """
    id: str = Field(..., description="Unique execution identifier")
    workflow_id: Optional[str] = Field(None, description="Workflow ID if workflow execution")
    agent_id: Optional[str] = Field(None, description="Agent ID if agent execution")
    status: WorkflowStatus = Field(..., description="Execution status")
    input_data: Dict[str, Any] = Field(..., description="Input data")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data")
    error: Optional[str] = Field(None, description="Error message if failed")
    started_at: datetime = Field(..., description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    duration_ms: Optional[int] = Field(None, description="Execution duration in milliseconds")
    citations: List['Citation'] = Field(default_factory=list, description="Citations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        """Convenience property for output data"""
        return self.output_data

    @property
    def is_complete(self) -> bool:
        """Check if execution is complete"""
        return self.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]

    @property
    def is_successful(self) -> bool:
        """Check if execution was successful"""
        return self.status == WorkflowStatus.COMPLETED

    class Config:
        json_schema_extra = {
            "example": {
                "id": "exec_xyz789",
                "workflow_id": "wf_abc123",
                "status": "completed",
                "input_data": {"query": "What is carbon footprint?"},
                "output_data": {"answer": "Carbon footprint is...", "confidence": 0.95},
                "started_at": "2025-01-01T00:00:00Z",
                "completed_at": "2025-01-01T00:00:05Z",
                "duration_ms": 5000
            }
        }


class Citation(BaseModel):
    """
    Citation model

    Represents a source citation for an execution result.
    """
    id: str = Field(..., description="Unique citation identifier")
    execution_id: str = Field(..., description="Execution ID")
    source_type: CitationType = Field(..., description="Citation type")
    source_url: Optional[HttpUrl] = Field(None, description="Source URL")
    source_title: Optional[str] = Field(None, description="Source title")
    source_author: Optional[str] = Field(None, description="Source author")
    published_date: Optional[datetime] = Field(None, description="Publication date")
    excerpt: Optional[str] = Field(None, description="Relevant excerpt")
    relevance_score: float = Field(..., description="Relevance score (0-1)", ge=0, le=1)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "cite_123",
                "execution_id": "exec_xyz789",
                "source_type": "web",
                "source_url": "https://example.com/carbon-footprint",
                "source_title": "Understanding Carbon Footprint",
                "excerpt": "A carbon footprint is the total amount of...",
                "relevance_score": 0.92
            }
        }


class PaginatedResponse(BaseModel):
    """
    Paginated response model

    Generic model for paginated API responses.
    """
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Current offset")
    has_more: bool = Field(..., description="Whether more items available")

    class Config:
        json_schema_extra = {
            "example": {
                "items": [...],
                "total": 100,
                "limit": 20,
                "offset": 0,
                "has_more": True
            }
        }


class APIError(BaseModel):
    """
    API error response model
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for debugging")


class AuthenticationError(APIError):
    """Authentication error response"""
    error: str = "authentication_error"


class RateLimitError(APIError):
    """Rate limit error response"""
    error: str = "rate_limit_error"
    retry_after: Optional[int] = Field(None, description="Seconds until retry allowed")


class ValidationError(APIError):
    """Validation error response"""
    error: str = "validation_error"
    field_errors: Optional[Dict[str, List[str]]] = Field(None, description="Field-specific errors")


# Update forward refs for Citation
ExecutionResult.model_rebuild()
