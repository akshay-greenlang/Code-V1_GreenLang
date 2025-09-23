"""
Core SDK Base Abstractions
===========================

Domain-agnostic base classes for building climate intelligence applications.
These are pure infrastructure components with no domain logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generic, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path

# Type variables for generic components
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")
TConfig = TypeVar("TConfig")

logger = logging.getLogger(__name__)


class Status(str, Enum):
    """Execution status"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Metadata:
    """Standard metadata for all components"""

    id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Result:
    """Standard result container"""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


class Agent(ABC, Generic[TInput, TOutput]):
    """
    Base Agent abstraction

    Agents are stateless computation units that:
    - Take typed input
    - Produce typed output
    - Can be composed into pipelines
    - Support validation and error handling
    """

    def __init__(self, metadata: Optional[Metadata] = None):
        """Initialize agent with metadata"""
        self.metadata = metadata or Metadata(
            id=self.__class__.__name__.lower(), name=self.__class__.__name__
        )
        self.logger = logging.getLogger(f"{__name__}.{self.metadata.id}")

    @abstractmethod
    def validate(self, input_data: TInput) -> bool:
        """
        Validate input data

        Args:
            input_data: Input to validate

        Returns:
            True if valid, False otherwise
        """

    @abstractmethod
    def process(self, input_data: TInput) -> TOutput:
        """
        Process input and produce output

        Args:
            input_data: Input data

        Returns:
            Processed output
        """

    def run(self, input_data: TInput) -> Result:
        """
        Run agent with validation and error handling

        Args:
            input_data: Input data

        Returns:
            Result container with output or error
        """
        try:
            # Validate input
            if not self.validate(input_data):
                return Result(success=False, error="Input validation failed")

            # Process
            output = self.process(input_data)

            return Result(
                success=True,
                data=output,
                metadata={"agent": self.metadata.id, "version": self.metadata.version},
            )

        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            return Result(success=False, error=str(e))

    def describe(self) -> Dict[str, Any]:
        """Describe agent capabilities"""
        return {
            "metadata": self.metadata.to_dict(),
            "input_schema": self.get_input_schema(),
            "output_schema": self.get_output_schema(),
        }

    def get_input_schema(self) -> Dict[str, Any]:
        """Get JSON schema for input"""
        # Override in subclasses for actual schema
        return {"type": "object"}

    def get_output_schema(self) -> Dict[str, Any]:
        """Get JSON schema for output"""
        # Override in subclasses for actual schema
        return {"type": "object"}


class Pipeline(ABC):
    """
    Base Pipeline abstraction

    Pipelines orchestrate multiple agents in sequence or parallel.
    """

    def __init__(self, metadata: Optional[Metadata] = None):
        """Initialize pipeline"""
        self.metadata = metadata or Metadata(
            id=self.__class__.__name__.lower(), name=self.__class__.__name__
        )
        self.agents: List[Agent] = []
        self.logger = logging.getLogger(f"{__name__}.{self.metadata.id}")

    def add_agent(self, agent: Agent) -> "Pipeline":
        """Add agent to pipeline"""
        self.agents.append(agent)
        return self

    @abstractmethod
    def execute(self, input_data: Any) -> Result:
        """
        Execute pipeline

        Args:
            input_data: Initial input

        Returns:
            Final result
        """

    def describe(self) -> Dict[str, Any]:
        """Describe pipeline structure"""
        return {
            "metadata": self.metadata.to_dict(),
            "agents": [agent.metadata.id for agent in self.agents],
            "flow": self.get_flow(),
        }

    def get_flow(self) -> Dict[str, Any]:
        """Get pipeline flow definition"""
        # Override in subclasses
        return {"type": "sequential"}


class Connector(ABC):
    """
    Base Connector abstraction

    Connectors integrate with external systems (APIs, databases, files).
    """

    def __init__(self, config: Optional[TConfig] = None):
        """Initialize connector with configuration"""
        self.config = config
        self.connected = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection

        Returns:
            True if connected successfully
        """

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close connection

        Returns:
            True if disconnected successfully
        """

    @abstractmethod
    def read(self, query: Any) -> Result:
        """
        Read data from source

        Args:
            query: Query specification

        Returns:
            Result with data or error
        """

    @abstractmethod
    def write(self, data: Any) -> Result:
        """
        Write data to destination

        Args:
            data: Data to write

        Returns:
            Result with success status
        """

    def is_connected(self) -> bool:
        """Check if connected"""
        return self.connected

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


class Dataset(ABC):
    """
    Base Dataset abstraction

    Datasets provide data access with metadata and provenance.
    """

    def __init__(self, metadata: Optional[Metadata] = None):
        """Initialize dataset"""
        self.metadata = metadata or Metadata(
            id=self.__class__.__name__.lower(), name=self.__class__.__name__
        )
        self.logger = logging.getLogger(f"{__name__}.{self.metadata.id}")

    @abstractmethod
    def load(self) -> Any:
        """Load dataset"""

    @abstractmethod
    def save(self, data: Any) -> bool:
        """Save dataset"""

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """Describe dataset"""

    def get_card(self) -> Optional[str]:
        """Get dataset card (documentation)"""
        return None

    def get_schema(self) -> Dict[str, Any]:
        """Get data schema"""
        return {"type": "object"}

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {}


class Report(ABC):
    """
    Base Report abstraction

    Reports generate formatted output from processed data.
    """

    def __init__(self, metadata: Optional[Metadata] = None):
        """Initialize report"""
        self.metadata = metadata or Metadata(
            id=self.__class__.__name__.lower(), name=self.__class__.__name__
        )
        self.logger = logging.getLogger(f"{__name__}.{self.metadata.id}")

    @abstractmethod
    def generate(self, data: Any, format: str = "markdown") -> str:
        """
        Generate report

        Args:
            data: Data to report on
            format: Output format (markdown, html, pdf, json)

        Returns:
            Formatted report
        """

    @abstractmethod
    def save(self, content: str, path: Path) -> bool:
        """
        Save report to file

        Args:
            content: Report content
            path: Output path

        Returns:
            True if saved successfully
        """

    def get_templates(self) -> List[str]:
        """Get available report templates"""
        return ["default"]

    def get_formats(self) -> List[str]:
        """Get supported output formats"""
        return ["markdown", "html", "json"]


class Transform(ABC, Generic[TInput, TOutput]):
    """
    Base Transform abstraction

    Transforms are pure functions for data transformation.
    """

    @abstractmethod
    def apply(self, data: TInput) -> TOutput:
        """
        Apply transformation

        Args:
            data: Input data

        Returns:
            Transformed data
        """

    def __call__(self, data: TInput) -> TOutput:
        """Make transform callable"""
        return self.apply(data)


class Validator(ABC, Generic[TInput]):
    """
    Base Validator abstraction

    Validators ensure data quality and compliance.
    """

    @abstractmethod
    def validate(self, data: TInput) -> Result:
        """
        Validate data

        Args:
            data: Data to validate

        Returns:
            Result with validation status and errors
        """

    def __call__(self, data: TInput) -> bool:
        """Make validator callable"""
        return self.validate(data).success
