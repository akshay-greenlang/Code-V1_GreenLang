"""
Base Infrastructure Components
===============================

Base classes for all infrastructure components in GreenLang.

Author: Infrastructure Team
Created: 2025-11-21
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Status of an infrastructure component."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class InfrastructureConfig:
    """Configuration for infrastructure components."""
    component_name: str
    enabled: bool = True
    debug_mode: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseInfrastructureComponent(ABC):
    """
    Abstract base class for all infrastructure components.

    Provides common functionality for initialization, lifecycle management,
    and status tracking.
    """

    def __init__(self, config: Optional[InfrastructureConfig] = None):
        """Initialize infrastructure component."""
        self.config = config or InfrastructureConfig(
            component_name=self.__class__.__name__
        )
        self.status = ComponentStatus.INITIALIZING
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self._metrics: Dict[str, Any] = {}

        logger.info(f"Initializing {self.config.component_name}")
        self._initialize()
        self.status = ComponentStatus.READY

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize component-specific resources."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start the component."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the component."""
        pass

    def get_status(self) -> ComponentStatus:
        """Get current component status."""
        return self.status

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()

    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics."""
        return {
            "component": self.config.component_name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            **self._metrics
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            is_healthy = self.status in [ComponentStatus.READY, ComponentStatus.RUNNING]
            return {
                "healthy": is_healthy,
                "status": self.status.value,
                "message": f"{self.config.component_name} is {self.status.value}"
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "healthy": False,
                "status": ComponentStatus.ERROR.value,
                "message": str(e)
            }