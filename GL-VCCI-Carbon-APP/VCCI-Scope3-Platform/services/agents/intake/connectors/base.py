"""Base ERP Connector - Abstract base class for ERP integrations."""
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseConnector(ABC):
    """Abstract base class for ERP connectors."""
    
    def __init__(self, credentials: Dict[str, str]):
        self.credentials = credentials
        self.connected = False
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to ERP system."""
        pass
    
    @abstractmethod
    def query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute query and return results."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        pass

__all__ = ["BaseConnector"]
