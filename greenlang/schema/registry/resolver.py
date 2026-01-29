"""
Schema Resolver Interface for GL-FOUND-X-002.

This module defines the abstract interface for schema resolution.

TODO Task 6.1:
    - Define SchemaRegistry protocol
    - Implement resolver factory
"""

from typing import Any, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod


class SchemaSource:
    """Container for resolved schema source."""

    def __init__(
        self,
        schema_id: str,
        version: str,
        content: Dict[str, Any],
        source_uri: str,
    ):
        self.schema_id = schema_id
        self.version = version
        self.content = content
        self.source_uri = source_uri


class SchemaRegistry(ABC):
    """
    Abstract base class for schema registries.

    Defines the interface that all registry implementations must follow.
    """

    @abstractmethod
    def resolve(self, schema_id: str, version: str) -> SchemaSource:
        """Resolve schema by ID and version."""
        pass

    @abstractmethod
    def list_versions(self, schema_id: str) -> List[str]:
        """List available versions for a schema."""
        pass

    @abstractmethod
    def get_latest(
        self,
        schema_id: str,
        constraint: Optional[str] = None,
    ) -> str:
        """Get latest version matching constraint."""
        pass
