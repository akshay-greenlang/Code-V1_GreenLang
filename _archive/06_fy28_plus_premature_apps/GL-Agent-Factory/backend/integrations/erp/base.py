"""
Base ERP Connector Interface.

This module provides the abstract base class for all ERP integrations,
ensuring consistent interface and error handling across connectors.

Example:
    >>> class MyERPConnector(BaseERPConnector):
    ...     async def connect(self) -> bool:
    ...         # Implementation
    ...     async def fetch_data(self, query: DataQuery) -> List[Dict]:
    ...         # Implementation
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ERPType(str, Enum):
    """Supported ERP systems."""

    SAP = "sap"
    ORACLE = "oracle"
    DYNAMICS = "dynamics"
    NETSUITE = "netsuite"
    SAGE = "sage"
    CUSTOM = "custom"


class ConnectionStatus(str, Enum):
    """Connection status states."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"


class DataQuery(BaseModel):
    """
    Query specification for ERP data extraction.

    Attributes:
        entity_type: Type of data to extract
        filters: Filter conditions
        fields: Fields to retrieve
        date_range: Date range filter
        limit: Maximum records
        offset: Pagination offset
    """

    entity_type: str = Field(..., description="Entity type (e.g., 'emissions', 'invoices')")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filter conditions")
    fields: List[str] = Field(default_factory=list, description="Fields to retrieve")
    date_from: Optional[datetime] = Field(None, description="Start date")
    date_to: Optional[datetime] = Field(None, description="End date")
    limit: int = Field(1000, ge=1, le=10000, description="Max records")
    offset: int = Field(0, ge=0, description="Pagination offset")


class ERPRecord(BaseModel):
    """
    Standardized ERP record format.

    Attributes:
        record_id: Unique record identifier
        entity_type: Type of entity
        data: Record data
        source_system: Source ERP system
        extracted_at: Extraction timestamp
    """

    record_id: str = Field(..., description="Record ID")
    entity_type: str = Field(..., description="Entity type")
    data: Dict[str, Any] = Field(..., description="Record data")
    source_system: str = Field(..., description="Source ERP")
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConnectionConfig(BaseModel):
    """
    ERP connection configuration.

    Attributes:
        erp_type: Type of ERP system
        host: Server hostname or URL
        port: Connection port
        username: Authentication username
        password: Authentication password
        client_id: Client/tenant ID
        database: Database name
        timeout_seconds: Connection timeout
        ssl_enabled: Whether to use SSL
    """

    erp_type: ERPType = Field(..., description="ERP system type")
    host: str = Field(..., description="Server hostname")
    port: int = Field(443, description="Connection port")
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    client_id: Optional[str] = Field(None, description="Client/tenant ID")
    database: Optional[str] = Field(None, description="Database name")
    timeout_seconds: int = Field(30, description="Connection timeout")
    ssl_enabled: bool = Field(True, description="Use SSL")
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class BaseERPConnector(ABC):
    """
    Abstract base class for ERP connectors.

    All ERP integrations must implement this interface to ensure
    consistent behavior and error handling.

    Attributes:
        config: Connection configuration
        status: Current connection status
        erp_type: Type of ERP system

    Subclasses must implement:
        - connect()
        - disconnect()
        - fetch_data()
        - test_connection()
    """

    def __init__(self, config: ConnectionConfig):
        """
        Initialize the ERP connector.

        Args:
            config: Connection configuration
        """
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self.erp_type = config.erp_type
        self._last_error: Optional[str] = None
        self._connected_at: Optional[datetime] = None

        logger.info(f"Initialized {self.erp_type.value} connector for {config.host}")

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the ERP system.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close the connection to the ERP system.
        """
        pass

    @abstractmethod
    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """
        Test the ERP connection.

        Returns:
            Tuple of (success, error_message)
        """
        pass

    @abstractmethod
    async def fetch_data(
        self,
        query: DataQuery,
    ) -> List[ERPRecord]:
        """
        Fetch data from the ERP system.

        Args:
            query: Data query specification

        Returns:
            List of ERP records

        Raises:
            ValueError: If query is invalid
            ConnectionError: If not connected
        """
        pass

    async def fetch_emissions_data(
        self,
        date_from: datetime,
        date_to: datetime,
        facility_ids: Optional[List[str]] = None,
    ) -> List[ERPRecord]:
        """
        Fetch emissions-related data.

        Args:
            date_from: Start date
            date_to: End date
            facility_ids: Optional facility filter

        Returns:
            List of emissions records
        """
        filters = {"facility_ids": facility_ids} if facility_ids else {}

        query = DataQuery(
            entity_type="emissions",
            filters=filters,
            date_from=date_from,
            date_to=date_to,
        )

        return await self.fetch_data(query)

    async def fetch_energy_consumption(
        self,
        date_from: datetime,
        date_to: datetime,
        meter_ids: Optional[List[str]] = None,
    ) -> List[ERPRecord]:
        """
        Fetch energy consumption data.

        Args:
            date_from: Start date
            date_to: End date
            meter_ids: Optional meter filter

        Returns:
            List of energy consumption records
        """
        filters = {"meter_ids": meter_ids} if meter_ids else {}

        query = DataQuery(
            entity_type="energy_consumption",
            filters=filters,
            date_from=date_from,
            date_to=date_to,
        )

        return await self.fetch_data(query)

    async def fetch_procurement_data(
        self,
        date_from: datetime,
        date_to: datetime,
        supplier_ids: Optional[List[str]] = None,
    ) -> List[ERPRecord]:
        """
        Fetch procurement/purchase data for Scope 3.

        Args:
            date_from: Start date
            date_to: End date
            supplier_ids: Optional supplier filter

        Returns:
            List of procurement records
        """
        filters = {"supplier_ids": supplier_ids} if supplier_ids else {}

        query = DataQuery(
            entity_type="procurement",
            filters=filters,
            date_from=date_from,
            date_to=date_to,
        )

        return await self.fetch_data(query)

    def get_status(self) -> Dict[str, Any]:
        """
        Get current connection status.

        Returns:
            Status dictionary
        """
        return {
            "erp_type": self.erp_type.value,
            "host": self.config.host,
            "status": self.status.value,
            "connected_at": self._connected_at.isoformat() if self._connected_at else None,
            "last_error": self._last_error,
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the connection.

        Returns:
            Health check result
        """
        try:
            success, error = await self.test_connection()
            return {
                "healthy": success,
                "status": self.status.value,
                "error": error,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "healthy": False,
                "status": ConnectionStatus.ERROR.value,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
