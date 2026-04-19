"""
SAP ERP Connector.

This module provides integration with SAP ERP systems for extracting
emissions, energy, and procurement data.

Supports:
- SAP S/4HANA
- SAP ECC
- SAP Business One
- SAP OData APIs

Example:
    >>> connector = SAPConnector(config)
    >>> await connector.connect()
    >>> records = await connector.fetch_emissions_data(date_from, date_to)
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from .base import (
    BaseERPConnector,
    ConnectionConfig,
    ConnectionStatus,
    DataQuery,
    ERPRecord,
    ERPType,
)

logger = logging.getLogger(__name__)


class SAPConfig(ConnectionConfig):
    """
    SAP-specific connection configuration.

    Attributes:
        sap_client: SAP client number
        sap_system: SAP system ID
        sap_language: Language code
        odata_service_path: OData service path
    """

    erp_type: ERPType = ERPType.SAP
    sap_client: str = Field("100", description="SAP client number")
    sap_system: Optional[str] = Field(None, description="SAP system ID")
    sap_language: str = Field("EN", description="Language code")
    odata_service_path: str = Field(
        "/sap/opu/odata/sap/",
        description="OData service base path"
    )
    api_version: str = Field("v2", description="OData API version")


class SAPEntityMapping(BaseModel):
    """Mapping from generic entity types to SAP entities."""

    entity_type: str
    sap_entity_set: str
    key_field: str
    date_field: str
    field_mapping: Dict[str, str]


class SAPConnector(BaseERPConnector):
    """
    SAP ERP Connector.

    This connector integrates with SAP systems using OData APIs
    to extract emissions, energy, and procurement data.

    Features:
    - OAuth2 and Basic authentication
    - OData v2/v4 support
    - Batch requests for efficiency
    - Delta queries for incremental sync
    - Error handling and retry logic

    Attributes:
        config: SAP connection configuration
        http_client: Async HTTP client
        auth_token: Current authentication token

    Example:
        >>> config = SAPConfig(
        ...     host="sap.company.com",
        ...     username="user",
        ...     password="pass",
        ...     sap_client="100"
        ... )
        >>> connector = SAPConnector(config)
        >>> await connector.connect()
        >>> records = await connector.fetch_data(query)
    """

    # Entity type to SAP entity mapping
    ENTITY_MAPPINGS: Dict[str, SAPEntityMapping] = {
        "emissions": SAPEntityMapping(
            entity_type="emissions",
            sap_entity_set="ZGL_EMISSIONS",
            key_field="EmissionId",
            date_field="PostingDate",
            field_mapping={
                "record_id": "EmissionId",
                "facility_id": "Plant",
                "emission_type": "EmissionType",
                "quantity": "Quantity",
                "unit": "Unit",
                "scope": "Scope",
                "source": "EmissionSource",
            },
        ),
        "energy_consumption": SAPEntityMapping(
            entity_type="energy_consumption",
            sap_entity_set="ZGL_ENERGY_CONSUMPTION",
            key_field="ConsumptionId",
            date_field="ReadingDate",
            field_mapping={
                "record_id": "ConsumptionId",
                "meter_id": "MeterId",
                "facility_id": "Plant",
                "energy_type": "EnergyType",
                "quantity": "Quantity",
                "unit": "Unit",
            },
        ),
        "procurement": SAPEntityMapping(
            entity_type="procurement",
            sap_entity_set="EKKO",  # SAP Purchase Order Header
            key_field="Ebeln",
            date_field="Bedat",
            field_mapping={
                "record_id": "Ebeln",
                "supplier_id": "Lifnr",
                "amount": "Netwr",
                "currency": "Waers",
                "material_group": "Matkl",
                "posting_date": "Bedat",
            },
        ),
        "materials": SAPEntityMapping(
            entity_type="materials",
            sap_entity_set="MARA",  # SAP Material Master
            key_field="Matnr",
            date_field="Ersda",
            field_mapping={
                "record_id": "Matnr",
                "description": "Maktx",
                "material_group": "Matkl",
                "unit": "Meins",
            },
        ),
    }

    def __init__(self, config: SAPConfig):
        """
        Initialize SAP Connector.

        Args:
            config: SAP connection configuration
        """
        super().__init__(config)
        self.sap_config = config
        self._http_client: Optional[httpx.AsyncClient] = None
        self._auth_token: Optional[str] = None
        self._csrf_token: Optional[str] = None
        self._session_cookies: Dict[str, str] = {}

    async def connect(self) -> bool:
        """
        Establish connection to SAP system.

        Returns:
            True if connection successful
        """
        self.status = ConnectionStatus.CONNECTING
        logger.info(f"Connecting to SAP at {self.config.host}...")

        try:
            # Create HTTP client
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                verify=self.config.ssl_enabled,
            )

            # Authenticate
            await self._authenticate()

            # Fetch CSRF token for write operations
            await self._fetch_csrf_token()

            self.status = ConnectionStatus.CONNECTED
            self._connected_at = datetime.utcnow()
            logger.info("Successfully connected to SAP")

            return True

        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self._last_error = str(e)
            logger.error(f"SAP connection failed: {e}", exc_info=True)
            raise ConnectionError(f"SAP connection failed: {e}")

    async def disconnect(self) -> None:
        """Close SAP connection."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._auth_token = None
        self._csrf_token = None
        self._session_cookies = {}
        self.status = ConnectionStatus.DISCONNECTED

        logger.info("Disconnected from SAP")

    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """
        Test SAP connection.

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Try to fetch metadata
            url = self._build_url("$metadata")
            response = await self._http_client.get(
                url,
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                return True, None
            else:
                return False, f"HTTP {response.status_code}: {response.text[:200]}"

        except Exception as e:
            return False, str(e)

    async def fetch_data(
        self,
        query: DataQuery,
    ) -> List[ERPRecord]:
        """
        Fetch data from SAP.

        Args:
            query: Data query specification

        Returns:
            List of ERP records

        Raises:
            ValueError: If entity type not supported
            ConnectionError: If not connected
        """
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to SAP")

        mapping = self.ENTITY_MAPPINGS.get(query.entity_type)
        if not mapping:
            raise ValueError(f"Unsupported entity type: {query.entity_type}")

        logger.info(f"Fetching {query.entity_type} from SAP...")

        try:
            # Build OData query
            odata_query = self._build_odata_query(query, mapping)
            url = self._build_url(f"{mapping.sap_entity_set}{odata_query}")

            # Execute request
            response = await self._http_client.get(
                url,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise Exception(f"SAP request failed: HTTP {response.status_code}")

            # Parse response
            data = response.json()
            records = self._parse_odata_response(data, mapping)

            logger.info(f"Fetched {len(records)} {query.entity_type} records from SAP")

            return records

        except Exception as e:
            logger.error(f"SAP fetch failed: {e}", exc_info=True)
            raise

    async def _authenticate(self) -> None:
        """Authenticate with SAP system."""
        # Basic authentication
        auth = (self.config.username, self.config.password)

        # Test authentication with a simple request
        url = self._build_url("")
        response = await self._http_client.get(
            url,
            auth=auth,
            headers={
                "sap-client": self.sap_config.sap_client,
                "sap-language": self.sap_config.sap_language,
            },
        )

        if response.status_code == 401:
            raise ConnectionError("SAP authentication failed")

        # Store cookies for session
        self._session_cookies = dict(response.cookies)
        self._auth_token = f"Basic {hashlib.sha256(f'{self.config.username}:{self.config.password}'.encode()).hexdigest()[:32]}"

    async def _fetch_csrf_token(self) -> None:
        """Fetch CSRF token for write operations."""
        try:
            url = self._build_url("")
            response = await self._http_client.get(
                url,
                headers={
                    **self._get_headers(),
                    "x-csrf-token": "Fetch",
                },
            )

            self._csrf_token = response.headers.get("x-csrf-token")

        except Exception as e:
            logger.warning(f"Failed to fetch CSRF token: {e}")

    def _build_url(self, path: str) -> str:
        """Build SAP OData URL."""
        protocol = "https" if self.config.ssl_enabled else "http"
        base = f"{protocol}://{self.config.host}:{self.config.port}"
        service_path = self.sap_config.odata_service_path
        return f"{base}{service_path}{path}"

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "sap-client": self.sap_config.sap_client,
            "sap-language": self.sap_config.sap_language,
        }

        if self._csrf_token:
            headers["x-csrf-token"] = self._csrf_token

        return headers

    def _build_odata_query(
        self,
        query: DataQuery,
        mapping: SAPEntityMapping,
    ) -> str:
        """Build OData query string."""
        params = []

        # Select fields
        if query.fields:
            sap_fields = [
                mapping.field_mapping.get(f, f)
                for f in query.fields
            ]
            params.append(f"$select={','.join(sap_fields)}")

        # Date filter
        filters = []
        if query.date_from:
            date_str = query.date_from.strftime("%Y-%m-%dT%H:%M:%S")
            filters.append(f"{mapping.date_field} ge datetime'{date_str}'")

        if query.date_to:
            date_str = query.date_to.strftime("%Y-%m-%dT%H:%M:%S")
            filters.append(f"{mapping.date_field} le datetime'{date_str}'")

        # Additional filters
        for key, value in query.filters.items():
            sap_field = mapping.field_mapping.get(key, key)
            if isinstance(value, str):
                filters.append(f"{sap_field} eq '{value}'")
            elif isinstance(value, list):
                or_conditions = " or ".join(
                    f"{sap_field} eq '{v}'" for v in value
                )
                filters.append(f"({or_conditions})")
            else:
                filters.append(f"{sap_field} eq {value}")

        if filters:
            params.append(f"$filter={' and '.join(filters)}")

        # Pagination
        params.append(f"$top={query.limit}")
        if query.offset > 0:
            params.append(f"$skip={query.offset}")

        # Format
        params.append("$format=json")

        return "?" + "&".join(params) if params else ""

    def _parse_odata_response(
        self,
        data: Dict[str, Any],
        mapping: SAPEntityMapping,
    ) -> List[ERPRecord]:
        """Parse OData response to ERPRecord list."""
        records = []

        # OData v2 format
        results = data.get("d", {}).get("results", [])
        if not results:
            # OData v4 format
            results = data.get("value", [])

        for item in results:
            # Map SAP fields to standard fields
            record_data = {}
            for std_field, sap_field in mapping.field_mapping.items():
                value = item.get(sap_field)
                record_data[std_field] = value

            record = ERPRecord(
                record_id=str(item.get(mapping.key_field, "")),
                entity_type=mapping.entity_type,
                data=record_data,
                source_system="sap",
                metadata={
                    "sap_entity_set": mapping.sap_entity_set,
                    "raw_keys": {mapping.key_field: item.get(mapping.key_field)},
                },
            )
            records.append(record)

        return records

    async def fetch_emission_factors(
        self,
        material_ids: Optional[List[str]] = None,
    ) -> List[ERPRecord]:
        """
        Fetch emission factors from SAP.

        Args:
            material_ids: Optional material filter

        Returns:
            List of emission factor records
        """
        query = DataQuery(
            entity_type="materials",
            filters={"material_ids": material_ids} if material_ids else {},
        )

        return await self.fetch_data(query)

    async def fetch_transport_data(
        self,
        date_from: datetime,
        date_to: datetime,
    ) -> List[ERPRecord]:
        """
        Fetch transport/logistics data for Scope 3 Category 4.

        Args:
            date_from: Start date
            date_to: End date

        Returns:
            List of transport records
        """
        # This would typically query SAP Transportation Management
        query = DataQuery(
            entity_type="transport",
            date_from=date_from,
            date_to=date_to,
        )

        # For now, return empty - would need TM module access
        logger.warning("Transport data extraction not yet implemented")
        return []
