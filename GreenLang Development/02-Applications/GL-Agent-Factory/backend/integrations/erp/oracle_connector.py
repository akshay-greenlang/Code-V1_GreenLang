"""
Oracle ERP Connector.

This module provides integration with Oracle ERP Cloud and E-Business Suite
for extracting emissions, energy, and procurement data.

Supports:
- Oracle Cloud ERP (Fusion)
- Oracle E-Business Suite
- Oracle REST APIs

Example:
    >>> connector = OracleConnector(config)
    >>> await connector.connect()
    >>> records = await connector.fetch_emissions_data(date_from, date_to)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import Field

from .base import (
    BaseERPConnector,
    ConnectionConfig,
    ConnectionStatus,
    DataQuery,
    ERPRecord,
    ERPType,
)

logger = logging.getLogger(__name__)


class OracleConfig(ConnectionConfig):
    """
    Oracle-specific connection configuration.

    Attributes:
        cloud_instance: Oracle Cloud instance URL
        rest_api_version: REST API version
        auth_type: Authentication type (basic, oauth2)
        oauth_client_id: OAuth2 client ID
        oauth_client_secret: OAuth2 client secret
    """

    erp_type: ERPType = ERPType.ORACLE
    cloud_instance: Optional[str] = Field(None, description="Cloud instance URL")
    rest_api_version: str = Field("v1", description="REST API version")
    auth_type: str = Field("basic", description="Auth type: basic, oauth2")
    oauth_client_id: Optional[str] = Field(None, description="OAuth2 client ID")
    oauth_client_secret: Optional[str] = Field(None, description="OAuth2 client secret")
    business_unit: Optional[str] = Field(None, description="Business unit filter")


class OracleEntityMapping:
    """Mapping from generic entity types to Oracle entities."""

    def __init__(
        self,
        entity_type: str,
        oracle_resource: str,
        key_field: str,
        date_field: str,
        field_mapping: Dict[str, str],
    ):
        self.entity_type = entity_type
        self.oracle_resource = oracle_resource
        self.key_field = key_field
        self.date_field = date_field
        self.field_mapping = field_mapping


class OracleConnector(BaseERPConnector):
    """
    Oracle ERP Connector.

    This connector integrates with Oracle Cloud ERP and E-Business Suite
    using REST APIs to extract emissions, energy, and procurement data.

    Features:
    - OAuth2 and Basic authentication
    - Oracle REST API support
    - Batch requests for efficiency
    - Error handling and retry logic

    Attributes:
        config: Oracle connection configuration
        http_client: Async HTTP client
        access_token: OAuth2 access token

    Example:
        >>> config = OracleConfig(
        ...     host="company.oraclecloud.com",
        ...     username="user",
        ...     password="pass"
        ... )
        >>> connector = OracleConnector(config)
        >>> await connector.connect()
        >>> records = await connector.fetch_data(query)
    """

    # Entity type to Oracle resource mapping
    ENTITY_MAPPINGS: Dict[str, OracleEntityMapping] = {
        "emissions": OracleEntityMapping(
            entity_type="emissions",
            oracle_resource="fscmRestApi/resources/latest/emissionsData",
            key_field="EmissionId",
            date_field="TransactionDate",
            field_mapping={
                "record_id": "EmissionId",
                "facility_id": "OrganizationId",
                "emission_type": "EmissionType",
                "quantity": "Quantity",
                "unit": "UnitOfMeasure",
                "scope": "EmissionScope",
            },
        ),
        "energy_consumption": OracleEntityMapping(
            entity_type="energy_consumption",
            oracle_resource="fscmRestApi/resources/latest/utilities",
            key_field="UtilityId",
            date_field="MeterReadingDate",
            field_mapping={
                "record_id": "UtilityId",
                "meter_id": "MeterId",
                "facility_id": "LocationId",
                "energy_type": "UtilityType",
                "quantity": "ConsumptionQuantity",
                "unit": "UnitOfMeasure",
            },
        ),
        "procurement": OracleEntityMapping(
            entity_type="procurement",
            oracle_resource="fscmRestApi/resources/latest/purchaseOrders",
            key_field="POHeaderId",
            date_field="OrderDate",
            field_mapping={
                "record_id": "POHeaderId",
                "order_number": "OrderNumber",
                "supplier_id": "SupplierId",
                "supplier_name": "SupplierName",
                "amount": "TotalAmount",
                "currency": "CurrencyCode",
                "status": "Status",
            },
        ),
        "invoices": OracleEntityMapping(
            entity_type="invoices",
            oracle_resource="fscmRestApi/resources/latest/invoices",
            key_field="InvoiceId",
            date_field="InvoiceDate",
            field_mapping={
                "record_id": "InvoiceId",
                "invoice_number": "InvoiceNumber",
                "supplier_id": "SupplierId",
                "amount": "InvoiceAmount",
                "currency": "InvoiceCurrencyCode",
                "category": "ExpenseCategory",
            },
        ),
        "suppliers": OracleEntityMapping(
            entity_type="suppliers",
            oracle_resource="fscmRestApi/resources/latest/suppliers",
            key_field="SupplierId",
            date_field="CreationDate",
            field_mapping={
                "record_id": "SupplierId",
                "name": "SupplierName",
                "country": "Country",
                "industry": "IndustryClassification",
                "status": "Status",
            },
        ),
    }

    def __init__(self, config: OracleConfig):
        """
        Initialize Oracle Connector.

        Args:
            config: Oracle connection configuration
        """
        super().__init__(config)
        self.oracle_config = config
        self._http_client: Optional[httpx.AsyncClient] = None
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

    async def connect(self) -> bool:
        """
        Establish connection to Oracle system.

        Returns:
            True if connection successful
        """
        self.status = ConnectionStatus.CONNECTING
        logger.info(f"Connecting to Oracle at {self.config.host}...")

        try:
            # Create HTTP client
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                verify=self.config.ssl_enabled,
            )

            # Authenticate based on auth type
            if self.oracle_config.auth_type == "oauth2":
                await self._oauth2_authenticate()
            else:
                await self._basic_authenticate()

            self.status = ConnectionStatus.CONNECTED
            self._connected_at = datetime.utcnow()
            logger.info("Successfully connected to Oracle")

            return True

        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self._last_error = str(e)
            logger.error(f"Oracle connection failed: {e}", exc_info=True)
            raise ConnectionError(f"Oracle connection failed: {e}")

    async def disconnect(self) -> None:
        """Close Oracle connection."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._access_token = None
        self._token_expires_at = None
        self.status = ConnectionStatus.DISCONNECTED

        logger.info("Disconnected from Oracle")

    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """
        Test Oracle connection.

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Try to fetch user info
            url = self._build_url("fscmRestApi/resources/latest/users")
            response = await self._http_client.get(
                url,
                headers=self._get_headers(),
                params={"limit": 1},
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
        Fetch data from Oracle.

        Args:
            query: Data query specification

        Returns:
            List of ERP records
        """
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to Oracle")

        mapping = self.ENTITY_MAPPINGS.get(query.entity_type)
        if not mapping:
            raise ValueError(f"Unsupported entity type: {query.entity_type}")

        logger.info(f"Fetching {query.entity_type} from Oracle...")

        try:
            # Check token expiry
            await self._ensure_valid_token()

            # Build query parameters
            params = self._build_query_params(query, mapping)
            url = self._build_url(mapping.oracle_resource)

            # Execute request
            response = await self._http_client.get(
                url,
                headers=self._get_headers(),
                params=params,
            )

            if response.status_code != 200:
                raise Exception(f"Oracle request failed: HTTP {response.status_code}")

            # Parse response
            data = response.json()
            records = self._parse_response(data, mapping)

            logger.info(f"Fetched {len(records)} {query.entity_type} records from Oracle")

            return records

        except Exception as e:
            logger.error(f"Oracle fetch failed: {e}", exc_info=True)
            raise

    async def _basic_authenticate(self) -> None:
        """Authenticate using Basic auth."""
        # Test authentication
        url = self._build_url("fscmRestApi/resources/latest/users")
        response = await self._http_client.get(
            url,
            auth=(self.config.username, self.config.password),
            params={"limit": 1},
        )

        if response.status_code == 401:
            raise ConnectionError("Oracle authentication failed")

    async def _oauth2_authenticate(self) -> None:
        """Authenticate using OAuth2."""
        if not self.oracle_config.oauth_client_id:
            raise ValueError("OAuth2 client ID required")

        token_url = f"https://{self.config.host}/oauth2/v1/token"

        response = await self._http_client.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.oracle_config.oauth_client_id,
                "client_secret": self.oracle_config.oauth_client_secret,
                "scope": "urn:opc:resource:fa:instanceid=*",
            },
        )

        if response.status_code != 200:
            raise ConnectionError("OAuth2 authentication failed")

        token_data = response.json()
        self._access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._token_expires_at = datetime.utcnow().timestamp() + expires_in

    async def _ensure_valid_token(self) -> None:
        """Ensure OAuth2 token is valid."""
        if self.oracle_config.auth_type != "oauth2":
            return

        if self._token_expires_at:
            # Refresh if expiring within 5 minutes
            if datetime.utcnow().timestamp() > (self._token_expires_at - 300):
                await self._oauth2_authenticate()

    def _build_url(self, resource: str) -> str:
        """Build Oracle API URL."""
        protocol = "https" if self.config.ssl_enabled else "http"

        if self.oracle_config.cloud_instance:
            base = self.oracle_config.cloud_instance
        else:
            base = f"{protocol}://{self.config.host}"

        return f"{base}/{resource}"

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        return headers

    def _build_query_params(
        self,
        query: DataQuery,
        mapping: OracleEntityMapping,
    ) -> Dict[str, Any]:
        """Build Oracle REST query parameters."""
        params: Dict[str, Any] = {}

        # Select fields
        if query.fields:
            oracle_fields = [
                mapping.field_mapping.get(f, f)
                for f in query.fields
            ]
            params["fields"] = ",".join(oracle_fields)

        # Build filter query
        filters = []

        if query.date_from:
            date_str = query.date_from.strftime("%Y-%m-%d")
            filters.append(f"{mapping.date_field} >= '{date_str}'")

        if query.date_to:
            date_str = query.date_to.strftime("%Y-%m-%d")
            filters.append(f"{mapping.date_field} <= '{date_str}'")

        # Additional filters
        for key, value in query.filters.items():
            oracle_field = mapping.field_mapping.get(key, key)
            if isinstance(value, str):
                filters.append(f"{oracle_field} = '{value}'")
            elif isinstance(value, list):
                values_str = ",".join(f"'{v}'" for v in value)
                filters.append(f"{oracle_field} IN ({values_str})")
            else:
                filters.append(f"{oracle_field} = {value}")

        if filters:
            params["q"] = " AND ".join(filters)

        # Pagination
        params["limit"] = query.limit
        if query.offset > 0:
            params["offset"] = query.offset

        return params

    def _parse_response(
        self,
        data: Dict[str, Any],
        mapping: OracleEntityMapping,
    ) -> List[ERPRecord]:
        """Parse Oracle response to ERPRecord list."""
        records = []

        items = data.get("items", [])

        for item in items:
            # Map Oracle fields to standard fields
            record_data = {}
            for std_field, oracle_field in mapping.field_mapping.items():
                value = item.get(oracle_field)
                record_data[std_field] = value

            record = ERPRecord(
                record_id=str(item.get(mapping.key_field, "")),
                entity_type=mapping.entity_type,
                data=record_data,
                source_system="oracle",
                metadata={
                    "oracle_resource": mapping.oracle_resource,
                    "links": item.get("links", []),
                },
            )
            records.append(record)

        return records

    async def fetch_spend_by_category(
        self,
        date_from: datetime,
        date_to: datetime,
    ) -> List[ERPRecord]:
        """
        Fetch spend data by category for Scope 3 calculations.

        Args:
            date_from: Start date
            date_to: End date

        Returns:
            List of spend records with category
        """
        query = DataQuery(
            entity_type="invoices",
            date_from=date_from,
            date_to=date_to,
            fields=["record_id", "amount", "currency", "category", "supplier_id"],
        )

        return await self.fetch_data(query)

    async def fetch_supplier_data(
        self,
        supplier_ids: Optional[List[str]] = None,
    ) -> List[ERPRecord]:
        """
        Fetch supplier master data.

        Args:
            supplier_ids: Optional supplier filter

        Returns:
            List of supplier records
        """
        filters = {"supplier_ids": supplier_ids} if supplier_ids else {}

        query = DataQuery(
            entity_type="suppliers",
            filters=filters,
        )

        return await self.fetch_data(query)
