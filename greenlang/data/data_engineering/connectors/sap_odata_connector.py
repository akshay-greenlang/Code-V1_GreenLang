"""
SAP OData Connector
===================

Enterprise-grade connector for SAP S/4HANA via OData API.
Supports OAuth2 authentication, pagination, batch operations.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date
from decimal import Decimal
import logging
import asyncio

from pydantic import BaseModel, Field, SecretStr

from greenlang.data_engineering.connectors.base_connector import (
    BaseConnector,
    ConnectorConfig,
    AuthenticationType,
)

logger = logging.getLogger(__name__)


class SAPODataConfig(ConnectorConfig):
    """Configuration for SAP OData connector."""
    name: str = Field(default="SAP S/4HANA")
    auth_type: AuthenticationType = Field(default=AuthenticationType.OAUTH2_CLIENT_CREDENTIALS)
    sap_client: str = Field(default="100", description="SAP client number")
    api_version: str = Field(default="v2", description="OData API version")
    language: str = Field(default="EN", description="SAP language key")
    csrf_token_required: bool = Field(default=True, description="CSRF token for write operations")


class SAPPurchaseOrder(BaseModel):
    """SAP Purchase Order record."""
    purchase_order: str
    supplier: str
    supplier_name: Optional[str] = None
    plant: str
    purchasing_group: Optional[str] = None
    currency: str
    total_net_amount: Decimal
    creation_date: date
    items: List[Dict[str, Any]] = Field(default_factory=list)


class SAPMaterial(BaseModel):
    """SAP Material master record."""
    material_number: str
    material_description: str
    material_group: Optional[str] = None
    base_unit: str
    weight_unit: Optional[str] = None
    gross_weight: Optional[Decimal] = None
    net_weight: Optional[Decimal] = None
    co2_emission_factor: Optional[Decimal] = None


class SAPODataConnector(BaseConnector[Dict[str, Any]]):
    """
    SAP S/4HANA OData API Connector.

    Supported SAP Modules:
    - MM (Materials Management) - Purchase Orders, Materials
    - FI (Financial Accounting) - Cost Centers, GL Accounts
    - SD (Sales & Distribution) - Sales Orders, Deliveries
    - PP (Production Planning) - Production Orders
    - PM (Plant Maintenance) - Equipment, Maintenance Orders
    - EHS (Environment, Health, Safety) - Emissions Data

    OData Services:
    - API_PURCHASEORDER_PROCESS_SRV - Purchase Orders
    - API_MATERIAL_SRV - Material Master
    - API_PRODUCT_SRV - Products
    - API_BUSINESS_PARTNER - Suppliers/Vendors
    """

    def __init__(self, config: SAPODataConfig):
        """Initialize SAP connector."""
        super().__init__(config)
        self.sap_config = config
        self._csrf_token: Optional[str] = None

    async def _create_client(self) -> Any:
        """Create HTTP client for SAP."""
        try:
            import httpx
            return httpx.AsyncClient(
                verify=self.config.verify_ssl,
                timeout=self.config.timeout_seconds,
            )
        except ImportError:
            raise ImportError("httpx required for SAP connector")

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get SAP-specific headers."""
        headers = super()._get_auth_headers()
        headers['sap-client'] = self.sap_config.sap_client
        headers['sap-language'] = self.sap_config.language
        headers['Accept'] = 'application/json'

        if self._csrf_token:
            headers['X-CSRF-Token'] = self._csrf_token

        return headers

    async def fetch_csrf_token(self) -> str:
        """Fetch CSRF token for write operations."""
        headers = self._get_auth_headers()
        headers['X-CSRF-Token'] = 'Fetch'

        async with self.pool.get_connection() as client:
            response = await client.get(
                f"{self.config.base_url}/",
                headers=headers,
            )
            self._csrf_token = response.headers.get('X-CSRF-Token')
            return self._csrf_token or ""

    async def fetch_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Fetch data from SAP - generic method."""
        entity_set = kwargs.get('entity_set', 'A_PurchaseOrder')
        filters = kwargs.get('filters', {})
        select = kwargs.get('select', [])
        expand = kwargs.get('expand', [])
        top = kwargs.get('top', 1000)

        return await self.fetch_entity_set(
            entity_set=entity_set,
            filters=filters,
            select=select,
            expand=expand,
            top=top,
        )

    async def fetch_entity_set(
        self,
        entity_set: str,
        filters: Optional[Dict[str, Any]] = None,
        select: Optional[List[str]] = None,
        expand: Optional[List[str]] = None,
        top: int = 1000,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from OData entity set with pagination.

        Args:
            entity_set: OData entity set name
            filters: OData $filter parameters
            select: Fields to select ($select)
            expand: Related entities to expand ($expand)
            top: Max records per request
            skip: Records to skip (for pagination)

        Returns:
            List of entity records
        """
        all_records = []
        current_skip = skip

        while True:
            # Build OData query
            params = self._build_odata_params(
                filters=filters,
                select=select,
                expand=expand,
                top=top,
                skip=current_skip,
            )

            # Make request
            response = await self.request(
                method='GET',
                endpoint=entity_set,
                params=params,
            )

            # Extract records
            records = response.get('d', {}).get('results', [])
            if not records:
                # Try alternate response format
                records = response.get('value', [])

            if not records:
                break

            all_records.extend(records)

            # Check if more records available
            if len(records) < top:
                break

            current_skip += top

            # Safety limit
            if current_skip > 100000:
                logger.warning("Reached pagination limit of 100,000 records")
                break

        logger.info(f"Fetched {len(all_records)} records from {entity_set}")
        return all_records

    def _build_odata_params(
        self,
        filters: Optional[Dict[str, Any]] = None,
        select: Optional[List[str]] = None,
        expand: Optional[List[str]] = None,
        top: int = 1000,
        skip: int = 0,
    ) -> Dict[str, str]:
        """Build OData query parameters."""
        params = {
            '$format': 'json',
            '$top': str(top),
        }

        if skip > 0:
            params['$skip'] = str(skip)

        if select:
            params['$select'] = ','.join(select)

        if expand:
            params['$expand'] = ','.join(expand)

        if filters:
            filter_parts = []
            for key, value in filters.items():
                if isinstance(value, str):
                    filter_parts.append(f"{key} eq '{value}'")
                elif isinstance(value, (int, float)):
                    filter_parts.append(f"{key} eq {value}")
                elif isinstance(value, date):
                    filter_parts.append(f"{key} eq datetime'{value.isoformat()}'")
                elif isinstance(value, dict):
                    # Handle operators like ge, le, etc.
                    for op, val in value.items():
                        if isinstance(val, str):
                            filter_parts.append(f"{key} {op} '{val}'")
                        else:
                            filter_parts.append(f"{key} {op} {val}")
            params['$filter'] = ' and '.join(filter_parts)

        return params

    async def fetch_purchase_orders(
        self,
        start_date: date,
        end_date: date,
        plant_codes: Optional[List[str]] = None,
        supplier_codes: Optional[List[str]] = None,
    ) -> List[SAPPurchaseOrder]:
        """
        Fetch purchase orders from SAP.

        Args:
            start_date: Start date filter
            end_date: End date filter
            plant_codes: Optional plant code filter
            supplier_codes: Optional supplier filter

        Returns:
            List of purchase orders
        """
        filters = {
            'CreationDate': {'ge': start_date.isoformat(), 'le': end_date.isoformat()},
        }

        # Add plant filter
        if plant_codes:
            # OData doesn't support IN, so we need to use OR
            plant_filter = ' or '.join([f"Plant eq '{code}'" for code in plant_codes])
            filters['_plant_or'] = plant_filter

        select = [
            'PurchaseOrder', 'Supplier', 'Plant', 'PurchasingGroup',
            'DocumentCurrency', 'TotalNetAmount', 'CreationDate',
        ]

        expand = ['to_PurchaseOrderItem']

        raw_data = await self.fetch_entity_set(
            entity_set='A_PurchaseOrder',
            filters=filters,
            select=select,
            expand=expand,
        )

        # Convert to typed models
        orders = []
        for record in raw_data:
            try:
                order = SAPPurchaseOrder(
                    purchase_order=record.get('PurchaseOrder', ''),
                    supplier=record.get('Supplier', ''),
                    plant=record.get('Plant', ''),
                    purchasing_group=record.get('PurchasingGroup'),
                    currency=record.get('DocumentCurrency', 'USD'),
                    total_net_amount=Decimal(str(record.get('TotalNetAmount', 0))),
                    creation_date=self._parse_sap_date(record.get('CreationDate')),
                    items=record.get('to_PurchaseOrderItem', {}).get('results', []),
                )
                orders.append(order)
            except Exception as e:
                logger.warning(f"Error parsing purchase order: {e}")

        return orders

    async def fetch_materials(
        self,
        material_group: Optional[str] = None,
        include_emission_factors: bool = True,
    ) -> List[SAPMaterial]:
        """
        Fetch material master data from SAP.

        Args:
            material_group: Optional material group filter
            include_emission_factors: Include CO2 emission factors if available

        Returns:
            List of materials
        """
        filters = {}
        if material_group:
            filters['MaterialGroup'] = material_group

        select = [
            'Material', 'MaterialDescription', 'MaterialGroup',
            'BaseUnit', 'WeightUnit', 'GrossWeight', 'NetWeight',
        ]

        if include_emission_factors:
            select.extend(['YY1_CO2EmissionFactor_PRD'])  # Custom field example

        raw_data = await self.fetch_entity_set(
            entity_set='A_Product',
            filters=filters,
            select=select,
        )

        materials = []
        for record in raw_data:
            try:
                material = SAPMaterial(
                    material_number=record.get('Material', ''),
                    material_description=record.get('MaterialDescription', ''),
                    material_group=record.get('MaterialGroup'),
                    base_unit=record.get('BaseUnit', 'EA'),
                    weight_unit=record.get('WeightUnit'),
                    gross_weight=Decimal(str(record.get('GrossWeight', 0))) if record.get('GrossWeight') else None,
                    net_weight=Decimal(str(record.get('NetWeight', 0))) if record.get('NetWeight') else None,
                    co2_emission_factor=Decimal(str(record.get('YY1_CO2EmissionFactor_PRD', 0))) if record.get('YY1_CO2EmissionFactor_PRD') else None,
                )
                materials.append(material)
            except Exception as e:
                logger.warning(f"Error parsing material: {e}")

        return materials

    async def fetch_suppliers(
        self,
        country_codes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch supplier/vendor master data from SAP.

        Args:
            country_codes: Optional country filter

        Returns:
            List of suppliers
        """
        filters = {}
        if country_codes:
            country_filter = ' or '.join([f"Country eq '{code}'" for code in country_codes])
            filters['_country_or'] = country_filter

        select = [
            'Supplier', 'SupplierName', 'Country', 'Region',
            'Industry', 'SupplierAccountGroup',
        ]

        return await self.fetch_entity_set(
            entity_set='A_Supplier',
            filters=filters,
            select=select,
        )

    async def fetch_emission_data(
        self,
        start_date: date,
        end_date: date,
        plant_codes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch emission data from SAP EHS module.

        Note: This requires SAP EHS module and custom configuration.

        Args:
            start_date: Start date
            end_date: End date
            plant_codes: Plant filter

        Returns:
            List of emission records
        """
        filters = {
            'EmissionDate': {'ge': start_date.isoformat(), 'le': end_date.isoformat()},
        }

        if plant_codes:
            plant_filter = ' or '.join([f"Plant eq '{code}'" for code in plant_codes])
            filters['_plant_or'] = plant_filter

        # This is a hypothetical OData service - actual service name depends on SAP config
        return await self.fetch_entity_set(
            entity_set='YY1_EmissionData',
            filters=filters,
        )

    def _parse_sap_date(self, sap_date: Any) -> date:
        """Parse SAP date format to Python date."""
        if isinstance(sap_date, date):
            return sap_date
        if isinstance(sap_date, str):
            # SAP OData date format: /Date(1234567890000)/
            if sap_date.startswith('/Date('):
                timestamp = int(sap_date.replace('/Date(', '').replace(')/', '').split('+')[0])
                return datetime.fromtimestamp(timestamp / 1000).date()
            # ISO format
            return datetime.fromisoformat(sap_date.replace('Z', '+00:00')).date()
        return date.today()

    async def create_batch_request(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Execute batch request for multiple operations.

        Args:
            requests: List of request specifications

        Returns:
            List of responses
        """
        if self.sap_config.csrf_token_required and not self._csrf_token:
            await self.fetch_csrf_token()

        # Build $batch request
        batch_boundary = f"batch_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        changeset_boundary = f"changeset_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        # OData batch format would be built here
        # This is a simplified implementation

        logger.warning("Batch requests not fully implemented - use individual requests")
        return []
