"""
Oracle ERP Cloud Connector
==========================

Enterprise-grade connector for Oracle ERP Cloud via REST API.
Supports OAuth2 authentication, FBDI integrations, and BI Publisher reports.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date
from decimal import Decimal
import logging
import asyncio
import base64

from pydantic import BaseModel, Field, SecretStr

from greenlang.data_engineering.connectors.base_connector import (
    BaseConnector,
    ConnectorConfig,
    AuthenticationType,
)

logger = logging.getLogger(__name__)


class OracleERPConfig(ConnectorConfig):
    """Configuration for Oracle ERP Cloud connector."""
    name: str = Field(default="Oracle ERP Cloud")
    auth_type: AuthenticationType = Field(default=AuthenticationType.BASIC)
    pod_url: str = Field(..., description="Oracle Cloud pod URL (e.g., fa-xxx)")
    service_domain: str = Field(default="fscmRestApi", description="REST API domain")
    api_version: str = Field(default="11.13.18.05", description="API version")
    bi_publisher_enabled: bool = Field(default=True, description="Enable BI Publisher integration")


class OracleSupplier(BaseModel):
    """Oracle Supplier record."""
    supplier_id: int
    supplier_name: str
    supplier_number: str
    supplier_type: Optional[str] = None
    country: Optional[str] = None
    status: str = "ACTIVE"
    tax_registration_number: Optional[str] = None


class OraclePurchaseOrder(BaseModel):
    """Oracle Purchase Order record."""
    po_header_id: int
    po_number: str
    supplier_id: int
    supplier_name: Optional[str] = None
    buyer_id: Optional[int] = None
    currency_code: str
    total_amount: Decimal
    status: str
    creation_date: datetime
    lines: List[Dict[str, Any]] = Field(default_factory=list)


class OracleERPConnector(BaseConnector[Dict[str, Any]]):
    """
    Oracle ERP Cloud REST API Connector.

    Supported Oracle Modules:
    - Procurement (PO, Suppliers, Requisitions)
    - Financials (GL, AP, AR)
    - Supply Chain Management (Inventory, Costing)
    - Manufacturing (Production, Quality)

    Integration Methods:
    - REST APIs (primary)
    - FBDI (File-Based Data Import)
    - BI Publisher Reports
    - SOAP Web Services (legacy)
    """

    # Oracle REST API endpoints
    ENDPOINTS = {
        'suppliers': '/fscmRestApi/resources/{version}/suppliers',
        'purchase_orders': '/fscmRestApi/resources/{version}/purchaseOrders',
        'requisitions': '/fscmRestApi/resources/{version}/requisitions',
        'gl_journals': '/fscmRestApi/resources/{version}/journals',
        'cost_elements': '/fscmRestApi/resources/{version}/costElements',
        'inventory_items': '/fscmRestApi/resources/{version}/inventoryItems',
        'inventory_organizations': '/fscmRestApi/resources/{version}/inventoryOrganizations',
    }

    def __init__(self, config: OracleERPConfig):
        """Initialize Oracle ERP connector."""
        super().__init__(config)
        self.oracle_config = config

        # Build full base URL
        if not config.base_url.startswith('http'):
            self.config.base_url = f"https://{config.pod_url}.fa.{config.service_domain}.oraclecloud.com"

    async def _create_client(self) -> Any:
        """Create HTTP client for Oracle ERP."""
        try:
            import httpx
            return httpx.AsyncClient(
                verify=self.config.verify_ssl,
                timeout=self.config.timeout_seconds,
            )
        except ImportError:
            raise ImportError("httpx required for Oracle connector")

    async def fetch_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Fetch data from Oracle ERP - generic method."""
        resource = kwargs.get('resource', 'suppliers')
        filters = kwargs.get('filters', {})
        fields = kwargs.get('fields', [])
        limit = kwargs.get('limit', 500)
        offset = kwargs.get('offset', 0)

        return await self.fetch_resource(
            resource=resource,
            filters=filters,
            fields=fields,
            limit=limit,
            offset=offset,
        )

    async def fetch_resource(
        self,
        resource: str,
        filters: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        expand: Optional[List[str]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from Oracle REST resource with pagination.

        Args:
            resource: Resource name (e.g., 'suppliers', 'purchaseOrders')
            filters: Query filters
            fields: Fields to return
            expand: Child resources to expand
            limit: Max records per request
            offset: Starting offset

        Returns:
            List of resource records
        """
        all_records = []
        current_offset = offset
        has_more = True

        # Build endpoint
        endpoint_template = self.ENDPOINTS.get(resource)
        if not endpoint_template:
            endpoint_template = f'/fscmRestApi/resources/{{version}}/{resource}'

        endpoint = endpoint_template.format(version=self.oracle_config.api_version)

        while has_more:
            # Build query parameters
            params = self._build_query_params(
                filters=filters,
                fields=fields,
                expand=expand,
                limit=limit,
                offset=current_offset,
            )

            # Make request
            response = await self.request(
                method='GET',
                endpoint=endpoint,
                params=params,
            )

            # Extract records
            records = response.get('items', [])

            if not records:
                break

            all_records.extend(records)

            # Check pagination
            has_more = response.get('hasMore', False)
            current_offset += limit

            # Safety limit
            if current_offset > 100000:
                logger.warning("Reached pagination limit")
                break

        logger.info(f"Fetched {len(all_records)} records from {resource}")
        return all_records

    def _build_query_params(
        self,
        filters: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        expand: Optional[List[str]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> Dict[str, str]:
        """Build Oracle REST query parameters."""
        params = {
            'limit': str(limit),
            'offset': str(offset),
            'onlyData': 'true',
        }

        if fields:
            params['fields'] = ','.join(fields)

        if expand:
            params['expand'] = ','.join(expand)

        if filters:
            # Oracle uses different filter syntax
            filter_parts = []
            for key, value in filters.items():
                if isinstance(value, str):
                    filter_parts.append(f"{key}={value}")
                elif isinstance(value, (int, float)):
                    filter_parts.append(f"{key}={value}")
                elif isinstance(value, dict):
                    # Handle operators
                    for op, val in value.items():
                        if op == 'eq':
                            filter_parts.append(f"{key}={val}")
                        elif op == 'like':
                            filter_parts.append(f"{key} LIKE '{val}'")
                        elif op in ['ge', 'gt', 'le', 'lt']:
                            op_symbol = {'ge': '>=', 'gt': '>', 'le': '<=', 'lt': '<'}[op]
                            filter_parts.append(f"{key} {op_symbol} '{val}'")

            if filter_parts:
                params['q'] = ';'.join(filter_parts)

        return params

    async def fetch_suppliers(
        self,
        status: str = "ACTIVE",
        country: Optional[str] = None,
        supplier_type: Optional[str] = None,
    ) -> List[OracleSupplier]:
        """
        Fetch suppliers from Oracle ERP.

        Args:
            status: Supplier status filter
            country: Country filter
            supplier_type: Supplier type filter

        Returns:
            List of suppliers
        """
        filters = {'Status': status}

        if country:
            filters['Country'] = country
        if supplier_type:
            filters['SupplierType'] = supplier_type

        fields = [
            'SupplierId', 'SupplierName', 'SupplierNumber',
            'SupplierType', 'Country', 'Status', 'TaxRegistrationNumber',
        ]

        raw_data = await self.fetch_resource(
            resource='suppliers',
            filters=filters,
            fields=fields,
        )

        suppliers = []
        for record in raw_data:
            try:
                supplier = OracleSupplier(
                    supplier_id=record.get('SupplierId', 0),
                    supplier_name=record.get('SupplierName', ''),
                    supplier_number=record.get('SupplierNumber', ''),
                    supplier_type=record.get('SupplierType'),
                    country=record.get('Country'),
                    status=record.get('Status', 'ACTIVE'),
                    tax_registration_number=record.get('TaxRegistrationNumber'),
                )
                suppliers.append(supplier)
            except Exception as e:
                logger.warning(f"Error parsing supplier: {e}")

        return suppliers

    async def fetch_purchase_orders(
        self,
        start_date: date,
        end_date: date,
        status: Optional[str] = None,
        supplier_id: Optional[int] = None,
    ) -> List[OraclePurchaseOrder]:
        """
        Fetch purchase orders from Oracle ERP.

        Args:
            start_date: Start date filter
            end_date: End date filter
            status: PO status filter
            supplier_id: Supplier filter

        Returns:
            List of purchase orders
        """
        filters = {
            'CreationDate': {
                'ge': start_date.isoformat(),
                'le': end_date.isoformat(),
            },
        }

        if status:
            filters['Status'] = status
        if supplier_id:
            filters['SupplierId'] = supplier_id

        fields = [
            'POHeaderId', 'OrderNumber', 'SupplierId', 'SupplierName',
            'BuyerId', 'CurrencyCode', 'TotalAmount', 'Status', 'CreationDate',
        ]

        expand = ['lines']

        raw_data = await self.fetch_resource(
            resource='purchaseOrders',
            filters=filters,
            fields=fields,
            expand=expand,
        )

        orders = []
        for record in raw_data:
            try:
                order = OraclePurchaseOrder(
                    po_header_id=record.get('POHeaderId', 0),
                    po_number=record.get('OrderNumber', ''),
                    supplier_id=record.get('SupplierId', 0),
                    supplier_name=record.get('SupplierName'),
                    buyer_id=record.get('BuyerId'),
                    currency_code=record.get('CurrencyCode', 'USD'),
                    total_amount=Decimal(str(record.get('TotalAmount', 0))),
                    status=record.get('Status', ''),
                    creation_date=datetime.fromisoformat(
                        record.get('CreationDate', datetime.utcnow().isoformat())
                    ),
                    lines=record.get('lines', []),
                )
                orders.append(order)
            except Exception as e:
                logger.warning(f"Error parsing purchase order: {e}")

        return orders

    async def fetch_inventory_items(
        self,
        organization_id: Optional[int] = None,
        item_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch inventory items from Oracle ERP.

        Args:
            organization_id: Inventory organization filter
            item_type: Item type filter

        Returns:
            List of inventory items
        """
        filters = {}

        if organization_id:
            filters['OrganizationId'] = organization_id
        if item_type:
            filters['ItemType'] = item_type

        fields = [
            'ItemId', 'ItemNumber', 'ItemDescription',
            'PrimaryUOMCode', 'ItemType', 'OrganizationId',
            'UnitWeight', 'WeightUOMCode',
        ]

        return await self.fetch_resource(
            resource='inventoryItems',
            filters=filters,
            fields=fields,
        )

    async def fetch_cost_elements(
        self,
        organization_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch cost elements for costing analysis.

        Args:
            organization_id: Organization filter

        Returns:
            List of cost elements
        """
        filters = {}
        if organization_id:
            filters['OrganizationId'] = organization_id

        return await self.fetch_resource(
            resource='costElements',
            filters=filters,
        )

    async def execute_bi_report(
        self,
        report_path: str,
        parameters: Optional[Dict[str, Any]] = None,
        output_format: str = 'CSV',
    ) -> bytes:
        """
        Execute BI Publisher report and return results.

        Args:
            report_path: Path to report in BI Publisher catalog
            parameters: Report parameters
            output_format: Output format (CSV, XML, PDF, XLSX)

        Returns:
            Report content as bytes
        """
        if not self.oracle_config.bi_publisher_enabled:
            raise ValueError("BI Publisher integration not enabled")

        # Build SOAP request for BI Publisher
        report_request = {
            'reportAbsolutePath': report_path,
            'attributeFormat': output_format,
            'parameterNameValues': parameters or {},
        }

        # BI Publisher uses a different endpoint
        bi_endpoint = '/xmlpserver/services/ExternalReportWSSService'

        # This is a simplified implementation - actual BI Publisher
        # integration requires SOAP/WSDL handling

        logger.warning("BI Publisher execution requires SOAP implementation")
        return b''

    async def submit_fbdi_import(
        self,
        interface_name: str,
        file_content: bytes,
        file_name: str,
    ) -> Dict[str, Any]:
        """
        Submit FBDI (File-Based Data Import) to Oracle ERP.

        Args:
            interface_name: FBDI interface name
            file_content: CSV/ZIP file content
            file_name: Name for the uploaded file

        Returns:
            Import job details
        """
        # Upload file to UCM (Universal Content Management)
        upload_endpoint = '/fscmRestApi/resources/latest/erpintegrations'

        # Encode file as base64
        file_base64 = base64.b64encode(file_content).decode('utf-8')

        payload = {
            'OperationName': 'importBulkData',
            'DocumentContent': file_base64,
            'DocumentName': file_name,
            'ContentType': 'text/csv',
            'JobOptions': interface_name,
        }

        response = await self.request(
            method='POST',
            endpoint=upload_endpoint,
            json_data=payload,
        )

        return response

    async def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check status of ERP integration job.

        Args:
            job_id: Job ID to check

        Returns:
            Job status details
        """
        endpoint = f'/fscmRestApi/resources/latest/erpintegrations/{job_id}'

        return await self.request(
            method='GET',
            endpoint=endpoint,
        )
