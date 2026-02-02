# -*- coding: utf-8 -*-
"""
SAP S/4HANA OData API Connector for GreenLang

Production-ready connector for SAP S/4HANA using OData REST API.
Handles OAuth2 authentication, rate limiting, retry logic, and pagination.
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urljoin, quote

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field, HttpUrl, SecretStr

from .base import BaseConnector

logger = logging.getLogger(__name__)


class SAPConfig(BaseModel):
    """SAP connection configuration."""
    base_url: HttpUrl = Field(..., description="SAP S/4HANA base URL")
    client_id: str = Field(..., description="OAuth2 client ID")
    client_secret: SecretStr = Field(..., description="OAuth2 client secret")
    oauth_token_url: HttpUrl = Field(..., description="OAuth2 token endpoint")
    company_code: str = Field(default="1000", description="SAP company code")
    api_version: str = Field(default="v1", description="API version")
    timeout_seconds: int = Field(default=30, description="Request timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    rate_limit_requests_per_minute: int = Field(default=100, description="Rate limit")
    page_size: int = Field(default=1000, description="Page size for pagination")


class SAPConnector(BaseConnector):
    """
    SAP S/4HANA OData API Connector.

    Features:
    - OAuth2 authentication with automatic token refresh
    - Rate limiting and throttling
    - Retry logic with exponential backoff
    - Pagination for large datasets
    - Comprehensive error handling
    - Connection pooling for performance
    """

    def __init__(self, credentials: Optional[Dict[str, str]] = None):
        """Initialize SAP connector with credentials from environment or config."""
        # Get credentials from environment variables or passed config
        if credentials is None:
            credentials = {
                'base_url': os.getenv('SAP_BASE_URL', 'https://api.s4hana.example.com'),
                'client_id': os.getenv('SAP_CLIENT_ID', ''),
                'client_secret': os.getenv('SAP_CLIENT_SECRET', ''),
                'oauth_token_url': os.getenv('SAP_OAUTH_URL', 'https://auth.s4hana.example.com/oauth/token'),
                'company_code': os.getenv('SAP_COMPANY_CODE', '1000')
            }

        super().__init__(credentials)

        # Initialize configuration
        self.config = SAPConfig(
            base_url=credentials.get('base_url'),
            client_id=credentials.get('client_id'),
            client_secret=credentials.get('client_secret'),
            oauth_token_url=credentials.get('oauth_token_url'),
            company_code=credentials.get('company_code', '1000')
        )

        # HTTP client with connection pooling
        self.client = httpx.Client(
            timeout=self.config.timeout_seconds,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            follow_redirects=True
        )

        # OAuth token management
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None

        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window_start = time.time()

        # Deduplication cache (if available)
        self._init_deduplication_cache()

        logger.info("SAP connector initialized")

    def _init_deduplication_cache(self):
        """Initialize deduplication cache if available."""
        try:
            from connectors.sap.utils.deduplication import DeduplicationCache
            self.dedup_cache = DeduplicationCache(ttl_days=7)
            logger.info("Deduplication cache initialized for SAP connector")
        except ImportError:
            self.dedup_cache = None
            logger.warning("Deduplication cache not available")

    def connect(self) -> bool:
        """Establish connection to SAP and authenticate."""
        try:
            self._authenticate()
            # Test connection with a simple query
            test_url = urljoin(str(self.config.base_url), "/sap/opu/odata/sap/API_BUSINESS_PARTNER/A_BusinessPartner?$top=1")
            response = self._make_request('GET', test_url)

            if response.status_code == 200:
                self.connected = True
                logger.info("Successfully connected to SAP S/4HANA")
                return True
            else:
                logger.error(f"SAP connection test failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to SAP: {str(e)}")
            self.connected = False
            return False

    def _authenticate(self) -> str:
        """Obtain or refresh OAuth2 access token."""
        # Check if token is still valid
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at - timedelta(minutes=5):
                return self.access_token

        # Request new token
        token_data = {
            'grant_type': 'client_credentials',
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret.get_secret_value()
        }

        try:
            response = self.client.post(
                str(self.config.oauth_token_url),
                data=token_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            response.raise_for_status()

            token_response = response.json()
            self.access_token = token_response['access_token']
            expires_in = token_response.get('expires_in', 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

            logger.info("SAP OAuth2 authentication successful")
            return self.access_token

        except httpx.HTTPStatusError as e:
            logger.error(f"SAP authentication failed: {e.response.status_code}")
            raise Exception(f"SAP OAuth2 failed: {e.response.text}")

    def _enforce_rate_limit(self):
        """Enforce rate limiting to avoid overwhelming SAP API."""
        current_time = time.time()

        # Reset window if needed
        if current_time - self.rate_limit_window_start >= 60:
            self.request_count = 0
            self.rate_limit_window_start = current_time

        # Check if rate limit exceeded
        if self.request_count >= self.config.rate_limit_requests_per_minute:
            sleep_time = 60 - (current_time - self.rate_limit_window_start)
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_count = 0
                self.rate_limit_window_start = time.time()

        self.request_count += 1

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPStatusError)
    )
    def _make_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make HTTP request with retry logic and rate limiting."""
        self._enforce_rate_limit()

        # Ensure authentication
        token = self._authenticate()

        # Add auth header
        headers = kwargs.get('headers', {})
        headers['Authorization'] = f'Bearer {token}'
        headers['Accept'] = 'application/json'
        kwargs['headers'] = headers

        # Make request
        response = self.client.request(method, url, **kwargs)
        response.raise_for_status()

        return response

    def query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute query against SAP OData API.

        Args:
            query: Query parameters including:
                - entity_type: 'purchase_orders', 'shipments', 'suppliers', 'materials'
                - filters: OData filter parameters
                - start_date: Start date for date range
                - end_date: End date for date range
                - fields: Fields to select
                - limit: Maximum records to return

        Returns:
            List of records matching query
        """
        if not self.connected:
            self.connect()

        entity_type = query.get('entity_type', 'purchase_orders')

        # Map entity types to SAP OData endpoints
        entity_map = {
            'purchase_orders': '/sap/opu/odata/sap/API_PURCHASEORDER_PROCESS_SRV/A_PurchaseOrder',
            'purchase_order_items': '/sap/opu/odata/sap/API_PURCHASEORDER_PROCESS_SRV/A_PurchaseOrderItem',
            'suppliers': '/sap/opu/odata/sap/API_BUSINESS_PARTNER/A_Supplier',
            'materials': '/sap/opu/odata/sap/API_MATERIAL_DOCUMENT_SRV/A_MaterialDocument',
            'deliveries': '/sap/opu/odata/sap/API_OUTBOUND_DELIVERY_SRV/A_OutbDeliveryHeader',
            'shipments': '/sap/opu/odata/sap/LE_SHIPMENT_SRV/ShipmentSet',
            'invoices': '/sap/opu/odata/sap/API_SUPPLIERINVOICE_PROCESS_SRV/A_SupplierInvoice'
        }

        endpoint = entity_map.get(entity_type)
        if not endpoint:
            logger.error(f"Unknown entity type: {entity_type}")
            return []

        # Build OData query parameters
        params = self._build_odata_params(query)

        # Fetch data with pagination
        all_records = []
        skip = 0

        while True:
            params['$skip'] = skip
            params['$top'] = self.config.page_size

            url = urljoin(str(self.config.base_url), endpoint)

            try:
                response = self._make_request('GET', url, params=params)
                data = response.json()

                # Extract records from OData response
                records = data.get('d', {}).get('results', [])
                if not records:
                    break

                # Apply deduplication if available
                if self.dedup_cache:
                    # Filter out already processed records
                    id_field = self._get_id_field(entity_type)
                    transaction_ids = [r.get(id_field) for r in records if r.get(id_field)]
                    new_ids = self.dedup_cache.filter_duplicates(transaction_ids, entity_type)
                    records = [r for r in records if r.get(id_field) in new_ids]

                    # Mark as processed
                    self.dedup_cache.mark_batch_processed(new_ids, entity_type)

                # Transform SAP data to standard format
                transformed = [self._transform_record(r, entity_type) for r in records]
                all_records.extend(transformed)

                # Check limit
                limit = query.get('limit')
                if limit and len(all_records) >= limit:
                    all_records = all_records[:limit]
                    break

                # Check for more data
                if len(records) < self.config.page_size:
                    break

                skip += self.config.page_size

            except Exception as e:
                logger.error(f"Error querying SAP {entity_type}: {str(e)}")
                break

        logger.info(f"Retrieved {len(all_records)} {entity_type} records from SAP")
        return all_records

    def _build_odata_params(self, query: Dict[str, Any]) -> Dict[str, str]:
        """Build OData query parameters from query dict."""
        params = {
            '$format': 'json',
            '$inlinecount': 'allpages'
        }

        # Add date filters
        filters = []
        if query.get('start_date'):
            filters.append(f"CreationDate ge datetime'{query['start_date']}T00:00:00'")
        if query.get('end_date'):
            filters.append(f"CreationDate le datetime'{query['end_date']}T23:59:59'")

        # Add custom filters
        if query.get('filters'):
            if isinstance(query['filters'], dict):
                for field, value in query['filters'].items():
                    if isinstance(value, list):
                        # Multiple values - use OR
                        or_filters = [f"{field} eq '{v}'" for v in value]
                        filters.append(f"({' or '.join(or_filters)})")
                    else:
                        filters.append(f"{field} eq '{value}'")
            elif isinstance(query['filters'], str):
                filters.append(query['filters'])

        if filters:
            params['$filter'] = ' and '.join(filters)

        # Add field selection
        if query.get('fields'):
            params['$select'] = ','.join(query['fields'])

        # Add expansion for related entities
        if query.get('expand'):
            params['$expand'] = ','.join(query['expand'])

        return params

    def _get_id_field(self, entity_type: str) -> str:
        """Get the ID field name for an entity type."""
        id_fields = {
            'purchase_orders': 'PurchaseOrder',
            'purchase_order_items': 'PurchaseOrderItem',
            'suppliers': 'Supplier',
            'materials': 'Material',
            'deliveries': 'DeliveryDocument',
            'shipments': 'TransportationOrder',
            'invoices': 'SupplierInvoice'
        }
        return id_fields.get(entity_type, 'ID')

    def _transform_record(self, record: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Transform SAP record to GreenLang standard format."""
        if entity_type == 'purchase_orders':
            return self._transform_purchase_order(record)
        elif entity_type == 'suppliers':
            return self._transform_supplier(record)
        elif entity_type == 'materials':
            return self._transform_material(record)
        elif entity_type == 'deliveries':
            return self._transform_delivery(record)
        else:
            return record

    def _transform_purchase_order(self, po: Dict[str, Any]) -> Dict[str, Any]:
        """Transform SAP purchase order to standard format."""
        return {
            'id': po.get('PurchaseOrder'),
            'type': 'purchase_order',
            'source_system': 'SAP',
            'document_number': po.get('PurchaseOrder'),
            'supplier_id': po.get('Supplier'),
            'supplier_name': po.get('SupplierName'),
            'created_date': po.get('CreationDate'),
            'delivery_date': po.get('PurchaseOrderDate'),
            'total_amount': float(po.get('TotalNetAmount', 0)),
            'currency': po.get('DocumentCurrency'),
            'status': po.get('PurchasingProcessingStatus'),
            'company_code': po.get('CompanyCode'),
            'plant': po.get('Plant'),
            'items': po.get('to_PurchaseOrderItem', {}).get('results', []),
            'metadata': {
                'purchasing_org': po.get('PurchasingOrganization'),
                'purchasing_group': po.get('PurchasingGroup'),
                'payment_terms': po.get('PaymentTerms'),
                'incoterms': po.get('IncotermsClassification')
            },
            'extracted_at': datetime.utcnow().isoformat()
        }

    def _transform_supplier(self, supplier: Dict[str, Any]) -> Dict[str, Any]:
        """Transform SAP supplier to standard format."""
        return {
            'id': supplier.get('Supplier'),
            'type': 'supplier',
            'source_system': 'SAP',
            'supplier_id': supplier.get('Supplier'),
            'name': supplier.get('SupplierName'),
            'tax_number': supplier.get('TaxNumber1'),
            'address': {
                'street': supplier.get('StreetName'),
                'city': supplier.get('CityName'),
                'postal_code': supplier.get('PostalCode'),
                'country': supplier.get('Country'),
                'region': supplier.get('Region')
            },
            'contact': {
                'phone': supplier.get('PhoneNumber1'),
                'email': supplier.get('EmailAddress'),
                'fax': supplier.get('FaxNumber')
            },
            'payment_terms': supplier.get('PaymentTerms'),
            'industry': supplier.get('Industry'),
            'created_date': supplier.get('CreationDate'),
            'extracted_at': datetime.utcnow().isoformat()
        }

    def _transform_material(self, material: Dict[str, Any]) -> Dict[str, Any]:
        """Transform SAP material to standard format."""
        return {
            'id': material.get('Material'),
            'type': 'material',
            'source_system': 'SAP',
            'material_number': material.get('Material'),
            'description': material.get('MaterialDocumentItemText'),
            'quantity': float(material.get('Quantity', 0)),
            'unit': material.get('MaterialBaseUnit'),
            'plant': material.get('Plant'),
            'storage_location': material.get('StorageLocation'),
            'batch': material.get('Batch'),
            'movement_type': material.get('GoodsMovementType'),
            'posting_date': material.get('PostingDate'),
            'document_date': material.get('DocumentDate'),
            'extracted_at': datetime.utcnow().isoformat()
        }

    def _transform_delivery(self, delivery: Dict[str, Any]) -> Dict[str, Any]:
        """Transform SAP delivery to standard format."""
        return {
            'id': delivery.get('DeliveryDocument'),
            'type': 'delivery',
            'source_system': 'SAP',
            'delivery_number': delivery.get('DeliveryDocument'),
            'shipping_point': delivery.get('ShippingPoint'),
            'delivery_date': delivery.get('DeliveryDate'),
            'planned_goods_issue': delivery.get('PlannedGoodsIssueDate'),
            'actual_goods_issue': delivery.get('ActualGoodsMovementDate'),
            'ship_to_party': delivery.get('ShipToParty'),
            'sold_to_party': delivery.get('SoldToParty'),
            'total_weight': float(delivery.get('HeaderGrossWeight', 0)),
            'weight_unit': delivery.get('HeaderWeightUnit'),
            'total_volume': float(delivery.get('HeaderVolume', 0)),
            'volume_unit': delivery.get('HeaderVolumeUnit'),
            'delivery_status': delivery.get('OverallDeliveryStatus'),
            'transportation_mode': delivery.get('TransportationMode'),
            'extracted_at': datetime.utcnow().isoformat()
        }

    def get_purchase_orders(self, start_date: str, end_date: str,
                           plant_codes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get purchase orders for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            plant_codes: Optional list of plant codes to filter

        Returns:
            List of purchase order records
        """
        query = {
            'entity_type': 'purchase_orders',
            'start_date': start_date,
            'end_date': end_date,
            'expand': ['to_PurchaseOrderItem']
        }

        if plant_codes:
            query['filters'] = {'Plant': plant_codes}

        return self.query(query)

    def get_suppliers(self, supplier_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get supplier information.

        Args:
            supplier_ids: Optional list of supplier IDs to fetch

        Returns:
            List of supplier records
        """
        query = {'entity_type': 'suppliers'}

        if supplier_ids:
            query['filters'] = {'Supplier': supplier_ids}

        return self.query(query)

    def get_shipments(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Get shipment/logistics data for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of shipment records
        """
        # Get deliveries as proxy for shipments
        query = {
            'entity_type': 'deliveries',
            'start_date': start_date,
            'end_date': end_date
        }

        return self.query(query)

    def disconnect(self) -> None:
        """Close connection to SAP."""
        if self.client:
            self.client.close()
        self.connected = False
        logger.info("Disconnected from SAP")


__all__ = ["SAPConnector"]