# -*- coding: utf-8 -*-
"""
Workday REST API Connector for GreenLang

Production-ready connector for Workday HCM/Financial Management using REST API.
Handles OAuth2 authentication, RAAS reports, and spend analytics extraction.
"""

import os
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from urllib.parse import urljoin, quote
import hashlib
import hmac

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field, HttpUrl, SecretStr

from .base import BaseConnector

logger = logging.getLogger(__name__)


class WorkdayConfig(BaseModel):
    """Workday connection configuration."""
    base_url: HttpUrl = Field(..., description="Workday base URL")
    tenant: str = Field(..., description="Workday tenant name")
    username: str = Field(..., description="Workday username")
    password: SecretStr = Field(..., description="Workday password")
    client_id: str = Field(..., description="OAuth2 client ID")
    client_secret: SecretStr = Field(..., description="OAuth2 client secret")
    refresh_token: Optional[str] = Field(None, description="OAuth2 refresh token")
    api_version: str = Field(default="v35.0", description="API version")
    timeout_seconds: int = Field(default=60, description="Request timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    rate_limit_requests_per_minute: int = Field(default=60, description="Rate limit")
    page_size: int = Field(default=100, description="Page size for pagination")


class WorkdayConnector(BaseConnector):
    """
    Workday REST API and RAAS Connector.

    Features:
    - OAuth2 authentication with refresh token support
    - RAAS (Report-as-a-Service) integration
    - REST API for HCM and Financial Management
    - Spend analytics and vendor management
    - Rate limiting and retry logic
    - Pagination for large datasets
    - Support for custom reports
    """

    def __init__(self, credentials: Optional[Dict[str, str]] = None):
        """Initialize Workday connector with credentials from environment or config."""
        # Get credentials from environment variables or passed config
        if credentials is None:
            credentials = {
                'base_url': os.getenv('WORKDAY_BASE_URL', 'https://wd2-impl-services1.workday.com'),
                'tenant': os.getenv('WORKDAY_TENANT', ''),
                'username': os.getenv('WORKDAY_USERNAME', ''),
                'password': os.getenv('WORKDAY_PASSWORD', ''),
                'client_id': os.getenv('WORKDAY_CLIENT_ID', ''),
                'client_secret': os.getenv('WORKDAY_CLIENT_SECRET', ''),
                'refresh_token': os.getenv('WORKDAY_REFRESH_TOKEN', '')
            }

        super().__init__(credentials)

        # Initialize configuration
        self.config = WorkdayConfig(
            base_url=credentials.get('base_url'),
            tenant=credentials.get('tenant'),
            username=credentials.get('username'),
            password=credentials.get('password'),
            client_id=credentials.get('client_id'),
            client_secret=credentials.get('client_secret'),
            refresh_token=credentials.get('refresh_token')
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

        logger.info("Workday connector initialized")

    def connect(self) -> bool:
        """Establish connection to Workday."""
        try:
            self._authenticate()

            # Test connection with workers endpoint
            test_url = self._build_url(f"/api/{self.config.api_version}/workers?limit=1")
            response = self._make_request('GET', test_url)

            if response.status_code == 200:
                self.connected = True
                logger.info("Successfully connected to Workday")
                return True
            else:
                logger.error(f"Workday connection test failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Workday: {str(e)}")
            self.connected = False
            return False

    def _build_url(self, path: str) -> str:
        """Build full URL with tenant."""
        base = str(self.config.base_url)
        if self.config.tenant:
            return f"{base}/ccx/service/{self.config.tenant}{path}"
        return urljoin(base, path)

    def _authenticate(self) -> str:
        """Obtain or refresh OAuth2 access token."""
        # Check if token is still valid
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at - timedelta(minutes=5):
                return self.access_token

        # Use refresh token if available
        if self.config.refresh_token:
            return self._refresh_access_token()

        # Otherwise, get new token with client credentials
        token_url = self._build_url("/oauth2/token")
        token_data = {
            'grant_type': 'client_credentials',
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret.get_secret_value()
        }

        try:
            response = self.client.post(
                token_url,
                data=token_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            response.raise_for_status()

            token_response = response.json()
            self.access_token = token_response['access_token']
            expires_in = token_response.get('expires_in', 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

            # Store refresh token if provided
            if 'refresh_token' in token_response:
                self.config.refresh_token = token_response['refresh_token']

            logger.info("Workday OAuth2 authentication successful")
            return self.access_token

        except httpx.HTTPStatusError as e:
            logger.error(f"Workday authentication failed: {e.response.status_code}")
            raise Exception(f"Workday OAuth2 failed: {e.response.text}")

    def _refresh_access_token(self) -> str:
        """Refresh access token using refresh token."""
        token_url = self._build_url("/oauth2/token")
        token_data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.config.refresh_token,
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret.get_secret_value()
        }

        try:
            response = self.client.post(
                token_url,
                data=token_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            response.raise_for_status()

            token_response = response.json()
            self.access_token = token_response['access_token']
            expires_in = token_response.get('expires_in', 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

            # Update refresh token if new one provided
            if 'refresh_token' in token_response:
                self.config.refresh_token = token_response['refresh_token']

            logger.info("Workday token refresh successful")
            return self.access_token

        except httpx.HTTPStatusError as e:
            logger.error(f"Token refresh failed: {e.response.status_code}")
            # Fall back to full authentication
            self.config.refresh_token = None
            return self._authenticate()

    def _enforce_rate_limit(self):
        """Enforce rate limiting to avoid overwhelming Workday API."""
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

        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logger.warning(f"Workday rate limit hit, waiting {retry_after}s")
            time.sleep(retry_after)
            response = self.client.request(method, url, **kwargs)

        response.raise_for_status()
        return response

    def query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute query against Workday REST API.

        Args:
            query: Query parameters including:
                - entity_type: 'spend_categories', 'suppliers', 'purchase_orders', 'invoices', etc.
                - filters: API query parameters
                - start_date: Start date for date range
                - end_date: End date for date range
                - fields: Fields to select
                - limit: Maximum records to return

        Returns:
            List of records matching query
        """
        if not self.connected:
            self.connect()

        entity_type = query.get('entity_type', 'suppliers')

        # Map entity types to Workday REST API endpoints
        entity_map = {
            # Financial Management
            'suppliers': f'/api/{self.config.api_version}/suppliers',
            'spend_categories': f'/api/{self.config.api_version}/spend_categories',
            'purchase_orders': f'/api/{self.config.api_version}/purchaseOrders',
            'requisitions': f'/api/{self.config.api_version}/requisitions',
            'invoices': f'/api/{self.config.api_version}/supplier_invoices',
            'expense_reports': f'/api/{self.config.api_version}/expense_reports',
            'supplier_contracts': f'/api/{self.config.api_version}/supplier_contracts',
            'procurement_cards': f'/api/{self.config.api_version}/procurement_card_transactions',

            # HCM (for vendor/contractor spend)
            'workers': f'/api/{self.config.api_version}/workers',
            'contingent_workers': f'/api/{self.config.api_version}/contingentWorkers',

            # Analytics
            'spend_analytics': '/api/financialManagement/v1/spendAnalytics'
        }

        endpoint = entity_map.get(entity_type)
        if not endpoint:
            # Try RAAS report if not a standard endpoint
            return self._get_raas_report(entity_type, query)

        # Build query parameters
        params = self._build_query_params(query)

        # Fetch data with pagination
        all_records = []
        offset = 0

        while True:
            params['offset'] = offset
            params['limit'] = self.config.page_size

            url = self._build_url(endpoint)

            try:
                response = self._make_request('GET', url, params=params)
                data = response.json()

                # Extract records from Workday response
                if 'data' in data:
                    records = data['data']
                elif isinstance(data, list):
                    records = data
                else:
                    records = [data]

                if not records:
                    break

                # Transform Workday data to standard format
                transformed = [self._transform_record(r, entity_type) for r in records]
                all_records.extend(transformed)

                # Check limit
                limit = query.get('limit')
                if limit and len(all_records) >= limit:
                    all_records = all_records[:limit]
                    break

                # Check for more data
                total = data.get('total', 0)
                if offset + len(records) >= total or len(records) < self.config.page_size:
                    break

                offset += self.config.page_size

            except Exception as e:
                logger.error(f"Error querying Workday {entity_type}: {str(e)}")
                break

        logger.info(f"Retrieved {len(all_records)} {entity_type} records from Workday")
        return all_records

    def _get_raas_report(self, report_name: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get data from Workday RAAS (Report-as-a-Service).

        Args:
            report_name: Name of the RAAS report
            query: Query parameters for filtering

        Returns:
            List of records from RAAS report
        """
        raas_url = self._build_url(f"/raas/customreport2/{self.config.tenant}/{report_name}")

        # Build RAAS parameters
        params = {
            'format': 'json'
        }

        # Add date filters
        if query.get('start_date'):
            params['Start_Date'] = query['start_date']
        if query.get('end_date'):
            params['End_Date'] = query['end_date']

        # Add custom parameters
        if query.get('filters'):
            params.update(query['filters'])

        try:
            response = self._make_request('GET', raas_url, params=params)
            data = response.json()

            # RAAS reports typically return data in 'Report_Entry' field
            records = data.get('Report_Entry', [])

            # Transform to standard format
            transformed = [self._transform_raas_record(r, report_name) for r in records]

            logger.info(f"Retrieved {len(transformed)} records from RAAS report {report_name}")
            return transformed

        except Exception as e:
            logger.error(f"Error getting RAAS report {report_name}: {str(e)}")
            return []

    def _build_query_params(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Build Workday API query parameters."""
        params = {}

        # Add date filters (Workday format: YYYY-MM-DD)
        if query.get('start_date'):
            params['fromDate'] = query['start_date']
        if query.get('end_date'):
            params['toDate'] = query['end_date']

        # Add custom filters
        if query.get('filters'):
            if isinstance(query['filters'], dict):
                params.update(query['filters'])

        # Add field selection (if supported by endpoint)
        if query.get('fields'):
            params['fields'] = ','.join(query['fields'])

        return params

    def _transform_record(self, record: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Transform Workday record to GreenLang standard format."""
        if entity_type == 'suppliers':
            return self._transform_supplier(record)
        elif entity_type == 'purchase_orders':
            return self._transform_purchase_order(record)
        elif entity_type == 'spend_categories':
            return self._transform_spend_category(record)
        elif entity_type == 'invoices':
            return self._transform_invoice(record)
        elif entity_type == 'expense_reports':
            return self._transform_expense_report(record)
        else:
            return record

    def _transform_supplier(self, supplier: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Workday supplier to standard format."""
        return {
            'id': supplier.get('id'),
            'type': 'supplier',
            'source_system': 'Workday',
            'supplier_id': supplier.get('supplierID'),
            'supplier_name': supplier.get('supplierName'),
            'supplier_category': supplier.get('supplierCategory'),
            'tax_id': supplier.get('taxID'),
            'duns_number': supplier.get('dunsNumber'),
            'status': supplier.get('supplierStatus'),
            'vendor_type': supplier.get('vendorType'),
            'payment_terms': supplier.get('paymentTerms', {}).get('descriptor'),
            'currency': supplier.get('defaultCurrency'),
            'addresses': supplier.get('addresses', []),
            'contacts': supplier.get('contacts', []),
            'bank_accounts': supplier.get('settlementAccounts', []),
            'spend_categories': supplier.get('spendCategories', []),
            'metadata': {
                'supplier_group': supplier.get('supplierGroup', {}).get('descriptor'),
                'worktags': supplier.get('worktags', []),
                'custom_fields': supplier.get('customFields', {})
            },
            'created_date': supplier.get('createdOn'),
            'last_updated': supplier.get('lastUpdated'),
            'extracted_at': datetime.utcnow().isoformat()
        }

    def _transform_purchase_order(self, po: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Workday purchase order to standard format."""
        return {
            'id': po.get('id'),
            'type': 'purchase_order',
            'source_system': 'Workday',
            'po_number': po.get('purchaseOrderNumber'),
            'supplier_id': po.get('supplier', {}).get('id'),
            'supplier_name': po.get('supplier', {}).get('descriptor'),
            'requester': po.get('requester', {}).get('descriptor'),
            'buyer': po.get('buyer', {}).get('descriptor'),
            'created_date': po.get('purchaseOrderDate'),
            'due_date': po.get('dueDate'),
            'total_amount': float(po.get('totalAmount', {}).get('value', 0)),
            'currency': po.get('totalAmount', {}).get('currency'),
            'status': po.get('purchaseOrderStatus'),
            'company': po.get('company', {}).get('descriptor'),
            'cost_center': po.get('costCenter', {}).get('descriptor'),
            'ship_to_address': po.get('shipToAddress', {}).get('descriptor'),
            'bill_to_address': po.get('billToAddress', {}).get('descriptor'),
            'lines': po.get('purchaseOrderLines', []),
            'metadata': {
                'spend_category': po.get('spendCategory', {}).get('descriptor'),
                'worktags': po.get('worktags', []),
                'custom_fields': po.get('customFields', {})
            },
            'extracted_at': datetime.utcnow().isoformat()
        }

    def _transform_spend_category(self, category: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Workday spend category to standard format."""
        return {
            'id': category.get('id'),
            'type': 'spend_category',
            'source_system': 'Workday',
            'category_code': category.get('spendCategoryCode'),
            'category_name': category.get('spendCategoryName'),
            'parent_category': category.get('parentSpendCategory', {}).get('descriptor'),
            'category_hierarchy': category.get('spendCategoryHierarchy'),
            'description': category.get('description'),
            'active': category.get('inactive', False) is False,
            'metadata': {
                'resource_category': category.get('resourceCategory'),
                'custom_fields': category.get('customFields', {})
            },
            'extracted_at': datetime.utcnow().isoformat()
        }

    def _transform_invoice(self, invoice: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Workday supplier invoice to standard format."""
        return {
            'id': invoice.get('id'),
            'type': 'invoice',
            'source_system': 'Workday',
            'invoice_number': invoice.get('supplierInvoiceNumber'),
            'invoice_date': invoice.get('invoiceDate'),
            'supplier_id': invoice.get('supplier', {}).get('id'),
            'supplier_name': invoice.get('supplier', {}).get('descriptor'),
            'po_number': invoice.get('purchaseOrder', {}).get('descriptor'),
            'total_amount': float(invoice.get('totalAmount', {}).get('value', 0)),
            'paid_amount': float(invoice.get('paidAmount', {}).get('value', 0)),
            'currency': invoice.get('totalAmount', {}).get('currency'),
            'payment_status': invoice.get('paymentStatus'),
            'approval_status': invoice.get('approvalStatus'),
            'due_date': invoice.get('dueDate'),
            'payment_terms': invoice.get('paymentTerms', {}).get('descriptor'),
            'company': invoice.get('company', {}).get('descriptor'),
            'lines': invoice.get('invoiceLines', []),
            'metadata': {
                'invoice_type': invoice.get('invoiceType'),
                'spend_category': invoice.get('spendCategory', {}).get('descriptor'),
                'worktags': invoice.get('worktags', []),
                'custom_fields': invoice.get('customFields', {})
            },
            'extracted_at': datetime.utcnow().isoformat()
        }

    def _transform_expense_report(self, expense: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Workday expense report to standard format."""
        return {
            'id': expense.get('id'),
            'type': 'expense_report',
            'source_system': 'Workday',
            'report_number': expense.get('expenseReportNumber'),
            'employee': expense.get('worker', {}).get('descriptor'),
            'submit_date': expense.get('submitDate'),
            'approval_date': expense.get('approvalDate'),
            'total_amount': float(expense.get('totalAmount', {}).get('value', 0)),
            'reimbursable_amount': float(expense.get('reimbursableAmount', {}).get('value', 0)),
            'currency': expense.get('totalAmount', {}).get('currency'),
            'status': expense.get('expenseReportStatus'),
            'payment_status': expense.get('paymentStatus'),
            'expense_lines': expense.get('expenseLines', []),
            'metadata': {
                'expense_type': expense.get('expenseType'),
                'business_purpose': expense.get('businessPurpose'),
                'worktags': expense.get('worktags', []),
                'custom_fields': expense.get('customFields', {})
            },
            'extracted_at': datetime.utcnow().isoformat()
        }

    def _transform_raas_record(self, record: Dict[str, Any], report_name: str) -> Dict[str, Any]:
        """Transform RAAS report record to standard format."""
        # Generic transformation for RAAS reports
        return {
            'id': record.get('WID') or record.get('ID'),
            'type': 'raas_report',
            'source_system': 'Workday',
            'report_name': report_name,
            'data': record,
            'extracted_at': datetime.utcnow().isoformat()
        }

    def get_spend_analytics(self, start_date: str, end_date: str,
                           spend_category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get spend analytics data for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            spend_category: Optional spend category filter

        Returns:
            List of spend analytics records
        """
        query = {
            'entity_type': 'spend_analytics',
            'start_date': start_date,
            'end_date': end_date
        }

        if spend_category:
            query['filters'] = {'spendCategory': spend_category}

        return self.query(query)

    def get_suppliers(self, status: str = 'ACTIVE') -> List[Dict[str, Any]]:
        """
        Get supplier information.

        Args:
            status: Supplier status filter

        Returns:
            List of supplier records
        """
        query = {
            'entity_type': 'suppliers',
            'filters': {'supplierStatus': status}
        }

        return self.query(query)

    def get_purchase_orders(self, start_date: str, end_date: str,
                          company: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get purchase orders for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            company: Optional company filter

        Returns:
            List of purchase order records
        """
        query = {
            'entity_type': 'purchase_orders',
            'start_date': start_date,
            'end_date': end_date
        }

        if company:
            query['filters'] = {'company': company}

        return self.query(query)

    def disconnect(self) -> None:
        """Close connection to Workday."""
        if self.client:
            self.client.close()
        self.connected = False
        logger.info("Disconnected from Workday")


__all__ = ["WorkdayConnector"]