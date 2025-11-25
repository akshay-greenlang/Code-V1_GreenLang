# -*- coding: utf-8 -*-
"""
GreenLang ERP Connectors Architecture
Version: 1.0.0
Supports: SAP, Oracle, Workday, Microsoft Dynamics 365, NetSuite
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import httpx
from pydantic import BaseModel, Field, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from greenlang.determinism import FinancialDecimal
from greenlang.determinism import deterministic_random

logger = logging.getLogger(__name__)

# ==============================================
# SAP CONNECTOR (OData, RFC, BAPI)
# ==============================================

class SAPConnector:
    """
    SAP S/4HANA Integration Connector
    Supports OData API, RFC, and BAPI protocols
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize SAP connector with configuration."""
        self.base_url = config['base_url']
        self.client_id = config['client_id']
        self.client_secret = config['client_secret']
        self.oauth_url = config['oauth_url']
        self.sap_client = config.get('sap_client', '100')

        self.session = httpx.AsyncClient(
            timeout=30,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
        )
        self.access_token = None
        self.token_expires_at = None

    async def authenticate(self) -> str:
        """Authenticate with SAP OAuth2 server."""
        if self.access_token and self.token_expires_at > DeterministicClock.now():
            return self.access_token

        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': 'API_BUSINESS_PARTNER_0001 API_PURCHASEORDER_PROCESS_SRV'
        }

        response = await self.session.post(self.oauth_url, data=auth_data)
        response.raise_for_status()

        token_data = response.json()
        self.access_token = token_data['access_token']
        expires_in = token_data.get('expires_in', 3600)
        self.token_expires_at = DeterministicClock.now() + timedelta(seconds=expires_in - 60)

        return self.access_token

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_purchase_orders(
        self,
        start_date: str,
        end_date: str,
        company_code: Optional[str] = None,
        batch_size: int = 1000
    ) -> List[Dict]:
        """
        Fetch purchase orders from SAP.

        SAP OData Entity: /sap/opu/odata/sap/API_PURCHASEORDER_PROCESS_SRV/A_PurchaseOrder
        """
        await self.authenticate()

        # Build OData query
        filter_parts = [
            f"CreationDate ge datetime'{start_date}T00:00:00'",
            f"CreationDate le datetime'{end_date}T23:59:59'"
        ]
        if company_code:
            filter_parts.append(f"CompanyCode eq '{company_code}'")

        filter_query = " and ".join(filter_parts)

        url = f"{self.base_url}/sap/opu/odata/sap/API_PURCHASEORDER_PROCESS_SRV/A_PurchaseOrder"

        all_records = []
        skip = 0

        while True:
            params = {
                '$filter': filter_query,
                '$select': 'PurchaseOrder,Supplier,PurchasingDocument,CompanyCode,PurchasingOrganization,TotalNetAmount,DocumentCurrency,CreationDate',
                '$expand': 'to_PurchaseOrderItem',
                '$top': batch_size,
                '$skip': skip,
                '$format': 'json',
                'sap-client': self.sap_client
            }

            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json',
                'x-csrf-token': 'Fetch'
            }

            response = await self.session.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            results = data.get('d', {}).get('results', [])

            if not results:
                break

            all_records.extend(results)
            skip += batch_size

            # Check if there are more results
            if len(results) < batch_size:
                break

        logger.info(f"Fetched {len(all_records)} purchase orders from SAP")
        return all_records

    async def fetch_material_master(
        self,
        material_codes: List[str],
        plant: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch material master data from SAP.

        SAP OData Entity: /sap/opu/odata/sap/API_MATERIAL_STOCK_SRV/A_Product
        """
        await self.authenticate()

        # Build filter for material codes
        material_filter = " or ".join([f"Product eq '{code}'" for code in material_codes])

        url = f"{self.base_url}/sap/opu/odata/sap/API_PRODUCT_SRV/A_Product"

        params = {
            '$filter': f"({material_filter})",
            '$select': 'Product,ProductType,ProductGroup,BaseUnit,GrossWeight,NetWeight,WeightUnit',
            '$expand': 'to_ProductPlant,to_ProductSalesDelivery',
            '$format': 'json',
            'sap-client': self.sap_client
        }

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }

        response = await self.session.get(url, params=params, headers=headers)
        response.raise_for_status()

        data = response.json()
        return data.get('d', {}).get('results', [])

    async def fetch_business_partners(
        self,
        partner_type: str = 'Supplier',
        country: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch business partner (supplier/customer) data.

        SAP OData Entity: /sap/opu/odata/sap/API_BUSINESS_PARTNER/A_BusinessPartner
        """
        await self.authenticate()

        filter_parts = [f"BusinessPartnerCategory eq '{partner_type}'"]
        if country:
            filter_parts.append(f"Country eq '{country}'")

        url = f"{self.base_url}/sap/opu/odata/sap/API_BUSINESS_PARTNER/A_BusinessPartner"

        params = {
            '$filter': " and ".join(filter_parts),
            '$select': 'BusinessPartner,BusinessPartnerName,BusinessPartnerCategory,Country,Region,PostalCode',
            '$expand': 'to_BusinessPartnerAddress',
            '$format': 'json',
            'sap-client': self.sap_client
        }

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }

        response = await self.session.get(url, params=params, headers=headers)
        response.raise_for_status()

        data = response.json()
        return data.get('d', {}).get('results', [])

    async def post_journal_entry(self, journal_entry: Dict) -> Dict:
        """
        Post journal entry to SAP for carbon accounting.

        SAP OData Entity: /sap/opu/odata/sap/API_ODATA_JOURNAL_ENTRY_SRV/JournalEntryBulkCreateRequest
        """
        await self.authenticate()

        url = f"{self.base_url}/sap/opu/odata/sap/API_ODATA_JOURNAL_ENTRY_SRV/JournalEntryBulkCreateRequest"

        # Get CSRF token
        csrf_response = await self.session.head(url, headers={
            'Authorization': f'Bearer {self.access_token}',
            'x-csrf-token': 'Fetch'
        })
        csrf_token = csrf_response.headers.get('x-csrf-token')

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'x-csrf-token': csrf_token
        }

        response = await self.session.post(url, json=journal_entry, headers=headers)
        response.raise_for_status()

        return response.json()

# ==============================================
# ORACLE ERP CLOUD CONNECTOR
# ==============================================

class OracleERPConnector:
    """
    Oracle ERP Cloud REST API Connector
    Supports Fusion Applications REST APIs
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Oracle ERP connector."""
        self.base_url = config['base_url']
        self.username = config['username']
        self.password = config['password']
        self.tenant_id = config.get('tenant_id')

        self.session = httpx.AsyncClient(
            timeout=30,
            auth=(self.username, self.password)
        )

    async def fetch_purchase_orders(
        self,
        start_date: str,
        end_date: str,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch purchase orders from Oracle ERP.

        Oracle REST API: /fscmRestApi/resources/11.13.18.05/purchaseOrders
        """
        url = f"{self.base_url}/fscmRestApi/resources/11.13.18.05/purchaseOrders"

        # Build query
        query_parts = [
            f"CreationDate >= '{start_date}'",
            f"CreationDate <= '{end_date}'"
        ]
        if status:
            query_parts.append(f"DocumentStatus = '{status}'")

        params = {
            'q': ';'.join(query_parts),
            'fields': 'POHeaderId,OrderNumber,Supplier,SupplierSite,CurrencyCode,Total,CreationDate,DocumentStatus',
            'expand': 'lines',
            'limit': 500,
            'offset': 0
        }

        all_records = []

        while True:
            response = await self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            items = data.get('items', [])

            if not items:
                break

            all_records.extend(items)

            # Check for more pages
            if data.get('hasMore', False):
                params['offset'] += params['limit']
            else:
                break

        logger.info(f"Fetched {len(all_records)} purchase orders from Oracle ERP")
        return all_records

    async def fetch_suppliers(
        self,
        supplier_type: Optional[str] = None,
        country: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch supplier data from Oracle ERP.

        Oracle REST API: /fscmRestApi/resources/11.13.18.05/suppliers
        """
        url = f"{self.base_url}/fscmRestApi/resources/11.13.18.05/suppliers"

        query_parts = []
        if supplier_type:
            query_parts.append(f"SupplierType = '{supplier_type}'")
        if country:
            query_parts.append(f"Country = '{country}'")

        params = {
            'fields': 'SupplierId,SupplierNumber,SupplierName,SupplierType,TaxpayerIdentificationNumber,DUNSNumber',
            'expand': 'addresses,sites,contacts',
            'limit': 500
        }

        if query_parts:
            params['q'] = ';'.join(query_parts)

        response = await self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return data.get('items', [])

    async def fetch_invoices(
        self,
        start_date: str,
        end_date: str,
        invoice_type: str = 'STANDARD'
    ) -> List[Dict]:
        """
        Fetch AP invoices from Oracle ERP.

        Oracle REST API: /fscmRestApi/resources/11.13.18.05/invoices
        """
        url = f"{self.base_url}/fscmRestApi/resources/11.13.18.05/invoices"

        params = {
            'q': f"InvoiceDate >= '{start_date}';InvoiceDate <= '{end_date}';InvoiceType = '{invoice_type}'",
            'fields': 'InvoiceId,InvoiceNumber,InvoiceDate,InvoiceAmount,InvoiceCurrency,Supplier,Description',
            'expand': 'invoiceLines,invoiceDistributions',
            'limit': 500
        }

        response = await self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return data.get('items', [])

    async def fetch_general_ledger_balances(
        self,
        period_name: str,
        ledger_id: int,
        account_segments: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Fetch GL balances for carbon accounting.

        Oracle REST API: /fscmRestApi/resources/11.13.18.05/ledgerBalances
        """
        url = f"{self.base_url}/fscmRestApi/resources/11.13.18.05/ledgerBalances"

        query_parts = [
            f"PeriodName = '{period_name}'",
            f"LedgerId = {ledger_id}"
        ]

        if account_segments:
            for segment, value in account_segments.items():
                query_parts.append(f"{segment} = '{value}'")

        params = {
            'q': ';'.join(query_parts),
            'fields': 'AccountCombination,PeriodName,Currency,BeginningBalance,PeriodActivity,EndingBalance',
            'limit': 1000
        }

        response = await self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return data.get('items', [])

# ==============================================
# WORKDAY CONNECTOR
# ==============================================

class WorkdayConnector:
    """
    Workday REST and SOAP API Connector
    Supports HCM, Financial Management, and Procurement
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Workday connector."""
        self.base_url = config['base_url']
        self.tenant = config['tenant']
        self.username = config['username']
        self.password = config['password']
        self.api_version = config.get('api_version', 'v35.0')

        self.session = httpx.AsyncClient(
            timeout=30,
            auth=(f"{self.username}@{self.tenant}", self.password)
        )

    async def fetch_workers(
        self,
        effective_date: Optional[str] = None,
        location: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch worker data from Workday HCM.

        Workday REST API: /ccx/service/customreport2/{tenant}/{report_owner}/{report_name}
        """
        if not effective_date:
            effective_date = DeterministicClock.now().strftime('%Y-%m-%d')

        url = f"{self.base_url}/ccx/service/customreport2/{self.tenant}/ISU_Sustainability/Workers_Report"

        params = {
            'Effective_Date': effective_date,
            'format': 'json'
        }

        if location:
            params['Location'] = location

        response = await self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return data.get('Report_Entry', [])

    async def fetch_purchase_orders(
        self,
        start_date: str,
        end_date: str,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch purchase orders from Workday Procurement.

        Workday REST API: Resource Management domain
        """
        url = f"{self.base_url}/ccx/api/{self.api_version}/{self.tenant}/purchaseOrders"

        params = {
            'fromDate': start_date,
            'toDate': end_date,
            'limit': 100,
            'offset': 0
        }

        if status:
            params['status'] = status

        all_records = []

        while True:
            response = await self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            records = data.get('data', [])

            if not records:
                break

            all_records.extend(records)

            # Check for more pages
            if len(records) == params['limit']:
                params['offset'] += params['limit']
            else:
                break

        return all_records

    async def fetch_suppliers(
        self,
        supplier_category: Optional[str] = None,
        include_inactive: bool = False
    ) -> List[Dict]:
        """
        Fetch supplier data from Workday.

        Workday SOAP API: Get_Suppliers operation
        """
        soap_url = f"{self.base_url}/ccx/service/{self.tenant}/Resource_Management/{self.api_version}"

        # Build SOAP request
        soap_body = f"""
        <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"
                         xmlns:wd="urn:com.workday/bsvc">
            <soapenv:Header/>
            <soapenv:Body>
                <wd:Get_Suppliers_Request>
                    <wd:Request_Criteria>
                        <wd:Include_Inactive_Suppliers>{str(include_inactive).lower()}</wd:Include_Inactive_Suppliers>
                        {f'<wd:Supplier_Category>{supplier_category}</wd:Supplier_Category>' if supplier_category else ''}
                    </wd:Request_Criteria>
                    <wd:Response_Group>
                        <wd:Include_Reference>true</wd:Include_Reference>
                        <wd:Include_Supplier_Data>true</wd:Include_Supplier_Data>
                    </wd:Response_Group>
                </wd:Get_Suppliers_Request>
            </soapenv:Body>
        </soapenv:Envelope>
        """

        headers = {
            'Content-Type': 'text/xml; charset=utf-8',
            'SOAPAction': 'Get_Suppliers'
        }

        response = await self.session.post(soap_url, content=soap_body, headers=headers)
        response.raise_for_status()

        # Parse SOAP response
        root = ET.fromstring(response.text)
        suppliers = []

        for supplier_elem in root.findall('.//wd:Supplier', namespaces={'wd': 'urn:com.workday/bsvc'}):
            supplier_data = {
                'supplier_id': supplier_elem.find('.//wd:Supplier_ID', namespaces={'wd': 'urn:com.workday/bsvc'}).text,
                'supplier_name': supplier_elem.find('.//wd:Supplier_Name', namespaces={'wd': 'urn:com.workday/bsvc'}).text,
                'tax_id': supplier_elem.find('.//wd:Tax_ID', namespaces={'wd': 'urn:com.workday/bsvc'}).text if supplier_elem.find('.//wd:Tax_ID', namespaces={'wd': 'urn:com.workday/bsvc'}) else None
            }
            suppliers.append(supplier_data)

        return suppliers

    async def fetch_expense_reports(
        self,
        start_date: str,
        end_date: str,
        expense_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch expense reports for travel emissions tracking.

        Workday REST API: Expense Management
        """
        url = f"{self.base_url}/ccx/api/{self.api_version}/{self.tenant}/expenseReports"

        params = {
            'fromDate': start_date,
            'toDate': end_date,
            'limit': 100
        }

        if expense_type:
            params['expenseType'] = expense_type

        response = await self.session.get(url, params=params)
        response.raise_for_status()

        return response.json().get('data', [])

# ==============================================
# MICROSOFT DYNAMICS 365 CONNECTOR
# ==============================================

class Dynamics365Connector:
    """
    Microsoft Dynamics 365 Web API Connector
    Supports Finance, Supply Chain, and Customer Engagement
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Dynamics 365 connector."""
        self.base_url = config['base_url']
        self.tenant_id = config['tenant_id']
        self.client_id = config['client_id']
        self.client_secret = config['client_secret']
        self.resource = config.get('resource', 'https://api.businesscentral.dynamics.com')

        self.session = httpx.AsyncClient(timeout=30)
        self.access_token = None
        self.token_expires_at = None

    async def authenticate(self) -> str:
        """Authenticate with Azure AD for Dynamics 365."""
        if self.access_token and self.token_expires_at > DeterministicClock.now():
            return self.access_token

        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/token"

        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'resource': self.resource
        }

        response = await self.session.post(token_url, data=auth_data)
        response.raise_for_status()

        token_data = response.json()
        self.access_token = token_data['access_token']
        expires_in = token_data.get('expires_in', 3600)
        self.token_expires_at = DeterministicClock.now() + timedelta(seconds=expires_in - 60)

        return self.access_token

    async def fetch_purchase_orders(
        self,
        company_id: str,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """
        Fetch purchase orders from Dynamics 365.

        Dynamics 365 Web API: /api/v2.0/companies({id})/purchaseOrders
        """
        await self.authenticate()

        url = f"{self.base_url}/api/v2.0/companies({company_id})/purchaseOrders"

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json',
            'OData-Version': '4.0'
        }

        # OData filter
        filter_query = f"orderDate ge {start_date} and orderDate le {end_date}"

        params = {
            '$filter': filter_query,
            '$expand': 'purchaseOrderLines',
            '$top': 100
        }

        all_records = []

        while True:
            response = await self.session.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            records = data.get('value', [])

            if not records:
                break

            all_records.extend(records)

            # Check for next page
            next_link = data.get('@odata.nextLink')
            if next_link:
                url = next_link
                params = {}  # Next link includes all params
            else:
                break

        return all_records

    async def fetch_vendors(
        self,
        company_id: str,
        country_code: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch vendor data from Dynamics 365.

        Dynamics 365 Web API: /api/v2.0/companies({id})/vendors
        """
        await self.authenticate()

        url = f"{self.base_url}/api/v2.0/companies({company_id})/vendors"

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }

        params = {'$top': 100}

        if country_code:
            params['$filter'] = f"countryRegionCode eq '{country_code}'"

        response = await self.session.get(url, params=params, headers=headers)
        response.raise_for_status()

        return response.json().get('value', [])

    async def fetch_general_ledger_entries(
        self,
        company_id: str,
        posting_date_from: str,
        posting_date_to: str,
        account_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch GL entries for carbon accounting.

        Dynamics 365 Web API: /api/v2.0/companies({id})/generalLedgerEntries
        """
        await self.authenticate()

        url = f"{self.base_url}/api/v2.0/companies({company_id})/generalLedgerEntries"

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }

        filter_parts = [
            f"postingDate ge {posting_date_from}",
            f"postingDate le {posting_date_to}"
        ]

        if account_filter:
            filter_parts.append(f"accountNumber eq '{account_filter}'")

        params = {
            '$filter': ' and '.join(filter_parts),
            '$top': 1000
        }

        response = await self.session.get(url, params=params, headers=headers)
        response.raise_for_status()

        return response.json().get('value', [])

# ==============================================
# NETSUITE CONNECTOR
# ==============================================

class NetSuiteConnector:
    """
    NetSuite SuiteTalk REST API Connector
    Supports ERP, CRM, and E-commerce operations
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize NetSuite connector."""
        self.account_id = config['account_id']
        self.consumer_key = config['consumer_key']
        self.consumer_secret = config['consumer_secret']
        self.token_id = config['token_id']
        self.token_secret = config['token_secret']
        self.base_url = f"https://{config['account_id']}.suitetalk.api.netsuite.com"

        self.session = httpx.AsyncClient(timeout=30)

    def _generate_oauth_header(self, method: str, url: str) -> str:
        """Generate OAuth 1.0a header for NetSuite."""
        import hashlib
        import hmac
        import base64
        import time
        import random
        from urllib.parse import quote

        timestamp = str(int(time.time()))
        nonce = str(deterministic_random().randint(100000, 999999))

        # Create signature base string
        params = {
            'oauth_consumer_key': self.consumer_key,
            'oauth_nonce': nonce,
            'oauth_signature_method': 'HMAC-SHA256',
            'oauth_timestamp': timestamp,
            'oauth_token': self.token_id,
            'oauth_version': '1.0'
        }

        param_string = '&'.join([f"{k}={quote(v)}" for k, v in sorted(params.items())])
        base_string = f"{method}&{quote(url)}&{quote(param_string)}"

        # Create signature
        signing_key = f"{self.consumer_secret}&{self.token_secret}"
        signature = base64.b64encode(
            hmac.new(
                signing_key.encode(),
                base_string.encode(),
                hashlib.sha256
            ).digest()
        ).decode()

        # Build OAuth header
        oauth_header = 'OAuth realm="' + self.account_id + '", '
        oauth_header += ', '.join([f'{k}="{v}"' for k, v in params.items()])
        oauth_header += f', oauth_signature="{quote(signature)}"'

        return oauth_header

    async def fetch_purchase_orders(
        self,
        start_date: str,
        end_date: str,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch purchase orders from NetSuite.

        NetSuite REST API: /services/rest/record/v1/purchaseOrder
        """
        url = f"{self.base_url}/services/rest/record/v1/purchaseOrder"

        headers = {
            'Authorization': self._generate_oauth_header('GET', url),
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

        # Build query
        query_parts = [
            f"tranDate AFTER {start_date}",
            f"tranDate BEFORE {end_date}"
        ]

        if status:
            query_parts.append(f"status IS {status}")

        params = {
            'q': ' AND '.join(query_parts),
            'limit': 100,
            'offset': 0
        }

        all_records = []

        while True:
            response = await self.session.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            items = data.get('items', [])

            if not items:
                break

            all_records.extend(items)

            # Check for more records
            if data.get('hasMore'):
                params['offset'] += params['limit']
            else:
                break

        return all_records

    async def fetch_vendors(
        self,
        category: Optional[str] = None,
        subsidiary: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch vendor data from NetSuite.

        NetSuite REST API: /services/rest/record/v1/vendor
        """
        url = f"{self.base_url}/services/rest/record/v1/vendor"

        headers = {
            'Authorization': self._generate_oauth_header('GET', url),
            'Accept': 'application/json'
        }

        query_parts = []
        if category:
            query_parts.append(f"category IS {category}")
        if subsidiary:
            query_parts.append(f"subsidiary IS {subsidiary}")

        params = {'limit': 100}
        if query_parts:
            params['q'] = ' AND '.join(query_parts)

        response = await self.session.get(url, params=params, headers=headers)
        response.raise_for_status()

        return response.json().get('items', [])

    async def fetch_carbon_transactions(
        self,
        start_date: str,
        end_date: str,
        account_type: str = 'CarbonOffset'
    ) -> List[Dict]:
        """
        Fetch carbon-related transactions from NetSuite.

        Custom record type for sustainability tracking
        """
        url = f"{self.base_url}/services/rest/record/v1/customrecord_carbon_transaction"

        headers = {
            'Authorization': self._generate_oauth_header('GET', url),
            'Accept': 'application/json'
        }

        params = {
            'q': f"custrecord_date AFTER {start_date} AND custrecord_date BEFORE {end_date} AND custrecord_type IS {account_type}",
            'limit': 100
        }

        response = await self.session.get(url, params=params, headers=headers)
        response.raise_for_status()

        return response.json().get('items', [])

# ==============================================
# UNIFIED ERP CONNECTOR INTERFACE
# ==============================================

class UnifiedERPConnector:
    """
    Unified interface for all ERP connectors.
    Provides abstraction layer for multiple ERP systems.
    """

    def __init__(self):
        """Initialize unified connector."""
        self.connectors = {}

    def register_connector(self, system: str, connector: Any):
        """Register an ERP connector."""
        self.connectors[system.lower()] = connector

    async def fetch_purchase_orders(
        self,
        system: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> List[Dict]:
        """Fetch purchase orders from any registered ERP system."""
        connector = self.connectors.get(system.lower())
        if not connector:
            raise ValueError(f"No connector registered for system: {system}")

        # Normalize data from different ERPs
        raw_data = await connector.fetch_purchase_orders(start_date, end_date, **kwargs)

        normalized_data = []
        for record in raw_data:
            normalized = self._normalize_purchase_order(system, record)
            normalized_data.append(normalized)

        return normalized_data

    def _normalize_purchase_order(self, system: str, record: Dict) -> Dict:
        """Normalize purchase order data from different ERPs."""
        normalized = {
            'source_system': system,
            'original_data': record,
            'extracted_at': DeterministicClock.now().isoformat()
        }

        # Map fields based on source system
        if system.lower() == 'sap':
            normalized.update({
                'po_number': record.get('PurchaseOrder'),
                'supplier_id': record.get('Supplier'),
                'total_amount': FinancialDecimal.from_string(record.get('TotalNetAmount', 0)),
                'currency': record.get('DocumentCurrency'),
                'created_date': record.get('CreationDate')
            })
        elif system.lower() == 'oracle':
            normalized.update({
                'po_number': record.get('OrderNumber'),
                'supplier_id': record.get('Supplier'),
                'total_amount': FinancialDecimal.from_string(record.get('Total', 0)),
                'currency': record.get('CurrencyCode'),
                'created_date': record.get('CreationDate')
            })
        elif system.lower() == 'workday':
            normalized.update({
                'po_number': record.get('purchaseOrderNumber'),
                'supplier_id': record.get('supplierId'),
                'total_amount': FinancialDecimal.from_string(record.get('totalAmount', 0)),
                'currency': record.get('currency'),
                'created_date': record.get('createdDate')
            })
        # Add more system mappings as needed

        return normalized

# ==============================================
# USAGE EXAMPLE
# ==============================================

async def main():
    """Example usage of ERP connectors."""

    # Initialize SAP connector
    sap_config = {
        'base_url': 'https://my-sap-system.com',
        'client_id': 'client_id',
        'client_secret': 'client_secret',
        'oauth_url': 'https://auth.sap.com/oauth/token',
        'sap_client': '100'
    }
    sap = SAPConnector(sap_config)

    # Fetch purchase orders from SAP
    po_data = await sap.fetch_purchase_orders(
        start_date='2024-01-01',
        end_date='2024-01-31',
        company_code='1000'
    )

    # Initialize unified connector
    unified = UnifiedERPConnector()
    unified.register_connector('SAP', sap)

    # Fetch normalized data
    normalized_pos = await unified.fetch_purchase_orders(
        system='SAP',
        start_date='2024-01-01',
        end_date='2024-01-31'
    )

    print(f"Fetched {len(normalized_pos)} purchase orders")

if __name__ == "__main__":
    asyncio.run(main())