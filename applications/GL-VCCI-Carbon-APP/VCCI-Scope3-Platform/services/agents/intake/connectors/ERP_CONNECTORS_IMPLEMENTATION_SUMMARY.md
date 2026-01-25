# ERP Connectors Implementation Summary

## Overview
Production-ready ERP connectors have been implemented for SAP S/4HANA, Oracle ERP Cloud, and Workday, providing robust data integration capabilities for GreenLang's data intake system.

## Implementation Status ✅

### 1. SAP S/4HANA Connector (`sap_connector.py`)

#### Features Implemented:
- **Authentication**: OAuth2 with automatic token refresh
- **API Integration**: OData REST API for S/4HANA and ECC
- **Data Extraction**:
  - Purchase Orders (EKKO/EKPO tables)
  - Suppliers (LFA1)
  - Materials (MARA)
  - Deliveries/Shipments
  - Invoices
- **Production Features**:
  - Connection pooling (10 connections, 5 keepalive)
  - Rate limiting (100 requests/minute default)
  - Retry logic with exponential backoff
  - Pagination support (1000 records/page)
  - Deduplication cache integration
  - Comprehensive error handling

#### API Endpoints:
```python
# Purchase Orders
/sap/opu/odata/sap/API_PURCHASEORDER_PROCESS_SRV/A_PurchaseOrder

# Suppliers
/sap/opu/odata/sap/API_BUSINESS_PARTNER/A_Supplier

# Materials
/sap/opu/odata/sap/API_MATERIAL_DOCUMENT_SRV/A_MaterialDocument

# Deliveries
/sap/opu/odata/sap/API_OUTBOUND_DELIVERY_SRV/A_OutbDeliveryHeader

# Invoices
/sap/opu/odata/sap/API_SUPPLIERINVOICE_PROCESS_SRV/A_SupplierInvoice
```

#### Authentication Flow:
1. OAuth2 client credentials grant
2. Token caching with automatic refresh (5 min before expiry)
3. Bearer token in Authorization header

#### Sample Usage:
```python
from services.agents.intake.connectors.sap_connector import SAPConnector

# Initialize with environment variables
connector = SAPConnector()

# Or with explicit credentials
credentials = {
    'base_url': 'https://api.s4hana.example.com',
    'client_id': 'your_client_id',
    'client_secret': 'your_secret',
    'oauth_token_url': 'https://auth.s4hana.example.com/oauth/token',
    'company_code': '1000'
}
connector = SAPConnector(credentials)

# Connect and authenticate
if connector.connect():
    # Get purchase orders
    pos = connector.get_purchase_orders(
        start_date='2024-01-01',
        end_date='2024-01-31',
        plant_codes=['PL01', 'PL02']
    )

    # Get suppliers
    suppliers = connector.get_suppliers(
        supplier_ids=['0000100000', '0000100001']
    )

    # Get shipments/deliveries
    shipments = connector.get_shipments(
        start_date='2024-01-01',
        end_date='2024-01-31'
    )

    connector.disconnect()
```

---

### 2. Oracle ERP Cloud Connector (`oracle_connector.py`)

#### Features Implemented:
- **Authentication**: Basic Auth and JWT support
- **API Integration**: Oracle REST Data Services (ORDS)
- **Data Extraction**:
  - Purchase Orders & Lines
  - Requisitions
  - Suppliers & Sites
  - Invoices
  - Inventory Transactions
  - Receipts
  - Shipments
  - Work Orders
  - Projects & Contracts
- **Production Features**:
  - Connection pooling
  - Rate limiting with 429 response handling
  - Automatic retry with exponential backoff
  - Pagination support (500 records/page)
  - Bulk data extraction
  - Multiple Oracle modules support (SCM, FIN, HCM)

#### API Endpoints:
```python
# Procurement
/fscmRestApi/resources/11.13.18.05/purchaseOrders
/fscmRestApi/resources/11.13.18.05/requisitions
/fscmRestApi/resources/11.13.18.05/suppliers

# Finance
/fscmRestApi/resources/11.13.18.05/invoices
/fscmRestApi/resources/11.13.18.05/receipts

# Supply Chain
/fscmRestApi/resources/11.13.18.05/inventoryTransactions
/fscmRestApi/resources/11.13.18.05/shipments
/fscmRestApi/resources/11.13.18.05/workOrders

# Projects
/pjfRestApi/resources/11.13.18.05/projects
```

#### Authentication Flow:
1. Basic Authentication (default)
2. JWT token support (configurable)
3. Automatic 429 rate limit handling with Retry-After header

#### Sample Usage:
```python
from services.agents.intake.connectors.oracle_connector import OracleConnector

# Initialize with environment variables
connector = OracleConnector()

# Or with explicit credentials
credentials = {
    'base_url': 'https://example.fa.us2.oraclecloud.com',
    'username': 'your_username',
    'password': 'your_password',
    'tenant_name': 'your_tenant'
}
connector = OracleConnector(credentials)

# Connect
if connector.connect():
    # Get purchase orders
    pos = connector.get_purchase_orders(
        start_date='2024-01-01',
        end_date='2024-01-31',
        business_unit='US_BU'
    )

    # Get active suppliers
    suppliers = connector.get_suppliers(status='ACTIVE')

    # Get inventory transactions
    transactions = connector.get_inventory_transactions(
        start_date='2024-01-01',
        end_date='2024-01-31',
        organization='ORG1'
    )

    connector.disconnect()
```

---

### 3. Workday Connector (`workday_connector.py`)

#### Features Implemented:
- **Authentication**: OAuth2 with refresh token support
- **API Integration**: REST API v35.0 and RAAS
- **Data Extraction**:
  - Suppliers & Spend Categories
  - Purchase Orders & Requisitions
  - Invoices & Expense Reports
  - Supplier Contracts
  - Procurement Card Transactions
  - Workers & Contingent Workers
  - Spend Analytics
  - Custom RAAS Reports
- **Production Features**:
  - OAuth2 with automatic refresh
  - RAAS (Report-as-a-Service) integration
  - Rate limiting (60 requests/minute)
  - Retry logic with exponential backoff
  - Pagination support
  - Custom report support

#### API Endpoints:
```python
# Financial Management
/api/v35.0/suppliers
/api/v35.0/spend_categories
/api/v35.0/purchaseOrders
/api/v35.0/requisitions
/api/v35.0/supplier_invoices
/api/v35.0/expense_reports

# HCM
/api/v35.0/workers
/api/v35.0/contingentWorkers

# Analytics
/api/financialManagement/v1/spendAnalytics

# RAAS Custom Reports
/raas/customreport2/{tenant}/{report_name}
```

#### Authentication Flow:
1. OAuth2 client credentials grant
2. Refresh token support for extended sessions
3. Automatic token refresh before expiry

#### Sample Usage:
```python
from services.agents.intake.connectors.workday_connector import WorkdayConnector

# Initialize with environment variables
connector = WorkdayConnector()

# Or with explicit credentials
credentials = {
    'base_url': 'https://wd2-impl-services1.workday.com',
    'tenant': 'your_tenant',
    'username': 'your_username',
    'password': 'your_password',
    'client_id': 'your_client_id',
    'client_secret': 'your_client_secret'
}
connector = WorkdayConnector(credentials)

# Connect
if connector.connect():
    # Get spend analytics
    analytics = connector.get_spend_analytics(
        start_date='2024-01-01',
        end_date='2024-03-31',
        spend_category='IT Equipment'
    )

    # Get active suppliers
    suppliers = connector.get_suppliers(status='ACTIVE')

    # Get purchase orders
    pos = connector.get_purchase_orders(
        start_date='2024-01-01',
        end_date='2024-01-31',
        company='Acme Corporation'
    )

    # Get custom RAAS report
    custom_report = connector.query({
        'entity_type': 'supplier_spend_report',
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'filters': {'Spend_Threshold': 100000}
    })

    connector.disconnect()
```

---

## Common Features Across All Connectors

### 1. Environment Variable Support
All connectors support configuration via environment variables:
```bash
# SAP
export SAP_BASE_URL="https://api.s4hana.example.com"
export SAP_CLIENT_ID="your_client_id"
export SAP_CLIENT_SECRET="your_client_secret"
export SAP_OAUTH_URL="https://auth.s4hana.example.com/oauth/token"
export SAP_COMPANY_CODE="1000"

# Oracle
export ORACLE_BASE_URL="https://example.fa.us2.oraclecloud.com"
export ORACLE_USERNAME="your_username"
export ORACLE_PASSWORD="your_password"
export ORACLE_TENANT="your_tenant"

# Workday
export WORKDAY_BASE_URL="https://wd2-impl-services1.workday.com"
export WORKDAY_TENANT="your_tenant"
export WORKDAY_USERNAME="your_username"
export WORKDAY_PASSWORD="your_password"
export WORKDAY_CLIENT_ID="your_client_id"
export WORKDAY_CLIENT_SECRET="your_client_secret"
```

### 2. Rate Limiting
- Configurable requests per minute
- Automatic throttling when limits reached
- Sleep/wait implementation to respect API limits

### 3. Retry Logic
- Exponential backoff (2-10 seconds)
- Default 3 retry attempts
- Handles transient network errors

### 4. Pagination
- Automatic handling of large datasets
- Configurable page sizes
- Support for limit parameter

### 5. Data Transformation
All connectors transform ERP-specific data to GreenLang standard format:
```python
{
    'id': 'unique_identifier',
    'type': 'entity_type',
    'source_system': 'SAP/Oracle/Workday',
    'extracted_at': 'ISO timestamp',
    # Entity-specific fields
    'metadata': {
        # Additional ERP-specific data
    }
}
```

### 6. Connection Pooling
- HTTP connection reuse
- Configurable pool sizes
- Keepalive connections

### 7. Error Handling
- Comprehensive exception handling
- Detailed logging
- Graceful degradation

---

## Testing Coverage

### Integration Tests Implemented:
1. **Authentication Tests**
   - Success scenarios
   - Failure handling
   - Token refresh

2. **Data Extraction Tests**
   - Purchase orders
   - Suppliers
   - Inventory/Materials
   - Custom queries

3. **Pagination Tests**
   - Large dataset handling
   - Page size limits

4. **Rate Limiting Tests**
   - Throttling enforcement
   - 429 response handling

5. **Retry Logic Tests**
   - Transient failure recovery
   - Exponential backoff

6. **Transformation Tests**
   - Data format conversion
   - Field mapping validation

---

## Data Flow Examples

### 1. Purchase Order Extraction Flow
```
ERP System → OAuth/Auth → API Request → Pagination →
Transform → Deduplication → Standard Format → Return
```

### 2. Incremental Sync Flow
```
Query with Date Range → Filter Changed Records →
Deduplicate → Transform → Update Cache → Return New Records
```

### 3. Error Recovery Flow
```
API Request → Error → Retry with Backoff →
Still Error → Log → Return Partial Results →
Dead Letter Queue
```

---

## Security Considerations

1. **Credential Management**
   - Never hardcode credentials
   - Use environment variables or vault
   - SecretStr type for sensitive data

2. **Authentication**
   - OAuth2 preferred over basic auth
   - Automatic token refresh
   - Secure token storage

3. **Network Security**
   - HTTPS only
   - Connection pooling with limits
   - Request signing where supported

---

## Performance Metrics

### Expected Performance:
- **SAP**: 100 requests/minute, ~5000 records/minute
- **Oracle**: 60 requests/minute, ~3000 records/minute
- **Workday**: 60 requests/minute, ~2000 records/minute

### Optimization Features:
- Connection pooling reduces latency
- Batch processing for bulk operations
- Deduplication cache prevents reprocessing
- Parallel processing support

---

## Monitoring & Observability

All connectors include:
- Structured logging with levels
- Request/response timing
- Error tracking
- Rate limit monitoring
- Connection health checks

---

## Next Steps & Recommendations

1. **Production Deployment**:
   - Set up credential vault integration
   - Configure monitoring dashboards
   - Establish SLAs for data freshness

2. **Performance Tuning**:
   - Adjust rate limits based on ERP capacity
   - Optimize page sizes for network conditions
   - Implement caching for reference data

3. **Additional Features**:
   - WebSocket support for real-time updates
   - GraphQL endpoint support
   - Custom field mapping configuration

4. **Compliance**:
   - Audit logging for data access
   - GDPR compliance for personal data
   - Data retention policies

---

## Files Delivered

1. **Connectors**:
   - `/services/agents/intake/connectors/sap_connector.py` - SAP S/4HANA connector
   - `/services/agents/intake/connectors/oracle_connector.py` - Oracle ERP Cloud connector
   - `/services/agents/intake/connectors/workday_connector.py` - Workday connector

2. **Tests**:
   - `/services/agents/intake/connectors/tests/test_sap_connector.py` - SAP tests
   - `/services/agents/intake/connectors/tests/test_oracle_connector.py` - Oracle tests
   - `/services/agents/intake/connectors/tests/test_workday_connector.py` - Workday tests

3. **Documentation**:
   - This summary document

---

## Contact & Support

For questions or issues with the ERP connectors:
- Review test files for usage examples
- Check environment variable configuration
- Verify network connectivity to ERP systems
- Enable debug logging for troubleshooting

The connectors are designed to be production-ready with comprehensive error handling, retry logic, and monitoring capabilities suitable for enterprise deployment.