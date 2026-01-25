"""
SAP Ariba Connector.

This module provides integration with SAP Ariba for supplier master data
and procurement transaction import.

Supported APIs:
- Supplier Management APIs (Supplier Data)
- Procurement APIs (Purchase Orders, Contracts)
- Analytics APIs (Spend Analysis)

Authentication:
- OAuth 2.0 with Client Credentials
- API Key authentication for legacy endpoints

Example:
    >>> from greenlang.supply_chain.connectors import SAPAribaConnector
    >>> connector = SAPAribaConnector(
    ...     api_base_url="https://api.ariba.com",
    ...     realm="my-realm",
    ...     client_id="your-client-id",
    ...     client_secret="your-client-secret"
    ... )
    >>> suppliers = connector.get_suppliers()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any, Generator
import json

from greenlang.supply_chain.models.entity import (
    Supplier,
    Address,
    ExternalIdentifiers,
    ContactInfo,
    SupplierTier,
    SupplierStatus,
)

logger = logging.getLogger(__name__)


class AribaEndpoint(Enum):
    """SAP Ariba API endpoints."""
    SUPPLIERS = "/slm/sourcing/suppliers"
    SUPPLIER_DATA = "/slm/sourcing/supplierdata"
    PURCHASE_ORDERS = "/procurement/purchaseorders"
    CONTRACTS = "/contracts/ws"
    SPEND_ANALYSIS = "/analytics/spend"
    SUPPLIER_RISK = "/slm/sourcing/supplier-risk"


@dataclass
class AribaSupplierRecord:
    """
    Supplier record from SAP Ariba.

    Attributes:
        system_id: Ariba internal system ID
        an_id: Ariba Network ID (ANID)
        vendor_id: SAP Vendor ID (if linked)
        company_name: Legal company name
        duns_number: D-U-N-S number
        tax_id: Tax identification number
        country: Country code
        address: Address details
        contact_email: Primary contact email
        qualification_status: Ariba qualification status
        risk_score: Supplier risk score
        spend_category: Primary spend category
        created_date: Record creation date
        last_modified: Last modification date
        custom_fields: Custom field values
    """
    system_id: str
    company_name: str
    an_id: Optional[str] = None
    vendor_id: Optional[str] = None
    duns_number: Optional[str] = None
    tax_id: Optional[str] = None
    country: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    qualification_status: str = "pending"
    risk_score: Optional[float] = None
    spend_category: Optional[str] = None
    industry_code: Optional[str] = None
    created_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_supplier(self) -> Supplier:
        """
        Convert Ariba record to Supplier entity.

        Returns:
            Supplier entity
        """
        # Map qualification status
        status_mapping = {
            "qualified": SupplierStatus.QUALIFIED,
            "approved": SupplierStatus.QUALIFIED,
            "pending": SupplierStatus.PENDING_QUALIFICATION,
            "blocked": SupplierStatus.BLOCKED,
            "inactive": SupplierStatus.INACTIVE,
        }
        status = status_mapping.get(
            self.qualification_status.lower(),
            SupplierStatus.PENDING_QUALIFICATION
        )

        # Build address
        address = None
        if self.street or self.city:
            address = Address(
                street_line_1=self.street or "",
                city=self.city or "",
                state_province=self.state,
                postal_code=self.postal_code,
                country_code=self.country or "",
            )

        # Build external identifiers
        external_ids = ExternalIdentifiers(
            duns=self.duns_number,
            vat_number=self.tax_id,
            ariba_network_id=self.an_id,
            sap_vendor_id=self.vendor_id,
        )

        # Build contact info
        contact = None
        if self.contact_email or self.contact_name:
            contact = ContactInfo(
                primary_contact_name=self.contact_name,
                primary_email=self.contact_email,
                primary_phone=self.contact_phone,
            )

        # Industry codes
        industry_codes = {}
        if self.industry_code:
            industry_codes["NAICS"] = self.industry_code

        return Supplier(
            id=f"ARIBA-{self.system_id}",
            name=self.company_name,
            tier=SupplierTier.TIER_1,  # Ariba suppliers are typically Tier 1
            country_code=self.country,
            status=status,
            external_ids=external_ids,
            address=address,
            contact=contact,
            industry_codes=industry_codes,
            metadata={
                "source": "sap_ariba",
                "ariba_system_id": self.system_id,
                "risk_score": self.risk_score,
                "spend_category": self.spend_category,
                **self.custom_fields,
            },
        )


@dataclass
class AribaPORecord:
    """
    Purchase Order record from SAP Ariba.

    Attributes:
        po_number: Purchase Order number
        supplier_id: Supplier system ID
        supplier_name: Supplier name
        total_amount: Total PO amount
        currency: Currency code
        status: PO status
        created_date: PO creation date
        items: PO line items
    """
    po_number: str
    supplier_id: str
    supplier_name: str
    total_amount: Decimal
    currency: str = "USD"
    status: str = "open"
    created_date: Optional[datetime] = None
    items: List[Dict[str, Any]] = field(default_factory=list)


class SAPAribaConnector:
    """
    Connector for SAP Ariba integration.

    Provides methods to fetch supplier data, purchase orders, and spend
    information from SAP Ariba APIs.

    Example:
        >>> connector = SAPAribaConnector(
        ...     api_base_url="https://api.ariba.com",
        ...     realm="my-realm",
        ...     client_id="your-client-id",
        ...     client_secret="your-client-secret"
        ... )
        >>>
        >>> # Get all suppliers
        >>> suppliers = connector.get_suppliers()
        >>>
        >>> # Get spend by supplier
        >>> spend = connector.get_spend_by_supplier(year=2024)
    """

    def __init__(
        self,
        api_base_url: str,
        realm: str,
        client_id: str,
        client_secret: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize the SAP Ariba connector.

        Args:
            api_base_url: Base URL for Ariba API
            realm: Ariba realm identifier
            client_id: OAuth client ID
            client_secret: OAuth client secret
            api_key: Optional API key for legacy endpoints
            timeout: Request timeout in seconds
        """
        self.api_base_url = api_base_url.rstrip("/")
        self.realm = realm
        self.client_id = client_id
        self.client_secret = client_secret
        self.api_key = api_key
        self.timeout = timeout

        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        logger.info(f"SAPAribaConnector initialized for realm: {realm}")

    def _get_access_token(self) -> str:
        """
        Get OAuth access token.

        Returns:
            Access token string

        Raises:
            ConnectionError: If token request fails
        """
        # Check if we have a valid token
        if self._access_token and self._token_expiry:
            if datetime.utcnow() < self._token_expiry:
                return self._access_token

        # In production, this would make an actual OAuth request
        # For now, we provide structure for integration
        logger.info("Requesting new OAuth access token")

        # OAuth token request would be:
        # POST to {api_base_url}/oauth/token
        # with client_credentials grant type

        # Placeholder for actual implementation
        self._access_token = "placeholder_token"
        self._token_expiry = datetime.utcnow()

        return self._access_token

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an authenticated API request.

        Args:
            endpoint: API endpoint path
            method: HTTP method
            params: Query parameters
            data: Request body data

        Returns:
            Response JSON data

        Raises:
            ConnectionError: If request fails
        """
        # Get access token
        token = self._get_access_token()

        url = f"{self.api_base_url}/{self.realm}{endpoint}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.api_key:
            headers["APIKey"] = self.api_key

        logger.debug(f"Making {method} request to {url}")

        # In production, this would use requests/httpx
        # Returning placeholder structure
        return {"items": [], "totalCount": 0}

    def get_suppliers(
        self,
        status: Optional[str] = None,
        modified_since: Optional[datetime] = None,
        page_size: int = 100,
    ) -> Generator[AribaSupplierRecord, None, None]:
        """
        Fetch suppliers from SAP Ariba.

        Args:
            status: Filter by qualification status
            modified_since: Filter by modification date
            page_size: Number of records per page

        Yields:
            AribaSupplierRecord objects
        """
        params: Dict[str, Any] = {"pageSize": page_size}

        if status:
            params["qualificationStatus"] = status
        if modified_since:
            params["modifiedSince"] = modified_since.isoformat()

        offset = 0
        while True:
            params["offset"] = offset
            response = self._make_request(
                AribaEndpoint.SUPPLIERS.value,
                params=params
            )

            items = response.get("items", [])
            if not items:
                break

            for item in items:
                yield self._parse_supplier_record(item)

            offset += page_size
            if offset >= response.get("totalCount", 0):
                break

    def get_supplier_by_id(
        self,
        supplier_id: str
    ) -> Optional[AribaSupplierRecord]:
        """
        Get a specific supplier by ID.

        Args:
            supplier_id: Ariba supplier system ID

        Returns:
            AribaSupplierRecord or None if not found
        """
        response = self._make_request(
            f"{AribaEndpoint.SUPPLIERS.value}/{supplier_id}"
        )

        if response:
            return self._parse_supplier_record(response)
        return None

    def get_supplier_by_anid(
        self,
        anid: str
    ) -> Optional[AribaSupplierRecord]:
        """
        Get a supplier by Ariba Network ID.

        Args:
            anid: Ariba Network ID (AN...)

        Returns:
            AribaSupplierRecord or None if not found
        """
        response = self._make_request(
            AribaEndpoint.SUPPLIERS.value,
            params={"anId": anid}
        )

        items = response.get("items", [])
        if items:
            return self._parse_supplier_record(items[0])
        return None

    def get_purchase_orders(
        self,
        supplier_id: Optional[str] = None,
        status: Optional[str] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        page_size: int = 100,
    ) -> Generator[AribaPORecord, None, None]:
        """
        Fetch purchase orders from SAP Ariba.

        Args:
            supplier_id: Filter by supplier
            status: Filter by PO status
            from_date: Filter by creation date (from)
            to_date: Filter by creation date (to)
            page_size: Number of records per page

        Yields:
            AribaPORecord objects
        """
        params: Dict[str, Any] = {"pageSize": page_size}

        if supplier_id:
            params["supplierId"] = supplier_id
        if status:
            params["status"] = status
        if from_date:
            params["createdDateFrom"] = from_date.isoformat()
        if to_date:
            params["createdDateTo"] = to_date.isoformat()

        offset = 0
        while True:
            params["offset"] = offset
            response = self._make_request(
                AribaEndpoint.PURCHASE_ORDERS.value,
                params=params
            )

            items = response.get("items", [])
            if not items:
                break

            for item in items:
                yield self._parse_po_record(item)

            offset += page_size
            if offset >= response.get("totalCount", 0):
                break

    def get_spend_by_supplier(
        self,
        year: int,
        currency: str = "USD",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get spend aggregation by supplier.

        Args:
            year: Fiscal year
            currency: Currency for spend amounts

        Returns:
            Dictionary mapping supplier ID to spend data
        """
        params = {
            "fiscalYear": year,
            "currency": currency,
            "groupBy": "supplier",
        }

        response = self._make_request(
            AribaEndpoint.SPEND_ANALYSIS.value,
            params=params
        )

        spend_data: Dict[str, Dict[str, Any]] = {}
        for item in response.get("items", []):
            supplier_id = item.get("supplierId")
            if supplier_id:
                spend_data[supplier_id] = {
                    "supplier_name": item.get("supplierName"),
                    "total_spend": Decimal(str(item.get("totalSpend", 0))),
                    "invoice_count": item.get("invoiceCount", 0),
                    "po_count": item.get("poCount", 0),
                    "currency": currency,
                }

        return spend_data

    def get_supplier_risk_data(
        self,
        supplier_id: str
    ) -> Dict[str, Any]:
        """
        Get risk assessment data for a supplier.

        Args:
            supplier_id: Supplier system ID

        Returns:
            Risk assessment data
        """
        response = self._make_request(
            f"{AribaEndpoint.SUPPLIER_RISK.value}/{supplier_id}"
        )

        return {
            "supplier_id": supplier_id,
            "overall_risk_score": response.get("overallRiskScore"),
            "financial_risk": response.get("financialRisk"),
            "operational_risk": response.get("operationalRisk"),
            "compliance_risk": response.get("complianceRisk"),
            "esg_risk": response.get("esgRisk"),
            "last_assessed": response.get("lastAssessedDate"),
        }

    def _parse_supplier_record(
        self,
        data: Dict[str, Any]
    ) -> AribaSupplierRecord:
        """Parse API response to AribaSupplierRecord."""
        return AribaSupplierRecord(
            system_id=data.get("systemId", data.get("id", "")),
            company_name=data.get("companyName", data.get("name", "")),
            an_id=data.get("anId"),
            vendor_id=data.get("vendorId"),
            duns_number=data.get("dunsNumber"),
            tax_id=data.get("taxId"),
            country=data.get("countryCode"),
            street=data.get("address", {}).get("street"),
            city=data.get("address", {}).get("city"),
            state=data.get("address", {}).get("state"),
            postal_code=data.get("address", {}).get("postalCode"),
            contact_name=data.get("primaryContact", {}).get("name"),
            contact_email=data.get("primaryContact", {}).get("email"),
            contact_phone=data.get("primaryContact", {}).get("phone"),
            qualification_status=data.get("qualificationStatus", "pending"),
            risk_score=data.get("riskScore"),
            spend_category=data.get("primaryCategory"),
            industry_code=data.get("industryCode"),
            created_date=datetime.fromisoformat(data["createdDate"]) if data.get("createdDate") else None,
            last_modified=datetime.fromisoformat(data["lastModified"]) if data.get("lastModified") else None,
            custom_fields=data.get("customFields", {}),
        )

    def _parse_po_record(self, data: Dict[str, Any]) -> AribaPORecord:
        """Parse API response to AribaPORecord."""
        return AribaPORecord(
            po_number=data.get("poNumber", ""),
            supplier_id=data.get("supplierId", ""),
            supplier_name=data.get("supplierName", ""),
            total_amount=Decimal(str(data.get("totalAmount", 0))),
            currency=data.get("currency", "USD"),
            status=data.get("status", "open"),
            created_date=datetime.fromisoformat(data["createdDate"]) if data.get("createdDate") else None,
            items=data.get("lineItems", []),
        )

    def sync_suppliers_to_graph(
        self,
        supply_chain_graph: Any,
        status_filter: Optional[str] = "qualified",
    ) -> int:
        """
        Sync suppliers from Ariba to supply chain graph.

        Args:
            supply_chain_graph: SupplyChainGraph instance
            status_filter: Only sync suppliers with this status

        Returns:
            Number of suppliers synced
        """
        count = 0
        for ariba_record in self.get_suppliers(status=status_filter):
            supplier = ariba_record.to_supplier()
            supply_chain_graph.add_supplier(supplier)
            count += 1

        logger.info(f"Synced {count} suppliers from SAP Ariba")
        return count

    def export_for_scope3(
        self,
        year: int
    ) -> List[Dict[str, Any]]:
        """
        Export spend data formatted for Scope 3 calculation.

        Args:
            year: Fiscal year

        Returns:
            List of spend records for Scope3Allocator
        """
        spend_data = self.get_spend_by_supplier(year)
        records = []

        for supplier_id, data in spend_data.items():
            records.append({
                "supplier_id": supplier_id,
                "supplier_name": data["supplier_name"],
                "amount": data["total_spend"],
                "currency": data["currency"],
            })

        return records
