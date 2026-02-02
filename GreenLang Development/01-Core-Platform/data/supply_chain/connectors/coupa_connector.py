"""
Coupa Connector.

This module provides integration with Coupa for supplier master data
and procurement transaction import.

Supported APIs:
- Supplier API (Supplier records)
- Invoice API (Invoice data)
- Purchase Order API
- Spend Analysis API

Authentication:
- OAuth 2.0 with API Key
- API Token authentication

Example:
    >>> from greenlang.supply_chain.connectors import CoupaConnector
    >>> connector = CoupaConnector(
    ...     instance_url="https://company.coupacloud.com",
    ...     api_key="your-api-key"
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

from greenlang.supply_chain.models.entity import (
    Supplier,
    Address,
    ExternalIdentifiers,
    ContactInfo,
    SupplierTier,
    SupplierStatus,
)

logger = logging.getLogger(__name__)


class CoupaEndpoint(Enum):
    """Coupa API endpoints."""
    SUPPLIERS = "/api/suppliers"
    INVOICES = "/api/invoices"
    INVOICE_LINES = "/api/invoice_lines"
    PURCHASE_ORDERS = "/api/purchase_orders"
    PO_LINES = "/api/order_lines"
    CONTRACTS = "/api/contracts"
    SPEND_ANALYSIS = "/api/analytics/spend"


@dataclass
class CoupaSupplierRecord:
    """
    Supplier record from Coupa.

    Attributes:
        id: Coupa internal ID
        name: Supplier name
        display_name: Display name
        number: Supplier number
        status: Supplier status
        duns: D-U-N-S number
        tax_id: Tax identification number
        payment_term_id: Payment term ID
        primary_address: Primary address
        primary_contact: Primary contact
        custom_fields: Custom field values
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    id: int
    name: str
    display_name: Optional[str] = None
    number: Optional[str] = None
    status: str = "active"
    duns: Optional[str] = None
    tax_id: Optional[str] = None
    payment_term_id: Optional[int] = None
    po_method: Optional[str] = None
    website: Optional[str] = None
    # Address
    street1: Optional[str] = None
    street2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country_code: Optional[str] = None
    # Contact
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    # Sustainability
    sustainability_score: Optional[float] = None
    sustainability_tier: Optional[str] = None
    certifications: List[str] = field(default_factory=list)
    # Metadata
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_supplier(self) -> Supplier:
        """
        Convert Coupa record to Supplier entity.

        Returns:
            Supplier entity
        """
        # Map status
        status_mapping = {
            "active": SupplierStatus.ACTIVE,
            "inactive": SupplierStatus.INACTIVE,
            "pending": SupplierStatus.PENDING_QUALIFICATION,
            "blocked": SupplierStatus.BLOCKED,
            "approved": SupplierStatus.QUALIFIED,
        }
        supplier_status = status_mapping.get(
            self.status.lower(),
            SupplierStatus.ACTIVE
        )

        # Build address
        address = None
        if self.street1 or self.city:
            address = Address(
                street_line_1=self.street1 or "",
                street_line_2=self.street2,
                city=self.city or "",
                state_province=self.state,
                postal_code=self.postal_code,
                country_code=self.country_code or "",
            )

        # Build external identifiers
        external_ids = ExternalIdentifiers(
            duns=self.duns,
            vat_number=self.tax_id,
            coupa_supplier_id=str(self.id),
            custom_ids={"coupa_number": self.number} if self.number else {},
        )

        # Build contact info
        contact = None
        if self.contact_email or self.contact_name:
            contact = ContactInfo(
                primary_contact_name=self.contact_name,
                primary_email=self.contact_email,
                primary_phone=self.contact_phone,
            )

        return Supplier(
            id=f"COUPA-{self.id}",
            name=self.name,
            tier=SupplierTier.TIER_1,
            country_code=self.country_code,
            status=supplier_status,
            external_ids=external_ids,
            address=address,
            contact=contact,
            certifications=self.certifications,
            sustainability_rating=self.sustainability_tier,
            metadata={
                "source": "coupa",
                "coupa_id": self.id,
                "coupa_number": self.number,
                "display_name": self.display_name,
                "sustainability_score": self.sustainability_score,
                "website": self.website,
                **self.custom_fields,
            },
        )


@dataclass
class CoupaInvoiceRecord:
    """
    Invoice record from Coupa.

    Attributes:
        id: Coupa invoice ID
        invoice_number: Invoice number
        supplier_id: Supplier ID
        supplier_name: Supplier name
        total_amount: Total invoice amount
        currency: Currency code
        invoice_date: Invoice date
        status: Invoice status
        lines: Invoice line items
    """
    id: int
    invoice_number: str
    supplier_id: int
    supplier_name: str
    total_amount: Decimal
    currency: str = "USD"
    invoice_date: Optional[date] = None
    due_date: Optional[date] = None
    status: str = "pending"
    po_number: Optional[str] = None
    lines: List[Dict[str, Any]] = field(default_factory=list)
    created_at: Optional[datetime] = None


class CoupaConnector:
    """
    Connector for Coupa integration.

    Provides methods to fetch supplier data, invoices, and spend
    information from Coupa APIs.

    Example:
        >>> connector = CoupaConnector(
        ...     instance_url="https://company.coupacloud.com",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> # Get all suppliers
        >>> suppliers = connector.get_suppliers()
        >>>
        >>> # Get invoices for a date range
        >>> invoices = connector.get_invoices(
        ...     from_date=date(2024, 1, 1),
        ...     to_date=date(2024, 12, 31)
        ... )
    """

    def __init__(
        self,
        instance_url: str,
        api_key: str,
        oauth_token: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize the Coupa connector.

        Args:
            instance_url: Coupa instance URL
            api_key: API key for authentication
            oauth_token: Optional OAuth token
            timeout: Request timeout in seconds
        """
        self.instance_url = instance_url.rstrip("/")
        self.api_key = api_key
        self.oauth_token = oauth_token
        self.timeout = timeout

        logger.info(f"CoupaConnector initialized for: {instance_url}")

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
        url = f"{self.instance_url}{endpoint}"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if self.oauth_token:
            headers["Authorization"] = f"Bearer {self.oauth_token}"
        else:
            headers["X-COUPA-API-KEY"] = self.api_key

        logger.debug(f"Making {method} request to {url}")

        # In production, this would use requests/httpx
        # Returning placeholder structure
        return {"suppliers": [], "invoices": [], "total": 0}

    def get_suppliers(
        self,
        status: Optional[str] = None,
        updated_since: Optional[datetime] = None,
        limit: int = 50,
    ) -> Generator[CoupaSupplierRecord, None, None]:
        """
        Fetch suppliers from Coupa.

        Args:
            status: Filter by supplier status
            updated_since: Filter by update date
            limit: Records per page

        Yields:
            CoupaSupplierRecord objects
        """
        params: Dict[str, Any] = {"limit": limit}

        if status:
            params["status"] = status
        if updated_since:
            params["updated-at[gt]"] = updated_since.isoformat()

        offset = 0
        while True:
            params["offset"] = offset
            response = self._make_request(
                CoupaEndpoint.SUPPLIERS.value,
                params=params
            )

            items = response.get("suppliers", [])
            if not items:
                break

            for item in items:
                yield self._parse_supplier_record(item)

            offset += limit
            if len(items) < limit:
                break

    def get_supplier_by_id(
        self,
        supplier_id: int
    ) -> Optional[CoupaSupplierRecord]:
        """
        Get a specific supplier by ID.

        Args:
            supplier_id: Coupa supplier ID

        Returns:
            CoupaSupplierRecord or None if not found
        """
        response = self._make_request(
            f"{CoupaEndpoint.SUPPLIERS.value}/{supplier_id}"
        )

        if response:
            return self._parse_supplier_record(response)
        return None

    def get_invoices(
        self,
        supplier_id: Optional[int] = None,
        status: Optional[str] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        limit: int = 50,
    ) -> Generator[CoupaInvoiceRecord, None, None]:
        """
        Fetch invoices from Coupa.

        Args:
            supplier_id: Filter by supplier
            status: Filter by invoice status
            from_date: Filter by invoice date (from)
            to_date: Filter by invoice date (to)
            limit: Records per page

        Yields:
            CoupaInvoiceRecord objects
        """
        params: Dict[str, Any] = {"limit": limit}

        if supplier_id:
            params["supplier_id"] = supplier_id
        if status:
            params["status"] = status
        if from_date:
            params["invoice-date[gte]"] = from_date.isoformat()
        if to_date:
            params["invoice-date[lte]"] = to_date.isoformat()

        offset = 0
        while True:
            params["offset"] = offset
            response = self._make_request(
                CoupaEndpoint.INVOICES.value,
                params=params
            )

            items = response.get("invoices", [])
            if not items:
                break

            for item in items:
                yield self._parse_invoice_record(item)

            offset += limit
            if len(items) < limit:
                break

    def get_spend_by_supplier(
        self,
        from_date: date,
        to_date: date,
        currency: str = "USD",
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get spend aggregation by supplier.

        Args:
            from_date: Start date
            to_date: End date
            currency: Currency for spend amounts

        Returns:
            Dictionary mapping supplier ID to spend data
        """
        spend_data: Dict[int, Dict[str, Any]] = {}

        for invoice in self.get_invoices(
            from_date=from_date,
            to_date=to_date,
        ):
            if invoice.supplier_id not in spend_data:
                spend_data[invoice.supplier_id] = {
                    "supplier_name": invoice.supplier_name,
                    "total_spend": Decimal("0"),
                    "invoice_count": 0,
                    "currency": currency,
                }

            spend_data[invoice.supplier_id]["total_spend"] += invoice.total_amount
            spend_data[invoice.supplier_id]["invoice_count"] += 1

        return spend_data

    def get_spend_by_category(
        self,
        from_date: date,
        to_date: date,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get spend aggregation by procurement category.

        Args:
            from_date: Start date
            to_date: End date

        Returns:
            Dictionary mapping category to spend data
        """
        # This would typically use the analytics API
        params = {
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "group_by": "commodity",
        }

        response = self._make_request(
            CoupaEndpoint.SPEND_ANALYSIS.value,
            params=params
        )

        category_spend: Dict[str, Dict[str, Any]] = {}
        for item in response.get("data", []):
            category = item.get("commodity", "Uncategorized")
            category_spend[category] = {
                "total_spend": Decimal(str(item.get("total", 0))),
                "supplier_count": item.get("supplier_count", 0),
                "invoice_count": item.get("invoice_count", 0),
            }

        return category_spend

    def get_supplier_sustainability_data(
        self,
        supplier_id: int
    ) -> Dict[str, Any]:
        """
        Get sustainability data for a supplier.

        Args:
            supplier_id: Coupa supplier ID

        Returns:
            Sustainability data
        """
        response = self._make_request(
            f"{CoupaEndpoint.SUPPLIERS.value}/{supplier_id}/sustainability"
        )

        return {
            "supplier_id": supplier_id,
            "sustainability_score": response.get("score"),
            "sustainability_tier": response.get("tier"),
            "carbon_disclosure": response.get("carbon_disclosure"),
            "scope1_emissions": response.get("scope1_emissions"),
            "scope2_emissions": response.get("scope2_emissions"),
            "scope3_emissions": response.get("scope3_emissions"),
            "certifications": response.get("certifications", []),
            "last_updated": response.get("last_updated"),
        }

    def _parse_supplier_record(
        self,
        data: Dict[str, Any]
    ) -> CoupaSupplierRecord:
        """Parse API response to CoupaSupplierRecord."""
        primary_address = data.get("primary-address", {})
        primary_contact = data.get("primary-contact", {})

        return CoupaSupplierRecord(
            id=data.get("id", 0),
            name=data.get("name", ""),
            display_name=data.get("display-name"),
            number=data.get("number"),
            status=data.get("status", "active"),
            duns=data.get("duns"),
            tax_id=data.get("tax-id"),
            payment_term_id=data.get("payment-term", {}).get("id"),
            po_method=data.get("po-method"),
            website=data.get("website"),
            street1=primary_address.get("street1"),
            street2=primary_address.get("street2"),
            city=primary_address.get("city"),
            state=primary_address.get("state"),
            postal_code=primary_address.get("postal-code"),
            country_code=primary_address.get("country", {}).get("code"),
            contact_name=primary_contact.get("name"),
            contact_email=primary_contact.get("email"),
            contact_phone=primary_contact.get("phone"),
            sustainability_score=data.get("sustainability-score"),
            sustainability_tier=data.get("sustainability-tier"),
            certifications=data.get("certifications", []),
            custom_fields=data.get("custom-fields", {}),
            created_at=datetime.fromisoformat(data["created-at"]) if data.get("created-at") else None,
            updated_at=datetime.fromisoformat(data["updated-at"]) if data.get("updated-at") else None,
        )

    def _parse_invoice_record(
        self,
        data: Dict[str, Any]
    ) -> CoupaInvoiceRecord:
        """Parse API response to CoupaInvoiceRecord."""
        supplier = data.get("supplier", {})

        return CoupaInvoiceRecord(
            id=data.get("id", 0),
            invoice_number=data.get("invoice-number", ""),
            supplier_id=supplier.get("id", 0),
            supplier_name=supplier.get("name", ""),
            total_amount=Decimal(str(data.get("total", 0))),
            currency=data.get("currency", {}).get("code", "USD"),
            invoice_date=date.fromisoformat(data["invoice-date"]) if data.get("invoice-date") else None,
            due_date=date.fromisoformat(data["due-date"]) if data.get("due-date") else None,
            status=data.get("status", "pending"),
            po_number=data.get("po-number"),
            lines=data.get("invoice-lines", []),
            created_at=datetime.fromisoformat(data["created-at"]) if data.get("created-at") else None,
        )

    def sync_suppliers_to_graph(
        self,
        supply_chain_graph: Any,
        status_filter: Optional[str] = "active",
    ) -> int:
        """
        Sync suppliers from Coupa to supply chain graph.

        Args:
            supply_chain_graph: SupplyChainGraph instance
            status_filter: Only sync suppliers with this status

        Returns:
            Number of suppliers synced
        """
        count = 0
        for coupa_record in self.get_suppliers(status=status_filter):
            supplier = coupa_record.to_supplier()
            supply_chain_graph.add_supplier(supplier)
            count += 1

        logger.info(f"Synced {count} suppliers from Coupa")
        return count

    def export_for_scope3(
        self,
        from_date: date,
        to_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Export spend data formatted for Scope 3 calculation.

        Args:
            from_date: Start date
            to_date: End date

        Returns:
            List of spend records for Scope3Allocator
        """
        spend_data = self.get_spend_by_supplier(from_date, to_date)
        records = []

        for supplier_id, data in spend_data.items():
            records.append({
                "supplier_id": f"COUPA-{supplier_id}",
                "supplier_name": data["supplier_name"],
                "amount": data["total_spend"],
                "currency": data["currency"],
            })

        return records
