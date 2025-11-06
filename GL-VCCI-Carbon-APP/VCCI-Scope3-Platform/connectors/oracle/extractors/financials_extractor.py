"""
Oracle Financials Cloud Extractor

Extracts data from Oracle Fusion Financials Cloud including:
    - Fixed Assets (/fixedAssets)

Supports delta extraction by LastUpdateDate field and provides field selection
for performance optimization. Used for Category 2 (Capital Goods) emissions.

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 22-24) - Oracle Connector Implementation
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class FixedAssetData(BaseModel):
    """Oracle Fixed Asset data model.

    Maps to /fixedAssets REST endpoint response.
    """
    AssetId: int
    AssetNumber: str
    AssetDescription: Optional[str] = None
    AssetCategory: Optional[str] = None
    AssetType: Optional[str] = None
    SerialNumber: Optional[str] = None
    TagNumber: Optional[str] = None
    AssetKey: Optional[str] = None
    BookTypeCode: Optional[str] = None
    CostCenter: Optional[str] = None
    LocationId: Optional[int] = None
    LocationName: Optional[str] = None
    LocationAddress: Optional[str] = None
    LocationCity: Optional[str] = None
    LocationState: Optional[str] = None
    LocationCountry: Optional[str] = None
    LocationPostalCode: Optional[str] = None
    OriginalCost: Optional[float] = None
    CurrentCost: Optional[float] = None
    CostCurrency: Optional[str] = None
    DatePlacedInService: Optional[str] = None
    DateAcquired: Optional[str] = None
    DepreciationMethod: Optional[str] = None
    UsefulLife: Optional[float] = None
    SalvageValue: Optional[float] = None
    AccumulatedDepreciation: Optional[float] = None
    NetBookValue: Optional[float] = None
    ManufacturerId: Optional[int] = None
    ManufacturerName: Optional[str] = None
    SupplierName: Optional[str] = None
    SupplierId: Optional[int] = None
    SupplierSiteId: Optional[int] = None
    PurchaseOrderNumber: Optional[str] = None
    InvoiceNumber: Optional[str] = None
    WarrantyExpirationDate: Optional[str] = None
    AssetStatus: Optional[str] = None
    Capitalized: Optional[bool] = None
    ProjectId: Optional[int] = None
    ProjectNumber: Optional[str] = None
    CreatedBy: Optional[str] = None
    CreationDate: Optional[str] = None
    LastUpdatedBy: Optional[str] = None
    LastUpdateDate: Optional[str] = None  # For delta extraction


class AssetBookData(BaseModel):
    """Oracle Asset Book data model.

    Maps to /assetBooks REST endpoint response (asset depreciation books).
    """
    AssetId: int
    BookTypeCode: str
    AssetNumber: str
    Cost: Optional[float] = None
    DatePlacedInService: Optional[str] = None
    DepreciationMethod: Optional[str] = None
    LifeInMonths: Optional[int] = None
    RecoverableAmount: Optional[float] = None
    SalvageValue: Optional[float] = None
    CurrentFiscalYearDepreciation: Optional[float] = None
    AccumulatedDepreciation: Optional[float] = None
    NetBookValue: Optional[float] = None
    DepreciationReserve: Optional[float] = None
    Currency: Optional[str] = None
    LastUpdateDate: Optional[str] = None


class FinancialsExtractor(BaseExtractor):
    """Financials Cloud Extractor.

    Extracts fixed asset data from Oracle Fusion Financials Cloud.
    """

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        """Initialize Financials extractor.

        Args:
            client: Oracle REST client instance
            config: Extraction configuration
        """
        super().__init__(client, config)
        self.base_url = "/fscmRestApi/resources/11.13.18.05"  # Oracle Financials REST API base
        self._current_resource = "/fixedAssets"  # Default

    def get_resource_path(self) -> str:
        """Get current REST resource path."""
        return f"{self.base_url}{self._current_resource}"

    def get_changed_on_field(self) -> str:
        """Get field name for delta extraction."""
        return "LastUpdateDate"

    def extract_fixed_assets(
        self,
        asset_category: Optional[str] = None,
        asset_type: Optional[str] = None,
        location_id: Optional[int] = None,
        cost_center: Optional[str] = None,
        status: Optional[str] = None,
        date_placed_from: Optional[str] = None,
        date_placed_to: Optional[str] = None,
        capitalized_only: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """Extract Fixed Assets from Oracle.

        Args:
            asset_category: Filter by asset category (e.g., 'MACHINERY', 'EQUIPMENT')
            asset_type: Filter by asset type
            location_id: Filter by location ID
            cost_center: Filter by cost center
            status: Filter by asset status (e.g., 'ACTIVE', 'RETIRED')
            date_placed_from: Filter by date placed in service from (ISO format)
            date_placed_to: Filter by date placed in service to (ISO format)
            capitalized_only: Only include capitalized assets (default: True)

        Yields:
            Fixed Asset records as dictionaries
        """
        self._current_resource = "/fixedAssets"

        additional_filters = []

        if asset_category:
            additional_filters.append(f"AssetCategory='{asset_category}'")
        if asset_type:
            additional_filters.append(f"AssetType='{asset_type}'")
        if location_id:
            additional_filters.append(f"LocationId={location_id}")
        if cost_center:
            additional_filters.append(f"CostCenter='{cost_center}'")
        if status:
            additional_filters.append(f"AssetStatus='{status}'")
        if date_placed_from:
            additional_filters.append(f"DatePlacedInService>='{date_placed_from}'")
        if date_placed_to:
            additional_filters.append(f"DatePlacedInService<='{date_placed_to}'")
        if capitalized_only:
            additional_filters.append("Capitalized=true")

        logger.info(f"Extracting Fixed Assets with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="DatePlacedInService:desc"
        )

    def extract_asset_books(
        self,
        asset_id: Optional[int] = None,
        book_type_code: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Asset Books (depreciation books) from Oracle.

        Args:
            asset_id: Filter by specific asset ID
            book_type_code: Filter by book type code (e.g., 'CORPORATE', 'TAX')

        Yields:
            Asset Book records as dictionaries
        """
        self._current_resource = "/assetBooks"

        additional_filters = []
        if asset_id:
            additional_filters.append(f"AssetId={asset_id}")
        if book_type_code:
            additional_filters.append(f"BookTypeCode='{book_type_code}'")

        logger.info(f"Extracting Asset Books for Asset: {asset_id or 'All'}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="AssetId:asc"
        )
