# -*- coding: utf-8 -*-
"""
SAP Financial Accounting (FI) Extractor

Extracts data from SAP S/4HANA Financial Accounting module including:
    - Fixed Assets (API_FIXEDASSET_SRV)

Supports delta extraction for tracking changes to fixed asset master data,
which is relevant for Scope 3 Category 8 (Upstream Leased Assets) and
Category 13 (Downstream Leased Assets).

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 19-22) - SAP Connector Implementation
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class FixedAssetData(BaseModel):
    """SAP Fixed Asset Master data model.

    Maps to A_FixedAsset entity from API_FIXEDASSET_SRV.
    """
    CompanyCode: str
    MasterFixedAsset: str
    FixedAsset: str
    FixedAssetDescription: Optional[str] = None
    AssetClass: Optional[str] = None
    AssetClassName: Optional[str] = None
    AssetCapitalizationDate: Optional[str] = None
    AssetRetirementDate: Optional[str] = None
    AssetIsInactive: Optional[bool] = None
    Plant: Optional[str] = None
    Location: Optional[str] = None
    Room: Optional[str] = None
    ResponsibleCostCenter: Optional[str] = None
    AssetOwner: Optional[str] = None
    ManufacturerPartNumber: Optional[str] = None
    ManufacturerSerialNumber: Optional[str] = None
    Manufacturer: Optional[str] = None
    ConstructionYear: Optional[str] = None
    InventoryNumber: Optional[str] = None
    AssetQuantity: Optional[float] = None
    AssetQuantityUnit: Optional[str] = None
    AssetAcquisitionDate: Optional[str] = None
    OriginalAcquisitionValue: Optional[float] = None
    AcquisitionValueCurrency: Optional[str] = None
    CreatedByUser: Optional[str] = None
    CreationDate: Optional[str] = None
    LastChangedByUser: Optional[str] = None
    LastChangeDate: Optional[str] = None
    ChangedOn: Optional[str] = None  # For delta extraction


class FixedAssetDepreciationData(BaseModel):
    """SAP Fixed Asset Depreciation Area data model.

    Maps to A_FixedAssetDepreciationArea entity.
    Contains depreciation information for different areas (e.g., book, tax).
    """
    CompanyCode: str
    MasterFixedAsset: str
    FixedAsset: str
    DepreciationArea: str
    DepreciationAreaDescription: Optional[str] = None
    AssetIsFullyDepreciated: Optional[bool] = None
    AssetDepreciationStartDate: Optional[str] = None
    PlannedUsefulLife: Optional[int] = None
    UsefulLifeInYears: Optional[int] = None
    UsefulLifeInMonths: Optional[int] = None
    AccumulatedDepreciation: Optional[float] = None
    BookValue: Optional[float] = None
    Currency: Optional[str] = None


class FixedAssetLeaseData(BaseModel):
    """SAP Fixed Asset Lease information data model.

    Custom extension for tracking leased assets (Scope 3 Category 8 & 13).
    This may need to be extracted from custom Z-tables or extensions.
    """
    CompanyCode: str
    MasterFixedAsset: str
    FixedAsset: str
    LeaseType: Optional[str] = None  # 'OPERATING' or 'FINANCE'
    LeaseDirection: Optional[str] = None  # 'INBOUND' (upstream) or 'OUTBOUND' (downstream)
    LessorName: Optional[str] = None
    LessorID: Optional[str] = None
    LesseeNamee: Optional[str] = None
    LesseeID: Optional[str] = None
    LeaseStartDate: Optional[str] = None
    LeaseEndDate: Optional[str] = None
    MonthlyLeasePayment: Optional[float] = None
    AnnualLeasePayment: Optional[float] = None
    LeaseCurrency: Optional[str] = None
    LeaseContractNumber: Optional[str] = None


class FIExtractor(BaseExtractor):
    """Financial Accounting (FI) Extractor.

    Extracts fixed asset data from SAP S/4HANA FI module.
    """

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        """Initialize FI extractor.

        Args:
            client: SAP OData client instance
            config: Extraction configuration
        """
        super().__init__(client, config)
        self.service_name = "FI"
        self._current_entity_set = "A_FixedAsset"  # Default

    def get_entity_set_name(self) -> str:
        """Get current entity set name."""
        return self._current_entity_set

    def get_changed_on_field(self) -> str:
        """Get field name for delta extraction."""
        return "ChangedOn"

    def extract_fixed_assets(
        self,
        company_code: Optional[str] = None,
        asset_class: Optional[str] = None,
        plant: Optional[str] = None,
        cost_center: Optional[str] = None,
        include_inactive: bool = False
    ) -> Iterator[Dict[str, Any]]:
        """Extract Fixed Asset master data from SAP.

        Args:
            company_code: Filter by company code
            asset_class: Filter by asset class
            plant: Filter by plant
            cost_center: Filter by responsible cost center
            include_inactive: Whether to include inactive/retired assets

        Yields:
            Fixed Asset master records as dictionaries
        """
        self._current_entity_set = "A_FixedAsset"

        additional_filters = []

        if company_code:
            additional_filters.append(f"CompanyCode eq '{company_code}'")
        if asset_class:
            additional_filters.append(f"AssetClass eq '{asset_class}'")
        if plant:
            additional_filters.append(f"Plant eq '{plant}'")
        if cost_center:
            additional_filters.append(f"ResponsibleCostCenter eq '{cost_center}'")
        if not include_inactive:
            additional_filters.append("AssetIsInactive eq false")

        logger.info(f"Extracting Fixed Assets with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="CompanyCode asc, MasterFixedAsset asc, FixedAsset asc"
        )

    def extract_fixed_asset_depreciation(
        self,
        company_code: Optional[str] = None,
        master_fixed_asset: Optional[str] = None,
        fixed_asset: Optional[str] = None,
        depreciation_area: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Fixed Asset Depreciation Area data from SAP.

        Args:
            company_code: Filter by company code
            master_fixed_asset: Filter by master fixed asset number
            fixed_asset: Filter by sub-asset number
            depreciation_area: Filter by depreciation area (e.g., '01' for book)

        Yields:
            Fixed Asset Depreciation records as dictionaries
        """
        self._current_entity_set = "A_FixedAssetDepreciationArea"

        additional_filters = []

        if company_code:
            additional_filters.append(f"CompanyCode eq '{company_code}'")
        if master_fixed_asset:
            additional_filters.append(f"MasterFixedAsset eq '{master_fixed_asset}'")
        if fixed_asset:
            additional_filters.append(f"FixedAsset eq '{fixed_asset}'")
        if depreciation_area:
            additional_filters.append(f"DepreciationArea eq '{depreciation_area}'")

        logger.info(f"Extracting Fixed Asset Depreciation data")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    def extract_leased_assets(
        self,
        company_code: Optional[str] = None,
        lease_direction: Optional[str] = None,
        active_leases_only: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """Extract leased assets for Scope 3 Category 8 & 13.

        Note: This method assumes custom lease tracking fields or Z-tables exist
        in the SAP system. The actual implementation may need to be customized
        based on the specific SAP configuration.

        Args:
            company_code: Filter by company code
            lease_direction: Filter by lease direction ('INBOUND' or 'OUTBOUND')
            active_leases_only: Whether to include only active leases

        Yields:
            Leased asset records as dictionaries
        """
        self._current_entity_set = "Z_FixedAssetLease"  # Custom entity set

        additional_filters = []

        if company_code:
            additional_filters.append(f"CompanyCode eq '{company_code}'")
        if lease_direction:
            additional_filters.append(f"LeaseDirection eq '{lease_direction}'")

        if active_leases_only:
            # Only active leases (start date <= today and end date >= today)
            from datetime import datetime
            today = DeterministicClock.now().strftime("%Y-%m-%d")
            additional_filters.append(f"LeaseStartDate le datetime'{today}'")
            additional_filters.append(f"LeaseEndDate ge datetime'{today}'")

        logger.info(f"Extracting Leased Assets with filters: {additional_filters}")

        try:
            yield from self.get_all(
                additional_filters=additional_filters if additional_filters else None,
                order_by="CompanyCode asc, LeaseStartDate desc"
            )
        except Exception as e:
            logger.warning(
                f"Leased asset extraction failed. This may require custom SAP configuration: {e}"
            )
            # Return empty iterator if custom entity doesn't exist
            return iter([])

    def extract_assets_by_capitalization_period(
        self,
        company_code: Optional[str] = None,
        capitalization_date_from: Optional[str] = None,
        capitalization_date_to: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract assets capitalized within a specific period.

        Useful for tracking new asset acquisitions for emissions accounting.

        Args:
            company_code: Filter by company code
            capitalization_date_from: Filter by capitalization date from (ISO format)
            capitalization_date_to: Filter by capitalization date to (ISO format)

        Yields:
            Fixed Asset records capitalized in the specified period
        """
        self._current_entity_set = "A_FixedAsset"

        additional_filters = []

        if company_code:
            additional_filters.append(f"CompanyCode eq '{company_code}'")
        if capitalization_date_from:
            additional_filters.append(
                f"AssetCapitalizationDate ge datetime'{capitalization_date_from}'"
            )
        if capitalization_date_to:
            additional_filters.append(
                f"AssetCapitalizationDate le datetime'{capitalization_date_to}'"
            )

        logger.info(
            f"Extracting assets capitalized between {capitalization_date_from} "
            f"and {capitalization_date_to}"
        )

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="AssetCapitalizationDate desc"
        )
