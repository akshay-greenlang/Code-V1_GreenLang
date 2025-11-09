"""
SAP Production Planning (PP) Extractor

Extracts data from SAP S/4HANA Production Planning module including:
    - Production Orders (API_PRODUCTION_ORDER_SRV)
    - Planned Orders (API_PLANNED_ORDER_SRV)
    - Production Order Components (materials consumed)
    - Production Order Operations (work centers, times)

Supports delta extraction by ChangedOn field for carbon accounting of manufactured goods.

Use Cases:
    - Category 1: Manufactured goods emissions tracking
    - Direct emissions from production processes
    - Energy consumption by production order
    - Waste and scrap tracking for sustainability reporting

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
Phase: Team 4 Mission - 48 Missing ERP Connectors
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class ProductionOrderData(BaseModel):
    """SAP Production Order data model.

    Maps to A_ProductionOrder entity from API_PRODUCTION_ORDER_SRV.
    """
    ManufacturingOrder: str
    ManufacturingOrderType: Optional[str] = None
    Material: str
    MaterialName: Optional[str] = None
    ProductionPlant: str
    ProductionVersion: Optional[str] = None
    MfgOrderPlannedTotalQty: float
    ProductionUnit: str
    MfgOrderPlannedStartDate: Optional[str] = None
    MfgOrderPlannedEndDate: Optional[str] = None
    MfgOrderActualStartDate: Optional[str] = None
    MfgOrderActualEndDate: Optional[str] = None
    MfgOrderConfirmedYieldQty: Optional[float] = None
    MfgOrderScrapQty: Optional[float] = None
    OrderIsCreated: Optional[bool] = None
    OrderIsReleased: Optional[bool] = None
    OrderIsConfirmed: Optional[bool] = None
    OrderIsClosed: Optional[bool] = None
    ProductionSupervisor: Optional[str] = None
    ProductionSchedulingProfile: Optional[str] = None
    CreatedByUser: Optional[str] = None
    CreationDate: Optional[str] = None
    LastChangedByUser: Optional[str] = None
    ChangedOn: Optional[str] = None  # For delta extraction


class ProductionOrderItemData(BaseModel):
    """SAP Production Order Item/Component data model.

    Maps to A_ProductionOrderComponent entity.
    """
    ManufacturingOrder: str
    ManufacturingOrderItem: str
    Material: Optional[str] = None
    MaterialName: Optional[str] = None
    Plant: Optional[str] = None
    StorageLocation: Optional[str] = None
    Batch: Optional[str] = None
    GoodsMovementType: Optional[str] = None
    RequiredQuantity: Optional[float] = None
    WithdrawnQuantity: Optional[float] = None
    BaseUnit: Optional[str] = None
    ComponentScrapInPercent: Optional[float] = None
    QuantityIsFixed: Optional[bool] = None


class ProductionOrderOperationData(BaseModel):
    """SAP Production Order Operation data model.

    Maps to A_ProductionOrderOperation entity.
    """
    ManufacturingOrder: str
    ManufacturingOrderOperation: str
    WorkCenter: str
    WorkCenterTypeCode: Optional[str] = None
    OperationText: Optional[str] = None
    Plant: Optional[str] = None
    OperationControlProfile: Optional[str] = None
    OpPlannedTotalQuantity: Optional[float] = None
    OpTotalConfirmedYieldQty: Optional[float] = None
    OpTotalConfirmedScrapQty: Optional[float] = None
    OpPlannedSetupTime: Optional[float] = None
    OpPlannedMachineTime: Optional[float] = None
    OpPlannedLaborTime: Optional[float] = None
    OpActualSetupTime: Optional[float] = None
    OpActualMachineTime: Optional[float] = None
    OpActualLaborTime: Optional[float] = None
    WorkCenterCapacityUnit: Optional[str] = None


class PlannedOrderData(BaseModel):
    """SAP Planned Order data model.

    Maps to A_PlannedOrder entity from API_PLANNED_ORDER_SRV.
    """
    PlannedOrder: str
    Material: str
    ProductionPlant: str
    MRPArea: Optional[str] = None
    PlannedOrderType: Optional[str] = None
    TotalQuantity: float
    ProductionUnit: str
    PlannedOrderPlannedStartDate: Optional[str] = None
    PlannedOrderPlannedEndDate: Optional[str] = None
    ProductionVersion: Optional[str] = None
    StorageLocation: Optional[str] = None
    ProductionSupervisor: Optional[str] = None
    IsConvertedToProductionOrder: Optional[bool] = None
    CreationDate: Optional[str] = None
    LastChangedByUser: Optional[str] = None
    ChangedOn: Optional[str] = None  # For delta extraction


class PPExtractor(BaseExtractor):
    """Production Planning (PP) Extractor.

    Extracts manufacturing and production data from SAP S/4HANA PP module.
    Critical for Category 1 emissions tracking and manufacturing carbon accounting.
    """

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        """Initialize PP extractor.

        Args:
            client: SAP OData client instance
            config: Extraction configuration
        """
        super().__init__(client, config)
        self.service_name = "PP"
        self._current_entity_set = "A_ProductionOrder"  # Default

    def get_entity_set_name(self) -> str:
        """Get current entity set name."""
        return self._current_entity_set

    def get_changed_on_field(self) -> str:
        """Get field name for delta extraction."""
        return "ChangedOn"

    def extract_production_orders(
        self,
        plant: Optional[str] = None,
        material: Optional[str] = None,
        order_type: Optional[str] = None,
        start_date_from: Optional[str] = None,
        start_date_to: Optional[str] = None,
        status_filter: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Production Orders from SAP.

        Args:
            plant: Filter by production plant
            material: Filter by material number
            order_type: Filter by order type (e.g., 'YP01', 'PP01')
            start_date_from: Filter by planned start date from (ISO format)
            start_date_to: Filter by planned start date to (ISO format)
            status_filter: Filter by status ('CREATED', 'RELEASED', 'CONFIRMED', 'CLOSED')

        Yields:
            Production Order records as dictionaries
        """
        self._current_entity_set = "A_ProductionOrder"

        additional_filters = []

        if plant:
            additional_filters.append(f"ProductionPlant eq '{plant}'")
        if material:
            additional_filters.append(f"Material eq '{material}'")
        if order_type:
            additional_filters.append(f"ManufacturingOrderType eq '{order_type}'")
        if start_date_from:
            additional_filters.append(f"MfgOrderPlannedStartDate ge datetime'{start_date_from}'")
        if start_date_to:
            additional_filters.append(f"MfgOrderPlannedStartDate le datetime'{start_date_to}'")

        # Status filters
        if status_filter == 'RELEASED':
            additional_filters.append("OrderIsReleased eq true")
        elif status_filter == 'CONFIRMED':
            additional_filters.append("OrderIsConfirmed eq true")
        elif status_filter == 'CLOSED':
            additional_filters.append("OrderIsClosed eq true")

        logger.info(f"Extracting Production Orders with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="MfgOrderPlannedStartDate desc"
        )

    def extract_production_order_components(
        self,
        manufacturing_order: Optional[str] = None,
        plant: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Production Order Components (materials consumed) from SAP.

        Args:
            manufacturing_order: Filter by specific production order number
            plant: Filter by plant

        Yields:
            Production Order Component records as dictionaries
        """
        self._current_entity_set = "A_ProductionOrderComponent"

        additional_filters = []
        if manufacturing_order:
            additional_filters.append(f"ManufacturingOrder eq '{manufacturing_order}'")
        if plant:
            additional_filters.append(f"Plant eq '{plant}'")

        logger.info(f"Extracting Production Order Components for order: {manufacturing_order or 'All'}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    def extract_production_order_operations(
        self,
        manufacturing_order: Optional[str] = None,
        work_center: Optional[str] = None,
        plant: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Production Order Operations from SAP.

        Operations track work center usage, labor hours, and machine time - critical
        for calculating energy consumption and direct emissions.

        Args:
            manufacturing_order: Filter by specific production order number
            work_center: Filter by work center
            plant: Filter by plant

        Yields:
            Production Order Operation records as dictionaries
        """
        self._current_entity_set = "A_ProductionOrderOperation"

        additional_filters = []
        if manufacturing_order:
            additional_filters.append(f"ManufacturingOrder eq '{manufacturing_order}'")
        if work_center:
            additional_filters.append(f"WorkCenter eq '{work_center}'")
        if plant:
            additional_filters.append(f"Plant eq '{plant}'")

        logger.info(f"Extracting Production Order Operations")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    def extract_planned_orders(
        self,
        plant: Optional[str] = None,
        material: Optional[str] = None,
        mrp_area: Optional[str] = None,
        start_date_from: Optional[str] = None,
        start_date_to: Optional[str] = None,
        converted_only: bool = False
    ) -> Iterator[Dict[str, Any]]:
        """Extract Planned Orders from SAP.

        Planned orders are the result of MRP runs and represent future production needs.

        Args:
            plant: Filter by production plant
            material: Filter by material number
            mrp_area: Filter by MRP area
            start_date_from: Filter by planned start date from (ISO format)
            start_date_to: Filter by planned start date to (ISO format)
            converted_only: If True, only return orders converted to production orders

        Yields:
            Planned Order records as dictionaries
        """
        self._current_entity_set = "A_PlannedOrder"

        additional_filters = []

        if plant:
            additional_filters.append(f"ProductionPlant eq '{plant}'")
        if material:
            additional_filters.append(f"Material eq '{material}'")
        if mrp_area:
            additional_filters.append(f"MRPArea eq '{mrp_area}'")
        if start_date_from:
            additional_filters.append(f"PlannedOrderPlannedStartDate ge datetime'{start_date_from}'")
        if start_date_to:
            additional_filters.append(f"PlannedOrderPlannedStartDate le datetime'{start_date_to}'")
        if converted_only:
            additional_filters.append("IsConvertedToProductionOrder eq true")

        logger.info(f"Extracting Planned Orders with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="PlannedOrderPlannedStartDate desc"
        )

    def extract_production_order_with_details(
        self,
        manufacturing_order: str
    ) -> Dict[str, Any]:
        """Extract complete production order with components and operations.

        This method fetches the production order header along with all related
        components and operations in a single call for comprehensive analysis.

        Args:
            manufacturing_order: Production order number

        Returns:
            Dictionary with 'header', 'components', and 'operations' keys
        """
        logger.info(f"Extracting complete production order: {manufacturing_order}")

        result = {
            'header': None,
            'components': [],
            'operations': []
        }

        # Get header
        self._current_entity_set = "A_ProductionOrder"
        try:
            result['header'] = self.get_by_id(manufacturing_order)
        except Exception as e:
            logger.error(f"Error fetching production order header: {e}")
            return result

        # Get components
        try:
            result['components'] = list(self.extract_production_order_components(
                manufacturing_order=manufacturing_order
            ))
            logger.debug(f"Found {len(result['components'])} components")
        except Exception as e:
            logger.error(f"Error fetching components: {e}")

        # Get operations
        try:
            result['operations'] = list(self.extract_production_order_operations(
                manufacturing_order=manufacturing_order
            ))
            logger.debug(f"Found {len(result['operations'])} operations")
        except Exception as e:
            logger.error(f"Error fetching operations: {e}")

        return result

    def extract_manufacturing_emissions_data(
        self,
        plant: str,
        date_from: str,
        date_to: str
    ) -> Iterator[Dict[str, Any]]:
        """Extract production orders optimized for emissions calculations.

        This method extracts production orders with pre-selected fields most relevant
        for carbon accounting and emissions calculations.

        Args:
            plant: Production plant code
            date_from: Start date for production orders (ISO format)
            date_to: End date for production orders (ISO format)

        Yields:
            Production order records with emissions-relevant fields
        """
        # Override field selection for emissions calculation
        original_fields = self.config.select_fields

        self.config.select_fields = [
            'ManufacturingOrder',
            'Material',
            'MaterialName',
            'ProductionPlant',
            'MfgOrderPlannedTotalQty',
            'MfgOrderConfirmedYieldQty',
            'MfgOrderScrapQty',
            'ProductionUnit',
            'MfgOrderActualStartDate',
            'MfgOrderActualEndDate',
            'ChangedOn'
        ]

        try:
            logger.info(f"Extracting emissions data for plant {plant} from {date_from} to {date_to}")

            yield from self.extract_production_orders(
                plant=plant,
                start_date_from=date_from,
                start_date_to=date_to,
                status_filter='CONFIRMED'  # Only confirmed orders have actual emissions
            )
        finally:
            # Restore original field selection
            self.config.select_fields = original_fields
