# -*- coding: utf-8 -*-
"""
SAP Quality Management (QM) Extractor

Extracts data from SAP S/4HANA Quality Management module including:
    - Quality Inspections (API_INSPECTIONLOT_SRV)
    - Inspection Results
    - Quality Notifications
    - Inspection Characteristics
    - Non-Conforming Materials

Supports delta extraction by ChangedOn field for quality control and emissions tracking.

Use Cases:
    - Category 1: Quality-related scrap and rework emissions
    - Non-conforming material waste tracking
    - Product quality impact on sustainability
    - Defect rate correlation with environmental metrics
    - Rework energy consumption tracking

Carbon Impact: MEDIUM
- Direct: Scrap materials from failed inspections
- Indirect: Energy consumed in rework processes
- Waste: Non-conforming materials disposal emissions

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
Phase: Team 4 Mission - Priority ERP Connector Modules
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class InspectionLotData(BaseModel):
    """SAP Quality Inspection Lot data model.

    Maps to A_InspectionLot entity from API_INSPECTIONLOT_SRV.
    """
    InspectionLot: str
    Material: Optional[str] = None
    MaterialName: Optional[str] = None
    Batch: Optional[str] = None
    Plant: str
    InspectionLotOrigin: Optional[str] = None  # '01'=Goods Receipt, '04'=Production
    InspectionLotType: Optional[str] = None
    MaterialDocument: Optional[str] = None
    MaterialDocumentYear: Optional[str] = None
    PurchaseOrder: Optional[str] = None
    PurchaseOrderItem: Optional[str] = None
    ManufacturingOrder: Optional[str] = None
    InspectionLotQuantity: float
    InspectionLotQuantityUnit: str
    InspectionLotActualQuantity: Optional[float] = None
    InspectionLotRejectedQuantity: Optional[float] = None
    InspectionLotScrapQuantity: Optional[float] = None
    InspectionLotStartDate: Optional[str] = None
    InspectionLotEndDate: Optional[str] = None
    InspLotUsageDecisionCode: Optional[str] = None  # 'A'=Accepted, 'R'=Rejected
    InspLotUsageDecisionValuation: Optional[str] = None
    InspectionLotIsReleased: Optional[bool] = None
    InspectionLotIsCompleted: Optional[bool] = None
    InspectionLotIsClosed: Optional[bool] = None
    Supplier: Optional[str] = None
    CreatedByUser: Optional[str] = None
    CreationDate: Optional[str] = None
    LastChangedByUser: Optional[str] = None
    ChangedOn: Optional[str] = None  # For delta extraction


class InspectionResultData(BaseModel):
    """SAP Inspection Result data model.

    Maps to A_InspectionResult entity.
    """
    InspectionLot: str
    InspectionCharacteristic: str
    InspectionSpecification: Optional[str] = None
    InspectionOperation: Optional[str] = None
    InspResultMeanValue: Optional[float] = None
    InspResultLowerLimit: Optional[float] = None
    InspResultUpperLimit: Optional[float] = None
    InspectionValuationResult: Optional[str] = None  # 'A'=Accepted, 'R'=Rejected
    InspResultRecordedValue: Optional[float] = None
    InspResultUnit: Optional[str] = None


class QualityNotificationData(BaseModel):
    """SAP Quality Notification data model.

    Maps to A_QualityNotification entity from API_QUALITYNOTIFICATION_SRV.
    """
    QualityNotification: str
    QualityNotificationType: Optional[str] = None
    Material: Optional[str] = None
    Batch: Optional[str] = None
    Plant: Optional[str] = None
    NotificationDescription: Optional[str] = None
    DefectCategory: Optional[str] = None
    DefectCode: Optional[str] = None
    Priority: Optional[str] = None
    QualityNotificationStatus: Optional[str] = None
    DefectQuantity: Optional[float] = None
    DefectQuantityUnit: Optional[str] = None
    Supplier: Optional[str] = None
    PurchaseOrder: Optional[str] = None
    ManufacturingOrder: Optional[str] = None
    CreatedByUser: Optional[str] = None
    CreationDate: Optional[str] = None
    LastChangedByUser: Optional[str] = None
    ChangedOn: Optional[str] = None


class InspectionCharacteristicData(BaseModel):
    """SAP Inspection Characteristic data model.

    Maps to A_InspectionCharacteristic entity.
    """
    InspectionLot: str
    InspectionCharacteristic: str
    InspectionSpecification: Optional[str] = None
    CharacteristicText: Optional[str] = None
    InspectionMethod: Optional[str] = None
    TargetValue: Optional[float] = None
    LowerSpecificationLimit: Optional[float] = None
    UpperSpecificationLimit: Optional[str] = None
    InspectionUnit: Optional[str] = None
    SamplingProcedure: Optional[str] = None
    SampleSize: Optional[int] = None


class QMExtractor(BaseExtractor):
    """Quality Management (QM) Extractor.

    Extracts quality control and inspection data from SAP S/4HANA QM module.
    Critical for tracking quality-related waste, scrap, and rework emissions.
    """

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        """Initialize QM extractor.

        Args:
            client: SAP OData client instance
            config: Extraction configuration
        """
        super().__init__(client, config)
        self.service_name = "QM"
        self._current_entity_set = "A_InspectionLot"  # Default

    def get_entity_set_name(self) -> str:
        """Get current entity set name."""
        return self._current_entity_set

    def get_changed_on_field(self) -> str:
        """Get field name for delta extraction."""
        return "ChangedOn"

    def extract_inspection_lots(
        self,
        plant: Optional[str] = None,
        material: Optional[str] = None,
        origin: Optional[str] = None,
        usage_decision: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        status_filter: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Inspection Lots from SAP.

        Args:
            plant: Filter by plant code
            material: Filter by material number
            origin: Filter by origin ('01'=GR, '04'=Production, '05'=Stock, '07'=Delivery)
            usage_decision: Filter by decision ('A'=Accepted, 'R'=Rejected, 'Q'=Quality Hold)
            date_from: Filter by inspection start date from (ISO format)
            date_to: Filter by inspection start date to (ISO format)
            status_filter: Filter by status ('RELEASED', 'COMPLETED', 'CLOSED')

        Yields:
            Inspection Lot records as dictionaries
        """
        self._current_entity_set = "A_InspectionLot"

        additional_filters = []

        if plant:
            additional_filters.append(f"Plant eq '{plant}'")
        if material:
            additional_filters.append(f"Material eq '{material}'")
        if origin:
            additional_filters.append(f"InspectionLotOrigin eq '{origin}'")
        if usage_decision:
            additional_filters.append(f"InspLotUsageDecisionCode eq '{usage_decision}'")
        if date_from:
            additional_filters.append(f"InspectionLotStartDate ge datetime'{date_from}'")
        if date_to:
            additional_filters.append(f"InspectionLotStartDate le datetime'{date_to}'")

        # Status filters
        if status_filter == 'RELEASED':
            additional_filters.append("InspectionLotIsReleased eq true")
        elif status_filter == 'COMPLETED':
            additional_filters.append("InspectionLotIsCompleted eq true")
        elif status_filter == 'CLOSED':
            additional_filters.append("InspectionLotIsClosed eq true")

        logger.info(f"Extracting Inspection Lots with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="InspectionLotStartDate desc"
        )

    def extract_inspection_results(
        self,
        inspection_lot: Optional[str] = None,
        valuation_result: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Inspection Results from SAP.

        Args:
            inspection_lot: Filter by specific inspection lot number
            valuation_result: Filter by valuation ('A'=Accepted, 'R'=Rejected)

        Yields:
            Inspection Result records as dictionaries
        """
        self._current_entity_set = "A_InspectionResult"

        additional_filters = []
        if inspection_lot:
            additional_filters.append(f"InspectionLot eq '{inspection_lot}'")
        if valuation_result:
            additional_filters.append(f"InspectionValuationResult eq '{valuation_result}'")

        logger.info(f"Extracting Inspection Results for lot: {inspection_lot or 'All'}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    def extract_quality_notifications(
        self,
        plant: Optional[str] = None,
        notification_type: Optional[str] = None,
        material: Optional[str] = None,
        supplier: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Quality Notifications from SAP.

        Quality notifications track defects, complaints, and quality issues
        that can result in scrap, rework, or returns - all carbon-intensive activities.

        Args:
            plant: Filter by plant code
            notification_type: Filter by notification type ('Q1'=Internal, 'Q2'=Supplier, 'Q3'=Customer)
            material: Filter by material number
            supplier: Filter by supplier/vendor
            date_from: Filter by creation date from (ISO format)
            date_to: Filter by creation date to (ISO format)

        Yields:
            Quality Notification records as dictionaries
        """
        self._current_entity_set = "A_QualityNotification"

        additional_filters = []

        if plant:
            additional_filters.append(f"Plant eq '{plant}'")
        if notification_type:
            additional_filters.append(f"QualityNotificationType eq '{notification_type}'")
        if material:
            additional_filters.append(f"Material eq '{material}'")
        if supplier:
            additional_filters.append(f"Supplier eq '{supplier}'")
        if date_from:
            additional_filters.append(f"CreationDate ge datetime'{date_from}'")
        if date_to:
            additional_filters.append(f"CreationDate le datetime'{date_to}'")

        logger.info(f"Extracting Quality Notifications with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="CreationDate desc"
        )

    def extract_inspection_characteristics(
        self,
        inspection_lot: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Inspection Characteristics from SAP.

        Args:
            inspection_lot: Filter by specific inspection lot number

        Yields:
            Inspection Characteristic records as dictionaries
        """
        self._current_entity_set = "A_InspectionCharacteristic"

        additional_filters = []
        if inspection_lot:
            additional_filters.append(f"InspectionLot eq '{inspection_lot}'")

        logger.info(f"Extracting Inspection Characteristics")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    def extract_inspection_lot_with_details(
        self,
        inspection_lot: str
    ) -> Dict[str, Any]:
        """Extract complete inspection lot with results and characteristics.

        This method fetches the inspection lot header along with all related
        results and characteristics for comprehensive quality analysis.

        Args:
            inspection_lot: Inspection lot number

        Returns:
            Dictionary with 'header', 'results', and 'characteristics' keys
        """
        logger.info(f"Extracting complete inspection lot: {inspection_lot}")

        result = {
            'header': None,
            'results': [],
            'characteristics': []
        }

        # Get header
        self._current_entity_set = "A_InspectionLot"
        try:
            result['header'] = self.get_by_id(inspection_lot)
        except Exception as e:
            logger.error(f"Error fetching inspection lot header: {e}")
            return result

        # Get results
        try:
            result['results'] = list(self.extract_inspection_results(
                inspection_lot=inspection_lot
            ))
            logger.debug(f"Found {len(result['results'])} inspection results")
        except Exception as e:
            logger.error(f"Error fetching results: {e}")

        # Get characteristics
        try:
            result['characteristics'] = list(self.extract_inspection_characteristics(
                inspection_lot=inspection_lot
            ))
            logger.debug(f"Found {len(result['characteristics'])} characteristics")
        except Exception as e:
            logger.error(f"Error fetching characteristics: {e}")

        return result

    def extract_rejected_materials_data(
        self,
        plant: str,
        date_from: str,
        date_to: str
    ) -> Iterator[Dict[str, Any]]:
        """Extract inspection lots for rejected materials (scrap tracking).

        This method extracts inspection lots where materials were rejected,
        which is critical for calculating waste-related emissions.

        Args:
            plant: Plant code
            date_from: Start date for inspections (ISO format)
            date_to: End date for inspections (ISO format)

        Yields:
            Inspection lot records for rejected materials
        """
        # Override field selection for scrap calculation
        original_fields = self.config.select_fields

        self.config.select_fields = [
            'InspectionLot',
            'Material',
            'MaterialName',
            'Plant',
            'InspectionLotOrigin',
            'InspectionLotQuantity',
            'InspectionLotRejectedQuantity',
            'InspectionLotScrapQuantity',
            'InspectionLotQuantityUnit',
            'InspLotUsageDecisionCode',
            'InspLotUsageDecisionValuation',
            'Supplier',
            'ManufacturingOrder',
            'InspectionLotStartDate',
            'InspectionLotEndDate',
            'ChangedOn'
        ]

        try:
            logger.info(f"Extracting rejected materials for plant {plant} from {date_from} to {date_to}")

            yield from self.extract_inspection_lots(
                plant=plant,
                date_from=date_from,
                date_to=date_to,
                usage_decision='R',  # Only rejected materials
                status_filter='COMPLETED'
            )
        finally:
            # Restore original field selection
            self.config.select_fields = original_fields

    def extract_quality_scrap_emissions_data(
        self,
        plant: str,
        date_from: str,
        date_to: str
    ) -> Dict[str, Any]:
        """Extract aggregated quality scrap data for emissions calculations.

        This method provides a high-level summary of quality-related scrap
        for carbon accounting purposes.

        Args:
            plant: Plant code
            date_from: Start date (ISO format)
            date_to: End date (ISO format)

        Returns:
            Dictionary with scrap summary statistics
        """
        logger.info(f"Calculating quality scrap emissions data for {plant}")

        summary = {
            'plant': plant,
            'period_from': date_from,
            'period_to': date_to,
            'total_inspections': 0,
            'total_rejected_lots': 0,
            'total_scrap_quantity': 0.0,
            'scrap_by_material': {},
            'scrap_by_origin': {},
            'rejection_rate_percent': 0.0
        }

        try:
            # Get all inspections for the period
            all_lots = list(self.extract_inspection_lots(
                plant=plant,
                date_from=date_from,
                date_to=date_to,
                status_filter='COMPLETED'
            ))

            summary['total_inspections'] = len(all_lots)

            # Analyze rejections
            for lot in all_lots:
                usage_decision = lot.get('InspLotUsageDecisionCode')
                scrap_qty = lot.get('InspectionLotScrapQuantity', 0.0) or 0.0
                rejected_qty = lot.get('InspectionLotRejectedQuantity', 0.0) or 0.0

                if usage_decision == 'R' or scrap_qty > 0 or rejected_qty > 0:
                    summary['total_rejected_lots'] += 1

                    # Sum scrap quantities
                    total_qty = scrap_qty + rejected_qty
                    summary['total_scrap_quantity'] += total_qty

                    # Group by material
                    material = lot.get('Material', 'UNKNOWN')
                    summary['scrap_by_material'][material] = \
                        summary['scrap_by_material'].get(material, 0.0) + total_qty

                    # Group by origin
                    origin = lot.get('InspectionLotOrigin', 'UNKNOWN')
                    summary['scrap_by_origin'][origin] = \
                        summary['scrap_by_origin'].get(origin, 0.0) + total_qty

            # Calculate rejection rate
            if summary['total_inspections'] > 0:
                summary['rejection_rate_percent'] = \
                    (summary['total_rejected_lots'] / summary['total_inspections']) * 100

            logger.info(
                f"Quality scrap summary: {summary['total_rejected_lots']} rejected lots, "
                f"{summary['total_scrap_quantity']:.2f} total scrap quantity, "
                f"{summary['rejection_rate_percent']:.2f}% rejection rate"
            )

        except Exception as e:
            logger.error(f"Error calculating scrap emissions data: {e}", exc_info=True)

        return summary
