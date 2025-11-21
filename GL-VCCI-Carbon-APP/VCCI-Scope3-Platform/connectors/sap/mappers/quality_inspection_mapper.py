# -*- coding: utf-8 -*-
"""
Quality Inspection Mapper

Maps SAP S/4HANA Quality Management (QM) Inspection Lot data to VCCI waste_v1.0.json schema.

This mapper transforms SAP QM Inspection Lot data into the standardized
waste/scrap schema used by the VCCI Scope 3 Carbon Platform for:
    - Category 1: Quality-related scrap emissions
    - Waste disposal emissions from rejected materials
    - Rework emissions tracking
    - Quality cost environmental impact

Field Mappings:
    SAP Field → VCCI Schema Field
    - InspectionLot → waste_transaction_id
    - Material → material_code
    - MaterialName → material_name
    - Plant → facility_id
    - InspectionLotRejectedQuantity + InspectionLotScrapQuantity → waste_quantity
    - InspectionLotQuantityUnit → unit
    - InspectionLotEndDate → transaction_date
    - InspLotUsageDecisionCode → waste_category ('R'=Rejected, 'Q'=Quality Hold)
    - InspectionLotOrigin → source_type ('01'=GR, '04'=Production)

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
Phase: Team 4 Mission - Priority ERP Connector Modules
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class WasteRecord(BaseModel):
    """VCCI Waste data model matching waste_v1.0.json schema."""

    waste_transaction_id: str
    transaction_date: str
    facility_id: str
    material_code: str
    material_name: str
    waste_quantity: float
    unit: str
    waste_category: str

    # Optional fields
    tenant_id: Optional[str] = None
    reporting_year: Optional[int] = None
    facility_name: Optional[str] = None
    facility_country: Optional[str] = None
    facility_region: Optional[str] = None
    waste_type: Optional[str] = None  # 'Scrap', 'Rework', 'Rejected'
    disposal_method: Optional[str] = None  # 'Landfill', 'Recycle', 'Incinerate'
    source_document_type: Optional[str] = None
    source_document_id: Optional[str] = None
    supplier_id: Optional[str] = None
    manufacturing_order: Optional[str] = None
    purchase_order: Optional[str] = None
    inspection_result: Optional[str] = None
    defect_code: Optional[str] = None
    emissions_kg_co2e: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class QualityInspectionMapper:
    """Maps SAP Quality Inspection Lot data to VCCI waste schema."""

    # SAP to VCCI unit mapping
    UNIT_MAPPING = {
        "KG": "kg",
        "G": "kg",  # Convert grams to kg
        "TO": "tonnes",
        "T": "tonnes",
        "LB": "lbs",
        "L": "liters",
        "GAL": "gallons",
        "M3": "m3",
        "EA": "items",
        "PC": "items",
        "ST": "items",
        "UN": "units",
    }

    # SAP Usage Decision to Waste Category
    USAGE_DECISION_MAPPING = {
        "R": "Rejected",  # Rejected
        "Q": "Quality Hold",  # Quality inspection hold
        "S": "Scrap",  # Scrap
        "X": "Sample Destruction",  # Sample destroyed
    }

    # Inspection Origin to Source Type
    ORIGIN_MAPPING = {
        "01": "Goods Receipt",
        "04": "Production",
        "05": "Stock",
        "07": "Delivery",
        "08": "Goods Issue",
    }

    # Default disposal methods by waste type (configurable)
    DISPOSAL_METHOD_DEFAULTS = {
        "Rejected": "Return to Supplier",
        "Scrap": "Recycle",
        "Quality Hold": "Pending",
        "Sample Destruction": "Incinerate"
    }

    # Emission factors (kg CO2e per kg of waste) - simplified
    # In production, these would come from a comprehensive database
    EMISSION_FACTORS = {
        "Landfill": 0.5,
        "Recycle": 0.1,
        "Incinerate": 0.8,
        "Return to Supplier": 0.3,  # Transport emissions
        "Pending": 0.0
    }

    def __init__(self, tenant_id: Optional[str] = None):
        """Initialize Quality Inspection mapper.

        Args:
            tenant_id: Tenant identifier for multi-tenant deployment
        """
        self.tenant_id = tenant_id
        logger.info(f"Initialized QualityInspectionMapper for tenant: {tenant_id}")

    def _standardize_unit(self, sap_unit: Optional[str]) -> str:
        """Convert SAP unit to VCCI standard unit.

        Args:
            sap_unit: SAP unit of measure

        Returns:
            Standardized VCCI unit
        """
        if not sap_unit:
            return "units"

        sap_unit_upper = sap_unit.upper().strip()
        return self.UNIT_MAPPING.get(sap_unit_upper, "units")

    def _map_waste_category(self, usage_decision: Optional[str]) -> str:
        """Map SAP usage decision to waste category.

        Args:
            usage_decision: SAP usage decision code

        Returns:
            Waste category string
        """
        if not usage_decision:
            return "Unknown"

        return self.USAGE_DECISION_MAPPING.get(usage_decision.upper(), "Other")

    def _map_source_type(self, origin: Optional[str]) -> str:
        """Map SAP inspection origin to source type.

        Args:
            origin: SAP inspection lot origin

        Returns:
            Source type string
        """
        if not origin:
            return "Unknown"

        return self.ORIGIN_MAPPING.get(origin, "Other")

    def _determine_disposal_method(self, waste_category: str) -> str:
        """Determine disposal method based on waste category.

        Args:
            waste_category: Waste category

        Returns:
            Disposal method
        """
        return self.DISPOSAL_METHOD_DEFAULTS.get(waste_category, "Unknown")

    def _calculate_emissions(
        self,
        waste_quantity: float,
        disposal_method: str,
        unit: str
    ) -> float:
        """Calculate estimated emissions from waste disposal.

        Args:
            waste_quantity: Quantity of waste
            disposal_method: How waste is disposed
            unit: Unit of measure

        Returns:
            Estimated emissions in kg CO2e
        """
        # Convert to kg if necessary
        quantity_kg = waste_quantity
        if unit.lower() == 'tonnes':
            quantity_kg = waste_quantity * 1000
        elif unit.lower() == 'g':
            quantity_kg = waste_quantity / 1000
        elif unit.lower() == 'lbs':
            quantity_kg = waste_quantity * 0.453592

        # Get emission factor
        emission_factor = self.EMISSION_FACTORS.get(disposal_method, 0.5)

        # Calculate emissions
        emissions = quantity_kg * emission_factor

        return round(emissions, 2)

    def _extract_reporting_year(self, date_str: str) -> int:
        """Extract reporting year from date string.

        Args:
            date_str: ISO date string (YYYY-MM-DD)

        Returns:
            Year as integer
        """
        try:
            return int(date_str[:4])
        except (ValueError, TypeError):
            return DeterministicClock.now().year

    def map_inspection_lot(
        self,
        inspection_lot: Dict[str, Any],
        facility_master: Optional[Dict[str, Any]] = None
    ) -> Optional[WasteRecord]:
        """Map SAP Inspection Lot to VCCI waste record.

        Only creates a waste record if there is rejected or scrap quantity.

        Args:
            inspection_lot: SAP Inspection Lot data
            facility_master: Optional facility/plant master data

        Returns:
            WasteRecord if waste exists, None otherwise

        Raises:
            ValueError: If required fields are missing
        """
        # Required fields validation
        if not inspection_lot.get("InspectionLot"):
            raise ValueError("Missing required field: InspectionLot")
        if not inspection_lot.get("Plant"):
            raise ValueError("Missing required field: Plant")

        # Calculate total waste quantity
        rejected_qty = inspection_lot.get("InspectionLotRejectedQuantity", 0.0) or 0.0
        scrap_qty = inspection_lot.get("InspectionLotScrapQuantity", 0.0) or 0.0
        total_waste = rejected_qty + scrap_qty

        # Only create waste record if there is actual waste
        if total_waste <= 0:
            logger.debug(f"Inspection lot {inspection_lot['InspectionLot']}: No waste to record")
            return None

        # Generate waste transaction ID
        waste_transaction_id = f"QM-WASTE-{inspection_lot['InspectionLot']}"

        # Transaction date
        transaction_date = inspection_lot.get("InspectionLotEndDate") or \
                          inspection_lot.get("InspectionLotStartDate", "")

        if not transaction_date:
            transaction_date = DeterministicClock.now().strftime("%Y-%m-%d")
            logger.warning(f"Inspection lot {inspection_lot['InspectionLot']}: Missing date")

        # Facility information
        facility_id = inspection_lot["Plant"]
        facility_name = None
        facility_country = None
        facility_region = None

        if facility_master:
            facility_name = facility_master.get("PlantName")
            facility_country = facility_master.get("Country")
            # Map country to region
            if facility_country in ["US", "CA", "MX"]:
                facility_region = "US"
            elif facility_country in ["DE", "FR", "GB", "IT", "ES", "NL"]:
                facility_region = "EU"
            elif facility_country in ["CN", "JP", "KR", "IN", "SG"]:
                facility_region = "APAC"

        # Material information
        material_code = inspection_lot.get("Material", "UNKNOWN")
        material_name = inspection_lot.get("MaterialName", f"Material {material_code}")

        # Unit
        sap_unit = inspection_lot.get("InspectionLotQuantityUnit", "")
        unit = self._standardize_unit(sap_unit)

        # Waste categorization
        usage_decision = inspection_lot.get("InspLotUsageDecisionCode", "")
        waste_category = self._map_waste_category(usage_decision)

        # Source type
        origin = inspection_lot.get("InspectionLotOrigin", "")
        source_type = self._map_source_type(origin)

        # Waste type
        if scrap_qty > 0 and rejected_qty > 0:
            waste_type = "Mixed (Scrap + Rejected)"
        elif scrap_qty > 0:
            waste_type = "Scrap"
        elif rejected_qty > 0:
            waste_type = "Rejected"
        else:
            waste_type = "Other"

        # Disposal method
        disposal_method = self._determine_disposal_method(waste_category)

        # Calculate emissions
        emissions = self._calculate_emissions(total_waste, disposal_method, unit)

        # Metadata
        metadata = {
            "source_system": "SAP_S4HANA_QM",
            "source_document_id": f"INSP-{inspection_lot['InspectionLot']}",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "Validated",
            "validation_errors": [],
            "manual_review_required": False,
            "created_by": "sap-qm-extractor",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Custom fields (SAP-specific)
        custom_fields = {
            "inspection_lot": inspection_lot.get("InspectionLot"),
            "inspection_lot_origin": origin,
            "inspection_lot_origin_desc": source_type,
            "inspection_lot_type": inspection_lot.get("InspectionLotType"),
            "usage_decision_code": usage_decision,
            "usage_decision_valuation": inspection_lot.get("InspLotUsageDecisionValuation"),
            "batch": inspection_lot.get("Batch"),
            "material_document": inspection_lot.get("MaterialDocument"),
            "material_document_year": inspection_lot.get("MaterialDocumentYear"),
            "inspection_quantity_total": inspection_lot.get("InspectionLotQuantity"),
            "inspection_quantity_actual": inspection_lot.get("InspectionLotActualQuantity"),
            "rejected_quantity": rejected_qty,
            "scrap_quantity": scrap_qty,
            "is_released": inspection_lot.get("InspectionLotIsReleased"),
            "is_completed": inspection_lot.get("InspectionLotIsCompleted"),
            "is_closed": inspection_lot.get("InspectionLotIsClosed"),
            "created_by_user": inspection_lot.get("CreatedByUser"),
            "creation_date": inspection_lot.get("CreationDate"),
            "changed_by_user": inspection_lot.get("LastChangedByUser"),
            "changed_on": inspection_lot.get("ChangedOn"),
        }

        # Build waste record
        record = WasteRecord(
            waste_transaction_id=waste_transaction_id,
            tenant_id=self.tenant_id,
            transaction_date=transaction_date,
            reporting_year=self._extract_reporting_year(transaction_date),
            facility_id=facility_id,
            facility_name=facility_name,
            facility_country=facility_country,
            facility_region=facility_region,
            material_code=material_code,
            material_name=material_name,
            waste_quantity=total_waste,
            unit=unit,
            waste_category=waste_category,
            waste_type=waste_type,
            disposal_method=disposal_method,
            source_document_type="Inspection Lot",
            source_document_id=inspection_lot.get("InspectionLot"),
            supplier_id=inspection_lot.get("Supplier"),
            manufacturing_order=inspection_lot.get("ManufacturingOrder"),
            purchase_order=inspection_lot.get("PurchaseOrder"),
            inspection_result=usage_decision,
            emissions_kg_co2e=emissions,
            metadata=metadata,
            custom_fields=custom_fields,
        )

        logger.debug(
            f"Mapped Inspection Lot {waste_transaction_id}: {total_waste} {unit} "
            f"{waste_type}, {emissions:.2f} kg CO2e"
        )

        return record

    def map_batch(
        self,
        inspection_lots: List[Dict[str, Any]],
        facility_lookup: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[WasteRecord]:
        """Map a batch of inspection lot records.

        Args:
            inspection_lots: List of SAP inspection lot dicts
            facility_lookup: Optional dict mapping plant ID to plant master data

        Returns:
            List of mapped WasteRecord objects (only those with waste)
        """
        records = []
        facility_lookup = facility_lookup or {}
        total_lots = len(inspection_lots)
        lots_with_waste = 0

        for lot in inspection_lots:
            try:
                plant_id = lot.get("Plant")
                facility_data = facility_lookup.get(plant_id) if plant_id else None

                record = self.map_inspection_lot(lot, facility_data)

                # Only append if waste was generated
                if record:
                    records.append(record)
                    lots_with_waste += 1

            except Exception as e:
                logger.error(f"Error mapping inspection lot: {e}", exc_info=True)
                continue

        logger.info(
            f"Mapped {lots_with_waste} inspection lots with waste "
            f"from {total_lots} total lots ({(lots_with_waste/total_lots*100):.1f}% rejection rate)"
        )
        return records

    def calculate_total_waste_emissions(
        self,
        waste_records: List[WasteRecord]
    ) -> Dict[str, Any]:
        """Calculate aggregate waste emissions statistics.

        Args:
            waste_records: List of waste records

        Returns:
            Dictionary with aggregate statistics
        """
        stats = {
            'total_waste_quantity': 0.0,
            'total_emissions_kg_co2e': 0.0,
            'waste_by_category': {},
            'waste_by_type': {},
            'waste_by_disposal_method': {},
            'emissions_by_facility': {},
            'top_waste_materials': {}
        }

        for record in waste_records:
            qty = record.waste_quantity
            emissions = record.emissions_kg_co2e or 0.0

            stats['total_waste_quantity'] += qty
            stats['total_emissions_kg_co2e'] += emissions

            # By category
            category = record.waste_category or 'Unknown'
            stats['waste_by_category'][category] = \
                stats['waste_by_category'].get(category, 0.0) + qty

            # By type
            waste_type = record.waste_type or 'Unknown'
            stats['waste_by_type'][waste_type] = \
                stats['waste_by_type'].get(waste_type, 0.0) + qty

            # By disposal method
            disposal = record.disposal_method or 'Unknown'
            stats['waste_by_disposal_method'][disposal] = \
                stats['waste_by_disposal_method'].get(disposal, 0.0) + qty

            # By facility
            facility = record.facility_id
            if facility not in stats['emissions_by_facility']:
                stats['emissions_by_facility'][facility] = {
                    'waste_quantity': 0.0,
                    'emissions_kg_co2e': 0.0
                }
            stats['emissions_by_facility'][facility]['waste_quantity'] += qty
            stats['emissions_by_facility'][facility]['emissions_kg_co2e'] += emissions

            # Top materials
            material = record.material_code
            if material not in stats['top_waste_materials']:
                stats['top_waste_materials'][material] = {
                    'material_name': record.material_name,
                    'waste_quantity': 0.0,
                    'emissions_kg_co2e': 0.0
                }
            stats['top_waste_materials'][material]['waste_quantity'] += qty
            stats['top_waste_materials'][material]['emissions_kg_co2e'] += emissions

        # Sort top materials by quantity
        stats['top_waste_materials'] = dict(
            sorted(
                stats['top_waste_materials'].items(),
                key=lambda x: x[1]['waste_quantity'],
                reverse=True
            )[:10]  # Top 10
        )

        logger.info(
            f"Total waste: {stats['total_waste_quantity']:.2f}, "
            f"Total emissions: {stats['total_emissions_kg_co2e']:.2f} kg CO2e"
        )

        return stats
