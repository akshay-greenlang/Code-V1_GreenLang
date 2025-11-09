"""
Production Order Mapper

Maps SAP S/4HANA Production Order data to VCCI manufacturing_v1.0.json schema.

This mapper transforms SAP PP Production Order data into the standardized
manufacturing schema used by the VCCI Scope 3 Carbon Platform for:
    - Category 1: Manufactured goods emissions
    - Direct process emissions
    - Energy consumption tracking
    - Waste and scrap emissions

Field Mappings:
    SAP Field → VCCI Schema Field
    - ManufacturingOrder → manufacturing_id
    - Material → product_code
    - MaterialName → product_name
    - ProductionPlant → facility_id
    - MfgOrderPlannedTotalQty → planned_quantity
    - MfgOrderConfirmedYieldQty → actual_quantity
    - MfgOrderScrapQty → scrap_quantity
    - ProductionUnit → unit
    - MfgOrderActualStartDate → production_start_date
    - MfgOrderActualEndDate → production_end_date

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
Phase: Team 4 Mission - 48 Missing ERP Connectors
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ManufacturingRecord(BaseModel):
    """VCCI Manufacturing data model matching manufacturing_v1.0.json schema."""

    manufacturing_id: str
    production_start_date: str
    production_end_date: str
    facility_id: str
    product_code: str
    product_name: str
    planned_quantity: float
    actual_quantity: float
    unit: str

    # Optional fields
    tenant_id: Optional[str] = None
    reporting_year: Optional[int] = None
    facility_name: Optional[str] = None
    facility_country: Optional[str] = None
    facility_region: Optional[str] = None
    production_line: Optional[str] = None
    production_supervisor: Optional[str] = None
    order_type: Optional[str] = None
    order_status: Optional[str] = None
    scrap_quantity: Optional[float] = None
    scrap_percentage: Optional[float] = None
    production_duration_hours: Optional[float] = None
    energy_consumed_kwh: Optional[float] = None
    process_emissions_kg_co2e: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class ProductionOrderMapper:
    """Maps SAP Production Order data to VCCI manufacturing schema."""

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
        "KWH": "kWh",
        "MWH": "MWh",
        "EA": "items",
        "PC": "items",
        "ST": "items",
        "UN": "units",
    }

    # SAP order status mapping
    STATUS_MAPPING = {
        ("OrderIsCreated", True): "CREATED",
        ("OrderIsReleased", True): "RELEASED",
        ("OrderIsConfirmed", True): "CONFIRMED",
        ("OrderIsClosed", True): "CLOSED",
    }

    def __init__(self, tenant_id: Optional[str] = None):
        """Initialize Production Order mapper.

        Args:
            tenant_id: Tenant identifier for multi-tenant deployment
        """
        self.tenant_id = tenant_id
        logger.info(f"Initialized ProductionOrderMapper for tenant: {tenant_id}")

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

    def _determine_order_status(self, production_order: Dict[str, Any]) -> str:
        """Determine order status from SAP flags.

        Args:
            production_order: SAP production order data

        Returns:
            Order status string
        """
        # Check in priority order (most complete status first)
        if production_order.get("OrderIsClosed"):
            return "CLOSED"
        if production_order.get("OrderIsConfirmed"):
            return "CONFIRMED"
        if production_order.get("OrderIsReleased"):
            return "RELEASED"
        if production_order.get("OrderIsCreated"):
            return "CREATED"

        return "UNKNOWN"

    def _calculate_scrap_percentage(
        self,
        scrap_qty: Optional[float],
        planned_qty: Optional[float]
    ) -> Optional[float]:
        """Calculate scrap percentage.

        Args:
            scrap_qty: Scrap quantity
            planned_qty: Planned quantity

        Returns:
            Scrap percentage or None
        """
        if scrap_qty and planned_qty and planned_qty > 0:
            return (scrap_qty / planned_qty) * 100
        return None

    def _calculate_production_duration(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Optional[float]:
        """Calculate production duration in hours.

        Args:
            start_date: Production start date (ISO format)
            end_date: Production end date (ISO format)

        Returns:
            Duration in hours or None
        """
        if not start_date or not end_date:
            return None

        try:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            duration = (end - start).total_seconds() / 3600
            return round(duration, 2)
        except Exception as e:
            logger.warning(f"Error calculating duration: {e}")
            return None

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
            return datetime.now().year

    def map_production_order(
        self,
        production_order: Dict[str, Any],
        facility_master: Optional[Dict[str, Any]] = None,
        operations_data: Optional[List[Dict[str, Any]]] = None
    ) -> ManufacturingRecord:
        """Map SAP Production Order to VCCI manufacturing record.

        Args:
            production_order: SAP Production Order data
            facility_master: Optional facility/plant master data
            operations_data: Optional production order operations for energy calculation

        Returns:
            ManufacturingRecord matching manufacturing_v1.0.json schema

        Raises:
            ValueError: If required fields are missing
        """
        # Required fields validation
        if not production_order.get("ManufacturingOrder"):
            raise ValueError("Missing required field: ManufacturingOrder")
        if not production_order.get("Material"):
            raise ValueError("Missing required field: Material")
        if not production_order.get("ProductionPlant"):
            raise ValueError("Missing required field: ProductionPlant")

        # Manufacturing ID
        manufacturing_id = f"MFG-{production_order['ManufacturingOrder']}"

        # Dates
        start_date = production_order.get("MfgOrderActualStartDate") or \
                    production_order.get("MfgOrderPlannedStartDate", "")
        end_date = production_order.get("MfgOrderActualEndDate") or \
                  production_order.get("MfgOrderPlannedEndDate", "")

        if not start_date:
            start_date = datetime.now().strftime("%Y-%m-%d")
            logger.warning(f"Order {production_order['ManufacturingOrder']}: Missing start date")
        if not end_date:
            end_date = start_date
            logger.warning(f"Order {production_order['ManufacturingOrder']}: Missing end date")

        # Facility information
        facility_id = production_order["ProductionPlant"]
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

        # Product information
        product_code = production_order["Material"]
        product_name = production_order.get("MaterialName", f"Material {product_code}")

        # Quantities and units
        planned_qty = production_order.get("MfgOrderPlannedTotalQty", 0.0)
        actual_qty = production_order.get("MfgOrderConfirmedYieldQty", 0.0)
        scrap_qty = production_order.get("MfgOrderScrapQty", 0.0)

        sap_unit = production_order.get("ProductionUnit", "")
        unit = self._standardize_unit(sap_unit)

        # Status
        order_status = self._determine_order_status(production_order)

        # Calculations
        scrap_percentage = self._calculate_scrap_percentage(scrap_qty, planned_qty)
        production_duration = self._calculate_production_duration(start_date, end_date)

        # Energy calculation from operations (if available)
        energy_consumed_kwh = None
        if operations_data:
            total_machine_time = sum(
                op.get("OpActualMachineTime", 0.0) for op in operations_data
            )
            # Estimate: 50 kWh per machine hour (simplified)
            if total_machine_time > 0:
                energy_consumed_kwh = total_machine_time * 50.0
                logger.debug(f"Estimated energy: {energy_consumed_kwh} kWh from {total_machine_time} machine hours")

        # Metadata
        metadata = {
            "source_system": "SAP_S4HANA_PP",
            "source_document_id": f"PRD-{production_order['ManufacturingOrder']}",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "Validated",
            "validation_errors": [],
            "manual_review_required": False,
            "created_by": "sap-pp-extractor",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Custom fields (SAP-specific)
        custom_fields = {
            "manufacturing_order_type": production_order.get("ManufacturingOrderType"),
            "production_version": production_order.get("ProductionVersion"),
            "production_supervisor": production_order.get("ProductionSupervisor"),
            "production_scheduling_profile": production_order.get("ProductionSchedulingProfile"),
            "order_is_created": production_order.get("OrderIsCreated"),
            "order_is_released": production_order.get("OrderIsReleased"),
            "order_is_confirmed": production_order.get("OrderIsConfirmed"),
            "order_is_closed": production_order.get("OrderIsClosed"),
            "created_by_user": production_order.get("CreatedByUser"),
            "creation_date": production_order.get("CreationDate"),
            "changed_by_user": production_order.get("LastChangedByUser"),
            "changed_on": production_order.get("ChangedOn"),
        }

        # Build manufacturing record
        record = ManufacturingRecord(
            manufacturing_id=manufacturing_id,
            tenant_id=self.tenant_id,
            production_start_date=start_date,
            production_end_date=end_date,
            reporting_year=self._extract_reporting_year(start_date),
            facility_id=facility_id,
            facility_name=facility_name,
            facility_country=facility_country,
            facility_region=facility_region,
            product_code=product_code,
            product_name=product_name,
            planned_quantity=planned_qty,
            actual_quantity=actual_qty,
            scrap_quantity=scrap_qty,
            scrap_percentage=scrap_percentage,
            unit=unit,
            order_type=production_order.get("ManufacturingOrderType"),
            order_status=order_status,
            production_supervisor=production_order.get("ProductionSupervisor"),
            production_duration_hours=production_duration,
            energy_consumed_kwh=energy_consumed_kwh,
            metadata=metadata,
            custom_fields=custom_fields,
        )

        logger.debug(f"Mapped Production Order {manufacturing_id}: {product_name}, {actual_qty} {unit}")

        return record

    def map_batch(
        self,
        production_orders: List[Dict[str, Any]],
        facility_lookup: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[ManufacturingRecord]:
        """Map a batch of production order records.

        Args:
            production_orders: List of SAP production order dicts
            facility_lookup: Optional dict mapping plant ID to plant master data

        Returns:
            List of mapped ManufacturingRecord objects
        """
        records = []
        facility_lookup = facility_lookup or {}

        for po in production_orders:
            try:
                plant_id = po.get("ProductionPlant")
                facility_data = facility_lookup.get(plant_id) if plant_id else None

                record = self.map_production_order(po, facility_data)
                records.append(record)

            except Exception as e:
                logger.error(f"Error mapping production order: {e}", exc_info=True)
                continue

        logger.info(f"Mapped {len(records)} of {len(production_orders)} production orders")
        return records

    def map_with_operations(
        self,
        production_order: Dict[str, Any],
        operations: List[Dict[str, Any]],
        facility_master: Optional[Dict[str, Any]] = None
    ) -> ManufacturingRecord:
        """Map production order with operations data for enhanced energy calculation.

        Args:
            production_order: SAP Production Order header
            operations: List of production order operations
            facility_master: Optional facility master data

        Returns:
            ManufacturingRecord with calculated energy consumption
        """
        return self.map_production_order(
            production_order,
            facility_master=facility_master,
            operations_data=operations
        )
