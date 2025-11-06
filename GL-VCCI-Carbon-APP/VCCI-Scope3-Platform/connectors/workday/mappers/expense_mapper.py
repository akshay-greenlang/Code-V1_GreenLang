"""
Workday Expense to Logistics Mapper
GL-VCCI Scope 3 Platform

Maps Workday expense report data to logistics_v1.0.json schema
for Category 6: Business Travel emissions.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

from ..extractors.hcm_extractor import ExpenseReportData

logger = logging.getLogger(__name__)


class ExpenseMapper:
    """
    Maps Workday expense reports to logistics schema for business travel.

    Maps expense categories to transport modes and generates logistics records
    conforming to logistics_v1.0.json schema.
    """

    # Mapping of expense categories to transport modes
    CATEGORY_TO_TRANSPORT_MODE = {
        "Flight": "Air_Freight_ShortHaul",  # Default to short haul
        "Car Rental": "Road_Truck_LessThan7.5t",
        "Ground Transportation": "Road_Truck_LessThan7.5t",
        "Taxi": "Road_Truck_LessThan7.5t",
        "Rideshare": "Road_Truck_LessThan7.5t",
        "Train": "Rail_Freight",
        "Bus": "Road_Truck_LessThan7.5t",
    }

    # Distance thresholds for flight classification (km)
    FLIGHT_DISTANCE_THRESHOLD = 2500  # Short haul vs long haul

    def __init__(self, tenant_id: str = "tenant-default"):
        """
        Initialize expense mapper.

        Args:
            tenant_id: Tenant identifier for multi-tenant environments
        """
        self.tenant_id = tenant_id

    def map_expense_to_logistics(
        self,
        expense: ExpenseReportData
    ) -> Optional[Dict[str, Any]]:
        """
        Map expense report to logistics schema.

        Args:
            expense: Expense report data

        Returns:
            Logistics record dict or None if not mappable

        Example:
            {
                "shipment_id": "SHIP-EXP-12345",
                "transport_mode": "Air_Freight_ShortHaul",
                "calculation_method": "distance_based",
                ...
            }
        """
        # Only map travel-related expenses
        if not self._is_travel_expense(expense.expense_category):
            logger.debug(
                f"Skipping non-travel expense: {expense.expense_category}"
            )
            return None

        # Determine transport mode
        transport_mode = self._get_transport_mode(expense)

        # Generate shipment ID
        shipment_id = self._generate_shipment_id(expense)

        # Build logistics record
        logistics_record = {
            "shipment_id": shipment_id,
            "tenant_id": self.tenant_id,
            "shipment_date": expense.expense_date.isoformat(),
            "reporting_year": expense.expense_date.year,
            "transport_mode": transport_mode,
            "calculation_method": self._get_calculation_method(expense),
        }

        # Add origin/destination if available
        if expense.origin_city:
            logistics_record["origin"] = {
                "location_name": expense.origin_city,
                "city": expense.origin_city,
            }

        if expense.destination_city:
            logistics_record["destination"] = {
                "location_name": expense.destination_city,
                "city": expense.destination_city,
            }

        # Add distance if available
        if expense.distance_km:
            logistics_record["distance_km"] = expense.distance_km
            logistics_record["distance_source"] = "Carrier_Provided"
            logistics_record["weight_tonnes"] = 0.1  # Assume average passenger luggage
            logistics_record["weight_source"] = "Estimated"

        # Add spend data for spend-based method
        logistics_record["spend_usd"] = expense.amount
        logistics_record["spend_currency_original"] = expense.currency
        logistics_record["spend_amount_original"] = expense.amount

        # Add metadata
        logistics_record["metadata"] = {
            "source_system": "Workday",
            "source_document_id": expense.expense_id,
            "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
            "validation_status": "Validated",
            "created_by": "workday-connector",
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        # Add data quality indicators
        logistics_record["data_quality_indicators"] = self._get_dqi(expense)

        return logistics_record

    def map_expenses_batch(
        self,
        expenses: List[ExpenseReportData]
    ) -> List[Dict[str, Any]]:
        """
        Map batch of expense reports to logistics records.

        Args:
            expenses: List of expense reports

        Returns:
            List of logistics records
        """
        logistics_records = []

        for expense in expenses:
            try:
                record = self.map_expense_to_logistics(expense)
                if record:
                    logistics_records.append(record)
            except Exception as e:
                logger.error(
                    f"Failed to map expense {expense.expense_id}: {e}"
                )
                continue

        logger.info(
            f"Mapped {len(logistics_records)} logistics records "
            f"from {len(expenses)} expenses"
        )

        return logistics_records

    def _is_travel_expense(self, category: str) -> bool:
        """Check if expense category is travel-related."""
        return category in self.CATEGORY_TO_TRANSPORT_MODE

    def _get_transport_mode(self, expense: ExpenseReportData) -> str:
        """
        Determine transport mode from expense.

        Args:
            expense: Expense report data

        Returns:
            Transport mode string
        """
        # Get base transport mode
        transport_mode = self.CATEGORY_TO_TRANSPORT_MODE.get(
            expense.expense_category,
            "Road_Truck_LessThan7.5t"  # Default
        )

        # For flights, classify as short haul or long haul based on distance
        if expense.expense_category == "Flight" and expense.distance_km:
            if expense.distance_km > self.FLIGHT_DISTANCE_THRESHOLD:
                transport_mode = "Air_Freight_LongHaul"
            else:
                transport_mode = "Air_Freight_ShortHaul"

        return transport_mode

    def _get_calculation_method(self, expense: ExpenseReportData) -> str:
        """
        Determine calculation method based on available data.

        Args:
            expense: Expense report data

        Returns:
            Calculation method string
        """
        # Prefer distance-based if distance is available
        if expense.distance_km:
            return "distance_based"

        # Fall back to spend-based
        return "spend_based"

    def _generate_shipment_id(self, expense: ExpenseReportData) -> str:
        """
        Generate unique shipment ID from expense data.

        Args:
            expense: Expense report data

        Returns:
            Shipment ID string
        """
        # Create hash from expense ID and date
        hash_input = f"{expense.expense_id}_{expense.expense_date}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:12].upper()

        return f"SHIP-WD-{hash_value}"

    def _get_dqi(self, expense: ExpenseReportData) -> Dict[str, Any]:
        """
        Calculate Data Quality Indicators for expense.

        Args:
            expense: Expense report data

        Returns:
            DQI dictionary
        """
        # Score based on data completeness
        reliability = 4 if expense.distance_km else 3
        completeness = 5 if all([
            expense.origin_city,
            expense.destination_city,
            expense.distance_km
        ]) else 3

        temporal_correlation = 5  # Current data
        geographical_correlation = 4  # Specific locations
        technological_correlation = 3  # Generic transport modes

        dqi_score = (
            reliability + completeness + temporal_correlation +
            geographical_correlation + technological_correlation
        ) / 5.0

        if dqi_score >= 4.5:
            dqi_rating = "Excellent"
        elif dqi_score >= 3.5:
            dqi_rating = "Good"
        elif dqi_score >= 2.5:
            dqi_rating = "Fair"
        else:
            dqi_rating = "Poor"

        return {
            "reliability": reliability,
            "completeness": completeness,
            "temporal_correlation": temporal_correlation,
            "geographical_correlation": geographical_correlation,
            "technological_correlation": technological_correlation,
            "dqi_score": round(dqi_score, 2),
            "dqi_rating": dqi_rating
        }
