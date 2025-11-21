# -*- coding: utf-8 -*-
"""
Workday HCM Data Extractor
GL-VCCI Scope 3 Platform

Extracts expense reports and commute survey data from Workday HCM module
for Scope 3 Categories 6 (Business Travel) and 7 (Employee Commuting).

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import logging
import time
from typing import List, Dict, Any, Optional, Generator
from datetime import date, datetime
from pydantic import BaseModel, Field, validator

from .base import BaseExtractor
from ..exceptions import WorkdayDataError
from greenlang.determinism import FinancialDecimal

logger = logging.getLogger(__name__)


class ExpenseReportData(BaseModel):
    """
    Pydantic model for expense report data.

    Maps to Category 6: Business Travel emissions.
    """
    employee_id: str = Field(..., description="Employee ID")
    employee_name: str = Field(..., description="Employee name")
    expense_date: date = Field(..., description="Expense date")
    expense_category: str = Field(..., description="Expense category (Flight, Hotel, Car Rental, etc.)")
    amount: float = Field(..., ge=0, description="Expense amount")
    currency: str = Field(default="USD", description="Currency code")
    origin_city: Optional[str] = Field(None, description="Origin city (for travel)")
    destination_city: Optional[str] = Field(None, description="Destination city (for travel)")
    distance_km: Optional[float] = Field(None, ge=0, description="Travel distance in km")
    expense_id: str = Field(..., description="Unique expense ID")
    description: Optional[str] = Field(None, description="Expense description")

    @validator('expense_category')
    def validate_category(cls, v):
        """Validate expense category."""
        valid_categories = [
            "Flight", "Hotel", "Car Rental", "Ground Transportation",
            "Taxi", "Rideshare", "Train", "Bus", "Meal", "Other"
        ]
        if v not in valid_categories:
            logger.warning(f"Unknown expense category: {v}")
        return v

    class Config:
        json_encoders = {
            date: lambda v: v.isoformat()
        }


class CommuteData(BaseModel):
    """
    Pydantic model for commute survey data.

    Maps to Category 7: Employee Commuting emissions.
    """
    employee_id: str = Field(..., description="Employee ID")
    employee_name: str = Field(..., description="Employee name")
    survey_date: date = Field(..., description="Survey date")
    home_location: str = Field(..., description="Home location (city or address)")
    office_location: str = Field(..., description="Office location")
    commute_mode: str = Field(..., description="Commute mode (Car, Bus, Train, Bike, Walk, etc.)")
    commute_frequency_days_per_week: int = Field(..., ge=0, le=7, description="Days per week commuting")
    distance_km_one_way: Optional[float] = Field(None, ge=0, description="One-way distance in km")
    vehicle_type: Optional[str] = Field(None, description="Vehicle type if driving")
    carpool_size: Optional[int] = Field(None, ge=1, le=10, description="Number of people in carpool")

    @validator('commute_mode')
    def validate_mode(cls, v):
        """Validate commute mode."""
        valid_modes = [
            "Car", "Bus", "Train", "Metro", "Bike", "Walk",
            "Motorcycle", "Carpool", "Remote", "Other"
        ]
        if v not in valid_modes:
            logger.warning(f"Unknown commute mode: {v}")
        return v

    class Config:
        json_encoders = {
            date: lambda v: v.isoformat()
        }


class HCMExtractor(BaseExtractor):
    """
    Extractor for Workday HCM data.

    Extracts expense reports and commute surveys for Scope 3 emissions.
    """

    def extract_expense_reports(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[ExpenseReportData]:
        """
        Extract expense report data.

        Args:
            from_date: Start date for extraction
            to_date: End date for extraction

        Returns:
            List of expense report records

        Raises:
            WorkdayDataError: If extraction fails
        """
        from_date, to_date = self.validate_date_range(from_date, to_date)

        start_time = time.time()
        records = []

        try:
            # Extract data from Workday
            raw_data = self.client.get_report(
                report_name="expense_reports",
                from_date=from_date,
                to_date=to_date
            )

            # Transform to Pydantic models
            for raw_record in raw_data:
                try:
                    record = self._transform_expense_record(raw_record)
                    records.append(record)
                except Exception as e:
                    self.logger.error(
                        f"Failed to transform expense record: {e}. "
                        f"Raw data: {raw_record}"
                    )
                    continue

            elapsed = time.time() - start_time
            self.log_extraction_summary(len(records), from_date, to_date, elapsed)

            return records

        except Exception as e:
            self.logger.error(f"Expense report extraction failed: {e}")
            raise WorkdayDataError(
                data_type="ExpenseReport",
                reason=str(e),
                original_exception=e
            )

    def extract_commute_surveys(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[CommuteData]:
        """
        Extract commute survey data.

        Args:
            from_date: Start date for extraction
            to_date: End date for extraction

        Returns:
            List of commute survey records

        Raises:
            WorkdayDataError: If extraction fails
        """
        from_date, to_date = self.validate_date_range(from_date, to_date)

        start_time = time.time()
        records = []

        try:
            # Extract data from Workday
            raw_data = self.client.get_report(
                report_name="commute_surveys",
                from_date=from_date,
                to_date=to_date
            )

            # Transform to Pydantic models
            for raw_record in raw_data:
                try:
                    record = self._transform_commute_record(raw_record)
                    records.append(record)
                except Exception as e:
                    self.logger.error(
                        f"Failed to transform commute record: {e}. "
                        f"Raw data: {raw_record}"
                    )
                    continue

            elapsed = time.time() - start_time
            self.log_extraction_summary(len(records), from_date, to_date, elapsed)

            return records

        except Exception as e:
            self.logger.error(f"Commute survey extraction failed: {e}")
            raise WorkdayDataError(
                data_type="CommuteData",
                reason=str(e),
                original_exception=e
            )

    def extract(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract both expense and commute data.

        Args:
            from_date: Start date for extraction
            to_date: End date for extraction
            **kwargs: Additional parameters

        Returns:
            Dictionary with 'expenses' and 'commutes' keys
        """
        expenses = self.extract_expense_reports(from_date, to_date)
        commutes = self.extract_commute_surveys(from_date, to_date)

        return {
            "expenses": [e.dict() for e in expenses],
            "commutes": [c.dict() for c in commutes]
        }

    def extract_paginated(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        **kwargs
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Extract data with pagination (not implemented for this extractor).

        Use extract() instead which handles pagination internally.
        """
        yield self.extract(from_date, to_date, **kwargs)

    def _transform_expense_record(self, raw: Dict[str, Any]) -> ExpenseReportData:
        """
        Transform raw Workday expense record to ExpenseReportData.

        Args:
            raw: Raw record from Workday

        Returns:
            ExpenseReportData instance
        """
        # Parse date
        expense_date = self._parse_date(raw.get("Expense_Date"))

        return ExpenseReportData(
            employee_id=raw.get("Employee_ID", ""),
            employee_name=raw.get("Employee", ""),
            expense_date=expense_date,
            expense_category=raw.get("Expense_Category", "Other"),
            amount=FinancialDecimal.from_string(raw.get("Amount", 0)),
            currency=raw.get("Currency", "USD"),
            origin_city=raw.get("Origin_City"),
            destination_city=raw.get("Destination_City"),
            distance_km=float(raw.get("Distance_KM")) if raw.get("Distance_KM") else None,
            expense_id=raw.get("Expense_ID", ""),
            description=raw.get("Description")
        )

    def _transform_commute_record(self, raw: Dict[str, Any]) -> CommuteData:
        """
        Transform raw Workday commute record to CommuteData.

        Args:
            raw: Raw record from Workday

        Returns:
            CommuteData instance
        """
        # Parse date
        survey_date = self._parse_date(raw.get("Survey_Date"))

        return CommuteData(
            employee_id=raw.get("Employee_ID", ""),
            employee_name=raw.get("Employee", ""),
            survey_date=survey_date,
            home_location=raw.get("Home_Location", ""),
            office_location=raw.get("Office_Location", ""),
            commute_mode=raw.get("Commute_Mode", "Other"),
            commute_frequency_days_per_week=int(raw.get("Days_Per_Week", 5)),
            distance_km_one_way=float(raw.get("Distance_KM")) if raw.get("Distance_KM") else None,
            vehicle_type=raw.get("Vehicle_Type"),
            carpool_size=int(raw.get("Carpool_Size")) if raw.get("Carpool_Size") else None
        )

    def _parse_date(self, date_str: str) -> date:
        """
        Parse date string from Workday.

        Args:
            date_str: Date string (ISO format expected)

        Returns:
            date object

        Raises:
            ValueError: If date cannot be parsed
        """
        if not date_str:
            raise ValueError("Date string is empty")

        try:
            # Try ISO format first
            return datetime.fromisoformat(date_str.replace("Z", "")).date()
        except Exception:
            # Try other common formats
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except Exception:
                    continue

            raise ValueError(f"Cannot parse date: {date_str}")
