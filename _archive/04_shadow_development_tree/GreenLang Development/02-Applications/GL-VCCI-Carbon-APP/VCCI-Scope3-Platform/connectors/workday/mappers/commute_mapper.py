# -*- coding: utf-8 -*-
"""
Workday Commute to Category 7 Mapper
GL-VCCI Scope 3 Platform

Maps Workday commute survey data to Category 7: Employee Commuting emissions.
Creates custom schema format for commute data.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

from ..extractors.hcm_extractor import CommuteData
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class CommuteMapper:
    """
    Maps Workday commute surveys to Category 7 emissions format.

    Creates structured commute records for employee commuting emissions
    calculations (GHG Protocol Scope 3, Category 7).
    """

    # Emission factors (kg CO2e per km) - placeholder values
    # In production, these would come from a factor database
    EMISSION_FACTORS = {
        "Car": 0.192,  # Average car
        "Bus": 0.089,
        "Train": 0.041,
        "Metro": 0.041,
        "Motorcycle": 0.113,
        "Carpool": 0.096,  # Car / 2
        "Bike": 0.0,
        "Walk": 0.0,
        "Remote": 0.0,
        "Other": 0.15,  # Conservative estimate
    }

    def __init__(self, tenant_id: str = "tenant-default"):
        """
        Initialize commute mapper.

        Args:
            tenant_id: Tenant identifier for multi-tenant environments
        """
        self.tenant_id = tenant_id

    def map_commute_to_category7(
        self,
        commute: CommuteData
    ) -> Dict[str, Any]:
        """
        Map commute survey to Category 7 emissions format.

        Args:
            commute: Commute survey data

        Returns:
            Category 7 commute record dict

        Example:
            {
                "commute_id": "COMMUTE-12345",
                "employee_id": "EMP-001",
                "commute_mode": "Car",
                "annual_emissions_kg_co2e": 1250.5,
                ...
            }
        """
        # Generate commute ID
        commute_id = self._generate_commute_id(commute)

        # Calculate annual distance
        annual_distance_km = self._calculate_annual_distance(commute)

        # Calculate emissions
        emission_factor = self.EMISSION_FACTORS.get(
            commute.commute_mode,
            self.EMISSION_FACTORS["Other"]
        )

        # Adjust for carpool
        if commute.carpool_size and commute.carpool_size > 1:
            emission_factor = emission_factor / commute.carpool_size

        annual_emissions_kg_co2e = annual_distance_km * emission_factor

        # Build commute record
        commute_record = {
            "commute_id": commute_id,
            "tenant_id": self.tenant_id,
            "employee_id": commute.employee_id,
            "employee_name": commute.employee_name,
            "survey_date": commute.survey_date.isoformat(),
            "reporting_year": commute.survey_date.year,
            "home_location": commute.home_location,
            "office_location": commute.office_location,
            "commute_mode": commute.commute_mode,
            "commute_frequency_days_per_week": commute.commute_frequency_days_per_week,
            "distance_km_one_way": commute.distance_km_one_way,
            "annual_distance_km": round(annual_distance_km, 2),
            "emission_factor_kg_co2e_per_km": round(emission_factor, 4),
            "annual_emissions_kg_co2e": round(annual_emissions_kg_co2e, 2),
            "vehicle_type": commute.vehicle_type,
            "carpool_size": commute.carpool_size,
            "calculation_method": "distance_based",
            "ghg_category": "Scope3_Category7_EmployeeCommuting",
        }

        # Add metadata
        commute_record["metadata"] = {
            "source_system": "Workday",
            "extraction_timestamp": DeterministicClock.utcnow().isoformat() + "Z",
            "validation_status": "Validated",
            "created_by": "workday-connector",
            "created_at": DeterministicClock.utcnow().isoformat() + "Z",
        }

        # Add data quality indicators
        commute_record["data_quality_indicators"] = self._get_dqi(commute)

        return commute_record

    def map_commutes_batch(
        self,
        commutes: List[CommuteData]
    ) -> List[Dict[str, Any]]:
        """
        Map batch of commute surveys to Category 7 records.

        Args:
            commutes: List of commute survey data

        Returns:
            List of Category 7 commute records
        """
        commute_records = []

        for commute in commutes:
            try:
                record = self.map_commute_to_category7(commute)
                commute_records.append(record)
            except Exception as e:
                logger.error(
                    f"Failed to map commute for employee {commute.employee_id}: {e}"
                )
                continue

        logger.info(
            f"Mapped {len(commute_records)} commute records "
            f"from {len(commutes)} surveys"
        )

        return commute_records

    def _calculate_annual_distance(self, commute: CommuteData) -> float:
        """
        Calculate annual commute distance.

        Args:
            commute: Commute survey data

        Returns:
            Annual distance in km
        """
        if not commute.distance_km_one_way:
            # If distance not provided, use default estimate
            return 0.0

        # Calculate: one_way_distance * 2 (round trip) * days_per_week * 52 weeks
        annual_distance = (
            commute.distance_km_one_way * 2 *
            commute.commute_frequency_days_per_week * 52
        )

        return annual_distance

    def _generate_commute_id(self, commute: CommuteData) -> str:
        """
        Generate unique commute ID from survey data.

        Args:
            commute: Commute survey data

        Returns:
            Commute ID string
        """
        # Create hash from employee ID and survey date
        hash_input = f"{commute.employee_id}_{commute.survey_date}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:12].upper()

        return f"COMMUTE-WD-{hash_value}"

    def _get_dqi(self, commute: CommuteData) -> Dict[str, Any]:
        """
        Calculate Data Quality Indicators for commute.

        Args:
            commute: Commute survey data

        Returns:
            DQI dictionary
        """
        # Score based on data completeness
        reliability = 4 if commute.distance_km_one_way else 2
        completeness = 5 if all([
            commute.distance_km_one_way,
            commute.commute_frequency_days_per_week,
            commute.home_location,
            commute.office_location
        ]) else 3

        temporal_correlation = 5  # Current survey data
        geographical_correlation = 4  # Specific locations
        technological_correlation = 4  # Specific transport modes

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
