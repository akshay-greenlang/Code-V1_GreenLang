"""
Time Entry Mapper

Maps Workday Time Tracking data to VCCI commute_v1.0.json schema.

This mapper transforms Workday Time Entry data into the standardized
commute schema used by the VCCI Scope 3 Carbon Platform for:
    - Category 7: Employee commuting emissions
    - Remote work impact analysis
    - Office occupancy-based facility emissions
    - Business travel time tracking

Field Mappings:
    Workday Field → VCCI Schema Field
    - TimeEntryID → commute_transaction_id
    - EmployeeID → employee_id
    - EmployeeName → employee_name
    - WorkDate → date
    - LocationType → commute_type (ONSITE → 'Office', REMOTE → 'Remote')
    - HoursWorked → hours_at_location
    - WorkLocation → facility_location

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
Phase: Team 4 Mission - Priority ERP Connector Modules
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CommuteRecord(BaseModel):
    """VCCI Commute data model matching commute_v1.0.json schema."""

    commute_transaction_id: str
    employee_id: str
    date: str
    commute_type: str  # 'Office', 'Remote', 'Hybrid', 'Travel'
    commute_occurred: bool

    # Optional fields
    tenant_id: Optional[str] = None
    reporting_year: Optional[int] = None
    employee_name: Optional[str] = None
    employee_department: Optional[str] = None
    employee_cost_center: Optional[str] = None
    facility_location: Optional[str] = None
    facility_id: Optional[str] = None
    home_location: Optional[str] = None
    commute_mode: Optional[str] = None  # 'Car', 'Public Transit', 'Bicycle', 'Walk'
    commute_distance_km: Optional[float] = None
    commute_duration_minutes: Optional[float] = None
    hours_at_location: Optional[float] = None
    is_remote_work_day: Optional[bool] = None
    emissions_kg_co2e: Optional[float] = None
    emissions_avoided_kg_co2e: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class TimeEntryMapper:
    """Maps Workday Time Entry data to VCCI commute schema."""

    # Location type to commute type mapping
    LOCATION_TYPE_MAPPING = {
        "ONSITE": "Office",
        "REMOTE": "Remote",
        "HYBRID": "Hybrid",
        "TRAVEL": "Travel",
    }

    # Default commute assumptions (configurable per tenant)
    DEFAULT_COMMUTE_DISTANCE_KM = 25.0
    DEFAULT_COMMUTE_MODE = "Car"

    # Emission factors (kg CO2e per km) by commute mode
    EMISSION_FACTORS = {
        "Car": 0.171,  # Average passenger car
        "Car - Gasoline": 0.192,
        "Car - Diesel": 0.171,
        "Car - Electric": 0.053,
        "Car - Hybrid": 0.109,
        "Public Transit": 0.089,
        "Bus": 0.105,
        "Train": 0.041,
        "Metro/Subway": 0.028,
        "Bicycle": 0.0,
        "E-Bike": 0.005,
        "Walk": 0.0,
        "Motorcycle": 0.103,
        "Carpool": 0.057,  # Shared car
    }

    def __init__(
        self,
        tenant_id: Optional[str] = None,
        default_commute_distance_km: float = 25.0,
        default_commute_mode: str = "Car"
    ):
        """Initialize Time Entry mapper.

        Args:
            tenant_id: Tenant identifier for multi-tenant deployment
            default_commute_distance_km: Default commute distance assumption
            default_commute_mode: Default commute mode assumption
        """
        self.tenant_id = tenant_id
        self.default_commute_distance_km = default_commute_distance_km
        self.default_commute_mode = default_commute_mode
        logger.info(
            f"Initialized TimeEntryMapper for tenant: {tenant_id}, "
            f"default distance: {default_commute_distance_km} km, "
            f"default mode: {default_commute_mode}"
        )

    def _map_commute_type(self, location_type: Optional[str]) -> str:
        """Map Workday location type to commute type.

        Args:
            location_type: Workday location type

        Returns:
            Commute type string
        """
        if not location_type:
            return "Unknown"

        location_type_upper = location_type.upper()
        return self.LOCATION_TYPE_MAPPING.get(location_type_upper, "Other")

    def _determine_commute_occurred(self, commute_type: str) -> bool:
        """Determine if a commute occurred based on commute type.

        Args:
            commute_type: Commute type

        Returns:
            True if commute occurred, False otherwise
        """
        # Commute occurred for office work or travel
        return commute_type in ["Office", "Hybrid", "Travel"]

    def _calculate_emissions(
        self,
        commute_occurred: bool,
        commute_distance_km: float,
        commute_mode: str
    ) -> tuple[float, float]:
        """Calculate commute emissions and avoided emissions.

        Args:
            commute_occurred: Whether commute occurred
            commute_distance_km: Distance of commute in km
            commute_mode: Mode of transportation

        Returns:
            Tuple of (emissions_kg_co2e, emissions_avoided_kg_co2e)
        """
        if not commute_occurred:
            # Remote work - no emissions, but calculate avoided emissions
            # Assume default mode if they had commuted
            emission_factor = self.EMISSION_FACTORS.get(self.default_commute_mode, 0.171)
            avoided_emissions = commute_distance_km * 2 * emission_factor  # Round trip
            return 0.0, round(avoided_emissions, 2)

        # Commute occurred - calculate actual emissions
        emission_factor = self.EMISSION_FACTORS.get(commute_mode, 0.171)
        emissions = commute_distance_km * 2 * emission_factor  # Round trip
        return round(emissions, 2), 0.0

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

    def map_time_entry(
        self,
        time_entry: Dict[str, Any],
        employee_master: Optional[Dict[str, Any]] = None,
        commute_profile: Optional[Dict[str, Any]] = None
    ) -> CommuteRecord:
        """Map Workday Time Entry to VCCI commute record.

        Args:
            time_entry: Workday Time Entry data
            employee_master: Optional employee master data for enrichment
            commute_profile: Optional employee commute profile with distance/mode

        Returns:
            CommuteRecord matching commute_v1.0.json schema

        Raises:
            ValueError: If required fields are missing
        """
        # Required fields validation
        if not time_entry.get("TimeEntryID"):
            raise ValueError("Missing required field: TimeEntryID")
        if not time_entry.get("EmployeeID"):
            raise ValueError("Missing required field: EmployeeID")
        if not time_entry.get("WorkDate"):
            raise ValueError("Missing required field: WorkDate")

        # Generate commute transaction ID
        commute_transaction_id = f"COMMUTE-{time_entry['TimeEntryID']}"

        # Employee information
        employee_id = time_entry["EmployeeID"]
        employee_name = time_entry.get("EmployeeName") or \
                       (employee_master.get("EmployeeName") if employee_master else None) or \
                       f"Employee {employee_id}"

        # Date
        work_date = time_entry["WorkDate"]

        # Commute type and occurrence
        location_type = time_entry.get("LocationType", "ONSITE")
        commute_type = self._map_commute_type(location_type)
        commute_occurred = self._determine_commute_occurred(commute_type)
        is_remote = commute_type == "Remote"

        # Hours worked
        hours_at_location = time_entry.get("HoursWorked", 0.0)

        # Location information
        work_location = time_entry.get("WorkLocation")
        facility_id = None
        home_location = None

        if employee_master:
            home_location = employee_master.get("HomeAddress") or employee_master.get("City")

        # Commute details from profile or defaults
        commute_distance_km = self.default_commute_distance_km
        commute_mode = self.default_commute_mode

        if commute_profile:
            commute_distance_km = commute_profile.get("CommuteDistanceKm", commute_distance_km)
            commute_mode = commute_profile.get("CommuteMode", commute_mode)

        # Calculate emissions
        emissions, emissions_avoided = self._calculate_emissions(
            commute_occurred,
            commute_distance_km,
            commute_mode
        )

        # Department and cost center
        employee_department = employee_master.get("Department") if employee_master else None
        employee_cost_center = time_entry.get("CostCenter") or \
                              (employee_master.get("CostCenter") if employee_master else None)

        # Metadata
        metadata = {
            "source_system": "WORKDAY_TIME_TRACKING",
            "source_document_id": f"TIME-{time_entry['TimeEntryID']}",
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "Validated",
            "validation_errors": [],
            "manual_review_required": False,
            "created_by": "workday-time-extractor",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "calculation_method": "Default assumptions" if not commute_profile else "Employee profile"
        }

        # Custom fields (Workday-specific)
        custom_fields = {
            "time_entry_id": time_entry.get("TimeEntryID"),
            "time_type": time_entry.get("TimeType"),
            "location_type": location_type,
            "project_code": time_entry.get("ProjectCode"),
            "project_name": time_entry.get("ProjectName"),
            "task_code": time_entry.get("TaskCode"),
            "submitted_date": time_entry.get("SubmittedDate"),
            "approved_date": time_entry.get("ApprovedDate"),
            "approval_status": time_entry.get("ApprovalStatus"),
            "last_modified_date": time_entry.get("LastModifiedDate"),
            "commute_assumptions_used": not bool(commute_profile)
        }

        # Build commute record
        record = CommuteRecord(
            commute_transaction_id=commute_transaction_id,
            tenant_id=self.tenant_id,
            employee_id=employee_id,
            employee_name=employee_name,
            employee_department=employee_department,
            employee_cost_center=employee_cost_center,
            date=work_date,
            reporting_year=self._extract_reporting_year(work_date),
            commute_type=commute_type,
            commute_occurred=commute_occurred,
            facility_location=work_location,
            facility_id=facility_id,
            home_location=home_location,
            commute_mode=commute_mode if commute_occurred else None,
            commute_distance_km=commute_distance_km if commute_occurred else None,
            hours_at_location=hours_at_location,
            is_remote_work_day=is_remote,
            emissions_kg_co2e=emissions,
            emissions_avoided_kg_co2e=emissions_avoided,
            metadata=metadata,
            custom_fields=custom_fields,
        )

        logger.debug(
            f"Mapped Time Entry {commute_transaction_id}: {employee_name}, "
            f"{commute_type}, {emissions:.2f} kg CO2e"
        )

        return record

    def map_batch(
        self,
        time_entries: List[Dict[str, Any]],
        employee_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
        commute_profile_lookup: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[CommuteRecord]:
        """Map a batch of time entry records.

        Args:
            time_entries: List of Workday time entry dicts
            employee_lookup: Optional dict mapping employee ID to employee master data
            commute_profile_lookup: Optional dict mapping employee ID to commute profile

        Returns:
            List of mapped CommuteRecord objects
        """
        records = []
        employee_lookup = employee_lookup or {}
        commute_profile_lookup = commute_profile_lookup or {}

        for entry in time_entries:
            try:
                employee_id = entry.get("EmployeeID")
                employee_data = employee_lookup.get(employee_id) if employee_id else None
                commute_profile = commute_profile_lookup.get(employee_id) if employee_id else None

                record = self.map_time_entry(entry, employee_data, commute_profile)
                records.append(record)

            except Exception as e:
                logger.error(f"Error mapping time entry: {e}", exc_info=True)
                continue

        logger.info(f"Mapped {len(records)} of {len(time_entries)} time entries")
        return records

    def calculate_commute_statistics(
        self,
        commute_records: List[CommuteRecord]
    ) -> Dict[str, Any]:
        """Calculate aggregate commute statistics.

        Args:
            commute_records: List of commute records

        Returns:
            Dictionary with aggregate statistics
        """
        stats = {
            'total_work_days': len(commute_records),
            'total_commute_days': 0,
            'total_remote_days': 0,
            'total_emissions_kg_co2e': 0.0,
            'total_emissions_avoided_kg_co2e': 0.0,
            'net_emissions_kg_co2e': 0.0,
            'remote_work_percentage': 0.0,
            'unique_employees': set(),
            'emissions_by_commute_type': {},
            'emissions_by_employee': {},
            'emissions_by_department': {},
            'emissions_by_commute_mode': {},
            'average_emissions_per_commute_day': 0.0,
            'average_emissions_avoided_per_remote_day': 0.0,
        }

        for record in commute_records:
            employee_id = record.employee_id
            stats['unique_employees'].add(employee_id)

            if record.commute_occurred:
                stats['total_commute_days'] += 1
            else:
                stats['total_remote_days'] += 1

            emissions = record.emissions_kg_co2e or 0.0
            avoided = record.emissions_avoided_kg_co2e or 0.0

            stats['total_emissions_kg_co2e'] += emissions
            stats['total_emissions_avoided_kg_co2e'] += avoided

            # By commute type
            commute_type = record.commute_type
            if commute_type not in stats['emissions_by_commute_type']:
                stats['emissions_by_commute_type'][commute_type] = {
                    'count': 0,
                    'emissions': 0.0,
                    'avoided': 0.0
                }
            stats['emissions_by_commute_type'][commute_type]['count'] += 1
            stats['emissions_by_commute_type'][commute_type]['emissions'] += emissions
            stats['emissions_by_commute_type'][commute_type]['avoided'] += avoided

            # By employee
            if employee_id not in stats['emissions_by_employee']:
                stats['emissions_by_employee'][employee_id] = {
                    'employee_name': record.employee_name,
                    'commute_days': 0,
                    'remote_days': 0,
                    'emissions': 0.0,
                    'avoided': 0.0
                }
            employee_stats = stats['emissions_by_employee'][employee_id]
            if record.commute_occurred:
                employee_stats['commute_days'] += 1
            else:
                employee_stats['remote_days'] += 1
            employee_stats['emissions'] += emissions
            employee_stats['avoided'] += avoided

            # By department
            department = record.employee_department or 'Unknown'
            if department not in stats['emissions_by_department']:
                stats['emissions_by_department'][department] = {
                    'emissions': 0.0,
                    'avoided': 0.0
                }
            stats['emissions_by_department'][department]['emissions'] += emissions
            stats['emissions_by_department'][department]['avoided'] += avoided

            # By commute mode
            if record.commute_mode:
                mode = record.commute_mode
                if mode not in stats['emissions_by_commute_mode']:
                    stats['emissions_by_commute_mode'][mode] = {
                        'count': 0,
                        'emissions': 0.0
                    }
                stats['emissions_by_commute_mode'][mode]['count'] += 1
                stats['emissions_by_commute_mode'][mode]['emissions'] += emissions

        # Calculate derived metrics
        if stats['total_work_days'] > 0:
            stats['remote_work_percentage'] = \
                (stats['total_remote_days'] / stats['total_work_days']) * 100

        if stats['total_commute_days'] > 0:
            stats['average_emissions_per_commute_day'] = \
                stats['total_emissions_kg_co2e'] / stats['total_commute_days']

        if stats['total_remote_days'] > 0:
            stats['average_emissions_avoided_per_remote_day'] = \
                stats['total_emissions_avoided_kg_co2e'] / stats['total_remote_days']

        stats['net_emissions_kg_co2e'] = \
            stats['total_emissions_kg_co2e'] - stats['total_emissions_avoided_kg_co2e']

        stats['unique_employees'] = len(stats['unique_employees'])

        logger.info(
            f"Commute statistics: {stats['total_work_days']} work days, "
            f"{stats['remote_work_percentage']:.1f}% remote, "
            f"{stats['total_emissions_kg_co2e']:.2f} kg CO2e emitted, "
            f"{stats['total_emissions_avoided_kg_co2e']:.2f} kg CO2e avoided"
        )

        return stats
