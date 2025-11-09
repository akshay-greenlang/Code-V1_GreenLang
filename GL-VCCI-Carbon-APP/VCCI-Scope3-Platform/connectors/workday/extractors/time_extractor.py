"""
Workday Time Tracking Extractor

Extracts data from Workday Time Tracking including:
    - Employee Time Entries
    - Time Off Requests
    - Absence Records
    - Project Time Allocations
    - Overtime Hours

Use Cases:
    - Category 7: Employee commuting emissions (work-from-home vs office)
    - Category 6: Business travel time analysis
    - Remote work impact on carbon footprint
    - Facility energy consumption correlation with headcount
    - Overtime energy usage tracking

Carbon Impact: MEDIUM-HIGH
- Direct: Reduced commuting emissions during remote work
- Indirect: Office energy consumption based on occupancy
- Travel: Business travel time and frequency patterns

Workday API Endpoints:
    - Time_Entry (RaaS report or Time_Tracking web service)
    - Worker_Time_Off (Absence_Management web service)
    - Time_Off_Plan_Balance

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
Phase: Team 4 Mission - Priority ERP Connector Modules
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class TimeEntryData(BaseModel):
    """Workday Time Entry data model.

    Maps to Time_Entry data from Workday Time Tracking.
    """
    TimeEntryID: str
    EmployeeID: str
    EmployeeName: Optional[str] = None
    WorkDate: str
    TimeType: str  # 'Regular', 'Overtime', 'Remote', 'OnSite', 'Travel'
    HoursWorked: float
    ProjectCode: Optional[str] = None
    ProjectName: Optional[str] = None
    TaskCode: Optional[str] = None
    CostCenter: Optional[str] = None
    WorkLocation: Optional[str] = None  # 'Office', 'Home', 'Client Site', 'Travel'
    LocationType: Optional[str] = None  # 'ONSITE', 'REMOTE', 'HYBRID'
    SubmittedDate: Optional[str] = None
    ApprovedDate: Optional[str] = None
    ApprovalStatus: Optional[str] = None  # 'Pending', 'Approved', 'Rejected'
    LastModifiedDate: Optional[str] = None


class TimeOffData(BaseModel):
    """Workday Time Off data model.

    Maps to Worker_Time_Off from Absence_Management.
    """
    TimeOffRequestID: str
    EmployeeID: str
    EmployeeName: Optional[str] = None
    TimeOffType: str  # 'Vacation', 'Sick', 'Personal', 'Sabbatical'
    StartDate: str
    EndDate: str
    TotalDays: float
    TotalHours: Optional[float] = None
    RequestDate: Optional[str] = None
    ApprovalStatus: Optional[str] = None
    ApprovedBy: Optional[str] = None
    LastModifiedDate: Optional[str] = None


class ProjectTimeAllocationData(BaseModel):
    """Workday Project Time Allocation data model."""
    AllocationID: str
    EmployeeID: str
    ProjectCode: str
    ProjectName: Optional[str] = None
    AllocationStartDate: str
    AllocationEndDate: Optional[str] = None
    AllocatedHoursPerWeek: float
    AllocatedPercentage: float
    WorkLocationType: Optional[str] = None  # 'ONSITE', 'REMOTE', 'HYBRID'
    LastModifiedDate: Optional[str] = None


class EmployeeWorkPatternData(BaseModel):
    """Employee work pattern for carbon footprint analysis."""
    EmployeeID: str
    EmployeeName: Optional[str] = None
    WorkWeek: str  # ISO week (e.g., '2024-W12')
    TotalHoursWorked: float
    OnsiteHours: float
    RemoteHours: float
    TravelHours: float
    OvertimeHours: float
    DaysInOffice: int
    DaysRemote: int
    DaysTravel: int
    DaysOff: int
    PrimaryLocation: Optional[str] = None
    CostCenter: Optional[str] = None


class TimeExtractor(BaseExtractor):
    """Workday Time Tracking Extractor.

    Extracts employee time data from Workday for carbon emissions analysis,
    particularly commuting and facility energy consumption patterns.
    """

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        """Initialize Time extractor.

        Args:
            client: Workday API client instance
            config: Extraction configuration
        """
        super().__init__(client, config)
        self.service_name = "TIME"
        self._current_entity_set = "Time_Entry"  # Default

    def get_entity_set_name(self) -> str:
        """Get current entity set name."""
        return self._current_entity_set

    def get_changed_on_field(self) -> str:
        """Get field name for delta extraction."""
        return "LastModifiedDate"

    def extract_time_entries(
        self,
        employee_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        location_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        approval_status: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Time Entries from Workday.

        Args:
            employee_id: Filter by employee ID
            cost_center: Filter by cost center
            location_type: Filter by location type ('ONSITE', 'REMOTE', 'HYBRID', 'TRAVEL')
            date_from: Filter by work date from (ISO format YYYY-MM-DD)
            date_to: Filter by work date to (ISO format YYYY-MM-DD)
            approval_status: Filter by approval status

        Yields:
            Time Entry records as dictionaries
        """
        self._current_entity_set = "Time_Entry"

        additional_filters = []

        if employee_id:
            additional_filters.append(f"EmployeeID = '{employee_id}'")
        if cost_center:
            additional_filters.append(f"CostCenter = '{cost_center}'")
        if location_type:
            additional_filters.append(f"LocationType = '{location_type}'")
        if date_from:
            additional_filters.append(f"WorkDate >= '{date_from}'")
        if date_to:
            additional_filters.append(f"WorkDate <= '{date_to}'")
        if approval_status:
            additional_filters.append(f"ApprovalStatus = '{approval_status}'")

        logger.info(f"Extracting Time Entries with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="WorkDate desc"
        )

    def extract_time_off_requests(
        self,
        employee_id: Optional[str] = None,
        time_off_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        approval_status: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Time Off Requests from Workday.

        Args:
            employee_id: Filter by employee ID
            time_off_type: Filter by time off type
            date_from: Filter by start date from (ISO format)
            date_to: Filter by start date to (ISO format)
            approval_status: Filter by approval status

        Yields:
            Time Off records as dictionaries
        """
        self._current_entity_set = "Worker_Time_Off"

        additional_filters = []

        if employee_id:
            additional_filters.append(f"EmployeeID = '{employee_id}'")
        if time_off_type:
            additional_filters.append(f"TimeOffType = '{time_off_type}'")
        if date_from:
            additional_filters.append(f"StartDate >= '{date_from}'")
        if date_to:
            additional_filters.append(f"StartDate <= '{date_to}'")
        if approval_status:
            additional_filters.append(f"ApprovalStatus = '{approval_status}'")

        logger.info(f"Extracting Time Off Requests with filters: {additional_filters}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None,
            order_by="StartDate desc"
        )

    def extract_project_time_allocations(
        self,
        employee_id: Optional[str] = None,
        project_code: Optional[str] = None,
        date_from: Optional[str] = None,
        work_location_type: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract Project Time Allocations from Workday.

        Args:
            employee_id: Filter by employee ID
            project_code: Filter by project code
            date_from: Filter by allocation start date from (ISO format)
            work_location_type: Filter by work location type

        Yields:
            Project Time Allocation records as dictionaries
        """
        self._current_entity_set = "Project_Time_Allocation"

        additional_filters = []

        if employee_id:
            additional_filters.append(f"EmployeeID = '{employee_id}'")
        if project_code:
            additional_filters.append(f"ProjectCode = '{project_code}'")
        if date_from:
            additional_filters.append(f"AllocationStartDate >= '{date_from}'")
        if work_location_type:
            additional_filters.append(f"WorkLocationType = '{work_location_type}'")

        logger.info(f"Extracting Project Time Allocations")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    def extract_remote_vs_onsite_hours(
        self,
        cost_center: Optional[str] = None,
        date_from: str = None,
        date_to: str = None
    ) -> Dict[str, Any]:
        """Extract and aggregate remote vs onsite working hours.

        This method calculates the distribution of remote vs onsite work,
        which is critical for commuting emissions calculations (Category 7).

        Args:
            cost_center: Filter by cost center
            date_from: Start date (ISO format)
            date_to: End date (ISO format)

        Returns:
            Dictionary with work location summary statistics
        """
        logger.info(f"Calculating remote vs onsite hours from {date_from} to {date_to}")

        summary = {
            'period_from': date_from,
            'period_to': date_to,
            'cost_center': cost_center,
            'total_hours': 0.0,
            'onsite_hours': 0.0,
            'remote_hours': 0.0,
            'travel_hours': 0.0,
            'overtime_hours': 0.0,
            'total_employees': set(),
            'remote_percentage': 0.0,
            'onsite_percentage': 0.0,
            'hours_by_location_type': {},
            'employees_by_location_type': {}
        }

        try:
            # Get all time entries for the period
            time_entries = list(self.extract_time_entries(
                cost_center=cost_center,
                date_from=date_from,
                date_to=date_to,
                approval_status='Approved'
            ))

            for entry in time_entries:
                employee_id = entry.get('EmployeeID')
                hours = entry.get('HoursWorked', 0.0)
                location_type = entry.get('LocationType', 'UNKNOWN')
                time_type = entry.get('TimeType', 'Regular')

                summary['total_hours'] += hours
                summary['total_employees'].add(employee_id)

                # Categorize by location
                if location_type == 'ONSITE':
                    summary['onsite_hours'] += hours
                elif location_type == 'REMOTE':
                    summary['remote_hours'] += hours
                elif location_type == 'TRAVEL':
                    summary['travel_hours'] += hours

                # Track overtime
                if time_type == 'Overtime':
                    summary['overtime_hours'] += hours

                # Aggregate by location type
                summary['hours_by_location_type'][location_type] = \
                    summary['hours_by_location_type'].get(location_type, 0.0) + hours

                if employee_id:
                    if location_type not in summary['employees_by_location_type']:
                        summary['employees_by_location_type'][location_type] = set()
                    summary['employees_by_location_type'][location_type].add(employee_id)

            # Calculate percentages
            if summary['total_hours'] > 0:
                summary['remote_percentage'] = (summary['remote_hours'] / summary['total_hours']) * 100
                summary['onsite_percentage'] = (summary['onsite_hours'] / summary['total_hours']) * 100

            # Convert sets to counts
            summary['total_employees'] = len(summary['total_employees'])
            summary['employees_by_location_type'] = {
                k: len(v) for k, v in summary['employees_by_location_type'].items()
            }

            logger.info(
                f"Work location summary: {summary['total_hours']:.1f} total hours, "
                f"{summary['remote_percentage']:.1f}% remote, "
                f"{summary['onsite_percentage']:.1f}% onsite, "
                f"{summary['total_employees']} employees"
            )

        except Exception as e:
            logger.error(f"Error calculating remote vs onsite hours: {e}", exc_info=True)

        return summary

    def extract_employee_work_patterns(
        self,
        employee_id: str,
        date_from: str,
        date_to: str
    ) -> List[EmployeeWorkPatternData]:
        """Extract weekly work patterns for a specific employee.

        This provides detailed work pattern analysis for individual carbon
        footprint calculations including commuting frequency.

        Args:
            employee_id: Employee identifier
            date_from: Start date (ISO format)
            date_to: End date (ISO format)

        Returns:
            List of weekly work pattern records
        """
        logger.info(f"Extracting work patterns for employee {employee_id}")

        weekly_patterns = {}

        try:
            # Get time entries
            time_entries = list(self.extract_time_entries(
                employee_id=employee_id,
                date_from=date_from,
                date_to=date_to,
                approval_status='Approved'
            ))

            # Get time off
            time_off = list(self.extract_time_off_requests(
                employee_id=employee_id,
                date_from=date_from,
                date_to=date_to,
                approval_status='Approved'
            ))

            # Group by ISO week
            for entry in time_entries:
                work_date = entry.get('WorkDate', '')
                if not work_date:
                    continue

                # Calculate ISO week (simplified - use proper date library in production)
                week_key = work_date[:8] + 'W' + str(int(work_date[8:10]) // 7 + 1)

                if week_key not in weekly_patterns:
                    weekly_patterns[week_key] = {
                        'total_hours': 0.0,
                        'onsite_hours': 0.0,
                        'remote_hours': 0.0,
                        'travel_hours': 0.0,
                        'overtime_hours': 0.0,
                        'days_onsite': set(),
                        'days_remote': set(),
                        'days_travel': set(),
                        'days_off': 0,
                        'primary_location': None,
                        'cost_center': entry.get('CostCenter')
                    }

                pattern = weekly_patterns[week_key]
                hours = entry.get('HoursWorked', 0.0)
                location_type = entry.get('LocationType', 'UNKNOWN')
                time_type = entry.get('TimeType', 'Regular')

                pattern['total_hours'] += hours

                if location_type == 'ONSITE':
                    pattern['onsite_hours'] += hours
                    pattern['days_onsite'].add(work_date)
                elif location_type == 'REMOTE':
                    pattern['remote_hours'] += hours
                    pattern['days_remote'].add(work_date)
                elif location_type == 'TRAVEL':
                    pattern['travel_hours'] += hours
                    pattern['days_travel'].add(work_date)

                if time_type == 'Overtime':
                    pattern['overtime_hours'] += hours

            # Process time off into weekly patterns
            for time_off_entry in time_off:
                # Simplified - distribute days off across weeks
                total_days = time_off_entry.get('TotalDays', 0.0)
                start_date = time_off_entry.get('StartDate', '')
                if start_date:
                    week_key = start_date[:8] + 'W' + str(int(start_date[8:10]) // 7 + 1)
                    if week_key in weekly_patterns:
                        weekly_patterns[week_key]['days_off'] += int(total_days)

            # Convert to EmployeeWorkPatternData objects
            results = []
            for week, pattern in weekly_patterns.items():
                # Determine primary location
                primary_location = 'REMOTE' if pattern['remote_hours'] > pattern['onsite_hours'] else 'ONSITE'

                work_pattern = EmployeeWorkPatternData(
                    EmployeeID=employee_id,
                    WorkWeek=week,
                    TotalHoursWorked=pattern['total_hours'],
                    OnsiteHours=pattern['onsite_hours'],
                    RemoteHours=pattern['remote_hours'],
                    TravelHours=pattern['travel_hours'],
                    OvertimeHours=pattern['overtime_hours'],
                    DaysInOffice=len(pattern['days_onsite']),
                    DaysRemote=len(pattern['days_remote']),
                    DaysTravel=len(pattern['days_travel']),
                    DaysOff=pattern['days_off'],
                    PrimaryLocation=primary_location,
                    CostCenter=pattern['cost_center']
                )
                results.append(work_pattern)

            logger.info(f"Generated {len(results)} weekly work patterns for employee {employee_id}")

        except Exception as e:
            logger.error(f"Error extracting work patterns: {e}", exc_info=True)

        return results

    def extract_commuting_emissions_data(
        self,
        cost_center: Optional[str] = None,
        date_from: str = None,
        date_to: str = None
    ) -> Dict[str, Any]:
        """Extract data optimized for Category 7 commuting emissions calculations.

        This method provides aggregated data specifically for calculating
        employee commuting emissions based on office attendance patterns.

        Args:
            cost_center: Filter by cost center
            date_from: Start date (ISO format)
            date_to: End date (ISO format)

        Returns:
            Dictionary with commuting analysis data
        """
        logger.info(f"Extracting commuting emissions data")

        # Get remote vs onsite summary
        work_location_summary = self.extract_remote_vs_onsite_hours(
            cost_center=cost_center,
            date_from=date_from,
            date_to=date_to
        )

        # Calculate commuting days (onsite days = commuting days)
        commuting_data = {
            **work_location_summary,
            'estimated_commuting_days': 0,
            'estimated_remote_work_days': 0,
            'commuting_emissions_avoided_kg_co2e': 0.0,
            'assumptions': {
                'avg_hours_per_day': 8.0,
                'avg_commute_distance_km': 25.0,  # Configurable
                'avg_commute_emissions_kg_co2e_per_km': 0.171  # Average car emissions
            }
        }

        # Estimate commuting days from onsite hours
        avg_hours_per_day = commuting_data['assumptions']['avg_hours_per_day']
        commuting_data['estimated_commuting_days'] = int(
            work_location_summary.get('onsite_hours', 0.0) / avg_hours_per_day
        )
        commuting_data['estimated_remote_work_days'] = int(
            work_location_summary.get('remote_hours', 0.0) / avg_hours_per_day
        )

        # Calculate emissions avoided by remote work
        remote_days = commuting_data['estimated_remote_work_days']
        avg_distance = commuting_data['assumptions']['avg_commute_distance_km']
        emissions_per_km = commuting_data['assumptions']['avg_commute_emissions_kg_co2e_per_km']

        commuting_data['commuting_emissions_avoided_kg_co2e'] = \
            remote_days * avg_distance * 2 * emissions_per_km  # Round trip

        logger.info(
            f"Commuting analysis: {commuting_data['estimated_commuting_days']} office days, "
            f"{commuting_data['estimated_remote_work_days']} remote days, "
            f"{commuting_data['commuting_emissions_avoided_kg_co2e']:.2f} kg CO2e avoided"
        )

        return commuting_data
