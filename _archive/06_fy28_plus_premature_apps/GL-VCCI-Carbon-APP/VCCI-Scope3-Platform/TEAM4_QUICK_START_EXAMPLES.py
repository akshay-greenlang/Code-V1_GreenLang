# -*- coding: utf-8 -*-
"""
Team 4 ERP Connector Modules - Quick Start Examples

This file provides copy-paste ready code examples for using the new
SAP QM and Workday Time connector modules.

Author: GL-VCCI Team 4
Date: 2025-11-09
"""

# ============================================================================
# SAP QM (Quality Management) - Example Usage
# ============================================================================

def example_sap_qm_extraction():
    """Example: Extract quality inspection data from SAP for scrap tracking."""

    from connectors.sap.client import SAPClient
    from connectors.sap.extractors.qm_extractor import QMExtractor
    from connectors.sap.extractors.base import ExtractionConfig
    from connectors.sap.mappers.quality_inspection_mapper import QualityInspectionMapper

    # Step 1: Initialize SAP client
    client = SAPClient(
        base_url="https://your-sap-server.com/sap/opu/odata/sap",
        username="your_username",
        password="your_password"
    )

    # Step 2: Configure extraction
    config = ExtractionConfig(
        batch_size=1000,
        enable_delta=True,
        last_sync_timestamp="2024-01-01T00:00:00Z"
    )

    # Step 3: Initialize extractor
    extractor = QMExtractor(client, config)

    # Step 4: Extract rejected materials (for waste tracking)
    print("Extracting rejected materials...")
    rejected_lots = []
    for lot in extractor.extract_rejected_materials_data(
        plant="1000",
        date_from="2024-01-01",
        date_to="2024-12-31"
    ):
        rejected_lots.append(lot)
        print(f"  Rejected: {lot['Material']} - {lot['InspectionLotRejectedQuantity']} {lot['InspectionLotQuantityUnit']}")

    print(f"Total rejected lots: {len(rejected_lots)}")

    # Step 5: Get scrap summary for emissions calculation
    print("\nCalculating scrap emissions summary...")
    summary = extractor.extract_quality_scrap_emissions_data(
        plant="1000",
        date_from="2024-01-01",
        date_to="2024-12-31"
    )

    print(f"Plant: {summary['plant']}")
    print(f"Total inspections: {summary['total_inspections']}")
    print(f"Total rejected lots: {summary['total_rejected_lots']}")
    print(f"Total scrap quantity: {summary['total_scrap_quantity']:.2f}")
    print(f"Rejection rate: {summary['rejection_rate_percent']:.2f}%")
    print(f"\nScrap by origin:")
    for origin, qty in summary['scrap_by_origin'].items():
        print(f"  {origin}: {qty:.2f}")

    return rejected_lots, summary


def example_sap_qm_mapping():
    """Example: Map SAP QM data to VCCI waste schema."""

    from connectors.sap.mappers.quality_inspection_mapper import QualityInspectionMapper

    # Step 1: Initialize mapper
    mapper = QualityInspectionMapper(tenant_id="ACME_CORP")

    # Step 2: Prepare sample data
    inspection_lot = {
        "InspectionLot": "1000001",
        "Material": "MAT-12345",
        "MaterialName": "Steel Sheet 2mm",
        "Plant": "1000",
        "InspectionLotOrigin": "04",  # Production
        "InspectionLotRejectedQuantity": 50.0,
        "InspectionLotScrapQuantity": 10.0,
        "InspectionLotQuantityUnit": "KG",
        "InspectionLotEndDate": "2024-06-15",
        "InspLotUsageDecisionCode": "R",  # Rejected
        "Supplier": None,
        "ManufacturingOrder": "MO-12345",
    }

    facility_master = {
        "PlantName": "Main Manufacturing Plant",
        "Country": "US",
    }

    # Step 3: Map to waste record
    print("Mapping inspection lot to waste record...")
    waste_record = mapper.map_inspection_lot(
        inspection_lot=inspection_lot,
        facility_master=facility_master
    )

    if waste_record:
        print(f"\nWaste Record Created:")
        print(f"  ID: {waste_record.waste_transaction_id}")
        print(f"  Material: {waste_record.material_name}")
        print(f"  Quantity: {waste_record.waste_quantity} {waste_record.unit}")
        print(f"  Category: {waste_record.waste_category}")
        print(f"  Type: {waste_record.waste_type}")
        print(f"  Disposal: {waste_record.disposal_method}")
        print(f"  Emissions: {waste_record.emissions_kg_co2e} kg CO2e")
    else:
        print("No waste generated (lot was accepted)")

    # Step 4: Map batch of inspection lots
    inspection_lots = [inspection_lot]  # Add more lots here
    plant_lookup = {"1000": facility_master}

    print("\nMapping batch of inspection lots...")
    waste_records = mapper.map_batch(
        inspection_lots=inspection_lots,
        facility_lookup=plant_lookup
    )
    print(f"Mapped {len(waste_records)} waste records")

    # Step 5: Calculate aggregate statistics
    if waste_records:
        print("\nCalculating waste statistics...")
        stats = mapper.calculate_total_waste_emissions(waste_records)

        print(f"Total waste: {stats['total_waste_quantity']:.2f}")
        print(f"Total emissions: {stats['total_emissions_kg_co2e']:.2f} kg CO2e")
        print(f"\nWaste by category:")
        for category, qty in stats['waste_by_category'].items():
            print(f"  {category}: {qty:.2f}")
        print(f"\nTop waste materials:")
        for material, data in list(stats['top_waste_materials'].items())[:5]:
            print(f"  {material}: {data['waste_quantity']:.2f} ({data['emissions_kg_co2e']:.2f} kg CO2e)")

    return waste_records, stats


# ============================================================================
# Workday Time Tracking - Example Usage
# ============================================================================

def example_workday_time_extraction():
    """Example: Extract time tracking data from Workday for commute analysis."""

    from connectors.workday.client import WorkdayClient
    from connectors.workday.extractors.time_extractor import TimeExtractor
    from connectors.workday.extractors.base import ExtractionConfig

    # Step 1: Initialize Workday client
    client = WorkdayClient(
        tenant_url="https://impl.workday.com/acme/services",
        username="your_username",
        password="your_password"
    )

    # Step 2: Configure extraction
    config = ExtractionConfig(
        batch_size=1000,
        enable_delta=True,
        last_sync_timestamp="2024-01-01T00:00:00Z"
    )

    # Step 3: Initialize extractor
    extractor = TimeExtractor(client, config)

    # Step 4: Extract remote vs onsite hours
    print("Extracting remote vs onsite work summary...")
    summary = extractor.extract_remote_vs_onsite_hours(
        cost_center="CC-1000",
        date_from="2024-01-01",
        date_to="2024-12-31"
    )

    print(f"\nWork Location Summary:")
    print(f"Period: {summary['period_from']} to {summary['period_to']}")
    print(f"Cost Center: {summary['cost_center']}")
    print(f"Total hours: {summary['total_hours']:.1f}")
    print(f"Onsite hours: {summary['onsite_hours']:.1f} ({summary['onsite_percentage']:.1f}%)")
    print(f"Remote hours: {summary['remote_hours']:.1f} ({summary['remote_percentage']:.1f}%)")
    print(f"Travel hours: {summary['travel_hours']:.1f}")
    print(f"Overtime hours: {summary['overtime_hours']:.1f}")
    print(f"Total employees: {summary['total_employees']}")

    # Step 5: Extract commuting emissions data (Category 7)
    print("\n\nExtracting commuting emissions data...")
    commute_data = extractor.extract_commuting_emissions_data(
        cost_center="CC-1000",
        date_from="2024-01-01",
        date_to="2024-12-31"
    )

    print(f"\nCommuting Analysis:")
    print(f"Estimated commuting days: {commute_data['estimated_commuting_days']}")
    print(f"Estimated remote work days: {commute_data['estimated_remote_work_days']}")
    print(f"Emissions avoided by remote work: {commute_data['commuting_emissions_avoided_kg_co2e']:.2f} kg CO2e")
    print(f"\nAssumptions:")
    print(f"  Avg commute distance: {commute_data['assumptions']['avg_commute_distance_km']} km")
    print(f"  Avg emissions per km: {commute_data['assumptions']['avg_commute_emissions_kg_co2e_per_km']} kg CO2e/km")

    # Step 6: Extract employee work patterns
    print("\n\nExtracting employee work patterns...")
    patterns = extractor.extract_employee_work_patterns(
        employee_id="EMP-001",
        date_from="2024-01-01",
        date_to="2024-03-31"
    )

    print(f"\nEmployee Work Patterns (showing first 5 weeks):")
    for pattern in patterns[:5]:
        print(f"Week {pattern.WorkWeek}:")
        print(f"  Total hours: {pattern.TotalHoursWorked:.1f}")
        print(f"  Onsite: {pattern.DaysInOffice} days ({pattern.OnsiteHours:.1f} hrs)")
        print(f"  Remote: {pattern.DaysRemote} days ({pattern.RemoteHours:.1f} hrs)")
        print(f"  Primary location: {pattern.PrimaryLocation}")

    return summary, commute_data, patterns


def example_workday_time_mapping():
    """Example: Map Workday time data to VCCI commute schema."""

    from connectors.workday.mappers.time_entry_mapper import TimeEntryMapper

    # Step 1: Initialize mapper with tenant-specific defaults
    mapper = TimeEntryMapper(
        tenant_id="ACME_CORP",
        default_commute_distance_km=30.0,  # Company average
        default_commute_mode="Car"
    )

    # Step 2: Prepare sample time entries
    time_entries = [
        {
            "TimeEntryID": "TE-001",
            "EmployeeID": "EMP-001",
            "EmployeeName": "John Smith",
            "WorkDate": "2024-06-01",
            "TimeType": "Regular",
            "HoursWorked": 8.0,
            "LocationType": "ONSITE",
            "WorkLocation": "Main Office",
            "CostCenter": "CC-1000",
            "ApprovalStatus": "Approved"
        },
        {
            "TimeEntryID": "TE-002",
            "EmployeeID": "EMP-001",
            "EmployeeName": "John Smith",
            "WorkDate": "2024-06-02",
            "TimeType": "Regular",
            "HoursWorked": 8.0,
            "LocationType": "REMOTE",
            "WorkLocation": "Home",
            "CostCenter": "CC-1000",
            "ApprovalStatus": "Approved"
        }
    ]

    # Step 3: Prepare employee master data (optional)
    employee_lookup = {
        "EMP-001": {
            "EmployeeName": "John Smith",
            "Department": "Engineering",
            "CostCenter": "CC-1000",
            "City": "San Francisco"
        }
    }

    # Step 4: Prepare employee-specific commute profiles (optional)
    commute_profile_lookup = {
        "EMP-001": {
            "CommuteDistanceKm": 35.0,
            "CommuteMode": "Car - Hybrid"
        }
    }

    # Step 5: Map time entries to commute records
    print("Mapping time entries to commute records...")
    commute_records = mapper.map_batch(
        time_entries=time_entries,
        employee_lookup=employee_lookup,
        commute_profile_lookup=commute_profile_lookup
    )

    print(f"\nMapped {len(commute_records)} commute records:")
    for record in commute_records:
        print(f"\n{record.date} - {record.employee_name}:")
        print(f"  Type: {record.commute_type}")
        print(f"  Commute occurred: {record.commute_occurred}")
        print(f"  Hours: {record.hours_at_location}")
        if record.commute_occurred:
            print(f"  Mode: {record.commute_mode}")
            print(f"  Distance: {record.commute_distance_km} km")
            print(f"  Emissions: {record.emissions_kg_co2e} kg CO2e")
        else:
            print(f"  Remote work - Emissions avoided: {record.emissions_avoided_kg_co2e} kg CO2e")

    # Step 6: Calculate aggregate statistics
    print("\n\nCalculating commute statistics...")
    stats = mapper.calculate_commute_statistics(commute_records)

    print(f"\nCommute Statistics:")
    print(f"Total work days: {stats['total_work_days']}")
    print(f"Commute days: {stats['total_commute_days']}")
    print(f"Remote days: {stats['total_remote_days']}")
    print(f"Remote work percentage: {stats['remote_work_percentage']:.1f}%")
    print(f"\nEmissions:")
    print(f"  Total emissions: {stats['total_emissions_kg_co2e']:.2f} kg CO2e")
    print(f"  Total avoided: {stats['total_emissions_avoided_kg_co2e']:.2f} kg CO2e")
    print(f"  Net emissions: {stats['net_emissions_kg_co2e']:.2f} kg CO2e")
    print(f"\nAverages:")
    print(f"  Per commute day: {stats['average_emissions_per_commute_day']:.2f} kg CO2e")
    print(f"  Per remote day (avoided): {stats['average_emissions_avoided_per_remote_day']:.2f} kg CO2e")

    print(f"\nEmissions by commute type:")
    for ctype, data in stats['emissions_by_commute_type'].items():
        print(f"  {ctype}: {data['count']} days, {data['emissions']:.2f} kg CO2e")

    print(f"\nEmissions by employee:")
    for emp_id, data in list(stats['emissions_by_employee'].items())[:5]:
        print(f"  {data['employee_name']}:")
        print(f"    Commute: {data['commute_days']} days, {data['emissions']:.2f} kg CO2e")
        print(f"    Remote: {data['remote_days']} days, {data['avoided']:.2f} kg CO2e avoided")

    return commute_records, stats


# ============================================================================
# Combined Example: Full Pipeline
# ============================================================================

def example_full_pipeline():
    """Example: Complete data pipeline from extraction to emissions calculation."""

    print("=" * 80)
    print("SAP QM MODULE - FULL PIPELINE")
    print("=" * 80)

    # SAP QM: Extract and map quality scrap data
    rejected_lots, scrap_summary = example_sap_qm_extraction()
    waste_records, waste_stats = example_sap_qm_mapping()

    print("\n" + "=" * 80)
    print("WORKDAY TIME MODULE - FULL PIPELINE")
    print("=" * 80)

    # Workday Time: Extract and map commuting data
    work_summary, commute_data, patterns = example_workday_time_extraction()
    commute_records, commute_stats = example_workday_time_mapping()

    print("\n" + "=" * 80)
    print("COMBINED EMISSIONS SUMMARY")
    print("=" * 80)

    # Calculate total Scope 3 emissions impact
    total_waste_emissions = waste_stats.get('total_emissions_kg_co2e', 0.0) if waste_stats else 0.0
    total_commute_emissions = commute_stats.get('total_emissions_kg_co2e', 0.0) if commute_stats else 0.0
    total_emissions_avoided = commute_stats.get('total_emissions_avoided_kg_co2e', 0.0) if commute_stats else 0.0

    print(f"\nScope 3 Emissions Summary:")
    print(f"  Category 1 (Quality Scrap/Waste): {total_waste_emissions:.2f} kg CO2e")
    print(f"  Category 7 (Commuting): {total_commute_emissions:.2f} kg CO2e")
    print(f"  Category 7 (Avoided by Remote Work): -{total_emissions_avoided:.2f} kg CO2e")
    print(f"\nNet Scope 3 Emissions: {total_waste_emissions + total_commute_emissions - total_emissions_avoided:.2f} kg CO2e")

    return {
        'waste_emissions': total_waste_emissions,
        'commute_emissions': total_commute_emissions,
        'emissions_avoided': total_emissions_avoided,
        'net_emissions': total_waste_emissions + total_commute_emissions - total_emissions_avoided
    }


# ============================================================================
# Advanced Examples
# ============================================================================

def example_advanced_sap_qm():
    """Advanced example: Extract specific inspection lot with all details."""

    from connectors.sap.extractors.qm_extractor import QMExtractor
    from connectors.sap.client import SAPClient

    client = SAPClient("https://sap-server.com", "user", "pass")
    extractor = QMExtractor(client)

    # Get complete inspection lot with results and characteristics
    inspection_lot_number = "1000001"
    print(f"Extracting complete inspection lot: {inspection_lot_number}")

    complete_lot = extractor.extract_inspection_lot_with_details(inspection_lot_number)

    print(f"\nInspection Lot Header:")
    print(f"  Material: {complete_lot['header']['Material']}")
    print(f"  Quantity: {complete_lot['header']['InspectionLotQuantity']}")
    print(f"  Rejected: {complete_lot['header']['InspectionLotRejectedQuantity']}")

    print(f"\nInspection Results: {len(complete_lot['results'])} characteristics")
    for result in complete_lot['results'][:3]:  # Show first 3
        print(f"  {result['InspectionCharacteristic']}: {result['InspectionValuationResult']}")

    print(f"\nInspection Characteristics: {len(complete_lot['characteristics'])}")

    return complete_lot


def example_advanced_workday_time():
    """Advanced example: Analyze work patterns for emissions forecasting."""

    from connectors.workday.extractors.time_extractor import TimeExtractor
    from connectors.workday.client import WorkdayClient

    client = WorkdayClient("https://workday.com/acme", "user", "pass")
    extractor = TimeExtractor(client)

    # Analyze multiple employees for department-level insights
    employee_ids = ["EMP-001", "EMP-002", "EMP-003"]

    all_patterns = []
    for emp_id in employee_ids:
        patterns = extractor.extract_employee_work_patterns(
            employee_id=emp_id,
            date_from="2024-01-01",
            date_to="2024-12-31"
        )
        all_patterns.extend(patterns)

    # Calculate department averages
    total_weeks = len(all_patterns)
    avg_remote_days = sum(p.DaysRemote for p in all_patterns) / total_weeks if total_weeks > 0 else 0
    avg_onsite_days = sum(p.DaysInOffice for p in all_patterns) / total_weeks if total_weeks > 0 else 0

    print(f"\nDepartment Work Pattern Analysis:")
    print(f"Total employee-weeks: {total_weeks}")
    print(f"Average remote days per week: {avg_remote_days:.1f}")
    print(f"Average onsite days per week: {avg_onsite_days:.1f}")
    print(f"Remote work adoption: {(avg_remote_days / (avg_remote_days + avg_onsite_days) * 100):.1f}%")

    return all_patterns


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("""
    ========================================================================
    Team 4 ERP Connector Modules - Quick Start Examples
    ========================================================================

    This script demonstrates the usage of the new SAP QM and Workday Time
    connector modules for Scope 3 emissions tracking.

    Note: This is example code. Update credentials and endpoints before running.
    ========================================================================
    """)

    # Uncomment the examples you want to run:

    # SAP QM Examples
    # example_sap_qm_extraction()
    # example_sap_qm_mapping()
    # example_advanced_sap_qm()

    # Workday Time Examples
    # example_workday_time_extraction()
    # example_workday_time_mapping()
    # example_advanced_workday_time()

    # Full Pipeline
    # example_full_pipeline()

    print("\n✅ Examples loaded. Uncomment function calls in main() to run.")
