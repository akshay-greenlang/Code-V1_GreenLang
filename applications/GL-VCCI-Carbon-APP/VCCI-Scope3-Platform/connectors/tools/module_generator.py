# -*- coding: utf-8 -*-
"""
ERP Connector Module Generator

Automated code generation for ERP connector modules (extractors, mappers, tests).

This tool generates boilerplate code for new ERP connector modules following
the established architecture patterns, significantly accelerating development.

Usage:
    python module_generator.py --erp sap --module wm --name "Warehouse Management"

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from greenlang.determinism import DeterministicClock


# Module specifications database
MODULE_SPECS = {
    "sap": {
        "wm": {
            "name": "Warehouse Management",
            "code": "WM",
            "service": "API_WHSE_ORDER_SRV",
            "entity_sets": ["A_WarehouseOrder", "A_WarehouseTask", "A_WarehouseStorageBin"],
            "primary_entity": "A_WarehouseOrder",
            "key_field": "WarehouseOrder",
            "changed_field": "ChangedOn",
            "schema": "logistics_v1.0.json",
            "use_cases": ["Category 4: Warehouse-to-warehouse transfers", "Material handling equipment emissions"],
            "carbon_impact": "MEDIUM"
        },
        "co": {
            "name": "Controlling",
            "code": "CO",
            "service": "API_CONTROLLINGAREA_SRV",
            "entity_sets": ["A_CostCenter", "A_CostElement", "A_InternalOrder", "A_ActivityType"],
            "primary_entity": "A_CostCenter",
            "key_field": "CostCenter",
            "changed_field": "ChangedOn",
            "schema": "financials_v1.0.json",
            "use_cases": ["Cost center emissions allocation", "Activity-based costing for carbon"],
            "carbon_impact": "HIGH"
        },
        "qm": {
            "name": "Quality Management",
            "code": "QM",
            "service": "API_QUALITY_MANAGEMENT_SRV",
            "entity_sets": ["A_InspectionLot", "A_InspectionOperation", "A_UsageDecision"],
            "primary_entity": "A_InspectionLot",
            "key_field": "InspectionLot",
            "changed_field": "ChangedOn",
            "schema": "quality_v1.0.json",
            "use_cases": ["Quality-related rework emissions", "Scrap and waste tracking"],
            "carbon_impact": "LOW"
        },
        "pm": {
            "name": "Plant Maintenance",
            "code": "PM",
            "service": "API_MAINTENANCE_ORDER_SRV",
            "entity_sets": ["A_MaintenanceOrder", "A_MaintenanceOrderOperation", "A_MaintenanceItem"],
            "primary_entity": "A_MaintenanceOrder",
            "key_field": "MaintenanceOrder",
            "changed_field": "ChangedOn",
            "schema": "asset_management_v1.0.json",
            "use_cases": ["Equipment energy consumption", "Maintenance-related emissions"],
            "carbon_impact": "MEDIUM"
        },
        "ps": {
            "name": "Project System",
            "code": "PS",
            "service": "API_PROJECT_SRV",
            "entity_sets": ["A_Project", "A_WBSElement", "A_Network", "A_NetworkActivity"],
            "primary_entity": "A_Project",
            "key_field": "Project",
            "changed_field": "ChangedOn",
            "schema": "project_v1.0.json",
            "use_cases": ["Project-level carbon accounting", "Capital project emissions"],
            "carbon_impact": "MEDIUM"
        }
    },
    "oracle": {
        "manufacturing": {
            "name": "Manufacturing",
            "code": "MFG",
            "service": "/fscmRestApi/resources/11.13.18.05/workOrders",
            "entity_sets": ["workOrders", "workOrderOperations", "workOrderMaterials"],
            "primary_entity": "workOrders",
            "key_field": "WorkOrderId",
            "changed_field": "LastUpdateDate",
            "schema": "manufacturing_v1.0.json",
            "use_cases": ["Production emissions tracking", "Manufacturing energy consumption"],
            "carbon_impact": "HIGH"
        },
        "inventory": {
            "name": "Inventory Management",
            "code": "INV",
            "service": "/fscmRestApi/resources/11.13.18.05/materialTransactions",
            "entity_sets": ["materialTransactions", "onhandQuantities", "lots"],
            "primary_entity": "materialTransactions",
            "key_field": "TransactionId",
            "changed_field": "LastUpdateDate",
            "schema": "logistics_v1.0.json",
            "use_cases": ["Stock movement emissions", "Inventory holding emissions"],
            "carbon_impact": "MEDIUM"
        },
        "projects": {
            "name": "Projects",
            "code": "PRJ",
            "service": "/fscmRestApi/resources/11.13.18.05/projects",
            "entity_sets": ["projects", "projectTasks", "projectCosts"],
            "primary_entity": "projects",
            "key_field": "ProjectId",
            "changed_field": "LastUpdateDate",
            "schema": "project_v1.0.json",
            "use_cases": ["Project carbon footprint", "Capital expenditure emissions"],
            "carbon_impact": "MEDIUM"
        }
    },
    "workday": {
        "recruiting": {
            "name": "Recruiting",
            "code": "REC",
            "service": "Recruiting",
            "entity_sets": ["Candidates", "JobApplications", "Interviews"],
            "primary_entity": "Candidates",
            "key_field": "CandidateID",
            "changed_field": "LastModified",
            "schema": "hr_v1.0.json",
            "use_cases": ["Candidate travel emissions", "Recruitment event emissions"],
            "carbon_impact": "LOW"
        },
        "time_tracking": {
            "name": "Time Tracking",
            "code": "TIME",
            "service": "Time_Tracking",
            "entity_sets": ["TimeEntries", "TimeOffRequests", "TimeBlocks"],
            "primary_entity": "TimeEntries",
            "key_field": "TimeEntryID",
            "changed_field": "LastModified",
            "schema": "hr_v1.0.json",
            "use_cases": ["Work-from-home emissions", "Employee commute tracking"],
            "carbon_impact": "MEDIUM"
        }
    }
}


class ModuleGenerator:
    """Generates ERP connector module code."""

    def __init__(self, erp: str, module_key: str, output_dir: str):
        """Initialize module generator.

        Args:
            erp: ERP system (sap, oracle, workday)
            module_key: Module key (e.g., wm, co, manufacturing)
            output_dir: Output directory path
        """
        self.erp = erp.lower()
        self.module_key = module_key.lower()
        self.output_dir = Path(output_dir)

        if self.erp not in MODULE_SPECS:
            raise ValueError(f"Unknown ERP: {self.erp}")
        if self.module_key not in MODULE_SPECS[self.erp]:
            raise ValueError(f"Unknown module: {self.module_key} for ERP: {self.erp}")

        self.spec = MODULE_SPECS[self.erp][self.module_key]

    def generate_extractor(self) -> str:
        """Generate extractor code."""
        template = f'''"""
{self.spec['name']} Extractor

Extracts data from {self.erp.upper()} {self.spec['name']} module.

Service: {self.spec['service']}
Entity Sets: {', '.join(self.spec['entity_sets'])}

Use Cases:
{chr(10).join(f'    - {uc}' for uc in self.spec['use_cases'])}

Carbon Impact: {self.spec['carbon_impact']}

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
Generated: {DeterministicClock.now().strftime('%Y-%m-%d')}
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionConfig

logger = logging.getLogger(__name__)


class {self.spec['code']}Data(BaseModel):
    """{self.spec['name']} primary data model.

    Maps to {self.spec['primary_entity']} entity.
    """
    {self.spec['key_field']}: str
    # Add additional fields based on API documentation
    {self.spec['changed_field']}: Optional[str] = None  # For delta extraction


class {self.spec['code']}Extractor(BaseExtractor):
    """{self.spec['name']} Extractor.

    Extracts {self.spec['name'].lower()} data from {self.erp.upper()}.
    """

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        """Initialize {self.spec['code']} extractor.

        Args:
            client: {self.erp.upper()} client instance
            config: Extraction configuration
        """
        super().__init__(client, config)
        self.service_name = "{self.spec['code']}"
        self._current_entity_set = "{self.spec['primary_entity']}"

    def get_entity_set_name(self) -> str:
        """Get current entity set name."""
        return self._current_entity_set

    def get_changed_on_field(self) -> str:
        """Get field name for delta extraction."""
        return "{self.spec['changed_field']}"

    def extract_primary_data(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        **filters
    ) -> Iterator[Dict[str, Any]]:
        """Extract primary {self.spec['name'].lower()} data.

        Args:
            date_from: Filter by date from (ISO format)
            date_to: Filter by date to (ISO format)
            **filters: Additional filters

        Yields:
            {self.spec['name']} records as dictionaries
        """
        self._current_entity_set = "{self.spec['primary_entity']}"

        additional_filters = []

        if date_from:
            additional_filters.append(f"Date ge datetime'{{date_from}}'")
        if date_to:
            additional_filters.append(f"Date le datetime'{{date_to}}'")

        logger.info(f"Extracting {self.spec['name']} data with filters: {{additional_filters}}")

        yield from self.get_all(
            additional_filters=additional_filters if additional_filters else None
        )

    # Add additional extraction methods for other entity sets
'''
        return template

    def generate_mapper(self) -> str:
        """Generate mapper code."""
        template = f'''"""
{self.spec['name']} Mapper

Maps {self.erp.upper()} {self.spec['name']} data to VCCI {self.spec['schema']} schema.

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
Generated: {DeterministicClock.now().strftime('%Y-%m-%d')}
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class {self.spec['code']}Record(BaseModel):
    """VCCI {self.spec['name']} data model matching {self.spec['schema']} schema."""

    id: str
    # Add schema fields based on target schema
    tenant_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class {self.spec['code']}Mapper:
    """Maps {self.erp.upper()} {self.spec['name']} data to VCCI schema."""

    def __init__(self, tenant_id: Optional[str] = None):
        """Initialize {self.spec['code']} mapper.

        Args:
            tenant_id: Tenant identifier for multi-tenant deployment
        """
        self.tenant_id = tenant_id
        logger.info(f"Initialized {self.spec['code']}Mapper for tenant: {{tenant_id}}")

    def map_record(
        self,
        source_data: Dict[str, Any],
        master_data: Optional[Dict[str, Any]] = None
    ) -> {self.spec['code']}Record:
        """Map {self.erp.upper()} record to VCCI schema.

        Args:
            source_data: Source {self.erp.upper()} data
            master_data: Optional master data for enrichment

        Returns:
            {self.spec['code']}Record matching {self.spec['schema']} schema

        Raises:
            ValueError: If required fields are missing
        """
        # Required fields validation
        if not source_data.get("{self.spec['key_field']}"):
            raise ValueError("Missing required field: {self.spec['key_field']}")

        # Generate ID
        record_id = f"{self.spec['code']}-{{source_data['{self.spec['key_field']}]}}"

        # Metadata
        metadata = {{
            "source_system": "{self.erp.upper()}_{self.spec['code']}",
            "source_document_id": source_data.get("{self.spec['key_field']}"),
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_status": "Validated",
            "created_by": "{self.erp.lower()}-{self.module_key.lower()}-extractor",
        }}

        # Custom fields
        custom_fields = {{
            "{self.spec['changed_field'].lower()}": source_data.get("{self.spec['changed_field']}"),
            # Add additional custom fields
        }}

        # Build record
        record = {self.spec['code']}Record(
            id=record_id,
            tenant_id=self.tenant_id,
            metadata=metadata,
            custom_fields=custom_fields,
        )

        logger.debug(f"Mapped {{record_id}}")
        return record

    def map_batch(
        self,
        records: List[Dict[str, Any]],
        master_lookup: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[{self.spec['code']}Record]:
        """Map a batch of records.

        Args:
            records: List of source records
            master_lookup: Optional master data lookup

        Returns:
            List of mapped records
        """
        mapped = []
        master_lookup = master_lookup or {{}}

        for record in records:
            try:
                key = record.get("{self.spec['key_field']}")
                master_data = master_lookup.get(key) if key else None

                mapped_record = self.map_record(record, master_data)
                mapped.append(mapped_record)

            except Exception as e:
                logger.error(f"Error mapping record: {{e}}", exc_info=True)
                continue

        logger.info(f"Mapped {{len(mapped)}} of {{len(records)}} records")
        return mapped
'''
        return template

    def generate_tests(self) -> str:
        """Generate test code."""
        template = f'''"""
Tests for {self.erp.upper()} {self.spec['name']} Extractor

Author: GL-VCCI Team 4 - ERP Integration Expansion
Version: 1.0.0
Generated: {DeterministicClock.now().strftime('%Y-%m-%d')}
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from connectors.{self.erp}.extractors.{self.module_key}_extractor import (
    {self.spec['code']}Extractor,
    {self.spec['code']}Data
)
from connectors.{self.erp}.extractors.base import ExtractionConfig


@pytest.fixture
def mock_client():
    """Create a mock {self.erp.upper()} client."""
    client = Mock()
    client.get = MagicMock()
    client.get_by_key = MagicMock()
    return client


@pytest.fixture
def extractor(mock_client):
    """Create extractor instance."""
    config = ExtractionConfig(batch_size=100, enable_delta=False)
    return {self.spec['code']}Extractor(mock_client, config)


@pytest.fixture
def sample_data():
    """Sample {self.spec['name'].lower()} data."""
    return {{
        "{self.spec['key_field']}": "TEST-001",
        "{self.spec['changed_field']}": "2024-01-20T10:00:00Z"
    }}


class Test{self.spec['code']}Extractor:
    """Tests for {self.spec['code']} Extractor."""

    def test_initialization(self, mock_client):
        """Test extractor initialization."""
        extractor = {self.spec['code']}Extractor(mock_client)
        assert extractor.service_name == "{self.spec['code']}"
        assert extractor.client == mock_client
        assert extractor.get_entity_set_name() == "{self.spec['primary_entity']}"
        assert extractor.get_changed_on_field() == "{self.spec['changed_field']}"

    def test_extract_primary_data(self, extractor, mock_client, sample_data):
        """Test extracting primary data."""
        mock_client.get.return_value = {{
            "value": [sample_data]
        }}

        results = list(extractor.extract_primary_data())

        assert len(results) == 1
        assert results[0]["{self.spec['key_field']}"] == "TEST-001"

    def test_delta_extraction(self, extractor, mock_client, sample_data):
        """Test delta extraction."""
        extractor.config.enable_delta = True
        mock_client.get.return_value = {{"value": [sample_data]}}

        results = list(extractor.get_delta(last_sync_timestamp="2024-01-19T00:00:00Z"))
        assert len(results) == 1

    def test_error_handling(self, extractor, mock_client):
        """Test error handling."""
        mock_client.get.side_effect = Exception("Connection error")

        with pytest.raises(Exception) as exc_info:
            list(extractor.extract_primary_data())

        assert "Connection error" in str(exc_info.value)


class Test{self.spec['code']}Data:
    """Tests for {self.spec['code']} data models."""

    def test_data_model(self, sample_data):
        """Test data model."""
        data = {self.spec['code']}Data(**sample_data)
        assert data.{self.spec['key_field']} == "TEST-001"
'''
        return template

    def generate_all(self):
        """Generate all module files."""
        # Create directories
        extractor_dir = self.output_dir / self.erp / "extractors"
        mapper_dir = self.output_dir / self.erp / "mappers"
        test_dir = self.output_dir / self.erp / "tests"

        extractor_dir.mkdir(parents=True, exist_ok=True)
        mapper_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Generate files
        files_created = []

        # Extractor
        extractor_path = extractor_dir / f"{self.module_key}_extractor.py"
        extractor_path.write_text(self.generate_extractor())
        files_created.append(str(extractor_path))

        # Mapper
        mapper_path = mapper_dir / f"{self.module_key}_mapper.py"
        mapper_path.write_text(self.generate_mapper())
        files_created.append(str(mapper_path))

        # Tests
        test_path = test_dir / f"test_{self.module_key}_extractor.py"
        test_path.write_text(self.generate_tests())
        files_created.append(str(test_path))

        return files_created


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate ERP connector module")
    parser.add_argument("--erp", required=True, choices=["sap", "oracle", "workday"],
                       help="ERP system")
    parser.add_argument("--module", required=True, help="Module key (e.g., wm, co)")
    parser.add_argument("--output", default=".", help="Output directory")

    args = parser.parse_args()

    try:
        generator = ModuleGenerator(args.erp, args.module, args.output)
        files = generator.generate_all()

        print(f"✅ Successfully generated {len(files)} files:")
        for file_path in files:
            print(f"   - {file_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
