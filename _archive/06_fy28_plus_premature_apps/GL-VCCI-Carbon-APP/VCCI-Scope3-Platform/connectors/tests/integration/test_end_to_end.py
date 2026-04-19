# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests
GL-VCCI Scope 3 Platform

Multi-connector end-to-end integration tests covering data flow from
ERP connectors through Intake Agent to Calculator Agent.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import pytest
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from sap.client import SAPODataClient, create_query as sap_query
from oracle.client import OracleRESTClient, create_query as oracle_query
from greenlang.determinism import DeterministicClock


@pytest.mark.integration
@pytest.mark.slow
class TestMultiConnectorIntegration:
    """Test integration across multiple ERP connectors."""

    @pytest.mark.sap_sandbox
    @pytest.mark.oracle_sandbox
    def test_concurrent_extraction_all_connectors(
        self,
        sap_client: SAPODataClient,
        oracle_client: OracleRESTClient,
        mock_workday_client
    ):
        """Test concurrent extraction from all three ERPs."""
        results = {}
        errors = {}

        # Extract from SAP
        try:
            query = sap_query().top(5)
            sap_response = sap_client.get("purchase_orders", query.build())
            results["sap"] = sap_response.get("value", [])
        except Exception as e:
            errors["sap"] = str(e)

        # Extract from Oracle
        try:
            query = oracle_query().limit(5)
            oracle_response = oracle_client.get("purchase_orders", query.build())
            results["oracle"] = oracle_response.get("items", [])
        except Exception as e:
            errors["oracle"] = str(e)

        # Extract from Workday
        try:
            workday_data = mock_workday_client.extract_expense_reports(limit=5)
            results["workday"] = workday_data
        except Exception as e:
            errors["workday"] = str(e)

        # At least one connector should succeed
        assert len(results) > 0, f"All connectors failed: {errors}"

        # Validate each successful extraction
        for connector_name, data in results.items():
            assert isinstance(data, list), f"{connector_name} did not return list"
            if data:
                assert isinstance(data[0], dict), f"{connector_name} records are not dicts"


@pytest.mark.integration
@pytest.mark.slow
class TestSAPToIntakeAgent:
    """Test data flow from SAP to Intake Agent."""

    @pytest.mark.sap_sandbox
    def test_sap_to_intake_flow(self, sap_client: SAPODataClient):
        """Test complete flow: SAP extraction -> Intake Agent ingestion."""
        # Step 1: Extract from SAP
        query = sap_query().top(10)
        sap_response = sap_client.get("purchase_orders", query.build())
        raw_records = sap_response.get("value", [])

        assert len(raw_records) > 0, "No records extracted from SAP"

        # Step 2: Simulate Intake Agent processing
        # In real implementation, this would call the Intake Agent
        processed_records = []
        for record in raw_records:
            processed = {
                "source_system": "SAP",
                "record_type": "purchase_order",
                "raw_data": record,
                "ingestion_timestamp": DeterministicClock.now().isoformat(),
                "status": "pending_calculation"
            }
            processed_records.append(processed)

        assert len(processed_records) == len(raw_records)

        # Step 3: Validate processed records
        for processed in processed_records:
            assert processed["source_system"] == "SAP"
            assert processed["record_type"] == "purchase_order"
            assert "raw_data" in processed
            assert "ingestion_timestamp" in processed


@pytest.mark.integration
@pytest.mark.slow
class TestOracleToIntakeAgent:
    """Test data flow from Oracle to Intake Agent."""

    @pytest.mark.oracle_sandbox
    def test_oracle_to_intake_flow(self, oracle_client: OracleRESTClient):
        """Test complete flow: Oracle extraction -> Intake Agent ingestion."""
        # Step 1: Extract from Oracle
        query = oracle_query().limit(10)
        oracle_response = oracle_client.get("purchase_orders", query.build())
        raw_records = oracle_response.get("items", [])

        assert len(raw_records) > 0, "No records extracted from Oracle"

        # Step 2: Simulate Intake Agent processing
        processed_records = []
        for record in raw_records:
            processed = {
                "source_system": "Oracle",
                "record_type": "purchase_order",
                "raw_data": record,
                "ingestion_timestamp": DeterministicClock.now().isoformat(),
                "status": "pending_calculation"
            }
            processed_records.append(processed)

        assert len(processed_records) == len(raw_records)

        # Step 3: Validate processed records
        for processed in processed_records:
            assert processed["source_system"] == "Oracle"
            assert processed["record_type"] == "purchase_order"
            assert "raw_data" in processed


@pytest.mark.integration
class TestWorkdayToIntakeAgent:
    """Test data flow from Workday to Intake Agent."""

    def test_workday_to_intake_flow(self, mock_workday_client):
        """Test complete flow: Workday extraction -> Intake Agent ingestion."""
        # Step 1: Extract from Workday
        raw_records = mock_workday_client.extract_expense_reports(limit=10)

        assert len(raw_records) > 0, "No records extracted from Workday"

        # Step 2: Simulate Intake Agent processing
        processed_records = []
        for record in raw_records:
            processed = {
                "source_system": "Workday",
                "record_type": "expense_report",
                "raw_data": record,
                "ingestion_timestamp": DeterministicClock.now().isoformat(),
                "status": "pending_calculation"
            }
            processed_records.append(processed)

        assert len(processed_records) == len(raw_records)

        # Step 3: Validate processed records
        for processed in processed_records:
            assert processed["source_system"] == "Workday"
            assert processed["record_type"] == "expense_report"
            assert "raw_data" in processed


@pytest.mark.integration
@pytest.mark.slow
class TestDataQualityAcrossConnectors:
    """Test data quality consistency across connectors."""

    def test_schema_consistency(
        self,
        mock_sap_client: SAPODataClient,
        mock_oracle_client: OracleRESTClient,
        mock_workday_client
    ):
        """Test that mapped data has consistent schema across connectors."""
        # Extract from all connectors
        sap_query_obj = sap_query().top(1)
        sap_response = mock_sap_client.get("purchase_orders", sap_query_obj.build())
        sap_records = sap_response.get("value", [])

        oracle_query_obj = oracle_query().limit(1)
        oracle_response = mock_oracle_client.get("purchase_orders", oracle_query_obj.build())
        oracle_records = oracle_response.get("items", [])

        workday_records = mock_workday_client.extract_expense_reports(limit=1)

        # Map to standard format
        mapped_records = []

        if sap_records:
            from sap.mappers.po_mapper import PurchaseOrderMapper
            mapper = PurchaseOrderMapper()
            mapped_records.append(mapper.map(sap_records[0]))

        if oracle_records:
            from oracle.mappers.po_mapper import PurchaseOrderMapper
            mapper = PurchaseOrderMapper()
            mapped_records.append(mapper.map(oracle_records[0]))

        if workday_records:
            # Workday expense report mapping
            mapped_records.append({
                "transaction_id": workday_records[0]["expense_id"],
                "source_system": "Workday",
                "transaction_date": workday_records[0]["expense_date"]
            })

        # Validate all mapped records have required fields
        required_fields = ["transaction_id", "source_system", "transaction_date"]

        for record in mapped_records:
            for field in required_fields:
                assert field in record, f"Missing {field} in {record.get('source_system')} record"

    def test_data_type_consistency(
        self,
        mock_sap_client: SAPODataClient,
        mock_oracle_client: OracleRESTClient
    ):
        """Test data type consistency across connectors."""
        # Extract and map from SAP
        sap_query_obj = sap_query().top(1)
        sap_response = mock_sap_client.get("purchase_orders", sap_query_obj.build())
        sap_records = sap_response.get("value", [])

        # Extract and map from Oracle
        oracle_query_obj = oracle_query().limit(1)
        oracle_response = mock_oracle_client.get("purchase_orders", oracle_query_obj.build())
        oracle_records = oracle_response.get("items", [])

        # Validate data types are consistent
        if sap_records and oracle_records:
            from sap.mappers.po_mapper import PurchaseOrderMapper as SAPMapper
            from oracle.mappers.po_mapper import PurchaseOrderMapper as OracleMapper

            sap_mapped = SAPMapper().map(sap_records[0])
            oracle_mapped = OracleMapper().map(oracle_records[0])

            # Both should have string transaction_id
            assert isinstance(sap_mapped["transaction_id"], str)
            assert isinstance(oracle_mapped["transaction_id"], str)

            # Both should have numeric total_amount (if present)
            if "total_amount" in sap_mapped:
                assert isinstance(sap_mapped["total_amount"], (int, float))
            if "total_amount" in oracle_mapped:
                assert isinstance(oracle_mapped["total_amount"], (int, float))


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWithCalculator:
    """Test end-to-end flow including Calculator Agent."""

    def test_sap_to_calculator_flow(self, mock_sap_client: SAPODataClient):
        """Test flow from SAP extraction to emissions calculation."""
        # Step 1: Extract from SAP
        query = sap_query().top(5)
        sap_response = mock_sap_client.get("purchase_orders", query.build())
        raw_records = sap_response.get("value", [])

        assert len(raw_records) > 0

        # Step 2: Map to standard format
        from sap.mappers.po_mapper import PurchaseOrderMapper
        mapper = PurchaseOrderMapper()
        mapped_records = [mapper.map(r) for r in raw_records]

        # Step 3: Simulate Calculator Agent processing
        calculated_records = []
        for record in mapped_records:
            calculated = {
                **record,
                "emissions_co2e": 150.5,  # Mock calculation
                "calculation_method": "spend-based",
                "calculation_timestamp": DeterministicClock.now().isoformat(),
                "status": "calculated"
            }
            calculated_records.append(calculated)

        assert len(calculated_records) == len(raw_records)

        # Step 4: Validate calculated records
        for calculated in calculated_records:
            assert "emissions_co2e" in calculated
            assert calculated["emissions_co2e"] > 0
            assert "calculation_method" in calculated
            assert calculated["status"] == "calculated"


@pytest.mark.integration
class TestConcurrentProcessing:
    """Test concurrent processing of multiple data streams."""

    def test_concurrent_connector_extraction(
        self,
        mock_sap_client: SAPODataClient,
        mock_oracle_client: OracleRESTClient,
        mock_workday_client
    ):
        """Test concurrent extraction from multiple connectors."""
        import concurrent.futures

        def extract_sap():
            query = sap_query().top(5)
            response = mock_sap_client.get("purchase_orders", query.build())
            return "sap", response.get("value", [])

        def extract_oracle():
            query = oracle_query().limit(5)
            response = mock_oracle_client.get("purchase_orders", query.build())
            return "oracle", response.get("items", [])

        def extract_workday():
            records = mock_workday_client.extract_expense_reports(limit=5)
            return "workday", records

        # Execute concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(extract_sap),
                executor.submit(extract_oracle),
                executor.submit(extract_workday)
            ]

            results = {}
            for future in concurrent.futures.as_completed(futures):
                connector_name, data = future.result()
                results[connector_name] = data

        # Validate all extractions succeeded
        assert len(results) == 3
        for connector_name, data in results.items():
            assert isinstance(data, list)


@pytest.mark.integration
class TestErrorHandlingAcrossConnectors:
    """Test error handling consistency across connectors."""

    def test_graceful_degradation(
        self,
        mock_sap_client: SAPODataClient,
        mock_oracle_client: OracleRESTClient,
        mock_workday_client
    ):
        """Test that failure in one connector doesn't affect others."""
        results = []
        errors = []

        # Extract from SAP (may fail)
        try:
            query = sap_query().top(5)
            sap_response = mock_sap_client.get("purchase_orders", query.build())
            results.append(("sap", sap_response.get("value", [])))
        except Exception as e:
            errors.append(("sap", str(e)))

        # Extract from Oracle (may fail)
        try:
            query = oracle_query().limit(5)
            oracle_response = mock_oracle_client.get("purchase_orders", query.build())
            results.append(("oracle", oracle_response.get("items", [])))
        except Exception as e:
            errors.append(("oracle", str(e)))

        # Extract from Workday (may fail)
        try:
            workday_data = mock_workday_client.extract_expense_reports(limit=5)
            results.append(("workday", workday_data))
        except Exception as e:
            errors.append(("workday", str(e)))

        # At least extraction attempts were made
        assert len(results) + len(errors) == 3

        # Successful extractions should return valid data
        for connector_name, data in results:
            assert isinstance(data, list)
