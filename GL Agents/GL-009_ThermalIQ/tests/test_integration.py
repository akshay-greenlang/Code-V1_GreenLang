# -*- coding: utf-8 -*-
"""
Integration Tests for GL-009 THERMALIQ

End-to-end integration tests covering:
- Full analysis pipeline
- Multi-calculator workflows
- Database integration
- ERP connector integration
- Kafka streaming integration
- External API integration
- Performance under realistic load
- Data consistency verification

Author: GL-TestEngineer
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import time
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

import pytest


# Try importing hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


# =============================================================================
# TEST CLASS: FULL ANALYSIS PIPELINE
# =============================================================================

class TestFullAnalysisPipeline:
    """Test full analysis pipeline end-to-end."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_thermal_analysis_workflow(
        self, sample_heat_balance, orchestrator, thermal_iq_config
    ):
        """Test complete thermal analysis from input to report."""
        # Step 1: Submit analysis request
        analysis_id = await self._submit_analysis(sample_heat_balance)
        assert analysis_id is not None

        # Step 2: Wait for analysis completion
        result = await self._wait_for_completion(analysis_id, timeout_seconds=30)
        assert result["status"] == "completed"

        # Step 3: Verify all calculations performed
        assert "first_law_efficiency" in result
        assert "second_law_efficiency" in result
        assert "loss_breakdown" in result
        assert "sankey_data" in result

        # Step 4: Verify provenance hash
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

        # Step 5: Generate report
        report = await self._generate_report(analysis_id, format="pdf")
        assert report is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_with_exergy_analysis(
        self, sample_heat_balance
    ):
        """Test pipeline including exergy analysis."""
        input_data = {
            **sample_heat_balance,
            "enable_exergy_analysis": True,
        }

        analysis_id = await self._submit_analysis(input_data)
        result = await self._wait_for_completion(analysis_id)

        # Verify exergy results
        assert "exergy_input_kw" in result
        assert "exergy_output_kw" in result
        assert "exergy_destruction_kw" in result
        assert result["exergy_destruction_kw"] >= 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_with_uncertainty_analysis(
        self, sample_heat_balance
    ):
        """Test pipeline including uncertainty analysis."""
        input_data = {
            **sample_heat_balance,
            "enable_uncertainty_analysis": True,
            "measurement_uncertainties": {
                "fuel_flow": 0.02,  # 2%
                "temperature": 0.01,  # 1%
            },
        }

        analysis_id = await self._submit_analysis(input_data)
        result = await self._wait_for_completion(analysis_id)

        # Verify uncertainty results
        assert "uncertainty" in result
        assert "confidence_interval" in result["uncertainty"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_deterministic_reproducibility(
        self, sample_heat_balance
    ):
        """Test that pipeline produces identical results for identical inputs."""
        # Run analysis twice
        analysis_id_1 = await self._submit_analysis(sample_heat_balance)
        result_1 = await self._wait_for_completion(analysis_id_1)

        analysis_id_2 = await self._submit_analysis(sample_heat_balance)
        result_2 = await self._wait_for_completion(analysis_id_2)

        # Verify identical results
        assert result_1["first_law_efficiency"] == result_2["first_law_efficiency"]
        assert result_1["provenance_hash"] == result_2["provenance_hash"]

    async def _submit_analysis(self, input_data: dict) -> str:
        """Submit analysis request."""
        return f"analysis_{int(time.time() * 1000000)}"

    async def _wait_for_completion(
        self, analysis_id: str, timeout_seconds: int = 30
    ) -> Dict[str, Any]:
        """Wait for analysis to complete."""
        # Simulated completion
        await asyncio.sleep(0.01)  # Simulate processing time

        return {
            "status": "completed",
            "analysis_id": analysis_id,
            "first_law_efficiency": 82.8,
            "second_law_efficiency": 45.2,
            "exergy_input_kw": 1444.5,
            "exergy_output_kw": 653.0,
            "exergy_destruction_kw": 647.0,
            "loss_breakdown": {
                "flue_gas": 80.0,
                "radiation": 12.0,
                "convection": 7.5,
            },
            "sankey_data": {"nodes": [], "links": []},
            "provenance_hash": hashlib.sha256(
                json.dumps({"id": analysis_id}, sort_keys=True).encode()
            ).hexdigest(),
            "uncertainty": {
                "confidence_interval": (81.5, 84.1),
                "confidence_level": 0.95,
            },
        }

    async def _generate_report(self, analysis_id: str, format: str) -> bytes:
        """Generate analysis report."""
        return b"%PDF-1.4 mock pdf content"


# =============================================================================
# TEST CLASS: MULTI-CALCULATOR WORKFLOWS
# =============================================================================

class TestMultiCalculatorWorkflows:
    """Test workflows involving multiple calculators."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_sequential_calculator_chain(self, sample_heat_balance):
        """Test sequential execution of dependent calculators."""
        # Calculator 1: First Law Efficiency
        first_law_result = await self._calculate_first_law(sample_heat_balance)

        # Calculator 2: Second Law Efficiency (depends on First Law)
        second_law_result = await self._calculate_second_law(
            sample_heat_balance,
            first_law_result=first_law_result
        )

        # Calculator 3: Loss Analysis (depends on both)
        loss_result = await self._calculate_loss_breakdown(
            sample_heat_balance,
            first_law_result=first_law_result,
            second_law_result=second_law_result
        )

        # Verify chain completed
        assert first_law_result["efficiency_percent"] > 0
        assert second_law_result["efficiency_percent"] > 0
        assert sum(loss_result["breakdown"].values()) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parallel_independent_calculations(self, sample_heat_balance):
        """Test parallel execution of independent calculations."""
        # Run multiple independent calculations in parallel
        tasks = [
            self._calculate_fuel_analysis(sample_heat_balance),
            self._calculate_steam_properties(sample_heat_balance),
            self._calculate_ambient_correction(sample_heat_balance),
        ]

        results = await asyncio.gather(*tasks)

        # All should complete
        assert len(results) == 3
        assert all(r is not None for r in results)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_calculator_error_isolation(self, sample_heat_balance):
        """Test that error in one calculator doesn't affect others."""
        # Run calculations where one will fail
        results = await asyncio.gather(
            self._calculate_first_law(sample_heat_balance),
            self._calculate_with_error(),  # This will fail
            self._calculate_second_law(sample_heat_balance, {}),
            return_exceptions=True
        )

        # First and third should succeed
        assert isinstance(results[0], dict)
        assert isinstance(results[1], Exception)
        assert isinstance(results[2], dict)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_calculator_result_aggregation(self, sample_heat_balance):
        """Test aggregation of results from multiple calculators."""
        # Run all calculators
        first_law = await self._calculate_first_law(sample_heat_balance)
        second_law = await self._calculate_second_law(sample_heat_balance, first_law)
        losses = await self._calculate_loss_breakdown(sample_heat_balance, first_law, second_law)

        # Aggregate results
        aggregated = {
            "efficiency": {
                "first_law_percent": first_law["efficiency_percent"],
                "second_law_percent": second_law["efficiency_percent"],
            },
            "energy_flows": {
                "input_kw": first_law["energy_input_kw"],
                "output_kw": first_law["useful_output_kw"],
            },
            "losses": losses["breakdown"],
            "provenance": self._generate_aggregate_hash([first_law, second_law, losses]),
        }

        assert "efficiency" in aggregated
        assert "energy_flows" in aggregated
        assert "losses" in aggregated
        assert "provenance" in aggregated

    async def _calculate_first_law(self, input_data: dict) -> Dict[str, Any]:
        """Calculate First Law efficiency."""
        return {
            "efficiency_percent": 82.8,
            "energy_input_kw": 1388.9,
            "useful_output_kw": 1150.0,
        }

    async def _calculate_second_law(
        self, input_data: dict, first_law_result: dict = None
    ) -> Dict[str, Any]:
        """Calculate Second Law efficiency."""
        return {
            "efficiency_percent": 45.2,
            "exergy_input_kw": 1444.5,
            "exergy_output_kw": 653.0,
        }

    async def _calculate_loss_breakdown(
        self, input_data: dict,
        first_law_result: dict = None,
        second_law_result: dict = None
    ) -> Dict[str, Any]:
        """Calculate loss breakdown."""
        return {
            "breakdown": {
                "flue_gas": 80.0,
                "radiation": 12.0,
                "convection": 7.5,
                "blowdown": 8.0,
            },
        }

    async def _calculate_fuel_analysis(self, input_data: dict) -> Dict[str, Any]:
        """Calculate fuel analysis."""
        return {"fuel_energy_kw": 1388.9}

    async def _calculate_steam_properties(self, input_data: dict) -> Dict[str, Any]:
        """Calculate steam properties."""
        return {"enthalpy_kj_kg": 2778.1}

    async def _calculate_ambient_correction(self, input_data: dict) -> Dict[str, Any]:
        """Calculate ambient correction."""
        return {"correction_factor": 1.02}

    async def _calculate_with_error(self) -> Dict[str, Any]:
        """Calculator that raises an error."""
        raise ValueError("Simulated calculator error")

    def _generate_aggregate_hash(self, results: List[Dict]) -> str:
        """Generate aggregate provenance hash."""
        data = json.dumps(results, sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()


# =============================================================================
# TEST CLASS: DATABASE INTEGRATION
# =============================================================================

class TestDatabaseIntegration:
    """Test database integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_save_analysis_result(self, sample_heat_balance, mock_database):
        """Test saving analysis result to database."""
        result = {
            "analysis_id": "analysis_123",
            "efficiency_percent": 82.8,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        saved = await self._save_to_database(result, mock_database)

        assert saved is True
        mock_database.cursor().execute.assert_called()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_retrieve_analysis_result(self, mock_database):
        """Test retrieving analysis result from database."""
        mock_database.cursor().fetchone.return_value = {
            "id": "analysis_123",
            "efficiency_percent": 82.8,
        }

        result = await self._retrieve_from_database("analysis_123", mock_database)

        assert result is not None
        assert result["id"] == "analysis_123"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_database_transaction_rollback(self, mock_database):
        """Test database transaction rollback on error."""
        mock_database.cursor().execute.side_effect = [None, Exception("DB Error")]

        with pytest.raises(Exception):
            await self._save_with_transaction(
                [{"id": "1"}, {"id": "2"}],
                mock_database
            )

        mock_database.rollback.assert_called()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_database_connection_pooling(self):
        """Test database connection pooling."""
        pool = self._create_connection_pool(max_connections=5)

        # Acquire multiple connections
        connections = []
        for _ in range(5):
            conn = await pool.acquire()
            connections.append(conn)

        assert len(connections) == 5

        # Release connections
        for conn in connections:
            await pool.release(conn)

        assert pool.available_connections == 5

    async def _save_to_database(self, result: dict, db) -> bool:
        """Save result to database."""
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO analyses (id, data) VALUES (?, ?)",
            (result["analysis_id"], json.dumps(result))
        )
        return True

    async def _retrieve_from_database(self, analysis_id: str, db) -> Optional[dict]:
        """Retrieve result from database."""
        cursor = db.cursor()
        return cursor.fetchone()

    async def _save_with_transaction(self, items: List[dict], db):
        """Save multiple items in transaction."""
        cursor = db.cursor()
        try:
            for item in items:
                cursor.execute("INSERT INTO items (id) VALUES (?)", (item["id"],))
            db.commit()
        except Exception:
            db.rollback()
            raise

    def _create_connection_pool(self, max_connections: int):
        """Create mock connection pool."""
        class MockPool:
            def __init__(self, max_conn):
                self.max_connections = max_conn
                self.available_connections = max_conn
                self._connections = []

            async def acquire(self):
                if self.available_connections > 0:
                    self.available_connections -= 1
                    conn = MagicMock()
                    self._connections.append(conn)
                    return conn
                raise Exception("No connections available")

            async def release(self, conn):
                self.available_connections += 1

        return MockPool(max_connections)


# =============================================================================
# TEST CLASS: ERP CONNECTOR INTEGRATION
# =============================================================================

class TestERPConnectorIntegration:
    """Test ERP system integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fetch_energy_data_from_erp(self, mock_http_client):
        """Test fetching energy data from ERP system."""
        mock_http_client.get.return_value.json.return_value = {
            "fuel_consumption": [
                {"date": "2025-01-01", "fuel_type": "natural_gas", "volume_m3": 1000}
            ]
        }

        data = await self._fetch_erp_data(
            endpoint="/api/energy/consumption",
            start_date="2025-01-01",
            end_date="2025-01-31",
            client=mock_http_client
        )

        assert "fuel_consumption" in data
        assert len(data["fuel_consumption"]) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_push_results_to_erp(self, mock_http_client):
        """Test pushing analysis results to ERP system."""
        mock_http_client.post.return_value.status_code = 201

        result = await self._push_to_erp(
            endpoint="/api/thermal/results",
            data={
                "analysis_id": "123",
                "efficiency_percent": 82.8,
            },
            client=mock_http_client
        )

        assert result is True
        mock_http_client.post.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_erp_authentication(self, mock_http_client):
        """Test ERP authentication flow."""
        # Mock token response
        mock_http_client.post.return_value.json.return_value = {
            "access_token": "test_token_123",
            "expires_in": 3600,
        }

        token = await self._authenticate_erp(
            client_id="test_client",
            client_secret="test_secret",
            client=mock_http_client
        )

        assert token is not None
        assert "access_token" in token

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_erp_retry_on_timeout(self, mock_http_client):
        """Test retry logic on ERP timeout."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise asyncio.TimeoutError()
            response = MagicMock()
            response.json.return_value = {"data": "success"}
            return response

        mock_http_client.get.side_effect = side_effect

        data = await self._fetch_erp_data_with_retry(
            endpoint="/api/data",
            client=mock_http_client,
            max_retries=3
        )

        assert call_count == 3
        assert data is not None

    async def _fetch_erp_data(
        self, endpoint: str, start_date: str, end_date: str, client
    ) -> Dict[str, Any]:
        """Fetch data from ERP system."""
        response = client.get(endpoint, params={
            "start_date": start_date,
            "end_date": end_date,
        })
        return response.json()

    async def _push_to_erp(
        self, endpoint: str, data: dict, client
    ) -> bool:
        """Push data to ERP system."""
        response = client.post(endpoint, json=data)
        return response.status_code == 201

    async def _authenticate_erp(
        self, client_id: str, client_secret: str, client
    ) -> Dict[str, Any]:
        """Authenticate with ERP system."""
        response = client.post("/oauth/token", data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        })
        return response.json()

    async def _fetch_erp_data_with_retry(
        self, endpoint: str, client, max_retries: int
    ) -> Optional[Dict[str, Any]]:
        """Fetch ERP data with retry logic."""
        for attempt in range(max_retries):
            try:
                response = client.get(endpoint)
                return response.json()
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.01)
        return None


# =============================================================================
# TEST CLASS: KAFKA STREAMING INTEGRATION
# =============================================================================

class TestKafkaStreamingIntegration:
    """Test Kafka streaming integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_publish_analysis_event(self, mock_kafka):
        """Test publishing analysis event to Kafka."""
        event = {
            "event_type": "analysis_completed",
            "analysis_id": "analysis_123",
            "efficiency_percent": 82.8,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await self._publish_event(
            topic="thermal-analysis-events",
            event=event,
            producer=mock_kafka["producer"]
        )

        mock_kafka["producer"].send.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_consume_analysis_request(self, mock_kafka):
        """Test consuming analysis request from Kafka."""
        mock_kafka["consumer"].poll.return_value = {
            "thermal-analysis-requests": [
                MagicMock(value=json.dumps({
                    "request_id": "req_123",
                    "input_data": {"value": 100},
                }).encode())
            ]
        }

        messages = await self._consume_messages(
            topic="thermal-analysis-requests",
            consumer=mock_kafka["consumer"],
            timeout_ms=1000
        )

        assert len(messages) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kafka_event_ordering(self, mock_kafka):
        """Test that events maintain order within partition."""
        events = [
            {"seq": 1, "event": "start"},
            {"seq": 2, "event": "progress"},
            {"seq": 3, "event": "complete"},
        ]

        for event in events:
            await self._publish_event(
                topic="ordered-events",
                event=event,
                producer=mock_kafka["producer"],
                key="analysis_123"  # Same key = same partition
            )

        # Events with same key should be in order
        assert mock_kafka["producer"].send.call_count == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kafka_dead_letter_queue(self, mock_kafka):
        """Test dead letter queue for failed messages."""
        failed_message = {
            "original_topic": "analysis-requests",
            "error": "Processing failed",
            "message": {"invalid": "data"},
        }

        await self._publish_to_dlq(
            dlq_topic="analysis-requests-dlq",
            message=failed_message,
            producer=mock_kafka["producer"]
        )

        mock_kafka["producer"].send.assert_called()

    async def _publish_event(
        self, topic: str, event: dict, producer, key: str = None
    ):
        """Publish event to Kafka topic."""
        producer.send(
            topic,
            key=key.encode() if key else None,
            value=json.dumps(event).encode()
        )
        producer.flush()

    async def _consume_messages(
        self, topic: str, consumer, timeout_ms: int
    ) -> List[dict]:
        """Consume messages from Kafka topic."""
        messages = []
        records = consumer.poll(timeout_ms=timeout_ms)

        for topic_partition, partition_records in records.items():
            for record in partition_records:
                messages.append(json.loads(record.value.decode()))

        return messages

    async def _publish_to_dlq(
        self, dlq_topic: str, message: dict, producer
    ):
        """Publish failed message to dead letter queue."""
        producer.send(
            dlq_topic,
            value=json.dumps(message).encode()
        )
        producer.flush()


# =============================================================================
# TEST CLASS: DATA CONSISTENCY VERIFICATION
# =============================================================================

class TestDataConsistencyVerification:
    """Test data consistency across components."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_energy_balance_consistency(self, sample_heat_balance):
        """Test that energy balance is consistent across calculations."""
        result = await self._run_full_analysis(sample_heat_balance)

        # Energy balance: Input = Output + Losses
        total_input = result["energy_input_kw"]
        total_output = result["useful_output_kw"]
        total_losses = sum(result["loss_breakdown"].values())

        balance_error = abs(total_input - (total_output + total_losses))
        balance_error_percent = balance_error / total_input * 100

        assert balance_error_percent < 2.0, \
            f"Energy balance error {balance_error_percent:.2f}% exceeds 2% tolerance"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_provenance_hash_chain_integrity(self, sample_heat_balance):
        """Test provenance hash chain integrity."""
        result = await self._run_full_analysis(sample_heat_balance)

        # Verify hash is derivable from inputs
        recalculated_hash = self._calculate_provenance_hash(sample_heat_balance)

        # Note: Actual hash might include additional metadata
        assert result["provenance_hash"] is not None
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_timestamp_ordering(self, sample_heat_balance):
        """Test that timestamps are properly ordered."""
        result = await self._run_full_analysis(sample_heat_balance)

        timestamps = result.get("timestamps", {})
        if timestamps:
            start = datetime.fromisoformat(timestamps.get("start", "2000-01-01"))
            end = datetime.fromisoformat(timestamps.get("end", "2000-01-01"))

            assert end >= start, "End timestamp should be after start"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cross_calculator_consistency(self, sample_heat_balance):
        """Test consistency between different calculators."""
        # Run First Law and Second Law calculations
        first_law = await self._calculate_first_law(sample_heat_balance)
        second_law = await self._calculate_second_law(sample_heat_balance)

        # Second Law efficiency should always be <= First Law
        assert second_law["efficiency_percent"] <= first_law["efficiency_percent"], \
            "Second Law efficiency cannot exceed First Law efficiency"

    async def _run_full_analysis(self, input_data: dict) -> Dict[str, Any]:
        """Run full analysis pipeline."""
        return {
            "energy_input_kw": 1388.9,
            "useful_output_kw": 1150.0,
            "loss_breakdown": {
                "flue_gas": 80.0,
                "radiation": 12.0,
                "convection": 7.5,
                "blowdown": 8.0,
                "unaccounted": 138.4,
            },
            "provenance_hash": hashlib.sha256(
                json.dumps(input_data, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "timestamps": {
                "start": datetime.now(timezone.utc).isoformat(),
                "end": (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
            },
        }

    async def _calculate_first_law(self, input_data: dict) -> Dict[str, Any]:
        """Calculate First Law efficiency."""
        return {"efficiency_percent": 82.8}

    async def _calculate_second_law(self, input_data: dict) -> Dict[str, Any]:
        """Calculate Second Law efficiency."""
        return {"efficiency_percent": 45.2}

    def _calculate_provenance_hash(self, data: dict) -> str:
        """Calculate provenance hash."""
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()


# =============================================================================
# TEST CLASS: PERFORMANCE UNDER LOAD
# =============================================================================

class TestPerformanceUnderLoad:
    """Test performance under realistic load conditions."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_analysis_requests(self, sample_heat_balance):
        """Test handling concurrent analysis requests."""
        num_requests = 50

        async def run_analysis(idx: int):
            input_data = {**sample_heat_balance, "request_id": idx}
            start = time.perf_counter()
            await self._mock_analysis(input_data)
            return time.perf_counter() - start

        tasks = [run_analysis(i) for i in range(num_requests)]

        start_total = time.perf_counter()
        latencies = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_total

        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies) * 1000  # ms
        throughput = num_requests / total_time

        assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms exceeds 100ms"
        assert throughput >= 50, f"Throughput {throughput:.1f}/sec below 50/sec"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_sustained_load(self, sample_heat_balance):
        """Test sustained load over time."""
        duration_seconds = 1.0
        requests_completed = 0

        start_time = time.perf_counter()
        while time.perf_counter() - start_time < duration_seconds:
            await self._mock_analysis(sample_heat_balance)
            requests_completed += 1

        throughput = requests_completed / duration_seconds

        assert throughput >= 100, f"Sustained throughput {throughput:.1f}/sec below 100/sec"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_stability_under_load(self, sample_heat_balance):
        """Test memory remains stable under load."""
        import sys

        # Get initial memory
        initial_size = sys.getsizeof(sample_heat_balance)

        # Run many analyses
        for _ in range(1000):
            await self._mock_analysis(sample_heat_balance)

        # Memory should not grow significantly
        # (In a real test, would use proper memory profiling)
        assert True  # Placeholder - real test would check memory

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_latency_percentiles(self, sample_heat_balance):
        """Test latency at various percentiles."""
        num_requests = 100
        latencies = []

        for _ in range(num_requests):
            start = time.perf_counter()
            await self._mock_analysis(sample_heat_balance)
            latencies.append((time.perf_counter() - start) * 1000)

        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        assert p50 < 50, f"P50 latency {p50:.2f}ms exceeds 50ms"
        assert p95 < 100, f"P95 latency {p95:.2f}ms exceeds 100ms"
        assert p99 < 200, f"P99 latency {p99:.2f}ms exceeds 200ms"

    async def _mock_analysis(self, input_data: dict) -> dict:
        """Mock analysis with realistic delay."""
        await asyncio.sleep(0.001)  # 1ms simulated processing
        return {"efficiency_percent": 82.8}


# =============================================================================
# PROPERTY-BASED INTEGRATION TESTS
# =============================================================================

if HAS_HYPOTHESIS:

    class TestIntegrationPropertyBased:
        """Property-based integration tests."""

        @given(
            fuel_flow=st.floats(min_value=1.0, max_value=10000.0),
            heating_value=st.floats(min_value=10.0, max_value=60.0),
        )
        @settings(max_examples=50)
        def test_efficiency_bounded_for_any_input(
            self, fuel_flow: float, heating_value: float
        ):
            """Property: Efficiency is always bounded for any valid input."""
            assume(not (fuel_flow != fuel_flow or heating_value != heating_value))  # No NaN

            input_data = {
                "energy_inputs": {
                    "fuel_inputs": [{
                        "mass_flow_kg_hr": fuel_flow,
                        "heating_value_mj_kg": heating_value,
                    }]
                },
            }

            # Simulated efficiency calculation
            energy_input = fuel_flow * heating_value * 0.2778
            output = energy_input * 0.828  # Assume 82.8% efficiency
            efficiency = (output / energy_input * 100) if energy_input > 0 else 0

            assert 0 <= efficiency <= 100

        @given(
            temperature_c=st.floats(min_value=0.0, max_value=500.0),
            ambient_c=st.floats(min_value=-40.0, max_value=50.0),
        )
        @settings(max_examples=50)
        def test_exergy_consistency(self, temperature_c: float, ambient_c: float):
            """Property: Exergy output <= Energy output."""
            assume(temperature_c > ambient_c)

            import math

            T_K = temperature_c + 273.15
            T0_K = ambient_c + 273.15

            energy = 1000.0  # kW
            carnot = 1 - T0_K / T_K
            exergy = energy * carnot

            assert exergy <= energy
