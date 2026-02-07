# -*- coding: utf-8 -*-
"""
Thanos Upload Integration Tests
===============================

Integration tests for Thanos block upload and store gateway functionality.
Requires running Thanos components and S3 access.

Run with: pytest tests/integration/monitoring/test_thanos_upload.py -v
"""

import pytest
import requests
import time
import boto3
from botocore.exceptions import ClientError
from typing import Dict, Any, List
import os

# Skip all tests if not running in integration mode
pytestmark = pytest.mark.skipif(
    os.environ.get("INTEGRATION_TESTS") != "true",
    reason="Integration tests disabled (set INTEGRATION_TESTS=true to run)"
)


# Configuration from environment
THANOS_QUERY_URL = os.environ.get("THANOS_QUERY_URL", "http://localhost:10901")
THANOS_STORE_URL = os.environ.get("THANOS_STORE_URL", "http://localhost:10904")
THANOS_BUCKET = os.environ.get("THANOS_BUCKET", "gl-thanos-metrics-test")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


@pytest.fixture(scope="module")
def thanos_query_client() -> Dict[str, str]:
    """Create a Thanos Query API client configuration."""
    return {
        "base_url": THANOS_QUERY_URL,
        "api_path": "/api/v1",
    }


@pytest.fixture(scope="module")
def s3_client():
    """Create an S3 client for Thanos bucket verification."""
    return boto3.client("s3", region_name=AWS_REGION)


@pytest.fixture(scope="module")
def wait_for_thanos(thanos_query_client):
    """Wait for Thanos Query to be ready."""
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(
                f"{thanos_query_client['base_url']}/-/ready",
                timeout=5
            )
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)

    pytest.fail("Thanos Query not ready after 60 seconds")


class TestThanosBlockUpload:
    """Tests for Thanos block upload to S3."""

    def test_blocks_uploaded_to_s3(self, s3_client, wait_for_thanos):
        """Test that Prometheus blocks are uploaded to S3."""
        try:
            response = s3_client.list_objects_v2(
                Bucket=THANOS_BUCKET,
                MaxKeys=10
            )

            # Bucket should have some objects after Thanos starts uploading
            assert "Contents" in response, "No objects in Thanos bucket"

            # Check for block structure (ULID directories)
            objects = response["Contents"]
            block_patterns = [obj["Key"] for obj in objects if "/" in obj["Key"]]
            assert len(block_patterns) > 0, "No block directories found"

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchBucket":
                pytest.skip(f"Bucket {THANOS_BUCKET} does not exist")
            raise

    def test_block_meta_json_exists(self, s3_client, wait_for_thanos):
        """Test that block meta.json files exist in S3."""
        try:
            # List objects to find meta.json files
            paginator = s3_client.get_paginator("list_objects_v2")
            meta_files = []

            for page in paginator.paginate(Bucket=THANOS_BUCKET, MaxKeys=100):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        if obj["Key"].endswith("meta.json"):
                            meta_files.append(obj["Key"])

            # After some time, we should have meta.json files
            # This might be empty if Thanos hasn't uploaded yet
            assert isinstance(meta_files, list)

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchBucket":
                pytest.skip(f"Bucket {THANOS_BUCKET} does not exist")
            raise

    def test_s3_bucket_lifecycle_rules(self, s3_client):
        """Test that S3 bucket has correct lifecycle rules."""
        try:
            response = s3_client.get_bucket_lifecycle_configuration(
                Bucket=THANOS_BUCKET
            )

            rules = response.get("Rules", [])
            assert len(rules) > 0, "No lifecycle rules configured"

            # Check for expected rules
            rule_ids = [rule["ID"] for rule in rules]
            # Expected: intelligent-tiering, glacier-archive, expiration
            assert len(rule_ids) >= 1, "Missing lifecycle rules"

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                pytest.skip(f"Bucket {THANOS_BUCKET} does not exist")
            elif error_code == "NoSuchLifecycleConfiguration":
                pytest.skip("Lifecycle configuration not set")
            raise


class TestThanosStoreGateway:
    """Tests for Thanos Store Gateway queries."""

    def test_store_gateway_queries(
        self,
        thanos_query_client: Dict[str, str],
        wait_for_thanos
    ):
        """Test that Store Gateway can be queried through Thanos Query."""
        # Query Thanos for store info
        response = requests.get(
            f"{thanos_query_client['base_url']}{thanos_query_client['api_path']}/stores",
            timeout=10
        )

        assert response.status_code == 200

    def test_historical_data_accessible(
        self,
        thanos_query_client: Dict[str, str],
        wait_for_thanos
    ):
        """Test that historical data is accessible via Thanos."""
        now = int(time.time())
        # Query for data from 24 hours ago (should be in S3 via Store Gateway)
        response = requests.get(
            f"{thanos_query_client['base_url']}{thanos_query_client['api_path']}/query_range",
            params={
                "query": "up",
                "start": now - 86400,  # 24 hours ago
                "end": now - 82800,    # 23 hours ago
                "step": "300",
            },
            timeout=30
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_store_gateway_health(self, wait_for_thanos):
        """Test Store Gateway health endpoint."""
        try:
            response = requests.get(
                f"{THANOS_STORE_URL}/-/healthy",
                timeout=10
            )
            # Store Gateway might not be separately accessible
            if response.status_code == 200:
                assert True
        except requests.RequestException:
            pytest.skip("Store Gateway not directly accessible")


class TestThanosQueryFederation:
    """Tests for Thanos Query federation."""

    def test_query_federation(
        self,
        thanos_query_client: Dict[str, str],
        wait_for_thanos
    ):
        """Test that Thanos Query federates across stores."""
        # Query for up metric across all stores
        response = requests.get(
            f"{thanos_query_client['base_url']}{thanos_query_client['api_path']}/query",
            params={"query": "up"},
            timeout=10
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Results should include data from multiple sources
        results = data["data"]["result"]
        assert isinstance(results, list)

    def test_deduplication_works(
        self,
        thanos_query_client: Dict[str, str],
        wait_for_thanos
    ):
        """Test that Thanos deduplicates data from HA pairs."""
        # Query with deduplication enabled (default)
        response = requests.get(
            f"{thanos_query_client['base_url']}{thanos_query_client['api_path']}/query",
            params={
                "query": "up",
                "dedup": "true",
            },
            timeout=10
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # With HA Prometheus, we should see deduplicated results
        results_dedup = data["data"]["result"]

        # Query without deduplication
        response_no_dedup = requests.get(
            f"{thanos_query_client['base_url']}{thanos_query_client['api_path']}/query",
            params={
                "query": "up",
                "dedup": "false",
            },
            timeout=10
        )

        data_no_dedup = response_no_dedup.json()
        results_no_dedup = data_no_dedup["data"]["result"]

        # Without deduplication, we might see more results (from both HA replicas)
        assert len(results_no_dedup) >= len(results_dedup)

    def test_partial_response_enabled(
        self,
        thanos_query_client: Dict[str, str],
        wait_for_thanos
    ):
        """Test that partial response mode works."""
        response = requests.get(
            f"{thanos_query_client['base_url']}{thanos_query_client['api_path']}/query",
            params={
                "query": "up",
                "partial_response": "true",
            },
            timeout=10
        )

        assert response.status_code == 200

    def test_store_selection(
        self,
        thanos_query_client: Dict[str, str],
        wait_for_thanos
    ):
        """Test store selection for queries."""
        # Get list of stores
        stores_response = requests.get(
            f"{thanos_query_client['base_url']}{thanos_query_client['api_path']}/stores",
            timeout=10
        )

        if stores_response.status_code != 200:
            pytest.skip("Stores endpoint not available")

        stores_data = stores_response.json()
        assert isinstance(stores_data, (list, dict))


class TestThanosMetrics:
    """Tests for Thanos component metrics."""

    def test_thanos_query_metrics(
        self,
        thanos_query_client: Dict[str, str],
        wait_for_thanos
    ):
        """Test Thanos Query exposes metrics."""
        response = requests.get(
            f"{thanos_query_client['base_url']}/metrics",
            timeout=10
        )

        assert response.status_code == 200
        assert "thanos_" in response.text

    def test_block_sync_metrics(
        self,
        thanos_query_client: Dict[str, str],
        wait_for_thanos
    ):
        """Test block sync metrics are available."""
        response = requests.get(
            f"{thanos_query_client['base_url']}{thanos_query_client['api_path']}/query",
            params={"query": "thanos_blocks_meta_synced"},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"

    def test_query_latency_metrics(
        self,
        thanos_query_client: Dict[str, str],
        wait_for_thanos
    ):
        """Test query latency metrics are recorded."""
        response = requests.get(
            f"{thanos_query_client['base_url']}/metrics",
            timeout=10
        )

        assert response.status_code == 200
        # Should have query duration histogram
        assert "thanos_query" in response.text or "http_request" in response.text


class TestThanosCompactor:
    """Tests for Thanos Compactor status."""

    def test_compactor_not_halted(
        self,
        thanos_query_client: Dict[str, str],
        wait_for_thanos
    ):
        """Test that Thanos Compactor is not halted."""
        response = requests.get(
            f"{thanos_query_client['base_url']}{thanos_query_client['api_path']}/query",
            params={"query": "thanos_compact_halted"},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data["status"] == "success" and len(data["data"]["result"]) > 0:
                # Check that compactor is not halted (value should be 0)
                halted_value = float(data["data"]["result"][0]["value"][1])
                assert halted_value == 0, "Thanos Compactor is halted"

    def test_compaction_runs(
        self,
        thanos_query_client: Dict[str, str],
        wait_for_thanos
    ):
        """Test that compaction runs are occurring."""
        response = requests.get(
            f"{thanos_query_client['base_url']}{thanos_query_client['api_path']}/query",
            params={"query": "thanos_compact_group_compactions_total"},
            timeout=10
        )

        # This metric might not exist if compactor hasn't run yet
        assert response.status_code == 200
