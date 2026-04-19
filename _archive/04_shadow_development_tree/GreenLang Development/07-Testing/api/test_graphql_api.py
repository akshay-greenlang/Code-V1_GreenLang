# -*- coding: utf-8 -*-
"""
Unit Tests for GraphQL API

Tests GraphQL queries and mutations for emission calculations.
"""

import pytest
from unittest.mock import Mock
import json


class TestGraphQLCalculationQueries:
    """Test GraphQL calculation queries"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup GraphQL client"""
        try:
            from greenlang.api.graphql.client import GraphQLClient
            self.client = GraphQLClient()
        except ImportError:
            pytest.skip("GraphQL module not available")

    def test_calculate_emission_query(self):
        """Test calculateEmission query"""
        query = """
        query {
            calculateEmission(
                factorId: "diesel-us-stationary"
                activityAmount: 100
                activityUnit: "liters"
            ) {
                emissionsKgCo2e
                provenanceHash
                status
            }
        }
        """

        result = self.client.execute(query)

        assert "data" in result
        assert "calculateEmission" in result["data"]
        calc = result["data"]["calculateEmission"]

        assert "emissionsKgCo2e" in calc
        assert "provenanceHash" in calc

    def test_batch_calculate_mutation(self):
        """Test batchCalculate mutation"""
        mutation = """
        mutation {
            batchCalculate(requests: [
                {
                    factorId: "diesel-us-stationary"
                    activityAmount: 100
                    activityUnit: "liters"
                },
                {
                    factorId: "natural_gas-us-stationary"
                    activityAmount: 500
                    activityUnit: "cubic_meters"
                }
            ]) {
                results {
                    emissionsKgCo2e
                    status
                }
                totalEmissions
            }
        }
        """

        result = self.client.execute(mutation)

        assert "data" in result
        assert "batchCalculate" in result["data"]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
