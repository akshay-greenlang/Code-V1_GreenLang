"""
Tests for GL-001 ThermalCommand GraphQL API

Integration tests for GraphQL queries, mutations, and subscriptions.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from strawberry.test import BaseGraphQLTestClient

from api.graphql_api import schema, data_store


# =============================================================================
# Test Client Setup
# =============================================================================

class GraphQLTestClient(BaseGraphQLTestClient):
    """GraphQL test client for ThermalCommand API."""

    async def execute_query(self, query: str, variables: dict = None):
        """Execute a GraphQL query."""
        return await schema.execute(
            query,
            variable_values=variables,
            context_value={"request": None, "user": None},
        )


@pytest.fixture
def client():
    """Create GraphQL test client."""
    return GraphQLTestClient(schema)


@pytest.fixture
def mock_user():
    """Create mock authenticated user."""
    from api.api_auth import ThermalCommandUser, Role
    return ThermalCommandUser(
        user_id=uuid4(),
        username="test_user",
        email="test@example.com",
        tenant_id=uuid4(),
        roles=[Role.OPERATOR],
        created_at=datetime.utcnow(),
    )


# =============================================================================
# Query Tests
# =============================================================================

class TestCurrentPlanQuery:
    """Tests for currentPlan query."""

    @pytest.mark.asyncio
    async def test_get_current_plan(self, client):
        """Test retrieving current dispatch plan."""
        query = """
        query GetCurrentPlan {
            currentPlan {
                planId
                planName
                planVersion
                objective
                planningHorizonHours
                resolutionMinutes
                isActive
                totalThermalOutputMwh
                totalCost
                totalEmissionsKg
                averageEfficiency
                optimizationScore
                solverStatus
            }
        }
        """
        result = await schema.execute(query, context_value={"request": None})

        assert result.errors is None
        assert result.data is not None
        plan = result.data["currentPlan"]
        assert plan["planName"] == "Day-Ahead Dispatch Plan"
        assert plan["isActive"] is True
        assert plan["objective"] == "BALANCE_COST_EMISSIONS"

    @pytest.mark.asyncio
    async def test_get_current_plan_with_recommendations(self, client):
        """Test retrieving plan with setpoint recommendations."""
        query = """
        query GetCurrentPlanWithRecommendations {
            currentPlan {
                planId
                planName
                setpointRecommendations {
                    assetId
                    assetName
                    currentSetpointMw
                    recommendedSetpointMw
                    confidence
                    reason
                }
            }
        }
        """
        result = await schema.execute(query, context_value={"request": None})

        assert result.errors is None
        plan = result.data["currentPlan"]
        assert len(plan["setpointRecommendations"]) > 0
        rec = plan["setpointRecommendations"][0]
        assert rec["assetName"] == "CHP Unit 1"
        assert rec["confidence"] > 0.9


class TestAssetStatesQuery:
    """Tests for assetStates query."""

    @pytest.mark.asyncio
    async def test_get_all_assets(self, client):
        """Test retrieving all asset states."""
        query = """
        query GetAllAssets {
            assetStates {
                items {
                    assetId
                    assetName
                    assetType
                    status
                    currentOutputMw
                    currentSetpointMw
                    supplyTemperatureC
                    returnTemperatureC
                }
                totalCount
                page
                pageSize
            }
        }
        """
        result = await schema.execute(query, context_value={"request": None})

        assert result.errors is None
        data = result.data["assetStates"]
        assert data["totalCount"] >= 1
        asset = data["items"][0]
        assert asset["assetName"] == "CHP Unit 1"
        assert asset["assetType"] == "CHP"

    @pytest.mark.asyncio
    async def test_get_assets_with_filter(self, client):
        """Test filtering assets by type."""
        query = """
        query GetCHPAssets($types: [AssetTypeEnum!]) {
            assetStates(assetTypes: $types) {
                items {
                    assetId
                    assetName
                    assetType
                }
                totalCount
            }
        }
        """
        result = await schema.execute(
            query,
            variable_values={"types": ["CHP"]},
            context_value={"request": None},
        )

        assert result.errors is None
        data = result.data["assetStates"]
        for asset in data["items"]:
            assert asset["assetType"] == "CHP"

    @pytest.mark.asyncio
    async def test_get_asset_with_capacity(self, client):
        """Test retrieving asset with capacity details."""
        query = """
        query GetAssetCapacity {
            assetStates {
                items {
                    assetName
                    capacity {
                        thermalCapacityMw
                        minOutputMw
                        maxOutputMw
                        rampUpRateMwMin
                        rampDownRateMwMin
                    }
                }
            }
        }
        """
        result = await schema.execute(query, context_value={"request": None})

        assert result.errors is None
        asset = result.data["assetStates"]["items"][0]
        assert asset["capacity"]["thermalCapacityMw"] == 100.0
        assert asset["capacity"]["minOutputMw"] == 20.0


class TestConstraintsQuery:
    """Tests for constraints query."""

    @pytest.mark.asyncio
    async def test_get_all_constraints(self, client):
        """Test retrieving all constraints."""
        query = """
        query GetConstraints {
            constraints {
                items {
                    constraintId
                    name
                    constraintType
                    priority
                    isActive
                    isViolated
                }
                totalCount
            }
        }
        """
        result = await schema.execute(query, context_value={"request": None})
        assert result.errors is None

    @pytest.mark.asyncio
    async def test_get_active_constraints(self, client):
        """Test filtering active constraints."""
        query = """
        query GetActiveConstraints {
            constraints(isActive: true) {
                items {
                    name
                    isActive
                }
            }
        }
        """
        result = await schema.execute(query, context_value={"request": None})

        assert result.errors is None
        for constraint in result.data["constraints"]["items"]:
            assert constraint["isActive"] is True


class TestKPIsQuery:
    """Tests for KPIs query."""

    @pytest.mark.asyncio
    async def test_get_all_kpis(self, client):
        """Test retrieving all KPIs."""
        query = """
        query GetKPIs {
            kpis {
                kpiId
                name
                category
                currentValue
                targetValue
                unit
                measurementTimestamp
            }
        }
        """
        result = await schema.execute(query, context_value={"request": None})

        assert result.errors is None
        kpis = result.data["kpis"]
        assert len(kpis) >= 1
        kpi = kpis[0]
        assert kpi["name"] == "System Efficiency"
        assert kpi["unit"] == "%"

    @pytest.mark.asyncio
    async def test_get_kpis_by_category(self, client):
        """Test filtering KPIs by category."""
        query = """
        query GetEfficiencyKPIs($category: String) {
            kpis(category: $category) {
                name
                category
                currentValue
            }
        }
        """
        result = await schema.execute(
            query,
            variable_values={"category": "efficiency"},
            context_value={"request": None},
        )

        assert result.errors is None
        for kpi in result.data["kpis"]:
            assert kpi["category"] == "efficiency"


class TestExplainabilitySummaryQuery:
    """Tests for explainabilitySummary query."""

    @pytest.mark.asyncio
    async def test_get_explainability_summary(self, client):
        """Test retrieving explainability summary."""
        # First get a plan ID
        plan_query = """
        query GetPlanId {
            currentPlan {
                planId
            }
        }
        """
        plan_result = await schema.execute(plan_query, context_value={"request": None})
        plan_id = plan_result.data["currentPlan"]["planId"]

        # Now get explainability
        query = """
        query GetExplainability($planId: ID!) {
            explainabilitySummary(planId: $planId) {
                summaryId
                executiveSummary
                keyDrivers
                globalFeatureImportance {
                    featureName
                    importanceScore
                    direction
                    description
                }
                plainEnglishSummary
            }
        }
        """
        result = await schema.execute(
            query,
            variable_values={"planId": plan_id},
            context_value={"request": None},
        )

        assert result.errors is None
        summary = result.data["explainabilitySummary"]
        assert summary["executiveSummary"] is not None
        assert len(summary["keyDrivers"]) > 0
        assert len(summary["globalFeatureImportance"]) > 0


class TestLatestForecastQuery:
    """Tests for latestForecast query."""

    @pytest.mark.asyncio
    async def test_get_demand_forecast(self, client):
        """Test retrieving demand forecast."""
        query = """
        query GetDemandForecast {
            latestForecast(forecastType: DEMAND) {
                forecastId
                forecastType
                forecastHorizonHours
                resolutionMinutes
                values
                unit
                confidenceLevel
                modelName
                modelVersion
            }
        }
        """
        result = await schema.execute(query, context_value={"request": None})

        assert result.errors is None
        forecast = result.data["latestForecast"]
        assert forecast["forecastType"] == "DEMAND"
        assert forecast["unit"] == "MW"
        assert len(forecast["values"]) > 0


# =============================================================================
# Mutation Tests
# =============================================================================

class TestSubmitDemandUpdateMutation:
    """Tests for submitDemandUpdate mutation."""

    @pytest.mark.asyncio
    async def test_submit_demand_update(self, client):
        """Test submitting demand update."""
        now = datetime.utcnow().isoformat() + "Z"
        timestamps = [
            (datetime.utcnow() + timedelta(minutes=15 * i)).isoformat() + "Z"
            for i in range(4)
        ]

        query = """
        mutation SubmitDemand($input: DemandUpdateInput!) {
            submitDemandUpdate(input: $input) {
                requestId
                success
                message
                recordsReceived
                recordsValidated
                dataQualityScore
            }
        }
        """
        result = await schema.execute(
            query,
            variable_values={
                "input": {
                    "forecastType": "DEMAND",
                    "forecastHorizonHours": 24,
                    "resolutionMinutes": 15,
                    "demandMw": [50.0, 55.0, 60.0, 58.0],
                    "demandTimestamps": timestamps,
                    "sourceSystem": "SCADA",
                }
            },
            context_value={"request": None},
        )

        assert result.errors is None
        response = result.data["submitDemandUpdate"]
        assert response["success"] is True
        assert response["recordsReceived"] == 4


class TestRequestAllocationMutation:
    """Tests for requestAllocation mutation."""

    @pytest.mark.asyncio
    async def test_request_allocation(self, client):
        """Test requesting heat allocation."""
        query = """
        mutation RequestAlloc($input: AllocationRequestInput!) {
            requestAllocation(input: $input) {
                requestId
                responseId
                success
                statusMessage
                allocatedOutputMw
                allocationGapMw
                assetAllocations {
                    assetId
                    assetName
                    recommendedSetpointMw
                    confidence
                }
                estimatedCost
                estimatedEmissionsKg
                optimizationTimeMs
            }
        }
        """
        result = await schema.execute(
            query,
            variable_values={
                "input": {
                    "targetOutputMw": 80.0,
                    "timeWindowMinutes": 15,
                    "objective": "BALANCE_COST_EMISSIONS",
                    "costWeight": 0.5,
                    "emissionsWeight": 0.5,
                }
            },
            context_value={"request": None},
        )

        assert result.errors is None
        response = result.data["requestAllocation"]
        assert response["success"] is True
        assert response["allocatedOutputMw"] > 0


class TestAcknowledgeAlarmMutation:
    """Tests for acknowledgeAlarm mutation."""

    @pytest.mark.asyncio
    async def test_acknowledge_alarm(self, client):
        """Test acknowledging an alarm."""
        query = """
        mutation AckAlarm($input: AlarmAcknowledgementInput!) {
            acknowledgeAlarm(input: $input) {
                alarmId
                success
                message
                acknowledgedAt
                acknowledgedBy
            }
        }
        """
        result = await schema.execute(
            query,
            variable_values={
                "input": {
                    "alarmId": str(uuid4()),
                    "acknowledgedBy": "operator1",
                    "acknowledgementNote": "Investigating",
                }
            },
            context_value={"request": None},
        )

        assert result.errors is None
        response = result.data["acknowledgeAlarm"]
        assert response["success"] is True
        assert response["acknowledgedBy"] == "operator1"


# =============================================================================
# Complex Query Tests
# =============================================================================

class TestComplexQueries:
    """Tests for complex GraphQL queries."""

    @pytest.mark.asyncio
    async def test_multiple_queries_in_one_request(self, client):
        """Test executing multiple queries in a single request."""
        query = """
        query DashboardData {
            currentPlan {
                planName
                totalCost
                totalEmissionsKg
            }
            assetStates {
                items {
                    assetName
                    status
                    currentOutputMw
                }
                totalCount
            }
            kpis {
                name
                currentValue
                unit
            }
        }
        """
        result = await schema.execute(query, context_value={"request": None})

        assert result.errors is None
        assert result.data["currentPlan"] is not None
        assert result.data["assetStates"]["totalCount"] >= 1
        assert len(result.data["kpis"]) >= 1

    @pytest.mark.asyncio
    async def test_deep_nested_query(self, client):
        """Test deeply nested query structure."""
        query = """
        query FullAssetDetails {
            assetStates {
                items {
                    assetName
                    assetType
                    status
                    capacity {
                        thermalCapacityMw
                        maxOutputMw
                        rampUpRateMwMin
                    }
                    efficiency {
                        thermalEfficiency
                        electricalEfficiency
                    }
                    emissions {
                        co2KgPerMwh
                        noxKgPerMwh
                    }
                    cost {
                        fuelCostPerMwh
                        startupCost
                    }
                    health {
                        healthScore
                        faultIndicators
                    }
                }
            }
        }
        """
        result = await schema.execute(query, context_value={"request": None})

        assert result.errors is None
        asset = result.data["assetStates"]["items"][0]
        assert asset["capacity"]["thermalCapacityMw"] == 100.0
        assert asset["efficiency"]["thermalEfficiency"] == 0.88
        assert asset["health"]["healthScore"] == 92.5


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for GraphQL error handling."""

    @pytest.mark.asyncio
    async def test_invalid_query(self, client):
        """Test handling invalid query."""
        query = """
        query InvalidQuery {
            nonExistentField {
                id
            }
        }
        """
        result = await schema.execute(query, context_value={"request": None})
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_invalid_variable_type(self, client):
        """Test handling invalid variable type."""
        query = """
        query GetAssets($types: [AssetTypeEnum!]) {
            assetStates(assetTypes: $types) {
                items {
                    assetName
                }
            }
        }
        """
        result = await schema.execute(
            query,
            variable_values={"types": ["INVALID_TYPE"]},
            context_value={"request": None},
        )
        assert result.errors is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
