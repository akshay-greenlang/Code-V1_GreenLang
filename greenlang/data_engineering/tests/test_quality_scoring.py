"""
Test Data Quality Scoring Framework
===================================

Unit tests for the data quality scoring framework.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

import pytest
from datetime import datetime

from greenlang.data_engineering.quality.scoring import (
    DataQualityScorer,
    QualityDimension,
    QualityScore,
    create_emission_factor_scorer,
    create_cbam_scorer,
)


class TestDataQualityScorer:
    """Test suite for DataQualityScorer."""

    @pytest.fixture
    def scorer(self):
        """Create default quality scorer."""
        return DataQualityScorer()

    @pytest.fixture
    def emission_factor_scorer(self):
        """Create emission factor specific scorer."""
        return create_emission_factor_scorer()

    @pytest.fixture
    def valid_records(self):
        """Create list of valid emission factor records."""
        return [
            {
                "factor_id": f"ef-{i:03d}",
                "factor_hash": f"hash{i:03d}",
                "factor_value": 0.5 + i * 0.1,
                "factor_unit": "kgCO2e/kWh",
                "industry": "electricity",
                "region": "europe",
                "scope_type": "scope_2_location",
                "reference_year": 2024,
                "valid_from": "2024-01-01",
            }
            for i in range(100)
        ]

    def test_score_perfect_dataset(self, emission_factor_scorer, valid_records):
        """Test scoring of perfect dataset."""
        result = emission_factor_scorer.score_dataset(valid_records)

        assert result.overall_score >= 80
        assert result.grade in ["A+", "A", "B+", "B"]
        assert result.total_records == 100
        assert result.records_passed == result.total_records

    def test_score_empty_dataset(self, scorer):
        """Test scoring of empty dataset."""
        result = scorer.score_dataset([])

        assert result.overall_score == 0
        assert result.grade == "F"
        assert result.total_records == 0

    def test_completeness_score(self, scorer, valid_records):
        """Test completeness scoring."""
        # Remove required field from half the records
        for i in range(0, 50):
            valid_records[i]["factor_value"] = None

        result = scorer.score_dataset(valid_records)

        # Completeness should be affected
        assert result.dimension_scores.get("completeness", 100) < 100
        assert result.overall_score < 100

    def test_validity_score(self, emission_factor_scorer, valid_records):
        """Test validity scoring."""
        # Add invalid values
        for i in range(0, 20):
            valid_records[i]["factor_value"] = -1  # Invalid negative value

        result = emission_factor_scorer.score_dataset(valid_records)

        # Validity should be affected
        assert result.dimension_scores.get("validity", 100) < 100

    def test_uniqueness_score(self, scorer, valid_records):
        """Test uniqueness scoring."""
        # Create duplicates
        for i in range(50, 100):
            valid_records[i]["factor_hash"] = valid_records[0]["factor_hash"]

        result = scorer.score_dataset(valid_records)

        # Uniqueness should be significantly affected
        assert result.dimension_scores.get("uniqueness", 100) < 100

    def test_timeliness_score(self, emission_factor_scorer, valid_records):
        """Test timeliness scoring."""
        # Make records old
        for record in valid_records:
            record["reference_year"] = 2015  # Old data

        result = emission_factor_scorer.score_dataset(valid_records)

        # Timeliness should be affected
        assert result.dimension_scores.get("timeliness", 100) < 100

    def test_quality_grade_calculation(self, scorer):
        """Test quality grade assignment."""
        # Test various scores
        test_cases = [
            (95, "A+"),
            (92, "A"),
            (87, "B+"),
            (82, "B"),
            (77, "C+"),
            (72, "C"),
            (67, "D+"),
            (62, "D"),
            (55, "F"),
        ]

        for score, expected_grade in test_cases:
            quality_score = QualityScore(
                overall_score=score,
                total_records=100,
                records_passed=score,
                records_failed=100 - score,
            )
            assert quality_score.to_grade() == expected_grade

    def test_single_record_scoring(self, emission_factor_scorer):
        """Test scoring a single record."""
        valid_record = {
            "factor_id": "ef-001",
            "factor_value": 0.5,
            "factor_unit": "kgCO2e/kWh",
            "industry": "electricity",
            "region": "europe",
            "scope_type": "scope_2_location",
            "reference_year": 2024,
        }

        result = emission_factor_scorer.score_single_record(valid_record)

        assert result["passed"] is True
        assert result["score"] >= 80
        assert len(result["issues"]) == 0

    def test_single_record_with_issues(self, emission_factor_scorer):
        """Test scoring a single record with issues."""
        invalid_record = {
            "factor_id": "ef-001",
            "factor_value": -1,  # Invalid
            # Missing required fields
        }

        result = emission_factor_scorer.score_single_record(invalid_record)

        assert result["passed"] is False
        assert result["score"] < 100
        assert len(result["issues"]) > 0


class TestCBAMScorer:
    """Test CBAM-specific scoring."""

    @pytest.fixture
    def cbam_scorer(self):
        """Create CBAM scorer."""
        return create_cbam_scorer()

    def test_cbam_product_code_validation(self, cbam_scorer):
        """Test CBAM CN code validation."""
        records = [
            {
                "factor_id": "cbam-001",
                "factor_value": 2.0,
                "factor_unit": "tCO2e/t",
                "product_code": "72061000",  # Valid 8-digit CN code
                "country_code": "UA",
                "production_route": "BF-BOF",
            }
        ]

        result = cbam_scorer.score_dataset(records)
        assert result.dimension_scores.get("validity", 0) > 80

    def test_cbam_missing_production_route(self, cbam_scorer):
        """Test CBAM completeness for production route."""
        records = [
            {
                "factor_id": "cbam-001",
                "factor_value": 2.0,
                "factor_unit": "tCO2e/t",
                "product_code": "72061000",
                "country_code": "UA",
                # Missing production_route
            }
        ]

        result = cbam_scorer.score_dataset(records)
        # Should have completeness issue
        assert result.dimension_scores.get("completeness", 100) < 100


class TestQualityDimensions:
    """Test individual quality dimensions."""

    def test_dimension_weights_sum_to_one(self):
        """Verify dimension weights sum to 1.0."""
        scorer = DataQualityScorer()
        total_weight = sum(scorer.weights.values())
        assert abs(total_weight - 1.0) < 0.001

    def test_all_dimensions_scored(self):
        """Verify all dimensions are scored."""
        scorer = DataQualityScorer()
        records = [{"factor_id": "test", "factor_value": 1.0, "factor_unit": "kg"}]
        result = scorer.score_dataset(records)

        expected_dimensions = [
            "completeness", "validity", "accuracy",
            "consistency", "uniqueness", "timeliness"
        ]

        for dim in expected_dimensions:
            assert dim in result.dimension_scores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
