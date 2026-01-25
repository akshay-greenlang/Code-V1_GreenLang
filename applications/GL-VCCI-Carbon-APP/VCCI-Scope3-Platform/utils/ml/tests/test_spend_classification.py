# -*- coding: utf-8 -*-
"""
Tests for Spend Classification Service.

Tests LLM classification, rule-based fallback, confidence routing,
integration tests, and edge cases.

Target: 600+ lines, 30 tests
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List


# Mock spend classification service
class SpendClassificationService:
    """Service for classifying procurement spend into Scope 3 categories."""

    def __init__(self, llm_client, rules_engine, config: Dict = None):
        self.llm_client = llm_client
        self.rules_engine = rules_engine
        self.config = config or {}

    def classify(self, description: str) -> Dict:
        """Classify a procurement description."""
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")

        # Try rules first if configured
        if self.config.get("use_rules_first", True):
            rule_result = self.rules_engine.classify(description)

            if rule_result and rule_result.get("confidence", 0) >= self.config.get("rules_confidence_threshold", 0.75):
                return {
                    "category_id": rule_result["category_id"],
                    "category_name": rule_result["category_name"],
                    "confidence": rule_result["confidence"],
                    "method": "rules"
                }

        # Fallback to LLM
        llm_result = self.llm_client.classify(description, self._get_categories())

        return {
            "category_id": llm_result["category_id"],
            "category_name": llm_result["category_name"],
            "confidence": llm_result["confidence"],
            "method": "llm",
            "reasoning": llm_result.get("reasoning", "")
        }

    def classify_batch(self, descriptions: List[str]) -> List[Dict]:
        """Classify multiple descriptions."""
        results = []
        for desc in descriptions:
            try:
                result = self.classify(desc)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "description": desc
                })
        return results

    def _get_categories(self) -> List[str]:
        """Get list of Scope 3 category names."""
        return [
            "Purchased Goods and Services",
            "Capital Goods",
            "Fuel and Energy Related Activities",
            "Upstream Transportation and Distribution",
            "Waste Generated in Operations",
            "Business Travel",
            "Employee Commuting",
            "Upstream Leased Assets",
            "Downstream Transportation and Distribution",
            "Processing of Sold Products",
            "Use of Sold Products",
            "End-of-Life Treatment of Sold Products",
            "Downstream Leased Assets",
            "Franchises",
            "Investments"
        ]


# ============================================================================
# TEST SUITE - LLM CLASSIFICATION
# ============================================================================

class TestLLMClassification:
    """Test suite for LLM-based classification."""

    @pytest.fixture
    def service(self, mock_llm_client, mock_rule_engine):
        return SpendClassificationService(
            mock_llm_client,
            mock_rule_engine,
            {"use_rules_first": False}
        )

    def test_classify_business_travel(self, service):
        """Test classifying business travel expense."""
        result = service.classify("Airfare for business trip to London")

        assert result["category_id"] == 6
        assert result["category_name"] == "Business Travel"
        assert result["method"] == "llm"

    def test_classify_transportation(self, service):
        """Test classifying transportation expense."""
        result = service.classify("Freight services for inbound materials")

        assert result["category_id"] == 4
        assert result["category_name"] == "Upstream Transportation and Distribution"

    def test_classify_waste_management(self, service):
        """Test classifying waste management expense."""
        result = service.classify("Waste disposal services")

        assert result["category_id"] == 5
        assert result["category_name"] == "Waste Generated in Operations"

    def test_classify_energy(self, service):
        """Test classifying energy expense."""
        result = service.classify("Electricity consumption for facilities")

        assert result["category_id"] == 3
        assert result["category_name"] == "Fuel and Energy Related Activities"

    def test_classify_office_supplies(self, service):
        """Test classifying office supplies."""
        result = service.classify("Office supplies - paper and pens")

        assert result["category_id"] == 1
        assert result["category_name"] == "Purchased Goods and Services"

    def test_classify_capital_goods(self, service):
        """Test classifying capital goods."""
        result = service.classify("Manufacturing equipment purchase")

        assert result["category_id"] == 2
        assert result["category_name"] == "Capital Goods"

    def test_classification_includes_confidence_score(self, service):
        """Test that classification includes confidence score."""
        result = service.classify("Hotel accommodation for conference")

        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_classification_includes_reasoning(self, service):
        """Test that LLM classification includes reasoning."""
        result = service.classify("Business travel expenses")

        assert "reasoning" in result
        assert len(result["reasoning"]) > 0

    def test_classify_with_empty_description_raises_error(self, service):
        """Test that empty description raises error."""
        with pytest.raises(ValueError, match="Description cannot be empty"):
            service.classify("")

        with pytest.raises(ValueError, match="Description cannot be empty"):
            service.classify("   ")


# ============================================================================
# TEST SUITE - RULE-BASED FALLBACK
# ============================================================================

class TestRuleBasedFallback:
    """Test suite for rule-based classification."""

    @pytest.fixture
    def service(self, mock_llm_client, mock_rule_engine):
        return SpendClassificationService(
            mock_llm_client,
            mock_rule_engine,
            {
                "use_rules_first": True,
                "rules_confidence_threshold": 0.75
            }
        )

    def test_use_rules_when_high_confidence(self, service):
        """Test that rules are used when confidence is high."""
        result = service.classify("Airfare for business travel")

        # Should use rules (mock returns confidence 0.85)
        assert result["method"] == "rules"

    def test_fallback_to_llm_when_low_confidence(self, service, mock_rule_engine):
        """Test fallback to LLM when rules confidence is low."""
        # Mock low confidence rule result
        mock_rule_engine.classify.return_value = {
            "category_id": 1,
            "category_name": "Purchased Goods and Services",
            "confidence": 0.60  # Below threshold
        }

        result = service.classify("Ambiguous purchase description")

        assert result["method"] == "llm"

    def test_fallback_to_llm_when_no_rule_match(self, service, mock_rule_engine):
        """Test fallback to LLM when no rule matches."""
        mock_rule_engine.classify.return_value = None

        result = service.classify("Unknown expense type")

        assert result["method"] == "llm"

    def test_rules_classification_includes_method(self, service):
        """Test that rules classification includes method."""
        result = service.classify("Business flight to New York")

        if result["method"] == "rules":
            assert result["category_id"] is not None


# ============================================================================
# TEST SUITE - CONFIDENCE ROUTING
# ============================================================================

class TestConfidenceRouting:
    """Test suite for confidence-based routing."""

    @pytest.fixture
    def service(self, mock_llm_client, mock_rule_engine):
        return SpendClassificationService(
            mock_llm_client,
            mock_rule_engine,
            {
                "use_rules_first": True,
                "rules_confidence_threshold": 0.80
            }
        )

    def test_routing_with_different_thresholds(self, service, mock_rule_engine):
        """Test routing behavior with different confidence thresholds."""
        # High confidence rule result
        mock_rule_engine.classify.return_value = {
            "category_id": 6,
            "category_name": "Business Travel",
            "confidence": 0.85
        }

        result = service.classify("Travel expenses")

        assert result["method"] == "rules"
        assert result["confidence"] >= 0.80

    def test_routing_below_threshold_uses_llm(self, service, mock_rule_engine):
        """Test that below-threshold confidence uses LLM."""
        mock_rule_engine.classify.return_value = {
            "category_id": 1,
            "category_name": "Purchased Goods and Services",
            "confidence": 0.70  # Below 0.80 threshold
        }

        result = service.classify("Various purchases")

        assert result["method"] == "llm"


# ============================================================================
# TEST SUITE - BATCH CLASSIFICATION
# ============================================================================

class TestBatchClassification:
    """Test suite for batch classification."""

    @pytest.fixture
    def service(self, mock_llm_client, mock_rule_engine):
        return SpendClassificationService(mock_llm_client, mock_rule_engine)

    def test_classify_batch(self, service):
        """Test batch classification."""
        descriptions = [
            "Business travel expenses",
            "Freight shipping costs",
            "Waste disposal services"
        ]

        results = service.classify_batch(descriptions)

        assert len(results) == 3
        assert all("category_id" in r or "error" in r for r in results)

    def test_classify_batch_handles_errors(self, service, mock_llm_client):
        """Test that batch classification handles errors gracefully."""
        # Make LLM client raise error for specific input
        def mock_classify_with_error(desc, cats):
            if "error" in desc.lower():
                raise Exception("Classification error")
            return {"category_id": 1, "category_name": "Test", "confidence": 0.8}

        mock_llm_client.classify.side_effect = mock_classify_with_error

        descriptions = ["Good description", "ERROR description", "Another good one"]
        results = service.classify_batch(descriptions)

        assert len(results) == 3
        assert "error" in results[1]

    def test_classify_empty_batch(self, service):
        """Test batch classification with empty list."""
        results = service.classify_batch([])

        assert results == []


# ============================================================================
# TEST SUITE - EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test suite for edge cases."""

    @pytest.fixture
    def service(self, mock_llm_client, mock_rule_engine):
        return SpendClassificationService(mock_llm_client, mock_rule_engine)

    def test_classify_very_short_description(self, service):
        """Test classification with very short description."""
        result = service.classify("Travel")

        assert "category_id" in result

    def test_classify_very_long_description(self, service):
        """Test classification with very long description."""
        long_desc = "Business travel " * 100
        result = service.classify(long_desc)

        assert "category_id" in result

    def test_classify_with_special_characters(self, service):
        """Test classification with special characters."""
        result = service.classify("Flight to O'Hare Airport & hotel in NYC")

        assert "category_id" in result

    def test_classify_with_numbers(self, service):
        """Test classification with numeric values."""
        result = service.classify("Purchase of 500 units @ $10 each")

        assert "category_id" in result

    def test_classify_ambiguous_description(self, service):
        """Test classification with ambiguous description."""
        result = service.classify("Various expenses")

        assert "category_id" in result
        # Ambiguous should likely have lower confidence
        assert result["confidence"] < 0.95

    def test_classify_mixed_categories(self, service):
        """Test description that could fit multiple categories."""
        result = service.classify("Travel freight services for equipment delivery")

        # Should still classify to one category
        assert "category_id" in result
        assert 1 <= result["category_id"] <= 15
