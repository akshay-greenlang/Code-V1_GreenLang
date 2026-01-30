"""
GL-EUDR-001: LLM Integration Service Tests

Comprehensive test suite for the LLM integration service covering:
- Entity extraction from text
- Fuzzy matching assistance
- Natural language query parsing
- Gap materiality assessment
- Provider fallback handling

Run with: pytest test_llm_service.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from .llm_service import (
    LLMProvider,
    LLMIntegrationService,
    ExtractedEntity,
    FuzzyMatchSuggestion,
    ParsedQuery,
    GapMateriality,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def local_service():
    """Create LLM service in local mode (no API calls)."""
    return LLMIntegrationService(provider=LLMProvider.LOCAL)


@pytest.fixture
def mock_claude_service():
    """Create LLM service with mocked Claude client."""
    service = LLMIntegrationService(
        provider=LLMProvider.CLAUDE,
        api_key="test-key"
    )
    return service


# =============================================================================
# ENTITY EXTRACTION TESTS
# =============================================================================

class TestEntityExtraction:
    """Test entity extraction from unstructured text."""

    def test_extract_country_local(self, local_service):
        """Test country extraction in local mode."""
        text = "Coffee shipment from Brazil to Germany via Rotterdam"
        entities = local_service.extract_entities(text, ["COUNTRY"])

        country_entities = [e for e in entities if e.entity_type == "COUNTRY"]
        assert len(country_entities) >= 1
        assert any(e.value == "BR" for e in country_entities)

    def test_extract_quantity_local(self, local_service):
        """Test quantity extraction in local mode."""
        text = "Order: 5,000 kg of Arabica coffee beans"
        entities = local_service.extract_entities(text, ["QUANTITY"])

        qty_entities = [e for e in entities if e.entity_type == "QUANTITY"]
        assert len(qty_entities) == 1
        assert qty_entities[0].value == "5000"
        assert qty_entities[0].metadata.get("unit") == "kg"

    def test_extract_certification_local(self, local_service):
        """Test certification extraction in local mode."""
        text = "All products are Rainforest Alliance certified and Fair Trade approved"
        entities = local_service.extract_entities(text, ["CERTIFICATION"])

        cert_entities = [e for e in entities if e.entity_type == "CERTIFICATION"]
        assert len(cert_entities) >= 2
        cert_names = [e.value for e in cert_entities]
        assert "Rainforest Alliance" in cert_names
        assert "Fair Trade" in cert_names

    def test_extract_commodity_local(self, local_service):
        """Test commodity extraction in local mode."""
        text = "Palm oil production from sustainable plantations"
        entities = local_service.extract_entities(text, ["COMMODITY"])

        commodity_entities = [e for e in entities if e.entity_type == "COMMODITY"]
        assert len(commodity_entities) == 1
        assert commodity_entities[0].value == "PALM_OIL"

    def test_extract_multiple_quantities(self, local_service):
        """Test extraction of multiple quantities."""
        text = "Shipment contains 1000 kg cocoa and 500 tonnes coffee"
        entities = local_service.extract_entities(text, ["QUANTITY"])

        qty_entities = [e for e in entities if e.entity_type == "QUANTITY"]
        assert len(qty_entities) == 2

    def test_extract_empty_text(self, local_service):
        """Test extraction from empty text."""
        entities = local_service.extract_entities("", ["COUNTRY", "QUANTITY"])
        assert entities == []

    def test_extract_no_entities_found(self, local_service):
        """Test extraction when no entities are present."""
        text = "This is a simple document without supply chain information"
        entities = local_service.extract_entities(text, ["QUANTITY"])

        qty_entities = [e for e in entities if e.entity_type == "QUANTITY"]
        assert len(qty_entities) == 0

    def test_confidence_scores(self, local_service):
        """Test that confidence scores are within valid range."""
        text = "Coffee from Colombia, 1000 tonnes"
        entities = local_service.extract_entities(
            text,
            ["COUNTRY", "QUANTITY", "COMMODITY"]
        )

        for entity in entities:
            assert 0 <= entity.confidence <= 1


# =============================================================================
# FUZZY MATCHING TESTS
# =============================================================================

class TestFuzzyMatching:
    """Test fuzzy matching assistance for entity resolution."""

    def test_assess_match_same_tax_id(self, local_service):
        """Test matching with identical tax ID."""
        candidate_a = {
            "name": "ABC Trading GmbH",
            "country_code": "DE",
            "tax_id": "DE123456789"
        }
        candidate_b = {
            "name": "ABC Trade",
            "country_code": "DE",
            "tax_id": "DE123456789"
        }

        result = local_service.assess_entity_match(candidate_a, candidate_b)

        assert result.is_same_entity is True
        assert result.confidence >= 0.9
        assert "Matching tax ID" in result.key_factors

    def test_assess_match_similar_names(self, local_service):
        """Test matching with similar names."""
        candidate_a = {
            "name": "Global Coffee Trading",
            "country_code": "NL"
        }
        candidate_b = {
            "name": "Global Coffee Trade",
            "country_code": "NL"
        }

        result = local_service.assess_entity_match(candidate_a, candidate_b)

        assert "Similar names" in result.key_factors
        assert "Same country" in result.key_factors

    def test_assess_match_different_countries(self, local_service):
        """Test matching with different countries."""
        candidate_a = {
            "name": "Test Company",
            "country_code": "DE"
        }
        candidate_b = {
            "name": "Test Company",
            "country_code": "FR"
        }

        result = local_service.assess_entity_match(candidate_a, candidate_b)

        assert "Same country" not in result.key_factors

    def test_assess_match_no_matching_factors(self, local_service):
        """Test matching with no matching factors."""
        candidate_a = {
            "name": "Alpha Corp",
            "country_code": "US"
        }
        candidate_b = {
            "name": "Beta Ltd",
            "country_code": "UK"
        }

        result = local_service.assess_entity_match(candidate_a, candidate_b)

        assert result.is_same_entity is False
        assert result.confidence < 0.7

    def test_suggestion_structure(self, local_service):
        """Test FuzzyMatchSuggestion structure."""
        candidate_a = {"name": "Test A", "country_code": "DE"}
        candidate_b = {"name": "Test B", "country_code": "DE"}

        result = local_service.assess_entity_match(candidate_a, candidate_b)

        assert isinstance(result, FuzzyMatchSuggestion)
        assert result.candidate_a_name == "Test A"
        assert result.candidate_b_name == "Test B"
        assert isinstance(result.is_same_entity, bool)
        assert isinstance(result.confidence, float)
        assert isinstance(result.reasoning, str)
        assert isinstance(result.key_factors, list)


# =============================================================================
# NATURAL LANGUAGE QUERY TESTS
# =============================================================================

class TestNaturalLanguageQueryParsing:
    """Test natural language query parsing."""

    def test_parse_country_query(self, local_service):
        """Test parsing query with country."""
        result = local_service.parse_natural_language_query(
            "Show all suppliers from Indonesia"
        )

        assert result.filters.get("country_code") == "ID"

    def test_parse_node_type_producer(self, local_service):
        """Test parsing query for producers."""
        result = local_service.parse_natural_language_query(
            "Find all producers in Brazil"
        )

        assert result.filters.get("node_type") == "PRODUCER"
        assert result.filters.get("country_code") == "BR"

    def test_parse_node_type_processor(self, local_service):
        """Test parsing query for processors."""
        result = local_service.parse_natural_language_query(
            "List all processor facilities"
        )

        assert result.filters.get("node_type") == "PROCESSOR"

    def test_parse_verification_status(self, local_service):
        """Test parsing query for verification status."""
        result = local_service.parse_natural_language_query(
            "Show unverified suppliers"
        )

        assert result.filters.get("verification_status") == "UNVERIFIED"

    def test_parse_risk_level_high(self, local_service):
        """Test parsing query for high risk."""
        result = local_service.parse_natural_language_query(
            "Find high risk suppliers"
        )

        assert result.filters.get("min_risk_score") == 0.7

    def test_parse_risk_level_low(self, local_service):
        """Test parsing query for low risk."""
        result = local_service.parse_natural_language_query(
            "Show low risk producers"
        )

        assert result.filters.get("max_risk_score") == 0.3

    def test_parse_tier_query(self, local_service):
        """Test parsing query with tier number."""
        result = local_service.parse_natural_language_query(
            "Show all tier 2 suppliers"
        )

        assert result.filters.get("tier") == 2

    def test_parse_commodity_query(self, local_service):
        """Test parsing query with commodity."""
        result = local_service.parse_natural_language_query(
            "Find all cocoa suppliers"
        )

        assert result.filters.get("commodity") == "COCOA"

    def test_parse_expired_certifications(self, local_service):
        """Test parsing query for expired certifications."""
        result = local_service.parse_natural_language_query(
            "Show suppliers with expired certification"
        )

        assert result.filters.get("expired_certifications") is True

    def test_parse_complex_query(self, local_service):
        """Test parsing complex multi-filter query."""
        result = local_service.parse_natural_language_query(
            "Find unverified coffee producers in Colombia tier 3"
        )

        assert result.filters.get("verification_status") == "UNVERIFIED"
        assert result.filters.get("node_type") == "PRODUCER"
        assert result.filters.get("commodity") == "COFFEE"
        assert result.filters.get("country_code") == "CO"
        assert result.filters.get("tier") == 3

    def test_parse_empty_query(self, local_service):
        """Test parsing empty query."""
        result = local_service.parse_natural_language_query("")

        assert result.filters == {}
        assert result.confidence < 0.5

    def test_parsed_query_structure(self, local_service):
        """Test ParsedQuery structure."""
        result = local_service.parse_natural_language_query("test query")

        assert isinstance(result, ParsedQuery)
        assert result.original_query == "test query"
        assert isinstance(result.interpretation, str)
        assert isinstance(result.filters, dict)
        assert isinstance(result.confidence, float)
        assert isinstance(result.suggested_refinements, list)


# =============================================================================
# GAP MATERIALITY TESTS
# =============================================================================

class TestGapMaterialityAssessment:
    """Test gap materiality assessment."""

    def test_assess_critical_gap(self, local_service):
        """Test assessment of critical gap type."""
        result = local_service.assess_gap_materiality(
            gap_type="MISSING_PLOT_DATA",
            gap_description="Producer has no geolocation data",
            context={"gap_id": "gap-123", "tier": 3}
        )

        assert result.is_material is True
        assert result.severity_assessment == "CRITICAL"

    def test_assess_high_gap(self, local_service):
        """Test assessment of high severity gap."""
        result = local_service.assess_gap_materiality(
            gap_type="UNVERIFIED_SUPPLIER",
            gap_description="Supplier not verified",
            context={"gap_id": "gap-456", "tier": 1}
        )

        assert result.is_material is True
        assert result.severity_assessment == "HIGH"

    def test_assess_medium_gap_close_tier(self, local_service):
        """Test assessment of medium gap close to importer."""
        result = local_service.assess_gap_materiality(
            gap_type="EXPIRED_CERTIFICATION",
            gap_description="Certification expired",
            context={"gap_id": "gap-789", "tier": 1}
        )

        assert result.is_material is True  # Close tier makes it material

    def test_assess_medium_gap_far_tier(self, local_service):
        """Test assessment of medium gap far from importer."""
        result = local_service.assess_gap_materiality(
            gap_type="EXPIRED_CERTIFICATION",
            gap_description="Certification expired",
            context={"gap_id": "gap-101", "tier": 5}
        )

        assert result.is_material is False  # Far tier, not material

    def test_assess_high_volume_gap(self, local_service):
        """Test assessment with high volume percentage."""
        result = local_service.assess_gap_materiality(
            gap_type="EXPIRED_CERTIFICATION",
            gap_description="Certification expired",
            context={"gap_id": "gap-102", "tier": 5, "volume_percentage": 25}
        )

        assert result.is_material is True  # High volume makes it material

    def test_materiality_structure(self, local_service):
        """Test GapMateriality structure."""
        result = local_service.assess_gap_materiality(
            gap_type="TEST_GAP",
            gap_description="Test description",
            context={"gap_id": "gap-test"}
        )

        assert isinstance(result, GapMateriality)
        assert result.gap_id == "gap-test"
        assert isinstance(result.is_material, bool)
        assert result.severity_assessment in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        assert isinstance(result.risk_factors, list)
        assert isinstance(result.recommended_action, str)
        assert 0 <= result.confidence <= 1


# =============================================================================
# PROVIDER FALLBACK TESTS
# =============================================================================

class TestProviderFallback:
    """Test provider fallback behavior."""

    def test_fallback_to_local_on_missing_package(self):
        """Test fallback to local when package not installed."""
        service = LLMIntegrationService(
            provider=LLMProvider.CLAUDE,
            api_key="test-key"
        )

        # Should fall back to local extraction
        entities = service.extract_entities(
            "Coffee from Brazil",
            ["COUNTRY", "COMMODITY"]
        )

        # Should still return results from local fallback
        assert isinstance(entities, list)

    def test_local_provider_no_client(self, local_service):
        """Test local provider doesn't create client."""
        client = local_service._get_client()
        assert client is None


# =============================================================================
# UTILITY METHOD TESTS
# =============================================================================

class TestUtilityMethods:
    """Test utility methods."""

    def test_validate_response_valid(self, local_service):
        """Test response validation with valid response."""
        response = {
            "entity_type": "COUNTRY",
            "value": "BR",
            "confidence": 0.9
        }
        schema = {
            "entity_type": str,
            "value": str,
            "confidence": float
        }

        is_valid, errors = local_service.validate_response(response, schema)

        assert is_valid is True
        assert errors == []

    def test_validate_response_missing_field(self, local_service):
        """Test response validation with missing field."""
        response = {
            "entity_type": "COUNTRY",
            "value": "BR"
            # missing confidence
        }
        schema = {
            "entity_type": str,
            "value": str,
            "confidence": float
        }

        is_valid, errors = local_service.validate_response(response, schema)

        assert is_valid is False
        assert len(errors) == 1
        assert "confidence" in errors[0]

    def test_validate_response_wrong_type(self, local_service):
        """Test response validation with wrong type."""
        response = {
            "entity_type": "COUNTRY",
            "value": "BR",
            "confidence": "high"  # Should be float
        }
        schema = {
            "entity_type": str,
            "value": str,
            "confidence": float
        }

        is_valid, errors = local_service.validate_response(response, schema)

        assert is_valid is False
        assert len(errors) == 1

    def test_get_confidence_level_high(self, local_service):
        """Test confidence level for high confidence."""
        level = local_service.get_confidence_level(0.95)
        assert level == "HIGH"

    def test_get_confidence_level_medium(self, local_service):
        """Test confidence level for medium confidence."""
        level = local_service.get_confidence_level(0.75)
        assert level == "MEDIUM"

    def test_get_confidence_level_low(self, local_service):
        """Test confidence level for low confidence."""
        level = local_service.get_confidence_level(0.55)
        assert level == "LOW"

    def test_get_confidence_level_very_low(self, local_service):
        """Test confidence level for very low confidence."""
        level = local_service.get_confidence_level(0.3)
        assert level == "VERY_LOW"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extract_case_insensitive_country(self, local_service):
        """Test country extraction is case insensitive."""
        text = "COFFEE FROM BRAZIL"
        entities = local_service.extract_entities(text, ["COUNTRY"])

        country_entities = [e for e in entities if e.entity_type == "COUNTRY"]
        assert len(country_entities) >= 1

    def test_extract_multiple_countries(self, local_service):
        """Test extraction of multiple countries."""
        text = "Trade route from Indonesia to Germany via Malaysia"
        entities = local_service.extract_entities(text, ["COUNTRY"])

        country_entities = [e for e in entities if e.entity_type == "COUNTRY"]
        country_codes = [e.value for e in country_entities]
        assert "ID" in country_codes
        assert "MY" in country_codes

    def test_parse_query_special_characters(self, local_service):
        """Test query parsing with special characters."""
        result = local_service.parse_natural_language_query(
            "Find suppliers in Côte d'Ivoire"
        )

        # Should handle without error even if not extracting perfectly
        assert isinstance(result, ParsedQuery)

    def test_default_model_claude(self):
        """Test default model for Claude provider."""
        service = LLMIntegrationService(provider=LLMProvider.CLAUDE)
        assert "claude" in service.model.lower()

    def test_default_model_openai(self):
        """Test default model for OpenAI provider."""
        service = LLMIntegrationService(provider=LLMProvider.OPENAI)
        assert service.model == "gpt-4"

    def test_default_model_local(self, local_service):
        """Test default model for local provider."""
        assert local_service.model == "local"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
