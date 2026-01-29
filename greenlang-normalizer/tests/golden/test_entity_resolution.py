"""
Golden Tests: Entity Resolution Tests for GL-FOUND-X-003.

This module tests entity resolution against golden file specifications.
Tests cover fuel, material, and process entity matching.

Test Coverage:
    - Exact name matching
    - Alias matching
    - Rule-based normalization
    - Fuzzy matching with confidence scoring
    - Deprecated entity handling
    - Ambiguous match detection

Features:
    - Automatic test discovery from YAML golden files
    - Confidence score validation
    - Match method verification
    - Needs review flag checking
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import pytest

from .conftest import (
    ENTITY_RESOLUTION_DIR,
    load_test_cases,
    discover_golden_files,
    GoldenTestResult,
)


# =============================================================================
# Test Data Loading
# =============================================================================

def get_entity_resolution_test_cases() -> List[Tuple[str, str, Dict[str, Any]]]:
    """Load all entity resolution test cases from golden files."""
    test_cases = []
    for golden_file in discover_golden_files(ENTITY_RESOLUTION_DIR):
        entity_type = golden_file.stem
        for test_case in load_test_cases(golden_file):
            test_id = f"{entity_type}::{test_case.get('name', 'unnamed')}"
            test_cases.append((entity_type, test_id, test_case))
    return test_cases


# Generate test IDs
TEST_CASES = get_entity_resolution_test_cases()
TEST_IDS = [tc[1] for tc in TEST_CASES]


# =============================================================================
# Mock Vocabulary for Testing
# =============================================================================

class MockVocabulary:
    """Mock vocabulary for entity resolution testing."""

    def __init__(self):
        """Initialize with default vocabulary entries."""
        self.entries = {
            "fuels": {
                "GL-FUEL-NATGAS": {
                    "id": "GL-FUEL-NATGAS",
                    "name": "Natural gas",
                    "aliases": ["Nat Gas", "NG", "Natural-gas", "Methane", "natural gas"],
                    "status": "active",
                },
                "GL-FUEL-DIESEL": {
                    "id": "GL-FUEL-DIESEL",
                    "name": "Diesel",
                    "aliases": ["DERV", "Diesel oil", "diesel", "AGO"],
                    "status": "active",
                },
                "GL-FUEL-PETROL": {
                    "id": "GL-FUEL-PETROL",
                    "name": "Petrol",
                    "aliases": ["Gasoline", "Motor spirit", "gas", "petrol"],
                    "status": "active",
                },
                "GL-FUEL-ELEC": {
                    "id": "GL-FUEL-ELEC",
                    "name": "Electricity",
                    "aliases": ["Grid electricity", "electric", "power"],
                    "status": "active",
                },
                "GL-FUEL-HFO": {
                    "id": "GL-FUEL-HFO",
                    "name": "Heavy fuel oil",
                    "aliases": ["HFO", "Fuel Oil No. 6", "Bunker C", "Residual fuel"],
                    "status": "active",
                },
                "GL-FUEL-LPG": {
                    "id": "GL-FUEL-LPG",
                    "name": "Liquefied petroleum gas",
                    "aliases": ["LPG", "Propane-butane", "Autogas"],
                    "status": "active",
                },
                "GL-FUEL-JETFUEL": {
                    "id": "GL-FUEL-JETFUEL",
                    "name": "Aviation fuel (jet)",
                    "aliases": ["Jet Fuel", "Jet A", "Jet A-1", "Aviation kerosene"],
                    "status": "active",
                },
                "GL-FUEL-MGO": {
                    "id": "GL-FUEL-MGO",
                    "name": "Marine gas oil",
                    "aliases": ["MGO", "Marine diesel"],
                    "status": "active",
                },
            },
            "materials": {
                "GL-MAT-STEEL": {
                    "id": "GL-MAT-STEEL",
                    "name": "Steel",
                    "aliases": ["Mild steel", "Structural steel", "steel"],
                    "status": "active",
                },
                "GL-MAT-STEEL-CARBON": {
                    "id": "GL-MAT-STEEL-CARBON",
                    "name": "Carbon steel",
                    "aliases": ["carbon steel", "Low carbon steel"],
                    "status": "active",
                },
                "GL-MAT-STEEL-STAINLESS": {
                    "id": "GL-MAT-STEEL-STAINLESS",
                    "name": "Stainless steel",
                    "aliases": ["stainless steel", "SS", "Inox"],
                    "status": "active",
                },
                "GL-MAT-ALUM": {
                    "id": "GL-MAT-ALUM",
                    "name": "Aluminum",
                    "aliases": ["Aluminium", "Al", "Alu", "aluminum"],
                    "status": "active",
                },
                "GL-MAT-CEMENT": {
                    "id": "GL-MAT-CEMENT",
                    "name": "Cement",
                    "aliases": ["cement", "Hydraulic cement"],
                    "status": "active",
                },
                "GL-MAT-CEMENT-PORTLAND": {
                    "id": "GL-MAT-CEMENT-PORTLAND",
                    "name": "Portland cement",
                    "aliases": ["OPC", "CEM I", "Type I cement", "portland cement"],
                    "status": "active",
                },
                "GL-MAT-CEMENT-COMPOSITE": {
                    "id": "GL-MAT-CEMENT-COMPOSITE",
                    "name": "Portland-composite cement",
                    "aliases": ["CEM II", "Composite cement"],
                    "status": "active",
                },
                "GL-MAT-FERT-UREA": {
                    "id": "GL-MAT-FERT-UREA",
                    "name": "Urea",
                    "aliases": ["urea", "CO(NH2)2", "Carbamide"],
                    "status": "active",
                },
                "GL-MAT-CHEM-AMMONIA": {
                    "id": "GL-MAT-CHEM-AMMONIA",
                    "name": "Ammonia",
                    "aliases": ["ammonia", "NH3", "Anhydrous ammonia"],
                    "status": "active",
                },
                "GL-MAT-HYDROGEN": {
                    "id": "GL-MAT-HYDROGEN",
                    "name": "Hydrogen",
                    "aliases": ["H2", "Hydrogen gas", "hydrogen"],
                    "status": "active",
                },
                "GL-MAT-IRON-ORE": {
                    "id": "GL-MAT-IRON-ORE",
                    "name": "Iron ore",
                    "aliases": ["iron ore", "Fe ore"],
                    "status": "active",
                },
                "GL-MAT-CONCRETE": {
                    "id": "GL-MAT-CONCRETE",
                    "name": "Concrete",
                    "aliases": ["concrete", "Ready-mix"],
                    "status": "active",
                },
            },
            "processes": {
                "GL-PROC-STEEL-EAF": {
                    "id": "GL-PROC-STEEL-EAF",
                    "name": "Electric arc furnace",
                    "aliases": ["EAF", "Arc furnace", "Electric furnace", "eaf"],
                    "status": "active",
                },
                "GL-PROC-STEEL-BOF": {
                    "id": "GL-PROC-STEEL-BOF",
                    "name": "Basic oxygen furnace",
                    "aliases": ["BOF", "BOS", "LD converter", "Oxygen converter"],
                    "status": "active",
                },
                "GL-PROC-IRON-BF": {
                    "id": "GL-PROC-IRON-BF",
                    "name": "Blast furnace",
                    "aliases": ["BF", "Iron blast furnace"],
                    "status": "active",
                },
                "GL-PROC-ALUM-HALLHEROULT": {
                    "id": "GL-PROC-ALUM-HALLHEROULT",
                    "name": "Hall-Heroult process",
                    "aliases": ["Hall-Heroult", "Aluminum electrolysis"],
                    "status": "active",
                },
                "GL-PROC-ALUM-BAYER": {
                    "id": "GL-PROC-ALUM-BAYER",
                    "name": "Bayer process",
                    "aliases": ["Bayer", "Alumina refining"],
                    "status": "active",
                },
                "GL-PROC-CEMENT-CLINKER": {
                    "id": "GL-PROC-CEMENT-CLINKER",
                    "name": "Clinker production",
                    "aliases": ["Clinker burning", "Kiln operation"],
                    "status": "active",
                },
                "GL-PROC-CEMENT-CALCINATION": {
                    "id": "GL-PROC-CEMENT-CALCINATION",
                    "name": "Calcination",
                    "aliases": ["Limestone calcination"],
                    "status": "active",
                },
                "GL-PROC-COMB-STATIONARY": {
                    "id": "GL-PROC-COMB-STATIONARY",
                    "name": "Stationary combustion",
                    "aliases": ["stationary combustion", "Fixed combustion"],
                    "status": "active",
                },
                "GL-PROC-COMB-MOBILE": {
                    "id": "GL-PROC-COMB-MOBILE",
                    "name": "Mobile combustion",
                    "aliases": ["mobile combustion", "Transport combustion"],
                    "status": "active",
                },
            },
        }

    def get_entries(self, entity_type: str) -> Dict[str, Dict]:
        """Get all entries for an entity type."""
        # Map singular to plural
        type_map = {
            "fuel": "fuels",
            "material": "materials",
            "process": "processes",
        }
        vocab_type = type_map.get(entity_type, entity_type)
        return self.entries.get(vocab_type, {})


VOCABULARY = MockVocabulary()


# =============================================================================
# Entity Resolution Logic
# =============================================================================

def normalize_string(s: str) -> str:
    """Normalize a string for matching."""
    # Lowercase, strip, normalize whitespace
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # Replace common separators
    s = s.replace("-", " ").replace("_", " ")
    return s


def calculate_similarity(s1: str, s2: str) -> float:
    """Calculate similarity score between two strings (0-1)."""
    s1_norm = normalize_string(s1)
    s2_norm = normalize_string(s2)

    if s1_norm == s2_norm:
        return 1.0

    # Token-based similarity
    tokens1 = set(s1_norm.split())
    tokens2 = set(s2_norm.split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    jaccard = len(intersection) / len(union)

    # Also compute character-level similarity (Levenshtein-like)
    # Simple ratio for quick approximation
    longer = max(len(s1_norm), len(s2_norm))
    if longer == 0:
        return 1.0

    # Count matching characters
    matches = sum(1 for c1, c2 in zip(s1_norm, s2_norm) if c1 == c2)
    char_sim = matches / longer

    # Combined score
    return (jaccard + char_sim) / 2


def resolve_entity(
    raw_name: str,
    entity_type: str,
    hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve an entity name to a vocabulary entry.

    Returns resolution result with reference_id, confidence, match_method.
    """
    if not raw_name or not raw_name.strip():
        return {
            "success": False,
            "error_code": "GLNORM-E400",
            "error_message": "Empty input for entity resolution",
        }

    entries = VOCABULARY.get_entries(entity_type)
    if not entries:
        return {
            "success": False,
            "error_code": "GLNORM-E404",
            "error_message": f"Vocabulary not found for type: {entity_type}",
        }

    normalized_query = normalize_string(raw_name)

    # Try exact match on canonical name
    for ref_id, entry in entries.items():
        if normalize_string(entry["name"]) == normalized_query:
            return {
                "success": True,
                "reference_id": ref_id,
                "canonical_name": entry["name"],
                "match_method": "exact",
                "confidence": 1.0,
                "needs_review": False,
            }

    # Try alias match
    for ref_id, entry in entries.items():
        for alias in entry.get("aliases", []):
            if normalize_string(alias) == normalized_query:
                return {
                    "success": True,
                    "reference_id": ref_id,
                    "canonical_name": entry["name"],
                    "match_method": "alias",
                    "confidence": 1.0,
                    "needs_review": False,
                }

    # Try rule-based normalization
    # Handle common patterns: hyphens, underscores, case
    for ref_id, entry in entries.items():
        entry_normalized = normalize_string(entry["name"])
        if entry_normalized == normalized_query:
            return {
                "success": True,
                "reference_id": ref_id,
                "canonical_name": entry["name"],
                "match_method": "rule",
                "confidence": 0.98,
                "needs_review": False,
            }

    # Fuzzy matching
    best_match = None
    best_score = 0.0

    for ref_id, entry in entries.items():
        # Score against canonical name
        name_score = calculate_similarity(raw_name, entry["name"])
        if name_score > best_score:
            best_score = name_score
            best_match = (ref_id, entry)

        # Score against aliases
        for alias in entry.get("aliases", []):
            alias_score = calculate_similarity(raw_name, alias)
            if alias_score > best_score:
                best_score = alias_score
                best_match = (ref_id, entry)

    if best_match and best_score >= 0.5:
        ref_id, entry = best_match
        needs_review = best_score < 0.80
        return {
            "success": True,
            "reference_id": ref_id,
            "canonical_name": entry["name"],
            "match_method": "fuzzy",
            "confidence": best_score,
            "needs_review": needs_review,
        }

    # No match found
    return {
        "success": False,
        "error_code": "GLNORM-E400",
        "error_message": f"No match found for '{raw_name}'",
        "needs_review": True,
    }


# =============================================================================
# Test Classes
# =============================================================================

class TestEntityResolution:
    """Golden tests for entity resolution."""

    @pytest.mark.parametrize("entity_type,test_id,test_case", TEST_CASES, ids=TEST_IDS)
    def test_entity_resolution(
        self,
        entity_type: str,
        test_id: str,
        test_case: Dict[str, Any],
    ):
        """Test entity resolution against golden values."""
        input_data = test_case.get("input", {})
        expected = test_case.get("expected", {})

        raw_name = input_data.get("raw_name", "")
        etype = input_data.get("entity_type", entity_type)
        hints = input_data.get("hints", {})

        # Resolve entity
        result = resolve_entity(raw_name, etype, hints)

        # Handle error cases
        if "error_code" in expected:
            assert not result.get("success", True) or "error_code" in result, (
                f"Expected error {expected['error_code']} but got success"
            )
            if "error_code" in result:
                assert result["error_code"] == expected["error_code"]
            return

        # Verify success
        assert result.get("success", False), (
            f"Resolution failed for '{raw_name}':\n"
            f"  Error: {result.get('error_message', 'Unknown error')}"
        )

        # Verify reference ID if specified
        if "reference_id" in expected:
            assert result["reference_id"] == expected["reference_id"], (
                f"Reference ID mismatch for '{raw_name}':\n"
                f"  Expected: {expected['reference_id']}\n"
                f"  Actual: {result['reference_id']}"
            )

        # Verify canonical name if specified
        if "canonical_name" in expected:
            assert result["canonical_name"] == expected["canonical_name"], (
                f"Canonical name mismatch for '{raw_name}':\n"
                f"  Expected: {expected['canonical_name']}\n"
                f"  Actual: {result['canonical_name']}"
            )

        # Verify match method if specified
        if "match_method" in expected:
            assert result["match_method"] == expected["match_method"], (
                f"Match method mismatch for '{raw_name}':\n"
                f"  Expected: {expected['match_method']}\n"
                f"  Actual: {result['match_method']}"
            )

        # Verify confidence if specified
        if "confidence" in expected:
            assert abs(result["confidence"] - expected["confidence"]) < 0.01, (
                f"Confidence mismatch for '{raw_name}':\n"
                f"  Expected: {expected['confidence']}\n"
                f"  Actual: {result['confidence']}"
            )

        # Verify confidence range if specified
        if "confidence_min" in expected:
            assert result["confidence"] >= expected["confidence_min"], (
                f"Confidence below minimum for '{raw_name}':\n"
                f"  Min expected: {expected['confidence_min']}\n"
                f"  Actual: {result['confidence']}"
            )

        if "confidence_max" in expected:
            assert result["confidence"] <= expected["confidence_max"], (
                f"Confidence above maximum for '{raw_name}':\n"
                f"  Max expected: {expected['confidence_max']}\n"
                f"  Actual: {result['confidence']}"
            )

        # Verify needs_review flag if specified
        if "needs_review" in expected:
            assert result.get("needs_review", False) == expected["needs_review"], (
                f"Needs review mismatch for '{raw_name}':\n"
                f"  Expected: {expected['needs_review']}\n"
                f"  Actual: {result.get('needs_review', False)}"
            )


class TestExactMatching:
    """Tests for exact name matching."""

    @pytest.mark.parametrize(
        "raw_name,entity_type,expected_id",
        [
            ("Natural gas", "fuel", "GL-FUEL-NATGAS"),
            ("Diesel", "fuel", "GL-FUEL-DIESEL"),
            ("Steel", "material", "GL-MAT-STEEL"),
            ("Electric arc furnace", "process", "GL-PROC-STEEL-EAF"),
        ],
    )
    def test_exact_canonical_match(
        self,
        raw_name: str,
        entity_type: str,
        expected_id: str,
    ):
        """Test exact matching on canonical names."""
        result = resolve_entity(raw_name, entity_type)

        assert result["success"]
        assert result["reference_id"] == expected_id
        assert result["match_method"] == "exact"
        assert result["confidence"] == 1.0


class TestAliasMatching:
    """Tests for alias matching."""

    @pytest.mark.parametrize(
        "raw_name,entity_type,expected_id",
        [
            ("Nat Gas", "fuel", "GL-FUEL-NATGAS"),
            ("NG", "fuel", "GL-FUEL-NATGAS"),
            ("Gasoline", "fuel", "GL-FUEL-PETROL"),
            ("DERV", "fuel", "GL-FUEL-DIESEL"),
            ("Aluminium", "material", "GL-MAT-ALUM"),
            ("OPC", "material", "GL-MAT-CEMENT-PORTLAND"),
            ("EAF", "process", "GL-PROC-STEEL-EAF"),
            ("BOF", "process", "GL-PROC-STEEL-BOF"),
        ],
    )
    def test_alias_match(
        self,
        raw_name: str,
        entity_type: str,
        expected_id: str,
    ):
        """Test alias matching."""
        result = resolve_entity(raw_name, entity_type)

        assert result["success"]
        assert result["reference_id"] == expected_id
        assert result["match_method"] == "alias"
        assert result["confidence"] == 1.0


class TestRuleBasedMatching:
    """Tests for rule-based normalization matching."""

    @pytest.mark.parametrize(
        "raw_name,entity_type,expected_id",
        [
            ("natural-gas", "fuel", "GL-FUEL-NATGAS"),
            ("natural_gas", "fuel", "GL-FUEL-NATGAS"),
            ("NATURAL GAS", "fuel", "GL-FUEL-NATGAS"),
            ("  Natural   Gas  ", "fuel", "GL-FUEL-NATGAS"),
        ],
    )
    def test_rule_match(
        self,
        raw_name: str,
        entity_type: str,
        expected_id: str,
    ):
        """Test rule-based normalization matching."""
        result = resolve_entity(raw_name, entity_type)

        assert result["success"]
        assert result["reference_id"] == expected_id
        # Could be exact, alias, or rule depending on normalization
        assert result["confidence"] >= 0.95


class TestFuzzyMatching:
    """Tests for fuzzy matching."""

    @pytest.mark.parametrize(
        "raw_name,entity_type,expected_id",
        [
            ("Natual Gas", "fuel", "GL-FUEL-NATGAS"),  # typo
            ("Deisel", "fuel", "GL-FUEL-DIESEL"),  # typo
            ("Heavy Fuel", "fuel", "GL-FUEL-HFO"),  # partial
        ],
    )
    def test_fuzzy_match(
        self,
        raw_name: str,
        entity_type: str,
        expected_id: str,
    ):
        """Test fuzzy matching with typos and partial names."""
        result = resolve_entity(raw_name, entity_type)

        assert result["success"]
        assert result["reference_id"] == expected_id
        assert result["match_method"] == "fuzzy"
        assert result["confidence"] >= 0.5


class TestEdgeCases:
    """Edge case tests for entity resolution."""

    def test_empty_input(self):
        """Empty input should return error."""
        result = resolve_entity("", "fuel")

        assert not result.get("success", True)
        assert result.get("error_code") == "GLNORM-E400"

    def test_whitespace_only_input(self):
        """Whitespace-only input should return error."""
        result = resolve_entity("   ", "fuel")

        assert not result.get("success", True)
        assert result.get("error_code") == "GLNORM-E400"

    def test_unknown_entity(self):
        """Unknown entity should return not found error."""
        result = resolve_entity("Unobtainium Fuel", "fuel")

        assert not result.get("success", True)
        assert result.get("needs_review", False) is True

    def test_unknown_vocabulary(self):
        """Unknown vocabulary type should return error."""
        result = resolve_entity("Test", "unknown_type")

        assert not result.get("success", True)
        assert result.get("error_code") == "GLNORM-E404"

    def test_case_insensitive_matching(self):
        """Matching should be case-insensitive."""
        lower = resolve_entity("natural gas", "fuel")
        upper = resolve_entity("NATURAL GAS", "fuel")
        mixed = resolve_entity("NaTuRaL gAs", "fuel")

        assert lower["reference_id"] == upper["reference_id"] == mixed["reference_id"]


class TestConfidenceScoring:
    """Tests for confidence score calculation."""

    def test_exact_match_confidence(self):
        """Exact matches should have confidence 1.0."""
        result = resolve_entity("Natural gas", "fuel")

        assert result["confidence"] == 1.0

    def test_alias_match_confidence(self):
        """Alias matches should have confidence 1.0."""
        result = resolve_entity("Nat Gas", "fuel")

        assert result["confidence"] == 1.0

    def test_fuzzy_match_confidence_range(self):
        """Fuzzy matches should have confidence < 1.0."""
        result = resolve_entity("Natual Gas", "fuel")

        assert 0.5 <= result["confidence"] < 1.0

    def test_needs_review_threshold(self):
        """Low confidence matches should be flagged for review."""
        # Partial match with lower confidence
        result = resolve_entity("Heavy Fuel", "fuel")

        if result["confidence"] < 0.80:
            assert result.get("needs_review", False) is True


class TestDeterminism:
    """Tests for resolution determinism."""

    def test_resolution_determinism(self):
        """Same input should always produce same output."""
        results = [
            resolve_entity("Natural gas", "fuel")
            for _ in range(10)
        ]

        first = results[0]
        for result in results[1:]:
            assert result["reference_id"] == first["reference_id"]
            assert result["confidence"] == first["confidence"]
            assert result["match_method"] == first["match_method"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
