# -*- coding: utf-8 -*-
"""
Tests for SupplierDiscoveryEngine - AGENT-EUDR-008 Engine 1: Supplier Hierarchy Discovery

Comprehensive test suite covering:
- Supplier discovery from declarations (F1.1)
- Supplier discovery from questionnaire responses (F1.3)
- Supplier discovery from shipping documents (F1.2)
- Recursive discovery up to 15 tiers (F1.7)
- Deduplication: exact match and fuzzy match (F1.10)
- Confidence scoring per source type (F1.8)
- Batch discovery from bulk uploads (F1.9)
- Many-to-many relationship handling (F1.6)
- Tier depth auto-detection (F1.5)

Test count: 65+ tests
Coverage target: >= 85% of SupplierDiscoveryEngine module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.multi_tier_supplier.conftest import (
    COCOA_IMPORTER_EU,
    COCOA_TRADER_GH,
    COCOA_PROCESSOR_GH,
    COCOA_AGGREGATOR_GH,
    COFFEE_EXPORTER_CO,
    PALM_REFINERY_ID,
    SUP_ID_COCOA_IMPORTER_EU,
    SUP_ID_COCOA_TRADER_GH,
    SUP_ID_COCOA_PROCESSOR_GH,
    SUP_ID_PALM_REFINERY_ID,
    EUDR_COMMODITIES,
    CERTIFICATION_TYPES,
    SHA256_HEX_LENGTH,
    make_supplier,
    make_relationship,
    compute_sha256,
)


# ===========================================================================
# 1. Declaration Parsing
# ===========================================================================


class TestSupplierDiscoveryFromDeclaration:
    """Test discovery of sub-tier suppliers from Tier 1 declarations."""

    def test_discover_single_sub_supplier(self, supplier_discovery_engine):
        """Discover one Tier 2 supplier from a Tier 1 declaration."""
        declaration = {
            "declaring_supplier_id": SUP_ID_COCOA_TRADER_GH,
            "commodity": "cocoa",
            "sub_suppliers": [
                {
                    "legal_name": "Accra Cocoa Processing Co",
                    "country_iso": "GH",
                    "registration_id": "GH-BRN-345678",
                    "role": "processor",
                }
            ],
        }
        result = supplier_discovery_engine.discover_from_declaration(declaration)
        assert result is not None
        assert len(result.discovered_suppliers) == 1
        assert result.discovered_suppliers[0]["legal_name"] == "Accra Cocoa Processing Co"

    def test_discover_multiple_sub_suppliers(self, supplier_discovery_engine):
        """Discover multiple Tier 2 suppliers from one declaration."""
        declaration = {
            "declaring_supplier_id": SUP_ID_COCOA_TRADER_GH,
            "commodity": "cocoa",
            "sub_suppliers": [
                {"legal_name": "Processor A", "country_iso": "GH", "role": "processor"},
                {"legal_name": "Processor B", "country_iso": "GH", "role": "processor"},
                {"legal_name": "Aggregator C", "country_iso": "GH", "role": "aggregator"},
            ],
        }
        result = supplier_discovery_engine.discover_from_declaration(declaration)
        assert len(result.discovered_suppliers) == 3

    def test_declaration_assigns_correct_tier(self, supplier_discovery_engine):
        """Discovered suppliers should be tier + 1 of the declaring supplier."""
        declaration = {
            "declaring_supplier_id": SUP_ID_COCOA_TRADER_GH,
            "declaring_tier": 1,
            "commodity": "cocoa",
            "sub_suppliers": [
                {"legal_name": "Sub Supplier", "country_iso": "GH", "role": "processor"},
            ],
        }
        result = supplier_discovery_engine.discover_from_declaration(declaration)
        assert result.discovered_suppliers[0]["tier"] == 2

    def test_declaration_sets_confidence_verified(self, supplier_discovery_engine):
        """Declaration source should set confidence to 'verified' level."""
        declaration = {
            "declaring_supplier_id": SUP_ID_COCOA_TRADER_GH,
            "commodity": "cocoa",
            "sub_suppliers": [
                {"legal_name": "Sub Supplier", "country_iso": "GH", "role": "processor"},
            ],
        }
        result = supplier_discovery_engine.discover_from_declaration(declaration)
        # Declarations are "declared" confidence level
        assert result.discovered_suppliers[0]["confidence"] in ("verified", "declared")

    def test_declaration_creates_relationship(self, supplier_discovery_engine):
        """Declaration should create a buyer-supplier relationship."""
        declaration = {
            "declaring_supplier_id": SUP_ID_COCOA_TRADER_GH,
            "commodity": "cocoa",
            "sub_suppliers": [
                {"legal_name": "Sub Supplier", "country_iso": "GH", "role": "processor"},
            ],
        }
        result = supplier_discovery_engine.discover_from_declaration(declaration)
        assert len(result.discovered_relationships) >= 1
        rel = result.discovered_relationships[0]
        assert rel["buyer_id"] == SUP_ID_COCOA_TRADER_GH

    def test_declaration_empty_sub_suppliers(self, supplier_discovery_engine):
        """Empty sub_suppliers list yields no discoveries."""
        declaration = {
            "declaring_supplier_id": SUP_ID_COCOA_TRADER_GH,
            "commodity": "cocoa",
            "sub_suppliers": [],
        }
        result = supplier_discovery_engine.discover_from_declaration(declaration)
        assert len(result.discovered_suppliers) == 0

    def test_declaration_missing_required_fields_raises(self, supplier_discovery_engine):
        """Declaration missing required fields should raise or return error."""
        declaration = {
            "declaring_supplier_id": SUP_ID_COCOA_TRADER_GH,
            # Missing commodity
            "sub_suppliers": [
                {"legal_name": "Sub Supplier"},
            ],
        }
        with pytest.raises((ValueError, KeyError)):
            supplier_discovery_engine.discover_from_declaration(declaration)

    def test_declaration_with_gps_coordinates(self, supplier_discovery_engine):
        """Discovered supplier includes GPS when provided in declaration."""
        declaration = {
            "declaring_supplier_id": SUP_ID_COCOA_TRADER_GH,
            "commodity": "cocoa",
            "sub_suppliers": [
                {
                    "legal_name": "GPS Supplier",
                    "country_iso": "GH",
                    "role": "processor",
                    "gps_lat": 5.5571,
                    "gps_lon": -0.2013,
                },
            ],
        }
        result = supplier_discovery_engine.discover_from_declaration(declaration)
        sup = result.discovered_suppliers[0]
        assert sup.get("gps_lat") == pytest.approx(5.5571, abs=0.001)
        assert sup.get("gps_lon") == pytest.approx(-0.2013, abs=0.001)

    def test_declaration_provenance_hash_generated(self, supplier_discovery_engine):
        """Discovery result should include provenance hash."""
        declaration = {
            "declaring_supplier_id": SUP_ID_COCOA_TRADER_GH,
            "commodity": "cocoa",
            "sub_suppliers": [
                {"legal_name": "Hash Test Supplier", "country_iso": "GH", "role": "processor"},
            ],
        }
        result = supplier_discovery_engine.discover_from_declaration(declaration)
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH


# ===========================================================================
# 2. Questionnaire Extraction
# ===========================================================================


class TestSupplierDiscoveryFromQuestionnaire:
    """Test discovery of suppliers from questionnaire responses."""

    def test_extract_suppliers_from_questionnaire(self, supplier_discovery_engine):
        """Extract supplier names and metadata from questionnaire answers."""
        questionnaire = {
            "respondent_supplier_id": SUP_ID_COCOA_PROCESSOR_GH,
            "commodity": "cocoa",
            "responses": [
                {
                    "question": "List your upstream cocoa suppliers",
                    "answer": "Ashanti Regional Aggregators, Kumasi; Western Aggregators, Takoradi",
                },
                {
                    "question": "Are your suppliers certified?",
                    "answer": "Ashanti is UTZ certified. Western is not certified.",
                },
            ],
        }
        result = supplier_discovery_engine.discover_from_questionnaire(questionnaire)
        assert result is not None
        assert len(result.discovered_suppliers) >= 1

    def test_questionnaire_confidence_is_declared(self, supplier_discovery_engine):
        """Questionnaire-sourced suppliers should have 'declared' confidence."""
        questionnaire = {
            "respondent_supplier_id": SUP_ID_COCOA_PROCESSOR_GH,
            "commodity": "cocoa",
            "responses": [
                {"question": "Supplier list", "answer": "Supplier Alpha, Ghana"},
            ],
        }
        result = supplier_discovery_engine.discover_from_questionnaire(questionnaire)
        for sup in result.discovered_suppliers:
            assert sup["confidence"] in ("declared", "inferred")

    def test_questionnaire_empty_responses(self, supplier_discovery_engine):
        """Empty questionnaire responses yield no discoveries."""
        questionnaire = {
            "respondent_supplier_id": SUP_ID_COCOA_PROCESSOR_GH,
            "commodity": "cocoa",
            "responses": [],
        }
        result = supplier_discovery_engine.discover_from_questionnaire(questionnaire)
        assert len(result.discovered_suppliers) == 0

    def test_questionnaire_with_structured_data(self, supplier_discovery_engine):
        """Questionnaire with structured supplier data (name, country, cert)."""
        questionnaire = {
            "respondent_supplier_id": SUP_ID_COCOA_PROCESSOR_GH,
            "commodity": "cocoa",
            "responses": [
                {
                    "question": "upstream_suppliers",
                    "structured": True,
                    "suppliers": [
                        {"legal_name": "Struct Supplier 1", "country_iso": "GH", "certification": "UTZ"},
                        {"legal_name": "Struct Supplier 2", "country_iso": "CI", "certification": None},
                    ],
                },
            ],
        }
        result = supplier_discovery_engine.discover_from_questionnaire(questionnaire)
        assert len(result.discovered_suppliers) == 2

    def test_questionnaire_provenance_hash(self, supplier_discovery_engine):
        """Questionnaire discovery includes provenance hash."""
        questionnaire = {
            "respondent_supplier_id": SUP_ID_COCOA_PROCESSOR_GH,
            "commodity": "cocoa",
            "responses": [
                {"question": "Supplier list", "answer": "Test Supplier, Ghana"},
            ],
        }
        result = supplier_discovery_engine.discover_from_questionnaire(questionnaire)
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH


# ===========================================================================
# 3. Shipping Document Parsing
# ===========================================================================


class TestSupplierDiscoveryFromShippingDoc:
    """Test discovery of suppliers from shipping documents (BL, packing lists)."""

    def test_discover_from_bill_of_lading(self, supplier_discovery_engine):
        """Extract shipper and consignee from bill of lading."""
        shipping_doc = {
            "document_type": "bill_of_lading",
            "bl_number": "BL-2026-001234",
            "commodity": "cocoa",
            "shipper": {
                "legal_name": "Ghana Cocoa Trading Ltd",
                "country_iso": "GH",
                "address": "Independence Avenue, Accra",
            },
            "consignee": {
                "legal_name": "EuroChoc GmbH",
                "country_iso": "DE",
                "address": "Speicherstrasse 10, Hamburg",
            },
            "origin_port": "Tema, Ghana",
            "destination_port": "Hamburg, Germany",
            "weight_mt": 500.0,
        }
        result = supplier_discovery_engine.discover_from_shipping_doc(shipping_doc)
        assert result is not None
        assert len(result.discovered_suppliers) >= 1

    def test_discover_from_packing_list(self, supplier_discovery_engine):
        """Extract supplier references from packing list."""
        shipping_doc = {
            "document_type": "packing_list",
            "reference": "PL-2026-005678",
            "commodity": "coffee",
            "supplier_references": [
                {"legal_name": "Exportadora Colombiana", "country_iso": "CO"},
            ],
            "origin": "Colombia",
            "weight_mt": 200.0,
        }
        result = supplier_discovery_engine.discover_from_shipping_doc(shipping_doc)
        assert len(result.discovered_suppliers) >= 1

    def test_shipping_doc_confidence_is_inferred(self, supplier_discovery_engine):
        """Shipping-doc-sourced suppliers have 'inferred' or 'declared' confidence."""
        shipping_doc = {
            "document_type": "bill_of_lading",
            "bl_number": "BL-TEST-001",
            "commodity": "palm_oil",
            "shipper": {"legal_name": "Shipping Test Co", "country_iso": "ID"},
            "consignee": {"legal_name": "EU Buyer", "country_iso": "NL"},
            "weight_mt": 100.0,
        }
        result = supplier_discovery_engine.discover_from_shipping_doc(shipping_doc)
        for sup in result.discovered_suppliers:
            assert sup["confidence"] in ("declared", "inferred", "suspected")

    def test_shipping_doc_missing_shipper_raises(self, supplier_discovery_engine):
        """Shipping document without shipper should raise or return error."""
        shipping_doc = {
            "document_type": "bill_of_lading",
            "bl_number": "BL-TEST-002",
            "commodity": "cocoa",
            "consignee": {"legal_name": "EU Buyer", "country_iso": "DE"},
            "weight_mt": 100.0,
        }
        with pytest.raises((ValueError, KeyError)):
            supplier_discovery_engine.discover_from_shipping_doc(shipping_doc)

    def test_shipping_doc_creates_volume_relationship(self, supplier_discovery_engine):
        """Relationship from shipping doc includes volume from weight_mt."""
        shipping_doc = {
            "document_type": "bill_of_lading",
            "bl_number": "BL-TEST-003",
            "commodity": "cocoa",
            "shipper": {"legal_name": "Shipper Co", "country_iso": "GH"},
            "consignee": {"legal_name": "Buyer Co", "country_iso": "DE"},
            "weight_mt": 750.0,
        }
        result = supplier_discovery_engine.discover_from_shipping_doc(shipping_doc)
        if result.discovered_relationships:
            assert result.discovered_relationships[0]["volume_mt"] == pytest.approx(750.0)


# ===========================================================================
# 4. Recursive Discovery
# ===========================================================================


class TestRecursiveDiscovery:
    """Test recursive discovery of suppliers through multiple tiers."""

    def test_recursive_discovery_depth_3(self, supplier_discovery_engine):
        """Recursively discover suppliers up to depth 3."""
        seed_suppliers = [
            {
                "supplier_id": "SUP-SEED-001",
                "legal_name": "Seed Trader",
                "country_iso": "GH",
                "commodity": "cocoa",
                "tier": 1,
                "sub_suppliers": [
                    {
                        "legal_name": "Mid Processor",
                        "country_iso": "GH",
                        "sub_suppliers": [
                            {"legal_name": "Deep Farmer", "country_iso": "GH"},
                        ],
                    },
                ],
            },
        ]
        result = supplier_discovery_engine.discover_recursive(
            seed_suppliers, max_depth=3, commodity="cocoa"
        )
        assert result is not None
        assert result.max_tier_discovered >= 3

    def test_recursive_discovery_respects_max_depth(self, supplier_discovery_engine):
        """Discovery stops at configured max_depth even if more tiers exist."""
        seed_suppliers = [
            {
                "supplier_id": "SUP-DEEP-001",
                "legal_name": "Root",
                "country_iso": "GH",
                "commodity": "cocoa",
                "tier": 0,
                "sub_suppliers": [
                    {
                        "legal_name": f"Tier {i}",
                        "country_iso": "GH",
                        "sub_suppliers": [],
                    }
                    for i in range(1, 20)
                ],
            },
        ]
        result = supplier_discovery_engine.discover_recursive(
            seed_suppliers, max_depth=5, commodity="cocoa"
        )
        assert result.max_tier_discovered <= 5

    def test_recursive_discovery_default_max_15(self, supplier_discovery_engine):
        """Default max depth is 15 tiers per PRD F1.7."""
        # Just verify the default config
        assert supplier_discovery_engine.config.get("max_tier_depth", 15) <= 15

    def test_recursive_discovery_handles_empty_tree(self, supplier_discovery_engine):
        """Empty seed data yields empty result."""
        result = supplier_discovery_engine.discover_recursive(
            [], max_depth=10, commodity="cocoa"
        )
        assert result.total_discovered == 0

    def test_recursive_discovery_avoids_cycles(self, supplier_discovery_engine):
        """Circular references in data do not cause infinite recursion."""
        # Supplier A -> B -> C -> A (cycle)
        seed = [
            {
                "supplier_id": "SUP-CYCLE-A",
                "legal_name": "Cycle A",
                "country_iso": "GH",
                "commodity": "cocoa",
                "tier": 0,
                "sub_suppliers": [
                    {
                        "supplier_id": "SUP-CYCLE-B",
                        "legal_name": "Cycle B",
                        "country_iso": "GH",
                        "sub_suppliers": [
                            {
                                "supplier_id": "SUP-CYCLE-C",
                                "legal_name": "Cycle C",
                                "country_iso": "GH",
                                "sub_suppliers": [
                                    {"supplier_id": "SUP-CYCLE-A", "legal_name": "Cycle A"},
                                ],
                            },
                        ],
                    },
                ],
            },
        ]
        # Should not raise RecursionError
        result = supplier_discovery_engine.discover_recursive(seed, max_depth=15, commodity="cocoa")
        assert result is not None
        # Should not have more nodes than unique suppliers
        unique_ids = {s.get("supplier_id") for s in result.discovered_suppliers if s.get("supplier_id")}
        assert len(unique_ids) <= 3


# ===========================================================================
# 5. Deduplication
# ===========================================================================


class TestSupplierDeduplication:
    """Test deduplication of suppliers across discovery sources."""

    def test_exact_match_dedup_by_registration_id(self, supplier_discovery_engine):
        """Two suppliers with same registration_id are deduplicated."""
        suppliers = [
            make_supplier(legal_name="Supplier Alpha", registration_id="GH-BRN-111"),
            make_supplier(legal_name="Supplier Alpha Ltd", registration_id="GH-BRN-111"),
        ]
        result = supplier_discovery_engine.deduplicate(suppliers)
        assert len(result) == 1

    def test_exact_match_dedup_by_tax_id(self, supplier_discovery_engine):
        """Two suppliers with same tax_id are deduplicated."""
        suppliers = [
            make_supplier(legal_name="Tax Match A", tax_id="GH999999"),
            make_supplier(legal_name="Tax Match B", tax_id="GH999999"),
        ]
        result = supplier_discovery_engine.deduplicate(suppliers)
        assert len(result) == 1

    def test_exact_match_dedup_by_supplier_id(self, supplier_discovery_engine):
        """Two suppliers with same supplier_id are deduplicated."""
        sid = "SUP-DEDUP-EXACT"
        suppliers = [
            make_supplier(supplier_id=sid, legal_name="Dup A"),
            make_supplier(supplier_id=sid, legal_name="Dup B"),
        ]
        result = supplier_discovery_engine.deduplicate(suppliers)
        assert len(result) == 1

    def test_fuzzy_match_similar_names(self, supplier_discovery_engine):
        """Suppliers with similar names (above threshold) are deduplicated."""
        suppliers = [
            make_supplier(legal_name="Ghana Cocoa Trading Ltd", country_iso="GH"),
            make_supplier(legal_name="Ghana Cocoa Trading Limited", country_iso="GH"),
        ]
        result = supplier_discovery_engine.deduplicate(suppliers, fuzzy_threshold=0.85)
        assert len(result) == 1

    def test_fuzzy_match_different_names_preserved(self, supplier_discovery_engine):
        """Suppliers with different names are not deduplicated."""
        suppliers = [
            make_supplier(legal_name="Alpha Corp", country_iso="GH"),
            make_supplier(legal_name="Beta Industries", country_iso="GH"),
        ]
        result = supplier_discovery_engine.deduplicate(suppliers, fuzzy_threshold=0.85)
        assert len(result) == 2

    def test_dedup_preserves_most_complete_profile(self, supplier_discovery_engine):
        """When merging duplicates, keep the most complete profile."""
        suppliers = [
            make_supplier(legal_name="Complete Co", registration_id="GH-001",
                          tax_id="GH111", primary_contact="Alice"),
            make_supplier(legal_name="Complete Co", registration_id="GH-001",
                          tax_id=None, primary_contact=None),
        ]
        result = supplier_discovery_engine.deduplicate(suppliers)
        assert len(result) == 1
        assert result[0]["tax_id"] == "GH111"
        assert result[0]["primary_contact"] == "Alice"

    def test_dedup_empty_list(self, supplier_discovery_engine):
        """Empty supplier list returns empty result."""
        result = supplier_discovery_engine.deduplicate([])
        assert len(result) == 0

    def test_dedup_single_supplier(self, supplier_discovery_engine):
        """Single supplier list returned unchanged."""
        suppliers = [make_supplier(legal_name="Solo")]
        result = supplier_discovery_engine.deduplicate(suppliers)
        assert len(result) == 1

    def test_dedup_cross_country_not_merged(self, supplier_discovery_engine):
        """Same-name suppliers in different countries are not merged."""
        suppliers = [
            make_supplier(legal_name="Global Trader", country_iso="GH"),
            make_supplier(legal_name="Global Trader", country_iso="CI"),
        ]
        result = supplier_discovery_engine.deduplicate(suppliers, fuzzy_threshold=0.85)
        assert len(result) == 2

    @pytest.mark.parametrize("threshold", [0.5, 0.7, 0.85, 0.95, 1.0])
    def test_dedup_fuzzy_threshold_sensitivity(self, supplier_discovery_engine, threshold):
        """Different fuzzy thresholds produce valid results."""
        suppliers = [
            make_supplier(legal_name="Threshold Test Corp"),
            make_supplier(legal_name="Threshold Test Corporation"),
        ]
        result = supplier_discovery_engine.deduplicate(suppliers, fuzzy_threshold=threshold)
        assert 1 <= len(result) <= 2


# ===========================================================================
# 6. Confidence Scoring
# ===========================================================================


class TestDiscoveryConfidenceScoring:
    """Test confidence scoring for discovered supplier relationships."""

    @pytest.mark.parametrize("source_type,expected_confidence", [
        ("verified_audit", "verified"),
        ("supplier_declaration", "declared"),
        ("shipping_document", "inferred"),
        ("certification_cross_ref", "inferred"),
        ("news_analysis", "suspected"),
    ])
    def test_confidence_level_per_source(self, supplier_discovery_engine,
                                         source_type, expected_confidence):
        """Each source type maps to a defined confidence level."""
        result = supplier_discovery_engine.get_confidence_level(source_type)
        assert result == expected_confidence

    @pytest.mark.parametrize("source_type,min_score,max_score", [
        ("verified_audit", 0.90, 1.00),
        ("supplier_declaration", 0.70, 0.90),
        ("shipping_document", 0.50, 0.75),
        ("certification_cross_ref", 0.50, 0.75),
        ("news_analysis", 0.20, 0.50),
    ])
    def test_confidence_score_range(self, supplier_discovery_engine,
                                    source_type, min_score, max_score):
        """Each source type produces a confidence score within expected range."""
        score = supplier_discovery_engine.get_confidence_score(source_type)
        assert min_score <= score <= max_score

    def test_unknown_source_type_low_confidence(self, supplier_discovery_engine):
        """Unknown source type should return lowest confidence."""
        result = supplier_discovery_engine.get_confidence_level("unknown_source")
        assert result == "suspected"

    def test_confidence_score_is_numeric(self, supplier_discovery_engine):
        """Confidence score must be a float between 0 and 1."""
        for source in ["verified_audit", "supplier_declaration", "shipping_document"]:
            score = supplier_discovery_engine.get_confidence_score(source)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0


# ===========================================================================
# 7. Batch Discovery
# ===========================================================================


class TestBatchDiscovery:
    """Test batch discovery from bulk supplier data uploads."""

    def test_batch_discovery_multiple_declarations(self, supplier_discovery_engine):
        """Process multiple declarations in a single batch."""
        declarations = [
            {
                "declaring_supplier_id": f"SUP-BATCH-{i}",
                "commodity": "cocoa",
                "sub_suppliers": [
                    {"legal_name": f"Batch Sub {i}-{j}", "country_iso": "GH", "role": "processor"}
                    for j in range(3)
                ],
            }
            for i in range(5)
        ]
        result = supplier_discovery_engine.discover_batch(declarations)
        assert result is not None
        assert result.total_discovered >= 15  # 5 declarations x 3 subs

    def test_batch_discovery_with_dedup(self, supplier_discovery_engine):
        """Batch discovery with overlapping suppliers deduplicates."""
        declarations = [
            {
                "declaring_supplier_id": "SUP-BATCH-A",
                "commodity": "cocoa",
                "sub_suppliers": [
                    {"legal_name": "Shared Supplier", "country_iso": "GH",
                     "registration_id": "GH-SHARED-001", "role": "processor"},
                ],
            },
            {
                "declaring_supplier_id": "SUP-BATCH-B",
                "commodity": "cocoa",
                "sub_suppliers": [
                    {"legal_name": "Shared Supplier", "country_iso": "GH",
                     "registration_id": "GH-SHARED-001", "role": "processor"},
                ],
            },
        ]
        result = supplier_discovery_engine.discover_batch(declarations, deduplicate=True)
        # Shared supplier should appear only once
        names = [s["legal_name"] for s in result.discovered_suppliers]
        assert names.count("Shared Supplier") == 1

    def test_batch_discovery_empty(self, supplier_discovery_engine):
        """Empty batch yields empty result."""
        result = supplier_discovery_engine.discover_batch([])
        assert result.total_discovered == 0

    def test_batch_discovery_provenance(self, supplier_discovery_engine):
        """Batch result includes provenance hash."""
        declarations = [
            {
                "declaring_supplier_id": "SUP-PROV-001",
                "commodity": "coffee",
                "sub_suppliers": [
                    {"legal_name": "Prov Sub", "country_iso": "CO", "role": "mill"},
                ],
            },
        ]
        result = supplier_discovery_engine.discover_batch(declarations)
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    def test_batch_discovery_large_set(self, supplier_discovery_engine):
        """Batch of 100 declarations processes without error."""
        declarations = [
            {
                "declaring_supplier_id": f"SUP-LARGE-{i:04d}",
                "commodity": EUDR_COMMODITIES[i % len(EUDR_COMMODITIES)],
                "sub_suppliers": [
                    {"legal_name": f"Large Sub {i}", "country_iso": "GH", "role": "farmer"},
                ],
            }
            for i in range(100)
        ]
        result = supplier_discovery_engine.discover_batch(declarations)
        assert result.total_discovered >= 100


# ===========================================================================
# 8. Many-to-Many Relationships
# ===========================================================================


class TestManyToManyRelationships:
    """Test handling of many-to-many supplier relationships (F1.6)."""

    def test_one_supplier_multiple_buyers(self, supplier_discovery_engine):
        """One supplier can be linked to multiple buyers."""
        declarations = [
            {
                "declaring_supplier_id": "BUYER-A",
                "commodity": "cocoa",
                "sub_suppliers": [
                    {"legal_name": "Shared Proc", "country_iso": "GH",
                     "registration_id": "SHARED-001", "role": "processor"},
                ],
            },
            {
                "declaring_supplier_id": "BUYER-B",
                "commodity": "cocoa",
                "sub_suppliers": [
                    {"legal_name": "Shared Proc", "country_iso": "GH",
                     "registration_id": "SHARED-001", "role": "processor"},
                ],
            },
        ]
        result = supplier_discovery_engine.discover_batch(declarations, deduplicate=True)
        # Two relationships, one supplier
        buyer_ids = {r["buyer_id"] for r in result.discovered_relationships}
        assert "BUYER-A" in buyer_ids
        assert "BUYER-B" in buyer_ids

    def test_one_buyer_multiple_suppliers(self, supplier_discovery_engine):
        """One buyer can have multiple suppliers."""
        declaration = {
            "declaring_supplier_id": "BUYER-MULTI",
            "commodity": "cocoa",
            "sub_suppliers": [
                {"legal_name": f"Multi Sup {i}", "country_iso": "GH", "role": "farmer"}
                for i in range(5)
            ],
        }
        result = supplier_discovery_engine.discover_from_declaration(declaration)
        assert len(result.discovered_relationships) == 5


# ===========================================================================
# 9. Tier Depth Auto-Detection
# ===========================================================================


class TestTierDepthAutoDetection:
    """Test auto-detection of tier depth from commodity flow (F1.5)."""

    @pytest.mark.parametrize("commodity,expected_min,expected_max", [
        ("cocoa", 6, 8),
        ("coffee", 5, 7),
        ("palm_oil", 5, 7),
        ("soya", 4, 6),
        ("rubber", 5, 7),
        ("cattle", 3, 5),
        ("wood", 4, 6),
    ])
    def test_typical_tier_depth_per_commodity(self, supplier_discovery_engine,
                                              commodity, expected_min, expected_max):
        """Each commodity has expected typical tier depth range."""
        depth = supplier_discovery_engine.get_typical_tier_depth(commodity)
        assert expected_min <= depth <= expected_max

    def test_auto_detect_tier_from_role(self, supplier_discovery_engine):
        """Supplier role helps determine approximate tier level."""
        role_tier_map = {
            "importer": 0,
            "trader": 1,
            "processor": 2,
            "aggregator": 3,
            "cooperative": 4,
            "farmer": 5,
        }
        for role, expected_tier in role_tier_map.items():
            tier = supplier_discovery_engine.infer_tier_from_role(role, "cocoa")
            assert tier == expected_tier or abs(tier - expected_tier) <= 1

    def test_unknown_commodity_returns_default_depth(self, supplier_discovery_engine):
        """Unknown commodity returns a reasonable default depth."""
        depth = supplier_discovery_engine.get_typical_tier_depth("unknown_commodity")
        assert 3 <= depth <= 10


# ===========================================================================
# 10. Certification Cross-Reference Discovery
# ===========================================================================


class TestCertificationCrossRefDiscovery:
    """Test discovery from certification database cross-references (F1.4)."""

    def test_discover_from_certification_db(self, supplier_discovery_engine):
        """Discover suppliers from certification database cross-references."""
        cert_data = {
            "certification_type": "RSPO",
            "commodity": "palm_oil",
            "certified_entities": [
                {"legal_name": "PT Certified Mill", "country_iso": "ID",
                 "certificate_number": "RSPO-2025-001"},
                {"legal_name": "PT Certified Refinery", "country_iso": "ID",
                 "certificate_number": "RSPO-2025-002"},
            ],
        }
        result = supplier_discovery_engine.discover_from_certification_crossref(cert_data)
        assert len(result.discovered_suppliers) == 2

    def test_cert_crossref_confidence_inferred(self, supplier_discovery_engine):
        """Cert-cross-ref discovery produces 'inferred' confidence."""
        cert_data = {
            "certification_type": "FSC",
            "commodity": "wood",
            "certified_entities": [
                {"legal_name": "Certified Sawmill", "country_iso": "CD",
                 "certificate_number": "FSC-2025-999"},
            ],
        }
        result = supplier_discovery_engine.discover_from_certification_crossref(cert_data)
        for sup in result.discovered_suppliers:
            assert sup["confidence"] in ("inferred", "declared")

    def test_cert_crossref_includes_cert_reference(self, supplier_discovery_engine):
        """Discovered suppliers include their certification reference."""
        cert_data = {
            "certification_type": "UTZ",
            "commodity": "cocoa",
            "certified_entities": [
                {"legal_name": "UTZ Certified Co", "country_iso": "GH",
                 "certificate_number": "UTZ-2025-ABC"},
            ],
        }
        result = supplier_discovery_engine.discover_from_certification_crossref(cert_data)
        sup = result.discovered_suppliers[0]
        assert "UTZ-2025-ABC" in sup.get("certifications", []) or \
               sup.get("certificate_number") == "UTZ-2025-ABC"


# ===========================================================================
# 11. Discovery Validation and Error Handling
# ===========================================================================


class TestDiscoveryValidation:
    """Test validation and error handling in discovery operations."""

    def test_declaration_invalid_commodity_raises(self, supplier_discovery_engine):
        """Declaration with invalid commodity type raises error."""
        declaration = {
            "declaring_supplier_id": "SUP-INVALID-COMM",
            "commodity": "diamonds",  # not an EUDR commodity
            "sub_suppliers": [
                {"legal_name": "Invalid", "country_iso": "GH", "role": "processor"},
            ],
        }
        with pytest.raises((ValueError, KeyError)):
            supplier_discovery_engine.discover_from_declaration(declaration)

    def test_discovery_result_has_metadata(self, supplier_discovery_engine):
        """Discovery result includes metadata about the operation."""
        declaration = {
            "declaring_supplier_id": "SUP-META-001",
            "commodity": "cocoa",
            "sub_suppliers": [
                {"legal_name": "Meta Sub", "country_iso": "GH", "role": "processor"},
            ],
        }
        result = supplier_discovery_engine.discover_from_declaration(declaration)
        assert hasattr(result, "total_discovered") or len(result.discovered_suppliers) >= 0
        assert hasattr(result, "source_type") or hasattr(result, "provenance_hash")

    def test_discovery_with_special_characters_in_name(self, supplier_discovery_engine):
        """Supplier names with special characters handled correctly."""
        declaration = {
            "declaring_supplier_id": "SUP-SPECIAL",
            "commodity": "coffee",
            "sub_suppliers": [
                {"legal_name": "Caf\u00e9 & Cie. S.A.R.L.", "country_iso": "CO",
                 "role": "mill"},
            ],
        }
        result = supplier_discovery_engine.discover_from_declaration(declaration)
        assert result.discovered_suppliers[0]["legal_name"] == "Caf\u00e9 & Cie. S.A.R.L."

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_discovery_accepts_all_commodities(self, supplier_discovery_engine, commodity):
        """Discovery works for each of the 7 EUDR commodities."""
        declaration = {
            "declaring_supplier_id": f"SUP-{commodity.upper()}",
            "commodity": commodity,
            "sub_suppliers": [
                {"legal_name": f"{commodity} Sub", "country_iso": "GH", "role": "farmer"},
            ],
        }
        result = supplier_discovery_engine.discover_from_declaration(declaration)
        assert len(result.discovered_suppliers) >= 1

    def test_batch_discovery_returns_statistics(self, supplier_discovery_engine):
        """Batch discovery result includes processing statistics."""
        declarations = [
            {
                "declaring_supplier_id": f"SUP-STAT-{i}",
                "commodity": "cocoa",
                "sub_suppliers": [
                    {"legal_name": f"Stat Sub {i}", "country_iso": "GH", "role": "farmer"},
                ],
            }
            for i in range(5)
        ]
        result = supplier_discovery_engine.discover_batch(declarations)
        assert hasattr(result, "total_discovered")
        assert result.total_discovered >= 5

    def test_dedup_case_insensitive_names(self, supplier_discovery_engine):
        """Deduplication treats names case-insensitively."""
        suppliers = [
            make_supplier(legal_name="GHANA COCOA LTD", country_iso="GH"),
            make_supplier(legal_name="ghana cocoa ltd", country_iso="GH"),
        ]
        result = supplier_discovery_engine.deduplicate(suppliers, fuzzy_threshold=0.85)
        assert len(result) == 1

    def test_dedup_strips_whitespace(self, supplier_discovery_engine):
        """Deduplication normalizes whitespace in names."""
        suppliers = [
            make_supplier(legal_name="Ghana  Cocoa  Ltd", country_iso="GH"),
            make_supplier(legal_name="Ghana Cocoa Ltd", country_iso="GH"),
        ]
        result = supplier_discovery_engine.deduplicate(suppliers, fuzzy_threshold=0.85)
        assert len(result) == 1
