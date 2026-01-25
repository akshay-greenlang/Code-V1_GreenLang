# -*- coding: utf-8 -*-
"""
Pytest fixtures for Spend Classification ML tests.

This module provides shared fixtures for testing spend classification components
including mock LLM clients, rule engines, sample procurement data, and test utilities.

Target: 300+ lines, comprehensive fixture coverage
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, List


# ============================================================================
# SCOPE 3 CATEGORIES
# ============================================================================

@pytest.fixture
def scope3_categories():
    """All 15 Scope 3 GHG Protocol categories."""
    return [
        {"id": 1, "name": "Purchased Goods and Services"},
        {"id": 2, "name": "Capital Goods"},
        {"id": 3, "name": "Fuel and Energy Related Activities"},
        {"id": 4, "name": "Upstream Transportation and Distribution"},
        {"id": 5, "name": "Waste Generated in Operations"},
        {"id": 6, "name": "Business Travel"},
        {"id": 7, "name": "Employee Commuting"},
        {"id": 8, "name": "Upstream Leased Assets"},
        {"id": 9, "name": "Downstream Transportation and Distribution"},
        {"id": 10, "name": "Processing of Sold Products"},
        {"id": 11, "name": "Use of Sold Products"},
        {"id": 12, "name": "End-of-Life Treatment of Sold Products"},
        {"id": 13, "name": "Downstream Leased Assets"},
        {"id": 14, "name": "Franchises"},
        {"id": 15, "name": "Investments"}
    ]


# ============================================================================
# MOCK LLM CLIENT
# ============================================================================

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for classification."""
    client = MagicMock()

    def mock_classify(description: str, categories: List[str]) -> Dict:
        """Mock classification logic based on keywords."""
        desc_lower = description.lower()

        # Simple rule-based mock for testing
        if any(word in desc_lower for word in ["travel", "flight", "hotel", "airfare"]):
            return {
                "category_id": 6,
                "category_name": "Business Travel",
                "confidence": 0.92,
                "reasoning": "Description contains travel-related keywords"
            }
        elif any(word in desc_lower for word in ["freight", "shipping", "logistics", "transport"]):
            return {
                "category_id": 4,
                "category_name": "Upstream Transportation and Distribution",
                "confidence": 0.89,
                "reasoning": "Description contains transportation keywords"
            }
        elif any(word in desc_lower for word in ["waste", "disposal", "recycling"]):
            return {
                "category_id": 5,
                "category_name": "Waste Generated in Operations",
                "confidence": 0.87,
                "reasoning": "Description contains waste management keywords"
            }
        elif any(word in desc_lower for word in ["electricity", "energy", "power", "fuel"]):
            return {
                "category_id": 3,
                "category_name": "Fuel and Energy Related Activities",
                "confidence": 0.90,
                "reasoning": "Description contains energy-related keywords"
            }
        elif any(word in desc_lower for word in ["office", "supplies", "stationery", "materials"]):
            return {
                "category_id": 1,
                "category_name": "Purchased Goods and Services",
                "confidence": 0.85,
                "reasoning": "Description contains goods/services keywords"
            }
        elif any(word in desc_lower for word in ["equipment", "machinery", "capital", "asset"]):
            return {
                "category_id": 2,
                "category_name": "Capital Goods",
                "confidence": 0.88,
                "reasoning": "Description contains capital goods keywords"
            }
        else:
            return {
                "category_id": 1,
                "category_name": "Purchased Goods and Services",
                "confidence": 0.60,
                "reasoning": "Default category for ambiguous description"
            }

    client.classify.side_effect = mock_classify
    return client


# ============================================================================
# SAMPLE PROCUREMENT DESCRIPTIONS (200+ variations)
# ============================================================================

@pytest.fixture
def sample_procurement_descriptions():
    """Sample procurement descriptions covering all 15 Scope 3 categories."""
    return {
        # Category 1: Purchased Goods and Services (20 examples)
        "category_1": [
            "Office supplies - paper, pens, folders",
            "IT equipment procurement - laptops and monitors",
            "Cleaning services for office building",
            "Marketing materials printing services",
            "Software licenses for productivity tools",
            "Catering services for corporate events",
            "Professional consulting services",
            "Legal advisory services",
            "Accounting and audit services",
            "HR recruitment services",
            "Stationery and office materials",
            "Furniture for office spaces",
            "Telecommunications services",
            "Security guard services",
            "Maintenance and repair services",
            "Laboratory testing services",
            "Design and creative services",
            "Translation services",
            "Training and development programs",
            "Market research services"
        ],

        # Category 2: Capital Goods (15 examples)
        "category_2": [
            "Manufacturing equipment purchase",
            "Heavy machinery for production line",
            "Industrial robots for automation",
            "Building construction - new warehouse",
            "Vehicle fleet purchase - delivery trucks",
            "Computer servers and data center equipment",
            "HVAC system installation",
            "Production line equipment",
            "Material handling equipment",
            "Quality control instruments",
            "Warehouse storage systems",
            "Forklift and lifting equipment",
            "Industrial ventilation systems",
            "Water treatment plant equipment",
            "Power generation equipment"
        ],

        # Category 3: Fuel and Energy Related Activities (15 examples)
        "category_3": [
            "Electricity consumption for facilities",
            "Natural gas for heating",
            "Diesel fuel for backup generators",
            "Energy efficiency consulting",
            "Renewable energy certificates",
            "Steam generation services",
            "Fuel oil for industrial processes",
            "Coal for power generation",
            "Biomass fuel purchase",
            "Solar panel installation",
            "Wind energy procurement",
            "Energy audit services",
            "Power factor correction equipment",
            "Energy monitoring systems",
            "Grid connection fees"
        ],

        # Category 4: Upstream Transportation (20 examples)
        "category_4": [
            "Freight services - inbound materials",
            "Shipping costs for raw materials",
            "Trucking services for supplier deliveries",
            "Air freight for urgent deliveries",
            "Ocean freight for imported goods",
            "Rail transport for bulk materials",
            "Courier services for documents",
            "Logistics and warehousing services",
            "Customs clearance services",
            "Freight forwarding services",
            "Container shipping fees",
            "Distribution center operations",
            "Cross-docking services",
            "Intermodal transportation",
            "Last-mile delivery services",
            "Cold chain logistics",
            "Hazardous materials transport",
            "Palletizing and packaging",
            "Material handling at ports",
            "Transit insurance costs"
        ],

        # Category 5: Waste Management (15 examples)
        "category_5": [
            "Waste disposal services",
            "Recycling program fees",
            "Hazardous waste treatment",
            "Landfill disposal costs",
            "Composting services",
            "Electronic waste recycling",
            "Industrial waste management",
            "Medical waste disposal",
            "Wastewater treatment",
            "Scrap metal recycling",
            "Paper waste collection",
            "Plastic waste processing",
            "Chemical waste disposal",
            "Construction debris removal",
            "Organic waste management"
        ],

        # Category 6: Business Travel (20 examples)
        "category_6": [
            "Airfare for business trip to London",
            "Hotel accommodation for conference",
            "Rental car for site visit",
            "Train tickets for client meeting",
            "Taxi and ride-sharing expenses",
            "Per diem allowances for travelers",
            "International flight - business class",
            "Domestic flights for team meeting",
            "Airport parking fees",
            "Travel insurance",
            "Conference registration fees",
            "Meals during business travel",
            "Airport lounge access",
            "Baggage fees",
            "Visa application fees",
            "Corporate travel card expenses",
            "Travel agent fees",
            "Hotel parking charges",
            "Business center services",
            "Emergency travel expenses"
        ],

        # Category 7: Employee Commuting (10 examples)
        "category_7": [
            "Shuttle bus service for employees",
            "Parking allowances",
            "Public transit subsidies",
            "Bike-sharing program fees",
            "Carpool program incentives",
            "Electric vehicle charging stations",
            "Commuter benefits administration",
            "Remote work technology support",
            "Company car leases for employees",
            "Fuel reimbursement for commuting"
        ],

        # Category 8: Upstream Leased Assets (10 examples)
        "category_8": [
            "Office space lease payments",
            "Warehouse rental fees",
            "Equipment leasing costs",
            "Vehicle leasing for operations",
            "Storage facility rental",
            "Co-working space membership",
            "Retail space lease",
            "Manufacturing facility rental",
            "Land lease for operations",
            "IT infrastructure leasing"
        ],

        # Category 9: Downstream Transportation (15 examples)
        "category_9": [
            "Product delivery to customers",
            "E-commerce shipping costs",
            "Retail distribution services",
            "Customer return shipping",
            "Final mile delivery services",
            "Freight to distribution centers",
            "Export shipping costs",
            "Cold storage during transport",
            "Packaging materials for shipping",
            "Tracking and monitoring services",
            "Insurance for shipped goods",
            "Customer pickup services",
            "Drop shipping fees",
            "White glove delivery",
            "Assembly and installation services"
        ],

        # Category 10: Processing of Sold Products (10 examples)
        "category_10": [
            "Ingredient processing by buyer",
            "Raw material refining services",
            "Component assembly by customer",
            "Product finishing services",
            "Coating and treatment processes",
            "Quality testing by purchaser",
            "Packaging by third-party",
            "Labeling services downstream",
            "Blending and mixing operations",
            "Sterilization services"
        ],

        # Category 11: Use of Sold Products (10 examples)
        "category_11": [
            "Energy consumption during product use",
            "Fuel for sold vehicles",
            "Electricity for sold appliances",
            "Consumables for sold equipment",
            "Maintenance parts for sold products",
            "Replacement filters",
            "Operating supplies",
            "Refills and cartridges",
            "Batteries for electronics",
            "Software updates and licenses"
        ],

        # Category 12: End-of-Life Treatment (10 examples)
        "category_12": [
            "Product recycling at end-of-life",
            "Disposal of sold products",
            "Take-back program costs",
            "Product disassembly services",
            "Material recovery operations",
            "Landfill for disposed products",
            "Incineration with energy recovery",
            "Hazardous component disposal",
            "Electronics recycling program",
            "Battery recycling services"
        ],

        # Category 13: Downstream Leased Assets (10 examples)
        "category_13": [
            "Franchisee facility operations",
            "Leased retail locations by others",
            "Equipment leased to customers",
            "Property leased to third parties",
            "Downstream warehouse leasing",
            "Showroom leased to partners",
            "Storage leased to distributors",
            "Office space leased to tenants",
            "Vehicle fleet leased to customers",
            "Machinery leased to operators"
        ],

        # Category 14: Franchises (10 examples)
        "category_14": [
            "Franchise royalty payments",
            "Franchisee support services",
            "Franchise marketing fees",
            "Franchise training programs",
            "Franchise quality audits",
            "Franchise system development",
            "Franchise operations manuals",
            "Franchise territory fees",
            "Franchise renewal costs",
            "Franchise legal services"
        ],

        # Category 15: Investments (10 examples)
        "category_15": [
            "Equity investments in companies",
            "Project finance investments",
            "Real estate investment trust holdings",
            "Corporate bond investments",
            "Venture capital funding",
            "Private equity stakes",
            "Infrastructure investments",
            "Green bond purchases",
            "Mutual fund investments",
            "Pension fund allocations"
        ]
    }


# ============================================================================
# MOCK LLM RESPONSES
# ============================================================================

@pytest.fixture
def mock_llm_responses():
    """Sample LLM API responses."""
    return {
        "business_travel": {
            "category_id": 6,
            "category_name": "Business Travel",
            "confidence": 0.92,
            "reasoning": "The description mentions airfare and hotel, which are typical business travel expenses."
        },
        "transportation": {
            "category_id": 4,
            "category_name": "Upstream Transportation and Distribution",
            "confidence": 0.89,
            "reasoning": "Freight and shipping services for inbound materials fall under upstream transportation."
        },
        "waste": {
            "category_id": 5,
            "category_name": "Waste Generated in Operations",
            "confidence": 0.87,
            "reasoning": "Waste disposal and recycling services are categorized under waste management."
        }
    }


# ============================================================================
# MOCK RULE ENGINE
# ============================================================================

@pytest.fixture
def mock_rule_engine():
    """Mock rule-based classification engine."""
    engine = MagicMock()

    def mock_classify_by_rules(description: str) -> Dict:
        """Simple rule-based classification."""
        desc_lower = description.lower()

        # Check for keywords
        if "travel" in desc_lower or "flight" in desc_lower or "hotel" in desc_lower:
            return {
                "category_id": 6,
                "category_name": "Business Travel",
                "confidence": 0.85,
                "rule_matched": "travel_keywords"
            }
        elif "freight" in desc_lower or "shipping" in desc_lower:
            return {
                "category_id": 4,
                "category_name": "Upstream Transportation and Distribution",
                "confidence": 0.80,
                "rule_matched": "shipping_keywords"
            }
        else:
            return None

    engine.classify.side_effect = mock_classify_by_rules
    return engine


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def spend_classification_config():
    """Sample spend classification configuration."""
    return {
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 500,
            "api_key": "test-api-key"
        },
        "rules": {
            "enabled": True,
            "fallback_to_llm": True,
            "confidence_threshold": 0.75
        },
        "routing": {
            "high_confidence_threshold": 0.85,
            "use_rules_first": True
        },
        "categories": 15
    }


# ============================================================================
# TEST DATA HELPERS
# ============================================================================

@pytest.fixture
def create_procurement_item():
    """Factory function to create test procurement items."""
    def _create(description: str, expected_category: int = None, amount: float = 1000.0):
        return {
            "description": description,
            "amount": amount,
            "expected_category": expected_category
        }
    return _create
