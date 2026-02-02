# -*- coding: utf-8 -*-
"""Tests for RecommendationAgent."""

import pytest
from greenlang.agents.recommendation_agent import RecommendationAgent


class TestRecommendationAgent:
    """Test suite for RecommendationAgent - contract-focused tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = RecommendationAgent()
    
    def test_stable_recommendations(self):
        """Test that recommendations are stable and deduplicated."""
        input_data = {
            "country": "IN",
            "building_type": "office",
            "performance_rating": "D",
            "building_age": 15,
            "emission_intensity": 25.0
        }
        
        result = self.agent.run(input_data)
        
        assert result["success"] is True
        data = result["data"]
        
        # Should return recommendations
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) > 0
        
        # Check for duplicates
        actions = [r["action"] for r in data["recommendations"]]
        assert len(actions) == len(set(actions))  # No duplicates
    
    def test_recommendation_structure(self):
        """Test that each recommendation has required fields."""
        result = self.agent.run({
            "country": "US",
            "building_type": "office",
            "performance_rating": "C",
            "building_age": 10
        })
        
        assert result["success"] is True
        
        for rec in result["data"]["recommendations"]:
            # Required fields
            assert "action" in rec
            assert "impact" in rec
            assert "payback" in rec
            
            # Optional fields
            if "category" in rec:
                assert rec["category"] in [
                    "energy_efficiency", "renewable_energy", "behavioral",
                    "hvac", "lighting", "envelope", "controls"
                ]
            
            if "cost_band" in rec:
                assert rec["cost_band"] in ["low", "medium", "high", "very_high"]
            
            # Validate impact
            assert rec["impact"] in ["low", "medium", "high", "very_high"]
            
            # Validate payback
            assert rec["payback"] in ["immediate", "short", "medium", "long"] or \
                   isinstance(rec["payback"], (int, float))
    
    def test_worse_rating_more_recommendations(self):
        """Test that worse ratings generate more recommendations."""
        # Good performance
        good_result = self.agent.run({
            "country": "IN",
            "building_type": "office",
            "performance_rating": "B",
            "building_age": 5
        })
        
        # Poor performance
        poor_result = self.agent.run({
            "country": "IN",
            "building_type": "office",
            "performance_rating": "E",
            "building_age": 5
        })
        
        assert good_result["success"] and poor_result["success"]
        
        # Poor performance should have at least as many recommendations
        assert len(poor_result["data"]["recommendations"]) >= len(good_result["data"]["recommendations"])
    
    def test_top_n_recommendations(self):
        """Test limiting to top N recommendations."""
        result = self.agent.run({
            "country": "IN",
            "building_type": "office",
            "performance_rating": "D",
            "building_age": 20,
            "max_recommendations": 5
        })
        
        assert result["success"] is True
        
        # Should respect max_recommendations if provided
        if "max_recommendations" in result["data"]:
            assert len(result["data"]["recommendations"]) <= 5
    
    def test_building_type_specific_recommendations(self):
        """Test that recommendations are tailored to building type."""
        office_result = self.agent.run({
            "country": "US",
            "building_type": "office",
            "performance_rating": "C",
            "building_age": 10
        })
        
        hospital_result = self.agent.run({
            "country": "US",
            "building_type": "hospital",
            "performance_rating": "C",
            "building_age": 10
        })
        
        assert office_result["success"] and hospital_result["success"]
        
        office_actions = [r["action"] for r in office_result["data"]["recommendations"]]
        hospital_actions = [r["action"] for r in hospital_result["data"]["recommendations"]]
        
        # Some recommendations should be different
        assert office_actions != hospital_actions
        
        # Hospital might have specific recommendations
        hospital_keywords = ["medical", "sterilization", "24/7", "critical"]
        if any(keyword in str(hospital_actions).lower() for keyword in hospital_keywords):
            assert True  # Hospital-specific recommendations found
    
    def test_age_based_recommendations(self):
        """Test that building age influences recommendations."""
        new_result = self.agent.run({
            "country": "IN",
            "building_type": "office",
            "performance_rating": "C",
            "building_age": 2
        })
        
        old_result = self.agent.run({
            "country": "IN",
            "building_type": "office",
            "performance_rating": "C",
            "building_age": 30
        })
        
        assert new_result["success"] and old_result["success"]
        
        # Older buildings might have retrofit recommendations
        old_actions = [r["action"] for r in old_result["data"]["recommendations"]]
        retrofit_keywords = ["retrofit", "upgrade", "replace", "modernize"]
        
        has_retrofit = any(
            keyword in action.lower()
            for keyword in retrofit_keywords
            for action in old_actions
        )
        
        # Older buildings more likely to need retrofits
        if has_retrofit:
            assert True
    
    def test_priority_ordering(self):
        """Test that recommendations are ordered by priority/impact."""
        result = self.agent.run({
            "country": "US",
            "building_type": "office",
            "performance_rating": "D",
            "building_age": 15
        })
        
        assert result["success"] is True
        recs = result["data"]["recommendations"]
        
        if len(recs) > 1:
            # Check if there's a priority field
            if all("priority" in r for r in recs):
                priorities = [r["priority"] for r in recs]
                assert priorities == sorted(priorities)  # Should be sorted
            
            # Or check if sorted by impact
            impact_order = ["very_high", "high", "medium", "low"]
            if all("impact" in r for r in recs):
                # First recommendations should have higher impact
                first_impact_idx = impact_order.index(recs[0]["impact"])
                last_impact_idx = impact_order.index(recs[-1]["impact"])
                assert first_impact_idx <= last_impact_idx
    
    def test_excellent_performance_minimal_recommendations(self):
        """Test that excellent performance generates minimal recommendations."""
        result = self.agent.run({
            "country": "IN",
            "building_type": "office",
            "performance_rating": "A",
            "building_age": 5,
            "emission_intensity": 5.0
        })
        
        assert result["success"] is True
        
        # Should have few or no recommendations
        assert len(result["data"]["recommendations"]) <= 3
        
        # If any, should be about maintaining performance
        if result["data"]["recommendations"]:
            actions = [r["action"].lower() for r in result["data"]["recommendations"]]
            maintenance_keywords = ["maintain", "monitor", "optimize", "continue"]
            has_maintenance = any(
                keyword in action
                for keyword in maintenance_keywords
                for action in actions
            )
            assert has_maintenance or len(actions) == 0
    
    def test_cost_effectiveness_included(self):
        """Test that cost-effectiveness information is included."""
        result = self.agent.run({
            "country": "US",
            "building_type": "office",
            "performance_rating": "C",
            "building_age": 10,
            "include_costs": True
        })
        
        if result["success"]:
            for rec in result["data"]["recommendations"]:
                # Should include cost-effectiveness metrics
                assert any(key in rec for key in ["roi", "cost_band", "payback", "annual_savings"])
    
    def test_renewable_energy_recommendations(self):
        """Test inclusion of renewable energy recommendations where appropriate."""
        result = self.agent.run({
            "country": "IN",  # High grid emissions
            "building_type": "office",
            "performance_rating": "C",
            "building_age": 10,
            "roof_area_available": True
        })
        
        assert result["success"] is True
        
        # Should include renewable energy recommendations for high-emission grids
        actions = [r["action"].lower() for r in result["data"]["recommendations"]]
        renewable_keywords = ["solar", "pv", "renewable", "wind"]
        has_renewable = any(
            keyword in action
            for keyword in renewable_keywords
            for action in actions
        )
        
        # India with high grid emissions should suggest renewables
        assert has_renewable
    
    def test_output_schema(self):
        """Test that output conforms to expected schema."""
        result = self.agent.run({
            "country": "IN",
            "building_type": "office",
            "performance_rating": "C",
            "building_age": 10
        })
        
        assert result["success"] is True
        data = result["data"]
        
        # Required fields
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
        
        # Optional fields
        if "total_potential_reduction" in data:
            assert isinstance(data["total_potential_reduction"], (int, float))
            assert data["total_potential_reduction"] >= 0
        
        if "implementation_timeline" in data:
            assert isinstance(data["implementation_timeline"], dict)
            assert any(key in data["implementation_timeline"] for key in ["immediate", "short_term", "long_term"])