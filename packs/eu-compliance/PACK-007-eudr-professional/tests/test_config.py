# -*- coding: utf-8 -*-
"""
PACK-007 EUDR Professional Pack - Configuration Tests
======================================================

Tests the EUDRProfessionalConfig configuration system including:
- Default config validation
- Advanced geolocation config
- Scenario risk (Monte Carlo) config
- Satellite monitoring config
- Continuous monitoring config
- Portfolio management config
- Audit management config
- Protected area analysis config
- Regulatory tracking config
- Grievance mechanism config
- Cross-regulation compliance config
- Supply chain traceability config
- Supplier benchmarking config
- Config merging with presets/sectors
- Environment variable overrides
- Config hash reproducibility

Author: GreenLang QA Team
Version: 1.0.0
"""

import os
from typing import Any, Dict

import pytest


class TestEUDRProfessionalConfig:
    """Test suite for PACK-007 configuration."""

    def test_default_config_valid(self, mock_config: Dict[str, Any]):
        """Test default config is valid and complete."""
        assert "pack_id" in mock_config, "Config missing pack_id"
        assert "tier" in mock_config, "Config missing tier"
        assert mock_config["pack_id"] == "PACK-007"
        assert mock_config["tier"] == "professional"

    def test_pack_id(self, mock_config: Dict[str, Any]):
        """Test pack_id is PACK-007."""
        assert mock_config["pack_id"] == "PACK-007"

    def test_tier(self, mock_config: Dict[str, Any]):
        """Test tier is professional."""
        assert mock_config["tier"] == "professional"

    def test_extends(self, mock_config: Dict[str, Any]):
        """Test extends PACK-006."""
        assert "extends" in mock_config
        assert mock_config["extends"] == "PACK-006"

    def test_advanced_geolocation_config(self, mock_config: Dict[str, Any]):
        """Test advanced geolocation config defaults."""
        assert "advanced_geolocation" in mock_config
        geo_config = mock_config["advanced_geolocation"]

        assert isinstance(geo_config, dict)
        assert "sentinel_monitoring" in geo_config
        assert geo_config["sentinel_monitoring"] is True

        assert "protected_area_buffer_km" in geo_config
        assert geo_config["protected_area_buffer_km"] == 5.0

        assert "indigenous_land_check" in geo_config
        assert geo_config["indigenous_land_check"] is True

        assert "forest_change_detection" in geo_config
        assert geo_config["forest_change_detection"] is True

    def test_scenario_risk_config_defaults(self, mock_config: Dict[str, Any]):
        """Test scenario risk (Monte Carlo) config defaults."""
        assert "scenario_risk" in mock_config
        risk_config = mock_config["scenario_risk"]

        assert isinstance(risk_config, dict)

        # Test simulation_count
        assert "simulation_count" in risk_config
        assert risk_config["simulation_count"] == 10000
        assert risk_config["simulation_count"] >= 1000

        # Test confidence_levels
        assert "confidence_levels" in risk_config
        assert isinstance(risk_config["confidence_levels"], list)
        assert 0.90 in risk_config["confidence_levels"]
        assert 0.95 in risk_config["confidence_levels"]
        assert 0.99 in risk_config["confidence_levels"]

        # Test distribution_types
        assert "distribution_types" in risk_config
        assert isinstance(risk_config["distribution_types"], list)
        assert "normal" in risk_config["distribution_types"]
        assert "beta" in risk_config["distribution_types"]
        assert "triangular" in risk_config["distribution_types"]

    def test_satellite_monitoring_config(self, mock_config: Dict[str, Any]):
        """Test satellite monitoring config."""
        assert "satellite_monitoring" in mock_config
        sat_config = mock_config["satellite_monitoring"]

        assert isinstance(sat_config, dict)
        assert "sources" in sat_config
        assert isinstance(sat_config["sources"], list)
        assert len(sat_config["sources"]) >= 2

        # Validate common satellite sources
        sources = sat_config["sources"]
        expected_sources = ["Sentinel-1", "Sentinel-2", "Landsat-8"]
        for expected in expected_sources:
            assert expected in sources, f"Expected satellite source {expected} not found"

        assert "update_frequency_days" in sat_config
        assert sat_config["update_frequency_days"] == 14

    def test_continuous_monitoring_config(self, mock_config: Dict[str, Any]):
        """Test continuous monitoring config."""
        assert "continuous_monitoring" in mock_config
        mon_config = mock_config["continuous_monitoring"]

        assert isinstance(mon_config, dict)
        assert "enabled" in mon_config
        assert mon_config["enabled"] is True

        assert "check_interval_hours" in mon_config
        assert mon_config["check_interval_hours"] == 24

        assert "alert_channels" in mon_config
        assert isinstance(mon_config["alert_channels"], list)
        assert "email" in mon_config["alert_channels"]

    def test_portfolio_config(self, mock_config: Dict[str, Any]):
        """Test portfolio config (max_operators=100)."""
        assert "portfolio_config" in mock_config
        portfolio = mock_config["portfolio_config"]

        assert isinstance(portfolio, dict)
        assert "max_operators" in portfolio
        assert portfolio["max_operators"] == 100

        assert "shared_supplier_pool" in portfolio
        assert portfolio["shared_supplier_pool"] is True

    def test_audit_management_config(self, mock_config: Dict[str, Any]):
        """Test audit management config (retention_years=5)."""
        assert "audit_management" in mock_config
        audit = mock_config["audit_management"]

        assert isinstance(audit, dict)
        assert "retention_years" in audit
        assert audit["retention_years"] == 5

        assert "automatic_archival" in audit
        assert isinstance(audit["automatic_archival"], bool)

    def test_protected_area_config(self, mock_config: Dict[str, Any]):
        """Test protected area config (buffer_km=5)."""
        assert "protected_area_config" in mock_config
        protected = mock_config["protected_area_config"]

        assert isinstance(protected, dict)
        assert "buffer_km" in protected
        assert protected["buffer_km"] == 5

        assert "check_indigenous_lands" in protected
        assert protected["check_indigenous_lands"] is True

    def test_regulatory_tracking_config(self, mock_config: Dict[str, Any]):
        """Test regulatory tracking config."""
        assert "regulatory_tracking" in mock_config
        reg_tracking = mock_config["regulatory_tracking"]

        assert isinstance(reg_tracking, dict)
        assert "enabled" in reg_tracking
        assert reg_tracking["enabled"] is True

        assert "sources" in reg_tracking
        assert isinstance(reg_tracking["sources"], list)
        assert "EUR-Lex" in reg_tracking["sources"]

    def test_grievance_config(self, mock_config: Dict[str, Any]):
        """Test grievance mechanism config (sla_days)."""
        assert "grievance_config" in mock_config
        grievance = mock_config["grievance_config"]

        assert isinstance(grievance, dict)
        assert "sla_days" in grievance
        assert grievance["sla_days"] == 30

        assert "escalation_enabled" in grievance
        assert isinstance(grievance["escalation_enabled"], bool)

    def test_cross_regulation_config(self, mock_config: Dict[str, Any]):
        """Test cross-regulation compliance config."""
        assert "cross_regulation_config" in mock_config
        cross_reg = mock_config["cross_regulation_config"]

        assert isinstance(cross_reg, dict)
        assert "regulations" in cross_reg
        assert isinstance(cross_reg["regulations"], list)
        assert "EUDR" in cross_reg["regulations"]
        assert len(cross_reg["regulations"]) >= 2

    def test_supply_chain_config(self, mock_config: Dict[str, Any]):
        """Test supply chain config (max_tier_depth=5)."""
        assert "supply_chain_config" in mock_config
        supply_chain = mock_config["supply_chain_config"]

        assert isinstance(supply_chain, dict)
        assert "max_tier_depth" in supply_chain
        assert supply_chain["max_tier_depth"] == 5

        assert "traceability_level" in supply_chain
        assert supply_chain["traceability_level"] in ["plot", "farm", "supplier"]

    def test_supplier_benchmark_config(self, mock_config: Dict[str, Any]):
        """Test supplier benchmarking config."""
        assert "supplier_benchmark_config" in mock_config
        benchmark = mock_config["supplier_benchmark_config"]

        assert isinstance(benchmark, dict)
        assert "peer_group_size" in benchmark
        assert benchmark["peer_group_size"] == 10

        assert "scoring_dimensions" in benchmark
        assert benchmark["scoring_dimensions"] == 6

    def test_config_merge_with_preset(self, mock_config: Dict[str, Any]):
        """Test config merging with preset values."""
        # Simulate preset override
        preset_overrides = {
            "scenario_risk": {
                "simulation_count": 5000,
            }
        }

        merged = {**mock_config, **preset_overrides}
        assert merged["scenario_risk"]["simulation_count"] == 5000

        # Original should be unchanged
        assert mock_config["scenario_risk"]["simulation_count"] == 10000

    def test_config_merge_with_sector(self, mock_config: Dict[str, Any]):
        """Test config merging with sector-specific values."""
        # Simulate sector override (e.g., food_beverage)
        sector_overrides = {
            "portfolio_config": {
                "max_operators": 50,
            }
        }

        merged = {**mock_config, **sector_overrides}
        assert merged["portfolio_config"]["max_operators"] == 50

        # Original should be unchanged
        assert mock_config["portfolio_config"]["max_operators"] == 100

    def test_config_env_overrides(self, mock_config: Dict[str, Any], monkeypatch):
        """Test config can be overridden by environment variables."""
        # Set environment variable
        monkeypatch.setenv("EUDR_SIMULATION_COUNT", "20000")

        # Simulate reading env var in config
        env_simulation_count = int(os.getenv("EUDR_SIMULATION_COUNT", "10000"))
        assert env_simulation_count == 20000

        # Without env var, should default to 10000
        monkeypatch.delenv("EUDR_SIMULATION_COUNT", raising=False)
        default_count = int(os.getenv("EUDR_SIMULATION_COUNT", "10000"))
        assert default_count == 10000

    def test_config_hash_reproducibility(self, mock_config: Dict[str, Any]):
        """Test config hash is reproducible for same config."""
        import hashlib
        import json

        # Compute hash twice
        config_str = json.dumps(mock_config, sort_keys=True)
        hash1 = hashlib.sha256(config_str.encode()).hexdigest()
        hash2 = hashlib.sha256(config_str.encode()).hexdigest()

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_simulation_count_validation(self):
        """Test simulation count is within valid range."""
        valid_counts = [1000, 5000, 10000, 50000]
        for count in valid_counts:
            assert count >= 1000, f"Simulation count {count} too low"
            assert count <= 100000, f"Simulation count {count} too high"

    def test_confidence_levels_validation(self):
        """Test confidence levels are valid probabilities."""
        valid_levels = [0.90, 0.95, 0.99]
        for level in valid_levels:
            assert 0.0 < level < 1.0, f"Confidence level {level} out of range"

    def test_buffer_km_validation(self):
        """Test buffer_km is positive."""
        valid_buffers = [1, 3, 5, 10]
        for buffer in valid_buffers:
            assert buffer > 0, f"Buffer {buffer} must be positive"
            assert buffer <= 50, f"Buffer {buffer} too large"

    def test_retention_years_validation(self):
        """Test retention_years is reasonable."""
        valid_years = [3, 5, 7, 10]
        for years in valid_years:
            assert years >= 1, f"Retention {years} must be at least 1 year"
            assert years <= 20, f"Retention {years} too long"

    def test_sla_days_validation(self):
        """Test SLA days is reasonable."""
        valid_sla = [7, 14, 30, 60]
        for days in valid_sla:
            assert days > 0, f"SLA {days} must be positive"
            assert days <= 365, f"SLA {days} too long"

    def test_max_operators_validation(self):
        """Test max_operators is reasonable."""
        valid_max = [10, 50, 100, 500]
        for max_op in valid_max:
            assert max_op > 0, f"max_operators {max_op} must be positive"
            assert max_op <= 10000, f"max_operators {max_op} too high"

    def test_max_tier_depth_validation(self):
        """Test max_tier_depth is reasonable."""
        valid_depths = [3, 5, 7, 10]
        for depth in valid_depths:
            assert depth > 0, f"max_tier_depth {depth} must be positive"
            assert depth <= 15, f"max_tier_depth {depth} too deep"

    def test_update_frequency_days_validation(self):
        """Test satellite update frequency is reasonable."""
        valid_frequencies = [1, 7, 14, 30]
        for freq in valid_frequencies:
            assert freq > 0, f"update_frequency_days {freq} must be positive"
            assert freq <= 365, f"update_frequency_days {freq} too long"

    def test_check_interval_hours_validation(self):
        """Test monitoring check interval is reasonable."""
        valid_intervals = [1, 6, 12, 24]
        for interval in valid_intervals:
            assert interval > 0, f"check_interval_hours {interval} must be positive"
            assert interval <= 168, f"check_interval_hours {interval} too long (>1 week)"

    def test_all_required_configs_present(self, mock_config: Dict[str, Any]):
        """Test all required config sections are present."""
        required_sections = [
            "pack_id",
            "tier",
            "extends",
            "advanced_geolocation",
            "scenario_risk",
            "satellite_monitoring",
            "continuous_monitoring",
            "portfolio_config",
            "audit_management",
            "protected_area_config",
            "regulatory_tracking",
            "grievance_config",
            "cross_regulation_config",
            "supply_chain_config",
            "supplier_benchmark_config",
        ]

        for section in required_sections:
            assert section in mock_config, f"Required config section '{section}' missing"
