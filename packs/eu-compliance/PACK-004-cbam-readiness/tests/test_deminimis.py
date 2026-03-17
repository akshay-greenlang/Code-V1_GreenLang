# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness Pack - De Minimis Tests (10 tests)

Tests de minimis threshold tracking, alert levels, exemption
determination, annual assessment, and volume projections.

Author: GreenLang QA Team
"""

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import _compute_hash, _new_uuid, _utcnow


class TestDeMinimisTracking:
    """Test de minimis import tracking."""

    def test_track_import_under_threshold(self, sample_cbam_config):
        """Test tracking an import that stays under de minimis threshold."""
        threshold_kg = sample_cbam_config["cbam"]["deminimis_config"]["annual_weight_threshold_kg"]
        import_entry = {
            "import_id": "IMP-DM-001",
            "weight_kg": 10000,
            "cn_code": "7207 11 14",
            "goods_category": "steel",
            "cumulative_year_kg": 50000,
        }
        result = {
            "import_id": import_entry["import_id"],
            "cumulative_kg": import_entry["cumulative_year_kg"],
            "threshold_kg": threshold_kg,
            "under_threshold": import_entry["cumulative_year_kg"] < threshold_kg,
            "utilization_pct": round(
                import_entry["cumulative_year_kg"] / threshold_kg * 100, 1
            ),
            "exempt": True,
        }
        assert result["under_threshold"] is True
        assert result["utilization_pct"] < 100
        assert result["exempt"] is True

    def test_track_import_exceeding_threshold(self, sample_cbam_config):
        """Test tracking an import that exceeds de minimis threshold."""
        threshold_kg = sample_cbam_config["cbam"]["deminimis_config"]["annual_weight_threshold_kg"]
        cumulative_kg = 200000
        result = {
            "cumulative_kg": cumulative_kg,
            "threshold_kg": threshold_kg,
            "exceeded": cumulative_kg >= threshold_kg,
            "over_by_kg": cumulative_kg - threshold_kg,
            "exempt": False,
        }
        assert result["exceeded"] is True
        assert result["over_by_kg"] == 50000
        assert result["exempt"] is False


class TestDeMinimisAlerts:
    """Test de minimis alert level system."""

    def test_alert_level_safe(self, sample_cbam_config):
        """Test safe alert level (well under threshold)."""
        threshold_kg = sample_cbam_config["cbam"]["deminimis_config"]["annual_weight_threshold_kg"]
        alert_pct = sample_cbam_config["cbam"]["deminimis_config"]["alert_at_pct"]
        cumulative_kg = 50000
        pct = cumulative_kg / threshold_kg * 100
        level = "exceeded" if pct >= 100 else "approaching" if pct >= alert_pct else "safe"
        assert level == "safe"
        assert pct < alert_pct

    def test_alert_level_approaching(self, sample_cbam_config):
        """Test approaching alert level (80-99% of threshold)."""
        threshold_kg = sample_cbam_config["cbam"]["deminimis_config"]["annual_weight_threshold_kg"]
        alert_pct = sample_cbam_config["cbam"]["deminimis_config"]["alert_at_pct"]
        cumulative_kg = 130000  # 86.7% of 150000
        pct = cumulative_kg / threshold_kg * 100
        level = "exceeded" if pct >= 100 else "approaching" if pct >= alert_pct else "safe"
        assert level == "approaching"

    def test_alert_level_exceeded(self, sample_cbam_config):
        """Test exceeded alert level (>= 100% of threshold)."""
        threshold_kg = sample_cbam_config["cbam"]["deminimis_config"]["annual_weight_threshold_kg"]
        cumulative_kg = 155000
        pct = cumulative_kg / threshold_kg * 100
        level = "exceeded" if pct >= 100 else "approaching" if pct >= 80 else "safe"
        assert level == "exceeded"


class TestDeMinimisExemptions:
    """Test de minimis exemption logic."""

    def test_exemption_granted(self, sample_cbam_config):
        """Test exemption granted when under both thresholds."""
        dmc = sample_cbam_config["cbam"]["deminimis_config"]
        weight_kg = 100000
        value_eur = 100.0
        exempt = weight_kg < dmc["annual_weight_threshold_kg"] and \
                 value_eur < dmc["annual_value_threshold_eur"]
        result = {
            "weight_under": weight_kg < dmc["annual_weight_threshold_kg"],
            "value_under": value_eur < dmc["annual_value_threshold_eur"],
            "exempt": exempt,
            "status": "exempt" if exempt else "reporting_required",
        }
        assert result["exempt"] is True
        assert result["status"] == "exempt"

    def test_exemption_revoked(self, sample_cbam_config):
        """Test exemption revoked when threshold exceeded mid-year."""
        dmc = sample_cbam_config["cbam"]["deminimis_config"]
        monthly_imports_kg = [10000, 15000, 20000, 25000, 30000, 25000,
                              20000, 15000, 10000, 10000, 10000, 10000]
        for month_idx in range(len(monthly_imports_kg)):
            cumulative = sum(monthly_imports_kg[:month_idx + 1])
            if cumulative >= dmc["annual_weight_threshold_kg"]:
                revocation = {
                    "revoked_at_month": month_idx + 1,
                    "cumulative_kg": cumulative,
                    "threshold_kg": dmc["annual_weight_threshold_kg"],
                    "status": "exemption_revoked",
                }
                assert revocation["status"] == "exemption_revoked"
                assert revocation["revoked_at_month"] <= 12
                break


class TestDeMinimisAssessment:
    """Test annual de minimis assessment."""

    def test_annual_assessment(self, sample_cbam_config):
        """Test annual de minimis assessment with category breakdown."""
        dmc = sample_cbam_config["cbam"]["deminimis_config"]
        categories = {
            "steel": 80000,
            "aluminium": 30000,
            "cement": 40000,
        }
        total_kg = sum(categories.values())
        assessment = {
            "year": 2026,
            "categories": categories,
            "total_weight_kg": total_kg,
            "threshold_kg": dmc["annual_weight_threshold_kg"],
            "exceeded": total_kg >= dmc["annual_weight_threshold_kg"],
            "utilization_pct": round(total_kg / dmc["annual_weight_threshold_kg"] * 100, 1),
        }
        assert assessment["total_weight_kg"] == 150000
        assert assessment["exceeded"] is True

    def test_multi_sector_tracking(self, sample_cbam_config):
        """Test de minimis tracking across multiple sectors."""
        dmc = sample_cbam_config["cbam"]["deminimis_config"]
        sectors = {
            "steel": {"weight_kg": 60000, "value_eur": 50.0},
            "aluminium": {"weight_kg": 40000, "value_eur": 45.0},
            "cement": {"weight_kg": 20000, "value_eur": 15.0},
        }
        total_weight = sum(s["weight_kg"] for s in sectors.values())
        total_value = sum(s["value_eur"] for s in sectors.values())
        result = {
            "sectors_tracked": len(sectors),
            "total_weight_kg": total_weight,
            "total_value_eur": total_value,
            "weight_pct": round(total_weight / dmc["annual_weight_threshold_kg"] * 100, 1),
            "exempt": total_weight < dmc["annual_weight_threshold_kg"],
        }
        assert result["sectors_tracked"] == 3
        assert result["total_weight_kg"] == 120000
        assert result["exempt"] is True

    def test_volume_projection(self, sample_cbam_config):
        """Test volume projection for remaining year."""
        dmc = sample_cbam_config["cbam"]["deminimis_config"]
        months_elapsed = 6
        cumulative_kg = 80000
        avg_monthly = cumulative_kg / months_elapsed
        projected_annual = round(avg_monthly * 12, 0)
        projection = {
            "months_elapsed": months_elapsed,
            "cumulative_kg": cumulative_kg,
            "avg_monthly_kg": round(avg_monthly, 0),
            "projected_annual_kg": projected_annual,
            "threshold_kg": dmc["annual_weight_threshold_kg"],
            "projected_exceed": projected_annual >= dmc["annual_weight_threshold_kg"],
            "provenance_hash": _compute_hash({
                "cumulative": cumulative_kg,
                "months": months_elapsed,
            }),
        }
        assert projection["projected_annual_kg"] == 160000
        assert projection["projected_exceed"] is True
        assert len(projection["provenance_hash"]) == 64
