# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack - Cutoff Date Verification Tests
==============================================================

Validates the cutoff date engine including deforestation-free
verification against the 2020-12-31 cutoff date, temporal evidence
validation, land use history analysis, exemption handling, batch
verification, and cutoff summary generation.

Test count: 10
Author: GreenLang QA Team
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import date, datetime
from typing import Any, Dict, List

import pytest

from conftest import (
    EUDR_CUTOFF_DATE,
    _compute_hash,
)


# ---------------------------------------------------------------------------
# Cutoff Date Engine Simulator
# ---------------------------------------------------------------------------

class CutoffDateEngineSimulator:
    """Simulates cutoff date verification engine operations."""

    CUTOFF_DATE = EUDR_CUTOFF_DATE  # 2020-12-31

    def verify_cutoff_compliance(self, plot: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that a plot complies with the EUDR cutoff date.

        A plot is compliant if no deforestation or forest degradation
        has occurred on the land after 2020-12-31.
        """
        deforestation_free_since = plot.get("deforestation_free_since", "")
        if not deforestation_free_since:
            return {
                "compliant": False,
                "reason": "No deforestation-free date provided",
                "cutoff_date": str(self.CUTOFF_DATE),
                "plot_id": plot.get("plot_id", "unknown"),
            }
        try:
            free_since = date.fromisoformat(deforestation_free_since)
        except ValueError:
            return {
                "compliant": False,
                "reason": f"Invalid date format: {deforestation_free_since}",
                "cutoff_date": str(self.CUTOFF_DATE),
                "plot_id": plot.get("plot_id", "unknown"),
            }

        compliant = free_since <= self.CUTOFF_DATE
        return {
            "compliant": compliant,
            "deforestation_free_since": str(free_since),
            "cutoff_date": str(self.CUTOFF_DATE),
            "plot_id": plot.get("plot_id", "unknown"),
            "reason": "Deforestation-free since before cutoff" if compliant
                      else "Deforestation occurred after cutoff date",
        }

    def check_deforestation_free(self, plot: Dict[str, Any]) -> Dict[str, Any]:
        """Check deforestation-free status with evidence type."""
        compliance = self.verify_cutoff_compliance(plot)
        evidence_type = plot.get("satellite_verification_date")
        return {
            **compliance,
            "evidence_available": evidence_type is not None,
            "evidence_type": "satellite_imagery" if evidence_type else "none",
        }

    def validate_temporal_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Validate temporal evidence (satellite imagery, aerial photos)."""
        evidence_date = evidence.get("date", "")
        evidence_type = evidence.get("type", "")
        covers_cutoff = False
        if evidence_date:
            try:
                ev_date = date.fromisoformat(evidence_date)
                # Evidence must cover the period around the cutoff date
                covers_cutoff = ev_date >= self.CUTOFF_DATE
            except ValueError:
                pass
        valid_types = ["satellite_imagery", "aerial_photography", "ground_survey", "forest_inventory"]
        return {
            "valid": evidence_type in valid_types and covers_cutoff,
            "evidence_type": evidence_type,
            "evidence_date": evidence_date,
            "covers_cutoff_period": covers_cutoff,
            "accepted_types": valid_types,
        }

    def analyze_land_use_history(self, plot: Dict[str, Any],
                                  history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze land use history for a plot around the cutoff date."""
        pre_cutoff = [h for h in history if h.get("date", "9999") <= str(self.CUTOFF_DATE)]
        post_cutoff = [h for h in history if h.get("date", "0000") > str(self.CUTOFF_DATE)]

        forest_loss_post_cutoff = any(
            h.get("event") == "deforestation" or h.get("event") == "forest_degradation"
            for h in post_cutoff
        )

        return {
            "plot_id": plot.get("plot_id", "unknown"),
            "pre_cutoff_events": len(pre_cutoff),
            "post_cutoff_events": len(post_cutoff),
            "forest_loss_post_cutoff": forest_loss_post_cutoff,
            "compliant": not forest_loss_post_cutoff,
            "cutoff_date": str(self.CUTOFF_DATE),
        }

    def generate_cutoff_declaration(self, plot: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a cutoff date compliance declaration."""
        compliance = self.verify_cutoff_compliance(plot)
        return {
            "declaration_type": "cutoff_compliance",
            "plot_id": plot.get("plot_id", "unknown"),
            "cutoff_date": str(self.CUTOFF_DATE),
            "compliant": compliance["compliant"],
            "generated_at": datetime.now().isoformat(),
            "provenance_hash": _compute_hash({
                "plot_id": plot.get("plot_id"),
                "compliant": compliance["compliant"],
                "cutoff_date": str(self.CUTOFF_DATE),
            }),
        }

    def check_exemptions(self, plot: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a plot qualifies for any cutoff date exemptions."""
        # EUDR allows exemptions for certain land uses if documented
        land_use = plot.get("land_use", "")
        exempted = land_use in ["agroforestry_pre2020", "managed_plantation_pre2015"]
        return {
            "plot_id": plot.get("plot_id", "unknown"),
            "exemption_applies": exempted,
            "land_use": land_use,
            "reason": f"Land use '{land_use}' qualifies for exemption" if exempted
                      else "No exemption applicable",
        }

    def batch_verify(self, plots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch verify cutoff compliance for multiple plots."""
        results = []
        compliant_count = 0
        for plot in plots:
            result = self.verify_cutoff_compliance(plot)
            results.append(result)
            if result["compliant"]:
                compliant_count += 1
        return {
            "total": len(plots),
            "compliant": compliant_count,
            "non_compliant": len(plots) - compliant_count,
            "compliance_rate_pct": round(compliant_count / len(plots) * 100, 1) if plots else 0.0,
            "results": results,
        }

    def cutoff_summary(self, plots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of cutoff date compliance across all plots."""
        batch = self.batch_verify(plots)
        return {
            "cutoff_date": str(self.CUTOFF_DATE),
            "total_plots": batch["total"],
            "compliant_plots": batch["compliant"],
            "non_compliant_plots": batch["non_compliant"],
            "compliance_rate_pct": batch["compliance_rate_pct"],
            "summary_generated_at": datetime.now().isoformat(),
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCutoffDate:
    """Tests for the cutoff date verification engine."""

    @pytest.fixture
    def engine(self) -> CutoffDateEngineSimulator:
        return CutoffDateEngineSimulator()

    # 1
    def test_verify_cutoff_compliance_pass(self, engine, sample_plot):
        """Plot with deforestation-free date before cutoff passes."""
        result = engine.verify_cutoff_compliance(sample_plot)
        assert result["compliant"] is True
        assert result["cutoff_date"] == "2020-12-31"

    # 2
    def test_verify_cutoff_compliance_fail(self, engine):
        """Plot with deforestation after cutoff fails."""
        plot = {
            "plot_id": "test-fail",
            "deforestation_free_since": "2021-06-15",
        }
        result = engine.verify_cutoff_compliance(plot)
        assert result["compliant"] is False

    # 3
    def test_check_deforestation_free(self, engine, sample_plot):
        """Deforestation-free check includes evidence status."""
        result = engine.check_deforestation_free(sample_plot)
        assert result["compliant"] is True
        assert result["evidence_available"] is True
        assert result["evidence_type"] == "satellite_imagery"

    # 4
    def test_cutoff_date_is_20201231(self, engine):
        """EUDR cutoff date is 2020-12-31."""
        assert engine.CUTOFF_DATE == date(2020, 12, 31)

    # 5
    def test_temporal_evidence(self, engine):
        """Valid temporal evidence is accepted."""
        evidence = {
            "date": "2021-03-15",
            "type": "satellite_imagery",
        }
        result = engine.validate_temporal_evidence(evidence)
        assert result["valid"] is True
        assert result["covers_cutoff_period"] is True

    # 6
    def test_temporal_evidence_invalid_type(self, engine):
        """Invalid evidence type is rejected."""
        evidence = {
            "date": "2021-03-15",
            "type": "social_media_post",
        }
        result = engine.validate_temporal_evidence(evidence)
        assert result["valid"] is False

    # 7
    def test_land_use_history(self, engine, sample_plot):
        """Land use history analysis detects post-cutoff changes."""
        history = [
            {"date": "2018-01-01", "event": "palm_oil_plantation"},
            {"date": "2019-06-15", "event": "land_clearing"},
            {"date": "2022-03-01", "event": "replanting"},
        ]
        result = engine.analyze_land_use_history(sample_plot, history)
        assert result["pre_cutoff_events"] == 2
        assert result["post_cutoff_events"] == 1
        assert result["compliant"] is True  # no deforestation event post-cutoff

    # 8
    def test_generate_cutoff_declaration(self, engine, sample_plot):
        """Cutoff declaration includes provenance hash."""
        declaration = engine.generate_cutoff_declaration(sample_plot)
        assert declaration["declaration_type"] == "cutoff_compliance"
        assert len(declaration["provenance_hash"]) == 64

    # 9
    def test_exemptions(self, engine):
        """Exemption check identifies qualifying land uses."""
        exempted_plot = {"plot_id": "ex-001", "land_use": "agroforestry_pre2020"}
        result = engine.check_exemptions(exempted_plot)
        assert result["exemption_applies"] is True

        normal_plot = {"plot_id": "norm-001", "land_use": "palm_oil_plantation"}
        result = engine.check_exemptions(normal_plot)
        assert result["exemption_applies"] is False

    # 10
    def test_batch_verify(self, engine, sample_plots_list):
        """Batch verification processes all plots."""
        result = engine.batch_verify(sample_plots_list)
        assert result["total"] == len(sample_plots_list)
        assert result["compliant"] + result["non_compliant"] == result["total"]
        assert 0 <= result["compliance_rate_pct"] <= 100

    def test_cutoff_summary(self, engine, sample_plots_list):
        """Cutoff summary provides aggregate statistics."""
        summary = engine.cutoff_summary(sample_plots_list)
        assert summary["cutoff_date"] == "2020-12-31"
        assert summary["total_plots"] == len(sample_plots_list)
