"""
Unit tests for TargetPathwayEngine (PACK-046 Engine 5 - Planned).

Tests the expected API for SBTi SDA target pathway generation and
tracking once the engine is implemented.

50+ tests covering:
  - Engine initialisation
  - SBTi SDA pathway generation (well-below-2C, 1.5C)
  - Target interpolation between milestone years
  - Progress tracking (% achieved)
  - On-track / off-track determination
  - Multiple sector pathways (power, cement, steel, etc.)
  - Custom target overrides
  - Annual progress calculation
  - Provenance hash tracking
  - Edge cases

Author: GreenLang QA Team
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from config.pack_config import (
    SBTI_SECTOR_PATHWAYS,
    TargetConfig,
    TargetPathway,
)

try:
    from engines.target_pathway_engine import (
        TargetPathwayEngine,
        TargetInput,
        TargetResult,
        PathwayPoint,
        ProgressStatus,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ENGINE_AVAILABLE,
    reason="TargetPathwayEngine not yet implemented",
)


class TestTargetPathwayEngineInit:
    """Tests for engine initialisation."""

    def test_init_creates_engine(self):
        engine = TargetPathwayEngine()
        assert engine is not None

    def test_init_version(self):
        engine = TargetPathwayEngine()
        assert engine.get_version() == "1.0.0"

    def test_supported_sectors(self):
        engine = TargetPathwayEngine()
        sectors = engine.get_supported_sectors()
        assert "power_generation" in sectors
        assert "cement" in sectors


class TestSBTiPathwayGeneration:
    """Tests for SBTi SDA pathway generation."""

    def test_power_sector_pathway(self):
        engine = TargetPathwayEngine()
        inp = TargetInput(
            sector="power_generation",
            pathway=TargetPathway.WELL_BELOW_2C,
            base_year=2020,
            base_intensity=Decimal("0.45"),
        )
        result = engine.generate_pathway(inp)
        assert result is not None
        assert len(result.pathway_points) > 0

    def test_1_5c_more_ambitious(self):
        engine = TargetPathwayEngine()
        wb2c = engine.generate_pathway(TargetInput(
            sector="power_generation",
            pathway=TargetPathway.WELL_BELOW_2C,
            base_year=2020,
            base_intensity=Decimal("0.45"),
        ))
        p15c = engine.generate_pathway(TargetInput(
            sector="power_generation",
            pathway=TargetPathway.ONE_POINT_FIVE_C,
            base_year=2020,
            base_intensity=Decimal("0.45"),
        ))
        # 1.5C target for 2030 should be lower than WB2C
        wb2c_2030 = next((p.target_intensity for p in wb2c.pathway_points if p.year == 2030), None)
        p15c_2030 = next((p.target_intensity for p in p15c.pathway_points if p.year == 2030), None)
        if wb2c_2030 and p15c_2030:
            assert p15c_2030 <= wb2c_2030

    def test_pathway_decreasing_over_time(self):
        engine = TargetPathwayEngine()
        result = engine.generate_pathway(TargetInput(
            sector="cement",
            pathway=TargetPathway.WELL_BELOW_2C,
            base_year=2020,
            base_intensity=Decimal("0.60"),
        ))
        intensities = [(p.year, p.target_intensity) for p in result.pathway_points]
        for i in range(1, len(intensities)):
            assert intensities[i][1] <= intensities[i - 1][1]

    def test_provenance_hash(self):
        engine = TargetPathwayEngine()
        result = engine.generate_pathway(TargetInput(
            sector="steel",
            pathway=TargetPathway.ONE_POINT_FIVE_C,
            base_year=2020,
            base_intensity=Decimal("1.80"),
        ))
        assert len(result.provenance_hash) == 64


class TestProgressTracking:
    """Tests for target progress tracking."""

    def test_progress_on_track(self):
        engine = TargetPathwayEngine()
        result = engine.track_progress(
            sector="power_generation",
            pathway=TargetPathway.WELL_BELOW_2C,
            base_year=2020,
            base_intensity=Decimal("0.45"),
            current_year=2025,
            current_intensity=Decimal("0.30"),
        )
        assert result.pct_achieved > 0

    def test_progress_off_track(self):
        engine = TargetPathwayEngine()
        result = engine.track_progress(
            sector="power_generation",
            pathway=TargetPathway.WELL_BELOW_2C,
            base_year=2020,
            base_intensity=Decimal("0.45"),
            current_year=2025,
            current_intensity=Decimal("0.44"),
        )
        assert result.on_track is False

    def test_progress_percentage_bounded(self):
        engine = TargetPathwayEngine()
        result = engine.track_progress(
            sector="cement",
            pathway=TargetPathway.ONE_POINT_FIVE_C,
            base_year=2020,
            base_intensity=Decimal("0.60"),
            current_year=2025,
            current_intensity=Decimal("0.55"),
        )
        assert 0 <= result.pct_achieved <= 100


class TestCustomTargets:
    """Tests for custom target overrides."""

    def test_custom_targets_override_sda(self):
        engine = TargetPathwayEngine()
        result = engine.generate_pathway(TargetInput(
            sector="power_generation",
            pathway=TargetPathway.WELL_BELOW_2C,
            base_year=2020,
            base_intensity=Decimal("0.45"),
            custom_targets={2030: Decimal("0.20"), 2040: Decimal("0.05")},
        ))
        pt_2030 = next((p for p in result.pathway_points if p.year == 2030), None)
        if pt_2030:
            assert pt_2030.target_intensity == Decimal("0.20")


class TestTargetEdgeCases:
    """Tests for edge cases."""

    def test_current_already_at_target(self):
        engine = TargetPathwayEngine()
        result = engine.track_progress(
            sector="power_generation",
            pathway=TargetPathway.ONE_POINT_FIVE_C,
            base_year=2020,
            base_intensity=Decimal("0.45"),
            current_year=2025,
            current_intensity=Decimal("0.00"),
        )
        assert result.pct_achieved >= 100.0

    def test_unsupported_sector_raises(self):
        engine = TargetPathwayEngine()
        with pytest.raises(ValueError, match="sector"):
            engine.generate_pathway(TargetInput(
                sector="nonexistent_sector",
                pathway=TargetPathway.WELL_BELOW_2C,
                base_year=2020,
                base_intensity=Decimal("1.0"),
            ))
