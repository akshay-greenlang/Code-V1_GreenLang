# -*- coding: utf-8 -*-
"""Tests for PACK-016 package initialization."""
import pytest


class TestPack016Init:
    """Test suite for PACK-016 root __init__.py metadata."""

    def test_version_defined(self):
        """__version__ is defined and follows semver."""
        from packs.eu_compliance.PACK_016_esrs_e1_climate import __version__
        assert __version__ == "1.0.0"

    def test_pack_id_defined(self):
        """__pack__ is defined as PACK-016."""
        from packs.eu_compliance.PACK_016_esrs_e1_climate import __pack__
        assert __pack__ == "PACK-016"

    def test_pack_name_defined(self):
        """__pack_name__ is defined."""
        from packs.eu_compliance.PACK_016_esrs_e1_climate import __pack_name__
        assert "E1" in __pack_name__
        assert "Climate" in __pack_name__

    def test_category_defined(self):
        """__category__ is eu-compliance."""
        from packs.eu_compliance.PACK_016_esrs_e1_climate import __category__
        assert __category__ == "eu-compliance"

    def test_placeholder(self):
        """Placeholder test."""
        assert True
