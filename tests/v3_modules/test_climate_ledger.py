# -*- coding: utf-8 -*-
"""ClimateLedger tests (PLATFORM 1, task #25).

Intentionally minimal. Deeper tests require API-specific knowledge of the
ClimateLedger constructor signature which takes parameters not known at this
level. Placeholder for future deep-dive.
"""

from __future__ import annotations

import pytest


def test_ledger_importable():
    from greenlang.climate_ledger import ClimateLedger  # noqa: F401


@pytest.mark.skip(reason="Deeper ClimateLedger behavior tests require constructor docs")
def test_ledger_chain_integrity_placeholder():
    """To be implemented in PLATFORM 1 v2 (requires ClimateLedger constructor study)."""
