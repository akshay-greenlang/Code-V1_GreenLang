# -*- coding: utf-8 -*-
"""Phase 3 / Wave 1.5 — synthetic DEFRA Excel reference fixtures.

This package carries the deterministic, byte-stable DEFRA workbook
fixture used by the Phase 3 ingestion pipeline e2e suite. The xlsx
binary is regenerated at test-collection time by
:mod:`_build_defra_fixture` if the file is missing — see that module's
docstring for the exact deterministic-write contract.
"""
