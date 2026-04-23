# -*- coding: utf-8 -*-
"""Release-gate tests enforcing the 7 CTO non-negotiables for GreenLang Factors FY27.

These tests are the binary go/no-go signal for any external pilot. Each of the
seven test modules in this package maps 1:1 to a CTO non-negotiable:

    N1. No CO2e-only Certified factors — gas-level vectors required.
    N2. Factor rows are immutable; changes create new, chain-linked versions.
    N3. Fallback is always visible (rank, step label, why-chosen, alternates).
    N4. License classes never mix — open / licensed / customer-private / OEM.
    N5. Release signoff blocks when required metadata is missing.
    N6. Policy workflows never call raw factor lookup without a method_profile.
    N7. Open-core boundary — Community never receives Premium / Private / OEM.

Tests that are xfail today are honest signal to the CTO about which gates
still require production code to close.
"""
