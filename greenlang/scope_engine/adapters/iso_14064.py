# -*- coding: utf-8 -*-
"""ISO 14064-1:2018 projection.

ISO uses a category-based reporting structure (Categories 1-6) rather than
the GHG Protocol's scope-based one.
"""

from __future__ import annotations

from greenlang.scope_engine.models import Framework, FrameworkView, ScopeComputation

# ISO 14064-1 mapping:
#   Cat 1 = direct emissions (Scope 1)
#   Cat 2 = indirect from imported energy (Scope 2)
#   Cat 3 = indirect from transportation (subset of Scope 3 Cat 4+9+6+7)
#   Cat 4 = indirect from products used by the org (Scope 3 Cat 1+2+3+8)
#   Cat 5 = indirect from use of products from the org (Scope 3 Cat 10+11+12)
#   Cat 6 = other indirect
_CAT_3_SCOPE3 = {4, 9, 6, 7}
_CAT_4_SCOPE3 = {1, 2, 3, 8}
_CAT_5_SCOPE3 = {10, 11, 12}
_CAT_6_SCOPE3 = {5, 13, 14, 15}


class ISO14064Adapter:
    framework = Framework.ISO_14064

    def project(self, computation: ScopeComputation) -> FrameworkView:
        b = computation.breakdown
        scope3_by_cat = b.scope_3_by_category or {}

        def sum_cats(cats: set[int]):
            return sum((scope3_by_cat.get(c, 0) for c in cats), 0)

        rows = [
            {"category": "1", "label": "Direct GHG emissions and removals",
             "co2e_kg": str(b.scope_1_co2e_kg)},
            {"category": "2", "label": "Indirect from imported energy",
             "co2e_kg": str(b.scope_2_location_co2e_kg)},
            {"category": "3", "label": "Indirect from transportation",
             "co2e_kg": str(sum_cats(_CAT_3_SCOPE3))},
            {"category": "4", "label": "Indirect from products used by the organization",
             "co2e_kg": str(sum_cats(_CAT_4_SCOPE3))},
            {"category": "5", "label": "Indirect associated with use of products from the organization",
             "co2e_kg": str(sum_cats(_CAT_5_SCOPE3))},
            {"category": "6", "label": "Other indirect GHG emissions",
             "co2e_kg": str(sum_cats(_CAT_6_SCOPE3))},
        ]
        return FrameworkView(
            framework=self.framework,
            rows=rows,
            metadata={"standard": "ISO 14064-1:2018"},
        )
