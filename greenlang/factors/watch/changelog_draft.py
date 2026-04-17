# -*- coding: utf-8 -*-
"""Machine changelog draft before human approval (U5)."""

from __future__ import annotations

from typing import Any, Dict, List


def draft_changelog_lines(compare: Dict[str, Any]) -> List[str]:
    lines = [
        f"edition diff {compare.get('left_edition_id')} -> {compare.get('right_edition_id')}",
        f"added: {len(compare.get('added_factor_ids') or [])}",
        f"removed: {len(compare.get('removed_factor_ids') or [])}",
        f"changed: {len(compare.get('changed_factor_ids') or [])}",
    ]
    return lines
