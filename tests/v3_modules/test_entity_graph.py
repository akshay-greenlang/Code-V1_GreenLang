# -*- coding: utf-8 -*-
"""EntityGraph tests (PLATFORM 1, task #25).

Minimal smoke tests only. Full API tests deferred to PLATFORM 1 v2.
"""

from __future__ import annotations

import pytest


def test_entity_graph_imports():
    from greenlang.entity_graph import (  # noqa: F401
        EntityGraph,
        EntityNode,
        EntityEdge,
        NodeType,
        EdgeType,
    )


@pytest.mark.skip(reason="EntityNode/EntityEdge construction semantics need docs review")
def test_entity_graph_add_and_fetch_placeholder():
    """To be implemented after EntityGraph API finalized."""
