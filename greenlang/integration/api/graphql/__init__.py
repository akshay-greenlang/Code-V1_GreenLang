# -*- coding: utf-8 -*-
"""
GreenLang GraphQL API Module
Phase 4 - Comprehensive GraphQL API Layer

GAP-12 (2026-04-20):
    * Factors types + resolvers registered below via Strawberry
      multi-inheritance composition.  The ``Query`` / ``Mutation`` types
      exported here are the *merged* roots; :func:`create_graphql_app`
      will pick them up automatically without any other changes.
"""

# Factors types + resolvers (GAP-12) — always importable.
from greenlang.integration.api.graphql.resolvers_factors import (
    FactorsQuery,
    FactorsMutation,
)
from greenlang.integration.api.graphql import types_factors  # re-export namespace


def build_merged_roots():
    """Return ``(Query, Mutation, Subscription)`` with Factors merged in.

    Called lazily so importing this package doesn't force a load of the
    base resolver graph (which transitively pulls orchestrator / auth
    modules that aren't always available).

    Usage from :func:`create_graphql_app`::

        from greenlang.integration.api.graphql import build_merged_roots
        Query, Mutation, Subscription = build_merged_roots()
        schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)
    """
    import strawberry

    try:
        from greenlang.integration.api.graphql.resolvers import (
            Query as _BaseQuery,
            Mutation as _BaseMutation,
        )
        from greenlang.integration.api.graphql.subscriptions import Subscription
    except Exception:  # pragma: no cover — fall back to Factors-only schema
        _BaseQuery = None
        _BaseMutation = None
        Subscription = None

    if _BaseQuery is not None and _BaseMutation is not None:

        @strawberry.type
        class Query(_BaseQuery, FactorsQuery):  # type: ignore[misc,valid-type]
            pass

        @strawberry.type
        class Mutation(_BaseMutation, FactorsMutation):  # type: ignore[misc,valid-type]
            pass

    else:

        @strawberry.type
        class Query(FactorsQuery):
            pass

        @strawberry.type
        class Mutation(FactorsMutation):
            pass

    return Query, Mutation, Subscription


__all__ = [
    "FactorsQuery",
    "FactorsMutation",
    "types_factors",
    "build_merged_roots",
]

__version__ = "1.0.1"
