# -*- coding: utf-8 -*-
"""GreenLang Factors GraphQL surface (W4-C / API15).

Single source of truth is the REST 16-field contract exposed by
:mod:`greenlang.factors.resolution.result`. This package projects the
same envelopes into a Strawberry-style GraphQL schema so neither
surface can leak fields the other omits.
"""
from .schema import build_schema  # noqa: F401
from .resolvers import Query  # noqa: F401
from .routes import graphql_router  # noqa: F401

__all__ = ["build_schema", "graphql_router", "Query"]
