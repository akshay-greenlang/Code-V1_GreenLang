# -*- coding: utf-8 -*-
"""ETL helpers for GreenLang Factors (lazy exports)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

__all__ = ["ingest_builtin_database", "ingest_from_paths", "validate_factor_dict"]


def __getattr__(name: str) -> Any:
    if name == "ingest_builtin_database":
        from greenlang.factors.etl.ingest import ingest_builtin_database

        return ingest_builtin_database
    if name == "ingest_from_paths":
        from greenlang.factors.etl.ingest import ingest_from_paths

        return ingest_from_paths
    if name == "validate_factor_dict":
        from greenlang.factors.etl.qa import validate_factor_dict

        return validate_factor_dict
    raise AttributeError(name)


if TYPE_CHECKING:
    from greenlang.factors.etl.ingest import ingest_builtin_database, ingest_from_paths
    from greenlang.factors.etl.qa import validate_factor_dict
