# -*- coding: utf-8 -*-
"""
GreenLang Factors Exceptions - Factor Catalog and Governance Errors

This module provides exception classes for factors-related errors
in the factor catalog, quality gates, ingestion pipeline, matching
engine, governance workflow, and source registry.

Features:
- Factor validation and QA gate failures
- Ingestion / ETL pipeline errors
- Governance and promotion gate blocks
- Edition manifest and release errors
- License and export guard violations
- Matching pipeline failures
- Source registry and parser errors
- Quality engine rejections
- Source monitoring watch failures

Author: GreenLang Team
Date: 2026-04-17
"""

from typing import Any, Dict, List, Optional

from greenlang.exceptions.base import GreenLangException


class FactorsException(GreenLangException):
    """Base exception for all factors-related errors.

    Raised when the factor catalog, governance, ingestion, or matching
    subsystems encounter errors.
    """
    ERROR_PREFIX = "GL_FACTORS"


class FactorValidationError(FactorsException):
    """Factor QA gate validation failed.

    Raised when a factor dict or record fails one or more quality
    gate checks (Q1-Q6) before catalog insertion.

    Example:
        >>> raise FactorValidationError(
        ...     message="Factor missing CO2 vector",
        ...     factor_id="EF:US:diesel:2024:v1",
        ...     gate="Q1_schema",
        ...     violations=["vectors.CO2 is required"],
        ... )
    """

    def __init__(
        self,
        message: str,
        factor_id: Optional[str] = None,
        gate: Optional[str] = None,
        violations: Optional[List[str]] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        context = context or {}
        if factor_id:
            context["factor_id"] = factor_id
        if gate:
            context["gate"] = gate
        if violations:
            context["violations"] = violations
        super().__init__(message, agent_name=agent_name, context=context)


class FactorIngestionError(FactorsException):
    """ETL / parser pipeline failure.

    Raised when factors cannot be ingested due to file parsing,
    normalization, or database write errors.

    Example:
        >>> raise FactorIngestionError(
        ...     message="CBAM JSON parse failed",
        ...     source_id="eu_cbam",
        ...     artifact_id="abc-123",
        ... )
    """

    def __init__(
        self,
        message: str,
        source_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        context = context or {}
        if source_id:
            context["source_id"] = source_id
        if artifact_id:
            context["artifact_id"] = artifact_id
        super().__init__(message, agent_name=agent_name, context=context)


class FactorGovernanceError(FactorsException):
    """Approval gate or promotion block.

    Raised when a governance workflow step (legal sign-off, methodology
    review, approval gate) blocks an operation.

    Example:
        >>> raise FactorGovernanceError(
        ...     message="Legal sign-off missing for electricity_maps",
        ...     source_id="electricity_maps",
        ...     blockers=["legal_signoff_artifact is None"],
        ... )
    """

    def __init__(
        self,
        message: str,
        source_id: Optional[str] = None,
        blockers: Optional[List[str]] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        context = context or {}
        if source_id:
            context["source_id"] = source_id
        if blockers:
            context["blockers"] = blockers
        super().__init__(message, agent_name=agent_name, context=context)


class FactorEditionError(FactorsException):
    """Edition manifest or release error.

    Raised when building, publishing, or resolving an edition
    manifest fails.

    Example:
        >>> raise FactorEditionError(
        ...     message="Unknown edition_id: 'does-not-exist'",
        ...     edition_id="does-not-exist",
        ... )
    """

    def __init__(
        self,
        message: str,
        edition_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        context = context or {}
        if edition_id:
            context["edition_id"] = edition_id
        super().__init__(message, agent_name=agent_name, context=context)


class FactorLicenseError(FactorsException):
    """License or export guard violation.

    Raised when a factor's license terms prohibit the requested
    operation (bulk export, redistribution, commercial use).

    Example:
        >>> raise FactorLicenseError(
        ...     message="Factor from connector_only source cannot be exported",
        ...     factor_id="EF:EM:grid:DE:2024:v1",
        ...     license_class="commercial_api",
        ... )
    """

    def __init__(
        self,
        message: str,
        factor_id: Optional[str] = None,
        license_class: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        context = context or {}
        if factor_id:
            context["factor_id"] = factor_id
        if license_class:
            context["license_class"] = license_class
        super().__init__(message, agent_name=agent_name, context=context)


class FactorMatchingError(FactorsException):
    """Matching pipeline failure.

    Raised when the deterministic matching pipeline (facet filter,
    lexical search, rerank) encounters an unrecoverable error.

    Example:
        >>> raise FactorMatchingError(
        ...     message="No factors matched query",
        ...     query="unknown_fuel_xyz",
        ...     edition_id="builtin-v1.0.0",
        ... )
    """

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        edition_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        context = context or {}
        if query:
            context["query"] = query
        if edition_id:
            context["edition_id"] = edition_id
        super().__init__(message, agent_name=agent_name, context=context)


class SourceRegistryError(FactorsException):
    """Source registry YAML load or validation error.

    Raised when the source_registry.yaml cannot be loaded or
    contains validation issues.

    Example:
        >>> raise SourceRegistryError(
        ...     message="Duplicate source_id: electricity_maps",
        ...     issues=["duplicate source_id: electricity_maps"],
        ... )
    """

    def __init__(
        self,
        message: str,
        issues: Optional[List[str]] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        context = context or {}
        if issues:
            context["issues"] = issues
        super().__init__(message, agent_name=agent_name, context=context)


class ParserError(FactorsException):
    """Source-specific parser failure.

    Raised when a parser (CBAM, DEFRA, EPA, etc.) fails to
    extract factor dicts from raw artifacts.

    Example:
        >>> raise ParserError(
        ...     message="DEFRA scope1 JSON missing 'units' key",
        ...     parser_id="defra_scope1",
        ...     source_id="defra_uk",
        ... )
    """

    def __init__(
        self,
        message: str,
        parser_id: Optional[str] = None,
        source_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        context = context or {}
        if parser_id:
            context["parser_id"] = parser_id
        if source_id:
            context["source_id"] = source_id
        super().__init__(message, agent_name=agent_name, context=context)


class QualityGateError(FactorsException):
    """Quality engine rejection.

    Raised when the quality engine rejects a factor or batch
    due to plausibility, completeness, or consistency checks.

    Example:
        >>> raise QualityGateError(
        ...     message="Outlier: co2e_total > 1e7",
        ...     factor_id="EF:US:coal:2024:v1",
        ...     gate_name="Q2_plausibility",
        ... )
    """

    def __init__(
        self,
        message: str,
        factor_id: Optional[str] = None,
        gate_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        context = context or {}
        if factor_id:
            context["factor_id"] = factor_id
        if gate_name:
            context["gate_name"] = gate_name
        super().__init__(message, agent_name=agent_name, context=context)


class WatchError(FactorsException):
    """Source monitoring / watch failure.

    Raised when a source watch check (HTTP HEAD, API poll,
    document diff) fails or detects an unexpected state.

    Example:
        >>> raise WatchError(
        ...     message="EPA Hub endpoint returned 503",
        ...     source_id="epa_hub",
        ...     watch_mechanism="http_head",
        ... )
    """

    def __init__(
        self,
        message: str,
        source_id: Optional[str] = None,
        watch_mechanism: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        context = context or {}
        if source_id:
            context["source_id"] = source_id
        if watch_mechanism:
            context["watch_mechanism"] = watch_mechanism
        super().__init__(message, agent_name=agent_name, context=context)


__all__ = [
    'FactorsException',
    'FactorValidationError',
    'FactorIngestionError',
    'FactorGovernanceError',
    'FactorEditionError',
    'FactorLicenseError',
    'FactorMatchingError',
    'SourceRegistryError',
    'ParserError',
    'QualityGateError',
    'WatchError',
]
